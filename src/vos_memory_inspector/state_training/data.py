"""Checksummed v2 state-only caches and bounded record loading."""
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import io
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from ..state_schema import CanonicalState, StateSpec, validate_paired_state_contract
from .storage import digest, sha256
from .usage import MemoryPolicy, record_usage

OFFICIAL_SCHEMA = 'cmmt.official_state_training_index.v1'
SAMPLE_SCHEMA = 'cmmt.state_training_sample_index.v1'


def file_stamp(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns)


@dataclass
class TrainingIndex:
    entries: list
    identity: dict
    official: bool
    normalization: dict | None

    def role(self, role):
        return [e for e in self.entries if e['case']['training_role'] == role]


def load_index(path, root_map=None, *, allow_sample=False):
    original = json.loads(Path(path).read_text())
    schema = original.get('schema_version')
    official = schema == OFFICIAL_SCHEMA
    if not official and not (allow_sample and schema == SAMPLE_SCHEMA):
        raise ValueError('official training index required; sample indexes need explicit sample mode')
    if 'content_sha256' in original:
        unsigned = {k: v for k, v in original.items() if k != 'content_sha256'}
        if digest(unsigned) != original['content_sha256']:
            raise ValueError('index content hash mismatch')
    if official and original.get('state') != 'tensor_validated':
        raise ValueError('official index is not tensor_validated')
    mappings = sorted([(Path(a), Path(b).resolve()) for a, b in (root_map or {}).items()],
                      key=lambda pair: len(pair[0].parts), reverse=True)
    entries, keys, paths = [], set(), set()
    videos = {'train': set(), 'validation': set()}
    for source in original['entries']:
        e = dict(source)
        case = e['case']
        role = case['training_role']
        if role not in videos or case['case_id'] in keys:
            raise ValueError('invalid role or duplicate case ID')
        if official:
            if case['dataset'] not in ('MOSEv2', 'LVOS v2'):
                raise ValueError('unknown official dataset')
            if (role == 'train' and case['official_split'] != 'train') or (
                role == 'validation' and (case['dataset'] != 'LVOS v2' or
                case['official_split'] not in ('val', 'valid', 'validation'))):
                raise ValueError('invalid official dataset role')
        keys.add(case['case_id'])
        videos[role].add((case['dataset'], case['video_id']))
        positions = e['valid_indices']
        if e['records'] < 1 or e['records'] != len(positions) or len({tuple(p) for p in positions}) != len(positions):
            raise ValueError('invalid indexed record identities')
        if len(e.get('sha256', '')) != 64:
            raise ValueError('missing cache content hash')
        p = Path(e['path'])
        e['original_path'] = str(p)
        for before, after in mappings:
            if p.is_relative_to(before):
                p = after / p.relative_to(before)
                break
        e['path'] = str(p.resolve())
        if e['path'] in paths:
            raise ValueError('duplicate cache path after root mapping')
        paths.add(e['path'])
        entries.append(e)
    if not entries or videos['train'] & videos['validation']:
        raise ValueError('empty index or train/validation video overlap')
    if official:
        if [e['case']['case_id'] for e in entries] != original['identity']['expected_case_ids']:
            raise ValueError('official index coverage mismatch')
        if not videos['validation']:
            raise ValueError('official validation is empty')
        if original.get('synthetic_declared'):
            raise ValueError('synthetic data cannot be official training')
    identity = dict(index_sha256=sha256(path), content_sha256=original.get('content_sha256'),
                    official=official, cases=len(entries), records=sum(e['records'] for e in entries))
    return TrainingIndex(entries, identity, official, original.get('normalization'))


def sidecar_hash(path):
    path = Path(path)
    if not path.is_file():
        return None
    fields = path.read_text().split()
    return fields[0] if fields else None


def load_pair(entry, *, expected_stamp=None, expected_spec=None):
    path = Path(entry['path'])
    before = file_stamp(path)
    if expected_stamp is None:
        expected_stamp = entry.get('verified_stamp')
    if expected_stamp is not None and before != expected_stamp:
        raise ValueError('cache changed since verification')
    if Path(str(path) + '.writer.lock').exists():
        raise ValueError('cache has active writer')
    raw = path.read_bytes()
    import hashlib
    checksum = hashlib.sha256(raw).hexdigest()
    sidecar = Path(str(path) + '.sha256')
    if checksum != entry['sha256'] or sidecar_hash(sidecar) != checksum:
        raise ValueError('cache checksum mismatch')
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if file_stamp(path) != before:
        raise ValueError('cache changed during read')
    if not isinstance(value, dict) or value.get('schema_version') != 'cmmt.prepared_handoff_case.v2':
        raise ValueError('v2 prepared handoff cache required')
    s, t, md = value['source_canonical'], value['target_canonical'], value['metadata']
    if not isinstance(s, CanonicalState) or not isinstance(t, CanonicalState) or not isinstance(md, dict):
        raise TypeError('cache must contain canonical states and metadata mapping')
    valid = validate_paired_state_contract(s, t)
    if s.spec != t.spec or (expected_spec is not None and s.spec != expected_spec):
        raise ValueError('cache tensor spec mismatch')
    for state in (s, t):
        if state.spatial_memory.device.type != 'cpu' or state.object_pointer.device.type != 'cpu':
            raise ValueError('cache must be CPU offloaded')
        if state.spatial_memory.dtype != torch.bfloat16 or state.object_pointer.dtype != torch.float32:
            raise ValueError('unexpected runtime tensor dtype')
        for tensor in (state.spatial_memory, state.object_pointer, state.presence_logits):
            if not bool(torch.isfinite(tensor[valid]).all()):
                raise ValueError('nonfinite valid record')
    if md.get('cache_mode') != 'state_only':
        raise ValueError('state_only cache required')
    if (md.get('source_model_id'), md.get('target_model_id')) != ('sam2.1-small', 'sam2.1-base-plus'):
        raise ValueError('cache model direction mismatch')
    if torch.nonzero(valid).tolist() != entry['valid_indices']:
        raise ValueError('index validity mismatch')
    case = entry['case']
    if md.get('video_id') != case['video_id'] or str(md.get('object_id')) != str(case['object_id']):
        raise ValueError('cache case identity mismatch')
    if tuple(map(str, s.object_ids)) != (str(case['object_id']),):
        raise ValueError('cache object IDs mismatch')
    if s.switch_frame != entry['runtime_switch_frame']:
        raise ValueError('runtime switch mismatch; filename frame is not runtime frame')
    if md.get('num_frames', s.switch_frame + 1) <= s.switch_frame:
        raise ValueError('runtime frame range mismatch')
    if s.spatial_memory.shape[:2] != (1, 1):
        raise ValueError('single object state-pair cache required')
    return value, before


def resident_bytes(value):
    return sum(t.numel() * t.element_size() for key in ('source_canonical', 'target_canonical')
               for t in vars(value[key]).values() if isinstance(t, torch.Tensor))


class RecordDataset(Dataset):
    def __init__(self, entries, *, cache_cases=2, cache_bytes=64 * 1024 * 1024):
        if cache_cases < 1 or cache_bytes < 1:
            raise ValueError('positive cache limits required')
        self.entries = entries
        self.ends = list(itertools.accumulate(e['records'] for e in entries))
        self.cache_cases, self.cache_bytes = cache_cases, cache_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.stamps = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def location(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        segment = bisect_right(self.ends, i)
        entry = self.entries[segment]
        relative = i - (self.ends[segment - 1] if segment else 0)
        position = entry.get('selected_positions', range(len(entry['valid_indices'])))[relative]
        return entry, position

    def obtain(self, entry):
        path = entry['path']
        if path in self.cache:
            if file_stamp(path) != self.stamps[path] or Path(path + '.writer.lock').exists():
                raise ValueError('cached file changed or has active writer')
            sidecar = Path(path + '.sha256')
            if sidecar_hash(sidecar) != entry['sha256']:
                raise ValueError('cache sidecar changed')
            value, size = self.cache.pop(path)
            self.cache[path] = (value, size)
            return value
        value, stamp = load_pair(entry, expected_stamp=self.stamps.get(path))
        self.stamps[path] = stamp
        size = resident_bytes(value)
        while self.cache and (len(self.cache) >= self.cache_cases or self.bytes + size > self.cache_bytes):
            _, (_, old_size) = self.cache.popitem(last=False)
            self.bytes -= old_size
        if size <= self.cache_bytes:
            self.cache[path] = (value, size)
            self.bytes += size
        return value

    def __getitem__(self, i):
        return self.__getitems__([i])[0]

    def __getitems__(self, indices):
        rows = []
        previous_path, value = None, None
        for i in indices:
            entry, position = self.location(i)
            if entry["path"] != previous_path:
                value = self.obtain(entry)
                previous_path = entry["path"]
            s, t = value['source_canonical'], value['target_canonical']
            b, o, k = entry['valid_indices'][position]
            policy = entry.get('memory_policy', {})
            sm, pm, supported = record_usage(s.frame_indices, s.is_conditioning, s.validity,
                s.switch_frame, MemoryPolicy(num_maskmem=policy.get('num_maskmem', 7),
                max_obj_ptrs=policy.get('max_obj_ptrs_in_encoder', 16),
                stride=policy.get('memory_temporal_stride_for_eval', 1),
                reverse=policy.get('reverse', False), correction=policy.get('correction', False)))
            row = {f'{role}_{name}': tensor[b, o, k].clone() for role, state in (('source', s), ('target', t))
                   for name, tensor in (('spatial', state.spatial_memory), ('pointer', state.object_pointer))}
            row.update(source_present=s.presence_logits[b, o, k, 0] > 0,
                       target_present=t.presence_logits[b, o, k, 0] > 0,
                       conditioning=s.is_conditioning[b, o, k], spatial_used=sm[b, o, k],
                       pointer_used=pm[b, o, k], usage_supported=torch.tensor(supported),
                       frame=s.frame_indices[b, o, k],
                       ref=(entry['case']['case_id'], b, o, k))
            rows.append(row)
        return rows


def collate(rows):
    return {key: ([r[key] for r in rows] if key == 'ref' else torch.stack([r[key] for r in rows]))
            for key in rows[0]}


def record_loader(entries, config, *, windows=None, cuda=False):
    dataset = RecordDataset(entries, cache_cases=config.cache_cases, cache_bytes=config.cache_bytes)
    options = dict(num_workers=config.workers, collate_fn=collate, pin_memory=cuda and config.pin_memory)
    if windows is None:
        options['batch_size'] = config.microbatch
    else:
        def batches():
            offset = 0
            for count in windows:
                for start in range(0, count, config.microbatch):
                    yield range(offset + start, offset + min(start + config.microbatch, count))
                offset += count
            if offset != len(dataset):
                raise ValueError('window loader coverage mismatch')
        options['batch_sampler'] = list(batches())
    if config.workers:
        options.update(prefetch_factor=config.prefetch_factor, multiprocessing_context='spawn')
    # An independent generator keeps loader worker creation out of model RNG.
    return DataLoader(dataset, generator=torch.Generator().manual_seed(config.seed), **options)
