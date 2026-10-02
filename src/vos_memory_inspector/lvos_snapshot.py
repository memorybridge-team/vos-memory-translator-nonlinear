"""원본 보존 audit와 bounded whole-record tensor view. Epoch에서 shard당 한 번 읽는다."""
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import math
import multiprocessing
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from .cache_migration import inspect_cache, _check_frozen
from .collection_contract import require, SEMANTICS
from .training_data import read_manifest, save_manifest
from .training_storage import sha256, content_hash, write_json, write_tensor_file, ExclusiveWriter
from .transformer_translator import SAM21_MEMORY_SPEC
from .lvos_contract import json_read

FIELDS = ('source_spatial', 'target_spatial', 'source_pointer', 'target_pointer', 'frame', 'slot', 'conditioning')


def diagnostic_json(value):
    if isinstance(value,list):
        return [diagnostic_json(v) for v in value]
    return value if math.isfinite(value) else str(value)


def stamp(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns, value.st_ino]


def build_snapshot(requests, selection, output, *, trusted=False, stable_seconds=60, max_shard_mib=256, pilot_ids=None):
    """전체 선정 metadata/tensor audit 후 완료된 원본 밖 derived view만 동결한다."""
    selection = read_manifest(selection)
    require(selection['dataset']=='LVOS v2', 'LVOS_ONLY')
    canonical = read_manifest(Path(__file__).resolve().parents[2]/'manifests/lvosv2_train_v1.json')
    require(selection['source_manifest_content_sha256']==canonical['content_sha256'], 'FROZEN_SOURCE_REVISION')
    require(pilot_ids is not None or {c['case_id'] for c in selection['cases']}=={c['case_id'] for c in canonical['cases']},
            'PARTIAL_SELECTION_REQUIRES_PILOT')
    require(selection['fit_membership']==read_manifest(Path(__file__).resolve().parents[2]/'manifests/lvosv2_train_v1_fit.json') and
            selection['development_membership']==read_manifest(Path(__file__).resolve().parents[2]/'manifests/lvosv2_train_v1_development.json'),
            'FROZEN_MEMBERSHIP_REVISION')
    require(max_shard_mib>0, 'SHARD_SIZE')
    expected = {c['case_id']: c for c in selection['cases']}
    if pilot_ids is not None:
        require(bool(pilot_ids) and len(pilot_ids)==len(set(pilot_ids)) and set(pilot_ids)<=expected.keys(), 'PILOT_SELECTION')
        expected = {key: expected[key] for key in pilot_ids}
    supplied = {r['case_id']: r for r in requests}
    require(len(supplied)==len(requests) and set(supplied)<=expected.keys(), 'REQUEST_CASE_DUPLICATE_OR_FOREIGN')
    output = Path(output).resolve()
    require(all(not output.is_relative_to(Path(r['path']).resolve().parent) for r in requests), 'OUTPUT_INSIDE_ORIGINAL')
    report = {'schema_version': 'cmmt.lvos_audit.v1', 'selection_digest': selection['selection_digest'],
              'pilot': pilot_ids is not None, 'results': [], 'splits': {}, 'duplicates_shared_prefix_records': 0}
    cases, shards, buffer, byte_count = [], [], [], 0
    origins, policy, models, common_digest, synthetic = [], None, None, None, None
    duplicate_keys = Counter()
    totals = {role: {'expected': sum(c['paired_split']==role for c in expected.values()), 'completed': 0,
        'rejected': 0, 'unknown': 0, 'valid_records': 0, 'padded_records': 0, 'raw_file_bytes': 0,
        'train_tensor_bytes': 0, 'videos': set()} for role in ('fit','development')}
    with ExclusiveWriter(output):
        require(not (output/'snapshot.json').exists(), 'SNAPSHOT_EXISTS')
        require(not (output/'audit.json').exists() and not (output/'views').exists(),'AUDIT_OUTPUT_EXISTS')
        def flush():
            nonlocal buffer, byte_count
            if not buffer:
                return
            split = buffer[0]['split']
            payload = {name: torch.stack([r[name] for r in buffer]) for name in FIELDS}
            payload['refs'] = [r['ref'] for r in buffer]
            info = write_tensor_file(output/f'views/{split}-{len(shards):05d}.pt', payload)
            info['path'] = 'views/'+info['path']
            require(len({r['ref'][0] for r in buffer})==1,'VIEW_MIXED_CASE')
            info.update(split=split, case_index=buffer[0]['ref'][0], records=len(buffer), stamp=stamp(output/info['path']))
            write_json(str(output/info['path'])+'.complete.json', info)
            shards.append(info)
            buffer, byte_count = [], 0
        for case_id, raw in expected.items():
            role = raw['paired_split']
            if case_id not in supplied:
                report['results'].append({'case_id':case_id,'decision':'PENDING_EVIDENCE','reason_codes':['EXPECTED_CASE_MISSING']})
                totals[role]['unknown'] += 1
                continue
            request = supplied[case_id]
            result, payload, gen = inspect_cache(request, trusted_team_legacy=trusted, stable_seconds=stable_seconds)
            result['case_id'] = case_id
            if result['decision']=='CONVERTIBLE':
                try:
                    _check_frozen(gen['case'], selection)
                    require(gen['case']['case_id']==case_id and gen['case']['pair_mode']=='native_history', 'CASE_MODE')
                    require(gen['case']['object_semantics']==SEMANTICS and len(gen['case']['object_ids'])==1, 'OBJECT_SEMANTICS')
                    source, target = payload['source_canonical'], payload['target_canonical']
                    require(source.spec==target.spec==SAM21_MEMORY_SPEC, 'FROZEN_MODEL_SPEC')
                    require(source.spatial_memory.dtype==target.spatial_memory.dtype==torch.bfloat16 and
                            source.object_pointer.dtype==target.object_pointer.dtype==torch.float32, 'RUNTIME_DTYPES')
                    record_bytes = 2*64**3*2+2*256*4+17
                    require(record_bytes<=max_shard_mib*1024**2, 'RECORD_EXCEEDS_SHARD')
                    this_policy = gen['case']['memory_policy']
                    common = content_hash({k:gen.get(k) for k in ('models','seed','collection_mode','collector_source_hashes',
                                                  'effective_model_policy','preprocessing','environment','synthetic')})
                    if models is None:
                        models, policy, common_digest, synthetic = gen['models'], this_policy, common, gen.get('synthetic',False)
                    require(models==gen['models'] and policy==this_policy and common_digest==common, 'MIXED_COLLECTION_CONDITIONS')
                    origin = {'path':str(Path(request['path']).resolve()), 'sha256':result['evidence']['cache_sha256'],
                              'stamp':stamp(request['path'])}
                    origins.append(origin)
                    index = len(cases)
                    cases.append({'case':gen['case'], 'generating':gen, 'origin':origin, 'split':role,
                                  'original_validity':source.validity.tolist(), 'original_slots':source.slot_order.tolist(),
                                  'original_frames':source.frame_indices.tolist(),'original_conditioning':source.is_conditioning.tolist(),
                                  'presence_diagnostic':diagnostic_json(source.presence_logits.detach().cpu().tolist())})
                    totals[role]['completed'] += 1
                    totals[role]['videos'].add(raw['video_id'])
                    totals[role]['padded_records'] += source.validity.numel()-int(source.validity.sum())
                    totals[role]['raw_file_bytes'] += Path(request['path']).stat().st_size
                    for batch, obj, record in torch.nonzero(source.validity).tolist():
                        item = {'source_spatial':source.spatial_memory[batch,obj,record].detach().cpu().clone(),
                                'target_spatial':target.spatial_memory[batch,obj,record].detach().cpu().clone(),
                                'source_pointer':source.object_pointer[batch,obj,record].detach().cpu().clone(),
                                'target_pointer':target.object_pointer[batch,obj,record].detach().cpu().clone(),
                                'frame':source.frame_indices[batch,obj,record].cpu(), 'slot':source.slot_order[batch,obj,record].cpu(),
                                'conditioning':source.is_conditioning[batch,obj,record].cpu(), 'split':role,
                                'ref':[index,batch,obj,record]}
                        size = sum(item[f].numel()*item[f].element_size() for f in FIELDS)
                        require(size<=max_shard_mib*1024**2, 'RECORD_EXCEEDS_SHARD')
                        if buffer and (buffer[0]['split']!=role or byte_count+size>max_shard_mib*1024**2):
                            flush()
                        buffer.append(item); byte_count += size
                        totals[role]['valid_records'] += 1
                        totals[role]['train_tensor_bytes'] += size
                        duplicate_keys[(raw['video_id'],str(raw['object_id']),int(item['frame']),bool(item['conditioning']))] += 1
                    flush()  # 완성 paired case 경계. 하나의 case만 여러 bounded shard로 나눌 수 있다.
                except (ValueError, KeyError, TypeError) as exc:
                    result.update(decision='NEEDS_RECOLLECTION', reason_codes=[getattr(exc,'code','SNAPSHOT_CASE_INVALID')], detail=str(exc))
            if result['decision']!='CONVERTIBLE':
                totals[role]['rejected' if result['decision']=='NEEDS_RECOLLECTION' else 'unknown'] += 1
            report['results'].append(result)
        flush()
        require(not ({c['case']['video_id'] for c in cases if c['split']=='fit'} &
                     {c['case']['video_id'] for c in cases if c['split']=='development'}), 'FIT_DEV_VIDEO_OVERLAP')
        report['duplicates_shared_prefix_records'] = sum(n-1 for n in duplicate_keys.values())
        report['splits'] = {k:{**v,'videos':sorted(v['videos'])} for k,v in totals.items()}
        report['record_weight_policy'] = 'all_frozen_valid_records_retained_including_shared_prefixes'
        complete = len(cases)==len(expected) and all(r['decision']=='CONVERTIBLE' for r in report['results'])
        report['status'] = 'PASS' if complete else 'FAIL'
        write_json(output/'audit.json', report)
        require(complete, 'AUDIT_INCOMPLETE', str(output/'audit.json'))
        snapshot = {'schema_version':'cmmt.lvos_snapshot.v1','state':'pilot_ready' if pilot_ids is not None else 'ready',
            'selection':selection, 'expected_case_ids':list(expected), 'models':models, 'memory_policy':policy,
            'object_semantics':SEMANTICS, 'pair_mode':'native_history', 'synthetic':synthetic,
            'cases':cases,'shards':shards,'origins':origins,'audit_sha256':sha256(output/'audit.json'),
            'generating_configuration_digest':common_digest, 'statistics':report['splits']}
        snapshot['sampling_policy']='shuffle_complete_cases_only; preserve_record_slot_order_inside_case'
        save_manifest(output/'snapshot.json', snapshot)
        return read_manifest(output/'snapshot.json')


def verify_snapshot(root, *, hashes=True):
    root = Path(root)
    value = read_manifest(root/'snapshot.json')
    require(value['state'] in {'ready','pilot_ready'} and sha256(root/'audit.json')==value['audit_sha256'], 'SNAPSHOT_AUDIT_CHANGED')
    for entry in value['shards']:
        path = root/entry['path']
        require(stamp(path)==entry['stamp'] and Path(str(path)+'.complete.json').is_file(), 'SNAPSHOT_CHANGED', str(path))
        require(json_read(str(path)+'.complete.json')==entry, 'VIEW_COMPLETION_CHANGED')
        if hashes:
            require(sha256(path)==entry['sha256'], 'VIEW_CHECKSUM', str(path))
    for origin in value['origins']:
        require(stamp(origin['path'])==origin['stamp'], 'ORIGINAL_CHANGED', origin['path'])
        if hashes:
            require(sha256(origin['path'])==origin['sha256'], 'ORIGINAL_CHECKSUM')
    return value


@dataclass
class RecordBatch:
    tensors: dict
    refs: list
    def pin_memory(self):
        self.tensors = {k:v.pin_memory() for k,v in self.tensors.items()}
        return self
    def to(self, device):
        return RecordBatch({k:v.to(device, non_blocking=True) for k,v in self.tensors.items()}, self.refs)
    def __len__(self):
        return len(self.refs)


def collate_records(rows):
    return RecordBatch({k:torch.stack([r[k] for r in rows]) for k in FIELDS}, [r['ref'] for r in rows])


class RecordStream(IterableDataset):
    def __init__(self, root, split, epoch=0, seed=7, shuffle=True):
        self.root, self.split, self.seed, self.shuffle = Path(root), split, seed, shuffle
        self.epoch_shared = multiprocessing.Value('q', epoch)
        self.manifest = verify_snapshot(root, hashes=False)
        self.manifest_hash = sha256(self.root/'snapshot.json')
    @property
    def epoch(self):
        return self.epoch_shared.value
    def set_epoch(self, epoch):
        self.epoch_shared.value = epoch
    def order(self):
        return [entry for group in self.group_order() for entry in group]
    def group_order(self):
        shards = [e for e in self.manifest['shards'] if e['split']==self.split]
        grouped={}
        for entry in shards:
            grouped.setdefault(entry['case_index'],[]).append(entry)
        groups=list(grouped.values())
        if self.shuffle:
            ids = torch.randperm(len(groups), generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            groups = [groups[i] for i in ids]
        return groups
    def __iter__(self):
        require(sha256(self.root/'snapshot.json')==self.manifest_hash,'SNAPSHOT_MANIFEST_CHANGED')
        info = get_worker_info()
        worker, count = (info.id,info.num_workers) if info else (0,1)
        assigned=[e for group in self.group_order()[worker::count] for e in group]
        for entry in assigned:
            path = self.root/entry['path']
            require(stamp(path)==entry['stamp'],'SNAPSHOT_CHANGED',str(path))
            payload = torch.load(path,map_location='cpu',weights_only=True)
            require(stamp(path)==entry['stamp'],'SNAPSHOT_CHANGED_DURING_READ')
            for i in range(entry['records']):
                yield {**{k:payload[k][i] for k in FIELDS},'ref':payload['refs'][i]}
    def __len__(self):
        return sum(e['records'] for e in self.order())
    def batch_count(self, microbatch, workers):
        groups = self.group_order()
        return sum(math.ceil(sum(e['records'] for group in groups[i::max(1,workers)] for e in group)/microbatch) for i in range(max(1,workers)))


def record_loader(root, split, *, epoch=0, seed=7, microbatch=16, workers=4, prefetch_factor=2, pin_memory=True, shuffle=True):
    stream = RecordStream(root,split,epoch,seed,shuffle)
    kwargs = {'num_workers':workers,'pin_memory':pin_memory,'collate_fn':collate_records,'drop_last':False}
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor,persistent_workers=True)
    return DataLoader(stream,batch_size=microbatch,generator=torch.Generator().manual_seed(seed),**kwargs)
