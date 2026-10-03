"""완료 운영 cache의 CPU 검사/index. 과거 provenance와 tensor 유효성을 분리한다.

원본을 변경하거나 unsafe pickle로 fallback하지 않는다. index는 학습용 derived
관리 파일이며 공식 research snapshot/G1 승인 자료로 사용하지 않는다.
"""
from pathlib import Path
import hashlib
import io
import math
import time
import torch

from .case_cache import validate_case_cache
from .state_schema import CanonicalState, StateSpec
from .transformer_translator import SAM21_MEMORY_SPEC
from .collection_contract import require
from .training_data import read_manifest, save_manifest
from .training_storage import ExclusiveWriter, write_json, content_hash
from .lvos_snapshot import stamp, collate_records
from .lvos_budget import Deadline, check

SCHEMA = 'cmmt.lvos_operational_cache_index.v1'
HISTORY_FIELDS = ('collector_revision', 'source_checkpoint_sha256',
                  'target_checkpoint_sha256', 'prompt_history', 'pair_mode')


def load_raw(path, *, expected_sha=None, expected_stamp=None, stable_seconds=0):
    path = Path(path)
    before = stamp(path)
    require(not Path(str(path)+'.writer.lock').exists(), 'RAW_ACTIVE_WRITER', str(path))
    require(time.time()-path.stat().st_mtime >= stable_seconds, 'RAW_NOT_STABLE', str(path))
    if expected_stamp is not None:
        require(before == expected_stamp, 'RAW_CHANGED', str(path))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = Path(str(path)+'.sha256')
    require(sidecar.is_file() and sidecar.read_text().split()[0] == digest, 'RAW_CHECKSUM', str(path))
    if expected_sha is not None:
        require(digest == expected_sha, 'RAW_INDEX_CHECKSUM', str(path))
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    require(stamp(path) == before, 'RAW_CHANGED_DURING_READ', str(path))
    validate_case_cache(value)
    source, target = value['source_canonical'], value['target_canonical']
    require(source.spec == target.spec == SAM21_MEMORY_SPEC, 'FROZEN_MODEL_SPEC')
    require(source.spatial_memory.dtype == target.spatial_memory.dtype == torch.bfloat16 and
            source.object_pointer.dtype == target.object_pointer.dtype == torch.float32, 'RUNTIME_DTYPES')
    require(source.validity.shape[:2] == (1, 1), 'SINGLE_OBJECT_CACHE_REQUIRED')
    return value, digest, before


def frozen_cases():
    root = Path(__file__).resolve().parents[2]/'manifests'
    source = read_manifest(root/'lvosv2_train_v1.json')
    fit = read_manifest(root/'lvosv2_train_v1_fit.json')
    dev = read_manifest(root/'lvosv2_train_v1_development.json')
    require(not set(fit['videos']) & set(dev['videos']), 'FIT_DEV_VIDEO_OVERLAP')
    require(set(fit['videos']) | set(dev['videos']) == {c['video_id'] for c in source['cases']}, 'MEMBERSHIP_COVERAGE')
    cases = [{**c, 'paired_split': 'fit' if c['video_id'] in fit['videos'] else 'development'} for c in source['cases']]
    return cases, {'source': source['content_sha256'], 'fit': fit['content_sha256'], 'development': dev['content_sha256']}


def inspect_entry(case, path, rgb_root, *, stable_seconds=60, accept_observed_prompt=False):
    value, digest, before = load_raw(path, stable_seconds=stable_seconds)
    md = value['metadata']; source = value['source_canonical']; target = value['target_canonical']
    require(md.get('source_model_id') == 'sam2.1-small' and md.get('target_model_id') == 'sam2.1-base-plus', 'MODEL_DIRECTION')
    require(md.get('video_id') == case['video_id'] and str(md.get('object_id')) == str(case['object_id']) and
            tuple(map(str, source.object_ids)) == (str(case['object_id']),), 'RAW_CASE_IDENTITY')
    frames = sorted(int(p.stem) for p in (Path(rgb_root)/case['video_id']).glob('*.jpg'))
    require(frames and len(frames) == len(set(frames)), 'RGB_FRAME_MAP_REQUIRED')
    require(case['switch_frame'] in frames and case['first_prompt_frame'] in frames, 'OFFICIAL_FRAME_MISSING')
    require(source.switch_frame == frames.index(case['switch_frame']) and md['num_frames'] == len(frames), 'SPARSE_FRAME_MAPPING')
    prompt = frames.index(case['first_prompt_frame'])
    valid_frames = source.frame_indices[source.validity]
    prompt_matches = bool((valid_frames >= prompt).all())
    require(prompt_matches or accept_observed_prompt, 'RECORD_BEFORE_PROMPT',
            f"case={case['case_id']}, official_prompt={case['first_prompt_frame']}, runtime_prompt={prompt}, records={valid_frames.tolist()}")
    require(bool((source.is_conditioning & source.validity).any()), 'EMPTY_PROMPT_STATE')
    policy = {k: md.get(k) for k in ('cache_mode','active_memory_only','num_maskmem','max_obj_ptrs_in_encoder')}
    require(policy == {'cache_mode':'state_only','active_memory_only':True,'num_maskmem':7,'max_obj_ptrs_in_encoder':16}, 'MEMORY_POLICY')
    known = {k: md[k] for k in HISTORY_FIELDS if k in md}
    unknown = [k for k in HISTORY_FIELDS if k not in md]
    entry = {'case':case, 'path':str(Path(path).resolve()), 'sha256':digest, 'stamp':before,
             'valid_indices':torch.nonzero(source.validity).tolist(), 'records':int(source.validity.sum()),
             'padded_records':source.validity.numel()-int(source.validity.sum()), 'memory_policy':policy,
             'runtime_switch_frame':source.switch_frame, 'runtime_prompt_frame':prompt,
             'prompt_contract_status':'MATCH' if prompt_matches else 'ACCEPTED_EXISTING_HISTORY_FOR_STATE_TRAINING_ONLY',
             'conditioning_frames_observed':source.frame_indices[source.validity & source.is_conditioning].tolist(),
             'frame_map_digest':content_hash(frames), 'tensor_contract':{
                 'spatial_shape':list(source.spatial_memory.shape), 'pointer_shape':list(source.object_pointer.shape),
                 'spatial_dtype':str(source.spatial_memory.dtype), 'pointer_dtype':str(source.object_pointer.dtype)},
             'historical_provenance':{'known':known, 'unknown':unknown, 'status':'UNKNOWN' if unknown else 'OBSERVED_UNVERIFIED'},
             'synthetic_declared':bool(md.get('synthetic',False)), 'normalization_sums':{}}
    if case['paired_split'] == 'fit':
        for name, tensor in (('spatial',target.spatial_memory),('pointer',target.object_pointer)):
            # BF16 -> FP64 accumulation; development never contributes.
            v = tensor[target.validity].double()
            entry['normalization_sums'][name] = [float(v.square().sum()), v.numel()]
    return entry


def build_index(fit_root, dev_root, rgb_root, output, *, operator_completed, case_ids=None,
                stable_seconds=60, max_wall_seconds=None, prompt_decision=None, exclusion_plan=None):
    require(bool(operator_completed), 'OPERATOR_COMPLETION_CONFIRMATION_REQUIRED')
    output = Path(output).resolve(); deadline = Deadline(max_wall_seconds)
    roots = {'fit':Path(fit_root).resolve(), 'development':Path(dev_root).resolve()}
    require(all(not output.is_relative_to(r) for r in roots.values()), 'OUTPUT_INSIDE_ORIGINAL')
    cases, splits = frozen_cases()
    accepted = set()
    if prompt_decision is not None:
        require(prompt_decision.get('schema_version') == 'cmmt.state_training_prompt_decision.v1' and
                prompt_decision.get('scope') == 'state_supervised_only' and prompt_decision.get('approved_by') and
                prompt_decision.get('reason') and prompt_decision.get('recorded_at'), 'PROMPT_DECISION_APPROVAL_REQUIRED')
        accepted = set(prompt_decision['case_ids'])
        require(len(accepted) == len(prompt_decision['case_ids']) and accepted <= {c['case_id'] for c in cases}, 'PROMPT_DECISION_CASES')
    if case_ids:
        require(len(case_ids) == len(set(case_ids)) and set(case_ids) <= {c['case_id'] for c in cases}, 'FOREIGN_CASE')
        cases = [c for c in cases if c['case_id'] in case_ids]
    expected = {c['paired_split']:set() for c in cases}
    for c in cases:
        expected[c['paired_split']].add(f"lvos_{c['video_id']}_obj{c['object_id']}_switch{c['switch_frame']}.pt")
    inventory = {role:{p.name for p in root.glob('*.pt')} for role,root in roots.items()}
    for role in expected:
        require(expected[role] <= inventory[role], 'EXPECTED_CACHE_MISSING', role)
        if not case_ids:
            require(expected[role] == inventory[role], 'FOREIGN_CACHE_OR_VALIDATION', role)
    excluded = set()
    if exclusion_plan is not None:
        require(not case_ids and exclusion_plan.get('schema_version') == 'cmmt.state_training_exclusions.v1' and
                exclusion_plan.get('scope') == 'state_supervised_only', 'EXCLUSION_PLAN_SCHEMA')
        rows = exclusion_plan['cases']; excluded = {e['case_id'] for e in rows}
        require(len(rows) == len(excluded) and all(e.get('reason') for e in rows) and
                excluded <= {c['case_id'] for c in cases if c['paired_split']=='fit'}, 'EXCLUSION_FIT_CASES_ONLY')
        cases = [c for c in cases if c['case_id'] not in excluded]
    identity = {'splits':splits,'case_ids':[c['case_id'] for c in cases],
                'roots':{k:str(v) for k,v in roots.items()},'rgb_root':str(Path(rgb_root).resolve()),
                'operator_completion':operator_completed, 'prompt_policy_decision':prompt_decision,
                'exclusion_plan':exclusion_plan,
                'scope':'subset' if case_ids else 'full_frozen_fit_development_with_exclusions' if excluded else 'full_frozen_fit_development'}
    entries = []; failures = []
    with ExclusiveWriter(output):
        control = output/'index_inputs.json'
        if control.exists():
            from .lvos_contract import json_read
            require(json_read(control) == identity, 'INDEX_RESUME_INPUT_CHANGED')
        else: write_json(control, identity)
        if (output/'raw_index.json').exists():
            return read_index(output/'raw_index.json')
        for i,case in enumerate(cases):
            check(deadline, 'raw_index_case')
            path = roots[case['paired_split']]/f"lvos_{case['video_id']}_obj{case['object_id']}_switch{case['switch_frame']}.pt"
            journal = output/f'cases/{i:05d}.json'
            if journal.exists():
                entry = read_manifest(journal)
                require(entry['case'] == case and entry['path'] == str(path) and stamp(path) == entry['stamp'], 'INDEX_RESUME_CACHE_CHANGED')
            else:
                try:
                    entry = inspect_entry(case,path,rgb_root,stable_seconds=stable_seconds,
                                          accept_observed_prompt=case['case_id'] in accepted)
                    save_manifest(journal,entry)
                except (ValueError,TypeError,KeyError,RuntimeError) as exc:
                    failures.append({'case_id':case['case_id'],'split':case['paired_split'],
                                     'path':str(path),'reason':getattr(exc,'code',type(exc).__name__),'detail':str(exc)})
                    write_json(output/'FAILURES.json',{'status':'FAIL','failed_cases':failures,'training_ready':False})
                    print(f"입력 제외 없이 학습 중단 대상 기록: {case['case_id']}: {exc}",flush=True)
                    continue
            entry = {k:v for k,v in entry.items() if k != 'content_sha256'}
            entries.append(entry)
            require(entry['memory_policy'] == entries[0]['memory_policy'], 'MIXED_MEMORY_POLICY')
            if i < 3 or (i+1)%20 == 0 or i+1 == len(cases):
                elapsed = deadline.elapsed()
                print(f'CPU cache 검사 {i+1}/{len(cases)}; {elapsed:.1f}s; 예상 잔여 {(len(cases)-i-1)*elapsed/(i+1):.1f}s',flush=True)
        write_json(output/'INSPECTION.json',{'status':'FAIL' if failures else 'PASS','expected_cases':len(cases),
                  'validated_cases':len(entries),'failures':failures,'duration_seconds':deadline.elapsed(),
                  'training_ready':not failures,'historical_provenance':'UNKNOWN','original_modified':False})
        require(not failures,'CACHE_ALIGNMENT_FAILURES',str(output/'FAILURES.json'))
        require(accepted == {e['case']['case_id'] for e in entries if e['prompt_contract_status'] != 'MATCH'}, 'PROMPT_DECISION_EXACT_DEVIATIONS')
        sums = {k:[sum(e['normalization_sums'].get(k,[0,0])[0] for e in entries),
                   sum(e['normalization_sums'].get(k,[0,0])[1] for e in entries)] for k in ('spatial','pointer')}
        require(all(n > 0 for _,n in sums.values()), 'NO_FIT_NORMALIZATION')
        value = {'schema_version':SCHEMA, 'state':'tensor_validated', 'identity':identity, 'entries':entries,
                 'normalization':{'scope':'fit_only','scales':{k:math.sqrt(s/n) for k,(s,n) in sums.items()},
                                  'sums':sums,'computations':1},
                 'statistics':{role:{'cases':sum(e['case']['paired_split']==role for e in entries),
                                    'records':sum(e['records'] for e in entries if e['case']['paired_split']==role)} for role in roots},
                 'synthetic_declared':any(e['synthetic_declared'] for e in entries),
                 'historical_provenance_status':'UNKNOWN', 'research_gates_passed':False,
                 'original_modified':False, 'duration_seconds':deadline.elapsed()}
        save_manifest(output/'raw_index.json',value)
        return read_manifest(output/'raw_index.json')


def read_index(path):
    value = read_manifest(path)
    require(value['schema_version'] == SCHEMA and value['state'] == 'tensor_validated', 'RAW_INDEX_SCHEMA')
    cases, splits = frozen_cases()
    require(value['identity']['splits'] == splits, 'FROZEN_SPLIT_CHANGED')
    known = {c['case_id']:c for c in cases}
    require(len({e['case']['case_id'] for e in value['entries']}) == len(value['entries']), 'DUPLICATE_INDEX_CASE')
    for e in value['entries']:
        require(e['case'] == known.get(e['case']['case_id']), 'RAW_INDEX_FOREIGN_CASE')
        require(stamp(e['path']) == e['stamp'], 'RAW_CHANGED')
        require(e['records'] == len(e['valid_indices']) > 0 and
                len({tuple(x) for x in e['valid_indices']}) == e['records'], 'INDEX_RECORD_IDENTITIES')
    if value['identity']['scope'] == 'full_frozen_fit_development':
        require({e['case']['case_id'] for e in value['entries']} == set(known), 'INDEX_FULL_COVERAGE')
    elif value['identity']['scope'] == 'full_frozen_fit_development_with_exclusions':
        excluded = {e['case_id'] for e in value['identity']['exclusion_plan']['cases']}
        used = {e['case']['case_id'] for e in value['entries']}
        require(not used & excluded and used | excluded == set(known) and
                all(known[c]['paired_split']=='fit' for c in excluded), 'INDEX_EXCLUSION_COVERAGE')
    require(value['normalization']['scope'] == 'fit_only', 'NORMALIZATION_SCOPE')
    return value


def records(entries):
    """case 단위 읽기/순서 유지. 원본 SHA를 실제 읽은 bytes에서 매번 확인한다."""
    previous = None; value = None
    for entry in entries:
        if previous != entry['path']:
            value,_,_ = load_raw(entry['path'],expected_sha=entry['sha256'],expected_stamp=entry['stamp'])
            previous = entry['path']
        s,t = value['source_canonical'],value['target_canonical']
        require(torch.nonzero(s.validity).tolist() == entry['valid_indices'], 'INDEX_VALIDITY_CHANGED')
        selected = entry.get('selected_positions',range(len(entry['valid_indices'])))
        for position in selected:
            b,o,k = entry['valid_indices'][position]
            yield {'source_spatial':s.spatial_memory[b,o,k], 'target_spatial':t.spatial_memory[b,o,k],
                   'source_pointer':s.object_pointer[b,o,k], 'target_pointer':t.object_pointer[b,o,k],
                   'frame':s.frame_indices[b,o,k], 'slot':s.slot_order[b,o,k], 'conditioning':s.is_conditioning[b,o,k],
                   'ref':[entry['case']['case_id'],b,o,k]}
