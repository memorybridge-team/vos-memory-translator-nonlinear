"""완료 cache의 읽기 전용 discovery. 파일명으로 case/provenance를 추정하지 않는다."""
from pathlib import Path
import time
import torch
from .collection_contract import require, validate_generating
from .cache_migration import inspect_cache, _check_frozen
from .training_data import read_manifest
from .training_storage import sha256, content_hash, write_json, ExclusiveWriter
from .lvos_contract import json_read
from .lvos_budget import Deadline, check, BudgetStop


def discover(cache_roots, selection_path, output, *, evidence_roots=(), rgb_root=None,
             trusted=False, stable_seconds=60, run_evidence=None, max_wall_seconds=None):
    deadline=Deadline(max_wall_seconds)
    roots=[Path(p).resolve() for p in cache_roots]
    evidence_roots=[Path(p).resolve() for p in evidence_roots]
    output=Path(output).resolve()
    require(roots and len(roots)==len(set(roots)), 'DISCOVERY_ROOTS')
    require(all(not output.is_relative_to(p) for p in roots+evidence_roots), 'OUTPUT_INSIDE_ORIGINAL')
    require(stable_seconds>=0,'STABILITY_LIMIT')
    selection=read_manifest(selection_path)
    require(selection['dataset']=='LVOS v2','LVOS_ONLY')
    expected={c['case_id']:c for c in selection['cases']}
    conditions={}; bindings={}; metadata=[]; findings=[]; requests=[]; seen={}; bytes_total=valid_records=0
    proof=None; proof_path=None; duplicate_ids=set()
    if run_evidence:
        proof_path=Path(run_evidence).resolve(); proof=json_read(proof_path)
        require(proof.get('status')=='reviewed_immutable' and proof.get('reviewer') and
                proof.get('reviewed_at') and proof.get('source_evidence'),'RUN_EVIDENCE_UNREVIEWED')
    with ExclusiveWriter(output):
        require(not (output/'inventory.json').exists(),'DISCOVERY_OUTPUT_EXISTS')
        stopped=None
        try:
            for root in roots+evidence_roots:
                if not root.is_dir():
                    findings.append({'path':str(root),'status':'PENDING_EVIDENCE','reason':'RUNTIME_ROOT_UNAVAILABLE'})
                    continue
                for path in sorted(root.rglob('*.json')):
                    check(deadline,'discovery_metadata')
                    if path.name not in {'status.json','case-bindings.json'} and path.parent.name!='conditions':
                        continue
                    require(path.resolve().is_relative_to(root),'DISCOVERY_SYMLINK_ESCAPE')
                    require(path.stat().st_size<=16*1024**2,'DISCOVERY_METADATA_TOO_LARGE')
                    value=json_read(path); digest=sha256(path)
                    metadata.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
                    if path.parent.name=='conditions':
                        validate_generating(value); key=content_hash(value)
                        require(key not in conditions or conditions[key]==value,'CONDITIONS_CONFLICT')
                        conditions[key]=value
                    elif path.name=='case-bindings.json':
                        for case_id,binding in value.items():
                            require(case_id not in bindings or bindings[case_id]==binding,'CASE_BINDING_CONFLICT')
                            bindings[case_id]=binding
            candidates=set()
            for root in roots:
                if root.is_dir(): candidates.update(root.rglob('*.pt'))
            for path in sorted(candidates):
                check(deadline,'discovery_cache')
                record={'path':str(path.resolve()),'status':'PENDING_EVIDENCE','case_id':None}
                try:
                    require(any(path.resolve().is_relative_to(r) for r in roots),'DISCOVERY_SYMLINK_ESCAPE')
                    require(not Path(str(path)+'.writer.lock').exists(),'ACTIVE_WRITER')
                    before=path.stat()
                    require(time.time()-before.st_mtime>=stable_seconds,'UNSTABLE_FILE')
                    marker_path=Path(str(path)+'.complete.json'); sidecar=Path(str(path)+'.sha256')
                    require(sidecar.is_file(),'MISSING_CHECKSUM')
                    require(marker_path.is_file() or proof is not None,'COMPLETED_EVIDENCE_MISSING')
                    digest=sha256(path); record.update(sha256=digest,bytes=before.st_size)
                    require(sidecar.read_text(encoding='ascii').split()[0]==digest,'CHECKSUM_TAMPERED')
                    marker=json_read(marker_path) if marker_path.is_file() else None
                    if marker:
                        require(marker['sha256']==digest and marker['bytes']==before.st_size,'COMPLETION_MISMATCH')
                    require(trusted,'TRUST_REQUIRED')
                    # Authorization applies only to this team legacy format; no auto safe-load fallback.
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    require((path.stat().st_size,path.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns) and
                            sha256(path)==digest,'FILE_CHANGED')
                    generating=payload.get('metadata',{}).get('generating')
                    if generating is None and marker:
                        generating=conditions.get(marker.get('generating_sha256'))
                    request={'path':str(path.resolve()),'sha256':digest}
                    if proof is not None:
                        matching=[b for b in proof['bindings'] if b['cache_sha256']==digest]
                        require(len(matching)==1,'RUN_EVIDENCE_BINDING')
                        require(generating is None or generating==matching[0]['generating'],'GENERATING_MISMATCH')
                        generating=matching[0]['generating']
                        request.update(run_evidence={'path':str(proof_path),'sha256':sha256(proof_path)},completed_evidence=True)
                    require(generating is not None,'LEGACY_PROVENANCE_UNKNOWN')
                    validate_generating(generating); case_id=generating['case']['case_id']; record['case_id']=case_id
                    _check_frozen(generating['case'],selection)
                    require(case_id in expected and generating['case']['pair_mode']=='native_history','FOREIGN_OR_CONTROLLED_CASE')
                    if case_id in bindings:
                        require(bindings[case_id]==content_hash(generating),'CASE_BINDING_MISMATCH')
                    request.update(case_id=case_id,expected_generating=generating)
                    result,loaded,_=inspect_cache(request,trusted_team_legacy=True,stable_seconds=stable_seconds)
                    require(result['decision']=='CONVERTIBLE',result['reason_codes'][0])
                    if case_id in seen:
                        duplicate_ids.add(case_id)
                        require(False,'DUPLICATE_SEMANTIC_CASE')
                    seen[case_id]=str(path); requests.append(request)
                    count=loaded['source_canonical'].valid_record_count(); valid_records+=count; bytes_total+=before.st_size
                    record.update(status='COMPLETED_PAIR_VERIFIED',valid_records=count,paired_split=generating['case']['paired_split'],
                                  generating_digest=content_hash(generating),models=generating['models'],
                                  provenance='embedded_or_explicit_reviewed_binding; checkpoint bytes not independently available')
                except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
                    record['reason']=getattr(exc,'code','DISCOVERY_INPUT_ERROR')
                    if record['reason'] in {'CHECKSUM_TAMPERED','COMPLETION_MISMATCH','DUPLICATE_SEMANTIC_CASE','FOREIGN_OR_CONTROLLED_CASE',
                                            'PAIR_ALIGNMENT','INVALID_CACHE_PAIR','PROMPT_TIMING'}:
                        record['status']='REJECTED'
                findings.append(record)
        except BudgetStop as exc:
            stopped=exc.stage
        requests=[r for r in requests if r['case_id'] not in duplicate_ids]
        for finding in findings:
            if finding.get('case_id') in duplicate_ids:
                finding.update(status='REJECTED',reason='DUPLICATE_SEMANTIC_CASE')
        completed=[f for f in findings if f['status']=='COMPLETED_PAIR_VERIFIED']
        bytes_total=sum(f['bytes'] for f in completed); valid_records=sum(f['valid_records'] for f in completed)
        ids={r['case_id'] for r in requests}
        result={'schema_version':'cmmt.lvos_discovery.v1','execution_kind':'cpu_fixture' if requests and
                all(r['expected_generating'].get('synthetic') for r in requests) else 'cpu_runtime_inventory',
                'selection_digest':selection['selection_digest'],'selection_file_sha256':sha256(selection_path),
                'cache_roots':[str(r) for r in roots],'evidence_artifacts':metadata,
                'rgb_root':{'path':str(rgb_root) if rgb_root else None,'exists':bool(rgb_root and Path(rgb_root).is_dir()),
                            'content_verified':False},
                'expected_cases':len(expected),'completed_pairs':len(requests),'valid_memory_records':valid_records,
                'completed_cache_bytes':bytes_total,'missing_case_ids':sorted(expected.keys()-ids),
                'duplicate_case_ids':sorted(duplicate_ids),'rejected':[f for f in findings if f['status']=='REJECTED'],
                'unknown':[f for f in findings if f['status']=='PENDING_EVIDENCE'],'inventory':findings,
                'status':'STOPPED' if stopped else 'READY' if ids==expected.keys() and not any(f['status']!='COMPLETED_PAIR_VERIFIED' for f in findings)
                         else 'BLOCKED','incomplete_stage':stopped,'trusted_team_legacy':trusted,
                'note':'Directory/status counts are not collection evidence; only verified pairs are counted. Originals are read-only.'}
        write_json(output/'requests.json',requests); write_json(output/'inventory.json',result)
        return result
