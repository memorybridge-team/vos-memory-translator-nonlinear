"""명시적으로 시작하는 evaluation worker/merge. Scheduler·cloud lifecycle 기능 없음."""
from pathlib import Path
from datetime import datetime, timezone
import time
import uuid
from contextlib import nullcontext
from .collection_contract import require
from .training_storage import content_hash, sha256, write_json, ExclusiveWriter
from .training_data import read_manifest
from .training_runner import code_provenance
from .lvos_contract import json_read, load_complete
from .lvos_budget import Deadline, BudgetStop, check
from .lvos_evaluation import case_assignments, read_result, validate_rows, merge_results


class GPULease(ExclusiveWriter):
    """공유 root의 host/physical UUID 단위 advisory lease; stale lock 자동 제거 없음."""
    def __init__(self,root,host,gpu_uuid,owner):
        require(root and host and gpu_uuid and owner,'GPU_LEASE_INPUTS')
        self.identity={'host':host,'physical_gpu_uuid':gpu_uuid,'owner':owner}
        self.root=Path(root)/content_hash({'host':host,'gpu':gpu_uuid})
        super().__init__(self.root)
    def __enter__(self):
        super().__enter__()
        try:
            write_json(self.root/'lease.json',{**self.identity,'timestamp_utc':datetime.now(timezone.utc).isoformat()})
        except Exception:
            super().__exit__(); raise
        return self
    def __exit__(self,*args):
        try: return super().__exit__(*args)
        finally: del self.fd


def publish_request(run,checkpoint,protocol_path,metric_contract):
    run=Path(run).resolve(); checkpoint=Path(checkpoint).resolve()
    require(checkpoint.is_relative_to(run/'checkpoints'),'REQUEST_CHECKPOINT_OUTSIDE_RUN')
    payload,entry=load_complete(checkpoint); protocol=read_manifest(protocol_path); contract=json_read(metric_contract)
    require(payload['identity']['run_id']==run.name and payload['identity']['collection_digest']==protocol['collection_digest'] and
            payload['identity']['metric_contract_digest']==content_hash(contract),'REQUEST_IDENTITY')
    require(protocol['scope']=='monitor','REQUEST_MONITOR_PROTOCOL')
    value={'schema_version':'cmmt.lvos_eval_request.v2','epoch':payload['epoch'],'untrained':payload['epoch']==0,
        'identity':payload['identity'],'checkpoint':str(checkpoint),'checkpoint_sha':entry['sha256'],
        'protocol':str(Path(protocol_path).resolve()),'protocol_digest':protocol['content_sha256'],
        'protocol_file_sha256':sha256(protocol_path),'metric_contract':str(Path(metric_contract).resolve()),
        'metric_contract_digest':content_hash(contract),'metric_file_sha256':sha256(metric_contract)}
    value['request_digest']=content_hash(value)
    path=run/f"evaluation/requests/epoch-{payload['epoch']:05d}.json"
    if path.exists():
        require(json_read(path)==value and json_read(str(path)+'.complete.json')['sha256']==sha256(path),'REQUEST_CONFLICT')
    else:
        write_json(path,value); write_json(str(path)+'.complete.json',{'sha256':sha256(path)})
    return path


def read_request(run,path):
    run=Path(run).resolve(); path=Path(path).resolve()
    require(path.is_relative_to(run/'evaluation/requests'),'REQUEST_FOREIGN_RUN')
    require(json_read(str(path)+'.complete.json')['sha256']==sha256(path),'REQUEST_NOT_READY')
    value=json_read(path)
    require(value['schema_version']=='cmmt.lvos_eval_request.v2' and value['request_digest']==
            content_hash({k:v for k,v in value.items() if k!='request_digest'}),'REQUEST_SCHEMA_OR_DIGEST')
    require(value['identity']==json_read(run/'config.json')['identity'] and
            value['identity']['code_sha']==code_provenance()['package_source_sha256'],'REQUEST_STALE_OR_FOREIGN')
    require(Path(value['checkpoint']).resolve().is_relative_to(run/'checkpoints'),'REQUEST_FOREIGN_CHECKPOINT')
    payload,entry=load_complete(value['checkpoint'])
    require(entry['sha256']==value['checkpoint_sha'] and payload['identity']==value['identity'] and
            payload['epoch']==value['epoch'] and payload['boundary']=='complete_epoch','REQUEST_CHECKPOINT_CHANGED')
    protocol=read_manifest(value['protocol'])
    require(sha256(value['protocol'])==value['protocol_file_sha256'] and protocol['content_sha256']==value['protocol_digest'] and
            protocol['collection_digest']==value['identity']['collection_digest'] and
            sha256(value['metric_contract'])==value['metric_file_sha256'] and
            content_hash(json_read(value['metric_contract']))==value['metric_contract_digest'],'REQUEST_PROTOCOL_OR_METRIC_CHANGED')
    return value,protocol


def verify_worker_result(result,request,protocol,index,count):
    validate_rows(result,protocol)
    require(result['epoch']==request['epoch'] and result['method']=='learned' and
            result.get('code_sha')==request['identity']['code_sha'] and
            result['checkpoint_sha']==request['checkpoint_sha'] and result['protocol_digest']==request['protocol_digest'] and
            result['metric_contract_digest']==request['metric_contract_digest'] and
            result['collection_digest']==request['identity']['collection_digest'] and
            result['model_config_digest']==request['identity']['model_config_digest'] and
            result['shard_index']==index and result['shard_count']==count and
            result['expected_case_ids']==case_assignments(protocol,count)[index],'WORKER_RESULT_STALE_OR_FOREIGN')


def merge_request(run,request_path,benchmark_root,*,count=1):
    run=Path(run).resolve(); request,protocol=read_request(run,request_path)
    paths=[]
    for index in range(count):
        pointer=run/f"evaluation/workers/worker-{index}/{request['request_digest']}/verified.json"
        require(pointer.is_file(),'WORKER_SHARD_PENDING')
        entry=json_read(pointer); require(entry['sha256']==sha256(entry['path']),'WORKER_POINTER_CHANGED')
        path=Path(entry['path']).resolve()
        require(path.is_relative_to(run/f'evaluation/workers/worker-{index}'),'WORKER_POINTER_FOREIGN')
        result=read_result(path); verify_worker_result(result,request,protocol,index,count); paths.append(path)
    output=run/f"evaluation/results/epoch-{request['epoch']:05d}"
    if (output/'result.json').exists():
        existing=read_result(output/'result.json'); validate_rows(existing,protocol)
        expected=[{'path':str(p),'sha256':sha256(p)} for p in paths]
        require(existing.get('merged') and existing['checkpoint_sha']==request['checkpoint_sha'] and
                existing['source_artifacts']==expected and existing['protocol_digest']==request['protocol_digest'] and
                existing['metric_contract_digest']==request['metric_contract_digest'],'MERGE_CONFLICT')
        return existing  # exactly once, verified idempotent reuse.
    return merge_results(paths,request['protocol'],request['metric_contract'],benchmark_root,output)


def pending_requests(run):
    run=Path(run); pending=[]
    for path in sorted((run/'evaluation/requests').glob('epoch-[0-9][0-9][0-9][0-9][0-9].json')):
        request,protocol=read_request(run,path)
        result_path=run/f"evaluation/results/epoch-{request['epoch']:05d}/result.json"
        if not result_path.exists(): pending.append(str(path)); continue
        result=read_result(result_path); validate_rows(result,protocol)
        require(result.get('merged') and result['checkpoint_sha']==request['checkpoint_sha'] and
                result.get('code_sha')==request['identity']['code_sha'] and
                result['epoch']==request['epoch'] and result['method']=='learned' and
                result['collection_digest']==request['identity']['collection_digest'] and
                result['model_config_digest']==request['identity']['model_config_digest'] and
                result['protocol_digest']==request['protocol_digest'] and result['metric_contract_digest']==request['metric_contract_digest'],
                'PENDING_RESULT_CONFLICT')
    return pending


def run_worker(run,snapshot,benchmark_root,*,index=0,count=1,dry_run=True,drain=False,approved=False,
               device='cuda:0',max_wall_seconds=None,deadline=None,executor=None,merge=False,lease=None):
    run=Path(run).resolve(); worker=run/f'evaluation/workers/worker-{index}'
    require(0<=index<count,'WORKER_SHARD_INDEX')
    deadline=deadline or Deadline(max_wall_seconds)
    require(dry_run or executor is not None or (approved and lease is not None and hasattr(lease,'fd')),'WORKER_EXPLICIT_EXECUTION_APPROVAL')
    records=[]; stopped=None
    context=nullcontext() if dry_run else ExclusiveWriter(worker)
    with context:
        try:
            while True:
                check(deadline,'evaluation_worker_scan')
                requests=sorted((run/'evaluation/requests').glob('epoch-[0-9][0-9][0-9][0-9][0-9].json'))
                for path in requests:
                    check(deadline,'evaluation_worker_request')
                    request,protocol=read_request(run,path); assigned=case_assignments(protocol,count)[index]
                    require(assigned,'WORKER_EMPTY_ASSIGNMENT')
                    base=worker/request['request_digest']; pointer=base/'verified.json'
                    if dry_run:
                        records.append({'request':str(path),'request_digest':request['request_digest'],'case_ids':assigned,
                                        'shard_index':index,'shard_count':count,'status':'DRY_RUN'}); continue
                    if pointer.exists():
                        entry=json_read(pointer); require(entry['sha256']==sha256(entry['path']),'WORKER_POINTER_CHANGED')
                        verified=read_result(entry['path']); verify_worker_result(verified,request,protocol,index,count)
                        records.append({'request':str(path),'status':'VERIFIED_SKIP'}); continue
                    recovered=False
                    for candidate in sorted(base.glob('attempt-*/result.json')):
                        if not (candidate.parent/'READY.json').exists(): continue
                        result=read_result(candidate); verify_worker_result(result,request,protocol,index,count)
                        write_json(pointer,{'path':str(candidate),'sha256':sha256(candidate)})
                        records.append({'request':str(path),'status':'RECOVERED_VERIFIED_RESULT'})
                        recovered=True; break
                    if recovered: continue
                    attempt=base/('attempt-'+uuid.uuid4().hex)
                    try:
                        if executor is None:
                            from .lvos_evaluation import evaluate
                            result=evaluate(snapshot,request['protocol'],request['checkpoint'],request['metric_contract'],benchmark_root,attempt,
                                shard_index=index,shard_count=count,device=device,approved=approved,
                                max_wall_seconds=max_wall_seconds,deadline=deadline)
                        else:
                            result=executor(request,protocol,index,count,attempt,deadline)
                            require(result['execution_kind']=='cpu_synthetic','FAKE_WORKER_CANNOT_CLAIM_GPU')
                        result=read_result(attempt/'result.json'); verify_worker_result(result,request,protocol,index,count)
                        check(deadline,'evaluation_worker_verify_complete')
                        write_json(pointer,{'path':str(attempt/'result.json'),'sha256':sha256(attempt/'result.json')})
                        records.append({'request':str(path),'status':'VERIFIED_COMPLETE'})
                    except BudgetStop: raise
                    except Exception as exc:
                        write_json(attempt/'FAILED.json',{'status':'FAILED','reason':getattr(exc,'code',str(exc))})
                        records.append({'request':str(path),'status':'FAILED','attempt':str(attempt)}); continue
                if merge and not dry_run:
                    for path in requests:
                        check(deadline,'evaluation_worker_merge')
                        try: merge_request(run,path,benchmark_root,count=count)
                        except ValueError as exc:
                            if getattr(exc,'code',None)!='WORKER_SHARD_PENDING': raise
                if dry_run or not drain: break
                require(max_wall_seconds is not None,'WORKER_DRAIN_REQUIRES_DEADLINE')
                pending=pending_requests(run)
                training=json_read(run/'STATUS.json').get('training_status','RUNNING')
                if not pending and training in {'COMPLETED','STOPPED','FAILED'}: break
                check(deadline,'evaluation_worker_wait'); time.sleep(min(1,max(0,deadline.ends-deadline.clock())))
        except BudgetStop as exc:
            stopped=exc.stage
        summary={'schema_version':'cmmt.lvos_worker.v1','dry_run':dry_run,'execution_performed':not dry_run,
                 'status':'STOPPED' if stopped else 'FAIL' if any(r['status']=='FAILED' for r in records) else 'DONE',
                 'incomplete_stage':stopped,'run':str(run),'shard_index':index,'shard_count':count,'records':records,
                 'execution_kind':'cpu_synthetic' if executor is not None else 'plan_only' if dry_run else 'gpu_real_checkpoint'}
        if not dry_run:
            write_json(worker/'STATUS.json',summary)
        return summary
