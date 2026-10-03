"""한 translator의 same-host DDP state 학습. 외부 VOS/J&F gate와 독립.

실행: python -m vos_memory_inspector.lvos_ddp --help
GPU 실행은 torchrun + 명시 승인/Pod 전체 단가/상한/UUID 필요.
"""
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from contextlib import nullcontext, ExitStack
import argparse
import json
import math
import os
import random
import subprocess
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .collection_contract import require
from .training_storage import ExclusiveWriter, write_json, content_hash, sha256
from .training_runner import rng_state, restore_rng, code_provenance
from .lvos_contract import model_lock, validate_model_lock, json_read, load_complete
from .lvos_training import LVOSConfig, component_objective, optimizer_groups, make_scheduler, publish_state_export
from .lvos_checkpoint import CheckpointStore
from .lvos_snapshot import collate_records
from .lvos_raw_cache import build_index, read_index, records
from .lvos_budget import BudgetStop, Deadline


class TensorModule(nn.Module):
    """DDP.forward를 통과하는 adapter. 고정 translator body/API는 변경하지 않는다."""
    def __init__(self, translator):
        super().__init__(); self.translator = translator
    def forward(self, spatial, pointer):
        return self.translator.translate_tensors(spatial[None,None].float(),pointer[None,None].float())


def schedule(entries, world, global_batch, *, seed=7, epoch=0, shuffle=True):
    require(world > 0 and global_batch > 0 and global_batch % world == 0, 'GLOBAL_BATCH_WORLD_DIVISIBILITY')
    order = list(entries)
    if shuffle:
        indices = torch.randperm(len(order),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        order = [order[i] for i in indices]
    assigned = [[] for _ in range(world)]; quota = global_batch//world
    windows = []; current = [0]*world
    for e in order:
        remaining = e['records']; position = 0
        while remaining:
            owner = sum(current)//quota
            n = min(remaining,quota-current[owner])
            assigned[owner].append({**e,'records':n,'selected_positions':list(range(position,position+n))})
            current[owner] += n; remaining -= n; position += n
            if sum(current) == global_batch:
                windows.append(current); current = [0]*world
    if sum(current): windows.append(current)
    require(sum(map(sum,windows)) == sum(e['records'] for e in entries), 'SCHEDULE_COVERAGE')
    return assigned, windows


def reduce_values(values, device):
    tensor = torch.tensor(values,dtype=torch.float64,device=device)
    dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def collective_deadline(deadline, stage, device):
    expired = 0
    try: deadline.check(stage)
    except BudgetStop: expired = 1
    vote = torch.tensor(expired,device=device)
    dist.all_reduce(vote,op=dist.ReduceOp.MAX)
    if int(vote): raise BudgetStop(stage)


def validate_main_index(index):
    require(index['identity']['scope'] in ('full_frozen_fit_development','full_frozen_fit_development_with_exclusions'), 'TRAIN_REQUIRES_FULL_INDEX')
    plan = index['identity'].get('exclusion_plan')
    require(plan is None or (plan.get('approved_by') and plan.get('recorded_at')), 'EXCLUSION_PLAN_NOT_APPROVED')


def gpu_status():
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        return r.stdout.strip().splitlines() if r.returncode == 0 else ['unavailable']
    except (FileNotFoundError, subprocess.TimeoutExpired): return ['unavailable']


def gradients(model):
    result = {}
    for part in ('spatial','pointer'):
        values = [p.grad for n,p in model.named_parameters() if n.startswith(part+'.') and p.grad is not None]
        require(bool(values) and all(bool(torch.isfinite(g).all()) for g in values), 'BRANCH_NONFINITE_OR_MISSING_GRADIENT')
        result[part] = math.sqrt(sum(float(g.float().square().sum()) for g in values))
    return result


def same_across_ranks(model, device):
    maximum = 0.
    for p in model.parameters():
        reference = p.detach().clone(); dist.broadcast(reference,src=0)
        maximum = max(maximum,float((reference-p.detach()).abs().max()))
    value = torch.tensor(maximum,device=device); dist.all_reduce(value,op=dist.ReduceOp.MAX)
    require(float(value) == 0, 'DDP_PARAMETERS_DIVERGED')
    return float(value)


def evaluate(model, entries, world, rank, microbatch, scales, config, device, deadline):
    assigned,_ = schedule(entries,world,max(world,64),shuffle=False)
    totals = [0.]*6; batch = []; model.eval()
    def consume(rows):
        t = collate_records(rows).to(device).tensors
        with torch.no_grad():
            s,p = model(t['source_spatial'],t['source_pointer'])
            raw,norm,loss = component_objective(s[0,0],p[0,0],t['target_spatial'],t['target_pointer'],
                         torch.ones(len(rows),dtype=torch.bool,device=device),scales,config)
        values = [float(raw['spatial']),float(raw['pointer']),float(norm['spatial']),float(norm['pointer']),float(loss),1.]
        for i,v in enumerate(values): totals[i] += v*len(rows)
    for row in records(assigned[rank]):
        deadline.check('development_record'); batch.append(row)
        if len(batch) == microbatch: consume(batch); batch = []
    if batch: consume(batch)
    total = reduce_values(totals,device)
    return {k:total[i]/total[-1] if total[-1] else None for i,k in enumerate(
        ('spatial_mse','pointer_mse','normalized_spatial','normalized_pointer','loss'))} | {'records':int(total[-1])}


def train(index_path, output, *, config=None, global_batch=64, mode='smoke', device='cpu', resume=False,
          max_wall_seconds=600, smoke_evidence=None, gpu_uuids=None, pod_hourly_rate=None, budget=None,
          execute_approved=False, lease_root=None, host_id=None, stop_after_epoch=None):
    config = config or LVOSConfig(); output = Path(output).resolve(); deadline = Deadline(max_wall_seconds)
    rank = int(os.environ.get('RANK','0')); world = int(os.environ.get('WORLD_SIZE','1'))
    local = int(os.environ.get('LOCAL_RANK','0'))
    cuda = device == 'cuda'
    require(world in (1,2,4) and config.microbatch > 0 and global_batch % (world*config.microbatch) == 0, 'DDP_WORLD_OR_BATCH')
    require(config.accumulation == global_batch//(world*config.microbatch),'DDP_ACCUMULATION_GLOBAL_BATCH')
    require(config.precision == 'fp32' and config.augmentation == 'none' and config.lambda_cos == 0, 'FROZEN_TRAINING_POLICY')
    require(config.max_epochs > 0 and max_wall_seconds > 0 and config.workers == 0, 'DDP_LIMITS_WORKERS_ZERO_V1')
    if cuda:
        uuids = gpu_uuids or []
        require(execute_approved and world in (2,4) and len(set(uuids)) == len(uuids) == world,
                'DDP_EXPLICIT_GPU_APPROVAL_UUIDS')
        require(all(u.startswith('GPU-') for u in uuids),'DDP_PHYSICAL_GPU_UUIDS_REQUIRED')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(uuids) and torch.cuda.device_count() == world,
                'DDP_VISIBLE_GPU_MAPPING')
        require(pod_hourly_rate is not None and budget is not None and pod_hourly_rate > 0 and budget > 0 and
                max_wall_seconds/3600*pod_hourly_rate <= budget, 'POD_PRICE_BUDGET_REQUIRED')
        require(lease_root and host_id,'DDP_LEASE_HOST_REQUIRED')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8'),'DETERMINISTIC_CUBLAS_CONFIG_REQUIRED')
        torch.cuda.set_device(local); actual_device = torch.device('cuda',local)
    else: actual_device = torch.device('cpu')
    torch.set_num_threads(1)
    init = {}
    if not cuda and os.environ.get('LVOS_CPU_STORE'):
        init = {'init_method':Path(os.environ['LVOS_CPU_STORE']).resolve().as_uri(),'rank':rank,'world_size':world}
    dist.init_process_group('nccl' if cuda else 'gloo',timeout=timedelta(seconds=120),**init)
    try:
        return _train(index_path,output,config,global_batch,mode,resume,deadline,smoke_evidence,
                      rank,world,actual_device,cuda,gpu_uuids,pod_hourly_rate,lease_root,host_id,stop_after_epoch)
    except BudgetStop as exc:
        if rank == 0:
            write_json(output/'STATUS.json',{'status':'STOPPED','reason':'deadline','stage':exc.stage,
                       'resume':'last complete epoch only; partial epoch discarded','research_gates_passed':False})
        raise
    except Exception as exc:
        if rank == 0:
            write_json(output/'FAILED.json',{'status':'FAIL','reason':str(exc),'research_gates_passed':False})
        raise
    finally:
        dist.destroy_process_group()


def _train(index_path,output,cfg,global_batch,mode,resume,deadline,smoke_evidence,rank,world,device,cuda,uuids,rate,
           lease_root,host_id,stop_after_epoch):
    index = read_index(index_path); lock = model_lock()
    require(not cuda or not index.get('synthetic_declared',True),'GPU_REQUIRES_REAL_CACHE_INDEX')
    provenance = code_provenance()
    contract = {'index_sha256':sha256(index_path),'model_config_digest':lock['digest'],
                'package_sha256':provenance['package_source_sha256'], 'global_batch':global_batch,
                'optimization_policy':{k:getattr(cfg,k) for k in ('lr','weight_decay','lambda_spatial','lambda_pointer',
                    'microbatch','clip_norm','warmup_fraction','min_lr','seed')}}
    if mode == 'train':
        validate_main_index(index)
        require(cuda and smoke_evidence is not None, 'TRAIN_REQUIRES_REAL_DDP_SMOKE')
        report = json_read(smoke_evidence)
        require(report.get('status') == 'REAL_DDP_SMOKE_PASS' and report.get('world_size') in (2,4) and
                report.get('training_contract') == contract and report.get('both_branches_updated') and
                report.get('strict_reload_equal') and report.get('optimizer_reload_equal') and
                report.get('rank_parameter_max_difference') == 0, 'REAL_DDP_SMOKE_BINDING')
        body,_ = load_complete(Path(smoke_evidence).parent/report['checkpoint']['path'])
        require(body['identity']['training_contract'] == contract,'SMOKE_CHECKPOINT_BINDING')
    entries = index['entries']; fit = [e for e in entries if e['case']['paired_split']=='fit']
    dev = [e for e in entries if e['case']['paired_split']=='development']
    if mode == 'smoke':
        require(sum(e['records'] for e in fit) >= world, 'SMOKE_REQUIRES_RECORD_PER_RANK')
        fit = fit[:max(world,math.ceil(global_batch/16))]
        dev = dev[:1]  # diagnostic smoke는 full-dev 반복 비용을 지불하지 않는다.
    require(fit and (mode != 'train' or dev), 'EMPTY_FIT_DEV')
    # Single index calculation; rank 0 broadcasts fit-only scalars to all ranks.
    scale_message = [index['normalization']['scales'] if rank == 0 else None]
    dist.broadcast_object_list(scale_message,src=0); scales = scale_message[0]
    require(scales == index['normalization']['scales'],'NORMALIZATION_RANK_MISMATCH')
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    import numpy as np
    np.random.seed(cfg.seed)
    if cuda: torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    model = validate_model_lock(lock,require_approval=False).to(device)
    adapter = TensorModule(model)
    ddp = DistributedDataParallel(adapter,device_ids=[device.index] if cuda else None,broadcast_buffers=False)
    groups,_ = optimizer_groups(model,cfg.weight_decay)
    optimizer = torch.optim.AdamW(groups,lr=cfg.lr)
    planned_updates = math.ceil(sum(e['records'] for e in fit)/global_batch)*cfg.max_epochs
    scheduler = make_scheduler(optimizer,cfg,planned_updates)
    identity = {'run_id':output.name, 'code_sha':provenance['commit'], 'model_config_digest':lock['digest'],
                'collection_digest':contract['index_sha256'], 'training_contract':contract,
                'configuration':{**asdict(cfg),'accumulation':'global_batch/(world_size*microbatch)'},
                'scope':'state_supervised_ddp','mode':mode}
    with ExitStack() as stack:
        if rank == 0:
            if cuda:
                from .lvos_jobs import GPULease
                for uuid in uuids: stack.enter_context(GPULease(lease_root,host_id,uuid,str(output)))
            stack.enter_context(ExclusiveWriter(output))
            require(resume or not (output/'run.json').exists(),'RUN_EXISTS_USE_RESUME')
            write_json(output/'run.json',{'identity':identity,'model_lock':lock,'code':provenance,'world_size':world,
                'gpu_uuids':uuids,'pod_hourly_rate':rate,'formal_model_approval':lock['approved_by'],
                'historical_provenance':index['historical_provenance_status'],'research_gates_passed':False,
                'normalization':index['normalization'],'JF_early_stopping_applied':False})
        dist.barrier()
        store = CheckpointStore(output,identity)
        loaded = [store.reconcile() if rank == 0 and resume else None]
        dist.broadcast_object_list(loaded,src=0)
        epoch_start = step = 0; history = []
        if resume:
            require(loaded[0] is not None,'RESUME_NO_COMPLETE_CHECKPOINT')
            payload,_ = loaded[0]
            model.load_state_dict(payload['export']['state_dict'],strict=True)
            optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
            require(payload['scales'] == scales,'RESUME_NORMALIZATION_CHANGED')
            epoch_start = payload['epoch']; step = payload['optimizer_step']; history = payload['history']
            saved = payload['rank_rng']
            restore_rng(saved[rank] if len(saved) == world else payload['rng'])
            if epoch_start == cfg.max_epochs:
                status = json_read(output/'STATUS.json')
                require(status.get('complete_epochs') == cfg.max_epochs,'COMPLETED_RESUME_STATUS')
                dist.barrier()
                return status
        start_weights = {part:{n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(part+'.')}
                         for part in ('spatial','pointer')}
        def checkpoint(epoch):
            states = [None]*world
            dist.all_gather_object(states,rng_state())
            entry = [None]
            if rank == 0:
                payload = {'schema_version':'cmmt.lvos_checkpoint.v1','identity':identity,'model_lock':lock,
                    'export':model.to_payload(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'rng':states[0],'rank_rng':states,'world_size':world,'scales':scales,'epoch':epoch,
                    'optimizer_step':step,'history':history,'boundary':'complete_epoch',
                    'resume_guarantee':'complete epoch only; changed world size preserves global batch, not exact trajectory'}
                entry[0] = store.publish(payload); publish_state_export(output,payload,entry[0])
            dist.broadcast_object_list(entry,src=0)
            return entry[0]
        if not resume: checkpoint(0)
        started = time.perf_counter(); last_entry = loaded[0][1] if resume else None
        for epoch in range(epoch_start,cfg.max_epochs):
            collective_deadline(deadline,'epoch_start',device); model.train()
            assigned,windows = schedule(fit,world,global_batch,seed=cfg.seed,epoch=epoch)
            if rank == 0:
                write_json(output/f'assignments/epoch-{epoch+1:05d}.json',{'global_batch':global_batch,
                   'ranks':[[{'case_id':e['case']['case_id'],'records':e['records'],
                             'selected_positions':e['selected_positions']} for e in group] for group in assigned],
                   'window_rank_records':windows,'tail_global_records':sum(windows[-1]),
                   'coverage':sum(map(sum,windows)),'expected':sum(e['records'] for e in fit)})
            iterator = iter(records(assigned[rank])); seen = []; sums = [0.]*6; epoch_time = time.perf_counter()
            for window in windows:
                collective_deadline(deadline,'optimizer_window',device); optimizer.zero_grad(set_to_none=True)
                iterations = max(math.ceil(n/cfg.microbatch) for n in window)
                left = window[rank]; local_sums = [0.]*6
                for micro in range(iterations):
                    n = min(left,cfg.microbatch); rows = [next(iterator) for _ in range(n)]; left -= n
                    context = ddp.no_sync() if micro+1 < iterations else nullcontext()
                    with context:
                        if rows:
                            batch = collate_records(rows).to(device); t = batch.tensors
                            s,p = ddp(t['source_spatial'],t['source_pointer'])
                            raw,norm,loss = component_objective(s[0,0],p[0,0],t['target_spatial'],t['target_pointer'],
                              torch.ones(n,dtype=torch.bool,device=device),scales,cfg)
                            (loss*n).backward(); seen.extend(batch.refs)
                            values = [float(raw['spatial'].detach()),float(raw['pointer'].detach()),
                                      float(norm['spatial'].detach()),float(norm['pointer'].detach()),float(loss.detach()),1.]
                            for i,v in enumerate(values): local_sums[i] += v*n
                        else:
                            s,p = ddp(torch.zeros((1,64,64,64),device=device),torch.zeros((1,256),device=device))
                            (s.sum()*0+p.sum()*0).backward()  # participate without duplicated/padded data loss
                total = reduce_values(local_sums,device); require(int(total[-1]) == sum(window),'WINDOW_RECORD_COUNT')
                # DDP averages summed gradients over W; undo W, divide by actual global valid records.
                for param in model.parameters():
                    if param.grad is not None: param.grad.mul_(world/total[-1])
                grads = gradients(model); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_norm)
                optimizer.step(); scheduler.step(); step += 1
                for i,v in enumerate(total): sums[i] += v
                if step <= 3: same_across_ranks(model,device)
                if rank == 0:
                    elapsed = time.perf_counter()-epoch_time; speed = sums[-1]/elapsed
                    row = {'epoch':epoch+1,'step':step,'global_valid_records':int(total[-1]),
                       'spatial_loss':total[0]/total[-1],'pointer_loss':total[1]/total[-1],
                       'normalized_spatial':total[2]/total[-1],'normalized_pointer':total[3]/total[-1],
                       'gradient_norms':grads,'lr':optimizer.param_groups[0]['lr'],'records_per_second':speed,
                       'remaining_seconds_estimate':(sum(e['records'] for e in fit)*(cfg.max_epochs-epoch)-sums[-1])/speed,
                       'gpu':gpu_status() if cuda else [],'checkpoint':last_entry,
                       'micro_iterations':iterations,'rank_valid_records':window}
                    target = output/'logs/steps.jsonl'; target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                    print(f"epoch={epoch+1} step={step} spatial={row['spatial_loss']:.6f} pointer={row['pointer_loss']:.6f} records/s={speed:.2f}",flush=True)
            require(next(iterator,None) is None,'RANK_UNCONSUMED_RECORDS')
            require(len({tuple(x) for x in seen}) == len(seen),'RANK_DUPLICATE_RECORD')
            rank_evidence = [None]*world; dist.all_gather_object(rank_evidence,seen)
            all_refs = [tuple(x) for items in rank_evidence for x in items]
            require(len(set(all_refs)) == len(all_refs) == sum(e['records'] for e in fit), 'DDP_DISJOINT_COVERAGE')
            dev_metrics = evaluate(adapter,dev,world,rank,cfg.microbatch,scales,cfg,device,deadline)
            history.append({'epoch':epoch+1,'optimizer_step':step,'fit':{k:sums[i]/sums[-1] for i,k in enumerate(
                ('spatial_mse','pointer_mse','normalized_spatial','normalized_pointer','loss'))},
                'dev':dev_metrics,'records':int(sums[-1]),'epoch_seconds':time.perf_counter()-epoch_time})
            last_entry = checkpoint(epoch+1)
            if rank == 0:
                write_json(output/'metrics/history.json',history)
                write_json(output/f'assignments/epoch-{epoch+1:05d}-coverage.json',{'status':'PASS','rank_records':[len(x) for x in rank_evidence],
                    'unique_records':len(set(all_refs)),'expected_records':len(all_refs)})
                if dev_metrics['records'] and dev_metrics['loss'] == min(h['dev']['loss'] for h in history):
                    write_json(output/'best_state_loss.ckpt.json',last_entry)
            if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
                if rank == 0:
                    write_json(output/'STATUS.json',{'status':'STOPPED','reason':'requested_epoch_boundary',
                        'complete_epochs':epoch+1,'checkpoint':last_entry,'research_gates_passed':False})
                dist.barrier()
                return {'status':'STOPPED','checkpoint':last_entry}
        require(last_entry is not None,'NO_COMPLETED_EPOCH')
        # Strict model/optimizer/scheduler reload on every rank, plus actual output equality.
        from .lvos_pilot import equal_state
        payload,_ = load_complete(output/last_entry['path'])
        reloaded = validate_model_lock(payload['model_lock'],require_approval=False).to(device)
        reloaded.load_state_dict(payload['export']['state_dict'],strict=True)
        opt2 = torch.optim.AdamW(optimizer_groups(reloaded,cfg.weight_decay)[0],lr=cfg.lr)
        sch2 = make_scheduler(opt2,cfg,planned_updates)
        opt2.load_state_dict(payload['optimizer']); sch2.load_state_dict(payload['scheduler'])
        require(equal_state(optimizer.state_dict(),opt2.state_dict()) and
                equal_state(scheduler.state_dict(),sch2.state_dict()),'OPTIMIZER_SCHEDULER_RELOAD')
        sample = collate_records([next(iter(records(fit[:1])))]).to(device).tensors
        model.eval(); reloaded.eval()
        with torch.no_grad():
            before = adapter(sample['source_spatial'],sample['source_pointer'])
            after = TensorModule(reloaded)(sample['source_spatial'],sample['source_pointer'])
        require(all(torch.equal(a,b) for a,b in zip(before,after)), 'STRICT_RELOAD_OUTPUT')
        changed = {part:any(not torch.equal(start_weights[part][n],p.detach().cpu()) for n,p in model.named_parameters()
                    if n.startswith(part+'.')) for part in ('spatial','pointer')}
        difference = same_across_ranks(model,device)
        if mode == 'smoke': require(all(changed.values()),'SMOKE_BOTH_BRANCH_UPDATES')
        report = {'status':('REAL_DDP_SMOKE_PASS' if mode=='smoke' else 'REAL_DDP_TRAIN_COMPLETED') if cuda else 'CPU_DDP_SYNTHETIC_OR_DATA_PASS',
                  'execution_kind':'REAL_CACHE_GPU' if cuda else 'CPU_SYNTHETIC' if index.get('synthetic_declared') else 'REAL_CACHE_CPU', 'world_size':world,
                  'training_contract':contract,'both_branches_updated':all(changed.values()),'branch_updates':changed,
                  'strict_reload_equal':True,'optimizer_reload_equal':True,'scheduler_reload_equal':True,
                  'rank_parameter_max_difference':difference,'checkpoint':last_entry,'complete_epochs':cfg.max_epochs,
                  'optimizer_step':step,'duration_seconds':time.perf_counter()-started,
                  'historical_provenance':'UNKNOWN','research_gates_passed':False,'JF_early_stopping_applied':False,
                  'vos_best_model_selected':False,'resume':'complete epoch only','normalization_scope':index['normalization']['scope']}
        report['development_scope'] = 'diagnostic_first_case' if mode == 'smoke' else 'full_development_state_loss'
        if rank == 0: write_json(output/'STATUS.json',report)
        dist.barrier()
        return report


def parser():
    p = argparse.ArgumentParser(description='LVOS 기존 cache CPU 호환 검사 + 단일 translator DDP (J&F 독립)')
    commands = p.add_subparsers(dest='command',required=True)
    item = commands.add_parser('index',help='원본 read-only CPU 검사, 재개 가능한 derived raw_index.json')
    for name in ('fit-root','development-root','rgb-root','output'): item.add_argument('--'+name,required=True,type=Path)
    item.add_argument('--operator-completed',required=True,help='실제 팀 완료 확인 문구/날짜; 과거 생성 증명 아님')
    item.add_argument('--case-id',action='append'); item.add_argument('--stable-seconds',type=float,default=60)
    item.add_argument('--prompt-decision',type=Path,help='특정 case의 관측 prompt 조건을 state 학습에만 수용하는 실제 운영 승인; 기본 거부')
    item.add_argument('--exclusion-plan',type=Path,help='frozen membership을 보존한 run별 fit case 제외안; 미승인안은 본 학습 불가')
    item.add_argument('--max-wall-seconds',required=True,type=float)
    item = commands.add_parser('train',help='torchrun same-host 2/4 rank; 기본 smoke, main train에는 실제 DDP smoke 필요')
    item.add_argument('--index',required=True,type=Path); item.add_argument('--output',required=True,type=Path)
    item.add_argument('--config',type=Path); item.add_argument('--global-batch',type=int,default=64)
    item.add_argument('--mode',choices=('smoke','train'),default='smoke'); item.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    item.add_argument('--resume',action='store_true'); item.add_argument('--smoke-evidence',type=Path)
    item.add_argument('--max-wall-seconds',required=True,type=float); item.add_argument('--execute-approved',action='store_true')
    item.add_argument('--gpu-uuid',action='append'); item.add_argument('--pod-hourly-rate',type=float); item.add_argument('--budget-usd',type=float)
    item.add_argument('--lease-root',type=Path); item.add_argument('--host-id')
    item.add_argument('--stop-after-epoch',type=int,help='완성 epoch 경계에서 저장 후 중단; --resume 검사/운영용')
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == 'index':
            build_index(args.fit_root,args.development_root,args.rgb_root,args.output,operator_completed=args.operator_completed,
                        case_ids=args.case_id,stable_seconds=args.stable_seconds,max_wall_seconds=args.max_wall_seconds,
                        prompt_decision=json_read(args.prompt_decision) if args.prompt_decision else None,
                        exclusion_plan=json_read(args.exclusion_plan) if args.exclusion_plan else None)
        else:
            world = int(os.environ.get('WORLD_SIZE','1'))
            cfg = LVOSConfig(**json_read(args.config)) if args.config else LVOSConfig(workers=0,
                accumulation=args.global_batch//(world*16),max_epochs=3 if args.mode=='smoke' else 30)
            train(args.index,args.output,config=cfg,global_batch=args.global_batch,mode=args.mode,device=args.device,
                  resume=args.resume,max_wall_seconds=args.max_wall_seconds,smoke_evidence=args.smoke_evidence,
                  gpu_uuids=args.gpu_uuid,pod_hourly_rate=args.pod_hourly_rate,budget=args.budget_usd,execute_approved=args.execute_approved,
                  lease_root=args.lease_root,host_id=args.host_id,stop_after_epoch=args.stop_after_epoch)
        return 0
    except BudgetStop:
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
