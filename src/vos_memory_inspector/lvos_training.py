"""고정 spatial Transformer의 whole-record 학습·원자적 epoch resume."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import math
import time
import random
from collections import Counter
from contextlib import nullcontext
import torch

from .training_storage import content_hash, write_json, ExclusiveWriter
from .training_runner import rng_state, restore_rng, code_provenance
from .lvos_contract import validate_model_lock, save_complete, load_complete, Events, resources, json_read
from .lvos_snapshot import verify_snapshot, record_loader, RecordStream
from .collection_contract import require
from .lvos_budget import Deadline, BudgetStop, check
from .lvos_checkpoint import CheckpointStore


@dataclass
class LVOSConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_spatial: float = 1.0
    lambda_pointer: float = 1.0
    lambda_cos: float = 0.0
    microbatch: int = 16
    accumulation: int = 4
    max_epochs: int = 30
    warmup_fraction: float = 0.05
    min_lr: float = 1e-5
    clip_norm: float = 1.0
    seed: int = 7
    workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    precision: str = 'fp32'
    augmentation: str = 'none'
    max_wall_seconds: float | None = None
    max_micro_iterations: int | None = None
    monitor_every: int = 2
    min_epochs: int = 6
    patience: int = 5
    min_delta: float = 0.001
    max_outstanding_evaluations: int = 2
    quoted_hourly_rate: float | None = None


def component_objective(pred_spatial, pred_pointer, target_spatial, target_pointer, validity, scales, config):
    """NaN padding을 산술 전에 gather. Component마다 record mean을 독립 계산한다."""
    require(validity.dtype==torch.bool and validity.shape==pred_spatial.shape[:-3], 'LOSS_RECORD_ALIGNMENT')
    require(pred_spatial.shape==target_spatial.shape and pred_pointer.shape==target_pointer.shape and
            pred_pointer.shape[:-1]==validity.shape, 'LOSS_PAIR_ALIGNMENT')
    require(bool(validity.any()), 'NO_VALID_RECORDS')
    spatial = (pred_spatial[validity].float()-target_spatial[validity].float()).square().flatten(1).mean(1)
    pointer = (pred_pointer[validity].float()-target_pointer[validity].float()).square().mean(1)
    require(bool(torch.isfinite(spatial).all() and torch.isfinite(pointer).all()), 'NONFINITE_LOSS')
    raw = {'spatial':spatial.mean(), 'pointer':pointer.mean()}
    normalized = {k:v/max(float(scales[k])**2,1e-12) for k,v in raw.items()}
    total = config.lambda_spatial*normalized['spatial']+config.lambda_pointer*normalized['pointer']
    return raw, normalized, total


def fit_rms(root, config, deadline=None):
    totals = {'spatial':[0.,0],'pointer':[0.,0]}
    for batch in record_loader(root,'fit',workers=0,microbatch=config.microbatch,pin_memory=False,shuffle=False,deadline=deadline):
        check(deadline,'fit_rms_batch')
        for name in totals:
            value = batch.tensors['target_'+name].double()
            totals[name][0] += float(value.square().sum()); totals[name][1] += value.numel()
    require(all(n for _,n in totals.values()), 'EMPTY_FIT_STATS')
    check(deadline,'fit_rms_complete')
    return {k:math.sqrt(total/count) for k,(total,count) in totals.items()}


def optimizer_groups(model, weight_decay):
    norm_ids = {id(p) for m in model.modules() if isinstance(m,torch.nn.LayerNorm) for p in m.parameters(recurse=False)}
    decay, excluded, names = [], [], {'decay':[], 'no_decay':[]}
    for name,p in model.named_parameters():
        no_decay = name.endswith('.bias') or id(p) in norm_ids or name.endswith('.alpha') or 'pos_embed' in name
        (excluded if no_decay else decay).append(p)
        names['no_decay' if no_decay else 'decay'].append(name)
    return [{'params':decay,'weight_decay':weight_decay}, {'params':excluded,'weight_decay':0.}], names


def make_scheduler(optimizer, config, planned_updates):
    warmup = max(1,math.ceil(planned_updates*config.warmup_fraction))
    def multiplier(step):
        if step<warmup:
            return (step+1)/warmup
        progress = min(1.,(step-warmup)/max(1,planned_updates-warmup))
        floor = config.min_lr/config.lr
        return floor+(1-floor)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)


def forward_batch(model, batch, device, scales, config, timing=None, synchronize=False):
    if synchronize and torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)
    started=time.perf_counter()
    batch = batch.to(device)
    if synchronize and torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)
    transferred=time.perf_counter()
    t = batch.tensors
    spatial, pointer = model.translate_tensors(t['source_spatial'][None,None].float(), t['source_pointer'][None,None].float())
    raw, normalized, loss = component_objective(spatial[0,0],pointer[0,0],t['target_spatial'],t['target_pointer'],
                                               torch.ones(len(batch),dtype=torch.bool,device=device), scales, config)
    if synchronize and torch.device(device).type=='cuda':
        torch.cuda.synchronize(device)
    if timing is not None:
        timing['h2d_seconds']=transferred-started
        timing['forward_loss_seconds']=time.perf_counter()-transferred
    return raw, normalized, loss


def state_dev(model, root, device, scales, config, deadline=None):
    model.eval(); totals = {'spatial_mse':0.,'pointer_mse':0.,'loss':0.}; count = 0
    with torch.no_grad():
        for batch in record_loader(root,'development',workers=0,microbatch=config.microbatch,pin_memory=False,shuffle=False,deadline=deadline):
            check(deadline,'state_dev_batch')
            raw,_,loss = forward_batch(model,batch,device,scales,config)
            for key,value in (('spatial_mse',raw['spatial']),('pointer_mse',raw['pointer']),('loss',loss)):
                totals[key] += float(value)*len(batch)
            count += len(batch)
    check(deadline,'state_dev_complete')
    return {k:v/count for k,v in totals.items()}|{'records':count} if count else {'records':0,'loss':None,'reason':'NO_DEVELOPMENT_RECORDS'}


def checkpoint_payload(model, optimizer, scheduler, identity, lock, scales, epoch, step, history, early_state):
    return {'schema_version':'cmmt.lvos_checkpoint.v1','identity':identity,'model_lock':lock,'export':model.to_payload(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),'scales':scales,
            'epoch':epoch,'optimizer_step':step,'history':history,'early_state':early_state,'boundary':'complete_epoch'}


def load_model_checkpoint(path, device='cpu'):
    payload, entry = load_complete(path)
    require(payload['schema_version']=='cmmt.lvos_checkpoint.v1','CHECKPOINT_SCHEMA')
    model = validate_model_lock(payload['model_lock'])
    model.load_state_dict(payload['export']['state_dict'],strict=True)
    return model.to(device).eval(), payload, entry


def train_snapshot(root, lock, output, config: LVOSConfig, *, deadline=None, **kwargs):
    deadline = deadline or Deadline(config.max_wall_seconds)
    output = Path(output)
    with ExclusiveWriter(output):
        try:
            return _train_snapshot(root,lock,output,config,deadline=deadline,**kwargs)
        except BudgetStop as exc:
            last = json_read(output/'last.ckpt.json') if (output/'last.ckpt.json').exists() else None
            value = {'status':'STOPPED','reason':'budget_stop','incomplete_stage':exc.stage,
                     'training_status':'STOPPED','evaluation_status':'PENDING_OR_NOT_REQUESTED',
                     'last_complete_checkpoint':last,'duration_seconds':deadline.elapsed(),
                     'partial_dev_is_complete':False}
            write_json(output/'STOPPED.json',value); write_json(output/'STATUS.json',value)
            return []


def _train_snapshot(root, lock, output, config: LVOSConfig, *, device='cuda:0', mode='train', resume=None,
                   gates=None, monitor_protocol=None, metric_contract=None, deadline=None, evaluation_callback=None):
    require(config.lr>0 and config.microbatch>0 and config.accumulation>0 and config.max_epochs>0 and config.workers>=0,
            'TRAIN_CONFIG')
    require(config.precision=='fp32' and config.augmentation=='none' and config.lambda_cos==0, 'V1_PRECISION_OR_AUGMENTATION')
    require(config.max_wall_seconds is None or config.max_wall_seconds>0,'TIME_LIMIT')
    require(config.max_micro_iterations is None or config.max_micro_iterations>0,'MICRO_LIMIT')
    require(config.monitor_every>0 and config.max_outstanding_evaluations>0 and config.min_lr<=config.lr and
            config.clip_norm>0 and 0<=config.warmup_fraction<1 and config.lambda_spatial>=0 and config.lambda_pointer>=0,'TRAIN_CONFIG')
    require(not monitor_protocol or metric_contract,'METRIC_CONTRACT_REQUIRED')
    snapshot = verify_snapshot(root,hashes=True,deadline=deadline)
    if mode=='train' and not snapshot['synthetic']:
        from .lvos_gates import authorize_training
        authorize_training(gates,snapshot['content_sha256'],lock['digest'])
        require(snapshot['state']=='ready' and monitor_protocol and metric_contract, 'MAIN_TRAIN_INPUTS')
        from .lvos_evaluation import validate_protocol_snapshot
        validate_protocol_snapshot(snapshot,json_read(monitor_protocol))
    if mode=='overfit':
        require(len({c['case']['video_id'] for c in snapshot['cases'] if c['split']=='fit'})==1, 'OVERFIT_ONE_VIDEO')
    if torch.device(device).type=='cuda':
        require(torch.cuda.device_count()==1 and bool(__import__('os').environ.get('CUDA_VISIBLE_DEVICES')), 'ONE_VISIBLE_GPU')
    kind = 'cpu_synthetic' if snapshot['synthetic'] else ('gpu_real_checkpoint' if torch.device(device).type=='cuda' else 'cpu_real_cache')
    random.seed(config.seed); torch.manual_seed(config.seed)
    __import__('numpy').random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = validate_model_lock(lock).to(device)
    groups,names = optimizer_groups(model,config.weight_decay)
    optimizer = torch.optim.AdamW(groups,lr=config.lr)
    planned = sum(math.ceil(RecordStream(root,'fit',epoch,config.seed,deadline=deadline).batch_count(config.microbatch,config.workers)/config.accumulation)
                  for epoch in range(config.max_epochs))
    require(planned>0,'EMPTY_FIT')
    scheduler = make_scheduler(optimizer,config,planned)
    identity = {'run_id':Path(output).name,'code_sha':code_provenance()['package_source_sha256'],
        'code_revision':code_provenance()['commit'],
        'model_config_digest':lock['digest'],'collection_digest':snapshot['content_sha256'],
        'config':asdict(config),'mode':mode,'planned_updates':planned,
        'monitor_protocol_digest':content_hash(json_read(monitor_protocol)) if monitor_protocol else None,
        'metric_contract_digest':content_hash(json_read(metric_contract)) if metric_contract else None}
    output = Path(output)
    with nullcontext():  # public wrapper owns the exclusive run lock, including preparation.
        require(resume is not None or mode=='initialize' or not (output/'config.json').exists(),'RUN_EXISTS')
        events = Events(output,identity,kind)
        started = deadline.started; last_heartbeat = time.perf_counter()
        start_epoch,step,micro,history = 0,0,0,[]
        from .lvos_evaluation import EarlyStopping
        early = EarlyStopping(config.min_epochs,config.patience,config.min_delta)
        store = CheckpointStore(output,identity)
        check(deadline,'checkpoint_reconciliation')
        recovered = store.reconcile() if resume or mode=='initialize' else None
        if resume or recovered:
            require(recovered is not None,'RESUME_RUN_MISSING')
            payload,_ = recovered
            require(mode!='initialize' or payload['epoch']==0,'INITIALIZE_EXISTING_TRAINED_RUN')
            if resume and str(resume)!='latest':
                requested,_ = load_complete(resume)
                require(requested['identity']==identity and requested['epoch']<=payload['epoch'],'RESUME_IDENTITY')
            model.load_state_dict(payload['export']['state_dict'],strict=True)
            optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
            restore_rng(payload['rng'])
            scales,history,start_epoch,step = payload['scales'],payload['history'],payload['epoch'],payload['optimizer_step']
            early.restore(payload['early_state'])
            require(start_epoch<=config.max_epochs,'RESUME_EPOCH_LIMIT')
        else:
            scales = fit_rms(root,config,deadline=deadline)
        check(deadline,'training_setup_complete')
        write_json(output/'config.json', {'identity':identity,'model_lock':lock,'scales':scales,'parameter_groups':names,
            'group_parameter_counts':[sum(p.numel() for p in g['params']) for g in groups],'augmentation':'none'})
        write_json(output/'provenance.json',resources())
        if history:
            # A completed body may precede metric/pointer side effects at the crash boundary.
            write_json(output/'metrics/train.json',history)
            best_row=min(history,key=lambda r:r['dev'].get('loss') if r['dev'].get('loss') is not None else r['fit']['loss'])
            _,best_entry=load_complete(output/f"checkpoints/epoch-{best_row['epoch']:05d}.ckpt")
            write_json(output/'best_state_loss.ckpt.json',{k:('checkpoints/'+best_entry[k] if k=='path' else best_entry[k])
                                                         for k in ('path','sha256','bytes')})
        if recovered is None:
            init_payload = checkpoint_payload(model,optimizer,scheduler,identity,lock,scales,0,0,[],early.state())
            init_entry=store.publish(init_payload)
            if monitor_protocol:
                from .lvos_jobs import publish_request
                publish_request(output,output/'checkpoints/epoch-00000.ckpt',monitor_protocol,metric_contract)
        events.emit('run_start',metrics={'resources':resources(),'planned_updates':planned,'scales':scales},evidence_paths=[str(output/'config.json')])
        write_json(output/'STATUS.json',{'status':'RUNNING','execution_kind':kind,'resume_epoch':start_epoch})
        if mode=='initialize':
            init_entry = json_read(output/'last.ckpt.json')
            write_json(output/'INITIALIZED.json',{'epoch':0,'untrained':True,'execution_kind':kind,'checkpoint':'checkpoints/epoch-00000.ckpt'})
            write_json(output/'STATUS.json',{'status':'INITIALIZED','execution_kind':kind,'complete_epochs':0})
            events.emit('initialization_complete',epoch=0,checkpoint_sha=init_entry['sha256'],evidence_paths=[str(output/'INITIALIZED.json')])
            return []
        best_state = min((r['dev'].get('loss') if r['dev'].get('loss') is not None else r['fit']['loss'] for r in history),default=math.inf)
        reason = 'normal_completion'
        loader = record_loader(root,'fit',seed=config.seed,microbatch=config.microbatch,workers=config.workers,
                    prefetch_factor=config.prefetch_factor,pin_memory=config.pin_memory,deadline=deadline)
        if torch.device(device).type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        try:
            for epoch in range(start_epoch,config.max_epochs):
                model.train(); optimizer.zero_grad(set_to_none=True)
                epoch_started = time.perf_counter(); window_count=window_micro=count=0
                sums={'spatial_mse':0.,'pointer_mse':0.,'normalized_spatial':0.,'normalized_pointer':0.,'loss':0.}
                case_counts=Counter(); seen=set()
                timings=Counter()
                loader.dataset.set_epoch(epoch)
                expected_batches=loader.dataset.batch_count(config.microbatch,config.workers)
                events.emit('epoch_start',epoch=epoch+1)
                previous=time.perf_counter()
                for batch_index,batch in enumerate(loader):
                    check(deadline,'training_batch')
                    now=time.perf_counter(); data_wait=now-previous
                    if config.max_micro_iterations and micro>=config.max_micro_iterations:
                        reason='profile_complete' if mode=='profile' else 'budget_stop'; break
                    timing={}
                    raw,norm,loss=forward_batch(model,batch,device,scales,config,timing,synchronize=mode=='profile')
                    compute_started=time.perf_counter()
                    (loss*len(batch)).backward()
                    if mode=='profile' and torch.device(device).type=='cuda':
                        torch.cuda.synchronize(device)
                    timing['backward_seconds']=time.perf_counter()-compute_started
                    timings['data_wait_seconds']+=data_wait
                    for name,value in timing.items():
                        timings[name]+=value
                    window_count+=len(batch); window_micro+=1; count+=len(batch); micro+=1
                    for ref in batch.refs:
                        key=tuple(ref); require(key not in seen,'DUPLICATE_EPOCH_RECORD'); seen.add(key); case_counts[ref[0]]+=1
                    for key,value in (('spatial_mse',raw['spatial']),('pointer_mse',raw['pointer']),
                        ('normalized_spatial',norm['spatial']),('normalized_pointer',norm['pointer']),('loss',loss)):
                        sums[key]+=float(value.detach())*len(batch)
                    if window_micro==config.accumulation or batch_index+1==expected_batches:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.div_(window_count)
                                require(bool(torch.isfinite(p.grad).all()),'NONFINITE_GRADIENT')
                        gradients={part:math.sqrt(sum(float(p.grad.float().square().sum()) for name,p in model.named_parameters()
                                    if name.startswith(part+'.') and p.grad is not None)) for part in ('spatial','pointer')}
                        clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip_norm))
                        lr=optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step(); step+=1
                        optimizer.zero_grad(set_to_none=True); window_count=window_micro=0
                        if step<=3 or step%20==0 or time.perf_counter()-last_heartbeat>=60 or batch_index+1==expected_batches:
                            elapsed=time.perf_counter()-started
                            events.emit('optimizer_update',epoch=epoch+1,micro_iteration=micro,optimizer_step=step,
                                metrics={**{k:v/count for k,v in sums.items()},'lr':lr,'gradient_norms':gradients,
                                    'clip_input_norm':clip,'alpha_norm':float(model.spatial.alpha.detach().norm()),
                                    'weighted_spatial_loss':config.lambda_spatial*sums['normalized_spatial']/count,
                                    'weighted_pointer_loss':config.lambda_pointer*sums['normalized_pointer']/count,
                                    'skipped_updates':0,'records':count,'padded_records':0,'processed_cases':len(case_counts),
                                    'processed_videos':len({snapshot['cases'][i]['case']['video_id'] for i in case_counts}),
                                    'record_shapes':{k:list(v.shape) for k,v in batch.tensors.items()},
                                    'record_dtypes':{k:str(v.dtype) for k,v in batch.tensors.items()},'device':device,
                                    'records_per_second':count/(time.perf_counter()-epoch_started),
                                    'optimizer_updates_per_second':step/elapsed,
                                    'epoch_timing_sums':dict(timings),'last_micro_timing':timing,
                                    'cpu_process_rss_bytes':resources()['cpu_process_rss_bytes'],
                                    'estimated_remaining_fit_seconds':(len(loader.dataset)-count+(config.max_epochs-epoch-1)*len(loader.dataset))
                                          /(count/(time.perf_counter()-epoch_started)),
                                    'cuda_allocated_peak':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
                                    'cuda_reserved_peak':torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None,
                                    'timing_kind':'measured_synchronized_profile' if mode=='profile' else 'measured_host_wall_no_cuda_sync',
                                    'quoted_cost':elapsed/3600*config.quoted_hourly_rate if config.quoted_hourly_rate is not None else None})
                            print(f'epoch {epoch+1}, step {step}: 유효 record={count}, loss={sums["loss"]/count:.6f}',flush=True)
                            last_heartbeat=time.perf_counter()
                    previous=time.perf_counter()
                complete=count==len(loader.dataset)
                events.emit('epoch_end',epoch=epoch+1,optimizer_step=step,metrics={'records':count,'expected_records':len(loader.dataset),
                    'coverage_complete':complete,'case_record_counts':dict(case_counts),'epoch_timing_sums':dict(timings)},duration_seconds=time.perf_counter()-epoch_started)
                if not complete:
                    require(reason!='normal_completion','EPOCH_COVERAGE_INCOMPLETE')
                    break  # partial epoch는 checkpoint에 완성 epoch로 저장하지 않는다.
                check(deadline,'state_dev_start')
                dev=state_dev(model,root,device,scales,config,deadline=deadline) if mode=='train' else {'loss':None,'reason':'FIT_ONLY_DIAGNOSTIC'}
                check(deadline,'state_dev_complete')
                row={'epoch':epoch+1,'fit':{k:v/count for k,v in sums.items()},'dev':dev,'records':count,'optimizer_step':step}
                history.append(row)
                payload=checkpoint_payload(model,optimizer,scheduler,identity,lock,scales,epoch+1,step,history,early.state())
                checkpoint_started=time.perf_counter()
                entry=store.publish(payload)
                events.emit('checkpoint_complete',epoch=epoch+1,optimizer_step=step,checkpoint_sha=entry['sha256'],
                    metrics={'checkpoint_seconds':time.perf_counter()-checkpoint_started},evidence_paths=[str(output/entry['path'])])
                state_score=dev.get('loss') if dev.get('loss') is not None else row['fit']['loss']
                if state_score<best_state:
                    best_state=state_score; write_json(output/'best_state_loss.ckpt.json',entry)
                write_json(output/'metrics/train.json',history)
                with (output/'metrics/train.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=['epoch','fit_loss','dev_loss','records','optimizer_step']); writer.writeheader()
                    writer.writerows({'epoch':r['epoch'],'fit_loss':r['fit']['loss'],'dev_loss':r['dev']['loss'],
                                      'records':r['records'],'optimizer_step':r['optimizer_step']} for r in history)
                write_json(output/'metrics/case_records.json',{'epoch':epoch+1,'rows':[{'case_id':snapshot['cases'][i]['case']['case_id'],
                            'video_id':snapshot['cases'][i]['case']['video_id'],'records':n} for i,n in sorted(case_counts.items())]})
                if monitor_protocol and (epoch+1)%config.monitor_every==0:
                    from .lvos_jobs import publish_request
                    publish_request(output,output/entry['path'],monitor_protocol,metric_contract)
                    if evaluation_callback:
                        saved_rng=rng_state()
                        try:
                            check(deadline,'sequential_evaluation_start')
                            evaluation_callback(output,deadline)
                            check(deadline,'sequential_evaluation_complete')
                        finally:
                            restore_rng(saved_rng)
                    from .lvos_evaluation import consume_monitor_results
                    outstanding=consume_monitor_results(output,early,identity,events)
                    while outstanding>=config.max_outstanding_evaluations:
                        check(deadline,'evaluation_wait')
                        require(config.max_wall_seconds is not None,'ASYNC_WAIT_REQUIRES_TIME_LIMIT')
                        time.sleep(1)
                        outstanding=consume_monitor_results(output,early,identity,events)
                        if time.perf_counter()-last_heartbeat>=60:
                            events.emit('evaluation_wait',epoch=epoch+1,metrics={'outstanding':outstanding})
                            last_heartbeat=time.perf_counter()
                    write_json(output/'early_stopping.json',early.state())
                    if early.should_stop:
                        reason='early_stopping'
                    if reason!='normal_completion':
                        break
                if config.max_micro_iterations and micro>=config.max_micro_iterations and mode=='profile':
                    reason='profile_complete'; break
            check(deadline,'finalization')
            pending=[]
            if monitor_protocol:
                from .lvos_jobs import pending_requests
                from .lvos_evaluation import consume_monitor_results
                consume_monitor_results(output,early,identity,events)
                pending=pending_requests(output)
            marker='COMPLETED.json' if reason=='normal_completion' else 'STOPPED.json'
            write_json(output/marker,{'reason':reason,'complete_epochs':len(history),'optimizer_step':step,
                'partial_epoch_restart':'last complete epoch; partial optimizer updates are discarded on resume',
                'duration_seconds':time.perf_counter()-started,'execution_kind':kind})
            write_json(output/'STATUS.json',{'status':'TRAINING_COMPLETED_EVALUATION_PENDING' if reason=='normal_completion' and pending else marker.removesuffix('.json'),
                'training_status':'COMPLETED' if reason=='normal_completion' else 'STOPPED',
                'evaluation_status':'PENDING' if pending else 'COMPLETED' if monitor_protocol else 'NOT_REQUESTED',
                'pending_requests':pending,'reason':reason,'complete_epochs':len(history),'execution_kind':kind})
            events.emit('run_end',optimizer_step=step,metrics={'reason':reason,'complete_epochs':len(history)},evidence_paths=[str(output/marker)])
            return history
        except BudgetStop:
            raise
        except Exception as exc:
            write_json(output/'FAILED.json',{'reason_code':getattr(exc,'code','TRAIN_FAILURE'),'detail':str(exc),'optimizer_step':step})
            write_json(output/'STATUS.json',{'status':'FAILED','training_status':'FAILED','evaluation_status':'PENDING_OR_NOT_REQUESTED',
                                            'reason_code':getattr(exc,'code','TRAIN_FAILURE'),'execution_kind':kind})
            events.emit('run_failure',status='FAIL',reason_code=getattr(exc,'code','TRAIN_FAILURE'),evidence_paths=[str(output/'FAILED.json')])
            raise
