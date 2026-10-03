"""One-case state-supervised diagnostic. Never authorizes official/full training."""
from pathlib import Path
import math
import random
import time
import numpy as np
import torch
from .collection_contract import require, validate_generating
from .cache_migration import _check_frozen
from .lvos_budget import Deadline, BudgetStop, check
from .lvos_contract import validate_model_lock, save_complete, load_complete, Events
from .lvos_snapshot import verify_snapshot, record_loader
from .lvos_training import LVOSConfig, fit_rms, forward_batch, optimizer_groups, make_scheduler
from .training_storage import write_json, content_hash, sha256, ExclusiveWriter
from .training_runner import code_provenance, rng_state, restore_rng
from .transformer_translator import SAM21_MEMORY_SPEC


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def validated_input(root, deadline, *, synthetic_test=False):
    snapshot = verify_snapshot(root, deadline=deadline)
    require(not snapshot['synthetic'] or synthetic_test, 'PILOT_REQUIRES_REAL_INPUT')
    require(snapshot['state'] == 'pilot_ready' and len(snapshot['cases']) == 1 and
            len(snapshot['expected_case_ids']) == 1, 'PILOT_ONE_CASE_ONLY')
    case = snapshot['cases'][0]
    require(case['split'] == 'fit' and all(s['split'] == 'fit' for s in snapshot['shards']), 'PILOT_FIT_ONLY')
    require(snapshot['expected_case_ids'] == [case['case']['case_id']], 'PILOT_CASE_ID')
    validate_generating(case['generating']); _check_frozen(case['case'], snapshot['selection'])
    require(bool(case['generating'].get('synthetic', False)) == snapshot['synthetic'], 'PILOT_SYNTHETIC_MISMATCH')
    seen = set(); first = None
    for batch in record_loader(root, 'fit', workers=0, microbatch=1, pin_memory=False, shuffle=False, deadline=deadline):
        check(deadline, 'pilot_input_record')
        ref = tuple(batch.refs[0]); require(ref not in seen, 'PILOT_DUPLICATE_RECORD'); seen.add(ref)
        index, b, o, r = ref
        require(index == 0 and case['original_validity'][b][o][r], 'PILOT_INVALID_REFERENCE')
        tensors = batch.tensors
        for name, original in [('frame','original_frames'), ('slot','original_slots'), ('conditioning','original_conditioning')]:
            require(tensors[name].item() == case[original][b][o][r], 'PILOT_RECORD_ALIGNMENT', name)
        spec = SAM21_MEMORY_SPEC
        for role in ('source', 'target'):
            spatial, pointer = tensors[role+'_spatial'], tensors[role+'_pointer']
            require(spatial.shape == (1,spec.feature_channels,spec.height,spec.width) and
                    pointer.shape == (1,spec.pointer_dim), 'PILOT_TENSOR_SHAPE')
            require(spatial.dtype == torch.bfloat16 and pointer.dtype == torch.float32, 'PILOT_TENSOR_DTYPE')
            require(bool(torch.isfinite(spatial).all() and torch.isfinite(pointer).all()), 'PILOT_NONFINITE_INPUT')
        if first is None:
            first = batch
    require(len(seen) == snapshot['statistics']['fit']['valid_records'] and first is not None, 'PILOT_RECORD_COVERAGE')
    return snapshot, first, len(seen)


def pilot_preflight(root, lock, output, *, device='cuda:0', max_wall_seconds=900,
                    deadline=None, synthetic_test=False):
    deadline = deadline or Deadline(max_wall_seconds)
    output = Path(output)
    with ExclusiveWriter(output):
        require(not (output/'STATUS.json').exists(), 'PILOT_OUTPUT_EXISTS')
        try:
            return _pilot(root, lock, output, device, deadline, synthetic_test)
        except Exception as exc:
            stopped = isinstance(exc, BudgetStop)
            write_json(output/'STATUS.json', {'schema_version':'cmmt.lvos_pilot.v1', 'scope':'diagnostic_only',
                'status':'STOPPED' if stopped else 'FAIL', 'reason_code':'BUDGET_STOP' if stopped else getattr(exc,'code','PILOT_FAILURE'),
                'detail':str(exc), 'unattended_training_ready':False, 'state_training_ready':False})
            raise


def _pilot(root, lock, output, device, deadline, synthetic_test):
    require(not synthetic_test or torch.device(device).type == 'cpu', 'SYNTHETIC_TEST_CPU_ONLY')
    check(deadline, 'pilot_start')
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == 'cuda':
        require(torch.cuda.device_count() == 1, 'ONE_VISIBLE_GPU')
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    snapshot, batch, records = validated_input(root, deadline, synthetic_test=synthetic_test)
    kind = 'cpu_synthetic' if snapshot['synthetic'] else ('gpu_real_checkpoint' if torch.device(device).type=='cuda' else 'cpu_real_cache')
    cfg = LVOSConfig(microbatch=1, accumulation=1, workers=0, pin_memory=False, max_epochs=1, seed=7)
    identity = {'run_id':output.name, 'code_sha':code_provenance()['package_source_sha256'],
        'model_config_digest':lock['digest'], 'collection_digest':snapshot['content_sha256'],
        'scope':'diagnostic_only', 'case_id':snapshot['expected_case_ids'][0]}
    events = Events(output, identity, kind)
    events.emit('pilot_input', metrics={'load_seconds':time.perf_counter()-load_started, 'records':records,
        'padded_records':snapshot['statistics']['fit']['padded_records'], 'case':snapshot['cases'][0]['case'],
        'input_sha256':sha256(Path(root)/'snapshot.json'), 'views':snapshot['shards'],
        'record_alignment':'PASS', 'scope':'diagnostic_only', 'synthetic':snapshot['synthetic']})
    stats_started = time.perf_counter()
    scales = fit_rms(root, cfg, deadline)
    stats = {'schema_version':'cmmt.lvos_pilot_normalization.v1', 'scope':'pilot_fit_only',
        'records':records, 'case_id':identity['case_id'], 'view_digest':snapshot['content_sha256'],
        'memory_policy':snapshot['memory_policy'], 'scales':scales, 'computations':1}
    write_json(output/'normalization.json', stats)
    events.emit('pilot_normalization', metrics={'seconds':time.perf_counter()-stats_started, **stats})
    # Only this diagnostic path may omit formal approval; source/config pins remain strict.
    model = validate_model_lock(lock, require_approval=False).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model,cfg.weight_decay)[0],lr=cfg.lr)
    scheduler = make_scheduler(optimizer,cfg,4)
    initial = {name:p.detach().clone() for name,p in model.named_parameters()}
    def sync():
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
    def update(m, opt, sched, step):
        check(deadline, 'pilot_update_'+str(step))
        events.emit('pilot_update_start',optimizer_step=step)
        opt.zero_grad(set_to_none=True); timing={}
        raw, norm, loss = forward_batch(m,batch,device,scales,cfg,timing,synchronize=True)
        sync(); started=time.perf_counter(); loss.backward(); sync()
        timing['backward_seconds']=time.perf_counter()-started
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()), 'PILOT_NONFINITE_GRADIENT')
        gradient = {part:math.sqrt(sum(float(p.grad.square().sum()) for n,p in m.named_parameters()
                    if n.startswith(part+'.') and p.grad is not None)) for part in ('spatial','pointer')}
        deep = math.sqrt(sum(float(p.grad.square().sum()) for n,p in m.named_parameters()
                    if n.startswith('spatial.blocks.') and p.grad is not None))
        started=time.perf_counter(); lr=opt.param_groups[0]['lr']; opt.step(); sched.step(); sync()
        timing['update_seconds']=time.perf_counter()-started
        delta = {part:math.sqrt(sum(float((p.detach()-initial[n]).square().sum()) for n,p in m.named_parameters()
                    if n.startswith(part+'.'))) for part in ('spatial','pointer')}
        values = {'step':step, 'raw_spatial_mse':float(raw['spatial'].detach()), 'raw_pointer_mse':float(raw['pointer'].detach()),
            'normalized_spatial':float(norm['spatial'].detach()), 'normalized_pointer':float(norm['pointer'].detach()),
            'total':float(loss.detach()), 'gradient_norms':gradient, 'deep_gradient_norm':deep,
            'delta_norms':delta, 'lr':lr, 'timings':timing, 'valid_records':len(batch), 'device':device,
            'shapes':{k:list(v.shape) for k,v in batch.tensors.items()}, 'dtypes':{k:str(v.dtype) for k,v in batch.tensors.items()}}
        events.emit('pilot_update',optimizer_step=step,metrics=values)
        return values
    steps = [update(model,optimizer,scheduler,i) for i in range(1,4)]
    require(all(value['gradient_norms'][part]>0 for value in steps for part in ('spatial','pointer')) and
            all(steps[-1]['delta_norms'][part]>0 for part in ('spatial','pointer')) and
            steps[-1]['deep_gradient_norm']>0, 'PILOT_BRANCH_OR_DEEP_UPDATE')
    write_json(output/'steps.json', {'scope':'diagnostic_only', 'steps':steps})
    check(deadline, 'pilot_checkpoint')
    events.emit('pilot_checkpoint_start',optimizer_step=3)
    started=time.perf_counter()
    payload={'schema_version':'cmmt.lvos_pilot_checkpoint.v1', 'scope':'diagnostic_only',
        'identity':identity,'model_lock':lock,'export':model.to_payload(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'rng':rng_state(),'normalization':stats,'step':3,
        'config':cfg.__dict__,'boundary':'complete_optimizer_boundary','synthetic':snapshot['synthetic']}
    checkpoint=output/'diagnostic.ckpt'; entry=save_complete(checkpoint,payload)
    events.emit('pilot_checkpoint',optimizer_step=3,checkpoint_sha=entry['sha256'],metrics={'seconds':time.perf_counter()-started})
    loaded,_=load_complete(checkpoint)
    events.emit('pilot_reload_start',optimizer_step=3)
    require(loaded['scope']=='diagnostic_only' and loaded['identity']==identity and loaded['normalization']==stats and
            loaded['step']==3 and loaded['config']==cfg.__dict__, 'PILOT_RELOAD_IDENTITY')
    restored=validate_model_lock(loaded['model_lock'], require_approval=False).to(device)
    restored.load_state_dict(loaded['export']['state_dict'],strict=True)
    restored_optimizer=torch.optim.AdamW(optimizer_groups(restored,cfg.weight_decay)[0],lr=cfg.lr)
    restored_scheduler=make_scheduler(restored_optimizer,cfg,4)
    restored_optimizer.load_state_dict(loaded['optimizer']); restored_scheduler.load_state_dict(loaded['scheduler'])
    require(equal_state(model.state_dict(),restored.state_dict()) and
            equal_state(optimizer.state_dict(),restored_optimizer.state_dict()) and
            equal_state(scheduler.state_dict(),restored_scheduler.state_dict()), 'PILOT_RELOAD_PARITY')
    with torch.no_grad():
        a=model.translate_tensors(batch.tensors['source_spatial'][None,None].to(device).float(),batch.tensors['source_pointer'][None,None].to(device).float())
        b=restored.translate_tensors(batch.tensors['source_spatial'][None,None].to(device).float(),batch.tensors['source_pointer'][None,None].to(device).float())
    output_diffs=[float((x-y).abs().max()) for x,y in zip(a,b)]
    write_json(output/'reload_comparison.json',{'max_absolute_output_difference':output_diffs,'comparison':'torch.equal; zero tolerance'})
    require(equal_state(a,b), 'PILOT_RELOAD_OUTPUT_PARITY')
    restore_rng(loaded['rng']); expected_rng=(random.random(),float(np.random.rand()),torch.rand(5))
    update(model,optimizer,scheduler,4)
    restore_rng(loaded['rng']); actual_rng=(random.random(),float(np.random.rand()),torch.rand(5))
    require(equal_state(expected_rng,actual_rng), 'PILOT_RNG_PARITY')
    update(restored,restored_optimizer,restored_scheduler,4)
    next_diff=max(float((p-restored.state_dict()[n]).abs().max()) for n,p in model.state_dict().items())
    write_json(output/'resume_comparison.json',{'next_step_max_absolute_difference':next_diff,'comparison':'torch.equal; zero tolerance'})
    require(equal_state(model.state_dict(),restored.state_dict()) and equal_state(optimizer.state_dict(),restored_optimizer.state_dict()) and
            equal_state(scheduler.state_dict(),restored_scheduler.state_dict()), 'PILOT_NEXT_UPDATE_PARITY')
    status = 'CPU_SYNTHETIC_PILOT_PREFLIGHT_PASS' if snapshot['synthetic'] else ('REAL_PILOT_PREFLIGHT_PASS' if torch.device(device).type=='cuda' else 'CPU_REAL_INPUT_DIAGNOSTIC_PASS')
    result={'schema_version':'cmmt.lvos_pilot.v1','scope':'diagnostic_only','status':status,'identity':identity,
        'synthetic':snapshot['synthetic'],'execution_kind':kind,'model_formally_approved':bool(lock.get('approved_by')),
        'unattended_training_ready':False,'state_training_ready':False,'optimizer_training_updates':3,
        'resume_comparison_updates_per_copy':1,'records':records,'checkpoint':entry,'normalization_sha256':sha256(output/'normalization.json'),
        'reload_output_difference':output_diffs,'next_update_difference':next_diff,'duration_seconds':deadline.elapsed(),
        'cuda_peak_allocated':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
        'cuda_peak_reserved':torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None}
    write_json(output/'STATUS.json',result); events.emit('pilot_end',status=status,metrics=result)
    return result
