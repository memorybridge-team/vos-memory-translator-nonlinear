"""실행 근거를 checksum으로 연결하는 LVOS gate runner. Synthetic은 GPU PASS가 아니다."""
from pathlib import Path
import math
import time
import copy
import torch
from .collection_contract import require
from .training_storage import write_json, content_hash, sha256, ExclusiveWriter
from .training_runner import code_provenance, rng_state, restore_rng
from .lvos_contract import (model_lock, validate_model_lock, json_read, evidence, verify_evidence,
                            resources, save_complete, load_complete, Events)
from .lvos_snapshot import verify_snapshot, record_loader
from .lvos_training import LVOSConfig, component_objective, fit_rms, forward_batch, optimizer_groups, make_scheduler


def loss_gate():
    # 작은 독립 scalar loop reference. Spatial 크기로 pointer weight가 희석되지 않는다.
    ps=torch.tensor([[[[2.,4.]]],[[[float('nan'),float('nan')]]],[[[1.,3.]]]])
    pp=torch.tensor([[2.,4.,6.],[float('nan')]*3,[1.,2.,3.]])
    ts=torch.zeros_like(ps); tp=torch.zeros_like(pp); mask=torch.tensor([True,False,True])
    cfg=LVOSConfig(); scales={'spatial':2.,'pointer':3.}
    raw,normalized,total=component_objective(ps,pp,ts,tp,mask,scales,cfg)
    ref_s=sum(sum(float(v)**2 for v in ps[i].flatten())/ps[i].numel() for i in (0,2))/2
    ref_p=sum(sum(float(v)**2 for v in pp[i])/pp[i].numel() for i in (0,2))/2
    require(abs(float(raw['spatial'])-ref_s)<1e-6 and abs(float(raw['pointer'])-ref_p)<1e-6,'LOSS_REFERENCE')
    require(abs(float(total)-(ref_s/4+ref_p/9))<1e-6,'LOSS_NORMALIZATION')
    expanded=ps.repeat(1,2,3,4)
    require(torch.equal(component_objective(expanded,pp,torch.zeros_like(expanded),tp,mask,scales,cfg)[2],total),'COMPONENT_ELEMENT_COUNT')
    # Unequal 2 + 1 record windows: multiplying mean by N and dividing total N.
    w=torch.tensor(1.,requires_grad=True); x=torch.tensor([1.,2.,4.])
    for piece in (x[:2],x[2:]):
        (((w*piece)**2).mean()*len(piece)).backward()
    require(abs(float(w.grad)/3-2*sum(float(v)**2 for v in x)/3)<1e-6,'ACCUMULATION_WEIGHT')
    try:
        component_objective(ps,pp,tp,tp,mask,scales,cfg)
    except ValueError:
        pass
    else:
        raise ValueError('LOSS_ALIGNMENT_FAILURE_NOT_REJECTED')
    return {'scalar_reference':{'spatial':ref_s,'pointer':ref_p,'total':float(total)},'padding_nan_excluded':True,
            'pointer_element_weight_independent':True,'partial_window_records':3}


def connectivity_gate(root,lock,output,device):
    snapshot=verify_snapshot(root)
    model=validate_model_lock(lock).to(device); cfg=LVOSConfig(microbatch=1,workers=0,pin_memory=False)
    scales=fit_rms(root,cfg); batch=next(iter(record_loader(root,'fit',microbatch=1,workers=0,pin_memory=False,shuffle=False)))
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg.weight_decay)[0],lr=cfg.lr)
    trajectory=[]; deltas=[]; initial={n:p.detach().clone() for n,p in model.named_parameters()}
    norms=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True); _,_,loss=forward_batch(model,batch,device,scales,cfg)
        require(torch.isfinite(loss),'G2_NONFINITE_LOSS'); loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'G2_NONFINITE_GRAD')
        norms.append({part:math.sqrt(sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
                     if n.startswith(part+'.') and p.grad is not None)) for part in ('spatial','pointer')})
        optimizer.step(); trajectory.append(float(loss.detach()))
        deltas.append({part:math.sqrt(sum(float((p.detach()-initial[n]).square().sum()) for n,p in model.named_parameters()
                       if n.startswith(part+'.'))) for part in ('spatial','pointer')})
    require(all(n['spatial']>0 and n['pointer']>0 for n in norms) and deltas[-1]['spatial']>0 and deltas[-1]['pointer']>0,'G2_UPDATE')
    require(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters()
                if n.startswith('spatial.blocks.')), 'G2_DEEP_GRADIENT')
    write_json(output,{'loss_trajectory':trajectory,'gradient_norms':norms,'delta_norms':deltas,
        'source_shapes':{k:list(v.shape) for k,v in batch.tensors.items()},'source_dtypes':{k:str(v.dtype) for k,v in batch.tensors.items()},
        'records':len(batch),'synthetic':snapshot['synthetic'],'device':device,'alpha_norm':float(model.spatial.alpha.detach().norm())})
    return snapshot


def roundtrip_gate(root,lock,output,device):
    model=validate_model_lock(lock).to(device); cfg=LVOSConfig(microbatch=1,workers=0,pin_memory=False)
    scales=fit_rms(root,cfg); batch=next(iter(record_loader(root,'fit',microbatch=1,workers=0,pin_memory=False,shuffle=False)))
    opt=torch.optim.AdamW(optimizer_groups(model,cfg.weight_decay)[0],lr=cfg.lr); sched=make_scheduler(opt,cfg,5)
    def step(m,o,s):
        o.zero_grad(set_to_none=True); forward_batch(m,batch,device,scales,cfg)[2].backward(); o.step(); s.step()
    step(model,opt,sched)
    path=Path(output)/'roundtrip.ckpt'
    save_complete(path,{'export':model.to_payload(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
        'rng':rng_state(),'scales':scales,'config':cfg.__dict__,'data_digest':verify_snapshot(root)['content_sha256']})
    payload,_=load_complete(path); restored=validate_model_lock(lock).to(device); restored.load_state_dict(payload['export']['state_dict'],strict=True)
    restored_opt=torch.optim.AdamW(optimizer_groups(restored,cfg.weight_decay)[0],lr=cfg.lr)
    restored_sched=make_scheduler(restored_opt,cfg,5); restored_opt.load_state_dict(payload['optimizer']); restored_sched.load_state_dict(payload['scheduler'])
    require(all(torch.equal(p,restored.state_dict()[n]) for n,p in model.state_dict().items()),'G4_WEIGHT_PARITY')
    require(torch.equal(forward_batch(model,batch,device,scales,cfg)[2],forward_batch(restored,batch,device,scales,cfg)[2]),'G4_OUTPUT_PARITY')
    restore_rng(payload['rng']); expected_rand=torch.rand(5); step(model,opt,sched)
    restore_rng(payload['rng']); require(torch.equal(torch.rand(5),expected_rand),'G4_RNG_PARITY'); step(restored,restored_opt,restored_sched)
    require(all(torch.equal(p,restored.state_dict()[n]) for n,p in model.state_dict().items()),'G4_UPDATE_PARITY')
    require(opt.state_dict()['param_groups']==restored_opt.state_dict()['param_groups'] and sched.state_dict()==restored_sched.state_dict(),'G4_OPT_SCHED_PARITY')
    write_json(Path(output)/'roundtrip.json',{'weights_output_optimizer_scheduler_rng_scales_parity':True,'boundary':'complete_optimizer_boundary',
                                           'checkpoint_sha256':sha256(path),'device':device})


def run_gates(output, *, root=None, lock=None, device='cpu', metric_contract=None, handoff_result=None, cpu_test_report=None,
              max_wall_seconds=None,direct_copy_result=None):
    output=Path(output); records=[]
    identity={'run_id':output.name,'code_sha':code_provenance()['package_source_sha256'],
              'model_config_digest':lock.get('digest') if lock else None,'collection_digest':None}
    snapshot=None
    if root and (Path(root)/'snapshot.json').exists():
        identity['collection_digest']=json_read(Path(root)/'snapshot.json')['content_sha256']
    events=Events(output,identity,'cpu_synthetic')
    started_all=time.perf_counter()
    with ExclusiveWriter(output):
        require(not (output/'gates.json').exists(),'GATE_OUTPUT_EXISTS')
        def run(gate,fn,paths,kind='cpu_synthetic',missing=None):
            started=time.perf_counter()
            if missing:
                record={'gate_id':gate,'status':'BLOCKED','reason_code':missing,'execution_kind':kind,'evidence':[]}
            else:
                try:
                    require(max_wall_seconds is None or time.perf_counter()-started_all<max_wall_seconds,'GATE_TIME_LIMIT')
                    fn()
                    record={'gate_id':gate,'status':'PASS','reason_code':None,'execution_kind':kind,'evidence':evidence(paths)}
                except Exception as exc:
                    path=output/f'{gate}-failure.json'; write_json(path,{'detail':str(exc),'reason_code':getattr(exc,'code','GATE_FAILURE')})
                    record={'gate_id':gate,'status':'FAIL','reason_code':getattr(exc,'code','GATE_FAILURE'),'execution_kind':kind,'evidence':evidence([path])}
            record['duration_seconds']=time.perf_counter()-started; records.append(record)
            events.emit('gate_end',**{k:v for k,v in record.items() if k!='evidence'},evidence_paths=[e['path'] for e in record['evidence']])
        def config_gate():
            validate_model_lock(lock)
            require(metric_contract and json_read(metric_contract).get('approved_by'),'METRIC_FREEZE_INPUT_MISSING')
            write_json(output/'G0.json',{'model_lock':lock,'metric_contract':json_read(metric_contract),'resources':resources(),'output':str(output.resolve())})
        run('G0',config_gate,[output/'G0.json'],missing='MODEL_AND_METRIC_FREEZE_INPUT_MISSING' if not lock or not lock.get('approved_by') or not metric_contract or not json_read(metric_contract).get('approved_by') else None)
        def audit():
            nonlocal snapshot
            snapshot=verify_snapshot(root); require(not snapshot['synthetic'] and snapshot['state']=='ready','REAL_FULL_AUDIT_REQUIRED')
            write_json(output/'G1.json',{'snapshot_digest':snapshot['content_sha256'],'audit':json_read(Path(root)/'audit.json')})
        run('G1',audit,[output/'G1.json'],'cpu_real_cache',missing='REAL_CACHE_SNAPSHOT_REQUIRED' if not root or not (Path(root)/'snapshot.json').exists() or json_read(Path(root)/'snapshot.json').get('synthetic') else None)
        real_gpu=bool(root and snapshot and not snapshot['synthetic'] and torch.device(device).type=='cuda' and torch.cuda.is_available() and lock and lock.get('approved_by'))
        missing=None if real_gpu else 'REAL_CACHE_APPROVED_MODEL_AND_GPU_REQUIRED'
        run('G2',lambda:connectivity_gate(root,lock,output/'G2.json',device),[output/'G2.json'],'gpu_real_checkpoint',missing)
        run('G3',lambda:write_json(output/'G3.json',loss_gate()),[output/'G3.json'])
        run('G4',lambda:roundtrip_gate(root,lock,output/'G4',device),[output/'G4/roundtrip.json',output/'G4/roundtrip.ckpt'],'gpu_real_checkpoint',missing)
        def handoff():
            from .lvos_evaluation import read_result, validate_rows
            from .training_data import read_manifest
            result=read_result(handoff_result)
            protocol=read_manifest(result['protocol_path']); validate_rows(result,protocol)
            require(set(result['expected_case_ids'])==set(protocol['expected_case_ids']),'G5_COMPLETE_PROTOCOL')
            require(result['execution_kind']=='gpu_real_checkpoint' and result.get('merged') and
                result['collection_digest']==identity['collection_digest'] and result['model_config_digest']==identity['model_config_digest'] and
                result['scope']=='monitor' and result['epoch']==0 and result['method']=='learned' and
                result['rows'] and all(r['trace']['past_encoder_call_count']==0 and r['trace']['metadata_preserved'] and
                    not any(f<=next(c['switch_runtime_index'] for c in protocol['cases'] if c['case_id']==r['case_id'])
                        for f in r['trace']['encoder_frame_ids']) and
                    r['trace']['continuation_frames']==next(c['expected_runtime_indices'] for c in protocol['cases'] if c['case_id']==r['case_id'])
                    for r in result['rows']),'REAL_HANDOFF_EVIDENCE')
            direct=read_result(direct_copy_result); validate_rows(direct,protocol)
            require(direct['method']=='direct_copy' and direct['epoch']==0 and direct.get('merged') and
                direct['execution_kind']=='gpu_real_checkpoint' and set(direct['expected_case_ids'])==set(protocol['expected_case_ids']) and
                all(direct[k]==result[k] for k in ('protocol_digest','metric_contract_digest','collection_digest','checkpoint_sha')),
                'G5_DIRECT_COPY_BASELINE')
            write_json(output/'G5.json',result)
            write_json(output/'G5-direct.json',direct)
        run('G5',handoff,[output/'G5.json',output/'G5-direct.json'],'gpu_real_checkpoint',missing='REAL_TARGET_IDENTITY_AND_DIRECT_COPY_RESULTS_REQUIRED' if not handoff_result or not direct_copy_result else None)
        def safety():
            value=json_read(cpu_test_report)
            verify_evidence(value['evidence']); require(value['exit_code']==0 and value['suite']=='lvos_pipeline' and value['tests']>0,'CPU_TEST_REPORT')
            require(value['tested_source_sha256']==identity['code_sha'] and
                    value['tested_suite_sha256']==sha256(Path(__file__).resolve().parents[2]/'tests/test_lvos_pipeline.py'),'STALE_CPU_TEST_EVIDENCE')
            write_json(output/'G6.json',value)
        run('G6',safety,[output/'G6.json'],missing='EXECUTED_CPU_SAFETY_TEST_REPORT_REQUIRED' if not cpu_test_report else None)
        report={'schema_version':'cmmt.lvos_gates.v1','identity':identity,'gates':records,'unattended_training_ready':False}
        write_json(output/'gates.json',report)
        try:
            authorize_training(output/'gates.json',identity['collection_digest'],identity['model_config_digest'])
            report['unattended_training_ready']=True; write_json(output/'gates.json',report)
        except ValueError:
            pass
        return report


def authorize_training(path, data_digest, model_digest):
    require(path,'REAL_GATES_REQUIRED'); report=json_read(path)
    require(report['identity']['collection_digest']==data_digest and report['identity']['model_config_digest']==model_digest and
            report['identity']['code_sha']==code_provenance()['package_source_sha256'],'GATE_IDENTITY_CHANGED')
    gates={g['gate_id']:g for g in report['gates']}
    for gate in ('G0','G1','G2','G3','G4','G5','G6'):
        require(gates[gate]['status']=='PASS','GATE_NOT_PASSED',gate); verify_evidence(gates[gate]['evidence'])
    for gate in ('G2','G4','G5'):
        require(gates[gate]['execution_kind']=='gpu_real_checkpoint','SYNTHETIC_GATE_CANNOT_AUTHORIZE',gate)
    require(gates['G1']['execution_kind']=='cpu_real_cache','REAL_AUDIT_REQUIRED')
