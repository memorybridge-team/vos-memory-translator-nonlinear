"""운영 fault/benchmark bridge CPU 증거. Real cache·GPU 검증과 구분한다."""
import copy
import json
from pathlib import Path
import pytest
import torch
from test_lvos_pipeline import fixture_snapshot, fixture_data, selected
from vos_memory_inspector.training_storage import write_json, sha256, source_sha256, content_hash
from vos_memory_inspector.lvos_contract import model_lock, load_complete
from vos_memory_inspector.lvos_checkpoint import CheckpointStore
from vos_memory_inspector.lvos_budget import Deadline
from vos_memory_inspector.lvos_training import LVOSConfig, train_snapshot
from vos_memory_inspector import lvos_training as tr, lvos_cli as cli
from vos_memory_inspector.collection_contract import Rejection


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_equal(x,y)
    else: assert a==b


def test_canonical_source_bytes_and_real_change(tmp_path):
    a=tmp_path/'a.py'; b=tmp_path/'b.py'
    a.write_bytes(b'x=1\r\ny=2\r\n'); b.write_bytes(b'x=1\ny=2\n')
    assert sha256(a)!=sha256(b) and source_sha256(a)==source_sha256(b)
    b.write_bytes(b'x=2\ny=2\n'); assert source_sha256(a)!=source_sha256(b)


@pytest.mark.parametrize('stage',['before_body','after_body','after_marker','after_pointer'])
def test_checkpoint_fault_recovery_parity(fixture_snapshot,tmp_path,monkeypatch,stage):
    root,_,_,_=fixture_snapshot; lock=model_lock('CPU fixture only')
    cfg=LVOSConfig(microbatch=1,accumulation=3,max_epochs=2,workers=0,pin_memory=False)
    continuous=tmp_path/'continuous'; interrupted=tmp_path/'interrupted'
    train_snapshot(root,lock,continuous,cfg,device='cpu')
    original=CheckpointStore.publish; once=[False]
    def publish(self,payload):
        self._fault_epoch=payload['epoch']; return original(self,payload)
    def boundary(self,point):
        if self.root==interrupted and getattr(self,'_fault_epoch',None)==1 and point==stage and not once[0]:
            once[0]=True; raise RuntimeError('injected checkpoint crash')
    monkeypatch.setattr(CheckpointStore,'publish',publish); monkeypatch.setattr(CheckpointStore,'boundary',boundary)
    with pytest.raises(RuntimeError,match='injected checkpoint crash'):
        train_snapshot(root,lock,interrupted,cfg,device='cpu')
    monkeypatch.setattr(CheckpointStore,'boundary',lambda *a:None)
    train_snapshot(root,lock,interrupted,cfg,device='cpu',resume=Path('latest'))
    a,_=load_complete(continuous/'checkpoints/epoch-00002.ckpt')
    b,_=load_complete(interrupted/'checkpoints/epoch-00002.ckpt')
    for key in ('export','optimizer','scheduler','rng','scales','history','epoch','optimizer_step'):
        assert_equal(a[key],b[key])
    assert json.loads((interrupted/'last.ckpt.json').read_text())['path']=='checkpoints/epoch-00002.ckpt'


@pytest.mark.parametrize('stage',['before_body','after_body','after_marker','after_pointer'])
def test_epoch_zero_initialize_recovery(fixture_snapshot,tmp_path,monkeypatch,stage):
    root,_,_,_=fixture_snapshot; out=tmp_path/'init'; once=[False]
    def boundary(self,point):
        if point==stage and not once[0]: once[0]=True; raise RuntimeError('init crash')
    monkeypatch.setattr(CheckpointStore,'boundary',boundary)
    with pytest.raises(RuntimeError,match='init crash'):
        train_snapshot(root,model_lock('CPU only'),out,LVOSConfig(workers=0),device='cpu',mode='initialize')
    monkeypatch.setattr(CheckpointStore,'boundary',lambda *a:None)
    train_snapshot(root,model_lock('CPU only'),out,LVOSConfig(workers=0),device='cpu',mode='initialize')
    payload,_=load_complete(out/'checkpoints/epoch-00000.ckpt')
    assert payload['epoch']==0 and json.loads((out/'STATUS.json').read_text())['status']=='INITIALIZED'


def test_recovery_preserves_corrupt_foreign_and_orphan(tmp_path):
    ident={'id':'ours'}; store=CheckpointStore(tmp_path,ident)
    payload={'epoch':0,'identity':ident,'boundary':'complete_epoch','value':torch.ones(1)}
    store.publish(payload)
    path=tmp_path/'checkpoints/epoch-00000.ckpt'; old=path.read_bytes(); path.write_bytes(old+b'x')
    with pytest.raises(Rejection,match='CHECKPOINT_TRANSACTION_CHECKSUM'): store.reconcile()
    assert path.read_bytes()==old+b'x'
    path.write_bytes(old)
    with pytest.raises(Rejection,match='FOREIGN_CHECKPOINT_JOURNAL'): CheckpointStore(tmp_path,{'id':'foreign'}).reconcile()
    orphan=tmp_path/'checkpoints/epoch-00001.ckpt'; torch.save(payload,orphan)
    with pytest.raises(Rejection,match='INCOMPLETE_ARTIFACT'): store.reconcile()
    assert orphan.exists()


@pytest.mark.parametrize('stage',['snapshot_shard_hash','fit_rms_batch','loader_shard_after','state_dev_batch','finalization'])
def test_whole_run_deadline_safe_boundaries(fixture_snapshot,tmp_path,stage):
    root,_,_,_=fixture_snapshot; now=[0.]; deadline=Deadline(5,clock=lambda:now[0])
    original=deadline.check
    def check(point):
        if point==stage: now[0]=6.
        original(point)
    deadline.check=check
    out=tmp_path/'budget'
    train_snapshot(root,model_lock('CPU only'),out,LVOSConfig(microbatch=1,max_epochs=1,workers=0,pin_memory=False),
                   device='cpu',deadline=deadline)
    status=json.loads((out/'STATUS.json').read_text())
    assert status['status']=='STOPPED' and status['reason']=='budget_stop' and status['incomplete_stage']==stage
    assert not (out/'COMPLETED.json').exists() and not status['partial_dev_is_complete']
    if stage=='state_dev_batch':
        assert not (out/'checkpoints/epoch-00001.ckpt').exists()


@pytest.mark.parametrize('statuses,ready,required,expected',[
    (['PASS'],True,False,0),(['PASS'],True,True,0),(['FAIL'],False,False,2),
    (['FAIL','BLOCKED'],False,True,2),(['PASS','BLOCKED'],False,False,0),
    (['BLOCKED'],False,True,3),(['PASS'],False,True,3)])
def test_gate_exit_semantics(tmp_path,monkeypatch,capsys,statuses,ready,required,expected):
    import vos_memory_inspector.lvos_gates as g
    monkeypatch.setattr(g,'run_gates',lambda *a,**kw:{'gates':[{'status':s} for s in statuses],'unattended_training_ready':ready})
    args=['gates','--output',str(tmp_path/'gates')]+(['--require-ready'] if required else [])
    assert cli.main(args)==expected
    assert 'DONE' not in capsys.readouterr().out


def test_real_bad_model_gate_returns_nonzero(tmp_path):
    lock=model_lock('CPU fixture, not approval'); lock['config']['d_model']=32
    lock['digest']=content_hash({k:v for k,v in lock.items() if k!='digest'})
    p=tmp_path/'lock.json'; c=tmp_path/'metric.json'; write_json(p,lock); write_json(c,{'approved_by':'CPU fixture'})
    out=tmp_path/'bad'
    assert cli.main(['gates','--output',str(out),'--model-lock',str(p),'--metric-contract',str(c)])==2
    g=json.loads((out/'gates.json').read_text())['gates'][0]
    assert g['status']=='FAIL' and g['reason_code']=='FROZEN_ARCHITECTURE'


def fixture_module(name):
    import importlib.util
    path=Path(__file__).parent/'fixtures'/('benchmark_'+name+'.py')
    lock=json.loads((Path(__file__).parents[1]/'configs/lvos_benchmark_reference_lock.json').read_text())
    assert source_sha256(path)==lock['sources'][name]
    spec=importlib.util.spec_from_file_location('pinned_'+name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_discovery_readonly_exact_requests_and_unknown(fixture_snapshot,tmp_path):
    from vos_memory_inspector.lvos_discovery import discover
    root,_,_,requests=fixture_snapshot; originals=Path(requests[0]['path']).parent
    before={str(p):sha256(p) for p in originals.iterdir()}
    selection=tmp_path/'selection.json'; from vos_memory_inspector.training_data import save_manifest
    save_manifest(selection,selected())
    out=tmp_path/'discovery'; r=discover([originals],selection,out,trusted=True,stable_seconds=0)
    assert r['completed_pairs']==2 and r['valid_memory_records']==4 and r['status']=='BLOCKED'
    assert json.loads((out/'requests.json').read_text())==requests
    assert before=={str(p):sha256(p) for p in originals.iterdir()}
    untrusted=discover([originals],selection,tmp_path/'untrusted',trusted=False,stable_seconds=0)
    assert untrusted['completed_pairs']==0 and len(untrusted['unknown'])==2
    missing=discover([tmp_path/'missing_runtime'],selection,tmp_path/'missing_report')
    assert missing['completed_pairs']==0 and missing['unknown'][0]['reason']=='RUNTIME_ROOT_UNAVAILABLE'


def test_discovery_duplicate_incomplete_corrupt(fixture_snapshot,tmp_path):
    import shutil
    from vos_memory_inspector.lvos_discovery import discover
    from vos_memory_inspector.training_data import save_manifest
    _,_,_,req=fixture_snapshot; originals=Path(req[0]['path']).parent
    selection=tmp_path/'selection.json'; save_manifest(selection,selected())
    for suffix in ('','.sha256','.complete.json'):
        shutil.copyfile(req[0]['path']+suffix,originals/('duplicate.pt'+suffix))
    (originals/'partial.pt').write_bytes(b'partial')
    Path(req[1]['path']).write_bytes(b'corrupt')
    report=discover([originals],selection,tmp_path/'discovery',trusted=True,stable_seconds=0)
    assert report['completed_pairs']==0 and len(report['duplicate_case_ids'])==1
    assert len(report['rejected'])>=2 and any(f['reason']=='MISSING_CHECKSUM' for f in report['unknown'])
    assert (originals/'partial.pt').read_bytes()==b'partial'


def test_frozen_split_counterexample_no_independent_resplit():
    import hashlib
    root=Path(__file__).parents[1]/'manifests'
    fit=json.loads((root/'lvosv2_train_v1_fit.json').read_text())['videos']
    dev=json.loads((root/'lvosv2_train_v1_development.json').read_text())['videos']
    # Exact pinned baseline split formula; settings.DEV_FRACTION=.2.
    is_dev=lambda v:int(hashlib.md5(v.encode('utf-8')).hexdigest()[:8],16)/0xFFFFFFFF<.2
    assert len(fit)==347 and len(dev)==73 and not set(fit)&set(dev)
    assert sum(is_dev(v) for v in fit)==64 and sum(not is_dev(v) for v in dev)==60


def test_canonical_metric_counterexample_and_no_clipping():
    from vos_memory_inspector.lvos_metrics import canonical_retention
    metric=fixture_module('metric'); legacy=fixture_module('legacy_metrics')
    replay=[{'case_id':str(f),'video':'a','fraction':f,'object_id':1,'jf':v,'j':v,'f':v}
            for f,v in zip((.25,.5,.75),(.9,.3,.6))]
    rows=[{**r,'jf':v,'j':v,'f':v} for r,v in zip(replay,(.6,.3,.6))]
    result=canonical_retention(metric,rows,replay)
    assert result['score']==pytest.approx(88.8888888889)
    assert legacy.retention(rows,replay,'jf')==pytest.approx(83.3333333333)
    high=[{**r,'jf':1.,'j':1.,'f':1.} for r in replay]
    assert canonical_retention(metric,high,replay)['score']>100
    with pytest.raises(Rejection,match='RETENTION_CASE_COVERAGE'): canonical_retention(metric,rows[:-1],replay)
    zero=[{**r,'jf':0.} for r in replay]
    assert canonical_retention(metric,rows,zero)['score'] is None
    missing=[{**r,'jf':None} for r in replay]
    with pytest.raises(Rejection,match='RETENTION_UNDEFINED_MISMATCH'): canonical_retention(metric,rows,missing)


def test_baseline_row_requires_real_provenance():
    from vos_memory_inspector.lvos_benchmark import adapt_baseline_row
    raw={'video':'a','object':1,'baseline':'full_replay','switch_frame':5,'switch_name':'50','j':.5,'f':.5,'jf':.5}
    fixed={'video_id':'a','object_id':1,'switch_runtime_index':5,'fraction':.5,'case_id':'case',
           'prompt':{'sha256':'prompt'},'scored_frames':[{'runtime_index':6}]}
    with pytest.raises(Rejection,match='BASELINE_ROW_PROVENANCE_MISSING'): adapt_baseline_row(raw,fixed,{})


def test_full_replay_late_prompt_sparse_scoring_once(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    import vos_memory_inspector.sam2_state as state
    from vos_memory_inspector.lvos_benchmark import replay_group
    from vos_memory_inspector.lvos_evaluation import score_predictions
    prompt=tmp_path/'prompt.png'; Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(prompt)
    gt=tmp_path/'gt.png'; Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(gt)
    monkeypatch.setattr(state,'init_sam2_inference_state_without_warmup',lambda *a,**kw:{})
    class Predictor:
        prompts=[]
        def _get_image_feature(self,state,frame_idx,batch_size): return self.forward_image(torch.ones(1))
        def forward_image(self,image): return image
        def add_new_mask(self,state,frame_idx,obj_id,mask):
            self.prompts.append(frame_idx); self._get_image_feature(state,frame_idx,1)
        def propagate_in_video(self,state,start_frame_idx,max_frame_num_to_track,reverse):
            for i in range(start_frame_idx,start_frame_idx+max_frame_num_to_track+1):
                self._get_image_feature(state,i,1); yield i,[1],torch.ones(1,1,4,4)
    rows=[{'case_id':str(t),'video_id':'a','video_dir':'CPU_fixture','object_id':1,'frame_map_sha256':'fixture',
        'prompt':{'runtime_index':3,'official_id':16,'path':str(prompt),'sha256':sha256(prompt)},
        'switch_runtime_index':t,'expected_runtime_indices':list(range(t+1,10)),
        'scored_frames':[{'runtime_index':9,'official_id':46,'path':str(gt),'sha256':sha256(gt),'visible':True}]}
        for t in (4,6,8)]
    predictor=Predictor(); predictions,trace=replay_group(predictor,rows,fixture_module('full_replay'),{'name':'full_history'})
    assert predictor.prompts==[3] and trace['rollout_frames']==list(range(3,10))
    assert min(trace['encoder_frame_ids'])==3 and set(predictions)=={9}
    for row in rows:
        assert score_predictions(predictions,row,fixture_module('metric'))['jf']==1.


def test_final_merge_and_table_use_one_canonical_reducer(tmp_path,monkeypatch):
    from vos_memory_inspector.lvos_evaluation import merge_results
    from vos_memory_inspector.lvos_benchmark import comparison_table
    import vos_memory_inspector.lvos_evaluation as evaluation
    import vos_memory_inspector.lvos_benchmark as benchmark
    from vos_memory_inspector.training_data import save_manifest,read_manifest
    metric=fixture_module('metric')
    monkeypatch.setattr(evaluation,'benchmark_module',lambda *a:metric)
    monkeypatch.setattr(benchmark,'benchmark_module',lambda *a:metric)
    fixed=[{'case_id':str(f),'video_id':'a','object_id':1,'fraction':f,
            'scored_frames':[{'visible':True}]} for f in (.25,.5,.75)]
    p=tmp_path/'protocol.json'; save_manifest(p,{'scope':'full_development','collection_digest':'fixture',
        'cases':fixed,'expected_case_ids':[r['case_id'] for r in fixed],'synthetic':True})
    protocol=read_manifest(p); c=tmp_path/'metric.json'; write_json(c,{'approved_by':'CPU fixture'})
    identity={'scope':'full_development','status':'PASS','collection_digest':'fixture','model_config_digest':'fixture',
        'checkpoint_sha':'fixture','epoch':2,'shard_count':1,'shard_index':0,'method':'learned',
        'execution_kind':'cpu_synthetic','protocol_digest':protocol['content_sha256'],'protocol_path':str(p),
        'target_identity_digest':'target','target_policy_digest':'read','metric_contract_digest':content_hash(json.loads(c.read_text())),
        'expected_case_ids':protocol['expected_case_ids']}
    def save(folder,method,values):
        rows=[{'case_id':r['case_id'],'video':'a','object_id':1,'fraction':r['fraction'],
               'j':v,'f':v,'jf':v,'visible_frames':1,'absent_frames':0} for r,v in zip(fixed,values)]
        result={**identity,'method':method,'rows':rows}; root=tmp_path/folder
        write_json(root/'result.json',result); write_json(root/'READY.json',{'sha256':sha256(root/'result.json')})
        return root/'result.json'
    replay=save('replay','full_replay',[.9,.3,.6]); learned=save('learned','learned',[.6,.3,.6])
    merge_results([replay],p,c,tmp_path,tmp_path/'replay_merge')
    denominator=tmp_path/'replay_merge/result.json'
    result=merge_results([learned],p,c,tmp_path,tmp_path/'method_merge',replay=denominator)
    table=comparison_table([tmp_path/'method_merge/result.json'],denominator,p,c,tmp_path,tmp_path/'table.json')
    assert result['primary_score']==table['rows'][0]['canonical_retention_percent']==pytest.approx(88.8888888889)
    assert table['rows'][0]['raw_j_f_jf']['jf']==pytest.approx(.5)
    assert result['monitor_metric'] is None and result['monitor_score'] is None


@pytest.fixture
def worker_fixture(fixture_snapshot,tmp_path,monkeypatch):
    from vos_memory_inspector.training_data import save_manifest,read_manifest
    import vos_memory_inspector.lvos_evaluation as ev
    root,snapshot,_,_=fixture_snapshot
    protocol_path=tmp_path/'queue_protocol.json'; contract=tmp_path/'queue_metric.json'
    rows=[{'case_id':'CPU_QUEUE_FIXTURE_'+v,'video_id':v,'object_id':1,'fraction':.5,
           'scored_frames':[{'visible':True}],'expected_runtime_indices':[6]} for v in ('a','b')]
    save_manifest(protocol_path,{'scope':'monitor','collection_digest':snapshot['content_sha256'],'cases':rows,
        'expected_case_ids':[r['case_id'] for r in rows],'synthetic':True})
    write_json(contract,{'approved_by':'CPU fake-worker fixture only'})
    monkeypatch.setattr(ev,'benchmark_module',lambda *a:fixture_module('metric'))
    run=tmp_path/'run'; cfg=LVOSConfig(workers=0,pin_memory=False)
    train_snapshot(root,model_lock('CPU fake-worker only'),run,cfg,mode='initialize',device='cpu',
                   monitor_protocol=protocol_path,metric_contract=contract)
    request=run/'evaluation/requests/epoch-00000.json'
    return root,run,request,protocol_path,contract


def fake_executor(request,protocol,index,count,output,deadline):
    from vos_memory_inspector.lvos_evaluation import case_assignments
    ids=case_assignments(protocol,count)[index]
    rows=[{'case_id':r['case_id'],'video':r['video_id'],'object_id':r['object_id'],'fraction':r['fraction'],
           'j':.5,'f':.5,'jf':.5,'visible_frames':sum(s['visible'] for s in r['scored_frames']),
           'absent_frames':sum(not s['visible'] for s in r['scored_frames'])} for r in protocol['cases'] if r['case_id'] in ids]
    result={'status':'PASS','execution_kind':'cpu_synthetic','epoch':request['epoch'],'method':'learned',
        'checkpoint_sha':request['checkpoint_sha'],'checkpoint':request['checkpoint'],'protocol_path':request['protocol'],
        'protocol_digest':request['protocol_digest'],'metric_contract_digest':request['metric_contract_digest'],
        'collection_digest':request['identity']['collection_digest'],'model_config_digest':request['identity']['model_config_digest'],
        'scope':'monitor','expected_case_ids':ids,'rows':rows,'shard_index':index,'shard_count':count,
        'target_identity_digest':'CPU_FIXTURE_TARGET','target_policy_digest':'CPU_FIXTURE_POLICY'}
    result['code_sha']=request['identity']['code_sha']
    write_json(output/'result.json',result); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
    return result


def test_worker_disjoint_success_explicit_merge_once(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import run_worker,merge_request,pending_requests
    root,run,request,_,_=worker_fixture
    dry=run_worker(run,root,tmp_path,index=0,count=2)
    assert dry['dry_run'] and not dry['execution_performed'] and not (run/'evaluation/workers').exists()
    run_worker(run,root,tmp_path,index=0,count=2,dry_run=False,executor=fake_executor)
    with pytest.raises(Rejection,match='WORKER_SHARD_PENDING'): merge_request(run,request,tmp_path,count=2)
    run_worker(run,root,tmp_path,index=1,count=2,dry_run=False,executor=fake_executor)
    result=merge_request(run,request,tmp_path,count=2)
    assert result['monitor_jf_proxy']==.5 and len(result['rows'])==2 and pending_requests(run)==[]
    path=run/'evaluation/results/epoch-00000/result.json'; original=sha256(path)
    assert merge_request(run,request,tmp_path,count=2)==result and sha256(path)==original


def test_worker_crash_retry_and_foreign_result(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import run_worker,merge_request
    root,run,request,_,_=worker_fixture
    def crash(*args): raise RuntimeError('CPU simulated worker crash')
    failed=run_worker(run,root,tmp_path,dry_run=False,executor=crash)
    assert failed['status']=='FAIL' and list((run/'evaluation/workers').rglob('FAILED.json'))
    def foreign(*args):
        value=fake_executor(*args); value['checkpoint_sha']='foreign'; output=args[4]
        write_json(output/'result.json',value); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
        return value
    # Preserve foreign evidence in a separate fixture namespace, rather than auto replacing it.
    done=run_worker(run,root,tmp_path,dry_run=False,executor=fake_executor)
    assert done['status']=='DONE'; merge_request(run,request,tmp_path)
    skips=run_worker(run,root,tmp_path,dry_run=False,executor=foreign)
    assert skips['records'][0]['status']=='VERIFIED_SKIP'


def test_worker_timeout_after_ready_recovers_without_reexecution(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import run_worker
    root,run,_,_,_=worker_fixture; now=[0.]; deadline=Deadline(5,clock=lambda:now[0])
    def delayed(*args):
        value=fake_executor(*args); now[0]=6.; return value
    stopped=run_worker(run,root,tmp_path,dry_run=False,executor=delayed,deadline=deadline)
    assert stopped['status']=='STOPPED' and list((run/'evaluation/workers').rglob('READY.json'))
    def never(*args): raise AssertionError('verified READY must be reused')
    recovered=run_worker(run,root,tmp_path,dry_run=False,executor=never)
    assert recovered['records'][0]['status']=='RECOVERED_VERIFIED_RESULT'


def test_stale_request_and_run_ownership(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import read_request,run_worker,GPULease
    from vos_memory_inspector.training_storage import ExclusiveWriter
    root,run,request,protocol,_=worker_fixture
    with pytest.raises(Rejection,match='REQUEST_FOREIGN_RUN'): read_request(tmp_path/'other_run',request)
    with ExclusiveWriter(run/'evaluation/workers/worker-0'):
        with pytest.raises(FileExistsError): run_worker(run,root,tmp_path,dry_run=False,executor=fake_executor)
    with GPULease(tmp_path/'leases','CPU_HOST','GPU-CPU_FIXTURE','one'):
        with pytest.raises(FileExistsError):
            with GPULease(tmp_path/'leases','CPU_HOST','GPU-CPU_FIXTURE','two'): pass
    write_json(protocol,{'changed':True})
    with pytest.raises((ValueError,KeyError)): read_request(run,request)


def test_training_completion_is_distinct_from_pending_eval(fixture_snapshot,tmp_path):
    from vos_memory_inspector.training_data import save_manifest
    root,snapshot,_,_=fixture_snapshot; p=tmp_path/'protocol.json'; c=tmp_path/'metric.json'
    rows=[{'case_id':'CPU_FIXTURE','video_id':'a','object_id':1,'fraction':.5,'scored_frames':[{'visible':True}]}]
    save_manifest(p,{'scope':'monitor','collection_digest':snapshot['content_sha256'],'cases':rows,
        'expected_case_ids':['CPU_FIXTURE'],'synthetic':True}); write_json(c,{'approved_by':'CPU fixture'})
    run=tmp_path/'queued'
    train_snapshot(root,model_lock('CPU queue fixture'),run,LVOSConfig(max_epochs=1,microbatch=1,workers=0,pin_memory=False,
        monitor_every=1,max_outstanding_evaluations=2),device='cpu',monitor_protocol=p,metric_contract=c)
    status=json.loads((run/'STATUS.json').read_text())
    assert status['training_status']=='COMPLETED' and status['evaluation_status']=='PENDING' and len(status['pending_requests'])==2


def test_sequential_train_eval_exclusive_lease_cpu_fixture(fixture_snapshot,tmp_path,monkeypatch):
    from vos_memory_inspector.lvos_jobs import run_worker,GPULease
    from vos_memory_inspector.lvos_evaluation import freeze_protocol
    import vos_memory_inspector.lvos_evaluation as ev
    root,_,settings,_=fixture_snapshot; p=tmp_path/'monitor.json'; c=tmp_path/'metric.json'
    runtime={'approved_by':'CPU fixture only','ground_truth_format':'palette_object_id_png','monitor_video_count':1,'videos':settings}
    freeze_protocol(root,Path(__file__).parents[1]/'manifests/lvosv2_train_v1.json',runtime,p,videos=['1umdNtrE'])
    write_json(c,{'approved_by':'CPU fixture only'}); monkeypatch.setattr(ev,'benchmark_module',lambda *a:fixture_module('metric'))
    run=tmp_path/'sequential'
    with GPULease(tmp_path/'leases','CPU_HOST','GPU-CPU_FIXTURE',str(run)) as lease:
        def callback(run,deadline):
            result=run_worker(run,root,tmp_path,dry_run=False,executor=fake_executor,merge=True,deadline=deadline,lease=lease)
            assert result['status']=='DONE'
        train_snapshot(root,model_lock('CPU sequential fixture'),run,
            LVOSConfig(max_epochs=2,microbatch=1,workers=0,pin_memory=False,monitor_every=1,max_outstanding_evaluations=1),
            device='cpu',monitor_protocol=p,metric_contract=c,evaluation_callback=callback)
    status=json.loads((run/'STATUS.json').read_text())
    assert status['training_status']==status['evaluation_status']=='COMPLETED' and status['pending_requests']==[]


def test_paid_gpu_budget_and_uuid_guard_cpu_mock(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True); monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-CPU_FIXTURE')
    args=SimpleNamespace(device='cuda:0',execute_approved=True,max_wall_seconds=3600,lease_root=tmp_path,
        host_id='CPU_HOST',physical_gpu_uuid='GPU-CPU_FIXTURE',quoted_hourly_rate=2,total_budget=1,budget_currency='USD')
    with pytest.raises(Rejection,match='GPU_BUDGET_EXCEEDED'): cli.approved_gpu(args)
    args.total_budget=2; cli.approved_gpu(args)
    args.physical_gpu_uuid='GPU-FOREIGN'
    with pytest.raises(Rejection,match='EXPLICIT_FREE_GPU_UUID'): cli.approved_gpu(args)


def test_worker_foreign_output_is_rejected(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import run_worker
    root,run,_,_,_=worker_fixture
    def foreign(*args):
        result=fake_executor(*args); result['checkpoint_sha']='FOREIGN'
        write_json(args[4]/'result.json',result); write_json(args[4]/'READY.json',{'sha256':sha256(args[4]/'result.json')})
        return result
    value=run_worker(run,root,tmp_path,dry_run=False,executor=foreign)
    assert value['status']=='FAIL' and not list((run/'evaluation/workers').rglob('verified.json'))
    failed=json.loads(next((run/'evaluation/workers').rglob('FAILED.json')).read_text())
    assert failed['reason']=='WORKER_RESULT_STALE_OR_FOREIGN'


@pytest.mark.parametrize('slow_stage',['fit_rms','state_dev','evaluation_wait'])
def test_whole_deadline_after_slow_operations(fixture_snapshot,tmp_path,monkeypatch,slow_stage):
    from vos_memory_inspector.training_data import save_manifest
    import vos_memory_inspector.lvos_evaluation as ev
    root,snapshot,_,_=fixture_snapshot; now=[0.]; deadline=Deadline(5,clock=lambda:now[0])
    p=tmp_path/'protocol.json'; c=tmp_path/'metric.json'
    save_manifest(p,{'scope':'monitor','collection_digest':snapshot['content_sha256'],'cases':[],
                     'expected_case_ids':[],'synthetic':True}); write_json(c,{'approved_by':'CPU fixture'})
    if slow_stage in ('fit_rms','state_dev'):
        original=getattr(tr,slow_stage)
        def slow(*args,**kw):
            result=original(*args,**kw); now[0]=6.; return result
        monkeypatch.setattr(tr,slow_stage,slow)
    else:
        monkeypatch.setattr(ev,'consume_monitor_results',lambda *a:2)
        original=deadline.check
        def check(stage):
            if stage=='evaluation_wait': now[0]=6.
            original(stage)
        deadline.check=check
    run=tmp_path/'budget'
    kwargs={'monitor_protocol':p,'metric_contract':c} if slow_stage=='evaluation_wait' else {}
    train_snapshot(root,model_lock('CPU only'),run,LVOSConfig(max_epochs=1,microbatch=1,workers=0,pin_memory=False,
        max_wall_seconds=5,monitor_every=1,max_outstanding_evaluations=1),device='cpu',deadline=deadline,**kwargs)
    status=json.loads((run/'STATUS.json').read_text())
    assert status['status']=='STOPPED' and status['reason']=='budget_stop' and not (run/'COMPLETED.json').exists()
    if slow_stage=='state_dev': assert not (run/'checkpoints/epoch-00001.ckpt').exists()
    if slow_stage=='evaluation_wait': assert (run/'checkpoints/epoch-00001.ckpt.complete.json').exists()


def test_small_reproducibility_bundle_strict_load_and_sanitized(worker_fixture,tmp_path):
    import zipfile
    from vos_memory_inspector.lvos_release import bundle_release
    from vos_memory_inspector.lvos_contract import save_complete
    from vos_memory_inspector.transformer_translator import TransformerStateTranslator
    root,run,_,_,_=worker_fixture; release=tmp_path/'release'
    payload,_=load_complete(run/'checkpoints/epoch-00000.ckpt'); save_complete(release/'best_model.pth',payload['export'])
    write_json(release/'selection.json',{'selection_claim':'best among evaluated shortlist; not global best epoch',
                                      'execution_kind':'cpu_synthetic','protocol_digest':'CPU_FIXTURE_FULL_DEV'})
    with pytest.raises(Rejection,match='SYNTHETIC_BUNDLE_EXPLICIT_ONLY'):
        bundle_release(release,run,tmp_path/'blocked')
    out=tmp_path/'bundle'; result=bundle_release(release,run,out,include_resume=True,allow_synthetic=True)
    assert result['execution_kind']=='cpu_synthetic' and result['sanitized'] and len(result['files'])==5
    with zipfile.ZipFile(out/'bundle.zip') as archive:
        assert all(not Path(n).is_absolute() for n in archive.namelist())
        archive.extractall(tmp_path/'restore')
    restored,_=load_complete(tmp_path/'restore/files/resume.ckpt')
    TransformerStateTranslator.from_payload(restored['export'])
    assert 'C:' not in (out/'artifact-index.json').read_text()


def test_protocol_rejects_foreign_prompt_and_window(fixture_snapshot,tmp_path):
    from vos_memory_inspector.lvos_evaluation import freeze_protocol,validate_protocol_snapshot
    root,snapshot,settings,_=fixture_snapshot
    runtime={'approved_by':'CPU fixture','ground_truth_format':'palette_object_id_png','monitor_video_count':1,'videos':settings}
    protocol=freeze_protocol(root,Path(__file__).parents[1]/'manifests/lvosv2_train_v1.json',runtime,tmp_path/'p.json',videos=['1umdNtrE'])
    validate_protocol_snapshot(snapshot,protocol)
    wrong=copy.deepcopy(protocol); wrong['cases'][0]['prompt']['runtime_index']+=1
    with pytest.raises(Rejection,match='PROTOCOL_PROMPT_IDENTITY'): validate_protocol_snapshot(snapshot,wrong)
    wrong=copy.deepcopy(protocol); wrong['cases'][0]['expected_runtime_indices']=wrong['cases'][0]['expected_runtime_indices'][1:]
    with pytest.raises(Rejection,match='PROTOCOL_FROZEN_WINDOW'): validate_protocol_snapshot(snapshot,wrong)


def test_monitor_consumption_verifies_immutable_requests(worker_fixture):
    from vos_memory_inspector.lvos_contract import Events
    from vos_memory_inspector.lvos_evaluation import consume_monitor_results,EarlyStopping
    _,run,request,_,_=worker_fixture
    identity=json.loads((run/'config.json').read_text())['identity']
    request.write_bytes(request.read_bytes()+b' ')
    early=EarlyStopping()
    with pytest.raises(Rejection,match='REQUEST_NOT_READY'):
        consume_monitor_results(run,early,identity,Events(run,identity,'cpu_synthetic'))
    assert early.consumed==[] and early.bad_events==0


def test_worker_rejects_foreign_evaluator_code(worker_fixture,tmp_path):
    from vos_memory_inspector.lvos_jobs import run_worker
    root,run,_,_,_=worker_fixture
    def foreign(*args):
        result=fake_executor(*args); result['code_sha']='foreign_source'
        output=args[4]; write_json(output/'result.json',result)
        write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
        return result
    result=run_worker(run,root,tmp_path,dry_run=False,executor=foreign)
    assert result['status']=='FAIL' and not list((run/'evaluation/workers').rglob('verified.json'))


def test_cpu_driver_preserves_existing_temp(tmp_path):
    existing=tmp_path/'existing'; existing.mkdir(); sentinel=existing/'preserve.txt'
    sentinel.write_text('original')
    with pytest.raises(Rejection,match='TEST_TEMP_EXISTS'):
        cli.cpu_test(tmp_path/'report',basetemp=existing)
    assert sentinel.read_text()=='original'
