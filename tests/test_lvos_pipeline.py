"""CPU/synthetic LVOS logic gates. 실제 SAM checkpoint/GPU 품질·속도 증거가 아니다."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import copy
import json
import ast
import subprocess
import numpy as np
from PIL import Image
import pytest
import torch

from vos_memory_inspector.collection_contract import frozen_selection, memory_policy, SEMANTICS, jpeg_map, Rejection
from vos_memory_inspector.case_cache import write_case_cache
from vos_memory_inspector.training_data import read_manifest, save_manifest
from vos_memory_inspector.training_storage import write_json, sha256, source_sha256, content_hash, ExclusiveWriter
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.lvos_contract import model_lock, validate_model_lock, load_complete, save_complete, Events, verify_evidence
from vos_memory_inspector.lvos_snapshot import build_snapshot, record_loader, verify_snapshot
from vos_memory_inspector.lvos_training import LVOSConfig, train_snapshot, component_objective, optimizer_groups, load_model_checkpoint
from vos_memory_inspector.lvos_gates import loss_gate, connectivity_gate, roundtrip_gate, run_gates, authorize_training
from vos_memory_inspector import lvos_evaluation as ev
from vos_memory_inspector import lvos_training as training
from vos_memory_inspector.lvos_cli import main, launch_plan, parser

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def selected():
    prefix=ROOT/'manifests/lvosv2_train_v1'
    return frozen_selection(str(prefix)+'.json',str(prefix)+'_fit.json',str(prefix)+'_development.json')


def fixture_data(tmp_path):
    selection=selected(); source_path=tmp_path/'selection.json'; save_manifest(source_path,selection)
    wanted=['train:0ClBYzYm:obj1:switch1221','train:1umdNtrE:obj1:switch961']
    models={role:{'architecture':architecture,'version':'sam2.1','synthetic':True,
        'checkpoint_sha256':content_hash(['synthetic weights',role]),'config_sha256':content_hash(['synthetic config',role]),
        'upstream_commit':'2b90b9f5ceec907a1c18123530e92e794ad901a4'} for role,architecture in [('source','small'),('target','base_plus')]}
    requests=[]; settings=[]
    for case_id in wanted:
        raw=next(c for c in selection['cases'] if c['case_id']==case_id)
        video_dir=tmp_path/'rgb'/raw['video_id']; video_dir.mkdir(parents=True)
        gt_dir=tmp_path/'gt'/raw['video_id']; gt_dir.mkdir(parents=True)
        prompt_path=gt_dir/'000001.png'
        Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(prompt_path)
        for frame in range(1,raw['future_end_frame']+1,5):
            Image.fromarray(np.full((4,4,3),20,dtype=np.uint8)).save(video_dir/f'{frame:06d}.jpg')
        paths,frame_map=jpeg_map(video_dir)
        switch=next(r['runtime_index'] for r in frame_map if r['official_id']==raw['switch_frame'])
        for row in frame_map[switch+1:switch+65:5]:
            Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(gt_dir/f"{row['official_id']:06d}.png")
        case={'dataset':'LVOS v2','release':'v2','official_split':'train','video_id':raw['video_id'],
            'object_ids':[1],'switch_frame':switch,'pair_mode':'native_history','prompt_conditions':[{'frame_index':0,'object_id':1,
                'kind':'mask','sha256':sha256(prompt_path)}], 'preprocessing':{'image_size':4,'scope':'synthetic_cpu_fixture'},
            'video_sha256':content_hash(frame_map),'frame_map_sha256':content_hash(frame_map),'num_frames':len(frame_map),'case_id':case_id,
            'official_prompt_frame':1,'official_switch_frame':raw['switch_frame'],'paired_split':raw['paired_split'],
            'source_manifest_content_sha256':selection['source_manifest_content_sha256'],'object_semantics':SEMANTICS,
            'memory_policy':memory_policy('active_window_v1',2,2)}
        read={'num_maskmem':2,'max_obj_ptrs_in_encoder':2,'memory_temporal_stride_for_eval':1,'max_cond_frames_in_attn':-1,
            'only_obj_ptrs_in_the_past_for_eval':True,'use_obj_ptrs_in_encoder':True,'add_all_frames_to_correct_as_cond':False}
        gen={'schema_version':'cmmt.paired_generating.v1','synthetic':True,'case':case,'models':models,'frame_map':frame_map,
            'seed':7,'collection_mode':'state_only','collector_source_hashes':{'test_lvos_pipeline.py':sha256(__file__)},
            'effective_model_policy':{'source':read,'target':read},'preprocessing':case['preprocessing']}
        shape=(1,1,3); valid=torch.tensor([[[True,True,False]]]); frames=torch.tensor([[[0,switch,-1]]])
        def state(value):
            spatial=torch.full((*shape,64,64,64),value,dtype=torch.bfloat16); spatial[:,:,2]=float('nan')
            pointer=torch.full((*shape,256),value,dtype=torch.float32); pointer[:,:,2]=float('nan')
            return CanonicalState(spatial,pointer,torch.full((*shape,1),-10.),frames,torch.tensor([[[0,1,2]]]),
                torch.tensor([[[True,False,False]]]),valid,(1,),switch)
        path=tmp_path/'originals'/f'{len(requests)}.pt'
        write_case_cache(path,source_canonical=state(1.),target_canonical=state(2. if raw['paired_split']=='fit' else 3.),
            metadata={'switch_frame':switch,'cache_mode':'state_only','seed':7,'prompt_frame_index':0,'generating':gen})
        requests.append({'path':str(path),'case_id':case_id,'sha256':sha256(path),'expected_generating':gen})
        settings.append({'video_id':raw['video_id'],'video_dir':str(video_dir),'annotation_dir':str(gt_dir)})
    return source_path,requests,wanted,settings


@pytest.fixture
def fixture_snapshot(tmp_path):
    selection,requests,ids,settings=fixture_data(tmp_path)
    root=tmp_path/'snapshot'
    value=build_snapshot(requests,selection,root,trusted=True,stable_seconds=0,max_shard_mib=2,pilot_ids=ids)
    return root,value,settings,requests


def test_frozen_architecture_sources():
    # Frozen pin source is available in this repository's fetched model ref.
    revision='746ea3e7d84c366c2d7ac06159e90a1f684bca56'
    lock=json.loads((ROOT/'configs/lvos_reference_lock.json').read_text(encoding='utf-8'))
    path=ROOT/'src/vos_memory_inspector/transformer_translator.py'
    assert source_sha256(path)==lock['ported_model_sha256']
    assert source_sha256(path.with_name('frozen_tensor_api.py'))==lock['ported_tensor_api_sha256']
    assert lock['architecture_ast_unchanged'] and lock['model_revision']==revision
    assert model_lock()['config']['d_model']==64
    assert model_lock()['config']['num_layers']==2


def test_model_identity_dtype_autograd_padding():
    torch.manual_seed(7); m=validate_model_lock(model_lock('CPU synthetic test only'))
    x=torch.randn(1,1,2,64,64,64).bfloat16(); p=torch.randn(1,1,2,256)
    x[:,:,1]=float('nan'); p[:,:,1]=float('nan'); valid=torch.tensor([[[True,False]]])
    s,t=m.translate_tensors(x,p,valid)
    assert s.dtype==t.dtype==torch.float32 and torch.equal(s[:,:,0],x[:,:,0].float()) and torch.equal(t[:,:,0],p[:,:,0])
    assert torch.isnan(s[:,:,1]).all()
    optimizer=torch.optim.AdamW(optimizer_groups(m,1e-4)[0],lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(); s,t=m.translate_tensors(x,p,valid)
        loss=component_objective(s,t,torch.ones_like(s),torch.ones_like(t),valid,{'spatial':1.,'pointer':1.},LVOSConfig())[2]
        loss.backward(); optimizer.step()
    assert m.spatial.alpha.grad.abs().sum()>0 and m.pointer.fc2.weight.grad.abs().sum()>0
    assert m.spatial.blocks[0].attn.qkv.weight.grad.abs().sum()>0
    a,b=m.translate_handoff_tensors(x,p,valid); assert a.dtype==torch.bfloat16 and b.dtype==torch.float32
    assert not m.supports_position_sampling


def test_frozen_lock_changes_rejected():
    lock=model_lock();
    with pytest.raises(Rejection,match='MODEL_FREEZE_INPUT_MISSING'):
        validate_model_lock(lock)
    lock=model_lock('CPU synthetic test only'); lock['config']['d_model']=32
    lock['digest']=content_hash({k:v for k,v in lock.items() if k!='digest'})
    with pytest.raises(Rejection,match='FROZEN_ARCHITECTURE'):
        validate_model_lock(lock)


def test_loss_reference_partial_window_and_padding():
    assert loss_gate()['partial_window_records']==3
    x=torch.tensor([[[[1.]]],[[[float('nan')]]]])
    with pytest.raises(Rejection,match='NONFINITE_LOSS'):
        component_objective(x,torch.ones(2,3),torch.zeros_like(x),torch.ones(2,3),torch.ones(2,dtype=torch.bool),
                            {'spatial':1.,'pointer':1.},LVOSConfig())


def test_snapshot_stream_counts_absent_records_fit_only_stats(fixture_snapshot):
    root,snapshot,_,_=fixture_snapshot
    assert snapshot['state']=='pilot_ready' and snapshot['synthetic']
    assert snapshot['statistics']['fit']['valid_records']==2 and snapshot['statistics']['fit']['padded_records']==1
    assert snapshot['statistics']['development']['valid_records']==2
    loader=record_loader(root,'fit',workers=0,microbatch=1,pin_memory=False)
    batches=list(loader); assert len(batches)==2 and len(set(tuple(b.refs[0]) for b in batches))==2
    assert all(tuple(b.tensors['source_spatial'].shape)==(1,64,64,64) for b in batches)
    assert training.fit_rms(root,LVOSConfig(workers=0))=={'spatial':2.,'pointer':2.}
    assert next(iter(record_loader(root,'development',workers=0,microbatch=1,pin_memory=False))).tensors['target_pointer'].mean()==3
    assert not ({c['case']['video_id'] for c in snapshot['cases'] if c['split']=='fit'} &
                {c['case']['video_id'] for c in snapshot['cases'] if c['split']=='development'})
    source=ev.reconstruct_source(root,snapshot,0); assert source.valid_record_count()==2 and source.validity.shape[-1]==3


def test_multiworker_persistent_epoch_coverage(fixture_snapshot):
    root,snapshot,_,_=fixture_snapshot
    loader=record_loader(root,'fit',workers=2,microbatch=1,pin_memory=False)
    assert loader.persistent_workers and loader.prefetch_factor==2
    records=[]
    for epoch in range(2):
        loader.dataset.set_epoch(epoch); records.append([tuple(b.refs[0]) for b in loader])
    assert records[0]==records[1] and len(records[0])==2 and [r[-1] for r in records[0]]==[0,1]


def test_pin_memory_custom_batch(monkeypatch):
    from vos_memory_inspector.lvos_snapshot import RecordBatch
    called=[]
    monkeypatch.setattr(torch.Tensor,'pin_memory',lambda self:(called.append(self.shape),self)[1])
    RecordBatch({'source_spatial':torch.zeros(1,64,64,64),'source_pointer':torch.zeros(1,256)},[[0,0,0,0]]).pin_memory()
    assert len(called)==2


def test_snapshot_checksum_completion_and_original_invalidation(fixture_snapshot):
    root,snapshot,_,requests=fixture_snapshot
    marker=Path(str(root/snapshot['shards'][0]['path'])+'.complete.json'); marker.write_text('{}',encoding='utf-8')
    with pytest.raises(Rejection,match='VIEW_COMPLETION_CHANGED'):
        verify_snapshot(root)
    write_json(marker,snapshot['shards'][0])
    with Path(requests[0]['path']).open('ab') as f:
        f.write(b'x')
    with pytest.raises(Rejection,match='ORIGINAL_CHANGED'):
        verify_snapshot(root,hashes=False)


def test_incomplete_and_misaligned_cache_audit_preserves_source(tmp_path):
    selection,requests,ids,_=fixture_data(tmp_path)
    before=[sha256(r['path']) for r in requests]
    requests[1]['expected_generating']=copy.deepcopy(requests[1]['expected_generating'])
    requests[1]['expected_generating']['case']['switch_frame']+=1
    with pytest.raises(Rejection,match='AUDIT_INCOMPLETE'):
        build_snapshot(requests,selection,tmp_path/'bad',trusted=True,stable_seconds=0,pilot_ids=ids)
    report=json.loads((tmp_path/'bad/audit.json').read_text(encoding='utf-8'))
    assert report['status']=='FAIL' and report['splits']['development']['unknown']==1
    assert before==[sha256(r['path']) for r in requests] and not (tmp_path/'bad/snapshot.json').exists()


def test_partial_selection_and_split_change_rejected(tmp_path):
    selection,requests,ids,_=fixture_data(tmp_path); value=read_manifest(selection)
    value['cases']=[c for c in value['cases'] if c['case_id'] in ids]; save_manifest(selection,value)
    with pytest.raises(Rejection,match='PARTIAL_SELECTION_REQUIRES_PILOT'):
        build_snapshot(requests,selection,tmp_path/'bad',trusted=True,stable_seconds=0)
    value['development_membership']['videos'].append(value['fit_membership']['videos'][0]); save_manifest(selection,value)
    with pytest.raises(Rejection,match='FROZEN_MEMBERSHIP_REVISION'):
        build_snapshot(requests,selection,tmp_path/'bad2',trusted=True,stable_seconds=0,pilot_ids=ids)


def test_connectivity_and_checkpoint_resume_gate_cpu_synthetic(fixture_snapshot,tmp_path):
    root,_,_,_=fixture_snapshot; lock=model_lock('CPU synthetic test only')
    connectivity_gate(root,lock,tmp_path/'G2.json','cpu')
    assert json.loads((tmp_path/'G2.json').read_text())['synthetic']
    roundtrip_gate(root,lock,tmp_path/'G4','cpu')
    assert json.loads((tmp_path/'G4/roundtrip.json').read_text())['weights_output_optimizer_scheduler_rng_scales_parity']


def test_actual_train_partial_restart_epoch_resume(fixture_snapshot,tmp_path,monkeypatch):
    root,_,_,_=fixture_snapshot; lock=model_lock('CPU synthetic test only')
    config=LVOSConfig(microbatch=1,accumulation=3,max_epochs=2,workers=0,pin_memory=False)
    expected=tmp_path/'continuous'; interrupted=tmp_path/'interrupted'
    train_snapshot(root,lock,expected,config,device='cpu')
    original_save=training.CheckpointStore.publish
    def interrupted_save(store,payload):
        if payload['epoch']==2:
            raise RuntimeError('Synthetic interrupted before epoch-2 commit')
        return original_save(store,payload)
    monkeypatch.setattr(training.CheckpointStore,'publish',interrupted_save)
    with pytest.raises(RuntimeError,match='Synthetic interrupted'):
        train_snapshot(root,lock,interrupted,config,device='cpu')
    assert not (interrupted/'checkpoints/epoch-00002.ckpt').exists()
    monkeypatch.setattr(training.CheckpointStore,'publish',original_save)
    train_snapshot(root,lock,interrupted,config,device='cpu',resume=interrupted/'checkpoints/epoch-00001.ckpt')
    a,_=load_complete(expected/'checkpoints/epoch-00002.ckpt'); b,_=load_complete(interrupted/'checkpoints/epoch-00002.ckpt')
    assert all(torch.equal(v,b['export']['state_dict'][n]) for n,v in a['export']['state_dict'].items())
    assert a['optimizer_step']==b['optimizer_step']==2 and a['history']==b['history']
    assert not (expected/'best_model.pth').exists()
    events=[json.loads(s) for s in (expected/'logs/events.jsonl').read_text(encoding='utf-8').splitlines()]
    required={'schema_version','timestamp_utc','run_id','task_id','execution_kind','code_sha','model_config_digest','collection_digest',
              'epoch','micro_iteration','optimizer_step','checkpoint_sha','eval_id','metrics','evidence_paths','duration_seconds'}
    assert all(required<=e.keys() for e in events) and all(e['execution_kind']=='cpu_synthetic' for e in events)
    with pytest.raises(Rejection,match='FOREIGN_CHECKPOINT'):
        train_snapshot(root,lock,interrupted,replace(config,lr=1e-4),device='cpu',resume=interrupted/'checkpoints/epoch-00002.ckpt')


def test_partial_epoch_is_stopped_not_completed(fixture_snapshot,tmp_path):
    root,_,_,_=fixture_snapshot; out=tmp_path/'partial'
    train_snapshot(root,model_lock('CPU synthetic test only'),out,LVOSConfig(microbatch=1,accumulation=3,max_epochs=2,
        workers=0,pin_memory=False,max_micro_iterations=1),device='cpu')
    assert (out/'STOPPED.json').exists() and not (out/'COMPLETED.json').exists()
    assert json.loads((out/'last.ckpt.json').read_text())['path']=='checkpoints/epoch-00000.ckpt'


def test_corruption_and_incomplete_checkpoints(tmp_path):
    path=tmp_path/'x.ckpt'; torch.save({'x':torch.ones(2)},path)
    with pytest.raises(Rejection,match='INCOMPLETE_ARTIFACT'):
        load_complete(path)
    path.unlink(); save_complete(path,{'x':torch.ones(2)})
    with path.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError):
        load_complete(path)


def test_decay_groups():
    m=validate_model_lock(model_lock('CPU synthetic test only')); _,names=optimizer_groups(m,1e-4)
    assert 'spatial.alpha' in names['no_decay'] and 'spatial.pos_embed' in names['no_decay']
    assert all(n in names['no_decay'] for n,p in m.named_parameters() if n.endswith('.bias'))


def test_monitor_sparse_protocol_and_unchanged_selection(fixture_snapshot,tmp_path):
    root,snapshot,settings,_=fixture_snapshot
    runtime={'approved_by':'CPU synthetic fixture only','ground_truth_format':'palette_object_id_png','monitor_video_count':1,
             'videos':settings,'sam2_repo':'UNKNOWN','target_checkpoint':'UNKNOWN','target_config':'UNKNOWN'}
    path=tmp_path/'monitor.json'
    protocol=ev.freeze_protocol(root,ROOT/'manifests/lvosv2_train_v1.json',runtime,path,videos=['1umdNtrE'])
    assert protocol['cases'][0]['fraction']==0.5 and len(protocol['cases'][0]['expected_runtime_indices'])==64
    assert protocol['cases'][0]['scored_frames'][0]['official_id']!=protocol['cases'][0]['scored_frames'][0]['runtime_index']
    assert sha256(root/'snapshot.json')
    with pytest.raises(Rejection,match='FULL_DEV_REQUIRES_FULL_SNAPSHOT'):
        ev.freeze_protocol(root,ROOT/'manifests/lvosv2_train_v1.json',runtime,tmp_path/'full.json',scope='full_development')
    for p in Path(settings[1]['annotation_dir']).glob('*.png'):
        if p.name!='000001.png': p.unlink()
    with pytest.raises(Rejection,match='INSUFFICIENT_SCORED_FRAMES'):
        ev.freeze_protocol(root,ROOT/'manifests/lvosv2_train_v1.json',runtime,tmp_path/'empty.json',videos=['1umdNtrE'])


def evaluation_fixture(tmp_path, *, scope='monitor', epochs=(2,)):
    rows=[{'case_id':f'{v}:{f}','video_id':v,'fraction':f,'expected_runtime_indices':[5,6],
           'scored_frames':[{'visible':True}]} for v in ('a','b') for f in ((0.5,) if scope=='monitor' else (0.25,0.5,0.75))]
    protocol={'scope':scope,'collection_digest':'fixture_data','cases':rows,'expected_case_ids':[r['case_id'] for r in rows],
              'synthetic':True}; p=tmp_path/'protocol.json'; save_manifest(p,protocol); protocol=read_manifest(p)
    contract={'approved_by':'CPU synthetic fixture','export_approved_by':'CPU synthetic fixture',
        'export_loader':'vos_memory_inspector.transformer_translator:TransformerStateTranslator.from_payload'}
    c=tmp_path/'metric.json'; write_json(c,contract); paths=[]
    for epoch in epochs:
        output=tmp_path/f'e{epoch}'; output.mkdir()
        result={'status':'PASS','scope':scope,'collection_digest':'fixture_data','model_config_digest':'fixture_model',
            'protocol_digest':protocol['content_sha256'],'metric_contract_digest':content_hash(contract),'checkpoint_sha':f'sha{epoch}',
            'checkpoint':f'NOT_REAL_CKPT{epoch}','epoch':epoch,'method':'learned','execution_kind':'cpu_synthetic','shard_count':1,
            'shard_index':0,'expected_case_ids':protocol['expected_case_ids'],'rows':[{'case_id':r['case_id'],'video':r['video_id'],
                'fraction':r['fraction'],'jf':0.5,'j':0.5,'f':0.5,'visible_frames':1,'absent_frames':0} for r in rows]}
        write_json(output/'result.json',result); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')}); paths.append(output/'result.json')
    return p,c,paths


def test_eval_video_partition_merge_coverage_and_no_worker_mean(tmp_path,monkeypatch):
    p,c,paths=evaluation_fixture(tmp_path)
    monkeypatch.setattr(ev,'benchmark_module',lambda *args:None)
    protocol=read_manifest(p); assignment=ev.case_assignments(protocol,2)
    assert set(assignment[0]).isdisjoint(assignment[1]) and set(sum(assignment,[]))==set(protocol['expected_case_ids'])
    original=json.loads(paths[0].read_text()); shard_paths=[]
    for i in range(2):
        out=tmp_path/f'shard{i}'; out.mkdir(); r=copy.deepcopy(original)
        r.update(shard_count=2,shard_index=i,expected_case_ids=assignment[i]); r['rows']=[x for x in r['rows'] if x['case_id'] in assignment[i]]
        r['rows'][0]['jf']=0.2 if i==0 else 0.8
        r['rows'][0]['j']=r['rows'][0]['f']=r['rows'][0]['jf']
        write_json(out/'result.json',r); write_json(out/'READY.json',{'sha256':sha256(out/'result.json')}); shard_paths.append(out/'result.json')
    merged=ev.merge_results(shard_paths,p,c,tmp_path,tmp_path/'merged')
    assert merged['monitor_score']==0.5 and merged['primary_score'] is None
    with pytest.raises(Rejection,match='EVAL_MISSING_SHARD'):
        ev.merge_results(shard_paths[:1],p,c,tmp_path,tmp_path/'missing')
    with pytest.raises(Rejection,match='EVAL_MISSING_SHARD'):
        ev.merge_results([shard_paths[0],shard_paths[0]],p,c,tmp_path,tmp_path/'duplicate')


def test_early_stopping_order_once_rawmax_and_delta():
    e=ev.EarlyStopping(min_epochs=6,patience=2,min_delta=.001)
    e.accept(2,.5); e.accept(4,.5005); assert e.raw_max==.5005 and e.stopping_reference==.5 and e.bad_events==0
    e.accept(6,.5008); e.accept(8,.5009); assert e.should_stop and e.raw_max==.5009
    with pytest.raises(Rejection,match='ASYNC_EVENT_ORDER'):
        e.accept(6,.8)


def test_async_incomplete_foreign_does_not_increment_patience(tmp_path,monkeypatch):
    p,c,paths=evaluation_fixture(tmp_path,epochs=(2,4)); identity={'run_id':'test','code_sha':'synthetic','collection_digest':'fixture_data',
        'model_config_digest':'fixture_model','metric_contract_digest':content_hash(json.loads(c.read_text()))}
    out=tmp_path/'run'; events=Events(out,identity,'cpu_synthetic'); early=ev.EarlyStopping()
    for epoch in (2,4):
        write_json(out/f'evaluation/requests/epoch-{epoch:05d}.json',{'epoch':epoch,'checkpoint':'synthetic','checkpoint_sha':f'sha{epoch}','protocol':str(p)})
    # Ordering unit fixture; actual immutable-request verification has integration coverage.
    import vos_memory_inspector.lvos_jobs as jobs
    monkeypatch.setattr(jobs,'read_request',lambda run,path:(json.loads(path.read_text()),read_manifest(p)))
    assert ev.consume_monitor_results(out,early,identity,events)==2 and not early.consumed
    # Later result cannot jump ahead of epoch 2.
    for epoch,path in zip((2,4),paths):
        dest=out/f'evaluation/results/epoch-{epoch:05d}'; dest.mkdir(parents=True)
        r=json.loads(path.read_text()); r.update(merged=True,monitor_score=.6,code_sha=identity['code_sha'])
        if epoch==2:
            r['collection_digest']='foreign'
        write_json(dest/'result.json',r); write_json(dest/'READY.json',{'sha256':sha256(dest/'result.json')})
    assert ev.consume_monitor_results(out,early,identity,events)==2 and not early.consumed and early.bad_events==0
    r=json.loads((out/'evaluation/results/epoch-00002/result.json').read_text()); r['collection_digest']='fixture_data'
    write_json(out/'evaluation/results/epoch-00002/result.json',r)
    write_json(out/'evaluation/results/epoch-00002/READY.json',{'sha256':sha256(out/'evaluation/results/epoch-00002/result.json')})
    assert ev.consume_monitor_results(out,early,identity,events)==0 and early.consumed==[2,4]
    assert ev.consume_monitor_results(out,early,identity,events)==0 and early.consumed==[2,4]


def test_best_cannot_promote_monitor_or_synthetic(tmp_path):
    p,c,paths=evaluation_fixture(tmp_path)
    with pytest.raises(Rejection,match='BEST_REQUIRES_REAL_FULL_DEV'):
        ev.select_best(paths,p,c,tmp_path/'best',shortlist_path=tmp_path/'missing.json')
    assert not (tmp_path/'best/best_model.pth').exists()


def test_full_dev_candidate_tie_and_undefined_rejection():
    rows=[{'epoch':4,'primary_score':80.},{'epoch':2,'primary_score':80.0000005},{'epoch':6,'primary_score':79.}]
    assert ev.best_candidate(rows)['epoch']==2
    with pytest.raises(Rejection,match='NO_VALID_TRAINED_CANDIDATE'):
        ev.best_candidate([{'epoch':2,'primary_score':None}])
    with pytest.raises(Rejection):
        ev.best_candidate([{'epoch':0,'primary_score':100.}])


def test_target_config_hash_path_matches_actual_producer(tmp_path):
    from vos_memory_inspector.training_collection import CONFIGS
    assert ev.target_config_path(tmp_path,CONFIGS['target'])==tmp_path/'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
    with pytest.raises(Rejection,match='PINNED_TARGET_CONFIG'):
        ev.target_config_path(tmp_path,'/foreign/config.yaml')


def test_shared_volume_namespace_collision_across_pods(tmp_path):
    job={'job_id':'A','host':'pod1','physical_gpu_uuid':'GPU-TEST-A','output':'/run/shared','python':'python',
         'argv':['python','-m','vos_memory_inspector.lvos_cli','train','--snapshot','snapshot','--model-lock','lock','--config','cfg',
                 '--output','/run/shared','--device','cuda:0']}
    with pytest.raises(Rejection,match='DUPLICATE_DEVICE_OR_NAMESPACE'):
        launch_plan({'jobs':[job,{**job,'job_id':'B','host':'pod2','physical_gpu_uuid':'GPU-TEST-B'}]},tmp_path/'plan.json')


def test_real_shapes_pair_metadata_mismatch(tmp_path):
    _,requests,_,_=fixture_data(tmp_path)
    value=torch.load(requests[0]['path'],weights_only=False)
    source=value['source_canonical']; target=value['target_canonical']
    for field in ('frame_indices','slot_order','validity','is_conditioning'):
        broken=copy.deepcopy(target)
        current=getattr(broken,field)
        current[0,0,1]=not bool(current[0,0,1]) if current.dtype==torch.bool else current[0,0,1]+1
        with pytest.raises(ValueError):
            write_case_cache(tmp_path/f'bad-{field}.pt',source_canonical=source,target_canonical=broken,metadata=value['metadata'])
    with pytest.raises(ValueError):
        write_case_cache(tmp_path/'bad-object.pt',source_canonical=source,target_canonical=replace(target,object_ids=(2,)),metadata=value['metadata'])


def test_missing_cache_completion_stops_snapshot(tmp_path):
    selection,requests,ids,_=fixture_data(tmp_path)
    Path(str(requests[0]['path'])+'.complete.json').unlink()
    with pytest.raises(Rejection,match='AUDIT_INCOMPLETE'):
        build_snapshot(requests,selection,tmp_path/'bad',trusted=True,stable_seconds=0,pilot_ids=ids)
    audit=json.loads((tmp_path/'bad/audit.json').read_text(encoding='utf-8'))
    assert audit['results'][0]['reason_codes']==['COMPLETED_EVIDENCE_MISSING']
    assert not (tmp_path/'bad/snapshot.json').exists()


def test_full_dev_export_adapter_strict_roundtrip_without_gpu_claim(tmp_path):
    # Model-team export schema only; this does not call select_best or invent a GPU result.
    m=validate_model_lock(model_lock('CPU synthetic export fixture'))
    path=tmp_path/'model.pth'; save_complete(path,m.to_payload())
    from vos_memory_inspector.transformer_translator import TransformerStateTranslator
    restored=TransformerStateTranslator.from_payload(load_complete(path)[0])
    assert all(torch.equal(v,restored.state_dict()[n]) for n,v in m.state_dict().items())
    bad=m.to_payload(); del bad['config']['num_layers']
    with pytest.raises(ValueError,match='missing fields'):
        TransformerStateTranslator.from_payload(bad)


def test_full_dev_selection_incomplete_and_foreign_model_blocked(tmp_path):
    p,c,paths=evaluation_fixture(tmp_path,scope='full_development',epochs=(2,4))
    protocol=read_manifest(p); protocol['synthetic']=False; save_manifest(p,protocol)
    short=tmp_path/'short.json'; write_json(short,{'identity':{'collection_digest':'fixture_data','model_config_digest':'fixture_model',
        'metric_contract_digest':content_hash(json.loads(c.read_text()))},'candidates':[{'epoch':e,'checkpoint_sha':f'sha{e}'} for e in (2,4)]})
    with pytest.raises(Rejection,match='FULL_DEV_SHORTLIST_INCOMPLETE'):
        ev.select_best(paths[:1],p,c,tmp_path/'best',shortlist_path=short)
    with pytest.raises(Rejection,match='BEST_RESULT_CONTRACT'):
        ev.select_best(paths,p,c,tmp_path/'best',shortlist_path=short)
    assert not (tmp_path/'best/best_model.pth').exists()


def test_initialize_and_cpu_cli_help(fixture_snapshot,tmp_path):
    root,_,_,_=fixture_snapshot
    output=tmp_path/'initialize'; lock_path=tmp_path/'lock.json'; cfg=tmp_path/'cfg.json'
    write_json(lock_path,model_lock('CPU synthetic init')); write_json(cfg,{'workers':0,'pin_memory':False})
    assert main(['initialize','--snapshot',str(root),'--model-lock',str(lock_path),'--config',str(cfg),'--output',str(output)])==0
    _,value,_=load_model_checkpoint(output/'checkpoints/epoch-00000.ckpt'); assert value['epoch']==0
    assert (output/'INITIALIZED.json').exists() and not (output/'metrics/train.json').exists()
    for cmd in parser()._subparsers._group_actions[0].choices:
        with pytest.raises(SystemExit) as exc:
            main([cmd,'--help'])
        assert exc.value.code==0


def test_namespace_locks_launcher_and_cli_failure(tmp_path):
    root=tmp_path/'run'
    with ExclusiveWriter(root):
        with pytest.raises(FileExistsError):
            with ExclusiveWriter(root):
                pass
    assert main(['train','--snapshot',str(tmp_path),'--model-lock','missing','--config','missing','--output',str(root),
                 '--device','cuda:0'])==2
    job={'job_id':'A','host':'synthetic_host','physical_gpu_uuid':'GPU-TEST','output':'/run/A','python':'python',
         'argv':['python','-m','vos_memory_inspector.lvos_cli','train','--snapshot','snapshot','--model-lock','lock','--config','cfg',
                 '--output','/run/A','--device','cuda:0']}
    plan=launch_plan({'jobs':[job]},tmp_path/'plan.json'); assert plan['dry_run'] and not plan['execution_performed']
    with pytest.raises(Rejection,match='DUPLICATE_DEVICE_OR_NAMESPACE'):
        launch_plan({'jobs':[job,{**job,'job_id':'B','output':'/run/B','argv':job['argv'][:-3]+['/run/B','--device','cuda:0']}]},tmp_path/'bad.json')


def test_gates_require_actual_evidence_and_do_not_claim_gpu(tmp_path):
    report=run_gates(tmp_path/'gates')
    g={v['gate_id']:v for v in report['gates']}
    assert g['G3']['status']=='PASS' and g['G3']['execution_kind']=='cpu_synthetic'
    verify_evidence(g['G3']['evidence'])
    assert all(g[k]['status']=='BLOCKED' for k in ('G0','G1','G2','G4','G5','G6'))
    with pytest.raises(Rejection):
        authorize_training(tmp_path/'gates/gates.json','x','y')


def test_cpu_fake_predictor_no_replay_frame_ids(fixture_snapshot,monkeypatch):
    root,snapshot,_,_=fixture_snapshot; source=ev.reconstruct_source(root,snapshot,0)
    import vos_memory_inspector.sam2_state as sam_state
    fresh={'device':torch.device('cpu'),'storage_device':torch.device('cpu'),'constants':{},'obj_ids':[],
        'output_dict_per_obj':{},'temp_output_dict_per_obj':{},'num_frames':source.switch_frame+3}
    monkeypatch.setattr(sam_state,'init_sam2_inference_state_without_warmup',lambda *a,**kw:copy.deepcopy(fresh))
    class Predictor:
        device=torch.device('cpu')
        memory_encoder=SimpleNamespace(position_encoding=lambda x:torch.zeros_like(x))
        def forward_image(self,image):
            return image
        def _get_image_feature(self,state,frame_idx,batch_size):
            return self.forward_image(torch.ones(1,3,4,4))
        def propagate_in_video(self,state,start_frame_idx,max_frame_num_to_track,reverse):
            for i in range(start_frame_idx,start_frame_idx+max_frame_num_to_track+1):
                self._get_image_feature(state,i,1)
                yield i,[1],torch.ones(1,1,4,4)
    row={'video_dir':'synthetic','expected_runtime_indices':[source.switch_frame+1,source.switch_frame+2],
         'scored_frames':[{'runtime_index':source.switch_frame+1}]}
    pred,trace=ev.no_replay_case(Predictor(),source,validate_model_lock(model_lock('CPU synthetic only')),row)
    assert trace['encoder_frame_ids']==row['expected_runtime_indices'] and trace['past_encoder_call_count']==0
    assert trace['metadata_preserved'] and list(pred)==[source.switch_frame+1]
