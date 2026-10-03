"""호환 raw cache / DDP CPU fixture 검증. 실제 GPU 통과 증거가 아니다."""
from pathlib import Path
from dataclasses import replace
import json
import os
import subprocess
import sys
import pytest
import torch
torch.set_num_threads(1)

from vos_memory_inspector.lvos_raw_cache import frozen_cases, build_index, load_raw, read_index, records
from vos_memory_inspector.lvos_ddp import schedule
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.case_cache import write_case_cache
from vos_memory_inspector.collection_contract import Rejection
from vos_memory_inspector.training_storage import write_json


def fixture_raw(root, unequal=False, late=False, prompt_accept=False):
    cases,_ = frozen_cases()
    first = next(c for c in cases if c['paired_split']=='fit' and (not late or c['first_prompt_frame'] > 1))
    wanted = [first,
              next(c for c in cases if c['paired_split']=='fit' and c['video_id']!=first['video_id']),
              next(c for c in cases if c['paired_split']=='development')]
    for case_index,case in enumerate(wanted):
        video = root/'rgb'/case['video_id']; video.mkdir(parents=True)
        frame_ids = sorted({1,case['first_prompt_frame'],case['switch_frame'],case['future_end_frame']})
        runtime_switch = frame_ids.index(case['switch_frame'])
        for frame in frame_ids:
            (video/f'{frame:06d}.jpg').write_bytes(b'CPU fixture filename map only')
        shape = (1,1,3)
        def state(v):
            x = torch.full((*shape,64,64,64),v,dtype=torch.bfloat16); x[:,:,2] = float('nan')
            p = torch.full((*shape,256),v,dtype=torch.float32); p[:,:,2] = float('nan')
            valid = torch.tensor([[[True,not(unequal and case_index == 1),False]]])
            return CanonicalState(x,p,torch.zeros((*shape,1)),torch.tensor([[[0,1,-1]]]),
                       torch.tensor([[[0,1,2]]]),torch.tensor([[[True,False,False]]]),
                       valid,(int(case['object_id']),),runtime_switch)
        path = root/case['paired_split']/f"lvos_{case['video_id']}_obj{case['object_id']}_switch{case['switch_frame']}.pt"
        target_value = (4 if unequal and case_index == 1 else 2) if case['paired_split']=='fit' else 100
        write_case_cache(path,source_canonical=state(1),target_canonical=state(target_value),
                         metadata={'source_model_id':'sam2.1-small','target_model_id':'sam2.1-base-plus',
                         'video_id':case['video_id'],'object_id':case['object_id'],'switch_frame':runtime_switch,'num_frames':len(frame_ids),
                         'cache_mode':'state_only','active_memory_only':True,'num_maskmem':7,'max_obj_ptrs_in_encoder':16,
                         'synthetic':True})
        # 자체 fixture의 marker만 제거해 실제 legacy 입력 조건을 재현한다.
        Path(str(path)+'.complete.json').unlink()
    decision = {'schema_version':'cmmt.state_training_prompt_decision.v1','scope':'state_supervised_only',
                'approved_by':'CPU synthetic test only','reason':'explicit observed-history acceptance fixture',
                'recorded_at':'2026-10-03','case_ids':[first['case_id']]} if prompt_accept else None
    index = build_index(root/'fit',root/'development',root/'rgb',root/'index',operator_completed='CPU synthetic fixture',
                        case_ids=[c['case_id'] for c in wanted],stable_seconds=0,max_wall_seconds=120,prompt_decision=decision)
    return root/'index/raw_index.json', index


def test_raw_compat_no_completion_no_fake_provenance_fit_only(tmp_path):
    path,value = fixture_raw(tmp_path)
    assert value['normalization']['scales'] == {'spatial':2.,'pointer':2.}
    assert value['normalization']['computations'] == 1
    assert value['historical_provenance_status'] == 'UNKNOWN' and not value['research_gates_passed']
    assert all(e['historical_provenance']['unknown'] for e in value['entries'])
    assert not list((tmp_path/'fit').glob('*.complete.json'))
    rows = list(records(read_index(path)['entries']))
    assert len(rows) == 6 and all(torch.isfinite(r['source_spatial']).all() for r in rows)
    resumed = build_index(tmp_path/'fit',tmp_path/'development',tmp_path/'rgb',tmp_path/'index',
                          operator_completed='CPU synthetic fixture',case_ids=value['identity']['case_ids'],stable_seconds=0)
    assert resumed['normalization'] == value['normalization']


@pytest.mark.parametrize('world',[2,4])
def test_global_batch_unique_case_partition_variable_counts(world):
    entries = [{'case':{'case_id':str(i)},'records':n} for i,n in enumerate((16,9,5,11,16,16,3,16,16,16))]
    assigned, windows = schedule(entries,world,64,seed=7,epoch=0)
    ids = [(e['case']['case_id'],p) for group in assigned for e in group for p in e['selected_positions']]
    assert len(set(ids)) == sum(e['records'] for e in entries) == len(ids)
    assert all(sum(w) == 64 for w in windows[:-1])
    assert all(w == [64//world]*world for w in windows[:-1])
    assert sum(map(sum,windows)) == sum(e['records'] for e in entries)
    assert [sum(w[r] for w in windows) for r in range(world)] == [sum(e['records'] for e in group) for group in assigned]


def test_checksum_corruption_stops(tmp_path):
    _,index = fixture_raw(tmp_path); path = Path(index['entries'][0]['path'])
    with path.open('ab') as f: f.write(b'corrupt')
    with pytest.raises(Rejection,match='RAW_CHECKSUM'): load_raw(path)


def test_prompt_decision_requires_actual_approval_and_exact_cases(tmp_path,monkeypatch):
    path,index = fixture_raw(tmp_path)
    template = {'schema_version':'cmmt.state_training_prompt_decision.v1','scope':'state_supervised_only',
                'approved_by':None,'reason':'CPU fixture','recorded_at':'2026-10-03','case_ids':[]}
    with pytest.raises(Rejection,match='PROMPT_DECISION_APPROVAL_REQUIRED'):
        build_index(tmp_path/'fit',tmp_path/'development',tmp_path/'rgb',tmp_path/'rejected',
                    operator_completed='CPU fixture',case_ids=index['identity']['case_ids'],prompt_decision=template)
    template['approved_by'] = 'CPU synthetic test only'
    template['case_ids'] = [index['entries'][0]['case']['case_id']]
    with pytest.raises(Rejection,match='PROMPT_DECISION_EXACT_DEVIATIONS'):
        build_index(tmp_path/'fit',tmp_path/'development',tmp_path/'rgb',tmp_path/'unused-approval',
                    operator_completed='CPU fixture',case_ids=index['identity']['case_ids'],prompt_decision=template,stable_seconds=0)
    with pytest.raises(Rejection,match='CACHE_ALIGNMENT_FAILURES'):
        fixture_raw(tmp_path/'late-rejected',late=True)
    _,accepted = fixture_raw(tmp_path/'late-accepted',late=True,prompt_accept=True)
    deviation = next(e for e in accepted['entries'] if e['prompt_contract_status'] != 'MATCH')
    assert deviation['prompt_contract_status'] == 'ACCEPTED_EXISTING_HISTORY_FOR_STATE_TRAINING_ONLY'
    assert deviation['conditioning_frames_observed'] == [0]
    assert not accepted['research_gates_passed']
    from vos_memory_inspector.lvos_ddp import train
    from vos_memory_inspector.lvos_training import LVOSConfig
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(Rejection,match='DDP_PHYSICAL_GPU_UUIDS_REQUIRED'):
        train(path,tmp_path/'fake-gpu-rejected',config=LVOSConfig(accumulation=2,workers=0),device='cuda',
              gpu_uuids=['0','1'],execute_approved=True)


def test_exclusion_manifest_preserves_membership_and_blocks_unapproved_training(tmp_path,monkeypatch):
    from vos_memory_inspector import lvos_raw_cache as raw
    from vos_memory_inspector.lvos_ddp import validate_main_index
    _,original = fixture_raw(tmp_path)
    minimal = [e['case'] for e in original['entries']]
    monkeypatch.setattr(raw,'frozen_cases',lambda:(minimal,original['identity']['splits']))
    plan = {'schema_version':'cmmt.state_training_exclusions.v1','scope':'state_supervised_only',
            'approved_by':None,'recorded_at':None,'cases':[{'case_id':minimal[0]['case_id'],'reason':'CPU fixture pending prompt check'}]}
    index = build_index(tmp_path/'fit',tmp_path/'development',tmp_path/'rgb',tmp_path/'proposed',
                        operator_completed='CPU fixture',exclusion_plan=plan,stable_seconds=0)
    assert index['identity']['splits'] == original['identity']['splits']
    assert len(read_index(tmp_path/'proposed/raw_index.json')['entries']) == 2
    with pytest.raises(Rejection,match='EXCLUSION_PLAN_NOT_APPROVED'): validate_main_index(index)
    plan['approved_by']='CPU synthetic test only'; plan['recorded_at']='2026-10-03'
    index['identity']['exclusion_plan']=plan
    validate_main_index(index)


@pytest.mark.parametrize('world,unequal',[(2,False),(2,True),(4,False)])
def test_actual_two_rank_gloo_branches_reload_and_coverage(tmp_path,world,unequal):
    path,index = fixture_raw(tmp_path,unequal=unequal)
    cfg = tmp_path/'config.json'; write_json(cfg,{'microbatch':1,'accumulation':4//world,'workers':0,'max_epochs':3,'pin_memory':False})
    out = tmp_path/'ddp'
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--rdzv_conf=use_libuv=0','--nproc_per_node='+str(world),
               '-m','vos_memory_inspector.lvos_ddp','train','--index',str(path),'--output',str(out),
               '--config',str(cfg),'--global-batch','4','--device','cpu','--max-wall-seconds','240']
    if os.name == 'nt':
        # 현재 Windows Torch build의 TCPStore/libuv 조합 대신 동일 Gloo runner를 FileStore로 검증.
        script = tmp_path/'cpu_spawn.py'
        script.write_text('''import os, torch.multiprocessing as mp
from vos_memory_inspector.lvos_ddp import train
from vos_memory_inspector.lvos_training import LVOSConfig
def worker(rank):
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(WORLD),LVOS_CPU_STORE=STORE)
    train(INDEX,OUTPUT,config=LVOSConfig(microbatch=1,accumulation=4//WORLD,workers=0,max_epochs=3,pin_memory=False),global_batch=4,max_wall_seconds=240)
INDEX = '''+repr(str(path))+'''\nOUTPUT = '''+repr(str(out))+'''\nSTORE = '''+repr(str(tmp_path/'gloo_store'))+'''\nWORLD = '''+repr(world)+'''
if __name__ == '__main__': mp.spawn(worker,nprocs=WORLD,join=True)
''',encoding='utf-8')
        command = [sys.executable,str(script)]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',USE_LIBUV='0')
    result = subprocess.run(command,capture_output=True,text=True,env=env,timeout=270)
    (tmp_path/'torchrun.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    status = json.loads((out/'STATUS.json').read_text())
    assert status['execution_kind'] == 'CPU_SYNTHETIC' and status['world_size'] == world
    assert status['both_branches_updated'] and status['strict_reload_equal'] and status['optimizer_reload_equal']
    assert status['optimizer_step'] == 3 and status['rank_parameter_max_difference'] == 0
    coverage = json.loads((out/'assignments/epoch-00003-coverage.json').read_text())
    assert coverage['rank_records'] == ([2,1] if unequal else [4//world]*world)
    assert coverage['unique_records'] == (3 if unequal else 4)
    assert len(list((out/'translator').glob('epoch-*.pth'))) == 4  # only rank 0; epoch 0..3
    # 실제 runner의 첫 optimizer update와 모든 valid record를 한 번에 처리한 기준 비교.
    from vos_memory_inspector.lvos_contract import model_lock, validate_model_lock, load_complete
    from vos_memory_inspector.lvos_training import LVOSConfig, component_objective, optimizer_groups
    from vos_memory_inspector.lvos_snapshot import collate_records
    torch.manual_seed(7); reference = validate_model_lock(model_lock(),require_approval=False)
    config = LVOSConfig(microbatch=1,accumulation=4//world,workers=0,max_epochs=3,pin_memory=False)
    optimizer = torch.optim.AdamW(optimizer_groups(reference,config.weight_decay)[0],lr=config.lr)
    data = collate_records(list(records([e for e in index['entries'] if e['case']['paired_split']=='fit']))).tensors
    s,p = reference.translate_tensors(data['source_spatial'][None,None].float(),data['source_pointer'][None,None].float())
    loss = component_objective(s[0,0],p[0,0],data['target_spatial'],data['target_pointer'],
                    torch.ones(len(data['frame']),dtype=torch.bool),index['normalization']['scales'],config)[2]
    loss.backward(); torch.nn.utils.clip_grad_norm_(reference.parameters(),config.clip_norm); optimizer.step()
    first,_ = load_complete(out/'checkpoints/epoch-00001.ckpt')
    for name,v in reference.state_dict().items():
        torch.testing.assert_close(v,first['export']['state_dict'][name],rtol=1e-5,atol=2e-7)
    if not unequal and world == 2:
        # 같은 config/3 epoch plan 아래 epoch 1 경계 중단 후 실제 2-rank 재개.
        resumed = tmp_path/'resumed'
        if os.name == 'nt':
            contents = script.read_text().replace(repr(str(out)),repr(str(resumed)))
            contents = contents.replace(repr(str(tmp_path/'gloo_store')),repr(str(tmp_path/'gloo_pause')))
            contents = contents.replace('max_wall_seconds=240)', 'max_wall_seconds=240,stop_after_epoch=1)')
            script.write_text(contents,encoding='utf-8')
            paused = subprocess.run(command,capture_output=True,text=True,env=env,timeout=270)
            assert paused.returncode == 0, paused.stderr
            script.write_text(contents.replace('stop_after_epoch=1','resume=True').replace(
                repr(str(tmp_path/'gloo_pause')),repr(str(tmp_path/'gloo_resume'))),encoding='utf-8')
            continued = subprocess.run(command,capture_output=True,text=True,env=env,timeout=270)
        else:
            updated = list(command); updated[updated.index(str(out))] = str(resumed)
            paused = subprocess.run(updated+['--stop-after-epoch','1'],capture_output=True,text=True,env=env,timeout=270)
            assert paused.returncode == 0, paused.stderr
            continued = subprocess.run(updated+['--resume'],capture_output=True,text=True,env=env,timeout=270)
        (tmp_path/'resume.log').write_text(paused.stdout+paused.stderr+continued.stdout+continued.stderr,encoding='utf-8')
        assert continued.returncode == 0, continued.stdout+continued.stderr
        final,_ = load_complete(out/'checkpoints/epoch-00003.ckpt')
        resumed_final,_ = load_complete(resumed/'checkpoints/epoch-00003.ckpt')
        from vos_memory_inspector.lvos_pilot import equal_state
        for field in ('export','optimizer','scheduler','rank_rng','scales'):
            assert equal_state(final[field],resumed_final[field]), field
