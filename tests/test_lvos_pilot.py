"""Pilot isolation and official dependencies: CPU mocks/fixtures, never real GPU."""
from pathlib import Path
import pytest
import torch
from vos_memory_inspector import lvos_gates as gates
from vos_memory_inspector.lvos_contract import model_lock
from vos_memory_inspector.training_storage import write_json
from vos_memory_inspector.training_storage import sha256, content_hash
from vos_memory_inspector.training_runner import code_provenance
from vos_memory_inspector.lvos_pilot import pilot_preflight, equal_state
from vos_memory_inspector.lvos_contract import load_complete, json_read
from vos_memory_inspector.lvos_training import train_snapshot, LVOSConfig
from vos_memory_inspector.lvos_snapshot import build_snapshot
from vos_memory_inspector.collection_contract import Rejection
from test_lvos_pipeline import fixture_data, fixture_snapshot


def mock_official(tmp_path, monkeypatch, *, state='pilot_ready', g0='pass', g1='pass'):
    root = tmp_path / 'input'
    snapshot = {'synthetic': False, 'state': state, 'content_sha256': 'mock-digest'}
    write_json(root / 'snapshot.json', snapshot)
    write_json(root / 'audit.json', {})
    metric = tmp_path / 'metric.json'
    write_json(metric, {'approved_by': 'CPU mock only'})
    lock = model_lock('CPU mock only') if g0 != 'blocked' else None
    calls = []
    def validate(lock):
        if g0 == 'fail':
            raise ValueError('MOCK_G0_FAILURE')
    def verify(*args, **kwargs):
        if g1 == 'fail':
            raise ValueError('MOCK_G1_FAILURE')
        return snapshot
    def connectivity(root, lock, output, device, deadline):
        calls.append('G2')
        write_json(output, {'mock': True})
    def roundtrip(root, lock, output, device, deadline):
        calls.append('G4')
        write_json(Path(output) / 'roundtrip.json', {'mock': True})
        (Path(output) / 'roundtrip.ckpt').write_bytes(b'CPU mock only')
    monkeypatch.setattr(gates, 'validate_model_lock', validate)
    monkeypatch.setattr(gates, 'verify_snapshot', verify)
    monkeypatch.setattr(gates, 'resources', lambda: {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(gates, 'connectivity_gate', connectivity)
    monkeypatch.setattr(gates, 'roundtrip_gate', roundtrip)
    report = gates.run_gates(tmp_path / 'gates', root=None if g1=='blocked' else root, lock=lock,
                            metric_contract=metric, device='cuda:0')
    return report, calls


def test_pilot_ready_cannot_trigger_official_gpu(tmp_path, monkeypatch):
    report, calls = mock_official(tmp_path, monkeypatch)
    records = {gate['gate_id']: gate for gate in report['gates']}
    assert records['G1']['status'] == 'FAIL'
    assert calls == []
    assert records['G2']['status'] == records['G4']['status'] == 'BLOCKED'
    assert report['unattended_training_ready'] is False


@pytest.mark.parametrize('g0,g1', [('blocked','pass'), ('fail','pass'), ('pass','fail'), ('pass','blocked')])
def test_failed_or_blocked_dependencies_never_call_gpu(tmp_path, monkeypatch, g0, g1):
    report, calls = mock_official(tmp_path, monkeypatch, state='ready', g0=g0, g1=g1)
    assert calls == []
    assert all(g['status'] == 'BLOCKED' for g in report['gates'] if g['gate_id'] in ('G2','G4'))


def test_valid_full_dependencies_retain_official_path(tmp_path, monkeypatch):
    report, calls = mock_official(tmp_path, monkeypatch, state='ready')
    assert calls == ['G2','G4']
    assert all(g['status'] == 'PASS' for g in report['gates'] if g['gate_id'] in ('G0','G1','G2','G4'))
    assert report['unattended_training_ready'] is False  # Missing G5/G6 evidence.


def test_pilot_real_required_and_cpu_synthetic_strict_resume(tmp_path):
    selection, requests, ids, _ = fixture_data(tmp_path)
    root=tmp_path/'one-fit'
    build_snapshot(requests[:1],selection,root,trusted=True,stable_seconds=0,pilot_ids=ids[:1])
    lock=model_lock()  # No fabricated model-team approval.
    with pytest.raises(Rejection,match='PILOT_REQUIRES_REAL_INPUT'):
        pilot_preflight(root,lock,tmp_path/'rejected',device='cpu')
    result=pilot_preflight(root,lock,tmp_path/'pilot',device='cpu',synthetic_test=True)
    assert result['status']=='CPU_SYNTHETIC_PILOT_PREFLIGHT_PASS' and result['scope']=='diagnostic_only'
    assert not result['unattended_training_ready'] and not result['state_training_ready']
    assert not result['model_formally_approved']
    assert result['next_update_difference']==0 and result['reload_output_difference']==[0,0]
    body,_=load_complete(tmp_path/'pilot/diagnostic.ckpt')
    assert body['step']==3 and body['normalization']['records']==2
    assert body['normalization']['computations']==1
    with pytest.raises(Rejection,match='DIAGNOSTIC_ARTIFACT_NOT_TRAINING_AUTHORIZATION'):
        gates.authorize_training(tmp_path/'pilot/STATUS.json','x','y')


def test_pilot_dev_or_multiple_cases_rejected(fixture_snapshot,tmp_path):
    root,_,_,_=fixture_snapshot
    with pytest.raises(Rejection,match='PILOT_ONE_CASE_ONLY'):
        pilot_preflight(root,model_lock(),tmp_path/'pilot',device='cpu',synthetic_test=True)


def test_single_development_pilot_rejected(tmp_path):
    selection,requests,ids,_=fixture_data(tmp_path)
    root=tmp_path/'one-dev'
    build_snapshot(requests[1:],selection,root,trusted=True,stable_seconds=0,pilot_ids=ids[1:])
    with pytest.raises(Rejection,match='PILOT_FIT_ONLY'):
        pilot_preflight(root,model_lock(),tmp_path/'pilot',device='cpu',synthetic_test=True)


def test_pilot_deadline_records_stopped_without_training(tmp_path):
    from vos_memory_inspector.lvos_budget import Deadline, BudgetStop
    deadline=Deadline(1,clock=lambda:2,started=0)
    with pytest.raises(BudgetStop):
        pilot_preflight(tmp_path/'missing',model_lock(),tmp_path/'stopped',device='cpu',deadline=deadline)
    status=json_read(tmp_path/'stopped/STATUS.json')
    assert status['status']=='STOPPED' and not status['unattended_training_ready']


def test_state_training_no_evaluation_and_epoch_exports(fixture_snapshot,tmp_path,monkeypatch):
    root,_,_,_=fixture_snapshot
    def forbidden(*args,**kwargs):
        raise AssertionError('State training must never invoke evaluation')
    from vos_memory_inspector import lvos_jobs
    monkeypatch.setattr(lvos_jobs,'publish_request',forbidden)
    out=tmp_path/'state-training'
    cfg=LVOSConfig(microbatch=1,accumulation=2,workers=0,max_epochs=2,pin_memory=False)
    rows=train_snapshot(root,model_lock('CPU synthetic only'),out,cfg,device='cpu',training_scope='state_supervised')
    assert len(rows)==2 and rows[-1]['dev']['records']==2
    assert {'normalized_spatial','normalized_pointer','spatial_mse','pointer_mse'}<=rows[-1]['dev'].keys()
    status=json_read(out/'STATUS.json'); assert status['evaluation_status']=='NOT_REQUESTED'
    assert not status['JF_early_stopping_applied'] and not status['vos_best_model_selected']
    assert not status['unattended_training_ready'] and (out/'best_state_loss.ckpt.json').is_file()
    assert not (out/'best_model.pth').exists()
    for epoch in range(3):
        body,_=load_complete(out/f'checkpoints/epoch-{epoch:05d}.ckpt')
        exported,_=load_complete(out/f'translator/epoch-{epoch:05d}.pth')
        assert equal_state(body['export'],exported)
    assert json_read(out/'translator/normalization.json')['scales']['spatial']==2
    assert 'dev_normalized_pointer' in (out/'metrics/train.csv').read_text(encoding='utf-8')
    with pytest.raises(Rejection,match='STATE_TRAINING_HAS_NO_JF_EVALUATION'):
        train_snapshot(root,model_lock('CPU synthetic only'),tmp_path/'bad',cfg,device='cpu',
                       training_scope='state_supervised',monitor_protocol='unused')


def mocked_training_report(tmp_path):
    proof=tmp_path/'proof.json'; write_json(proof,{'cpu_mock_only':True})
    ev=[{'path':str(proof),'sha256':sha256(proof)}]
    return {'schema_version':'cmmt.lvos_gates.v1','identity':{'collection_digest':'data',
        'model_config_digest':'model','code_sha':code_provenance()['package_source_sha256']},
        'gates':[{'gate_id':name,'status':'BLOCKED' if name=='G5' else 'PASS',
                  'execution_kind':'gpu_real_checkpoint' if name in ('G2','G4','G5') else
                    'cpu_real_cache' if name=='G1' else 'cpu_synthetic',
                  'evidence':[] if name=='G5' else ev} for name in ('G0','G1','G2','G3','G4','G5','G6')]},proof


def test_state_authorization_is_independent_from_full_research(tmp_path):
    report,_=mocked_training_report(tmp_path); path=tmp_path/'gates.json'; write_json(path,report)
    gates.authorize_state_training(path,'data','model')  # Pure CPU authorization logic mock.
    with pytest.raises(Rejection,match='GATE_NOT_PASSED'):
        gates.authorize_training(path,'data','model')
    report['scope']='diagnostic_only'; write_json(path,report)
    with pytest.raises(Rejection,match='DIAGNOSTIC_ARTIFACT_NOT_TRAINING_AUTHORIZATION'):
        gates.authorize_state_training(path,'data','model')


@pytest.mark.parametrize('change',['identity','data','model','checksum','synthetic','failed_G0','failed_G1'])
def test_state_authorization_preserves_required_guards(tmp_path,change):
    report,proof=mocked_training_report(tmp_path)
    if change=='identity': report['identity']['code_sha']='stale'
    if change=='data': report['identity']['collection_digest']='stale'
    if change=='model': report['identity']['model_config_digest']='stale'
    if change=='checksum': write_json(proof,{'changed':True})
    if change=='synthetic': report['gates'][2]['execution_kind']='cpu_synthetic'
    if change.startswith('failed_'):
        next(g for g in report['gates'] if g['gate_id']==change[7:])['status']='FAIL'
    path=tmp_path/'gates.json'; write_json(path,report)
    with pytest.raises(Rejection): gates.authorize_state_training(path,'data','model')
