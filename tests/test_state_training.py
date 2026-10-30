"""CPU coverage of data integrity, batch optimization, initialization and resume."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from vos_memory_inspector.state_training.data import (
    load_index, load_pair, RecordDataset, record_loader,
)
from vos_memory_inspector.state_training.policy import TrainConfig, schedule
from vos_memory_inspector.state_training.preflight import inspect, require_ready
from vos_memory_inspector.state_training.ridge import (
    fit_ridge, load_ridge, StreamingRidgeTranslator,
)
from vos_memory_inspector.state_training.statistics import Moments, solve
from vos_memory_inspector.state_training.storage import save_json, sha256
from vos_memory_inspector.state_training.trainer import train, construct_model, restore_model
from vos_memory_inspector.state_training.usage import MemoryPolicy, record_usage, require_selective_handoff_verified
from vos_memory_inspector.translator_artifact import load_learned_translator


OPTIONS = dict(d_model=4, num_heads=2, num_layers=1, patch_size=2,
               pointer_hidden_dim=10, fusion_hidden_dim=8)


@pytest.fixture
def sample_index(tmp_path, make_state):
    torch.manual_seed(44)
    entries = []
    (tmp_path / 'cache').mkdir()
    for i, (role, records) in enumerate((('train', 3), ('train', 4), ('validation', 3))):
        s = make_state(torch.randn(1, 1, records, 4, 4, 4).bfloat16(), torch.randn(1, 1, records, 6))
        s.presence_logits[..., 0] = torch.tensor([1. if k % 2 else -1. for k in range(records)])
        t = s.with_continuous(spatial_memory=(s.spatial_memory.float() * 0.6 + 0.2).bfloat16(),
                              object_pointer=s.object_pointer * 0.7 - 0.1, presence_logits=s.presence_logits.clone())
        # File frame deliberately differs from canonical runtime frame.
        path = tmp_path / 'cache' / f'video{i}_obj0_switch900.pt'
        torch.save(dict(schema_version='cmmt.prepared_handoff_case.v2', source_canonical=s,
            target_canonical=t, metadata=dict(cache_mode='state_only', source_model_id='sam2.1-small',
            target_model_id='sam2.1-base-plus', video_id=f'video{i}', object_id=0, num_frames=records + 2)), path)
        checksum = sha256(path)
        Path(str(path) + '.sha256').write_text(checksum + '  ' + path.name)
        entries.append(dict(path=str(path), sha256=checksum, valid_indices=torch.nonzero(s.validity).tolist(),
            runtime_switch_frame=s.switch_frame, records=records,
            case=dict(case_id=f'case{i}', dataset='LVOS v2', training_role=role, video_id=f'video{i}',
                      object_id=0, switch_frame=900, official_split='train' if role == 'train' else 'valid')))
    path = tmp_path / 'sample-index.json'
    save_json(path, dict(schema_version='cmmt.state_training_sample_index.v1', entries=entries))
    return path


@pytest.fixture
def cpu_config():
    return TrainConfig(workers=0, global_batch=6, microbatch=2, max_epochs=2, cache_bytes=4096, clip_norm=100.)


def test_index_mapping_integrity_and_missing_readiness(sample_index, cpu_config, tmp_path):
    with pytest.raises(ValueError, match='official training index'):
        load_index(sample_index)
    index = load_index(sample_index, allow_sample=True)
    entry = index.entries[0]
    pair, _ = load_pair(entry)
    assert pair['source_canonical'].switch_frame != entry['case']['switch_frame']
    report = inspect(index, cpu_config)
    assert report['sample_data_ready'] and not report['full_data_ready']
    require_ready(report, allow_sample=True)
    with pytest.raises(ValueError, match='not ready'):
        require_ready(report)
    report = inspect(index, cpu_config, sample_cases=1)
    assert not report['sample_data_ready']
    relocated = load_index(sample_index, {str(tmp_path): str(tmp_path / 'not-here')}, allow_sample=True)
    assert relocated.identity == index.identity
    assert relocated.entries[0]['case'] == entry['case']
    assert len(inspect(relocated, cpu_config)['missing']) == 3
    data = json.loads(sample_index.read_text())
    data['entries'][2]['case']['video_id'] = data['entries'][0]['case']['video_id']
    save_json(tmp_path / 'overlap.json', data)
    with pytest.raises(ValueError, match='overlap'):
        load_index(tmp_path / 'overlap.json', allow_sample=True)


def test_cache_errors_and_bounded_record_loading(sample_index, cpu_config):
    index = load_index(sample_index, allow_sample=True)
    dataset = RecordDataset(index.entries, cache_cases=1, cache_bytes=cpu_config.cache_bytes)
    rows = dataset.__getitems__([0, 1, 3, 4, 7])
    assert len(rows) == 5
    assert dataset.bytes <= cpu_config.cache_bytes and len(dataset.cache) <= 1
    assert all(row['source_spatial'].shape == (4, 4, 4) for row in rows)
    entry = index.entries[0]
    bad = dict(entry, runtime_switch_frame=100)
    with pytest.raises(ValueError, match='runtime switch'):
        load_pair(bad)
    path = Path(entry['path'])
    payload = torch.load(path, weights_only=False)
    payload['target_canonical'].slot_order[..., 1] = 9
    torch.save(payload, path)
    with pytest.raises(ValueError, match='checksum'):
        load_pair(entry)
    entry = dict(entry, sha256=sha256(path))
    Path(str(path) + '.sha256').write_text(entry['sha256'])
    with pytest.raises(ValueError, match='slot_order'):
        load_pair(entry)


def test_train_only_rms_and_presence_strata(sample_index, cpu_config):
    index = load_index(sample_index, allow_sample=True)
    before = inspect(index, cpu_config)['normalization']['scales']
    entry = index.role('validation')[0]
    value, _ = load_pair(entry)
    value['target_canonical'].object_pointer.mul_(1000)
    torch.save(value, entry['path'])
    entry['sha256'] = sha256(entry['path'])
    Path(entry['path'] + '.sha256').write_text(entry['sha256'])
    after = inspect(index, cpu_config)['normalization']['scales']
    assert before == after


def test_schedule_tail_has_exactly_once_coverage(sample_index):
    index = load_index(sample_index, allow_sample=True)
    assigned, windows = schedule(index.role('train'), 2, 6)
    assert windows[-1] == [1, 0]
    observed = [(e['case']['case_id'], p) for rank in assigned for e in rank for p in e['selected_positions']]
    expected = [(e['case']['case_id'], p) for e in index.role('train') for p in range(e['records'])]
    assert sorted(observed) == sorted(expected)


def test_streaming_ridge_equals_direct_centered_solution():
    torch.manual_seed(8)
    x = torch.randn(31, 5, dtype=torch.float64)
    y = torch.randn(31, 3, dtype=torch.float64)
    moments = Moments(5, 3)
    for a, b in zip(x.split(7), y.split(7)):
        moments.update(a, b)
    for alpha in (0., 1e-4, 1.):
        affine = solve(moments, alpha)
        xc, yc = x - x.mean(0), y - y.mean(0)
        gram = xc.T @ xc / len(x)
        penalty = alpha * gram.trace() / 5
        expected = torch.linalg.solve(gram + penalty * torch.eye(5, dtype=torch.float64), xc.T @ yc / len(x))
        torch.testing.assert_close(affine.weight, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(affine.bias, y.mean(0) - x.mean(0) @ expected)


@pytest.mark.parametrize('name', ['residual_mlp', 'mlp_context'])
def test_affine_initialization_gradients_export_and_artifact_binding(sample_index, cpu_config, tmp_path, name):
    index = load_index(sample_index, allow_sample=True)
    ridge_path = tmp_path / 'ridge.pth'
    ridge = fit_ridge(index, cpu_config, ridge_path)
    baseline = StreamingRidgeTranslator(ridge)
    options = dict(skip_mode='affine', **(dict(hidden_dim=8, pointer_hidden_dim=10) if name == 'residual_mlp' else OPTIONS))
    model = construct_model(name, baseline.source_spec, options)
    model.initialize_from_ridge(ridge)
    pair, _ = load_pair(index.entries[0])
    s = pair['source_canonical']
    inputs = (s.spatial_memory, s.object_pointer, s.validity)
    for a, b in zip(model.translate_tensors(*inputs), baseline.translate_tensors(*inputs)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-7)
    assert model.to_payload()['schema_version'].endswith('v3' if name == 'residual_mlp' else 'v2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for step in range(2):
        optimizer.zero_grad()
        a, b = model.translate_tensors(*inputs)
        (a.square().mean() + b.square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if step:
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters()
                       if 'weight' in n)
        optimizer.step()
    artifact = tmp_path / 'model.pt'
    torch.save({name: model.to_payload()}, artifact)
    restored = load_learned_translator(artifact, name)
    for a, b in zip(model.translate_tensors(*inputs), restored.translate_tensors(*inputs)):
        assert torch.equal(a, b)
    altered = dict(ridge, provenance=dict(ridge['provenance'], data_identity={'different': True}))
    torch.save(altered, tmp_path / 'foreign.pth')
    with pytest.raises(ValueError, match='different index'):
        train(index, cpu_config, name, tmp_path / 'bad', options=options,
              ridge_path=tmp_path / 'foreign.pth', allow_sample=True)


def test_cpu_batch_training_resume_selection_and_protocol_rejection(sample_index, cpu_config, tmp_path):
    index = load_index(sample_index, allow_sample=True)
    options = dict(hidden_dim=8, pointer_hidden_dim=10)
    continuous, history = train(index, replace(cpu_config, max_epochs=4), 'residual_mlp',
                                tmp_path / 'continuous', options=options, allow_sample=True)
    train(index, cpu_config, 'residual_mlp', tmp_path / 'resume', options=options, allow_sample=True)
    resumed, resumed_history = train(index, replace(cpu_config, max_epochs=4), 'residual_mlp',
        tmp_path / 'resume', options=options, resume=tmp_path / 'resume/last.ckpt', allow_sample=True)
    assert history[-1]['step'] == 8
    for key, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    for a, b in zip(history, resumed_history):
        assert a['validation'] == b['validation']
        assert a['train'] == b['train']
    last = torch.load(tmp_path / 'resume/last.ckpt', weights_only=True)
    selected = min(history, key=lambda row: row['validation']['all']['state_loss'])
    assert last['best_epoch'] == selected['epoch']
    assert last['best_loss'] == selected['validation']['all']['state_loss']
    metrics = history[0]['validation']
    assert metrics['present']['records'] + metrics['absent']['records'] == metrics['all']['records']
    with pytest.raises(ValueError, match='resume model/data/skip'):
        train(index, replace(cpu_config, global_batch=4), 'residual_mlp', tmp_path / 'resume',
              options=options, resume=tmp_path / 'resume/last.ckpt', allow_sample=True)
    with pytest.raises(ValueError, match='sample mode|sample indexes'):
        train(index, cpu_config, 'residual_mlp', tmp_path / 'cuda', device='cuda', allow_sample=True)


def test_usage_boundaries_and_unverified_skip_is_disabled():
    frames = torch.tensor([[[0] + list(range(36, 51))]])
    conditioning = frames.eq(0)
    valid = torch.ones_like(frames, dtype=torch.bool)
    spatial, pointer, supported = record_usage(frames, conditioning, valid, 50)
    assert supported
    assert frames[spatial].tolist() == [0, 45, 46, 47, 48, 49, 50]
    assert pointer.all()
    for policy in (MemoryPolicy(stride=2), MemoryPolicy(reverse=True), MemoryPolicy(correction=True)):
        sm, pm, supported = record_usage(frames, conditioning, valid, 50, policy)
        assert not supported and sm.all() and pm.all()
    with pytest.raises(ValueError, match='disabled'):
        require_selective_handoff_verified()


def _distributed_worker(rank, index_path, output, store):
    from datetime import timedelta
    import torch.distributed as dist
    import os
    import sys
    if sys.platform == 'darwin':
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        index = load_index(index_path, allow_sample=True)
        config = TrainConfig(workers=0, global_batch=6, microbatch=2, max_epochs=2, clip_norm=100.)
        model, _ = train(index, config, 'residual_mlp', output,
                         options=dict(hidden_dim=8, pointer_hidden_dim=10), allow_sample=True)
        torch.save(model.state_dict(), str(output) + f'.rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_gloo_two_process_tail_matches_single_process(sample_index, tmp_path):
    import torch.multiprocessing as mp
    index = load_index(sample_index, allow_sample=True)
    config = TrainConfig(workers=0, global_batch=6, microbatch=2, max_epochs=2, clip_norm=100.)
    train(index, config, 'residual_mlp', tmp_path / 'single',
          options=dict(hidden_dim=8, pointer_hidden_dim=10), allow_sample=True)
    mp.spawn(_distributed_worker, args=(str(sample_index), str(tmp_path / 'distributed'),
             str(tmp_path / 'gloo-store')), nprocs=2, join=True)
    rank0 = torch.load(str(tmp_path / 'distributed') + '.rank0.pt', weights_only=True)
    rank1 = torch.load(str(tmp_path / 'distributed') + '.rank1.pt', weights_only=True)
    assert all(torch.equal(rank0[k], rank1[k]) for k in rank0)
    single = torch.load(tmp_path / 'single/last.ckpt', weights_only=True)
    distributed = torch.load(tmp_path / 'distributed/last.ckpt', weights_only=True)
    assert single['step'] == distributed['step'] == 4
    for key, tensor in single['model']['state_dict'].items():
        torch.testing.assert_close(tensor, distributed['model']['state_dict'][key], atol=2e-6, rtol=2e-6)
    for a, b in zip(single['history'], distributed['history']):
        assert a['validation']['all']['records'] == b['validation']['all']['records'] == 3
        assert abs(a['validation']['all']['state_loss'] - b['validation']['all']['state_loss']) < 1e-6


@pytest.mark.parametrize('name', ['transformer', 'mlp_context'])
def test_common_trainer_supports_transformer_and_affine_context(sample_index, cpu_config, tmp_path, name):
    index = load_index(sample_index, allow_sample=True)
    options = dict(OPTIONS)
    ridge_path = None
    if name == 'mlp_context':
        options['skip_mode'] = 'affine'
        ridge_path = tmp_path / 'ridge.pth'
        fit_ridge(index, cpu_config, ridge_path)
    _, history = train(index, cpu_config, name, tmp_path / 'trained', options=options,
                       ridge_path=ridge_path, allow_sample=True)
    assert history[-1]['step'] == 4
    assert all(row['validation']['all']['records'] == 3 for row in history)
    model = load_learned_translator(tmp_path / 'trained/paired_translators.pt', name)
    assert model.skip_mode == ('affine' if ridge_path else 'identity')


def test_cli_preflight_and_cpu_smoke_do_not_invent_full_readiness(sample_index, tmp_path, cpu_config):
    from vos_memory_inspector.state_training.cli import main
    config = tmp_path / 'config.json'
    save_json(config, dict(model='residual_mlp', model_options=dict(hidden_dim=8, pointer_hidden_dim=10, skip_mode='affine'),
                          training=dict(workers=0, global_batch=6, microbatch=2)))
    common = ['--index', str(sample_index), '--sample-index', '--config', str(config)]
    assert main(['preflight', *common, '--output', str(tmp_path / 'preflight.json')]) == 2
    report = json.loads((tmp_path / 'preflight.json').read_text())
    assert not report['full_data_ready'] and not report['launch_ready']
    assert main(['cpu-smoke', *common, '--output', str(tmp_path / 'smoke'), '--sample-cases', '2']) == 0
    smoke = json.loads((tmp_path / 'smoke/cpu_smoke_report.json').read_text())
    assert not smoke['validation_or_benchmark_score'] and not smoke['cuda_training_executed']
    assert len(smoke['samples']) == 2
    initialization = load_ridge(tmp_path / 'smoke/sample_initialization.pth')
    assert initialization['provenance']['data_identity']['sample_only']
    index = load_index(sample_index, allow_sample=True)
    with pytest.raises(ValueError, match='different index'):
        train(index, cpu_config, 'residual_mlp', tmp_path / 'invalid-init',
              options=dict(hidden_dim=8, pointer_hidden_dim=10, skip_mode='affine'),
              ridge_path=tmp_path / 'smoke/sample_initialization.pth', allow_sample=True)


def test_state_evaluation_accepts_legacy_raw_and_new_ridge_artifacts(sample_index, cpu_config, tmp_path):
    from vos_memory_inspector.state_training.evaluation import evaluate_artifact
    index = load_index(sample_index, allow_sample=True)
    pair, _ = load_pair(index.entries[0])
    model = construct_model('transformer', pair['source_canonical'].spec, OPTIONS)
    raw = tmp_path / 'legacy.pth'
    torch.save(model.to_payload(), raw)
    result = evaluate_artifact(index, cpu_config, 'transformer', raw, tmp_path / 'legacy_eval.json', allow_sample=True)
    assert result['metrics']['all']['records'] == 3 and not result['vos_score']
    ridge = tmp_path / 'ridge.pth'
    fit_ridge(index, cpu_config, ridge)
    result = evaluate_artifact(index, cpu_config, 'ridge', ridge, tmp_path / 'ridge_eval.json', allow_sample=True)
    assert result['metrics']['all']['state_loss'] < 0.001
    assert load_learned_translator(ridge, 'ridge').source_spec == model.source_spec


def test_worker_spawn_loader_coverage(sample_index, cpu_config):
    index = load_index(sample_index, allow_sample=True)
    rows = list(record_loader(index.role('train'), replace(cpu_config, workers=1)))
    refs = [ref for batch in rows for ref in batch['ref']]
    assert len(refs) == len(set(refs)) == 7


def test_used_spatial_loss_keeps_pointer_training(sample_index, cpu_config, tmp_path):
    index = load_index(sample_index, allow_sample=True)
    _, history = train(index, replace(cpu_config, spatial_loss_records='used'), 'residual_mlp',
                       tmp_path / 'used', options=dict(hidden_dim=8, pointer_hidden_dim=10), allow_sample=True)
    assert history[-1]['train']['all']['records'] == 7
    assert history[-1]['training_objective'] > 0
