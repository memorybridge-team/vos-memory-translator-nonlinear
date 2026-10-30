"""Execution policy is explicit and bound to epoch-boundary resume."""
import os

import pytest
import torch

from test_state_training import sample_index, cpu_config
from vos_memory_inspector.state_training.data import load_index
from vos_memory_inspector.state_training.determinism import configure_determinism, execution_profile, CUBLAS_WORKSPACE
from vos_memory_inspector.state_training.preflight import inspect
from vos_memory_inspector.state_training.storage import digest
from vos_memory_inspector.state_training.trainer import train


@pytest.fixture(autouse=True)
def preserve_process_settings():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def test_cpu_determinism_sets_required_flags_without_initializing_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: pytest.fail('CPU policy must not initialize CUDA'))
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    profile = configure_determinism('cpu')
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
    assert profile['cublas_workspace_config'] is None
    assert profile['broadcast_buffers'] is False


def test_cuda_environment_is_set_before_initialization_and_conflicts_are_rejected(monkeypatch):
    # Test setup ordering without requiring or executing any CUDA operations.
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    assert execution_profile('cuda')['cublas_workspace_config'] == CUBLAS_WORKSPACE
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    profile = configure_determinism('cuda')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == profile['cublas_workspace_config'] == CUBLAS_WORKSPACE
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    with pytest.raises(ValueError, match='must be :4096:8'):
        configure_determinism('cuda')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='before CUDA initialization'):
        configure_determinism('cuda')
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ


def test_resume_rejects_changed_or_legacy_execution_profiles(sample_index, cpu_config, tmp_path):
    index = load_index(sample_index, allow_sample=True)
    readiness = inspect(index, cpu_config)
    options = dict(hidden_dim=8, pointer_hidden_dim=10)
    train(index, cpu_config, 'residual_mlp', tmp_path / 'original', options=options,
          readiness=readiness, allow_sample=True)
    original = torch.load(tmp_path / 'original/last.ckpt', weights_only=True)
    assert original['identity']['determinism']['deterministic_algorithms']
    original['identity']['determinism']['broadcast_buffers'] = True
    original['signature'] = digest(original['identity'])
    changed = tmp_path / 'changed.ckpt'
    torch.save(original, changed)
    from dataclasses import replace
    config = replace(cpu_config, max_epochs=3)
    with pytest.raises(ValueError, match='determinism changed'):
        train(index, config, 'residual_mlp', tmp_path / 'changed', options=options,
              readiness=readiness, allow_sample=True, resume=changed)
    original['identity'].pop('determinism')
    original['signature'] = digest(original['identity'])
    legacy = tmp_path / 'legacy.ckpt'
    torch.save(original, legacy)
    with pytest.raises(ValueError, match='no deterministic execution profile'):
        train(index, config, 'residual_mlp', tmp_path / 'legacy', options=options,
              readiness=readiness, allow_sample=True, resume=legacy)
