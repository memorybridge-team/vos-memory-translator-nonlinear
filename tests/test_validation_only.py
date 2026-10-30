"""Validation uses train normalization, never train cache IO."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_state_training import sample_index, cpu_config, OPTIONS
from vos_memory_inspector.state_training import data, preflight
from vos_memory_inspector.state_training.data import load_index, load_pair
from vos_memory_inspector.state_training.evaluation import evaluate_artifact
from vos_memory_inspector.state_training.trainer import construct_model


@pytest.mark.parametrize('normalization_source', ['index', 'preflight_report'])
def test_evaluation_does_not_touch_train_files(sample_index, cpu_config, tmp_path, monkeypatch, normalization_source):
    index = load_index(sample_index, allow_sample=True)
    report = preflight.inspect(index, cpu_config)
    value, _ = load_pair(index.role('validation')[0])
    model = construct_model('transformer', value['source_canonical'].spec, OPTIONS)
    artifact = tmp_path / 'model.pth'
    torch.save(model.to_payload(), artifact)
    if normalization_source == 'index':
        index = replace(index, normalization=report['normalization'])
    # A validation-only machine need not have the training caches mounted.
    train_paths = {Path(e['path']) for e in index.role('train')}
    for path in train_paths:
        path.unlink()
        Path(str(path) + '.sha256').unlink()
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        assert path not in train_paths, 'evaluation must not even stat train data'
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    original_load = data.load_pair
    loaded = []
    def load(entry, **kwargs):
        assert entry['case']['training_role'] == 'validation'
        loaded.append(entry['case']['case_id'])
        return original_load(entry, **kwargs)
    monkeypatch.setattr(data, 'load_pair', load)
    monkeypatch.setattr(preflight, 'inspect', lambda *a, **k: pytest.fail('unexpected full preflight'))
    result = evaluate_artifact(index, cpu_config, 'transformer', artifact, tmp_path / 'evaluation.json',
        allow_sample=True, readiness=report if normalization_source == 'preflight_report' else None)
    assert loaded == [e['case']['case_id'] for e in index.role('validation')]
    assert result['normalization'] == report['normalization']
    assert result['normalization_source'] == normalization_source
    assert result['metrics']['all']['records'] == 3
    assert not result['vos_score']


def test_sample_evaluation_cannot_estimate_rms_from_validation(sample_index, cpu_config, tmp_path):
    index = load_index(sample_index, allow_sample=True)
    with pytest.raises(ValueError, match='train normalization missing'):
        evaluate_artifact(index, cpu_config, 'transformer', tmp_path / 'unused.pth',
                          tmp_path / 'out.json', allow_sample=True)
