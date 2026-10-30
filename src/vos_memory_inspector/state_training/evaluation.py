"""State-only validation of legacy exports, selected checkpoints and ridge."""
from pathlib import Path
import torch

from .preflight import inspect, require_ready
from .ridge import SCHEMA, load_ridge, StreamingRidgeTranslator
from .storage import save_json, sha256
from .trainer import restore_model, evaluate


def evaluate_artifact(index, config, name, artifact, output, *, allow_sample=False):
    report = inspect(index, config)
    require_ready(report, allow_sample=allow_sample)
    raw = torch.load(artifact, map_location='cpu', weights_only=True)
    if raw.get('schema_version') == SCHEMA:
        if name != 'ridge':
            raise ValueError('ridge artifact requires model ridge')
        model = StreamingRidgeTranslator(load_ridge(artifact))
    else:
        payload = raw.get(name, raw)
        if raw.get('schema_version') == 'cmmt.record_training_checkpoint.v1':
            if raw['identity']['model'] != name:
                raise ValueError('checkpoint model mismatch')
            payload = raw['model']
        model = restore_model(name, payload)
        expected_fusion = 'linear' if name == 'transformer' else 'additive_mlp'
        if name != 'residual_mlp' and model.config.local_fusion != expected_fusion:
            raise ValueError('artifact architecture mismatch')
    if model.source_spec.to_dict() != report['source_spec']:
        raise ValueError('artifact spec mismatch')
    if Path(output).exists():
        raise ValueError('evaluation output exists')
    if any(Path(output).resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('evaluation output must be outside cache roots')
    result = dict(schema_version='cmmt.record_state_evaluation.v1', data_identity=index.identity,
                  artifact_sha256=sha256(artifact), normalization=report['normalization'], model=name,
                  evaluation_role='validation', sample_only=not index.official,
                  metrics=evaluate(model, index.role('validation'), config,
                                   report['normalization']['scales'], torch.device('cpu')),
                  historical_comparison_data_match=report['historical_comparison_data_match'],
                  vos_score=False, cuda_training_executed=False)
    save_json(output, result)
    return result
