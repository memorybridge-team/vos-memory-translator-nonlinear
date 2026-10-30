"""State-only validation of legacy exports, selected checkpoints and ridge."""
from pathlib import Path
import torch

from .preflight import validate_readiness, normalization_scales, REFERENCE_INDEX_SHA256
from .ridge import SCHEMA, load_ridge, StreamingRidgeTranslator
from .storage import save_json, sha256
from .trainer import restore_model, evaluate


def evaluate_artifact(index, config, name, artifact, output, *, allow_sample=False, readiness=None):
    config.validate()
    if not index.official and not allow_sample:
        raise ValueError('sample evaluation requires explicit sample mode')
    if not index.role('validation'):
        raise ValueError('validation split is empty')
    report = None
    if readiness is not None:
        # Do not stat, hash or load train caches during validation evaluation.
        report = validate_readiness(index, readiness, roles=('validation',), allow_sample=allow_sample).report
    normalization = report['normalization'] if report is not None else index.normalization
    if normalization is None:
        raise ValueError('train normalization missing: supply a verified preflight report')
    scales = normalization_scales(normalization)
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
    if report is not None and model.source_spec.to_dict() != report['source_spec']:
        raise ValueError('artifact spec mismatch')
    if model.source_spec != model.target_spec:
        raise ValueError('state-pair evaluation requires matching source/target specs')
    if index.official:
        from ..models._spatial import SAM21_MEMORY_SPEC
        if model.source_spec != SAM21_MEMORY_SPEC:
            raise ValueError('official artifact memory spec mismatch')
    if Path(output).exists():
        raise ValueError('evaluation output exists')
    if any(Path(output).resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('evaluation output must be outside cache roots')
    result = dict(schema_version='cmmt.record_state_evaluation.v1', data_identity=index.identity,
                  artifact_sha256=sha256(artifact), normalization=normalization, model=name,
                  normalization_source='preflight_report' if report is not None else 'index',
                  preflight_receipt_digest=report['receipt_digest'] if report is not None else None,
                  evaluation_role='validation', sample_only=not index.official,
                  metrics=evaluate(model, index.role('validation'), config,
                                   scales, torch.device('cpu')),
                  historical_comparison_data_match=index.official and index.identity['index_sha256'] == REFERENCE_INDEX_SHA256,
                  vos_score=False, cuda_training_executed=False)
    save_json(output, result)
    return result
