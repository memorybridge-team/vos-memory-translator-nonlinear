"""Train-only FP64 sufficient statistics, validation selection, FP32 affine maps."""
from pathlib import Path
import torch
from torch import nn

from ..models._base import _LearnedStateTranslator
from ..state_schema import StateSpec
from .data import record_loader
from .statistics import Moments, solve, metrics, select_alpha
from .storage import save_torch, sha256

ALPHAS = (0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1)
SCHEMA = 'cmmt.streaming_ridge.v1'


class StreamingRidgeTranslator(_LearnedStateTranslator):
    name = 'ridge'
    _spatial_module_prefix = 'feature'

    def __init__(self, payload):
        if payload.get('schema_version') != SCHEMA:
            raise ValueError('streaming ridge artifact required')
        super().__init__(StateSpec(**payload['source_spec']), StateSpec(**payload['target_spec']), output_dtype='source')
        self.payload = payload
        self.feature = nn.Linear(self.source_spec.feature_channels, self.target_spec.feature_channels)
        self.pointer = nn.Linear(self.source_spec.pointer_dim, self.target_spec.pointer_dim)
        from ..models.affine import copy_affine
        copy_affine(self.feature, payload['spatial'])
        copy_affine(self.pointer, payload['pointer'])
        self.requires_grad_(False)

    def _feature_head(self, x):
        return self.feature(x.to(self.feature.weight))

    def _pointer_head(self, x):
        return self.pointer(x.to(self.pointer.weight))

    def macs_per_frame(self):
        s, t = self.source_spec, self.target_spec
        return t.height * t.width * s.feature_channels * t.feature_channels

    def pointer_macs_per_record(self):
        return self.source_spec.pointer_dim * self.target_spec.pointer_dim


def load_ridge(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema_version') != SCHEMA:
        raise ValueError('invalid streaming ridge artifact')
    if not payload.get('provenance', {}).get('data_identity'):
        raise ValueError('ridge data identity missing')
    payload['provenance'] = {**payload['provenance'], 'artifact_sha256': sha256(path)}
    return payload


def collect_statistics(entries, config, *, chunk_positions=16384):
    spatial, pointer = None, None
    spec = None
    from .data import load_pair
    for batch in record_loader(entries, config):
        if spec is None:
            s = load_pair(entries[0])[0]['source_canonical']
            spec = s.spec
            spatial = Moments(spec.feature_channels, spec.feature_channels)
            pointer = Moments(spec.pointer_dim, spec.pointer_dim)
        x = batch['source_spatial'].movedim(1, -1).reshape(-1, spec.feature_channels)
        y = batch['target_spatial'].movedim(1, -1).reshape(-1, spec.feature_channels)
        for start in range(0, len(x), chunk_positions):
            spatial.update(x[start:start + chunk_positions], y[start:start + chunk_positions])
        pointer.update(batch['source_pointer'], batch['target_pointer'])
    if spec is None:
        raise ValueError('nonempty split required for ridge')
    return spec, {'spatial': spatial, 'pointer': pointer}


def fit_ridge(index, config, output):
    config.validate()
    output = Path(output)
    if output.exists() or any(output.resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('ridge output must be new and outside original cache roots')
    train_spec, train = collect_statistics(index.role('train'), config)
    valid_spec, valid = collect_statistics(index.role('validation'), config)
    if train_spec != valid_spec:
        raise ValueError('train/validation spec mismatch')
    selected, reports = {}, {}
    for component in ('spatial', 'pointer'):
        candidates = []
        for alpha in ALPHAS:
            affine = solve(train[component], alpha)
            # Select the coefficients actually stored for initialization.
            affine.weight = affine.weight.float()
            affine.bias = affine.bias.float()
            score = metrics(valid[component], affine, train[component].rms2())
            candidates.append(dict(alpha=alpha, development_mse=score['mse'], metrics=score,
                                   affine=affine))
        choice = select_alpha(candidates)
        affine = choice['affine']
        selected[component] = dict(weight=affine.weight, bias=affine.bias,
                                   diagnostics=affine.diagnostics)
        reports[component] = [dict(alpha=c['alpha'], validation=c['metrics'],
                                  selected=c is choice) for c in candidates]
    payload = dict(schema_version=SCHEMA, source_spec=train_spec.to_dict(), target_spec=train_spec.to_dict(),
                   **selected, candidates=reports,
                   normalization=dict(scope='train_only', scales={c: train[c].rms2() ** 0.5 for c in train}),
                   provenance=dict(data_identity=index.identity, selection_role='validation',
                                   refit_after_selection=False),
                   statistics={role: {c: stats.payload() for c, stats in values.items()}
                               for role, values in (('train', train), ('validation', valid))})
    save_torch(output, payload)
    return load_ridge(output)
