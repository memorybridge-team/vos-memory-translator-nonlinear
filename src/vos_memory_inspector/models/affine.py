"""Ridge initialization helpers; row-vector artifact weights become Linear weights."""
import torch
from torch import nn


def make_skip(inputs, outputs):
    layer = nn.Linear(inputs, outputs)
    with torch.no_grad():
        nn.init.eye_(layer.weight)
        layer.bias.zero_()
    return layer


def copy_affine(layer, values):
    weight, bias = values['weight'], values['bias']
    if weight.shape != (layer.in_features, layer.out_features) or bias.shape != (layer.out_features,):
        raise ValueError('ridge initialization shape mismatch')
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('nonfinite ridge initialization')
    with torch.no_grad():
        layer.weight.copy_(weight.T.to(layer.weight))
        layer.bias.copy_(bias.to(layer.bias))


def initialize_skips(model, spatial, pointer, ridge):
    if ridge.get('schema_version') != 'cmmt.streaming_ridge.v1':
        raise ValueError('streaming ridge artifact required')
    if ridge['source_spec'] != model.source_spec.to_dict() or ridge['target_spec'] != model.target_spec.to_dict():
        raise ValueError('ridge specs mismatch')
    provenance = ridge.get('provenance', {})
    if not provenance.get('artifact_sha256') or not provenance.get('data_identity'):
        raise ValueError('ridge hash and data identity required')
    copy_affine(spatial, ridge['spatial'])
    copy_affine(pointer, ridge['pointer'])
    model.ridge_initialization = dict(provenance)
