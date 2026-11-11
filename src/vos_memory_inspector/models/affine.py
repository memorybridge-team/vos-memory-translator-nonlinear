"""Copy closed-form (ridge) affine maps into ``nn.Linear`` layers."""
import torch


def copy_affine(layer, values):
    """Row-vector artifact weights ``[in, out]`` become ``Linear`` weights ``[out, in]``."""
    weight, bias = values['weight'], values['bias']
    if weight.shape != (layer.in_features, layer.out_features) or bias.shape != (layer.out_features,):
        raise ValueError('ridge initialization shape mismatch')
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('nonfinite ridge initialization')
    with torch.no_grad():
        layer.weight.copy_(weight.T.to(layer.weight))
        layer.bias.copy_(bias.to(layer.bias))
