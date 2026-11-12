"""The borrowed operators, checked numerically against the cited papers.

Each reference below is written from the paper's equations (or, where the paper
gives none, from the official code it points to), independently of the module
under test. ``docs/paper_check_report.md`` lists the maximum differences.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vos_memory_inspector.models import ResidualMLPStateTranslator, TransformerStateTranslator
from vos_memory_inspector.models._stitched import (
    RIDGE_SCHEMA,
    FeedForward,
    SpatialReductionAttention,
    StitchingLayer,
    TranslatorBlock,
    sine_position_encoding,
)
from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.state_training.statistics import Moments, solve


def _randomize(module: nn.Module, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))


# --- Spatial-reduction attention: PVT (Wang et al., ICCV 2021), Eq. 1-4 -----------------


def _pvt_sra_reference(module: SpatialReductionAttention, tokens: torch.Tensor) -> torch.Tensor:
    """SRA as the paper writes it, with no position signal.

    SR(x) = Norm(Reshape(x, R) W_S); head_j = Attention(Q W_Q, SR(K) W_K, SR(V) W_V);
    Attention(q, k, v) = Softmax(q k^T / sqrt(d_head)) v; output = Concat(heads) W_O.
    """

    count, positions, dim = tokens.shape
    height, width = module.grid
    ratio, heads = module.sr_ratio, module.num_heads
    head_dim = dim // heads
    # Reshape(x, R): group each R x R neighbourhood into one row of R^2 * C values.
    grouped = tokens.reshape(count, height // ratio, ratio, width // ratio, ratio, dim)
    grouped = grouped.permute(0, 1, 3, 2, 4, 5).reshape(count, -1, ratio * ratio * dim)
    # W_S in R^{(R^2 C) x C}: the strided convolution's weight, flattened in the same order.
    w_s = module.sr.weight.permute(2, 3, 1, 0).reshape(ratio * ratio * dim, dim)
    reduced = F.layer_norm(
        grouped @ w_s + module.sr.bias, (dim,), module.sr_norm.weight, module.sr_norm.bias,
        module.sr_norm.eps,
    )
    query = tokens @ module.q.weight.T + module.q.bias
    key = reduced @ module.k.weight.T + module.k.bias
    value = reduced @ module.v.weight.T + module.v.bias
    outputs = []
    for head in range(heads):
        part = slice(head * head_dim, (head + 1) * head_dim)
        scores = query[..., part] @ key[..., part].transpose(1, 2) / math.sqrt(head_dim)
        outputs.append(scores.softmax(dim=-1) @ value[..., part])
    return torch.cat(outputs, dim=-1) @ module.proj.weight.T + module.proj.bias


@pytest.mark.parametrize("heads,ratio", [(1, 4), (1, 2), (2, 2), (4, 4)])
def test_attention_without_position_signal_is_the_spatial_reduction_attention_of_pvt(heads, ratio):
    module = SpatialReductionAttention(8, (8, 8), num_heads=heads, sr_ratio=ratio).double()
    _randomize(module)
    module.pos_query.zero_()
    module.pos_key.zero_()
    tokens = torch.randn(3, 64, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        difference = (module(tokens) - _pvt_sra_reference(module, tokens)).abs().max()
    assert float(difference) < 1e-12
    # The reduction divides each spatial side by R, so the key count falls by R^2 (Eq. 3).
    assert module.reduced_grid == (8 // ratio, 8 // ratio)
    assert module.sr.kernel_size == module.sr.stride == (ratio, ratio)
    assert isinstance(module.sr_norm, nn.LayerNorm)


def test_position_signal_enters_queries_and_keys_only_as_in_detr():
    """DETR adds the encoding to queries and keys and leaves values untouched
    (appendix A.3; ``q = k = with_pos_embed(src, pos)``, ``value=src``)."""

    module = SpatialReductionAttention(8, (8, 8), num_heads=1, sr_ratio=2).double()
    _randomize(module)
    tokens = torch.randn(2, 64, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        with_position = module(tokens)
        # The same computation written out: position added before the q and k projections only.
        grid = tokens.transpose(1, 2).reshape(2, 8, 8, 8)
        reduced = module.sr_norm(module.sr(grid).flatten(2).transpose(1, 2))
        query = module.q(tokens + module.pos_query)
        key = module.k(reduced + module.pos_key)
        value = module.v(reduced)  # no position here
        scores = query @ key.transpose(1, 2) / math.sqrt(8)
        expected = module.proj(scores.softmax(-1) @ value)
        module.pos_query.zero_()
        module.pos_key.zero_()
        without_position = module(tokens)
    assert float((with_position - expected).abs().max()) < 1e-12
    assert float((with_position - without_position).abs().max()) > 1e-3


# --- Sine position encoding: DETR / SAM 2 code and Vaswani et al. (2017) ----------------


def _detr_sine_reference(y_embed, x_embed, num_pos_feats, temperature=10000.0):
    """``PositionEmbeddingSine`` of DETR (and of SAM 2) for given coordinates.

    dim_t = T^(2 * (i // 2) / N); sine on even and cosine on odd channels;
    the y half comes first.
    """

    index = torch.arange(num_pos_feats, dtype=torch.float64)
    dim_t = temperature ** (2 * (index // 2) / num_pos_feats)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    return torch.cat((pos_y, pos_x), dim=2)


@pytest.mark.parametrize("height,width,dim", [(64, 64, 64), (16, 16, 64), (8, 4, 16)])
def test_sine_encoding_is_the_detr_formula_at_cell_centres(height, width, dim):
    ours = sine_position_encoding(height, width, dim).double().reshape(height, width, dim)
    rows = torch.arange(height, dtype=torch.float64)
    cols = torch.arange(width, dtype=torch.float64)
    scale = 2 * math.pi
    # Same formula, coordinates at cell centres: (i + 0.5) / H.
    centre_y = ((rows + 0.5) / height * scale)[:, None].expand(height, width)
    centre_x = ((cols + 0.5) / width * scale)[None, :].expand(height, width)
    centred = _detr_sine_reference(centre_y, centre_x, dim // 2)
    assert float((ours - centred).abs().max()) < 1e-6
    # DETR's own normalisation uses (i + 1) / H, half a cell later; that is the stated difference.
    detr_y = ((rows + 1) / (height + 1e-6) * scale)[:, None].expand(height, width)
    detr_x = ((cols + 1) / (width + 1e-6) * scale)[None, :].expand(height, width)
    detr = _detr_sine_reference(detr_y, detr_x, dim // 2)
    assert float((ours - detr).abs().max()) > 1e-3
    half_cell_y = ((rows + 0.5) / (height + 1e-6) * scale)[:, None].expand(height, width)
    half_cell_x = ((cols + 0.5) / (width + 1e-6) * scale)[None, :].expand(height, width)
    shifted = _detr_sine_reference(half_cell_y, half_cell_x, dim // 2)
    assert float((ours - shifted).abs().max()) < 1e-5


def test_sine_encoding_follows_the_transformer_formula_on_each_axis():
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""

    dim = 32
    half = dim // 2
    ours = sine_position_encoding(8, 8, dim).double().reshape(8, 8, dim)
    position = (torch.arange(8, dtype=torch.float64) + 0.5) / 8 * 2 * math.pi
    for i in range(half // 2):
        frequency = 10000.0 ** (2 * i / half)
        assert torch.allclose(ours[:, 0, 2 * i], (position / frequency).sin(), atol=1e-6)
        assert torch.allclose(ours[:, 0, 2 * i + 1], (position / frequency).cos(), atol=1e-6)
        assert torch.allclose(ours[0, :, half + 2 * i], (position / frequency).sin(), atol=1e-6)


def test_reduced_key_grid_sits_at_the_centres_of_its_cells():
    """With centre coordinates a reduced key lies at the mean position of the
    R x R positions it covers; with DETR's (i + 1) / H it would not."""

    rows = torch.arange(64, dtype=torch.float64)
    fine = (rows + 0.5) / 64
    coarse = (torch.arange(16, dtype=torch.float64) + 0.5) / 16
    assert torch.allclose(fine.reshape(16, 4).mean(1), coarse)
    detr_fine = (rows + 1) / 64
    detr_coarse = (torch.arange(16, dtype=torch.float64) + 1) / 16
    assert not torch.allclose(detr_fine.reshape(16, 4).mean(1), detr_coarse)


# --- Block: Pre-LN order (ViT; Xiong et al.), zero start (Fixup rule 1) -----------------


def test_block_is_pre_ln_with_identity_shortcuts():
    block = TranslatorBlock(
        8, (8, 8), mlp_ratio=4.0, attention=True, num_heads=1, sr_ratio=2
    ).double()
    _randomize(block)
    tokens = torch.randn(2, 64, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        middle = tokens + block.attn(block.norm_attn(tokens))  # z' = MSA(LN(z)) + z
        expected = middle + block.mlp(block.norm_mlp(middle))  # z = MLP(LN(z')) + z'
        assert float((block(tokens) - expected).abs().max()) < 1e-12


@pytest.mark.parametrize("attention", [False, True])
def test_zero_start_makes_the_block_an_exact_identity(attention):
    block = TranslatorBlock(
        8, (8, 8), mlp_ratio=4.0, attention=attention, num_heads=1, sr_ratio=2
    )
    # Rule 1 of Fixup: the last layer of each residual branch starts at zero.
    assert not block.mlp.fc2.weight.any() and not block.mlp.fc2.bias.any()
    if attention:
        assert not block.attn.proj.weight.any() and not block.attn.proj.bias.any()
    tokens = torch.randn(2, 64, 8, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.equal(block(tokens), tokens)


def test_feed_forward_is_two_linear_layers_with_gelu_and_ratio_four():
    """FFN(x) = W2 act(W1 x + b1) + b2 (Vaswani et al., Eq. 2), act = GELU (ViT),
    inner width 4x; GELU(x) = x * Phi(x) (Hendrycks & Gimpel)."""

    mlp = FeedForward(8, 32).double()
    _randomize(mlp)
    block = TranslatorBlock(64, (64, 64), mlp_ratio=4.0, attention=True, num_heads=1, sr_ratio=4)
    assert block.mlp.fc1.out_features == 4 * 64
    assert block.attn.dim // block.attn.num_heads == 64  # head size 64, as in Vaswani et al.
    tokens = torch.randn(5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        hidden = tokens @ mlp.fc1.weight.T + mlp.fc1.bias
        gelu = hidden * 0.5 * (1 + torch.erf(hidden / math.sqrt(2)))
        expected = gelu @ mlp.fc2.weight.T + mlp.fc2.bias
        assert float((mlp(tokens) - expected).abs().max()) < 1e-12


# --- Stitching layer and its least-squares start ---------------------------------------


def test_stitching_layer_is_a_1x1_convolution_with_bias_and_no_normalisation():
    """Bansal et al. and Csiszarik et al. use a 1x1 convolution; Bansal et al. add
    BatchNorm before and after it, which this layer does not have."""

    layer = StitchingLayer(6, 6).double()
    assert [name for name, _ in layer.named_modules() if name] == ["linear"]
    _randomize(layer)
    conv = nn.Conv2d(6, 6, kernel_size=1).double()
    with torch.no_grad():
        conv.weight.copy_(layer.linear.weight[:, :, None, None])
        conv.bias.copy_(layer.linear.bias)
        frames = torch.randn(2, 6, 5, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
        ours = layer(frames.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(2, 6, 5, 5)
        assert float((ours - conv(frames)).abs().max()) < 1e-12


@pytest.mark.parametrize("model_type", [ResidualMLPStateTranslator, TransformerStateTranslator])
def test_ridge_start_with_no_penalty_is_the_least_squares_solution(model_type):
    """Csiszarik et al. and Pan et al. start the stitching layer from the
    least-squares solution; with penalty 0 the ridge solver must return it."""

    generator = torch.Generator().manual_seed(7)
    spec = StateSpec(feature_channels=8, height=4, width=4, pointer_dim=5)
    records = 40
    spatial = torch.randn(1, 1, records, 8, 4, 4, dtype=torch.float64, generator=generator)
    pointer = torch.randn(1, 1, records, 5, dtype=torch.float64, generator=generator)
    true_w = torch.randn(8, 8, dtype=torch.float64, generator=generator)
    true_a = torch.randn(5, 5, dtype=torch.float64, generator=generator)
    x = spatial[0, 0].movedim(1, -1).reshape(-1, 8)
    y = x @ true_w + 0.3 + 0.05 * torch.randn(x.shape, dtype=torch.float64, generator=generator)
    p = pointer[0, 0]
    q = p @ true_a - 0.2 + 0.05 * torch.randn(p.shape, dtype=torch.float64, generator=generator)

    def least_squares(inputs, targets):
        design = torch.cat((inputs, torch.ones(len(inputs), 1, dtype=torch.float64)), dim=1)
        solution = torch.linalg.lstsq(design, targets).solution
        return solution[:-1], solution[-1]

    artifact = {
        "schema_version": RIDGE_SCHEMA,
        "source_spec": spec.to_dict(),
        "target_spec": spec.to_dict(),
        "provenance": {"artifact_sha256": "0" * 64, "data_identity": {"test": True}},
    }
    for name, inputs, targets in (("spatial", x, y), ("pointer", p, q)):
        stats = Moments(inputs.shape[1], targets.shape[1])
        stats.update(inputs, targets)
        affine = solve(stats, 0.0)
        weight, bias = least_squares(inputs, targets)
        assert float((affine.weight - weight).abs().max()) < 1e-9
        assert float((affine.bias - bias).abs().max()) < 1e-9
        artifact[name] = {"weight": affine.weight.float(), "bias": affine.bias.float()}

    model = model_type(spec, spec)
    model.initialize_from_ridge(artifact)
    with torch.no_grad():
        out_spatial, out_pointer = model.translate_tensors(spatial.float(), pointer.float())
    weight, bias = least_squares(x, y)
    expected = (x @ weight + bias).reshape(records, 4, 4, 8).movedim(-1, 1)
    assert float((out_spatial[0, 0].double() - expected).abs().max()) < 1e-4
    weight, bias = least_squares(p, q)
    assert float((out_pointer[0, 0].double() - (p @ weight + bias)).abs().max()) < 1e-4
