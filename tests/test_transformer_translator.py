from __future__ import annotations

import re
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from vos_memory_inspector.transformer_translator import (
    SAM21_MEMORY_SPEC,
    SpatialMemoryTranslator,
    ResidualPointerTranslator,
    SpatialSelfAttention,
    SpatialTransformerBlock,
    SpatialTransformerConfig,
    build_translator,
    main,
    sincos_2d_embedding,
)


def _randomize_alpha(module: SpatialMemoryTranslator, seed: int = 3) -> None:
    """Make Delta visible in the output (alpha starts at zero)."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.alpha.copy_(torch.randn(module.alpha.shape, generator=generator))


# --------------------------------------------------------------------------
# Task 2: config, positional embedding, Pre-LN block
# --------------------------------------------------------------------------


def test_config_defaults_match_base_and_round_trip() -> None:
    config = SpatialTransformerConfig()
    assert (config.channels, config.height, config.width) == (64, 64, 64)
    assert (config.d_model, config.num_heads, config.num_layers) == (64, 4, 2)
    assert (config.mlp_ratio, config.dropout, config.patch_size) == (2.0, 0.0, 4)
    assert (config.patch_embed, config.pos_embed, config.local_fusion) == ("conv", "learnable", "linear")
    assert config.spatial_context and config.use_attention and config.use_residual
    assert config.residual_scale_init == 0.0
    assert (config.pointer_dim, config.pointer_hidden_dim) == (256, 512)
    assert config.output_dtype == "source"
    assert config.grid_size == (16, 16) and config.num_tokens == 256 and config.ffn_hidden_dim == 128

    custom = SpatialTransformerConfig(patch_embed="avgpool", pos_embed="sincos", patch_size=8)
    assert SpatialTransformerConfig.from_dict(custom.to_dict()) == custom
    with pytest.raises(ValueError, match="unknown"):
        SpatialTransformerConfig.from_dict({**config.to_dict(), "extra": 1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"patch_size": 5},  # 64 is not divisible by 5
        {"num_heads": 3},  # 64 is not divisible by 3
        {"spatial_context": False, "local_fusion": "none"},
        {"patch_embed": "linear"},
        {"pos_embed": "rope"},
        {"local_fusion": "concat"},
        {"output_dtype": "bfloat16"},
        {"dropout": 1.0},
        {"mlp_ratio": 0.0},
        {"mlp_ratio": 1.3},  # 64 * 1.3 is not an integer
        {"d_model": 6, "num_heads": 2, "pos_embed": "sincos"},
        {"num_layers": 0},
        {"channels": True},
        {"residual_scale_init": float("nan")},
    ],
)
def test_config_rejects_invalid_settings(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SpatialTransformerConfig(**overrides)


def test_sincos_embedding_is_deterministic() -> None:
    first = sincos_2d_embedding(16, 16, 64)
    second = sincos_2d_embedding(16, 16, 64)
    assert first.shape == (256, 64) and first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.unique(first, dim=0).shape[0] == 256  # every position is distinct
    assert first.abs().max() <= 1.0
    # Position (0, 0): sin terms are 0 and cos terms are 1.
    quarter = 16
    assert torch.equal(first[0, :quarter], torch.zeros(quarter))
    assert torch.equal(first[0, quarter : 2 * quarter], torch.ones(quarter))
    with pytest.raises(ValueError):
        sincos_2d_embedding(16, 16, 62)


def test_block_shapes_and_parameter_count() -> None:
    block = SpatialTransformerBlock(64, 4, mlp_ratio=2.0)
    assert sum(p.numel() for p in block.parameters()) == 33_472
    tokens = torch.randn(3, 256, 64)
    assert block(tokens).shape == (3, 256, 64)
    assert SpatialSelfAttention(64, 4)(tokens).shape == (3, 256, 64)

    ffn_only = SpatialTransformerBlock(64, 4, use_attention=False)
    assert ffn_only.attn is None and ffn_only.norm1 is None
    assert sum(p.numel() for p in ffn_only.parameters()) == 16_704


def test_block_keeps_samples_in_a_batch_independent() -> None:
    torch.manual_seed(0)
    block = SpatialTransformerBlock(64, 4).eval()
    tokens = torch.randn(3, 256, 64)
    perturbed = tokens.clone()
    perturbed[1] += torch.randn(256, 64)
    with torch.no_grad():
        reference, changed = block(tokens), block(perturbed)
    torch.testing.assert_close(changed[0], reference[0])
    torch.testing.assert_close(changed[2], reference[2])
    assert not torch.allclose(changed[1], reference[1])


# --------------------------------------------------------------------------
# Task 3: SpatialMemoryTranslator
# --------------------------------------------------------------------------


def test_spatial_translator_shapes_and_identity_at_init() -> None:
    torch.manual_seed(0)
    module = SpatialMemoryTranslator()
    frames = torch.randn(3, 64, 64, 64)
    records = torch.randn(1, 2, 3, 64, 64, 64)
    with torch.no_grad():
        output, tokens = module(frames, return_tokens=True)
        output_6d, tokens_6d = module(records, return_tokens=True)
    assert output.shape == frames.shape and torch.equal(output, frames)
    assert output_6d.shape == records.shape and torch.equal(output_6d, records)
    assert tokens.shape == (3, 256, 64)
    assert tokens_6d.shape == (1, 2, 3, 256, 64)

    local_only = SpatialMemoryTranslator(SpatialTransformerConfig(spatial_context=False))
    with torch.no_grad():
        output, tokens = local_only(frames, return_tokens=True)
    assert tokens is None and torch.equal(output, frames)


def test_spatial_translator_fails_closed_on_grid_mismatch() -> None:
    module = SpatialMemoryTranslator()
    with pytest.raises(ValueError, match="does not match"):
        module(torch.randn(1, 64, 32, 32))
    with pytest.raises(ValueError, match="does not match"):
        module(torch.randn(1, 32, 64, 64))
    with pytest.raises(ValueError, match="N,C,H,W"):
        module(torch.randn(1, 1, 64, 64, 64))


def _naive_concat_delta(
    module: SpatialMemoryTranslator, memory: torch.Tensor
) -> torch.Tensor:
    """Reference fusion: 1x1 conv on [M || Up(ctx)] at full resolution."""

    context, _ = module._context(memory)  # [N, d, H/p, W/p]
    upsampled = F.interpolate(context, size=memory.shape[-2:], mode="bilinear", align_corners=False)
    stacked = torch.cat((memory, upsampled), dim=1)  # [N, C + d, H, W]
    weight = torch.cat((module.local_in.weight, module.context_proj.weight), dim=1)
    fused = F.conv2d(stacked, weight, module.local_in.bias)
    if module.local_out is not None:
        fused = module.local_out(F.gelu(fused))
    return fused


@pytest.mark.parametrize("fusion", ["linear", "mlp"])
def test_efficient_fusion_matches_naive_concat(fusion: str) -> None:
    torch.manual_seed(0)
    module = SpatialMemoryTranslator(SpatialTransformerConfig(local_fusion=fusion)).eval()
    _randomize_alpha(module)
    memory = torch.randn(2, 64, 64, 64)
    with torch.no_grad():
        expected = memory + module.alpha * _naive_concat_delta(module, memory)
        actual = module(memory)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_batched_frames_match_per_frame_and_stay_independent() -> None:
    torch.manual_seed(0)
    module = SpatialMemoryTranslator().eval()
    _randomize_alpha(module)
    frames = torch.randn(3, 64, 64, 64)
    with torch.no_grad():
        batched = module(frames)
        single = torch.cat([module(frames[index : index + 1]) for index in range(3)])
        perturbed = frames.clone()
        perturbed[1] += torch.randn(64, 64, 64)
        changed = module(perturbed)
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(changed[0], batched[0])
    torch.testing.assert_close(changed[2], batched[2])
    assert not torch.allclose(changed[1], batched[1])


@pytest.mark.parametrize("records", [1, 3, 7])
def test_attention_uses_256_tokens_per_frame_for_any_record_count(records: int, make_state) -> None:
    torch.manual_seed(0)
    translator = build_translator("base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    attention_inputs: list[tuple[int, ...]] = []
    qkv_outputs: list[tuple[int, ...]] = []
    hooks = []
    for block in translator.spatial.blocks:
        hooks.append(
            block.attn.register_forward_pre_hook(
                lambda _m, inputs: attention_inputs.append(tuple(inputs[0].shape))
            )
        )
        hooks.append(
            block.attn.qkv.register_forward_hook(
                lambda _m, _i, output: qkv_outputs.append(tuple(output.shape))
            )
        )
    state = make_state(
        torch.randn(1, 1, records, 64, 64, 64).to(torch.bfloat16),
        torch.randn(1, 1, records, 256),
    )
    with torch.no_grad():
        translator.translate(state)
    for hook in hooks:
        hook.remove()
    # Frames stay on the batch axis; every attention sees exactly 256 tokens.
    assert attention_inputs == [(records, 256, 64)] * 2
    assert qkv_outputs == [(records, 256, 192)] * 2


def test_rezero_gradients_on_a_deterministic_fixture() -> None:
    torch.manual_seed(0)
    module = SpatialMemoryTranslator()
    generator = torch.Generator().manual_seed(1)
    memory = torch.randn(2, 64, 64, 64, generator=generator)
    target = torch.randn(2, 64, 64, 64, generator=generator)

    F.mse_loss(module(memory), target).backward()
    assert module.alpha.grad.abs().max() > 1e-8
    for name, parameter in module.named_parameters():
        if name != "alpha":
            assert parameter.grad is not None, name
            assert torch.count_nonzero(parameter.grad) == 0, name

    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(module(memory), target).backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad.abs().max() > 1e-8, name


def test_pointer_head_starts_at_identity_and_learns() -> None:
    torch.manual_seed(0)
    pointer = ResidualPointerTranslator(256, 512)
    assert pointer.parameter_count() == 262_912
    assert ResidualPointerTranslator(256, 128).parameter_count() == 65_920
    values = torch.randn(5, 256)
    output = pointer(values)
    assert torch.equal(output, values)
    F.mse_loss(output, torch.randn(5, 256)).backward()
    assert torch.count_nonzero(pointer.fc1.weight.grad) == 0
    assert pointer.fc2.weight.grad.abs().max() > 1e-8


def test_base_parameters_and_analytic_macs_match_plan() -> None:
    module = SpatialMemoryTranslator()
    assert module.parameter_count() == 194_304
    assert module.macs_breakdown() == {
        "patch_embed": 16_777_216,
        "attention": 25_165_824,
        "ffn": 8_388_608,
        "context_conv": 9_437_184,
        "context_proj": 1_048_576,
        "local_in": 16_777_216,
    }
    assert module.macs_per_frame() == 77_594_624
    assert round(module.context_branch_macs() / 1e6, 1) == 60.8
    assert round(module.per_pixel_macs() / 1e6, 1) == 16.8


@pytest.mark.parametrize("preset", ["base", "base_mlp", "res8", "no_attn_context"])
def test_flop_counter_cross_checks_matmul_and_conv_terms(preset: str) -> None:
    """FLOPs are a separate measurement; only the matmul/conv groups are compared."""

    translator = build_translator(preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    module = translator.spatial
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), counter:
        module(torch.randn(1, 64, 64, 64))
    counts = {str(op): value for op, value in counter.get_flop_counts()["Global"].items()}
    macs = module.macs_breakdown()
    conv_macs = sum(macs.get(key, 0) for key in ("patch_embed", "context_conv", "context_proj", "local_in", "local_out"))
    matmul_macs = macs.get("attention", 0) + macs.get("ffn", 0)
    conv_flops = counts.get("aten.convolution", 0)
    matmul_flops = sum(counts.get(op, 0) for op in ("aten.addmm", "aten.mm", "aten.bmm"))
    assert conv_flops / 2 == pytest.approx(conv_macs, rel=0.01)
    assert matmul_flops / 2 == pytest.approx(matmul_macs, rel=0.01)


# --------------------------------------------------------------------------
# Task 6: dummy forward summary
# --------------------------------------------------------------------------


def test_cli_prints_dummy_forward_summary(capsys) -> None:
    arguments = ["--preset", "base", "--batch", "1", "--objects", "2", "--records", "3", "--invalid", "1"]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    expected_lines = [
        "preset             : base (Spatial-Context Transformer, linear fusion)",
        "input spatial      : (1, 2, 3, 64, 64, 64) bfloat16  valid 5/6 -> frames (5, 64, 64, 64)",
        "intermediate tokens: (5, 256, 64)",
        "output spatial     : (1, 2, 3, 64, 64, 64) bfloat16",
        "object pointer     : (1, 2, 3, 256) -> (1, 2, 3, 256)",
        "trainable params   : spatial 194,304 | pointer 262,912 | total 457,216",
        "MACs per frame     : ~77.6M (context branch 60.8M + per-pixel 16.8M)",
        "identity at init   : max|T(M)-M| = 0 | max|T(p)-p| = 0",
        "attention scope    : per frame, 256x256 (no cross-frame attention)",
    ]
    for line in expected_lines:
        assert line in output
    assert re.search(r"PyTorch FLOPs/frame: 155\.2M \(FlopCounterMode", output)
    assert re.search(r"^blocks\s+66,944\s+33\.6M", output, flags=re.MULTILINE)
    assert re.search(r"^pointer\s+262,912\s+0\.3M", output, flags=re.MULTILINE)


def test_cli_reports_skipped_forward_without_valid_records(capsys) -> None:
    assert main(["--preset", "no_context", "--records", "2", "--objects", "1", "--invalid", "2"]) == 0
    output = capsys.readouterr().out
    assert "valid 0/2 -> frames none (translator not called)" in output
    assert "intermediate tokens: n/a (no spatial context)" in output
    assert "attention scope    : none (position-wise head)" in output


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vos_memory_inspector.transformer_translator", "--preset", "depth1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "preset             : depth1 (Spatial-Context Transformer, linear fusion)" in result.stdout
