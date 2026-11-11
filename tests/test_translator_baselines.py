"""Task 5: Moment-Matched Copy, component ablation and the preset registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.transformer_translator import (
    PRESETS,
    SAM21_MEMORY_SPEC,
    SpatialTransformerConfig,
    TransformerStateTranslator,
    build_translator,
    preset_names,
)
from vos_memory_inspector.translators import (
    ComponentAblationTranslator,
    DirectCopyTranslator,
    LinearStateTranslator,
    MomentMatchedCopyTranslator,
    ResidualMLPStateTranslator,
)


SPEC = StateSpec(4, 6, 6, 5)
# Per-role affine target maps: row 0 conditioning, row 1 non-conditioning.
SPATIAL_SCALE = torch.tensor([[2.0, 0.5, 1.5, 3.0], [0.25, 4.0, 1.0, 2.0]])
SPATIAL_SHIFT = torch.tensor([[1.0, -2.0, 0.5, 0.0], [3.0, 0.0, -1.0, 2.0]])
POINTER_SCALE = torch.tensor([[1.5, 2.0, 0.5, 3.0, 1.0], [0.5, 0.25, 2.0, 1.0, 4.0]])
POINTER_SHIFT = torch.tensor([[0.0, 1.0, -1.0, 2.0, 0.5], [-2.0, 0.0, 1.0, 0.5, 3.0]])


def _affine_pair(make_state, seed: int, *, poison_padding: bool = False):
    generator = torch.Generator().manual_seed(seed)
    spatial = torch.randn(1, 2, 4, 4, 6, 6, generator=generator) * 1.7 + 0.4
    pointer = torch.randn(1, 2, 4, 5, generator=generator) * 0.6 - 0.3
    validity = torch.tensor([[[True, True, True, False], [True, True, False, False]]])
    if poison_padding:
        spatial[~validity] = 1e6
        pointer[~validity] = -1e6
    source = make_state(spatial, pointer, validity=validity)
    role = (~source.is_conditioning).long()  # [B,O,K]
    target_spatial = spatial * SPATIAL_SCALE[role][..., None, None] + SPATIAL_SHIFT[role][..., None, None]
    target_pointer = pointer * POINTER_SCALE[role] + POINTER_SHIFT[role]
    target = source.with_continuous(
        spatial_memory=target_spatial,
        object_pointer=target_pointer,
        presence_logits=source.presence_logits.clone(),
    )
    return source, target


def test_moment_match_recovers_a_synthetic_affine_map(make_state) -> None:
    fit_pairs = [_affine_pair(make_state, seed, poison_padding=True) for seed in range(4)]
    translator = MomentMatchedCopyTranslator.fit(fit_pairs)
    source, target = _affine_pair(make_state, seed=99)
    translated = translator.translate(source)
    valid = source.validity
    # Positive scales make the per-role affine map exactly recoverable from moments.
    torch.testing.assert_close(translated.spatial_memory[valid], target.spatial_memory[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(translated.object_pointer[valid], target.object_pointer[valid], rtol=1e-4, atol=1e-4)
    # Padding keeps the Direct Copy value; poisoned fit padding did not leak into statistics.
    assert torch.equal(translated.spatial_memory[~valid], source.spatial_memory[~valid])
    assert torch.equal(translated.object_pointer[~valid], source.object_pointer[~valid])
    assert translator.statistics["spatial_source_mean"].abs().max() < 10
    assert translator.fit_summary == {
        "pairs": 4,
        "valid_records": {"conditioning": 8, "non_conditioning": 12},
    }
    metadata = translated.metadata["translation"]
    assert metadata["statistics"] == "fit_split_only"
    assert metadata["per_frame_independent"] is True


def test_moment_match_uses_only_fit_pair_statistics(make_state) -> None:
    fit_pairs = [_affine_pair(make_state, seed) for seed in range(2)]
    translator = MomentMatchedCopyTranslator.fit(fit_pairs)
    before = {key: value.clone() for key, value in translator.statistics.items()}
    test_source, test_target = _affine_pair(make_state, seed=50)
    shifted_target = test_target.with_continuous(
        spatial_memory=test_target.spatial_memory + 100,
        object_pointer=test_target.object_pointer - 100,
        presence_logits=test_target.presence_logits,
    )
    first = translator.translate(test_source)
    assert all(torch.equal(before[key], value) for key, value in translator.statistics.items())
    # Output depends on the source and the fit statistics only, never on test targets.
    refit = MomentMatchedCopyTranslator.fit(fit_pairs)
    assert torch.equal(refit.translate(test_source).spatial_memory, first.spatial_memory)
    assert not torch.allclose(first.spatial_memory, shifted_target.spatial_memory)
    # Changing the fit set changes the statistics.
    other = MomentMatchedCopyTranslator.fit([_affine_pair(make_state, seed) for seed in (7, 8)])
    assert not torch.equal(other.statistics["spatial_source_mean"], before["spatial_source_mean"])


def test_moment_match_eps_guards_constant_channels(make_state) -> None:
    source, target = _affine_pair(make_state, seed=1)
    source.spatial_memory[:, :, :, 0] = 2.0  # constant channel: sigma_source = 0
    translator = MomentMatchedCopyTranslator.fit([(source, target)], eps=1e-3)
    assert torch.count_nonzero(translator.statistics["spatial_source_std"][:, 0]) == 0
    translated = translator.translate(source)
    assert torch.isfinite(translated.spatial_memory).all()


def test_moment_match_payload_bytes_and_dtype(tmp_path: Path, make_state) -> None:
    fit_pairs = [_affine_pair(make_state, seed) for seed in range(2)]
    translator = MomentMatchedCopyTranslator.fit(fit_pairs)
    # 2 roles x (4 channels + 5 dims) x (mu_s, sigma_s, mu_t, sigma_t)
    assert translator.statistics_count() == 2 * (4 + 5) * 4
    assert translator.statistics_bytes() == translator.statistics_count() * 4
    assert translator.parameter_count() == 0
    torch.save(translator.to_payload(), tmp_path / "moments.pt")
    restored = MomentMatchedCopyTranslator.from_payload(torch.load(tmp_path / "moments.pt", weights_only=True))
    source, _ = _affine_pair(make_state, seed=3)
    bf16_source = source.with_continuous(
        spatial_memory=source.spatial_memory.to(torch.bfloat16),
        object_pointer=source.object_pointer,
        presence_logits=source.presence_logits,
    )
    expected, actual = translator.translate(bf16_source), restored.translate(bf16_source)
    assert torch.equal(actual.spatial_memory, expected.spatial_memory)
    assert actual.spatial_memory.dtype == torch.bfloat16
    assert actual.handoff_bytes() == DirectCopyTranslator(SPEC).translate(bf16_source).handoff_bytes()
    # Base+ contract size: 2 x (64 + 256) x 4 statistics.
    sam_stats = {
        key: torch.ones(2, 64 if key.startswith("spatial") else 256)
        for key in translator.statistics
    }
    sam = MomentMatchedCopyTranslator(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, sam_stats)
    assert sam.statistics_count() == 2_560
    assert sum(sam.statistics[key].numel() for key in sam.statistics if key.startswith("spatial")) == 512
    assert sum(sam.statistics[key].numel() for key in sam.statistics if key.startswith("pointer")) == 2_048


def test_moment_match_rejects_missing_roles(make_state) -> None:
    source, target = _affine_pair(make_state, seed=1)
    only_conditioning = torch.zeros_like(source.validity)
    only_conditioning[..., 0] = True
    source.validity = only_conditioning
    target.validity = only_conditioning.clone()
    with pytest.raises(ValueError, match="non_conditioning"):
        MomentMatchedCopyTranslator.fit([(source, target)])
    with pytest.raises(ValueError, match="at least one"):
        MomentMatchedCopyTranslator.fit([])


@pytest.mark.parametrize(
    "zero",
    [("spatial_memory",), ("object_pointer",), ("spatial_memory", "object_pointer")],
)
def test_component_ablation_zeroes_valid_records_only(zero: tuple[str, ...], make_state) -> None:
    source, _ = _affine_pair(make_state, seed=4)
    wrapper = ComponentAblationTranslator(DirectCopyTranslator(SPEC), zero=zero)
    translated = wrapper.translate(source)
    valid = source.validity
    for component in ("spatial_memory", "object_pointer"):
        output, original = getattr(translated, component), getattr(source, component)
        if component in zero:
            assert torch.count_nonzero(output[valid]) == 0
        else:
            assert torch.equal(output[valid], original[valid])
        assert torch.equal(output[~valid], original[~valid])
    assert translated.metadata["translation"]["ablation"]["zeroed_components"] == sorted(zero)
    assert translated.metadata["translation"]["inner"]["translator"] == "direct"
    assert torch.equal(translated.frame_indices, source.frame_indices)
    with pytest.raises(ValueError):
        ComponentAblationTranslator(DirectCopyTranslator(SPEC), zero=("presence_logits",))
    with pytest.raises(ValueError):
        ComponentAblationTranslator(DirectCopyTranslator(SPEC), zero=())


def test_component_ablation_wraps_learned_translators(make_state) -> None:
    config = SpatialTransformerConfig(
        channels=4, height=8, width=8, d_model=8, num_heads=2, pointer_dim=5, pointer_hidden_dim=6
    )
    spec = StateSpec(4, 8, 8, 5)
    inner = TransformerStateTranslator(spec, spec, config=config)
    source = make_state(torch.randn(1, 1, 2, 4, 8, 8).to(torch.bfloat16), torch.randn(1, 1, 2, 5))
    with torch.no_grad():
        translated = ComponentAblationTranslator(inner, zero=("spatial_memory",)).translate(source)
    assert translated.spatial_memory.dtype == torch.bfloat16
    assert torch.count_nonzero(translated.spatial_memory) == 0
    assert torch.equal(translated.object_pointer, source.object_pointer)  # pointer head at identity


EXPECTED = {
    # name: (spatial params, pointer params, MACs per frame in millions)
    "linear": (4_160, 65_792, 16.8),
    "residual_mlp": (16_576, 262_912, 67.1),
    "linear_local": (4_224, 262_912, 16.8),
    "base": (194_304, 262_912, 77.6),
    "no_context": (16_640, 262_912, 67.1),
    "base_mlp": (210_816, 262_912, 129.0),
    "context_only": (186_048, 262_912, 59.8),
    "no_attn_context": (160_768, 262_912, 52.4),
    "res8": (116_480, 262_912, 24.9),
    "res16": (116_480, 262_912, 61.9),
    "res32": (116_480, 262_912, 398.5),
    "grid8": (378_624, 262_912, 41.4),
    "grid32": (194_304, 262_912, 411.0),
    "depth1": (160_832, 262_912, 60.8),
    "depth4": (261_248, 262_912, 111.1),
    "heads2": (194_304, 262_912, 77.6),
    "heads8": (194_304, 262_912, 77.6),
    "no_residual": (194_240, 262_912, 77.6),
    "mlp256": (33_152, 262_912, 134.2),
    "mlp_param": (194_273, 262_912, 789.1),
    "no_pos": (177_920, 262_912, 77.6),
    "sincos_pos": (177_920, 262_912, 77.6),
    "ptr128": (194_304, 65_920, 77.6),
    "no_cos": (194_304, 262_912, 77.6),
}


def test_registry_covers_every_plan_preset() -> None:
    assert set(preset_names(learned=True)) == set(EXPECTED)
    assert set(PRESETS) == set(EXPECTED) | {"direct", "moment_match"}
    assert PRESETS["no_cos"].training == {"lambda_cos": 0.0}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_learned_preset_builds_forwards_and_matches_the_plan(name: str, make_state) -> None:
    torch.manual_seed(0)
    translator = build_translator(name, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    spatial_params, pointer_params, macs = EXPECTED[name]
    breakdown = translator.parameter_breakdown()
    assert (breakdown["spatial"], breakdown["pointer"]) == (spatial_params, pointer_params)
    assert round(translator.macs_per_frame() / 1e6, 1) == macs
    assert translator.preset == name
    validity = torch.tensor([[[True, False]]])
    source = make_state(
        torch.randn(1, 1, 2, 64, 64, 64).to(torch.bfloat16), torch.randn(1, 1, 2, 256), validity=validity
    )
    with torch.no_grad():
        translated = translator.translate(source)
    assert translated.spatial_memory.shape == source.spatial_memory.shape
    assert translated.spatial_memory.dtype == torch.bfloat16
    assert translated.object_pointer.dtype == torch.float32
    assert torch.isfinite(translated.spatial_memory.float()).all()
    assert torch.equal(translated.spatial_memory[~validity], source.spatial_memory[~validity])


def test_build_translator_overrides_and_errors(make_state) -> None:
    custom = build_translator("base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, num_layers=3)
    assert custom.config.num_layers == 3 and custom.preset == "base"
    explicit = build_translator(
        "base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, config=SpatialTransformerConfig(patch_size=8)
    )
    assert explicit.config.patch_size == 8
    assert isinstance(build_translator("linear", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC), LinearStateTranslator)
    assert isinstance(build_translator("residual_mlp", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC), ResidualMLPStateTranslator)
    assert isinstance(build_translator("direct", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC), DirectCopyTranslator)
    with pytest.raises(ValueError, match="unknown preset"):
        build_translator("huge", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    with pytest.raises(ValueError, match="unknown overrides"):
        build_translator("base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, depth=3)
    with pytest.raises(ValueError, match="unknown overrides"):
        build_translator("linear", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, hidden_dim=3)
    with pytest.raises(ValueError, match="fit_pairs"):
        build_translator("moment_match", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    fit_pairs = [_affine_pair(make_state, seed) for seed in range(2)]
    moment = build_translator("moment_match", SPEC, SPEC, fit_pairs=fit_pairs)
    assert isinstance(moment, MomentMatchedCopyTranslator) and moment.preset == "moment_match"
    with pytest.raises(ValueError, match="fit_pairs"):
        build_translator("base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC, fit_pairs=fit_pairs)
