"""Design contract of the two translators (docs/translator_model_design.md)."""

from __future__ import annotations

import pytest
import torch

from vos_memory_inspector.models import (
    SAM21_MEMORY_SPEC,
    ResidualMLPStateTranslator,
    TransformerStateTranslator,
    TranslatorConfig,
)
from vos_memory_inspector.models._stitched import (
    PAYLOAD_SCHEMA,
    RIDGE_SCHEMA,
    SpatialReductionAttention,
    sine_position_encoding,
)
from vos_memory_inspector.models.registry import PRESETS, build_translator
from vos_memory_inspector.state_schema import StateSpec

MODELS = (ResidualMLPStateTranslator, TransformerStateTranslator)
SMALL_SPEC = StateSpec(feature_channels=8, height=8, width=8, pointer_dim=12)
# Spatial and pointer RMS measured on stored states of each SAM 2.1 size
# (gpu-server /workspace/states, case pumavos_pimples_obj1_switch405).
SCALES = {"small": (1.03, 0.25), "base_plus": (0.76, 0.58), "large": (1.39, 0.62)}
PAIRS = (("small", "base_plus"), ("base_plus", "large"), ("small", "large"))


def _records(spec: StateSpec, records: int, *, scale=(1.0, 1.0), seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    spatial = scale[0] * torch.randn(
        1, 1, records, spec.feature_channels, spec.height, spec.width, generator=generator
    )
    pointer = scale[1] * torch.randn(1, 1, records, spec.pointer_dim, generator=generator)
    return spatial, pointer


def _perturb(model: torch.nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))


def _ridge(spec: StateSpec, seed: int = 3) -> dict:
    generator = torch.Generator().manual_seed(seed)
    channels, dim = spec.feature_channels, spec.pointer_dim
    return {
        "schema_version": RIDGE_SCHEMA,
        "source_spec": spec.to_dict(),
        "target_spec": spec.to_dict(),
        # Row-vector convention of the ridge artifact: weight is [in, out].
        "spatial": {
            "weight": torch.randn(channels, channels, generator=generator),
            "bias": torch.randn(channels, generator=generator),
        },
        "pointer": {
            "weight": torch.randn(dim, dim, generator=generator),
            "bias": torch.randn(dim, generator=generator),
        },
        "provenance": {"artifact_sha256": "0" * 64, "data_identity": {"test": True}},
    }


@pytest.mark.parametrize("model_type", MODELS)
def test_untrained_model_is_the_identity_and_keeps_shape(model_type):
    model = model_type(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    spatial, pointer = _records(SAM21_MEMORY_SPEC, 2)
    out_spatial, out_pointer = model.translate_tensors(spatial, pointer)
    assert out_spatial.shape == spatial.shape and out_pointer.shape == pointer.shape
    assert torch.equal(out_spatial, spatial) and torch.equal(out_pointer, pointer)


@pytest.mark.parametrize("model_type", MODELS)
def test_ridge_start_equals_the_closed_form_linear_baseline(model_type):
    model = model_type(SMALL_SPEC, SMALL_SPEC)
    ridge = _ridge(SMALL_SPEC)
    model.initialize_from_ridge(ridge)
    spatial, pointer = _records(SMALL_SPEC, 3)
    out_spatial, out_pointer = model.translate_tensors(spatial, pointer)
    expected_spatial = (
        torch.einsum("bokchw,cd->bokdhw", spatial, ridge["spatial"]["weight"])
        + ridge["spatial"]["bias"][None, None, None, :, None, None]
    )
    expected_pointer = pointer @ ridge["pointer"]["weight"] + ridge["pointer"]["bias"]
    assert torch.allclose(out_spatial, expected_spatial, atol=1e-5)
    assert torch.allclose(out_pointer, expected_pointer, atol=1e-5)
    assert model.ridge_initialization == ridge["provenance"]
    assert model.architecture_summary()["start"] == "ridge"


@pytest.mark.parametrize("model_type", MODELS)
def test_ridge_start_rejects_other_specs_and_missing_provenance(model_type):
    model = model_type(SMALL_SPEC, SMALL_SPEC)
    other = _ridge(StateSpec(feature_channels=8, height=8, width=8, pointer_dim=16))
    with pytest.raises(ValueError, match="specs mismatch"):
        model.initialize_from_ridge(other)
    unsigned = _ridge(SMALL_SPEC)
    unsigned["provenance"] = {}
    with pytest.raises(ValueError, match="hash and data identity"):
        model.initialize_from_ridge(unsigned)


@pytest.mark.parametrize("model_type", MODELS)
def test_last_layer_of_each_branch_receives_gradient_at_the_start(model_type):
    model = model_type(SMALL_SPEC, SMALL_SPEC)
    spatial, pointer = _records(SMALL_SPEC, 2)
    out_spatial, out_pointer = model.translate_tensors(spatial, pointer)
    (out_spatial.square().mean() + out_pointer.square().mean()).backward()
    block = model.spatial.blocks[0]
    assert float(block.mlp.fc2.weight.grad.abs().sum()) > 0
    assert float(model.spatial.stitch.linear.weight.grad.abs().sum()) > 0
    assert float(model.pointer.linear.weight.grad.abs().sum()) > 0
    if model.config.attention:
        assert float(block.attn.proj.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("model_type", MODELS)
def test_records_are_translated_independently(model_type):
    model = model_type(SMALL_SPEC, SMALL_SPEC).eval()
    _perturb(model)
    spatial, pointer = _records(SMALL_SPEC, 4)
    with torch.no_grad():
        full_spatial, full_pointer = model.translate_tensors(spatial, pointer)
        changed = spatial.clone()
        changed[:, :, 0] += 5.0
        changed_pointer = pointer.clone()
        changed_pointer[:, :, 0] -= 3.0
        other_spatial, other_pointer = model.translate_tensors(changed, changed_pointer)
        alone_spatial, alone_pointer = model.translate_tensors(spatial[:, :, 2:3], pointer[:, :, 2:3])
    # Changing record 0 leaves records 1..3 untouched.
    assert torch.equal(full_spatial[:, :, 1:], other_spatial[:, :, 1:])
    assert torch.equal(full_pointer[:, :, 1:], other_pointer[:, :, 1:])
    assert not torch.equal(full_spatial[:, :, 0], other_spatial[:, :, 0])
    # A record translated alone equals the same record translated in a batch.
    assert torch.allclose(full_spatial[:, :, 2:3], alone_spatial, atol=1e-5)
    assert torch.allclose(full_pointer[:, :, 2:3], alone_pointer, atol=1e-6)


def test_mlp_model_is_position_wise_and_transformer_model_is_not():
    spatial, pointer = _records(SMALL_SPEC, 1)
    changed = spatial.clone()
    changed[..., 0, 0] += 4.0  # one position of the grid
    results = {}
    for model_type in MODELS:
        model = model_type(SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(sr_ratio=2)).eval()
        _perturb(model)
        with torch.no_grad():
            before, _ = model.translate_tensors(spatial, pointer)
            after, _ = model.translate_tensors(changed, pointer)
        moved = (before - after).abs().amax(dim=3)[0, 0, 0] > 1e-7  # [H, W]
        results[model_type.name] = int(moved.sum())
    assert results["residual_mlp"] == 1
    assert results["transformer"] == SMALL_SPEC.height * SMALL_SPEC.width


def test_models_differ_only_in_the_attention_sublayer():
    mlp = ResidualMLPStateTranslator(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    transformer = TransformerStateTranslator(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    mlp_keys, transformer_keys = set(mlp.state_dict()), set(transformer.state_dict())
    assert mlp_keys < transformer_keys
    extra = transformer_keys - mlp_keys
    assert all(".attn." in key or ".norm_attn." in key for key in extra)
    assert {key: mlp.state_dict()[key].shape for key in mlp_keys} == {
        key: transformer.state_dict()[key].shape for key in mlp_keys
    }
    assert mlp.config == TranslatorConfig(attention=False)
    assert transformer.config == TranslatorConfig(attention=True)

    # With the attention output projection at zero the Transformer model is the MLP model.
    _perturb(transformer)
    with torch.no_grad():
        for block in transformer.spatial.blocks:
            block.attn.proj.weight.zero_()
            block.attn.proj.bias.zero_()
    mlp.load_state_dict({key: transformer.state_dict()[key] for key in mlp_keys})
    spatial, pointer = _records(SAM21_MEMORY_SPEC, 2)
    with torch.no_grad():
        a_spatial, a_pointer = mlp.translate_tensors(spatial, pointer)
        b_spatial, b_pointer = transformer.translate_tensors(spatial, pointer)
    assert torch.equal(a_spatial, b_spatial) and torch.equal(a_pointer, b_pointer)


def test_parameter_and_mac_counts_match_the_design_note():
    mlp = build_translator("residual_mlp", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    transformer = build_translator("transformer", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    matched = build_translator("residual_mlp_matched", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    assert mlp.parameter_breakdown() == {"spatial": 37_376, "pointer": 65_792, "total": 103_168}
    assert transformer.parameter_breakdown() == {
        "spatial": 119_872, "pointer": 65_792, "total": 185_664,
    }
    assert mlp.spatial.parameter_breakdown() == {
        "stitch": 4_160, "feed_forward": 33_216, "attention": 0, "total": 37_376,
    }
    assert transformer.spatial.parameter_breakdown()["attention"] == 82_496
    assert mlp.macs_per_frame() == 150_994_944
    assert transformer.macs_per_frame() == 337_641_472
    assert mlp.pointer_macs_per_record() == transformer.pointer_macs_per_record() == 65_536
    # The control is within 0.1% of the Transformer model's spatial parameters.
    assert matched.parameter_breakdown()["spatial"] == 119_936
    for model in (mlp, transformer, matched):
        counted = sum(parameter.numel() for parameter in model.parameters())
        assert counted == model.parameter_breakdown()["total"]


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("source,target", PAIRS)
def test_same_structure_serves_all_three_pairs(model_type, source, target):
    """Small, Base+ and Large states share one spec; only their scale differs."""

    model = model_type(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    reference = model_type(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    assert model.config == reference.config
    assert {k: v.shape for k, v in model.state_dict().items()} == {
        k: v.shape for k, v in reference.state_dict().items()
    }
    _perturb(model)
    spatial, pointer = _records(SAM21_MEMORY_SPEC, 2, scale=SCALES[source])
    with torch.no_grad():
        out_spatial, out_pointer = model.translate_handoff_tensors(
            spatial.to(torch.bfloat16), pointer
        )
    assert out_spatial.shape == (1, 1, 2, 64, 64, 64) and out_spatial.dtype == torch.bfloat16
    assert out_pointer.shape == (1, 1, 2, 256) and out_pointer.dtype == torch.float32
    assert torch.isfinite(out_spatial.float()).all() and torch.isfinite(out_pointer).all()
    assert SCALES[target]  # the target only changes the weights, never the structure


@pytest.mark.parametrize("model_type", MODELS)
def test_padding_records_keep_the_source_and_never_enter_the_model(model_type):
    model = model_type(SMALL_SPEC, SMALL_SPEC).eval()
    _perturb(model)
    spatial, pointer = _records(SMALL_SPEC, 3)
    spatial[:, :, 2] = float("nan")
    pointer[:, :, 2] = float("nan")
    validity = torch.tensor([[[True, True, False]]])
    with torch.no_grad():
        out_spatial, out_pointer = model.translate_tensors(spatial, pointer, validity)
    assert torch.isfinite(out_spatial[:, :, :2]).all() and torch.isfinite(out_pointer[:, :, :2]).all()
    assert torch.isnan(out_spatial[:, :, 2]).all() and torch.isnan(out_pointer[:, :, 2]).all()


@pytest.mark.parametrize("model_type", MODELS)
def test_payload_round_trip_is_weights_only_loadable(model_type, tmp_path):
    model = model_type(SMALL_SPEC, SMALL_SPEC, preset=model_type.name)
    model.initialize_from_ridge(_ridge(SMALL_SPEC))
    _perturb(model)
    path = tmp_path / "model.pt"
    torch.save(model.to_payload(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == PAYLOAD_SCHEMA and payload["translator"] == model_type.name
    restored = model_type.from_payload(payload)
    assert not restored.training
    assert restored.config == model.config and restored.preset == model_type.name
    assert restored.ridge_initialization == model.ridge_initialization
    spatial, pointer = _records(SMALL_SPEC, 2)
    with torch.no_grad():
        expected = model.eval().translate_tensors(spatial, pointer)
        actual = restored.translate_tensors(spatial, pointer)
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])


def test_payloads_are_bound_to_their_model_and_old_schemas_are_rejected():
    mlp = ResidualMLPStateTranslator(SMALL_SPEC, SMALL_SPEC)
    with pytest.raises(ValueError, match="not 'transformer'"):
        TransformerStateTranslator.from_payload(mlp.to_payload())
    for schema in (
        "cmmt.residual_mlp_translator.v2",
        "cmmt.residual_mlp_translator.v3",
        "cmmt.spatial_context_transformer_translator.v1",
        "cmmt.spatial_context_transformer_translator.v2",
    ):
        with pytest.raises(ValueError, match="before the translator redesign"):
            ResidualMLPStateTranslator.from_payload({"schema_version": schema})
        with pytest.raises(ValueError, match="before the translator redesign"):
            TransformerStateTranslator.from_payload({"schema_version": schema})


def test_payload_with_an_incomplete_config_is_rejected():
    payload = ResidualMLPStateTranslator(SMALL_SPEC, SMALL_SPEC).to_payload()
    payload["config"] = {k: v for k, v in payload["config"].items() if k != "mlp_ratio"}
    with pytest.raises(ValueError, match="missing fields: \\['mlp_ratio'\\]"):
        ResidualMLPStateTranslator.from_payload(payload)


def test_attention_switch_belongs_to_the_class_not_to_the_caller():
    mlp = ResidualMLPStateTranslator(
        SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(attention=True)
    )
    assert mlp.config.attention is False
    assert all(block.attn is None for block in mlp.spatial.blocks)
    with pytest.raises(ValueError, match="unknown overrides"):
        build_translator("residual_mlp", SMALL_SPEC, SMALL_SPEC, attention=True)
    assert PRESETS["residual_mlp"].kind == "residual_mlp"
    assert PRESETS["transformer"].kind == "transformer"
    assert "mlp_context" not in PRESETS and "base" not in PRESETS


def test_position_sampling_is_only_offered_without_attention():
    mlp = ResidualMLPStateTranslator(SMALL_SPEC, SMALL_SPEC)
    transformer = TransformerStateTranslator(SMALL_SPEC, SMALL_SPEC)
    assert mlp.supports_position_sampling and not transformer.supports_position_sampling
    _perturb(mlp)
    spatial, pointer = _records(SMALL_SPEC, 1)
    with torch.no_grad():
        full, _ = mlp.translate_tensors(spatial, pointer)
        vectors = spatial[0, 0, 0].movedim(0, -1).reshape(-1, SMALL_SPEC.feature_channels)
        sampled = mlp._feature_head(vectors)
    expected = full[0, 0, 0].movedim(0, -1).reshape(-1, SMALL_SPEC.feature_channels)
    assert torch.allclose(sampled, expected, atol=1e-6)
    with pytest.raises(ValueError, match="whole records"):
        transformer._feature_head(vectors)


def test_config_validation_and_grid_requirements():
    with pytest.raises(ValueError, match="num_layers"):
        TranslatorConfig(num_layers=0)
    with pytest.raises(ValueError, match="mlp_ratio"):
        TranslatorConfig(mlp_ratio=0.0)
    with pytest.raises(ValueError, match="unknown TranslatorConfig fields"):
        TranslatorConfig.from_dict({"hidden_dim": 128})
    with pytest.raises(ValueError, match="divisible by sr_ratio"):
        TransformerStateTranslator(SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(sr_ratio=3))
    with pytest.raises(ValueError, match="positive integer"):
        ResidualMLPStateTranslator(SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(mlp_ratio=0.3))
    wider = StateSpec(feature_channels=8, height=16, width=16, pointer_dim=12)
    with pytest.raises(ValueError, match="does not resample"):
        ResidualMLPStateTranslator(SMALL_SPEC, wider)
    deep = TransformerStateTranslator(SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(num_layers=2))
    assert len(deep.spatial.blocks) == 2


def test_different_channel_and_pointer_sizes_are_stitched_without_resampling():
    source = StateSpec(feature_channels=8, height=8, width=8, pointer_dim=12)
    target = StateSpec(feature_channels=16, height=8, width=8, pointer_dim=20)
    for model_type in MODELS:
        model = model_type(source, target)
        spatial, pointer = _records(source, 2)
        out_spatial, out_pointer = model.translate_tensors(spatial, pointer)
        assert out_spatial.shape == (1, 1, 2, 16, 8, 8) and out_pointer.shape == (1, 1, 2, 20)


def test_sine_position_encoding_separates_positions_and_shares_a_frame():
    fine = sine_position_encoding(8, 8, 16)
    coarse = sine_position_encoding(4, 4, 16)
    assert fine.shape == (64, 16) and coarse.shape == (16, 16)
    assert torch.unique(fine, dim=0).shape[0] == 64
    assert float(fine.abs().max()) <= 1.0
    # A coarse cell centre lies between its four fine cells, in the same coordinates.
    cell = fine.reshape(8, 8, 16)[:2, :2].reshape(4, 16)
    assert float((coarse[0] - cell.mean(0)).abs().max()) < 0.5
    with pytest.raises(ValueError, match="divisible by 4"):
        sine_position_encoding(4, 4, 6)


def test_attention_position_buffers_are_not_saved_and_reduction_is_optional():
    attention = SpatialReductionAttention(8, (8, 8), num_heads=2, sr_ratio=1)
    assert attention.sr is None and attention.reduced_grid == (8, 8)
    reduced = SpatialReductionAttention(8, (8, 8), num_heads=1, sr_ratio=4)
    assert reduced.reduced_grid == (2, 2)
    assert not any("pos_" in key for key in reduced.state_dict())
    with pytest.raises(ValueError, match="divisible by num_heads"):
        SpatialReductionAttention(8, (8, 8), num_heads=3, sr_ratio=1)


@pytest.mark.parametrize("model_type", MODELS)
def test_a_few_updates_lower_the_loss_on_a_nonlinear_target(model_type):
    torch.manual_seed(0)
    model = model_type(SMALL_SPEC, SMALL_SPEC)
    spatial, pointer = _records(SMALL_SPEC, 6)
    target_spatial = spatial + 0.5 * torch.tanh(2.0 * spatial.roll(1, dims=3))
    target_pointer = 0.5 * pointer + 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        out_spatial, out_pointer = model.translate_tensors(spatial, pointer)
        loss = (out_spatial - target_spatial).square().mean() + (out_pointer - target_pointer).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.7 * losses[0]
    # The nonlinear branch has moved away from its zero start.
    assert float(model.spatial.blocks[0].mlp.fc2.weight.abs().sum()) > 0
