"""Task 4: shared learned-translator handoff path (valid gather/scatter, dtype, payloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from vos_memory_inspector.models._stitched import PAYLOAD_SCHEMA
from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.transformer_translator import (
    SAM21_MEMORY_SPEC,
    TransformerStateTranslator,
    TranslatorConfig,
    build_translator,
)
from vos_memory_inspector.translator_training import fit_gradient_translator
from vos_memory_inspector.translators import (
    DirectCopyTranslator,
    LearnedComponentPolicyTranslator,
    LinearStateTranslator,
    ResidualMLPStateTranslator,
)


SMALL_SPEC = StateSpec(8, 16, 16, 12)
SMALL = TranslatorConfig(num_heads=2, sr_ratio=2)
VALIDITY = torch.tensor([[[True, True, False], [True, False, False]]])  # 3 of 6 valid


def _randomize(module: torch.nn.Module, seed: int = 5) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)


def _transformer(**options) -> TransformerStateTranslator:
    torch.manual_seed(0)
    config = TranslatorConfig(num_heads=2, sr_ratio=2, **options)
    translator = TransformerStateTranslator(SMALL_SPEC, SMALL_SPEC, config=config).eval()
    _randomize(translator)
    return translator


def _residual_mlp(**options) -> ResidualMLPStateTranslator:
    torch.manual_seed(0)
    translator = ResidualMLPStateTranslator(
        SMALL_SPEC, SMALL_SPEC, config=TranslatorConfig(mlp_ratio=1.5, **options)
    ).eval()
    _randomize(translator)
    return translator


def _linear() -> LinearStateTranslator:
    torch.manual_seed(0)
    translator = LinearStateTranslator(SMALL_SPEC, SMALL_SPEC).eval()
    _randomize(translator)
    return translator


TRANSLATORS = {"transformer": _transformer, "residual_mlp": _residual_mlp, "linear": _linear}
MODELS = ("transformer", "residual_mlp")


def _inputs(dtype: torch.dtype = torch.float32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    spatial = torch.randn(1, 2, 3, 8, 16, 16, generator=generator).to(dtype)
    pointer = torch.randn(1, 2, 3, 12, generator=generator)
    return spatial, pointer


@pytest.mark.parametrize("kind", MODELS)
def test_valid_output_is_the_model_applied_to_the_gathered_records(kind: str, make_state) -> None:
    translator = TRANSLATORS[kind]()
    spatial, pointer = _inputs()
    with torch.no_grad():
        translated = translator.translate(make_state(spatial, pointer, validity=VALIDITY))
        expected_spatial = translator.spatial(spatial[VALIDITY])  # [N, C, H, W]
        expected_pointer = translator.pointer(pointer[VALIDITY])
    torch.testing.assert_close(translated.spatial_memory[VALIDITY], expected_spatial)
    torch.testing.assert_close(translated.object_pointer[VALIDITY], expected_pointer)


@pytest.mark.parametrize("kind", sorted(TRANSLATORS))
def test_nan_padding_never_reaches_valid_outputs(kind: str, make_state) -> None:
    translator = TRANSLATORS[kind]()
    spatial, pointer = _inputs()
    poisoned_spatial, poisoned_pointer = spatial.clone(), pointer.clone()
    poisoned_spatial[~VALIDITY] = float("nan")
    poisoned_pointer[~VALIDITY] = float("nan")
    with torch.no_grad():
        clean = translator.translate(make_state(spatial, pointer, validity=VALIDITY))
        poisoned = translator.translate(
            make_state(poisoned_spatial, poisoned_pointer, validity=VALIDITY)
        )
    assert torch.equal(poisoned.spatial_memory[VALIDITY], clean.spatial_memory[VALIDITY])
    assert torch.equal(poisoned.object_pointer[VALIDITY], clean.object_pointer[VALIDITY])
    assert not torch.isnan(poisoned.spatial_memory[VALIDITY]).any()
    # Padding is an exact source copy (NaN == NaN only under equal_nan).
    torch.testing.assert_close(
        poisoned.spatial_memory[~VALIDITY], poisoned_spatial[~VALIDITY], rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(
        poisoned.object_pointer[~VALIDITY], poisoned_pointer[~VALIDITY], rtol=0, atol=0, equal_nan=True
    )
    policy = poisoned.metadata["translation"]["invalid_records"]
    assert policy == {
        "translated": "valid_records_only",
        "spatial_memory": "source_copy",
        "object_pointer": "source_copy",
    }


@pytest.mark.parametrize("kind", sorted(TRANSLATORS))
def test_forward_sees_only_valid_records(kind: str, make_state) -> None:
    translator = TRANSLATORS[kind]()
    spatial_module = translator.feature if kind == "linear" else translator.spatial
    pointer_module = translator.pointer
    seen: dict[str, list[int]] = {"spatial": [], "pointer": []}
    hooks = [
        spatial_module.register_forward_pre_hook(
            lambda _m, inputs: seen["spatial"].append(inputs[0].shape[0])
        ),
        pointer_module.register_forward_pre_hook(
            lambda _m, inputs: seen["pointer"].append(inputs[0].shape[0])
        ),
    ]
    spatial, pointer = _inputs()
    with torch.no_grad():
        translator.translate(make_state(spatial, pointer, validity=VALIDITY))
        assert seen == {"spatial": [3], "pointer": [3]}
        empty = torch.zeros_like(VALIDITY)
        translated = translator.translate(make_state(spatial, pointer, validity=empty))
    for hook in hooks:
        hook.remove()
    assert seen == {"spatial": [3], "pointer": [3]}  # no call for zero valid records
    assert torch.equal(translated.spatial_memory, spatial)
    assert torch.equal(translated.object_pointer, pointer)


def test_zero_fill_padding_when_shapes_differ(make_state) -> None:
    torch.manual_seed(0)
    translator = ResidualMLPStateTranslator(SMALL_SPEC, StateSpec(6, 16, 16, 5)).eval()
    _randomize(translator)
    spatial, pointer = _inputs()
    with torch.no_grad():
        translated = translator.translate(make_state(spatial, pointer, validity=VALIDITY))
    assert translated.spatial_memory.shape == (1, 2, 3, 6, 16, 16)
    assert torch.count_nonzero(translated.spatial_memory[~VALIDITY]) == 0
    assert torch.count_nonzero(translated.object_pointer[~VALIDITY]) == 0
    assert translated.metadata["translation"]["invalid_records"]["spatial_memory"] == "zero_fill"


@pytest.mark.parametrize("kind", MODELS)
def test_bf16_handoff_keeps_source_dtype_and_direct_copy_bytes(kind: str, make_state) -> None:
    translator = TRANSLATORS[kind](output_dtype="source")
    spatial, pointer = _inputs(torch.bfloat16)
    state = make_state(spatial, pointer, validity=VALIDITY)
    with torch.no_grad():
        translated = translator.translate(state)
    direct = DirectCopyTranslator(SMALL_SPEC).translate(state)
    assert translated.spatial_memory.dtype == torch.bfloat16
    assert translated.object_pointer.dtype == torch.float32
    assert translated.handoff_bytes() == direct.handoff_bytes() == state.handoff_bytes()
    metadata = translated.metadata["translation"]
    assert metadata["per_frame_independent"] is True
    assert metadata["output_dtype"] == {
        "policy": "source",
        "spatial_memory": "bfloat16",
        "object_pointer": "float32",
    }
    assert translated.metadata["sentinel"] == "preserve"
    assert torch.equal(translated.frame_indices, state.frame_indices)
    assert torch.equal(translated.presence_logits, state.presence_logits)


@pytest.mark.parametrize("kind", MODELS)
def test_float32_output_policy_is_available_explicitly(kind: str, make_state) -> None:
    translator = TRANSLATORS[kind](output_dtype="float32")
    spatial, pointer = _inputs(torch.bfloat16)
    state = make_state(spatial, pointer, validity=VALIDITY)
    with torch.no_grad():
        translated = translator.translate(state)
    assert translator.output_dtype == "float32"
    assert translated.spatial_memory.dtype == torch.float32
    assert translated.handoff_bytes() > DirectCopyTranslator(SMALL_SPEC).translate(state).handoff_bytes()


@pytest.mark.parametrize("kind", MODELS)
def test_translate_tensors_keeps_gradients_in_float32(kind: str) -> None:
    translator = TRANSLATORS[kind]().train()
    spatial, pointer = _inputs(torch.bfloat16)
    spatial_hat, pointer_hat = translator.translate_tensors(spatial, pointer, VALIDITY)
    assert spatial_hat.dtype == pointer_hat.dtype == torch.float32
    assert spatial_hat.shape == spatial.shape and pointer_hat.shape == pointer.shape
    loss = spatial_hat[VALIDITY].square().mean() + pointer_hat[VALIDITY].square().mean()
    loss.backward()
    assert translator.spatial.stitch.linear.weight.grad.abs().max() > 0
    assert translator.spatial.blocks[0].mlp.fc2.weight.grad.abs().max() > 0
    assert translator.pointer.linear.weight.grad.abs().max() > 0
    # validity=None treats every record as valid.
    everything, _ = translator.translate_tensors(spatial, pointer)
    assert not torch.equal(everything[~VALIDITY], spatial[~VALIDITY].float())


def test_translate_tensors_validates_inputs() -> None:
    translator = _transformer()
    spatial, pointer = _inputs()
    with pytest.raises(ValueError, match="B,O,K,C,H,W"):
        translator.translate_tensors(spatial[0], pointer)
    with pytest.raises(ValueError, match="source spec"):
        translator.translate_tensors(spatial[..., :8], pointer)
    with pytest.raises(ValueError, match="pointer dim"):
        translator.translate_tensors(spatial, pointer[..., :4])
    with pytest.raises(ValueError, match="validity"):
        translator.translate_tensors(spatial, pointer, VALIDITY.int())


@pytest.mark.parametrize("kind", MODELS)
def test_payload_round_trip_with_weights_only(kind: str, tmp_path: Path, make_state) -> None:
    translator = TRANSLATORS[kind](output_dtype="float32")
    translator.preset = "small"
    payload = translator.to_payload()
    assert payload["schema_version"] == PAYLOAD_SCHEMA == "cmmt.stitched_translator.v1"
    path = tmp_path / "translator.pt"
    torch.save(payload, path)
    restored = type(translator).from_payload(torch.load(path, weights_only=True))
    assert restored.config == translator.config and restored.preset == "small"
    assert restored.output_dtype == "float32"
    spatial, pointer = _inputs(torch.bfloat16)
    state = make_state(spatial, pointer, validity=VALIDITY)
    with torch.no_grad():
        expected, actual = translator.translate(state), restored.translate(state)
    assert torch.equal(actual.spatial_memory, expected.spatial_memory)
    assert torch.equal(actual.object_pointer, expected.object_pointer)

    broken = dict(payload)
    broken["config"] = {**payload["config"], "hidden_dim": 7}
    with pytest.raises(ValueError, match="unknown"):
        type(translator).from_payload(broken)
    with pytest.raises(ValueError, match="unsupported"):
        type(translator).from_payload({**payload, "schema_version": "v0"})


def test_component_policy_accepts_the_transformer(make_state) -> None:
    translator = _transformer()
    spatial, pointer = _inputs(torch.bfloat16)
    state = make_state(spatial, pointer, validity=VALIDITY)
    policy = LearnedComponentPolicyTranslator(translator, learned_components=("spatial_memory",))
    with torch.no_grad():
        translated = policy.translate(state)
        learned = translator.translate(state)
    assert torch.equal(translated.spatial_memory, learned.spatial_memory)
    assert torch.equal(translated.object_pointer, pointer)
    assert not torch.equal(learned.object_pointer[VALIDITY], pointer[VALIDITY])
    assert policy.parameter_count() == translator.parameter_count()


def test_spec_mismatch_and_sampled_training_fail_closed(make_state) -> None:
    with pytest.raises(ValueError, match="grids must be equal"):
        TransformerStateTranslator(SMALL_SPEC, StateSpec(8, 8, 8, 12), config=SMALL)
    translator = _transformer()
    other = make_state(torch.randn(1, 1, 2, 8, 8, 8), torch.randn(1, 1, 2, 12))
    with pytest.raises(ValueError, match="does not match"):
        translator.translate(other)

    spatial, pointer = _inputs()
    source = make_state(spatial, pointer, validity=VALIDITY)
    target = source.with_continuous(
        spatial_memory=spatial * 1.1, object_pointer=pointer - 0.1, presence_logits=source.presence_logits
    )
    with pytest.raises(ValueError, match="spatial_samples_per_pair"):
        fit_gradient_translator(translator, [(source, target)], epochs=1, device="cpu", spatial_samples_per_pair=3)
    # Whole-frame training through the legacy trainer still works.
    history = fit_gradient_translator(translator, [(source, target)], epochs=2, learning_rate=1e-3, device="cpu")
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(history)))
    # The model without attention also accepts sampled positions.
    mlp = _residual_mlp().train()
    history = fit_gradient_translator(
        mlp, [(source, target)], epochs=2, learning_rate=1e-3, device="cpu", spatial_samples_per_pair=3
    )
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(history)))


@pytest.mark.parametrize("kind", MODELS)
def test_whole_frame_training_reduces_loss(kind: str) -> None:
    """translate_tensors is enough to train either model end to end."""

    torch.manual_seed(0)
    translator = build_translator(kind, SMALL_SPEC, SMALL_SPEC).train()
    generator = torch.Generator().manual_seed(2)
    spatial = torch.randn(1, 2, 3, 8, 16, 16, generator=generator)
    pointer = torch.randn(1, 2, 3, 12, generator=generator)
    target_spatial = spatial * 0.5 + 0.3
    target_pointer = pointer * 1.5
    optimizer = torch.optim.Adam(translator.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        spatial_hat, pointer_hat = translator.translate_tensors(spatial, pointer, VALIDITY)
        loss = F.mse_loss(spatial_hat[VALIDITY], target_spatial[VALIDITY]) + F.mse_loss(
            pointer_hat[VALIDITY], target_pointer[VALIDITY]
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0]


def test_package_exports_new_symbols() -> None:
    import vos_memory_inspector as package

    assert package.TransformerStateTranslator is TransformerStateTranslator
    assert package.build_translator is build_translator
    for name in package.__all__:
        assert getattr(package, name) is not None, name
