"""Model separation must keep legacy imports, artifacts and entrypoints usable."""

import pickle
import subprocess
import sys

import pytest
import torch

import vos_memory_inspector as package
from vos_memory_inspector.models import (
    MLPContextStateTranslator,
    ResidualMLPStateTranslator,
    SpatialTransformerConfig,
    TransformerStateTranslator,
    build_translator,
)
from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.translator_artifact import load_learned_translator


SPEC = StateSpec(4, 8, 8, 6)
CONFIG = SpatialTransformerConfig(
    channels=4, height=8, width=8, d_model=4, num_heads=2, num_layers=1,
    patch_size=2, pointer_dim=6, pointer_hidden_dim=9, fusion_hidden_dim=7,
)


def test_old_import_and_pickle_paths_resolve_to_model_classes():
    from vos_memory_inspector.translators import ResidualMLPStateTranslator as old_mlp
    from vos_memory_inspector.transformer_translator import TransformerStateTranslator as old_transformer

    assert old_mlp is package.ResidualMLPStateTranslator is ResidualMLPStateTranslator
    assert old_transformer is package.TransformerStateTranslator is TransformerStateTranslator
    assert package.MLPContextStateTranslator is MLPContextStateTranslator
    for module, name, expected in (
        ("translators", "ResidualMLPStateTranslator", ResidualMLPStateTranslator),
        ("transformer_translator", "TransformerStateTranslator", TransformerStateTranslator),
    ):
        # Protocol 0 GLOBAL references used in older whole-model pickles.
        reference = f"cvos_memory_inspector.{module}\n{name}\n.".encode()
        assert pickle.loads(reference) is expected


def test_factory_uses_separate_model_modules():
    mlp = build_translator("residual_mlp", SPEC, SPEC)
    transformer = build_translator("base", SPEC, SPEC, config=CONFIG)
    context = build_translator("mlp_context", SPEC, SPEC, config=CONFIG, local_fusion="additive_mlp")
    assert type(mlp) is ResidualMLPStateTranslator
    assert type(transformer) is TransformerStateTranslator
    assert type(context) is MLPContextStateTranslator
    assert len({type(model).__module__ for model in (mlp, transformer, context)}) == 3
    assert context.config.local_fusion == "additive_mlp"
    with pytest.raises(ValueError, match="additive_mlp"):
        MLPContextStateTranslator(SPEC, SPEC, config=CONFIG)


def test_legacy_context_payload_dispatch_preserves_weights_outputs_and_metadata(tmp_path, make_state):
    # Before the split, context artifacts were made by this generic wrapper.
    config = SpatialTransformerConfig.from_dict({**CONFIG.to_dict(), "local_fusion": "additive_mlp"})
    old = TransformerStateTranslator(SPEC, SPEC, config=config, preset="mlp_context")
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.uniform_(-0.2, 0.2)
    payload = old.to_payload()
    artifact = tmp_path / "legacy_context.pt"
    torch.save({"mlp_context": payload}, artifact)
    restored = load_learned_translator(artifact, "mlp_context")
    assert type(restored) is MLPContextStateTranslator
    assert type(TransformerStateTranslator.from_payload(payload)) is MLPContextStateTranslator
    assert restored.to_payload().keys() == payload.keys()
    assert restored.to_payload()["translator"] == payload["translator"]
    assert restored.state_dict().keys() == old.state_dict().keys()
    for key, tensor in old.state_dict().items():
        assert torch.equal(restored.state_dict()[key], tensor)

    source = make_state(
        torch.randn(1, 1, 2, 4, 8, 8).bfloat16(), torch.randn(1, 1, 2, 6),
        validity=torch.tensor([[[True, False]]]),
    )
    for method in ("translate_tensors", "translate_handoff_tensors"):
        inputs = (source.spatial_memory, source.object_pointer, source.validity)
        expected = getattr(old, method)(*inputs)
        actual = getattr(restored, method)(*inputs)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert old.translate(source).metadata == restored.translate(source).metadata


@pytest.mark.parametrize("module", [
    "models.residual_mlp", "models.transformer", "models.mlp_context", "models.registry",
])
def test_model_modules_import_without_legacy_modules_loaded_first(module):
    # A fresh interpreter catches cycles hidden by pytest's import order.
    subprocess.run(
        [sys.executable, "-c", f"import vos_memory_inspector.{module}"],
        check=True, capture_output=True, text=True,
    )
