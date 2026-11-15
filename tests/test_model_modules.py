"""Model separation must keep legacy imports, artifacts and entrypoints usable."""

import pickle
import subprocess
import sys

import pytest

import vos_memory_inspector as package
from vos_memory_inspector.models import (
    ResidualMLPStateTranslator,
    TransformerStateTranslator,
    TranslatorConfig,
    build_translator,
)
from vos_memory_inspector.state_schema import StateSpec


SPEC = StateSpec(4, 8, 8, 6)
CONFIG = TranslatorConfig(num_heads=2, sr_ratio=2)


def test_old_import_and_pickle_paths_resolve_to_model_classes():
    from vos_memory_inspector.translators import ResidualMLPStateTranslator as old_mlp
    from vos_memory_inspector.transformer_translator import TransformerStateTranslator as old_transformer

    assert old_mlp is package.ResidualMLPStateTranslator is ResidualMLPStateTranslator
    assert old_transformer is package.TransformerStateTranslator is TransformerStateTranslator
    for module, name, expected in (
        ("translators", "ResidualMLPStateTranslator", ResidualMLPStateTranslator),
        ("transformer_translator", "TransformerStateTranslator", TransformerStateTranslator),
    ):
        # Protocol 0 GLOBAL references used in older whole-model pickles.
        reference = f"cvos_memory_inspector.{module}\n{name}\n.".encode()
        assert pickle.loads(reference) is expected


def test_compatibility_module_exports_the_new_names_only():
    import vos_memory_inspector.transformer_translator as compat

    for name in ("StitchedStateTranslator", "TranslatorConfig", "SAM21_MEMORY_SPEC",
                 "OUTPUT_DTYPES", "PAYLOAD_SCHEMA", "build_translator", "PRESETS"):
        assert hasattr(compat, name), name
    for name in ("SpatialTransformerConfig", "SpatialMemoryTranslator", "ResidualPointerTranslator",
                 "sincos_2d_embedding", "describe_config"):
        assert not hasattr(compat, name), name


def test_factory_uses_separate_model_modules():
    mlp = build_translator("residual_mlp", SPEC, SPEC)
    transformer = build_translator("transformer", SPEC, SPEC, config=CONFIG)
    assert type(mlp) is ResidualMLPStateTranslator
    assert type(transformer) is TransformerStateTranslator
    assert len({type(model).__module__ for model in (mlp, transformer)}) == 2


@pytest.mark.parametrize("module", [
    "models.residual_mlp", "models.transformer", "models.registry",
])
def test_model_modules_import_without_legacy_modules_loaded_first(module):
    # A fresh interpreter catches cycles hidden by pytest's import order.
    subprocess.run(
        [sys.executable, "-c", f"import vos_memory_inspector.{module}"],
        check=True, capture_output=True, text=True,
    )
