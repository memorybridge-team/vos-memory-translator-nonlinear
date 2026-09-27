"""SAM 2.1 Small to Base+ memory handoff.

Heavy torch-backed symbols are loaded lazily so commands can run before the
GPU research environment is installed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CanonicalState",
    "ComponentAblationTranslator",
    "DirectCopyTranslator",
    "LightweightSpatialMemoryTranslator",
    "LinearStateTranslator",
    "MomentMatchedCopyTranslator",
    "PRESETS",
    "ResidualMLPStateTranslator",
    "ResidualPointerTranslator",
    "RidgeStateTranslator",
    "SpatialTransformerConfig",
    "StateSpec",
    "SUPPORTED_SAM2_COMMIT",
    "TransformerStateTranslator",
    "benchmark_translator",
    "build_translator",
]


_EXPORTS = {
    "CanonicalState": (".state_schema", "CanonicalState"),
    "StateSpec": (".state_schema", "StateSpec"),
    "ComponentAblationTranslator": (".translators", "ComponentAblationTranslator"),
    "DirectCopyTranslator": (".translators", "DirectCopyTranslator"),
    "LinearStateTranslator": (".translators", "LinearStateTranslator"),
    "MomentMatchedCopyTranslator": (".translators", "MomentMatchedCopyTranslator"),
    "ResidualMLPStateTranslator": (".translators", "ResidualMLPStateTranslator"),
    "RidgeStateTranslator": (".translators", "RidgeStateTranslator"),
    "LightweightSpatialMemoryTranslator": (
        ".transformer_translator",
        "LightweightSpatialMemoryTranslator",
    ),
    "PRESETS": (".transformer_translator", "PRESETS"),
    "ResidualPointerTranslator": (".transformer_translator", "ResidualPointerTranslator"),
    "SpatialTransformerConfig": (".transformer_translator", "SpatialTransformerConfig"),
    "TransformerStateTranslator": (".transformer_translator", "TransformerStateTranslator"),
    "build_translator": (".transformer_translator", "build_translator"),
    "benchmark_translator": (".translator_benchmark", "benchmark_translator"),
    "SUPPORTED_SAM2_COMMIT": (".upstream", "SUPPORTED_SAM2_COMMIT"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
