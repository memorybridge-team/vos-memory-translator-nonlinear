"""Model-specific modules with lazy exports for callers and tools."""

from importlib import import_module
from typing import Any

__all__ = [
    "ResidualMLPStateTranslator", "TransformerStateTranslator",
    "MLPContextStateTranslator", "SpatialTransformerConfig", "build_translator",
]

_EXPORTS = {
    "ResidualMLPStateTranslator": (".residual_mlp", "ResidualMLPStateTranslator"),
    "TransformerStateTranslator": (".transformer", "TransformerStateTranslator"),
    "MLPContextStateTranslator": (".mlp_context", "MLPContextStateTranslator"),
    "SpatialTransformerConfig": ("._spatial", "SpatialTransformerConfig"),
    "build_translator": (".registry", "build_translator"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
