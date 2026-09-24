"""SAM 2.1 Small to Base+ memory handoff.

Heavy torch-backed symbols are loaded lazily so commands can run before the
GPU research environment is installed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CanonicalState",
    "DirectCopyTranslator",
    "LinearStateTranslator",
    "ResidualMLPStateTranslator",
    "RidgeStateTranslator",
    "StateSpec",
    "SUPPORTED_SAM2_COMMIT",
]


_EXPORTS = {
    "CanonicalState": (".state_schema", "CanonicalState"),
    "StateSpec": (".state_schema", "StateSpec"),
    "DirectCopyTranslator": (".translators", "DirectCopyTranslator"),
    "LinearStateTranslator": (".translators", "LinearStateTranslator"),
    "ResidualMLPStateTranslator": (".translators", "ResidualMLPStateTranslator"),
    "RidgeStateTranslator": (".translators", "RidgeStateTranslator"),
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
