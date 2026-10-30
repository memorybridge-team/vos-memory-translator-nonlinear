"""Load a named learned model saved by the paired-state experiment."""

from pathlib import Path
from typing import Any

import torch

from .models.residual_mlp import ResidualMLPStateTranslator


TRANSFORMER_TRANSLATORS = ("transformer", "mlp_context")
LEARNED_ARTIFACT_TRANSLATORS = ("residual_mlp", *TRANSFORMER_TRANSLATORS)


def load_learned_translator(path: Path, name: str) -> Any:
    if name not in LEARNED_ARTIFACT_TRANSLATORS:
        raise ValueError(f"unsupported learned translator: {name}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    payload = artifact.get(name) if isinstance(artifact, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"translator artifact does not contain a {name} payload")
    if name == "residual_mlp":
        return ResidualMLPStateTranslator.from_payload(payload).eval()

    from .models.transformer import TransformerStateTranslator

    translator = TransformerStateTranslator.from_payload(payload)
    expected_fusion = "additive_mlp" if name == "mlp_context" else "linear"
    if translator.config.local_fusion != expected_fusion:
        raise ValueError(f"{name} payload has incompatible local_fusion")
    return translator
