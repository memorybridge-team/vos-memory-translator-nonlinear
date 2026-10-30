"""Independent local residual MLP plus gated spatial Transformer context.

The spatial backbone and pointer head are shared with the Transformer model.
Only the context correction is gated: M_hat = M + LocalMLP(M) + alpha * Context(M).
Payload keys remain compatible with artifacts saved before the module split.
"""

from __future__ import annotations

from ..state_schema import StateSpec
from ._spatial import SpatialTransformerConfig
from .transformer import TransformerStateTranslator


class MLPContextStateTranslator(TransformerStateTranslator):
    """State translator with an independent, ungated local MLP correction."""

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        config: SpatialTransformerConfig | None = None,
        preset: str | None = "mlp_context",
    ):
        if config is None:
            config = SpatialTransformerConfig(local_fusion="additive_mlp")
        if config.local_fusion != "additive_mlp":
            raise ValueError("mlp_context requires local_fusion='additive_mlp'")
        super().__init__(source_spec, target_spec, config=config, preset=preset)
