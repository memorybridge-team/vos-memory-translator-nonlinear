"""Spatial Transformer state translator and compatible payload loading."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch

from ..state_schema import CanonicalState, StateSpec
from ._base import _LearnedStateTranslator
from .affine import initialize_skips
from ._spatial import (
    PAYLOAD_SCHEMA, ResidualPointerTranslator, SpatialMemoryTranslator,
    SpatialTransformerConfig, describe_config,
)


class TransformerStateTranslator(_LearnedStateTranslator):
    """CanonicalState translator: spatial-context memory + residual pointer MLP.

    ``translate`` gathers valid records ``[B,O,K,...] -> [N,...]``, translates
    each frame independently, casts to the source dtype (bf16 spatial, fp32
    pointer by default) and scatters back. Padding records keep the source.
    ``translate_tensors`` is the gradient-preserving fp32 training API.
    """

    name = "spatial_context_transformer"
    grid_adapter = "none"
    # The context branch reads the whole frame, so single positions cannot be sampled.
    supports_position_sampling = False
    _spatial_module_prefix = "spatial"
    _pointer_module_prefix = "pointer"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        config: SpatialTransformerConfig | None = None,
        preset: str | None = None,
    ):
        config = SpatialTransformerConfig() if config is None else config
        super().__init__(source_spec, target_spec, output_dtype=config.output_dtype)
        expected = StateSpec(
            config.channels, config.height, config.width, config.pointer_dim
        )
        for role, spec in (("source", source_spec), ("target", target_spec)):
            actual = (spec.feature_channels, spec.height, spec.width, spec.pointer_dim)
            wanted = (
                expected.feature_channels,
                expected.height,
                expected.width,
                expected.pointer_dim,
            )
            if actual != wanted:
                raise ValueError(
                    f"{role} spec [C,H,W,D]={actual} does not match the translator "
                    f"config {wanted}"
                )
        self.skip_mode = config.skip_mode
        self.ridge_initialization = None
        self.config = config
        self.preset = preset
        self.spatial = SpatialMemoryTranslator(config)
        self.pointer = ResidualPointerTranslator(config.pointer_dim, config.pointer_hidden_dim, config.skip_mode)

    def initialize_from_ridge(self, ridge):
        if self.skip_mode != "affine":
            raise ValueError("ridge initialization requires affine skip")
        initialize_skips(self, self.spatial.linear_skip, self.pointer.linear_skip, ridge)

    def _translate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.spatial(frames)  # [N, C, H, W]

    def _translate_pointers(self, pointers: torch.Tensor) -> torch.Tensor:
        return self.pointer(pointers)  # [N, D]

    def macs_per_frame(self) -> int:
        return self.spatial.macs_per_frame()

    def pointer_macs_per_record(self) -> int:
        return self.pointer.macs_per_record()

    @property
    def description(self) -> str:
        return describe_config(self.config)

    def architecture_summary(self) -> dict[str, Any]:
        """JSON-safe architecture, parameter and analytic-cost summary."""

        config = self.config
        attention = None
        if config.spatial_context and config.use_attention:
            attention = {
                "scope": "per_frame",
                "matrix": [config.num_tokens, config.num_tokens],
                "heads": config.num_heads,
                "cross_frame": False,
            }
        return {
            "translator": self.name,
            "preset": self.preset,
            "description": self.description,
            "config": config.to_dict(),
            "token_grid": list(config.grid_size) if config.spatial_context else None,
            "tokens_per_frame": config.num_tokens if config.spatial_context else 0,
            "attention": attention,
            "parameters": self.parameter_breakdown(),
            "spatial_parameters_by_module": self.spatial.parameter_breakdown(),
            "macs_per_frame": self.macs_per_frame(),
            "macs_by_module": self.spatial.macs_breakdown(),
            "context_branch_macs": self.spatial.context_branch_macs(),
            "per_pixel_macs": self.spatial.per_pixel_macs(),
            "pointer_macs_per_record": self.pointer_macs_per_record(),
            "identity_at_init": config.skip_mode == "identity" and config.use_residual and config.residual_scale_init == 0.0,
            "residual_gate_scope": "context_only" if config.local_fusion == "additive_mlp" else "combined_delta",
        }

    def _translation_metadata(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "architecture": self.description,
            "attention_scope": (
                "per_frame_tokens"
                if self.config.spatial_context and self.config.use_attention
                else "none"
            ),
            "tokens_per_frame": self.config.num_tokens if self.config.spatial_context else 0,
            "target_positional_encoding": "regenerated_by_target",
        }

    def to_payload(self) -> dict[str, Any]:
        """Plain dict with CPU tensors; loadable with ``torch.load(weights_only=True)``."""

        config_values = self.config.to_dict()
        if self.skip_mode == "identity":
            config_values.pop("skip_mode")
        elif not self.ridge_initialization:
            raise ValueError("affine model has not been initialized from ridge")
        return {
            **({"ridge_initialization": self.ridge_initialization} if self.skip_mode == "affine" else {}),
            "schema_version": "cmmt.spatial_context_transformer_translator.v2" if self.skip_mode == "affine" else PAYLOAD_SCHEMA,
            "translator": self.name,
            "preset": self.preset,
            "config": config_values,
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TransformerStateTranslator":
        schema = payload.get("schema_version")
        if schema not in (PAYLOAD_SCHEMA, "cmmt.spatial_context_transformer_translator.v2"):
            raise ValueError("unsupported spatial-context transformer payload")
        config_values = payload.get("config")
        if not isinstance(config_values, Mapping):
            raise TypeError("payload config must be a mapping")
        missing = sorted({item.name for item in fields(SpatialTransformerConfig) if item.name != "skip_mode"} - set(config_values))
        if missing:
            raise ValueError(f"payload config is missing fields: {missing}")
        if schema != PAYLOAD_SCHEMA and (config_values.get("skip_mode") != "affine" or not payload.get("ridge_initialization")):
            raise ValueError("affine payload initialization is missing")
        if schema == PAYLOAD_SCHEMA and config_values.get("skip_mode", "identity") != "identity":
            raise ValueError("legacy payload cannot declare affine skip")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("payload state_dict must be a mapping")
        preset = payload.get("preset")
        model_type = cls
        if cls is TransformerStateTranslator and config_values.get("local_fusion") == "additive_mlp":
            # Older artifacts used the same wrapper for both architectures.
            from .mlp_context import MLPContextStateTranslator

            model_type = MLPContextStateTranslator
        translator = model_type(
            StateSpec(**dict(payload["source_spec"])),
            StateSpec(**dict(payload["target_spec"])),
            config=SpatialTransformerConfig.from_dict(config_values),
            preset=None if preset is None else str(preset),
        )
        translator.ridge_initialization = payload.get("ridge_initialization")
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator
