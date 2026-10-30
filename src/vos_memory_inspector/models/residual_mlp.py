"""Position-wise residual MLP for spatial memory and object pointers."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..state_schema import StateSpec
from ._base import _LearnedStateTranslator
from .affine import make_skip, initialize_skips


class ResidualMLPStateTranslator(_LearnedStateTranslator):
    """MLP-only translator with independent, position-shared residual heads.

    Defaults: spatial 64->128->64 and pointer 256->512->256 for SAM 2.1.
    Both heads use GELU and zero-initialized output layers. Equal-shape inputs
    start at identity; no attention, convolution or cross-record mixing occurs.
    Explicit ``pointer_hidden_dim=None`` retains the legacy shared hidden width.
    """

    name = "residual_mlp"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        hidden_dim: int = 128,
        pointer_hidden_dim: int | None = 512,
        output_dtype: str = "source",
        skip_mode: str = "identity",
    ):
        super().__init__(source_spec, target_spec, output_dtype=output_dtype)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        pointer_hidden_dim = hidden_dim if pointer_hidden_dim is None else pointer_hidden_dim
        if pointer_hidden_dim < 1:
            raise ValueError("pointer_hidden_dim must be positive")
        if skip_mode not in ("identity", "affine"):
            raise ValueError("unknown skip_mode")
        self.skip_mode = skip_mode
        self.ridge_initialization = None
        self.hidden_dim = hidden_dim
        self.pointer_hidden_dim = pointer_hidden_dim
        self.feature = nn.Sequential(
            nn.Linear(source_spec.feature_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_spec.feature_channels),
        )
        self.pointer = nn.Sequential(
            nn.Linear(source_spec.pointer_dim, pointer_hidden_dim),
            nn.GELU(),
            nn.Linear(pointer_hidden_dim, target_spec.pointer_dim),
        )
        # 보정값만 0에서 시작하고 첫 층은 일반 초기화를 유지한다.
        for head in (self.feature, self.pointer):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.feature_residual = (
            source_spec.feature_channels == target_spec.feature_channels
            and source_spec.height == target_spec.height
            and source_spec.width == target_spec.width
        )
        self.pointer_residual = source_spec.pointer_dim == target_spec.pointer_dim
        self.feature_skip = make_skip(source_spec.feature_channels, target_spec.feature_channels) if skip_mode == "affine" else None
        self.pointer_skip = make_skip(source_spec.pointer_dim, target_spec.pointer_dim) if skip_mode == "affine" else None

    def initialize_from_ridge(self, ridge):
        if self.skip_mode != "affine":
            raise ValueError("ridge initialization requires affine skip")
        initialize_skips(self, self.feature_skip, self.pointer_skip, ridge)

    def parameter_breakdown(self):
        result = super().parameter_breakdown()
        if self.skip_mode == "affine":
            result["spatial"] += sum(p.numel() for p in self.feature_skip.parameters())
            result["pointer"] += sum(p.numel() for p in self.pointer_skip.parameters())
        return result


    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.feature[0].weight.device,
            dtype=self.feature[0].weight.dtype,
        )
        output = self.feature(calibrated)
        if self.feature_skip is not None:
            return output + self.feature_skip(calibrated)
        return output + calibrated if self.feature_residual else output

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.pointer[0].weight.device,
            dtype=self.pointer[0].weight.dtype,
        )
        output = self.pointer(calibrated)
        if self.pointer_skip is not None:
            return output + self.pointer_skip(calibrated)
        return output + calibrated if self.pointer_residual else output

    def macs_per_frame(self) -> int:
        positions = self.target_spec.height * self.target_spec.width
        return positions * (
            self.source_spec.feature_channels * self.hidden_dim
            + self.hidden_dim * self.target_spec.feature_channels
            + (self.source_spec.feature_channels * self.target_spec.feature_channels if self.skip_mode == "affine" else 0)
        )

    def pointer_macs_per_record(self) -> int:
        return (
            self.source_spec.pointer_dim * self.pointer_hidden_dim
            + self.pointer_hidden_dim * self.target_spec.pointer_dim
            + (self.source_spec.pointer_dim * self.target_spec.pointer_dim if self.skip_mode == "affine" else 0)
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize architecture metadata with CPU weights for safe reuse."""

        if self.skip_mode == "affine" and not self.ridge_initialization:
            raise ValueError("affine model has not been initialized from ridge")
        return {
            **({"skip_mode": self.skip_mode, "ridge_initialization": self.ridge_initialization} if self.skip_mode == "affine" else {}),
            "schema_version": "cmmt.residual_mlp_translator.v3" if self.skip_mode == "affine" else "cmmt.residual_mlp_translator.v2",
            "translator": self.name,
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "hidden_dim": self.hidden_dim,
            "pointer_hidden_dim": self.pointer_hidden_dim,
            "output_dtype": self.output_dtype,
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualMLPStateTranslator":
        schema = payload.get("schema_version")
        if schema not in ("cmmt.residual_mlp_translator.v2", "cmmt.residual_mlp_translator.v3"):
            raise ValueError("unsupported residual MLP translator payload")
        if schema.endswith("v3") and (payload.get("skip_mode") != "affine" or not payload.get("ridge_initialization")):
            raise ValueError("affine payload initialization is missing")
        if schema.endswith("v2") and payload.get("skip_mode", "identity") != "identity":
            raise ValueError("legacy payload cannot declare affine skip")
        source_spec = StateSpec(**dict(payload["source_spec"]))
        target_spec = StateSpec(**dict(payload["target_spec"]))
        hidden_dim = int(payload["hidden_dim"])
        # Payloads written before pointer_hidden_dim existed used hidden_dim
        # for both heads and always produced float32.
        translator = cls(
            source_spec,
            target_spec,
            hidden_dim=hidden_dim,
            pointer_hidden_dim=int(payload.get("pointer_hidden_dim", hidden_dim)),
            output_dtype=str(payload.get("output_dtype", "float32")),
            skip_mode="affine" if schema.endswith("v3") else "identity",
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("residual MLP payload state_dict must be a mapping")
        translator.ridge_initialization = payload.get("ridge_initialization")
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator
