"""Component-wise CMMT baselines for SAM 2-style canonical state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .state_schema import CanonicalState, StateSpec, validate_paired_state_contract


_OUTPUT_DTYPE_POLICIES = ("source", "float32")


def _resample_spatial(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim != 6:
        raise ValueError("spatial input must be [B,O,K,C,H,W]")
    if tensor.shape[-2:] == (height, width):
        return tensor
    batch, objects, records, channels, old_h, old_w = tensor.shape
    flat = tensor.reshape(batch * objects * records, channels, old_h, old_w)
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(batch, objects, records, channels, height, width)


def _resample_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear grid adapter for a frame batch ``[N,C,H,W]``; no-op on equal grids."""

    if frames.shape[-2:] == (height, width):
        return frames
    return F.interpolate(
        frames, size=(height, width), mode="bilinear", align_corners=False
    )


def _valid_record_index(validity: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Row-major flat indices of valid ``[B,O,K]`` records (int64 on ``device``).

    ``nonzero`` synchronizes with the device. Latency-critical callers can
    compute this index once and call ``_translate_indexed`` directly.
    """

    return validity.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(device)


def _gather_records(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``[B,O,K,*F] -> [N,*F]`` for the flat record ``index``."""

    return tensor.reshape(-1, *tensor.shape[3:]).index_select(0, index)


def _scatter_records(
    values: torch.Tensor,
    index: torch.Tensor,
    record_shape: tuple[int, ...],
    fill: torch.Tensor | None,
) -> torch.Tensor:
    """Write ``[N,*F]`` rows into a new ``[B,O,K,*F]`` tensor.

    Slots outside ``index`` come from ``fill`` (``None`` means zeros). The copy is
    out of place, so ``fill`` is never modified and gradients reach ``values``.
    """

    feature_shape = tuple(values.shape[1:])
    if fill is None:
        base = values.new_zeros((math.prod(record_shape), *feature_shape))
    else:
        base = fill.reshape(-1, *feature_shape).to(
            device=values.device, dtype=values.dtype
        )
    return base.index_copy(0, index, values).reshape(*record_shape, *feature_shape)


def _padding_only(
    fill: torch.Tensor | None,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Output for a state without valid records: a source copy or zeros."""

    if fill is None:
        return torch.zeros(shape, device=device, dtype=dtype)
    return fill.to(device=device, dtype=dtype, copy=True)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _pad_or_truncate(tensor: torch.Tensor, output_dim: int, axis: int) -> torch.Tensor:
    axis = axis if axis >= 0 else tensor.ndim + axis
    current = tensor.shape[axis]
    if current == output_dim:
        return tensor.clone()
    if current > output_dim:
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(0, output_dim)
        return tensor[tuple(slices)].clone()
    shape = list(tensor.shape)
    shape[axis] = output_dim - current
    padding = tensor.new_zeros(shape)
    return torch.cat((tensor, padding), dim=axis)


def _target_positional(spec: StateSpec) -> dict[str, str]:
    return {"policy": spec.positional_policy}


def _pair_guard(source: CanonicalState, target: CanonicalState) -> torch.Tensor:
    return validate_paired_state_contract(source, target)


class DirectCopyTranslator:
    """No-learning baseline with explicit grid resize and zero-pad/truncate rules."""

    name = "direct"

    def __init__(self, target_spec: StateSpec):
        self.target_spec = target_spec

    def translate_handoff_tensors(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy every record (padding included), adapting grid and widths only."""

        del validity  # Direct Copy treats valid and padding records alike.
        spatial = _resample_spatial(
            spatial, self.target_spec.height, self.target_spec.width
        )
        return (
            _pad_or_truncate(spatial, self.target_spec.feature_channels, axis=3),
            _pad_or_truncate(pointer, self.target_spec.pointer_dim, axis=-1),
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        spatial, pointer = self.translate_handoff_tensors(
            source.spatial_memory, source.object_pointer
        )
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "channel_adapter": "truncate_or_zero_pad",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return 0


@dataclass(frozen=True)
class AffineMap:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=tensor.device, dtype=tensor.dtype)
        bias = self.bias.to(device=tensor.device, dtype=tensor.dtype)
        return tensor @ weight + bias

    @property
    def parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()


def fit_affine_closed_form(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_lambda: float,
    fit_dtype: torch.dtype = torch.float64,
) -> AffineMap:
    """Fit centered affine OLS (lambda=0) or ridge in closed form."""

    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("affine calibration expects X=[N,Din], Y=[N,Dout]")
    if inputs.shape[0] < 1:
        raise ValueError("affine calibration has no observations")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")
    x = inputs.detach().to(device="cpu", dtype=fit_dtype)
    y = targets.detach().to(device="cpu", dtype=fit_dtype)
    mean_x = x.mean(dim=0, keepdim=True)
    mean_y = y.mean(dim=0, keepdim=True)
    xc = x - mean_x
    yc = y - mean_y
    if ridge_lambda == 0:
        weight = torch.linalg.lstsq(xc, yc).solution
    else:
        gram = xc.T @ xc
        identity = torch.eye(gram.shape[0], dtype=gram.dtype)
        weight = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, xc.T @ yc
        )
    bias = (mean_y - mean_x @ weight).squeeze(0)
    output_dtype = inputs.dtype if inputs.is_floating_point() else torch.float32
    return AffineMap(weight.to(output_dtype), bias.to(output_dtype))


class RidgeStateTranslator:
    name = "ridge"

    def __init__(
        self,
        target_spec: StateSpec,
        feature_map: AffineMap,
        pointer_map: AffineMap,
        ridge_lambda: float,
    ):
        self.target_spec = target_spec
        self.feature_map = feature_map
        self.pointer_map = pointer_map
        self.ridge_lambda = ridge_lambda

    @classmethod
    def fit(
        cls,
        pairs: Iterable[tuple[CanonicalState, CanonicalState]],
        *,
        ridge_lambda: float = 0.01,
    ) -> "RidgeStateTranslator":
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("at least one paired state is required")
        target_spec = pair_list[0][1].spec
        feature_x: list[torch.Tensor] = []
        feature_y: list[torch.Tensor] = []
        pointer_x: list[torch.Tensor] = []
        pointer_y: list[torch.Tensor] = []
        for source, target in pair_list:
            if target.spec != target_spec:
                raise ValueError("all target states must share one runtime StateSpec")
            valid = _pair_guard(source, target)
            src_spatial = _resample_spatial(
                source.spatial_memory, target_spec.height, target_spec.width
            )
            sx = src_spatial.permute(0, 1, 2, 4, 5, 3)[valid]
            sy = target.spatial_memory.permute(0, 1, 2, 4, 5, 3)[
                valid.to(target.validity.device)
            ]
            feature_x.append(sx.reshape(-1, sx.shape[-1]))
            feature_y.append(sy.reshape(-1, sy.shape[-1]))
            pointer_x.append(source.object_pointer[valid])
            pointer_y.append(target.object_pointer[valid.to(target.validity.device)])
        return cls(
            target_spec=target_spec,
            feature_map=fit_affine_closed_form(
                torch.cat(feature_x), torch.cat(feature_y), ridge_lambda=ridge_lambda
            ),
            pointer_map=fit_affine_closed_form(
                torch.cat(pointer_x), torch.cat(pointer_y), ridge_lambda=ridge_lambda
            ),
            ridge_lambda=ridge_lambda,
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        spatial = _resample_spatial(
            source.spatial_memory, self.target_spec.height, self.target_spec.width
        )
        spatial = self.feature_map(spatial.movedim(3, -1)).movedim(-1, 3)
        pointer = self.pointer_map(source.object_pointer)
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "ridge_lambda": self.ridge_lambda,
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.feature_map.parameter_count + self.pointer_map.parameter_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_spec": self.target_spec.to_dict(),
            "ridge_lambda": self.ridge_lambda,
            "feature_weight": self.feature_map.weight.detach().cpu(),
            "feature_bias": self.feature_map.bias.detach().cpu(),
            "pointer_weight": self.pointer_map.weight.detach().cpu(),
            "pointer_bias": self.pointer_map.bias.detach().cpu(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RidgeStateTranslator":
        required = {
            "target_spec",
            "ridge_lambda",
            "feature_weight",
            "feature_bias",
            "pointer_weight",
            "pointer_bias",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Ridge payload is missing fields: {missing}")
        spec_value = payload["target_spec"]
        if not isinstance(spec_value, Mapping):
            raise TypeError("Ridge target_spec must be a mapping")
        target_spec = StateSpec(**dict(spec_value))
        tensor_fields = {
            name: payload[name]
            for name in required
            if name.endswith("weight") or name.endswith("bias")
        }
        invalid = [name for name, value in tensor_fields.items() if not isinstance(value, torch.Tensor)]
        if invalid:
            raise TypeError(f"Ridge payload fields must be tensors: {invalid}")
        return cls(
            target_spec=target_spec,
            feature_map=AffineMap(payload["feature_weight"], payload["feature_bias"]),
            pointer_map=AffineMap(payload["pointer_weight"], payload["pointer_bias"]),
            ridge_lambda=float(payload["ridge_lambda"]),
        )


class RidgeDirectPresenceTranslator:
    """Use Ridge for memory/pointer and preserve the source presence logit."""

    name = "ridge_spatial_pointer_direct_presence"

    def __init__(self, ridge: RidgeStateTranslator):
        self.ridge = ridge
        self.target_spec = ridge.target_spec

    def translate(self, source: CanonicalState) -> CanonicalState:
        ridge_state = self.ridge.translate(source)
        return ridge_state.with_continuous(
            spatial_memory=ridge_state.spatial_memory,
            object_pointer=ridge_state.object_pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "ridge_lambda": self.ridge.ridge_lambda,
                "spatial_memory": "ridge",
                "object_pointer": "ridge",
                "presence_logits": "direct",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.ridge.feature_map.parameter_count + self.ridge.pointer_map.parameter_count


class _LearnedStateTranslator(nn.Module):
    """Shared handoff path for learned translators.

    Subclasses map a frame batch ``[N,C,H,W]`` (``_translate_frames``) and a
    pointer batch ``[N,D]`` (``_translate_pointers``), where ``N`` counts only
    valid records. The same map is applied to every frame and nothing mixes
    the ``N`` axis. Padding never enters a forward pass. Padding slots copy the
    source when source and target shapes match; otherwise they are zero.
    """

    name = "learned"
    preset: str | None = None
    grid_adapter = "bilinear"
    # The legacy sampled trainer draws single spatial positions; that is only
    # meaningful for translators whose spatial head is position-wise.
    supports_position_sampling = True
    _spatial_module_prefix = "feature"
    _pointer_module_prefix = "pointer"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        output_dtype: str = "float32",
    ):
        super().__init__()
        if output_dtype not in _OUTPUT_DTYPE_POLICIES:
            raise ValueError(
                f"output_dtype must be one of {_OUTPUT_DTYPE_POLICIES}, got {output_dtype!r}"
            )
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.output_dtype = output_dtype

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _translate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``[N,Cs,Hs,Ws] -> [N,Ct,Ht,Wt]``; the default applies a position-wise head."""

        frames = _resample_frames(
            frames, self.target_spec.height, self.target_spec.width
        )
        return self._feature_head(frames.movedim(1, -1)).movedim(-1, 1)

    def _translate_pointers(self, pointers: torch.Tensor) -> torch.Tensor:
        """``[N,Ds] -> [N,Dt]``."""

        return self._pointer_head(pointers)

    def _compute_reference(self) -> torch.Tensor:
        for parameter in self.parameters():
            return parameter
        raise RuntimeError(f"{type(self).__name__} has no parameters")

    def _spatial_padding_copies_source(self) -> bool:
        source, target = self.source_spec, self.target_spec
        return (source.feature_channels, source.height, source.width) == (
            target.feature_channels,
            target.height,
            target.width,
        )

    def _pointer_padding_copies_source(self) -> bool:
        return self.source_spec.pointer_dim == self.target_spec.pointer_dim

    def invalid_record_policy(self) -> dict[str, str]:
        return {
            "translated": "valid_records_only",
            "spatial_memory": (
                "source_copy" if self._spatial_padding_copies_source() else "zero_fill"
            ),
            "object_pointer": (
                "source_copy" if self._pointer_padding_copies_source() else "zero_fill"
            ),
        }

    def output_dtypes(
        self, spatial_dtype: torch.dtype, pointer_dtype: torch.dtype
    ) -> tuple[torch.dtype, torch.dtype]:
        """Handoff output dtypes for the given source dtypes."""

        if self.output_dtype == "source":
            return spatial_dtype, pointer_dtype
        return torch.float32, torch.float32

    def _check_tensors(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        validity: torch.Tensor | None,
    ) -> torch.Tensor:
        if spatial.ndim != 6:
            raise ValueError(
                f"spatial must be [B,O,K,C,H,W], got {tuple(spatial.shape)}"
            )
        if pointer.ndim != 4 or pointer.shape[:3] != spatial.shape[:3]:
            raise ValueError(
                "pointer must be [B,O,K,D] with the spatial [B,O,K]; got "
                f"{tuple(pointer.shape)} for spatial {tuple(spatial.shape)}"
            )
        expected = (
            self.source_spec.feature_channels,
            self.source_spec.height,
            self.source_spec.width,
        )
        if tuple(spatial.shape[3:]) != expected:
            raise ValueError(
                f"spatial [C,H,W]={tuple(spatial.shape[3:])} does not match "
                f"source spec {expected}"
            )
        if pointer.shape[-1] != self.source_spec.pointer_dim:
            raise ValueError(
                f"pointer dim {pointer.shape[-1]} does not match source spec "
                f"{self.source_spec.pointer_dim}"
            )
        if validity is None:
            return torch.ones(spatial.shape[:3], dtype=torch.bool, device=spatial.device)
        if validity.dtype != torch.bool or validity.shape != spatial.shape[:3]:
            raise ValueError(
                "validity must be a bool [B,O,K] tensor matching spatial; got "
                f"{validity.dtype} {tuple(validity.shape)}"
            )
        return validity

    def _translate_indexed(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        index: torch.Tensor,
        *,
        spatial_dtype: torch.dtype,
        pointer_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """valid gather -> compute dtype -> translator -> output cast -> scatter.

        ``index`` holds flat ``[B*O*K]`` record positions. This path does not
        synchronize with the device, so it can be captured in a CUDA graph.
        """

        reference = self._compute_reference()
        device, compute_dtype = reference.device, reference.dtype
        spatial = spatial.to(device)
        pointer = pointer.to(device)
        index = index.to(device)
        record_shape = tuple(spatial.shape[:3])
        target = self.target_spec
        spatial_fill = spatial if self._spatial_padding_copies_source() else None
        pointer_fill = pointer if self._pointer_padding_copies_source() else None
        if index.numel() == 0:
            # No valid record: never call the translator.
            return (
                _padding_only(
                    spatial_fill,
                    (*record_shape, target.feature_channels, target.height, target.width),
                    device=device,
                    dtype=spatial_dtype,
                ),
                _padding_only(
                    pointer_fill,
                    (*record_shape, target.pointer_dim),
                    device=device,
                    dtype=pointer_dtype,
                ),
            )
        frames = _gather_records(spatial, index).to(compute_dtype)  # [N,C,H,W]
        pointers = _gather_records(pointer, index).to(compute_dtype)  # [N,D]
        translated_frames = self._translate_frames(frames).to(spatial_dtype)
        translated_pointers = self._translate_pointers(pointers).to(pointer_dtype)
        return (
            _scatter_records(translated_frames, index, record_shape, spatial_fill),
            _scatter_records(translated_pointers, index, record_shape, pointer_fill),
        )

    def translate_tensors(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training API: ``[B,O,K,C,H,W]``, ``[B,O,K,D]`` -> compute-dtype tensors.

        Gradients are kept. Outputs use the parameter dtype (fp32 by default)
        and the parameter device. ``validity=None`` treats every record as valid.
        """

        validity = self._check_tensors(spatial, pointer, validity)
        reference = self._compute_reference()
        index = _valid_record_index(validity, reference.device)
        return self._translate_indexed(
            spatial,
            pointer,
            index,
            spatial_dtype=reference.dtype,
            pointer_dtype=reference.dtype,
        )

    def translate_handoff_tensors(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handoff API: as ``translate_tensors``, then cast by ``output_dtype``.

        The cast happens before the scatter, so padding copies keep the source
        bits when ``output_dtype="source"``.
        """

        validity = self._check_tensors(spatial, pointer, validity)
        reference = self._compute_reference()
        index = _valid_record_index(validity, reference.device)
        spatial_dtype, pointer_dtype = self.output_dtypes(spatial.dtype, pointer.dtype)
        return self._translate_indexed(
            spatial,
            pointer,
            index,
            spatial_dtype=spatial_dtype,
            pointer_dtype=pointer_dtype,
        )

    def _translation_metadata(self) -> dict[str, Any]:
        return {}

    def forward(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        if source.spec != self.source_spec:
            raise ValueError(
                f"source runtime spec {source.spec} does not match {self.source_spec}"
            )
        spatial, pointer = self.translate_handoff_tensors(
            source.spatial_memory, source.object_pointer, source.validity
        )
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "grid_adapter": self.grid_adapter,
                "per_frame_independent": True,
                "invalid_records": self.invalid_record_policy(),
                "output_dtype": {
                    "policy": self.output_dtype,
                    "spatial_memory": _dtype_name(spatial.dtype),
                    "object_pointer": _dtype_name(pointer.dtype),
                },
                "presence_logits": "diagnostic_only",
                **self._translation_metadata(),
            },
        )

    translate = forward

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        def count(prefix: str) -> int:
            return sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name.split(".", 1)[0] == prefix
            )

        return {
            "spatial": count(self._spatial_module_prefix),
            "pointer": count(self._pointer_module_prefix),
            "total": self.parameter_count(),
        }

    def macs_per_frame(self) -> int:
        """Analytic spatial MACs per translated frame (pointer excluded)."""

        raise NotImplementedError

    def pointer_macs_per_record(self) -> int:
        raise NotImplementedError


class LinearStateTranslator(_LearnedStateTranslator):
    name = "linear"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        output_dtype: str = "float32",
    ):
        super().__init__(source_spec, target_spec, output_dtype=output_dtype)
        self.feature = nn.Linear(
            source_spec.feature_channels, target_spec.feature_channels
        )
        self.pointer = nn.Linear(source_spec.pointer_dim, target_spec.pointer_dim)

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.feature(
            tensor.to(device=self.feature.weight.device, dtype=self.feature.weight.dtype)
        )

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.pointer(
            tensor.to(device=self.pointer.weight.device, dtype=self.pointer.weight.dtype)
        )

    def macs_per_frame(self) -> int:
        positions = self.target_spec.height * self.target_spec.width
        return positions * self.feature.in_features * self.feature.out_features

    def pointer_macs_per_record(self) -> int:
        return self.pointer.in_features * self.pointer.out_features


class ResidualMLPStateTranslator(_LearnedStateTranslator):
    """Separate two-layer MLP heads; identity residuals only when shapes match.

    ``hidden_dim`` sizes the spatial head. ``pointer_hidden_dim`` defaults to
    ``hidden_dim`` for compatibility; the capacity ladder uses 128 / 512.
    """

    name = "residual_mlp"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        hidden_dim: int = 128,
        pointer_hidden_dim: int | None = None,
        output_dtype: str = "float32",
    ):
        super().__init__(source_spec, target_spec, output_dtype=output_dtype)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        pointer_hidden_dim = hidden_dim if pointer_hidden_dim is None else pointer_hidden_dim
        if pointer_hidden_dim < 1:
            raise ValueError("pointer_hidden_dim must be positive")
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
        self.feature_residual = (
            source_spec.feature_channels == target_spec.feature_channels
            and source_spec.height == target_spec.height
            and source_spec.width == target_spec.width
        )
        self.pointer_residual = source_spec.pointer_dim == target_spec.pointer_dim

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.feature[0].weight.device,
            dtype=self.feature[0].weight.dtype,
        )
        output = self.feature(calibrated)
        return output + calibrated if self.feature_residual else output

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.pointer[0].weight.device,
            dtype=self.pointer[0].weight.dtype,
        )
        output = self.pointer(calibrated)
        return output + calibrated if self.pointer_residual else output

    def macs_per_frame(self) -> int:
        positions = self.target_spec.height * self.target_spec.width
        return positions * (
            self.source_spec.feature_channels * self.hidden_dim
            + self.hidden_dim * self.target_spec.feature_channels
        )

    def pointer_macs_per_record(self) -> int:
        return (
            self.source_spec.pointer_dim * self.pointer_hidden_dim
            + self.pointer_hidden_dim * self.target_spec.pointer_dim
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize architecture metadata with CPU weights for safe reuse."""

        return {
            "schema_version": "cmmt.residual_mlp_translator.v2",
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
        if payload.get("schema_version") != "cmmt.residual_mlp_translator.v2":
            raise ValueError("unsupported residual MLP translator payload")
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
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("residual MLP payload state_dict must be a mapping")
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator


class LearnedComponentPolicyTranslator:
    """Use selected learned continuous components and Direct Copy for the rest."""

    _ALLOWED_COMPONENTS = {
        "spatial_memory",
        "object_pointer",
    }

    def __init__(
        self,
        learned: _LearnedStateTranslator,
        *,
        learned_components: tuple[str, ...],
    ):
        selected = frozenset(learned_components)
        if not selected or not selected <= self._ALLOWED_COMPONENTS:
            raise ValueError(
                "learned_components must be a non-empty subset of "
                f"{sorted(self._ALLOWED_COMPONENTS)}"
            )
        self.learned = learned
        self.learned_components = selected
        self.target_spec = learned.target_spec
        self.name = "learned_" + "_".join(
            component
            for component in sorted(self._ALLOWED_COMPONENTS)
            if component in selected
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        learned_state = self.learned.translate(source)
        direct_state = DirectCopyTranslator(self.target_spec).translate(source)

        def choose(name: str) -> torch.Tensor:
            learned_tensor = getattr(learned_state, name)
            if name in self.learned_components:
                return learned_tensor
            return getattr(direct_state, name).to(
                device=learned_tensor.device,
                dtype=learned_tensor.dtype,
            )

        return learned_state.with_continuous(
            spatial_memory=choose("spatial_memory"),
            object_pointer=choose("object_pointer"),
            presence_logits=choose("presence_logits"),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "component_policy": {
                    component: (
                        "learned" if component in self.learned_components else "direct"
                    )
                    for component in sorted(self._ALLOWED_COMPONENTS)
                },
                "presence_logits": "diagnostic_only",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.learned.parameter_count()



_CONTINUOUS_COMPONENTS = ("spatial_memory", "object_pointer")
_MOMENT_ROLES = ("conditioning", "non_conditioning")
_MOMENT_STATISTICS = (
    "spatial_source_mean",
    "spatial_source_std",
    "spatial_target_mean",
    "spatial_target_std",
    "pointer_source_mean",
    "pointer_source_std",
    "pointer_target_mean",
    "pointer_target_std",
)


class _RunningMoments:
    """Streaming per-feature mean/variance (Chan et al.) in float64."""

    def __init__(self, features: int):
        self.count = 0
        self.mean = torch.zeros(features, dtype=torch.float64)
        self.m2 = torch.zeros(features, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        """``values`` is ``[M, features]``: M observations of every feature."""

        values = values.detach().to(device="cpu", dtype=torch.float64)
        count = values.shape[0]
        if count == 0:
            return
        mean = values.mean(dim=0)
        m2 = (values - mean).square().sum(dim=0)
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * (count / total)
        self.m2 = self.m2 + m2 + delta.square() * (self.count * count / total)
        self.count = total

    def std(self) -> torch.Tensor:
        """Population standard deviation (ddof=0)."""

        return (self.m2 / self.count).clamp_min(0).sqrt()


class MomentMatchedCopyTranslator:
    """Train-split moment matching of spatial memory and object pointer.

    Frozen baseline rule (docs/design/01 section 3.2): statistics are computed
    once on fit pairs only, from valid records, separately for conditioning and
    non-conditioning records, per spatial channel and per pointer dimension::

        x_target_like = ((x - mu_source) / max(sigma_source, eps)) * sigma_target + mu_target

    There are no learned parameters and translation never sees target states.
    Padding records keep the Direct Copy value. Other fields follow Direct Copy.
    """

    name = "moment_matched_copy"
    preset: str | None = None
    payload_schema = "cmmt.moment_matched_copy.v1"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        statistics: Mapping[str, torch.Tensor],
        *,
        eps: float = 1e-6,
        output_dtype: str = "source",
        fit_summary: Mapping[str, Any] | None = None,
    ):
        if source_spec.feature_channels != target_spec.feature_channels:
            raise ValueError("moment matching needs equal source/target channel counts")
        if source_spec.pointer_dim != target_spec.pointer_dim:
            raise ValueError("moment matching needs equal source/target pointer dims")
        if output_dtype not in _OUTPUT_DTYPE_POLICIES:
            raise ValueError(f"output_dtype must be one of {_OUTPUT_DTYPE_POLICIES}")
        if not eps > 0:
            raise ValueError("eps must be positive")
        missing = sorted(set(_MOMENT_STATISTICS) - set(statistics))
        if missing:
            raise ValueError(f"moment statistics are missing: {missing}")
        roles = len(_MOMENT_ROLES)
        stats: dict[str, torch.Tensor] = {}
        for key in _MOMENT_STATISTICS:
            value = statistics[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a tensor")
            width = (
                source_spec.feature_channels
                if key.startswith("spatial")
                else source_spec.pointer_dim
            )
            if tuple(value.shape) != (roles, width):
                raise ValueError(f"{key} must have shape {(roles, width)}")
            stats[key] = value.detach().to(device="cpu", dtype=torch.float32).clone()
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.statistics = MappingProxyType(stats)
        self.eps = float(eps)
        self.output_dtype = output_dtype
        self.fit_summary = dict(fit_summary or {})
        self._affine_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def fit(
        cls,
        pairs: Iterable[tuple[CanonicalState, CanonicalState]],
        *,
        eps: float = 1e-6,
        output_dtype: str = "source",
    ) -> "MomentMatchedCopyTranslator":
        """Estimate statistics from fit-split pairs (never pass test pairs)."""

        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("at least one paired state is required")
        source_spec = pair_list[0][0].spec
        target_spec = pair_list[0][1].spec
        channels, pointer_dim = target_spec.feature_channels, target_spec.pointer_dim
        moments = {
            (component, side, role): _RunningMoments(
                channels if component == "spatial" else pointer_dim
            )
            for component in ("spatial", "pointer")
            for side in ("source", "target")
            for role in _MOMENT_ROLES
        }
        records = {role: 0 for role in _MOMENT_ROLES}
        for source, target in pair_list:
            if source.spec != source_spec or target.spec != target_spec:
                raise ValueError("all fit pairs must share one source and one target spec")
            valid = _pair_guard(source, target).cpu()
            conditioning = source.is_conditioning.cpu()
            source_spatial = _resample_spatial(
                source.spatial_memory, target_spec.height, target_spec.width
            ).cpu()
            sides = {
                "source": (source_spatial, source.object_pointer.cpu()),
                "target": (target.spatial_memory.cpu(), target.object_pointer.cpu()),
            }
            for role in _MOMENT_ROLES:
                selected = valid & (conditioning == (role == "conditioning"))
                records[role] += int(selected.sum())
                for side, (spatial, pointer) in sides.items():
                    # [n,C,H,W] -> [n*H*W, C]: one observation per position.
                    frames = spatial[selected]
                    moments[("spatial", side, role)].update(
                        frames.movedim(1, -1).reshape(-1, channels)
                    )
                    moments[("pointer", side, role)].update(pointer[selected])
        empty = [role for role, count in records.items() if count == 0]
        if empty:
            raise ValueError(f"fit pairs contain no valid {empty} records")
        statistics = {}
        for component in ("spatial", "pointer"):
            for side in ("source", "target"):
                per_role = [moments[(component, side, role)] for role in _MOMENT_ROLES]
                statistics[f"{component}_{side}_mean"] = torch.stack(
                    [moment.mean for moment in per_role]
                )
                statistics[f"{component}_{side}_std"] = torch.stack(
                    [moment.std() for moment in per_role]
                )
        return cls(
            source_spec,
            target_spec,
            statistics,
            eps=eps,
            output_dtype=output_dtype,
            fit_summary={"pairs": len(pair_list), "valid_records": records},
        )

    def _affine(
        self, component: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-role ``scale``, ``shift`` with ``x_hat = x * scale + shift``: ``[2, F]``.

        Computed once per device, so translation does no host-to-device copy.
        """

        key = (component, str(device))
        cached = self._affine_cache.get(key)
        if cached is None:
            stats = self.statistics
            scale = stats[f"{component}_target_std"].double() / stats[
                f"{component}_source_std"
            ].double().clamp_min(self.eps)
            shift = stats[f"{component}_target_mean"].double() - stats[
                f"{component}_source_mean"
            ].double() * scale
            cached = (scale.float().to(device), shift.float().to(device))
            self._affine_cache[key] = cached
        return cached

    def translate_handoff_tensors(
        self,
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        validity: torch.Tensor | None,
        is_conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,O,K,C,H,W]``, ``[B,O,K,D]`` -> moment-matched copies.

        Does not synchronize with the device (no data-dependent shapes).
        """

        if validity is None:
            validity = torch.ones(spatial.shape[:3], dtype=torch.bool, device=spatial.device)
        spatial_dtype = spatial.dtype if self.output_dtype == "source" else torch.float32
        pointer_dtype = pointer.dtype if self.output_dtype == "source" else torch.float32
        spatial = _resample_spatial(spatial, self.target_spec.height, self.target_spec.width)
        device = spatial.device
        role = (~is_conditioning.to(device)).long()  # [B,O,K]: 0 cond, 1 non-cond
        valid = validity.to(device)
        outputs = []
        for component, tensor, trailing in (
            ("spatial", spatial, (1, 1)),
            ("pointer", pointer.to(device), ()),
        ):
            scale, shift = self._affine(component, device)
            view = (*role.shape, scale.shape[-1], *trailing)
            source = tensor.float()
            matched = source * scale[role].view(view) + shift[role].view(view)
            keep = valid.view(*valid.shape, *([1] * (source.ndim - 3)))
            outputs.append(torch.where(keep, matched, source))
        return outputs[0].to(spatial_dtype), outputs[1].to(pointer_dtype)

    def translate(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        if source.spec != self.source_spec:
            raise ValueError(
                f"source runtime spec {source.spec} does not match {self.source_spec}"
            )
        spatial, pointer = self.translate_handoff_tensors(
            source.spatial_memory,
            source.object_pointer,
            source.validity,
            source.is_conditioning,
        )
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "statistics": "fit_split_only",
                "groups": "conditioning/non_conditioning x channel|dimension",
                "eps": self.eps,
                "per_frame_independent": True,
                "invalid_records": {
                    "translated": "valid_records_only",
                    "spatial_memory": "direct_copy",
                    "object_pointer": "direct_copy",
                },
                "output_dtype": {
                    "policy": self.output_dtype,
                    "spatial_memory": _dtype_name(spatial.dtype),
                    "object_pointer": _dtype_name(pointer.dtype),
                },
                "presence_logits": "diagnostic_only",
                "grid_adapter": "bilinear",
                "fit_summary": dict(self.fit_summary),
            },
        )

    def parameter_count(self) -> int:
        return 0

    def statistics_count(self) -> int:
        return sum(value.numel() for value in self.statistics.values())

    def statistics_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.statistics.values())

    def macs_per_frame(self) -> int:
        """One multiply-add per spatial element."""

        spec = self.target_spec
        return spec.feature_channels * spec.height * spec.width

    def pointer_macs_per_record(self) -> int:
        return self.target_spec.pointer_dim

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload_schema,
            "translator": self.name,
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "eps": self.eps,
            "output_dtype": self.output_dtype,
            "roles": list(_MOMENT_ROLES),
            "statistics": {key: value.clone() for key, value in self.statistics.items()},
            "fit_summary": {
                "pairs": int(self.fit_summary.get("pairs", 0)),
                "valid_records": {
                    str(key): int(value)
                    for key, value in dict(self.fit_summary.get("valid_records", {})).items()
                },
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MomentMatchedCopyTranslator":
        if payload.get("schema_version") != cls.payload_schema:
            raise ValueError("unsupported moment-matched copy payload")
        if list(payload.get("roles", [])) != list(_MOMENT_ROLES):
            raise ValueError(f"moment payload roles must be {list(_MOMENT_ROLES)}")
        statistics = payload.get("statistics")
        if not isinstance(statistics, Mapping):
            raise TypeError("moment payload statistics must be a mapping")
        return cls(
            StateSpec(**dict(payload["source_spec"])),
            StateSpec(**dict(payload["target_spec"])),
            statistics,
            eps=float(payload["eps"]),
            output_dtype=str(payload["output_dtype"]),
            fit_summary=payload.get("fit_summary"),
        )


class ComponentAblationTranslator:
    """Zero selected continuous components of another translator's output.

    Used for injection-sensitivity checks and the zero-memory arm. Only valid
    records are zeroed; padding keeps the wrapped translator's value.
    """

    def __init__(self, inner: Any, *, zero: Iterable[str] = ("spatial_memory",)):
        selected = tuple(dict.fromkeys(zero))
        if not selected or not set(selected) <= set(_CONTINUOUS_COMPONENTS):
            raise ValueError(
                f"zero must be a non-empty subset of {list(_CONTINUOUS_COMPONENTS)}, "
                f"got {list(selected)}"
            )
        self.inner = inner
        self.zero = frozenset(selected)
        self.target_spec = inner.target_spec
        self.preset = getattr(inner, "preset", None)
        inner_name = getattr(inner, "name", type(inner).__name__)
        self.name = f"{inner_name}_zero_" + "_".join(
            component for component in _CONTINUOUS_COMPONENTS if component in self.zero
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        state = self.inner.translate(source)
        spatial = state.spatial_memory
        pointer = state.object_pointer
        if "spatial_memory" in self.zero:
            valid = state.validity.to(spatial.device)
            spatial = spatial.masked_fill(valid[..., None, None, None], 0)
        if "object_pointer" in self.zero:
            valid = state.validity.to(pointer.device)
            pointer = pointer.masked_fill(valid[..., None], 0)
        return state.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=state.presence_logits.clone(),
            positional_information=state.positional_information,
            translation_metadata={
                "translator": self.name,
                "ablation": {
                    "zeroed_components": sorted(self.zero),
                    "records": "valid_only",
                },
                "inner": dict(state.metadata.get("translation", {})),
            },
        )

    def parameter_count(self) -> int:
        counter = getattr(self.inner, "parameter_count", None)
        return int(counter()) if callable(counter) else 0
