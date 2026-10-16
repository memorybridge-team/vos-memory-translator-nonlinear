"""모델팀 746ea3e의 tensor API 선별 이관. 클래스/함수 body는 원본과 동일."""
from __future__ import annotations
from typing import Any
import math
import torch
from torch import nn
from torch.nn import functional as F
from .state_schema import CanonicalState, StateSpec
_OUTPUT_DTYPE_POLICIES = ("source", "float32")

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

def _target_positional(spec: StateSpec) -> dict[str, str]:
    return {"policy": spec.positional_policy}

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
