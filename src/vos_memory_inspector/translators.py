"""Component-wise CMMT baselines for SAM 2-style canonical state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

# Re-export the old names for existing callers and serialized Python objects.
from .models._base import (
    _OUTPUT_DTYPE_POLICIES,
    _LearnedStateTranslator,
    _dtype_name,
    _gather_records,
    _padding_only,
    _resample_frames,
    _scatter_records,
    _target_positional,
    _valid_record_index,
)
from .models.residual_mlp import ResidualMLPStateTranslator
from .state_schema import CanonicalState, StateSpec, validate_paired_state_contract


def _resample_spatial(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim != 6:
        raise ValueError("spatial input must be [B,O,K,C,H,W]")
    if tensor.shape[-2:] == (height, width):
        return tensor
    batch, objects, records, channels, old_h, old_w = tensor.shape
    flat = tensor.reshape(batch * objects * records, channels, old_h, old_w)
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(batch, objects, records, channels, height, width)


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
