"""Shared model/preset registry used by training, handoff and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..state_schema import CanonicalState, StateSpec
from ..translators import DirectCopyTranslator, LinearStateTranslator, MomentMatchedCopyTranslator
from ._spatial import SpatialTransformerConfig
from .mlp_context import MLPContextStateTranslator
from .residual_mlp import ResidualMLPStateTranslator
from .transformer import TransformerStateTranslator


@dataclass(frozen=True)
class TranslatorPreset:
    """Registry entry. ``config`` holds overrides of ``SpatialTransformerConfig()``."""

    name: str
    kind: str  # "spatial_transformer" | "linear" | "residual_mlp" | "direct" | "moment_match"
    group: str
    description: str
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    training: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def learned(self) -> bool:
        return self.kind in {"spatial_transformer", "linear", "residual_mlp"}


def _preset(
    name: str,
    kind: str,
    group: str,
    description: str,
    *,
    config: Mapping[str, Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> TranslatorPreset:
    return TranslatorPreset(
        name,
        kind,
        group,
        description,
        MappingProxyType(dict(config or {})),
        MappingProxyType(dict(kwargs or {})),
        MappingProxyType(dict(training or {})),
    )


_ST = "spatial_transformer"
_AVG_SINCOS = {"patch_embed": "avgpool", "pos_embed": "sincos"}
_NO_CONTEXT_MLP = {"spatial_context": False, "local_fusion": "mlp"}

PRESETS: Mapping[str, TranslatorPreset] = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            # Capacity ladder (repository classes).
            _preset("direct", "direct", "ladder", "Direct Copy"),
            _preset("moment_match", "moment_match", "ladder", "Moment-Matched Copy (fit split only)"),
            _preset("linear", "linear", "ladder", "position-shared Linear 64->64, pointer 256->256"),
            _preset(
                "residual_mlp",
                "residual_mlp",
                "ladder",
                "Residual MLP 64->128->64, pointer 256->512->256",
                kwargs={"hidden_dim": 128, "pointer_hidden_dim": 512},
            ),
            # Main 2x2: local head {linear, MLP} x spatial context {off, on}.
            _preset("linear_local", _ST, "2x2", "linear local head, no context", config={"spatial_context": False}),
            _preset("base", _ST, "2x2", "linear local head + spatial context"),
            _preset("no_context", _ST, "2x2", "MLP local head (128), no context", config=_NO_CONTEXT_MLP),
            _preset("base_mlp", _ST, "2x2", "MLP local head + spatial context", config={"local_fusion": "mlp"}),
            _preset("mlp_context", _ST, "structure", "independent residual MLP + gated spatial context", config={"local_fusion": "additive_mlp"}),
            # Ablations of base (one change each unless noted).
            _preset("context_only", _ST, "structure", "Delta = Up(Conv3x3(ctx))", config={"local_fusion": "none"}),
            _preset("no_attn_context", _ST, "structure", "attention sublayers removed", config={"use_attention": False}),
            _preset("res8", _ST, "A_resolution", "avgpool 8 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 8}),
            _preset("res16", _ST, "A_resolution", "avgpool 4 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 4}),
            _preset("res32", _ST, "A_resolution", "avgpool 2 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 2}),
            _preset("grid8", _ST, "A_prime_grid", "conv patch 8, learnable PE", config={"patch_size": 8}),
            _preset("grid32", _ST, "A_prime_grid", "conv patch 2, learnable PE", config={"patch_size": 2}),
            _preset("depth1", _ST, "B_depth", "1 layer", config={"num_layers": 1}),
            _preset("depth4", _ST, "B_depth", "4 layers", config={"num_layers": 4}),
            _preset("heads2", _ST, "C_heads", "2 heads", config={"num_heads": 2}),
            _preset("heads8", _ST, "C_heads", "8 heads", config={"num_heads": 8}),
            _preset("no_residual", _ST, "D_residual", "M_hat = Delta", config={"use_residual": False}),
            _preset(
                "mlp256",
                _ST,
                "E_control",
                "no_context with hidden 256 (base_mlp compute)",
                config={**_NO_CONTEXT_MLP, "fusion_hidden_dim": 256},
            ),
            _preset(
                "mlp_param",
                _ST,
                "E_control",
                "no_context with hidden 1505 (base params)",
                config={**_NO_CONTEXT_MLP, "fusion_hidden_dim": 1505},
            ),
            _preset("no_pos", _ST, "F_pe", "no positional embedding", config={"pos_embed": "none"}),
            _preset("sincos_pos", _ST, "F_pe", "fixed sincos PE", config={"pos_embed": "sincos"}),
            _preset("ptr128", _ST, "pointer", "pointer hidden 128", config={"pointer_hidden_dim": 128}),
            _preset(
                "no_cos",
                _ST,
                "loss",
                "base model trained with lambda_cos = 0 (pipeline setting)",
                training={"lambda_cos": 0.0},
            ),
        )
    }
)

_LADDER_OVERRIDES = {
    "linear": {"output_dtype"},
    "residual_mlp": {"hidden_dim", "pointer_hidden_dim", "output_dtype"},
    "moment_match": {"eps", "output_dtype"},
    "direct": set(),
}


def preset_names(*, learned: bool | None = None, kind: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, preset in PRESETS.items()
        if (learned is None or preset.learned == learned) and (kind is None or preset.kind == kind)
    )


def build_translator(
    name: str,
    source_spec: StateSpec,
    target_spec: StateSpec,
    *,
    config: SpatialTransformerConfig | None = None,
    fit_pairs: Iterable[tuple[CanonicalState, CanonicalState]] | None = None,
    **overrides: Any,
) -> Any:
    """Instantiate a registered preset.

    Spatial-transformer presets accept ``config=`` (replaces the preset config)
    and ``SpatialTransformerConfig`` field overrides. Ladder translators use
    ``output_dtype="source"`` unless overridden so that handoff bytes match
    Direct Copy. ``moment_match`` requires ``fit_pairs`` from the fit split.
    """

    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from exc
    if fit_pairs is not None and preset.kind != "moment_match":
        raise ValueError("fit_pairs is only used by the moment_match preset")
    if config is not None and preset.kind != _ST:
        raise ValueError(f"config= applies only to spatial-transformer presets, not {name!r}")
    if preset.kind == _ST:
        base = config if config is not None else SpatialTransformerConfig(**dict(preset.config))
        known = {item.name for item in fields(SpatialTransformerConfig)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown overrides for {name!r}: {unknown}")
        effective_config = replace(base, **overrides)
        model_type = (
            MLPContextStateTranslator
            if effective_config.local_fusion == "additive_mlp"
            else TransformerStateTranslator
        )
        translator: Any = model_type(
            source_spec, target_spec, config=effective_config, preset=name
        )
        return translator
    unknown = sorted(set(overrides) - _LADDER_OVERRIDES[preset.kind])
    if unknown:
        raise ValueError(f"unknown overrides for {name!r}: {unknown}")
    options = {**dict(preset.kwargs), **overrides}
    if preset.kind == "direct":
        translator = DirectCopyTranslator(target_spec)
    elif preset.kind == "moment_match":
        if fit_pairs is None:
            raise ValueError(
                "moment_match needs fit_pairs (fit split only); it has no untrained form"
            )
        options.setdefault("output_dtype", "source")
        translator = MomentMatchedCopyTranslator.fit(fit_pairs, **options)
        if translator.source_spec != source_spec or translator.target_spec != target_spec:
            raise ValueError("fit_pairs specs do not match source_spec/target_spec")
    elif preset.kind == "linear":
        options.setdefault("output_dtype", "source")
        translator = LinearStateTranslator(source_spec, target_spec, **options)
    else:
        options.setdefault("output_dtype", "source")
        translator = ResidualMLPStateTranslator(source_spec, target_spec, **options)
    translator.preset = name
    return translator
