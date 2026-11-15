"""Shared model/preset registry used by training, handoff and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..state_schema import CanonicalState, StateSpec
from ..translators import DirectCopyTranslator, LinearStateTranslator, MomentMatchedCopyTranslator
from ._stitched import TranslatorConfig
from .residual_mlp import ResidualMLPStateTranslator
from .transformer import TransformerStateTranslator


@dataclass(frozen=True)
class TranslatorPreset:
    """Registry entry. ``config`` holds overrides of ``TranslatorConfig()``."""

    name: str
    kind: str  # "transformer" | "residual_mlp" | "linear" | "direct" | "moment_match"
    group: str
    description: str
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    training: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def learned(self) -> bool:
        return self.kind in {"transformer", "linear", "residual_mlp"}


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


_STITCHED = {
    "residual_mlp": ResidualMLPStateTranslator,
    "transformer": TransformerStateTranslator,
}

PRESETS: Mapping[str, TranslatorPreset] = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            # Capacity ladder: copy, moment matching, linear stitching, then the two models.
            _preset("direct", "direct", "ladder", "Direct Copy"),
            _preset("moment_match", "moment_match", "ladder", "Moment-Matched Copy (fit split only)"),
            _preset("linear", "linear", "ladder", "position-shared Linear 64->64, pointer 256->256"),
            _preset(
                "residual_mlp",
                "residual_mlp",
                "model",
                "stitching layer + Pre-LN feed-forward block (64->256->64), linear pointer",
            ),
            _preset(
                "transformer",
                "transformer",
                "model",
                "stitching layer + Pre-LN Transformer block (spatial-reduction attention, "
                "feed-forward 64->256->64), linear pointer",
            ),
            # Control for the comparison of the two models: the MLP model widened to
            # the spatial parameter count of the Transformer model (119,936 vs 119,872).
            _preset(
                "residual_mlp_matched",
                "residual_mlp",
                "control",
                "residual_mlp with feed-forward ratio 14 (parameter-matched to transformer)",
                config={"mlp_ratio": 14.0},
            ),
        )
    }
)

_LADDER_OVERRIDES = {
    "linear": {"output_dtype"},
    "moment_match": {"eps", "output_dtype"},
    "direct": set(),
}
# ``attention`` is fixed by the model class and cannot be overridden.
_CONFIG_OVERRIDES = {item.name for item in fields(TranslatorConfig)} - {"attention"}


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
    config: TranslatorConfig | None = None,
    fit_pairs: Iterable[tuple[CanonicalState, CanonicalState]] | None = None,
    **overrides: Any,
) -> Any:
    """Instantiate a registered preset.

    ``residual_mlp`` and ``transformer`` presets accept ``config=`` (replaces
    the preset config) and ``TranslatorConfig`` field overrides. They default
    to ``output_dtype="source"`` so that handoff bytes match Direct Copy, as do
    the ladder translators. ``moment_match`` requires ``fit_pairs`` from the
    fit split.
    """

    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from exc
    if fit_pairs is not None and preset.kind != "moment_match":
        raise ValueError("fit_pairs is only used by the moment_match preset")
    if preset.kind in _STITCHED:
        base = config if config is not None else TranslatorConfig(**dict(preset.config))
        unknown = sorted(set(overrides) - _CONFIG_OVERRIDES)
        if unknown:
            raise ValueError(f"unknown overrides for {name!r}: {unknown}")
        return _STITCHED[preset.kind](
            source_spec, target_spec, config=replace(base, **overrides), preset=name
        )
    if config is not None:
        raise ValueError(f"config= applies only to residual_mlp/transformer presets, not {name!r}")
    unknown = sorted(set(overrides) - _LADDER_OVERRIDES[preset.kind])
    if unknown:
        raise ValueError(f"unknown overrides for {name!r}: {unknown}")
    options = {**dict(preset.kwargs), **overrides}
    if preset.kind == "direct":
        translator: Any = DirectCopyTranslator(target_spec)
    elif preset.kind == "moment_match":
        if fit_pairs is None:
            raise ValueError(
                "moment_match needs fit_pairs (fit split only); it has no untrained form"
            )
        options.setdefault("output_dtype", "source")
        translator = MomentMatchedCopyTranslator.fit(fit_pairs, **options)
        if translator.source_spec != source_spec or translator.target_spec != target_spec:
            raise ValueError("fit_pairs specs do not match source_spec/target_spec")
    else:
        options.setdefault("output_dtype", "source")
        translator = LinearStateTranslator(source_spec, target_spec, **options)
    translator.preset = name
    return translator
