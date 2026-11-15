"""Compatibility imports and dummy-forward CLI for translator models.

Model implementations live in ``vos_memory_inspector.models``. Existing imports
and ``python -m vos_memory_inspector.transformer_translator`` remain supported.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import torch

from .state_schema import CanonicalState, StateSpec
from .models._stitched import (
    OUTPUT_DTYPES, PAYLOAD_SCHEMA, SAM21_MEMORY_SPEC, StitchedStateTranslator,
    TranslatorConfig,
)
from .models.residual_mlp import ResidualMLPStateTranslator
from .models.transformer import TransformerStateTranslator
from .models.registry import PRESETS, TranslatorPreset, build_translator, preset_names


def _millions(value: float) -> str:
    return f"{value / 1e6:.1f}M"


def _dummy_state(
    *, batch: int, objects: int, records: int, invalid: int, seed: int, spec: StateSpec
) -> CanonicalState:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, objects, records)
    spatial = torch.randn(
        *shape, spec.feature_channels, spec.height, spec.width, generator=generator
    ).to(torch.bfloat16)  # runtime dtype of maskmem_features
    pointer = torch.randn(*shape, spec.pointer_dim, generator=generator)
    validity = torch.ones(shape, dtype=torch.bool)
    if invalid:
        validity.view(-1)[-invalid:] = False  # padding at the end of the record order
    frames = torch.arange(records).view(1, 1, records).expand(shape).clone()
    conditioning = torch.zeros(shape, dtype=torch.bool)
    conditioning[..., 0] = True
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=torch.zeros(*shape, 1),
        frame_indices=frames,
        slot_order=frames.clone(),
        is_conditioning=conditioning,
        validity=validity,
        object_ids=tuple(range(1, objects + 1)),
        switch_frame=records - 1,
        metadata={"synthetic": True},
    ).validate()


def pytorch_flops(function: Any, *args: Any) -> int:
    """FlopCounterMode total (matmul/conv only). SDPA runs on the math backend
    so attention matmuls are counted on every device."""

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        counter = FlopCounterMode(display=False)
        with counter:
            function(*args)
    return int(counter.get_total_flops())


def summarize_dummy_forward(
    translator: StitchedStateTranslator,
    *,
    batch: int = 1,
    objects: int = 2,
    records: int = 3,
    invalid: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    total = batch * objects * records
    if min(batch, objects, records) < 1:
        raise ValueError("batch, objects and records must be positive")
    if not 0 <= invalid <= total:
        raise ValueError(f"invalid must be in [0, {total}]")
    state = _dummy_state(
        batch=batch,
        objects=objects,
        records=records,
        invalid=invalid,
        seed=seed,
        spec=translator.source_spec,
    )
    captured: dict[str, Any] = {"frames": None, "attention": []}
    hooks = [
        translator.spatial.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("frames", tuple(inputs[0].shape))
        )
    ]
    for block in translator.spatial.blocks:
        if getattr(block, "attn", None) is not None:
            hooks.append(
                block.attn.register_forward_pre_hook(
                    lambda _module, inputs: captured["attention"].append(tuple(inputs[0].shape))
                )
            )
    try:
        with torch.inference_mode():
            translated = translator.translate(state)
    finally:
        for hook in hooks:
            hook.remove()
    valid = state.validity
    if int(valid.sum()):
        spatial_error = (
            translated.spatial_memory[valid].float() - state.spatial_memory[valid].float()
        ).abs().max().item()
        pointer_error = (
            translated.object_pointer[valid].float() - state.object_pointer[valid].float()
        ).abs().max().item()
    else:
        spatial_error = pointer_error = 0.0
    spec = translator.source_spec
    one_frame = torch.zeros(
        1, spec.feature_channels, spec.height, spec.width,
        device=next(translator.parameters()).device,
    )
    return {
        "preset": translator.preset,
        "description": translator.description,
        "input_spatial": tuple(state.spatial_memory.shape),
        "input_dtype": str(state.spatial_memory.dtype).removeprefix("torch."),
        "valid_records": int(valid.sum()),
        "total_records": total,
        "frames": captured["frames"],
        "attention_inputs": captured["attention"],
        "output_spatial": tuple(translated.spatial_memory.shape),
        "output_dtype": str(translated.spatial_memory.dtype).removeprefix("torch."),
        "pointer_in": tuple(state.object_pointer.shape),
        "pointer_out": tuple(translated.object_pointer.shape),
        "identity_spatial_max_abs": spatial_error,
        "identity_pointer_max_abs": pointer_error,
        "pytorch_flops_per_frame": pytorch_flops(translator.spatial, one_frame),
        "architecture": translator.architecture_summary(),
    }


def format_dummy_forward_summary(summary: Mapping[str, Any]) -> str:
    arch = summary["architecture"]
    params = arch["parameters"]
    macs = arch["macs_by_module"]
    frames = summary["frames"]
    lines = [
        f"preset             : {summary['preset']} ({summary['description']})",
        f"input spatial      : {summary['input_spatial']} {summary['input_dtype']}  "
        f"valid {summary['valid_records']}/{summary['total_records']} -> frames "
        + (str(frames) if frames is not None else "none (translator not called)"),
        f"output spatial     : {summary['output_spatial']} {summary['output_dtype']}",
        f"object pointer     : {summary['pointer_in']} -> {summary['pointer_out']}",
        f"config             : {arch['config']}  (start: {arch['start']})",
        f"trainable params   : spatial {params['spatial']:,} | pointer {params['pointer']:,} "
        f"| total {params['total']:,}",
        f"MACs per frame     : ~{_millions(arch['macs_per_frame'])} (stitch "
        f"{_millions(macs['stitch'])} + feed-forward {_millions(macs['feed_forward'])} "
        f"+ attention {_millions(macs['attention'])})",
        f"PyTorch FLOPs/frame: {_millions(summary['pytorch_flops_per_frame'])} (FlopCounterMode, "
        "matmul/conv only; ~2 FLOPs per MAC, a different unit from MACs)",
        f"identity at init   : max|T(M)-M| = {summary['identity_spatial_max_abs']:.3g} | "
        f"max|T(p)-p| = {summary['identity_pointer_max_abs']:.3g}",
    ]
    attention = arch["attention"]
    if attention is not None:
        lines.append(
            f"attention scope    : per frame, {attention['queries']} queries x "
            f"{attention['keys']} keys, {attention['heads']} head(s), sr_ratio "
            f"{attention['sr_ratio']} (no cross-frame attention)"
        )
    else:
        lines.append("attention scope    : none (position-wise feed-forward)")
    lines.extend(["", f"{'module':<24}{'params':>12}{'MACs/frame':>14}"])
    params_by_module = arch["spatial_parameters_by_module"]
    for module, count in params_by_module.items():
        cell = _millions(macs[module]) if module in macs else "-"
        lines.append(f"{module:<24}{count:>12,}{cell:>14}")
    lines.append(f"{'pointer':<24}{params['pointer']:>12,}{_millions(arch['pointer_macs_per_record']):>14}  (per record)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vos_memory_inspector.transformer_translator",
        description="Dummy forward summary of a residual_mlp or transformer translator preset.",
    )
    parser.add_argument(
        "--preset",
        default="transformer",
        choices=(*preset_names(kind="residual_mlp"), *preset_names(kind="transformer")),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--records", type=int, default=3)
    parser.add_argument("--invalid", type=int, default=1, help="padding records (end of order)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    translator = build_translator(args.preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    try:
        summary = summarize_dummy_forward(
            translator,
            batch=args.batch,
            objects=args.objects,
            records=args.records,
            invalid=args.invalid,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_dummy_forward_summary(summary))
    return 0


if __name__ == "__main__":
    # Run the package-qualified module so its classes are the ones other
    # modules import.
    from vos_memory_inspector.transformer_translator import main as _main

    raise SystemExit(_main())
