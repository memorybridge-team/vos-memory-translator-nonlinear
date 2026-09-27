"""Translator cost benchmark (plan Section 8 latency protocol; Task 7).

Primary number, ``latency_e2e_eager``: the handoff path

    valid gather -> dtype conversion -> translator -> output cast -> scatter

run eagerly on inputs that are already on the device, with
``torch.cuda.synchronize()`` before and after every call (warmup 50, 300
timed calls, median and p90). Not timed: host<->device copies, CanonicalState
validation, translation metadata, Base+ injection and continuation.

``latency_e2e_optimized`` replays the same path from a CUDA graph captured per
input with a precomputed valid-record index. It is reported separately and is
not used for model selection. Local GPU numbers are development references; the
selection rule uses one fixed benchmark environment.

Run ``python -m vos_memory_inspector.translator_benchmark --help`` for the
preset cost table.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import datetime as _dt
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .device import resolve_device
from .state_schema import CanonicalState, StateSpec
from .transformer_translator import (
    PRESETS,
    SAM21_MEMORY_SPEC,
    _dummy_state,
    build_translator,
    pytorch_flops,
)
from .translators import (
    DirectCopyTranslator,
    MomentMatchedCopyTranslator,
    _LearnedStateTranslator,
    _valid_record_index,
)


SCHEMA_VERSION = "cmmt.translator_benchmark.v1"
COST_TABLE_SCHEMA = "cmmt.translator_cost_table.v1"
PROTOCOL_ID = "cmmt.translator_latency_protocol.v1"
PROTOCOL_WARMUP = 50
PROTOCOL_ITERS = 300
PROTOCOL_VALID_RECORDS = (1, 7, 11, 16)
PROTOCOL_TF32 = {"matmul_allow_tf32": False, "cudnn_allow_tf32": True}
PATH_STEPS = ("valid_gather", "dtype_conversion", "translator", "output_cast", "scatter")
EXCLUDED_STEPS = (
    "host_device_copy",
    "canonical_state_validation",
    "translation_metadata",
    "base_plus_injection_and_continuation",
)
COST_UNITS = {
    "analytic_macs": (
        "multiply-accumulates of convolutions, linear layers and the attention "
        "matmuls (QK^T, AV); norms, softmax, GELU, upsampling and additions excluded"
    ),
    "pytorch_flops": (
        "torch.utils.flop_counter.FlopCounterMode total with SDPA on the math "
        "backend; counts matmul/conv only (about 2 FLOPs per MAC); a different "
        "unit from analytic MACs"
    ),
}

Callables = tuple[Callable[[], tuple[torch.Tensor, torch.Tensor]], Callable[[], tuple[torch.Tensor, torch.Tensor]]]


# --------------------------------------------------------------------------
# Protocol settings and environment
# --------------------------------------------------------------------------


# Per-backend precision attributes of torch>=2.9, parents before children.
_PRECISION_PATHS = (
    "fp32_precision",
    "cuda.matmul.fp32_precision",
    "cudnn.fp32_precision",
    "cudnn.conv.fp32_precision",
    "cudnn.rnn.fp32_precision",
    "mkldnn.fp32_precision",
    "mkldnn.matmul.fp32_precision",
    "mkldnn.conv.fp32_precision",
    "mkldnn.rnn.fp32_precision",
)


def _precision_owner(path: str) -> tuple[Any, str] | None:
    *parents, attribute = path.split(".")
    owner: Any = torch.backends
    for part in parents:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner, attribute


def backend_precision_snapshot() -> dict[str, str]:
    """Readable per-backend ``fp32_precision`` values (empty before torch 2.9)."""

    snapshot = {}
    for path in _PRECISION_PATHS:
        located = _precision_owner(path)
        value = getattr(located[0], located[1], None) if located is not None else None
        if isinstance(value, str):
            snapshot[path] = value
    return snapshot


def restore_backend_precision(snapshot: Mapping[str, str]) -> None:
    for path in _PRECISION_PATHS:
        if path in snapshot:
            owner, attribute = _precision_owner(path)
            setattr(owner, attribute, snapshot[path])


@contextmanager
def protocol_backend_flags() -> Iterator[dict[str, Any]]:
    """Set the protocol TF32 flags explicitly; restore the caller's flags after.

    Protocol: matmul TF32 off, cuDNN TF32 on (the PyTorch defaults). With the
    per-backend ``fp32_precision`` API (torch>=2.9) the flags are set and
    restored through it: mixing it with the legacy ``allow_tf32`` flags can make
    the legacy getters raise.
    """

    saved = backend_precision_snapshot()
    new_api = "cuda.matmul.fp32_precision" in saved
    if new_api:
        for path, value in (
            ("cuda.matmul.fp32_precision", "ieee"),  # matmul_allow_tf32 = False
            ("cudnn.conv.fp32_precision", "tf32"),  # cudnn_allow_tf32 = True
            ("cudnn.rnn.fp32_precision", "tf32"),
        ):
            owner, attribute = _precision_owner(path)
            setattr(owner, attribute, value)
        legacy_saved = None
    else:
        legacy_saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = PROTOCOL_TF32["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = PROTOCOL_TF32["cudnn_allow_tf32"]
    try:
        precision = backend_precision_snapshot()
        if new_api:
            matmul_tf32 = precision["cuda.matmul.fp32_precision"] == "tf32"
            cudnn_tf32 = precision.get("cudnn.conv.fp32_precision") == "tf32"
        else:
            matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
            cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        yield {
            "inference_mode": True,
            "compute_dtype": "float32",
            "tf32": {
                "matmul_allow_tf32": matmul_tf32,
                "cudnn_allow_tf32": cudnn_tf32,
                "api": "fp32_precision" if new_api else "allow_tf32",
                "backend_fp32_precision": precision or None,
            },
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }
    finally:
        if legacy_saved is None:
            restore_backend_precision(saved)
        else:
            torch.backends.cuda.matmul.allow_tf32 = legacy_saved[0]
            torch.backends.cudnn.allow_tf32 = legacy_saved[1]


def _run(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _code_revision() -> dict[str, Any] | None:
    root = Path(__file__).resolve().parent
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if commit is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def benchmark_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "code_revision": _code_revision(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        driver = _run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(device.index if device.index is not None else torch.cuda.current_device()),
            ]
        )
        environment.update(
            gpu=properties.name,
            gpu_capability=f"{properties.major}.{properties.minor}",
            gpu_memory_bytes=int(properties.total_memory),
            gpu_multiprocessors=int(properties.multi_processor_count),
            driver=driver,
        )
    return environment


def _protocol_block(
    *, device: torch.device, warmup: int, iters: int, synthetic_records: Sequence[int] | None
) -> dict[str, Any]:
    deviations = []
    if warmup != PROTOCOL_WARMUP:
        deviations.append(f"warmup={warmup} (protocol {PROTOCOL_WARMUP})")
    if iters != PROTOCOL_ITERS:
        deviations.append(f"iters={iters} (protocol {PROTOCOL_ITERS})")
    if device.type != "cuda":
        deviations.append(f"device={device} (CUDA is the reference; CPU is informational)")
    if synthetic_records is not None and tuple(synthetic_records) != PROTOCOL_VALID_RECORDS:
        deviations.append(
            f"synthetic valid_records={list(synthetic_records)} "
            f"(protocol {list(PROTOCOL_VALID_RECORDS)})"
        )
    return {
        "id": PROTOCOL_ID,
        "primary": "latency_e2e_eager",
        "path": list(PATH_STEPS),
        "excluded": list(EXCLUDED_STEPS),
        "warmup": warmup,
        "iters": iters,
        "statistics": ["median", "p90"],
        "timer": "time.perf_counter_ns wall clock per call",
        "synchronize": "torch.cuda.synchronize() before and after every call (CUDA)",
        "gc_disabled_during_timing": True,
        "optimized": "CUDA graph per input, precomputed valid-record index; not used for selection",
        "conforming": not deviations,
        "deviations": deviations,
    }


# --------------------------------------------------------------------------
# Translator adapters
# --------------------------------------------------------------------------


def _normalized_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(resolve_device(device))
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _module_device(translator: Any) -> torch.device | None:
    if isinstance(translator, nn.Module):
        for parameter in translator.parameters():
            return _normalized_device(parameter.device)
    return None


def _source_spec(translator: Any) -> StateSpec:
    spec = getattr(translator, "source_spec", None)
    return spec if spec is not None else translator.target_spec


def _handoff_callables(
    translator: Any,
    spatial: torch.Tensor,
    pointer: torch.Tensor,
    validity: torch.Tensor,
    conditioning: torch.Tensor,
) -> Callables:
    """(eager, static) callables for the handoff path on device tensors.

    ``eager`` is the public handoff API: it derives the valid index from
    ``validity`` on every call (``nonzero`` synchronizes). ``static`` performs
    the same computation from a precomputed index, so it can be graph-captured.
    """

    if isinstance(translator, _LearnedStateTranslator):
        index = _valid_record_index(validity, spatial.device)
        spatial_dtype, pointer_dtype = translator.output_dtypes(spatial.dtype, pointer.dtype)

        def eager() -> tuple[torch.Tensor, torch.Tensor]:
            return translator.translate_handoff_tensors(spatial, pointer, validity)

        def static() -> tuple[torch.Tensor, torch.Tensor]:
            return translator._translate_indexed(
                spatial,
                pointer,
                index,
                spatial_dtype=spatial_dtype,
                pointer_dtype=pointer_dtype,
            )

        return eager, static
    if isinstance(translator, MomentMatchedCopyTranslator):

        def moment() -> tuple[torch.Tensor, torch.Tensor]:
            return translator.translate_handoff_tensors(spatial, pointer, validity, conditioning)

        return moment, moment
    if isinstance(translator, DirectCopyTranslator):

        def direct() -> tuple[torch.Tensor, torch.Tensor]:
            return translator.translate_handoff_tensors(spatial, pointer, validity)

        return direct, direct
    raise TypeError(
        "benchmark_translator supports learned translators, DirectCopyTranslator and "
        f"MomentMatchedCopyTranslator; got {type(translator).__name__}"
    )


def _describe(translator: Any) -> dict[str, Any]:
    description = getattr(translator, "description", None)
    return {
        "name": getattr(translator, "name", type(translator).__name__),
        "class": type(translator).__name__,
        "preset": getattr(translator, "preset", None),
        "description": description,
        "output_dtype": getattr(translator, "output_dtype", "source"),
    }


def _parameters(translator: Any) -> dict[str, int]:
    if isinstance(translator, _LearnedStateTranslator):
        return translator.parameter_breakdown()
    return {"spatial": 0, "pointer": 0, "total": 0}


def _analytic_costs(translator: Any) -> tuple[int, int]:
    """(spatial MACs per frame, pointer MACs per record)."""

    if isinstance(translator, (_LearnedStateTranslator, MomentMatchedCopyTranslator)):
        return int(translator.macs_per_frame()), int(translator.pointer_macs_per_record())
    return 0, 0


def _flops_per_frame(translator: Any, device: torch.device) -> int:
    """FlopCounterMode FLOPs of the spatial translation of one frame."""

    spec = _source_spec(translator)
    if isinstance(translator, _LearnedStateTranslator):
        reference = translator._compute_reference()
        frame = torch.zeros(
            1, spec.feature_channels, spec.height, spec.width,
            device=device, dtype=reference.dtype,
        )
        return pytorch_flops(translator._translate_frames, frame)
    # Direct Copy and Moment Match: elementwise only, measured on one record.
    state = _dummy_state(batch=1, objects=1, records=1, invalid=0, seed=0, spec=spec)
    eager, _ = _handoff_callables(
        translator,
        state.spatial_memory.to(device),
        state.object_pointer.to(device),
        state.validity.to(device),
        state.is_conditioning.to(device),
    )
    return pytorch_flops(eager)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _latency_statistics(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),  # linear interpolation
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples": int(values.size),
    }


def _time_calls(
    function: Callable[[], Any], *, device: torch.device, warmup: int, iters: int
) -> dict[str, Any]:
    cuda = device.type == "cuda"

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        function()
    synchronize()
    samples: list[float] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iters):
            synchronize()
            start = time.perf_counter_ns()
            function()
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        if gc_enabled:
            gc.enable()
    return _latency_statistics(samples)


def _peak_memory(function: Callable[[], Any], device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    outputs = function()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del outputs
    return {
        "peak_allocated_bytes": int(peak),
        "baseline_allocated_bytes": int(baseline),
        "peak_delta_bytes": int(peak - baseline),
    }


def _capture_cuda_graph(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, torch.Tensor]]:
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    return graph, outputs


def _max_abs_difference(
    first: tuple[torch.Tensor, torch.Tensor], second: tuple[torch.Tensor, torch.Tensor]
) -> float:
    return max(
        float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        for left, right in zip(first, second, strict=True)
    )


def _benchmark_entry(
    translator: Any,
    state: CanonicalState,
    *,
    label: str,
    device: torch.device,
    warmup: int,
    iters: int,
    optimized: bool,
    macs: tuple[int, int],
) -> dict[str, Any]:
    state.validate()
    # Inputs are placed on the device before any timing (not part of the path).
    spatial = state.spatial_memory.to(device)
    pointer = state.object_pointer.to(device)
    validity = state.validity.to(device)
    conditioning = state.is_conditioning.to(device)
    eager, static = _handoff_callables(translator, spatial, pointer, validity, conditioning)
    valid = int(state.validity.sum())
    reference = eager()
    output = {
        "spatial_shape": list(reference[0].shape),
        "spatial_dtype": str(reference[0].dtype).removeprefix("torch."),
        "pointer_shape": list(reference[1].shape),
        "pointer_dtype": str(reference[1].dtype).removeprefix("torch."),
    }
    output["matches_source_layout"] = (
        tuple(reference[0].shape) == tuple(state.spatial_memory.shape)
        and reference[0].dtype == state.spatial_memory.dtype
        and tuple(reference[1].shape) == tuple(state.object_pointer.shape)
        and reference[1].dtype == state.object_pointer.dtype
    )
    translated = translator.translate(
        dataclasses.replace(state, spatial_memory=spatial, object_pointer=pointer)
    )
    direct = DirectCopyTranslator(translator.target_spec).translate(state)
    spatial_macs, pointer_macs = macs
    entry: dict[str, Any] = {
        "label": label,
        "records": list(state.spatial_memory.shape[:3]),
        "valid_records": valid,
        "total_records": int(state.validity.numel()),
        "analytic_macs_spatial": valid * spatial_macs,
        "analytic_macs_pointer": valid * pointer_macs,
        "analytic_macs": valid * (spatial_macs + pointer_macs),
        "pytorch_flops": pytorch_flops(eager),
        "handoff_bytes": translated.handoff_bytes(),
        "direct_copy_handoff_bytes": direct.handoff_bytes(),
        "output": output,
    }
    del translated, direct
    entry["latency_e2e_eager"] = _time_calls(eager, device=device, warmup=warmup, iters=iters)
    if device.type == "cuda":
        entry["peak_memory"] = _peak_memory(eager, device)
        entry["peak_memory_reason"] = None
    else:
        entry["peak_memory"] = None
        entry["peak_memory_reason"] = "peak memory is tracked for CUDA devices only"
    entry["latency_e2e_optimized"] = None
    if not optimized:
        entry["latency_e2e_optimized_reason"] = "disabled by caller"
    elif device.type != "cuda":
        entry["latency_e2e_optimized_reason"] = f"CUDA graph needs a CUDA device (device={device})"
    else:
        try:
            graph, graph_outputs = _capture_cuda_graph(static, device)
        except Exception as exc:  # capture failures are reported, not fatal
            torch.cuda.synchronize(device)
            entry["latency_e2e_optimized_reason"] = f"CUDA graph capture failed: {exc!r}"[:300]
        else:
            timing = _time_calls(graph.replay, device=device, warmup=warmup, iters=iters)
            graph.replay()
            torch.cuda.synchronize(device)
            entry["latency_e2e_optimized"] = {
                "method": "cuda_graph",
                **timing,
                "max_abs_diff_vs_eager": _max_abs_difference(graph_outputs, reference),
            }
            entry["latency_e2e_optimized_reason"] = None
            del graph, graph_outputs
    return entry


def synthetic_handoff_state(
    valid_records: int, spec: StateSpec = SAM21_MEMORY_SPEC, *, seed: int = 0
) -> CanonicalState:
    """``B=O=1, K=N`` all-valid state: spatial bf16, pointer fp32, record 0 conditioning."""

    if isinstance(valid_records, bool) or not isinstance(valid_records, int) or valid_records < 1:
        raise ValueError(f"valid_records must be positive ints, got {valid_records!r}")
    return _dummy_state(batch=1, objects=1, records=valid_records, invalid=0, seed=seed, spec=spec)


def _median_over_entries(entries: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [entry[key]["median_ms"] for entry in entries if entry.get(key) is not None]
    return float(np.median(values)) if values else None


def benchmark_translator(
    translator: Any,
    valid_records: Sequence[int] = PROTOCOL_VALID_RECORDS,
    *,
    device: str | torch.device | None = None,
    states: Sequence[CanonicalState] | None = None,
    labels: Sequence[str] | None = None,
    warmup: int = PROTOCOL_WARMUP,
    iters: int = PROTOCOL_ITERS,
    optimized: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure one translator under the fixed latency protocol; JSON-safe result.

    Synthetic inputs (default) use ``B=O=1, K=N`` for each ``valid_records``
    value. Pass ``states`` (for example the source state of every dev case) to
    measure real ``[B,O,K]`` and validity; ``summary`` then holds the median
    over cases used by the selection rule. Learned translators must already be
    on ``device`` (default: their parameter device, else CUDA when available).
    """

    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be >= 0 and iters >= 1")
    module_device = _module_device(translator)
    target = _normalized_device(device) if device is not None else (
        module_device if module_device is not None else _normalized_device(None)
    )
    if module_device is not None and module_device != target:
        raise ValueError(
            f"translator parameters are on {module_device}; move it with "
            f".to({str(target)!r}) before benchmarking on {target}"
        )
    if states is None:
        records = [int(value) if not isinstance(value, bool) else value for value in valid_records]
        state_list = [synthetic_handoff_state(value, _source_spec(translator), seed=seed) for value in records]
        entry_labels = [f"N={value}" for value in records]
        synthetic_records: Sequence[int] | None = records
    else:
        state_list = list(states)
        if not state_list:
            raise ValueError("states must not be empty")
        entry_labels = list(labels) if labels is not None else [f"state[{i}]" for i in range(len(state_list))]
        if len(entry_labels) != len(state_list):
            raise ValueError("labels must match states")
        synthetic_records = None
    was_training = translator.training if isinstance(translator, nn.Module) else None
    if isinstance(translator, nn.Module):
        translator.eval()
    macs = _analytic_costs(translator)
    try:
        with protocol_backend_flags() as settings, torch.inference_mode():
            flops_per_frame = _flops_per_frame(translator, target)
            entries = [
                _benchmark_entry(
                    translator,
                    state,
                    label=label,
                    device=target,
                    warmup=warmup,
                    iters=iters,
                    optimized=optimized,
                    macs=macs,
                )
                for state, label in zip(state_list, entry_labels, strict=True)
            ]
    finally:
        if was_training is not None:
            translator.train(was_training)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": _protocol_block(
            device=target, warmup=warmup, iters=iters, synthetic_records=synthetic_records
        ),
        "settings": settings,
        "environment": benchmark_environment(target),
        "translator": _describe(translator),
        "params": _parameters(translator),
        "analytic_macs_per_frame": macs[0],
        "pointer_macs_per_record": macs[1],
        "pytorch_flops_per_frame": flops_per_frame,
        "cost_units": dict(COST_UNITS),
        "inputs": {
            "source": "synthetic" if states is None else "states",
            "valid_records": [entry["valid_records"] for entry in entries],
            "seed": seed,
        },
        "entries": entries,
        "summary": {
            "latency_e2e_eager_median_ms": _median_over_entries(entries, "latency_e2e_eager"),
            "latency_e2e_optimized_median_ms": _median_over_entries(entries, "latency_e2e_optimized"),
            "definition": "median over entries of the per-entry median latency",
        },
    }
    if isinstance(translator, MomentMatchedCopyTranslator):
        report["statistics_count"] = translator.statistics_count()
        report["statistics_bytes"] = translator.statistics_bytes()
    return report


def write_benchmark_json(report: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Preset cost table
# --------------------------------------------------------------------------


def _placeholder_moment_translator(spec: StateSpec) -> MomentMatchedCopyTranslator:
    """Cost-only Moment Match (mu=0, sigma=1); cost does not depend on values."""

    statistics = {}
    for component, width in (("spatial", spec.feature_channels), ("pointer", spec.pointer_dim)):
        for side in ("source", "target"):
            statistics[f"{component}_{side}_mean"] = torch.zeros(2, width)
            statistics[f"{component}_{side}_std"] = torch.ones(2, width)
    translator = MomentMatchedCopyTranslator(
        spec, spec, statistics, fit_summary={"placeholder": "cost only"}
    )
    translator.preset = "moment_match"
    return translator


def _cost_translator(name: str, spec: StateSpec, device: torch.device, seed: int) -> Any:
    if PRESETS[name].kind == "moment_match":
        return _placeholder_moment_translator(spec)
    torch.manual_seed(seed)
    translator = build_translator(name, spec, spec)
    if isinstance(translator, nn.Module):
        translator = translator.to(device).eval()
    return translator


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    return value / reference


def _cost_rows(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference = results.get("residual_mlp")

    def by_n(report: Mapping[str, Any], key: str, field: str = "median_ms") -> dict[str, Any]:
        values = {}
        for entry in report["entries"]:
            measurement = entry.get(key)
            values[str(entry["valid_records"])] = None if measurement is None else measurement[field]
        return values

    rows = []
    for name, report in results.items():
        preset = PRESETS[name]
        eager = by_n(report, "latency_e2e_eager")
        optimized = by_n(report, "latency_e2e_optimized")
        row = {
            "preset": name,
            "group": preset.group,
            "kind": preset.kind,
            "description": preset.description,
            "spatial_params": report["params"]["spatial"],
            "pointer_params": report["params"]["pointer"],
            "total_params": report["params"]["total"],
            "statistics_count": report.get("statistics_count"),
            "analytic_macs_per_frame": report["analytic_macs_per_frame"],
            "pytorch_flops_per_frame": report["pytorch_flops_per_frame"],
            "latency_e2e_eager_median_ms": eager,
            "latency_e2e_eager_p90_ms": by_n(report, "latency_e2e_eager", "p90_ms"),
            "latency_e2e_optimized_median_ms": optimized,
            "peak_memory_delta_bytes": {
                str(entry["valid_records"]): (
                    None if entry["peak_memory"] is None else entry["peak_memory"]["peak_delta_bytes"]
                )
                for entry in report["entries"]
            },
            "handoff_bytes": {
                str(entry["valid_records"]): entry["handoff_bytes"] for entry in report["entries"]
            },
        }
        if reference is not None:
            reference_eager = by_n(reference, "latency_e2e_eager")
            reference_optimized = by_n(reference, "latency_e2e_optimized")
            row["aux_relative_latency_vs_residual_mlp"] = {
                "eager": {n: _ratio(eager[n], reference_eager.get(n)) for n in eager},
                "optimized": {n: _ratio(optimized[n], reference_optimized.get(n)) for n in optimized},
            }
        else:
            row["aux_relative_latency_vs_residual_mlp"] = None
        rows.append(row)
    return rows


def benchmark_presets(
    names: Sequence[str] | None = None,
    *,
    device: str | torch.device | None = None,
    valid_records: Sequence[int] = PROTOCOL_VALID_RECORDS,
    warmup: int = PROTOCOL_WARMUP,
    iters: int = PROTOCOL_ITERS,
    optimized: bool = True,
    seed: int = 0,
    spec: StateSpec = SAM21_MEMORY_SPEC,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cost table for registered presets (default: all, including Direct and Moment Match)."""

    selected = list(PRESETS) if names is None else list(dict.fromkeys(names))
    unknown = sorted(set(selected) - set(PRESETS))
    if unknown:
        raise ValueError(f"unknown presets: {unknown}")
    target = _normalized_device(device)
    results: dict[str, dict[str, Any]] = {}
    shared: dict[str, Any] = {}
    for position, name in enumerate(selected, start=1):
        if progress is not None:
            progress(f"[{position}/{len(selected)}] {name}")
        translator = _cost_translator(name, spec, target, seed)
        report = benchmark_translator(
            translator,
            valid_records,
            device=target,
            warmup=warmup,
            iters=iters,
            optimized=optimized,
            seed=seed,
        )
        for key in ("protocol", "settings", "environment"):
            shared.setdefault(key, report[key])
            report.pop(key)
        report.pop("cost_units")
        results[name] = report
        del translator
        if target.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema_version": COST_TABLE_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **shared,
        "cost_units": dict(COST_UNITS),
        "valid_records": [int(value) for value in valid_records],
        "notes": [
            "Primary latency is latency_e2e_eager; optimized (CUDA graph) is reported separately.",
            "aux_relative_latency_vs_residual_mlp is an auxiliary indicator, not a selection budget.",
            "moment_match uses placeholder statistics (cost does not depend on their values).",
            "Selection uses dev-case inputs measured in the fixed benchmark environment.",
        ],
        "rows": _cost_rows(results),
        "presets": results,
    }


def _format_count(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _format_millions(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e6:.1f}M"


def _format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_cost_table(table: Mapping[str, Any]) -> str:
    """Markdown cost tables from a ``benchmark_presets`` result."""

    records = [str(value) for value in table["valid_records"]]
    largest = records[-1]
    environment = table.get("environment", {})
    protocol = table.get("protocol", {})
    device = environment.get("gpu") or environment.get("cpu") or environment.get("device")
    lines = [
        "# Translator cost table",
        "",
        f"- Device: {device} ({environment.get('device')}); driver {environment.get('driver')}; "
        f"torch {environment.get('torch')}; CUDA {environment.get('cuda_runtime')}",
        f"- Protocol: {protocol.get('id')}; warmup {protocol.get('warmup')}, iters {protocol.get('iters')}; "
        f"conforming: {protocol.get('conforming')}"
        + (f" ({'; '.join(protocol.get('deviations', []))})" if protocol.get("deviations") else ""),
        f"- TF32: {table.get('settings', {}).get('tf32')}",
        f"- Inputs: synthetic B=O=1, K=N valid records, spatial bf16, pointer fp32; seed-fixed",
        "- Latency: end-to-end handoff path (valid gather -> dtype conversion -> translator -> "
        "output cast -> scatter), median ms. Optimized = CUDA graph replay (not used for selection).",
        "- MACs are analytic; FLOPs are FlopCounterMode (matmul/conv only). Different units.",
        "",
        "## Size and analytic cost",
        "",
        f"| preset | group | spatial params | pointer params | MACs/frame | FLOPs/frame "
        f"| peak mem Δ @N={largest} | handoff bytes @N={largest} |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table["rows"]:
        spatial = (
            f"stats {row['statistics_count']:,}"
            if row.get("statistics_count")
            else _format_count(row["spatial_params"])
        )
        peak = row["peak_memory_delta_bytes"].get(largest)
        lines.append(
            f"| {row['preset']} | {row['group']} | {spatial} | {_format_count(row['pointer_params'])} "
            f"| {_format_millions(row['analytic_macs_per_frame'])} "
            f"| {_format_millions(row['pytorch_flops_per_frame'])} "
            f"| {'-' if peak is None else f'{peak / 2**20:.1f} MiB'} "
            f"| {_format_count(row['handoff_bytes'].get(largest))} |"
        )
    header = " | ".join(f"N={value}" for value in records)
    divider = "|".join(["---"] + ["---:"] * len(records))
    lines.extend(
        [
            "",
            "## End-to-end eager latency (primary), median ms [p90]",
            "",
            f"| preset | {header} |",
            f"|{divider}|",
        ]
    )
    for row in table["rows"]:
        cells = " | ".join(
            f"{_format_ms(row['latency_e2e_eager_median_ms'][n])} [{_format_ms(row['latency_e2e_eager_p90_ms'][n])}]"
            for n in records
        )
        lines.append(f"| {row['preset']} | {cells} |")
    lines.extend(
        [
            "",
            "## Optimized latency (CUDA graph), median ms",
            "",
            f"| preset | {header} |",
            f"|{divider}|",
        ]
    )
    for row in table["rows"]:
        cells = " | ".join(_format_ms(row["latency_e2e_optimized_median_ms"][n]) for n in records)
        lines.append(f"| {row['preset']} | {cells} |")
    if any(row["aux_relative_latency_vs_residual_mlp"] for row in table["rows"]):
        lines.extend(
            [
                "",
                "## Auxiliary: latency relative to Residual MLP (eager / optimized)",
                "",
                f"| preset | {header} |",
                f"|{divider}|",
            ]
        )
        for row in table["rows"]:
            ratios = row["aux_relative_latency_vs_residual_mlp"]
            cells = " | ".join(
                f"{_format_ratio(ratios['eager'][n])} / {_format_ratio(ratios['optimized'][n])}"
                for n in records
            )
            lines.append(f"| {row['preset']} | {cells} |")
    return "\n".join(lines) + "\n"


def _parse_presets(value: str) -> list[str] | None:
    if value == "all":
        return None
    if value == "learned":
        return [name for name, preset in PRESETS.items() if preset.learned]
    return [name.strip() for name in value.split(",") if name.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vos_memory_inspector.translator_benchmark",
        description="Translator cost table under the fixed latency protocol.",
    )
    parser.add_argument("--presets", default="all", help="'all', 'learned', or comma-separated names")
    parser.add_argument("--device", default=None, help="default: CUDA when available")
    parser.add_argument("--valid-records", type=int, nargs="+", default=list(PROTOCOL_VALID_RECORDS))
    parser.add_argument("--warmup", type=int, default=PROTOCOL_WARMUP)
    parser.add_argument("--iters", type=int, default=PROTOCOL_ITERS)
    parser.add_argument("--no-optimized", action="store_true", help="skip CUDA graph timing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write the JSON result here")
    parser.add_argument("--markdown", type=Path, help="write the Markdown tables here")
    args = parser.parse_args(argv)
    try:
        table = benchmark_presets(
            _parse_presets(args.presets),
            device=args.device,
            valid_records=args.valid_records,
            warmup=args.warmup,
            iters=args.iters,
            optimized=not args.no_optimized,
            seed=args.seed,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except ValueError as exc:
        parser.error(str(exc))
    text = format_cost_table(table)
    if args.output is not None:
        write_benchmark_json(table, args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    from vos_memory_inspector.translator_benchmark import main as _main

    raise SystemExit(_main())
