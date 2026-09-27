"""Task 7: translator cost benchmark under the fixed latency protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.transformer_translator import (
    SAM21_MEMORY_SPEC,
    SpatialTransformerConfig,
    TransformerStateTranslator,
    build_translator,
)
from vos_memory_inspector.translator_benchmark import (
    PATH_STEPS,
    PROTOCOL_ID,
    _placeholder_moment_translator,
    backend_precision_snapshot,
    benchmark_translator,
    main,
    protocol_backend_flags,
    restore_backend_precision,
)
from vos_memory_inspector.translators import ComponentAblationTranslator, DirectCopyTranslator


SMALL = SpatialTransformerConfig(
    channels=8, height=16, width=16, d_model=16, num_heads=2, pointer_dim=12, pointer_hidden_dim=24
)
SMALL_SPEC = StateSpec(8, 16, 16, 12)
FAST = {"warmup": 0, "iters": 2}


def _small() -> TransformerStateTranslator:
    torch.manual_seed(0)
    return TransformerStateTranslator(SMALL_SPEC, SMALL_SPEC, config=SMALL)


def test_cpu_benchmark_reports_every_field() -> None:
    translator = _small()
    original = backend_precision_snapshot()
    caller = dict(original)
    if caller:  # torch>=2.9: a caller setting that must survive the benchmark
        caller["cuda.matmul.fp32_precision"] = "tf32"
        restore_backend_precision(caller)
    try:
        report = benchmark_translator(translator, (1, 2, 4), device="cpu", **FAST)
        assert backend_precision_snapshot() == caller
    finally:
        restore_backend_precision(original)
    json.dumps(report)  # JSON-safe
    assert report["schema_version"] == "cmmt.translator_benchmark.v1"
    protocol = report["protocol"]
    assert protocol["id"] == PROTOCOL_ID and protocol["path"] == list(PATH_STEPS)
    assert protocol["primary"] == "latency_e2e_eager"
    assert protocol["conforming"] is False
    assert any("device=cpu" in item for item in protocol["deviations"])
    assert report["settings"]["tf32"]["matmul_allow_tf32"] is False
    assert report["settings"]["tf32"]["cudnn_allow_tf32"] is True
    assert report["settings"]["inference_mode"] is True
    assert report["environment"]["torch"] == torch.__version__
    assert report["params"] == translator.parameter_breakdown()
    assert report["analytic_macs_per_frame"] == translator.macs_per_frame()
    assert report["pytorch_flops_per_frame"] > 0  # separate field, different unit
    for entry in report["entries"]:
        eager = entry["latency_e2e_eager"]
        assert eager["samples"] == 2 and eager["median_ms"] > 0 and eager["p90_ms"] >= eager["min_ms"]
        assert entry["latency_e2e_optimized"] is None
        assert "CUDA" in entry["latency_e2e_optimized_reason"]
        assert entry["peak_memory"] is None and "CUDA" in entry["peak_memory_reason"]
        assert entry["output"] == {
            "spatial_shape": [1, 1, entry["valid_records"], 8, 16, 16],
            "spatial_dtype": "bfloat16",
            "pointer_shape": [1, 1, entry["valid_records"], 12],
            "pointer_dtype": "float32",
            "matches_source_layout": True,
        }
        assert entry["handoff_bytes"] == entry["direct_copy_handoff_bytes"]
    assert report["summary"]["latency_e2e_eager_median_ms"] > 0
    assert report["summary"]["latency_e2e_optimized_median_ms"] is None


def test_protocol_flags_apply_and_restore_from_a_mixed_caller_state() -> None:
    original = backend_precision_snapshot()
    if not original:
        pytest.skip("per-backend fp32_precision API is not available")
    # set_float32_matmul_precision("high") followed by a legacy flag leaves torch
    # in a mixed state where the legacy getters raise; the benchmark must cope.
    mixed = {**original, "cuda.matmul.fp32_precision": "tf32", "mkldnn.matmul.fp32_precision": "tf32",
             "cudnn.conv.fp32_precision": "ieee"}
    restore_backend_precision(mixed)
    try:
        with protocol_backend_flags() as settings:
            assert settings["tf32"]["matmul_allow_tf32"] is False
            assert settings["tf32"]["cudnn_allow_tf32"] is True
            assert torch.backends.cuda.matmul.fp32_precision == "ieee"
            assert torch.backends.cudnn.conv.fp32_precision == "tf32"
        assert backend_precision_snapshot() == mixed
    finally:
        restore_backend_precision(original)
    assert backend_precision_snapshot() == original


def test_macs_and_flops_grow_linearly_with_valid_records() -> None:
    translator = _small()
    report = benchmark_translator(translator, (1, 3, 7), device="cpu", **FAST)
    per_record = translator.macs_per_frame() + translator.pointer_macs_per_record()
    first = report["entries"][0]
    for entry in report["entries"]:
        count = entry["valid_records"]
        assert entry["analytic_macs"] == count * per_record
        assert entry["analytic_macs_spatial"] == count * translator.macs_per_frame()
        assert entry["pytorch_flops"] == count * first["pytorch_flops"]


def test_benchmark_accepts_dev_case_states(make_state) -> None:
    translator = _small()
    validity = torch.tensor([[[True, True, False], [True, False, False]]])
    states = [
        make_state(torch.randn(1, 2, 3, 8, 16, 16).to(torch.bfloat16), torch.randn(1, 2, 3, 12), validity=validity),
        make_state(torch.randn(1, 1, 5, 8, 16, 16).to(torch.bfloat16), torch.randn(1, 1, 5, 12)),
    ]
    report = benchmark_translator(translator, device="cpu", states=states, labels=["walking", "india"], **FAST)
    assert report["inputs"] == {"source": "states", "valid_records": [3, 5], "seed": 0}
    assert [entry["label"] for entry in report["entries"]] == ["walking", "india"]
    assert [entry["records"] for entry in report["entries"]] == [[1, 2, 3], [1, 1, 5]]
    assert [entry["total_records"] for entry in report["entries"]] == [6, 5]
    medians = [entry["latency_e2e_eager"]["median_ms"] for entry in report["entries"]]
    assert report["summary"]["latency_e2e_eager_median_ms"] == pytest.approx(sum(medians) / 2)
    # Synthetic-only deviation is not reported for dev-case inputs.
    assert not any("valid_records" in item for item in report["protocol"]["deviations"])


def test_benchmark_supports_direct_copy_and_moment_match() -> None:
    direct = benchmark_translator(DirectCopyTranslator(SMALL_SPEC), (2,), device="cpu", **FAST)
    assert direct["params"] == {"spatial": 0, "pointer": 0, "total": 0}
    assert direct["analytic_macs_per_frame"] == 0 and direct["pytorch_flops_per_frame"] == 0
    assert direct["entries"][0]["output"]["matches_source_layout"] is True

    moment = benchmark_translator(_placeholder_moment_translator(SMALL_SPEC), (2,), device="cpu", **FAST)
    assert moment["analytic_macs_per_frame"] == 8 * 16 * 16
    assert moment["statistics_count"] == 2 * (8 + 12) * 4
    assert moment["entries"][0]["handoff_bytes"] == moment["entries"][0]["direct_copy_handoff_bytes"]


def test_benchmark_rejects_bad_inputs_and_restores_training_mode() -> None:
    translator = _small().train()
    benchmark_translator(translator, (1,), device="cpu", **FAST)
    assert translator.training is True
    with pytest.raises(ValueError, match="iters"):
        benchmark_translator(translator, (1,), device="cpu", warmup=0, iters=0)
    with pytest.raises(ValueError, match="valid_records"):
        benchmark_translator(translator, (0,), device="cpu", **FAST)
    with pytest.raises(TypeError, match="supports"):
        benchmark_translator(
            ComponentAblationTranslator(DirectCopyTranslator(SMALL_SPEC)), (1,), device="cpu", **FAST
        )
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="move it"):
            benchmark_translator(translator, (1,), device="cuda", **FAST)


def test_cost_table_cli_writes_json_and_markdown(tmp_path: Path, capsys) -> None:
    arguments = [
        "--presets", "residual_mlp,linear_local,direct",
        "--device", "cpu",
        "--valid-records", "1", "2",
        "--warmup", "0",
        "--iters", "1",
        "--no-optimized",
        "--output", str(tmp_path / "cost.json"),
        "--markdown", str(tmp_path / "cost.md"),
    ]
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    table = json.loads((tmp_path / "cost.json").read_text(encoding="utf-8"))
    assert table["schema_version"] == "cmmt.translator_cost_table.v1"
    assert table["valid_records"] == [1, 2]
    assert [row["preset"] for row in table["rows"]] == ["residual_mlp", "linear_local", "direct"]
    rows = {row["preset"]: row for row in table["rows"]}
    assert rows["residual_mlp"]["aux_relative_latency_vs_residual_mlp"]["eager"] == {"1": 1.0, "2": 1.0}
    assert rows["residual_mlp"]["spatial_params"] == 16_576
    assert rows["linear_local"]["analytic_macs_per_frame"] == 64 * 64 * 64 * 64
    assert rows["direct"]["latency_e2e_optimized_median_ms"] == {"1": None, "2": None}
    assert all("environment" not in report for report in table["presets"].values())
    markdown = (tmp_path / "cost.md").read_text(encoding="utf-8")
    assert markdown == printed
    assert "## End-to-end eager latency (primary), median ms [p90]" in markdown
    assert "## Auxiliary: latency relative to Residual MLP (eager / optimized)" in markdown
    assert "| residual_mlp | ladder | 16,576 | 262,912 | 67.1M |" in markdown


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_benchmark_measures_optimized_path_and_memory() -> None:
    torch.manual_seed(0)
    translator = build_translator("base", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).cuda().eval()
    report = benchmark_translator(translator, (1, 3), device="cuda", warmup=2, iters=5)
    assert report["environment"]["gpu"]
    for entry in report["entries"]:
        optimized = entry["latency_e2e_optimized"]
        assert optimized["method"] == "cuda_graph" and optimized["median_ms"] > 0
        assert optimized["max_abs_diff_vs_eager"] == 0.0
        assert entry["peak_memory"]["peak_delta_bytes"] > 0
        assert entry["output"]["matches_source_layout"] is True
    moment = benchmark_translator(
        _placeholder_moment_translator(SAM21_MEMORY_SPEC), (3,), device="cuda", warmup=1, iters=3
    )
    assert moment["entries"][0]["latency_e2e_optimized"]["max_abs_diff_vs_eager"] == 0.0
