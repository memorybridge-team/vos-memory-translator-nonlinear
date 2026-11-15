from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from vos_memory_inspector.transformer_translator import (
    SAM21_MEMORY_SPEC,
    build_translator,
    main,
    pytorch_flops,
)

MODEL_PRESETS = ["residual_mlp", "transformer", "residual_mlp_matched"]


@pytest.mark.parametrize("records", [1, 3, 7])
def test_attention_uses_one_token_grid_per_record_for_any_record_count(records: int, make_state) -> None:
    torch.manual_seed(0)
    translator = build_translator("transformer", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    attention_inputs: list[tuple[int, ...]] = []
    hooks = [
        block.attn.register_forward_pre_hook(
            lambda _m, inputs: attention_inputs.append(tuple(inputs[0].shape))
        )
        for block in translator.spatial.blocks
    ]
    state = make_state(
        torch.randn(1, 1, records, 64, 64, 64).to(torch.bfloat16),
        torch.randn(1, 1, records, 256),
    )
    with torch.no_grad():
        translator.translate(state)
    for hook in hooks:
        hook.remove()
    # Records stay on the batch axis; every attention sees one 64x64 grid of tokens.
    assert attention_inputs == [(records, 4096, 64)]


@pytest.mark.parametrize("preset", MODEL_PRESETS)
def test_analytic_macs_match_the_flop_counter(preset: str) -> None:
    """The analytic count covers linear layers, the reduction convolution and the
    two attention products; the flop counter sees the same operations."""

    translator = build_translator(preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    module = translator.spatial
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]), counter:
        module(torch.randn(1, 64, 64, 64))
    counts = {str(op): value for op, value in counter.get_flop_counts()["Global"].items()}
    flops = sum(
        counts.get(op, 0)
        for op in ("aten.addmm", "aten.mm", "aten.bmm", "aten.convolution", "aten._scaled_dot_product_attention_math")
    )
    assert flops / 2 == pytest.approx(translator.macs_per_frame(), rel=0.01)
    assert module.macs_breakdown()["total"] == translator.macs_per_frame()


@pytest.mark.parametrize("preset", ["residual_mlp", "transformer"])
def test_pytorch_flops_of_a_frame_is_about_twice_the_analytic_macs(preset: str) -> None:
    translator = build_translator(preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    flops = pytorch_flops(translator.spatial, torch.zeros(1, 64, 64, 64))
    assert flops / 2 == pytest.approx(translator.macs_per_frame(), rel=0.01)


def test_cli_prints_dummy_forward_summary_for_the_transformer(capsys) -> None:
    arguments = ["--preset", "transformer", "--batch", "1", "--objects", "2", "--records", "3", "--invalid", "1"]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    translator = build_translator("transformer", SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC)
    summary = translator.architecture_summary()
    assert f"preset             : transformer ({summary['description']})" in output
    assert "input spatial      : (1, 2, 3, 64, 64, 64) bfloat16  valid 5/6 -> frames (5, 64, 64, 64)" in output
    assert "output spatial     : (1, 2, 3, 64, 64, 64) bfloat16" in output
    assert "object pointer     : (1, 2, 3, 256) -> (1, 2, 3, 256)" in output
    params = summary["parameters"]
    assert (
        f"trainable params   : spatial {params['spatial']:,} | pointer {params['pointer']:,} "
        f"| total {params['total']:,}"
    ) in output
    assert f"~{summary['macs_per_frame'] / 1e6:.1f}M" in output
    assert "identity at init   : max|T(M)-M| = 0 | max|T(p)-p| = 0" in output
    assert "no cross-frame attention" in output
    assert "PyTorch FLOPs/frame" in output


def test_cli_reports_skipped_forward_without_valid_records(capsys) -> None:
    assert main(["--preset", "residual_mlp", "--records", "2", "--objects", "1", "--invalid", "2"]) == 0
    output = capsys.readouterr().out
    assert "valid 0/2 -> frames none (translator not called)" in output
    assert "attention scope    : none (position-wise feed-forward)" in output


def test_cli_offers_only_the_two_model_kinds(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--preset", "linear"])
    capsys.readouterr()


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vos_memory_inspector.transformer_translator", "--preset", "residual_mlp"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "preset             : residual_mlp (" in result.stdout
