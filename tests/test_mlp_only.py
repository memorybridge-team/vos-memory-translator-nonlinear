"""MLP-only architecture, learning and collection CLI contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from vos_memory_inspector.paired_state_cache import write_paired_state_cache
from vos_memory_inspector.state_schema import StateSpec
from vos_memory_inspector.translator_training import fit_gradient_translator, state_mse_loss
from vos_memory_inspector.translators import ResidualMLPStateTranslator


def test_default_mlp_identity_and_sam_memory_contract(make_state):
    spec = StateSpec(64, 64, 64, 256)
    model = ResidualMLPStateTranslator(spec, spec)
    assert model.parameter_count() == 279_488
    assert [type(m) for m in model.feature] == [nn.Linear, nn.GELU, nn.Linear]
    assert [type(m) for m in model.pointer] == [nn.Linear, nn.GELU, nn.Linear]
    assert (model.hidden_dim, model.pointer_hidden_dim) == (128, 512)
    source = make_state(
        torch.randn(1, 1, 2, 64, 64, 64).bfloat16(),
        torch.randn(1, 1, 2, 256),
        validity=torch.tensor([[[True, False]]]),
    )
    with torch.no_grad():
        result = model.translate(source)
        spatial, pointer = model.translate_tensors(
            source.spatial_memory, source.object_pointer, source.validity
        )
    assert torch.equal(result.spatial_memory, source.spatial_memory)
    assert torch.equal(result.object_pointer, source.object_pointer)
    assert result.spatial_memory.dtype == torch.bfloat16
    assert spatial.dtype == pointer.dtype == torch.float32
    for name in ("frame_indices", "slot_order", "is_conditioning", "validity", "presence_logits"):
        assert torch.equal(getattr(result, name), getattr(source, name))
    assert result.object_ids == source.object_ids
    assert result.switch_frame == source.switch_frame


@pytest.mark.parametrize("samples", [None, 12])
def test_zero_initialized_mlp_learns_both_heads_and_reloads(tmp_path, make_state, samples):
    torch.manual_seed(7)
    valid = torch.tensor([[[True, True, False]]])
    spatial = torch.randn(1, 1, 3, 4, 3, 3)
    pointer = torch.randn(1, 1, 3, 5)
    spatial[~valid] = float("nan")
    pointer[~valid] = float("nan")
    source = make_state(spatial, pointer, validity=valid)
    target = source.with_continuous(
        spatial_memory=spatial * 0.5 + 0.2,
        object_pointer=pointer * 0.7 - 0.3,
        presence_logits=source.presence_logits,
    )
    model = ResidualMLPStateTranslator(source.spec, target.spec, hidden_dim=12, pointer_hidden_dim=16)
    first_weights = [model.feature[0].weight.detach().clone(), model.pointer[0].weight.detach().clone()]
    history = fit_gradient_translator(
        model, [(source, target)], epochs=35, learning_rate=0.02,
        device="cpu", spatial_samples_per_pair=samples,
    )
    assert history[-1] < history[0] * 0.2
    for head, initial in zip((model.feature, model.pointer), first_weights):
        assert not torch.equal(head[0].weight, initial)
        assert torch.count_nonzero(head[-1].weight) > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    path = tmp_path / "mlp.pt"
    torch.save(model.to_payload(), path)
    restored = ResidualMLPStateTranslator.from_payload(torch.load(path, weights_only=True))
    with torch.no_grad():
        expected, actual = model.translate(source), restored.translate(source)
    assert state_mse_loss(actual, target).item() < history[0] * 0.2
    torch.testing.assert_close(actual.spatial_memory, expected.spatial_memory, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(actual.object_pointer, expected.object_pointer, equal_nan=True, rtol=0, atol=0)


def test_mlp_positions_and_records_remain_independent_after_training(make_state):
    torch.manual_seed(7)
    source = make_state(torch.randn(1, 2, 2, 4, 3, 3), torch.randn(1, 2, 2, 5))
    model = ResidualMLPStateTranslator(source.spec, source.spec, hidden_dim=8, pointer_hidden_dim=9)
    with torch.no_grad():
        for head in (model.feature, model.pointer):
            head[-1].weight.normal_()
        before_s, before_p = model.translate_tensors(source.spatial_memory, source.object_pointer)
        spatial, pointer = source.spatial_memory.clone(), source.object_pointer.clone()
        spatial[0, 0, 0, :, 1, 1] += 4
        pointer[0, 0, 0] += 4
        after_s, after_p = model.translate_tensors(spatial, pointer)
    changed = (before_s != after_s).any(dim=3)
    assert changed.sum().item() == 1
    assert changed[0, 0, 0, 1, 1]
    assert (before_p != after_p).any(dim=-1).sum().item() == 1


@pytest.mark.parametrize("samples", [None, 2])
def test_training_rejects_empty_records(make_state, samples):
    source = make_state(
        torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3),
        validity=torch.zeros(1, 1, 1, dtype=torch.bool),
    )
    model = ResidualMLPStateTranslator(source.spec, source.spec)
    with pytest.raises(ValueError, match="no valid records"):
        fit_gradient_translator(model, [(source, source)], epochs=1, spatial_samples_per_pair=samples)


def test_whole_frame_training_uses_fp32_before_handoff_cast(make_state):
    source = make_state(torch.ones(1, 1, 1, 2, 2, 2).bfloat16(), torch.ones(1, 1, 1, 3))
    target = source.with_continuous(
        spatial_memory=source.spatial_memory.float() + 0.0001,
        object_pointer=source.object_pointer,
        presence_logits=source.presence_logits,
    )
    model = ResidualMLPStateTranslator(source.spec, source.spec, hidden_dim=4, pointer_hidden_dim=6)
    fit_gradient_translator(model, [(source, target)], epochs=1, learning_rate=0.0001, device="cpu")
    with torch.no_grad():
        spatial, _ = model.translate_tensors(source.spatial_memory, source.object_pointer)
        handoff, _ = model.translate_handoff_tensors(source.spatial_memory, source.object_pointer)
    assert spatial.dtype == torch.float32
    assert (spatial != source.spatial_memory.float()).any()
    assert handoff.dtype == torch.bfloat16


@pytest.mark.parametrize("sampling", ["default", "whole_frames", "sampled"])
def test_collection_cli_trains_and_exports_mlp(tmp_path, make_state, sampling):
    source = make_state(torch.randn(1, 1, 2, 4, 3, 3), torch.randn(1, 1, 2, 5))
    target = source.with_continuous(
        spatial_memory=source.spatial_memory + 0.2,
        object_pointer=source.object_pointer - 0.1,
        presence_logits=source.presence_logits,
    )
    cases = []
    for split in ("train", "validation"):
        case = dict(sequence=f"video_{split}", object_id=0, switch_frame=1, paired_split=split)
        cases.append(case)
        write_paired_state_cache(
            tmp_path / "cache" / split / f"video_{split}_obj0_switch1.pt",
            source_canonical=source, target_canonical=target, metadata={},
        )
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(dict(schema_version="cmmt.paired_state_selection.v1", cases=cases)))
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts/train_nonlinear_from_collection.py"),
        "--selection-manifest", str(manifest), "--cache-root", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "output"), "--epochs", "2", "--device", "cpu",
    ]
    if sampling == "whole_frames":
        command.append("--whole-frames")
    elif sampling == "sampled":
        command.extend(["--spatial-samples-per-pair", "5"])
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "output/paired_report.json").read_text())
    assert report["translator_names"] == ["residual_mlp"]
    assert report["training"]["pointer_hidden_dim"] == 512
    assert report["training"]["hidden_dim"] == 128
    assert report["training"]["spatial_samples_per_pair"] == (5 if sampling == "sampled" else None)
    assert len(report["training"]["mlp_loss_history"]) == 2
    saved = torch.load(tmp_path / "output/paired_translators.pt", weights_only=True)
    model = ResidualMLPStateTranslator.from_payload(saved["residual_mlp"])
    assert (model.hidden_dim, model.pointer_hidden_dim, model.output_dtype) == (128, 512, "source")
    assert report["actual_sam2_injection"]["ran"] is False
    assert report["evaluation_split"] == "validation"
    assert report["test_pairs"] == 0
    assert report["validation_pairs"] == 1
    selection = saved["training_selection"]["residual_mlp"]
    assert report["results"]["residual_mlp"]["state_loss"] == selection["selected_validation_loss"]
    assert selection["selected_validation_loss"] == min(row["state_loss"] for row in selection["validation_history"])
