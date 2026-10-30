"""Independent local correction, context learning and training/export contracts."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from vos_memory_inspector import cli
from vos_memory_inspector.paired_experiment import run_paired_experiment
from vos_memory_inspector.paired_state_cache import write_paired_state_cache
from vos_memory_inspector.translator_artifact import load_learned_translator
from vos_memory_inspector.translator_training import evaluate_state_pairs
from vos_memory_inspector.transformer_translator import (
    SpatialTransformerConfig, TransformerStateTranslator, build_translator,
)
from vos_memory_inspector.translators import ResidualMLPStateTranslator


def _config():
    return SpatialTransformerConfig(
        channels=4, height=8, width=8, pointer_dim=6, d_model=8,
        num_heads=2, num_layers=1, patch_size=2, fusion_hidden_dim=10,
        pointer_hidden_dim=12, local_fusion="additive_mlp",
    )


def _pair(make_state):
    source = make_state(torch.randn(1, 2, 2, 4, 8, 8), torch.randn(1, 2, 2, 6))
    target = source.with_continuous(
        spatial_memory=source.spatial_memory * 0.7 + 0.2,
        object_pointer=source.object_pointer * 0.8 - 0.1,
        presence_logits=source.presence_logits,
    )
    return source, target


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_zero_matches_mlp_and_context_does_not_mix_records(make_state):
    torch.manual_seed(7)
    source, _ = _pair(make_state)
    model = build_translator("mlp_context", source.spec, source.spec, config=_config()).eval()
    mlp = ResidualMLPStateTranslator(source.spec, source.spec, hidden_dim=10, pointer_hidden_dim=12)
    with torch.no_grad():
        # 로컬 MLP를 학습된 상태처럼 만든다. alpha=0이어도 출력에 반영되어야 한다.
        mlp.feature[-1].weight.normal_(std=0.1)
        for conv, linear in ((model.spatial.local_in, mlp.feature[0]),
                             (model.spatial.local_out, mlp.feature[-1])):
            conv.weight.copy_(linear.weight[:, :, None, None])
            conv.bias.copy_(linear.bias)
        got, _ = model.translate_tensors(source.spatial_memory, source.object_pointer)
        expected, _ = mlp.translate_tensors(source.spatial_memory, source.object_pointer)
        torch.testing.assert_close(got, expected)
        assert not torch.equal(got, source.spatial_memory)
        model.spatial.alpha.fill_(0.2)
        context_result, _ = model.translate_tensors(source.spatial_memory, source.object_pointer)
        assert not torch.allclose(context_result, got)
        changed = source.spatial_memory.clone()
        changed[0, 0, 0, :, 0, 0] += 5
        after, _ = model.translate_tensors(changed, source.object_pointer)
        # 문맥은 같은 기록 안의 다른 위치에 영향을 준다. 다른 기록에는 영향이 없다.
        assert not torch.allclose(after[0, 0, 0, :, 4:, 4:], context_result[0, 0, 0, :, 4:, 4:])
        torch.testing.assert_close(after[0, 0, 1], context_result[0, 0, 1])
        torch.testing.assert_close(after[0, 1], context_result[0, 1])


def test_identity_init_opens_both_learning_paths_without_nan_padding(make_state):
    torch.manual_seed(7)
    source, _ = _pair(make_state)
    source.validity[0, 0, 1] = False
    source.spatial_memory[~source.validity] = float("nan")
    source.object_pointer[~source.validity] = float("nan")
    source.spatial_memory = source.spatial_memory.bfloat16()
    model = build_translator("mlp_context", source.spec, source.spec, config=_config())
    with torch.no_grad():
        initial = model.translate(source)
    torch.testing.assert_close(initial.spatial_memory, source.spatial_memory, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(initial.object_pointer, source.object_pointer, equal_nan=True, rtol=0, atol=0)
    assert initial.spatial_memory.dtype == torch.bfloat16
    valid = source.validity
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad()
        spatial, pointer = model.translate_tensors(source.spatial_memory, source.object_pointer, valid)
        loss = F.mse_loss(spatial[valid], source.spatial_memory[valid].float() * 0.7 + 0.2)
        loss = loss + F.mse_loss(pointer[valid], source.object_pointer[valid] * 0.8 - 0.1)
        loss.backward()
        active = [model.spatial.local_out.weight, model.spatial.alpha, model.pointer.fc2.weight]
        if step == 1:
            active += [model.spatial.local_in.weight, model.spatial.context_proj.weight,
                       model.spatial.blocks[0].attn.qkv.weight, model.pointer.fc1.weight]
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in active)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()


def test_both_transformers_train_select_export_and_reload(tmp_path, make_state):
    torch.manual_seed(7)
    pair = _pair(make_state)
    validation = [_pair(make_state)]
    report = run_paired_experiment(
        [pair], None, tmp_path, validation_pairs=validation,
        translator_names=("transformer", "mlp_context"), transformer_config=_config(),
        hidden_dim=10, pointer_hidden_dim=12, epochs=5, learning_rate=0.01, device="cpu",
    )
    for name, fusion in (("transformer", "linear"), ("mlp_context", "additive_mlp")):
        restored = load_learned_translator(tmp_path / "paired_translators.pt", name)
        assert restored.config.local_fusion == fusion
        fit = report["training"]["model_selection"][name]
        metrics = evaluate_state_pairs(restored, validation)
        assert metrics["state_loss"] == fit["selected_validation_loss"]
        assert metrics["state_loss"] == min(row["state_loss"] for row in fit["validation_history"])
        assert metrics["state_loss"] == report["results"][name]["state_loss"]
        assert fit["train_loss_history"][-1] < fit["train_loss_history"][0]
    with pytest.raises(ValueError, match="whole frames"):
        run_paired_experiment([pair], None, tmp_path, validation_pairs=validation,
                              translator_names=("mlp_context",), spatial_samples_per_pair=4)
    with pytest.raises(ValueError, match="test_pairs cannot select"):
        run_paired_experiment([pair], [pair], tmp_path, translator_names=("transformer",))


@pytest.mark.parametrize("name", ["transformer", "mlp_context"])
def test_collection_entrypoint_trains_standard_transformer(tmp_path, make_state, monkeypatch, capsys, name):
    source = make_state(torch.randn(1, 1, 1, 64, 64, 64).bfloat16(), torch.randn(1, 1, 1, 256))
    target = source.with_continuous(
        spatial_memory=source.spatial_memory.float() + 0.1,
        object_pointer=source.object_pointer + 0.1, presence_logits=source.presence_logits,
    )
    cases = []
    for split in ("train", "validation"):
        cases.append(dict(sequence=split, object_id=0, switch_frame=0, paired_split=split))
        write_paired_state_cache(
            tmp_path / split / f"{split}_obj0_switch0.pt",
            source_canonical=source, target_canonical=target, metadata={},
        )
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(dict(schema_version="cmmt.paired_state_selection.v1", cases=cases)))
    module = _script("train_nonlinear_from_collection")
    argv = ["train", "--selection-manifest", str(manifest), "--cache-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"), "--translator", name, "--epochs", "2", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    restored = load_learned_translator(tmp_path / "out/paired_translators.pt", name)
    assert (restored.config.patch_size, restored.config.num_tokens, restored.config.num_layers,
            restored.config.num_heads, restored.config.pointer_hidden_dim) == (4, 256, 2, 4, 512)
    report = json.loads((tmp_path / "out/paired_report.json").read_text())
    assert report["training"]["spatial_samples_per_pair"] is None
    assert report["training"]["model_selection"][name]["selected_validation_loss"] == evaluate_state_pairs(restored, [(source, target)])["state_loss"]
    monkeypatch.setattr(sys, "argv", argv + ["--spatial-samples-per-pair", "10"])
    with pytest.raises(SystemExit, match="2"):
        module.main()
    with pytest.raises(SystemExit, match="2"):
        cli.paired_experiment_main(["--output-dir", str(tmp_path), "--translator", name,
                                   "--spatial-samples-per-pair", "10"])
    capsys.readouterr()


def test_collection_evaluation_loads_correct_model_and_separates_cache(tmp_path, make_state, monkeypatch, capsys):
    source, _ = _pair(make_state)
    path = tmp_path / "models.pt"
    torch.save({name: build_translator(
        "base" if name == "transformer" else name, source.spec, source.spec, config=_config(),
        local_fusion="linear" if name == "transformer" else "additive_mlp",
    ).to_payload() for name in ("transformer", "mlp_context")}, path)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"cases": [dict(sequence="video", object_id=0, switch_frame=1, paired_split="validation")]}))
    module = _script("evaluate_nonlinear_collection")
    monkeypatch.setattr(module, "_stage_sequence", lambda *a: tmp_path)
    calls = []
    def handoff(**kw):
        calls.append(kw)
        return {"resources_candidate_only": {"wall_time_seconds": 0.1}}
    monkeypatch.setattr(module, "run_cached_translator_handoff", handoff)
    for name in ("transformer", "mlp_context", "mlp_context"):
        monkeypatch.setattr(sys, "argv", [
            "evaluate", "--selection-manifest", str(manifest), "--case-cache-root", str(tmp_path),
            "--dataset-root", str(tmp_path), "--sam2-repo", str(tmp_path), "--target-config", "unused",
            "--target-checkpoint", "unused", "--target-model-id", "unused", "--translator", name,
            "--translator-artifact", str(path), "--output-root", str(tmp_path / "evaluation"), "--device", "cpu",
        ])
        module.main()
    assert len(calls) == 2  # 같은 파일이라도 모델 이름이 바뀌면 다시 평가한다.
    assert [call["translator"].config.local_fusion for call in calls] == ["linear", "additive_mlp"]
    assert all(isinstance(call["translator"], TransformerStateTranslator) for call in calls)
    capsys.readouterr()
