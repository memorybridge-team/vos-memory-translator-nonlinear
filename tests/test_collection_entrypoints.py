"""Collection training and evaluation scripts for the two stitched translators."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from vos_memory_inspector import cli
from vos_memory_inspector.models import (
    ResidualMLPStateTranslator,
    TransformerStateTranslator,
    TranslatorConfig,
    build_translator,
)
from vos_memory_inspector.paired_state_cache import write_paired_state_cache
from vos_memory_inspector.translator_artifact import load_learned_translator
from vos_memory_inspector.translator_training import evaluate_state_pairs

MODEL_CLASSES = {"residual_mlp": ResidualMLPStateTranslator, "transformer": TransformerStateTranslator}


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(make_state):
    source = make_state(torch.randn(1, 2, 2, 4, 8, 8), torch.randn(1, 2, 2, 6))
    target = source.with_continuous(
        spatial_memory=source.spatial_memory * 0.7 + 0.2,
        object_pointer=source.object_pointer * 0.8 - 0.1,
        presence_logits=source.presence_logits,
    )
    return source, target


@pytest.mark.parametrize("name", ["transformer", "residual_mlp"])
def test_collection_entrypoint_trains_standard_model(tmp_path, make_state, monkeypatch, capsys, name):
    source, target = _pair(make_state)
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
    assert type(restored) is MODEL_CLASSES[name]
    assert restored.config == TranslatorConfig(attention=(name == "transformer"))
    report = json.loads((tmp_path / "out/paired_report.json").read_text())
    assert report["training"]["spatial_samples_per_pair"] is None
    assert report["training"]["translator_config"] is None
    assert "hidden_dim" not in report["training"] and "pointer_hidden_dim" not in report["training"]
    assert report["training"]["model_selection"][name]["selected_validation_loss"] == evaluate_state_pairs(restored, [(source, target)])["state_loss"]
    monkeypatch.setattr(sys, "argv", argv + ["--spatial-samples-per-pair", "10"])
    if name == "transformer":
        with pytest.raises(SystemExit, match="2"):
            module.main()
        with pytest.raises(SystemExit, match="2"):
            cli.paired_experiment_main(["--output-dir", str(tmp_path), "--translator", name,
                                       "--spatial-samples-per-pair", "10"])
    for flag in ("--hidden-dim", "--pointer-hidden-dim"):
        monkeypatch.setattr(sys, "argv", argv + [flag, "8"])
        with pytest.raises(SystemExit, match="2"):
            module.main()
    capsys.readouterr()


def test_collection_evaluation_loads_correct_model_and_separates_cache(tmp_path, make_state, monkeypatch, capsys):
    source, _ = _pair(make_state)
    path = tmp_path / "models.pt"
    torch.save({name: build_translator(name, source.spec, source.spec).to_payload()
                for name in ("transformer", "residual_mlp")}, path)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"cases": [dict(sequence="video", object_id=0, switch_frame=1, paired_split="validation")]}))
    module = _script("evaluate_nonlinear_collection")
    monkeypatch.setattr(module, "_stage_sequence", lambda *a: tmp_path)
    calls = []

    def handoff(**kw):
        calls.append(kw)
        return {"resources_candidate_only": {"wall_time_seconds": 0.1}}

    monkeypatch.setattr(module, "run_cached_translator_handoff", handoff)
    for name in ("transformer", "residual_mlp", "residual_mlp"):
        monkeypatch.setattr(sys, "argv", [
            "evaluate", "--selection-manifest", str(manifest), "--case-cache-root", str(tmp_path),
            "--dataset-root", str(tmp_path), "--sam2-repo", str(tmp_path), "--target-config", "unused",
            "--target-checkpoint", "unused", "--target-model-id", "unused", "--translator", name,
            "--translator-artifact", str(path), "--output-root", str(tmp_path / "evaluation"), "--device", "cpu",
        ])
        module.main()
    assert len(calls) == 2  # The same file is evaluated again when the model name changes.
    assert [type(call["translator"]) for call in calls] == [TransformerStateTranslator, ResidualMLPStateTranslator]
    assert [call["translator"].config.attention for call in calls] == [True, False]
    capsys.readouterr()
