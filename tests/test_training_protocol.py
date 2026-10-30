"""Regression checks for FP32 metrics, validation selection and CLI defaults."""

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from vos_memory_inspector import cli
from vos_memory_inspector.paired_experiment import run_paired_experiment
from vos_memory_inspector.translator_training import (
    evaluate_state_pairs, fit_validated_translator, state_mse_loss,
)
from vos_memory_inspector.translators import ResidualMLPStateTranslator


def _target(source, spatial_offset, pointer_offset):
    return source.with_continuous(
        spatial_memory=source.spatial_memory.float() + spatial_offset,
        object_pointer=source.object_pointer.float() + pointer_offset,
        presence_logits=source.presence_logits,
    )


def test_metrics_are_component_sum_not_mean(make_state):
    source = make_state(torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3))
    target = _target(source, 2, 3)
    model = ResidualMLPStateTranslator(source.spec, source.spec)
    result = evaluate_state_pairs(model, [(source, target)])
    assert result == {"spatial_mse": 4, "pointer_mse": 9, "state_loss": 13}
    assert state_mse_loss(model(source), target).item() == result["state_loss"]


def test_fp32_evaluation_does_not_round_through_bf16_handoff(make_state):
    source = make_state(torch.ones(1, 1, 1, 2, 2, 2).bfloat16(), torch.ones(1, 1, 1, 3))
    target = _target(source, 0.001, 0)
    model = ResidualMLPStateTranslator(source.spec, source.spec)
    with torch.no_grad():
        model.feature[-1].bias.fill_(0.001)
    model.train()
    metrics = evaluate_state_pairs(model, [(source, target)])
    assert model.training  # Validation must not leave the training loop in eval mode.
    assert metrics["state_loss"] == 0
    assert state_mse_loss(model(source), target).item() > 0


def test_best_epoch_weights_are_restored_and_survive_export(tmp_path, make_state):
    source = make_state(torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3))
    train = [(source, _target(source, 1, 1))]
    validation = [(source, _target(source, 0.01, 0.01))]
    model = ResidualMLPStateTranslator(source.spec, source.spec, hidden_dim=4, pointer_hidden_dim=4)
    # A bias-only learning problem has its validation optimum after the first update.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    fit = fit_validated_translator(
        model, train, validation, epochs=5, learning_rate=0.01, device="cpu"
    )
    assert fit["selected_epoch"] == 1
    assert fit["validation_history"][-1]["state_loss"] > fit["selected_validation_loss"]
    path = tmp_path / "selected.pt"
    torch.save(model.to_payload(), path)
    restored = ResidualMLPStateTranslator.from_payload(torch.load(path, weights_only=True))
    metrics = evaluate_state_pairs(restored, validation)
    assert metrics["state_loss"] == fit["selected_validation_loss"]
    assert metrics["state_loss"] == min(row["state_loss"] for row in fit["validation_history"])


def test_equal_validation_losses_keep_earlier_epoch(make_state):
    source = make_state(torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3))
    model = ResidualMLPStateTranslator(source.spec, source.spec)
    fit = fit_validated_translator(model, [(source, source)], [(source, source)], epochs=3, device="cpu")
    assert fit["selected_epoch"] == 1
    assert [row["state_loss"] for row in fit["validation_history"]] == [0, 0, 0]


def test_test_targets_cannot_change_selection_or_saved_weights(tmp_path, make_state):
    source = make_state(torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3))
    train = [(source, _target(source, 1, 1))]
    validation = [(source, _target(source, 0.01, 0.01))]
    outputs = []
    for offset in (0, 100):
        directory = tmp_path / str(offset)
        report = run_paired_experiment(
            train, [(source, _target(source, offset, offset))], directory,
            validation_pairs=validation, translator_names=("residual_mlp",),
            epochs=3, hidden_dim=4, pointer_hidden_dim=4, device="cpu",
        )
        artifact = torch.load(directory / "paired_translators.pt", weights_only=True)
        assert report["evaluation_split"] == "test"
        assert report["schema_version"] == "cmmt.paired_experiment.v2"
        assert "aggregate_mse" not in report["results"]["residual_mlp"]
        outputs.append((report, artifact))
    first, second = outputs
    assert first[1]["training_selection"] == second[1]["training_selection"]
    for key, tensor in first[1]["residual_mlp"]["state_dict"].items():
        assert torch.equal(tensor, second[1]["residual_mlp"]["state_dict"][key])
    assert first[0]["results"]["residual_mlp"]["state_loss"] < second[0]["results"]["residual_mlp"]["state_loss"]
    with pytest.raises(ValueError, match="test_pairs cannot select"):
        run_paired_experiment(train, validation, tmp_path / "missing", translator_names=("residual_mlp",))


def test_entry_points_share_whole_frame_lr_and_epoch_defaults(tmp_path, make_state, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts/train_nonlinear_from_collection.py"
    spec = importlib.util.spec_from_file_location("collection_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parser().parse_args([
        "--selection-manifest", "selection.json", "--cache-root", "cache", "--output-dir", "out"
    ])
    assert (args.epochs, args.learning_rate, args.spatial_samples_per_pair) == (8, 0.001, None)
    source = make_state(torch.zeros(1, 1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 3))
    monkeypatch.setattr(cli, "load_canonical_state", lambda path: source)
    calls = []
    monkeypatch.setattr(cli, "run_paired_experiment", lambda *a, **kw: calls.append(kw) or {})
    cli.paired_experiment_main([
        "--train-source", "train-source.pt", "--train-target", "train-target.pt",
        "--validation-source", "val-source.pt", "--validation-target", "val-target.pt",
        "--output-dir", str(tmp_path), "--translator", "residual_mlp",
    ])
    assert calls[0]["epochs"] == args.epochs
    assert calls[0]["learning_rate"] == args.learning_rate
    assert calls[0]["spatial_samples_per_pair"] is None
    signature = inspect.signature(run_paired_experiment).parameters
    assert signature["epochs"].default == args.epochs
    assert signature["learning_rate"].default == args.learning_rate
    capsys.readouterr()
