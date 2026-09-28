from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from vos_memory_inspector.training_collection import freeze_model, prepare_case, run_prefix
from vos_memory_inspector.training_data import (
    COLLECTION_SCHEMA, CONTRACT, PairedStateDataset, check_disjoint, load_pair,
    read_manifest, save_manifest, split_collection, validate_pair, write_pair,
)
from vos_memory_inspector.training_plan import build_plan
from vos_memory_inspector.training_rollout import inject_checkpoint
from vos_memory_inspector.training_runner import (
    CheckpointTranslator, TrainConfig, component_loss, fit_scales, guarded_forward,
    read_checkpoint, train, move_state,
)
from vos_memory_inspector.training_smoke import (
    SyntheticPredictor, fresh_synthetic_runtime, synthetic_case, synthetic_models,
    synthetic_pair,
)
from vos_memory_inspector.training_storage import ExclusiveWriter, checked_path, write_json
from vos_memory_inspector.translators import LinearStateTranslator
from vos_memory_inspector.translators import ResidualMLPStateTranslator


def existing_mlp_test_factory(*, source_spec, target_spec, hidden_dim):
    """Use the repository's existing model solely as an external factory fixture."""
    return ResidualMLPStateTranslator(source_spec, target_spec, hidden_dim=hidden_dim)


def collection(tmp_path, modes=("native_history",)):
    models = synthetic_models()
    entries = [write_pair(tmp_path, synthetic_case(video, mode), *synthetic_pair(i+7),
                         models, allow_synthetic=True)
               for mode in modes for i, video in enumerate(("a", "b", "c"))]
    save_manifest(tmp_path / "manifest.json", {"schema_version": COLLECTION_SCHEMA,
                  "contract": CONTRACT, "models": models, "pairs": entries, "state": "complete"})
    return read_manifest(tmp_path / "manifest.json")


def datasets(tmp_path):
    manifest = collection(tmp_path / "collection")
    paths = split_collection(manifest, tmp_path / "splits")
    return tuple(PairedStateDataset(tmp_path / "collection", paths[f"native_history.{role}"],
                 role=role, pair_mode="native_history", allow_synthetic=True) for role in ("fit", "dev"))


@pytest.mark.parametrize("name", ["frame_indices", "slot_order", "is_conditioning", "validity"])
def test_pair_rejects_record_mismatch(name):
    source, target = synthetic_pair()
    changed = getattr(target, name).clone()
    if changed.dtype == torch.bool:
        changed[0, 0, 0] = ~changed[0, 0, 0]
    else:
        changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match=name):
        validate_pair(source, replace(target, **{name: changed}))


def test_duplicate_records_and_nonfinite_valid_values_rejected():
    source, _ = synthetic_pair()
    frames = source.frame_indices.clone()
    frames[0, 0, 1] = frames[0, 0, 0]
    duplicate = replace(source, frame_indices=frames)
    with pytest.raises(ValueError, match="duplicate record"):
        validate_pair(duplicate, duplicate)
    source.spatial_memory[0, 0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_pair(source, source)


def test_integrity_and_path_escape_rejected(tmp_path):
    manifest = collection(tmp_path)
    entry = manifest["pairs"][0]
    load_pair(tmp_path, entry, manifest["models"], allow_synthetic=True)
    with (tmp_path / entry["path"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_pair(tmp_path, entry, manifest["models"], allow_synthetic=True)
    with pytest.raises(ValueError, match="escapes"):
        checked_path(tmp_path, "../outside.pt")


def test_video_disjoint_and_separate_pair_modes(tmp_path):
    manifest = collection(tmp_path, ("native_history", "controlled_same_mask"))
    paths = split_collection(manifest, tmp_path / "splits")
    sets = {}
    for mode in ("native_history", "controlled_same_mask"):
        for role in ("fit", "dev"):
            ds = PairedStateDataset(tmp_path, paths[f"{mode}.{role}"], role=role,
                                   pair_mode=mode, allow_synthetic=True)
            sets[mode, role] = {entry["case"]["video_id"] for entry in ds.entries}
    assert not sets["native_history", "fit"] & sets["native_history", "dev"]
    assert sets["native_history", "fit"] == sets["controlled_same_mask", "fit"]
    fit = PairedStateDataset(tmp_path, paths["native_history.fit"], role="fit",
                            pair_mode="native_history", allow_synthetic=True)
    same = read_manifest(paths["native_history.fit"])
    same["role"] = "dev"
    save_manifest(tmp_path / "leaked.json", same)
    dev = PairedStateDataset(tmp_path, tmp_path / "leaked.json", role="dev",
                            pair_mode="native_history", allow_synthetic=True)
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint(fit, dev)


@pytest.mark.parametrize("dataset,official_split", [("MOSEv2", "valid"), ("LVOS v2", "val"), ("VOST", "train"), ("DAVIS", "train")])
def test_forbidden_training_roles(tmp_path, dataset, official_split):
    case = {**synthetic_case(), "dataset": dataset, "official_split": official_split}
    with pytest.raises(ValueError, match="official train"):
        write_pair(tmp_path, case, *synthetic_pair(), synthetic_models())


def test_validity_loss_masks_padding_and_diagnostics():
    source, target = synthetic_pair()
    prediction = target.with_continuous(spatial_memory=target.spatial_memory.clone(),
                    object_pointer=target.object_pointer.clone(), presence_logits=target.presence_logits+100)
    prediction.spatial_memory[~prediction.validity] = float("nan")
    prediction.object_pointer[~prediction.validity] = float("nan")
    parts, loss = component_loss(prediction, target, {"spatial_memory": 1, "object_pointer": 1})
    assert loss.item() == 0
    assert set(parts) == {"spatial_memory", "object_pointer"}
    source.spatial_memory[~source.validity] = float("nan")
    source.object_pointer[~source.validity] = float("nan")
    model = LinearStateTranslator(source.spec, target.spec)
    safe = move_state(source, "cpu")
    prediction = guarded_forward(model, safe, target.spec)
    _, loss = component_loss(prediction, target, {"spatial_memory": 1, "object_pointer": 1})
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_resume_exact_checkpoint_and_actual_injector(tmp_path):
    torch.set_num_threads(1)
    fit, dev = datasets(tmp_path)
    config = TrainConfig(epochs=2, learning_rate=.02)
    train(fit, dev, tmp_path / "resumed", config)
    train(fit, dev, tmp_path / "resumed", replace(config, epochs=4),
          resume=tmp_path / "resumed" / "latest.json")
    train(fit, dev, tmp_path / "full", replace(config, epochs=4))
    resumed, full = [read_checkpoint(tmp_path / name / "latest.json") for name in ("resumed", "full")]
    for name, value in resumed["model"].items():
        assert torch.equal(value, full["model"][name])
    assert resumed["scales"] == fit_scales(fit, "fit_rms")
    runtime = fresh_synthetic_runtime()
    translator = CheckpointTranslator(tmp_path / "resumed" / "latest.json")
    source, target, _ = fit[0]
    expected = translator.model(move_state(source, "cpu"))
    actual = translator.translate(source)
    assert torch.allclose(actual.spatial_memory, expected.spatial_memory)
    assert torch.allclose(actual.object_pointer, expected.object_pointer)
    result = inject_checkpoint(translator, source, target, SyntheticPredictor(), runtime)
    assert result["records"] == source.valid_record_count()
    for history in runtime["output_dict_per_obj"].values():
        for outputs in history.values():
            for record in outputs.values():
                assert set(record) == {"maskmem_features", "maskmem_pos_enc", "obj_ptr"}
                assert torch.all(record["maskmem_pos_enc"][0] == 3)
    with pytest.raises(ValueError, match="differs"):
        train(fit, dev, tmp_path / "resumed", replace(config, epochs=5, seed=99),
              resume=tmp_path / "resumed" / "latest.json")


def test_freezing_keeps_input_autograd():
    target = freeze_model(torch.nn.Linear(3, 2))
    value = torch.ones(1, 3, requires_grad=True)
    target(value).sum().backward()
    assert value.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in target.parameters())


def test_factory_interface_rejects_diagnostic_mutation():
    source, target = synthetic_pair()
    class Bad(torch.nn.Module):
        def forward(self, state):
            return state.with_continuous(spatial_memory=state.spatial_memory,
                   object_pointer=state.object_pointer, presence_logits=state.presence_logits+1)
    with pytest.raises(ValueError, match="diagnostic"):
        guarded_forward(Bad(), source, target.spec)


def test_manifest_corruption_and_writer_lock(tmp_path):
    manifest = collection(tmp_path)
    value = json.loads((tmp_path / "manifest.json").read_text())
    value["state"] = "edited"
    write_json(tmp_path / "manifest.json", value)
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(tmp_path / "manifest.json")
    with ExclusiveWriter(tmp_path):
        with pytest.raises(FileExistsError):
            with ExclusiveWriter(tmp_path):
                pass


def test_lvos_sparse_frame_staging_and_switch_mapping(tmp_path):
    dataset = tmp_path / "dataset"
    frames = dataset / "JPEGImages" / "video"
    annotations = dataset / "Annotations" / "video"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for index in (1, 6, 11, 16):
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(frames / f"{index:08d}.jpg")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(annotations / "00000001.png")
    value = {"dataset": "LVOS v2", "release": "v2", "split": "train", "cases": [
             {"video_id": "video", "object_id": "1", "switch_frame": 11, "first_prompt_frame": 1}]}
    save_manifest(tmp_path / "source.json", value)
    plan = build_plan(tmp_path / "source.json", dataset, tmp_path, tmp_path / "plan.json", stage_method="copy")
    assert plan["cases"][0]["switch_frame"] == 2
    prepared = prepare_case(tmp_path, plan["cases"][0])
    assert prepared["num_frames"] == 4
    raw = {**plan["cases"][0], "pair_mode": "controlled_same_mask"}
    with pytest.raises(ValueError, match="every active"):
        prepare_case(tmp_path, raw)


def test_collection_resume_verifies_committed_shards(tmp_path, monkeypatch):
    from vos_memory_inspector import training_collection as module
    video = tmp_path / "video"
    video.mkdir()
    for index in range(7):
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(video / f"{index:06d}.jpg")
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[2:] = 9
    Image.fromarray(mask).save(tmp_path / "mask.png")
    raw = {"dataset": "MOSEv2", "release": "v2", "official_split": "train", "video_id": "video",
           "video_dir": "video", "switch_frame": 5, "pair_mode": "native_history",
           "preprocessing": {"image_size": 1024}, "prompt_events": [
               {"frame_index": frame, "object_id": obj, "kind": "mask", "mask_path": "mask.png"}
               for frame, obj in ((0, 1), (2, 9))]}
    write_json(tmp_path / "plan.json", {"schema_version": "cmmt.collection_plan.v1", "cases": [raw]})
    monkeypatch.setattr(module, "model_provenance", lambda *_: synthetic_models())
    calls = []
    def collect_model(*args):
        calls.append(args[3])
        return synthetic_pair()[0 if args[3] == "source" else 1]
    monkeypatch.setattr(module, "collect_model", collect_model)
    first = module.collect(tmp_path / "plan.json", tmp_path, tmp_path / "collected", tmp_path, device="cpu")
    module.collect(tmp_path / "plan.json", tmp_path, tmp_path / "collected", tmp_path, device="cpu")
    assert calls == ["source", "target"]
    entry = first["pairs"][0]
    with (tmp_path / "collected" / entry["path"]).open("ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="integrity"):
        module.collect(tmp_path / "plan.json", tmp_path, tmp_path / "collected", tmp_path, device="cpu")


def test_prefix_boundary_and_controlled_empty_mask(tmp_path):
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "empty.png")
    class Predictor:
        def __init__(self):
            self.frames = []
            self.masks = []
        def add_new_mask(self, state, *, frame_idx, obj_id, mask):
            self.masks.append(mask)
            state.setdefault("obj_idx_to_id", {})[0] = obj_id
            state.setdefault("output_dict_per_obj", {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}})
            state["output_dict_per_obj"][0]["cond_frame_outputs"][frame_idx] = {
                "maskmem_features": torch.ones(1, 4, 3, 3), "obj_ptr": torch.ones(1, 6)}
        def propagate_in_video(self, state, *, start_frame_idx, max_frame_num_to_track, reverse):
            assert not reverse
            for frame in range(start_frame_idx, start_frame_idx+max_frame_num_to_track+1):
                self.frames.append(frame)
                yield frame, [1], torch.zeros(1, 1, 4, 4)
    predictor = Predictor()
    events = [{"frame_index": frame, "object_id": 1, "mask_path": "empty.png"} for frame in range(4)]
    state = {"device": "cpu", "storage_device": "cpu"}
    canonical = run_prefix(predictor, state, events, tmp_path, 3, allow_empty_masks=True)
    assert canonical.valid_record_count() == 4
    assert max(predictor.frames) == 3
    assert all(not mask.any() for mask in predictor.masks)


def test_mose_train_manifest_uses_position_for_prompt(tmp_path):
    dataset = tmp_path / "dataset"
    frames = dataset / "JPEGImages" / "video"
    masks = dataset / "Annotations" / "video"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    for index in (10, 20, 30, 40):
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(frames / f"{index:05d}.jpg")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(masks / "00010.png")
    save_manifest(tmp_path / "source.json", {"dataset": "MOSEv2", "release": "v2", "split": "train",
                  "cases": [{"video_id": "video", "object_id": 1, "switch_frame": 2, "first_prompt_frame": 0}]})
    plan = build_plan(tmp_path / "source.json", dataset, tmp_path, tmp_path / "plan.json", stage_method="copy")
    assert plan["cases"][0]["switch_frame"] == 2
    assert plan["cases"][0]["prompt_events"][0]["mask_path"].endswith("00010.png")


def test_external_nonlinear_factory_checkpoint(tmp_path):
    fit, dev = datasets(tmp_path)
    config = TrainConfig(factory=f"{__name__}:existing_mlp_test_factory",
                         factory_kwargs={"hidden_dim": 8}, epochs=1)
    train(fit, dev, tmp_path / "nonlinear-fixture", config)
    translator = CheckpointTranslator(tmp_path / "nonlinear-fixture" / "latest.json")
    source, target, _ = fit[0]
    prediction = translator.translate(source)
    assert prediction.spec == target.spec
    assert prediction.object_ids == source.object_ids


def test_legacy_compact_cache_now_rejects_record_alignment(tmp_path):
    from vos_memory_inspector.paired_state_cache import write_paired_state_cache
    source, target = synthetic_pair()
    target.frame_indices = target.frame_indices+1
    with pytest.raises(ValueError, match="frame_indices"):
        write_paired_state_cache(tmp_path / "bad.pt", source_canonical=source,
                                 target_canonical=target, metadata={})
