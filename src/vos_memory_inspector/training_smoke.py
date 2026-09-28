"""Synthetic engineering checks only; never substitutes for real VOS/GPU gates."""

from pathlib import Path
import time

import torch

from .sam2_state import canonicalize_sam2_inference_state
from .state_schema import CanonicalState
from .training_data import (COLLECTION_SCHEMA, CONTRACT, PairedStateDataset,
                            read_manifest, save_manifest, split_collection, write_pair)
from .training_rollout import inject_checkpoint
from .training_runner import CheckpointTranslator, TrainConfig, read_checkpoint, train
from .training_storage import write_json


def synthetic_models():
    return {role: {"architecture": architecture, "version": "sam2.1",
                   "checkpoint_sha256": char*64, "config_sha256": char*64,
                   "upstream_commit": "a"*40, "synthetic": True}
            for role, architecture, char in (("source", "small", "b"), ("target", "base_plus", "c"))}


def synthetic_pair(seed=7):
    generator = torch.Generator().manual_seed(seed)
    valid = torch.ones(1, 2, 6, dtype=torch.bool)
    valid[:, 1, :2] = False
    source = CanonicalState(
        spatial_memory=torch.randn(1, 2, 6, 4, 3, 3, generator=generator),
        object_pointer=torch.randn(1, 2, 6, 6, generator=generator),
        presence_logits=torch.zeros(1, 2, 6, 1),
        frame_indices=torch.arange(6).view(1, 1, 6).expand(1, 2, 6).clone(),
        slot_order=torch.arange(6).view(1, 1, 6).expand(1, 2, 6).clone(),
        is_conditioning=torch.zeros_like(valid), validity=valid, object_ids=(1, 9), switch_frame=5)
    source.is_conditioning[:, 0, 0] = True
    source.is_conditioning[:, 1, 2] = True
    target = source.with_continuous(spatial_memory=source.spatial_memory*.7+.2,
               object_pointer=source.object_pointer*.4-.1, presence_logits=source.presence_logits.clone())
    return source, target


def synthetic_case(video="video-a", mode="native_history"):
    conditions = [{"frame_index": frame, "object_id": obj, "kind": "mask", "sha256": "d"*64}
                  for obj, first in ((1, 0), (9, 2))
                  for frame in (range(first, 6) if mode == "controlled_same_mask" else [first])]
    return {"dataset": "synthetic", "release": "test-v1", "official_split": "train",
            "video_id": video, "switch_frame": 5, "pair_mode": mode, "object_ids": [1, 9],
            "prompt_conditions": conditions,
            "preprocessing": {"image_size": "synthetic_grid"}, "video_sha256": "e"*64}


class SyntheticPosition(torch.nn.Module):
    def forward(self, tensor):
        return torch.full_like(tensor, 3)


class SyntheticPredictor:
    def __init__(self):
        self.memory_encoder = type("Encoder", (), {})()
        self.memory_encoder.position_encoding = SyntheticPosition()


def fresh_synthetic_runtime():
    return {"device": "cpu", "storage_device": "cpu", "constants": {}, "obj_ids": [],
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {}}


def synthetic_smoke(output):
    output = Path(output)
    if (output / "collection" / "manifest.json").exists():
        raise ValueError("synthetic smoke output already exists; choose a new directory")
    torch.set_num_threads(1)
    started = time.perf_counter()
    root = output / "collection"
    models = synthetic_models()
    entries = [write_pair(root, synthetic_case(video), *synthetic_pair(7+i), models,
                         allow_synthetic=True) for i, video in enumerate(("video-a", "video-b", "video-c"))]
    save_manifest(root / "manifest.json", {"schema_version": COLLECTION_SCHEMA, "contract": CONTRACT,
                  "models": models, "pairs": entries, "state": "complete"})
    manifest = read_manifest(root / "manifest.json")
    paths = split_collection(manifest, output / "splits")
    save_manifest(output / "overfit.json", {"schema_version": COLLECTION_SCHEMA, "contract": CONTRACT,
                  "models": models, "pairs": entries[:1], "role": "overfit", "pair_mode": "native_history"})
    fit_one = PairedStateDataset(root, output / "overfit.json", role="overfit",
                  pair_mode="native_history", allow_synthetic=True)
    overfit = train(fit_one, None, output / "overfit-run", TrainConfig(epochs=80,
                    learning_rate=.05, record_batch_size=0), overfit=True)
    if overfit[-1]["fit_probe"]["normalized_loss"] >= overfit[0]["fit_probe"]["normalized_loss"]*.02:
        raise AssertionError("synthetic one-video overfit gate did not converge")
    fit = PairedStateDataset(root, paths["native_history.fit"], role="fit",
                            pair_mode="native_history", allow_synthetic=True)
    dev = PairedStateDataset(root, paths["native_history.dev"], role="dev",
                            pair_mode="native_history", allow_synthetic=True)
    train(fit, dev, output / "dev-run", TrainConfig(epochs=3, learning_rate=.01))
    resumed = train(fit, dev, output / "dev-run", TrainConfig(epochs=6, learning_rate=.01),
                    resume=output / "dev-run" / "latest.json")
    train(fit, dev, output / "uninterrupted-run", TrainConfig(epochs=6, learning_rate=.01))
    a, b = (read_checkpoint(output / run / "latest.json") for run in ("dev-run", "uninterrupted-run"))
    if not all(torch.equal(value, b["model"][name]) for name, value in a["model"].items()):
        raise AssertionError("epoch-boundary resume differs from uninterrupted training")
    translator = CheckpointTranslator(output / "dev-run" / "latest.json")
    source, target, _ = fit[0]
    runtime = fresh_synthetic_runtime()
    injection = inject_checkpoint(translator, source, target, SyntheticPredictor(), runtime)
    exported = canonicalize_sam2_inference_state(runtime, switch_frame=source.switch_frame)
    if exported.valid_record_count() != source.valid_record_count():
        raise AssertionError("actual injector lost records")
    report = {"scope": "synthetic CPU engineering smoke; no real SAM 2 or VOS evaluation",
              "overfit_first_loss": overfit[0]["fit_probe"]["normalized_loss"],
              "overfit_final_loss": overfit[-1]["fit_probe"]["normalized_loss"],
              "dev_final": resumed[-1]["dev"], "resume_weights_exact": True,
              "actual_materializer_injection": injection,
              "shard_bytes": sum(e["bytes"] for e in entries),
              "checkpoint_bytes": sum(p.stat().st_size for p in output.rglob("epoch-*.pt")),
              "wall_time_seconds": time.perf_counter()-started}
    write_json(output / "smoke_report.json", report)
    return report
