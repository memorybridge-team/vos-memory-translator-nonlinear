"""Checkpoint-backed, prefix-only collection at declared Small -> Base+ switches."""

from __future__ import annotations

import gc
import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .runner import load_binary_prompt
from .sam2_state import canonicalize_sam2_inference_state
from .training_data import (COLLECTION_SCHEMA, CONTRACT, load_pair, pair_id,
                            read_manifest, save_manifest, validate_case, write_pair)
from .training_storage import ExclusiveWriter, checked_path, content_hash, sha256
from .upstream import verify_sam2_checkout

CONFIGS = {"source": "configs/sam2.1/sam2.1_hiera_s.yaml",
           "target": "configs/sam2.1/sam2.1_hiera_b+.yaml"}


def freeze_model(model):
    """Freeze parameters only; future distillation must keep input autograd enabled."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def video_fingerprint(video: Path) -> tuple[str, int]:
    frames = sorted(video.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames or [int(p.stem) for p in frames] != list(range(len(frames))):
        raise ValueError("stage video as contiguous numeric JPEG frames 0..N-1")
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(f"{frame.name}:{sha256(frame)}\n".encode())
    return digest.hexdigest(), len(frames)


def prepare_case(root: Path, raw: dict) -> dict:
    video = checked_path(root, raw["video_dir"])
    fingerprint, count = video_fingerprint(video)
    if raw.get("video_sha256") and raw["video_sha256"] != fingerprint:
        raise ValueError("video content changed")
    if not 0 <= raw["switch_frame"] < count - 1:
        raise ValueError("switch must leave at least one continuation frame")
    events = sorted(raw["prompt_events"], key=lambda e: (e["frame_index"], e["object_id"]))
    seen = set()
    conditions = []
    for event in events:
        key = event["frame_index"], event["object_id"]
        if key in seen or not 0 <= key[0] <= raw["switch_frame"] or event["kind"] != "mask":
            raise ValueError("invalid/duplicate mask prompt event")
        seen.add(key)
        mask = checked_path(root, event["mask_path"])
        digest = sha256(mask)
        if event.get("sha256") and event["sha256"] != digest:
            raise ValueError("prompt content changed")
        conditions.append({"frame_index": key[0], "object_id": key[1], "kind": "mask",
                           "sha256": digest})
    ids = list(dict.fromkeys(e["object_id"] for e in events))
    if not ids:
        raise ValueError("empty prompt timeline")
    if raw["pair_mode"] == "controlled_same_mask":
        first = {obj: min(e["frame_index"] for e in events if e["object_id"] == obj) for obj in ids}
        required = {(frame, obj) for obj in ids for frame in range(first[obj], raw["switch_frame"] + 1)}
        if seen != required:
            raise ValueError("controlled_same_mask requires a shared mask at every active object/frame")
    return {"dataset": raw["dataset"], "release": raw["release"],
            "official_split": raw["official_split"], "video_id": raw["video_id"],
            "switch_frame": raw["switch_frame"], "pair_mode": raw["pair_mode"],
            "object_ids": ids, "prompt_conditions": conditions,
            "preprocessing": raw["preprocessing"], "video_sha256": fingerprint,
            "num_frames": count}


def run_prefix(predictor, state, events: list, root: Path, switch_frame: int, *, allow_empty_masks=False):
    by_frame = {}
    for event in sorted(events, key=lambda e: (e["frame_index"], e["object_id"])):
        by_frame.setdefault(event["frame_index"], []).append(event)
    cursor = min(by_frame)

    def propagate(stop):
        if stop < cursor:
            return
        for frame, _, _ in predictor.propagate_in_video(
                state, start_frame_idx=cursor, max_frame_num_to_track=stop-cursor, reverse=False):
            if int(frame) > stop:
                raise RuntimeError("upstream propagation exceeded declared prefix")

    for frame in sorted(by_frame):
        if cursor < frame:
            propagate(frame-1)
        for event in by_frame[frame]:
            path = checked_path(root, event["mask_path"])
            if allow_empty_masks:
                with Image.open(path) as image:
                    labels = np.asarray(image)
                if labels.ndim != 2:
                    raise ValueError("controlled masks must be 2D object-label images")
                mask = labels == event["object_id"]
            else:
                mask = load_binary_prompt(path, event["object_id"])
            predictor.add_new_mask(state, frame_idx=frame, obj_id=event["object_id"],
                                   mask=mask)
        cursor = frame
    propagate(switch_frame)
    return canonicalize_sam2_inference_state(state, switch_frame=switch_frame, strict=True)


def model_provenance(root: Path, plan: dict, sam2_repo: Path) -> dict:
    commit = verify_sam2_checkout(sam2_repo)
    result = {}
    for role, architecture in (("source", "small"), ("target", "base_plus")):
        value = plan["models"][role]
        if value["config"] != CONFIGS[role]:
            raise ValueError("model config must be the pinned SAM 2.1 Small/Base+ config")
        config = sam2_repo / "sam2" / value["config"]
        checkpoint = checked_path(root, value["checkpoint"])
        digest = sha256(checkpoint)
        if value.get("checkpoint_sha256") and value["checkpoint_sha256"] != digest:
            raise ValueError("checkpoint hash mismatch")
        result[role] = {"architecture": architecture, "version": "sam2.1",
                        "config": value["config"], "config_sha256": sha256(config),
                        "checkpoint_sha256": digest, "upstream_commit": commit}
    return result


def collect_model(root, raw, plan, role, sam2_repo, device, seed):
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = freeze_model(build_sam2_video_predictor(
        config_file=plan["models"][role]["config"],
        ckpt_path=str(checked_path(root, plan["models"][role]["checkpoint"])), device=device))
    if raw["preprocessing"].get("image_size") != model.image_size:
        raise ValueError("manifest preprocessing image_size differs from predictor")
    context = torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        state = model.init_state(video_path=str(checked_path(root, raw["video_dir"])),
                                 offload_video_to_cpu=True, offload_state_to_cpu=True)
        canonical = run_prefix(model, state, raw["prompt_events"], root, raw["switch_frame"],
                               allow_empty_masks=raw["pair_mode"] == "controlled_same_mask")
    # Shards archive CPU snapshots, not predictor containers or model parameters.
    from .training_data import pack_state, unpack_state
    canonical = unpack_state(pack_state(canonical))
    del state, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return canonical


def collect(plan_path, root, collection, sam2_repo, *, device="cuda", seed=7):
    root, collection, sam2_repo = Path(root).resolve(), Path(collection).resolve(), Path(sam2_repo).resolve()
    if not collection.is_relative_to(root):
        raise ValueError("collection must reside on configured persistent root")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan["schema_version"] != "cmmt.collection_plan.v1":
        raise ValueError("unsupported collection plan")
    models = model_provenance(root, plan, sam2_repo)
    # Validate all plan inputs before any GPU work; identity contains actual content hashes.
    cases = [prepare_case(root, raw) for raw in plan["cases"]]
    for case in cases:
        validate_case(case)
    if not cases or len({pair_id(c) for c in cases}) != len(cases):
        raise ValueError("empty or duplicate collection plan")
    plan_hash = content_hash({"cases": cases, "models": models, "seed": seed})
    from .training_runner import code_provenance
    provenance = {"code": code_provenance(), "device": device}
    path = collection / "manifest.json"
    with ExclusiveWriter(collection):
        manifest = read_manifest(path) if path.exists() else {
            "schema_version": COLLECTION_SCHEMA, "contract": CONTRACT, "models": models,
            "plan_sha256": plan_hash, "seed": seed, "pairs": [], "state": "collecting",
            "collector_provenance": provenance,
            "resources": {"wall_time_seconds": 0.0, "shard_bytes": 0}}
        if manifest["plan_sha256"] != plan_hash or manifest["models"] != models:
            raise ValueError("resume plan/model/seed differs from committed collection")
        if manifest["collector_provenance"] != provenance:
            raise ValueError("resume collector code/environment differs")
        existing = {entry["pair_id"]: entry for entry in manifest["pairs"]}
        save_manifest(path, manifest)
        for raw, case in zip(plan["cases"], cases):
            identity = pair_id(case)
            if identity in existing:
                load_pair(collection, existing[identity], models)
                print(f"verified resume shard {identity}", flush=True)
                continue
            started = time.perf_counter()
            source = collect_model(root, raw, plan, "source", sam2_repo, device, seed)
            target = collect_model(root, raw, plan, "target", sam2_repo, device, seed)
            entry = write_pair(collection, case, source, target, models)
            # Shard is durable before manifest commits it. Orphans after interruption are rewritten.
            load_pair(collection, entry, models)
            manifest["pairs"].append(entry)
            manifest["resources"]["wall_time_seconds"] += time.perf_counter()-started
            manifest["resources"]["shard_bytes"] += entry["bytes"]
            save_manifest(path, manifest)
            print(f"committed shard {identity}: {entry['bytes']} bytes", flush=True)
        manifest["state"] = "complete"
        save_manifest(path, manifest)
    return manifest
