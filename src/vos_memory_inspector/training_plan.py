"""Convert current MOSE/LVOS train case manifests into explicit collection plans."""

from __future__ import annotations

import shutil
from pathlib import Path

from .training_data import read_manifest, video_key
from .training_storage import content_hash, sha256, write_json


def build_plan(manifest_path, dataset_root, persistent_root, output, *,
               stage_method="symlink", video_ids=None, fit_videos=None, dev_videos=None,
               videos_per_split=None, switch_limit=None,
               source_checkpoint="checkpoints/sam2.1_hiera_small.pt",
               target_checkpoint="checkpoints/sam2.1_hiera_base_plus.pt"):
    manifest = read_manifest(manifest_path)
    if manifest["dataset"] not in {"MOSEv2", "LVOS v2"} or manifest["split"] != "train":
        raise ValueError("collection plan accepts only MOSEv2/LVOS v2 official train")
    root, dataset = Path(persistent_root).resolve(), Path(dataset_root).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError("dataset must reside under persistent root")
    split_root = dataset / "train" if (dataset / "train").is_dir() else dataset
    cases = manifest["cases"]
    if video_ids:
        cases = [c for c in cases if c["video_id"] in video_ids]
    if videos_per_split is not None:
        if videos_per_split < 1 or fit_videos is None or dev_videos is None:
            raise ValueError("videos_per_split requires fit/dev manifests and positive count")
        selected = set()
        for split, expected in ((fit_videos, "fit"), (dev_videos, "development")):
            if (split["dataset"] != manifest["dataset"] or split["source_split"] != "train"
                    or split["split"] != expected):
                raise ValueError("official video manifest role/dataset mismatch")
            available = sorted(set(split["videos"]) & {c["video_id"] for c in cases})
            selected.update(available[:videos_per_split])
        cases = [c for c in cases if c["video_id"] in selected]
    groups = {}
    for case in cases:
        groups.setdefault(case["video_id"], []).append(case)
    if not groups:
        raise ValueError("empty collection selection")
    result = []
    for video_id, rows in sorted(groups.items()):
        frames = sorted((split_root / "JPEGImages" / video_id).glob("*.jpg"), key=lambda p: int(p.stem))
        if not frames:
            raise FileNotFoundError(f"no RGB frames: {video_id}")
        stems = [p.stem for p in frames]
        by_stem = {int(stem): index for index, stem in enumerate(stems)}
        staged = root / "staging" / manifest["dataset"].replace(" ", "_") / video_id
        staged.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            target = staged / f"{index:06d}.jpg"
            if target.exists():
                if sha256(target) != sha256(frame):
                    raise ValueError("existing staged frame differs")
            elif stage_method == "symlink":
                target.symlink_to(frame)
            elif stage_method == "copy":
                shutil.copy2(frame, target)
            else:
                raise ValueError("stage_method must be symlink or copy")
        mapping = {"official_frame_stems": stems, "source_manifest_sha256": manifest["content_sha256"]}
        write_json(staged / "frame_map.json", mapping)
        switches = sorted({row["switch_frame"] for row in rows})
        if switch_limit:
            switches = switches[:switch_limit]
        for official_switch in switches:
            selected = [row for row in rows if row["switch_frame"] == official_switch]
            events = []
            for row in selected:
                if manifest["dataset"] == "LVOS v2":
                    switch = by_stem[official_switch]
                    first = by_stem[row["first_prompt_frame"]]
                    annotation_stem = stems[first]
                else:
                    switch = official_switch
                    if "first_prompt_frame_stem" in row:
                        annotation_stem = row["first_prompt_frame_stem"]
                        first = stems.index(annotation_stem)
                    else:
                        first = int(row["first_prompt_frame"])
                        annotation_stem = stems[first]
                annotation = split_root / "Annotations" / video_id / f"{annotation_stem}.png"
                if not annotation.is_file():
                    raise FileNotFoundError(f"declared prompt annotation unavailable: {annotation}")
                event = {"frame_index": first, "object_id": int(row["object_id"]), "kind": "mask",
                         "mask_path": annotation.relative_to(root).as_posix(), "sha256": sha256(annotation)}
                if event not in events:
                    events.append(event)
            result.append({"dataset": manifest["dataset"], "release": manifest["release"],
                           "official_split": "train", "video_id": video_id,
                           "video_dir": staged.relative_to(root).as_posix(),
                           "switch_frame": switch, "official_switch_frame": official_switch,
                           "pair_mode": "native_history", "prompt_events": events,
                           "preprocessing": {"image_size": 1024,
                              "frame_indexing": "zero_based_staged_rgb_order",
                              "frame_map_sha256": content_hash(mapping)}})
    value = {"schema_version": "cmmt.collection_plan.v1", "source_manifest_sha256": manifest["content_sha256"],
             "models": {"source": {"config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                                      "checkpoint": source_checkpoint},
                        "target": {"config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                                      "checkpoint": target_checkpoint}},
             "cases": result}
    write_json(output, value)
    return value
