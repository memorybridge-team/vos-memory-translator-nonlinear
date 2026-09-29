"""Strict paired-state shards and video-disjoint fit/dev datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .state_schema import CanonicalState, validate_paired_state_contract
from .training_storage import content_hash, read_checked, write_json, write_tensor_file

CONTRACT = "cmmt.small_to_base_plus.io.v1.1"
COLLECTION_SCHEMA = "cmmt.training_collection.v1"
MODES = {"native_history", "controlled_same_mask"}
TENSORS = ("spatial_memory", "object_pointer", "presence_logits", "frame_indices",
           "slot_order", "is_conditioning", "validity")


def case_lineage(case):
    # 기존 training.v1은 full history의 prompt registry를 joint로 실행했다.
    return {"memory_policy": case.get("memory_policy", {"name": "full_history", "version": 1}),
            "object_semantics": case.get("object_semantics", "prompt_registry_joint_v1"),
            "pair_mode": case["pair_mode"]}


def pack_state(state: CanonicalState) -> dict:
    state.validate()
    return {**{name: getattr(state, name).detach().cpu().clone() for name in TENSORS},
            "object_ids": list(state.object_ids), "switch_frame": state.switch_frame,
            "schema_version": state.schema_version}


def unpack_state(value: dict) -> CanonicalState:
    return CanonicalState(**{name: value[name] for name in TENSORS},
                          object_ids=tuple(value["object_ids"]),
                          switch_frame=value["switch_frame"],
                          schema_version=value["schema_version"]).validate()


def validate_pair(source: CanonicalState, target: CanonicalState) -> None:
    valid = validate_paired_state_contract(source, target)
    if not valid.any():
        raise ValueError("pair contains no valid records")
    if len(set(source.object_ids)) != len(source.object_ids):
        raise ValueError("duplicate object IDs")
    for state in (source, target):
        for name in ("frame_indices", "slot_order"):
            if getattr(state, name).dtype != torch.int64:
                raise ValueError(f"{name} must be int64")
        for batch in range(valid.shape[0]):
            for obj in range(valid.shape[1]):
                frames = state.frame_indices[batch, obj][valid[batch, obj]]
                slots = state.slot_order[batch, obj][valid[batch, obj]]
                if ((frames < 0) | (frames > state.switch_frame)).any():
                    raise ValueError("record outside switch prefix")
                if len(frames.unique()) != len(frames) or len(slots.unique()) != len(slots):
                    raise ValueError("duplicate record identity")
                if (slots < 0).any():
                    raise ValueError("negative valid slot")
        for name in ("spatial_memory", "object_pointer"):
            tensor = getattr(state, name)
            if not tensor.is_floating_point() or not torch.isfinite(tensor[valid]).all():
                raise ValueError(f"non-finite or non-floating valid {name}")


def video_key(case: dict) -> tuple:
    return case["dataset"], case["release"], case["video_id"]


def validate_case(case: dict, *, allow_synthetic: bool = False) -> None:
    allowed = {"MOSEv2", "LVOS v2"} | ({"synthetic"} if allow_synthetic else set())
    if case["dataset"] not in allowed or case["official_split"] != "train":
        raise ValueError("fit/dev accept only MOSEv2/LVOSv2 official train")
    if case["pair_mode"] not in MODES:
        raise ValueError("unknown pair_mode")
    for key in ("release", "video_id", "prompt_conditions", "preprocessing", "video_sha256"):
        if not case.get(key):
            raise ValueError(f"missing case provenance: {key}")
    if not isinstance(case["switch_frame"], int) or case["switch_frame"] < 0:
        raise ValueError("invalid switch_frame")
    ids = case["object_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty/duplicate case object IDs")
    seen = set()
    for event in case["prompt_conditions"]:
        identity = event["frame_index"], event["object_id"]
        if (identity in seen or identity[1] not in ids or event["kind"] != "mask"
                or not 0 <= identity[0] <= case["switch_frame"]):
            raise ValueError("invalid prompt conditions")
        seen.add(identity)
        digest = event["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid prompt checksum")
    if {obj for _, obj in seen} != set(ids):
        raise ValueError("prompt conditions do not cover object registry")
    if case["pair_mode"] == "controlled_same_mask":
        first = {obj: min(frame for frame, key in seen if key == obj) for obj in ids}
        if seen != {(frame, obj) for obj in ids for frame in range(first[obj], case["switch_frame"]+1)}:
            raise ValueError("controlled pairs require shared masks at every active frame")


def pair_id(case: dict) -> str:
    return content_hash(case)


def read_manifest(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = value.pop("content_sha256", None)
    if digest != content_hash(value):
        raise ValueError("manifest content checksum mismatch")
    value["content_sha256"] = digest
    return value


def save_manifest(path: str | Path, manifest: dict) -> None:
    value = {key: val for key, val in manifest.items() if key != "content_sha256"}
    value["content_sha256"] = content_hash(value)
    write_json(path, value)


def write_pair(root: str | Path, case: dict, source: CanonicalState,
               target: CanonicalState, models: dict, *, allow_synthetic=False) -> dict:
    validate_case(case, allow_synthetic=allow_synthetic)
    validate_models(models)
    validate_pair(source, target)
    if source.switch_frame != case["switch_frame"]:
        raise ValueError("case/state switch mismatch")
    if list(source.object_ids) != case["object_ids"]:
        raise ValueError("case/state object mismatch")
    identity = pair_id(case)
    relative = f"shards/{case['pair_mode']}/{identity}.pt"
    payload = {"contract": CONTRACT, "pair_id": identity, "case": case, "models": models,
               "source": pack_state(source), "target": pack_state(target)}
    info = write_tensor_file(Path(root) / relative, payload)
    return {**info, "path": relative, "pair_id": identity, "case": case,
            "source_contract": unpack_state(payload["source"]).contract_dict(),
            "target_contract": unpack_state(payload["target"]).contract_dict()}


def validate_models(models: dict) -> None:
    for role, architecture in (("source", "small"), ("target", "base_plus")):
        model = models[role]
        if model["architecture"] != architecture or model["version"] != "sam2.1":
            raise ValueError("only SAM 2.1 Small -> Base+ is supported")
        for field in ("checkpoint_sha256", "upstream_commit", "config_sha256"):
            digest = model[field]
            length = 40 if field == "upstream_commit" else 64
            if len(digest) != length or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid model provenance: {field}")
    if models["source"]["upstream_commit"] != models["target"]["upstream_commit"]:
        raise ValueError("source/target upstream versions differ")


def load_pair(root: str | Path, entry: dict, models: dict, *, allow_synthetic=False):
    payload = read_checked(root, entry)
    if payload["contract"] != CONTRACT or payload["models"] != models:
        raise ValueError("shard contract/model provenance mismatch")
    case = payload["case"]
    validate_case(case, allow_synthetic=allow_synthetic)
    if case != entry["case"] or pair_id(case) != entry["pair_id"] or payload["pair_id"] != entry["pair_id"]:
        raise ValueError("shard identity mismatch")
    source, target = unpack_state(payload["source"]), unpack_state(payload["target"])
    validate_pair(source, target)
    if source.switch_frame != case["switch_frame"] or list(source.object_ids) != case["object_ids"]:
        raise ValueError("shard case alignment mismatch")
    if source.contract_dict() != entry["source_contract"] or target.contract_dict() != entry["target_contract"]:
        raise ValueError("tensor inventory mismatch")
    return source, target, case


def split_collection(manifest: dict, output: str | Path, *, seed=7, dev_fraction=.2,
                     fit_videos: dict | None = None, dev_videos: dict | None = None) -> dict:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    result = {}
    for mode in sorted({e["case"]["pair_mode"] for e in manifest["pairs"]}):
        entries = [e for e in manifest["pairs"] if e["case"]["pair_mode"] == mode]
        videos = sorted({video_key(e["case"]) for e in entries})
        if len(videos) < 2:
            raise ValueError("video-disjoint fit/dev needs at least two videos per mode")
        if (fit_videos is None) != (dev_videos is None):
            raise ValueError("provide both official fit/dev manifests")
        if fit_videos is not None:
            for value in (fit_videos, dev_videos):
                if value["source_split"] != "train":
                    raise ValueError("split source must be official train")
            fit = {(fit_videos["dataset"], fit_videos["release"], v) for v in fit_videos["videos"]}
            dev = {(dev_videos["dataset"], dev_videos["release"], v) for v in dev_videos["videos"]}
            if fit & dev or not set(videos) <= fit | dev:
                raise ValueError("split overlap or undeclared video")
        else:
            ranked = sorted(videos, key=lambda v: content_hash([seed, *v]))
            count = max(1, min(len(videos)-1, round(len(videos)*dev_fraction)))
            dev, fit = set(ranked[:count]), set(ranked[count:])
        for role, keys in (("fit", fit), ("dev", dev)):
            selected = [e for e in entries if video_key(e["case"]) in keys]
            if not selected:
                raise ValueError("empty fit/dev selection")
            value = {"schema_version": COLLECTION_SCHEMA, "contract": CONTRACT,
                     "models": manifest["models"], "pairs": selected, "role": role,
                     "pair_mode": mode, "seed": seed,
                     "parent_content_sha256": manifest["content_sha256"]}
            path = Path(output) / f"{mode}.{role}.json"
            save_manifest(path, value)
            result[f"{mode}.{role}"] = str(path)
    return result


class PairedStateDataset(Dataset):
    """Loads one CPU shard at a time; never caches a whole dataset in RAM."""

    def __init__(self, root, manifest, *, role, pair_mode, allow_synthetic=False):
        self.root = Path(root)
        self.manifest = read_manifest(manifest)
        value = self.manifest
        if value["schema_version"] != COLLECTION_SCHEMA or value["contract"] != CONTRACT:
            raise ValueError("unsupported training manifest")
        if role not in {"fit", "dev", "overfit"} or value["role"] != role or value["pair_mode"] != pair_mode:
            raise ValueError("dataset role/mode mismatch")
        validate_models(value["models"])
        self.entries = value["pairs"]
        self.allow_synthetic = allow_synthetic
        if not self.entries or len({e["pair_id"] for e in self.entries}) != len(self.entries):
            raise ValueError("empty or duplicate pairs")
        self.lineage = case_lineage(self.entries[0]["case"])
        if any(case_lineage(e["case"]) != self.lineage for e in self.entries):
            raise ValueError("mixed memory policies or object semantics")
        if "lineage" in value and value["lineage"] != self.lineage:
            raise ValueError("manifest/case lineage mismatch")
        for entry in self.entries:
            validate_case(entry["case"], allow_synthetic=allow_synthetic)
            if entry["case"]["pair_mode"] != pair_mode:
                raise ValueError("mixed controlled/native pairs")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return load_pair(self.root, self.entries[index], self.manifest["models"],
                         allow_synthetic=self.allow_synthetic)


def check_disjoint(fit: PairedStateDataset, dev: PairedStateDataset) -> None:
    if fit.lineage != dev.lineage:
        raise ValueError("fit/dev memory policy or object semantics differs")
    if fit.manifest["models"] != dev.manifest["models"]:
        raise ValueError("fit/dev model provenance differs")
    if fit.manifest["pair_mode"] != dev.manifest["pair_mode"]:
        raise ValueError("fit/dev pair modes differ")
    if {video_key(e["case"]) for e in fit.entries} & {video_key(e["case"]) for e in dev.entries}:
        raise ValueError("fit/dev video leakage")
