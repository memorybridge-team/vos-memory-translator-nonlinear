"""ZIP 단일 객체 수집의 명시적 조건·frame map·memory 정책 계약."""
from __future__ import annotations

from pathlib import Path
import json
import torch

from .training_data import read_manifest, validate_case, validate_models, validate_pair
from .training_storage import content_hash, sha256

SEMANTICS = "single_object_independent_v1"
POLICIES = {"full_history", "active_window_v1"}
READ_FIELDS = ("num_maskmem", "max_obj_ptrs_in_encoder", "memory_temporal_stride_for_eval",
               "max_cond_frames_in_attn", "only_obj_ptrs_in_the_past_for_eval",
               "use_obj_ptrs_in_encoder", "add_all_frames_to_correct_as_cond")


class Rejection(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition, code, detail=""):
    if not condition:
        raise Rejection(code, detail)


def jpeg_map(video: Path, *, hash_pixels=True):
    """JPEG만 numeric 정렬한다. sparse 공식 ID는 연속 runtime index와 다르다."""
    frames = [p for p in Path(video).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    require(bool(frames), "NO_JPEG", str(video))
    require(all(p.stem.isascii() and p.stem.isdigit() for p in frames), "NON_NUMERIC_JPEG")
    frames.sort(key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in frames]
    require(len(set(ids)) == len(ids), "DUPLICATE_FRAME_ID")
    records = [{"runtime_index": i, "official_id": int(p.stem), "filename": p.name,
                **({"sha256": sha256(p)} if hash_pixels else {})} for i, p in enumerate(frames)]
    return frames, records


def resolve_indices(case, records):
    if case["dataset"] == "MOSEv2":
        if "frame_count" in case:
            require(len(records) == case["frame_count"], "FRAME_COUNT_MISMATCH")
        prompt, switch = int(case["first_prompt_frame"]), int(case["switch_frame"])
    else:
        lookup = {r["official_id"]: r["runtime_index"] for r in records}
        for label in ("first_prompt_frame", "switch_frame"):
            require(int(case[label]) in lookup, "MISSING_FRAME_ID", f"{label}={case[label]}")
        prompt, switch = lookup[int(case["first_prompt_frame"])], lookup[int(case["switch_frame"])]
    require(0 <= prompt <= switch < len(records)-1, "PROMPT_SWITCH_BOUNDS")
    return prompt, switch


def frozen_selection(source_path, fit_path, dev_path, *, paired_split="all", case_ids=None):
    source, fit, dev = [read_manifest(p) for p in (source_path, fit_path, dev_path)]
    require(source["dataset"] in {"MOSEv2", "LVOS v2"}, "FORBIDDEN_DATASET")
    require(source.get("official_split", source.get("split")) == "train", "FORBIDDEN_SPLIT")
    for membership, role in ((fit, "fit"), (dev, "development")):
        require(membership["source_split"] == "train" and membership["split"] == role,
                "MEMBERSHIP_ROLE")
        require(membership["dataset"] == source["dataset"] and membership["release"] == source["release"],
                "MEMBERSHIP_DATASET")
        require(membership["source_manifest_content_sha256"] == source["content_sha256"],
                "MEMBERSHIP_SOURCE")
    a, b = set(fit["videos"]), set(dev["videos"])
    require(not a & b, "SPLIT_OVERLAP")
    require(len(a) == len(fit["videos"]) and len(b) == len(dev["videos"]), "DUPLICATE_VIDEO")
    cases = []
    for c in source["cases"]:
        require(c["official_split"] == "train" and c["dataset"] == source["dataset"], "FORBIDDEN_SPLIT")
        require(c["video_id"] in a | b, "UNDECLARED_VIDEO", c["video_id"])
        role = "fit" if c["video_id"] in a else "development"
        if paired_split == "all" or role == paired_split:
            cases.append({**c, "paired_split": role})
    # ZIP의 modulo 순서는 그대로 보존한다. 먼저 global selection을 동결한다.
    cases.sort(key=lambda c: (c["paired_split"], c["case_id"]))
    if case_ids is not None:
        requested = set(case_ids)
        cases = [c for c in cases if c["case_id"] in requested]
        require({c["case_id"] for c in cases} == requested, "UNKNOWN_CASE_ID")
    require(cases and len({c["case_id"] for c in cases}) == len(cases), "EMPTY_DUPLICATE_SELECTION")
    return {"schema_version": "cmmt.paired_global_selection.v1", "dataset": source["dataset"],
            "source_manifest_content_sha256": source["content_sha256"],
            "fit_membership": fit, "development_membership": dev, "cases": cases,
            "selection_digest": content_hash(cases)}


def shard_plan(selection, count=8):
    require(count >= 1, "INVALID_SHARD_COUNT")
    require(selection["selection_digest"] == content_hash(selection["cases"]), "SELECTION_DIGEST")
    ids = [c["case_id"] for c in selection["cases"]]
    require(len(set(ids)) == len(ids), "DUPLICATE_SELECTION")
    assignments = [ids[i::count] for i in range(count)]
    flat = [key for shard in assignments for key in shard]
    require(len(flat) == len(set(flat)) and set(flat) == set(ids), "SHARD_COVERAGE")
    cost_by_id = {c["case_id"]: int(c.get("frame_count", c.get("frame_count_for_object", 0))) for c in selection["cases"]}
    return {"selection_digest": selection["selection_digest"], "shard_count": count,
            "assignment_policy": "legacy_modulo_v1", "coverage_exact": True,
            "shards": [{"index": i, "case_ids": keys, "case_count": len(keys),
                        "cost_proxy": sum(cost_by_id[key] for key in keys)}
                       for i, keys in enumerate(assignments)],
            "cost_note": "총 frame 수 기반 proxy; sparse runtime prefix·실측 throughput 아님"}


def memory_policy(name, num_maskmem=7, max_obj_ptrs_in_encoder=16):
    require(name in POLICIES and num_maskmem > 0 and max_obj_ptrs_in_encoder > 0, "MEMORY_POLICY")
    return {"name": name, "version": 1, "num_maskmem": num_maskmem,
            "max_obj_ptrs_in_encoder": max_obj_ptrs_in_encoder,
            "selection_rule": "all_conditioning_plus_latest_nonconditioning" if name == "active_window_v1" else "all_valid_records",
            "equivalence": "provisional_requires_gpu_gate" if name == "active_window_v1" else "full_history_reference"}


def observed_read_policy(predictor):
    return {key: getattr(predictor, key) for key in READ_FIELDS}


def apply_memory_policy(state, policy):
    """ZIP의 선택 휴리스틱을 보존한다. SAM read set의 보편적 증명이 아니다."""
    state.validate()
    require(state.spatial_memory.shape[0] == 1, "POLICY_BATCH")
    selections, mapping = [], []
    limit = max(policy["num_maskmem"]-1, policy["max_obj_ptrs_in_encoder"]-1)
    for obj in range(len(state.object_ids)):
        valid = torch.where(state.validity[0, obj])[0].tolist()
        cond = [i for i in valid if bool(state.is_conditioning[0, obj, i])]
        noncond = sorted([i for i in valid if i not in cond], key=lambda i: int(state.frame_indices[0, obj, i]))
        keep = valid if policy["name"] == "full_history" else cond + (noncond[-limit:] if limit else [])
        keep.sort(key=lambda i: (int(state.frame_indices[0, obj, i]), not bool(state.is_conditioning[0, obj, i])))
        require(bool(keep), "EMPTY_SELECTED_MEMORY")
        selections.append(keep)
        mapping.append([{"old_slot": int(state.slot_order[0, obj, i]), "new_slot":
                         int(state.slot_order[0, obj, i]) if policy["name"] == "full_history" else j,
                         "runtime_frame": int(state.frame_indices[0, obj, i]),
                         "conditioning": bool(state.is_conditioning[0, obj, i])} for j, i in enumerate(keep)])
    if policy["name"] == "full_history":
        result = state
    else:
        from .training_data import pack_state, unpack_state, TENSORS
        packed = pack_state(state)
        size = max(map(len, selections))
        for name in TENSORS:
            tensor = packed[name]
            value = tensor.new_full((*tensor.shape[:2], size, *tensor.shape[3:]),
                                    -1 if name in {"frame_indices", "slot_order"} else 0)
            for obj, keep in enumerate(selections):
                value[0, obj, :len(keep)] = tensor[0, obj, keep]
                if name == "slot_order":
                    value[0, obj, :len(keep)] = torch.arange(len(keep))
            packed[name] = value
        result = unpack_state(packed)
    result.metadata = {**result.metadata, "memory_selection": {"policy": policy, "old_to_new_slots": mapping,
                        "records_before": state.valid_record_count(), "records_after": result.valid_record_count()}}
    return result


def validate_generating(value):
    require(value.get("schema_version") == "cmmt.paired_generating.v1", "GENERATING_SCHEMA")
    try:
        validate_models(value["models"])
        validate_case(value["case"], allow_synthetic=value.get("synthetic", False))
        case = value["case"]
        require(case["object_semantics"] == SEMANTICS and len(case["object_ids"]) == 1, "OBJECT_SEMANTICS")
        require(case["memory_policy"] == memory_policy(case["memory_policy"]["name"],
                case["memory_policy"]["num_maskmem"], case["memory_policy"]["max_obj_ptrs_in_encoder"]), "MEMORY_POLICY")
        for key in ("frame_map_sha256", "source_manifest_content_sha256", "case_id", "paired_split"):
            require(bool(case[key]), "MISSING_GENERATING", key)
        records = value["frame_map"]
        require(case["frame_map_sha256"] == content_hash(records), "FRAME_MAP_HASH")
        require([r["runtime_index"] for r in records] == list(range(len(records))), "FRAME_MAP_INDEX")
        require(len({r["official_id"] for r in records}) == len(records), "DUPLICATE_FRAME_ID")
        require(case["video_sha256"] == content_hash(records), "VIDEO_PROVENANCE")
        for record in records:
            require(len(record["sha256"]) == 64 and all(c in "0123456789abcdef" for c in record["sha256"]),
                    "FRAME_CONTENT_HASH")
        require(case["num_frames"] == len(records) and case["switch_frame"] < len(records)-1, "FRAME_MAP_BOUNDS")
        first = min(event["frame_index"] for event in case["prompt_conditions"])
        if case["dataset"] == "LVOS v2":
            require(records[first]["official_id"] == int(case["official_prompt_frame"]) and
                    records[case["switch_frame"]]["official_id"] == int(case["official_switch_frame"]), "OFFICIAL_RUNTIME_MAPPING")
        elif case["dataset"] == "MOSEv2":
            require(first == int(case["official_prompt_frame"]) and case["switch_frame"] == int(case["official_switch_frame"]),
                    "OFFICIAL_RUNTIME_MAPPING")
        for label in ("source", "target"):
            observed = value["effective_model_policy"][label]
            require(all(key in observed for key in READ_FIELDS), "MISSING_READ_POLICY")
            require(observed["num_maskmem"] == case["memory_policy"]["num_maskmem"] and
                    observed["max_obj_ptrs_in_encoder"] == case["memory_policy"]["max_obj_ptrs_in_encoder"], "READ_POLICY_MISMATCH")
            if case["memory_policy"]["name"] == "active_window_v1":
                require(observed["memory_temporal_stride_for_eval"] == 1, "UNSUPPORTED_ACTIVE_STRIDE")
        require(value["collection_mode"] in {"state_only", "handoff_full"}, "COLLECTION_MODE")
        require(isinstance(value["seed"], int) and value["collector_source_hashes"] and value["preprocessing"],
                "MISSING_GENERATING")
        require(value["preprocessing"] == case["preprocessing"], "PREPROCESSING_MISMATCH")
        for digest in value["collector_source_hashes"].values():
            require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "SOURCE_HASH")
    except (KeyError, TypeError) as exc:
        raise Rejection("MISSING_GENERATING", str(exc)) from exc
    return value


def validate_cache_generating(payload, generating):
    validate_generating(generating)
    source, target = payload["source_canonical"], payload["target_canonical"]
    try:
        validate_pair(source, target)
    except ValueError as exc:
        raise Rejection("PAIR_ALIGNMENT", str(exc)) from exc
    case = generating["case"]
    require(source.switch_frame == case["switch_frame"] and list(source.object_ids) == case["object_ids"], "CASE_ALIGNMENT")
    first = min(e["frame_index"] for e in case["prompt_conditions"])
    for state in (source, target):
        require(not (state.frame_indices[state.validity] < first).any(), "RECORD_BEFORE_PROMPT")
        for event in case["prompt_conditions"]:
            require(((state.frame_indices == event["frame_index"]) & state.is_conditioning & state.validity).any(),
                    "PROMPT_CONDITIONING_MISMATCH")
        # ZIP production은 객체 하나의 최초 mask 이후 매 runtime frame을 추적한다.
        cond = {e["frame_index"] for e in case["prompt_conditions"]}
        require(set(state.frame_indices[state.validity & state.is_conditioning].tolist()) == cond,
                "CONDITIONING_TIMELINE_MISMATCH")
        noncond = [f for f in range(first, case["switch_frame"]+1) if f not in cond]
        limit = max(case["memory_policy"]["num_maskmem"]-1, case["memory_policy"]["max_obj_ptrs_in_encoder"]-1)
        expected = cond | set(noncond if case["memory_policy"]["name"] == "full_history" else (noncond[-limit:] if limit else []))
        require(set(state.frame_indices[state.validity].tolist()) == expected, "MISSING_SELECTED_MEMORY")
    metadata = payload["metadata"]
    require(metadata["switch_frame"] == source.switch_frame, "METADATA_SWITCH")
    require(metadata["cache_mode"] == generating["collection_mode"], "COLLECTION_MODE")
    if "seed" in metadata:
        require(metadata["seed"] == generating["seed"], "SEED_MISMATCH")
    if "num_frames" in metadata:
        require(metadata["num_frames"] == case["num_frames"], "VIDEO_COUNT_MISMATCH")
    if "prompt_frame_index" in metadata:
        require(metadata["prompt_frame_index"] == first, "PROMPT_TIMING")
    if "upstream_commit" in metadata:
        require(metadata["upstream_commit"] == generating["models"]["source"]["upstream_commit"], "UPSTREAM_MISMATCH")
    if "active_memory_only" in metadata:
        require(metadata["active_memory_only"] == (case["memory_policy"]["name"] == "active_window_v1"), "MEMORY_POLICY_MISMATCH")
    if metadata.get("memory_policy"):
        require(metadata["memory_policy"] == case["memory_policy"], "MEMORY_POLICY_MISMATCH")
    for role in ("source", "target"):
        for key in ("checkpoint_sha256", "config_sha256"):
            if f"{role}_{key}" in metadata:
                require(metadata[f"{role}_{key}"] == generating["models"][role][key], "MODEL_HASH_MISMATCH")
    # legacy metadata에 실재하는 값은 외부 근거가 있어도 덮어쓰지 않는다.
    if "generating" in metadata:
        require(metadata["generating"] == generating, "GENERATING_MISMATCH")
    return source, target
