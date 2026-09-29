"""명시적으로 신뢰한 팀 legacy cache의 CPU 감사·원본 보존 변환."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import time
import uuid
import pickle
import torch

from .case_cache import validate_case_cache
from .collection_contract import (Rejection, require, validate_cache_generating,
    validate_generating, apply_memory_policy, memory_policy)
from .training_data import (COLLECTION_SCHEMA, CONTRACT, read_manifest, save_manifest,
    write_pair, load_pair, pair_id)
from .training_storage import sha256, content_hash, write_json, ExclusiveWriter

DECISIONS = ("REUSABLE", "CONVERTIBLE", "NEEDS_RECOLLECTION", "PENDING_EVIDENCE")


def append_event(root, value):
    """event별 immutable JSON. 공유 ledger는 한 writer만 사용한다."""
    root = Path(root)
    with ExclusiveWriter(root):
        event = {**value, "event_id": uuid.uuid4().hex, "recorded_at_ns": time.time_ns()}
        path = root / "events" / f"{event['event_id']}.json"
        require(not path.exists(), "LEDGER_EVENT_EXISTS")
        write_json(path, event)
    return path


def _run_evidence(request, digest):
    proof = request.get("run_evidence")
    require(proof is not None, "LEGACY_RUN_EVIDENCE_MISSING")
    path = Path(proof["path"])
    require(sha256(path) == proof["sha256"], "RUN_EVIDENCE_HASH")
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record.get("status") == "reviewed_immutable" and record.get("reviewer") and
            record.get("reviewed_at") and record.get("source_evidence"), "RUN_EVIDENCE_UNREVIEWED")
    bindings = [r for r in record["bindings"] if r["cache_sha256"] == digest]
    require(len(bindings) == 1, "RUN_EVIDENCE_BINDING")
    return bindings[0]["generating"], {"path": str(path), "sha256": proof["sha256"],
                                      "reviewer": record["reviewer"]}


def inspect_cache(request, *, trusted_team_legacy=False, stable_seconds=60):
    """명시된 소수의 완료 파일만 읽는다. recursive tensor scan은 하지 않는다."""
    path = Path(request["path"]).resolve()
    result = {"path": str(path), "case_id": request.get("case_id"),
              "decision": "PENDING_EVIDENCE", "reason_codes": [], "evidence": {}}
    payload = generating = None
    try:
        require(path.is_file(), "MISSING_CACHE")
        require(not Path(str(path) + ".writer.lock").exists(), "ACTIVE_WRITER")
        require(stable_seconds >= 0, "STABILITY_LIMIT")
        before = path.stat()
        require(time.time()-before.st_mtime >= stable_seconds, "UNSTABLE_FILE")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        require(sidecar.is_file(), "MISSING_CHECKSUM")
        tokens = sidecar.read_text(encoding="ascii").split()
        require(tokens and len(tokens[0]) == 64, "CHECKSUM_FORMAT")
        digest = sha256(path)
        require(digest == tokens[0], "CHECKSUM_TAMPERED")
        if request.get("sha256"):
            require(digest == request["sha256"], "EXPECTED_CHECKSUM_MISMATCH")
        result["evidence"] = {"cache_sha256": digest, "bytes": before.st_size,
                              "mtime_ns": before.st_mtime_ns, "stable_seconds": stable_seconds}
        marker = Path(str(path) + ".complete.json")
        if marker.exists():
            complete = json.loads(marker.read_text(encoding="utf-8"))
            require(complete["sha256"] == digest and complete["bytes"] == before.st_size, "COMPLETION_MISMATCH")
        require(trusted_team_legacy, "TRUST_REQUIRED", "CanonicalState pickle는 팀 내부 신뢰 입력에만 허용")
        # 원본 directory에는 read lock도 만들지 않는다. legacy writer는 새 lock을
        # 존중하지 않으므로 operator의 완료 증거와 안정 구간이 필수다.
        require(marker.exists() or request.get("completed_evidence"), "COMPLETED_EVIDENCE_MISSING")
        prior_proof = _run_evidence(request, digest) if request.get("run_evidence") else None
        if prior_proof:
            validate_generating(prior_proof[0])  # 가능한 생성 근거는 pickle 실행 전에 검증한다.
        require((path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns), "FILE_CHANGED")
        require(sha256(path) == digest, "FILE_CHANGED")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise Rejection("INVALID_SERIALIZATION", str(exc)) from exc
        try:
            validate_case_cache(payload)
        except (ValueError, TypeError) as exc:
            raise Rejection("INVALID_CACHE_PAIR", str(exc)) from exc
        require(not Path(str(path) + ".writer.lock").exists() and sha256(path) == digest and
                path.stat().st_mtime_ns == before.st_mtime_ns, "FILE_CHANGED")
        metadata = payload["metadata"]
        result["metadata_summary"] = {"schema_version": payload["schema_version"],
            "cache_mode": metadata.get("cache_mode", "handoff_full"),
            "switch_frame": payload["source_canonical"].switch_frame,
            "object_ids": list(payload["source_canonical"].object_ids),
            "memory_policy": metadata.get("memory_policy", metadata.get("active_memory_only", "UNKNOWN")),
            "has_generating": "generating" in metadata}
        expected = request.get("expected_generating")
        require(expected is not None, "EXPECTED_CONDITIONS_MISSING")
        validate_generating(expected)
        if "prompt_frame_index" in metadata:
            require(metadata["prompt_frame_index"] == min(e["frame_index"] for e in expected["case"]["prompt_conditions"]),
                    "PROMPT_TIMING")
        if "generating" in metadata:
            generating = metadata["generating"]
            result["evidence"]["generating_source"] = "embedded"
            if marker.is_file():
                require(complete["generating_sha256"] == content_hash(generating), "COMPLETION_MISSING")
            else:
                supplied, proof = prior_proof or _run_evidence(request, digest)
                require(supplied == generating, "GENERATING_MISMATCH")
                result["evidence"]["generating_source"] = {"embedded_and_reviewed_run": proof}
        else:
            generating, proof = prior_proof or _run_evidence(request, digest)
            result["evidence"]["generating_source"] = proof
        validate_cache_generating(payload, generating)
        require(generating == expected, "GENERATING_MISMATCH")
        result["case_id"] = generating["case"]["case_id"]
        result["semantic_digest"] = content_hash(generating)
        result["decision"] = "CONVERTIBLE"
        result["reason_codes"] = ["VALIDATED_LEGACY_TO_TENSOR_DICT"]
    except Rejection as exc:
        result["reason_codes"] = [exc.code]
        result["detail"] = str(exc)
        # 생성 이력 부재/변경 중인 파일은 전체 재수집 근거가 아니다.
        if exc.code in {"MISSING_CACHE", "CHECKSUM_TAMPERED", "INVALID_CACHE_PAIR", "INVALID_SERIALIZATION", "PAIR_ALIGNMENT",
                        "CASE_ALIGNMENT", "METADATA_SWITCH", "PROMPT_TIMING", "RECORD_BEFORE_PROMPT",
                        "PROMPT_CONDITIONING_MISMATCH", "MISSING_SELECTED_MEMORY"}:
            result["decision"] = "NEEDS_RECOLLECTION"
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        result["reason_codes"] = ["AUDIT_INPUT_ERROR"]
        result["detail"] = str(exc)
    return result, payload, generating


def audit_requests(requests, ledger, *, trusted_team_legacy=False, stable_seconds=60):
    results = []
    for request in requests:
        result, _, _ = inspect_cache(request, trusted_team_legacy=trusted_team_legacy, stable_seconds=stable_seconds)
        result["ledger_event"] = str(append_event(ledger, result))
        results.append(result)
    return {"schema_version": "cmmt.cache_reuse_plan.v1", "results": results,
            "counts": {label: sum(r["decision"] == label for r in results) for label in DECISIONS},
            "recollection_case_ids": sorted({r["case_id"] for r in results
                                              if r["decision"] == "NEEDS_RECOLLECTION" and r["case_id"]}),
            "scope": "명시한 완료 sample만 검사; 배포 전체 상태를 추정하지 않음"}


def _check_frozen(case, selection):
    require(selection["selection_digest"] == content_hash(selection["cases"]), "SELECTION_DIGEST")
    matches = [c for c in selection["cases"] if c["case_id"] == case["case_id"]]
    require(len(matches) == 1, "UNKNOWN_CASE_ID")
    raw = matches[0]
    require(case["dataset"] == raw["dataset"] and case["release"] == raw["release"] and
            case["official_split"] == raw["official_split"] == "train" and
            case["video_id"] == raw["video_id"] and case["object_ids"] == [int(raw["object_id"])] and
            case["official_switch_frame"] == raw["switch_frame"] and
            case["official_prompt_frame"] == raw["first_prompt_frame"] and
            case["paired_split"] == raw["paired_split"] and
            case["source_manifest_content_sha256"] == selection["source_manifest_content_sha256"], "FROZEN_CASE_MISMATCH")
    for key, split in (("fit_membership", "fit"), ("development_membership", "development")):
        membership = selection[key]
        require(membership["source_split"] == "train", "FORBIDDEN_SPLIT")
        if case["video_id"] in membership["videos"]:
            require(case["paired_split"] == split, "FROZEN_SPLIT_MISMATCH")


def import_requests(requests, output, selection, *, trusted_team_legacy=False,
                    stable_seconds=60, target_policy=None):
    """검증 후 새 root로만 import한다. 동일 의미+생성 조건은 하나의 shard다."""
    output = Path(output).resolve()
    for r in requests:
        require(not output.is_relative_to(Path(r["path"]).resolve().parent), "OUTPUT_INSIDE_ORIGINAL")
    results = []
    with ExclusiveWriter(output):
        path = output / "manifest.json"
        manifest = read_manifest(path) if path.exists() else None
        for request in requests:
            result, payload, generating = inspect_cache(request, trusted_team_legacy=trusted_team_legacy, stable_seconds=stable_seconds)
            if result["decision"] != "CONVERTIBLE":
                result["ledger_event"] = str(append_event(output / "migration-ledger", result))
                results.append(result)
                continue
            try:
                case = deepcopy(generating["case"])
                _check_frozen(case, selection)
                source, target = payload["source_canonical"], payload["target_canonical"]
                conversion = None
                if target_policy and target_policy != case["memory_policy"]["name"]:
                    require(case["memory_policy"]["name"] == "full_history" and target_policy == "active_window_v1",
                            "MISSING_MEMORY_NOT_RECOVERABLE")
                    require(all(generating["effective_model_policy"][role]["memory_temporal_stride_for_eval"] == 1
                                for role in ("source", "target")), "UNSUPPORTED_ACTIVE_STRIDE")
                    old = case["memory_policy"]
                    new = memory_policy(target_policy, old["num_maskmem"], old["max_obj_ptrs_in_encoder"])
                    source, target = apply_memory_policy(source, new), apply_memory_policy(target, new)
                    case["memory_policy"] = new
                    conversion = {"source_policy": old, "target_policy": new,
                                  "source_selection": source.metadata["memory_selection"],
                                  "target_selection": target.metadata["memory_selection"]}
                lineage = {"memory_policy": case["memory_policy"], "object_semantics": case["object_semantics"],
                           "pair_mode": case["pair_mode"]}
                case["generating_sha256"] = content_hash(generating)
                models = generating["models"]
                common = {key: generating.get(key) for key in ("models", "seed", "collection_mode", "collector_source_hashes",
                                                               "effective_model_policy", "preprocessing", "synthetic", "environment")}
                if manifest is None:
                    manifest = {"schema_version": COLLECTION_SCHEMA, "contract": CONTRACT, "models": models,
                                "lineage": lineage, "selection_digest": selection["selection_digest"], "pairs": [],
                                "state": "imported_partial", "generating_configuration_sha256": content_hash(common)}
                require(manifest["models"] == models and manifest["lineage"] == lineage and
                        manifest["selection_digest"] == selection["selection_digest"], "MIXED_COLLECTION_POLICY")
                require(manifest["generating_configuration_sha256"] == content_hash(common), "MIXED_GENERATING_CONFIGURATION")
                identity = pair_id(case)
                existing = [e for e in manifest["pairs"] if e["pair_id"] == identity]
                if existing:
                    old_source, old_target, _ = load_pair(output, existing[0], models, allow_synthetic=generating.get("synthetic", False))
                    from .training_data import TENSORS
                    require(all(torch.equal(getattr(a, name), getattr(b, name)) for a, b in
                        ((source, old_source), (target, old_target)) for name in TENSORS), "SEMANTIC_CONTENT_COLLISION")
                    result["decision"], result["reason_codes"] = "REUSABLE", ["ALREADY_IMPORTED_VERIFIED"]
                else:
                    entry = write_pair(output, case, source, target, models, allow_synthetic=generating.get("synthetic", False))
                    entry["migration"] = {"source_cache": result["evidence"], "generating": generating,
                                          "conversion": conversion,
                                          "memory_selection": payload["metadata"].get("memory_selection"),
                                          "selected_records": [{"runtime_frame": int(source.frame_indices[0,0,i]),
                                               "official_frame": generating["frame_map"][int(source.frame_indices[0,0,i])]["official_id"],
                                               "stored_slot": int(source.slot_order[0,0,i])} for i in
                                               torch.where(source.validity[0,0])[0].tolist()],
                                          "legacy_old_slots": "UNKNOWN unless retained in memory_selection/run evidence"}
                    load_pair(output, entry, models, allow_synthetic=generating.get("synthetic", False))
                    manifest["pairs"].append(entry)
                    save_manifest(path, manifest)
                    result["reason_codes"] = ["IMPORTED_SAFE_TENSOR_DICT"]
                result["pair_id"] = identity
                result["ledger_event"] = str(append_event(output / "migration-ledger", result))
                results.append(result)
            except Rejection as exc:
                result["decision"], result["reason_codes"] = "PENDING_EVIDENCE", [exc.code]
                result["ledger_event"] = str(append_event(output / "migration-ledger", result))
                results.append(result)
        if manifest is not None:
            for label, role in (("fit", "fit"), ("development", "dev")):
                value = {**manifest, "role": role, "pair_mode": manifest["lineage"]["pair_mode"],
                         "pairs": [e for e in manifest["pairs"] if e["case"]["paired_split"] == label]}
                save_manifest(output / "splits" / f"{role}.json", value)
    return {"results": results, "counts": {d: sum(r["decision"] == d for r in results) for d in DECISIONS}}
