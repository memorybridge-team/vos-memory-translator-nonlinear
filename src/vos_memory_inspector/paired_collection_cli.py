"""ZIP 객체별 수집 경로의 최소 repair·CPU audit/import 진입점."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import hashlib
import torch
import numpy as np
import PIL

from .collection_contract import (SEMANTICS, POLICIES, Rejection, require, jpeg_map,
    resolve_indices, frozen_selection, shard_plan, memory_policy, validate_generating)
from .cache_migration import audit_requests, import_requests, inspect_cache, append_event
from .training_data import read_manifest, save_manifest, load_pair
from .training_storage import write_json, content_hash, sha256, ExclusiveWriter
from .training_collection import CONFIGS, model_provenance


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_snapshot():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "pillow": PIL.__version__, "cuda": torch.version.cuda,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def parser():
    p = argparse.ArgumentParser(description="ZIP 단일 객체 paired-state 수집 repair; 기본은 CPU 계획/검증")
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="기존 frozen membership을 그대로 읽어 global selection 생성")
    for flag in ("source-manifest", "fit-split", "development-split", "output"):
        plan.add_argument(f"--{flag}", required=True, type=Path)
    plan.add_argument("--paired-split", choices=("all", "fit", "development"), default="all")
    plan.add_argument("--case-id", action="append", help="새 제한 smoke selection에만 사용; 기존 실행 선택을 변경하지 않음")
    shards = sub.add_parser("shard-plan", help="고정 global selection의 modulo disjoint/full-union 증명")
    shards.add_argument("--plan", required=True, type=Path)
    shards.add_argument("--shard-count", type=int, default=8)
    shards.add_argument("--output", required=True, type=Path)
    for name in ("audit", "import"):
        item = sub.add_parser(name, help="신뢰한 팀 legacy cache 소수 sample의 CPU 검사" if name == "audit" else "원본 보존 safe tensor/dict 변환")
        item.add_argument("--requests", required=True, type=Path, help="검사할 명시적 완료 파일 목록 JSON; recursive scan 없음")
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--trusted-team-legacy", action="store_true", help="CanonicalState pickle 실행을 명시적으로 신뢰")
        item.add_argument("--stable-seconds", type=float, default=60)
        if name == "import":
            item.add_argument("--plan", required=True, type=Path)
            item.add_argument("--target-policy", choices=sorted(POLICIES))
    verify = sub.add_parser("verify", help="imported collection의 weights_only=True shard 검사")
    verify.add_argument("--collection", required=True, type=Path)
    collect = sub.add_parser("collect", help="ZIP 단일 객체 prefix 경로; 새 namespace에서만 실행")
    for flag in ("plan", "dataset-root", "sam2-repo", "source-checkpoint", "target-checkpoint",
                 "read-policy-json", "cache-root", "run-root"):
        collect.add_argument(f"--{flag}", required=True, type=Path)
    collect.add_argument("--shard-count", type=int, default=8)
    collect.add_argument("--shard-index", type=int, required=True)
    collect.add_argument("--memory-policy", choices=sorted(POLICIES), default="active_window_v1")
    collect.add_argument("--num-maskmem", type=int, default=7)
    collect.add_argument("--max-obj-ptrs-in-encoder", type=int, default=16)
    collect.add_argument("--seed", type=int, default=7)
    collect.add_argument("--device", default="cuda:0", help="visible GPU remapping 후 worker 내부 device")
    collect.add_argument("--max-cases", type=int, help="승인된 제한 smoke의 worker별 상한")
    collect.add_argument("--worklist", type=Path, help="audit의 confirmed recollection case IDs; global assignment는 그대로")
    collect.add_argument("--case-timeout-seconds", type=float, default=900, help="한 CLI subprocess의 wall 상한")
    collect.add_argument("--dry-run", action="store_true", help="frame-map/provenance/skip 검사만; SAM 모델 실행 없음")
    return p


def dataset_layout(root):
    for candidate in (root, root / "train", root / "extracted" / "train"):
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate / "JPEGImages", candidate / "Annotations"
    raise Rejection("DATASET_LAYOUT", str(root))


def numeric_mask(directory, official_id):
    masks = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png" and
             p.stem.isascii() and p.stem.isdigit() and int(p.stem) == official_id]
    require(len(masks) == 1, "PROMPT_MASK_MISSING_DUPLICATE", str(official_id))
    return masks[0]


def build_generating(raw, video, annotation, selection, models, *, policy, read_policy, seed, sam2_repo):
    _, records = jpeg_map(video)
    prompt, switch = resolve_indices(raw, records)
    official_prompt = records[prompt]["official_id"]
    mask = numeric_mask(annotation, official_prompt)
    obj = int(raw["object_id"])
    from .runner import load_binary_prompt
    load_binary_prompt(mask, obj)  # 첫 prompt만 CPU 검증; 미래 annotation은 열지 않는다.
    case = {"dataset": raw["dataset"], "release": raw["release"], "official_split": "train",
            "video_id": raw["video_id"], "object_ids": [obj], "switch_frame": switch,
            "pair_mode": "native_history", "prompt_conditions": [{"frame_index": prompt, "object_id": obj,
               "kind": "mask", "sha256": sha256(mask)}], "preprocessing": {
                   "image_size": 1024, "precision": "float32_no_autocast", "offload_video_to_cpu": True,
                   "offload_state_to_cpu": True}, "video_sha256": content_hash(records),
            "frame_map_sha256": content_hash(records), "num_frames": len(records),
            "case_id": raw["case_id"], "official_prompt_frame": raw["first_prompt_frame"],
            "official_switch_frame": raw["switch_frame"], "paired_split": raw["paired_split"],
            "source_manifest_content_sha256": selection["source_manifest_content_sha256"],
            "object_semantics": SEMANTICS, "memory_policy": policy}
    files = [Path(__file__), Path(__file__).with_name("roundtrip.py"),
             Path(__file__).with_name("sam2_state.py"), Path(__file__).with_name("case_cache.py"),
             Path(__file__).with_name("collection_contract.py"), Path(__file__).with_name("state_schema.py")]
    upstream_files = [sam2_repo / "sam2" / "sam2_video_predictor.py", sam2_repo / "sam2" / "modeling" / "sam2_base.py",
                      sam2_repo/"sam2"/"build_sam.py", sam2_repo/"sam2"/"utils"/"misc.py"]
    generating = {"schema_version": "cmmt.paired_generating.v1", "case": case, "models": models,
                  "frame_map": records, "seed": seed, "collection_mode": "state_only",
                  "collector_source_hashes": {f.name: sha256(f) for f in files + upstream_files},
                  "effective_model_policy": read_policy, "preprocessing": case["preprocessing"],
                  "environment": environment_snapshot()}
    validate_generating(generating)
    return generating, mask


def stage_video(video, records, root):
    """새 checkout의 staging view만 생성. 원본 이름·index map은 generating에 보존."""
    destination = root / "staging" / content_hash(records)
    mapping_path = destination / "frame-map.json"
    if mapping_path.exists():
        require(json_read(mapping_path) == records, "STAGING_MAP_CHANGED")
    else:
        require(not destination.exists(), "PARTIAL_STAGING", "새 run-root로 재시도; 임의 cleanup 금지")
        destination.mkdir(parents=True)
        for record in records:
            original = (video / record["filename"]).resolve()
            (destination / f"{record['runtime_index']:06d}.jpg").symlink_to(original)
        write_json(mapping_path, records)
    for record in records:
        staged = destination / f"{record['runtime_index']:06d}.jpg"
        require(staged.is_file() and sha256(staged) == record["sha256"], "STAGING_CONTENT_CHANGED")
    return destination


def prepare_command(generating, args, video, mask, cache, condition_path):
    case = generating["case"]
    command = [sys.executable, "-c", "from vos_memory_inspector.cli import prepare_handoff_case_main; prepare_handoff_case_main()",
        "--sam2-repo", str(args.sam2_repo), "--source-config", CONFIGS["source"],
        "--source-checkpoint", str(args.source_checkpoint), "--source-model-id", "sam2.1_small",
        "--target-config", CONFIGS["target"], "--target-checkpoint", str(args.target_checkpoint),
        "--target-model-id", "sam2.1_base_plus", "--video-dir", str(video), "--prompt-mask", str(mask),
        "--object-id", str(case["object_ids"][0]), "--prompt-frame-index", str(case["prompt_conditions"][0]["frame_index"]),
        "--switch-frame", str(case["switch_frame"]), "--output", str(cache),
        "--generating-json", str(condition_path), "--state-only", "--device", args.device, "--seed", str(args.seed),
        "--num-maskmem", str(args.num_maskmem), "--max-obj-ptrs-in-encoder", str(args.max_obj_ptrs_in_encoder)]
    if args.memory_policy == "active_window_v1":
        command.append("--active-memory-only")
    return command


def collect(args):
    selection = read_manifest(args.plan)
    assignments = shard_plan(selection, args.shard_count)
    require(0 <= args.shard_index < args.shard_count, "SHARD_INDEX")
    require(args.case_timeout_seconds > 0, "CASE_TIMEOUT")
    if not args.dry_run:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        require(visible and "," not in visible and visible != "-1" and
                args.device in {"cuda", "cuda:0"}, "ONE_VISIBLE_GPU_REQUIRED")
        require(torch.cuda.device_count() == 1, "ONE_VISIBLE_GPU_REQUIRED")
    keys = set(assignments["shards"][args.shard_index]["case_ids"])
    if args.worklist:
        work = set(json_read(args.worklist)["recollection_case_ids"])
        require(work <= {c["case_id"] for c in selection["cases"]}, "WORKLIST_SELECTION")
        keys &= work
    cases = [c for c in selection["cases"] if c["case_id"] in keys]
    if args.max_cases is not None:
        require(args.max_cases > 0, "CASE_LIMIT")
        cases = cases[:args.max_cases]
    frames, annotations = dataset_layout(args.dataset_root.resolve())
    models = model_provenance(Path("/"), {"models": {role: {"checkpoint": str(getattr(args, f"{role}_checkpoint").resolve()),
                               "config": CONFIGS[role]} for role in ("source", "target")}}, args.sam2_repo.resolve())
    for relative in ("sam2/sam2_video_predictor.py", "sam2/modeling/sam2_base.py", *["sam2/"+c for c in CONFIGS.values()]):
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=args.sam2_repo,
                                   check=True, capture_output=True).stdout
        require(hashlib.sha256(committed).hexdigest() == sha256(args.sam2_repo/relative),
                "DIRTY_PINNED_SAM_SOURCE", relative)
    status = subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "sam2"], cwd=args.sam2_repo,
                            capture_output=True)
    require(status.returncode == 0, "DIRTY_PINNED_SAM_SOURCE", "sam2 tracked source는 pinned HEAD와 같아야 함")
    policy = memory_policy(args.memory_policy, args.num_maskmem, args.max_obj_ptrs_in_encoder)
    read_policy = json_read(args.read_policy_json)
    run_identity = {"selection_digest": selection["selection_digest"], "models": models, "policy": policy,
                    "seed": args.seed, "read_policy": read_policy, "shard_count": args.shard_count,
                    "collector_source_hashes": {name: sha256(Path(__file__).with_name(name)) for name in
                    ("paired_collection_cli.py", "collection_contract.py", "roundtrip.py", "case_cache.py", "sam2_state.py", "state_schema.py")},
                    "environment": environment_snapshot()}
    namespace = content_hash(run_identity)
    root = args.run_root.resolve() / namespace / f"shard-{args.shard_index:02d}"
    cache_root = args.cache_root.resolve() / namespace / f"shard-{args.shard_index:02d}"
    summary = {"namespace": namespace, "shard_index": args.shard_index, "case_count": len(cases),
               "selection_digest": selection["selection_digest"], "results": [], "dry_run": args.dry_run,
               "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    with ExclusiveWriter(root):
        identity_file = root / "run-identity.json"
        if identity_file.exists():
            require(json_read(identity_file) == run_identity, "RUN_IDENTITY_CHANGED")
        else:
            write_json(identity_file, run_identity)
        binding_file = root / "case-bindings.json"
        bindings = json_read(binding_file) if binding_file.exists() else {}
        for raw in cases:
            started = time.perf_counter()
            try:
                generating, mask = build_generating(raw, frames / raw["video_id"], annotations / raw["video_id"],
                        selection, models, policy=policy, read_policy=read_policy, seed=args.seed, sam2_repo=args.sam2_repo.resolve())
                identity = content_hash(generating)
                if raw["case_id"] in bindings:
                    require(bindings[raw["case_id"]] == identity, "CASE_INPUT_CHANGED", raw["case_id"])
                else:
                    bindings[raw["case_id"]] = identity
                    write_json(binding_file, bindings)
                cache = cache_root / f"{identity}.pt"
                # 기존 namespace에 파일이 하나라도 있으면 실제 검증 없이는 skip하지 않는다.
                if cache.exists() or cache.with_suffix(".pt.sha256").exists():
                    result, _, _ = inspect_cache({"path": str(cache), "expected_generating": generating},
                              trusted_team_legacy=True, stable_seconds=0)
                    require(result["decision"] == "CONVERTIBLE", "RESUME_REJECTED", str(result))
                    record = {"case_id": raw["case_id"], "state": "verified_complete", "cache": str(cache)}
                elif args.dry_run:
                    record = {"case_id": raw["case_id"], "state": "dry_run_ready", "generating": generating,
                              "cache": str(cache)}
                else:
                    staged = stage_video(frames / raw["video_id"], generating["frame_map"], root)
                    conditions = root / "conditions" / f"{identity}.json"
                    write_json(conditions, generating)
                    command = prepare_command(generating, args, staged, mask, cache, conditions)
                    log = root / "logs" / f"{identity}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("ab") as stream:
                        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.case_timeout_seconds)
                    result, _, _ = inspect_cache({"path": str(cache), "expected_generating": generating},
                                                  trusted_team_legacy=True, stable_seconds=0)
                    require(result["decision"] == "CONVERTIBLE", "COMPLETION_REJECTED", str(result))
                    record = {"case_id": raw["case_id"], "state": "completed", "cache": str(cache),
                              "generating_sha256": identity, "cache_sha256": result["evidence"]["cache_sha256"]}
            except (Rejection, OSError, subprocess.SubprocessError) as exc:
                record = {"case_id": raw["case_id"], "state": "rejected", "reason": str(exc)}
            record["wall_time_seconds"] = time.perf_counter()-started
            # event가 먼저 durable, status는 복구 가능한 projection이다.
            append_event(root / "events-ledger", record)
            summary["results"].append(record)
            write_json(root / "status.json", summary)
        save_manifest(root / "worker-manifest.json", {**summary, "run_identity": run_identity})
    require(all(r["state"] != "rejected" for r in summary["results"]), "WORKER_REJECTIONS", str(root / "status.json"))
    return {"run_root": str(root), "cache_root": str(cache_root),
            "case_count": len(cases), "selection_digest": selection["selection_digest"], "dry_run": args.dry_run}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        selection = frozen_selection(args.source_manifest, args.fit_split, args.development_split,
                                    paired_split=args.paired_split, case_ids=args.case_id)
        require(not args.output.exists(), "PLAN_ALREADY_EXISTS", "동결된 selection을 덮어쓰지 않음")
        save_manifest(args.output, selection)
        result = {"output": str(args.output), "case_count": len(selection["cases"]),
                  "selection_digest": selection["selection_digest"]}
    elif args.command == "shard-plan":
        result = shard_plan(read_manifest(args.plan), args.shard_count)
        write_json(args.output, result)
        result = {"output": str(args.output), "selection_digest": result["selection_digest"],
                  "coverage_exact": True, "case_counts": [s["case_count"] for s in result["shards"]]}
    elif args.command in {"audit", "import"}:
        requests = json_read(args.requests)["requests"]
        if args.command == "audit":
            result = audit_requests(requests, args.output / "ledger", trusted_team_legacy=args.trusted_team_legacy,
                                    stable_seconds=args.stable_seconds)
            write_json(args.output / "reuse-plan.json", result)
        else:
            result = import_requests(requests, args.output, read_manifest(args.plan),
                    trusted_team_legacy=args.trusted_team_legacy, stable_seconds=args.stable_seconds,
                    target_policy=args.target_policy)
            write_json(args.output / "import-report.json", result)
    elif args.command == "verify":
        manifest = read_manifest(args.collection / "manifest.json")
        for entry in manifest["pairs"]:
            load_pair(args.collection, entry, manifest["models"])
        result = {"verified_pairs": len(manifest["pairs"]), "manifest_sha256": sha256(args.collection / "manifest.json")}
    else:
        result = collect(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
