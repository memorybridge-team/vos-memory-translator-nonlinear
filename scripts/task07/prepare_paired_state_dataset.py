"""Collect future-GT-free Small/Base+ paired states for Task 07.

This is intentionally a local orchestration script.  It reads only the frozen
MOSEv2/LVOS v2 train manifests and their video-level fit/development split
manifests.  A mask is read solely at the object's recorded first-prompt frame;
no later annotation is supplied to either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--fit-split", required=True, type=Path)
    parser.add_argument("--development-split", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--network-volume-root", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--paired-split",
        choices=("all", "fit", "development"),
        default="all",
        help="collect one frozen paired split or both",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of deterministic independent collection shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based shard index to collect",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--max-frame-count",
        type=int,
        help="optional pilot/smoke filter; production runs should leave this unset",
    )
    parser.add_argument(
        "--dynamic-queue",
        action="store_true",
        help="claim cases atomically from a shared queue instead of static shards",
    )
    parser.add_argument(
        "--queue-root",
        type=Path,
        help="shared directory for dynamic-queue claim files (required with --dynamic-queue)",
    )
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--claim-lease-seconds",
        type=int,
        default=1800,
        help="reclaim a claim only after this many seconds without a heartbeat",
    )
    parser.add_argument(
        "--claim-heartbeat-seconds",
        type=int,
        default=30,
        help="how often an active worker refreshes its claim lease",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_checked(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_sha256", None)
    if digest is not None:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"content_sha256 mismatch: {path}")
        payload["content_sha256"] = digest
    return payload


def _dataset_layout(root: Path) -> tuple[Path, Path]:
    candidates = (root, root / "train", root / "extracted" / "train")
    for candidate in candidates:
        frames = candidate / "JPEGImages"
        masks = candidate / "Annotations"
        if frames.is_dir() and masks.is_dir():
            return frames, masks
    raise FileNotFoundError(
        "expected JPEGImages/ and Annotations/ under dataset root, train/, or extracted/train/"
    )


def _numeric_frame(directory: Path, frame: int, suffix: str) -> Path:
    matches = [path for path in directory.glob(f"*{suffix}") if path.stem.isdigit() and int(path.stem) == frame]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {suffix} frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def _frame_id_to_video_index(video: Path, frame_id: int) -> int:
    """Map a manifest frame ID to SAM 2's contiguous video-frame index.

    MOSEv2 uses consecutive numeric frame names, so its frame ID and index are
    usually identical.  LVOS v2 uses sparse numeric names; passing a raw LVOS
    ID to SAM 2 would select a wrong frame or go out of range.
    """
    numeric_ids = sorted(
        int(path.stem)
        for path in video.iterdir()
        if path.is_file() and path.stem.isdigit()
    )
    try:
        return numeric_ids.index(frame_id)
    except ValueError as error:
        raise FileNotFoundError(
            f"manifest switch frame ID {frame_id} is absent from {video}"
        ) from error


def _case_slug(case: dict[str, Any]) -> str:
    dataset = str(case["dataset"]).lower().replace(" ", "").replace("v2", "")
    return f"{dataset}_{case['video_id']}_obj{case['object_id']}_switch{case['switch_frame']}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a small run ledger without exposing a partially-written JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _cache_checksum_matches(cache: Path) -> bool:
    """Return True only when the cache and its SHA-256 sidecar agree."""

    checksum = cache.with_suffix(cache.suffix + ".sha256")
    if not cache.is_file() or not checksum.is_file():
        return False
    fields = checksum.read_text(encoding="ascii").split()
    if not fields:
        return False
    digest = hashlib.sha256()
    with cache.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == fields[0]


def _producer_revision() -> str:
    """Use an explicit deployment revision; never guess it from a live checkout."""

    return os.environ.get("CMMT_CODE_REVISION", "unknown")


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_checked(args.source_manifest.resolve())
    fit = _load_checked(args.fit_split.resolve())
    development = _load_checked(args.development_split.resolve())
    if source.get("dataset") not in {"MOSEv2", "LVOS v2"}:
        raise ValueError("Task 07 accepts only the MOSEv2 or LVOS v2 train manifest")
    if fit.get("source_manifest_content_sha256") != source.get("content_sha256"):
        raise ValueError("fit split does not belong to source manifest")
    if development.get("source_manifest_content_sha256") != source.get("content_sha256"):
        raise ValueError("development split does not belong to source manifest")
    fit_videos = set(map(str, fit["videos"]))
    development_videos = set(map(str, development["videos"]))
    if fit_videos & development_videos:
        raise ValueError("fit and development videos overlap")
    cases = []
    for case in source["cases"]:
        video_id = str(case["video_id"])
        split = "fit" if video_id in fit_videos else "development" if video_id in development_videos else None
        if split is not None:
            cases.append({**case, "paired_split": split})
    if args.paired_split != "all":
        cases = [case for case in cases if case["paired_split"] == args.paired_split]
    if args.max_frame_count is not None:
        if args.max_frame_count < 1:
            raise ValueError("--max-frame-count must be positive")
        cases = [
            case
            for case in cases
            if int(case.get("frame_count", case.get("frame_count_for_object", 0)))
            <= args.max_frame_count
        ]
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    if args.dynamic_queue:
        if args.queue_root is None:
            raise ValueError("--queue-root is required with --dynamic-queue")
        if args.claim_lease_seconds <= args.claim_heartbeat_seconds:
            raise ValueError("--claim-lease-seconds must exceed --claim-heartbeat-seconds")
        # Long cases are claimed first, reducing end-of-run tail latency. The
        # fallback keeps manifests without frame counts deterministic.
        cases.sort(
            key=lambda item: (
                -int(item.get("num_frames", 0)) - int(item.get("switch_frame", 0)),
                str(item["paired_split"]),
                str(item["case_id"]),
            )
        )
    else:
        cases.sort(key=lambda item: (item["paired_split"], item["case_id"]))
        cases = cases[args.shard_index :: args.shard_count]
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        cases = cases[: args.max_cases]
    selection: dict[str, Any] = {
        "schema_version": "cmmt.task07.paired_state_selection.v2",
        "dataset": source["dataset"],
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_content_sha256": source["content_sha256"],
        "fit_split": str(args.fit_split.resolve()),
        "development_split": str(args.development_split.resolve()),
        "future_gt_policy": "first_prompt_mask_only; no future annotation is read or passed to SAM 2",
        "paired_split_request": args.paired_split,
        "max_frame_count": args.max_frame_count,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "queue_mode": "dynamic_cost_descending" if args.dynamic_queue else "static_round_robin",
        "producer_revision": _producer_revision(),
        "state_policy": {
            "cache_mode": "state_only",
            "active_memory_only": True,
            "num_maskmem": 7,
            "max_obj_ptrs_in_encoder": 16,
        },
        "cases": cases,
        "split_case_counts": {
            "fit": sum(case["paired_split"] == "fit" for case in cases),
            "development": sum(case["paired_split"] == "development" for case in cases),
        },
    }
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":"))
    selection["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return selection


def _queue_contract(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    """Return the run contract every worker must share before claiming work."""

    payload = {
        "schema_version": "cmmt.task07.dynamic_queue_contract.v1",
        "dataset": selection["dataset"],
        "source_manifest_content_sha256": selection["source_manifest_content_sha256"],
        "paired_split_request": selection["paired_split_request"],
        "source_model_id": args.source_model_id,
        "target_model_id": args.target_model_id,
        "source_config": args.source_config,
        "target_config": args.target_config,
        "source_checkpoint_name": args.source_checkpoint.name,
        "target_checkpoint_name": args.target_checkpoint.name,
        "seed": args.seed,
        "state_policy": selection["state_policy"],
        "producer_revision": selection["producer_revision"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _ensure_queue_contract(queue_root: Path, contract: dict[str, Any]) -> None:
    """Create one immutable queue contract or reject a mismatched worker."""

    path = queue_root / "queue_contract.json"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") != contract["content_sha256"]:
            raise ValueError(
                "dynamic queue contract differs from the worker contract; use a separate queue root"
            )
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")


def _claim_case(
    queue_root: Path,
    slug: str,
    worker_id: str,
    *,
    reclaim_stale: bool,
    lease_seconds: int,
) -> Path | None:
    """Atomically claim one case on a shared filesystem.

    O_EXCL prevents two workers from running the same case concurrently. A
    heartbeat keeps valid long-running work fresh. An expired, non-heartbeating
    claim may be atomically moved aside and reclaimed on a later worker run.
    """

    claims = queue_root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / f"{slug}.claim"
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age_seconds = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            return None
        if not reclaim_stale or age_seconds <= lease_seconds:
            return None
        stale = claim.with_name(f"{claim.name}.stale-{uuid.uuid4().hex}")
        try:
            os.replace(claim, stale)
        except FileNotFoundError:
            return None
        try:
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"worker_id": worker_id, "claimed_at": _now()}) + "\n")
    return claim


def _refresh_claim_until_stopped(claim: Path, stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        try:
            os.utime(claim, None)
        except FileNotFoundError:
            return


def _prepare_command(
    args: argparse.Namespace,
    case: dict[str, Any],
    video: Path,
    prompt: Path,
    cache: Path,
    *,
    runtime_switch_index: int,
) -> list[str]:
    executable = str(Path(sys.executable).with_name("cmmt-sam2-prepare-case"))
    return [
        executable, "--sam2-repo", str(args.sam2_repo),
        "--source-config", args.source_config, "--source-checkpoint", str(args.source_checkpoint),
        "--source-model-id", args.source_model_id, "--target-config", args.target_config,
        "--target-checkpoint", str(args.target_checkpoint), "--target-model-id", args.target_model_id,
        "--video-dir", str(video), "--prompt-mask", str(prompt),
        "--object-id", str(case["object_id"]), "--switch-frame", str(runtime_switch_index),
        "--output", str(cache), "--report-json", str(cache.with_suffix(".prepare.json")),
        "--device", args.device, "--seed", str(args.seed), "--state-only",
        "--active-memory-only", "--num-maskmem", "7",
        "--max-obj-ptrs-in-encoder", "16",
    ]


def main() -> None:
    args = _parser().parse_args()
    selection = _selection(args)
    if args.dry_run:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return
    frames_root, masks_root = _dataset_layout(args.dataset_root.resolve())
    volume = args.network_volume_root.resolve()
    run_directory = args.run_directory.resolve()
    volume.mkdir(parents=True, exist_ok=True)
    if args.dynamic_queue:
        assert args.queue_root is not None
    worker_id = args.worker_id or f"pid-{os.getpid()}"
    if args.dynamic_queue:
        queue_root = args.queue_root.resolve()
        queue_root.mkdir(parents=True, exist_ok=True)
        _ensure_queue_contract(queue_root, _queue_contract(args, selection))
        # Separate ledgers avoid multiple workers overwriting one status file.
        run_directory = run_directory / worker_id
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(run_directory / "selection_manifest.json", selection)
    status: dict[str, Any] = {
        "schema_version": "cmmt.task07.paired_state_collection_status.v2",
        "state": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "producer_revision": selection["producer_revision"],
        "source_manifest_content_sha256": selection["source_manifest_content_sha256"],
        "paired_split_request": selection["paired_split_request"],
        "shard_count": selection["shard_count"],
        "shard_index": selection["shard_index"],
        "worker_id": worker_id,
        "queue_mode": selection["queue_mode"],
        "cases": {},
    }
    status_path = run_directory / "collection_status.json"
    for index, case in enumerate(selection["cases"], start=1):
        manifest_switch_frame = int(case["switch_frame"])
        runtime_switch_index: int | None = None
        cache = volume / str(case["paired_split"]) / f"{_case_slug(case)}.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        checksum = cache.with_suffix(".pt.sha256")
        claim: Path | None = None
        if _cache_checksum_matches(cache):
            state = "skipped_complete"
        elif args.dynamic_queue:
            claim = _claim_case(
                args.queue_root.resolve(),
                _case_slug(case),
                worker_id,
                reclaim_stale=True,
                lease_seconds=args.claim_lease_seconds,
            )
            if claim is None:
                status["cases"][_case_slug(case)] = {
                    "state": "claimed_by_other_worker",
                    "paired_split": case["paired_split"],
                    "updated_at": _now(),
                }
                status["updated_at"] = _now()
                _write_json_atomic(status_path, status)
                continue
            state = "claimed"
        else:
            state = "pending"

        # Do not touch video directories until this worker owns the case. This
        # avoids N workers repeatedly listing the same long-video directory.
        if state != "skipped_complete":
            video = frames_root / str(case["video_id"])
            annotations = masks_root / str(case["video_id"])
            prompt = _numeric_frame(annotations, int(case["first_prompt_frame"]), ".png")
            if not video.is_dir():
                raise FileNotFoundError(video)
            runtime_switch_index = _frame_id_to_video_index(video, manifest_switch_frame)

        if args.dynamic_queue and claim is not None:
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=_refresh_claim_until_stopped,
                args=(claim, heartbeat_stop, args.claim_heartbeat_seconds),
                daemon=True,
            )
            heartbeat.start()
            try:
                subprocess.run(
                    _prepare_command(
                        args,
                        case,
                        video,
                        prompt,
                        cache,
                        runtime_switch_index=runtime_switch_index,
                    ),
                    check=True,
                )
                if not _cache_checksum_matches(cache):
                    raise RuntimeError(f"collector finished without an intact cache checksum: {cache}")
                state = "completed"
            except Exception:
                status["cases"][_case_slug(case)] = {
                    "state": "failed_claim_retained",
                    "paired_split": case["paired_split"],
                    "updated_at": _now(),
                }
                status["updated_at"] = _now()
                _write_json_atomic(status_path, status)
                raise
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=args.claim_heartbeat_seconds + 1)
            if state == "completed":
                claim.unlink(missing_ok=True)
        elif state == "pending":
            subprocess.run(
                _prepare_command(
                    args,
                    case,
                    video,
                    prompt,
                    cache,
                    runtime_switch_index=runtime_switch_index,
                ),
                check=True,
            )
            state = "completed"
        status["cases"][_case_slug(case)] = {
            "state": state,
            "paired_split": case["paired_split"],
            "manifest_switch_frame_id": manifest_switch_frame,
            "runtime_switch_frame_index": runtime_switch_index,
            "cache_sha256": (
                checksum.read_text(encoding="ascii").split()[0]
                if checksum.is_file()
                else None
            ),
            "updated_at": _now(),
        }
        status["updated_at"] = _now()
        _write_json_atomic(status_path, status)
        print(f"[{index}/{len(selection['cases'])}] {_case_slug(case)}: {state}", flush=True)
    status["state"] = "completed"
    status["updated_at"] = _now()
    _write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
