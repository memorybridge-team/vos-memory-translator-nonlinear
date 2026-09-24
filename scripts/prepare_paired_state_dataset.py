"""Prepare resumable source/target SAM 2 paired-state caches without baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vos_memory_inspector.case_cache import load_case_cache
from vos_memory_inspector.paired_state_cache import write_paired_state_cache

RARE_EVENT_TAGS = frozenset(
    {
        "full_occlusion_entry",
        "reappearance",
        "strong_area_drop",
        "strong_area_growth",
    }
)
_DAVIS_MANIFEST_SCHEMA = "cmmt.davis_evaluation_manifest.v1"


def case_slug(case: Mapping[str, Any]) -> str:
    return (
        f"{case['sequence']}_obj{int(case['object_id'])}_"
        f"switch{int(case['switch_frame'])}"
    )


def select_cases(
    manifest: Mapping[str, Any],
    *,
    tags: Iterable[str] = RARE_EVENT_TAGS,
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected_tags = frozenset(tags)
    requested_ids = None if case_ids is None else frozenset(case_ids)
    cases = []
    seen: set[str] = set()
    for raw_case in manifest["cases"]:
        case = dict(raw_case)
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id in manifest: {case_id}")
        seen.add(case_id)
        if requested_ids is not None:
            include = case_id in requested_ids
        else:
            include = bool(selected_tags.intersection(case["tags"]))
        if include:
            cases.append(case)
    if requested_ids is not None:
        missing = sorted(requested_ids - seen)
        if missing:
            raise ValueError(f"requested case ids are absent from manifest: {missing}")
    return cases


def selection_manifest(
    source_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    tags: Iterable[str],
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    policy = (
        {"match": "explicit_case_ids", "case_ids": sorted(set(case_ids))}
        if case_ids is not None
        else {"match": "any_tag", "tags": sorted(set(tags))}
    )
    payload: dict[str, Any] = {
        "schema_version": "cmmt.paired_state_selection.v1",
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "selection_policy": policy,
        "case_count": len(cases),
        "cases": [dict(case) for case in cases],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_json_atomic(payload: Mapping[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(output)


def cache_checksum_matches(path: str | Path) -> bool:
    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not checksum.is_file():
        return False
    tokens = checksum.read_text(encoding="ascii").split()
    if not tokens:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == tokens[0]


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _DAVIS_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported evaluation manifest: {manifest.get('schema_version')!r}"
        )
    digest = manifest.pop("content_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest["content_sha256"] = digest
    if digest != expected:
        raise ValueError("evaluation manifest content_sha256 does not match its content")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Source/Target canonical-state pairs from a DAVIS manifest. "
            "Cases are split by sequence, never by frame or object."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--compact-root", type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--hot-cache-root", type=Path)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_sequences(
    cases: list[dict[str, Any]], *, seed: int, validation_fraction: float
) -> tuple[list[str], list[str]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    sequences = sorted({str(case["sequence"]) for case in cases})
    if len(sequences) < 2:
        raise ValueError("at least two sequences are required for a video-level split")
    ranked = sorted(
        sequences,
        key=lambda sequence: hashlib.sha256(
            f"{seed}:{sequence}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    validation = sorted(ranked[:validation_count])
    train = sorted(set(ranked) - set(validation))
    return train, validation


def _stage_sequence(
    dataset_root: Path, hot_cache_root: Path | None, sequence: str
) -> tuple[Path, Path]:
    video = dataset_root / "JPEGImages" / "480p" / sequence
    annotation = dataset_root / "Annotations" / "480p" / sequence
    if not video.is_dir() or not annotation.is_dir():
        raise FileNotFoundError(f"DAVIS sequence is incomplete: {sequence}")
    if hot_cache_root is None:
        return video, annotation
    staged_video = hot_cache_root / "videos" / sequence
    staged_annotation = hot_cache_root / "annotations" / sequence
    shutil.copytree(video, staged_video, dirs_exist_ok=True, copy_function=shutil.copy2)
    shutil.copytree(
        annotation, staged_annotation, dirs_exist_ok=True, copy_function=shutil.copy2
    )
    return staged_video, staged_annotation


def _prepare_command(
    args: argparse.Namespace,
    *,
    case: dict[str, Any],
    video: Path,
    annotation: Path,
    cache: Path,
) -> list[str]:
    executable = str(Path(sys.executable).with_name("cmmt-sam2-prepare-case"))
    return [
        executable,
        "--sam2-repo",
        str(args.sam2_repo),
        "--source-config",
        args.source_config,
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--source-model-id",
        args.source_model_id,
        "--target-config",
        args.target_config,
        "--target-checkpoint",
        str(args.target_checkpoint),
        "--target-model-id",
        args.target_model_id,
        "--video-dir",
        str(video),
        "--prompt-mask",
        str(annotation / "00000.png"),
        "--object-id",
        str(case["object_id"]),
        "--switch-frame",
        str(case["switch_frame"]),
        "--output",
        str(cache),
        "--report-json",
        str(cache.with_suffix(".prepare.json")),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = _parser().parse_args()
    manifest = load_evaluation_manifest(args.manifest)
    tags = tuple(args.tag) or RARE_EVENT_TAGS
    cases = select_cases(manifest, tags=tags, case_ids=args.case_id)
    if not cases:
        raise ValueError("paired-state selection is empty")
    train_sequences, validation_sequences = _split_sequences(
        cases, seed=args.seed, validation_fraction=args.validation_fraction
    )
    split_by_sequence = {
        sequence: "validation" if sequence in validation_sequences else "train"
        for sequence in train_sequences + validation_sequences
    }
    selection = selection_manifest(
        manifest, cases, tags=tags, case_ids=args.case_id
    )
    selection["schema_version"] = "cmmt.paired_state_selection.v1"
    selection.pop("content_sha256", None)
    selection["video_level_split"] = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
    }
    selection["cases"] = [
        {**case, "paired_split": split_by_sequence[str(case["sequence"])]}
        for case in selection["cases"]
    ]
    selection["split_case_counts"] = {
        "train": sum(case["paired_split"] == "train" for case in selection["cases"]),
        "validation": sum(
            case["paired_split"] == "validation" for case in selection["cases"]
        ),
    }
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":"))
    selection["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return

    output_root = args.output_root.resolve()
    compact_root = None if args.compact_root is None else args.compact_root.resolve()
    run_directory = args.run_directory.resolve()
    hot_cache_root = (
        None if args.hot_cache_root is None else args.hot_cache_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(selection, run_directory / "selection_manifest.json")
    status: dict[str, Any] = {
        "schema_version": "cmmt.paired_state_collection_status.v1",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "state": "running",
        "planned_cases": len(cases),
        "cases": {},
    }
    status_path = run_directory / "collection_status.json"
    write_json_atomic(status, status_path)

    for index, case in enumerate(cases, start=1):
        slug = case_slug(case)
        split = split_by_sequence[str(case["sequence"])]
        cache = output_root / split / f"{slug}.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache_checksum_matches(cache):
            state = "skipped_complete"
        else:
            video, annotation = _stage_sequence(
                args.dataset_root.resolve(), hot_cache_root, str(case["sequence"])
            )
            subprocess.run(
                _prepare_command(
                    args,
                    case=case,
                    video=video,
                    annotation=annotation,
                    cache=cache,
                ),
                check=True,
            )
            state = "completed"
        compact_cache = None
        if compact_root is not None:
            compact_cache = compact_root / split / f"{slug}.pt"
            if not cache_checksum_matches(compact_cache):
                full_payload = load_case_cache(cache)
                write_paired_state_cache(
                    compact_cache,
                    source_canonical=full_payload["source_canonical"],
                    target_canonical=full_payload["target_canonical"],
                    metadata={
                        **dict(full_payload["metadata"]),
                        "source_case_cache": str(cache),
                    },
                )
        status["cases"][slug] = {
            "state": state,
            "paired_split": split,
            "compact_cache": None if compact_cache is None else str(compact_cache),
            "updated_at": _utc_now(),
        }
        status["updated_at"] = _utc_now()
        write_json_atomic(status, status_path)
        print(f"[{index}/{len(cases)}] {slug}: {state} ({split})", flush=True)

    status["state"] = "completed"
    status["updated_at"] = _utc_now()
    write_json_atomic(status, status_path)


if __name__ == "__main__":
    main()
