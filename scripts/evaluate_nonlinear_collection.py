"""Evaluate a trained nonlinear translator on internal validation videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from vos_memory_inspector.device import resolve_device
from vos_memory_inspector.roundtrip import run_cached_translator_handoff
from vos_memory_inspector.translators import (
    LearnedComponentPolicyTranslator,
    ResidualMLPStateTranslator,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate nonlinear handoff on the collection validation split."
    )
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--case-cache-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument(
        "--translator", choices=("residual_mlp", "direct"), default="residual_mlp"
    )
    parser.add_argument(
        "--component-policy",
        choices=("full", "spatial_pointer", "spatial_only"),
        default="full",
        help="Residual MLP components to use; remaining components use Direct Copy.",
    )
    parser.add_argument("--translator-artifact", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hot-cache-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(case: dict[str, Any]) -> str:
    return (
        f"{case['sequence']}_obj{int(case['object_id'])}_"
        f"switch{int(case['switch_frame'])}"
    )


def _stage_sequence(
    dataset_root: Path, hot_cache_root: Path | None, sequence: str
) -> Path:
    video = dataset_root / "JPEGImages" / "480p" / sequence
    if hot_cache_root is None:
        return video
    staged_video = hot_cache_root / "videos" / sequence
    shutil.copytree(video, staged_video, dirs_exist_ok=True, copy_function=shutil.copy2)
    return staged_video


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "cmmt.nonlinear_validation_aggregate.v3",
        "translator": rows[0].get("translator") if rows else None,
        "case_count": len(rows),
        "mean_candidate_wall_time_seconds": mean(
            float(row["handoff"]["resources_candidate_only"]["wall_time_seconds"])
            for row in rows
        )
        if rows
        else None,
        "cases": rows,
    }


def _write_aggregate(rows: list[dict[str, Any]], output_root: Path) -> None:
    aggregate = _aggregate_rows(rows)
    (output_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (output_root / "aggregate.md").write_text(
        "\n".join(
            [
                "# CMMT internal validation",
                "",
                f"- Cases: {aggregate['case_count']}",
                "- Mean candidate wall time (s): "
                f"{aggregate['mean_candidate_wall_time_seconds']}",
                "",
                "Handoff continuation only. This file does not score masks against ground truth.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = _parser().parse_args()
    requested_device = args.device
    translator_device = resolve_device(requested_device)
    sam2_device = resolve_device(requested_device, allow_mps=False)
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    cases = [
        case for case in selection["cases"] if case.get("paired_split") == "validation"
    ]
    if not cases:
        raise ValueError("selection has no internal validation cases")
    if args.translator == "residual_mlp":
        if args.translator_artifact is None:
            raise ValueError("--translator-artifact is required for residual_mlp")
        artifact_path = args.translator_artifact.resolve()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        payload = artifact.get("residual_mlp") if isinstance(artifact, dict) else None
        if not isinstance(payload, dict):
            raise ValueError("artifact has no residual_mlp payload")
        learned = (
            ResidualMLPStateTranslator.from_payload(payload).to(translator_device).eval()
        )
        artifact_sha = _sha256(artifact_path)
        if args.component_policy == "full":
            translator = learned
            evaluation_id = artifact_sha
        else:
            learned_components = (
                ("spatial_memory", "object_pointer")
                if args.component_policy == "spatial_pointer"
                else ("spatial_memory",)
            )
            translator = LearnedComponentPolicyTranslator(
                learned,
                learned_components=learned_components,
            )
            evaluation_id = f"{artifact_sha}:{args.component_policy}:v1"
    else:
        if args.translator_artifact is not None:
            raise ValueError("--translator-artifact is not used for direct")
        if args.component_policy != "full":
            raise ValueError("--component-policy is only used for residual_mlp")
        artifact_path = None
        translator = None
        evaluation_id = "direct-copy-v1"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hot_cache_root = (
        None if args.hot_cache_root is None else args.hot_cache_root.resolve()
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        slug = _slug(case)
        case_output = output_root / slug
        summary_path = case_output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_id = summary.get(
                "translator_evaluation_id",
                summary.get("translator_artifact_sha256"),
            )
            if summary_id == evaluation_id:
                summary.setdefault("translator", args.translator)
                summary.setdefault("translator_evaluation_id", evaluation_id)
                rows.append(summary)
                print(f"[{index}/{len(cases)}] {slug}: skipped_complete", flush=True)
                continue
        sequence = str(case["sequence"])
        video = _stage_sequence(
            args.dataset_root.resolve(), hot_cache_root, sequence
        )
        case_cache = args.case_cache_root.resolve() / "validation" / f"{slug}.pt"
        handoff = run_cached_translator_handoff(
            case_cache=case_cache,
            sam2_repo=args.sam2_repo,
            target_config_file=args.target_config,
            target_checkpoint=args.target_checkpoint,
            target_model_id=args.target_model_id,
            video_dir=video,
            device=sam2_device,
            seed=args.seed,
            translator=translator,
            translator_name=args.translator,
        )
        summary = {
            "case": case,
            "translator": (
                args.translator
                if args.translator == "direct"
                else f"{args.translator}:{args.component_policy}"
            ),
            "translator_artifact": (
                None if artifact_path is None else str(artifact_path)
            ),
            "translator_evaluation_id": evaluation_id,
            "handoff": handoff,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(summary)
        _write_aggregate(rows, output_root)
        print(f"[{index}/{len(cases)}] {slug}: completed", flush=True)
    _write_aggregate(rows, output_root)


if __name__ == "__main__":
    main()
