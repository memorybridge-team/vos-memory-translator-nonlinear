"""Train the nonlinear translator from a completed paired-state collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vos_memory_inspector.paired_experiment import run_paired_experiment
from vos_memory_inspector.paired_state_cache import load_paired_state_cache
from vos_memory_inspector.translator_artifact import (
    LEARNED_ARTIFACT_TRANSLATORS, TRANSFORMER_TRANSLATORS,
)
from vos_memory_inspector.translator_training import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the video-level split from a paired-state selection manifest "
            "and train a residual MLP or spatial Transformer."
        )
    )
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--translator", choices=LEARNED_ARTIFACT_TRANSLATORS, default="residual_mlp")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pointer-hidden-dim", type=int, default=512)
    spatial = parser.add_mutually_exclusive_group()
    spatial.add_argument(
        "--spatial-samples-per-pair", type=int,
        help="Opt in to spatial subsampling; default uses all valid positions.",
    )
    spatial.add_argument(
        "--whole-frames", action="store_true",
        help="Use every valid spatial position (default).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _case_slug(case: dict[str, Any]) -> str:
    return (
        f"{case['sequence']}_obj{int(case['object_id'])}_"
        f"switch{int(case['switch_frame'])}"
    )


def _split_paths(
    selection: dict[str, Any], cache_root: Path
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"train": [], "validation": []}
    video_splits: dict[str, str] = {}
    seen: set[Path] = set()
    for case in selection["cases"]:
        split = str(case.get("paired_split"))
        if split not in result:
            raise ValueError(f"case has unsupported paired_split: {split!r}")
        video = str(case["sequence"])
        if video in video_splits and video_splits[video] != split:
            raise ValueError(f"video occurs in both train and validation: {video}")
        video_splits[video] = split
        path = cache_root / split / f"{_case_slug(case)}.pt"
        if path in seen:
            raise ValueError(f"duplicate paired case: {path}")
        seen.add(path)
        checksum = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not checksum.is_file():
            raise FileNotFoundError(f"paired case is incomplete: {path}")
        result[split].append(path)
    if not result["train"] or not result["validation"]:
        raise ValueError("selection needs non-empty train and validation cases")
    return result


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.translator in TRANSFORMER_TRANSLATORS and args.spatial_samples_per_pair is not None:
        parser.error("Transformer translators require whole frames; remove --spatial-samples-per-pair")
    selection_path = args.selection_manifest.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "cmmt.paired_state_selection.v1":
        raise ValueError("unsupported paired-state selection manifest")
    paths = _split_paths(selection, args.cache_root.resolve())
    train_pairs = [load_paired_state_cache(path)[:2] for path in paths["train"]]
    validation_pairs = [
        load_paired_state_cache(path)[:2] for path in paths["validation"]
    ]
    report = run_paired_experiment(
        train_pairs,
        None,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        pointer_hidden_dim=args.pointer_hidden_dim,
        translator_names=(args.translator,),
        device=args.device,
        spatial_samples_per_pair=(None if args.whole_frames else args.spatial_samples_per_pair),
        validation_pairs=validation_pairs,
    )
    report["input_collection"] = {
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "train": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths["train"]
        ],
        "validation": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths["validation"]
        ],
    }
    report_path = args.output_dir.resolve() / "paired_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
