"""Validate the split-less PUMaVOS archive before external evaluation.

PUMaVOS is a benchmark-only dataset: it has no official train/validation/test
split.  The archive's paired ``JPEGImages/<sequence>`` and
``Annotations/<sequence>`` directories are therefore the source of truth for
the frozen external-evaluation inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def files_by_stem(path: Path, suffix: str) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    return {
        file.stem: file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() == suffix and not file.name.startswith(".")
    }


def sequence_names(root: Path) -> list[str]:
    names: set[str] = set()
    for directory in (root / "JPEGImages", root / "Annotations"):
        if directory.is_dir():
            names.update(child.name for child in directory.iterdir() if child.is_dir())
    return sorted(names)


def build_inventory(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failure_count = 0
    for sequence in sequence_names(root):
        frames = files_by_stem(root / "JPEGImages" / sequence, ".jpg")
        masks = files_by_stem(root / "Annotations" / sequence, ".png")
        missing_masks = sorted(set(frames) - set(masks))
        missing_frames = sorted(set(masks) - set(frames))
        valid = bool(frames) and not missing_masks and not missing_frames
        failure_count += int(not valid)
        rows.append(
            {
                "sequence": sequence,
                "frames_dir_exists": (root / "JPEGImages" / sequence).is_dir(),
                "annotations_dir_exists": (root / "Annotations" / sequence).is_dir(),
                "frame_count": len(frames),
                "annotation_count": len(masks),
                "missing_masks": missing_masks,
                "missing_frames": missing_frames,
                "valid": valid,
            }
        )
    return {
        "schema_version": "cmmt.pumavos_inventory.v1",
        "dataset": "PUMaVOS",
        "root": str(root),
        "official_split": None,
        "sequence_count": len(rows),
        "frame_count": sum(int(row["frame_count"]) for row in rows),
        "annotation_count": sum(int(row["annotation_count"]) for row in rows),
        "failure_count": failure_count,
        "sequences": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="PUMaVOS root containing JPEGImages and Annotations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sequence_count": inventory["sequence_count"],
                "frame_count": inventory["frame_count"],
                "annotation_count": inventory["annotation_count"],
                "failure_count": inventory["failure_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
