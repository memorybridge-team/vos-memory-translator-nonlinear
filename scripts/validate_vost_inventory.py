"""Validate VOST split membership and annotation/frame correspondence.

This is an offline inventory check only. It does not read pixels or create a
training split; VOST remains an external zero-shot benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def files_by_stem(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    return {p.stem: p for p in path.iterdir() if p.is_file() and not p.name.startswith(".")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="extracted VOST root containing ImageSets")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    result: dict[str, object] = {"root": str(root), "splits": {}, "failure_count": 0}

    for split in ("train", "val", "test"):
        split_file = root / "ImageSets" / f"{split}.txt"
        sequences = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        rows = []
        for sequence in sequences:
            frames = files_by_stem(root / "JPEGImages" / sequence)
            masks = files_by_stem(root / "Annotations" / sequence)
            missing_masks = sorted(set(frames) - set(masks))
            missing_frames = sorted(set(masks) - set(frames))
            row = {
                "sequence": sequence,
                "frames_dir_exists": (root / "JPEGImages" / sequence).is_dir(),
                "annotations_dir_exists": (root / "Annotations" / sequence).is_dir(),
                "frame_count": len(frames),
                "annotation_count": len(masks),
                "missing_masks": missing_masks,
                "missing_frames": missing_frames,
            }
            rows.append(row)
            result["failure_count"] = int(result["failure_count"]) + bool(missing_masks or missing_frames)
        result["splits"][split] = {"sequence_count": len(sequences), "sequences": rows}

    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
