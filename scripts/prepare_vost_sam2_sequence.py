"""Expose a VOST sequence to SAM 2 without changing the original dataset.

SAM 2's JPEG loader accepts integer-stem filenames only, whereas VOST uses
names such as ``frame00102.jpg``.  This script creates a numeric symlink view
and records the reversible original-stem to SAM 2 index mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(dataset_root: Path, sequence: str, output: Path) -> dict:
    image_dir = dataset_root / "JPEGImages" / sequence
    annotation_dir = dataset_root / "Annotations" / sequence
    frames = sorted(path for path in image_dir.glob("*.jpg") if not path.name.startswith("."))
    if not frames or not annotation_dir.is_dir():
        raise FileNotFoundError(f"VOST sequence is incomplete: {sequence}")
    missing = [path.stem for path in frames if not (annotation_dir / f"{path.stem}.png").is_file()]
    if missing:
        raise ValueError(f"frames without annotations: {missing[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        target = output / f"{index:05d}.jpg"
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == frame.resolve():
                continue
            raise FileExistsError(f"refusing to replace unrelated staging file: {target}")
        target.symlink_to(frame.resolve())
    with Image.open(annotation_dir / f"{frames[0].stem}.png") as image:
        labels = np.asarray(image)
    object_ids = sorted(int(value) for value in np.unique(labels) if int(value) not in {0, 255})
    if not object_ids:
        raise ValueError(f"no promptable object in first VOST annotation: {sequence}")
    return {
        "schema_version": "cmmt.vost_sam2_staging.v1",
        "sequence": sequence,
        "source_frames_directory": str(image_dir.resolve()),
        "sam2_frames_directory": str(output.resolve()),
        "prompt_annotation": str((annotation_dir / f"{frames[0].stem}.png").resolve()),
        "prompt_object_ids": object_ids,
        "frame_count": len(frames),
        "original_stems": [path.stem for path in frames],
        "original_stem_to_sam2_index": {path.stem: index for index, path in enumerate(frames)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.dataset_root, args.sequence, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
