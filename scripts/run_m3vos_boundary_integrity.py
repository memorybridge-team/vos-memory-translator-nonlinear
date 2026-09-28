#!/usr/bin/env python3
"""Record a non-primary boundary F/J&F integrity check for M³-VOS.

M³-VOS reports J/J_last/J_cc. This utility never promotes F/J&F to an
official metric; it checks only PNG serialization and expected sensitivity to
resize or one-pixel morphology on a GT-copy control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _score(db_eval_iou, db_eval_boundary, gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    j = float(db_eval_iou(gt, prediction))
    f = float(db_eval_boundary(gt, prediction))
    return {"J": j, "F": f, "J_and_F": (j + f) / 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.evaluator_root))
    from source.metrics import db_eval_boundary, db_eval_iou

    annotation_dir = args.dataset_root / "Annotations" / args.sequence
    paths = sorted(annotation_dir.glob("*.png"))[: args.max_frames]
    if not paths:
        raise ValueError(f"no annotations for {args.sequence}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    kernel = np.ones((3, 3), np.uint8)
    for path in paths:
        gt = np.asarray(Image.open(path)) == args.object_id
        export = args.scratch / f"{path.stem}.png"
        Image.fromarray(np.where(gt, args.object_id, 0).astype(np.uint8)).save(export)
        reloaded = np.asarray(Image.open(export)) == args.object_id
        if not np.array_equal(gt, reloaded):
            raise AssertionError(f"PNG export/reload changed {path.name}")
        half = cv2.resize(gt.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        resize_roundtrip = cv2.resize(half, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        dilated = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1).astype(bool)
        eroded = cv2.erode(gt.astype(np.uint8), kernel, iterations=1).astype(bool)
        rows.append(
            {
                "frame": path.stem,
                "gt_copy": _score(db_eval_iou, db_eval_boundary, gt, reloaded),
                "resize_roundtrip": _score(db_eval_iou, db_eval_boundary, gt, resize_roundtrip),
                "one_pixel_dilation": _score(db_eval_iou, db_eval_boundary, gt, dilated),
                "one_pixel_erosion": _score(db_eval_iou, db_eval_boundary, gt, eroded),
            }
        )
    if any(row["gt_copy"]["J"] != 1.0 or row["gt_copy"]["F"] != 1.0 for row in rows):
        raise AssertionError("GT-copy boundary smoke must be exactly J=F=1.0")
    print(json.dumps({
        "schema_version": "cmmt.m3vos_boundary_integrity.v1",
        "scope": "non-official engineering integrity check; not a CMMT model result",
        "sequence": args.sequence,
        "object_id": args.object_id,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
