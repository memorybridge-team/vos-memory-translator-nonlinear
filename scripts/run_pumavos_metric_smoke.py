#!/usr/bin/env python3
"""Check the PUMaVOS J/F/J&F input contract with GT copied as predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    args = parser.parse_args()

    metrics_root = args.evaluator_root.resolve()
    sys.path.insert(0, str(metrics_root))
    from davis2017.metrics import db_eval_boundary, db_eval_iou

    ann_dir = args.dataset_root / "Annotations" / args.sequence
    paths = sorted(ann_dir.glob("*.png"))[: args.max_frames]
    if not paths:
        raise ValueError(f"no annotation masks for {args.sequence}")

    rows = []
    for path in paths:
        gt = np.asarray(Image.open(path)) == args.object_id
        # Intentional GT copy: validates only evaluator and label handling.
        prediction = gt.copy()
        j = float(db_eval_iou(gt, prediction))
        f = float(db_eval_boundary(gt, prediction))
        rows.append({"frame": path.stem, "J": j, "F": f, "J_and_F": (j + f) / 2})
    if any(row["J"] != 1.0 or row["F"] != 1.0 for row in rows):
        raise AssertionError("GT-copy smoke must be exactly J=F=1.0")

    metrics_sha256 = hashlib.sha256((metrics_root / "davis2017" / "metrics.py").read_bytes()).hexdigest()
    print(json.dumps({
        "schema_version": "cmmt.pumavos_metric_smoke.v1",
        "scope": "GT-copy evaluator contract only; not a model result",
        "metric_source": "davisvideochallenge/davis2017-evaluation local checkout",
        "metrics_py_sha256": metrics_sha256,
        "sequence": args.sequence,
        "object_id": args.object_id,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
