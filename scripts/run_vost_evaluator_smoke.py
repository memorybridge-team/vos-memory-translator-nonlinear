"""Run the official VOST evaluator on one ground-truth-as-prediction smoke case.

The check proves the evaluator path, mask names, and metric output contract. It
does not measure CMMT performance and must never be reported as a model score.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--python", default="python3", help="interpreter with evaluator dependencies")
    parser.add_argument("--resize-max", type=int, help="optional smoke-only maximum image side")
    args = parser.parse_args()

    sequences = [x.strip() for x in (args.dataset_root / "ImageSets" / "val.txt").read_text().splitlines() if x.strip()]
    sequence = args.sequence or sequences[0]
    if sequence not in sequences:
        raise ValueError(f"not a VOST val sequence: {sequence}")

    dataset = args.workdir / "dataset"
    results = args.workdir / "results"
    for path in (dataset / "ImageSets", dataset / "JPEGImages", dataset / "Annotations", results):
        path.mkdir(parents=True, exist_ok=True)
    (dataset / "ImageSets" / "val.txt").write_text(sequence + "\n")
    if args.resize_max:
        for name, interpolation in (("JPEGImages", Image.Resampling.BILINEAR), ("Annotations", Image.Resampling.NEAREST)):
            source_dir = args.dataset_root / name / sequence
            target_dir = dataset / name / sequence
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in source_dir.glob("*.*"):
                if source.name.startswith("."):
                    continue
                with Image.open(source) as image:
                    scale = args.resize_max / max(image.size)
                    size = tuple(max(1, round(side * min(scale, 1.0))) for side in image.size)
                    image.resize(size, interpolation).save(target_dir / source.name)
        shutil.copytree(dataset / "Annotations" / sequence, results / sequence, dirs_exist_ok=True)
    else:
        for name in ("JPEGImages", "Annotations"):
            target = dataset / name / sequence
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(args.dataset_root / name / sequence, target_is_directory=True)
        shutil.copytree(args.dataset_root / "Annotations" / sequence, results / sequence, dirs_exist_ok=True)

    completed = subprocess.run(
        [args.python, "evaluation_method.py", "--dataset_path", str(dataset), "--set", "val", "--results_path", str(results), "--re"],
        cwd=args.evaluator_root / "evaluation",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with (results / "global_results-val.csv").open(newline="") as handle:
        metrics = next(csv.DictReader(handle))
    summary = {
        "sequence": sequence,
        "prediction_source": "ground_truth_copy_for_evaluator_contract_smoke_only",
        "resize_max": args.resize_max,
        "official_evaluator_metrics": metrics,
        "stdout": completed.stdout,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
