"""Build deterministic VOST switch manifests from extracted frame names.

Only val is intended for the local external benchmark. Test names without
frames are rejected rather than silently converted into fake cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def frame_names(path: Path) -> list[str]:
    return sorted(p.stem for p in path.iterdir() if p.is_file() and not p.name.startswith("."))


def build(root: Path, split: str, quantiles: tuple[float, ...]) -> dict:
    sequences = [x.strip() for x in (root / "ImageSets" / f"{split}.txt").read_text().splitlines() if x.strip()]
    cases = []
    for sequence in sequences:
        frames = frame_names(root / "JPEGImages" / sequence)
        annotations = {p.stem for p in (root / "Annotations" / sequence).iterdir() if p.is_file() and not p.name.startswith(".")}
        if not frames or set(frames) != annotations:
            raise ValueError(f"frame/annotation mismatch or empty sequence: {sequence}")
        for q in quantiles:
            index = min(len(frames) - 1, max(0, int(round((len(frames) - 1) * q))))
            cases.append({
                "dataset": "VOST",
                "split": split,
                "sequence": sequence,
                "prompt_frame": frames[0],
                "switch_frame": frames[index],
                "switch_quantile": q,
                "frame_count": len(frames),
                "annotation_count": len(annotations),
                "future_gt_used": False,
            })
    result = {"schema_version": "cmmt.vost_switch_manifest.v1", "dataset": "VOST", "split": split, "quantiles": list(quantiles), "cases": cases}
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.split, (0.25, 0.50, 0.75))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"split": args.split, "cases": len(result["cases"]), "content_sha256": result["content_sha256"]}))


if __name__ == "__main__":
    main()
