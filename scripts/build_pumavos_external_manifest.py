#!/usr/bin/env python3
"""Build a GT-prompt-only external evaluation manifest for PUMaVOS.

PUMaVOS has no official train/validation/test split.  This builder does not
create one: it enumerates the verified public archive, finds each object's
first non-empty GT mask for the standard VOS prompt, and chooses fixed
frame-index switch points.  Later GT masks are not used as handoff input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


SCHEMA = "cmmt.pumavos_external_manifest.v1"
SWITCH_QUANTILES = (0.25, 0.50, 0.75)


def _frame_paths(directory: Path, suffix: str) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.glob(f"*{suffix}"))}


def _non_background_labels(mask_path: Path) -> set[int]:
    with Image.open(mask_path) as image:
        # PUMaVOS masks are indexed instance-label PNGs. Background is 0.
        return {int(value) for value in image.getdata() if int(value) != 0}


def _content_sha256(document: dict[str, Any]) -> str:
    stable = dict(document)
    stable.pop("content_sha256", None)
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(root: Path) -> dict[str, Any]:
    frames_root = root / "JPEGImages"
    masks_root = root / "Annotations"
    if not frames_root.is_dir() or not masks_root.is_dir():
        raise ValueError("root must contain JPEGImages and Annotations")

    cases: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for sequence_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        mask_dir = masks_root / sequence
        frame_by_stem = _frame_paths(sequence_dir, ".jpg")
        mask_by_stem = _frame_paths(mask_dir, ".png")
        if not frame_by_stem or set(frame_by_stem) != set(mask_by_stem):
            raise ValueError(f"unpaired RGB/masks in sequence {sequence}")

        stems = sorted(frame_by_stem)
        first_prompt: dict[int, str] = {}
        for stem in stems:
            for object_id in _non_background_labels(mask_by_stem[stem]):
                first_prompt.setdefault(object_id, stem)

        sequences.append(
            {
                "sequence": sequence,
                "frame_count": len(stems),
                "object_ids": sorted(first_prompt),
            }
        )
        for object_id, prompt_stem in sorted(first_prompt.items()):
            prompt_index = stems.index(prompt_stem)
            for quantile in SWITCH_QUANTILES:
                switch_index = max(prompt_index + 1, round((len(stems) - 1) * quantile))
                if switch_index >= len(stems):
                    continue
                cases.append(
                    {
                        "case_id": f"pumavos:{sequence}:obj{object_id}:q{int(quantile * 100):02d}",
                        "sequence": sequence,
                        "object_id": object_id,
                        "prompt_frame": prompt_stem,
                        "switch_frame": stems[switch_index],
                        "switch_quantile": quantile,
                        "input_policy": "first_nonempty_gt_prompt_only",
                        "future_gt_policy": "evaluation_only",
                    }
                )

    document: dict[str, Any] = {
        "schema_version": SCHEMA,
        "dataset": "PUMaVOS",
        "official_split": None,
        "role": "external_zero_shot_secondary_stress",
        "root_contract": "JPEGImages/<sequence>/*.jpg paired with Annotations/<sequence>/*.png",
        "prompt_policy": "one first-nonempty GT mask per object; later GT is evaluation-only",
        "switch_policy": "fixed frame-index quantiles 25/50/75%, strictly after prompt",
        "sequences": sequences,
        "cases": cases,
    }
    document["content_sha256"] = _content_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"sequences={len(manifest['sequences'])} cases={len(manifest['cases'])} "
        f"sha256={manifest['content_sha256']}"
    )


if __name__ == "__main__":
    main()
