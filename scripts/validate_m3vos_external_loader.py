#!/usr/bin/env python3
"""Validate that every frozen M³-VOS case can be loaded without future-GT input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _labels(mask_path: Path) -> set[int]:
    from PIL import Image

    with Image.open(mask_path) as image:
        colors = image.getcolors(maxcolors=65_536)
    if colors is None:
        raise ValueError(f"too many label values for indexed mask: {mask_path}")
    return {int(value) for _count, value in colors}


def validate_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check first-prompt input paths and post-prompt switch paths for every case.

    This validator intentionally reads only the prompt GT mask. It proves the
    prompt object exists but does not inspect future GT pixels, preserving the
    frozen external-evaluation input contract.
    """

    data = root / "data"
    failures: list[dict[str, str]] = []
    checked = 0
    for case in manifest.get("cases", []):
        case_id = str(case.get("case_id", "<missing>"))
        sequence = str(case.get("sequence", ""))
        object_id = case.get("object_id")
        prompt = str(case.get("prompt_frame", ""))
        switch = str(case.get("switch_frame", ""))
        rgb_prompt = data / "JPEGImages" / sequence / f"{prompt}.jpg"
        mask_prompt = data / "Annotations" / sequence / f"{prompt}.png"
        rgb_switch = data / "JPEGImages" / sequence / f"{switch}.jpg"
        if not rgb_prompt.is_file() or not mask_prompt.is_file() or not rgb_switch.is_file():
            failures.append({"case_id": case_id, "reason": "missing_prompt_or_switch_path"})
            continue
        if not isinstance(object_id, int) or object_id <= 0 or object_id == 255:
            failures.append({"case_id": case_id, "reason": "invalid_or_void_object_id"})
            continue
        if object_id not in _labels(mask_prompt):
            failures.append({"case_id": case_id, "reason": "prompt_mask_lacks_object_label"})
            continue
        if prompt >= switch:
            failures.append({"case_id": case_id, "reason": "switch_not_strictly_after_prompt"})
            continue
        if case.get("input_policy") != "first_nonempty_gt_prompt_only":
            failures.append({"case_id": case_id, "reason": "unexpected_input_policy"})
            continue
        if case.get("future_gt_policy") != "evaluation_only":
            failures.append({"case_id": case_id, "reason": "unexpected_future_gt_policy"})
            continue
        checked += 1
    return {
        "schema_version": "cmmt.m3vos_loader_validation.v1",
        "dataset": "M3-VOS",
        "manifest_content_sha256": manifest.get("content_sha256"),
        "case_count": len(manifest.get("cases", [])),
        "checked_case_count": checked,
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "input_contract": "only first prompt GT mask is read; switch RGB is read; future GT is not read",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_manifest(args.root, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "checked_case_count", "failure_count")}))


if __name__ == "__main__":
    main()
