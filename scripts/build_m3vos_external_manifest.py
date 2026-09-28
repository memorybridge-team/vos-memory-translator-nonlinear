#!/usr/bin/env python3
"""Build a fixed, GT-prompt-only M³-VOS external-evaluation manifest.

Only each declared object's first non-empty GT mask is used as a VOS prompt.
Later masks never select the switch point and remain evaluation-only.  The
immutable delivery's ``val.txt`` is the full evaluation population; its
``all_core_seqs.txt`` is retained as a reported subset, not a model-selection
split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "cmmt.m3vos_external_manifest.v1"
SWITCH_QUANTILES = (0.25, 0.50, 0.75)
OBJECT_ID = re.compile(r"obj_(\d+)$")


def _content_sha256(document: dict[str, Any]) -> str:
    stable = dict(document)
    stable.pop("content_sha256", None)
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stems(directory: Path, suffix: str) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix and not path.name.startswith(".")
    )


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _declared_object_ids(record: object, sequence: str) -> list[int]:
    if not isinstance(record, dict):
        raise ValueError(f"target_object metadata for {sequence} is not a mapping")
    object_ids: list[int] = []
    for key in record:
        match = OBJECT_ID.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            raise ValueError(f"invalid target object ID in {sequence}: {key!r}")
        object_ids.append(int(match.group(1)))
    if not object_ids:
        raise ValueError(f"no declared objects in {sequence}")
    return sorted(object_ids)


def _labels(mask_path: Path) -> set[int]:
    """Read indexed PNG labels, preserving 255 so callers can handle void explicitly."""

    from PIL import Image

    with Image.open(mask_path) as image:
        colors = image.getcolors(maxcolors=65_536)
    if colors is None:
        raise ValueError(f"too many label values for indexed mask: {mask_path}")
    return {int(value) for _count, value in colors}


def build_manifest(root: Path, *, revision: str | None = None) -> dict[str, Any]:
    data = root / "data"
    frames_root = data / "JPEGImages"
    masks_root = data / "Annotations"
    split_path = data / "ImageSets" / "val.txt"
    target_path = root / "meta" / "target_object.json"
    core_path = root / "meta" / "all_core_seqs.txt"
    if not frames_root.is_dir() or not masks_root.is_dir() or not split_path.is_file() or not target_path.is_file():
        raise ValueError("root must contain data/{JPEGImages,Annotations,ImageSets/val.txt} and meta/target_object.json")

    declared_sequences = _read_lines(split_path)
    target_objects = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(target_objects, dict):
        raise ValueError("target_object.json must be a mapping")
    core_sequences = _read_lines(core_path) if core_path.is_file() else []
    if not set(core_sequences).issubset(declared_sequences):
        raise ValueError("all_core_seqs.txt contains sequences outside val.txt")

    sequences: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for sequence in declared_sequences:
        frame_stems = _stems(frames_root / sequence, ".jpg")
        mask_stems = _stems(masks_root / sequence, ".png")
        if not frame_stems or frame_stems != mask_stems:
            raise ValueError(f"unpaired or empty RGB/mask stems in {sequence}")
        declared_object_ids = _declared_object_ids(target_objects.get(sequence), sequence)
        first_prompt: dict[int, str] = {}
        void_seen = False
        observed_object_ids: set[int] = set()
        for stem in frame_stems:
            labels = _labels(masks_root / sequence / f"{stem}.png")
            void_seen = void_seen or 255 in labels
            frame_object_ids = {label for label in labels if label not in {0, 255}}
            observed_object_ids.update(frame_object_ids)
            for object_id in frame_object_ids:
                first_prompt.setdefault(object_id, stem)
        if not observed_object_ids:
            raise ValueError(f"no non-void object labels appear in {sequence}")
        # The annotation labels are the only objects that can be prompted and
        # scored. Metadata-only names are retained as an explicit discrepancy,
        # rather than being silently converted into empty-object cases.
        object_ids = sorted(observed_object_ids)
        declared_but_unannotated = sorted(set(declared_object_ids) - observed_object_ids)
        annotated_but_undeclared = sorted(observed_object_ids - set(declared_object_ids))

        sequences.append(
            {
                "sequence": sequence,
                "frame_count": len(frame_stems),
                "object_ids": object_ids,
                "declared_object_ids": declared_object_ids,
                "declared_but_unannotated_object_ids": declared_but_unannotated,
                "annotated_but_undeclared_object_ids": annotated_but_undeclared,
                "is_core": sequence in set(core_sequences),
                "void_label_255_seen_before_all_prompts": void_seen,
            }
        )
        index_by_stem = {stem: index for index, stem in enumerate(frame_stems)}
        for object_id in object_ids:
            prompt_stem = first_prompt[object_id]
            prompt_index = index_by_stem[prompt_stem]
            for quantile in SWITCH_QUANTILES:
                switch_index = max(prompt_index + 1, round((len(frame_stems) - 1) * quantile))
                if switch_index >= len(frame_stems):
                    continue
                cases.append(
                    {
                        "case_id": f"m3vos:{sequence}:obj{object_id}:q{int(quantile * 100):02d}",
                        "sequence": sequence,
                        "object_id": object_id,
                        "prompt_frame": prompt_stem,
                        "switch_frame": frame_stems[switch_index],
                        "switch_quantile": quantile,
                        "input_policy": "first_nonempty_gt_prompt_only",
                        "future_gt_policy": "evaluation_only",
                    }
                )

    document: dict[str, Any] = {
        "schema_version": SCHEMA,
        "dataset": "M3-VOS",
        "dataset_revision": revision,
        "role": "external_zero_shot_secondary_material_phase_transition_stress",
        "official_split": "data/ImageSets/val.txt",
        "core_subset": "meta/all_core_seqs.txt",
        "root_contract": "data/JPEGImages/<sequence>/*.jpg paired with data/Annotations/<sequence>/*.png",
        "object_contract": "meta/target_object.json keys obj_<positive integer>; label 255 is never a prompt object",
        "metadata_discrepancy_policy": "prompt and score only non-void labels present in annotations; record metadata-only and annotation-only labels per sequence",
        "prompt_policy": "one actual first-nonempty GT mask per declared object; later GT is evaluation-only",
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
    parser.add_argument("--revision")
    args = parser.parse_args()
    manifest = build_manifest(args.root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sequences={len(manifest['sequences'])} cases={len(manifest['cases'])} sha256={manifest['content_sha256']}")


if __name__ == "__main__":
    main()
