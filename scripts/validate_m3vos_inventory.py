"""Validate an M³-VOS delivery before using it as an external benchmark.

The Hugging Face delivery contains a ``data`` tree plus repository metadata.
This verifier treats the supplied ``ImageSets/val.txt`` as the declared split
and checks that it agrees with the RGB tree, annotations, object metadata and
viewer metadata.  It deliberately records observed counts instead of silently
assuming a count quoted for another release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _file_stems(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        file.stem
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() == suffix and not file.name.startswith(".")
    }


def _directories(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {child.name for child in directory.iterdir() if child.is_dir() and not child.name.startswith(".")}


def _read_split(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _viewer_video_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    video_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = row.get("video_id")
        if isinstance(video_id, str) and video_id:
            video_ids.add(video_id)
    return video_ids


def _examples(items: set[str], limit: int = 20) -> list[str]:
    return sorted(items)[:limit]


def _object_ids(value: Any) -> list[str]:
    """Normalize the release's object record without discarding its labels.

    The current M³-VOS release stores ``{"obj_1": {...}}`` per sequence,
    whereas early mirror descriptions represented object IDs as a list.
    Supporting both makes the validator an inventory check rather than an
    undocumented release-format assumption.
    """

    if isinstance(value, dict):
        return sorted(key for key in value if isinstance(key, str))
    if isinstance(value, list):
        return sorted(item for item in value if isinstance(item, str))
    return []


def build_inventory(root: Path, *, revision: str | None = None) -> dict[str, object]:
    """Return a compact, deterministic inventory for an M³-VOS delivery."""

    data = root / "data"
    meta = root / "meta"
    split_path = data / "ImageSets" / "val.txt"
    target_objects = _read_json(meta / "target_object.json")
    split_sequences = _read_split(split_path)
    image_sequences = _directories(data / "JPEGImages")
    annotation_sequences = _directories(data / "Annotations")
    viewer_sequences = _viewer_video_ids(root / "m3vos_viewer_data_with_paths.jsonl")
    metadata_sequences = set(target_objects)
    core_sequences = _read_split(meta / "all_core_seqs.txt")

    declared_sets = {
        "val_split": split_sequences,
        "jpegimages": image_sequences,
        "annotations": annotation_sequences,
        "viewer_metadata": viewer_sequences,
        "target_object_metadata": metadata_sequences,
    }
    union = set().union(*declared_sets.values())
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    total_frames = 0
    total_annotations = 0
    total_objects = 0

    for sequence in sorted(union):
        frame_stems = _file_stems(data / "JPEGImages" / sequence, ".jpg")
        annotation_stems = _file_stems(data / "Annotations" / sequence, ".png")
        missing_annotations = frame_stems - annotation_stems
        missing_frames = annotation_stems - frame_stems
        object_ids = _object_ids(target_objects.get(sequence, []))
        memberships = {name: sequence in values for name, values in declared_sets.items()}
        valid = (
            all(memberships.values())
            and bool(frame_stems)
            and not missing_annotations
            and not missing_frames
            and bool(object_ids)
        )
        if not valid:
            failures.append(sequence)
        total_frames += len(frame_stems)
        total_annotations += len(annotation_stems)
        total_objects += len(object_ids)
        rows.append(
            {
                "sequence": sequence,
                "memberships": memberships,
                "frame_count": len(frame_stems),
                "annotation_count": len(annotation_stems),
                "object_ids": object_ids,
                "missing_annotation_count": len(missing_annotations),
                "missing_frame_count": len(missing_frames),
                "missing_annotation_examples": _examples(missing_annotations),
                "missing_frame_examples": _examples(missing_frames),
                "valid": valid,
            }
        )

    core_not_in_split = core_sequences - split_sequences
    if core_not_in_split:
        failures.append("__core_not_in_val_split__")
    return {
        "schema_version": "cmmt.m3vos_inventory.v1",
        "dataset": "M3-VOS",
        "root": str(root),
        "dataset_revision": revision,
        "declared_split": "data/ImageSets/val.txt",
        "sequence_count": len(rows),
        "frame_count": total_frames,
        "annotation_count": total_annotations,
        "object_record_count": total_objects,
        "core_sequence_count": len(core_sequences),
        "core_not_in_split": _examples(core_not_in_split),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "sequences": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="M³-VOS root containing data/ and meta/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", help="Immutable source revision used for this delivery")
    args = parser.parse_args()
    inventory = build_inventory(args.root, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: inventory[key] for key in ("sequence_count", "frame_count", "annotation_count", "object_record_count", "core_sequence_count", "failure_count")}))


if __name__ == "__main__":
    main()
