import json
from pathlib import Path

from scripts.validate_m3vos_inventory import build_inventory


def _write_m3_sequence(root: Path, sequence: str, *, with_annotation: bool = True) -> None:
    image_dir = root / "data" / "JPEGImages" / sequence
    annotation_dir = root / "data" / "Annotations" / sequence
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    (image_dir / "0000001.jpg").write_bytes(b"rgb")
    if with_annotation:
        (annotation_dir / "0000001.png").write_bytes(b"mask")


def _write_metadata(root: Path, sequences: list[str]) -> None:
    (root / "data" / "ImageSets").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "ImageSets" / "val.txt").write_text("\n".join(sequences), encoding="utf-8")
    (root / "meta" / "all_core_seqs.txt").write_text(sequences[0], encoding="utf-8")
    (root / "meta" / "target_object.json").write_text(
        json.dumps({sequence: ["obj_1"] for sequence in sequences}), encoding="utf-8"
    )
    (root / "m3vos_viewer_data_with_paths.jsonl").write_text(
        "\n".join(json.dumps({"video_id": sequence, "obj_id": "obj_1"}) for sequence in sequences),
        encoding="utf-8",
    )


def test_inventory_accepts_matching_delivery_sets(tmp_path: Path) -> None:
    root = tmp_path / "M3VOS"
    _write_m3_sequence(root, "video_a")
    _write_metadata(root, ["video_a"])

    result = build_inventory(root, revision="abc")

    assert result["dataset_revision"] == "abc"
    assert result["sequence_count"] == 1
    assert result["frame_count"] == result["annotation_count"] == 1
    assert result["object_record_count"] == 1
    assert result["failure_count"] == 0


def test_inventory_rejects_missing_mask_and_metadata_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "M3VOS"
    _write_m3_sequence(root, "video_a", with_annotation=False)
    _write_metadata(root, ["video_a"])
    (root / "m3vos_viewer_data_with_paths.jsonl").write_text(
        json.dumps({"video_id": "other_video", "obj_id": "obj_1"}), encoding="utf-8"
    )

    result = build_inventory(root)

    assert result["failure_count"] == 1
    assert result["sequences"][0]["missing_annotation_count"] == 1
    assert result["sequences"][0]["memberships"]["viewer_metadata"] is False
