from pathlib import Path

from scripts.validate_pumavos_inventory import build_inventory


def test_inventory_reports_exact_paired_stems(tmp_path: Path) -> None:
    root = tmp_path / "PUMaVOS"
    for directory in (root / "JPEGImages" / "video", root / "Annotations" / "video"):
        directory.mkdir(parents=True)
    (root / "JPEGImages" / "video" / "frame_000001.jpg").write_bytes(b"rgb")
    (root / "Annotations" / "video" / "frame_000001.png").write_bytes(b"mask")

    result = build_inventory(root)

    assert result["sequence_count"] == 1
    assert result["frame_count"] == result["annotation_count"] == 1
    assert result["failure_count"] == 0


def test_inventory_rejects_missing_annotation_or_frame(tmp_path: Path) -> None:
    root = tmp_path / "PUMaVOS"
    (root / "JPEGImages" / "video").mkdir(parents=True)
    (root / "JPEGImages" / "video" / "frame_000001.jpg").write_bytes(b"rgb")

    result = build_inventory(root)

    assert result["failure_count"] == 1
    assert result["sequences"][0]["missing_masks"] == ["frame_000001"]
