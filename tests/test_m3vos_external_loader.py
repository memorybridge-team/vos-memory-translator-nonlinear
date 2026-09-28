from pathlib import Path

import scripts.validate_m3vos_external_loader as loader


def _case() -> dict[str, object]:
    return {
        "case_id": "m3vos:s:obj1:q50",
        "sequence": "s",
        "object_id": 1,
        "prompt_frame": "0000000",
        "switch_frame": "0000002",
        "input_policy": "first_nonempty_gt_prompt_only",
        "future_gt_policy": "evaluation_only",
    }


def test_loader_accepts_prompt_and_switch_without_future_gt_read(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "M3VOS"
    for directory in (root / "data" / "JPEGImages" / "s", root / "data" / "Annotations" / "s"):
        directory.mkdir(parents=True)
    (root / "data" / "JPEGImages" / "s" / "0000000.jpg").write_bytes(b"rgb")
    (root / "data" / "Annotations" / "s" / "0000000.png").write_bytes(b"mask")
    (root / "data" / "JPEGImages" / "s" / "0000002.jpg").write_bytes(b"rgb")
    monkeypatch.setattr(loader, "_labels", lambda _path: {0, 1})

    result = loader.validate_manifest(root, {"content_sha256": "x", "cases": [_case()]})

    assert result["checked_case_count"] == 1
    assert result["failure_count"] == 0


def test_loader_rejects_void_or_missing_prompt_label(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "M3VOS"
    for directory in (root / "data" / "JPEGImages" / "s", root / "data" / "Annotations" / "s"):
        directory.mkdir(parents=True)
    (root / "data" / "JPEGImages" / "s" / "0000000.jpg").write_bytes(b"rgb")
    (root / "data" / "Annotations" / "s" / "0000000.png").write_bytes(b"mask")
    (root / "data" / "JPEGImages" / "s" / "0000002.jpg").write_bytes(b"rgb")
    monkeypatch.setattr(loader, "_labels", lambda _path: {0})

    result = loader.validate_manifest(root, {"cases": [_case()]})

    assert result["failure_count"] == 1
    assert result["failure_examples"][0]["reason"] == "prompt_mask_lacks_object_label"
