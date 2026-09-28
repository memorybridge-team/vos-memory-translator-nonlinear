import json
from pathlib import Path

import scripts.build_m3vos_external_manifest as m3


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "M3VOS"
    (root / "data" / "JPEGImages" / "sequence").mkdir(parents=True)
    (root / "data" / "Annotations" / "sequence").mkdir(parents=True)
    for stem in ("0000000", "0000001", "0000002", "0000003", "0000004"):
        (root / "data" / "JPEGImages" / "sequence" / f"{stem}.jpg").write_bytes(b"rgb")
        (root / "data" / "Annotations" / "sequence" / f"{stem}.png").write_bytes(b"mask")
    (root / "data" / "ImageSets").mkdir()
    (root / "data" / "ImageSets" / "val.txt").write_text("sequence\n", encoding="utf-8")
    (root / "meta").mkdir()
    (root / "meta" / "all_core_seqs.txt").write_text("sequence\n", encoding="utf-8")
    (root / "meta" / "target_object.json").write_text(
        json.dumps({"sequence": {"obj_1": {}, "obj_2": {}}}), encoding="utf-8"
    )
    return root


def test_manifest_uses_actual_first_prompt_and_fixed_switches(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    labels = {
        "0000000": {0, 255},
        "0000001": {0, 2},
        "0000002": {0, 1, 2},
        "0000003": {0, 1, 2},
        "0000004": {0, 1, 2},
    }
    monkeypatch.setattr(m3, "_labels", lambda path: labels[path.stem])

    result = m3.build_manifest(root, revision="revision")

    assert result["dataset_revision"] == "revision"
    assert len(result["sequences"]) == 1
    assert len(result["cases"]) == 6
    by_object = {case["object_id"]: case for case in result["cases"] if case["switch_quantile"] == 0.5}
    assert by_object[1]["prompt_frame"] == "0000002"
    assert by_object[2]["prompt_frame"] == "0000001"
    assert all(case["future_gt_policy"] == "evaluation_only" for case in result["cases"])


def test_manifest_rejects_declared_object_without_gt_pixels(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    monkeypatch.setattr(m3, "_labels", lambda _path: {0})

    try:
        m3.build_manifest(root)
    except ValueError as error:
        assert "never appear" in str(error)
    else:
        raise AssertionError("expected missing declared object to fail")
