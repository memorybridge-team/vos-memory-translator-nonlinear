from pathlib import Path

from PIL import Image

from scripts.build_pumavos_external_manifest import build_manifest


def _mask(path: Path, values: list[int]) -> None:
    image = Image.new("L", (len(values), 1))
    image.putdata(values)
    image.save(path)


def test_manifest_uses_first_nonempty_prompt_and_future_gt_only(tmp_path: Path) -> None:
    root = tmp_path / "PUBLIC_PUMaVOS"
    rgb = root / "JPEGImages" / "example"
    ann = root / "Annotations" / "example"
    rgb.mkdir(parents=True)
    ann.mkdir(parents=True)
    for stem, labels in [("00000", [0, 0]), ("00001", [0, 2]), ("00002", [2, 7]), ("00003", [7, 0])]:
        (rgb / f"{stem}.jpg").write_bytes(b"rgb")
        _mask(ann / f"{stem}.png", labels)

    manifest = build_manifest(root)

    assert manifest["schema_version"] == "cmmt.pumavos_external_manifest.v1"
    assert manifest["sequences"][0]["object_ids"] == [2, 7]
    prompts = {case["object_id"]: case["prompt_frame"] for case in manifest["cases"]}
    assert prompts[2] == "00001"
    assert prompts[7] == "00002"
    assert all(case["future_gt_policy"] == "evaluation_only" for case in manifest["cases"])
