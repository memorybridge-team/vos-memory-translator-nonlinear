"""Task 8: post-switch VOS metrics, DAVIS PNG I/O and the official adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from vos_memory_inspector import vos_metrics
from vos_memory_inspector.vos_metrics import (
    DAVIS_PALETTE,
    DAVIS2017_EVALUATION_COMMIT,
    VENDORED_METRICS_SHA256,
    evaluate_case,
    evaluate_case_pngs,
    frame_pngs,
    load_official_metrics,
    object_masks,
    read_indexed_png,
    region_jaccard,
    write_indexed_png,
)


SHAPE = (12, 16)


def _box(top: int, left: int, height: int = 4, width: int = 5) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


EMPTY = np.zeros(SHAPE, dtype=bool)
OBJECT = _box(3, 4)
SHIFTED = _box(3, 6)  # J(OBJECT, SHIFTED) = 12 / 28


def _official_or_skip():
    metrics, reason = load_official_metrics()
    if metrics is None:
        pytest.skip(reason)
    return metrics


# --------------------------------------------------------------------------
# J rule and PNG I/O
# --------------------------------------------------------------------------


def test_region_jaccard_follows_the_db_eval_iou_rule() -> None:
    assert region_jaccard(OBJECT, OBJECT) == 1.0
    assert region_jaccard(OBJECT, SHIFTED) == pytest.approx(12 / 28)
    assert region_jaccard(EMPTY, EMPTY) == 1.0  # empty union scores 1
    assert region_jaccard(OBJECT, EMPTY) == 0.0
    assert region_jaccard(EMPTY, OBJECT) == 0.0
    void = np.zeros(SHAPE, dtype=bool)
    void[:, 9:] = True  # hides the non-overlapping part of SHIFTED
    assert region_jaccard(OBJECT, SHIFTED, void) == pytest.approx(12 / 20)
    with pytest.raises(ValueError):
        region_jaccard(OBJECT, OBJECT[:-1])
    with pytest.raises(ValueError):
        region_jaccard(OBJECT[None], OBJECT[None])


def test_indexed_png_round_trip_preserves_labels_and_palette(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    labels = generator.choice(np.array([0, 1, 2, 3, 255], dtype=np.uint8), size=SHAPE)
    path = write_indexed_png(tmp_path / "masks" / "00007.png", labels)
    restored = read_indexed_png(path)
    assert restored.dtype == np.uint8 and np.array_equal(restored, labels)
    with Image.open(path) as image:
        assert image.mode == "P"
        palette = image.getpalette()
    assert palette[:12] == [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0]  # DAVIS / VOC colours
    assert tuple(palette[3 * 255 : 3 * 256]) == (224, 224, 192)
    assert tuple(palette[: 3 * 256]) == DAVIS_PALETTE
    mask, void = object_masks(restored, 2)
    assert np.array_equal(mask, labels == 2) and np.array_equal(void, labels == 255)
    write_indexed_png(tmp_path / "binary.png", labels == 1)
    assert np.array_equal(read_indexed_png(tmp_path / "binary.png"), (labels == 1).astype(np.uint8))

    Image.fromarray(labels).save(tmp_path / "gray.png")  # mode "L"
    assert np.array_equal(read_indexed_png(tmp_path / "gray.png"), labels)
    Image.fromarray(np.zeros((*SHAPE, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="indexed"):
        read_indexed_png(tmp_path / "rgb.png")
    with pytest.raises(ValueError):
        write_indexed_png(tmp_path / "bad.png", np.full(SHAPE, 300))
    with pytest.raises(ValueError):
        object_masks(labels, 255)


def test_frame_png_names(tmp_path: Path) -> None:
    for name in ("00003.png", "frame_00004.png", "notes.png"):
        write_indexed_png(tmp_path / name, np.zeros(SHAPE, dtype=np.uint8))
    assert sorted(frame_pngs(tmp_path)) == [3, 4]
    write_indexed_png(tmp_path / "frame_00003.png", np.zeros(SHAPE, dtype=np.uint8))
    with pytest.raises(ValueError, match="duplicate"):
        frame_pngs(tmp_path)


# --------------------------------------------------------------------------
# Diagnostics edge cases (repo J only, deterministic)
# --------------------------------------------------------------------------


def _case(gt: dict[int, np.ndarray | None], pred: dict[int, np.ndarray], **options):
    return evaluate_case(pred, gt, switch_frame=10, official=None, **options)


def test_no_reappearance_event() -> None:
    frames = range(11, 16)
    report = _case({f: OBJECT for f in frames}, {f: OBJECT for f in frames})
    reappearance = report["reappearance"]
    assert reappearance["events"] == [] and reappearance["count"] == 0
    assert reappearance["no_recovery_rate"] is None
    assert reappearance["mean_recovery_length"] is None
    assert report["post_switch"]["J"] == 1.0
    assert report["false_positives"]["rate"] is None  # no GT-absent frame
    json.dumps(report)


def test_recovery_failure_until_the_end_is_censored() -> None:
    gt = {11: EMPTY, 12: EMPTY, **{f: OBJECT for f in range(13, 21)}}
    report = _case(gt, {f: EMPTY for f in range(11, 21)})
    (event,) = report["reappearance"]["events"]
    assert event == {
        "absence_start_frame": 11,
        "reappearance_frame": 13,
        "visible_until_frame": 20,
        "recovered": False,
        "recovery_frame": None,
        "recovery_length": None,
        "censored_at_frame": 20,
    }
    assert report["reappearance"]["no_recovery_rate"] == 1.0
    assert report["reappearance"]["gt_present_at_switch"] is None


def test_recovery_length_counts_frames_until_threshold() -> None:
    gt = {10: OBJECT, 11: EMPTY, 12: EMPTY, **{f: OBJECT for f in range(13, 21)}}
    pred = {11: EMPTY, 12: EMPTY, 13: EMPTY, 14: SHIFTED, **{f: OBJECT for f in range(15, 21)}}
    report = _case(gt, pred)
    (event,) = report["reappearance"]["events"]
    assert (event["recovered"], event["recovery_frame"], event["recovery_length"]) == (True, 15, 2)
    assert report["reappearance"]["gt_present_at_switch"] is True
    assert report["reappearance"]["no_recovery_rate"] == 0.0
    assert report["reappearance"]["mean_recovery_length"] == 2
    # Lower threshold: the shifted mask (J = 0.43) already counts.
    lenient = _case(gt, pred, recovery_threshold=0.4)
    assert lenient["reappearance"]["events"][0]["recovery_length"] == 1


def test_absent_at_switch_makes_the_first_visible_frame_a_reappearance() -> None:
    gt = {10: EMPTY, 11: OBJECT, 12: OBJECT, 13: EMPTY, 14: OBJECT}
    pred = {11: OBJECT, 12: OBJECT, 13: EMPTY, 14: EMPTY}
    events = _case(gt, pred)["reappearance"]["events"]
    assert [(e["absence_start_frame"], e["reappearance_frame"], e["recovery_length"]) for e in events] == [
        (None, 11, 0),
        (13, 14, None),
    ]


def test_empty_gt_and_prediction_and_false_positive_area() -> None:
    small = _box(0, 0, 2, 3)  # 6 pixels
    gt = {11: EMPTY, 12: EMPTY, 13: OBJECT}
    pred = {11: EMPTY, 12: small, 13: OBJECT}
    report = _case(gt, pred)
    rows = {row["frame"]: row for row in report["frames"]}
    assert rows[11]["J"] == 1.0 and rows[11]["false_positive_pixels"] == 0
    assert rows[12]["J"] == 0.0 and rows[12]["false_positive_pixels"] == 6
    fp = report["false_positives"]
    assert (fp["gt_absent_frames"], fp["false_positive_frames"], fp["rate"]) == (2, 1, 0.5)
    assert fp["mean_area_pixels"] == 3.0
    assert fp["mean_area_fraction"] == pytest.approx(3.0 / (12 * 16))
    assert report["gt_absent"]["J"] == 0.5 and report["gt_visible"]["J"] == 1.0


def test_unannotated_frames_are_excluded_everywhere() -> None:
    frames = range(11, 41)  # offsets +1 .. +30
    gt: dict[int, np.ndarray | None] = {f: OBJECT for f in frames}
    gt[15] = None  # explicit "not annotated" (offset +5)
    del gt[16]  # missing key is also "not annotated"
    pred = {f: (SHIFTED if f in (15, 16) else OBJECT) for f in frames}
    pred[35] = EMPTY  # offset +25 lies in the remaining interval
    report = _case(gt, pred)
    assert report["frame_counts"] == {
        "evaluated": 30, "annotated": 28, "not_annotated": 2, "gt_visible": 28, "gt_absent": 0,
    }
    assert report["checkpoints"]["+5"] == {
        "frame": 15, "evaluated": True, "annotated": False, "gt_present": None,
        "J": None, "F": None, "J_and_F": None,
    }
    assert report["checkpoints"]["+1"]["J"] == 1.0 and report["checkpoints"]["+20"]["J"] == 1.0
    assert report["post_switch"]["J"] == pytest.approx(27 / 28)  # SHIFTED frames never count
    remaining = report["remaining"]
    assert remaining["frames"] == 10 and remaining["J"] == pytest.approx(0.9)
    assert report["post_switch"]["F"] is None and report["post_switch"]["J_and_F"] is None
    assert report["metric_sources"]["F_unavailable_reason"] == "official metrics disabled by caller"
    assert report["identity"]["reported"] is False


def test_checkpoint_beyond_the_case_and_input_errors() -> None:
    report = _case({11: OBJECT, 12: OBJECT}, {11: OBJECT, 12: OBJECT})
    assert report["checkpoints"]["+20"]["evaluated"] is False
    assert report["remaining"]["frames"] == 0 and report["remaining"]["J"] is None
    with pytest.raises(ValueError, match="contiguous"):
        _case({11: OBJECT, 13: OBJECT}, {11: OBJECT, 13: OBJECT})
    with pytest.raises(ValueError, match="no predictions"):
        _case({9: OBJECT}, {9: OBJECT})
    with pytest.raises(ValueError, match="annotation"):
        _case({11: OBJECT[:-1]}, {11: OBJECT})


# --------------------------------------------------------------------------
# Official functions
# --------------------------------------------------------------------------


def test_vendored_official_metrics_are_byte_identical_and_licensed() -> None:
    vendor = Path(vos_metrics.__file__).resolve().parent / "_vendor"
    digest = hashlib.sha256((vendor / "davis2017_metrics.py").read_bytes()).hexdigest()
    assert digest == VENDORED_METRICS_SHA256
    license_text = (vendor / "LICENSE.davis2017-evaluation").read_text(encoding="utf-8")
    assert license_text.startswith("BSD 3-Clause License")
    assert "DAVIS: Densely Annotated VIdeo Segmentation" in license_text


def test_official_j_matches_repo_j_when_installed() -> None:
    metrics = _official_or_skip()
    assert metrics.commit == DAVIS2017_EVALUATION_COMMIT
    generator = np.random.default_rng(3)
    for _ in range(50):
        annotation = generator.random(SHAPE) < 0.3
        segmentation = generator.random(SHAPE) < 0.3
        void = generator.random(SHAPE) < 0.1
        assert float(metrics.db_eval_iou(annotation, segmentation, void)) == region_jaccard(
            annotation, segmentation, void
        )
        assert float(metrics.db_eval_iou(annotation, segmentation)) == region_jaccard(annotation, segmentation)
    with np.errstate(invalid="ignore"):  # upstream computes 0/0 before its empty-union rule
        assert float(metrics.db_eval_iou(EMPTY, EMPTY)) == region_jaccard(EMPTY, EMPTY) == 1.0
    assert float(metrics.db_eval_boundary(EMPTY, EMPTY)) == 1.0


def test_case_report_uses_official_f_when_installed() -> None:
    metrics = _official_or_skip()
    gt = {11: OBJECT, 12: OBJECT, 13: EMPTY, 14: OBJECT}
    pred = {11: OBJECT, 12: SHIFTED, 13: EMPTY, 14: EMPTY}
    report = evaluate_case(pred, gt, switch_frame=10, official=metrics)
    assert report["j_parity_with_official"] == {"frames": 4, "max_abs_difference": 0.0}
    sources = report["metric_sources"]
    assert sources["F"]["matches_pinned_commit"] is True and sources["F_unavailable_reason"] is None
    rows = {row["frame"]: row for row in report["frames"]}
    assert rows[11]["F"] == 1.0 and rows[13]["F"] == 1.0 and rows[14]["F"] == 0.0
    assert 0.0 < rows[12]["F"] < 1.0
    post = report["post_switch"]
    assert post["J_and_F"] == pytest.approx((post["J"] + post["F"]) / 2)
    assert report["checkpoints"]["+1"]["J_and_F"] == 1.0
    json.dumps(report)


def test_checkout_source_loads_without_sys_path_changes(tmp_path: Path) -> None:
    _official_or_skip()
    vendored = Path(vos_metrics.__file__).resolve().parent / "_vendor" / "davis2017_metrics.py"
    (tmp_path / "davis2017").mkdir()
    shutil.copy(vendored, tmp_path / "davis2017" / "metrics.py")
    metrics, reason = load_official_metrics("checkout", evaluation_repo=tmp_path)
    assert reason is None and metrics.source == "checkout"
    assert metrics.sha256 == VENDORED_METRICS_SHA256 and metrics.commit is None
    assert metrics.to_dict()["matches_pinned_commit"] is False
    missing, why = load_official_metrics("checkout", evaluation_repo=tmp_path / "nowhere")
    assert missing is None and "not found" in why


def test_missing_eval_extra_records_null_and_reason(monkeypatch) -> None:
    vos_metrics._load_official.cache_clear()
    monkeypatch.setattr(vos_metrics, "_missing_dependencies", lambda: ["cv2", "skimage"])
    try:
        metrics, reason = load_official_metrics()
        assert metrics is None and "eval" in reason
        report = evaluate_case({11: OBJECT}, {11: OBJECT}, switch_frame=10)  # official="auto"
        assert report["post_switch"]["J"] == 1.0
        assert report["post_switch"]["F"] is None and report["post_switch"]["J_and_F"] is None
        assert report["metric_sources"]["F"] is None
        assert "eval" in report["metric_sources"]["F_unavailable_reason"]
        assert report["j_parity_with_official"] is None
    finally:
        vos_metrics._load_official.cache_clear()


def test_evaluate_case_pngs_reads_davis_label_maps(tmp_path: Path) -> None:
    annotations, predictions = tmp_path / "gt", tmp_path / "pred"
    for frame in range(10, 15):
        labels = np.zeros(SHAPE, dtype=np.uint8)
        labels[OBJECT] = 2
        labels[0, :] = 255  # void row
        if frame != 13:  # frame 13 has no annotation file
            write_indexed_png(annotations / f"{frame:05d}.png", labels)
        prediction = np.zeros(SHAPE, dtype=np.uint8)
        prediction[OBJECT] = 1
        prediction[0, :3] = 1  # lies in void: ignored by J
        write_indexed_png(predictions / f"frame_{frame:05d}.png", prediction)
    report = evaluate_case_pngs(predictions, annotations, object_id=2, switch_frame=10, official=None)
    assert report["frame_counts"]["evaluated"] == 4 and report["frame_counts"]["not_annotated"] == 1
    assert report["post_switch"]["J"] == 1.0
    assert report["inputs"]["object_id"] == 2
