"""Post-switch VOS metrics for handoff cases (plan Section 8; Task 8).

- ``J``: in-repo region similarity following the official ``db_eval_iou`` rule
  (void pixels excluded; an empty union scores 1).
- ``F`` and ``J&F``: the official davis2017-evaluation functions
  (``davis2017/metrics.py`` vendored byte-identically at commit ``ac7c43f``).
  They need the ``eval`` extra (``cv2``, ``skimage``); without it the fields are
  ``None`` and the report records why.
- Diagnostics (CMMT research definitions): ``J@+1/+5/+20``, remaining-interval
  J&F, reappearance recovery length (frames until ``J >= tau``, ``tau = 0.5``),
  no-recovery rate and GT-absent false positives. Identity maintenance is not
  reported until a decision rule is fixed.

Scores cover the case's post-switch annotated frames only. Report them as
"post-switch J&F (official metric functions)", never as a benchmark score.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = "cmmt.post_switch_vos_metrics.v1"
DAVIS2017_EVALUATION_COMMIT = "ac7c43fca936f9722837b7fbd337d284ba37004b"
VENDORED_METRICS_SHA256 = "a71bfb6d2da563ebf50251bda9876cf0b4b6193842543b0a3a286190529e26be"
VOID_LABEL = 255
DEFAULT_CHECKPOINTS = (1, 5, 20)
DEFAULT_RECOVERY_THRESHOLD = 0.5
METRIC_NAME = "post-switch J&F (official metric functions)"

_VENDORED_METRICS = Path(__file__).resolve().parent / "_vendor" / "davis2017_metrics.py"


# --------------------------------------------------------------------------
# Region similarity (J)
# --------------------------------------------------------------------------


def region_jaccard(
    annotation: np.ndarray, segmentation: np.ndarray, void: np.ndarray | None = None
) -> float:
    """J for one 2-D frame with the official ``db_eval_iou`` rule."""

    annotation = np.asarray(annotation).astype(bool)
    segmentation = np.asarray(segmentation).astype(bool)
    if annotation.ndim != 2 or annotation.shape != segmentation.shape:
        raise ValueError(
            f"annotation {annotation.shape} and segmentation {segmentation.shape} must be "
            "equal 2-D masks"
        )
    keep = np.ones_like(annotation) if void is None else ~_void_mask(void, annotation.shape)
    union = np.count_nonzero((annotation | segmentation) & keep)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(annotation & segmentation & keep) / union)


def _void_mask(void: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    void = np.asarray(void).astype(bool)
    if void.shape != shape:
        raise ValueError(f"void mask {void.shape} does not match {shape}")
    return void


# --------------------------------------------------------------------------
# Official davis2017 functions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialMetrics:
    """``db_eval_iou`` / ``db_eval_boundary`` from davis2017-evaluation."""

    db_eval_iou: Callable[..., Any]
    db_eval_boundary: Callable[..., Any]
    source: str  # "vendored" | "installed" | "checkout"
    location: str
    commit: str | None
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": "davis2017-evaluation",
            "functions": ["db_eval_iou", "db_eval_boundary"],
            "source": self.source,
            "location": self.location,
            "commit": self.commit,
            "pinned_commit": DAVIS2017_EVALUATION_COMMIT,
            "matches_pinned_commit": self.commit == DAVIS2017_EVALUATION_COMMIT,
            "metrics_sha256": self.sha256,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _missing_dependencies() -> list[str]:
    return [name for name in ("cv2", "skimage") if importlib.util.find_spec(name) is None]


def _load_file_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _installed_commit() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        import json

        text = distribution("davis2017").read_text("direct_url.json")
    except (ImportError, PackageNotFoundError, FileNotFoundError):
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


@functools.lru_cache(maxsize=None)
def _load_official(source: str, evaluation_repo: str | None) -> tuple[OfficialMetrics | None, str | None]:
    missing = _missing_dependencies()
    if missing:
        return None, (
            f"official davis2017 metrics need {missing}; install the 'eval' extra "
            "(pip install -e '.[eval]')"
        )
    try:
        if source == "vendored":
            digest = _sha256(_VENDORED_METRICS)
            if digest != VENDORED_METRICS_SHA256:
                return None, f"vendored davis2017 metrics.py was modified (sha256 {digest})"
            module = importlib.import_module("vos_memory_inspector._vendor.davis2017_metrics")
            location, commit = str(_VENDORED_METRICS), DAVIS2017_EVALUATION_COMMIT
        elif source == "installed":
            module = importlib.import_module("davis2017.metrics")
            location = str(Path(module.__file__).resolve())
            digest = _sha256(Path(location))
            commit = _installed_commit() or _git_commit(Path(location).parent)
        elif source == "checkout":
            if evaluation_repo is None:
                return None, "source='checkout' needs evaluation_repo"
            path = Path(evaluation_repo).resolve() / "davis2017" / "metrics.py"
            if not path.is_file():
                return None, f"official metrics file not found: {path}"
            module = _load_file_module("cmmt_davis2017_checkout_metrics", path)
            location, digest, commit = str(path), _sha256(path), _git_commit(path.parent)
        else:
            raise ValueError(f"unknown official metrics source {source!r}")
    except ImportError as exc:
        return None, f"official davis2017 metrics are not importable: {exc}"
    return (
        OfficialMetrics(module.db_eval_iou, module.db_eval_boundary, source, location, commit, digest),
        None,
    )


def load_official_metrics(
    source: str = "vendored", *, evaluation_repo: str | Path | None = None
) -> tuple[OfficialMetrics | None, str | None]:
    """Return ``(metrics, None)`` or ``(None, reason)``; never raises for a missing install.

    ``vendored`` (default) is the pinned byte-identical copy; ``installed``
    imports a separately installed ``davis2017`` package; ``checkout`` loads
    ``<evaluation_repo>/davis2017/metrics.py`` without touching ``sys.path``.
    """

    if source not in {"vendored", "installed", "checkout"}:
        raise ValueError(f"unknown official metrics source {source!r}")
    return _load_official(source, None if evaluation_repo is None else str(evaluation_repo))


# --------------------------------------------------------------------------
# DAVIS indexed PNG I/O
# --------------------------------------------------------------------------


def _voc_palette() -> tuple[int, ...]:
    """PASCAL VOC colour map written by the official ``save_mask`` (``davis2017.utils.color_map``).

    Most DAVIS 2017 annotation files carry this palette; some use a variant
    (e.g. 191 instead of 192). Only label indices matter for scoring.
    """

    palette: list[int] = []
    for index in range(256):
        red = green = blue = 0
        value = index
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend((red, green, blue))
    return tuple(palette)


DAVIS_PALETTE = _voc_palette()


def read_indexed_png(path: str | Path) -> np.ndarray:
    """Label map ``[H,W] uint8`` from a palette (``P``) or grayscale (``L``) PNG."""

    with Image.open(path) as image:
        if image.mode not in {"P", "L"}:
            raise ValueError(
                f"{path}: expected an indexed ('P') or grayscale ('L') PNG, got mode {image.mode!r}"
            )
        return np.array(image, dtype=np.uint8)


def write_indexed_png(
    path: str | Path, labels: np.ndarray, *, palette: Sequence[int] = DAVIS_PALETTE
) -> Path:
    """Write a label map as a DAVIS-style palette PNG (indices preserved exactly)."""

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2-D array, got shape {labels.shape}")
    if labels.dtype == bool:
        labels = labels.astype(np.uint8)
    if not np.issubdtype(labels.dtype, np.integer) or labels.min(initial=0) < 0 or labels.max(initial=0) > 255:
        raise ValueError("labels must be integers in [0, 255]")
    image = Image.fromarray(labels.astype(np.uint8))  # mode "L"
    image.putpalette(list(palette))  # "L" + palette -> "P"; pixel values stay indices
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def object_masks(labels: np.ndarray, object_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(binary object mask, void mask) from a DAVIS label map."""

    if object_id in (0, VOID_LABEL):
        raise ValueError(f"object_id must not be background (0) or void ({VOID_LABEL})")
    labels = np.asarray(labels)
    return labels == object_id, labels == VOID_LABEL


def frame_pngs(directory: str | Path) -> dict[int, Path]:
    """``{frame: path}`` for ``00012.png`` or ``frame_00012.png`` style names."""

    frames: dict[int, Path] = {}
    for path in sorted(Path(directory).glob("*.png")):
        stem = path.stem.removeprefix("frame_")
        if stem.isdigit():
            frame = int(stem)
            if frame in frames:
                raise ValueError(f"duplicate PNG for frame {frame} in {directory}")
            frames[frame] = path
    return frames


# --------------------------------------------------------------------------
# Case evaluation
# --------------------------------------------------------------------------


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def _scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    j = _mean_or_none(row["J"] for row in rows)
    f = _mean_or_none(row["F"] for row in rows) if all(row["F"] is not None for row in rows) else None
    return {
        "frames": len(rows),
        "J": j,
        "F": f,
        "J_and_F": None if j is None or f is None else (j + f) / 2,
    }


def _reappearance_events(
    rows: Sequence[Mapping[str, Any]], *, context_present: bool | None, threshold: float
) -> list[dict[str, Any]]:
    """Absent -> present transitions over annotated frames and their recovery."""

    annotated = [row for row in rows if row["annotated"]]
    events: list[dict[str, Any]] = []
    previous = context_present
    absent_since: int | None = None
    for position, row in enumerate(annotated):
        if not row["gt_present"]:
            if previous is not False:
                absent_since = row["frame"]
            previous = False
            continue
        if previous is False:
            window = []
            for candidate in annotated[position:]:
                if not candidate["gt_present"]:
                    break
                window.append(candidate)
            recovered = next((item for item in window if item["J"] >= threshold), None)
            events.append(
                {
                    "absence_start_frame": absent_since,
                    "reappearance_frame": row["frame"],
                    "visible_until_frame": window[-1]["frame"],
                    "recovered": recovered is not None,
                    "recovery_frame": None if recovered is None else recovered["frame"],
                    "recovery_length": None if recovered is None else recovered["frame"] - row["frame"],
                    "censored_at_frame": None if recovered is not None else window[-1]["frame"],
                }
            )
        previous = True
    return events


def evaluate_case(
    predictions: Mapping[int, np.ndarray],
    annotations: Mapping[int, np.ndarray | None],
    *,
    switch_frame: int,
    voids: Mapping[int, np.ndarray | None] | None = None,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD,
    official: OfficialMetrics | str | None = "auto",
) -> dict[str, Any]:
    """Score one ``(video, object, switch)`` continuation; JSON-safe result.

    ``predictions`` maps every frame after ``switch_frame`` (contiguous) to a
    binary mask of the object. ``annotations`` maps frames to binary GT masks;
    a missing key or ``None`` means the frame is not annotated and is excluded
    from every score. The latest annotated frame at or before the switch, if
    given, only tells whether the object was absent when the switch happened.
    ``official="auto"`` loads the vendored official functions; ``None`` skips F.
    """

    if any(offset < 1 for offset in checkpoints):
        raise ValueError("checkpoint offsets must be positive")
    if not 0.0 <= recovery_threshold <= 1.0:
        raise ValueError("recovery_threshold must be in [0, 1]")
    frames = sorted(frame for frame in predictions if frame > switch_frame)
    if not frames:
        raise ValueError(f"no predictions after switch frame {switch_frame}")
    missing = sorted(set(range(switch_frame + 1, frames[-1] + 1)) - set(frames))
    if missing:
        raise ValueError(f"post-switch predictions are not contiguous; missing frames {missing}")
    if official == "auto":
        metrics, official_reason = load_official_metrics()
    elif official is None:
        metrics, official_reason = None, "official metrics disabled by caller"
    elif isinstance(official, OfficialMetrics):
        metrics, official_reason = official, None
    else:
        raise TypeError("official must be 'auto', None or OfficialMetrics")
    voids = voids or {}

    rows: list[dict[str, Any]] = []
    parity: list[float] = []
    for frame in frames:
        prediction = np.asarray(predictions[frame]).astype(bool)
        annotation = annotations.get(frame)
        row: dict[str, Any] = {
            "frame": frame,
            "offset": frame - switch_frame,
            "annotated": annotation is not None,
            "prediction_present": bool(prediction.any()),
            "gt_present": None,
            "J": None,
            "F": None,
            "J_and_F": None,
            "false_positive_pixels": None,
        }
        if annotation is not None:
            annotation = np.asarray(annotation).astype(bool)
            if annotation.shape != prediction.shape:
                raise ValueError(
                    f"frame {frame}: prediction {prediction.shape} != annotation {annotation.shape}"
                )
            void = voids.get(frame)
            void = None if void is None else _void_mask(void, annotation.shape)
            row["gt_present"] = bool(annotation.any())
            row["J"] = region_jaccard(annotation, prediction, void)
            if metrics is not None:
                # Upstream divides 0/0 before applying the empty-union rule.
                with np.errstate(invalid="ignore", divide="ignore"):
                    official_j = float(metrics.db_eval_iou(annotation, prediction, void))
                    row["F"] = float(metrics.db_eval_boundary(annotation, prediction, void))
                parity.append(abs(official_j - row["J"]))
                row["J_and_F"] = (row["J"] + row["F"]) / 2
            if not row["gt_present"]:
                keep = np.ones_like(prediction) if void is None else ~void
                row["false_positive_pixels"] = int(np.count_nonzero(prediction & keep))
        rows.append(row)

    annotated = [row for row in rows if row["annotated"]]
    visible = [row for row in annotated if row["gt_present"]]
    absent = [row for row in annotated if not row["gt_present"]]
    last_checkpoint = max(checkpoints) if checkpoints else 0
    checkpoint_scores = {}
    by_frame = {row["frame"]: row for row in rows}
    for offset in checkpoints:
        row = by_frame.get(switch_frame + offset)
        checkpoint_scores[f"+{offset}"] = {
            "frame": switch_frame + offset,
            "evaluated": row is not None,
            "annotated": bool(row and row["annotated"]),
            "gt_present": None if row is None else row["gt_present"],
            "J": None if row is None else row["J"],
            "F": None if row is None else row["F"],
            "J_and_F": None if row is None else row["J_and_F"],
        }

    context = [frame for frame, mask in annotations.items() if frame <= switch_frame and mask is not None]
    context_present = bool(np.asarray(annotations[max(context)]).any()) if context else None
    events = _reappearance_events(rows, context_present=context_present, threshold=recovery_threshold)
    recovered = [event for event in events if event["recovered"]]
    pixels = int(np.asarray(predictions[frames[0]]).size)
    fp_frames = [row for row in absent if row["false_positive_pixels"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "name": METRIC_NAME,
        "scope": "post-switch annotated frames of one case; not a benchmark score",
        "switch_frame": switch_frame,
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "frame_counts": {
            "evaluated": len(rows),
            "annotated": len(annotated),
            "not_annotated": len(rows) - len(annotated),
            "gt_visible": len(visible),
            "gt_absent": len(absent),
        },
        "metric_sources": {
            "J": "in-repo implementation of the official db_eval_iou rule",
            "F": None if metrics is None else metrics.to_dict(),
            "F_unavailable_reason": official_reason,
        },
        "j_parity_with_official": (
            None if metrics is None else {"frames": len(parity), "max_abs_difference": max(parity, default=0.0)}
        ),
        "post_switch": _scores(annotated),
        "gt_visible": _scores(visible),
        "gt_absent": _scores(absent),
        "checkpoints": checkpoint_scores,
        "remaining": {
            "definition": f"annotated frames with offset > +{last_checkpoint}",
            **_scores([row for row in annotated if row["offset"] > last_checkpoint]),
        },
        "reappearance": {
            "definition": (
                "absent -> present transitions over annotated frames; recovery length is "
                "frames from the reappearance to the first visible frame with J >= threshold "
                "(0 = recovered on reappearance); unrecovered events are censored at the end "
                "of the visible run; absence_start_frame None = already absent at the switch"
            ),
            "threshold": recovery_threshold,
            "threshold_note": "research definition, not an official metric",
            "gt_present_at_switch": context_present,
            "events": events,
            "count": len(events),
            "recovered": len(recovered),
            "no_recovery_rate": None if not events else (len(events) - len(recovered)) / len(events),
            "mean_recovery_length": _mean_or_none(event["recovery_length"] for event in recovered),
        },
        "false_positives": {
            "definition": (
                "over annotated GT-absent frames: rate of frames with a non-empty prediction, "
                "and mean predicted area (void excluded) per GT-absent frame"
            ),
            "gt_absent_frames": len(absent),
            "false_positive_frames": len(fp_frames),
            "rate": None if not absent else len(fp_frames) / len(absent),
            "mean_area_pixels": _mean_or_none(row["false_positive_pixels"] for row in absent),
            "mean_area_fraction": _mean_or_none(
                row["false_positive_pixels"] / pixels for row in absent
            ),
        },
        "identity": {
            "reported": False,
            "reason": "no identity decision rule has been fixed (plan Section 8)",
        },
        "frames": rows,
    }


def evaluate_case_pngs(
    prediction_directory: str | Path,
    annotation_directory: str | Path,
    *,
    object_id: int,
    switch_frame: int,
    prediction_object_id: int | None = None,
    **options: Any,
) -> dict[str, Any]:
    """``evaluate_case`` from PNG directories.

    Annotations are DAVIS label maps (``object_id`` foreground, 255 void); a
    missing annotation file means "not annotated". Predictions are binary
    (``> 0``) unless ``prediction_object_id`` selects one label.
    """

    predictions = {}
    for frame, path in frame_pngs(prediction_directory).items():
        labels = read_indexed_png(path)
        predictions[frame] = labels > 0 if prediction_object_id is None else labels == prediction_object_id
    annotations: dict[int, np.ndarray] = {}
    voids: dict[int, np.ndarray] = {}
    for frame, path in frame_pngs(annotation_directory).items():
        annotations[frame], voids[frame] = object_masks(read_indexed_png(path), object_id)
    report = evaluate_case(predictions, annotations, switch_frame=switch_frame, voids=voids, **options)
    report["inputs"] = {
        "prediction_directory": str(Path(prediction_directory).resolve()),
        "annotation_directory": str(Path(annotation_directory).resolve()),
        "object_id": object_id,
        "prediction_object_id": prediction_object_id,
    }
    return report
