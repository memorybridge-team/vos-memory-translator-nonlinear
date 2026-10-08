"""운영 v1/v2 cache의 read-only, weights_only allowlist adapter.

completion/provenance를 만들어 넣지 않는다. 실제 tensor 정렬 검사와 미확인
생성 이력을 분리한다. SAM 2 모델을 생성하거나 호출하지 않는다.
"""
from pathlib import Path
import hashlib
import io
import time
import torch
from ..state_schema import CanonicalState, StateSpec
from ..transformer_translator import SAM21_MEMORY_SPEC
from .common import require, stamp, digest

SCHEMAS = {"cmmt.prepared_handoff_case.v1", "cmmt.prepared_handoff_case.v2"}
HISTORY = ("collector_revision", "source_checkpoint_sha256", "target_checkpoint_sha256", "prompt_history", "pair_mode")


def load_raw(path, *, expected_sha=None, expected_stamp=None, stable_seconds=60):
    path = Path(path)
    before = stamp(path)
    require(not Path(str(path) + ".writer.lock").exists(), "CACHE_WRITER_ACTIVE")
    require(time.time() - path.stat().st_mtime >= stable_seconds, "CACHE_NOT_STABLE")
    require(expected_stamp is None or expected_stamp == before, "CACHE_STAMP_CHANGED")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    require(sidecar.is_file() and sidecar.read_text().split()[0] == actual, "CACHE_CHECKSUM")
    require(expected_sha is None or expected_sha == actual, "CACHE_INDEX_SHA_CHANGED")
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    require(stamp(path) == before, "CACHE_CHANGED_DURING_READ")
    require(isinstance(value, dict) and value.get("schema_version") in SCHEMAS, "CACHE_SCHEMA")
    s, t = value["source_canonical"], value["target_canonical"]
    require(isinstance(s, CanonicalState) and isinstance(t, CanonicalState), "CANONICAL_STATE_REQUIRED")
    s.validate(); t.validate()
    require(s.spec == t.spec == SAM21_MEMORY_SPEC and s.validity.shape[:2] == (1, 1), "FROZEN_STATE_SPEC_SINGLE_OBJECT")
    require(s.object_ids == t.object_ids and s.switch_frame == t.switch_frame, "OBJECT_SWITCH_ALIGNMENT")
    for name in ("frame_indices", "slot_order", "is_conditioning", "validity"):
        require(torch.equal(getattr(s, name), getattr(t, name)), "RECORD_ALIGNMENT:" + name)
    require(s.frame_indices.dtype == t.frame_indices.dtype == torch.int64 and
            s.slot_order.dtype == t.slot_order.dtype == torch.int64, "DISCRETE_DTYPES")
    require(bool(s.validity.any()), "EMPTY_VALID_RECORDS")
    for state in (s, t):
        require(state.spatial_memory.dtype == torch.bfloat16 and state.object_pointer.dtype == torch.float32, "RUNTIME_DTYPES")
        for name in ("spatial_memory", "object_pointer", "presence_logits"):
            tensor = getattr(state, name)
            require(tensor.device.type == "cpu" and bool(torch.isfinite(tensor[state.validity]).all()), "NONFINITE_VALID_TENSOR:" + name)
        valid_slots = state.slot_order[state.validity].tolist()
        require(len(valid_slots) == len(set(valid_slots)) and min(valid_slots) >= 0, "DUPLICATE_OR_NEGATIVE_SLOT")
        require(bool((state.frame_indices[state.validity] <= state.switch_frame).all()), "FUTURE_STATE_RECORD")
    md = value["metadata"]
    require(md.get("source_model_id") == "sam2.1-small" and md.get("target_model_id") == "sam2.1-base-plus", "SMALL_BASE_DIRECTION")
    require(md.get("switch_frame") == s.switch_frame, "METADATA_SWITCH")
    require(md.get("cache_mode") == "state_only" and md.get("active_memory_only") is True and
            md.get("num_maskmem") == 7 and md.get("max_obj_ptrs_in_encoder") == 16, "MEMORY_POLICY")
    require(md.get("pair_mode", "native_history") == "native_history", "CONTROLLED_NATIVE_MIXING_FORBIDDEN")
    return value, actual, before


def inspect(case, path, rgb_root, *, stable_seconds=60):
    value, actual, before = load_raw(path, stable_seconds=stable_seconds)
    s = value["source_canonical"]
    md = value["metadata"]
    require(md.get("video_id") == case["video_id"] and str(md.get("object_id")) == str(case["object_id"]) and
            tuple(map(str, s.object_ids)) == (str(case["object_id"]),), "CASE_OBJECT_IDENTITY")
    images = [p for p in (Path(rgb_root) / case["video_id"]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    require(images and all(p.stem.isascii() and p.stem.isdigit() for p in images), "RGB_NUMERIC_FRAME_MAP")
    frames = sorted(int(p.stem) for p in images)
    require(len(frames) == len(set(frames)) and md["num_frames"] == len(frames), "RGB_FRAME_COUNT")
    if case["dataset"] == "MOSEv2":
        prompt, switch = int(case["first_prompt_frame"]), int(case["switch_frame"])
    else:
        require(case["first_prompt_frame"] in frames and case["switch_frame"] in frames, "LVOS_OFFICIAL_FRAME_MISSING")
        prompt, switch = frames.index(case["first_prompt_frame"]), frames.index(case["switch_frame"])
    require(0 <= prompt <= switch < len(frames) - 1 and s.switch_frame == switch, "RUNTIME_SWITCH_MAPPING")
    observed = s.frame_indices[s.validity & s.is_conditioning].tolist()
    require(observed and set(observed) == {prompt}, "CONDITIONING_PROMPT_RUNTIME_MISMATCH")
    require(bool((s.frame_indices[s.validity] >= prompt).all()), "RECORD_BEFORE_LATE_PROMPT")
    if "prompt_frame_index" in md:
        require(md["prompt_frame_index"] == prompt, "PROMPT_METADATA_RUNTIME")
    return dict(case=case, path=str(Path(path).resolve()), sha256=actual, stamp=before,
                valid_indices=torch.nonzero(s.validity).tolist(), records=int(s.validity.sum()),
                padded_records=s.validity.numel() - int(s.validity.sum()),
                runtime_prompt_frame=prompt, runtime_switch_frame=switch, conditioning_frames_observed=observed,
                frame_map_digest=digest(frames), policy={k: md.get(k) for k in
                    ("cache_mode", "active_memory_only", "num_maskmem", "max_obj_ptrs_in_encoder")},
                shapes=dict(spatial=list(s.spatial_memory.shape), pointer=list(s.object_pointer.shape)),
                dtypes=dict(spatial=str(s.spatial_memory.dtype), pointer=str(s.object_pointer.dtype)),
                historical_provenance=dict(known={k: md[k] for k in HISTORY if k in md}, unknown=[k for k in HISTORY if k not in md]),
                mask_rgb_injection_history="UNKNOWN_IF_NO_ORIGINAL_MASK_DIGEST",
                synthetic_declared=bool(md.get("synthetic", False)))
