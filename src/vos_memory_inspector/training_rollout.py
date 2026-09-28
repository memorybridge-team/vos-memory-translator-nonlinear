"""Load a trained checkpoint into the existing no-replay SAM 2 handoff path."""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .sam2_state import init_sam2_inference_state_without_warmup, inject_sam2_canonical_state
from .training_collection import freeze_model, model_provenance, prepare_case, run_prefix
from .training_data import load_pair, pair_id, read_manifest
from .training_runner import CheckpointTranslator
from .training_storage import write_json


def inject_checkpoint(translator, source, target_reference, predictor, inference_state):
    translated = translator.translate(source)
    # Final runtime cast uses observed target dtypes, not guessed SAM dimensions/dtypes.
    translated = translated.with_continuous(
        spatial_memory=translated.spatial_memory.to(dtype=target_reference.spatial_memory.dtype),
        object_pointer=translated.object_pointer.to(dtype=target_reference.object_pointer.dtype),
        presence_logits=translated.presence_logits,
        positional_information={"policy": "regenerate_at_target"})
    return inject_sam2_canonical_state(translated, predictor=predictor, inference_state=inference_state)


def rollout(plan_path, persistent_root, collection, checkpoint, sam2_repo, output, *,
            pair_index=0, device="cuda"):
    import json
    root, sam2_repo = Path(persistent_root).resolve(), Path(sam2_repo).resolve()
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    manifest = read_manifest(Path(collection) / "manifest.json")
    models = model_provenance(root, plan, sam2_repo)
    translator = CheckpointTranslator(checkpoint, device=device)
    if translator.identity["models"] != models or manifest["models"] != models:
        raise ValueError("runtime models differ from checkpoint/collection provenance")
    entry = manifest["pairs"][pair_index]
    source, target, case = load_pair(collection, entry, models)
    if case["pair_mode"] != "native_history" or translator.identity["pair_mode"] != case["pair_mode"]:
        raise ValueError("downstream native rollout requires native-history checkpoint and pair")
    matching = [raw for raw in plan["cases"] if pair_id(prepare_case(root, raw)) == entry["pair_id"]]
    if len(matching) != 1:
        raise ValueError("runtime plan does not uniquely match shard")
    raw = matching[0]
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor
    from .training_storage import checked_path
    from .roundtrip import _compare_future_masks

    config = plan["models"]["target"]
    predictor = freeze_model(build_sam2_video_predictor(
        config_file=config["config"], ckpt_path=str(checked_path(root, config["checkpoint"])), device=device))
    calls = []
    original = predictor.forward_image
    def counted(image):
        calls.append(tuple(image.shape))
        return original(image)
    predictor.forward_image = counted
    def collect_future(state, start):
        result = {}
        for frame, ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=start,
                max_frame_num_to_track=state["num_frames"]-start-1, reverse=False):
            if list(ids) != case["object_ids"] or masks.shape[0] != len(ids):
                raise ValueError("rollout object registry/order mismatch")
            if int(frame) in result:
                raise ValueError("duplicate rollout frame")
            result[int(frame)] = masks.detach().cpu().float()
        return result
    context = torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()
    started = time.perf_counter()
    with torch.inference_mode(), context:
        state = init_sam2_inference_state_without_warmup(predictor,
                  video_path=str(checked_path(root, raw["video_dir"])),
                  offload_video_to_cpu=True, offload_state_to_cpu=True)
        before = len(calls)
        injection = inject_checkpoint(translator, source, target, predictor, state)
        during = len(calls)-before
        if before or during:
            raise RuntimeError("past backbone call in no-replay handoff")
        start = case["switch_frame"]+1
        candidate = collect_future(state, start)
        candidate_calls = len(calls)
        del state
        # Native reference is evaluated separately; its prefix work is not handoff work.
        native = predictor.init_state(video_path=str(checked_path(root, raw["video_dir"])),
                                      offload_video_to_cpu=True, offload_state_to_cpu=True)
        run_prefix(predictor, native, raw["prompt_events"], root, case["switch_frame"])
        oracle = collect_future(native, start)
    if set(candidate) != set(range(start, case["num_frames"])) or set(candidate) != set(oracle):
        raise ValueError("rollout frame coverage mismatch")
    report = {"pair_id": entry["pair_id"], "injection": injection,
              "backbone_calls_before_injection": before, "backbone_calls_during_injection": during,
              "candidate_backbone_calls": candidate_calls,
              "mask_agreement_to_target_native": _compare_future_masks(oracle, candidate),
              "interpretation": "engineering rollout agreement, not ground-truth VOS performance",
              "wall_time_seconds_including_native_reference": time.perf_counter()-started}
    write_json(output, report)
    return report
