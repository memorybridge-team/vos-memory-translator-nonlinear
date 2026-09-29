"""향후 승인된 free GPU에서만 실행할 bounded same-checkpoint memory-policy gate."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from .collection_contract import jpeg_map, memory_policy, apply_memory_policy, require
from .training_storage import sha256, write_json


def parser():
    p = argparse.ArgumentParser(description="Base+ full/selected history의 제한된 no-replay continuation gate")
    for key in ("sam2-repo", "checkpoint", "video-dir", "prompt-mask", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--prompt-frame-index", type=int, required=True)
    p.add_argument("--switch-frame", type=int, required=True)
    p.add_argument("--object-id", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--future-frames", type=int, default=3)
    p.add_argument("--max-video-frames", type=int, default=1000)
    p.add_argument("--max-input-bytes", type=int, default=1000000000)
    p.add_argument("--max-wall-seconds", type=float, default=300)
    p.add_argument("--max-cuda-bytes", type=int, default=8589934592)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--require-exact", action="store_true", help="exact 동등성을 사전에 요구하는 실험에만 지정")
    p.add_argument("--execute", action="store_true", help="현재 endpoint/free GPU/시간/비용에 대한 운영자 승인 후에만 지정")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def execute(args):
    import torch
    from .upstream import verify_sam2_checkout
    from .runner import load_binary_prompt
    from .training_collection import freeze_model, CONFIGS
    from .roundtrip import _collect_prefix_reference, _seed_everything, _compare_future_masks
    from .sam2_state import init_sam2_inference_state_without_warmup, inject_sam2_canonical_state
    require(torch.device(args.device).type == "cuda", "GPU_DEVICE_REQUIRED")
    upstream = verify_sam2_checkout(args.sam2_repo)
    sys.path.insert(0, str(args.sam2_repo.resolve()))
    from sam2.build_sam import build_sam2_video_predictor
    device = torch.device(args.device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., args.max_cuda_bytes/total), device)
    torch.cuda.reset_peak_memory_stats(device)
    _seed_everything(args.seed)
    predictor = freeze_model(build_sam2_video_predictor(config_file=CONFIGS["target"],
                 ckpt_path=str(args.checkpoint.resolve()), device=args.device))
    require(predictor.memory_temporal_stride_for_eval == 1, "UNSUPPORTED_ACTIVE_STRIDE")
    started = time.perf_counter()
    with torch.inference_mode():
        native = predictor.init_state(video_path=str(args.video_dir.resolve()),
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        trace = {}
        full, _ = _collect_prefix_reference(predictor, native,
            mask=load_binary_prompt(args.prompt_mask, args.object_id), object_id=args.object_id,
            switch_frame=args.switch_frame, prompt_frame_index=args.prompt_frame_index,
            capture_masks=False, trace=trace)
        policy = memory_policy("active_window_v1", predictor.num_maskmem, predictor.max_obj_ptrs_in_encoder)
        selected = apply_memory_policy(full, policy)
        del native
        futures, calls = {}, []
        original = predictor.forward_image
        def counted(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        predictor.forward_image = counted
        try:
            for name, canonical in (("full_history", full), ("active_window_v1", selected)):
                calls.clear()
                state = init_sam2_inference_state_without_warmup(predictor, video_path=str(args.video_dir.resolve()),
                                 offload_video_to_cpu=True, offload_state_to_cpu=True)
                inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
                require(not calls, "PAST_BACKBONE_CALL")
                output = {}
                first = args.switch_frame+1
                for f, _, masks in predictor.propagate_in_video(state, start_frame_idx=first,
                     max_frame_num_to_track=args.future_frames-1, reverse=False):
                    require(first <= int(f) < first+args.future_frames, "FUTURE_BOUNDARY")
                    output[int(f)] = masks.detach().cpu().float()
                require(len(output) == args.future_frames, "FUTURE_COVERAGE")
                futures[name] = output
                del state
        finally:
            predictor.forward_image = original
    full_masks, chosen = futures["full_history"], futures["active_window_v1"]
    exact = all(torch.equal(full_masks[f], chosen[f]) for f in full_masks)
    report = {"upstream_commit": upstream, "checkpoint_sha256": sha256(args.checkpoint),
        "policy": policy, "selection": selected.metadata["memory_selection"], "prefix_trace": trace,
        "future_frames": args.future_frames, "full_vs_selected_exact": exact,
        "agreement": _compare_future_masks(full_masks, chosen), "injection_past_backbone_calls": 0,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device), "wall_seconds": time.perf_counter()-started,
        "interpretation": "same-checkpoint memory 정책 gate; translator/GT VOS 성능 증거 아님"}
    write_json(args.output, report)
    if args.require_exact:
        require(exact, "POLICY_PARITY_NOT_EXACT", "결과 보존 후 명시적 연구 판단 필요")
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    frames, _ = jpeg_map(args.video_dir, hash_pixels=False)
    require(0 <= args.prompt_frame_index <= args.switch_frame and
            args.switch_frame+args.future_frames < len(frames), "SMOKE_BOUNDS")
    require(1 <= args.future_frames <= 10 and len(frames) <= args.max_video_frames and
            sum(p.stat().st_size for p in frames) <= args.max_input_bytes and args.max_wall_seconds > 0 and
            args.max_cuda_bytes > 0, "SMOKE_RESOURCE_LIMIT")
    if not args.execute:
        print(json.dumps({"state": "dry_run_only", "video_frames": len(frames),
            "future_frames": args.future_frames, "limits": {"wall_seconds": args.max_wall_seconds,
            "cuda_bytes": args.max_cuda_bytes, "input_bytes": args.max_input_bytes},
            "approval_required": "현재 endpoint·free GPU assignment·시간·비용 한도; 배포 실행 권한 없음"}, ensure_ascii=False))
    elif args.worker:
        print(json.dumps(execute(args), ensure_ascii=False))
    else:
        command = [sys.executable, "-m", "vos_memory_inspector.memory_policy_smoke", *(sys.argv[1:] if argv is None else argv), "--worker"]
        subprocess.run(command, check=True, timeout=args.max_wall_seconds)


if __name__ == "__main__":
    main()
