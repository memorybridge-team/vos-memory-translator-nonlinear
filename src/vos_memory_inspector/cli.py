from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compatibility import compare_manifests, write_compatibility_report
from .runner import run_video_probe
from .roundtrip import (
    MaskPromptEvent,
    prepare_cross_model_case_reference,
    run_cached_baseline,
    run_cached_translator_handoff,
    run_cross_model_direct_handoff,
    run_cross_model_translator_handoff,
    run_same_checkpoint_correction_roundtrip,
    run_same_checkpoint_prompt_timeline_roundtrip,
    run_same_checkpoint_repeated_switch_roundtrip,
    run_same_checkpoint_roundtrip,
)
from .state_inspector import inspect_state, write_inspection_report
from .translators import (
    ResidualMLPStateTranslator,
    RidgeDirectPresenceTranslator,
    RidgeStateTranslator,
)


def _probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe stored and consumed SAM 2 temporal-memory tensors."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument(
        "--dump-tensor",
        action="append",
        default=[],
        help=(
            "Opt-in tensor name to save on CPU; repeat for multiple names. "
            "Without this option only statistics are written."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--canonical-state",
        type=Path,
        help="Opt-in .pt export of the continuation-oriented canonical state.",
    )
    return parser


def probe_main(argv: list[str] | None = None) -> None:
    args = _probe_parser().parse_args(argv)
    summary = run_video_probe(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        jsonl_path=args.jsonl,
        csv_path=args.csv,
        dump_dir=args.dump_dir,
        dump_tensors=tuple(args.dump_tensor),
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        allow_upstream_mismatch=args.allow_upstream_mismatch,
        seed=args.seed,
        canonical_state_path=args.canonical_state,
    )
    print(json.dumps(summary, indent=2))


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare matching rows from sequential source and target manifests."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    report = compare_manifests(args.source, args.target)
    write_compatibility_report(report, args.json, args.markdown)
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))


def state_inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensors in a nested .pt state, HF cache, or canonical state."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", help="Optional top-level mapping key to inspect")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    # State files are pickle-backed. Only load files from a trusted source.
    value = __import__("torch").load(args.input, map_location="cpu", weights_only=False)
    if args.key is not None:
        value = value[args.key]
    report = inspect_state(value)
    write_inspection_report(report, args.json, args.markdown)
    print(json.dumps(report.to_dict(), indent=2))


def roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run same-checkpoint SAM 2 export/inject continuation validation."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--active-memory-only", action="store_true")
    parser.add_argument("--num-maskmem", type=int, default=7)
    parser.add_argument("--max-obj-ptrs-in-encoder", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run_same_checkpoint_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        active_memory_only=args.active_memory_only,
        num_maskmem=args.num_maskmem,
        max_obj_ptrs_in_encoder=args.max_obj_ptrs_in_encoder,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def prompt_timeline_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run same-checkpoint SAM 2 export/inject validation for a multi-object "
            "mask-prompt timeline."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
        help="Repeat for each user mask prompt in the timeline.",
    )
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(
            frame_index=int(frame),
            object_id=int(object_id),
            mask_path=Path(mask_path),
        )
        for frame, object_id, mask_path in args.prompt_event
    ]
    report = run_same_checkpoint_prompt_timeline_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def correction_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a same-checkpoint correction after handoff or the safe "
            "Target replay fallback for a correction before handoff."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument(
        "--correction-event",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(int(frame), int(object_id), Path(mask_path))
        for frame, object_id, mask_path in args.prompt_event
    ]
    correction = MaskPromptEvent(
        int(args.correction_event[0]),
        int(args.correction_event[1]),
        Path(args.correction_event[2]),
    )
    report = run_same_checkpoint_correction_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        correction_event=correction,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def repeated_switch_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate two consecutive same-checkpoint SAM 2 handoffs."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument(
        "--switch-frames",
        nargs=2,
        type=int,
        metavar=("FIRST", "SECOND"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(int(frame), int(object_id), Path(mask_path))
        for frame, object_id, mask_path in args.prompt_event
    ]
    report = run_same_checkpoint_repeated_switch_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        switch_frames=(args.switch_frames[0], args.switch_frames[1]),
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def prepare_handoff_case_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one source prefix and target oracle once, then cache them for "
            "reuse by all handoff baselines."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", required=True, type=int)
    parser.add_argument("--switch-frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="store paired canonical states only; omit prediction masks and future GT-derived payloads",
    )
    parser.add_argument(
        "--active-memory-only",
        action="store_true",
        help="keep conditioning records and SAM 2's immediate non-conditioning memory window only",
    )
    parser.add_argument("--num-maskmem", type=int, default=7)
    parser.add_argument("--max-obj-ptrs-in-encoder", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    report = prepare_cross_model_case_reference(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        output=args.output,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        store_masks=not args.state_only,
        active_memory_only=args.active_memory_only,
        num_maskmem=args.num_maskmem,
        max_obj_ptrs_in_encoder=args.max_obj_ptrs_in_encoder,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cached_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Direct, Ridge, or residual-MLP target continuation from a "
            "prepared case cache."
        )
    )
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument(
        "--translator",
        choices=("direct", "ridge", "residual_mlp"),
        default="direct",
    )
    parser.add_argument("--translator-artifact", type=Path)
    parser.add_argument(
        "--presence-policy", choices=("direct", "ridge"), default="direct"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)

    translator = None
    translator_name = "direct_copy"
    candidate_label = "Direct Copy"
    if args.translator == "ridge":
        if args.translator_artifact is None:
            parser.error("--translator-artifact is required for Ridge")
        payload = __import__("torch").load(
            args.translator_artifact, map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("ridge"), dict):
            parser.error("translator artifact does not contain a Ridge payload")
        ridge = RidgeStateTranslator.from_payload(payload["ridge"])
        if args.presence_policy == "direct":
            translator = RidgeDirectPresenceTranslator(ridge)
            translator_name = translator.name
            candidate_label = "Ridge memory/pointer + Direct presence"
        else:
            translator = ridge
            translator_name = ridge.name
            candidate_label = "Ridge"
    elif args.translator == "residual_mlp":
        if args.translator_artifact is None:
            parser.error("--translator-artifact is required for residual_mlp")
        payload = __import__("torch").load(
            args.translator_artifact, map_location="cpu", weights_only=True
        )
        mlp_payload = payload.get("residual_mlp") if isinstance(payload, dict) else None
        if not isinstance(mlp_payload, dict):
            parser.error("translator artifact does not contain a residual_mlp payload")
        translator = ResidualMLPStateTranslator.from_payload(mlp_payload)
        translator_name = translator.name
        candidate_label = "Nonlinear Residual MLP"
    report = run_cached_translator_handoff(
        case_cache=args.case_cache,
        sam2_repo=args.sam2_repo,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        translator=translator,
        translator_name=translator_name,
        candidate_label=candidate_label,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cached_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Target Reset, Last-Mask, Replay-k, or Full Replay against a "
            "prepared SAM 2 case cache."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=("target_reset", "last_mask", "replay_k", "full_replay"),
    )
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-mask",
        type=Path,
        help="First-frame ground-truth mask; required only for full_replay.",
    )
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument(
        "--replay-frames",
        type=int,
        help=(
            "Number of target-processed prefix frames including the switch frame. "
            "Required for replay_k; replay-1 intentionally equals Last-Mask."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    if args.baseline == "replay_k" and args.replay_frames is None:
        parser.error("--replay-frames is required for replay_k")
    if args.baseline == "full_replay" and args.prompt_mask is None:
        parser.error("--prompt-mask is required for full_replay")
    report = run_cached_baseline(
        baseline=args.baseline,
        case_cache=args.case_cache,
        sam2_repo=args.sam2_repo,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        annotation_dir=args.annotation_dir,
        replay_frames=args.replay_frames,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def direct_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint-backed cross-model Direct Copy handoff."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Write browsable mask PNGs plus report.json/report.md.",
    )
    args = parser.parse_args(argv)
    report = run_cross_model_direct_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def ridge_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint-backed cross-model handoff using a saved Ridge model."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--translator-artifact", required=True, type=Path)
    parser.add_argument(
        "--presence-policy",
        choices=("direct", "ridge"),
        default="direct",
        help="Use source presence logits directly or apply the fitted Ridge head.",
    )
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)

    payload = __import__("torch").load(
        args.translator_artifact, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("ridge"), dict):
        parser.error("translator artifact does not contain a Ridge payload")
    ridge = RidgeStateTranslator.from_payload(payload["ridge"])
    if args.presence_policy == "direct":
        translator = RidgeDirectPresenceTranslator(ridge)
        translator_name = translator.name
        candidate_label = "Ridge memory/pointer + Direct presence"
    else:
        translator = ridge
        translator_name = ridge.name
        candidate_label = "Ridge"
    report = run_cross_model_translator_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        translator=translator,
        translator_name=translator_name,
        candidate_label=candidate_label,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
