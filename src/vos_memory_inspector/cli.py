from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paired_experiment import (
    load_case_cache_pair,
    load_canonical_state,
    run_paired_experiment,
    run_synthetic_experiment,
)
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
from .translators import (
    ResidualMLPStateTranslator,
    RidgeDirectPresenceTranslator,
    RidgeStateTranslator,
)


def synthetic_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an explicitly synthetic paired-state translator smoke test."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_synthetic_experiment(
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(report, indent=2))


def paired_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit translators from paired canonical states and evaluate held-out pairs."
    )
    parser.add_argument("--train-source", action="append", type=Path, default=[])
    parser.add_argument("--train-target", action="append", type=Path, default=[])
    parser.add_argument("--test-source", action="append", type=Path, default=[])
    parser.add_argument("--test-target", action="append", type=Path, default=[])
    parser.add_argument("--train-case-cache", action="append", type=Path, default=[])
    parser.add_argument("--test-case-cache", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument(
        "--spatial-samples-per-pair",
        type=int,
        help="Sample this many valid spatial tokens per state and epoch.",
    )
    parser.add_argument(
        "--translator",
        action="append",
        choices=("direct", "ridge", "linear", "residual_mlp"),
        default=[],
        help="Translator to run; repeat as needed. Default: all.",
    )
    args = parser.parse_args(argv)
    if len(args.train_source) != len(args.train_target):
        parser.error("provide the same number of --train-source/--train-target")
    if len(args.test_source) != len(args.test_target):
        parser.error("provide the same number of --test-source/--test-target")
    train_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.train_source, args.train_target, strict=True)
    ]
    train_pairs.extend(load_case_cache_pair(path) for path in args.train_case_cache)
    test_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.test_source, args.test_target, strict=True)
    ]
    test_pairs.extend(load_case_cache_pair(path) for path in args.test_case_cache)
    if not train_pairs or not test_pairs:
        parser.error("provide at least one train pair and one test pair")
    report = run_paired_experiment(
        train_pairs,
        test_pairs,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
        translator_names=tuple(args.translator) or None,
        device=args.device,
        spatial_samples_per_pair=args.spatial_samples_per_pair,
    )
    print(json.dumps(report, indent=2))


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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
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
    parser.add_argument(
        "--translator",
        choices=("direct", "ridge", "residual_mlp"),
        default="direct",
    )
    parser.add_argument("--translator-artifact", type=Path)
    parser.add_argument(
        "--presence-policy", choices=("direct", "ridge"), default="direct"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    translator = None
    translator_name = "direct_copy"
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
        else:
            translator = ridge
            translator_name = ridge.name
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
    report = run_cached_translator_handoff(
        case_cache=args.case_cache,
        sam2_repo=args.sam2_repo,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        translator=translator,
        translator_name=translator_name,
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
    parser.add_argument(
        "--replay-frames",
        type=int,
        help=(
            "Number of target-processed prefix frames including the switch frame. "
            "Required for replay_k; replay-1 intentionally equals Last-Mask."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
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
        replay_frames=args.replay_frames,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
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
    else:
        translator = ridge
        translator_name = ridge.name
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
        translator=translator,
        translator_name=translator_name,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
