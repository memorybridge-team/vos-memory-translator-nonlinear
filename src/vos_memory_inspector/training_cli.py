"""Reproducible collection/training commands, with no benchmark training path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .training_data import (PairedStateDataset, load_pair, read_manifest,
                            save_manifest, split_collection)
from .training_storage import checked_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path,
                        default=Path(os.environ.get("CMMT_VOLUME_ROOT", "/workspace")))
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-plan")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--dataset-root", required=True, type=Path)
    build.add_argument("--output", required=True)
    build.add_argument("--stage-method", choices=("symlink", "copy"), default="symlink")
    build.add_argument("--video-id", action="append")
    build.add_argument("--fit-videos", type=Path)
    build.add_argument("--dev-videos", type=Path)
    build.add_argument("--videos-per-split", type=int)
    build.add_argument("--switch-limit", type=int)
    build.add_argument("--source-checkpoint", default="checkpoints/sam2.1_hiera_small.pt")
    build.add_argument("--target-checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    collect = sub.add_parser("collect")
    collect.add_argument("--plan", required=True, type=Path)
    collect.add_argument("--collection", required=True)
    collect.add_argument("--sam2-repo", required=True, type=Path)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=7)
    verify = sub.add_parser("verify")
    verify.add_argument("--collection", required=True)
    verify.add_argument("--allow-synthetic", action="store_true")
    split = sub.add_parser("split")
    split.add_argument("--collection", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=7)
    split.add_argument("--dev-fraction", type=float, default=.2)
    split.add_argument("--fit-videos", type=Path)
    split.add_argument("--dev-videos", type=Path)
    overfit = sub.add_parser("overfit-manifest")
    overfit.add_argument("--collection", required=True)
    overfit.add_argument("--video-id", required=True)
    overfit.add_argument("--pair-mode", choices=("native_history", "controlled_same_mask"), default="native_history")
    overfit.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--collection", required=True)
    training.add_argument("--fit", required=True, type=Path)
    training.add_argument("--dev", type=Path)
    training.add_argument("--output", required=True)
    training.add_argument("--factory", default="vos_memory_inspector.training_runner:linear_factory")
    training.add_argument("--factory-kwargs", default="{}")
    training.add_argument("--pair-mode", choices=("native_history", "controlled_same_mask"), default="native_history")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--learning-rate", type=float, default=.001)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--device", default="cuda")
    training.add_argument("--record-batch-size", type=int, default=4)
    training.add_argument("--normalization", choices=("none", "fit_rms"), default="fit_rms")
    training.add_argument("--resume", type=Path)
    training.add_argument("--overfit", action="store_true")
    training.add_argument("--allow-synthetic", action="store_true")
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--plan", required=True, type=Path)
    rollout.add_argument("--collection", required=True)
    rollout.add_argument("--checkpoint", required=True, type=Path)
    rollout.add_argument("--sam2-repo", required=True, type=Path)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--pair-index", type=int, default=0)
    rollout.add_argument("--device", default="cuda")
    smoke = sub.add_parser("synthetic-smoke")
    smoke.add_argument("--output", default="synthetic-smoke")
    args = parser.parse_args()
    root = args.persistent_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    local = lambda relative: checked_path(root, relative)
    if args.command == "build-plan":
        from .training_plan import build_plan
        result = build_plan(args.manifest, args.dataset_root, root, local(args.output),
                  stage_method=args.stage_method, video_ids=args.video_id,
                  fit_videos=read_manifest(args.fit_videos) if args.fit_videos else None,
                  dev_videos=read_manifest(args.dev_videos) if args.dev_videos else None,
                  videos_per_split=args.videos_per_split, switch_limit=args.switch_limit,
                  source_checkpoint=args.source_checkpoint, target_checkpoint=args.target_checkpoint)
        print(f"planned {len(result['cases'])} native-history cases")
    elif args.command == "collect":
        from .training_collection import collect
        collect(args.plan, root, local(args.collection), args.sam2_repo, device=args.device, seed=args.seed)
    elif args.command == "verify":
        collection = local(args.collection)
        manifest = read_manifest(collection / "manifest.json")
        for entry in manifest["pairs"]:
            load_pair(collection, entry, manifest["models"], allow_synthetic=args.allow_synthetic)
        print(json.dumps({"readable_pairs": len(manifest["pairs"]),
                         "manifest_sha256": manifest["content_sha256"],
                         "shard_bytes": sum(e["bytes"] for e in manifest["pairs"])}))
    elif args.command == "split":
        manifest = read_manifest(local(args.collection) / "manifest.json")
        print(json.dumps(split_collection(manifest, local(args.output), seed=args.seed,
                     dev_fraction=args.dev_fraction,
                     fit_videos=read_manifest(args.fit_videos) if args.fit_videos else None,
                     dev_videos=read_manifest(args.dev_videos) if args.dev_videos else None)))
    elif args.command == "overfit-manifest":
        manifest = read_manifest(local(args.collection) / "manifest.json")
        entries = [e for e in manifest["pairs"] if e["case"]["video_id"] == args.video_id
                   and e["case"]["pair_mode"] == args.pair_mode]
        from .training_data import video_key
        if not entries or len({video_key(e["case"]) for e in entries}) != 1:
            raise ValueError("choose one uniquely identified video")
        save_manifest(local(args.output), {"schema_version": manifest["schema_version"],
                      "contract": manifest["contract"], "models": manifest["models"],
                      "pairs": entries, "role": "overfit", "pair_mode": args.pair_mode,
                      "parent_content_sha256": manifest["content_sha256"]})
    elif args.command == "train":
        from .training_runner import TrainConfig, train
        fit = PairedStateDataset(local(args.collection), args.fit,
                  role="overfit" if args.overfit else "fit", pair_mode=args.pair_mode,
                  allow_synthetic=args.allow_synthetic)
        dev = PairedStateDataset(local(args.collection), args.dev, role="dev",
                  pair_mode=args.pair_mode, allow_synthetic=args.allow_synthetic) if args.dev else None
        config = TrainConfig(factory=args.factory, factory_kwargs=json.loads(args.factory_kwargs),
                  epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed,
                  device=args.device, record_batch_size=args.record_batch_size,
                  normalization=args.normalization)
        train(fit, dev, local(args.output), config, resume=args.resume, overfit=args.overfit)
    elif args.command == "rollout":
        from .training_rollout import rollout
        print(json.dumps(rollout(args.plan, root, local(args.collection), args.checkpoint,
                  args.sam2_repo, local(args.output), pair_index=args.pair_index, device=args.device)))
    else:
        from .training_smoke import synthetic_smoke
        print(json.dumps(synthetic_smoke(local(args.output)), indent=2))


if __name__ == "__main__":
    main()
