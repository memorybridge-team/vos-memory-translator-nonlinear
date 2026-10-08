"""help/metadata 명령은 Torch를 import하지 않는다. 실제 tensor 연산은 RunPod에서 수행한다."""
import argparse
import json
from pathlib import Path
from .common import require, read_json, read_bound, save_bound, write_json, writer


def parser():
    p = argparse.ArgumentParser(description="고정 MOSE 90/5/5 + MOSE/LVOS cached state MSE 학습 · validation R² 선정")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inventory", help="원본 official 영상 목록과 RGB directory metadata 대조; cache를 입력으로 사용하지 않음")
    q.add_argument("--dataset", choices=("MOSEv2", "LVOSv2"), required=True)
    q.add_argument("--official-split", choices=("train", "valid"), required=True)
    q.add_argument("--video-root", required=True, type=Path)
    q.add_argument("--official-list", required=True, type=Path)
    q.add_argument("--groups", type=Path, help="확인한 video_id:original_group JSON; 미확인 관계는 UNKNOWN")
    q.add_argument("--output", required=True, type=Path)
    q = sub.add_parser("freeze", help="원본 MOSE official train inventory의 영구 split 생성/동일 입력 재사용")
    q.add_argument("--inventory", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--seed", default=7, type=int)
    q = sub.add_parser("plan", help="tensor를 읽지 않는 새 역할/제외/저장 위치 dry-run")
    q.add_argument("--datasets", required=True, type=Path)
    q.add_argument("--split", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q = sub.add_parser("index", help="RunPod CPU에서 실제 tensor·SHA·시간/객체/slot 대응 검사; 원본 수정 없음")
    q.add_argument("--datasets", required=True, type=Path)
    q.add_argument("--split", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--operator-completed", required=True)
    q.add_argument("--stable-seconds", default=60, type=float)
    q = sub.add_parser("statistics", help="RunPod CPU streaming: train-only RMS와 validation target 통계 한 번 계산")
    q.add_argument("--index", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)
    q.add_argument("--config", type=Path)
    q = sub.add_parser("train", help="torchrun single-host 1/2/4 rank; 기본 CPU smoke; 공식 CUDA 학습은 별도 승인 필요")
    for n in ("index", "statistics", "config", "output"):
        q.add_argument("--" + n, required=True, type=Path)
    q.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    q.add_argument("--resume", action="store_true")
    q.add_argument("--smoke-evidence", type=Path)
    q.add_argument("--max-wall-seconds", type=float, required=True)
    q.add_argument("--stop-after-epoch", type=int)
    q.add_argument("--execute-approved", action="store_true")
    q.add_argument("--gpu-uuid", action="append")
    q.add_argument("--deadline-utc")
    q.add_argument("--approval-start-utc")
    q.add_argument("--pod-hourly-rate", type=float)
    q.add_argument("--budget-usd", type=float)
    q = sub.add_parser("monitor", help="LIVE_STATUS.json read-only 출력; 작업 제어 없음")
    q.add_argument("--run", required=True, type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        for v in vars(args).values():
            require(not isinstance(v, Path) or ("<" not in str(v) and "PLACEHOLDER" not in str(v)), "UNFILLED_PATH_PLACEHOLDER")
        if args.command == "inventory":
            from .splits import inventory
            require(not args.output.resolve().is_relative_to(args.video_root.resolve()), "INVENTORY_OUTPUT_INSIDE_RAW_RGB")
            value = inventory(args.dataset, args.official_split, args.video_root, args.official_list, args.groups)
            with writer(args.output.parent):
                save_bound(args.output, value)
        elif args.command == "freeze":
            from .splits import freeze
            value = freeze(read_bound(args.inventory), args.output, args.seed)
        elif args.command == "plan":
            from .splits import read_split
            from .index import plan
            items, sources, _ = plan(read_json(args.datasets), read_split(args.split))
            value = dict(status="DRY_RUN_NOT_TENSOR_PASS", cases=items, sources=sources)
            write_json(args.output, value, immutable=True)
        elif args.command == "index":
            require(__import__("os").environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_INDEX_MUST_HIDE_CUDA")
            from .index import build_index
            value = build_index(args.datasets, args.split, args.output, operator_completed=args.operator_completed, stable_seconds=args.stable_seconds)
        elif args.command == "statistics":
            require(__import__("os").environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_STATS_MUST_HIDE_CUDA")
            from .statistics import prepare
            from .policy import Config
            value = prepare(args.index, args.output, Config(**read_json(args.config)) if args.config else None)
        elif args.command == "train":
            require(args.mode != "smoke" or args.max_wall_seconds <= 600, "SMOKE_MAX_600_SECONDS")
            from .training import train
            train(args)
            return 0
        else:
            value = read_json(args.run / "LIVE_STATUS.json")
        print(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"중단: {type(exc).__name__}: {exc}")
        return 3 if type(exc).__name__ == "LimitReached" else 2


if __name__ == "__main__":
    raise SystemExit(main())
