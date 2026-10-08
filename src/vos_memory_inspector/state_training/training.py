"""고정 split의 fresh translator DDP 학습. epoch 경계에서만 재개한다."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import math
import os
import platform
import random
import subprocess
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import CONTRACT
from .common import require, sha256, digest, read_json, write_json, writer, outside_originals
from .checkpoint import SCHEMA, publish, latest, rng_state, restore_rng, equal
from .index import read_index
from .statistics import read_statistics
from .policy import Config, WarmupPlateau, Selection
from .model import fresh_model, lock, provenance, optimizer_groups, TensorModule
from .loader import loader
from .r2 import SPEC, DATASETS
from .schedule import schedule
from .evaluation import objective, evaluate, verify_coverage


class LimitReached(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds, *, deadline_utc=None, approval_start_utc=None, rate=None, budget=None):
        require(seconds > 0, "WALL_LIMIT_REQUIRED")
        self.end = time.time() + seconds
        self.rate, self.started = rate, None
        if deadline_utc:
            self.end = min(self.end, datetime.fromisoformat(deadline_utc.replace("Z", "+00:00")).timestamp())
        if rate is not None:
            require(rate > 0 and budget is not None and budget > 0 and approval_start_utc, "APPROVAL_RATE_BUDGET_START_REQUIRED")
            self.started = datetime.fromisoformat(approval_start_utc.replace("Z", "+00:00")).timestamp()
            self.end = min(self.end, self.started + budget / rate * 3600)

    def check(self, stage):
        if time.time() >= self.end:
            raise LimitReached(stage)

    def collective(self, device, stage):
        expired = torch.tensor(int(time.time() >= self.end), device=device)
        dist.all_reduce(expired, op=dist.ReduceOp.MAX)
        if int(expired):
            raise LimitReached(stage)

    def cost(self):
        return max(0, time.time() - self.started) * self.rate / 3600 if self.started is not None else None


def gpu_status():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip().splitlines() if r.returncode == 0 else ["UNAVAILABLE"]
    except (OSError, subprocess.TimeoutExpired):
        return ["UNAVAILABLE"]


def verify_reload(payload, sample, device):
    from ..transformer_translator import TransformerStateTranslator
    model = TransformerStateTranslator.from_payload(payload["export"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    require(equal(model.to_payload(), payload["export"]), "STRICT_MODEL_RELOAD_DIFFERENCE")
    optimizer = torch.optim.AdamW(optimizer_groups(model, payload["configuration"]["weight_decay"]), lr=payload["configuration"]["lr"])
    scheduler = WarmupPlateau(optimizer, payload["updates_per_epoch"], payload["configuration"]["lr"], payload["configuration"]["min_delta"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    require(equal(optimizer.state_dict(), payload["optimizer"]) and equal(scheduler.state_dict(), payload["scheduler"]), "OPTIMIZER_SCHEDULER_RELOAD_DIFFERENCE")
    reference = TensorModule(model).eval()
    with torch.no_grad():
        got = reference(sample["source_spatial"].to(device), sample["source_pointer"].to(device))
    require(all(torch.equal(x.cpu(), y) for x, y in zip(got, payload["reload_outputs"])), "STRICT_OUTPUT_RELOAD_DIFFERENCE")


def rank_difference(model, device):
    vector = torch.cat([p.detach().flatten() for p in model.parameters()])
    reference = vector.clone()
    dist.broadcast(reference, src=0)
    error = (vector - reference).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    require(float(error) == 0, "RANK_PARAMETERS_DIFFER")
    return float(error)


def append_log(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def train(args):
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
    cfg = Config(**read_json(args.config))
    cfg.validate(world)
    output = Path(args.output).resolve()
    outside_originals(output, read_index(args.index))
    require(args.resume or not (output / "RUN.json").exists(), "FRESH_RUN_OUTPUT_EXISTS")
    cuda = args.device == "cuda"
    if cuda:
        require(args.execute_approved and args.gpu_uuid and len(args.gpu_uuid) == world and len(set(args.gpu_uuid)) == world,
                "GPU_EXECUTION_REQUIRES_NEW_OPERATOR_APPROVAL_UUIDS")
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "DETERMINISTIC_CUBLAS_CONFIG_REQUIRED")
        require(torch.cuda.is_available() and torch.cuda.device_count() == world, "SAME_HOST_VISIBLE_GPU_COUNT")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        require(str(torch.cuda.get_device_properties(device).uuid) == args.gpu_uuid[local], "GPU_UUID_MAPPING")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_TEST_MUST_HIDE_CUDA")
        device = torch.device("cpu")
    deadline = Deadline(args.max_wall_seconds, deadline_utc=args.deadline_utc, approval_start_utc=args.approval_start_utc,
                        rate=args.pod_hourly_rate, budget=args.budget_usd)
    require(not cuda or deadline.started is not None, "CUDA_SERVER_COST_LIMIT_REQUIRED")
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=180))
    try:
        # rank 0이 writer lock을 획득한 뒤 나머지를 진행시킨다.
        with writer(output) if rank == 0 else nullcontext():
            dist.barrier()
            return run(args, output, cfg, device, rank, world, deadline)
    except Exception as exc:
        if rank == 0:
            last = read_json(output / "last_complete.json") if (output / "last_complete.json").exists() else None
            write_json(output / "LIVE_STATUS.json", dict(status="LIMIT_REACHED" if isinstance(exc, LimitReached) else "FAIL",
                       error=str(exc), last_complete=last, partial_epoch_is_complete=False,
                       resume="last complete epoch only", pod_stopped=False, approved_cost_usd=deadline.cost()))
        raise
    finally:
        dist.destroy_process_group()


def run(args, output, cfg, device, rank, world, deadline):
    index = read_index(args.index)
    outside_originals(output, index)
    require(not output.is_relative_to(Path(args.index).parent), "RUN_MUST_BE_SEPARATE_FROM_INDEX")
    bindings = dict(index_sha256=sha256(args.index), split_sha256=index["identity"]["split_sha256"], r2_spec_sha256=digest(SPEC))
    norm, targets, spec = read_statistics(args.statistics, bindings)
    source = provenance(); model_lock = lock()
    identity = dict(contract=CONTRACT, **bindings, normalization_sha256=sha256(Path(args.statistics) / "normalization.json"),
                    validation_targets_sha256=sha256(Path(args.statistics) / "validation_targets.json"),
                    model_digest=model_lock["digest"], configuration=cfg.logical(), source_package_sha256=source["package_source_sha256"], world_size=world,
                    environment=dict(torch=str(torch.__version__), python=platform.python_version(), cuda=torch.version.cuda, device=device.type))
    if args.mode == "train":
        require(device.type == "cuda" and not index["synthetic_declared"] and args.smoke_evidence, "OFFICIAL_TRAIN_REQUIRES_REAL_BOUND_SMOKE")
        smoke = read_json(args.smoke_evidence)
        require(smoke["status"] == "REAL_R2_DDP_SMOKE_PASS" and smoke["identity"] == identity and smoke["both_branches_updated"] and
                smoke["strict_reload"] and smoke["rank_parameter_max_difference"] == 0, "NEW_DATA_SOURCE_SMOKE_BINDING")
    all_train = [e for e in index["entries"] if e["case"]["training_role"] == "train"]
    all_validation = [e for e in index["entries"] if e["case"]["training_role"] == "validation"]
    training, validation = all_train, all_validation
    if args.mode == "smoke":
        training = sum(([e for e in all_train if e["case"]["dataset"] == d][:2] for d in DATASETS), [])
        validation = sum(([e for e in all_validation if e["case"]["dataset"] == d][:2] for d in DATASETS), [])
        # Smoke의 target 통계는 별도 subset에서 계산한다. 공식 stats/score와 혼동 금지.
        from .statistics import tensor_moments
        from .r2 import Moments
        smoke_targets = {}
        for d in DATASETS:
            moments = {"spatial": Moments(64), "pointer": Moments(256)}
            selected = [e for e in validation if e["case"]["dataset"] == d]
            for batch in loader(selected, Config(workers=0, microbatch=cfg.microbatch)):
                for b, m in moments.items():
                    m.merge(*tensor_moments(batch["tensors"]["target_" + b], b))
            smoke_targets[d] = {b: m.payload() for b, m in moments.items()}
        targets = dict(targets, targets=smoke_targets, records={d: sum(e["records"] for e in validation if e["case"]["dataset"] == d) for d in DATASETS})
    updates_per_epoch = math.ceil(sum(e["records"] for e in training) / cfg.global_batch)
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    model = fresh_model().to(device)
    adapter = TensorModule(model)
    ddp = DistributedDataParallel(adapter, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = WarmupPlateau(optimizer, updates_per_epoch, cfg.lr, cfg.min_delta)
    selection = Selection(cfg.min_epochs, cfg.patience, cfg.min_delta)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    history, step, start = [], 0, 1
    sample = next(iter(loader(training[:1], Config(workers=0, microbatch=1))))["tensors"]
    if args.resume:
        require(args.mode == "train", "SMOKE_RESUME_NOT_SUPPORTED")
        payload, _ = latest(output, identity)
        require(payload["mode"] == "train" and payload["world_size"] == world, "RESUME_CONTRACT_WORLD_CHANGED")
        verify_reload(payload, sample, device)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        selection.load_state_dict(payload["selection"])
        history, step, start, initial = payload["history"], payload["optimizer_step"], payload["epoch"] + 1, payload["initial_weights"]
        recorded = read_json(output / "history.json") if (output / "history.json").exists() else []
        for row in history:
            old = next((h for h in recorded if h["epoch"] == row["epoch"]), None)
            if old is not None:
                require(all(equal(v, old.get(k)) for k, v in row.items()), "RESUME_HISTORY_DIFFERENCE")
                row.update(old)
            marker = read_json(str(output / row["checkpoint"]) + ".complete.json")
            row["checkpoint_sha256"] = marker["sha256"]
        restore_rng(payload["rank_rng"][rank])
    else:
        require(not (output / "RUN.json").exists() and not list((output / "checkpoints").glob("*.pt")), "FRESH_RUN_ONLY_NO_OLD_WARMSTART")
    if rank == 0:
        write_json(output / "RUN.json", dict(identity=identity, source=source, configuration=asdict(cfg), mode=args.mode,
                   SAM2="not instantiated; immutable cached Small/Base+ state only", fresh_initialization=not args.resume,
                   input_index=str(Path(args.index).resolve()), statistics=str(Path(args.statistics).resolve()),
                   accumulation=cfg.global_batch // (world * cfg.microbatch), shuffle="python_random_case_permutation_seed_plus_epoch_v1"))
    reason = "max_epochs"
    started = time.perf_counter()
    last_entry = None
    effective_max = min(cfg.max_epochs, 2) if args.mode == "smoke" else cfg.max_epochs
    if selection.last_epoch > cfg.min_epochs and selection.bad_checks >= cfg.patience:
        effective_max = start - 1
        reason = "validation_r2_early_stopping_already_satisfied"
    for epoch in range(start, effective_max + 1):
        deadline.collective(device, "epoch_start")
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        assigned, windows = schedule(training, world, cfg.global_batch, seed=cfg.seed, epoch=epoch)
        batches = iter(loader(assigned[rank], cfg, cuda=device.type == "cuda", window_counts=[w[rank] for w in windows]))
        totals = torch.zeros(5, dtype=torch.float64, device=device)
        refs = []
        gradient = {"spatial": 0.0, "pointer": 0.0}
        ddp.train()
        for window in windows:
            deadline.collective(device, "optimizer_window")
            nlocal, total = window[rank], sum(window)
            iterations = max(1, math.ceil(nlocal / cfg.microbatch))
            optimizer.zero_grad(set_to_none=True)
            for i in range(iterations):
                with ddp.no_sync() if i < iterations - 1 else nullcontext():
                    if nlocal:
                        batch = next(batches)
                        n = len(batch["refs"]); refs.extend(batch["refs"])
                        t = {k: v.to(device, non_blocking=True) for k, v in batch["tensors"].items()}
                        s, p = ddp(t["source_spatial"], t["source_pointer"])
                        metrics, loss = objective(s, p, t["target_spatial"], t["target_pointer"], norm["scales"])
                        totals[:4] += metrics.detach().double() * n; totals[4] += n
                        (loss * (world * n / total)).backward()
                    else:
                        # empty tail rank도 동일한 DDP collective에 양 branch를 참여시킨다.
                        s, p = ddp(sample["source_spatial"].to(device), sample["source_pointer"].to(device))
                        (s.sum() * 0 + p.sum() * 0).backward()
            for branch in gradient:
                grads = [p.grad for name, p in model.named_parameters() if name.startswith(branch + ".") and p.grad is not None]
                require(grads and all(bool(torch.isfinite(g).all()) for g in grads), "NONFINITE_OR_MISSING_BRANCH_GRADIENT")
                gradient[branch] = max(gradient[branch], math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm, error_if_nonfinite=True)
            optimizer.step(); scheduler.after_update(); step += 1
            if rank == 0 and (step % cfg.log_every == 0 or window is windows[-1]):
                write_json(output / "LIVE_STATUS.json", dict(status="TRAINING", epoch=epoch, optimizer_step=step,
                    complete_epochs=len(history), lr=optimizer.param_groups[0]["lr"], gradient_norm=gradient,
                    latest_complete=last_entry, gpu=gpu_status() if device.type == "cuda" else [],
                    approved_cost_usd=deadline.cost(), effective_global_batch=total, planned_global_batch=cfg.global_batch))
        records, rank_counts = verify_coverage(refs, training, world)
        dist.all_reduce(totals)
        require(int(totals[4]) == records and bool(torch.isfinite(totals).all()), "TRAIN_METRIC_RECORD_COVERAGE")
        fit = {k: float(totals[i] / records) for i, k in enumerate(("spatial_mse", "pointer_mse", "normalized_spatial", "normalized_pointer"))}
        fit["loss"] = fit["normalized_spatial"] + fit["normalized_pointer"]
        fit_seconds = time.perf_counter() - epoch_start
        if rank == 0:
            write_json(output / "LIVE_STATUS.json", dict(status="VALIDATING", epoch=epoch, optimizer_step=step, complete_epochs=len(history), train=fit))
        result = evaluate(adapter, validation, cfg, targets, device, rank, world, deadline)
        scheduler.after_validation(result["validation_r2_score"])
        promote, stop = selection.observe(epoch, result["validation_r2_score"])
        difference = rank_difference(model, device)
        changes = {b: any(not torch.equal(initial[n], p.detach().cpu()) for n, p in model.named_parameters() if n.startswith(b + ".")) for b in ("spatial", "pointer")}
        require(all(changes.values()) and all(x > 0 for x in gradient.values()), "BOTH_BRANCH_UPDATE_REQUIRED")
        rank_rng = [None] * world; dist.all_gather_object(rank_rng, rng_state())
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        peaks = [None] * world; dist.all_gather_object(peaks, peak)
        row = dict(epoch=epoch, optimizer_step=step, train=fit, validation=result, gradient_norm=gradient,
                   lr=optimizer.param_groups[0]["lr"], best_epoch=selection.best_epoch, best_validation_r2=selection.best_score,
                   early_stopping=selection.state_dict(), fit_records=records, rank_fit_records=rank_counts,
                   fit_seconds=fit_seconds, fit_records_per_second=records / fit_seconds,
                   optimizer_updates_per_second=len(windows) / fit_seconds,
                   rank_parameter_max_difference=difference, both_branches_updated=changes,
                   peak_vram_bytes_per_rank=peaks,
                   checkpoint=f"checkpoints/epoch-{epoch:05d}.pt")
        history.append(row)
        if rank == 0:
            adapter.eval()
            with torch.no_grad():
                reference = adapter(sample["source_spatial"].to(device), sample["source_pointer"].to(device))
            payload = dict(schema_version=SCHEMA, identity=identity, epoch=epoch, optimizer_step=step,
                           model={n: t.detach().cpu() for n, t in model.state_dict().items()}, export=model.to_payload(),
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), selection=selection.state_dict(),
                           rank_rng=rank_rng, world_size=world, mode=args.mode, history=history, initial_weights=initial,
                           configuration=asdict(cfg), updates_per_epoch=updates_per_epoch, model_lock=model_lock,
                           normalization=norm, source=source, reload_outputs=[x.cpu() for x in reference])
            entry = publish(output / row["checkpoint"], payload, lambda saved: verify_reload(saved, sample, device))
            last_entry = dict(entry, path=str(output / row["checkpoint"]))
            row["checkpoint_sha256"] = entry["sha256"]
            row["wall_seconds_including_checkpoint"] = time.perf_counter() - epoch_start
            row["gpu"] = gpu_status() if device.type == "cuda" else []
            row["peak_vram_bytes_rank0"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            write_json(output / "last_complete.json", last_entry)
            if promote:
                selected_path = output / f"checkpoints/epoch-{selection.best_epoch:05d}.pt"
                selected_entry = dict(read_json(str(selected_path) + ".complete.json"), path=str(selected_path))
                write_json(output / "best_validation_r2.json", dict(epoch=selection.best_epoch, score=selection.best_score, checkpoint=selected_entry,
                           scope="diagnostic_subset" if args.mode == "smoke" else "full_validation", selection_metric="validation_r2_score"))
            write_json(output / "history.json", history)
            append_log(output / "epochs.jsonl", row)
            remaining_seconds = (effective_max - epoch) * sum(h["wall_seconds_including_checkpoint"] for h in history if "wall_seconds_including_checkpoint" in h) / max(1, sum("wall_seconds_including_checkpoint" in h for h in history))
            write_json(output / "LIVE_STATUS.json", dict(status="EPOCH_COMPLETE", complete_epochs=epoch, optimizer_step=step,
                last=row, latest_complete=last_entry, best_epoch=selection.best_epoch, best_validation_r2=selection.best_score,
                eta_seconds=remaining_seconds, approved_cost_usd=deadline.cost(), checkpoint_ready=True))
            print(json.dumps(dict(event="EPOCH_COMPLETE", **row), ensure_ascii=False, allow_nan=False), flush=True)
        # rank 0의 verification은 RNG를 소비한다. 저장한 epoch boundary로 모두 복원한다.
        restore_rng(rank_rng[rank])
        dist.barrier()
        if stop:
            reason = "validation_r2_early_stopping"
            break
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            reason = "operator_epoch_boundary_stop"
            break
    if rank == 0:
        status = dict(status=("REAL_R2_DDP_SMOKE_PASS" if device.type == "cuda" else "CPU_SYNTHETIC_R2_SMOKE_PASS" if index["synthetic_declared"] else "CPU_REAL_CACHE_R2_SMOKE_PASS") if args.mode == "smoke" else "TRAINING_STOPPED_AT_COMPLETE_EPOCH",
                      identity=identity, complete_epochs=selection.last_epoch, optimizer_step=step, stop_reason=reason,
                      best_epoch=selection.best_epoch, best_validation_r2=selection.best_score, both_branches_updated=True,
                      strict_reload=True, rank_parameter_max_difference=0, seconds=time.perf_counter() - started,
                      approved_cost_usd=deadline.cost(), research_gates_passed=False, JF_evaluation=False, SAM2_updated=False)
        write_json(output / "STATUS.json", status)
        if args.mode == "train" and reason != "operator_epoch_boundary_stop" and not (output / "DELIVERY_ARCHIVE_READY.json").exists():
            from .delivery import export_delivery
            export_delivery(output, args.index, args.statistics)
        write_json(output / "LIVE_STATUS.json", status)
    dist.barrier()
