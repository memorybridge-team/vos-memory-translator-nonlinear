"""실제 state 예측의 full validation. VOS rollout/J&F는 실행하지 않는다."""
import time
import torch
import torch.distributed as dist
from .common import require
from .loader import loader
from .schedule import schedule, refs_for
from .r2 import DATASETS, BRANCHES, Coverage, summarize, score


def objective(s, p, ts, tp, scales):
    require(s.shape == ts.shape and p.shape == tp.shape, "LOSS_PAIR_ALIGNMENT")
    spatial = (s.float() - ts.float()).square().flatten(1).mean()
    pointer = (p.float() - tp.float()).square().mean()
    normalized_s = spatial / max(scales["spatial"] ** 2, 1e-12)
    normalized_p = pointer / max(scales["pointer"] ** 2, 1e-12)
    return torch.stack((spatial, pointer, normalized_s, normalized_p)), normalized_s + normalized_p


def verify_coverage(refs, entries, world):
    groups = [None] * world
    dist.all_gather_object(groups, refs)
    coverage = Coverage(refs_for(entries))
    for values in groups:
        coverage.add(values)
    return coverage.finish(), [len(values) for values in groups]


def evaluate(adapter, entries, cfg, targets, device, rank, world, deadline):
    adapter.eval()
    assigned, _ = schedule(entries, world, cfg.global_batch, shuffle=False)
    errors = {d: {"spatial": torch.zeros(64, dtype=torch.float64, device=device),
                  "pointer": torch.zeros(256, dtype=torch.float64, device=device)} for d in DATASETS}
    counts = torch.zeros(2, dtype=torch.int64, device=device)
    refs = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader(assigned[rank], cfg, cuda=device.type == "cuda"):
            deadline.check("validation_batch")
            refs.extend(batch["refs"])
            t = {k: v.to(device, non_blocking=True) for k, v in batch["tensors"].items()}
            s, p = adapter(t["source_spatial"], t["source_pointer"])
            require(bool(torch.isfinite(s).all() and torch.isfinite(p).all()), "NONFINITE_VALIDATION_PREDICTION")
            for i, dataset in enumerate(DATASETS):
                selected = [j for j, d in enumerate(batch["datasets"]) if d == dataset]
                if not selected:
                    continue
                counts[i] += len(selected)
                errors[dataset]["spatial"] += (s[selected].double() - t["target_spatial"][selected].double()).square().sum(dim=(0, 2, 3))
                errors[dataset]["pointer"] += (p[selected].double() - t["target_pointer"][selected].double()).square().sum(dim=0)
    _, per_rank = verify_coverage(refs, entries, world)
    dist.all_reduce(counts)
    metrics = {}
    for i, dataset in enumerate(DATASETS):
        require(int(counts[i]) == targets["records"][dataset] > 0, "TARGET_STAT_RECORD_BINDING")
        metrics[dataset] = {}
        for branch in BRANCHES:
            dist.all_reduce(errors[dataset][branch])
            metrics[dataset][branch] = summarize(targets["targets"][dataset][branch], errors[dataset][branch].cpu().tolist())
    return dict(datasets=metrics, validation_r2_score=score(metrics), records=int(counts.sum()),
                per_rank_records=per_rank, seconds=time.perf_counter() - started, coverage="FULL_EXACT_NO_DUPLICATE")
