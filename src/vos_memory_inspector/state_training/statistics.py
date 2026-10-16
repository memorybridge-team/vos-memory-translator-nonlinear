"""한 번의 CPU streaming pass: train RMS와 고정 validation target 통계."""
from pathlib import Path
import math
import torch
from .common import require, sha256, digest, read_bound, save_bound, writer, write_json, outside_originals
from .index import read_index
from .loader import loader
from .policy import Config
from .r2 import Moments, SPEC, DATASETS, BRANCHES, Coverage
from .schedule import refs_for


def tensor_moments(tensor, branch):
    x = tensor.double()
    if branch == "spatial":
        require(x.ndim == 4, "SPATIAL_RECORD_SHAPE")
        count, axes = x.shape[0] * x.shape[2] * x.shape[3], (0, 2, 3)
    else:
        require(x.ndim == 2, "POINTER_RECORD_SHAPE")
        count, axes = x.shape[0], (0,)
    variance, mean = torch.var_mean(x, dim=axes, correction=0)
    return count, mean.cpu().tolist(), (variance * count).cpu().tolist()


def prepare(index_path, output, config=None):
    cfg = config or Config(workers=0)
    torch.set_num_threads(1)
    index = read_index(index_path)
    output = Path(output)
    outside_originals(output, index)
    binding = dict(index_sha256=sha256(index_path), split_sha256=index["identity"]["split_sha256"], r2_spec_sha256=digest(SPEC))
    with writer(output):
        if (output / "STATISTICS_READY.json").exists():
            return read_statistics(output, binding)
        require(not any((output / n).exists() for n in ("normalization.json", "validation_targets.json", "r2_spec.json")), "PARTIAL_STATS_USE_NEW_NAMESPACE")
        sums = {b: [0.0, 0] for b in BRANCHES}
        training = [e for e in index["entries"] if e["case"]["training_role"] == "train"]
        coverage = Coverage(refs_for(training))
        for batch in loader(training, cfg):
            coverage.add(batch["refs"])
            for branch in BRANCHES:
                x = batch["tensors"]["target_" + branch].double()
                sums[branch][0] += float(x.square().sum())
                sums[branch][1] += x.numel()
        coverage.finish()
        require(all(n > 0 and math.isfinite(s) for s, n in sums.values()), "TRAIN_NORMALIZATION_NONFINITE_EMPTY")
        norm = dict(schema_version="cmmt.train_only_rms.v1", binding=binding, scope="train_only", computations=1,
                    train_records=len(coverage.seen), sums=sums,
                    scales={k: math.sqrt(s / n) for k, (s, n) in sums.items()}, usage="loss scale only; inference inputs unchanged")
        targets = {}
        record_counts = {}
        for dataset in DATASETS:
            validation = [e for e in index["entries"] if e["case"]["training_role"] == "validation" and e["case"]["dataset"] == dataset]
            coverage = Coverage(refs_for(validation))
            moments = {"spatial": Moments(64), "pointer": Moments(256)}
            for batch in loader(validation, cfg):
                coverage.add(batch["refs"])
                for branch in BRANCHES:
                    moments[branch].merge(*tensor_moments(batch["tensors"]["target_" + branch], branch))
            record_counts[dataset] = coverage.finish()
            require(record_counts[dataset] > 0, "VALIDATION_DATASET_EMPTY")
            targets[dataset] = {k: m.payload() for k, m in moments.items()}
        values = dict(schema_version="cmmt.fixed_validation_targets.v1", binding=binding, records=record_counts, targets=targets)
        save_bound(output / "normalization.json", norm)
        save_bound(output / "validation_targets.json", values)
        save_bound(output / "r2_spec.json", SPEC)
        write_json(output / "STATISTICS_READY.json", dict(status="STATISTICS_READY", binding=binding,
                   files={n: sha256(output / n) for n in ("normalization.json", "validation_targets.json", "r2_spec.json")}), immutable=True)
        return read_statistics(output, binding)


def read_statistics(root, binding):
    root = Path(root)
    ready = read_bound(root / "normalization.json"), read_bound(root / "validation_targets.json"), read_bound(root / "r2_spec.json")
    from .common import read_json
    marker = read_json(root / "STATISTICS_READY.json")
    require(marker["status"] == "STATISTICS_READY" and marker["binding"] == binding and
            all(sha256(root / n) == h for n, h in marker["files"].items()), "STATISTICS_BINDING_CHANGED")
    norm, targets, spec = ready
    require(norm["binding"] == targets["binding"] == binding and norm["scope"] == "train_only" and spec == SPEC, "TRAIN_NORM_TARGET_BINDING")
    return norm, targets, spec
