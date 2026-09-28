"""Offline state supervision with explicit factories and epoch-boundary resume."""

from __future__ import annotations

import importlib
import inspect
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .state_schema import CanonicalState, StateSpec, validate_paired_state_contract
from .training_data import (CONTRACT, PairedStateDataset, TENSORS, check_disjoint,
                            validate_pair)
from .training_storage import (ExclusiveWriter, content_hash, read_checked,
                               sha256, write_json, write_tensor_file)
from .translators import LinearStateTranslator

CHECKPOINT_SCHEMA = "cmmt.training_checkpoint.v1"


def linear_factory(*, source_spec: StateSpec, target_spec: StateSpec):
    return LinearStateTranslator(source_spec, target_spec)


def create_translator(factory: str, source_spec: StateSpec, target_spec: StateSpec,
                      kwargs: dict) -> nn.Module:
    module_name, function_name = factory.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    model = function(source_spec=source_spec, target_spec=target_spec, **kwargs)
    if not isinstance(model, nn.Module) or not any(p.requires_grad for p in model.parameters()):
        raise TypeError("translator factory must return a trainable torch.nn.Module")
    return model


def factory_hash(factory: str) -> str:
    module_name, _ = factory.split(":", 1)
    path = inspect.getsourcefile(importlib.import_module(module_name))
    if not path:
        raise ValueError("factory module must have a source file for provenance")
    return sha256(path)


def architecture_hash(model: nn.Module) -> str:
    files = {}
    for module in model.modules():
        try:
            path = inspect.getsourcefile(type(module))
        except TypeError:
            path = None
        if path:
            key = type(module).__module__ + "." + type(module).__qualname__
            files[key] = sha256(path)
    return content_hash(files)


def guarded_forward(model: nn.Module, source: CanonicalState, target_spec: StateSpec):
    result = model(source)
    if not isinstance(result, CanonicalState):
        raise TypeError("forward(CanonicalState) must return CanonicalState")
    validate_paired_state_contract(source, result)
    if result.spec != target_spec:
        raise ValueError("translator output spec differs from target spec")
    if not torch.allclose(source.presence_logits.cpu(), result.presence_logits.cpu(),
                          rtol=0, atol=0, equal_nan=True):
        raise ValueError("presence_logits must remain diagnostic-only")
    if result.positional_information.get("policy") != "regenerate_at_target":
        raise ValueError("translator must request target positional regeneration")
    for name in ("spatial_memory", "object_pointer"):
        if not torch.isfinite(getattr(result, name)[result.validity]).all():
            raise ValueError(f"non-finite translator output: {name}")
    return result


def move_state(state, device):
    moved = replace(state, **{name: getattr(state, name).to(device) for name in TENSORS})
    # Padding must not contaminate matrix gradients (NaN * zero is still NaN).
    return replace(moved,
        spatial_memory=moved.spatial_memory.masked_fill(~moved.validity[..., None, None, None], 0),
        object_pointer=moved.object_pointer.masked_fill(~moved.validity[..., None], 0))


def chunks(source, target, size):
    records = source.validity.shape[2]
    size = size or records
    for start in range(0, records, size):
        def slice_state(state):
            return replace(state, **{name: getattr(state, name)[:, :, start:start+size]
                                     for name in TENSORS})
        a, b = slice_state(source), slice_state(target)
        if a.validity.any():
            yield a, b


def component_loss(prediction, target, scales):
    valid = validate_paired_state_contract(prediction, target)
    losses = {}
    for name in ("spatial_memory", "object_pointer"):
        predicted = getattr(prediction, name)[valid]
        expected = getattr(target, name)[valid].to(predicted.device)
        losses[name] = ((predicted.float() - expected.float()) ** 2).mean()
    return losses, sum(losses[name] / scales[name]**2 for name in losses)


def fit_scales(dataset, normalization):
    if normalization == "none":
        return {"spatial_memory": 1.0, "object_pointer": 1.0}
    totals = {name: [0.0, 0] for name in ("spatial_memory", "object_pointer")}
    for source, target, _ in dataset:
        validate_pair(source, target)
        for name, value in totals.items():
            tensor = getattr(target, name)[target.validity].double()
            value[0] += tensor.square().sum().item()
            value[1] += tensor.numel()
    return {name: max((total/count)**.5, 1e-6) for name, (total, count) in totals.items()}


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value["python"])
    state = value["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def code_provenance():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                                cwd=Path(__file__).resolve().parents[2])
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    source_hashes = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
    return {"commit": git("rev-parse", "HEAD"), "tracked_diff_sha256": content_hash(git("diff", "HEAD")),
            "package_source_sha256": content_hash(source_hashes),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                         if torch.cuda.is_available() else []}


@dataclass
class TrainConfig:
    factory: str = "vos_memory_inspector.training_runner:linear_factory"
    factory_kwargs: dict | None = None
    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 7
    device: str = "cpu"
    record_batch_size: int = 4
    normalization: str = "fit_rms"


def read_checkpoint(path):
    path = Path(path)
    if path.name in {"latest.json", "best.json"}:
        entry = json.loads(path.read_text(encoding="utf-8"))
        return read_checked(path.parent, entry)
    sidecar = path.with_suffix(path.suffix + ".json")
    entry = json.loads(sidecar.read_text(encoding="utf-8"))
    if entry["path"] != path.name:
        raise ValueError("checkpoint sidecar path mismatch")
    return read_checked(path.parent, entry)


def _run_pass(model, dataset, config, scales, target_spec, optimizer=None, order=None):
    totals = {"spatial_memory_mse": 0.0, "object_pointer_mse": 0.0, "normalized_loss": 0.0}
    records = 0
    model.train(optimizer is not None)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for index in (range(len(dataset)) if order is None else order):
            source, target, _ = dataset[index]
            for a, b in chunks(source, target, config.record_batch_size):
                a, b = move_state(a, config.device), move_state(b, config.device)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                prediction = guarded_forward(model, a, target_spec)
                loss_parts, loss = component_loss(prediction, b, scales)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss")
                if optimizer is not None:
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError("non-finite translator gradients")
                    optimizer.step()
                count = a.valid_record_count()
                records += count
                for name, value in loss_parts.items():
                    totals[name + "_mse"] += value.detach().item()*count
                totals["normalized_loss"] += loss.detach().item()*count
    return {**{name: value/records for name, value in totals.items()}, "valid_records": records}


def train(fit, dev, output, config: TrainConfig, *, resume=None, overfit=False):
    if config.epochs < 1 or config.learning_rate <= 0 or config.record_batch_size < 0:
        raise ValueError("invalid training configuration")
    if config.normalization not in {"none", "fit_rms"}:
        raise ValueError("normalization must be none or fit_rms")
    if overfit:
        from .training_data import video_key
        if fit.manifest["role"] != "overfit" or len({video_key(e["case"]) for e in fit.entries}) != 1 or dev is not None:
            raise ValueError("overfit smoke requires one video and no dev set")
    else:
        if dev is None:
            raise ValueError("training requires a video-disjoint dev set")
        check_disjoint(fit, dev)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    first_source, first_target, _ = fit[0]
    source_spec, target_spec = first_source.spec, first_target.spec
    for dataset in (fit, dev):
        if dataset is not None:
            for source, target, _ in dataset:
                if source.spec != source_spec or target.spec != target_spec:
                    raise ValueError("dataset changes source/target tensor spec")
    config.factory_kwargs = config.factory_kwargs or {}
    model = create_translator(config.factory, source_spec, target_spec, config.factory_kwargs).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    immutable_config = {key: value for key, value in asdict(config).items() if key != "epochs"}
    identity = {"contract": CONTRACT, "config": immutable_config,
                "factory_source_sha256": factory_hash(config.factory),
                "architecture_source_sha256": architecture_hash(model),
                "fit_sha256": fit.manifest["content_sha256"],
                "dev_sha256": dev.manifest["content_sha256"] if dev else None,
                "source_spec": source_spec.to_dict(), "target_spec": target_spec.to_dict(),
                "models": fit.manifest["models"], "pair_mode": fit.manifest["pair_mode"],
                "overfit": overfit}
    output = Path(output)
    with ExclusiveWriter(output):
        history, start_epoch, best_loss = [], 0, float("inf")
        if resume:
            payload = read_checkpoint(resume)
            if payload["schema_version"] != CHECKPOINT_SCHEMA or payload["identity"] != identity:
                raise ValueError("resume contract/data/factory/config differs")
            if payload["provenance"] != code_provenance():
                raise ValueError("resume code/environment provenance differs")
            if not (output / "latest.json").exists() or read_checkpoint(output / "latest.json")["epoch"] != payload["epoch"]:
                raise ValueError("resume must use the latest checkpoint of this output run")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(config.device)
            scales = payload["scales"]
            history, start_epoch, best_loss = payload["history"], payload["epoch"], payload["best_loss"]
            restore_rng(payload["rng"])
        else:
            if (output / "latest.json").exists():
                raise ValueError("run already exists; use resume or a new output directory")
            scales = fit_scales(fit, config.normalization)
        if start_epoch >= config.epochs:
            raise ValueError("requested epochs must exceed checkpoint epoch")
        write_json(output / "run.json", {"identity": identity, "provenance": code_provenance(),
                   "scales": scales, "epochs_requested": config.epochs,
                   "sam2_parameters": "not loaded in offline state supervision; frozen during collection"})
        write_json(output / "metrics.json", history)
        for epoch in range(start_epoch, config.epochs):
            started = time.perf_counter()
            if torch.device(config.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            order = torch.randperm(len(fit)).tolist()
            fit_metrics = _run_pass(model, fit, config, scales, target_spec, optimizer, order)
            dev_metrics = _run_pass(model, dev, config, scales, target_spec) if dev else None
            # Overfit probes are fit-only diagnostics, never evidence of generalization.
            probe = dev_metrics or _run_pass(model, fit, config, scales, target_spec)
            row = {"epoch": epoch+1, "fit": fit_metrics, "dev": dev_metrics,
                   "fit_probe": probe if overfit else None,
                   "wall_time_seconds": time.perf_counter()-started,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.device(config.device).type == "cuda" else None}
            history.append(row)
            improved = probe["normalized_loss"] < best_loss
            best_loss = min(best_loss, probe["normalized_loss"])
            payload = {"schema_version": CHECKPOINT_SCHEMA, "identity": identity,
                       "provenance": code_provenance(), "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "epoch": epoch+1, "rng": rng_state(),
                       "scales": scales, "history": history, "best_loss": best_loss}
            entry = write_tensor_file(output / f"epoch-{epoch+1:05d}.pt", payload)
            write_json(output / (entry["path"] + ".json"), entry)
            write_json(output / "latest.json", entry)
            if improved:
                write_json(output / "best.json", entry)
            write_json(output / "metrics.json", history)
            print(json.dumps(row), flush=True)
    model.eval()
    return history


class CheckpointTranslator:
    """Runtime facade accepted by the existing actual SAM 2 no-replay injector."""

    def __init__(self, path, *, device="cpu"):
        payload = read_checkpoint(path)
        if payload["schema_version"] != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        self.identity = payload["identity"]
        if self.identity["contract"] != CONTRACT:
            raise ValueError("unsupported state contract")
        config = self.identity["config"]
        if factory_hash(config["factory"]) != self.identity["factory_source_sha256"]:
            raise ValueError("factory source differs from trained checkpoint")
        self.source_spec = StateSpec(**self.identity["source_spec"])
        self.target_spec = StateSpec(**self.identity["target_spec"])
        self.model = create_translator(config["factory"], self.source_spec, self.target_spec,
                                       config["factory_kwargs"]).to(device)
        if architecture_hash(self.model) != self.identity["architecture_source_sha256"]:
            raise ValueError("translator architecture source differs from trained checkpoint")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.device = device
        self.record_batch_size = config["record_batch_size"]
        self.name = config["factory"]

    def translate(self, source):
        if source.spec != self.source_spec:
            raise ValueError("runtime source spec differs from training")
        with torch.no_grad():
            records = source.validity.shape[2]
            size = self.record_batch_size or records
            spatial, pointer = [], []
            for start in range(0, records, size):
                chunk = replace(source, **{name: getattr(source, name)[:, :, start:start+size]
                                          for name in TENSORS})
                result = guarded_forward(self.model, move_state(chunk, self.device), self.target_spec)
                spatial.append(result.spatial_memory.cpu())
                pointer.append(result.object_pointer.cpu())
            return source.with_continuous(spatial_memory=torch.cat(spatial, dim=2),
                    object_pointer=torch.cat(pointer, dim=2),
                    presence_logits=source.presence_logits.clone(),
                    positional_information={"policy": "regenerate_at_target"})

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())
