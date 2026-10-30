"""Record-batch FP32 trainer with weighted DDP tails and epoch-boundary resume."""
from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..models.registry import build_translator
from ..models._spatial import SpatialTransformerConfig
from ..models.residual_mlp import ResidualMLPStateTranslator
from ..models.transformer import TransformerStateTranslator
from ..state_schema import StateSpec
from .data import record_loader
from .metrics import Metrics
from .policy import WarmupPlateau, ValidationStop, optimizer_groups, schedule
from .preflight import inspect, require_ready
from .ridge import load_ridge
from .storage import digest, save_json, save_torch, writer_lock


class TensorModel(nn.Module):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    def forward(self, spatial, pointer):
        s, p = self.translator.translate_tensors(spatial[None, None], pointer[None, None])
        return s[0, 0], p[0, 0]


def construct_model(name, spec, options=None):
    options = dict(options or {})
    if name == 'residual_mlp':
        return build_translator(name, spec, spec, **options)
    if name not in ('transformer', 'mlp_context'):
        raise ValueError('unsupported training model')
    options.setdefault("local_fusion", "linear" if name == "transformer" else "additive_mlp")
    config = SpatialTransformerConfig(channels=spec.feature_channels, height=spec.height,
        width=spec.width, pointer_dim=spec.pointer_dim, **options)
    return build_translator('base' if name == 'transformer' else name, spec, spec, config=config,
                            local_fusion='linear' if name == 'transformer' else 'additive_mlp')


def restore_model(name, payload):
    cls = ResidualMLPStateTranslator if name == 'residual_mlp' else TransformerStateTranslator
    return cls.from_payload(payload)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(value):
    random.setstate(value['python'])
    state = value['numpy']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate(model, entries, config, scales, device, rank=0, world=1):
    assigned, _ = schedule(entries, world, config.global_batch, shuffle=False)
    totals = Metrics()
    model.eval()
    with torch.no_grad():
        for batch in record_loader(assigned[rank], config, cuda=device.type == 'cuda'):
            batch = _to_device(batch, device)
            s, p = model.translate_tensors(batch['source_spatial'][None, None], batch['source_pointer'][None, None])
            totals.add(s[0, 0], p[0, 0], batch)
    totals.reduce(device)
    return totals.report(scales)


def _rank_zero(function, rank, world):
    """Propagate rank-0 IO failures instead of leaving peers waiting forever."""
    result = [None, None]
    if rank == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    if result[1]:
        raise ValueError(result[1])
    return result[0]


def train(index, config, name, output, *, options=None, ridge_path=None, device='cpu',
          resume=None, allow_sample=False, readiness=None):
    device = torch.device(device)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    config.validate(world)
    if not index.role('train') or not index.role('validation'):
        raise ValueError('training requires nonempty train and validation roles')
    if not index.official and (not allow_sample or device.type != 'cpu'):
        raise ValueError('sample indexes are CPU smoke only')
    if allow_sample and device.type != 'cpu':
        raise ValueError('sample mode cannot run CUDA training')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('record training supports CPU or CUDA only')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
    if readiness is not None:
        # A supplied report is advisory; inspect actual files at every new launch.
        if readiness['data_identity'] != index.identity:
            raise ValueError('readiness data identity mismatch')
    report = _rank_zero(lambda: inspect(index, config), rank, world)
    require_ready(report, allow_sample=allow_sample)
    scales = report['normalization']['scales']
    spec = StateSpec(**report['source_spec'])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = construct_model(name, spec, options)
    ridge = load_ridge(ridge_path) if ridge_path is not None else None
    if getattr(model, 'skip_mode', 'identity') == 'affine':
        if ridge is None:
            raise ValueError('affine training requires --ridge-init')
        if ridge['provenance']['data_identity'] != index.identity:
            raise ValueError('ridge was fitted/selected on a different index')
        model.initialize_from_ridge(ridge)
    elif ridge is not None:
        raise ValueError('ridge initialization is only supported in affine mode')
    identity = dict(data=index.identity, config=config.logical(), model=name, options=options or {},
                    spec=spec.to_dict(), normalization=scales, world_size=world,
                    ridge_sha256=ridge['provenance']['artifact_sha256'] if ridge else None)
    signature = digest(identity)
    output = Path(output)
    # Outputs may never be placed in an original cache directory.
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)
    if any(output.resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('output must not be inside original cache roots')
    context = writer_lock(output) if rank == 0 else nullcontext()
    # Acquire the lock collectively, so peers do not proceed when it fails.
    _rank_zero(lambda: context.__enter__(), rank, world)
    try:
        if resume is None:
            _rank_zero(lambda: _require_new_run(output), rank, world)
        model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(optimizer_groups(model, config.weight_decay), lr=config.lr)
        train_entries, valid_entries = index.role('train'), index.role('validation')
        total_records = sum(e['records'] for e in train_entries)
        scheduler = WarmupPlateau(optimizer, math.ceil(total_records / config.global_batch), config.lr)
        early = ValidationStop(config.min_epochs, config.early_stop_checks, config.relative_improvement)
        history, step, start, best_loss, best_payload, best_epoch = [], 0, 0, math.inf, None, 0
        checkpoint = None
        if resume is not None:
            checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
            if checkpoint.get('schema_version') != 'cmmt.record_training_checkpoint.v1' or checkpoint['signature'] != signature:
                raise ValueError('resume model/data/skip mode/protocol changed')
            if checkpoint['identity'] != identity:
                raise ValueError('resume identity mismatch')
            loaded = restore_model(name, checkpoint['model'])
            model.load_state_dict(loaded.state_dict(), strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            early.load_state_dict(checkpoint['early'])
            history, step, start = checkpoint['history'], checkpoint['step'], checkpoint['epoch']
            if config.max_epochs <= start:
                raise ValueError('max_epochs must exceed the completed checkpoint epoch')
            if early.last_epoch > early.min_epochs and early.bad_checks >= early.checks:
                raise ValueError('checkpoint has already early-stopped; policy is not reset')
            best_loss, best_payload, best_epoch = checkpoint['best_loss'], checkpoint['best_model'], checkpoint['best_epoch']
        tensor_model = TensorModel(model)
        ddp = DistributedDataParallel(tensor_model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else tensor_model
        if checkpoint is not None:
            restore_rng(checkpoint['rng'][rank])
        for epoch in range(start, config.max_epochs):
            if checkpoint is not None and early.last_epoch > early.min_epochs and early.bad_checks >= early.checks:
                break
            model.train()
            assigned, windows = schedule(train_entries, world, config.global_batch, seed=config.seed, epoch=epoch)
            loader = iter(record_loader(assigned[rank], config, windows=[w[rank] for w in windows], cuda=device.type == 'cuda'))
            metrics = Metrics()
            objective_total = 0.
            started = time.monotonic()
            for counts in windows:
                local = counts[rank]
                batches = [_to_device(next(loader), device) for _ in range(math.ceil(local / config.microbatch))]
                used = sum(int(b['spatial_used'].sum()) for b in batches)
                unsupported = torch.tensor(int(any(not bool(b['usage_supported'].all()) for b in batches)), device=device)
                if world > 1:
                    dist.all_reduce(unsupported, op=dist.ReduceOp.MAX)
                if config.spatial_loss_records == 'used' and int(unsupported):
                    raise ValueError('used spatial loss needs a supported memory policy')
                spatial_denominator = torch.tensor(used if config.spatial_loss_records == 'used' else local, device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(spatial_denominator)
                pointer_denominator = sum(counts)
                optimizer.zero_grad(set_to_none=True)
                micro_iterations = max(math.ceil(n / config.microbatch) for n in counts)
                for micro in range(micro_iterations):
                    batch = batches[micro] if micro < len(batches) else None
                    sync = ddp.no_sync() if world > 1 and micro + 1 < micro_iterations else nullcontext()
                    with sync:
                        if batch is None:
                            x = torch.zeros(1, spec.feature_channels, spec.height, spec.width, device=device)
                            y = torch.zeros(1, spec.pointer_dim, device=device)
                            s, p = ddp(x, y)
                            loss = (s.sum() + p.sum()) * 0
                        else:
                            s, p = ddp(batch['source_spatial'], batch['source_pointer'])
                            se = (s - batch['target_spatial'].float()).square().flatten(1).mean(1)
                            pe = (p - batch['target_pointer'].float()).square().mean(1)
                            mask = batch['spatial_used'] if config.spatial_loss_records == 'used' else torch.ones_like(batch['spatial_used'])
                            loss = se[mask].sum() / max(float(spatial_denominator), 1) / scales['spatial'] ** 2
                            loss = loss + pe.sum() / pointer_denominator / scales['pointer'] ** 2
                            metrics.add(s, p, batch)
                            objective_total += float(loss.detach()) * pointer_denominator
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('nonfinite training loss')
                        (loss * world).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.after_update()
                step += 1
            if next(loader, None) is not None:
                raise ValueError('unconsumed training records')
            metrics.reduce(device)
            objective = torch.tensor(objective_total, dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(objective)
            validation = evaluate(model, valid_entries, config, scales, device, rank, world)
            val_loss = validation['all']['state_loss']
            scheduler.after_validation(val_loss)
            stopped = early.observe(epoch + 1, val_loss)
            if val_loss < best_loss:
                best_loss, best_payload, best_epoch = val_loss, model.to_payload(), epoch + 1
                # Payload tensors must own storage; future optimizer updates must not alter best.
                best_payload = _clone_payload(best_payload)
            history.append(dict(epoch=epoch + 1, step=step, train=metrics.report(scales), validation=validation,
                                training_objective=float(objective) / total_records,
                                lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - started))
            local_rng = rng_state()
            all_rng = [None] * world
            if world > 1:
                dist.all_gather_object(all_rng, local_rng)
            else:
                all_rng[0] = local_rng
            state = dict(schema_version='cmmt.record_training_checkpoint.v1', signature=signature, identity=identity,
                         model=model.to_payload(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         early=early.state_dict(), rng=all_rng, epoch=epoch + 1, step=step, history=history,
                         best_loss=best_loss, best_model=best_payload, best_epoch=best_epoch)
            def publish():
                save_torch(output / 'last.ckpt', state)
                if best_epoch == epoch + 1:
                    save_torch(output / 'best.ckpt', state)
                save_torch(output / 'paired_translators.pt', {name: best_payload,
                           'training_selection': dict(selected_epoch=best_epoch, selected_validation_loss=best_loss)})
                save_json(output / 'training_report.json', dict(identity=identity, signature=signature,
                          history=history, selected_epoch=best_epoch, selected_validation_loss=best_loss,
                          preflight=report, cuda_training_executed=device.type == 'cuda',
                          hardware_validation='CPU only' if device.type == 'cpu' else 'CUDA run',
                          historical_comparison_eligible=report['historical_comparison_data_match'] and
                              config.spatial_loss_records == 'all',
                          inference_record_skipping_enabled=False))
                return None
            _rank_zero(publish, rank, world)
            if rank == 0:
                print(f'epoch={epoch+1} step={step} validation={val_loss:.8f}', flush=True)
            if stopped:
                break
        if best_payload is None:
            raise ValueError('no completed epoch or checkpoint')
        model.load_state_dict(restore_model(name, best_payload).state_dict(), strict=True)
        model.eval()
        return model, history
    finally:
        if rank == 0:
            context.__exit__(None, None, None)


def _require_new_run(output):
    if any((output / name).exists() for name in ('last.ckpt', 'best.ckpt', 'training_report.json', 'paired_translators.pt')):
        raise ValueError('output already contains a run; use --resume or a new directory')


def _clone_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_payload(v) for v in value]
    return value
