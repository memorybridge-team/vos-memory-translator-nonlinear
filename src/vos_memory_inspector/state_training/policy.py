"""Training policies copied from the frozen first-run protocol, without its runtime dependencies."""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    microbatch: int = 32
    global_batch: int = 64
    workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    cache_cases: int = 2
    cache_bytes: int = 64 * 1024 * 1024
    max_epochs: int = 60
    min_epochs: int = 15
    early_stop_checks: int = 8
    relative_improvement: float = 0.002
    clip_norm: float = 1.0
    seed: int = 7
    spatial_loss_records: str = 'all'

    def validate(self, world=1):
        integer_fields = ('microbatch', 'global_batch', 'workers', 'prefetch_factor',
                          'cache_cases', 'cache_bytes', 'max_epochs', 'min_epochs', 'early_stop_checks', 'seed')
        if not isinstance(world, int) or isinstance(world, bool) or any(
            not isinstance(getattr(self, key), int) or isinstance(getattr(self, key), bool)
            for key in integer_fields):
            raise ValueError('integer training/loader settings required')
        if not isinstance(self.pin_memory, bool):
            raise ValueError('pin_memory must be boolean')
        if self.global_batch < world or self.global_batch % world or min(self.microbatch, world) < 1:
            raise ValueError('global batch must be divisible by world size and at least world size')
        if min(self.max_epochs, self.min_epochs, self.early_stop_checks, self.cache_cases,
               self.cache_bytes, self.prefetch_factor) < 1 or self.workers < 0:
            raise ValueError('invalid positive training/loader limits')
        if not all(math.isfinite(v) for v in (self.lr, self.weight_decay, self.relative_improvement, self.clip_norm)):
            raise ValueError('nonfinite training config')
        if self.lr <= 0 or self.weight_decay < 0 or self.clip_norm <= 0 or not 0 <= self.relative_improvement < 1:
            raise ValueError('invalid optimizer/early stop setting')
        if self.spatial_loss_records not in ('all', 'used'):
            raise ValueError('spatial_loss_records must be all or used')

    def logical(self):
        return {k: v for k, v in asdict(self).items() if k not in (
            'workers', 'microbatch', 'prefetch_factor', 'pin_memory', 'cache_cases', 'cache_bytes', 'max_epochs')}


def optimizer_groups(model, weight_decay):
    norm_ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
                for p in m.parameters(recurse=False)}
    decay, excluded = [], []
    for name, parameter in model.named_parameters():
        no_decay = name.endswith('.bias') or id(parameter) in norm_ids or name.endswith('.alpha') or 'pos_embed' in name
        (excluded if no_decay else decay).append(parameter)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=excluded, weight_decay=0.)]


class WarmupPlateau:
    def __init__(self, optimizer, updates_per_epoch, base_lr=3e-4):
        if updates_per_epoch < 1:
            raise ValueError('empty train updates')
        self.optimizer, self.base_lr = optimizer, base_lr
        self.warmup_updates = min(updates_per_epoch, 1000)
        self.completed_updates = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, threshold=0.002,
            threshold_mode='rel', cooldown=1, min_lr=1e-5)
        for group in optimizer.param_groups:
            group['lr'] = base_lr / self.warmup_updates

    def after_update(self):
        self.completed_updates += 1
        if self.completed_updates <= self.warmup_updates:
            for group in self.optimizer.param_groups:
                group['lr'] = self.base_lr * min(self.completed_updates + 1, self.warmup_updates) / self.warmup_updates

    def after_validation(self, metric):
        if not math.isfinite(metric):
            raise ValueError('nonfinite validation loss')
        if self.completed_updates >= self.warmup_updates:
            self.plateau.step(metric)

    def state_dict(self):
        return dict(warmup_updates=self.warmup_updates, completed_updates=self.completed_updates,
                    base_lr=self.base_lr, plateau=self.plateau.state_dict())

    def load_state_dict(self, state):
        if state['warmup_updates'] != self.warmup_updates or state['base_lr'] != self.base_lr:
            raise ValueError('scheduler plan changed')
        self.completed_updates = state['completed_updates']
        self.plateau.load_state_dict(state['plateau'])


class ValidationStop:
    def __init__(self, min_epochs=15, checks=8, threshold=0.002):
        self.min_epochs, self.checks, self.threshold = min_epochs, checks, threshold
        self.best_reference = None
        self.bad_checks = 0
        self.last_epoch = 0

    def observe(self, epoch, metric):
        if epoch != self.last_epoch + 1 or not math.isfinite(metric):
            raise ValueError('invalid full-validation sequence')
        if self.best_reference is None or metric < self.best_reference * (1 - self.threshold):
            self.best_reference, self.bad_checks = metric, 0
        elif epoch > self.min_epochs:
            self.bad_checks += 1
        self.last_epoch = epoch
        return epoch > self.min_epochs and self.bad_checks >= self.checks

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ('min_epochs', 'checks', 'threshold')):
            raise ValueError('early stop policy changed')
        self.__dict__.update(state)


def schedule(entries, world, global_batch, *, seed=7, epoch=0, shuffle=True):
    if world < 1 or global_batch < world or global_batch % world:
        raise ValueError('invalid distributed global batch')
    order = list(entries)
    if shuffle:
        indices = torch.randperm(len(order), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        order = [order[i] for i in indices]
    assigned, windows = [[] for _ in range(world)], []
    quota, current = global_batch // world, [0] * world
    for entry in order:
        position, remaining = 0, entry['records']
        while remaining:
            owner = sum(current) // quota
            n = min(remaining, quota - current[owner])
            assigned[owner].append({**entry, 'records': n, 'selected_positions': list(range(position, position + n))})
            current[owner] += n
            remaining -= n
            position += n
            if sum(current) == global_batch:
                windows.append(current)
                current = [0] * world
    if sum(current):
        windows.append(current)
    return assigned, windows
