"""R² best와 meaningful improvement는 별개의 상태다. MSE는 진단용이다."""
from dataclasses import asdict, dataclass
import math
from .common import require
from .r2 import SPEC


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    microbatch: int = 32
    global_batch: int = 64
    workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    cache_cases: int = 2
    cache_bytes_per_worker: int = 64 * 1024 * 1024
    max_epochs: int = 60
    min_epochs: int = 15
    patience: int = 8
    min_delta: float = 0.001
    clip_norm: float = 1.0
    seed: int = 7
    precision: str = "fp32"
    augmentation: str = "none"
    log_every: int = 20

    def validate(self, world):
        require(world in (1, 2, 4) and self.microbatch > 0 and self.global_batch == 64 and
                self.global_batch % (world * self.microbatch) == 0, "GLOBAL_BATCH_CONTRACT")
        require(60 >= self.max_epochs >= self.min_epochs > 0 and self.patience > 0 and math.isfinite(self.min_delta) and self.min_delta >= 0, "EARLY_POLICY")
        require(self.workers >= 0 and self.prefetch_factor > 0 and self.cache_cases > 0 and self.cache_bytes_per_worker > 0, "BOUNDED_LOADER")
        require(self.precision == "fp32" and self.augmentation == "none" and self.lr == 3e-4 and self.weight_decay == 1e-4 and self.clip_norm == 1.0, "FROZEN_TRAIN_METHOD")

    def logical(self):
        return {k: v for k, v in asdict(self).items() if k not in
                ("workers", "prefetch_factor", "pin_memory", "cache_cases", "cache_bytes_per_worker", "log_every")}


class Selection:
    def __init__(self, min_epochs=15, patience=8, min_delta=0.001, tie_tolerance=SPEC["tie_tolerance"]):
        self.min_epochs, self.patience, self.min_delta, self.tie_tolerance = min_epochs, patience, min_delta, tie_tolerance
        self.best_score = None
        self.best_epoch = None
        self.reference = None
        self.bad_checks = 0
        self.last_epoch = 0
        self.scores = []

    def observe(self, epoch, metric):
        require(epoch == self.last_epoch + 1 and math.isfinite(metric), "FULL_VALIDATION_SEQUENCE")
        self.scores.append(metric)
        maximum = max(self.scores)
        selected = next(i + 1 for i, value in enumerate(self.scores) if value >= maximum - self.tie_tolerance)
        promote = selected != self.best_epoch
        self.best_epoch, self.best_score = selected, self.scores[selected - 1]
        meaningful = self.reference is None or metric > self.reference + self.min_delta
        if meaningful:
            self.reference, self.bad_checks = metric, 0
        elif epoch > self.min_epochs:
            self.bad_checks += 1
        self.last_epoch = epoch
        return promote, epoch > self.min_epochs and self.bad_checks >= self.patience

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        require(all(state[k] == getattr(self, k) for k in ("min_epochs", "patience", "min_delta", "tie_tolerance")), "EARLY_POLICY_CHANGED")
        require(len(state["scores"]) == state["last_epoch"] and all(math.isfinite(x) for x in state["scores"]) and state["bad_checks"] >= 0, "SELECTION_HISTORY_CORRUPT")
        self.__dict__.update(state)


class WarmupPlateau:
    def __init__(self, optimizer, updates_per_epoch, base_lr=3e-4, min_delta=0.001):
        import torch
        self.optimizer, self.base_lr = optimizer, base_lr
        self.warmup_updates = min(updates_per_epoch, 1000)
        require(self.warmup_updates > 0, "EMPTY_TRAIN")
        self.completed_updates = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, threshold=min_delta,
            threshold_mode="abs", cooldown=1, min_lr=1e-5)
        for g in optimizer.param_groups:
            g["lr"] = base_lr / self.warmup_updates

    def after_update(self):
        self.completed_updates += 1
        if self.completed_updates <= self.warmup_updates:
            for g in self.optimizer.param_groups:
                g["lr"] = self.base_lr * min(1.0, (self.completed_updates + 1) / self.warmup_updates)

    def after_validation(self, metric):
        require(math.isfinite(metric), "NONFINITE_R2")
        if self.completed_updates >= self.warmup_updates:
            self.plateau.step(metric)

    def state_dict(self):
        return dict(schema_version="cmmt.r2_warmup_plateau.v1", warmup_updates=self.warmup_updates,
                    completed_updates=self.completed_updates, base_lr=self.base_lr, plateau=self.plateau.state_dict())

    def load_state_dict(self, state):
        require(state["schema_version"] == "cmmt.r2_warmup_plateau.v1" and state["warmup_updates"] == self.warmup_updates and
                state["base_lr"] == self.base_lr and state["plateau"]["mode"] == "max" and
                state["plateau"]["threshold_mode"] == "abs", "SCHEDULER_PLAN_CHANGED")
        self.completed_updates = state["completed_updates"]
        self.plateau.load_state_dict(state["plateau"])
