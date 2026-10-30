"""Read masks for the next frame, separately for spatial memory and pointers.

These masks do not authorize selective handoff. Runtime logits equivalence has
not been established, so the default translator still computes every record.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MemoryPolicy:
    num_maskmem: int = 7
    max_obj_ptrs: int = 16
    stride: int = 1
    reverse: bool = False
    correction: bool = False

    def __post_init__(self):
        if min(self.num_maskmem, self.max_obj_ptrs, self.stride) < 1:
            raise ValueError('memory policy sizes must be positive')

    @property
    def supported(self):
        return self.stride == 1 and not self.reverse and not self.correction


def record_usage(frames, conditioning, validity, switch_frame, policy=MemoryPolicy()):
    # Multiple conditioning frames need max_cond_frames_in_attn selection;
    # until verified, conservatively mark every record as potentially consumed.
    supported = policy.supported and bool((conditioning & validity).sum(-1).eq(1).all())
    if not supported:
        return validity.clone(), validity.clone(), False
    distance = switch_frame + 1 - frames
    spatial = validity & (conditioning | ((distance >= 1) & (distance < policy.num_maskmem)))
    pointer = validity & (conditioning | ((distance >= 1) & (distance < policy.max_obj_ptrs)))
    return spatial, pointer, True


def require_selective_handoff_verified(*args, **kwargs):
    # No boolean flag or user-written receipt can stand in for runtime proof.
    raise ValueError('selective handoff is disabled: predictor logits equivalence is unverified')
