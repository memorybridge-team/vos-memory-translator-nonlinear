from __future__ import annotations

from typing import Callable

import pytest
import torch

from vos_memory_inspector.state_schema import CanonicalState


StateFactory = Callable[..., CanonicalState]


@pytest.fixture
def make_state() -> StateFactory:
    """Build a validated CanonicalState from spatial ``[B,O,K,C,H,W]`` and pointer ``[B,O,K,D]``.

    Record 0 of every object is a conditioning record unless ``conditioning`` is given.
    """

    def factory(
        spatial: torch.Tensor,
        pointer: torch.Tensor,
        *,
        validity: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        switch_frame: int | None = None,
    ) -> CanonicalState:
        batch, objects, records = spatial.shape[:3]
        shape = (batch, objects, records)
        frames = torch.arange(records).view(1, 1, records).expand(shape).clone()
        if conditioning is None:
            conditioning = torch.zeros(shape, dtype=torch.bool)
            conditioning[..., 0] = True
        return CanonicalState(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=torch.randn(*shape, 1),
            frame_indices=frames,
            slot_order=frames.clone(),
            is_conditioning=conditioning,
            validity=torch.ones(shape, dtype=torch.bool) if validity is None else validity,
            object_ids=tuple(range(objects)),
            switch_frame=records - 1 if switch_frame is None else switch_frame,
            metadata={"sentinel": "preserve"},
        ).validate()

    return factory
