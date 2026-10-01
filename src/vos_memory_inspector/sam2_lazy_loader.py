"""Bounded-memory JPEG frame loader for SAM2 Task 07 collection.

SAM2's stock JPEG loader materializes every resized frame in a float32 tensor.
That is convenient for demos but unsafe for very long videos.  This module
installs a small LRU-backed loader in the predictor modules used by the
collector.  It preserves the predictor's sequence interface (``len`` and
indexing) while keeping only a bounded number of frames in host/device memory.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class LazyJPEGFrames:
    """Sequence-compatible, bounded LRU cache of normalized JPEG frames."""

    def __init__(
        self,
        paths: list[str],
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: torch.Tensor,
        img_std: torch.Tensor,
        compute_device: torch.device,
        cache_size: int = 8,
    ) -> None:
        from sam2.utils.misc import _load_img_as_tensor

        if not paths:
            raise RuntimeError("no images found")
        self._paths = paths
        self._image_size = image_size
        self._offload = offload_video_to_cpu
        self._mean = img_mean
        self._std = img_std
        self._device = compute_device
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._load_img_as_tensor = _load_img_as_tensor
        first, self.video_height, self.video_width = self._load(0)
        self._cache[0] = first

    def _load(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, height, width = self._load_img_as_tensor(
            self._paths[index], self._image_size
        )
        image -= self._mean
        image /= self._std
        if not self._offload:
            image = image.to(self._device, non_blocking=True)
        return image, height, width

    def __getitem__(self, index: int) -> torch.Tensor:
        index = int(index)
        if index < 0:
            index += len(self._paths)
        if index < 0 or index >= len(self._paths):
            raise IndexError(index)
        value = self._cache.pop(index, None)
        if value is None:
            value, height, width = self._load(index)
            self.video_height, self.video_width = height, width
        self._cache[index] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._paths)


def install_sam2_lazy_loader(*, cache_size: int = 8) -> None:
    """Patch SAM2 predictor modules for the current process.

    The patch is opt-in and affects only the collector subprocess.  MP4 input
    and non-JPEG paths retain SAM2's original loader.
    """

    import sam2.sam2_video_predictor as predictor
    try:
        import sam2.sam2_video_predictor_legacy as legacy
    except ImportError:  # pragma: no cover - version dependent
        legacy = None
    import sam2.utils.misc as misc

    original = predictor.load_video_frames

    def load_video_frames_bounded(
        video_path: Any,
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames: bool = False,
        compute_device: torch.device = torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            folder = Path(video_path)
            names = sorted(
                (p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}),
                key=lambda p: int(p.stem),
            )
            if not names:
                raise RuntimeError(f"no images found in {folder}")
            mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            frames = LazyJPEGFrames(
                [str(p) for p in names],
                image_size,
                offload_video_to_cpu,
                mean,
                std,
                compute_device,
                cache_size=cache_size,
            )
            return frames, frames.video_height, frames.video_width
        return original(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            async_loading_frames,
            compute_device,
        )

    predictor.load_video_frames = load_video_frames_bounded
    if legacy is not None:
        legacy.load_video_frames = load_video_frames_bounded
