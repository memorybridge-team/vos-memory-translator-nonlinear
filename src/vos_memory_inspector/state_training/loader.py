"""검증된 case의 유효 record만 로드한다. 원본 SHA 검사는 생략하지 않는다."""
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from .cache import load_raw
from .common import require, stamp
from .schedule import window_batches


class RawRecordDataset(Dataset):
    def __init__(self, entries, cache_cases=2, cache_bytes=64 * 1024 * 1024):
        self.entries = entries
        self.ends = list(itertools.accumulate(e["records"] for e in entries))
        self.cache_cases, self.cache_bytes = cache_cases, cache_bytes
        self.cache, self.bytes = OrderedDict(), 0

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def location(self, i):
        require(0 <= i < len(self), "RECORD_INDEX_RANGE")
        segment = bisect_right(self.ends, i)
        e = self.entries[segment]
        relative = i - (self.ends[segment - 1] if segment else 0)
        return e, e.get("selected_positions", range(len(e["valid_indices"])))[relative]

    def obtain(self, entry):
        path = entry["path"]
        if path in self.cache:
            result, size = self.cache.pop(path)
            self.cache[path] = result, size
            return result
        result, _, _ = load_raw(path, expected_sha=entry["sha256"], expected_stamp=entry["stamp"], stable_seconds=0)
        require(torch.nonzero(result["source_canonical"].validity).tolist() == entry["valid_indices"], "INDEX_VALIDITY_CHANGED")
        size = sum(t.numel() * t.element_size() for key in ("source_canonical", "target_canonical")
                   for t in vars(result[key]).values() if isinstance(t, torch.Tensor))
        while self.cache and (len(self.cache) >= self.cache_cases or self.bytes + size > self.cache_bytes):
            _, (_, old) = self.cache.popitem(last=False)
            self.bytes -= old
        if size <= self.cache_bytes:
            self.cache[path] = result, size
            self.bytes += size
        return result

    def __getitems__(self, indices):
        checked, rows = set(), []
        previous, value = None, None
        for i in indices:
            e, position = self.location(i)
            path = e["path"]
            if path not in checked:
                require(stamp(path) == e["stamp"] and not Path(path + ".writer.lock").exists(), "CACHE_CHANGED_OR_WRITER")
                require(Path(path + ".sha256").read_text().split()[0] == e["sha256"], "CACHE_SIDECAR_CHANGED")
                checked.add(path)
            if path != previous:
                value, previous = self.obtain(e), path
            s, t = value["source_canonical"], value["target_canonical"]
            b, o, k = e["valid_indices"][position]
            rows.append(dict(source_spatial=s.spatial_memory[b, o, k].clone(), target_spatial=t.spatial_memory[b, o, k].clone(),
                             source_pointer=s.object_pointer[b, o, k].clone(), target_pointer=t.object_pointer[b, o, k].clone(),
                             frame=s.frame_indices[b, o, k], slot=s.slot_order[b, o, k], conditioning=s.is_conditioning[b, o, k],
                             ref=[e["case"]["case_id"], b, o, k], dataset=e["case"]["dataset"]))
        return rows

    def __getitem__(self, i):
        return self.__getitems__([i])[0]


def collate(rows):
    keys = ("source_spatial", "target_spatial", "source_pointer", "target_pointer", "frame", "slot", "conditioning")
    return dict(tensors={k: torch.stack([r[k] for r in rows]) for k in keys}, refs=[r["ref"] for r in rows],
                datasets=[r["dataset"] for r in rows])


def loader(entries, config, *, cuda=False, window_counts=None):
    data = RawRecordDataset(entries, config.cache_cases, config.cache_bytes_per_worker)
    options = dict(num_workers=config.workers, collate_fn=collate, pin_memory=bool(cuda and config.pin_memory))
    if window_counts is None:
        options.update(batch_size=config.microbatch, shuffle=False, drop_last=False)
    else:
        require(sum(window_counts) == len(data), "LOADER_WINDOW_COVERAGE")
        options["batch_sampler"] = list(window_batches(window_counts, config.microbatch))
    if config.workers:
        options.update(prefetch_factor=config.prefetch_factor, persistent_workers=False, multiprocessing_context="spawn")
    return DataLoader(data, **options)
