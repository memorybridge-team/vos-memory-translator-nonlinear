"""중복 없는 DDP global window. tail에서도 유효 record 수로 gradient 가중한다."""
import random
from .common import require


def schedule(entries, world, global_batch=64, *, seed=7, epoch=0, shuffle=True):
    require(world > 0 and global_batch > 0 and global_batch % world == 0, "GLOBAL_BATCH_WORLD")
    order = list(entries)
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    assigned = [[] for _ in range(world)]
    quota = global_batch // world
    windows, current = [], [0] * world
    for e in order:
        remaining, position = e["records"], 0
        while remaining:
            owner = sum(current) // quota
            n = min(remaining, quota - current[owner])
            positions = e.get("selected_positions", range(len(e["valid_indices"])))
            assigned[owner].append(dict(e, records=n, selected_positions=list(positions[position:position + n])))
            current[owner] += n
            remaining -= n
            position += n
            if sum(current) == global_batch:
                windows.append(current)
                current = [0] * world
    if sum(current):
        windows.append(current)
    require(sum(map(sum, windows)) == sum(e["records"] for e in entries), "SCHEDULE_COVERAGE")
    return assigned, windows


def refs_for(entries):
    return [[e["case"]["case_id"], *e["valid_indices"][i]] for e in entries
            for i in e.get("selected_positions", range(len(e["valid_indices"])))]


def window_batches(counts, microbatch):
    offset = 0
    for count in counts:
        for start in range(0, count, microbatch):
            yield list(range(offset + start, offset + min(start + microbatch, count)))
        offset += count
