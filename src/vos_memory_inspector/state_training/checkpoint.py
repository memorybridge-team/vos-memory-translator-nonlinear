"""저장 → checksum → weights_only strict reload → complete marker 순서."""
from pathlib import Path
import os
import tempfile
import torch
from .common import require, sha256, read_json, write_json

SCHEMA = "cmmt.r2_training_checkpoint.v1"


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def publish(path, payload, verifier):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not Path(str(path) + ".complete.json").exists(), "CHECKPOINT_ALREADY_EXISTS")
    fd, name = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())
        saved = torch.load(name, map_location="cpu", weights_only=True)
        require(equal(saved, payload), "CHECKPOINT_SERIALIZATION_DIFFERENCE")
        verifier(saved)  # 검증이 실패하면 marker와 promotion은 생성하지 않는다.
        digest = sha256(name)
        os.replace(name, path)
        entry = dict(path=path.name, sha256=digest, bytes=path.stat().st_size, strict_reload_verified=True,
                     epoch=payload.get("epoch"), schema_version=payload.get("schema_version"))
        write_json(str(path) + ".complete.json", entry, immutable=True)
        return entry
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load(path, identity=None):
    path = Path(path)
    marker = read_json(str(path) + ".complete.json")
    require(marker["path"] == path.name and marker["strict_reload_verified"] and sha256(path) == marker["sha256"], "CHECKPOINT_COMPLETION_OR_SHA")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if identity is not None:
        require(value.get("schema_version") == SCHEMA and value.get("identity") == identity, "OLD_OR_DIFFERENT_CONTRACT_CHECKPOINT")
    return value, marker


def latest(root, identity):
    root = Path(root)
    candidates = sorted((root / "checkpoints").glob("epoch-*.pt.complete.json"))
    require(candidates, "NO_COMPLETE_CHECKPOINT")
    path = Path(str(candidates[-1])[:-len(".complete.json")])
    value, entry = load(path, identity)
    return value, dict(entry, path=str(path))


def rng_state():
    import random
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    import random
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])
