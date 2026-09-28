"""Durable local/RunPod-volume writes; immutable files precede JSON pointers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_write(path: str | Path, writer: Callable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial",
                                           dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda stream: stream.write(encoded))


def write_tensor_file(path: str | Path, value: Any) -> dict:
    path = Path(path)
    atomic_write(path, lambda stream: torch.save(value, stream))
    return {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def checked_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest path escapes persistent root")
    return path


def read_checked(root: str | Path, entry: dict) -> Any:
    path = checked_path(root, entry["path"])
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"integrity check failed: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class ExclusiveWriter:
    """One writer per collection/run. Stale locks require explicit operator removal."""

    def __init__(self, root: str | Path):
        self.path = Path(root) / ".writer.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, f"pid={os.getpid()}\n".encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, *_):
        os.close(self.fd)
        self.path.unlink()
