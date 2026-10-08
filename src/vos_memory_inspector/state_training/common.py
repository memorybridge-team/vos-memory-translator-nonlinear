"""Torch를 import하지 않는 계약, checksum, atomic 파일 도구."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import tempfile


def require(condition, code):
    if not condition:
        raise ValueError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, data, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not immutable or not path.exists(), "IMMUTABLE_FILE_EXISTS:" + str(path))
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        require(not immutable or not path.exists(), "IMMUTABLE_FILE_EXISTS:" + str(path))
        os.replace(name, path)
        # Linux Network Volume에서도 file과 directory commit 순서를 보존한다.
        if os.name != "nt":
            dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value, *, immutable=False):
    atomic(path, encoded(value), immutable=immutable)


def save_bound(path, value):
    write_json(path, value, immutable=True)
    atomic(str(path) + ".sha256", (sha256(path) + "\n").encode(), immutable=True)
    return sha256(path)


def read_bound(path):
    expected = Path(str(path) + ".sha256").read_text().split()[0]
    require(sha256(path) == expected, "JSON_CHECKSUM:" + str(path))
    return read_json(path)


@contextmanager
def writer(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "WRITER.lock"
    # lock가 남으면 자동 삭제하지 않는다. operator가 정확한 소유자를 확인한다.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def stamp(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def reject_placeholders(value):
    if isinstance(value, dict):
        for item in value.values():
            reject_placeholders(item)
    elif isinstance(value, list):
        for item in value:
            reject_placeholders(item)
    elif isinstance(value, str):
        require("<" not in value and "PLACEHOLDER" not in value, "UNFILLED_CONFIGURATION_PLACEHOLDER:" + value)


def outside_originals(output, index):
    output = Path(output).resolve()
    for source in index["identity"]["sources"]:
        spec = source["spec"]
        roots = list(spec["cache_roots"].values())
        roots.append(read_bound(spec["inventory"])["source"]["rgb_root"])
        require(all(not output.is_relative_to(Path(root).resolve()) for root in roots), "OUTPUT_INSIDE_ORIGINAL_CACHE_OR_RGB")
