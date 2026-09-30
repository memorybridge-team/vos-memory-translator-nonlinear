from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _collector_module():
    path = Path(__file__).parents[1] / "scripts" / "task07" / "prepare_paired_state_dataset.py"
    spec = importlib.util.spec_from_file_location("task07_collector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_checksum_matches_only_for_intact_sidecar(tmp_path: Path) -> None:
    collector = _collector_module()
    cache = tmp_path / "case.pt"
    cache.write_bytes(b"state")
    digest = hashlib.sha256(b"state").hexdigest()
    cache.with_suffix(".pt.sha256").write_text(f"{digest}  case.pt\n", encoding="ascii")

    assert collector._cache_checksum_matches(cache)
    cache.write_bytes(b"changed")
    assert not collector._cache_checksum_matches(cache)


def test_atomic_json_write_does_not_leave_partial_file(tmp_path: Path) -> None:
    collector = _collector_module()
    output = tmp_path / "status.json"
    collector._write_json_atomic(output, {"state": "running"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"state": "running"}
    assert not output.with_suffix(".json.partial").exists()


def test_dynamic_queue_claim_is_exclusive(tmp_path: Path) -> None:
    collector = _collector_module()
    first = collector._claim_case(tmp_path, "mose_case_1", "worker-a")
    second = collector._claim_case(tmp_path, "mose_case_1", "worker-b")

    assert first is not None
    assert second is None
    assert "worker-a" in first.read_text(encoding="utf-8")


def test_dynamic_queue_claim_can_be_released_after_success(tmp_path: Path) -> None:
    collector = _collector_module()
    first = collector._claim_case(tmp_path, "mose_case_2", "worker-a")
    assert first is not None
    first.unlink()

    second = collector._claim_case(tmp_path, "mose_case_2", "worker-b")
    assert second is not None
