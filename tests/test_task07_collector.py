from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
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
    first = collector._claim_case(
        tmp_path, "mose_case_1", "worker-a", reclaim_stale=True, lease_seconds=60
    )
    second = collector._claim_case(
        tmp_path, "mose_case_1", "worker-b", reclaim_stale=True, lease_seconds=60
    )

    assert first is not None
    assert second is None
    assert "worker-a" in first.read_text(encoding="utf-8")


def test_dynamic_queue_claim_can_be_released_after_success(tmp_path: Path) -> None:
    collector = _collector_module()
    first = collector._claim_case(
        tmp_path, "mose_case_2", "worker-a", reclaim_stale=True, lease_seconds=60
    )
    assert first is not None
    first.unlink()

    second = collector._claim_case(
        tmp_path, "mose_case_2", "worker-b", reclaim_stale=True, lease_seconds=60
    )
    assert second is not None


def test_stale_claim_is_quarantined_then_reclaimed(tmp_path: Path) -> None:
    collector = _collector_module()
    first = collector._claim_case(
        tmp_path, "mose_case_3", "dead-worker", reclaim_stale=True, lease_seconds=60
    )
    assert first is not None
    old = time.time() - 61
    os.utime(first, (old, old))

    second = collector._claim_case(
        tmp_path, "mose_case_3", "new-worker", reclaim_stale=True, lease_seconds=60
    )
    assert second is not None
    assert "new-worker" in second.read_text(encoding="utf-8")
    assert list((tmp_path / "claims").glob("mose_case_3.claim.stale-*"))


def test_queue_contract_rejects_mixed_runs(tmp_path: Path) -> None:
    collector = _collector_module()
    first = {"content_sha256": "a"}
    collector._ensure_queue_contract(tmp_path, first)
    collector._ensure_queue_contract(tmp_path, first)

    try:
        collector._ensure_queue_contract(tmp_path, {"content_sha256": "b"})
    except ValueError as error:
        assert "contract differs" in str(error)
    else:
        raise AssertionError("mixed dynamic queue contracts must be rejected")
