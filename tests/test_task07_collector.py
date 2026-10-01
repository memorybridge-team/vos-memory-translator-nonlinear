from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_cache_skip_requires_matching_generating_contract(tmp_path: Path) -> None:
    collector = _collector_module()
    cache = tmp_path / "case.pt"
    cache.write_bytes(b"state")
    digest = hashlib.sha256(b"state").hexdigest()
    cache.with_suffix(".pt.sha256").write_text(f"{digest}  case.pt\n", encoding="ascii")
    generating = {"case": "a"}
    assert not collector._cache_checksum_matches(cache, generating)
    Path(str(cache) + ".complete.json").write_text(
        json.dumps({"sha256": digest, "generating_sha256": collector.content_hash(generating)}),
        encoding="utf-8",
    )
    assert collector._cache_checksum_matches(cache, generating)
    assert not collector._cache_checksum_matches(cache, {"case": "b"})


def _signed(path: Path, payload: dict) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps({**payload, "content_sha256": digest}), encoding="utf-8")
    return path


def _selection_args(tmp_path: Path, *, dataset: str = "MOSEv2") -> SimpleNamespace:
    cases = [{"case_id": "a", "video_id": "video-a", "dataset": dataset,
              "official_split": "train", "object_id": 1, "switch_frame": 5,
              "first_prompt_frame": 0},
             {"case_id": "b", "video_id": "video-b", "dataset": dataset,
              "official_split": "train", "object_id": 1, "switch_frame": 5,
              "first_prompt_frame": 0}]
    source = _signed(tmp_path / "source.json", {"dataset": dataset, "split": "train",
                                               "release": "v2", "cases": cases})
    source_digest = json.loads(source.read_text())["content_sha256"]
    common = {"dataset": dataset, "release": "v2", "source_split": "train",
              "source_manifest_content_sha256": source_digest}
    fit = _signed(tmp_path / "fit.json", {**common, "split": "fit", "videos": ["video-a"]})
    dev = _signed(tmp_path / "dev.json", {**common, "split": "development", "videos": ["video-b"]})
    return SimpleNamespace(source_manifest=source, fit_split=fit, development_split=dev,
                           paired_split="all", max_frame_count=None, shard_count=1,
                           shard_index=0, dynamic_queue=False, queue_root=None,
                           max_cases=None, source_model_id="sam2.1_small",
                           target_model_id="sam2.1_base_plus")


def test_selection_rejects_missing_hash_davis_and_split_overlap(tmp_path: Path) -> None:
    collector = _collector_module()
    args = _selection_args(tmp_path)
    selected = collector._selection(args)
    assert selected["split_case_counts"] == {"fit": 1, "development": 1}
    args.fit_split.write_text(json.dumps({"videos": ["video-a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing content_sha256"):
        collector._selection(args)
    args = _selection_args(tmp_path, dataset="DAVIS 2017")
    with pytest.raises(ValueError, match="MOSEv2 or LVOS v2"):
        collector._selection(args)
    args = _selection_args(tmp_path)
    dev = json.loads(args.development_split.read_text())
    dev.pop("content_sha256")
    _signed(args.development_split, {**dev, "videos": ["video-a"]})
    with pytest.raises(ValueError, match="overlap"):
        collector._selection(args)


def test_sparse_frame_mapping_and_command_preserve_prompt_index(tmp_path: Path) -> None:
    collector = _collector_module()
    video = tmp_path / "JPEGImages"
    video.mkdir()
    for frame_id in (1, 26, 1491, 2001):
        (video / f"{frame_id:06d}.jpg").write_bytes(b"jpeg")
    (video / "000010.png").write_bytes(b"not rgb input")
    assert collector._frame_id_to_video_index(video, 1491) == 2
    args = SimpleNamespace(sam2_repo=tmp_path, source_config="small.yaml",
                           source_checkpoint=tmp_path / "s.pt", source_model_id="small",
                           target_config="base.yaml", target_checkpoint=tmp_path / "b.pt",
                           target_model_id="base", device="cpu", seed=7)
    command = collector._prepare_command(
        args, {"object_id": 1}, video, tmp_path / "000026.png", tmp_path / "case.pt",
        runtime_switch_index=2, prompt_frame_index=1,
        generating_path=tmp_path / "generating.json",
    )
    assert command[command.index("--prompt-frame-index") + 1] == "1"
    assert command[command.index("--switch-frame") + 1] == "2"
    assert command[command.index("--generating-json") + 1].endswith("generating.json")
    assert "--state-only" in command and "--active-memory-only" in command
    assert "--future-mask" not in command


def test_deterministic_shards_are_disjoint_and_cover_selection(tmp_path: Path) -> None:
    collector = _collector_module()
    args = _selection_args(tmp_path)
    expected = {case["case_id"] for case in collector._selection(args)["cases"]}
    args.shard_count = 8
    shards = []
    for shard_index in range(args.shard_count):
        args.shard_index = shard_index
        shards.append({case["case_id"] for case in collector._selection(args)["cases"]})
    assert set.union(*shards) == expected
    assert sum(map(len, shards)) == len(expected)


def test_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    collector = _collector_module()
    path = _signed(tmp_path / "manifest.json", {"dataset": "MOSEv2"})
    payload = json.loads(path.read_text())
    payload["dataset"] = "LVOS v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        collector._load_checked(path)


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
