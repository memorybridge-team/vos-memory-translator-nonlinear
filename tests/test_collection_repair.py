"""CPU synthetic 계약 회귀. 실제 SAM weights/배포/continuation parity 증거는 아니다."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time

import numpy as np
import pytest
import torch

from vos_memory_inspector.collection_contract import (
    Rejection, jpeg_map, resolve_indices, frozen_selection, shard_plan,
    memory_policy, apply_memory_policy, SEMANTICS, validate_generating)
from vos_memory_inspector.cache_migration import inspect_cache, import_requests, audit_requests
from vos_memory_inspector.case_cache import write_case_cache, validate_case_cache, CaseOwnership
from vos_memory_inspector.training_storage import sha256, content_hash, write_json, atomic_write
from vos_memory_inspector.training_data import (read_manifest, save_manifest, PairedStateDataset,
    pack_state, load_pair, check_disjoint)
from vos_memory_inspector.training_runner import TrainConfig, train, CheckpointTranslator
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector import roundtrip

ROOT = Path(__file__).resolve().parents[1]


def state(prompt=0, switch=20, value=1.):
    frames = torch.arange(prompt, switch+1).view(1, 1, -1)
    size = frames.shape[2]
    cond = torch.zeros_like(frames, dtype=torch.bool)
    cond[:, :, 0] = True
    return CanonicalState(spatial_memory=torch.full((1,1,size,2,2,2), value),
        object_pointer=torch.full((1,1,size,3), value), presence_logits=torch.zeros(1,1,size,1),
        frame_indices=frames, slot_order=torch.arange(size).view(1,1,size),
        is_conditioning=cond, validity=torch.ones_like(cond), object_ids=(1,), switch_frame=switch)


def selection(dataset="mosev2", ids=None):
    prefix = ROOT / "manifests" / f"{dataset}_train_v1"
    return frozen_selection(str(prefix)+".json", str(prefix)+"_fit.json",
                            str(prefix)+"_development.json", case_ids=ids)


def fixtures(tmp_path, *, legacy=False):
    selected = selection()
    raw = [next(c for c in selected["cases"] if c["paired_split"] == role and int(c["object_id"]) == 1)
           for role in ("fit", "development")]
    selected = selection(ids=[c["case_id"] for c in raw])
    files = tmp_path / "fixture-models"
    files.mkdir(exist_ok=True)
    models = {}
    from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT
    for role, arch in (("source", "small"), ("target", "base_plus")):
        ckpt, cfg = files / f"{role}.pt", files / f"{role}.yaml"
        ckpt.write_bytes(f"synthetic fixture {role}: not SAM weights".encode())
        cfg.write_text("synthetic fixture config: not a SAM config", encoding="utf-8")
        models[role] = {"architecture": arch, "version": "sam2.1", "synthetic": True,
            "checkpoint_sha256": sha256(ckpt), "config_sha256": sha256(cfg),
            "upstream_commit": SUPPORTED_SAM2_COMMIT}
    requests = []
    for c in selected["cases"]:
        frames = [{"runtime_index": i, "official_id": i, "filename": f"{i:06d}.jpg",
                   "sha256": content_hash(["synthetic pixel fixture", i])} for i in range(c["frame_count"])]
        case = {"dataset": "MOSEv2", "release": "v2", "official_split": "train", "video_id": c["video_id"],
            "object_ids": [1], "switch_frame": c["switch_frame"], "pair_mode": "native_history",
            "prompt_conditions": [{"frame_index": 0, "object_id": 1, "kind": "mask",
                                   "sha256": content_hash("synthetic prompt fixture")}],
            "preprocessing": {"image_size": 4, "scope": "synthetic_cpu_fixture"},
            "video_sha256": content_hash(frames), "frame_map_sha256": content_hash(frames),
            "num_frames": len(frames), "case_id": c["case_id"], "official_prompt_frame": c["first_prompt_frame"],
            "official_switch_frame": c["switch_frame"], "paired_split": c["paired_split"],
            "source_manifest_content_sha256": selected["source_manifest_content_sha256"],
            "object_semantics": SEMANTICS, "memory_policy": memory_policy("full_history")}
        read = {"num_maskmem": 7, "max_obj_ptrs_in_encoder": 16, "memory_temporal_stride_for_eval": 1,
                "max_cond_frames_in_attn": -1, "only_obj_ptrs_in_the_past_for_eval": True,
                "use_obj_ptrs_in_encoder": True, "add_all_frames_to_correct_as_cond": False}
        gen = {"schema_version": "cmmt.paired_generating.v1", "synthetic": True, "case": case,
               "models": models, "frame_map": frames, "seed": 7, "collection_mode": "state_only",
               "collector_source_hashes": {"test_collection_repair.py": sha256(__file__)},
               "effective_model_policy": {"source": read, "target": read}, "preprocessing": case["preprocessing"]}
        s, t = state(switch=c["switch_frame"]), state(switch=c["switch_frame"], value=2.)
        output = tmp_path / "original" / (content_hash(c)+".pt")
        meta = {"switch_frame": s.switch_frame, "cache_mode": "state_only", "seed": 7, "prompt_frame_index": 0}
        if not legacy:
            meta["generating"] = gen
        write_case_cache(output, source_canonical=s, target_canonical=t, metadata=meta)
        request = {"path": str(output), "case_id": c["case_id"], "sha256": sha256(output), "expected_generating": gen}
        if legacy:
            proof_path = tmp_path / (output.stem+".run.json")
            write_json(proof_path, {"status": "reviewed_immutable", "reviewer": "synthetic_fixture_only",
                "reviewed_at": "2026-09-29", "source_evidence": "이 테스트에서 cache와 함께 만든 synthetic trace",
                "bindings": [{"cache_sha256": sha256(output), "generating": gen}]})
            request["run_evidence"] = {"path": str(proof_path), "sha256": sha256(proof_path)}
            # ZIP v2와 같이 generating/완료 marker가 없는 legacy 직렬화 fixture.
            Path(str(output)+".complete.json").unlink()
            request["completed_evidence"] = "synthetic test writer 종료 확인"
        requests.append(request)
    return selected, requests


def test_sparse_mapping_jpeg_only_and_late_prompt(tmp_path):
    for i in (1, 26, 51, 1491, 1496):
        (tmp_path/f"{i:05d}.jpg").write_bytes(b"jpeg fixture")
    (tmp_path/"26.png").write_bytes(b"not RGB")
    (tmp_path/"other.txt").touch()
    _, records = jpeg_map(tmp_path)
    case = {"dataset": "LVOS v2", "first_prompt_frame": 26, "switch_frame": 1491}
    assert resolve_indices(case, records) == (1, 3)
    case["first_prompt_frame"] = 1
    assert resolve_indices(case, records) == (0, 3)
    case["first_prompt_frame"] = 2
    with pytest.raises(Rejection, match="MISSING_FRAME_ID"):
        resolve_indices(case, records)
    (tmp_path/"026.jpeg").write_bytes(b"duplicate")
    with pytest.raises(Rejection, match="DUPLICATE_FRAME_ID"):
        jpeg_map(tmp_path)
    (tmp_path/"026.jpeg").unlink()
    (tmp_path/"abc.jpg").touch()
    with pytest.raises(Rejection, match="NON_NUMERIC_JPEG"):
        jpeg_map(tmp_path)


def test_invalid_serialization_keeps_original_and_reports_reason(tmp_path):
    _, requests = fixtures(tmp_path)
    request = requests[0]
    path = Path(request["path"])
    broken = b"not a tensor archive or pickle"
    path.write_bytes(broken)
    request["sha256"] = sha256(path)  # checksum 일치 이후의 직렬화 실패를 검사한다.
    path.with_suffix(path.suffix+".sha256").write_text(sha256(path), encoding="ascii")
    marker = Path(str(path)+".complete.json")
    value = json.loads(marker.read_text())
    value.update(sha256=sha256(path), bytes=len(broken))
    write_json(marker, value)
    result, _, _ = inspect_cache(request, trusted_team_legacy=True, stable_seconds=0)
    assert result["decision"] == "NEEDS_RECOLLECTION"
    assert result["reason_codes"] == ["INVALID_SERIALIZATION"]
    assert path.read_bytes() == broken


def test_inventory_redacts_inline_python_after_options():
    from scripts.inspect_collection_workers import entrypoint
    assert entrypoint(["/venv/bin/python", "-u", "-c", "secret code", "extra"]) == [
        "/venv/bin/python", "-u", "-c [redacted]"]
    assert entrypoint(["/venv/bin/python", "-u", "collect.py", "--shard-index", "0"]) == [
        "/venv/bin/python", "-u", "collect.py"]


class Predictor:
    def __init__(self):
        self.prompts, self.processed = [], []
    def add_new_mask(self, container, *, frame_idx, **kwargs):
        self.prompts.append(frame_idx)
        self._run_single_frame_inference(frame_idx=frame_idx)
    def forward_image(self, x):
        return x
    def _run_single_frame_inference(self, *, frame_idx):
        self.forward_image(torch.zeros(1))
    def propagate_in_video(self, container, *, start_frame_idx, max_frame_num_to_track, **kwargs):
        for i in range(start_frame_idx, min(start_frame_idx+max_frame_num_to_track, container["num_frames"]-1)+1):
            self._run_single_frame_inference(frame_idx=i)
            self.processed.append(i)
            yield i, [1], torch.zeros(1,1,2,2)


@pytest.mark.parametrize("prompt", [0, 5])
def test_prompt_insertion_and_inclusive_prefix_boundary(monkeypatch, prompt):
    monkeypatch.setattr(roundtrip, "canonicalize_sam2_inference_state", lambda *_args, **kw: state(prompt, kw["switch_frame"]))
    for native in (False, True):
        predictor, trace = Predictor(), {}
        fn = roundtrip._collect_native if native else roundtrip._collect_prefix_reference
        extra = {"stop_at_switch": True} if native else {}
        canonical, masks = fn(predictor, {"num_frames": 30}, mask=np.ones((2,2)), object_id=1,
            switch_frame=20, prompt_frame_index=prompt, capture_masks=False, trace=trace, **extra)
        assert predictor.prompts == [prompt] and predictor.processed == list(range(prompt,21))
        assert canonical.frame_indices.min() == prompt and not masks
        assert trace["backbone_calls"] == 22-prompt
        assert max(trace["inference_frame_indices"]) == 20


def test_full_reference_keeps_future_path(monkeypatch):
    monkeypatch.setattr(roundtrip, "canonicalize_sam2_inference_state", lambda *_args, **kw: state(0, kw["switch_frame"]))
    predictor = Predictor()
    _, future = roundtrip._collect_native(predictor, {"num_frames": 25}, mask=np.ones((2,2)), object_id=1, switch_frame=20)
    assert list(future) == [21,22,23,24]


@pytest.mark.parametrize("field", ["frame_indices", "is_conditioning", "validity", "object_ids", "switch_frame"])
def test_common_state_only_alignment_validation(field):
    s, t = state(), state(value=2.)
    if field == "object_ids":
        t.object_ids = (2,)
    elif field == "switch_frame":
        t.switch_frame = 21
    elif field == "frame_indices":
        t.frame_indices[0,0,-1] = 19
    else:
        getattr(t, field)[0,0,0] = not bool(getattr(t, field)[0,0,0])
    with pytest.raises(ValueError):
        validate_case_cache({"schema_version": "cmmt.prepared_handoff_case.v2", "source_canonical": s,
            "target_canonical": t, "metadata": {"cache_mode": "state_only", "switch_frame": 20}})


def test_state_only_metadata_and_finite_rejection():
    s, t = state(), state()
    p = {"schema_version": "cmmt.prepared_handoff_case.v2", "source_canonical": s, "target_canonical": t,
         "metadata": {"cache_mode": "state_only", "switch_frame": 19}}
    with pytest.raises(ValueError, match="metadata switch"):
        validate_case_cache(p)
    p["metadata"]["switch_frame"] = 20
    p["metadata"]["num_frames"] = 21
    with pytest.raises(ValueError, match="no continuation"):
        validate_case_cache(p)
    p["metadata"].pop("num_frames")
    t.spatial_memory[0,0,0,0,0,0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_case_cache(p)


def test_checksum_generating_and_explicit_pickle_trust(tmp_path):
    _, requests = fixtures(tmp_path)
    request = requests[0]
    assert inspect_cache(request, stable_seconds=0)[0]["reason_codes"] == ["TRUST_REQUIRED"]
    changed = deepcopy(request)
    changed["expected_generating"]["seed"] = 8
    result, _, _ = inspect_cache(changed, trusted_team_legacy=True, stable_seconds=0)
    assert result["decision"] == "PENDING_EVIDENCE" and result["reason_codes"] == ["GENERATING_MISMATCH"]
    with Path(request["path"]).open("ab") as stream:
        stream.write(b"tamper")
    assert inspect_cache(request, trusted_team_legacy=True, stable_seconds=0)[0]["reason_codes"] == ["CHECKSUM_TAMPERED"]


def test_ownership_and_interrupted_atomic_write(tmp_path):
    path = tmp_path / "case.pt"
    with CaseOwnership(path):
        with pytest.raises(FileExistsError):
            with CaseOwnership(path):
                pass
    path.write_bytes(b"old complete data")
    def interrupted(stream):
        stream.write(b"partial new data")
        raise RuntimeError("synthetic interrupt")
    with pytest.raises(RuntimeError):
        atomic_write(path, interrupted)
    assert path.read_bytes() == b"old complete data" and not list(tmp_path.glob("*.partial"))


def test_missing_completion_and_uncertain_legacy_not_recollected(tmp_path):
    _, requests = fixtures(tmp_path, legacy=True)
    request = deepcopy(requests[0])
    request.pop("run_evidence")
    result, _, _ = inspect_cache(request, trusted_team_legacy=True, stable_seconds=0)
    assert result["decision"] == "PENDING_EVIDENCE" and result["reason_codes"] == ["LEGACY_RUN_EVIDENCE_MISSING"]
    request.pop("completed_evidence")
    assert inspect_cache(request, trusted_team_legacy=True, stable_seconds=0)[0]["reason_codes"] == ["COMPLETED_EVIDENCE_MISSING"]


def test_valid_legacy_import_idempotent_loader_and_optimizer(tmp_path):
    torch.set_num_threads(1)
    selected, requests = fixtures(tmp_path, legacy=True)
    originals = {r["path"]: sha256(r["path"]) for r in requests}
    imported = tmp_path / "imported"
    result = import_requests(requests, imported, selected, trusted_team_legacy=True, stable_seconds=0)
    assert result["counts"]["CONVERTIBLE"] == 2
    result = import_requests(requests, imported, selected, trusted_team_legacy=True, stable_seconds=0)
    assert result["counts"]["REUSABLE"] == 2
    manifest = read_manifest(imported / "manifest.json")
    assert len(manifest["pairs"]) == 2
    for entry in manifest["pairs"]:
        payload = torch.load(imported / entry["path"], weights_only=True)
        assert isinstance(payload["source"], dict)
    fit = PairedStateDataset(imported, imported/"splits/fit.json", role="fit", pair_mode="native_history")
    dev = PairedStateDataset(imported, imported/"splits/dev.json", role="dev", pair_mode="native_history")
    check_disjoint(fit, dev)
    history = train(fit, dev, tmp_path/"train", TrainConfig(epochs=1, device="cpu", normalization="none"))
    facade = CheckpointTranslator(tmp_path/"train/latest.json", device="cpu")
    source, _, _ = fit[0]
    assert facade.translate(source).spatial_memory.shape == source.spatial_memory.shape
    assert facade.identity["collection_lineage"]["object_semantics"] == SEMANTICS
    assert {p: sha256(p) for p in originals} == originals
    assert len(list((imported/"migration-ledger/events").glob("*.json"))) == 4


def test_policy_conversion_and_policy_mixing_rejection(tmp_path):
    selected, requests = fixtures(tmp_path)
    imported = tmp_path/"active"
    result = import_requests(requests, imported, selected, trusted_team_legacy=True, stable_seconds=0,
                             target_policy="active_window_v1")
    assert result["counts"]["CONVERTIBLE"] == 2
    changed = import_requests(requests[:1], imported, selected, trusted_team_legacy=True, stable_seconds=0)
    assert changed["results"][0]["reason_codes"] == ["MIXED_COLLECTION_POLICY"]
    m = read_manifest(imported/"splits/fit.json")
    duplicate = deepcopy(m["pairs"][0])
    duplicate["case"]["memory_policy"] = memory_policy("full_history")
    duplicate["pair_id"] = "different-pair"
    m["pairs"].append(duplicate)
    save_manifest(imported/"mixed.json", m)
    with pytest.raises(ValueError, match="mixed memory policies"):
        PairedStateDataset(imported, imported/"mixed.json", role="fit", pair_mode="native_history")


def test_selected_memory_archives_slots_and_zero_noncond_limit():
    original = state()
    selected = apply_memory_policy(original, memory_policy("active_window_v1"))
    assert selected.frame_indices.flatten().tolist() == [0]+list(range(6,21))
    maps = selected.metadata["memory_selection"]["old_to_new_slots"][0]
    assert maps[1] == {"old_slot": 6, "new_slot": 1, "runtime_frame": 6, "conditioning": False}
    assert apply_memory_policy(original, memory_policy("active_window_v1",1,1)).frame_indices.item() == 0


@pytest.mark.parametrize("dataset,count", [("mosev2",20841),("lvosv2",1803)])
def test_exact_frozen_splits_and_eight_way_union(dataset, count):
    chosen = selection(dataset)
    assert len(chosen["cases"]) == count
    plan = shard_plan(chosen, 8)
    groups = [set(s["case_ids"]) for s in plan["shards"]]
    assert sum(map(len,groups)) == len(set.union(*groups)) == count
    fit = set(chosen["fit_membership"]["videos"])
    dev = set(chosen["development_membership"]["videos"])
    assert not fit & dev
    assert all((c["video_id"] in fit) == (c["paired_split"] == "fit") for c in chosen["cases"])


def test_append_only_plan_and_selective_recollection(tmp_path):
    _, requests = fixtures(tmp_path)
    missing = {"path": str(tmp_path/"missing.pt"), "case_id": "confirmed-missing-fixture"}
    result = audit_requests([requests[0], missing], tmp_path/"ledger", trusted_team_legacy=True, stable_seconds=0)
    assert result["counts"]["CONVERTIBLE"] == 1 and result["counts"]["NEEDS_RECOLLECTION"] == 1
    assert result["recollection_case_ids"] == ["confirmed-missing-fixture"]
    old = {p: sha256(p) for p in (tmp_path/"ledger/events").glob("*.json")}
    audit_requests([requests[0]], tmp_path/"ledger", trusted_team_legacy=True, stable_seconds=0)
    assert all(sha256(p)==digest for p,digest in old.items())


def test_stride_must_be_declared_and_supported(tmp_path):
    _, requests = fixtures(tmp_path)
    gen = deepcopy(requests[0]["expected_generating"])
    gen["case"]["memory_policy"] = memory_policy("active_window_v1")
    gen["effective_model_policy"]["target"]["memory_temporal_stride_for_eval"] = 5
    with pytest.raises(Rejection, match="UNSUPPORTED_ACTIVE_STRIDE"):
        validate_generating(gen)


def test_confirmed_late_prompt_defect_is_selective_recollection(tmp_path):
    _, requests = fixtures(tmp_path)
    request = deepcopy(requests[0])
    gen = request["expected_generating"]
    gen["case"]["official_prompt_frame"] = 2
    gen["case"]["prompt_conditions"][0]["frame_index"] = 2
    result, _, _ = inspect_cache(request, trusted_team_legacy=True, stable_seconds=0)
    assert result["decision"] == "NEEDS_RECOLLECTION" and result["reason_codes"] == ["PROMPT_TIMING"]


def test_same_semantic_pair_at_second_path_is_not_duplicated(tmp_path):
    import shutil
    chosen, requests = fixtures(tmp_path)
    alias = deepcopy(requests[0])
    original = Path(alias["path"])
    clone = original.with_name("alias.pt")
    for suffix in ("", ".sha256", ".complete.json"):
        shutil.copyfile(str(original)+suffix, str(clone)+suffix)
    alias["path"] = str(clone)
    result = import_requests([requests[0], alias], tmp_path/"imported", chosen,
                              trusted_team_legacy=True, stable_seconds=0)
    assert result["counts"]["REUSABLE"] == 1
    assert len(read_manifest(tmp_path/"imported/manifest.json")["pairs"]) == 1


def test_atomic_completion_failure_preserves_incomplete_original(tmp_path, monkeypatch):
    from vos_memory_inspector import case_cache
    path = tmp_path/"case.pt"
    monkeypatch.setattr(case_cache, "write_json", lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic completion interrupt")))
    with pytest.raises(RuntimeError):
        write_case_cache(path, source_canonical=state(), target_canonical=state(),
                         metadata={"switch_frame": 20, "cache_mode": "state_only"})
    assert path.is_file() and path.with_suffix(".pt.sha256").is_file()
    assert not Path(str(path)+".complete.json").exists()
    with pytest.raises(FileExistsError):
        write_case_cache(path, source_canonical=state(), target_canonical=state(),
                         metadata={"switch_frame": 20, "cache_mode": "state_only"})


def test_active_to_full_does_not_recover_deleted_records(tmp_path):
    chosen, requests = fixtures(tmp_path)
    request = requests[0]
    gen = deepcopy(request["expected_generating"])
    gen["case"]["memory_policy"] = memory_policy("active_window_v1")
    output = tmp_path/"original/active.pt"
    canonical = apply_memory_policy(state(switch=gen["case"]["switch_frame"]), gen["case"]["memory_policy"])
    write_case_cache(output, source_canonical=canonical, target_canonical=canonical,
        metadata={"switch_frame": canonical.switch_frame, "cache_mode": "state_only", "generating": gen})
    req = {"path": str(output), "expected_generating": gen}
    result = import_requests([req], tmp_path/"want-full", chosen, trusted_team_legacy=True,
                             stable_seconds=0, target_policy="full_history")
    assert result["results"][0]["reason_codes"] == ["MISSING_MEMORY_NOT_RECOVERABLE"]


def test_prepared_cli_threads_prompt_and_state_only_flags(tmp_path, monkeypatch):
    from vos_memory_inspector import cli
    captured = {}
    monkeypatch.setattr(cli, "prepare_cross_model_case_reference", lambda **kw: captured.update(kw) or {})
    cli.prepare_handoff_case_main(["--sam2-repo", str(tmp_path), "--source-config", "source", "--source-checkpoint", "s.pt",
        "--source-model-id", "small", "--target-config", "target", "--target-checkpoint", "t.pt",
        "--target-model-id", "base_plus", "--video-dir", str(tmp_path), "--prompt-mask", "mask.png", "--object-id", "2",
        "--switch-frame", "20", "--output", "case.pt", "--prompt-frame-index", "5", "--state-only", "--active-memory-only"])
    assert captured["prompt_frame_index"] == 5 and captured["store_masks"] is False and captured["active_memory_only"] is True


def test_collect_dry_run_resolves_conditions_and_rejects_changed_input(tmp_path, monkeypatch):
    """SAM preflight만 mock한다. JPEG/prompt/hash/binding/CLI 조건은 실제 CPU 경로다."""
    from PIL import Image
    from vos_memory_inspector import paired_collection_cli as cli
    selected, requests = fixtures(tmp_path)
    selected["cases"] = selected["cases"][:1]
    selected["selection_digest"] = content_hash(selected["cases"])
    raw = selected["cases"][0]
    save_manifest(tmp_path/"plan.json", selected)
    video = tmp_path/"data/JPEGImages"/raw["video_id"]
    annotations = tmp_path/"data/Annotations"/raw["video_id"]
    video.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for i in range(raw["frame_count"]):
        Image.fromarray(np.full((4,4,3), i, dtype=np.uint8)).save(video/f"{i:06d}.jpg")
    Image.fromarray(np.ones((4,4), dtype=np.uint8)).save(annotations/"000000.png")
    sam = tmp_path/"synthetic-preflight-only"
    for rel in ("sam2/sam2_video_predictor.py", "sam2/modeling/sam2_base.py", "sam2/build_sam.py", "sam2/utils/misc.py",
                *["sam2/"+c for c in cli.CONFIGS.values()]):
        path = sam/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic preflight placeholder; not executable SAM")
    models = requests[0]["expected_generating"]["models"]
    monkeypatch.setattr(cli, "model_provenance", lambda *_: models)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout=b"synthetic preflight placeholder; not executable SAM", returncode=0))
    write_json(tmp_path/"read.json", requests[0]["expected_generating"]["effective_model_policy"])
    args = cli.parser().parse_args(["collect", "--plan", str(tmp_path/"plan.json"), "--dataset-root", str(tmp_path/"data"),
        "--sam2-repo", str(sam), "--source-checkpoint", "s.pt", "--target-checkpoint", "t.pt",
        "--read-policy-json", str(tmp_path/"read.json"), "--cache-root", str(tmp_path/"new-cache"),
        "--run-root", str(tmp_path/"run"), "--shard-count", "1", "--shard-index", "0", "--dry-run"])
    result = cli.collect(args)
    status = json.loads((Path(result["run_root"])/"status.json").read_text())
    assert status["results"][0]["state"] == "dry_run_ready"
    assert status["results"][0]["generating"]["case"]["object_semantics"] == SEMANTICS
    assert not (tmp_path/"new-cache").exists()
    Image.fromarray(np.full((4,4,3), 99, dtype=np.uint8)).save(video/"000001.jpg")
    with pytest.raises(Rejection, match="WORKER_REJECTIONS"):
        cli.collect(args)
    status = json.loads((Path(result["run_root"])/"status.json").read_text())
    assert "CASE_INPUT_CHANGED" in status["results"][0]["reason"]
