"""새 역할의 selection/index. frozen split은 입력이며 cache 상황으로 바꾸지 않는다."""
from collections import Counter
from pathlib import Path
from .common import require, sha256, digest, read_json, read_bound, save_bound, write_json, writer, stamp, reject_placeholders
from .splits import read_split, role_for

SCHEMA = "cmmt.frozen_split_state_index.v1"
SOURCES = {("MOSEv2", "train"), ("LVOSv2", "train"), ("LVOSv2", "valid")}


def plan(config, split):
    reject_placeholders(config)
    specs = config["datasets"]
    require(len(specs) == 3 and {(s["dataset"], s["official_split"]) for s in specs} == SOURCES, "THREE_OFFICIAL_SOURCES_REQUIRED")
    items, inventories, bindings = [], {}, []
    for spec in specs:
        original = read_bound(spec["inventory"])
        dataset, official = spec["dataset"], spec["official_split"]
        require(original["dataset"] == dataset and original["official_split"] == official, "INVENTORY_SOURCE_ROLE")
        videos = {r["video_id"] for r in original["videos"]}
        inventories[(dataset, official)] = original
        if dataset == "MOSEv2":
            require(digest(original) == split["inventory_digest"], "SPLIT_INVENTORY_BINDING")
        selected = read_json(spec["case_manifest"])
        require(selected["dataset"] in (dataset, "LVOS v2" if dataset == "LVOSv2" else dataset) and
                selected["split"] in ({"train"} if official == "train" else {"val", "valid", "validation"}), "CASE_MANIFEST_OFFICIAL_SPLIT")
        expected_indexing = "zero_based_sorted_rgb_order" if dataset == "MOSEv2" else "official_lvos_frame_id"
        require(selected["selection_policy"]["switch_frame_indexing"] == expected_indexing, "CASE_FRAME_INDEXING")
        parts = {}
        if official == "train":
            require(set(spec["cache_roots"]) == {"fit", "development"} and set(spec["storage_membership"]) == {"fit", "development"}, "STORAGE_LOCATIONS")
            for origin, path in spec["storage_membership"].items():
                m = read_json(path)
                require(m["dataset"] == selected["dataset"] and m["source_split"] == "train" and m["split"] == origin, "STORAGE_MANIFEST_SOURCE")
                require(len(m["videos"]) == len(set(m["videos"])), "DUPLICATE_STORAGE_VIDEO")
                parts[origin] = set(m["videos"])
            require(not parts["fit"] & parts["development"] and (parts["fit"] | parts["development"]) <= videos, "STORAGE_MEMBERSHIP")
        else:
            require(set(spec["cache_roots"]) == {"validation"}, "VALIDATION_STORAGE")
        bindings.append(dict(spec=spec, inventory_sha256=sha256(spec["inventory"]),
                             manifest_sha256=sha256(spec["case_manifest"]), storage_sha256={k: sha256(p) for k, p in spec.get("storage_membership", {}).items()}))
        for c in selected["cases"]:
            require(c["video_id"] in videos and c["official_split"] in ({"train"} if official == "train" else {"val", "valid", "validation"}) and
                    c["dataset"] == selected["dataset"], "FOREIGN_CASE_OFFICIAL_MEMBERSHIP")
            role = role_for(split, dataset, c["video_id"]) if dataset == "MOSEv2" else ("train" if official == "train" else "validation")
            origins = [k for k, v in parts.items() if c["video_id"] in v] if official == "train" else ["validation"]
            require(len(origins) == 1, "CASE_STORAGE_MEMBERSHIP_MISSING")
            case = dict(c, dataset=dataset, original_case_id=c["case_id"], case_id=dataset + "|" + c["case_id"],
                        training_role=role, origin_storage=origins[0])
            name = spec["filename_pattern"].format(**case)
            require(Path(name).name == name and name.endswith(".pt"), "CACHE_FILENAME")
            items.append(dict(case=case, path=str((Path(spec["cache_roots"][origins[0]]) / name).resolve()),
                              rgb_root=original["source"]["rgb_root"]))
    ids = [e["case"]["case_id"] for e in items]
    require(len(ids) == len(set(ids)), "DUPLICATE_CASE")
    train = {r["video_id"] for r in inventories[("LVOSv2", "train")]["videos"]}
    valid = {r["video_id"] for r in inventories[("LVOSv2", "valid")]["videos"]}
    require(not train & valid, "LVOS_TRAIN_VALID_VIDEO_OVERLAP")
    require(not {r["original_group"] for r in inventories[("LVOSv2", "train")]["videos"]} &
            {r["original_group"] for r in inventories[("LVOSv2", "valid")]["videos"]}, "LVOS_ORIGINAL_GROUP_OVERLAP")
    return items, bindings, inventories


def prior_entries(config):
    """명시적으로 SHA를 지정한 기존 tensor-validated index 증빙만 재사용한다."""
    if "verified_prior_index" not in config:
        return {}, None
    source = config["verified_prior_index"]
    require(sha256(source["path"]) == source["sha256"], "PRIOR_INDEX_SHA_CHANGED")
    value = read_json(source["path"])
    require(value.get("schema_version") == "cmmt.official_state_training_index.v1" and
            value.get("state") == "tensor_validated", "VERIFIED_PRIOR_INDEX_SCHEMA")
    entries = {e["path"]: e for e in value["entries"]}
    require(len(entries) == len(value["entries"]), "PRIOR_DUPLICATE_PATH")
    return entries, dict(path=source["path"], sha256=source["sha256"], status="OLD_INSPECTION_EVIDENCE_NOT_NEW_R2_CONTRACT")


def reuse_prior(prior, item):
    """content/semantic binding이 같을 때 역할만 새 manifest에서 적용한다."""
    c, old = item["case"], prior["case"]
    require(prior["path"] == item["path"] and stamp(item["path"]) == prior["stamp"] and
            Path(item["path"] + ".sha256").read_text().split()[0] == prior["sha256"] and
            not Path(item["path"] + ".writer.lock").exists(), "PRIOR_CACHE_CHANGED_INSPECTION_REQUIRED")
    require(all(str(old[k]) == str(c[k]) for k in ("video_id", "object_id", "first_prompt_frame", "switch_frame")) and
            old["dataset"] in (c["dataset"], "LVOS v2" if c["dataset"] == "LVOSv2" else c["dataset"]), "PRIOR_SEMANTIC_BINDING")
    require(prior["prompt_contract_status"] == "MATCH" and set(prior["conditioning_frames_observed"]) == {prior["runtime_prompt_frame"]}, "PRIOR_PROMPT_INSPECTION_NOT_VERIFIED")
    require(prior["historical_provenance"].get("known", {}).get("pair_mode", "native_history") == "native_history", "PRIOR_CONTROLLED_PAIR_FORBIDDEN")
    require(prior["records"] == len(prior["valid_indices"]) > 0 and len({tuple(x) for x in prior["valid_indices"]}) == prior["records"], "PRIOR_RECORD_ALIGNMENT")
    import hashlib
    import json
    files = [p for p in (Path(item["rgb_root"]) / c["video_id"]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    require(files and all(p.stem.isascii() and p.stem.isdigit() for p in files), "PRIOR_RGB_MAP_REQUIRED")
    frames = sorted(int(p.stem) for p in files)
    legacy_digest = hashlib.sha256(json.dumps(frames, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(prior["frame_map_digest"] == legacy_digest and len(set(frames)) == len(frames), "PRIOR_RGB_FRAME_MAP_CHANGED")
    prompt, switch = (int(c["first_prompt_frame"]), int(c["switch_frame"])) if c["dataset"] == "MOSEv2" else (frames.index(c["first_prompt_frame"]), frames.index(c["switch_frame"]))
    require(prompt == prior["runtime_prompt_frame"] and switch == prior["runtime_switch_frame"], "PRIOR_RGB_PROMPT_SWITCH_CHANGED")
    # 이전 normalization은 재사용하지 않는다. 새 train의 실제 tensor로 계산한다.
    value = {k: v for k, v in prior.items() if k not in ("case", "normalization_sums", "content_sha256")}
    value.update(case=c, policy=prior["memory_policy"], frame_map_digest=digest(frames),
                 inspection_evidence="PRIOR_TENSOR_INSPECTION_REUSED", prior_index_case_id=old["case_id"])
    return value


def assert_roles(entries, membership):
    seen = set()
    for e in entries:
        c = e["case"]
        require(c["training_role"] in ("train", "validation"), "INTERNAL_TEST_FORBIDDEN_IN_INDEX")
        if c["dataset"] == "MOSEv2":
            require(c["official_split"] == "train", "MOSE_OFFICIAL_TRAIN_ONLY")
            require(c["training_role"] == role_for(membership, "MOSEv2", c["video_id"]), "CACHE_SPLIT_ROLE_MISMATCH")
        else:
            require(c["dataset"] == "LVOSv2" and c["official_split"] in ("train", "val", "valid", "validation") and
                    c["training_role"] == ("train" if c["official_split"] == "train" else "validation"), "EXTERNAL_OR_ROLE_MISMATCH")
        key = (c["dataset"], c["video_id"])
        require((key, "validation" if c["training_role"] == "train" else "train") not in seen, "VIDEO_ROLE_OVERLAP")
        seen.add((key, c["training_role"]))


def build_index(config_path, split_root, output, *, operator_completed, stable_seconds=60):
    from .cache import inspect
    import torch
    torch.set_num_threads(1)
    require(bool(operator_completed), "OPERATOR_COMPLETION_CONFIRMATION_REQUIRED")
    config = read_json(config_path)
    split = read_split(split_root)
    items, sources, originals = plan(config, split)
    prior, prior_binding = prior_entries(config)
    output = Path(output).resolve()
    roots = [Path(r).resolve() for s in config["datasets"] for r in s["cache_roots"].values()]
    roots.extend(Path(inv["source"]["rgb_root"]).resolve() for inv in originals.values())
    require(all(not output.is_relative_to(r) for r in roots), "OUTPUT_INSIDE_RAW_CACHE")
    exclusions = config.get("exclusions", [])
    excluded = {e["case_id"]: e["reason"] for e in exclusions}
    require(len(excluded) == len(exclusions) and all(excluded.values()) and set(excluded) <= {e["case"]["case_id"] for e in items}, "EXCLUSIONS_EXPLICIT_CASE_REASON")
    identity = dict(contract="cmmt.frozen_split_validation_r2.v1", split_sha256=sha256(Path(split_root) / "split_manifest.json"),
                    split_root=str(Path(split_root).resolve()), config_sha256=sha256(config_path), sources=sources,
                    operator_completed=operator_completed, exclusions=exclusions, stable_seconds=stable_seconds, prior_index=prior_binding)
    with writer(output):
        if (output / "INPUTS.json").exists():
            require(read_json(output / "INPUTS.json") == identity, "INDEX_INPUT_CHANGED_USE_NEW_NAMESPACE")
        else:
            write_json(output / "INPUTS.json", identity, immutable=True)
        if (output / "index.json").exists():
            return read_index(output / "index.json")
        entries, inventory, failures = [], [], []
        for i, item in enumerate(items):
            c = item["case"]
            reason = "INTERNAL_TEST_SEALED" if c["training_role"] == "internal-test" else excluded.get(c["case_id"])
            available = Path(item["path"]).is_file()
            row = dict(case=c, path=item["path"], cache_present=available, included=False, reason=reason)
            if not reason and not available:
                # missing을 명시한다. membership을 이동하거나 보충하지 않는다.
                row["reason"] = "MISSING_CACHE"
            elif not reason:
                journal = output / f"cases/{i:06d}.json"
                try:
                    if journal.exists():
                        e = read_bound(journal)
                        require(e["case"] == c and e["path"] == item["path"] and stamp(item["path"]) == e["stamp"], "INDEX_RESUME_CACHE_CHANGED")
                        require(Path(item["path"] + ".sha256").read_text().split()[0] == e["sha256"], "INDEX_RESUME_CACHE_SHA_CHANGED")
                    else:
                        if item["path"] in prior and stamp(item["path"]) == prior[item["path"]]["stamp"] and Path(item["path"] + ".sha256").read_text().split()[0] == prior[item["path"]]["sha256"]:
                            e = reuse_prior(prior[item["path"]], item)
                        else:
                            e = inspect(c, item["path"], item["rgb_root"], stable_seconds=stable_seconds)
                        save_bound(journal, e)
                    entries.append(e)
                    row["included"] = True
                    row["records"] = e["records"]
                except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                    row["reason"] = str(exc)
                    failures.append(row)
            inventory.append(row)
            if (i + 1) % 50 == 0:
                print(f"CPU cache 검사 {i + 1}/{len(items)}; valid_cases={len(entries)}; failures={len(failures)}", flush=True)
        write_json(output / "INSPECTION.json", dict(status="FAIL" if failures else "TENSOR_INDEXED", failures=failures,
                   original_modified=False, historical_provenance="SEE_CASE_KNOWN_UNKNOWN_SEPARATELY"))
        write_json(output / "used_excluded.json", inventory)
        require(not failures, "CACHE_ALIGNMENT_FAILURES_SEE_INSPECTION")
        assert_roles(entries, split)
        require({(e["case"]["dataset"], e["case"]["training_role"]) for e in entries} ==
                {(d, r) for d in ("MOSEv2", "LVOSv2") for r in ("train", "validation")}, "REQUIRED_DATASET_ROLE_EMPTY")
        require(len({digest(e["policy"]) for e in entries}) == 1, "MIXED_MEMORY_POLICY")
        stats = {}
        for dataset in ("MOSEv2", "LVOSv2"):
            for role in ("train", "validation", "internal-test"):
                rows = [r for r in inventory if (r["case"]["dataset"], r["case"]["training_role"]) == (dataset, role)]
                original_rows = [r for inv in originals.values() for r in inv["videos"] if r["dataset"] == dataset and
                                 (role_for(split, dataset, r["video_id"]) if dataset == "MOSEv2" else ("train" if r["official_split"] == "train" else "validation")) == role]
                used = [e for e in entries if (e["case"]["dataset"], e["case"]["training_role"]) == (dataset, role)]
                stats[dataset + "/" + role] = dict(original_videos=len(original_rows), original_groups=len({r["original_group"] for r in original_rows}),
                    available_videos=len({r["case"]["video_id"] for r in rows if r["cache_present"]}),
                    videos_without_selected_cases=len({r["video_id"] for r in original_rows} - {r["case"]["video_id"] for r in rows}),
                    videos_without_available_cache=len({r["video_id"] for r in original_rows} - {r["case"]["video_id"] for r in rows if r["cache_present"]}),
                    available_cases=sum(r["cache_present"] for r in rows), used_videos=len({e["case"]["video_id"] for e in used}),
                    expected_cases=len(rows), used_cases=len(used), valid_records=sum(e["records"] for e in used),
                    missing_cache=sum(not r["cache_present"] for r in rows), exclusion_counts=dict(Counter(r["reason"] for r in rows if r["reason"])))
        value = dict(schema_version=SCHEMA, identity=identity, entries=entries, statistics=stats,
                     used_excluded_sha256=sha256(output / "used_excluded.json"),
                     synthetic_declared=any(e["synthetic_declared"] for e in entries),
                     inspection_counts=dict(Counter(e.get("inspection_evidence", "NEW_TENSOR_INSPECTION") for e in entries)),
                     normalization_not_computed=True, validation_target_stats_not_computed=True)
        save_bound(output / "index.json", value)
        return read_index(output / "index.json")


def read_index(path):
    value = read_bound(path)
    require(value["schema_version"] == SCHEMA and value["identity"]["contract"] == "cmmt.frozen_split_validation_r2.v1", "NEW_CONTRACT_INDEX_REQUIRED")
    split = read_split(value["identity"]["split_root"])
    require(sha256(Path(value["identity"]["split_root"]) / "split_manifest.json") == value["identity"]["split_sha256"], "INDEX_SPLIT_SHA_CHANGED")
    require(sha256(Path(path).parent / "used_excluded.json") == value["used_excluded_sha256"], "USED_EXCLUDED_CHANGED")
    assert_roles(value["entries"], split)
    require({(e["case"]["dataset"], e["case"]["training_role"]) for e in value["entries"]} ==
            {(d, r) for d in ("MOSEv2", "LVOSv2") for r in ("train", "validation")}, "REQUIRED_DATASET_ROLE_EMPTY")
    for source in value["identity"]["sources"]:
        spec = source["spec"]
        require(sha256(spec["inventory"]) == source["inventory_sha256"] and sha256(spec["case_manifest"]) == source["manifest_sha256"] and
                all(sha256(spec["storage_membership"][k]) == h for k, h in source["storage_sha256"].items()), "SOURCE_MANIFEST_CHANGED")
    prior = value["identity"].get("prior_index")
    require(prior is None or sha256(prior["path"]) == prior["sha256"], "PRIOR_INDEX_EVIDENCE_CHANGED")
    require(len({e["case"]["case_id"] for e in value["entries"]}) == len(value["entries"]), "DUPLICATE_INDEX_CASE")
    for e in value["entries"]:
        require(e["records"] > 0 and e["records"] == len(e["valid_indices"]) == len({tuple(x) for x in e["valid_indices"]}), "INDEX_RECORD_ALIGNMENT")
        require(stamp(e["path"]) == e["stamp"], "CACHE_CHANGED_NEW_INDEX_REQUIRED")
        require(not Path(e["path"] + ".writer.lock").exists() and Path(e["path"] + ".sha256").read_text().split()[0] == e["sha256"], "CACHE_WRITER_OR_SHA_CHANGED")
    return value
