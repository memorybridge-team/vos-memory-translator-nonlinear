"""원본 official 목록을 먼저 고정한다. cache 존재 여부는 membership에 관여하지 않는다."""
from collections import Counter
from pathlib import Path
import hashlib

from .common import require, digest, sha256, read_json, read_bound, save_bound, write_json, atomic, writer

ALGORITHM = "sha256_seed_group_sort_largest_remainder_v1"
ROLES = ("train", "validation", "internal-test")
RATIOS = (90, 5, 5)


def inventory(dataset, official_split, video_root, official_list, groups=None):
    require(dataset in ("MOSEv2", "LVOSv2"), "UNSUPPORTED_DATASET")
    require(official_split in ("train", "valid"), "OFFICIAL_SPLIT")
    root = Path(video_root).resolve()
    source = Path(official_list)
    if source.suffix == ".json":
        value = read_json(source)
        require(value.get("dataset") in (dataset, "LVOS v2" if dataset == "LVOSv2" else dataset), "OFFICIAL_DATASET")
        require(value.get("split") in ({"train"} if official_split == "train" else {"val", "valid", "validation"}), "OFFICIAL_MEMBERSHIP")
        require(isinstance(value.get("videos"), list), "FULL_VIDEO_LIST_REQUIRED_NOT_CASE_DERIVED")
        videos = value["videos"]
    else:
        videos = [v.strip() for v in source.read_text(encoding="utf-8").splitlines() if v.strip()]
    require(videos and len(videos) == len(set(videos)), "DUPLICATE_OR_EMPTY_OFFICIAL_LIST")
    require(all(isinstance(v, str) and v not in (".", "..") and Path(v).name == v and "/" not in v and "\\" not in v for v in videos), "VIDEO_ID")
    present = {p.name for p in root.iterdir() if p.is_dir()}
    require(set(videos) == present, "OFFICIAL_LIST_RGB_INVENTORY_MISMATCH")
    mapping = read_json(groups) if groups else {}
    require(set(mapping) <= set(videos), "GROUP_MAP_UNKNOWN_VIDEO")
    rows = [dict(dataset=dataset, video_id=v, official_split=official_split,
                 original_group=str(mapping.get(v, v)),
                 lineage_status="CONFIRMED" if v in mapping else "UNKNOWN_SINGLE_VIDEO_GROUP") for v in sorted(videos)]
    return dict(schema_version="cmmt.original_video_inventory.v1", dataset=dataset, official_split=official_split,
                videos=rows, source=dict(official_list=str(source.resolve()), sha256=sha256(source),
                rgb_root=str(root), group_map_sha256=sha256(groups) if groups else None),
                duplicate_lineage_status="NOT_FULLY_VERIFIED" if len(mapping) < len(videos) else "OPERATOR_GROUP_MAP")


def allocate(n):
    base = [n * r // 100 for r in RATIOS]
    # 나머지가 같은 경우 train, validation, internal-test 순서.
    order = sorted(range(3), key=lambda i: (-(n * RATIOS[i] % 100), i))
    for i in order[:n - sum(base)]:
        base[i] += 1
    return base


def build_split(original, seed=7):
    require(original.get("schema_version") == "cmmt.original_video_inventory.v1" and
            original.get("dataset") == "MOSEv2" and original.get("official_split") == "train", "MOSE_OFFICIAL_TRAIN_INVENTORY")
    rows = sorted(original["videos"], key=lambda r: (r["dataset"], r["video_id"]))
    require(rows and all(r["dataset"] == "MOSEv2" and r["official_split"] == "train" and r["original_group"] for r in rows), "INVENTORY_ROWS")
    require(len({(r["dataset"], r["video_id"]) for r in rows}) == len(rows), "DUPLICATE_VIDEO")
    groups = sorted({r["original_group"] for r in rows}, key=lambda g: (
        hashlib.sha256(f"{seed}\0MOSEv2\0{g}".encode()).hexdigest(), g))
    counts = allocate(len(groups))
    group_roles = {}
    start = 0
    for role, count in zip(ROLES, counts):
        for g in groups[start:start + count]:
            group_roles[g] = role
        start += count
    membership = [{**r, "role": group_roles[r["original_group"]]} for r in rows]
    return dict(schema_version="cmmt.mose_permanent_split.v1", inventory_digest=digest(original),
                seed=seed, algorithm=ALGORITHM, ratios_percent=dict(zip(ROLES, RATIOS)),
                rounding="largest_remainder_on_groups; tie=train,validation,internal-test",
                sorted_inventory="dataset,video_id", membership=membership,
                group_counts=dict(zip(ROLES, counts)), video_counts=dict(Counter(r["role"] for r in membership)),
                lineage_status=original.get("duplicate_lineage_status", "UNKNOWN"))


def freeze(original, root, seed=7):
    root = Path(root)
    if original.get("source", {}).get("rgb_root"):
        require(not root.resolve().is_relative_to(Path(original["source"]["rgb_root"]).resolve()), "SPLIT_OUTPUT_INSIDE_RAW_RGB")
    proposed = build_split(original, seed)
    with writer(root):
        marker = root / "SPLIT_FROZEN.json"
        if marker.exists():
            manifest = read_split(root)
            require(manifest == proposed, "FROZEN_INPUT_CHANGED_CREATE_EXPLICIT_EXTENSION")
            require(read_bound(root / "original_inventory.json") == original, "FROZEN_INVENTORY_CHANGED")
            return manifest
        require(not any((root / p).exists() for p in ("original_inventory.json", "split_manifest.json", "split_manifest.sha256")), "INCOMPLETE_FREEZE_DO_NOT_OVERWRITE")
        original_sha = save_bound(root / "original_inventory.json", original)
        manifest_sha = save_bound(root / "split_manifest.json", proposed)
        atomic(root / "split_manifest.sha256", (manifest_sha + "\n").encode(), immutable=True)
        write_json(root / "split_report.json", dict(group_counts=proposed["group_counts"], video_counts=proposed["video_counts"],
                   group_overlap=0, cache_used_for_split=False, lineage_status=proposed["lineage_status"],
                   complete_leakage_prevention_claimed=False), immutable=True)
        write_json(marker, dict(status="SPLIT_FROZEN", inventory_sha256=original_sha,
                   split_sha256=manifest_sha, algorithm=ALGORITHM, seed=seed), immutable=True)
        return proposed


def read_split(root):
    root = Path(root)
    frozen = read_json(root / "SPLIT_FROZEN.json")
    manifest = read_bound(root / "split_manifest.json")
    original = read_bound(root / "original_inventory.json")
    require(sha256(root / "split_manifest.json") == frozen["split_sha256"] == (root / "split_manifest.sha256").read_text().strip(), "FROZEN_SHA_CHANGED")
    require(sha256(root / "original_inventory.json") == frozen["inventory_sha256"], "FROZEN_INVENTORY_SHA_CHANGED")
    require(manifest == build_split(original, frozen["seed"]), "FROZEN_MEMBERSHIP_CHANGED")
    return manifest


def role_for(manifest, dataset, video_id):
    rows = [r for r in manifest["membership"] if (r["dataset"], r["video_id"]) == (dataset, video_id)]
    require(len(rows) == 1, "UNKNOWN_VIDEO:" + dataset + ":" + video_id)
    return rows[0]["role"]
