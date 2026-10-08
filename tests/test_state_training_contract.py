"""소규모 stdlib fixture 검사. 실제 데이터 처리나 tensor 학습을 실행하지 않는다."""
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest

from vos_memory_inspector.state_training.common import save_bound, write_json
from vos_memory_inspector.state_training.splits import build_split, freeze, read_split, role_for, allocate
from vos_memory_inspector.state_training.index import assert_roles, plan
from vos_memory_inspector.state_training.r2 import Moments, summarize, score, Coverage, SPEC
from vos_memory_inspector.state_training.policy import Selection, Config
from vos_memory_inspector.state_training.schedule import schedule, refs_for, window_batches


def temporary():
    return tempfile.TemporaryDirectory(dir=os.environ.get("CMMT_TEST_TMP"))


def original(n=100):
    return dict(schema_version="cmmt.original_video_inventory.v1", dataset="MOSEv2", official_split="train",
                videos=[dict(dataset="MOSEv2", video_id=f"v{i:03d}", official_split="train", original_group=f"g{i:03d}") for i in range(n)])


def entry(case, n):
    return dict(case=dict(case_id=case), records=n, valid_indices=[[0, 0, i] for i in range(n)])


def metric(targets, predictions):
    moments = Moments(len(targets[0])); moments.add(targets)
    sse = [math.fsum((p[i] - t[i]) ** 2 for p, t in zip(predictions, targets)) for i in range(len(targets[0]))]
    return summarize(moments.payload(), sse)


class SplitTests(unittest.TestCase):
    def test_deterministic_order_and_counts(self):
        value = original(); reordered = copy.deepcopy(value); reordered["videos"].reverse()
        a, b = build_split(value), build_split(reordered)
        self.assertEqual(a["membership"], b["membership"])
        self.assertEqual(a["group_counts"], {"train": 90, "validation": 5, "internal-test": 5})

    def test_group_never_crosses_roles(self):
        value = original(); value["videos"][1]["original_group"] = "g000"
        split = build_split(value)
        roles = {r["role"] for r in split["membership"] if r["original_group"] == "g000"}
        self.assertEqual(len(roles), 1)
        groups = [{r["original_group"] for r in split["membership"] if r["role"] == role} for role in ("train", "validation", "internal-test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_same_video_objects_cases_records_have_one_role(self):
        split = build_split(original())
        self.assertEqual(len({role_for(split, "MOSEv2", "v010") for _ in range(6)}), 1)

    def test_largest_remainder_small_inventory(self):
        self.assertEqual(allocate(3), [3, 0, 0])
        self.assertEqual(allocate(20), [18, 1, 1])

    def test_frozen_reuse_and_no_seed_overwrite(self):
        with temporary() as d:
            a = freeze(original(), d)
            sha = (Path(d) / "split_manifest.sha256").read_bytes()
            self.assertEqual(freeze(original(), d), a)
            with self.assertRaisesRegex(ValueError, "FROZEN_INPUT_CHANGED"):
                freeze(original(), d, seed=8)
            self.assertEqual((Path(d) / "split_manifest.sha256").read_bytes(), sha)

    def test_changed_inventory_rejected(self):
        with temporary() as d:
            freeze(original(), d)
            with self.assertRaises(ValueError):
                freeze(original(101), d)

    def test_partial_freeze_is_not_overwritten(self):
        with temporary() as d:
            write_json(Path(d) / "split_manifest.json", {"uncommitted": True})
            with self.assertRaisesRegex(ValueError, "INCOMPLETE_FREEZE"):
                freeze(original(), d)

    def test_frozen_tamper_and_unknown_id(self):
        with temporary() as d:
            split = freeze(original(), d)
            with self.assertRaisesRegex(ValueError, "UNKNOWN_VIDEO"):
                role_for(split, "MOSEv2", "unknown")
            (Path(d) / "split_manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "JSON_CHECKSUM"):
                read_split(d)

    def test_cache_exclusion_does_not_change_membership(self):
        split = build_split(original())
        membership = copy.deepcopy(split["membership"])
        available = membership[15:]
        for row in available:
            self.assertEqual(role_for(split, row["dataset"], row["video_id"]), row["role"])
        self.assertEqual(split["membership"], membership)

    def test_internal_test_and_wrong_role_blocked(self):
        split = build_split(original())
        row = next(r for r in split["membership"] if r["role"] == "internal-test")
        for role in ("internal-test", "train", "validation"):
            e = dict(case=dict(row, training_role=role))
            with self.assertRaises(ValueError):
                assert_roles([e], split)

    def test_external_dataset_blocked(self):
        with self.assertRaises(ValueError):
            assert_roles([dict(case=dict(dataset="DAVIS", video_id="x", official_split="train", training_role="train"))], build_split(original()))


class R2Tests(unittest.TestCase):
    def test_exact_prediction_is_one(self):
        y = [[1, 2], [3, 7], [5, 8]]
        self.assertEqual(metric(y, y)["r2"], 1)

    def test_mean_prediction_is_zero(self):
        y = [[1, 2], [3, 7], [5, 9]]
        p = [[3, 6]] * 3
        self.assertAlmostEqual(metric(y, p)["r2"], 0)

    def test_worse_than_mean_is_negative(self):
        self.assertLess(metric([[1], [3], [5]], [[20], [20], [20]])["r2"], 0)

    def test_variance_weighted_not_dimension_average(self):
        m = metric([[0, 0], [2, 20]], [[0, 0], [0, 20]])
        self.assertAlmostEqual(m["r2"], 1 - 4 / 202)

    def test_constant_and_small_variance_excluded(self):
        y = [[2, 3, 0], [2, 3 + 1e-8, 2]]
        m = metric(y, [[0, 0, 0], [0, 0, 2]])
        self.assertEqual(m["excluded_dimensions"], 2)
        self.assertEqual(m["r2"], 1)
        self.assertGreater(m["excluded_dimension_mse"], 0)
        self.assertIsNone(metric([[1, 1], [1, 1]], [[1, 1], [1, 1]])["r2"])

    def test_batch_and_shard_partition_invariant(self):
        y = [[float(i), float(i * i)] for i in range(13)]
        a = Moments(2); a.add(y)
        shards = []
        for section in (y[:2], y[2:7], y[7:]):
            m = Moments(2); m.add(section); shards.append(m)
        b = Moments(2)
        for m in shards:
            b.merge(**m.payload())
        self.assertEqual(a.count, b.count)
        for x, z in zip(a.mean + a.m2, b.mean + b.m2):
            self.assertAlmostEqual(x, z, places=9)

    def test_stable_large_offset(self):
        m = Moments(1); m.add([[1e12 + i] for i in range(5)])
        self.assertEqual(m.m2, [10.0])

    def test_four_required_equal_weights(self):
        metrics = {"MOSEv2": {"spatial": {"r2": 0.8}, "pointer": {"r2": 0.6}},
                   "LVOSv2": {"spatial": {"r2": 0.4}, "pointer": {"r2": -0.2}}}
        self.assertAlmostEqual(score(metrics), 0.4)
        with self.assertRaises(ValueError):
            score({"MOSEv2": metrics["MOSEv2"]})
        del metrics["LVOSv2"]["pointer"]
        with self.assertRaises(ValueError):
            score(metrics)

    def test_nonfinite_and_undefined_block_promotion(self):
        for bad in (None, math.nan, math.inf):
            values = {d: {b: {"r2": bad} for b in ("spatial", "pointer")} for d in ("MOSEv2", "LVOSv2")}
            with self.assertRaises(ValueError):
                score(values)
        with self.assertRaises(ValueError):
            summarize(dict(count=2, mean=[0], m2=[1]), [math.nan])

    def test_duplicate_and_tail_rejected(self):
        cov = Coverage([[0], [1], [2]])
        cov.add([[0], [1]])
        with self.assertRaises(ValueError):
            cov.finish()
        with self.assertRaises(ValueError):
            cov.add([[1]])
        cov.add([[2]]); self.assertEqual(cov.finish(), 3)


class PlanTests(unittest.TestCase):
    def fixture(self, root):
        specs = []
        mose = original(20)
        mose["source"] = dict(rgb_root=str(root / "mose-rgb"))
        for dataset, official, inv in (("MOSEv2", "train", mose),
                ("LVOSv2", "train", dict(videos=[dict(dataset="LVOSv2", video_id="ltrain", original_group="ltrain", official_split="train")], source=dict(rgb_root="unused"))),
                ("LVOSv2", "valid", dict(videos=[dict(dataset="LVOSv2", video_id="lvalid", original_group="lvalid", official_split="valid")], source=dict(rgb_root="unused")))):
            inv = dict(inv, schema_version="cmmt.original_video_inventory.v1", dataset=dataset, official_split=official)
            path = root / (dataset + official + "_inventory.json"); save_bound(path, inv)
            names = [r["video_id"] for r in inv["videos"]]
            case_path = root / (dataset + official + "_cases.json")
            write_json(case_path, dict(dataset=dataset, split=official,
                selection_policy=dict(switch_frame_indexing="zero_based_sorted_rgb_order" if dataset == "MOSEv2" else "official_lvos_frame_id"),
                cases=[dict(dataset=dataset, official_split=official, video_id=v, object_id=1,
                       first_prompt_frame=0, switch_frame=5, case_id=v + ":obj1:switch5") for v in names]))
            spec = dict(dataset=dataset, official_split=official, inventory=str(path), case_manifest=str(case_path),
                        filename_pattern="{video_id}_obj{object_id}_switch{switch_frame}.pt")
            if official == "train":
                spec["storage_membership"] = {}; spec["cache_roots"] = {}
                for origin, values in (("fit", names[::2]), ("development", names[1::2])):
                    membership = root / (dataset + "_" + origin + ".json")
                    write_json(membership, dict(dataset=dataset, source_split="train", split=origin, videos=values))
                    spec["storage_membership"][origin] = str(membership)
                    spec["cache_roots"][origin] = str(root / origin)
            else:
                spec["cache_roots"] = dict(validation=str(root / "validation"))
            specs.append(spec)
        return dict(datasets=specs), build_split(mose)

    def test_storage_folder_is_not_training_role(self):
        with temporary() as d:
            cfg, split = self.fixture(Path(d))
            items, _, _ = plan(cfg, split)
            self.assertEqual(len(items), 22)
            for item in items:
                c = item["case"]
                if c["dataset"] == "MOSEv2":
                    self.assertEqual(c["training_role"], role_for(split, "MOSEv2", c["video_id"]))
            self.assertEqual(sum(e["case"]["training_role"] == "internal-test" for e in items), 1)
            self.assertTrue(any(e["case"]["origin_storage"] == "development" and e["case"]["training_role"] == "train" for e in items))

    def test_unknown_case_video_and_changed_inventory_rejected(self):
        with temporary() as d:
            cfg, split = self.fixture(Path(d))
            path = Path(cfg["datasets"][0]["case_manifest"])
            import json
            value = json.loads(path.read_text())
            value["cases"][0]["video_id"] = "unknown"
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "FOREIGN_CASE"):
                plan(cfg, split)

    def test_lvos_train_valid_group_overlap_rejected(self):
        with temporary() as d:
            cfg, split = self.fixture(Path(d))
            spec = cfg["datasets"][2]
            from vos_memory_inspector.state_training.common import read_bound
            value = read_bound(spec["inventory"])
            value["videos"][0]["original_group"] = "ltrain"
            import json
            write_json(spec["inventory"], value)
            from vos_memory_inspector.state_training.common import sha256
            Path(spec["inventory"] + ".sha256").write_text(sha256(spec["inventory"]) + "\n")
            with self.assertRaisesRegex(ValueError, "LVOS_ORIGINAL_GROUP_OVERLAP"):
                plan(cfg, split)


class PolicyTests(unittest.TestCase):
    def test_max_direction_and_tie_lower_epoch(self):
        state = Selection()
        self.assertTrue(state.observe(1, -2)[0])
        self.assertTrue(state.observe(2, -1)[0])
        self.assertFalse(state.observe(3, -1 + SPEC["tie_tolerance"] / 2)[0])
        self.assertEqual(state.best_epoch, 2)

    def test_best_and_meaningful_improvement_are_distinct(self):
        state = Selection(min_epochs=1, patience=2)
        state.observe(1, 0)
        promoted, stop = state.observe(2, 0.0005)
        self.assertTrue(promoted); self.assertFalse(stop)
        self.assertEqual(state.reference, 0)
        self.assertTrue(state.observe(3, 0.0006)[1])

    def test_resume_early_stopping_no_reset(self):
        first = Selection(min_epochs=1, patience=2); first.observe(1, -1); first.observe(2, -1)
        resumed = Selection(min_epochs=1, patience=2); resumed.load_state_dict(first.state_dict())
        self.assertTrue(resumed.observe(3, -1)[1])
        with self.assertRaises(ValueError):
            Selection(min_delta=0.2).load_state_dict(first.state_dict())

    def test_ties_use_lowest_epoch_within_running_max_band(self):
        state = Selection()
        state.observe(1, 0)
        state.observe(2, 0.8e-12)
        self.assertEqual(state.best_epoch, 1)
        state.observe(3, 1.5e-12)
        self.assertEqual(state.best_epoch, 2)

    def test_config_world_accumulation(self):
        cfg = Config(microbatch=16)
        for world in (1, 2, 4):
            cfg.validate(world)
            self.assertEqual(cfg.global_batch // (world * cfg.microbatch), {1: 4, 2: 2, 4: 1}[world])
        with self.assertRaises(ValueError):
            Config(global_batch=32).validate(2)
        self.assertEqual(Config().global_batch // (2 * Config().microbatch), 1)


class ScheduleTests(unittest.TestCase):
    def test_world_tail_exact_coverage(self):
        entries = [entry("a", 17), entry("b", 16), entry("c", 36)]
        expected = {tuple(r) for r in refs_for(entries)}
        for world in (1, 2, 4):
            parts, windows = schedule(entries, world)
            cov = Coverage(list(expected))
            for part in parts:
                cov.add(refs_for(part))
            self.assertEqual(cov.finish(), 69)
            self.assertEqual(list(map(sum, windows)), [64, 5])

    def test_internal_record_order_and_complete_case_shuffle(self):
        parts, _ = schedule([entry(str(i), 7) for i in range(20)], 2)
        for part in parts:
            for segment in part:
                self.assertEqual(segment["selected_positions"], sorted(segment["selected_positions"]))

    def test_accumulation_window_does_not_cross_boundary(self):
        batches = list(window_batches([32, 5], 16))
        self.assertEqual(list(map(len, batches)), [16, 16, 5])
        self.assertEqual(sum(batches, []), list(range(37)))


if __name__ == "__main__":
    unittest.main()
