from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S3_orchestrator as orch
from stage_S3_amendment_recovery import _tree_digest, revalidate_wave2_archive, validate_receipt
from stage_S3_protocol import (
    ACCESS, FULL, FRESH_SAMPLE_FULL_ROLE, FRESH_TARGET_ROLE,
    build_fresh_matrix, build_reuse_matrix, fresh_formal_key,
    sample_full_replacement_key,
)


S1_ROOT = os.path.join(HERE, "stage_S1_raw_results")
S2_ROOT = os.path.join(HERE, "stage_S2_raw_results")
RAW_ROOT = os.path.join(HERE, "stage_S3_raw_results")
QUEUE_PATH = os.path.join(
    RAW_ROOT, ".stage_S3_archives", "s3_kuairand_natural_seed2022",
    "parent_abort_79c649abb1994ee81d49_wave_1", "queue_snapshot.json")
MODEL_KEY = "m3oe/MDMTRec/models/multi_domain/model.py"
MODEL_SHA = "a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1"


class NonfiniteInventoryTests(unittest.TestCase):
    def _with_value(self, value):
        temp = tempfile.TemporaryDirectory(prefix="stage_s3_nonfinite_")
        self.addCleanup(temp.cleanup)
        path = os.path.join(temp.name, "nonfinite_inventory.json")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(value)
        return path

    def test_empty_array_is_legal_and_legacy_empty_object_is_opt_in_only(self):
        empty_list = self._with_value("[]")
        empty_object = self._with_value("{}")
        self.assertEqual(orch._read_nonfinite_inventory(empty_list), [])
        self.assertIsNone(orch._nonfinite_inventory_failure(empty_list))
        with self.assertRaisesRegex(ValueError, "JSON array"):
            orch._read_nonfinite_inventory(empty_object)
        self.assertEqual(orch._read_nonfinite_inventory(empty_object, allow_empty_object=True), [])
        self.assertEqual(orch._nonfinite_inventory_failure(
            empty_object, allow_empty_object=True), None)

    def test_wrong_types_nonempty_events_duplicate_keys_and_nonstandard_constants_fail(self):
        valid_event = {"file": "metrics.csv", "row_index": 2, "row_key": {"seed": 2022},
                       "field": "loss", "value": "NaN", "reason": "nonfinite_training_loss"}
        cases = [
            "null", "1", '"[]"',
            json.dumps([dict(valid_event, row_index=True)]),
            json.dumps([dict(valid_event, extra="wrong")]),
            json.dumps([dict(valid_event, value="Infinity")]),
            json.dumps([dict(valid_event, reason="unknown")]),
            '{"x": 1, "x": 2}', '[NaN]',
        ]
        for body in cases:
            path = self._with_value(body)
            with self.subTest(body=body):
                self.assertEqual(orch._nonfinite_inventory_failure(path),
                                 "nonfinite_inventory_invalid")
        nonempty = self._with_value(json.dumps([valid_event]))
        self.assertEqual(orch._nonfinite_inventory_failure(nonempty),
                         "nonfinite_inventory_not_empty")


class AmendmentPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage_s3_source_pair_")
        self.addCleanup(self.temp.cleanup)
        self.raw_root = os.path.join(self.temp.name, "raw")
        self.tasks = build_fresh_matrix(self.raw_root)
        self.task_by_key = {str(task["formal_run_key"]): task for task in self.tasks}
        reuse = build_reuse_matrix(S1_ROOT, S2_ROOT)
        self.equal_full = next(task for task in reuse if task["method"] == "equal_full"
                               and task["dataset"] == "kuairand"
                               and task["setting"] == "natural" and task["seed"] == 2022)
        with open(os.path.join(self.equal_full["source_run_dir"], "run_manifest.json"),
                  "r", encoding="utf-8") as stream:
            self.equal_full_manifest = json.load(stream)

    def _write_manifest(self, task, manifest):
        run_dir = os.path.join(self.temp.name, "fixtures", str(task["formal_run_key"]).replace("|", "_"))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True)
        return run_dir

    def test_sample_target_and_fresh_full_replacement_strict_pairing_and_corruption(self):
        sample_target = self.task_by_key[fresh_formal_key("sample", "kuairand", "natural", 2022)]
        sample_full = self.task_by_key[sample_full_replacement_key("kuairand", "natural", 2022)]
        target_manifest = dict(self.equal_full_manifest)
        target_manifest.update({"objective": "sample_population_mean_task_bce",
                                "objective_formula": "mean_i(mean_t(BCE(p_it,y_it)))",
                                "shared_access": ACCESS,
                                "formal_run_key": sample_target["formal_run_key"]})
        full_manifest = dict(target_manifest)
        full_manifest.update({"shared_access": FULL,
                              "formal_run_key": sample_full["formal_run_key"]})
        target_dir = self._write_manifest(sample_target, target_manifest)
        full_dir = self._write_manifest(sample_full, full_manifest)
        target_for_pair = dict(sample_target, run_dir=target_dir)
        full_for_pair = dict(sample_full, run_dir=full_dir, target_only_run_dir=target_dir)
        valid, reasons = orch.validate_fresh_reuse_pair(full_for_pair, S1_ROOT, S2_ROOT)
        self.assertTrue(valid, reasons)
        corrupt = dict(full_manifest)
        corrupt["dataset_file_hash"] = "changed"
        self._write_manifest(sample_full, corrupt)
        valid, reasons = orch.validate_fresh_reuse_pair(full_for_pair, S1_ROOT, S2_ROOT)
        self.assertFalse(valid)
        self.assertTrue(any("dataset_file_hash" in reason for reason in reasons))
        self.assertEqual(target_for_pair["run_dir"], target_dir)

    def test_strict_four_way_gate_requires_all_four_exact_run_keys_and_model_hashes(self):
        dataset, setting, seed = "kuairand", "natural", 2022
        keys = {
            "sample_full": sample_full_replacement_key(dataset, setting, seed),
            "sample_target_only_shared": fresh_formal_key("sample", dataset, setting, seed),
            "equal_target_only_shared": fresh_formal_key("equal", dataset, setting, seed),
        }
        run_dirs = {}
        for role, key in keys.items():
            task = self.task_by_key[key]
            manifest = dict(self.equal_full_manifest)
            access = FULL if role == "sample_full" else ACCESS
            manifest.update({"formal_run_key": key, "stage": "S3", "shared_access": access,
                             "fresh_role": task["fresh_role"],
                             "provenance_type": task["fresh_role"],
                             "method": task["objective_weighting"]})
            run_dirs[key] = self._write_manifest(task, manifest)
        result = orch.validate_strict_four_way_block(
            dataset, setting, seed, self.tasks, run_dirs=run_dirs,
            s1_root=S1_ROOT, s2_root=S2_ROOT)
        self.assertTrue(result["strict_source_identity"])
        self.assertEqual(result["forward_model_sha256"], MODEL_SHA)
        corrupt_path = os.path.join(run_dirs[keys["sample_full"]], "run_manifest.json")
        with open(corrupt_path, "r", encoding="utf-8") as stream:
            corrupted = json.load(stream)
        corrupted["code_hashes"] = dict(corrupted["code_hashes"])
        corrupted["code_hashes"][MODEL_KEY] = "wrong-model-hash"
        with open(corrupt_path, "w", encoding="utf-8") as stream:
            json.dump(corrupted, stream)
        with self.assertRaisesRegex(RuntimeError, "forward source hash mismatch"):
            orch.validate_strict_four_way_block(
                dataset, setting, seed, self.tasks, run_dirs=run_dirs,
                s1_root=S1_ROOT, s2_root=S2_ROOT)

    def test_fresh_matrix_cardinality_and_lineage_corruption_fail_closed(self):
        bad = [dict(task) for task in self.tasks]
        bad.pop()
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            orch.validate_fresh_matrix(bad)
        bad = [dict(task) for task in self.tasks]
        replacement = next(task for task in bad if task["fresh_role"] == FRESH_SAMPLE_FULL_ROLE)
        replacement["source_formal_run_key"] = "wrong-target"
        with self.assertRaisesRegex(ValueError, "exact Sample target run"):
            orch.validate_fresh_matrix(bad)


class RealArchiveRevalidationTests(unittest.TestCase):
    def test_live_wave_two_archive_is_revalidated_read_only_with_hash_bound_receipt(self):
        with open(QUEUE_PATH, "rb") as stream:
            queue_before = hashlib.sha256(stream.read()).hexdigest()
        archive_root = os.path.join(RAW_ROOT, ".stage_S3_archives",
                                    "s3_kuairand_natural_seed2022", "wave_2")
        archive_before = _tree_digest(archive_root)
        tasks = build_fresh_matrix(RAW_ROOT)
        tasks = orch.annotate_pairs(tasks)
        canonical_before = [os.path.exists(str(task["run_dir"])) for task in tasks
                            if task["dataset"] == "kuairand" and task["setting"] == "natural"
                            and int(task["seed"]) == 2022]
        receipt = revalidate_wave2_archive(
            queue_path=QUEUE_PATH, raw_root=RAW_ROOT, tasks=tasks,
            current_source_root=ROOT, process_inventory=False)
        validate_receipt(receipt)
        self.assertEqual(receipt["sample_full_pairing"], "PENDING_SAMPLE_FULL_REPLACEMENT")
        self.assertEqual(set(receipt["member_checks"]), {"equal", "sample"})
        self.assertEqual(receipt["reviewed_model_sha256"], MODEL_SHA)
        with open(QUEUE_PATH, "rb") as stream:
            self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), queue_before)
        self.assertEqual(_tree_digest(archive_root), archive_before)
        canonical_after = [os.path.exists(str(task["run_dir"])) for task in tasks
                           if task["dataset"] == "kuairand" and task["setting"] == "natural"
                           and int(task["seed"]) == 2022]
        self.assertEqual(canonical_after, canonical_before)
        self.assertFalse(receipt["live_promotion_performed"])


if __name__ == "__main__":
    unittest.main()
