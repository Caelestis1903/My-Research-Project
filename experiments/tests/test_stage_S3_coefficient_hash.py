from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S3_orchestrator as orchestrator  # noqa: E402
from stage_S2_objectives import compute_equal_coefficients  # noqa: E402
from stage_S2_protocol import coefficient_table_hash  # noqa: E402
from stage_S3_run import _sample_objective_table  # noqa: E402


class StageS3CoefficientHashTests(unittest.TestCase):
    """Regression tests for immutable artifact and table-hash semantics."""

    def _path(self, *parts):
        return os.path.join(ROOT, *parts)

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)

    def _sample_paths(self):
        legacy = self._path(
            "experiments", "stage_S3_raw_results", "target_only_shared", "sample",
            "kuairand", "natural", "m3oe", "seed2022", "joint")
        fresh = self._path(
            "experiments", "stage_S3_raw_results", "full_replacement", "sample",
            "kuairand", "natural", "m3oe", "seed2022", "joint")
        if not all(os.path.isfile(os.path.join(path, name))
                   for path in (legacy, fresh) for name in (
                       "run_manifest.json", "objective_coefficients.json")):
            self.skipTest("promoted Stage S3 coefficient artifacts are unavailable")
        return legacy, fresh

    def test_generator_and_shared_helper_use_the_same_table_content_hash(self):
        equal = compute_equal_coefficients({"0": 8, "1": 2, "2": 1})
        sample = _sample_objective_table(equal)
        self.assertEqual(sample["domain_num"], 3)
        self.assertEqual(sample["task_num"], 2)
        self.assertEqual(sample["coefficient_table_hash"], coefficient_table_hash(sample))

    def test_legacy_schema_normalizes_without_mutating_or_pairing_file_bytes(self):
        legacy_dir, fresh_dir = self._sample_paths()
        legacy_manifest = self._load(os.path.join(legacy_dir, "run_manifest.json"))
        fresh_manifest = self._load(os.path.join(fresh_dir, "run_manifest.json"))
        legacy = orchestrator._coefficient_artifact_identity(legacy_dir, legacy_manifest)
        fresh = orchestrator._coefficient_artifact_identity(fresh_dir, fresh_manifest)

        self.assertEqual(legacy["reasons"], [])
        self.assertEqual(fresh["reasons"], [])
        self.assertEqual(legacy["schema_fields_added"], ["domain_num", "task_num"])
        self.assertEqual(legacy["normalized_table_hash"], fresh["normalized_table_hash"])
        self.assertNotEqual(legacy["raw_table_hash"], fresh["raw_table_hash"])
        self.assertNotEqual(legacy["file_sha256"], fresh["file_sha256"])
        self.assertNotEqual(orchestrator._digest(legacy["table"]), legacy["raw_table_hash"])
        self.assertEqual(orchestrator._coefficient_pair_reasons(
            legacy_manifest, fresh_manifest, legacy_dir, fresh_dir), [])

    def test_same_table_with_different_json_bytes_is_valid_but_tampered_table_fails(self):
        _legacy_dir, fresh_dir = self._sample_paths()
        manifest = self._load(os.path.join(fresh_dir, "run_manifest.json"))
        table = self._load(os.path.join(fresh_dir, "objective_coefficients.json"))
        original_file_sha = orchestrator._file_hash(
            os.path.join(fresh_dir, "objective_coefficients.json"))
        with tempfile.TemporaryDirectory(prefix="stage_s3_coefficient_hash_") as temp:
            path = os.path.join(temp, "objective_coefficients.json")
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(table, stream, ensure_ascii=False, indent=4)
                stream.write("\n")
            rewritten_manifest = dict(manifest)
            rewritten_manifest["artifact_hashes"] = {
                "objective_coefficients.json": orchestrator._file_hash(path)
            }
            rewritten = orchestrator._coefficient_artifact_identity(temp, rewritten_manifest)
            self.assertEqual(rewritten["reasons"], [])
            self.assertNotEqual(rewritten["file_sha256"], original_file_sha)
            self.assertEqual(rewritten["raw_table_hash"], table["coefficient_table_hash"])

            tampered = copy.deepcopy(table)
            tampered["loss_coefficient_w"]["0"] = 1.5
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(tampered, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
            tampered_manifest = dict(rewritten_manifest)
            tampered_manifest["artifact_hashes"] = {
                "objective_coefficients.json": orchestrator._file_hash(path)
            }
            invalid = orchestrator._coefficient_artifact_identity(temp, tampered_manifest)
            self.assertIn("coefficient_table_hash_mismatch", invalid["reasons"])
            self.assertIn("coefficient_table_loss_coefficient_mismatch:0", invalid["reasons"])

            semantic_tamper = copy.deepcopy(table)
            semantic_tamper["objective_formula"] = "non_frozen_formula"
            semantic_tamper["coefficient_table_hash"] = coefficient_table_hash(
                semantic_tamper)
            with open(path, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(semantic_tamper, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
            semantic_manifest = dict(manifest)
            semantic_manifest["artifact_hashes"] = {
                "objective_coefficients.json": orchestrator._file_hash(path)
            }
            semantic_manifest["coefficient_table_hash"] = semantic_tamper[
                "coefficient_table_hash"]
            invalid = orchestrator._coefficient_artifact_identity(temp, semantic_manifest)
            self.assertIn("coefficient_table_objective_formula_mismatch",
                          invalid["reasons"])

    def test_completed_queue_rebind_is_exact_and_uses_only_a_temporary_copy(self):
        queue_path = self._path("experiments", "stage_S3_queue_manifest_A1.json")
        queue_hash = orchestrator._file_hash(queue_path)
        self.assertEqual(queue_hash, orchestrator._COEFFICIENT_REPAIR_QUEUE_SHA256)
        raw_root = self._path("experiments", "stage_S3_raw_results")
        base = orchestrator.build_fresh_matrix(raw_root)
        tasks = orchestrator.annotate_pairs(base, orchestrator.build_launch_plan(base))
        current = orchestrator.source_inventory(orchestrator.ROOT)
        authority = {
            "review_report_sha256": "fixture-review",
            "reviewed_source_inventory_sha256":
                orchestrator.source_inventory_digest(current),
        }
        args = Namespace(
            raw_root=raw_root,
            s1_raw_root=self._path("experiments", "stage_S1_raw_results"),
            s2_raw_root=self._path("experiments", "stage_S2_raw_results"),
            allow_reviewed_coefficient_rebind=True,
        )
        with tempfile.TemporaryDirectory(prefix="stage_s3_coefficient_rebind_") as root:
            temporary_queue = os.path.join(root, "queue.json")
            shutil.copy2(queue_path, temporary_queue)
            queue = orchestrator._read_json(temporary_queue)
            orchestrator._apply_reviewed_coefficient_rebind(
                temporary_queue, queue, args, current, authority, tasks,
                process_inventory=lambda: [])
            rebound = orchestrator._read_json(temporary_queue)
            self.assertEqual(orchestrator._validate_amendment_queue_history(rebound), [])
            self.assertEqual(rebound["status"], "RUNNING")
            self.assertEqual(rebound["resume_state"], "PLANNED")
            self.assertEqual(len(rebound["records"]), 24)
            self.assertEqual(rebound["strict_block_validations"], {})
            self.assertEqual(rebound["artifact_source_inventory_digest"],
                             orchestrator._COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST)
            self.assertEqual(rebound["coefficient_repair_rebind"][
                "artifact_disposition"],
                "READ_ONLY_FINAL_VALIDATION_NO_RETRAIN_NO_PROMOTION")
            reconciliation = orchestrator._reconcile_amendment_queue(
                tasks, args, temporary_queue, rebound, process_inventory=lambda: [])
            self.assertIsNone(reconciliation["first_pending_group"])
            self.assertEqual(reconciliation["remaining_run_count"], 0)
            self.assertEqual(reconciliation["remaining_launch_group_count"], 0)
            orchestrator._finalize_strict_blocks(rebound, tasks, args)
            self.assertEqual(rebound["status"], "COMPLETE")
            self.assertEqual(rebound["resume_state"], "COMPLETE")
            self.assertEqual(len(rebound["strict_block_validations"]), 8)
            self.assertTrue(all(row["pairing_valid"] is True for row in
                                rebound["strict_block_validations"].values()))
        self.assertEqual(orchestrator._file_hash(queue_path), queue_hash)


if __name__ == "__main__":
    unittest.main()
