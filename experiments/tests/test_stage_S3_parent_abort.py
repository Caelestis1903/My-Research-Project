from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S3_orchestrator as s3
from stage_S3_protocol import (
    FRESH_SAMPLE_FULL_ROLE, FRESH_TARGET_ROLE, SETTINGS, SEEDS,
    build_fresh_matrix, build_launch_plan,
    sample_target_pending_key,
)


class AmendmentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage_s3_a1_recovery_")
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.raw_root = os.path.join(self.root, "raw")
        os.makedirs(self.raw_root)
        self.tasks = build_fresh_matrix(self.raw_root)
        self.launch_plan = build_launch_plan(self.tasks)
        self.tasks = s3.annotate_pairs(self.tasks, self.launch_plan)
        self.task_by_key = {str(task["formal_run_key"]): task for task in self.tasks}
        self.args = Namespace(
            raw_root=self.raw_root, queue_manifest=os.path.join(self.root, "a1_queue.json"),
            s1_raw_root=os.path.join(self.root, "s1"),
            s2_raw_root=os.path.join(self.root, "s2"), max_workers=2,
        )

    def test_pre_amendment_recovery_entrypoints_are_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "pre-amendment parent-abort recovery is disabled"):
            s3.resume_parent_abort(self.tasks, self.args, reviewer_report="unused")
        with self.assertRaisesRegex(RuntimeError, "pre-amendment validator recovery is disabled"):
            s3.recover_validator_bug(self.tasks, self.args, reviewer_report="unused",
                                     recovery_review_report="unused")

    def test_reviewer_gate_is_complete_hash_bound_and_requires_zero_blockers(self):
        review_path = os.path.join(self.root, "reports", "stage_S3_amendment_review.md")
        os.makedirs(os.path.dirname(review_path))
        args = Namespace(review_report=review_path)
        with open(review_path, "w", encoding="utf-8") as stream:
            stream.write("Verdict: PASS\n")
        with self.assertRaisesRegex(PermissionError, "incomplete or ambiguous"):
            s3._require_queue_execution_authority(args)
        with open(review_path, "w", encoding="utf-8") as stream:
            stream.write("Verdict: PASS WITH FIXES\nBlocking issues: 1\n")
        with self.assertRaisesRegex(PermissionError, "unresolved blocking issues"):
            s3._require_queue_execution_authority(args)
        with open(review_path, "w", encoding="utf-8") as stream:
            stream.write("Verdict: PASS\nBlocking issues: 0\n")
        authority = s3._require_queue_execution_authority(args)
        self.assertEqual(authority["review_report_sha256"], s3._file_hash(review_path))

    def test_interrupted_group_archive_is_pair_scoped_append_only_and_preserves_locks(self):
        group = self.launch_plan[0]
        group_tasks = [self.task_by_key[str(key)] for key in group["formal_run_keys"]]
        queue_path = self.args.queue_manifest
        queue = {"attempts": [], "archives": [], "interrupted_group_waves": []}
        state_digest = s3._amendment_queue_state_digest(queue)
        queue["amendment_initial_state_digest"] = state_digest
        queue["amendment_state_digest"] = state_digest
        attempt_dirs = []
        lock_paths = []
        for task in group_tasks:
            member = s3._group_member_dir(self.args, group, task, 1)
            run_dir = os.path.join(member, "run")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "partial.txt"), "w", encoding="utf-8") as stream:
                stream.write("parent-aborted\n")
            with open(os.path.join(member, "launcher_stdout.log"), "w", encoding="utf-8") as stream:
                stream.write("incomplete\n")
            attempt_dirs.append(member)
            lock = s3.lock_path_for_task(task, self.raw_root)
            os.makedirs(os.path.dirname(lock), exist_ok=True)
            with open(lock, "w", encoding="utf-8") as stream:
                stream.write(str(task["formal_run_key"]) + "\n")
            lock_paths.append(lock)
        archive = s3._archive_interrupted_group_wave(
            group, group_tasks, self.args, queue, 1, process_inventory=lambda: [])
        self.assertEqual(set(os.listdir(archive)), {"worker-1", "worker-2",
                                                    "locks_original", "locks_invalidated"})
        self.assertEqual(len(queue["attempts"]), 2)
        self.assertTrue(all(row["status"] == "ABORTED_PARENT"
                            and row["scientific_result"] is False for row in queue["attempts"]))
        self.assertTrue(all(not os.path.exists(path) for path in attempt_dirs + lock_paths))
        archive_record = queue["interrupted_group_waves"][0]
        self.assertEqual(archive_record["archive_tree_digest"], s3._tree_digest(archive))
        self.assertEqual(len(archive_record["invalidated_locks"]), 2)
        s3._write_amendment_queue_state(queue_path, queue, "fixture_archive_recorded")
        s3._validate_amendment_queue_history(s3._read_json(queue_path))

    def _make_completed_queue_fixture(self):
        records = []
        states = {}
        for group in self.launch_plan:
            group_id = str(group["launch_group_id"])
            group_tasks = [self.task_by_key[str(key)] for key in group["formal_run_keys"]]
            if group["group_kind"] == "SAMPLE_FULL_RESOURCE_PAIR":
                states[group_id] = "COMPLETE"
            elif group_id == str(self.launch_plan[0]["launch_group_id"]):
                states[group_id] = "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED"
            else:
                states[group_id] = "COMPLETE_STRICT_SOURCE_PAIRED"
            for task in group_tasks:
                key = str(task["formal_run_key"])
                run_dir = str(task["run_dir"])
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
                    json.dump({"formal_run_key": key}, stream)
                records.append({
                    "formal_run_key": key, "status": "VALID_RETAINED", "artifact_valid": True,
                    "scientific_pair_state": "STRICT_SOURCE_PAIRED", "run_dir": run_dir,
                    "source_archive_tree_digest": s3._tree_digest(run_dir),
                    "canonical_tree_digest": s3._tree_digest(run_dir),
                    "resolved_by_sample_full_key": (
                        str(task.get("source_formal_run_key", ""))
                        if task.get("fresh_role") == FRESH_TARGET_ROLE
                        and task.get("objective_weighting") == "sample" else ""),
                })
        strict = {}
        for dataset, setting in SETTINGS:
            for seed in SEEDS:
                row = {"dataset": dataset, "setting": setting, "seed": int(seed),
                       "strict_source_identity": True, "pairing_valid": True}
                strict["{}|{}|{}".format(dataset, setting, seed)] = row
        return {"launch_plan": self.launch_plan, "records": records,
                "launch_group_states": states, "pending_sample_full_replacements": [],
                "strict_block_validations": strict}

    def test_completed_amendment_resume_revalidates_and_is_idempotent(self):
        queue = self._make_completed_queue_fixture()
        before_records = json.dumps(queue["records"], sort_keys=True)
        before_trees = {str(task["formal_run_key"]): s3._tree_digest(str(task["run_dir"]))
                        for task in self.tasks}
        block_rows = {key: dict(value) for key, value in queue[
            "strict_block_validations"].items()}

        def four_way(dataset, setting, seed, *_args, **_kwargs):
            return block_rows["{}|{}|{}".format(dataset, setting, seed)]

        with mock.patch.object(s3, "validate_fresh_run", return_value=(True, [])), \
                mock.patch.object(s3, "validate_fresh_reuse_pair", return_value=(True, [])), \
                mock.patch.object(s3, "validate_strict_four_way_block", side_effect=four_way):
            result = s3._reconcile_amendment_queue(
                self.tasks, self.args, self.args.queue_manifest, queue, process_inventory=lambda: [])
        self.assertIsNone(result["first_pending_group"])
        self.assertEqual(result["completed_group_count"], 12)
        self.assertEqual(json.dumps(queue["records"], sort_keys=True), before_records)
        self.assertEqual({str(task["formal_run_key"]): s3._tree_digest(str(task["run_dir"]))
                          for task in self.tasks}, before_trees)

    def test_sample_target_pending_state_resolves_only_after_full_replacement(self):
        target = next(task for task in self.tasks if task.get("fresh_role") == FRESH_TARGET_ROLE
                      and task.get("objective_weighting") == "sample"
                      and task.get("dataset") == "kuairand" and task.get("seed") == 2022)
        replacement = next(task for task in self.tasks
                           if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE
                           and task.get("dataset") == "kuairand" and task.get("seed") == 2022)
        group_id = str(target["launch_group_id"])
        queue = {
            "records": [{"formal_run_key": target["formal_run_key"],
                         "launch_group_id": group_id,
                         "scientific_pair_state": "PENDING_SAMPLE_FULL_REPLACEMENT"}],
            "pending_sample_full_replacements": [target["formal_run_key"]],
            "launch_group_states": {group_id: "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT"},
        }
        s3._resolve_sample_pending_records(queue, [replacement])
        row = queue["records"][0]
        self.assertEqual(queue["pending_sample_full_replacements"], [])
        self.assertEqual(row["scientific_pair_state"], "STRICT_SOURCE_PAIRED")
        self.assertEqual(row["resolved_by_sample_full_key"], replacement["formal_run_key"])
        self.assertEqual(queue["launch_group_states"][group_id],
                         "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED")

    def test_template_pending_keys_flow_through_recovered_sample_replacement(self):
        binding = {
            "source_inventory": {"fixture.py": "hash"},
            "source_inventory_digest": "source-digest",
            "legacy_queue_sha256": "legacy-queue-hash",
            "amendment_reuse_audit_json_sha256": "audit-hash",
        }
        receipt = {
            "parent_abort_archive": "fixture-parent-abort",
            "wave2_archive_path": "fixture-wave2",
            "wave2_archive_tree_digest": "fixture-wave2-digest",
            "receipt_digest": "fixture-receipt",
        }
        preflight = {
            "binding": binding, "reuse_audit": {"audit_digest": "reuse-digest"},
            "wave2_revalidation_receipt": receipt,
        }
        queue = s3._amendment_queue_template(
            self.tasks, self.args, {}, {}, preflight, {})
        canonical = sample_target_pending_key("kuairand", "natural", 2022)
        replacement_key = next(
            task["formal_run_key"] for task in self.tasks
            if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE
            and task["dataset"] == "kuairand" and task["seed"] == 2022)
        self.assertIn(canonical, queue["pending_sample_full_replacements"])
        self.assertNotIn(replacement_key, queue["pending_sample_full_replacements"])
        target = next(task for task in self.tasks
                      if task["formal_run_key"] == canonical)
        queue["records"] = [{
            "formal_run_key": canonical, "launch_group_id": target["launch_group_id"],
            "status": "VALID_RETAINED", "artifact_valid": True,
            "scientific_pair_state": "PENDING_SAMPLE_FULL_REPLACEMENT",
            "run_dir": target["run_dir"],
        }]
        queue["launch_group_states"] = {
            target["launch_group_id"]: "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT"
        }
        full_task = next(task for task in self.tasks
                         if task["formal_run_key"] == replacement_key)
        s3._resolve_sample_pending_records(queue, [full_task])
        self.assertNotIn(canonical, queue["pending_sample_full_replacements"])
        self.assertNotIn(replacement_key, queue["pending_sample_full_replacements"])
        self.assertEqual(queue["records"][0]["scientific_pair_state"], "STRICT_SOURCE_PAIRED")
        self.assertEqual(queue["records"][0]["resolved_by_sample_full_key"], replacement_key)


if __name__ == "__main__":
    unittest.main()
