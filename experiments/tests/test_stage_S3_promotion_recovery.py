from __future__ import annotations

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
    CHECKPOINTS, EXPECTED_FRESH_RUNS, EXPECTED_LAUNCH_GROUPS, SEEDS,
    FRESH_TARGET_ROLE, build_fresh_matrix, build_launch_plan,
)


class PairPromotionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage_s3_pair_txn_")
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.raw_root = os.path.join(self.root, "raw")
        os.makedirs(self.raw_root)
        raw_tasks = build_fresh_matrix(self.raw_root)
        self.launch_plan = build_launch_plan(raw_tasks)
        self.tasks = s3.annotate_pairs(raw_tasks, self.launch_plan)
        self.task_by_key = {str(task["formal_run_key"]): task for task in self.tasks}
        self.group = self.launch_plan[0]
        self.group_tasks = [self.task_by_key[str(key)] for key in self.group["formal_run_keys"]]
        self.queue_path = os.path.join(self.root, "a1_queue.json")
        self.args = Namespace(
            raw_root=self.raw_root, queue_manifest=self.queue_path, max_workers=2,
            s1_raw_root=os.path.join(self.root, "s1"),
            s2_raw_root=os.path.join(self.root, "s2"),
        )
        self.queue = {
            "stage": "S3", "amendment": "A1", "queue_schema_version": 3,
            "status": "RUNNING", "resume_state": "PLANNED",
            "raw_root": self.raw_root, "tasks": self.tasks, "launch_plan": self.launch_plan,
            "expected_run_count": EXPECTED_FRESH_RUNS,
            "expected_launch_group_count": EXPECTED_LAUNCH_GROUPS,
            "expected_seed_list": list(SEEDS), "checkpoint_labels": list(CHECKPOINTS),
            "records": [], "attempts": [], "archives": [],
            "launch_group_states": {}, "strict_block_validations": {},
            "pending_sample_full_replacements": s3._sample_target_pending_keys(self.tasks),
            "retained_artifact_count": 0, "formal_launches": 0, "launched": 0,
        }
        s3._refresh_amendment_runtime_state(self.queue)
        initial = s3._amendment_queue_state_digest(self.queue)
        self.queue["amendment_initial_state_digest"] = initial
        self.queue["amendment_state_digest"] = initial
        self.queue["amendment_history"] = []
        self.queue["amendment_history_digest"] = s3._digest([])
        s3._write_amendment_queue_state(self.queue_path, self.queue, "temporary_fixture_created")
        self.queue = s3._read_json(self.queue_path)

    def _success_attempts(self):
        group_id = str(self.group["launch_group_id"])
        wave = 1
        attempt_id = group_id + "-wave1"
        snapshot = s3._amendment_runtime_state(
            self.tasks, self.queue["records"], self.queue["launch_group_states"],
            self.launch_plan)
        rows = []
        for task in self.group_tasks:
            attempt_dir = s3._group_member_dir(self.args, self.group, task, wave)
            run_dir = os.path.join(attempt_dir, "run")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
                json.dump({"formal_run_key": task["formal_run_key"]}, stream)
            rows.append({
                "formal_run_key": task["formal_run_key"],
                "launch_group_id": group_id, "co_run_pair_id": group_id,
                "planned_peer_run_key": task["planned_peer_run_key"],
                "worker_id": task["worker_id"], "attempt_id": attempt_id,
                "attempt_wave": wave, "attempt_run_dir": run_dir,
                "returncode": 0, "validator_valid": True,
                "validator_reasons": [], "queue_runtime_snapshot": snapshot,
                "status": "CHILD_OK",
            })
        self.queue["attempts"].extend(rows)
        self.queue["launched"] += 2
        self.queue["formal_launches"] = self.queue["launched"]
        s3._write_amendment_queue_state(
            self.queue_path, self.queue, "temporary_successful_children_recorded")
        self.queue = s3._read_json(self.queue_path)
        return snapshot, rows

    def test_crash_before_promotion_reconciles_recorded_valid_attempts_without_relaunch(self):
        snapshot, rows = self._success_attempts()
        with mock.patch.object(s3, "_run_group_validation",
                               return_value={"pairing_valid": True}) as validate:
            s3._reconcile_amendment_queue(
                self.tasks, self.args, self.queue_path, self.queue,
                process_inventory=lambda: [])
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(all(os.path.isdir(str(task["run_dir"])) for task in self.group_tasks))
        self.assertTrue(all(not os.path.isdir(row["attempt_run_dir"]) for row in rows))
        self.assertEqual(len(self.queue["records"]), 2)
        self.assertEqual(self.queue["launch_group_states"][self.group["launch_group_id"]],
                         "COMPLETE_PENDING_SAMPLE_FULL_REPLACEMENT")
        self.assertEqual(self.queue["remaining_run_count"], EXPECTED_FRESH_RUNS - 2)
        self.assertEqual(self.queue["remaining_launch_group_count"], EXPECTED_LAUNCH_GROUPS - 1)
        self.assertEqual(self.queue["checkpoint_labels"], list(CHECKPOINTS))
        self.assertEqual(self.queue["records"][0]["launch_runtime_snapshot"], snapshot)

    def test_parent_crash_after_two_member_promotion_before_queue_acceptance_is_idempotent(self):
        child_launches = []

        def child(task, _args, attempt_dir, attempt_id, wave):
            child_launches.append(str(task["formal_run_key"]))
            run_dir = os.path.join(attempt_dir, "run")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
                json.dump({"formal_run_key": task["formal_run_key"]}, stream)
            return {
                "formal_run_key": task["formal_run_key"], "attempt_id": attempt_id,
                "attempt_wave": wave, "attempt_run_dir": run_dir,
                "returncode": 0, "validator_valid": True, "validator_reasons": [],
            }

        with mock.patch.object(s3, "_require_queue_execution_authority", return_value={}), \
                mock.patch.object(s3, "_run_group_validation",
                                  return_value={"pairing_valid": True}), \
                mock.patch.object(s3, "_record_group_completion",
                                  side_effect=RuntimeError("simulated post-promotion crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated post-promotion crash"):
                s3._execute_amendment_group(
                    self.group, self.group_tasks, self.tasks, self.args, self.queue_path,
                    self.queue, child_runner=child, process_inventory=lambda: [])
        self.assertEqual(len(child_launches), 2)
        self.assertTrue(all(os.path.isfile(os.path.join(str(task["run_dir"]),
                                                       "run_manifest.json"))
                            for task in self.group_tasks))
        self.assertFalse(self.queue["launch_group_states"])
        receipt_path = s3._promotion_receipt_path(
            self.raw_root, self.group["launch_group_id"], 1)
        self.assertEqual(s3._read_json(receipt_path)["state"], "PROMOTED")

        with mock.patch.object(s3, "_run_group_validation",
                               return_value={"pairing_valid": True}), \
                mock.patch.object(s3, "validate_fresh_run", return_value=(True, [])):
            s3._reconcile_amendment_queue(
                self.tasks, self.args, self.queue_path, self.queue,
                process_inventory=lambda: [])
            before_records = json.dumps(self.queue["records"], sort_keys=True)
            before_history = len(self.queue["amendment_history"])
            s3._reconcile_amendment_queue(
                self.tasks, self.args, self.queue_path, self.queue,
                process_inventory=lambda: [])
        self.assertEqual(len(child_launches), 2)
        self.assertEqual(len(self.queue["records"]), 2)
        self.assertEqual(json.dumps(self.queue["records"], sort_keys=True), before_records)
        self.assertEqual(len(self.queue["amendment_history"]), before_history)
        s3._validate_amendment_queue_history(self.queue)

    def test_partial_move_between_pair_members_fails_closed_on_resume(self):
        _snapshot, _rows = self._success_attempts()
        real_promote = s3._atomic_pair_promote

        def simulated_interruption(source_dirs, canonical_dirs):
            os.makedirs(os.path.dirname(canonical_dirs[0]), exist_ok=True)
            os.replace(source_dirs[0], canonical_dirs[0])
            raise RuntimeError("simulated termination between pair-member moves")

        with mock.patch.object(s3, "_run_group_validation",
                               return_value={"pairing_valid": True}), \
                mock.patch.object(s3, "_atomic_pair_promote", side_effect=simulated_interruption):
            with self.assertRaisesRegex(RuntimeError, "cannot reconcile"):
                s3._reconcile_amendment_queue(
                    self.tasks, self.args, self.queue_path, self.queue,
                    process_inventory=lambda: [])
        self.assertTrue(os.path.isdir(str(self.group_tasks[0]["run_dir"])))
        self.assertFalse(os.path.exists(str(self.group_tasks[1]["run_dir"])))
        with mock.patch.object(s3, "_atomic_pair_promote", wraps=real_promote) as promote:
            with self.assertRaisesRegex(RuntimeError, "partial canonical promotion"):
                s3._reconcile_amendment_queue(
                    self.tasks, self.args, self.queue_path, self.queue,
                    process_inventory=lambda: [])
        promote.assert_not_called()
        self.assertFalse(self.queue["records"])

    def test_tampered_complete_canonical_pair_is_not_accepted_or_overwritten(self):
        self._success_attempts()
        with mock.patch.object(s3, "_run_group_validation",
                               return_value={"pairing_valid": True}):
            s3._promote_or_reconcile_group(
                self.group, self.group_tasks, self.tasks, self.args, self.queue, 1)
        changed = os.path.join(str(self.group_tasks[0]["run_dir"]), "tampered.txt")
        with open(changed, "w", encoding="utf-8") as stream:
            stream.write("not in promotion receipt")
        with mock.patch.object(s3, "_run_group_validation",
                               return_value={"pairing_valid": True}):
            with self.assertRaisesRegex(RuntimeError, "canonical pair hash differs"):
                s3._promote_or_reconcile_group(
                    self.group, self.group_tasks, self.tasks, self.args, self.queue, 1)
        self.assertTrue(os.path.isfile(changed))


if __name__ == "__main__":
    unittest.main()
