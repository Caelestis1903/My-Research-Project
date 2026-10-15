from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S3_orchestrator import _child_output_failure_markers, _execute_pair_with_recovery


def _pair_fixture(root: str):
    pair_id = "pair_test_seed2022"
    tasks = [
        {"formal_run_key": "target_only_shared|equal|test", "objective_weighting": "equal",
         "run_dir": os.path.join(root, "canonical", "equal")},
        {"formal_run_key": "target_only_shared|sample|test", "objective_weighting": "sample",
         "run_dir": os.path.join(root, "canonical", "sample")},
    ]
    pair = {"co_run_pair_id": pair_id}
    args = Namespace(raw_root=root)
    queue_path = os.path.join(root, "queue.json")
    base = {"launched": 0, "attempts": [], "archives": []}
    return pair, tasks, args, queue_path, base


class PairRecoveryTests(unittest.TestCase):
    def test_child_protocol_and_oom_diagnostics_are_nonretryable(self):
        protocol = _child_output_failure_markers("RuntimeError: Stage S3 protocol mismatch")
        deviation = _child_output_failure_markers(
            "ValueError: formal Stage S3 does not allow protocol or seed deviations")
        oom = _child_output_failure_markers("RuntimeError: CUDA out of memory")
        self.assertIn("protocol_or_validator_failure", protocol)
        self.assertIn("protocol_or_validator_failure", deviation)
        self.assertIn("deterministic_or_capacity_failure", oom)

    def _run_case(self, outcomes, *, expect_failure=False):
        temp = tempfile.TemporaryDirectory(prefix="stage_s3_pair_recovery_")
        self.addCleanup(temp.cleanup)
        root = temp.name
        pair, tasks, args, queue_path, base = _pair_fixture(root)
        records = []
        calls = []
        calls_lock = threading.Lock()

        def child_runner(task, _args, attempt_dir, attempt_id, wave):
            # Each wave gets a never-before-used attempt directory; the child starts at epoch 0.
            self.assertFalse(os.path.exists(attempt_dir))
            os.makedirs(os.path.join(attempt_dir, "run"))
            with open(os.path.join(attempt_dir, "run", "epoch_start.txt"), "w",
                      encoding="utf-8") as stream:
                stream.write("0\n")
            objective = str(task["objective_weighting"])
            outcome = outcomes[(wave, objective)]
            row = {
                "formal_run_key": task["formal_run_key"],
                "returncode": outcome["returncode"],
                "validator_valid": outcome["validator_valid"],
                "validator_reasons": list(outcome.get("validator_reasons", [])),
                "attempt_id": attempt_id, "attempt_wave": wave,
                "attempt_run_dir": os.path.join(attempt_dir, "run"),
            }
            with calls_lock:
                calls.append((wave, objective, attempt_dir, row["returncode"]))
            return row

        if expect_failure:
            with self.assertRaises(RuntimeError):
                _execute_pair_with_recovery(pair, tasks, args, queue_path, base, records,
                                            child_runner=child_runner)
            launches = None
        else:
            launches = _execute_pair_with_recovery(pair, tasks, args, queue_path, base, records,
                                                   child_runner=child_runner)
        return root, pair, tasks, base, records, calls, launches

    @staticmethod
    def _success():
        return {"returncode": 0, "validator_valid": True, "validator_reasons": []}

    @staticmethod
    def _retryable():
        return {"returncode": 1, "validator_valid": False, "validator_reasons": ["child_exit:1"]}

    def test_mixed_success_and_retryable_failure_restarts_whole_pair(self):
        outcomes = {
            (1, "equal"): self._success(), (1, "sample"): self._retryable(),
            (2, "equal"): self._success(), (2, "sample"): self._success(),
        }
        root, pair, tasks, base, records, calls, launches = self._run_case(outcomes)
        self.assertEqual(launches, 4)
        self.assertEqual({(wave, objective) for wave, objective, _, _ in calls},
                         {(1, "equal"), (1, "sample"), (2, "equal"), (2, "sample")})
        self.assertEqual(base["archives"], [os.path.join(
            root, ".stage_S3_archives", pair["co_run_pair_id"], "wave_1")])
        archived_members = os.listdir(base["archives"][0])
        self.assertEqual(set(archived_members), {"equal", "sample"})
        self.assertEqual(len(records), 2)
        self.assertEqual({row["attempt_wave"] for row in records}, {2})
        self.assertTrue(all(os.path.isfile(os.path.join(task["run_dir"], "epoch_start.txt"))
                            for task in tasks))
        self.assertEqual(base["launched"], 4)

    def test_two_retryable_failures_retry_both_and_promote_wave_two(self):
        outcomes = {
            (1, "equal"): self._retryable(), (1, "sample"): self._retryable(),
            (2, "equal"): self._success(), (2, "sample"): self._success(),
        }
        _, _, tasks, base, records, calls, launches = self._run_case(outcomes)
        self.assertEqual(launches, 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(os.path.isdir(task["run_dir"]) for task in tasks))
        self.assertEqual(len(base["archives"]), 1)

    def test_fatal_oom_protocol_and_validator_failures_stop_after_wave_one(self):
        fatal_outcomes = (
            {"returncode": 1, "validator_valid": False,
             "validator_reasons": ["child_exit:1", "deterministic_or_capacity_failure"]},
            {"returncode": 1, "validator_valid": False,
             "validator_reasons": ["formal_protocol_invalid"]},
            {"returncode": 1, "validator_valid": False,
             "validator_reasons": ["child_exit:1", "protocol_or_validator_failure"]},
            {"returncode": 0, "validator_valid": False,
             "validator_reasons": ["checkpoint_artifact_missing"]},
        )
        for fatal in fatal_outcomes:
            with self.subTest(fatal=fatal["validator_reasons"]):
                outcomes = {(1, "equal"): self._success(), (1, "sample"): fatal}
                root, pair, _, base, _, calls, _ = self._run_case(outcomes, expect_failure=True)
                self.assertEqual(len(calls), 2)
                self.assertEqual(base["launched"], 2)
                self.assertEqual(len(base["archives"]), 1)
                self.assertTrue(os.path.isdir(os.path.join(
                    root, ".stage_S3_archives", pair["co_run_pair_id"], "wave_1", "equal")))
                self.assertTrue(os.path.isdir(os.path.join(
                    root, ".stage_S3_archives", pair["co_run_pair_id"], "wave_1", "sample")))

    def test_wave_two_failure_is_archived_and_stops_without_third_attempt(self):
        outcomes = {
            (1, "equal"): self._retryable(), (1, "sample"): self._success(),
            (2, "equal"): self._success(), (2, "sample"): self._retryable(),
        }
        root, pair, _, base, _, calls, _ = self._run_case(outcomes, expect_failure=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual({wave for wave, _, _, _ in calls}, {1, 2})
        self.assertEqual(base["launched"], 4)
        self.assertEqual(len(base["archives"]), 2)
        for wave in (1, 2):
            for objective in ("equal", "sample"):
                self.assertTrue(os.path.isdir(os.path.join(
                    root, ".stage_S3_archives", pair["co_run_pair_id"],
                    "wave_{}".format(wave), objective)))

    def test_complete_pair_promotes_both_members_together(self):
        outcomes = {(1, "equal"): self._success(), (1, "sample"): self._success()}
        _, _, tasks, base, records, calls, launches = self._run_case(outcomes)
        self.assertEqual(launches, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(row["attempt_wave"] == 1 for row in records))
        self.assertTrue(all(os.path.isdir(task["run_dir"]) for task in tasks))
        self.assertEqual(base["archives"], [])
        self.assertEqual(base["launched"], 2)
        queue_path = os.path.join(os.path.dirname(os.path.dirname(tasks[0]["run_dir"])),
                                  "queue.json")
        with open(queue_path, "r", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream)["resume_state"], "RUNNING")


if __name__ == "__main__":
    unittest.main()
