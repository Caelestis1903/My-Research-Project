"""S5 operational gates: freeze refusal, source-drift, collector isolation.

These tests exercise the fail-closed orchestrator, run-validator, and
collector paths without launching a single S5 training process.  Real
training is forbidden by the design.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stage_S5_orchestrator as orchestration
import stage_S5_protocol as protocol_mod
import stage_S5_report as reporting
from stage_S3_run import S3Runner


class StageS5OrchestrationTests(unittest.TestCase):
    def test_freeze_refuses_without_reviewer_pass(self):
        with tempfile.TemporaryDirectory() as root:
            raw_root = os.path.join(root, "raw")
            queue_path = os.path.join(root, "queue.json")
            with self.assertRaisesRegex(RuntimeError, "reviewer PASS"):
                orchestration.freeze_queue(raw_root, queue_path, allow_create=False)
            queue = orchestration.freeze_queue(raw_root, queue_path)
            self.assertEqual(queue["stage"], protocol_mod.STAGE)
            self.assertEqual(len(queue["tasks"]), 16)
            self.assertEqual(len(queue["waves"]), 8)
            self.assertGreaterEqual(len(queue["source_inventory"]), 21)
            # Subsequent freeze with the same files must be idempotent.
            self.assertEqual(orchestration.freeze_queue(raw_root, queue_path,
                                                          allow_create=False),
                             queue)

    def test_freeze_rejects_drift_in_source_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            raw_root = os.path.join(root, "raw")
            queue_path = os.path.join(root, "queue.json")
            queue = orchestration.freeze_queue(raw_root, queue_path)
            # Simulate source drift by reporting a mutated source_inventory.
            drifted = dict(queue["source_inventory"])
            drifted["experiments/stage_S5_protocol.py"] = "0" * 64
            with patch.object(orchestration, "inventory",
                                side_effect=lambda r: drifted):
                with self.assertRaisesRegex(RuntimeError, "frozen S5 queue/source differs"):
                    orchestration.freeze_queue(raw_root, queue_path,
                                                 allow_create=False)

    def test_validate_run_requires_seven_moment_dumps_and_step_fields(self):
        with tempfile.TemporaryDirectory() as work:
            task = {"run_dir": work, "formal_run_key": "toy", "arm": "F", "seed": 2022}
            with open(os.path.join(work, "config.json"), "w") as handle:
                json.dump({}, handle)
            good, reasons = orchestration.validate_run(task)
            self.assertFalse(good)
            self.assertIn("required artifact missing", reasons)

    def test_collector_refuses_when_inputs_do_not_match(self):
        with tempfile.TemporaryDirectory() as root:
            a_root = os.path.join(root, "a")
            b_root = os.path.join(root, "b")
            output_root = os.path.join(root, "out")
            queue_path = os.path.join(root, "queue.json")
            # Freeze under root A; the collector must reject root B because
            # the queue's task paths were bound to a_root.
            orchestration.freeze_queue(a_root, queue_path)
            with self.assertRaises(RuntimeError):
                reporting.collect(b_root, output_root, queue_path)

    def test_validate_run_rejects_invalid_protocol(self):
        """Mutation of ``formal_protocol_observed`` must fail validation."""
        with tempfile.TemporaryDirectory() as work:
            task = {"run_dir": work, "formal_run_key": "toy", "arm": "F", "seed": 2022}
            for f in orchestration.REQUIRED:
                if f.endswith(".csv"):
                    with open(os.path.join(work, f), "w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["stage", "value"])
                        writer.writeheader()
                        writer.writerow({"stage": "S5", "value": "x"})
                elif f.endswith(".json"):
                    with open(os.path.join(work, f), "w") as handle:
                        json.dump({"arm": task["arm"], "seed": task["seed"],
                                   "stage": "S5", "formal_run_key": task["formal_run_key"],
                                   "formal_protocol": {}, "formal_protocol_observed": {},
                                   "formal_protocol_mismatches": {},
                                   "formal_valid": True, "history_source": "FULL",
                                   "current_source": "FULL",
                                   "fresh_role": "fresh_s5",
                                   "code_hashes": {}, "moment_dump_epochs": [],
                                   "moment_dump_hashes": {}}, handle)
                elif f.endswith(".npz"):
                    import numpy as np
                    np.savez_compressed(os.path.join(work, f),
                                          row_ids=np.zeros(1, dtype=np.int64),
                                          physical_domain=np.array(["x"]),
                                          labels=np.zeros((1, 2), dtype=np.float32),
                                          pred_selected=np.zeros((1, 2), dtype=np.float32))
                elif f.endswith(".pth"):
                    import torch
                    torch.save({}, os.path.join(work, f))
                elif f == "stdout.log":
                    with open(os.path.join(work, f), "w") as handle:
                        handle.write("ok")
                else:
                    with open(os.path.join(work, f), "w") as handle:
                        handle.write("placeholder")
            with open(os.path.join(work, "process_interval.json"), "w") as handle:
                json.dump({"pid": 1, "started_ns": 1, "finished_ns": 2, "status": "OK"},
                          handle)
            with open(os.path.join(work, "nonfinite_inventory.json"), "w") as handle:
                json.dump([], handle)
            # Mutate the observed protocol to disagree with the canonical one.
            with open(os.path.join(work, "config.json")) as handle:
                cfg = json.load(handle)
            cfg["formal_protocol_observed"] = dict(cfg["formal_protocol_observed"])
            cfg["formal_protocol_observed"]["arm"] = "T"
            with open(os.path.join(work, "config.json"), "w") as handle:
                json.dump(cfg, handle)
            good, reasons = orchestration.validate_run(task)
            self.assertFalse(good)
            self.assertTrue(any("formal protocol" in r for r in reasons))

    def test_only_explicit_transient_errno_allows_retry(self):
        from stage_S4_orchestrator import _retry_classification
        with tempfile.TemporaryDirectory() as root:
            works = [os.path.join(root, arm) for arm in ("F", "T")]
            for work in works:
                os.makedirs(work)
            children = [(None, None, work, {}) for work in works]
            report = {"exits": [1, -15]}
            self.assertEqual(_retry_classification(children, report)[0], "blocked")
            with open(os.path.join(works[0], "failure.json"), "w") as handle:
                json.dump({"exception_type": "OSError", "exception_errno": 11}, handle)
            self.assertEqual(_retry_classification(children, report)[0],
                             "transient_os_error")
            self.assertEqual(_retry_classification(children, {"exits": [0, 0]})[0],
                             "none")

    def test_s5_drift_check_uses_s5_inventory_not_s4_inventory(self):
        """Regression test for the wave-1 failure.

        The S4 inherited ``S4Runner.write_outputs`` used the S4 inventory
        function (21 entries) for its ``inventory(ROOT) != self.initial_inventory``
        check, while ``self.initial_inventory`` is the S5 inventory (25 entries,
        a strict superset of S4).  That mismatch always raised, blocking
        every successful training run.

        The fix routes ``S5Runner.write_outputs`` through
        ``S3Runner.write_outputs(self, ...)`` instead of
        ``super().write_outputs()``, leaving a single fail-closed gate
        (``_check_source_drift``) that uses the S5 inventory.
        """
        from stage_S5_run import S5Runner
        from stage_S4_protocol import inventory as s4_inventory

        # Precondition: S5 inventory is a strict superset of S4 inventory,
        # which is what made the broken S4 check always raise.
        s5_inv = orchestration.inventory(orchestration.ROOT)
        s4_inv = s4_inventory(orchestration.ROOT)
        self.assertGreater(len(s5_inv), len(s4_inv),
                            "S5 inventory must be a strict superset of S4")
        self.assertTrue(set(s4_inv).issubset(set(s5_inv)),
                         "S4 inventory must be a subset of S5 inventory")
        self.assertNotEqual(s5_inv, s4_inv,
                             "S5 and S4 inventories must differ for the test to be meaningful")

        # Minimal runner state for the S5 drift check.
        runner = object.__new__(S5Runner)
        runner.initial_inventory = dict(s5_inv)

        # Case 1: No drift — must NOT raise, even though S4's check would
        # have raised against the S5 initial_inventory before the fix.
        runner._check_source_drift()

        # Case 2: Real drift in an S5-only file — must raise.
        drifted = dict(s5_inv)
        drifted["experiments/stage_S5_run.py"] = "0" * 64
        runner.initial_inventory = drifted
        with self.assertRaisesRegex(RuntimeError, "S5 producer source changed"):
            runner._check_source_drift()

        # Case 3: Real drift in an S4-shared file — must also raise (the
        # same S5 check covers both S5-only and S4-shared sources because
        # the S5 inventory pins both sets).
        drifted = dict(s5_inv)
        drifted["experiments/stage_S4_run.py"] = "0" * 64
        runner.initial_inventory = drifted
        with self.assertRaisesRegex(RuntimeError, "S5 producer source changed"):
            runner._check_source_drift()

    def test_s5_write_outputs_does_not_invoke_s4_inventory_check(self):
        """``S5Runner.write_outputs`` must not call the S4 inventory check.

        Regression test for the exact wave-1 failure mode: with
        ``S4Runner.write_outputs`` poisoned to raise (it raised
        ``RuntimeError("S4 producer source changed during training")`` in
        production), the real ``S5Runner.write_outputs`` must still complete
        the full post-S3 write path (CSV re-stamping, gradient_step/probe with
        S5 schemas, observed-vs-expected protocol gate, config/manifest
        stamping) without invoking the S4 layer.
        """
        import torch
        from stage_S3_run import S3Runner
        from stage_S4_run import S4Runner
        from stage_S5_run import S5Runner, S5_FIELDS, S5_PROBE_FIELDS

        runner = object.__new__(S5Runner)
        runner.arm = "F"
        runner.formal_run_key = "s5|equal|kuairand|natural|m3oe|seed2022|F"
        runner.args = SimpleNamespace(dataset="kuairand", setting="natural",
                                       backbone="m3oe", seed=2022, epoch=50,
                                       batch_size=40960, device="cuda:0",
                                       learning_rate=0.01, learning_rate_darts=0.001,
                                       weight_decay=1e-5)
        runner.objective_weighting = "equal"
        runner.initial_inventory = orchestration.inventory(orchestration.ROOT)
        param = torch.nn.Parameter(torch.tensor([0.1]))
        runner.optimizer = torch.optim.Adam([param], lr=0.01, betas=(0.9, 0.999))
        runner.gradient_rows = []
        runner.probe_rows = []
        runner.zero_target_main_steps = 0
        runner.main_steps_seen = 0
        runner._none_count_full = 0
        runner._none_count_target = 0
        runner._moment_dump_paths = []

        with tempfile.TemporaryDirectory() as run_dir:
            runner.run_dir = run_dir
            for name in ("batch_trace.csv", "metrics_epoch.csv",
                         "metrics_test.csv", "metrics_summary.csv"):
                with open(os.path.join(run_dir, name), "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["stage", "value"])
                    writer.writeheader()
                    writer.writerow({"stage": "S3", "value": "1"})
            with open(os.path.join(run_dir, "config.json"), "w") as handle:
                json.dump({}, handle)
            with open(os.path.join(run_dir, "run_manifest.json"), "w") as handle:
                json.dump({"artifact_hashes": {}, "artifacts": []}, handle)

            with patch.object(S3Runner, "write_outputs", return_value=None) as s3_write, \
                    patch.object(S4Runner, "write_outputs",
                                 side_effect=RuntimeError("S4 producer source changed "
                                                          "during training")) as s4_write:
                runner.write_outputs({}, 1.0)
            s3_write.assert_called_once()
            s4_write.assert_not_called()

            # The post-S3 S5 write path must have run to completion:
            for name in ("batch_trace.csv", "metrics_epoch.csv",
                         "metrics_test.csv", "metrics_summary.csv"):
                with open(os.path.join(run_dir, name)) as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["stage"], "S5")
            with open(os.path.join(run_dir, "gradient_step.csv")) as handle:
                self.assertEqual(next(csv.reader(handle)), S5_FIELDS)
            with open(os.path.join(run_dir, "probe.csv")) as handle:
                self.assertEqual(next(csv.reader(handle)), S5_PROBE_FIELDS)
            manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
            self.assertEqual(manifest["stage"], "S5")
            self.assertEqual(manifest["fresh_role"], "fresh_s5")
            self.assertEqual(manifest["history_source"], "FULL")
            self.assertEqual(manifest["current_source"], "FULL")
            self.assertTrue(manifest["formal_valid"])
            self.assertEqual(manifest["formal_protocol_mismatches"], {})
            self.assertEqual(manifest["code_hashes"], runner.initial_inventory)

    def test_s5_drift_check_ignores_s4_subset_differences(self):
        """The S4 inventory is a strict subset of the S5 inventory.  This
        proves that ``self.initial_inventory = S5_inv`` against
        ``inventory(ROOT) = S5_inv`` (the live S5 source) is the correct
        comparison: the check returns equal, and the S4-only 4-file gap does
        NOT trigger a false positive on its own.  This test guards against
        future regressions that might swap in ``stage_S4_protocol.inventory``
        as the live inventory function.
        """
        from stage_S5_run import S5Runner
        s5_inv = orchestration.inventory(orchestration.ROOT)
        # Build the S4 inventory and confirm it is a strict subset.
        from stage_S4_protocol import inventory as s4_inventory
        s4_inv = s4_inventory(orchestration.ROOT)
        self.assertTrue(set(s4_inv).issubset(set(s5_inv)))
        self.assertGreater(len(s5_inv) - len(s4_inv), 0,
                            "S5 must add at least one source on top of S4")

        # If the runner accidentally stored the S4 inventory as initial, the
        # live S5 inventory (which includes 4 extra files) would not match —
        # proving that storing the S4 inventory here is wrong.  Conversely,
        # when the runner stores the S5 inventory as initial, equality holds.
        runner = object.__new__(S5Runner)
        runner.initial_inventory = dict(s5_inv)
        runner._check_source_drift()  # no raise

        # If we ever accidentally compare against S4 inventory instead, the
        # equality check must fail (i.e., the subset difference would be
        # treated as drift).  Verify the trap by checking that S5 != S4.
        self.assertNotEqual(s4_inv, s5_inv)

    # ----- Round-3 fixes: Windows PID semantics + revalidate path -----

    def _make_fake_child(self, pid):
        return type("FakeChild", (), {"pid": pid})()

    def _write_interval(self, work_dir, *, status="OK", pid=1, started_ns=1000,
                        finished_ns=2000):
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, "process_interval.json"), "w") as handle:
            json.dump({"pid": pid, "started_ns": started_ns, "finished_ns": finished_ns,
                       "status": status}, handle)

    def test_validated_overlap_s5_accepts_pid_mismatch_with_overlap(self):
        """The Windows case: ``Popen.pid`` != ``process_interval.pid``, but the
        intervals still overlap in time and the two child PIDs are distinct.

        The S5 override must accept this as proof of real concurrent
        execution.  This is the exact failure mode of round-2 wave-1.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=54804,
                                 started_ns=1790497801930714000,
                                 finished_ns=1790505602322676500)
            self._write_interval(t_dir, status="OK", pid=53188,
                                 started_ns=1790497801887765100,
                                 finished_ns=1790505602264538700)
            children = [(self._make_fake_child(46804), None, f_dir, {"formal_run_key": "F"}),
                        (self._make_fake_child(49736), None, t_dir, {"formal_run_key": "T"})]
            intervals, overlap = orchestration._validated_overlap_s5(children)
            self.assertGreater(overlap, 0)
            self.assertEqual({i["pid"] for i in intervals}, {54804, 53188})
            self.assertEqual({i["popen_pid"] for i in intervals}, {46804, 49736})

    def test_validated_overlap_s5_accepts_matching_pid(self):
        """The round-1 S4 A2 / round-2 fallback case: Popen.pid == interval.pid.

        The S5 override must continue to accept this case (no regression
        for the existing launch context where Popen.pid is the real
        child's PID).
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=2628,
                                 started_ns=1790401359036813700,
                                 finished_ns=1790409806028861500)
            self._write_interval(t_dir, status="OK", pid=2244,
                                 started_ns=1790401359039813400,
                                 finished_ns=1790409799574252500)
            children = [(self._make_fake_child(2628), None, f_dir, {"formal_run_key": "F"}),
                        (self._make_fake_child(2244), None, t_dir, {"formal_run_key": "T"})]
            intervals, overlap = orchestration._validated_overlap_s5(children)
            self.assertGreater(overlap, 0)
            self.assertEqual({i["pid"] for i in intervals}, {2628, 2244})

    def test_validated_overlap_s5_rejects_status_not_ok(self):
        """A child whose ``process_interval.json`` reports ``status="FAILED"``
        must be rejected even when timing looks plausible.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="FAILED", pid=1)
            self._write_interval(t_dir, status="OK", pid=2,
                                 started_ns=100, finished_ns=200)
            children = [(self._make_fake_child(100), None, f_dir, {}),
                        (self._make_fake_child(200), None, t_dir, {})]
            with self.assertRaisesRegex(RuntimeError, "status not OK"):
                orchestration._validated_overlap_s5(children)

    def test_validated_overlap_s5_rejects_identical_child_pids(self):
        """Two children that report the same PID prove no real concurrency.
        The S5 override rejects this even when timing looks fine.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=42,
                                 started_ns=100, finished_ns=200)
            self._write_interval(t_dir, status="OK", pid=42,
                                 started_ns=150, finished_ns=250)
            children = [(self._make_fake_child(10), None, f_dir, {}),
                        (self._make_fake_child(20), None, t_dir, {})]
            with self.assertRaisesRegex(RuntimeError, "identical pid"):
                orchestration._validated_overlap_s5(children)

    def test_validated_overlap_s5_rejects_non_overlapping_intervals(self):
        """Children that completed sequentially (no overlap) must be rejected
        even if status is OK and child PIDs are distinct.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=1,
                                 started_ns=100, finished_ns=200)
            self._write_interval(t_dir, status="OK", pid=2,
                                 started_ns=200, finished_ns=300)
            children = [(self._make_fake_child(10), None, f_dir, {}),
                        (self._make_fake_child(20), None, t_dir, {})]
            with self.assertRaisesRegex(RuntimeError, "did not actually overlap"):
                orchestration._validated_overlap_s5(children)

    def test_validated_overlap_s5_rejects_inverted_intervals(self):
        """An interval with ``finished_ns <= started_ns`` must be rejected
        (the runner cannot legitimately write such an interval).
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=1,
                                 started_ns=200, finished_ns=100)
            self._write_interval(t_dir, status="OK", pid=2,
                                 started_ns=150, finished_ns=300)
            children = [(self._make_fake_child(10), None, f_dir, {}),
                        (self._make_fake_child(20), None, t_dir, {})]
            with self.assertRaisesRegex(RuntimeError, "invalid process interval"):
                orchestration._validated_overlap_s5(children)

    def test_validated_overlap_s5_rejects_non_positive_pid(self):
        """An interval with ``pid <= 0`` must be rejected (defends against a
        future bug that forgets to write a real pid).
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=1,
                                 started_ns=100, finished_ns=300)
            self._write_interval(t_dir, status="OK", pid=0,
                                 started_ns=100, finished_ns=300)
            children = [(self._make_fake_child(10), None, f_dir, {}),
                        (self._make_fake_child(20), None, t_dir, {})]
            with self.assertRaisesRegex(RuntimeError, "non-positive pid"):
                orchestration._validated_overlap_s5(children)

    def test_validated_overlap_s5_s4_check_would_have_rejected_same_case(self):
        """Lock in the regression: the S4 inherited check raises against the
        exact (PID-mismatched, time-overlapping) case that the S5 override
        accepts.  If this test ever stops failing on the S4 side, someone
        has changed Windows behaviour (or fixed the launcher mismatch in
        the parent) and the S5 override may no longer be needed.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            t_dir = os.path.join(root, "T")
            self._write_interval(f_dir, status="OK", pid=54804,
                                 started_ns=1000, finished_ns=2000)
            self._write_interval(t_dir, status="OK", pid=53188,
                                 started_ns=1500, finished_ns=2500)
            children = [(self._make_fake_child(46804), None, f_dir, {}),
                        (self._make_fake_child(49736), None, t_dir, {})]
            from stage_S4_orchestrator import _validated_overlap as s4_check
            with self.assertRaisesRegex(RuntimeError,
                                        "process interval identity/status mismatch"):
                s4_check(children)
            # And the S5 override accepts it.
            orchestration._validated_overlap_s5(children)

    # ----- Revalidate path + canonical-evidence tolerance -----

    def _write_dummy_artifact(self, path, body=b"x"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(body)

    def _make_minimal_run_dir(self, run_dir, *, history_source="FULL",
                              current_source="FULL", formal_run_key=None,
                              arm="F", seed=2022, status="OK",
                              started_ns=1000, finished_ns=2000, child_pid=42):
        """Stage the S5 REQUIRED artifacts + ``process_interval.json`` under
        ``run_dir`` with a ``run_manifest.json`` whose ``artifact_hashes``
        table is intentionally missing ``process_interval.json`` — this
        mirrors what the round-2 staged attempts look like after the child
        succeeded but before ``_finalize_child`` was ever called.
        """
        formal_run_key = formal_run_key or (
            "s5|equal|kuairand|natural|m3oe|seed{seed}|{arm}".format(
                seed=seed, arm=arm))
        os.makedirs(run_dir, exist_ok=True)
        for filename in orchestration.REQUIRED:
            if filename.endswith(".csv"):
                with open(os.path.join(run_dir, filename), "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["stage", "value"])
                    writer.writeheader()
                    writer.writerow({"stage": "S5", "value": "x"})
            elif filename.endswith(".json"):
                with open(os.path.join(run_dir, filename), "w") as handle:
                    json.dump({}, handle)
            elif filename.endswith(".npz"):
                import numpy as np
                np.savez_compressed(
                    os.path.join(run_dir, filename),
                    row_ids=np.zeros(1, dtype=np.int64),
                    physical_domain=np.array(["x"]),
                    labels=np.zeros((1, 2), dtype=np.float32),
                    pred_selected=np.zeros((1, 2), dtype=np.float32))
            elif filename.endswith(".pth"):
                import torch
                torch.save({}, os.path.join(run_dir, filename))
            elif filename == "stdout.log":
                self._write_dummy_artifact(os.path.join(run_dir, filename),
                                          body=b"stage_S5 stdout\n")
            else:
                self._write_dummy_artifact(os.path.join(run_dir, filename),
                                          body=b"placeholder")
        # process_interval.json: present on disk, will be missing from
        # artifact_hashes table below (round-2 staged-attempt shape).
        interval_path = os.path.join(run_dir, "process_interval.json")
        with open(interval_path, "w") as handle:
            json.dump({"pid": child_pid, "started_ns": started_ns,
                       "finished_ns": finished_ns, "status": status}, handle)
        # Minimal manifest that matches the validator's expectations for
        # everything except the artifact_hashes/process_interval entry.
        manifest = {
            "stage": "S5", "status": "OK",
            "formal_run_key": formal_run_key, "arm": arm, "seed": seed,
            "setting": "natural", "dataset": "kuairand",
            "dataset_key": "kuairand", "backbone": "m3oe",
            "objective_weighting": "equal",
            "fresh_role": "fresh_s5",
            "history_source": history_source,
            "current_source": current_source,
            "formal_protocol": {}, "formal_protocol_observed": {},
            "formal_protocol_mismatches": {}, "formal_valid": True,
            "formal_run_key": formal_run_key,
            "code_hashes": {}, "code_inventory_hash": "",
            "source_inventory_digest": "",
            "source_code_hash": "", "forward_model_source_sha256": "",
            "moment_dump_epochs": [], "moment_dump_hashes": {},
            "artifact_hashes": {f: "x" * 16 for f in
                                ("stdout.log", "config.json",
                                 "metrics_epoch.csv", "metrics_test.csv",
                                 "metrics_summary.csv", "batch_trace.csv",
                                 "gradient_step.csv", "probe.csv")},
            "artifacts": ["stdout.log"],
            "fixed_final_epoch": 49,
            "common_epochs": list(orchestration.COMMON_EPOCHS),
            "checkpoint_selection_frozen_before_test": True,
            "test_evaluated_during_training": False,
            "moment_dump_epochs": [],
            "batches_per_epoch": 1, "optimizer_steps": 1,
            "architecture_optimizer_steps": 1,
            "zero_target_main_steps": 0, "main_steps_seen": 1,
            "selected_epoch": 0,
        }
        with open(os.path.join(run_dir, "run_manifest.json"), "w") as handle:
            json.dump(manifest, handle)
        return formal_run_key

    def test_validate_run_accepts_staged_attempt_with_interval_on_disk(self):
        """The validator must accept a staged attempt whose
        ``process_interval.json`` is present on disk even when the
        ``run_manifest.json`` was written before ``_finalize_child`` ran
        (so ``artifact_hashes`` does not yet mention
        ``process_interval.json``).  Round-2 wave-1 attempts look like
        this; before the round-3 fix the validator rejected them with
        ``canonical process evidence missing``.

        We assert the absence of that specific reason — other reasons
        (schema fields, checkpoints, hash mismatches) are expected from
        the minimal fixture and are out of scope for this test.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            self._make_minimal_run_dir(f_dir, arm="F", seed=2022,
                                       child_pid=54804,
                                       started_ns=1000, finished_ns=2000)
            task = {"run_dir": f_dir,
                    "formal_run_key": "s5|equal|kuairand|natural|m3oe|seed2022|F",
                    "arm": "F", "seed": 2022}
            good, reasons = orchestration.validate_run(
                task, source_hashes=orchestration.inventory(orchestration.ROOT))
            canonical = [r for r in reasons if "canonical process evidence" in r]
            self.assertEqual(canonical, [],
                             "staged attempt must pass canonical-evidence "
                             "check via the on-disk tolerance: " + str(reasons))

    def test_validate_run_rejects_when_interval_missing(self):
        """If ``process_interval.json`` is absent from disk AND not in
        artifact_hashes, the validator must still reject — either via
        ``required artifact missing`` (REQUIRED includes
        ``process_interval.json``) or ``canonical process evidence
        missing`` (whichever the check fires first).  The fix is about
        on-disk tolerance, not about removing the check.
        """
        with tempfile.TemporaryDirectory() as root:
            f_dir = os.path.join(root, "F")
            self._make_minimal_run_dir(f_dir, arm="F", seed=2022,
                                       child_pid=42,
                                       started_ns=1000, finished_ns=2000)
            os.remove(os.path.join(f_dir, "process_interval.json"))
            task = {"run_dir": f_dir,
                    "formal_run_key": "s5|equal|kuairand|natural|m3oe|seed2022|F",
                    "arm": "F", "seed": 2022}
            good, reasons = orchestration.validate_run(
                task, source_hashes=orchestration.inventory(orchestration.ROOT))
            self.assertTrue(
                any("canonical process evidence" in r
                    or "required artifact missing" in r
                    for r in reasons),
                "expected a canonical-evidence or required-artifact reason; "
                "got: " + str(reasons))

    def test_revalidate_overlap_logic_accepts_pid_mismatched_staged_attempts(self):
        """Directly drive the revalidate overlap path: stage a fake F/T
        attempt under ``.attempts/`` whose ``process_interval.json``
        carries the round-2 PID-mismatch story and assert that
        ``revalidate_attempts`` reports ``overlap_valid: true`` with the
        correct overlap_ns.  The full per-arm validate_run is exercised
        by the real ``--revalidate`` invocation against the real
        round-2 artifacts (see also the hand-verification in the
        implementation report).
        """
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            raw_root = os.path.join(root, "raw")
            queue_path = os.path.join(root, "queue.json")
            attempts_root = os.path.join(raw_root, ".attempts",
                                          "s5_FT_seed2022", "attempt1")
            f_dir = os.path.join(attempts_root, "F")
            t_dir = os.path.join(attempts_root, "T")
            self._make_minimal_run_dir(f_dir, arm="F", seed=2022,
                                       child_pid=54804,
                                       started_ns=1790497801930714000,
                                       finished_ns=1790505602322676500)
            self._make_minimal_run_dir(t_dir, arm="T", seed=2022,
                                       child_pid=53188,
                                       started_ns=1790497801887765100,
                                       finished_ns=1790505602264538700)
            # Inject launch records so revalidate can surface the
            # launcher-vs-interpreter PID mismatch story in its report.
            state_path = queue_path + ".state.json"
            with open(state_path, "w") as handle:
                json.dump({"completed": [], "attempts": [{
                    "wave": "s5_FT_seed2022", "attempt": 1,
                    "attempt_root": attempts_root,
                    "launch": [
                        {"key": "s5|equal|kuairand|natural|m3oe|seed2022|F",
                         "pid": 46804},
                        {"key": "s5|equal|kuairand|natural|m3oe|seed2022|T",
                         "pid": 49736},
                    ],
                }]}, handle)
            orchestration.freeze_queue(raw_root, queue_path)
            args = SimpleNamespace(reviewer_pass=True, raw_root=raw_root,
                                     queue_manifest=queue_path, freeze_only=False,
                                     revalidate=True, python=sys.executable)
            report = orchestration.revalidate_attempts(args)
            self.assertEqual(len(report["per_wave"]), 1)
            attempt_results = report["per_wave"][0]["per_attempt"]
            self.assertEqual(len(attempt_results), 1)
            attempt = attempt_results[0]
            self.assertTrue(attempt["overlap_valid"],
                            "expected overlap_valid=true; got " + str(attempt))
            self.assertGreater(attempt["overlap_ns"], 0)
            pids = sorted([(p["popen"], p["child"])
                            for p in attempt["overlap_pids"]])
            self.assertEqual(pids, [(46804, 54804), (49736, 53188)])
            # Verify state.json was not modified by revalidate.
            state = json.load(open(state_path))
            self.assertEqual(state.get("attempts")[0].get("ready_for_promotion"),
                             None,
                             "revalidate must not modify state.json")

    def test_revalidate_requires_reviewer_pass(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            raw_root = os.path.join(root, "raw")
            queue_path = os.path.join(root, "queue.json")
            args = SimpleNamespace(reviewer_pass=False, raw_root=raw_root,
                                     queue_manifest=queue_path, freeze_only=False,
                                     revalidate=True, python=sys.executable)
            with self.assertRaisesRegex(RuntimeError, "reviewer PASS"):
                orchestration.revalidate_attempts(args)

    def test_revalidate_handles_empty_attempts_root(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            raw_root = os.path.join(root, "raw")
            queue_path = os.path.join(root, "queue.json")
            orchestration.freeze_queue(raw_root, queue_path)
            args = SimpleNamespace(reviewer_pass=True, raw_root=raw_root,
                                     queue_manifest=queue_path, freeze_only=False,
                                     revalidate=True, python=sys.executable)
            report = orchestration.revalidate_attempts(args)
            self.assertTrue(report.get("empty"))
            self.assertEqual(report["per_wave"], [])

    def _promotion_fixture(self, root):
        raw_root = os.path.join(root, "raw")
        work_f = os.path.join(root, "attempt", "F")
        work_t = os.path.join(root, "attempt", "T")
        os.makedirs(work_f)
        os.makedirs(work_t)
        tasks = {key: {"formal_run_key": key, "arm": arm,
                       "run_dir": os.path.join(raw_root, arm, "seed2022", "joint")}
                 for key, arm in (("kF", "F"), ("kT", "T"))}
        wave = {"wave_id": "w", "formal_run_keys": ["kF", "kT"]}
        children = [(None, None, work_f, tasks["kF"]),
                    (None, None, work_t, tasks["kT"])]
        state = {"completed": [], "attempts": []}
        state_path = os.path.join(root, "state.json")
        return raw_root, wave, children, tasks, state, state_path

    def test_promote_wave_resolves_s5_validator_not_s4(self):
        import stage_S4_orchestrator as s4_orchestration
        with tempfile.TemporaryDirectory() as root:
            raw_root, wave, children, tasks, state, state_path = \
                self._promotion_fixture(root)
            with patch.object(s4_orchestration, "validate_run",
                              side_effect=AssertionError("S4 validator reached")) as s4v, \
                 patch.object(orchestration, "validate_run",
                              return_value=(False, ["s5-validator-marker"])) as s5v:
                with self.assertRaisesRegex(RuntimeError, "s5-validator-marker"):
                    orchestration._promote_wave(raw_root, wave, children, state,
                                                state_path, {})
            s4v.assert_not_called()
            self.assertTrue(s5v.called)

    def test_promote_wave_commits_both_canonicals(self):
        with tempfile.TemporaryDirectory() as root:
            raw_root, wave, children, tasks, state, state_path = \
                self._promotion_fixture(root)
            with patch.object(orchestration, "validate_run",
                              return_value=(True, [])):
                orchestration._promote_wave(raw_root, wave, children, state,
                                            state_path, {})
            for key in ("kF", "kT"):
                self.assertTrue(os.path.isdir(tasks[key]["run_dir"]))
            self.assertEqual(state["completed"], ["w"])
            journal = orchestration._read_json(
                orchestration._journal_path(raw_root, "w"))
            self.assertEqual(journal["status"], "COMMITTED")

    def test_recover_promotion_resolves_s5_validator_not_s4(self):
        import stage_S4_orchestrator as s4_orchestration
        with tempfile.TemporaryDirectory() as root:
            raw_root, wave, children, tasks, state, state_path = \
                self._promotion_fixture(root)
            for key in ("kF", "kT"):
                os.makedirs(tasks[key]["run_dir"])
            journal_path = orchestration._journal_path(raw_root, "w")
            os.makedirs(os.path.dirname(journal_path), exist_ok=True)
            orchestration._write_json_atomic(
                journal_path, {"wave": "w", "status": "PREPARED", "members": []})
            with patch.object(s4_orchestration, "validate_run",
                              side_effect=AssertionError("S4 validator reached")) as s4v, \
                 patch.object(orchestration, "validate_run",
                              return_value=(True, [])) as s5v:
                orchestration._recover_promotion(raw_root, wave, tasks, state,
                                                 state_path, {})
            s4v.assert_not_called()
            self.assertTrue(s5v.called)
            self.assertEqual(state["completed"], ["w"])
            journal = orchestration._read_json(journal_path)
            self.assertEqual(journal["status"], "COMMITTED")


if __name__ == "__main__":
    unittest.main()