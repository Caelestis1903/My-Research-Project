"""Failure injection for S4 operational gates; never launches formal training."""

import json
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage_S4_orchestrator import (_finalize_child, _monitor_children, _promote_wave,
                                   _recover_promotion, _retry_classification,
                                   _shared_formula_reference_norm, _validated_overlap,
                                   validate_probe_trace, sha)
from stage_S4_run import FIELDS
from stage_S4_run import S4Runner
from stage_S3_run import S3Runner
from stage_S3_protocol import formal_protocol_for, FRESH_TARGET_ROLE
from stage_S4_protocol import matrix, waves
from stage_S4_protocol import inventory
import torch
import stage_S4_orchestrator as orchestration
import stage_S4_report as reporting


def sample_trace():
    rows, probes = [], []
    for epoch in range(50):
        for kind, step in (("main", 0), ("architecture", -1)):
            row = {field: "" for field in FIELDS}
            row.update({"kind": kind, "epoch": str(epoch), "step": str(step),
                        "formal_run_key": "toy", "arm": "F", "seed": "2022",
                        "batch_fingerprint": "fingerprint-{}-{}".format(kind, epoch),
                        "g_full_norm": "2", "g_target_norm": "1",
                        "delta_full_norm": "0.2", "delta_target_norm": "0.1",
                        "delta_N_predicted_norm": "0.15",
                        "gradient_cosine": "0.5", "r_U": "0.5"})
            rows.append(row)
            probe = dict(row, delta_cosine="0.5", target_gain_full="0.1",
                         target_gain_target="0.05", target_gain_equal_norm_full="0.05",
                         delta_N_norm="0.15")
            probes.append(probe)
    return rows, probes


class StageS4OrchestrationTests(unittest.TestCase):
    def test_disposable_complete_s4_writer_and_validator(self):
        source_run = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "stage_S3_raw_results", "target_only_shared", "equal",
                                  "kuairand", "natural", "m3oe", "seed2023", "joint")
        with tempfile.TemporaryDirectory() as root:
            work = os.path.join(root, "completed_fixture")
            shutil.copytree(source_run, work)
            runner = S4Runner.__new__(S4Runner)
            runner.run_dir = work
            runner.arm = "F"
            runner.formal_run_key = "s4|equal|kuairand|natural|m3oe|seed2023|F"
            runner.args = type("Args", (), {
                "seed": 2023, "dataset": "kuairand", "setting": "natural",
                "backbone": "m3oe", "epoch": 50, "batch_size": 40960,
                "device": "cuda:0", "learning_rate": .01,
                "learning_rate_darts": .001, "weight_decay": 1e-5,
                "co_run_pair_id": "s4_FT_seed2023",
                "planned_peer_run_key": "s4|equal|kuairand|natural|m3oe|seed2023|T",
                "worker_id": "worker-1", "attempt_id": "disposable"})()
            runner.objective_weighting = "equal"
            runner.dataset = "kuairand"
            runner.setting = "natural"
            runner.shared_access = "FULL"
            runner.fresh_role = "target_only_shared"
            runner.source_formal_run_key = ""
            runner.target_local_domain = 1
            runner.device = torch.device("cuda:0")
            runner.optimizer = type("Optimizer", (), {"param_groups": [{"betas": (.9, .999)}]})()
            runner.initial_inventory = inventory(orchestration.ROOT)
            runner.zero_target_main_steps = 0
            runner.main_steps_seen = 2450
            runner.objective_calls = [{"coefficient_table_hash": orchestration.EQUAL_TABLE_SHA256}]
            for attr, name in (("partition", "gradient_partition.json"),
                               ("reachability", "reachability_audit.json"),
                               ("objective_table", "objective_coefficients.json")):
                with open(os.path.join(work, name)) as handle:
                    setattr(runner, attr, json.load(handle))
            with open(os.path.join(work, "batch_trace.csv"), newline="") as handle:
                runner.batch_rows = list(csv.DictReader(handle))
            runner.gradient_rows = []
            runner.probe_rows = []
            for row in runner.batch_rows:
                kind = "architecture" if row["kind"] == "architecture" else "main"
                step = {field: "" for field in FIELDS}
                step.update({"stage": "S4", "formal_run_key": runner.formal_run_key,
                             "arm": "F", "seed": "2023", "epoch": row["epoch"],
                             "step": row["step"], "kind": kind,
                             "batch_fingerprint": row["batch_fingerprint"],
                             "batch_size": row["batch_size"],
                             "domain_counts": row["domain_counts"],
                             "L_full": row["L_full"], "L_target": row["L_target"],
                             "L_non": row["L_non"], "decomposition_residual": "0",
                             "g_full_norm": "2", "g_target_norm": "1", "g_non_norm": "1",
                             "gradient_cosine": "0.5", "r_N": "0.5", "r_U": "0.5",
                             "actual_gradient_norm": "2", "delta_full_norm": "0.2",
                             "delta_target_norm": "0.1", "delta_N_predicted_norm": "0.1",
                             "delta_actual_norm": "0.2", "matching_residual": "0",
                             "private_delta_norm": "0.1", "decay_norm": "0",
                             "adam_state_step_min": "1", "adam_state_step_max": "1",
                             "optimizer_steps": "1", "architecture_optimizer_steps": "1",
                             "zero_case": "none", "full_direction_cosine": "1",
                             "private_full_residual": "0", "shared_formula_residual": "0"})
                runner.gradient_rows.append(step)
                if kind == "architecture" or int(row["step"]) == 0:
                    runner.probe_rows.append(dict(step, delta_cosine="0.5",
                                                  target_gain_full="0", target_gain_target="0",
                                                  target_gain_equal_norm_full="0", delta_N_norm="0.1"))
            with patch("stage_S2_run.S2Runner.write_outputs", return_value=None):
                runner.write_outputs({}, 1.0)
            interval = os.path.join(work, "process_interval.json")
            with open(interval, "w") as handle:
                json.dump({"pid": 1, "started_ns": 1, "finished_ns": 2, "status": "OK"}, handle)
            console = os.path.join(root, "console.log")
            with open(console, "w") as handle:
                handle.write("disposable completed fixture")
            _finalize_child(work, console)
            task = matrix(root)[4]
            task["run_dir"] = work
            good, reasons = orchestration.validate_run(task)
            self.assertTrue(good, reasons)
            probe = os.path.join(work, "probe.csv")
            renamed = probe + ".missing"
            os.replace(probe, renamed)
            try:
                self.assertFalse(orchestration.validate_run(task)[0])
            finally:
                os.replace(renamed, probe)
            with open(probe, "rb") as handle:
                probe_bytes = handle.read()
            try:
                with open(probe, "ab") as handle:
                    handle.write(b"corrupt\n")
                self.assertFalse(orchestration.validate_run(task)[0])
            finally:
                with open(probe, "wb") as handle:
                    handle.write(probe_bytes)
            manifest_path = os.path.join(work, "run_manifest.json")
            with open(manifest_path, "rb") as handle:
                manifest_bytes = handle.read()
            for mutation in (lambda m: m["formal_protocol_observed"].update(seed=9999),
                             lambda m: m.pop("formal_protocol"),
                             lambda m: m["code_hashes"].update(
                                 {"experiments/stage_S4_run.py": "0" * 64}),
                             lambda m: m.update(fresh_role="target_only_shared")):
                try:
                    with open(manifest_path, "w") as handle:
                        modified = json.loads(manifest_bytes)
                        mutation(modified)
                        json.dump(modified, handle)
                    self.assertFalse(orchestration.validate_run(task)[0])
                finally:
                    with open(manifest_path, "wb") as handle:
                        handle.write(manifest_bytes)
            self.assertTrue(orchestration.validate_run(task)[0])

    def test_s4_output_serializer_writes_protocol_probe_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            runner = S4Runner.__new__(S4Runner)
            runner.run_dir = root
            runner.arm = "F"
            runner.formal_run_key = "s4|equal|kuairand|natural|m3oe|seed2022|F"
            runner.args = type("Args", (), {
                "seed": 2022, "dataset": "kuairand", "setting": "natural",
                "backbone": "m3oe", "epoch": 50, "batch_size": 40960,
                "device": "cuda:0", "learning_rate": .01,
                "learning_rate_darts": .001, "weight_decay": 1e-5})()
            runner.objective_weighting = "equal"
            runner.optimizer = type("Optimizer", (), {"param_groups": [{"betas": (.9, .999)}]})()
            runner.initial_inventory = inventory(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            runner.gradient_rows, runner.probe_rows = sample_trace()
            runner.zero_target_main_steps = 0
            runner.main_steps_seen = 50
            for name in ("batch_trace.csv", "metrics_epoch.csv", "metrics_test.csv",
                         "metrics_summary.csv"):
                with open(os.path.join(root, name), "w", newline="") as handle:
                    writer = csv.DictWriter(handle, ["stage", "value"])
                    writer.writeheader()
                    writer.writerow({"stage": "S3", "value": "fixture"})
            for name in ("config.json", "run_manifest.json"):
                with open(os.path.join(root, name), "w") as handle:
                    json.dump({"artifact_hashes": {}, "artifacts": [], "status": "OK"}, handle)
            with patch.object(S3Runner, "write_outputs", return_value=None):
                runner.write_outputs({}, 1.0)
            with open(os.path.join(root, "run_manifest.json")) as handle:
                manifest = json.load(handle)
            with open(os.path.join(root, "config.json")) as handle:
                config = json.load(handle)
            self.assertEqual(manifest["formal_protocol"], manifest["formal_protocol_observed"])
            self.assertEqual(manifest["formal_protocol_mismatches"], {})
            self.assertEqual(config["formal_protocol"], manifest["formal_protocol"])
            self.assertEqual(manifest["stage"], "S4")
            self.assertEqual(manifest["probe_count"], 100)
            with open(os.path.join(root, "probe.csv")) as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 100)
            self.assertEqual(manifest["artifact_hashes"]["probe.csv"],
                             sha(os.path.join(root, "probe.csv")))
            self.assertEqual(manifest["artifact_hashes"]["gradient_step.csv"],
                             sha(os.path.join(root, "gradient_step.csv")))

    def test_s3_writer_protocol_default_and_s4_dispatch(self):
        ordinary = S3Runner.__new__(S3Runner)
        ordinary.fresh_role = FRESH_TARGET_ROLE
        self.assertEqual(ordinary._output_formal_protocol(),
                         formal_protocol_for(FRESH_TARGET_ROLE))
        s4 = S4Runner.__new__(S4Runner)
        s4.arm = "F"
        s4.args = type("Args", (), {"seed": 2022})()
        self.assertEqual(s4._output_formal_protocol()["stage"], "S4")
        self.assertEqual(s4._output_formal_protocol()["arm"], "F")

    def test_a2_freeze_preserves_a1_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as root:
            a1_raw = os.path.join(root, "a1_raw")
            a2_raw = os.path.join(root, "a2_raw")
            a1_queue_path = os.path.join(root, "a1_queue.json")
            a2_queue_path = os.path.join(root, "a2_queue.json")
            attempt = os.path.join(a1_raw, ".attempts", "wave", "attempt1")
            os.makedirs(attempt)
            with open(os.path.join(attempt, "failure.json"), "w") as handle:
                json.dump({"error": "frozen"}, handle)
            tasks = matrix(a1_raw)
            with open(a1_queue_path, "w") as handle:
                json.dump({"stage": "S4", "tasks": tasks, "waves": waves(tasks),
                           "aborted_predecessor": {"queue_sha256": "old"}}, handle)
            with open(a1_queue_path + ".state.json", "w") as handle:
                json.dump({"completed": [], "attempts": [{"retry_class": "blocked"}]}, handle)
            patches = [patch.object(orchestration, name, value) for name, value in (
                ("A1_RAW_ROOT", a1_raw), ("A1_QUEUE_PATH", a1_queue_path),
                ("A1_QUEUE_SHA256", sha(a1_queue_path)),
                ("A1_STATE_SHA256", sha(a1_queue_path + ".state.json")),
                ("A1_OUTPUT_ROOT", os.path.join(root, "a1_outputs")),
                ("A2_RAW_ROOT", a2_raw), ("A2_QUEUE_PATH", a2_queue_path))]
            patches.append(patch.object(orchestration, "_a1_predecessor_receipt",
                                        return_value=({}, {"queue_sha256": "old"})))
            for item in patches:
                item.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "reviewer PASS"):
                    orchestration.freeze_a2_queue(allow_create=False)
                queue = orchestration.freeze_a2_queue()
                self.assertEqual(queue["amendment"], "A2")
                self.assertEqual(queue["waves"], waves(tasks))
                self.assertEqual(len(queue["tasks"]), 16)
                self.assertEqual(len(queue["source_inventory"]), 21)
                self.assertFalse(os.path.exists(a2_raw))
                self.assertEqual(orchestration.freeze_a2_queue(allow_create=False), queue)
                with self.assertRaisesRegex(RuntimeError, "isolated"):
                    reporting.collect(a1_raw, os.path.join(root, "mixed_outputs"),
                                      a2_queue_path)
                os.makedirs(tasks[0]["run_dir"])
                try:
                    with self.assertRaisesRegex(RuntimeError, "canonical"):
                        orchestration.freeze_a2_queue(allow_create=False)
                finally:
                    os.rmdir(tasks[0]["run_dir"])
                with open(os.path.join(attempt, "failure.json"), "a") as handle:
                    handle.write("drift")
                with self.assertRaisesRegex(RuntimeError, "predecessor"):
                    orchestration.freeze_a2_queue(allow_create=False)
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_probe_support_content_and_math(self):
        rows, probes = sample_trace()
        task = {"formal_run_key": "toy", "arm": "F", "seed": 2022}
        self.assertEqual(validate_probe_trace(rows, probes, task), [])
        self.assertIn("probe support incomplete", validate_probe_trace(rows, probes[:-1], task))
        duplicated = probes[:-1] + [dict(probes[0])]
        self.assertIn("duplicate probe key", validate_probe_trace(rows, duplicated, task))
        corrupt = [dict(row) for row in probes]
        corrupt[0]["batch_fingerprint"] = "other"
        self.assertIn("probe/step content mismatch", validate_probe_trace(rows, corrupt, task))
        corrupt = [dict(row) for row in probes]
        corrupt[0]["target_gain_equal_norm_full"] = "0.2"
        self.assertIn("probe equal-norm identity mismatch", validate_probe_trace(rows, corrupt, task))
        corrupt = [dict(row) for row in probes]
        corrupt[0]["target_gain_full"] = "nan"
        self.assertIn("nonfinite probe quantity", validate_probe_trace(rows, corrupt, task))

    def test_n_formula_reference_is_predicted_delta_not_actual(self):
        row = {"delta_N_predicted_norm": "0.01", "delta_actual_norm": "0.02",
               "delta_target_norm": "0.03", "delta_full_norm": "0.04"}
        self.assertEqual(_shared_formula_reference_norm(row, "N"), .01)
        self.assertEqual(_shared_formula_reference_norm(row, "F"), .04)
        self.assertEqual(_shared_formula_reference_norm(row, "T"), .03)
        self.assertEqual(_shared_formula_reference_norm(row, "U"), .03)
        residual = 1.5e-6
        self.assertGreater(residual, max(1e-6, 1e-4 * _shared_formula_reference_norm(row, "N")))
        self.assertLess(residual, max(1e-6, 1e-4 * float(row["delta_actual_norm"])))

    def test_failed_child_terminates_live_peer(self):
        failed = subprocess.Popen([sys.executable, "-c", "import sys;sys.exit(7)"])
        peer = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        started = time.monotonic()
        finished, terminated, _ = _monitor_children(
            [(failed, None, "", {}), (peer, None, "", {})])
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(finished[failed.pid]["returncode"], 7)
        self.assertIn(peer.pid, terminated)
        self.assertIsNotNone(peer.poll())

    def test_overlap_uses_child_intervals_not_popen_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            children = []
            for i, interval in enumerate(((10, 30), (20, 40))):
                work = os.path.join(root, str(i))
                os.makedirs(work)
                with open(os.path.join(work, "process_interval.json"), "w") as handle:
                    json.dump({"pid": 100 + i, "status": "OK",
                               "started_ns": interval[0], "finished_ns": interval[1]}, handle)
                children.append((type("Process", (), {"pid": 100+i})(), None, work, {}))
            _, overlap = _validated_overlap(children)
            self.assertEqual(overlap, 10)
            with open(os.path.join(root, "1", "process_interval.json"), "w") as handle:
                json.dump({"pid": 101, "status": "OK", "started_ns": 30,
                           "finished_ns": 40}, handle)
            with self.assertRaisesRegex(RuntimeError, "did not actually overlap"):
                _validated_overlap(children)

    def test_stdout_is_retained_and_hashed(self):
        with tempfile.TemporaryDirectory() as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            with open(os.path.join(work, "stdout.log"), "wb") as handle:
                handle.write(b"runner summary")
            with open(os.path.join(work, "process_interval.json"), "wb") as handle:
                handle.write(b"interval")
            with open(os.path.join(work, "run_manifest.json"), "w") as handle:
                json.dump({"artifact_hashes": {}, "artifacts": []}, handle)
            console = os.path.join(root, "console.log")
            with open(console, "wb") as handle:
                handle.write(b"process output")
            _finalize_child(work, console)
            with open(os.path.join(work, "stdout.log"), "rb") as handle:
                self.assertIn(b"process output", handle.read())
            with open(os.path.join(work, "run_manifest.json")) as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["artifact_hashes"]["stdout.log"],
                             sha(os.path.join(work, "stdout.log")))

    def test_second_promotion_failure_rolls_back_first(self):
        with tempfile.TemporaryDirectory() as root:
            wave = {"wave_id": "toy-wave", "formal_run_keys": ["F", "T"]}
            tasks = {}
            children = []
            for arm in ("F", "T"):
                work = os.path.join(root, "attempt", arm)
                os.makedirs(work)
                with open(os.path.join(work, "evidence.txt"), "w") as handle:
                    handle.write(arm)
                task = {"arm": arm, "run_dir": os.path.join(root, "canonical", arm)}
                tasks[arm] = task
                children.append((None, None, work, task))
            state, state_path = {"completed": [], "attempts": []}, os.path.join(root, "state.json")
            with patch("stage_S4_orchestrator.validate_run", return_value=(True, [])):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    _promote_wave(root, wave, children, state, state_path, {}, fail_after_first=True)
                self.assertFalse(any(os.path.exists(task["run_dir"]) for task in tasks.values()))
                self.assertTrue(all(os.path.exists(work) for _, _, work, _ in children))
                _promote_wave(root, wave, children, state, state_path, {})
            self.assertEqual(state["completed"], ["toy-wave"])
            self.assertTrue(all(os.path.isfile(os.path.join(task["run_dir"], "evidence.txt"))
                                for task in tasks.values()))

    def test_recover_partial_crash_rolls_back_canonical(self):
        with tempfile.TemporaryDirectory() as root:
            wave = {"wave_id": "toy-wave", "formal_run_keys": ["F", "T"]}
            tasks = {arm: {"arm": arm, "run_dir": os.path.join(root, "canonical", arm)}
                     for arm in ("F", "T")}
            os.makedirs(tasks["F"]["run_dir"])
            os.makedirs(os.path.join(root, ".promotions"))
            journal_path = os.path.join(root, ".promotions", "toy-wave.json")
            with open(journal_path, "w") as handle:
                json.dump({"status": "PREPARED", "wave": "toy-wave"}, handle)
            state = {"completed": []}
            _recover_promotion(root, wave, tasks, state, os.path.join(root, "state.json"), {})
            self.assertFalse(os.path.exists(tasks["F"]["run_dir"]))
            with open(journal_path) as handle:
                self.assertEqual(json.load(handle)["status"], "ROLLED_BACK")

    def test_only_explicit_transient_errno_allows_retry(self):
        with tempfile.TemporaryDirectory() as root:
            works = [os.path.join(root, arm) for arm in ("F", "T")]
            for work in works:
                os.makedirs(work)
            children = [(None, None, work, {}) for work in works]
            report = {"exits": [1, -15]}
            self.assertEqual(_retry_classification(children, report)[0], "blocked")
            with open(os.path.join(works[0], "failure.json"), "w") as handle:
                json.dump({"exception_type": "AssertionError", "exception_errno": None}, handle)
            self.assertEqual(_retry_classification(children, report)[0], "blocked")
            with open(os.path.join(works[0], "failure.json"), "w") as handle:
                json.dump({"exception_type": "OSError", "exception_errno": 11}, handle)
            self.assertEqual(_retry_classification(children, report)[0], "transient_os_error")
            self.assertEqual(_retry_classification(children, {"exits": [0, 0]})[0], "none")

    def test_a1_freeze_isolated_and_bound_to_old_abort_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            old_root = os.path.join(root, "old_raw")
            a1_root = os.path.join(root, "a1_raw")
            old_queue_path = os.path.join(root, "old_queue.json")
            a1_queue_path = os.path.join(root, "a1_queue.json")
            attempts = os.path.join(old_root, ".attempts", "s4_FT_seed2022", "attempt1")
            os.makedirs(attempts)
            with open(os.path.join(attempts, "failure.json"), "w") as handle:
                json.dump({"exception_type": "AssertionError"}, handle)
            old_tasks = matrix(old_root)
            with open(old_queue_path, "w") as handle:
                json.dump({"stage": "S4", "tasks": old_tasks, "waves": waves(old_tasks)}, handle)
            state_path = old_queue_path + ".state.json"
            with open(state_path, "w") as handle:
                json.dump({"completed": [], "attempts": [{"retry_class": "blocked"}]}, handle)
            abort_report = os.path.join(root, "old_abort_report.md")
            with open(abort_report, "w") as handle:
                handle.write("immutable predecessor abort report\n")
            patches = [patch.object(orchestration, name, value) for name, value in (
                ("OLD_QUEUE_PATH", old_queue_path), ("OLD_QUEUE_SHA256", sha(old_queue_path)),
                ("OLD_STATE_SHA256", sha(state_path)), ("OLD_RAW_ROOT", old_root),
                ("OLD_ABORT_REPORT_PATH", abort_report),
                ("OLD_ABORT_REPORT_SHA256", sha(abort_report)),
                ("A1_RAW_ROOT", a1_root), ("A1_QUEUE_PATH", a1_queue_path),
                ("HERE", root))]
            for item in patches:
                item.start()
            try:
                queue = orchestration.freeze_a1_queue()
                self.assertEqual(queue["amendment"], "A1")
                self.assertEqual(queue["waves"], waves(old_tasks))
                self.assertEqual(queue["aborted_predecessor"]["canonical_count"], 0)
                self.assertEqual(len(queue["aborted_predecessor"]["attempt_files"]), 1)
                self.assertTrue(os.path.isfile(a1_queue_path))
                self.assertFalse(os.path.exists(a1_root))
                self.assertEqual(orchestration.freeze_a1_queue(allow_create=False), queue)
                with open(os.path.join(attempts, "failure.json"), "a") as handle:
                    handle.write(" ")
                with self.assertRaisesRegex(RuntimeError, "predecessor"):
                    orchestration.freeze_a1_queue(allow_create=False)
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_a1_repeated_collection_preserves_predecessor_report(self):
        with tempfile.TemporaryDirectory() as root:
            old_root = os.path.join(root, "old_raw")
            a1_root = os.path.join(root, "a1_raw")
            output_root = os.path.join(root, "a1_outputs")
            old_queue_path = os.path.join(root, "old_queue.json")
            a1_queue_path = os.path.join(root, "a1_queue.json")
            attempt = os.path.join(old_root, ".attempts", "wave1", "attempt1")
            os.makedirs(attempt)
            with open(os.path.join(attempt, "failure.json"), "w") as handle:
                json.dump({"exception_type": "AssertionError"}, handle)
            old_tasks = matrix(old_root)
            with open(old_queue_path, "w") as handle:
                json.dump({"stage": "S4", "tasks": old_tasks, "waves": waves(old_tasks)}, handle)
            state_path = old_queue_path + ".state.json"
            with open(state_path, "w") as handle:
                json.dump({"completed": [], "attempts": [{"retry_class": "blocked"}]}, handle)
            abort_report = os.path.join(root, "old_abort_report.md")
            with open(abort_report, "w") as handle:
                handle.write("historical abort evidence\n")
            old_sha = sha(abort_report)
            patches = [patch.object(orchestration, name, value) for name, value in (
                ("OLD_QUEUE_PATH", old_queue_path), ("OLD_QUEUE_SHA256", sha(old_queue_path)),
                ("OLD_STATE_SHA256", sha(state_path)), ("OLD_RAW_ROOT", old_root),
                ("OLD_ABORT_REPORT_PATH", abort_report), ("OLD_ABORT_REPORT_SHA256", old_sha),
                ("A1_RAW_ROOT", a1_root), ("A1_QUEUE_PATH", a1_queue_path),
                ("HERE", root))]
            patches += [patch.object(reporting, name, value) for name, value in (
                ("A1_RAW_ROOT", a1_root), ("A1_QUEUE_PATH", a1_queue_path),
                ("A1_OUTPUT_ROOT", output_root))]
            for item in patches:
                item.start()
            try:
                queue = orchestration.freeze_a1_queue()
                checkpoints = ("selected", "final", "common_epoch_9", "common_epoch_19",
                               "common_epoch_29", "common_epoch_39", "common_epoch_49")
                def fake_csv(path):
                    name = os.path.basename(path)
                    if name == "metrics_test.csv":
                        rows = [{"checkpoint": label, "domain": str(domain),
                                 "task": str(task), "physical_domain_identity": "d" + str(domain),
                                 "auc": "0.5", "logloss": "0.7"}
                                for label in checkpoints for domain in range(3)
                                for task in range(2)]
                    elif name == "probe.csv":
                        rows = [{"epoch": str(epoch), "kind": kind}
                                for epoch in range(50) for kind in ("main", "architecture")]
                    else:
                        rows = [{"kind": "main", "epoch": "0"}]
                    return [], rows
                def fake_json(path):
                    if os.path.basename(path) == "run_manifest.json":
                        return {"selected_epoch": 9, "optimizer_steps": 1,
                                "architecture_optimizer_steps": 1, "code_inventory_hash": "fixture"}
                    raise AssertionError("unexpected JSON read: " + path)
                with patch.object(reporting, "validate_block", return_value=(True, [])), \
                     patch.object(reporting, "_read_csv", side_effect=fake_csv), \
                     patch.object(reporting, "_read_json", side_effect=fake_json):
                    for _ in range(2):
                        manifest = reporting.collect(a1_root, output_root, a1_queue_path)
                        self.assertEqual(manifest["cardinalities"]["raw_results"], 672)
                        self.assertEqual(orchestration.freeze_a1_queue(allow_create=False), queue)
                        self.assertEqual(sha(abort_report), old_sha)
                self.assertTrue(os.path.isfile(os.path.join(output_root, "reports",
                                                            "stage_S4_A1_run_report.md")))
                self.assertFalse(os.path.exists(os.path.join(root, "stage_S4_runs.csv")))
                with self.assertRaisesRegex(RuntimeError, "isolated"):
                    reporting.collect(a1_root, root, a1_queue_path)
                self.assertEqual(sha(abort_report), old_sha)
                with open(abort_report, "a") as handle:
                    handle.write("tampered\n")
                with self.assertRaisesRegex(RuntimeError, "run report SHA mismatch"):
                    orchestration.freeze_a1_queue(allow_create=False)
            finally:
                for item in reversed(patches):
                    item.stop()
