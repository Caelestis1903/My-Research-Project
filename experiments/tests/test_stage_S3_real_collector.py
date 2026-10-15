from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S3_protocol import (
    CHECKPOINTS, EXPECTED_FRESH_RUNS, EXPECTED_LAUNCH_GROUPS,
    EXPECTED_INTERACTION_ROWS, EXPECTED_INVENTORY_ROWS, EXPECTED_PAIRED_ROWS,
    EXPECTED_RAW_ROWS, EXPECTED_STRICT_PAIRED_ROWS, EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS,
    FRESH_SAMPLE_FULL_ROLE, FULL, build_fresh_matrix, build_launch_plan,
    build_historical_audit_matrix, build_reuse_matrix,
    SEEDS, SETTINGS, sample_full_replacement_key, sample_target_pending_key,
)
import stage_S3_orchestrator as orchestrator
from stage_S3_report import (
    RAW_FIELDS, _read_csv, _read_json, _row_for_metric, build_paired_rows, collect_and_write,
)


S1_ROOT = os.path.join(HERE, "stage_S1_raw_results")
S2_ROOT = os.path.join(HERE, "stage_S2_raw_results")
REVIEWED_MODEL_SHA256 = "a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1"


class RealCollectorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reuse_tasks = build_reuse_matrix(S1_ROOT, S2_ROOT)
        cls.joint_task = next(task for task in cls.reuse_tasks
                              if task["method"] == "equal_full"
                              and task["dataset"] == "kuairand"
                              and task["setting"] == "natural" and task["seed"] == 2022)
        cls.sd_task = next(task for task in cls.reuse_tasks
                           if task["method"] == "single_domain_mtl_reference"
                           and task["dataset"] == "kuairand"
                           and task["setting"] == "natural" and task["seed"] == 2022
                           and task["physical_domain"] == 1)
        cls.joint_manifest = _read_json(os.path.join(cls.joint_task["source_run_dir"],
                                                     "run_manifest.json"))
        cls.sd_manifest = _read_json(os.path.join(cls.sd_task["source_run_dir"],
                                                  "run_manifest.json"))
        cls.joint_metric = next(row for row in _read_csv(os.path.join(
            cls.joint_task["source_run_dir"], "metrics_test.csv"))
                                if row["checkpoint"] == "selected" and row["domain"] == "1"
                                and row["task"] == "0")
        cls.sd_metric = next(row for row in _read_csv(os.path.join(
            cls.sd_task["source_run_dir"], "metrics_test.csv"))
                             if row["checkpoint"] == "selected" and row["task"] == "0")
        cls.fresh_sample_task = next(task for task in build_fresh_matrix("unused")
                                     if task["objective_weighting"] == "sample"
                                     and task["dataset"] == "kuairand"
                                     and task["setting"] == "natural" and task["seed"] == 2022
                                     and task["shared_access"] != FULL)

    def _write_completed_queue(self, raw_root, queue_path, fresh_tasks):
        tasks = orchestrator.annotate_pairs(fresh_tasks)
        launch_plan = build_launch_plan(tasks)
        queue = {
            "stage": "S3", "amendment": "A1", "queue_schema_version": 3,
            "status": "RUNNING", "resume_state": "PLANNED",
            "raw_root": os.path.abspath(raw_root),
            "expected_run_count": EXPECTED_FRESH_RUNS,
            "expected_launch_group_count": EXPECTED_LAUNCH_GROUPS,
            "expected_seed_list": list(SEEDS), "checkpoint_labels": list(CHECKPOINTS),
            "tasks": tasks, "launch_plan": launch_plan,
            "records": [], "attempts": [], "launch_group_states": {},
            "strict_block_validations": {},
            "pending_sample_full_replacements": sorted(
                sample_target_pending_key(dataset, setting, seed)
                for dataset, setting in SETTINGS for seed in SEEDS),
            "retained_artifact_count": 0, "formal_launches": 24, "launched": 24,
        }
        orchestrator._refresh_amendment_runtime_state(queue)
        initial = orchestrator._amendment_queue_state_digest(queue)
        queue["amendment_initial_state_digest"] = initial
        queue["amendment_state_digest"] = initial
        queue["amendment_history"] = []
        queue["amendment_history_digest"] = orchestrator._digest([])
        group_states = {}
        for group in launch_plan:
            if group["group_kind"] == "SAMPLE_FULL_RESOURCE_PAIR":
                group_states[group["launch_group_id"]] = "COMPLETE"
            elif group == launch_plan[0]:
                group_states[group["launch_group_id"]] = (
                    "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED")
            else:
                group_states[group["launch_group_id"]] = "COMPLETE_STRICT_SOURCE_PAIRED"
        full_keys = {sample_full_replacement_key(dataset, setting, seed)
                     for dataset, setting in SETTINGS for seed in SEEDS}
        queue["records"] = [{
            "formal_run_key": task["formal_run_key"],
            "source_formal_run_key": task.get("source_formal_run_key", ""),
            "launch_group_id": task["launch_group_id"],
            "group_kind": task["launch_group_kind"],
            "status": "VALID_ACCEPTED", "artifact_valid": True,
            "scientific_pair_state": "STRICT_SOURCE_PAIRED",
            "resolved_by_sample_full_key": (
                task["source_formal_run_key"] if task["formal_run_key"] in
                {sample_target_pending_key(task["dataset"], task["setting"], task["seed"])}
                else ""),
            "run_dir": task["run_dir"],
            "canonical_tree_digest": orchestrator._tree_digest(task["run_dir"]),
        } for task in fresh_tasks]
        queue["launch_group_states"] = group_states
        queue["pending_sample_full_replacements"] = []
        queue["strict_block_validations"] = {
            "{}|{}|{}".format(dataset, setting, seed): {
                "dataset": dataset, "setting": setting, "seed": seed,
                "strict_source_identity": True, "pairing_valid": True,
            }
            for dataset, setting in SETTINGS for seed in SEEDS
        }
        queue["records"] = [
            dict(row, resolved_by_sample_full_key=(
                sample_full_replacement_key(task["dataset"], task["setting"], task["seed"])
                if task["formal_run_key"] in {
                    sample_target_pending_key(task["dataset"], task["setting"], task["seed"])}
                else ""))
            for task, row in zip(fresh_tasks, queue["records"])
        ]
        queue["status"] = "COMPLETE"; queue["resume_state"] = "COMPLETE"
        orchestrator._refresh_amendment_runtime_state(queue)
        orchestrator._write_amendment_queue_state(queue_path, queue, "fixture_complete")
        return queue

    def _prepare_full_collector_fixture(self, temp_root, *, missing_prediction=None,
                                        tamper_prediction=None):
        raw_root = os.path.join(temp_root, "fresh_raw")
        output_root = os.path.join(temp_root, "output")
        queue_path = os.path.join(temp_root, "a1_queue.json")
        fresh_tasks = build_fresh_matrix(raw_root)
        reuse_by_factor = {
            (task["dataset"], task["setting"], task["seed"], task["objective_weighting"]): task
            for task in self.reuse_tasks if task["run_type"] == "joint"
        }
        historical_sample_full = {
            (task["dataset"], task["setting"], task["seed"]): task
            for task in build_historical_audit_matrix(S1_ROOT, S2_ROOT)
            if task.get("method") == "sample_full"
        }
        for task in fresh_tasks:
            if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE \
                    or task["objective_weighting"] == "sample":
                source = historical_sample_full[
                    (task["dataset"], task["setting"], task["seed"])]
            else:
                source = reuse_by_factor[(task["dataset"], task["setting"], task["seed"],
                                          task["objective_weighting"])]
            source_dir = str(source["source_run_dir"])
            run_dir = str(task["run_dir"])
            os.makedirs(run_dir, exist_ok=False)
            manifest = _read_json(os.path.join(source_dir, "run_manifest.json"))
            manifest.update({
                "stage": "S3", "status": "OK", "formal_run_key": task["formal_run_key"],
                "fresh_role": task["fresh_role"], "provenance_type": task["fresh_role"],
                "shared_access": task["shared_access"],
            })
            manifest["code_hashes"] = dict(manifest.get("code_hashes", {}))
            manifest["code_hashes"]["m3oe/MDMTRec/models/multi_domain/model.py"] = \
                REVIEWED_MODEL_SHA256
            for name in ("metrics_test.csv", "metrics_summary.csv", "predictions_test.npz"):
                if name == "predictions_test.npz" and task["formal_run_key"] == missing_prediction:
                    continue
                shutil.copyfile(os.path.join(source_dir, name), os.path.join(run_dir, name))
            if task["formal_run_key"] == tamper_prediction:
                import numpy as np
                prediction_path = os.path.join(run_dir, "predictions_test.npz")
                with np.load(prediction_path, allow_pickle=False) as payload:
                    arrays = {key: np.asarray(payload[key]) for key in payload.files}
                arrays["pred_selected"] = np.clip(arrays["pred_selected"] + 0.03, 1e-5, 1 - 1e-5)
                np.savez_compressed(prediction_path, **arrays)
            manifest["artifact_hashes"] = {}
            for name in ("metrics_test.csv", "metrics_summary.csv", "predictions_test.npz"):
                path = os.path.join(run_dir, name)
                if os.path.isfile(path):
                    with open(path, "rb") as stream:
                        manifest["artifact_hashes"][name] = hashlib.sha256(stream.read()).hexdigest()
            with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True)
        self._write_completed_queue(raw_root, queue_path, fresh_tasks)
        return raw_root, output_root, queue_path

    def test_real_list_manifest_maps_joint_and_single_domain_to_same_physical_domain(self):
        joint = _row_for_metric(self.joint_task, self.joint_manifest, self.joint_metric,
                                self.joint_task["source_run_dir"])
        single = _row_for_metric(self.sd_task, self.sd_manifest, self.sd_metric,
                                 self.sd_task["source_run_dir"])
        self.assertEqual(joint["physical_domain"], 1)
        self.assertEqual(single["physical_domain"], 1)
        self.assertEqual(joint["physical_domain_identity"], "tab1")
        self.assertEqual(single["physical_domain_identity"], "tab1")
        self.assertTrue({
            "objective_weighting", "shared_access", "target_local_domain",
            "target_physical_domain", "gradient_partition_hash", "access_formula",
        }.issubset(set(RAW_FIELDS)))
        self.assertEqual(joint["objective_weighting"], "equal")
        self.assertEqual(joint["shared_access"], FULL)
        self.assertEqual(joint["target_local_domain"], 1)
        self.assertEqual(joint["target_physical_domain"], "tab1")
        self.assertEqual(single["shared_access"], "NA")
        self.assertEqual(single["target_physical_domain"], "tab1")

    def test_mapping_shaped_domain_fields_remain_compatible(self):
        manifest = copy.deepcopy(self.joint_manifest)
        manifest["local_to_physical"] = {str(i): value for i, value in
                                          enumerate(manifest["local_to_physical"])}
        manifest["physical_domain_identity"] = {str(i): value for i, value in
                                                 enumerate(manifest["physical_domain_identity"])}
        row = _row_for_metric(self.joint_task, manifest, self.joint_metric,
                              self.joint_task["source_run_dir"])
        self.assertEqual(row["physical_domain"], 1)
        self.assertEqual(row["physical_domain_identity"], "tab1")

    def test_missing_wrong_length_and_task_identity_mismatch_fail_closed(self):
        cases = []
        missing = copy.deepcopy(self.joint_manifest)
        missing.pop("physical_domain_identity")
        cases.append(missing)
        wrong_length = copy.deepcopy(self.joint_manifest)
        wrong_length["local_to_physical"] = [0, 1]
        cases.append(wrong_length)
        wrong_identity_length = copy.deepcopy(self.joint_manifest)
        wrong_identity_length["physical_domain_identity"] = ["tab0", "tab1"]
        cases.append(wrong_identity_length)
        for manifest in cases:
            with self.subTest(manifest=manifest.get("local_to_physical")):
                with self.assertRaises(ValueError):
                    _row_for_metric(self.joint_task, manifest, self.joint_metric,
                                    self.joint_task["source_run_dir"])
        mismatched_task = dict(self.sd_task, physical_domain=2)
        with self.assertRaisesRegex(ValueError, "physical ID disagrees"):
            _row_for_metric(mismatched_task, self.sd_manifest, self.sd_metric,
                            self.sd_task["source_run_dir"])

    def test_pair_identity_mismatch_is_rejected_for_real_manifest_rows(self):
        target = _row_for_metric(self.fresh_sample_task, self.joint_manifest,
                                 self.joint_metric, self.joint_task["source_run_dir"])
        reference = _row_for_metric(self.sd_task, self.sd_manifest, self.sd_metric,
                                    self.sd_task["source_run_dir"])
        reference["physical_domain_identity"] = "tab4"
        with self.assertRaisesRegex(ValueError, "physical-domain reference missing"):
            build_paired_rows([target, reference])

    def test_full_collector_dry_run_uses_real_reuse_artifacts_and_temp_outputs(self):
        with tempfile.TemporaryDirectory(prefix="stage_s3_real_collector_") as temp_root:
            raw_root, output_root, queue_path = self._prepare_full_collector_fixture(temp_root)
            with mock.patch.object(orchestrator, "validate_fresh_run", return_value=(True, [])):
                result = collect_and_write(raw_root, S1_ROOT, S2_ROOT,
                                           output_root=output_root, require_complete=True,
                                           queue_manifest=queue_path)
            expected = {
                "raw": EXPECTED_RAW_ROWS,
                "paired": EXPECTED_PAIRED_ROWS,
                "interactions": EXPECTED_INTERACTION_ROWS,
                "inventory": EXPECTED_INVENTORY_ROWS,
                "strict_paired": EXPECTED_STRICT_PAIRED_ROWS,
                "legacy_context_paired": EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS,
            }
            self.assertEqual(result["cardinalities"], expected)
            raw_rows = _read_csv(result["paths"]["raw"])
            sample_full = next(row for row in raw_rows
                               if row["condition"] == "sample_full")
            self.assertEqual(sample_full["objective_weighting"], "sample")
            self.assertEqual(sample_full["shared_access"], FULL)
            self.assertEqual(sample_full["target_local_domain"], "1")
            self.assertEqual(sample_full["target_physical_domain"], "tab1")
            self.assertTrue(sample_full["access_formula"])
            for name in ("raw", "paired", "interactions", "runs", "aggregates"):
                self.assertTrue(os.path.isfile(result["paths"][name]))
                self.assertTrue(os.path.commonpath((temp_root, result["paths"][name])) == temp_root)
            # A parent-abort recovery may preserve evidence under the raw root;
            # this collector fixture must not create any canonical accepted run.
            self.assertFalse(os.path.exists(os.path.join(
                HERE, "stage_S3_raw_results", "target_only_shared")))

    def test_collector_rejects_missing_predictions_even_when_metrics_and_queue_exist(self):
        with tempfile.TemporaryDirectory(prefix="stage_s3_missing_predictions_") as temp_root:
            task_key = str(build_fresh_matrix(os.path.join(temp_root, "fresh"))[0][
                "formal_run_key"])
            raw_root, output_root, queue_path = self._prepare_full_collector_fixture(
                temp_root, missing_prediction=task_key)
            with mock.patch.object(orchestrator, "validate_fresh_run", return_value=(True, [])):
                with self.assertRaisesRegex(ValueError, "missing artifact predictions_test.npz"):
                    collect_and_write(raw_root, S1_ROOT, S2_ROOT, output_root=output_root,
                                      require_complete=True, queue_manifest=queue_path)

    def test_collector_recomputes_metrics_and_rejects_tampered_prediction_values(self):
        with tempfile.TemporaryDirectory(prefix="stage_s3_tampered_predictions_") as temp_root:
            task_key = str(build_fresh_matrix(os.path.join(temp_root, "fresh"))[0][
                "formal_run_key"])
            raw_root, output_root, queue_path = self._prepare_full_collector_fixture(
                temp_root, tamper_prediction=task_key)
            with mock.patch.object(orchestrator, "validate_fresh_run", return_value=(True, [])):
                with self.assertRaisesRegex(ValueError, "prediction-level metric validation failed"):
                    collect_and_write(raw_root, S1_ROOT, S2_ROOT, output_root=output_root,
                                      require_complete=True, queue_manifest=queue_path)


if __name__ == "__main__":
    unittest.main()
