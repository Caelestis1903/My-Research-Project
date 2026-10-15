from __future__ import annotations

import os
import sys

import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S3_protocol import (
    ACCESS, CHECKPOINTS, EXPECTED_FRESH_RUNS, EXPECTED_INTERACTION_ROWS, EXPECTED_INVENTORY_ROWS,
    EXPECTED_PAIRED_ROWS, EXPECTED_RAW_ROWS, build_co_run_plan,
    FULL, FRESH_SAMPLE_FULL_ROLE, EXPECTED_SAMPLE_FULL_REPLACEMENTS,
    EXPECTED_STRICT_PAIRED_ROWS, EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS,
    build_fresh_matrix, build_historical_audit_matrix, build_launch_plan,
    build_reuse_matrix, validate_fresh_matrix, validate_reuse_matrix,
)
from stage_S3_report import (
    PAIRED_FIELDS, build_interaction_rows, build_paired_rows, build_run_inventory,
    validate_fresh_reuse_pair,
)


def _synthetic_raw_rows():
    rows = []
    conditions = ("equal_full", "sample_full", "equal_target_only_shared",
                  "sample_target_only_shared")
    for dataset, setting in (("kuairand", "natural"), ("movielens", "d2_large")):
        for seed in (2022, 2023, 2024, 2025):
            for domain in range(3):
                physical = "{}-physical-{}".format(dataset, domain)
                for task in range(2):
                    for checkpoint in CHECKPOINTS:
                        epoch = -1 if checkpoint == "selected" else (
                            49 if checkpoint == "final" else int(checkpoint.rsplit("_", 1)[1]))
                        for condition in conditions:
                            rows.append({
                                "dataset": dataset, "setting": setting, "seed": seed,
                                "evaluated_local_domain": domain, "task": task,
                                "checkpoint": checkpoint, "evaluated_epoch": epoch,
                                "condition": condition, "physical_domain_identity": physical,
                                "physical_domain": domain,
                                "strict_source_identity": True,
                                "forward_model_source_sha256": "a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1",
                                "formal_run_key": "{}|{}|{}|{}|{}|{}".format(
                                    condition, dataset, setting, seed, domain, task),
                                "selected_epoch": 10, "auc": 0.6 + seed * 1e-8,
                                "logloss": 0.5, "checkpoint_hash": "hash-{}".format(condition),
                            })
                        rows.append({
                            "dataset": dataset, "setting": setting, "seed": seed,
                            "evaluated_local_domain": 0, "task": task,
                            "checkpoint": checkpoint, "evaluated_epoch": epoch,
                            "condition": "single_domain_mtl_reference",
                            "physical_domain_identity": physical, "physical_domain": domain,
                            "strict_source_identity": False,
                            "forward_model_source_sha256": "",
                            "formal_run_key": "sd|{}|{}|{}|{}|{}".format(
                                dataset, setting, seed, domain, task),
                            "selected_epoch": 10, "auc": 0.59, "logloss": 0.51,
                            "checkpoint_hash": "sdhash-{}".format(domain),
                        })
    return rows


class PairingTests(unittest.TestCase):
    def test_exact_amended_matrix_reuse_inventory_and_twelve_launch_groups(self):
        fresh = build_fresh_matrix("fresh-root")
        reuse = build_reuse_matrix("s1-root", "s2-root")
        historical = build_historical_audit_matrix("s1-root", "s2-root")
        validate_fresh_matrix(fresh); validate_reuse_matrix(historical)
        self.assertEqual(len(fresh), EXPECTED_FRESH_RUNS)
        self.assertEqual(len(reuse), 32)
        self.assertEqual(sum(task["fresh_role"] == FRESH_SAMPLE_FULL_ROLE for task in fresh),
                         EXPECTED_SAMPLE_FULL_REPLACEMENTS)
        self.assertEqual(sum(task["shared_access"] == FULL for task in fresh), 8)
        self.assertEqual(sum(task["method"] == "sample_full" for task in historical), 8)
        self.assertEqual(sum(task["method"] == "equal_full" for task in reuse), 8)
        self.assertEqual(sum(task["method"] == "single_domain_mtl_reference" for task in reuse), 24)
        plan = build_co_run_plan(fresh)
        self.assertEqual(len(plan), 8)
        launch_plan = build_launch_plan(fresh)
        self.assertEqual(len(launch_plan), 12)
        self.assertEqual(sum(group["group_kind"] == "SAMPLE_FULL_RESOURCE_PAIR"
                             for group in launch_plan), 4)
        for pair in plan:
            self.assertEqual(len(pair["formal_run_keys"]), 2)
            self.assertIn("equal", pair["formal_run_keys"][0])
            self.assertIn("sample", pair["formal_run_keys"][1])
            self.assertIn(pair["seed"], (2022, 2023, 2024, 2025))


    def test_structure_contract_is_1680_2688_336_and_56(self):
        raw = _synthetic_raw_rows()
        self.assertEqual(len(raw), EXPECTED_RAW_ROWS)
        paired = build_paired_rows(raw)
        interactions = build_interaction_rows(raw)
        inventory = build_run_inventory(build_fresh_matrix("fresh"),
                                        build_reuse_matrix("s1", "s2"))
        self.assertEqual(len(paired), EXPECTED_PAIRED_ROWS)
        self.assertEqual(len(interactions), EXPECTED_INTERACTION_ROWS)
        self.assertEqual(len(inventory), EXPECTED_INVENTORY_ROWS)
        strict = [row for row in paired if row["strict_source_identity"]]
        legacy = [row for row in paired if not row["strict_source_identity"]]
        self.assertEqual(len(strict), EXPECTED_STRICT_PAIRED_ROWS)
        self.assertEqual(len(legacy), EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS)
        self.assertTrue({"lhs_forward_model_source_sha256",
                         "rhs_forward_model_source_sha256"}.issubset(set(PAIRED_FIELDS)))
        self.assertTrue(all(row["lhs_forward_model_source_sha256"]
                            == row["rhs_forward_model_source_sha256"]
                            for row in strict))
        self.assertTrue(all(row["contrast_evidence_class"] == "legacy_sd_context"
                            for row in legacy))
        self.assertTrue(all(row["contrast_evidence_class"] == "strict_joint_source_identity"
                            for row in interactions))
        self.assertTrue(all(row["pairing_valid"] for row in paired))


    def test_collector_rejects_duplicate_raw_identity(self):
        rows = _synthetic_raw_rows()
        with self.assertRaisesRegex(ValueError, "duplicate Stage S3 raw identity"):
            build_interaction_rows(rows + [dict(rows[0])])


    def test_pairing_audit_rejects_changed_immutable_dataset_identity(self):
        full = {
        "dataset": "kuairand", "dataset_key": "kuairand", "setting": "natural",
        "backbone": "m3oe", "seed": 2022, "domain_num": 3, "task_num": 2,
        "local_to_physical": [0, 1, 2], "physical_domain_identity": ["a", "b", "c"],
        "features": ["f"], "split_row_fingerprints": {"train": "x"},
        "split_row_id_hashes": {"train": "x"}, "base_split_fingerprints": {},
        "base_split_row_id_hashes": {}, "physical_view_fingerprints": {"train": "x"},
        "train_domain_counts": {"0": 1}, "domain_shares": {"0": 1.0}, "cell_stats": {},
        "preprocessing_metadata": {}, "dataset_file_hash": "same",
        "initial_parameter_state_hash": "init", "model_parameter_manifest": {},
        "forward_flops_estimate": 1, "optimizer": {}, "architecture_optimizer": {},
        "batches_per_epoch": 1, "examples_processed": 1, "architecture_examples_processed": 1,
        "optimizer_steps": 1, "architecture_optimizer_steps": 1,
        "batch_order_hash": "batch", "architecture_batch_order_hash": "arch",
        "batch_fingerprint_count": 1, "architecture_batch_fingerprint_count": 1,
        "fixed_final_epoch": 49, "common_epochs": [9, 19, 29, 39, 49],
        "metric_implementation": {"auc": "tie"}, "checkpoint_selection_frozen_before_test": True,
        "test_evaluated_during_training": False, "test_metric_in_epoch_history": False,
        "code_hashes": {"m3oe/MDMTRec/models/multi_domain/model.py": "model-source"},
        }
        self.assertEqual(validate_fresh_reuse_pair(full, dict(full)), [])
        corrupt = dict(full); corrupt["dataset_file_hash"] = "changed"
        self.assertIn("pairing_field_mismatch:dataset_file_hash",
                      validate_fresh_reuse_pair(corrupt, full))


if __name__ == "__main__":
    unittest.main()
