"""Stage S2 matrix and structural pairing tests; no formal training."""

from __future__ import annotations

import os
import sys
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S2_orchestrator import (  # noqa: E402
    S2_BATCH_FIELDS,
    build_tasks,
    command_for,
    lock_path_for_task,
    validate_batch_trace_contract,
    validate_matrix_tasks,
)
from stage_S2_protocol import (  # noqa: E402
    BACKBONE,
    COMMON_EPOCHS,
    SEEDS,
    equal_run_dir,
    build_reuse_matrix,
    validate_equal_matrix,
    validate_reuse_matrix,
)
from stage_S2_report import (  # noqa: E402
    CHECKPOINTS,
    PAIRED_FIELDS,
    RAW_FIELDS,
    aggregate_paired_rows,
    build_paired_rows,
    checkpoint_hash_for_label,
    fixed_physical_rotation_summary_rows,
    paired_training_epoch_mean_rows,
    paired_training_stage_summary_rows,
)


class Args:
    python = "python"
    dataset_path = "dataset"
    raw_root = "stage_S2_raw_results"


def test_fresh_matrix_is_exactly_16_equal_m3oe_keys(tmp_path):
    tasks = build_tasks(str(tmp_path))
    validate_equal_matrix(tasks)
    assert len(tasks) == 16
    assert Counter(task["seed"] for task in tasks) == {seed: 4 for seed in SEEDS}
    assert {task["backbone"] for task in tasks} == {BACKBONE}
    assert {task["method"] for task in tasks} == {"equal"}
    assert len({task["formal_run_key"] for task in tasks}) == 16
    ok, reasons = validate_matrix_tasks(tasks)
    assert ok and reasons == []


def test_fresh_paths_have_explicit_equal_namespace(tmp_path):
    path = equal_run_dir(str(tmp_path), "movielens", "d0_large", 2022)
    parts = Path(path).parts
    assert "equal" in parts
    assert parts[-4:] == ("d0_large", "m3oe", "seed2022", "joint")


def test_scheduler_lock_is_outside_canonical_child_dir(tmp_path):
    task = {"formal_run_key": "equal|movielens|d0_large|m3oe|2022",
            "run_dir": str(tmp_path / "equal" / "movielens" / "d0_large" / "m3oe" / "seed2022" / "joint")}
    args = Args()
    args.raw_root = str(tmp_path)
    lock = lock_path_for_task(task, args)
    assert Path(lock).parent != Path(task["run_dir"])
    assert Path(lock).parent.name == ".stage_S2_locks"


def test_reuse_matrix_is_exactly_64_frozen_s1_m3oe_keys(tmp_path):
    tasks = build_reuse_matrix(str(tmp_path))
    validate_reuse_matrix(tasks)
    assert len(tasks) == 64
    assert Counter(task["run_type"] for task in tasks) == {"joint": 16, "single_domain": 48}
    assert len({task["source_formal_run_key"] for task in tasks}) == 64


def test_command_is_equal_only_and_formal_budget_is_locked(tmp_path):
    command = command_for(build_tasks(str(tmp_path))[0], Args())
    joined = " ".join(command)
    assert "stage_S2_run.py" in joined
    assert "--epoch 50" in joined and "--batch_size 40960" in joined
    assert "--backbone m3oe" in joined
    assert "momentum" not in joined.lower() and "curriculum" not in joined.lower()


def _raw_row(method, dataset, setting, seed, physical, task, checkpoint):
    return {
        "stage": "S2", "method": method, "source_stage": "S2" if method == "equal" else "S1",
        "provenance_type": "fresh" if method == "equal" else "reused",
        "formal_run_key": "{}|{}|{}|{}|{}".format(method, dataset, setting, seed, physical),
        "source_formal_run_key": "", "dataset": dataset, "setting": setting, "backbone": "m3oe",
        "seed": seed, "physical_domain": physical, "physical_domain_identity": str(physical),
        "size_role": "L", "task": task, "task_identity": "task{}".format(task),
        "checkpoint": checkpoint, "evaluated_epoch": 49, "selected_epoch": 0,
        "auc": 0.6 + 0.001 * physical, "logloss": 0.5, "validation_auc_at_stage": 0.6,
        "validation_logloss_at_stage": 0.5, "validation_macro_auc_at_stage": 0.6,
        "validation_macro_logloss_at_stage": 0.5, "train_sample_count": 10,
        "validation_sample_count": 10, "test_sample_count": 10, "domain_share_p": 1 / 3,
        "loss_coefficient_w": 1.0, "effective_domain_coefficient_a": 1 / 3,
        "effective_cell_coefficient": 1 / 6, "objective_name": method,
        "objective_formula": "test", "initial_parameter_state_hash": "init",
        "batch_order_hash": "batch", "architecture_batch_order_hash": "arch",
        "split_row_id_hash": "split", "physical_view_fingerprint": "view",
        "dataset_file_hash": "dataset", "checkpoint_hash": "checkpoint",
        "code_inventory_hash": "code", "runtime_fingerprint": "runtime", "status": "OK",
    }


def test_pairing_contract_has_three_contrasts_and_exact_cardinality():
    rows = []
    for dataset, settings in (("movielens", ("d0_large", "d1_large", "d2_large")),
                              ("kuairand", ("natural",))):
        for setting in settings:
            for seed in SEEDS:
                for physical in range(3):
                    for task in range(2):
                        for checkpoint in CHECKPOINTS:
                            for method in ("equal", "sample", "single_domain_mtl_reference"):
                                rows.append(_raw_row(method, dataset, setting, seed, physical, task, checkpoint))
    paired = build_paired_rows(rows)
    assert len(paired) == 2016
    assert {row["comparison"] for row in paired} == {
        "equal_vs_single_domain", "sample_vs_single_domain", "equal_vs_sample",
    }
    assert all(row["pairing_valid"] == 1 for row in paired)
    assert set(PAIRED_FIELDS).issubset(paired[0])
    assert len(RAW_FIELDS) > 0


def test_pairing_corruption_is_fail_closed():
    rows = []
    for dataset, settings in (("movielens", ("d0_large", "d1_large", "d2_large")),
                              ("kuairand", ("natural",))):
        for setting in settings:
            for seed in SEEDS:
                for physical in range(3):
                    for task in range(2):
                        for checkpoint in CHECKPOINTS:
                            for method in ("equal", "sample", "single_domain_mtl_reference"):
                                rows.append(_raw_row(method, dataset, setting, seed, physical, task, checkpoint))
    rows[0]["dataset_file_hash"] = "corrupt"
    paired = build_paired_rows(rows)
    assert any(row["pairing_valid"] == 0 for row in paired)


def test_joint_reference_pairing_does_not_require_batch_order_identity():
    rows = []
    for method in ("equal", "sample", "single_domain_mtl_reference"):
        row = _raw_row(method, "kuairand", "natural", 2022, 0, 0, "final")
        if method == "sample":
            row["batch_order_hash"] = "joint-sample"
        if method == "single_domain_mtl_reference":
            row["batch_order_hash"] = "d1-reference"
        rows.append(row)
    # The helper expects the complete matrix; repeat this deliberate hash
    # difference across all keys while leaving physical-view evidence equal.
    full = []
    for dataset, settings in (("movielens", ("d0_large", "d1_large", "d2_large")),
                              ("kuairand", ("natural",))):
        for setting in settings:
            for seed in SEEDS:
                for physical in range(3):
                    for task in range(2):
                        for checkpoint in CHECKPOINTS:
                            for method in ("equal", "sample", "single_domain_mtl_reference"):
                                row = _raw_row(method, dataset, setting, seed, physical, task, checkpoint)
                                if method == "sample":
                                    row["batch_order_hash"] = "joint-sample"
                                if method == "single_domain_mtl_reference":
                                    row["batch_order_hash"] = "d1-reference"
                                full.append(row)
    paired = build_paired_rows(full)
    sample_reference = [row for row in paired if row["comparison"] == "sample_vs_single_domain"]
    assert sample_reference and all(row["pairing_valid"] == 1 for row in sample_reference)


def test_invalid_pair_rows_never_enter_aggregate_or_rotation_summaries():
    rows = []
    for dataset, settings in (("movielens", ("d0_large", "d1_large", "d2_large")),
                              ("kuairand", ("natural",))):
        for setting in settings:
            for seed in SEEDS:
                for physical in range(3):
                    for task in range(2):
                        for checkpoint in CHECKPOINTS:
                            for method in ("equal", "sample", "single_domain_mtl_reference"):
                                rows.append(_raw_row(method, dataset, setting, seed, physical, task, checkpoint))
    paired = build_paired_rows(rows)
    target = next(row for row in paired if row["comparison"] == "equal_vs_sample"
                  and row["dataset"] == "movielens" and row["setting"] == "d0_large"
                  and row["seed"] == 2022 and row["physical_domain"] == 0
                  and row["task"] == 0 and row["checkpoint"] == "final")
    target["pairing_valid"] = 0
    aggregate = aggregate_paired_rows(paired)
    group = next(row for row in aggregate if row["comparison"] == "equal_vs_sample"
                 and row["dataset"] == "movielens" and row["setting"] == "d0_large"
                 and row["physical_domain"] == 0 and row["task"] == 0 and row["checkpoint"] == "final")
    assert group["seed_count"] == 3
    assert all(row["pairing_valid_count"] == row["seed_count"]
               for row in aggregate)
    rotation = fixed_physical_rotation_summary_rows(paired)
    assert rotation
    assert {row["rotation_scope"] for row in rotation if row["dataset"] == "kuairand"} == {
        "natural_fixed_physical_domain"
    }
    assert paired_training_stage_summary_rows(paired)


def test_batch_trace_contract_recomputes_hashes_and_counters():
    coefficients = {"coefficient_table_hash": "coeff"}
    rows = []
    for epoch in range(2):
        rows.append({"stage": "S2", "kind": "train", "epoch": epoch, "step": 0,
                     "batch_fingerprint": "train-{}".format(epoch), "batch_size": 4,
                     "total_weighted_mass": 4.0, "train_objective": 0.5,
                     "objective_name": "equal_domain_task_mean_bce",
                     "coefficient_table_hash": "coeff"})
        rows.append({"stage": "S2", "kind": "architecture", "epoch": epoch, "step": -1,
                     "batch_fingerprint": "arch-{}".format(epoch), "batch_size": 4,
                     "total_weighted_mass": 4.0, "train_objective": 0.5,
                     "objective_name": "equal_domain_task_mean_bce",
                     "coefficient_table_hash": "coeff"})
    train = ["train-0", "train-1"]
    arch = ["arch-0", "arch-1"]
    manifest = {
        "batch_order_hash": hashlib.sha256(json.dumps(train).encode()).hexdigest(),
        "architecture_batch_order_hash": hashlib.sha256(json.dumps(arch).encode()).hexdigest(),
        "train_batches_total": 2, "batch_fingerprint_count": 2,
        "architecture_batch_fingerprint_count": 2, "optimizer_steps": 2,
        "examples_processed": 8, "architecture_optimizer_steps": 2,
        "architecture_examples_processed": 8,
    }
    assert validate_batch_trace_contract(S2_BATCH_FIELDS, rows, manifest, coefficients,
                                         {"0": 4}, epochs=2, batch_size=4) == []
    rows[0]["batch_fingerprint"] = ""
    assert "batch_fingerprint_missing" in validate_batch_trace_contract(
        S2_BATCH_FIELDS, rows, manifest, coefficients, {"0": 4}, epochs=2, batch_size=4)


def test_checkpoint_label_maps_selected_and_final_artifacts():
    checkpoints = {"selected_checkpoint.pth": "selected-hash", "final_checkpoint.pth": "final-hash",
                   "common_epoch_49.pth": "common-hash"}
    assert checkpoint_hash_for_label(checkpoints, "selected") == "selected-hash"
    assert checkpoint_hash_for_label(checkpoints, "final") == "final-hash"
    assert checkpoint_hash_for_label(checkpoints, "common_epoch_49") == "common-hash"


def test_training_stage_gate_aggregates_four_seeds_before_counting_epochs():
    rows = []
    negative_mean_epochs = {9, 19, 29}
    for epoch in COMMON_EPOCHS:
        values = ([-0.10, -0.10, 0.01, 0.01] if epoch in negative_mean_epochs
                  else [-0.01, 0.01, 0.01, 0.01])
        for seed, delta in zip(SEEDS, values):
            rows.append({
                "comparison": "equal_vs_single_domain", "dataset": "movielens",
                "setting": "d0_large", "physical_domain": 0, "task": 0,
                "checkpoint": "common_epoch_{}".format(epoch), "seed": seed,
                "delta_auc": delta, "delta_logloss": 0.01, "pairing_valid": 1,
            })
    epoch_rows = paired_training_epoch_mean_rows(rows)
    summary = paired_training_stage_summary_rows(rows)
    assert len(epoch_rows) == len(COMMON_EPOCHS)
    assert all(row["seed_count"] == 4 for row in epoch_rows)
    assert summary[0]["common_epoch_count"] == 5
    assert summary[0]["negative_matched_epoch_auc_count"] == 3
    assert summary[0]["stage_support_3_of_5"] == 1
    # An incorrect seed-epoch sign count would report 8 negatives here.
    assert sum(float(row["delta_auc"]) < 0 for row in rows) == 8


def test_training_stage_gate_rejects_many_negative_seeds_when_epoch_means_are_positive():
    rows = []
    for epoch in COMMON_EPOCHS:
        for seed, delta in zip(SEEDS, [-0.10, 0.04, 0.04, 0.04]):
            rows.append({
                "comparison": "equal_vs_sample", "dataset": "kuairand",
                "setting": "natural", "physical_domain": 1, "task": 1,
                "checkpoint": "common_epoch_{}".format(epoch), "seed": seed,
                "delta_auc": delta, "delta_logloss": 0.0, "pairing_valid": 1,
            })
    summary = paired_training_stage_summary_rows(rows)
    assert summary[0]["negative_matched_epoch_auc_count"] == 0
    assert summary[0]["stage_support_3_of_5"] == 0
    assert sum(float(row["delta_auc"]) < 0 for row in rows) == 5
