"""Focused Stage E6 matrix, comparator, and raw report checks."""

import csv
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E6_calibration import E6_ARM_KEYS, E6_FORMAL_SETTING, E6_EXPECTED_SEEDS  # noqa: E402
from stage_E6_orchestrator import (  # noqa: E402
    build_tasks,
    compare_e6_block,
    validate_matrix_tasks,
)
import stage_E6_report as e6_report  # noqa: E402
from stage_E6_report import collect  # noqa: E402


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _trace(path: Path, fingerprint: str = "same", epochs: int = 1):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "kind", "epoch", "step", "batch_fingerprint", "batch_size",
            "domain_counts", "task_counts", "train_loss",
        ])
        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow({"kind": "train", "epoch": epoch, "step": 0,
                             "batch_fingerprint": fingerprint, "batch_size": 6,
                             "domain_counts": "[2,2,2]", "task_counts": "[6,6]",
                             "train_loss": 1.0})
            writer.writerow({"kind": "architecture", "epoch": epoch, "step": -1,
                             "batch_fingerprint": fingerprint, "batch_size": 6,
                             "domain_counts": "[2,2,2]", "task_counts": "[6,6]",
                             "train_loss": 1.0})


def _make_block(root: Path, mismatch_arm: str = ""):
    for arm in E6_ARM_KEYS:
        run_dir = root / arm / E6_FORMAL_SETTING / "seed2022"
        run_dir.mkdir(parents=True)
        common = {
            "stage": "E6", "status": "OK", "arm_key": arm, "setting": E6_FORMAL_SETTING,
            "seed": 2022, "dataset": "MovieLens-1M", "dataset_identity": "dataset",
            "device_identity": "cpu", "domain_num": 3, "task_num": 2,
            "role_by_domain": {"0": "L", "1": "M", "2": "S"},
            "role_sizes": {"L": 150000, "M": 34500, "S": 3000},
            "train_domain_counts": {"0": 150000, "1": 34500, "2": 3000},
            "domain_shares": [0.8, 0.184, 0.016], "split_counts": {},
            "split_row_fingerprints": {"train": "train", "validation": "val", "test": "test"},
            "positive_rates": {}, "initial_parameter_state_hash": "initial",
            "partition": {"shared": ["p"]}, "model_identity": {"class": "MDMTRec"},
            "formal_protocol": {"stage": "E6"}, "oracle_audit": {},
            "observed_epochs": [0], "formal_valid": False,
        }
        _write_json(run_dir / "run_manifest.json", common)
        _write_json(run_dir / "config.json", {"arm_key": arm, "setting": E6_FORMAL_SETTING,
                                                "formal_protocol": {"stage": "E6"},
                                                "formal_valid": False})
        _trace(run_dir / "batch_trace.csv", "bad" if arm == mismatch_arm else "same")


def test_fixed_matrix_is_six_arms_by_four_seeds():
    tasks = build_tasks("raw")
    valid, reasons = validate_matrix_tasks(tasks)
    assert valid, reasons
    assert len(tasks) == 24
    assert {task["arm_key"] for task in tasks} == set(E6_ARM_KEYS)
    assert {task["setting"] for task in tasks} == {E6_FORMAL_SETTING}
    assert {task["seed"] for task in tasks} == set(E6_EXPECTED_SEEDS)


def test_block_comparator_detects_common_batch_identity_mismatch(tmp_path):
    _make_block(tmp_path, mismatch_arm="adaptive_ema")
    dirs = {arm: str(tmp_path / arm / E6_FORMAL_SETTING / "seed2022") for arm in E6_ARM_KEYS}
    result = compare_e6_block(dirs, require_formal=False)
    assert result["status"] == "INVALID"
    assert any(item.get("reason") == "batch_identity_mismatch" for item in result["reasons"])


def test_report_enumerates_all_missing_paths(tmp_path):
    rows, manifest = collect(str(tmp_path / "raw"))
    assert rows == []
    assert manifest["expected_run_count"] == 24
    assert manifest["missing_run_count"] == 24
    assert len(manifest["runs"]) == 24


def _make_report_fixture(root: Path):
    """Create one complete seed block with minimal raw metric rows."""
    for arm in E6_ARM_KEYS:
        run_dir = root / arm / E6_FORMAL_SETTING / "seed2022"
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "run_manifest.json", {
            "stage": "E6", "status": "OK", "formal_valid": True,
            "arm_key": arm, "setting": E6_FORMAL_SETTING, "seed": 2022,
            "block_status": "VALID", "block_valid": True,
        })
        _write_json(run_dir / "config.json", {"stage": "E6", "arm_key": arm,
                                                "setting": E6_FORMAL_SETTING,
                                                "formal_valid": True})
        for name in ("stdout.log", "model.pth"):
            (run_dir / name).write_text("raw\n", encoding="utf-8")
        _write_json(run_dir / "nonfinite_inventory.json", {"unexplained_count": 0})
        for name in ("controller_step.csv", "controller_epoch.csv", "shared_gradient_step.csv",
                     "shared_gradient_epoch.csv", "batch_trace.csv"):
            (run_dir / name).write_text("field\n", encoding="utf-8")
        with (run_dir / "metrics_epoch.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "epoch", "cell", "test_auc", "test_logloss", "transfer_auc",
                "transfer_logloss", "stopped_censored",
            ])
            writer.writeheader()
            for epoch in range(3):
                for cell in range(6):
                    writer.writerow({"epoch": epoch, "cell": cell,
                                     "test_auc": 0.70 + 0.01 * cell + 0.001 * epoch,
                                     "test_logloss": 0.40, "transfer_auc": 0.10,
                                     "transfer_logloss": 0.01, "stopped_censored": 0})
        with (run_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["arm_key", "cell", "selected_epoch",
                                                         "common_support", "best_test_auc"])
            writer.writeheader()
            for cell in range(6):
                writer.writerow({"arm_key": arm, "cell": cell, "selected_epoch": 0,
                                 "common_support": 0, "best_test_auc": 0.70 + 0.01 * cell})


def test_report_rejects_invalid_block_and_writes_last_common_endpoint(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _make_report_fixture(raw)

    def valid_block(_root, seed):
        if int(seed) == 2022:
            return {"status": "VALID", "valid": True, "common_epochs": [0, 1, 2],
                    "common_support_count": 6, "last_common_epoch": 2}
        return {"status": "PENDING", "valid": None}

    monkeypatch.setattr(e6_report, "_seed_block", valid_block)
    rows, manifest = e6_report.collect(str(raw))
    valid_rows = [row for row in rows if row["seed"] == 2022]
    assert manifest["available_run_count"] == 6
    assert all(row["status"] == "OK" for row in valid_rows)
    assert all(str(row["common_support"]) == "1" for row in valid_rows)
    assert all(str(row["last_common_epoch"]) == "2" for row in valid_rows)
    assert float(valid_rows[0]["last_common_test_auc"]) > 0.70

    def invalid_block(_root, seed):
        if int(seed) == 2022:
            return {"status": "INVALID", "valid": False,
                    "common_epochs": [], "common_support_count": 0,
                    "last_common_epoch": None}
        return {"status": "PENDING", "valid": None}

    monkeypatch.setattr(e6_report, "_seed_block", invalid_block)
    invalid_rows, invalid_manifest = e6_report.collect(str(raw))
    invalid_seed_rows = [row for row in invalid_rows if row["seed"] == 2022]
    assert invalid_manifest["available_run_count"] == 0
    assert invalid_manifest["invalid_run_count"] == 6
    assert all(row["status"] == "INVALID" for row in invalid_seed_rows)
    assert all(row["common_support"] == "" for row in invalid_seed_rows)
