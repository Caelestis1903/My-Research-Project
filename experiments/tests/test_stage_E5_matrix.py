"""Focused E5 queue and block identity checks (no dataset or GPU)."""

import csv
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E5_orchestrator import build_tasks  # noqa: E402
from stage_E5_schedule import (  # noqa: E402
    E5_ARM_KEYS,
    E5_EPOCHS,
    E5_FORMAL_SETTING,
    compare_e5_block,
    validate_matrix_tasks,
)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_trace(path: Path, fingerprint: str = "same"):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "kind", "epoch", "step", "batch_fingerprint", "batch_size",
            "domain_counts", "task_counts",
        ])
        writer.writeheader()
        for epoch in range(E5_EPOCHS):
            writer.writerow({"kind": "train", "epoch": epoch, "step": 0,
                             "batch_fingerprint": fingerprint, "batch_size": 2,
                             "domain_counts": "[2,0,0]", "task_counts": "[2,2]"})
            writer.writerow({"kind": "architecture", "epoch": epoch, "step": -1,
                             "batch_fingerprint": fingerprint, "batch_size": 2,
                             "domain_counts": "[2,0,0]", "task_counts": "[2,2]"})


def _make_block(root: Path, mismatch_arm: str = ""):
    for arm in E5_ARM_KEYS:
        run_dir = root / arm / E5_FORMAL_SETTING / "seed2022"
        run_dir.mkdir(parents=True)
        manifest = {
            "stage": "E5", "status": "OK", "arm_key": arm, "setting": E5_FORMAL_SETTING,
            "seed": 2022, "observed_epochs": list(range(E5_EPOCHS)),
            "formal_valid": False, "epoch": E5_EPOCHS,
        }
        config = {"arm_key": arm, "epoch": E5_EPOCHS, "formal_valid": False}
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(run_dir / "config.json", config)
        _write_trace(run_dir / "batch_trace.csv", "mismatch" if arm == mismatch_arm else "same")


def test_matrix_is_exactly_five_arms_by_four_seeds(tmp_path):
    tasks = build_tasks(str(tmp_path))
    valid, reasons = validate_matrix_tasks(tasks)
    assert valid, reasons
    assert len(tasks) == 20
    assert {task["arm_key"] for task in tasks} == set(E5_ARM_KEYS)
    assert {task["seed"] for task in tasks} == {2022, 2023, 2024, 2025}
    assert {task["setting"] for task in tasks} == {E5_FORMAL_SETTING}


def test_block_comparator_detects_batch_identity_mismatch(tmp_path):
    _make_block(tmp_path, mismatch_arm="forward")
    dirs = {arm: str(tmp_path / arm / E5_FORMAL_SETTING / "seed2022") for arm in E5_ARM_KEYS}
    result = compare_e5_block(dirs, require_formal=False)
    assert result["status"] == "INVALID"
    assert any(item.get("reason") == "batch_identity_mismatch" for item in result["reasons"])


def test_block_comparator_accepts_matching_ordered_support(tmp_path):
    _make_block(tmp_path)
    dirs = {arm: str(tmp_path / arm / E5_FORMAL_SETTING / "seed2022") for arm in E5_ARM_KEYS}
    result = compare_e5_block(dirs, require_formal=False)
    assert result["status"] == "VALID"
    assert result["trace_count"]["forward"] == 36
