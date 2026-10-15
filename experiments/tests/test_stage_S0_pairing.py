import os
import sys


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import json
from types import SimpleNamespace

import stage_S0_orchestrator as orchestrator
import pytest
from stage_S0_report import _mean, historical_reproduction_audit, select_control_family


def arm_rows(validation_auc, test_auc):
    rows = []
    for epoch in range(50):
        for cell in range(6):
            rows.append({
                "epoch": str(epoch), "selected_checkpoint": "1" if epoch == 4 else "0",
                "domain": str(cell % 3), "task": str(cell // 3), "cell": str(cell),
                "validation_auc": str(validation_auc), "test_auc": str(test_auc),
            })
    return rows


def test_control_family_selection_is_validation_only():
    block = {
        "uniform_090": arm_rows(0.70, 0.99),
        "uniform_p_tau": arm_rows(0.72, 0.10),
        "uniform_eq_tau": arm_rows(0.71, 0.80),
        "sample_adam": arm_rows(0.60, 0.99),
        "adam_p_tau": arm_rows(0.62, 0.10),
        "adam_eq_tau": arm_rows(0.61, 0.80),
    }
    assert select_control_family(block, "C_val") == "uniform_p_tau"
    assert select_control_family(block, "D_val") == "adam_p_tau"
    for rows in block.values():
        for row in rows:
            row["test_auc"] = "0.0" if row["test_auc"] == "0.99" else "1.0"
    assert select_control_family(block, "C_val") == "uniform_p_tau"
    assert select_control_family(block, "D_val") == "adam_p_tau"


def test_family_ties_use_preregistered_order():
    block = {arm: arm_rows(0.7, 0.5) for arm in (
        "uniform_090", "uniform_p_tau", "uniform_eq_tau",
        "sample_adam", "adam_p_tau", "adam_eq_tau")}
    assert select_control_family(block, "C_val") == "uniform_090"
    assert select_control_family(block, "D_val") == "sample_adam"


def test_block_audit_rejects_pair_hash_mismatch(tmp_path, monkeypatch):
    tasks = []
    common = {
        field: "same" for field in orchestrator.PAIR_FIELDS
    }
    common.update({
        "examples_processed": 100, "optimizer_steps": 10, "scheduler_steps": 0,
        "architecture_optimizer_steps": 50, "domain_shares": [0.8, 0.184, 0.016],
        "train_domain_counts": {"0": 150000, "1": 34500, "2": 3000},
        "ema_update_counts": [0, 0, 0], "shared_beta1": 0.9,
    })
    for arm in orchestrator.ARM_SPECS:
        run_dir = tmp_path / arm
        run_dir.mkdir()
        manifest = dict(common)
        manifest["shared_beta1"] = orchestrator.ARM_SPECS[arm]["shared_beta1"]
        if orchestrator.ARM_SPECS[arm]["kind"].startswith("ema"):
            manifest["ema_update_counts"] = [10, 10, 10]
        if arm == "adam_eq_tau":
            manifest["architecture_batch_order_hash"] = "different"
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tasks.append({"arm": arm, "rotation": "d0_large", "seed": 2022,
                      "run_dir": str(run_dir)})
    monkeypatch.setattr(orchestrator, "validate_run", lambda task, require_formal=True: (True, []))
    result = orchestrator.audit_block(tasks)
    assert result["valid"] is False
    assert any(item.get("field") == "architecture_batch_order_hash"
               for item in result["reasons"])


def test_report_rejects_nonfinite_instead_of_shrinking_group():
    with pytest.raises(ValueError, match="NaN/Inf"):
        _mean([0.7, float("nan")])


def test_resume_classification_uses_current_artifact_not_master(tmp_path, monkeypatch):
    task = {"formal_run_key": "sample_adam|d0_large|2022",
            "run_dir": str(tmp_path / "missing")}
    monkeypatch.setattr(orchestrator, "validate_run", lambda item: (False, ["missing"]))
    valid, pending, blocked = orchestrator.classify_task_state([task], rerun_failed=False)
    assert valid == 0 and pending == [task] and blocked == []
    os.makedirs(task["run_dir"])
    valid, pending, blocked = orchestrator.classify_task_state([task], rerun_failed=False)
    assert valid == 0 and pending == [] and len(blocked) == 1
    valid, pending, blocked = orchestrator.classify_task_state([task], rerun_failed=True)
    assert pending == [task] and blocked == []


def test_orchestrator_stops_after_first_failed_child(tmp_path, monkeypatch):
    tasks = [
        {"formal_run_key": "a|r|1", "arm": "sample_adam", "rotation": "d0_large",
         "seed": 2022, "run_dir": str(tmp_path / "one")},
        {"formal_run_key": "b|r|1", "arm": "comp1_domain", "rotation": "d0_large",
         "seed": 2022, "run_dir": str(tmp_path / "two")},
    ]
    args = SimpleNamespace(
        raw_root=str(tmp_path / "raw"), master=str(tmp_path / "master.csv"),
        summary_manifest=str(tmp_path / "queue.json"), python="python", device="cuda:0",
        dataset_path="dataset", oracle_root="oracles", role_sizes="150000,34500,3000",
        batch_size=40960, epoch=50, learning_rate=0.01, learning_rate_darts=0.001,
        weight_decay=1e-5, expert_num=4, shadow_patience=10, require_oracles=1,
        instrument=1, dry_run=False, max_runs=84, rerun_failed=False,
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    calls = []
    monkeypatch.setattr(orchestrator, "build_argparser", lambda: parser)
    monkeypatch.setattr(orchestrator, "build_tasks", lambda root: tasks)
    monkeypatch.setattr(orchestrator, "classify_task_state",
                        lambda items, rerun_failed: (0, list(items), []))
    monkeypatch.setattr(orchestrator, "runtime_preflight", lambda value: None)
    monkeypatch.setattr(orchestrator, "command_for", lambda task, value: ["fake", task["arm"]])
    monkeypatch.setattr(orchestrator, "validate_run", lambda task: (False, ["forced_failure"]))

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    assert orchestrator.main() == 1
    assert len(calls) == 1


def test_historical_ab_audit_is_descriptive_and_fail_closed(tmp_path):
    audits = {(rotation, seed): {"valid": False}
              for rotation in orchestrator.ROTATIONS for seed in orchestrator.SEEDS}
    rows = historical_reproduction_audit(str(tmp_path), {}, audits)
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == {"sample_adam", "comp1_domain"}
    assert all(row["status"] == "MISSING" for row in rows)
    assert all(row["included_in_formal_gate"] == 0 for row in rows)
