import os
import sys
from collections import Counter
import csv
import json


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pytest

from stage_S0_orchestrator import (ARM_SPECS, REQUIRED_FILES, ROTATIONS, SEEDS,
                                   assert_formal_launcher, build_tasks, command_for,
                                   validate_run)
from stage_S0_protocol import FORMAL_PROTOCOL


class Args:
    python = "python"
    device = "cuda:0"
    dataset_path = "dataset"
    oracle_root = "oracles"
    role_sizes = "150000,34500,3000"
    batch_size = 40960
    epoch = 50
    learning_rate = 0.01
    learning_rate_darts = 0.001
    weight_decay = 1e-5
    expert_num = 4
    shadow_patience = 10
    require_oracles = 1
    instrument = 1


def test_fixed_matrix_is_exactly_84_unique_tasks(tmp_path):
    tasks = build_tasks(str(tmp_path))
    assert len(tasks) == 84
    assert len({task["formal_run_key"] for task in tasks}) == 84
    assert Counter(task["arm"] for task in tasks) == {arm: 12 for arm in ARM_SPECS}
    assert Counter(task["rotation"] for task in tasks) == {rotation: 28 for rotation in ROTATIONS}
    assert Counter(task["seed"] for task in tasks) == {seed: 21 for seed in SEEDS}


def test_commands_lock_formal_budget_and_do_not_add_forbidden_methods(tmp_path):
    command = command_for(build_tasks(str(tmp_path))[0], Args())
    joined = " ".join(command)
    assert "--epoch 50" in joined
    assert "--batch_size 40960" in joined
    assert "--role_sizes 150000,34500,3000" in joined
    assert "curriculum" not in joined.lower()
    assert "controller" not in joined.lower()
    assert "schedule" not in joined.lower()


def test_formal_launcher_rejects_critical_protocol_deviation():
    args = Args()
    assert_formal_launcher(args)
    args.batch_size = 1024
    with pytest.raises(ValueError, match="batch_size"):
        assert_formal_launcher(args)
    assert "architecture_optimizer_state.pth" in REQUIRED_FILES


def test_validator_explicitly_rejects_nonfinite_metric(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = {"arm": "sample_adam", "rotation": "d0_large", "seed": 2022,
            "formal_run_key": "sample_adam|d0_large|2022", "run_dir": str(run_dir)}
    config = dict(FORMAL_PROTOCOL)
    config.update({"formal_valid": True, "formal_protocol": dict(FORMAL_PROTOCOL),
                   "sample_objective": "w_d=D*p_d; sum over domain-task mean BCE",
                   "control_family_selection_uses_test": False})
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {"stage": "S0", "status": "OK", **{k: task[k] for k in
                ("arm", "rotation", "seed", "formal_run_key")}, "fixed_final_epoch": 49,
                "formal_valid": True, "formal_protocol": dict(FORMAL_PROTOCOL)}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = ["epoch", "domain", "task", "cell", "domain_size", "domain_share",
              "sample_weighted_train_loss", "validation_auc", "validation_logloss",
              "test_auc", "test_logloss", "oracle_auc", "oracle_logloss", "transfer_auc",
              "transfer_logloss", "raw_auc_nt", "raw_logloss_nt", "practical_auc_nt",
              "concordant_practical_nt", "selected_checkpoint", "fixed_final", "common_epoch"]
    with open(run_dir / "metrics_epoch.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(50):
            for task_id in range(2):
                for domain in range(3):
                    row = {field: 0.5 for field in fields}
                    row.update({"epoch": epoch, "domain": domain, "task": task_id,
                                "cell": task_id * 3 + domain,
                                "selected_checkpoint": int(epoch == 0)})
                    if epoch == 7 and task_id == 1 and domain == 2:
                        row["test_auc"] = float("nan")
                    writer.writerow(row)
    _, reasons = validate_run(task)
    assert "metrics_nonfinite_or_missing" in reasons
