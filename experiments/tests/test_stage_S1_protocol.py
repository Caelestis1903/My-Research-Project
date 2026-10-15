"""Stage S1 protocol tests; no dataset or formal training is launched."""

import os
import sys
from collections import Counter

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_orchestrator import build_tasks, command_for
from stage_S1_protocol import (BACKBONES, COMMON_EPOCHS, EXPECTED_RUN_COUNT,
                               FORMAL_PROTOCOL, SEEDS, validate_matrix)


class Args:
    python = "python"
    device = "cuda:0"
    dataset_path = "dataset"
    batch_size = 40960
    epoch = 50
    learning_rate = 0.01
    learning_rate_darts = 0.001
    weight_decay = 1e-5
    expert_num = 4
    allow_protocol_deviation = 0


def test_stage_s1_matrix_is_exactly_192_unique_tasks(tmp_path):
    tasks = build_tasks(str(tmp_path))
    validate_matrix(tasks)
    assert len(tasks) == EXPECTED_RUN_COUNT == 192
    assert len({task["formal_run_key"] for task in tasks}) == 192
    assert Counter(task["run_type"] for task in tasks) == {
        "joint": 48, "single_domain": 144,
    }
    assert Counter(task["seed"] for task in tasks) == {seed: 48 for seed in SEEDS}
    assert {task["backbone"] for task in tasks} == set(BACKBONES)


def test_formal_command_locks_budget_and_no_prohibited_method(tmp_path):
    command = command_for(build_tasks(str(tmp_path))[0], Args())
    joined = " ".join(command)
    assert "--batch_size 40960" in joined
    assert "--epoch 50" in joined
    assert "--learning_rate 0.01" in joined
    assert "curriculum" not in joined.lower()
    assert "momentum" not in joined.lower()
    assert "safe_mtl" not in joined.lower()


def test_common_epoch_protocol_is_frozen():
    assert COMMON_EPOCHS == (9, 19, 29, 39, 49)
    assert FORMAL_PROTOCOL["test_evaluation"] == "deferred_after_selection_freeze"
    assert FORMAL_PROTOCOL["scheduler"] is None

