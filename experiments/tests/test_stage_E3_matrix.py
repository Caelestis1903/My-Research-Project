"""Focused queue/block/report checks for Stage E3 (no dataset or GPU)."""

import json
import os
import sys

from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E3_orchestrator import build_tasks, build_oracle_tasks  # noqa: E402
from stage_E3_report import collect  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_KEYS,
    E3_FORMAL_PROTOCOL,
    ROLE_MAPS,
    compare_alpha_block,
    write_json,
    write_rows_csv,
)


def _make_arm(path: Path, key: str, *, fingerprint="same"):
    path.mkdir(parents=True)
    alpha = ALPHA_GRID[ALPHA_KEYS.index(key)]
    common_manifest = {
        "stage": "E3", "status": "OK", "seed": 2022, "setting": "d0_large",
        "alpha": alpha, "alpha_key": key, "dataset": "MovieLens-1M", "domain_num": 3, "task_num": 2,
        "role_by_domain": {"0": "L", "1": "M", "2": "S"},
        "role_sizes": {"L": 150000, "M": 34500, "S": 3000},
        "dataset_identity": "c:/data/ml-1m",
        "device_identity": "cpu",
        "split_row_fingerprints": {"train": "train", "validation": "validation", "test": "test"},
        "train_domain_counts": {"0": 150000, "1": 34500, "2": 3000},
        "domain_shares": [0.8, 0.184, 0.016],
        "split_counts": {"train": {"0": 1}, "validation": {"0": 1}, "test": {"0": 1}},
        "positive_rates": {"train": {"0": 0.5}, "validation": {"0": 0.5}, "test": {"0": 0.5}},
        "initial_parameter_state_hash": "initial", "partition": {"shared": ["p"]},
        "model_identity": {"class": "MDMTRec"}, "formal_protocol": dict(E3_FORMAL_PROTOCOL),
        "observed_epochs": [0], "formal_valid": False,
    }
    write_json(str(path / "run_manifest.json"), common_manifest)
    config = dict(E3_FORMAL_PROTOCOL)
    config.update({"alpha": alpha, "alpha_key": key, "setting": "d0_large", "seed": 2022,
                   "sampling": E3_FORMAL_PROTOCOL["sampling"], "formal_valid": False,
                   "epoch": 1,
                   "dataset_identity": "c:/data/ml-1m", "device_identity": "cpu",
                   "split_row_fingerprints": common_manifest["split_row_fingerprints"]})
    write_json(str(path / "config.json"), config)
    fields = ["seed", "setting", "alpha", "alpha_key", "epoch", "step", "kind",
              "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "is_probe", "train_loss"]
    write_rows_csv(str(path / "batch_trace.csv"), [{
        "seed": 2022, "setting": "d0_large", "alpha_key": key, "epoch": 0, "step": 0,
        "kind": "train", "batch_fingerprint": fingerprint, "batch_size": 2,
        "domain_counts": "[2,0,0]", "task_counts": "[2,2]", "is_probe": 1, "train_loss": 1.0,
    }, {
        "seed": 2022, "setting": "d0_large", "alpha_key": key, "epoch": 0, "step": -1,
        "kind": "architecture", "batch_fingerprint": fingerprint, "batch_size": 2,
        "domain_counts": "[2,0,0]", "task_counts": "[2,2]", "is_probe": 0, "train_loss": 1.0,
    }], fields)


def test_fixed_joint_and_oracle_cardinality(tmp_path):
    tasks = build_tasks(str(tmp_path / "raw"))
    assert len(tasks) == 60
    assert {task["alpha_key"] for task in tasks} == set(ALPHA_KEYS)
    assert len({(task["alpha_key"], task["setting"], task["seed"]) for task in tasks}) == 60
    oracle_tasks = build_oracle_tasks(str(tmp_path / "oracle"))
    assert len(oracle_tasks) == 36
    assert len({(task["setting"], task["domain"], task["seed"]) for task in oracle_tasks}) == 36


def test_alpha_block_common_support_and_early_short_tail(tmp_path):
    paths = {}
    for key in ALPHA_KEYS:
        path = tmp_path / key
        _make_arm(path, key)
        paths[key] = str(path)
    block = compare_alpha_block(paths, require_formal=False)
    assert block["status"] == "VALID"
    assert block["valid"] is True
    _make_arm(tmp_path / "bad", ALPHA_KEYS[1], fingerprint="bad")
    paths[ALPHA_KEYS[1]] = str(tmp_path / "bad")
    invalid = compare_alpha_block(paths, require_formal=False)
    assert invalid["status"] == "INVALID"
    assert invalid["valid"] is False


def test_alpha_block_rejects_unrecorded_interior_or_tail_support(tmp_path):
    paths = {}
    for key in ALPHA_KEYS:
        path = tmp_path / key
        _make_arm(path, key)
        manifest_path = path / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["observed_epochs"] = [0, 1]
        write_json(str(manifest_path), manifest)
        config_path = path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["epoch"] = 2
        write_json(str(config_path), config)
        paths[key] = str(path)
    invalid = compare_alpha_block(paths, require_formal=False)
    assert invalid["status"] == "INVALID"
    assert any(item.get("reason") == "trace_epoch_support_mismatch"
               for item in invalid["mismatches"])


def test_report_always_enumerates_all_60_paths(tmp_path):
    rows, manifest = collect(str(tmp_path / "raw"))
    assert rows == []
    assert manifest["expected_run_count"] == 60
    assert manifest["missing_run_count"] == 60
    assert len(manifest["runs"]) == 60
