import csv
import json
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_E4_orchestrator as orchestrator  # noqa: E402
import stage_E4_report as report  # noqa: E402
from stage_E4_orchestrator import JOINT_REQUIRED_ARTIFACTS  # noqa: E402
from stage_E4_weighting import EXPECTED_SEEDS, WEIGHTING_KEYS  # noqa: E402


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_joint_fixture(root, *, formal=True):
    for weighting in WEIGHTING_KEYS:
        for seed in EXPECTED_SEEDS:
            run_dir = root / weighting / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "run_manifest.json", {
                "stage": "E4", "status": "OK", "weighting": weighting,
                "seed": seed, "formal_valid": formal, "best_epoch": 1,
            })
            _write_json(run_dir / "config.json", {"formal_valid": formal})
            for name in JOINT_REQUIRED_ARTIFACTS:
                target = run_dir / name
                if target.name not in {"run_manifest.json", "config.json"}:
                    target.touch()
            with (run_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "weighting", "selected_epoch", "selected_checkpoint",
                    "best_test_auc", "best_test_logloss",
                ])
                writer.writeheader()
                writer.writerow({
                    "weighting": weighting, "selected_epoch": 1,
                    "selected_checkpoint": 1, "best_test_auc": 0.7,
                    "best_test_logloss": 0.4,
                })


def test_joint_queue_keeps_pending_until_three_arms_and_finishes(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    # The real comparator must classify an empty seed block as pending.
    empty_dirs = {key: str(raw_root / key / "seed2022") for key in WEIGHTING_KEYS}
    assert orchestrator.compare_joint_block(empty_dirs, SimpleNamespace()).get("status") == "PENDING"
    partial = raw_root / "equal" / "seed2022"
    partial.mkdir(parents=True, exist_ok=True)
    _write_json(partial / "run_manifest.json", {"stage": "E4"})
    _write_json(partial / "config.json", {})
    assert orchestrator.compare_joint_block(empty_dirs, SimpleNamespace()).get("status") == "PENDING"

    def fake_process(command, **_kwargs):
        run_dir = tmp_path / "raw"
        run_dir = run_dir / command[command.index("--weighting") + 1]
        run_dir = run_dir / f"seed{command[command.index('--seed') + 1]}"
        _write_json(run_dir / "run_manifest.json", {"stage": "E4", "status": "OK"})
        _write_json(run_dir / "config.json", {"formal_valid": True})
        for name in JOINT_REQUIRED_ARTIFACTS:
            target = run_dir / name
            if name not in {"run_manifest.json", "config.json", "stdout.log"}:
                if name == "metrics_summary.csv":
                    with target.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "weighting", "selected_epoch", "selected_checkpoint",
                                "common_support",
                            ],
                        )
                        writer.writeheader()
                        writer.writerow({
                            "weighting": command[command.index("--weighting") + 1],
                            "selected_epoch": 1,
                            "selected_checkpoint": 1,
                            "common_support": "",
                        })
                else:
                    target.touch()
        return SimpleNamespace(returncode=0)

    def fake_compare(run_dirs, _bundle, **_kwargs):
        present = sum(os.path.exists(os.path.join(path, "run_manifest.json"))
                      for path in run_dirs.values())
        if present < len(WEIGHTING_KEYS):
            return {"status": "PENDING", "valid": None, "common_epochs": []}
        return {
            "status": "VALID", "valid": True, "common_epochs": [0, 1],
            "common_epoch_count": 2, "common_trace_count": 4,
            "selected_epoch_in_common_support": {key: True for key in WEIGHTING_KEYS},
        }

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_process)
    monkeypatch.setattr(orchestrator, "validate_joint_artifact", lambda *_args, **_kwargs: (True, []))
    monkeypatch.setattr(orchestrator, "compare_joint_block", fake_compare)
    original_set_block_status = orchestrator._set_block_status
    snapshots = []

    def recording_set_block_status(task, root, block):
        original_set_block_status(task, root, block)
        states = {}
        for weighting in WEIGHTING_KEYS:
            summary_path = raw_root / weighting / f"seed{task['seed']}" / "metrics_summary.csv"
            if not summary_path.exists():
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                states[weighting] = next(csv.DictReader(handle))
        snapshots.append((task["weighting"], task["seed"], block["status"], states))

    monkeypatch.setattr(orchestrator, "_set_block_status", recording_set_block_status)
    args = SimpleNamespace(
        raw_root=str(raw_root), master=str(tmp_path / "master.csv"), python=sys.executable,
        dataset_path="dataset.csv", oracle_root=str(tmp_path / "oracle"), device="cpu",
        batch_size=40960, epoch=50, learning_rate=1e-2, learning_rate_darts=1e-3,
        weight_decay=1e-5, expert_num=4, earlystop_patience=10, probe_interval=10,
        instrument=1,
    )
    result = orchestrator._run_joint_queue(
        orchestrator.build_tasks(str(raw_root)), args, SimpleNamespace())
    assert result == 0
    seed2022 = [item for item in snapshots if int(item[1]) == 2022]
    assert [item[0] for item in seed2022] == ["equal", "sqrt", "sample"]
    assert seed2022[0][2] == "PENDING"
    assert seed2022[0][3]["equal"]["block_status"] == "PENDING"
    assert seed2022[0][3]["equal"]["common_support"] == ""
    assert seed2022[1][2] == "PENDING"
    assert seed2022[1][3]["equal"]["block_validity"] == "PENDING"
    assert seed2022[1][3]["sqrt"]["block_status"] == "PENDING"
    assert seed2022[2][2] == "VALID"
    assert all(seed2022[2][3][weighting]["block_validity"] == "VALID"
               for weighting in WEIGHTING_KEYS)
    assert all(seed2022[2][3][weighting]["common_support"] == "2"
               for weighting in WEIGHTING_KEYS)
    manifests = [
        json.loads((raw_root / weighting / f"seed{seed}" / "run_manifest.json").read_text())
        for weighting in WEIGHTING_KEYS for seed in EXPECTED_SEEDS
    ]
    assert all(item.get("block_status") == "VALID" for item in manifests)
    with (tmp_path / "master.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(row["status"] == "OK" for row in rows)
    assert not any(row["status"] == "INVALID_BLOCK" for row in rows)


def _patch_report_inputs(monkeypatch, block):
    monkeypatch.setattr(report, "load_processed_dataset",
                        lambda *_args, **_kwargs: SimpleNamespace(dataset_identity="tiny"))
    monkeypatch.setattr(report, "validate_oracle_artifact",
                        lambda *_args, **_kwargs: (False, ["run_directory_missing"]))
    monkeypatch.setattr(report, "validate_joint_artifact",
                        lambda *_args, **_kwargs: (True, []))
    monkeypatch.setattr(report, "compare_joint_block",
                        lambda *_args, **_kwargs: dict(block))


def test_report_downgrades_cross_arm_invalid_block(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    _write_joint_fixture(raw_root, formal=True)
    _patch_report_inputs(monkeypatch, {
        "status": "INVALID", "valid": False, "mismatches": [{"field": "batch_fingerprint"}],
        "common_epochs": [0, 1], "common_epoch_count": 2, "common_trace_count": 4,
        "selected_epoch_in_common_support": {key: True for key in WEIGHTING_KEYS},
    })
    rows, manifest = report.collect(str(raw_root), str(raw_root / "oracle_reference"),
                                    "tiny.csv", formal=True)
    assert manifest["joint_formal_valid_count"] == 0
    assert all(item["status"] == "INVALID_BLOCK" for item in manifest["joint"])
    assert all(row["artifact_validity"] == "INVALID_BLOCK" for row in rows)
    assert all(row["block_validity"] == "INVALID" for row in rows)
    assert all(row["common_support"] == 2 for row in rows)
    run_report = tmp_path / "run_report.md"
    report.write_run_report(str(run_report), manifest)
    report_text = run_report.read_text(encoding="utf-8")
    assert "INVALID_BLOCK" in report_text
    assert "2/4" in report_text


def test_report_propagates_selected_checkpoint_and_common_support(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    _write_joint_fixture(raw_root, formal=True)
    _patch_report_inputs(monkeypatch, {
        "status": "VALID", "valid": True, "mismatches": [],
        "common_epochs": [0, 1], "common_epoch_count": 2, "common_trace_count": 4,
        "selected_epoch_in_common_support": {key: True for key in WEIGHTING_KEYS},
    })
    rows, manifest = report.collect(str(raw_root), str(raw_root / "oracle_reference"),
                                    "tiny.csv", formal=True)
    assert manifest["joint_formal_valid_count"] == 12
    assert all(item["status"] == "FORMAL_VALID" for item in manifest["joint"])
    assert all(row["selected_checkpoint"] == "1" for row in rows)
    assert all(row["selected_epoch_in_common_support"] == 1 for row in rows)
    assert all(row["common_support"] == 2 for row in rows)
    assert all(row["common_epoch_count"] == 2 for row in rows)
    assert all(row["common_trace_count"] == 4 for row in rows)
