"""Regression tests for the approval-gated S2 scheduler boundaries."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S2_orchestrator as orchestrator  # noqa: E402


def _args(tmp_path, tasks=None):
    args = argparse.Namespace(
        raw_root=str(tmp_path / "raw"),
        s1_raw_root=str(tmp_path / "s1"),
        queue_manifest=str(tmp_path / "queue.json"),
        max_workers=2,
        rerun_failed=False,
        python=sys.executable,
        dataset_path="",
    )
    if tasks is not None:
        plan = orchestrator.build_co_run_plan(tasks)
        with open(args.queue_manifest, "w", encoding="utf-8") as handle:
            json.dump({"tasks": list(tasks), "co_run_plan": plan,
                       "co_run_plan_digest": orchestrator.co_run_plan_digest(plan)}, handle)
    return args


def _passing_audit(*args, **kwargs):
    return {
        "expected_count": 64, "observed_count": 64, "valid_count": 64,
        "rejected_count": 0, "all_reuse_ok": True, "rows": [],
    }


def test_approved_queue_fresh_reuse_gate_fails_before_child(monkeypatch, tmp_path):
    tasks = orchestrator.build_tasks(str(tmp_path / "raw"))
    called = {"child": False}

    def rejected_audit(*args, **kwargs):
        return {"expected_count": 64, "observed_count": 64, "valid_count": 63,
                "rejected_count": 1, "all_reuse_ok": False, "rows": []}

    def forbidden_child(*args, **kwargs):
        called["child"] = True
        raise AssertionError("child launch must not happen after failed reuse audit")

    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", rejected_audit)
    monkeypatch.setattr(orchestrator, "_run_one", forbidden_child)
    with pytest.raises(RuntimeError, match="fresh 64-key S1 reuse audit failed"):
        orchestrator.run_formal_queue(tasks, _args(tmp_path, tasks), approved=True)
    assert called["child"] is False


def test_formal_queue_requires_pre_frozen_plan(monkeypatch, tmp_path):
    tasks = orchestrator.build_tasks(str(tmp_path / "raw"))
    args = _args(tmp_path)

    def forbidden_audit(*args, **kwargs):
        raise AssertionError("reuse audit must not substitute for a missing frozen co-run plan")

    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", forbidden_audit)
    with pytest.raises(RuntimeError, match="pre-frozen co-run plan"):
        orchestrator.run_formal_queue(tasks, args, approved=True)


def test_co_run_plan_is_eight_stable_within_setting_pairs():
    tasks = orchestrator.build_tasks("stage_S2_raw_results")
    plan = orchestrator.build_co_run_plan(tasks)
    assert len(plan) == 8
    assert all(len(pair["formal_run_keys"]) == 2 for pair in plan)
    assert all(pair["seed_pair"] in ([2022, 2023], [2024, 2025]) for pair in plan)
    assert all(pair["formal_run_keys"][0].split("|")[1:3] ==
               pair["formal_run_keys"][1].split("|")[1:3] for pair in plan)
    assert all(task.get("co_run_pair_id") and task.get("planned_peer_run_key")
               and task.get("worker_id") for task in tasks)


def test_command_and_child_provenance_fields_are_explicit(tmp_path):
    task = orchestrator.build_tasks(str(tmp_path / "raw"))[0]
    command = orchestrator.command_for(task, _args(tmp_path))
    assert "--co_run_pair_id" in command
    assert "--planned_peer_run_key" in command
    assert "--worker_id" in command
    manifest = {field: task[field] for field in ("co_run_pair_id", "planned_peer_run_key", "worker_id")}
    config = dict(manifest)
    assert orchestrator.validate_co_run_provenance(task, manifest, config) == []
    config["worker_id"] = "wrong-worker"
    assert "co_run_provenance_mismatch:worker_id" in orchestrator.validate_co_run_provenance(
        task, manifest, config)


def test_retained_sibling_stops_instead_of_singleton_recovery(monkeypatch, tmp_path):
    tasks = orchestrator.build_tasks(str(tmp_path / "raw"))
    args = _args(tmp_path, tasks)
    args.rerun_failed = True
    first, second = tasks[0], tasks[1]
    os.makedirs(first["run_dir"], exist_ok=True)
    with open(os.path.join(first["run_dir"], "run_manifest.json"), "w", encoding="utf-8") as handle:
        handle.write("{}")

    def fake_validate(task, require_formal=True):
        return (str(task["formal_run_key"]) == str(first["formal_run_key"]), [])

    called = {"child": False}

    def forbidden_child(*args, **kwargs):
        called["child"] = True
        raise AssertionError("partial frozen pair must stop before child launch")

    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", _passing_audit)
    monkeypatch.setattr(orchestrator, "validate_equal_run", fake_validate)
    monkeypatch.setattr(orchestrator, "_run_one", forbidden_child)
    with pytest.raises(RuntimeError, match="singleton recovery"):
        orchestrator.run_formal_queue(tasks, args, approved=True)
    assert called["child"] is False


def test_existing_cross_setting_or_odd_plan_is_rejected(monkeypatch, tmp_path):
    tasks = orchestrator.build_tasks(str(tmp_path / "raw"))
    plan = orchestrator.build_co_run_plan(tasks)
    plan[0] = dict(plan[0])
    plan[0]["formal_run_keys"] = [plan[0]["formal_run_keys"][0], plan[2]["formal_run_keys"][0]]
    plan[0]["workers"] = []
    queue_path = tmp_path / "cross_setting_queue.json"
    args = _args(tmp_path, tasks)
    args.queue_manifest = str(queue_path)
    queue_path.write_text(json.dumps({
        "co_run_plan": plan,
        "co_run_plan_digest": orchestrator.co_run_plan_digest(plan),
    }), encoding="utf-8")
    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", _passing_audit)
    with pytest.raises(RuntimeError, match="frozen co-run plan"):
        orchestrator.run_formal_queue(tasks, args, approved=True)


def test_audit_cli_honors_custom_s1_root(monkeypatch, tmp_path):
    captured = {}

    def capture_audit(tasks, **kwargs):
        captured["source_root"] = tasks[0]["source_run_dir"]
        return _passing_audit()

    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", capture_audit)
    monkeypatch.setattr(sys, "argv", [
        "stage_S2_orchestrator.py", "--audit_reuse",
        "--s1_raw_root", str(tmp_path / "custom_s1"),
    ])
    assert orchestrator.main() == 0
    assert str(tmp_path / "custom_s1") in captured["source_root"]


def test_failed_pair_manifest_reports_actual_launch_count(monkeypatch, tmp_path):
    tasks = orchestrator.build_tasks(str(tmp_path / "raw"))
    args = _args(tmp_path, tasks)

    def fake_child(task, _args_value, worker_id):
        return {
            "formal_run_key": task["formal_run_key"], "co_run_pair_id": task["co_run_pair_id"],
            "planned_peer_run_key": task["planned_peer_run_key"], "worker_id": worker_id,
            "pid": 1, "start_time": 0.0, "end_time": 0.0, "returncode": 1,
            "validator_valid": False, "validator_reasons": ["synthetic_failure"],
            "run_dir": task["run_dir"], "lock_path": "outside-lock",
        }

    monkeypatch.setattr(orchestrator, "audit_reuse_tasks", _passing_audit)
    monkeypatch.setattr(orchestrator, "_run_one", fake_child)
    result = orchestrator.run_formal_queue(tasks, args, approved=True)
    assert result["resume_state"] == "FAILED"
    assert result["launched"] == 2
    assert result["completed_record_count"] == 2
    assert result["launched"] < result["expected_run_count"]
    queue = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert len(queue["co_run_plan"]) == 8
    assert queue["co_run_plan_digest"]
    assert all(record["co_run_pair_id"] and record["planned_peer_run_key"]
               for record in queue["records"])


def test_formal_code_inventory_includes_actual_m3oe_forward_source():
    assert "m3oe/MDMTRec/models/multi_domain/model.py" in orchestrator.S2_CODE_HASH_FILES
    assert os.path.isfile(orchestrator._code_source_path(
        "m3oe/MDMTRec/models/multi_domain/model.py"))
