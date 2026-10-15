from __future__ import annotations

import hashlib
import base64
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import stage_F0_postabnormal_v14 as v14
from stage_F0_worker_control import PRESERVED_H49_RUN_IDS, PRESERVED_H49_SHA256


def test_exact_h51_is_h49_plus_both_new_keys_and_rejects_substitutions():
    assert len(v14.H51_RUN_IDS) == 51
    assert set(v14.H51_RUN_IDS) == set(PRESERVED_H49_RUN_IDS) | {
        v14.SIGSEGV_KEY, v14.PEER_KEY
    }
    assert v14.H51_SHA256 == v14.compact_sha(list(v14.H51_RUN_IDS))
    assert PRESERVED_H49_SHA256 == "60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e"
    for malformed in (
        list(v14.H51_RUN_IDS[:-1]),
        list(v14.H51_RUN_IDS) + [v14.H51_RUN_IDS[0]],
        sorted(set(v14.H51_RUN_IDS[:-1]) | {"a" * 20}),
        list(PRESERVED_H49_RUN_IDS),
    ):
        with pytest.raises(PermissionError):
            v14.require_exact_h51(malformed)
    assert v14.require_exact_h51(set(v14.H51_RUN_IDS)) == list(v14.H51_RUN_IDS)
    with pytest.raises(PermissionError):
        v14.require_exact_h51(set(v14.H51_RUN_IDS) | {"outside"})


def _review_record(snapshot, snapshot_sha, *, verdict="PASS"):
    from stage_F0_worker_control import control_hash_inventory
    repo = EXPERIMENTS.parent
    control_hashes = control_hash_inventory(repo)
    return {
        "schema": v14.IMPLEMENTATION_REVIEW_SCHEMA,
        "verdict": verdict,
        "reviewer_role": "independent reviewer",
        "reviewer_identity": "fixture-independent-reviewer",
        "design_sha256": v14.DESIGN_SHA256,
        "source_snapshot_sha256": snapshot_sha,
        "coder_report_path": v14.IMPLEMENTATION_REPORT_REL,
        "coder_report_sha256": v14.sha256_file(repo / v14.IMPLEMENTATION_REPORT_REL),
        "control_hashes": control_hashes,
        "control_hash_inventory_sha256": v14.compact_sha(control_hashes),
        "historical_v13_runtime_sha256": snapshot.get(
            "historical_v13_runtime_sha256", "fixture-historical-graph"),
        "reviewed_implementation_sha256": control_hashes[
            "experiments/stage_F0_postabnormal_v14.py"],
    }


def test_structured_current_review_binds_design_report_control_and_history():
    snapshot = {"historical_v13_runtime_sha256": "historical-fixture"}
    review = _review_record(snapshot, "a" * 64)
    v14._validate_implementation_review(review, snapshot=snapshot,
                                        snapshot_sha256="a" * 64)
    old_review = dict(review)
    old_review.update({"schema": "F0-V13-REVIEW", "coder_report_sha256": "b" * 64})
    with pytest.raises(PermissionError, match="review identity"):
        v14._validate_implementation_review(old_review, snapshot=snapshot,
                                            snapshot_sha256="a" * 64)


def test_v14_gate_refuses_h49_and_same_count_substitution_before_filesystem_reads():
    base = {
        "schema": "F0-EXECUTION-ISOLATION-v1.4",
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256,
    }
    with pytest.raises(PermissionError, match="H51"):
        v14.validate_current_gate(base, runs_root="/no/such/f0/root")
    substituted = dict(base)
    substituted["preserved_held_run_ids"] = list(v14.H51_RUN_IDS[:-1]) + ["f" * 20]
    substituted["preserved_held_run_ids_sha256"] = v14.H51_SHA256
    with pytest.raises(PermissionError, match="H51"):
        v14.validate_current_gate(substituted, runs_root="/no/such/f0/root")


def test_status_projection_changes_only_administrative_fields():
    source = {
        "run_id": v14.SIGSEGV_KEY,
        "status": "RUNNING",
        "attempt_id": "attempt-00",
        "canonical_key_json": "immutable canonical identity",
        "run_id_full_sha256": "1" * 64,
        "recovery_count": 0,
        "resume_checkpoint": None,
        "pid": 123, "supervisor_pid": 124, "start_ticks": 100,
        "opaque_existing_field": {"preserve": True},
    }
    changed = v14.held_status_projection(source, run_id=v14.SIGSEGV_KEY)
    assert changed["status"] == "RECOVERY_BLOCKED"
    assert changed["canonical_key_json"] == source["canonical_key_json"]
    assert changed["run_id_full_sha256"] == source["run_id_full_sha256"]
    assert changed["recovery_count"] == 0 and changed["resume_checkpoint"] is None
    assert changed["opaque_existing_field"] == source["opaque_existing_field"]
    assert changed["retry_authorized"] is False
    assert changed["checkpoint_load_authorized"] is False
    with pytest.raises(PermissionError):
        v14.held_status_projection({**source, "status": "UNCLAIMED"},
                                   run_id=v14.SIGSEGV_KEY)


def test_read_only_source_snapshot_matches_frozen_c97_f140_and_historical_h49():
    current_hold = v14._runs_root() / "execution_control/global_hold.json"
    if v14.sha256_file(current_hold) != v14.EXPECTED_SOURCE_HOLD_SHA256:
        with pytest.raises(PermissionError, match="active source hold differs"):
            v14.source_snapshot()
        return
    snapshot = v14.source_snapshot()
    assert len(snapshot["completed_ids"]) == 97
    assert snapshot["completed_ids_sha256"] == v14.COMPLETED_KEYS_SHA256
    assert len(snapshot["fresh_ids"]) == 140
    assert snapshot["fresh_ids_sha256"] == v14.FRESH_KEYS_SHA256
    assert snapshot["h49_sha256"] == PRESERVED_H49_SHA256
    assert snapshot["h51_sha256"] == v14.H51_SHA256
    assert snapshot["hold_active"] is True
    assert snapshot["launch_authorized"] is False
    v14.validate_source_snapshot(snapshot)
    changed = dict(snapshot)
    changed["fresh_ids"] = list(snapshot["fresh_ids"][:-1])
    with pytest.raises(PermissionError):
        v14.validate_source_snapshot(changed)


def _migration_fixture(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    control = runs / "execution_control"
    statuses = runs / "queue_status"
    control.mkdir(parents=True)
    statuses.mkdir(parents=True)
    source_statuses = {}
    for rid in v14.NEW_INCIDENT_KEYS:
        original_status = "RUNNING" if rid == v14.SIGSEGV_KEY else "INTERRUPTED"
        status = {"run_id": rid, "status": original_status,
                  "canonical_key_json": "frozen", "run_id_full_sha256": "a" * 64,
                  "recovery_count": 0, "resume_checkpoint": None,
                  "attempt_id": "attempt-00"}
        source_statuses[rid] = {"projection": status}
        (statuses / (rid + ".json")).write_text(json.dumps(status), encoding="utf-8")
    hold = {"active": True, "events": [{"event": "historic"}],
            "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
            "preserved_held_run_ids_sha256": PRESERVED_H49_SHA256}
    hold_path = control / "global_hold.json"
    hold_path.write_text(json.dumps(hold), encoding="utf-8")
    monkeypatch.setattr(v14, "EXPECTED_SOURCE_HOLD_SHA256", v14.sha256_file(hold_path))
    monkeypatch.setattr(v14, "EXPECTED_STATUS_SHA256", {
        rid: v14.sha256_file(statuses / (rid + ".json")) for rid in v14.NEW_INCIDENT_KEYS
    })
    snapshot = {
        "schema": "F0-POSTABNORMAL-SOURCE-SNAPSHOT-v1.4",
        "design_sha256": v14.DESIGN_SHA256,
        "historical_v13_runtime_sha256": "fixture-historical-graph",
        "h49_ids": list(PRESERVED_H49_RUN_IDS), "h49_sha256": PRESERVED_H49_SHA256,
        "h51_ids": list(v14.H51_RUN_IDS), "h51_sha256": v14.H51_SHA256,
        "fresh_ids": ["x"], "fresh_ids_sha256": v14.FRESH_KEYS_SHA256,
        "completed_ids": ["c"] * 97, "completed_ids_sha256": v14.COMPLETED_KEYS_SHA256,
        "source_files": [{"path": "execution_control/global_hold.json",
                          "sha256": v14.sha256_file(hold_path),
                          "size": hold_path.stat().st_size,
                          "raw_base64": base64.b64encode(hold_path.read_bytes()).decode()}],
        "incident_statuses": source_statuses,
        "launch_authorized": False, "recovery_authorized": False,
    }
    snapshot["snapshot_sha256"] = v14.compact_sha(snapshot)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    reviewer_path = tmp_path / "review.json"
    reviewer_path.write_text(json.dumps(_review_record(
        snapshot, v14.sha256_file(snapshot_path))), encoding="utf-8")
    reviewer_sha = v14.sha256_file(reviewer_path)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "schema": "F0-EXECUTION-ISOLATION-v1.4",
        "source_snapshot_sha256": v14.sha256_file(snapshot_path),
        "independent_review_sha256": reviewer_sha,
        "independent_review_verdict": "PASS",
        "independent_review_path": str(reviewer_path),
        "launch_authorized": False,
        "preserved_held_run_ids": list(v14.H51_RUN_IDS),
        "preserved_held_run_ids_sha256": v14.H51_SHA256,
    }), encoding="utf-8")
    monkeypatch.setattr(v14, "validate_source_snapshot", lambda *_a, **_kw: None)
    return runs, snapshot_path, gate_path, reviewer_path, reviewer_sha


@pytest.mark.parametrize("fault", [
    "after_prepared",
    "after_status_write_0d2e6555b049c012c0e5",
    "after_status_0d2e6555b049c012c0e5",
    "after_status_a80a232a3c0068a48a08",
    "before_hold_write",
    "after_hold_write",
    "after_hold",
])
def test_admin_migration_failure_is_fail_closed_and_reentry_is_idempotent(
        tmp_path, monkeypatch, fault):
    runs, snapshot, gate, reviewer, reviewer_sha = _migration_fixture(tmp_path, monkeypatch)
    with pytest.raises(OSError):
        v14.apply_administrative_migration(
            runs_root=runs, snapshot_path=snapshot, gate_path=gate,
            reviewer_path=reviewer, reviewer_sha256=reviewer_sha, fault=fault)
    hold_path = runs / "execution_control/global_hold.json"
    assert json.loads(hold_path.read_text())["active"] is True
    result = v14.apply_administrative_migration(
        runs_root=runs, snapshot_path=snapshot, gate_path=gate,
        reviewer_path=reviewer, reviewer_sha256=reviewer_sha)
    assert result["status"] == "COMMITTED"
    assert json.loads(hold_path.read_text())["preserved_held_run_ids"] == list(v14.H51_RUN_IDS)
    for rid in v14.NEW_INCIDENT_KEYS:
        status = json.loads((runs / "queue_status" / (rid + ".json")).read_text())
        assert status["status"] == "RECOVERY_BLOCKED"
        assert status["attempt_id"] == "attempt-00"
        assert status["recovery_count"] == 0 and status["resume_checkpoint"] is None
    ledger = (runs / "execution_control/postabnormal_v14_migration.jsonl").read_text()
    events = [json.loads(line) for line in ledger.splitlines()]
    assert sum(item["event"] == "PREPARED" for item in events) == 1
    assert sum(item["event"] == "COMMITTED" for item in events) == 1
    again = v14.apply_administrative_migration(
        runs_root=runs, snapshot_path=snapshot, gate_path=gate,
        reviewer_path=reviewer, reviewer_sha256=reviewer_sha)
    assert again["idempotent"] is True
    assert len((runs / "execution_control/postabnormal_v14_migration.jsonl").read_text().splitlines()) == len(events)


def test_committed_migration_replay_detects_status_or_hold_drift(tmp_path, monkeypatch):
    runs, snapshot, gate, reviewer, reviewer_sha = _migration_fixture(tmp_path, monkeypatch)
    result = v14.apply_administrative_migration(
        runs_root=runs, snapshot_path=snapshot, gate_path=gate,
        reviewer_path=reviewer, reviewer_sha256=reviewer_sha)
    assert result["status"] == "COMMITTED"
    hold_path = runs / "execution_control/global_hold.json"
    hold = json.loads(hold_path.read_text())
    hold["active"] = False
    hold_path.write_text(json.dumps(hold), encoding="utf-8")
    with pytest.raises(PermissionError, match="committed v1.4 migration"):
        v14.apply_administrative_migration(
            runs_root=runs, snapshot_path=snapshot, gate_path=gate,
            reviewer_path=reviewer, reviewer_sha256=reviewer_sha)


def test_partial_hold_reentry_rejects_historical_event_and_release_timestamp_tampering(
        tmp_path, monkeypatch):
    runs, snapshot, gate, reviewer, reviewer_sha = _migration_fixture(tmp_path, monkeypatch)
    with pytest.raises(OSError, match="after H51 hold projection"):
        v14.apply_administrative_migration(
            runs_root=runs, snapshot_path=snapshot, gate_path=gate,
            reviewer_path=reviewer, reviewer_sha256=reviewer_sha,
            fault="after_hold_write")
    hold_path = runs / "execution_control/global_hold.json"
    hold = json.loads(hold_path.read_text())
    hold["events"][0]["event"] = "tampered-historical-event"
    hold["released_at_utc"] = "2099-01-01T00:00:00Z"
    hold_path.write_text(json.dumps(hold), encoding="utf-8")
    with pytest.raises(PermissionError, match="frozen source pointer changed"):
        v14.apply_administrative_migration(
            runs_root=runs, snapshot_path=snapshot, gate_path=gate,
            reviewer_path=reviewer, reviewer_sha256=reviewer_sha)


def test_missing_production_hold_blocks_claim_even_for_caller_supplied_row(tmp_path, monkeypatch):
    import stage_F0_queue as queue_module
    import stage_F0_worker_control as control_module

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(control_module, "RUNS", root)
    row = {"run_id": "outside_f140_key", "run_id_full_sha256": "0" * 64,
           "canonical_key_json": "{}", "phase": "search"}
    with pytest.raises(PermissionError, match="hold is missing"):
        queue_module.claim_next_entry({"entries": [row]}, str(root),
                                      held_run_ids=v14.H51_RUN_IDS)


def test_release_api_rejects_unbound_or_broad_principal_intent_before_mutation(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="principal release intent"):
        v14.commit_principal_release(
            runs_root=tmp_path / "runs", gate_path=gate,
            principal_identity={"approved_by_role": "principal"})


def test_actual_source_validator_admin_migration_release_and_claim_on_isolated_cpu_clone(
        tmp_path):
    """Keep historical v1.4 evidence read-only and reject it as current authority."""
    import stage_F0_successor_v15 as v15

    repo = EXPERIMENTS.parent
    output = repo / "experiments/stage_F0_outputs"
    source_runs = output / "runs"
    frozen = v15._assert_frozen_artifacts(repo)
    release_rel = "experiments/stage_F0_outputs/runs/execution_control/global_hold_release_v14.json"
    release_ledger_rel = "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_release.jsonl"
    migration_ledger_rel = "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_migration.jsonl"
    snapshot_rel = "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json"
    assert frozen[release_rel] == v15.V14_RELEASE_SHA256
    assert frozen[release_ledger_rel] == v15.V14_RELEASE_LEDGER_SHA256
    assert frozen[migration_ledger_rel] == v15.V14_MIGRATION_LEDGER_SHA256
    assert frozen[snapshot_rel] == v15.V14_SOURCE_SNAPSHOT_SHA256

    release_path = repo / release_rel
    release = v14._read_json(release_path)
    release_events = v15._read_jsonl_exact(repo / release_ledger_rel)
    migration_events = v15._read_jsonl_exact(repo / migration_ledger_rel)
    assert [row.get("event") for row in release_events] == [
        "PREPARED", "RELEASE_ARTIFACT_DURABLE", "HOLD_RELEASED", "COMMITTED"]
    assert release_events[0]["release_id"] == release["release_id"]
    assert release_events[-1]["release_sha256"] == v15.V14_RELEASE_SHA256
    assert release_events[-1]["released_hold_projection_sha256"] == release[
        "released_hold_projection_sha256"]
    assert migration_events[0]["event"] == "PREPARED"
    assert migration_events[-1]["event"] == "COMMITTED"
    assert migration_events[-1]["h51_ids"] == list(v14.H51_RUN_IDS)
    assert migration_events[-1]["h51_sha256"] == v14.H51_SHA256
    assert migration_events[-1]["source_snapshot_sha256"] == v15.V14_SOURCE_SNAPSHOT_SHA256

    # The preserved v1.4 canonical gate is historical raw evidence. Its frozen
    # control inventory predates v1.5 source edits, so the current v1.4
    # validator must fail closed at that exact boundary without changing the
    # clone's gate, ledgers, hold, or queue.
    gate_path = output / "execution_isolation_gate.json"
    gate = v14._read_json(gate_path)
    clone_output = tmp_path / "stage_F0_outputs"
    clone_runs = clone_output / "runs"
    shutil.copytree(source_runs, clone_runs, copy_function=shutil.copy2)
    shutil.copytree(output / "queues", clone_output / "queues", copy_function=shutil.copy2)
    shutil.copy2(output / "run_gate_search.json", clone_output / "run_gate_search.json")
    clone_gate_path = clone_output / "execution_isolation_gate.json"
    shutil.copy2(gate_path, clone_gate_path)
    protected = [
        clone_gate_path,
        clone_output / "queues/F0-B1-ML-search-queue.json",
        clone_runs / "execution_control/global_hold.json",
        clone_runs / "execution_control/global_hold_release_v14.json",
        clone_runs / "execution_control/postabnormal_v14_release.jsonl",
        clone_runs / "execution_control/postabnormal_v14_migration.jsonl",
    ]
    before = [path.read_bytes() for path in protected]
    with pytest.raises(PermissionError, match="current control source inventory is stale"):
        v14.validate_current_gate(gate, runs_root=clone_runs, require_release=False)
    assert [path.read_bytes() for path in protected] == before


def test_current_reheld_graph_and_runner_agree_on_approved_amendment_predicate(
        monkeypatch):
    """Run both real amendment validators read-only against the current graph."""
    import stage_F0_queue as queue_module
    import stage_F0_run as runner
    from stage_F0_protocol import sha256_json
    from stage_F0_worker_control import control_hash_inventory

    effects = []
    def forbidden(*args, **kwargs):
        effects.append("claim-or-data")
        raise AssertionError("validation-only integration must not claim or load data")
    monkeypatch.setattr(runner, "load_b1_bundles", forbidden)
    monkeypatch.setattr(runner, "claim_next_entry", forbidden)

    output = v14._runs_root().parent
    gate = v14._read_json(output / "execution_isolation_gate.json")
    search = v14._read_json(output / "run_gate_search.json")
    runtime_hash = queue_module.code_hash(require_tests=True)
    gate["control_hashes"] = control_hash_inventory(EXPERIMENTS.parent)
    gate["control_hash_inventory_sha256"] = sha256_json(gate["control_hashes"])
    amendment = gate["scientific_code_hash_amendment"]
    amendment.update({"status": "APPROVED",
                      "reviewer_verdict": "PASS",
                      "frozen_scientific_code_hash": search["code_hash"],
                      "runtime_control_code_hash": runtime_hash})
    for verdict in ("PASS", "PASS WITH FIXES"):
        amendment["reviewer_verdict"] = verdict
        runner.validate_runtime_control_amendment(
            gate, frozen_scientific_hash=search["code_hash"],
            runtime_code_hash=runtime_hash)
        with pytest.raises(PermissionError) as rejected:
            v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                      require_release=False)
        assert "approval metadata" not in str(rejected.value)
        assert "runtime control amendment is stale" not in str(rejected.value)
    approved = dict(amendment)
    invalid_metadata = [
        ({key: value for key, value in approved.items() if key != "status"}),
        ({**approved, "status": "PENDING"}),
        ({**approved, "status": "REJECTED"}),
        ({key: value for key, value in approved.items() if key != "reviewer_verdict"}),
        ({**approved, "reviewer_verdict": "PENDING"}),
        ({**approved, "reviewer_verdict": "FAIL"}),
        ({key: value for key, value in approved.items()
          if key not in ("status", "reviewer_verdict")}),
    ]
    for invalid in invalid_metadata:
        gate["scientific_code_hash_amendment"] = invalid
        with pytest.raises(PermissionError, match="approval metadata"):
            v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                      require_release=False)
        with pytest.raises(PermissionError, match="runtime code hash amendment"):
            runner.validate_runtime_control_amendment(
                gate, frozen_scientific_hash=search["code_hash"],
                runtime_code_hash=runtime_hash)
    gate["scientific_code_hash_amendment"] = {
        key: value for key, value in approved.items()
        if key not in ("status", "reviewer_verdict")}
    amendment = gate["scientific_code_hash_amendment"]
    # Correct digest fields alone are insufficient (the exact observed shape).
    amendment["frozen_scientific_code_hash"] = search["code_hash"]
    amendment["runtime_control_code_hash"] = runtime_hash
    with pytest.raises(PermissionError, match="approval metadata"):
        v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                  require_release=False)
    amendment.update(approved)
    amendment["frozen_scientific_code_hash"] = "0" * 64
    with pytest.raises(PermissionError, match="scientific identity"):
        v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                  require_release=False)
    with pytest.raises(PermissionError, match="runtime code hash amendment"):
        runner.validate_runtime_control_amendment(
            gate, frozen_scientific_hash=search["code_hash"],
            runtime_code_hash=runtime_hash)
    amendment.update(approved)
    amendment["runtime_control_code_hash"] = "0" * 64
    with pytest.raises(PermissionError, match="runtime control amendment is stale"):
        v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                  require_release=False)
    amendment["runtime_control_code_hash"] = gate["control_hash_inventory_sha256"]
    with pytest.raises(PermissionError, match="runtime control amendment is stale"):
        v14.validate_current_gate(gate, runs_root=v14._runs_root(),
                                  require_release=False)
    assert effects == []


def test_current_reheld_clone_rejects_old_and_new_release_requests_without_mutation(
        tmp_path):
    """Exercise both committed-ledger rejection paths only in an isolated copy."""
    source_runs = v14._runs_root()
    output = source_runs.parent
    clone_output = tmp_path / "stage_F0_outputs"
    clone_runs = clone_output / "runs"
    shutil.copytree(source_runs, clone_runs, copy_function=shutil.copy2)
    shutil.copy2(output / "execution_isolation_gate.json",
                 clone_output / "execution_isolation_gate.json")
    hold = clone_runs / "execution_control/global_hold.json"
    ledger = clone_runs / "execution_control/postabnormal_v14_release.jsonl"
    release = clone_runs / "execution_control/global_hold_release_v14.json"
    release_pointer = clone_runs / "execution_control/global_hold_release.json"
    gate = clone_output / "execution_isolation_gate.json"
    protected = (hold, ledger, release, release_pointer, gate)
    before = {path: v14.sha256_file(path) for path in protected}
    hold_before = v14._read_json(hold)
    assert hold_before["active"] is True
    assert hold_before["hold_id"] == "cd2dbf50-3a50-4b81-b976-38f1b03991a3"
    old_release = v14._read_json(release)
    identity_fields = ("approved_by_role", "approved_by", "principal_intent",
                       "request_nonce", "approved_at_utc", "authorization")
    old_identity = {field: old_release[field] for field in identity_fields}
    for identity in (old_identity,
                     {**old_identity, "request_nonce": "different-request-nonce-0001"}):
        with pytest.raises(PermissionError):
            v14.commit_principal_release(
                runs_root=clone_runs, gate_path=gate, principal_identity=identity)
        assert {path: v14.sha256_file(path) for path in protected} == before
        assert v14._read_json(hold)["active"] is True
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [event["event"] for event in events] == [
            "PREPARED", "RELEASE_ARTIFACT_DURABLE", "HOLD_RELEASED", "COMMITTED"]


def test_immutable_evidence_refuses_different_overwrite(tmp_path):
    path = tmp_path / "fixed.json"
    v14._create_immutable_json(path, {"v": 1})
    with pytest.raises(FileExistsError):
        v14._create_immutable_json(path, {"v": 2})


def test_matching_immutable_evidence_is_refsynced_and_partial_jsonl_tail_reconciles(
        tmp_path, monkeypatch):
    path = tmp_path / "fixed.json"
    payload = {"v": 1}
    v14._create_immutable_json(path, payload)
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(v14.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    v14._create_immutable_json(path, payload)
    assert len(synced) >= 2  # existing file and its containing directory

    ledger = tmp_path / "txn.jsonl"
    ledger.write_bytes(b'{"event":"PREPARED"}\n{"event":"COM')
    rows = v14._read_jsonl(ledger, repair_partial=True)
    assert rows == [{"event": "PREPARED"}]
    assert ledger.read_bytes() == b'{"event":"PREPARED"}\n'
    v14._append_durable_jsonl(ledger, {"event": "COMMITTED"})
    assert v14._read_jsonl(ledger) == [
        {"event": "PREPARED"}, {"event": "COMMITTED"}]


def test_mutable_clone_pointer_replacement_breaks_same_filesystem_hardlink(tmp_path):
    source = tmp_path / "production-pointer.json"
    clone = tmp_path / "clone-pointer.json"
    source.write_text('{"source":true}\n', encoding="utf-8")
    os.link(source, clone)
    assert os.path.samefile(source, clone)
    v14._atomic_json(clone, {"source": False, "clone": True})
    assert not os.path.samefile(source, clone)
    assert source.read_text(encoding="utf-8") == '{"source":true}\n'
    assert json.loads(clone.read_text(encoding="utf-8")) == {
        "source": False, "clone": True}


def test_migration_partial_hold_reentry_fsyncs_hold_before_durable_record(
        tmp_path, monkeypatch):
    runs, snapshot, gate, reviewer, reviewer_sha = _migration_fixture(tmp_path, monkeypatch)
    with pytest.raises(OSError, match="after H51 hold projection"):
        v14.apply_administrative_migration(runs_root=runs, snapshot_path=snapshot,
            gate_path=gate, reviewer_path=reviewer, reviewer_sha256=reviewer_sha,
            fault="after_hold_write")
    hold_path = runs / "execution_control/global_hold.json"
    seen_fsync = []
    real_fsync_parent = v14._fsync_file_parent
    real_append = v14._append_durable_jsonl

    def track_fsync(path):
        real_fsync_parent(path)
        seen_fsync.append(Path(path))

    def check_before_hold_durable(path, item):
        if item.get("event") == "H51_HOLD_DURABLE":
            assert hold_path in seen_fsync
        return real_append(path, item)

    monkeypatch.setattr(v14, "_fsync_file_parent", track_fsync)
    monkeypatch.setattr(v14, "_append_durable_jsonl", check_before_hold_durable)
    result = v14.apply_administrative_migration(runs_root=runs, snapshot_path=snapshot,
        gate_path=gate, reviewer_path=reviewer, reviewer_sha256=reviewer_sha)
    assert result["status"] == "COMMITTED"


def test_manual_first_epoch_entrypoint_is_disabled():
    import importlib.util

    path = EXPERIMENTS / "stage_F0_outputs/preflight/run_first_search_epoch.py"
    spec = importlib.util.spec_from_file_location("f0_manual_epoch_disabled", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(PermissionError, match="manual first-epoch execution is disabled"):
        module.main()
