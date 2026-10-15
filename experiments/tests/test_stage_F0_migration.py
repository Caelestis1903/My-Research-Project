"""v1.3 administrative migration fixtures; no runner or checkpoint load."""

from __future__ import annotations

import hashlib
import json
import os
import base64
import shutil
import sys
import uuid
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import stage_F0_migration as migration
from stage_F0_worker_control import (
    PRESERVED_H49_RUN_IDS,
    PRESERVED_V12_H47_RUN_IDS,
    held_run_ids_sha256,
)


def _copy(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise AssertionError(f"fixture source must be a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_stat = source.stat()
    target_stat = target.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino):
        raise AssertionError(f"fixture shares source inode: {source}")
    if migration.sha256_file(source) != migration.sha256_file(target):
        raise AssertionError(f"fixture copy differs from source bytes: {source}")


def _fixture_tree(tmp_path: Path, monkeypatch, request=None) -> Path:
    # Mutable incident/control inputs are independent byte copies. The 52
    # completed runs are checked by a read-only per-file source hash/inode
    # verifier to avoid duplicating 4.7 GB for each short failure fixture.
    artifact_root = tmp_path / ("stage_F0_v13_fixture_" + uuid.uuid4().hex)
    if request is not None:
        request.addfinalizer(lambda: shutil.rmtree(artifact_root, ignore_errors=True))
    output = artifact_root / "experiments/stage_F0_outputs"
    runs = output / "runs"
    queue_dir = output / "queues"
    (runs / "queue_status").mkdir(parents=True)
    (runs / "execution_control").mkdir(parents=True)
    queue_dir.mkdir(parents=True)
    (artifact_root / "experiments/queues").mkdir(parents=True, exist_ok=True)
    source_runs = ROOT / "experiments/stage_F0_outputs/runs"
    source_queue = ROOT / "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json"
    _copy(source_queue, queue_dir / source_queue.name)
    rows = migration._load_search_queue(runs)
    source_rows = migration._load_search_queue(source_runs)
    completed_inventory, completed_sha = migration._asset_inventory(source_runs, source_rows)
    source_identities = {}
    for item in completed_inventory:
        relative = str(item["path"]).removeprefix("experiments/stage_F0_outputs/runs/")
        source = source_runs / relative
        stat_value = source.stat()
        source_identities[relative] = (stat_value.st_dev, stat_value.st_ino,
                                       stat_value.st_size, stat_value.st_mtime_ns,
                                       str(item["sha256"]))
    assert len(completed_inventory) == 573
    assert completed_sha == migration.EXPECTED_ASSET_INVENTORY_SHA256

    for run_id in (*PRESERVED_V12_H47_RUN_IDS, *migration.TARGET_IDS,
                   *migration.EXPECTED_COMPLETED_RUN_IDS):
        _copy(source_runs / "queue_status" / f"{run_id}.json",
              runs / "queue_status" / f"{run_id}.json")

    # Copy every frozen incident artifact: both full attempt trees and all
    # v4-referenced native stderr/phase/failure/receipt evidence.
    incident_rows, _ = migration._incident_asset_inventory(source_runs)
    for item in incident_rows:
        relative = str(item["path"]).removeprefix("experiments/stage_F0_outputs/runs/")
        _copy(source_runs / relative, runs / relative)
    _copy(source_runs / "execution_control/global_hold.json",
          runs / "execution_control/global_hold.json")
    _copy(source_runs / migration.V4_PATH, runs / migration.V4_PATH)
    # Production now contains the v1.4 H51 administrative head. These tests
    # exercise the immutable v1.3 graph, so reconstruct its exact PREPARED
    # source bytes from the append-only v1.3 ledger instead of copying the
    # already-migrated production projections into a historical fixture.
    source_ledger = source_runs / migration.MIGRATION_PATH
    prepared_rows = [json.loads(line) for line in source_ledger.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    prepared_rows = [row for row in prepared_rows if row.get("event") == "PREPARED"]
    if len(prepared_rows) != 1:
        raise AssertionError("frozen v1.3 PREPARED source row is missing or duplicated")
    prepared = prepared_rows[0]
    for run_id in migration.TARGET_IDS:
        raw = base64.b64decode(prepared["original_statuses"][run_id]["raw_bytes_base64"])
        (runs / "queue_status" / f"{run_id}.json").write_bytes(raw)
    original_hold = base64.b64decode(
        prepared["original_active_hold"]["raw_bytes_base64"])
    (runs / "execution_control/global_hold.json").write_bytes(original_hold)
    for relative in (migration.DESIGN_PATH, migration.IMPLEMENTATION_REPORT_PATH):
        _copy(ROOT / relative, artifact_root / relative)
    _copy(ROOT / "experiments/reports/stage_F0_signal_cleanup_review_v3.md",
          artifact_root / "experiments/reports/stage_F0_signal_cleanup_review_v3.md")

    def readonly_completed_inventory(_runs_root, _rows, completed_run_ids=None):
        ids = tuple(migration.EXPECTED_COMPLETED_RUN_IDS if completed_run_ids is None
                    else completed_run_ids)
        if (len(ids) != 52 or set(ids) != set(migration.EXPECTED_COMPLETED_RUN_IDS)
                or {str(row.get("run_id", "")) for row in _rows}
                != {str(row.get("run_id", "")) for row in source_rows}):
            raise PermissionError("fixture changed completed-key or search-queue identity")
        for relative, frozen in source_identities.items():
            source = source_runs / relative
            stat_value = source.stat()
            if (source.is_symlink()
                    or (stat_value.st_dev, stat_value.st_ino, stat_value.st_size,
                        stat_value.st_mtime_ns) != frozen[:4]):
                raise PermissionError(f"read-only production asset drift: {relative}")
        return completed_inventory, completed_sha

    monkeypatch.setattr(migration, "_asset_inventory", readonly_completed_inventory)
    asset_rows, asset_sha = migration._asset_inventory(runs, rows)
    incident_rows, incident_sha = migration._incident_asset_inventory(runs)
    assert len(asset_rows) == 573
    assert asset_sha == migration.EXPECTED_ASSET_INVENTORY_SHA256
    assert incident_sha == migration.EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256
    return runs


def _fixture_release(runs: Path, reviewer_verdict="PASS") -> dict:
    artifact_root = migration._artifact_root_for_runs(runs)
    ledger = runs / migration.MIGRATION_PATH
    events, prepared, committed = migration._load_migration_events(ledger)
    hold_path = runs / "execution_control/global_hold.json"
    hold = json.loads(hold_path.read_text(encoding="utf-8"))
    hold["active"] = False
    hold["release_id"] = "fixture-reviewed-release"
    hold_path.write_text(json.dumps(hold, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    hold_sha = migration.sha256_file(hold_path)
    reviewer_path = artifact_root / "experiments/reports/fixture_v13_review.md"
    reviewer_path.parent.mkdir(parents=True, exist_ok=True)
    reviewer_path.write_text("fixture reviewer verdict: PASS WITH FIXES\n", encoding="utf-8")
    reviewer_relative = "experiments/reports/fixture_v13_review.md"
    reviewer_sha = migration.sha256_file(reviewer_path)
    migration._atomic_json(
        artifact_root / "experiments/stage_F0_outputs/execution_isolation_gate.json",
        {
            "schema": "F0-EXECUTION-ISOLATION-v1",
            "reviewer_verdict": "PASS WITH FIXES",
            "reviewer_report_path": reviewer_relative,
            "reviewer_report_sha256": reviewer_sha,
            "v13_protection": {
                "reviewer_report_path": reviewer_relative,
                "reviewer_report_sha256": reviewer_sha,
            },
        })
    queue_path = migration._search_queue_path(runs)
    queue_entries_sha = migration._sha256_bytes(
        migration._compact_json(migration._load_search_queue(runs)))
    completed, completed_sha = migration._asset_inventory(
        runs, migration._load_search_queue(runs))
    incident, incident_sha = migration._incident_asset_inventory(runs)
    release = {
        "schema": migration.RELEASE_SCHEMA,
        "release_id": "fixture-reviewed-release",
        "reviewer_verdict": reviewer_verdict,
        "approved_by": "fixture principal",
        "approved_by_role": "principal",
        "active_hold_released": True,
        "hold_id": hold["hold_id"],
        "hold_sha256": hold_sha,
        "source_active_hold_sha256": committed["active_hold_sha256"],
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS),
        "preserved_held_run_ids_sha256": migration.PRESERVED_H49_SHA256,
        "design_path": migration.DESIGN_PATH,
        "design_sha256": migration.DESIGN_SHA256,
        "implementation_report_path": migration.IMPLEMENTATION_REPORT_PATH,
        "implementation_report_sha256": migration.sha256_file(
            artifact_root / migration.IMPLEMENTATION_REPORT_PATH),
        "reviewer_report_path": reviewer_relative,
        "reviewer_report_sha256": reviewer_sha,
        "original_source_path": migration.V4_PATH,
        "original_source_sha256": migration.V4_SHA256,
        "provenance_correction_path": str(migration.CORRECTION_PATH),
        "provenance_correction_sha256": migration.sha256_file(runs / migration.CORRECTION_PATH),
        "migration_audit_path": str(migration.MIGRATION_PATH),
        "migration_audit_sha256": migration.sha256_file(ledger),
        "search_queue_path": "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json",
        "search_queue_file_sha256": migration.sha256_file(queue_path),
        "search_queue_entries_sha256": queue_entries_sha,
        "completed_asset_inventory_sha256": completed_sha,
        "incident_asset_inventory_sha256": incident_sha,
        "fresh_only_scope": migration.FRESH_ONLY_SCOPE,
        "authorization": dict(migration.MIGRATED_STATUS_AUTHORIZATION),
    }
    release_path = runs / "execution_control/global_hold_release.json"
    release_path.write_text(json.dumps(release, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    return release


@pytest.mark.parametrize("failure", [
    "after_prepared",
    "status0_before_rename", "status0_before_dir_fsync", "after_status0",
    "status1_before_rename", "status1_before_dir_fsync", "after_status1",
    "before_hold_write", "hold_before_rename", "hold_before_dir_fsync",
])
def test_migration_faults_preserve_active_hold_and_never_change_completed_assets(
        tmp_path, monkeypatch, request, failure):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    queue = migration._load_search_queue(runs)
    before_assets, before_hash = migration._asset_inventory(runs, queue)
    hold_path = runs / "execution_control/global_hold.json"
    old_hold_sha = migration.sha256_file(hold_path)
    old_hold_raw = hold_path.read_bytes()
    with pytest.raises(OSError, match="injected failure"):
        migration.apply_migration(runs_root=runs, fault=failure)
    hold = json.loads(hold_path.read_text(encoding="utf-8"))
    assert hold["active"] is True
    if failure == "hold_before_dir_fsync":
        assert hold["preserved_held_run_ids"] == list(PRESERVED_H49_RUN_IDS)
    else:
        assert migration.sha256_file(hold_path) == old_hold_sha
    after_assets, after_hash = migration._asset_inventory(runs, queue)
    assert after_hash == before_hash
    assert after_assets == before_assets
    for run_id in migration.TARGET_IDS:
        status = json.loads((runs / "queue_status" / f"{run_id}.json").read_text())
        assert status["attempt_id"] == "attempt-00"
        assert status["recovery_count"] == 0
        assert status["resume_checkpoint"] is None
        assert status["status"] in ("RUNNING", "INTERRUPTED", "RECOVERY_BLOCKED")
    assert not (runs / migration.TARGET_SIGSEGV / "attempts" / "attempt-01").exists()
    # An authorized administrative retry may only converge the same prepared
    # facts; it cannot create an attempt or touch the active hold state.
    result = migration.apply_migration(runs_root=runs)
    assert result["counts"] == {
        "VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 49, "UNCLAIMED": 187}
    assert json.loads(hold_path.read_text(encoding="utf-8"))["active"] is True
    if failure == "after_prepared":
        events = [json.loads(line) for line in
                  (runs / migration.MIGRATION_PATH).read_text(encoding="utf-8").splitlines()]
        prepared = next(event for event in events if event["event"] == "PREPARED")
        snapshot = prepared["original_active_hold"]
        assert migration.base64.b64decode(snapshot["raw_bytes_base64"], validate=True) == old_hold_raw
        assert snapshot["sha256"] == old_hold_sha


def test_retry_after_second_status_rename_fsyncs_file_and_directory_before_commit(
        tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    with pytest.raises(OSError, match="injected failure before directory fsync"):
        migration.apply_migration(runs_root=runs, fault="status1_before_dir_fsync")

    timeline = []
    original_fsync = os.fsync
    original_append = migration._append_jsonl

    def traced_fsync(fd):
        try:
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            target = Path("<unknown>")
        timeline.append(("fsync", target))
        return original_fsync(fd)

    def traced_append(path, event):
        if event.get("event") in ("STATUS_TRANSITION_DURABLE", "COMMITTED"):
            timeline.append(("append", event.get("event"), event.get("run_id")))
        return original_append(path, event)

    monkeypatch.setattr(os, "fsync", traced_fsync)
    monkeypatch.setattr(migration, "_append_jsonl", traced_append)
    result = migration.apply_migration(runs_root=runs)

    target_status = migration._status_path(runs, migration.TARGET_PEER)
    transition_append = next(i for i, row in enumerate(timeline)
                             if row == ("append", "STATUS_TRANSITION_DURABLE",
                                        migration.TARGET_PEER))
    status_file_fsync = max(i for i, row in enumerate(timeline[:transition_append])
                            if row == ("fsync", target_status))
    status_dir_fsync = max(i for i, row in enumerate(timeline[:transition_append])
                           if row == ("fsync", target_status.parent))
    committed_append = next(i for i, row in enumerate(timeline)
                            if row[0:2] == ("append", "COMMITTED"))
    assert status_file_fsync < status_dir_fsync < transition_append < committed_append
    assert result["counts"] == {
        "VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 49, "UNCLAIMED": 187}


def test_retry_after_hold_rename_fsyncs_hold_file_and_directory_before_commit(
        tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    with pytest.raises(OSError, match="injected failure before directory fsync"):
        migration.apply_migration(runs_root=runs, fault="hold_before_dir_fsync")
    hold_path = runs / "execution_control/global_hold.json"
    hold_before_retry = migration.sha256_file(hold_path)
    assert json.loads(hold_path.read_text(encoding="utf-8"))["active"] is True

    timeline = []
    original_fsync = os.fsync
    original_append = migration._append_jsonl

    def traced_fsync(fd):
        try:
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            target = Path("<unknown>")
        timeline.append(("fsync", target))
        return original_fsync(fd)

    def traced_append(path, event):
        if event.get("event") in ("STATUS_TRANSITION_DURABLE", "COMMITTED"):
            timeline.append(("append", event.get("event"), event.get("run_id")))
        return original_append(path, event)

    monkeypatch.setattr(os, "fsync", traced_fsync)
    monkeypatch.setattr(migration, "_append_jsonl", traced_append)
    migration.apply_migration(runs_root=runs)
    hold_file_fsync = next(i for i, row in enumerate(timeline)
                           if row == ("fsync", hold_path))
    hold_dir_fsync = next(i for i, row in enumerate(timeline)
                          if row == ("fsync", hold_path.parent))
    committed_append = next(i for i, row in enumerate(timeline)
                            if row[0:2] == ("append", "COMMITTED"))
    assert hold_file_fsync < hold_dir_fsync < committed_append
    assert migration.sha256_file(hold_path) == hold_before_retry


def test_successful_admin_migration_sets_exact_h49_without_touching_checkpoint_or_manifest(
        tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    queue = migration._load_search_queue(runs)
    before_assets, before_hash = migration._asset_inventory(runs, queue)
    checkpoint = runs / migration.TARGET_PEER / "attempts/attempt-00/resume_epoch_017.pt"
    manifest = runs / migration.TARGET_PEER / "attempts/attempt-00/run_manifest.json"
    checkpoint_sha = migration.sha256_file(checkpoint)
    manifest_sha = migration.sha256_file(manifest)
    result = migration.apply_migration(runs_root=runs)
    assert result["counts"] == {
        "VALIDATION_COMPLETE": 52, "RECOVERY_BLOCKED": 49, "UNCLAIMED": 187}
    assert result["asset_file_count"] == 573
    assert result["asset_inventory_sha256"] == before_hash
    assert result["migration_sha256"] == migration.sha256_file(runs / migration.MIGRATION_PATH)
    hold = json.loads((runs / "execution_control/global_hold.json").read_text())
    assert hold["active"] is True
    assert hold["preserved_held_run_ids"] == list(PRESERVED_H49_RUN_IDS)
    assert hold["preserved_held_run_ids_sha256"] == held_run_ids_sha256(PRESERVED_H49_RUN_IDS)
    after_assets, after_hash = migration._asset_inventory(runs, queue)
    assert after_assets == before_assets
    assert after_hash == before_hash
    assert migration.sha256_file(checkpoint) == checkpoint_sha
    assert migration.sha256_file(manifest) == manifest_sha
    validated = migration.validate_h49_runtime(
        runs, correction_sha256=result["correction_sha256"],
        migration_sha256=result["migration_sha256"],
        expected_asset_sha256=before_hash)
    assert validated["counts"] == result["counts"]
    assert all(json.loads((runs / "queue_status" / f"{run_id}.json").read_text())[
               "status"] == "RECOVERY_BLOCKED" for run_id in PRESERVED_H49_RUN_IDS)
    # A second administrative invocation is read-only and returns the same
    # committed migration without appending another transaction.
    ledger_sha = migration.sha256_file(runs / migration.MIGRATION_PATH)
    again = migration.apply_migration(runs_root=runs)
    assert again["idempotent"] is True
    assert migration.sha256_file(runs / migration.MIGRATION_PATH) == ledger_sha


@pytest.mark.parametrize("caller_matches", [True, False], ids=["matching", "mismatched"])
def test_apply_migration_binds_caller_frozen_correction_sha(
        tmp_path, monkeypatch, request, caller_matches):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    correction_path, actual_correction_sha = migration.ensure_provenance_correction(runs)
    caller_sha = actual_correction_sha if caller_matches else "0" * 64
    hold_path = runs / "execution_control/global_hold.json"
    hold_before = hold_path.read_bytes()
    status_before = {
        run_id: migration.sha256_file(migration._status_path(runs, run_id))
        for run_id in migration.TARGET_IDS
    }
    migration_path = runs / migration.MIGRATION_PATH

    if caller_matches:
        result = migration.apply_migration(
            runs_root=runs, correction_sha256=caller_sha)
        assert result["correction_sha256"] == actual_correction_sha
        assert migration.sha256_file(correction_path) == caller_sha
        events, _, committed = migration._load_migration_events(migration_path)
        assert events[-1]["event"] == "COMMITTED"
        assert committed["provenance_correction_sha256"] == caller_sha
        assert json.loads(hold_path.read_text(encoding="utf-8"))["active"] is True
    else:
        with pytest.raises(PermissionError, match="differs from caller freeze"):
            migration.apply_migration(runs_root=runs, correction_sha256=caller_sha)
        assert migration.sha256_file(correction_path) == actual_correction_sha
        assert not migration_path.exists()
        assert {
            run_id: migration.sha256_file(migration._status_path(runs, run_id))
            for run_id in migration.TARGET_IDS
        } == status_before
        assert hold_path.read_bytes() == hold_before
        assert json.loads(hold_before)["active"] is True


def test_historical_v13_release_does_not_authorize_current_h51_claims(tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    result = migration.apply_migration(runs_root=runs)
    initial = migration.validate_h49_runtime(
        runs, correction_sha256=result["correction_sha256"],
        migration_sha256=result["migration_sha256"],
        expected_asset_sha256=result["asset_inventory_sha256"],
        expected_incident_asset_sha256=migration.EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256,
        require_active_hold=False)
    assert initial["counts"]["UNCLAIMED"] == 187

    import stage_F0_queue as queue_module
    with pytest.raises(PermissionError, match="global execution hold is active"):
        queue_module.claim_next_entry(
            {"entries": migration._load_search_queue(runs)}, str(runs),
            held_run_ids=PRESERVED_H49_RUN_IDS)

    release = _fixture_release(runs)
    assert migration.validate_h49_runtime(
        runs, correction_sha256=release["provenance_correction_sha256"],
        migration_sha256=release["migration_audit_sha256"],
        expected_asset_sha256=release["completed_asset_inventory_sha256"],
        expected_incident_asset_sha256=release["incident_asset_inventory_sha256"],
        require_active_hold=False)["counts"]["RECOVERY_BLOCKED"] == 49
    queue = {"entries": migration._load_search_queue(runs)}
    # This fixture validates the historical v1.3 release graph only. Its H49
    # release must no longer authorize the current v1.4 claim boundary.
    with pytest.raises(PermissionError, match="exact frozen H51"):
        queue_module.claim_next_entry(
            queue, str(runs), host="fixture", pid=os.getpid(), gpu_uuid="gpu-fixture",
            gpu_index=0, held_run_ids=PRESERVED_H49_RUN_IDS)


def test_reviewed_release_binds_every_required_hash_path_and_scope(tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    migration.apply_migration(runs_root=runs)
    valid = _fixture_release(runs)
    import stage_F0_worker_control as control
    hold_path = runs / "execution_control/global_hold.json"
    hold = json.loads(hold_path.read_text())
    release_path = runs / "execution_control/global_hold_release.json"
    assert control._hold_is_released(hold, runs) is True
    fields = {
        "release_id": "",
        "approved_by": "",
        "hold_id": "substituted-hold",
        "hold_sha256": "0" * 64,
        "source_active_hold_sha256": "0" * 64,
        "design_sha256": "0" * 64,
        "implementation_report_sha256": "0" * 64,
        "reviewer_report_sha256": "0" * 64,
        "original_source_sha256": "0" * 64,
        "provenance_correction_sha256": "0" * 64,
        "migration_audit_sha256": "0" * 64,
        "search_queue_file_sha256": "0" * 64,
        "search_queue_entries_sha256": "0" * 64,
        "completed_asset_inventory_sha256": "0" * 64,
        "incident_asset_inventory_sha256": "0" * 64,
        "design_path": "experiments/reports/substituted.md",
        "implementation_report_path": "experiments/reports/substituted.md",
        "reviewer_report_path": "experiments/reports/substituted.md",
        "original_source_path": "execution_control/substituted.json",
        "provenance_correction_path": "execution_control/substituted.json",
        "migration_audit_path": "execution_control/substituted.json",
        "search_queue_path": "experiments/queues/substituted.json",
        "fresh_only_scope": "all keys",
        "reviewer_verdict": "FAIL",
        "authorization": {"resume": True, "retry": True, "new_attempt": True,
                          "checkpoint_load": True, "formal": True, "test": True},
        "preserved_held_run_ids": list(PRESERVED_H49_RUN_IDS[:-1]),
        "preserved_held_run_ids_sha256": "0" * 64,
        "active_hold_released": False,
    }
    for field, wrong_value in fields.items():
        for malformed in (None, wrong_value):
            candidate = dict(valid)
            if malformed is None:
                candidate.pop(field, None)
            else:
                candidate[field] = malformed
            release_path.write_text(json.dumps(candidate, sort_keys=True) + "\n")
            assert control._hold_is_released(hold, runs) is False, field
    release_path.write_text(json.dumps(valid, sort_keys=True, indent=2) + "\n")
    assert control._hold_is_released(hold, runs) is True

    # A real old PASS cannot replace the current reviewer frozen by the v1.3
    # gate and matching protection record.
    artifact_root = migration._artifact_root_for_runs(runs)
    old_review_rel = "experiments/reports/stage_F0_signal_cleanup_review_v3.md"
    old_review_sha = migration.sha256_file(artifact_root / old_review_rel)
    assert old_review_sha == "a5fe1f766ae969bc9d307e135d73afe1d39cefcb2a6cbf94a1e8133efcc42b96"
    stale_review = dict(valid)
    stale_review["reviewer_report_path"] = old_review_rel
    stale_review["reviewer_report_sha256"] = old_review_sha
    release_path.write_text(json.dumps(stale_review, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    assert control._hold_is_released(hold, runs) is False


def test_migration_refuses_source_drift_before_status_mutation(tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    receipt = runs / migration.PEER_RECEIPT_REL
    original = receipt.read_bytes()
    receipt.write_bytes(original + b" ")
    status_before = {run_id: migration.sha256_file(migration._status_path(runs, run_id))
                     for run_id in migration.TARGET_IDS}
    with pytest.raises(PermissionError, match="peer receipt actual hash"):
        migration.apply_migration(runs_root=runs)
    assert {run_id: migration.sha256_file(migration._status_path(runs, run_id))
            for run_id in migration.TARGET_IDS} == status_before
    assert json.loads((runs / "execution_control/global_hold.json").read_text())["active"] is True


def test_incident_inventory_covers_both_attempts_and_all_v4_evidence(tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    incident_rows, incident_sha = migration._incident_asset_inventory(runs)
    assert incident_sha == migration.EXPECTED_INCIDENT_ASSET_INVENTORY_SHA256
    included = {
        str(row["path"]).removeprefix("experiments/stage_F0_outputs/runs/")
        for row in incident_rows
    }
    required = {
        f"{migration.TARGET_SIGSEGV}/attempts/attempt-00/{name}"
        for name in ("metrics_validation.jsonl", "resource_epoch_metrics.jsonl",
                     "run_manifest.json", "resume_epoch_013.pt")
    }
    required.update({
        f"{migration.TARGET_PEER}/attempts/attempt-00/{name}"
        for name in ("metrics_validation.jsonl", "resource_epoch_metrics.jsonl",
                     "run_manifest.json", "resume_epoch_017.pt")
    })
    v4 = migration._read_json(runs / migration.V4_PATH)
    for worker_key, evidence_keys in (
            ("sigsegv_worker", ("failure_audit_path", "stderr_path", "phase_log_path")),
            ("peer_stopped_worker", ("receipt_path", "stderr_path", "phase_log_path"))):
        for key in evidence_keys:
            required.add(str(v4[worker_key][key]).removeprefix(
                "experiments/stage_F0_outputs/runs/"))
    hold = migration._read_json(runs / "execution_control/global_hold.json")
    peer_failure = [event for event in hold["events"]
                    if event.get("event") == "WORKER_FAILURE_HOLD"
                    and event.get("canonical_run_id") == migration.TARGET_PEER]
    assert len(peer_failure) == 1
    assert peer_failure[0]["nonce"] == v4["peer_stopped_worker"]["nonce"]
    peer_failure_relative = Path(peer_failure[0]["failure_path"]).relative_to(
        ROOT / "experiments/stage_F0_outputs/runs").as_posix()
    assert peer_failure_relative in included
    assert required <= included

    # Mutating the peer's hold-event-bound failure audit must fail before either
    # administrative status can be rewritten.
    drift_path = runs / peer_failure_relative
    before_statuses = {run_id: migration.sha256_file(migration._status_path(runs, run_id))
                       for run_id in migration.TARGET_IDS}
    hold_path = runs / "execution_control/global_hold.json"
    before_hold = migration.sha256_file(hold_path)
    drift_path.write_bytes(drift_path.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="peer durable failure audit"):
        migration.apply_migration(runs_root=runs)
    assert {run_id: migration.sha256_file(migration._status_path(runs, run_id))
            for run_id in migration.TARGET_IDS} == before_statuses
    assert migration.sha256_file(hold_path) == before_hold


@pytest.mark.parametrize("stage", ["prepared_retry", "committed_runtime"])
def test_full_target_projection_drift_is_rejected(tmp_path, monkeypatch, request, stage):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    if stage == "prepared_retry":
        with pytest.raises(OSError, match="after target status 0"):
            migration.apply_migration(runs_root=runs, fault="after_status0")
        run_id = migration.TARGET_SIGSEGV
    else:
        result = migration.apply_migration(runs_root=runs)
        run_id = migration.TARGET_PEER
    status_path = migration._status_path(runs, run_id)
    status = migration._read_json(status_path)
    status["canonical_key_json"] += " "
    migration._atomic_json(status_path, status)
    if stage == "prepared_retry":
        expected = f"status is neither frozen source nor this migration: {run_id}"
        with pytest.raises(PermissionError, match=expected):
            migration.apply_migration(runs_root=runs)
    else:
        with pytest.raises(PermissionError, match="full migrated status projection"):
            migration.validate_h49_runtime(
                runs, correction_sha256=result["correction_sha256"],
                migration_sha256=result["migration_sha256"],
                expected_asset_sha256=result["asset_inventory_sha256"],
                expected_incident_asset_sha256=result["incident_asset_inventory_sha256"])


@pytest.mark.parametrize("mutation", [
    "original_audit_path", "original_audit_sha", "original_audit_raw_value",
    "correction_original_value", "correction_source_path", "correction_source_hash",
    "correction_source_field", "correction_source_value",
])
def test_correction_consumer_rejects_each_forged_provenance_edge(
        tmp_path, monkeypatch, request, mutation):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    result = migration.apply_migration(runs_root=runs)
    correction_path = runs / migration.CORRECTION_PATH
    correction = migration._read_json(correction_path)
    if mutation == "original_audit_path":
        correction["original_audit"]["path"] = "execution_control/substituted.json"
    elif mutation == "original_audit_sha":
        correction["original_audit"]["sha256"] = "0" * 64
    elif mutation == "original_audit_raw_value":
        correction["original_audit"]["raw_value"]["schema"] = "forged"
    elif mutation == "correction_original_value":
        correction["corrections"][0]["original_value"] = "forged"
    elif mutation == "correction_source_path":
        correction["corrections"][0]["source_path"] = "worker_logs/other.receipt.json"
    elif mutation == "correction_source_hash":
        correction["corrections"][0]["source_sha256"] = "0" * 64
    elif mutation == "correction_source_field":
        correction["corrections"][1]["sources"][1]["field"] = "other"
    else:
        correction["corrections"][1]["sources"][1]["value"] = "forged"
    correction_path.write_text(json.dumps(correction, sort_keys=True, indent=2) + "\n",
                               encoding="utf-8")
    with pytest.raises(PermissionError, match="does not bind exact original audit and source fields"):
        migration.validate_h49_runtime(
            runs, correction_sha256=migration.sha256_file(correction_path),
            migration_sha256=result["migration_sha256"],
            expected_asset_sha256=result["asset_inventory_sha256"],
            expected_incident_asset_sha256=result["incident_asset_inventory_sha256"])


def test_runtime_validator_rejects_audit_that_self_substitutes_for_h49(tmp_path, monkeypatch, request):
    runs = _fixture_tree(tmp_path, monkeypatch, request)
    result = migration.apply_migration(runs_root=runs)
    ledger = runs / migration.MIGRATION_PATH
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    malformed = list(PRESERVED_H49_RUN_IDS)
    malformed[malformed.index(migration.TARGET_SIGSEGV)] = "ffffffffffffffffffff"
    malformed.sort()
    for event in events:
        event["held_run_ids"] = malformed
        event["held_run_ids_sha256"] = held_run_ids_sha256(malformed)
    prepared = next(event for event in events if event["event"] == "PREPARED")
    committed = next(event for event in events if event["event"] == "COMMITTED")
    committed["prepared_record_sha256"] = migration._sha256_bytes(
        migration._compact_json(prepared))
    ledger.write_text("".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                               for event in events), encoding="utf-8")

    with pytest.raises(PermissionError, match="migration audit does not freeze the exact independent H49"):
        migration.validate_h49_runtime(
            runs, correction_sha256=result["correction_sha256"],
            migration_sha256=migration.sha256_file(ledger),
            expected_asset_sha256=result["asset_inventory_sha256"])
