import json
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_protocol import (
    ADDON_SEEDS,
    DEVELOPMENT_SEEDS,
    FORMAL_SEEDS,
    all_b1_units,
    config_grid,
    config_id,
    protocol_hash,
    width_for,
)
from stage_F0_queue import (
    QueueFileLock,
    acquire_process_slot,
    build_resource_queue,
    build_search_queue,
    code_hash_inventory,
    build_formal_queue,
    claim_next_entry as _claim_next_entry,
    matrix_ledger,
    select_search_configs,
    write_queue,
)
from stage_F0_worker_control import PRESERVED_H51_RUN_IDS


def claim_next_entry(queue, output_root, **kwargs):
    """Tests that exercise current search claims use the immutable H51 set."""
    if any(str(row.get("phase", "")) == "search" for row in queue.get("entries", [])):
        kwargs.setdefault("held_run_ids", list(PRESERVED_H51_RUN_IDS))
    return _claim_next_entry(queue, output_root, **kwargs)


class _Bundle:
    def split_fingerprints(self):
        return {"train": "train", "validation": "val", "test": "test"}

    def row_id_fingerprints(self):
        return {"train": "train-ids", "validation": "val-ids", "test": "test-ids"}


def _queue():
    bundles = {scope: _Bundle() for scope in
               ("mdmt", "sdmt-d0", "sdmt-d1", "sdmt-d2")}
    return build_search_queue(bundles, "code-hash")


def test_search_queue_full_canonical_coverage_and_user_resource_override(tmp_path):
    queue = _queue()
    assert len(queue) == 288
    assert len({row["run_id"] for row in queue}) == 288
    assert all(row["test_access"] is False for row in queue)
    path = tmp_path / "search.json"
    payload = write_queue(str(path), queue, phase="search")
    loaded = json.loads(path.read_text())
    assert loaded["queue_hash"] == payload["queue_hash"]
    assert loaded["resource_policy"]["max_concurrent_python_processes"] == 5
    assert loaded["resource_policy"]["max_training_processes"] == \
        "min(5, preflight_visible_usable_gpu_count)"
    assert matrix_ledger()["required_totals"]["trajectories"] == 3416


def test_resource_queue_covers_three_backbones_by_two_batch_sizes():
    bundles = {scope: _Bundle() for scope in
               ("mdmt", "sdmt-d0", "sdmt-d1", "sdmt-d2")}
    rows = build_resource_queue(bundles, "code-hash")
    assert len(rows) == 6
    assert {(row["backbone"], row["config"]["batch_size"]) for row in rows} == {
        (model, batch) for model in ("m3oe", "mmoe", "cgc1")
        for batch in (4096, 40960)}


def test_code_hash_inventory_includes_all_reused_model_data_metric_and_test_dependencies():
    inventory = code_hash_inventory(require_tests=True)
    required = {
        "stage_F0_protocol.py", "stage_F0_data.py", "stage_F0_models.py",
        "stage_F0_objectives.py", "stage_F0_queue.py", "stage_F0_run.py",
        "tests/conftest.py", "tests/test_stage_F0_run.py", "stage_S1_data.py",
        "stage_S1_models.py", "stage_S1_protocol.py", "stage_S1_metrics.py",
        "../m3oe/MDMTRec/__init__.py",
        "history_exp/first/p7_models.py", "../m3oe/MDMTRec/basic/activation.py",
        "../m3oe/MDMTRec/basic/features.py", "../m3oe/MDMTRec/basic/initializers.py",
        "../m3oe/MDMTRec/basic/layers.py", "../m3oe/MDMTRec/utils/data.py",
        "../m3oe/MDMTRec/models/multi_domain/model.py",
    }
    assert required.issubset(inventory)
    assert all(len(value) == 64 for value in inventory.values())


def test_search_selection_requires_all_288_and_tie_breaks_numerically():
    rows = []
    for unit in all_b1_units():
        for config in config_grid():
            for seed in DEVELOPMENT_SEEDS:
                rows.append({
                    "unit_key": unit["unit_key"], "status": "VALIDATION_COMPLETE",
                    "seed": seed, **config, "config_id": config_id(config),
                    "selected_validation_macro_auc": 0.75,
                })
    selected = select_search_configs(rows)
    assert len(selected) == 12
    assert all(value["status"] == "SELECTED" for value in selected.values())
    assert all(value["lr"] == 0.001 and value["batch_size"] == 4096
               and value["weight_decay"] == 0.0 for value in selected.values())
    with pytest.raises(ValueError):
        select_search_configs(rows[:-1])


def test_seed_trigger_expands_the_entire_ml_formal_addon_or_none():
    bundles = {scope: _Bundle() for scope in
               ("mdmt", "sdmt-d0", "sdmt-d1", "sdmt-d2")}
    selected = {unit["unit_key"]: {
        "status": "SELECTED", "lr": 0.001, "batch_size": 4096,
        "weight_decay": 0.0, "selection_artifact_hash": "frozen-selection-hash",
    } for unit in all_b1_units()}
    no_trigger = [[0.0] * 4 for _ in range(5)]
    base = build_formal_queue(bundles, selected, "code-hash",
                              primary_validation_deltas=no_trigger)
    assert len(base) == 48
    assert all(row["seed_expansion_trigger_evaluated"] is True
               and row["seed_expansion_trigger"] is False for row in base)

    triggered = [[-0.01, -0.01, 0.01, 0.01]] + [[0.0] * 4 for _ in range(4)]
    expanded = build_formal_queue(bundles, selected, "code-hash",
                                  primary_validation_deltas=triggered)
    assert len(expanded) == 96
    assert len({row["run_id"] for row in expanded}) == 96
    assert {row["seed"] for row in expanded if row["seed_role"] == "formal"} == set(FORMAL_SEEDS)
    assert {row["seed"] for row in expanded if row["seed_role"] == "formal_addon"} == set(ADDON_SEEDS)
    assert {row["unit_key"] for row in expanded if row["seed_role"] == "formal_addon"} == {
        unit["unit_key"] for unit in all_b1_units()}
    assert all(row["seed_expansion_trigger"] is True for row in expanded)

    incomplete = list(expanded[:-1])
    with pytest.raises(ValueError):
        from stage_F0_queue import validate_queue
        validate_queue(incomplete, expected_count=95, phase="formal")


def test_canonical_flock_claims_distinct_runs_and_recovers_checkpoint(tmp_path):
    queue = _queue()
    first, lock1, resume1 = claim_next_entry({"entries": queue[:2]}, str(tmp_path),
                                              gpu_uuid="gpu-a", gpu_index=0)
    assert first["run_id"] == queue[0]["run_id"]
    assert resume1 is None
    second, lock2, _ = claim_next_entry({"entries": queue[:2]}, str(tmp_path),
                                        gpu_uuid="gpu-b", gpu_index=1)
    assert second["run_id"] == queue[1]["run_id"]
    with pytest.raises(BlockingIOError):
        QueueFileLock(os.path.join(tmp_path, "canonical_locks", first["run_id"] + ".lock"))
    lock2.close()
    lock1.close()

    status_file = os.path.join(tmp_path, "queue_status", first["run_id"] + ".json")
    status = json.loads(open(status_file, encoding="utf-8").read())
    status["status"] = "INTERRUPTED"
    with open(status_file, "w", encoding="utf-8") as handle:
        json.dump(status, handle)
    attempt_dir = os.path.join(tmp_path, first["run_id"], "attempts", first["attempt_id"])
    os.makedirs(attempt_dir, exist_ok=True)
    checkpoint = os.path.join(attempt_dir, "resume_epoch_003.pt")
    open(checkpoint, "wb").close()
    not_recovered, no_lock, _ = claim_next_entry({"entries": queue[:1]}, str(tmp_path),
                                                 gpu_uuid="gpu-a", gpu_index=0)
    assert not_recovered is None and no_lock is None
    checkpoint_sha = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
    authorization = {
        "review_verdict": "PASS", "release_id": "reviewed-release-1",
        "approved_by": "principal", "run_id": first["run_id"],
        "canonical_key_json_sha256": hashlib.sha256(
            first["canonical_key_json"].encode("utf-8")).hexdigest(),
        "attempt_id": first["attempt_id"], "recovery_count": 0,
        "resume_checkpoint_sha256": checkpoint_sha,
    }
    recovered, lock3, resume3 = claim_next_entry(
        {"entries": queue[:1]}, str(tmp_path), gpu_uuid="gpu-a", gpu_index=0,
        recovery_authorizations={first["run_id"]: authorization})
    assert recovered["attempt_id"] == first["attempt_id"]
    assert resume3 == checkpoint
    lock3.close()

    with open(status_file, encoding="utf-8") as handle:
        status = json.load(handle)
    status["status"] = "INTERRUPTED"
    with open(status_file, "w", encoding="utf-8") as handle:
        json.dump(status, handle)
    held_entry = {**queue[0], "run_id": PRESERVED_H51_RUN_IDS[0]}
    held, held_lock, _ = claim_next_entry(
        {"entries": [held_entry]}, str(tmp_path), held_run_ids=list(PRESERVED_H51_RUN_IDS),
        recovery_authorizations={first["run_id"]: authorization})
    assert held is None and held_lock is None


def test_search_claim_boundary_rejects_omitted_or_incomplete_h51(tmp_path):
    queue = _queue()
    with pytest.raises(PermissionError, match="H51"):
        _claim_next_entry({"entries": queue[:1]}, str(tmp_path), gpu_uuid="gpu-a")
    with pytest.raises(PermissionError, match="H51"):
        _claim_next_entry({"entries": queue[:1]}, str(tmp_path), gpu_uuid="gpu-a",
                          held_run_ids=list(PRESERVED_H51_RUN_IDS[:-1]))


def test_unrecognized_or_incomplete_history_never_enters_ordinary_claim(tmp_path):
    queue = _queue()
    entry = queue[0]
    status_dir = tmp_path / 'queue_status'
    status_dir.mkdir()
    status_path = status_dir / (entry['run_id'] + '.json')
    for state in ('FAILED', 'UNKNOWN'):
        status_path.write_text(json.dumps({'run_id': entry['run_id'], 'status': state,
                                           'attempt_id': 'attempt-00',
                                           'recovery_count': 0}), encoding='utf-8')
        claimed, lock, resume = claim_next_entry(
            {'entries': [entry]}, str(tmp_path), gpu_uuid='gpu-fixture')
        assert claimed is None and lock is None and resume is None

    status_path.write_text('{malformed-json', encoding='utf-8')
    with pytest.raises(PermissionError, match='unreadable'):
        claim_next_entry({'entries': [entry]}, str(tmp_path), gpu_uuid='gpu-fixture')

    status_path.unlink()
    checkpoint_dir = (tmp_path / entry['run_id'] / 'attempts' / 'attempt-00')
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / 'resume_epoch_001.pt').write_bytes(b'historical checkpoint')
    claimed, lock, resume = claim_next_entry(
        {'entries': [entry]}, str(tmp_path), gpu_uuid='gpu-fixture')
    assert claimed is None and lock is None and resume is None


def test_process_scheduler_enforces_five_live_task_slots(tmp_path):
    leases = [acquire_process_slot(role="test", blocking=False,
                                   scheduler_dir=str(tmp_path))
              for _ in range(5)]
    try:
        with pytest.raises(RuntimeError, match="five"):
            acquire_process_slot(role="sixth", blocking=False,
                                 scheduler_dir=str(tmp_path))
    finally:
        for lease in leases:
            lease.close()


def test_shared_global_hold_blocks_canonical_claim_before_lock_or_status(tmp_path):
    from stage_F0_worker_control import record_global_hold, global_hold_path

    queue = _queue()
    record_global_hold(output_root=str(tmp_path), event={
        "event": "TEST_HOLD", "canonical_run_id": None,
        "reason": "unit-test hold gate",
    })
    with pytest.raises(PermissionError, match="global execution hold"):
        claim_next_entry({"entries": queue[:1]}, str(tmp_path), gpu_uuid="gpu-a")
    assert os.path.isfile(global_hold_path(str(tmp_path)))
    assert not os.path.exists(os.path.join(tmp_path, "queue_status",
                                           queue[0]["run_id"] + ".json"))
