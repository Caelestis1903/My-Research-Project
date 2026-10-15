import os
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_data import split_hashes
from stage_F0_protocol import canonical_queue_entry, protocol_hash, width_for
from stage_F0_queue import QueueFileLock
from stage_F0_run import (F0Runner, build_global_order, frozen_selected_checkpoint_identity,
                          gpu_memory_preflight, release_selected_test, resource_preflight_record,
                          stable_state_hash, validate_runtime_control_amendment)


@pytest.mark.parametrize("free_bytes, expected", [
    (101, True),  # 89.9% used: strict threshold permits it.
    (100, False), # 90% used is a NO-GO.
    (0, False),   # 100% used is a NO-GO.
])
def test_gpu_memory_preflight_applies_strict_per_device_threshold(free_bytes, expected):
    record = gpu_memory_preflight({"gpus": [{"uuid": "GPU-fixture",
                                              "total_memory_bytes": 1000,
                                              "free_memory_bytes": free_bytes}],
                                   "nvidia_smi": {"available": False}})
    assert record["devices"][0]["uuid"] == "GPU-fixture"
    assert record["devices"][0]["eligible"] is expected
    assert bool(record["eligible_gpu_indices"]) is expected


def test_gpu_memory_preflight_fails_closed_when_memory_is_unreadable():
    record = gpu_memory_preflight({"gpus": [{"uuid": "GPU-fixture",
                                              "total_memory_bytes": 1000,
                                              "free_memory_bytes": None}],
                                   "nvidia_smi": {"available": False}})
    assert record["go"] is False
    assert record["eligible_gpu_indices"] == []
    assert record["devices"][0]["readable"] is False
from stage_S1_data import DataBundle, FeatureSchema


def test_runtime_amendment_uses_28_file_code_hash_separate_from_control_inventory():
    from stage_F0_queue import code_hash
    from stage_F0_worker_control import control_hash_inventory
    from stage_F0_protocol import sha256_json

    runtime_hash = code_hash(require_tests=True)
    control_hash = sha256_json(control_hash_inventory())
    assert runtime_hash != control_hash
    amendment = {
        'status': 'APPROVED', 'reviewer_verdict': 'PASS WITH FIXES',
        'frozen_scientific_code_hash': 'frozen-scientific-identity',
        'runtime_control_code_hash': runtime_hash,
    }
    gate = {'scientific_code_hash_amendment': amendment}
    for verdict in ('PASS', 'PASS WITH FIXES'):
        amendment['reviewer_verdict'] = verdict
        validate_runtime_control_amendment(
            gate, frozen_scientific_hash='frozen-scientific-identity',
            runtime_code_hash=runtime_hash)

    invalid_cases = [
        ({key: value for key, value in amendment.items() if key != 'status'},
         'status missing'),
        ({**amendment, 'status': 'PENDING'}, 'status pending'),
        ({**amendment, 'status': 'REJECTED'}, 'status rejected'),
        ({key: value for key, value in amendment.items() if key != 'reviewer_verdict'},
         'verdict missing'),
        ({**amendment, 'reviewer_verdict': 'PENDING'}, 'verdict pending'),
        ({**amendment, 'reviewer_verdict': 'FAIL'}, 'verdict fail'),
        ({**amendment, 'frozen_scientific_code_hash': '0' * 64}, 'wrong frozen digest'),
        ({**amendment, 'runtime_control_code_hash': '0' * 64}, 'wrong runtime digest'),
        ({**amendment, 'runtime_control_code_hash': control_hash},
         'inventory digest is not runtime digest'),
    ]
    # The exact observed two-field amendment is also rejected despite matching
    # both digests: approval metadata is part of the runner contract.
    invalid_cases.append(({
        'frozen_scientific_code_hash': 'frozen-scientific-identity',
        'runtime_control_code_hash': runtime_hash,
    }, 'observed two-field amendment'))
    for invalid, _case in invalid_cases:
        with pytest.raises(PermissionError, match='runtime code hash amendment'):
            validate_runtime_control_amendment(
                {'scientific_code_hash_amendment': invalid},
                frozen_scientific_hash='frozen-scientific-identity',
                runtime_code_hash=runtime_hash)


def test_real_runner_entrypoint_uses_gate_default_and_stops_before_claim(tmp_path):
    from stage_F0_worker_control import (PRESERVED_H51_RUN_IDS,
                                         control_hash_inventory, _sha256_json)

    root = os.path.dirname(HERE)
    wrapper = os.path.join(HERE, 'stage_F0_outputs', 'runs',
                           'run_f0_python_task.sh')
    gate_path = tmp_path / 'pending-execution-gate.json'
    control_hashes = control_hash_inventory()
    gate_path.write_text(json.dumps({
        'schema': 'F0-EXECUTION-ISOLATION-v1.4',
        'control_hashes': control_hashes,
        'control_hash_inventory_sha256': _sha256_json(control_hashes),
        'preserved_held_run_ids': list(PRESERVED_H51_RUN_IDS),
        'reviewer_verdict': 'PENDING',
    }), encoding='utf-8')
    queue_path = tmp_path / 'must-not-be-read-queue.json'
    output_root = tmp_path / 'must-not-be-created'
    result = subprocess.run([
        wrapper, 'f0-run-entrypoint-regression', sys.executable,
        os.path.join(HERE, 'stage_F0_run.py'),
        '--queue', str(queue_path), '--run-id', 'safe-gate-probe',
        '--output-root', str(output_root), '--run-gate', str(tmp_path / 'search.json'),
        '--execution-isolation-gate', str(gate_path),
    ], cwd=root, capture_output=True, text=True, timeout=30)

    assert result.returncode != 0
    assert ('v1.4 gate H51 digest mismatch' in result.stderr
            or 'v1.4 source snapshot is missing or changed' in result.stderr
            or 'new execution-isolation reviewer verdict is not PASS' in result.stderr
            or 'v1.3 provenance/status/asset protection binding is missing' in result.stderr)
    assert 'NameError' not in result.stderr
    assert 'must-not-be-read-queue.json' not in result.stderr
    assert not output_root.exists()


def _synthetic_bundle():
    columns = ("user_id", "movie_id", "gender", "age", "occupation", "zip",
               "cate_id", "domain_indicator")
    sparse = ("user_id", "movie_id", "gender", "occupation", "zip",
              "cate_id", "domain_indicator")
    features = FeatureSchema(
        dataset="movielens", columns=columns, dense_columns=("age",),
        sparse_columns=sparse,
        vocab_sizes={name: (3 if name == "domain_indicator" else 8) for name in sparse},
    )
    def make_x(count, offset):
        x = np.zeros((count, len(columns)), dtype=np.float32)
        for col in range(len(columns)):
            if col == 3:
                x[:, col] = np.linspace(0.0, 1.0, count, dtype=np.float32)
            elif col != 7:
                x[:, col] = (np.arange(count) + col + offset) % 4
        return x

    train_x = make_x(24, 0)
    val_x = make_x(8, 7)
    test_x = make_x(4, 11)
    train_y = (np.arange(48).reshape(24, 2) % 2).astype(np.float32)
    val_y = np.tile(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (2, 1))
    test_y = np.tile(np.asarray([[0, 0], [1, 1]], dtype=np.float32), (2, 1))
    return DataBundle(
        dataset="movielens", setting="natural", seed=2022, features=features,
        train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        train_row_ids=np.arange(24, dtype=np.int64),
        val_row_ids=np.arange(100, 108, dtype=np.int64),
        test_row_ids=np.arange(200, 204, dtype=np.int64),
        domain_num=1, task_num=2, local_to_physical=(0,),
        preprocessing_metadata={"global_base_train_size": 24,
                                "raw_file_sha256": "synthetic-dataset-hash"},
    )


def _synthetic_mdmt_bundle():
    from copy import deepcopy
    bundle = _synthetic_bundle()

    def expand(x_values, y_values, id_start):
        base_x = np.asarray(x_values[:4], dtype=np.float32)
        base_y = np.asarray(bundle.val_y[:4] if len(y_values) == 8 else y_values[:4],
                            dtype=np.float32)
        out_x = np.tile(base_x, (3, 1))
        out_y = np.tile(base_y, (3, 1))
        out_x[:, -1] = np.repeat(np.arange(3, dtype=np.float32), 4)
        out_ids = np.arange(id_start, id_start + 12, dtype=np.int64)
        return out_x, out_y, out_ids

    result = deepcopy(bundle)
    result.train_x, result.train_y, result.train_row_ids = expand(bundle.train_x, bundle.train_y, 0)
    result.val_x, result.val_y, result.val_row_ids = expand(bundle.val_x, bundle.val_y, 100)
    result.test_x, result.test_y, result.test_row_ids = expand(bundle.test_x, bundle.test_y, 200)
    result.domain_num = 3
    result.local_to_physical = (0, 1, 2)
    result.preprocessing_metadata["global_base_train_size"] = 12
    return result


def _entry(bundle, phase="search", max_epochs=2):
    seed_role = "development" if phase == "search" else "formal"
    seed = 2018 if phase == "search" else 2022
    entry = canonical_queue_entry(
        phase=phase, intent=phase, model="mmoe", scope="sdmt-d0",
        seed_role=seed_role, seed=seed,
        width={"c": 8, "b": 32, "a": 0, "non_embedding": 0},
        split_hashes=split_hashes(bundle), protocol_digest=protocol_hash(),
        code_digest="unit-test-code",
        config={"lr": 0.001, "batch_size": 4096, "weight_decay": 0.0},
    )
    entry["max_epochs"] = max_epochs
    entry["test_access"] = False
    return entry


def _runner(bundle, entry, path):
    return F0Runner(entry, bundle, str(path), device="cpu",
                    code_digest="unit-test-code", protocol_digest=protocol_hash(),
                    enforce_cuda=False, test_mode=True)


def test_global_natural_permutation_is_seeded_and_scope_filterable():
    selected = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    first = build_global_order(10, selected, 2022, 0)
    second = build_global_order(10, selected, 2022, 0)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == set(selected.tolist())
    assert len(set(first.tolist())) == len(selected)


def test_run_preserves_gpu_uuid_mapping_in_both_lock_owners(tmp_path, monkeypatch):
    bundle = _synthetic_bundle()
    entry = _entry(bundle)
    attempt_dir = tmp_path / entry["run_id"] / "attempts" / "attempt-00"
    runner = _runner(bundle, entry, attempt_dir)
    # Exercise the CUDA identity path without allocating a CUDA context or
    # running an epoch; this validates only lock-owner metadata propagation.
    runner.device = torch.device("cuda:2")
    runner.selected_score = 0.5
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid="GPU-test-uuid-2"))

    canonical_lock = QueueFileLock(str(tmp_path / "canonical_locks" /
                                        (entry["run_id"] + ".lock")))
    gpu_lock = QueueFileLock(str(tmp_path / "gpu_locks" / "gpu2.lock"))
    result = runner.run(max_epochs=0, canonical_lock=canonical_lock, gpu_lock=gpu_lock)
    assert result["status"] == "PARTIAL_RESOURCE_TRAJECTORY"

    canonical_owner = json.loads((tmp_path / "canonical_locks" /
                                  (entry["run_id"] + ".lock")).read_text())
    gpu_owner = json.loads((tmp_path / "gpu_locks" / "gpu2.lock").read_text())
    assert canonical_owner["gpu_index"] == gpu_owner["gpu_index"] == 2
    assert canonical_owner["gpu_uuid"] == gpu_owner["gpu_uuid"] == "GPU-test-uuid-2"


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    bundle = _synthetic_bundle()
    entry = _entry(bundle)
    full_dir = tmp_path / "full" / entry["run_id"] / "attempts" / "attempt-00"
    full = _runner(bundle, entry, full_dir)
    full_result = full.run(max_epochs=2)
    assert full_result["status"] == "VALIDATION_COMPLETE"

    partial_dir = tmp_path / "resumed" / entry["run_id"] / "attempts" / "attempt-00"
    partial = _runner(bundle, entry, partial_dir)
    partial_result = partial.run(max_epochs=1)
    assert partial_result["status"] == "PARTIAL_RESOURCE_TRAJECTORY"
    resume_path = partial_dir / "resume_epoch_001.pt"
    assert resume_path.is_file()

    resumed = _runner(bundle, entry, partial_dir)
    resumed_result = resumed.run(resume_from=str(resume_path), max_epochs=2)
    assert resumed_result["status"] == "VALIDATION_COMPLETE"
    full_state = torch.load(full_dir / "final_checkpoint.pt", map_location="cpu",
                            weights_only=False)
    resumed_state = torch.load(partial_dir / "final_checkpoint.pt", map_location="cpu",
                               weights_only=False)
    assert stable_state_hash(full_state["model_state"]) == \
        stable_state_hash(resumed_state["model_state"])
    assert stable_state_hash(full_state["main_optimizer_state"]) == \
        stable_state_hash(resumed_state["main_optimizer_state"])
    def without_timing(rows):
        resource_fields = {"epoch_wall_seconds", "examples_per_second", "cpu_peak_rss_bytes",
                           "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"}
        return [{key: value for key, value in row.items() if key not in resource_fields}
                for row in rows]
    assert without_timing(full_state["metrics_rows"]) == \
        without_timing(resumed_state["metrics_rows"])
    resource_rows = [json.loads(line) for line in
                     (full_dir / "resource_epoch_metrics.jsonl").read_text().splitlines()]
    assert len(resource_rows) == 2
    required_resource_fields = {
        "epoch_wall_seconds", "examples_per_second", "cpu_peak_rss_bytes",
        "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
        "storage_free_bytes_after_commit", "resume_checkpoint_bytes",
        "resource_measurement_status",
    }
    assert required_resource_fields.issubset(resource_rows[0])
    assert resource_rows[0]["resource_measurement_status"] == "MEASURED"
    assert resource_rows[0]["examples_per_second"] > 0
    assert resource_rows[0]["resume_checkpoint_bytes"] > 0


def test_resource_preflight_schema_has_measurement_fields_and_pending_values():
    bundle = _synthetic_bundle()
    entry = canonical_queue_entry(
        phase="resource", intent="resource", model="mmoe", scope="mdmt",
        seed_role="resource_preflight", seed=2018,
        width=width_for("mmoe", "mdmt"), split_hashes=split_hashes(bundle),
        protocol_digest=protocol_hash(), code_digest="unit-test-code",
        config={"lr": 0.001, "batch_size": 4096, "weight_decay": 0.0})
    runtime = {"ram": {"total_bytes": 1000, "available_bytes": 500},
               "disk": {"free_bytes": 900, "total_bytes": 1000},
               "gpus": [{"index": 0, "name": "test-gpu", "free_memory_bytes": 100}],
               "resource_concurrency_policy": {"max_concurrent_python_processes": 5}}
    row = resource_preflight_record(entry, bundle, runtime)
    required = {"epoch_wall_seconds", "examples_per_second", "cpu_ram_total_bytes",
                "cpu_peak_rss_bytes", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
                "storage_free_bytes", "checkpoint_bytes", "measurement_status",
                "backbone", "batch_size"}
    assert required.issubset(row)
    assert row["measurement_status"] == "PENDING_FIRST_SEARCH_EPOCH"
    assert row["epoch_wall_seconds"] is None
    assert row["examples_per_second"] is None
    assert row["gpu_peak_allocated_bytes"] is None
    assert row["storage_free_bytes"] == 900
    assert row["backbone"] == "mmoe" and row["batch_size"] == 4096


def test_selected_test_api_requires_the_frozen_selected_checkpoint_identity(tmp_path):
    bundle = _synthetic_bundle()
    entry = canonical_queue_entry(
        phase="formal", intent="formal", model="mmoe", scope="sdmt-d0",
        seed_role="formal", seed=2022, width=width_for("mmoe", "sdmt-d0"),
        split_hashes=split_hashes(bundle), protocol_digest=protocol_hash(),
        code_digest="unit-test-code",
        config={"lr": 0.001, "batch_size": 4096, "weight_decay": 0.0})
    entry["test_access"] = False
    runner = F0Runner(entry, bundle, str(tmp_path / "source"), device="cpu",
                      code_digest="unit-test-code", protocol_digest=protocol_hash(),
                      enforce_cuda=False, test_mode=True)
    runner.selected_epoch = 0
    runner.selected_score = 0.75
    runner.validation_history = [{"epoch": 0, "macro_auc": 0.75,
                                  "macro_logloss": 0.69}]
    checkpoint_path = tmp_path / "run" / "selected_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    selected_payload = runner._state_payload(1, None)
    selected_payload["snapshot_role"] = "selected"
    torch.save(selected_payload, checkpoint_path)
    identity = frozen_selected_checkpoint_identity(entry, str(checkpoint_path))
    gate = {
        "status": "TEST_RELEASED", "all_search_complete": True,
        "formal_first_four_complete": True, "validation_expansion_decision_frozen": True,
        "p_trigger_decision_frozen": True, "protocol_hash": entry["protocol_hash"],
        "code_hash": entry["code_hash"],
        "dataset_file_sha256": bundle.preprocessing_metadata["raw_file_sha256"],
        "split_hashes": entry["split_hashes"],
        "selected_checkpoint_identities": {entry["run_id"]: identity},
    }
    result = release_selected_test(
        bundle, release_gate=gate, entry=entry, output_dir=str(tmp_path / "test"),
        selected_checkpoint_path=str(checkpoint_path),
        selected_checkpoint_identity=identity, device="cpu")
    assert result["selected_checkpoint_identity_hash"]
    assert result["selected_checkpoint_sha256"] == identity["selected_checkpoint_sha256"]
    assert (tmp_path / "test" / "metrics_test_selected.json").is_file()

    wrong_identity = dict(identity)
    wrong_identity["selected_checkpoint_sha256"] = "0" * 64
    wrong_gate = {**gate, "selected_checkpoint_identities": {entry["run_id"]: wrong_identity}}
    rejected_output = tmp_path / "rejected"
    with pytest.raises((PermissionError, ValueError)):
        release_selected_test(
            bundle, release_gate=wrong_gate, entry=entry, output_dir=str(rejected_output),
            selected_checkpoint_path=str(checkpoint_path),
            selected_checkpoint_identity=wrong_identity, device="cpu")
    assert not rejected_output.exists()


@pytest.mark.parametrize("commit_point", [
    "after_epoch_commit",
    "after_metrics_projection",
    "after_resume_prune",
    "after_selected_projection",
    "after_fixed_projection",
    "after_resource_projection",
    "after_manifest_projection",
])
def test_formal_checkpoint_commit_points_recover_selected_and_fixed_snapshots(
        tmp_path, commit_point):
    bundle = _synthetic_bundle()
    entry = _entry(bundle, phase="formal", max_epochs=10)
    baseline_dir = tmp_path / "baseline" / entry["run_id"] / "attempts" / "attempt-00"
    baseline = _runner(bundle, entry, baseline_dir)
    def increasing_validation_score():
        return {"macro_auc": 0.6 + 0.01 * baseline.completed_epochs,
                "macro_logloss": 0.7, "cells": {}}
    baseline._eval_validation = increasing_validation_score
    baseline_result = baseline.run(max_epochs=10)
    assert baseline_result["status"] == "VALIDATION_COMPLETE"
    assert baseline_result["routing_probe_status"] == "COMPLETE"
    assert (baseline_dir / "routing_probe_selected.json").is_file()
    assert (baseline_dir / "routing_probe_final.json").is_file()
    probe_audit = json.loads((baseline_dir / "routing_probe_noop_audit.json").read_text())
    assert probe_audit["runner_state_before_and_after_all_probes_identical"] is True
    for role in ("selected", "final"):
        audit = probe_audit["no_op_state_audit"][role]
        assert audit["complete_state_unchanged_after_each_probe"] is True
        assert audit["repeated_probe_outputs_identical"] is True
        assert audit["state_before"] == audit["state_after_first_probe"]
        assert audit["state_before"] == audit["state_after_repeat_probe"]
        assert set(probe_audit["snapshots"][role]["train"]["cells"]) == {"d0t0", "d0t1"}
        assert set(probe_audit["snapshots"][role]["validation"]["cells"]) == {"d0t0", "d0t1"}

    interrupted_dir = tmp_path / commit_point / entry["run_id"] / "attempts" / "attempt-00"
    interrupted = _runner(bundle, entry, interrupted_dir)
    def interrupted_increasing_validation_score():
        return {"macro_auc": 0.6 + 0.01 * interrupted.completed_epochs,
                "macro_logloss": 0.7, "cells": {}}
    interrupted._eval_validation = interrupted_increasing_validation_score

    class InjectedCrash(RuntimeError):
        pass

    def inject(point):
        if point == commit_point and interrupted.completed_epochs == 10:
            raise InjectedCrash(commit_point)

    interrupted.fault_injector = inject
    with pytest.raises(InjectedCrash, match=commit_point):
        interrupted.run(max_epochs=10)
    assert (interrupted_dir / "resume_epoch_010.pt").is_file()

    resumed = _runner(bundle, entry, interrupted_dir)
    resumed_result = resumed.run(
        resume_from=str(interrupted_dir / "resume_epoch_010.pt"), max_epochs=10)
    assert resumed_result["status"] == "VALIDATION_COMPLETE"

    for filename in ("selected_checkpoint.pt", "checkpoint_epoch_09.pt"):
        baseline_state = torch.load(baseline_dir / filename, map_location="cpu",
                                    weights_only=False)
        recovered_state = torch.load(interrupted_dir / filename, map_location="cpu",
                                     weights_only=False)
        assert baseline_state["selected_epoch"] == recovered_state["selected_epoch"]
        assert baseline_state["completed_epochs"] == recovered_state["completed_epochs"]
        assert baseline_state["model_state_hash"] == recovered_state["model_state_hash"]
        assert baseline_state["main_optimizer_state_hash"] == \
            recovered_state["main_optimizer_state_hash"]
        assert baseline_state["architecture_optimizer_state_hash"] == \
            recovered_state["architecture_optimizer_state_hash"]
    selected_baseline = torch.load(baseline_dir / "selected_checkpoint.pt", map_location="cpu",
                                   weights_only=False)
    selected_recovered = torch.load(interrupted_dir / "selected_checkpoint.pt", map_location="cpu",
                                    weights_only=False)
    assert selected_baseline["selected_epoch"] == selected_recovered["selected_epoch"] == 9
    assert selected_recovered["snapshot_role"] == "selected"
    assert len((interrupted_dir / "metrics_validation.jsonl").read_text().splitlines()) == 10


@pytest.mark.parametrize("backbone", ["m3oe", "mmoe", "cgc1"])
def test_formal_routing_probe_supports_all_backbones_without_state_changes(backbone, tmp_path):
    bundle = _synthetic_bundle()
    entry = _entry(bundle)
    entry["backbone"] = backbone
    entry["model_width"] = {"c": 8, "b": 32, "a": 64 if backbone == "m3oe" else 0,
                            "non_embedding": 0}
    # Recompute canonical identity because backbone is part of the queue key.
    entry = canonical_queue_entry(
        phase="formal", intent="formal", model=backbone, scope="sdmt-d0",
        seed_role="formal", seed=2022, width=entry["model_width"],
        split_hashes=split_hashes(bundle), protocol_digest=protocol_hash(),
        code_digest="unit-test-code",
        config={"lr": 0.001, "batch_size": 4096, "weight_decay": 0.0})
    entry["max_epochs"] = 2
    entry["test_access"] = False
    runner = _runner(bundle, entry, tmp_path / backbone)
    before = runner._diagnostic_state(0)
    first = runner._routing_probe_split("validation", runner.val_x, runner.val_ids)
    second = runner._routing_probe_split("validation", runner.val_x, runner.val_ids)
    after = runner._diagnostic_state(0)
    assert before == after
    assert first == second
    assert set(first["cells"]) == {"d0t0", "d0t1"}
    assert first["probe_cap_per_cell"] == 1024
    assert all(cell["expert_output_pairwise_cosine"]["pair_count"] > 0
               for cell in first["cells"].values())
    assert all(value["calls"] == 1 and value["rows"] == first["sampled_row_count"]
               for value in first["dense_execution_counts"].values())
    if backbone == "m3oe":
        assert 0.0 < first["fusion_coefficients"]["alpha_domain"] < 1.0
        assert first["fusion_coefficients"]["beta_domain"] is None


@pytest.mark.parametrize("backbone", ["m3oe", "mmoe", "cgc1"])
def test_mdmt_routing_probe_selects_each_physical_domain(backbone, tmp_path):
    bundle = _synthetic_mdmt_bundle()
    width = {"c": 8, "b": 32, "a": 64 if backbone == "m3oe" else 0,
             "non_embedding": 0}
    entry = canonical_queue_entry(
        phase="formal", intent="formal", model=backbone, scope="mdmt",
        seed_role="formal", seed=2022, width=width,
        split_hashes=split_hashes(bundle), protocol_digest=protocol_hash(),
        code_digest="unit-test-code",
        config={"lr": 0.001, "batch_size": 4096, "weight_decay": 0.0})
    entry["max_epochs"] = 2
    entry["test_access"] = False
    runner = _runner(bundle, entry, tmp_path / (backbone + "-mdmt"))
    result = runner._routing_probe_split("validation", runner.val_x, runner.val_ids)
    assert set(result["cells"]) == {
        "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1"}
    assert set(result["selected_probe_row_ids_by_physical_domain"]) == {"d0", "d1", "d2"}
    assert all(value["probe_row_count"] == 4 for value in result["cells"].values())
