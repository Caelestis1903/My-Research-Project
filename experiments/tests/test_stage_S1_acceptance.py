"""Focused Stage S1 acceptance tests that do not launch formal training."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_orchestrator import (BATCH_FIELDS, METRICS_EPOCH_FIELDS, REQUIRED_FILES,
                                   METRICS_TEST_FIELDS, archive_failed_attempt,
                                   assert_formal_launcher, audit_pairing_fingerprints,
                                   build_tasks, classify_task_state,
                                   expected_train_domain_counts,
                                   finalize_artifact_hashes,
                                   required_artifact_hash_names,
                                   validate_artifact_hash_inventory,
                                   validate_raw_prediction_metrics)
from stage_S1_protocol import canonical_dataset
from stage_S1_metrics import LOGLOSS_EPS, cell_metrics, safe_auc, safe_logloss, task_domain_cell_index
from stage_S1_report import (_safe_logloss, contradiction_rows,
                             difficulty_summary_rows, prediction_aggregate_rows)


def test_formal_launcher_rejects_every_deviation():
    class Args:
        batch_size = 40960
        epoch = 50
        learning_rate = 0.01
        learning_rate_darts = 0.001
        weight_decay = 1e-5
        expert_num = 99
        shadow_patience = 10
        device = "cuda:0"
        num_workers = 0
        fcn_dims = [512, 256, 256, 64]
        allow_protocol_deviation = 1
        allow_nonformal_seed = 0
    with pytest.raises(ValueError):
        assert_formal_launcher(Args())


def test_retry_archives_invalid_canonical_and_keeps_path(tmp_path):
    task = build_tasks(str(tmp_path))[0]
    os.makedirs(task["run_dir"])
    with open(os.path.join(task["run_dir"], "run_manifest.json"), "w", encoding="utf-8") as handle:
        handle.write("not-json")
    _, pending, blocked = classify_task_state([task], rerun_failed=True)
    assert not blocked and len(pending) == 1
    assert pending[0]["run_dir"] == task["run_dir"]
    first = os.path.join(task["run_dir"], "failed_attempts", "attempt_001", "run_manifest.json")
    assert os.path.isfile(first)
    with open(task["run_dir"] + "/run_manifest.json", "w", encoding="utf-8") as handle:
        handle.write("second-invalid-attempt")
    _, pending, blocked = classify_task_state([task], rerun_failed=True)
    second = os.path.join(task["run_dir"], "failed_attempts", "attempt_002", "run_manifest.json")
    assert not blocked and len(pending) == 1
    assert os.path.isfile(first) and os.path.isfile(second)
    with open(first, encoding="utf-8") as handle:
        assert handle.read() == "not-json"
    with open(second, encoding="utf-8") as handle:
        assert handle.read() == "second-invalid-attempt"


def test_exact_schema_contract_has_no_test_in_epoch_history():
    assert len(set(METRICS_EPOCH_FIELDS)) == len(METRICS_EPOCH_FIELDS)
    assert len(set(METRICS_TEST_FIELDS)) == len(METRICS_TEST_FIELDS)
    assert len(set(BATCH_FIELDS)) == len(BATCH_FIELDS)
    assert not any(field.startswith("test_") for field in METRICS_EPOCH_FIELDS)
    assert "predictions_test.npz" not in METRICS_EPOCH_FIELDS


def _write_prediction(path, physical, offset):
    row_ids = np.arange(8, dtype=np.int64) + physical * 100
    labels = np.stack([np.arange(8) % 2, (np.arange(8) + 1) % 2], axis=1).astype(np.float32)
    pred = np.clip(0.15 + 0.7 * labels + offset, 1e-3, 1 - 1e-3).astype(np.float32)
    payload = {"row_ids": row_ids, "physical_domain": np.full(8, physical, dtype=np.int64), "labels": labels}
    for checkpoint in ("selected", "final", "common_epoch_9", "common_epoch_19",
                       "common_epoch_29", "common_epoch_39", "common_epoch_49"):
        payload["pred_" + checkpoint] = pred
    np.savez_compressed(os.path.join(path, "predictions_test.npz"), **payload)


def test_prediction_aggregates_keep_tasks_separate_and_assemble_references(tmp_path):
    rows = []
    for physical in (0, 1, 2):
        run_dir = tmp_path / ("ref{}".format(physical))
        run_dir.mkdir()
        _write_prediction(str(run_dir), physical, 0.0)
        for task in (0, 1):
            rows.append({"formal_run_key": "single_domain|x|natural|mmoe|{}|2022".format(physical),
                         "dataset": "kuairand", "setting": "natural", "backbone": "mmoe",
                         "seed": 2022, "run_type": "single_domain", "physical_domain": physical,
                         "task": task, "task_identity": "ref_task{}".format(task),
                         "run_dir": str(run_dir)})
    joint_dir = tmp_path / "joint"
    joint_dir.mkdir()
    # The synthetic joint row IDs must match each physical reference; use one
    # concatenated payload to exercise assembled row alignment.
    ids = np.concatenate([np.arange(8) + d * 100 for d in (0, 1, 2)])
    labels_one = np.stack([np.arange(24) % 2, (np.arange(24) + 1) % 2], axis=1).astype(np.float32)
    pred = np.clip(0.2 + 0.6 * labels_one, 1e-3, 1 - 1e-3).astype(np.float32)
    payload = {"row_ids": ids, "physical_domain": np.repeat(np.arange(3), 8), "labels": labels_one}
    for checkpoint in ("selected", "final", "common_epoch_9", "common_epoch_19",
                       "common_epoch_29", "common_epoch_39", "common_epoch_49"):
        payload["pred_" + checkpoint] = pred
    np.savez_compressed(joint_dir / "predictions_test.npz", **payload)
    for task in (0, 1):
        rows.append({"formal_run_key": "joint|x|natural|mmoe|2022", "dataset": "kuairand",
                     "setting": "natural", "backbone": "mmoe", "seed": 2022,
                     "run_type": "joint", "physical_domain": "", "task": task,
                     "task_identity": "joint_task{}".format(task), "run_dir": str(joint_dir)})
    aggregates = prediction_aggregate_rows(rows)
    joint_selected = [row for row in aggregates if row["run_type"] == "joint" and row["checkpoint"] == "selected"]
    assert {row["task"] for row in joint_selected} == {0, 1}
    assert {row["task_identity"] for row in joint_selected} == {"joint_task0", "joint_task1"}
    assert all(row["assembled_reference_pooled_auc"] != "" for row in joint_selected)


def test_tie_helper_and_cuda_smoke_are_executable_when_torch_exists():
    torch = pytest.importorskip("torch")
    from stage_S1_run import select_checkpoint_epoch
    assert select_checkpoint_epoch({0: 0.3, 1: 0.3, 2: 0.2}) == 0


def _raw_metric_fixture():
    labels = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    prediction = np.asarray([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    payload = {"row_ids": np.arange(4, dtype=np.int64),
               "physical_domain": np.full(4, 2, dtype=np.int64), "labels": labels}
    checkpoints = ("selected", "final", "common_epoch_9", "common_epoch_19",
                   "common_epoch_29", "common_epoch_39", "common_epoch_49")
    for checkpoint in checkpoints:
        payload["pred_" + checkpoint] = prediction.copy()
    manifest = {
        "domain_num": 1, "local_to_physical": [2], "physical_domain": 2,
        "physical_domain_identity": ["tab4"], "domain_shares": {"0": 1.0},
        "preprocessing_metadata": {"label_definitions": {"task0": "alpha", "task1": "beta"}},
    }
    rows, summaries = [], []
    for checkpoint in checkpoints:
        for task in (0, 1):
            metrics = cell_metrics(labels[:, task], prediction[:, task])
            row = {"checkpoint": checkpoint, "domain": 0, "task": task,
                   "cell": task_domain_cell_index(task, 0, 1), "physical_domain": 2,
                   "physical_domain_identity": "tab4", "domain_share": 1.0,
                   "task_identity": "alpha" if task == 0 else "beta",
                   "test_sample_count": metrics["sample_count"],
                   "test_positive_count": metrics["positive_count"],
                   "test_positive_rate": metrics["positive_rate"],
                   "auc": metrics["auc"], "logloss": metrics["logloss"]}
            rows.append(row)
            if checkpoint == "selected":
                summaries.append(dict(row))
    return payload, manifest, rows, summaries


def test_shared_metric_and_task_major_cell_index_fixture():
    labels = [0, 1, 0, 1]
    scores = [1.0, 0.0, 1.0, 0.0]
    assert LOGLOSS_EPS == 1e-12
    assert _safe_logloss is safe_logloss
    clipped = np.clip(np.asarray(scores, dtype=np.float64), LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    expected = -np.mean(np.asarray(labels) * np.log(clipped)
                        + (1 - np.asarray(labels)) * np.log(1 - clipped))
    assert safe_logloss(labels, scores) == pytest.approx(expected, rel=0, abs=1e-12)
    assert safe_auc(labels, scores) == pytest.approx(0.0)
    assert task_domain_cell_index(0, 0, 3) == 0
    assert task_domain_cell_index(1, 0, 3) == 3
    assert task_domain_cell_index(1, 2, 3) == 5


def test_validator_recomputes_every_raw_cell_and_rejects_corruption():
    payload, manifest, rows, summaries = _raw_metric_fixture()
    assert validate_raw_prediction_metrics(payload, manifest,
                                           {"run_type": "single_domain"},
                                           rows, summaries) == []
    corrupted_rows = [dict(row) for row in rows]
    corrupted_rows[0]["auc"] = float(corrupted_rows[0]["auc"]) + 0.01
    assert validate_raw_prediction_metrics(payload, manifest,
                                           {"run_type": "single_domain"},
                                           corrupted_rows, summaries)
    corrupted_payload = dict(payload)
    corrupted_payload["pred_selected"] = np.asarray(payload["pred_selected"]).copy()
    corrupted_payload["pred_selected"][0, 0] = 0.99
    assert validate_raw_prediction_metrics(corrupted_payload, manifest,
                                           {"run_type": "single_domain"},
                                           rows, summaries)


def test_logging_on_off_equivalence():
    torch = pytest.importorskip("torch")
    from stage_S1_run import run_instrumentation_equivalence_fixture
    result = run_instrumentation_equivalence_fixture(torch.device("cuda:0")
                                                     if torch.cuda.is_available() else torch.device("cpu"))
    assert result["parameter_equal"] and result["optimizer_equal"]
    assert result["metrics_equal"] and result["rng_equal"]
    assert result["batch_trajectory_equal"]
    assert result["on_log_rows"] > 0 and result["off_log_rows"] == 0


def test_pairing_negative_and_exact_count_contracts():
    joint = {"physical_view_fingerprints": {"1": {split: "same-" + split
                                                    for split in ("train", "validation", "test")}}}
    reference = {"split_row_fingerprints": {split: "same-" + split
                                             for split in ("train", "validation", "test")},
                 "preprocessing_metadata": {"pairing_fingerprints": {
                     split: {"joint_remapped": "same-" + split, "reference": "same-" + split}
                     for split in ("train", "validation", "test")}}}
    assert audit_pairing_fingerprints(joint, reference, 1)[0]
    broken = json.loads(json.dumps(reference))
    broken["split_row_fingerprints"]["test"] = "corrupted"
    assert not audit_pairing_fingerprints(joint, broken, 1)[0]
    assert expected_train_domain_counts({"dataset": "movielens", "setting": "d0_large",
                                         "run_type": "joint"}) == {"0": 150000, "1": 34500, "2": 3000}
    assert expected_train_domain_counts({"dataset": "kuairand", "setting": "natural",
                                         "run_type": "single_domain", "physical_domain": 2}) == {"0": 89452}
    names = required_artifact_hash_names({"architecture_optimizer_artifact": "architecture_optimizer_state.pth"})
    assert "optimizer_state.pth" in names and "run_manifest.json" not in names


def _paired_fixture_row(dataset, setting, physical, role, delta, checkpoint="selected",
                        seed=2022, task=0, gap=0.01):
    return {"dataset": dataset, "setting": setting, "backbone": "mmoe",
            "physical_domain": physical, "task": task, "checkpoint": checkpoint,
            "seed": seed, "size_role": role, "delta_auc": delta,
            "gap_logloss": gap, "raw_auc_nt": int(delta < 0),
            "raw_logloss_nt": int(gap > 0), "practical_auc_nt": int(delta <= -0.003),
            "practical_logloss_nt": int(gap >= 0.005),
            "concordant_practical_nt": int(delta <= -0.003 and gap > 0),
            "concordant_strong_nt": int(delta <= -0.005 and gap >= 0.005)}


def test_contradiction_axes_require_matched_dataset_strata():
    # Negative dataset fixture: the opposing signs are only across
    # MovieLens rotations (different roles); KuaiRand matches one role but
    # has the same sign.  This must not be reported as a dataset flip.
    rotation_only = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01),
        _paired_fixture_row("movielens", "d1_large", 0, "S", 0.01),
        _paired_fixture_row("kuairand", "natural", 0, "M", 0.02),
    ]
    assert not any(row["comparison_axis"] == "dataset"
                   for row in contradiction_rows(rotation_only))

    # Positive cross-dataset fixture: same physical/task/backbone/checkpoint,
    # and the same frozen size role, with opposite dataset signs.
    cross_dataset = [
        _paired_fixture_row("movielens", "d1_large", 1, "L", -0.01),
        _paired_fixture_row("kuairand", "natural", 1, "L", 0.01),
    ]
    assert any(row["comparison_axis"] == "dataset"
               for row in contradiction_rows(cross_dataset))

    # Positive rotation and training-stage fixtures are independent axes.
    rotation = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01),
        _paired_fixture_row("movielens", "d1_large", 0, "S", 0.01),
    ]
    assert any(row["comparison_axis"] == "rotation"
               for row in contradiction_rows(rotation))
    rotation_stable = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01),
        _paired_fixture_row("movielens", "d1_large", 0, "S", -0.02),
    ]
    assert not any(row["comparison_axis"] == "rotation"
                   for row in contradiction_rows(rotation_stable))
    stage = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01, "selected"),
        _paired_fixture_row("movielens", "d0_large", 0, "L", 0.01, "common_epoch_9"),
    ]
    assert any(row["comparison_axis"] == "stage"
               for row in contradiction_rows(stage))
    stage_stable = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01, "selected"),
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.02, "common_epoch_9"),
    ]
    assert not any(row["comparison_axis"] == "stage"
                   for row in contradiction_rows(stage_stable))

    # All matched strata with the same sign must remain contradiction-free.
    stable = [
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01),
        _paired_fixture_row("kuairand", "natural", 0, "L", -0.02),
        _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01, "common_epoch_9"),
    ]
    axes = {row["comparison_axis"] for row in contradiction_rows(stable)}
    assert not ({"dataset", "rotation", "stage"} & axes)


def test_difficulty_summary_uses_one_validation_split():
    row = _paired_fixture_row("movielens", "d0_large", 0, "L", -0.01)
    row.update({"reference_validation_difficulty": 0.71,
                "reference_validation_difficulty_logloss": 0.42,
                # This deliberately differs and must not enter the difficulty summary.
                "reference_logloss": 9.99})
    summary = difficulty_summary_rows([row])
    assert len(summary) == 1
    assert summary[0]["difficulty_evaluation_split"] == "validation"
    assert summary[0]["reference_validation_auc_mean"] == 0.71
    assert summary[0]["reference_validation_logloss_mean"] == 0.42
    assert "reference_test_logloss_mean" not in summary[0]


def test_artifact_hash_corruption_is_rejected(tmp_path):
    import hashlib
    names = [name for name in REQUIRED_FILES if name != "run_manifest.json"]
    hashes = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"stage-s1-artifact")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"architecture_optimizer_artifact": "",
                "artifact_hashes": hashes,
                "artifacts": ["run_manifest.json"] + names}
    assert validate_artifact_hash_inventory(str(tmp_path), manifest) == []
    (tmp_path / "stdout.log").write_bytes(b"corrupted")
    assert any("artifact_hash_mismatch:stdout.log" == reason
               for reason in validate_artifact_hash_inventory(str(tmp_path), manifest))
    self_referential = dict(manifest)
    self_referential["artifact_hashes"] = dict(hashes)
    self_referential["artifact_hashes"]["run_manifest.json"] = "0" * 64
    assert "artifact_hash_self_reference" in validate_artifact_hash_inventory(
        str(tmp_path), self_referential)


def test_formal_launcher_identity_and_stdout_write_order_regression(tmp_path):
    # The failed first formal task exposed the human label in the manifest
    # while the queue carried the canonical key.  Both spellings must resolve
    # to one identity before validation compares them.
    assert canonical_dataset("MovieLens-1M") == "movielens"
    assert canonical_dataset("movielens") == "movielens"
    with pytest.raises(ValueError):
        canonical_dataset("MovieLens-1M-copy")
    with pytest.raises(ValueError):
        canonical_dataset("movie")

    # Reproduce the stdout race: the runner attests the file, then the parent
    # appends the child completion line through its inherited handle.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    names = [name for name in REQUIRED_FILES if name != "run_manifest.json"]
    hashes = {}
    for name in names:
        path = run_dir / name
        path.write_bytes(b"artifact-before-child-exit")
        hashes[name] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    stdout_path = run_dir / "stdout.log"
    stdout_path.write_bytes(b"[launch]\n[runner]")
    hashes["stdout.log"] = __import__("hashlib").sha256(stdout_path.read_bytes()).hexdigest()
    manifest = {"status": "OK", "architecture_optimizer_artifact": "",
                "artifact_hashes": hashes,
                "artifacts": ["run_manifest.json"] + names}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stdout_path.write_bytes(stdout_path.read_bytes() + b"\n[stage_S1] completed\n")
    assert any(reason == "artifact_hash_mismatch:stdout.log"
               for reason in validate_artifact_hash_inventory(str(run_dir), manifest))

    # Finalization runs only after the child/parent stdout handle is closed;
    # it repairs the stale pre-exit hash without hashing run_manifest.json.
    finalized = finalize_artifact_hashes(str(run_dir))
    assert finalized["artifact_hash_finalization"]["method"] == \
        "stage_S1_orchestrator_after_child_exit"
    reloaded = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert validate_artifact_hash_inventory(str(run_dir), reloaded) == []
    assert reloaded["artifact_hash_finalization"]["stdout_log_hash"] == \
        reloaded["artifact_hashes"]["stdout.log"]
    assert not list(run_dir.glob("run_manifest.json.finalizing-*.tmp"))

    # A same-size post-finalization mutation remains fail-closed even if its
    # mtime is restored to the value used by a prior hash-cache entry.
    prior_stat = stdout_path.stat()
    prior_bytes = stdout_path.read_bytes()
    stdout_path.write_bytes(bytes(byte ^ 1 for byte in prior_bytes))
    os.utime(stdout_path, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns))
    assert stdout_path.stat().st_size == prior_stat.st_size
    assert stdout_path.stat().st_mtime_ns == prior_stat.st_mtime_ns
    assert any(reason == "artifact_hash_mismatch:stdout.log"
               for reason in validate_artifact_hash_inventory(str(run_dir), reloaded))
