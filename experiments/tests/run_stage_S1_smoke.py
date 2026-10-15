"""Dependency-light Stage S1 acceptance smoke.

Run this script with the formal EDDAenv/CUDA Python.  It intentionally uses a
small synthetic batch and never writes a formal run or consumes a dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _schema(FeatureSchema):
    return FeatureSchema(
        dataset="synthetic", columns=("dense", "cat", "domain_indicator"),
        dense_columns=("dense",), sparse_columns=("cat", "domain_indicator"),
        vocab_sizes={"cat": 9, "domain_indicator": 3}, embedding_dim=16,
        domain_column="domain_indicator", encoder_hash="synthetic", scaler_hash="synthetic")


def _metric_validator_smoke():
    from stage_S1_metrics import LOGLOSS_EPS, cell_metrics, safe_auc, safe_logloss, task_domain_cell_index
    from stage_S1_orchestrator import validate_raw_prediction_metrics

    labels = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    prediction = np.asarray([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    assert LOGLOSS_EPS == 1e-12
    assert safe_auc(labels[:, 0], prediction[:, 0]) == 1.0
    assert safe_logloss([0, 1], [1, 0]) > 20.0
    payload = {"row_ids": np.arange(4, dtype=np.int64),
               "physical_domain": np.full(4, 2, dtype=np.int64), "labels": labels}
    checkpoints = ("selected", "final", "common_epoch_9", "common_epoch_19",
                   "common_epoch_29", "common_epoch_39", "common_epoch_49")
    for checkpoint in checkpoints:
        payload["pred_" + checkpoint] = prediction.copy()
    manifest = {"domain_num": 1, "local_to_physical": [2], "physical_domain": 2,
                "physical_domain_identity": ["tab4"], "domain_shares": {"0": 1.0},
                "preprocessing_metadata": {"label_definitions": {"task0": "alpha", "task1": "beta"}}}
    rows, summaries = [], []
    for checkpoint in checkpoints:
        for task in (0, 1):
            metric = cell_metrics(labels[:, task], prediction[:, task])
            row = {"checkpoint": checkpoint, "domain": 0, "task": task,
                   "cell": task_domain_cell_index(task, 0, 1), "physical_domain": 2,
                   "physical_domain_identity": "tab4", "domain_share": 1.0,
                   "task_identity": "alpha" if task == 0 else "beta",
                   "test_sample_count": metric["sample_count"],
                   "test_positive_count": metric["positive_count"],
                   "test_positive_rate": metric["positive_rate"], "auc": metric["auc"],
                   "logloss": metric["logloss"]}
            rows.append(row)
            if checkpoint == "selected":
                summaries.append(dict(row))
    assert validate_raw_prediction_metrics(payload, manifest, {"run_type": "single_domain"},
                                           rows, summaries) == []
    corrupted = [dict(row) for row in rows]
    corrupted[0]["auc"] += 0.01
    assert validate_raw_prediction_metrics(payload, manifest, {"run_type": "single_domain"},
                                           corrupted, summaries)


def _logging_equivalence_smoke(torch):
    from stage_S1_run import run_instrumentation_equivalence_fixture
    result = run_instrumentation_equivalence_fixture(torch.device("cuda:0"))
    assert result["parameter_equal"] and result["optimizer_equal"]
    assert result["metrics_equal"] and result["rng_equal"]
    assert result["batch_trajectory_equal"]
    assert result["on_log_rows"] > 0 and result["off_log_rows"] == 0


def _artifact_hash_closure_smoke():
    import hashlib
    import json

    from stage_S1_orchestrator import (REQUIRED_FILES, finalize_artifact_hashes,
                                       validate_artifact_hash_inventory)

    with tempfile.TemporaryDirectory(prefix="stage_s1_hash_closure_") as root:
        names = [name for name in REQUIRED_FILES if name != "run_manifest.json"]
        hashes = {}
        for name in names:
            path = Path(root) / name
            path.write_bytes(b"artifact-before-child-exit")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        stdout_path = Path(root) / "stdout.log"
        stdout_path.write_bytes(b"[launch]\n[runner]")
        hashes["stdout.log"] = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
        manifest = {"status": "OK", "architecture_optimizer_artifact": "",
                    "artifact_hashes": hashes,
                    "artifacts": ["run_manifest.json"] + names}
        (Path(root) / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        stdout_path.write_bytes(stdout_path.read_bytes() + b"\n[stage_S1] completed\n")
        assert any(reason == "artifact_hash_mismatch:stdout.log"
                   for reason in validate_artifact_hash_inventory(root, manifest))
        finalized = finalize_artifact_hashes(root)
        assert finalized["artifact_hash_finalization"]["method"] == \
            "stage_S1_orchestrator_after_child_exit"
        reloaded = json.loads((Path(root) / "run_manifest.json").read_text(encoding="utf-8"))
        assert validate_artifact_hash_inventory(root, reloaded) == []
        prior_stat = stdout_path.stat()
        prior_bytes = stdout_path.read_bytes()
        stdout_path.write_bytes(bytes(byte ^ 1 for byte in prior_bytes))
        os.utime(stdout_path, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns))
        assert any(reason == "artifact_hash_mismatch:stdout.log"
                   for reason in validate_artifact_hash_inventory(root, reloaded))


def _resume_safety_smoke():
    from stage_S1_orchestrator import (archive_failed_attempt, build_tasks,
                                       classify_task_state)
    import stage_S1_orchestrator as orchestrator

    with tempfile.TemporaryDirectory(prefix="stage_s1_resume_safety_") as root:
        run_dir = Path(root) / "started"
        run_dir.mkdir()
        stdout_path = str(run_dir / "stdout.log")
        assert orchestrator._run_child_process(
            [sys.executable, "-c", "print('child-complete')"], stdout_path) == 0
        archive = archive_failed_attempt(str(run_dir))
        assert archive and (Path(archive) / "stdout.log").is_file()

        task = build_tasks(str(Path(root) / "empty_raw"))[0]
        Path(task["run_dir"]).mkdir(parents=True)
        _, pending, blocked = classify_task_state([task], rerun_failed=True)
        assert pending and not blocked
        assert not (Path(task["run_dir"]) / "failed_attempts").exists()

        header_only = build_tasks(str(Path(root) / "header_raw"))[0]
        Path(header_only["run_dir"]).mkdir(parents=True)
        (Path(header_only["run_dir"]) / "stdout.log").write_text(
            "[launch 123] command-not-started\n", encoding="utf-8")
        _, pending, blocked = classify_task_state([header_only], rerun_failed=True)
        assert pending and not blocked
        assert not (Path(header_only["run_dir"]) / "failed_attempts").exists()


def _contradiction_axis_smoke():
    from stage_S1_report import contradiction_rows

    def row(dataset, setting, physical, role, delta, checkpoint="selected"):
        return {"dataset": dataset, "setting": setting, "backbone": "mmoe",
                "physical_domain": physical, "task": 0, "checkpoint": checkpoint,
                "seed": 2022, "size_role": role, "delta_auc": delta,
                "gap_logloss": 0.01}

    # Rotation-only opposition must not become a cross-dataset contradiction.
    negative = [row("movielens", "d0_large", 0, "L", -0.01),
                row("movielens", "d1_large", 0, "S", 0.01),
                row("kuairand", "natural", 0, "M", 0.02)]
    assert not any(item["comparison_axis"] == "dataset"
                   for item in contradiction_rows(negative))
    positive = [row("movielens", "d1_large", 1, "L", -0.01),
                row("kuairand", "natural", 1, "L", 0.01)]
    assert any(item["comparison_axis"] == "dataset"
               for item in contradiction_rows(positive))
    rotation = [row("movielens", "d0_large", 0, "L", -0.01),
                row("movielens", "d1_large", 0, "S", 0.01)]
    assert any(item["comparison_axis"] == "rotation"
               for item in contradiction_rows(rotation))
    rotation_stable = [row("movielens", "d0_large", 0, "L", -0.01),
                       row("movielens", "d1_large", 0, "S", -0.02)]
    assert not any(item["comparison_axis"] == "rotation"
                   for item in contradiction_rows(rotation_stable))
    stage = [row("movielens", "d0_large", 0, "L", -0.01, "selected"),
             row("movielens", "d0_large", 0, "L", 0.01, "common_epoch_9")]
    assert any(item["comparison_axis"] == "stage"
               for item in contradiction_rows(stage))
    stage_stable = [row("movielens", "d0_large", 0, "L", -0.01, "selected"),
                    row("movielens", "d0_large", 0, "L", -0.02, "common_epoch_9")]
    assert not any(item["comparison_axis"] == "stage"
                   for item in contradiction_rows(stage_stable))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        import torch.nn.functional as F
        from stage_S1_data import FeatureSchema
        from stage_S1_models import apply_semantic_initialization, build_model
        from stage_S1_run import direct_per_example_bce, normalized_cell_bce, select_checkpoint_epoch
    except Exception as exc:
        if args.allow_skip:
            print("SKIP: EDDAenv dependencies unavailable:", exc)
            return 0
        raise

    if not torch.cuda.is_available():
        if args.allow_skip:
            print("SKIP: CUDA unavailable")
            return 0
        raise RuntimeError("formal Stage S1 smoke requires CUDA")
    device = torch.device("cuda:0")
    schema = _schema(FeatureSchema)
    batch_size = 12
    x = torch.zeros((3, batch_size), dtype=torch.float32, device=device)
    x[0] = torch.rand(batch_size, device=device)
    x[1] = torch.randint(0, 9, (batch_size,), device=device).float()
    x[2] = torch.arange(batch_size, device=device).remainder(3).float()
    y = torch.randint(0, 2, (batch_size, 2), device=device).float()
    for backbone in ("m3oe", "mmoe", "ple"):
        for domain_num in (3, 1):
            model = build_model(schema, domain_num, 2, backbone, seed=2022).to(device)
            model_x = x.clone()
            model_x[2] = model_x[2].remainder(domain_num)
            prediction = model(model_x)
            assert prediction.device == device, (backbone, domain_num, prediction.device)
            assert prediction.shape == (batch_size, 2)
            loss = direct_per_example_bce(prediction, y)
            assert torch.isfinite(loss)
            loss.backward()
            assert any(parameter.grad is not None for parameter in model.parameters()
                       if parameter.requires_grad)
    domain_ids = x[2].long()
    oracle = normalized_cell_bce(
        torch.sigmoid(torch.randn_like(y)), y, domain_ids, 3, 2)
    direct = direct_per_example_bce(torch.sigmoid(torch.randn_like(y)), y)
    # Reuse one prediction for the exact equality check.
    pred = torch.sigmoid(torch.randn_like(y))
    assert torch.allclose(direct_per_example_bce(pred, y),
                          normalized_cell_bce(pred, y, domain_ids, 3, 2), atol=1e-6)
    assert select_checkpoint_epoch({0: 0.5, 1: 0.5, 2: 0.4}) == 0

    target_schema = _schema(FeatureSchema)
    target_schema.vocab_sizes = dict(target_schema.vocab_sizes)
    target_schema.vocab_sizes["domain_indicator"] = 1
    for backbone in ("m3oe", "mmoe", "ple"):
        source = build_model(schema, 3, 2, backbone, seed=2022)
        for physical_domain in (0, 1, 2):
            target = build_model(target_schema, 1, 2, backbone, seed=2023)
            audit = apply_semantic_initialization(
                source, target, backbone=backbone, physical_domain=physical_domain,
                base_seed=2022)
            for entry in audit.mapping_table:
                if entry.get("source") and entry.get("kind") != "name_derived_subseed":
                    assert entry.get("hash_equal") is True, entry
            if backbone == "m3oe":
                mappings = {item["target"]: item["source"] for item in audit.mapping_table}
                assert mappings["gate.0.0.weight"] == "gate.{}.0.weight".format(physical_domain)
                assert mappings["gate.1.0.weight"] == "gate.{}.0.weight".format(3 + physical_domain)
    _metric_validator_smoke()
    _logging_equivalence_smoke(torch)
    _artifact_hash_closure_smoke()
    _resume_safety_smoke()
    _contradiction_axis_smoke()
    print("PASS: Stage S1 CUDA forward/loss/backward, objective, tie, and semantic mapping smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
