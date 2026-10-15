"""Nonformal Stage S2 smoke: tiny objective/model/artifact checks only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except Exception as exc:
        if args.allow_skip:
            print("SKIP: PyTorch unavailable:", exc)
            return 0
        raise
    if not torch.cuda.is_available():
        if args.allow_skip:
            print("SKIP: CUDA unavailable for the two-process M3OE boundary smoke")
            return 0
        raise RuntimeError("CUDA is required for the Stage S2 two-process smoke")

    from stage_S2_objectives import (assert_rejects_doubled_domain_share,
                                     batch_weight_audit, compute_equal_coefficients,
                                     objective_oracle)
    from stage_S2_orchestrator import (S2_BATCH_FIELDS, concurrent_resource_smoke,
                                       validate_batch_trace_contract)
    from stage_S2_report import CHECKPOINTS, build_paired_rows
    from stage_S2_protocol import SEEDS
    from stage_S2_run import run_tiny_runner_fixture
    from stage_S1_data import FeatureSchema
    from stage_S1_models import build_model

    table = compute_equal_coefficients([8, 2, 1])
    prediction = torch.sigmoid(torch.randn((12, 2), dtype=torch.float32)).requires_grad_(True)
    target = torch.randint(0, 2, (12, 2), dtype=torch.float32)
    domains = torch.arange(12, dtype=torch.long).remainder(3)
    # The fixed-prediction oracle uses a batch whose empirical shares match
    # the immutable fixture counts; production Equal still uses full counts.
    oracle = objective_oracle(prediction, target, domains, [4, 4, 4])
    assert oracle["sample_equivalent"] and oracle["equal_equivalent"]
    assert_rejects_doubled_domain_share(prediction, target, domains, [4, 4, 4])
    batch = batch_weight_audit(domains, table)
    assert batch["coefficient_table_hash"] == table["coefficient_table_hash"]

    tiny = run_tiny_runner_fixture(torch.device("cuda:0"))
    assert tiny["main_batch_fingerprints"] == 3
    assert tiny["architecture_batch_fingerprints"] == 1
    assert tiny["main_optimizer_steps"] == 3
    assert tiny["architecture_optimizer_steps"] == 1

    trace_rows = []
    for epoch in range(2):
        trace_rows.extend([
            {"stage": "S2", "kind": "train", "epoch": epoch, "step": 0,
             "batch_fingerprint": "train-{}".format(epoch), "batch_size": 4,
             "total_weighted_mass": 4.0, "train_objective": 0.5,
             "objective_name": "equal_domain_task_mean_bce", "coefficient_table_hash": "coeff"},
            {"stage": "S2", "kind": "architecture", "epoch": epoch, "step": -1,
             "batch_fingerprint": "arch-{}".format(epoch), "batch_size": 4,
             "total_weighted_mass": 4.0, "train_objective": 0.5,
             "objective_name": "equal_domain_task_mean_bce", "coefficient_table_hash": "coeff"},
        ])
    trace_manifest = {
        "batch_order_hash": hashlib.sha256(json.dumps(["train-0", "train-1"]).encode()).hexdigest(),
        "architecture_batch_order_hash": hashlib.sha256(json.dumps(["arch-0", "arch-1"]).encode()).hexdigest(),
        "train_batches_total": 2, "batch_fingerprint_count": 2,
        "architecture_batch_fingerprint_count": 2, "optimizer_steps": 2,
        "examples_processed": 8, "architecture_optimizer_steps": 2,
        "architecture_examples_processed": 8,
    }
    assert validate_batch_trace_contract(S2_BATCH_FIELDS, trace_rows, trace_manifest,
                                         {"coefficient_table_hash": "coeff"},
                                         {"0": 4}, epochs=2, batch_size=4) == []

    synthetic_rows = []
    for dataset, settings in (("movielens", ("d0_large", "d1_large", "d2_large")),
                              ("kuairand", ("natural",))):
        for setting in settings:
            for seed in SEEDS:
                for physical in range(3):
                    for task in range(2):
                        for checkpoint in CHECKPOINTS:
                            for method in ("equal", "sample", "single_domain_mtl_reference"):
                                synthetic_rows.append({
                                    "dataset": dataset, "setting": setting, "seed": seed,
                                    "physical_domain": physical, "task": task,
                                    "checkpoint": checkpoint, "method": method,
                                    "dataset_file_hash": "dataset", "physical_domain_identity": str(physical),
                                    "physical_view_fingerprint": "view", "split_row_id_hash": "split",
                                    "task_identity": "task{}".format(task), "initial_parameter_state_hash": "init",
                                    "batch_order_hash": method + "-batch",
                                    "architecture_batch_order_hash": method + "-arch",
                                    "auc": 0.6, "logloss": 0.5,
                                    "formal_run_key": method + "|{}|{}|{}".format(dataset, setting, seed),
                                })
    synthetic_rows[0]["physical_view_fingerprint"] = "corrupt"
    paired = build_paired_rows(synthetic_rows)
    assert any(int(row["pairing_valid"]) == 0 for row in paired)

    schema = FeatureSchema(
        dataset="synthetic", columns=("dense", "cat", "domain_indicator"),
        dense_columns=("dense",), sparse_columns=("cat", "domain_indicator"),
        vocab_sizes={"cat": 9, "domain_indicator": 3}, embedding_dim=16,
        domain_column="domain_indicator", encoder_hash="synthetic", scaler_hash="synthetic")
    model = build_model(schema, 3, 2, "m3oe", seed=2022)
    x = torch.zeros((3, 12), dtype=torch.float32)
    x[0] = torch.rand(12)
    x[1] = torch.arange(12).remainder(9).float()
    x[2] = domains.float()
    output = model(x)
    assert output.shape == (12, 2) and torch.isfinite(output).all()
    output.sum().backward()

    with tempfile.TemporaryDirectory(prefix="stage_s2_smoke_") as root:
        path = Path(root) / "objective_coefficients.json"
        path.write_text(json.dumps(table), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["coefficient_table_hash"] == table["coefficient_table_hash"]
    resource = concurrent_resource_smoke(python=sys.executable)
    assert resource["passed"]
    print("PASS: Stage S2 tiny objective/M3OE/artifact/concurrency smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
