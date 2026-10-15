"""Focused, dataset-free checks for Stage E3's weighting algorithm."""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E3_weighting import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_KEYS,
    E3_FORMAL_PROTOCOL,
    SharedGradientEpochAccumulator,
    alpha_key,
    collect_shared_cell_gradients,
    make_shared_gradient_step_rows,
    normalise_dataset_path,
    resolve_oracle_metrics,
    strict_validation_macro_auc,
    validate_alpha_coefficients,
    validate_oracle_artifact,
    weighted_loss_from_terms,
    weighting_coefficients_numpy,
)
from stage_E1_instrumentation import (  # noqa: E402
    build_parameter_partition,
    contribution_views,
    write_json,
    write_rows_csv,
)
from stage_E3_oracle_run import oracle_main_loss  # noqa: E402


def _make_formal_oracle_fixture(
    path: Path, *, device: str = "cuda:0",
    split_rows=None,
) -> tuple[str, dict[str, str]]:
    path.mkdir(parents=True)
    dataset_path = str(path / "dataset")
    split_rows = split_rows or {"train": "train-a", "validation": "val-a", "test": "test-a"}
    protocol = dict(E3_FORMAL_PROTOCOL)
    common = {
        "stage": "E3_ORACLE", "status": "OK", "oracle": True,
        "seed": 2022, "setting": "d0_large", "original_physical_domain": 0,
        "role": "L", "train_count": 150000, "role_sizes": {"L": 150000, "M": 34500, "S": 3000},
        "domain_num": 1, "task_num": 2,
        "split_counts": {
            "train": {"0": 150000, "1": 150000},
            "validation": {"0": 100, "1": 100},
            "test": {"0": 100, "1": 100},
        },
        "positive_rates": {
            "train": {"0": 0.5, "1": 0.5},
            "validation": {"0": 0.5, "1": 0.5},
            "test": {"0": 0.5, "1": 0.5},
        },
        "dataset": "MovieLens-1M", "dataset_path": dataset_path,
        "dataset_identity": normalise_dataset_path(dataset_path),
        "device_identity": device, "split_row_fingerprints": split_rows,
        "partition": {"shared": ["shared.weight"]},
        "model_identity": {"class": "MDMTRec", "backbone": "mdmoe",
                           "domain_num": 1, "task_num": 2, "expert_num": 4},
        "initial_parameter_state_hash": "initial",
        "observed_epochs": list(range(50)), "best_epoch": 0,
        "stopped_epoch": "", "censored_after_epoch": None,
        "alpha": None, "formal_protocol": protocol, "formal_valid": True,
        "selected_checkpoint_complete": True, "selected_checkpoint_cells": ["0", "1"],
    }
    config = dict(protocol)
    config.update({
        "stage": "E3_ORACLE", "oracle": True, "alpha": None,
        "original_physical_domain": 0, "role": "L", "train_count": 150000,
        "task_num": 2, "domain_num": 1, "role_sizes_parsed": common["role_sizes"],
        "model_identity": common["model_identity"], "formal_protocol": protocol,
        "formal_valid": True, "probe_interval": 1, "instrument": 1,
        "fix_optim_grouping": 1, "backbone": "mdmoe", "sampling": protocol["sampling"],
        "task": "all", "domain": "all", "dataset_path": dataset_path,
        "dataset_identity": common["dataset_identity"], "device_identity": device,
        "split_row_fingerprints": split_rows, "selected_checkpoint_complete": True,
        "selected_checkpoint_cells": ["0", "1"],
    })
    write_json(str(path / "run_manifest.json"), common)
    write_json(str(path / "config.json"), config)
    metrics_fields = [
        "seed", "setting", "original_physical_domain", "role", "epoch", "domain", "task",
        "cell", "train_sample_count", "validation_sample_count", "test_sample_count",
        "train_auc", "train_logloss", "validation_auc", "validation_logloss", "test_auc",
        "test_logloss", "selected_checkpoint", "stopped_censored",
    ]
    metrics = []
    for epoch in range(50):
        for task in (0, 1):
            metrics.append({
                "seed": 2022, "setting": "d0_large", "original_physical_domain": 0,
                "role": "L", "epoch": epoch, "domain": 0, "task": task, "cell": task,
                "train_sample_count": 150000, "validation_sample_count": 100,
                "test_sample_count": 100, "train_auc": 0.7, "train_logloss": 0.4,
                "validation_auc": 0.7, "validation_logloss": 0.4, "test_auc": 0.68,
                "test_logloss": 0.42, "selected_checkpoint": int(epoch == 0),
                "stopped_censored": 0,
            })
    write_rows_csv(str(path / "metrics_epoch.csv"), metrics, metrics_fields)
    summary_fields = [
        "seed", "setting", "original_physical_domain", "role", "task", "domain", "cell",
        "train_count", "best_epoch", "stopped_epoch", "best_val_auc", "best_val_logloss",
        "best_test_auc", "best_test_logloss", "oracle_auc", "oracle_logloss",
    ]
    summary = [{
        "seed": 2022, "setting": "d0_large", "original_physical_domain": 0, "role": "L",
        "task": task, "domain": 0, "cell": task, "train_count": 150000, "best_epoch": 0,
        "stopped_epoch": "", "best_val_auc": 0.7, "best_val_logloss": 0.4,
        "best_test_auc": 0.68, "best_test_logloss": 0.42, "oracle_auc": 0.68,
        "oracle_logloss": 0.42,
    } for task in (0, 1)]
    write_rows_csv(str(path / "metrics_summary.csv"), summary, summary_fields)
    (path / "stdout.log").write_text("formal fixture\n", encoding="utf-8")
    (path / "model.pth").write_bytes(b"checkpoint")
    return dataset_path, split_rows


def test_registered_alpha_grid_and_coefficients():
    shares = [0.8, 0.184, 0.016]
    assert [alpha_key(value) for value in ALPHA_GRID] == list(ALPHA_KEYS)
    np.testing.assert_allclose(
        weighting_coefficients_numpy(0.0, shares), np.ones(6), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        weighting_coefficients_numpy(1.0, shares),
        [2.4, 0.552, 0.048, 2.4, 0.552, 0.048], atol=1e-12, rtol=0,
    )
    np.testing.assert_allclose(
        weighting_coefficients_numpy(0.25, shares),
        [1.450270421, 1.004340436, 0.545389143] * 2, atol=1e-8, rtol=0,
    )
    np.testing.assert_allclose(
        weighting_coefficients_numpy(0.50, shares),
        [1.850704291, 0.887566598, 0.261729111] * 2, atol=1e-8, rtol=0,
    )
    np.testing.assert_allclose(
        weighting_coefficients_numpy(0.75, shares),
        [2.165590261, 0.719237241, 0.115172498] * 2, atol=1e-8, rtol=0,
    )
    for value in ALPHA_GRID:
        coeff = weighting_coefficients_numpy(value, shares)
        check = validate_alpha_coefficients(value, shares, coeff)
        assert np.isclose(check["domain_sum"], 3.0)
        assert np.all(coeff > 0)
        assert np.isclose(coeff.sum(), 6.0)


def test_explicit_dot_product_and_shared_observer_do_not_write_grad():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_weight = torch.nn.Parameter(torch.tensor([[0.4]]))

        def forward(self, x):
            return torch.sigmoid(x @ self.shared_weight.t()).reshape(-1)

    model = Tiny()
    partition = build_parameter_partition(model, strict=True)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pred = model(x)
    # Six independent terms share one forward-derived scalar so the exact
    # weighted loss is easy to audit while preserving one graph.
    base = torch.nn.functional.binary_cross_entropy(pred, y)
    terms = [base * (1.0 + 0.1 * idx) for idx in range(6)]
    coeff = torch.tensor(weighting_coefficients_numpy(0.5, [0.8, 0.184, 0.016]), dtype=torch.float32)
    expected = sum(term * coeff[idx] for idx, term in enumerate(terms))
    actual = weighted_loss_from_terms(terms, coeff)
    assert torch.allclose(actual, expected)
    raw, unused = collect_shared_cell_gradients(terms, model, partition, retain_graph=True)
    assert all(model.named_parameters().__iter__().__next__()[1].grad is None for _ in [0])
    before = model.shared_weight.detach().clone()
    model.zero_grad(set_to_none=True)
    actual.backward()
    assert model.shared_weight.grad is not None
    combined = sum(raw[idx] * coeff[idx] for idx in range(6))
    assert torch.allclose(model.shared_weight.grad.reshape(-1), combined, atol=1e-6, rtol=1e-6)
    model.zero_grad(set_to_none=True)
    assert torch.equal(model.shared_weight.detach(), before)
    assert all(value == 0 for value in unused.values())


def test_oracle_main_loss_is_task_sum_not_mean():
    criterion = torch.nn.BCELoss()
    output = torch.tensor([[0.2, 0.8], [0.7, 0.4]], dtype=torch.float32)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    task_terms = [criterion(output[:, task], target[:, task]) for task in (0, 1)]
    actual = oracle_main_loss(output, target, criterion)
    assert torch.allclose(actual, task_terms[0] + task_terms[1])
    assert not torch.allclose(actual, criterion(output, target))


def test_checkpoint_macro_rejects_missing_or_nonfinite_cell():
    stats = {
        0: {"auc": 0.7, "logloss": 0.4},
        1: {"auc": 0.8, "logloss": 0.3},
    }
    assert np.isclose(strict_validation_macro_auc(stats, [0, 1]), 0.75)
    with np.testing.assert_raises(RuntimeError):
        strict_validation_macro_auc({0: stats[0], 1: {"auc": np.nan, "logloss": 0.3}}, [0, 1])
    with np.testing.assert_raises(RuntimeError):
        strict_validation_macro_auc({0: stats[0]}, [0, 1])


def test_shared_contribution_rows_and_epoch_accumulator():
    raw = {idx: torch.tensor([float(idx + 1), float(6 - idx)])
           for idx in range(6)}
    coeff = torch.tensor(weighting_coefficients_numpy(0.5, [0.8, 0.184, 0.016]))
    views = contribution_views(raw, coeff, domain_num=3, task_num=2)
    assert np.isclose(sum(views["normalized"].values()), 1.0)
    assert np.isclose(sum(views["magnitude"].values()), 1.0)
    rows = make_shared_gradient_step_rows(
        raw, {idx: 0 for idx in range(6)}, alpha=0.5, coefficients=coeff,
        equal_coefficients=torch.ones(6), sample_coefficients=torch.tensor(
            weighting_coefficients_numpy(1.0, [0.8, 0.184, 0.016])),
        domain_num=3, task_num=2, shared_param_count=2,
        domain_shares=[0.8, 0.184, 0.016], seed=2022, setting="d0_large",
        batch_fingerprint="fp", epoch=0, step=0,
        cell_sample_counts=[10, 10, 3, 3, 1, 1],
        actual_update=torch.tensor([1.0, 1.0]), reconstruction_error=0.0,
    )
    assert len(rows) == 6
    assert all(row["weighted_gradient_norm"] == row["loss_coefficient"] * row["raw_gradient_norm"]
               for row in rows)
    assert np.isclose(sum(row["normalized_contribution"] for row in rows), 1.0)
    assert all(row["parameter_group"] == "shared" for row in rows)
    accumulator = SharedGradientEpochAccumulator(3, 2)
    accumulator.update(raw, views["normalized"], 0.2)
    accumulator.update({idx: value * 2 for idx, value in raw.items()}, views["normalized"], 0.3)
    epoch_rows = accumulator.rows(alpha=0.5, coefficients=coeff, seed=2022,
                                  setting="d0_large", epoch=0)
    assert len(epoch_rows) == 6
    assert all(int(row["probe_count"]) == 2 for row in epoch_rows)


def test_oracle_resolver_requires_explicit_seed_identity(tmp_path):
    root = tmp_path / "oracle"
    path = root / "d0_large" / "d0" / "seed2022"
    path.mkdir(parents=True)
    manifest = {
        "stage": "E3_ORACLE", "status": "OK", "oracle": True,
        "setting": "d0_large", "seed": 2022, "original_physical_domain": 0,
        "role": "L", "train_count": 150000, "alpha": None,
    }
    write_json(str(path / "run_manifest.json"), manifest)
    write_json(str(path / "config.json"), {})
    for name in ("metrics_epoch.csv", "stdout.log"):
        if name.endswith("csv"):
            write_rows_csv(str(path / name), [], ["domain", "task"])
        else:
            (path / name).write_text("ok\n", encoding="utf-8")
    # Missing split/identity fields make the artifact invalid; resolver must
    # leave both cells uncovered rather than trusting the directory name.
    from stage_E3_weighting import resolve_oracle_metrics
    metrics, audit = resolve_oracle_metrics(str(root), "d0_large", 2022)
    assert audit["coverage_cells"] == 0
    assert all(value["coverage"] is False for value in metrics.values())


def test_oracle_external_device_and_row_identity_are_required(tmp_path):
    root = tmp_path / "oracle"
    path = root / "d0_large" / "d0" / "seed2022"
    dataset_path, split_rows = _make_formal_oracle_fixture(path)
    valid, reason, _ = validate_oracle_artifact(
        str(path), setting="d0_large", seed=2022, domain=0,
        dataset_identity=dataset_path, device_identity="cuda:0",
        expected_split_row_fingerprints=split_rows)
    assert valid, reason

    wrong_device, wrong_device_reason, _ = validate_oracle_artifact(
        str(path), setting="d0_large", seed=2022, domain=0,
        dataset_identity=dataset_path, device_identity="cpu",
        expected_split_row_fingerprints=split_rows)
    assert not wrong_device
    assert "external_device_identity" in wrong_device_reason

    wrong_rows = dict(split_rows)
    wrong_rows["validation"] = "different-current-validation-rows"
    wrong_row, wrong_row_reason, _ = validate_oracle_artifact(
        str(path), setting="d0_large", seed=2022, domain=0,
        dataset_identity=dataset_path, device_identity="cuda:0",
        expected_split_row_fingerprints=wrong_rows)
    assert not wrong_row
    assert "external_split_row_identity" in wrong_row_reason

    _metrics, audit = resolve_oracle_metrics(
        str(root), "d0_large", 2022, dataset_identity=dataset_path,
        device_identity="cpu", expected_split_row_fingerprints={0: split_rows})
    assert audit["coverage_cells"] == 0
    _metrics, stale_audit = resolve_oracle_metrics(
        str(root), "d0_large", 2022, dataset_identity=dataset_path,
        device_identity="cuda:0", expected_split_row_fingerprints={0: wrong_rows})
    assert stale_audit["coverage_cells"] == 0
