"""Dependency-light S2 objective and coefficient tests.

These tests never construct a formal dataset or launch training.  The Torch
cases are skipped when the repository's CUDA/EDDA environment is unavailable.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S2_objectives import (  # noqa: E402
    assert_rejects_doubled_domain_share,
    batch_weight_audit,
    compute_equal_coefficients,
    objective_oracle,
    validate_coefficient_invariants,
)


def test_equal_coefficients_match_movielens_roles_and_population_scale():
    table = compute_equal_coefficients({0: 150000, 1: 34500, 2: 3000})
    assert table["domain_share_p"] == {
        "0": 0.8, "1": 34500 / 187500, "2": 3000 / 187500,
    }
    for domain in range(3):
        assert abs(float(table["effective_domain_coefficient_a"][str(domain)]) - 1 / 3) < 1e-12
        assert abs(float(table["effective_cell_coefficient"][str(domain)]) - 1 / 6) < 1e-12
    assert abs(float(table["population_mean_weight"]) - 1.0) < 1e-12
    validate_coefficient_invariants(table)


def test_equal_coefficients_match_kuairand_counts():
    table = compute_equal_coefficients({0: 172782, 1: 1743898, 2: 89452})
    assert table["train_counts"] == {"0": 172782, "1": 1743898, "2": 89452}
    for domain in range(3):
        assert abs(float(table["domain_share_p"][str(domain)]) *
                   float(table["loss_coefficient_w"][str(domain)]) - 1 / 3) < 1e-12


def test_torch_oracle_proves_sample_equal_and_rejects_double_weighting():
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([
        [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1],
        [0.35, 0.65], [0.55, 0.45],
    ], dtype=torch.float32)
    target = torch.tensor([
        [0., 1.], [1., 0.], [0., 1.], [1., 0.], [1., 1.], [0., 1.],
    ], dtype=torch.float32)
    domains = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    oracle = objective_oracle(prediction, target, domains, [2, 1, 3])
    assert oracle["sample_equivalent"] and oracle["equal_equivalent"]
    assert oracle["effective_domain_coefficients"] == pytest.approx([1 / 3] * 3)
    assert oracle["population_mean_weight"] == pytest.approx(1.0)
    assert_rejects_doubled_domain_share(prediction, target, domains, [2, 1, 3])


def test_batch_audit_records_raw_and_weighted_mass_without_renormalizing():
    torch = pytest.importorskip("torch")
    table = compute_equal_coefficients([8, 2, 1])
    audit = batch_weight_audit(torch.tensor([0, 0, 0, 2]), table)
    assert audit["raw_domain_counts"] == [3, 0, 1]
    assert audit["total_weighted_mass"] == pytest.approx(
        3 * float(table["loss_coefficient_w"]["0"]) + float(table["loss_coefficient_w"]["2"]))
    assert audit["coefficient_table_hash"] == table["coefficient_table_hash"]
