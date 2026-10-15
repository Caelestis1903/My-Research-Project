import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_E4_weighting import (  # noqa: E402
    DOMAIN_NUM,
    TASK_NUM,
    TRAIN_DOMAIN_SHARES,
    WEIGHTING_KEYS,
    coefficient_table,
    contribution_views,
    make_shared_gradient_step_rows,
    validate_weighting_coefficients,
    weighted_loss_from_terms,
    weighting_coefficients,
    weighting_coefficients_numpy,
)


def test_registered_coefficients_match_audit_table_and_task_major_order():
    table = coefficient_table(TRAIN_DOMAIN_SHARES)
    assert np.allclose(table["equal"], [1.0] * 6, atol=1e-12)
    assert np.allclose(table["sqrt"][:3],
                       [0.612684844677, 1.946473203086, 0.440841952237], atol=1e-8)
    assert np.allclose(table["sample"][:3],
                       [0.258380804454, 2.607851327829, 0.133767867718], atol=1e-8)
    for key in WEIGHTING_KEYS:
        coeff = weighting_coefficients_numpy(key, TRAIN_DOMAIN_SHARES)
        assert np.all(coeff > 0)
        assert np.isclose(coeff[:DOMAIN_NUM].sum(), DOMAIN_NUM)
        assert np.isclose(coeff.sum(), DOMAIN_NUM * TASK_NUM)
        assert np.allclose(coeff[:DOMAIN_NUM], coeff[DOMAIN_NUM:])
        validate_weighting_coefficients(key, TRAIN_DOMAIN_SHARES, coeff)


def test_explicit_six_loss_dot_product_controls_real_gradient():
    parameter = torch.tensor([1.5, -0.5], dtype=torch.float64, requires_grad=True)
    terms = [parameter[0] ** 2, parameter[1] ** 2,
             (parameter[0] + parameter[1]) ** 2]
    coefficients = torch.tensor([0.25, 2.0, 1.5], dtype=torch.float64)
    loss = weighted_loss_from_terms(terms, coefficients)
    loss.backward()
    expected = torch.autograd.grad(
        coefficients[0] * terms[0].detach().requires_grad_(), parameter,
        allow_unused=True) if False else None
    # d/dp0 = 2*0.25*p0 + 2*1.5*(p0+p1); d/dp1 =
    # 2*2*p1 + 2*1.5*(p0+p1).
    assert torch.allclose(parameter.grad, torch.tensor([3.75, 1.0], dtype=torch.float64))


def test_shared_contribution_reconstruction_and_weighted_norm_identity():
    raw = {idx: torch.tensor([float(idx + 1), -0.5 * float(idx + 1)])
           for idx in range(6)}
    shares = TRAIN_DOMAIN_SHARES
    coeff = weighting_coefficients("sqrt", shares)
    equal = weighting_coefficients("equal", shares)
    sqrt = weighting_coefficients("sqrt", shares)
    sample = weighting_coefficients("sample", shares)
    views = contribution_views(raw, coeff, 3, 2)
    expected_combined = sum((raw[i] * coeff[i] for i in range(6)),
                            torch.zeros_like(views["combined"]))
    assert torch.allclose(views["combined"], expected_combined)
    rows = make_shared_gradient_step_rows(
        raw, {idx: 0 for idx in range(6)}, weighting="sqrt", coefficients=coeff,
        equal_coefficients=equal, sqrt_coefficients=sqrt,
        sample_coefficients=sample, domain_shares=shares, shared_param_count=2,
        seed=2022, epoch=0, step=0, batch_fingerprint_value="fp",
        cell_sample_counts=[2] * 6, actual_update=views["combined"],
        reconstruction_error=0.0)
    assert len(rows) == 6
    for row in rows:
        assert np.isclose(row["weighted_gradient_norm"],
                          row["loss_coefficient"] * row["raw_gradient_norm"], atol=1e-12)
        assert row["gradient_reconstruction_error"] == 0.0
    assert np.isfinite([row["mismatch_index"] for row in rows]).all()
