"""Focused E5 schedule and objective-path tests (no dataset or GPU)."""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E5_schedule import (  # noqa: E402
    E5_ARM_KEYS,
    E5_EPOCHS,
    alpha_trajectory,
    arm_alpha,
    exposure_table,
    scheduled_coefficients_numpy,
    validate_registry,
)
from stage_E3_weighting import weighted_loss_from_terms  # noqa: E402


def test_registered_boundaries_and_mean_exposure():
    registry = validate_registry()
    assert registry["arms"] == list(E5_ARM_KEYS)
    assert arm_alpha("forward", 0) == 0.75
    assert arm_alpha("forward", 17) == 1.0
    assert arm_alpha("reverse", 0) == 1.0
    assert arm_alpha("reverse", 17) == 0.75
    assert np.isclose(np.mean(alpha_trajectory("forward")), 0.875)
    assert np.isclose(np.mean(alpha_trajectory("reverse")), 0.875)
    assert all(np.isclose(
        arm_alpha("forward", epoch) + arm_alpha("reverse", epoch), 1.75
    ) for epoch in range(E5_EPOCHS))


def test_e3_normalisation_and_task_copy_order():
    shares = [0.8, 0.184, 0.016]
    for arm in E5_ARM_KEYS:
        for epoch in (0, 8, 9, 17):
            coeff = scheduled_coefficients_numpy(arm, epoch, shares, 2)
            assert coeff.shape == (6,)
            assert np.isclose(coeff.sum(), 6.0)
            assert np.all(coeff[:3] == coeff[3:])
            assert np.all(coeff > 0)
    # The domain order is task-major: [t0 d0,d1,d2,t1 d0,d1,d2].
    coeff = scheduled_coefficients_numpy("forward", 0, shares, 2)
    assert np.allclose(coeff[:3], coeff[3:])


def test_toy_loss_weighted_sum_and_backward_gradient():
    parameter = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    terms = [parameter[0] ** 2, 3.0 * parameter[1] ** 2,
             parameter[0] * parameter[1], parameter[0] ** 2 + parameter[1] ** 2,
             parameter[0] - parameter[1], parameter[0] + 2.0 * parameter[1]]
    coeff = torch.tensor(
        scheduled_coefficients_numpy("forward", 8, [0.8, 0.184, 0.016], 2),
        dtype=torch.float64,
    )
    loss = weighted_loss_from_terms(terms, coeff)
    expected = sum(float(coeff[i]) * terms[i] for i in range(6))
    assert torch.allclose(loss, expected)
    gradient = torch.autograd.grad(loss, parameter)[0]
    assert gradient.shape == parameter.shape
    assert torch.isfinite(gradient).all()


def test_exposure_table_reports_step_weighted_integrals():
    table = exposure_table("static_matched", [0.8, 0.184, 0.016], 2, steps_per_epoch=7)
    assert table["epochs"] == 18
    assert table["total_main_steps"] == 18 * 7
    assert np.isclose(table["alpha_mean"], 0.875)
    assert len(table["coefficient_integral_by_cell"]) == 6
    assert len(table["step_weighted_coefficient_integral_by_domain"]) == 3
    assert np.allclose(
        np.asarray(table["step_weighted_coefficient_integral_by_cell"]),
        7.0 * np.asarray(table["coefficient_integral_by_cell"]),
    )
