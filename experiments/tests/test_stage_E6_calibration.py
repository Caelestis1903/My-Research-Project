"""Focused Stage E6 algorithm checks; no dataset, GPU, or formal matrix run."""

import os
import sys

import numpy as np
import torch
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from stage_E1_instrumentation import build_parameter_partition  # noqa: E402
from stage_E6_calibration import (  # noqa: E402
    E6_ARM_KEYS,
    E6_BETA,
    EMAMagnitudeState,
    calibration_metrics,
    closed_form_coefficients,
    coefficient_adjusted_magnitude_share,
    coefficient_variation,
    raw_domain_magnitudes,
    static_coefficients,
    target_shares,
)
from stage_E6_run import E6Runner  # noqa: E402


def test_raw_domain_magnitude_is_task_sum_of_l2_norms():
    raw = {
        0: torch.tensor([3.0, 4.0]), 1: torch.tensor([0.0, 0.0]),
        2: torch.tensor([6.0]), 3: torch.tensor([8.0]),
        4: torch.tensor([5.0, 12.0]), 5: torch.tensor([0.0, 5.0]),
    }
    actual = raw_domain_magnitudes(raw, domain_num=3, task_num=2)
    # Task-major cells are (t0,d0..d2) then (t1,d0..d2):
    # d0=5+8, d1=0+13, d2=6+5.
    torch.testing.assert_close(actual, torch.tensor([13.0, 13.0, 11.0], dtype=torch.float64))
    assert not actual.requires_grad


def test_closed_form_coefficients_sum_to_d_and_track_target():
    q = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)
    target = [0.8, 0.184, 0.016]
    coeff, details = closed_form_coefficients(q, target)
    assert not coeff.requires_grad
    assert np.isclose(float(coeff[:3].sum()), 3.0, atol=1e-12)
    assert np.all(coeff.detach().cpu().numpy() > 0)
    share = coefficient_adjusted_magnitude_share(q.detach(), coeff)
    np.testing.assert_allclose(share.cpu().numpy(), target, atol=1e-12, rtol=0)
    assert np.isclose(details["coefficient_sum_cell"], 6.0, atol=1e-12)


def test_ema_initializes_then_uses_preregistered_beta():
    state = EMAMagnitudeState()
    first = state.update([1.0, 2.0, 4.0])
    second = state.update([3.0, 4.0, 8.0])
    np.testing.assert_allclose(first.numpy(), [1.0, 2.0, 4.0], atol=0, rtol=0)
    np.testing.assert_allclose(second.numpy(),
                               E6_BETA * np.asarray([1.0, 2.0, 4.0]) +
                               (1.0 - E6_BETA) * np.asarray([3.0, 4.0, 8.0]),
                               atol=1e-14, rtol=0)
    assert state.update_count == 2
    assert state.value.dtype == torch.float64
    assert not state.value.requires_grad


def test_uniform_ablation_changes_only_target():
    shares = [0.8, 0.184, 0.016]
    np.testing.assert_allclose(target_shares("adaptive_ema", shares), shares)
    np.testing.assert_allclose(target_shares("adaptive_ema_uniform", shares), [1 / 3] * 3)
    equal = static_coefficients("equal_static", shares)
    sample = static_coefficients("sample_static", shares)
    np.testing.assert_allclose(equal.numpy(), np.ones(6))
    np.testing.assert_allclose(sample.numpy(), np.tile(np.asarray(shares) * 3, 2))


def test_coefficient_variation_is_domain_only_and_excludes_first_transition():
    previous = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    current = torch.tensor([1.2, 1.8, 3.3, 1.2, 1.8, 3.3], dtype=torch.float64)
    assert np.isclose(coefficient_variation(None, current, 3), np.nan, equal_nan=True)
    # (|.2|+|-.2|+|.3|)/3; the task copy is not counted twice.
    assert np.isclose(coefficient_variation(previous, current, 3), 0.7 / 3.0)


def test_autograd_raw_observer_does_not_pollute_parameter_grad():
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
    base = torch.nn.functional.binary_cross_entropy(pred, y)
    raw = {}
    for idx in range(6):
        (grad,) = torch.autograd.grad(base * (idx + 1.0), [model.shared_weight], retain_graph=True)
        raw[idx] = grad.detach()
    assert model.shared_weight.grad is None
    magnitudes = raw_domain_magnitudes(raw, 3, 2)
    assert torch.all(magnitudes > 0)


def test_shared_step_rows_keep_first_variation_blank_then_log_transition():
    runner = object.__new__(E6Runner)
    runner.seed = 2022
    runner.setting = "d0_large"
    runner.arm_key = "adaptive_noema"
    runner.domain_num = 3
    runner.task_num = 2
    runner.domain_shares = np.asarray([0.8, 0.184, 0.016], dtype=np.float64)
    runner.target = runner.domain_shares.copy()
    runner.ema_state = None
    runner.partition = SimpleNamespace(counts={"shared": 2})
    runner.static_equal = torch.ones(6, dtype=torch.float64)
    runner.static_sample = torch.tensor([2.4, 0.552, 0.048] * 2, dtype=torch.float64)
    raw = {idx: torch.tensor([float(idx + 1), 1.0]) for idx in range(6)}
    coeff = torch.tensor([2.0, 0.8, 0.2] * 2, dtype=torch.float64)
    raw_mag = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    calibration = {"data_share_mismatch": 0.1, "target_calibration_error": 0.2}
    kwargs = dict(
        raw_by_cell=raw, unused_by_cell={idx: 0 for idx in range(6)},
        coefficients=coeff, raw_magnitude=raw_mag, q_magnitude=raw_mag,
        ema_magnitude=raw_mag, counts=[1] * 6, epoch=0, step=0,
        fingerprint="fp", actual_update=torch.ones(2), reconstruction_error=0.0,
        calibration=calibration, probe=True,
    )
    first = runner._shared_step_rows(coeff_var="", **kwargs)
    second = runner._shared_step_rows(coeff_var=0.125, **kwargs)
    assert all(row["coefficient_variation"] == "" for row in first)
    assert all(row["coefficient_variation"] == 0.125 for row in second)
