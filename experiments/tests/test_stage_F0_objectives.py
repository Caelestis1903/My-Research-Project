import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_objectives import (
    equal_diagnostic_bce,
    sample_bce,
    split_weighted_sample_loss,
)


def test_sample_batch_means_recombine_to_exact_full_pool_mean():
    prediction = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    target = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    whole = sample_bce(prediction, target)
    first = sample_bce(prediction[:3], target[:3])
    second = sample_bce(prediction[3:], target[3:])
    combined = split_weighted_sample_loss([first, second], [3, 2])
    torch.testing.assert_close(combined, whole, rtol=0, atol=1e-7)


def test_equal_diagnostic_uses_fixed_training_shares_and_row_denominator():
    prediction = torch.tensor([[0.25], [0.75], [0.5]], dtype=torch.float64)
    target = torch.tensor([[0.], [1.], [0.]], dtype=torch.float64)
    domains = torch.tensor([0, 0, 1])
    shares = {0: 2 / 3, 1: 1 / 3}
    actual = equal_diagnostic_bce(prediction, target, domains, shares, 2)
    bce = F.binary_cross_entropy(prediction, target, reduction="none").flatten()
    expected = (bce * torch.tensor([0.75, 0.75, 1.5], dtype=torch.float64)).sum() / 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
