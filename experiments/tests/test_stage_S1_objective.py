"""Direct per-example BCE objective checks."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_run import objective_equivalence


def test_direct_sample_objective_equals_normalized_cell_expression():
    prediction = torch.tensor([
        [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.9, 0.1], [0.35, 0.65],
    ], dtype=torch.float32)
    target = torch.tensor([
        [0., 1.], [1., 0.], [0., 1.], [1., 0.], [1., 1.],
    ], dtype=torch.float32)
    domain_ids = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    result = objective_equivalence(prediction, target, domain_ids, 3, 2)
    assert result["equivalent"]
    assert result["absolute_difference"] < 1e-6

