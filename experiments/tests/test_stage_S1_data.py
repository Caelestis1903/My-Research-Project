"""Synthetic data/reference pairing checks; no formal data load."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_data import role_reversal_sample, split_row_fingerprint


def test_role_reversal_uses_frozen_randomstate_rule():
    x = np.concatenate([
        np.column_stack([np.arange(10), np.zeros(10)]),
        np.column_stack([np.arange(10, 20), np.ones(10)]),
        np.column_stack([np.arange(20, 30), np.full(10, 2)]),
    ]).astype(np.float32)
    y = np.zeros((len(x), 2), dtype=np.float32)
    selected_x, selected_y, selected_ids = role_reversal_sample(
        x, y, "d0_large", 2022, {"L": 5, "M": 3, "S": 2}, np.arange(len(x)))
    assert selected_x.shape[0] == 10
    assert selected_y.shape[0] == 10
    assert selected_ids.shape[0] == 10
    assert {int(v): int((selected_x[:, -1] == v).sum()) for v in (0, 1, 2)} == {
        0: 5, 1: 3, 2: 2,
    }


def test_remapped_reference_fingerprint_is_distinct_from_physical_route():
    x = np.asarray([[10, 2], [11, 2], [12, 0]], dtype=np.float32)
    y = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    ids = np.asarray([10, 11, 12], dtype=np.int64)
    physical = x[x[:, -1] == 2].copy()
    labels = y[x[:, -1] == 2]
    physical[:, -1] = 0
    assert split_row_fingerprint(physical, labels, ids[:2])
    assert not np.array_equal(x[x[:, -1] == 2], physical)

