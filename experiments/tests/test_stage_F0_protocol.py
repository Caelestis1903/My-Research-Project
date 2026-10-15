import os
import sys
import math
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_protocol import (
    FROZEN_WIDTHS,
    all_b1_units,
    config_grid,
    expected_matrix_totals,
    matrix_ledger,
    paired_delta_t_interval,
    primary_search_expansion_trigger,
    T_CRITICAL_95_DF3,
    width_for,
)


def test_frozen_b1_grid_widths_and_matrix_ledger():
    assert len(all_b1_units()) == 12
    assert len(config_grid()) == 12
    assert width_for("m3oe", "mdmt") == FROZEN_WIDTHS[("m3oe", "mdmt")]
    assert width_for("mmoe", "sdmt-d1") == FROZEN_WIDTHS[("mmoe", "sdmt")]
    assert expected_matrix_totals() == {
        "units": 122, "search": 2928, "formal": 488, "trajectories": 3416,
    }
    ledger = matrix_ledger()
    assert ledger["required_blocks"][0]["status"] == "IMPLEMENTED_NOT_RUN_APPROVAL_PENDING"
    assert all(row["status"].startswith("BLOCKED_NOT_IMPLEMENTED")
               for row in ledger["required_blocks"][1:])


def test_validation_only_seed_expansion_trigger():
    assert not primary_search_expansion_trigger([[0.0] * 4 for _ in range(5)])
    crossed = [[-0.01, -0.01, 0.01, 0.01]] + [[0.0] * 4 for _ in range(4)]
    assert primary_search_expansion_trigger(crossed)


def test_expansion_trigger_validates_all_five_comparisons_before_deciding():
    already_triggered = [-0.01, -0.01, 0.01, 0.01]
    valid = [already_triggered] + [[0.0] * 4 for _ in range(4)]
    assert primary_search_expansion_trigger(valid)

    nonfinite_tail = [already_triggered] + [[0.0] * 4 for _ in range(3)] + [
        [0.0, 0.0, float("nan"), 0.0]
    ]
    with pytest.raises(ValueError, match="finite"):
        primary_search_expansion_trigger(nonfinite_tail)

    wrong_length_tail = [already_triggered] + [[0.0] * 4 for _ in range(3)] + [
        [0.0, 0.0, 0.0]
    ]
    with pytest.raises(ValueError, match="exactly four"):
        primary_search_expansion_trigger(wrong_length_tail)


def test_first_four_interval_uses_df3_t_critical_and_exact_trigger_boundary():
    assert math.isclose(T_CRITICAL_95_DF3, 3.1824463053, rel_tol=0, abs_tol=1e-10)
    epsilon = 0.003
    amplitude = epsilon * math.sqrt(3.0) / T_CRITICAL_95_DF3
    lower, upper = paired_delta_t_interval([-amplitude, -amplitude, amplitude, amplitude])
    assert math.isclose(lower, -epsilon, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(upper, epsilon, rel_tol=0, abs_tol=1e-12)
    above = amplitude * (1.0 + 1e-8)
    below = amplitude * (1.0 - 1e-8)
    assert primary_search_expansion_trigger(
        [[-above, -above, above, above]] + [[0.0] * 4 for _ in range(4)])
    assert not primary_search_expansion_trigger(
        [[-below, -below, below, below]] + [[0.0] * 4 for _ in range(4)])


def test_first_four_interval_rejects_nonfinite_values():
    for values in ([0.0, 0.0, float("nan"), 0.0],
                   [0.0, float("inf"), 0.0, 0.0]):
        try:
            paired_delta_t_interval(values)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite paired deltas must fail closed")
