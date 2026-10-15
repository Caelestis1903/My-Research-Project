"""Fast unit tests for the Stage E1 observation math (no dataset/GPU)."""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
sys.path.insert(0, BASE)

from stage_E1_instrumentation import (  # noqa: E402
    FORMAL_PROTOCOL,
    GradientEpochAccumulator,
    WelfordVector,
    build_parameter_partition,
    collect_cell_gradients,
    collect_nonfinite_inventory,
    compare_on_off_runs,
    compare_pair_runs,
    contribution_views,
    make_gradient_step_rows,
    masked_bce_terms,
    parameter_group_for_name,
    weighting_coefficients_numpy,
    write_json,
    write_rows_csv,
)


def _write_comparison_identity(run_dir, *, weighting="equal", setting="d0_large",
                               seed=2022, instrument=1, observed_epochs=None,
                               stopped_epoch=None, censored_after_epoch=None,
                               initial_hash="initial", final_hash="final"):
    """Write the minimum explicit identity required by pair comparators."""

    if observed_epochs is None:
        observed_epochs = [0]
    partition = {
        "groups": {"shared": ["p"]}, "param_count": 1,
        "param_shapes": {"p": [1]}, "manifest_hash": "partition",
        "state_hash": initial_hash,
    }
    manifest = {
        "stage": "E1", "status": "OK", "seed": seed,
        "setting": setting, "weighting": weighting,
        "formal_run_key": f"{weighting}|{setting}|{seed}",
        "dataset": "MovieLens-1M", "domain_num": 2, "task_num": 1,
        "role_by_domain": {"0": "L", "1": "S"},
        "role_sizes": {"L": 2, "M": 1, "S": 1},
        "train_domain_counts": {"0": 2, "1": 1},
        "domain_shares": [2.0 / 3.0, 1.0 / 3.0],
        "split_counts": {"train": {"0": 2, "1": 1},
                          "validation": {"0": 2, "1": 1},
                          "test": {"0": 2, "1": 1}},
        "positive_rates": {"train": {"0": 0.5, "1": 0.5},
                            "validation": {"0": 0.5, "1": 0.5},
                            "test": {"0": 0.5, "1": 0.5}},
        "initial_parameter_state_hash": initial_hash,
        "final_parameter_state_hash": final_hash,
        "partition": partition,
        "observed_epochs": list(observed_epochs),
        "stopped_epoch": "" if stopped_epoch is None else stopped_epoch,
        "censored_after_epoch": censored_after_epoch,
        "rng_restore_evidence": [{"epoch": 0, "scope": "test_eval",
                                   "before": "rng", "after": "rng", "restored": True}],
    }
    config = {
        "weighting": weighting, "loss_weighting": weighting,
        "setting": setting, "seed": seed, "run_dir": str(run_dir),
        "config_hash": f"config-{weighting}", "instrument": instrument,
        "instrumentation": bool(instrument), "dataset_path": "same-data",
        "model": "tiny", "partition_marker": "same",
    }
    write_json(str(run_dir / "run_manifest.json"), manifest)
    write_json(str(run_dir / "config.json"), config)


def _write_metrics_for_comparison(run_dir):
    write_rows_csv(str(run_dir / "metrics_epoch.csv"), [{
        "seed": 2022, "epoch": 0, "domain": 0, "task": 0, "cell": 0,
        "AUC": 0.5, "LogLoss": 0.5,
    }], ["seed", "epoch", "domain", "task", "cell", "AUC", "LogLoss"])
    write_rows_csv(str(run_dir / "metrics_summary.csv"), [{
        "seed": 2022, "weighting": "equal", "setting": "d0_large",
        "domain": 0, "task": 0, "cell": 0, "AUC": 0.5,
    }], ["seed", "weighting", "setting", "domain", "task", "cell", "AUC"])


class TinyPartitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 2, bias=False)
        self.shared_weight = torch.nn.Parameter(torch.ones(2, 2))
        self.shared_bias = torch.nn.Parameter(torch.zeros(2))
        self.skip_conn = torch.nn.Linear(2, 2)
        self.star_mlp = torch.nn.Linear(2, 2)
        self.expert = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.slot_weight = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2, 2))])
        self.slot_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2))])
        self.domain_expert = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.task_expert = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.tower = torch.nn.ModuleList([torch.nn.Linear(2, 1)])
        self._weight_dummy = torch.nn.Parameter(torch.ones(1))


def test_weighting_coefficients_and_scale():
    shares = [0.8, 0.184, 0.016]
    equal = weighting_coefficients_numpy("equal", shares, task_num=2)
    sample = weighting_coefficients_numpy("sample", shares, task_num=2)
    np.testing.assert_allclose(equal, np.ones(6))
    np.testing.assert_allclose(sample, [2.4, 0.552, 0.048, 2.4, 0.552, 0.048])
    assert np.isclose(equal.sum(), 6.0)
    assert np.isclose(sample.sum(), 6.0)


def test_parameter_partition_is_explicit_and_disjoint():
    model = TinyPartitionModel()
    partition = build_parameter_partition(model)
    assert set(partition.groups) == {
        "shared", "domain_specific", "task_specific", "cell_specific", "architecture"
    }
    assert "domain_expert.0.weight" in partition.groups["domain_specific"]
    assert "task_expert.0.weight" in partition.groups["task_specific"]
    assert "_weight_dummy" in partition.groups["architecture"]
    assert "gate.0.weight" in partition.groups["cell_specific"]
    flattened = [name for group in partition.groups.values() for name in group]
    assert len(flattened) == len(set(flattened)) == len(list(model.named_parameters()))
    assert parameter_group_for_name("domain_expert.0.weight") == "domain_specific"
    assert parameter_group_for_name("task_expert.0.weight") == "task_specific"


def test_contribution_projection_and_welford():
    raw = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
        2: torch.tensor([1.0, 1.0]),
        3: torch.tensor([0.0, 0.0]),
    }
    coefficients = torch.tensor([1.0, 2.0, 1.0, 2.0])
    views = contribution_views(raw, coefficients, domain_num=2, task_num=2)
    assert np.isclose(sum(views["normalized"].values()), 1.0)
    assert np.isclose(sum(views["magnitude"].values()), 1.0)
    assert torch.allclose(views["combined"], torch.tensor([2.0, 3.0]))

    accumulator = WelfordVector()
    accumulator.update(torch.tensor([1.0, 0.0]))
    accumulator.update(torch.tensor([3.0, 0.0]))
    assert accumulator.count == 2
    assert np.isclose(accumulator.norm_of_mean, 2.0)
    assert np.isclose(accumulator.variance_trace, 2.0)
    assert np.isclose(accumulator.snr, 2.0)


def test_masked_terms_and_step_schema():
    pred = torch.tensor([[0.5, 0.25], [0.75, 0.5], [0.25, 0.75]], requires_grad=True)
    y = torch.tensor([[0, 1], [1, 1], [0, 0]])
    domains = torch.tensor([0, 1, 0])
    terms, counts = masked_bce_terms(pred, y, domains, domain_num=2, task_num=2)
    assert counts == [2, 1, 2, 1]
    assert len(terms) == 4
    model = TinyPartitionModel()
    partition = build_parameter_partition(model)
    raw_by_group = {
        group: {idx: torch.ones(partition.counts[group])
                for idx in range(4)}
        for group in ("shared", "domain_specific", "task_specific", "cell_specific", "architecture")
    }
    unused = {group: {idx: 0 for idx in range(4)} for group in raw_by_group}
    rows = make_gradient_step_rows(
        raw_by_group, unused,
        actual_coefficients=torch.ones(4),
        equal_coefficients=torch.ones(4),
        sample_coefficients=torch.tensor([1.5, 0.5, 1.5, 0.5]),
        domain_num=2, task_num=2, partition=partition,
        domain_shares=[0.75, 0.25], weighting="equal", epoch=0, step=0,
        batch_fingerprint="abc", cell_sample_counts=counts,
    )
    assert len(rows) == 4 * 5
    required = {
        "raw_gradient_norm", "weighted_gradient_norm", "gradient_reconstruction_error",
        "equal_cf_contribution", "sample_cf_contribution", "unused_param_count",
    }
    assert required.issubset(rows[0])
    assert "domain_signed_contribution" in rows[0]


def test_pair_comparator_and_nonfinite_inventory(tmp_path):
    fields = ["seed", "epoch", "step", "kind", "batch_fingerprint",
              "batch_size", "domain_counts", "task_counts", "is_probe", "train_loss"]
    rows = [{
        "seed": 2022, "epoch": 0, "step": 0, "kind": "train",
        "batch_fingerprint": "same", "batch_size": 2,
        "domain_counts": "[1,1]", "task_counts": "[2]", "is_probe": 1,
        "train_loss": 0.25,
    }, {
        "seed": 2022, "epoch": 0, "step": -1, "kind": "architecture",
        "batch_fingerprint": "arch", "batch_size": 2,
        "domain_counts": "[1,1]", "task_counts": "[2]", "is_probe": 0,
        "train_loss": 0.5,
    }]
    from stage_E1_instrumentation import write_rows_csv
    equal_dir = tmp_path / "equal"
    sample_dir = tmp_path / "sample"
    equal_dir.mkdir()
    sample_dir.mkdir()
    write_rows_csv(str(equal_dir / "batch_trace.csv"), rows, fields)
    write_rows_csv(str(sample_dir / "batch_trace.csv"), rows, fields)
    _write_comparison_identity(equal_dir, weighting="equal")
    _write_comparison_identity(sample_dir, weighting="sample")
    pair = compare_pair_runs(str(equal_dir), str(sample_dir), require_formal=False)
    assert pair["status"] == "VALID"
    assert pair["valid"] is True
    changed = list(rows)
    changed[0] = dict(changed[0], train_loss=0.2500002)
    write_rows_csv(str(sample_dir / "batch_trace.csv"), changed, fields)
    invalid = compare_pair_runs(str(equal_dir), str(sample_dir),
                                compare_losses=True, require_formal=False)
    assert invalid["status"] == "INVALID"
    assert invalid["valid"] is False
    assert invalid["max_abs_loss_diff"] > 1e-7
    # ON/OFF uses the same objective and explicitly ignores the expected
    # instrumentation-only is_probe column difference.
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    off_rows = [dict(row) for row in rows]
    on_rows = [dict(row) for row in rows]
    off_rows[0]["is_probe"], on_rows[0]["is_probe"] = 0, 1
    write_rows_csv(str(off_dir / "batch_trace.csv"), off_rows, fields)
    write_rows_csv(str(on_dir / "batch_trace.csv"), on_rows, fields)
    _write_comparison_identity(off_dir, instrument=0)
    _write_comparison_identity(on_dir, instrument=1)
    _write_metrics_for_comparison(off_dir)
    _write_metrics_for_comparison(on_dir)
    assert compare_on_off_runs(str(off_dir), str(on_dir))["status"] == "VALID"

    changed_fingerprint = list(rows)
    changed_fingerprint[0] = dict(changed_fingerprint[0], batch_fingerprint="different")
    write_rows_csv(str(sample_dir / "batch_trace.csv"), changed_fingerprint, fields)
    invalid_fingerprint = compare_pair_runs(str(equal_dir), str(sample_dir),
                                            require_formal=False)
    assert invalid_fingerprint["status"] == "INVALID"
    assert invalid_fingerprint["valid"] is False

    pending = compare_pair_runs(str(equal_dir), str(tmp_path / "missing"),
                                require_formal=False)
    assert pending["status"] == "PENDING"
    assert pending["valid"] is None

    # A legal early-stop tail is censored: compare the shared epoch/step
    # support and record both arm lengths rather than invalidating the pair.
    long_dir = tmp_path / "long"
    short_dir = tmp_path / "short"
    long_dir.mkdir()
    short_dir.mkdir()
    long_rows = []
    for epoch in (0, 1):
        long_rows.append(dict(rows[0], epoch=epoch, batch_fingerprint=f"fp{epoch}"))
    short_rows = [dict(rows[0], epoch=0, batch_fingerprint="fp0")]
    write_rows_csv(str(long_dir / "batch_trace.csv"), long_rows, fields)
    write_rows_csv(str(short_dir / "batch_trace.csv"), short_rows, fields)
    _write_comparison_identity(long_dir, observed_epochs=[0, 1])
    _write_comparison_identity(short_dir, observed_epochs=[0], stopped_epoch=0,
                               censored_after_epoch=1)
    early = compare_pair_runs(str(long_dir), str(short_dir), require_formal=False)
    assert early["status"] == "VALID"
    assert early["length_difference_allowed"] is True
    assert early["equal_trace_row_count"] == 2
    assert early["sample_trace_row_count"] == 1
    assert early["sample_stopped_epoch"] == 0

    # A mismatch on the common support remains invalid even if the tail is
    # legally censored.
    short_rows[0]["batch_fingerprint"] = "wrong"
    write_rows_csv(str(short_dir / "batch_trace.csv"), short_rows, fields)
    assert compare_pair_runs(str(long_dir), str(short_dir), require_formal=False)["status"] == "INVALID"

    # Identity differences cannot be hidden by matching fingerprints.
    _write_comparison_identity(sample_dir, weighting="sample", initial_hash="other")
    write_rows_csv(str(sample_dir / "batch_trace.csv"), rows, fields)
    identity_bad = compare_pair_runs(str(equal_dir), str(sample_dir), require_formal=False)
    assert identity_bad["status"] == "INVALID"
    assert any("initial_parameter_state_hash" in str(item)
               for item in identity_bad["mismatches"])

    inventory = collect_nonfinite_inventory({
        "gradient_step.csv": [{
            "epoch": 0, "parameter_group": "architecture",
            "actual_update_alignment": float("nan"),
        }],
        "gradient_epoch.csv": [{
            "epoch": 0, "parameter_group": "domain_specific",
            "variance_scope": "norm_scalar", "gradient_SNR": float("nan"),
        }],
    })
    assert len(inventory) == 2
    reasons = {entry["reason"] for entry in inventory}
    assert "architecture_group_has_no_main_optimizer_update" in reasons
    assert "vector_snr_not_defined_for_norm_scalar_variance" in reasons


if __name__ == "__main__":
    test_weighting_coefficients_and_scale()
    test_parameter_partition_is_explicit_and_disjoint()
    test_contribution_projection_and_welford()
    test_masked_terms_and_step_schema()
    print("[PASS] Stage E1 instrumentation unit tests")
