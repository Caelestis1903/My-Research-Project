"""Instrumentation ON/OFF equivalence check on a deterministic tiny model.

This is intentionally a short CPU test.  It verifies the critical contract:
observer autograd calls leave the actual weighted backward/Adam update
unchanged and do not populate ``Parameter.grad`` before the real backward.
"""

import copy
import json
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
    build_parameter_partition,
    collect_cell_gradients,
    contribution_views,
    masked_bce_terms,
    weighting_coefficients,
    write_json,
    write_rows_csv,
)

from stage_E1_orchestrator import (  # noqa: E402
    MASTER_FIELDS,
    REQUIRED_ARTIFACTS,
    _append_master,
    _finalize_task,
    build_tasks,
    done_set,
)
from stage_E1_report import SUMMARY_FIELDS, collect  # noqa: E402


def _write_formal_run(task, *, fingerprint="fp", initial_hash="initial",
                      epoch=50, observed_epochs=None, formal_valid=True):
    """Create a complete synthetic formal run for resume/pair tests."""

    run_dir = task["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    if observed_epochs is None:
        observed_epochs = list(range(epoch))
    weighting = str(task["weighting"])
    setting = str(task["setting"])
    seed = int(task["seed"])
    partition = {
        "groups": {"shared": ["p"]}, "param_count": 1,
        "param_shapes": {"p": [1]}, "manifest_hash": "partition",
        "state_hash": initial_hash,
    }
    split = {name: {str(idx): 10 for idx in range(6)}
             for name in ("train", "validation", "test")}
    positives = {name: {str(idx): 0.5 for idx in range(6)}
                 for name in ("train", "validation", "test")}
    role_by_domain = {"0": "L", "1": "M", "2": "S"}
    manifest = {
        "stage": "E1", "status": "OK", "seed": seed,
        "setting": setting, "weighting": weighting,
        "formal_run_key": f"{weighting}|{setting}|{seed}",
        "dataset": "MovieLens-1M", "domain_num": 3, "task_num": 2,
        "role_by_domain": role_by_domain,
        "role_sizes": {"L": 150000, "M": 34500, "S": 3000},
        "train_domain_counts": {"0": 150000, "1": 34500, "2": 3000},
        "domain_shares": [150000 / 187500.0, 34500 / 187500.0, 3000 / 187500.0],
        "split_counts": split, "positive_rates": positives,
        "partition": partition,
        "initial_parameter_state_hash": initial_hash,
        "final_parameter_state_hash": "final",
        "observed_epochs": list(observed_epochs), "best_epoch": 0,
        "stopped_epoch": "", "censored_after_epoch": None,
        "formal_protocol": dict(FORMAL_PROTOCOL),
        "formal_valid": bool(formal_valid),
        "rng_restore_evidence": [{"epoch": 0, "scope": "test_eval",
                                   "before": "rng", "after": "rng", "restored": True}],
        "pair_status": "PENDING", "pair_valid": None,
    }
    config = dict(FORMAL_PROTOCOL)
    config.update({
        "epoch": epoch,
        "seed": seed, "setting": setting, "weighting": weighting,
        "loss_weighting": weighting, "run_dir": str(run_dir),
        "config_hash": f"config-{weighting}",
        "dataset_path": "/xzf/data/movielens",
        "oracle_root": "oracle", "stdout_log": None,
        "instrumentation": True, "formal_protocol": dict(FORMAL_PROTOCOL),
        "formal_valid": bool(formal_valid),
    })
    write_json(os.path.join(run_dir, "run_manifest.json"), manifest)
    write_json(os.path.join(run_dir, "config.json"), config)

    headers = {
        "metrics_epoch.csv": ["seed", "epoch", "domain", "task", "cell"],
        "gradient_step.csv": ["epoch", "step", "domain", "task", "cell",
                               "parameter_group", "domain_signed_contribution"],
        "gradient_epoch.csv": ["seed", "epoch", "domain", "task", "cell",
                                "parameter_group"],
        "gradient_cosine_epoch.csv": ["seed", "epoch", "domain_pair"],
        "gate_epoch.csv": ["seed", "epoch", "domain", "task", "cell",
                            "routing_scope", "routing_sample_count"],
        "batch_trace.csv": ["seed", "epoch", "step", "kind",
                             "batch_fingerprint", "train_loss"],
        "metrics_summary.csv": ["seed", "weighting", "setting", "domain", "task", "cell"],
    }
    metric_rows = [{"seed": seed, "epoch": current_epoch,
                    "domain": domain, "task": task_id,
                    "cell": task_id * 3 + domain}
                   for current_epoch in range(50)
                   for task_id in range(2) for domain in range(3)]
    batch_rows = [{"seed": seed, "epoch": current_epoch, "step": 0,
                  "kind": "train", "batch_fingerprint": f"{fingerprint}-{current_epoch}",
                  "train_loss": 0.5}
                 for current_epoch in range(50)]
    batch_rows.extend({"seed": seed, "epoch": current_epoch, "step": -1,
                       "kind": "architecture", "batch_fingerprint": f"arch-{current_epoch}",
                       "train_loss": 0.4}
                      for current_epoch in range(50))
    for name, fields in headers.items():
        rows = metric_rows if name == "metrics_epoch.csv" else batch_rows if name == "batch_trace.csv" else []
        write_rows_csv(os.path.join(run_dir, name), rows, fields)
    write_rows_csv(os.path.join(run_dir, "metrics_summary.csv"), [
        {"seed": seed, "weighting": weighting, "setting": setting,
         "domain": 0, "task": 0, "cell": 0}], headers["metrics_summary.csv"])
    with open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    write_json(os.path.join(run_dir, "nonfinite_inventory.json"), {
        "count": 0, "unexplained_count": 0, "entries": [],
    })


class TinyE1Model(torch.nn.Module):
    """Two-domain, one-task model with shared and cell-private parameters."""

    def __init__(self):
        super().__init__()
        self.shared_weight = torch.nn.Parameter(torch.tensor(0.2))
        self.shared_bias = torch.nn.Parameter(torch.tensor(-0.1))
        self.tower = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        with torch.no_grad():
            for tower in self.tower:
                tower.weight.fill_(0.5)
                tower.bias.fill_(0.0)

    def forward(self, x):
        values = x[0].float() * self.shared_weight + self.shared_bias
        outputs = []
        domain_ids = x[-1].long()
        for domain in range(2):
            transformed = self.tower[domain](values.reshape(-1, 1)).reshape(-1)
            outputs.append(torch.sigmoid(transformed))
        return torch.stack([torch.where(domain_ids == 0, outputs[0], outputs[1])], dim=1)


def _run(model, x, y, instrument):
    partition = build_parameter_partition(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    prediction = model(x)
    terms, counts = masked_bce_terms(prediction, y, x[-1], 2, 1, criterion)
    coeff = weighting_coefficients("sample", [0.75, 0.25], task_num=1)
    loss = (torch.stack(terms) * coeff).sum()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    if instrument:
        raw_cells, unused_cells = collect_cell_gradients(terms, model, partition, retain_graph=True)
        raw_shared = {idx: raw_cells[idx]["shared"] for idx in range(2)}
        view = contribution_views(raw_shared, coeff, domain_num=2, task_num=1)
        assert np.isclose(sum(view["normalized"].values()), 1.0)
        assert all(value.grad is None for value in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_snapshot = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    optimizer.step()
    after = {name: value.detach().clone() for name, value in model.named_parameters()}
    return float(loss.detach()), before, grad_snapshot, after, counts


def test_instrumentation_on_off_is_update_equivalent():
    torch.manual_seed(2022)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]])
    x[-1] = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([[0], [1], [1], [0]])
    base = TinyE1Model()
    off = copy.deepcopy(base)
    on = copy.deepcopy(base)
    off_out = _run(off, x, y, instrument=False)
    on_out = _run(on, x, y, instrument=True)
    assert off_out[4] == on_out[4]
    assert off_out[0] == on_out[0]
    for name in off.state_dict():
        assert torch.equal(off.state_dict()[name], on.state_dict()[name]), name
    for name in off_out[2]:
        assert torch.equal(off_out[2][name], on_out[2][name]), name


def test_resume_revalidates_stale_ok_and_propagates_failure(tmp_path):
    raw_root = tmp_path / "raw"
    task = build_tasks(str(raw_root))[0]
    _write_formal_run(task)
    master = tmp_path / "master.csv"
    _append_master(str(master), {
        "stage": "E1", "weighting": "equal", "setting": "d0_large", "seed": 2022,
        "status": "OK", "return_code": 0, "run_dir": task["run_dir"],
    })
    # A lone arm is not complete until its paired arm has a VALID comparison.
    assert done_set(str(master), str(raw_root)) == set()

    sample_task = build_tasks(str(raw_root))[12]
    _write_formal_run(sample_task)
    assert done_set(str(master), str(raw_root)) == {("equal", "d0_large", 2022)}

    # A completed pair is resumable only with the full 50-epoch protocol.
    _append_master(str(master), {
        "stage": "E1", "weighting": "sample", "setting": "d0_large", "seed": 2022,
        "status": "OK", "return_code": 0, "run_dir": sample_task["run_dir"],
    })
    assert done_set(str(master), str(raw_root)) == {
        ("equal", "d0_large", 2022), ("sample", "d0_large", 2022)
    }

    # Replacing one arm with a 3-epoch smoke output must invalidate resume.
    _write_formal_run(sample_task, epoch=3, observed_epochs=[0, 1, 2], formal_valid=False)
    assert done_set(str(master), str(raw_root)) == set()

    failed = _finalize_task(task, 7, 0.0, "child failed", str(tmp_path / "failed.csv"))
    assert failed["status"] == "FAILED"
    assert failed["return_code"] == 7


def test_report_enumerates_matrix_and_reads_split_routing(tmp_path):
    raw_root = tmp_path / "report_raw"
    run_dir = raw_root / "equal" / "d0_large" / "seed2022"
    run_dir.mkdir(parents=True)
    write_json(str(run_dir / "run_manifest.json"), {
        "stage": "E1", "status": "OK", "seed": 2022,
        "setting": "d0_large", "weighting": "equal",
    })
    metric_fields = ["seed", "epoch", "domain", "task", "cell", "weighting", "setting",
                     "physical_domain", "size_role", "domain_size", "domain_share",
                     "train_auc", "train_logloss", "validation_auc", "validation_logloss",
                     "test_auc", "test_logloss", "oracle_source"]
    write_rows_csv(str(run_dir / "metrics_epoch.csv"), [{
        "seed": 2022, "epoch": 0, "domain": 0, "task": 0, "cell": 0,
        "weighting": "equal", "setting": "d0_large", "physical_domain": 0,
        "size_role": "L", "domain_size": 10, "domain_share": 0.75,
        "train_auc": 0.60, "train_logloss": 0.50,
        "validation_auc": 0.61, "validation_logloss": 0.49,
        "test_auc": 0.62, "test_logloss": 0.48, "oracle_source": "missing",
    }], metric_fields)
    gradient_fields = ["seed", "epoch", "domain", "task", "cell", "weighting",
                       "parameter_group", "gradient_norm", "variance_scope"]
    write_rows_csv(str(run_dir / "gradient_epoch.csv"), [{
        "seed": 2022, "epoch": 0, "domain": 0, "task": 0, "cell": 0,
        "weighting": "equal", "parameter_group": "shared", "gradient_norm": 1.0,
        "variance_scope": "vector_trace",
    }], gradient_fields)
    gate_fields = ["seed", "weighting", "setting", "epoch", "domain", "task", "cell",
                   "routing_scope", "routing_sample_count", "gate_entropy",
                   "effective_expert_count", "top1_expert_frequency", "expert_0_mean_probability"]
    write_rows_csv(str(run_dir / "gate_epoch.csv"), [{
        "seed": 2022, "weighting": "equal", "setting": "d0_large", "epoch": 0,
        "domain": 0, "task": 0, "cell": 0, "routing_scope": "cell",
        "routing_sample_count": 10, "gate_entropy": 0.5,
        "effective_expert_count": 1.6, "top1_expert_frequency": 0.8,
        "expert_0_mean_probability": 0.7,
    }, {
        "seed": 2022, "weighting": "equal", "setting": "d0_large", "epoch": 0,
        "domain": 0, "task": -1, "cell": -1, "routing_scope": "domain_task_average",
        "routing_sample_count": 10, "gate_entropy": 0.55,
        "effective_expert_count": 1.73, "top1_expert_frequency": 0.75,
        "expert_0_mean_probability": 0.65,
    }], gate_fields)
    rows, manifest = collect(str(raw_root))
    assert manifest["expected_run_count"] == 24
    assert manifest["available_run_count"] == 1
    assert manifest["missing_run_count"] == 23
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert "gate_entropy" in SUMMARY_FIELDS
    assert any(row["gate_entropy"] == "0.5" for row in rows)
    assert any(row["domain_task_average_gate_entropy"] == "0.55" for row in rows)


def test_orchestrator_marks_mismatched_pair_invalid(tmp_path):
    raw_root = tmp_path / "pair_raw"
    tasks = build_tasks(str(raw_root))
    equal_task = tasks[0]
    sample_task = tasks[12]
    _write_formal_run(equal_task, fingerprint="equal_fp")
    _write_formal_run(sample_task, fingerprint="sample_fp")
    master = tmp_path / "pair_master.csv"
    _append_master(str(master), {
        "stage": "E1", "weighting": "equal", "setting": "d0_large", "seed": 2022,
        "status": "OK", "return_code": 0, "run_dir": equal_task["run_dir"],
    })
    result = _finalize_task(sample_task, 0, 0.0, "", str(master))
    assert result["status"] == "INVALID_PAIR"
    assert result["pair_status"] == "INVALID"
    assert result["pair_valid"] is False
    assert done_set(str(master), str(raw_root)) == set()
    equal_manifest = json.loads(open(os.path.join(equal_task["run_dir"],
                                                   "run_manifest.json"), encoding="utf-8").read())
    sample_manifest = json.loads(open(os.path.join(sample_task["run_dir"],
                                                    "run_manifest.json"), encoding="utf-8").read())
    assert equal_manifest["pair_valid"] is False
    assert sample_manifest["pair_valid"] is False
    assert equal_manifest["status"] == "INVALID_PAIR"
    assert sample_manifest["status"] == "INVALID_PAIR"

    # A rerun with a repaired counterpart must be able to clear the stale
    # INVALID_PAIR correction rather than forcing the pair to remain pending.
    _write_formal_run(sample_task, fingerprint="equal_fp")
    repaired = _finalize_task(sample_task, 0, 0.0, "", str(master))
    assert repaired["status"] == "OK"
    assert repaired["pair_status"] == "VALID"
    assert done_set(str(master), str(raw_root)) == {
        ("equal", "d0_large", 2022), ("sample", "d0_large", 2022)
    }


if __name__ == "__main__":
    test_instrumentation_on_off_is_update_equivalent()
    print("[PASS] Stage E1 instrumentation ON/OFF equivalence")
