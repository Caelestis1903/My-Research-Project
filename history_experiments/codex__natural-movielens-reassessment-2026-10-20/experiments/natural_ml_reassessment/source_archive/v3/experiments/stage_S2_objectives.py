"""Pure Stage S2 objective helpers.

The only mathematical change introduced by S2 is a fixed inverse-share
coefficient applied to unreduced per-example BCE.  Sampling remains the
ordinary shuffled concatenated loader inherited from S1.  Keeping this module
free of model/data/training code makes the no-double-weight and fixed-count
contracts straightforward to test.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
try:  # Keep protocol/reuse-audit imports usable in dependency-light Python.
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only without Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from stage_S2_protocol import coefficient_table_hash


OBJECTIVE_NAME = "equal_domain_task_mean_bce"
SAMPLE_OBJECTIVE_NAME = "direct_per_example_task_mean_bce"
OBJECTIVE_FORMULA = "mean_i( w[d_i] * mean_t(BCE(p_it,y_it)) )"
SAMPLE_FORMULA = "mean_i,t(BCE(p_it,y_it))"
TASK_NUM = 2
DOMAIN_NUM = 3


def _require_torch():
    if torch is None or F is None:
        raise RuntimeError("Stage S2 objective execution requires PyTorch")
    return torch, F


def _as_counts(train_counts: Mapping[int | str, int] | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(train_counts, Mapping):
        values = [train_counts.get(d, train_counts.get(str(d))) for d in range(DOMAIN_NUM)]
    else:
        values = list(train_counts)
    if len(values) != DOMAIN_NUM:
        raise ValueError("Equal objective requires exactly three domain counts")
    result = []
    for value in values:
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("domain counts must be integer") from exc
        if count <= 0:
            raise ValueError("domain counts must be positive")
        result.append(count)
    return tuple(result)  # type: ignore[return-value]


def compute_equal_coefficients(train_counts: Mapping[int | str, int] | Sequence[int],
                               *, task_num: int = TASK_NUM,
                               domain_num: int = DOMAIN_NUM) -> Dict[str, object]:
    """Create the immutable full-training coefficient table.

    ``p`` and ``w`` are represented as float64 metadata.  The training tensor
    is cast to the prediction dtype/device only at the point of application.
    No batch-local normalisation is performed here or by the runner.
    """

    if int(domain_num) != DOMAIN_NUM or int(task_num) != TASK_NUM:
        raise ValueError("S2 Equal is fixed at D=3,T=2")
    counts = _as_counts(train_counts)
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("domain count total must be positive")
    p = {str(d): float(counts[d]) / float(total) for d in range(DOMAIN_NUM)}
    w = {str(d): 1.0 / (float(DOMAIN_NUM) * p[str(d)]) for d in range(DOMAIN_NUM)}
    effective = {str(d): p[str(d)] * w[str(d)] for d in range(DOMAIN_NUM)}
    cell = {str(d): effective[str(d)] / float(TASK_NUM) for d in range(DOMAIN_NUM)}
    payload: Dict[str, object] = {
        "domain_num": DOMAIN_NUM,
        "task_num": TASK_NUM,
        "train_counts": {str(d): int(counts[d]) for d in range(DOMAIN_NUM)},
        "total_train_count": total,
        "domain_share_p": p,
        "loss_coefficient_w": w,
        "effective_domain_coefficient_a": effective,
        "effective_cell_coefficient": cell,
        "population_mean_weight": float(sum(p[str(d)] * w[str(d)] for d in range(DOMAIN_NUM))),
        "objective_name": OBJECTIVE_NAME,
        "objective_formula": OBJECTIVE_FORMULA,
        "sampling": "ordinary_shuffled_concatenated_loader",
        "double_weighting": False,
    }
    payload["coefficient_table_hash"] = coefficient_table_hash(payload)
    validate_coefficient_invariants(payload)
    return payload


def validate_coefficient_invariants(table: Mapping[str, object], *, atol: float = 1e-12) -> None:
    """Fail closed on missing, nonfinite, or double-weighted coefficients."""

    if int(table.get("domain_num", -1)) != DOMAIN_NUM or int(table.get("task_num", -1)) != TASK_NUM:
        raise ValueError("coefficient table has unexpected D/T")
    counts = table.get("train_counts")
    p = table.get("domain_share_p")
    w = table.get("loss_coefficient_w")
    a = table.get("effective_domain_coefficient_a")
    cell = table.get("effective_cell_coefficient")
    if not all(isinstance(value, Mapping) for value in (counts, p, w, a, cell)):
        raise ValueError("coefficient table is incomplete")
    total = sum(int(counts[str(d)]) for d in range(DOMAIN_NUM))  # type: ignore[index]
    if total <= 0:
        raise ValueError("coefficient table has no training examples")
    for d in range(DOMAIN_NUM):
        pd = float(p[str(d)])  # type: ignore[index]
        wd = float(w[str(d)])  # type: ignore[index]
        ad = float(a[str(d)])  # type: ignore[index]
        cd = float(cell[str(d)])  # type: ignore[index]
        if not all(math.isfinite(value) for value in (pd, wd, ad, cd)):
            raise ValueError("coefficient table contains nonfinite value")
        if abs(pd - int(counts[str(d)]) / float(total)) > atol:  # type: ignore[index]
            raise ValueError("domain share is not derived from immutable counts")
        if abs(pd * wd - (1.0 / DOMAIN_NUM)) > atol:
            raise ValueError("Equal effective domain coefficient is not 1/3")
        if abs(ad - pd * wd) > atol:
            raise ValueError("effective domain coefficient mismatch")
        if abs(cd - ad / TASK_NUM) > atol:
            raise ValueError("effective cell coefficient mismatch")
    if abs(sum(float(p[str(d)]) for d in range(DOMAIN_NUM)) - 1.0) > atol:  # type: ignore[index]
        raise ValueError("domain shares do not sum to one")
    if abs(float(table.get("population_mean_weight", float("nan"))) - 1.0) > atol:
        raise ValueError("population mean Equal weight is not one")
    if table.get("double_weighting") is not False:
        raise ValueError("double weighting is explicitly forbidden")
    declared_hash = table.get("coefficient_table_hash")
    if declared_hash:
        actual = coefficient_table_hash(table)
        if str(declared_hash) != actual:
            raise ValueError("coefficient table hash mismatch")


def _validate_batch(prediction: torch.Tensor, target: torch.Tensor,
                    domain_ids: torch.Tensor) -> None:
    _require_torch()
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction/target must have matching [B,T] shape")
    if prediction.shape[1] != TASK_NUM:
        raise ValueError("S2 objective requires two tasks")
    if domain_ids.ndim != 1 or int(domain_ids.shape[0]) != int(prediction.shape[0]):
        raise ValueError("domain IDs must have shape [B]")
    if prediction.shape[0] == 0:
        raise ValueError("empty objective batch")
    if not bool(torch.isfinite(prediction).all().detach().cpu()) \
            or not bool(torch.isfinite(target).all().detach().cpu()):
        raise FloatingPointError("objective prediction/target is non-finite")
    if not bool(((domain_ids >= 0) & (domain_ids < DOMAIN_NUM)).all().detach().cpu()):
        raise ValueError("unexpected domain ID in objective batch")


def sample_objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The frozen S1 Sample objective: unweighted per-example/task mean BCE."""

    _, functional = _require_torch()
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != TASK_NUM:
        raise ValueError("Sample objective requires matching [B,2] tensors")
    if prediction.shape[0] == 0:
        raise ValueError("empty objective batch")
    return functional.binary_cross_entropy(prediction, target.float(), reduction="mean")


def equal_objective(prediction: torch.Tensor, target: torch.Tensor,
                    domain_ids: torch.Tensor,
                    coefficient_table: Mapping[str, object] | None = None,
                    train_counts: Mapping[int | str, int] | Sequence[int] | None = None,
                    *, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Apply fixed full-dataset Equal weights exactly once.

    ``weights`` is an internal/testing escape hatch for a precomputed tensor;
    it is never combined with domain shares.  Production calls pass the
    immutable coefficient table and therefore cannot accidentally use
    per-batch shares.
    """

    torch_module, functional = _require_torch()
    _validate_batch(prediction, target, domain_ids)
    if coefficient_table is None:
        if train_counts is None:
            raise ValueError("Equal objective needs a coefficient table or counts")
        coefficient_table = compute_equal_coefficients(train_counts)
    else:
        validate_coefficient_invariants(coefficient_table)
    if weights is None:
        weight_values = coefficient_table["loss_coefficient_w"]  # type: ignore[index]
        weight_vector = torch_module.tensor(
            [float(weight_values[str(d)]) for d in range(DOMAIN_NUM)],  # type: ignore[index]
            dtype=prediction.dtype, device=prediction.device)
        weights = weight_vector[domain_ids.to(device=prediction.device, dtype=torch.long)]
    else:
        if weights.ndim != 1 or tuple(weights.shape) != (prediction.shape[0],):
            raise ValueError("per-example Equal weights must have shape [B]")
        if weights.device != prediction.device:
            weights = weights.to(device=prediction.device)
        if not bool(torch.isfinite(weights).all().detach().cpu()):
            raise FloatingPointError("Equal weight tensor is non-finite")
    unreduced = functional.binary_cross_entropy(prediction, target.float(), reduction="none")
    per_example = unreduced.mean(dim=1)
    result = (per_example * weights).mean()
    if not bool(torch.isfinite(result).detach().cpu()):
        raise FloatingPointError("Equal objective is non-finite")
    return result


def batch_weight_audit(domain_ids: torch.Tensor,
                       coefficient_table: Mapping[str, object]) -> Dict[str, object]:
    """Return logging-only batch exposure and weighted-mass diagnostics."""

    if domain_ids.ndim != 1 or domain_ids.numel() == 0:
        raise ValueError("cannot audit an empty batch")
    validate_coefficient_invariants(coefficient_table)
    ids = domain_ids.detach().cpu().numpy().astype(np.int64)
    if np.any((ids < 0) | (ids >= DOMAIN_NUM)):
        raise ValueError("unexpected domain ID in batch audit")
    counts = [int(np.sum(ids == d)) for d in range(DOMAIN_NUM)]
    size = int(len(ids))
    weights = coefficient_table["loss_coefficient_w"]  # type: ignore[index]
    weighted_mass = {str(d): float(counts[d]) * float(weights[str(d)]) for d in range(DOMAIN_NUM)}  # type: ignore[index]
    return {
        "raw_domain_counts": counts,
        "raw_domain_fractions": [float(value) / float(size) for value in counts],
        "weighted_mass_by_domain": weighted_mass,
        "total_weighted_mass": float(sum(weighted_mass.values())),
        "batch_size": size,
        "coefficient_table_hash": str(coefficient_table["coefficient_table_hash"]),
    }


def objective_oracle(prediction: torch.Tensor, target: torch.Tensor,
                     domain_ids: torch.Tensor,
                     train_counts: Mapping[int | str, int] | Sequence[int]) -> Dict[str, object]:
    """Fixed-prediction Sample/Equal equivalence oracle for tests and smokes."""

    torch_module, functional = _require_torch()
    table = compute_equal_coefficients(train_counts)
    sample = sample_objective(prediction, target)
    equal = equal_objective(prediction, target, domain_ids, table)
    unreduced = functional.binary_cross_entropy(prediction, target.float(), reduction="none")
    ids = domain_ids.to(device=prediction.device, dtype=torch.long)
    cell_means = []
    sample_expected = prediction.new_zeros(())
    equal_expected = prediction.new_zeros(())
    shares = table["domain_share_p"]  # type: ignore[index]
    for domain in range(DOMAIN_NUM):
        mask = ids == domain
        if not bool(mask.any()):
            continue
        cell = unreduced[mask].mean(dim=0).mean()
        cell_means.append(float(cell.detach().cpu()))
        sample_expected = sample_expected + float(shares[str(domain)]) * cell  # type: ignore[index]
        equal_expected = equal_expected + cell / float(DOMAIN_NUM)
    return {
        "sample": float(sample.detach().cpu()),
        "sample_expected": float(sample_expected.detach().cpu()),
        "equal": float(equal.detach().cpu()),
        "equal_expected": float(equal_expected.detach().cpu()),
        "sample_equivalent": bool(torch.allclose(sample, sample_expected, atol=1e-7, rtol=1e-7)),
        "equal_equivalent": bool(torch.allclose(equal, equal_expected, atol=1e-7, rtol=1e-7)),
        "effective_domain_coefficients": [float(table["effective_domain_coefficient_a"][str(d)])  # type: ignore[index]
                                           for d in range(DOMAIN_NUM)],
        "population_mean_weight": float(table["population_mean_weight"]),
        "cell_means": cell_means,
    }


def assert_rejects_doubled_domain_share(prediction: torch.Tensor, target: torch.Tensor,
                                        domain_ids: torch.Tensor,
                                        train_counts: Mapping[int | str, int] | Sequence[int]) -> None:
    """Negative fixture: applying p twice must not pass the Equal oracle."""

    table = compute_equal_coefficients(train_counts)
    shares = table["domain_share_p"]  # type: ignore[index]
    doubled = torch.tensor([2.0 * float(shares[str(d)]) for d in range(DOMAIN_NUM)],
                           dtype=prediction.dtype, device=prediction.device)
    torch_module, _ = _require_torch()
    observed = equal_objective(prediction, target, domain_ids, table, weights=doubled[
        domain_ids.to(device=prediction.device, dtype=torch.long)])
    expected = equal_objective(prediction, target, domain_ids, table)
    if bool(torch.allclose(observed, expected, atol=1e-7, rtol=1e-7)):
        raise AssertionError("deliberately doubled p_d fixture was accepted")


# Descriptive aliases used by focused tests/review tooling.
fixed_equal_objective = equal_objective
equal_domain_bce = equal_objective
build_coefficient_table = compute_equal_coefficients


__all__ = [
    "OBJECTIVE_NAME", "SAMPLE_OBJECTIVE_NAME", "OBJECTIVE_FORMULA", "SAMPLE_FORMULA",
    "compute_equal_coefficients", "build_coefficient_table", "validate_coefficient_invariants",
    "sample_objective", "equal_objective", "fixed_equal_objective", "equal_domain_bce",
    "batch_weight_audit", "objective_oracle", "assert_rejects_doubled_domain_share",
]
