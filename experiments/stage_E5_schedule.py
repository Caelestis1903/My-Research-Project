"""Closed schedule registry and audit helpers for Stage E5.

The E5 experiment changes only the exponent used by the E3 contribution
normalisation.  This module is deliberately independent from the training
entry point so that the schedule can be tested with a deterministic toy loss
and audited before a run is launched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from stage_E3_weighting import ROLE_MAPS, EXPECTED_SEEDS


E5_EPOCHS = 18
E5_EPOCH_INDICES: Tuple[int, ...] = tuple(range(E5_EPOCHS))
E5_ARM_KEYS: Tuple[str, ...] = (
    "static_sample",
    "static_interior",
    "static_matched",
    "forward",
    "reverse",
)
E5_STATIC_ALPHAS: Dict[str, float] = {
    "static_sample": 1.0,
    "static_interior": 0.75,
    "static_matched": 0.875,
}
E5_FORMAL_SETTING = "d0_large"
E5_FORMAL_ROLE_SIZES: Dict[str, int] = {"L": 150000, "M": 34500, "S": 3000}
E5_FORMAL_PROTOCOL: Dict[str, object] = {
    "stage": "E5",
    "dataset": "MovieLens-1M",
    "setting": E5_FORMAL_SETTING,
    "epoch": E5_EPOCHS,
    "fixed_epoch_budget": True,
    "batch_size": 40960,
    "learning_rate": 1.0e-2,
    "learning_rate_darts": 1.0e-3,
    "weight_decay": 1.0e-5,
    "expert_num": 4,
    "probe_interval": 1,
    "instrument": 1,
    "role_sizes": "150000,34500,3000",
    "role_sizes_parsed": dict(E5_FORMAL_ROLE_SIZES),
    "fix_optim_grouping": 1,
    "earlystop_patience": 10,
    "backbone": "mdmoe",
    "scheduler": None,
    "task": "all",
    "domain": "all",
    "sampling": "DataLoader(shuffle=True,num_workers=0,drop_last=False)",
    "schedule_version": "E5-v1",
    "schedule_registry": list(E5_ARM_KEYS),
}


def _check_arm(arm_key: str) -> str:
    value = str(arm_key)
    if value not in E5_ARM_KEYS:
        raise ValueError("arm_key must be one of {}".format(list(E5_ARM_KEYS)))
    return value


def _check_epoch(epoch: int) -> int:
    try:
        value = int(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError("epoch must be an integer in 0..17") from exc
    if value != epoch or value not in E5_EPOCH_INDICES:
        raise ValueError("epoch must be an integer in 0..17")
    return value


def phase_for_epoch(epoch: int) -> str:
    """Return the preregistered early/middle/late label."""

    value = _check_epoch(epoch)
    if value <= 5:
        return "early"
    if value <= 11:
        return "middle"
    return "late"


def arm_alpha(arm_key: str, epoch: int) -> float:
    """Return the exact exponent for one registered arm and epoch."""

    key = _check_arm(arm_key)
    value = _check_epoch(epoch)
    if key in E5_STATIC_ALPHAS:
        return float(E5_STATIC_ALPHAS[key])
    u = float(value) / float(E5_EPOCHS - 1)
    if key == "forward":
        return 0.75 + 0.25 * u
    if key == "reverse":
        return 1.0 - 0.25 * u
    raise AssertionError("unreachable arm registry branch")


def alpha_trajectory(arm_key: str, epochs: int = E5_EPOCHS) -> List[float]:
    """Return the registered trajectory; shorter trajectories are smoke-only."""

    _check_arm(arm_key)
    count = int(epochs)
    if count <= 0 or count > E5_EPOCHS:
        raise ValueError("epochs must be in 1..{}".format(E5_EPOCHS))
    return [arm_alpha(arm_key, epoch) for epoch in range(count)]


def schedule_formula(arm_key: str) -> str:
    key = _check_arm(arm_key)
    if key in E5_STATIC_ALPHAS:
        return "alpha(e)={:.3f}".format(E5_STATIC_ALPHAS[key])
    if key == "forward":
        return "alpha(e)=0.750+0.250*(e/17)"
    return "alpha(e)=1.000-0.250*(e/17)"


def _normalise_shares(domain_shares: Sequence[float]) -> np.ndarray:
    shares = np.asarray(domain_shares, dtype=np.float64)
    if shares.ndim != 1 or len(shares) == 0:
        raise ValueError("domain_shares must be a non-empty vector")
    if not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("domain_shares must be finite and strictly positive")
    total = float(shares.sum())
    if total <= 0.0:
        raise ValueError("domain_shares must have positive sum")
    return shares / total


def scheduled_coefficients(
    arm_key: str,
    epoch: int,
    domain_shares: Sequence[float],
    task_num: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute E3-normalised coefficients for the current scheduled alpha.

    E3's public helper intentionally accepts only its static grid.  Forward
    and reverse contain intermediate values, so the same formula is evaluated
    here without imposing a new alpha grid.
    """

    alpha = arm_alpha(arm_key, epoch)
    shares = _normalise_shares(domain_shares)
    tasks = int(task_num)
    if tasks <= 0:
        raise ValueError("task_num must be positive")
    powers = np.power(shares, alpha)
    per_domain = len(shares) * powers / float(powers.sum())
    return torch.as_tensor(np.tile(per_domain, tasks), dtype=dtype, device=device)


def scheduled_coefficients_numpy(
    arm_key: str, epoch: int, domain_shares: Sequence[float], task_num: int = 2
) -> np.ndarray:
    return scheduled_coefficients(
        arm_key, epoch, domain_shares, task_num, dtype=torch.float64
    ).detach().cpu().numpy()


def validate_coefficients(
    arm_key: str,
    epoch: int,
    domain_shares: Sequence[float],
    coefficients: Sequence[float],
    task_num: int = 2,
    *,
    atol: float = 1.0e-8,
) -> Dict[str, object]:
    expected = scheduled_coefficients_numpy(arm_key, epoch, domain_shares, task_num)
    if isinstance(coefficients, torch.Tensor):
        actual = coefficients.detach().cpu().numpy().reshape(-1).astype(np.float64)
    else:
        actual = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError("scheduled coefficients do not match E3 normalisation")
    domain_num = len(_normalise_shares(domain_shares))
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise AssertionError("scheduled coefficients must be finite and positive")
    expected_sum = float(domain_num * int(task_num))
    if not np.isclose(float(actual.sum()), expected_sum, atol=1.0e-7, rtol=0):
        raise AssertionError("six-cell coefficient sum is not domain_num*task_num")
    return {
        "arm_key": _check_arm(arm_key),
        "epoch": _check_epoch(epoch),
        "alpha": arm_alpha(arm_key, epoch),
        "expected": expected.tolist(),
        "actual": actual.tolist(),
        "max_abs_error": float(np.max(np.abs(expected - actual))) if actual.size else 0.0,
        "cell_sum": float(actual.sum()),
    }


def exposure_table(
    arm_key: str,
    domain_shares: Sequence[float],
    task_num: int = 2,
    *,
    steps_per_epoch: Optional[Sequence[int] | int] = None,
    epochs: int = E5_EPOCHS,
) -> Dict[str, object]:
    """Compute alpha/coefficient integrals used by the audit and report."""

    key = _check_arm(arm_key)
    count = int(epochs)
    if count <= 0 or count > E5_EPOCHS:
        raise ValueError("epochs must be in 1..{}".format(E5_EPOCHS))
    shares = _normalise_shares(domain_shares)
    coefficients = [scheduled_coefficients_numpy(key, e, shares, task_num) for e in range(count)]
    if steps_per_epoch is None:
        steps = [1 for _ in range(count)]
    elif isinstance(steps_per_epoch, (int, np.integer)):
        steps = [int(steps_per_epoch) for _ in range(count)]
    else:
        steps = [int(item) for item in steps_per_epoch]
        if len(steps) != count:
            raise ValueError("steps_per_epoch must contain {} values".format(count))
    if any(item <= 0 for item in steps):
        raise ValueError("steps_per_epoch entries must be positive")
    matrix = np.asarray(coefficients, dtype=np.float64)
    alpha_values = np.asarray(alpha_trajectory(key, count), dtype=np.float64)
    weighted = matrix * np.asarray(steps, dtype=np.float64)[:, None]
    domain_num = len(shares)
    task_count = int(task_num)
    reshaped = matrix.reshape(count, task_count, domain_num)
    weighted_reshaped = weighted.reshape(count, task_count, domain_num)
    # Each task receives the same c_d.  The primary domain integral reports
    # that one coefficient; the all-task aggregate is retained separately so
    # neither interpretation is silently conflated.
    per_domain = reshaped[:, 0, :]
    weighted_domain = weighted_reshaped[:, 0, :]
    per_domain_all_tasks = reshaped.sum(axis=1)
    weighted_domain_all_tasks = weighted_reshaped.sum(axis=1)
    return {
        "arm_key": key,
        "epochs": count,
        "alpha_sum": float(alpha_values.sum()),
        "alpha_mean": float(alpha_values.mean()),
        "alpha_values": alpha_values.tolist(),
        "coefficients_by_epoch": matrix.tolist(),
        "coefficient_integral_by_cell": matrix.sum(axis=0).tolist(),
        "coefficient_integral_by_domain": per_domain.sum(axis=0).tolist(),
        "coefficient_integral_by_domain_all_tasks": per_domain_all_tasks.sum(axis=0).tolist(),
        "step_weighted_coefficient_integral_by_cell": weighted.sum(axis=0).tolist(),
        "step_weighted_coefficient_integral_by_domain": weighted_domain.sum(axis=0).tolist(),
        "step_weighted_coefficient_integral_by_domain_all_tasks": weighted_domain_all_tasks.sum(axis=0).tolist(),
        "steps_per_epoch": steps,
        "total_main_steps": int(sum(steps)),
    }


def schedule_registry_hash() -> str:
    payload = {
        key: {"formula": schedule_formula(key), "alpha": alpha_trajectory(key)}
        for key in E5_ARM_KEYS
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_registry() -> Dict[str, object]:
    """Validate preregistered boundary and exposure contracts."""

    trajectories = {key: alpha_trajectory(key) for key in E5_ARM_KEYS}
    if trajectories["forward"][0] != 0.75 or trajectories["forward"][-1] != 1.0:
        raise AssertionError("forward schedule boundary mismatch")
    if trajectories["reverse"][0] != 1.0 or trajectories["reverse"][-1] != 0.75:
        raise AssertionError("reverse schedule boundary mismatch")
    for key in ("forward", "reverse"):
        if not math.isclose(float(np.mean(trajectories[key])), 0.875, abs_tol=1.0e-12):
            raise AssertionError("{} mean alpha is not 0.875".format(key))
    if any(abs(trajectories["forward"][i] + trajectories["reverse"][i] - 1.75) > 1.0e-12
           for i in E5_EPOCH_INDICES):
        raise AssertionError("forward/reverse are not complementary")
    return {
        "arms": list(E5_ARM_KEYS),
        "epochs": E5_EPOCHS,
        "trajectories": trajectories,
        "registry_hash": schedule_registry_hash(),
    }


def expected_matrix_tasks(raw_root: str) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for arm_key in E5_ARM_KEYS:
        for seed in EXPECTED_SEEDS:
            run_dir = os.path.join(os.path.abspath(raw_root), arm_key, E5_FORMAL_SETTING,
                                    "seed{}".format(int(seed)))
            tasks.append({
                "stage": "E5",
                "arm_key": arm_key,
                "setting": E5_FORMAL_SETTING,
                "seed": int(seed),
                "run_dir": run_dir,
                "stdout_log": os.path.join(run_dir, "stdout.log"),
            })
    return tasks


def validate_matrix_tasks(tasks: Sequence[Mapping[str, object]]) -> Tuple[bool, List[str]]:
    expected = {(arm, E5_FORMAL_SETTING, int(seed))
                for arm in E5_ARM_KEYS for seed in EXPECTED_SEEDS}
    actual = {(str(task.get("arm_key", "")), str(task.get("setting", "")),
               int(task.get("seed", -1))) for task in tasks}
    reasons: List[str] = []
    if len(tasks) != len(expected):
        reasons.append("task_count:{}".format(len(tasks)))
    if actual != expected:
        reasons.append("task_identity_mismatch")
    return not reasons, reasons


def compare_e5_block(run_dirs: Mapping[str, str], *, require_formal: bool = False) -> Dict[str, object]:
    """Compare one seed's five arms on identity and ordered batch support."""

    import csv

    missing = [key for key in E5_ARM_KEYS if not os.path.isdir(str(run_dirs.get(key, "")))]
    if missing:
        return {"status": "PENDING", "valid": None, "missing_arm_keys": missing}

    def read_json(path: str) -> Dict[str, object]:
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def read_csv(path: str) -> List[Dict[str, str]]:
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return []

    manifests = {key: read_json(os.path.join(str(run_dirs[key]), "run_manifest.json"))
                 for key in E5_ARM_KEYS}
    configs = {key: read_json(os.path.join(str(run_dirs[key]), "config.json"))
              for key in E5_ARM_KEYS}
    reasons: List[Dict[str, object]] = []
    if any(not manifests[key] or not configs[key] for key in E5_ARM_KEYS):
        return {"status": "INVALID", "valid": False, "reason": "manifest_or_config_missing"}
    base = manifests[E5_ARM_KEYS[0]]
    identity_fields = (
        "seed", "setting", "dataset", "dataset_identity", "device_identity", "domain_num",
        "task_num", "role_by_domain", "role_sizes", "train_domain_counts", "domain_shares",
        "split_counts", "split_row_fingerprints", "positive_rates", "initial_parameter_state_hash",
        "partition", "model_identity", "formal_protocol", "oracle_audit",
    )
    for key in E5_ARM_KEYS:
        if manifests[key].get("arm_key") != key or configs[key].get("arm_key") != key:
            reasons.append({"arm_key": key, "reason": "arm_identity_mismatch"})
        if require_formal and (manifests[key].get("formal_valid") is not True
                               or configs[key].get("formal_valid") is not True):
            reasons.append({"arm_key": key, "reason": "formal_invalid"})
        for field in identity_fields:
            if json.dumps(manifests[key].get(field), sort_keys=True) != json.dumps(
                    base.get(field), sort_keys=True):
                reasons.append({"arm_key": key, "field": field, "reason": "identity_mismatch"})
        if manifests[key].get("observed_epochs") != list(E5_EPOCH_INDICES):
            reasons.append({"arm_key": key, "reason": "observed_epochs_invalid"})
        if configs[key].get("epoch") != E5_EPOCHS:
            reasons.append({"arm_key": key, "reason": "epoch_budget_invalid"})

    traces: Dict[str, Dict[Tuple[str, int, int], Mapping[str, str]]] = {}
    for key in E5_ARM_KEYS:
        rows = read_csv(os.path.join(str(run_dirs[key]), "batch_trace.csv"))
        traces[key] = {}
        for row in rows:
            kind = str(row.get("kind", ""))
            if kind not in ("train", "architecture"):
                continue
            try:
                support = (kind, int(row.get("epoch", -1)), int(row.get("step", -1)))
            except (TypeError, ValueError):
                reasons.append({"arm_key": key, "reason": "malformed_batch_trace"})
                continue
            if support in traces[key]:
                reasons.append({"arm_key": key, "support": support, "reason": "duplicate_batch_trace"})
            traces[key][support] = row
        if not rows:
            reasons.append({"arm_key": key, "reason": "batch_trace_missing_or_empty"})
    first_key = E5_ARM_KEYS[0]
    first_support = set(traces[first_key])
    for key in E5_ARM_KEYS[1:]:
        if set(traces[key]) != first_support:
            reasons.append({"arm_key": key, "reason": "batch_support_mismatch"})
    for support in sorted(first_support, key=lambda value: (value[1], value[0], value[2])):
        reference = traces[first_key][support]
        for key in E5_ARM_KEYS[1:]:
            candidate = traces[key].get(support, {})
            for field in ("batch_fingerprint", "batch_size", "domain_counts", "task_counts"):
                if str(candidate.get(field, "")) != str(reference.get(field, "")):
                    reasons.append({"arm_key": key, "support": support, "field": field,
                                    "reason": "batch_identity_mismatch"})
    return {
        "status": "VALID" if not reasons else "INVALID",
        "valid": not reasons,
        "reasons": reasons,
        "arm_keys": list(E5_ARM_KEYS),
        "observed_epochs": {key: manifests[key].get("observed_epochs", []) for key in E5_ARM_KEYS},
        "trace_count": {key: len(traces[key]) for key in E5_ARM_KEYS},
    }


__all__ = [
    "E5_EPOCHS", "E5_EPOCH_INDICES", "E5_ARM_KEYS", "E5_STATIC_ALPHAS",
    "E5_FORMAL_SETTING", "E5_FORMAL_ROLE_SIZES", "E5_FORMAL_PROTOCOL",
    "arm_alpha", "alpha_trajectory", "phase_for_epoch", "schedule_formula",
    "scheduled_coefficients", "scheduled_coefficients_numpy", "validate_coefficients",
    "exposure_table", "schedule_registry_hash", "validate_registry",
    "expected_matrix_tasks", "validate_matrix_tasks", "compare_e5_block",
    "ROLE_MAPS",
]
