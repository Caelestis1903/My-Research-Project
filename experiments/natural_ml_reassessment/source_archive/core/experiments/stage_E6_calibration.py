"""Algorithmic primitives for Stage E6 adaptive domain calibration.

The module is intentionally independent of the training entry point.  It owns
the closed six-arm registry, raw shared-gradient magnitude state, the
pre-registered EMA estimator, and the coefficient rule from the E6 design
specification.  No object in this module is a trainable parameter and no
function writes ``Parameter.grad``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from stage_E1_instrumentation import cell_index, iter_cells


E6_ARM_KEYS: Tuple[str, ...] = (
    "equal_static",
    "sample_static",
    "sample_ema_control",
    "adaptive_noema",
    "adaptive_ema",
    "adaptive_ema_uniform",
)
E6_FORMAL_SETTING = "d0_large"
E6_EXPECTED_SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
E6_ROLE_SIZES: Dict[str, int] = {"L": 150000, "M": 34500, "S": 3000}
E6_STEPS_PER_EPOCH = 5
E6_BETA = 2.0 ** (-1.0 / float(E6_STEPS_PER_EPOCH))
E6_FORMAL_PROTOCOL: Dict[str, object] = {
    "stage": "E6",
    "dataset": "MovieLens-1M",
    "setting": E6_FORMAL_SETTING,
    "epoch": 50,
    "batch_size": 40960,
    "learning_rate": 1.0e-2,
    "learning_rate_darts": 1.0e-3,
    "weight_decay": 1.0e-5,
    "expert_num": 4,
    "probe_interval": 1,
    "instrument": 1,
    "role_sizes": "150000,34500,3000",
    "role_sizes_parsed": dict(E6_ROLE_SIZES),
    "fix_optim_grouping": 1,
    "earlystop_patience": 10,
    "backbone": "mdmoe",
    "scheduler": None,
    "task": "all",
    "domain": "all",
    "sampling": "DataLoader(shuffle=True,num_workers=0,drop_last=False)",
    "controller_version": "E6-v1",
    "controller_inputs": ["train_p_d", "detached_shared_raw_gradient_magnitude"],
    "controller_formula": "c_d=D*(tau_d/(q_d+epsilon))/sum_j(tau_j/(q_j+epsilon))",
    "ema_half_life_epochs": 1,
    "ema_steps_per_epoch": E6_STEPS_PER_EPOCH,
    "ema_beta": E6_BETA,
    "arm_registry": list(E6_ARM_KEYS),
}

# The dictionary is deliberately explicit.  In particular, ``sample_ema_control``
# has an EMA statistic but does not feed that statistic back into its loss.
E6_ARM_REGISTRY: Dict[str, Dict[str, object]] = {
    "equal_static": {
        "coefficient_behavior": "equal_static",
        "dynamic_statistic": "none",
        "target": "data_share",
        "uses_ema": False,
        "feedback": False,
    },
    "sample_static": {
        "coefficient_behavior": "sample_static",
        "dynamic_statistic": "none",
        "target": "data_share",
        "uses_ema": False,
        "feedback": False,
    },
    "sample_ema_control": {
        "coefficient_behavior": "sample_static",
        "dynamic_statistic": "ema_raw_shared_magnitude",
        "target": "data_share",
        "uses_ema": True,
        "feedback": False,
    },
    "adaptive_noema": {
        "coefficient_behavior": "adaptive_closed_form",
        "dynamic_statistic": "raw_shared_magnitude",
        "target": "data_share",
        "uses_ema": False,
        "feedback": True,
    },
    "adaptive_ema": {
        "coefficient_behavior": "adaptive_closed_form",
        "dynamic_statistic": "ema_raw_shared_magnitude",
        "target": "data_share",
        "uses_ema": True,
        "feedback": True,
    },
    "adaptive_ema_uniform": {
        "coefficient_behavior": "adaptive_closed_form",
        "dynamic_statistic": "ema_raw_shared_magnitude",
        "target": "uniform_share",
        "uses_ema": True,
        "feedback": True,
    },
}


def validate_arm_key(arm_key: str) -> str:
    key = str(arm_key)
    if key not in E6_ARM_KEYS:
        raise ValueError("arm_key must be one of {}".format(list(E6_ARM_KEYS)))
    return key


def normalise_shares(domain_shares: Sequence[float]) -> np.ndarray:
    values = np.asarray(domain_shares, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("domain_shares must be finite and strictly positive")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("domain_shares must have a positive sum")
    return values / total


def target_shares(arm_key: str, domain_shares: Sequence[float]) -> np.ndarray:
    """Return the fixed target for the arm (data share or uniform share)."""

    key = validate_arm_key(arm_key)
    shares = normalise_shares(domain_shares)
    if key == "adaptive_ema_uniform":
        return np.full(len(shares), 1.0 / float(len(shares)), dtype=np.float64)
    return shares.copy()


def static_domain_coefficients(
    arm_key: str,
    domain_shares: Sequence[float],
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the registered fixed coefficient vector for static/control arms."""

    key = validate_arm_key(arm_key)
    shares = normalise_shares(domain_shares)
    domains = len(shares)
    if key == "equal_static":
        values = np.ones(domains, dtype=np.float64)
    elif key in ("sample_static", "sample_ema_control"):
        values = float(domains) * shares
    else:
        raise ValueError("{} has no static coefficient vector".format(key))
    return torch.as_tensor(np.tile(values, 2), dtype=dtype, device=device)


def static_coefficients(
    arm_key: str,
    domain_shares: Sequence[float],
    task_num: int = 2,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Static coefficient vector with task-major domain replication."""

    key = validate_arm_key(arm_key)
    shares = normalise_shares(domain_shares)
    domains = len(shares)
    if key == "equal_static":
        per_domain = np.ones(domains, dtype=np.float64)
    elif key in ("sample_static", "sample_ema_control"):
        per_domain = float(domains) * shares
    else:
        raise ValueError("{} has no static coefficient vector".format(key))
    tasks = int(task_num)
    if tasks <= 0:
        raise ValueError("task_num must be positive")
    return torch.as_tensor(np.tile(per_domain, tasks), dtype=dtype, device=device)


def raw_domain_magnitudes(
    raw_by_cell: Mapping[int, torch.Tensor],
    domain_num: int = 3,
    task_num: int = 2,
) -> torch.Tensor:
    """Compute ``r_d=sum_t ||g^S_dt||_2`` in task-major cell order.

    Inputs are detached by the caller or are detached again here.  The result
    is a detached float64 statistic and therefore cannot form a second-order
    graph when it is used to construct coefficients.
    """

    domains = int(domain_num)
    tasks = int(task_num)
    if domains <= 0 or tasks <= 0:
        raise ValueError("domain_num and task_num must be positive")
    result: List[torch.Tensor] = []
    for domain in range(domains):
        total: Optional[torch.Tensor] = None
        for task in range(tasks):
            idx = cell_index(task, domain, domains)
            if idx not in raw_by_cell:
                raise ValueError("raw gradients missing cell {}".format(idx))
            value = raw_by_cell[idx]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            value = value.detach()
            norm = torch.linalg.vector_norm(value).to(dtype=torch.float64)
            total = norm if total is None else total + norm
        if total is None:
            raise ValueError("no task gradients for domain {}".format(domain))
        result.append(total.detach())
    output = torch.stack(result).detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("raw shared domain magnitudes must be finite")
    if bool((output < 0).any()) or not bool((output > 0).any()):
        raise ValueError("raw shared domain magnitudes must be non-negative and nonzero")
    return output


def domain_magnitudes_from_cells(
    raw_by_cell: Mapping[int, torch.Tensor], domain_num: int = 3, task_num: int = 2
) -> torch.Tensor:
    """Backward-compatible descriptive alias for :func:`raw_domain_magnitudes`."""

    return raw_domain_magnitudes(raw_by_cell, domain_num, task_num)


def coefficient_adjusted_magnitude_share(
    raw_magnitude: Sequence[float] | torch.Tensor,
    coefficients: Sequence[float] | torch.Tensor,
    *,
    domain_num: Optional[int] = None,
    task_num: int = 2,
) -> torch.Tensor:
    """Return coefficient-adjusted domain magnitude shares."""

    raw = torch.as_tensor(raw_magnitude, dtype=torch.float64).reshape(-1).detach()
    coeff = torch.as_tensor(coefficients, dtype=torch.float64).reshape(-1).detach()
    tasks = int(task_num)
    if domain_num is None:
        if len(coeff) % tasks:
            raise ValueError("coefficient vector length is not divisible by task_num")
        domains = len(coeff) // tasks
    else:
        domains = int(domain_num)
    if len(raw) != domains or len(coeff) != domains * tasks:
        raise ValueError("raw magnitude/coefficient dimensions differ")
    per_domain = coeff.reshape(tasks, domains)[0]
    if not torch.isfinite(raw).all() or not torch.isfinite(coeff).all():
        raise ValueError("raw magnitude and coefficients must be finite")
    weighted = raw * per_domain.abs()
    total = weighted.sum()
    if float(total) <= 0.0:
        raise ValueError("weighted magnitude must be nonzero")
    return (weighted / total).detach()


def closed_form_coefficients(
    raw_magnitude: Sequence[float] | torch.Tensor,
    target: Sequence[float] | torch.Tensor,
    *,
    task_num: int = 2,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Compute the E6 detached ``c_d ∝ tau_d/(q_d+epsilon)`` rule.

    ``raw_magnitude`` is the q-state supplied by the caller: current raw
    magnitude for NoEMA, or detached EMA magnitude for EMA.  Epsilon follows
    the dtype-specific rule in the design specification.
    """

    q = torch.as_tensor(raw_magnitude).reshape(-1).detach()
    if not q.is_floating_point():
        q = q.to(dtype=torch.float64)
    target_tensor = torch.as_tensor(target, dtype=q.dtype, device=q.device).reshape(-1).detach()
    domains = len(q)
    tasks = int(task_num)
    if domains <= 0 or len(target_tensor) != domains or tasks <= 0:
        raise ValueError("q/target/task dimensions are invalid")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("q must be finite and non-negative")
    if not bool(torch.isfinite(target_tensor).all()) or bool((target_tensor <= 0).any()):
        raise ValueError("target must be finite and strictly positive")
    target_tensor = target_tensor / target_tensor.sum()
    if float(q.sum()) <= 0.0:
        raise ValueError("all raw shared magnitudes are zero")
    eps = float(torch.finfo(q.dtype).eps) * max(1.0, float(q.mean().detach().cpu()))
    epsilon = torch.as_tensor(eps, dtype=q.dtype, device=q.device)
    z = target_tensor / (q + epsilon)
    denominator = z.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ValueError("closed-form normalization is non-finite")
    per_domain = float(domains) * z / denominator
    coeff = torch.tile(per_domain, (tasks,)).detach()
    diagnostics = {
        "epsilon": eps,
        "q_dtype": str(q.dtype),
        "target": target_tensor.detach().cpu().tolist(),
        "q": q.detach().cpu().tolist(),
        "coefficient_sum_domain": float(per_domain.sum().detach().cpu()),
        "coefficient_sum_cell": float(coeff.sum().detach().cpu()),
        "coefficients": coeff.detach().cpu().tolist(),
    }
    return coeff, diagnostics


def calibrate_coefficients(
    raw_magnitude: Sequence[float] | torch.Tensor,
    target: Sequence[float] | torch.Tensor,
    *,
    task_num: int = 2,
) -> torch.Tensor:
    """Small public alias used by focused tests and the runner."""

    return closed_form_coefficients(raw_magnitude, target, task_num=task_num)[0]


@dataclass
class EMAMagnitudeState:
    """Detached float64 EMA of domain raw shared-gradient magnitudes."""

    beta: float = E6_BETA
    value: Optional[torch.Tensor] = None
    update_count: int = 0

    def __post_init__(self) -> None:
        self.beta = float(self.beta)
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("EMA beta must be in [0,1)")
        if self.value is not None:
            self.value = torch.as_tensor(self.value, dtype=torch.float64).detach().clone()

    def update(self, raw_magnitude: Sequence[float] | torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(raw_magnitude, dtype=torch.float64).reshape(-1).detach()
        if raw.numel() == 0 or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
            raise ValueError("EMA raw magnitudes must be finite and non-negative")
        if self.value is None:
            self.value = raw.clone()
        else:
            if self.value.shape != raw.shape:
                raise ValueError("EMA state and raw magnitude shapes differ")
            self.value = (self.beta * self.value + (1.0 - self.beta) * raw).detach()
        self.update_count += 1
        return self.value.detach().clone()

    @property
    def initialized(self) -> bool:
        return self.value is not None


def ema_update(
    previous: Optional[Sequence[float] | torch.Tensor],
    current: Sequence[float] | torch.Tensor,
    beta: float = E6_BETA,
) -> torch.Tensor:
    """Functional EMA helper with first-observation initialization."""

    state = EMAMagnitudeState(beta=beta)
    if previous is not None:
        state.value = torch.as_tensor(previous, dtype=torch.float64).reshape(-1).detach().clone()
    return state.update(current)


def coefficient_variation(
    previous: Optional[torch.Tensor], current: torch.Tensor, domain_num: int,
    task_num: int = 2,
) -> float:
    """Return one transition of the preregistered domain-only ``V_c``.

    E6 stores the same domain coefficient once for every task.  The stability
    metric is defined on the D-dimensional domain vector, not on the
    task-replicated D*T cell vector.  The first observation has no transition
    and therefore returns NaN so callers can exclude it from ``K-1`` means.
    """

    if previous is None:
        return float("nan")
    left = previous.detach().reshape(-1)
    right = current.detach().reshape(-1)
    domains = int(domain_num)
    tasks = int(task_num)
    if domains <= 0 or tasks <= 0:
        raise ValueError("domain_num and task_num must be positive")
    if left.shape != right.shape:
        raise ValueError("coefficient vectors have different shapes")
    if len(left) == domains:
        left_domain, right_domain = left, right
    elif len(left) == domains * tasks:
        # Task-major replication is intentional; only one domain copy enters
        # the D-dimensional variation formula.
        left_domain = left.reshape(tasks, domains)[0]
        right_domain = right.reshape(tasks, domains)[0]
    else:
        raise ValueError("coefficient vector does not cover D or D*T entries")
    return float(torch.sum(torch.abs(right_domain - left_domain)).cpu()) / float(domains)


def calibration_metrics(
    raw_magnitude: Sequence[float] | torch.Tensor,
    coefficients: Sequence[float] | torch.Tensor,
    domain_shares: Sequence[float],
    target: Sequence[float] | torch.Tensor,
    *,
    task_num: int = 2,
) -> Dict[str, object]:
    """Return raw/weighted share and target/data-share errors for one step."""

    shares = normalise_shares(domain_shares)
    raw = torch.as_tensor(raw_magnitude, dtype=torch.float64).reshape(-1).detach()
    target_np = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(raw) != len(shares) or len(target_np) != len(shares):
        raise ValueError("share vectors have different lengths")
    coeff = torch.as_tensor(coefficients, dtype=torch.float64).reshape(-1).detach()
    adjusted = coefficient_adjusted_magnitude_share(raw, coeff,
                                                     domain_num=len(shares), task_num=task_num)
    adjusted_np = adjusted.cpu().numpy()
    target_np = target_np / target_np.sum()
    return {
        "raw_domain_magnitude": raw.cpu().tolist(),
        "applied_domain_coefficients": coeff.reshape(int(task_num), len(shares))[0].cpu().tolist(),
        "actual_magnitude_share": adjusted_np.tolist(),
        "data_share": shares.tolist(),
        "target_share": target_np.tolist(),
        "data_share_mismatch": float(0.5 * np.abs(adjusted_np - shares).sum()),
        "target_calibration_error": float(0.5 * np.abs(adjusted_np - target_np).sum()),
    }


def registry_hash() -> str:
    payload = {"arms": list(E6_ARM_KEYS), "registry": E6_ARM_REGISTRY,
               "protocol": E6_FORMAL_PROTOCOL}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_registry() -> Dict[str, object]:
    if tuple(E6_ARM_REGISTRY) != E6_ARM_KEYS:
        raise AssertionError("E6 arm registry order is not closed")
    for key in E6_ARM_KEYS:
        entry = E6_ARM_REGISTRY[key]
        if bool(entry["feedback"]) and entry["coefficient_behavior"] != "adaptive_closed_form":
            raise AssertionError("adaptive arm registry is inconsistent")
    if not math.isclose(E6_BETA, 2.0 ** (-1.0 / 5.0), rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("E6 EMA beta derivation changed")
    return {"arms": list(E6_ARM_KEYS), "registry_hash": registry_hash(),
            "beta": E6_BETA, "steps_per_epoch": E6_STEPS_PER_EPOCH}


__all__ = [
    "E6_ARM_KEYS", "E6_ARM_REGISTRY", "E6_EXPECTED_SEEDS", "E6_FORMAL_SETTING",
    "E6_ROLE_SIZES", "E6_STEPS_PER_EPOCH", "E6_BETA", "E6_FORMAL_PROTOCOL",
    "validate_arm_key", "normalise_shares", "target_shares", "static_coefficients",
    "static_domain_coefficients", "raw_domain_magnitudes", "domain_magnitudes_from_cells",
    "coefficient_adjusted_magnitude_share", "closed_form_coefficients",
    "calibrate_coefficients", "EMAMagnitudeState", "ema_update", "coefficient_variation",
    "calibration_metrics", "registry_hash", "validate_registry", "iter_cells",
]
