"""Instrumentation primitives for Stage E1.

The module is deliberately independent from the repository's trainer.  A
caller supplies the six masked BCE terms from one forward pass and the module
returns detached diagnostics.  In particular, :func:`collect_cell_gradients`
uses ``torch.autograd.grad`` and therefore never writes observer gradients to
``Parameter.grad``.  The training runner can subsequently perform the normal
``loss.backward(); optimizer.step()`` path.

The public helpers are small enough to be used by unit tests without loading
MovieLens or constructing the full MDMTRec model.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch


EPS = 1.0e-12
GROUP_NAMES = ("shared", "domain_specific", "task_specific", "cell_specific", "architecture")
MAIN_GROUP_NAMES = GROUP_NAMES[:4]
CELL_ORDER_NOTE = "task-major: idx = task * domain_num + domain"

# These are the protocol values used by the formal E1 matrix.  They are kept
# here so a pair checker cannot accidentally treat a short smoke run as a
# completed formal pair.  ``dataset_path`` is checked by the orchestrator
# against its normalized repository path because it is host-specific.
FORMAL_PROTOCOL = {
    "epoch": 50,
    "batch_size": 40960,
    "learning_rate": 1.0e-2,
    "learning_rate_darts": 1.0e-3,
    "weight_decay": 1.0e-5,
    "expert_num": 4,
    "probe_interval": 1,
    "instrument": 1,
    "role_sizes": "150000,34500,3000",
    "role_sizes_parsed": {"L": 150000, "M": 34500, "S": 3000},
    "fix_optim_grouping": 1,
    "earlystop_patience": 10,
    "backbone": "mdmoe",
    "scheduler": None,
    "task": "all",
    "domain": "all",
}


def cell_index(task: int, domain: int, domain_num: int) -> int:
    """Return the fixed task-major cell index used throughout E1."""

    return int(task) * int(domain_num) + int(domain)


def iter_cells(domain_num: int, task_num: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(cell_index, domain, task)`` in task-major order."""

    for task in range(task_num):
        for domain in range(domain_num):
            yield cell_index(task, domain, domain_num), domain, task


def _is_architecture(name: str) -> bool:
    return name.startswith("_weight_") or name.startswith("architecture.")


def parameter_group_for_name(name: str) -> str:
    """Map an MDMTRec parameter name to the E1 mutually-exclusive group.

    The mapping intentionally does not use the historical P1 definition:
    ``domain_expert`` and ``task_expert`` are private in E1.  Unknown names
    are rejected so that a new model cannot silently contaminate the shared
    contribution analysis.
    """

    if _is_architecture(name):
        return "architecture"
    if name.startswith(("embedding.", "shared_weight", "shared_bias", "skip_conn.",
                        "star_mlp.", "expert.")):
        return "shared"
    if name.startswith(("slot_weight.", "slot_bias.", "domain_expert.")):
        return "domain_specific"
    if name.startswith("task_expert."):
        return "task_specific"
    if name.startswith(("gate.", "tower.")):
        return "cell_specific"
    raise ValueError(
        "unclassified parameter {!r}; update the explicit E1 partition before "
        "running this model".format(name)
    )


@dataclass(frozen=True)
class ParameterPartition:
    """Auditable parameter-name partition used by the observer."""

    groups: Mapping[str, Tuple[str, ...]]
    counts: Mapping[str, int]
    shapes: Mapping[str, Tuple[int, ...]]
    manifest_hash: str
    state_hash: str = ""

    @property
    def all_names(self) -> Tuple[str, ...]:
        return tuple(name for group in GROUP_NAMES for name in self.groups[group])

    @property
    def main_names(self) -> Tuple[str, ...]:
        return tuple(name for group in MAIN_GROUP_NAMES for name in self.groups[group])

    def names(self, group: str) -> Tuple[str, ...]:
        return tuple(self.groups[group])

    def as_dict(self) -> Dict[str, object]:
        return {
            "groups": {g: list(self.groups[g]) for g in GROUP_NAMES},
            "param_count": {g: int(self.counts[g]) for g in GROUP_NAMES},
            "param_shapes": {n: list(self.shapes[n]) for n in self.all_names},
            "manifest_hash": self.manifest_hash,
            "state_hash": self.state_hash,
            "cell_order": CELL_ORDER_NOTE,
        }


def _hash_parameter_manifest(named_parameters: Mapping[str, torch.nn.Parameter],
                             groups: Mapping[str, Sequence[str]]) -> str:
    payload = []
    for group in GROUP_NAMES:
        for name in groups[group]:
            p = named_parameters[name]
            payload.append({"name": name, "group": group,
                            "shape": list(p.shape), "numel": int(p.numel())})
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def parameter_state_hash(model: torch.nn.Module) -> str:
    """Hash parameter names/shapes/values for run manifests.

    This is a diagnostic identity hash, not a checkpoint format.  Values are
    converted to CPU bytes in deterministic named-parameter order.
    """

    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("ascii"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_parameter_partition(model: torch.nn.Module, *, strict: bool = True) -> ParameterPartition:
    """Build and validate the E1 parameter partition.

    ``strict=True`` is the production default.  ``strict=False`` is useful for
    tiny synthetic tests that use arbitrary names; such names are assigned to
    ``shared`` only in the test fixture, never by the production runner.
    """

    named = dict(model.named_parameters())
    groups: Dict[str, List[str]] = {g: [] for g in GROUP_NAMES}
    shapes: Dict[str, Tuple[int, ...]] = {}
    for name, p in named.items():
        try:
            group = parameter_group_for_name(name)
        except ValueError:
            if strict:
                raise
            group = "shared"
        groups[group].append(name)
        shapes[name] = tuple(int(x) for x in p.shape)
    for group in GROUP_NAMES:
        groups[group].sort()
    flattened = [n for group in GROUP_NAMES for n in groups[group]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(named):
        raise AssertionError("E1 parameter partition is not a complete disjoint partition")
    counts = {g: sum(int(named[n].numel()) for n in groups[g]) for g in GROUP_NAMES}
    frozen = ParameterPartition(
        groups={g: tuple(groups[g]) for g in GROUP_NAMES},
        counts=counts,
        shapes=shapes,
        manifest_hash=_hash_parameter_manifest(named, groups),
    )
    return ParameterPartition(
        groups=frozen.groups, counts=frozen.counts, shapes=frozen.shapes,
        manifest_hash=frozen.manifest_hash, state_hash=parameter_state_hash(model)
    )


def _normalise_shares(domain_shares: Sequence[float]) -> np.ndarray:
    shares = np.asarray(domain_shares, dtype=np.float64)
    if shares.ndim != 1 or len(shares) == 0:
        raise ValueError("domain_shares must be a non-empty one-dimensional sequence")
    if not np.isfinite(shares).all() or (shares < 0).any() or float(shares.sum()) <= 0:
        raise ValueError("domain_shares must be finite, non-negative, and sum to > 0")
    shares = shares / shares.sum()
    if (shares <= 0).any():
        raise ValueError("all E1 domains need a non-zero sample share")
    return shares


def weighting_coefficients(weighting: str, domain_shares: Sequence[float],
                           task_num: int = 2, *, dtype: torch.dtype = torch.float32,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """Return task-major coefficients for E1 equal/sample weighting.

    Equal has coefficient 1 for every cell.  Sample uses ``D * p_d`` for
    every task, so both six-cell coefficient sums are exactly six.
    """

    if weighting not in ("equal", "sample"):
        raise ValueError("E1 weighting must be 'equal' or 'sample'")
    shares = _normalise_shares(domain_shares)
    per_domain = np.ones_like(shares) if weighting == "equal" else len(shares) * shares
    flat = np.tile(per_domain, int(task_num))
    return torch.as_tensor(flat, dtype=dtype, device=device)


def weighting_coefficients_numpy(weighting: str, domain_shares: Sequence[float],
                                 task_num: int = 2) -> np.ndarray:
    return weighting_coefficients(weighting, domain_shares, task_num,
                                  dtype=torch.float64).cpu().numpy()


def flatten_parameter_tensors(named_parameters: Mapping[str, torch.nn.Parameter],
                              names: Sequence[str],
                              grads: Optional[Sequence[Optional[torch.Tensor]]] = None,
                              *, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, int]:
    """Flatten gradients in a deterministic parameter order.

    ``None`` gradients are represented by zero vectors and counted.  This is
    required for per-cell private parameters and is intentionally not hidden.
    """

    if device is None:
        for name in names:
            device = named_parameters[name].device
            break
    if device is None:
        device = torch.device("cpu")
    pieces: List[torch.Tensor] = []
    unused = 0
    if grads is None:
        grads = [named_parameters[name].grad for name in names]
    if len(grads) != len(names):
        raise ValueError("gradient/name lengths differ")
    for name, grad in zip(names, grads):
        p = named_parameters[name]
        if grad is None:
            unused += 1
            pieces.append(torch.zeros(p.numel(), dtype=p.dtype, device=device))
        else:
            pieces.append(grad.detach().reshape(-1).to(device=device))
    if not pieces:
        return torch.zeros(0, dtype=torch.float32, device=device), unused
    return torch.cat(pieces), unused


def collect_cell_gradients(
    loss_terms: Sequence[torch.Tensor],
    model: torch.nn.Module,
    partition: ParameterPartition,
    *,
    retain_graph: bool = True,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, int]]]:
    """Collect raw per-cell gradients with one forward graph.

    The six terms must be in task-major order.  The returned vectors are raw
    gradients of each masked-mean BCE term, before any loss coefficient.  All
    calls keep the graph by default so the caller can immediately execute the
    real weighted ``backward`` path.
    """

    # The partition does not encode D/T; callers may use any D*T.  An empty
    # term vector is the only universally invalid shape here.
    if len(loss_terms) == 0:
        raise ValueError("loss_terms must contain at least one cell")
    named = dict(model.named_parameters())
    active_names = [n for n in partition.all_names if named[n].requires_grad]
    active_params = [named[n] for n in active_names]
    group_sets = {group: set(partition.groups[group]) for group in GROUP_NAMES}
    by_group_indices = {
        group: [i for i, name in enumerate(active_names) if name in group_sets[group]]
        for group in GROUP_NAMES
    }
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    unused_out: Dict[int, Dict[str, int]] = {}
    for idx, term in enumerate(loss_terms):
        grads = torch.autograd.grad(term, active_params, retain_graph=retain_graph,
                                    allow_unused=True)
        out[idx] = {}
        unused_out[idx] = {}
        for group in GROUP_NAMES:
            selected_names = [active_names[i] for i in by_group_indices[group]]
            selected_grads = [grads[i] for i in by_group_indices[group]]
            vec, unused = flatten_parameter_tensors(named, selected_names,
                                                    selected_grads, device=term.device)
            out[idx][group] = vec
            # Frozen parameters do not participate in autograd and are not
            # counted as unused active parameters.  All returned None values
            # are genuine allow_unused results.
            unused_out[idx][group] = int(unused)
    return out, unused_out


def _sum_vectors(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    items = list(vectors)
    if not items:
        return torch.zeros(0)
    result = torch.zeros_like(items[0])
    for item in items:
        result = result + item
    return result


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector).detach().cpu()) if vector.numel() else 0.0


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.reshape(-1), b.reshape(-1)).detach().cpu()) if a.numel() else 0.0


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> float:
    if not a.numel() or not b.numel():
        return float("nan")
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    denom = na * nb
    if float(denom.detach().cpu()) <= eps:
        return float("nan")
    value = torch.dot(a.reshape(-1), b.reshape(-1)) / denom
    return float(torch.clamp(value, -1.0, 1.0).detach().cpu())


def _group_contribution(raw: Mapping[int, torch.Tensor], coeffs: torch.Tensor,
                        domain_num: int, task_num: int,
                        *, eps: float = EPS) -> Tuple[Dict[int, float], Dict[int, float],
                                                      Dict[int, float], torch.Tensor]:
    """Return signed projections, normalized contributions, magnitude shares."""

    coeffs = coeffs.reshape(-1)
    weighted = {i: raw[i] * coeffs[i].to(raw[i].device) for i in raw}
    combined = _sum_vectors(weighted.values())
    combined_norm = torch.linalg.vector_norm(combined) if combined.numel() else torch.tensor(0.0)
    combined_sq = float((combined_norm * combined_norm).detach().cpu())
    denom_proj = float(combined_norm.detach().cpu()) + eps
    projections: Dict[int, float] = {}
    normalised: Dict[int, float] = {}
    raw_norms = {i: _norm(weighted[i]) for i in raw}
    total_mag = sum(raw_norms.values()) + eps
    magnitude_share = {i: raw_norms[i] / total_mag for i in raw}
    for i, vector in weighted.items():
        projections[i] = _dot(vector, combined) / denom_proj
        normalised[i] = _dot(vector, combined) / (combined_sq + eps)
    return projections, normalised, magnitude_share, combined


def contribution_views(raw: Mapping[int, torch.Tensor], coefficients: torch.Tensor,
                       domain_num: int, task_num: int, *, eps: float = EPS) -> Dict[str, object]:
    """Public wrapper for the contribution decomposition used by the runner.

    ``projection`` is the signed projection onto the combined direction,
    ``normalized`` sums to one (up to the epsilon convention), ``magnitude``
    is the non-negative weighted-norm share, and ``combined`` is the summed
    weighted vector.
    """

    projection, normalized, magnitude, combined = _group_contribution(
        raw, coefficients, domain_num, task_num, eps=eps)
    return {"projection": projection, "normalized": normalized,
            "magnitude": magnitude, "combined": combined}


def _rms(norm: float, count: int) -> float:
    return float(norm / math.sqrt(count)) if count > 0 else 0.0


def make_gradient_step_rows(
    raw_by_group: Mapping[str, Mapping[int, torch.Tensor]],
    unused_by_group: Mapping[str, Mapping[int, int]],
    *,
    actual_coefficients: torch.Tensor,
    equal_coefficients: torch.Tensor,
    sample_coefficients: torch.Tensor,
    domain_num: int,
    task_num: int,
    partition: ParameterPartition,
    domain_shares: Sequence[float],
    weighting: str,
    epoch: int,
    step: int,
    batch_fingerprint: str,
    cell_sample_counts: Sequence[int],
    actual_update_by_group: Optional[Mapping[str, torch.Tensor]] = None,
    reconstruction_error_by_group: Optional[Mapping[str, float]] = None,
    eps: float = EPS,
) -> List[Dict[str, object]]:
    """Create long-form ``gradient_step.csv`` rows for one probe.

    The row contains actual and both fixed-state counterfactual contribution
    values.  ``actual_update_by_group`` is the post-step ``before-after``
    vector; it is optional so the same function can be tested before an
    optimizer step.
    """

    shares = _normalise_shares(domain_shares)
    actual = actual_coefficients.reshape(-1)
    equal = equal_coefficients.reshape(-1)
    sample = sample_coefficients.reshape(-1)
    if len(actual) != domain_num * task_num:
        raise ValueError("actual coefficient vector does not cover all cells")
    rows: List[Dict[str, object]] = []
    for group in GROUP_NAMES:
        raw = dict(raw_by_group[group])
        actual_proj, actual_q, actual_share_cell, actual_combined = _group_contribution(
            raw, actual, domain_num, task_num, eps=eps)
        equal_proj, equal_q, equal_share_cell, equal_combined = _group_contribution(
            raw, equal, domain_num, task_num, eps=eps)
        sample_proj, sample_q, sample_share_cell, sample_combined = _group_contribution(
            raw, sample, domain_num, task_num, eps=eps)
        update = None if actual_update_by_group is None else actual_update_by_group.get(group)
        update_norm = _norm(update) if update is not None else float("nan")
        # Domain magnitude shares are the sum of the two task-cell weighted
        # norms.  They are identical for each task row of that domain.
        domain_mag = {d: sum(actual_share_cell.get(cell_index(t, d, domain_num), 0.0)
                             for t in range(task_num)) for d in range(domain_num)}
        mismatch = 0.5 * sum(abs(domain_mag[d] - float(shares[d])) for d in range(domain_num))
        for idx, domain, task in iter_cells(domain_num, task_num):
            vector = raw[idx]
            raw_norm = _norm(vector)
            coeff = float(actual[idx].detach().cpu())
            weighted_norm = abs(coeff) * raw_norm
            raw_count = int(partition.counts[group])
            row: Dict[str, object] = {
                "epoch": int(epoch),
                "step": int(step),
                "domain": int(domain),
                "task": int(task),
                "cell": int(idx),
                "parameter_group": group,
                "weighting": weighting,
                "batch_fingerprint": batch_fingerprint,
                "cell_sample_count": int(cell_sample_counts[idx]),
                "loss_coefficient": coeff,
                "raw_gradient_norm": raw_norm,
                "raw_gradient_rms": _rms(raw_norm, raw_count),
                "weighted_gradient_norm": weighted_norm,
                "weighted_gradient_rms": _rms(weighted_norm, raw_count),
                "effective_projection": actual_proj[idx],
                "normalized_contribution": actual_q[idx],
                # Keep the signed domain contribution explicit.  It is the
                # sum of the two task-cell signed contributions and can be
                # audited without reconstructing a second aggregation.
                "domain_signed_contribution": sum(
                    actual_q.get(cell_index(t, domain, domain_num), 0.0)
                    for t in range(task_num)),
                "magnitude_share": domain_mag[domain],
                "overrepresentation_ratio": domain_mag[domain] / (float(shares[domain]) + eps),
                "mismatch_index": mismatch,
                "equal_cf_projection": equal_proj[idx],
                "equal_cf_contribution": equal_q[idx],
                "sample_cf_projection": sample_proj[idx],
                "sample_cf_contribution": sample_q[idx],
                "equal_cf_magnitude_share": sum(
                    equal_share_cell.get(cell_index(t, domain, domain_num), 0.0)
                    for t in range(task_num)),
                "sample_cf_magnitude_share": sum(
                    sample_share_cell.get(cell_index(t, domain, domain_num), 0.0)
                    for t in range(task_num)),
                "unused_param_count": int(unused_by_group[group].get(idx, 0)),
                "actual_update_norm": update_norm,
                "actual_update_alignment": float("nan"),
                "actual_update_projection": float("nan"),
                "gradient_reconstruction_error": float(
                    (reconstruction_error_by_group or {}).get(group, float("nan"))),
            }
            if update is not None:
                row["actual_update_alignment"] = _cosine(vector * actual[idx].to(vector.device), update)
                row["actual_update_projection"] = _dot(
                    vector * actual[idx].to(vector.device), update) / (update_norm + eps)
            rows.append(row)
    return rows


@dataclass
class WelfordVector:
    """Online coordinate-wise Welford accumulator for one gradient vector."""

    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    count: int = 0
    sum_norm: float = 0.0
    sum_norm_sq: float = 0.0

    def update(self, value: torch.Tensor) -> None:
        x = value.detach()
        if self.mean is None:
            self.mean = torch.zeros_like(x)
            self.m2 = torch.zeros_like(x)
        if tuple(x.shape) != tuple(self.mean.shape):
            raise ValueError("Welford vector shape changed")
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2
        n = float(torch.linalg.vector_norm(x).detach().cpu())
        self.sum_norm += n
        self.sum_norm_sq += n * n

    @property
    def norm_of_mean(self) -> float:
        return _norm(self.mean) if self.mean is not None else float("nan")

    @property
    def variance_trace(self) -> float:
        if self.count < 2 or self.m2 is None:
            return float("nan")
        return float(self.m2.sum().detach().cpu()) / (self.count - 1)

    @property
    def mean_norm(self) -> float:
        return self.sum_norm / self.count if self.count else float("nan")

    @property
    def snr(self) -> float:
        variance = self.variance_trace
        if not np.isfinite(variance) or variance <= 0:
            return float("nan") if not self.count else self.norm_of_mean ** 2 / (variance + EPS)
        return self.norm_of_mean ** 2 / (variance + EPS)

    def as_dict(self) -> Dict[str, float]:
        return {
            "probe_count": int(self.count),
            "mean_gradient_norm": self.mean_norm,
            "norm_of_mean_gradient": self.norm_of_mean,
            "gradient_variance": self.variance_trace,
            "gradient_SNR": self.snr,
        }


@dataclass
class WelfordNorm:
    """Low-memory dispersion accumulator for private-group norm statistics."""

    count: int = 0
    sum_value: float = 0.0
    sum_sq: float = 0.0

    def update(self, vector: torch.Tensor) -> None:
        value = _norm(vector)
        self.count += 1
        self.sum_value += value
        self.sum_sq += value * value

    def as_dict(self) -> Dict[str, float]:
        if self.count < 1:
            return {"probe_count": 0, "mean_gradient_norm": float("nan"),
                    "norm_of_mean_gradient": float("nan"),
                    "gradient_variance": float("nan"), "gradient_SNR": float("nan"),
                    "variance_scope": "norm_scalar"}
        mean = self.sum_value / self.count
        variance = ((self.sum_sq - self.count * mean * mean) /
                     (self.count - 1)) if self.count >= 2 else float("nan")
        return {"probe_count": self.count, "mean_gradient_norm": mean,
                "norm_of_mean_gradient": mean, "gradient_variance": max(0.0, variance)
                if np.isfinite(variance) else float("nan"),
                "gradient_SNR": float("nan"), "variance_scope": "norm_scalar"}


class GradientEpochAccumulator:
    """Per-epoch gradient statistics with full Welford for shared vectors."""

    def __init__(self, partition: ParameterPartition, domain_num: int, task_num: int,
                 *, full_vector_groups: Sequence[str] = ("shared",)) -> None:
        self.partition = partition
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.full_vector_groups = set(full_vector_groups)
        self.stats: Dict[Tuple[str, int], object] = {}
        for group in GROUP_NAMES:
            for idx, _, _ in iter_cells(domain_num, task_num):
                self.stats[(group, idx)] = (
                    WelfordVector() if group in self.full_vector_groups else WelfordNorm())
        self.contribution: Dict[Tuple[str, int], List[float]] = {
            (g, i): [] for g in GROUP_NAMES for i, _, _ in iter_cells(domain_num, task_num)
        }
        self.mismatch: Dict[str, List[float]] = {g: [] for g in GROUP_NAMES}
        self.weighted_coefficients: Dict[str, float] = {}

    def update(self, raw_by_group: Mapping[str, Mapping[int, torch.Tensor]],
               normalized_contributions: Mapping[str, Mapping[int, float]],
               mismatches: Mapping[str, float], actual_coefficients: torch.Tensor) -> None:
        for group in GROUP_NAMES:
            self.weighted_coefficients[group] = float("nan")
            for idx, _, _ in iter_cells(self.domain_num, self.task_num):
                self.stats[(group, idx)].update(raw_by_group[group][idx])
                self.contribution[(group, idx)].append(
                    float(normalized_contributions[group][idx]))
            self.mismatch[group].append(float(mismatches[group]))

    def rows(self, *, epoch: int, seed: int, weighting: str,
             actual_coefficients: torch.Tensor) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        coeff = actual_coefficients.detach().cpu().numpy().reshape(-1)
        for group in GROUP_NAMES:
            mismatch = (float(np.mean(self.mismatch[group]))
                        if self.mismatch[group] else float("nan"))
            for idx, domain, task in iter_cells(self.domain_num, self.task_num):
                stat = self.stats[(group, idx)].as_dict()
                values = np.asarray(self.contribution[(group, idx)], dtype=np.float64)
                row: Dict[str, object] = {
                    "seed": int(seed), "epoch": int(epoch), "domain": int(domain),
                    "task": int(task), "cell": int(idx), "weighting": weighting,
                    "parameter_group": group,
                    "gradient_norm": stat["mean_gradient_norm"],
                    "weighted_gradient_norm": stat["mean_gradient_norm"] * abs(float(coeff[idx]))
                    if np.isfinite(stat["mean_gradient_norm"]) else float("nan"),
                    "gradient_variance": stat["gradient_variance"],
                    "weighted_gradient_variance": stat["gradient_variance"] * float(coeff[idx] ** 2)
                    if np.isfinite(stat["gradient_variance"]) else float("nan"),
                    "gradient_SNR": stat["gradient_SNR"],
                    "norm_of_mean_gradient": stat["norm_of_mean_gradient"],
                    "probe_count": stat["probe_count"],
                    "contribution_mean": float(np.mean(values)) if len(values) else float("nan"),
                    "contribution_std": float(np.std(values)) if len(values) else float("nan"),
                    "mismatch_index": mismatch,
                    "variance_scope": stat.get("variance_scope", "vector_trace"),
                }
                rows.append(row)
        return rows


class RoutingObserver:
    """Read-only gate observer that accumulates scalar routing summaries."""

    def __init__(self, model: torch.nn.Module, domain_num: int, task_num: int,
                 *, enabled: bool = True) -> None:
        self.model = model
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.enabled = bool(enabled) and hasattr(model, "gate")
        # MDMTRec exposes ``expert_num`` (number of shared experts), while
        # ``len(model.gate)`` is the number of cell gates.  Keep those notions
        # separate; the hook index is mapped to a cell below.
        self.expert_num = int(getattr(model, "expert_num", 0))
        self.handles: List[object] = []
        self._domain_ids: Optional[torch.Tensor] = None
        self._epoch = -1
        self._reset_epoch()
        if self.enabled:
            try:
                _ = len(model.gate)
            except TypeError:
                self.enabled = False
            if self.enabled:
                for idx, module in enumerate(model.gate):
                    self.handles.append(module.register_forward_hook(self._hook(idx)))

    def _reset_epoch(self) -> None:
        self._sums: Dict[int, torch.Tensor] = {}
        self._counts: Dict[int, int] = {}
        self._top1: Dict[int, torch.Tensor] = {}

    def _hook(self, idx: int):
        def callback(_module, _inputs, output):
            if not self.enabled or self._domain_ids is None:
                return
            values = output.detach()
            if values.ndim != 2:
                return
            if self.expert_num == 0:
                self.expert_num = int(values.shape[1])
            domain = idx % self.domain_num
            mask = self._domain_ids.to(values.device) == domain
            if not bool(mask.any()):
                return
            selected = values[mask].to(dtype=torch.float64)
            self._sums[idx] = self._sums.get(
                idx, torch.zeros(values.shape[1], dtype=torch.float64)) + selected.sum(dim=0).cpu()
            self._counts[idx] = self._counts.get(idx, 0) + int(selected.shape[0])
            top = selected.argmax(dim=1)
            counts = torch.bincount(top.cpu(), minlength=values.shape[1]).to(dtype=torch.float64)
            self._top1[idx] = self._top1.get(
                idx, torch.zeros(values.shape[1], dtype=torch.float64)) + counts
        return callback

    def start_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._reset_epoch()

    def start_batch(self, domain_ids: torch.Tensor) -> None:
        self._domain_ids = domain_ids.detach()

    def stop_batch(self) -> None:
        self._domain_ids = None

    def rows(self, *, seed: int, weighting: str, setting: str, epoch: int) -> List[Dict[str, object]]:
        if not self.enabled:
            return []
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            count = self._counts.get(idx, 0)
            if count <= 0:
                continue
            probs = self._sums[idx] / count
            probs_np = probs.numpy()
            clipped = np.clip(probs_np, EPS, 1.0)
            entropy = float(-(probs_np * np.log(clipped)).sum())
            top_counts = self._top1.get(idx, torch.zeros(self.expert_num, dtype=torch.float64)).numpy()
            row: Dict[str, object] = {
                "seed": int(seed), "weighting": weighting, "setting": setting,
                "epoch": int(epoch), "domain": int(domain), "task": int(task),
                "cell": int(idx), "routing_scope": "cell",
                "routing_sample_count": int(count),
                "gate_entropy": entropy,
                "effective_expert_count": float(math.exp(entropy)),
                "top1_expert_frequency": float(top_counts.max() / count)
                if len(top_counts) else float("nan"),
            }
            for expert, value in enumerate(probs_np):
                row[f"expert_{expert}_mean_probability"] = float(value)
            rows.append(row)

        # The cell rows above preserve the physical-domain mask and task
        # identity.  Also emit a direct domain-level task-average row so the
        # report layer does not need to infer this diagnostic from raw gate
        # rows.  The probability average is unweighted across the task rows;
        # the sample count remains the sum of the contributing cell counts.
        for domain in range(self.domain_num):
            cell_indices = [cell_index(task, domain, self.domain_num)
                            for task in range(self.task_num)
                            if self._counts.get(cell_index(task, domain, self.domain_num), 0) > 0]
            if not cell_indices:
                continue
            probability_rows = []
            top1_values = []
            total_count = 0
            for idx in cell_indices:
                count = self._counts[idx]
                probs = (self._sums[idx] / count).numpy()
                probability_rows.append(probs)
                total_count += int(count)
                top_counts = self._top1.get(
                    idx, torch.zeros(self.expert_num, dtype=torch.float64)).numpy()
                if len(top_counts):
                    top1_values.append(float(top_counts.max() / count))
            probs_np = np.mean(np.asarray(probability_rows, dtype=np.float64), axis=0)
            clipped = np.clip(probs_np, EPS, 1.0)
            entropy = float(-(probs_np * np.log(clipped)).sum())
            average_row: Dict[str, object] = {
                "seed": int(seed), "weighting": weighting, "setting": setting,
                "epoch": int(epoch), "domain": int(domain), "task": -1,
                "cell": -1, "routing_scope": "domain_task_average",
                "routing_sample_count": int(total_count),
                "gate_entropy": entropy,
                "effective_expert_count": float(math.exp(entropy)),
                "top1_expert_frequency": (float(np.mean(top1_values))
                                           if top1_values else float("nan")),
            }
            for expert, value in enumerate(probs_np):
                average_row[f"expert_{expert}_mean_probability"] = float(value)
            rows.append(average_row)
        return rows

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        self._domain_ids = None


def batch_fingerprint(x: torch.Tensor, y: torch.Tensor) -> str:
    """Return a stable digest of one CPU batch, including shape and dtype."""

    h = hashlib.sha256()
    for value in (x, y):
        cpu = value.detach().cpu().contiguous()
        h.update(str(cpu.dtype).encode("ascii"))
        h.update(str(tuple(cpu.shape)).encode("ascii"))
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _trace_rows(path: str) -> List[Dict[str, str]]:
    """Read the comparable train/architecture rows from a batch trace."""

    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    return [row for row in rows if row.get("kind") in ("train", "architecture")]


def _read_json_dict(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _optional_int(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_key(row: Mapping[str, str]) -> Optional[Tuple[str, int, int]]:
    try:
        return (str(row.get("kind", "")), int(row.get("epoch", "")),
                int(row.get("step", "")))
    except (TypeError, ValueError):
        return None


def _trace_metadata(run_dir: str, rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    manifest = _read_json_dict(os.path.join(run_dir, "run_manifest.json"))
    observed_value = manifest.get("observed_epochs", [])
    observed = []
    if isinstance(observed_value, (list, tuple)):
        for value in observed_value:
            parsed = _optional_int(value)
            if parsed is not None:
                observed.append(parsed)
    epochs = sorted({key[1] for row in rows if (key := _trace_key(row)) is not None})
    return {
        "manifest": manifest,
        "observed_epochs": observed,
        "trace_epochs": epochs,
        "stopped_epoch": _optional_int(manifest.get("stopped_epoch")),
        "censored_after_epoch": _optional_int(manifest.get("censored_after_epoch")),
    }


def _config_differences(left: Mapping[str, object], right: Mapping[str, object],
                        allowed: Set[str]) -> List[Dict[str, object]]:
    differences = []
    for field in sorted(set(left) | set(right)):
        if field in allowed:
            continue
        left_value, right_value = left.get(field), right.get(field)
        try:
            equal = _canonical(left_value) == _canonical(right_value)
        except (TypeError, ValueError):
            equal = left_value == right_value
        if not equal:
            differences.append({"field": field, "equal": left_value, "sample": right_value})
    return differences


def _manifest_identity_differences(left: Mapping[str, object],
                                   right: Mapping[str, object]) -> List[Dict[str, object]]:
    fields = (
        "dataset", "domain_num", "task_num", "role_by_domain", "role_sizes",
        "train_domain_counts", "domain_shares", "split_counts", "positive_rates",
        "initial_parameter_state_hash",
    )
    differences = []
    for field in fields:
        if field not in left or field not in right:
            differences.append({"field": "manifest." + field,
                                "reason": "missing_identity_field"})
            continue
        if field == "initial_parameter_state_hash" and (
                not left.get(field) or not right.get(field)):
            differences.append({"field": "manifest." + field,
                                "reason": "missing_identity_field"})
            continue
        try:
            equal = _canonical(left[field]) == _canonical(right[field])
        except (TypeError, ValueError):
            equal = left[field] == right[field]
        if not equal:
            differences.append({"field": "manifest." + field,
                                "equal": left[field], "sample": right[field]})
    left_partition = left.get("partition")
    right_partition = right.get("partition")
    if not isinstance(left_partition, Mapping) or not isinstance(right_partition, Mapping):
        differences.append({"field": "manifest.partition", "reason": "missing_identity_field"})
    else:
        for field in ("groups", "param_count", "param_shapes", "manifest_hash", "state_hash"):
            if field not in left_partition or field not in right_partition:
                differences.append({"field": "manifest.partition." + field,
                                    "reason": "missing_identity_field"})
            elif field in ("manifest_hash", "state_hash") and (
                    not left_partition.get(field) or not right_partition.get(field)):
                differences.append({"field": "manifest.partition." + field,
                                    "reason": "missing_identity_field"})
            elif _canonical(left_partition[field]) != _canonical(right_partition[field]):
                differences.append({"field": "manifest.partition." + field,
                                    "equal": left_partition[field],
                                    "sample": right_partition[field]})
    return differences


def _formal_protocol_differences(config: Mapping[str, object],
                                 manifest: Mapping[str, object]) -> List[Dict[str, object]]:
    differences = []
    for field, expected in FORMAL_PROTOCOL.items():
        actual = config.get(field)
        if field == "dataset_path":
            continue
        if actual != expected:
            differences.append({"field": "config." + field,
                                "expected": expected, "actual": actual})
    if config.get("loss_weighting") != config.get("weighting"):
        differences.append({"field": "config.loss_weighting",
                            "reason": "must_match_weighting_arm"})
    if config.get("instrumentation") is not True:
        differences.append({"field": "config.instrumentation",
                            "expected": True, "actual": config.get("instrumentation")})
    if manifest.get("formal_valid") is not True:
        differences.append({"field": "manifest.formal_valid",
                            "expected": True, "actual": manifest.get("formal_valid")})
    protocol = manifest.get("formal_protocol")
    if isinstance(protocol, Mapping):
        for field, expected in FORMAL_PROTOCOL.items():
            if field not in protocol:
                differences.append({"field": "manifest.formal_protocol." + field,
                                    "reason": "missing_identity_field"})
            elif protocol[field] != expected:
                differences.append({"field": "manifest.formal_protocol." + field,
                                    "expected": expected, "actual": protocol[field]})
    else:
        differences.append({"field": "manifest.formal_protocol",
                            "reason": "missing_identity_field"})
    return differences


def _numeric_value(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _compare_csv_files(left_path: str, right_path: str,
                       *, numeric_tolerance: float) -> Tuple[List[Dict[str, object]], float]:
    def read(path: str) -> Optional[List[Dict[str, str]]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return None

    left_rows, right_rows = read(left_path), read(right_path)
    differences: List[Dict[str, object]] = []
    if left_rows is None or right_rows is None:
        differences.append({"reason": "metric_file_missing",
                            "left": left_rows is not None, "right": right_rows is not None})
        return differences, 0.0
    if len(left_rows) != len(right_rows):
        differences.append({"reason": "metric_row_count_mismatch",
                            "left": len(left_rows), "right": len(right_rows)})
    if not left_rows or not right_rows:
        differences.append({"reason": "metric_file_empty",
                            "left_count": len(left_rows), "right_count": len(right_rows)})
    maximum = 0.0
    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        fields = set(left) | set(right)
        for field in sorted(fields):
            left_value, right_value = left.get(field, ""), right.get(field, "")
            left_number, right_number = _numeric_value(left_value), _numeric_value(right_value)
            if left_number is not None and right_number is not None:
                difference = abs(left_number - right_number)
                maximum = max(maximum, difference)
                if difference > numeric_tolerance:
                    differences.append({"row": index, "field": field,
                                        "reason": "numeric_tolerance_exceeded",
                                        "difference": difference,
                                        "tolerance": numeric_tolerance})
            elif str(left_value) != str(right_value):
                differences.append({"row": index, "field": field,
                                    "reason": "value_mismatch",
                                    "left": left_value, "right": right_value})
    return differences, maximum


def _trace_comparison(equal_rows: Sequence[Mapping[str, str]],
                      sample_rows: Sequence[Mapping[str, str]],
                      equal_meta: Mapping[str, object],
                      sample_meta: Mapping[str, object], *,
                      loss_tolerance: float, compare_losses: bool,
                      ignored_fields: Set[str]) -> Dict[str, object]:
    equal_keys = [_trace_key(row) for row in equal_rows]
    sample_keys = [_trace_key(row) for row in sample_rows]
    mismatches: List[Dict[str, object]] = []
    if any(key is None for key in equal_keys + sample_keys):
        mismatches.append({"reason": "invalid_trace_key"})
    equal_keys_valid = [key for key in equal_keys if key is not None]
    sample_keys_valid = [key for key in sample_keys if key is not None]
    if len(equal_keys_valid) != len(set(equal_keys_valid)):
        mismatches.append({"reason": "duplicate_equal_trace_key"})
    if len(sample_keys_valid) != len(set(sample_keys_valid)):
        mismatches.append({"reason": "duplicate_sample_trace_key"})
    equal_by_key = {key: row for key, row in zip(equal_keys, equal_rows) if key is not None}
    sample_by_key = {key: row for key, row in zip(sample_keys, sample_rows) if key is not None}
    equal_set, sample_set = set(equal_by_key), set(sample_by_key)
    common_set = equal_set & sample_set
    equal_only = equal_set - sample_set
    sample_only = sample_set - equal_set

    length_difference_allowed = True
    length_reason = "same_trace_length"
    if equal_only or sample_only:
        length_difference_allowed = False
        length_reason = "trace_length_or_support_mismatch"
        if len(equal_rows) <= len(sample_rows):
            short_meta, short_only, short_last = equal_meta, equal_only, max(
                (key[1] for key in equal_set), default=None)
            short_name = "equal"
        else:
            short_meta, short_only, short_last = sample_meta, sample_only, max(
                (key[1] for key in sample_set), default=None)
            short_name = "sample"
        stopped = short_meta.get("stopped_epoch")
        censored_after = short_meta.get("censored_after_epoch")
        observed = short_meta.get("observed_epochs", [])
        legal_stop = (stopped is not None and short_last == stopped
                      and stopped in observed
                      and censored_after in (None, stopped + 1))
        asymmetric_before_stop = [key for key in (equal_only | sample_only)
                                  if stopped is None or key[1] <= stopped]
        if legal_stop and not asymmetric_before_stop:
            length_difference_allowed = True
            length_reason = "{}_early_stop_censoring_tail".format(short_name)
        else:
            mismatches.append({
                "reason": "trace_length_mismatch_unexplained",
                "equal_only_count": len(equal_only), "sample_only_count": len(sample_only),
                "short_arm": short_name, "short_stopped_epoch": stopped,
                "short_censored_after_epoch": censored_after,
                "short_trace_last_epoch": short_last,
            })

    equal_common_order = [key for key in equal_keys_valid if key in common_set]
    sample_common_order = [key for key in sample_keys_valid if key in common_set]
    if equal_common_order != sample_common_order:
        mismatches.append({"reason": "common_support_order_mismatch"})

    maximum = 0.0
    maximum_train = 0.0
    maximum_architecture = 0.0
    for index, key in enumerate(equal_common_order):
        left, right = equal_by_key[key], sample_by_key[key]
        for field in ("kind", "epoch", "step", "batch_fingerprint", "batch_size",
                      "domain_counts", "task_counts", "is_probe"):
            if field in ignored_fields:
                continue
            if str(left.get(field, "")) != str(right.get(field, "")):
                mismatches.append({"index": index, "reason": "{} mismatch".format(field),
                                   "field": field, "equal": left.get(field, ""),
                                   "sample": right.get(field, "")})
        left_loss, right_loss = left.get("train_loss", ""), right.get("train_loss", "")
        try:
            left_value, right_value = float(left_loss), float(right_loss)
        except (TypeError, ValueError):
            mismatches.append({"index": index, "reason": "missing_or_non_numeric_train_loss",
                               "equal": left_loss, "sample": right_loss})
            continue
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            mismatches.append({"index": index, "reason": "nonfinite_train_loss",
                               "equal": left_loss, "sample": right_loss})
            continue
        difference = abs(left_value - right_value)
        maximum = max(maximum, difference)
        if key[0] == "train":
            maximum_train = max(maximum_train, difference)
        else:
            maximum_architecture = max(maximum_architecture, difference)
        if compare_losses and difference > float(loss_tolerance):
            mismatches.append({"index": index, "reason": "train_loss_tolerance_exceeded",
                               "difference": difference, "tolerance": float(loss_tolerance)})

    return {
        "mismatches": mismatches,
        "equal_trace_row_count": len(equal_rows),
        "sample_trace_row_count": len(sample_rows),
        "common_trace_row_count": len(common_set),
        "equal_only_count": len(equal_only), "sample_only_count": len(sample_only),
        "length_difference_allowed": bool(length_difference_allowed),
        "length_reason": length_reason,
        "equal_observed_epochs": list(equal_meta.get("observed_epochs", [])),
        "sample_observed_epochs": list(sample_meta.get("observed_epochs", [])),
        "equal_stopped_epoch": equal_meta.get("stopped_epoch"),
        "sample_stopped_epoch": sample_meta.get("stopped_epoch"),
        "equal_censored_after_epoch": equal_meta.get("censored_after_epoch"),
        "sample_censored_after_epoch": sample_meta.get("censored_after_epoch"),
        "max_abs_loss_diff": maximum,
        "max_abs_train_loss_diff": maximum_train,
        "max_abs_architecture_loss_diff": maximum_architecture,
    }


def compare_pair_runs(equal_run_dir: str, sample_run_dir: str,
                      *, loss_tolerance: float = 1.0e-7,
                      compare_losses: bool = False,
                      ignored_trace_fields: Sequence[str] = (),
                      require_formal: bool = True,
                      allowed_config_keys: Sequence[str] = ()) -> Dict[str, object]:
    """Compare paired E1 traces and make mismatches explicitly invalid.

    The comparison is intentionally mechanical: train and architecture batch
    order/fingerprints must agree on their common support.  Equal/sample arms
    intentionally have different objective scalars, so their losses are
    reported but are only checked when ``compare_losses=True``.  A missing
    counterpart is ``PENDING``; any present-but-different identity or trace is
    ``INVALID`` and must not be treated as a completed pair.
    """

    equal_run_dir = os.path.abspath(equal_run_dir)
    sample_run_dir = os.path.abspath(sample_run_dir)
    if not os.path.isdir(equal_run_dir) or not os.path.isdir(sample_run_dir):
        return {
            "status": "PENDING", "valid": None,
            "reason": "counterpart_missing",
            "equal_run_dir": equal_run_dir, "sample_run_dir": sample_run_dir,
            "mismatch_count": 0, "max_abs_loss_diff": "",
            "equal_trace_row_count": "", "sample_trace_row_count": "",
        }
    equal_trace_path = os.path.join(equal_run_dir, "batch_trace.csv")
    sample_trace_path = os.path.join(sample_run_dir, "batch_trace.csv")
    if not os.path.exists(equal_trace_path) or not os.path.exists(sample_trace_path):
        return {
            "status": "INVALID", "valid": False,
            "reason": "batch_trace_missing", "equal_run_dir": equal_run_dir,
            "sample_run_dir": sample_run_dir, "mismatch_count": 1,
            "max_abs_loss_diff": "",
        }
    equal_rows = _trace_rows(equal_trace_path)
    sample_rows = _trace_rows(sample_trace_path)
    equal_meta = _trace_metadata(equal_run_dir, equal_rows)
    sample_meta = _trace_metadata(sample_run_dir, sample_rows)
    equal_manifest = equal_meta["manifest"]
    sample_manifest = sample_meta["manifest"]
    equal_config = _read_json_dict(os.path.join(equal_run_dir, "config.json"))
    sample_config = _read_json_dict(os.path.join(sample_run_dir, "config.json"))
    mismatches = []
    if not equal_rows or not sample_rows:
        mismatches.append({"reason": "batch_trace_empty",
                           "equal_count": len(equal_rows),
                           "sample_count": len(sample_rows)})
    if not equal_manifest or not sample_manifest:
        mismatches.append({"reason": "manifest_missing_or_unreadable"})
    if not equal_config or not sample_config:
        mismatches.append({"reason": "config_missing_or_unreadable"})
    config_diffs = _config_differences(
        equal_config, sample_config,
        set(("weighting", "loss_weighting", "run_dir", "config_hash"))
        | set(allowed_config_keys))
    mismatches.extend(config_diffs)
    identity_diffs = _manifest_identity_differences(equal_manifest, sample_manifest)
    mismatches.extend(identity_diffs)
    formal_diffs = []
    if require_formal:
        formal_diffs.extend(_formal_protocol_differences(equal_config, equal_manifest))
        formal_diffs.extend(_formal_protocol_differences(sample_config, sample_manifest))
        mismatches.extend(formal_diffs)
    trace_result = _trace_comparison(
        equal_rows, sample_rows, equal_meta, sample_meta,
        loss_tolerance=loss_tolerance, compare_losses=compare_losses,
        ignored_fields=set(ignored_trace_fields))
    mismatches.extend(trace_result["mismatches"])
    result = {
        "status": "VALID" if not mismatches else "INVALID",
        "valid": not bool(mismatches),
        "reason": "" if not mismatches else mismatches[0].get("reason", "identity_mismatch"),
        "equal_run_dir": equal_run_dir, "sample_run_dir": sample_run_dir,
        "trace_row_count": trace_result["common_trace_row_count"],
        "equal_trace_row_count": trace_result["equal_trace_row_count"],
        "sample_trace_row_count": trace_result["sample_trace_row_count"],
        "common_trace_row_count": trace_result["common_trace_row_count"],
        "equal_only_count": trace_result["equal_only_count"],
        "sample_only_count": trace_result["sample_only_count"],
        "length_difference_allowed": trace_result["length_difference_allowed"],
        "length_reason": trace_result["length_reason"],
        "equal_observed_epochs": trace_result["equal_observed_epochs"],
        "sample_observed_epochs": trace_result["sample_observed_epochs"],
        "equal_stopped_epoch": trace_result["equal_stopped_epoch"],
        "sample_stopped_epoch": trace_result["sample_stopped_epoch"],
        "equal_censored_after_epoch": trace_result["equal_censored_after_epoch"],
        "sample_censored_after_epoch": trace_result["sample_censored_after_epoch"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "identity_mismatch_count": len(config_diffs) + len(identity_diffs) + len(formal_diffs),
        "controlled_config_differences": config_diffs,
        "manifest_identity_differences": identity_diffs,
        "formal_protocol_differences": formal_diffs,
        "max_abs_loss_diff": trace_result["max_abs_loss_diff"],
        "max_abs_train_loss_diff": trace_result["max_abs_train_loss_diff"],
        "max_abs_architecture_loss_diff": trace_result["max_abs_architecture_loss_diff"],
        "loss_tolerance": float(loss_tolerance),
        "loss_checked": bool(compare_losses),
        "ignored_trace_fields": sorted(set(ignored_trace_fields)),
        "require_formal": bool(require_formal),
    }
    return result


def compare_on_off_runs(off_run_dir: str, on_run_dir: str,
                        *, loss_tolerance: float = 1.0e-7) -> Dict[str, object]:
    """Check same-objective ON/OFF invariants without comparing probe flags."""

    result = compare_pair_runs(
        off_run_dir, on_run_dir, loss_tolerance=loss_tolerance,
        compare_losses=True, ignored_trace_fields=("is_probe",),
        require_formal=False, allowed_config_keys=("instrument", "instrumentation"))
    if result.get("status") == "PENDING":
        # Preserve the missing-counterpart state; a missing ON/OFF arm is not
        # a completed equivalence failure and must be rerun by the caller.
        return result
    mismatches = list(result.get("mismatches", []))
    off_manifest = _read_json_dict(os.path.join(os.path.abspath(off_run_dir), "run_manifest.json"))
    on_manifest = _read_json_dict(os.path.join(os.path.abspath(on_run_dir), "run_manifest.json"))
    metric_differences = []
    metric_maximum = 0.0
    for filename in ("metrics_epoch.csv", "metrics_summary.csv"):
        differences, maximum = _compare_csv_files(
            os.path.join(os.path.abspath(off_run_dir), filename),
            os.path.join(os.path.abspath(on_run_dir), filename),
            numeric_tolerance=1.0e-8)
        metric_differences.extend([dict(item, file=filename) for item in differences])
        metric_maximum = max(metric_maximum, maximum)
    mismatches.extend(metric_differences)
    parameter_hash_equal = True
    for field in ("initial_parameter_state_hash", "final_parameter_state_hash"):
        off_hash, on_hash = off_manifest.get(field), on_manifest.get(field)
        if not off_hash or not on_hash:
            parameter_hash_equal = False
            mismatches.append({"reason": "{}_missing".format(field),
                               "off": off_hash, "on": on_hash})
        elif off_hash != on_hash:
            parameter_hash_equal = False
            mismatches.append({"reason": "{} mismatch".format(field),
                               "off": off_hash, "on": on_hash})
    off_rng = off_manifest.get("rng_restore_evidence")
    on_rng = on_manifest.get("rng_restore_evidence")
    rng_evidence_valid = True
    if not isinstance(off_rng, list) or not isinstance(on_rng, list) \
            or not off_rng or not on_rng:
        rng_evidence_valid = False
        mismatches.append({"reason": "rng_restore_evidence_missing"})
    else:
        for label, evidence in (("off", off_rng), ("on", on_rng)):
            if any(not isinstance(row, Mapping) for row in evidence):
                rng_evidence_valid = False
                mismatches.append({"reason": "rng_restore_evidence_malformed",
                                   "arm": label})
                continue
            for row in evidence:
                if (not row.get("before") or not row.get("after")
                        or row.get("restored") is not True
                        or row.get("before") != row.get("after")):
                    rng_evidence_valid = False
                    mismatches.append({"reason": "rng_restore_evidence_failed",
                                       "arm": label, "epoch": row.get("epoch")})
        if rng_evidence_valid and _canonical(off_rng) != _canonical(on_rng):
            rng_evidence_valid = False
            mismatches.append({"reason": "rng_restore_evidence_mismatch"})
    result.update({
        "status": "VALID" if not mismatches else "INVALID",
        "valid": not bool(mismatches),
        "reason": "" if not mismatches else mismatches[0].get("reason", "equivalence_mismatch"),
        "mismatches": mismatches[:50], "mismatch_count": len(mismatches),
        "metrics_max_abs_diff": metric_maximum,
        "metrics_tolerance": 1.0e-8,
        "parameter_hash_equal": parameter_hash_equal,
        "rng_evidence_equal": rng_evidence_valid,
        "ignored_trace_fields": ["is_probe"],
    })
    return result


def _is_nonfinite(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf",
                                         "infinity", "+infinity", "-infinity"}
    if isinstance(value, (float, np.floating)):
        return not bool(np.isfinite(value))
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return not bool(torch.isfinite(value).detach().cpu().item())
    return False


def nonfinite_reason(file_name: str, row: Mapping[str, object], field: str) -> str:
    """Return a concrete reason for a non-finite value in a raw row."""

    if bool(row.get("stopped_censored", 0)):
        return "early_stop_censored_epoch"
    group = str(row.get("parameter_group", ""))
    if field in ("actual_update_alignment", "actual_update_projection") and group == "architecture":
        return "architecture_group_has_no_main_optimizer_update"
    if field == "gradient_SNR":
        if str(row.get("variance_scope", "")) == "norm_scalar":
            return "vector_snr_not_defined_for_norm_scalar_variance"
        try:
            if int(float(row.get("probe_count", 0))) < 2:
                return "insufficient_probes_for_unbiased_snr"
        except (TypeError, ValueError):
            return "insufficient_probes_for_unbiased_snr"
    if field in ("gradient_variance", "weighted_gradient_variance"):
        try:
            if int(float(row.get("probe_count", 0))) < 2:
                return "insufficient_probes_for_unbiased_variance"
        except (TypeError, ValueError):
            return "insufficient_probes_for_unbiased_variance"
    if field in ("auc", "train_auc", "validation_auc", "test_auc", "AUC"):
        try:
            if int(float(row.get("sample_count", row.get("train_sample_count", 0)))) <= 0:
                return "empty_metric_cell"
        except (TypeError, ValueError):
            return "empty_or_single_class_metric_cell"
        return "single_class_metric_cell"
    if field in ("positive_rate", "train_positive_rate", "validation_positive_rate",
                 "test_positive_rate"):
        return "empty_metric_cell"
    if field in ("logloss", "train_logloss", "validation_logloss", "test_logloss",
                 "LogLoss"):
        return "empty_or_invalid_metric_cell"
    if field in ("gradient_norm", "weighted_gradient_norm", "mean_gradient_norm",
                 "norm_of_mean_gradient", "contribution_mean", "contribution_std",
                 "mismatch_index"):
        try:
            if int(float(row.get("probe_count", 0))) < 1:
                return "insufficient_probes_for_epoch_statistic"
        except (TypeError, ValueError):
            return "insufficient_probes_for_epoch_statistic"
    if file_name == "batch_trace.csv":
        return "nonfinite_training_loss"
    if field.startswith("oracle_") or field.startswith("transfer_"):
        return "nonfinite_oracle_artifact_value"
    if file_name == "gate_epoch.csv":
        return "nonfinite_routing_statistic"
    if "cosine" in field:
        return "zero_norm_gradient_or_no_valid_cosine_probe"
    return "nonfinite_numeric_value"


def collect_nonfinite_inventory(file_rows: Mapping[str, Sequence[Mapping[str, object]]]) -> List[Dict[str, object]]:
    """Inventory every NaN/Inf before CSV conversion, with row and reason."""

    entries: List[Dict[str, object]] = []
    for file_name, rows in file_rows.items():
        for row_index, row in enumerate(rows, start=2):
            for field, value in row.items():
                if not _is_nonfinite(value):
                    continue
                key_fields = ("seed", "epoch", "step", "domain", "task", "cell",
                              "parameter_group", "kind", "split")
                row_key = {key: row[key] for key in key_fields if key in row}
                if isinstance(value, (float, np.floating)):
                    numeric_value = float(value)
                    if math.isnan(numeric_value):
                        inventory_value = "NaN"
                    elif numeric_value > 0:
                        inventory_value = "+Inf"
                    else:
                        inventory_value = "-Inf"
                else:
                    inventory_value = json_safe(value)
                entries.append({
                    "file": str(file_name), "row_index": int(row_index),
                    "row_key": row_key, "field": str(field),
                    "value": inventory_value,
                    "reason": nonfinite_reason(file_name, row, field),
                })
    return entries


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Preserve CPU and all CUDA RNG streams across eval-only work."""

    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def rng_state_digest() -> str:
    """Return a non-consuming digest of the current CPU/CUDA RNG states."""

    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def relative_l2_error(actual: torch.Tensor, expected: torch.Tensor, eps: float = EPS) -> float:
    if not actual.numel() and not expected.numel():
        return 0.0
    return float((torch.linalg.vector_norm(actual - expected) /
                  (torch.linalg.vector_norm(actual) + eps)).detach().cpu())


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)


def _csv_value(value):
    if isinstance(value, (float, np.floating)) and (math.isnan(float(value)) or math.isinf(float(value))):
        return "NA"
    return value


def write_rows_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Optional[Sequence[str]] = None) -> None:
    """Write a stable CSV schema, including an empty file's header."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fields is None:
        field_set = set()
        for row in rows:
            field_set.update(row.keys())
        fields = sorted(field_set)
    fields = list(fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def masked_bce_terms(y_pred: torch.Tensor, y: torch.Tensor, domain_ids: torch.Tensor,
                     domain_num: int, task_num: int,
                     criterion: Optional[torch.nn.Module] = None) -> Tuple[List[torch.Tensor], List[int]]:
    """Build exact per-cell sample-mean BCE terms and counts."""

    if criterion is None:
        criterion = torch.nn.BCELoss()
    terms: List[torch.Tensor] = []
    counts: List[int] = []
    for task in range(task_num):
        for domain in range(domain_num):
            mask = domain_ids == domain
            count = int(mask.sum().detach().cpu())
            counts.append(count)
            if count == 0:
                # A zero term keeps the six-cell vector well-defined without
                # introducing a NaN.  It has no gradient to any parameter.
                terms.append(y_pred[:, task].sum() * 0.0)
            else:
                terms.append(criterion(y_pred[mask, task], y[mask, task].float()))
    return terms, counts


def safe_auc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score
    if len(y_true) == 0 or len(set(float(x) for x in y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def safe_logloss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1.0 - EPS)
    return float(-(true * np.log(pred) + (1.0 - true) * np.log(1.0 - pred)).mean())


def ensure_required_fields(path: str, required: Sequence[str]) -> Tuple[bool, List[str]]:
    """Small schema check used by the runner/report and unit tests."""

    if not os.path.exists(path):
        return False, list(required)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            fields = next(reader)
        except StopIteration:
            fields = []
    missing = [field for field in required if field not in fields]
    return not missing, missing
