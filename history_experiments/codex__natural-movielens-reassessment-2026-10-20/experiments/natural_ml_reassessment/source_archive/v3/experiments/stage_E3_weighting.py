"""Algorithmic primitives for Stage E3's static alpha response surface.

The implementation in this module is intentionally small and deterministic.  It
contains the only alpha-to-coefficient mapping used by the E3 runner and the
shared-parameter observer used for the contribution diagnostics.  The observer
uses ``torch.autograd.grad`` on the already-built six-cell graph; it never
touches ``Parameter.grad`` and never performs an optimizer update.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from stage_E1_instrumentation import (  # noqa: F401
    EPS,
    GROUP_NAMES,
    MAIN_GROUP_NAMES,
    ParameterPartition,
    build_parameter_partition,
    cell_index,
    contribution_views,
    flatten_parameter_tensors,
    iter_cells,
    masked_bce_terms,
    parameter_state_hash,
    preserve_rng_state,
    relative_l2_error,
    rng_state_digest,
    safe_auc,
    safe_logloss,
    write_json,
    write_rows_csv,
)


ALPHA_GRID: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
ALPHA_KEYS: Tuple[str, ...] = (
    "alpha_0p00", "alpha_0p25", "alpha_0p50", "alpha_0p75", "alpha_1p00"
)
ALPHA_KEY_BY_VALUE = dict(zip(ALPHA_GRID, ALPHA_KEYS))
VALUE_BY_ALPHA_KEY = dict(zip(ALPHA_KEYS, ALPHA_GRID))

# The E3 formal protocol is intentionally duplicated as a read-only contract so
# an E3 smoke cannot be mistaken for the formal 50-epoch matrix.
E3_FORMAL_PROTOCOL: Dict[str, object] = {
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
    "sampling": "DataLoader(shuffle=True,num_workers=0,drop_last=False)",
}

ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}
EXPECTED_SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
EXPECTED_SETTINGS: Tuple[str, ...] = tuple(ROLE_MAPS)


def _normalise_shares(domain_shares: Sequence[float]) -> np.ndarray:
    shares = np.asarray(domain_shares, dtype=np.float64)
    if shares.ndim != 1 or len(shares) == 0:
        raise ValueError("domain_shares must be a non-empty one-dimensional sequence")
    if not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("domain_shares must be finite and strictly positive")
    total = float(shares.sum())
    if total <= 0:
        raise ValueError("domain_shares must have a positive sum")
    return shares / total


def canonical_alpha(alpha: float) -> float:
    """Return the registered grid value, rejecting unregistered values."""

    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError("alpha must be finite")
    for candidate in ALPHA_GRID:
        if abs(value - candidate) <= 1.0e-10:
            return candidate
    raise ValueError("alpha must be one of {}".format(list(ALPHA_GRID)))


def alpha_key(alpha: float) -> str:
    return ALPHA_KEY_BY_VALUE[canonical_alpha(alpha)]


def alpha_from_key(key: str) -> float:
    try:
        return float(VALUE_BY_ALPHA_KEY[str(key)])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unknown alpha key {!r}".format(key)) from exc


def weighting_coefficients(
    alpha: float,
    domain_shares: Sequence[float],
    task_num: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the task-major E3 coefficient vector.

    ``c_d = D * p_d**alpha / sum_j p_j**alpha`` is the sole weighting rule.
    The same domain coefficient is copied for every task; there is no second
    task normalization.
    """

    value = canonical_alpha(alpha)
    shares = _normalise_shares(domain_shares)
    task_num = int(task_num)
    if task_num <= 0:
        raise ValueError("task_num must be positive")
    domain_num = len(shares)
    powers = np.power(shares, value)
    per_domain = domain_num * powers / float(powers.sum())
    flat = np.tile(per_domain, task_num)
    return torch.as_tensor(flat, dtype=dtype, device=device)


def weighting_coefficients_numpy(
    alpha: float, domain_shares: Sequence[float], task_num: int = 2
) -> np.ndarray:
    return weighting_coefficients(alpha, domain_shares, task_num,
                                  dtype=torch.float64).detach().cpu().numpy()


def alpha_coefficient_table(domain_shares: Sequence[float], task_num: int = 2) -> Dict[str, List[float]]:
    return {alpha_key(value): weighting_coefficients_numpy(value, domain_shares, task_num).tolist()
            for value in ALPHA_GRID}


def validate_alpha_coefficients(
    alpha: float,
    domain_shares: Sequence[float],
    coefficients: Sequence[float],
    task_num: int = 2,
    *,
    atol: float = 1.0e-8,
) -> Dict[str, object]:
    """Return an auditable coefficient check and raise on a failed contract."""

    expected = weighting_coefficients_numpy(alpha, domain_shares, task_num)
    if isinstance(coefficients, torch.Tensor):
        actual = coefficients.detach().cpu().numpy().astype(np.float64).reshape(-1)
    else:
        actual = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError("alpha coefficient vector does not match the registered formula")
    domain_num = len(_normalise_shares(domain_shares))
    if not np.all(actual > 0):
        raise AssertionError("all E3 coefficients must be positive")
    if not np.isclose(float(actual.sum()), domain_num * task_num, atol=1.0e-7):
        raise AssertionError("E3 six-cell coefficient sum is not task_num*domain_num")
    return {
        "alpha": canonical_alpha(alpha),
        "alpha_key": alpha_key(alpha),
        "expected": expected.tolist(),
        "actual": actual.tolist(),
        "max_abs_error": float(np.max(np.abs(actual - expected))) if len(actual) else 0.0,
        "domain_sum": float(actual[:domain_num].sum()),
        "cell_sum": float(actual.sum()),
    }


def weighted_loss_from_terms(loss_terms: Sequence[torch.Tensor], coefficients: torch.Tensor) -> torch.Tensor:
    """Form the exact explicit dot product consumed by the real backward path."""

    terms = list(loss_terms)
    coeff = coefficients.reshape(-1)
    if len(terms) != len(coeff):
        raise ValueError("loss term/coefficient lengths differ")
    if not terms:
        raise ValueError("loss_terms cannot be empty")
    return torch.stack(terms).to(device=coeff.device, dtype=coeff.dtype).dot(coeff)


# Short aliases make the intended API discoverable without adding another
# implementation of the formula.
alpha_to_coefficients = weighting_coefficients
loss_from_terms = weighted_loss_from_terms


def collect_shared_cell_gradients(
    loss_terms: Sequence[torch.Tensor],
    model: torch.nn.Module,
    partition: ParameterPartition,
    *,
    retain_graph: bool = True,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Collect raw shared gradients only, without mutating ``Parameter.grad``."""

    terms = list(loss_terms)
    named = dict(model.named_parameters())
    names = [name for name in partition.names("shared") if named[name].requires_grad]
    params = [named[name] for name in names]
    result: Dict[int, torch.Tensor] = {}
    unused: Dict[int, int] = {}
    for idx, term in enumerate(terms):
        if params:
            grads = torch.autograd.grad(term, params, retain_graph=retain_graph,
                                        allow_unused=True)
            vector, count = flatten_parameter_tensors(named, names, grads, device=term.device)
        else:
            vector, count = torch.zeros(0, dtype=term.dtype, device=term.device), 0
        result[idx] = vector.detach()
        unused[idx] = int(count)
    return result, unused


def _norm(value: Optional[torch.Tensor]) -> float:
    if value is None or not value.numel():
        return 0.0
    return float(torch.linalg.vector_norm(value.detach()).cpu())


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or not right.numel():
        return 0.0
    return float(torch.dot(left.reshape(-1), right.reshape(-1)).detach().cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or not right.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.detach().cpu()) <= EPS:
        return float("nan")
    return float(torch.clamp(torch.dot(left.reshape(-1), right.reshape(-1)) / denom,
                             -1.0, 1.0).detach().cpu())


def _contribution(raw: Mapping[int, torch.Tensor], coefficients: torch.Tensor,
                  domain_num: int, task_num: int) -> Dict[str, object]:
    return contribution_views(raw, coefficients, domain_num, task_num, eps=EPS)


E3_GRADIENT_STEP_FIELDS = [
    "seed", "epoch", "step", "domain", "task", "cell", "alpha", "alpha_key",
    "parameter_group", "weighting", "batch_fingerprint", "cell_sample_count",
    "loss_coefficient", "raw_gradient_norm", "raw_gradient_rms",
    "weighted_gradient_norm", "weighted_gradient_rms", "effective_projection",
    "normalized_contribution", "signed_contribution", "domain_signed_contribution",
    "magnitude_share", "overrepresentation_ratio", "mismatch_index",
    "equal_cf_contribution", "sample_cf_contribution", "unused_param_count",
    "actual_update_norm", "actual_update_alignment", "actual_update_projection",
    "gradient_reconstruction_error",
]


def make_shared_gradient_step_rows(
    raw_by_cell: Mapping[int, torch.Tensor],
    unused_by_cell: Mapping[int, int],
    *,
    alpha: float,
    coefficients: torch.Tensor,
    equal_coefficients: torch.Tensor,
    sample_coefficients: torch.Tensor,
    domain_num: int,
    task_num: int,
    shared_param_count: int,
    domain_shares: Sequence[float],
    seed: int,
    setting: str,
    batch_fingerprint: str,
    epoch: int,
    step: int,
    cell_sample_counts: Sequence[int],
    actual_update: Optional[torch.Tensor] = None,
    reconstruction_error: float = float("nan"),
) -> List[Dict[str, object]]:
    """Create one long-form shared contribution row per cell at a probe."""

    value = canonical_alpha(alpha)
    key = alpha_key(value)
    shares = _normalise_shares(domain_shares)
    coeff = coefficients.reshape(-1)
    if len(coeff) != domain_num * task_num or len(cell_sample_counts) != len(coeff):
        raise ValueError("coefficient/count vectors do not cover all cells")
    actual = _contribution(raw_by_cell, coeff, domain_num, task_num)
    equal = _contribution(raw_by_cell, equal_coefficients, domain_num, task_num)
    sample = _contribution(raw_by_cell, sample_coefficients, domain_num, task_num)
    magnitude = actual["magnitude"]
    normalized = actual["normalized"]
    projection = actual["projection"]
    domain_magnitude = {
        domain: sum(float(magnitude[cell_index(task, domain, domain_num)])
                    for task in range(task_num))
        for domain in range(domain_num)
    }
    domain_signed = {
        domain: sum(float(normalized[cell_index(task, domain, domain_num)])
                    for task in range(task_num))
        for domain in range(domain_num)
    }
    mismatch = 0.5 * sum(abs(domain_magnitude[d] - float(shares[d]))
                         for d in range(domain_num))
    update_norm = _norm(actual_update)
    rows: List[Dict[str, object]] = []
    for idx, domain, task in iter_cells(domain_num, task_num):
        raw_norm = _norm(raw_by_cell[idx])
        coefficient = float(coeff[idx].detach().cpu())
        weighted_norm = abs(coefficient) * raw_norm
        weighted_vec = raw_by_cell[idx] * coeff[idx].to(raw_by_cell[idx].device)
        rows.append({
            "seed": int(seed), "epoch": int(epoch), "step": int(step),
            "domain": int(domain), "task": int(task), "cell": int(idx),
            "alpha": value, "alpha_key": key, "parameter_group": "shared",
            "weighting": key, "setting": str(setting),
            "batch_fingerprint": str(batch_fingerprint),
            "cell_sample_count": int(cell_sample_counts[idx]),
            "loss_coefficient": coefficient,
            "raw_gradient_norm": raw_norm,
            "raw_gradient_rms": (raw_norm / math.sqrt(shared_param_count)
                                  if shared_param_count > 0 else 0.0),
            "weighted_gradient_norm": weighted_norm,
            "weighted_gradient_rms": (weighted_norm / math.sqrt(shared_param_count)
                                      if shared_param_count > 0 else 0.0),
            "effective_projection": float(projection[idx]),
            "normalized_contribution": float(normalized[idx]),
            "signed_contribution": float(normalized[idx]),
            "domain_signed_contribution": domain_signed[domain],
            "magnitude_share": float(domain_magnitude[domain]),
            "overrepresentation_ratio": domain_magnitude[domain] / (float(shares[domain]) + EPS),
            "mismatch_index": float(mismatch),
            "equal_cf_contribution": float(equal["normalized"][idx]),
            "sample_cf_contribution": float(sample["normalized"][idx]),
            "unused_param_count": int(unused_by_cell.get(idx, 0)),
            "actual_update_norm": update_norm,
            "actual_update_alignment": (_cosine(weighted_vec, actual_update)
                                         if actual_update is not None else float("nan")),
            "actual_update_projection": (_dot(weighted_vec, actual_update) / (update_norm + EPS)
                                         if actual_update is not None else float("nan")),
            "gradient_reconstruction_error": float(reconstruction_error),
        })
    return rows


@dataclass
class SharedGradientEpochAccumulator:
    """Full-vector Welford statistics for shared raw gradients only."""

    domain_num: int
    task_num: int

    def __post_init__(self) -> None:
        self.count = 0
        self.stats: Dict[int, Dict[str, object]] = {
            idx: {"mean": None, "m2": None, "sum_norm": 0.0, "sum_norm_sq": 0.0,
                  "values": [], "magnitude": [], "mismatch": []}
            for idx, _, _ in iter_cells(self.domain_num, self.task_num)
        }

    def update(self, raw_by_cell: Mapping[int, torch.Tensor],
               normalized: Mapping[int, float], mismatch: float,
               magnitude: Optional[Mapping[int, float]] = None) -> None:
        self.count += 1
        # ``magnitude_share`` is a domain-level quantity in both the step and
        # epoch schemas.  Repeat that domain value for the two task rows so a
        # downstream join cannot accidentally compare a cell share with a
        # domain share under the same field name.
        domain_magnitude = {}
        if magnitude is not None:
            domain_magnitude = {
                domain: sum(float(magnitude[cell_index(task, domain, self.domain_num)])
                            for task in range(self.task_num))
                for domain in range(self.domain_num)
            }
        for idx, domain, _ in iter_cells(self.domain_num, self.task_num):
            value = raw_by_cell[idx].detach()
            state = self.stats[idx]
            if state["mean"] is None:
                state["mean"] = torch.zeros_like(value)
                state["m2"] = torch.zeros_like(value)
            mean = state["mean"]
            m2 = state["m2"]
            delta = value - mean
            mean = mean + delta / self.count
            m2 = m2 + delta * (value - mean)
            state["mean"], state["m2"] = mean, m2
            norm = _norm(value)
            state["sum_norm"] += norm
            state["sum_norm_sq"] += norm * norm
            state["values"].append(float(normalized[idx]))
            if magnitude is not None:
                state["magnitude"].append(float(domain_magnitude[domain]))
            state["mismatch"].append(float(mismatch))

    def rows(self, *, alpha: float, coefficients: torch.Tensor, seed: int,
             setting: str, epoch: int) -> List[Dict[str, object]]:
        coeff = coefficients.detach().cpu().numpy().reshape(-1)
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            state = self.stats[idx]
            count = int(self.count)
            if count:
                mean_norm = float(state["sum_norm"]) / count
                norm_mean = _norm(state["mean"])
                if count >= 2:
                    variance = float(state["m2"].sum().detach().cpu()) / (count - 1)
                else:
                    variance = float("nan")
                values = np.asarray(state["values"], dtype=np.float64)
                mismatch = float(np.mean(state["mismatch"]))
            else:
                mean_norm = norm_mean = variance = float("nan")
                values = np.asarray([], dtype=np.float64)
                mismatch = float("nan")
            coefficient = float(coeff[idx])
            rows.append({
                "seed": int(seed), "setting": str(setting), "epoch": int(epoch),
                "domain": int(domain), "task": int(task), "cell": int(idx),
                "alpha": canonical_alpha(alpha), "alpha_key": alpha_key(alpha),
                "parameter_group": "shared", "weighting": alpha_key(alpha),
                "raw_gradient_norm": mean_norm,
                "weighted_gradient_norm": (mean_norm * abs(coefficient)
                                             if np.isfinite(mean_norm) else float("nan")),
                "gradient_variance": variance,
                "weighted_gradient_variance": (variance * coefficient * coefficient
                                                if np.isfinite(variance) else float("nan")),
                "gradient_SNR": (norm_mean * norm_mean / (variance + EPS)
                                  if np.isfinite(norm_mean) and np.isfinite(variance)
                                  else float("nan")),
                "norm_of_mean_gradient": norm_mean,
                "probe_count": count,
                "contribution_mean": float(values.mean()) if len(values) else float("nan"),
                "contribution_std": float(values.std()) if len(values) else float("nan"),
                "magnitude_share": (float(np.mean(state["magnitude"]))
                                    if state["magnitude"] else float("nan")),
                "mismatch_index": mismatch,
                "variance_scope": "shared_vector",
            })
        return rows


E3_GRADIENT_EPOCH_FIELDS = [
    "seed", "setting", "epoch", "domain", "task", "cell", "alpha", "alpha_key",
    "parameter_group", "weighting", "raw_gradient_norm", "weighted_gradient_norm",
    "gradient_variance", "weighted_gradient_variance", "gradient_SNR",
    "norm_of_mean_gradient", "probe_count", "contribution_mean", "contribution_std",
    "magnitude_share", "mismatch_index", "variance_scope",
]


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _numeric(value: object) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def normalise_dataset_path(path: object) -> str:
    """Return the stable path identity used by every E3 artifact.

    A raw command-line path is not a sufficient resume identity: the same
    dataset can be spelled with relative segments or different case on
    Windows.  The normalized absolute path is recorded alongside the human
    readable ``dataset_path`` field.
    """

    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def split_row_fingerprint(x_values: object, y_values: object) -> str:
    """Hash ordered feature/label rows for a split-level resume identity."""

    hasher = hashlib.sha256()
    for value in (x_values, y_values):
        array = np.ascontiguousarray(np.asarray(value))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        if array.dtype.hasobject:
            hasher.update(repr(array.tolist()).encode("utf-8"))
        else:
            hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def strict_validation_macro_auc(
    stats: Mapping[int, Mapping[str, object]],
    expected_cells: Iterable[int],
    *,
    label: str = "validation",
) -> float:
    """Compute a checkpoint macro AUC only when every required cell is valid.

    This is deliberately strict at the checkpoint boundary.  Dropping a
    missing/non-finite cell before taking a macro mean can silently select a
    checkpoint on a different population, so callers must fail the run and
    make the invalid artifact visible instead.
    """

    values: List[float] = []
    invalid: List[str] = []
    for cell in expected_cells:
        record = stats.get(int(cell))
        if not isinstance(record, Mapping):
            invalid.append(f"{int(cell)}:missing")
            continue
        auc = _numeric(record.get("auc"))
        logloss = _numeric(record.get("logloss"))
        if auc is None:
            invalid.append(f"{int(cell)}:auc")
        if logloss is None:
            invalid.append(f"{int(cell)}:logloss")
        if auc is not None:
            values.append(auc)
    if invalid:
        raise RuntimeError(
            f"{label} checkpoint cells are incomplete/nonfinite: {','.join(invalid)}")
    if not values:
        raise RuntimeError(f"{label} checkpoint has no finite cells")
    return float(np.mean(values))


def _manifest_identity(manifest: Mapping[str, object]) -> Dict[str, object]:
    return {
        "dataset": manifest.get("dataset"),
        "dataset_identity": manifest.get("dataset_identity"),
        "device_identity": manifest.get("device_identity"),
        "domain_num": manifest.get("domain_num"),
        "task_num": manifest.get("task_num"),
        "role_by_domain": manifest.get("role_by_domain"),
        "role_sizes": manifest.get("role_sizes"),
        "train_domain_counts": manifest.get("train_domain_counts"),
        "domain_shares": manifest.get("domain_shares"),
        "split_counts": manifest.get("split_counts"),
        "split_row_fingerprints": manifest.get("split_row_fingerprints"),
        "positive_rates": manifest.get("positive_rates"),
        "initial_parameter_state_hash": manifest.get("initial_parameter_state_hash"),
        "partition": manifest.get("partition"),
        "model_identity": manifest.get("model_identity"),
    }


def _int_value(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _contiguous_epochs(value: object) -> Optional[List[int]]:
    if not isinstance(value, list) or not value:
        return None
    parsed = [_int_value(item) for item in value]
    if any(item is None for item in parsed):
        return None
    result = [int(item) for item in parsed if item is not None]
    return result if result == list(range(len(result))) else None


def _oracle_selected_cells(rows: Sequence[Mapping[str, str]]) -> Dict[int, Mapping[str, str]]:
    selected: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        if _int_value(row.get("selected_checkpoint"), 0) != 1:
            continue
        task = _int_value(row.get("task"))
        if task in (0, 1):
            selected[int(task)] = row
    return selected


def _oracle_metric_rows_valid(
    rows: Sequence[Mapping[str, str]], observed_epochs: Sequence[int],
) -> Tuple[bool, str, Dict[int, Mapping[str, str]]]:
    by_epoch: Dict[int, List[Mapping[str, str]]] = {}
    for row in rows:
        epoch = _int_value(row.get("epoch"))
        task = _int_value(row.get("task"))
        domain = _int_value(row.get("domain"))
        if epoch is None or task not in (0, 1) or domain != 0:
            return False, "metrics_row_identity", {}
        by_epoch.setdefault(epoch, []).append(row)
    for epoch in observed_epochs:
        epoch_rows = by_epoch.get(int(epoch), [])
        if len(epoch_rows) != 2 or {int(_int_value(row.get("task"), -1)) for row in epoch_rows} != {0, 1}:
            return False, "metrics_epoch_cell_count", {}
        for row in epoch_rows:
            for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"):
                if _numeric(row.get(field)) is None:
                    return False, f"metrics_nonfinite:{epoch}:{field}", {}
    selected = _oracle_selected_cells(rows)
    if set(selected) != {0, 1}:
        return False, "selected_checkpoint_cells", selected
    selected_epochs = {_int_value(row.get("epoch")) for row in selected.values()}
    if len(selected_epochs) != 1 or next(iter(selected_epochs)) not in set(observed_epochs):
        return False, "selected_checkpoint_epoch", selected
    for row in selected.values():
        for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"):
            if _numeric(row.get(field)) is None:
                return False, f"selected_nonfinite:{field}", selected
    return True, "OK", selected


def _valid_oracle_dir(path: str, *, setting: str, seed: int, domain: int,
                      role_sizes: Mapping[str, int],
                      dataset_identity: Optional[str] = None,
                      device_identity: Optional[str] = None,
                      expected_split_row_fingerprints: Optional[Mapping[str, str]] = None,
                      ) -> Tuple[bool, str, Dict[str, object]]:
    """Validate a seed-level *formal* oracle artifact.

    The resolver is intentionally stricter than the runner's smoke path.  A
    short run, even if it has plausible metrics, cannot become a reference
    checkpoint because formal protocol, row identity, observed support, and
    the saved model must all be present and internally consistent.
    """

    manifest = _read_json(os.path.join(path, "run_manifest.json"))
    if not manifest:
        return False, "manifest_missing", {}
    required = ("config.json", "run_manifest.json", "metrics_epoch.csv", "metrics_summary.csv",
                "stdout.log", "model.pth")
    missing = [name for name in required if not os.path.exists(os.path.join(path, name))]
    if missing:
        return False, "missing_files:" + ",".join(missing), manifest
    config = _read_json(os.path.join(path, "config.json"))
    if not config:
        return False, "config_missing_or_unreadable", manifest
    role = manifest.get("role")
    split_counts = manifest.get("split_counts")
    positive_rates = manifest.get("positive_rates")

    def _positive_split_count(name: str, task: int) -> bool:
        try:
            value = split_counts[name].get(str(task), split_counts[name].get(task, 0))
            return int(value) > 0
        except (AttributeError, TypeError, ValueError):
            return False

    split_identity_ok = (
        isinstance(split_counts, Mapping)
        and all(name in split_counts for name in ("train", "validation", "test"))
        and all(isinstance(split_counts[name], Mapping) for name in ("train", "validation", "test"))
        and all(_positive_split_count(name, task)
                for name in ("train", "validation", "test") for task in range(2))
    )
    positive_identity_ok = isinstance(positive_rates, Mapping) and all(
        name in positive_rates and isinstance(positive_rates[name], Mapping)
        for name in ("train", "validation", "test"))
    role_count_ok = role in role_sizes and _int_value(manifest.get("train_count"), -1) == int(role_sizes[role])
    normalized_dataset = normalise_dataset_path(manifest.get("dataset_path", "")) \
        if manifest.get("dataset_path") else ""
    manifest_dataset_identity = str(manifest.get("dataset_identity", ""))
    config_dataset_identity = str(config.get("dataset_identity", ""))
    expected_dataset = normalise_dataset_path(dataset_identity) if dataset_identity else ""
    split_rows = manifest.get("split_row_fingerprints")
    config_split_rows = config.get("split_row_fingerprints")
    split_row_identity_ok = (
        isinstance(split_rows, Mapping)
        and all(isinstance(split_rows.get(name), str) and bool(split_rows.get(name))
                for name in ("train", "validation", "test"))
        and split_rows == config_split_rows
    )
    external_device_identity_ok = (
        bool(device_identity) and manifest.get("device_identity") == str(device_identity))
    external_split_row_identity_ok = (
        isinstance(split_rows, Mapping)
        and isinstance(expected_split_row_fingerprints, Mapping)
        and dict(split_rows) == dict(expected_split_row_fingerprints)
    )
    observed = _contiguous_epochs(manifest.get("observed_epochs"))
    stopped = manifest.get("stopped_epoch")
    stopped_epoch = None if stopped in (None, "") else _int_value(stopped)
    configured_epoch = _int_value(config.get("epoch"))
    stop_identity_ok = (
        observed is not None and configured_epoch == int(E3_FORMAL_PROTOCOL["epoch"])
        and (stopped_epoch is None and observed[-1] == configured_epoch - 1
             or stopped_epoch is not None and stopped_epoch == observed[-1]
             and _int_value(manifest.get("censored_after_epoch")) == stopped_epoch + 1)
    )
    checks = {
        "stage": manifest.get("stage") == "E3_ORACLE",
        "status": manifest.get("status") == "OK",
        "oracle": manifest.get("oracle") is True,
        "setting": str(manifest.get("setting")) == str(setting),
        "seed": _int_value(manifest.get("seed"), -1) == int(seed),
        "domain": _int_value(manifest.get("original_physical_domain"), -1) == int(domain),
        "role": role in set(role_sizes),
        "train_count": role_count_ok,
        "domain_task_schema": _int_value(manifest.get("domain_num"), -1) == 1
        and _int_value(manifest.get("task_num"), -1) == 2,
        "split_identity": split_identity_ok,
        "positive_rate_identity": positive_identity_ok,
        "split_row_identity": split_row_identity_ok,
        "dataset_identity": bool(expected_dataset) and bool(manifest_dataset_identity)
        and manifest_dataset_identity == normalized_dataset
        and config_dataset_identity == manifest_dataset_identity
        and (not expected_dataset or manifest_dataset_identity == expected_dataset),
        "device_identity": bool(manifest.get("device_identity"))
        and str(config.get("device_identity", "")) == str(manifest.get("device_identity")),
        "external_device_identity": external_device_identity_ok,
        "external_split_row_identity": external_split_row_identity_ok,
        "initial_state_identity": bool(manifest.get("initial_parameter_state_hash")),
        "partition_identity": isinstance(manifest.get("partition"), Mapping),
        "model_identity": isinstance(manifest.get("model_identity"), Mapping),
        "alpha_absent": manifest.get("alpha") in (None, ""),
        "formal_protocol": manifest.get("formal_protocol") == E3_FORMAL_PROTOCOL
        and config.get("formal_protocol") == E3_FORMAL_PROTOCOL,
        "formal_hyperparameters": all(
            config.get(field) == E3_FORMAL_PROTOCOL.get(field)
            for field in (
                "epoch", "batch_size", "learning_rate", "learning_rate_darts",
                "weight_decay", "expert_num", "earlystop_patience", "role_sizes",
                "probe_interval", "instrument", "fix_optim_grouping", "backbone",
                "sampling", "task", "domain",
            )
        ),
        "formal_valid": manifest.get("formal_valid") is True
        and config.get("formal_valid") is True,
        "formal_stop_support": stop_identity_ok,
    }
    bad = [name for name, passed in checks.items() if not passed]
    if bad:
        return False, "identity:" + ",".join(bad), manifest

    metrics = _read_csv(os.path.join(path, "metrics_epoch.csv"))
    metric_ok, metric_reason, selected = _oracle_metric_rows_valid(metrics, observed or [])
    if not metric_ok:
        return False, metric_reason, manifest
    summary = _read_csv(os.path.join(path, "metrics_summary.csv"))
    summary_cells: Dict[int, Mapping[str, str]] = {}
    for row in summary:
        task = _int_value(row.get("task"))
        row_domain = _int_value(row.get("domain"))
        if task in (0, 1) and row_domain == 0:
            summary_cells[int(task)] = row
    if set(summary_cells) != {0, 1}:
        return False, "summary_missing_cell", manifest
    for task, row in summary_cells.items():
        for field in ("best_val_auc", "best_val_logloss", "best_test_auc", "best_test_logloss"):
            if _numeric(row.get(field)) is None:
                return False, f"summary_nonfinite:{task}:{field}", manifest
        if _int_value(row.get("best_epoch")) != _int_value(selected[task].get("epoch")):
            return False, "summary_checkpoint_mismatch", manifest
    selected_epochs = {_int_value(row.get("epoch")) for row in selected.values()}
    selected_complete = manifest.get("selected_checkpoint_complete") is True \
        and set(manifest.get("selected_checkpoint_cells", [])) in ({"0", "1"}, {0, 1})
    if not selected_complete or len(selected_epochs) != 1:
        return False, "selected_checkpoint_incomplete", manifest
    return True, "OK", manifest


def validate_oracle_artifact(
    path: str, *, setting: str, seed: int, domain: int,
    role_sizes: Mapping[str, int] = E3_FORMAL_PROTOCOL["role_sizes_parsed"],  # type: ignore[arg-type]
    dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
    expected_split_row_fingerprints: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str, Dict[str, object]]:
    """Public formal-oracle validator shared by the runner and orchestrator."""

    return _valid_oracle_dir(path, setting=setting, seed=seed, domain=domain,
                             role_sizes=role_sizes, dataset_identity=dataset_identity,
                             device_identity=device_identity,
                             expected_split_row_fingerprints=expected_split_row_fingerprints)


def resolve_oracle_metrics(
    roots: Sequence[str] | str,
    setting: str,
    seed: int,
    *,
    role_sizes: Mapping[str, int] = E3_FORMAL_PROTOCOL["role_sizes_parsed"],  # type: ignore[arg-type]
    domain_num: int = 3,
    task_num: int = 2,
    dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
    expected_split_row_fingerprints: Optional[Mapping[int, Mapping[str, str]]] = None,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    """Resolve only complete seed-level oracle artifacts.

    Markdown summaries and legacy aggregate CSVs without an explicit manifest
    are intentionally rejected.  Reuse also requires the caller's external
    device and remapped split-row identities; artifact-local self-consistency
    is not sufficient.  The returned coverage is metadata only; it is never
    used to substitute a joint endpoint or an equal-arm value.
    """

    root_list = [roots] if isinstance(roots, (str, os.PathLike)) else list(roots)
    result = {idx: {"auc": "", "logloss": "", "source": "missing", "coverage": False}
              for idx, _, _ in iter_cells(domain_num, task_num)}
    audit: Dict[str, object] = {"setting": setting, "seed": int(seed), "domains": [],
                                "coverage_cells": 0, "expected_cells": domain_num * task_num}
    for domain in range(domain_num):
        chosen = None
        reason = "missing"
        for root in root_list:
            if not root:
                continue
            candidates = [
                os.path.join(os.path.abspath(str(root)), setting, f"d{domain}", f"seed{seed}"),
                os.path.join(os.path.abspath(str(root)), "oracle_reference", setting,
                             f"d{domain}", f"seed{seed}"),
                os.path.join(os.path.abspath(str(root)), "ratio_50", setting,
                             f"oracle_d{domain}", f"seed{seed}"),
            ]
            for candidate in candidates:
                expected_rows = (
                    expected_split_row_fingerprints.get(domain,
                        expected_split_row_fingerprints.get(str(domain)))
                    if isinstance(expected_split_row_fingerprints, Mapping) else None)
                ok, why, manifest = _valid_oracle_dir(candidate, setting=setting,
                                                       seed=seed, domain=domain,
                                                       role_sizes=role_sizes,
                                                       dataset_identity=dataset_identity,
                                                       device_identity=device_identity,
                                                       expected_split_row_fingerprints=expected_rows)
                if ok:
                    chosen = candidate
                    break
                if why != "manifest_missing":
                    reason = why
            if chosen:
                break
        entry = {"domain": domain, "status": "reused" if chosen else "missing",
                 "source": chosen or "", "reason": "OK" if chosen else reason}
        if chosen:
            rows = _read_csv(os.path.join(chosen, "metrics_summary.csv"))
            manifest = _read_json(os.path.join(chosen, "run_manifest.json"))
            for task in range(task_num):
                row = next((item for item in rows if int(item.get("task", -1)) == task), None)
                if row is None:
                    row = next((item for item in rows
                                if int(item.get("original_physical_domain", -1)) == domain
                                and int(item.get("task", -1)) == task), None)
                if row is None:
                    continue
                idx = cell_index(task, domain, domain_num)
                result[idx] = {
                    "auc": row.get("best_test_auc", row.get("oracle_auc", "")),
                    "logloss": row.get("best_test_logloss", row.get("oracle_logloss", "")),
                    "source": os.path.abspath(chosen), "coverage": True,
                    "original_physical_domain": domain, "setting": setting, "seed": int(seed),
                    "role": manifest.get("role", ROLE_MAPS.get(setting, {}).get(domain, "")),
                    "train_count": manifest.get("train_count", ""),
                }
        audit["domains"].append(entry)
    audit["coverage_cells"] = sum(1 for value in result.values() if value.get("coverage"))
    audit["coverage_fraction"] = float(audit["coverage_cells"]) / float(audit["expected_cells"] or 1)
    audit["complete"] = bool(audit["coverage_cells"] == audit["expected_cells"])
    return result, audit


def compare_alpha_block(
    run_dirs: Mapping[str, str] | Sequence[Tuple[str, str]],
    *,
    require_formal: bool = False,
    loss_atol: float = 1.0e-7,
) -> Dict[str, object]:
    """Mechanically compare five alpha arms for one setting/seed block.

    The only support difference permitted by E3 is a legal early-stop tail.
    Every observed epoch must be a complete prefix with all train steps and one
    architecture step; arbitrary set intersection is deliberately rejected.
    """

    if isinstance(run_dirs, Mapping):
        paths = {str(key): str(value) for key, value in run_dirs.items()}
    else:
        paths = {str(key): str(value) for key, value in run_dirs}
    missing = [key for key in ALPHA_KEYS if not os.path.isdir(paths.get(key, ""))]
    if missing:
        return {"status": "PENDING", "valid": None, "missing_alpha_keys": missing}
    mismatches: List[Dict[str, object]] = []
    manifests = {key: _read_json(os.path.join(paths[key], "run_manifest.json"))
                 for key in ALPHA_KEYS}
    configs = {key: _read_json(os.path.join(paths[key], "config.json"))
              for key in ALPHA_KEYS}
    if any(not manifests[key] or not configs[key] for key in ALPHA_KEYS):
        return {"status": "INVALID", "valid": False, "reason": "manifest_or_config_missing"}
    base_m = manifests[ALPHA_KEYS[0]]
    base_c = configs[ALPHA_KEYS[0]]
    identity_fields = (
        "seed", "setting", "dataset", "dataset_identity", "device_identity",
        "domain_num", "task_num", "role_by_domain", "role_sizes",
        "train_domain_counts", "domain_shares", "split_counts",
        "split_row_fingerprints", "positive_rates", "initial_parameter_state_hash",
        "partition", "model_identity", "formal_protocol",
    )
    required_identity_fields = ("dataset_identity", "device_identity", "split_row_fingerprints")
    for key in ALPHA_KEYS:
        for field in required_identity_fields:
            value = manifests[key].get(field)
            if field == "split_row_fingerprints":
                present = isinstance(value, Mapping) and all(
                    isinstance(value.get(name), str) and bool(value.get(name))
                    for name in ("validation", "test"))
            else:
                present = bool(value)
            if not present:
                mismatches.append({"field": field, "alpha_key": key, "reason": "identity_missing"})
    for key in ALPHA_KEYS[1:]:
        try:
            expected_alpha = alpha_from_key(key)
            if abs(float(manifests[key].get("alpha")) - expected_alpha) > 1.0e-10:
                mismatches.append({"alpha_key": key, "field": "manifest_alpha"})
            if abs(float(configs[key].get("alpha")) - expected_alpha) > 1.0e-10:
                mismatches.append({"alpha_key": key, "field": "config_alpha"})
        except (TypeError, ValueError):
            mismatches.append({"alpha_key": key, "field": "alpha_malformed"})
        for field in identity_fields:
            if json.dumps(manifests[key].get(field), sort_keys=True) != json.dumps(
                    base_m.get(field), sort_keys=True):
                mismatches.append({"field": field, "alpha_key": key})
        for field in ("epoch", "batch_size", "learning_rate", "learning_rate_darts", "weight_decay",
                      "expert_num", "probe_interval", "instrument", "earlystop_patience", "role_sizes",
                      "fix_optim_grouping", "backbone", "sampling", "dataset_identity",
                      "device_identity", "split_row_fingerprints", "task", "domain"):
            if json.dumps(configs[key].get(field), sort_keys=True) != json.dumps(
                    base_c.get(field), sort_keys=True):
                mismatches.append({"config_field": field, "alpha_key": key})
        for field in required_identity_fields:
            if json.dumps(configs[key].get(field), sort_keys=True) != json.dumps(
                    manifests[key].get(field), sort_keys=True):
                mismatches.append({"config_manifest_field": field, "alpha_key": key})
    try:
        if abs(float(base_m.get("alpha")) - alpha_from_key(ALPHA_KEYS[0])) > 1.0e-10:
            mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "manifest_alpha"})
        if abs(float(base_c.get("alpha")) - alpha_from_key(ALPHA_KEYS[0])) > 1.0e-10:
            mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "config_alpha"})
    except (TypeError, ValueError):
        mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "alpha_malformed"})
    if json.dumps(base_c.get("dataset_identity"), sort_keys=True) != json.dumps(
            base_m.get("dataset_identity"), sort_keys=True):
        mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "config_manifest_dataset_identity"})
    if json.dumps(base_c.get("device_identity"), sort_keys=True) != json.dumps(
            base_m.get("device_identity"), sort_keys=True):
        mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "config_manifest_device_identity"})
    if json.dumps(base_c.get("split_row_fingerprints"), sort_keys=True) != json.dumps(
            base_m.get("split_row_fingerprints"), sort_keys=True):
        mismatches.append({"alpha_key": ALPHA_KEYS[0], "field": "config_manifest_split_rows"})

    traces: Dict[str, List[Dict[str, str]]] = {}
    trace_maps: Dict[str, Dict[Tuple[str, int, int], Mapping[str, str]]] = {}
    epoch_support: Dict[str, Dict[int, Dict[Tuple[str, int], Mapping[str, str]]]] = {}
    observed_epochs: Dict[str, List[int]] = {}
    for key in ALPHA_KEYS:
        traces[key] = _read_csv(os.path.join(paths[key], "batch_trace.csv"))
        if not traces[key]:
            mismatches.append({"alpha_key": key, "reason": "batch_trace_missing_or_empty"})
        parsed_observed = _contiguous_epochs(manifests[key].get("observed_epochs"))
        if parsed_observed is None:
            mismatches.append({"alpha_key": key, "reason": "observed_epochs_not_contiguous_prefix"})
            parsed_observed = []
        observed_epochs[key] = parsed_observed
        stopped = manifests[key].get("stopped_epoch")
        stopped_epoch = None if stopped in (None, "") else _int_value(stopped)
        censored = manifests[key].get("censored_after_epoch")
        if stopped_epoch is not None:
            if not parsed_observed or stopped_epoch != parsed_observed[-1] \
                    or _int_value(censored) != stopped_epoch + 1:
                mismatches.append({"alpha_key": key, "reason": "illegal_stop_tail"})
        configured_epoch = _int_value(configs[key].get("epoch"))
        if parsed_observed and configured_epoch is not None and parsed_observed[-1] + 1 < configured_epoch \
                and stopped_epoch is None:
            mismatches.append({"alpha_key": key, "reason": "short_run_without_early_stop"})
        trace_maps[key] = {}
        epoch_support[key] = {}
        for row in traces[key]:
            if row.get("kind") not in ("train", "architecture"):
                continue
            try:
                kind = str(row.get("kind", ""))
                epoch = int(row.get("epoch", -1))
                step = int(row.get("step", -1))
                support = (kind, epoch, step)
            except (TypeError, ValueError):
                mismatches.append({"alpha_key": key, "reason": "malformed_batch_trace_row"})
                continue
            if support in trace_maps[key]:
                mismatches.append({"alpha_key": key, "support": support, "reason": "duplicate_trace_row"})
                continue
            trace_maps[key][support] = row
            epoch_support[key].setdefault(epoch, {})[(kind, step)] = row
        # Enforce complete ordered support for every observed epoch.  Extra
        # epochs/rows are also invalid, so support cannot be rescued by an
        # arbitrary intersection with another arm.
        if set(epoch_support[key]) != set(parsed_observed):
            mismatches.append({"alpha_key": key, "reason": "trace_epoch_support_mismatch"})
        for epoch in parsed_observed:
            support = epoch_support[key].get(epoch, {})
            train_steps = sorted(step for kind, step in support if kind == "train")
            architecture_steps = sorted(step for kind, step in support if kind == "architecture")
            if train_steps != list(range(len(train_steps))) or not train_steps:
                mismatches.append({"alpha_key": key, "epoch": epoch,
                                   "reason": "train_steps_not_complete"})
            if architecture_steps != [-1]:
                mismatches.append({"alpha_key": key, "epoch": epoch,
                                   "reason": "architecture_step_count"})

    common_end = min((epochs[-1] for epochs in observed_epochs.values() if epochs), default=-1)
    common = list(range(common_end + 1)) if common_end >= 0 else []
    if not common:
        mismatches.append({"reason": "empty_common_trace_support"})
    # A shorter arm is legal only when its missing epochs are a suffix caused
    # by an explicitly recorded early stop/censored tail.
    max_end = max((epochs[-1] for epochs in observed_epochs.values() if epochs), default=-1)
    for key in ALPHA_KEYS:
        epochs = observed_epochs[key]
        if epochs and epochs != list(range(epochs[-1] + 1)):
            mismatches.append({"alpha_key": key, "reason": "non_prefix_observed_epochs"})
        if epochs and epochs[-1] < max_end:
            stopped = manifests[key].get("stopped_epoch")
            censored = manifests[key].get("censored_after_epoch")
            if stopped in (None, "") or _int_value(stopped) != epochs[-1] \
                    or _int_value(censored) != epochs[-1] + 1:
                mismatches.append({"alpha_key": key, "reason": "tail_difference_not_censored"})
    # Compare the complete common support in deterministic order.
    common_trace_count = 0
    for epoch in common:
        support_lists = {
            key: sorted(epoch_support[key].get(epoch, {}).keys()) for key in ALPHA_KEYS
        }
        first_support = support_lists[ALPHA_KEYS[0]]
        if any(support_lists[key] != first_support for key in ALPHA_KEYS[1:]):
            mismatches.append({"epoch": epoch, "reason": "common_support_shape_mismatch"})
            continue
        common_trace_count += len(first_support)
        for kind, step in first_support:
            first = epoch_support[ALPHA_KEYS[0]][epoch][(kind, step)]
            for key in ALPHA_KEYS[1:]:
                row = epoch_support[key][epoch][(kind, step)]
                for field in ("batch_fingerprint", "batch_size", "domain_counts", "task_counts"):
                    if str(row.get(field, "")) != str(first.get(field, "")):
                        mismatches.append({"epoch": epoch, "kind": kind, "step": step,
                                           "field": field, "alpha_key": key})
    formal_bad = []
    if require_formal:
        for key in ALPHA_KEYS:
            if manifests[key].get("formal_valid") is not True \
                    or configs[key].get("formal_valid") is not True:
                formal_bad.append(key)
    if formal_bad:
        mismatches.append({"reason": "formal_invalid", "alpha_keys": formal_bad})
    status = "VALID" if not mismatches else "INVALID"
    return {
        "status": status, "valid": status == "VALID", "mismatches": mismatches,
        "common_epochs": common, "common_trace_count": common_trace_count,
        "common_support_valid": status == "VALID", "tail_only_difference": status == "VALID",
        "alpha_keys": list(ALPHA_KEYS),
        "observed_epoch_counts": {key: len(observed_epochs[key]) for key in ALPHA_KEYS},
        "censored": {key: manifests[key].get("censored_after_epoch") for key in ALPHA_KEYS},
    }


def json_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = [
    "ALPHA_GRID", "ALPHA_KEYS", "ALPHA_KEY_BY_VALUE", "VALUE_BY_ALPHA_KEY",
    "E3_FORMAL_PROTOCOL", "ROLE_MAPS", "EXPECTED_SEEDS", "EXPECTED_SETTINGS",
    "canonical_alpha", "alpha_key", "alpha_from_key", "weighting_coefficients",
    "weighting_coefficients_numpy", "alpha_coefficient_table",
    "validate_alpha_coefficients", "weighted_loss_from_terms", "loss_from_terms",
    "normalise_dataset_path", "split_row_fingerprint", "strict_validation_macro_auc",
    "collect_shared_cell_gradients", "make_shared_gradient_step_rows",
    "SharedGradientEpochAccumulator", "E3_GRADIENT_STEP_FIELDS", "E3_GRADIENT_EPOCH_FIELDS",
    "resolve_oracle_metrics", "validate_oracle_artifact", "compare_alpha_block", "json_hash", "masked_bce_terms",
    "build_parameter_partition", "parameter_state_hash", "preserve_rng_state",
    "rng_state_digest", "relative_l2_error", "safe_auc", "safe_logloss", "write_json",
    "write_rows_csv", "iter_cells", "cell_index",
]
