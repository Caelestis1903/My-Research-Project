"""Single-run implementation for Stage E6 adaptive contribution calibration.

This entry point keeps the E3 MovieLens model, data, sampler, optimizers,
architecture cadence, and checkpoint rule.  The only main-objective change is
the registered E6 arm coefficient chosen from the current batch's detached
shared raw-gradient magnitudes.  It emits observations and does not interpret
the matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)

from stage_E1_instrumentation import (  # noqa: E402
    CELL_ORDER_NOTE,
    EPS,
    RoutingObserver,
    batch_fingerprint,
    collect_nonfinite_inventory,
    contribution_views,
    flatten_parameter_tensors,
    parameter_state_hash,
    preserve_rng_state,
    relative_l2_error,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_E3_run import (  # noqa: E402
    E3Runner,
    E3_FORMAL_PROTOCOL,
    METRICS_FIELDS as E3_METRICS_FIELDS,
    _cpu_state_dict,
    _finite_number,
    _hash_json,
    _json_counts,
    _metric_records,
    parse_role_sizes,
    role_reversal_sample,
    setup_seed,
    strict_validation_macro_auc,
)
from stage_E3_weighting import (  # noqa: E402
    ROLE_MAPS,
    collect_shared_cell_gradients,
    iter_cells,
    masked_bce_terms,
    normalise_dataset_path,
    resolve_oracle_metrics,
    split_row_fingerprint,
    weighted_loss_from_terms,
)
from stage_E6_calibration import (  # noqa: E402
    E6_ARM_KEYS,
    E6_ARM_REGISTRY,
    E6_BETA,
    E6_FORMAL_PROTOCOL,
    E6_ROLE_SIZES,
    E6_FORMAL_SETTING,
    EMAMagnitudeState,
    calibration_metrics,
    closed_form_coefficients,
    coefficient_variation,
    normalise_shares,
    raw_domain_magnitudes,
    registry_hash,
    static_coefficients,
    target_shares,
    validate_arm_key,
)


E6_METRICS_FIELDS = [
    "seed", "epoch", "arm_key", "controller", "target", "domain", "task", "cell",
    "setting", "physical_domain", "size_role", "domain_size", "domain_share",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate",
    "train_auc", "train_logloss", "validation_auc", "validation_logloss",
    "test_auc", "test_logloss", "selected_checkpoint", "stopped_censored",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage",
    "transfer_auc", "transfer_logloss",
]

E6_METRICS_SUMMARY_FIELDS = [
    "seed", "setting", "arm_key", "controller", "target", "domain", "physical_domain",
    "task", "cell", "size_role", "domain_size", "domain_share", "selected_epoch",
    "stopped_epoch", "best_val_auc", "best_val_logloss", "best_test_auc",
    "best_test_logloss", "oracle_auc", "oracle_logloss", "oracle_source",
    "oracle_coverage", "transfer_auc", "transfer_logloss", "overall_auc",
    "overall_logloss", "worst_domain_auc", "worst_cell_auc", "negative_transfer",
    "shared_raw_gradient_norm", "shared_weighted_gradient_norm", "shared_magnitude_share",
    "shared_signed_contribution", "shared_mismatch_index", "target_calibration_error",
    "coefficient_variation", "ema_tracking_error", "gradient_variance", "gradient_SNR",
    "shared_update_alignment", "common_support", "last_common_epoch",
    "last_common_test_auc", "last_common_test_logloss", "last_common_transfer_auc",
    "last_common_transfer_logloss", "censored",
]

E6_BATCH_FIELDS = [
    "seed", "setting", "arm_key", "epoch", "step", "kind", "batch_fingerprint",
    "batch_size", "domain_counts", "task_counts", "is_probe", "train_loss",
]

E6_CONTROLLER_STEP_FIELDS = [
    "seed", "setting", "arm_key", "epoch", "step", "batch_fingerprint", "batch_size",
    "domain_counts", "p_d", "tau_d", "raw_r_d", "ema_hat_r_d", "q_d", "beta",
    "controller_statistic", "controller_feedback", "proposed_c_d", "applied_c_d",
    "coefficient_sum_domain", "coefficient_sum_cell", "actual_magnitude_share",
    "data_share_mismatch", "target_calibration_error", "ema_tracking_error",
    "coefficient_variation", "fallback_flag", "error_flag", "main_loss",
    "gradient_reconstruction_error", "actual_update_alignment", "actual_update_norm",
]

E6_SHARED_STEP_FIELDS = [
    "seed", "setting", "arm_key", "epoch", "step", "domain", "task", "cell",
    "parameter_group", "batch_fingerprint", "cell_sample_count", "p_d", "tau_d",
    "raw_domain_magnitude", "ema_domain_magnitude", "q_domain_magnitude",
    "loss_coefficient", "raw_gradient_norm", "raw_gradient_rms",
    "weighted_gradient_norm", "weighted_gradient_rms", "effective_projection",
    "normalized_contribution", "signed_contribution", "domain_signed_contribution",
    "magnitude_share", "overrepresentation_ratio", "mismatch_index",
    "target_calibration_error", "equal_cf_contribution", "sample_cf_contribution",
    "unused_param_count", "actual_update_norm", "actual_update_alignment",
    "actual_update_projection", "gradient_reconstruction_error", "probe",
]

E6_SHARED_EPOCH_FIELDS = [
    "seed", "setting", "arm_key", "epoch", "domain", "task", "cell",
    "parameter_group", "raw_gradient_norm", "weighted_gradient_norm",
    "gradient_variance", "weighted_gradient_variance", "gradient_SNR",
    "norm_of_mean_gradient", "probe_count", "contribution_mean", "contribution_std",
    "magnitude_share", "mismatch_index", "target_calibration_error",
    "ema_tracking_error", "coefficient_variation", "coefficient_mean",
    "update_alignment_mean", "variance_scope",
]

E6_CONTROLLER_EPOCH_FIELDS = [
    "seed", "setting", "arm_key", "epoch", "controller_statistic", "target",
    "update_count", "raw_r_mean", "q_mean", "target_calibration_error_mean",
    "data_share_mismatch_mean", "ema_tracking_error_mean", "coefficient_variation_mean",
    "coefficient_min", "coefficient_max", "coefficient_mean", "coefficient_sum_domain",
    "coefficient_sum_cell", "fallback_count", "error_count",
]


def _json_vector(values: object) -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return json.dumps(values, separators=(",", ":"), ensure_ascii=False)


def _norm(value: Optional[torch.Tensor]) -> float:
    if value is None or not value.numel():
        return 0.0
    return float(torch.linalg.vector_norm(value.detach()).cpu())


def _dot(left: torch.Tensor, right: Optional[torch.Tensor]) -> float:
    if right is None or not left.numel() or not right.numel():
        return 0.0
    return float(torch.dot(left.reshape(-1), right.reshape(-1)).detach().cpu())


def _cosine(left: torch.Tensor, right: Optional[torch.Tensor]) -> float:
    if right is None or not left.numel() or not right.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.detach().cpu()) <= EPS:
        return float("nan")
    return float(torch.clamp(torch.dot(left.reshape(-1), right.reshape(-1)) / denom,
                             -1.0, 1.0).detach().cpu())


def _mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


class _EpochGradientAccumulator:
    """Small raw-norm/vector diagnostic accumulator for one training epoch."""

    def __init__(self, domain_num: int, task_num: int):
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.raw_vectors: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.rows: Dict[int, List[Mapping[str, object]]] = defaultdict(list)

    def update(self, raw_by_cell: Mapping[int, torch.Tensor], rows: Sequence[Mapping[str, object]]) -> None:
        for idx, _, _ in iter_cells(self.domain_num, self.task_num):
            self.raw_vectors[idx].append(raw_by_cell[idx].detach().cpu().clone())
        for row in rows:
            self.rows[int(row["cell"])].append(dict(row))

    def output(self, *, seed: int, setting: str, arm_key: str, epoch: int,
               parameter_count: int) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            vectors = self.raw_vectors[idx]
            cell_rows = self.rows[idx]
            raw_norms = np.asarray([_norm(value) for value in vectors], dtype=np.float64)
            weighted_norms = np.asarray([
                float(row.get("weighted_gradient_norm", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            shares = np.asarray([
                float(row.get("normalized_contribution", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            magnitude = np.asarray([
                float(row.get("magnitude_share", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            mismatch = np.asarray([
                float(row.get("mismatch_index", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            target_error = np.asarray([
                float(row.get("target_calibration_error", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            ema_error = np.asarray([
                float(row.get("ema_tracking_error"))
                if _finite_number(row.get("ema_tracking_error")) else np.nan
                for row in cell_rows
            ], dtype=np.float64)
            coeff_var = np.asarray([
                float(row.get("coefficient_variation"))
                if _finite_number(row.get("coefficient_variation")) else np.nan
                for row in cell_rows
            ], dtype=np.float64)
            align = np.asarray([
                float(row.get("actual_update_alignment", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            coeff = np.asarray([
                float(row.get("loss_coefficient", np.nan)) for row in cell_rows
            ], dtype=np.float64)
            if vectors:
                mean_vector = torch.stack(vectors).mean(dim=0)
                norm_mean = _norm(mean_vector)
            else:
                norm_mean = float("nan")
            variance = float(np.var(raw_norms, ddof=1)) if len(raw_norms) >= 2 else float("nan")
            weighted_variance = (float(np.var(weighted_norms[np.isfinite(weighted_norms)], ddof=1))
                                 if int(np.isfinite(weighted_norms).sum()) >= 2 else float("nan"))
            result.append({
                "seed": int(seed), "setting": str(setting), "arm_key": str(arm_key),
                "epoch": int(epoch), "domain": int(domain), "task": int(task),
                "cell": int(idx), "parameter_group": "shared",
                "raw_gradient_norm": _mean_or_nan(raw_norms),
                "weighted_gradient_norm": _mean_or_nan(weighted_norms),
                "gradient_variance": variance,
                "weighted_gradient_variance": weighted_variance,
                "gradient_SNR": (float(np.nanmean(raw_norms) ** 2 / (variance + EPS))
                                  if np.isfinite(variance) and len(raw_norms) else float("nan")),
                "norm_of_mean_gradient": norm_mean,
                "probe_count": len(vectors),
                "contribution_mean": _mean_or_nan(shares),
                "contribution_std": (float(np.nanstd(shares)) if np.isfinite(shares).any()
                                      else float("nan")),
                "magnitude_share": _mean_or_nan(magnitude),
                "mismatch_index": _mean_or_nan(mismatch),
                "target_calibration_error": _mean_or_nan(target_error),
                "ema_tracking_error": (_mean_or_nan(ema_error)
                                        if np.isfinite(ema_error).any() else ""),
                "coefficient_variation": (_mean_or_nan(coeff_var)
                                          if np.isfinite(coeff_var).any() else ""),
                "coefficient_mean": _mean_or_nan(coeff),
                "update_alignment_mean": _mean_or_nan(align),
                "variance_scope": "raw_gradient_norm",
            })
        return result


class E6Runner(E3Runner):
    """One E6 arm/setting/seed run."""

    def __init__(self, args: argparse.Namespace):
        self.arm_key = validate_arm_key(args.arm_key)
        self.arm = dict(E6_ARM_REGISTRY[self.arm_key])
        # E3Runner's loader/build path expects an alpha field.  It is never used
        # as an E6 control variable and is replaced by arm identity in outputs.
        args.alpha = 0.0
        # E6 accepts only the formal E3 oracle root; do not silently fall back
        # to historical/non-formal oracle directories supplied by a caller.
        args.legacy_oracle_root = ""
        super().__init__(args)
        self.alpha = 0.0
        self.alpha_key = self.arm_key
        self.weighting = self.arm_key
        self.target = target_shares(self.arm_key, self.domain_shares)
        self.static_equal = static_coefficients("equal_static", self.domain_shares,
                                                self.task_num, dtype=torch.float64,
                                                device=self.device).detach()
        self.static_sample = static_coefficients("sample_static", self.domain_shares,
                                                 self.task_num, dtype=torch.float64,
                                                 device=self.device).detach()
        if self.arm_key in ("equal_static", "sample_static", "sample_ema_control"):
            self.initial_coefficients = static_coefficients(
                self.arm_key, self.domain_shares, self.task_num,
                dtype=torch.float64, device=self.device).detach()
        else:
            # The first batch replaces this proposal before it is used.
            self.initial_coefficients = self.static_sample.clone().detach()
        self.actual_coefficients = self.initial_coefficients.clone().detach()

    def _init_outputs(self) -> None:
        self.metrics_rows: List[Dict[str, object]] = []
        self.shared_gradient_step_rows: List[Dict[str, object]] = []
        self.shared_gradient_epoch_rows: List[Dict[str, object]] = []
        self.controller_step_rows: List[Dict[str, object]] = []
        self.controller_epoch_rows: List[Dict[str, object]] = []
        self.batch_rows: List[Dict[str, object]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: Optional[int] = None
        self.best_val_auc = float("-inf")
        self.stopped_epoch: Optional[int] = None
        self.observed_epochs: List[int] = []
        self.batch_fingerprints: Dict[int, List[str]] = {}
        self.arch_batch_fingerprints: List[str] = []
        self.rng_restore_evidence: List[Dict[str, object]] = []
        self.nonfinite_inventory: List[Dict[str, object]] = []
        self.ema_state: Optional[EMAMagnitudeState] = (
            EMAMagnitudeState(beta=E6_BETA) if bool(self.arm.get("uses_ema")) else None
        )
        self.previous_coefficients: Optional[torch.Tensor] = None
        self.controller_updates = 0
        self.coefficient_values: List[float] = []
        self.coefficient_variations: List[float] = []
        self.controller_failures: List[str] = []
        self._current_epoch_controller: List[Mapping[str, object]] = []
        self._current_epoch_accumulator: Optional[_EpochGradientAccumulator] = None

    def _record_batch(self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
                      y_batch: torch.Tensor, counts: Sequence[int], is_probe: bool,
                      train_loss: Optional[object] = None) -> str:
        fp = batch_fingerprint(x_batch, y_batch)
        self.batch_rows.append({
            "seed": int(self.seed), "setting": self.setting, "arm_key": self.arm_key,
            "epoch": int(epoch), "step": int(step), "kind": str(kind),
            "batch_fingerprint": fp, "batch_size": int(len(y_batch)),
            "domain_counts": _json_counts(counts),
            "task_counts": _json_counts([int(y_batch[:, task].numel()) for task in range(self.task_num)]),
            "is_probe": int(bool(is_probe)),
            "train_loss": ("" if train_loss is None else float(
                train_loss.detach().cpu() if isinstance(train_loss, torch.Tensor) else train_loss)),
        })
        return fp

    def _controller_coefficients(
        self, raw_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Return applied c, q, EMA state, and formula diagnostics."""

        raw = raw_magnitude.detach().to(dtype=torch.float64)
        ema = torch.empty_like(raw)
        if self.ema_state is not None:
            ema = self.ema_state.update(raw).to(device=self.device).detach()
            q = ema
        else:
            q = raw.to(device=self.device).detach()
        if bool(self.arm.get("feedback")):
            proposed, diag = closed_form_coefficients(q, self.target,
                                                      task_num=self.task_num)
            proposed = proposed.to(device=self.device).detach()
        else:
            proposed = static_coefficients(self.arm_key, self.domain_shares,
                                           self.task_num, dtype=torch.float64,
                                           device=self.device).detach()
            diag = {
                "epsilon": "",
                "q_dtype": str(q.dtype),
                "target": self.target.tolist(),
                "q": q.detach().cpu().tolist(),
                "coefficient_sum_domain": float(proposed[:self.domain_num].sum().detach().cpu()),
                "coefficient_sum_cell": float(proposed.sum().detach().cpu()),
                "coefficients": proposed.detach().cpu().tolist(),
            }
        if not bool(torch.isfinite(proposed).all()) or bool((proposed <= 0).any()):
            raise ValueError("applied E6 coefficients are not finite positive")
        domain_sum = proposed.reshape(self.task_num, self.domain_num)[0].sum()
        if not torch.isclose(domain_sum, torch.tensor(float(self.domain_num),
                                                       dtype=proposed.dtype,
                                                       device=proposed.device), atol=1e-9, rtol=0):
            raise AssertionError("E6 domain coefficient sum is not D")
        return proposed.detach(), q.detach(), ema.detach(), diag

    def _shared_step_rows(
        self, raw_by_cell: Mapping[int, torch.Tensor], unused_by_cell: Mapping[int, int],
        coefficients: torch.Tensor, raw_magnitude: torch.Tensor, q_magnitude: torch.Tensor,
        ema_magnitude: torch.Tensor, counts: Sequence[int], *, epoch: int, step: int,
        fingerprint: str, actual_update: Optional[torch.Tensor], reconstruction_error: float,
        calibration: Mapping[str, object], coeff_var: object, probe: bool,
    ) -> List[Dict[str, object]]:
        actual = contribution_views(raw_by_cell, coefficients, self.domain_num, self.task_num)
        equal = contribution_views(raw_by_cell, self.static_equal, self.domain_num, self.task_num)
        sample = contribution_views(raw_by_cell, self.static_sample, self.domain_num, self.task_num)
        domain_magnitude = {
            domain: sum(float(actual["magnitude"][idx]) for idx, d, _ in
                        iter_cells(self.domain_num, self.task_num) if d == domain)
            for domain in range(self.domain_num)
        }
        domain_signed = {
            domain: sum(float(actual["normalized"][idx]) for idx, d, _ in
                        iter_cells(self.domain_num, self.task_num) if d == domain)
            for domain in range(self.domain_num)
        }
        data_mismatch = float(calibration["data_share_mismatch"])
        target_error = float(calibration["target_calibration_error"])
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            raw_vector = raw_by_cell[idx].detach()
            weighted_vector = raw_vector * coefficients[idx].to(raw_vector.device)
            raw_norm = _norm(raw_vector)
            weighted_norm = _norm(weighted_vector)
            row = {
                "seed": int(self.seed), "setting": self.setting, "arm_key": self.arm_key,
                "epoch": int(epoch), "step": int(step), "domain": int(domain),
                "task": int(task), "cell": int(idx), "parameter_group": "shared",
                "batch_fingerprint": fingerprint, "cell_sample_count": int(counts[idx]),
                "p_d": float(self.domain_shares[domain]), "tau_d": float(self.target[domain]),
                "raw_domain_magnitude": float(raw_magnitude[domain].detach().cpu()),
                "ema_domain_magnitude": float(ema_magnitude[domain].detach().cpu())
                if self.ema_state is not None else "",
                "q_domain_magnitude": float(q_magnitude[domain].detach().cpu()),
                "loss_coefficient": float(coefficients[idx].detach().cpu()),
                "raw_gradient_norm": raw_norm,
                "raw_gradient_rms": raw_norm / np.sqrt(max(1, self.partition.counts["shared"])),
                "weighted_gradient_norm": weighted_norm,
                "weighted_gradient_rms": weighted_norm / np.sqrt(max(1, self.partition.counts["shared"])),
                "effective_projection": (float(actual["projection"][idx]) if idx in actual["projection"] else float("nan")),
                "normalized_contribution": float(actual["normalized"][idx]),
                "signed_contribution": float(actual["normalized"][idx]),
                "domain_signed_contribution": domain_signed[domain],
                "magnitude_share": domain_magnitude[domain],
                "overrepresentation_ratio": domain_magnitude[domain] / (float(self.domain_shares[domain]) + EPS),
                "mismatch_index": data_mismatch,
                "target_calibration_error": target_error,
                "equal_cf_contribution": float(equal["normalized"][idx]),
                "sample_cf_contribution": float(sample["normalized"][idx]),
                "unused_param_count": int(unused_by_cell.get(idx, 0)),
                "actual_update_norm": _norm(actual_update),
                "actual_update_alignment": _cosine(weighted_vector, actual_update),
                "actual_update_projection": (_dot(weighted_vector, actual_update) /
                                               (_norm(actual_update) + EPS)
                                               if actual_update is not None else float("nan")),
                "gradient_reconstruction_error": float(reconstruction_error),
                # The first batch has no k>0 transition and must remain blank;
                # later calls carry the real domain-only float transition.
                "coefficient_variation": coeff_var,
                "ema_tracking_error": (abs(float(q_magnitude[domain].detach().cpu()) -
                                            float(raw_magnitude[domain].detach().cpu())) /
                                        (float(raw_magnitude[domain].detach().cpu()) + EPS)
                                        if self.ema_state is not None else ""),
                "probe": int(bool(probe)),
            }
            rows.append(row)
        return rows

    def _train_epoch(self, epoch: int) -> Tuple[Dict[int, Dict[str, object]], List[str]]:
        self.model.train()
        train_truth: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_prediction: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_loss_sum = {idx: 0.0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_counts = {idx: 0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        accumulator = _EpochGradientAccumulator(self.domain_num, self.task_num)
        self._current_epoch_accumulator = accumulator
        self._current_epoch_controller = []
        fingerprints: List[str] = []
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            domain_cpu = x_batch[:, -1].detach().cpu().numpy().astype(int)
            domain_counts = [int((domain_cpu == domain).sum()) for domain in range(self.domain_num)]
            if any(value <= 0 for value in domain_counts):
                raise RuntimeError("formal train batch is missing a domain")
            cell_counts = [domain_counts[domain] for _, domain, _ in
                           iter_cells(self.domain_num, self.task_num)]
            is_probe = bool(int(self.args.instrument)) and (
                step % int(self.args.probe_interval) == 0)
            fingerprint = self._record_batch(epoch, step, "train", x_batch, y_batch,
                                             cell_counts, is_probe)
            fingerprints.append(fingerprint)
            x_device = x_batch.transpose(0, 1).to(self.device)
            y_device = y_batch.to(self.device)
            domain_ids = x_device[-1, :].detach()
            y_pred = self.model(x_device).to(self.device)
            terms, counts = masked_bce_terms(
                y_pred, y_device, domain_ids, self.domain_num, self.task_num, self.criterion)
            for idx, domain, task in iter_cells(self.domain_num, self.task_num):
                if counts[idx] > 0:
                    train_loss_sum[idx] += float(terms[idx].detach().cpu()) * counts[idx]
                    train_counts[idx] += counts[idx]
                    mask = domain_ids == domain
                    train_truth[idx].extend(y_device[mask, task].detach().cpu().tolist())
                    train_prediction[idx].extend(y_pred[mask, task].detach().cpu().tolist())

            # The observer uses the same forward graph and only detached shared
            # raw gradients.  autograd.grad never touches Parameter.grad.
            raw_by_cell, unused_by_cell = collect_shared_cell_gradients(
                terms, self.model, self.partition, retain_graph=True)
            raw_magnitude = raw_domain_magnitudes(raw_by_cell, self.domain_num, self.task_num)
            coefficients, q_magnitude, ema_magnitude, formula = self._controller_coefficients(raw_magnitude)
            has_transition = self.previous_coefficients is not None
            coeff_var_value = coefficient_variation(
                self.previous_coefficients, coefficients, self.domain_num)
            # V_c is defined only over real transitions k>0.  The first
            # observation is intentionally blank and excluded from all means.
            coeff_var = coeff_var_value if has_transition else ""
            if has_transition and not np.isfinite(coeff_var_value):
                raise RuntimeError("non-finite coefficient transition")
            self.previous_coefficients = coefficients.clone().detach()
            self.actual_coefficients = coefficients.clone().detach()
            self.controller_updates += 1
            self.coefficient_values.extend(float(value) for value in coefficients.detach().cpu().tolist())
            if has_transition:
                self.coefficient_variations.append(float(coeff_var_value))
            calibration = calibration_metrics(raw_magnitude, coefficients, self.domain_shares,
                                              self.target, task_num=self.task_num)
            loss = weighted_loss_from_terms(terms, coefficients)
            self.batch_rows[-1]["train_loss"] = float(loss.detach().cpu())

            before = self._vectors_shared()
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            named = dict(self.model.named_parameters())
            shared_names = list(self.partition.names("shared"))
            actual_grad, _ = flatten_parameter_tensors(
                named, shared_names,
                [named[name].grad for name in shared_names], device=self.device)
            combined = contribution_views(raw_by_cell, coefficients,
                                          self.domain_num, self.task_num)["combined"]
            reconstruction = relative_l2_error(actual_grad, combined)
            self.optimizer.step()
            after = self._vectors_shared()
            update = before - after
            shared_rows = self._shared_step_rows(
                raw_by_cell, unused_by_cell, coefficients, raw_magnitude, q_magnitude,
                ema_magnitude, counts, epoch=epoch, step=step, fingerprint=fingerprint,
                actual_update=update, reconstruction_error=reconstruction,
                calibration=calibration, coeff_var=coeff_var, probe=is_probe)
            self.shared_gradient_step_rows.extend(shared_rows)
            accumulator.update(raw_by_cell, shared_rows)
            ema_errors = [float(row["ema_tracking_error"]) for row in shared_rows
                          if _finite_number(row.get("ema_tracking_error"))]
            controller_row = {
                "seed": int(self.seed), "setting": self.setting, "arm_key": self.arm_key,
                "epoch": int(epoch), "step": int(step), "batch_fingerprint": fingerprint,
                "batch_size": int(len(y_batch)), "domain_counts": _json_vector(domain_counts),
                "p_d": _json_vector(self.domain_shares), "tau_d": _json_vector(self.target),
                "raw_r_d": _json_vector(raw_magnitude),
                "ema_hat_r_d": (_json_vector(ema_magnitude) if self.ema_state is not None else ""),
                "q_d": _json_vector(q_magnitude), "beta": (E6_BETA if self.ema_state is not None else ""),
                "controller_statistic": self.arm["dynamic_statistic"],
                "controller_feedback": int(bool(self.arm["feedback"])),
                "proposed_c_d": _json_vector(formula["coefficients"]),
                "applied_c_d": _json_vector(coefficients),
                "coefficient_sum_domain": float(coefficients[:self.domain_num].sum().detach().cpu()),
                "coefficient_sum_cell": float(coefficients.sum().detach().cpu()),
                "actual_magnitude_share": _json_vector(calibration["actual_magnitude_share"]),
                "data_share_mismatch": float(calibration["data_share_mismatch"]),
                "target_calibration_error": float(calibration["target_calibration_error"]),
                "ema_tracking_error": (_mean_or_nan(ema_errors)
                                        if self.ema_state is not None else ""),
                "coefficient_variation": coeff_var,
                "fallback_flag": 0,
                "error_flag": 0,
                "main_loss": float(loss.detach().cpu()),
                "gradient_reconstruction_error": float(reconstruction),
                "actual_update_alignment": _cosine(combined, update),
                "actual_update_norm": _norm(update),
            }
            self.controller_step_rows.append(controller_row)
            self._current_epoch_controller.append(controller_row)
        self.shared_gradient_epoch_rows.extend(accumulator.output(
            seed=self.seed, setting=self.setting, arm_key=self.arm_key, epoch=epoch,
            parameter_count=self.partition.counts["shared"]))
        if self._current_epoch_controller:
            rows = self._current_epoch_controller
            def nums(field: str) -> List[float]:
                return [float(row[field]) for row in rows if _finite_number(row.get(field))]
            all_coeff = [value for row in rows for value in json.loads(str(row["applied_c_d"]))]
            self.controller_epoch_rows.append({
                "seed": int(self.seed), "setting": self.setting, "arm_key": self.arm_key,
                "epoch": int(epoch), "controller_statistic": self.arm["dynamic_statistic"],
                "target": self.arm["target"], "update_count": len(rows),
                "raw_r_mean": _mean_or_nan([float(np.mean(json.loads(str(row["raw_r_d"])))) for row in rows]),
                "q_mean": _mean_or_nan([float(np.mean(json.loads(str(row["q_d"])))) for row in rows]),
                "target_calibration_error_mean": _mean_or_nan(nums("target_calibration_error")),
                "data_share_mismatch_mean": _mean_or_nan(nums("data_share_mismatch")),
                "ema_tracking_error_mean": (_mean_or_nan(nums("ema_tracking_error"))
                                             if self.ema_state is not None else ""),
                "coefficient_variation_mean": (_mean_or_nan(nums("coefficient_variation"))
                                                if nums("coefficient_variation") else ""),
                "coefficient_min": float(np.min(all_coeff)), "coefficient_max": float(np.max(all_coeff)),
                "coefficient_mean": float(np.mean(all_coeff)),
                "coefficient_sum_domain": _mean_or_nan(nums("coefficient_sum_domain")),
                "coefficient_sum_cell": _mean_or_nan(nums("coefficient_sum_cell")),
                "fallback_count": int(sum(int(row["fallback_flag"]) for row in rows)),
                "error_count": int(sum(int(row["error_flag"]) for row in rows)),
            })
        train_stats = {
            idx: {
                "sample_count": train_counts[idx],
                "positive_rate": (float(np.mean(train_truth[idx])) if train_truth[idx] else float("nan")),
                "auc": self._safe_auc(train_truth[idx], train_prediction[idx]),
                "logloss": (train_loss_sum[idx] / train_counts[idx]
                            if train_counts[idx] else float("nan")),
            }
            for idx, _, _ in iter_cells(self.domain_num, self.task_num)
        }
        return train_stats, fingerprints

    @staticmethod
    def _safe_auc(truth: Sequence[float], prediction: Sequence[float]) -> float:
        from stage_E1_instrumentation import safe_auc
        return safe_auc(truth, prediction)

    def _append_metric_rows(self, epoch: int, train_stats: Mapping[int, Mapping[str, object]],
                            val_stats: Mapping[int, Mapping[str, object]],
                            test_stats: Mapping[int, Mapping[str, object]], selected: bool,
                            censored: bool = False) -> None:
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            train = train_stats.get(idx, {})
            val = val_stats.get(idx, {})
            test = test_stats.get(idx, {})
            oracle = self.oracle_metrics.get(idx, {})
            row: Dict[str, object] = {
                "seed": int(self.seed), "epoch": int(epoch), "arm_key": self.arm_key,
                "controller": self.arm["coefficient_behavior"], "target": self.arm["target"],
                "domain": int(domain), "task": int(task), "cell": int(idx),
                "setting": self.setting, "physical_domain": int(domain),
                "size_role": self.role_by_domain[domain], "domain_size": self.train_counts[domain],
                "domain_share": self.domain_shares[domain],
                "train_sample_count": train.get("sample_count", self.split_counts["train"].get(idx, "")),
                "validation_sample_count": val.get("sample_count", self.split_counts["validation"].get(idx, "")),
                "test_sample_count": test.get("sample_count", self.split_counts["test"].get(idx, "")),
                "train_positive_rate": train.get("positive_rate", self.positive_rates["train"].get(idx, "")),
                "validation_positive_rate": val.get("positive_rate", self.positive_rates["validation"].get(idx, "")),
                "test_positive_rate": test.get("positive_rate", self.positive_rates["test"].get(idx, "")),
                "train_auc": train.get("auc", ""), "train_logloss": train.get("logloss", ""),
                "validation_auc": val.get("auc", ""), "validation_logloss": val.get("logloss", ""),
                "test_auc": test.get("auc", ""), "test_logloss": test.get("logloss", ""),
                "selected_checkpoint": int(bool(selected)), "stopped_censored": int(bool(censored)),
                "oracle_auc": oracle.get("auc", ""), "oracle_logloss": oracle.get("logloss", ""),
                "oracle_source": oracle.get("source", "missing"),
                "oracle_coverage": int(bool(oracle.get("coverage", False))),
                "transfer_auc": "", "transfer_logloss": "",
            }
            if _finite_number(row["oracle_auc"]) and _finite_number(row["test_auc"]):
                row["transfer_auc"] = float(row["test_auc"]) - float(row["oracle_auc"])
            if _finite_number(row["oracle_logloss"]) and _finite_number(row["test_logloss"]):
                row["transfer_logloss"] = float(row["test_logloss"]) - float(row["oracle_logloss"])
            self.metrics_rows.append(row)

    def _selected_metric_rows(self) -> Dict[int, Dict[str, object]]:
        return {int(row["cell"]): row for row in self.metrics_rows
                if int(row.get("selected_checkpoint", 0)) == 1}

    def _selected_checkpoint_complete(self) -> Tuple[bool, List[str]]:
        expected = {idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        selected = self._selected_metric_rows()
        required = ("validation_auc", "validation_logloss", "test_auc", "test_logloss")
        good = set(selected) == expected and all(_finite_number(row.get(field))
                                                  for row in selected.values() for field in required)
        return bool(good), [str(idx) for idx in sorted(selected)]

    def _gradient_epoch_for_cell(self, cell: int, epoch: object) -> Mapping[str, object]:
        rows = [row for row in self.shared_gradient_epoch_rows
                if int(row.get("cell", -1)) == int(cell) and int(row.get("epoch", -1)) == int(epoch)]
        return rows[-1] if rows else {}

    def _selected_step_for_cell(self, cell: int, epoch: object) -> Mapping[str, object]:
        rows = [row for row in self.shared_gradient_step_rows
                if int(row.get("cell", -1)) == int(cell) and int(row.get("epoch", -1)) == int(epoch)]
        return rows[-1] if rows else {}

    def _write_metrics_summary(self, fit_info: Mapping[str, object]) -> List[Dict[str, object]]:
        selected = self._selected_metric_rows()
        expected = {idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        complete = set(selected) == expected and all(
            _finite_number(row.get("test_auc")) and _finite_number(row.get("test_logloss"))
            for row in selected.values())
        test_auc = {idx: float(row["test_auc"]) for idx, row in selected.items()} if complete else {}
        test_ll = {idx: float(row["test_logloss"]) for idx, row in selected.items()} if complete else {}
        domain_auc = {d: float(np.mean([test_auc[idx] for idx, dd, _ in
                                        iter_cells(self.domain_num, self.task_num) if dd == d]))
                      for d in range(self.domain_num) if test_auc and any(dd == d for _, dd, _ in
                                                                          iter_cells(self.domain_num, self.task_num))}
        overall_auc = float(np.mean(list(test_auc.values()))) if test_auc else float("nan")
        overall_ll = float(np.mean(list(test_ll.values()))) if test_ll else float("nan")
        worst_domain = min(domain_auc.values()) if domain_auc else float("nan")
        worst_cell = min(test_auc.values()) if test_auc else float("nan")
        target_epoch = fit_info.get("best_epoch", "")
        own_last_epoch = max(self.observed_epochs) if self.observed_epochs else ""
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            metric = selected.get(idx, {})
            oracle = self.oracle_metrics.get(idx, {})
            grad = self._gradient_epoch_for_cell(idx, target_epoch) if target_epoch not in ("", None) else {}
            step = self._selected_step_for_cell(idx, target_epoch) if target_epoch not in ("", None) else {}
            last_common = next((item for item in reversed(self.metrics_rows)
                                if int(item.get("cell", -1)) == int(idx)
                                and int(item.get("epoch", -1)) == int(own_last_epoch)
                                and not int(item.get("stopped_censored", 0))), {}) \
                if own_last_epoch not in ("", None) else {}
            transfer = metric.get("transfer_auc", "")
            rows.append({
                "seed": int(self.seed), "setting": self.setting, "arm_key": self.arm_key,
                "controller": self.arm["coefficient_behavior"], "target": self.arm["target"],
                "domain": int(domain), "physical_domain": int(domain), "task": int(task),
                "cell": int(idx), "size_role": self.role_by_domain[domain],
                "domain_size": self.train_counts[domain], "domain_share": self.domain_shares[domain],
                "selected_epoch": fit_info.get("best_epoch", ""),
                "stopped_epoch": fit_info.get("stopped_epoch", ""),
                "best_val_auc": metric.get("validation_auc", ""),
                "best_val_logloss": metric.get("validation_logloss", ""),
                "best_test_auc": metric.get("test_auc", ""),
                "best_test_logloss": metric.get("test_logloss", ""),
                "oracle_auc": oracle.get("auc", ""), "oracle_logloss": oracle.get("logloss", ""),
                "oracle_source": oracle.get("source", "missing"),
                "oracle_coverage": int(bool(oracle.get("coverage", False))),
                "transfer_auc": transfer, "transfer_logloss": metric.get("transfer_logloss", ""),
                "overall_auc": overall_auc, "overall_logloss": overall_ll,
                "worst_domain_auc": worst_domain, "worst_cell_auc": worst_cell,
                "negative_transfer": int(_finite_number(transfer) and float(transfer) < 0),
                "shared_raw_gradient_norm": grad.get("raw_gradient_norm", ""),
                "shared_weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                "shared_magnitude_share": step.get("magnitude_share", grad.get("magnitude_share", "")),
                "shared_signed_contribution": grad.get("contribution_mean", ""),
                "shared_mismatch_index": grad.get("mismatch_index", ""),
                "target_calibration_error": grad.get("target_calibration_error", ""),
                "coefficient_variation": grad.get("coefficient_variation", ""),
                "ema_tracking_error": grad.get("ema_tracking_error", ""),
                "gradient_variance": grad.get("gradient_variance", ""),
                "gradient_SNR": grad.get("gradient_SNR", ""),
                "shared_update_alignment": grad.get("update_alignment_mean", ""),
                # The report collector replaces this local endpoint flag with
                # six-arm common support after the cross-arm block is known.
                "common_support": int(target_epoch in self.observed_epochs),
                "last_common_epoch": own_last_epoch,
                "last_common_test_auc": last_common.get("test_auc", ""),
                "last_common_test_logloss": last_common.get("test_logloss", ""),
                "last_common_transfer_auc": last_common.get("transfer_auc", ""),
                "last_common_transfer_logloss": last_common.get("transfer_logloss", ""),
                "censored": int(self.stopped_epoch is not None),
            })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), rows,
                       E6_METRICS_SUMMARY_FIELDS)
        return rows

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows,
                       E6_METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "controller_step.csv"), self.controller_step_rows,
                       E6_CONTROLLER_STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "controller_epoch.csv"), self.controller_epoch_rows,
                       E6_CONTROLLER_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_step.csv"),
                       self.shared_gradient_step_rows, E6_SHARED_STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_epoch.csv"),
                       self.shared_gradient_epoch_rows, E6_SHARED_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       E6_BATCH_FIELDS)
        summary_rows = self._write_metrics_summary(fit_info)
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "metrics_summary.csv": summary_rows,
            "controller_step.csv": self.controller_step_rows,
            "controller_epoch.csv": self.controller_epoch_rows,
            "shared_gradient_step.csv": self.shared_gradient_step_rows,
            "shared_gradient_epoch.csv": self.shared_gradient_epoch_rows,
            "batch_trace.csv": self.batch_rows,
        })
        inventory = {
            "stage": "E6", "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(1 for item in self.nonfinite_inventory
                                      if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
            "scope_note": "raw non-finite values inventoried before CSV conversion",
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory)
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_complete, selected_cells = self._selected_checkpoint_complete()
        config = vars(self.args).copy()
        config.update({
            "stage": "E6", "seed": int(self.seed), "setting": self.setting,
            "arm_key": self.arm_key, "arm": self.arm, "controller_version": E6_FORMAL_PROTOCOL["controller_version"],
            "controller_formula": E6_FORMAL_PROTOCOL["controller_formula"],
            "controller_inputs_whitelist": list(E6_FORMAL_PROTOCOL["controller_inputs"]),
            "controller_statistic": self.arm["dynamic_statistic"],
            "controller_feedback": bool(self.arm["feedback"]), "target": self.arm["target"],
            "target_vector": self.target.tolist(), "beta": E6_BETA if self.ema_state is not None else None,
            "beta_derivation": "2**(-1/ceil(187500/40960))" if self.ema_state is not None else None,
            "loss_weighting": self.arm["coefficient_behavior"], "fix_optim_grouping": 1,
            "earlystop_patience": int(self.args.earlystop_patience), "backbone": "mdmoe",
            "scheduler": None, "task": "all", "domain": "all",
            "role_sizes_parsed": self.sizes, "domain_shares": self.domain_shares.tolist(),
            "train_domain_counts": self.train_counts,
            "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.static_equal.detach().cpu().tolist(),
            "sample_coefficients": self.static_sample.detach().cpu().tolist(),
            "cell_order": CELL_ORDER_NOTE, "instrumentation": bool(int(self.args.instrument)),
            "prohibited_mechanisms": ["curriculum", "optimizer_momentum_modification",
                                       "parameter_ema", "adaptive_sampling", "new_router",
                                       "validation_controller_input", "test_controller_input",
                                       "oracle_controller_input", "gradient_variance_controller_input",
                                       "snr_controller_input", "training_stage_controller_input"],
            "sampling": E6_FORMAL_PROTOCOL["sampling"],
            "dataset_identity": self.dataset_identity, "device_identity": self.device_identity,
            "split_row_fingerprints": self.split_row_fingerprints,
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": selected_cells,
            "controller_update_count": int(self.controller_updates),
            "ema_update_count": int(self.ema_state.update_count if self.ema_state is not None else 0),
            "coefficient_min": (float(np.min(self.coefficient_values)) if self.coefficient_values else float("nan")),
            "coefficient_max": (float(np.max(self.coefficient_values)) if self.coefficient_values else float("nan")),
            "coefficient_variation_mean": (_mean_or_nan(self.coefficient_variations)
                                            if self.coefficient_variations else ""),
            "controller_registry_hash": registry_hash(),
        })
        config["formal_protocol"] = dict(E6_FORMAL_PROTOCOL)
        config["formal_valid"] = bool(
            config.get("epoch") == E6_FORMAL_PROTOCOL["epoch"]
            and config.get("batch_size") == E6_FORMAL_PROTOCOL["batch_size"]
            and config.get("learning_rate") == E6_FORMAL_PROTOCOL["learning_rate"]
            and config.get("learning_rate_darts") == E6_FORMAL_PROTOCOL["learning_rate_darts"]
            and config.get("weight_decay") == E6_FORMAL_PROTOCOL["weight_decay"]
            and config.get("expert_num") == E6_FORMAL_PROTOCOL["expert_num"]
            and config.get("probe_interval") == E6_FORMAL_PROTOCOL["probe_interval"]
            and config.get("instrument") == E6_FORMAL_PROTOCOL["instrument"]
            and config.get("role_sizes") == E6_FORMAL_PROTOCOL["role_sizes"]
            and config.get("fix_optim_grouping") == E6_FORMAL_PROTOCOL["fix_optim_grouping"]
            and config.get("earlystop_patience") == E6_FORMAL_PROTOCOL["earlystop_patience"]
            and config.get("setting") == E6_FORMAL_SETTING
            and self.arm_key in E6_ARM_KEYS
            and self.train_counts == {d: int(E6_ROLE_SIZES[self.role_by_domain[d]])
                                      for d in range(self.domain_num)}
            and selected_complete
            and self.controller_updates > 0
            and os.path.exists(os.path.join(self.run_dir, "model.pth"))
            and self.dataset_identity == normalise_dataset_path(config.get("dataset_path", ""))
        )
        config["config_hash"] = _hash_json(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E6", "status": "OK", "seed": int(self.seed), "setting": self.setting,
            "arm_key": self.arm_key, "arm": self.arm,
            "formal_run_key": f"{self.arm_key}|{self.setting}|{self.seed}",
            "dataset": "MovieLens-1M", "dataset_path": os.path.abspath(self.args.dataset_path),
            "dataset_identity": self.dataset_identity, "device_identity": self.device_identity,
            "domain_num": self.domain_num, "task_num": self.task_num,
            "role_by_domain": self.role_by_domain, "role_sizes": self.sizes,
            "train_domain_counts": self.train_counts, "domain_shares": self.domain_shares.tolist(),
            "split_counts": self.split_counts, "positive_rates": self.positive_rates,
            "oracle_audit": self.oracle_audit, "oracle_metrics": self.oracle_metrics,
            "partition": self.partition.as_dict(),
            "model_identity": {"class": "MDMTRec", "backbone": "mdmoe", "fcn_dims": [512, 256, 256, 64],
                               "expert_num": int(self.args.expert_num), "domain_num": self.domain_num,
                               "task_num": self.task_num},
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.static_equal.detach().cpu().tolist(),
            "sample_coefficients": self.static_sample.detach().cpu().tolist(),
            "target": self.arm["target"], "target_vector": self.target.tolist(),
            "controller_version": E6_FORMAL_PROTOCOL["controller_version"],
            "controller_formula": E6_FORMAL_PROTOCOL["controller_formula"],
            "controller_inputs_whitelist": list(E6_FORMAL_PROTOCOL["controller_inputs"]),
            "controller_statistic": self.arm["dynamic_statistic"],
            "controller_feedback": bool(self.arm["feedback"]), "ema_beta": E6_BETA if self.ema_state is not None else None,
            "ema_half_life_epochs": 1 if self.ema_state is not None else None,
            "observed_epochs": list(self.observed_epochs), "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": (self.stopped_epoch + 1 if self.stopped_epoch is not None else None),
            "formal_protocol": dict(E6_FORMAL_PROTOCOL), "formal_valid": bool(config["formal_valid"]),
            "selected_checkpoint_complete": bool(selected_complete), "selected_checkpoint_cells": selected_cells,
            "split_row_fingerprints": self.split_row_fingerprints,
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.arch_batch_fingerprints,
            "controller_update_count": int(self.controller_updates),
            "controller_step_count": len(self.controller_step_rows),
            "controller_epoch_count": len(self.controller_epoch_rows),
            "shared_gradient_step_count": len(self.shared_gradient_step_rows),
            "shared_gradient_epoch_count": len(self.shared_gradient_epoch_rows),
            "coefficient_min": config["coefficient_min"], "coefficient_max": config["coefficient_max"],
            "coefficient_variation_mean": config["coefficient_variation_mean"],
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory["unexplained_count"],
            "block_status": "PENDING", "block_valid": None,
            "wall_seconds": float(wall_seconds),
            "scope_note": "raw outputs only; no scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E6 one-run adaptive calibration")
    parser.add_argument("--arm_key", choices=list(E6_ARM_KEYS), default="sample_static")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--setting", choices=[E6_FORMAL_SETTING], default=E6_FORMAL_SETTING)
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "stage_E3_raw_results", "oracle_reference"))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--instrument", type=int, choices=[0, 1], default=1)
    parser.add_argument("--probe_interval", type=int, default=1)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--stdout_log", default=None)
    return parser


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args: argparse.Namespace) -> int:
    if int(args.probe_interval) <= 0:
        raise ValueError("--probe_interval must be positive")
    started = time.time()
    runner = E6Runner(args)
    try:
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        with preserve_rng_state():
            final_test = _metric_records(runner.model, runner.test_eval_loader,
                                         runner.device, runner.domain_num, runner.task_num)
        print("[stage_E6] arm:", runner.arm_key, flush=True)
        print("[stage_E6] FINAL per-(d,t) test auc:",
              [final_test[idx]["auc"] for idx, _, _ in iter_cells(runner.domain_num, runner.task_num)],
              flush=True)
        print("[stage_E6] raw outputs:", runner.run_dir, flush=True)
        return 0
    finally:
        runner.close()


def main() -> None:
    args = build_argparser().parse_args()
    if args.stdout_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.stdout_log)), exist_ok=True)
        with open(args.stdout_log, "w", encoding="utf-8", errors="replace") as handle:
            old = sys.stdout
            sys.stdout = _Tee(old, handle)
            try:
                raise SystemExit(run(args))
            finally:
                sys.stdout = old
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
