"""Single-run raw-output entry point for Stage E5.

Stage E5 reuses the E3 data loading, model, optimiser, architecture update and
shared-gradient observer.  The only run-time change is the closed five-arm
alpha trajectory from :mod:`stage_E5_schedule`; the architecture objective and
the natural sampler remain the E3 paths.  This module emits raw observations
and integrity metadata only.  Cross-arm comparisons and scientific decisions
belong to the orchestrator and main analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_E3_run as e3_run  # noqa: E402
from stage_E3_run import E3Runner  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    E3_GRADIENT_EPOCH_FIELDS,
    E3_GRADIENT_STEP_FIELDS,
    ROLE_MAPS,
    alpha_key as e3_alpha_key,
    iter_cells,
    normalise_dataset_path,
    parameter_state_hash,
    strict_validation_macro_auc,
    write_json,
    write_rows_csv,
)
from stage_E1_instrumentation import collect_nonfinite_inventory  # noqa: E402
from stage_E5_schedule import (  # noqa: E402
    E5_ARM_KEYS,
    E5_EPOCHS,
    E5_EPOCH_INDICES,
    E5_FORMAL_PROTOCOL,
    E5_FORMAL_ROLE_SIZES,
    E5_FORMAL_SETTING,
    alpha_trajectory,
    arm_alpha,
    exposure_table,
    phase_for_epoch,
    schedule_formula,
    schedule_registry_hash,
    scheduled_coefficients,
    scheduled_coefficients_numpy,
    validate_coefficients,
)


METRICS_FIELDS = [
    "stage", "arm_key", "seed", "epoch", "domain", "task", "cell", "alpha",
    "alpha_key", "weighting", "setting", "physical_domain", "size_role", "domain_size",
    "domain_share", "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate", "train_auc",
    "train_logloss", "validation_auc", "validation_logloss", "test_auc", "test_logloss",
    "selected_checkpoint", "stopped_censored", "oracle_auc", "oracle_logloss",
    "oracle_source", "oracle_coverage", "transfer_auc", "transfer_logloss",
]

SUMMARY_FIELDS = [
    "stage", "arm_key", "seed", "view", "epoch", "domain", "physical_domain", "task",
    "cell", "size_role", "domain_size", "domain_share", "train_sample_count",
    "validation_sample_count", "test_sample_count", "train_auc", "train_logloss",
    "validation_auc", "validation_logloss", "test_auc", "test_logloss", "oracle_auc",
    "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc", "transfer_logloss",
    "negative_transfer", "overall_auc", "overall_logloss", "domain_macro_auc", "task_macro_auc",
    "worst_domain_auc", "worst_cell_auc", "peak_epoch", "fixed_final_epoch", "selected_epoch",
    "selected_checkpoint", "common_support", "censored", "alpha_at_epoch", "alpha_exposure_total",
    "coefficient_integral_cell", "step_weighted_coefficient_integral_cell",
    "shared_raw_gradient_norm", "shared_weighted_gradient_norm", "shared_gradient_variance",
    "shared_gradient_snr", "shared_magnitude_share", "shared_signed_contribution",
    "shared_mismatch_index", "shared_update_alignment",
]

SCHEDULE_FIELDS = [
    "stage", "arm_key", "seed", "epoch", "phase", "alpha", "alpha_formula",
    "coefficient_sum", "cumulative_alpha_exposure", "main_step_count", "optimizer_step_count",
    "processed_examples", "planned_alpha", "observed_alpha", "schedule_formula_version",
]
for _cell in range(6):
    SCHEDULE_FIELDS.extend([
        "coefficient_cell_{}".format(_cell),
        "cumulative_coefficient_cell_{}".format(_cell),
        "step_weighted_coefficient_cell_{}".format(_cell),
    ])
for _domain in range(3):
    SCHEDULE_FIELDS.extend([
        "coefficient_domain_{}".format(_domain),
        "cumulative_coefficient_domain_{}".format(_domain),
        "step_weighted_coefficient_domain_{}".format(_domain),
    ])

BATCH_FIELDS = [
    "seed", "setting", "arm_key", "alpha", "alpha_key", "epoch", "step", "kind",
    "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "is_probe", "train_loss",
]

STEP_FIELDS = ["stage", "arm_key"] + [
    field for field in E3_GRADIENT_STEP_FIELDS if field not in ("stage", "arm_key")
]

EPOCH_FIELDS = list(E3_GRADIENT_EPOCH_FIELDS)
for _field in ("stage", "arm_key", "loss_coefficient", "step_weighted_loss_coefficient"):
    if _field not in EPOCH_FIELDS:
        EPOCH_FIELDS.append(_field)


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_hash(payload: object) -> str:
    return __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _float_or_blank(value: object) -> object:
    return float(value) if _finite(value) else ""


class E5Runner(E3Runner):
    """E3 runner with a preregistered per-epoch contribution exponent."""

    def __init__(self, args: argparse.Namespace):
        self.arm_key = str(args.arm_key)
        if self.arm_key not in E5_ARM_KEYS:
            raise ValueError("unknown E5 arm {!r}".format(self.arm_key))
        if int(args.epoch) <= 0 or int(args.epoch) > E5_EPOCHS:
            raise ValueError("E5 epoch must be in 1..{}".format(E5_EPOCHS))
        if str(args.setting) != E5_FORMAL_SETTING:
            raise ValueError("E5 is registered only for setting {}".format(E5_FORMAL_SETTING))
        # E3Runner initialises the common data/model/optimiser path and requires
        # a registered static alpha.  It is used only during construction; the
        # scheduled coefficient vector replaces it before every train epoch.
        args.alpha = 0.75
        super().__init__(args)
        self.arm_key = str(args.arm_key)
        self.alpha = arm_alpha(self.arm_key, 0)
        self.alpha_key = self.arm_key
        self.schedule_rows: List[Dict[str, object]] = []
        self._last_epoch_parameter_state_hash = ""
        self._planned_trajectory = alpha_trajectory(self.arm_key, int(args.epoch))

    def _record_batch(self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
                      y_batch: torch.Tensor, counts: Sequence[int], is_probe: bool,
                      train_loss: Optional[object] = None) -> str:
        fingerprint = super()._record_batch(
            epoch, step, kind, x_batch, y_batch, counts, is_probe, train_loss)
        self.batch_rows[-1]["arm_key"] = self.arm_key
        return fingerprint

    def _schedule_row(self, epoch: int, coefficients: torch.Tensor) -> Dict[str, object]:
        coeff = coefficients.detach().cpu().numpy().reshape(-1).astype(np.float64)
        step_count = int(len(self.train_loader))
        cumulative_alpha = float(sum(float(row["alpha"]) for row in self.schedule_rows)
                                 + float(self.alpha))
        prior_coeff = np.zeros_like(coeff)
        prior_step_coeff = np.zeros_like(coeff)
        prior_steps = 0
        for row in self.schedule_rows:
            for cell in range(min(6, len(coeff))):
                prior_coeff[cell] += float(row.get("coefficient_cell_{}".format(cell), 0.0))
                prior_step_coeff[cell] += float(row.get(
                    "step_weighted_coefficient_cell_{}".format(cell), 0.0))
            prior_steps += int(row.get("optimizer_step_count", 0))
        row: Dict[str, object] = {
            "stage": "E5", "arm_key": self.arm_key, "seed": self.seed,
            "epoch": int(epoch), "phase": phase_for_epoch(epoch), "alpha": float(self.alpha),
            "alpha_formula": schedule_formula(self.arm_key), "coefficient_sum": float(coeff.sum()),
            "cumulative_alpha_exposure": cumulative_alpha, "main_step_count": step_count,
            "optimizer_step_count": step_count, "processed_examples": int(sum(self.train_counts.values())),
            "planned_alpha": float(self._planned_trajectory[epoch]), "observed_alpha": float(self.alpha),
            "schedule_formula_version": "E5-v1",
        }
        for cell, value in enumerate(coeff):
            row["coefficient_cell_{}".format(cell)] = float(value)
            row["cumulative_coefficient_cell_{}".format(cell)] = float(prior_coeff[cell] + value)
            row["step_weighted_coefficient_cell_{}".format(cell)] = float(
                prior_step_coeff[cell] + value * step_count)
        for domain in range(3):
            # Both tasks intentionally copy c_d; the domain field reports the
            # per-task coefficient, while the six cell fields preserve the
            # task-major values needed for exact reconstruction.
            domain_value = float(coeff[domain])
            prior_domain = float(prior_coeff[domain])
            prior_step_domain = float(prior_step_coeff[domain])
            row["coefficient_domain_{}".format(domain)] = domain_value
            row["cumulative_coefficient_domain_{}".format(domain)] = prior_domain + domain_value
            row["step_weighted_coefficient_domain_{}".format(domain)] = (
                prior_step_domain + domain_value * step_count)
        return row

    def _remap_epoch_metadata(self, start_batch: int, start_step: int, start_epoch: int) -> None:
        """Replace construction-time alpha labels with the current arm labels."""

        current_alpha = float(self.alpha)
        for row in self.batch_rows[start_batch:]:
            row.update({"arm_key": self.arm_key, "alpha": current_alpha,
                        "alpha_key": self.arm_key})
        for row in self.shared_gradient_step_rows[start_step:]:
            row.update({"stage": "E5", "arm_key": self.arm_key, "alpha": current_alpha,
                        "alpha_key": self.arm_key, "weighting": self.arm_key})
        for row in self.shared_gradient_epoch_rows[start_epoch:]:
            row.update({"stage": "E5", "arm_key": self.arm_key, "alpha": current_alpha,
                        "alpha_key": self.arm_key, "weighting": self.arm_key})
            cell = int(row.get("cell", -1))
            if 0 <= cell < 6:
                row["loss_coefficient"] = float(self.actual_coefficients[cell].detach().cpu())
                row["step_weighted_loss_coefficient"] = float(
                    self.actual_coefficients[cell].detach().cpu()) * len(self.train_loader)

    def _train_epoch(self, epoch: int):
        self.alpha = arm_alpha(self.arm_key, epoch)
        self.alpha_key = self.arm_key
        self.actual_coefficients = scheduled_coefficients(
            self.arm_key, epoch, self.domain_shares, self.task_num,
            dtype=torch.float64, device=self.device)
        validate_coefficients(self.arm_key, epoch, self.domain_shares,
                              self.actual_coefficients, self.task_num)
        schedule = self._schedule_row(epoch, self.actual_coefficients)
        self.schedule_rows.append(schedule)
        batch_start = len(self.batch_rows)
        step_start = len(self.shared_gradient_step_rows)
        epoch_start = len(self.shared_gradient_epoch_rows)
        # E3's helper currently accepts static alpha labels.  Run its exact
        # gradient path with the valid interior label 0.75 and remap only the
        # metadata; all coefficient-dependent arithmetic uses actual_coefficients.
        self.alpha = 0.75
        self.alpha_key = e3_alpha_key(0.75)
        train_stats, fingerprints = super()._train_epoch(epoch)
        self.alpha = arm_alpha(self.arm_key, epoch)
        self.alpha_key = self.arm_key
        self._remap_epoch_metadata(batch_start, step_start, epoch_start)
        return train_stats, fingerprints

    def _append_metric_rows(self, epoch: int, train_stats: Mapping[int, Mapping[str, object]],
                            val_stats: Mapping[int, Mapping[str, object]],
                            test_stats: Mapping[int, Mapping[str, object]], selected: bool,
                            censored: bool = False) -> None:
        start = len(self.metrics_rows)
        super()._append_metric_rows(epoch, train_stats, val_stats, test_stats, selected, censored)
        for row in self.metrics_rows[start:]:
            row.update({"stage": "E5", "arm_key": self.arm_key,
                        "alpha_key": self.arm_key, "weighting": self.arm_key})

    def fit(self) -> Dict[str, object]:
        """Run exactly the configured fixed epoch budget; never early-stop."""

        arch_iter = iter(self.train_loader)
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            train_stats, fingerprints = self._train_epoch(epoch)
            self.batch_fingerprints[epoch] = list(fingerprints)
            # Architecture update is deliberately after the main epoch, as in
            # E3.  It consumes the same natural batch sequence for every arm.
            arch_iter = self._architecture_update(epoch, arch_iter)
            val_stats = e3_run._metric_records(self.model, self.val_loader, self.device,
                                               self.domain_num, self.task_num)
            before = e3_run.rng_state_digest()
            with e3_run.preserve_rng_state():
                test_stats = e3_run._metric_records(
                    self.model, self.test_eval_loader, self.device,
                    self.domain_num, self.task_num)
            after = e3_run.rng_state_digest()
            self.rng_restore_evidence.append({
                "epoch": int(epoch), "scope": "test_eval", "before": before,
                "after": after, "restored": bool(before == after),
            })
            val_mean = strict_validation_macro_auc(
                val_stats, (idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)),
                label="E5 joint validation")
            is_best = bool(np.isfinite(val_mean) and val_mean > self.best_val_auc)
            if is_best:
                for row in self.metrics_rows:
                    if int(row.get("selected_checkpoint", 0)):
                        row["selected_checkpoint"] = 0
                self.best_val_auc = val_mean
                self.best_epoch = int(epoch)
                self.best_state = e3_run._cpu_state_dict(self.model)
            self._append_metric_rows(epoch, train_stats, val_stats, test_stats, is_best)
        self._last_epoch_parameter_state_hash = parameter_state_hash(self.model)
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {
            "best_epoch": self.best_epoch if self.best_epoch is not None else "",
            "stopped_epoch": "",
            "best_val_auc": self.best_val_auc if np.isfinite(self.best_val_auc) else "",
            "observed_epochs": list(self.observed_epochs),
            "fixed_final_epoch": int(self.args.epoch) - 1,
            "last_epoch_parameter_state_hash": self._last_epoch_parameter_state_hash,
        }

    def _summary_metric_map(self) -> Tuple[Dict[int, Dict[str, object]], Dict[int, Dict[str, object]], int]:
        selected = {
            int(row["cell"]): row for row in self.metrics_rows
            if int(row.get("selected_checkpoint", 0)) == 1
        }
        final_epoch = int(self.args.epoch) - 1
        final = {
            int(row["cell"]): row for row in self.metrics_rows
            if int(row.get("epoch", -1)) == final_epoch
        }
        return selected, final, final_epoch

    def _aggregate_view(self, rows: Mapping[int, Mapping[str, object]]) -> Dict[str, object]:
        aucs = [float(row["test_auc"]) for row in rows.values() if _finite(row.get("test_auc"))]
        losses = [float(row["test_logloss"]) for row in rows.values() if _finite(row.get("test_logloss"))]
        overall_auc = float(np.mean(aucs)) if len(aucs) == 6 else float("nan")
        overall_loss = float(np.mean(losses)) if len(losses) == 6 else float("nan")
        domain_auc: Dict[int, float] = {}
        task_auc: Dict[int, float] = {}
        for domain in range(self.domain_num):
            values = [float(row["test_auc"]) for cell, row in rows.items()
                      if cell % self.domain_num == domain and _finite(row.get("test_auc"))]
            if len(values) == self.task_num:
                domain_auc[domain] = float(np.mean(values))
        for task in range(self.task_num):
            values = [float(row["test_auc"]) for cell, row in rows.items()
                      if cell // self.domain_num == task and _finite(row.get("test_auc"))]
            if len(values) == self.domain_num:
                task_auc[task] = float(np.mean(values))
        return {
            "overall_auc": overall_auc, "overall_logloss": overall_loss,
            "domain_macro_auc": float(np.mean(list(domain_auc.values()))) if len(domain_auc) == self.domain_num else float("nan"),
            "task_macro_auc": float(np.mean(list(task_auc.values()))) if len(task_auc) == self.task_num else float("nan"),
            "worst_domain_auc": min(domain_auc.values()) if len(domain_auc) == self.domain_num else float("nan"),
            "worst_cell_auc": min(aucs) if len(aucs) == 6 else float("nan"),
        }

    def _write_metrics_summary(self, fit_info: Mapping[str, object]) -> List[Dict[str, object]]:
        selected, final, final_epoch = self._summary_metric_map()
        views = [("selected", selected, fit_info.get("best_epoch", "")),
                 ("fixed-final", final, final_epoch)]
        summary_rows: List[Dict[str, object]] = []
        exposure = exposure_table(self.arm_key, self.domain_shares, self.task_num,
                                  steps_per_epoch=len(self.train_loader),
                                  epochs=int(self.args.epoch))
        gradients = {int(row["cell"]): row for row in self.shared_gradient_epoch_rows}
        for view, cell_rows, target_epoch in views:
            aggregate = self._aggregate_view(cell_rows)
            for idx, domain, task in iter_cells(self.domain_num, self.task_num):
                metric = cell_rows.get(idx, {})
                try:
                    target = int(target_epoch)
                except (TypeError, ValueError):
                    target = final_epoch
                grad = gradients.get(idx, {})
                if target_epoch not in ("", None):
                    grad = next((row for row in self.shared_gradient_epoch_rows
                                 if int(row.get("epoch", -1)) == target and int(row.get("cell", -1)) == idx), grad)
                coefficient_integral = exposure["coefficient_integral_by_cell"][idx]
                step_integral = exposure["step_weighted_coefficient_integral_by_cell"][idx]
                transfer = metric.get("transfer_auc", "")
                summary_rows.append({
                    "stage": "E5", "arm_key": self.arm_key, "seed": self.seed,
                    "view": view, "epoch": target_epoch, "domain": domain,
                    "physical_domain": domain, "task": task, "cell": idx,
                    "size_role": self.role_by_domain[domain], "domain_size": self.train_counts[domain],
                    "domain_share": self.domain_shares[domain],
                    "train_sample_count": metric.get("train_sample_count", ""),
                    "validation_sample_count": metric.get("validation_sample_count", ""),
                    "test_sample_count": metric.get("test_sample_count", ""),
                    "train_auc": metric.get("train_auc", ""), "train_logloss": metric.get("train_logloss", ""),
                    "validation_auc": metric.get("validation_auc", ""), "validation_logloss": metric.get("validation_logloss", ""),
                    "test_auc": metric.get("test_auc", ""), "test_logloss": metric.get("test_logloss", ""),
                    "oracle_auc": metric.get("oracle_auc", ""), "oracle_logloss": metric.get("oracle_logloss", ""),
                    "oracle_source": metric.get("oracle_source", "missing"),
                    "oracle_coverage": metric.get("oracle_coverage", 0),
                    "transfer_auc": transfer, "transfer_logloss": metric.get("transfer_logloss", ""),
                    "negative_transfer": int(_finite(transfer) and float(transfer) < 0),
                    **aggregate,
                    "peak_epoch": fit_info.get("best_epoch", ""),
                    "fixed_final_epoch": final_epoch, "selected_epoch": fit_info.get("best_epoch", ""),
                    "selected_checkpoint": int(view == "selected"), "common_support": 1,
                    "censored": 0, "alpha_at_epoch": arm_alpha(self.arm_key, target),
                    "alpha_exposure_total": exposure["alpha_sum"],
                    "coefficient_integral_cell": coefficient_integral,
                    "step_weighted_coefficient_integral_cell": step_integral,
                    "shared_raw_gradient_norm": grad.get("raw_gradient_norm", ""),
                    "shared_weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                    "shared_gradient_variance": grad.get("gradient_variance", ""),
                    "shared_gradient_snr": grad.get("gradient_SNR", ""),
                    "shared_magnitude_share": grad.get("magnitude_share", ""),
                    "shared_signed_contribution": grad.get("contribution_mean", ""),
                    "shared_mismatch_index": grad.get("mismatch_index", ""),
                    "shared_update_alignment": self._selected_update_alignment(domain, task, target),
                })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), summary_rows,
                       SUMMARY_FIELDS)
        return summary_rows

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows,
                       METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_step.csv"),
                       self.shared_gradient_step_rows, STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_epoch.csv"),
                       self.shared_gradient_epoch_rows, EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       BATCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "schedule_epoch.csv"), self.schedule_rows,
                       SCHEDULE_FIELDS)
        summary_rows = self._write_metrics_summary(fit_info)
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "metrics_summary.csv": summary_rows,
            "shared_gradient_step.csv": self.shared_gradient_step_rows,
            "shared_gradient_epoch.csv": self.shared_gradient_epoch_rows,
            "batch_trace.csv": self.batch_rows,
            "schedule_epoch.csv": self.schedule_rows,
        })
        inventory = {
            "stage": "E5", "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(1 for item in self.nonfinite_inventory
                                      if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
            "scope_note": "raw non-finite values inventoried before CSV conversion",
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory)
        # model.pth stores the validation-selected state, while the fixed-final
        # test metrics remain in metrics_epoch.csv and are never used for choice.
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_rows = [row for row in self.metrics_rows if int(row.get("selected_checkpoint", 0)) == 1]
        selected_complete = (len(selected_rows) == 6
                             and {int(row.get("cell", -1)) for row in selected_rows} == set(range(6))
                             and all(_finite(row.get(field)) for row in selected_rows
                                     for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss")))
        formal_valid = self._formal_valid(selected_complete, inventory)
        schedule_payload = {
            "arm_key": self.arm_key,
            "alpha": [row["alpha"] for row in self.schedule_rows],
            "coefficients": [[row["coefficient_cell_{}".format(cell)] for cell in range(6)]
                             for row in self.schedule_rows],
        }
        observed_hash = _json_hash(schedule_payload)
        planned_payload = {
            "arm_key": self.arm_key,
            "alpha": self._planned_trajectory,
            "coefficients": [scheduled_coefficients_numpy(self.arm_key, epoch,
                                                            self.domain_shares, self.task_num).tolist()
                             for epoch in range(int(self.args.epoch))],
        }
        planned_hash = _json_hash(planned_payload)
        exposure = exposure_table(self.arm_key, self.domain_shares, self.task_num,
                                  steps_per_epoch=len(self.train_loader),
                                  epochs=int(self.args.epoch))
        config = vars(self.args).copy()
        config.update({
            "stage": "E5", "arm_key": self.arm_key, "alpha": None,
            "alpha_key": self.arm_key, "weighting": self.arm_key,
            "setting": self.setting, "seed": self.seed, "backbone": "mdmoe",
            "scheduler": None, "task": "all", "domain": "all", "fixed_epoch_budget": True,
            "role_sizes_parsed": self.sizes, "domain_shares": self.domain_shares.tolist(),
            "train_domain_counts": self.train_counts,
            "schedule_formula": schedule_formula(self.arm_key),
            "schedule_formula_version": "E5-v1", "schedule_registry_hash": schedule_registry_hash(),
            "planned_alpha_trajectory": self._planned_trajectory,
            "observed_alpha_trajectory": [row["alpha"] for row in self.schedule_rows],
            "planned_schedule_hash": planned_hash, "observed_schedule_hash": observed_hash,
            "coefficient_integral_by_cell": exposure["coefficient_integral_by_cell"],
            "coefficient_integral_by_domain": exposure["coefficient_integral_by_domain"],
            "step_weighted_coefficient_integral_by_cell": exposure["step_weighted_coefficient_integral_by_cell"],
            "step_weighted_coefficient_integral_by_domain": exposure["step_weighted_coefficient_integral_by_domain"],
            "cell_order": e3_run.CELL_ORDER_NOTE, "instrumentation": True,
            "prohibited_mechanisms": ["adaptive_controller", "EMA", "momentum", "new_sampling",
                                       "new_router", "warmup", "role_reversal", "schedule_grid"],
            "architecture_objective": "unmasked BCE; E5 coefficients apply only to main six-cell loss",
            "sampling": E5_FORMAL_PROTOCOL["sampling"],
            "dataset_identity": self.dataset_identity, "device_identity": self.device_identity,
            "split_row_fingerprints": self.split_row_fingerprints,
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": sorted(int(row["cell"]) for row in selected_rows),
            "formal_protocol": dict(E5_FORMAL_PROTOCOL), "formal_valid": bool(formal_valid),
            "config_hash": "",
        })
        config["config_hash"] = _json_hash(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E5", "status": "OK", "seed": self.seed, "arm_key": self.arm_key,
            "setting": self.setting, "dataset": "MovieLens-1M",
            "dataset_path": os.path.abspath(self.args.dataset_path),
            "dataset_identity": self.dataset_identity, "device_identity": self.device_identity,
            "domain_num": self.domain_num, "task_num": self.task_num,
            "role_by_domain": self.role_by_domain, "role_sizes": self.sizes,
            "train_domain_counts": self.train_counts, "domain_shares": self.domain_shares.tolist(),
            "split_counts": self.split_counts, "positive_rates": self.positive_rates,
            "split_row_fingerprints": self.split_row_fingerprints,
            "oracle_audit": self.oracle_audit, "oracle_metrics": self.oracle_metrics,
            "partition": self.partition.as_dict(),
            "model_identity": {"class": "MDMTRec", "backbone": "mdmoe", "fcn_dims": [512, 256, 256, 64],
                               "expert_num": int(self.args.expert_num), "domain_num": self.domain_num,
                               "task_num": self.task_num},
            "initial_parameter_state_hash": self.partition.state_hash,
            "last_epoch_parameter_state_hash": self._last_epoch_parameter_state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "observed_epochs": list(self.observed_epochs), "epoch": int(self.args.epoch),
            "fixed_final_epoch": int(self.args.epoch) - 1,
            "best_epoch": fit_info.get("best_epoch", ""), "stopped_epoch": "",
            "total_main_steps": int(sum(int(row["main_step_count"]) for row in self.schedule_rows)),
            "total_architecture_steps": int(len(self.arch_batch_fingerprints)),
            "total_processed_examples": int(sum(self.train_counts.values()) * len(self.schedule_rows)),
            "formal_protocol": dict(E5_FORMAL_PROTOCOL), "formal_valid": bool(formal_valid),
            "fixed_epoch_budget": True, "schedule_formula": schedule_formula(self.arm_key),
            "schedule_formula_version": "E5-v1", "schedule_registry_hash": schedule_registry_hash(),
            "planned_alpha_trajectory": self._planned_trajectory,
            "observed_alpha_trajectory": [row["alpha"] for row in self.schedule_rows],
            "planned_schedule_hash": planned_hash, "observed_schedule_hash": observed_hash,
            "coefficient_integral_by_cell": exposure["coefficient_integral_by_cell"],
            "coefficient_integral_by_domain": exposure["coefficient_integral_by_domain"],
            "step_weighted_coefficient_integral_by_cell": exposure["step_weighted_coefficient_integral_by_cell"],
            "step_weighted_coefficient_integral_by_domain": exposure["step_weighted_coefficient_integral_by_domain"],
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": sorted(int(row["cell"]) for row in selected_rows),
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.arch_batch_fingerprints,
            "shared_gradient_step_count": len(self.shared_gradient_step_rows),
            "shared_gradient_epoch_count": len(self.shared_gradient_epoch_rows),
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory["unexplained_count"],
            "oracle_coverage": self.oracle_audit.get("coverage_fraction", 0.0),
            "block_status": "PENDING", "block_valid": None,
            "wall_seconds": float(wall_seconds),
            "scope_note": "raw outputs only; no cross-arm scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)

    def _formal_valid(self, selected_complete: bool, inventory: Mapping[str, object]) -> bool:
        args = self.args
        expected = E5_FORMAL_PROTOCOL
        return bool(
            str(args.setting) == E5_FORMAL_SETTING
            and int(args.epoch) == E5_EPOCHS
            and int(args.batch_size) == int(expected["batch_size"])
            and float(args.learning_rate) == float(expected["learning_rate"])
            and float(args.learning_rate_darts) == float(expected["learning_rate_darts"])
            and float(args.weight_decay) == float(expected["weight_decay"])
            and int(args.expert_num) == int(expected["expert_num"])
            and int(args.probe_interval) == int(expected["probe_interval"])
            and int(args.instrument) == 1
            and str(args.role_sizes) == str(expected["role_sizes"])
            and self.sizes == E5_FORMAL_ROLE_SIZES
            and self.train_counts == {domain: int({"L": 150000, "M": 34500, "S": 3000}[self.role_by_domain[domain]])
                                      for domain in range(self.domain_num)}
            and self.observed_epochs == list(E5_EPOCH_INDICES)
            and len(self.schedule_rows) == E5_EPOCHS
            and selected_complete
            and int(inventory.get("unexplained_count", 1)) == 0
            and bool(self.oracle_audit.get("complete", False))
            and all(bool(item.get("restored")) for item in self.rng_restore_evidence)
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E5 one-run conditional contribution schedule")
    parser.add_argument("--arm_key", choices=list(E5_ARM_KEYS), required=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--setting", choices=[E5_FORMAL_SETTING], default=E5_FORMAL_SETTING)
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=E5_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "stage_E3_raw_results", "oracle_reference"))
    parser.add_argument("--legacy_oracle_root", default=os.path.join(HERE, "P0_2_ratio_sweep"))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--instrument", type=int, choices=[0, 1], default=1)
    parser.add_argument("--probe_interval", type=int, default=1)
    # Retained as an identity field for parity with E3; E5 fit never consults
    # it because the fixed 18-epoch budget is an identification constraint.
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

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args: argparse.Namespace) -> int:
    os.makedirs(os.path.abspath(args.run_dir), exist_ok=True)
    log_path = args.stdout_log or os.path.join(os.path.abspath(args.run_dir), "stdout.log")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    started = time.time()
    runner: Optional[E5Runner] = None
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
            sys.stdout = _Tee(old_stdout, log_handle)
            sys.stderr = _Tee(old_stderr, log_handle)
            runner = E5Runner(args)
            fit_info = runner.fit()
            runner.write_outputs(fit_info, time.time() - started)
            print("[stage_E5] raw outputs:", runner.run_dir, flush=True)
            return 0
    except Exception:
        # The log handle is owned by the ``with`` block and is closed before
        # control reaches this handler; restore console streams before printing
        # the traceback so an error cannot mask the original exception.
        sys.stdout, sys.stderr = old_stdout, old_stderr
        traceback.print_exc()
        return 1
    finally:
        if runner is not None:
            runner.close()
        sys.stdout, sys.stderr = old_stdout, old_stderr


def main() -> None:
    args = build_argparser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
