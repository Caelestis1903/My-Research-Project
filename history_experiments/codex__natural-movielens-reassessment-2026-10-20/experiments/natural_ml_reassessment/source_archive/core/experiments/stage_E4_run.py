"""Single formal/smoke joint run for Stage E4 KuaiRand.

The runner owns one ``(weighting, seed)`` arm.  It emits observations and
integrity metadata only; cross-arm comparison and scientific interpretation
belong to the orchestrator and main analysis respectively.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)

from MDMTRec.models.multi_domain import MDMTRec  # noqa: E402
from MDMTRec.utils.data import DataGenerator, TorchDataset  # noqa: E402

from stage_E1_instrumentation import (  # noqa: E402
    CELL_ORDER_NOTE,
    collect_nonfinite_inventory,
    contribution_views,
    flatten_parameter_tensors,
    masked_bce_terms,
    parameter_state_hash,
    preserve_rng_state,
    relative_l2_error,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_E4_weighting import (  # noqa: E402
    DOMAIN_NUM,
    E4_FORMAL_PROTOCOL,
    E4_DATASET_PATH,
    SHARED_EPOCH_FIELDS,
    SHARED_STEP_FIELDS,
    TASK_NUM,
    WEIGHTING_KEYS,
    DatasetBundle,
    SharedGradientEpochAccumulator,
    audit_dataset_bundle,
    batch_fingerprint,
    build_parameter_partition,
    canonical_weighting,
    cell_index,
    collect_shared_raw_gradients,
    filter_split_first_domain,
    iter_cells,
    load_processed_dataset,
    make_shared_gradient_step_rows,
    metric_records,
    normalise_dataset_path,
    setup_seed,
    split_row_fingerprint,
    strict_validation_macro_auc,
    weighting_alpha,
    weighting_coefficients,
    weighting_coefficients_numpy,
    weighting_display,
    weighted_loss_from_terms,
    json_hash,
)

# The import list above deliberately keeps all E4 primitives visible to code
# review.  Remove the compatibility aliases created by older prototypes.


METRICS_FIELDS = [
    "seed", "weighting", "weighting_display", "alpha", "epoch", "domain", "task", "cell",
    "physical_domain", "domain_share", "train_sample_count", "validation_sample_count",
    "test_sample_count", "train_positive_rate", "validation_positive_rate",
    "test_positive_rate", "train_auc", "train_logloss", "validation_auc",
    "validation_logloss", "test_auc", "test_logloss", "train_pooled_auc",
    "train_pooled_logloss", "validation_pooled_auc", "validation_pooled_logloss",
    "test_pooled_auc", "test_pooled_logloss", "selected_checkpoint", "stopped_censored",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc",
    "transfer_logloss",
]

METRICS_SUMMARY_FIELDS = [
    "seed", "weighting", "weighting_display", "alpha", "domain", "physical_domain", "task", "cell",
    "domain_share", "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate", "selected_epoch",
    "stopped_epoch", "best_val_auc", "best_val_logloss", "best_test_auc", "best_test_logloss",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc",
    "transfer_logloss", "negative_transfer", "domain_macro_auc", "task_macro_auc",
    "overall_auc", "overall_logloss", "worst_domain_auc", "worst_cell_auc", "pooled_auc",
    "pooled_logloss", "shared_raw_gradient_norm", "shared_weighted_gradient_norm",
    "shared_magnitude_share", "shared_signed_contribution", "shared_mismatch_index",
    "shared_update_alignment", "selected_checkpoint", "selected_epoch_in_common_support",
    "common_support", "common_epoch_count", "common_trace_count", "block_status",
    "block_valid", "block_validity", "censored", "artifact_validity",
]

BATCH_FIELDS = [
    "seed", "weighting", "weighting_display", "alpha", "epoch", "step", "kind",
    "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "is_probe", "train_loss",
]


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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


def _oracle_candidates(root: str, domain: int, seed: int) -> List[str]:
    base = os.path.abspath(str(root))
    return [
        os.path.join(base, f"d{domain}", f"seed{seed}"),
        os.path.join(base, "oracle_reference", f"d{domain}", f"seed{seed}"),
    ]


def _resolve_oracle_metrics(
    root: str, bundle: DatasetBundle, seed: int,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    """Resolve only matching formal E4 oracle artifacts; missing is explicit."""

    result: Dict[int, Dict[str, object]] = {
        idx: {"auc": "", "logloss": "", "source": "missing", "coverage": False}
        for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)
    }
    audit: Dict[str, object] = {"seed": int(seed), "domains": [], "coverage_cells": 0,
                                "expected_cells": DOMAIN_NUM * TASK_NUM}
    expected_by_domain: Dict[int, Mapping[str, object]] = {}
    for domain in range(DOMAIN_NUM):
        _filtered, info = filter_split_first_domain(bundle, domain)
        expected_by_domain[domain] = info
        chosen = None
        reason = "missing"
        manifest: Dict[str, object] = {}
        summary: List[Dict[str, str]] = []
        for candidate in _oracle_candidates(root, domain, seed) if root else []:
            candidate_manifest = _read_json(os.path.join(candidate, "run_manifest.json"))
            candidate_config = _read_json(os.path.join(candidate, "config.json"))
            candidate_summary = _read_csv(os.path.join(candidate, "metrics_summary.csv"))
            required = ("config.json", "run_manifest.json", "metrics_epoch.csv",
                        "metrics_summary.csv", "stdout.log", "model.pth")
            if not os.path.isdir(candidate) or any(
                    not os.path.exists(os.path.join(candidate, name)) for name in required):
                reason = "missing_or_incomplete"
                continue
            if candidate_manifest.get("stage") != "E4_ORACLE" \
                    or candidate_manifest.get("status") != "OK" \
                    or candidate_manifest.get("formal_valid") is not True \
                    or candidate_config.get("formal_valid") is not True:
                reason = "formal_invalid"
                continue
            if int(candidate_manifest.get("seed", -1)) != int(seed) \
                    or int(candidate_manifest.get("original_physical_domain", -1)) != domain:
                reason = "identity_mismatch"
                continue
            if candidate_manifest.get("dataset_identity") != bundle.dataset_identity \
                    or candidate_config.get("dataset_identity") != bundle.dataset_identity:
                reason = "dataset_identity_mismatch"
                continue
            if candidate_manifest.get("split_row_fingerprints") != info["split_row_fingerprints"]:
                reason = "filtered_split_identity_mismatch"
                continue
            if candidate_manifest.get("global_split_row_fingerprints") \
                    != info["global_split_row_fingerprints"]:
                reason = "global_split_identity_mismatch"
                continue
            selected = [row for row in candidate_summary
                        if row.get("selected_checkpoint") in ("1", 1)]
            # Oracle summaries are one row per task and have no required
            # selected_checkpoint column in older smoke fixtures; use finite
            # best_test fields when the marker is absent.
            if not selected:
                selected = [row for row in candidate_summary
                            if _finite(row.get("best_test_auc"))]
            by_task = {int(row.get("task", -1)): row for row in selected
                       if str(row.get("task", "")) in ("0", "1")}
            if set(by_task) != {0, 1} or any(not _finite(by_task[t].get("best_test_auc"))
                                            or not _finite(by_task[t].get("best_test_logloss"))
                                            for t in (0, 1)):
                reason = "oracle_summary_incomplete"
                continue
            chosen, manifest, summary = candidate, candidate_manifest, candidate_summary
            break
        entry = {"domain": domain, "status": "reused" if chosen else "missing",
                 "source": chosen or "", "reason": "OK" if chosen else reason}
        audit["domains"].append(entry)
        if chosen:
            selected = [row for row in summary if _finite(row.get("best_test_auc"))]
            by_task = {int(row.get("task", -1)): row for row in selected
                       if str(row.get("task", "")) in ("0", "1")}
            for task in range(TASK_NUM):
                row = by_task.get(task)
                if row is None:
                    continue
                idx = cell_index(task, domain)
                result[idx] = {
                    "auc": row.get("best_test_auc", ""),
                    "logloss": row.get("best_test_logloss", ""),
                    "source": os.path.abspath(chosen), "coverage": True,
                    "original_physical_domain": domain, "seed": int(seed),
                }
    audit["coverage_cells"] = sum(1 for row in result.values() if row.get("coverage"))
    audit["coverage_fraction"] = audit["coverage_cells"] / float(audit["expected_cells"])
    audit["complete"] = audit["coverage_cells"] == audit["expected_cells"]
    return result, audit


class E4Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = int(args.seed)
        self.weighting = canonical_weighting(args.weighting)
        if args.alpha is not None and abs(float(args.alpha) - weighting_alpha(self.weighting)) > 1.0e-10:
            raise ValueError("--alpha disagrees with the registered E4 weighting")
        self.alpha = weighting_alpha(self.weighting)
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {args.device!r}, but CUDA is unavailable")
        setup_seed(self.seed)
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.bundle = load_processed_dataset(args.dataset_path, validate=True)
        self.dataset_audit = audit_dataset_bundle(self.bundle, strict=True)
        self.domain_num = DOMAIN_NUM
        self.task_num = TASK_NUM
        self.domain_shares = np.asarray([
            self.dataset_audit["split_counts"]["train"][d]
            / float(sum(self.dataset_audit["split_counts"]["train"].values()))
            for d in range(DOMAIN_NUM)
        ], dtype=np.float64)
        self.actual_coefficients = weighting_coefficients(
            self.weighting, self.domain_shares, TASK_NUM, dtype=torch.float64, device=self.device)
        self.equal_coefficients = weighting_coefficients(
            "equal", self.domain_shares, TASK_NUM, dtype=torch.float64, device=self.device)
        self.sqrt_coefficients = weighting_coefficients(
            "sqrt", self.domain_shares, TASK_NUM, dtype=torch.float64, device=self.device)
        self.sample_coefficients = weighting_coefficients(
            "sample", self.domain_shares, TASK_NUM, dtype=torch.float64, device=self.device)
        self._build_loaders()
        self._build_model()
        self.oracle_metrics, self.oracle_audit = _resolve_oracle_metrics(
            str(args.oracle_root), self.bundle, self.seed)
        self._init_outputs()

    def _build_loaders(self) -> None:
        generator = DataGenerator(self.bundle.x_train, self.bundle.y_train)
        self.train_loader, self.val_loader, self.test_loader = generator.generate_dataloader(
            x_val=self.bundle.x_validation, y_val=self.bundle.y_validation,
            x_test=self.bundle.x_test, y_test=self.bundle.y_test,
            batch_size=int(self.args.batch_size), num_workers=0, drop_last_flag=False)
        self.test_eval_loader = DataLoader(
            TorchDataset(self.bundle.x_test, self.bundle.y_test),
            batch_size=int(self.args.batch_size), shuffle=False, num_workers=0, drop_last=False)

    def _build_model(self) -> None:
        self.model = MDMTRec(
            list(self.bundle.dense_features) + list(self.bundle.sparse_features),
            domain_num=DOMAIN_NUM, task_num=TASK_NUM,
            fcn_dims=[512, 256, 256, 64], expert_num=int(self.args.expert_num),
            exp_d=1, exp_t=1, bal_d=1, bal_t=1,
        ).to(self.device)
        self.partition = build_parameter_partition(self.model, strict=True)
        named = dict(self.model.named_parameters())
        main_params = [named[name] for name in self.partition.main_names if named[name].requires_grad]
        arch_params = [named[name] for name in self.partition.names("architecture") if named[name].requires_grad]
        if set(main_params).intersection(set(arch_params)):
            raise AssertionError("main and architecture optimizer groups overlap")
        self.optimizer = torch.optim.Adam(
            main_params, lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = torch.optim.Adam(
            arch_params, lr=float(self.args.learning_rate_darts),
            weight_decay=float(self.args.weight_decay)) if arch_params else None
        self.criterion = torch.nn.BCELoss()

    def _init_outputs(self) -> None:
        self.metrics_rows: List[Dict[str, object]] = []
        self.shared_gradient_step_rows: List[Dict[str, object]] = []
        self.shared_gradient_epoch_rows: List[Dict[str, object]] = []
        self.batch_rows: List[Dict[str, object]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: Optional[int] = None
        self.best_val_auc = float("-inf")
        self.stopped_epoch: Optional[int] = None
        self.observed_epochs: List[int] = []
        self.batch_fingerprints: Dict[int, List[str]] = {}
        self.architecture_batch_fingerprints: List[str] = []
        self.rng_restore_evidence: List[Dict[str, object]] = []
        self.nonfinite_inventory: List[Dict[str, object]] = []

    def _record_batch(
        self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
        y_batch: torch.Tensor, counts: Sequence[int], is_probe: bool,
        train_loss: Optional[object] = None,
    ) -> str:
        fingerprint = batch_fingerprint(x_batch, y_batch)
        self.batch_rows.append({
            "seed": self.seed, "weighting": self.weighting,
            "weighting_display": weighting_display(self.weighting), "alpha": self.alpha,
            "epoch": int(epoch), "step": int(step), "kind": str(kind),
            "batch_fingerprint": fingerprint, "batch_size": int(len(y_batch)),
            "domain_counts": _json_counts(counts),
            "task_counts": _json_counts([int(y_batch[:, task].numel()) for task in range(TASK_NUM)]),
            "is_probe": int(bool(is_probe)),
            "train_loss": "" if train_loss is None else float(
                train_loss.detach().cpu() if isinstance(train_loss, torch.Tensor) else train_loss),
        })
        return fingerprint

    def _shared_vector(self) -> torch.Tensor:
        named = dict(self.model.named_parameters())
        names = list(self.partition.names("shared"))
        if not names:
            return torch.zeros(0, device=self.device)
        return torch.cat([named[name].detach().reshape(-1).clone() for name in names])

    def _cell_counts(self, x_batch: torch.Tensor) -> List[int]:
        domains = x_batch[:, -1].detach().cpu().numpy().astype(int)
        return [int((domains == domain).sum()) for _, domain, _ in iter_cells(DOMAIN_NUM, TASK_NUM)]

    def _append_metric_rows(
        self, epoch: int, train_stats: Mapping[int, Mapping[str, object]],
        val_stats: Mapping[int, Mapping[str, object]], test_stats: Mapping[int, Mapping[str, object]],
        train_pooled: Mapping[str, object], val_pooled: Mapping[str, object],
        test_pooled: Mapping[str, object], selected: bool, censored: bool = False,
    ) -> None:
        for idx, domain, task in iter_cells(DOMAIN_NUM, TASK_NUM):
            train, val, test = train_stats.get(idx, {}), val_stats.get(idx, {}), test_stats.get(idx, {})
            oracle = self.oracle_metrics.get(idx, {})
            row: Dict[str, object] = {
                "seed": self.seed, "weighting": self.weighting,
                "weighting_display": weighting_display(self.weighting), "alpha": self.alpha,
                "epoch": int(epoch), "domain": domain, "task": task, "cell": idx,
                "physical_domain": domain, "domain_share": self.domain_shares[domain],
                "train_sample_count": train.get("sample_count", self.dataset_audit["split_counts"]["train"][domain]),
                "validation_sample_count": val.get("sample_count", self.dataset_audit["split_counts"]["validation"][domain]),
                "test_sample_count": test.get("sample_count", self.dataset_audit["split_counts"]["test"][domain]),
                "train_positive_rate": train.get("positive_rate", self.dataset_audit["positive_rates"]["train"][domain][task]),
                "validation_positive_rate": val.get("positive_rate", self.dataset_audit["positive_rates"]["validation"][domain][task]),
                "test_positive_rate": test.get("positive_rate", self.dataset_audit["positive_rates"]["test"][domain][task]),
                "train_auc": train.get("auc", ""), "train_logloss": train.get("logloss", ""),
                "validation_auc": val.get("auc", ""), "validation_logloss": val.get("logloss", ""),
                "test_auc": test.get("auc", ""), "test_logloss": test.get("logloss", ""),
                "train_pooled_auc": train_pooled.get("auc", ""),
                "train_pooled_logloss": train_pooled.get("logloss", ""),
                "validation_pooled_auc": val_pooled.get("auc", ""),
                "validation_pooled_logloss": val_pooled.get("logloss", ""),
                "test_pooled_auc": test_pooled.get("auc", ""),
                "test_pooled_logloss": test_pooled.get("logloss", ""),
                "selected_checkpoint": int(bool(selected)), "stopped_censored": int(bool(censored)),
                "oracle_auc": oracle.get("auc", ""), "oracle_logloss": oracle.get("logloss", ""),
                "oracle_source": oracle.get("source", "missing"),
                "oracle_coverage": int(bool(oracle.get("coverage", False))),
                "transfer_auc": "", "transfer_logloss": "",
            }
            if _finite(row["oracle_auc"]) and _finite(row["test_auc"]):
                row["transfer_auc"] = float(row["test_auc"]) - float(row["oracle_auc"])
            if _finite(row["oracle_logloss"]) and _finite(row["test_logloss"]):
                row["transfer_logloss"] = float(row["test_logloss"]) - float(row["oracle_logloss"])
            self.metrics_rows.append(row)

    def _architecture_update(self, epoch: int, arch_iter):
        try:
            x_arch, y_arch = next(arch_iter)
        except StopIteration:
            arch_iter = iter(self.train_loader)
            x_arch, y_arch = next(arch_iter)
        fingerprint = self._record_batch(
            epoch, -1, "architecture", x_arch, y_arch, self._cell_counts(x_arch), False)
        self.architecture_batch_fingerprints.append(fingerprint)
        if self.arch_optimizer is None:
            return arch_iter
        self.model.train()
        x_device = x_arch.transpose(0, 1).to(self.device)
        y_device = y_arch.to(self.device).float()
        prediction = self.model(x_device).to(self.device)
        arch_loss = self.criterion(prediction, y_device)
        self.batch_rows[-1]["train_loss"] = float(arch_loss.detach().cpu())
        self.model.zero_grad(set_to_none=True)
        arch_loss.backward()
        self.arch_optimizer.step()
        return arch_iter

    def _train_epoch(self, epoch: int) -> Tuple[Dict[int, Dict[str, object]], Dict[str, float], List[str]]:
        self.model.train()
        train_truth = {idx: [] for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        train_prediction = {idx: [] for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        train_loss_sum = {idx: 0.0 for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        train_counts = {idx: 0 for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        accumulator = SharedGradientEpochAccumulator()
        fingerprints: List[str] = []
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            counts_for_cells = self._cell_counts(x_batch)
            is_probe = bool(int(self.args.instrument)) and step % int(self.args.probe_interval) == 0
            fingerprint = self._record_batch(
                epoch, step, "train", x_batch, y_batch, counts_for_cells, is_probe)
            fingerprints.append(fingerprint)
            x_device = x_batch.transpose(0, 1).to(self.device)
            y_device = y_batch.to(self.device).float()
            domain_ids = x_device[-1, :].detach()
            prediction = self.model(x_device).to(self.device)
            terms, counts = masked_bce_terms(
                prediction, y_device, domain_ids, DOMAIN_NUM, TASK_NUM, self.criterion)
            for idx, domain, task in iter_cells(DOMAIN_NUM, TASK_NUM):
                if counts[idx] > 0:
                    train_counts[idx] += counts[idx]
                    train_loss_sum[idx] += float(terms[idx].detach().cpu()) * counts[idx]
                    mask = domain_ids == domain
                    train_truth[idx].extend(y_device[mask, task].detach().cpu().tolist())
                    train_prediction[idx].extend(prediction[mask, task].detach().cpu().tolist())
            loss = weighted_loss_from_terms(terms, self.actual_coefficients)
            self.batch_rows[-1]["train_loss"] = float(loss.detach().cpu())
            if is_probe:
                raw_by_cell, unused_by_cell = collect_shared_raw_gradients(
                    terms, self.model, self.partition)
                views = contribution_views(raw_by_cell, self.actual_coefficients,
                                           DOMAIN_NUM, TASK_NUM)
                domain_magnitude = {
                    d: sum(float(views["magnitude"].get(cell_index(t, d), 0.0))
                           for t in range(TASK_NUM)) for d in range(DOMAIN_NUM)
                }
                mismatch = 0.5 * sum(abs(domain_magnitude[d] - float(self.domain_shares[d]))
                                     for d in range(DOMAIN_NUM))
                accumulator.update(raw_by_cell, views, mismatch)
                before = self._shared_vector()
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                named = dict(self.model.named_parameters())
                shared_names = list(self.partition.names("shared"))
                actual_grad, _ = flatten_parameter_tensors(
                    named, shared_names,
                    [named[name].grad for name in shared_names], device=self.device)
                reconstruction = relative_l2_error(actual_grad, views["combined"])
                self.optimizer.step()
                after = self._shared_vector()
                self.shared_gradient_step_rows.extend(make_shared_gradient_step_rows(
                    raw_by_cell, unused_by_cell, weighting=self.weighting,
                    coefficients=self.actual_coefficients,
                    equal_coefficients=self.equal_coefficients,
                    sqrt_coefficients=self.sqrt_coefficients,
                    sample_coefficients=self.sample_coefficients,
                    domain_shares=self.domain_shares,
                    shared_param_count=self.partition.counts["shared"], seed=self.seed,
                    epoch=epoch, step=step, batch_fingerprint_value=fingerprint,
                    cell_sample_counts=counts, actual_update=before - after,
                    reconstruction_error=reconstruction))
            else:
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        if bool(int(self.args.instrument)):
            self.shared_gradient_epoch_rows.extend(
                accumulator.rows(weighting=self.weighting, coefficients=self.actual_coefficients,
                                 seed=self.seed, epoch=epoch))
        train_stats = {
            idx: {
                "sample_count": train_counts[idx],
                "positive_rate": float(np.mean(train_truth[idx])) if train_truth[idx] else float("nan"),
                "auc": self._safe_auc(train_truth[idx], train_prediction[idx]),
                "logloss": train_loss_sum[idx] / train_counts[idx]
                if train_counts[idx] else float("nan"),
            } for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)
        }
        flat_truth = [value for idx in sorted(train_truth) for value in train_truth[idx]]
        flat_pred = [value for idx in sorted(train_prediction) for value in train_prediction[idx]]
        pooled = {"auc": self._safe_auc(flat_truth, flat_pred),
                  "logloss": self._safe_logloss(flat_truth, flat_pred)}
        return train_stats, pooled, fingerprints

    @staticmethod
    def _safe_auc(truth: Sequence[float], prediction: Sequence[float]) -> float:
        from stage_E4_weighting import safe_auc
        return safe_auc(truth, prediction)

    @staticmethod
    def _safe_logloss(truth: Sequence[float], prediction: Sequence[float]) -> float:
        from stage_E4_weighting import safe_logloss
        return safe_logloss(truth, prediction)

    def fit(self) -> Dict[str, object]:
        arch_iter = iter(self.train_loader)
        patience = int(self.args.earlystop_patience)
        bad_epochs = 0
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            train_stats, train_pooled, fingerprints = self._train_epoch(epoch)
            self.batch_fingerprints[epoch] = list(fingerprints)
            arch_iter = self._architecture_update(epoch, arch_iter)
            val_stats, val_pooled = metric_records(
                self.model, self.val_loader, self.device, DOMAIN_NUM, TASK_NUM)
            before = rng_state_digest()
            with preserve_rng_state():
                test_stats, test_pooled = metric_records(
                    self.model, self.test_eval_loader, self.device, DOMAIN_NUM, TASK_NUM)
            after = rng_state_digest()
            self.rng_restore_evidence.append({
                "epoch": int(epoch), "scope": "test_eval", "before": before,
                "after": after, "restored": bool(before == after),
            })
            val_mean = strict_validation_macro_auc(
                val_stats, range(DOMAIN_NUM * TASK_NUM))
            is_best = bool(np.isfinite(val_mean) and val_mean > self.best_val_auc)
            if is_best:
                for row in self.metrics_rows:
                    if row.get("selected_checkpoint"):
                        row["selected_checkpoint"] = 0
                self.best_val_auc = val_mean
                self.best_epoch = epoch
                self.best_state = {name: value.detach().cpu().clone()
                                   for name, value in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            self._append_metric_rows(
                epoch, train_stats, val_stats, test_stats,
                train_pooled, val_pooled, test_pooled, is_best)
            if bad_epochs >= patience:
                self.stopped_epoch = epoch
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.stopped_epoch is not None and self.stopped_epoch + 1 < int(self.args.epoch):
            for censored_epoch in range(self.stopped_epoch + 1, int(self.args.epoch)):
                self._append_metric_rows(
                    censored_epoch, {}, {}, {}, {}, {}, {}, False, censored=True)
        return {
            "best_epoch": self.best_epoch if self.best_epoch is not None else "",
            "stopped_epoch": self.stopped_epoch if self.stopped_epoch is not None else "",
            "best_val_auc": self.best_val_auc if np.isfinite(self.best_val_auc) else "",
            "observed_epochs": list(self.observed_epochs),
        }

    def _selected_rows(self) -> Dict[int, Dict[str, object]]:
        return {int(row["cell"]): row for row in self.metrics_rows
                if int(row.get("selected_checkpoint", 0)) == 1}

    def _selected_checkpoint_complete(self) -> Tuple[bool, List[str]]:
        expected = set(range(DOMAIN_NUM * TASK_NUM))
        selected = self._selected_rows()
        if set(selected) != expected or len(selected) != len(expected):
            return False, [str(idx) for idx in sorted(selected)]
        fields = ("validation_auc", "validation_logloss", "test_auc", "test_logloss")
        if any(not _finite(row.get(field)) for row in selected.values() for field in fields):
            return False, [str(idx) for idx in sorted(selected)]
        return True, [str(idx) for idx in sorted(selected)]

    def _selected_gradient_row(self, domain: int, task: int, epoch: object) -> Dict[str, object]:
        if epoch in (None, ""):
            return {}
        rows = [row for row in self.shared_gradient_epoch_rows
                if int(row.get("epoch", -1)) == int(epoch)
                and int(row.get("domain", -1)) == domain
                and int(row.get("task", -1)) == task]
        return rows[-1] if rows else {}

    def _selected_step_values(self, domain: int, task: int, epoch: object) -> Dict[str, object]:
        if epoch in (None, ""):
            return {}
        rows = [row for row in self.shared_gradient_step_rows
                if int(row.get("epoch", -1)) == int(epoch)
                and int(row.get("domain", -1)) == domain
                and int(row.get("task", -1)) == task]
        if not rows:
            return {}
        alignment = [float(row["actual_update_alignment"]) for row in rows
                     if _finite(row.get("actual_update_alignment"))]
        return {
            "shared_magnitude_share": rows[-1].get("magnitude_share", ""),
            "shared_update_alignment": float(np.mean(alignment)) if alignment else float("nan"),
        }

    def _write_metrics_summary(self, fit_info: Mapping[str, object]) -> List[Dict[str, object]]:
        selected = self._selected_rows()
        complete = set(selected) == set(range(DOMAIN_NUM * TASK_NUM)) and all(
            _finite(row.get("test_auc")) and _finite(row.get("test_logloss"))
            for row in selected.values())
        aucs = {idx: float(row["test_auc"]) for idx, row in selected.items()} if complete else {}
        losses = {idx: float(row["test_logloss"]) for idx, row in selected.items()} if complete else {}
        domain_auc = {
            d: float(np.mean([aucs[cell_index(t, d)] for t in range(TASK_NUM)]))
            for d in range(DOMAIN_NUM) if all(cell_index(t, d) in aucs for t in range(TASK_NUM))
        }
        task_auc = {
            t: float(np.mean([aucs[cell_index(t, d)] for d in range(DOMAIN_NUM)]))
            for t in range(TASK_NUM) if all(cell_index(t, d) in aucs for d in range(DOMAIN_NUM))
        }
        overall_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        overall_ll = float(np.mean(list(losses.values()))) if losses else float("nan")
        best_epoch = fit_info.get("best_epoch", "")
        selected_metric = next(iter(selected.values()), {})
        pooled_auc = selected_metric.get("test_pooled_auc", "")
        pooled_ll = selected_metric.get("test_pooled_logloss", "")
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(DOMAIN_NUM, TASK_NUM):
            metric = selected.get(idx, {})
            grad = self._selected_gradient_row(domain, task, best_epoch)
            step = self._selected_step_values(domain, task, best_epoch)
            oracle = self.oracle_metrics.get(idx, {})
            transfer = metric.get("transfer_auc", "")
            rows.append({
                "seed": self.seed, "weighting": self.weighting,
                "weighting_display": weighting_display(self.weighting), "alpha": self.alpha,
                "domain": domain, "physical_domain": domain, "task": task, "cell": idx,
                "domain_share": self.domain_shares[domain],
                "train_sample_count": self.dataset_audit["split_counts"]["train"][domain],
                "validation_sample_count": self.dataset_audit["split_counts"]["validation"][domain],
                "test_sample_count": self.dataset_audit["split_counts"]["test"][domain],
                "train_positive_rate": self.dataset_audit["positive_rates"]["train"][domain][task],
                "validation_positive_rate": self.dataset_audit["positive_rates"]["validation"][domain][task],
                "test_positive_rate": self.dataset_audit["positive_rates"]["test"][domain][task],
                "selected_epoch": best_epoch, "stopped_epoch": fit_info.get("stopped_epoch", ""),
                "best_val_auc": metric.get("validation_auc", ""),
                "best_val_logloss": metric.get("validation_logloss", ""),
                "best_test_auc": metric.get("test_auc", ""),
                "best_test_logloss": metric.get("test_logloss", ""),
                "oracle_auc": oracle.get("auc", ""), "oracle_logloss": oracle.get("logloss", ""),
                "oracle_source": oracle.get("source", "missing"),
                "oracle_coverage": int(bool(oracle.get("coverage", False))),
                "transfer_auc": transfer, "transfer_logloss": metric.get("transfer_logloss", ""),
                "negative_transfer": int(_finite(transfer) and float(transfer) < 0),
                "domain_macro_auc": domain_auc.get(domain, float("nan")),
                "task_macro_auc": task_auc.get(task, float("nan")),
                "overall_auc": overall_auc, "overall_logloss": overall_ll,
                "worst_domain_auc": min(domain_auc.values()) if domain_auc else float("nan"),
                "worst_cell_auc": min(aucs.values()) if aucs else float("nan"),
                "pooled_auc": pooled_auc, "pooled_logloss": pooled_ll,
                "shared_raw_gradient_norm": grad.get("raw_gradient_norm", ""),
                "shared_weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                "shared_magnitude_share": step.get("shared_magnitude_share", grad.get("magnitude_share", "")),
                "shared_signed_contribution": grad.get("contribution_mean", ""),
                "shared_mismatch_index": grad.get("mismatch_index", ""),
                "shared_update_alignment": step.get("shared_update_alignment", ""),
                # Cross-arm support is filled by the orchestrator once all
                # three weighting arms for this seed are complete.  Before
                # that point it is intentionally unknown, rather than a
                # misleading constant zero or a single-arm proxy.
                "selected_checkpoint": int(best_epoch not in (None, "")),
                "selected_epoch_in_common_support": "",
                "common_support": "",
                "common_epoch_count": "", "common_trace_count": "",
                "block_status": "PENDING", "block_valid": "",
                "block_validity": "PENDING", "censored": int(self.stopped_epoch is not None),
                "artifact_validity": "SMOKE_OR_FORMAL_PENDING",
            })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), rows,
                       METRICS_SUMMARY_FIELDS)
        return rows

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows,
                       METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_step.csv"),
                       self.shared_gradient_step_rows, SHARED_STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_epoch.csv"),
                       self.shared_gradient_epoch_rows, SHARED_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       BATCH_FIELDS)
        summary_rows = self._write_metrics_summary(fit_info)
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "shared_gradient_step.csv": self.shared_gradient_step_rows,
            "shared_gradient_epoch.csv": self.shared_gradient_epoch_rows,
            "batch_trace.csv": self.batch_rows,
            "metrics_summary.csv": summary_rows,
        })
        inventory = {
            "stage": "E4", "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(1 for item in self.nonfinite_inventory
                                      if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
            "scope_note": "raw non-finite values inventoried before CSV conversion",
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory)
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_complete, selected_cells = self._selected_checkpoint_complete()
        formal_fields = (
            "epoch", "batch_size", "learning_rate", "learning_rate_darts", "weight_decay",
            "expert_num", "earlystop_patience", "probe_interval", "instrument",
        )
        config = vars(self.args).copy()
        config.update({
            "stage": "E4", "dataset": "KuaiRand-Pure", "dataset_path": self.bundle.dataset_path,
            "dataset_identity": self.bundle.dataset_identity,
            "dataset_file_identity": dict(self.bundle.dataset_file_identity),
            "device_identity": str(self.device), "seed": self.seed,
            "weighting": self.weighting, "weighting_display": weighting_display(self.weighting),
            "alpha": self.alpha, "domain_num": DOMAIN_NUM, "task_num": TASK_NUM,
            "domain_mapping": {"tab0": 0, "tab1": 1, "tab4": 2},
            "task_mapping": {"is_click": 0, "long_view": 1},
            "domain_shares": self.domain_shares.tolist(),
            "train_domain_counts": self.dataset_audit["split_counts"]["train"],
            "split_counts": self.dataset_audit["split_counts"],
            "positive_counts": self.dataset_audit["positive_counts"],
            "positive_rates": self.dataset_audit["positive_rates"],
            "split_row_fingerprints": self.dataset_audit["split_row_fingerprints"],
            "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.equal_coefficients.detach().cpu().tolist(),
            "sqrt_coefficients": self.sqrt_coefficients.detach().cpu().tolist(),
            "sample_coefficients": self.sample_coefficients.detach().cpu().tolist(),
            "cell_order": CELL_ORDER_NOTE, "instrumentation": bool(int(self.args.instrument)),
            "sampling": E4_FORMAL_PROTOCOL["sampling"], "scheduler": None,
            "backbone": "mdmoe", "model": "MDMTRec", "fcn_dims": [512, 256, 256, 64],
            "architecture_update": "once_per_epoch_first_train_batch",
            "optimizer_grouping": self.partition.as_dict(),
            "prohibited_mechanisms": list(E4_FORMAL_PROTOCOL["prohibited_mechanisms"]),
            "oracle_audit": self.oracle_audit,
            "oracle_coverage_complete": bool(self.oracle_audit.get("complete", False)),
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": selected_cells,
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "observed_epochs": list(self.observed_epochs),
            "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": self.stopped_epoch + 1 if self.stopped_epoch is not None else None,
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.architecture_batch_fingerprints,
        })
        config["formal_protocol"] = dict(E4_FORMAL_PROTOCOL)
        config["formal_valid"] = bool(
            all(config.get(field) == E4_FORMAL_PROTOCOL.get(field) for field in formal_fields)
            and config.get("batch_size") == E4_FORMAL_PROTOCOL["batch_size"]
            and config.get("setting", "all") == "all"
            and config.get("instrument") == 1
            and config.get("probe_interval") == E4_FORMAL_PROTOCOL["probe_interval"]
            and config.get("weighting") in WEIGHTING_KEYS
            and config.get("domain_num") == DOMAIN_NUM
            and config.get("task_num") == TASK_NUM
            and self.dataset_audit.get("valid") is True
            and selected_complete
            and bool(self.oracle_audit.get("complete", False))
            and len(self.observed_epochs) > 0
            and (self.stopped_epoch is not None or len(self.observed_epochs) == int(E4_FORMAL_PROTOCOL["epoch"]))
            and os.path.exists(os.path.join(self.run_dir, "model.pth"))
            and self.bundle.dataset_identity == normalise_dataset_path(config.get("dataset_path", ""))
        )
        summary_status = "FORMAL_VALID" if config["formal_valid"] else "SMOKE_ONLY"
        for row in summary_rows:
            row["artifact_validity"] = summary_status
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), summary_rows,
                       METRICS_SUMMARY_FIELDS)
        config["config_hash"] = json_hash(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E4", "status": "OK", "seed": self.seed,
            "weighting": self.weighting, "weighting_display": weighting_display(self.weighting),
            "alpha": self.alpha, "formal_run_key": f"{self.weighting}|{self.seed}",
            "dataset": "KuaiRand-Pure", "dataset_path": self.bundle.dataset_path,
            "dataset_identity": self.bundle.dataset_identity,
            "dataset_file_identity": dict(self.bundle.dataset_file_identity),
            "device_identity": str(self.device), "domain_num": DOMAIN_NUM, "task_num": TASK_NUM,
            "domain_mapping": {"tab0": 0, "tab1": 1, "tab4": 2},
            "task_mapping": {"is_click": 0, "long_view": 1},
            "train_domain_counts": self.dataset_audit["split_counts"]["train"],
            "domain_shares": self.domain_shares.tolist(),
            "split_counts": self.dataset_audit["split_counts"],
            "positive_counts": self.dataset_audit["positive_counts"],
            "positive_rates": self.dataset_audit["positive_rates"],
            "split_row_fingerprints": self.dataset_audit["split_row_fingerprints"],
            "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.equal_coefficients.detach().cpu().tolist(),
            "sqrt_coefficients": self.sqrt_coefficients.detach().cpu().tolist(),
            "sample_coefficients": self.sample_coefficients.detach().cpu().tolist(),
            "partition": self.partition.as_dict(),
            "model_identity": {
                "class": "MDMTRec", "backbone": "mdmoe", "fcn_dims": [512, 256, 256, 64],
                "expert_num": int(self.args.expert_num), "domain_num": DOMAIN_NUM, "task_num": TASK_NUM,
                "parameter_count": int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
            },
            "optimizer_identity": {
                "main": {"class": "Adam", "lr": float(self.args.learning_rate),
                         "weight_decay": float(self.args.weight_decay)},
                "architecture": {"class": "Adam", "lr": float(self.args.learning_rate_darts),
                                  "weight_decay": float(self.args.weight_decay)},
            },
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "observed_epochs": list(self.observed_epochs), "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": self.stopped_epoch + 1 if self.stopped_epoch is not None else None,
            "formal_protocol": dict(E4_FORMAL_PROTOCOL),
            "formal_valid": bool(config["formal_valid"]),
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": selected_cells,
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.architecture_batch_fingerprints,
            "shared_gradient_step_count": len(self.shared_gradient_step_rows),
            "shared_gradient_epoch_count": len(self.shared_gradient_epoch_rows),
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory["unexplained_count"],
            "oracle_audit": self.oracle_audit, "oracle_coverage_complete": bool(self.oracle_audit.get("complete", False)),
            "block_status": "PENDING", "block_valid": None,
            "wall_seconds": float(wall_seconds),
            "scope_note": "raw outputs only; no scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)
        if not os.path.exists(os.path.join(self.run_dir, "stdout.log")):
            with open(os.path.join(self.run_dir, "stdout.log"), "a", encoding="utf-8") as handle:
                handle.write("[stage_E4] raw outputs written\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E4 one-run KuaiRand static weighting")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--weighting", default="equal",
                        help="Equal/Sqrt/Sample (case-insensitive)")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default=E4_DATASET_PATH)
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "stage_E4_raw_results", "oracle_reference"))
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--instrument", type=int, choices=[0, 1], default=1)
    parser.add_argument("--probe_interval", type=int, default=10)
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
    runner = E4Runner(args)
    fit_info = runner.fit()
    runner.write_outputs(fit_info, time.time() - started)
    print("[stage_E4] weighting:", runner.weighting, "alpha:", runner.alpha, flush=True)
    print("[stage_E4] raw outputs:", runner.run_dir, flush=True)
    return 0


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
