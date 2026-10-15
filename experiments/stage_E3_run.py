"""Single-run implementation for Stage E3's five-point alpha sweep.

Each invocation trains exactly one ``(alpha, setting, seed)`` arm.  The data,
model, optimizer, architecture-update cadence, validation checkpoint rule and
RNG-safe evaluation path mirror the E1/P0-4 protocol.  The only objective
change is the registered E3 alpha coefficient vector.

This file emits raw observations only.  It deliberately does not select an
alpha across runs or make a scientific claim.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)

from MDMTRec.models.multi_domain import MDMTRec  # noqa: E402
from MDMTRec.utils.data import DataGenerator, TorchDataset  # noqa: E402
from run_movielens import get_movielens_data_rank_multidomain  # noqa: E402

from stage_E1_instrumentation import (  # noqa: E402
    CELL_ORDER_NOTE,
    RoutingObserver,
    build_parameter_partition,
    flatten_parameter_tensors,
    parameter_state_hash,
    preserve_rng_state,
    relative_l2_error,
    rng_state_digest,
    safe_auc,
    safe_logloss,
    write_json,
    write_rows_csv,
)
from stage_E3_weighting import (  # noqa: E402
    ALPHA_GRID,
    E3_FORMAL_PROTOCOL,
    E3_GRADIENT_EPOCH_FIELDS,
    E3_GRADIENT_STEP_FIELDS,
    ROLE_MAPS,
    SharedGradientEpochAccumulator,
    alpha_key,
    canonical_alpha,
    collect_shared_cell_gradients,
    contribution_views,
    iter_cells,
    masked_bce_terms,
    make_shared_gradient_step_rows,
    resolve_oracle_metrics,
    normalise_dataset_path,
    split_row_fingerprint,
    strict_validation_macro_auc,
    weighted_loss_from_terms,
    weighting_coefficients,
)


METRICS_FIELDS = [
    "seed", "epoch", "domain", "task", "cell", "alpha", "alpha_key", "weighting",
    "setting", "physical_domain", "size_role", "domain_size", "domain_share",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate",
    "train_auc", "train_logloss", "validation_auc", "validation_logloss",
    "test_auc", "test_logloss", "selected_checkpoint", "stopped_censored",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage",
    "transfer_auc", "transfer_logloss",
]

METRICS_SUMMARY_FIELDS = [
    "seed", "setting", "alpha", "alpha_key", "weighting", "domain", "physical_domain",
    "task", "cell", "size_role", "domain_size", "domain_share", "selected_epoch",
    "stopped_epoch", "best_val_auc", "best_val_logloss", "best_test_auc",
    "best_test_logloss", "oracle_auc", "oracle_logloss", "oracle_source",
    "oracle_coverage", "transfer_auc", "transfer_logloss", "overall_auc",
    "overall_logloss", "worst_domain_auc", "worst_cell_auc", "negative_transfer",
    "shared_raw_gradient_norm", "shared_weighted_gradient_norm", "shared_magnitude_share",
    "shared_signed_contribution", "shared_mismatch_index", "shared_update_alignment",
    "common_support", "censored",
]

BATCH_FIELDS = [
    "seed", "setting", "alpha", "alpha_key", "epoch", "step", "kind",
    "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "is_probe",
    "train_loss",
]


def setup_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True


def parse_role_sizes(value: str) -> Dict[str, int]:
    parts = [int(item.strip()) for item in str(value).split(",")]
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise ValueError("--role_sizes must be three positive integers: L,M,S")
    return dict(zip(("L", "M", "S"), parts))


def role_reversal_sample(x_train: np.ndarray, y_train: np.ndarray, setting: str,
                         seed: int, sizes: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    if setting not in ROLE_MAPS:
        raise ValueError("unknown role setting {!r}".format(setting))
    domain_ids = x_train[:, -1].astype(int)
    keep = np.zeros(len(x_train), dtype=bool)
    for domain, role in ROLE_MAPS[setting].items():
        pool = np.where(domain_ids == domain)[0]
        requested = int(sizes[role])
        if requested > len(pool):
            raise ValueError("role size {} exceeds physical domain {} pool {}".format(
                requested, domain, len(pool)))
        permutation = np.random.RandomState(int(seed) * 10 + domain).permutation(len(pool))
        keep[pool[permutation[:requested]]] = True
    return x_train[keep], y_train[keep]


def oracle_split_row_fingerprints(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    setting: str, seed: int, sizes: Mapping[str, int],
) -> Dict[int, Dict[str, str]]:
    """Fingerprint the exact per-domain rows used by the single-domain oracle.

    This mirrors the oracle loader: apply the current setting/seed role
    reversal, select one physical domain, and remap its domain column to zero
    in fresh copies before hashing ordered feature/label rows.
    """

    selected_x, selected_y = role_reversal_sample(x_train, y_train, setting, seed, sizes)
    result: Dict[int, Dict[str, str]] = {}
    for domain in range(len(ROLE_MAPS[setting])):
        train_mask = selected_x[:, -1].astype(int) == domain
        val_mask = x_val[:, -1].astype(int) == domain
        test_mask = x_test[:, -1].astype(int) == domain
        train_view = np.array(selected_x[train_mask], copy=True)
        val_view = np.array(x_val[val_mask], copy=True)
        test_view = np.array(x_test[test_mask], copy=True)
        for view in (train_view, val_view, test_view):
            view[:, -1] = 0
        result[domain] = {
            "train": split_row_fingerprint(train_view, selected_y[train_mask]),
            "validation": split_row_fingerprint(val_view, y_val[val_mask]),
            "test": split_row_fingerprint(test_view, y_test[test_mask]),
        }
    return result


def _positive_rate(values: np.ndarray, task: int) -> float:
    return float(np.asarray(values[:, task], dtype=np.float64).mean()) if len(values) else float("nan")


def _parameter_vector(named: Mapping[str, torch.nn.Parameter], names: Sequence[str]) -> torch.Tensor:
    if not names:
        return torch.zeros(0)
    return torch.cat([named[name].detach().reshape(-1).clone() for name in names])


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _metric_records(model: torch.nn.Module, data_loader: DataLoader, device: torch.device,
                    domain_num: int, task_num: int) -> Dict[int, Dict[str, object]]:
    """Exact per-cell metrics from one deterministic, non-shuffling loader."""

    truth: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    prediction: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    counts: Dict[int, int] = {idx: 0 for idx, _, _ in iter_cells(domain_num, task_num)}
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_device = x_batch.transpose(0, 1).to(device)
            y_device = y_batch.to(device)
            pred = model(x_device).to(device)
            domain_ids = x_device[-1, :].detach()
            for idx, domain, task in iter_cells(domain_num, task_num):
                mask = domain_ids == domain
                if bool(mask.any()):
                    truth[idx].extend(y_device[mask, task].detach().cpu().tolist())
                    prediction[idx].extend(pred[mask, task].detach().cpu().tolist())
                    counts[idx] += int(mask.sum().detach().cpu())
    return {
        idx: {
            "sample_count": counts[idx],
            "positive_rate": (float(np.mean(truth[idx])) if truth[idx] else float("nan")),
            "auc": safe_auc(truth[idx], prediction[idx]),
            "logloss": safe_logloss(truth[idx], prediction[idx]),
        }
        for idx, _, _ in iter_cells(domain_num, task_num)
    }


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class E3Runner:
    """One alpha/setting/seed run with shared-only contribution logging."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = int(args.seed)
        self.setting = str(args.setting)
        self.alpha = canonical_alpha(args.alpha)
        self.alpha_key = alpha_key(self.alpha)
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device {!r}, but CUDA is unavailable".format(args.device))
        setup_seed(self.seed)
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.sizes = parse_role_sizes(args.role_sizes)
        self._load_data()
        self._build_loaders()
        self._build_model()
        self._init_outputs()

    def _load_data(self) -> None:
        dense, sparse, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = \
            get_movielens_data_rank_multidomain("all", "all", self.args.dataset_path)
        self.dense_features = dense
        self.sparse_features = sparse
        self.x_train, self.y_train = role_reversal_sample(
            x_train, y_train, self.setting, self.seed, self.sizes)
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.domain_num = int(domain_num)
        self.task_num = 2
        self.dataset_identity = normalise_dataset_path(self.args.dataset_path)
        self.device_identity = str(self.device)
        self.split_row_fingerprints = {
            "train": split_row_fingerprint(self.x_train, self.y_train),
            "validation": split_row_fingerprint(self.x_val, self.y_val),
            "test": split_row_fingerprint(self.x_test, self.y_test),
        }
        self.oracle_expected_split_row_fingerprints = oracle_split_row_fingerprints(
            x_train, y_train, x_val, y_val, x_test, y_test,
            self.setting, self.seed, self.sizes)
        domain_ids = self.x_train[:, -1].astype(int)
        self.train_counts = {d: int((domain_ids == d).sum()) for d in range(self.domain_num)}
        total = float(sum(self.train_counts.values()))
        self.domain_shares = np.asarray(
            [self.train_counts[d] / total for d in range(self.domain_num)], dtype=np.float64)
        self.role_by_domain = ROLE_MAPS[self.setting]
        self.oracle_metrics, self.oracle_audit = resolve_oracle_metrics(
            [self.args.oracle_root, getattr(self.args, "legacy_oracle_root", "")],
            self.setting, self.seed, role_sizes=self.sizes,
            domain_num=self.domain_num, task_num=self.task_num,
            dataset_identity=self.dataset_identity,
            device_identity=self.device_identity,
            expected_split_row_fingerprints=self.oracle_expected_split_row_fingerprints)
        self.split_counts: Dict[str, Dict[int, int]] = {}
        self.positive_rates: Dict[str, Dict[int, float]] = {}
        for split, x_values, y_values in (("train", self.x_train, self.y_train),
                                           ("validation", self.x_val, self.y_val),
                                           ("test", self.x_test, self.y_test)):
            ids = x_values[:, -1].astype(int)
            self.split_counts[split] = {}
            self.positive_rates[split] = {}
            for idx, domain, task in iter_cells(self.domain_num, self.task_num):
                mask = ids == domain
                self.split_counts[split][idx] = int(mask.sum())
                self.positive_rates[split][idx] = _positive_rate(y_values[mask], task)
        expected = {"L": self.sizes["L"], "M": self.sizes["M"], "S": self.sizes["S"]}
        observed = {self.role_by_domain[d]: self.train_counts[d] for d in range(self.domain_num)}
        if observed != expected:
            raise AssertionError("role sampling counts differ: expected {} observed {}".format(
                expected, observed))
        self.actual_coefficients = weighting_coefficients(
            self.alpha, self.domain_shares, self.task_num,
            dtype=torch.float64, device=self.device)
        self.equal_coefficients = weighting_coefficients(
            0.0, self.domain_shares, self.task_num,
            dtype=torch.float64, device=self.device)
        self.sample_coefficients = weighting_coefficients(
            1.0, self.domain_shares, self.task_num,
            dtype=torch.float64, device=self.device)

    def _build_loaders(self) -> None:
        data_generator = DataGenerator(self.x_train, self.y_train)
        self.train_loader, self.val_loader, self.test_loader = data_generator.generate_dataloader(
            x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test,
            batch_size=int(self.args.batch_size), num_workers=0)
        self.test_eval_loader = DataLoader(
            TorchDataset(self.x_test, self.y_test), batch_size=int(self.args.batch_size),
            shuffle=False, num_workers=0)
        if bool(int(self.args.instrument)):
            probe_count = (len(self.train_loader) + int(self.args.probe_interval) - 1) // int(self.args.probe_interval)
            if probe_count < 2:
                raise ValueError("instrumented E3 epoch must expose at least two fixed probes")

    def _build_model(self) -> None:
        self.model = MDMTRec(
            self.dense_features + self.sparse_features,
            domain_num=self.domain_num, task_num=self.task_num,
            fcn_dims=[512, 256, 256, 64], expert_num=int(self.args.expert_num),
            exp_d=1, exp_t=1, bal_d=1, bal_t=1,
        ).to(self.device)
        self.partition = build_parameter_partition(self.model, strict=True)
        named = dict(self.model.named_parameters())
        main_names = list(self.partition.main_names)
        arch_names = list(self.partition.names("architecture"))
        main_params = [named[name] for name in main_names if named[name].requires_grad]
        arch_params = [named[name] for name in arch_names if named[name].requires_grad]
        self.optimizer = torch.optim.Adam(
            main_params, lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = torch.optim.Adam(
            arch_params, lr=float(self.args.learning_rate_darts),
            weight_decay=float(self.args.weight_decay)) if arch_params else None
        self.criterion = torch.nn.BCELoss()
        # E3 has no routing/private observer output.  The disabled observer is
        # retained only to make the distinction from the E1 full diagnostic
        # explicit and to avoid registering hooks that could alter a path.
        self.routing = RoutingObserver(self.model, self.domain_num, self.task_num, enabled=False)

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
        self.arch_batch_fingerprints: List[str] = []
        self.rng_restore_evidence: List[Dict[str, object]] = []
        self.nonfinite_inventory: List[Dict[str, object]] = []

    def _record_batch(self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
                      y_batch: torch.Tensor, counts: Sequence[int], is_probe: bool,
                      train_loss: Optional[object] = None) -> str:
        from stage_E1_instrumentation import batch_fingerprint
        fp = batch_fingerprint(x_batch, y_batch)
        self.batch_rows.append({
            "seed": self.seed, "setting": self.setting, "alpha": self.alpha,
            "alpha_key": self.alpha_key, "epoch": int(epoch), "step": int(step),
            "kind": str(kind), "batch_fingerprint": fp, "batch_size": int(len(y_batch)),
            "domain_counts": _json_counts(counts), "task_counts": _json_counts(
                [int(y_batch[:, task].numel()) for task in range(self.task_num)]),
            "is_probe": int(bool(is_probe)),
            "train_loss": ("" if train_loss is None else float(
                train_loss.detach().cpu() if isinstance(train_loss, torch.Tensor) else train_loss)),
        })
        return fp

    def _vectors_shared(self) -> torch.Tensor:
        named = dict(self.model.named_parameters())
        return _parameter_vector(named, self.partition.names("shared"))

    def _append_metric_rows(self, epoch: int, train_stats: Mapping[int, Mapping[str, object]],
                            val_stats: Mapping[int, Mapping[str, object]],
                            test_stats: Mapping[int, Mapping[str, object]], selected: bool,
                            censored: bool = False) -> None:
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            train, val, test = train_stats.get(idx, {}), val_stats.get(idx, {}), test_stats.get(idx, {})
            oracle = self.oracle_metrics.get(idx, {})
            row: Dict[str, object] = {
                "seed": self.seed, "epoch": int(epoch), "domain": domain, "task": task,
                "cell": idx, "alpha": self.alpha, "alpha_key": self.alpha_key,
                "weighting": self.alpha_key, "setting": self.setting,
                "physical_domain": domain, "size_role": self.role_by_domain[domain],
                "domain_size": self.train_counts[domain], "domain_share": self.domain_shares[domain],
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

    def _architecture_update(self, epoch: int, arch_iter):
        try:
            x_arch, y_arch = next(arch_iter)
        except StopIteration:
            arch_iter = iter(self.train_loader)
            x_arch, y_arch = next(arch_iter)
        domain_cpu = x_arch[:, -1].detach().cpu().numpy().astype(int)
        counts = [int((domain_cpu == domain).sum()) for _, domain, _ in
                  iter_cells(self.domain_num, self.task_num)]
        fp = self._record_batch(epoch, -1, "architecture", x_arch, y_arch, counts, False)
        self.arch_batch_fingerprints.append(fp)
        if self.arch_optimizer is None:
            return arch_iter
        x_device = x_arch.transpose(0, 1).to(self.device)
        y_device = y_arch.to(self.device).float()
        pred = self.model(x_device).to(self.device)
        # Same unmasked architecture objective/cadence as E1.  Alpha only
        # controls the main six-cell loss, never this optimizer path.
        arch_loss = self.criterion(pred, y_device)
        self.batch_rows[-1]["train_loss"] = float(arch_loss.detach().cpu())
        self.model.zero_grad(set_to_none=True)
        arch_loss.backward()
        self.arch_optimizer.step()
        return arch_iter

    def _train_epoch(self, epoch: int) -> Tuple[Dict[int, Dict[str, object]], List[str]]:
        self.model.train()
        train_truth: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_prediction: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_loss_sum = {idx: 0.0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_counts = {idx: 0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        accumulator = SharedGradientEpochAccumulator(self.domain_num, self.task_num)
        fingerprints: List[str] = []
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            domain_cpu = x_batch[:, -1].detach().cpu().numpy().astype(int)
            cell_counts = [int((domain_cpu == domain).sum()) for _, domain, _ in
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
            for idx, _, _ in iter_cells(self.domain_num, self.task_num):
                if counts[idx] > 0:
                    train_loss_sum[idx] += float(terms[idx].detach().cpu()) * counts[idx]
                    train_counts[idx] += counts[idx]
                    mask = domain_ids == (idx % self.domain_num)
                    train_truth[idx].extend(y_device[mask, idx // self.domain_num].detach().cpu().tolist())
                    train_prediction[idx].extend(y_pred[mask, idx // self.domain_num].detach().cpu().tolist())
            loss = weighted_loss_from_terms(terms, self.actual_coefficients)
            self.batch_rows[-1]["train_loss"] = float(loss.detach().cpu())
            if is_probe:
                raw_by_cell, unused_by_cell = collect_shared_cell_gradients(
                    terms, self.model, self.partition, retain_graph=True)
                actual_views = contribution_views(
                    raw_by_cell, self.actual_coefficients, self.domain_num, self.task_num)
                domain_magnitude = {
                    domain: sum(float(actual_views["magnitude"][idx])
                                for idx, d, _ in iter_cells(self.domain_num, self.task_num)
                                if d == domain)
                    for domain in range(self.domain_num)
                }
                mismatch = 0.5 * sum(
                    abs(domain_magnitude[domain] - float(self.domain_shares[domain]))
                    for domain in range(self.domain_num))
                accumulator.update(raw_by_cell, actual_views["normalized"],
                                   mismatch,
                                   actual_views["magnitude"])
                before = self._vectors_shared()
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                named = dict(self.model.named_parameters())
                shared_names = list(self.partition.names("shared"))
                grad_values = [named[name].grad for name in shared_names]
                actual_grad, _ = flatten_parameter_tensors(
                    named, shared_names, grad_values, device=self.device)
                combined = actual_views["combined"]
                reconstruction = relative_l2_error(actual_grad, combined)
                self.optimizer.step()
                after = self._vectors_shared()
                update = before - after
                self.shared_gradient_step_rows.extend(make_shared_gradient_step_rows(
                    raw_by_cell, unused_by_cell, alpha=self.alpha,
                    coefficients=self.actual_coefficients,
                    equal_coefficients=self.equal_coefficients,
                    sample_coefficients=self.sample_coefficients,
                    domain_num=self.domain_num, task_num=self.task_num,
                    shared_param_count=self.partition.counts["shared"],
                    domain_shares=self.domain_shares, seed=self.seed, setting=self.setting,
                    batch_fingerprint=fingerprint, epoch=epoch, step=step,
                    cell_sample_counts=counts, actual_update=update,
                    reconstruction_error=reconstruction))
            else:
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        if bool(int(self.args.instrument)):
            self.shared_gradient_epoch_rows.extend(accumulator.rows(
                alpha=self.alpha, coefficients=self.actual_coefficients, seed=self.seed,
                setting=self.setting, epoch=epoch))
        train_stats = {
            idx: {
                "sample_count": train_counts[idx],
                "positive_rate": (float(np.mean(train_truth[idx])) if train_truth[idx] else float("nan")),
                "auc": safe_auc(train_truth[idx], train_prediction[idx]),
                "logloss": (train_loss_sum[idx] / train_counts[idx]
                            if train_counts[idx] else float("nan")),
            }
            for idx, _, _ in iter_cells(self.domain_num, self.task_num)
        }
        return train_stats, fingerprints

    def fit(self) -> Dict[str, object]:
        arch_iter = iter(self.train_loader)
        patience = int(self.args.earlystop_patience)
        bad_epochs = 0
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            train_stats, fingerprints = self._train_epoch(epoch)
            self.batch_fingerprints[epoch] = list(fingerprints)
            arch_iter = self._architecture_update(epoch, arch_iter)
            val_stats = _metric_records(self.model, self.val_loader, self.device,
                                        self.domain_num, self.task_num)
            before = rng_state_digest()
            with preserve_rng_state():
                test_stats = _metric_records(self.model, self.test_eval_loader, self.device,
                                             self.domain_num, self.task_num)
            after = rng_state_digest()
            self.rng_restore_evidence.append({
                "epoch": int(epoch), "scope": "test_eval", "before": before,
                "after": after, "restored": bool(before == after),
            })
            val_mean = strict_validation_macro_auc(
                val_stats,
                (idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)),
                label="joint validation",
            )
            is_best = bool(np.isfinite(val_mean) and val_mean > self.best_val_auc)
            if is_best:
                for row in self.metrics_rows:
                    if row["selected_checkpoint"]:
                        row["selected_checkpoint"] = 0
                self.best_val_auc = val_mean
                self.best_epoch = epoch
                self.best_state = _cpu_state_dict(self.model)
                bad_epochs = 0
            else:
                bad_epochs += 1
            self._append_metric_rows(epoch, train_stats, val_stats, test_stats, is_best)
            if bad_epochs >= patience:
                self.stopped_epoch = epoch
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.stopped_epoch is not None and self.stopped_epoch + 1 < int(self.args.epoch):
            for censored_epoch in range(self.stopped_epoch + 1, int(self.args.epoch)):
                self._append_metric_rows(censored_epoch, {}, {}, {}, False, censored=True)
        return {
            "best_epoch": self.best_epoch if self.best_epoch is not None else "",
            "stopped_epoch": self.stopped_epoch if self.stopped_epoch is not None else "",
            "best_val_auc": self.best_val_auc if np.isfinite(self.best_val_auc) else "",
            "observed_epochs": list(self.observed_epochs),
        }

    def _selected_metric_rows(self) -> Dict[int, Dict[str, object]]:
        return {int(row["cell"]): row for row in self.metrics_rows
                if int(row.get("selected_checkpoint", 0)) == 1}

    def _selected_checkpoint_complete(self) -> Tuple[bool, List[str]]:
        expected = {idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        selected_rows = [row for row in self.metrics_rows
                         if int(row.get("selected_checkpoint", 0)) == 1]
        selected = self._selected_metric_rows()
        if len(selected_rows) != len(expected) or set(selected) != expected:
            return False, [str(idx) for idx in sorted(selected)]
        required = ("validation_auc", "validation_logloss", "test_auc", "test_logloss")
        if any(not _finite_number(row.get(field)) for row in selected.values() for field in required):
            return False, [str(idx) for idx in sorted(selected)]
        return True, [str(idx) for idx in sorted(selected)]

    def _write_metrics_summary(self, fit_info: Mapping[str, object]) -> List[Dict[str, object]]:
        selected = self._selected_metric_rows()
        expected = {idx for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        selected_complete = (
            set(selected) == expected
            and all(_finite_number(row.get(field))
                    for row in selected.values()
                    for field in ("test_auc", "test_logloss"))
        )
        # Keep summary macro fields invalid as a whole when any selected cell
        # is missing/non-finite; never average a reduced population.
        test_auc = ({idx: float(row["test_auc"]) for idx, row in selected.items()}
                    if selected_complete else {})
        test_ll = ({idx: float(row["test_logloss"]) for idx, row in selected.items()}
                   if selected_complete else {})
        domain_auc = {d: float(np.mean([test_auc[idx] for idx, dd, _ in
                                        iter_cells(self.domain_num, self.task_num) if dd == d and idx in test_auc]))
                      for d in range(self.domain_num)
                      if any(dd == d and idx in test_auc for idx, dd, _ in iter_cells(self.domain_num, self.task_num))}
        overall_auc = float(np.mean(list(test_auc.values()))) if test_auc else float("nan")
        overall_ll = float(np.mean(list(test_ll.values()))) if test_ll else float("nan")
        worst_domain = min(domain_auc.values()) if domain_auc else float("nan")
        worst_cell = min(test_auc.values()) if test_auc else float("nan")
        grad_by_cell: Dict[int, Dict[str, object]] = {}
        target_epoch = fit_info.get("best_epoch", "")
        if target_epoch not in ("", None):
            for row in self.shared_gradient_epoch_rows:
                if int(row.get("epoch", -1)) == int(target_epoch):
                    grad_by_cell[int(row["cell"])] = row
        if not grad_by_cell and self.shared_gradient_epoch_rows:
            last = max(int(row["epoch"]) for row in self.shared_gradient_epoch_rows)
            grad_by_cell = {int(row["cell"]): row for row in self.shared_gradient_epoch_rows
                            if int(row["epoch"]) == last}
        rows: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            metric = selected.get(idx, {})
            oracle = self.oracle_metrics.get(idx, {})
            transfer = metric.get("transfer_auc", "")
            negative = int(_finite_number(transfer) and float(transfer) < 0)
            grad = grad_by_cell.get(idx, {})
            rows.append({
                "seed": self.seed, "setting": self.setting, "alpha": self.alpha,
                "alpha_key": self.alpha_key, "weighting": self.alpha_key,
                "domain": domain, "physical_domain": domain, "task": task, "cell": idx,
                "size_role": self.role_by_domain[domain], "domain_size": self.train_counts[domain],
                "domain_share": self.domain_shares[domain],
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
                "negative_transfer": negative,
                "shared_raw_gradient_norm": grad.get("raw_gradient_norm", ""),
                "shared_weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                "shared_magnitude_share": (self._selected_shared_magnitude(domain, task, target_epoch)),
                "shared_signed_contribution": grad.get("contribution_mean", ""),
                "shared_mismatch_index": grad.get("mismatch_index", ""),
                "shared_update_alignment": self._selected_update_alignment(domain, task, target_epoch),
                # Pairwise common support is assigned by the orchestrator only
                # after all five arms are available; a single run cannot claim
                # it merely because it observed one epoch.
                "common_support": 0,
                "censored": int(self.stopped_epoch is not None),
            })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), rows,
                       METRICS_SUMMARY_FIELDS)
        return rows

    def _selected_shared_magnitude(self, domain: int, task: int, epoch: object) -> object:
        if epoch in ("", None):
            return ""
        candidates = [row for row in self.shared_gradient_step_rows
                      if int(row["epoch"]) == int(epoch) and int(row["domain"]) == domain
                      and int(row["task"]) == task]
        return candidates[-1].get("magnitude_share", "") if candidates else ""

    def _selected_update_alignment(self, domain: int, task: int, epoch: object) -> object:
        if epoch in ("", None):
            return ""
        candidates = [row for row in self.shared_gradient_step_rows
                      if int(row["epoch"]) == int(epoch) and int(row["domain"]) == domain
                      and int(row["task"]) == task]
        if not candidates:
            return ""
        values = [float(row["actual_update_alignment"]) for row in candidates
                  if _finite_number(row.get("actual_update_alignment"))]
        return float(np.mean(values)) if values else float("nan")

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows,
                       METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_step.csv"),
                       self.shared_gradient_step_rows, E3_GRADIENT_STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "shared_gradient_epoch.csv"),
                       self.shared_gradient_epoch_rows, E3_GRADIENT_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       BATCH_FIELDS)
        summary_rows = self._write_metrics_summary(fit_info)
        from stage_E1_instrumentation import collect_nonfinite_inventory
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "shared_gradient_step.csv": self.shared_gradient_step_rows,
            "shared_gradient_epoch.csv": self.shared_gradient_epoch_rows,
            "batch_trace.csv": self.batch_rows,
            "metrics_summary.csv": summary_rows,
        })
        inventory = {
            "stage": "E3", "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(1 for item in self.nonfinite_inventory
                                      if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
            "scope_note": "raw non-finite values inventoried before CSV conversion",
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory)
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_checkpoint_complete, selected_checkpoint_cells = self._selected_checkpoint_complete()
        config = vars(self.args).copy()
        config.update({
            "stage": "E3", "seed": self.seed, "setting": self.setting,
            "alpha": self.alpha, "alpha_key": self.alpha_key, "weighting": self.alpha_key,
            "loss_weighting": self.alpha_key, "fix_optim_grouping": 1,
            "earlystop_patience": int(self.args.earlystop_patience), "backbone": "mdmoe",
            "scheduler": None, "task": "all", "domain": "all",
            "role_sizes_parsed": self.sizes, "domain_shares": self.domain_shares.tolist(),
            "train_domain_counts": self.train_counts, "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.equal_coefficients.detach().cpu().tolist(),
            "sample_coefficients": self.sample_coefficients.detach().cpu().tolist(),
            "cell_order": CELL_ORDER_NOTE, "instrumentation": bool(int(self.args.instrument)),
            "prohibited_mechanisms": ["curriculum", "momentum", "EMA", "adaptive_controller",
                                       "temporal_schedule", "new_router", "new_sampling"],
            "sampling": E3_FORMAL_PROTOCOL["sampling"],
            "dataset_identity": self.dataset_identity,
            "device_identity": self.device_identity,
            "split_row_fingerprints": self.split_row_fingerprints,
            "selected_checkpoint_complete": bool(selected_checkpoint_complete),
            "selected_checkpoint_cells": selected_checkpoint_cells,
        })
        config["formal_protocol"] = dict(E3_FORMAL_PROTOCOL)
        config["formal_valid"] = bool(
            config.get("epoch") == E3_FORMAL_PROTOCOL["epoch"]
            and config.get("batch_size") == E3_FORMAL_PROTOCOL["batch_size"]
            and config.get("learning_rate") == E3_FORMAL_PROTOCOL["learning_rate"]
            and config.get("learning_rate_darts") == E3_FORMAL_PROTOCOL["learning_rate_darts"]
            and config.get("weight_decay") == E3_FORMAL_PROTOCOL["weight_decay"]
            and config.get("expert_num") == E3_FORMAL_PROTOCOL["expert_num"]
            and config.get("probe_interval") == E3_FORMAL_PROTOCOL["probe_interval"]
            and config.get("instrument") == E3_FORMAL_PROTOCOL["instrument"]
            and config.get("role_sizes") == E3_FORMAL_PROTOCOL["role_sizes"]
            and config.get("fix_optim_grouping") == E3_FORMAL_PROTOCOL["fix_optim_grouping"]
            and config.get("earlystop_patience") == E3_FORMAL_PROTOCOL["earlystop_patience"]
            and config.get("setting") in ROLE_MAPS
            and self.alpha in ALPHA_GRID
            and self.train_counts == {d: int({"L": 150000, "M": 34500, "S": 3000}[self.role_by_domain[d]])
                                      for d in range(self.domain_num)}
            and selected_checkpoint_complete
            and os.path.exists(os.path.join(self.run_dir, "model.pth"))
            and self.dataset_identity == normalise_dataset_path(config.get("dataset_path", ""))
        )
        config["config_hash"] = _hash_json(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E3", "status": "OK", "seed": self.seed, "setting": self.setting,
            "alpha": self.alpha, "alpha_key": self.alpha_key, "weighting": self.alpha_key,
            "formal_run_key": f"{self.alpha_key}|{self.setting}|{self.seed}",
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
            "equal_coefficients": self.equal_coefficients.detach().cpu().tolist(),
            "sample_coefficients": self.sample_coefficients.detach().cpu().tolist(),
            "observed_epochs": list(self.observed_epochs), "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": (self.stopped_epoch + 1 if self.stopped_epoch is not None else None),
            "formal_protocol": dict(E3_FORMAL_PROTOCOL), "formal_valid": bool(config["formal_valid"]),
            "selected_checkpoint_complete": bool(selected_checkpoint_complete),
            "selected_checkpoint_cells": selected_checkpoint_cells,
            "split_row_fingerprints": self.split_row_fingerprints,
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.arch_batch_fingerprints,
            "shared_gradient_step_count": len(self.shared_gradient_step_rows),
            "shared_gradient_epoch_count": len(self.shared_gradient_epoch_rows),
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory["unexplained_count"],
            "block_status": "PENDING", "block_valid": None,
            "wall_seconds": float(wall_seconds),
            "scope_note": "raw outputs only; no scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)

    def close(self) -> None:
        self.routing.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E3 one-run alpha instrumentation")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--setting", choices=sorted(ROLE_MAPS), default="d0_large")
    parser.add_argument("--alpha", type=float, choices=list(ALPHA_GRID), default=0.0)
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
    parser.add_argument("--legacy_oracle_root", default=os.path.join(HERE, "P0_2_ratio_sweep"))
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
    if args.probe_interval <= 0:
        raise ValueError("--probe_interval must be positive")
    started = time.time()
    runner = E3Runner(args)
    try:
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        with preserve_rng_state():
            final_test = _metric_records(runner.model, runner.test_eval_loader,
                                         runner.device, runner.domain_num, runner.task_num)
        print("[stage_E3] alpha:", runner.alpha, "key:", runner.alpha_key, flush=True)
        print("[stage_E3] FINAL per-(d,t) test auc:",
              [final_test[idx]["auc"] for idx, _, _ in iter_cells(runner.domain_num, runner.task_num)],
              flush=True)
        print("[stage_E3] raw outputs:", runner.run_dir, flush=True)
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
