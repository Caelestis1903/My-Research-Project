"""Single-run implementation for Stage E1.

This runner keeps the existing MovieLens/MDMTRec data and model path but
owns the E1 observation loop.  The only training arm switch is
``--weighting {equal,sample}``; role rotation and seed are run-level factors.
The normal weighted loss backward and Adam steps remain the source of truth.
Observer gradients are collected with ``autograd.grad`` before that backward
and never copied into ``Parameter.grad``.

The module is intentionally importable without a dataset.  Unit tests use
``build_argparser`` and the small helper functions, while the formal runner
invokes this file once per setting/weighting/seed.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    FORMAL_PROTOCOL,
    GROUP_NAMES,
    GradientEpochAccumulator,
    RoutingObserver,
    batch_fingerprint,
    build_parameter_partition,
    collect_cell_gradients,
    collect_nonfinite_inventory,
    contribution_views,
    iter_cells,
    make_gradient_step_rows,
    masked_bce_terms,
    parameter_state_hash,
    preserve_rng_state,
    rng_state_digest,
    relative_l2_error,
    safe_auc,
    safe_logloss,
    weighting_coefficients,
    write_json,
    write_rows_csv,
)


ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}

METRICS_FIELDS = [
    "seed", "epoch", "domain", "task", "cell", "weighting", "setting",
    "physical_domain", "size_role", "domain_size", "domain_share",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate",
    "train_auc", "train_logloss", "validation_auc", "validation_logloss",
    "test_auc", "test_logloss", "selected_checkpoint", "stopped_censored",
    "oracle_auc", "oracle_logloss", "oracle_source", "transfer_auc",
    "transfer_logloss",
]

GRADIENT_STEP_FIELDS = [
    "epoch", "step", "domain", "task", "cell", "parameter_group", "weighting",
    "batch_fingerprint", "cell_sample_count", "loss_coefficient",
    "raw_gradient_norm", "raw_gradient_rms", "weighted_gradient_norm",
    "weighted_gradient_rms", "effective_projection", "normalized_contribution",
    "domain_signed_contribution",
    "magnitude_share", "overrepresentation_ratio", "mismatch_index",
    "equal_cf_projection", "equal_cf_contribution", "sample_cf_projection",
    "sample_cf_contribution", "equal_cf_magnitude_share",
    "sample_cf_magnitude_share", "unused_param_count", "actual_update_norm",
    "actual_update_alignment", "actual_update_projection",
    "gradient_reconstruction_error",
]

GRADIENT_EPOCH_FIELDS = [
    "seed", "epoch", "domain", "task", "cell", "weighting", "parameter_group",
    "gradient_norm", "weighted_gradient_norm", "gradient_variance",
    "weighted_gradient_variance", "gradient_SNR", "norm_of_mean_gradient",
    "probe_count", "contribution_mean", "contribution_std", "mismatch_index",
    "variance_scope",
]

COSINE_FIELDS = ["seed", "epoch", "weighting", "setting", "domain_pair",
                 "cosine_mean", "cosine_std", "probe_count"]

GATE_FIELDS_BASE = ["seed", "weighting", "setting", "epoch", "domain", "task",
                    "cell", "routing_scope", "routing_sample_count", "gate_entropy",
                    "effective_expert_count", "top1_expert_frequency"]

BATCH_FIELDS = ["seed", "epoch", "step", "kind", "batch_fingerprint", "batch_size",
                "domain_counts", "task_counts", "is_probe", "train_loss"]

SUMMARY_FIELDS = [
    "seed", "epoch", "domain", "task", "cell", "weighting", "setting",
    "size_role", "domain_size", "domain_share", "parameter_group", "AUC",
    "LogLoss", "train_auc", "train_logloss", "validation_auc",
    "validation_logloss", "test_auc", "test_logloss", "gradient_norm",
    "weighted_gradient_norm", "gradient_variance", "gradient_SNR",
    "norm_of_mean_gradient", "probe_count", "contribution_mean",
    "contribution_std", "mismatch_index", "variance_scope", "oracle_auc",
    "oracle_logloss", "oracle_source", "transfer_auc", "transfer_logloss",
]


def setup_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    import random
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True


def parse_role_sizes(value: str) -> Dict[str, int]:
    parts = [int(item.strip()) for item in str(value).split(",")]
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise ValueError("--role_sizes must be three positive integers: L,M,S")
    return dict(zip(("L", "M", "S"), parts))


def role_reversal_sample(x_train: np.ndarray, y_train: np.ndarray, setting: str,
                         seed: int, sizes: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Exact P0-4 role-reversal sampling rule, kept local and auditable."""

    if setting not in ROLE_MAPS:
        raise ValueError("unknown role setting {!r}".format(setting))
    roles = ROLE_MAPS[setting]
    domain_ids = x_train[:, -1].astype(int)
    keep = np.zeros(len(x_train), dtype=bool)
    for domain, role in roles.items():
        pool = np.where(domain_ids == domain)[0]
        requested = int(sizes[role])
        if requested > len(pool):
            raise ValueError("role size {} exceeds physical domain {} pool {}".format(
                requested, domain, len(pool)))
        permutation = np.random.RandomState(int(seed) * 10 + domain).permutation(len(pool))
        keep[pool[permutation[:requested]]] = True
    return x_train[keep], y_train[keep]


def _positive_rate(values: np.ndarray, task: int) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.asarray(values[:, task], dtype=np.float64).mean())


def _parameter_vector(named: Mapping[str, torch.nn.Parameter], names: Sequence[str]) -> torch.Tensor:
    if not names:
        return torch.zeros(0)
    return torch.cat([named[name].detach().reshape(-1).clone() for name in names])


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _metric_records(model: torch.nn.Module, data_loader: DataLoader, device: torch.device,
                    domain_num: int, task_num: int) -> Dict[int, Dict[str, object]]:
    """Evaluate exact per-cell AUC/LogLoss without changing RNG state."""

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
    result: Dict[int, Dict[str, object]] = {}
    for idx, _, _ in iter_cells(domain_num, task_num):
        result[idx] = {
            "sample_count": counts[idx],
            "positive_rate": (float(np.mean(truth[idx])) if truth[idx] else float("nan")),
            "auc": safe_auc(truth[idx], prediction[idx]),
            "logloss": safe_logloss(truth[idx], prediction[idx]),
        }
    return result


def load_p02_oracle_metrics(oracle_root: str, setting: str, seed: int,
                            domain_num: int = 3, task_num: int = 2) -> Dict[int, Dict[str, object]]:
    """Read existing P0-2 ratio-50 oracle rows, if present.

    E1 never launches or reconstructs an oracle.  Missing artifacts return
    explicit ``source=missing`` records so downstream analysis cannot mistake
    an unavailable baseline for a zero-valued baseline.
    """

    result = {idx: {"auc": "", "logloss": "", "source": "missing"}
              for idx, _, _ in iter_cells(domain_num, task_num)}
    root = os.path.abspath(oracle_root) if oracle_root else ""
    for domain in range(domain_num):
        path = os.path.join(root, "ratio_50", setting, f"oracle_d{domain}", f"seed{seed}",
                            "metrics_summary.csv")
        rows = []
        if os.path.exists(path):
            try:
                with open(path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, csv.Error):
                rows = []
        if not rows:
            continue
        row = rows[0]
        for task in range(task_num):
            # P0-2's oracle writer uses the global physical-domain column;
            # tolerate a single-domain writer as a read-only fallback.
            auc = row.get(f"auc_d{domain}t{task}", "")
            logloss = row.get(f"log_d{domain}t{task}", "")
            if domain == 0:
                auc = auc or row.get(f"auc_d0t{task}", "")
                logloss = logloss or row.get(f"log_d0t{task}", "")
            idx = task * domain_num + domain
            if auc not in (None, "") or logloss not in (None, ""):
                result[idx] = {"auc": auc or "", "logloss": logloss or "",
                               "source": "P0-2_ratio_50_oracle"}
    return result


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class E1Runner:
    """One fixed setting/seed/weighting run and its raw diagnostic outputs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = int(args.seed)
        self.setting = str(args.setting)
        self.weighting = str(args.weighting)
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
        domain_ids = self.x_train[:, -1].astype(int)
        self.train_counts = {d: int((domain_ids == d).sum()) for d in range(self.domain_num)}
        total = float(sum(self.train_counts.values()))
        self.domain_shares = np.asarray([self.train_counts[d] / total for d in range(self.domain_num)],
                                        dtype=np.float64)
        self.role_by_domain = ROLE_MAPS[self.setting]
        self.oracle_metrics = load_p02_oracle_metrics(
            self.args.oracle_root, self.setting, self.seed,
            self.domain_num, self.task_num)
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
            self.weighting, self.domain_shares, self.task_num, device=self.device)
        self.equal_coefficients = weighting_coefficients(
            "equal", self.domain_shares, self.task_num, device=self.device)
        self.sample_coefficients = weighting_coefficients(
            "sample", self.domain_shares, self.task_num, device=self.device)

    def _build_loaders(self) -> None:
        data_generator = DataGenerator(self.x_train, self.y_train)
        self.train_loader, self.val_loader, self.test_loader = data_generator.generate_dataloader(
            x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test,
            batch_size=int(self.args.batch_size), num_workers=0)
        # Explicit non-shuffling loader is used for the per-epoch test bypass.
        self.test_eval_loader = DataLoader(
            TorchDataset(self.x_test, self.y_test), batch_size=int(self.args.batch_size),
            shuffle=False, num_workers=0)

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
        self.routing = RoutingObserver(
            self.model, self.domain_num, self.task_num,
            enabled=bool(int(self.args.instrument)),
        )

    def _init_outputs(self) -> None:
        self.metrics_rows: List[Dict[str, object]] = []
        self.gradient_step_rows: List[Dict[str, object]] = []
        self.gradient_epoch_rows: List[Dict[str, object]] = []
        self.cosine_rows: List[Dict[str, object]] = []
        self.gate_rows: List[Dict[str, object]] = []
        self.batch_rows: List[Dict[str, object]] = []
        self.epoch_accumulators: Dict[int, GradientEpochAccumulator] = {}
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: Optional[int] = None
        self.best_val_auc = float("-inf")
        self.stopped_epoch: Optional[int] = None
        self.observed_epochs: List[int] = []
        self.arch_batch_fingerprints: List[str] = []
        self.batch_fingerprints: Dict[int, List[str]] = {}
        self.nonfinite_inventory: List[Dict[str, object]] = []
        self.rng_restore_evidence: List[Dict[str, object]] = []

    def _record_batch(self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
                      y_batch: torch.Tensor, counts: Sequence[int], is_probe: bool,
                      train_loss: Optional[object] = None) -> str:
        fp = batch_fingerprint(x_batch, y_batch)
        task_counts = []
        for task in range(self.task_num):
            task_counts.append(int(y_batch[:, task].numel()))
        self.batch_rows.append({
            "seed": self.seed, "epoch": int(epoch), "step": int(step), "kind": kind,
            "batch_fingerprint": fp, "batch_size": int(len(y_batch)),
            "domain_counts": _json_counts(counts), "task_counts": _json_counts(task_counts),
            "is_probe": int(bool(is_probe)),
            "train_loss": ("" if train_loss is None
                           else float(train_loss.detach().cpu()
                                      if isinstance(train_loss, torch.Tensor)
                                      else train_loss)),
        })
        return fp

    def _raw_by_group(self, raw_cells: Mapping[int, Mapping[str, torch.Tensor]]) -> Dict[str, Dict[int, torch.Tensor]]:
        return {group: {idx: raw_cells[idx][group]
                        for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
                for group in GROUP_NAMES}

    def _unused_by_group(self, unused_cells: Mapping[int, Mapping[str, int]]) -> Dict[str, Dict[int, int]]:
        return {group: {idx: int(unused_cells[idx][group])
                        for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
                for group in GROUP_NAMES}

    def _vectors_by_group(self, named: Mapping[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
        return {group: _parameter_vector(named, self.partition.names(group))
                for group in GROUP_NAMES}

    def _cosine_epoch_values(self, raw_by_group: Mapping[str, Mapping[int, torch.Tensor]]) -> List[Tuple[str, float]]:
        shared = raw_by_group["shared"]
        domain_vectors = {
            d: sum((shared[cell] for task in range(self.task_num)
                    for cell in [self._cell(task, d)]), torch.zeros_like(next(iter(shared.values()))))
            for d in range(self.domain_num)
        }
        values = []
        for left in range(self.domain_num):
            for right in range(left + 1, self.domain_num):
                left_vec, right_vec = domain_vectors[left], domain_vectors[right]
                denom = torch.linalg.vector_norm(left_vec) * torch.linalg.vector_norm(right_vec)
                if float(denom.detach().cpu()) <= 1.0e-12:
                    value = float("nan")
                else:
                    value = float(torch.clamp(torch.dot(left_vec, right_vec) / denom,
                                              -1.0, 1.0).detach().cpu())
                values.append((f"{left}-{right}", value))
        return values

    def _cell(self, task: int, domain: int) -> int:
        return int(task) * self.domain_num + int(domain)

    def _architecture_update(self, epoch: int, arch_iter) -> object:
        try:
            x_arch, y_arch = next(arch_iter)
        except StopIteration:
            arch_iter = iter(self.train_loader)
            x_arch, y_arch = next(arch_iter)
        domain_cpu = x_arch[:, -1].detach().cpu().numpy().astype(int)
        arch_counts = [int((domain_cpu == domain).sum()) for _, domain, _ in
                       iter_cells(self.domain_num, self.task_num)]
        arch_fingerprint = self._record_batch(
            epoch, -1, "architecture", x_arch, y_arch, arch_counts, False)
        self.arch_batch_fingerprints.append(arch_fingerprint)
        if self.arch_optimizer is None:
            return arch_iter
        x_device = x_arch.transpose(0, 1).to(self.device)
        y_device = y_arch.to(self.device)
        # MDMTRec's legacy forward allocates its final result tensor on CPU;
        # preserve the existing model while making the trainer-side loss
        # device explicit (the normal train/eval paths already use this cast).
        pred = self.model(x_device).to(self.device)
        # This is the existing architecture-parameter path: unmasked batch
        # BCE and the separate _weight_* optimizer.  It is identical in both
        # E1 arms and is not part of the shared contribution analysis.
        arch_loss = self.criterion(pred, y_device.float())
        self.batch_rows[-1]["train_loss"] = float(arch_loss.detach().cpu())
        self.model.zero_grad(set_to_none=True)
        arch_loss.backward()
        self.arch_optimizer.step()
        return arch_iter

    def _append_metric_rows(self, epoch: int, train_stats: Mapping[int, Mapping[str, object]],
                            val_stats: Mapping[int, Mapping[str, object]],
                            test_stats: Mapping[int, Mapping[str, object]],
                            selected: bool, censored: bool = False) -> None:
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            train = train_stats.get(idx, {})
            val = val_stats.get(idx, {})
            test = test_stats.get(idx, {})
            row = {
                "seed": self.seed, "epoch": int(epoch), "domain": domain, "task": task,
                "cell": idx, "weighting": self.weighting, "setting": self.setting,
                "physical_domain": domain, "size_role": self.role_by_domain[domain],
                "domain_size": self.train_counts[domain],
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
                "selected_checkpoint": int(bool(selected)),
                "stopped_censored": int(bool(censored)),
                "oracle_auc": "", "oracle_logloss": "", "oracle_source": "missing",
                "transfer_auc": "", "transfer_logloss": "",
            }
            oracle = self.oracle_metrics.get(idx, {})
            row["oracle_auc"] = oracle.get("auc", "")
            row["oracle_logloss"] = oracle.get("logloss", "")
            row["oracle_source"] = oracle.get("source", "missing")
            # Transfer is deliberately a data field, not an interpreted
            # result: it is the direct current-test-minus-oracle difference
            # only when a matching oracle artifact exists.
            if _finite_number(row["oracle_auc"]) and _finite_number(row["test_auc"]):
                row["transfer_auc"] = float(row["test_auc"]) - float(row["oracle_auc"])
            if _finite_number(row["oracle_logloss"]) and _finite_number(row["test_logloss"]):
                row["transfer_logloss"] = float(row["test_logloss"]) - float(row["oracle_logloss"])
            self.metrics_rows.append(row)

    def _train_epoch(self, epoch: int) -> Tuple[Dict[int, Dict[str, object]], Dict[int, List[str]]]:
        self.model.train()
        if bool(int(self.args.instrument)):
            self.routing.start_epoch(epoch)
        train_truth: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_prediction: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_loss_sum = {idx: 0.0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        train_counts = {idx: 0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        accumulator = GradientEpochAccumulator(self.partition, self.domain_num, self.task_num)
        self.epoch_accumulators[epoch] = accumulator
        fingerprints: List[str] = []
        cosine_values: Dict[str, List[float]] = {
            f"{left}-{right}": [] for left in range(self.domain_num) for right in range(left + 1, self.domain_num)
        }
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            domain_cpu = x_batch[:, -1].detach().cpu().numpy().astype(int)
            cell_counts = [int((domain_cpu == domain).sum()) for _, domain, _ in
                           iter_cells(self.domain_num, self.task_num)]
            is_probe = bool(int(self.args.instrument)) and (step % int(self.args.probe_interval) == 0)
            fingerprint = self._record_batch(epoch, step, "train", x_batch, y_batch,
                                             cell_counts, is_probe)
            fingerprints.append(fingerprint)
            x_device = x_batch.transpose(0, 1).to(self.device)
            y_device = y_batch.to(self.device)
            domain_ids = x_device[-1, :].detach()
            if bool(int(self.args.instrument)):
                self.routing.start_batch(domain_ids)
            y_pred = self.model(x_device).to(self.device)
            if bool(int(self.args.instrument)):
                self.routing.stop_batch()
            terms, counts = masked_bce_terms(
                y_pred, y_device, domain_ids, self.domain_num, self.task_num, self.criterion)
            for idx, _, _ in iter_cells(self.domain_num, self.task_num):
                if counts[idx] > 0:
                    train_loss_sum[idx] += float(terms[idx].detach().cpu()) * counts[idx]
                    train_counts[idx] += counts[idx]
                    mask = domain_ids == (idx % self.domain_num)
                    train_truth[idx].extend(y_device[mask, idx // self.domain_num].detach().cpu().tolist())
                    train_prediction[idx].extend(y_pred[mask, idx // self.domain_num].detach().cpu().tolist())
            loss = (torch.stack(terms) * self.actual_coefficients).sum()
            # Keep the exact scalar consumed by the real backward path.  The
            # paired comparator uses this field to enforce the <=1e-7
            # instrumentation ON/OFF and equal/sample trace contract.
            self.batch_rows[-1]["train_loss"] = float(loss.detach().cpu())
            if is_probe:
                raw_cells, unused_cells = collect_cell_gradients(
                    terms, self.model, self.partition, retain_graph=True)
                raw_by_group = self._raw_by_group(raw_cells)
                unused_by_group = self._unused_by_group(unused_cells)
                views = {
                    "actual": {group: contribution_views(
                        raw_by_group[group], self.actual_coefficients,
                        self.domain_num, self.task_num) for group in GROUP_NAMES},
                    "equal": {group: contribution_views(
                        raw_by_group[group], self.equal_coefficients,
                        self.domain_num, self.task_num) for group in GROUP_NAMES},
                    "sample": {group: contribution_views(
                        raw_by_group[group], self.sample_coefficients,
                        self.domain_num, self.task_num) for group in GROUP_NAMES},
                }
                normalized = {}
                mismatches = {}
                for group in GROUP_NAMES:
                    normalized[group] = views["actual"][group]["normalized"]
                    magnitude = views["actual"][group]["magnitude"]
                    domain_magnitude = {
                        domain: sum(magnitude[self._cell(task, domain)]
                                    for task in range(self.task_num))
                        for domain in range(self.domain_num)
                    }
                    mismatches[group] = 0.5 * sum(
                        abs(domain_magnitude[d] - float(self.domain_shares[d]))
                        for d in range(self.domain_num))
                accumulator.update(raw_by_group, normalized, mismatches,
                                   self.actual_coefficients)
                for pair, value in self._cosine_epoch_values(raw_by_group):
                    cosine_values[pair].append(value)
                named = dict(self.model.named_parameters())
                combined_by_group = {
                    group: views["actual"][group]["combined"] for group in GROUP_NAMES}
                before_vectors = self._vectors_by_group(named)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                reconstruction = {}
                actual_grad_vectors = {}
                for group in GROUP_NAMES:
                    actual_grad, _ = self._flatten_current_gradients(group)
                    actual_grad_vectors[group] = actual_grad
                    reconstruction[group] = relative_l2_error(actual_grad, combined_by_group[group])
                update_before = before_vectors
                self.optimizer.step()
                after_vectors = self._vectors_by_group(dict(self.model.named_parameters()))
                update = {group: update_before[group] - after_vectors[group] for group in GROUP_NAMES}
                self.gradient_step_rows.extend(make_gradient_step_rows(
                    raw_by_group, unused_by_group,
                    actual_coefficients=self.actual_coefficients,
                    equal_coefficients=self.equal_coefficients,
                    sample_coefficients=self.sample_coefficients,
                    domain_num=self.domain_num, task_num=self.task_num,
                    partition=self.partition, domain_shares=self.domain_shares,
                    weighting=self.weighting, epoch=epoch, step=step,
                    batch_fingerprint=fingerprint, cell_sample_counts=counts,
                    actual_update_by_group=update,
                    reconstruction_error_by_group=reconstruction,
                ))
            else:
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        train_stats = {}
        for idx, _, _ in iter_cells(self.domain_num, self.task_num):
            train_stats[idx] = {
                "sample_count": train_counts[idx],
                "positive_rate": (float(np.mean(train_truth[idx])) if train_truth[idx] else float("nan")),
                "auc": safe_auc(train_truth[idx], train_prediction[idx]),
                "logloss": (train_loss_sum[idx] / train_counts[idx]
                            if train_counts[idx] else float("nan")),
            }
        if bool(int(self.args.instrument)):
            self.gate_rows.extend(self.routing.rows(
                seed=self.seed, weighting=self.weighting, setting=self.setting, epoch=epoch))
            for pair, values in cosine_values.items():
                finite = [value for value in values if np.isfinite(value)]
                self.cosine_rows.append({
                    "seed": self.seed, "epoch": epoch, "weighting": self.weighting,
                    "setting": self.setting, "domain_pair": pair,
                    "cosine_mean": float(np.mean(finite)) if finite else float("nan"),
                    "cosine_std": float(np.std(finite)) if finite else float("nan"),
                    "probe_count": len(finite),
                })
            self.gradient_epoch_rows.extend(accumulator.rows(
                epoch=epoch, seed=self.seed, weighting=self.weighting,
                actual_coefficients=self.actual_coefficients))
        return train_stats, {epoch: fingerprints}

    def _flatten_current_gradients(self, group: str) -> Tuple[torch.Tensor, int]:
        named = dict(self.model.named_parameters())
        grads = [named[name].grad for name in self.partition.names(group)]
        from stage_E1_instrumentation import flatten_parameter_tensors
        return flatten_parameter_tensors(named, self.partition.names(group), grads,
                                         device=self.device)

    def fit(self) -> Dict[str, object]:
        # Constructing the iterator mirrors the existing architecture-update
        # path and is done before the first training iterator in both arms.
        arch_iter = iter(self.train_loader)
        patience = 10
        bad_epochs = 0
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            train_stats, fp_map = self._train_epoch(epoch)
            self.batch_fingerprints[epoch] = fp_map[epoch]
            arch_iter = self._architecture_update(epoch, arch_iter)
            # Validation is part of the P0-4 cadence and its DataLoader
            # iterator consumption must remain in the normal RNG stream.
            # Only the extra test evaluation is rolled back.
            val_stats = _metric_records(self.model, self.val_loader, self.device,
                                        self.domain_num, self.task_num)
            rng_before = rng_state_digest()
            with preserve_rng_state():
                test_stats = _metric_records(self.model, self.test_eval_loader, self.device,
                                             self.domain_num, self.task_num)
            rng_after = rng_state_digest()
            self.rng_restore_evidence.append({
                "epoch": int(epoch), "scope": "test_eval",
                "before": rng_before, "after": rng_after,
                "restored": bool(rng_before == rng_after),
            })
            val_values = [float(val_stats[idx]["auc"]) for idx, _, _ in
                          iter_cells(self.domain_num, self.task_num)
                          if np.isfinite(val_stats[idx]["auc"])]
            val_mean = float(np.mean(val_values)) if val_values else float("nan")
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
            # Explicitly retain the censoring boundary.  No metric is
            # forward-filled for epochs after early stopping.
            for censored_epoch in range(self.stopped_epoch + 1, int(self.args.epoch)):
                self._append_metric_rows(censored_epoch, {}, {}, {}, False, censored=True)
        return {
            "best_epoch": self.best_epoch if self.best_epoch is not None else "",
            "stopped_epoch": self.stopped_epoch if self.stopped_epoch is not None else "",
            "best_val_auc": self.best_val_auc if np.isfinite(self.best_val_auc) else "",
            "observed_epochs": list(self.observed_epochs),
        }

    def _write_metrics_summary(self, fit_info: Mapping[str, object]) -> List[Dict[str, object]]:
        best_rows = {int(row["cell"]): row for row in self.metrics_rows
                     if int(row["selected_checkpoint"]) == 1}
        rows = []
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            row = best_rows.get(idx, {})
            rows.append({
                "seed": self.seed, "weighting": self.weighting, "setting": self.setting,
                "domain": domain, "task": task, "cell": idx,
                "best_epoch": fit_info.get("best_epoch", ""),
                "stopped_epoch": fit_info.get("stopped_epoch", ""),
                "best_val_auc": row.get("validation_auc", ""),
                "best_val_logloss": row.get("validation_logloss", ""),
                "best_train_auc": row.get("train_auc", ""),
                "best_train_logloss": row.get("train_logloss", ""),
                "best_test_auc": row.get("test_auc", ""),
                "best_test_logloss": row.get("test_logloss", ""),
                "domain_train_samples": self.train_counts[domain],
                "domain_share": self.domain_shares[domain],
                "size_role": self.role_by_domain[domain],
                "oracle_source": row.get("oracle_source", "missing"),
            })
        fields = ["seed", "weighting", "setting", "domain", "task", "cell", "best_epoch",
                  "stopped_epoch", "best_val_auc", "best_val_logloss", "best_train_auc",
                  "best_train_logloss", "best_test_auc", "best_test_logloss",
                  "domain_train_samples", "domain_share", "size_role", "oracle_source"]
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), rows, fields)
        return rows

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows,
                       METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"), self.gradient_step_rows,
                       GRADIENT_STEP_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "gradient_epoch.csv"), self.gradient_epoch_rows,
                       GRADIENT_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "gradient_cosine_epoch.csv"), self.cosine_rows,
                       COSINE_FIELDS)
        gate_fields = list(GATE_FIELDS_BASE)
        max_experts = int(getattr(self.model, "expert_num", 0))
        gate_fields.extend(f"expert_{i}_mean_probability" for i in range(max_experts))
        write_rows_csv(os.path.join(self.run_dir, "gate_epoch.csv"), self.gate_rows, gate_fields)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       BATCH_FIELDS)
        summary_rows = self._write_metrics_summary(fit_info)
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "gradient_step.csv": self.gradient_step_rows,
            "gradient_epoch.csv": self.gradient_epoch_rows,
            "gradient_cosine_epoch.csv": self.cosine_rows,
            "gate_epoch.csv": self.gate_rows,
            "batch_trace.csv": self.batch_rows,
            "metrics_summary.csv": summary_rows,
        })
        inventory_payload = {
            "stage": "E1",
            "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(
                1 for item in self.nonfinite_inventory
                if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
            "scope_note": "every NaN/Inf is inventoried before CSV conversion; blanks are not non-finite values",
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory_payload)
        model_path = os.path.join(self.run_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        config = vars(self.args).copy()
        config.update({
            "seed": self.seed, "setting": self.setting, "weighting": self.weighting,
            "loss_weighting": self.weighting, "fix_optim_grouping": 1,
            "earlystop_patience": 10, "backbone": "mdmoe", "scheduler": None,
            "task": "all", "domain": "all",
            "role_sizes_parsed": self.sizes, "domain_shares": self.domain_shares.tolist(),
            "train_domain_counts": self.train_counts, "cell_order": CELL_ORDER_NOTE,
            "prohibited_mechanisms": ["curriculum", "momentum", "EMA", "new_router", "new_sampling"],
            "instrumentation": bool(int(self.args.instrument)),
        })
        config["formal_protocol"] = dict(FORMAL_PROTOCOL)
        config["formal_valid"] = bool(
            all(config.get(field) == expected
                for field, expected in FORMAL_PROTOCOL.items())
            and config.get("loss_weighting") == config.get("weighting")
            and config.get("instrumentation") is True
        )
        config["config_hash"] = _hash_json(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E1",
            "status": "OK",
            "seed": self.seed,
            "setting": self.setting,
            "weighting": self.weighting,
            "formal_run_key": f"{self.weighting}|{self.setting}|{self.seed}",
            "dataset": "MovieLens-1M",
            "domain_num": self.domain_num,
            "task_num": self.task_num,
            "role_by_domain": self.role_by_domain,
            "role_sizes": self.sizes,
            "train_domain_counts": self.train_counts,
            "domain_shares": self.domain_shares.tolist(),
            "split_counts": self.split_counts,
            "positive_rates": self.positive_rates,
            "oracle_metrics": self.oracle_metrics,
            "partition": self.partition.as_dict(),
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "actual_coefficients": self.actual_coefficients.detach().cpu().tolist(),
            "equal_coefficients": self.equal_coefficients.detach().cpu().tolist(),
            "sample_coefficients": self.sample_coefficients.detach().cpu().tolist(),
            "observed_epochs": list(self.observed_epochs),
            "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": (self.stopped_epoch + 1
                                     if self.stopped_epoch is not None else None),
            "formal_protocol": dict(FORMAL_PROTOCOL),
            "formal_valid": bool(config["formal_valid"]),
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.arch_batch_fingerprints,
            "gradient_step_count": len(self.gradient_step_rows),
            "gradient_epoch_count": len(self.gradient_epoch_rows),
            "routing_row_count": len(self.gate_rows),
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory_payload["unexplained_count"],
            "pair_status": "PENDING",
            "pair_valid": None,
            "pair_comparison": {},
            "wall_seconds": float(wall_seconds),
            "scope_note": "raw outputs only; no scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)

    def close(self) -> None:
        self.routing.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E1 one-run instrumentation")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--setting", choices=sorted(ROLE_MAPS), default="d0_large")
    parser.add_argument("--weighting", choices=["equal", "sample"], default="equal")
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "P0_2_ratio_sweep"),
                        help="read-only existing P0-2 ratio-50 oracle root; missing is explicit")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--instrument", type=int, choices=[0, 1], default=1)
    parser.add_argument("--probe_interval", type=int, default=1)
    parser.add_argument("--stdout_log", default=None,
                        help="optional tee path for direct invocation; orchestrator captures stdout")
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
    runner = E1Runner(args)
    try:
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        with preserve_rng_state():
            final_test = _metric_records(runner.model, runner.test_eval_loader, runner.device,
                                          runner.domain_num, runner.task_num)
        aucs = [final_test[idx]["auc"] for idx, _, _ in iter_cells(runner.domain_num, runner.task_num)]
        logloss = [final_test[idx]["logloss"] for idx, _, _ in iter_cells(runner.domain_num, runner.task_num)]
        print("[stage_E1] FINAL per-(d,t) test auc:", aucs, flush=True)
        print("[stage_E1] FINAL per-(d,t) test logloss:", logloss, flush=True)
        print("[stage_E1] best_epoch:", fit_info.get("best_epoch", ""),
              "stopped_epoch:", fit_info.get("stopped_epoch", ""), flush=True)
        print("[stage_E1] raw outputs:", runner.run_dir, flush=True)
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
