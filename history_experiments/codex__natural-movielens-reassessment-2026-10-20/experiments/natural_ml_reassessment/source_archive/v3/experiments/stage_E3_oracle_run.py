"""Frozen single-domain oracle reference runner for Stage E3.

This entry point is used only when a complete seed-level P0-2 oracle artifact
cannot be verified.  It trains the original MDMTRec on one physical domain,
while retaining that physical-domain identity in the manifest and summary.
There is no alpha parameter and no sample/objective reweighting in this path.
"""

from __future__ import annotations

import argparse
import csv
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
    build_parameter_partition,
    parameter_state_hash,
    preserve_rng_state,
    safe_auc,
    safe_logloss,
    write_json,
    write_rows_csv,
)
from stage_E3_weighting import ROLE_MAPS, E3_FORMAL_PROTOCOL, iter_cells
from stage_E3_weighting import (
    normalise_dataset_path,
    split_row_fingerprint,
    strict_validation_macro_auc,
)


METRICS_FIELDS = [
    "seed", "setting", "original_physical_domain", "role", "epoch", "domain", "task",
    "cell", "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_auc", "train_logloss", "validation_auc", "validation_logloss", "test_auc",
    "test_logloss", "selected_checkpoint", "stopped_censored",
]

SUMMARY_FIELDS = [
    "seed", "setting", "original_physical_domain", "role", "task", "domain", "cell",
    "train_count", "best_epoch", "stopped_epoch", "best_val_auc", "best_val_logloss",
    "best_test_auc", "best_test_logloss", "oracle_auc", "oracle_logloss",
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
    domain_ids = x_train[:, -1].astype(int)
    keep = np.zeros(len(x_train), dtype=bool)
    for domain, role in ROLE_MAPS[setting].items():
        pool = np.where(domain_ids == domain)[0]
        requested = int(sizes[role])
        if requested > len(pool):
            raise ValueError("role size exceeds physical-domain pool")
        permutation = np.random.RandomState(int(seed) * 10 + domain).permutation(len(pool))
        keep[pool[permutation[:requested]]] = True
    return x_train[keep], y_train[keep]


def _positive_rate(values: np.ndarray, task: int) -> float:
    return float(np.asarray(values[:, task], dtype=np.float64).mean()) if len(values) else float("nan")


def _metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[int, Dict[str, object]]:
    model.eval()
    truth = {0: [], 1: []}
    pred = {0: [], 1: []}
    for x_batch, y_batch in loader:
        x_device = x_batch.transpose(0, 1).to(device)
        y_device = y_batch.to(device)
        with torch.no_grad():
            output = model(x_device).to(device)
        for task in (0, 1):
            truth[task].extend(y_device[:, task].detach().cpu().tolist())
            pred[task].extend(output[:, task].detach().cpu().tolist())
    return {task: {"sample_count": len(truth[task]),
                   "positive_rate": float(np.mean(truth[task])) if truth[task] else float("nan"),
                   "auc": safe_auc(truth[task], pred[task]),
                   "logloss": safe_logloss(truth[task], pred[task])}
            for task in (0, 1)}


def oracle_main_loss(
    output: torch.Tensor, target: torch.Tensor, criterion: torch.nn.Module,
) -> torch.Tensor:
    """Frozen P0-4 oracle objective: two task BCE means summed, never averaged."""

    if output.ndim != 2 or target.ndim != 2 or output.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("oracle_main_loss expects [batch, 2] output and target tensors")
    return criterion(output[:, 0], target[:, 0]) + criterion(output[:, 1], target[:, 1])


def _hash_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _int_value(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


class OracleRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = int(args.seed)
        self.setting = str(args.setting)
        self.domain = int(args.domain)
        self.role_sizes = parse_role_sizes(args.role_sizes)
        self.role = ROLE_MAPS[self.setting][self.domain]
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device but CUDA is unavailable")
        setup_seed(self.seed)
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self._load_data()
        self._build_loaders()
        self._build_model()
        self.metrics_rows: List[Dict[str, object]] = []
        self.observed_epochs: List[int] = []
        self.best_state = None
        self.best_epoch: Optional[int] = None
        self.best_auc = float("-inf")
        self.stopped_epoch: Optional[int] = None

    def _load_data(self) -> None:
        dense, sparse, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = \
            get_movielens_data_rank_multidomain("all", "all", self.args.dataset_path)
        selected_x, selected_y = role_reversal_sample(
            x_train, y_train, self.setting, self.seed, self.role_sizes)
        train_mask = selected_x[:, -1].astype(int) == self.domain
        val_mask = x_val[:, -1].astype(int) == self.domain
        test_mask = x_test[:, -1].astype(int) == self.domain
        # domain_num=1 is the model's local interface; preserve original id in
        # the manifest and use 0 only inside the oracle feature arrays.
        self.x_train = np.array(selected_x[train_mask], copy=True)
        self.y_train = selected_y[train_mask]
        self.x_val = np.array(x_val[val_mask], copy=True)
        self.y_val = y_val[val_mask]
        self.x_test = np.array(x_test[test_mask], copy=True)
        self.y_test = y_test[test_mask]
        for array in (self.x_train, self.x_val, self.x_test):
            array[:, -1] = 0
        self.dense_features, self.sparse_features = dense, sparse
        expected_count = self.role_sizes[self.role]
        if len(self.x_train) != expected_count:
            raise AssertionError("oracle train count does not match role size")
        self.train_count = int(len(self.x_train))
        self.split_counts = {
            "train": {task: int(len(self.x_train)) for task in (0, 1)},
            "validation": {task: int(len(self.x_val)) for task in (0, 1)},
            "test": {task: int(len(self.x_test)) for task in (0, 1)},
        }
        self.positive_rates = {
            "train": {task: _positive_rate(self.y_train, task) for task in (0, 1)},
            "validation": {task: _positive_rate(self.y_val, task) for task in (0, 1)},
            "test": {task: _positive_rate(self.y_test, task) for task in (0, 1)},
        }
        self.dataset_identity = normalise_dataset_path(self.args.dataset_path)
        self.device_identity = str(self.device)
        self.split_row_fingerprints = {
            "train": split_row_fingerprint(self.x_train, self.y_train),
            "validation": split_row_fingerprint(self.x_val, self.y_val),
            "test": split_row_fingerprint(self.x_test, self.y_test),
        }

    def _build_loaders(self) -> None:
        dg = DataGenerator(self.x_train, self.y_train)
        self.train_loader, self.val_loader, self.test_loader = dg.generate_dataloader(
            x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test,
            batch_size=int(self.args.batch_size), num_workers=0)
        self.test_eval_loader = DataLoader(
            TorchDataset(self.x_test, self.y_test), batch_size=int(self.args.batch_size),
            shuffle=False, num_workers=0)

    def _build_model(self) -> None:
        self.model = MDMTRec(
            self.dense_features + self.sparse_features,
            domain_num=1, task_num=2, fcn_dims=[512, 256, 256, 64],
            expert_num=int(self.args.expert_num), exp_d=1, exp_t=1, bal_d=1, bal_t=1,
        ).to(self.device)
        self.partition = build_parameter_partition(self.model, strict=True)
        named = dict(self.model.named_parameters())
        main = [named[name] for name in self.partition.main_names if named[name].requires_grad]
        arch = [named[name] for name in self.partition.names("architecture") if named[name].requires_grad]
        self.optimizer = torch.optim.Adam(main, lr=float(self.args.learning_rate),
                                          weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = torch.optim.Adam(arch, lr=float(self.args.learning_rate_darts),
                                               weight_decay=float(self.args.weight_decay)) if arch else None
        self.criterion = torch.nn.BCELoss()

    def _append_rows(self, epoch: int, train: Mapping[int, Mapping[str, object]],
                     val: Mapping[int, Mapping[str, object]], test: Mapping[int, Mapping[str, object]],
                     selected: bool, censored: bool = False) -> None:
        for task in (0, 1):
            tr, va, te = train.get(task, {}), val.get(task, {}), test.get(task, {})
            self.metrics_rows.append({
                "seed": self.seed, "setting": self.setting,
                "original_physical_domain": self.domain, "role": self.role,
                "epoch": epoch, "domain": 0, "task": task, "cell": task,
                "train_sample_count": tr.get("sample_count", self.split_counts["train"][task]),
                "validation_sample_count": va.get("sample_count", self.split_counts["validation"][task]),
                "test_sample_count": te.get("sample_count", self.split_counts["test"][task]),
                "train_auc": tr.get("auc", ""), "train_logloss": tr.get("logloss", ""),
                "validation_auc": va.get("auc", ""), "validation_logloss": va.get("logloss", ""),
                "test_auc": te.get("auc", ""), "test_logloss": te.get("logloss", ""),
                "selected_checkpoint": int(selected), "stopped_censored": int(censored),
            })

    def fit(self) -> Dict[str, object]:
        arch_iter = iter(self.train_loader)
        bad = 0
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            self.model.train()
            truth = {0: [], 1: []}
            preds = {0: [], 1: []}
            loss_sum = {0: 0.0, 1: 0.0}
            counts = {0: 0, 1: 0}
            for x_batch, y_batch in self.train_loader:
                x_device = x_batch.transpose(0, 1).to(self.device)
                y_device = y_batch.to(self.device).float()
                output = self.model(x_device).to(self.device)
                terms = [self.criterion(output[:, task], y_device[:, task]) for task in (0, 1)]
                for task in (0, 1):
                    n = int(len(y_batch))
                    counts[task] += n
                    loss_sum[task] += float(terms[task].detach().cpu()) * n
                    truth[task].extend(y_device[:, task].detach().cpu().tolist())
                    preds[task].extend(output[:, task].detach().cpu().tolist())
                # Frozen single-domain/P0-4 main-loss semantics: task BCE
                # means are summed.  The architecture update below retains its
                # original unmasked criterion call and is a separate optimizer.
                loss = oracle_main_loss(output, y_device, self.criterion)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            train = {task: {"sample_count": counts[task],
                            "auc": safe_auc(truth[task], preds[task]),
                            "logloss": loss_sum[task] / counts[task] if counts[task] else float("nan")}
                     for task in (0, 1)}
            try:
                x_arch, y_arch = next(arch_iter)
            except StopIteration:
                arch_iter = iter(self.train_loader)
                x_arch, y_arch = next(arch_iter)
            if self.arch_optimizer is not None:
                x_device = x_arch.transpose(0, 1).to(self.device)
                y_device = y_arch.to(self.device).float()
                arch_loss = self.criterion(self.model(x_device).to(self.device), y_device)
                self.model.zero_grad(set_to_none=True)
                arch_loss.backward()
                self.arch_optimizer.step()
            val = _metrics(self.model, self.val_loader, self.device)
            before = torch.get_rng_state().clone()
            with preserve_rng_state():
                test = _metrics(self.model, self.test_eval_loader, self.device)
            _ = before  # preserve_rng_state provides the restoration contract
            val_mean = strict_validation_macro_auc(
                val, (0, 1), label="oracle validation")
            is_best = bool(np.isfinite(val_mean) and val_mean > self.best_auc)
            if is_best:
                for row in self.metrics_rows:
                    if row["selected_checkpoint"]:
                        row["selected_checkpoint"] = 0
                self.best_auc = val_mean
                self.best_epoch = epoch
                self.best_state = {name: value.detach().cpu().clone()
                                   for name, value in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            self._append_rows(epoch, train, val, test, is_best)
            if bad >= int(self.args.earlystop_patience):
                self.stopped_epoch = epoch
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.stopped_epoch is not None and self.stopped_epoch + 1 < int(self.args.epoch):
            for epoch in range(self.stopped_epoch + 1, int(self.args.epoch)):
                self._append_rows(epoch, {}, {}, {}, False, censored=True)
        return {"best_epoch": self.best_epoch if self.best_epoch is not None else "",
                "stopped_epoch": self.stopped_epoch if self.stopped_epoch is not None else ""}

    def _selected_checkpoint_complete(
        self, selected: Mapping[int, Mapping[str, object]], fit_info: Mapping[str, object],
    ) -> bool:
        selected_rows = [row for row in self.metrics_rows
                         if int(row.get("selected_checkpoint", 0)) == 1]
        if len(selected_rows) != 2 or set(selected) != {0, 1}:
            return False
        epochs = {_int_value(row.get("epoch")) for row in selected.values()}
        if len(epochs) != 1 or next(iter(epochs)) != _int_value(fit_info.get("best_epoch")):
            return False
        return all(_finite(row.get(field)) for row in selected.values()
                   for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"))

    def write_outputs(self, fit_info: Mapping[str, object], wall: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows, METRICS_FIELDS)
        selected = {task: row for row in self.metrics_rows
                    if int(row["selected_checkpoint"]) == 1 for task in [int(row["task"])]}
        summary = []
        for task in (0, 1):
            row = selected.get(task, {})
            summary.append({
                "seed": self.seed, "setting": self.setting,
                "original_physical_domain": self.domain, "role": self.role,
                "task": task, "domain": 0, "cell": task, "train_count": self.train_count,
                "best_epoch": fit_info.get("best_epoch", ""),
                "stopped_epoch": fit_info.get("stopped_epoch", ""),
                "best_val_auc": row.get("validation_auc", ""),
                "best_val_logloss": row.get("validation_logloss", ""),
                "best_test_auc": row.get("test_auc", ""),
                "best_test_logloss": row.get("test_logloss", ""),
                "oracle_auc": row.get("test_auc", ""), "oracle_logloss": row.get("test_logloss", ""),
            })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), summary, SUMMARY_FIELDS)
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_checkpoint_complete = self._selected_checkpoint_complete(selected, fit_info)
        config = vars(self.args).copy()
        config.update({"stage": "E3_ORACLE", "oracle": True, "alpha": None,
                       "original_physical_domain": self.domain, "role": self.role,
                       "train_count": self.train_count, "task_num": 2, "domain_num": 1,
                       "model_identity": {"class": "MDMTRec", "backbone": "mdmoe",
                                          "domain_num": 1, "task_num": 2,
                                          "expert_num": int(self.args.expert_num)},
                       "prohibited_mechanisms": ["alpha_weighting", "curriculum", "momentum",
                                                   "EMA", "adaptive_controller", "temporal_schedule"],
                       "formal_protocol": dict(E3_FORMAL_PROTOCOL),
                       "probe_interval": E3_FORMAL_PROTOCOL["probe_interval"],
                       "instrument": E3_FORMAL_PROTOCOL["instrument"],
                       "fix_optim_grouping": E3_FORMAL_PROTOCOL["fix_optim_grouping"],
                       "backbone": E3_FORMAL_PROTOCOL["backbone"],
                       "sampling": E3_FORMAL_PROTOCOL["sampling"],
                       "task": E3_FORMAL_PROTOCOL["task"], "domain": E3_FORMAL_PROTOCOL["domain"],
                       "role_sizes_parsed": self.role_sizes,
                       "dataset_identity": self.dataset_identity,
                       "device_identity": self.device_identity,
                       "split_row_fingerprints": self.split_row_fingerprints,
                       "selected_checkpoint_complete": bool(selected_checkpoint_complete),
                       "selected_checkpoint_cells": [str(task) for task in sorted(selected)],
                       "formal_valid": False})
        formal_fields = (
            "epoch", "batch_size", "learning_rate", "learning_rate_darts", "weight_decay",
            "expert_num", "earlystop_patience", "role_sizes", "probe_interval", "instrument",
            "fix_optim_grouping", "backbone", "sampling", "task", "domain",
        )
        formal_args_ok = all(config.get(field) == E3_FORMAL_PROTOCOL.get(field)
                             for field in formal_fields)
        observed_ok = self.observed_epochs == list(range(len(self.observed_epochs))) \
            and bool(self.observed_epochs)
        if self.stopped_epoch is None:
            observed_ok = observed_ok and len(self.observed_epochs) == int(E3_FORMAL_PROTOCOL["epoch"])
        else:
            observed_ok = observed_ok and self.stopped_epoch == self.observed_epochs[-1] \
                and self.stopped_epoch + 1 == int(fit_info.get("stopped_epoch")) + 1
        config["formal_valid"] = bool(
            formal_args_ok and observed_ok and selected_checkpoint_complete
            and os.path.exists(os.path.join(self.run_dir, "model.pth"))
            and self.dataset_identity == normalise_dataset_path(config.get("dataset_path", ""))
            and self.split_row_fingerprints == config.get("split_row_fingerprints"))
        config["config_hash"] = _hash_json(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E3_ORACLE", "status": "OK", "oracle": True,
            "seed": self.seed, "setting": self.setting,
            "original_physical_domain": self.domain, "role": self.role,
            "train_count": self.train_count, "role_sizes": self.role_sizes,
            "domain_num": 1, "task_num": 2,
            "split_counts": self.split_counts, "positive_rates": self.positive_rates,
            "dataset": "MovieLens-1M", "dataset_path": os.path.abspath(self.args.dataset_path),
            "dataset_identity": self.dataset_identity, "device_identity": self.device_identity,
            "split_row_fingerprints": self.split_row_fingerprints,
            "partition": self.partition.as_dict(),
            "model_identity": config["model_identity"],
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "observed_epochs": self.observed_epochs, "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": (self.stopped_epoch + 1 if self.stopped_epoch is not None else None),
            "alpha": None, "weighting": "equal_tasks_no_domain_weighting",
            "oracle_source": "stage_E3_oracle_run",
            "formal_protocol": dict(E3_FORMAL_PROTOCOL), "formal_valid": bool(config["formal_valid"]),
            "selected_checkpoint_complete": bool(selected_checkpoint_complete),
            "selected_checkpoint_cells": [str(task) for task in sorted(selected)],
            "formal_run_key": f"oracle|{self.setting}|d{self.domain}|{self.seed}",
            "wall_seconds": float(wall),
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E3 frozen single-domain oracle")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--setting", choices=sorted(ROLE_MAPS), default="d0_large")
    parser.add_argument("--domain", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--run_dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    started = time.time()
    runner = OracleRunner(args)
    info = runner.fit()
    runner.write_outputs(info, time.time() - started)
    print("[stage_E3_oracle] domain:", runner.domain, "raw outputs:", runner.run_dir, flush=True)


if __name__ == "__main__":
    main()
