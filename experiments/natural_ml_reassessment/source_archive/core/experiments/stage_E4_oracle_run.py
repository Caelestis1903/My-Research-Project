"""Matched split-first/filter-second single-domain oracle for Stage E4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

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
    flatten_parameter_tensors,
    masked_bce_terms,
    parameter_state_hash,
    preserve_rng_state,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_E4_weighting import (  # noqa: E402
    E4_DATASET_PATH,
    E4_ORACLE_PROTOCOL,
    TASK_NUM,
    DatasetBundle,
    batch_fingerprint,
    build_parameter_partition,
    filter_split_first_domain,
    load_processed_dataset,
    metric_records,
    normalise_dataset_path,
    relative_l2_error,
    setup_seed,
    strict_validation_macro_auc,
    weighting_display,
    iter_cells,
    json_hash,
)


METRICS_FIELDS = [
    "seed", "original_physical_domain", "local_domain", "task", "cell", "epoch",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate",
    "train_auc", "train_logloss", "validation_auc", "validation_logloss",
    "test_auc", "test_logloss", "train_pooled_auc", "train_pooled_logloss",
    "validation_pooled_auc", "validation_pooled_logloss", "test_pooled_auc",
    "test_pooled_logloss", "selected_checkpoint", "stopped_censored",
]

SUMMARY_FIELDS = [
    "seed", "original_physical_domain", "local_domain", "task", "cell",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "train_positive_rate", "validation_positive_rate", "test_positive_rate",
    "selected_checkpoint", "best_epoch", "stopped_epoch", "best_val_auc",
    "best_val_logloss", "best_test_auc", "best_test_logloss", "oracle_auc",
    "oracle_logloss", "artifact_validity",
]

BATCH_FIELDS = [
    "seed", "original_physical_domain", "epoch", "step", "kind",
    "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "train_loss",
]


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


class OracleRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = int(args.seed)
        self.original_domain = int(args.domain)
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {args.device!r}, but CUDA is unavailable")
        setup_seed(self.seed)
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.bundle = load_processed_dataset(args.dataset_path, validate=True)
        self.filtered_splits, self.filter_info = filter_split_first_domain(
            self.bundle, self.original_domain)
        self.x_train, self.y_train = self.filtered_splits["train"]
        self.x_validation, self.y_validation = self.filtered_splits["validation"]
        self.x_test, self.y_test = self.filtered_splits["test"]
        self.domain_num = 1
        self.task_num = TASK_NUM
        self.split_counts = self.filter_info["split_counts"]
        self.positive_counts = self.filter_info["positive_counts"]
        self.positive_rates = {
            split: {t: float(self.y_values(split)[:, t].mean())
                    for t in range(TASK_NUM)}
            for split in ("train", "validation", "test")
        }
        self._build_loaders()
        self._build_model()
        self._init_outputs()

    def y_values(self, split: str) -> np.ndarray:
        return self.filtered_splits[split][1]

    def _build_loaders(self) -> None:
        generator = DataGenerator(self.x_train, self.y_train)
        self.train_loader, self.val_loader, self.test_loader = generator.generate_dataloader(
            x_val=self.x_validation, y_val=self.y_validation,
            x_test=self.x_test, y_test=self.y_test,
            batch_size=int(self.args.batch_size), num_workers=0, drop_last_flag=False)
        self.test_eval_loader = DataLoader(
            TorchDataset(self.x_test, self.y_test), batch_size=int(self.args.batch_size),
            shuffle=False, num_workers=0, drop_last=False)

    def _build_model(self) -> None:
        self.model = MDMTRec(
            list(self.bundle.dense_features) + list(self.bundle.sparse_features),
            domain_num=1, task_num=TASK_NUM, fcn_dims=[512, 256, 256, 64],
            expert_num=int(self.args.expert_num), exp_d=1, exp_t=1, bal_d=1, bal_t=1,
        ).to(self.device)
        self.partition = build_parameter_partition(self.model, strict=True)
        named = dict(self.model.named_parameters())
        main_params = [named[name] for name in self.partition.main_names if named[name].requires_grad]
        arch_params = [named[name] for name in self.partition.names("architecture") if named[name].requires_grad]
        if set(main_params).intersection(set(arch_params)):
            raise AssertionError("oracle main and architecture optimizer groups overlap")
        self.optimizer = torch.optim.Adam(
            main_params, lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = torch.optim.Adam(
            arch_params, lr=float(self.args.learning_rate_darts),
            weight_decay=float(self.args.weight_decay)) if arch_params else None
        self.criterion = torch.nn.BCELoss()

    def _init_outputs(self) -> None:
        self.metrics_rows: List[Dict[str, object]] = []
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

    def _record_batch(self, epoch: int, step: int, kind: str,
                      x_batch: torch.Tensor, y_batch: torch.Tensor,
                      train_loss: Optional[object] = None) -> str:
        fp = batch_fingerprint(x_batch, y_batch)
        self.batch_rows.append({
            "seed": self.seed, "original_physical_domain": self.original_domain,
            "epoch": int(epoch), "step": int(step), "kind": kind,
            "batch_fingerprint": fp, "batch_size": int(len(y_batch)),
            "domain_counts": _json_counts([int(len(y_batch)), int(len(y_batch))]),
            "task_counts": _json_counts([int(len(y_batch)), int(len(y_batch))]),
            "train_loss": "" if train_loss is None else float(
                train_loss.detach().cpu() if isinstance(train_loss, torch.Tensor) else train_loss),
        })
        return fp

    def _append_metrics(self, epoch: int, train: Mapping[int, Mapping[str, object]],
                        val: Mapping[int, Mapping[str, object]], test: Mapping[int, Mapping[str, object]],
                        train_pool: Mapping[str, object], val_pool: Mapping[str, object],
                        test_pool: Mapping[str, object], selected: bool, censored: bool = False) -> None:
        for task in range(TASK_NUM):
            tr, va, te = train.get(task, {}), val.get(task, {}), test.get(task, {})
            self.metrics_rows.append({
                "seed": self.seed, "original_physical_domain": self.original_domain,
                "local_domain": 0, "task": task, "cell": task, "epoch": int(epoch),
                "train_sample_count": tr.get("sample_count", self.split_counts["train"][task]),
                "validation_sample_count": va.get("sample_count", self.split_counts["validation"][task]),
                "test_sample_count": te.get("sample_count", self.split_counts["test"][task]),
                "train_positive_rate": tr.get("positive_rate", self.positive_rates["train"][task]),
                "validation_positive_rate": va.get("positive_rate", self.positive_rates["validation"][task]),
                "test_positive_rate": te.get("positive_rate", self.positive_rates["test"][task]),
                "train_auc": tr.get("auc", ""), "train_logloss": tr.get("logloss", ""),
                "validation_auc": va.get("auc", ""), "validation_logloss": va.get("logloss", ""),
                "test_auc": te.get("auc", ""), "test_logloss": te.get("logloss", ""),
                "train_pooled_auc": train_pool.get("auc", ""),
                "train_pooled_logloss": train_pool.get("logloss", ""),
                "validation_pooled_auc": val_pool.get("auc", ""),
                "validation_pooled_logloss": val_pool.get("logloss", ""),
                "test_pooled_auc": test_pool.get("auc", ""),
                "test_pooled_logloss": test_pool.get("logloss", ""),
                "selected_checkpoint": int(bool(selected)),
                "stopped_censored": int(bool(censored)),
            })

    def _architecture_update(self, epoch: int, arch_iter):
        try:
            x_arch, y_arch = next(arch_iter)
        except StopIteration:
            arch_iter = iter(self.train_loader)
            x_arch, y_arch = next(arch_iter)
        self.architecture_batch_fingerprints.append(
            self._record_batch(epoch, -1, "architecture", x_arch, y_arch))
        if self.arch_optimizer is None:
            return arch_iter
        x_device = x_arch.transpose(0, 1).to(self.device)
        y_device = y_arch.to(self.device).float()
        prediction = self.model(x_device).to(self.device)
        arch_loss = self.criterion(prediction, y_device)
        self.batch_rows[-1]["train_loss"] = float(arch_loss.detach().cpu())
        self.model.zero_grad(set_to_none=True)
        arch_loss.backward()
        self.arch_optimizer.step()
        return arch_iter

    def _train_epoch(self, epoch: int):
        self.model.train()
        truth = {0: [], 1: []}
        predictions = {0: [], 1: []}
        loss_sum = {0: 0.0, 1: 0.0}
        counts = {0: 0, 1: 0}
        fingerprints: List[str] = []
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            fp = self._record_batch(epoch, step, "train", x_batch, y_batch)
            fingerprints.append(fp)
            x_device = x_batch.transpose(0, 1).to(self.device)
            y_device = y_batch.to(self.device).float()
            prediction = self.model(x_device).to(self.device)
            terms, term_counts = masked_bce_terms(
                prediction, y_device, x_device[-1, :].detach(), 1, TASK_NUM, self.criterion)
            for task in range(TASK_NUM):
                counts[task] += term_counts[task]
                loss_sum[task] += float(terms[task].detach().cpu()) * term_counts[task]
                truth[task].extend(y_device[:, task].detach().cpu().tolist())
                predictions[task].extend(prediction[:, task].detach().cpu().tolist())
            loss = terms[0] + terms[1]
            self.batch_rows[-1]["train_loss"] = float(loss.detach().cpu())
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        train = {
            task: {"sample_count": counts[task], "positive_rate": float(np.mean(truth[task])),
                   "auc": self._safe_auc(truth[task], predictions[task]),
                   "logloss": loss_sum[task] / counts[task]}
            for task in range(TASK_NUM)
        }
        flat_truth = [value for task in range(TASK_NUM) for value in truth[task]]
        flat_pred = [value for task in range(TASK_NUM) for value in predictions[task]]
        pool = {"auc": self._safe_auc(flat_truth, flat_pred),
                "logloss": self._safe_logloss(flat_truth, flat_pred)}
        return train, pool, fingerprints

    @staticmethod
    def _safe_auc(truth, prediction):
        from stage_E4_weighting import safe_auc
        return safe_auc(truth, prediction)

    @staticmethod
    def _safe_logloss(truth, prediction):
        from stage_E4_weighting import safe_logloss
        return safe_logloss(truth, prediction)

    def fit(self) -> Dict[str, object]:
        arch_iter = iter(self.train_loader)
        bad_epochs = 0
        for epoch in range(int(self.args.epoch)):
            self.observed_epochs.append(epoch)
            train, train_pool, fingerprints = self._train_epoch(epoch)
            self.batch_fingerprints[epoch] = list(fingerprints)
            arch_iter = self._architecture_update(epoch, arch_iter)
            val, val_pool = metric_records(self.model, self.val_loader, self.device, 1, TASK_NUM)
            before = rng_state_digest()
            with preserve_rng_state():
                test, test_pool = metric_records(self.model, self.test_eval_loader, self.device, 1, TASK_NUM)
            after = rng_state_digest()
            self.rng_restore_evidence.append({
                "epoch": int(epoch), "scope": "test_eval", "before": before,
                "after": after, "restored": bool(before == after),
            })
            val_mean = strict_validation_macro_auc(val, (0, 1))
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
            self._append_metrics(epoch, train, val, test, train_pool, val_pool, test_pool, is_best)
            if bad_epochs >= int(self.args.earlystop_patience):
                self.stopped_epoch = epoch
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.stopped_epoch is not None and self.stopped_epoch + 1 < int(self.args.epoch):
            for epoch in range(self.stopped_epoch + 1, int(self.args.epoch)):
                self._append_metrics(epoch, {}, {}, {}, {}, {}, {}, False, censored=True)
        return {"best_epoch": self.best_epoch if self.best_epoch is not None else "",
                "stopped_epoch": self.stopped_epoch if self.stopped_epoch is not None else "",
                "observed_epochs": list(self.observed_epochs)}

    def _selected(self) -> Dict[int, Dict[str, object]]:
        return {int(row["task"]): row for row in self.metrics_rows
                if int(row.get("selected_checkpoint", 0)) == 1}

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metrics_rows, METRICS_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows, BATCH_FIELDS)
        selected = self._selected()
        summary = []
        for task in range(TASK_NUM):
            row = selected.get(task, {})
            summary.append({
                "seed": self.seed, "original_physical_domain": self.original_domain,
                "local_domain": 0, "task": task, "cell": task,
                "train_sample_count": self.split_counts["train"][task],
                "validation_sample_count": self.split_counts["validation"][task],
                "test_sample_count": self.split_counts["test"][task],
                "train_positive_rate": self.positive_rates["train"][task],
                "validation_positive_rate": self.positive_rates["validation"][task],
                "test_positive_rate": self.positive_rates["test"][task],
                "selected_checkpoint": int(bool(row)),
                "best_epoch": fit_info.get("best_epoch", ""),
                "stopped_epoch": fit_info.get("stopped_epoch", ""),
                "best_val_auc": row.get("validation_auc", ""),
                "best_val_logloss": row.get("validation_logloss", ""),
                "best_test_auc": row.get("test_auc", ""),
                "best_test_logloss": row.get("test_logloss", ""),
                "oracle_auc": row.get("test_auc", ""),
                "oracle_logloss": row.get("test_logloss", ""),
                "artifact_validity": "SMOKE_OR_FORMAL_PENDING",
            })
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), summary, SUMMARY_FIELDS)
        self.nonfinite_inventory = collect_nonfinite_inventory({
            "metrics_epoch.csv": self.metrics_rows,
            "batch_trace.csv": self.batch_rows,
            "metrics_summary.csv": summary,
        })
        inventory = {
            "stage": "E4_ORACLE", "count": len(self.nonfinite_inventory),
            "unexplained_count": sum(1 for item in self.nonfinite_inventory
                                      if item.get("reason") in ("", "nonfinite_numeric_value")),
            "entries": self.nonfinite_inventory,
        }
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), inventory)
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model.pth"))
        selected_complete = set(selected) == {0, 1} and all(
            _finite(row.get(field)) for row in selected.values()
            for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"))
        common = {
            "epoch": int(self.args.epoch), "batch_size": int(self.args.batch_size),
            "learning_rate": float(self.args.learning_rate),
            "learning_rate_darts": float(self.args.learning_rate_darts),
            "weight_decay": float(self.args.weight_decay), "expert_num": int(self.args.expert_num),
            "earlystop_patience": int(self.args.earlystop_patience),
        }
        formal_args_ok = all(common[field] == E4_ORACLE_PROTOCOL[field]
                             for field in common)
        observed_ok = self.observed_epochs == list(range(len(self.observed_epochs))) and bool(self.observed_epochs)
        if self.stopped_epoch is None:
            observed_ok = observed_ok and len(self.observed_epochs) == int(E4_ORACLE_PROTOCOL["epoch"])
        else:
            observed_ok = observed_ok and self.stopped_epoch == self.observed_epochs[-1]
        config = vars(self.args).copy()
        config.update({
            "stage": "E4_ORACLE", "oracle": True, "dataset": "KuaiRand-Pure",
            "dataset_path": self.bundle.dataset_path, "dataset_identity": self.bundle.dataset_identity,
            "dataset_file_identity": dict(self.bundle.dataset_file_identity),
            "device_identity": str(self.device), "seed": self.seed,
            "original_physical_domain": self.original_domain, "local_domain": 0,
            "domain_num": 1, "task_num": TASK_NUM, "task_mapping": {"is_click": 0, "long_view": 1},
            "local_domain_mapping": {str(self.original_domain): 0},
            "filter_after_global_split": True,
            "filter_method": self.filter_info["filter_method"],
            "split_counts": self.split_counts, "positive_counts": self.positive_counts,
            "positive_rates": self.positive_rates,
            "split_row_fingerprints": self.filter_info["split_row_fingerprints"],
            "global_split_row_fingerprints": self.filter_info["global_split_row_fingerprints"],
            "global_split_counts": self.bundle.metadata["split_counts"],
            "model_identity": {"class": "MDMTRec", "backbone": "mdmoe",
                               "fcn_dims": [512, 256, 256, 64], "expert_num": int(self.args.expert_num),
                               "domain_num": 1, "task_num": TASK_NUM,
                               "parameter_count": int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))},
            "partition": self.partition.as_dict(), "cell_order": CELL_ORDER_NOTE,
            "sampling": E4_ORACLE_PROTOCOL["sampling"], "scheduler": None,
            "prohibited_mechanisms": list(E4_ORACLE_PROTOCOL["prohibited_mechanisms"]),
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": [str(task) for task in sorted(selected)],
            "initial_parameter_state_hash": self.partition.state_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "observed_epochs": list(self.observed_epochs), "best_epoch": fit_info.get("best_epoch", ""),
            "stopped_epoch": fit_info.get("stopped_epoch", ""),
            "censored_after_epoch": self.stopped_epoch + 1 if self.stopped_epoch is not None else None,
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.architecture_batch_fingerprints,
            "oracle_reference": True,
        })
        config["formal_protocol"] = dict(E4_ORACLE_PROTOCOL)
        config["formal_valid"] = bool(
            formal_args_ok and observed_ok and selected_complete
            and self.filter_info.get("filter_after_global_split") is True
            and self.bundle.dataset_identity == normalise_dataset_path(config.get("dataset_path", ""))
            and os.path.exists(os.path.join(self.run_dir, "model.pth"))
        )
        summary_status = "FORMAL_VALID" if config["formal_valid"] else "SMOKE_ONLY"
        for row in summary:
            row["artifact_validity"] = summary_status
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), summary, SUMMARY_FIELDS)
        config["config_hash"] = json_hash(config)
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "E4_ORACLE", "status": "OK", "oracle": True, "seed": self.seed,
            "original_physical_domain": self.original_domain, "local_domain": 0,
            "dataset": "KuaiRand-Pure", "dataset_path": self.bundle.dataset_path,
            "dataset_identity": self.bundle.dataset_identity,
            "dataset_file_identity": dict(self.bundle.dataset_file_identity),
            "device_identity": str(self.device), "domain_num": 1, "task_num": TASK_NUM,
            "local_domain_mapping": {str(self.original_domain): 0},
            "filter_after_global_split": True,
            "filter_method": self.filter_info["filter_method"],
            "split_counts": self.split_counts, "positive_counts": self.positive_counts,
            "positive_rates": self.positive_rates,
            "split_row_fingerprints": self.filter_info["split_row_fingerprints"],
            "global_split_row_fingerprints": self.filter_info["global_split_row_fingerprints"],
            "global_split_counts": self.bundle.metadata["split_counts"],
            "partition": self.partition.as_dict(), "model_identity": config["model_identity"],
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
            "formal_protocol": dict(E4_ORACLE_PROTOCOL), "formal_valid": bool(config["formal_valid"]),
            "selected_checkpoint_complete": bool(selected_complete),
            "selected_checkpoint_cells": [str(task) for task in sorted(selected)],
            "rng_restore_evidence": list(self.rng_restore_evidence),
            "batch_fingerprints": self.batch_fingerprints,
            "architecture_batch_fingerprints": self.architecture_batch_fingerprints,
            "nonfinite_inventory_path": "nonfinite_inventory.json",
            "nonfinite_inventory_count": len(self.nonfinite_inventory),
            "nonfinite_inventory_unexplained_count": inventory["unexplained_count"],
            "formal_run_key": f"oracle|d{self.original_domain}|{self.seed}",
            "wall_seconds": float(wall_seconds),
            "scope_note": "matched oracle raw outputs only; no scientific conclusion is computed here",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)
        if not os.path.exists(os.path.join(self.run_dir, "stdout.log")):
            with open(os.path.join(self.run_dir, "stdout.log"), "a", encoding="utf-8") as handle:
                handle.write("[stage_E4_ORACLE] raw outputs written\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E4 split-first/filter-second oracle")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--domain", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default=E4_DATASET_PATH)
    parser.add_argument("--run_dir", required=True)
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
    started = time.time()
    runner = OracleRunner(args)
    fit_info = runner.fit()
    runner.write_outputs(fit_info, time.time() - started)
    print("[stage_E4_ORACLE] domain:", runner.original_domain, "raw outputs:", runner.run_dir, flush=True)
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
