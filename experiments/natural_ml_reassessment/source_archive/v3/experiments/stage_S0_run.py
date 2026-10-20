"""Stage S0 single-run trainer for the matched-momentum closure experiment.

This module deliberately contains only the seven pre-registered S0 arms.  It
reuses the audited MovieLens sampling, metric, fingerprint, RNG-preservation,
parameter-partition, and oracle-validation helpers from Stages E1/E3.  A run
always consumes its full epoch budget; early stopping is shadow state used
only to bound validation-only checkpoint selection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
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
for path in (HERE, M3OE_ROOT, EXAMPLES_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from MDMTRec.models.multi_domain import MDMTRec  # noqa: E402
from MDMTRec.utils.data import DataGenerator, TorchDataset  # noqa: E402
from run_movielens import get_movielens_data_rank_multidomain  # noqa: E402
from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint, build_parameter_partition, iter_cells,
    masked_bce_terms, parameter_state_hash, preserve_rng_state,
    rng_state_digest, safe_auc, safe_logloss, write_json, write_rows_csv,
)
from stage_E3_run import oracle_split_row_fingerprints  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    normalise_dataset_path, resolve_oracle_metrics, split_row_fingerprint,
)
from stage_S0_protocol import (  # noqa: E402
    ARM_SPECS, COMMON_EPOCHS, FORMAL_PROTOCOL, GAMMA_EQ, GAMMA_P, ROTATIONS,
    SEEDS, protocol_mismatches,
)


ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}

METRIC_FIELDS = [
    "arm", "seed", "rotation", "epoch", "domain", "physical_domain", "size_role",
    "task", "cell", "domain_size", "domain_share", "sample_weighted_train_loss",
    "validation_auc", "validation_logloss", "test_auc", "test_logloss",
    "oracle_auc", "oracle_logloss", "transfer_auc", "transfer_logloss",
    "raw_auc_nt", "raw_logloss_nt", "practical_auc_nt", "concordant_practical_nt",
    "selected_checkpoint", "fixed_final", "common_epoch",
]
MOMENTUM_FIELDS = [
    "arm", "seed", "rotation", "epoch", "step", "domain", "gamma",
    "raw_shared_gradient_norm", "ema_buffer_norm", "aggregate_shared_gradient_norm",
    "shared_update_norm", "adam_first_moment_norm", "adam_second_moment_norm",
    "ema_to_raw_ratio", "ema_update_count", "startup_coefficient", "learning_rate",
]
BATCH_FIELDS = ["epoch", "step", "kind", "batch_fingerprint", "batch_size",
                "domain_counts", "train_loss"]


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
        raise ValueError("role_sizes must be L,M,S positive integers")
    return dict(zip(("L", "M", "S"), parts))


def role_reversal_sample(x_train: np.ndarray, y_train: np.ndarray, rotation: str,
                         seed: int, sizes: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    if rotation not in ROLE_MAPS:
        raise ValueError("unknown rotation: {}".format(rotation))
    ids = x_train[:, -1].astype(int)
    keep = np.zeros(len(x_train), dtype=bool)
    for domain, role in ROLE_MAPS[rotation].items():
        pool = np.where(ids == domain)[0]
        requested = int(sizes[role])
        if requested > len(pool):
            raise ValueError("requested {} rows from domain {} with {} rows".format(
                requested, domain, len(pool)))
        order = np.random.RandomState(int(seed) * 10 + domain).permutation(len(pool))
        keep[pool[order[:requested]]] = True
    return x_train[keep], y_train[keep]


def gamma_vector(domain_shares: Sequence[float], arm: str) -> np.ndarray:
    spec = ARM_SPECS[arm]
    shares = np.asarray(domain_shares, dtype=np.float64)
    if spec["kind"] == "ema_domain":
        values = 1.0 - 0.1 * shares / shares.max()
        return np.clip(values, 0.9, 0.99)
    if spec["kind"] == "ema_uniform":
        return np.full(len(shares), float(spec["gamma"]), dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def memory_statistics(gammas: Sequence[float]) -> List[Dict[str, float]]:
    rows = []
    for gamma in gammas:
        g = float(gamma)
        rows.append({
            "gamma": g,
            "tau": 1.0 / (1.0 - g),
            "half_life": math.log(0.5) / math.log(g),
            "effective_n": (1.0 + g) / (1.0 - g),
        })
    return rows


def hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_group_norm(tensors: Iterable[Optional[torch.Tensor]]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is not None:
            total += float(torch.sum(tensor.detach().double() ** 2).cpu())
    return math.sqrt(total)


def require_finite_tensor(label: str, tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError("{} is disconnected (None)".format(label))
    if not bool(torch.isfinite(tensor.detach()).all().cpu()):
        raise FloatingPointError("{} contains NaN/Inf".format(label))
    return tensor


def require_finite_number(label: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FloatingPointError("{} is not numeric: {!r}".format(label, value)) from exc
    if not math.isfinite(result):
        raise FloatingPointError("{} is NaN/Inf".format(label))
    return result


def require_finite_parameters(label: str, parameters: Iterable[torch.nn.Parameter]) -> None:
    for index, parameter in enumerate(parameters):
        require_finite_tensor("{}[{}]".format(label, index), parameter)


class DomainEMA:
    """Per-domain shared-gradient EMA used identically by B and every C arm."""

    def __init__(self, params: Sequence[torch.nn.Parameter], gammas: Sequence[float]):
        self.params = list(params)
        self.gammas = np.asarray(gammas, dtype=np.float64)
        self.buffers = [[torch.zeros_like(param) for param in self.params]
                        for _ in range(len(self.gammas))]
        self.update_counts = [0 for _ in self.gammas]

    def update(self, domain_gradients: Sequence[Sequence[Optional[torch.Tensor]]]) -> None:
        if len(domain_gradients) != len(self.buffers):
            raise ValueError("one gradient list is required per domain")
        for domain, gradients in enumerate(domain_gradients):
            if len(gradients) != len(self.params):
                raise ValueError("gradient/parameter cardinality mismatch")
            gamma = float(self.gammas[domain])
            for index, (buffer, gradient) in enumerate(zip(self.buffers[domain], gradients)):
                gradient = require_finite_tensor(
                    "domain_{}_shared_gradient_{}".format(domain, index), gradient)
                buffer.mul_(gamma).add_(gradient.detach(), alpha=1.0 - gamma)
                require_finite_tensor(
                    "domain_{}_ema_buffer_{}".format(domain, index), buffer)
            self.update_counts[domain] += 1

    def aggregate_into_grads(self) -> None:
        for index, param in enumerate(self.params):
            aggregate = sum((self.buffers[d][index] for d in range(len(self.buffers))),
                            torch.zeros_like(param))
            require_finite_tensor("aggregate_shared_gradient_{}".format(index), aggregate)
            param.grad = aggregate.clone()

    def state_dict(self) -> Dict[str, object]:
        return {"gammas": self.gammas.tolist(), "update_counts": list(self.update_counts),
                "buffers": [[value.detach().cpu() for value in row] for row in self.buffers]}


def _metric_records(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    domain_num: int, task_num: int) -> Dict[int, Dict[str, float]]:
    truth = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    pred = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x = x_batch.transpose(0, 1).to(device)
            y = y_batch.to(device)
            output = model(x).to(device)
            require_finite_tensor("evaluation_prediction", output)
            require_finite_tensor("evaluation_target", y)
            ids = x[-1, :].detach()
            for idx, domain, task in iter_cells(domain_num, task_num):
                mask = ids == domain
                truth[idx].extend(y[mask, task].detach().cpu().tolist())
                pred[idx].extend(output[mask, task].detach().cpu().tolist())
    result = {}
    for idx in truth:
        if not truth[idx] or len(truth[idx]) != len(pred[idx]):
            raise RuntimeError("evaluation cell {} is empty or incomplete".format(idx))
        auc = require_finite_number("cell_{}_auc".format(idx), safe_auc(truth[idx], pred[idx]))
        logloss = require_finite_number(
            "cell_{}_logloss".format(idx), safe_logloss(truth[idx], pred[idx]))
        result[idx] = {"auc": auc, "logloss": logloss, "sample_count": len(truth[idx])}
    if len(result) != domain_num * task_num:
        raise RuntimeError("evaluation did not produce exactly six cells")
    return result


class S0Runner:
    def __init__(self, args: argparse.Namespace):
        if args.arm not in ARM_SPECS:
            raise ValueError("unregistered S0 arm: {}".format(args.arm))
        self.args = args
        self.arm = args.arm
        self.seed = int(args.seed)
        self.rotation = args.rotation
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        setup_seed(self.seed)
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.sizes = parse_role_sizes(args.role_sizes)
        self._load_data()
        self._build_loaders()
        self._build_model()
        self.metric_rows: List[Dict[str, object]] = []
        self.momentum_rows: List[Dict[str, object]] = []
        self.batch_rows: List[Dict[str, object]] = []
        self.rng_evidence: List[Dict[str, object]] = []
        self.batch_fingerprints: List[str] = []
        self.architecture_batch_fingerprints: List[str] = []
        self.examples_processed = 0
        self.architecture_examples_processed = 0
        self.optimizer_steps = 0
        self.architecture_steps = 0
        self.scheduler_steps = 0

    def _load_data(self) -> None:
        dense, sparse, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = \
            get_movielens_data_rank_multidomain("all", "all", self.args.dataset_path)
        self.dense_features, self.sparse_features = dense, sparse
        self.x_train, self.y_train = role_reversal_sample(
            x_train, y_train, self.rotation, self.seed, self.sizes)
        self.x_val, self.y_val, self.x_test, self.y_test = x_val, y_val, x_test, y_test
        self.domain_num, self.task_num = int(domain_num), 2
        ids = self.x_train[:, -1].astype(int)
        self.train_counts = {d: int((ids == d).sum()) for d in range(self.domain_num)}
        total = float(sum(self.train_counts.values()))
        self.domain_shares = np.asarray([self.train_counts[d] / total
                                         for d in range(self.domain_num)], dtype=np.float64)
        self.loss_coefficients = torch.as_tensor(
            np.tile(self.domain_shares / self.domain_shares.mean(), self.task_num),
            dtype=torch.float32, device=self.device)
        self.role_by_domain = ROLE_MAPS[self.rotation]
        self.split_fingerprints = {
            "train": split_row_fingerprint(self.x_train, self.y_train),
            "validation": split_row_fingerprint(self.x_val, self.y_val),
            "test": split_row_fingerprint(self.x_test, self.y_test),
        }
        oracle_fps = oracle_split_row_fingerprints(
            x_train, y_train, x_val, y_val, x_test, y_test,
            self.rotation, self.seed, self.sizes)
        self.dataset_identity = normalise_dataset_path(self.args.dataset_path)
        self.oracle_metrics, self.oracle_audit = resolve_oracle_metrics(
            self.args.oracle_root, self.rotation, self.seed, role_sizes=self.sizes,
            domain_num=self.domain_num, task_num=self.task_num,
            dataset_identity=self.dataset_identity, device_identity=str(self.device),
            expected_split_row_fingerprints=oracle_fps)
        if bool(int(self.args.require_oracles)) and not self.oracle_audit.get("complete"):
            raise RuntimeError("matching oracle coverage incomplete: {}".format(self.oracle_audit))

    def _build_loaders(self) -> None:
        generator = DataGenerator(self.x_train, self.y_train)
        self.train_loader, self.val_loader, _ = generator.generate_dataloader(
            x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test,
            batch_size=int(self.args.batch_size), num_workers=0)
        self.test_loader = DataLoader(TorchDataset(self.x_test, self.y_test),
                                      batch_size=int(self.args.batch_size), shuffle=False,
                                      num_workers=0)

    def _build_model(self) -> None:
        self.model = MDMTRec(
            self.dense_features + self.sparse_features, domain_num=self.domain_num,
            task_num=self.task_num, fcn_dims=[512, 256, 256, 64],
            expert_num=int(self.args.expert_num), exp_d=1, exp_t=1, bal_d=1, bal_t=1,
        ).to(self.device)
        self.partition = build_parameter_partition(self.model, strict=True)
        named = dict(self.model.named_parameters())
        self.shared_names = list(self.partition.names("shared"))
        self.shared_params = [named[name] for name in self.shared_names]
        arch_names = set(self.partition.names("architecture"))
        self.private_params = [param for name, param in self.model.named_parameters()
                               if name not in set(self.shared_names) and name not in arch_names]
        arch_params = [named[name] for name in self.partition.names("architecture")]
        beta1 = float(ARM_SPECS[self.arm]["shared_beta1"])
        self.optimizer = torch.optim.Adam([
            {"params": self.shared_params, "betas": (beta1, 0.999), "group_name": "shared"},
            {"params": self.private_params, "betas": (0.9, 0.999), "group_name": "private"},
        ], lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = torch.optim.Adam(
            arch_params, lr=float(self.args.learning_rate_darts),
            weight_decay=float(self.args.weight_decay), betas=(0.9, 0.999)) if arch_params else None
        self.criterion = torch.nn.BCELoss()
        self.gammas = gamma_vector(self.domain_shares, self.arm)
        self.ema = DomainEMA(self.shared_params, self.gammas) if len(self.gammas) else None
        self.initial_hash = parameter_state_hash(self.model)
        require_finite_parameters("initial_model_parameter", self.model.parameters())

    def _optimizer_state_norms(self) -> Tuple[float, float]:
        first, second = [], []
        for param in self.shared_params:
            state = self.optimizer.state.get(param, {})
            first.append(state.get("exp_avg"))
            second.append(state.get("exp_avg_sq"))
        return tensor_group_norm(first), tensor_group_norm(second)

    @staticmethod
    def _optimizer_audit(optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, object]:
        if optimizer is None:
            return {"group_count": 0, "groups": [], "state_parameter_count": 0,
                    "step_min": 0, "step_max": 0}
        groups = []
        for group in optimizer.param_groups:
            groups.append({
                "group_name": group.get("group_name", "architecture"),
                "betas": list(group.get("betas", ())),
                "parameter_count": len(group.get("params", [])),
            })
        steps = []
        for state in optimizer.state.values():
            for field in ("exp_avg", "exp_avg_sq"):
                if field in state:
                    require_finite_tensor("optimizer_state_{}".format(field), state[field])
            if "step" in state:
                step = state["step"]
                steps.append(int(step.detach().cpu()) if isinstance(step, torch.Tensor) else int(step))
        return {"group_count": len(groups), "groups": groups,
                "state_parameter_count": len(optimizer.state),
                "step_min": min(steps) if steps else 0, "step_max": max(steps) if steps else 0}

    def _architecture_update(self, iterator):
        try:
            x_batch, y_batch = next(iterator)
        except StopIteration:
            iterator = iter(self.train_loader)
            x_batch, y_batch = next(iterator)
        fingerprint = batch_fingerprint(x_batch, y_batch)
        self.architecture_batch_fingerprints.append(fingerprint)
        counts_per_domain = [int((x_batch[:, -1] == domain).sum().detach().cpu())
                             for domain in range(self.domain_num)]
        self.architecture_examples_processed += int(len(y_batch))
        if self.arch_optimizer is not None:
            x_device = x_batch.transpose(0, 1).to(self.device)
            y_device = y_batch.to(self.device)
            pred = self.model(x_device).to(self.device)
            terms, counts = masked_bce_terms(
                pred, y_device, x_device[-1, :].detach(), self.domain_num,
                self.task_num, self.criterion)
            if len(terms) != self.domain_num * self.task_num or any(count <= 0 for count in counts):
                raise RuntimeError("architecture batch does not contain six non-empty cells")
            for idx, term in enumerate(terms):
                require_finite_tensor("architecture_cell_loss_{}".format(idx), term)
            # Architecture parameters use the exact same D*p_d cell
            # objective (including its historical scale) as every main path.
            loss = (torch.stack(terms) * self.loss_coefficients).sum()
            require_finite_tensor("architecture_loss", loss)
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            for index, param in enumerate(self.partition.names("architecture")):
                require_finite_tensor(
                    "architecture_gradient_{}".format(index),
                    dict(self.model.named_parameters())[param].grad)
            self.arch_optimizer.step()
            self.architecture_steps += 1
            self._optimizer_audit(self.arch_optimizer)
            require_finite_parameters("post_architecture_parameter", self.model.parameters())
        self.batch_rows.append({
            "epoch": len(self.architecture_batch_fingerprints) - 1, "step": -1,
            "kind": "architecture", "batch_fingerprint": fingerprint,
            "batch_size": int(len(y_batch)),
            "domain_counts": json.dumps(counts_per_domain, separators=(",", ":")),
            "train_loss": float(loss.detach().cpu()) if self.arch_optimizer is not None else "",
        })
        return iterator

    def _train_epoch(self, epoch: int) -> Dict[int, float]:
        self.model.train()
        weighted_cell_sum = {idx: 0.0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        cell_counts = {idx: 0 for idx, _, _ in iter_cells(self.domain_num, self.task_num)}
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            fp = batch_fingerprint(x_batch, y_batch)
            self.batch_fingerprints.append(fp)
            x = x_batch.transpose(0, 1).to(self.device)
            y = y_batch.to(self.device)
            ids = x[-1, :].detach()
            pred = self.model(x).to(self.device)
            terms, counts = masked_bce_terms(
                pred, y, ids, self.domain_num, self.task_num, self.criterion)
            if len(terms) != self.domain_num * self.task_num or any(count <= 0 for count in counts):
                raise RuntimeError("training batch does not contain six non-empty cells")
            for idx, term in enumerate(terms):
                require_finite_tensor("train_cell_loss_{}".format(idx), term)
            loss = (torch.stack(terms) * self.loss_coefficients).sum()
            require_finite_tensor("sample_weighted_train_loss", loss)
            for idx, _, _ in iter_cells(self.domain_num, self.task_num):
                weighted_cell_sum[idx] += (float(terms[idx].detach().cpu())
                                           * float(self.loss_coefficients[idx].detach().cpu())
                                           * counts[idx])
                cell_counts[idx] += counts[idx]
            raw_by_domain: List[List[Optional[torch.Tensor]]] = []
            need_raw = self.ema is not None or bool(int(self.args.instrument))
            if need_raw:
                for domain in range(self.domain_num):
                    domain_loss = sum(self.loss_coefficients[task * self.domain_num + domain]
                                      * terms[task * self.domain_num + domain]
                                      for task in range(self.task_num))
                    gradients = torch.autograd.grad(
                        domain_loss, self.shared_params, retain_graph=True, allow_unused=True)
                    for index, gradient in enumerate(gradients):
                        require_finite_tensor(
                            "domain_{}_raw_shared_gradient_{}".format(domain, index), gradient)
                    raw_by_domain.append(list(gradients))
            before = [param.detach().clone() for param in self.shared_params] \
                if bool(int(self.args.instrument)) else []
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            for index, param in enumerate(self.shared_params):
                require_finite_tensor("full_shared_gradient_{}".format(index), param.grad)
            for index, param in enumerate(self.private_params):
                require_finite_tensor("full_private_gradient_{}".format(index), param.grad)
            if self.ema is not None:
                self.ema.update(raw_by_domain)
                self.ema.aggregate_into_grads()
            aggregate_norm = tensor_group_norm(param.grad for param in self.shared_params)
            self.optimizer.step()
            self._optimizer_audit(self.optimizer)
            require_finite_parameters("post_main_parameter", self.model.parameters())
            self.optimizer_steps += 1
            self.examples_processed += int(len(y_batch))
            update_norm = tensor_group_norm(
                old - param.detach() for old, param in zip(before, self.shared_params)) \
                if before else float("nan")
            first_norm, second_norm = self._optimizer_state_norms()
            if bool(int(self.args.instrument)):
                for domain in range(self.domain_num):
                    raw_norm = tensor_group_norm(raw_by_domain[domain])
                    ema_norm = (tensor_group_norm(self.ema.buffers[domain])
                                if self.ema is not None else "")
                    gamma = float(self.gammas[domain]) if self.ema is not None else ""
                    updates = self.ema.update_counts[domain] if self.ema is not None else 0
                    self.momentum_rows.append({
                        "arm": self.arm, "seed": self.seed, "rotation": self.rotation,
                        "epoch": epoch, "step": step, "domain": domain, "gamma": gamma,
                        "raw_shared_gradient_norm": raw_norm, "ema_buffer_norm": ema_norm,
                        "aggregate_shared_gradient_norm": aggregate_norm,
                        "shared_update_norm": update_norm,
                        "adam_first_moment_norm": first_norm,
                        "adam_second_moment_norm": second_norm,
                        "ema_to_raw_ratio": (float(ema_norm) / (raw_norm + 1e-12)
                                             if self.ema is not None else ""),
                        "ema_update_count": updates,
                        "startup_coefficient": (1.0 - float(gamma) ** updates
                                                if self.ema is not None else ""),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })
            counts_per_domain = [int((ids == domain).sum().detach().cpu())
                                 for domain in range(self.domain_num)]
            self.batch_rows.append({
                "epoch": epoch, "step": step, "kind": "train", "batch_fingerprint": fp,
                "batch_size": int(len(y_batch)),
                "domain_counts": json.dumps(counts_per_domain, separators=(",", ":")),
                "train_loss": float(loss.detach().cpu()),
            })
        return {idx: weighted_cell_sum[idx] / cell_counts[idx] if cell_counts[idx] else float("nan")
                for idx in weighted_cell_sum}

    def _append_metrics(self, epoch: int, train_loss: Mapping[int, float],
                        val_stats: Mapping[int, Mapping[str, float]],
                        test_stats: Mapping[int, Mapping[str, float]]) -> None:
        for idx, domain, task in iter_cells(self.domain_num, self.task_num):
            oracle = self.oracle_metrics.get(idx, {})
            oracle_auc, oracle_ll = oracle.get("auc", ""), oracle.get("logloss", "")
            test_auc = require_finite_number("test_auc", test_stats[idx]["auc"])
            test_ll = require_finite_number("test_logloss", test_stats[idx]["logloss"])
            require_finite_number("validation_auc", val_stats[idx]["auc"])
            require_finite_number("validation_logloss", val_stats[idx]["logloss"])
            require_finite_number("weighted_train_loss", train_loss[idx])
            has_oracle = oracle.get("coverage") is True
            delta_auc = (test_auc - require_finite_number("oracle_auc", oracle_auc)
                         if has_oracle else "")
            delta_ll = (test_ll - require_finite_number("oracle_logloss", oracle_ll)
                        if has_oracle else "")
            self.metric_rows.append({
                "arm": self.arm, "seed": self.seed, "rotation": self.rotation,
                "epoch": epoch, "domain": domain, "physical_domain": domain,
                "size_role": self.role_by_domain[domain], "task": task, "cell": idx,
                "domain_size": self.train_counts[domain], "domain_share": self.domain_shares[domain],
                "sample_weighted_train_loss": train_loss[idx],
                "validation_auc": val_stats[idx]["auc"],
                "validation_logloss": val_stats[idx]["logloss"],
                "test_auc": test_auc, "test_logloss": test_ll,
                "oracle_auc": oracle_auc, "oracle_logloss": oracle_ll,
                "transfer_auc": delta_auc, "transfer_logloss": delta_ll,
                "raw_auc_nt": int(has_oracle and float(delta_auc) < 0.0),
                "raw_logloss_nt": int(has_oracle and float(delta_ll) > 0.0),
                "practical_auc_nt": int(has_oracle and float(delta_auc) <= -0.003),
                "concordant_practical_nt": int(has_oracle and float(delta_auc) <= -0.003
                                                and float(delta_ll) > 0.0),
                "selected_checkpoint": 0, "fixed_final": int(epoch == int(self.args.epoch) - 1),
                "common_epoch": int(epoch in COMMON_EPOCHS),
            })

    def fit(self) -> Dict[str, object]:
        best_epoch, best_score = None, float("-inf")
        shadow_stop_epoch, bad_epochs = None, 0
        selected_state = None
        arch_iter = iter(self.train_loader)
        for epoch in range(int(self.args.epoch)):
            train_loss = self._train_epoch(epoch)
            arch_iter = self._architecture_update(arch_iter)
            val_stats = _metric_records(self.model, self.val_loader, self.device,
                                        self.domain_num, self.task_num)
            before = rng_state_digest()
            with preserve_rng_state():
                test_stats = _metric_records(self.model, self.test_loader, self.device,
                                             self.domain_num, self.task_num)
            after = rng_state_digest()
            self.rng_evidence.append({"epoch": epoch, "before": before, "after": after,
                                      "restored": before == after})
            val_score = float(np.mean([val_stats[idx]["auc"] for idx in val_stats]))
            if shadow_stop_epoch is None:
                if np.isfinite(val_score) and val_score > best_score:
                    best_score, best_epoch, bad_epochs = val_score, epoch, 0
                    selected_state = {name: value.detach().cpu().clone()
                                      for name, value in self.model.state_dict().items()}
                else:
                    bad_epochs += 1
                if bad_epochs >= int(self.args.shadow_patience):
                    shadow_stop_epoch = epoch
            self._append_metrics(epoch, train_loss, val_stats, test_stats)
        if best_epoch is None:
            raise RuntimeError("no finite validation checkpoint")
        for row in self.metric_rows:
            row["selected_checkpoint"] = int(int(row["epoch"]) == best_epoch)
        torch.save(selected_state, os.path.join(self.run_dir, "selected_checkpoint.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "final_checkpoint.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.run_dir, "optimizer_state.pth"))
        if self.arch_optimizer is not None:
            torch.save(self.arch_optimizer.state_dict(),
                       os.path.join(self.run_dir, "architecture_optimizer_state.pth"))
        if self.ema is not None:
            torch.save(self.ema.state_dict(), os.path.join(self.run_dir, "momentum_state.pth"))
        return {"selected_epoch": best_epoch, "selected_validation_macro_auc": best_score,
                "shadow_stop_epoch": shadow_stop_epoch, "fixed_final_epoch": int(self.args.epoch) - 1}

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metric_rows, METRIC_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "momentum_step.csv"), self.momentum_rows,
                       MOMENTUM_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows, BATCH_FIELDS)
        selected = [row for row in self.metric_rows if row["selected_checkpoint"] == 1]
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), selected, METRIC_FIELDS)
        source_paths = [__file__, os.path.join(HERE, "stage_S0_protocol.py"),
                        os.path.join(HERE, "stage_S0_orchestrator.py"),
                        os.path.join(HERE, "stage_S0_report.py")]
        code_fp = hashlib.sha256("".join(
            hashlib.sha256(open(path, "rb").read()).hexdigest()
            for path in source_paths if os.path.isfile(path)).encode()).hexdigest()
        gamma_stats = memory_statistics(self.gammas)
        config = vars(self.args).copy()
        config.update({
            "stage": "S0", "sample_objective": "w_d=D*p_d; sum over domain-task mean BCE",
            "domain_shares": self.domain_shares.tolist(), "loss_coefficients": self.loss_coefficients.tolist(),
            "shared_beta1": ARM_SPECS[self.arm]["shared_beta1"], "private_beta1": 0.9,
            "beta2": 0.999, "gamma_vector": self.gammas.tolist(), "memory_statistics": gamma_stats,
            "full_budget_no_early_stop": True, "checkpoint_metric": "validation_macro_auc",
            "checkpoint_tie_break": "first_argmax", "control_family_selection_uses_test": False,
            "formal_protocol": dict(FORMAL_PROTOCOL),
        })
        config["formal_protocol_mismatches"] = protocol_mismatches(config)
        config["formal_valid"] = not config["formal_protocol_mismatches"]
        write_json(os.path.join(self.run_dir, "config.json"), config)
        manifest = {
            "stage": "S0", "status": "OK", "arm": self.arm, "seed": self.seed,
            "rotation": self.rotation, "formal_run_key": f"{self.arm}|{self.rotation}|{self.seed}",
            "fresh_run": True, "dataset": "MovieLens-1M", "dataset_identity": self.dataset_identity,
            "device_identity": str(self.device),
            "split_row_fingerprints": self.split_fingerprints,
            "selected_training_indices_hash": self.split_fingerprints["train"],
            "code_fingerprint": code_fp, "role_by_domain": self.role_by_domain,
            "train_domain_counts": self.train_counts, "domain_shares": self.domain_shares.tolist(),
            "initial_parameter_state_hash": self.initial_hash,
            "final_parameter_state_hash": parameter_state_hash(self.model),
            "batch_order_hash": hash_json(self.batch_fingerprints),
            "batch_fingerprint_count": len(self.batch_fingerprints),
            "architecture_batch_order_hash": hash_json(self.architecture_batch_fingerprints),
            "architecture_batch_fingerprint_count": len(self.architecture_batch_fingerprints),
            "architecture_examples_processed": self.architecture_examples_processed,
            "examples_processed": self.examples_processed, "optimizer_steps": self.optimizer_steps,
            "scheduler_steps": self.scheduler_steps, "architecture_optimizer_steps": self.architecture_steps,
            "ema_update_counts": self.ema.update_counts if self.ema is not None else [0] * self.domain_num,
            "gamma_vector": self.gammas.tolist(), "memory_statistics": gamma_stats,
            "shared_beta1": ARM_SPECS[self.arm]["shared_beta1"], "private_beta1": 0.9,
            "beta2": 0.999, "selected_epoch": fit_info["selected_epoch"],
            "shadow_stop_epoch": fit_info["shadow_stop_epoch"],
            "fixed_final_epoch": fit_info["fixed_final_epoch"], "common_epochs": list(COMMON_EPOCHS),
            "oracle_audit": self.oracle_audit, "rng_restore_evidence": self.rng_evidence,
            "instrumentation": bool(int(self.args.instrument)), "wall_seconds": wall_seconds,
            "formal_protocol": dict(FORMAL_PROTOCOL), "formal_valid": bool(config["formal_valid"]),
            "formal_protocol_mismatches": config["formal_protocol_mismatches"],
            "main_optimizer_audit": self._optimizer_audit(self.optimizer),
            "architecture_optimizer_audit": self._optimizer_audit(self.arch_optimizer),
            "peak_memory_bytes": (torch.cuda.max_memory_allocated(self.device)
                                  if self.device.type == "cuda" else 0),
            "artifacts": ["config.json", "metrics_epoch.csv", "metrics_summary.csv",
                          "momentum_step.csv", "batch_trace.csv", "selected_checkpoint.pth",
                          "final_checkpoint.pth", "optimizer_state.pth",
                          "architecture_optimizer_state.pth"]
                         + (["momentum_state.pth"] if self.ema is not None else []),
            "scope_note": "raw S0 run; no scientific conclusion",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S0 matched-momentum single run")
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--rotation", choices=ROTATIONS, required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--learning_rate_darts", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--shadow_patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "stage_E3_raw_results", "oracle_reference"))
    parser.add_argument("--require_oracles", type=int, choices=(0, 1), default=1)
    parser.add_argument("--instrument", type=int, choices=(0, 1), default=1)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    started = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    runner: Optional[S0Runner] = None
    try:
        runner = S0Runner(args)
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        print("[stage_S0] completed", args.arm, args.rotation, args.seed, flush=True)
        return 0
    except Exception as exc:
        write_json(os.path.join(args.run_dir, "run_manifest.json"), {
            "stage": "S0", "status": "FAILED", "arm": args.arm, "seed": args.seed,
            "rotation": args.rotation, "exception_type": type(exc).__name__,
            "exception": str(exc), "wall_seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
