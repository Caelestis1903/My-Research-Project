"""One fixed-budget Stage S2 Equal M3OE run.

This runner is intentionally derived from ``stage_S1_run.S1Runner``.  The
data construction, M3OE adapter, optimizer partition, checkpoint contract,
and metric implementation remain S1 imports.  The only training-path change
is dispatching the same fixed Equal objective for both the main and M3OE
architecture updates, plus S2-specific audit fields and output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
for _path in (HERE, M3OE_ROOT, EXAMPLES_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.utils.data import TorchDataset  # noqa: E402
from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint,
    parameter_state_hash,
    preserve_rng_state,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_S1_metrics import LOGLOSS_EPS  # noqa: E402
from stage_S1_models import (  # noqa: E402
    estimate_forward_flops,
    initial_state_hash,
    is_architecture_parameter,
    parameter_manifest,
)
from stage_S1_run import (  # noqa: E402
    BATCH_FIELDS as S1_BATCH_FIELDS,
    METRICS_EPOCH_FIELDS,
    METRICS_TEST_FIELDS,
    S1Runner,
    _cell_records,
    _macro,
    _optimizer_audit,
    _raw_rng_state,
    _require_finite,
    _rows_for_bundle,
    _runtime_metadata,
)
from stage_S2_objectives import (  # noqa: E402
    OBJECTIVE_FORMULA,
    OBJECTIVE_NAME,
    batch_weight_audit,
    compute_equal_coefficients,
    equal_objective,
    validate_coefficient_invariants,
)
from stage_S2_protocol import (  # noqa: E402
    BACKBONE,
    COMMON_EPOCHS,
    EPOCHS,
    FORMAL_PROTOCOL,
    SEEDS,
    STAGE,
    THRESHOLDS,
    equal_formal_key,
    protocol_mismatches,
)


S2_BATCH_FIELDS = list(S1_BATCH_FIELDS) + [
    "raw_domain_fractions", "weighted_mass_by_domain", "total_weighted_mass",
    "objective_name", "objective_formula", "coefficient_table_hash",
]


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash_inventory() -> Dict[str, str]:
    """Hash every S2 file and every frozen S1 implementation imported by it."""

    names = (
        "stage_S2_run.py", "stage_S2_objectives.py", "stage_S2_protocol.py",
        "stage_S2_orchestrator.py", "stage_S2_report.py",
        "stage_S1_run.py", "stage_S1_models.py", "stage_S1_data.py",
        "stage_S1_protocol.py", "stage_S1_metrics.py",
    )
    result: Dict[str, str] = {}
    for name in names:
        path = os.path.join(HERE, name)
        if os.path.isfile(path):
            result[name] = _file_hash(path)
    model_path = os.path.join(ROOT, "m3oe", "MDMTRec", "models", "multi_domain", "model.py")
    if os.path.isfile(model_path):
        result["m3oe/MDMTRec/models/multi_domain/model.py"] = _file_hash(model_path)
    return result


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _json_mapping(values: Mapping[str, object]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


class S2Runner(S1Runner):
    """S1-compatible M3OE runner with the S2 Equal objective."""

    def _validate_args(self) -> None:
        if str(self.args.run_type) != "joint":
            raise ValueError("Stage S2 fresh runs are joint only")
        if str(self.args.backbone).lower() != BACKBONE:
            raise ValueError("Stage S2 is M3OE-only")
        if bool(int(getattr(self.args, "allow_protocol_deviation", 0))):
            raise ValueError("formal Stage S2 does not allow protocol deviation")
        if bool(int(getattr(self.args, "allow_nonformal_seed", 0))):
            raise ValueError("formal Stage S2 does not allow non-formal seeds")
        if int(self.args.seed) not in SEEDS:
            raise ValueError("formal Stage S2 seeds are {}".format(SEEDS))
        if int(self.args.epoch) != EPOCHS:
            raise ValueError("Stage S2 requires exactly 50 epochs")
        for field in ("co_run_pair_id", "planned_peer_run_key", "worker_id"):
            if not str(getattr(self.args, field, "")):
                raise ValueError("formal Stage S2 requires frozen co-run field {}".format(field))
        expected = {
            "batch_size": FORMAL_PROTOCOL["batch_size"],
            "learning_rate": FORMAL_PROTOCOL["learning_rate"],
            "learning_rate_darts": FORMAL_PROTOCOL["learning_rate_darts"],
            "weight_decay": FORMAL_PROTOCOL["weight_decay"],
            "expert_num": FORMAL_PROTOCOL["expert_num"],
            "shadow_patience": FORMAL_PROTOCOL["shadow_patience"],
            "num_workers": FORMAL_PROTOCOL["num_workers"],
            "device": FORMAL_PROTOCOL["device"],
            "fcn_dims": FORMAL_PROTOCOL["fcn_dims"],
        }
        observed = {
            "batch_size": int(self.args.batch_size),
            "learning_rate": float(self.args.learning_rate),
            "learning_rate_darts": float(self.args.learning_rate_darts),
            "weight_decay": float(self.args.weight_decay),
            "expert_num": int(self.args.expert_num),
            "shadow_patience": int(self.args.shadow_patience),
            "num_workers": int(self.args.num_workers),
            "device": str(self.args.device),
            "fcn_dims": [int(x) for x in self.args.fcn_dims],
        }
        for field, value in expected.items():
            if observed[field] != value:
                raise ValueError("Stage S2 protocol mismatch for {}: {} != {}".format(
                    field, observed[field], value))

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # S1's constructor has loaded the immutable joint bundle and built the
        # exact M3OE model/optimizer partition.  Derive coefficients only now,
        # from the full training counts, and never from a minibatch.
        self.coefficient_table = compute_equal_coefficients(self.bundle.train_counts)
        validate_coefficient_invariants(self.coefficient_table)
        self.objective_name = OBJECTIVE_NAME
        self.objective_formula = OBJECTIVE_FORMULA
        self.objective_calls: List[Dict[str, object]] = []

    def _objective(self, prediction: torch.Tensor, target: torch.Tensor,
                   domain_ids: torch.Tensor) -> torch.Tensor:
        """Single objective dispatch used by main and architecture updates."""

        result = equal_objective(prediction, target, domain_ids,
                                 coefficient_table=self.coefficient_table)
        self.objective_calls.append({
            "coefficient_table_hash": self.coefficient_table["coefficient_table_hash"],
            "objective_name": self.objective_name,
        })
        return result

    def _record_batch(self, epoch: int, step: int, kind: str,
                      x_batch: torch.Tensor, y_batch: torch.Tensor,
                      objective: object = "") -> str:
        fingerprint = super()._record_batch(epoch, step, kind, x_batch, y_batch, objective)
        if self.instrumentation_enabled:
            ids = x_batch[:, -1].detach().cpu().long()
            audit = batch_weight_audit(ids, self.coefficient_table)
            row = self.batch_rows[-1]
            row.update({
                "raw_domain_fractions": json.dumps(
                    audit["raw_domain_fractions"], separators=(",", ":")),
                "weighted_mass_by_domain": _json_mapping(audit["weighted_mass_by_domain"]),
                "total_weighted_mass": float(audit["total_weighted_mass"]),
                "objective_name": self.objective_name,
                "objective_formula": self.objective_formula,
                "coefficient_table_hash": audit["coefficient_table_hash"],
            })
        return fingerprint

    def _train_epoch(self, epoch: int):
        self.model.train()
        objective_sum = 0.0
        sample_count = 0
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            fingerprint = self._record_batch(epoch, step, "train", x_batch, y_batch)
            self.batch_fingerprints.append(fingerprint)
            x_device = x_batch.transpose(0, 1).to(device=self.device, dtype=torch.float32)
            y_device = y_batch.to(device=self.device, dtype=torch.float32)
            domain_ids = x_device[-1, :].long()
            pred = self.model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite training prediction")
            loss = self._objective(pred, y_device, domain_ids)
            _require_finite("train_objective", loss.detach().cpu())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            architecture_names = {name for name, _ in self.model.named_parameters()
                                  if is_architecture_parameter(name)}
            for name, param in self.model.named_parameters():
                if name not in architecture_names and param.grad is not None \
                        and not bool(torch.isfinite(param.grad).all().detach().cpu()):
                    raise FloatingPointError("non-finite gradient {}".format(name))
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if not bool(torch.isfinite(param).all().detach().cpu()):
                    raise FloatingPointError("non-finite parameter {}".format(name))
            self.optimizer_steps += 1
            count = int(len(y_batch))
            self.examples_processed += count
            sample_count += count
            objective_sum += float(loss.detach().cpu()) * count
            if self.instrumentation_enabled:
                self.batch_rows[-1]["train_objective"] = float(loss.detach().cpu())
        train_objective = objective_sum / float(sample_count or 1)
        train_loader = DataLoader(
            TorchDataset(self.bundle.train_x, self.bundle.train_y),
            batch_size=int(self.args.batch_size), shuffle=False, num_workers=0, drop_last=False)
        train_records = _cell_records(self.bundle, self.model, train_loader, self.device)
        return train_objective, train_records

    def _architecture_update(self, architecture_iterator: Optional[Iterator], epoch: int):
        if self.arch_optimizer is None:
            return architecture_iterator
        self.model.train()
        rng_before = rng_state_digest()
        with preserve_rng_state():
            try:
                if architecture_iterator is None:
                    architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
            except StopIteration:
                architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
        rng_after = rng_state_digest()
        self.rng_evidence.append({"epoch": int(epoch), "kind": "architecture_batch",
                                  "before": rng_before, "after": rng_after,
                                  "restored": rng_before == rng_after})
        self._record_batch(epoch, -1, "architecture", x_batch, y_batch)
        fingerprint = batch_fingerprint(x_batch, y_batch)
        # _record_batch already returned the same fingerprint; recording it
        # explicitly keeps the S1 order/hash contract unchanged.
        self.architecture_batch_fingerprints.append(fingerprint)
        self.architecture_examples_processed += int(len(y_batch))
        x_device = x_batch.transpose(0, 1).to(device=self.device, dtype=torch.float32)
        y_device = y_batch.to(device=self.device, dtype=torch.float32)
        domain_ids = x_device[-1, :].long()
        pred = self.model(x_device)
        loss = self._objective(pred, y_device, domain_ids)
        _require_finite("architecture_train_objective", loss.detach().cpu())
        self.arch_optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in self.model.named_parameters():
            if is_architecture_parameter(name) and param.grad is not None \
                    and not bool(torch.isfinite(param.grad).all().detach().cpu()):
                raise FloatingPointError("non-finite architecture gradient {}".format(name))
        self.arch_optimizer.step()
        self.architecture_optimizer_steps += 1
        if self.instrumentation_enabled:
            self.batch_rows[-1]["train_objective"] = float(loss.detach().cpu())
        _optimizer_audit(self.arch_optimizer)
        return architecture_iterator

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        # Stage S1 helper rows carry S1 as their stage marker because the
        # underlying metric implementation is intentionally frozen.  Convert
        # only the copied records at the S2 output boundary.
        metric_rows = [dict(row, stage=STAGE) for row in self.metric_rows]
        test_rows = [dict(row, stage=STAGE) for row in self.test_rows]
        batch_rows = [dict(row, stage=STAGE) for row in self.batch_rows]
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), metric_rows,
                       METRICS_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "metrics_test.csv"), test_rows,
                       METRICS_TEST_FIELDS)
        selected_test = [row for row in test_rows if row["checkpoint"] == "selected"]
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), selected_test,
                       METRICS_TEST_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), batch_rows,
                       S2_BATCH_FIELDS)
        if self.test_labels_array is None or self.test_row_ids_array is None \
                or self.test_physical_domains_array is None:
            raise RuntimeError("deferred prediction arrays were not collected")
        prediction_payload = {
            "row_ids": self.test_row_ids_array,
            "physical_domain": self.test_physical_domains_array,
            "labels": self.test_labels_array,
        }
        for checkpoint, values in self.test_prediction_arrays.items():
            prediction_payload["pred_{}".format(checkpoint)] = values
        np.savez_compressed(os.path.join(self.run_dir, "predictions_test.npz"),
                            **prediction_payload)

        coefficients_path = os.path.join(self.run_dir, "objective_coefficients.json")
        write_json(coefficients_path, self.coefficient_table)
        checkpoints = self._checkpoint_payloads()
        checkpoint_hashes: Dict[str, str] = {}
        for filename, state in checkpoints.items():
            path = os.path.join(self.run_dir, filename)
            torch.save(dict(state), path)
            checkpoint_hashes[filename] = _file_hash(path)
        write_json(os.path.join(self.run_dir, "initialization_audit.json"), self.initialization_audit)
        write_json(os.path.join(self.run_dir, "parameter_manifest.json"), parameter_manifest(self.model))
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), self.nonfinite_inventory)
        torch.save(self.optimizer.state_dict(), os.path.join(self.run_dir, "optimizer_state.pth"))
        if self.arch_optimizer is not None:
            torch.save(self.arch_optimizer.state_dict(),
                       os.path.join(self.run_dir, "architecture_optimizer_state.pth"))
        else:
            raise RuntimeError("S2 M3OE must have an architecture optimizer")
        torch.save(_raw_rng_state(), os.path.join(self.run_dir, "rng_state.pth"))
        with open(os.path.join(self.run_dir, "stdout.log"), "a", encoding="utf-8") as stdout:
            stdout.write("[stage_S2] run outputs written; status=OK\n")

        protocol_values = dict(FORMAL_PROTOCOL)
        protocol_mismatch = protocol_mismatches(protocol_values)
        runtime = _runtime_metadata(self.device)
        runtime_fingerprint = hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        code_hashes = _code_hash_inventory()
        config = vars(self.args).copy()
        config.update({
            "stage": STAGE, "method": "equal", "formal_protocol": dict(FORMAL_PROTOCOL),
            "formal_protocol_observed": protocol_values,
            "formal_protocol_mismatches": protocol_mismatch,
            "formal_valid": not protocol_mismatch,
            "co_run_pair_id": str(self.args.co_run_pair_id),
            "planned_peer_run_key": str(self.args.planned_peer_run_key),
            "worker_id": str(self.args.worker_id),
            "objective": OBJECTIVE_NAME, "objective_formula": OBJECTIVE_FORMULA,
            "objective_coefficients": "objective_coefficients.json",
            "coefficient_table_hash": self.coefficient_table["coefficient_table_hash"],
            "objective_equivalence_contract": "Sample=sum_d p_d mean_t(cell BCE); Equal=(1/3)sum_d mean_t(cell BCE)",
            "metric_implementation": {"module": "stage_S1_metrics.py",
                                       "auc": "tie_aware_average_rank",
                                       "logloss": "fixed_epsilon_clip",
                                       "logloss_epsilon": LOGLOSS_EPS},
            "runtime": runtime, "runtime_fingerprint": runtime_fingerprint,
            "code_hashes": code_hashes,
            "source_code_hash": code_hashes.get("stage_S2_run.py", ""),
            "test_evaluated_during_training": False,
            "test_evaluated_after_selection": True,
            "checkpoint_selection_frozen_before_test": True,
            "thresholds": dict(THRESHOLDS),
            "s1_derived_runner": "stage_S1_run.S1Runner",
            "s1_source_modules_read_only": ["stage_S1_data.py", "stage_S1_models.py",
                                              "stage_S1_metrics.py", "stage_S1_protocol.py"],
        })
        write_json(os.path.join(self.run_dir, "config.json"), config)
        artifact_names = [
            "config.json", "metrics_epoch.csv", "metrics_test.csv", "metrics_summary.csv",
            "batch_trace.csv", "predictions_test.npz", "objective_coefficients.json",
            "initialization_audit.json", "parameter_manifest.json", "nonfinite_inventory.json",
            "optimizer_state.pth", "architecture_optimizer_state.pth", "rng_state.pth", "stdout.log",
        ] + list(checkpoints)
        artifact_hashes = {name: _file_hash(os.path.join(self.run_dir, name))
                           for name in artifact_names}
        bundle_manifest = self.bundle.manifest_fields()
        manifest = {
            "stage": STAGE, "status": "OK", "method": "equal", "run_type": "joint",
            "formal_run_key": equal_formal_key(self.dataset, self.setting, self.seed),
            "source_formal_run_key": "",
            "source_stage": STAGE, "provenance_type": "fresh",
            "co_run_pair_id": str(self.args.co_run_pair_id),
            "planned_peer_run_key": str(self.args.planned_peer_run_key),
            "worker_id": str(self.args.worker_id),
            "dataset": self.dataset, "dataset_label": bundle_manifest["dataset"],
            "dataset_key": self.dataset, "reference_scope": "joint",
            "setting": self.setting, "backbone": self.backbone, "seed": self.seed,
            "physical_domain": None, "run_dir": self.run_dir,
            "device_identity": str(self.device), "runtime": runtime,
            "runtime_fingerprint": runtime_fingerprint,
            "dataset_identity": self.bundle.preprocessing_metadata.get("raw_file", ""),
            "dataset_file_hash": (_file_hash(self.bundle.preprocessing_metadata["raw_file"])
                                  if self.bundle.preprocessing_metadata.get("raw_file") else ""),
            "source_code_hash": code_hashes.get("stage_S2_run.py", ""),
            "code_hashes": code_hashes, **bundle_manifest,
            "formal_protocol": dict(FORMAL_PROTOCOL),
            "formal_protocol_observed": protocol_values,
            "formal_protocol_mismatches": protocol_mismatch,
            "formal_valid": not protocol_mismatch,
            "objective": OBJECTIVE_NAME, "objective_formula": OBJECTIVE_FORMULA,
            "objective_coefficients": "objective_coefficients.json",
            "coefficient_table_hash": self.coefficient_table["coefficient_table_hash"],
            "no_double_weighting": True,
            "metric_implementation": {"module": "stage_S1_metrics.py",
                                       "auc": "tie_aware_average_rank",
                                       "logloss": "fixed_epsilon_clip",
                                       "logloss_epsilon": LOGLOSS_EPS},
            "optimizer": {"name": "Adam", "lr": float(self.args.learning_rate),
                          "betas": [0.9, 0.999], "weight_decay": float(self.args.weight_decay),
                          "scheduler": None, "audit": _optimizer_audit(self.optimizer)},
            "architecture_optimizer": {"name": "Adam", "lr": float(self.args.learning_rate_darts),
                                         "betas": [0.9, 0.999], "weight_decay": float(self.args.weight_decay),
                                         "steps_per_epoch": 1, "audit": _optimizer_audit(self.arch_optimizer)},
            "model_parameter_manifest": parameter_manifest(self.model),
            "forward_flops_estimate": estimate_forward_flops(self.model),
            "initial_parameter_state_hash": self.initial_parameter_state_hash,
            "final_parameter_state_hash": initial_state_hash(self.model),
            "semantic_initialization_audit": "initialization_audit.json",
            "examples_processed": int(self.examples_processed),
            "architecture_examples_processed": int(self.architecture_examples_processed),
            "optimizer_steps": int(self.optimizer_steps),
            "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
            "observed_epochs": list(range(EPOCHS)),
            "batches_per_epoch": int(len(self.train_loader)),
            "train_batches_total": int(len(self.batch_fingerprints)),
            "scheduler_steps": int(self.scheduler_steps),
            "batch_order_hash": hashlib.sha256(json.dumps(self.batch_fingerprints).encode()).hexdigest(),
            "batch_fingerprint_count": len(self.batch_fingerprints),
            "architecture_batch_order_hash": hashlib.sha256(
                json.dumps(self.architecture_batch_fingerprints).encode()).hexdigest(),
            "architecture_batch_fingerprint_count": len(self.architecture_batch_fingerprints),
            "objective_call_count": len(self.objective_calls),
            "objective_call_hashes": sorted({str(item["coefficient_table_hash"])
                                               for item in self.objective_calls}),
            "rng_state_digest_after_run": rng_state_digest(),
            "rng_restore_evidence": self.rng_evidence,
            "selected_epoch": int(self.selected_epoch),
            "selected_validation_macro_auc": float(self.selected_val_auc),
            "shadow_stop_epoch": self.shadow_stop_epoch,
            "fixed_final_epoch": EPOCHS - 1, "common_epochs": list(COMMON_EPOCHS),
            "checkpoints": checkpoint_hashes, "artifact_hashes": artifact_hashes,
            "optimizer_artifact": "optimizer_state.pth",
            "architecture_optimizer_artifact": "architecture_optimizer_state.pth",
            "prediction_artifact": "predictions_test.npz",
            "prediction_row_ids_hash": hashlib.sha256(
                np.asarray(self.test_row_ids_array, dtype=np.int64).tobytes()).hexdigest(),
            "prediction_labels_hash": hashlib.sha256(
                np.asarray(self.test_labels_array, dtype=np.float32).tobytes()).hexdigest(),
            "prediction_checkpoint_keys": sorted(self.test_prediction_arrays),
            "checkpoint_selection_frozen_before_test": True,
            "test_evaluated_during_training": False,
            "test_evaluation_order": ["selection_frozen", "selected", "final"]
            + ["common_epoch_{}".format(epoch) for epoch in COMMON_EPOCHS],
            "test_metric_in_epoch_history": False,
            "nonfinite_inventory": "nonfinite_inventory.json",
            "wall_seconds": float(wall_seconds),
            "peak_memory_bytes": (int(torch.cuda.max_memory_allocated(self.device))
                                  if self.device.type == "cuda" else 0),
            "artifacts": ["config.json", "run_manifest.json"] + artifact_names,
            "scope_note": "raw Stage S2 observation; no scientific conclusion",
        }
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S2 one-run Equal M3OE trainer")
    parser.add_argument("--run_type", choices=("joint",), default="joint")
    parser.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--backbone", choices=(BACKBONE,), default=BACKBONE)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical_domain", type=int, default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--batch_size", type=int, default=int(FORMAL_PROTOCOL["batch_size"]))
    parser.add_argument("--epoch", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=float(FORMAL_PROTOCOL["learning_rate"]))
    parser.add_argument("--learning_rate_darts", type=float,
                        default=float(FORMAL_PROTOCOL["learning_rate_darts"]))
    parser.add_argument("--weight_decay", type=float, default=float(FORMAL_PROTOCOL["weight_decay"]))
    parser.add_argument("--expert_num", type=int, default=int(FORMAL_PROTOCOL["expert_num"]))
    parser.add_argument("--fcn_dims", nargs=4, type=int, default=[512, 256, 256, 64])
    parser.add_argument("--shadow_patience", type=int, default=int(FORMAL_PROTOCOL["shadow_patience"]))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=str(FORMAL_PROTOCOL["device"]))
    parser.add_argument("--allow_protocol_deviation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow_nonformal_seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--co_run_pair_id", default="")
    parser.add_argument("--planned_peer_run_key", default="")
    parser.add_argument("--worker_id", default="")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    started = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    try:
        runner = S2Runner(args)
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        print("[stage_S2] completed equal", args.dataset, args.setting, BACKBONE,
              args.seed, flush=True)
        return 0
    except Exception as exc:
        prior_manifest = os.path.join(args.run_dir, "run_manifest.json")
        if not os.path.exists(prior_manifest):
            write_json(prior_manifest, {
                "stage": STAGE, "status": "FAILED", "method": "equal",
                "run_type": "joint", "dataset": args.dataset, "setting": args.setting,
                "backbone": BACKBONE, "seed": int(args.seed),
                "exception_type": type(exc).__name__, "exception": str(exc),
                "co_run_pair_id": str(getattr(args, "co_run_pair_id", "")),
                "planned_peer_run_key": str(getattr(args, "planned_peer_run_key", "")),
                "worker_id": str(getattr(args, "worker_id", "")),
                "wall_seconds": time.time() - started,
            })
        raise


def run_tiny_runner_fixture(device: Optional[torch.device] = None,
                            representative: bool = False) -> Dict[str, object]:
    """Exercise the real S2Runner update paths on a tiny synthetic M3OE batch.

    This is a nonformal integration fixture: it runs one main epoch and one
    architecture update, uses the actual Stage S1 M3OE adapter and the S2
    objective dispatch, and writes no formal artifact or dataset result.
    """

    from types import SimpleNamespace
    from stage_S1_data import FeatureSchema
    from stage_S1_models import build_model

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if representative:
        # KuaiRand's frozen formal schema/parameter scale, but synthetic rows
        # and one batch only; this is never a dataset or formal run.
        columns = (
            "user_active_degree", "is_live_streamer", "is_video_author",
            "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
            "register_days_range", "onehot_feat0", "onehot_feat1", "onehot_feat2",
            "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
            "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10",
            "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
            "onehot_feat15", "onehot_feat16", "onehot_feat17", "video_type",
            "upload_dt", "upload_type", "server_width", "server_height",
            "music_type", "tag", "counts", "domain_indicator")
        vocab_sizes = {
            "counts": 126, "domain_indicator": 3, "fans_user_num_range": 10,
            "follow_user_num_range": 9, "friend_user_num_range": 8,
            "is_live_streamer": 3, "is_video_author": 3, "music_type": 7,
            "onehot_feat0": 3, "onehot_feat1": 8, "onehot_feat10": 6,
            "onehot_feat11": 6, "onehot_feat12": 4, "onehot_feat13": 4,
            "onehot_feat14": 4, "onehot_feat15": 4, "onehot_feat16": 4,
            "onehot_feat17": 4, "onehot_feat2": 50, "onehot_feat3": 1461,
            "onehot_feat4": 17, "onehot_feat5": 35, "onehot_feat6": 4,
            "onehot_feat7": 119, "onehot_feat8": 455, "onehot_feat9": 8,
            "register_days_range": 9, "server_height": 121, "server_width": 157,
            "tag": 112, "upload_dt": 4, "upload_type": 15,
            "user_active_degree": 10, "video_type": 4,
        }
        sample_count = 40960
        batch_size = 40960
        features = np.zeros((sample_count, len(columns)), dtype=np.float32)
        for index, column in enumerate(columns[:-1]):
            features[:, index] = np.arange(sample_count, dtype=np.float32) % max(
                1, int(vocab_sizes[column]))
        features[:, -1] = np.arange(sample_count, dtype=np.float32) % 3
        schema = FeatureSchema(
            dataset="kuairand", columns=columns, dense_columns=(),
            sparse_columns=columns, vocab_sizes=vocab_sizes, embedding_dim=16,
            domain_column="domain_indicator", encoder_hash="representative",
            scaler_hash="representative")
    else:
        columns = ("dense", "cat", "domain_indicator")
        vocab_sizes = {"cat": 9, "domain_indicator": 3}
        sample_count = 12
        batch_size = 4
        features = np.zeros((sample_count, 3), dtype=np.float32)
        features[:, 0] = np.arange(sample_count, dtype=np.float32) / float(sample_count)
        features[:, 1] = np.arange(sample_count, dtype=np.float32) % 9
        features[:, 2] = np.repeat(np.arange(3, dtype=np.float32), 4)
        schema = FeatureSchema(
            dataset="synthetic", columns=columns, dense_columns=("dense",),
            sparse_columns=("cat", "domain_indicator"), vocab_sizes=vocab_sizes,
            embedding_dim=16, domain_column="domain_indicator", encoder_hash="synthetic",
            scaler_hash="synthetic")
    labels = np.zeros((sample_count, 2), dtype=np.float32)
    labels[:, 0] = np.arange(sample_count) % 2
    labels[:, 1] = (np.arange(sample_count) + 1) % 2
    bundle = SimpleNamespace(domain_num=3, task_num=2, train_x=features, train_y=labels)
    runner = S2Runner.__new__(S2Runner)
    runner.args = SimpleNamespace(batch_size=batch_size)
    runner.dataset = "kuairand" if representative else "synthetic"
    runner.setting = "tiny_fixture"
    runner.backbone = BACKBONE
    runner.seed = 2022
    runner.run_type = "joint"
    runner.bundle = bundle
    runner.device = device
    runner.model = build_model(schema, 3, 2, BACKBONE, seed=2022).to(device)
    runner.train_loader = DataLoader(
        TorchDataset(features, labels), batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False)
    domain_counts = [int((features[:, -1] == domain).sum()) for domain in range(3)]
    runner.coefficient_table = compute_equal_coefficients(domain_counts)
    runner.objective_name = OBJECTIVE_NAME
    runner.objective_formula = OBJECTIVE_FORMULA
    runner.instrumentation_enabled = True
    runner.batch_rows = []
    runner.batch_fingerprints = []
    runner.architecture_batch_fingerprints = []
    runner.objective_calls = []
    runner.optimizer_steps = 0
    runner.examples_processed = 0
    runner.architecture_optimizer_steps = 0
    runner.architecture_examples_processed = 0
    runner.nonfinite_inventory = []
    runner.rng_evidence = []
    runner.scheduler_steps = 0
    architecture_parameters = [parameter for name, parameter in runner.model.named_parameters()
                               if is_architecture_parameter(name)]
    main_parameters = [parameter for name, parameter in runner.model.named_parameters()
                       if not is_architecture_parameter(name)]
    runner.optimizer = torch.optim.Adam(main_parameters, lr=0.01)
    runner.arch_optimizer = torch.optim.Adam(architecture_parameters, lr=0.001)

    train_objective, records = runner._train_epoch(0)
    runner._architecture_update(None, 0)
    if len(runner.batch_fingerprints) != len(runner.train_loader):
        raise AssertionError("tiny runner main fingerprint count mismatch")
    if len(runner.architecture_batch_fingerprints) != 1:
        raise AssertionError("tiny runner architecture fingerprint count mismatch")
    if runner.optimizer_steps != len(runner.train_loader) or runner.architecture_optimizer_steps != 1:
        raise AssertionError("tiny runner optimizer-step contract mismatch")
    if not runner.objective_calls or len(set(item["coefficient_table_hash"]
                                             for item in runner.objective_calls)) != 1:
        raise AssertionError("tiny runner objective dispatch/hash contract mismatch")
    if not math.isfinite(float(train_objective)) or len(records) != 6:
        raise AssertionError("tiny runner objective/metric fixture failed")
    return {
        "profile": "kuairand_representative" if representative else "tiny",
        "device": str(device), "main_batch_fingerprints": len(runner.batch_fingerprints),
        "architecture_batch_fingerprints": len(runner.architecture_batch_fingerprints),
        "main_optimizer_steps": runner.optimizer_steps,
        "architecture_optimizer_steps": runner.architecture_optimizer_steps,
        "objective_call_count": len(runner.objective_calls),
        "coefficient_table_hash": runner.coefficient_table["coefficient_table_hash"],
        "parameter_count": int(parameter_manifest(runner.model)["total"]),
    }


__all__ = ["S2Runner", "S2_BATCH_FIELDS", "build_argparser", "main", "run_tiny_runner_fixture"]


if __name__ == "__main__":
    raise SystemExit(main())
