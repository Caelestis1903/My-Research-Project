"""Stage S3 M3OE runner with source-controlled shared-gradient access."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

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
    batch_fingerprint, parameter_state_hash, preserve_rng_state, rng_state_digest,
    write_json, write_rows_csv,
)
from stage_S1_data import FeatureSchema  # noqa: E402
from stage_S1_metrics import LOGLOSS_EPS  # noqa: E402
from stage_S1_models import (  # noqa: E402
    build_model, estimate_forward_flops, initial_state_hash, parameter_manifest,
)
from stage_S1_run import (  # noqa: E402
    BATCH_FIELDS as S1_BATCH_FIELDS, EPOCHS as S1_EPOCHS, METRICS_EPOCH_FIELDS,
    METRICS_TEST_FIELDS, S1Runner, _cell_records, _macro, _optimizer_audit,
    _raw_rng_state, _require_finite, _runtime_metadata,
)
from stage_S2_objectives import compute_equal_coefficients  # noqa: E402
from stage_S2_run import S2Runner  # noqa: E402
from stage_S3_gradient_access import (  # noqa: E402
    GROUP_ARCH, GROUP_PRIVATE, GROUP_SHARED, apply_architecture_gradients,
    apply_main_gradients, assert_expected_parameter_counts, loss_components,
    partition_parameters, reachability_audit,
)
from stage_S3_protocol import (  # noqa: E402
    ACCESS, BACKBONE, COMMON_EPOCHS_S3, EPOCHS, FORMAL_PROTOCOL,
    FRESH_SAMPLE_FULL_ROLE, FRESH_TARGET_ROLE, FULL, MODEL_SOURCE_KEY,
    OBJECTIVES, S3_SEEDS, STAGE, fresh_formal_key, formal_protocol_for,
    sample_full_replacement_key, source_inventory, source_inventory_digest,
    target_for,
)

OBJECTIVE_NAMES = {
    "equal": "equal_domain_task_mean_bce",
    "sample": "sample_population_mean_task_bce",
}
OBJECTIVE_FORMULAS = {
    "equal": "mean_i(w[d_i]*mean_t(BCE(p_it,y_it))); w_d=1/(3*p_d)",
    "sample": "mean_i(mean_t(BCE(p_it,y_it)))",
}
S3_BATCH_FIELDS = list(S1_BATCH_FIELDS) + [
    "objective_weighting", "shared_access", "target_local_domain", "target_count",
    "L_full", "L_target", "L_non", "decomposition_residual",
    "applied_shared_gradient_norm", "applied_private_gradient_norm",
    "applied_architecture_gradient_norm", "removed_non_target_shared_gradient_norm",
    "removed_non_target_architecture_gradient_norm", "gradient_substitution_valid",
    "gradient_partition_hash", "main_optimizer_step_count",
    "architecture_optimizer_step_count",
]
GRADIENT_STEP_FIELDS = [
    "stage", "formal_run_key", "objective_weighting", "shared_access", "epoch", "step",
    "kind", "batch_fingerprint", "batch_size", "domain_counts", "L_full", "L_target",
    "L_non", "decomposition_residual", "applied_shared_gradient_norm",
    "applied_private_gradient_norm", "applied_architecture_gradient_norm",
    "removed_non_target_shared_gradient_norm", "removed_non_target_architecture_gradient_norm",
    "gradient_substitution_valid", "gradient_partition_hash", "optimizer_steps",
    "architecture_optimizer_steps",
]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash_inventory() -> Dict[str, str]:
    return source_inventory(ROOT)


def _sample_objective_table(equal_table: Mapping[str, object]) -> Dict[str, object]:
    counts = {str(k): int(v) for k, v in equal_table["train_counts"].items()}
    shares = {str(k): float(v) for k, v in equal_table["domain_share_p"].items()}
    weights = {str(d): 1.0 for d in range(3)}
    effective = dict(shares)
    result: Dict[str, object] = {
        "domain_num": 3, "task_num": 2, "train_counts": counts,
        "total_train_count": int(equal_table["total_train_count"]),
        "domain_share_p": shares, "loss_coefficient_w": weights,
        "effective_domain_coefficient_a": effective,
        "effective_cell_coefficient": {str(d): effective[str(d)] / 2.0 for d in range(3)},
        "population_mean_weight": 1.0,
        "objective_name": OBJECTIVE_NAMES["sample"],
        "objective_formula": OBJECTIVE_FORMULAS["sample"],
        "sampling": "ordinary_shuffled_concatenated_loader", "double_weighting": False,
    }
    result["coefficient_table_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


def _smoke_feature_schema(dataset: str) -> FeatureSchema:
    dataset = str(dataset).lower()
    if dataset == "movielens":
        columns = ("user_id", "movie_id", "gender", "age", "occupation", "zip",
                   "cate_id", "domain_indicator")
        dense = ("age",)
        sparse = tuple(name for name in columns if name not in dense)
        vocab = {"user_id": 5401, "movie_id": 3663, "gender": 3, "occupation": 22,
                 "zip": 3201, "cate_id": 19, "domain_indicator": 3}
    elif dataset == "kuairand":
        columns = (
            "user_active_degree", "is_live_streamer", "is_video_author",
            "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
            "register_days_range", "onehot_feat0", "onehot_feat1", "onehot_feat2",
            "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
            "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10",
            "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
            "onehot_feat15", "onehot_feat16", "onehot_feat17", "video_type",
            "upload_dt", "upload_type", "server_width", "server_height", "music_type",
            "tag", "counts", "domain_indicator",
        )
        dense = ()
        sparse = columns
        vocab = {
            "user_active_degree": 10, "is_live_streamer": 3, "is_video_author": 3,
            "follow_user_num_range": 9, "fans_user_num_range": 10,
            "friend_user_num_range": 8, "register_days_range": 9,
            "onehot_feat0": 3, "onehot_feat1": 8, "onehot_feat2": 50,
            "onehot_feat3": 1461, "onehot_feat4": 17, "onehot_feat5": 35,
            "onehot_feat6": 4, "onehot_feat7": 119, "onehot_feat8": 455,
            "onehot_feat9": 8, "onehot_feat10": 6, "onehot_feat11": 6,
            "onehot_feat12": 4, "onehot_feat13": 4, "onehot_feat14": 4,
            "onehot_feat15": 4, "onehot_feat16": 4, "onehot_feat17": 4,
            "video_type": 4, "upload_dt": 4, "upload_type": 15,
            "server_width": 157, "server_height": 121, "music_type": 7,
            "tag": 112, "counts": 126, "domain_indicator": 3,
        }
    else:
        raise ValueError("unknown Stage S3 smoke dataset")
    return FeatureSchema(
        dataset=dataset, columns=columns, dense_columns=dense, sparse_columns=sparse,
        vocab_sizes=vocab, embedding_dim=16, domain_column="domain_indicator",
        encoder_hash="stage_S3_smoke_schema", scaler_hash="stage_S3_smoke_schema", oov_index=0,
    )


class S3Runner(S2Runner):
    """S1/S2-compatible runner with the frozen S3 gradient-access factor."""

    def _validate_args(self) -> None:
        args = self.args
        if str(args.run_type) != "joint" or str(args.backbone).lower() != BACKBONE:
            raise ValueError("Stage S3 formal runs are joint M3OE only")
        if bool(int(getattr(args, "allow_protocol_deviation", 0))) \
                or bool(int(getattr(args, "allow_nonformal_seed", 0))):
            raise ValueError("formal Stage S3 does not allow protocol or seed deviations")
        if int(args.seed) not in S3_SEEDS or int(args.epoch) != EPOCHS:
            raise ValueError("Stage S3 requires seeds 2022-2025 and 50 epochs")
        expected = {
            "batch_size": 40960, "learning_rate": 0.01,
            "learning_rate_darts": 0.001, "weight_decay": 1e-5,
            "expert_num": 4, "shadow_patience": 10, "num_workers": 0,
            "device": "cuda:0", "fcn_dims": [512, 256, 256, 64],
        }
        observed = {
            "batch_size": int(args.batch_size), "learning_rate": float(args.learning_rate),
            "learning_rate_darts": float(args.learning_rate_darts),
            "weight_decay": float(args.weight_decay), "expert_num": int(args.expert_num),
            "shadow_patience": int(args.shadow_patience), "num_workers": int(args.num_workers),
            "device": str(args.device), "fcn_dims": [int(v) for v in args.fcn_dims],
        }
        if observed != expected:
            raise ValueError("Stage S3 protocol mismatch: {}".format(observed))
        objective = str(args.objective_weighting).lower()
        fresh_role = str(args.fresh_role)
        access = str(args.shared_access)
        dataset, setting, seed = str(args.dataset).lower(), str(args.setting), int(args.seed)
        target_for(dataset, setting)
        if fresh_role == FRESH_TARGET_ROLE:
            expected_access = ACCESS
            key = fresh_formal_key(objective, dataset, setting, seed)
        elif fresh_role == FRESH_SAMPLE_FULL_ROLE:
            if objective != "sample":
                raise ValueError("only Sample FULL replacements are an allowed fresh FULL path")
            expected_access = FULL
            key = sample_full_replacement_key(dataset, setting, seed)
        else:
            raise ValueError("unknown Stage S3 fresh role")
        if objective not in OBJECTIVES or access != expected_access:
            raise ValueError("Stage S3 objective/access combination is not in the frozen matrix")
        if str(args.formal_run_key) != key:
            raise ValueError("Stage S3 formal run key does not match its condition")
        for name in ("co_run_pair_id", "planned_peer_run_key", "worker_id", "attempt_id"):
            if not str(getattr(args, name, "")):
                raise ValueError("formal Stage S3 requires frozen launch field {}".format(name))

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.objective_weighting = str(args.objective_weighting).lower()
        self.shared_access = str(args.shared_access)
        self.fresh_role = str(args.fresh_role)
        self.target_local_domain = int(target_for(self.dataset, self.setting)["local_domain"])
        self.formal_run_key = str(args.formal_run_key)
        self.source_formal_run_key = str(getattr(args, "source_formal_run_key", ""))
        self.objective_name = OBJECTIVE_NAMES[self.objective_weighting]
        self.objective_formula = OBJECTIVE_FORMULAS[self.objective_weighting]
        self.objective_table = (self.coefficient_table if self.objective_weighting == "equal"
                                else _sample_objective_table(self.coefficient_table))
        self.coefficient_table = self.objective_table
        self.objective_calls: List[Dict[str, object]] = []
        self.partition = partition_parameters(self.model)
        assert_expected_parameter_counts(self.partition, self.dataset)
        self.parameter_by_name = dict(self.model.named_parameters())
        # Reachability must not advance the training model's mutable M3OE tau values.
        audit_model = copy.deepcopy(self.model)
        self.reachability = reachability_audit(audit_model, device=self.device, domain_num=3)
        self.gradient_rows: List[Dict[str, object]] = []

    def _record_batch(self, epoch: int, step: int, kind: str,
                      x_batch: torch.Tensor, y_batch: torch.Tensor,
                      objective: object = "") -> str:
        fingerprint = S1Runner._record_batch(self, epoch, step, kind, x_batch, y_batch, objective)
        row = self.batch_rows[-1]
        ids = x_batch[:, -1].detach().cpu().long()
        row.update({
            "stage": STAGE,
            "objective_weighting": self.objective_weighting,
            "shared_access": self.shared_access,
            "target_local_domain": self.target_local_domain,
            "target_count": int((ids == self.target_local_domain).sum()),
            "L_full": "", "L_target": "", "L_non": "", "decomposition_residual": "",
            "applied_shared_gradient_norm": "", "applied_private_gradient_norm": "",
            "applied_architecture_gradient_norm": "",
            "removed_non_target_shared_gradient_norm": "",
            "removed_non_target_architecture_gradient_norm": "",
            "gradient_substitution_valid": "", "gradient_partition_hash": self.partition["partition_hash"],
            "main_optimizer_step_count": self.optimizer_steps,
            "architecture_optimizer_step_count": self.architecture_optimizer_steps,
        })
        return fingerprint

    def _objective_components(self, prediction: torch.Tensor, target: torch.Tensor,
                              domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.objective_calls.append({
            "formal_run_key": self.formal_run_key,
            "objective_weighting": self.objective_weighting,
            "coefficient_table_hash": self.objective_table["coefficient_table_hash"],
        })
        return loss_components(prediction, target, domain_ids, self.target_local_domain,
                               self.objective_weighting, self.objective_table)

    def _record_gradient_event(self, *, epoch: int, step: int, kind: str,
                               fingerprint: str, x_batch: torch.Tensor,
                               losses: Mapping[str, torch.Tensor],
                               gradients: Mapping[str, float]) -> None:
        loss_values = {key: float(losses[key].detach().cpu())
                       for key in ("full", "target", "non_target", "equality_residual")}
        if not all(np.isfinite(value) for value in loss_values.values()):
            raise FloatingPointError("non-finite Stage S3 loss component")
        row = self.batch_rows[-1]
        row.update({
            "L_full": loss_values["full"], "L_target": loss_values["target"],
            "L_non": loss_values["non_target"],
            "decomposition_residual": loss_values["equality_residual"],
            "applied_shared_gradient_norm": gradients.get("applied_shared_gradient_norm", ""),
            "applied_private_gradient_norm": gradients.get("applied_private_gradient_norm", ""),
            "applied_architecture_gradient_norm": gradients.get("applied_architecture_gradient_norm", ""),
            "removed_non_target_shared_gradient_norm": gradients.get(
                "removed_non_target_shared_gradient_norm", ""),
            "removed_non_target_architecture_gradient_norm": gradients.get(
                "removed_non_target_architecture_gradient_norm", ""),
            "gradient_substitution_valid": int(gradients.get("gradient_substitution_valid", 0)),
            "main_optimizer_step_count": self.optimizer_steps,
            "architecture_optimizer_step_count": self.architecture_optimizer_steps,
        })
        ids = x_batch[:, -1].detach().cpu().long().tolist()
        domain_counts = [int(ids.count(domain)) for domain in range(3)]
        self.gradient_rows.append({
            "stage": STAGE, "formal_run_key": self.formal_run_key,
            "objective_weighting": self.objective_weighting,
            "shared_access": self.shared_access, "epoch": int(epoch), "step": int(step),
            "kind": kind, "batch_fingerprint": fingerprint,
            "batch_size": int(len(x_batch)),
            "domain_counts": json.dumps(domain_counts, separators=(",", ":")),
            "L_full": loss_values["full"], "L_target": loss_values["target"],
            "L_non": loss_values["non_target"],
            "decomposition_residual": loss_values["equality_residual"],
            "applied_shared_gradient_norm": gradients.get("applied_shared_gradient_norm", ""),
            "applied_private_gradient_norm": gradients.get("applied_private_gradient_norm", ""),
            "applied_architecture_gradient_norm": gradients.get("applied_architecture_gradient_norm", ""),
            "removed_non_target_shared_gradient_norm": gradients.get(
                "removed_non_target_shared_gradient_norm", ""),
            "removed_non_target_architecture_gradient_norm": gradients.get(
                "removed_non_target_architecture_gradient_norm", ""),
            "gradient_substitution_valid": int(gradients.get("gradient_substitution_valid", 0)),
            "gradient_partition_hash": self.partition["partition_hash"],
            "optimizer_steps": int(self.optimizer_steps),
            "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
        })

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
                raise FloatingPointError("non-finite Stage S3 training prediction")
            losses = self._objective_components(pred, y_device, domain_ids)
            _require_finite("S3_full_loss", losses["full"].detach().cpu())
            _require_finite("S3_target_loss", losses["target"].detach().cpu())
            self.optimizer.zero_grad(set_to_none=True)
            if self.arch_optimizer is not None:
                self.arch_optimizer.zero_grad(set_to_none=True)
            gradient = apply_main_gradients(
                losses["full"], losses["target"], self.parameter_by_name,
                self.partition, access=self.shared_access)
            for name in self.partition["groups"][GROUP_SHARED] + self.partition["groups"][GROUP_PRIVATE]:
                param = self.parameter_by_name[name]
                if param.grad is not None and not bool(torch.isfinite(param.grad).all().detach().cpu()):
                    raise FloatingPointError("non-finite Stage S3 gradient {}".format(name))
            self.optimizer.step()
            self.optimizer_steps += 1
            count = int(len(y_batch))
            self.examples_processed += count
            sample_count += count
            loss_value = float(losses["full"].detach().cpu())
            objective_sum += loss_value * count
            self.batch_rows[-1]["train_objective"] = loss_value
            self._record_gradient_event(epoch=epoch, step=step, kind="main",
                                        fingerprint=fingerprint, x_batch=x_batch,
                                        losses=losses, gradients=gradient)
            if not all(bool(torch.isfinite(param).all().detach().cpu())
                       for param in self.model.parameters()):
                raise FloatingPointError("non-finite Stage S3 parameter after main update")
        train_objective = objective_sum / float(sample_count or 1)
        train_loader = DataLoader(TorchDataset(self.bundle.train_x, self.bundle.train_y),
                                  batch_size=int(self.args.batch_size), shuffle=False,
                                  num_workers=0, drop_last=False)
        train_records = _cell_records(self.bundle, self.model, train_loader, self.device)
        return train_objective, train_records

    def _architecture_update(self, architecture_iterator: Optional[Iterator], epoch: int):
        if self.arch_optimizer is None:
            raise RuntimeError("Stage S3 requires the original M3OE architecture optimizer")
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
        fingerprint = self._record_batch(epoch, -1, "architecture", x_batch, y_batch)
        self.architecture_batch_fingerprints.append(fingerprint)
        self.architecture_examples_processed += int(len(y_batch))
        x_device = x_batch.transpose(0, 1).to(device=self.device, dtype=torch.float32)
        y_device = y_batch.to(device=self.device, dtype=torch.float32)
        domain_ids = x_device[-1, :].long()
        pred = self.model(x_device)
        if not bool(torch.isfinite(pred).all().detach().cpu()):
            raise FloatingPointError("non-finite Stage S3 architecture prediction")
        losses = self._objective_components(pred, y_device, domain_ids)
        self.optimizer.zero_grad(set_to_none=True)
        self.arch_optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        gradient = apply_architecture_gradients(
            losses["full"], losses["target"], self.parameter_by_name,
            self.partition, access=self.shared_access)
        for name in self.partition["groups"][GROUP_ARCH]:
            param = self.parameter_by_name[name]
            if param.grad is None or not bool(torch.isfinite(param.grad).all().detach().cpu()):
                raise FloatingPointError("missing/non-finite architecture gradient {}".format(name))
        self.arch_optimizer.step()
        self.architecture_optimizer_steps += 1
        self.batch_rows[-1]["train_objective"] = float(losses["full"].detach().cpu())
        self._record_gradient_event(epoch=epoch, step=-1, kind="architecture",
                                    fingerprint=fingerprint, x_batch=x_batch,
                                    losses=losses, gradients=gradient)
        _optimizer_audit(self.arch_optimizer)
        if not all(bool(torch.isfinite(param).all().detach().cpu())
                   for name, param in self.model.named_parameters()
                   if name in self.partition["groups"][GROUP_ARCH]):
            raise FloatingPointError("non-finite Stage S3 architecture parameter")
        return architecture_iterator

    def _output_formal_protocol(self):
        return formal_protocol_for(self.fresh_role)

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        # The S2 serializer is the audited S1-compatible artifact writer.  We
        # replace only S3 identity/access fields and add the intervention audit.
        super().write_outputs(fit_info, wall_seconds)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"),
                       [dict(row, stage=STAGE) for row in self.batch_rows], S3_BATCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"),
                       self.gradient_rows, GRADIENT_STEP_FIELDS)
        write_json(os.path.join(self.run_dir, "gradient_partition.json"), self.partition)
        write_json(os.path.join(self.run_dir, "reachability_audit.json"), self.reachability)
        write_json(os.path.join(self.run_dir, "objective_coefficients.json"), self.objective_table)
        inventory = _code_hash_inventory()
        if inventory.get(MODEL_SOURCE_KEY) != _sha256(os.path.join(ROOT, *MODEL_SOURCE_KEY.split("/"))):
            raise RuntimeError("M3OE forward source changed during the S3 run")
        runtime = _runtime_metadata(self.device)
        runtime_fingerprint = hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        protocol = self._output_formal_protocol()
        protocol["objective"] = OBJECTIVE_NAMES[self.objective_weighting]
        protocol["objective_formula"] = OBJECTIVE_FORMULAS[self.objective_weighting]
        code_digest = source_inventory_digest(inventory)
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        config.update({
            "stage": STAGE, "condition": self._condition_name(),
            "method": self.objective_weighting,
            "objective_weighting": self.objective_weighting,
            "objective": OBJECTIVE_NAMES[self.objective_weighting],
            "objective_formula": OBJECTIVE_FORMULAS[self.objective_weighting],
            "shared_access": self.shared_access, "fresh_role": self.fresh_role,
            "formal_run_key": self.formal_run_key,
            "source_formal_run_key": self.source_formal_run_key,
            "co_run_pair_id": str(self.args.co_run_pair_id),
            "planned_peer_run_key": str(self.args.planned_peer_run_key),
            "worker_id": str(self.args.worker_id), "attempt_id": str(self.args.attempt_id),
            "formal_protocol": protocol, "formal_protocol_observed": protocol,
            "formal_protocol_mismatches": {}, "formal_valid": True,
            "code_hashes": inventory, "code_inventory_hash": code_digest,
            "source_code_hash": inventory.get("experiments/stage_S3_run.py", ""),
            "runtime": runtime, "runtime_fingerprint": runtime_fingerprint,
            "target_local_domain": self.target_local_domain,
            "target_physical_domain": str(target_for(self.dataset, self.setting)["physical_domain"]),
            "target_size_role": str(target_for(self.dataset, self.setting)["size_role"]),
            "gradient_partition_hash": self.partition["partition_hash"],
            "gradient_partition": "gradient_partition.json",
            "reachability_audit": "reachability_audit.json",
            "shared_access_formula": self._access_formula(),
            "coefficient_table_hash": self.objective_table["coefficient_table_hash"],
            "objective_coefficients": "objective_coefficients.json",
            "checkpoint_selection_frozen_before_test": True,
            "test_evaluated_during_training": False,
            "test_evaluated_after_selection": True,
            "test_metric_in_epoch_history": False,
            "source_inventory_digest": code_digest,
        })
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, sort_keys=True, indent=2)

        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update({
            "stage": STAGE, "status": "OK", "method": self.objective_weighting,
            "objective_weighting": self.objective_weighting,
            "objective": OBJECTIVE_NAMES[self.objective_weighting],
            "objective_name": OBJECTIVE_NAMES[self.objective_weighting],
            "objective_formula": OBJECTIVE_FORMULAS[self.objective_weighting],
            "condition": self._condition_name(), "shared_access": self.shared_access,
            "fresh_role": self.fresh_role, "formal_run_key": self.formal_run_key,
            "source_formal_run_key": self.source_formal_run_key,
            "source_stage": STAGE,
            "provenance_type": self.fresh_role,
            "target_local_domain": self.target_local_domain,
            "target_physical_domain": str(target_for(self.dataset, self.setting)["physical_domain"]),
            "target_size_role": str(target_for(self.dataset, self.setting)["size_role"]),
            "target_task_names": target_for(self.dataset, self.setting)["tasks"],
            "shared_access_formula": self._access_formula(),
            "gradient_partition_hash": self.partition["partition_hash"],
            "gradient_partition_artifact": "gradient_partition.json",
            "reachability_audit_artifact": "reachability_audit.json",
            "gradient_step_artifact": "gradient_step.csv",
            "gradient_batch_count": len(self.gradient_rows),
            "objective_coefficients": "objective_coefficients.json",
            "coefficient_table_hash": self.objective_table["coefficient_table_hash"],
            "code_hashes": inventory, "code_inventory_hash": code_digest,
            "source_code_hash": inventory.get("experiments/stage_S3_run.py", ""),
            "forward_model_source_sha256": inventory[MODEL_SOURCE_KEY],
            "runtime": runtime, "runtime_fingerprint": runtime_fingerprint,
            "formal_protocol": protocol, "formal_protocol_observed": protocol,
            "formal_protocol_mismatches": {}, "formal_valid": True,
            "objective_call_count": len(self.objective_calls),
            "objective_call_hashes": sorted({str(row["coefficient_table_hash"])
                                              for row in self.objective_calls}),
            "source_inventory_digest": code_digest,
            "access_control_note": (
                "domain_expert and task_expert are cross-domain shared; slot/gate/tower are private"),
            "scope_note": "raw Stage S3 observation; no scientific conclusion",
        })
        for filename in ("config.json", "batch_trace.csv", "gradient_step.csv",
                         "gradient_partition.json", "reachability_audit.json",
                         "objective_coefficients.json"):
            manifest.setdefault("artifact_hashes", {})[filename] = _sha256(
                os.path.join(self.run_dir, filename))
        manifest["artifacts"] = sorted(set(manifest.get("artifacts", [])) | {
            "gradient_step.csv", "gradient_partition.json", "reachability_audit.json"})
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)

    def _condition_name(self) -> str:
        return "{}_full".format(self.objective_weighting) if self.shared_access == FULL \
            else "{}_target_only_shared".format(self.objective_weighting)

    def _access_formula(self) -> str:
        if self.shared_access == FULL:
            return "C=grad(L_full); P=grad(L_full); A_arch=grad(L_full)"
        return "C=grad(L_target); P=grad(L_full); A_arch=grad(L_target)"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S3 one-run M3OE trainer")
    parser.add_argument("--run_type", choices=("joint",), default="joint")
    parser.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--backbone", choices=(BACKBONE,), default=BACKBONE)
    parser.add_argument("--seed", type=int, choices=S3_SEEDS, required=True)
    parser.add_argument("--physical_domain", type=int, default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--learning_rate_darts", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--fcn_dims", nargs=4, type=int, default=[512, 256, 256, 64])
    parser.add_argument("--shadow_patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_protocol_deviation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow_nonformal_seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--co_run_pair_id", required=True)
    parser.add_argument("--planned_peer_run_key", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--attempt_id", required=True)
    parser.add_argument("--objective_weighting", choices=OBJECTIVES, required=True)
    parser.add_argument("--shared_access", choices=(FULL, ACCESS), required=True)
    parser.add_argument("--fresh_role", choices=(FRESH_TARGET_ROLE, FRESH_SAMPLE_FULL_ROLE), required=True)
    parser.add_argument("--formal_run_key", required=True)
    parser.add_argument("--source_formal_run_key", required=True)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--smoke_representative", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    started = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    try:
        if args.smoke_only:
            result = run_s3_update_smoke(
                torch.device(args.device), objective=args.objective_weighting,
                access=args.shared_access, representative=bool(args.smoke_representative))
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0 if result.get("passed") else 1
        runner = S3Runner(args)
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        print("[stage_S3] completed", runner.formal_run_key, flush=True)
        return 0
    except Exception as exc:
        prior_manifest = os.path.join(args.run_dir, "run_manifest.json")
        if not os.path.exists(prior_manifest):
            write_json(prior_manifest, {
                "stage": STAGE, "status": "FAILED", "formal_run_key": args.formal_run_key,
                "objective_weighting": args.objective_weighting,
                "shared_access": args.shared_access, "fresh_role": args.fresh_role,
                "dataset": args.dataset, "setting": args.setting, "seed": int(args.seed),
                "exception_type": type(exc).__name__, "exception": str(exc),
                "co_run_pair_id": args.co_run_pair_id,
                "planned_peer_run_key": args.planned_peer_run_key,
                "worker_id": args.worker_id, "attempt_id": args.attempt_id,
                "wall_seconds": time.time() - started,
            })
        raise


def _make_smoke_model(dataset: str, device: torch.device) -> torch.nn.Module:
    schema = _smoke_feature_schema(dataset)
    return build_model(schema, 3, 2, BACKBONE, expert_num=4,
                       fcn_dims=(512, 256, 256, 64), seed=2022).to(device)


def run_s3_reachability_smoke(dataset: str, *,
                              device: Optional[torch.device] = None) -> Dict[str, object]:
    device = device or torch.device("cpu")
    model = _make_smoke_model(dataset, device)
    model.eval()
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, dataset)
    result = reachability_audit(model, device=device, domain_num=3)
    result["reachability_passed"] = bool(result["passed"])
    return result


def run_s3_update_smoke(device: Optional[torch.device] = None, *, objective: str = "equal",
                        access: str = ACCESS, representative: bool = False) -> Dict[str, object]:
    device = device or torch.device("cpu")
    objective = str(objective).lower()
    if objective not in OBJECTIVES or access not in (FULL, ACCESS):
        raise ValueError("invalid S3 smoke objective/access")
    model = _make_smoke_model("kuairand", device)
    model.train()
    partition = partition_parameters(model)
    assert_expected_parameter_counts(partition, "kuairand")
    audit = reachability_audit(copy.deepcopy(model), device=device)
    counts = {0: 1, 1: 2, 2: 1}
    equal_table = compute_equal_coefficients(counts)
    table = equal_table if objective == "equal" else _sample_objective_table(equal_table)
    batch_size = 40960 if representative else 12
    torch.manual_seed(2022)
    x = torch.zeros((batch_size, len(model.features)), dtype=torch.float32, device=device)
    x[:, -1] = torch.arange(batch_size, device=device) % 3
    y = (torch.arange(batch_size, device=device).reshape(-1, 1)
         + torch.tensor([0, 1], device=device).reshape(1, -1)).remainder(2).float()
    x_input = x.transpose(0, 1)
    domain_ids = x[:, -1].long()
    fingerprint = batch_fingerprint(x.detach().cpu(), y.detach().cpu())

    # FULL is checked against an unmodified direct backward on an identical copy.
    direct_model = copy.deepcopy(model)
    direct_parameters = dict(direct_model.named_parameters())
    direct_partition = partition_parameters(direct_model)
    optimizer_names = list(direct_partition["groups"][GROUP_SHARED]) \
        + list(direct_partition["groups"][GROUP_PRIVATE])
    direct_opt = torch.optim.Adam([direct_parameters[name] for name in optimizer_names],
                                  lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
    model_opt = torch.optim.Adam([dict(model.named_parameters())[name] for name in optimizer_names],
                                 lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
    direct_pred = direct_model(x_input)
    direct_losses = loss_components(direct_pred, y, domain_ids, 1, objective, table)
    direct_opt.zero_grad(set_to_none=True)
    direct_losses["full"].backward()
    direct_opt.step()

    prediction = model(x_input)
    losses = loss_components(prediction, y, domain_ids, 1, objective, table)
    model_opt.zero_grad(set_to_none=True)
    gradient = apply_main_gradients(losses["full"], losses["target"],
                                    dict(model.named_parameters()), partition, access=access)
    model_opt.step()
    main_steps = 1
    full_equivalence = True
    if access == FULL:
        for name, value in direct_model.state_dict().items():
            if not torch.equal(value, model.state_dict()[name]):
                full_equivalence = False
                break
        if full_equivalence:
            for first, second in zip(direct_opt.param_groups, model_opt.param_groups):
                for left, right in zip(first["params"], second["params"]):
                    a, b = direct_opt.state[left], model_opt.state[right]
                    if set(a) != set(b) or any(
                            not torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor)
                            else a[k] != b[k] for k in a):
                        full_equivalence = False
                        break

    # One architecture update from the same routed objective; main parameters
    # are not in this optimizer, matching the frozen M3OE optimizer groups.
    model_parameters = dict(model.named_parameters())
    arch_names = list(partition["groups"][GROUP_ARCH])
    arch_optimizer = torch.optim.Adam([model_parameters[name] for name in arch_names],
                                      lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
    arch_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    arch_prediction = model(x_input)
    arch_losses = loss_components(arch_prediction, y, domain_ids, 1, objective, table)
    arch_gradient = apply_architecture_gradients(
        arch_losses["full"], arch_losses["target"], model_parameters, partition, access=access)
    arch_optimizer.step()
    architecture_steps = 1
    norms = {**gradient, **arch_gradient}
    finite = all(np.isfinite(float(value)) for value in norms.values())
    passed = bool(audit["passed"] and finite and (full_equivalence if access == FULL else True))
    return {
        "passed": passed, "objective": objective, "access": access,
        "main_steps": main_steps, "architecture_steps": architecture_steps,
        "reachability_passed": bool(audit["passed"]),
        "full_equivalence_passed": bool(full_equivalence),
        "batch_size": batch_size, "batch_fingerprint": fingerprint,
        "parameter_count": partition["parameter_count"],
        "partition_hash": partition["partition_hash"],
        "gradient_norms": norms,
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
