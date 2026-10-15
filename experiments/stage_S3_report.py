"""Read-only Stage S3 raw collection and preregistered structural contrasts.

This module emits evidence tables and mechanical summaries only.  It does not
interpret results or draw research conclusions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import stage_S3_orchestrator as orchestrator
from stage_S1_metrics import cell_metrics
from stage_S3_protocol import (
    ACCESS, FULL, CHECKPOINTS, CONTRASTS, EXPECTED_FRESH_RUNS,
    EXPECTED_INTERACTION_ROWS, EXPECTED_INVENTORY_ROWS, EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS,
    EXPECTED_PAIRED_ROWS, EXPECTED_RAW_ROWS, EXPECTED_STRICT_PAIRED_ROWS,
    FRESH_SAMPLE_FULL_ROLE, FRESH_TARGET_ROLE, LEGACY_CONTEXT_ROLE, MODEL_SOURCE_KEY,
    MODEL_SOURCE_SHA256, PRACTICAL_AUC, STRONG_AUC, S3_DATASETS, S3_SEEDS,
    build_fresh_matrix, build_reuse_matrix,
    fresh_formal_key, sample_full_replacement_key, target_for,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_FIELDS = (
    "stage", "condition", "method", "objective_weighting", "shared_access",
    "source_stage", "provenance_type", "target_local_domain", "target_physical_domain",
    "formal_run_key", "source_formal_run_key", "dataset", "setting", "backbone", "seed",
    "physical_domain", "physical_domain_identity", "evaluated_local_domain", "size_role",
    "task", "task_identity", "checkpoint", "evaluated_epoch", "selected_epoch",
    "auc", "logloss", "validation_auc_at_stage", "validation_logloss_at_stage",
    "validation_macro_auc_at_stage", "validation_macro_logloss_at_stage",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "domain_share_p", "loss_coefficient_w", "effective_domain_coefficient_a",
    "effective_cell_coefficient", "objective_name", "objective_formula", "access_formula",
    "gradient_partition_hash",
    "initial_parameter_state_hash", "batch_order_hash", "architecture_batch_order_hash",
    "split_row_id_hash", "physical_view_fingerprint", "dataset_file_hash",
    "checkpoint_hash", "code_inventory_hash", "forward_model_source_sha256",
    "runtime_fingerprint", "strict_source_identity", "status",
)
PAIRED_FIELDS = (
    "comparison", "lhs_condition", "rhs_condition", "dataset", "setting", "backbone",
    "seed", "physical_domain", "physical_domain_identity", "size_role", "task",
    "task_identity", "checkpoint", "evaluated_epoch", "lhs_formal_run_key",
    "rhs_formal_run_key", "lhs_auc", "rhs_auc", "delta_auc", "lhs_logloss",
    "rhs_logloss", "delta_logloss", "lhs_selected_epoch", "rhs_selected_epoch",
    "lhs_checkpoint_hash", "rhs_checkpoint_hash",
    "lhs_forward_model_source_sha256", "rhs_forward_model_source_sha256",
    "contrast_evidence_class",
    "practical_auc_flag", "strong_auc_flag", "raw_auc_negative_flag", "logloss_worse_flag",
    "concordant_flag", "strict_source_identity", "pairing_valid", "pairing_reasons",
)
INTERACTION_FIELDS = (
    "dataset", "setting", "seed", "physical_domain", "physical_domain_identity",
    "task", "checkpoint", "evaluated_epoch", "equal_full_run_key",
    "sample_full_run_key", "equal_target_run_key", "sample_target_run_key",
    "equal_full_checkpoint_hash", "sample_full_checkpoint_hash",
    "equal_target_checkpoint_hash", "sample_target_checkpoint_hash",
    "d_w_full_auc", "d_w_block_auc", "rescue_equal_auc", "rescue_sample_auc",
    "interaction_auc", "d_w_full_neg_logloss", "d_w_block_neg_logloss",
    "rescue_equal_neg_logloss", "rescue_sample_neg_logloss",
    "interaction_neg_logloss", "pairing_valid", "pairing_reasons",
)
RUN_FIELDS = (
    "stage", "condition", "formal_run_key", "source_formal_run_key", "dataset", "setting",
    "backbone", "seed", "run_type", "physical_domain", "source_stage", "provenance_type",
    "reuse_role", "shared_access", "objective_weighting", "strict_source_identity", "status",
    "run_dir", "forward_model_source_sha256", "code_inventory_hash", "gradient_partition_hash",
    "reachability_passed", "artifact_tree_digest", "canonical_tree_digest",
)


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV header missing: {}".format(path))
        return [dict(row) for row in reader]


def _write_json(path: str, value: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        stream.write("\n")


def _write_csv(path: str, rows: Sequence[Mapping[str, object]],
               fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_item(value: object, index: int, field: str) -> object:
    if isinstance(value, Mapping):
        if str(index) in value:
            return value[str(index)]
        if index in value:
            return value[index]
        raise ValueError("{} missing local domain {}".format(field, index))
    if not isinstance(value, (list, tuple)) or len(value) <= index:
        raise ValueError("{} missing or has invalid domain length".format(field))
    return value[index]


def _condition_for_task(task: Mapping[str, object]) -> str:
    if task.get("reuse_role") == LEGACY_CONTEXT_ROLE:
        return "single_domain_mtl_reference"
    if task.get("method") == "equal_full":
        return "equal_full"
    if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE:
        return "sample_full"
    if task.get("fresh_role") == FRESH_TARGET_ROLE:
        return "{}_target_only_shared".format(task.get("objective_weighting"))
    return str(task.get("condition", task.get("method", "")))


def _objective_metadata(task: Mapping[str, object], manifest: Mapping[str, object],
                        local_domain: int, run_dir: str) -> Dict[str, object]:
    coefficient_path = os.path.join(run_dir, "objective_coefficients.json")
    coefficients = _read_json(coefficient_path) if os.path.isfile(coefficient_path) else {}
    domain_shares = manifest.get("domain_shares", coefficients.get("domain_share_p", {}))
    weights = coefficients.get("loss_coefficient_w", manifest.get("loss_coefficient_w", {}))
    effective = coefficients.get("effective_domain_coefficient_a",
                                 manifest.get("effective_domain_coefficient_a", {}))
    cell = coefficients.get("effective_cell_coefficient",
                            manifest.get("effective_cell_coefficient", {}))
    if task.get("reuse_role") == LEGACY_CONTEXT_ROLE:
        return {
            "domain_share_p": 1.0, "loss_coefficient_w": 1.0,
            "effective_domain_coefficient_a": 1.0, "effective_cell_coefficient": 0.5,
            "objective_name": "single_domain_mtl_reference",
            "objective_formula": "mean_i,t BCE(p_it,y_it)",
        }
    def get(source):
        if isinstance(source, Mapping):
            return source.get(str(local_domain), source.get(local_domain, ""))
        return ""
    return {
        "domain_share_p": get(domain_shares), "loss_coefficient_w": get(weights),
        "effective_domain_coefficient_a": get(effective), "effective_cell_coefficient": get(cell),
        "objective_name": manifest.get("objective_name", manifest.get("objective", "")),
        "objective_formula": manifest.get("objective_formula", ""),
    }


def _row_for_metric(task: Mapping[str, object], manifest: Mapping[str, object],
                    metric_row: Mapping[str, object], run_dir: str) -> Dict[str, object]:
    """Convert one S1/S2/S3 deferred-test row into the common S3 raw schema."""
    domain_num = int(manifest.get("domain_num", 3))
    local_domain = int(float(metric_row.get("domain", 0)))
    if not 0 <= local_domain < domain_num:
        raise ValueError("metric local domain outside manifest domain count")
    local_to_physical = manifest.get("local_to_physical")
    identities = manifest.get("physical_domain_identity")
    if local_to_physical is None or identities is None:
        raise ValueError("physical domain mapping/identity missing from manifest")
    mapping_length = len(local_to_physical) if isinstance(
        local_to_physical, (list, tuple, Mapping)) else -1
    identity_length = len(identities) if isinstance(identities, (list, tuple, Mapping)) else -1
    if mapping_length != domain_num or identity_length != domain_num:
        raise ValueError("physical domain mapping/identity has invalid domain length")
    mapped_physical = int(_mapping_item(local_to_physical, local_domain, "local_to_physical"))
    identity = str(_mapping_item(identities, local_domain, "physical_domain_identity"))
    expected_physical = task.get("physical_domain")
    observed_physical = int(float(metric_row.get("physical_domain", mapped_physical)))
    if domain_num == 1 and expected_physical is not None:
        expected_physical = int(expected_physical)
        if mapped_physical != expected_physical or observed_physical != expected_physical:
            raise ValueError("physical ID disagrees with task identity")
        physical_domain = expected_physical
    else:
        if observed_physical != mapped_physical:
            raise ValueError("physical ID disagrees with manifest mapping")
        physical_domain = mapped_physical
        if expected_physical is not None and int(expected_physical) != physical_domain:
            raise ValueError("physical ID disagrees with task identity")
    condition = _condition_for_task(task)
    target = target_for(str(task["dataset"]), str(task["setting"]))
    legacy_context = task.get("reuse_role") == LEGACY_CONTEXT_ROLE
    objective_weighting = "NA" if legacy_context else task.get(
        "objective_weighting", manifest.get("objective_weighting", ""))
    shared_access = "NA" if legacy_context else task.get(
        "shared_access", manifest.get("shared_access", ""))
    access_formula = str(manifest.get("shared_access_formula", ""))
    if legacy_context:
        access_formula = "NA"
    elif not access_formula and shared_access == FULL:
        # S2 Equal FULL predates the S3 instrumentation fields. The condition's
        # full-access routing is fixed by the approved S3 protocol; do not
        # invent a partition hash that was not recorded by that source run.
        access_formula = "C=grad(L_full); P=grad(L_full); A_arch=grad(L_full)"
    source_hashes = manifest.get("code_hashes", {})
    model_hash = (str(source_hashes.get(MODEL_SOURCE_KEY, ""))
                  if isinstance(source_hashes, Mapping) else "")
    strict = (condition != "single_domain_mtl_reference"
              and bool(model_hash) and model_hash == MODEL_SOURCE_SHA256)
    objective = _objective_metadata(task, manifest, local_domain, run_dir)
    split_hashes = manifest.get("split_row_id_hashes", {})
    view_fingerprints = manifest.get("physical_view_fingerprints", {})
    view = view_fingerprints.get(str(physical_domain), {}) \
        if isinstance(view_fingerprints, Mapping) else {}
    if isinstance(view, Mapping):
        view = view.get("test", "")
    checkpoint = str(metric_row.get("checkpoint", ""))
    checkpoints = manifest.get("checkpoints", {})
    checkpoint_filename = {"selected": "selected_checkpoint.pth",
                           "final": "final_checkpoint.pth"}.get(
                               checkpoint, checkpoint + ".pth")
    checkpoint_hash = (checkpoints.get(checkpoint_filename, checkpoints.get(checkpoint, ""))
                       if isinstance(checkpoints, Mapping) else "")
    row = {
        "stage": "S3", "condition": condition, "method": condition,
        "objective_weighting": objective_weighting, "shared_access": shared_access,
        "source_stage": task.get("source_stage", manifest.get("stage", "")),
        "provenance_type": task.get("provenance_type", task.get("fresh_role", "reused")),
        "target_local_domain": task.get("target_local_domain", target["local_domain"]),
        "target_physical_domain": task.get("target_physical_domain", target["physical_domain"]),
        "formal_run_key": str(task.get("formal_run_key", manifest.get("formal_run_key", ""))),
        "source_formal_run_key": str(task.get("source_formal_run_key",
                                               task.get("formal_run_key", ""))),
        "dataset": str(task["dataset"]), "setting": str(task["setting"]),
        "backbone": str(task.get("backbone", manifest.get("backbone", "m3oe"))),
        "seed": int(task["seed"]), "physical_domain": physical_domain,
        "physical_domain_identity": identity, "evaluated_local_domain": local_domain,
        "size_role": metric_row.get("size_role", ""),
        "task": int(float(metric_row.get("task", 0))),
        "task_identity": metric_row.get("task_identity", ""),
        "checkpoint": checkpoint, "evaluated_epoch": metric_row.get("stage_epoch", ""),
        "selected_epoch": manifest.get("selected_epoch", metric_row.get("selected_epoch", "")),
        "auc": metric_row.get("auc", ""), "logloss": metric_row.get("logloss", ""),
        "validation_auc_at_stage": metric_row.get("validation_auc_at_stage", ""),
        "validation_logloss_at_stage": metric_row.get("validation_logloss_at_stage", ""),
        "validation_macro_auc_at_stage": metric_row.get("validation_macro_auc_at_stage", ""),
        "validation_macro_logloss_at_stage": metric_row.get("validation_macro_logloss_at_stage", ""),
        "train_sample_count": metric_row.get("train_sample_count", ""),
        "validation_sample_count": metric_row.get("validation_sample_count", ""),
        "test_sample_count": metric_row.get("test_sample_count", ""),
        **objective,
        "access_formula": access_formula,
        "gradient_partition_hash": manifest.get("gradient_partition_hash", ""),
        "initial_parameter_state_hash": manifest.get("initial_parameter_state_hash", ""),
        "batch_order_hash": manifest.get("batch_order_hash", ""),
        "architecture_batch_order_hash": manifest.get("architecture_batch_order_hash", ""),
        "split_row_id_hash": split_hashes.get("test", "") if isinstance(split_hashes, Mapping) else "",
        "physical_view_fingerprint": view, "dataset_file_hash": manifest.get("dataset_file_hash", ""),
        "checkpoint_hash": checkpoint_hash,
        "code_inventory_hash": manifest.get("code_inventory_hash", _digest(source_hashes)),
        "forward_model_source_sha256": model_hash,
        "runtime_fingerprint": manifest.get("runtime_fingerprint", ""),
        "strict_source_identity": bool(strict), "status": manifest.get("status", ""),
    }
    return row


def _prediction_metric_reasons(run_dir: str, manifest: Mapping[str, object],
                               metric_rows: Sequence[Mapping[str, object]]) -> List[str]:
    reasons = []
    path = os.path.join(run_dir, "predictions_test.npz")
    if not os.path.isfile(path):
        return ["missing artifact predictions_test.npz"]
    try:
        with np.load(path, allow_pickle=False) as payload:
            labels = np.asarray(payload["labels"])
            physical = np.asarray(payload["physical_domain"])
            if labels.ndim != 2 or labels.shape[1] != 2 or physical.shape != (labels.shape[0],):
                return ["prediction_shape_invalid"]
            domain_num = int(manifest.get("domain_num", 3))
            mapping = manifest.get("local_to_physical", [])
            mapping_length = len(mapping) if isinstance(mapping, (list, tuple, Mapping)) else -1
            if mapping_length != domain_num:
                return ["prediction_domain_mapping_invalid"]
            rows = {
                (str(row.get("checkpoint")), int(float(row.get("task", 0))),
                 int(float(row.get("domain", 0)))): row
                for row in metric_rows
            }
            for checkpoint in CHECKPOINTS:
                key = "pred_" + checkpoint
                if key not in payload.files:
                    reasons.append("prediction_key_missing:" + checkpoint)
                    continue
                prediction = np.asarray(payload[key])
                if prediction.shape != labels.shape or not np.isfinite(prediction).all():
                    reasons.append("prediction_shape_or_finite_invalid:" + checkpoint)
                    continue
                for local in range(domain_num):
                    physical_id = int(_mapping_item(mapping, local, "local_to_physical"))
                    mask = physical == physical_id
                    for task_id in range(2):
                        row = rows.get((checkpoint, task_id, local))
                        if row is None:
                            reasons.append("prediction_metric_row_missing:{}:{}:{}".format(
                                checkpoint, local, task_id))
                            continue
                        values = cell_metrics(labels[mask, task_id], prediction[mask, task_id])
                        for field in ("auc", "logloss"):
                            try:
                                if abs(float(row[field]) - float(values[field])) > 1e-10:
                                    reasons.append("prediction_metric_mismatch:{}:{}:{}:{}".format(
                                        checkpoint, local, task_id, field))
                            except (KeyError, TypeError, ValueError):
                                reasons.append("prediction_metric_non_numeric:{}:{}:{}:{}".format(
                                    checkpoint, local, task_id, field))
    except Exception as exc:
        reasons.append("prediction_metric_validation_failed:{}".format(type(exc).__name__))
    return sorted(set(reasons))


def _artifact_raw_rows(task: Mapping[str, object], run_dir: str,
                       manifest: Mapping[str, object]) -> Tuple[List[Dict[str, object]], List[str]]:
    metric_rows = _read_csv(os.path.join(run_dir, "metrics_test.csv"))
    if not metric_rows:
        return [], ["metrics_test_missing"]
    reasons = _prediction_metric_reasons(run_dir, manifest, metric_rows)
    rows = [_row_for_metric(task, manifest, row, run_dir) for row in metric_rows]
    return rows, reasons


def _raw_identity(row: Mapping[str, object]) -> Tuple[object, ...]:
    return tuple(row.get(field) for field in (
        "dataset", "setting", "seed", "condition", "physical_domain", "task", "checkpoint"))


def _index_raw(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[object, ...], Mapping[str, object]]:
    result = {}
    for row in rows:
        key = _raw_identity(row)
        if key in result:
            raise ValueError("duplicate Stage S3 raw identity: {}".format(key))
        result[key] = row
    return result


def _numeric(row: Optional[Mapping[str, object]], field: str) -> float:
    if row is None:
        return float("nan")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("non-finite metric {} for {}".format(field, _raw_identity(row)))
    return value


def _identity_reasons(left: Mapping[str, object], right: Mapping[str, object],
                      *, strict: bool) -> List[str]:
    reasons = []
    for field in ("dataset", "setting", "seed", "physical_domain", "task",
                  "checkpoint", "physical_domain_identity"):
        if left.get(field) != right.get(field):
            reasons.append(field + "_mismatch")
    if strict:
        left_hash, right_hash = (str(left.get("forward_model_source_sha256", "")),
                                 str(right.get("forward_model_source_sha256", "")))
        if not left_hash or not right_hash or left_hash != right_hash \
                or left_hash != MODEL_SOURCE_SHA256:
            reasons.append("model_source_hash_mismatch")
        # Manifest-level pairing has already checked the complete held-constant
        # inventory. The row-level source-identity gate is the nonempty forward
        # hash; sparse synthetic rows remain useful for table-shape tests.
    return sorted(set(reasons))


def _paired_row(comparison: str, lhs: Optional[Mapping[str, object]],
                rhs: Optional[Mapping[str, object]], *, dataset: str, setting: str,
                seed: int, physical: int, task: int, checkpoint: str,
                evidence_class: str, strict: bool) -> Dict[str, object]:
    reasons = []
    if lhs is None or rhs is None:
        reasons.append("paired_cell_missing")
    else:
        reasons.extend(_identity_reasons(lhs, rhs, strict=strict))
    left_auc, right_auc = _numeric(lhs, "auc"), _numeric(rhs, "auc")
    left_ll, right_ll = _numeric(lhs, "logloss"), _numeric(rhs, "logloss")
    if not (math.isfinite(left_auc) and math.isfinite(right_auc)
            and math.isfinite(left_ll) and math.isfinite(right_ll)):
        reasons.append("paired_metric_invalid")
    delta_auc = left_auc - right_auc
    delta_ll = left_ll - right_ll
    return {
        "comparison": comparison, "lhs_condition": lhs.get("condition", "") if lhs else "",
        "rhs_condition": rhs.get("condition", "") if rhs else "",
        "dataset": dataset, "setting": setting, "backbone": "m3oe", "seed": seed,
        "physical_domain": physical,
        "physical_domain_identity": lhs.get("physical_domain_identity", "") if lhs else "",
        "size_role": lhs.get("size_role", "") if lhs else "", "task": task,
        "task_identity": lhs.get("task_identity", "") if lhs else "",
        "checkpoint": checkpoint, "evaluated_epoch": lhs.get("evaluated_epoch", "") if lhs else "",
        "lhs_formal_run_key": lhs.get("formal_run_key", "") if lhs else "",
        "rhs_formal_run_key": rhs.get("formal_run_key", "") if rhs else "",
        "lhs_auc": left_auc, "rhs_auc": right_auc, "delta_auc": delta_auc,
        "lhs_logloss": left_ll, "rhs_logloss": right_ll, "delta_logloss": delta_ll,
        "lhs_selected_epoch": lhs.get("selected_epoch", "") if lhs else "",
        "rhs_selected_epoch": rhs.get("selected_epoch", "") if rhs else "",
        "lhs_checkpoint_hash": lhs.get("checkpoint_hash", "") if lhs else "",
        "rhs_checkpoint_hash": rhs.get("checkpoint_hash", "") if rhs else "",
        "lhs_forward_model_source_sha256": (
            lhs.get("forward_model_source_sha256", "") if lhs else ""),
        "rhs_forward_model_source_sha256": (
            rhs.get("forward_model_source_sha256", "") if rhs else ""),
        "contrast_evidence_class": evidence_class,
        "practical_auc_flag": int(math.isfinite(delta_auc) and delta_auc <= -PRACTICAL_AUC),
        "strong_auc_flag": int(math.isfinite(delta_auc) and delta_auc <= -STRONG_AUC),
        "raw_auc_negative_flag": int(math.isfinite(delta_auc) and delta_auc < 0.0),
        "logloss_worse_flag": int(math.isfinite(delta_ll) and delta_ll > 0.0),
        "concordant_flag": int(math.isfinite(delta_auc) and math.isfinite(delta_ll)
                                and delta_auc < 0.0 and delta_ll > 0.0),
        "strict_source_identity": bool(strict),
        "pairing_valid": int(not reasons), "pairing_reasons": ";".join(reasons),
    }


def build_paired_rows(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index = _index_raw(raw_rows)
    rows = []
    definitions = (
        ("equal_full_minus_sd", "equal_full", "single_domain_mtl_reference", False),
        ("sample_full_minus_sd", "sample_full", "single_domain_mtl_reference", False),
        ("equal_target_only_shared_minus_sd", "equal_target_only_shared",
         "single_domain_mtl_reference", False),
        ("sample_target_only_shared_minus_sd", "sample_target_only_shared",
         "single_domain_mtl_reference", False),
        ("equal_full_minus_sample_full", "equal_full", "sample_full", True),
        ("equal_target_only_shared_minus_sample_target_only_shared",
         "equal_target_only_shared", "sample_target_only_shared", True),
        ("equal_target_only_shared_minus_equal_full",
         "equal_target_only_shared", "equal_full", True),
        ("sample_target_only_shared_minus_sample_full",
         "sample_target_only_shared", "sample_full", True),
    )
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            for physical in range(3):
                for task_id in range(2):
                    for checkpoint in CHECKPOINTS:
                        common = (dataset, setting, seed)
                        for comparison, left, right, strict in definitions:
                            lhs = index.get(common + (left, physical, task_id, checkpoint))
                            rhs = index.get(common + (right, physical, task_id, checkpoint))
                            if left == "single_domain_mtl_reference" and rhs is None:
                                # A D=1 run is keyed by its physical domain, not joint local ID.
                                candidates = [value for key, value in index.items()
                                              if key[:3] == common and key[3] == "single_domain_mtl_reference"
                                              and key[4] == physical and key[5] == task_id
                                              and key[6] == checkpoint]
                                rhs = candidates[0] if len(candidates) == 1 else None
                            if right == "single_domain_mtl_reference" and rhs is None:
                                candidates = [value for key, value in index.items()
                                              if key[:3] == common and key[3] == "single_domain_mtl_reference"
                                              and key[4] == physical and key[5] == task_id
                                              and key[6] == checkpoint]
                                rhs = candidates[0] if len(candidates) == 1 else None
                            evidence = ("strict_joint_source_identity" if strict
                                        else "legacy_sd_context")
                            paired = _paired_row(comparison, lhs, rhs, dataset=dataset,
                                                 setting=setting, seed=seed, physical=physical,
                                                 task=task_id, checkpoint=checkpoint,
                                                 evidence_class=evidence, strict=strict)
                            if lhs is not None and rhs is not None \
                                    and lhs.get("physical_domain_identity") != rhs.get(
                                        "physical_domain_identity"):
                                raise ValueError("physical-domain reference missing or identity mismatch")
                            rows.append(paired)
    if len(rows) != EXPECTED_PAIRED_ROWS:
        raise AssertionError("Stage S3 paired cardinality {} != {}".format(
            len(rows), EXPECTED_PAIRED_ROWS))
    return rows


def build_interaction_rows(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index = _index_raw(raw_rows)
    rows = []
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            for physical in range(3):
                for task_id in range(2):
                    for checkpoint in CHECKPOINTS:
                        common = (dataset, setting, seed)
                        names = ("equal_full", "sample_full",
                                 "equal_target_only_shared", "sample_target_only_shared")
                        cells = [index.get(common + (name, physical, task_id, checkpoint))
                                 for name in names]
                        reasons = []
                        if any(cell is None for cell in cells):
                            reasons.append("four_way_cell_missing")
                        for cell in cells:
                            if cell is not None and (
                                not cell.get("strict_source_identity")
                                or cell.get("forward_model_source_sha256") != MODEL_SOURCE_SHA256
                            ):
                                reasons.append("model_source_hash_mismatch")
                        values_auc = [_numeric(cell, "auc") for cell in cells]
                        values_ll = [-_numeric(cell, "logloss") for cell in cells]
                        e_full, s_full, e_target, s_target = values_auc
                        e_qf, s_qf, e_qt, s_qt = values_ll
                        d_full = e_full - s_full
                        d_block = e_target - s_target
                        r_equal, r_sample = e_target - e_full, s_target - s_full
                        interaction = d_block - d_full
                        q_d_full, q_d_block = e_qf - s_qf, e_qt - s_qt
                        q_r_equal, q_r_sample = e_qt - e_qf, s_qt - s_qf
                        rows.append({
                            "dataset": dataset, "setting": setting, "seed": seed,
                            "physical_domain": physical,
                            "physical_domain_identity": cells[0].get(
                                "physical_domain_identity", "") if cells[0] else "",
                            "task": task_id, "checkpoint": checkpoint,
                            "evaluated_epoch": cells[0].get("evaluated_epoch", "") if cells[0] else "",
                            "equal_full_run_key": cells[0].get("formal_run_key", "") if cells[0] else "",
                            "sample_full_run_key": cells[1].get("formal_run_key", "") if cells[1] else "",
                            "equal_target_run_key": cells[2].get("formal_run_key", "") if cells[2] else "",
                            "sample_target_run_key": cells[3].get("formal_run_key", "") if cells[3] else "",
                            "equal_full_checkpoint_hash": cells[0].get("checkpoint_hash", "") if cells[0] else "",
                            "sample_full_checkpoint_hash": cells[1].get("checkpoint_hash", "") if cells[1] else "",
                            "equal_target_checkpoint_hash": cells[2].get("checkpoint_hash", "") if cells[2] else "",
                            "sample_target_checkpoint_hash": cells[3].get("checkpoint_hash", "") if cells[3] else "",
                            "d_w_full_auc": d_full, "d_w_block_auc": d_block,
                            "rescue_equal_auc": r_equal, "rescue_sample_auc": r_sample,
                            "interaction_auc": interaction,
                            "d_w_full_neg_logloss": q_d_full,
                            "d_w_block_neg_logloss": q_d_block,
                            "rescue_equal_neg_logloss": q_r_equal,
                            "rescue_sample_neg_logloss": q_r_sample,
                            "interaction_neg_logloss": q_r_equal - q_r_sample,
                            "pairing_valid": int(not reasons),
                            "pairing_reasons": ";".join(sorted(set(reasons))),
                            "contrast_evidence_class": "strict_joint_source_identity",
                        })
    if len(rows) != EXPECTED_INTERACTION_ROWS:
        raise AssertionError("Stage S3 interaction cardinality {} != {}".format(
            len(rows), EXPECTED_INTERACTION_ROWS))
    return rows


def validate_fresh_reuse_pair(full_task: Mapping[str, object],
                              target_task: Mapping[str, object]) -> List[str]:
    """Pair helper accepting either task paths or already-loaded manifests."""
    if "run_dir" not in full_task or "run_dir" not in target_task:
        full, target = full_task, target_task
        fields = (
            "dataset_file_hash", "dataset_identity", "local_to_physical",
            "physical_domain_identity", "split_row_fingerprints", "split_row_id_hashes",
            "base_split_fingerprints", "base_split_row_id_hashes", "physical_view_fingerprints",
            "preprocessing_metadata", "features", "train_domain_counts", "domain_shares",
            "initial_parameter_state_hash", "model_parameter_manifest", "forward_flops_estimate",
            "batch_size", "batches_per_epoch", "examples_processed",
            "architecture_examples_processed", "optimizer_steps", "architecture_optimizer_steps",
            "batch_order_hash", "architecture_batch_order_hash", "batch_fingerprint_count",
            "architecture_batch_fingerprint_count", "optimizer", "architecture_optimizer",
            "fixed_final_epoch", "common_epochs", "metric_implementation",
            "checkpoint_selection_frozen_before_test", "test_evaluated_during_training",
            "test_metric_in_epoch_history", "runtime_fingerprint",
        )
        return ["pairing_field_mismatch:{}".format(field) for field in fields
                if _digest(full.get(field)) != _digest(target.get(field))]
    try:
        full = _read_json(os.path.join(str(full_task["run_dir"]), "run_manifest.json"))
        target = _read_json(os.path.join(str(target_task["run_dir"]), "run_manifest.json"))
    except Exception as exc:
        return ["fresh_full_pair_manifest_invalid:{}".format(type(exc).__name__)]
    reasons = orchestrator._compare_fresh_full_pair(full, target)
    full_hashes, target_hashes = full.get("code_hashes", {}), target.get("code_hashes", {})
    full_hash = full_hashes.get(MODEL_SOURCE_KEY, "") if isinstance(full_hashes, Mapping) else ""
    target_hash = target_hashes.get(MODEL_SOURCE_KEY, "") if isinstance(target_hashes, Mapping) else ""
    if not full_hash or not target_hash or full_hash != target_hash \
            or full_hash != MODEL_SOURCE_SHA256:
        reasons.append("model_source_hash_mismatch")
    return sorted(set(reasons))


def build_run_inventory(fresh_tasks: Sequence[Mapping[str, object]],
                        reuse_tasks: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for task in list(fresh_tasks) + list(reuse_tasks):
        is_fresh = task in fresh_tasks
        run_dir = str(task["run_dir"] if is_fresh else task["source_run_dir"])
        row = {
            "stage": "S3", "condition": _condition_for_task(task),
            "formal_run_key": str(task.get("formal_run_key", "")),
            "source_formal_run_key": str(task.get("source_formal_run_key", "")),
            "dataset": str(task["dataset"]), "setting": str(task["setting"]),
            "backbone": str(task.get("backbone", "m3oe")), "seed": int(task["seed"]),
            "run_type": str(task.get("run_type", "joint")),
            "physical_domain": task.get("physical_domain", ""),
            "source_stage": str(task.get("source_stage", "S3" if is_fresh else "")),
            "provenance_type": str(task.get("provenance_type", "fresh" if is_fresh else "reused")),
            "reuse_role": str(task.get("reuse_role", "")), "run_dir": run_dir,
            "strict_source_identity": bool(task.get("reuse_role") != LEGACY_CONTEXT_ROLE),
            "shared_access": str(task.get("shared_access", "")),
            "objective_weighting": str(task.get("objective_weighting", "")),
            "status": "PLANNED" if is_fresh else "REUSED",
        }
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if os.path.isfile(manifest_path):
            manifest = _read_json(manifest_path)
            hashes = manifest.get("code_hashes", {})
            row["forward_model_source_sha256"] = (
                hashes.get(MODEL_SOURCE_KEY, "") if isinstance(hashes, Mapping) else "")
            row["code_inventory_hash"] = manifest.get("code_inventory_hash", "")
            row["gradient_partition_hash"] = manifest.get("gradient_partition_hash", "")
            row["reachability_passed"] = manifest.get("reachability_audit", "")
            row["artifact_tree_digest"] = orchestrator._tree_digest(run_dir)
            row["status"] = str(manifest.get("status", row["status"]))
        rows.append(row)
    if len(rows) != EXPECTED_INVENTORY_ROWS:
        raise AssertionError("Stage S3 inventory cardinality {} != {}".format(
            len(rows), EXPECTED_INVENTORY_ROWS))
    keys = [str(row["formal_run_key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Stage S3 run inventory key")
    return rows


def _aggregate_rows(paired: Sequence[Mapping[str, object]],
                    interactions: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for row in paired:
        if int(row.get("pairing_valid", 0)):
            key = (row["comparison"], row["dataset"], row["setting"], row["physical_domain"],
                   row["task"], row["checkpoint"], row.get("physical_domain_identity", ""),
                   row.get("size_role", ""))
            groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        auc = [float(row["delta_auc"]) for row in rows]
        ll = [float(row["delta_logloss"]) for row in rows]
        target_identity = str(target_for(str(key[1]), str(key[2]))["physical_domain"])
        worst = min(rows, key=lambda row: float(row["delta_auc"]))
        output.append({
            "comparison": key[0], "dataset": key[1], "setting": key[2],
            "physical_domain": key[3], "task": key[4], "checkpoint": key[5],
            "physical_domain_identity": key[6], "size_role": key[7],
            "target_domain_indicator": int(str(key[6]) == target_identity),
            "seed_count": len(rows), "delta_auc_mean": statistics.mean(auc),
            "delta_auc_sd": statistics.stdev(auc) if len(auc) > 1 else 0.0,
            "delta_auc_min": min(auc), "delta_auc_max": max(auc),
            "negative_auc_count": sum(value < 0 for value in auc),
            "positive_auc_count": sum(value > 0 for value in auc),
            "four_of_four_negative": int(len(auc) == 4 and all(value < 0 for value in auc)),
            "worst_seed": worst.get("seed", ""),
            "delta_logloss_mean": statistics.mean(ll),
            "delta_logloss_sd": statistics.stdev(ll) if len(ll) > 1 else 0.0,
            "delta_logloss_min": min(ll), "delta_logloss_max": max(ll),
            "auc_logloss_sign_concordant": int(statistics.mean(auc) < 0
                                                and statistics.mean(ll) > 0),
            "positive_logloss_count": sum(value > 0 for value in ll),
            "pairing_valid_count": len(rows),
        })
    for row in interactions:
        if int(row.get("pairing_valid", 0)):
            output.append({key: row.get(key, "") for key in (
                "dataset", "setting", "physical_domain", "task", "checkpoint",
                "interaction_auc", "interaction_neg_logloss", "seed")})
    return output


def _selected_epoch_summaries(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_run: Dict[Tuple[object, ...], Mapping[str, object]] = {}
    for row in raw_rows:
        if row.get("checkpoint") != "selected":
            continue
        key = (row.get("formal_run_key"), row.get("condition"), row.get("dataset"),
               row.get("setting"), row.get("seed"))
        by_run.setdefault(key, row)
    groups = defaultdict(list)
    for (_run_key, condition, dataset, setting, _seed), row in by_run.items():
        try:
            groups[(condition, dataset, setting)].append(int(float(row["selected_epoch"])))
        except (KeyError, TypeError, ValueError):
            raise ValueError("selected checkpoint row has no valid selected_epoch")
    return [{
        "condition": key[0], "dataset": key[1], "setting": key[2],
        "run_count": len(epochs), "selected_epoch_mean": statistics.mean(epochs),
        "selected_epoch_sd": statistics.stdev(epochs) if len(epochs) > 1 else 0.0,
        "selected_epoch_min": min(epochs), "selected_epoch_max": max(epochs),
        "selected_epoch_values": "|".join(str(value) for value in sorted(epochs)),
    } for key, epochs in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))]


def collect_and_write(raw_root: str, s1_raw_root: str, s2_raw_root: str, *,
                      output_root: Optional[str] = None, require_complete: bool = True,
                      queue_manifest: Optional[str] = None) -> Dict[str, object]:
    """Validate and collect frozen S3 inputs without modifying any raw artifacts."""
    raw_root = os.path.abspath(raw_root)
    output_root = os.path.abspath(output_root or HERE)
    fresh = build_fresh_matrix(raw_root)
    reuse = build_reuse_matrix(s1_raw_root, s2_raw_root)
    queue_path = os.path.abspath(queue_manifest or os.path.join(HERE, "stage_S3_queue_manifest.json"))
    if require_complete and not os.path.isfile(queue_path):
        raise ValueError("amended Stage S3 queue manifest is missing")
    queue = _read_json(queue_path) if os.path.isfile(queue_path) else {}
    if queue:
        queue_reasons = orchestrator._validate_amendment_queue_history(queue)
        if queue_reasons:
            raise ValueError("Stage S3 queue integrity failure: {}".format(queue_reasons))
        if queue.get("amendment") != "A1" or int(queue.get("expected_run_count", 0)) != EXPECTED_FRESH_RUNS:
            raise ValueError("Stage S3 queue is not the amended A1 queue")
        if require_complete and queue.get("resume_state") != "COMPLETE":
            raise ValueError("Stage S3 queue has not completed every frozen launch group")
    records = {str(row.get("formal_run_key")): row for row in queue.get("records", [])}
    if require_complete and set(records) != {str(task["formal_run_key"]) for task in fresh}:
        raise ValueError("Stage S3 queue record key support is incomplete or contains extras")
    artifact_sources = (orchestrator._artifact_source_inventory(queue) if queue
                        else orchestrator.source_inventory(orchestrator.ROOT))
    raw_rows: List[Dict[str, object]] = []
    run_reasons = []
    # Current accepted historical references are Equal FULL and S1 D=1 only.
    inputs = [(task, str(task["run_dir"]), True) for task in fresh]
    inputs += [(task, str(task["source_run_dir"]), False) for task in reuse]
    reuse_audit = orchestrator.audit_reuse_tasks(
        s1_raw_root=s1_raw_root, s2_raw_root=s2_raw_root, require_formal=True)
    if not reuse_audit.get("valid"):
        raise ValueError("Stage S3 historical reuse/source audit is not valid")
    reuse_status = {str(row.get("source_formal_run_key", "")): str(row.get("status", ""))
                    for row in reuse_audit.get("rows", [])}
    for task, run_dir, is_fresh in inputs:
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.isfile(manifest_path):
            run_reasons.append("{}:run_manifest_missing".format(task["formal_run_key"]))
            continue
        manifest = _read_json(manifest_path)
        if is_fresh:
            record = records.get(str(task["formal_run_key"]))
            if record is not None and record.get("revalidation_receipt_digest"):
                try:
                    from stage_S3_amendment_recovery import _validate_member
                    _validate_member(run_dir, task, artifact_sources,
                                     allow_legacy_empty_object=True,
                                     require_formal=False)
                    reasons = orchestrator._coefficient_artifact_reasons(run_dir, manifest)
                    valid = not reasons
                except Exception as exc:
                    valid = False
                    reasons = ["recovered_artifact_validation_failed:{}".format(
                        type(exc).__name__)]
            else:
                valid, reasons = orchestrator.validate_fresh_run(
                    task, require_formal=True,
                    expected_source_inventory=artifact_sources)
            if not valid:
                run_reasons.extend("{}:{}".format(task["formal_run_key"], reason)
                                   for reason in reasons)
            if require_complete and (record is None or record.get("artifact_valid") is not True):
                run_reasons.append("{}:queue_record_missing_or_invalid".format(
                    task["formal_run_key"]))
            if require_complete and record is not None:
                accepted_digest = str(record.get("canonical_tree_digest", ""))
                if not accepted_digest or accepted_digest != orchestrator._tree_digest(run_dir):
                    run_reasons.append("{}:canonical_tree_digest_mismatch".format(
                        task["formal_run_key"]))
        else:
            expected_status = ("REUSE_OK" if task.get("reuse_role") == "controlled_full_source"
                               else "LEGACY_CONTEXT_OK")
            if reuse_status.get(str(task.get("source_formal_run_key", ""))) != expected_status:
                run_reasons.append("{}:historical_reuse_audit_status_invalid".format(
                    task["formal_run_key"]))
        test_metrics = _read_csv(os.path.join(run_dir, "metrics_test.csv"))
        raw_rows.extend(_row_for_metric(task, manifest, metric, run_dir) for metric in test_metrics)
        run_reasons.extend("{}:{}".format(task["formal_run_key"], reason)
                           for reason in _prediction_metric_reasons(run_dir, manifest, test_metrics))
    fresh_by_key = {str(task["formal_run_key"]): task for task in fresh}
    equal_reuse = {(str(task["dataset"]), str(task["setting"]), int(task["seed"])): task
                   for task in reuse if task.get("reuse_role") == "controlled_full_source"}
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            sample_target = fresh_by_key[fresh_formal_key("sample", dataset, setting, seed)]
            sample_full = fresh_by_key[sample_full_replacement_key(dataset, setting, seed)]
            reasons = orchestrator.validate_fresh_reuse_pair(
                dict(sample_full, target_only_run_dir=str(sample_target["run_dir"])),
                s1_raw_root, s2_raw_root,
                target_only_run_dir=str(sample_target["run_dir"]))[1]
            run_reasons.extend("{}:{}".format(sample_full["formal_run_key"], reason)
                               for reason in reasons)
            equal_target = fresh_by_key[fresh_formal_key("equal", dataset, setting, seed)]
            equal_full = equal_reuse[(dataset, setting, seed)]
            equal_pair_task = dict(equal_full, run_dir=str(equal_full["source_run_dir"]),
                                   target_only_run_dir=str(equal_target["run_dir"]))
            reasons = orchestrator.validate_fresh_reuse_pair(
                equal_pair_task, s1_raw_root, s2_raw_root,
                target_only_run_dir=str(equal_target["run_dir"]))[1]
            run_reasons.extend("{}:{}".format(equal_target["formal_run_key"], reason)
                               for reason in reasons)
            try:
                four_way = orchestrator.validate_strict_four_way_block(
                    dataset, setting, seed, fresh, run_dirs={
                        key: str(task["run_dir"]) for key, task in fresh_by_key.items()},
                    s1_root=s1_raw_root, s2_root=s2_raw_root)
                if not four_way.get("strict_source_identity") or not four_way.get("pairing_valid"):
                    run_reasons.append("{}|{}|{}:strict_four_way_invalid".format(
                        dataset, setting, seed))
            except Exception as exc:
                run_reasons.append("{}|{}|{}:strict_four_way_invalid:{}".format(
                    dataset, setting, seed, type(exc).__name__))
    if run_reasons:
        prediction_failures = [reason for reason in run_reasons
                               if ":prediction_metric_" in reason]
        if prediction_failures:
            raise ValueError("prediction-level metric validation failed: {}".format(
                "; ".join(prediction_failures[:20])))
        raise ValueError("Stage S3 artifact validation failed: {}".format("; ".join(run_reasons[:30])))
    if len(raw_rows) != EXPECTED_RAW_ROWS:
        raise ValueError("Stage S3 raw cardinality {} != {}".format(len(raw_rows), EXPECTED_RAW_ROWS))
    if any(row.get("condition") == "sample_full"
           and str(row.get("source_stage", "")) == "S1" for row in raw_rows):
        raise ValueError("revoked historical S1 Sample FULL artifacts cannot enter S3 raw rows")
    paired = build_paired_rows(raw_rows)
    interactions = build_interaction_rows(raw_rows)
    inventory = build_run_inventory(fresh, reuse)
    strict_count = sum(int(row["strict_source_identity"]) for row in paired)
    legacy_count = sum(row["contrast_evidence_class"] == "legacy_sd_context" for row in paired)
    invalid_strict = [row for row in paired if row["strict_source_identity"]
                      and not int(row["pairing_valid"])]
    invalid_interactions = [row for row in interactions if not int(row["pairing_valid"])]
    if strict_count != EXPECTED_STRICT_PAIRED_ROWS or legacy_count != EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS:
        raise ValueError("Stage S3 evidence-class contrast cardinality mismatch")
    if invalid_strict or invalid_interactions:
        raise ValueError("Stage S3 strict pairing failed: {} strict, {} interaction rows".format(
            len(invalid_strict), len(invalid_interactions)))
    aggregates = _aggregate_rows(paired, interactions)
    paths = {
        "raw": os.path.join(output_root, "stage_S3_raw_results.csv"),
        "paired": os.path.join(output_root, "stage_S3_paired_results.csv"),
        "interactions": os.path.join(output_root, "stage_S3_interactions.csv"),
        "runs": os.path.join(output_root, "stage_S3_runs.csv"),
        "aggregates": os.path.join(output_root, "stage_S3_aggregates.csv"),
        "manifest": os.path.join(output_root, "stage_S3_summary_manifest.json"),
    }
    _write_csv(paths["raw"], raw_rows, RAW_FIELDS)
    _write_csv(paths["paired"], paired, PAIRED_FIELDS)
    _write_csv(paths["interactions"], interactions, INTERACTION_FIELDS)
    _write_csv(paths["runs"], inventory, RUN_FIELDS)
    aggregate_fields = list(dict.fromkeys(key for row in aggregates for key in row))
    _write_csv(paths["aggregates"], aggregates, aggregate_fields)
    cardinalities = {
        "raw": len(raw_rows), "paired": len(paired), "interactions": len(interactions),
        "inventory": len(inventory), "strict_paired": strict_count,
        "legacy_context_paired": legacy_count,
    }
    manifest = {
        "stage": "S3", "amendment": "A1", "status": "STRUCTURALLY_VALIDATED",
        "cardinalities": cardinalities, "paths": paths,
        "strict_invalid_count": len(invalid_strict),
        "interaction_invalid_count": len(invalid_interactions),
        "reuse_audit_summary": reuse_audit.get("summary", {}),
        "selected_epoch_summaries": _selected_epoch_summaries(raw_rows),
        "source_inventory": orchestrator.source_inventory(orchestrator.ROOT),
        "source_inventory_digest": orchestrator.source_inventory_digest(
            orchestrator.source_inventory(orchestrator.ROOT)),
        "scientific_interpretation": None,
    }
    _write_json(paths["manifest"], manifest)
    return {"cardinalities": cardinalities, "paths": paths, "manifest": manifest}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Collect Stage S3 raw evidence")
    parser.add_argument("--raw-root", default=os.path.join(HERE, "stage_S3_raw_results"))
    parser.add_argument("--s1-raw-root", default=os.path.join(HERE, "stage_S1_raw_results"))
    parser.add_argument("--s2-raw-root", default=os.path.join(HERE, "stage_S2_raw_results"))
    parser.add_argument("--output-root", default=HERE)
    parser.add_argument("--queue-manifest", default=os.path.join(HERE, "stage_S3_queue_manifest.json"))
    args = parser.parse_args()
    result = collect_and_write(args.raw_root, args.s1_raw_root, args.s2_raw_root,
                               output_root=args.output_root, queue_manifest=args.queue_manifest)
    print(json.dumps(result["cardinalities"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
