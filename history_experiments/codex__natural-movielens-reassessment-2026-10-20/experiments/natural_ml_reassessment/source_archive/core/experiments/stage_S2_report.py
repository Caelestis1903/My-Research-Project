"""Mechanical Stage S2 artifact audit and structural table construction.

This module does not interpret signs or decide whether negative transfer is
present.  It validates frozen S1 reuse, validates fresh S2 artifacts, copies
the shared S1 metrics into a common long-form schema, constructs preregistered
pairwise deltas, and emits cardinality/provenance summaries for the later
analysis stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from stage_S1_metrics import cell_metrics, task_domain_cell_index
from stage_S2_orchestrator import (
    audit_reuse_tasks,
    build_reuse_matrix,
    build_tasks,
    validate_s2_pairings,
    validate_equal_run,
    validate_s1_reuse_task,
    write_reuse_audit,
)
from stage_S2_objectives import compute_equal_coefficients
from stage_S2_protocol import (
    BACKBONE,
    COMMON_EPOCHS,
    DATASETS,
    EPOCHS,
    PHYSICAL_DOMAINS,
    SEEDS,
    STAGE,
    THRESHOLDS,
    canonical_dataset,
    domain_shares,
    expected_train_counts,
)


HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS: Tuple[str, ...] = ("selected", "final") + tuple(
    "common_epoch_{}".format(epoch) for epoch in COMMON_EPOCHS)
RAW_FIELDS: Tuple[str, ...] = (
    "stage", "method", "source_stage", "provenance_type", "formal_run_key",
    "source_formal_run_key", "dataset", "setting", "backbone", "seed",
    "physical_domain", "physical_domain_identity", "size_role", "task",
    "task_identity", "checkpoint", "evaluated_epoch", "selected_epoch", "auc",
    "logloss", "validation_auc_at_stage", "validation_logloss_at_stage",
    "validation_macro_auc_at_stage", "validation_macro_logloss_at_stage",
    "train_sample_count", "validation_sample_count", "test_sample_count",
    "domain_share_p", "loss_coefficient_w", "effective_domain_coefficient_a",
    "effective_cell_coefficient", "objective_name", "objective_formula",
    "initial_parameter_state_hash", "batch_order_hash", "architecture_batch_order_hash",
    "split_row_id_hash", "physical_view_fingerprint", "dataset_file_hash",
    "checkpoint_hash", "code_inventory_hash", "runtime_fingerprint", "status",
)
PAIRED_FIELDS: Tuple[str, ...] = (
    "comparison", "lhs_method", "rhs_method", "dataset", "setting", "backbone", "seed",
    "physical_domain", "physical_domain_identity", "size_role", "task", "task_identity",
    "checkpoint", "evaluated_epoch", "lhs_formal_run_key", "rhs_formal_run_key",
    "lhs_auc", "rhs_auc", "delta_auc", "lhs_logloss", "rhs_logloss", "delta_logloss",
    "practical_auc_flag", "strong_auc_flag", "raw_auc_negative_flag", "logloss_worse_flag",
    "concordant_flag", "lhs_selected_epoch", "rhs_selected_epoch", "lhs_checkpoint_hash",
    "rhs_checkpoint_hash", "pairing_valid", "pairing_reasons",
)


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except (OSError, csv.Error):
        return [], []


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def _metric_artifact_reasons(run_dir: str, manifest: Mapping[str, object],
                             test_rows: Sequence[Mapping[str, object]]) -> List[str]:
    """Recompute deferred AUC/LogLoss from row-aligned predictions."""

    reasons: List[str] = []
    prediction_path = os.path.join(run_dir, "predictions_test.npz")
    try:
        with np.load(prediction_path, allow_pickle=False) as payload:
            labels = np.asarray(payload["labels"])
            physical = np.asarray(payload["physical_domain"])
            if labels.ndim != 2 or labels.shape[1] != 2 or physical.ndim != 1 \
                    or labels.shape[0] != physical.shape[0]:
                return ["prediction_shape_invalid"]
            local_to_physical = [int(value) for value in manifest.get("local_to_physical", [])]
            domain_num = int(manifest.get("domain_num", 0))
            if domain_num not in (1, 3) or len(local_to_physical) != domain_num:
                return ["prediction_domain_mapping_invalid"]
            row_map = {(str(row.get("checkpoint")), int(float(row.get("task"))),
                       int(float(row.get("domain")))): row for row in test_rows}
            for checkpoint in CHECKPOINTS:
                key = "pred_" + checkpoint
                if key not in payload.files:
                    reasons.append("prediction_key_missing:" + checkpoint)
                    continue
                prediction = np.asarray(payload[key])
                if prediction.shape != labels.shape or not np.isfinite(prediction).all():
                    reasons.append("prediction_shape_or_finite_invalid:" + checkpoint)
                    continue
                for task in (0, 1):
                    for domain in range(domain_num):
                        physical_domain = local_to_physical[domain]
                        mask = physical == physical_domain
                        if not mask.any():
                            reasons.append("prediction_domain_empty:{}".format(domain))
                            continue
                        metric = cell_metrics(labels[mask, task], prediction[mask, task])
                        row = row_map.get((checkpoint, task, domain))
                        if row is None:
                            reasons.append("prediction_metric_row_missing:{}:{}:{}".format(checkpoint, task, domain))
                            continue
                        for field in ("auc", "logloss"):
                            try:
                                if abs(float(row[field]) - float(metric[field])) > 1e-10:
                                    reasons.append("prediction_metric_mismatch:{}:{}:{}:{}".format(
                                        checkpoint, task, domain, field))
                            except (KeyError, TypeError, ValueError):
                                reasons.append("prediction_metric_non_numeric:{}:{}:{}:{}".format(
                                    checkpoint, task, domain, field))
    except Exception as exc:
        reasons.append("prediction_artifact_invalid:{}".format(type(exc).__name__))
    return reasons


def _method_metadata(task: Mapping[str, object], manifest: Mapping[str, object],
                    coefficients: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    method = str(task.get("method"))
    domain_num = int(manifest.get("domain_num", 3))
    if method == "equal":
        if coefficients is None:
            coefficients = _read_json(os.path.join(str(task["run_dir"]), "objective_coefficients.json"))
        p = coefficients.get("domain_share_p", {})
        w = coefficients.get("loss_coefficient_w", {})
        a = coefficients.get("effective_domain_coefficient_a", {})
        cell = coefficients.get("effective_cell_coefficient", {})
        objective = str(coefficients.get("objective_name", "equal_domain_task_mean_bce"))
        formula = str(coefficients.get("objective_formula", ""))
    elif method == "sample":
        p = manifest.get("domain_shares", {})
        w = {str(domain): 1.0 for domain in range(domain_num)}
        a = {str(domain): float(p.get(str(domain), p.get(domain, 0.0))) for domain in range(domain_num)}
        cell = {str(domain): float(a[str(domain)]) / 2.0 for domain in range(domain_num)}
        objective = "direct_per_example_task_mean_bce"
        formula = "mean_i,t BCE(p_it,y_it)"
    else:
        p = {"0": 1.0}; w = {"0": 1.0}; a = {"0": 1.0}; cell = {"0": 0.5}
        objective = "single_domain_mtl_reference"
        formula = "mean_i,t BCE(p_it,y_it)"
    code_hashes = manifest.get("code_hashes", {})
    return {
        "objective_name": objective, "objective_formula": formula,
        "domain_share_p": p, "loss_coefficient_w": w,
        "effective_domain_coefficient_a": a, "effective_cell_coefficient": cell,
        "code_inventory_hash": _hash_json(code_hashes),
        "domain_num": domain_num,
    }


def _value_for_domain(value: object, local_domain: int, default: object = "") -> object:
    if isinstance(value, Mapping):
        if str(local_domain) in value:
            return value[str(local_domain)]
        if local_domain in value:  # type: ignore[operator]
            return value[local_domain]  # type: ignore[index]
    return default


def checkpoint_hash_for_label(checkpoints: Mapping[str, object], checkpoint: str) -> object:
    """Map report labels to the actual checkpoint artifact filenames."""

    filename = {"selected": "selected_checkpoint.pth", "final": "final_checkpoint.pth"}.get(
        str(checkpoint), str(checkpoint) + ".pth")
    return checkpoints.get(filename, checkpoints.get(str(checkpoint), ""))


def _raw_rows_for_artifact(task: Mapping[str, object], run_dir: str,
                           manifest: Mapping[str, object]) -> Tuple[List[Dict[str, object]], List[str]]:
    fields, rows = _read_csv(os.path.join(run_dir, "metrics_test.csv"))
    reasons: List[str] = []
    if not rows:
        return [], ["metrics_test_missing"]
    coefficients = _read_json(os.path.join(run_dir, "objective_coefficients.json")) \
        if task.get("method") == "equal" else None
    metadata = _method_metadata(task, manifest, coefficients)
    prediction_reasons = _metric_artifact_reasons(run_dir, manifest, rows)
    reasons.extend(prediction_reasons)
    formal_key = str(manifest.get("formal_run_key", ""))
    source_key = str(task.get("source_formal_run_key", ""))
    if task.get("provenance_type") == "reused":
        source_key = formal_key
    physical_identities = manifest.get("physical_domain_identity", [])
    selected_epoch = manifest.get("selected_epoch", "")
    checkpoints = manifest.get("checkpoints", {})
    # Construct the emitted rows without exposing internal audit fields in the
    # CSV.  All metrics come from the shared S1 deferred-test artifact.
    output: List[Dict[str, object]] = []
    for row in rows:
        local_domain = int(float(row.get("domain", 0)))
        physical_domain = int(float(row.get("physical_domain", 0)))
        task_id = int(float(row.get("task", 0)))
        checkpoint = str(row.get("checkpoint"))
        p_value = _value_for_domain(metadata["domain_share_p"], local_domain)
        w_value = _value_for_domain(metadata["loss_coefficient_w"], local_domain)
        a_value = _value_for_domain(metadata["effective_domain_coefficient_a"], local_domain)
        cell_value = _value_for_domain(metadata["effective_cell_coefficient"], local_domain)
        identity = (physical_identities[local_domain] if isinstance(physical_identities, list)
                    and local_domain < len(physical_identities) else str(physical_domain))
        split_ids = manifest.get("split_row_id_hashes", {})
        physical_views = manifest.get("physical_view_fingerprints", {})
        view = physical_views.get(str(physical_domain), {}) if isinstance(physical_views, Mapping) else {}
        physical_view = view.get("test", "") if isinstance(view, Mapping) else ""
        checkpoint_hash = checkpoint_hash_for_label(checkpoints, checkpoint) \
            if isinstance(checkpoints, Mapping) else ""
        output.append({
            "stage": STAGE, "method": task["method"], "source_stage": task.get("source_stage", STAGE),
            "provenance_type": task.get("provenance_type", "fresh"), "formal_run_key": formal_key,
            "source_formal_run_key": source_key, "dataset": task["dataset"], "setting": task["setting"],
            "backbone": BACKBONE, "seed": int(task["seed"]), "physical_domain": physical_domain,
            "physical_domain_identity": identity, "size_role": row.get("size_role", ""),
            "task": task_id, "task_identity": row.get("task_identity", "task{}".format(task_id)),
            "checkpoint": checkpoint, "evaluated_epoch": row.get("stage_epoch", ""),
            "selected_epoch": selected_epoch, "auc": row.get("auc", ""), "logloss": row.get("logloss", ""),
            "validation_auc_at_stage": row.get("validation_auc_at_stage", ""),
            "validation_logloss_at_stage": row.get("validation_logloss_at_stage", ""),
            "validation_macro_auc_at_stage": row.get("validation_macro_auc_at_stage", ""),
            "validation_macro_logloss_at_stage": row.get("validation_macro_logloss_at_stage", ""),
            "train_sample_count": row.get("train_sample_count", ""),
            "validation_sample_count": row.get("validation_sample_count", ""),
            "test_sample_count": row.get("test_sample_count", ""), "domain_share_p": p_value,
            "loss_coefficient_w": w_value, "effective_domain_coefficient_a": a_value,
            "effective_cell_coefficient": cell_value, "objective_name": metadata["objective_name"],
            "objective_formula": metadata["objective_formula"],
            "initial_parameter_state_hash": manifest.get("initial_parameter_state_hash", ""),
            "batch_order_hash": manifest.get("batch_order_hash", ""),
            "architecture_batch_order_hash": manifest.get("architecture_batch_order_hash", ""),
            "split_row_id_hash": split_ids.get("test", "") if isinstance(split_ids, Mapping) else "",
            "physical_view_fingerprint": physical_view, "dataset_file_hash": manifest.get("dataset_file_hash", ""),
            "checkpoint_hash": checkpoint_hash, "code_inventory_hash": metadata["code_inventory_hash"],
            "runtime_fingerprint": manifest.get("runtime_fingerprint", ""), "status": manifest.get("status", ""),
        })
    return output, reasons


def _index_raw(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[object, ...], Mapping[str, object]]:
    index: Dict[Tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("setting"), row.get("seed"), row.get("physical_domain"),
               row.get("task"), row.get("checkpoint"), row.get("method"))
        if key in index:
            raise ValueError("duplicate raw cell key: {}".format(key))
        index[key] = row
    return index


def build_paired_rows(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    index = _index_raw(raw_rows)
    paired: List[Dict[str, object]] = []
    for dataset in DATASETS:
        settings = ("d0_large", "d1_large", "d2_large") if dataset == "movielens" else ("natural",)
        for setting in settings:
            for seed in SEEDS:
                for physical_domain in PHYSICAL_DOMAINS:
                    for task in (0, 1):
                        for checkpoint in CHECKPOINTS:
                            base_key = (dataset, setting, seed, physical_domain, task, checkpoint)
                            comparisons = (
                                ("equal_vs_single_domain", "equal", "single_domain_mtl_reference", physical_domain),
                                ("sample_vs_single_domain", "sample", "single_domain_mtl_reference", physical_domain),
                                ("equal_vs_sample", "equal", "sample", physical_domain),
                            )
                            for comparison, lhs_method, rhs_method, domain_key in comparisons:
                                lhs = index.get(base_key + (lhs_method,))
                                rhs = index.get(base_key + (rhs_method,))
                                reasons: List[str] = []
                                if lhs is None or rhs is None:
                                    reasons.append("paired_cell_missing")
                                if lhs is not None and rhs is not None:
                                    if lhs.get("dataset_file_hash") != rhs.get("dataset_file_hash"):
                                        reasons.append("dataset_file_hash_mismatch")
                                    if lhs.get("physical_domain") != rhs.get("physical_domain"):
                                        reasons.append("physical_domain_mismatch")
                                    if lhs.get("physical_domain_identity") != rhs.get("physical_domain_identity"):
                                        reasons.append("physical_domain_identity_mismatch")
                                    if lhs.get("physical_view_fingerprint") != rhs.get("physical_view_fingerprint"):
                                        reasons.append("physical_view_fingerprint_mismatch")
                                    if lhs.get("task_identity") != rhs.get("task_identity"):
                                        reasons.append("task_identity_mismatch")
                                    # The joint Equal/Sample pair shares the
                                    # physical loader and initialization.  A
                                    # D=3 joint run and a D=1 reference do
                                    # not, by design, share batch order or
                                    # optimizer-step structure.
                                    if lhs_method == "equal" and rhs_method == "sample":
                                        for field in ("initial_parameter_state_hash",
                                                      "batch_order_hash",
                                                      "architecture_batch_order_hash"):
                                            if lhs.get(field) != rhs.get(field):
                                                reasons.append(field + "_mismatch")
                                        if lhs.get("split_row_id_hash") != rhs.get("split_row_id_hash"):
                                            reasons.append("split_row_id_hash_mismatch")
                                left_auc = float(lhs["auc"]) if lhs is not None else float("nan")
                                right_auc = float(rhs["auc"]) if rhs is not None else float("nan")
                                left_ll = float(lhs["logloss"]) if lhs is not None else float("nan")
                                right_ll = float(rhs["logloss"]) if rhs is not None else float("nan")
                                delta_auc = left_auc - right_auc
                                delta_ll = left_ll - right_ll
                                paired.append({
                                    "comparison": comparison, "lhs_method": lhs_method, "rhs_method": rhs_method,
                                    "dataset": dataset, "setting": setting, "backbone": BACKBONE, "seed": seed,
                                    "physical_domain": physical_domain,
                                    "physical_domain_identity": lhs.get("physical_domain_identity", "") if lhs else "",
                                    "size_role": lhs.get("size_role", "") if lhs else "",
                                    "task": task, "task_identity": lhs.get("task_identity", "") if lhs else "",
                                    "checkpoint": checkpoint,
                                    "evaluated_epoch": lhs.get("evaluated_epoch", "") if lhs else "",
                                    "lhs_formal_run_key": lhs.get("formal_run_key", "") if lhs else "",
                                    "rhs_formal_run_key": rhs.get("formal_run_key", "") if rhs else "",
                                    "lhs_auc": left_auc, "rhs_auc": right_auc, "delta_auc": delta_auc,
                                    "lhs_logloss": left_ll, "rhs_logloss": right_ll, "delta_logloss": delta_ll,
                                    "practical_auc_flag": int(math.isfinite(delta_auc) and delta_auc <= THRESHOLDS["practical_auc"]),
                                    "strong_auc_flag": int(math.isfinite(delta_auc) and delta_auc <= THRESHOLDS["strong_auc"]),
                                    "raw_auc_negative_flag": int(math.isfinite(delta_auc) and delta_auc < 0.0),
                                    "logloss_worse_flag": int(math.isfinite(delta_ll) and delta_ll > 0.0),
                                    "concordant_flag": int(math.isfinite(delta_auc) and math.isfinite(delta_ll)
                                                           and delta_auc < 0.0 and delta_ll > 0.0),
                                    "lhs_selected_epoch": lhs.get("selected_epoch", "") if lhs else "",
                                    "rhs_selected_epoch": rhs.get("selected_epoch", "") if rhs else "",
                                    "lhs_checkpoint_hash": lhs.get("checkpoint_hash", "") if lhs else "",
                                    "rhs_checkpoint_hash": rhs.get("checkpoint_hash", "") if rhs else "",
                                    "pairing_valid": int(not reasons and math.isfinite(delta_auc) and math.isfinite(delta_ll)),
                                    "pairing_reasons": ";".join(reasons),
                                })
    expected = 3 * 96 * len(CHECKPOINTS)
    if len(paired) != expected:
        raise AssertionError("paired row cardinality {} != {}".format(len(paired), expected))
    return paired


def aggregate_paired_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Mechanical four-seed summaries; no interpretation is included."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        if int(row.get("pairing_valid", 0)) != 1:
            continue
        key = (row.get("comparison"), row.get("dataset"), row.get("setting"), row.get("physical_domain"),
               row.get("task"), row.get("checkpoint"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        auc = [float(row["delta_auc"]) for row in rows if math.isfinite(float(row["delta_auc"]))]
        ll = [float(row["delta_logloss"]) for row in rows if math.isfinite(float(row["delta_logloss"]))]
        output.append({
            "comparison": key[0], "dataset": key[1], "setting": key[2],
            "physical_domain": key[3], "task": key[4], "checkpoint": key[5],
            "seed_count": len(rows), "delta_auc_mean": statistics.mean(auc) if auc else "",
            "delta_auc_sd": statistics.stdev(auc) if len(auc) > 1 else 0.0 if auc else "",
            "delta_auc_min": min(auc) if auc else "", "delta_auc_max": max(auc) if auc else "",
            "negative_auc_count": sum(value < 0 for value in auc),
            "four_of_four_negative": int(len(auc) == 4 and all(value < 0 for value in auc)),
            "worst_seed": (min(rows, key=lambda row: float(row["delta_auc"])).get("seed", "")
                           if auc else ""),
            "worst_group": "{}|{}|d{}|t{}|{}".format(
                key[1], key[2], key[3], key[4], key[5]),
            "delta_logloss_mean": statistics.mean(ll) if ll else "",
            "delta_logloss_sd": statistics.stdev(ll) if len(ll) > 1 else 0.0 if ll else "",
            "delta_logloss_min": min(ll) if ll else "", "delta_logloss_max": max(ll) if ll else "",
            "positive_logloss_count": sum(value > 0 for value in ll),
            "auc_logloss_sign_concordant": int(auc and ll and statistics.mean(auc) < 0 and statistics.mean(ll) > 0),
            "pairing_valid_count": sum(int(row.get("pairing_valid", 0)) for row in rows),
        })
    return output


def role_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Mechanical fixed-physical-domain/size-role rotation summaries."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        if int(row.get("pairing_valid", 0)) != 1:
            continue
        key = (row.get("comparison"), row.get("dataset"), row.get("setting"),
               row.get("physical_domain"), row.get("physical_domain_identity"),
               row.get("size_role"), row.get("task"), row.get("checkpoint"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        auc = [float(row["delta_auc"]) for row in rows if math.isfinite(float(row["delta_auc"]))]
        ll = [float(row["delta_logloss"]) for row in rows if math.isfinite(float(row["delta_logloss"]))]
        output.append({
            "comparison": key[0], "dataset": key[1], "setting": key[2],
            "physical_domain": key[3], "physical_domain_identity": key[4],
            "size_role": key[5], "task": key[6], "checkpoint": key[7],
            "seed_count": len(rows), "delta_auc_mean": statistics.mean(auc) if auc else "",
            "delta_logloss_mean": statistics.mean(ll) if ll else "",
            "negative_auc_count": sum(value < 0 for value in auc),
            "positive_logloss_count": sum(value > 0 for value in ll),
            "pairing_valid_count": sum(int(row.get("pairing_valid", 0)) for row in rows),
        })
    return output


def fixed_physical_rotation_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Summarize each physical domain across all allowed role rotations."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        if int(row.get("pairing_valid", 0)) != 1:
            continue
        key = (row.get("comparison"), row.get("dataset"), row.get("physical_domain"),
               row.get("physical_domain_identity"), row.get("task"), row.get("checkpoint"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        auc = [float(row["delta_auc"]) for row in rows if math.isfinite(float(row["delta_auc"]))]
        ll = [float(row["delta_logloss"]) for row in rows if math.isfinite(float(row["delta_logloss"]))]
        settings = sorted({str(row.get("setting")) for row in rows})
        rotation_scope = ("cross_setting_fixed_physical_domain"
                          if str(key[1]) == "movielens"
                          else "natural_fixed_physical_domain")
        output.append({
            "comparison": key[0], "dataset": key[1], "physical_domain": key[2],
            "physical_domain_identity": key[3], "task": key[4], "checkpoint": key[5],
            "rotation_scope": rotation_scope,
            "settings_seen": "|".join(settings), "setting_count": len(settings),
            "seed_setting_row_count": len(rows),
            "delta_auc_mean": statistics.mean(auc) if auc else "",
            "delta_logloss_mean": statistics.mean(ll) if ll else "",
            "negative_auc_count": sum(value < 0 for value in auc),
            "positive_logloss_count": sum(value > 0 for value in ll),
            "pairing_valid_count": len(rows),
        })
    return output


def paired_training_epoch_mean_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compute one four-seed mean for each paired group/common epoch."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    common_labels = {"common_epoch_{}".format(epoch): epoch for epoch in COMMON_EPOCHS}
    for row in paired_rows:
        if int(row.get("pairing_valid", 0)) != 1:
            continue
        checkpoint = str(row.get("checkpoint"))
        if checkpoint not in common_labels:
            continue
        key = (row.get("comparison"), row.get("dataset"), row.get("setting"),
               row.get("physical_domain"), row.get("task"), common_labels[checkpoint])
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        by_seed = {int(row["seed"]): row for row in rows}
        auc = [float(row["delta_auc"]) for row in by_seed.values()
               if math.isfinite(float(row["delta_auc"]))]
        ll = [float(row["delta_logloss"]) for row in by_seed.values()
              if math.isfinite(float(row["delta_logloss"]))]
        output.append({
            "comparison": key[0], "dataset": key[1], "setting": key[2],
            "physical_domain": key[3], "task": key[4], "common_epoch": key[5],
            "seed_count": len(by_seed), "expected_seed_count": len(SEEDS),
            "delta_auc_mean": statistics.mean(auc) if len(auc) == len(SEEDS) else "",
            "delta_logloss_mean": statistics.mean(ll) if len(ll) == len(SEEDS) else "",
            "seed_delta_auc": json.dumps({str(seed): by_seed[seed]["delta_auc"]
                                            for seed in sorted(by_seed)}, sort_keys=True),
            "seed_delta_logloss": json.dumps({str(seed): by_seed[seed]["delta_logloss"]
                                                for seed in sorted(by_seed)}, sort_keys=True),
            "pairing_valid_count": len(rows),
        })
    return output


def paired_training_stage_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Apply the preregistered 3/5 gate to five four-seed epoch means."""

    epoch_rows = paired_training_epoch_mean_rows(paired_rows)
    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in epoch_rows:
        key = (row.get("comparison"), row.get("dataset"), row.get("setting"),
               row.get("physical_domain"), row.get("task"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        complete = [row for row in rows if int(row.get("seed_count", 0)) == len(SEEDS)
                    and row.get("delta_auc_mean", "") != ""]
        auc_means = [float(row["delta_auc_mean"]) for row in complete]
        ll_means = [float(row["delta_logloss_mean"]) for row in complete
                    if row.get("delta_logloss_mean", "") != ""]
        epoch_mean_map = {str(int(row["common_epoch"])): row["delta_auc_mean"]
                          for row in complete}
        output.append({
            "comparison": key[0], "dataset": key[1], "setting": key[2],
            "physical_domain": key[3], "task": key[4],
            "common_epoch_count": len(complete), "expected_common_epoch_count": len(COMMON_EPOCHS),
            "expected_seed_count_per_epoch": len(SEEDS),
            "negative_matched_epoch_auc_count": sum(value < 0 for value in auc_means),
            "positive_matched_epoch_logloss_count": sum(value > 0 for value in ll_means),
            "stage_support_threshold": 3,
            "stage_support_3_of_5": int(len(auc_means) == len(COMMON_EPOCHS)
                                         and sum(value < 0 for value in auc_means) >= 3),
            "epoch_mean_delta_auc": json.dumps(epoch_mean_map, sort_keys=True),
            "delta_auc_mean": statistics.mean(auc_means) if auc_means else "",
            "delta_logloss_mean": statistics.mean(ll_means) if ll_means else "",
            "pairing_valid_count": sum(int(row.get("pairing_valid_count", 0)) for row in complete),
        })
    return output


def difficulty_summary_rows(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Expose independent-reference validation difficulty mechanically."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("method") != "single_domain_mtl_reference":
            continue
        key = (row.get("dataset"), row.get("setting"), row.get("physical_domain"),
               row.get("physical_domain_identity"), row.get("size_role"), row.get("task"),
               row.get("checkpoint"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        auc = [float(row["validation_macro_auc_at_stage"]) for row in rows]
        ll = [float(row["validation_macro_logloss_at_stage"]) for row in rows]
        output.append({
            "dataset": key[0], "setting": key[1], "physical_domain": key[2],
            "physical_domain_identity": key[3], "size_role": key[4], "task": key[5],
            "checkpoint": key[6], "seed_count": len(rows),
            "reference_validation_macro_auc_mean": statistics.mean(auc) if auc else "",
            "reference_validation_macro_logloss_mean": statistics.mean(ll) if ll else "",
            "difficulty_evaluation_split": "validation",
        })
    return output


def method_stage_summary_rows(raw_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Mechanical method/checkpoint macro, worst-domain, and worst-cell views."""

    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        key = (row.get("method"), row.get("dataset"), row.get("setting"), row.get("seed"), row.get("checkpoint"))
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        auc = [float(row["auc"]) for row in rows]
        ll = [float(row["logloss"]) for row in rows]
        per_domain = {int(row["physical_domain"]): [float(row["auc"]) for row in rows
                                                      if int(row["physical_domain"]) == domain]
                      for domain in PHYSICAL_DOMAINS}
        domain_means = [statistics.mean(values) for values in per_domain.values() if values]
        output.append({
            "method": key[0], "dataset": key[1], "setting": key[2], "seed": key[3],
            "checkpoint": key[4], "cell_count": len(rows),
            "macro_auc": statistics.mean(auc) if auc else "",
            "macro_logloss": statistics.mean(ll) if ll else "",
            "worst_domain_auc": min(domain_means) if domain_means else "",
            "worst_cell_auc": min(auc) if auc else "",
        })
    return output


def _run_inventory_rows(equal_tasks: Sequence[Mapping[str, object]], reuse_tasks: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task in list(equal_tasks) + list(reuse_tasks):
        if task.get("provenance_type") == "reused":
            valid, reasons = validate_s1_reuse_task(task, require_formal=True)
            path = str(task["source_run_dir"])
            key = task["source_formal_run_key"]
        else:
            valid, reasons = validate_equal_run(task, require_formal=True)
            path = str(task["run_dir"])
            key = task["formal_run_key"]
        rows.append({
            "stage": STAGE, "method": task["method"], "provenance_type": task["provenance_type"],
            "formal_run_key": key, "dataset": task["dataset"], "setting": task["setting"],
            "backbone": BACKBONE, "seed": task["seed"], "run_type": task["run_type"],
            "physical_domain": task.get("physical_domain", ""), "run_dir": path,
            "status": "OK" if valid else "INVALID", "validation_reasons": ";".join(reasons),
        })
    return rows


def collect(*, s2_raw_root: str = os.path.join(HERE, "stage_S2_raw_results"),
            s1_raw_root: str = os.path.join(HERE, "stage_S1_raw_results"),
            output_root: str = HERE, require_complete: bool = True,
            write_audit: bool = True) -> Dict[str, object]:
    """Collect validated raw artifacts and emit all structural S2 tables."""

    equal_tasks = build_tasks(s2_raw_root)
    reuse_tasks = build_reuse_matrix(s1_raw_root)
    audit = audit_reuse_tasks(reuse_tasks, require_formal=True)
    if write_audit and not os.path.exists(os.path.join(output_root, "stage_S2_reuse_audit.csv")):
        write_reuse_audit(audit, output_root)
    if not audit.get("all_reuse_ok"):
        raise RuntimeError("S2 reuse audit rejected one or more frozen S1 keys")
    pairing_audit = validate_s2_pairings(equal_tasks, reuse_tasks)
    if require_complete and not pairing_audit.get("valid"):
        raise RuntimeError("S2 Equal/Sample or frozen S1 block pairing failed")
    inventory = _run_inventory_rows(equal_tasks, reuse_tasks)
    if require_complete and any(row["status"] != "OK" for row in inventory if row["provenance_type"] == "fresh"):
        raise RuntimeError("S2 Equal artifacts are incomplete or invalid; report is fail-closed")

    raw_rows: List[Dict[str, object]] = []
    artifact_errors: List[Dict[str, object]] = []
    for task in equal_tasks:
        path = str(task["run_dir"])
        manifest = _read_json(os.path.join(path, "run_manifest.json"))
        if not manifest:
            artifact_errors.append({"formal_run_key": task["formal_run_key"], "reasons": ["manifest_missing"]})
            continue
        rows, reasons = _raw_rows_for_artifact(task, path, manifest)
        if reasons:
            artifact_errors.append({"formal_run_key": task["formal_run_key"], "reasons": reasons})
        raw_rows.extend(rows)
    for task in reuse_tasks:
        path = str(task["source_run_dir"])
        manifest = _read_json(os.path.join(path, "run_manifest.json"))
        rows, reasons = _raw_rows_for_artifact(task, path, manifest)
        if reasons:
            artifact_errors.append({"formal_run_key": task["source_formal_run_key"], "reasons": reasons})
        raw_rows.extend(rows)
    if require_complete and artifact_errors:
        raise RuntimeError("S2 raw artifact validation failed: {}".format(artifact_errors[:3]))
    if require_complete:
        expected_raw = 3 * 4 * 4 * 3 * 2 * len(CHECKPOINTS)
        if len(raw_rows) != expected_raw:
            raise RuntimeError("S2 raw cardinality {} != {}".format(len(raw_rows), expected_raw))
        # Ensure exact key uniqueness before pairing.
        _index_raw(raw_rows)
    expected_raw = 3 * 4 * 4 * 3 * 2 * len(CHECKPOINTS)
    paired_rows = build_paired_rows(raw_rows) if len(raw_rows) == expected_raw else []
    invalid_pair_rows = [row for row in paired_rows if int(row.get("pairing_valid", 0)) != 1]
    if require_complete and invalid_pair_rows:
        raise RuntimeError("S2 paired rows contain invalid pairings; refusing aggregation: {}".format(
            invalid_pair_rows[:3]))
    aggregate_rows = aggregate_paired_rows(paired_rows) if paired_rows else []
    role_rows = role_summary_rows(paired_rows) if paired_rows else []
    rotation_rows = fixed_physical_rotation_summary_rows(paired_rows) if paired_rows else []
    paired_training_epoch_rows = paired_training_epoch_mean_rows(paired_rows) if paired_rows else []
    paired_training_rows = paired_training_stage_summary_rows(paired_rows) if paired_rows else []
    difficulty_rows = difficulty_summary_rows(raw_rows)
    method_stage_rows = method_stage_summary_rows(raw_rows)
    _write_csv(os.path.join(output_root, "stage_S2_raw_results.csv"), raw_rows, RAW_FIELDS)
    _write_csv(os.path.join(output_root, "stage_S2_paired_results.csv"), paired_rows, PAIRED_FIELDS)
    _write_csv(os.path.join(output_root, "stage_S2_runs.csv"), inventory,
               tuple(inventory[0].keys()) if inventory else ("stage",))
    _write_csv(os.path.join(output_root, "stage_S2_four_seed_aggregates.csv"), aggregate_rows,
               tuple(aggregate_rows[0].keys()) if aggregate_rows else ("comparison",))
    _write_csv(os.path.join(output_root, "stage_S2_role_summaries.csv"), role_rows,
               tuple(role_rows[0].keys()) if role_rows else ("comparison",))
    _write_csv(os.path.join(output_root, "stage_S2_fixed_physical_rotation_summaries.csv"), rotation_rows,
               tuple(rotation_rows[0].keys()) if rotation_rows else ("comparison",))
    _write_csv(os.path.join(output_root, "stage_S2_paired_training_epoch_means.csv"), paired_training_epoch_rows,
               tuple(paired_training_epoch_rows[0].keys()) if paired_training_epoch_rows else ("comparison",))
    _write_csv(os.path.join(output_root, "stage_S2_paired_training_stage_summaries.csv"), paired_training_rows,
               tuple(paired_training_rows[0].keys()) if paired_training_rows else ("comparison",))
    _write_csv(os.path.join(output_root, "stage_S2_difficulty_summaries.csv"), difficulty_rows,
               tuple(difficulty_rows[0].keys()) if difficulty_rows else ("dataset",))
    _write_csv(os.path.join(output_root, "stage_S2_training_stage_summaries.csv"), method_stage_rows,
               tuple(method_stage_rows[0].keys()) if method_stage_rows else ("method",))
    manifest = {
        "stage": STAGE, "backbone": BACKBONE, "expected_fresh_equal_runs": 16,
        "expected_reused_s1_m3oe_runs": 64, "fresh_equal_run_count": 16,
        "reused_run_count": 64, "raw_row_count": len(raw_rows),
        "expected_raw_row_count": 2016, "paired_row_count": len(paired_rows),
        "expected_paired_row_count": 2016, "checkpoint_labels": list(CHECKPOINTS),
        "common_epochs": list(COMMON_EPOCHS), "seeds": list(SEEDS),
        "paired_training_epoch_mean_row_count": len(paired_training_epoch_rows),
        "paired_training_stage_summary_row_count": len(paired_training_rows),
        "invalid_paired_row_count": len(invalid_pair_rows),
        "reuse_audit": {key: audit[key] for key in ("expected_count", "observed_count", "valid_count", "rejected_count", "all_reuse_ok")},
        "pairing_audit_valid": bool(pairing_audit.get("valid")),
        "pairing_audit": pairing_audit,
        "artifact_errors": artifact_errors, "raw_fields": list(RAW_FIELDS),
        "paired_fields": list(PAIRED_FIELDS), "scope_note": "mechanical structural report; no scientific conclusion",
        "outputs": ["stage_S2_raw_results.csv", "stage_S2_paired_results.csv", "stage_S2_runs.csv",
                    "stage_S2_four_seed_aggregates.csv", "stage_S2_role_summaries.csv",
                    "stage_S2_fixed_physical_rotation_summaries.csv",
                    "stage_S2_paired_training_epoch_means.csv",
                    "stage_S2_paired_training_stage_summaries.csv",
                    "stage_S2_difficulty_summaries.csv", "stage_S2_training_stage_summaries.csv",
                    "stage_S2_summary_manifest.json",
                    "stage_S2_run_report.md"],
    }
    with open(os.path.join(output_root, "stage_S2_summary_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.write("\n")
    report_lines = [
        "# Stage S2 Structural Run Report", "",
        "This file is a mechanical artifact inventory. It contains no scientific interpretation.", "",
        "- Fresh Equal M3OE runs expected: {}".format(manifest["expected_fresh_equal_runs"]),
        "- Reused S1 M3OE runs audited: {} (valid: {})".format(manifest["expected_reused_s1_m3oe_runs"], audit["valid_count"]),
        "- Raw rows: {} / expected {}".format(len(raw_rows), manifest["expected_raw_row_count"]),
        "- Paired rows: {} / expected {}".format(len(paired_rows), manifest["expected_paired_row_count"]),
        "- Invalid paired rows excluded from aggregation: {}".format(len(invalid_pair_rows)),
        "- Checkpoint labels: {}".format(", ".join(CHECKPOINTS)),
        "- Artifact validation error records: {}".format(len(artifact_errors)),
        "",
        "Scientific analysis, evidence-strength labels, and GO/STOP decisions are outside this report.",
    ]
    with open(os.path.join(output_root, "stage_S2_run_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")
    return {"manifest": manifest, "raw_rows": raw_rows, "paired_rows": paired_rows,
            "aggregate_rows": aggregate_rows, "role_rows": role_rows,
            "rotation_rows": rotation_rows, "paired_training_epoch_rows": paired_training_epoch_rows,
            "paired_training_rows": paired_training_rows,
            "difficulty_rows": difficulty_rows, "method_stage_rows": method_stage_rows,
            "audit": audit, "pairing_audit": pairing_audit, "inventory": inventory}


__all__ = [
    "CHECKPOINTS", "RAW_FIELDS", "PAIRED_FIELDS", "build_paired_rows", "aggregate_paired_rows",
    "role_summary_rows", "fixed_physical_rotation_summary_rows",
    "paired_training_epoch_mean_rows", "paired_training_stage_summary_rows",
    "difficulty_summary_rows", "method_stage_summary_rows",
    "collect", "checkpoint_hash_for_label", "_metric_artifact_reasons", "_raw_rows_for_artifact",
]
