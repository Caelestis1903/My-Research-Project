"""Mechanical Stage S1 artifact collection and paired raw tables.

This module deliberately stops at structural aggregation.  It does not
assign evidence strength, choose a scientific explanation, or issue the
Stage 2 gate; those actions belong to the main analysis agent.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_S1_orchestrator import audit_block, build_tasks, validate_run
from stage_S1_metrics import LOGLOSS_EPS, safe_auc, safe_logloss
from stage_S1_protocol import COMMON_EPOCHS, THRESHOLDS

# Compatibility aliases used by older focused tests.  The implementation is
# shared with runner and validator; no report-local metric math is allowed.
_safe_auc = safe_auc
_safe_logloss = safe_logloss


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


def _num(value: object) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    if not fields:
        fields = ["status"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def collect(raw_root: str = "experiments/stage_S1_raw_results",
            *, require_formal: bool = True) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Collect valid per-run selected rows and enumerate all expected paths."""

    tasks = build_tasks(raw_root)
    run_rows: List[Dict[str, object]] = []
    inventory: List[Dict[str, object]] = []
    for task in tasks:
        valid, reasons = validate_run(task, require_formal=require_formal)
        entry = {"formal_run_key": task["formal_run_key"], "run_type": task["run_type"],
                 "reference_scope": ("single_domain_mtl_reference"
                                     if task["run_type"] == "single_domain" else "joint"),
                 "dataset": task["dataset"], "setting": task["setting"],
                 "backbone": task["backbone"], "seed": task["seed"],
                 "physical_domain": task.get("physical_domain"), "run_dir": task["run_dir"],
                 "status": "OK" if valid else "MISSING_OR_INVALID",
                 "validation_reasons": ";".join(reasons)}
        inventory.append(entry)
        if not valid:
            continue
        manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
        # metrics_test.csv is the complete deferred support (selected, final,
        # and all common epochs).  Using it here keeps checkpoint sensitivity
        # mechanically reproducible; metrics_summary.csv is selected-only.
        summary = _read_csv(os.path.join(str(task["run_dir"]), "metrics_test.csv"))
        for row in summary:
            output = dict(entry)
            output.update(row)
            output["selected_epoch_manifest"] = manifest.get("selected_epoch", "")
            output["formal_valid"] = int(bool(manifest.get("formal_valid")))
            run_rows.append(output)
    manifest = {
        "stage": "S1", "expected_run_count": len(tasks),
        "valid_run_count": sum(item["status"] == "OK" for item in inventory),
        "missing_run_count": sum(item["status"] != "OK" for item in inventory),
        "runs": inventory,
        "scope_note": "mechanical inventory only; no scientific conclusion",
    }
    if require_formal:
        invalid = [item for item in inventory if item["status"] != "OK"]
        if invalid:
            raise RuntimeError("formal Stage S1 report has invalid runs: {}".format(
                len(invalid)))
        blocks: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
        for task in tasks:
            blocks[(task["dataset"], task["setting"], task["backbone"], task["seed"])].append(task)
        block_failures = []
        for key, block_tasks in blocks.items():
            audit = audit_block(block_tasks, require_formal=True)
            if not audit.get("valid"):
                block_failures.append({"block": key, "audit": audit})
        if block_failures:
            raise RuntimeError("formal Stage S1 block audit failed: {}".format(block_failures[:1]))
    return run_rows, manifest


def _run_key(row: Mapping[str, object]) -> Tuple[str, str, str, int]:
    return (str(row.get("dataset")), str(row.get("setting")), str(row.get("backbone")),
            int(float(row.get("seed"))))


def _selected_lookup(run_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[object, ...], Mapping[str, object]]:
    result: Dict[Tuple[object, ...], Mapping[str, object]] = {}
    for row in run_rows:
        key = (row.get("dataset"), row.get("setting"), row.get("backbone"),
               _int(row.get("seed")), row.get("run_type"), _int(row.get("physical_domain")),
               _int(row.get("task")), row.get("checkpoint", "selected"))
        if key in result:
            raise ValueError("duplicate selected/raw metric key: {}".format(key))
        result[key] = row
    return result


def paired_cell_rows(run_rows: Sequence[Mapping[str, object]],
                     *, checkpoint: str = "selected") -> List[Dict[str, object]]:
    """Join joint cells to their matching single-domain reference cells."""

    lookup = _selected_lookup(run_rows)
    result: List[Dict[str, object]] = []
    for row in run_rows:
        if row.get("run_type") != "joint" or row.get("checkpoint", "selected") != checkpoint:
            continue
        dataset, setting, backbone = str(row["dataset"]), str(row["setting"]), str(row["backbone"])
        seed = _int(row.get("seed"))
        physical = _int(row.get("physical_domain"))
        task = _int(row.get("task"))
        reference = lookup.get((dataset, setting, backbone, seed, "single_domain",
                                physical, task, checkpoint))
        if reference is None:
            raise ValueError("missing paired reference for {} d{}t{} seed{}".format(
                backbone, physical, task, seed))
        joint_auc, ref_auc = _num(row.get("auc")), _num(reference.get("auc"))
        joint_ll, ref_ll = _num(row.get("logloss")), _num(reference.get("logloss"))
        if None in (joint_auc, ref_auc, joint_ll, ref_ll):
            raise ValueError("non-finite paired metrics")
        delta_auc = joint_auc - ref_auc
        gap_ll = joint_ll - ref_ll
        result.append({
            "dataset": dataset, "setting": setting, "backbone": backbone,
            "seed": seed, "physical_domain": physical, "task": task,
            "size_role": row.get("size_role", ""), "domain_size": row.get("train_sample_count", ""),
            "checkpoint": checkpoint, "joint_auc": joint_auc, "reference_auc": ref_auc,
            "joint_logloss": joint_ll, "reference_logloss": ref_ll,
            "delta_auc": delta_auc, "gap_logloss": gap_ll,
            "raw_auc_nt": int(delta_auc < 0), "raw_logloss_nt": int(gap_ll > 0),
            "practical_auc_nt": int(delta_auc <= THRESHOLDS["practical_auc"]),
            "strong_auc_nt": int(delta_auc <= THRESHOLDS["strong_auc"]),
            "practical_logloss_nt": int(gap_ll >= THRESHOLDS["practical_logloss"]),
            "concordant_practical_nt": int(delta_auc <= THRESHOLDS["practical_auc"] and gap_ll > 0),
            "concordant_strong_nt": int(delta_auc <= THRESHOLDS["strong_auc"] and
                                          gap_ll >= THRESHOLDS["practical_logloss"]),
            "reference_validation_difficulty": reference.get("selected_validation_macro_auc", ""),
            "reference_validation_difficulty_logloss": reference.get(
                "validation_macro_logloss_at_stage", ""),
            "joint_validation_auc_at_stage": row.get("validation_auc_at_stage", ""),
            "reference_validation_auc_at_stage": reference.get("validation_auc_at_stage", ""),
            "joint_validation_logloss_at_stage": row.get("validation_logloss_at_stage", ""),
            "reference_validation_logloss_at_stage": reference.get("validation_logloss_at_stage", ""),
            "domain_share": row.get("domain_share", ""),
            "physical_domain_identity": row.get("physical_domain_identity", ""),
            "task_identity": row.get("task_identity", ""),
            "train_label_entropy": row.get("train_label_entropy", ""),
            "validation_label_entropy": row.get("validation_label_entropy", ""),
            "test_label_entropy": row.get("test_label_entropy", ""),
            "reference_train_sample_count": reference.get("train_sample_count", ""),
            "reference_validation_sample_count": reference.get("validation_sample_count", ""),
            "reference_test_sample_count": reference.get("test_sample_count", ""),
            "reference_domain_share": reference.get("domain_share", ""),
            "reference_validation_positive_rate": reference.get("validation_positive_rate", ""),
            "reference_test_positive_rate": reference.get("test_positive_rate", ""),
            "reference_train_label_entropy": reference.get("train_label_entropy", ""),
            "reference_validation_label_entropy": reference.get("validation_label_entropy", ""),
            "reference_test_label_entropy": reference.get("test_label_entropy", ""),
            "parameter_count_joint": row.get("parameter_count", ""),
            "parameter_count_reference": reference.get("parameter_count", ""),
        })
    return result


def common_epoch_rows(run_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for epoch in COMMON_EPOCHS:
        checkpoint = "common_epoch_{}".format(epoch)
        new_rows = paired_cell_rows(run_rows, checkpoint=checkpoint)
        for row in new_rows:
            row["training_stage"] = checkpoint
        rows.extend(new_rows)
    return rows


def aggregate_paired_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Raw grouped means/counts; seeds remain visible as the repeat unit."""

    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("physical_domain"), row.get("task"), row.get("checkpoint"))].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        auc = [float(row["delta_auc"]) for row in rows]
        ll = [float(row["gap_logloss"]) for row in rows]
        output.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "physical_domain": key[3], "task": key[4], "checkpoint": key[5],
            "seed_count": len(rows), "seed_values": ",".join(str(row["seed"]) for row in rows),
            "mean_delta_auc": sum(auc) / len(auc),
            "sample_sd_delta_auc": (sum((value - sum(auc) / len(auc)) ** 2 for value in auc) /
                                     (len(auc) - 1)) ** 0.5 if len(auc) > 1 else 0.0,
            "range_delta_auc": max(auc) - min(auc),
            "negative_seed_count": sum(value < 0 for value in auc),
            "mean_gap_logloss": sum(ll) / len(ll),
            "positive_logloss_seed_count": sum(value > 0 for value in ll),
            "practical_nt_rate": sum(float(row["practical_auc_nt"]) for row in rows) / len(rows),
            "practical_logloss_nt_rate": sum(float(row["practical_logloss_nt"]) for row in rows) / len(rows),
            "strong_nt_rate": sum(float(row["strong_auc_nt"]) for row in rows) / len(rows),
            "concordant_practical_nt_rate": sum(float(row["concordant_practical_nt"]) for row in rows) / len(rows),
            "concordant_strong_nt_rate": sum(float(row["concordant_strong_nt"]) for row in rows) / len(rows),
        })
    return output


def _prediction_payload(row: Mapping[str, object]) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.join(str(row.get("run_dir", "")), "predictions_test.npz")
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError, KeyError):
        return None


def task_identity_lookup(run_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, int], str]:
    """Validate and index task labels independently for every formal run."""

    result: Dict[Tuple[str, int], str] = {}
    for row in run_rows:
        formal_key_value = str(row.get("formal_run_key", ""))
        task_value = _int(row.get("task"))
        identity = str(row.get("task_identity", "")).strip()
        if not formal_key_value or task_value not in (0, 1) or not identity:
            raise ValueError("missing task identity in validated raw metric rows")
        key = (formal_key_value, int(task_value))
        prior = result.get(key)
        if prior is not None and prior != identity:
            raise ValueError("inconsistent task identity for {} task {}".format(
                formal_key_value, task_value))
        result[key] = identity
    return result


def prediction_aggregate_rows(run_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Compute per-task pooled and assembled-reference pooled metrics mechanically."""

    runs: Dict[str, Mapping[str, object]] = {}
    for row in run_rows:
        runs.setdefault(str(row.get("formal_run_key")), row)
    identities = task_identity_lookup(run_rows)
    payloads: Dict[str, Dict[str, np.ndarray]] = {}
    for key, row in runs.items():
        payload = _prediction_payload(row)
        if payload is not None:
            payloads[key] = payload

    output: List[Dict[str, object]] = []
    checkpoints = ("selected", "final") + tuple("common_epoch_{}".format(e) for e in COMMON_EPOCHS)
    for key, row in sorted(runs.items()):
        payload = payloads.get(key)
        if payload is None:
            continue
        labels = payload["labels"]
        for checkpoint in checkpoints:
            prediction_key = "pred_{}".format(checkpoint)
            if prediction_key not in payload:
                continue
            prediction = payload[prediction_key]
            for task in range(labels.shape[1]):
                output.append({
                    "formal_run_key": key, "dataset": row.get("dataset"),
                    "setting": row.get("setting"), "backbone": row.get("backbone"),
                    "seed": row.get("seed"), "run_type": row.get("run_type"),
                    "physical_domain": row.get("physical_domain"), "checkpoint": checkpoint,
                    "task": task, "task_identity": identities[(key, task)],
                    "pooled_auc": _safe_auc(labels[:, task].tolist(), prediction[:, task].tolist()),
                    "pooled_logloss": _safe_logloss(labels[:, task].tolist(), prediction[:, task].tolist()),
                    "assembled_reference_pooled_auc": "",
                    "assembled_reference_pooled_logloss": "",
                    "calibration_sensitive_assembled_reference": 0,
                })

    # Build one row-aligned reference ensemble for every joint run.  This is
    # descriptive and keeps task labels separate; no cross-task AUC is made.
    for key, joint in sorted(runs.items()):
        if joint.get("run_type") != "joint" or key not in payloads:
            continue
        ref_by_domain: Dict[int, str] = {}
        for ref_key, ref in runs.items():
            if ref.get("run_type") == "single_domain" and \
                    ref.get("dataset") == joint.get("dataset") and \
                    ref.get("setting") == joint.get("setting") and \
                    ref.get("backbone") == joint.get("backbone") and \
                    ref.get("seed") == joint.get("seed"):
                ref_by_domain[int(ref.get("physical_domain"))] = ref_key
        if set(ref_by_domain) != {0, 1, 2}:
            continue
        joint_payload = payloads[key]
        physical = joint_payload["physical_domain"].astype(np.int64)
        row_ids = joint_payload["row_ids"].astype(np.int64)
        labels = joint_payload["labels"]
        for checkpoint in checkpoints:
            assembled = np.full_like(joint_payload["pred_{}".format(checkpoint)], np.nan)
            for domain, ref_key in ref_by_domain.items():
                ref_payload = payloads.get(ref_key)
                if ref_payload is None or "pred_{}".format(checkpoint) not in ref_payload:
                    assembled = None
                    break
                ref_ids = ref_payload["row_ids"].astype(np.int64)
                ref_values = ref_payload["pred_{}".format(checkpoint)]
                lookup = {int(value): index for index, value in enumerate(ref_ids)}
                mask = physical == domain
                indices = [lookup.get(int(value), -1) for value in row_ids[mask]]
                if any(index < 0 for index in indices):
                    assembled = None
                    break
                assembled[mask] = ref_values[np.asarray(indices, dtype=np.int64)]
            if assembled is None or not np.isfinite(assembled).all():
                continue
            for task in range(labels.shape[1]):
                matches = [item for item in output if item.get("formal_run_key") == key
                           and item.get("checkpoint") == checkpoint and item.get("task") == task]
                if matches:
                    matches[0]["assembled_reference_pooled_auc"] = _safe_auc(
                        labels[:, task].tolist(), assembled[:, task].tolist())
                    matches[0]["assembled_reference_pooled_logloss"] = _safe_logloss(
                        labels[:, task].tolist(), assembled[:, task].tolist())
                    matches[0]["calibration_sensitive_assembled_reference"] = 1
    return output


def aggregate_run_checkpoints(run_rows: Sequence[Mapping[str, object]],
                              prediction_rows: Optional[Sequence[Mapping[str, object]]] = None) -> List[Dict[str, object]]:
    """Compute descriptive per-run summaries without treating cells as replicates."""

    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in run_rows:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("seed"), row.get("run_type"), row.get("physical_domain"),
                row.get("checkpoint"))].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        if not rows:
            continue
        auc = [float(row["auc"]) for row in rows if _num(row.get("auc")) is not None]
        ll = [float(row["logloss"]) for row in rows if _num(row.get("logloss")) is not None]
        if not auc or not ll:
            continue
        # Rows are task-major; domain shares are training counts in the raw
        # row fields.  This is a descriptive summary of one run, never an n
        # value for seed-level inference.
        by_domain: Dict[object, List[float]] = defaultdict(list)
        for row in rows:
            by_domain[row.get("physical_domain")].append(float(row["auc"]))
        weighted_auc_num = 0.0
        weighted_ll_num = 0.0
        total_weight = 0.0
        for row in rows:
            weight = float(row.get("train_sample_count", 0) or 0)
            weighted_auc_num += weight * float(row["auc"])
            weighted_ll_num += weight * float(row["logloss"])
            total_weight += weight
        prediction_lookup = {(str(item.get("formal_run_key")), str(item.get("checkpoint")),
                              int(item.get("task"))): item
                             for item in (prediction_rows or [])}
        pooled = [prediction_lookup.get((str(rows[0].get("formal_run_key")), str(key[6]), task))
                  for task in range(2)]
        pooled = [item for item in pooled if item is not None]
        output.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "seed": key[3], "run_type": key[4], "physical_domain": key[5],
            "checkpoint": key[6], "cell_count": len(rows),
            "macro_auc": sum(auc) / len(auc), "macro_logloss": sum(ll) / len(ll),
            "target_weighted_auc": weighted_auc_num / total_weight if total_weight else "",
            "target_weighted_logloss": weighted_ll_num / total_weight if total_weight else "",
            "worst_domain_auc": min(sum(values) / len(values) for values in by_domain.values()),
            "worst_cell_auc": min(auc), "maximum_cell_logloss": max(ll),
            "pooled_logloss": (sum(float(row["test_sample_count"]) * float(row["logloss"])
                                    for row in rows) /
                               sum(float(row["test_sample_count"]) for row in rows)),
            "pooled_auc": (sum(float(item["pooled_auc"]) for item in pooled) / len(pooled)
                           if pooled else ""),
            "per_task_pooled_auc_task0": pooled[0]["pooled_auc"] if len(pooled) > 0 else "",
            "per_task_pooled_auc_task1": pooled[1]["pooled_auc"] if len(pooled) > 1 else "",
            "assembled_reference_pooled_auc": (
                sum(float(item["assembled_reference_pooled_auc"]) for item in pooled
                    if _num(item.get("assembled_reference_pooled_auc")) is not None) /
                sum(1 for item in pooled if _num(item.get("assembled_reference_pooled_auc")) is not None)
                if any(_num(item.get("assembled_reference_pooled_auc")) is not None for item in pooled)
                else ""),
            "pooled_auc_is_gate": 0,
        })
    return output


def contradiction_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Enumerate sign changes across every preregistered comparison axis."""

    result: List[Dict[str, object]] = []
    axes = {
        "seed": ("dataset", "setting", "backbone", "physical_domain", "task", "checkpoint"),
        "backbone": ("dataset", "setting", "physical_domain", "task", "checkpoint", "seed"),
        # Dataset comparisons deliberately do not require identical
        # dataset-specific setting labels (MovieLens rotations are d*_large;
        # KuaiRand has natural).  The observed dataset/setting levels remain
        # in the output for mechanical auditability.
        # A dataset comparison is valid only inside a matched size-role
        # stratum.  Without ``size_role``, the three MovieLens rotations can
        # contribute opposing signs to one nominal dataset level even when
        # KuaiRand has no opposing cross-dataset observation.
        "dataset": ("backbone", "physical_domain", "task", "checkpoint", "seed", "size_role"),
        "rotation": ("dataset", "backbone", "physical_domain", "task", "checkpoint", "seed"),
        "stage": ("dataset", "setting", "backbone", "physical_domain", "task", "seed"),
    }
    varying_fields = {"seed": "seed", "backbone": "backbone", "dataset": "dataset",
                      "rotation": "setting", "stage": "checkpoint"}
    for axis, group_fields in axes.items():
        groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
        for row in paired_rows:
            if axis == "dataset" and not str(row.get("size_role", "")).strip():
                # A missing role cannot support a valid cross-dataset match;
                # fail closed for that comparison axis rather than pooling
                # unmatched MovieLens rotations.
                continue
            groups[tuple(row.get(field) for field in group_fields)].append(row)
        for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            varying_field = varying_fields[axis]
            if len({row.get(varying_field) for row in rows}) < 2:
                continue
            for metric, field in (("AUC", "delta_auc"), ("LogLoss", "gap_logloss")):
                signs = {"positive" if float(row[field]) > 0 else
                         "negative" if float(row[field]) < 0 else "zero" for row in rows}
                if "positive" in signs and "negative" in signs:
                    levels = sorted({"{}:{}".format(row.get(varying_field), row.get("setting"))
                                     if axis == "dataset" else str(row.get(varying_field))
                                     for row in rows})
                    result.append({"comparison_axis": axis, "group_key": "|".join(str(x) for x in key),
                                   "varying_field": varying_field,
                                   "levels": ",".join(levels),
                                   "metric": metric, "signs": ",".join(sorted(signs)),
                                   "observation_count": len(rows), "sign_flip": 1})
    # AUC/LogLoss discordance is a separate metric-level contradiction even
    # when neither metric changes sign across a seed axis.
    for row in paired_rows:
        if (float(row.get("delta_auc", 0)) < 0) != (float(row.get("gap_logloss", 0)) > 0):
            result.append({"comparison_axis": "metric", "group_key": "|".join(
                str(row.get(field)) for field in ("dataset", "setting", "backbone",
                                                   "seed", "physical_domain", "task", "checkpoint")),
                "varying_field": "metric", "levels": "AUC,LogLoss",
                "metric": "AUC_vs_LogLoss", "signs": "discordant",
                "observation_count": 1, "sign_flip": 1})
    return result


def stable_inventory(paired_selected: Sequence[Mapping[str, object]],
                     paired_common: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_selected:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("physical_domain"), row.get("task"))].append(row)
    common_groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_common:
        common_groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                      row.get("physical_domain"), row.get("task"))].append(row)
    result: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        auc = [float(row["delta_auc"]) for row in rows]
        ll = [float(row["gap_logloss"]) for row in rows]
        common_negative = 0
        for epoch in COMMON_EPOCHS:
            values = [float(row["delta_auc"]) for row in common_groups.get(key, [])
                      if row.get("checkpoint") == "common_epoch_{}".format(epoch)]
            if values and sum(values) / len(values) < 0:
                common_negative += 1
        result.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "physical_domain": key[3], "task": key[4],
            "training_stage": "selected_plus_common_epochs", "seed_count": len(rows),
            "mean_delta_auc": sum(auc) / len(auc), "negative_auc_seed_count": sum(v < 0 for v in auc),
            "mean_gap_logloss": sum(ll) / len(ll), "positive_logloss_seed_count": sum(v > 0 for v in ll),
            "matched_common_epoch_negative_count": common_negative,
            "stable_practical_cell": int(len(rows) == 4 and sum(auc) / len(auc) <= THRESHOLDS["practical_auc"]
                and sum(v < 0 for v in auc) >= 3 and sum(ll) / len(ll) > 0
                and sum(v > 0 for v in ll) >= 3 and common_negative >= 3),
        })
    return result


def reference_baseline_variation(run_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in run_rows:
        if row.get("run_type") == "single_domain" and row.get("checkpoint") == "selected":
            groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                    row.get("physical_domain"), row.get("task"))].append(row)
    result: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        auc = [float(row["auc"]) for row in rows]
        ll = [float(row["logloss"]) for row in rows]
        mean_auc = sum(auc) / len(auc)
        mean_ll = sum(ll) / len(ll)
        result.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "physical_domain": key[3], "task": key[4], "seed_count": len(rows),
            "reference_auc_mean": mean_auc,
            "reference_auc_sample_sd": (sum((v - mean_auc) ** 2 for v in auc) / (len(auc) - 1)) ** 0.5 if len(auc) > 1 else 0.0,
            "reference_auc_range": max(auc) - min(auc), "reference_logloss_mean": mean_ll,
            "reference_logloss_sample_sd": (sum((v - mean_ll) ** 2 for v in ll) / (len(ll) - 1)) ** 0.5 if len(ll) > 1 else 0.0,
            "reference_logloss_range": max(ll) - min(ll),
            "reference_validation_difficulty_values": ",".join(str(row.get("validation_auc_at_stage", "")) for row in rows),
        })
    return result


def role_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate paired deltas by the frozen domain-size role labels."""

    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("size_role"), row.get("physical_domain"), row.get("task"),
                row.get("checkpoint"))].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        auc = [float(row["delta_auc"]) for row in rows]
        ll = [float(row["gap_logloss"]) for row in rows]
        output.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "size_role": key[3], "physical_domain": key[4], "task": key[5],
            "checkpoint": key[6], "role_row_count": len(rows),
            "seed_count": len({row.get("seed") for row in rows}),
            "role_domain_size": rows[0].get("domain_size", ""),
            "difficulty_evaluation_split": "validation",
            "mean_delta_auc": sum(auc) / len(auc),
            "mean_gap_logloss": sum(ll) / len(ll),
            "raw_auc_nt_rate": sum(float(row["raw_auc_nt"]) for row in rows) / len(rows),
            "raw_logloss_nt_rate": sum(float(row["raw_logloss_nt"]) for row in rows) / len(rows),
            "practical_nt_rate": sum(float(row["practical_auc_nt"]) for row in rows) / len(rows),
            "concordant_practical_nt_rate": (
                sum(float(row["concordant_practical_nt"]) for row in rows) / len(rows)),
            "reference_difficulty_mean": (
                sum(float(row["reference_validation_difficulty"]) for row in rows
                    if _num(row.get("reference_validation_difficulty")) is not None) /
                max(1, sum(1 for row in rows
                           if _num(row.get("reference_validation_difficulty")) is not None))),
        })
    return output


def difficulty_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Emit raw aggregates keyed by independently measured reference difficulty.

    Difficulty is never inferred from the joint result: the reference
    validation macro-AUC and LogLoss are carried from the independent
    single-domain run and summarized without a post-hoc scientific label.
    """

    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("physical_domain"), row.get("task"), row.get("checkpoint"))].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        # Both difficulty measures are taken from the independent reference's
        # validation split at the matched training stage.  Do not combine a
        # validation AUC with a deferred-test LogLoss in one summary.
        difficulty_auc = [float(row["reference_validation_difficulty"]) for row in rows
                          if _num(row.get("reference_validation_difficulty")) is not None]
        difficulty_logloss = [float(row["reference_validation_difficulty_logloss"]) for row in rows
                              if _num(row.get("reference_validation_difficulty_logloss")) is not None]
        deltas = [float(row["delta_auc"]) for row in rows]
        gaps = [float(row["gap_logloss"]) for row in rows]
        output.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "physical_domain": key[3], "task": key[4], "checkpoint": key[5],
            "difficulty_evaluation_split": "validation",
            "difficulty_row_count": len(rows),
            "seed_count": len({row.get("seed") for row in rows}),
            "reference_validation_auc_mean": (sum(difficulty_auc) / len(difficulty_auc)
                                               if difficulty_auc else ""),
            "reference_validation_auc_min": min(difficulty_auc) if difficulty_auc else "",
            "reference_validation_auc_max": max(difficulty_auc) if difficulty_auc else "",
            "reference_validation_logloss_mean": (sum(difficulty_logloss) / len(difficulty_logloss)
                                                   if difficulty_logloss else ""),
            "reference_validation_logloss_min": min(difficulty_logloss) if difficulty_logloss else "",
            "reference_validation_logloss_max": max(difficulty_logloss) if difficulty_logloss else "",
            "mean_delta_auc": sum(deltas) / len(deltas),
            "mean_gap_logloss": sum(gaps) / len(gaps),
            "raw_auc_nt_rate": sum(float(row["raw_auc_nt"]) for row in rows) / len(rows),
            "practical_nt_rate": sum(float(row["practical_auc_nt"]) for row in rows) / len(rows),
            "concordant_practical_nt_rate": (
                sum(float(row["concordant_practical_nt"]) for row in rows) / len(rows)),
        })
    return output


def training_stage_summary_rows(paired_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Keep a mechanical dataset × training-stage summary for raw auditing."""

    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        groups[(row.get("dataset"), row.get("setting"), row.get("backbone"),
                row.get("checkpoint"), row.get("physical_domain"), row.get("task"))].append(row)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        auc = [float(row["delta_auc"]) for row in rows]
        ll = [float(row["gap_logloss"]) for row in rows]
        output.append({
            "dataset": key[0], "setting": key[1], "backbone": key[2],
            "training_stage": key[3], "physical_domain": key[4], "task": key[5],
            "seed_count": len({row.get("seed") for row in rows}),
            "row_count": len(rows), "mean_delta_auc": sum(auc) / len(auc),
            "mean_gap_logloss": sum(ll) / len(ll),
            "negative_auc_seed_count": sum(value < 0 for value in auc),
            "positive_logloss_seed_count": sum(value > 0 for value in ll),
            "practical_nt_rate": sum(float(row["practical_auc_nt"]) for row in rows) / len(rows),
            "concordant_practical_nt_rate": (
                sum(float(row["concordant_practical_nt"]) for row in rows) / len(rows)),
        })
    return output


def write_outputs(raw_root: str = "experiments/stage_S1_raw_results",
                  output_root: str = "experiments", *, require_formal: bool = True) -> Dict[str, object]:
    run_rows, inventory_manifest = collect(raw_root, require_formal=require_formal)
    inventory = inventory_manifest["runs"]
    paired = paired_cell_rows(run_rows, checkpoint="selected") if run_rows else []
    final_paired = paired_cell_rows(run_rows, checkpoint="final") if run_rows else []
    common = common_epoch_rows(run_rows) if run_rows else []
    aggregates = aggregate_paired_rows(paired)
    prediction_rows = prediction_aggregate_rows(run_rows)
    checkpoint_aggregates = aggregate_run_checkpoints(run_rows, prediction_rows)
    all_paired_for_inventory = list(paired) + list(final_paired) + list(common)
    contradictions = contradiction_rows(all_paired_for_inventory)
    stable = stable_inventory(paired, common)
    baseline = reference_baseline_variation(run_rows)
    role_summaries = role_summary_rows(all_paired_for_inventory)
    difficulty_summaries = difficulty_summary_rows(all_paired_for_inventory)
    training_stage_summaries = training_stage_summary_rows(all_paired_for_inventory)
    _write_csv(os.path.join(output_root, "stage_S1_runs.csv"), inventory)
    _write_csv(os.path.join(output_root, "stage_S1_paired_cells.csv"), paired)
    _write_csv(os.path.join(output_root, "stage_S1_final_paired_cells.csv"), final_paired)
    _write_csv(os.path.join(output_root, "stage_S1_paired_aggregates.csv"), aggregates)
    _write_csv(os.path.join(output_root, "stage_S1_checkpoint_aggregates.csv"), checkpoint_aggregates)
    _write_csv(os.path.join(output_root, "stage_S1_prediction_aggregates.csv"), prediction_rows)
    _write_csv(os.path.join(output_root, "stage_S1_stable_cells.csv"), stable)
    _write_csv(os.path.join(output_root, "stage_S1_reference_baseline_variation.csv"), baseline)
    _write_csv(os.path.join(output_root, "stage_S1_role_summaries.csv"), role_summaries)
    _write_csv(os.path.join(output_root, "stage_S1_difficulty_summaries.csv"), difficulty_summaries)
    _write_csv(os.path.join(output_root, "stage_S1_training_stage_summaries.csv"), training_stage_summaries)
    _write_csv(os.path.join(output_root, "stage_S1_contradictions.csv"), contradictions)
    _write_csv(os.path.join(output_root, "stage_S1_common_epoch_trajectories.csv"), common)
    manifest = {**inventory_manifest, "paired_cell_count": len(paired),
                 "final_paired_cell_count": len(final_paired),
                 "aggregate_group_count": len(aggregates),
                 "checkpoint_aggregate_count": len(checkpoint_aggregates),
                 "prediction_aggregate_count": len(prediction_rows),
                 "stable_cell_count": len(stable),
                 "reference_baseline_variation_count": len(baseline),
                 "role_summary_count": len(role_summaries),
                 "difficulty_summary_count": len(difficulty_summaries),
                 "training_stage_summary_count": len(training_stage_summaries),
                "contradiction_count": len(contradictions),
                "common_epoch_row_count": len(common),
                 "outputs": ["stage_S1_runs.csv", "stage_S1_paired_cells.csv",
                 "stage_S1_final_paired_cells.csv",
                 "stage_S1_paired_aggregates.csv", "stage_S1_checkpoint_aggregates.csv",
                 "stage_S1_prediction_aggregates.csv", "stage_S1_stable_cells.csv",
                 "stage_S1_reference_baseline_variation.csv",
                "stage_S1_role_summaries.csv", "stage_S1_difficulty_summaries.csv",
                "stage_S1_training_stage_summaries.csv",
                "stage_S1_contradictions.csv",
                "stage_S1_common_epoch_trajectories.csv"]}
    with open(os.path.join(output_root, "stage_S1_summary_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def write_report(manifest: Mapping[str, object], path: str = "experiments/reports/stage_S1_run_report.md") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# Stage S1 Raw Run Report", "",
        "This report is a structural/raw artifact report. It does not make a scientific conclusion or a Stage 2 decision.",
        "", "- Expected formal runs: {}".format(manifest.get("expected_run_count", "")),
        "- Valid run artifacts: {}".format(manifest.get("valid_run_count", "")),
        "- Missing or invalid artifacts: {}".format(manifest.get("missing_run_count", "")),
        "- Paired cell rows: {}".format(manifest.get("paired_cell_count", "")),
        "- Role summary rows: {}".format(manifest.get("role_summary_count", "")),
        "- Independent-reference difficulty summary rows: {}".format(
            manifest.get("difficulty_summary_count", "")),
        "- Dataset/training-stage summary rows: {}".format(
            manifest.get("training_stage_summary_count", "")),
        "- Seed is retained as the repeat-unit field in all aggregate tables.",
        "- Test metrics are read only from deferred selected/final/common checkpoint files.",
        "",
        "The main agent must perform the AUC/LogLoss, per-domain/per-task, seed, size/difficulty, backbone, dataset, and training-stage analysis separately.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Stage S1 mechanical raw report")
    parser.add_argument("--raw_root", default=os.path.join(os.path.dirname(__file__), "stage_S1_raw_results"))
    parser.add_argument("--output_root", default=os.path.dirname(__file__))
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "reports", "stage_S1_run_report.md"))
    parser.add_argument("--nonformal_smoke", action="store_true",
                        help="write an explicitly isolated structural smoke report")
    args = parser.parse_args()
    manifest = write_outputs(args.raw_root, args.output_root,
                             require_formal=not args.nonformal_smoke)
    write_report(manifest, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect", "paired_cell_rows", "common_epoch_rows", "aggregate_paired_rows",
           "prediction_aggregate_rows", "task_identity_lookup", "aggregate_run_checkpoints",
           "contradiction_rows", "stable_inventory", "reference_baseline_variation",
           "role_summary_rows", "difficulty_summary_rows", "training_stage_summary_rows",
           "write_outputs", "write_report", "main"]
