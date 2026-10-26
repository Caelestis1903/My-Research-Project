"""Resume-safe Stage S1 queue, preflight, and raw-artifact validator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stage_S1_protocol import (
    BACKBONES,
    COMMON_EPOCHS,
    DATASETS,
    EPOCHS,
    FORMAL_PROTOCOL,
    ROLE_MAPS,
    ROLE_SIZES,
    SEEDS,
    build_matrix,
    canonical_dataset,
    formal_key,
    settings_for_dataset,
    validate_matrix,
    protocol_mismatches,
)
from stage_S1_metrics import cell_metrics, task_domain_cell_index

HERE = os.path.dirname(os.path.abspath(__file__))
_FILE_HASH_CACHE: Dict[Tuple[str, int, int], str] = {}
_ARCHIVE_MUTEX = threading.RLock()
_OWNED_STAGE_LOCKS: Dict[str, str] = {}


REQUIRED_FILES = (
    "config.json", "run_manifest.json", "metrics_epoch.csv", "metrics_test.csv",
    "metrics_summary.csv", "batch_trace.csv", "initialization_audit.json",
    "parameter_manifest.json", "nonfinite_inventory.json",
    "optimizer_state.pth", "rng_state.pth", "predictions_test.npz", "stdout.log",
    "selected_checkpoint.pth", "final_checkpoint.pth",
    "common_epoch_9.pth", "common_epoch_19.pth", "common_epoch_29.pth",
    "common_epoch_39.pth", "common_epoch_49.pth",
)
ARCHITECTURE_FILES = ("architecture_optimizer_state.pth", "architecture_optimizer_absent.json")
METRICS_EPOCH_FIELDS = (
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch",
    "domain", "physical_domain", "size_role", "task", "cell",
    "train_sample_count", "validation_sample_count", "train_positive_rate",
    "validation_positive_rate", "train_cell_logloss", "validation_auc",
    "validation_logloss", "validation_macro_auc", "validation_macro_logloss",
    "target_weighted_validation_auc", "target_weighted_validation_logloss",
    "worst_domain_validation_auc", "worst_cell_validation_auc", "train_objective",
    "main_optimizer_steps", "architecture_optimizer_steps", "learning_rate",
    "architecture_learning_rate", "selected_checkpoint", "fixed_final", "common_epoch",
)
METRICS_TEST_FIELDS = (
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "checkpoint",
    "stage_epoch", "training_stage", "domain", "physical_domain", "size_role", "task",
    "cell", "train_sample_count", "validation_sample_count", "test_sample_count",
    "test_positive_rate", "auc", "logloss", "validation_auc_at_stage",
    "validation_logloss_at_stage", "validation_macro_auc_at_stage",
    "validation_macro_logloss_at_stage", "selected_epoch", "selected_validation_macro_auc",
    "parameter_count", "training_stage_identifier", "domain_share", "physical_domain_identity",
    "task_identity", "train_positive_count", "validation_positive_count", "test_positive_count",
    "train_positive_rate", "validation_positive_rate", "train_label_entropy",
    "validation_label_entropy", "test_label_entropy", "reference_scope",
)
BATCH_FIELDS = (
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch", "step",
    "kind", "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "train_objective",
)
FORBIDDEN_METHOD_TOKENS = ("safe_mtl", "curriculum", "momentum", "ema", "controller", "schedule")
LOCK_FILENAME = ".stage_S1.lock"
LOCK_RECLAIM_LOG_FILENAME = ".stage_S1.lock_reclaims.json"
KUAIRAND_EXPECTED_TRAIN_COUNTS = {0: 172_782, 1: 1_743_898, 2: 89_452}
CODE_HASH_FILES = ("stage_S1_run.py", "stage_S1_models.py", "stage_S1_data.py",
                   "stage_S1_protocol.py", "stage_S1_report.py", "stage_S1_orchestrator.py",
                   "stage_S1_metrics.py")


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _file_hash(path: str, *, fresh: bool = False) -> str:
    absolute = os.path.abspath(path)
    stat = os.stat(absolute)
    cache_key = (absolute, int(stat.st_size), int(stat.st_mtime_ns))
    if fresh:
        # Artifact validation is a trust boundary.  Size and mtime alone are
        # not a sufficient content identity: a same-size replacement can
        # restore the previous timestamp and otherwise reuse a stale digest.
        # Clear every cached generation for this path and read the bytes.
        for prior_key in tuple(_FILE_HASH_CACHE):
            if prior_key[0] == absolute:
                _FILE_HASH_CACHE.pop(prior_key, None)
    elif cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _array_hash(value: object) -> str:
    """Match stage_S1_data.array_hash without importing the dataset loader."""

    import numpy as np
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json_any(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _read_csv_with_header(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except (OSError, csv.Error):
        return [], []


def _numeric(value: object) -> Optional[float]:
    if value in (None, "", "NA", "NaN", "nan", "Inf", "+Inf", "-Inf"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _formal_cells(task: Mapping[str, object]) -> Tuple[int, Tuple[int, ...]]:
    return (3, (0, 1, 2)) if task.get("run_type") == "joint" else (1, (0,))


def expected_train_domain_counts(task: Mapping[str, object]) -> Dict[str, int]:
    """Derive frozen train exposure counts for one formal matrix cell."""

    dataset = canonical_dataset(str(task.get("dataset")))
    setting = str(task.get("setting"))
    run_type = str(task.get("run_type"))
    if dataset == "movielens":
        if run_type == "joint":
            return {str(domain): int(ROLE_SIZES[ROLE_MAPS[setting][domain]])
                    for domain in (0, 1, 2)}
        physical = int(task.get("physical_domain"))
        return {"0": int(ROLE_SIZES[ROLE_MAPS[setting][physical]])}
    if setting != "natural":
        raise ValueError("unknown KuaiRand setting: {}".format(setting))
    if run_type == "joint":
        return {str(domain): int(KUAIRAND_EXPECTED_TRAIN_COUNTS[domain])
                for domain in (0, 1, 2)}
    physical = int(task.get("physical_domain"))
    return {"0": int(KUAIRAND_EXPECTED_TRAIN_COUNTS[physical])}


def audit_pairing_fingerprints(joint_manifest: Mapping[str, object],
                               reference_manifest: Mapping[str, object],
                               physical_domain: int) -> Tuple[bool, List[str]]:
    """Compare joint physical-view and single-domain pairing fingerprints."""

    reasons: List[str] = []
    physical_views = joint_manifest.get("physical_view_fingerprints", {})
    reference_preprocessing = reference_manifest.get("preprocessing_metadata", {})
    reference_pairing = (reference_preprocessing.get("pairing_fingerprints", {})
                         if isinstance(reference_preprocessing, Mapping) else {})
    if not isinstance(physical_views, Mapping):
        physical_views = {}
    reference_split_fingerprints = reference_manifest.get("split_row_fingerprints", {})
    if not isinstance(reference_split_fingerprints, Mapping):
        reference_split_fingerprints = {}
    for split in ("train", "validation", "test"):
        physical_view = physical_views.get(str(physical_domain), {})
        joint_view = physical_view.get(split) if isinstance(physical_view, Mapping) else None
        ref_view = reference_split_fingerprints.get(split)
        if not joint_view or not ref_view or joint_view != ref_view:
            reasons.append("pairing_fingerprint_mismatch:{}".format(split))
        # The reference bundle also records the same relation explicitly;
        # require it rather than trusting only one side of the comparison.
        pair = reference_pairing.get(split, {}) if isinstance(reference_pairing, Mapping) else {}
        if pair.get("joint_remapped") != pair.get("reference") or pair.get("reference") != ref_view:
            reasons.append("reference_pairing_evidence_mismatch:{}".format(split))
    return not reasons, reasons


def required_artifact_hash_names(manifest: Mapping[str, object]) -> set[str]:
    """Return the exact non-self files whose hashes must be inventoried."""

    architecture = str(manifest.get("architecture_optimizer_artifact") or "")
    return (set(REQUIRED_FILES) - {"run_manifest.json"}) | ({architecture} if architecture else set())


def validate_artifact_hash_inventory(run_dir: str,
                                     manifest: Mapping[str, object]) -> List[str]:
    """Validate declared artifact presence, hashes, and required coverage."""

    reasons: List[str] = []
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        return ["artifact_hash_inventory_missing"]
    required_hashes = required_artifact_hash_names(manifest)
    if not required_hashes.issubset({str(name) for name in artifact_hashes}):
        reasons.append("artifact_hash_inventory_incomplete")
    if "run_manifest.json" in {str(name) for name in artifact_hashes}:
        reasons.append("artifact_hash_self_reference")
    for filename, expected_hash in artifact_hashes.items():
        path = os.path.join(str(run_dir), str(filename))
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            reasons.append("artifact_hash_invalid:" + str(filename))
        elif not os.path.isfile(path):
            reasons.append("artifact_missing:" + str(filename))
        elif _file_hash(path, fresh=True) != expected_hash:
            reasons.append("artifact_hash_mismatch:" + str(filename))
    declared_artifacts = manifest.get("artifacts")
    if not isinstance(declared_artifacts, list) or "run_manifest.json" not in declared_artifacts \
            or set(str(item) for item in declared_artifacts if item != "run_manifest.json") != set(str(item) for item in artifact_hashes):
        reasons.append("artifact_declaration_mismatch")
    return reasons


def finalize_artifact_hashes(run_dir: str) -> Dict[str, object]:
    """Close the artifact hash inventory after the child process has exited.

    The launcher owns ``stdout.log``: it writes the launch header and captures
    the child's final line through the same file handle.  Consequently the
    runner cannot hash that file before the child exits without creating a
    write-order race.  This function is called only after the subprocess and
    its stdout handle are closed, then atomically replaces the manifest.  The
    manifest itself remains excluded from the inventory, so this is not a
    self-referential hash.
    """

    canonical_run_dir = os.path.abspath(str(run_dir))
    manifest_path = os.path.join(canonical_run_dir, "run_manifest.json")
    manifest = _read_json(manifest_path)
    if not manifest or manifest.get("status") != "OK":
        raise RuntimeError("cannot finalize artifact hashes for a non-OK manifest")
    declared = manifest.get("artifact_hashes")
    if not isinstance(declared, Mapping) or not declared:
        raise RuntimeError("artifact hash inventory missing before finalization")
    if "run_manifest.json" in {str(name) for name in declared}:
        raise RuntimeError("run_manifest.json must remain outside artifact hash inventory")

    finalized: Dict[str, str] = {}
    for filename in declared:
        name = str(filename)
        artifact_path = os.path.join(canonical_run_dir, name)
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError("declared artifact missing during finalization: {}".format(name))
        # A retry reuses the canonical directory after archiving the failed
        # attempt.  Invalidate any prior process-local cache entry even when a
        # replacement happens to have the same size and timestamp granularity.
        absolute_artifact = os.path.abspath(artifact_path)
        for cache_key in tuple(_FILE_HASH_CACHE):
            if cache_key[0] == absolute_artifact:
                _FILE_HASH_CACHE.pop(cache_key, None)
        finalized[name] = _file_hash(artifact_path)
    manifest["artifact_hashes"] = finalized
    stdout_path = os.path.join(canonical_run_dir, "stdout.log")
    if not os.path.isfile(stdout_path):
        raise FileNotFoundError("stdout.log missing during finalization")
    manifest["artifact_hash_finalization"] = {
        "method": "stage_S1_orchestrator_after_child_exit",
        "stdout_log_hash": finalized.get("stdout.log", ""),
        "stdout_log_size": int(os.path.getsize(stdout_path)),
    }

    temporary_path = manifest_path + ".finalizing-{}.tmp".format(os.getpid())
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, manifest_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return manifest


def _expected_epoch_support(config: Mapping[str, object], manifest: Mapping[str, object],
                            require_formal: bool) -> List[int]:
    if require_formal:
        return list(range(EPOCHS))
    observed = manifest.get("observed_epochs", config.get("observed_epochs"))
    if isinstance(observed, list) and observed:
        try:
            values = sorted(int(x) for x in observed)
            return list(range(values[-1] + 1)) if values == list(range(values[-1] + 1)) else []
        except (TypeError, ValueError):
            return []
    epoch = _integer(config.get("epoch", manifest.get("epoch")))
    return list(range(epoch)) if epoch is not None and epoch > 0 else []


def _check_finite_rows(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bool:
    for row in rows:
        for field in fields:
            if field not in row or _numeric(row.get(field)) is None:
                return False
    return True


def _csv_contract(path: str, expected_fields: Sequence[str], expected_rows: int,
                  numeric_fields: Sequence[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    header, rows = _read_csv_with_header(path)
    reasons: List[str] = []
    if header != list(expected_fields):
        reasons.append("header_mismatch")
    if len(rows) != int(expected_rows):
        reasons.append("row_count:{}:expected:{}".format(len(rows), expected_rows))
    if rows and not _check_finite_rows(rows, numeric_fields):
        reasons.append("numeric_field_missing_or_nonfinite")
    return rows, reasons


def _check_checkpoint_finite(path: str) -> bool:
    """Load a checkpoint if torch is available and reject non-finite tensors."""

    try:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=False)
        def finite(item):
            if hasattr(item, "detach"):
                return bool(torch.isfinite(item.detach()).all().cpu())
            if isinstance(item, Mapping):
                return all(finite(child) for child in item.values())
            if isinstance(item, (list, tuple)):
                return all(finite(child) for child in item)
            return True
        return finite(value)
    except TypeError:
        # PyTorch versions before weights_only support.
        try:
            import torch
            value = torch.load(path, map_location="cpu")
            def finite(item):
                if hasattr(item, "detach"):
                    return bool(torch.isfinite(item.detach()).all().cpu())
                if isinstance(item, Mapping):
                    return all(finite(child) for child in item.values())
                if isinstance(item, (list, tuple)):
                    return all(finite(child) for child in item)
                return True
            return finite(value)
        except Exception:
            return False
    except Exception:
        return False


def validate_raw_prediction_metrics(payload: Mapping[str, object],
                                    manifest: Mapping[str, object],
                                    task: Mapping[str, object],
                                    test_rows: Sequence[Mapping[str, object]],
                                    summary_rows: Sequence[Mapping[str, object]]) -> List[str]:
    """Recompute every deferred test cell from raw predictions.

    The validator intentionally does not trust ``metrics_test.csv`` or
    ``metrics_summary.csv`` for AUC/LogLoss.  It reconstructs task-major
    cells from the row-aligned NPZ payload and compares both CSV contracts
    within one fixed numeric tolerance.
    """

    import numpy as np

    reasons: List[str] = []
    checkpoints = ("selected", "final") + tuple("common_epoch_{}".format(e) for e in COMMON_EPOCHS)
    try:
        labels = np.asarray(payload["labels"])
        physical = np.asarray(payload["physical_domain"])
        domain_num = int(manifest.get("domain_num"))
        local_to_physical = [int(value) for value in manifest.get("local_to_physical", [])]
        if labels.ndim != 2 or labels.shape[1] != 2 or physical.ndim != 1 \
                or labels.shape[0] != physical.shape[0]:
            return ["raw_metric_shape_invalid"]
        if domain_num not in (1, 3) or len(local_to_physical) != domain_num:
            return ["raw_metric_domain_mapping_invalid"]
        task_definitions = manifest.get("preprocessing_metadata", {})
        if isinstance(task_definitions, Mapping):
            task_definitions = task_definitions.get("label_definitions", {})
        if not isinstance(task_definitions, Mapping):
            task_definitions = {}
        expected_task_identity = {
            task_id: str(task_definitions.get("task{}".format(task_id), "task{}".format(task_id)))
            for task_id in (0, 1)
        }
        physical_identities = manifest.get("physical_domain_identity", [])
        domain_shares = manifest.get("domain_shares", {})
        row_map: Dict[Tuple[str, int, int], Mapping[str, object]] = {}
        for row in test_rows:
            key = (str(row.get("checkpoint")), _integer(row.get("task")),
                   _integer(row.get("domain")))
            if key in row_map:
                reasons.append("raw_metric_duplicate_cell:{}".format(key))
            row_map[key] = row
        summary_map: Dict[Tuple[str, int, int], Mapping[str, object]] = {}
        for row in summary_rows:
            key = (str(row.get("checkpoint")), _integer(row.get("task")),
                   _integer(row.get("domain")))
            if key in summary_map:
                reasons.append("raw_metric_duplicate_summary_cell:{}".format(key))
            summary_map[key] = row
        if len(row_map) != len(checkpoints) * domain_num * 2:
            reasons.append("raw_metric_support_incomplete")
        if len(summary_map) != domain_num * 2:
            reasons.append("raw_metric_summary_support_incomplete")
        for checkpoint in checkpoints:
            prediction = np.asarray(payload.get("pred_{}".format(checkpoint)))
            if prediction.shape != labels.shape:
                reasons.append("raw_metric_prediction_shape:{}".format(checkpoint))
                continue
            for task_id in (0, 1):
                for domain in range(domain_num):
                    key = (checkpoint, task_id, domain)
                    row = row_map.get(key)
                    if row is None:
                        reasons.append("raw_metric_row_missing:{}".format(key))
                        continue
                    target_physical = (int(manifest.get("physical_domain"))
                                       if domain_num == 1 else local_to_physical[domain])
                    mask = physical == target_physical
                    try:
                        computed = cell_metrics(labels[mask, task_id], prediction[mask, task_id])
                    except Exception as exc:
                        reasons.append("raw_metric_recompute_failed:{}:{}".format(key, type(exc).__name__))
                        continue
                    expected_cell = task_domain_cell_index(task_id, domain, domain_num)
                    if _integer(row.get("cell")) != expected_cell:
                        reasons.append("raw_metric_cell_index:{}".format(key))
                    if _integer(row.get("physical_domain")) != target_physical:
                        reasons.append("raw_metric_physical_domain:{}".format(key))
                    expected_physical_identity = (physical_identities[domain]
                                                  if isinstance(physical_identities, list)
                                                  and len(physical_identities) == domain_num else "")
                    if str(row.get("physical_domain_identity", "")) != str(expected_physical_identity):
                        reasons.append("raw_metric_physical_identity:{}".format(key))
                    expected_share = (domain_shares.get(str(domain))
                                      if isinstance(domain_shares, Mapping) else None)
                    observed_share = _numeric(row.get("domain_share"))
                    if expected_share is None or observed_share is None \
                            or abs(observed_share - float(expected_share)) > 1e-12:
                        reasons.append("raw_metric_domain_share:{}".format(key))
                    if str(row.get("task_identity", "")) != expected_task_identity[task_id]:
                        reasons.append("raw_metric_task_identity:{}".format(key))
                    for field in ("sample_count", "positive_count"):
                        csv_field = "test_{}".format(field)
                        if _integer(row.get(csv_field)) != int(computed[field]):
                            reasons.append("raw_metric_{}:{}".format(csv_field, key))
                    for field in ("positive_rate", "auc", "logloss"):
                        csv_field = "test_positive_rate" if field == "positive_rate" else field
                        observed = _numeric(row.get(csv_field))
                        if observed is None or abs(observed - float(computed[field])) > 1e-12:
                            reasons.append("raw_metric_value:{}:{}".format(csv_field, key))
                    if checkpoint == "selected":
                        summary = summary_map.get(key)
                        if summary is None:
                            reasons.append("raw_metric_summary_missing:{}".format(key))
                        else:
                            for field in ("auc", "logloss"):
                                observed = _numeric(summary.get(field))
                                if observed is None or abs(observed - float(computed[field])) > 1e-12:
                                    reasons.append("raw_metric_summary_value:{}:{}".format(field, key))
    except Exception as exc:
        reasons.append("raw_metric_recompute_exception:{}".format(type(exc).__name__))
    return reasons


def validate_run(task: Mapping[str, object], require_formal: bool = True) -> Tuple[bool, List[str]]:
    """Fail-closed validation of every formal raw artifact and declared counter."""

    run_dir = os.path.abspath(str(task.get("run_dir", "")))
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing"]
    reasons: List[str] = []
    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(run_dir, name))]
    if missing:
        reasons.append("missing_files:" + ",".join(missing))
        return False, reasons
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    config = _read_json(os.path.join(run_dir, "config.json"))
    if manifest.get("status") != "OK":
        reasons.append("manifest_not_OK")
    if manifest.get("stage") != "S1" or config.get("stage") != "S1":
        reasons.append("stage_identity_mismatch")
    expected_key = str(task.get("formal_run_key", ""))
    if manifest.get("formal_run_key") != expected_key:
        reasons.append("formal_key_mismatch")
    for field in ("dataset", "setting", "backbone", "seed", "run_type"):
        if field == "dataset":
            try:
                observed_dataset = canonical_dataset(str(manifest.get(field)))
                expected_dataset = canonical_dataset(str(task.get(field)))
            except (TypeError, ValueError):
                observed_dataset = str(manifest.get(field))
                expected_dataset = str(task.get(field))
            if observed_dataset != expected_dataset:
                reasons.append("identity_mismatch:dataset")
        elif str(manifest.get(field)) != str(task.get(field)):
            reasons.append("identity_mismatch:" + field)
    try:
        expected_dataset = canonical_dataset(str(task.get("dataset")))
    except (TypeError, ValueError):
        expected_dataset = str(task.get("dataset"))
    if str(manifest.get("dataset_key", "")) != expected_dataset:
        reasons.append("dataset_key_mismatch")
    if task.get("run_type") == "single_domain" and _integer(manifest.get("physical_domain")) != int(task.get("physical_domain")):
        reasons.append("physical_domain_mismatch")
    if manifest.get("test_evaluated_during_training") is not False \
            or config.get("test_evaluated_during_training") is not False:
        reasons.append("test_before_selection")
    if manifest.get("checkpoint_selection_frozen_before_test") is not True \
            or config.get("test_evaluated_after_selection") is not True:
        reasons.append("selection_freeze_evidence_missing")
    if manifest.get("test_metric_in_epoch_history") is not False:
        reasons.append("test_metric_in_epoch_history")
    inventory = _read_json_any(os.path.join(run_dir, "nonfinite_inventory.json"))
    if not isinstance(inventory, list) or inventory:
        reasons.append("nonfinite_inventory_invalid_or_nonempty")
    if os.path.getsize(os.path.join(run_dir, "stdout.log")) <= 0:
        reasons.append("stdout_artifact_empty")

    # Formal runs have no deviation escape hatch.  Keep the non-formal switch
    # only for explicitly isolated structural smoke fixtures.
    observed_protocol = manifest.get("formal_protocol_observed")
    if require_formal:
        try:
            deviation_flag = int(config.get("allow_protocol_deviation", 0) or 0)
        except (TypeError, ValueError):
            deviation_flag = 1
        if deviation_flag != 0:
            reasons.append("protocol_deviation_flag_present")
        try:
            nonformal_seed_flag = int(config.get("allow_nonformal_seed", 0) or 0)
        except (TypeError, ValueError):
            nonformal_seed_flag = 1
        if nonformal_seed_flag != 0:
            reasons.append("nonformal_seed_flag_present")
        if manifest.get("formal_valid") is not True or config.get("formal_valid") is not True:
            reasons.append("formal_protocol_invalid")
        if manifest.get("formal_protocol") != dict(FORMAL_PROTOCOL):
            reasons.append("formal_protocol_mismatch")
        if config.get("formal_protocol") != dict(FORMAL_PROTOCOL):
            reasons.append("config_formal_protocol_mismatch")
        if observed_protocol != dict(FORMAL_PROTOCOL) or config.get("formal_protocol_observed") != dict(FORMAL_PROTOCOL):
            reasons.append("formal_protocol_observed_mismatch")
        if manifest.get("formal_protocol_mismatches") != {} or config.get("formal_protocol_mismatches") != {}:
            reasons.append("formal_protocol_mismatch_inventory_nonempty")
        if protocol_mismatches(observed_protocol if isinstance(observed_protocol, Mapping) else {}):
            reasons.append("formal_protocol_field_mismatch")
        config_protocol_fields = ("batch_size", "epoch", "learning_rate", "learning_rate_darts",
                                  "weight_decay", "expert_num", "shadow_patience", "num_workers",
                                  "device", "fcn_dims")
        expected_config_fields = {
            "batch_size": FORMAL_PROTOCOL["batch_size"], "epoch": FORMAL_PROTOCOL["epoch"],
            "learning_rate": FORMAL_PROTOCOL["learning_rate"],
            "learning_rate_darts": FORMAL_PROTOCOL["learning_rate_darts"],
            "weight_decay": FORMAL_PROTOCOL["weight_decay"], "expert_num": FORMAL_PROTOCOL["expert_num"],
            "shadow_patience": FORMAL_PROTOCOL["shadow_patience"],
            "num_workers": FORMAL_PROTOCOL["num_workers"], "device": FORMAL_PROTOCOL["device"],
            "fcn_dims": FORMAL_PROTOCOL["fcn_dims"],
        }
        for field in config_protocol_fields:
            if config.get(field) != expected_config_fields[field]:
                reasons.append("config_protocol_field_mismatch:" + field)

    domain_num, domains = _formal_cells(task)
    expected_cells = 2 * domain_num
    train_counts = manifest.get("train_domain_counts")
    if not isinstance(train_counts, Mapping) or len(train_counts) != domain_num:
        reasons.append("train_domain_counts_missing_or_wrong_cardinality")
        train_total = 0
    else:
        try:
            train_total = sum(int(value) for value in train_counts.values())
        except (TypeError, ValueError):
            train_total = 0
            reasons.append("train_domain_counts_noninteger")
        try:
            exact_counts = expected_train_domain_counts(task)
            observed_counts = {str(key): int(value) for key, value in train_counts.items()}
            if observed_counts != exact_counts:
                reasons.append("train_domain_counts_exact_mismatch")
            total_exact = sum(exact_counts.values())
            expected_shares = {str(key): float(value) / float(total_exact)
                               for key, value in exact_counts.items()}
            observed_shares = manifest.get("domain_shares")
            if not isinstance(observed_shares, Mapping) or set(str(k) for k in observed_shares) != set(expected_shares):
                reasons.append("domain_shares_support_mismatch")
            else:
                for key, expected_share in expected_shares.items():
                    observed_share = _numeric(observed_shares.get(key))
                    if observed_share is None or abs(observed_share - expected_share) > 1e-12:
                        reasons.append("domain_share_exact_mismatch:{}".format(key))
        except (TypeError, ValueError, KeyError):
            reasons.append("train_domain_counts_expected_contract_error")
    batch_size = int(FORMAL_PROTOCOL["batch_size"])
    batches_per_epoch = (train_total + batch_size - 1) // batch_size if train_total else 0
    if manifest.get("batches_per_epoch") != batches_per_epoch:
        reasons.append("batches_per_epoch_mismatch")

    metrics, metric_reasons = _csv_contract(
        os.path.join(run_dir, "metrics_epoch.csv"), METRICS_EPOCH_FIELDS,
        EPOCHS * expected_cells,
        ("seed", "epoch", "domain", "physical_domain", "task", "cell",
         "train_sample_count", "validation_sample_count", "train_positive_rate",
         "validation_positive_rate", "train_cell_logloss", "validation_auc",
         "validation_logloss", "validation_macro_auc", "validation_macro_logloss",
         "target_weighted_validation_auc", "target_weighted_validation_logloss",
         "worst_domain_validation_auc", "worst_cell_validation_auc", "train_objective",
         "main_optimizer_steps", "architecture_optimizer_steps", "learning_rate",
         "architecture_learning_rate", "selected_checkpoint", "fixed_final", "common_epoch"))
    reasons.extend("metrics_epoch_" + item for item in metric_reasons)
    expected_support = {(epoch, task_id, domain)
                        for epoch in range(EPOCHS)
                        for task_id in (0, 1) for domain in domains}
    support = set()
    for row in metrics:
        epoch, task_id, domain = _integer(row.get("epoch")), _integer(row.get("task")), _integer(row.get("domain"))
        if epoch is not None and task_id is not None and domain is not None:
            support.add((epoch, task_id, domain))
    if support != expected_support:
        reasons.append("metrics_epoch_cell_support_mismatch")
    if any(row.get("test_auc") or row.get("test_logloss") for row in metrics):
        reasons.append("test_fields_in_epoch_history")
    if metrics:
        first_epoch_rows = [row for row in metrics if _integer(row.get("epoch")) == 0 and _integer(row.get("task")) == 0]
        observed_counts = {_integer(row.get("domain")): _integer(row.get("train_sample_count"))
                           for row in first_epoch_rows}
        expected_counts = {int(key): int(value) for key, value in (train_counts or {}).items()}
        if observed_counts != expected_counts:
            reasons.append("metric_train_domain_counts_mismatch")
        cell_stats = manifest.get("cell_stats")
        if not isinstance(cell_stats, Mapping) or not isinstance(cell_stats.get("train"), Mapping):
            reasons.append("cell_stats_train_missing")
        else:
            for row in metrics:
                domain = _integer(row.get("domain"))
                task_id = _integer(row.get("task"))
                if domain is None or task_id is None:
                    continue
                stat = cell_stats["train"].get("d{}t{}".format(domain, task_id))
                if not isinstance(stat, Mapping) or _integer(row.get("train_sample_count")) != _integer(stat.get("count")):
                    reasons.append("metric_cell_train_count_mismatch")
                    break

    valid_checkpoints = {"selected", "final"} | {"common_epoch_{}".format(e) for e in COMMON_EPOCHS}
    test_rows, test_reasons = _csv_contract(
        os.path.join(run_dir, "metrics_test.csv"), METRICS_TEST_FIELDS,
        expected_cells * len(valid_checkpoints),
        ("seed", "stage_epoch", "domain", "physical_domain", "task", "cell",
         "train_sample_count", "validation_sample_count", "test_sample_count",
         "test_positive_rate", "auc", "logloss", "validation_auc_at_stage",
         "validation_logloss_at_stage", "validation_macro_auc_at_stage",
         "validation_macro_logloss_at_stage", "selected_epoch",
         "selected_validation_macro_auc", "parameter_count", "domain_share",
         "train_positive_count", "validation_positive_count", "test_positive_count",
         "train_positive_rate", "validation_positive_rate",
         "train_label_entropy", "validation_label_entropy", "test_label_entropy"))
    reasons.extend("metrics_test_" + item for item in test_reasons)
    if {row.get("checkpoint") for row in test_rows} != valid_checkpoints:
        reasons.append("test_checkpoint_support_mismatch")
    test_keys = [(row.get("checkpoint"), row.get("task"), row.get("domain")) for row in test_rows]
    if len(test_keys) != len(set(test_keys)):
        reasons.append("test_metric_duplicate_cell")
    summary, summary_reasons = _csv_contract(
        os.path.join(run_dir, "metrics_summary.csv"), METRICS_TEST_FIELDS,
        expected_cells, ("seed", "stage_epoch", "domain", "physical_domain", "task", "cell",
                          "test_sample_count", "test_positive_rate", "auc", "logloss"))
    reasons.extend("metrics_summary_" + item for item in summary_reasons)
    if any(row.get("checkpoint") != "selected" for row in summary):
        reasons.append("summary_not_selected_only")

    # Recompute validation selection from the full epoch history (strict
    # earliest-tie rule) rather than trusting a manifest boolean.
    by_epoch: Dict[int, float] = {}
    for row in metrics:
        epoch = _integer(row.get("epoch"))
        score = _numeric(row.get("validation_macro_auc"))
        if epoch is not None and score is not None:
            by_epoch.setdefault(epoch, score)
            if abs(by_epoch[epoch] - score) > 1e-12:
                reasons.append("validation_macro_auc_inconsistent_within_epoch")
    if len(by_epoch) != EPOCHS:
        reasons.append("validation_epoch_summary_missing")
        selected_epoch = None
    else:
        selected_epoch = min(by_epoch, key=lambda epoch: (-by_epoch[epoch], epoch))
        if _integer(manifest.get("selected_epoch")) != selected_epoch:
            reasons.append("selected_epoch_recomputed_mismatch")
        selected_score = _numeric(manifest.get("selected_validation_macro_auc"))
        if selected_score is None or abs(selected_score - by_epoch[selected_epoch]) > 1e-10:
            reasons.append("selected_validation_score_mismatch")
    if manifest.get("fixed_final_epoch") != 49 or manifest.get("common_epochs") != list(COMMON_EPOCHS):
        reasons.append("checkpoint_epoch_contract_invalid")

    batch_rows, batch_reasons = _csv_contract(
        os.path.join(run_dir, "batch_trace.csv"), BATCH_FIELDS,
        EPOCHS * (batches_per_epoch + (1 if str(task.get("backbone")) == "m3oe" else 0)),
        ("seed", "epoch", "step", "batch_size", "train_objective"))
    reasons.extend("batch_trace_" + item for item in batch_reasons)
    train_rows = [row for row in batch_rows if row.get("kind") == "train"]
    arch_rows = [row for row in batch_rows if row.get("kind") == "architecture"]
    if len(train_rows) != EPOCHS * batches_per_epoch:
        reasons.append("train_batch_row_count_mismatch")
    if len(arch_rows) != (EPOCHS if str(task.get("backbone")) == "m3oe" else 0):
        reasons.append("architecture_batch_row_count_mismatch")
    for epoch in range(EPOCHS):
        epoch_rows = [row for row in train_rows if _integer(row.get("epoch")) == epoch]
        if sorted(_integer(row.get("step")) for row in epoch_rows) != list(range(batches_per_epoch)):
            reasons.append("train_step_support_mismatch:{}".format(epoch))
        if sum(_integer(row.get("batch_size")) or 0 for row in epoch_rows) != train_total:
            reasons.append("train_exposure_mismatch:{}".format(epoch))
    if any(not row.get("batch_fingerprint") for row in batch_rows):
        reasons.append("batch_fingerprint_missing")
    ordered_train = sorted(train_rows, key=lambda row: (_integer(row.get("epoch")) or -1,
                                                         _integer(row.get("step")) or -1))
    ordered_arch = sorted(arch_rows, key=lambda row: (_integer(row.get("epoch")) or -1,
                                                        _integer(row.get("step")) or -1))
    train_hash = hashlib.sha256(json.dumps([row.get("batch_fingerprint") for row in ordered_train]).encode()).hexdigest()
    arch_hash = hashlib.sha256(json.dumps([row.get("batch_fingerprint") for row in ordered_arch]).encode()).hexdigest()
    if manifest.get("batch_order_hash") != train_hash or manifest.get("architecture_batch_order_hash") != arch_hash:
        reasons.append("batch_order_hash_mismatch")
    if manifest.get("train_batches_total") != len(train_rows) or manifest.get("batch_fingerprint_count") != len(train_rows):
        reasons.append("train_batch_counter_mismatch")
    if manifest.get("architecture_batch_fingerprint_count") != len(arch_rows):
        reasons.append("architecture_batch_counter_mismatch")
    expected_optimizer_steps = EPOCHS * batches_per_epoch
    expected_arch_steps = EPOCHS if str(task.get("backbone")) == "m3oe" else 0
    if manifest.get("optimizer_steps") != expected_optimizer_steps \
            or manifest.get("examples_processed") != EPOCHS * train_total:
        reasons.append("optimizer_or_example_counter_mismatch")
    if manifest.get("architecture_optimizer_steps") != expected_arch_steps \
            or manifest.get("scheduler_steps") != 0:
        reasons.append("architecture_or_scheduler_counter_mismatch")
    if manifest.get("architecture_examples_processed") != sum(_integer(row.get("batch_size")) or 0 for row in arch_rows):
        reasons.append("architecture_example_counter_mismatch")
    optimizer = manifest.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("name") != "Adam" \
            or _numeric(optimizer.get("lr")) != float(FORMAL_PROTOCOL["learning_rate"]) \
            or optimizer.get("betas") != list(FORMAL_PROTOCOL["betas"]) \
            or _numeric(optimizer.get("weight_decay")) != float(FORMAL_PROTOCOL["weight_decay"]) \
            or optimizer.get("scheduler") is not None:
        reasons.append("main_optimizer_protocol_mismatch")
    elif optimizer.get("audit", {}).get("step_max") != expected_optimizer_steps:
        reasons.append("main_optimizer_step_audit_mismatch")
    architecture_optimizer = manifest.get("architecture_optimizer")
    if str(task.get("backbone")) == "m3oe":
        if not isinstance(architecture_optimizer, Mapping) \
                or architecture_optimizer.get("name") != "Adam" \
                or _numeric(architecture_optimizer.get("lr")) != float(FORMAL_PROTOCOL["learning_rate_darts"]) \
                or architecture_optimizer.get("betas") != list(FORMAL_PROTOCOL["architecture_betas"]):
            reasons.append("architecture_optimizer_protocol_mismatch")
        elif architecture_optimizer.get("audit", {}).get("step_max") != expected_arch_steps:
            reasons.append("architecture_optimizer_step_audit_mismatch")
    elif architecture_optimizer is not None:
        reasons.append("non_m3oe_architecture_optimizer_metadata_present")

    # Validate the declared domain/task exposure for every recorded batch.
    import json as _json
    for row in batch_rows:
        try:
            domain_counts = [int(value) for value in _json.loads(row.get("domain_counts", "[]"))]
            task_counts = [int(value) for value in _json.loads(row.get("task_counts", "[]"))]
            size = _integer(row.get("batch_size"))
            if len(domain_counts) != domain_num or len(task_counts) != 2 \
                    or any(value < 0 for value in domain_counts + task_counts) \
                    or sum(domain_counts) != size or any(value != size for value in task_counts):
                reasons.append("batch_exposure_contract_mismatch")
                break
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("batch_exposure_json_invalid")
            break
    for row in metrics:
        epoch = _integer(row.get("epoch"))
        if epoch is None:
            continue
        expected_main = (epoch + 1) * batches_per_epoch
        expected_arch = epoch + 1 if str(task.get("backbone")) == "m3oe" else 0
        if _integer(row.get("main_optimizer_steps")) != expected_main \
                or _integer(row.get("architecture_optimizer_steps")) != expected_arch:
            reasons.append("epoch_optimizer_step_trace_mismatch")
            break
    selected_markers = {(int(row.get("epoch")), int(row.get("selected_checkpoint"))) for row in metrics
                        if _integer(row.get("epoch")) is not None and _integer(row.get("selected_checkpoint")) is not None}
    if selected_epoch is not None and selected_markers != {(selected_epoch, 1), *[(epoch, 0) for epoch in range(EPOCHS) if epoch != selected_epoch]}:
        reasons.append("selected_checkpoint_marker_mismatch")

    audit = _read_json(os.path.join(run_dir, "initialization_audit.json"))
    if not audit or not isinstance(audit.get("mapping_table"), list) \
            or not isinstance(audit.get("unmapped_tensors"), list) \
            or not isinstance(audit.get("unmapped_tensor_subseeds"), Mapping):
        reasons.append("initialization_audit_missing_or_incomplete")
    else:
        unmapped = set(audit.get("unmapped_tensors", []))
        subseeds = audit.get("unmapped_tensor_subseeds", {})
        if unmapped != set(subseeds):
            reasons.append("unmapped_subseed_inventory_mismatch")
        for target_name, observed_seed in subseeds.items():
            digest = hashlib.sha256((str(int(manifest.get("seed", 0))) + "|" + str(target_name)).encode("utf-8")).digest()
            expected_seed = int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)
            if int(observed_seed) != expected_seed:
                reasons.append("unmapped_subseed_not_name_derived:{}".format(target_name))
        mapped_targets = [entry.get("target") for entry in audit["mapping_table"]]
        if len(mapped_targets) != len(set(mapped_targets)):
            reasons.append("semantic_mapping_duplicate_target")
        target_names = set(audit.get("target_parameter_manifest", {}).get("names", []))
        if target_names and set(mapped_targets) != target_names:
            reasons.append("semantic_mapping_target_manifest_mismatch")
        for entry in audit["mapping_table"]:
            if not entry.get("target_hash"):
                reasons.append("semantic_target_hash_missing")
            if entry.get("source") and entry.get("kind") != "name_derived_subseed":
                if not entry.get("source_hash") or not entry.get("source_slice_hash") \
                        or not entry.get("target_slice_hash") or entry.get("hash_equal") is not True:
                    reasons.append("semantic_source_target_hash_mismatch:{}".format(entry.get("target")))
    parameter_manifest = _read_json(os.path.join(run_dir, "parameter_manifest.json"))
    if not parameter_manifest or not parameter_manifest.get("names"):
        reasons.append("parameter_manifest_missing")

    # Persisted predictions are the source for all pooled/assembled report
    # tables; reject missing keys, row-order drift, and non-finite values.
    try:
        import numpy as np
        with np.load(os.path.join(run_dir, "predictions_test.npz"), allow_pickle=False) as payload:
            required_prediction_keys = {"row_ids", "physical_domain", "labels"} \
                | {"pred_{}".format(checkpoint) for checkpoint in valid_checkpoints}
            if set(payload.files) != required_prediction_keys:
                reasons.append("prediction_key_support_mismatch")
            row_ids = np.asarray(payload["row_ids"])
            labels = np.asarray(payload["labels"])
            physical = np.asarray(payload["physical_domain"])
            if row_ids.ndim != 1 or labels.ndim != 2 or physical.shape != row_ids.shape \
                    or labels.shape[0] != row_ids.shape[0] or labels.shape[1] != 2:
                reasons.append("prediction_shape_mismatch")
            if not np.isfinite(labels).all() or not np.isfinite(physical).all():
                reasons.append("prediction_labels_nonfinite")
            for checkpoint in valid_checkpoints:
                values = np.asarray(payload["pred_{}".format(checkpoint)])
                if values.shape != labels.shape or not np.isfinite(values).all():
                    reasons.append("prediction_shape_or_finite_mismatch:{}".format(checkpoint))
            raw_payload = {name: np.asarray(payload[name]) for name in payload.files}
            reasons.extend(validate_raw_prediction_metrics(
                raw_payload, manifest, task, test_rows, summary))
            row_hash = hashlib.sha256(np.asarray(row_ids, dtype=np.int64).tobytes()).hexdigest()
            label_hash = hashlib.sha256(np.asarray(labels, dtype=np.float32).tobytes()).hexdigest()
            if manifest.get("prediction_row_ids_hash") != row_hash or manifest.get("prediction_labels_hash") != label_hash:
                reasons.append("prediction_hash_mismatch")
            row_array_hash = _array_hash(np.asarray(row_ids, dtype=np.int64))
            label_array_hash = _array_hash(np.asarray(labels, dtype=np.float32))
            if manifest.get("prediction_row_ids_array_hash") != row_array_hash \
                    or manifest.get("prediction_labels_array_hash") != label_array_hash:
                reasons.append("prediction_array_hash_mismatch")
            declared_split_ids = manifest.get("split_row_id_hashes", {})
            if isinstance(declared_split_ids, Mapping) \
                    and declared_split_ids.get("test") != row_array_hash:
                reasons.append("prediction_row_ids_split_hash_mismatch")
            if manifest.get("prediction_checkpoint_keys") != sorted(valid_checkpoints):
                reasons.append("prediction_checkpoint_manifest_mismatch")
            # Cross-check raw row/domain exposure against the deferred CSV
            # counts instead of trusting either artifact in isolation.
            for domain in domains:
                for task_id in (0, 1):
                    mask = physical == int(manifest.get("local_to_physical", list(domains))[domain]) \
                        if domain_num == 3 else physical == int(manifest.get("physical_domain"))
                    expected_count = int(mask.sum())
                    observed = [row for row in test_rows if row.get("checkpoint") == "selected"
                                and _integer(row.get("domain")) == domain and _integer(row.get("task")) == task_id]
                    if observed and _integer(observed[0].get("test_sample_count")) != expected_count:
                        reasons.append("prediction_test_exposure_mismatch")
                    if observed and int(np.asarray(labels[mask, task_id]).sum()) != _integer(observed[0].get("test_positive_count")):
                        reasons.append("prediction_test_label_count_mismatch")
    except Exception as exc:
        reasons.append("prediction_artifact_invalid:{}".format(type(exc).__name__))

    architecture_artifact = str(manifest.get("architecture_optimizer_artifact") or "")
    if architecture_artifact not in ARCHITECTURE_FILES or not os.path.isfile(os.path.join(run_dir, str(architecture_artifact))):
        reasons.append("architecture_optimizer_artifact_missing")
    if str(task.get("backbone")) == "m3oe" and architecture_artifact != "architecture_optimizer_state.pth":
        reasons.append("m3oe_architecture_optimizer_absent")
    if str(task.get("backbone")) != "m3oe" and architecture_artifact != "architecture_optimizer_absent.json":
        reasons.append("non_m3oe_architecture_optimizer_present")

    # Every declared artifact hash is checked, including optimizer/RNG/log and
    # raw prediction files.  Failed-attempt archives are intentionally ignored.
    reasons.extend(validate_artifact_hash_inventory(run_dir, manifest))
    finalization = manifest.get("artifact_hash_finalization")
    if require_formal:
        if not isinstance(finalization, Mapping):
            reasons.append("artifact_hash_finalization_missing")
        else:
            if finalization.get("method") != "stage_S1_orchestrator_after_child_exit":
                reasons.append("artifact_hash_finalization_method_mismatch")
            artifact_hashes_for_log = manifest.get("artifact_hashes", {})
            if not isinstance(artifact_hashes_for_log, Mapping) or \
                    finalization.get("stdout_log_hash") != artifact_hashes_for_log.get("stdout.log"):
                reasons.append("stdout_final_hash_cross_reference_mismatch")
            stdout_path = os.path.join(run_dir, "stdout.log")
            if os.path.isfile(stdout_path) and finalization.get("stdout_log_size") != os.path.getsize(stdout_path):
                reasons.append("stdout_final_size_mismatch")
    artifact_hashes = manifest.get("artifact_hashes")
    expected_checkpoint_names = {"selected_checkpoint.pth", "final_checkpoint.pth"} \
        | {"common_epoch_{}.pth".format(epoch) for epoch in COMMON_EPOCHS}
    if not isinstance(manifest.get("checkpoints"), Mapping) \
            or set(manifest.get("checkpoints", {})) != expected_checkpoint_names:
        reasons.append("checkpoint_hash_inventory_mismatch")
    elif isinstance(artifact_hashes, Mapping):
        for checkpoint_name in expected_checkpoint_names:
            if manifest["checkpoints"].get(checkpoint_name) != artifact_hashes.get(checkpoint_name):
                reasons.append("checkpoint_hash_cross_reference_mismatch:" + checkpoint_name)
    for filename in REQUIRED_FILES + ((architecture_artifact,) if architecture_artifact else ()):
        path = os.path.join(run_dir, filename)
        if filename.endswith(".pth") and os.path.isfile(path) and not _check_checkpoint_finite(path):
            reasons.append("checkpoint_nonfinite_or_corrupt:" + filename)

    # Raw RNG restoration evidence is part of the formal block, not an
    # optional self-reported diagnostic.
    evidence = manifest.get("rng_restore_evidence")
    expected_evidence = EPOCHS * (2 if str(task.get("backbone")) == "m3oe" else 1)
    if not isinstance(evidence, list) or len(evidence) != expected_evidence \
            or any(not isinstance(item, Mapping) or item.get("restored") is not True
                   or item.get("before") != item.get("after") for item in (evidence or [])):
        reasons.append("rng_restore_evidence_invalid")
    if not os.path.isfile(os.path.join(run_dir, "rng_state.pth")):
        reasons.append("raw_rng_state_missing")

    # Data/code provenance and all covariate fields must be present.
    for field in ("split_row_fingerprints", "split_row_id_hashes", "cell_stats",
                  "domain_shares", "physical_domain_identity", "preprocessing_metadata",
                  "features"):
        if not manifest.get(field):
            reasons.append("manifest_field_missing:" + field)
    raw_file = manifest.get("dataset_identity")
    if not raw_file or not os.path.isfile(str(raw_file)):
        reasons.append("dataset_file_missing")
    else:
        try:
            if manifest.get("dataset_file_hash") != _file_hash(str(raw_file)):
                reasons.append("dataset_file_hash_mismatch")
        except OSError:
            reasons.append("dataset_file_hash_unreadable")
    code_hashes = manifest.get("code_hashes")
    if not isinstance(code_hashes, Mapping) or set(str(key) for key in code_hashes) != set(CODE_HASH_FILES):
        reasons.append("code_hash_inventory_missing_or_incomplete")
    else:
        for filename in CODE_HASH_FILES:
            code_path = os.path.join(HERE, filename)
            try:
                if str(code_hashes.get(filename)) != _file_hash(code_path):
                    reasons.append("code_hash_mismatch:" + filename)
            except OSError:
                reasons.append("code_hash_unreadable:" + filename)
        if manifest.get("source_code_hash") != code_hashes.get("stage_S1_run.py"):
            reasons.append("source_code_hash_cross_reference_mismatch")
        if config.get("code_hashes") != dict(code_hashes):
            reasons.append("config_code_hash_inventory_mismatch")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        reasons.append("runtime_metadata_missing")
    else:
        required_runtime_fields = ("python_executable", "python_version", "torch_version",
                                   "cuda_available", "cuda_version", "device_identity",
                                   "device_index", "device_name", "torch_cudnn_version")
        if any(field not in runtime for field in required_runtime_fields):
            reasons.append("runtime_metadata_incomplete")
        if runtime.get("cuda_available") is not True or runtime.get("device_identity") != "cuda:0":
            reasons.append("runtime_cuda_contract_mismatch")
        fingerprint = hashlib.sha256(
            json.dumps(dict(runtime), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("runtime_fingerprint") != fingerprint or config.get("runtime_fingerprint") != fingerprint:
            reasons.append("runtime_fingerprint_mismatch")
        if config.get("runtime") != dict(runtime):
            reasons.append("config_runtime_mismatch")
    return not reasons, reasons


def build_tasks(raw_root: str = "experiments/stage_S1_raw_results") -> List[Dict[str, object]]:
    return build_matrix(raw_root)


def _validate_matrix_wrapper(tasks: Sequence[Mapping[str, object]]) -> bool:
    try:
        validate_matrix(tasks)
        return True
    except ValueError:
        return False


def validate_matrix_tasks(tasks: Sequence[Mapping[str, object]]) -> Tuple[bool, List[str]]:
    try:
        validate_matrix(tasks)
        return True, []
    except ValueError as exc:
        return False, [str(exc)]


def command_for(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    python_executable = str(getattr(args, "python", sys.executable))
    command = [python_executable, os.path.join(os.path.dirname(__file__), "stage_S1_run.py"),
               "--run_type", str(task["run_type"]), "--dataset", str(task["dataset"]),
               "--setting", str(task["setting"]), "--backbone", str(task["backbone"]),
                "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
                "--batch_size", str(FORMAL_PROTOCOL["batch_size"]),
                "--epoch", str(FORMAL_PROTOCOL["epoch"]),
                "--learning_rate", str(FORMAL_PROTOCOL["learning_rate"]),
                "--learning_rate_darts", str(FORMAL_PROTOCOL["learning_rate_darts"]),
                "--weight_decay", str(FORMAL_PROTOCOL["weight_decay"]),
                "--expert_num", str(FORMAL_PROTOCOL["expert_num"]),
                "--device", str(FORMAL_PROTOCOL["device"]),
               "--dataset_path", str(
                   getattr(args, "movielens_path" if task.get("dataset") == "movielens"
                           else "kuairand_path", "")
                   or getattr(args, "dataset_path", "")),
               "--num_workers", "0"]
    if task.get("run_type") == "single_domain":
        command.extend(["--physical_domain", str(task["physical_domain"])])
    command.extend(["--fcn_dims"] + [str(value) for value in FORMAL_PROTOCOL["fcn_dims"]])
    return command


def assert_formal_launcher(args: argparse.Namespace) -> None:
    expected = {
        "batch_size": 40960, "epoch": 50, "learning_rate": 0.01,
        "learning_rate_darts": 0.001, "weight_decay": 1e-5,
        "expert_num": 4, "shadow_patience": 10, "device": "cuda:0",
        "num_workers": 0,
    }
    for field, value in expected.items():
        observed = getattr(args, field, FORMAL_PROTOCOL.get(field))
        if observed != value:
            raise ValueError("Stage S1 formal launcher mismatch: {}".format(field))
    if tuple(getattr(args, "fcn_dims", FORMAL_PROTOCOL["fcn_dims"])) != tuple(FORMAL_PROTOCOL["fcn_dims"]):
        raise ValueError("Stage S1 formal launcher mismatch: fcn_dims")
    if int(getattr(args, "allow_protocol_deviation", 0)) != 0:
        raise ValueError("Stage S1 formal launcher does not allow protocol deviation")
    if int(getattr(args, "allow_nonformal_seed", 0)) != 0:
        raise ValueError("Stage S1 formal launcher does not allow non-formal seed")
    command_text = " ".join(str(getattr(args, field, "")) for field in vars(args))
    for token in FORBIDDEN_METHOD_TOKENS:
        if token in command_text.lower():
            raise ValueError("forbidden Stage S1 method token: {}".format(token))


def classify_task_state(tasks: Sequence[Mapping[str, object]], rerun_failed: bool = False,
                        require_formal: bool = True) -> Tuple[int, List[Dict[str, object]], List[Dict[str, object]]]:
    valid_count = 0
    pending: List[Dict[str, object]] = []
    blocked: List[Dict[str, object]] = []
    for raw_task in tasks:
        task = dict(raw_task)
        valid, reasons = validate_run(task, require_formal=require_formal)
        if valid:
            valid_count += 1
            continue
        run_dir = os.path.abspath(str(task.get("run_dir", "")))
        if not os.path.exists(run_dir):
            pending.append(task)
        elif not os.path.isdir(run_dir):
            task["validation_reasons"] = reasons + ["run_dir_not_directory"]
            blocked.append(task)
        else:
            try:
                material_attempt = _has_material_canonical_attempt(run_dir)
            except OSError as exc:
                # A canonical directory that cannot be inspected may still be
                # owned by a live/orphaned Windows writer.  Treating it as an
                # empty placeholder would permit a second launch into the same
                # formal path, so probe failures are always blocking.
                winerror = getattr(exc, "winerror", None)
                suffix = ":{}".format(winerror) if winerror is not None else ""
                task["validation_reasons"] = reasons + [
                    "canonical_attempt_probe_busy{}".format(suffix)]
                blocked.append(task)
                continue
            if not material_attempt:
                # A pre-created directory (or an empty stdout placeholder) is
                # not evidence that training started.  It is safe to reuse
                # directly; no empty failed_attempts entry is manufactured.
                pending.append(task)
                continue
            if rerun_failed:
                try:
                    archive = archive_failed_attempt(run_dir)
                except OSError as exc:
                    # A sharing violation means an active/orphaned writer still
                    # owns a file.  Fail closed and leave the canonical directory
                    # untouched for a later, safe resume.
                    winerror = getattr(exc, "winerror", None)
                    suffix = ":{}".format(winerror) if winerror is not None else ""
                    task["validation_reasons"] = reasons + [
                        "failed_attempt_archive_busy{}".format(suffix)]
                    blocked.append(task)
                else:
                    if archive:
                        task["failed_attempt_archive"] = archive
                    pending.append(task)
            else:
                task["validation_reasons"] = reasons
                blocked.append(task)
    return valid_count, pending, blocked


def _canonical_attempt_entries(canonical_run_dir: str) -> List[str]:
    """Return entries belonging to the current attempt, excluding its archive."""

    return sorted(name for name in os.listdir(canonical_run_dir)
                  if name != "failed_attempts")


def _has_material_canonical_attempt(canonical_run_dir: str) -> bool:
    """Whether a canonical directory contains evidence of a started attempt."""

    entries = _canonical_attempt_entries(canonical_run_dir)
    if not entries:
        return False
    # The launcher writes the launch command before Popen succeeds.  A log that
    # has only that header is still an unstarted canonical placeholder, not a
    # failed training attempt.  The child-started marker is written only after
    # a process handle has been successfully created.
    if entries == ["stdout.log"]:
        with open(os.path.join(canonical_run_dir, "stdout.log"),
                  encoding="utf-8", errors="replace") as handle:
            return "[stage_S1] child_started" in handle.read()
    return True


def archive_failed_attempt(canonical_run_dir: str) -> str:
    """Move one invalid canonical attempt to an append-only archive.

    The canonical path is deliberately retained for the successful retry so
    the matrix, validator, and report all discover the same formal key.  The
    move is transactional: entries first go to a staging directory, a sharing
    violation rolls all moved entries back, and only a complete staging tree
    is renamed to ``attempt_NNN``.  Thus a failed archive cannot manufacture a
    partial attempt directory or delete the canonical evidence.
    """

    base = os.path.abspath(str(canonical_run_dir))
    with _ARCHIVE_MUTEX:
        if not os.path.isdir(base):
            return ""
        entries = _canonical_attempt_entries(base)
        if not entries:
            return ""
        archive_root = os.path.join(base, "failed_attempts")
        os.makedirs(archive_root, exist_ok=True)
        prior_attempts = []
        for name in os.listdir(archive_root):
            if name.startswith("attempt_") and name[len("attempt_"):].isdigit():
                prior_attempts.append(int(name[len("attempt_"):]))
        # Append after the greatest observed attempt rather than filling a
        # historical gap, which would make chronological audit order false.
        attempt = max(prior_attempts, default=0) + 1
        while os.path.exists(os.path.join(archive_root, "attempt_{:03d}".format(attempt))):
            attempt += 1
        archive = os.path.join(archive_root, "attempt_{:03d}".format(attempt))
        staging = os.path.join(archive_root, ".staging_attempt_{:03d}_{}".format(
            attempt, uuid.uuid4().hex))
        os.makedirs(staging, exist_ok=False)
        moved: List[str] = []
        try:
            for name in entries:
                source = os.path.join(base, name)
                target = os.path.join(staging, name)
                # Staging is a child of the canonical directory and therefore
                # guaranteed to be on the same volume.  Use rename-only
                # semantics: shutil.move falls back to copy on WinError 32 and
                # can leave a partial destination file in staging.
                os.replace(source, target)
                moved.append(name)
            # The destination does not exist because the attempt number was
            # reserved under the process-wide mutex.  Same-volume replace is
            # atomic and exposes only a complete archive to readers.
            os.replace(staging, archive)
            return archive
        except BaseException:
            # Restore any entries already moved before surfacing the original
            # sharing/IO error.  If rollback itself is blocked, preserve the
            # staging tree for audit rather than deleting evidence.
            rollback_ok = True
            for name in reversed(moved):
                source = os.path.join(staging, name)
                target = os.path.join(base, name)
                try:
                    if os.path.exists(source):
                        # Never overwrite evidence newly created by an active
                        # writer while rollback is in progress.
                        if os.path.exists(target):
                            rollback_ok = False
                            continue
                        os.replace(source, target)
                except OSError:
                    rollback_ok = False
            if rollback_ok:
                try:
                    os.rmdir(staging)
                except OSError:
                    pass
            raise


def _dataset_paths(args: argparse.Namespace) -> Iterable[str]:
    ml = str(getattr(args, "movielens_path", "") or
             (getattr(args, "dataset_path", "") if getattr(args, "dataset_path", "")
              and (str(getattr(args, "dataset_path")).lower().endswith(".csv")
                   or os.path.exists(os.path.join(str(getattr(args, "dataset_path")), "ml-1m.csv")))
              else "/home/xzf/data/ml-1m"))
    kr = str(getattr(args, "kuairand_path", "") or
             (getattr(args, "dataset_path", "") if getattr(args, "dataset_path", "")
              and (str(getattr(args, "dataset_path")).lower().endswith(".csv")
                   or os.path.exists(os.path.join(str(getattr(args, "dataset_path")), "kuai_rand.csv")))
              else "/home/xzf/data/kuairand/processed"))
    yield os.path.join(ml, "ml-1m.csv") if os.path.isdir(ml) else ml
    yield os.path.join(kr, "kuai_rand.csv") if os.path.isdir(kr) else kr


def runtime_preflight(args: argparse.Namespace) -> None:
    assert_formal_launcher(args)
    child_probe = subprocess.run(
        [str(args.python), "-c",
         "import torch,sys; assert torch.cuda.is_available(); "
         "assert torch.cuda.device_count() >= 1; "
         "print(sys.executable, torch.__version__, torch.cuda.get_device_name(0))"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if child_probe.returncode != 0:
        raise RuntimeError("requested Stage S1 Python/CUDA preflight failed: {}".format(
            child_probe.stdout.strip()))
    for path in _dataset_paths(args):
        if not os.path.isfile(path):
            raise FileNotFoundError("dataset file missing: {}".format(path))
    raw_root = os.path.abspath(str(args.raw_root))
    os.makedirs(raw_root, exist_ok=True)
    probe = os.path.join(raw_root, ".write_probe_{}".format(os.getpid()))
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("S1")
    os.remove(probe)
    lock_path = os.path.join(raw_root, LOCK_FILENAME)
    if os.path.exists(lock_path):
        raise RuntimeError("another Stage S1 orchestrator/training process is active (lock exists)")


def _pid_is_deterministically_dead(pid: int) -> bool:
    """Return True only when we can prove the PID is not a running process.

    Access-denied on Windows (and PermissionError on POSIX) is treated as
    "alive".  Such a process is protected but exists, so the lock it owns may
    still be authoritative.  Any other uncertainty also returns False: a
    fail-closed orchestrator must not reclaim a lock from a process whose
    state it cannot deterministically read.
    """

    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
        except ImportError:
            return False
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        ERROR_INVALID_PARAMETER = 87
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_int)
        except OSError:
            return False
        if handle:
            try:
                kernel32.CloseHandle(handle)
            except OSError:
                pass
            # A non-zero handle means the process exists and is queryable.
            return False
        try:
            last_error = int(ctypes.GetLastError())
        except OSError:
            return False
        if last_error == ERROR_ACCESS_DENIED:
            # Process is alive but protected.  Never reclaim this lock.
            return False
        if last_error == ERROR_INVALID_PARAMETER:
            # No process matches this PID on this system.
            return True
        return False
    try:
        os.kill(pid_int, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False


def _record_lock_reclaim(raw_root: str, lock_path: str,
                         previous_payload: Mapping[str, object]) -> None:
    """Append a reclaim event to an append-only log without touching the queue.

    The queue manifest already records the resume state under the atomic stage
    lock, so the reclaim audit lives in a separate, lock-independent JSON file.
    A reclaim log failure must never block reclaim; we just lose that audit
    entry.
    """

    log_path = os.path.join(os.path.abspath(raw_root), LOCK_RECLAIM_LOG_FILENAME)
    event = {
        "stage": "S1",
        "lock_path": os.path.abspath(lock_path),
        "raw_root": os.path.abspath(raw_root),
        "previous_pid": previous_payload.get("pid"),
        "previous_token": previous_payload.get("token"),
        "reclaimed_by_pid": os.getpid(),
        "reclaimed_at": time.time(),
    }
    events: List[object] = []
    if os.path.isfile(log_path):
        try:
            with open(log_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                events = payload
        except (OSError, ValueError, TypeError):
            events = []
    events.append(event)
    temporary = log_path + ".tmp-{}-{}".format(os.getpid(), uuid.uuid4().hex)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(events, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, log_path)
    except OSError:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _try_reclaim_stale_lock(raw_root: str, path: str) -> bool:
    """Attempt to remove a lock whose previous owner is deterministically dead.

    Every condition below is required; a single missing piece returns False
    without modifying any state.  A successful reclaim deletes exactly the
    stale lock file and records a single audit event.
    """

    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    if str(payload.get("stage", "")) != "S1":
        return False
    token_value = payload.get("token")
    if not isinstance(token_value, str) or not token_value:
        return False
    pid_value = payload.get("pid")
    try:
        pid_int = int(pid_value)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if pid_int == int(os.getpid()):
        return False
    if not _pid_is_deterministically_dead(pid_int):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    _record_lock_reclaim(raw_root, path, payload)
    return True


def acquire_stage_lock(raw_root: str) -> str:
    os.makedirs(os.path.abspath(raw_root), exist_ok=True)
    path = os.path.join(os.path.abspath(raw_root), LOCK_FILENAME)
    token = uuid.uuid4().hex
    created = False
    try:
        with open(path, "x", encoding="utf-8") as handle:
            created = True
            handle.write(json.dumps({"pid": os.getpid(), "stage": "S1", "token": token}))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        # The only path that allows removing a lock we did not create is a
        # dead-PID reclaim.  Any uncertainty keeps the fail-closed RuntimeError.
        if not _try_reclaim_stale_lock(raw_root, path):
            raise RuntimeError("another Stage S1 orchestrator/training process is active")
        try:
            with open(path, "x", encoding="utf-8") as handle:
                created = True
                handle.write(json.dumps({"pid": os.getpid(), "stage": "S1", "token": token}))
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise RuntimeError("another Stage S1 orchestrator/training process is active") from exc
    except BaseException:
        # A failed write/fsync must not strand a partial lock.  Remove it only
        # when its persisted token proves this acquisition created it.
        if created:
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, Mapping) and payload.get("token") == token:
                    os.remove(path)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass
        raise
    _OWNED_STAGE_LOCKS[os.path.abspath(path)] = token
    return path


def release_stage_lock(lock_path: str) -> None:
    # Only remove the exact lock file created by this orchestrator.  The token
    # check prevents a delayed cleanup from deleting a newer owner's lock.
    path = os.path.abspath(lock_path)
    token = _OWNED_STAGE_LOCKS.get(path)
    try:
        if token is None:
            return
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping) or payload.get("token") != token:
            return
        os.remove(path)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        # A missing/corrupt/replaced lock is left in place.  Failing closed is
        # safer than removing a lock that may now belong to another owner.
        pass
    finally:
        _OWNED_STAGE_LOCKS.pop(path, None)


def audit_reference_coverage(tasks: Sequence[Mapping[str, object]],
                             require_formal: bool = True) -> Dict[str, object]:
    """Validate all three references before admitting any joint run."""

    refs = [task for task in tasks if task.get("run_type") == "single_domain"]
    missing: List[Dict[str, object]] = []
    for task in refs:
        valid, reasons = validate_run(task, require_formal=require_formal)
        if not valid:
            missing.append({"formal_run_key": task.get("formal_run_key"), "reasons": reasons})
    return {"valid": not missing, "reference_count": len(refs), "missing": missing}


def audit_block(tasks: Sequence[Mapping[str, object]], require_formal: bool = True) -> Dict[str, object]:
    """Check paired identities for one dataset/setting/backbone/seed block."""

    result: Dict[str, object] = {"valid": False, "reasons": []}
    refs = [task for task in tasks if task.get("run_type") == "single_domain"]
    joints = [task for task in tasks if task.get("run_type") == "joint"]
    reasons: List[object] = []
    if len(refs) != 3 or len(joints) != 1:
        reasons.append({"reason": "block_cardinality", "references": len(refs), "joint": len(joints)})
        result["reasons"] = reasons
        return result
    ref_manifests = []
    for task in refs + joints:
        valid, why = validate_run(task, require_formal=require_formal)
        if not valid:
            reasons.extend({"formal_run_key": task.get("formal_run_key"), "reason": item} for item in why)
        ref_manifests.append(_read_json(os.path.join(str(task["run_dir"]), "run_manifest.json")))
    if reasons:
        result["reasons"] = reasons
        return result
    common_fields = ("dataset", "setting", "backbone", "seed", "device_identity",
                     "formal_protocol", "formal_protocol_observed", "dataset_file_hash",
                     "source_code_hash", "code_hashes", "runtime", "runtime_fingerprint",
                     "metric_implementation")
    base = ref_manifests[-1]
    for manifest, task in zip(ref_manifests[:-1], refs):
        for field in common_fields:
            if json.dumps(manifest.get(field), sort_keys=True) != json.dumps(base.get(field), sort_keys=True):
                reasons.append({"formal_run_key": task.get("formal_run_key"),
                                "field": field, "reason": "pair_identity_mismatch"})
        base_pre = base.get("preprocessing_metadata", {})
        current_pre = manifest.get("preprocessing_metadata", {})
        for field in ("fit_scope", "split_rule", "label_definitions", "raw_file",
                      "base_train_row_ids_hash", "raw_file_size"):
            if json.dumps(current_pre.get(field), sort_keys=True) != json.dumps(
                    base_pre.get(field), sort_keys=True):
                reasons.append({"formal_run_key": task.get("formal_run_key"),
                                "field": "preprocessing_metadata." + field,
                 "reason": "pair_identity_mismatch"})
        pairing_valid, pairing_reasons = audit_pairing_fingerprints(
            base, manifest, int(task.get("physical_domain")))
        if not pairing_valid:
            reasons.extend({"formal_run_key": task.get("formal_run_key"),
                            "field": item.split(":", 1)[0],
                            "reason": item} for item in pairing_reasons)
    if reasons:
        result["reasons"] = reasons
        return result
    result.update({"valid": True, "reasons": [],
                   "reference_domains": sorted(int(task["physical_domain"]) for task in refs)})
    return result


def _write_queue_manifest(path: str, tasks: Sequence[Mapping[str, object]], **extra: object) -> None:
    payload = {"stage": "S1", "expected_run_count": len(tasks),
               "tasks": [dict(item) for item in tasks], **extra}
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = absolute_path + ".tmp-{}-{}".format(os.getpid(), uuid.uuid4().hex)
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A reader sees either the previous complete resume state or this
        # complete state, never a partially truncated JSON file.
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S1 192-run orchestrator")
    parser.add_argument("--raw_root", default=os.path.join(HERE, "stage_S1_raw_results"))
    parser.add_argument("--queue_manifest", default=os.path.join(HERE, "stage_S1_queue_manifest.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--movielens_path", default="")
    parser.add_argument("--kuairand_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--learning_rate_darts", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=192)
    parser.add_argument("--rerun_failed", action="store_true")
    parser.add_argument("--allow_protocol_deviation", type=int, choices=(0, 1), default=0)
    return parser


def _run_child_process(command: Sequence[str], stdout_path: str) -> int:
    """Run one child with deterministic log ownership and closed handles."""

    returncode = 1
    process: Optional[subprocess.Popen] = None
    with open(stdout_path, "a", encoding="utf-8") as stdout:
        stdout.write("\n[launch {}] {}\n".format(time.time(), " ".join(command)))
        stdout.flush()
        try:
            # The marker is written only after Popen successfully creates the
            # child.  The surrounding file context closes redirected stdout;
            # validation/finalization is called only after this function
            # returns.
            with subprocess.Popen(command, stdout=stdout,
                                  stderr=subprocess.STDOUT) as process:
                stdout.write("[stage_S1] child_started pid={}\n".format(process.pid))
                stdout.flush()
                returncode = int(process.wait())
        finally:
            # Python 3.8's Windows Popen context waits for termination but its
            # OS process HANDLE is owned by the private Handle wrapper until
            # object destruction.  Close that wrapper explicitly before the
            # caller can finalize/archive the canonical directory.
            platform_handle = getattr(process, "_handle", None)
            close_handle = getattr(platform_handle, "Close", None)
            if callable(close_handle):
                close_handle()
    return returncode


def main() -> int:
    args = build_argparser().parse_args()
    tasks = build_tasks(args.raw_root)
    validate_matrix(tasks)
    # The matrix dry-run is non-training, but it is still a formal launcher
    # artifact and may not be used to smuggle a protocol-deviation flag.
    assert_formal_launcher(args)
    if args.dry_run:
        # A dry-run does not touch task directories, but it does publish to the
        # queue manifest.  Use the same lock so it cannot overwrite a live
        # RUNNING/FAILED state from a formal orchestrator.
        dry_lock_path = acquire_stage_lock(args.raw_root)
        try:
            _write_queue_manifest(args.queue_manifest, tasks, dry_run=True,
                                  expected_run_count=len(tasks), launch_count=0,
                                  resume_state="DRY_RUN")
        finally:
            release_stage_lock(dry_lock_path)
        print("[stage_S1] dry-run matrix", len(tasks), flush=True)
        return 0
    runtime_preflight(args)
    # Classification and failed-attempt archiving must be protected by the
    # same atomic stage lock as launching.  The previous order left a window
    # where two resume processes could archive the same stdout.log.
    lock_path = acquire_stage_lock(args.raw_root)
    try:
        _write_queue_manifest(args.queue_manifest, tasks, resume_state="RUNNING",
                              lock_path=os.path.abspath(lock_path), lock_pid=os.getpid())
        valid_count, pending, blocked = classify_task_state(tasks, args.rerun_failed,
                                                            require_formal=True)
        if blocked:
            _write_queue_manifest(args.queue_manifest, tasks, blocked=blocked,
                                  valid_count=valid_count, pending_count=len(pending),
                                  resume_state="BLOCKED")
            return 1
        effective_tasks = {str(item["formal_run_key"]): dict(item) for item in tasks}
        for item in pending:
            effective_tasks[str(item["formal_run_key"])] = dict(item)
        launched = 0
        for task in pending:
            if launched >= int(args.max_runs):
                break
            if task.get("run_type") == "joint":
                references = [candidate for candidate in effective_tasks.values()
                              if candidate.get("run_type") == "single_domain"
                              and candidate.get("dataset") == task.get("dataset")
                              and candidate.get("setting") == task.get("setting")
                              and candidate.get("backbone") == task.get("backbone")
                              and candidate.get("seed") == task.get("seed")]
                coverage = audit_reference_coverage(references, require_formal=True)
                if not coverage["valid"]:
                    raise RuntimeError("joint launch blocked by missing reference coverage: {}".format(coverage))
            os.makedirs(str(task["run_dir"]), exist_ok=True)
            command = command_for(task, args)
            stdout_path = os.path.join(str(task["run_dir"]), "stdout.log")
            returncode = _run_child_process(command, stdout_path)
            launched += 1
            if returncode != 0:
                _write_queue_manifest(args.queue_manifest, tasks, valid_count=valid_count,
                                      launched=launched, failed_task=task,
                                      returncode=returncode,
                                      pending_count=max(0, len(pending) - launched),
                                      resume_state="FAILED_CHILD")
                return int(returncode or 1)
            try:
                # The child has closed the inherited stdout handle now.  Close
                # the hash inventory at this point so captured completion text
                # cannot mutate stdout.log after it has been attested.
                finalize_artifact_hashes(str(task["run_dir"]))
            except Exception as exc:
                _write_queue_manifest(args.queue_manifest, tasks, valid_count=valid_count,
                                      launched=launched, failed_task=task,
                                      reasons=["artifact_hash_finalization_failed:{}".format(
                                          type(exc).__name__)],
                                      pending_count=max(0, len(pending) - launched),
                                      resume_state="FAILED_FINALIZATION")
                return 1
            valid, reasons = validate_run(task, require_formal=True)
            if not valid:
                _write_queue_manifest(args.queue_manifest, tasks, valid_count=valid_count,
                                      launched=launched, failed_task=task, reasons=reasons,
                                      pending_count=max(0, len(pending) - launched),
                                      resume_state="FAILED_VALIDATION")
                return 1
            if task.get("run_type") == "joint":
                block = audit_block(references + [task], require_formal=True)
                if not block.get("valid"):
                    _write_queue_manifest(args.queue_manifest, tasks, valid_count=valid_count,
                                          launched=launched, failed_task=task,
                                          block_audit=block,
                                          pending_count=max(0, len(pending) - launched),
                                          resume_state="FAILED_BLOCK_AUDIT")
                    return 1
        pending_after = max(0, len(pending) - launched)
        _write_queue_manifest(
            args.queue_manifest, tasks, valid_count=valid_count,
            launched=launched, pending_after=pending_after,
            resume_state=("COMPLETE" if pending_after == 0 else "PARTIAL_MAX_RUNS"))
        return 0
    except Exception as exc:
        # Publish a terminal resume state before releasing the lock for any
        # unexpected orchestrator-side failure (including Windows sharing
        # errors not handled by the transactional archive path).  The write is
        # best-effort so the original exception remains visible.
        try:
            valid_so_far = int(locals().get("valid_count", 0) or 0)
            launched_so_far = int(locals().get("launched", 0) or 0)
            pending_so_far = locals().get("pending", [])
            _write_queue_manifest(
                args.queue_manifest, tasks, valid_count=valid_so_far,
                launched=launched_so_far,
                pending_count=max(0, len(pending_so_far) - launched_so_far),
                resume_state="FAILED_ORCHESTRATOR",
                exception_type=type(exc).__name__, exception=str(exc))
        except Exception:
            pass
        raise
    finally:
        release_stage_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_FILES", "build_tasks", "validate_matrix_tasks", "command_for", "assert_formal_launcher",
    "expected_train_domain_counts", "required_artifact_hash_names", "validate_artifact_hash_inventory",
    "finalize_artifact_hashes",
    "audit_pairing_fingerprints",
    "validate_raw_prediction_metrics", "validate_run", "classify_task_state",
    "archive_failed_attempt", "runtime_preflight", "audit_reference_coverage", "audit_block",
    "acquire_stage_lock", "release_stage_lock",
    "build_argparser", "main",
]
