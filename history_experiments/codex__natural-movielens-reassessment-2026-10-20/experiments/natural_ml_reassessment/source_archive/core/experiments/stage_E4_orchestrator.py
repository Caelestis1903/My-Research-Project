"""Resume-safe Stage E4 queue and integrity checks.

The queue is exactly 12 joint arms (Equal/Sqrt/Sample x four seeds) plus 12
matched single-domain oracle references.  This module records statuses and
mechanically checks identities; it does not interpret experimental results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E4_raw_results")
DEFAULT_MASTER = os.path.join(HERE, "stage_E4_runs.csv")
DEFAULT_ORACLE_MASTER = os.path.join(DEFAULT_RAW_ROOT, "oracle_reference_manifest.csv")
DEFAULT_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable

from stage_E4_weighting import (  # noqa: E402
    DOMAIN_NUM,
    E4_DATASET_PATH,
    E4_FORMAL_PROTOCOL,
    E4_ORACLE_PROTOCOL,
    EXPECTED_POSITIVE_COUNTS,
    EXPECTED_SEEDS,
    EXPECTED_SPLIT_COUNTS,
    TASK_NUM,
    WEIGHTING_KEYS,
    DatasetBundle,
    audit_dataset_bundle,
    canonical_weighting,
    filter_split_first_domain,
    json_hash,
    load_processed_dataset,
    normalise_dataset_path,
    validate_weighting_coefficients,
    weighting_alpha,
    weighting_coefficients_numpy,
    weighting_display,
    write_json,
)


JOINT_REQUIRED_ARTIFACTS = (
    "config.json", "run_manifest.json", "stdout.log", "metrics_epoch.csv",
    "metrics_summary.csv", "shared_gradient_step.csv", "shared_gradient_epoch.csv",
    "batch_trace.csv", "nonfinite_inventory.json", "model.pth",
)
ORACLE_REQUIRED_ARTIFACTS = (
    "config.json", "run_manifest.json", "stdout.log", "metrics_epoch.csv",
    "metrics_summary.csv", "batch_trace.csv", "nonfinite_inventory.json", "model.pth",
)

JOINT_MASTER_FIELDS = [
    "stage", "weighting", "alpha", "seed", "status", "return_code", "run_dir",
    "stdout_log", "wall_seconds", "best_epoch", "stopped_epoch", "observed_epoch_count",
    "shared_gradient_step_count", "shared_gradient_epoch_count", "block_status",
    "block_valid", "note",
]
ORACLE_MASTER_FIELDS = [
    "stage", "original_physical_domain", "seed", "status", "return_code", "run_dir",
    "stdout_log", "wall_seconds", "reason",
]


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    return [
        {
            "stage": "E4", "weighting": weighting, "weighting_display": weighting_display(weighting),
            "alpha": weighting_alpha(weighting), "seed": int(seed),
            "run_dir": os.path.join(raw_root, weighting, f"seed{seed}"),
            "stdout_log": os.path.join(raw_root, weighting, f"seed{seed}", "stdout.log"),
        }
        for weighting in WEIGHTING_KEYS for seed in EXPECTED_SEEDS
    ]


def build_oracle_tasks(oracle_root: str = os.path.join(DEFAULT_RAW_ROOT, "oracle_reference")) -> List[Dict[str, object]]:
    return [
        {
            "stage": "E4_ORACLE", "original_physical_domain": int(domain), "seed": int(seed),
            "run_dir": os.path.join(oracle_root, f"d{domain}", f"seed{seed}"),
            "stdout_log": os.path.join(oracle_root, f"d{domain}", f"seed{seed}", "stdout.log"),
        }
        for domain in range(DOMAIN_NUM) for seed in EXPECTED_SEEDS
    ]


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


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_csv(path: str, fields: Sequence[str], row: Mapping[str, object]) -> None:
    rows = _read_csv(path) if os.path.exists(path) else []
    rows.append(row)
    _write_csv(path, rows, fields)


def _int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _normalise_domain_map(value: object) -> Dict[int, int]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _normalise_domain_counts(value: object) -> Dict[int, int]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _normalise_nested_counts(value: object) -> Dict[str, Dict[int, int]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Dict[int, int]] = {}
    for split, item in value.items():
        result[str(split)] = _normalise_domain_counts(item)
    return result


def _normalise_positive_counts(value: object) -> Dict[str, Dict[int, Dict[int, int]]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Dict[int, Dict[int, int]]] = {}
    for split, domains in value.items():
        if not isinstance(domains, Mapping):
            result[str(split)] = {}
            continue
        result[str(split)] = {}
        for domain, tasks in domains.items():
            if not isinstance(tasks, Mapping):
                result[str(split)][int(domain)] = {}
                continue
            try:
                domain_key = int(domain)
            except (TypeError, ValueError):
                continue
            result[str(split)][domain_key] = {}
            for task, count in tasks.items():
                try:
                    result[str(split)][domain_key][int(task)] = int(count)
                except (TypeError, ValueError):
                    continue
    return result


def _normalise_task_counts(value: object) -> Dict[str, Dict[int, int]]:
    """Normalize oracle split -> task -> count mappings after JSON loading."""

    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Dict[int, int]] = {}
    for split, tasks in value.items():
        result[str(split)] = {}
        if not isinstance(tasks, Mapping):
            continue
        for task, count in tasks.items():
            try:
                result[str(split)][int(task)] = int(count)
            except (TypeError, ValueError):
                continue
    return result


def _contiguous_observed(manifest: Mapping[str, object], configured_epoch: int,
                         *, formal: bool = True) -> Tuple[bool, List[int]]:
    observed = manifest.get("observed_epochs")
    if not isinstance(observed, list) or not observed:
        return False, []
    parsed = [_int(value) for value in observed]
    if any(value is None for value in parsed):
        return False, []
    epochs = [int(value) for value in parsed if value is not None]
    if epochs != list(range(len(epochs))):
        return False, epochs
    if not formal:
        return True, epochs
    stopped = manifest.get("stopped_epoch")
    if stopped in (None, ""):
        return len(epochs) == int(configured_epoch), epochs
    stop = _int(stopped)
    censored = _int(manifest.get("censored_after_epoch"))
    return stop is not None and stop == epochs[-1] and censored == stop + 1, epochs


def _selected_rows(path: str, expected_cells: Sequence[int]) -> Tuple[bool, Dict[int, Dict[str, str]], List[str]]:
    rows = _read_csv(path)
    selected: Dict[int, Dict[str, str]] = {}
    reasons: List[str] = []
    for row in rows:
        if _int(row.get("selected_checkpoint"), 0) != 1:
            continue
        cell = _int(row.get("cell"), -1)
        if cell in selected:
            reasons.append("duplicate_selected_cell")
        else:
            selected[int(cell)] = row
    expected = set(int(value) for value in expected_cells)
    if set(selected) != expected:
        reasons.append("selected_checkpoint_cells")
    for row in selected.values():
        for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"):
            if _float(row.get(field)) is None:
                reasons.append("selected_nonfinite:" + field)
    return not reasons, selected, reasons


def _oracle_selected_rows(path: str) -> Tuple[bool, Dict[int, Dict[str, str]], List[str]]:
    rows = _read_csv(path)
    selected: Dict[int, Dict[str, str]] = {}
    reasons: List[str] = []
    for row in rows:
        if _int(row.get("selected_checkpoint"), 0) != 1:
            continue
        task = _int(row.get("task"), -1)
        if task in selected:
            reasons.append("duplicate_selected_task")
        else:
            selected[int(task)] = row
    if set(selected) != {0, 1}:
        reasons.append("selected_checkpoint_tasks")
    for row in selected.values():
        fields = (("best_val_auc", "best_val_logloss", "best_test_auc", "best_test_logloss")
                  if "best_test_auc" in row else
                  ("validation_auc", "validation_logloss", "test_auc", "test_logloss"))
        for field in fields:
            if _float(row.get(field)) is None:
                reasons.append("selected_nonfinite:" + field)
    return not reasons, selected, reasons


def _unexplained_nonfinite(run_dir: str) -> bool:
    inventory = _read_json(os.path.join(run_dir, "nonfinite_inventory.json"))
    if not inventory:
        return True
    return int(inventory.get("unexplained_count", 1)) != 0


def validate_oracle_artifact(
    task: Mapping[str, object], bundle: DatasetBundle, *, formal: bool = True,
) -> Tuple[bool, List[str]]:
    run_dir = str(task["run_dir"])
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_directory_missing"]
    missing = [name for name in ORACLE_REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        reasons.append("missing_files:" + ",".join(missing))
    config = _read_json(os.path.join(run_dir, "config.json"))
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if not config or not manifest:
        return False, reasons + ["config_or_manifest_missing"]
    domain = int(task["original_physical_domain"])
    seed = int(task["seed"])
    filtered, info = filter_split_first_domain(bundle, domain)
    checks = {
        "stage": manifest.get("stage") == "E4_ORACLE",
        "status": manifest.get("status") == "OK",
        "oracle": manifest.get("oracle") is True,
        "domain": _int(manifest.get("original_physical_domain"), -1) == domain,
        "seed": _int(manifest.get("seed"), -1) == seed,
        "dataset_identity": manifest.get("dataset_identity") == bundle.dataset_identity
        and config.get("dataset_identity") == bundle.dataset_identity,
        "dataset_file_identity": manifest.get("dataset_file_identity") == dict(bundle.dataset_file_identity)
        and config.get("dataset_file_identity") == dict(bundle.dataset_file_identity),
        "local_mapping": _normalise_domain_map(manifest.get("local_domain_mapping")) == {domain: 0},
        "filter_after_global_split": manifest.get("filter_after_global_split") is True
        and config.get("filter_after_global_split") is True,
        "filter_method": manifest.get("filter_method") == info["filter_method"],
        "split_rows": manifest.get("split_row_fingerprints") == info["split_row_fingerprints"]
        and config.get("split_row_fingerprints") == info["split_row_fingerprints"],
        "global_split_rows": manifest.get("global_split_row_fingerprints")
        == info["global_split_row_fingerprints"]
        and config.get("global_split_row_fingerprints") == info["global_split_row_fingerprints"],
        "split_counts": _normalise_nested_counts(manifest.get("split_counts")) == info["split_counts"]
        and _normalise_nested_counts(config.get("split_counts")) == info["split_counts"],
        "positive_counts": _normalise_task_counts(manifest.get("positive_counts")) == info["positive_counts"]
        and _normalise_task_counts(config.get("positive_counts")) == info["positive_counts"],
        "schema": _int(manifest.get("domain_num"), -1) == 1
        and _int(manifest.get("task_num"), -1) == TASK_NUM,
        "model": isinstance(manifest.get("model_identity"), Mapping)
        and _int(manifest["model_identity"].get("domain_num"), -1) == 1,
        "formal_protocol": manifest.get("formal_protocol") == E4_ORACLE_PROTOCOL
        and config.get("formal_protocol") == E4_ORACLE_PROTOCOL,
        "formal_valid": (not formal) or (manifest.get("formal_valid") is True
        and config.get("formal_valid") is True),
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    if _unexplained_nonfinite(run_dir):
        reasons.append("unexplained_nonfinite")
    selected_ok, selected, selected_reasons = _oracle_selected_rows(
        os.path.join(run_dir, "metrics_summary.csv"))
    if not selected_ok:
        reasons.extend(selected_reasons)
    observed_ok, observed = _contiguous_observed(
        manifest, int(E4_ORACLE_PROTOCOL["epoch"]), formal=formal)
    if not observed_ok:
        reasons.append("observed_epochs_invalid")
    metric_rows = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    for epoch in observed:
        rows = [row for row in metric_rows if _int(row.get("epoch"), -1) == epoch]
        if len(rows) != 2 or {_int(row.get("task"), -1) for row in rows} != {0, 1}:
            reasons.append("metrics_epoch_schema")
            break
        if any(_float(row.get(field)) is None
               for row in rows for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss")):
            reasons.append("metrics_epoch_nonfinite")
            break
    if formal and manifest.get("formal_valid") is not True:
        reasons.append("formal_invalid")
    return not reasons, reasons


def validate_joint_artifact(
    task: Mapping[str, object], bundle: DatasetBundle, *, formal: bool = True,
) -> Tuple[bool, List[str]]:
    run_dir = str(task["run_dir"])
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_directory_missing"]
    missing = [name for name in JOINT_REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        reasons.append("missing_files:" + ",".join(missing))
    config = _read_json(os.path.join(run_dir, "config.json"))
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if not config or not manifest:
        return False, reasons + ["config_or_manifest_missing"]
    weighting = canonical_weighting(task["weighting"])
    checks = {
        "stage": manifest.get("stage") == "E4",
        "status": manifest.get("status") == "OK",
        "weighting": manifest.get("weighting") == weighting
        and config.get("weighting") == weighting,
        "alpha": _float(manifest.get("alpha")) is not None
        and abs(float(manifest.get("alpha")) - weighting_alpha(weighting)) <= 1e-10,
        "seed": _int(manifest.get("seed"), -1) == int(task["seed"]),
        "dataset_identity": manifest.get("dataset_identity") == bundle.dataset_identity
        and config.get("dataset_identity") == bundle.dataset_identity,
        "dataset_file_identity": manifest.get("dataset_file_identity") == dict(bundle.dataset_file_identity)
        and config.get("dataset_file_identity") == dict(bundle.dataset_file_identity),
        "domain_mapping": manifest.get("domain_mapping") == {"tab0": 0, "tab1": 1, "tab4": 2},
        "split_rows": manifest.get("split_row_fingerprints") == bundle.metadata["split_row_fingerprints"]
        and config.get("split_row_fingerprints") == bundle.metadata["split_row_fingerprints"],
        "schema": _int(manifest.get("domain_num"), -1) == DOMAIN_NUM
        and _int(manifest.get("task_num"), -1) == TASK_NUM,
        "formal_protocol": manifest.get("formal_protocol") == E4_FORMAL_PROTOCOL
        and config.get("formal_protocol") == E4_FORMAL_PROTOCOL,
        "formal_valid": (not formal) or (manifest.get("formal_valid") is True
        and config.get("formal_valid") is True),
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    if _normalise_nested_counts(manifest.get("split_counts")) != bundle.metadata["split_counts"]:
        reasons.append("split_counts")
    if _normalise_positive_counts(manifest.get("positive_counts")) != bundle.metadata["positive_counts"]:
        reasons.append("positive_counts")
    expected_train_counts = {d: int(bundle.metadata["split_counts"]["train"][d])
                             for d in range(DOMAIN_NUM)}
    if _normalise_domain_counts(manifest.get("train_domain_counts")) != expected_train_counts:
        reasons.append("train_domain_counts")
    expected_shares = np.asarray([
        expected_train_counts[d] / float(sum(expected_train_counts.values()))
        for d in range(DOMAIN_NUM)], dtype=np.float64)
    actual_shares = np.asarray(manifest.get("domain_shares", []), dtype=np.float64)
    if actual_shares.shape != expected_shares.shape or not np.allclose(
            actual_shares, expected_shares, atol=1e-12, rtol=0):
        reasons.append("domain_shares")
    try:
        validate_weighting_coefficients(weighting, manifest.get("domain_shares", []),
                                        manifest.get("actual_coefficients", []))
    except (AssertionError, TypeError, ValueError):
        reasons.append("coefficient_identity")
    if _unexplained_nonfinite(run_dir):
        reasons.append("unexplained_nonfinite")
    selected_ok, _selected, selected_reasons = _selected_rows(
        os.path.join(run_dir, "metrics_epoch.csv"), range(DOMAIN_NUM * TASK_NUM))
    if not selected_ok:
        reasons.extend(selected_reasons)
    observed_ok, observed = _contiguous_observed(
        manifest, int(E4_FORMAL_PROTOCOL["epoch"]), formal=formal)
    if not observed_ok:
        reasons.append("observed_epochs_invalid")
    metric_rows = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    for epoch in observed:
        rows = [row for row in metric_rows if _int(row.get("epoch"), -1) == epoch]
        if len(rows) != DOMAIN_NUM * TASK_NUM or {
                _int(row.get("cell"), -1) for row in rows} != set(range(DOMAIN_NUM * TASK_NUM)):
            reasons.append("metrics_epoch_schema")
            break
        if any(_float(row.get(field)) is None
               for row in rows for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss")):
            reasons.append("metrics_epoch_nonfinite")
            break
    if formal and manifest.get("formal_valid") is not True:
        reasons.append("formal_invalid")
    return not reasons, reasons


def _trace_rows(path: str) -> List[Dict[str, str]]:
    return _read_csv(path)


def _trace_map(run_dir: str, manifest: Mapping[str, object]) -> Tuple[Dict[Tuple[str, int, int], Dict[str, str]], Dict[int, Dict[Tuple[str, int], Dict[str, str]]], List[str]]:
    rows = _trace_rows(os.path.join(run_dir, "batch_trace.csv"))
    result: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    by_epoch: Dict[int, Dict[Tuple[str, int], Dict[str, str]]] = {}
    reasons: List[str] = []
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind not in ("train", "architecture"):
            continue
        epoch, step = _int(row.get("epoch")), _int(row.get("step"))
        if epoch is None or step is None:
            reasons.append("malformed_trace_row")
            continue
        key = (kind, epoch, step)
        if key in result:
            reasons.append("duplicate_trace_row")
        result[key] = row
        by_epoch.setdefault(epoch, {})[(kind, step)] = row
    observed = manifest.get("observed_epochs")
    parsed = [_int(item) for item in observed] if isinstance(observed, list) else []
    epochs = [int(item) for item in parsed if item is not None]
    if set(by_epoch) != set(epochs):
        reasons.append("trace_epoch_support")
    for epoch in epochs:
        support = by_epoch.get(epoch, {})
        train_steps = sorted(step for kind, step in support if kind == "train")
        arch_steps = sorted(step for kind, step in support if kind == "architecture")
        if train_steps != list(range(len(train_steps))) or not train_steps:
            reasons.append(f"train_steps:{epoch}")
        if arch_steps != [-1]:
            reasons.append(f"architecture_steps:{epoch}")
    return result, by_epoch, reasons


def compare_joint_block(
    run_dirs: Mapping[str, str], bundle: DatasetBundle, *, require_formal: bool = True,
) -> Dict[str, object]:
    # A block is not comparable until every arm has emitted its complete raw
    # artifact set.  In particular, the first and second arms of a seed are a
    # normal queue state, not a failed block.  Treating an incomplete arm as
    # INVALID here would make weighting-first queue order impossible to
    # resume from an empty directory.
    missing: List[str] = []
    for key in WEIGHTING_KEYS:
        run_dir = str(run_dirs.get(key, ""))
        if not os.path.isdir(run_dir):
            missing.append(key)
            continue
        incomplete = [name for name in JOINT_REQUIRED_ARTIFACTS
                      if not os.path.exists(os.path.join(run_dir, name))]
        if incomplete:
            missing.append(f"{key}:{','.join(incomplete)}")
    if missing:
        return {
            "status": "PENDING", "valid": None, "block_validity": "PENDING",
            "missing_weightings": missing,
            "common_epochs": [], "common_epoch_count": 0, "common_trace_count": 0,
            "selected_epochs": {}, "selected_epoch_in_common_support": {},
        }
    mismatches: List[Dict[str, object]] = []
    manifests = {key: _read_json(os.path.join(str(run_dirs[key]), "run_manifest.json"))
                 for key in WEIGHTING_KEYS}
    configs = {key: _read_json(os.path.join(str(run_dirs[key]), "config.json"))
              for key in WEIGHTING_KEYS}
    if any(not manifests[key] or not configs[key] for key in WEIGHTING_KEYS):
        return {"status": "INVALID", "valid": False, "reason": "manifest_or_config_missing"}
    identity_fields = (
        "seed", "dataset", "dataset_identity", "dataset_file_identity", "device_identity",
        "domain_num", "task_num", "domain_mapping", "task_mapping", "train_domain_counts",
        "domain_shares", "split_counts", "positive_counts", "positive_rates",
        "split_row_fingerprints", "initial_parameter_state_hash", "partition", "model_identity",
        "optimizer_identity", "formal_protocol",
    )
    base = manifests[WEIGHTING_KEYS[0]]
    for key in WEIGHTING_KEYS:
        if manifests[key].get("weighting") != key:
            mismatches.append({"weighting": key, "field": "manifest_weighting"})
        try:
            validate_weighting_coefficients(key, manifests[key].get("domain_shares", []),
                                            manifests[key].get("actual_coefficients", []))
        except (AssertionError, TypeError, ValueError):
            mismatches.append({"weighting": key, "field": "coefficients"})
        if require_formal and (manifests[key].get("formal_valid") is not True
                                or configs[key].get("formal_valid") is not True):
            mismatches.append({"weighting": key, "field": "formal_valid"})
        for field in identity_fields:
            if json.dumps(manifests[key].get(field), sort_keys=True) != json.dumps(
                    base.get(field), sort_keys=True):
                mismatches.append({"weighting": key, "field": field})
        for field in ("epoch", "batch_size", "learning_rate", "learning_rate_darts",
                      "weight_decay", "expert_num", "earlystop_patience", "probe_interval",
                      "instrument", "dataset_identity", "device_identity", "split_row_fingerprints",
                      "sampling", "scheduler", "backbone", "model", "fcn_dims"):
            if json.dumps(configs[key].get(field), sort_keys=True) != json.dumps(
                    configs[WEIGHTING_KEYS[0]].get(field), sort_keys=True):
                mismatches.append({"weighting": key, "config_field": field})
    trace_maps: Dict[str, Dict[Tuple[str, int, int], Dict[str, str]]] = {}
    epoch_maps: Dict[str, Dict[int, Dict[Tuple[str, int], Dict[str, str]]]] = {}
    observed: Dict[str, List[int]] = {}
    for key in WEIGHTING_KEYS:
        trace, by_epoch, reasons = _trace_map(str(run_dirs[key]), manifests[key])
        trace_maps[key], epoch_maps[key] = trace, by_epoch
        mismatches.extend({"weighting": key, "reason": reason} for reason in reasons)
        parsed = manifests[key].get("observed_epochs")
        epochs = [_int(item) for item in parsed] if isinstance(parsed, list) else []
        observed[key] = [int(item) for item in epochs if item is not None]
    common_end = min((epochs[-1] for epochs in observed.values() if epochs), default=-1)
    max_end = max((epochs[-1] for epochs in observed.values() if epochs), default=-1)
    if common_end < 0:
        mismatches.append({"reason": "empty_common_support"})
    for key in WEIGHTING_KEYS:
        epochs = observed[key]
        if epochs and epochs[-1] < max_end:
            stop = manifests[key].get("stopped_epoch")
            censored = manifests[key].get("censored_after_epoch")
            if stop in (None, "") or _int(stop) != epochs[-1] or _int(censored) != epochs[-1] + 1:
                mismatches.append({"weighting": key, "reason": "illegal_tail_difference"})
    common_epochs = list(range(common_end + 1)) if common_end >= 0 else []
    common_count = 0
    for epoch in common_epochs:
        support_lists = {key: sorted(epoch_maps[key].get(epoch, {}).keys()) for key in WEIGHTING_KEYS}
        first_support = support_lists[WEIGHTING_KEYS[0]]
        if any(support_lists[key] != first_support for key in WEIGHTING_KEYS[1:]):
            mismatches.append({"epoch": epoch, "reason": "common_support_shape"})
            continue
        common_count += len(first_support)
        for kind, step in first_support:
            first = epoch_maps[WEIGHTING_KEYS[0]][epoch][(kind, step)]
            for key in WEIGHTING_KEYS[1:]:
                row = epoch_maps[key][epoch][(kind, step)]
                for field in ("batch_fingerprint", "batch_size", "domain_counts", "task_counts"):
                    if str(row.get(field, "")) != str(first.get(field, "")):
                        mismatches.append({"epoch": epoch, "kind": kind, "step": step,
                                           "weighting": key, "field": field})
    selected_epochs = {
        key: _int(manifests[key].get("best_epoch"))
        for key in WEIGHTING_KEYS
    }
    selected_epoch_in_common_support = {
        key: bool(selected_epochs[key] is not None
                  and selected_epochs[key] in common_epochs)
        for key in WEIGHTING_KEYS
    }
    status = "VALID" if not mismatches else "INVALID"
    return {
        "status": status, "valid": status == "VALID", "block_validity": status,
        "mismatches": mismatches,
        "common_epochs": common_epochs, "common_epoch_count": len(common_epochs),
        "common_trace_count": common_count,
        "observed_epoch_counts": {key: len(observed[key]) for key in WEIGHTING_KEYS},
        "censored": {key: manifests[key].get("censored_after_epoch") for key in WEIGHTING_KEYS},
        "selected_epochs": selected_epochs,
        "selected_epoch_in_common_support": selected_epoch_in_common_support,
    }


def _task_key(task: Mapping[str, object]) -> Tuple[str, int]:
    return str(task["weighting"]), int(task["seed"])


def _oracle_key(task: Mapping[str, object]) -> Tuple[int, int]:
    return int(task["original_physical_domain"]), int(task["seed"])


def _block_dirs(task: Mapping[str, object], raw_root: str) -> Dict[str, str]:
    return {key: os.path.join(raw_root, key, f"seed{int(task['seed'])}") for key in WEIGHTING_KEYS}


def _set_block_status(task: Mapping[str, object], raw_root: str, block: Mapping[str, object]) -> None:
    block_status = str(block.get("status", "PENDING"))
    block_valid = block.get("valid")
    common_epochs = [int(value) for value in block.get("common_epochs", [])
                     if _int(value) is not None]
    common_epoch_count = _int(block.get("common_epoch_count"), len(common_epochs))
    common_trace_count = _int(block.get("common_trace_count"), 0)
    selected_support_by_weighting = block.get("selected_epoch_in_common_support", {})

    def update_summary(run_dir: str) -> None:
        summary_path = os.path.join(run_dir, "metrics_summary.csv")
        rows = _read_csv(summary_path)
        if not rows:
            return
        weighting = str(rows[0].get("weighting", ""))
        support_map = selected_support_by_weighting
        if not isinstance(support_map, Mapping):
            support_map = {}
        selected_in_support = support_map.get(weighting)
        if selected_in_support is None:
            selected_in_support = support_map.get(
                weighting_display(weighting), False)
        manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        formal = manifest.get("formal_valid") is True
        for row in rows:
            row["block_status"] = block_status
            row["block_valid"] = "" if block_valid is None else int(bool(block_valid))
            row["block_validity"] = block_status
            row["selected_checkpoint"] = int(
                _int(row.get("selected_epoch"), -1) >= 0)
            row["selected_epoch_in_common_support"] = (
                "" if block_status == "PENDING" else int(bool(selected_in_support)))
            row["common_epoch_count"] = (common_epoch_count
                                          if block_status != "PENDING" else row.get("common_epoch_count", ""))
            row["common_trace_count"] = (common_trace_count
                                          if block_status != "PENDING" else row.get("common_trace_count", ""))
            if block_status != "PENDING":
                row["common_support"] = common_epoch_count
            if block_status == "VALID":
                row["artifact_validity"] = "FORMAL_VALID" if formal else "SMOKE_ONLY"
            elif block_status == "INVALID":
                row["artifact_validity"] = "INVALID_BLOCK"
            elif row.get("artifact_validity") == "FORMAL_VALID":
                row["artifact_validity"] = "BLOCK_PENDING"
        fields = list(rows[0].keys())
        for field in (
                "block_status", "block_valid", "block_validity", "selected_checkpoint",
                "selected_epoch_in_common_support", "common_epoch_count",
                "common_trace_count"):
            if field not in fields:
                fields.append(field)
        _write_csv(summary_path, rows, fields)

    for path in _block_dirs(task, raw_root).values():
        manifest_path = os.path.join(path, "run_manifest.json")
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        manifest["block_status"] = block_status
        manifest["block_valid"] = block_valid
        manifest["block_validity"] = block_status
        manifest["block_comparison"] = dict(block)
        write_json(manifest_path, manifest)
        update_summary(path)


def done_set(master: str, raw_root: str, bundle: DatasetBundle) -> set:
    done = set()
    for task in build_tasks(raw_root):
        valid, _ = validate_joint_artifact(task, bundle, formal=True)
        if not valid:
            continue
        block = compare_joint_block(_block_dirs(task, raw_root), bundle, require_formal=True)
        if block.get("status") == "VALID":
            done.add(_task_key(task))
    return done


def command_for_task(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E4_run.py"),
        "--weighting", str(task["weighting"]), "--seed", str(task["seed"]),
        "--run_dir", str(task["run_dir"]), "--dataset_path", str(args.dataset_path),
        "--oracle_root", str(args.oracle_root), "--device", str(args.device),
        "--batch_size", str(args.batch_size), "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate), "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--earlystop_patience", str(args.earlystop_patience),
        "--probe_interval", str(args.probe_interval), "--instrument", "1",
    ]


def command_for_oracle(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E4_oracle_run.py"),
        "--domain", str(task["original_physical_domain"]), "--seed", str(task["seed"]),
        "--run_dir", str(task["run_dir"]), "--dataset_path", str(args.dataset_path),
        "--device", str(args.device), "--batch_size", str(args.batch_size),
        "--epoch", str(args.epoch), "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--earlystop_patience", str(args.earlystop_patience),
    ]


def _runner_environment() -> Dict[str, str]:
    env = os.environ.copy()
    bundled = os.path.join(os.path.dirname(ROOT), ".venv", "Lib", "site-packages")
    if os.path.isdir(bundled):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bundled + (os.pathsep + existing if existing else "")
    return env


def audit_oracles(oracle_root: str, bundle: DatasetBundle) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    missing: List[Dict[str, object]] = []
    statuses: List[Dict[str, object]] = []
    for task in build_oracle_tasks(oracle_root):
        valid, reasons = validate_oracle_artifact(task, bundle, formal=True)
        status = "OK" if valid else "MISSING_OR_INVALID"
        row = dict(task, status=status, reason="OK" if valid else ";".join(reasons))
        statuses.append(row)
        if not valid:
            missing.append(task)
    return missing, statuses


def _finalize_oracle(task: Mapping[str, object], return_code: int, wall: float,
                     output: str, master: str, bundle: DatasetBundle) -> Dict[str, object]:
    valid, reasons = validate_oracle_artifact(task, bundle, formal=True)
    if return_code != 0:
        reasons.append("process_failed")
    status = "OK" if return_code == 0 and valid else "FAILED"
    row = {
        "stage": "E4_ORACLE", "original_physical_domain": task["original_physical_domain"],
        "seed": task["seed"], "status": status, "return_code": int(return_code),
        "run_dir": task["run_dir"], "stdout_log": task["stdout_log"],
        "wall_seconds": f"{wall:.3f}", "reason": ";".join(reasons),
    }
    _append_csv(master, ORACLE_MASTER_FIELDS, row)
    return row


def _finalize_joint(task: Mapping[str, object], return_code: int, wall: float,
                    output: str, master: str, raw_root: str, bundle: DatasetBundle) -> Dict[str, object]:
    valid, reasons = validate_joint_artifact(task, bundle, formal=True)
    status = "OK" if return_code == 0 and valid else "FAILED"
    block: Dict[str, object] = {"status": "PENDING", "valid": None,
                                "block_validity": "PENDING"}
    if status == "OK":
        block = compare_joint_block(_block_dirs(task, raw_root), bundle, require_formal=True)
        _set_block_status(task, raw_root, block)
        # PENDING is expected for the first two arms of each seed.  Keep the
        # single arm successful so the queue can continue to the remaining
        # weighting arms; only a complete triad can become INVALID_BLOCK.
        if block.get("status") == "INVALID":
            status = "INVALID_BLOCK"
    if return_code != 0:
        reasons.append("process_failed")
    if output.strip():
        reasons.append(output.strip().splitlines()[-1][:300])
    manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
    row = {
        "stage": "E4", "weighting": task["weighting"], "alpha": task["alpha"],
        "seed": task["seed"], "status": status, "return_code": int(return_code),
        "run_dir": task["run_dir"], "stdout_log": task["stdout_log"],
        "wall_seconds": f"{wall:.3f}", "best_epoch": manifest.get("best_epoch", ""),
        "stopped_epoch": manifest.get("stopped_epoch", ""),
        "observed_epoch_count": len(manifest.get("observed_epochs", [])),
        "shared_gradient_step_count": manifest.get("shared_gradient_step_count", ""),
        "shared_gradient_epoch_count": manifest.get("shared_gradient_epoch_count", ""),
        "block_status": block.get("status", "PENDING"), "block_valid": block.get("valid"),
        "note": ";".join(reasons),
    }
    _append_csv(master, JOINT_MASTER_FIELDS, row)
    return row


def _run_oracle_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace,
                      bundle: DatasetBundle) -> int:
    os.makedirs(args.oracle_root, exist_ok=True)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_oracle(task, args)
        print("[stage_E4_ORACLE][launch]", " ".join(command), flush=True)
        started = time.time()
        with open(str(task["stdout_log"]), "w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     text=True, env=_runner_environment())
        output = ""
        try:
            with open(str(task["stdout_log"]), encoding="utf-8", errors="replace") as handle:
                output = handle.read()
        except OSError:
            pass
        row = _finalize_oracle(task, process.returncode, time.time() - started,
                               output, args.oracle_master, bundle)
        print("[stage_E4_ORACLE][done]", task["original_physical_domain"], task["seed"], row["status"], flush=True)
        if row["status"] != "OK":
            return 1
    return 0


def _run_joint_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace,
                     bundle: DatasetBundle) -> int:
    os.makedirs(args.raw_root, exist_ok=True)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_task(task, args)
        print("[stage_E4][launch]", " ".join(command), flush=True)
        started = time.time()
        with open(str(task["stdout_log"]), "w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     text=True, env=_runner_environment())
        output = ""
        try:
            with open(str(task["stdout_log"]), encoding="utf-8", errors="replace") as handle:
                output = handle.read()
        except OSError:
            pass
        row = _finalize_joint(task, process.returncode, time.time() - started,
                              output, args.master, args.raw_root, bundle)
        print("[stage_E4][done]", task["weighting"], task["seed"], row["status"], flush=True)
        if row["status"] not in ("OK",):
            return 1
    return 0


def _write_queue_manifest(path: str, bundle: DatasetBundle, oracle_status: Sequence[Mapping[str, object]],
                          joint_status: Sequence[Mapping[str, object]]) -> None:
    write_json(path, {
        "stage": "E4", "scope": "queue_status_only",
        "dataset_identity": bundle.dataset_identity,
        "expected_joint_runs": len(WEIGHTING_KEYS) * len(EXPECTED_SEEDS),
        "expected_oracle_runs": DOMAIN_NUM * len(EXPECTED_SEEDS),
        "oracle_status_counts": {
            status: sum(1 for row in oracle_status if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in oracle_status})
        },
        "joint_status_counts": {
            status: sum(1 for row in joint_status if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in joint_status})
        },
        "scope_note": "No cross-run scientific analysis is produced by the orchestrator.",
    })


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E4 12+12 queue orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--oracle_root", default=os.path.join(DEFAULT_RAW_ROOT, "oracle_reference"))
    parser.add_argument("--oracle_master", default=DEFAULT_ORACLE_MASTER)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset_path", default=E4_DATASET_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--probe_interval", type=int, default=10)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--max_runs", type=int, default=10**9)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_manifest", default=os.path.join(HERE, "stage_E4_summary_manifest.json"))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.max_runs < 0:
        raise ValueError("--max_runs must be non-negative")
    bundle = load_processed_dataset(args.dataset_path, validate=True)
    audit_dataset_bundle(bundle, strict=True)
    oracle_missing, oracle_status = audit_oracles(args.oracle_root, bundle)
    _write_csv(args.oracle_master, oracle_status, ORACLE_MASTER_FIELDS)
    print("[stage_E4] oracle audit: 12 expected; missing/invalid:", len(oracle_missing), flush=True)
    if args.dry_run:
        for task in oracle_missing[:args.max_runs]:
            print("[stage_E4][dry-run][oracle]", " ".join(command_for_oracle(task, args)), flush=True)
        completed = done_set(args.master, args.raw_root, bundle)
        pending = [task for task in build_tasks(args.raw_root) if _task_key(task) not in completed]
        print("[stage_E4] fixed matrix: 3 weighting x 4 seeds = 12 joint; 3 domains x 4 seeds = 12 oracle", flush=True)
        print("[stage_E4] pending joint:", len(pending), flush=True)
        for task in pending[:max(0, args.max_runs - len(oracle_missing))]:
            print("[stage_E4][dry-run][joint]", " ".join(command_for_task(task, args)), flush=True)
        _write_queue_manifest(args.summary_manifest, bundle, oracle_status, [])
        return 0
    if oracle_missing:
        if _run_oracle_queue(oracle_missing, args, bundle) != 0:
            return 1
        oracle_missing, oracle_status = audit_oracles(args.oracle_root, bundle)
        _write_csv(args.oracle_master, oracle_status, ORACLE_MASTER_FIELDS)
        if oracle_missing:
            print("[stage_E4] oracle audit incomplete; joint queue not launched", flush=True)
            return 1
    completed = done_set(args.master, args.raw_root, bundle)
    pending = [task for task in build_tasks(args.raw_root) if _task_key(task) not in completed]
    print("[stage_E4] fixed matrix: 3 weighting x 4 seeds = 12 joint", flush=True)
    print("[stage_E4] pending joint:", len(pending), flush=True)
    result = _run_joint_queue(pending[:args.max_runs], args, bundle)
    _write_queue_manifest(args.summary_manifest, bundle, oracle_status, _read_csv(args.master))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
