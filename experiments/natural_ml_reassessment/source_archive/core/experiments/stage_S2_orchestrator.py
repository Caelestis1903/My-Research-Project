"""Stage S2 queue, immutable reuse audit gate, and two-worker scheduler.

The formal launcher is intentionally approval-gated.  A dry run and the
nonformal concurrent-resource fixture are safe during implementation/review;
the 16-run Equal queue cannot start without the caller supplying an explicit
approval flag after review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stage_S1_orchestrator import (
    REQUIRED_FILES as S1_REQUIRED_FILES,
    audit_block as _audit_s1_block,
    _file_hash as _s1_file_hash,
    validate_artifact_hash_inventory as _validate_s1_artifact_hash_inventory,
    validate_run as _validate_s1_run,
)
from stage_S2_objectives import validate_coefficient_invariants
from stage_S2_protocol import (
    BACKBONE,
    COMMON_EPOCHS,
    DATASETS,
    EPOCHS,
    FORMAL_PROTOCOL,
    SEEDS,
    STAGE,
    build_equal_matrix,
    build_reuse_matrix,
    canonical_dataset,
    domain_shares,
    equal_formal_key,
    expected_train_counts,
    settings_for_dataset,
    validate_equal_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S2_REQUIRED_FILES = tuple(name for name in S1_REQUIRED_FILES if name != "architecture_optimizer_absent.json") + (
    "objective_coefficients.json", "architecture_optimizer_state.pth")
S2_CODE_HASH_FILES = (
    "stage_S2_run.py", "stage_S2_objectives.py", "stage_S2_protocol.py",
    "stage_S2_orchestrator.py", "stage_S2_report.py",
    "stage_S1_run.py", "stage_S1_models.py", "stage_S1_data.py",
    "stage_S1_protocol.py", "stage_S1_metrics.py",
    "m3oe/MDMTRec/models/multi_domain/model.py",
)
S2_BATCH_FIELDS = (
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch", "step",
    "kind", "batch_fingerprint", "batch_size", "domain_counts", "task_counts", "train_objective",
    "raw_domain_fractions", "weighted_mass_by_domain", "total_weighted_mass",
    "objective_name", "objective_formula", "coefficient_table_hash",
)
_QUEUE_MUTEX = threading.RLock()


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_json_any(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except (OSError, csv.Error):
        return [], []


def _number(value: object) -> Optional[float]:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_source_path(filename: str) -> str:
    """Resolve S2-local and repository-root M3OE inventory entries."""

    relative = str(filename)
    if relative.replace("\\", "/").startswith("m3oe/"):
        return os.path.join(ROOT, *relative.replace("\\", "/").split("/"))
    return os.path.join(HERE, relative)


def build_tasks(raw_root: str = os.path.join(HERE, "stage_S2_raw_results")) -> List[Dict[str, object]]:
    tasks = build_equal_matrix(raw_root)
    return annotate_co_run_plan(tasks)


def _co_run_pair_id(dataset: str, setting: str, seed_a: int, seed_b: int) -> str:
    """Stable identifier for one preregistered within-setting seed pair."""

    return "S2|co_run|{}|{}|{}|{}-{}".format(
        canonical_dataset(dataset), setting, BACKBONE, int(seed_a), int(seed_b))


def build_co_run_plan(tasks: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Freeze the eight within-setting two-worker pairs for the 16-key queue."""

    validate_equal_matrix(tasks)
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    plan: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for setting in settings_for_dataset(dataset):
            block = [task for task in tasks
                     if canonical_dataset(str(task["dataset"])) == canonical_dataset(dataset)
                     and str(task["setting"]) == setting]
            if sorted(int(task["seed"]) for task in block) != sorted(int(seed) for seed in SEEDS):
                raise ValueError("co-run plan requires all four seeds in one setting: {} {}".format(
                    dataset, setting))
            for seed_a, seed_b in zip(SEEDS[::2], SEEDS[1::2]):
                key_a = equal_formal_key(dataset, setting, seed_a)
                key_b = equal_formal_key(dataset, setting, seed_b)
                if key_a not in by_key or key_b not in by_key:
                    raise ValueError("co-run plan key support mismatch: {}".format((key_a, key_b)))
                pair_id = _co_run_pair_id(dataset, setting, seed_a, seed_b)
                plan.append({
                    "co_run_pair_id": pair_id,
                    "dataset": canonical_dataset(dataset), "setting": setting,
                    "backbone": BACKBONE, "seed_pair": [int(seed_a), int(seed_b)],
                    "formal_run_keys": [key_a, key_b],
                    "workers": [
                        {"formal_run_key": key_a, "worker_id": "worker-1",
                         "planned_peer_run_key": key_b},
                        {"formal_run_key": key_b, "worker_id": "worker-2",
                         "planned_peer_run_key": key_a},
                    ],
                })
    if len(plan) != 8 or any(len(item.get("formal_run_keys", [])) != 2 for item in plan):
        raise ValueError("Stage S2 co-run plan must contain eight two-key pairs")
    return plan


def co_run_plan_digest(plan: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(json.dumps(list(plan), sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def annotate_co_run_plan(tasks: Sequence[Mapping[str, object]],
                         plan: Optional[Sequence[Mapping[str, object]]] = None
                         ) -> List[Dict[str, object]]:
    """Attach immutable pair, peer, and worker provenance to every task."""

    frozen_plan = list(plan) if plan is not None else build_co_run_plan(tasks)
    assignments: Dict[str, Dict[str, object]] = {}
    for pair in frozen_plan:
        pair_id = str(pair["co_run_pair_id"])
        for worker in pair["workers"]:
            key = str(worker["formal_run_key"])
            assignment = {
                "co_run_pair_id": pair_id,
                "planned_peer_run_key": str(worker["planned_peer_run_key"]),
                "worker_id": str(worker["worker_id"]),
            }
            if key in assignments and assignments[key] != assignment:
                raise ValueError("duplicate co-run assignment: {}".format(key))
            assignments[key] = assignment
    annotated: List[Dict[str, object]] = []
    for task in tasks:
        key = str(task["formal_run_key"])
        if key not in assignments:
            raise ValueError("task missing from frozen co-run plan: {}".format(key))
        value = dict(task)
        for field, expected in assignments[key].items():
            if field in task and str(task[field]) != str(expected):
                raise ValueError("task co-run provenance mismatch: {} {}".format(key, field))
            value[field] = expected
        annotated.append(value)
    if len(annotated) != 16 or len(assignments) != 16:
        raise ValueError("co-run plan must annotate exactly 16 tasks")
    return annotated


def validate_matrix_tasks(tasks: Sequence[Mapping[str, object]]) -> Tuple[bool, List[str]]:
    try:
        validate_equal_matrix(tasks)
        return True, []
    except ValueError as exc:
        return False, [str(exc)]


def command_for(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    if str(task.get("backbone")) != BACKBONE or str(task.get("method")) != "equal":
        raise ValueError("Stage S2 command builder accepts Equal M3OE tasks only")
    for field in ("co_run_pair_id", "planned_peer_run_key", "worker_id"):
        if not str(task.get(field, "")):
            raise ValueError("formal S2 task is missing frozen co-run field: {}".format(field))
    python_executable = str(getattr(args, "python", sys.executable))
    command = [python_executable, os.path.join(HERE, "stage_S2_run.py"),
               "--run_type", "joint", "--dataset", str(task["dataset"]),
               "--setting", str(task["setting"]), "--backbone", BACKBONE,
               "--seed", str(int(task["seed"])), "--run_dir", str(task["run_dir"]),
               "--dataset_path", str(getattr(args, "dataset_path", "")),
               "--batch_size", str(int(FORMAL_PROTOCOL["batch_size"])),
               "--epoch", str(EPOCHS), "--learning_rate", str(FORMAL_PROTOCOL["learning_rate"]),
               "--learning_rate_darts", str(FORMAL_PROTOCOL["learning_rate_darts"]),
               "--weight_decay", str(FORMAL_PROTOCOL["weight_decay"]),
               "--expert_num", str(FORMAL_PROTOCOL["expert_num"]),
               "--shadow_patience", str(FORMAL_PROTOCOL["shadow_patience"]),
               "--num_workers", "0", "--device", "cuda:0",
               "--co_run_pair_id", str(task["co_run_pair_id"]),
               "--planned_peer_run_key", str(task["planned_peer_run_key"]),
               "--worker_id", str(task["worker_id"])]
    command.extend(["--fcn_dims"] + [str(value) for value in FORMAL_PROTOCOL["fcn_dims"]])
    return command


def lock_path_for_task(task: Mapping[str, object], args: argparse.Namespace) -> str:
    """Return a deterministic scheduler lock outside the child run directory.

    ``S1Runner`` deliberately rejects any unexpected entry in a canonical
    directory before it starts.  Scheduler coordination therefore lives in a
    sibling lock namespace under ``raw_root`` rather than in the directory
    validated by the child.
    """

    raw_root = os.path.abspath(str(getattr(args, "raw_root", os.path.join(HERE, "stage_S2_raw_results"))))
    formal_key = str(task.get("formal_run_key", ""))
    digest = hashlib.sha256(formal_key.encode("utf-8")).hexdigest()
    return os.path.join(raw_root, ".stage_S2_locks", digest + ".lock")


def _expected_train_counts(task: Mapping[str, object]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in expected_train_counts(
        str(task["dataset"]), str(task["setting"])).items()}


def _required_artifact_hash_names(manifest: Mapping[str, object]) -> set[str]:
    declared = manifest.get("artifact_hashes")
    if not isinstance(declared, Mapping):
        return set()
    required = set(S2_REQUIRED_FILES) - {"run_manifest.json"}
    return required


def _validate_hash_inventory(run_dir: str, manifest: Mapping[str, object]) -> List[str]:
    reasons: List[str] = []
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        return ["artifact_hash_inventory_missing"]
    required = _required_artifact_hash_names(manifest)
    if not required.issubset({str(key) for key in hashes}):
        reasons.append("artifact_hash_inventory_incomplete")
    if "run_manifest.json" in {str(key) for key in hashes}:
        reasons.append("artifact_hash_self_reference")
    for name, expected in hashes.items():
        path = os.path.join(run_dir, str(name))
        if not isinstance(expected, str) or len(expected) != 64:
            reasons.append("artifact_hash_invalid:" + str(name))
        elif not os.path.isfile(path):
            reasons.append("artifact_missing:" + str(name))
        elif _file_hash(path) != expected:
            reasons.append("artifact_hash_mismatch:" + str(name))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or "run_manifest.json" not in artifacts \
            or set(str(name) for name in artifacts if name != "run_manifest.json") != set(str(name) for name in hashes):
        reasons.append("artifact_declaration_mismatch")
    return reasons


def validate_batch_trace_contract(batch_fields: Sequence[str],
                                  batch_rows: Sequence[Mapping[str, object]],
                                  manifest: Mapping[str, object],
                                  coefficients: Mapping[str, object],
                                  expected_counts: Mapping[str, int],
                                  *, epochs: int = EPOCHS,
                                  batch_size: int = int(FORMAL_PROTOCOL["batch_size"])) -> List[str]:
    """Recompute the main/architecture batch audit from emitted CSV rows."""

    reasons: List[str] = []
    if list(batch_fields) != list(S2_BATCH_FIELDS):
        reasons.append("batch_trace_header_mismatch")
    counts_total = sum(int(value) for value in expected_counts.values())
    batches = (counts_total + int(batch_size) - 1) // int(batch_size)
    if len(batch_rows) != int(epochs) * (batches + 1):
        reasons.append("batch_trace_row_count:{}".format(len(batch_rows)))
    for row in batch_rows:
        if row.get("stage") != STAGE or row.get("objective_name") != "equal_domain_task_mean_bce":
            reasons.append("batch_objective_identity_mismatch")
            break
        if row.get("coefficient_table_hash") != coefficients.get("coefficient_table_hash"):
            reasons.append("batch_coefficient_hash_mismatch")
            break
        if any(_number(row.get(field)) is None for field in ("batch_size", "total_weighted_mass", "train_objective")):
            reasons.append("batch_numeric_field_missing")
            break
    train_rows = [row for row in batch_rows if row.get("kind") == "train"]
    architecture_rows = [row for row in batch_rows if row.get("kind") == "architecture"]
    if len(train_rows) != int(epochs) * batches:
        reasons.append("train_batch_row_count_mismatch")
    if len(architecture_rows) != int(epochs):
        reasons.append("architecture_batch_row_count_mismatch")
    for epoch in range(int(epochs)):
        epoch_rows = [row for row in train_rows if _integer(row.get("epoch")) == epoch]
        observed_steps = sorted(step for step in (_integer(row.get("step")) for row in epoch_rows)
                                if step is not None)
        if observed_steps != list(range(batches)):
            reasons.append("train_step_support_mismatch:{}".format(epoch))
        if sum(_integer(row.get("batch_size")) or 0 for row in epoch_rows) != counts_total:
            reasons.append("train_exposure_mismatch:{}".format(epoch))
    traced_rows = train_rows + architecture_rows
    if any(not row.get("batch_fingerprint") for row in traced_rows):
        reasons.append("batch_fingerprint_missing")
    ordered_train = sorted(train_rows, key=lambda row: (_integer(row.get("epoch")) or -1,
                                                         _integer(row.get("step")) or -1))
    ordered_architecture = sorted(architecture_rows, key=lambda row: (_integer(row.get("epoch")) or -1,
                                                                        _integer(row.get("step")) or -1))
    train_hash = hashlib.sha256(json.dumps(
        [row.get("batch_fingerprint") for row in ordered_train]).encode()).hexdigest()
    architecture_hash = hashlib.sha256(json.dumps(
        [row.get("batch_fingerprint") for row in ordered_architecture]).encode()).hexdigest()
    if manifest.get("batch_order_hash") != train_hash:
        reasons.append("batch_order_hash_recomputed_mismatch")
    if manifest.get("architecture_batch_order_hash") != architecture_hash:
        reasons.append("architecture_batch_order_hash_recomputed_mismatch")
    if manifest.get("train_batches_total") != len(train_rows) \
            or manifest.get("batch_fingerprint_count") != len(train_rows):
        reasons.append("train_batch_counter_mismatch")
    if manifest.get("architecture_batch_fingerprint_count") != len(architecture_rows):
        reasons.append("architecture_batch_counter_mismatch")
    if manifest.get("optimizer_steps") != int(epochs) * batches \
            or manifest.get("examples_processed") != int(epochs) * counts_total:
        reasons.append("optimizer_or_example_counter_mismatch")
    architecture_examples = sum(_integer(row.get("batch_size")) or 0 for row in architecture_rows)
    if manifest.get("architecture_optimizer_steps") != int(epochs) \
            or manifest.get("architecture_examples_processed") != architecture_examples:
        reasons.append("architecture_optimizer_or_example_counter_mismatch")
    return sorted(set(reasons))


def validate_co_run_provenance(task: Mapping[str, object],
                               manifest: Mapping[str, object],
                               config: Mapping[str, object]) -> List[str]:
    """Cross-check frozen pair/peer/worker identity in child artifacts."""

    reasons: List[str] = []
    for field in ("co_run_pair_id", "planned_peer_run_key", "worker_id"):
        expected = str(task.get(field, ""))
        observed_manifest = str(manifest.get(field, ""))
        observed_config = str(config.get(field, ""))
        if not expected or observed_manifest != expected or observed_config != expected:
            reasons.append("co_run_provenance_mismatch:" + field)
    return reasons


def finalize_artifact_hashes(run_dir: str) -> Dict[str, object]:
    """Close the child-owned stdout hash after the subprocess exits."""

    run_dir = os.path.abspath(run_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "OK":
        raise RuntimeError("cannot finalize a non-OK S2 manifest")
    declared = manifest.get("artifact_hashes")
    if not isinstance(declared, Mapping) or not declared:
        raise RuntimeError("S2 artifact hash inventory missing")
    finalized: Dict[str, str] = {}
    for name in declared:
        path = os.path.join(run_dir, str(name))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        finalized[str(name)] = _file_hash(path)
    stdout_path = os.path.join(run_dir, "stdout.log")
    if not os.path.isfile(stdout_path):
        raise FileNotFoundError(stdout_path)
    manifest["artifact_hashes"] = finalized
    manifest["artifact_hash_finalization"] = {
        "method": "stage_S2_orchestrator_after_child_exit",
        "stdout_log_hash": finalized.get("stdout.log", ""),
        "stdout_log_size": os.path.getsize(stdout_path),
    }
    temporary = manifest_path + ".finalizing-{}".format(uuid.uuid4().hex)
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return manifest


def validate_equal_run(task: Mapping[str, object], require_formal: bool = True) -> Tuple[bool, List[str]]:
    """Fail-closed structural validation of a fresh Equal artifact."""

    run_dir = os.path.abspath(str(task.get("run_dir", "")))
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_dir_missing"]
    missing = [name for name in S2_REQUIRED_FILES if not os.path.isfile(os.path.join(run_dir, name))]
    if missing:
        return False, ["missing_files:" + ",".join(missing)]
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    config = _read_json(os.path.join(run_dir, "config.json"))
    coefficients = _read_json(os.path.join(run_dir, "objective_coefficients.json"))
    if manifest.get("status") != "OK":
        reasons.append("manifest_not_OK")
    if manifest.get("stage") != STAGE or config.get("stage") != STAGE:
        reasons.append("stage_identity_mismatch")
    if manifest.get("method") != "equal" or config.get("method") != "equal":
        reasons.append("method_identity_mismatch")
    expected_key = str(task.get("formal_run_key", ""))
    if manifest.get("formal_run_key") != expected_key:
        reasons.append("formal_key_mismatch")
    for field in ("dataset", "setting", "backbone", "seed", "run_type"):
        expected = str(task.get(field))
        observed = str(manifest.get(field))
        if field == "dataset":
            try:
                if canonical_dataset(observed) != canonical_dataset(expected):
                    reasons.append("identity_mismatch:dataset")
            except ValueError:
                reasons.append("identity_mismatch:dataset")
        elif observed != expected:
            reasons.append("identity_mismatch:" + field)
    reasons.extend(validate_co_run_provenance(task, manifest, config))
    if manifest.get("dataset_key") != canonical_dataset(str(task.get("dataset"))):
        reasons.append("dataset_key_mismatch")
    if manifest.get("domain_num") != 3 or manifest.get("task_num") != 2 \
            or manifest.get("local_to_physical") != [0, 1, 2]:
        reasons.append("joint_domain_contract_mismatch")
    if manifest.get("test_evaluated_during_training") is not False \
            or config.get("test_evaluated_during_training") is not False:
        reasons.append("test_before_selection")
    if manifest.get("checkpoint_selection_frozen_before_test") is not True \
            or config.get("checkpoint_selection_frozen_before_test") is not True:
        reasons.append("selection_freeze_evidence_missing")
    if manifest.get("test_metric_in_epoch_history") is not False:
        reasons.append("test_metric_in_epoch_history")
    nonfinite = _read_json_any(os.path.join(run_dir, "nonfinite_inventory.json"))
    if not isinstance(nonfinite, list) or nonfinite:
        reasons.append("nonfinite_inventory_invalid_or_nonempty")
    if require_formal:
        if manifest.get("formal_valid") is not True or config.get("formal_valid") is not True:
            reasons.append("formal_protocol_invalid")
        if manifest.get("formal_protocol") != dict(FORMAL_PROTOCOL) \
                or config.get("formal_protocol") != dict(FORMAL_PROTOCOL):
            reasons.append("formal_protocol_mismatch")
        if manifest.get("formal_protocol_observed") != dict(FORMAL_PROTOCOL) \
                or config.get("formal_protocol_observed") != dict(FORMAL_PROTOCOL):
            reasons.append("formal_protocol_observed_mismatch")
        if manifest.get("formal_protocol_mismatches") != {} \
                or config.get("formal_protocol_mismatches") != {}:
            reasons.append("formal_protocol_mismatch_inventory_nonempty")
        finalization = manifest.get("artifact_hash_finalization")
        if not isinstance(finalization, Mapping) \
                or finalization.get("method") != "stage_S2_orchestrator_after_child_exit":
            reasons.append("artifact_hash_finalization_missing")
        elif finalization.get("stdout_log_hash") != manifest.get("artifact_hashes", {}).get("stdout.log"):
            reasons.append("stdout_final_hash_cross_reference_mismatch")
        elif finalization.get("stdout_log_size") != os.path.getsize(os.path.join(run_dir, "stdout.log")):
            reasons.append("stdout_final_size_mismatch")
    try:
        validate_coefficient_invariants(coefficients)
    except Exception as exc:
        reasons.append("objective_coefficients_invalid:{}".format(type(exc).__name__))
    if manifest.get("coefficient_table_hash") != coefficients.get("coefficient_table_hash") \
            or config.get("coefficient_table_hash") != coefficients.get("coefficient_table_hash"):
        reasons.append("coefficient_table_hash_mismatch")
    if manifest.get("objective") != "equal_domain_task_mean_bce" \
            or config.get("objective") != "equal_domain_task_mean_bce":
        reasons.append("objective_identity_mismatch")
    if manifest.get("no_double_weighting") is not True:
        reasons.append("no_double_weighting_guard_missing")

    expected_counts = _expected_train_counts(task)
    observed_counts = manifest.get("train_domain_counts")
    if not isinstance(observed_counts, Mapping) \
            or {str(k): int(v) for k, v in observed_counts.items()} != expected_counts:
        reasons.append("train_domain_counts_exact_mismatch")
    observed_shares = manifest.get("domain_shares")
    expected_shares = {str(k): value for k, value in domain_shares(
        str(task["dataset"]), str(task["setting"])).items()}
    if not isinstance(observed_shares, Mapping) or set(str(k) for k in observed_shares) != set(expected_shares):
        reasons.append("domain_shares_support_mismatch")
    else:
        for key, expected in expected_shares.items():
            observed = _number(observed_shares.get(key))
            if observed is None or abs(observed - expected) > 1e-12:
                reasons.append("domain_share_exact_mismatch:" + key)

    metric_fields, metric_rows = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    if metric_fields != [
        "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch", "domain", "physical_domain", "size_role", "task", "cell",
        "train_sample_count", "validation_sample_count", "train_positive_rate", "validation_positive_rate", "train_cell_logloss", "validation_auc", "validation_logloss", "validation_macro_auc", "validation_macro_logloss", "target_weighted_validation_auc", "target_weighted_validation_logloss", "worst_domain_validation_auc", "worst_cell_validation_auc", "train_objective", "main_optimizer_steps", "architecture_optimizer_steps", "learning_rate", "architecture_learning_rate", "selected_checkpoint", "fixed_final", "common_epoch"]:
        reasons.append("metrics_epoch_header_mismatch")
    if len(metric_rows) != EPOCHS * 6:
        reasons.append("metrics_epoch_row_count:{}".format(len(metric_rows)))
    required_metric_numeric = ("epoch", "domain", "physical_domain", "task", "cell",
                               "train_sample_count", "validation_sample_count", "train_positive_rate",
                               "validation_positive_rate", "train_cell_logloss", "validation_auc",
                               "validation_logloss", "validation_macro_auc", "validation_macro_logloss",
                               "target_weighted_validation_auc", "target_weighted_validation_logloss",
                               "worst_domain_validation_auc", "worst_cell_validation_auc", "train_objective",
                               "main_optimizer_steps", "architecture_optimizer_steps", "learning_rate",
                               "architecture_learning_rate", "selected_checkpoint", "fixed_final", "common_epoch")
    for row in metric_rows:
        if any(_number(row.get(field)) is None for field in required_metric_numeric):
            reasons.append("metrics_epoch_nonfinite_or_missing")
            break
        if row.get("stage") != STAGE:
            reasons.append("metrics_epoch_stage_mismatch")
            break
    expected_metric_support = {(epoch, task_id, domain) for epoch in range(EPOCHS)
                               for task_id in (0, 1) for domain in (0, 1, 2)}
    observed_support = {(_integer(row.get("epoch")), _integer(row.get("task")), _integer(row.get("domain")))
                        for row in metric_rows}
    if observed_support != expected_metric_support:
        reasons.append("metrics_epoch_cell_support_mismatch")
    by_epoch: Dict[int, float] = {}
    for row in metric_rows:
        epoch = _integer(row.get("epoch")); score = _number(row.get("validation_macro_auc"))
        if epoch is not None and score is not None:
            if epoch in by_epoch and abs(by_epoch[epoch] - score) > 1e-12:
                reasons.append("validation_macro_auc_inconsistent_within_epoch")
            by_epoch.setdefault(epoch, score)
    if len(by_epoch) == EPOCHS:
        selected = min(by_epoch, key=lambda epoch: (-by_epoch[epoch], epoch))
        if _integer(manifest.get("selected_epoch")) != selected:
            reasons.append("selected_epoch_recomputed_mismatch")
    else:
        reasons.append("validation_epoch_summary_missing")
    if manifest.get("fixed_final_epoch") != 49 or manifest.get("common_epochs") != list(COMMON_EPOCHS):
        reasons.append("checkpoint_epoch_contract_invalid")

    test_fields, test_rows = _read_csv(os.path.join(run_dir, "metrics_test.csv"))
    if test_fields != [
        "stage", "run_type", "dataset", "setting", "backbone", "seed", "checkpoint", "stage_epoch", "training_stage", "domain", "physical_domain", "size_role", "task", "cell", "train_sample_count", "validation_sample_count", "test_sample_count", "test_positive_rate", "auc", "logloss", "validation_auc_at_stage", "validation_logloss_at_stage", "validation_macro_auc_at_stage", "validation_macro_logloss_at_stage", "selected_epoch", "selected_validation_macro_auc", "parameter_count", "training_stage_identifier", "domain_share", "physical_domain_identity", "task_identity", "train_positive_count", "validation_positive_count", "test_positive_count", "train_positive_rate", "validation_positive_rate", "train_label_entropy", "validation_label_entropy", "test_label_entropy", "reference_scope"]:
        reasons.append("metrics_test_header_mismatch")
    expected_checkpoints = {"selected", "final"} | {"common_epoch_{}".format(epoch) for epoch in COMMON_EPOCHS}
    if len(test_rows) != 6 * len(expected_checkpoints):
        reasons.append("metrics_test_row_count:{}".format(len(test_rows)))
    if {row.get("checkpoint") for row in test_rows} != expected_checkpoints:
        reasons.append("metrics_test_checkpoint_support_mismatch")
    test_keys = [(row.get("checkpoint"), row.get("task"), row.get("domain")) for row in test_rows]
    if len(test_keys) != len(set(test_keys)):
        reasons.append("metrics_test_duplicate_cell")
    summary_fields, summary_rows = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))
    if summary_fields != test_fields or len(summary_rows) != 6 \
            or any(row.get("checkpoint") != "selected" for row in summary_rows):
        reasons.append("metrics_summary_contract_mismatch")
    # final and common_epoch_49 are separate labels but must be numerically
    # identical views of the same frozen epoch/state.
    compare_fields = ("stage_epoch", "auc", "logloss", "validation_auc_at_stage",
                      "validation_logloss_at_stage", "validation_macro_auc_at_stage",
                      "validation_macro_logloss_at_stage")
    by_key = {(row.get("checkpoint"), row.get("task"), row.get("domain")): row for row in test_rows}
    for task_id in ("0", "1"):
        for domain in ("0", "1", "2"):
            left = by_key.get(("final", task_id, domain)); right = by_key.get(("common_epoch_49", task_id, domain))
            if left is None or right is None or any(str(left.get(field)) != str(right.get(field)) for field in compare_fields):
                reasons.append("final_common_epoch_49_mismatch")
                break

    batch_fields, batch_rows = _read_csv(os.path.join(run_dir, "batch_trace.csv"))
    reasons.extend(validate_batch_trace_contract(
        batch_fields, batch_rows, manifest, coefficients, expected_counts))
    reasons.extend(_validate_hash_inventory(run_dir, manifest))
    # Code inventory is checked against current source before formal use.  A
    # missing S2 report during a tiny unit smoke is tolerated only when the
    # caller explicitly asks for nonformal validation.
    code_hashes = manifest.get("code_hashes")
    if require_formal:
        if not isinstance(code_hashes, Mapping):
            reasons.append("code_hash_inventory_missing")
        else:
            for filename in S2_CODE_HASH_FILES:
                path = _code_source_path(filename)
                if not os.path.isfile(path):
                    reasons.append("code_hash_source_missing:" + filename)
                elif str(code_hashes.get(filename, "")) != _file_hash(path):
                    reasons.append("code_hash_mismatch:" + filename)
    return not reasons, reasons


def validate_s1_reuse_task(task: Mapping[str, object], require_formal: bool = True) -> Tuple[bool, List[str]]:
    """Delegate validation to the frozen S1 validator without writing to S1."""

    source_task = {
        "run_type": task["run_type"], "dataset": task["dataset"], "setting": task["setting"],
        "backbone": BACKBONE, "seed": task["seed"], "physical_domain": task.get("physical_domain"),
        "formal_run_key": task["source_formal_run_key"], "run_dir": task["source_run_dir"],
    }
    return _validate_s1_run(source_task, require_formal=require_formal)


def compare_equal_sample_manifests(equal_manifest: Mapping[str, object],
                                   sample_manifest: Mapping[str, object]) -> List[str]:
    """Check the held-constant fields of an Equal/Sample joint pair."""

    reasons: List[str] = []
    fields = (
        "dataset", "dataset_key", "setting", "backbone", "seed", "domain_num", "task_num",
        "local_to_physical", "physical_domain_identity", "features", "split_row_fingerprints",
        "split_row_id_hashes", "base_split_fingerprints", "base_split_row_id_hashes",
        "physical_view_fingerprints", "train_domain_counts", "domain_shares", "cell_stats",
        "preprocessing_metadata", "dataset_file_hash", "runtime_fingerprint",
        "initial_parameter_state_hash", "model_parameter_manifest", "forward_flops_estimate",
        "optimizer", "architecture_optimizer", "batches_per_epoch", "examples_processed",
        "architecture_examples_processed", "optimizer_steps", "architecture_optimizer_steps",
        "batch_order_hash", "architecture_batch_order_hash", "batch_fingerprint_count",
        "architecture_batch_fingerprint_count", "fixed_final_epoch", "common_epochs",
        "metric_implementation", "checkpoint_selection_frozen_before_test",
        "test_evaluated_during_training", "test_metric_in_epoch_history",
    )
    for field in fields:
        if json.dumps(equal_manifest.get(field), sort_keys=True, default=str) != \
                json.dumps(sample_manifest.get(field), sort_keys=True, default=str):
            reasons.append("pairing_field_mismatch:" + field)
    # Protocol identity is held constant except for the explicitly changed
    # stage/objective fields; compare every other protocol field exactly.
    equal_protocol = equal_manifest.get("formal_protocol", {})
    sample_protocol = sample_manifest.get("formal_protocol", {})
    if isinstance(equal_protocol, Mapping) and isinstance(sample_protocol, Mapping):
        for field in set(equal_protocol) & set(sample_protocol):
            if field in ("stage", "objective", "objective_formula", "objective_weight_formula",
                         "forbidden_methods"):
                continue
            if json.dumps(equal_protocol.get(field), sort_keys=True, default=str) != \
                    json.dumps(sample_protocol.get(field), sort_keys=True, default=str):
                reasons.append("formal_protocol_pairing_mismatch:" + str(field))
    else:
        reasons.append("formal_protocol_pairing_missing")
    return sorted(set(reasons))


def validate_s2_pairings(equal_tasks: Sequence[Mapping[str, object]],
                         reuse_tasks: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Validate Equal/Sample and frozen joint/reference block pairing."""

    sample_by_block = {(str(task["dataset"]), str(task["setting"]), int(task["seed"])): task
                       for task in reuse_tasks if task.get("run_type") == "joint"}
    equal_results: List[Dict[str, object]] = []
    for equal_task in equal_tasks:
        block_key = (str(equal_task["dataset"]), str(equal_task["setting"]), int(equal_task["seed"]))
        sample_task = sample_by_block.get(block_key)
        reasons: List[str] = []
        if sample_task is None:
            reasons.append("sample_joint_key_missing")
        else:
            equal_manifest = _read_json(os.path.join(str(equal_task["run_dir"]), "run_manifest.json"))
            sample_manifest = _read_json(os.path.join(str(sample_task["source_run_dir"]), "run_manifest.json"))
            if equal_manifest:
                reasons.extend(compare_equal_sample_manifests(equal_manifest, sample_manifest))
            else:
                reasons.append("equal_manifest_missing")
        equal_results.append({"block": block_key, "valid": not reasons, "reasons": sorted(set(reasons))})

    # Re-run the frozen S1 block audit over the 16 blocks. This does not write
    # S1 artifacts and retains the S1 validator's physical-view/reference map.
    block_groups: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = {}
    for task in reuse_tasks:
        block_groups.setdefault((str(task["dataset"]), str(task["setting"]), int(task["seed"])), []).append({
            "run_type": task["run_type"], "dataset": task["dataset"], "setting": task["setting"],
            "backbone": BACKBONE, "seed": task["seed"], "physical_domain": task.get("physical_domain"),
            "formal_run_key": task["source_formal_run_key"], "run_dir": task["source_run_dir"],
        })
    s1_blocks: List[Dict[str, object]] = []
    for block, values in sorted(block_groups.items()):
        audited = _audit_s1_block(values, require_formal=True)
        s1_blocks.append({"block": block, "valid": bool(audited.get("valid")),
                          "reasons": audited.get("reasons", [])})
    return {"valid": all(item["valid"] for item in equal_results + s1_blocks),
            "equal_sample": equal_results, "s1_blocks": s1_blocks}


def audit_reuse_tasks(tasks: Optional[Sequence[Mapping[str, object]]] = None,
                      *, require_formal: bool = True) -> Dict[str, object]:
    """Validate the immutable 64-key S1 subset and return hash-addressed evidence."""

    reuse_tasks = list(tasks) if tasks is not None else build_reuse_matrix(
        os.path.join(HERE, "stage_S1_raw_results"))
    if len(reuse_tasks) != 64:
        raise ValueError("S2 reuse audit requires exactly 64 tasks")
    block_audit = validate_s2_pairings([], reuse_tasks)
    s1_block_reasons = {tuple(item["block"]): item.get("reasons", [])
                        for item in block_audit.get("s1_blocks", [])}
    rows: List[Dict[str, object]] = []
    for task in reuse_tasks:
        valid, reasons = validate_s1_reuse_task(task, require_formal=require_formal)
        manifest = _read_json(os.path.join(str(task["source_run_dir"]), "run_manifest.json"))
        artifacts = manifest.get("artifact_hashes", {})
        row: Dict[str, object] = {
            "stage": STAGE, "method": task["method"], "source_stage": "S1",
            "provenance_type": "reused", "reuse_decision": "REUSE_OK" if valid else "REUSE_REJECT",
            "validation_reasons": ";".join(str(item) for item in reasons),
            "source_formal_run_key": task["source_formal_run_key"],
            "source_run_dir": os.path.abspath(str(task["source_run_dir"])),
            "dataset": task["dataset"], "setting": task["setting"],
            "backbone": BACKBONE, "seed": int(task["seed"]),
            "run_type": task["run_type"], "physical_domain": task.get("physical_domain", ""),
            "status": manifest.get("status", ""),
            "dataset_file_hash": manifest.get("dataset_file_hash", ""),
            "split_row_fingerprints": json.dumps(manifest.get("split_row_fingerprints", {}), sort_keys=True),
            "split_row_id_hashes": json.dumps(manifest.get("split_row_id_hashes", {}), sort_keys=True),
            "physical_view_fingerprints": json.dumps(manifest.get("physical_view_fingerprints", {}), sort_keys=True),
            "initial_parameter_state_hash": manifest.get("initial_parameter_state_hash", ""),
            "batch_order_hash": manifest.get("batch_order_hash", ""),
            "architecture_batch_order_hash": manifest.get("architecture_batch_order_hash", ""),
            "optimizer_steps": manifest.get("optimizer_steps", ""),
            "architecture_optimizer_steps": manifest.get("architecture_optimizer_steps", ""),
            "examples_processed": manifest.get("examples_processed", ""),
            "architecture_examples_processed": manifest.get("architecture_examples_processed", ""),
            "selected_epoch": manifest.get("selected_epoch", ""),
            "checkpoint_hashes": json.dumps(manifest.get("checkpoints", {}), sort_keys=True),
            "artifact_hashes": json.dumps(artifacts, sort_keys=True),
            "code_hashes": json.dumps(manifest.get("code_hashes", {}), sort_keys=True),
            "runtime_fingerprint": manifest.get("runtime_fingerprint", ""),
            "initialization_audit_hash": _file_hash(os.path.join(str(task["source_run_dir"]), "initialization_audit.json"))
            if os.path.isfile(os.path.join(str(task["source_run_dir"]), "initialization_audit.json")) else "",
            "train_domain_counts": json.dumps(manifest.get("train_domain_counts", {}), sort_keys=True),
            "domain_shares": json.dumps(manifest.get("domain_shares", {}), sort_keys=True),
            "formal_protocol": json.dumps(manifest.get("formal_protocol", {}), sort_keys=True),
            "objective": manifest.get("objective", ""),
        }
        block_key = (str(task["dataset"]), str(task["setting"]), int(task["seed"]))
        row["s1_block_pairing_valid"] = not bool(s1_block_reasons.get(block_key))
        row["s1_block_pairing_reasons"] = json.dumps(s1_block_reasons.get(block_key, []), sort_keys=True)
        if s1_block_reasons.get(block_key):
            row["reuse_decision"] = "REUSE_REJECT"
            row["validation_reasons"] = ";".join(filter(None, [str(row["validation_reasons"]),
                                                                      "s1_block_pairing_invalid"]))
        rows.append(row)
    result = {
        "stage": STAGE, "audit_type": "S1_M3OE_REUSE",
        "expected_count": 64, "observed_count": len(rows),
        "valid_count": sum(row["reuse_decision"] == "REUSE_OK" for row in rows),
        "rejected_count": sum(row["reuse_decision"] != "REUSE_OK" for row in rows),
        "source_stage": "S1", "rows": rows,
        "all_reuse_ok": all(row["reuse_decision"] == "REUSE_OK" for row in rows),
        "read_only": True,
    }
    return result


def write_reuse_audit(audit: Mapping[str, object], output_root: str = HERE) -> Tuple[str, str]:
    """Write the CSV/JSON audit atomically; existing audit files are immutable."""

    csv_path = os.path.join(output_root, "stage_S2_reuse_audit.csv")
    json_path = os.path.join(output_root, "stage_S2_reuse_audit.json")
    if os.path.exists(csv_path) or os.path.exists(json_path):
        raise FileExistsError("S2 reuse audit is immutable; remove neither existing audit nor evidence")
    rows = list(audit.get("rows", []))
    fields = sorted({str(key) for row in rows for key in row})
    for path in (csv_path, json_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_csv = csv_path + ".tmp-{}".format(uuid.uuid4().hex)
    temp_json = json_path + ".tmp-{}".format(uuid.uuid4().hex)
    try:
        with open(temp_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        with open(temp_json, "w", encoding="utf-8") as handle:
            json.dump(dict(audit), handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_csv, csv_path); os.replace(temp_json, json_path)
    finally:
        for path in (temp_csv, temp_json):
            if os.path.exists(path):
                os.remove(path)
    return csv_path, json_path


def _atomic_json(path: str, payload: Mapping[str, object]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp-{}".format(uuid.uuid4().hex)
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def archive_failed_attempt(run_dir: str) -> str:
    """Move a failed canonical attempt to an append-only numbered archive."""

    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    archive_root = os.path.join(run_dir, "failed_attempts")
    os.makedirs(archive_root, exist_ok=True)
    numbers = []
    for name in os.listdir(archive_root):
        if name.startswith("attempt_"):
            try: numbers.append(int(name.split("_", 1)[1]))
            except ValueError: pass
    attempt = "attempt_{:03d}".format(max(numbers, default=0) + 1)
    staging = os.path.join(archive_root, ".staging-{}".format(uuid.uuid4().hex))
    os.makedirs(staging)
    for name in os.listdir(run_dir):
        if name == "failed_attempts":
            continue
        shutil.move(os.path.join(run_dir, name), os.path.join(staging, name))
    os.replace(staging, os.path.join(archive_root, attempt))
    return os.path.join(archive_root, attempt)


def _run_one(task: Mapping[str, object], args: argparse.Namespace, worker_id: str) -> Dict[str, object]:
    run_dir = os.path.abspath(str(task["run_dir"]))
    os.makedirs(run_dir, exist_ok=True)
    planned_worker_id = str(task.get("worker_id", ""))
    if planned_worker_id and planned_worker_id != str(worker_id):
        raise RuntimeError("worker assignment differs from frozen co-run plan")
    lock = lock_path_for_task(task, args)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    if os.path.dirname(os.path.abspath(lock)) == run_dir:
        raise RuntimeError("scheduler lock must not be inside canonical run directory")
    try:
        with open(lock, "x", encoding="utf-8") as handle:
            json.dump({"stage": STAGE, "worker_id": worker_id, "pid": os.getpid(),
                       "co_run_pair_id": task.get("co_run_pair_id", ""),
                       "planned_peer_run_key": task.get("planned_peer_run_key", "")}, handle)
        start = time.time()
        stdout_path = os.path.join(run_dir, "stdout.log")
        command = command_for(task, args)
        with open(stdout_path, "a", encoding="utf-8") as stdout:
            stdout.write("\n[stage_S2 launch {} worker={} command={}]\n".format(
                start, worker_id, " ".join(command)))
            stdout.flush()
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT)
            returncode = int(process.wait())
        end = time.time()
        child_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        child_config = _read_json(os.path.join(run_dir, "config.json"))
        if returncode == 0:
            try:
                finalize_artifact_hashes(run_dir)
            except Exception as exc:
                valid, reasons = False, ["artifact_hash_finalization_failed:{}".format(type(exc).__name__)]
            else:
                valid, reasons = validate_equal_run(task, require_formal=True)
        else:
            valid, reasons = False, ["child_exit:{}".format(returncode)]
        return {"formal_run_key": task["formal_run_key"], "co_run_pair_id": task.get("co_run_pair_id", ""),
                "planned_peer_run_key": task.get("planned_peer_run_key", ""),
                "worker_id": worker_id,
                "co_run_state": "COMPLETED",
                "pid": int(process.pid), "start_time": start, "end_time": end,
                "returncode": returncode, "validator_valid": valid,
                "validator_reasons": reasons, "run_dir": run_dir,
                "lock_path": lock,
                "device_identity": child_manifest.get("device_identity", ""),
                "peak_memory_bytes": child_manifest.get("peak_memory_bytes", ""),
                "child_manifest_co_run_pair_id": child_manifest.get("co_run_pair_id", ""),
                "child_manifest_planned_peer_run_key": child_manifest.get("planned_peer_run_key", ""),
                "child_manifest_worker_id": child_manifest.get("worker_id", ""),
                "child_config_co_run_pair_id": child_config.get("co_run_pair_id", ""),
                "child_config_planned_peer_run_key": child_config.get("planned_peer_run_key", ""),
                "child_config_worker_id": child_config.get("worker_id", "")}
    finally:
        try:
            if os.path.isfile(lock):
                os.remove(lock)
        except OSError:
            pass


def _task_attempt_state(task: Mapping[str, object]) -> Tuple[str, List[str]]:
    """Classify a canonical key without mutating it."""

    run_dir = os.path.abspath(str(task["run_dir"]))
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        valid, reasons = validate_equal_run(task, require_formal=True)
        return ("VALID" if valid else "INVALID"), reasons
    if os.path.isdir(run_dir):
        contents = [name for name in os.listdir(run_dir) if name != "failed_attempts"]
        if contents:
            return "INVALID", ["canonical_manifest_missing_or_incomplete"]
    return "MISSING", []


def _retained_record(task: Mapping[str, object], prior: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    run_dir = os.path.abspath(str(task["run_dir"]))
    child_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    child_config = _read_json(os.path.join(run_dir, "config.json"))
    record = {
        "formal_run_key": task["formal_run_key"],
        "co_run_pair_id": task["co_run_pair_id"],
        "planned_peer_run_key": task["planned_peer_run_key"],
        "worker_id": task["worker_id"],
        "run_dir": run_dir,
        "status": "VALID_RETAINED", "validator_valid": True,
        "validator_reasons": [], "co_run_state": "RETAINED",
        "device_identity": child_manifest.get("device_identity", ""),
        "peak_memory_bytes": child_manifest.get("peak_memory_bytes", ""),
        "child_manifest_co_run_pair_id": child_manifest.get("co_run_pair_id", ""),
        "child_manifest_planned_peer_run_key": child_manifest.get("planned_peer_run_key", ""),
        "child_manifest_worker_id": child_manifest.get("worker_id", ""),
        "child_config_co_run_pair_id": child_config.get("co_run_pair_id", ""),
        "child_config_planned_peer_run_key": child_config.get("planned_peer_run_key", ""),
        "child_config_worker_id": child_config.get("worker_id", ""),
    }
    if prior:
        for field in ("pid", "start_time", "end_time", "returncode", "lock_path"):
            if field in prior:
                record[field] = prior[field]
    return record


def run_formal_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace,
                     *, approved: bool = False) -> Dict[str, object]:
    """Run the immutable within-setting pair plan with fail-fast recovery."""

    if not approved:
        raise PermissionError("Stage S2 formal queue requires explicit reviewer/user approval")
    validate_equal_matrix(tasks)
    if int(getattr(args, "max_workers", 2)) != 2:
        raise ValueError("Stage S2 formal concurrency is exactly two workers")
    planned_tasks = annotate_co_run_plan(tasks)
    co_run_plan = build_co_run_plan(planned_tasks)
    co_run_plan_hash = co_run_plan_digest(co_run_plan)
    task_by_key = {str(task["formal_run_key"]): task for task in planned_tasks}
    queue_manifest = str(args.queue_manifest)
    existing_queue = _read_json(queue_manifest) if os.path.isfile(queue_manifest) else {}
    if not existing_queue:
        raise RuntimeError(
            "formal queue requires a pre-frozen co-run plan; run the nonformal dry-run first")
    if existing_queue:
        existing_plan = existing_queue.get("co_run_plan")
        existing_tasks = existing_queue.get("tasks")
        existing_digest = str(existing_queue.get("co_run_plan_digest", ""))
        if not isinstance(existing_plan, list) or not isinstance(existing_tasks, list) \
                or existing_digest != co_run_plan_hash \
                or json.dumps(existing_plan, sort_keys=True, default=str) != \
                json.dumps(co_run_plan, sort_keys=True, default=str) \
                or json.dumps(existing_tasks, sort_keys=True, default=str) != \
                json.dumps(planned_tasks, sort_keys=True, default=str):
            raise RuntimeError("existing queue manifest has no matching frozen co-run plan")
    prior_records = {str(row.get("formal_run_key")): row
                     for row in existing_queue.get("records", [])
                     if isinstance(row, Mapping) and row.get("formal_run_key")}

    # This is intentionally unconditional for every approved launch.  The
    # reuse set is immutable, but its on-disk hashes are not assumed to be
    # fresh; a new audit must pass before even the first Equal subprocess is
    # started.
    s1_raw_root = str(getattr(args, "s1_raw_root", os.path.join(HERE, "stage_S1_raw_results")))
    reuse_tasks = build_reuse_matrix(s1_raw_root)
    reuse_audit = audit_reuse_tasks(reuse_tasks, require_formal=True)
    if not bool(reuse_audit.get("all_reuse_ok")):
        raise RuntimeError("fresh 64-key S1 reuse audit failed; formal queue is fail-closed")
    reuse_audit_digest = hashlib.sha256(json.dumps(
        reuse_audit, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()

    states: Dict[str, Tuple[str, List[str]]] = {
        str(task["formal_run_key"]): _task_attempt_state(task) for task in planned_tasks}
    for pair in co_run_plan:
        pair_keys = [str(key) for key in pair["formal_run_keys"]]
        pair_states = [states[key][0] for key in pair_keys]
        valid_count = sum(state == "VALID" for state in pair_states)
        if valid_count == 1:
            raise RuntimeError(
                "frozen co-run pair is partially complete; refusing singleton recovery: {}".format(
                    {"co_run_pair_id": pair["co_run_pair_id"], "run_keys": pair_keys,
                     "states": pair_states, "decision": "reviewer_user_required"}))
        if any(state == "INVALID" for state in pair_states) \
                and not bool(getattr(args, "rerun_failed", False)):
            raise RuntimeError(
                "existing invalid S2 key requires --rerun_failed within its frozen pair: {}".format(
                    {"co_run_pair_id": pair["co_run_pair_id"], "run_keys": pair_keys,
                     "reasons": {key: states[key][1] for key in pair_keys}}))

    base_payload = {
        "stage": STAGE, "expected_run_count": len(planned_tasks), "launched": 0,
        "completed_record_count": 0, "max_workers": 2, "records": [],
        "tasks": planned_tasks, "co_run_plan": co_run_plan,
        "co_run_plan_digest": co_run_plan_hash,
        "reuse_audit_digest": reuse_audit_digest,
        "reuse_audit_expected_count": reuse_audit.get("expected_count"),
        "reuse_audit_observed_count": reuse_audit.get("observed_count"),
        "reuse_audit_valid_count": reuse_audit.get("valid_count"),
        "reuse_audit_source_root": os.path.abspath(s1_raw_root),
        "resume_state": "PLANNED",
    }
    _atomic_json(queue_manifest, base_payload)

    records: List[Dict[str, object]] = []
    launch_index = 0
    for pair in co_run_plan:
        pair_keys = [str(key) for key in pair["formal_run_keys"]]
        pair_tasks = [task_by_key[key] for key in pair_keys]
        pair_states = [states[key][0] for key in pair_keys]
        if all(state == "VALID" for state in pair_states):
            records.extend(_retained_record(task, prior_records.get(str(task["formal_run_key"])))
                            for task in pair_tasks)
            continue
        if any(state == "VALID" for state in pair_states):
            # Defensive duplicate check after the initial scan; no singleton
            # can be scheduled even if filesystem state changes unexpectedly.
            raise RuntimeError("frozen co-run pair changed during recovery scan")
        for task, state in zip(pair_tasks, pair_states):
            if state == "INVALID":
                archive_failed_attempt(os.path.abspath(str(task["run_dir"])))
        # Every launch is a complete frozen pair.  No compressed pending list
        # and no singleton tail are possible here.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_run_one, task, args, str(task["worker_id"])): task
                for task in pair_tasks
            }
            pair_records = [future.result() for future in as_completed(futures)]
        records.extend(pair_records)
        launch_index += 2
        _atomic_json(queue_manifest, {
            **base_payload, "launched": launch_index,
            "completed_record_count": len(records), "records": records,
            "resume_state": "FAILED" if any(not row["validator_valid"] for row in pair_records) else "RUNNING",
        })
        if any(not row["validator_valid"] for row in pair_records):
            break
    complete = len(records) == len(planned_tasks) and all(row["validator_valid"] for row in records)
    payload = {**base_payload, "launched": launch_index,
               "completed_record_count": len(records), "records": records,
               "resume_state": "COMPLETE" if complete else "FAILED"}
    _atomic_json(queue_manifest, payload)
    return payload


def concurrent_resource_smoke(*, python: str = sys.executable) -> Dict[str, object]:
    """Run two isolated tiny S2Runner/M3OE subprocesses concurrently.

    This is deliberately a nonformal integration fixture: it uses synthetic
    data, one epoch, and temporary output files.  It still crosses the same
    Python subprocess boundary as the formal scheduler and requires CUDA so
    that the two-process resource path is actually exercised.
    """

    with tempfile.TemporaryDirectory(prefix="stage_s2_resource_smoke_") as root:
        output_paths = [os.path.join(root, "worker_{}.json".format(index)) for index in range(2)]
        canonical_dirs = [os.path.join(root, "canonical_{}".format(index)) for index in range(2)]
        child_code = (
            "import json, os, sys\n"
            "sys.path.insert(0, {here!r})\n"
            "import torch\n"
            "if not torch.cuda.is_available(): raise RuntimeError('CUDA required for S2 resource smoke')\n"
            "from types import SimpleNamespace\n"
            "from stage_S2_orchestrator import lock_path_for_task\n"
            "from stage_S2_run import run_tiny_runner_fixture\n"
            "canonical = sys.argv[2]\n"
            "os.makedirs(canonical, exist_ok=True)\n"
            "task = {{'formal_run_key': 'smoke|' + str(os.getpid()), 'run_dir': canonical}}\n"
            "lock = lock_path_for_task(task, SimpleNamespace(raw_root=sys.argv[3]))\n"
            "if os.path.dirname(os.path.abspath(lock)) == os.path.abspath(canonical): raise RuntimeError('lock leaked into canonical dir')\n"
            "os.makedirs(os.path.dirname(lock), exist_ok=True)\n"
            "open(lock, 'x', encoding='utf-8').close()\n"
            "result = run_tiny_runner_fixture(torch.device('cuda:0'), representative=True)\n"
            "os.remove(lock)\n"
            "if any(name.endswith('.lock') for name in os.listdir(canonical)): raise RuntimeError('canonical lock residue')\n"
            "result['pid'] = os.getpid()\n"
            "json.dump(result, open(sys.argv[1], 'w', encoding='utf-8'), sort_keys=True)\n"
            "print('S2_M3OE_CUDA_SMOKE_OK', flush=True)\n"
        ).format(here=HERE)
        processes = [subprocess.Popen([python, "-c", child_code, output_path,
                                       canonical_path, root],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                     for output_path, canonical_path in zip(output_paths, canonical_dirs)]
        outputs = [process.communicate()[0] for process in processes]
        codes = [int(process.returncode) for process in processes]
        records = [_read_json(path) for path in output_paths]
        files_ok = all(os.path.isfile(path) for path in output_paths)
        canonical_isolated = all(os.path.isdir(path)
                                 and not any(name.endswith(".lock") for name in os.listdir(path))
                                 for path in canonical_dirs)
        records_ok = all(record.get("profile") == "kuairand_representative"
                         and record.get("device") == "cuda:0"
                         and int(record.get("main_batch_fingerprints", 0)) == 1
                         and int(record.get("architecture_batch_fingerprints", 0)) == 1
                         for record in records)
        return {"process_count": 2, "returncodes": codes, "outputs": outputs,
                "isolated_output_count": sum(os.path.isfile(path) for path in output_paths),
                "canonical_lock_free": canonical_isolated,
                "cuda_required": True, "records": records,
                "passed": all(code == 0 for code in codes) and files_ok
                and canonical_isolated and records_ok}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S2 Equal two-worker orchestrator")
    parser.add_argument("--raw_root", default=os.path.join(HERE, "stage_S2_raw_results"))
    parser.add_argument("--queue_manifest", default=os.path.join(HERE, "stage_S2_queue_manifest.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--s1_raw_root", default=os.path.join(HERE, "stage_S1_raw_results"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--approved", action="store_true",
                        help="explicit approval to enter the formal luna_runner stage")
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--rerun_failed", action="store_true")
    parser.add_argument("--audit_reuse", action="store_true")
    parser.add_argument("--write_reuse_audit", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    tasks = build_tasks(args.raw_root)
    if args.audit_reuse or args.write_reuse_audit:
        audit = audit_reuse_tasks(build_reuse_matrix(args.s1_raw_root))
        if args.write_reuse_audit:
            write_reuse_audit(audit, HERE)
        print(json.dumps({key: audit[key] for key in ("expected_count", "valid_count", "rejected_count", "all_reuse_ok")}, sort_keys=True))
        # Audit-only invocation is a read-only implementation/review action;
        # it must not fall through into the approval-gated formal launcher.
        if not args.dry_run and not args.approved:
            return 0 if audit.get("all_reuse_ok") else 1
    if args.dry_run:
        co_run_plan = build_co_run_plan(tasks)
        _atomic_json(args.queue_manifest, {
            "stage": STAGE, "expected_run_count": len(tasks), "tasks": tasks,
            "launch_count": 0, "launched": 0, "max_workers": 2,
            "co_run_plan": co_run_plan, "co_run_plan_digest": co_run_plan_digest(co_run_plan),
            "resume_state": "DRY_RUN",
        })
        print("[stage_S2] dry-run matrix", len(tasks), flush=True)
        return 0
    if not args.approved:
        raise PermissionError("review complete is not user approval; pass --approved only after approval")
    result = run_formal_queue(tasks, args, approved=True)
    return 0 if result["resume_state"] == "COMPLETE" else 1


__all__ = [
    "S2_REQUIRED_FILES", "S2_BATCH_FIELDS", "build_tasks", "validate_matrix_tasks",
    "command_for", "lock_path_for_task", "build_co_run_plan", "co_run_plan_digest",
    "annotate_co_run_plan", "validate_batch_trace_contract", "validate_co_run_provenance",
    "validate_equal_run", "validate_s1_reuse_task", "audit_reuse_tasks",
    "write_reuse_audit", "finalize_artifact_hashes", "archive_failed_attempt", "run_formal_queue",
    "concurrent_resource_smoke", "build_argparser", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
