"""Stage S3 reuse audit, source/pair validators, and fail-closed queue runner."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_orchestrator import (  # noqa: E402
    _check_checkpoint_finite, validate_raw_prediction_metrics, validate_run as _validate_s1_run,
)
from stage_S2_orchestrator import (  # noqa: E402
    _read_json as _s2_read_json, validate_equal_run as _validate_equal_run,
)
from stage_S2_protocol import coefficient_table_hash as _coefficient_table_hash  # noqa: E402
from stage_S3_protocol import (  # noqa: E402
    ACCESS, AMENDMENT_REUSE_AUDIT_BASENAME, BACKBONE, CHECKPOINTS,
    EXPECTED_FRESH_RUNS, EXPECTED_INVENTORY_ROWS, EXPECTED_LAUNCH_GROUPS,
    EXPECTED_SAMPLE_FULL_REPLACEMENTS, EXPECTED_TARGET_ONLY_RUNS,
    FORMAL_PROTOCOL, FULL, FRESH_SAMPLE_FULL_ROLE, FRESH_TARGET_ROLE,
    MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, REVIEW_REPORT_NAME, S3_DATASETS,
    S3_SEEDS, STAGE, build_fresh_matrix, build_historical_audit_matrix,
    build_launch_plan, build_reuse_matrix, co_run_plan_digest,
    canonical_dataset,
    formal_protocol_for, fresh_formal_key, sample_full_replacement_key,
    sample_target_pending_key, source_inventory, source_inventory_digest,
    target_for, validate_fresh_matrix, validate_reuse_matrix,
)

_NONFINITE_REASONS = {
    "nonfinite_training_loss", "nonfinite_oracle_artifact_value",
    "nonfinite_routing_statistic", "zero_norm_gradient_or_no_valid_cosine_probe",
    "nonfinite_numeric_value",
}
_NONFINITE_ROW_KEYS = {
    "seed", "epoch", "step", "domain", "task", "cell", "parameter_group", "kind", "split",
}
_REVIEWED_FORWARD_MODEL_SHA256 = MODEL_SOURCE_SHA256
_VALIDATOR_REPAIR_QUEUE_SHA256 = "0edcc92249de8c07e556097ec3601d5e3223e8fcb00693748dda7c87a79c8766"
_VALIDATOR_REPAIR_PRIOR_SOURCE_DIGEST = "2f995321583594e16c96f63ecd23aa3938d7aac02ae81c1918881c44aadc0279"
_VALIDATOR_REPAIR_PRIOR_ORCHESTRATOR_SHA256 = "c06c5b0498292c2b264c5a8a4beedab38f42df4bbf4c7c96428cdc55e3d1a430"
_VALIDATOR_REPAIR_ARCHIVE_SHA256 = "770cea174992ee6e63fbe6d51a3a86ad6f3a818736ea22554ea9af08aa3f7d3b"
_VALIDATOR_REPAIR_SOURCE = "experiments/stage_S3_orchestrator.py"
_VALIDATOR_REPAIR_GROUP = "s3_kuairand_natural_seed2023"
_COEFFICIENT_REPAIR_QUEUE_SHA256 = "1f614dee34d3b5c29752bdc539d3a9bb44e6036b1a6dc4b6266eea335cf35efa"
_COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST = "4ec39077ed69b0483a051360dfee43e38ca7e18572680a1f7aa53a6924314fbe"
_COEFFICIENT_REPAIR_PRIOR_SOURCE_HASHES = {
    "experiments/stage_S3_orchestrator.py":
        "5b9e779fd246c3664d31e8f9a3d4fed942c5bda52b4108227858270215207635",
    "experiments/stage_S3_report.py":
        "0e003f3e15288b6f944ca5a3ab89079a7865ed984f570b35fc63e217d17591ee",
}
_COEFFICIENT_REPAIR_CANONICAL_TREE_DIGEST = \
    "5b47fb65136afead0ab543f90a92d186116aa64e2112fa15aa8c7df688aaf645"
_STRICT_FOUR_WAY_FIELDS = (
    "dataset_file_hash", "split_row_fingerprints", "split_row_id_hashes",
    "physical_view_fingerprints", "preprocessing_metadata", "local_to_physical",
    "physical_domain_identity", "train_domain_counts", "domain_shares",
    "initial_parameter_state_hash", "model_parameter_manifest", "forward_flops_estimate",
    "batch_order_hash", "architecture_batch_order_hash", "optimizer_steps",
    "architecture_optimizer_steps", "examples_processed", "architecture_examples_processed",
    "fixed_final_epoch", "common_epochs", "metric_implementation",
    "checkpoint_selection_frozen_before_test", "test_evaluated_during_training",
    "test_metric_in_epoch_history", "runtime_fingerprint",
)

# ``artifact_hashes[objective_coefficients.json]`` is a SHA-256 of the file
# bytes.  ``coefficient_table_hash`` is a canonical JSON hash of the table
# content with its self-declared hash removed.  The two values are deliberately
# kept separate: a legacy promoted Sample artifact omitted the fixed D/T
# presentation fields, so its bytes and raw table hash differ from a fresh
# replacement even though the normalized protocol table is identical.
_COEFFICIENT_TABLE_REQUIRED_FIELDS = (
    "domain_num", "task_num", "train_counts", "total_train_count",
    "domain_share_p", "loss_coefficient_w", "effective_domain_coefficient_a",
    "effective_cell_coefficient", "population_mean_weight", "objective_name",
    "objective_formula", "sampling", "double_weighting", "coefficient_table_hash",
)
_COEFFICIENT_SCHEMA_FIELDS = ("domain_num", "task_num")
_SAMPLE_OBJECTIVE_NAMES = frozenset((
    "sample", "sample_population_mean_task_bce", "direct_per_example_task_mean_bce",
))
_EQUAL_OBJECTIVE_NAMES = frozenset(("equal", "equal_domain_task_mean_bce"))


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object at {}".format(path))
    return value


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV header missing: {}".format(path))
        return list(reader.fieldnames), [dict(row) for row in reader]


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_json(path: str, payload: Mapping[str, object]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".s3-json-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_csv_atomic(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".s3-csv-", suffix=".tmp",
                                     dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _read_nonfinite_inventory(path: str, *, allow_empty_object: bool = False) -> List[Dict[str, object]]:
    def reject_constant(value: str):
        raise ValueError("non-standard JSON constant {}".format(value))
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key {}".format(key))
            result[key] = value
        return result
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicates)
    if allow_empty_object and value == {}:
        return []
    if not isinstance(value, list):
        raise ValueError("nonfinite inventory must be a JSON array")
    for event in value:
        if not isinstance(event, dict):
            raise ValueError("nonfinite inventory event must be an object")
        required = {"file", "row_index", "row_key", "field", "value", "reason"}
        if set(event) != required:
            raise ValueError("nonfinite inventory event schema mismatch")
        if isinstance(event["row_index"], bool) or not isinstance(event["row_index"], int):
            raise ValueError("nonfinite inventory row_index must be an integer")
        if not isinstance(event["row_key"], dict) or not isinstance(event["file"], str) \
                or not isinstance(event["field"], str) or not isinstance(event["value"], str) \
                or str(event["reason"]) not in _NONFINITE_REASONS:
            raise ValueError("nonfinite inventory event has invalid fields")
        if str(event["value"]).strip().lower() not in ("nan", "+nan", "-nan"):
            raise ValueError("nonfinite inventory value must use the canonical NaN spelling")
    return value


def _nonfinite_inventory_failure(path: str, *, allow_empty_object: bool = False) -> Optional[str]:
    try:
        value = _read_nonfinite_inventory(path, allow_empty_object=allow_empty_object)
    except Exception:
        return "nonfinite_inventory_invalid"
    return "nonfinite_inventory_not_empty" if value else None


def _queue_state_digest(queue: Mapping[str, object]) -> str:
    excluded = {"state_digest", "history", "history_digest", "updated_at"}
    return _digest({key: value for key, value in queue.items() if key not in excluded})


def _write_queue_state(path: str, queue: Dict[str, object], event: str = "update") -> None:
    history = list(queue.get("history", []))
    history.append({"event": str(event), "time": time.time(),
                    "prior_state_digest": queue.get("state_digest", "")})
    queue["history"] = history
    queue["history_digest"] = _digest(history)
    queue["state_digest"] = _queue_state_digest(queue)
    _write_atomic_json(path, queue)


def _validate_queue_history(queue: Mapping[str, object]) -> List[str]:
    reasons = []
    history = queue.get("history", [])
    if not isinstance(history, list) or _digest(history) != queue.get("history_digest"):
        reasons.append("queue_history_digest_invalid")
    if queue.get("state_digest") != _queue_state_digest(queue):
        reasons.append("queue_state_digest_invalid")
    return reasons


@contextlib.contextmanager
def _parent_queue_guard(queue_path: str) -> Iterator[None]:
    guard = os.path.abspath(queue_path) + ".guard"
    os.makedirs(os.path.dirname(guard), exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, (str(os.getpid()) + "\n").encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.remove(guard)
        except FileNotFoundError:
            pass


def _source_artifact_hashes(run_dir: str, manifest: Mapping[str, object]) -> Dict[str, str]:
    names = manifest.get("artifact_hashes", {})
    if not isinstance(names, Mapping):
        return {}
    result = {}
    for name in sorted(names):
        path = os.path.join(run_dir, str(name))
        if os.path.isfile(path):
            result[str(name)] = _file_hash(path)
    return result


def _s1_task_for_s3(task: Mapping[str, object]) -> Dict[str, object]:
    return {
        "run_type": task["run_type"], "dataset": task["dataset"],
        "setting": task["setting"], "backbone": BACKBONE, "seed": task["seed"],
        "physical_domain": task.get("physical_domain"),
        "formal_run_key": task["source_formal_run_key"],
        "run_dir": task["source_run_dir"],
    }


def _s2_equal_task_for_s3(task: Mapping[str, object], s2_raw_root: str) -> Dict[str, object]:
    """Recover the exact accepted S2 co-run identity for a reused Equal FULL key."""
    queue_path = os.path.join(os.path.dirname(os.path.abspath(s2_raw_root)),
                              "stage_S2_queue_manifest.json")
    if not os.path.isfile(queue_path):
        raise FileNotFoundError("accepted S2 queue is required for Equal FULL provenance")
    queue = _read_json(queue_path)
    if queue.get("stage") != "S2" or queue.get("resume_state") != "COMPLETE":
        raise ValueError("S2 Equal provenance queue is not complete")
    matches = [item for item in queue.get("tasks", [])
               if isinstance(item, Mapping)
               and str(item.get("formal_run_key", "")) == str(task["source_formal_run_key"])]
    if len(matches) != 1:
        raise ValueError("S2 queue does not identify exactly one Equal FULL source task")
    source = dict(matches[0])
    expected = {
        "method": "equal", "run_type": "joint", "dataset": task["dataset"],
        "setting": task["setting"], "backbone": BACKBONE, "seed": int(task["seed"]),
        "formal_run_key": task["source_formal_run_key"],
        "run_dir": str(task["source_run_dir"]),
    }
    for field, value in expected.items():
        observed = source.get(field)
        if field == "run_dir":
            observed = os.path.abspath(str(observed))
            value = os.path.abspath(str(value))
        if observed != value:
            raise ValueError("accepted S2 Equal queue task identity mismatch:{}".format(field))
    return source


def _source_metrics_audit(task: Mapping[str, object], manifest: Mapping[str, object]) -> List[str]:
    run_dir = str(task["source_run_dir"])
    reasons = []
    if not os.path.isfile(os.path.join(run_dir, "predictions_test.npz")):
        reasons.append("missing_predictions_test.npz")
    expected_model_hash = str(manifest.get("code_hashes", {}).get(MODEL_SOURCE_KEY, "")) \
        if isinstance(manifest.get("code_hashes"), Mapping) else ""
    if task.get("reuse_role") == "controlled_full_source" and not expected_model_hash:
        reasons.append("model_source_hash_missing")
    if task.get("reuse_role") == "controlled_full_source" \
            and expected_model_hash != MODEL_SOURCE_SHA256:
        reasons.append("model_source_hash_mismatch")
    return reasons


def audit_reuse_tasks(tasks: Optional[Sequence[Mapping[str, object]]] = None,
                      *, s1_raw_root: Optional[str] = None,
                      s2_raw_root: Optional[str] = None,
                      require_formal: bool = True) -> Dict[str, object]:
    historical = list(tasks) if tasks is not None else build_historical_audit_matrix(
        s1_raw_root or os.path.join(HERE, "stage_S1_raw_results"),
        s2_raw_root or os.path.join(HERE, "stage_S2_raw_results"))
    validate_reuse_matrix(historical)
    rows: List[Dict[str, object]] = []
    for task in historical:
        source_dir = str(task["source_run_dir"])
        if task.get("reuse_role") == "rejected_controlled_source":
            try:
                manifest = _read_json(os.path.join(source_dir, "run_manifest.json"))
                reasons = _source_metrics_audit(task, manifest)
                if not manifest.get("code_hashes", {}).get(MODEL_SOURCE_KEY):
                    reasons.append("model_source_hash_mismatch")
            except Exception as exc:
                manifest, reasons = {}, ["source_manifest_invalid:{}".format(type(exc).__name__)]
            status = "REJECTED_MISSING_MODEL_SOURCE_HASH" if "model_source_hash_mismatch" in reasons else "REJECTED"
        elif task.get("reuse_role") == "controlled_full_source":
            s2_task = _s2_equal_task_for_s3(task, s2_raw_root or os.path.join(HERE, "stage_S2_raw_results"))
            valid, reasons = _validate_equal_run(s2_task, require_formal=require_formal)
            try:
                manifest = _read_json(os.path.join(source_dir, "run_manifest.json"))
                reasons = list(reasons) + _source_metrics_audit(task, manifest)
            except Exception as exc:
                manifest = {}
                reasons.append("source_manifest_invalid:{}".format(type(exc).__name__))
            valid = bool(valid and not reasons)
            status = "REUSE_OK" if valid else "REJECTED"
        else:
            s1_task = _s1_task_for_s3(task)
            valid, reasons = _validate_s1_run(s1_task, require_formal=require_formal)
            try:
                manifest = _read_json(os.path.join(source_dir, "run_manifest.json"))
                reasons = list(reasons) + _source_metrics_audit(task, manifest)
            except Exception as exc:
                manifest = {}
                reasons.append("source_manifest_invalid:{}".format(type(exc).__name__))
            valid = bool(valid and not reasons)
            status = "LEGACY_CONTEXT_OK" if valid else "REJECTED"
        code_hashes = manifest.get("code_hashes", {}) if isinstance(manifest, Mapping) else {}
        rows.append({
            **{key: task.get(key, "") for key in (
                "method", "reuse_role", "dataset", "setting", "backbone", "seed",
                "physical_domain", "formal_run_key", "source_formal_run_key", "source_stage",
                "source_run_dir")},
            "status": status, "valid": status in ("REUSE_OK", "LEGACY_CONTEXT_OK"),
            "strict_source_identity": status == "REUSE_OK",
            "forward_model_source_sha256": (code_hashes.get(MODEL_SOURCE_KEY, "")
                                              if isinstance(code_hashes, Mapping) else ""),
            "artifact_hashes": _source_artifact_hashes(source_dir, manifest) if manifest else {},
            "reasons": sorted(set(str(reason) for reason in reasons)),
        })
    summary = {
        "historical_artifact_count": len(rows),
        "controlled_reuse_valid": sum(row["status"] == "REUSE_OK" for row in rows),
        "controlled_reuse_rejected": sum(row["status"] == "REJECTED_MISSING_MODEL_SOURCE_HASH"
                                          for row in rows),
        "legacy_context_valid": sum(row["status"] == "LEGACY_CONTEXT_OK" for row in rows),
        "all_required_audit_rows_present": len(rows) == 40,
    }
    expected = (summary["controlled_reuse_valid"] == 8
                and summary["controlled_reuse_rejected"] == 8
                and summary["legacy_context_valid"] == 24)
    return {"rows": rows, "summary": summary, "valid": bool(expected)}


def write_amendment_reuse_audit(audit: Mapping[str, object],
                                output_root: str = HERE) -> Tuple[str, str]:
    root = os.path.abspath(output_root)
    rows = list(audit.get("rows", []))
    basename = os.path.join(root, AMENDMENT_REUSE_AUDIT_BASENAME)
    csv_path, json_path = basename + ".csv", basename + ".json"
    fields = ["method", "reuse_role", "dataset", "setting", "backbone", "seed",
              "physical_domain", "formal_run_key", "source_formal_run_key", "source_stage",
              "source_run_dir", "status", "valid", "strict_source_identity",
              "forward_model_source_sha256", "artifact_hashes", "reasons"]
    _write_csv_atomic(csv_path, rows, fields)
    _write_atomic_json(json_path, {"summary": audit.get("summary", {}), "rows": rows,
                                   "valid": bool(audit.get("valid", False))})
    return csv_path, json_path


def write_reuse_audit(audit: Mapping[str, object], output_root: str = HERE) -> Tuple[str, str]:
    return write_amendment_reuse_audit(audit, output_root)


def annotate_pairs(tasks: Sequence[Mapping[str, object]],
                   launch_plan: Optional[Sequence[Mapping[str, object]]] = None) -> List[Dict[str, object]]:
    # Protocol builders return mutable task dicts; annotate those in place so
    # the exact approved objects flow unchanged into queue/task inventories.
    tasks = [task if isinstance(task, dict) else dict(task) for task in tasks]
    plan = list(launch_plan) if launch_plan is not None else build_launch_plan(tasks)
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    if len(by_key) != len(tasks):
        raise ValueError("duplicate formal keys before S3 launch annotation")
    visited = set()
    for group in plan:
        workers = list(group["workers"])
        if len(workers) != 2:
            raise ValueError("Stage S3 launch groups must contain exactly two workers")
        for worker in workers:
            key = str(worker["formal_run_key"])
            task = by_key.get(key)
            if task is None or key in visited:
                raise ValueError("Stage S3 launch plan has missing or duplicate tasks")
            task.update({
                "launch_group_id": group["launch_group_id"],
                "launch_group_kind": group["group_kind"],
                "co_run_pair_id": group["launch_group_id"],
                "planned_peer_run_key": worker["planned_peer_run_key"],
                "worker_id": worker["worker_id"],
            })
            visited.add(key)
    if visited != set(by_key):
        raise ValueError("not every Stage S3 fresh key is in one frozen launch group")
    return tasks


def command_for(task: Mapping[str, object], args: argparse.Namespace,
                *, run_dir: Optional[str] = None, attempt_id: str = "") -> List[str]:
    python = str(getattr(args, "python", sys.executable))
    target = target_for(str(task["dataset"]), str(task["setting"]))
    command = [
        python, os.path.join(HERE, "stage_S3_run.py"),
        "--run_type", "joint", "--dataset", str(task["dataset"]),
        "--setting", str(task["setting"]), "--backbone", BACKBONE,
        "--seed", str(task["seed"]), "--run_dir", os.path.abspath(run_dir or str(task["run_dir"])),
        "--device", str(getattr(args, "device", "cuda:0")),
        "--objective_weighting", str(task["objective_weighting"]),
        "--shared_access", str(task["shared_access"]),
        "--fresh_role", str(task["fresh_role"]),
        "--formal_run_key", str(task["formal_run_key"]),
        "--source_formal_run_key", str(task.get("source_formal_run_key", "")),
        "--co_run_pair_id", str(task["co_run_pair_id"]),
        "--planned_peer_run_key", str(task["planned_peer_run_key"]),
        "--worker_id", str(task["worker_id"]), "--attempt_id", str(attempt_id),
    ]
    if getattr(args, "dataset_path", ""):
        command.extend(["--dataset_path", str(args.dataset_path)])
    if target.get("local_domain") not in (1, 2):
        raise AssertionError("invalid S3 target mapping")
    return command


def lock_path_for_task(task: Mapping[str, object], raw_root: str) -> str:
    return os.path.join(os.path.abspath(raw_root), ".stage_S3_locks",
                        hashlib.sha256(str(task["formal_run_key"]).encode("utf-8")).hexdigest() + ".lock")


def _acquire_group_locks(tasks: Sequence[Mapping[str, object]], raw_root: str,
                         group_id: str, wave: int, attempt_id: str) -> List[str]:
    acquired = []
    try:
        for task in tasks:
            path = lock_path_for_task(task, raw_root)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({
                    "formal_run_key": str(task["formal_run_key"]),
                    "launch_group_id": str(group_id), "wave": int(wave),
                    "attempt_id": str(attempt_id), "parent_pid": os.getpid(),
                }, sort_keys=True) + "\n")
            acquired.append(path)
        return acquired
    except Exception:
        # Only release locks acquired by this exact call; pre-existing locks are
        # evidence of another process and are never removed here.
        for path in acquired:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def _release_group_locks(paths: Sequence[str]) -> None:
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def _artifact_hash_reasons(run_dir: str, manifest: Mapping[str, object]) -> List[str]:
    reasons = []
    hashes = manifest.get("artifact_hashes", {})
    if not isinstance(hashes, Mapping):
        return ["artifact_hash_inventory_missing"]
    for filename, expected in hashes.items():
        path = os.path.join(run_dir, str(filename))
        if not os.path.isfile(path):
            reasons.append("artifact_missing:{}".format(filename))
        elif _file_hash(path) != str(expected):
            reasons.append("artifact_hash_mismatch:{}".format(filename))
    return reasons


def _coefficient_objective_kind(table: Mapping[str, object],
                                manifest: Optional[Mapping[str, object]] = None) -> str:
    """Return the frozen objective family represented by a coefficient table."""

    candidates = [table.get("objective_name", "")]
    if isinstance(manifest, Mapping):
        candidates.extend((manifest.get("objective_weighting", ""),
                           manifest.get("objective", ""),
                           manifest.get("method", "")))
    for value in candidates:
        name = str(value or "").strip().lower()
        if name in _SAMPLE_OBJECTIVE_NAMES:
            return "sample"
        if name in _EQUAL_OBJECTIVE_NAMES:
            return "equal"
    return ""


def _normalize_coefficient_table(table: Mapping[str, object],
                                 manifest: Optional[Mapping[str, object]] = None
                                 ) -> Tuple[Dict[str, object], List[str], List[str]]:
    """Normalize only the fixed D/T metadata omitted by the retained legacy run.

    This function never changes an artifact on disk and never rewrites the
    declared hash.  It returns a copy used only for semantic pair comparison,
    the fields added to that copy, and schema errors.  Numeric coefficients are
    not inferred or repaired: a missing field outside the two fixed protocol
    metadata fields is invalid.
    """

    normalized = dict(table)
    kind = _coefficient_objective_kind(normalized, manifest)
    missing = [field for field in _COEFFICIENT_SCHEMA_FIELDS if field not in normalized]
    errors: List[str] = []
    if missing:
        if kind != "sample" or not set(missing).issubset(set(_COEFFICIENT_SCHEMA_FIELDS)):
            errors.append("coefficient_table_schema_missing:{}".format(",".join(sorted(missing))))
        else:
            normalized["domain_num"] = 3
            normalized["task_num"] = 2
    for field, expected in (("domain_num", 3), ("task_num", 2)):
        if field in normalized:
            try:
                if int(normalized[field]) != expected:
                    errors.append("coefficient_table_schema_mismatch:{}".format(field))
            except (TypeError, ValueError):
                errors.append("coefficient_table_schema_invalid:{}".format(field))
    return normalized, sorted(set(missing)), sorted(set(errors))


def _coefficient_table_semantic_reasons(table: Mapping[str, object],
                                        objective_kind: str) -> List[str]:
    """Check the immutable S3 coefficient semantics without touching artifacts."""

    reasons: List[str] = []
    missing = [field for field in _COEFFICIENT_TABLE_REQUIRED_FIELDS
               if field not in table]
    if missing:
        reasons.append("coefficient_table_fields_missing:{}".format(",".join(missing)))
        return sorted(set(reasons))
    if objective_kind not in ("sample", "equal"):
        reasons.append("coefficient_table_objective_unknown")
        return sorted(set(reasons))
    try:
        domain_num = int(table["domain_num"])
        task_num = int(table["task_num"])
    except (TypeError, ValueError):
        reasons.append("coefficient_table_schema_invalid:D_or_T")
        return sorted(set(reasons))
    if domain_num != 3:
        reasons.append("coefficient_table_schema_mismatch:domain_num")
    if task_num != 2:
        reasons.append("coefficient_table_schema_mismatch:task_num")

    mappings = {}
    for field in ("train_counts", "domain_share_p", "loss_coefficient_w",
                  "effective_domain_coefficient_a", "effective_cell_coefficient"):
        value = table.get(field)
        if not isinstance(value, Mapping):
            reasons.append("coefficient_table_mapping_missing:{}".format(field))
        else:
            mappings[field] = value
    if reasons:
        return sorted(set(reasons))
    expected_keys = {"0", "1", "2"}
    for field, values in mappings.items():
        observed_keys = {str(key) for key in values.keys()}
        if observed_keys != expected_keys:
            reasons.append("coefficient_table_domain_keys_mismatch:{}".format(field))
    if reasons:
        return sorted(set(reasons))

    try:
        counts = {key: int(mappings["train_counts"][key]) for key in expected_keys}
        total = int(table["total_train_count"])
        shares = {key: float(mappings["domain_share_p"][key]) for key in expected_keys}
        weights = {key: float(mappings["loss_coefficient_w"][key]) for key in expected_keys}
        effective = {key: float(mappings["effective_domain_coefficient_a"][key])
                     for key in expected_keys}
        cell = {key: float(mappings["effective_cell_coefficient"][key])
                for key in expected_keys}
        population_mean = float(table["population_mean_weight"])
    except (TypeError, ValueError, KeyError):
        return ["coefficient_table_numeric_invalid"]
    if any(count <= 0 for count in counts.values()) or total != sum(counts.values()):
        reasons.append("coefficient_table_train_counts_invalid")
    for key in expected_keys:
        values = (shares[key], weights[key], effective[key], cell[key])
        if not all(math.isfinite(value) for value in values):
            reasons.append("coefficient_table_nonfinite:{}".format(key))
            continue
        expected_share = counts[key] / float(total) if total else float("nan")
        expected_weight = 1.0 if objective_kind == "sample" else 1.0 / (3.0 * expected_share)
        expected_effective = expected_share if objective_kind == "sample" else 1.0 / 3.0
        expected_cell = expected_effective / 2.0
        if abs(shares[key] - expected_share) > 1e-12:
            reasons.append("coefficient_table_domain_share_mismatch:{}".format(key))
        if abs(weights[key] - expected_weight) > 1e-12:
            reasons.append("coefficient_table_loss_coefficient_mismatch:{}".format(key))
        if abs(effective[key] - expected_effective) > 1e-12:
            reasons.append("coefficient_table_effective_domain_mismatch:{}".format(key))
        if abs(cell[key] - expected_cell) > 1e-12:
            reasons.append("coefficient_table_effective_cell_mismatch:{}".format(key))
    if abs(sum(shares.values()) - 1.0) > 1e-12:
        reasons.append("coefficient_table_domain_shares_not_one")
    if abs(population_mean - 1.0) > 1e-12:
        reasons.append("coefficient_table_population_mean_not_one")
    if table.get("double_weighting") is not False:
        reasons.append("coefficient_table_double_weighting")
    expected_formula = {
        "sample": "mean_i(mean_t(BCE(p_it,y_it)))",
        "equal": "mean_i( w[d_i] * mean_t(BCE(p_it,y_it)) )",
    }[objective_kind]
    if str(table.get("objective_formula", "")) != expected_formula:
        reasons.append("coefficient_table_objective_formula_mismatch")
    if str(table.get("sampling", "")) != "ordinary_shuffled_concatenated_loader":
        reasons.append("coefficient_table_sampling_mismatch")
    return sorted(set(reasons))


def _coefficient_artifact_identity(run_dir: str,
                                   manifest: Mapping[str, object]) -> Dict[str, object]:
    """Describe one objective artifact using byte, raw-content, and normalized hashes.

    ``file_sha256`` is the byte-level artifact identity used by the manifest.
    ``raw_table_hash`` is the declared table-content identity recomputed from
    the JSON object.  ``normalized_table_hash`` is used only for A1 pair
    comparison after adding the two fixed D/T fields to the retained legacy
    Sample table.  No value returned here authorizes rewriting the artifact.
    """

    path = os.path.join(os.path.abspath(str(run_dir)), "objective_coefficients.json")
    result: Dict[str, object] = {
        "path": path, "file_sha256": "", "declared_table_hash": "",
        "manifest_table_hash": manifest.get("coefficient_table_hash", ""),
        "raw_table_hash": "", "normalized_table_hash": "", "objective_kind": "",
        "schema_fields_added": [], "table": {}, "normalized_table": {}, "reasons": [],
    }
    reasons: List[str] = []
    if not os.path.isfile(path):
        result["reasons"] = ["objective_coefficients_artifact_missing"]
        return result
    result["file_sha256"] = _file_hash(path)
    artifact_hashes = manifest.get("artifact_hashes", {})
    expected_file_hash = artifact_hashes.get("objective_coefficients.json") \
        if isinstance(artifact_hashes, Mapping) else None
    if not expected_file_hash:
        reasons.append("objective_coefficients_artifact_hash_missing")
    elif str(expected_file_hash) != str(result["file_sha256"]):
        reasons.append("artifact_hash_mismatch:objective_coefficients.json")
    try:
        table = _read_json(path)
    except Exception as exc:
        result["reasons"] = ["objective_coefficients_invalid:{}".format(type(exc).__name__)]
        return result
    result["table"] = table
    kind = _coefficient_objective_kind(table)
    result["objective_kind"] = kind
    manifest_kind = _coefficient_objective_kind({}, manifest)
    if not kind:
        reasons.append("coefficient_table_objective_unknown")
    if not manifest_kind or manifest_kind != kind:
        reasons.append("coefficient_table_objective_manifest_mismatch")
    declared = table.get("coefficient_table_hash", "")
    result["declared_table_hash"] = declared
    manifest_hash = manifest.get("coefficient_table_hash", "")
    if not declared:
        reasons.append("coefficient_table_hash_missing")
    if manifest_hash in (None, ""):
        reasons.append("manifest_mismatch:coefficient_table_hash")
    elif str(manifest_hash) != str(declared):
        reasons.append("manifest_mismatch:coefficient_table_hash")
    try:
        raw_hash = _coefficient_table_hash(table)
        result["raw_table_hash"] = raw_hash
        if str(declared) != raw_hash:
            reasons.append("coefficient_table_hash_mismatch")
    except (TypeError, ValueError):
        reasons.append("coefficient_table_canonicalization_failed")
    normalized, added, schema_errors = _normalize_coefficient_table(table)
    result["schema_fields_added"] = list(added)
    result["normalized_table"] = normalized
    reasons.extend(schema_errors)
    reasons.extend(_coefficient_table_semantic_reasons(normalized, kind))
    try:
        result["normalized_table_hash"] = _coefficient_table_hash(normalized)
    except (TypeError, ValueError):
        reasons.append("coefficient_table_normalization_hash_failed")
    result["reasons"] = sorted(set(reasons))
    return result


def _coefficient_artifact_reasons(run_dir: str,
                                  manifest: Mapping[str, object]) -> List[str]:
    return list(_coefficient_artifact_identity(run_dir, manifest).get("reasons", []))


def _coefficient_pair_reasons(full_manifest: Mapping[str, object],
                              target_manifest: Mapping[str, object],
                              full_run_dir: str, target_run_dir: str) -> List[str]:
    """Compare objective semantics while intentionally ignoring cross-file byte SHA."""

    full_path = os.path.join(os.path.abspath(str(full_run_dir)), "objective_coefficients.json")
    target_path = os.path.join(os.path.abspath(str(target_run_dir)), "objective_coefficients.json")
    full_exists, target_exists = os.path.isfile(full_path), os.path.isfile(target_path)
    # Pair callers normally run after the per-run artifact validator.  Keeping
    # the metadata fallback makes dependency-light synthetic pairing fixtures
    # useful without weakening the per-run fail-closed artifact check.
    if not full_exists or not target_exists:
        left = full_manifest.get("coefficient_table_hash", "")
        right = target_manifest.get("coefficient_table_hash", "")
        if left not in (None, "") and right not in (None, "") and left != right:
            return ["pairing_field_mismatch:coefficient_table_hash"]
        return []
    full = _coefficient_artifact_identity(full_run_dir, full_manifest)
    target = _coefficient_artifact_identity(target_run_dir, target_manifest)
    reasons: List[str] = []
    for label, identity in (("full", full), ("target", target)):
        reasons.extend("pairing_objective_table_invalid:{}:{}".format(label, reason)
                       for reason in identity.get("reasons", []))
    if full.get("objective_kind") != target.get("objective_kind"):
        reasons.append("pairing_objective_table_kind_mismatch")
    full_hash = str(full.get("normalized_table_hash", ""))
    target_hash = str(target.get("normalized_table_hash", ""))
    if full_hash and target_hash and full_hash != target_hash:
        reasons.append("pairing_table_content_hash_mismatch")
    return sorted(set(reasons))


def validate_batch_trace(task: Mapping[str, object], run_dir: str,
                         manifest: Mapping[str, object]) -> List[str]:
    path = os.path.join(run_dir, "batch_trace.csv")
    if not os.path.isfile(path):
        return ["batch_trace_missing"]
    fields, rows = _read_csv(path)
    reasons = []
    required = {"kind", "epoch", "batch_fingerprint", "target_count", "L_full",
                "L_target", "L_non", "decomposition_residual",
                "gradient_substitution_valid", "gradient_partition_hash"}
    if not required.issubset(fields):
        return ["batch_trace_schema_missing_fields"]
    main = [row for row in rows if row.get("kind") == "train"]
    architecture = [row for row in rows if row.get("kind") == "architecture"]
    expected_main = int(manifest.get("batches_per_epoch", 0)) * 50
    if len(main) != expected_main:
        reasons.append("main_batch_count_mismatch")
    if len(architecture) != 50:
        reasons.append("architecture_batch_count_mismatch")
    fingerprints = []
    for row in main:
        fingerprints.append(str(row["batch_fingerprint"]))
        try:
            if int(row["target_count"]) <= 0:
                reasons.append("target_absent_from_batch")
            residual = float(row["decomposition_residual"])
            full = float(row["L_full"]); target_loss = float(row["L_target"])
            non_target = float(row["L_non"])
            if not all(math.isfinite(value) for value in (residual, full, target_loss, non_target)):
                reasons.append("nonfinite_loss_component")
            if abs(full - target_loss - non_target) > 1e-6 or abs(residual) > 1e-6:
                reasons.append("loss_decomposition_mismatch")
            if str(row["gradient_substitution_valid"]).lower() not in ("1", "true"):
                reasons.append("gradient_substitution_invalid")
            if str(row["gradient_partition_hash"]) != str(manifest.get("gradient_partition_hash")):
                reasons.append("gradient_partition_hash_mismatch")
        except (TypeError, ValueError, KeyError):
            reasons.append("batch_trace_numeric_invalid")
    architecture_fingerprints = [str(row["batch_fingerprint"]) for row in architecture]
    if len(fingerprints) != int(manifest.get("batch_fingerprint_count", -1)):
        reasons.append("main_fingerprint_count_mismatch")
    if len(architecture_fingerprints) != int(manifest.get("architecture_batch_fingerprint_count", -1)):
        reasons.append("architecture_fingerprint_count_mismatch")
    # stage_S3_run hashes these lists with json.dumps(list). Keep byte-for-byte
    # parity with that persisted contract (the general object digest sorts keys).
    fingerprint_digest = lambda values: hashlib.sha256(
        json.dumps(values).encode("utf-8")).hexdigest()
    if fingerprint_digest(fingerprints) != str(manifest.get("batch_order_hash", "")):
        reasons.append("main_batch_order_hash_mismatch")
    if fingerprint_digest(architecture_fingerprints) != str(manifest.get("architecture_batch_order_hash", "")):
        reasons.append("architecture_batch_order_hash_mismatch")
    return sorted(set(reasons))


def math_isfinite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _dataset_identity_reasons(task: Mapping[str, object],
                              manifest: Mapping[str, object]) -> List[str]:
    """Validate the canonical key and the human-readable dataset label.

    S1/S2 bundle metadata stores ``dataset`` as a display label (for example
    ``KuaiRand-Pure``) and ``dataset_key`` as the canonical protocol key
    (``kuairand``).  S3 task identities are canonical keys, so comparing the
    display label byte-for-byte is incorrect; the S2 validator already uses
    this canonicalization contract.
    """
    expected = canonical_dataset(str(task.get("dataset", "")))
    reasons: List[str] = []
    observed_key = manifest.get("dataset_key")
    try:
        if canonical_dataset(str(observed_key)) != expected \
                or str(observed_key) != expected:
            reasons.append("manifest_mismatch:dataset_key")
    except ValueError:
        reasons.append("manifest_mismatch:dataset_key")
    try:
        if canonical_dataset(str(manifest.get("dataset", ""))) != expected:
            reasons.append("manifest_mismatch:dataset")
    except ValueError:
        reasons.append("manifest_mismatch:dataset")
    return sorted(set(reasons))


def _checkpoint_state_numerically_equal(left_path: str, right_path: str) -> Optional[bool]:
    """Compare two serialized checkpoint state dicts, not their zip bytes.

    ``torch.save`` may emit different archive bytes for two saves of the same
    state (e.g. serialization metadata/order), while A1 requires final and
    common-epoch-49 to be the same numerical model state.  Return ``None`` on
    an unavailable/unreadable checkpoint so the caller can fail closed with a
    distinct reason.
    """
    try:
        import torch

        def load(path: str):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")

        left, right = load(left_path), load(right_path)

        def equal(lhs: object, rhs: object) -> bool:
            if torch.is_tensor(lhs) or torch.is_tensor(rhs):
                if not torch.is_tensor(lhs) or not torch.is_tensor(rhs):
                    return False
                return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype \
                    and bool(torch.equal(lhs, rhs))
            if isinstance(lhs, Mapping) or isinstance(rhs, Mapping):
                if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping) \
                        or set(lhs) != set(rhs):
                    return False
                return all(equal(lhs[key], rhs[key]) for key in lhs)
            if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
                if not isinstance(lhs, type(rhs)) or len(lhs) != len(rhs):
                    return False
                return all(equal(a, b) for a, b in zip(lhs, rhs))
            return lhs == rhs

        return bool(equal(left, right))
    except Exception:
        return None


def validate_fresh_run(task: Mapping[str, object], require_formal: bool = True, *,
                       expected_source_inventory: Optional[Mapping[str, str]] = None
                       ) -> Tuple[bool, List[str]]:
    run_dir = os.path.abspath(str(task.get("run_dir", "")))
    reasons: List[str] = []
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        return False, ["run_manifest_missing"]
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return False, ["run_manifest_invalid:{}".format(type(exc).__name__)]
    expected = {
        "stage": STAGE, "status": "OK", "formal_run_key": str(task["formal_run_key"]),
        "setting": str(task["setting"]), "seed": int(task["seed"]),
        "backbone": BACKBONE,
        "objective_weighting": str(task["objective_weighting"]),
        "shared_access": str(task["shared_access"]), "fresh_role": str(task["fresh_role"]),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            reasons.append("manifest_mismatch:{}".format(field))
    reasons.extend(_dataset_identity_reasons(task, manifest))
    if manifest.get("formal_valid") is not True:
        reasons.append("formal_protocol_invalid")
    if int(manifest.get("fixed_final_epoch", -1)) != 49 \
            or list(manifest.get("common_epochs", [])) != [9, 19, 29, 39, 49]:
        reasons.append("checkpoint_contract_mismatch")
    if manifest.get("checkpoint_selection_frozen_before_test") is not True \
            or manifest.get("test_evaluated_during_training") is not False \
            or manifest.get("test_metric_in_epoch_history") is not False:
        reasons.append("test_boundary_invalid")
    source_hashes = manifest.get("code_hashes", {})
    model_hash = source_hashes.get(MODEL_SOURCE_KEY, "") if isinstance(source_hashes, Mapping) else ""
    if not model_hash or str(model_hash) != MODEL_SOURCE_SHA256 \
            or manifest.get("forward_model_source_sha256") != MODEL_SOURCE_SHA256:
        reasons.append("model_source_hash_mismatch")
    if require_formal:
        try:
            expected_sources = (dict(expected_source_inventory)
                                if expected_source_inventory is not None
                                else source_inventory(ROOT))
            for source, source_hash in expected_sources.items():
                if not isinstance(source_hashes, Mapping) or source_hashes.get(source) != source_hash:
                    reasons.append("code_hash_mismatch:{}".format(source))
        except Exception as exc:
            reasons.append("source_inventory_invalid:{}".format(type(exc).__name__))
    reasons.extend(_artifact_hash_reasons(run_dir, manifest))
    # Keep the manifest's byte-level artifact SHA separate from the
    # coefficient table's canonical content hash.  The latter is also used by
    # strict pair comparison after the permitted legacy schema normalization.
    reasons.extend(_coefficient_artifact_reasons(run_dir, manifest))
    nonfinite_path = os.path.join(run_dir, "nonfinite_inventory.json")
    if not os.path.isfile(nonfinite_path):
        reasons.append("nonfinite_inventory_missing")
    else:
        failure = _nonfinite_inventory_failure(nonfinite_path)
        if failure:
            reasons.append(failure)
    reasons.extend(validate_batch_trace(task, run_dir, manifest))
    prediction_path = os.path.join(run_dir, "predictions_test.npz")
    if not os.path.isfile(prediction_path):
        reasons.append("missing artifact predictions_test.npz")
    else:
        try:
            import numpy as np
            with np.load(prediction_path, allow_pickle=False) as payload_file:
                payload = {key: np.asarray(payload_file[key]) for key in payload_file.files}
            test_rows = _read_csv(os.path.join(run_dir, "metrics_test.csv"))[1]
            summary_rows = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))[1]
            reasons.extend(validate_raw_prediction_metrics(payload, manifest, task,
                                                           test_rows, summary_rows))
        except Exception as exc:
            reasons.append("prediction_metric_validation_failed:{}".format(type(exc).__name__))
    checkpoints = manifest.get("checkpoints", {})
    if not isinstance(checkpoints, Mapping):
        reasons.append("checkpoint_hashes_missing")
    else:
        labels = {
            "selected_checkpoint.pth": "selected", "final_checkpoint.pth": "final",
            **{"common_epoch_{}.pth".format(epoch): "common_epoch_{}".format(epoch)
               for epoch in (9, 19, 29, 39, 49)},
        }
        for filename, key in labels.items():
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path) or not checkpoints.get(filename):
                reasons.append("checkpoint_missing:{}".format(filename))
            elif _check_checkpoint_finite(path) is not True:
                reasons.append("checkpoint_nonfinite:{}".format(filename))
        final_path = os.path.join(run_dir, "final_checkpoint.pth")
        common49_path = os.path.join(run_dir, "common_epoch_49.pth")
        state_equal = _checkpoint_state_numerically_equal(final_path, common49_path)
        if state_equal is None:
            reasons.append("final_common49_checkpoint_compare_failed")
        elif not state_equal:
            reasons.append("final_common49_checkpoint_state_mismatch")
    return not reasons, sorted(set(reasons))


def _compare_fresh_full_pair(full_manifest: Mapping[str, object],
                             target_manifest: Mapping[str, object], *,
                             full_run_dir: Optional[str] = None,
                             target_run_dir: Optional[str] = None) -> List[str]:
    reasons = []
    objective_aliases = {
        "equal_domain_task_mean_bce": "equal", "equal": "equal",
        "sample_population_mean_task_bce": "sample", "direct_per_example_task_mean_bce": "sample",
        "sample": "sample",
    }
    def objective_factor(manifest):
        explicit = manifest.get("objective_weighting")
        if explicit not in (None, ""):
            return str(explicit).lower()
        for field in ("objective_name", "objective", "method"):
            candidate = str(manifest.get(field, "")).lower()
            if candidate in objective_aliases:
                return objective_aliases[candidate]
        return ""
    if not objective_factor(full_manifest) or objective_factor(full_manifest) \
            != objective_factor(target_manifest):
        reasons.append("objective_weighting_mismatch")
    full_hashes = full_manifest.get("code_hashes", {})
    target_hashes = target_manifest.get("code_hashes", {})
    full_model = full_hashes.get(MODEL_SOURCE_KEY, "") if isinstance(full_hashes, Mapping) else ""
    target_model = target_hashes.get(MODEL_SOURCE_KEY, "") if isinstance(target_hashes, Mapping) else ""
    if not full_model or not target_model or full_model != target_model:
        reasons.append("model_source_hash_mismatch")
    if full_model and full_model != MODEL_SOURCE_SHA256:
        reasons.append("model_source_hash_mismatch")
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
    for field in fields:
        if _digest(full_manifest.get(field)) != _digest(target_manifest.get(field)):
            reasons.append("pairing_field_mismatch:{}".format(field))
    if full_run_dir is not None and target_run_dir is not None:
        # Do not compare the two objective JSON byte SHAs.  They identify each
        # immutable artifact independently and may differ when a retained
        # legacy table is schema-normalized.  Compare the canonical table
        # content instead; per-run validation above still checks each byte SHA.
        reasons.extend(_coefficient_pair_reasons(
            full_manifest, target_manifest, full_run_dir, target_run_dir))
    else:
        # Dependency-light callers may provide only manifests.  Preserve the
        # old metadata fallback for those synthetic fixtures; real strict
        # validation always supplies both artifact directories.
        if _digest(full_manifest.get("objective_coefficients")) != _digest(
                target_manifest.get("objective_coefficients")):
            if full_manifest.get("objective_weighting") == target_manifest.get(
                    "objective_weighting"):
                reasons.append("pairing_field_mismatch:objective_coefficients")
        left, right = full_manifest.get("coefficient_table_hash"), target_manifest.get(
            "coefficient_table_hash")
        if left not in (None, "") and right not in (None, "") and left != right:
            reasons.append("pairing_field_mismatch:coefficient_table_hash")
    return sorted(set(reasons))


def validate_fresh_reuse_pair(full_task: Mapping[str, object], s1_root: str, s2_root: str,
                              *, target_only_run_dir: Optional[str] = None) -> Tuple[bool, List[str]]:
    full_dir = str(full_task.get("run_dir", ""))
    target_dir = target_only_run_dir or str(full_task.get("target_only_run_dir", ""))
    reasons = []
    if not full_dir or not target_dir:
        return False, ["fresh_full_pair_path_missing"]
    try:
        full_manifest = _read_json(os.path.join(full_dir, "run_manifest.json"))
        target_manifest = _read_json(os.path.join(target_dir, "run_manifest.json"))
    except Exception as exc:
        return False, ["fresh_full_pair_manifest_invalid:{}".format(type(exc).__name__)]
    reasons.extend(_compare_fresh_full_pair(
        full_manifest, target_manifest, full_run_dir=full_dir, target_run_dir=target_dir))
    return not reasons, sorted(set(reasons))


def validate_strict_four_way_block(dataset: str, setting: str, seed: int,
                                   tasks: Sequence[Mapping[str, object]], *,
                                   run_dirs: Mapping[str, str], s1_root: str, s2_root: str) -> Dict[str, object]:
    dataset, setting, seed = str(dataset), str(setting), int(seed)
    all_tasks = list(tasks)
    fresh = {str(task["formal_run_key"]): task for task in all_tasks}
    reuse = build_reuse_matrix(s1_root, s2_root)
    equal_full = next(task for task in reuse if task.get("reuse_role") == "controlled_full_source"
                      and task.get("dataset") == dataset and task.get("setting") == setting
                      and int(task.get("seed")) == seed)
    keys = {
        "sample_full": sample_full_replacement_key(dataset, setting, seed),
        "sample_target_only_shared": fresh_formal_key("sample", dataset, setting, seed),
        "equal_target_only_shared": fresh_formal_key("equal", dataset, setting, seed),
        "equal_full": str(equal_full["formal_run_key"]),
    }
    sources: Dict[str, Mapping[str, object]] = {}
    source_paths: Dict[str, str] = {}
    for label, key in keys.items():
        path = run_dirs.get(key)
        if label == "equal_full":
            path = str(equal_full["source_run_dir"])
        elif path is None:
            raise RuntimeError("strict four-way block missing run path for {}".format(label))
        if label != "equal_full" and key not in fresh:
            raise RuntimeError("strict four-way block references an unapproved fresh key")
        if not path or not os.path.isfile(os.path.join(str(path), "run_manifest.json")):
            raise RuntimeError("strict four-way block missing manifest for {}".format(label))
        source_paths[label] = str(path)
        sources[label] = _read_json(os.path.join(str(path), "run_manifest.json"))
    model_hashes = []
    for manifest in sources.values():
        hashes = manifest.get("code_hashes", {})
        model_hashes.append(str(hashes.get(MODEL_SOURCE_KEY, "")) if isinstance(hashes, Mapping) else "")
    if any(not item for item in model_hashes) or len(set(model_hashes)) != 1 \
            or model_hashes[0] != MODEL_SOURCE_SHA256:
        raise RuntimeError("forward source hash mismatch in strict four-way block")
    for left, right in (("sample_full", "sample_target_only_shared"),
                        ("equal_full", "equal_target_only_shared")):
        reasons = _compare_fresh_full_pair(
            sources[left], sources[right], full_run_dir=source_paths[left],
            target_run_dir=source_paths[right])
        if reasons:
            raise RuntimeError("strict four-way pair invalid: {}".format(reasons))
    for field in _STRICT_FOUR_WAY_FIELDS:
        values = [_digest(sources[label].get(field)) for label in sources]
        if len(set(values)) != 1:
            raise RuntimeError("strict four-way held-constant field mismatch:{}".format(field))
    return {
        "dataset": dataset, "setting": setting, "seed": seed,
        "condition_keys": keys, "forward_model_sha256": model_hashes[0],
        "strict_source_identity": True, "pairing_valid": True,
    }


def _tree_digest(path: str) -> str:
    """Hash an immutable tree by normalized relative path and file contents."""
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError("tree root is missing: {}".format(root))
    entries = []
    for current, directories, files in os.walk(root):
        directories.sort()
        for filename in sorted(files):
            absolute = os.path.join(current, filename)
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            entries.append((relative, _file_hash(absolute)))
    return _digest(entries)


def _amendment_queue_state_digest(queue: Mapping[str, object]) -> str:
    excluded = {"amendment_state_digest", "amendment_history",
                "amendment_history_digest", "updated_at"}
    return _digest({key: value for key, value in queue.items() if key not in excluded})


def _validate_amendment_queue_history(queue: Mapping[str, object]) -> List[str]:
    history = queue.get("amendment_history", [])
    reasons = []
    if not isinstance(history, list):
        reasons.append("amendment_history_invalid")
    elif _digest(history) != queue.get("amendment_history_digest"):
        reasons.append("amendment_history_digest_invalid")
    if queue.get("amendment_state_digest") != _amendment_queue_state_digest(queue):
        reasons.append("amendment_state_digest_invalid")
    return reasons


def _write_amendment_queue_state(path: str, queue: Dict[str, object],
                                 event: str = "update") -> None:
    history = list(queue.get("amendment_history", []))
    history.append({"event": str(event), "time": time.time(),
                    "prior_state_digest": queue.get("amendment_state_digest", "")})
    queue["amendment_history"] = history
    queue["amendment_history_digest"] = _digest(history)
    queue["amendment_state_digest"] = _amendment_queue_state_digest(queue)
    queue["updated_at"] = time.time()
    _write_atomic_json(path, queue)


def _sample_target_pending_keys(tasks: Sequence[Mapping[str, object]]) -> List[str]:
    return sorted(str(task["formal_run_key"]) for task in tasks
                  if task.get("fresh_role") == FRESH_TARGET_ROLE
                  and task.get("objective_weighting") == "sample")


def _refresh_amendment_runtime_state(queue: Dict[str, object]) -> Dict[str, object]:
    records = list(queue.get("records", []))
    accepted = {str(row.get("formal_run_key", "")) for row in records
                if row.get("artifact_valid") is True}
    tasks = list(queue.get("tasks", []))
    plan = list(queue.get("launch_plan", []))
    states = queue.get("launch_group_states", {})
    if not isinstance(states, Mapping):
        states = {}
    complete = sum(str(group.get("launch_group_id", "")) in states
                   and str(states[str(group.get("launch_group_id", ""))]).startswith(
                       ("COMPLETE", "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT",
                        "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED"))
                   for group in plan)
    queue["retained_artifact_count"] = len(accepted)
    queue["remaining_run_count"] = max(0, len(tasks) - len(accepted))
    queue["remaining_launch_group_count"] = max(0, len(plan) - complete)
    queue["expected_run_count"] = EXPECTED_FRESH_RUNS
    queue["expected_launch_group_count"] = EXPECTED_LAUNCH_GROUPS
    queue["expected_seed_list"] = list(S3_SEEDS)
    queue["checkpoint_labels"] = list(CHECKPOINTS)
    queue["maximum_concurrency"] = 2
    return queue


def _amendment_runtime_state(tasks: Sequence[Mapping[str, object]],
                             records: Sequence[Mapping[str, object]],
                             launch_group_states: Mapping[str, object],
                             launch_plan: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    accepted = {str(row.get("formal_run_key", "")) for row in records
                if row.get("artifact_valid") is True}
    completed = sum(str(group.get("launch_group_id", "")) in launch_group_states
                    and str(launch_group_states[str(group.get("launch_group_id", ""))]).startswith(
                        ("COMPLETE", "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT",
                         "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED"))
                    for group in launch_plan)
    return {
        "retained_artifact_count": len(accepted),
        "remaining_run_count": EXPECTED_FRESH_RUNS - len(accepted),
        "remaining_launch_group_count": EXPECTED_LAUNCH_GROUPS - completed,
        "expected_run_count": EXPECTED_FRESH_RUNS,
        "expected_launch_group_count": EXPECTED_LAUNCH_GROUPS,
        "expected_seed_list": list(S3_SEEDS), "checkpoint_labels": list(CHECKPOINTS),
    }


def _group_member_dir(args: argparse.Namespace, group: Mapping[str, object],
                      task: Mapping[str, object], wave: int) -> str:
    root = os.path.abspath(str(getattr(args, "raw_root", os.path.dirname(str(task["run_dir"])))))
    return os.path.join(root, ".stage_S3_attempts", str(group["launch_group_id"]),
                        "wave_{}".format(int(wave)), str(task["worker_id"]))


def _promotion_receipt_path(raw_root: str, group_id: str, wave: int) -> str:
    return os.path.join(os.path.abspath(raw_root), ".stage_S3_promotions",
                        str(group_id), "wave_{}.json".format(int(wave)))


def _atomic_pair_promote(source_dirs: Sequence[str], canonical_dirs: Sequence[str]) -> None:
    if len(source_dirs) != 2 or len(canonical_dirs) != 2:
        raise ValueError("Stage S3 promotion must move exactly two pair members")
    present = [os.path.lexists(path) for path in canonical_dirs]
    if any(present):
        raise RuntimeError("partial canonical promotion; refusing overwrite")
    if any(not os.path.isdir(path) for path in source_dirs):
        raise FileNotFoundError("both validated pair sources must exist before promotion")
    moved = []
    try:
        for source, target in zip(source_dirs, canonical_dirs):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(source, target)
            moved.append((source, target))
    except Exception as exc:
        # Never attempt a speculative rollback across an interrupted filesystem
        # transaction. The next reconciliation detects the partial promotion.
        raise RuntimeError("cannot reconcile two-member canonical promotion") from exc


def _resolve_sample_pending_records(queue: Dict[str, object],
                                    replacement_tasks: Sequence[Mapping[str, object]]) -> None:
    replacements = {str(task.get("source_formal_run_key", "")):
                    str(task.get("formal_run_key", "")) for task in replacement_tasks
                    if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE}
    pending = set(str(key) for key in queue.get("pending_sample_full_replacements", []))
    for row in queue.get("records", []):
        key = str(row.get("formal_run_key", ""))
        if key not in pending:
            continue
        replacement = replacements.get(key)
        if not replacement:
            continue
        row["scientific_pair_state"] = "STRICT_SOURCE_PAIRED"
        row["resolved_by_sample_full_key"] = replacement
        pending.discard(key)
        group_id = str(row.get("launch_group_id", ""))
        if group_id and isinstance(queue.get("launch_group_states"), dict):
            prior = str(queue["launch_group_states"].get(group_id, ""))
            queue["launch_group_states"][group_id] = (
                "RECOVERED_ARCHIVE_SAMPLE_FULL_REPLACEMENT_RESOLVED"
                if prior.startswith("RECOVERED_ARCHIVE") else "COMPLETE_STRICT_SOURCE_PAIRED")
    queue["pending_sample_full_replacements"] = sorted(pending)


def _archive_interrupted_group_wave(group: Mapping[str, object],
                                    tasks: Sequence[Mapping[str, object]],
                                    args: argparse.Namespace, queue: Dict[str, object], wave: int,
                                    *, process_inventory=None) -> str:
    raw_root = os.path.abspath(str(args.raw_root))
    archive = os.path.join(raw_root, ".stage_S3_archives", str(group["launch_group_id"]),
                           "wave_{}".format(int(wave)))
    if os.path.exists(archive):
        raise FileExistsError("append-only archive already exists: {}".format(archive))
    os.makedirs(archive)
    members = []
    for task in tasks:
        source = _group_member_dir(args, group, task, wave)
        destination = os.path.join(archive, str(task["worker_id"]))
        if os.path.exists(source):
            os.replace(source, destination)
        else:
            os.makedirs(destination)
        members.append({"formal_run_key": str(task["formal_run_key"]),
                        "attempt_dir": destination,
                        "tree_digest": _tree_digest(destination)})
        queue.setdefault("attempts", []).append({
            "formal_run_key": str(task["formal_run_key"]), "launch_group_id": group["launch_group_id"],
            "attempt_wave": int(wave), "status": "ABORTED_PARENT",
            "scientific_result": False, "attempt_dir": destination,
        })
    lock_original = os.path.join(archive, "locks_original")
    lock_invalidated = os.path.join(archive, "locks_invalidated")
    os.makedirs(lock_original); os.makedirs(lock_invalidated)
    invalidated = []
    for task in tasks:
        lock = lock_path_for_task(task, raw_root)
        if os.path.isfile(lock):
            lock_digest = _file_hash(lock)
            shutil.copy2(lock, os.path.join(lock_original, os.path.basename(lock)))
            os.replace(lock, os.path.join(lock_invalidated, os.path.basename(lock)))
            invalidated.append({"path": lock, "sha256": lock_digest})
    if process_inventory is not None:
        process_inventory()
    record = {"launch_group_id": str(group["launch_group_id"]),
              "wave": int(wave), "archive_path": archive, "members": members,
              "invalidated_locks": invalidated, "archive_tree_digest": _tree_digest(archive)}
    queue.setdefault("interrupted_group_waves", []).append(record)
    queue.setdefault("archives", []).append(record)
    return archive


def _child_output_failure_markers(output: str) -> List[str]:
    value = str(output).lower()
    reasons = []
    if "out of memory" in value or "oom" in value:
        reasons.append("deterministic_or_capacity_failure")
    if any(token in value for token in ("protocol mismatch", "validator", "partition mismatch",
                                        "source hash mismatch", "does not allow protocol")):
        reasons.append("protocol_or_validator_failure")
    return reasons


def _validate_group_args(tasks: Sequence[Mapping[str, object]],
                         group: Mapping[str, object]) -> None:
    keys = [str(task["formal_run_key"]) for task in tasks]
    if len(keys) != 2 or keys != [str(key) for key in group.get("formal_run_keys", [])]:
        raise ValueError("Stage S3 launch group must contain its two frozen workers")
    if len({str(task.get("launch_group_id", group["launch_group_id"])) for task in tasks}) != 1:
        raise ValueError("Stage S3 launch group identity mismatch")
    if len({str(task.get("worker_id", "")) for task in tasks}) != 2:
        raise ValueError("Stage S3 launch group requires two distinct worker ids")


def _require_queue_execution_authority(args: argparse.Namespace) -> Dict[str, object]:
    """Require the independent A1 reviewer gate; user approval is not a gate."""
    report_path = str(getattr(args, "review_report", os.path.join(
        HERE, "reports", REVIEW_REPORT_NAME)))
    if not os.path.isfile(report_path):
        raise PermissionError("Stage S3 amendment reviewer PASS is required before launch")
    with open(report_path, "r", encoding="utf-8") as stream:
        body = stream.read()
    verdict_rows = [line.strip() for line in body.splitlines()
                    if line.strip().lower().startswith("verdict:")]
    blocker_rows = [line.strip() for line in body.splitlines()
                    if line.strip().lower().startswith("blocking issues:")]
    if len(verdict_rows) != 1 or len(blocker_rows) != 1:
        raise PermissionError("Stage S3 amendment reviewer gate is incomplete or ambiguous")
    verdict, blockers = verdict_rows[0], blocker_rows[0]
    if verdict.lower() not in ("verdict: pass", "verdict: pass with fixes"):
        raise PermissionError("Stage S3 amendment reviewer verdict is not PASS")
    if blockers.lower() != "blocking issues: 0":
        raise PermissionError("Stage S3 amendment reviewer has unresolved blocking issues")
    source_rows = [line.strip().split(":", 1)[1].strip().lower()
                   for line in body.splitlines()
                   if line.strip().lower().startswith(
                       "reviewed source inventory sha-256:")]
    if len(source_rows) > 1 or (source_rows and (
            len(source_rows[0]) != 64
            or any(character not in "0123456789abcdef" for character in source_rows[0]))):
        raise PermissionError("Stage S3 reviewer source-inventory binding is invalid")
    return {"review_report": os.path.abspath(report_path),
            "review_report_sha256": _file_hash(report_path), "verdict": verdict,
            "reviewed_source_inventory_sha256": source_rows[0] if source_rows else ""}


def _apply_reviewed_validator_rebind(queue_path: str, queue: Dict[str, object],
                                     args: argparse.Namespace,
                                     current_source: Mapping[str, str],
                                     authority: Mapping[str, object],
                                     tasks: Sequence[Mapping[str, object]], *,
                                     process_inventory) -> None:
    """Rebind one exact fail-closed queue after the reviewed validator-only repair.

    The seed-2023 pair was archived because two validator contracts were wrong;
    the scientific runner and artifacts were not changed.  This transition is
    deliberately non-generic: any queue, archive, state, source delta, or review
    digest outside the independently reviewed frozen incident stops execution.
    The archived pair remains non-scientific and is rerun together as wave 2.
    """
    if not bool(getattr(args, "allow_reviewed_validator_rebind", False)):
        raise RuntimeError("A1 source inventory drift invalidates resume")
    if process_inventory():
        raise RuntimeError("active Stage S3 process prevents validator-repair rebind")
    if _file_hash(queue_path) != _VALIDATOR_REPAIR_QUEUE_SHA256:
        raise RuntimeError("validator-repair queue SHA-256 mismatch")
    if queue.get("source_inventory_digest") != _VALIDATOR_REPAIR_PRIOR_SOURCE_DIGEST:
        raise RuntimeError("validator-repair prior source digest mismatch")
    prior = queue.get("source_inventory", {})
    if not isinstance(prior, Mapping):
        raise RuntimeError("validator-repair prior source inventory missing")
    changed = {key for key in set(prior) | set(current_source)
               if prior.get(key) != current_source.get(key)}
    if changed != {_VALIDATOR_REPAIR_SOURCE} \
            or prior.get(_VALIDATOR_REPAIR_SOURCE) != _VALIDATOR_REPAIR_PRIOR_ORCHESTRATOR_SHA256:
        raise RuntimeError("validator-repair source delta is not the reviewed one-file change")
    current_digest = source_inventory_digest(current_source)
    if authority.get("reviewed_source_inventory_sha256") != current_digest:
        raise RuntimeError("validator-repair source digest is not bound by the reviewer report")
    expected_states = {
        "s3_kuairand_natural_seed2022": "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT",
        _VALIDATOR_REPAIR_GROUP: "ARCHIVED_WAVE_1",
    }
    if queue.get("status") != "RUNNING" or queue.get("resume_state") != "PLANNED" \
            or int(queue.get("launched", -1)) != 2 \
            or int(queue.get("formal_launches", -1)) != 2 \
            or queue.get("launch_group_states") != expected_states:
        raise RuntimeError("validator-repair queue state is not the frozen failed state")
    expected_record_keys = {
        fresh_formal_key("equal", "kuairand", "natural", 2022),
        fresh_formal_key("sample", "kuairand", "natural", 2022),
    }
    record_keys = {str(row.get("formal_run_key", "")) for row in queue.get("records", [])
                   if row.get("artifact_valid") is True}
    if record_keys != expected_record_keys or len(queue.get("records", [])) != 2:
        raise RuntimeError("validator-repair retained record support mismatch")
    group = next((item for item in queue.get("launch_plan", [])
                  if item.get("launch_group_id") == _VALIDATOR_REPAIR_GROUP), None)
    if not isinstance(group, Mapping):
        raise RuntimeError("validator-repair launch group is missing")
    task_by_key = {str(task["formal_run_key"]): task for task in tasks}
    group_tasks = [task_by_key[str(key)] for key in group.get("formal_run_keys", [])]
    if len(group_tasks) != 2:
        raise RuntimeError("validator-repair launch group membership mismatch")
    archive_root = os.path.join(os.path.abspath(str(args.raw_root)), ".stage_S3_archives",
                                _VALIDATOR_REPAIR_GROUP, "wave_1")
    if _tree_digest(archive_root) != _VALIDATOR_REPAIR_ARCHIVE_SHA256:
        raise RuntimeError("validator-repair failed-pair archive digest mismatch")
    for task, worker in zip(group_tasks, ("worker-1", "worker-2")):
        run_dir = os.path.join(archive_root, worker, "run")
        valid, reasons = validate_fresh_run(dict(task, run_dir=run_dir), require_formal=False)
        if not valid:
            raise RuntimeError("validator-repair archived member is invalid: {}".format(reasons))
    queue["validator_repair_rebind"] = {
        "schema": "stage_S3_A1_validator_rebind_v1",
        "frozen_queue_sha256": _VALIDATOR_REPAIR_QUEUE_SHA256,
        "failed_archive_tree_digest": _VALIDATOR_REPAIR_ARCHIVE_SHA256,
        "old_source_inventory_digest": _VALIDATOR_REPAIR_PRIOR_SOURCE_DIGEST,
        "new_source_inventory_digest": current_digest,
        "changed_source_files": sorted(changed),
        "review_report_sha256": authority.get("review_report_sha256", ""),
        "archived_pair_disposition": "NON_SCIENTIFIC_RERUN_BOTH_FROM_EPOCH_0_AS_WAVE_2",
    }
    queue["source_inventory"] = dict(current_source)
    queue["source_inventory_digest"] = current_digest
    preflight = queue.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("validator-repair preflight binding is missing")
    preflight["review_report_sha256"] = authority.get("review_report_sha256", "")
    _write_amendment_queue_state(queue_path, queue, "reviewed_validator_repair_rebound")


def _artifact_source_inventory(queue: Mapping[str, object]) -> Mapping[str, str]:
    """Return the reviewed source inventory that produced ordinary fresh artifacts."""

    inventory = queue.get("artifact_source_inventory", queue.get("source_inventory", {}))
    if not isinstance(inventory, Mapping) or not inventory:
        raise RuntimeError("Stage S3 artifact source inventory is missing")
    digest = queue.get("artifact_source_inventory_digest")
    if digest not in (None, "") and source_inventory_digest(inventory) != digest:
        raise RuntimeError("Stage S3 artifact source inventory digest mismatch")
    return inventory


def _canonical_tree_rows(tasks: Sequence[Mapping[str, object]],
                         queue: Mapping[str, object]) -> List[Tuple[str, str]]:
    records = queue.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Stage S3 queue records are missing")
    records_by_key = {str(row.get("formal_run_key", "")): row for row in records}
    expected_keys = {str(task["formal_run_key"]) for task in tasks}
    if len(records_by_key) != len(records) or set(records_by_key) != expected_keys:
        raise RuntimeError("coefficient-repair queue record support mismatch")
    rows = []
    for task in tasks:
        key = str(task["formal_run_key"])
        record = records_by_key[key]
        if record.get("artifact_valid") is not True:
            raise RuntimeError("coefficient-repair queue contains an invalid record")
        observed = _tree_digest(str(task["run_dir"]))
        if str(record.get("canonical_tree_digest", "")) != observed:
            raise RuntimeError("coefficient-repair canonical tree digest mismatch: " + key)
        rows.append((key, observed))
    return sorted(rows)


def _apply_reviewed_coefficient_rebind(queue_path: str, queue: Dict[str, object],
                                       args: argparse.Namespace,
                                       current_source: Mapping[str, str],
                                       authority: Mapping[str, object],
                                       tasks: Sequence[Mapping[str, object]], *,
                                       process_inventory) -> None:
    """Bind the completed artifacts to the reviewed coefficient-validator repair.

    The training artifacts retain their producer source inventory.  The current
    inventory identifies only the validator/collector used for final structural
    validation.  This transition cannot promote, rewrite, or retrain a run.
    """

    if not bool(getattr(args, "allow_reviewed_coefficient_rebind", False)):
        raise RuntimeError("A1 source inventory drift invalidates resume")
    if process_inventory():
        raise RuntimeError("active Stage S3 process prevents coefficient-repair rebind")
    if _file_hash(queue_path) != _COEFFICIENT_REPAIR_QUEUE_SHA256:
        raise RuntimeError("coefficient-repair queue SHA-256 mismatch")
    if queue.get("source_inventory_digest") != _COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST:
        raise RuntimeError("coefficient-repair prior source digest mismatch")
    prior = queue.get("source_inventory", {})
    if not isinstance(prior, Mapping) or source_inventory_digest(prior) \
            != _COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST:
        raise RuntimeError("coefficient-repair prior source inventory mismatch")
    changed = {key for key in set(prior) | set(current_source)
               if prior.get(key) != current_source.get(key)}
    if changed != set(_COEFFICIENT_REPAIR_PRIOR_SOURCE_HASHES) or any(
            prior.get(key) != expected
            for key, expected in _COEFFICIENT_REPAIR_PRIOR_SOURCE_HASHES.items()):
        raise RuntimeError("coefficient-repair source delta is not the reviewed change")
    current_digest = source_inventory_digest(current_source)
    if authority.get("reviewed_source_inventory_sha256") != current_digest:
        raise RuntimeError("coefficient-repair source digest is not reviewer-bound")
    if queue.get("status") != "RUNNING" or queue.get("resume_state") != "PLANNED" \
            or int(queue.get("retained_artifact_count", -1)) != EXPECTED_FRESH_RUNS \
            or int(queue.get("remaining_run_count", -1)) != 0 \
            or int(queue.get("remaining_launch_group_count", -1)) != 0 \
            or int(queue.get("launched", -1)) != EXPECTED_FRESH_RUNS \
            or int(queue.get("formal_launches", -1)) != EXPECTED_FRESH_RUNS \
            or queue.get("strict_block_validations") != {}:
        raise RuntimeError("coefficient-repair queue state is not the frozen finalization state")
    tree_rows = _canonical_tree_rows(tasks, queue)
    tree_digest = _digest(tree_rows)
    if tree_digest != _COEFFICIENT_REPAIR_CANONICAL_TREE_DIGEST:
        raise RuntimeError("coefficient-repair combined canonical tree digest mismatch")
    records_by_key = {str(row["formal_run_key"]): row for row in queue["records"]}
    recovered_keys = {key for key, row in records_by_key.items()
                      if row.get("revalidation_receipt_digest")}
    expected_recovered = {
        fresh_formal_key("equal", "kuairand", "natural", 2022),
        fresh_formal_key("sample", "kuairand", "natural", 2022),
    }
    if recovered_keys != expected_recovered:
        raise RuntimeError("coefficient-repair recovered-record support mismatch")
    for task in tasks:
        key = str(task["formal_run_key"])
        if key in recovered_keys:
            from stage_S3_amendment_recovery import _validate_member
            _validate_member(str(task["run_dir"]), task, prior,
                             allow_legacy_empty_object=True, require_formal=False)
            manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
            coefficient_reasons = _coefficient_artifact_reasons(str(task["run_dir"]), manifest)
            if coefficient_reasons:
                raise RuntimeError("recovered coefficient artifact invalid: {}".format(
                    coefficient_reasons))
        else:
            valid, reasons = validate_fresh_run(
                task, require_formal=True, expected_source_inventory=prior)
            if not valid:
                raise RuntimeError("coefficient-repair artifact invalid: {}".format(reasons))
    run_dirs = _strict_block_run_dirs(tasks)
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            result = validate_strict_four_way_block(
                dataset, setting, seed, tasks, run_dirs=run_dirs,
                s1_root=str(args.s1_raw_root), s2_root=str(args.s2_raw_root))
            if result.get("strict_source_identity") is not True \
                    or result.get("pairing_valid") is not True:
                raise RuntimeError("coefficient-repair strict block is invalid")
    queue["artifact_source_inventory"] = dict(prior)
    queue["artifact_source_inventory_digest"] = _COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST
    queue["coefficient_repair_rebind"] = {
        "schema": "stage_S3_A1_coefficient_rebind_v1",
        "frozen_queue_sha256": _COEFFICIENT_REPAIR_QUEUE_SHA256,
        "canonical_tree_count": len(tree_rows),
        "canonical_tree_digest": tree_digest,
        "artifact_source_inventory_digest": _COEFFICIENT_REPAIR_PRIOR_SOURCE_DIGEST,
        "validation_source_inventory_digest": current_digest,
        "changed_source_files": sorted(changed),
        "review_report_sha256": authority.get("review_report_sha256", ""),
        "artifact_disposition": "READ_ONLY_FINAL_VALIDATION_NO_RETRAIN_NO_PROMOTION",
    }
    queue["source_inventory"] = dict(current_source)
    queue["source_inventory_digest"] = current_digest
    preflight = queue.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("coefficient-repair preflight binding is missing")
    preflight["review_report_sha256"] = authority.get("review_report_sha256", "")
    _write_amendment_queue_state(queue_path, queue, "reviewed_coefficient_repair_rebound")


def _new_amendment_queue(tasks: Sequence[Mapping[str, object]],
                         launch_plan: Sequence[Mapping[str, object]], raw_root: str,
                         source_hashes: Mapping[str, str]) -> Dict[str, object]:
    return {
        "stage": STAGE, "amendment": "A1", "queue_schema_version": 3,
        "status": "RUNNING", "resume_state": "PLANNED",
        "raw_root": os.path.abspath(raw_root), "tasks": list(tasks),
        "launch_plan": list(launch_plan), "expected_run_count": EXPECTED_FRESH_RUNS,
        "expected_launch_group_count": EXPECTED_LAUNCH_GROUPS,
        "expected_seed_list": list(S3_SEEDS), "checkpoint_labels": list(CHECKPOINTS),
        "source_inventory": dict(source_hashes),
        "source_inventory_digest": source_inventory_digest(source_hashes),
        "records": [], "attempts": [], "archives": [],
        "launch_group_states": {}, "strict_block_validations": {},
        "pending_sample_full_replacements": _sample_target_pending_keys(tasks),
        "retained_artifact_count": 0, "formal_launches": 0, "launched": 0,
        "maximum_concurrency": 2, "launch_group_count": EXPECTED_LAUNCH_GROUPS,
    }


def _run_group_validation(group: Mapping[str, object], tasks: Sequence[Mapping[str, object]],
                          run_dirs: Optional[Mapping[str, str]] = None,
                          *, require_formal: bool = True) -> Dict[str, object]:
    results = []
    for task in tasks:
        local = dict(task)
        if run_dirs and str(task["formal_run_key"]) in run_dirs:
            local["run_dir"] = run_dirs[str(task["formal_run_key"])]
        valid, reasons = validate_fresh_run(local, require_formal=require_formal)
        results.append({"formal_run_key": str(task["formal_run_key"]),
                        "valid": bool(valid), "reasons": reasons})
    return {"launch_group_id": str(group["launch_group_id"]),
            "pairing_valid": all(row["valid"] for row in results),
            "members": results}


def _record_group_completion(queue: Dict[str, object], group: Mapping[str, object],
                             tasks: Sequence[Mapping[str, object]],
                             validation: Mapping[str, object], wave: int,
                             runtime_snapshot: Mapping[str, object]) -> None:
    group_id = str(group["launch_group_id"])
    members = list(validation.get("members", []))
    if len(members) != len(tasks):
        members = [{"valid": bool(validation.get("pairing_valid", False)),
                    "reasons": []} for _task in tasks]
    for task, member in zip(tasks, members):
        target_only_sample = (task.get("fresh_role") == FRESH_TARGET_ROLE
                              and task.get("objective_weighting") == "sample")
        row = {
            "formal_run_key": str(task["formal_run_key"]),
            "launch_group_id": group_id, "status": "VALID_RETAINED",
            "artifact_valid": bool(member.get("valid", validation.get("pairing_valid"))),
            "scientific_pair_state": ("PENDING_SAMPLE_FULL_REPLACEMENT"
                                      if target_only_sample else "STRICT_SOURCE_PAIRED"),
            "run_dir": str(task["run_dir"]), "attempt_wave": int(wave),
            "launch_runtime_snapshot": dict(runtime_snapshot),
            "canonical_tree_digest": _tree_digest(str(task["run_dir"])),
        }
        if target_only_sample:
            row["resolved_by_sample_full_key"] = ""
        queue.setdefault("records", []).append(row)
    if group.get("group_kind") == "SAMPLE_FULL_RESOURCE_PAIR":
        _resolve_sample_pending_records(queue, tasks)
        queue.setdefault("launch_group_states", {})[group_id] = "COMPLETE"
    elif group.get("group_kind") == "TARGET_ONLY_OBJECTIVE_PAIR":
        has_sample = any(task.get("objective_weighting") == "sample" for task in tasks)
        queue.setdefault("launch_group_states", {})[group_id] = (
            "COMPLETE_PENDING_SAMPLE_FULL_REPLACEMENT" if has_sample
            else "COMPLETE_STRICT_SOURCE_PAIRED")
    queue["launched"] = int(queue.get("launched", 0))
    queue["formal_launches"] = int(queue.get("launched", 0))
    _refresh_amendment_runtime_state(queue)


def _promote_or_reconcile_group(group: Mapping[str, object],
                                group_tasks: Sequence[Mapping[str, object]],
                                all_tasks: Sequence[Mapping[str, object]],
                                args: argparse.Namespace, queue: Dict[str, object], wave: int,
                                *, validation: Optional[Mapping[str, object]] = None,
                                runtime_snapshot: Optional[Mapping[str, object]] = None) -> None:
    _validate_group_args(group_tasks, group)
    group_id = str(group["launch_group_id"])
    attempt_rows = [row for row in queue.get("attempts", [])
                    if row.get("launch_group_id") == group_id
                    and int(row.get("attempt_wave", -1)) == int(wave)]
    if not attempt_rows:
        raise RuntimeError("no complete two-member attempt recorded for promotion")
    order = {str(task["formal_run_key"]): index for index, task in enumerate(group_tasks)}
    attempt_rows.sort(key=lambda row: order.get(str(row.get("formal_run_key", "")), 999))
    source_dirs = [str(row["attempt_run_dir"]) for row in attempt_rows]
    canonical_dirs = [str(task["run_dir"]) for task in group_tasks]
    receipt_path = _promotion_receipt_path(str(args.raw_root), group_id, wave)
    if all(os.path.isdir(path) for path in canonical_dirs):
        receipt = _read_json(receipt_path) if os.path.isfile(receipt_path) else {}
        expected = receipt.get("canonical_tree_digests", [])
        observed = [_tree_digest(path) for path in canonical_dirs]
        if expected and list(expected) != observed:
            raise RuntimeError("canonical pair hash differs from promotion receipt")
        recorded = {str(row.get("formal_run_key", "")) for row in queue.get("records", [])}
        keys = {str(task["formal_run_key"]) for task in group_tasks}
        if keys - recorded:
            checked = dict(validation or _run_group_validation(group, group_tasks))
            if not checked.get("pairing_valid"):
                raise RuntimeError("cannot reconcile promoted pair without valid artifacts")
            _record_group_completion(queue, group, group_tasks, checked, wave,
                                     runtime_snapshot or {})
        _release_group_locks([lock_path_for_task(task, str(args.raw_root))
                              for task in group_tasks])
        return
    if any(os.path.lexists(path) for path in canonical_dirs):
        interrupted = set(queue.get("promotion_interrupted_group_ids", []))
        if group_id in interrupted:
            raise RuntimeError("partial canonical promotion; refusing overwrite")
    checked = dict(validation or _run_group_validation(group, group_tasks, {
        str(row["formal_run_key"]): str(row["attempt_run_dir"]) for row in attempt_rows
    }))
    if not checked.get("pairing_valid"):
        raise RuntimeError("cannot promote invalid two-member launch group")
    try:
        _atomic_pair_promote(source_dirs, canonical_dirs)
    except Exception as exc:
        queue.setdefault("promotion_interrupted_group_ids", []).append(group_id)
        raise RuntimeError("cannot reconcile two-member canonical promotion") from exc
    receipt = {
        "schema": "stage_S3_pair_promotion_v1", "state": "PROMOTED",
        "launch_group_id": group_id, "wave": int(wave),
        "formal_run_keys": [str(task["formal_run_key"]) for task in group_tasks],
        "canonical_tree_digests": [_tree_digest(path) for path in canonical_dirs],
        "attempt_tree_digests": [_tree_digest(path) for path in canonical_dirs],
    }
    _write_atomic_json(receipt_path, receipt)
    _record_group_completion(queue, group, group_tasks, checked, wave,
                             runtime_snapshot or dict(attempt_rows[0].get(
                                 "queue_runtime_snapshot", {})))


def _execute_pair_with_recovery(pair: Mapping[str, object], tasks: Sequence[Mapping[str, object]],
                                args: argparse.Namespace, queue_path: str,
                                base: Dict[str, object], records: List[Dict[str, object]],
                                *, child_runner=None) -> int:
    """Compatibility executor: at most one whole-pair epoch-0 retry."""
    pair_id = str(pair.get("co_run_pair_id", pair.get("launch_group_id", "pair")))
    group = dict(pair, launch_group_id=pair_id,
                 formal_run_keys=[str(task["formal_run_key"]) for task in tasks])
    for task, worker in zip(tasks, ("worker-1", "worker-2")):
        task.setdefault("worker_id", worker)
        task.setdefault("launch_group_id", pair_id)
    launches = 0
    for wave in (1, 2):
        outcomes = []
        for task in tasks:
            attempt_id = "{}-wave{}-{}".format(pair_id, wave, uuid.uuid4().hex[:8])
            attempt_dir = os.path.join(str(args.raw_root), ".stage_S3_attempts", pair_id,
                                       "wave_{}".format(wave), str(task["worker_id"]))
            if os.path.exists(attempt_dir):
                raise FileExistsError("attempt directory already exists: {}".format(attempt_dir))
            if child_runner is None:
                os.makedirs(attempt_dir)
                row = _launch_one_child(task, args, attempt_dir, attempt_id, wave)
            else:
                row = child_runner(task, args, attempt_dir, attempt_id, wave)
            row = dict(row)
            row.setdefault("formal_run_key", task["formal_run_key"])
            row.setdefault("attempt_id", attempt_id); row.setdefault("attempt_wave", wave)
            row.setdefault("attempt_run_dir", os.path.join(attempt_dir, "run"))
            row["launch_group_id"] = pair_id
            row["returncode"] = int(row.get("returncode", 1))
            row["validator_valid"] = bool(row.get("validator_valid", False))
            outcomes.append(row); launches += 1
        base["launched"] = int(base.get("launched", 0)) + 2
        base["attempts"] = list(base.get("attempts", [])) + outcomes
        queue = {"resume_state": "RUNNING"}
        _write_atomic_json(queue_path, queue)
        if all(row["returncode"] == 0 and row["validator_valid"] for row in outcomes):
            source = [str(row["attempt_run_dir"]) for row in outcomes]
            targets = [str(task["run_dir"]) for task in tasks]
            _atomic_pair_promote(source, targets)
            records.extend(outcomes)
            return launches
        diagnostics = []
        for row in outcomes:
            diagnostics.extend(_child_output_failure_markers(str(row.get("stdout", ""))
                                                               + str(row.get("stderr", ""))))
            diagnostics.extend(str(reason) for reason in row.get("validator_reasons", []))
        archive = os.path.join(str(args.raw_root), ".stage_S3_archives", pair_id,
                               "wave_{}".format(wave))
        if os.path.exists(archive):
            raise FileExistsError("append-only archive already exists: {}".format(archive))
        os.makedirs(archive)
        for task, row in zip(tasks, outcomes):
            destination = os.path.join(archive, str(task.get("objective_weighting", "member")))
            source = os.path.dirname(str(row["attempt_run_dir"]))
            if os.path.exists(source):
                os.replace(source, destination)
        base.setdefault("archives", []).append(archive)
        fatal_markers = {"deterministic_or_capacity_failure", "protocol_or_validator_failure"}
        fatal_reasons = any(reason in fatal_markers for reason in diagnostics) or any(
            reason != "child_exit:1" and not str(reason).startswith("child_exit:")
            for reason in diagnostics)
        if fatal_reasons or wave == 2:
            raise RuntimeError("Stage S3 launch group failed: {}".format(sorted(set(diagnostics))))
    raise RuntimeError("Stage S3 pair recovery exhausted")


def _launch_one_child(task: Mapping[str, object], args: argparse.Namespace,
                      attempt_dir: str, attempt_id: str, wave: int) -> Dict[str, object]:
    member_dir = attempt_dir
    if not os.path.isdir(member_dir):
        os.makedirs(member_dir, exist_ok=False)
    run_dir = os.path.join(attempt_dir, "run")
    command = command_for(task, args, run_dir=run_dir, attempt_id=attempt_id)
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=False)
    pid = int(process.pid)
    stdout, stderr = process.communicate()
    ended = time.time()
    with open(os.path.join(member_dir, "launcher_stdout.log"), "w", encoding="utf-8") as stream:
        stream.write(stdout)
    with open(os.path.join(member_dir, "launcher_stderr.log"), "w", encoding="utf-8") as stream:
        stream.write(stderr)
    valid, reasons = validate_fresh_run(dict(task, run_dir=run_dir), require_formal=True) \
        if process.returncode == 0 else (False, ["child_exit:{}".format(process.returncode)])
    reasons.extend(_child_output_failure_markers(stdout + "\n" + stderr))
    peak_memory = ""
    if os.path.isfile(os.path.join(run_dir, "run_manifest.json")):
        try:
            runtime = _read_json(os.path.join(run_dir, "run_manifest.json")).get("runtime", {})
            if isinstance(runtime, Mapping):
                peak_memory = runtime.get("max_memory_allocated_bytes", "")
        except Exception:
            pass
    return {
        "formal_run_key": str(task["formal_run_key"]),
        "launch_group_id": str(task["launch_group_id"]),
        "co_run_pair_id": str(task["co_run_pair_id"]),
        "planned_peer_run_key": str(task["planned_peer_run_key"]),
        "worker_id": str(task["worker_id"]), "attempt_id": str(attempt_id),
        "attempt_wave": int(wave), "attempt_dir": member_dir,
        "attempt_run_dir": run_dir, "command": command, "pid": pid,
        "start_time": started, "end_time": ended,
        "returncode": int(process.returncode), "validator_valid": bool(valid),
        "validator_reasons": sorted(set(reasons)), "device": str(getattr(args, "device", "cuda:0")),
        "peak_memory_bytes": peak_memory, "stdout": stdout, "stderr": stderr,
    }


def concurrent_resource_smoke(*, python: Optional[str] = None,
                              device: str = "cuda:0") -> Dict[str, object]:
    """Run two concurrent access smoke children; this is not a formal training run."""
    executable = str(python or sys.executable)
    jobs = [("equal", ACCESS), ("sample", ACCESS)]
    commands = [[executable, os.path.join(HERE, "stage_S3_run.py"),
                 "--run_type", "joint", "--dataset", "kuairand", "--setting", "natural",
                 "--seed", "2022", "--run_dir", os.path.join(tempfile.gettempdir(),
                                                                "s3_smoke_{}".format(i)),
                 "--device", str(device), "--objective_weighting", objective,
                 "--shared_access", access, "--fresh_role", FRESH_TARGET_ROLE,
                 "--formal_run_key", fresh_formal_key(objective, "kuairand", "natural", 2022),
                 "--source_formal_run_key", "smoke", "--co_run_pair_id", "s3_smoke",
                 "--planned_peer_run_key", "smoke_peer", "--worker_id", "worker-{}".format(i+1),
                 "--attempt_id", "smoke", "--smoke_only", "--smoke_representative"]
                for i, (objective, access) in enumerate(jobs)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(subprocess.run, command, cwd=ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                   for command in commands]
        results = [future.result() for future in futures]
    members = [{"returncode": int(result.returncode), "stdout": result.stdout,
                "stderr": result.stderr} for result in results]
    return {"passed": all(row["returncode"] == 0 for row in members),
            "max_concurrency": 2, "members": members}


def _execute_amendment_group(group: Mapping[str, object],
                             group_tasks: Sequence[Mapping[str, object]],
                             all_tasks: Sequence[Mapping[str, object]],
                             args: argparse.Namespace, queue_path: str,
                             queue: Dict[str, object], *, child_runner=None,
                             process_inventory=None) -> int:
    _require_queue_execution_authority(args)
    _validate_group_args(group_tasks, group)
    if int(getattr(args, "max_workers", 2)) != 2:
        raise ValueError("Stage S3 formal concurrency is frozen at exactly two workers")
    if process_inventory is not None and process_inventory():
        raise RuntimeError("unaccounted Stage S3 child process detected before launch")
    total_launches = 0
    used_waves = [int(row.get("attempt_wave", 0)) for row in queue.get("attempts", [])
                  if row.get("launch_group_id") == str(group["launch_group_id"])]
    first_wave = max(used_waves, default=0) + 1
    if first_wave > 2:
        raise RuntimeError("Stage S3 launch group exhausted its single retry")
    for wave in range(first_wave, 3):
        attempt_id = "{}-wave{}-{}".format(group["launch_group_id"], wave, uuid.uuid4().hex[:8])
        attempt_dirs = [_group_member_dir(args, group, task, wave) for task in group_tasks]
        if any(os.path.exists(path) for path in attempt_dirs):
            raise FileExistsError("append-only Stage S3 attempt directory collision")
        lock_paths = _acquire_group_locks(group_tasks, str(args.raw_root),
                                          str(group["launch_group_id"]), wave, attempt_id)

        def run_one(task, attempt_dir):
            if child_runner is not None:
                return child_runner(task, args, attempt_dir, attempt_id, wave)
            return _launch_one_child(task, args, attempt_dir, attempt_id, wave)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_one, task, attempt_dir)
                       for task, attempt_dir in zip(group_tasks, attempt_dirs)]
            rows = [dict(future.result()) for future in futures]
        total_launches += 2
        queue["launched"] = int(queue.get("launched", 0)) + 2
        queue["formal_launches"] = int(queue.get("launched", 0))
        snapshot = _amendment_runtime_state(all_tasks, queue.get("records", []),
                                           queue.get("launch_group_states", {}),
                                           queue.get("launch_plan", []))
        for task, row, attempt_dir in zip(group_tasks, rows, attempt_dirs):
            row.setdefault("formal_run_key", str(task["formal_run_key"]))
            row.setdefault("attempt_id", attempt_id)
            row.setdefault("attempt_wave", wave)
            row.setdefault("launch_group_id", str(group["launch_group_id"]))
            row.setdefault("attempt_run_dir", os.path.join(attempt_dir, "run"))
            row.setdefault("returncode", 1)
            row.setdefault("validator_valid", False)
            row.setdefault("validator_reasons", [])
            row["queue_runtime_snapshot"] = snapshot
            row["status"] = "CHILD_OK" if int(row["returncode"]) == 0 else "CHILD_FAILED"
        queue.setdefault("attempts", []).extend(rows)
        _write_amendment_queue_state(queue_path, queue, "two_process_group_wave_finished")
        valid = all(int(row["returncode"]) == 0 and bool(row["validator_valid"])
                    for row in rows)
        reasons = []
        fatal = False
        if valid:
            run_dirs = {str(row["formal_run_key"]): str(row["attempt_run_dir"]) for row in rows}
            validation = _run_group_validation(group, group_tasks, run_dirs)
            if not validation.get("pairing_valid"):
                reasons = [reason for member in validation.get("members", [])
                           for reason in member.get("reasons", [])]
                fatal = True
            else:
                _promote_or_reconcile_group(group, group_tasks, all_tasks, args, queue, wave,
                                            validation=validation, runtime_snapshot=snapshot)
                _write_amendment_queue_state(queue_path, queue, "pair_promoted_and_recorded")
                _release_group_locks(lock_paths)
                return total_launches
        else:
            for row in rows:
                reasons.extend(str(value) for value in row.get("validator_reasons", []))
                reasons.extend(_child_output_failure_markers(
                    str(row.get("stdout", "")) + "\n" + str(row.get("stderr", ""))))
            fatal = any(not str(reason).startswith("child_exit:")
                        for row in rows for reason in row.get("validator_reasons", []))
        _archive_interrupted_group_wave(group, group_tasks, args, queue, wave,
                                        process_inventory=process_inventory)
        queue.setdefault("launch_group_states", {})[str(group["launch_group_id"])] = \
            "ARCHIVED_WAVE_{}".format(wave)
        _write_amendment_queue_state(queue_path, queue, "failed_group_wave_archived")
        if fatal or "deterministic_or_capacity_failure" in reasons \
                or "protocol_or_validator_failure" in reasons or wave == 2:
            raise RuntimeError("Stage S3 launch group stopped after failure: {}".format(
                sorted(set(reasons))))
    return total_launches


def _reconcile_amendment_queue(tasks: Sequence[Mapping[str, object]],
                               args: argparse.Namespace, queue_path: str,
                               queue: Dict[str, object], *, process_inventory=None) -> Dict[str, object]:
    plan = list(queue.get("launch_plan", []))
    artifact_sources = (_artifact_source_inventory(queue)
                        if queue.get("artifact_source_inventory")
                        or queue.get("source_inventory") else None)
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    completed_keys = {str(row.get("formal_run_key", "")) for row in queue.get("records", [])
                      if row.get("artifact_valid") is True}
    first_pending = None
    for group in plan:
        group_id = str(group["launch_group_id"])
        group_tasks = [by_key[str(key)] for key in group["formal_run_keys"]]
        canonical_presence = [os.path.lexists(str(task["run_dir"])) for task in group_tasks]
        if any(canonical_presence) and not all(os.path.isdir(str(task["run_dir"]))
                                                for task in group_tasks):
            raise RuntimeError("partial canonical promotion; refusing overwrite")
        state = queue.get("launch_group_states", {}).get(group_id)
        if state and all(str(key) in completed_keys for key in group["formal_run_keys"]):
            if not all(os.path.isdir(str(task["run_dir"])) for task in group_tasks):
                raise RuntimeError("accepted Stage S3 canonical artifact disappeared")
            records = [row for row in queue.get("records", [])
                       if str(row.get("formal_run_key", "")) in {
                           str(task["formal_run_key"]) for task in group_tasks}]
            if len(records) != 2 or len({str(row.get("formal_run_key", ""))
                                         for row in records}) != 2:
                raise RuntimeError("accepted Stage S3 queue records are missing or duplicated")
            records_by_key = {str(row["formal_run_key"]): row for row in records}
            for task in group_tasks:
                key = str(task["formal_run_key"])
                record = records_by_key[key]
                run_dir = str(task["run_dir"])
                expected_tree = str(record.get("canonical_tree_digest", ""))
                if not expected_tree or _tree_digest(run_dir) != expected_tree:
                    raise RuntimeError("accepted Stage S3 canonical tree digest changed: " + key)
                if record.get("revalidation_receipt_digest"):
                    from stage_S3_amendment_recovery import _validate_member
                    _validate_member(
                        run_dir, task, source_inventory(ROOT),
                        allow_legacy_empty_object=True, require_formal=False)
                else:
                    valid, reasons = validate_fresh_run(
                        task, require_formal=True,
                        expected_source_inventory=artifact_sources)
                    if not valid:
                        raise RuntimeError("accepted Stage S3 artifact failed resume validation: {}"
                                           .format(reasons))
            _release_group_locks([lock_path_for_task(task, str(args.raw_root))
                                  for task in group_tasks])
            continue
        attempt_waves = sorted({int(row.get("attempt_wave", 0)) for row in queue.get("attempts", [])
                                if row.get("launch_group_id") == group_id})
        promoted = False
        for wave in attempt_waves:
            rows = [row for row in queue.get("attempts", [])
                    if row.get("launch_group_id") == group_id
                    and int(row.get("attempt_wave", 0)) == wave]
            if len(rows) != 2 or not all(int(row.get("returncode", 1)) == 0
                                         and row.get("validator_valid") for row in rows):
                continue
            run_dirs = {str(row["formal_run_key"]): str(row["attempt_run_dir"]) for row in rows}
            precheck = _run_group_validation(group, group_tasks, run_dirs)
            if not precheck.get("pairing_valid"):
                continue
            _promote_or_reconcile_group(group, group_tasks, tasks, args, queue, wave)
            promoted = True
            completed_keys.update(str(key) for key in group["formal_run_keys"])
            break
        if not promoted:
            if process_inventory is not None and process_inventory():
                raise RuntimeError("active/unaccounted child process; refusing queue reconciliation")
            first_pending = group if first_pending is None else first_pending
    completed_replacements = [task for task in tasks
                              if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE
                              and str(task.get("formal_run_key")) in completed_keys]
    _resolve_sample_pending_records(queue, completed_replacements)
    _refresh_amendment_runtime_state(queue)
    completed_groups = sum(all(str(key) in completed_keys for key in group["formal_run_keys"])
                           for group in plan)
    return {"first_pending_group": first_pending,
            "completed_group_count": completed_groups,
            "remaining_run_count": queue.get("remaining_run_count"),
            "remaining_launch_group_count": queue.get("remaining_launch_group_count")}


def resume_parent_abort(*_args, **_kwargs):
    raise RuntimeError("pre-amendment parent-abort recovery is disabled")


def recover_validator_bug(*_args, **_kwargs):
    raise RuntimeError("pre-amendment validator recovery is disabled")


def _amendment_queue_template(tasks: Sequence[Mapping[str, object]],
                             args: argparse.Namespace,
                             source_hashes: Mapping[str, str],
                             reuse_audit: Mapping[str, object],
                             preflight: Mapping[str, object],
                             authority: Mapping[str, object]) -> Dict[str, object]:
    plan = build_launch_plan(tasks)
    queue = _new_amendment_queue(tasks, plan, str(args.raw_root), source_hashes)
    queue["preflight"] = {
        "reuse_audit_digest": reuse_audit.get("audit_digest", _digest(reuse_audit)),
        "preflight_digest": _digest(preflight),
        "review_report_sha256": authority.get("review_report_sha256", ""),
    }
    return queue


def _strict_block_run_dirs(tasks: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    return {str(task["formal_run_key"]): str(task["run_dir"]) for task in tasks}


def _finalize_strict_blocks(queue: Dict[str, object], tasks: Sequence[Mapping[str, object]],
                            args: argparse.Namespace) -> None:
    run_dirs = _strict_block_run_dirs(tasks)
    by_block = {}
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            result = validate_strict_four_way_block(
                dataset, setting, seed, tasks, run_dirs=run_dirs,
                s1_root=str(args.s1_raw_root), s2_root=str(args.s2_raw_root))
            by_block["{}|{}|{}".format(dataset, setting, seed)] = result
    queue["strict_block_validations"] = by_block
    if any(not (row.get("strict_source_identity") is True and row.get("pairing_valid") is True)
           for row in by_block.values()):
        raise RuntimeError("one or more Stage S3 strict four-way blocks failed")
    pending = set(str(key) for key in queue.get("pending_sample_full_replacements", []))
    queue["pending_sample_full_replacements"] = []
    for record in queue.get("records", []):
        key = str(record.get("formal_run_key", ""))
        if key in pending:
            task = next((task for task in tasks if str(task["formal_run_key"]) == key), None)
            if task is None:
                raise RuntimeError("pending Sample target key is not in frozen task matrix")
            record["scientific_pair_state"] = "STRICT_SOURCE_PAIRED"
            record["resolved_by_sample_full_key"] = str(task["source_formal_run_key"])
    for group in queue.get("launch_plan", []):
        if group.get("group_kind") == "TARGET_ONLY_OBJECTIVE_PAIR":
            queue.setdefault("launch_group_states", {})[str(group["launch_group_id"])] = \
                "COMPLETE_STRICT_SOURCE_PAIRED"
    queue["status"] = "COMPLETE"
    queue["resume_state"] = "COMPLETE"
    _refresh_amendment_runtime_state(queue)


def _existing_noncanonical_state(raw_root: str) -> List[str]:
    paths = [os.path.join(os.path.abspath(raw_root), name) for name in (
        ".stage_S3_archives", ".stage_S3_attempts", ".stage_S3_locks")]
    return [path for path in paths if os.path.isdir(path)
            and any(os.scandir(path))]


def _active_s3_processes() -> List[Dict[str, object]]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for the frozen S3 active-process audit") from exc
    active = []
    for process in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            info = process.info
            pid = int(info.get("pid", -1))
            if pid == os.getpid():
                continue
            command = " ".join(info.get("cmdline") or [])
            lowered = command.lower()
            if "stage_s3_run.py" in lowered or "stage_s3_orchestrator.py" in lowered:
                active.append({"pid": pid, "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
    return active


def _retain_revalidated_wave2(queue: Dict[str, object],
                              tasks: Sequence[Mapping[str, object]],
                              receipt: Mapping[str, object],
                              promotion: Mapping[str, object]) -> None:
    keys = list(promotion.get("canonical_run_keys", []))
    digests = list(promotion.get("canonical_tree_digests", []))
    if len(keys) != 2 or len(digests) != 2:
        raise RuntimeError("wave-2 promotion receipt is incomplete")
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    group = next((group for group in queue.get("launch_plan", [])
                  if all(str(key) in group.get("formal_run_keys", []) for key in keys)), None)
    if group is None:
        raise RuntimeError("revalidated wave-2 members do not match one frozen launch group")
    for key, tree_digest in zip(keys, digests):
        task = by_key[key]
        queue.setdefault("records", []).append({
            "formal_run_key": key, "source_formal_run_key": str(task.get(
                "source_formal_run_key", "")),
            "launch_group_id": str(group["launch_group_id"]),
            "group_kind": str(group.get("group_kind", "")),
            "status": "VALID_RETAINED", "artifact_valid": True,
            "scientific_pair_state": (
                "PENDING_SAMPLE_FULL_REPLACEMENT"
                if task.get("objective_weighting") == "sample"
                else "STRICT_SOURCE_PAIRED"),
            "resolved_by_sample_full_key": "",
            "run_dir": str(task["run_dir"]), "attempt_wave": 2,
            "source_archive_tree_digest": receipt.get("archive_tree_digest", ""),
            "revalidation_receipt_digest": receipt.get("receipt_digest", ""),
            "canonical_tree_digest": str(tree_digest),
            "recovery_promotion_digest": promotion.get("promotion_digest", ""),
        })
    queue.setdefault("launch_group_states", {})[str(group["launch_group_id"])] = \
        "RECOVERED_ARCHIVE_PENDING_SAMPLE_FULL_REPLACEMENT"
    queue["archive_recovery_state"] = "WAVE2_PAIR_RETAINED_SAMPLE_PENDING_FULL"
    queue["archive_revalidation_receipt"] = dict(receipt)
    queue["archive_promotion_receipt"] = dict(promotion)
    _refresh_amendment_runtime_state(queue)


def run_formal_queue(args: argparse.Namespace) -> Dict[str, object]:
    """Run the frozen A1 queue after review; this function is not used by the coder."""
    authority = _require_queue_execution_authority(args)
    if int(getattr(args, "max_workers", 2)) != 2:
        raise ValueError("Stage S3 concurrency is fixed at exactly two child processes")
    raw_root = os.path.abspath(str(args.raw_root))
    queue_path = os.path.abspath(str(args.queue_manifest))
    tasks = annotate_pairs(build_fresh_matrix(raw_root), build_launch_plan(build_fresh_matrix(raw_root)))
    plan = build_launch_plan(tasks)
    process_inventory = getattr(args, "process_inventory", None) or _active_s3_processes
    source_hashes = source_inventory(ROOT)
    model_path = os.path.join(ROOT, *MODEL_SOURCE_KEY.split("/"))
    if _file_hash(model_path) != MODEL_SOURCE_SHA256:
        raise RuntimeError("reviewed M3OE forward source hash drift; no automatic Equal retrain")
    if os.path.isfile(queue_path):
        queue = _read_json(queue_path)
        if queue.get("amendment") != "A1":
            raise RuntimeError("existing queue is not the amended A1 queue; preserve and stop")
        failures = _validate_amendment_queue_history(queue)
        if failures:
            raise RuntimeError("A1 queue integrity failed: {}".format(failures))
        if queue.get("source_inventory_digest") != source_inventory_digest(source_hashes):
            if bool(getattr(args, "allow_reviewed_coefficient_rebind", False)):
                _apply_reviewed_coefficient_rebind(
                    queue_path, queue, args, source_hashes, authority, tasks,
                    process_inventory=process_inventory)
            else:
                _apply_reviewed_validator_rebind(
                    queue_path, queue, args, source_hashes, authority, tasks,
                    process_inventory=process_inventory)
        expected_queue_identity = {
            "raw_root": raw_root,
            "tasks": tasks,
            "launch_plan": plan,
            "expected_run_count": EXPECTED_FRESH_RUNS,
            "expected_launch_group_count": EXPECTED_LAUNCH_GROUPS,
            "expected_seed_list": list(S3_SEEDS),
            "checkpoint_labels": list(CHECKPOINTS),
            "maximum_concurrency": 2,
        }
        for field, expected in expected_queue_identity.items():
            if _digest(queue.get(field)) != _digest(expected):
                raise RuntimeError("A1 queue frozen identity mismatch: {}".format(field))
        preflight = queue.get("preflight", {})
        if not isinstance(preflight, Mapping) or preflight.get(
                "review_report_sha256") != authority.get("review_report_sha256"):
            raise RuntimeError("A1 queue reviewer authorization hash mismatch")
    else:
        prior_state = _existing_noncanonical_state(raw_root)
        if prior_state:
            legacy_queue = str(getattr(args, "legacy_queue_manifest", ""))
            if not legacy_queue or not os.path.isfile(legacy_queue):
                raise FileNotFoundError(
                    "archived/attempt state exists but its historical queue is missing; "
                    "refusing to infer task identity or create a replacement queue")
            from stage_S3_amendment_recovery import revalidate_wave2_archive
            receipt = revalidate_wave2_archive(
                queue_path=legacy_queue, raw_root=raw_root, tasks=tasks,
                current_source_root=ROOT, process_inventory=process_inventory)
        else:
            receipt = None
        audit = audit_reuse_tasks(s1_raw_root=str(args.s1_raw_root),
                                  s2_raw_root=str(args.s2_raw_root), require_formal=True)
        write_amendment_reuse_audit(audit, output_root=HERE)
        if not audit.get("valid"):
            raise RuntimeError("historical source audit failed; formal queue must not launch")
        preflight = {"wave2_revalidation_receipt": receipt,
                     "source_inventory_digest": source_inventory_digest(source_hashes),
                     "launch_plan_digest": co_run_plan_digest(plan)}
        smoke = concurrent_resource_smoke(python=getattr(args, "python", None),
                                          device=str(getattr(args, "device", "cuda:0")))
        if not smoke.get("passed"):
            raise RuntimeError("reviewed two-process resource smoke failed")
        queue = _amendment_queue_template(tasks, args, source_hashes, audit, preflight, authority)
        if receipt:
            from stage_S3_amendment_recovery import promote_wave2_archive
            promotion = promote_wave2_archive(receipt=receipt, tasks=tasks, raw_root=raw_root)
            _retain_revalidated_wave2(queue, tasks, receipt, promotion)
        initial = _amendment_queue_state_digest(queue)
        queue["amendment_initial_state_digest"] = initial
        queue["amendment_state_digest"] = initial
        queue["amendment_history"] = []
        queue["amendment_history_digest"] = _digest([])
        _write_amendment_queue_state(queue_path, queue, "A1_queue_created_after_preflight")

    with _parent_queue_guard(queue_path):
        reconciliation = _reconcile_amendment_queue(
            tasks, args, queue_path, queue,
            process_inventory=process_inventory)
        _write_amendment_queue_state(queue_path, queue, "queue_reconciled")
        task_by_key = {str(task["formal_run_key"]): task for task in tasks}
        completed = {str(row.get("formal_run_key", "")) for row in queue.get("records", [])
                     if row.get("artifact_valid") is True}
        total_launches = int(queue.get("launched", 0))
        for group in plan:
            keys = [str(key) for key in group["formal_run_keys"]]
            if all(key in completed for key in keys):
                continue
            group_tasks = [task_by_key[key] for key in keys]
            total_launches += _execute_amendment_group(
                group, group_tasks, tasks, args, queue_path, queue,
                process_inventory=process_inventory)
            completed.update(keys)
        if total_launches != int(queue.get("launched", 0)):
            queue["launched"] = total_launches
            queue["formal_launches"] = total_launches
        _finalize_strict_blocks(queue, tasks, args)
        _write_amendment_queue_state(queue_path, queue, "all_strict_blocks_validated")
    return {"queue_path": queue_path, "status": queue.get("status"),
            "launch_groups": len(plan), "formal_launches": queue.get("formal_launches"),
            "strict_blocks": len(queue.get("strict_block_validations", {})),
            "reconciliation": reconciliation}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S3 A1 queue orchestrator")
    parser.add_argument("--execute", action="store_true",
                        help="required flag for formal queue execution after independent review")
    parser.add_argument("--raw-root", default=os.path.join(HERE, "stage_S3_raw_results"))
    parser.add_argument("--s1-raw-root", default=os.path.join(HERE, "stage_S1_raw_results"))
    parser.add_argument("--s2-raw-root", default=os.path.join(HERE, "stage_S2_raw_results"))
    parser.add_argument("--queue-manifest", default=os.path.join(HERE, "stage_S3_queue_manifest_A1.json"))
    parser.add_argument("--legacy-queue-manifest", default="")
    parser.add_argument("--review-report", default=os.path.join(HERE, "reports", REVIEW_REPORT_NAME))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--allow-reviewed-validator-rebind", action="store_true",
                        help="allow only the frozen, reviewer-bound seed-2023 validator repair")
    parser.add_argument("--allow-reviewed-coefficient-rebind", action="store_true",
                        help="allow only the frozen, reviewer-bound coefficient repair")
    parser.add_argument("--dataset-path", default="")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if not args.execute:
        raise SystemExit("formal queue is not launched without --execute")
    result = run_formal_queue(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
