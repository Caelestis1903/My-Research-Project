"""Resume-safe formal matrix launcher for Stage E1.

The matrix is intentionally limited to the registered E1 factors:
``2 weighting arms x 3 role rotations x 4 seeds = 24 runs``.  This module
only launches runs, records process/file status, and never computes or prints
scientific conclusions.  Use ``--dry_run`` to inspect the exact command
queue without starting a process.
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

try:  # direct script execution puts ``experiments`` on sys.path
    from stage_E1_instrumentation import (  # noqa: E402
        FORMAL_PROTOCOL,
        compare_pair_runs,
        ensure_required_fields,
        write_json,
    )
except ModuleNotFoundError:  # namespace-package imports used by tests/tools
    from experiments.stage_E1_instrumentation import (  # noqa: E402
        FORMAL_PROTOCOL,
        compare_pair_runs,
        ensure_required_fields,
        write_json,
    )


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RUNNER = os.path.join(BASE, "stage_E1_run.py")
DEFAULT_RAW_ROOT = os.path.join(BASE, "stage_E1_raw_results")
DEFAULT_MASTER = os.path.join(DEFAULT_RAW_ROOT, "stage_E1_runs.csv")
DEFAULT_LOG_ROOT = os.path.join(BASE, "logs")
DEFAULT_LOCK = os.path.join(DEFAULT_LOG_ROOT, "stage_E1_orchestrator.lock")
DEFAULT_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable

WEIGHTINGS = ("equal", "sample")
SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
ROLE_SIZES = "150000,34500,3000"

MASTER_FIELDS = [
    "stage", "weighting", "setting", "seed", "status", "return_code",
    "run_dir", "stdout_log", "wall_seconds", "best_epoch", "stopped_epoch",
    "observed_epoch_count", "gradient_step_count", "gradient_epoch_count",
    "routing_row_count", "pair_status", "pair_valid", "pair_max_abs_loss_diff",
    "note",
]

REQUIRED_ARTIFACTS = [
    "config.json", "stdout.log", "run_manifest.json", "metrics_epoch.csv",
    "gradient_step.csv", "gradient_epoch.csv", "gradient_cosine_epoch.csv",
    "gate_epoch.csv", "batch_trace.csv", "metrics_summary.csv",
    "nonfinite_inventory.json",
]

REQUIRED_HEADERS = {
    "metrics_epoch.csv": ("seed", "epoch", "domain", "task", "cell"),
    "gradient_step.csv": ("epoch", "step", "domain", "task", "cell",
                           "parameter_group", "domain_signed_contribution"),
    "gradient_epoch.csv": ("seed", "epoch", "domain", "task", "cell",
                            "parameter_group"),
    "gradient_cosine_epoch.csv": ("seed", "epoch", "domain_pair"),
    "gate_epoch.csv": ("seed", "epoch", "domain", "task", "cell",
                        "routing_scope", "routing_sample_count"),
    "batch_trace.csv": ("seed", "epoch", "step", "kind", "batch_fingerprint",
                         "train_loss"),
    "metrics_summary.csv": ("seed", "weighting", "setting", "domain", "task",
                             "cell"),
}


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    """Return the fixed E1 queue in weighting -> setting -> seed order."""

    tasks: List[Dict[str, object]] = []
    for weighting in WEIGHTINGS:
        for setting in SETTINGS:
            for seed in SEEDS:
                run_dir = os.path.join(raw_root, weighting, setting, f"seed{seed}")
                tasks.append({
                    "stage": "E1", "weighting": weighting, "setting": setting,
                    "seed": seed, "run_dir": run_dir,
                    "stdout_log": os.path.join(run_dir, "stdout.log"),
                })
    return tasks


def _read_master(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(weighting: object, setting: object, seed: object) -> Tuple[str, str, int]:
    return (str(weighting), str(setting), int(seed))


def _task_key(task: Mapping[str, object]) -> Tuple[str, str, int]:
    return _key(task.get("weighting"), task.get("setting"), task.get("seed", 0))


def _counterpart_task(task: Mapping[str, object]) -> Dict[str, object]:
    counterpart_weighting = "sample" if str(task["weighting"]) == "equal" else "equal"
    raw_root = os.path.dirname(os.path.dirname(os.path.dirname(str(task["run_dir"]))))
    run_dir = os.path.join(raw_root, counterpart_weighting, str(task["setting"]),
                           "seed{}".format(int(task["seed"])))
    return {
        "stage": "E1", "weighting": counterpart_weighting,
        "setting": task["setting"], "seed": int(task["seed"]),
        "run_dir": run_dir, "stdout_log": os.path.join(run_dir, "stdout.log"),
    }


def validate_run_artifacts(task: Mapping[str, object], *, formal: bool = True) -> Tuple[bool, List[str]]:
    """Revalidate a run before resume or finalization treats it as complete.

    ``formal=True`` is the default for the registered E1 matrix.  A short
    smoke invocation can still be inspected by callers with ``formal=False``;
    it is never eligible for ``done_set`` or a formal master-row ``OK``.
    """

    run_dir = str(task.get("run_dir", ""))
    reasons: List[str] = []
    if not run_dir or not os.path.isdir(run_dir):
        return False, ["run_dir_missing"]
    missing = [name for name in REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    reasons.extend("missing:{}".format(name) for name in missing)
    manifest = _last_manifest(run_dir)
    if not manifest:
        reasons.append("manifest_unreadable")
    if manifest.get("stage") != "E1":
        reasons.append("manifest_stage_mismatch")
    if manifest.get("status") != "OK":
        reasons.append("manifest_status_not_OK")
    expected_key = "{}|{}|{}".format(task.get("weighting"), task.get("setting"),
                                     int(task.get("seed", 0)))
    if manifest.get("formal_run_key") != expected_key:
        reasons.append("manifest_run_key_mismatch")
    if manifest.get("pair_valid") is False or manifest.get("pair_status") == "INVALID":
        reasons.append("pair_marked_invalid")
    inventory_path = os.path.join(run_dir, "nonfinite_inventory.json")
    if os.path.exists(inventory_path):
        try:
            with open(inventory_path, encoding="utf-8") as handle:
                inventory = json.load(handle)
            if not isinstance(inventory, dict):
                reasons.append("inventory_unreadable")
            elif int(inventory.get("unexplained_count", 0)) != 0:
                reasons.append("unexplained_nonfinite_values")
            elif int(inventory.get("count", -1)) != len(inventory.get("entries", [])):
                reasons.append("inventory_count_mismatch")
            else:
                if ("nonfinite_inventory_count" in manifest
                        and int(manifest.get("nonfinite_inventory_count", -1))
                        != int(inventory.get("count", -1))):
                    reasons.append("manifest_inventory_count_mismatch")
                if ("nonfinite_inventory_unexplained_count" in manifest
                        and int(manifest.get("nonfinite_inventory_unexplained_count", -1))
                        != int(inventory.get("unexplained_count", -1))):
                    reasons.append("manifest_inventory_unexplained_count_mismatch")
        except (OSError, ValueError, TypeError):
            reasons.append("inventory_unreadable")
    for filename, fields in REQUIRED_HEADERS.items():
        if os.path.exists(os.path.join(run_dir, filename)):
            valid, missing_fields = ensure_required_fields(
                os.path.join(run_dir, filename), fields)
            if not valid:
                reasons.append("schema:{}:{}".format(filename, ",".join(missing_fields)))
    batch_trace_path = os.path.join(run_dir, "batch_trace.csv")
    if os.path.exists(batch_trace_path):
        try:
            with open(batch_trace_path, newline="", encoding="utf-8") as handle:
                if not next(csv.DictReader(handle), None):
                    reasons.append("batch_trace_has_no_rows")
        except (OSError, csv.Error, StopIteration):
            reasons.append("batch_trace_unreadable")
    config: Dict[str, object] = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                loaded_config = json.load(handle)
            if not isinstance(loaded_config, dict):
                raise ValueError("config must be a JSON object")
            config = loaded_config
            for field, expected in (("weighting", task.get("weighting")),
                                    ("setting", task.get("setting")),
                                    ("seed", int(task.get("seed", 0)))):
                if config.get(field) != expected:
                    reasons.append("config_{}_mismatch".format(field))
        except (OSError, ValueError, TypeError):
            reasons.append("config_unreadable")

    if formal:
        # Every formal run must carry the exact registered protocol.  In
        # particular, checking ``epoch`` here prevents a 3-epoch smoke output
        # from being accepted as a completed 50-epoch matrix cell.
        for field, expected in FORMAL_PROTOCOL.items():
            if config.get(field) != expected:
                reasons.append("formal_config_{}_mismatch".format(field))
        if config.get("loss_weighting") != config.get("weighting"):
            reasons.append("formal_loss_weighting_mismatch")
        if config.get("instrumentation") is not True:
            reasons.append("formal_instrumentation_not_enabled")
        if config.get("formal_valid") is not True:
            reasons.append("formal_valid_marker_missing")
        protocol = config.get("formal_protocol")
        if not isinstance(protocol, Mapping):
            reasons.append("formal_protocol_missing")
        else:
            for field, expected in FORMAL_PROTOCOL.items():
                if protocol.get(field) != expected:
                    reasons.append("formal_protocol_{}_mismatch".format(field))
        expected_dataset = os.path.normcase(os.path.normpath(os.path.abspath(
            "/xzf/data/movielens")))
        actual_dataset = config.get("dataset_path")
        if not actual_dataset or os.path.normcase(os.path.normpath(os.path.abspath(
                str(actual_dataset)))) != expected_dataset:
            reasons.append("formal_dataset_path_mismatch")

        # Validate observed/censored epoch semantics and the expected formal
        # cardinality independently of the config marker.  This check is
        # deliberately based on raw rows, so a relabeled smoke run cannot
        # pass by merely editing config.json.
        observed_value = manifest.get("observed_epochs")
        observed: List[int] = []
        if isinstance(observed_value, (list, tuple)):
            for value in observed_value:
                try:
                    observed.append(int(value))
                except (TypeError, ValueError):
                    reasons.append("formal_observed_epochs_unreadable")
                    break
        else:
            reasons.append("formal_observed_epochs_missing")
        if observed != list(range(len(observed))):
            reasons.append("formal_observed_epochs_not_contiguous")
        stopped_value = manifest.get("stopped_epoch")
        stopped: Optional[int]
        if stopped_value in (None, ""):
            stopped = None
        else:
            try:
                stopped = int(stopped_value)
            except (TypeError, ValueError):
                stopped = None
                reasons.append("formal_stopped_epoch_unreadable")
        censored_value = manifest.get("censored_after_epoch")
        censored: Optional[int]
        if censored_value in (None, ""):
            censored = None
        else:
            try:
                censored = int(censored_value)
            except (TypeError, ValueError):
                censored = None
                reasons.append("formal_censored_after_epoch_unreadable")
        if stopped is None:
            if observed != list(range(FORMAL_PROTOCOL["epoch"])):
                reasons.append("formal_observed_epoch_count_mismatch")
            if censored is not None:
                reasons.append("formal_unexpected_censoring_marker")
        else:
            if stopped < 0 or stopped >= int(FORMAL_PROTOCOL["epoch"]):
                reasons.append("formal_stopped_epoch_out_of_range")
            elif observed != list(range(stopped + 1)):
                reasons.append("formal_observed_epochs_do_not_end_at_stop")
            if censored not in (None, stopped + 1):
                reasons.append("formal_censoring_boundary_mismatch")

        def _csv_rows(filename: str) -> Optional[List[Dict[str, str]]]:
            path = os.path.join(run_dir, filename)
            if not os.path.exists(path):
                return None
            try:
                with open(path, newline="", encoding="utf-8") as handle:
                    return list(csv.DictReader(handle))
            except (OSError, csv.Error):
                return None

        metric_rows = _csv_rows("metrics_epoch.csv")
        if metric_rows is not None:
            metric_epochs = []
            for row in metric_rows:
                try:
                    metric_epochs.append(int(row.get("epoch", "")))
                except (TypeError, ValueError):
                    reasons.append("formal_metrics_epoch_unreadable")
                    break
            expected_metric_epochs = set(range(int(FORMAL_PROTOCOL["epoch"])))
            if set(metric_epochs) != expected_metric_epochs:
                reasons.append("formal_metrics_epoch_cardinality_mismatch")
            expected_cells = int(manifest.get("domain_num", 3)) * int(
                manifest.get("task_num", 2))
            for epoch in expected_metric_epochs:
                if metric_epochs.count(epoch) != expected_cells:
                    reasons.append("formal_metrics_epoch_cell_count_mismatch")
                    break
        batch_rows = _csv_rows("batch_trace.csv")
        if batch_rows is not None:
            trace_epochs = set()
            for row in batch_rows:
                try:
                    trace_epochs.add(int(row.get("epoch", "")))
                except (TypeError, ValueError):
                    reasons.append("formal_batch_trace_epoch_unreadable")
                    break
            if not set(observed).issubset(trace_epochs):
                reasons.append("formal_batch_trace_missing_observed_epoch")

        for identity_field in ("initial_parameter_state_hash", "final_parameter_state_hash"):
            if not manifest.get(identity_field):
                reasons.append("manifest_{}_missing".format(identity_field))
        partition = manifest.get("partition")
        if not isinstance(partition, Mapping):
            reasons.append("manifest_partition_missing")
        else:
            for identity_field in ("groups", "param_count", "param_shapes",
                                   "manifest_hash", "state_hash"):
                if identity_field not in partition or (
                        identity_field in ("manifest_hash", "state_hash")
                        and not partition.get(identity_field)):
                    reasons.append("manifest_partition_{}_missing".format(identity_field))
    return not reasons, reasons


def done_set(path: str, raw_root: Optional[str] = None) -> set:
    """Return only latest, independently revalidated historical ``OK`` runs."""

    latest: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for row in _read_master(path):
        try:
            key = _key(row.get("weighting"), row.get("setting"), row.get("seed", 0))
        except (TypeError, ValueError):
            continue
        latest[key] = row
    done = set()
    for key, row in latest.items():
        if row.get("status") != "OK":
            continue
        root = raw_root or DEFAULT_RAW_ROOT
        run_dir = row.get("run_dir") or os.path.join(
            root, key[0], key[1], "seed{}".format(key[2]))
        task = {"weighting": key[0], "setting": key[1], "seed": key[2],
                "run_dir": run_dir}
        valid, _reasons = validate_run_artifacts(task)
        if not valid:
            continue
        counterpart = _counterpart_task(task)
        counterpart_manifest = _last_manifest(str(counterpart["run_dir"]))
        if (counterpart_manifest.get("pair_valid") is False
                or counterpart_manifest.get("pair_status") == "INVALID"):
            continue
        pair = compare_pair_runs(run_dir, str(counterpart["run_dir"]),
                                 require_formal=True)
        # A formally valid single arm is not complete until its paired arm is
        # present and the common-support/identity comparator is VALID.  This
        # keeps a stale OK row from skipping the counterpart on resume.
        if pair.get("status") != "VALID":
            continue
        done.add(key)
    return done


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidLock:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.held = False

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="ascii") as handle:
                    old_pid = int(handle.read().strip() or "0")
            except (OSError, ValueError):
                old_pid = 0
            if _pid_is_alive(old_pid):
                raise RuntimeError("another Stage E1 orchestrator is alive (pid={})".format(old_pid))
            os.remove(self.path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(self.path, flags)
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, _type, _value, _trace):
        self.release()


def _append_master(path: str, row: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    # Migrate a pre-fix master header once so appending the new pair fields
    # cannot silently produce rows with a different column layout.
    if exists:
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                old_fields = list(reader.fieldnames or [])
                old_rows = list(reader)
            if old_fields != MASTER_FIELDS:
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
                    writer.writeheader()
                    for old_row in old_rows:
                        writer.writerow({field: old_row.get(field, "")
                                         for field in MASTER_FIELDS})
        except (OSError, csv.Error):
            # The normal append below will surface an actual I/O error.  Do
            # not turn a transient read failure into a false completion.
            pass
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MASTER_FIELDS})


def _last_manifest(run_dir: str) -> Dict[str, object]:
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def command_for_task(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        args.python, "-u", RUNNER,
        "--seed", str(task["seed"]),
        "--setting", str(task["setting"]),
        "--weighting", str(task["weighting"]),
        "--role_sizes", ROLE_SIZES,
        "--batch_size", str(args.batch_size),
        "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay),
        "--expert_num", str(args.expert_num),
        "--device", args.device,
        "--dataset_path", args.dataset_path,
        "--run_dir", str(task["run_dir"]),
        "--instrument", "1",
        "--probe_interval", str(args.probe_interval),
    ]


def _runner_environment() -> Dict[str, str]:
    """Keep the repository's Python invocation compatible with its torch env.

    The checked-in ``venv`` uses the host Anaconda standard library, while
    this workspace's PyTorch wheel is installed in ``D:\\CODE\\.venv``.
    Adding that site-packages directory is a no-op when it is absent and
    avoids baking a machine-specific interpreter path into the experiment
    command itself.
    """

    environment = os.environ.copy()
    candidates = [os.path.join(os.path.dirname(ROOT), ".venv", "Lib", "site-packages")]
    existing = environment.get("PYTHONPATH", "")
    entries = [entry for entry in candidates if os.path.isdir(entry)]
    if existing:
        entries.append(existing)
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _set_pair_manifest(task: Mapping[str, object], pair: Mapping[str, object],
                       *, status: str, valid: Optional[bool]) -> None:
    run_dir = str(task["run_dir"])
    manifest = _last_manifest(run_dir)
    if not manifest:
        return
    counterpart = _counterpart_task(task)
    if status == "INVALID":
        manifest["status"] = "INVALID_PAIR"
    elif status == "VALID":
        # A clean rerun can recover a manifest that was previously marked
        # INVALID_PAIR; the pair marker and top-level status must agree.
        manifest["status"] = "OK"
    manifest.update({
        "pair_status": status,
        "pair_valid": valid,
        "pair_counterpart": str(counterpart["run_dir"]),
        "pair_comparison": dict(pair),
    })
    write_json(os.path.join(run_dir, "run_manifest.json"), manifest)


def _latest_master_row(master: str, key: Tuple[str, str, int]) -> Optional[Dict[str, str]]:
    latest = None
    for row in _read_master(master):
        try:
            if _key(row.get("weighting"), row.get("setting"), row.get("seed", 0)) == key:
                latest = row
        except (TypeError, ValueError):
            continue
    return latest


def _append_pair_invalid_correction(master: str, task: Mapping[str, object],
                                    pair: Mapping[str, object]) -> None:
    """Append a correction row so a previously OK counterpart is not stale."""

    previous = _latest_master_row(master, _task_key(task))
    if previous is None:
        return
    correction = dict(previous)
    correction.update({
        "status": "INVALID_PAIR", "pair_status": "INVALID",
        "pair_valid": False,
        "pair_max_abs_loss_diff": pair.get("max_abs_loss_diff", ""),
        "note": "pair_invalidated: {}".format(pair.get("reason", "mismatch")),
    })
    _append_master(master, correction)


def _append_pair_valid_correction(master: str, task: Mapping[str, object],
                                  pair: Mapping[str, object]) -> None:
    """Clear a historical INVALID_PAIR status after a clean rerun."""

    previous = _latest_master_row(master, _task_key(task))
    if previous is None or previous.get("status") not in ("INVALID_PAIR", "FAILED"):
        return
    correction = dict(previous)
    correction.update({
        "status": "OK", "return_code": 0,
        "pair_status": "VALID", "pair_valid": True,
        "pair_max_abs_loss_diff": pair.get("max_abs_loss_diff", ""),
        "note": "pair_revalidated: VALID",
    })
    _append_master(master, correction)


def _pair_for_completed_task(task: Mapping[str, object]) -> Dict[str, object]:
    counterpart = _counterpart_task(task)
    counterpart_dir = str(counterpart["run_dir"])
    if not os.path.isdir(counterpart_dir) or not os.path.exists(
            os.path.join(counterpart_dir, "run_manifest.json")):
        return {
            "status": "PENDING", "valid": None,
            "reason": "counterpart_missing", "equal_run_dir": "",
            "sample_run_dir": "", "mismatch_count": 0,
            "max_abs_loss_diff": "",
        }
    other_valid, other_reasons = validate_run_artifacts(counterpart)
    if not other_valid:
        # A partially written/stale counterpart should be resumed first.  If
        # the only stale marker is a previous pair invalidation, a clean
        # rerun may recover it; otherwise both traces can still make a
        # concrete mismatch invalid instead of hiding it behind PENDING.
        pair = compare_pair_runs(
            str(task["run_dir"]) if str(task["weighting"]) == "equal" else counterpart_dir,
            counterpart_dir if str(task["weighting"]) == "equal" else str(task["run_dir"]),
            require_formal=True)
        counterpart_manifest = _last_manifest(counterpart_dir)
        recoverable_pair_marker = (
            counterpart_manifest.get("status") == "INVALID_PAIR"
            and not [reason for reason in other_reasons
                     if reason not in ("pair_marked_invalid", "manifest_status_not_OK")])
        if pair.get("status") == "INVALID" or (
                recoverable_pair_marker and pair.get("status") == "VALID"):
            return pair
        return {
            "status": "PENDING", "valid": None,
            "reason": "counterpart_artifacts_not_validated",
            "equal_run_dir": (str(task["run_dir"]) if str(task["weighting"]) == "equal"
                               else counterpart_dir),
            "sample_run_dir": (counterpart_dir if str(task["weighting"]) == "equal"
                                else str(task["run_dir"])),
            "mismatch_count": 0, "max_abs_loss_diff": "",
        }
    equal_dir = str(task["run_dir"]) if str(task["weighting"]) == "equal" else counterpart_dir
    sample_dir = counterpart_dir if str(task["weighting"]) == "equal" else str(task["run_dir"])
    return compare_pair_runs(equal_dir, sample_dir, require_formal=True)


def _finalize_task(task: Mapping[str, object], return_code: int, wall: float,
                   output: str, master: str) -> Dict[str, object]:
    run_dir = str(task["run_dir"])
    manifest = _last_manifest(run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    artifact_ok, validation_reasons = validate_run_artifacts(task)
    ok = return_code == 0 and artifact_ok
    status = "OK" if ok else "FAILED"
    pair_status = manifest.get("pair_status", "NOT_READY")
    pair_valid = manifest.get("pair_valid", "")
    pair_max_abs_loss_diff = ""
    note_parts: List[str] = []
    if not ok:
        if missing:
            note_parts.append("missing=" + ",".join(missing))
        if validation_reasons:
            note_parts.append("invalid_artifacts=" + ",".join(validation_reasons))
        if return_code != 0:
            note_parts.append("process_failed")
        if output.strip():
            note_parts.append(output.strip().splitlines()[-1][:300])
    else:
        pair = _pair_for_completed_task(task)
        pair_status = str(pair.get("status", "PENDING"))
        pair_valid = pair.get("valid", None)
        pair_max_abs_loss_diff = pair.get("max_abs_loss_diff", "")
        if pair_status == "INVALID":
            status = "INVALID_PAIR"
            note_parts.append("pair_invalid=" + str(pair.get("reason", "mismatch")))
            _set_pair_manifest(task, pair, status="INVALID", valid=False)
            counterpart = _counterpart_task(task)
            _set_pair_manifest(counterpart, pair, status="INVALID", valid=False)
            _append_pair_invalid_correction(master, counterpart, pair)
        elif pair_status == "VALID":
            _set_pair_manifest(task, pair, status="VALID", valid=True)
            counterpart = _counterpart_task(task)
            _set_pair_manifest(counterpart, pair, status="VALID", valid=True)
            _append_pair_valid_correction(master, counterpart, pair)
        else:
            _set_pair_manifest(task, pair, status="PENDING", valid=None)
            note_parts.append("pair_pending")
    row = {
        "stage": "E1", "weighting": task["weighting"], "setting": task["setting"],
        "seed": task["seed"], "status": status,
        "return_code": return_code, "run_dir": task["run_dir"],
        "stdout_log": task["stdout_log"], "wall_seconds": "{:.3f}".format(wall),
        "best_epoch": manifest.get("best_epoch", ""),
        "stopped_epoch": manifest.get("stopped_epoch", ""),
        "observed_epoch_count": len(manifest.get("observed_epochs", [])),
        "gradient_step_count": manifest.get("gradient_step_count", ""),
        "gradient_epoch_count": manifest.get("gradient_epoch_count", ""),
        "routing_row_count": manifest.get("routing_row_count", ""),
        "pair_status": pair_status, "pair_valid": pair_valid,
        "pair_max_abs_loss_diff": pair_max_abs_loss_diff,
        "note": "; ".join(note_parts),
    }
    _append_master(master, row)
    return row


def run_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace) -> int:
    os.makedirs(args.raw_root, exist_ok=True)
    lock = PidLock(args.lock)
    with lock:
        launched = 0
        had_failure = False
        for task in tasks:
            if launched >= args.max_runs:
                break
            run_dir = str(task["run_dir"])
            os.makedirs(run_dir, exist_ok=True)
            command = command_for_task(task, args)
            print("[stage_E1][launch]", " ".join(command), flush=True)
            started = time.time()
            with open(str(task["stdout_log"]), "w", encoding="utf-8", errors="replace") as handle:
                process = subprocess.run(command, cwd=ROOT, stdout=handle,
                                         stderr=subprocess.STDOUT, text=True,
                                         env=_runner_environment())
            with open(str(task["stdout_log"]), encoding="utf-8", errors="replace") as handle:
                output = handle.read()
            row = _finalize_task(task, int(process.returncode), time.time() - started,
                                 output, args.master)
            print("[stage_E1][done]", row["weighting"], row["setting"],
                  row["seed"], row["status"], flush=True)
            if row["status"] not in ("OK",):
                had_failure = True
            launched += 1
    return 1 if had_failure else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E1 24-run orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--probe_interval", type=int, default=1)
    parser.add_argument("--max_runs", type=int, default=10**9)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.max_runs < 0:
        raise ValueError("--max_runs must be non-negative")
    tasks = build_tasks(args.raw_root)
    completed = done_set(args.master, args.raw_root)
    pending = [task for task in tasks if
               (task["weighting"], task["setting"], int(task["seed"])) not in completed]
    print("[stage_E1] fixed matrix: 2 weighting x 3 settings x 4 seeds = 24", flush=True)
    print("[stage_E1] pending:", len(pending), flush=True)
    if args.dry_run:
        for task in pending[:args.max_runs]:
            print("[stage_E1][dry-run]", " ".join(command_for_task(task, args)), flush=True)
        return 0
    return run_queue(pending, args)


if __name__ == "__main__":
    raise SystemExit(main())
