"""Resume-safe orchestration for Stage E3.

The queue is exactly 5 alpha values x 3 role rotations x 4 seeds = 60 joint
runs.  Before that queue is launched, the orchestrator audits seed-level oracle
artifacts and, only for missing/invalid references, queues the 36 frozen
single-domain oracle runs.  This module records statuses and does not interpret
experimental results.
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
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)
DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E3_raw_results")
DEFAULT_MASTER = os.path.join(HERE, "stage_E3_runs.csv")
DEFAULT_ORACLE_MASTER = os.path.join(DEFAULT_RAW_ROOT, "oracle_reference_runs.csv")
DEFAULT_LOCK = os.path.join(HERE, "logs", "stage_E3_orchestrator.lock")
DEFAULT_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable

from stage_E1_instrumentation import write_json  # noqa: E402
from run_movielens import get_movielens_data_rank_multidomain  # noqa: E402
from stage_E3_run import oracle_split_row_fingerprints  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_KEYS,
    E3_FORMAL_PROTOCOL,
    EXPECTED_SEEDS,
    EXPECTED_SETTINGS,
    ROLE_MAPS,
    alpha_key,
    compare_alpha_block,
    normalise_dataset_path,
    resolve_oracle_metrics,
    validate_alpha_coefficients,
    validate_oracle_artifact,
)


MASTER_FIELDS = [
    "stage", "alpha", "alpha_key", "setting", "seed", "status", "return_code",
    "run_dir", "stdout_log", "wall_seconds", "best_epoch", "stopped_epoch",
    "observed_epoch_count", "shared_gradient_step_count", "shared_gradient_epoch_count",
    "block_status", "block_valid", "note",
]
ORACLE_MASTER_FIELDS = [
    "stage", "setting", "domain", "seed", "status", "return_code", "run_dir",
    "stdout_log", "wall_seconds", "reason",
]
REQUIRED_ARTIFACTS = (
    "config.json", "stdout.log", "run_manifest.json", "metrics_epoch.csv",
    "metrics_summary.csv", "shared_gradient_step.csv", "shared_gradient_epoch.csv",
    "batch_trace.csv", "nonfinite_inventory.json", "model.pth",
)


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for key, value in zip(ALPHA_KEYS, ALPHA_GRID):
        for setting in EXPECTED_SETTINGS:
            for seed in EXPECTED_SEEDS:
                run_dir = os.path.join(raw_root, key, setting, f"seed{seed}")
                tasks.append({
                    "stage": "E3", "alpha": float(value), "alpha_key": key,
                    "setting": setting, "seed": int(seed), "run_dir": run_dir,
                    "stdout_log": os.path.join(run_dir, "stdout.log"),
                })
    return tasks


def build_oracle_tasks(oracle_root: str) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for setting in EXPECTED_SETTINGS:
        for domain in range(3):
            for seed in EXPECTED_SEEDS:
                run_dir = os.path.join(oracle_root, setting, f"d{domain}", f"seed{seed}")
                tasks.append({
                    "stage": "E3_ORACLE", "setting": setting, "domain": int(domain),
                    "seed": int(seed), "run_dir": run_dir,
                    "stdout_log": os.path.join(run_dir, "stdout.log"),
                })
    return tasks


def build_expected_oracle_split_fingerprints(
    dataset_path: str,
    *,
    role_sizes: Mapping[str, int] = E3_FORMAL_PROTOCOL["role_sizes_parsed"],  # type: ignore[arg-type]
) -> Dict[Tuple[str, int, int], Dict[str, str]]:
    """Build external oracle row identities from the current dataset once.

    The map is keyed by the current joint ``setting/seed/domain`` and uses the
    same role sampling and domain-column-to-zero view as the oracle runner.
    It is passed into every audit/resolve/validator call; artifact-local
    manifest/config agreement alone is intentionally insufficient.
    """

    _dense, _sparse, x_train, y_train, x_val, y_val, x_test, y_test, _domain_num = \
        get_movielens_data_rank_multidomain("all", "all", dataset_path)
    result: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    for setting in EXPECTED_SETTINGS:
        for seed in EXPECTED_SEEDS:
            per_domain = oracle_split_row_fingerprints(
                x_train, y_train, x_val, y_val, x_test, y_test,
                setting, int(seed), role_sizes)
            for domain, fingerprints in per_domain.items():
                result[(str(setting), int(seed), int(domain))] = dict(fingerprints)
    return result


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


def _task_key(task: Mapping[str, object]) -> Tuple[str, str, int]:
    return str(task["alpha_key"]), str(task["setting"]), int(task["seed"])


def _oracle_key(task: Mapping[str, object]) -> Tuple[str, int, int]:
    return str(task["setting"]), int(task["domain"]), int(task["seed"])


def validate_run_artifacts(
    task: Mapping[str, object], *, formal: bool = True,
    dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    run_dir = str(task["run_dir"])
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_directory_missing"]
    missing = [name for name in REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        reasons.append("missing_files:" + ",".join(missing))
    config = _read_json(os.path.join(run_dir, "config.json"))
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if manifest.get("stage") != "E3":
        reasons.append("manifest_stage_mismatch")
    if manifest.get("status") != "OK":
        reasons.append("manifest_status_not_OK")
    manifest_dataset_identity = str(manifest.get("dataset_identity", ""))
    config_dataset_identity = str(config.get("dataset_identity", ""))
    expected_dataset_identity = (
        normalise_dataset_path(dataset_identity) if dataset_identity else manifest_dataset_identity)
    if (not manifest_dataset_identity or manifest_dataset_identity != config_dataset_identity
            or manifest_dataset_identity != expected_dataset_identity):
        reasons.append("dataset_identity_invalid")
    manifest_device_identity = str(manifest.get("device_identity", ""))
    config_device_identity = str(config.get("device_identity", ""))
    expected_device_identity = str(device_identity) if device_identity else manifest_device_identity
    if (not manifest_device_identity or manifest_device_identity != config_device_identity
            or manifest_device_identity != expected_device_identity):
        reasons.append("device_identity_invalid")
    manifest_split_rows = manifest.get("split_row_fingerprints")
    if (not isinstance(manifest_split_rows, Mapping)
            or not all(isinstance(manifest_split_rows.get(name), str)
                       and bool(manifest_split_rows.get(name))
                       for name in ("validation", "test"))
            or manifest_split_rows != config.get("split_row_fingerprints")):
        reasons.append("split_row_identity_invalid")
    if str(manifest.get("alpha_key")) != str(task.get("alpha_key")):
        reasons.append("manifest_alpha_key_mismatch")
    if str(manifest.get("setting")) != str(task.get("setting")):
        reasons.append("manifest_setting_mismatch")
    if int(manifest.get("seed", -1)) != int(task.get("seed", -2)):
        reasons.append("manifest_seed_mismatch")
    try:
        validate_alpha_coefficients(float(task["alpha"]), manifest.get("domain_shares", []),
                                    manifest.get("actual_coefficients", []))
    except (AssertionError, TypeError, ValueError):
        reasons.append("coefficient_identity_invalid")
    expected_role_counts = {"L": 150000, "M": 34500, "S": 3000}
    role_map = manifest.get("role_by_domain", {})
    observed_counts = manifest.get("train_domain_counts", {})
    if not isinstance(role_map, Mapping) or not isinstance(observed_counts, Mapping):
        reasons.append("role_count_identity_missing")
    else:
        canonical_roles = ROLE_MAPS.get(str(task.get("setting")), {})
        for domain in ("0", "1", "2"):
            expected_role_name = canonical_roles.get(int(domain))
            declared_role = role_map.get(domain, role_map.get(int(domain)))
            expected = expected_role_counts.get(str(expected_role_name))
            observed = observed_counts.get(domain, observed_counts.get(int(domain)))
            if (declared_role != expected_role_name or expected is None
                    or int(observed or -1) != expected):
                reasons.append("train_count_identity_invalid")
                break
    inventory = _read_json(os.path.join(run_dir, "nonfinite_inventory.json"))
    if inventory and int(inventory.get("unexplained_count", 0)) != 0:
        reasons.append("unexplained_nonfinite")
    # A selected checkpoint must be a complete finite six-cell object even for
    # a non-formal smoke; otherwise a later summary could silently average a
    # reduced cell set.
    selected = []
    for row in _read_csv(os.path.join(run_dir, "metrics_epoch.csv")):
        try:
            if int(row.get("selected_checkpoint", 0)) == 1:
                selected.append(row)
        except (TypeError, ValueError):
            continue
    expected_cells = {0, 1, 2, 3, 4, 5}
    selected_cells = set()
    for row in selected:
        try:
            selected_cells.add(int(row.get("cell", -1)))
        except (TypeError, ValueError):
            selected_cells.add(-1)
    if len(selected) != len(expected_cells) or selected_cells != expected_cells:
        reasons.append("selected_checkpoint_incomplete")
    else:
        for row in selected:
            for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"):
                try:
                    if not bool(np.isfinite(float(row.get(field)))):
                        raise ValueError
                except (TypeError, ValueError):
                    reasons.append("selected_checkpoint_nonfinite")
                    break
    if formal:
        if manifest.get("formal_valid") is not True or config.get("formal_valid") is not True:
            reasons.append("formal_invalid")
        if manifest.get("formal_protocol") != E3_FORMAL_PROTOCOL:
            reasons.append("formal_protocol_mismatch")
        observed = manifest.get("observed_epochs")
        if not isinstance(observed, list) or not observed or int(observed[0]) != 0:
            reasons.append("observed_epochs_invalid")
    return not reasons, reasons


def _read_master(path: str) -> List[Dict[str, str]]:
    return _read_csv(path)


def _append_row(path: str, fields: Sequence[str], row: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    old = _read_csv(path) if os.path.exists(path) else []
    old.append({field: row.get(field, "") for field in fields})
    _write_csv(path, old, fields)


def _latest_status(path: str, key_fields: Sequence[str], key: Tuple[object, ...]) -> Optional[Dict[str, str]]:
    found = None
    for row in _read_master(path):
        if tuple(str(row.get(field, "")) for field in key_fields) == tuple(str(value) for value in key):
            found = row
    return found


def _formal_task_rows(master: str) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    return {_task_key({"alpha_key": row.get("alpha_key", ""), "setting": row.get("setting", ""),
                       "seed": row.get("seed", -1)}): row
            for row in _read_master(master) if row.get("stage") == "E3"}


def _block_tasks(task: Mapping[str, object], raw_root: str) -> Dict[str, str]:
    setting, seed = str(task["setting"]), int(task["seed"])
    return {key: os.path.join(raw_root, key, setting, f"seed{seed}") for key in ALPHA_KEYS}


def _set_block_status(task: Mapping[str, object], raw_root: str, block: Mapping[str, object]) -> None:
    status = str(block.get("status", "PENDING"))
    valid = block.get("valid")
    for key, run_dir in _block_tasks(task, raw_root).items():
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        manifest["block_status"] = status
        manifest["block_valid"] = valid
        manifest["block_comparison"] = dict(block)
        write_json(manifest_path, manifest)


def _block_for_task(task: Mapping[str, object], raw_root: str,
                    *, require_formal: bool = True) -> Dict[str, object]:
    return compare_alpha_block(_block_tasks(task, raw_root), require_formal=require_formal)


def done_set(master: str, raw_root: str = DEFAULT_RAW_ROOT,
             *, dataset_identity: Optional[str] = None,
             device_identity: Optional[str] = None) -> set:
    """Return only individually valid runs in a fully valid five-alpha block."""

    done = set()
    latest = _formal_task_rows(master)
    for task in build_tasks(raw_root):
        key = _task_key(task)
        row = latest.get(key)
        if not row or row.get("status") != "OK":
            continue
        valid, _ = validate_run_artifacts(
            task, formal=True, dataset_identity=dataset_identity,
            device_identity=device_identity)
        if not valid:
            continue
        block = _block_for_task(task, raw_root, require_formal=True)
        if block.get("status") == "VALID":
            done.add(key)
    return done


def command_for_task(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E3_run.py"),
        "--alpha", str(task["alpha"]), "--setting", str(task["setting"]),
        "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
        "--dataset_path", str(args.dataset_path), "--device", str(args.device),
        "--batch_size", str(args.batch_size), "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--probe_interval", str(args.probe_interval), "--instrument", "1",
        "--earlystop_patience", str(args.earlystop_patience),
        "--oracle_root", str(args.oracle_root),
        "--legacy_oracle_root", str(args.legacy_oracle_root),
    ]


def command_for_oracle(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E3_oracle_run.py"),
        "--setting", str(task["setting"]), "--domain", str(task["domain"]),
        "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
        "--dataset_path", str(args.dataset_path), "--device", str(args.device),
        "--batch_size", str(args.batch_size), "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--earlystop_patience", str(args.earlystop_patience),
    ]


def _runner_environment() -> Dict[str, str]:
    """Carry the repository's torch-enabled site-packages into child runs."""

    env = os.environ.copy()
    bundled = os.path.join(os.path.dirname(ROOT), ".venv", "Lib", "site-packages")
    if os.path.isdir(bundled):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bundled + (os.pathsep + existing if existing else "")
    return env


def audit_oracles(
    legacy_root: str, oracle_root: str, *, dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
    expected_oracle_split_fingerprints: Optional[
        Mapping[Tuple[str, int, int], Mapping[str, str]]
    ] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    all_tasks = build_oracle_tasks(oracle_root)
    missing: List[Dict[str, object]] = []
    statuses: List[Dict[str, object]] = []
    for task in all_tasks:
        expected_for_seed = None
        if isinstance(expected_oracle_split_fingerprints, Mapping):
            expected_for_seed = {
                candidate_domain: expected_oracle_split_fingerprints.get(
                    (str(task["setting"]), int(task["seed"]), candidate_domain), {})
                for candidate_domain in range(3)
            }
        _metrics, audit = resolve_oracle_metrics(
            [oracle_root, legacy_root], str(task["setting"]), int(task["seed"]),
            role_sizes=E3_FORMAL_PROTOCOL["role_sizes_parsed"], domain_num=3, task_num=2,
            dataset_identity=dataset_identity, device_identity=device_identity,
            expected_split_row_fingerprints=expected_for_seed)
        domain_status = next((item for item in audit.get("domains", [])
                              if int(item.get("domain", -1)) == int(task["domain"])), {})
        status = "REUSED" if domain_status.get("status") == "reused" else "MISSING"
        row = dict(task, status=status, reason=domain_status.get("reason", "missing"))
        statuses.append(row)
        if status != "REUSED":
            missing.append(task)
    return missing, statuses


def _finalize_oracle(task: Mapping[str, object], return_code: int, wall: float,
                     output: str, master: str, dataset_identity: Optional[str] = None,
                     device_identity: Optional[str] = None,
                     expected_split_row_fingerprints: Optional[Mapping[str, str]] = None,
                     ) -> Dict[str, object]:
    required = ("config.json", "run_manifest.json", "metrics_epoch.csv", "metrics_summary.csv", "stdout.log")
    missing = [name for name in required if not os.path.exists(os.path.join(str(task["run_dir"]), name))]
    manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
    artifact_valid, artifact_reason, _ = validate_oracle_artifact(
        str(task["run_dir"]), setting=str(task["setting"]), seed=int(task["seed"]),
        domain=int(task["domain"]), role_sizes=E3_FORMAL_PROTOCOL["role_sizes_parsed"],
        dataset_identity=dataset_identity, device_identity=device_identity,
        expected_split_row_fingerprints=expected_split_row_fingerprints)
    ok = return_code == 0 and not missing and artifact_valid
    status = "OK" if ok else "FAILED"
    note = "; ".join(["missing=" + ",".join(missing)] if missing else [])
    if return_code != 0:
        note = (note + "; " if note else "") + "process_failed"
    if not artifact_valid:
        note = (note + "; " if note else "") + artifact_reason
    row = {"stage": "E3_ORACLE", "setting": task["setting"], "domain": task["domain"],
           "seed": task["seed"], "status": status, "return_code": return_code,
           "run_dir": task["run_dir"], "stdout_log": task["stdout_log"],
           "wall_seconds": "{:.3f}".format(wall), "reason": note}
    _append_row(master, ORACLE_MASTER_FIELDS, row)
    return row


def _finalize_joint(task: Mapping[str, object], return_code: int, wall: float,
                    output: str, master: str, raw_root: str,
                    dataset_identity: Optional[str] = None,
                    device_identity: Optional[str] = None) -> Dict[str, object]:
    valid, reasons = validate_run_artifacts(
        task, formal=False, dataset_identity=dataset_identity,
        device_identity=device_identity)
    status = "OK" if return_code == 0 and valid else "FAILED"
    block = {"status": "PENDING", "valid": None}
    if status == "OK":
        block = _block_for_task(task, raw_root, require_formal=False)
        _set_block_status(task, raw_root, block)
        if block.get("status") == "INVALID":
            status = "INVALID_BLOCK"
    note_parts = list(reasons)
    if return_code != 0:
        note_parts.append("process_failed")
    if output.strip():
        note_parts.append(output.strip().splitlines()[-1][:300])
    manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
    row = {
        "stage": "E3", "alpha": task["alpha"], "alpha_key": task["alpha_key"],
        "setting": task["setting"], "seed": task["seed"], "status": status,
        "return_code": return_code, "run_dir": task["run_dir"],
        "stdout_log": task["stdout_log"], "wall_seconds": "{:.3f}".format(wall),
        "best_epoch": manifest.get("best_epoch", ""), "stopped_epoch": manifest.get("stopped_epoch", ""),
        "observed_epoch_count": len(manifest.get("observed_epochs", [])),
        "shared_gradient_step_count": manifest.get("shared_gradient_step_count", ""),
        "shared_gradient_epoch_count": manifest.get("shared_gradient_epoch_count", ""),
        "block_status": block.get("status", "PENDING"), "block_valid": block.get("valid"),
        "note": "; ".join(note_parts),
    }
    _append_row(master, MASTER_FIELDS, row)
    return row


def _run_oracle_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace) -> int:
    failed = False
    os.makedirs(args.oracle_root, exist_ok=True)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_oracle(task, args)
        print("[stage_E3_oracle][launch]", " ".join(command), flush=True)
        started = time.time()
        with open(str(task["stdout_log"]), "w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle,
                                     stderr=subprocess.STDOUT, text=True,
                                     env=_runner_environment())
        output = ""
        try:
            with open(str(task["stdout_log"]), encoding="utf-8", errors="replace") as handle:
                output = handle.read()
        except OSError:
            pass
        expected_rows = None
        expected_all = getattr(args, "expected_oracle_split_fingerprints", None)
        if isinstance(expected_all, Mapping):
            expected_rows = expected_all.get(
                (str(task["setting"]), int(task["seed"]), int(task["domain"])))
        row = _finalize_oracle(task, int(process.returncode), time.time() - started,
                               output, args.oracle_master,
                               dataset_identity=normalise_dataset_path(args.dataset_path),
                               device_identity=str(args.device),
                               expected_split_row_fingerprints=expected_rows)
        print("[stage_E3_oracle][done]", task["setting"], task["domain"], task["seed"],
              row["status"], flush=True)
        failed = failed or row["status"] != "OK"
        if failed:
            break
    return 1 if failed else 0


def _run_joint_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace) -> int:
    failed = False
    os.makedirs(args.raw_root, exist_ok=True)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_task(task, args)
        print("[stage_E3][launch]", " ".join(command), flush=True)
        started = time.time()
        with open(str(task["stdout_log"]), "w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle,
                                     stderr=subprocess.STDOUT, text=True,
                                     env=_runner_environment())
        try:
            with open(str(task["stdout_log"]), encoding="utf-8", errors="replace") as handle:
                output = handle.read()
        except OSError:
            output = ""
        row = _finalize_joint(task, int(process.returncode), time.time() - started,
                              output, args.master, args.raw_root,
                              dataset_identity=normalise_dataset_path(args.dataset_path),
                              device_identity=str(args.device))
        print("[stage_E3][done]", task["alpha_key"], task["setting"], task["seed"],
              row["status"], flush=True)
        failed = failed or row["status"] not in ("OK",)
    return 1 if failed else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E3 60-run alpha orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--oracle_root", default=os.path.join(DEFAULT_RAW_ROOT, "oracle_reference"))
    parser.add_argument("--legacy_oracle_root", default=os.path.join(HERE, "P0_2_ratio_sweep"))
    parser.add_argument("--oracle_master", default=DEFAULT_ORACLE_MASTER)
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
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--max_runs", type=int, default=10**9)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.max_runs < 0:
        raise ValueError("--max_runs must be non-negative")
    expected_oracle_split_fingerprints = build_expected_oracle_split_fingerprints(
        args.dataset_path)
    args.expected_oracle_split_fingerprints = expected_oracle_split_fingerprints
    oracle_missing, oracle_status = audit_oracles(
        args.legacy_oracle_root, args.oracle_root,
        dataset_identity=normalise_dataset_path(args.dataset_path),
        device_identity=str(args.device),
        expected_oracle_split_fingerprints=expected_oracle_split_fingerprints)
    _write_csv(args.oracle_master, oracle_status, ORACLE_MASTER_FIELDS)
    print("[stage_E3] oracle audit: 36 expected; missing/invalid:", len(oracle_missing), flush=True)
    if args.dry_run:
        for task in oracle_missing[:args.max_runs]:
            print("[stage_E3][dry-run][oracle]", " ".join(command_for_oracle(task, args)), flush=True)
        completed = done_set(
            args.master, args.raw_root,
            dataset_identity=normalise_dataset_path(args.dataset_path),
            device_identity=str(args.device))
        pending = [task for task in build_tasks(args.raw_root)
                   if _task_key(task) not in completed]
        print("[stage_E3] fixed joint matrix: 5 alpha x 3 settings x 4 seeds = 60", flush=True)
        print("[stage_E3] pending joint:", len(pending), flush=True)
        for task in pending[:max(0, args.max_runs - len(oracle_missing))]:
            print("[stage_E3][dry-run][joint]", " ".join(command_for_task(task, args)), flush=True)
        return 0
    if oracle_missing:
        oracle_rc = _run_oracle_queue(oracle_missing, args)
        if oracle_rc != 0:
            return oracle_rc
        oracle_missing, oracle_status = audit_oracles(
            args.legacy_oracle_root, args.oracle_root,
            dataset_identity=normalise_dataset_path(args.dataset_path),
            device_identity=str(args.device),
            expected_oracle_split_fingerprints=expected_oracle_split_fingerprints)
        _write_csv(args.oracle_master, oracle_status, ORACLE_MASTER_FIELDS)
        if oracle_missing:
            print("[stage_E3] oracle audit still incomplete; joint matrix not launched", flush=True)
            return 1
    completed = done_set(
        args.master, args.raw_root,
        dataset_identity=normalise_dataset_path(args.dataset_path),
        device_identity=str(args.device))
    pending = [task for task in build_tasks(args.raw_root) if _task_key(task) not in completed]
    print("[stage_E3] fixed joint matrix: 5 alpha x 3 settings x 4 seeds = 60", flush=True)
    print("[stage_E3] pending joint:", len(pending), flush=True)
    return _run_joint_queue(pending[:args.max_runs], args)


if __name__ == "__main__":
    raise SystemExit(main())
