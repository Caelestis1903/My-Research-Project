"""Resume-safe fixed matrix orchestration for Stage E5.

The formal queue is exactly five registered arms by four registered seeds on
canonical MovieLens ``d0_large``.  Existing E3 frozen single-domain oracle
artifacts are audited and reused; this orchestrator never launches oracle
training and never computes a scientific conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)

from run_movielens import get_movielens_data_rank_multidomain  # noqa: E402
from stage_E1_instrumentation import write_json  # noqa: E402
from stage_E3_run import oracle_split_row_fingerprints  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    resolve_oracle_metrics,
    normalise_dataset_path,
)
from stage_E5_schedule import (  # noqa: E402
    E5_ARM_KEYS,
    E5_EPOCHS,
    E5_FORMAL_PROTOCOL,
    E5_FORMAL_ROLE_SIZES,
    E5_FORMAL_SETTING,
    EXPECTED_SEEDS,
    compare_e5_block,
    arm_alpha,
    expected_matrix_tasks,
    scheduled_coefficients_numpy,
    validate_matrix_tasks,
)


DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E5_raw_results")
DEFAULT_MASTER = os.path.join(HERE, "stage_E5_runs.csv")
DEFAULT_ORACLE_ROOT = os.path.join(HERE, "stage_E3_raw_results", "oracle_reference")
DEFAULT_LEGACY_ORACLE_ROOT = os.path.join(HERE, "P0_2_ratio_sweep")
DEFAULT_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable

MASTER_FIELDS = [
    "stage", "arm_key", "setting", "seed", "status", "return_code", "run_dir",
    "stdout_log", "wall_seconds", "best_epoch", "fixed_final_epoch", "observed_epoch_count",
    "total_main_steps", "total_architecture_steps", "shared_gradient_step_count",
    "shared_gradient_epoch_count", "oracle_coverage", "formal_valid", "block_status",
    "block_valid", "note",
]
REQUIRED_ARTIFACTS = (
    "config.json", "stdout.log", "run_manifest.json", "metrics_epoch.csv",
    "metrics_summary.csv", "shared_gradient_step.csv", "shared_gradient_epoch.csv",
    "schedule_epoch.csv", "batch_trace.csv", "nonfinite_inventory.json", "model.pth",
)


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    return expected_matrix_tasks(raw_root)


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
    return str(task.get("arm_key", "")), str(task.get("setting", "")), int(task.get("seed", -1))


def _latest_rows(path: str) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    found: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for row in _read_csv(path):
        if row.get("stage") == "E5":
            try:
                key = (str(row.get("arm_key", "")), str(row.get("setting", "")), int(row.get("seed", -1)))
            except (TypeError, ValueError):
                continue
            found[key] = row
    return found


def build_expected_oracle_split_fingerprints(dataset_path: str) -> Dict[Tuple[int, int], Dict[str, str]]:
    """Build the external split identities required by E3 oracle validation."""

    _dense, _sparse, x_train, y_train, x_val, y_val, x_test, y_test, _domain_num = \
        get_movielens_data_rank_multidomain("all", "all", dataset_path)
    result: Dict[Tuple[int, int], Dict[str, str]] = {}
    for seed in EXPECTED_SEEDS:
        per_domain = oracle_split_row_fingerprints(
            x_train, y_train, x_val, y_val, x_test, y_test,
            E5_FORMAL_SETTING, int(seed), E5_FORMAL_ROLE_SIZES)
        for domain, fingerprints in per_domain.items():
            result[(int(seed), int(domain))] = dict(fingerprints)
    return result


def audit_oracles(
    oracle_root: str,
    legacy_root: str,
    *,
    dataset_path: str,
    device_identity: str,
    expected_split_fingerprints: Optional[Mapping[Tuple[int, int], Mapping[str, str]]] = None,
) -> Dict[str, object]:
    domains: List[Dict[str, object]] = []
    all_reused = True
    for seed in EXPECTED_SEEDS:
        expected_for_seed = {
            domain: (expected_split_fingerprints or {}).get((int(seed), domain), {})
            for domain in range(3)
        }
        _metrics, audit = resolve_oracle_metrics(
            [oracle_root, legacy_root], E5_FORMAL_SETTING, int(seed),
            role_sizes=E5_FORMAL_ROLE_SIZES, domain_num=3, task_num=2,
            dataset_identity=normalise_dataset_path(dataset_path),
            device_identity=str(device_identity),
            expected_split_row_fingerprints=expected_for_seed)
        for item in audit.get("domains", []):
            entry = dict(item)
            entry["seed"] = int(seed)
            domains.append(entry)
        all_reused = all_reused and bool(audit.get("complete", False))
    return {
        "stage": "E5", "setting": E5_FORMAL_SETTING, "expected_seeds": list(EXPECTED_SEEDS),
        "expected_domains": 3, "domains": domains, "complete": bool(all_reused),
        "coverage_fraction": float(sum(1 for item in domains if item.get("status") == "reused")) / 12.0,
    }


def _float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def validate_run_artifacts(
    task: Mapping[str, object], *, formal: bool = True,
    dataset_identity: Optional[str] = None, device_identity: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    run_dir = str(task.get("run_dir", ""))
    reasons: List[str] = []
    if not os.path.isdir(run_dir):
        return False, ["run_directory_missing"]
    missing = [name for name in REQUIRED_ARTIFACTS if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        reasons.append("missing_files:" + ",".join(missing))
    config = _read_json(os.path.join(run_dir, "config.json"))
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if manifest.get("stage") != "E5" or manifest.get("status") != "OK":
        reasons.append("manifest_stage_or_status_invalid")
    if str(manifest.get("arm_key", "")) != str(task.get("arm_key", "")):
        reasons.append("arm_key_mismatch")
    if str(manifest.get("setting", "")) != E5_FORMAL_SETTING:
        reasons.append("setting_mismatch")
    try:
        if int(manifest.get("seed", -1)) != int(task.get("seed", -2)):
            reasons.append("seed_mismatch")
    except (TypeError, ValueError):
        reasons.append("seed_malformed")
    for field in ("dataset_identity", "device_identity", "split_row_fingerprints"):
        if json.dumps(manifest.get(field), sort_keys=True) != json.dumps(config.get(field), sort_keys=True):
            reasons.append("config_manifest_{}".format(field))
    if dataset_identity and str(manifest.get("dataset_identity", "")) != normalise_dataset_path(dataset_identity):
        reasons.append("dataset_identity_invalid")
    if device_identity and str(manifest.get("device_identity", "")) != str(device_identity):
        reasons.append("device_identity_invalid")
    if config.get("arm_key") != manifest.get("arm_key"):
        reasons.append("config_arm_identity_invalid")
    if config.get("schedule_registry_hash") != manifest.get("schedule_registry_hash"):
        reasons.append("schedule_registry_hash_invalid")
    inventory = _read_json(os.path.join(run_dir, "nonfinite_inventory.json"))
    if int(inventory.get("unexplained_count", 0)) != 0:
        reasons.append("unexplained_nonfinite")
    schedule_rows = _read_csv(os.path.join(run_dir, "schedule_epoch.csv"))
    shares = manifest.get("domain_shares", [])
    if formal and len(schedule_rows) != E5_EPOCHS:
        reasons.append("schedule_epoch_count_invalid")
    for row in schedule_rows:
        try:
            epoch = int(row.get("epoch", -1))
            alpha = float(row.get("alpha"))
            if abs(alpha - arm_alpha(str(task.get("arm_key")), epoch)) > 1.0e-10:
                reasons.append("schedule_alpha_mismatch")
            coeff = [float(row.get("coefficient_cell_{}".format(cell))) for cell in range(6)]
            expected = scheduled_coefficients_numpy(str(task.get("arm_key")), epoch, shares, 2)
            if not np.allclose(coeff, expected, atol=1.0e-7, rtol=0):
                reasons.append("schedule_coefficient_mismatch")
            if abs(float(row.get("coefficient_sum")) - 6.0) > 1.0e-6:
                reasons.append("schedule_coefficient_sum_invalid")
        except (TypeError, ValueError, IndexError):
            reasons.append("schedule_row_malformed")
            break
    metrics = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    selected = [row for row in metrics if row.get("selected_checkpoint") in ("1", 1)]
    cells = set()
    for row in selected:
        try:
            cells.add(int(row.get("cell", -1)))
        except (TypeError, ValueError):
            cells.add(-1)
    if len(selected) != 6 or cells != set(range(6)):
        reasons.append("selected_checkpoint_incomplete")
    else:
        for row in selected:
            for field in ("validation_auc", "validation_logloss", "test_auc", "test_logloss"):
                if _float(row.get(field)) is None:
                    reasons.append("selected_checkpoint_nonfinite")
                    break
    observed = manifest.get("observed_epochs")
    if formal:
        if manifest.get("formal_valid") is not True or config.get("formal_valid") is not True:
            reasons.append("formal_invalid")
        if observed != list(range(E5_EPOCHS)):
            reasons.append("observed_epochs_invalid")
        if manifest.get("fixed_epoch_budget") is not True or config.get("fixed_epoch_budget") is not True:
            reasons.append("fixed_budget_invalid")
        if manifest.get("oracle_audit", {}).get("complete") is not True:
            reasons.append("oracle_coverage_incomplete")
    return not reasons, reasons


def done_set(
    master: str = DEFAULT_MASTER, raw_root: str = DEFAULT_RAW_ROOT,
    *, dataset_identity: Optional[str] = None, device_identity: Optional[str] = None,
) -> set:
    done = set()
    latest = _latest_rows(master)
    tasks = build_tasks(raw_root)
    for task in tasks:
        key = _task_key(task)
        if latest.get(key, {}).get("status") not in ("OK",):
            continue
        valid, _ = validate_run_artifacts(task, formal=True,
                                          dataset_identity=dataset_identity,
                                          device_identity=device_identity)
        if not valid:
            continue
        block = compare_e5_block(
            {arm: os.path.join(raw_root, arm, E5_FORMAL_SETTING,
                               "seed{}".format(int(task["seed"]))) for arm in E5_ARM_KEYS},
            require_formal=True)
        if block.get("status") == "VALID":
            done.add(key)
    return done


def _set_block_status(task: Mapping[str, object], raw_root: str, block: Mapping[str, object]) -> None:
    seed = int(task["seed"])
    for arm in E5_ARM_KEYS:
        path = os.path.join(raw_root, arm, E5_FORMAL_SETTING, "seed{}".format(seed), "run_manifest.json")
        manifest = _read_json(path)
        if not manifest:
            continue
        manifest["block_status"] = block.get("status", "PENDING")
        manifest["block_valid"] = block.get("valid")
        manifest["block_comparison"] = dict(block)
        write_json(path, manifest)


def command_for_task(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E5_run.py"),
        "--arm_key", str(task["arm_key"]), "--setting", E5_FORMAL_SETTING,
        "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
        "--dataset_path", str(args.dataset_path), "--device", str(args.device),
        "--batch_size", str(E5_FORMAL_PROTOCOL["batch_size"]), "--epoch", str(E5_EPOCHS),
        "--learning_rate", str(E5_FORMAL_PROTOCOL["learning_rate"]),
        "--learning_rate_darts", str(E5_FORMAL_PROTOCOL["learning_rate_darts"]),
        "--weight_decay", str(E5_FORMAL_PROTOCOL["weight_decay"]),
        "--expert_num", str(E5_FORMAL_PROTOCOL["expert_num"]),
        "--probe_interval", str(E5_FORMAL_PROTOCOL["probe_interval"]), "--instrument", "1",
        "--earlystop_patience", str(E5_FORMAL_PROTOCOL["earlystop_patience"]),
        "--role_sizes", str(E5_FORMAL_PROTOCOL["role_sizes"]),
        "--oracle_root", str(args.oracle_root), "--legacy_oracle_root", str(args.legacy_oracle_root),
    ]


def _runner_environment() -> Dict[str, str]:
    env = os.environ.copy()
    bundled = os.path.join(os.path.dirname(ROOT), ".venv", "Lib", "site-packages")
    if os.path.isdir(bundled):
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bundled + (os.pathsep + old if old else "")
    return env


def _finalize(
    task: Mapping[str, object], return_code: int, wall: float, output: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    valid, reasons = validate_run_artifacts(
        task, formal=True,
        dataset_identity=normalise_dataset_path(args.dataset_path),
        device_identity=str(args.device))
    block = compare_e5_block(
        {arm: os.path.join(args.raw_root, arm, E5_FORMAL_SETTING,
                           "seed{}".format(int(task["seed"]))) for arm in E5_ARM_KEYS},
        require_formal=True) if valid else {"status": "PENDING", "valid": None}
    if block.get("status") == "INVALID":
        reasons.append("block_invalid")
    status = "OK" if return_code == 0 and valid and block.get("status") != "INVALID" else "FAILED"
    if block.get("status") == "INVALID":
        status = "INVALID_BLOCK"
    manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
    note = list(reasons)
    if return_code != 0:
        note.append("process_failed")
    if output.strip():
        note.append(output.strip().splitlines()[-1][:300])
    row = {
        "stage": "E5", "arm_key": task["arm_key"], "setting": task["setting"],
        "seed": task["seed"], "status": status, "return_code": int(return_code),
        "run_dir": task["run_dir"], "stdout_log": task["stdout_log"],
        "wall_seconds": "{:.3f}".format(wall), "best_epoch": manifest.get("best_epoch", ""),
        "fixed_final_epoch": manifest.get("fixed_final_epoch", ""),
        "observed_epoch_count": len(manifest.get("observed_epochs", [])),
        "total_main_steps": manifest.get("total_main_steps", ""),
        "total_architecture_steps": manifest.get("total_architecture_steps", ""),
        "shared_gradient_step_count": manifest.get("shared_gradient_step_count", ""),
        "shared_gradient_epoch_count": manifest.get("shared_gradient_epoch_count", ""),
        "oracle_coverage": manifest.get("oracle_coverage", ""),
        "formal_valid": manifest.get("formal_valid", False),
        "block_status": block.get("status", "PENDING"), "block_valid": block.get("valid"),
        "note": "; ".join(note),
    }
    _append_row(args.master, row)
    if block.get("status") in ("VALID", "INVALID"):
        _set_block_status(task, args.raw_root, block)
    return row


def _append_row(path: str, row: Mapping[str, object]) -> None:
    old = _read_csv(path) if os.path.exists(path) else []
    old.append({field: row.get(field, "") for field in MASTER_FIELDS})
    _write_csv(path, old, MASTER_FIELDS)


def _run_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace) -> int:
    failed = False
    os.makedirs(args.raw_root, exist_ok=True)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_task(task, args)
        print("[stage_E5][launch]", " ".join(command), flush=True)
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
        row = _finalize(task, int(process.returncode), time.time() - started, output, args)
        print("[stage_E5][done]", task["arm_key"], task["seed"], row["status"], flush=True)
        if row["status"] in ("FAILED", "INVALID_BLOCK"):
            failed = True
            break
    return 1 if failed else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E5 fixed 20-run orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--legacy_oracle_root", default=DEFAULT_LEGACY_ORACLE_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_runs", type=int, default=10**9)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.max_runs < 0:
        raise ValueError("--max_runs must be non-negative")
    expected_tasks = build_tasks(args.raw_root)
    valid_matrix, matrix_reasons = validate_matrix_tasks(expected_tasks)
    if not valid_matrix:
        raise AssertionError("internal E5 matrix invalid: {}".format(matrix_reasons))
    expected_rows = build_expected_oracle_split_fingerprints(args.dataset_path)
    oracle_audit = audit_oracles(
        args.oracle_root, args.legacy_oracle_root,
        dataset_path=args.dataset_path, device_identity=str(args.device),
        expected_split_fingerprints=expected_rows)
    print("[stage_E5] oracle audit: 12 expected; reused:",
          sum(1 for item in oracle_audit["domains"] if item.get("status") == "reused"), flush=True)
    if not oracle_audit.get("complete"):
        print("[stage_E5] oracle audit incomplete; E5 queue not launched", flush=True)
        return 1
    completed = done_set(
        args.master, args.raw_root,
        dataset_identity=normalise_dataset_path(args.dataset_path), device_identity=str(args.device))
    pending = [task for task in expected_tasks if _task_key(task) not in completed]
    print("[stage_E5] fixed joint matrix: 5 arms x 4 seeds = 20", flush=True)
    print("[stage_E5] pending joint:", len(pending), flush=True)
    if args.dry_run:
        for task in pending[:args.max_runs]:
            print("[stage_E5][dry-run]", " ".join(command_for_task(task, args)), flush=True)
        return 0
    return _run_queue(pending[:args.max_runs], args)


if __name__ == "__main__":
    raise SystemExit(main())
