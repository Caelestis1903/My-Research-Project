"""Resume-safe Stage E6 matrix orchestration.

The formal queue is closed at six arms x one canonical setting x four seeds.
This module audits and reuses the twelve formal E3 oracle references, launches
only missing E6 runs, and compares arm identity/batch support.  It records
statuses; it does not interpret scientific results.
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
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)

from run_movielens import get_movielens_data_rank_multidomain  # noqa: E402
from stage_E1_instrumentation import write_json  # noqa: E402
from stage_E3_run import oracle_split_row_fingerprints  # noqa: E402
from stage_E3_weighting import (  # noqa: E402
    normalise_dataset_path,
    resolve_oracle_metrics,
    validate_oracle_artifact,
)
from stage_E6_calibration import (  # noqa: E402
    E6_ARM_KEYS,
    E6_BETA,
    E6_EXPECTED_SEEDS,
    E6_FORMAL_PROTOCOL,
    E6_FORMAL_SETTING,
    E6_ROLE_SIZES,
    normalise_shares,
    registry_hash,
)


DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E6_raw_results")
DEFAULT_MASTER = os.path.join(HERE, "stage_E6_runs.csv")
DEFAULT_SUMMARY = os.path.join(HERE, "stage_E6_summary.csv")
DEFAULT_ORACLE_ROOT = os.path.join(HERE, "stage_E3_raw_results", "oracle_reference")
DEFAULT_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable

REQUIRED_ARTIFACTS = (
    "config.json", "run_manifest.json", "stdout.log", "metrics_epoch.csv",
    "metrics_summary.csv", "controller_step.csv", "controller_epoch.csv",
    "shared_gradient_step.csv", "shared_gradient_epoch.csv", "batch_trace.csv",
    "nonfinite_inventory.json", "model.pth",
)
MASTER_FIELDS = [
    "stage", "arm_key", "setting", "seed", "status", "return_code", "run_dir",
    "stdout_log", "wall_seconds", "best_epoch", "stopped_epoch", "observed_epoch_count",
    "controller_step_count", "shared_gradient_step_count", "shared_gradient_epoch_count",
    "block_status", "block_valid", "note",
]


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for arm_key in E6_ARM_KEYS:
        for seed in E6_EXPECTED_SEEDS:
            run_dir = os.path.join(os.path.abspath(raw_root), arm_key, E6_FORMAL_SETTING,
                                    "seed{}".format(int(seed)))
            tasks.append({
                "stage": "E6", "arm_key": arm_key, "setting": E6_FORMAL_SETTING,
                "seed": int(seed), "run_dir": run_dir,
                "stdout_log": os.path.join(run_dir, "stdout.log"),
            })
    return tasks


def validate_matrix_tasks(tasks: Sequence[Mapping[str, object]]) -> Tuple[bool, List[str]]:
    expected = {(arm, E6_FORMAL_SETTING, int(seed))
                for arm in E6_ARM_KEYS for seed in E6_EXPECTED_SEEDS}
    actual = {(str(task.get("arm_key", "")), str(task.get("setting", "")),
               int(task.get("seed", -1))) for task in tasks}
    reasons: List[str] = []
    if len(tasks) != len(expected):
        reasons.append("task_count:{}".format(len(tasks)))
    if actual != expected:
        reasons.append("task_identity_mismatch")
    return not reasons, reasons


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


def _append_row(path: str, row: Mapping[str, object]) -> None:
    rows = _read_csv(path)
    rows.append(dict(row))
    _write_csv(path, rows, MASTER_FIELDS)


def _task_key(task: Mapping[str, object]) -> Tuple[str, str, int]:
    return str(task["arm_key"]), str(task["setting"]), int(task["seed"])


def _normalise(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def validate_run_artifacts(
    task: Mapping[str, object], *, formal: bool = True,
    dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Check one E6 run's identity and required raw artifacts."""

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
    if manifest.get("stage") != "E6" or manifest.get("status") != "OK":
        reasons.append("manifest_stage_or_status_invalid")
    if manifest.get("arm_key") != task.get("arm_key") or config.get("arm_key") != task.get("arm_key"):
        reasons.append("arm_identity_invalid")
    if manifest.get("setting") != task.get("setting") or config.get("setting") != task.get("setting"):
        reasons.append("setting_identity_invalid")
    try:
        if int(manifest.get("seed", -1)) != int(task.get("seed", -2)):
            reasons.append("seed_identity_invalid")
    except (TypeError, ValueError):
        reasons.append("seed_identity_invalid")
    if _normalise(manifest.get("formal_protocol")) != _normalise(config.get("formal_protocol")):
        reasons.append("formal_protocol_config_mismatch")
    if manifest.get("controller_registry_hash") not in ("", None, registry_hash()):
        reasons.append("controller_registry_hash_invalid")
    if _normalise(manifest.get("split_row_fingerprints")) != _normalise(config.get("split_row_fingerprints")):
        reasons.append("split_row_identity_invalid")
    if dataset_identity is not None and str(manifest.get("dataset_identity", "")) != str(dataset_identity):
        reasons.append("dataset_identity_invalid")
    if device_identity is not None and str(manifest.get("device_identity", "")) != str(device_identity):
        reasons.append("device_identity_invalid")
    if manifest.get("train_domain_counts") != {"0": E6_ROLE_SIZES["L"],
                                                "1": E6_ROLE_SIZES["M"],
                                                "2": E6_ROLE_SIZES["S"]}:
        # JSON object keys are strings, while the run's in-memory dictionary
        # is integer-keyed; accept either representation but not a reduced set.
        try:
            observed = {int(k): int(v) for k, v in dict(manifest.get("train_domain_counts", {})).items()}
        except (TypeError, ValueError):
            observed = {}
        if observed != {0: E6_ROLE_SIZES["L"], 1: E6_ROLE_SIZES["M"], 2: E6_ROLE_SIZES["S"]}:
            reasons.append("train_count_identity_invalid")
    observed_epochs = manifest.get("observed_epochs", [])
    if not isinstance(observed_epochs, list) or not observed_epochs or int(observed_epochs[0]) != 0:
        reasons.append("observed_epochs_invalid")
    inventory = _read_json(os.path.join(run_dir, "nonfinite_inventory.json"))
    if int(inventory.get("unexplained_count", -1)) != 0:
        reasons.append("unexplained_nonfinite")
    selected = []
    for row in _read_csv(os.path.join(run_dir, "metrics_epoch.csv")):
        try:
            if int(row.get("selected_checkpoint", 0)) == 1:
                selected.append(row)
        except (TypeError, ValueError):
            pass
    selected_cells = {int(row.get("cell", -1)) for row in selected}
    if len(selected) != 6 or selected_cells != set(range(6)):
        reasons.append("selected_checkpoint_incomplete")
    controller_rows = _read_csv(os.path.join(run_dir, "controller_step.csv"))
    if not controller_rows:
        reasons.append("controller_trace_missing")
    else:
        for row in controller_rows:
            try:
                applied = json.loads(row.get("applied_c_d", "[]"))
                if len(applied) != 6 or not np.isclose(sum(applied[:3]), 3.0, atol=1e-7):
                    reasons.append("coefficient_sum_invalid")
                    break
            except (TypeError, ValueError, json.JSONDecodeError):
                reasons.append("controller_trace_malformed")
                break
    if formal:
        if manifest.get("formal_valid") is not True or config.get("formal_valid") is not True:
            reasons.append("formal_invalid")
        if _normalise(manifest.get("formal_protocol")) != _normalise(E6_FORMAL_PROTOCOL):
            reasons.append("formal_protocol_invalid")
        if config.get("epoch") != E6_FORMAL_PROTOCOL["epoch"]:
            reasons.append("epoch_budget_invalid")
    return not reasons, reasons


def _trace_by_support(run_dir: str) -> Tuple[Dict[Tuple[str, int, int], Mapping[str, str]], List[str]]:
    traces: Dict[Tuple[str, int, int], Mapping[str, str]] = {}
    reasons: List[str] = []
    rows = _read_csv(os.path.join(run_dir, "batch_trace.csv"))
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind not in ("train", "architecture"):
            continue
        try:
            support = (kind, int(row.get("epoch", -1)), int(row.get("step", -1)))
        except (TypeError, ValueError):
            reasons.append("malformed_batch_trace")
            continue
        if support in traces:
            reasons.append("duplicate_batch_trace")
        traces[support] = row
    if not traces:
        reasons.append("batch_trace_missing_or_empty")
    return traces, reasons


def compare_e6_block(run_dirs: Mapping[str, str], *, require_formal: bool = False) -> Dict[str, object]:
    """Compare one seed's six arms on shared identity and common batch support."""

    missing = [key for key in E6_ARM_KEYS if not os.path.isdir(str(run_dirs.get(key, "")))]
    if missing:
        return {"status": "PENDING", "valid": None, "missing_arm_keys": missing}
    manifests = {key: _read_json(os.path.join(str(run_dirs[key]), "run_manifest.json"))
                 for key in E6_ARM_KEYS}
    configs = {key: _read_json(os.path.join(str(run_dirs[key]), "config.json"))
               for key in E6_ARM_KEYS}
    reasons: List[Dict[str, object]] = []
    if any(not manifests[key] or not configs[key] for key in E6_ARM_KEYS):
        return {"status": "INVALID", "valid": False, "reason": "manifest_or_config_missing"}
    base = manifests[E6_ARM_KEYS[0]]
    identity_fields = (
        "seed", "setting", "dataset", "dataset_identity", "device_identity", "domain_num",
        "task_num", "role_by_domain", "role_sizes", "train_domain_counts", "domain_shares",
        "split_counts", "split_row_fingerprints", "positive_rates", "initial_parameter_state_hash",
        "partition", "model_identity", "formal_protocol", "oracle_audit",
    )
    for key in E6_ARM_KEYS:
        if manifests[key].get("arm_key") != key or configs[key].get("arm_key") != key:
            reasons.append({"arm_key": key, "reason": "arm_identity_mismatch"})
        if require_formal and (manifests[key].get("formal_valid") is not True
                               or configs[key].get("formal_valid") is not True):
            reasons.append({"arm_key": key, "reason": "formal_invalid"})
        for field in identity_fields:
            if _normalise(manifests[key].get(field)) != _normalise(base.get(field)):
                reasons.append({"arm_key": key, "field": field, "reason": "identity_mismatch"})
    traces: Dict[str, Dict[Tuple[str, int, int], Mapping[str, str]]] = {}
    for key in E6_ARM_KEYS:
        traces[key], trace_reasons = _trace_by_support(str(run_dirs[key]))
        reasons.extend({"arm_key": key, "reason": item} for item in trace_reasons)
    first = E6_ARM_KEYS[0]
    common = set(traces[first])
    # Every arm may legally stop early, but its observed support must be a
    # prefix of the longest common protocol order (no missing interior
    # batches).  The first registry arm is not assumed to be the longest.
    def ordered(support: Mapping[Tuple[str, int, int], Mapping[str, str]]) -> List[Tuple[str, int, int]]:
        return sorted(support, key=lambda item: (item[1], 0 if item[0] == "train" else 1, item[2]))
    orders = {key: ordered(traces[key]) for key in E6_ARM_KEYS}
    ref_order = max(orders.values(), key=len)
    for key in E6_ARM_KEYS:
        candidate_order = orders[key]
        if candidate_order != ref_order[:len(candidate_order)]:
            reasons.append({"arm_key": key, "reason": "batch_support_nonprefix"})
        common &= set(traces[key])
    for support in sorted(common, key=lambda item: (item[1], 0 if item[0] == "train" else 1, item[2])):
        reference = traces[first][support]
        for key in E6_ARM_KEYS[1:]:
            candidate = traces[key][support]
            for field in ("batch_fingerprint", "batch_size", "domain_counts", "task_counts"):
                if str(candidate.get(field, "")) != str(reference.get(field, "")):
                    reasons.append({"arm_key": key, "support": support, "field": field,
                                    "reason": "batch_identity_mismatch"})
    # The plain sample and matched estimator control must have identical main
    # losses on common train supports up to the preregistered numeric tolerance.
    sample = traces.get("sample_static", {})
    control = traces.get("sample_ema_control", {})
    for support in set(sample).intersection(control):
        if support[0] != "train":
            continue
        try:
            if abs(float(sample[support].get("train_loss", "nan")) -
                   float(control[support].get("train_loss", "nan"))) > 1.0e-7:
                reasons.append({"support": support, "reason": "matched_control_loss_mismatch"})
                break
        except (TypeError, ValueError):
            reasons.append({"support": support, "reason": "matched_control_loss_malformed"})
            break
    common_epochs = sorted({int(support[1]) for support in common})
    common_trace_count = len(common)
    return {
        "status": "VALID" if not reasons else "INVALID", "valid": not reasons,
        "reasons": reasons, "arm_keys": list(E6_ARM_KEYS),
        "trace_count": {key: len(traces[key]) for key in E6_ARM_KEYS},
        "common_support_count": common_trace_count,
        "common_trace_count": common_trace_count,
        "common_epochs": common_epochs,
        "last_common_epoch": (common_epochs[-1] if common_epochs else None),
        "common_support_valid": not reasons,
        "observed_epochs": {key: manifests[key].get("observed_epochs", []) for key in E6_ARM_KEYS},
    }


def _block_for_seed(seed: int, raw_root: str, *, require_formal: bool = True) -> Dict[str, object]:
    dirs = {key: os.path.join(raw_root, key, E6_FORMAL_SETTING, "seed{}".format(int(seed)))
            for key in E6_ARM_KEYS}
    return compare_e6_block(dirs, require_formal=require_formal)


def _set_block_status(seed: int, raw_root: str, block: Mapping[str, object]) -> None:
    for key in E6_ARM_KEYS:
        path = os.path.join(raw_root, key, E6_FORMAL_SETTING, "seed{}".format(int(seed)), "run_manifest.json")
        manifest = _read_json(path)
        if not manifest:
            continue
        manifest["block_status"] = block.get("status", "PENDING")
        manifest["block_valid"] = block.get("valid")
        manifest["block_comparison"] = dict(block)
        write_json(path, manifest)


def done_set(master: str, raw_root: str = DEFAULT_RAW_ROOT, *, formal: bool = True,
             dataset_identity: Optional[str] = None, device_identity: Optional[str] = None) -> set:
    done = set()
    latest = {}
    for row in _read_csv(master):
        latest[_task_key(row)] = row
    for task in build_tasks(raw_root):
        key = _task_key(task)
        if latest.get(key, {}).get("status") not in ("OK", "VALID"):
            continue
        valid, _ = validate_run_artifacts(task, formal=formal,
                                          dataset_identity=dataset_identity,
                                          device_identity=device_identity)
        if not valid:
            continue
        block = _block_for_seed(int(task["seed"]), raw_root, require_formal=formal)
        if block.get("status") == "VALID":
            done.add(key)
    return done


def command_for_task(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_E6_run.py"),
        "--arm_key", str(task["arm_key"]), "--setting", E6_FORMAL_SETTING,
        "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
        "--dataset_path", str(args.dataset_path), "--device", str(args.device),
        "--batch_size", str(args.batch_size), "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--probe_interval", "1", "--instrument", "1",
        "--earlystop_patience", str(args.earlystop_patience),
        "--oracle_root", str(args.oracle_root),
    ]


def _runner_environment() -> Dict[str, str]:
    env = os.environ.copy()
    bundled = os.path.join(os.path.dirname(ROOT), ".venv", "Lib", "site-packages")
    if os.path.isdir(bundled):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bundled + (os.pathsep + existing if existing else "")
    return env


def audit_oracles(oracle_root: str, *, dataset_path: str, device: str) -> Tuple[bool, Dict[str, object]]:
    """Audit the twelve E3 references without launching new oracle runs."""

    dataset_identity = normalise_dataset_path(dataset_path)
    try:
        _dense, _sparse, x_train, y_train, x_val, y_val, x_test, y_test, _domain_num = \
            get_movielens_data_rank_multidomain("all", "all", dataset_path)
    except Exception as exc:  # pragma: no cover - environment-specific loader errors
        return False, {"status": "MISSING", "reason": "dataset_audit_failed:{}".format(exc)}
    expected: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    for seed in E6_EXPECTED_SEEDS:
        per_domain = oracle_split_row_fingerprints(
            x_train, y_train, x_val, y_val, x_test, y_test,
            E6_FORMAL_SETTING, int(seed), E6_ROLE_SIZES)
        for domain, rows in per_domain.items():
            expected[(E6_FORMAL_SETTING, int(seed), int(domain))] = dict(rows)
    statuses: List[Dict[str, object]] = []
    all_valid = True
    for seed in E6_EXPECTED_SEEDS:
        expected_for_seed = {domain: expected[(E6_FORMAL_SETTING, int(seed), domain)]
                             for domain in range(3)}
        metrics, audit = resolve_oracle_metrics(
            [oracle_root], E6_FORMAL_SETTING, int(seed),
            role_sizes=E6_ROLE_SIZES, domain_num=3, task_num=2,
            dataset_identity=dataset_identity, device_identity=str(device),
            expected_split_row_fingerprints=expected_for_seed)
        coverage = int(audit.get("coverage_cells", 0))
        valid = coverage == 6 and all(bool(value.get("coverage")) for value in metrics.values())
        statuses.append({"setting": E6_FORMAL_SETTING, "seed": int(seed),
                         "coverage_cells": coverage, "valid": valid, "audit": audit})
        all_valid = all_valid and valid
    return all_valid, {"status": "REUSED" if all_valid else "MISSING", "references": statuses,
                       "expected_count": 12, "coverage_cells": 24}


def _finalize_task(task: Mapping[str, object], return_code: int, wall: float,
                   output: str, master: str, raw_root: str, *, formal: bool,
                   dataset_identity: Optional[str], device_identity: Optional[str]) -> Dict[str, object]:
    valid, reasons = validate_run_artifacts(task, formal=formal,
                                            dataset_identity=dataset_identity,
                                            device_identity=device_identity)
    block = _block_for_seed(int(task["seed"]), raw_root, require_formal=formal)
    if valid and block.get("status") == "INVALID":
        valid = False
        reasons.append("block_invalid")
    status = "OK" if return_code == 0 and valid else "FAILED"
    if block.get("status") == "INVALID":
        status = "INVALID_BLOCK"
    _set_block_status(int(task["seed"]), raw_root, block)
    manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
    if return_code != 0:
        reasons.append("process_failed")
    if output.strip():
        reasons.append(output.strip().splitlines()[-1][:300])
    row = {
        "stage": "E6", "arm_key": task["arm_key"], "setting": task["setting"],
        "seed": task["seed"], "status": status, "return_code": int(return_code),
        "run_dir": task["run_dir"], "stdout_log": task["stdout_log"],
        "wall_seconds": "{:.3f}".format(wall), "best_epoch": manifest.get("best_epoch", ""),
        "stopped_epoch": manifest.get("stopped_epoch", ""),
        "observed_epoch_count": len(manifest.get("observed_epochs", [])),
        "controller_step_count": manifest.get("controller_step_count", ""),
        "shared_gradient_step_count": manifest.get("shared_gradient_step_count", ""),
        "shared_gradient_epoch_count": manifest.get("shared_gradient_epoch_count", ""),
        "block_status": block.get("status", "PENDING"), "block_valid": block.get("valid"),
        "note": "; ".join(reasons),
    }
    _append_row(master, row)
    return row


def _run_queue(tasks: Sequence[Mapping[str, object]], args: argparse.Namespace, *, formal: bool) -> int:
    failed = False
    os.makedirs(args.raw_root, exist_ok=True)
    dataset_identity = normalise_dataset_path(args.dataset_path)
    for task in tasks:
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        command = command_for_task(task, args)
        print("[stage_E6][launch]", " ".join(command), flush=True)
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
        row = _finalize_task(task, int(process.returncode), time.time() - started,
                             output, args.master, args.raw_root, formal=formal,
                             dataset_identity=dataset_identity, device_identity=str(args.device))
        print("[stage_E6][done]", task["arm_key"], task["seed"], row["status"], flush=True)
        failed = failed or row["status"] != "OK"
        if failed:
            break
    return 1 if failed else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E6 six-arm x four-seed orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--learning_rate_darts", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--earlystop_patience", type=int, default=10)
    parser.add_argument("--max_runs", type=int, default=10**9)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.max_runs < 0:
        raise ValueError("--max_runs must be non-negative")
    oracle_ok, oracle_audit = audit_oracles(
        args.oracle_root, dataset_path=args.dataset_path, device=args.device)
    print("[stage_E6] E3 oracle references: 12 expected; status:",
          oracle_audit.get("status"), flush=True)
    if not oracle_ok:
        write_json(os.path.join(os.path.dirname(os.path.abspath(args.master)),
                                "stage_E6_oracle_audit.json"), oracle_audit)
        print("[stage_E6] formal matrix not launched: E3 oracle audit incomplete", flush=True)
        return 1
    tasks = build_tasks(args.raw_root)
    matrix_ok, matrix_reasons = validate_matrix_tasks(tasks)
    if not matrix_ok:
        raise RuntimeError("closed E6 matrix invalid: {}".format(matrix_reasons))
    dataset_identity = normalise_dataset_path(args.dataset_path)
    completed = done_set(args.master, args.raw_root, formal=True,
                         dataset_identity=dataset_identity, device_identity=args.device)
    pending = [task for task in tasks if _task_key(task) not in completed]
    print("[stage_E6] fixed joint matrix: 6 arms x 1 setting x 4 seeds = 24", flush=True)
    print("[stage_E6] pending joint:", len(pending), flush=True)
    if args.dry_run:
        for task in pending[:args.max_runs]:
            print("[stage_E6][dry-run]", " ".join(command_for_task(task, args)), flush=True)
        return 0
    return _run_queue(pending[:args.max_runs], args, formal=True)


if __name__ == "__main__":
    raise SystemExit(main())
