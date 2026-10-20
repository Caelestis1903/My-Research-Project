"""Resume-safe launcher and protocol auditor for the 84-run Stage S0 matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S0_protocol import (  # noqa: E402
    ARM_SPECS, COMMON_EPOCHS, FORMAL_PROTOCOL, ROTATIONS, SEEDS,
    protocol_mismatches,
)

DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_S0_raw_results")
MASTER_FIELDS = ["attempt_id", "formal_run_key", "arm", "rotation", "seed", "status",
                 "return_code", "wall_seconds", "run_dir", "failure_archive", "message"]
REQUIRED_FILES = ("config.json", "run_manifest.json", "metrics_epoch.csv",
                  "metrics_summary.csv", "momentum_step.csv", "batch_trace.csv",
                  "selected_checkpoint.pth", "final_checkpoint.pth", "optimizer_state.pth",
                  "architecture_optimizer_state.pth")
PAIR_FIELDS = ("dataset_identity", "device_identity", "split_row_fingerprints", "selected_training_indices_hash",
               "code_fingerprint", "initial_parameter_state_hash", "batch_order_hash", "examples_processed",
               "optimizer_steps", "scheduler_steps", "architecture_optimizer_steps",
               "architecture_batch_order_hash", "architecture_batch_fingerprint_count",
               "architecture_examples_processed", "train_domain_counts", "domain_shares")


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: str, value: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(temporary, path)


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _formal_values(args: argparse.Namespace) -> Dict[str, object]:
    return {field: getattr(args, field) for field in FORMAL_PROTOCOL}


def assert_formal_launcher(args: argparse.Namespace) -> None:
    mismatches = protocol_mismatches(_formal_values(args))
    if mismatches:
        raise ValueError("formal S0 launcher protocol mismatch: {}".format(
            json.dumps(mismatches, sort_keys=True)))


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fields=MASTER_FIELDS) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_tasks(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    tasks = []
    for arm in ARM_SPECS:
        for rotation in ROTATIONS:
            for seed in SEEDS:
                key = f"{arm}|{rotation}|{seed}"
                tasks.append({
                    "arm": arm, "rotation": rotation, "seed": seed,
                    "formal_run_key": key,
                    "run_dir": os.path.join(os.path.abspath(raw_root), arm, rotation,
                                            f"seed{seed}"),
                })
    keys = [str(task["formal_run_key"]) for task in tasks]
    if len(tasks) != 84 or len(keys) != len(set(keys)):
        raise AssertionError("S0 matrix must contain exactly 84 unique tasks")
    return tasks


def command_for(task: Mapping[str, object], args: argparse.Namespace) -> List[str]:
    return [
        str(args.python), os.path.join(HERE, "stage_S0_run.py"),
        "--arm", str(task["arm"]), "--rotation", str(task["rotation"]),
        "--seed", str(task["seed"]), "--run_dir", str(task["run_dir"]),
        "--device", str(args.device), "--dataset_path", str(args.dataset_path),
        "--oracle_root", str(args.oracle_root), "--role_sizes", str(args.role_sizes),
        "--batch_size", str(args.batch_size), "--epoch", str(args.epoch),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_darts", str(args.learning_rate_darts),
        "--weight_decay", str(args.weight_decay), "--expert_num", str(args.expert_num),
        "--shadow_patience", str(args.shadow_patience),
        "--require_oracles", str(args.require_oracles), "--instrument", str(args.instrument),
    ]


def validate_run(task: Mapping[str, object], *, require_formal: bool = True) -> Tuple[bool, List[str]]:
    run_dir = str(task["run_dir"])
    reasons: List[str] = []
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    config = _read_json(os.path.join(run_dir, "config.json"))
    if manifest.get("stage") != "S0" or manifest.get("status") != "OK":
        reasons.append("manifest_status")
    for field in ("arm", "rotation", "seed", "formal_run_key"):
        if str(manifest.get(field)) != str(task.get(field)):
            reasons.append("identity:" + field)
    for filename in REQUIRED_FILES:
        artifact = os.path.join(run_dir, filename)
        if not os.path.isfile(artifact) or os.path.getsize(artifact) <= 0:
            reasons.append("missing:" + filename)
    if task["arm"] in ("comp1_domain", "uniform_090", "uniform_p_tau", "uniform_eq_tau") \
            and (not os.path.isfile(os.path.join(run_dir, "momentum_state.pth"))
                 or os.path.getsize(os.path.join(run_dir, "momentum_state.pth")) <= 0):
        reasons.append("missing:momentum_state.pth")
    expected_epochs = _int(config.get("epoch"))
    if require_formal:
        mismatches = protocol_mismatches(config)
        if mismatches or config.get("formal_valid") is not True \
                or manifest.get("formal_valid") is not True:
            reasons.append("formal_protocol:" + json.dumps(mismatches, sort_keys=True))
        if config.get("formal_protocol") != FORMAL_PROTOCOL \
                or manifest.get("formal_protocol") != FORMAL_PROTOCOL:
            reasons.append("formal_protocol_snapshot")
    if _int(manifest.get("fixed_final_epoch")) != expected_epochs - 1:
        reasons.append("fixed_final_epoch")
    metrics = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
    if len(metrics) != expected_epochs * 6:
        reasons.append("metrics_cardinality")
    metric_keys = {(row.get("epoch"), row.get("domain"), row.get("task"), row.get("cell"))
                   for row in metrics}
    if len(metric_keys) != len(metrics):
        reasons.append("metrics_duplicate_cells")
    expected_metric_keys = {(str(epoch), str(domain), str(task), str(task * 3 + domain))
                            for epoch in range(expected_epochs)
                            for task in range(2) for domain in range(3)}
    if metric_keys != expected_metric_keys:
        reasons.append("metrics_epoch_cell_support")
    required_metric_finite = (
        "epoch", "domain", "task", "cell", "domain_size", "domain_share",
        "sample_weighted_train_loss", "validation_auc", "validation_logloss",
        "test_auc", "test_logloss", "oracle_auc", "oracle_logloss", "transfer_auc",
        "transfer_logloss", "raw_auc_nt", "raw_logloss_nt", "practical_auc_nt",
        "concordant_practical_nt", "selected_checkpoint", "fixed_final", "common_epoch",
    )
    if any(not _finite(row.get(field)) for row in metrics for field in required_metric_finite):
        reasons.append("metrics_nonfinite_or_missing")
    selected = [row for row in metrics if row.get("selected_checkpoint") == "1"]
    if len(selected) != 6 or len({row.get("epoch") for row in selected}) != 1:
        reasons.append("selected_checkpoint")
    observed_metric_epochs = {_int(row.get("epoch")) for row in metrics
                              if _finite(row.get("epoch"))}
    if set(COMMON_EPOCHS).difference(observed_metric_epochs):
        reasons.append("common_epoch_support")
    steps = _int(manifest.get("optimizer_steps"))
    if steps <= 0 or _int(manifest.get("examples_processed")) <= 0:
        reasons.append("budget_counts")
    train_counts = manifest.get("train_domain_counts", {})
    try:
        total_examples = sum(int(value) for value in train_counts.values())
        steps_per_epoch = int(math.ceil(total_examples / float(config.get("batch_size", 0))))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        total_examples, steps_per_epoch = -1, -1
    if steps != steps_per_epoch * expected_epochs \
            or _int(manifest.get("examples_processed")) != total_examples * expected_epochs \
            or _int(manifest.get("scheduler_steps")) != 0:
        reasons.append("formal_budget_exactness")
    ema_counts = [_int(value) for value in manifest.get("ema_update_counts", [])]
    is_ema = ARM_SPECS[str(task["arm"])]["kind"].startswith("ema")
    if is_ema and (len(ema_counts) != 3 or any(value != steps for value in ema_counts)):
        reasons.append("ema_update_counts")
    if not is_ema and (len(ema_counts) != 3 or any(value != 0 for value in ema_counts)):
        reasons.append("unexpected_ema_updates")
    momentum = _read_csv(os.path.join(run_dir, "momentum_step.csv"))
    if len(momentum) != 3 * steps:
        reasons.append("momentum_cardinality")
    if len({(row.get("epoch"), row.get("step"), row.get("domain")) for row in momentum}) \
            != len(momentum) or any(row.get("arm") != str(task["arm"])
                                    or row.get("rotation") != str(task["rotation"])
                                    or row.get("seed") != str(task["seed"])
                                    for row in momentum):
        reasons.append("momentum_identity_or_duplicates")
    momentum_always = ("epoch", "step", "domain", "raw_shared_gradient_norm",
                       "aggregate_shared_gradient_norm", "shared_update_norm",
                       "adam_first_moment_norm", "adam_second_moment_norm",
                       "ema_update_count", "learning_rate")
    if any(not _finite(row.get(field)) for row in momentum for field in momentum_always):
        reasons.append("momentum_nonfinite_or_missing")
    if is_ema:
        ema_fields = ("gamma", "ema_buffer_norm", "ema_to_raw_ratio", "startup_coefficient")
        if any(not _finite(row.get(field)) for row in momentum for field in ema_fields):
            reasons.append("ema_diagnostics_nonfinite_or_missing")
    batches = _read_csv(os.path.join(run_dir, "batch_trace.csv"))
    train_batches = [row for row in batches if row.get("kind") == "train"]
    architecture_batches = [row for row in batches if row.get("kind") == "architecture"]
    architecture_steps = _int(manifest.get("architecture_optimizer_steps"))
    if architecture_steps != expected_epochs:
        reasons.append("architecture_step_budget")
    if len(train_batches) != steps or len(architecture_batches) != architecture_steps \
            or len(batches) != steps + architecture_steps:
        reasons.append("batch_trace_cardinality")
    if _int(manifest.get("batch_fingerprint_count")) != len(train_batches):
        reasons.append("train_batch_manifest_count")
    if any(not row.get("batch_fingerprint") or not _finite(row.get("batch_size"))
           or not _finite(row.get("train_loss")) for row in batches):
        reasons.append("batch_trace_nonfinite_or_missing")
    if sum(int(float(row["batch_size"])) for row in train_batches if _finite(row.get("batch_size"))) \
            != _int(manifest.get("examples_processed")) \
            or sum(int(float(row["batch_size"])) for row in architecture_batches
                   if _finite(row.get("batch_size"))) \
            != _int(manifest.get("architecture_examples_processed")):
        reasons.append("batch_trace_example_totals")
    if _int(manifest.get("architecture_batch_fingerprint_count")) != architecture_steps \
            or _int(manifest.get("architecture_examples_processed")) <= 0 \
            or not manifest.get("architecture_batch_order_hash"):
        reasons.append("architecture_batch_audit")
    expected_shared_beta1 = float(ARM_SPECS[str(task["arm"])]["shared_beta1"])
    if not _finite(manifest.get("shared_beta1")) or \
            abs(float(manifest.get("shared_beta1")) - expected_shared_beta1) > 1e-12:
        reasons.append("shared_beta1")
    if not _finite(manifest.get("private_beta1")) or \
            abs(float(manifest.get("private_beta1")) - 0.9) > 1e-12:
        reasons.append("private_beta1")
    main_audit = manifest.get("main_optimizer_audit", {})
    groups = main_audit.get("groups", []) if isinstance(main_audit, dict) else []
    if len(groups) != 2 or groups[0].get("group_name") != "shared" \
            or groups[1].get("group_name") != "private" \
            or groups[0].get("betas") != [expected_shared_beta1, 0.999] \
            or groups[1].get("betas") != [0.9, 0.999] \
            or _int(main_audit.get("state_parameter_count")) != sum(
                _int(group.get("parameter_count"), 0) for group in groups) \
            or _int(main_audit.get("step_min")) != steps \
            or _int(main_audit.get("step_max")) != steps:
        reasons.append("main_optimizer_state_audit")
    arch_audit = manifest.get("architecture_optimizer_audit", {})
    arch_groups = arch_audit.get("groups", []) if isinstance(arch_audit, dict) else []
    if not isinstance(arch_audit, dict) or _int(arch_audit.get("step_min")) != architecture_steps \
            or _int(arch_audit.get("step_max")) != architecture_steps \
            or _int(arch_audit.get("state_parameter_count")) != sum(
                _int(group.get("parameter_count"), 0) for group in arch_groups):
        reasons.append("architecture_optimizer_state_audit")
    if config.get("sample_objective") != "w_d=D*p_d; sum over domain-task mean BCE":
        reasons.append("sample_objective")
    if config.get("control_family_selection_uses_test") is not False:
        reasons.append("test_selection_guard")
    if not all(bool(item.get("restored")) for item in manifest.get("rng_restore_evidence", [])):
        reasons.append("rng_restore")
    if manifest.get("oracle_audit", {}).get("complete") is not True:
        reasons.append("oracle_coverage")
    return not reasons, reasons


def audit_block(tasks: Sequence[Mapping[str, object]], *, require_formal: bool = True) -> Dict[str, object]:
    manifests: Dict[str, Dict[str, object]] = {}
    reasons: List[object] = []
    for task in tasks:
        ok, run_reasons = validate_run(task, require_formal=require_formal)
        if not ok:
            reasons.append({"arm": task["arm"], "reasons": run_reasons})
        manifests[str(task["arm"])] = _read_json(os.path.join(str(task["run_dir"]),
                                                               "run_manifest.json"))
    if reasons:
        return {"status": "INVALID", "valid": False,
                "rotation": tasks[0]["rotation"], "seed": tasks[0]["seed"],
                "reasons": reasons, "paired_fields": list(PAIR_FIELDS)}
    base = manifests["sample_adam"]
    for arm, manifest in manifests.items():
        for field in PAIR_FIELDS:
            if json.dumps(manifest.get(field), sort_keys=True) != json.dumps(base.get(field), sort_keys=True):
                reasons.append({"arm": arm, "field": field, "reason": "pair_mismatch"})
    # B/C share the same optimizer and EMA path; only gamma is arm-varying.
    for arm in ("comp1_domain", "uniform_090", "uniform_p_tau", "uniform_eq_tau"):
        manifest = manifests[arm]
        if float(manifest.get("shared_beta1")) != 0.0 or manifest.get("ema_update_counts") != \
                manifests["comp1_domain"].get("ema_update_counts"):
            reasons.append({"arm": arm, "reason": "B_C_path_contract"})
    # A/D share the no-EMA path and may differ only in shared beta1.
    for arm in ("sample_adam", "adam_p_tau", "adam_eq_tau"):
        if any(int(v) != 0 for v in manifests[arm].get("ema_update_counts", [])):
            reasons.append({"arm": arm, "reason": "A_D_path_contract"})
    return {"status": "VALID" if not reasons else "INVALID", "valid": not reasons,
            "rotation": tasks[0]["rotation"], "seed": tasks[0]["seed"], "reasons": reasons,
            "paired_fields": list(PAIR_FIELDS)}


def audit_all_blocks(tasks: Sequence[Mapping[str, object]], require_formal=True) -> List[Dict[str, object]]:
    blocks = []
    for rotation in ROTATIONS:
        for seed in SEEDS:
            group = [task for task in tasks if task["rotation"] == rotation and task["seed"] == seed]
            blocks.append(audit_block(group, require_formal=require_formal))
    return blocks


def _archive_failed(run_dir: str, raw_root: str, key: str) -> str:
    if not os.path.exists(run_dir):
        return ""
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    destination = os.path.join(os.path.abspath(raw_root), "failed_attempts",
                               key.replace("|", "__"), stamp)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.move(run_dir, destination)
    return destination


def runtime_preflight(args: argparse.Namespace) -> None:
    code = ("import torch; "
            "assert hasattr(torch, '__version__'); "
            + ("assert torch.cuda.is_available()" if str(args.device).startswith("cuda") else ""))
    process = subprocess.run([str(args.python), "-c", code], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError("runner Python/PyTorch/CUDA preflight failed: {}".format(
            process.stdout.strip()))


def classify_task_state(tasks: Sequence[Mapping[str, object]], *, rerun_failed: bool):
    """Classify from current artifacts only; master history is never authority."""
    pending, blocked = [], []
    valid_count = 0
    for task in tasks:
        valid, reasons = validate_run(task)
        if valid:
            valid_count += 1
            continue
        exists = os.path.exists(str(task["run_dir"]))
        manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
        if exists and not rerun_failed:
            blocked.append({"formal_run_key": task["formal_run_key"],
                            "artifact_status": manifest.get("status", "MISSING"),
                            "reasons": reasons})
        else:
            pending.append(task)
    return valid_count, pending, blocked


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S0 84-run orchestrator")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--master", default=os.path.join(HERE, "stage_S0_runs.csv"))
    parser.add_argument("--summary_manifest", default=os.path.join(HERE, "stage_S0_queue_manifest.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset_path", default="/xzf/data/movielens")
    parser.add_argument("--oracle_root", default=os.path.join(HERE, "stage_E3_raw_results", "oracle_reference"))
    parser.add_argument("--role_sizes", default="150000,34500,3000")
    parser.add_argument("--batch_size", type=int, default=40960)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--learning_rate_darts", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--expert_num", type=int, default=4)
    parser.add_argument("--shadow_patience", type=int, default=10)
    parser.add_argument("--require_oracles", type=int, choices=(0, 1), default=1)
    parser.add_argument("--instrument", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=84)
    parser.add_argument("--rerun_failed", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    assert_formal_launcher(args)
    tasks = build_tasks(args.raw_root)
    old_rows = _read_csv(args.master)
    valid_count, pending, blocked = classify_task_state(
        tasks, rerun_failed=bool(args.rerun_failed))
    write_json(args.summary_manifest, {
        "stage": "S0", "expected_run_count": 84, "task_count_by_arm": dict(Counter(
            str(task["arm"]) for task in tasks)), "pending_count": len(pending),
            "completed_count": valid_count, "blocked_count": len(blocked),
            "blocked": blocked, "dry_run": bool(args.dry_run),
            "scope_note": "queue state only; no scientific conclusion",
        "tasks": tasks,
    })
    if args.dry_run:
        print("[stage_S0] fixed matrix: 7 arms x 3 rotations x 4 seeds = 84")
        for task in pending[:max(0, args.max_runs)]:
            print(" ".join(command_for(task, args)))
        return 0 if not blocked else 2
    if blocked:
        print("[stage_S0] invalid/failed existing artifacts require --rerun_failed:",
              json.dumps(blocked, ensure_ascii=False), flush=True)
        return 2
    runtime_preflight(args)
    rows: List[Mapping[str, object]] = list(old_rows)
    for task in pending[:max(0, args.max_runs)]:
        key = str(task["formal_run_key"])
        failure_archive = ""
        if os.path.exists(str(task["run_dir"])):
            failure_archive = _archive_failed(str(task["run_dir"]), args.raw_root, key)
        os.makedirs(str(task["run_dir"]), exist_ok=True)
        started = time.time()
        with open(os.path.join(str(task["run_dir"]), "stdout.log"), "w", encoding="utf-8") as log:
            process = subprocess.run(command_for(task, args), stdout=log, stderr=subprocess.STDOUT,
                                     check=False)
        valid, reasons = validate_run(task)
        status = "OK" if process.returncode == 0 and valid else "FAILED"
        row = {
            "attempt_id": dt.datetime.now().isoformat(), "formal_run_key": key,
            "arm": task["arm"], "rotation": task["rotation"], "seed": task["seed"],
            "status": status, "return_code": process.returncode,
            "wall_seconds": "{:.3f}".format(time.time() - started),
            "run_dir": task["run_dir"], "failure_archive": failure_archive,
            "message": json.dumps(reasons, separators=(",", ":")),
        }
        rows.append(row)
        _write_csv(args.master, rows)
        if status != "OK":
            print("[stage_S0] run failed:", key, reasons, flush=True)
            write_json(args.summary_manifest, {
                "stage": "S0", "expected_run_count": 84,
                "valid_run_count": sum(validate_run(item)[0] for item in tasks),
                "failed_fast_on": key, "failure_reasons": reasons,
                "task_count_by_arm": dict(Counter(str(item["arm"]) for item in tasks)),
                "scope_note": "execution stopped on first failed task",
            })
            return 1
    blocks = audit_all_blocks(tasks)
    write_json(args.summary_manifest, {
        "stage": "S0", "expected_run_count": 84,
        "valid_run_count": sum(validate_run(task)[0] for task in tasks),
        "block_audits": blocks, "all_blocks_valid": all(block["valid"] for block in blocks),
        "task_count_by_arm": dict(Counter(str(task["arm"]) for task in tasks)),
        "scope_note": "protocol audit only; no scientific conclusion",
    })
    return 0 if all(block["valid"] for block in blocks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
