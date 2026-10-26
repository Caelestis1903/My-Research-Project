"""Stage S8 orchestrator -- strictly serial, MAX_PROCESSES_PER_WAVE=1.

Three sub-commands:

- ``--phase base --task-index i``: Phase 1 base retrain.  Each task is a
  ``(dataset, seed)`` pair; the orchestrator launches the S8 base runner
  for that pair and validates the result with ``validate_base_run``.
  Order: all MovieLens tasks first (faster, doubles as a pipeline smoke),
  then KuaiRand.  Total = 8 tasks.

- ``--phase cont --task-index i``: Phase 2 continuation core (2-epoch
  horizons).  Each task is a ``(dataset, seed, anchor_class)`` triple;
  total = 24 tasks (4 seeds x 3 anchors x 2 datasets).  Order: ML first,
  then KR; seed-major; anchors in (selected, middle, late) order.

- ``--phase cont5 --task-index i``: Phase 3 5-epoch extension.  Tasks
  with ``anchor_epoch + 5 > 49`` are excluded; total = 23 tasks (design
  §10).  Triggered only by explicit main-agent approval after core
  succeeds.

After each task, ``stage_S8_report.validate_run`` (continuation) /
``stage_S8_report.validate_base_run`` (base) is invoked; FAIL halts the
wave and surfaces the failure for main-agent review.

Per Algorithm Specification ``stage_S8_design.md`` §4.4 and §10.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import stage_S8_protocol as proto  # noqa: E402

# Strict serial: exactly one subprocess per wave (design §4.4).
MAX_PROCESSES_PER_WAVE = 1


def _python_executable() -> str:
    candidates = [
        os.path.join(ROOT, "venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def _append_state(state_path: str, record: Dict[str, object]) -> None:
    payload: List[Dict[str, object]] = []
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as handle:
                decoded = json.load(handle)
                if isinstance(decoded, list):
                    payload = decoded
        except (OSError, ValueError):
            payload = []
    payload.append(record)
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _validate_base_run(run_dir: str) -> Tuple[bool, List[str]]:
    sys.path.insert(0, HERE)
    try:
        from stage_S8_report import validate_base_run  # noqa: E402
        return validate_base_run(run_dir)
    except Exception as exc:
        return False, ["validator_import_failed: {}".format(exc)]


def _validate_run(run_dir: str) -> Tuple[bool, List[str]]:
    sys.path.insert(0, HERE)
    try:
        from stage_S8_report import validate_run  # noqa: E402
        return validate_run(run_dir)
    except Exception as exc:
        return False, ["validator_import_failed: {}".format(exc)]


def _co_run_pair_id_for(dataset: str, seed: int, anchor_class: str = "") -> str:
    if anchor_class:
        return "s8-{}-{}-{}".format(dataset, int(seed), anchor_class)
    return "s8base-{}-{}".format(dataset, int(seed))


def _planned_peer_run_key_for(dataset: str, seed: int, anchor_class: str = "") -> str:
    if anchor_class:
        return proto.key_cont(dataset, int(seed), anchor_class)
    return proto.key_base(dataset, int(seed))


def _launch_base(dataset: str, seed: int, base_run_dir: str) -> subprocess.Popen:
    if os.path.isdir(base_run_dir):
        existing = [name for name in os.listdir(base_run_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S8 refusing to overwrite non-empty base run dir: " + base_run_dir)
    os.makedirs(base_run_dir, exist_ok=True)
    setting = proto.target_setting(dataset)
    cmd = [
        _python_executable(),
        os.path.join(HERE, "stage_S8_base_run.py"),
        "--dataset", dataset,
        "--setting", setting,
        "--backbone", "m3oe",
        "--run_type", "joint",
        "--seed", str(int(seed)),
        "--epoch", "50",
        "--batch_size", "40960",
        "--learning_rate", "0.01",
        "--learning_rate_darts", "0.001",
        "--weight_decay", "1e-5",
        "--expert_num", "4",
        "--num_workers", "0",
        "--device", "cuda:0",
        "--shadow_patience", "10",
        "--fcn_dims", "512", "256", "256", "64",
        "--allow_protocol_deviation", "0",
        "--allow_nonformal_seed", "0",
        "--co_run_pair_id", _co_run_pair_id_for(dataset, int(seed)),
        "--planned_peer_run_key", _planned_peer_run_key_for(dataset, int(seed)),
        "--worker_id", "s8base-orchestrator",
        "--attempt_id", "s8base-{}-seed{}".format(dataset, int(seed)),
        "--objective_weighting", "sample",
        "--shared_access", "FULL",
        "--fresh_role", "fresh_s8base",
        "--formal_run_key", proto.key_base(dataset, int(seed)),
        "--source_formal_run_key", "",
        "--run_dir", base_run_dir,
    ]
    stdout_path = os.path.join(base_run_dir, "stdout.log")
    stdout_handle = open(stdout_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=stdout_handle, stderr=subprocess.STDOUT, cwd=ROOT)


def _launch_continuation(dataset: str, seed: int, anchor_class: str,
                          out_dir: str, h_max_epochs: int) -> subprocess.Popen:
    if os.path.isdir(out_dir):
        existing = [name for name in os.listdir(out_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S8 refusing to overwrite non-empty continuation dir: " + out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        _python_executable(),
        os.path.join(HERE, "stage_S8_run.py"),
        "--dataset", dataset,
        "--seed", str(int(seed)),
        "--anchor-class", anchor_class,
        "--out-dir", out_dir,
        "--h-max-epochs", str(int(h_max_epochs)),
    ]
    stdout_path = os.path.join(out_dir, "stdout.log")
    stdout_handle = open(stdout_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=stdout_handle, stderr=subprocess.STDOUT, cwd=ROOT)


def run_task_base(task_index: int) -> Dict[str, object]:
    tasks = proto.base_tasks()
    if not 0 <= int(task_index) < len(tasks):
        raise ValueError("task_index out of range: {}".format(task_index))
    task = tasks[int(task_index)]
    started = time.time()
    dataset = str(task["dataset"])
    seed = int(task["seed"])
    base_run_dir = str(task["run_dir"])
    proc: Optional[subprocess.Popen] = None
    stdout_handle = None
    status = "FAILED"
    exit_code = -1
    abort_reason = ""
    try:
        proc = _launch_base(dataset, seed, base_run_dir)
        exit_code = proc.wait()
        if stdout_handle is not None:
            stdout_handle.close()
        if exit_code != 0:
            status = "FAILED"
            abort_reason = "base_runner_exit_code={}".format(exit_code)
        else:
            ok, reasons = _validate_base_run(base_run_dir)
            if ok:
                status = "OK"
            else:
                status = "FAILED"
                abort_reason = "vbase_validation_failed: " + json.dumps(reasons, ensure_ascii=False)
    except FileExistsError as exc:
        status = "REFUSED"
        abort_reason = str(exc)
    except Exception as exc:
        status = "FAILED"
        abort_reason = "{}: {}".format(type(exc).__name__, exc)
    wall = time.time() - started
    record = {
        "phase": "base", "task_index": int(task_index),
        "dataset": dataset, "seed": int(seed),
        "formal_run_key": str(task["formal_run_key"]),
        "run_dir": base_run_dir,
        "status": status, "abort_reason": abort_reason,
        "exit_code": int(exit_code), "wall_seconds": float(wall),
        "max_processes": int(MAX_PROCESSES_PER_WAVE),
        "timestamp": time.time(),
    }
    _append_state(proto.STATE_PATH_BASE, record)
    return record


def run_task_cont(task_index: int, h_max_epochs: int = proto.H_MAX_EPOCHS_CORE,
                   state_path: Optional[str] = None) -> Dict[str, object]:
    tasks = proto.cont_tasks(h_max_epochs=int(h_max_epochs))
    if not 0 <= int(task_index) < len(tasks):
        raise ValueError("task_index out of range: {}".format(task_index))
    task = tasks[int(task_index)]
    started = time.time()
    dataset = str(task["dataset"])
    seed = int(task["seed"])
    anchor_class = str(task["anchor_class"])
    out_dir = str(task["run_dir"])
    proc: Optional[subprocess.Popen] = None
    stdout_handle = None
    status = "FAILED"
    exit_code = -1
    abort_reason = ""
    try:
        proc = _launch_continuation(dataset, seed, anchor_class, out_dir, int(h_max_epochs))
        exit_code = proc.wait()
        if stdout_handle is not None:
            stdout_handle.close()
        if exit_code != 0:
            status = "FAILED"
            abort_reason = "continuation_runner_exit_code={}".format(exit_code)
        else:
            ok, reasons = _validate_run(out_dir)
            if ok:
                status = "OK"
            else:
                status = "FAILED"
                abort_reason = "continuation_validation_failed: " + json.dumps(reasons, ensure_ascii=False)
    except FileExistsError as exc:
        status = "REFUSED"
        abort_reason = str(exc)
    except Exception as exc:
        status = "FAILED"
        abort_reason = "{}: {}".format(type(exc).__name__, exc)
    wall = time.time() - started
    record = {
        "phase": "cont5" if int(h_max_epochs) == proto.H_MAX_EPOCHS_EXTENDED else "cont",
        "task_index": int(task_index),
        "dataset": dataset, "seed": int(seed),
        "anchor_class": anchor_class, "anchor_epoch": int(task["anchor_epoch"]),
        "h_max_epochs": int(h_max_epochs),
        "formal_run_key": str(task["formal_run_key"]),
        "run_dir": out_dir,
        "status": status, "abort_reason": abort_reason,
        "exit_code": int(exit_code), "wall_seconds": float(wall),
        "max_processes": int(MAX_PROCESSES_PER_WAVE),
        "timestamp": time.time(),
    }
    target_state_path = state_path or proto.STATE_PATH_CONT
    if int(h_max_epochs) == proto.H_MAX_EPOCHS_EXTENDED:
        target_state_path = state_path or proto.STATE_PATH_CONT5
    _append_state(target_state_path, record)
    return record


def print_plan(phase: str) -> None:
    if phase == "base":
        tasks = proto.base_tasks()
        print("[stage_S8_dry] phase=base task_count={}".format(len(tasks)))
        for idx, task in enumerate(tasks):
            print("    [{}] dataset={} seed={} key={} run_dir={}".format(
                idx, task["dataset"], int(task["seed"]),
                task["formal_run_key"], task["run_dir"]))
    elif phase == "cont":
        tasks = proto.cont_tasks(h_max_epochs=proto.H_MAX_EPOCHS_CORE)
        print("[stage_S8_dry] phase=cont h_max_epochs={} task_count={}".format(
            proto.H_MAX_EPOCHS_CORE, len(tasks)))
        for idx, task in enumerate(tasks):
            print("    [{}] dataset={} seed={} anchor={}/{} key={} run_dir={}".format(
                idx, task["dataset"], int(task["seed"]), task["anchor_class"],
                int(task["anchor_epoch"]), task["formal_run_key"], task["run_dir"]))
    elif phase == "cont5":
        tasks = proto.cont_tasks(h_max_epochs=proto.H_MAX_EPOCHS_EXTENDED)
        print("[stage_S8_dry] phase=cont5 h_max_epochs={} task_count={}".format(
            proto.H_MAX_EPOCHS_EXTENDED, len(tasks)))
        for idx, task in enumerate(tasks):
            print("    [{}] dataset={} seed={} anchor={}/{} key={} run_dir={}".format(
                idx, task["dataset"], int(task["seed"]), task["anchor_class"],
                int(task["anchor_epoch"]), task["formal_run_key"], task["run_dir"]))
    else:
        raise ValueError("unknown phase: " + phase)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S8 orchestrator (strict serial)")
    parser.add_argument("--phase", choices=("base", "cont", "cont5"), required=True)
    parser.add_argument("--task-index", type=int, default=None,
                        help="0..N-1 selecting a single task (default: dry-run all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the task list without launching subprocesses.")
    args = parser.parse_args()
    if args.dry_run:
        print_plan(args.phase)
        return 0
    if args.task_index is None:
        print("Stage S8: --task-index required for non-dry runs", file=sys.stderr)
        return 2
    if args.phase == "base":
        record = run_task_base(int(args.task_index))
    elif args.phase == "cont":
        record = run_task_cont(int(args.task_index), h_max_epochs=proto.H_MAX_EPOCHS_CORE)
    else:  # cont5
        record = run_task_cont(int(args.task_index), h_max_epochs=proto.H_MAX_EPOCHS_EXTENDED)
    print(json.dumps({
        "phase": record.get("phase"),
        "task_index": record.get("task_index"),
        "dataset": record.get("dataset"),
        "seed": record.get("seed"),
        "anchor_class": record.get("anchor_class", ""),
        "anchor_epoch": record.get("anchor_epoch", 0),
        "formal_run_key": record.get("formal_run_key"),
        "status": record.get("status"),
        "abort_reason": record.get("abort_reason"),
        "wall_seconds": record.get("wall_seconds"),
    }, ensure_ascii=False, indent=2))
    return 0 if record.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
