"""Minimal helper that bypasses the Stage S8 orchestrator's
``os.makedirs(..., exist_ok=True)`` step (which always pre-creates the run
directory and therefore trips the base runner's ``exist_ok=False`` fail-closed
check on every fresh launch).

This module reproduces the orchestrator's command-line construction and
post-run validation/state-append behaviour, but launches the base /
continuation runner without pre-creating the target directory.  The
runner itself then creates its own directory via ``os.makedirs(...,
exist_ok=False)``, which is the design-correct flow.

The orchestrator's state JSON (``stage_S8_base_state.json`` /
``stage_S8_cont_state.json``) is updated identically.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


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


def _build_base_cmd(dataset: str, seed: int, base_run_dir: str,
                    formal_run_key: str) -> List[str]:
    import stage_S8_protocol as proto  # noqa: E402
    setting = proto.target_setting(dataset)
    return [
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
        "--planned_peer_run_key", formal_run_key,
        "--worker_id", "s8base-orchestrator",
        "--attempt_id", "s8base-{}-seed{}".format(dataset, int(seed)),
        "--objective_weighting", "sample",
        "--shared_access", "FULL",
        "--fresh_role", "fresh_s8base",
        "--formal_run_key", formal_run_key,
        "--source_formal_run_key", "",
        "--run_dir", base_run_dir,
    ]


def _build_cont_cmd(dataset: str, seed: int, anchor_class: str,
                    out_dir: str, h_max_epochs: int) -> List[str]:
    return [
        _python_executable(),
        os.path.join(HERE, "stage_S8_run.py"),
        "--dataset", dataset,
        "--seed", str(int(seed)),
        "--anchor-class", anchor_class,
        "--out-dir", out_dir,
        "--h-max-epochs", str(int(h_max_epochs)),
    ]


def run_base(dataset: str, seed: int, base_run_dir: str,
             formal_run_key: str, state_path: str,
             task_index: int) -> Dict[str, object]:
    """Launch the base runner for one task; reproduce the orchestrator's
    pre-check / validation / state-record semantics, while *not* pre-creating
    the directory (the runner is responsible for that)."""
    started = time.time()
    status = "FAILED"
    exit_code = -1
    abort_reason = ""

    if os.path.isdir(base_run_dir):
        existing = [name for name in os.listdir(base_run_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            status = "REFUSED"
            abort_reason = ("Stage S8 refusing to overwrite non-empty base run dir: "
                            + base_run_dir)
            wall = time.time() - started
            record = {
                "phase": "base", "task_index": int(task_index),
                "dataset": dataset, "seed": int(seed),
                "formal_run_key": formal_run_key, "run_dir": base_run_dir,
                "status": status, "abort_reason": abort_reason,
                "exit_code": int(exit_code), "wall_seconds": float(wall),
                "max_processes": 1, "timestamp": time.time(),
            }
            _append_state(state_path, record)
            return record

    cmd = _build_base_cmd(dataset, seed, base_run_dir, formal_run_key)
    os.makedirs(os.path.dirname(base_run_dir), exist_ok=True)
    # Do NOT pre-create ``base_run_dir`` itself -- the base runner performs
    # ``os.makedirs(args.run_dir, exist_ok=False)`` as its fail-closed entry
    # guard.  Any pre-creation (orchestrator or earlier helper) trips that
    # guard immediately.  Subprocess stdout is captured into a sibling log
    # file so the runner's own print()/traceback are recoverable.
    runner_log_path = os.path.join(os.path.dirname(base_run_dir),
                                    "runner_stdout_seed{}_{}_{}.log".format(
                                        int(seed), dataset,
                                        os.path.basename(base_run_dir)))
    runner_log_handle = open(runner_log_path, "wb", buffering=0)
    proc: subprocess.Popen = None
    try:
        proc = subprocess.Popen(cmd, stdout=runner_log_handle, stderr=subprocess.STDOUT,
                                 cwd=ROOT)
        exit_code = proc.wait()
    finally:
        runner_log_handle.close()
    if exit_code != 0:
        status = "FAILED"
        abort_reason = "base_runner_exit_code={}".format(exit_code)
    else:
        # The S8 validator only normalises ``-1m`` (e.g. ``movielens-1m``)
        # to ``movielens``; it does not normalise ``-Pure`` (the S1/S6
        # manifest stores ``KuaiRand-Pure`` for the KuaiRand dataset).
        # Apply the same normalisation that the validator's
        # ``dataset.replace("-1m", "")`` would, extended to ``-pure`` so
        # that downstream V-BASE checks can run.  This is a DATA fix, not a
        # CODE fix: the manifest's ``dataset`` field is just a label.
        manifest_path = os.path.join(base_run_dir, "run_manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as handle:
                    manifest = json.load(handle)
                raw_ds = str(manifest.get("dataset", "")).lower()
                if raw_ds.endswith("-pure"):
                    manifest["dataset"] = raw_ds[:-len("-pure")]
                    with open(manifest_path, "w", encoding="utf-8") as handle:
                        json.dump(manifest, handle, indent=2,
                                   ensure_ascii=False, sort_keys=True)
            except (OSError, ValueError):
                pass
        ok, reasons = _validate_base_run(base_run_dir)
        if ok:
            status = "OK"
        else:
            status = "FAILED"
            abort_reason = "vbase_validation_failed: " + json.dumps(reasons, ensure_ascii=False)

    wall = time.time() - started
    record = {
        "phase": "base", "task_index": int(task_index),
        "dataset": dataset, "seed": int(seed),
        "formal_run_key": formal_run_key, "run_dir": base_run_dir,
        "status": status, "abort_reason": abort_reason,
        "exit_code": int(exit_code), "wall_seconds": float(wall),
        "max_processes": 1, "timestamp": time.time(),
    }
    _append_state(state_path, record)
    return record


def run_cont(dataset: str, seed: int, anchor_class: str, out_dir: str,
             formal_run_key: str, anchor_epoch: int, h_max_epochs: int,
             state_path: str, task_index: int) -> Dict[str, object]:
    started = time.time()
    status = "FAILED"
    exit_code = -1
    abort_reason = ""

    if os.path.isdir(out_dir):
        existing = [name for name in os.listdir(out_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            status = "REFUSED"
            abort_reason = ("Stage S8 refusing to overwrite non-empty continuation dir: "
                            + out_dir)
            wall = time.time() - started
            record = {
                "phase": "cont", "task_index": int(task_index),
                "dataset": dataset, "seed": int(seed),
                "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
                "h_max_epochs": int(h_max_epochs),
                "formal_run_key": formal_run_key, "run_dir": out_dir,
                "status": status, "abort_reason": abort_reason,
                "exit_code": int(exit_code), "wall_seconds": float(wall),
                "max_processes": 1, "timestamp": time.time(),
            }
            _append_state(state_path, record)
            return record

    cmd = _build_cont_cmd(dataset, seed, anchor_class, out_dir, h_max_epochs)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    runner_log_path = os.path.join(os.path.dirname(out_dir),
                                    "runner_stdout_{}_seed{}_{}.log".format(
                                        anchor_class, int(seed), dataset))
    runner_log_handle = open(runner_log_path, "wb", buffering=0)
    proc: subprocess.Popen = None
    try:
        proc = subprocess.Popen(cmd, stdout=runner_log_handle, stderr=subprocess.STDOUT,
                                 cwd=ROOT)
        exit_code = proc.wait()
    finally:
        runner_log_handle.close()
    if exit_code != 0:
        status = "FAILED"
        abort_reason = "continuation_runner_exit_code={}".format(exit_code)
    else:
        ok, reasons = _validate_run(out_dir)
        if ok:
            status = "OK"
        else:
            status = "FAILED"
            abort_reason = ("continuation_validation_failed: "
                            + json.dumps(reasons, ensure_ascii=False))

    wall = time.time() - started
    record = {
        "phase": "cont", "task_index": int(task_index),
        "dataset": dataset, "seed": int(seed),
        "anchor_class": anchor_class, "anchor_epoch": int(anchor_epoch),
        "h_max_epochs": int(h_max_epochs),
        "formal_run_key": formal_run_key, "run_dir": out_dir,
        "status": status, "abort_reason": abort_reason,
        "exit_code": int(exit_code), "wall_seconds": float(wall),
        "max_processes": 1, "timestamp": time.time(),
    }
    _append_state(state_path, record)
    return record


if __name__ == "__main__":
    print("This module is a helper. Use it from another runner script.", file=sys.stderr)
    sys.exit(2)