"""Stage S7 orchestrator: 4 waves x 2 concurrent subprocess probes per wave.

Simplified S6-style orchestrator.  Each wave pairs kuairand and movielens
probes for one seed; the two probes are launched concurrently and the
wave waits for both to exit 0 and pass ``validate_probe_run``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE_PATH = os.path.join(HERE, "stage_S7_state.json")
STATE_DIR = os.path.join(HERE, "stage_S7_raw_results")

KUAIRAND_SOURCE_ROOT = "stage_S6_raw_results"
MOVIELENS_SOURCE_ROOT = "stage_S6ML_raw_results"
WAVES: Tuple[int, ...] = (2022, 2023, 2024, 2025)


def _source_run_dir(dataset_root: str, seed: int) -> str:
    return os.path.abspath(os.path.join(HERE, dataset_root, "F",
                                        "seed{}".format(int(seed)), "joint"))


def _out_dir(dataset: str, seed: int) -> str:
    return os.path.abspath(os.path.join(STATE_DIR, dataset, "seed{}".format(int(seed))))


def _python_executable() -> str:
    candidates = [
        os.path.join(ROOT, "venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def _append_state(record: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    payload: List[Dict[str, object]] = []
    if os.path.isfile(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as handle:
                decoded = json.load(handle)
                if isinstance(decoded, list):
                    payload = decoded
        except (OSError, ValueError):
            payload = []
    payload.append(record)
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _validate_run(dataset: str, run_dir: str) -> Tuple[bool, List[str]]:
    """Best-effort import of the validator; tolerate missing in dry runs."""
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "history_exp", "first"))
    try:
        from stage_S7_report import validate_probe_run  # noqa: E402
        return validate_probe_run(run_dir)
    except Exception as exc:
        return False, ["validator_import_failed: {}".format(exc)]


def _launch_probe(dataset: str, seed: int) -> subprocess.Popen:
    out_dir = _out_dir(dataset, seed)
    if os.path.isdir(out_dir):
        existing = [name for name in os.listdir(out_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S7 refusing to overwrite non-empty run directory: {}".format(out_dir))
    source_run_dir = _source_run_dir(
        KUAIRAND_SOURCE_ROOT if dataset == "kuairand" else MOVIELENS_SOURCE_ROOT, seed)
    cmd = [
        _python_executable(),
        os.path.join(HERE, "stage_S7_probe.py"),
        "--dataset", dataset, "--seed", str(int(seed)),
        "--source-run-dir", source_run_dir,
        "--out-dir", out_dir,
    ]
    stdout_path = os.path.join(out_dir, "stdout.log")
    os.makedirs(out_dir, exist_ok=True)
    stdout_handle = open(stdout_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=stdout_handle, stderr=subprocess.STDOUT,
                            cwd=ROOT)


def run_wave(seed: int) -> Dict[str, object]:
    started = time.time()
    attempts: Dict[str, Dict[str, object]] = {}
    processes: Dict[str, subprocess.Popen] = {}
    for dataset in ("kuairand", "movielens"):
        try:
            processes[dataset] = _launch_probe(dataset, seed)
        except FileExistsError as exc:
            attempts[dataset] = {"status": "REFUSED", "reason": str(exc),
                                  "exit_code": None}
    for dataset, proc in processes.items():
        exit_code = proc.wait()
        attempts[dataset] = {"status": "RUNNING", "exit_code": int(exit_code),
                              "out_dir": _out_dir(dataset, seed)}
    completed: List[str] = []
    failed: List[str] = []
    for dataset, attempt in attempts.items():
        if attempt.get("status") == "REFUSED":
            failed.append(dataset)
            continue
        exit_code = attempt.get("exit_code", -1)
        ok, reasons = _validate_run(dataset, attempt["out_dir"])
        if exit_code == 0 and ok:
            attempts[dataset]["status"] = "OK"
            attempts[dataset]["validation_reasons"] = []
            completed.append(dataset)
        else:
            attempts[dataset]["status"] = "FAILED"
            attempts[dataset]["validation_reasons"] = list(reasons)
            failed.append(dataset)
    wall = time.time() - started
    record = {"stage": "S7", "seed": int(seed), "wall_seconds": float(wall),
              "attempts": attempts, "completed": completed, "failed": failed,
              "timestamp": time.time()}
    _append_state(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S7 orchestrator")
    parser.add_argument("--wave-index", type=int, default=None,
                        help="0..3 select a single wave (default: dry-run all waves)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list waves without launching subprocesses")
    args = parser.parse_args()
    if args.dry_run:
        for idx, seed in enumerate(WAVES):
            print("[stage_S7_dry] wave {} seed={} datasets=kuairand,movielens".format(idx, seed))
        return 0
    if args.wave_index is None:
        print("Stage S7: --wave-index required for non-dry runs", file=sys.stderr)
        return 2
    if not 0 <= int(args.wave_index) < len(WAVES):
        print("Stage S7: --wave-index out of range", file=sys.stderr)
        return 2
    seed = WAVES[int(args.wave_index)]
    record = run_wave(seed)
    print(json.dumps({"wave_index": int(args.wave_index), "seed": int(seed),
                      "completed": record["completed"], "failed": record["failed"]},
                     ensure_ascii=False, indent=2))
    return 0 if not record["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())