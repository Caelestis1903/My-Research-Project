"""Stage S7-B orchestrator: 8 waves x 2 concurrent subprocess continuations.

Wave matrix (one (seed, anchor) pair per wave, two datasets per wave):

- wave 0: seed 2022, anchor 9,  {kuairand, movielens}
- wave 1: seed 2022, anchor 39, {kuairand, movielens}
- wave 2: seed 2023, anchor 9,  {kuairand, movielens}
- wave 3: seed 2023, anchor 39, {kuairand, movielens}
- wave 4: seed 2024, anchor 9,  {kuairand, movielens}
- wave 5: seed 2024, anchor 39, {kuairand, movielens}
- wave 6: seed 2025, anchor 9,  {kuairand, movielens}
- wave 7: seed 2025, anchor 39, {kuairand, movielens}

Each wave launches exactly two concurrent sub-processes (one per dataset);
the wave waits for both to exit 0 and pass ``stage_S7B_report.validate_run``.
Per the project runner rule (max_training_processes=2; see S6's frozen
queue manifest) and Algorithm Spec §1, the F/T branches of each continuation
live in the SAME process via ``branch_state`` deepcopy, so each
``(dataset, seed, anchor)`` task is one sub-process.

Per Algorithm Specification -- Stage S7-B.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import stage_S7B_protocol as proto  # noqa: E402


# Two-process cap per wave (S6 frozen precedent).
MAX_PROCESSES_PER_WAVE = 2


def _wave_plan() -> List[Tuple[int, int]]:
    """Return the ordered list of (seed, anchor) pairs that define the 8
    waves.  Within each wave the two datasets launch in parallel.
    """
    return [(int(seed), int(anchor))
            for seed in proto.SEEDS
            for anchor in proto.ANCHORS]


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
    payload: List[Dict[str, object]] = []
    if os.path.isfile(proto.STATE_PATH):
        try:
            with open(proto.STATE_PATH, encoding="utf-8") as handle:
                decoded = json.load(handle)
                if isinstance(decoded, list):
                    payload = decoded
        except (OSError, ValueError):
            payload = []
    payload.append(record)
    with open(proto.STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _validate_run(run_dir: str) -> Tuple[bool, List[str]]:
    sys.path.insert(0, HERE)
    try:
        from stage_S7B_report import validate_run  # noqa: E402
        return validate_run(run_dir)
    except Exception as exc:
        return False, ["validator_import_failed: {}".format(exc)]


def _launch_continuation(dataset: str, seed: int, anchor: int) -> subprocess.Popen:
    out_dir = proto.out_run_dir(dataset, int(seed), int(anchor))
    if os.path.isdir(out_dir):
        existing = [name for name in os.listdir(out_dir)
                    if name not in ("failed_attempts", "stdout.log", "process_interval.json")]
        if existing:
            raise FileExistsError(
                "Stage S7-B refusing to overwrite non-empty run directory: {}".format(out_dir))
    cmd = [
        _python_executable(),
        os.path.join(HERE, "stage_S7B_run.py"),
        "--dataset", dataset, "--seed", str(int(seed)),
        "--anchor", str(int(anchor)),
        "--out-dir", out_dir,
    ]
    stdout_path = os.path.join(out_dir, "stdout.log")
    os.makedirs(out_dir, exist_ok=True)
    stdout_handle = open(stdout_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=stdout_handle, stderr=subprocess.STDOUT,
                            cwd=ROOT)


def run_wave(wave_index: int) -> Dict[str, object]:
    """Run one wave: launch up to ``MAX_PROCESSES_PER_WAVE`` sub-processes
    in parallel (one per dataset for a fixed (seed, anchor)) and validate the
    artifacts.
    """
    plan = _wave_plan()
    if not 0 <= int(wave_index) < len(plan):
        raise ValueError("wave_index out of range: {}".format(wave_index))
    seed, anchor = plan[int(wave_index)]
    started = time.time()
    attempts: Dict[str, Dict[str, object]] = {}
    processes: Dict[str, subprocess.Popen] = {}
    for dataset in sorted(proto.DATASETS.keys()):
        key = "{}_a{}".format(dataset, int(anchor))
        try:
            processes[key] = _launch_continuation(dataset, int(seed), int(anchor))
        except FileExistsError as exc:
            attempts[key] = {"status": "REFUSED", "reason": str(exc), "exit_code": None,
                              "dataset": dataset, "anchor": int(anchor)}
    if len(processes) > MAX_PROCESSES_PER_WAVE:
        # Defensive: the wave design is exactly two tasks; refuse to start a
        # wave that would exceed the cap.  This is a structural guard, not a
        # runtime check.
        raise RuntimeError(
            "Stage S7-B wave {} would launch {} processes (> cap {}); "
            "orchestrator design violation".format(
                wave_index, len(processes), MAX_PROCESSES_PER_WAVE))
    for task_key, proc in processes.items():
        exit_code = proc.wait()
        dataset, anchor = task_key.split("_a")
        anchor = int(anchor)
        out_dir = proto.out_run_dir(dataset, int(seed), anchor)
        attempts[task_key] = {"status": "RUNNING", "exit_code": int(exit_code),
                              "out_dir": out_dir, "dataset": dataset, "anchor": anchor}
    completed: List[str] = []
    failed: List[str] = []
    for task_key, attempt in attempts.items():
        if attempt.get("status") == "REFUSED":
            failed.append(task_key)
            continue
        exit_code = attempt.get("exit_code", -1)
        ok, reasons = _validate_run(attempt["out_dir"])
        attempt["validation_reasons"] = list(reasons)
        if exit_code == 0 and ok:
            attempt["status"] = "OK"
            completed.append(task_key)
        else:
            attempt["status"] = "FAILED"
            failed.append(task_key)
    wall = time.time() - started
    record = {"stage": "S7B", "wave_index": int(wave_index), "seed": int(seed),
              "anchor": int(anchor), "max_processes": int(MAX_PROCESSES_PER_WAVE),
              "wall_seconds": float(wall),
              "attempts": attempts, "completed": completed, "failed": failed,
              "timestamp": time.time()}
    _append_state(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage S7-B orchestrator")
    parser.add_argument("--wave-index", type=int, default=None,
                        help="0..7 select a single wave (default: dry-run all waves)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list waves without launching subprocesses")
    args = parser.parse_args()
    if args.dry_run:
        plan = _wave_plan()
        for idx, (seed, anchor) in enumerate(plan):
            print("[stage_S7B_dry] wave {} seed={} anchor={} tasks={}".format(
                idx, int(seed), int(anchor), len(proto.DATASETS)))
            for dataset in sorted(proto.DATASETS.keys()):
                print("    {}/seed{}/anchor{} -> {}".format(
                    dataset, int(seed), int(anchor),
                    proto.out_run_dir(dataset, int(seed), int(anchor))))
        return 0
    if args.wave_index is None:
        print("Stage S7-B: --wave-index required for non-dry runs", file=sys.stderr)
        return 2
    if not 0 <= int(args.wave_index) < len(_wave_plan()):
        print("Stage S7-B: --wave-index out of range", file=sys.stderr)
        return 2
    record = run_wave(int(args.wave_index))
    print(json.dumps({"wave_index": record["wave_index"], "seed": record["seed"],
                      "anchor": record["anchor"],
                      "completed": record["completed"], "failed": record["failed"]},
                     ensure_ascii=False, indent=2))
    return 0 if not record["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
