"""Wait for eight terminal successful B1 tasks, then run frozen CPU analysis.

This is a process supervisor, not a scientific verdict or a training launcher.
It never restarts training, changes model code, or publishes completion claims.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from b1_reconstruction_20261016 import OUT, ROOT, PYTHON, save_json, sha

ANALYZER = ROOT / "experiments/b1_analyze_20261016.py"
EXPECTED = {(seed, arm) for seed in (2022, 2023, 2024, 2025) for arm in ("F", "DH")}


def process_matches(pid, marker):
    try:
        return marker in Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    lock = open(OUT / "postrun_analysis.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_sha = sha(__file__)
    analysis_sha = sha(ANALYZER)
    status_path = OUT / "postrun_analysis_status.json"

    def status(phase, **extra):
        value = dict(phase=phase, pid=os.getpid(), updated=time.time(),
                     analysis_source_sha256=analysis_sha, supervisor_source_sha256=source_sha,
                     scientific_interpretation="PENDING_PRIMARY_AGENT", **extra)
        save_json(status_path, value)
        print(json.dumps(value), flush=True)

    while True:
        s = json.loads((OUT / "scheduler_status.json").read_text())
        tasks = s["tasks"]
        assert {(t["seed"], t["arm"]) for t in tasks} == EXPECTED and len(tasks) == 8
        if any(t["state"] == "FAILED" for t in tasks):
            status("NEEDS_TRAINING_FAILURE_REVIEW", scheduler_status=s["status"])
            return
        successful = all(t["state"] == "COMPLETE" and t.get("exit_code") == 0 for t in tasks)
        live_children = [t["pid"] for t in tasks if t.get("pid")
                         and process_matches(t["pid"], str(t["run_dir"]))]
        if successful and not live_children:
            break
        parent_live = process_matches(s["pid"], "b1_reconstruction_scheduler_20261016.py")
        if not parent_live:
            status("NEEDS_SCHEDULER_REVIEW", scheduler_status=s["status"], live_child_pids=live_children)
            return
        status("WAITING_VALID_FORMAL_MATRIX", scheduler_pid=s["pid"], scheduler_live=True,
               live_child_pids=live_children, successful_full_runs=sum(t["state"] == "COMPLETE" for t in tasks),
               pending_tasks=sum(t["state"] == "PENDING" for t in tasks))
        time.sleep(30)
    if sha(ANALYZER) != analysis_sha or sha(__file__) != source_sha:
        status("NEEDS_SOURCE_CHANGE_REVIEW")
        return
    existing = OUT / "analysis" / "analysis_receipt.json"
    if existing.exists():
        status("EXISTING_ANALYSIS_REQUIRES_PRIMARY_REVIEW", receipt=str(existing))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [PYTHON, "-u", str(ANALYZER)]
    status("RUNNING_NUMERICAL_AUDIT", command=command)
    with open(OUT / "postrun_analysis.stdout.log", "a", buffering=1) as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        status("NEEDS_NUMERICAL_FAILURE_REVIEW", exit_code=result.returncode)
        return
    receipt = json.loads(existing.read_text())
    assert receipt["scope"] == "B1_RECONSTRUCTION_FORENSIC" and receipt["formal_runs_completed"] == 8
    assert receipt["analysis_source_sha256"] == analysis_sha == sha(ANALYZER)
    for filename, digest in receipt["output_sha256"].items():
        assert sha(existing.parent / filename) == digest
    status("NUMERICAL_AUDIT_COMPLETE_PRIMARY_INTERPRETATION_REQUIRED",
           receipt=str(existing), receipt_sha256=sha(existing), formal_runs_completed=8)


if __name__ == "__main__":
    main()
