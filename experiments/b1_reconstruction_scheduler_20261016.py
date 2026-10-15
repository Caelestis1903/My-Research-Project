"""Persistent local process scheduler for the separately named B1 study."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import hashlib

from b1_reconstruction_20261016 import OUT, ROOT, PYTHON, STUDY, save_json, sha

WORKER = ROOT / "experiments/b1_reconstruction_20261016.py"


def gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    return [dict(zip(("index", "total", "used"), map(int, line.split(","))))
            for line in output.strip().splitlines()]


def scheduler():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / "scheduler.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = json.loads((OUT / "engineering_verification.json").read_text())
    assert gate["status"] == "PASS" and gate["source_sha256"] == sha(WORKER)
    publication_checkout = ROOT.parent / "phase0-3-git"
    code_commit = subprocess.check_output(["git", "-C", str(publication_checkout), "rev-parse", "HEAD"], text=True).strip()
    committed_worker = subprocess.check_output(["git", "-C", str(publication_checkout), "show",
        code_commit + ":experiments/b1_reconstruction_20261016.py"])
    assert hashlib.sha256(committed_worker).hexdigest() == sha(WORKER), "commit does not contain actual worker"
    # Reserve enough space for all 400 full-state files and endpoint products.
    estimated_disk = int(gate["estimated_400_checkpoint_bytes"] * 1.25) + 10 * 1024**3
    assert shutil.disk_usage(OUT).free > estimated_disk, (estimated_disk, shutil.disk_usage(OUT).free)
    tasks = [{"seed": seed, "arm": arm, "state": "PENDING",
              "code_commit": code_commit, "run_dir": str(OUT / "runs" / f"seed{seed}" / arm)}
             for seed in (2022, 2023, 2024, 2025) for arm in ("F", "DH")]
    children = {}
    history_path = OUT / "scheduler_events.jsonl"

    def event(value):
        with open(history_path, "a") as f:
            f.write(json.dumps(dict(timestamp=time.time(), **value)) + "\n")

    def status(state, gpu_snapshot):
        save_json(OUT / "scheduler_status.json", {"study_id": STUDY, "pid": os.getpid(),
            "status": state, "updated": time.time(), "gpu_snapshot": gpu_snapshot,
            "max_processes_per_gpu": 5, "memory_fraction_limit": .9,
            "memory_check_interval_seconds": 5, "cold_reservation_mib": 20000,
            "tasks": tasks, "independent_review": False,
            "worker_sha256": sha(WORKER), "estimated_disk_bytes": estimated_disk})

    # No automatic restart/overwrite of an earlier attempt.
    for task in tasks:
        directory = Path(task["run_dir"])
        if directory.exists():
            manifest = directory / "run_manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()).get("status") == "OK":
                progress = json.loads((directory / "progress.json").read_text())
                assert progress["phase"] == "COMPLETE"
                task["state"] = "COMPLETE"
            else:
                raise RuntimeError(f"Existing incomplete attempt requires inspection: {directory}")
    event({"event": "SCHEDULER_STARTED", "pid": os.getpid()})
    while True:
        snapshot = gpus()
        for pid, (process, task, log) in list(children.items()):
            code = process.poll()
            if code is not None:
                log.close()
                task.update(state="COMPLETE" if code == 0 else "FAILED", exit_code=code,
                            finished=time.time())
                event({"event": "WORKER_EXIT", "pid": pid, "exit_code": code,
                       "seed": task["seed"], "arm": task["arm"]})
                del children[pid]
        if any(task["state"] == "FAILED" for task in tasks):
            # Let valid live work finish, but do not launch unreviewed identical retries.
            if not children:
                status("NEEDS_FAILURE_REVIEW", snapshot)
                return
            status("DRAINING_AFTER_FAILURE", snapshot)
            time.sleep(5)
            continue
        if all(task["state"] == "COMPLETE" for task in tasks):
            status("COMPLETE_PENDING_ANALYSIS", snapshot)
            return
        for gpu in snapshot:
            active = [task for _, task, _ in children.values() if task["gpu"] == gpu["index"]]
            if active and gpu["used"] > .9 * gpu["total"]:
                # Stop only this study's newest local worker. External jobs are untouched.
                newest = max(active, key=lambda task: task["started"])
                event({"event": "MEMORY_LIMIT_STOP", "gpu": gpu, "pid": newest["pid"]})
                children[newest["pid"]][0].terminate()
                newest["stop_reason"] = "GPU_TOTAL_MEMORY_OVER_90_PERCENT"
                continue
            if len(active) >= 5:
                continue
            pending = next((task for task in tasks if task["state"] == "PENDING"), None)
            if pending is None:
                continue
            # Peak reserved memory, rounded up with safety margin; before empirical
            # observation use a conservative 20 GB reservation for the fixed batch.
            peaks = []
            for task in tasks:
                progress = Path(task["run_dir"]) / "progress.json"
                if progress.exists():
                    peak = json.loads(progress.read_text()).get("peak_reserved_bytes", 0)
                    if peak:
                        peaks.append(peak / 1024**2)
            reservation = max(peaks) * 1.15 + 512 if peaks else 20000
            # Unobserved CUDA initialization also needs its full reserved headroom.
            initializing = sum(reservation for task in active
                               if not (Path(task["run_dir"]) / "progress.json").exists())
            if gpu["used"] + initializing + reservation > .9 * gpu["total"]:
                continue
            run_dir = Path(pending["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=False)
            log = open(run_dir / "stdout.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu["index"]),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            process = subprocess.Popen([PYTHON, "-u", str(WORKER), "worker", "--arm", pending["arm"],
                "--seed", str(pending["seed"]), "--run-dir", str(run_dir)], cwd=ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pending.update(state="RUNNING", gpu=gpu["index"], pid=process.pid, started=time.time())
            children[process.pid] = (process, pending, log)
            event({"event": "WORKER_STARTED", "seed": pending["seed"], "arm": pending["arm"],
                   "pid": process.pid, "gpu": gpu, "reservation_mib": reservation})
        status("RUNNING" if children else "WAITING_GPU_HEADROOM", snapshot)
        time.sleep(5)


if __name__ == "__main__":
    scheduler()
