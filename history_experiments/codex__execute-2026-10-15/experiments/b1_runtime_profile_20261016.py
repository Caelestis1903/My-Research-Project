"""Bounded CUDA engineering profile after an unexplained native preflight abort.

Two original-size DH main updates plus one original architecture update;
outputs are engineering-only, not a formal seed or performance measurement.
"""
from __future__ import annotations
import csv
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from b1_reconstruction_20261016 import OUT, ROOT, run_args, runner_class, save_json, sha


def main():
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    torch.set_num_threads(2)
    folder = OUT / "native_diagnostics" / ("runtime_profile_" + str(time.time_ns()))
    folder.mkdir(parents=True, exist_ok=False)
    log = open(folder / "native_stack.log", "w", buffering=1)
    faulthandler.enable(file=log, all_threads=True)
    stop = threading.Event()
    samples = []

    def observe():
        with open(folder / "gpu_memory.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("timestamp", "used_mib", "total_mib"))
            while not stop.is_set():
                raw = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
                used, total = map(int, raw.strip().split(","))
                samples.append(used)
                writer.writerow((time.time(), used, total)); f.flush()
                if used > .9 * total:
                    save_json(folder / "memory_limit_stop.json", {"used": used, "total": total, "pid": os.getpid()})
                    os.kill(os.getpid(), signal.SIGTERM)
                stop.wait(.5)

    initial = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    used, total = map(int, initial.strip().split(","))
    assert used + 16384 <= .9 * total, "insufficient conservative engineering headroom"
    monitor = threading.Thread(target=observe, daemon=True); monitor.start()
    try:
        print("PROFILE_CONSTRUCTOR_BEGIN", flush=True)
        r = runner_class()(run_args("DH", 2022, folder / "runner"))
        print("PROFILE_CONSTRUCTOR_PASS", flush=True)
        r.model.train()
        iterator = iter(r.train_loader)
        for step in range(2):
            x_batch, y_batch = next(iterator)
            assert len(y_batch) == 40960
            fingerprint = r._record_batch(0, step, "train", x_batch, y_batch)
            r.batch_fingerprints.append(fingerprint)
            x = x_batch.T.to(r.device, dtype=torch.float32)
            y = y_batch.to(r.device, dtype=torch.float32)
            pred = r.model(x)
            losses = r._objective_components(pred, y, x[-1].long())
            r.optimizer.zero_grad(set_to_none=True); r.arch_optimizer.zero_grad(set_to_none=True)
            r._controlled_step(losses, fingerprint, x_batch, 0, step, "main")
            r.optimizer_steps += 1
            torch.cuda.synchronize()
            print(f"PROFILE_MAIN_STEP_{step}_PASS", flush=True)
        r._architecture_update(None, 0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        result = {"status": "PASS", "scope": "CUDA_ENGINEERING_ORIGINAL_BATCH_TWO_MAIN_ONE_ARCH",
            "formal_training_runs_completed": 0, "batch_size": 40960, "arm": "DH", "seed": 2022,
            "main_steps_checked": 2, "architecture_steps_checked": 1,
            "peak_reserved_bytes": peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "baseline_gpu_used_mib": used, "observed_gpu_total_peak_mib": max(samples),
            "conservative_reservation_mib": peak / 1024**2 * 1.25 + 768,
            "worker_sha256": sha(ROOT / "experiments/b1_reconstruction_20261016.py"),
            "profile_sha256": sha(__file__), "diagnostic_directory": str(folder),
            "original_native_abort_root_cause": "UNRESOLVED_NOT_REPRODUCED"}
        save_json(OUT / "native_diagnostics/cuda_runtime_profile_receipt.json", result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        stop.set(); monitor.join(timeout=5)


if __name__ == "__main__":
    main()
