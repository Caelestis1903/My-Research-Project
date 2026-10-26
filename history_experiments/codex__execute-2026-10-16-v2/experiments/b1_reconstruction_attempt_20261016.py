"""Attempt/diagnostic launcher around the frozen dense measurement algorithm."""
from __future__ import annotations
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time
import traceback

from b1_reconstruction_20261016 import run_args, runner_class, save_json, sha


def phase(name, **extra):
    print(json.dumps(dict(phase=name, pid=os.getpid(), timestamp=time.time(), **extra)), flush=True)


def main():
    faulthandler.enable(all_threads=True)
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("F", "DH"), required=True)
    p.add_argument("--seed", type=int, choices=(2022, 2023, 2024, 2025), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--attempt-id", type=int, required=True)
    p.add_argument("--code-commit", required=True)
    a = p.parse_args()
    assert a.attempt_id >= 1
    frozen_source = sha(__file__)
    started = time.time()
    try:
        phase("BEFORE_TORCH_IMPORT", attempt=a.attempt_id)
        import torch
        torch.set_num_threads(2)
        phase("TORCH_IMPORTED", torch_version=torch.__version__)
        args = run_args(a.arm, a.seed, a.run_dir)
        args.attempt_id = str(a.attempt_id)
        phase("BEFORE_CONSTRUCTOR")
        r = runner_class()(args)
        phase("CONSTRUCTOR_PASS", actual_init_hash=r.measured_init)
        assert sha(__file__) == frozen_source
        phase("BEFORE_FIT")
        info = r.fit()
        phase("FIT_PASS", selected_epoch=info["selected_epoch"])
        r.write_outputs(info, time.time() - started)
        assert sha(__file__) == frozen_source
        for filename in ("config.json", "run_manifest.json"):
            path = a.run_dir / filename
            value = json.loads(path.read_text())
            value.update(attempt_id=str(a.attempt_id), code_commit=a.code_commit,
                         launch_wrapper_sha256=frozen_source, native_trace_enabled=True)
            if filename == "run_manifest.json":
                value["artifact_hashes"]["config.json"] = sha(a.run_dir / "config.json")
            save_json(path, value)
        phase("COMPLETE")
    except BaseException as exc:
        save_json(a.run_dir / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
            "attempt_id": str(a.attempt_id), "code_commit": a.code_commit,
            "launch_wrapper_sha256": frozen_source, "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
