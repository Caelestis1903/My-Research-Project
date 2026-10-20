"""Serialize only N17's shared frozen-split preparation, then run an unchanged module.

This is a runner-side wrapper. It does not alter N17 source, config, seed, or budget.
The shared split writer always targets movielens_seed2022, independent of optimizer seed.
"""
from __future__ import annotations

import fcntl
import functools
import runpy
import sys
from pathlib import Path


REPO = Path("/home/xzf/Code/phase0-3-natural-git")


LOCKED_MODULES = {
    "experiments.natural_ml_extensions.n17_train",
    "experiments.natural_ml_extensions.n17_diagnostics",
    "experiments.natural_ml_extensions.n17_readout",
    "experiments.natural_ml_extensions.n17_basiccheck",
}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: n17_prepare_lock_runner.py <n17_module> [module arguments...]")
    module_name = sys.argv[1]
    if module_name not in LOCKED_MODULES:
        raise SystemExit(f"unsupported module for N17 prepare lock: {module_name}")
    module_args = sys.argv[2:]

    # Script execution sets sys.path[0] to this runner directory. Put the
    # canonical checkout first so imports resolve to the reviewed producer.
    sys.path.insert(0, str(REPO))

    from experiments.natural_ml_extensions import n17_bootstrap

    original_prepare = n17_bootstrap.prepare
    lock_path = n17_bootstrap.OUT / "history" / "splits" / "movielens_seed2022.prepare.lock"

    @functools.wraps(original_prepare)
    def serialized_prepare(*args, **kwargs):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return original_prepare(*args, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    n17_bootstrap.prepare = serialized_prepare
    sys.argv = [module_name, *module_args]
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
