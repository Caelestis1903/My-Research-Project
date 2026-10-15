"""Reviewer check: resume equivalence end-to-end.

Verifies that a run interrupted after epoch 0 and resumed from
``last_resume.pt`` reproduces a fresh uninterrupted 2-epoch run
bit-for-bit:

- ``batch_order_hash`` and ``architecture_batch_order_hash`` identical
- ``macro_auc_by_epoch`` identical (exact float equality)
- selected epoch identical
- model state at end of training identical tensor-by-tensor
- Adam ``exp_avg`` of the arm-specific parameter identical

Runs on CPU so the comparison is deterministic (CUDA embedding-backward
atomics are outside this test's scope; the formal runs execute on CUDA
with the same code path).

Run from project root:
    venv/Scripts/python.exe research_R2/tests/reviewer_check_resume.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from ao04_v12_runner import paired_init_phase, train_val_loop  # noqa
from ao04_v12_protocol import BATCH_SIZE  # noqa

DATASET = "movielens"
ARM = "B1"
SEED = 2022
SMOKE = 10 ** 9  # smoke channel (permits epochs != 50) without truncation


def _config(epochs: int) -> dict:
    return {
        "device": "cpu",
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "shadow_patience": 10,
        "dataset_path": None,
    }


def main() -> int:
    tmp_root = tempfile.mkdtemp(prefix="r2_resume_check_")
    try:
        init_dir = os.path.join(tmp_root, "init")
        _bundle, paired = paired_init_phase(dataset=DATASET, seed=SEED,
                                            init_dir=init_dir)
        print("paired_init: hash={}...".format(paired.initial_state_hash[:16]))

        # Run A, part 1: one epoch, then stop (leaves last_resume.pt).
        dir_a = os.path.join(tmp_root, "run_a")
        res_a1 = train_val_loop(
            arm=ARM, dataset=DATASET, seed=SEED, run_dir=dir_a,
            config=_config(1), paired_state_path=paired.init_state_path,
            smoke_max_batches=SMOKE)
        resume_path = os.path.join(dir_a, "last_resume.pt")
        assert os.path.exists(resume_path), "last_resume.pt missing"
        print("run A part 1: epochs_completed={} selected={}".format(
            res_a1.epochs_completed, res_a1.selected_epoch))

        # Run A, part 2: resume into the same run dir up to epoch 2.
        res_a = train_val_loop(
            arm=ARM, dataset=DATASET, seed=SEED, run_dir=dir_a,
            config=_config(2), paired_state_path=paired.init_state_path,
            smoke_max_batches=SMOKE, resume_from=resume_path)
        print("run A resumed: epochs_completed={} resumed_from={}".format(
            res_a.epochs_completed, res_a.resumed_from is not None))

        # Run B: fresh uninterrupted 2-epoch reference.
        dir_b = os.path.join(tmp_root, "run_b")
        res_b = train_val_loop(
            arm=ARM, dataset=DATASET, seed=SEED, run_dir=dir_b,
            config=_config(2), paired_state_path=paired.init_state_path,
            smoke_max_batches=SMOKE)
        print("run B reference: epochs_completed={}".format(
            res_b.epochs_completed))

        checks = []
        checks.append(("batch_order_hash",
                       res_a.batch_order_hash == res_b.batch_order_hash))
        checks.append(("architecture_batch_order_hash",
                       res_a.architecture_batch_order_hash
                       == res_b.architecture_batch_order_hash))
        checks.append(("macro_auc_by_epoch",
                       res_a.macro_auc_by_epoch == res_b.macro_auc_by_epoch))
        checks.append(("selected_epoch",
                       res_a.selected_epoch == res_b.selected_epoch))
        checks.append(("final_parameter_state_hash",
                       res_a.final_parameter_state_hash
                       == res_b.final_parameter_state_hash))
        checks.append(("initial_parameter_state_hash",
                       res_a.initial_parameter_state_hash
                       == res_b.initial_parameter_state_hash))

        ck_a = torch.load(os.path.join(dir_a, "last_resume.pt"),
                          map_location="cpu", weights_only=False)
        ck_b = torch.load(os.path.join(dir_b, "last_resume.pt"),
                          map_location="cpu", weights_only=False)
        state_a = ck_a["model_state_dict"]
        state_b = ck_b["model_state_dict"]
        same_keys = sorted(state_a) == sorted(state_b)
        tensor_equal = same_keys and all(
            torch.equal(state_a[k], state_b[k]) for k in state_a)
        checks.append(("model_state_tensors", tensor_equal))

        opt_a = ck_a["main_optimizer_state_dict"]
        opt_b = ck_b["main_optimizer_state_dict"]
        # q_rho for B1 has 6 elements; find its state entry by numel.
        exp_avg_equal = None
        for pid, st in opt_a["state"].items():
            if "exp_avg" in st and st["exp_avg"].numel() == 6:
                exp_avg_equal = torch.equal(
                    st["exp_avg"], opt_b["state"][pid]["exp_avg"])
                break
        checks.append(("optimizer_exp_avg(q_rho)", bool(exp_avg_equal)))

        ok = True
        for name, passed in checks:
            print("  [{}] {}".format("OK" if passed else "FAIL", name))
            ok = ok and bool(passed)
        print("RESUME CHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
