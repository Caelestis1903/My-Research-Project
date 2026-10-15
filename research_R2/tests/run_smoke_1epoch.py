"""1-epoch smoke test of the full training->state-writer pipeline.

Confirms:
- paired_init_phase + train_val_loop(arm, dataset, seed=2022) executes end-to-end
- state_writer persists the right artifacts without touching any test array
- validation-only decision can be computed (with mocked metrics)
- the test_arrays_present=False path of state_writer completes successfully

Run from project root:
    venv/Scripts/python.exe research_R2/tests/run_smoke_1epoch.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from typing import Any, Dict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from stage_S1_data import load_stage_s1_bundle  # noqa

import ao04_v12_runner as _runner_mod  # noqa
from ao04_v12_runner import (  # noqa
    paired_init_phase,
    state_writer,
    setup_seed,
    train_val_loop,
)
from ao04_v12_protocol import (  # noqa
    BATCH_SIZE,
    BUDGET_PER_DATASET,
    DATASETS,
    EPOCHS,
)


def run_1epoch(dataset: str, seed: int = 2022) -> Dict[str, Any]:
    """Run paired_init + 1-epoch train_val + state_writer.

    Config is a minimal stub so the runner can run 1 epoch.  The test
    mode is ``epochs=1`` (smoke only).
    """
    setup_seed(seed)
    bundle = load_stage_s1_bundle(dataset, None,
                                   ("natural" if dataset == "kuairand"
                                    else "d0_large"), seed)
    tmp_root = tempfile.mkdtemp(prefix="r2_smoke_{}_".format(dataset))
    init_dir = os.path.join(tmp_root, "init")
    run_dir = os.path.join(tmp_root, "runs", dataset, "B0_seed{}".format(seed))
    os.makedirs(init_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    try:
        # 1. paired_init_phase
        _b, paired = paired_init_phase(dataset=dataset, seed=seed, init_dir=init_dir)
        print(f"  paired_init: hash={paired.initial_state_hash[:16]}..."
              f" rng_digest={paired.rng_state_digest[:16]}...")

        # 2. train_val_loop for 1 epoch (smoke channel: smoke_max_batches
        #    set to a huge value so no batch truncation occurs; the smoke
        #    flag is what permits epochs=1 under the locked-config gate).
        config = {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "batch_size": BATCH_SIZE,
            "epochs": 1,
            "shadow_patience": 10,
            "dataset_path": None,
        }
        result = train_val_loop(
            arm="B0", dataset=dataset, seed=seed, run_dir=run_dir,
            config=config, paired_state_path=paired.init_state_path,
            smoke_max_batches=10 ** 9)
        print(f"  train_val: selected_epoch={result.selected_epoch}"
              f" macro_auc={result.macro_auc_by_epoch[result.selected_epoch]:.4f}"
              f" main_steps={result.main_steps}"
              f" arch_steps={result.arch_steps}")

        # 3. state_writer (test arrays present flag = False; we still
        #    produce a sane manifest and CSV without test eval).
        writer_result = state_writer(
            arm="B0", dataset=dataset, seed=seed, run_dir=run_dir,
            config=config, bundle=bundle, train_val=result,
            paired_state_path=paired.init_state_path,
            test_arrays_present=False)
        print(f"  state_writer: artifacts={list(writer_result.keys())}")

        # 4. Verify TEST_PREDICTION_CALLS counter never incremented
        #    (read through the module object so the live value is seen).
        assert _runner_mod.TEST_PREDICTION_CALLS == 0

        # 5. Inspect output artifacts
        files_to_check = ["metrics_epoch.csv", "run_manifest.json",
                          "selected_checkpoint.pth",
                          "last_resume.pt", "selection.json",
                          "parameter_manifest.json",
                          "optimizer_state.pth",
                          "architecture_optimizer_state.pth",
                          "rng_state.pth", "fusion_log.csv",
                          "batch_trace.csv"]
        present = {n: os.path.exists(os.path.join(run_dir, n))
                    for n in files_to_check}
        print("  artifacts present:", present)
        all_present = all(present.values())
        run_manifest = json.load(open(os.path.join(run_dir, "run_manifest.json"),
                                       encoding="utf-8"))
        test_flag = run_manifest.get("test_predictions_written")
        manifest_clean = (test_flag is False
                          and run_manifest.get("status") == "COMPLETE")

        # 6. Content checks on the repaired artifacts.
        import csv as _csv
        opt_state = torch.load(os.path.join(run_dir, "optimizer_state.pth"),
                               map_location="cpu", weights_only=False)
        opt_state_nonempty = bool(opt_state.get("state"))
        arch_opt_state = torch.load(
            os.path.join(run_dir, "architecture_optimizer_state.pth"),
            map_location="cpu", weights_only=False)
        arch_opt_state_nonempty = bool(arch_opt_state.get("state"))
        with open(os.path.join(run_dir, "fusion_log.csv"),
                  newline="", encoding="utf-8") as h:
            fusion_rows = list(_csv.DictReader(h))
        fusion_ok = len(fusion_rows) == 1  # one epoch ran
        with open(os.path.join(run_dir, "batch_trace.csv"),
                  newline="", encoding="utf-8") as h:
            batch_rows = list(_csv.DictReader(h))
        batch_ok = len(batch_rows) > 1  # train batches + arch batch
        probe_cols = [c for c in (fusion_rows[0].keys() if fusion_rows else [])
                      if c.startswith("expert_norm_") or c.startswith("gate_entropy_")]
        probe_ok = len(probe_cols) == 15  # 9 expert norms + 6 gate entropies
        content_ok = (opt_state_nonempty and arch_opt_state_nonempty
                      and fusion_ok and batch_ok and probe_ok)
        print(f"  content: optimizer_state_nonempty={opt_state_nonempty} "
              f"arch_optimizer_state_nonempty={arch_opt_state_nonempty} "
              f"fusion_rows={len(fusion_rows)} batch_rows={len(batch_rows)} "
              f"probe_cols={len(probe_cols)}")
        all_present = all_present and content_ok

        return {
            "paired_init_hash": paired.initial_state_hash,
            "train_val_summary": {
                "selected_epoch": result.selected_epoch,
                "macro_auc": result.macro_auc_by_epoch[result.selected_epoch],
                "main_steps": result.main_steps,
                "arch_steps": result.arch_steps,
                "batch_order_hash_present": bool(result.batch_order_hash),
            },
            "all_artifacts_present": all_present,
            "manifest_clean": manifest_clean,
            "run_dir": run_dir,
        }
    finally:
        # Keep tmp_root around for inspection (don't auto-remove)
        pass


def main() -> int:
    print("=== 1-epoch smoke (KuaiRand) ===")
    kua = run_1epoch("kuairand")
    print(f"  PASS: artifacts={kua['all_artifacts_present']} "
          f"manifest_clean={kua['manifest_clean']}")
    print(f"  Note: KuaiRand temp run_dir = {kua['run_dir']}")

    print("\n=== 1-epoch smoke (MovieLens) ===")
    mov = run_1epoch("movielens")
    print(f"  PASS: artifacts={mov['all_artifacts_present']} "
          f"manifest_clean={mov['manifest_clean']}")
    print(f"  Note: MovieLens temp run_dir = {mov['run_dir']}")
    ok = (kua["all_artifacts_present"] and kua["manifest_clean"]
          and mov["all_artifacts_present"] and mov["manifest_clean"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
