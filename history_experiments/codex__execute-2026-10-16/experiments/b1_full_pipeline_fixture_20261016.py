"""CPU integration fixture for dense measurement output, never a formal run.

Uses real class-covered row subsets, original full model and all 50 epochs.
No edit to the running scheduler or its frozen worker is needed.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from b1_reconstruction_20261016 import OUT, ROOT, STUDY, run_args, runner_class, save_json, sha
from stage_S1_run import _raw_rng_state
from stage_E1_instrumentation import preserve_rng_state
from MDMTRec.utils.data import TorchDataset


def row_subset(x, y):
    chosen = []
    for domain in range(3):
        for task in range(2):
            for label in (0, 1):
                ids = np.flatnonzero((x[:, -1] == domain) & (y[:, task] == label))
                assert len(ids) >= 4
                chosen.extend(ids[:4])
    return np.unique(chosen)


def main():
    assert not torch.cuda.is_available(), "fixture must run with CUDA_VISIBLE_DEVICES=''"
    torch.set_num_threads(2)
    worker_path = ROOT / "experiments/b1_reconstruction_20261016.py"
    frozen_sha = sha(worker_path)
    base = runner_class()
    fixture_root = OUT / "engineering_full_pipeline" / str(time.time_ns())
    # The production worker is CUDA-only. Zero-valued CUDA memory observations
    # apply only to this isolated CPU fixture, with no formal process affected.
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.max_memory_reserved = lambda *args, **kwargs: 0

    class EngineeringRunner(base):
        def _validate_args(self):
            original = self.args.device
            self.args.device = "cuda:0"
            try:
                super()._validate_args()
            finally:
                self.args.device = original

        def write_outputs(self, info, wall_seconds):
            # The inherited S6 writer rejects CPU in its frozen CUDA contract.
            # Use its serializer, then explicitly mark this fixture nonformal.
            original = self.args.device
            self.args.device = "cuda:0"
            try:
                super().write_outputs(info, wall_seconds)
            finally:
                self.args.device = original

    reports = []
    for arm in ("F", "DH"):
        args = run_args(arm, 2022, fixture_root / arm)
        args.device = "cpu"
        started = time.time()
        r = EngineeringRunner(args)
        b = r.bundle
        tr, va, te = (row_subset(x, y) for x, y in (
            (b.train_x, b.train_y), (b.val_x, b.val_y), (b.test_x, b.test_y)))
        r.bundle = replace(b, train_x=b.train_x[tr], train_y=b.train_y[tr],
                           train_row_ids=b.train_row_ids[tr], val_x=b.val_x[va],
                           val_y=b.val_y[va], val_row_ids=b.val_row_ids[va],
                           test_x=b.test_x[te], test_y=b.test_y[te], test_row_ids=b.test_row_ids[te])
        r.bundle_joint = r.bundle
        for field, x, y, shuffle in (
            ("train_loader", r.bundle.train_x, r.bundle.train_y, True),
            ("val_loader", r.bundle.val_x, r.bundle.val_y, False),
            ("test_loader", r.bundle.test_x, r.bundle.test_y, False)):
            setattr(r, field, DataLoader(TorchDataset(x, y), batch_size=40960,
                                        shuffle=shuffle, num_workers=0, drop_last=False))
        save_json(Path(r.run_dir) / "ENGINEERING_FIXTURE.json", {
            "formal_run": False, "study_id": "B1_ENGINEERING_INTEGRATION",
            "training_subset_rows": len(tr), "validation_subset_rows": len(va),
            "test_subset_rows": len(te), "no_scientific_inference": True,
            "worker_sha256": frozen_sha})
        info = r.fit()
        r.write_outputs(info, time.time() - started)
        for filename in ("config.json", "run_manifest.json"):
            path = Path(r.run_dir) / filename
            value = json.loads(path.read_text())
            value.update(formal_valid=False, engineering_fixture=True,
                         study_id="B1_ENGINEERING_INTEGRATION", device="cpu",
                         formal_protocol_mismatches={"device": {"expected": "cuda:0", "observed": "cpu"},
                             "dataset_exposure": "class-covered small subsets for engineering only"},
                         scope_note="small real-row CPU fixture; no scientific result")
            value["formal_protocol_observed"]["device"] = "cpu"
            if filename == "run_manifest.json":
                value["artifact_hashes"]["config.json"] = sha(Path(r.run_dir) / "config.json")
            save_json(path, value)
        manifest = json.loads((Path(r.run_dir) / "run_manifest.json").read_text())
        assert manifest["dense_state_count"] == 50
        for name, digest in manifest["dense_artifact_hashes"].items():
            assert sha(Path(r.run_dir) / name) == digest
        assert len(list(r.dense_dir.glob("epoch*.pt"))) == 50
        assert len(list(r.dense_dir.glob("test_epoch*.npz"))) == 50
        assert len(list(r.dense_dir.glob("validation_epoch*.npz"))) == 50
        reports.append({"arm": arm, "status": "PASS", "epochs": 50,
            "selected_epoch": info["selected_epoch"], "dense_states": 50,
            "test_prediction_files": 50, "validation_prediction_files": 50,
            "all_dense_hashes_checked": True, "formal_valid": False,
            "elapsed_seconds": time.time() - started, "fixture_path": str(Path(r.run_dir).relative_to(ROOT))})
        del r
    assert sha(worker_path) == frozen_sha
    result = {"status": "PASS", "scope": "CPU_FULL_PIPELINE_ENGINEERING_ONLY",
        "worker_sha256": frozen_sha, "fixture_sha256": sha(__file__), "formal_runs_completed": 0,
        "checks": reports, "timestamp": time.time()}
    save_json(OUT / "full_pipeline_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
