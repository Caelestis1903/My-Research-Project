"""B1 measurement reconstruction. Original S6 experiments remain untouched.

Run with the existing phase0-3-env Python. This is a separately identified
reconstruction with self-review, not an exact historical replication.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDY = "B1_reconstruction_20261016"
OUT = ROOT / "research_plans/2026-10-15/execution_20261015" / STUDY
PYTHON = "/home/xzf/miniconda3/envs/phase0-3-env/bin/python"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def run_args(arm, seed, run_dir):
    from stage_S6_run import build_argparser
    from stage_S6_protocol import key
    return build_argparser().parse_args([
        "--run_type", "joint", "--dataset", "kuairand", "--setting", "natural",
        "--backbone", "m3oe", "--seed", str(seed), "--arm", arm,
        "--run_dir", str(run_dir), "--dataset_path", "/home/xzf/data/kuairand/processed",
        "--formal_run_key", key(seed, arm), "--co_run_pair_id", STUDY + f"_seed{seed}",
        "--planned_peer_run_key", key(seed, "DH" if arm == "F" else "F"),
        "--worker_id", "primary_agent_local_scheduler", "--attempt_id", "1",
    ])


def runner_class():
    import numpy as np
    import torch
    from stage_S6_run import S6Runner, S6_FIELDS, S6_PROBE_FIELDS
    from stage_S6_protocol import INITIAL_PARAMETER_SHA256, KUAIRAND_DATA_SHA256, SAMPLE_TABLE_SHA256
    from stage_S1_run import (_cell_records, _macro, _state_cpu, _raw_rng_state,
                              _rows_for_bundle, _prediction_records, METRICS_EPOCH_FIELDS)
    from stage_E1_instrumentation import preserve_rng_state, rng_state_digest
    from stage_S3_run import write_rows_csv
    from stage_S1_models import initial_state_hash

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu(v) for v in value)
        return copy.deepcopy(value)

    class DenseRunner(S6Runner):
        def __init__(self, args):
            # S1 constructor accepts only stdout.log in a fresh run directory.
            self.measured_dataset_sha = sha(Path(args.dataset_path) / "kuai_rand.csv")
            assert self.measured_dataset_sha == KUAIRAND_DATA_SHA256, "dataset hash differs from frozen protocol"
            super().__init__(args)
            assert sha(Path(args.dataset_path) / "kuai_rand.csv") == self.measured_dataset_sha
            assert self.objective_table["coefficient_table_hash"] == SAMPLE_TABLE_SHA256
            self.measured_init = self.initial_parameter_state_hash
            self.dense_dir = Path(self.run_dir) / "dense"
            self.dense_dir.mkdir(exist_ok=False)
            self.extra_source_hash = sha(__file__)
            self.dense_index = []
            self.arch_creation_rng = None
            self.started_at = time.time()
            main_ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            arch_ids = {id(p) for g in self.arch_optimizer.param_groups for p in g["params"]}
            assert not main_ids & arch_ids
            assert main_ids | arch_ids == {id(p) for p in self.model.parameters()}
            save_json(Path(self.run_dir) / "reconstruction_identity.json", {
                "study_id": STUDY, "identity": "RECONSTRUCTION_NOT_HISTORICAL_EXACT",
                "review": "SELF_REVIEW", "seed": self.seed, "arm": self.arm,
                "actual_initial_parameter_hash": self.measured_init,
                "actual_dataset_sha256": self.measured_dataset_sha,
                "historical_initial_parameter_pin": INITIAL_PARAMETER_SHA256[self.seed],
                "matches_historical_init_pin": self.measured_init == INITIAL_PARAMETER_SHA256[self.seed],
                "source_hashes": dict(self.initial_inventory, **{str(Path(__file__).relative_to(ROOT)): self.extra_source_hash}),
                "dataset": self.bundle.manifest_fields(),
            })
            # Paired new runs must share measured initialization and split identity.
            peer = Path(self.run_dir).parent / ("DH" if self.arm == "F" else "F")
            peer_identity = peer / "reconstruction_identity.json"
            if peer_identity.exists():
                other = json.loads(peer_identity.read_text())
                assert other["actual_initial_parameter_hash"] == self.measured_init
                assert other["dataset"] == self.bundle.manifest_fields()

        def attrs(self):
            return {n: {k: getattr(m, k) for k in ("tau", "tau_step", "softmax_type") if hasattr(m, k)}
                    for n, m in self.model.named_modules() if hasattr(m, "tau")}

        def restore_attrs(self, values):
            modules = dict(self.model.named_modules())
            for name, attrs in values.items():
                for k, v in attrs.items():
                    setattr(modules[name], k, v)

        def _architecture_update(self, iterator, epoch):
            if iterator is None or iterator._num_yielded >= len(self.train_loader):
                self.arch_creation_rng = _raw_rng_state()
            return super()._architecture_update(iterator, epoch)

        def snapshot(self, epoch, iterator):
            before = rng_state_digest()
            model_hash = initial_state_hash(self.model)
            payload = {
                "study_id": STUDY, "epoch": epoch, "model": _state_cpu(self.model),
                "model_python_attributes": self.attrs(),
                "optimizer": cpu(self.optimizer.state_dict()),
                "arch_optimizer": cpu(self.arch_optimizer.state_dict()),
                "optimizer_parameter_names": [[next(n for n, q in self.model.named_parameters() if q is p)
                    for p in group["params"]] for group in self.optimizer.param_groups],
                "arch_optimizer_parameter_names": [[next(n for n, q in self.model.named_parameters() if q is p)
                    for p in group["params"]] for group in self.arch_optimizer.param_groups],
                "rng": _raw_rng_state(), "arch_iterator_creation_rng": self.arch_creation_rng,
                "arch_iterator_batches_consumed": iterator._num_yielded,
                "iterator_restore_recipe": "restore creation RNG; create iter(train_loader); consume saved batch count, preserving caller RNG; then restore end-of-epoch RNG",
                "resume_verified": False, "scheduler": None,
                "selected_epoch_so_far": self.selected_epoch, "selected_val_auc": self.selected_val_auc,
                "selected_state": self.selected_state,
                "validation_records": self.validation_records, "validation_summaries": self.validation_summaries,
                "optimizer_steps": self.optimizer_steps,
                "architecture_optimizer_steps": self.architecture_optimizer_steps,
                "examples_processed": self.examples_processed,
                "architecture_examples_processed": self.architecture_examples_processed,
                "batch_fingerprints": self.batch_fingerprints,
                "architecture_batch_fingerprints": self.architecture_batch_fingerprints,
                "initial_parameter_state_hash": self.measured_init,
            }
            for optimizer, state in ((self.optimizer, payload["optimizer"]),
                                     (self.arch_optimizer, payload["arch_optimizer"])):
                ids = [i for group in state["param_groups"] for i in group["params"]]
                assert len(ids) == len(set(ids))
                # Inactive parameters are explicitly represented by their param-group membership;
                # Adam legitimately has no state until a gradient first arrives.
                assert set(state["state"]).issubset(ids)
                for s in state["state"].values():
                    assert {"step", "exp_avg", "exp_avg_sq"}.issubset(s)
            path = self.dense_dir / f"epoch{epoch:02d}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(payload, path)
            assert before == rng_state_digest(), "dump changed RNG"
            assert model_hash == initial_state_hash(self.model), "dump changed parameters"
            self.dense_index.append({"epoch": epoch, "path": path.name, "sha256": sha(path),
                                     "bytes": path.stat().st_size, "model_hash": model_hash,
                                     "initial_parameter_hash": self.measured_init,
                                     "main_batch_hash": hashlib.sha256(json.dumps(self.batch_fingerprints).encode()).hexdigest(),
                                     "arch_batch_hash": hashlib.sha256(json.dumps(self.architecture_batch_fingerprints).encode()).hexdigest()})
            peer_index = Path(self.run_dir).parent / ("DH" if self.arm == "F" else "F") / "dense/index.json"
            if peer_index.exists():
                peers = json.loads(peer_index.read_text())
                matched = next((v for v in peers if v["epoch"] == epoch), None)
                if matched:
                    for field in ("initial_parameter_hash", "main_batch_hash", "arch_batch_hash"):
                        assert self.dense_index[-1][field] == matched[field], f"paired {field} mismatch at epoch {epoch}"
            save_json(self.dense_dir / "index.json", self.dense_index)
            write_rows_csv(str(Path(self.run_dir) / "metrics_epoch_partial.csv"), self.metric_rows, METRICS_EPOCH_FIELDS)
            save_json(Path(self.run_dir) / "progress.json", {
                "study_id": STUDY, "phase": "TRAIN", "epoch_completed": epoch,
                "epochs_total": 50, "selected_epoch_so_far": self.selected_epoch,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.device.type == "cuda" else 0,
                "elapsed_seconds": time.time() - self.started_at,
            })
            print(json.dumps({"epoch": epoch, "val_macro_auc": self.validation_summaries[epoch]["macro_auc"],
                              "selected_epoch": self.selected_epoch, "state_bytes": path.stat().st_size}), flush=True)

        def fit(self):
            iterator = None
            best_score, best_epoch, bad = -math.inf, None, 0
            for epoch in range(50):
                train_loss, train_records = self._train_epoch(epoch)
                iterator = self._architecture_update(iterator, epoch)
                before = rng_state_digest()
                with preserve_rng_state():
                    records = _cell_records(self.bundle, self.model, self.val_loader, self.device)
                after = rng_state_digest()
                self.rng_evidence.append({"epoch": epoch, "kind": "validation_eval", "before": before,
                                          "after": after, "restored": before == after})
                assert before == after
                score = _macro(records, "auc")
                if not math.isfinite(score):
                    raise FloatingPointError("nonfinite validation macro AUC")
                if score > best_score:
                    best_score, best_epoch, bad = score, epoch, 0
                    self.selected_state = _state_cpu(self.model)
                else:
                    bad += 1
                if self.shadow_stop_epoch is None and bad >= self.args.shadow_patience:
                    self.shadow_stop_epoch = epoch
                if epoch in (9, 19, 29, 39, 49):
                    self.checkpoint_states[epoch] = _state_cpu(self.model)
                self.validation_records[epoch] = records
                self.validation_summaries[epoch] = {"macro_auc": score, "macro_logloss": _macro(records, "logloss")}
                self.metric_rows.extend(_rows_for_bundle(
                    self.bundle, records, "validation", epoch, train_records=train_records,
                    selected=False, fixed_final=epoch == 49, train_objective=train_loss,
                    main_steps=self.optimizer_steps, architecture_steps=self.architecture_optimizer_steps,
                    learning_rate=self.args.learning_rate, architecture_learning_rate=self.args.learning_rate_darts))
                self.selected_epoch, self.selected_val_auc = best_epoch, best_score
                self.snapshot(epoch, iterator)
                self._check_source_drift()
                assert sha(__file__) == self.extra_source_hash
            assert best_epoch is not None
            for row in self.metric_rows:
                row["backbone"] = self.backbone
                if int(row["epoch"]) == best_epoch:
                    row["selected_checkpoint"] = 1
            rules = {}
            for epoch, rec in self.validation_records.items():
                auc = {(d, t): rec[t * 3 + d]["auc"] for d in range(3) for t in range(2)}
                rules[epoch] = [sum(auc.values()) / 6,
                    sum(auc[1, t] for t in range(2)) / 2,
                    sum(auc[d, t] for d in (0, 2) for t in range(2)) / 4,
                    min(sum(auc[d, t] for t in range(2)) / 2 for d in range(3)),
                    sum(rec[t * 3 + d]["sample_count"] * auc[d, t] for d in range(3) for t in range(2))
                    / sum(v["sample_count"] for v in rec.values())]
            selection = {f"R{i}": max(range(50), key=lambda e: (rules[e][i], -e)) for i in range(5)}
            assert selection["R0"] == best_epoch
            save_json(Path(self.run_dir) / "selection_frozen.json", {
                "study_id": STUDY, "selection": selection,
                "test_access_started": False, "validation_summaries": self.validation_summaries,
                "timestamp": time.time(), "retrospective_test": True})
            # Original sparse endpoint products remain available with original schema.
            self._evaluate_deferred_checkpoints()
            test_rows = []
            for epoch in range(50):
                payload = torch.load(self.dense_dir / f"epoch{epoch:02d}.pt", map_location="cpu", weights_only=False)
                self.model.load_state_dict(payload["model"], strict=True)
                self.restore_attrs(payload["model_python_attributes"])
                val_records, val_rows, val_domains, val_labels, val_predictions = _prediction_records(
                    self.bundle, self.model, self.val_loader, self.device, self.bundle.val_row_ids)
                for key, rec in val_records.items():
                    for field in ("auc", "logloss"):
                        assert abs(rec[field] - self.validation_records[epoch][key][field]) < 1e-10
                np.savez_compressed(self.dense_dir / f"validation_epoch{epoch:02d}.npz",
                    row_ids=val_rows, local_domains=val_domains, labels=val_labels, predictions=val_predictions)
                records, rows, domains, labels, predictions = _prediction_records(
                    self.bundle, self.model, self.test_loader, self.device, self.bundle.test_row_ids)
                np.savez_compressed(self.dense_dir / f"test_epoch{epoch:02d}.npz", row_ids=rows,
                    local_domains=domains, labels=labels, predictions=predictions)
                for key, rec in records.items():
                    test_rows.append(dict(rec, epoch=epoch, domain=key % 3, task=key // 3,
                                          study_id=STUDY, arm=self.arm, seed=self.seed))
                write_rows_csv(str(self.dense_dir / "metrics_test.csv"), test_rows, list(test_rows[0]))
                save_json(Path(self.run_dir) / "progress.json", {"phase": "TEST_FROZEN",
                    "test_epoch_completed": epoch, "selected_epoch": best_epoch,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            self.model.load_state_dict(payload["model"])
            self.restore_attrs(payload["model_python_attributes"])
            return {"selected_epoch": best_epoch, "selected_validation_macro_auc": best_score,
                    "shadow_stop_epoch": self.shadow_stop_epoch, "fixed_final_epoch": 49}

        def write_outputs(self, info, wall_seconds):
            assert sha(Path(self.args.dataset_path) / "kuai_rand.csv") == self.measured_dataset_sha
            super().write_outputs(info, wall_seconds)
            # S6 stamps its historical pin; retain the pin separately and replace with measurement.
            for filename in ("config.json", "run_manifest.json"):
                path = Path(self.run_dir) / filename
                value = json.loads(path.read_text())
                value.update(study_id=STUDY, replication_identity="RECONSTRUCTION_NOT_HISTORICAL_EXACT",
                             review="SELF_REVIEW", initial_parameter_state_hash=self.measured_init,
                             historical_initial_parameter_pin=INITIAL_PARAMETER_SHA256[self.seed],
                             reconstruction_source_sha256=self.extra_source_hash,
                             resume_verified=False, dense_state_count=len(self.dense_index))
                if filename == "run_manifest.json":
                    value["artifact_hashes"]["config.json"] = sha(Path(self.run_dir) / "config.json")
                    value["dense_artifact_hashes"] = {str(p.relative_to(self.run_dir)): sha(p)
                        for p in self.dense_dir.iterdir() if p.is_file()}
                save_json(path, value)
            save_json(Path(self.run_dir) / "progress.json", {"phase": "COMPLETE", "epochs": 50,
                "test_epochs": 50, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "wall_seconds": wall_seconds})

    return DenseRunner


def worker(arm, seed, run_dir):
    import torch
    torch.set_num_threads(2)
    started = time.time()
    try:
        runner = runner_class()(run_args(arm, seed, run_dir))
        info = runner.fit()
        runner.write_outputs(info, time.time() - started)
    except BaseException as exc:
        save_json(Path(run_dir) / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started})
        raise


def verify():
    """Actual CPU model, fixed real rows, loss/update/state and iterator checks.

    This is an engineering fixture, never a shortened formal training run.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from stage_S4_gradient_controls import validate_adam_delta_against_clone
    from stage_S1_run import _state_cpu, _raw_rng_state
    from stage_S1_models import initial_state_hash
    from stage_E1_instrumentation import preserve_rng_state, rng_state_digest
    from MDMTRec.utils.data import TorchDataset
    torch.set_num_threads(2)
    checked_source_sha = sha(__file__)
    base = runner_class()

    class EngineeringRunner(base):
        def _validate_args(self):
            # CPU is confined to this fixture. Formal worker never uses this class.
            original = self.args.device
            self.args.device = "cuda:0"
            try:
                super()._validate_args()
            finally:
                self.args.device = original

    assert validate_adam_delta_against_clone(torch.device("cpu"))
    fixture_root = OUT / "engineering" / str(time.time_ns())
    reports = []
    for arm in ("F", "DH"):
        args = run_args(arm, 2022, fixture_root / arm)
        args.device = "cpu"
        r = EngineeringRunner(args)
        before_init = r.measured_init
        ids = np.concatenate([np.flatnonzero(r.bundle.train_x[:, -1] == d)[:8] for d in range(3)])
        x = torch.tensor(r.bundle.train_x[ids], dtype=torch.float32)
        y = torch.tensor(r.bundle.train_y[ids], dtype=torch.float32)
        r.train_loader = DataLoader(TorchDataset(x.numpy(), y.numpy()), batch_size=12,
                                   shuffle=True, num_workers=0)
        # Fixed full 24-row batch includes all domains, preserving original denominator.
        r.model.train()
        prediction = r.model(x.T)
        losses = r._objective_components(prediction, y, x[:, -1].long())
        oracle = torch.nn.functional.binary_cross_entropy(prediction, y, reduction="none").mean(1)
        assert torch.allclose(losses["full"], oracle.mean(), atol=1e-7, rtol=1e-6)
        assert torch.allclose(losses["target"], (oracle * (x[:, -1] == 1)).mean(), atol=1e-7, rtol=1e-6)
        gf = torch.autograd.grad(losses["full"], prediction, retain_graph=True)[0]
        expected_g = torch.autograd.grad(oracle.mean(), prediction, retain_graph=True)[0]
        assert torch.allclose(gf, expected_g, atol=1e-7, rtol=1e-6)
        fingerprint = r._record_batch(0, 0, "main", x, y)
        r._controlled_step(losses, fingerprint, x, 0, 0, "main")
        r.optimizer_steps += 1
        r.examples_processed += len(x)
        iterator = r._architecture_update(None, 0)
        r.selected_state = _state_cpu(r.model)
        r.selected_epoch, r.selected_val_auc = 0, .5
        r.validation_summaries = {0: {"macro_auc": .5, "macro_logloss": .7}}
        state_hash = initial_state_hash(r.model)
        rng_before = rng_state_digest()
        r.snapshot(0, iterator)
        payload = torch.load(r.dense_dir / "epoch00.pt", map_location="cpu", weights_only=False)
        assert payload["initial_parameter_state_hash"] == before_init
        assert state_hash == initial_state_hash(r.model)
        assert rng_before == rng_state_digest()
        for opt, key in ((r.optimizer, "optimizer"), (r.arch_optimizer, "arch_optimizer")):
            actual = opt.state_dict()
            for pid, state in actual["state"].items():
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    assert torch.equal(state[field], payload[key]["state"][pid][field])
        # Rebuild the persistent architecture iterator from its creation RNG.
        def restore_rng(state):
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
        with preserve_rng_state():
            restore_rng(payload["arch_iterator_creation_rng"])
            with preserve_rng_state():
                replay = iter(r.train_loader)
                next(replay)
        with preserve_rng_state():
            actual_next = next(iterator)
        with preserve_rng_state():
            replay_next = next(replay)
        assert all(torch.equal(a, b) for a, b in zip(actual_next, replay_next))
        model_bytes = sum(t.numel() * t.element_size() for t in r.model.state_dict().values())
        reports.append({"arm": arm, "actual_init_hash": before_init,
            "post_update_hash": state_hash, "loss_gradient_oracle": "PASS",
            "controlled_update_invariants": "PASS", "snapshot_rng_parameters_optimizer": "PASS",
            "iterator_replay_next_batch": "PASS", "snapshot_bytes": (r.dense_dir / "epoch00.pt").stat().st_size,
            "model_tensor_bytes": model_bytes})
        del payload, r
    assert reports[0]["actual_init_hash"] == reports[1]["actual_init_hash"]
    assert checked_source_sha == sha(__file__), "verification source changed while executing"
    result = {"study_id": STUDY, "status": "PASS", "scope": "CPU_ENGINEERING_SELF_REVIEW",
        "formal_runs_completed": 0, "source_sha256": checked_source_sha, "checks": reports,
        "estimated_400_checkpoint_bytes": max(v["snapshot_bytes"] for v in reports) * 400,
        "free_disk_bytes": shutil.disk_usage(OUT).free, "timestamp": time.time()}
    save_json(OUT / "engineering_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("worker", "verify"))
    p.add_argument("--arm", choices=("F", "DH"))
    p.add_argument("--seed", type=int, choices=(2022, 2023, 2024, 2025))
    p.add_argument("--run-dir", type=Path)
    a = p.parse_args()
    if a.mode == "verify":
        verify()
    else:
        if a.arm is None or a.seed is None or a.run_dir is None:
            p.error("worker requires --arm --seed --run-dir")
        worker(a.arm, a.seed, a.run_dir)


if __name__ == "__main__":
    main()
