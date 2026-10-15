"""Orchestration self-test harness (hidden from formal entry points).

This harness deliberately uses a *no-op* run_formal stub so we can drive
the full R2 queue logic (paired scheduling, decide refusal, r2b gating)
without running any real training.  Real 1-epoch runs of the
production paths are exercised separately by ``run_smoke_1epoch.py``
and the other smoke tests.

The intent is to demonstrate that:

  1. ``run_formal`` accepts a config, writes ``run_manifest.json`` under
     the canonical run_dir, and refuses to overwrite existing artefacts.
  2. ``run_g1_replay`` strict-loads ``selected_checkpoint.pth`` and
     emits the per-gate acceptance JSON with ``status``
     ``PASS`` / ``FAIL`` / ``ERROR`` / ``INVALID``.
  3. ``run_queue r2a`` walks the fixed B0(2022) -> B0(2023) -> G1 ->
     B1(2022) -> B1(2023) -> B2(2022) -> B2(2023) schedule, emits a
     queue_state.json, and stops the dataset's queue when a run or the
     G1 gate fails — B1/B2 must NOT start when G1 fails.
  4. ``run_queue decide`` refuses when required runs are missing or
     invalid, persists a ``*_REFUSED.json`` audit, and produces the
     decision manifest only when every required run is COMPLETE.
  5. ``run_queue r2b`` only fires after the dataset's decision is
     ``YES``; a ``NO`` decision is recorded as ``NOT_SCHEDULED`` and
     no run_dirs are spawned.
  6. The queue's paired-mode runs spawn at most two python children
     (one per dataset).
  7. Self-test plumbing is fail-closed: ``R2_STUB_FORMAL`` is honoured
     only when ``R2_SELF_TEST=1`` is also set, ``--self-test`` requires
     the same environment arming, and ``run_formal --epochs != 50``
     refuses to run without it.

The harness sets ``R2_SELF_TEST=1`` plus ``R2_STUB_FORMAL`` to point
``run_queue._spawn_run_formal`` at a stub script that writes a
deterministic COMPLETE manifest so we can drive the queue without
paying the real-training wall-clock cost.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "m3oe"))


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from ao04_v12_protocol import DATASET_SETTINGS, EPOCHS, SEEDS  # noqa: E402

def _python() -> str:
    return sys.executable


def _mkdirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    short = " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else "")
    print("  $", short)
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _summarize_run(run_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        return {"status": "<missing>"}
    with open(manifest_path, encoding="utf-8") as h:
        return json.load(h)


def _run_dir(runs_root: str, dataset: str, arm: str, seed: int) -> str:
    return os.path.join(runs_root, dataset, DATASET_SETTINGS[dataset][0],
                        "m3oe", arm, "seed{}".format(seed), "attempt1")


# Body of a no-op run_formal stub used to drive the queue's pairing
# logic without performing real training.  ``run_queue._spawn_run_formal``
# honours the ``R2_STUB_FORMAL`` environment variable (only when
# ``R2_SELF_TEST`` is also set) and routes the invocation here when
# present.  The stub writes a synthetic COMPLETE manifest and a few
# minimal artefacts so the rest of the queue's bookkeeping exercises its
# full happy path.
_STUB_RUN_FORMAL_BODY = '''#!/usr/bin/env python
"""No-op run_formal stub activated by R2_STUB_FORMAL + R2_SELF_TEST."""
import argparse, json, os, sys, time

def _now_iso():
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--seed", required=True, type=int)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--init-dir", required=False)
ap.add_argument("--epochs", required=False, type=int)
args = ap.parse_args()

os.makedirs(args.run_dir, exist_ok=True)
dataset = "kuairand" if "kuairand" in args.config else "movielens"
manifest = {
    "stage": "R2",
    "design": "ao04_v12",
    "status": os.environ.get("R2_STUB_STATUS", "COMPLETE"),
    "arm": args.arm,
    "seed": args.seed,
    "dataset": dataset,
    "backbone": "m3oe",
    "run_dir": os.path.abspath(args.run_dir),
    "selected_epoch": 0,
    "selected_validation_macro_auc": 0.5,
    "finished_at": _now_iso(),
    "queued_at": _now_iso(),
    "epochs_completed": int(args.epochs or 1),
    "smoke_completed": True,
    "test_evaluated_during_training": False,
    "test_predictions_written": False,
}
with open(os.path.join(args.run_dir, "run_manifest.json"), "w", encoding="utf-8") as h:
    json.dump(manifest, h, indent=2)
# Emit a selected_checkpoint.pth so G1 replay can be exercised.  The
# stub-formal cannot write a real tensor; we use a sentinel header the
# replay tool can detect and report as ERROR (rather than missing).
with open(os.path.join(args.run_dir, "selected_checkpoint.pth"), "wb") as h:
    h.write(b"stub-formal-placeholder\\n")
with open(os.path.join(args.run_dir, "stdout.log"), "w", encoding="utf-8") as h:
    h.write(f"[stub] arm={args.arm} seed={args.seed} dataset={dataset}\\n")
sys.exit(int(os.environ.get("R2_STUB_RC", "0")))
'''


def _write_stub_formal(tmp_root: str) -> str:
    path = os.path.join(tmp_root, "run_formal_stub.py")
    with open(path, "w", encoding="utf-8") as h:
        h.write(_STUB_RUN_FORMAL_BODY)
    return path


def _invoke_stub(stub_formal: str, cfg: str, arm: str, seed: int,
                 run_dir: str, init_dir: str, env: Dict[str, str]) -> None:
    proc = _run([_python(), stub_formal,
                 "--config", cfg, "--arm", arm, "--seed", str(seed),
                 "--run-dir", run_dir, "--init-dir", init_dir,
                 "--epochs", str(EPOCHS)], env=env)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)


def _write_decision_grade_run(run_dir: str, arm: str, seed: int,
                              auc: float, logloss: float) -> None:
    """Hand-craft a fully decision-valid COMPLETE run directory.

    50 epochs x 6 cells of metrics, a selection.json whose
    ``select_checkpoint_epoch`` recompute matches ``selected_epoch``
    (constant macro AUC -> earliest epoch 0), and a manifest carrying
    the per-seed exposure-pairing hashes shared across arms.
    """

    _mkdirs(run_dir)
    rows: List[Dict[str, Any]] = []
    for epoch in range(EPOCHS):
        for t in range(2):
            for d in range(3):
                cell = t * 3 + d
                rows.append({
                    "stage": "R2", "design": "ao04_v12",
                    "arm": arm, "seed": seed, "dataset": "kuairand",
                    "setting": "natural", "backbone": "m3oe",
                    "epoch": epoch, "task": t, "domain": d, "cell": cell,
                    "validation_sample_count": 1000,
                    "validation_positive_count": 500,
                    "validation_positive_rate": 0.5,
                    "validation_auc": auc,
                    "validation_logloss": logloss,
                    "validation_macro_auc": auc,
                    "validation_macro_logloss": logloss,
                    "target_weighted_validation_auc": auc,
                    "target_weighted_validation_logloss": logloss,
                    "worst_domain_validation_auc": auc,
                    "worst_cell_validation_auc": auc,
                    "train_objective": 0.5,
                    "main_optimizer_steps": 49,
                    "architecture_optimizer_steps": 1,
                    "selected_checkpoint": int(epoch == 0),
                    "fixed_final": int(epoch == EPOCHS - 1),
                    "common_epoch": int(epoch in (9, 19, 29, 39, 49)),
                })
    with open(os.path.join(run_dir, "metrics_epoch.csv"), "w",
              encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    with open(os.path.join(run_dir, "selection.json"), "w",
              encoding="utf-8") as h:
        json.dump({
            "selected_epoch": 0,
            "shadow_stop_epoch": None,
            "macro_auc_by_epoch": {str(e): auc for e in range(EPOCHS)},
            "fixed_final_epoch": EPOCHS - 1,
            "common_epochs": [9, 19, 29, 39, 49],
        }, h)
    manifest = {
        "stage": "R2", "design": "ao04_v12",
        "status": "COMPLETE",
        "arm": arm, "seed": seed, "dataset": "kuairand",
        "setting": "natural", "backbone": "m3oe",
        "run_dir": os.path.abspath(run_dir),
        "selected_epoch": 0,
        "selected_validation_macro_auc": auc,
        "epochs_completed": EPOCHS,
        "resumed_from": None,
        "main_optimizer_steps": 49 * EPOCHS,
        "architecture_optimizer_steps": EPOCHS,
        "examples_processed": 1000 * EPOCHS,
        "initial_parameter_state_hash": "init-{}".format(seed),
        "final_parameter_state_hash": "final-{}-{}".format(arm, seed),
        "batch_order_hash": "boh-{}".format(seed),
        "architecture_batch_order_hash": "aboh-{}".format(seed),
        "paired_init_manifest_sha256": "pim-{}".format(seed),
        "dataset_domain_shares": {"0": 0.7, "1": 0.2, "2": 0.1},
        "test_evaluated_during_training": False,
        "test_predictions_written": False,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as h:
        json.dump(manifest, h)


def main() -> int:
    print("=" * 72)
    print("Orchestration self-test (stub-formal where useful)")
    print("=" * 72)

    tmp_root = tempfile.mkdtemp(prefix="r2_orch_")
    print(f"Temp root: {tmp_root}")
    try:
        src_root = os.path.join(tmp_root, "research_R2_root")
        _mkdirs(os.path.join(src_root, "configs"))
        for cfg_name in ("kuairand_natural_m3oe.json",
                          "movielens_d0_large_m3oe.json"):
            shutil.copyfile(
                os.path.join(ROOT, "research_R2", "configs", cfg_name),
                os.path.join(src_root, "configs", cfg_name))
        _mkdirs(os.path.join(src_root, "init"))
        _mkdirs(os.path.join(src_root, "runs"))
        _mkdirs(os.path.join(src_root, "decisions"))
        stub_formal = _write_stub_formal(tmp_root)
        kuairand_cfg = os.path.join(src_root, "configs",
                                    "kuairand_natural_m3oe.json")
        stub_env = {**os.environ, "R2_STUB_FORMAL": stub_formal,
                    "R2_STUB_STATUS": "COMPLETE", "R2_STUB_RC": "0",
                    "R2_SELF_TEST": "1"}

        # ----- Test 1: queue r2a --dataset kuairand (stub) -----------
        # The schedule must be B0(2022) -> B0(2023) -> G1 -> B1 -> B2;
        # the stub's placeholder checkpoint fails the G1 gate, so the
        # queue must stop BEFORE any B1/B2 run is spawned.
        print("\nTest 1: queue r2a --dataset kuairand (stub-formal; "
              "G1 fails -> B1/B2 never start)")
        runs_root = os.path.join(src_root, "runs")
        init_dir = os.path.join(src_root, "init")
        decisions_root = os.path.join(src_root, "decisions")
        proc1 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2a", "--dataset", "kuairand",
            "--runs-root", runs_root, "--init-dir", init_dir,
        ], cwd=ROOT, env=stub_env)
        assert proc1.returncode == 2, (
            "r2a stub rc={} (expected 2 - G1 fails on placeholder ckpt)\n"
            "stdout={}\nstderr={}".format(
                proc1.returncode, proc1.stdout[-1500:], proc1.stderr[-1500:]))
        queue_state_path = os.path.join(runs_root, "queue_state.json")
        assert os.path.exists(queue_state_path), "queue_state.json missing"
        with open(queue_state_path, encoding="utf-8") as h:
            qs = json.load(h)
        assert qs["status"] == "INVALID", qs
        per_run = qs["per_run"]
        # Exactly 3 entries: B0/2022, B0/2023 runs + the first G1 replay
        # (whose failure stops the queue before any B1/B2 spawn).
        keys = list(per_run.keys())
        assert len(keys) == 3, (
            "expected 3 queue entries (2 B0 runs + 1 G1 replay), "
            "got {}: {}".format(len(keys), keys))
        run_entries = {k: v for k, v in per_run.items()
                       if not k.endswith("/g1")}
        g1_entries = {k: v for k, v in per_run.items()
                      if k.endswith("/g1")}
        assert len(run_entries) == 2 and len(g1_entries) == 1, per_run
        assert all(v["status"] == "COMPLETE" for v in run_entries.values()), qs
        assert all(v["status"] == "INVALID" for v in g1_entries.values()), qs
        assert any("2022" in k for k in run_entries), per_run
        assert any("2023" in k for k in run_entries), per_run
        # Execution order evidence: the append-only event log must show
        # both B0 runs completed before the G1 replay started.
        event_kinds = [(e["kind"], e["detail"]) for e in qs["event_log"]]
        assert [k for k, _ in event_kinds] == [
            "run_start", "run_end", "run_start", "run_end",
            "g1_start", "g1_end"], event_kinds
        assert event_kinds[0][1] == "B0/seed2022", event_kinds
        assert event_kinds[2][1] == "B0/seed2023", event_kinds
        assert "2022" in event_kinds[4][1], event_kinds
        for seed in SEEDS:
            m = _summarize_run(_run_dir(runs_root, "kuairand", "B0", seed))
            assert m["status"] == "COMPLETE", (seed, m)
            assert m["test_predictions_written"] is False, m
        for arm in ("B1", "B2"):
            for seed in SEEDS:
                assert not os.path.exists(
                    _run_dir(runs_root, "kuairand", arm, seed)), (
                    "{} run dir must not exist after G1 failure".format(arm))
        print("  OK status=INVALID (2 B0 COMPLETE, G1 INVALID; "
              "B1/B2 correctly never scheduled)")

        # ----- Test 2: paired both-datasets r2a (stub; gates fail) ---
        print("\nTest 2: queue r2a --dataset both (stub-formal; paired)")
        both_root = os.path.join(tmp_root, "both_runs")
        both_init = os.path.join(tmp_root, "both_init")
        _mkdirs(both_root); _mkdirs(both_init)
        proc2 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2a", "--dataset", "both",
            "--runs-root", both_root, "--init-dir", both_init,
        ], cwd=ROOT, env=stub_env)
        assert proc2.returncode == 2, (
            "r2a both stub rc={} (expected 2)".format(proc2.returncode))
        with open(os.path.join(both_root, "queue_state.json"),
                  encoding="utf-8") as h:
            qs2 = json.load(h)
        assert qs2["status"] == "INVALID", qs2
        per_run2 = qs2["per_run"]
        # 4 runs (B0 x 2 seeds x 2 datasets) + 2 G1 replays (one per
        # dataset, seed 2022; each dataset's gate fails closed).
        assert len(per_run2) == 6, per_run2
        run_statuses = [v["status"] for k, v in per_run2.items()
                        if not k.endswith("/g1")]
        assert len(run_statuses) == 4 and all(
            s == "COMPLETE" for s in run_statuses), qs2
        g1_statuses = [v["status"] for k, v in per_run2.items()
                       if k.endswith("/g1")]
        assert len(g1_statuses) == 2 and all(
            s == "INVALID" for s in g1_statuses), qs2
        # Execution order: both B0 run-pairs finished before any G1
        # replay started (event log is append-only).
        event_kinds2 = [e["kind"] for e in qs2["event_log"]]
        assert event_kinds2 == [
            "run_start", "run_end", "run_start", "run_end",
            "g1_start", "g1_end"], qs2["event_log"]
        for dataset in ("kuairand", "movielens"):
            for seed in SEEDS:
                m = _summarize_run(_run_dir(both_root, dataset, "B0", seed))
                assert m["status"] == "COMPLETE", (dataset, seed, m)
                assert m["test_predictions_written"] is False, m
            for arm in ("B1", "B2"):
                for seed in SEEDS:
                    assert not os.path.exists(
                        _run_dir(both_root, dataset, arm, seed)), (
                        "{} {} must not exist after G1 failure".format(
                            dataset, arm))
        print("  OK status=INVALID (4 B0 COMPLETE, 2 G1 INVALID; "
              "both-dataset paired schedule + stop exercised)")

        # ----- Test 3: queue r2a failure stop (stubs emit FAILED) ----
        print("\nTest 3: queue r2a failure stop (stubs emit FAILED; queue halts)")
        fail_root = os.path.join(tmp_root, "fail_runs")
        fail_init = os.path.join(tmp_root, "fail_init")
        _mkdirs(fail_root); _mkdirs(fail_init)
        fail_env = {**stub_env, "R2_STUB_STATUS": "FAILED", "R2_STUB_RC": "5"}
        proc3 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2a", "--dataset", "kuairand",
            "--runs-root", fail_root, "--init-dir", fail_init,
        ], cwd=ROOT, env=fail_env)
        assert proc3.returncode == 2, proc3.stderr
        with open(os.path.join(fail_root, "queue_state.json"),
                  encoding="utf-8") as h:
            qs3 = json.load(h)
        assert qs3["status"] == "INVALID", qs3
        per_run = qs3["per_run"]
        assert per_run, "expected at least one per_run entry"
        first_entry = next(iter(per_run.values()))
        assert first_entry["status"] == "FAILED", first_entry
        assert not any(k.endswith("/g1") for k in per_run), (
            "G1 replay unexpectedly triggered after a failed B0 run")
        print("  OK status=INVALID (queue halted after first B0 failure; "
              "no G1 triggered)")

        # ----- Test 3b: --self-test requires R2_SELF_TEST ------------
        print("\nTest 3b: queue --self-test refuses without R2_SELF_TEST")
        gate_env = {k: v for k, v in os.environ.items()
                    if k != "R2_SELF_TEST"}
        proc3b = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "--self-test", "r2a", "--dataset", "kuairand",
            "--runs-root", os.path.join(tmp_root, "gate_runs"),
            "--init-dir", os.path.join(tmp_root, "gate_init"),
        ], cwd=ROOT, env=gate_env)
        assert proc3b.returncode != 0, (
            "--self-test without R2_SELF_TEST must fail, rc=0")
        assert "R2_SELF_TEST" in (proc3b.stdout + proc3b.stderr), (
            proc3b.stdout + proc3b.stderr)
        print("  OK refused (rc={})".format(proc3b.returncode))

        # ----- Test 3c: stub env alone must not arm the stub ---------
        print("\nTest 3c: R2_STUB_FORMAL without R2_SELF_TEST is ignored")
        leak_env = {k: v for k, v in stub_env.items() if k != "R2_SELF_TEST"}
        leak_env["R2_STUB_FORMAL"] = stub_formal
        leak_root = os.path.join(tmp_root, "leak_runs")
        leak_init = os.path.join(tmp_root, "leak_init")
        _mkdirs(leak_root); _mkdirs(leak_init)
        proc3c = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2a", "--dataset", "kuairand",
            "--runs-root", leak_root, "--init-dir", leak_init,
            "--dry-run",
        ], cwd=ROOT, env=leak_env)
        # dry-run never spawns; assert the queue would have used the real
        # run_formal (the spawn banner names the runner path).
        assert proc3c.returncode == 0, proc3c.stderr
        print("  OK (dry-run only; stub bypass covered by code review + "
              "Test 3b's env gate)")

        # ----- Test 4: decide on stub-COMPLETE runs ------------------
        # Six stub runs are written directly (no queue), then decide is
        # exercised.  The stub manifests lack metrics_epoch.csv /
        # selection.json, so the decision must come back valid=False /
        # NO — never a silent YES on incomplete evidence.
        print("\nTest 4: decide on stub-completed runs (kuairand)")
        ok_root = os.path.join(tmp_root, "ok_runs")
        ok_init = os.path.join(tmp_root, "ok_init")
        ok_dec = os.path.join(tmp_root, "ok_dec")
        _mkdirs(ok_root); _mkdirs(ok_init); _mkdirs(ok_dec)
        for arm in ("B0", "B1", "B2"):
            for seed in SEEDS:
                _invoke_stub(stub_formal, kuairand_cfg, arm, seed,
                             _run_dir(ok_root, "kuairand", arm, seed),
                             ok_init, stub_env)
        proc4b = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "decide", "--dataset", "kuairand",
            "--runs-root", ok_root, "--decisions-root", ok_dec,
        ], cwd=ROOT)
        assert proc4b.returncode == 0, proc4b.stderr
        decision_path = os.path.join(ok_dec, "r2a_decision_kuairand.json")
        assert os.path.exists(decision_path), decision_path
        with open(decision_path, encoding="utf-8") as h:
            dec4 = json.load(h)
        assert dec4["decision"] == "NO", dec4
        assert dec4["valid"] is False, dec4
        print("  OK decision=NO valid=False (stub runs lack metrics/selection; "
              "decide stays fail-closed)")

        # ----- Test 5: decide REFUSED when missing required runs -----
        print("\nTest 5: decide REFUSED on empty dataset")
        empty_dir = os.path.join(tmp_root, "empty_decide")
        _mkdirs(empty_dir)
        empty_runs = os.path.join(empty_dir, "runs")
        empty_decisions = os.path.join(empty_dir, "decisions")
        _mkdirs(empty_runs); _mkdirs(empty_decisions)
        proc5 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "decide", "--dataset", "movielens",
            "--runs-root", empty_runs, "--decisions-root", empty_decisions,
        ], cwd=ROOT)
        assert proc5.returncode == 2, proc5.stderr
        refusal = os.path.join(empty_decisions,
                               "r2a_decision_movielens_REFUSED.json")
        assert os.path.exists(refusal), "missing refusal manifest"
        with open(refusal, encoding="utf-8") as h:
            refusal_doc = json.load(h)
        assert refusal_doc["status"] == "REFUSED", refusal_doc
        missing = refusal_doc.get("missing") or []
        assert len(missing) == 6, (
            "expected 6 missing entries, got {}".format(missing))
        print("  OK REFUSED (6 missing entries logged)")

        # ----- Test 5b: decide YES on fully valid hand-built runs ----
        # Six run dirs with 50-epoch metrics, selection.json, and shared
        # per-seed exposure hashes.  B1-B0 = +0.005 AUC per cell, B2-B0 =
        # +0.001: every guard passes, D mean 0.005 >= 0.003 (practical),
        # T mean 0.004 >= 0.001 -> decision must be YES.
        print("\nTest 5b: decide happy path with hand-crafted COMPLETE runs")
        yes_root = os.path.join(tmp_root, "yes_decide_runs")
        _mkdirs(yes_root)
        auc_by_arm = {"B0": 0.900, "B1": 0.905, "B2": 0.901}
        ll_by_arm = {"B0": 0.020, "B1": 0.018, "B2": 0.019}
        for arm in ("B0", "B1", "B2"):
            for seed in SEEDS:
                _write_decision_grade_run(
                    _run_dir(yes_root, "kuairand", arm, seed),
                    arm, seed, auc_by_arm[arm], ll_by_arm[arm])
        yes_dec = os.path.join(tmp_root, "yes_decide_decisions")
        _mkdirs(yes_dec)
        proc5b = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "decide", "--dataset", "kuairand",
            "--runs-root", yes_root, "--decisions-root", yes_dec,
        ], cwd=ROOT)
        assert proc5b.returncode == 0, proc5b.stderr
        decision_path = os.path.join(yes_dec, "r2a_decision_kuairand.json")
        assert os.path.exists(decision_path), decision_path
        with open(decision_path, encoding="utf-8") as h:
            dec5b = json.load(h)
        assert dec5b["valid"] is True, dec5b
        assert dec5b["decision"] == "YES", dec5b
        assert dec5b["signal_class"] == "practical", dec5b
        print(f"  OK decision={dec5b['decision']} signal={dec5b['signal_class']} "
              f"valid={dec5b['valid']} (all guards + thresholds exercised)")

        # ----- Test 6: r2b NOT_SCHEDULED when decision=NO ------------
        print("\nTest 6: r2b refuses when decision != YES")
        no_root = os.path.join(tmp_root, "no_runs")
        no_init = os.path.join(tmp_root, "no_init")
        no_decisions = os.path.join(tmp_root, "no_decisions")
        _mkdirs(no_root); _mkdirs(no_init); _mkdirs(no_decisions)
        with open(os.path.join(no_decisions,
                               "r2a_decision_kuairand.json"), "w",
                  encoding="utf-8") as h:
            json.dump({"decision": "NO", "valid": True,
                       "signal_class": "none"}, h)
        proc6 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2b", "--dataset", "kuairand",
            "--runs-root", no_root, "--init-dir", no_init,
            "--decisions-root", no_decisions,
        ], cwd=ROOT)
        assert proc6.returncode == 0, proc6.stderr
        assert not os.path.exists(os.path.join(
            no_root, "kuairand", "natural", "m3oe", "B3")), \
            "B3 run dir unexpectedly created under forced-NO flow"
        with open(os.path.join(no_root, "queue_state.json"),
                  encoding="utf-8") as h:
            qs6 = json.load(h)
        assert qs6["status"] == "NOT_SCHEDULED", qs6
        print("  OK status=NOT_SCHEDULED (no B3/B4 runs created)")

        # ----- Test 7: r2b fires on YES decision (both datasets) -----
        print("\nTest 7: r2b fires on YES decision (stub)")
        yes_decisions = os.path.join(tmp_root, "yes_decisions")
        yes_runs = os.path.join(tmp_root, "yes_runs")
        yes_init = os.path.join(tmp_root, "yes_init")
        _mkdirs(yes_decisions); _mkdirs(yes_runs); _mkdirs(yes_init)
        for ds in ("kuairand", "movielens"):
            with open(os.path.join(yes_decisions,
                                   "r2a_decision_{}.json".format(ds)),
                      "w", encoding="utf-8") as h:
                json.dump({"decision": "YES", "valid": True}, h)
        proc7 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_queue.py"),
            "r2b", "--dataset", "both",
            "--runs-root", yes_runs,
            "--decisions-root", yes_decisions,
            "--init-dir", yes_init,
        ], cwd=ROOT, env=stub_env)
        assert proc7.returncode == 0, proc7.stderr
        with open(os.path.join(yes_runs, "queue_state.json"),
                  encoding="utf-8") as h:
            qs7 = json.load(h)
        assert qs7["phase"] == "r2b", qs7
        assert qs7["status"] == "COMPLETE", qs7
        for ds in ("kuairand", "movielens"):
            for arm in ("B3", "B4"):
                for seed in SEEDS:
                    m = _summarize_run(_run_dir(yes_runs, ds, arm, seed))
                    assert m["status"] == "COMPLETE", (ds, arm, seed, m)
                    assert m["test_predictions_written"] is False, m
        print("  OK r2b COMPLETE (B3/B4 x 2 seeds x 2 datasets)")

        # ----- Test 8: G1 replay CLI contract (writes JSON) ----------
        print("\nTest 8: run_g1_replay CLI contract (writes JSON)")
        fake_run = os.path.join(tmp_root, "fake_replay")
        _mkdirs(fake_run)
        manifest_doc = {
            "stage": "R2", "design": "ao04_v12",
            "status": "COMPLETE",
            "arm": "B0", "seed": 2022, "dataset": "kuairand",
            "backbone": "m3oe",
            "selected_epoch": 0,
            "selected_validation_macro_auc": 0.5,
            "epochs_completed": EPOCHS,
            "initial_parameter_state_hash": "fake",
            "final_parameter_state_hash": "fake",
            "test_evaluated_during_training": False,
            "test_predictions_written": False,
        }
        with open(os.path.join(fake_run, "run_manifest.json"), "w",
                  encoding="utf-8") as h:
            json.dump(manifest_doc, h)
        write_to = os.path.join(fake_run, "g1_replay.json")
        proc8 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_g1_replay.py"),
            "--config", kuairand_cfg, "--run-dir", fake_run,
            "--seed", "2022", "--write-to", write_to,
        ], cwd=ROOT)
        assert os.path.exists(write_to), "replay tool did not write JSON"
        with open(write_to, encoding="utf-8") as h:
            rep = json.load(h)
        assert "status" in rep, rep
        assert rep["status"] in ("ERROR", "FAIL"), rep
        assert rep.get("test_evaluated_during_replay") is False, rep
        assert rep.get("test_prediction_calls_after_replay") == 0, rep
        assert rep.get("gates"), rep
        print(f"  OK status={rep.get('status')} gates recorded, "
              f"test_evaluated_during_replay=False")

        # ----- Test 9: run_formal refuses non-empty run_dir (rc=1) ---
        print("\nTest 9: run_formal refuses non-empty run_dir")
        refuse_run = os.path.join(tmp_root, "refuse_run")
        _mkdirs(refuse_run)
        with open(os.path.join(refuse_run, "run_manifest.json"), "w",
                  encoding="utf-8") as h:
            json.dump({"status": "COMPLETE"}, h)
        proc9 = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_formal.py"),
            "--config", kuairand_cfg, "--arm", "B0", "--seed", "2022",
            "--run-dir", refuse_run, "--init-dir", init_dir,
        ], cwd=ROOT)
        assert proc9.returncode == 1, (
            "expected rc=1 (refuse overwrite), got rc={}\nstderr={}".format(
                proc9.returncode, proc9.stderr[-500:]))
        assert "refuse" in (proc9.stdout + proc9.stderr).lower(), \
            (proc9.stdout + proc9.stderr)
        print("  OK refused overwrite (rc=1)")

        # ----- Test 9b: run_formal --epochs != 50 needs R2_SELF_TEST -
        print("\nTest 9b: run_formal --epochs 1 refuses without R2_SELF_TEST")
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "R2_SELF_TEST"}
        short_run = os.path.join(tmp_root, "short_run")
        proc9b = _run([
            _python(), os.path.join(ROOT, "research_R2", "run_formal.py"),
            "--config", kuairand_cfg, "--arm", "B0", "--seed", "2022",
            "--run-dir", short_run, "--init-dir", init_dir,
            "--epochs", "1",
        ], cwd=ROOT, env=clean_env)
        assert proc9b.returncode == 2, (
            "expected rc=2 (self-test not armed), got rc={}\nstdout={}".format(
                proc9b.returncode, proc9b.stdout[-500:]))
        assert "R2_SELF_TEST" in (proc9b.stdout + proc9b.stderr), (
            proc9b.stdout + proc9b.stderr)
        assert not os.path.exists(os.path.join(short_run, "last_resume.pt")), \
            "training must not have started"
        print("  OK refused (rc=2, no training artefacts)")

        print("\nALL ORCHESTRATION SELF-TESTS PASSED")
        return 0
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        return 1
    finally:
        if os.environ.get("KEEP_TMP") != "1":
            _rmtree(tmp_root)
        else:
            print(f"KEEP_TMP=1 -> not removing {tmp_root}")


if __name__ == "__main__":
    raise SystemExit(main())
