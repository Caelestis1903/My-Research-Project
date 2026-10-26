"""Dual-process staggered orchestrator for the P7 backbone / D-T matrix.

Three task groups (seeds 2022..2026 each):
  a) backbone validation:
       backbone in {mmoe, ple, star, sharebottom}
       momentum_mode = none, domain_version = three
       4 * 5 = 20 tasks
  b) component-1 port:
       same 4 backbones, momentum_mode = per_domain_only,
       domain_version = three  (p7_run.py's --fix_optim_grouping defaults
       to 1, so we do not pass it explicitly)
       4 * 5 = 20 tasks
  c) D/T extension (D=5):
       backbone in {mdmtoe, mmoe}, domain_version = five,
       momentum_mode in {none, per_domain_only}
       2 * 2 * 5 = 20 tasks

Total: 60 tasks. Up to 2 concurrent subprocesses; every new launch is
staggered by >=30 s (STAGGER_SEC).

Resume-safe via experiments/results/p7/P7_backbones.csv (p7_run.py
appends one row per successful run). p7_run.py prints
"[p7] {<row dict>}" on success, which this orchestrator greps from the
log to extract test_auc_mean for the [done] status line. p7_run.py does
not write a row on failure, so the natural absence is the resume
signal.

All child stdout/stderr is redirected 100% to an individual file under
experiments/logs/. The orchestrator does NOT write a separate CSV; the
P7_backbones.csv is the single source of truth.

Usage:
  python -u experiments/p7_orchestrator.py [--concurrency 2] [--max_runs N]
"""
import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
RUN = os.path.join(BASE, "p7_run.py")
CWD = os.path.join(ROOT, "m3oe", "examples")
OUTDIR = os.path.join(BASE, "results", "p7")
CSV_PATH = os.path.join(OUTDIR, "P7_backbones.csv")
LOGDIR = os.path.join(BASE, "logs")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

STAGGER_SEC = 30
SEEDS = [2022, 2023, 2024, 2025, 2026]

BACKBONES_REPRO = ["mmoe", "ple", "star", "sharebottom"]
BACKBONES_DT = ["mdmtoe", "mmoe"]
MOMENTUM_MODES = ["none", "per_domain_only"]


def build_tasks():
    """60 tasks total, deduped against the existing P7_backbones.csv."""
    done = done_keys()
    tasks = []
    for bb in BACKBONES_REPRO:                                # group a + b
        for mom in MOMENTUM_MODES:
            for seed in SEEDS:
                key = (bb, "three", mom, seed)
                if key in done:
                    continue
                tasks.append({"backbone": bb, "domain_version": "three",
                              "momentum_mode": mom, "seed": seed})
    for bb in BACKBONES_DT:                                   # group c
        for mom in MOMENTUM_MODES:
            for seed in SEEDS:
                key = (bb, "five", mom, seed)
                if key in done:
                    continue
                tasks.append({"backbone": bb, "domain_version": "five",
                              "momentum_mode": mom, "seed": seed})
    return tasks


def done_keys():
    """Set of (backbone, domain_version, momentum, seed) already in CSV."""
    done = set()
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                done.add((r["backbone"], r["domain_version"], r["momentum"],
                          int(r["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def task_cmd(t):
    return [PY, "-u", RUN,
            "--backbone", t["backbone"],
            "--seed", str(t["seed"]),
            "--epoch", "50",
            "--momentum_mode", t["momentum_mode"],
            "--domain_version", t["domain_version"],
            "--tag", f"p7_{t['backbone']}_{t['momentum_mode']}_{t['domain_version']}",
            "--outdir", OUTDIR]


def parse_p7_line(out):
    """Find the trailing '[p7] {...}' dict on its own line in `out`.

    p7_run.py prints `print("[p7]", row, flush=True)` so the dict is on
    the final non-empty line. ast.literal_eval is safe (no exec).
    Returns the row dict or None.
    """
    p7_re = re.compile(r"^\[p7\]\s*(\{.*\})\s*$")
    for line in out.splitlines()[::-1]:
        line = line.strip()
        if not line.startswith("[p7]"):
            continue
        m = p7_re.match(line)
        if not m:
            continue
        try:
            row = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            return None
        if isinstance(row, dict):
            return row
        return None
    return None


def finalize(t, rc, log_path, wall):
    out = open(log_path, encoding="utf-8", errors="replace").read()
    row = parse_p7_line(out) if rc == 0 else None
    if row and "test_auc_mean" in row:
        return (f"p7 {t['backbone']} {t['momentum_mode']} "
                f"{t['domain_version']} {t['seed']}: OK "
                f"auc={row['test_auc_mean']}")
    last = ""
    if out.strip():
        last = out.strip().splitlines()[-1][:150]
    return (f"p7 {t['backbone']} {t['momentum_mode']} "
            f"{t['domain_version']} {t['seed']}: FAILED "
            f"last={last!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max_runs", type=int, default=10**9)
    args = ap.parse_args()
    assert args.concurrency <= 2, "hard cap: 8GB VRAM laptop, max 2 processes"

    n_total = (len(BACKBONES_REPRO) * len(MOMENTUM_MODES) * len(SEEDS)
               + len(BACKBONES_DT) * len(MOMENTUM_MODES) * len(SEEDS))
    queue = build_tasks()[:args.max_runs]
    print(f"[p7_orchestrator] queued {len(queue)} tasks (of {n_total} total, "
          f"{len(done_keys())} already in CSV)", flush=True)
    if not queue:
        print("P7_ORCHESTRATOR_DONE", flush=True)
        return

    slots = []
    last_launch = 0.0
    launched = 0
    while queue or slots:
        while queue and len(slots) < args.concurrency:
            wait = STAGGER_SEC - (time.time() - last_launch)
            if launched > 0 and wait > 0:
                time.sleep(wait)
            t = queue.pop(0)
            cmd = task_cmd(t)
            log_path = os.path.join(
                LOGDIR,
                f"p7_{t['backbone']}_{t['momentum_mode']}_"
                f"{t['domain_version']}_{t['seed']}.log")
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=CWD)
            slots.append({"task": t, "proc": proc, "fh": fh,
                          "log": log_path, "t0": time.time()})
            last_launch = time.time()
            launched += 1
            print(f"[launch] p7_{t['backbone']}_{t['momentum_mode']}_"
                  f"{t['domain_version']}_{t['seed']} (pid {proc.pid})",
                  flush=True)
        time.sleep(5)
        for s in list(slots):
            rc = s["proc"].poll()
            if rc is None:
                continue
            s["fh"].close()
            wall = time.time() - s["t0"]
            print(f"[done] {finalize(s['task'], rc, s['log'], wall)} "
                  f"wall={wall:.0f}s", flush=True)
            slots.remove(s)
    print("P7_ORCHESTRATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
