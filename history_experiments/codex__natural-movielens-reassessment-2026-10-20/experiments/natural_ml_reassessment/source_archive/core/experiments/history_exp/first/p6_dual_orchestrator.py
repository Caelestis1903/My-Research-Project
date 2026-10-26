"""Dual-process staggered orchestrator for the P6 ablation matrix.

Runs all 17 P6 variants (taken from p6_orchestrator.VARIANTS) x 5 seeds
(2022..2026) = 85 tasks, on a single fixed config (ml_r30 ->
--downsample_ratio 30.0). Up to 2 concurrent subprocesses; every new
launch is staggered by >=30 s (STAGGER_SEC) to avoid RAM/VRAM spikes
during dataset deserialization + CUDA context init.

Resume-safe via experiments/results/p6/P6_raw.csv: a task is queued only
if its (variant, seed) row is missing or has status != OK. All child
stdout/stderr is redirected 100% to an individual file under
experiments/logs/ (never echoed to the parent).

Usage:
  python -u experiments/p6_dual_orchestrator.py [--concurrency 2]
      [--max_runs N]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
RUN = os.path.join(BASE, "p4_run.py")
CWD = os.path.join(ROOT, "m3oe", "examples")
OUTDIR = os.path.join(BASE, "results", "p6")
RAW = os.path.join(OUTDIR, "P6_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

STAGGER_SEC = 30
SEEDS = [2022, 2023, 2024, 2025, 2026]

# (variant, p4_run flags) - exactly the list from p6_orchestrator.VARIANTS
FG = ["--fix_optim_grouping", "1"]
C2 = ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.5"] + FG
C1 = ["--momentum_mode", "per_domain_only"] + FG

VARIANTS = [
    # group 1: momentum tiering
    ("g1_uniform_gamma",
     ["--momentum_mode", "per_domain_only", "--gamma_uniform", "1"] + FG),
    # group 2: gamma source
    ("g2_manual_gamma",
     ["--momentum_mode", "per_domain_only", "--gamma_source", "manual",
      "--gamma_manual", "0.95,0.90,0.90"] + FG),
    ("g2_snr_gamma",
     ["--momentum_mode", "per_domain_only", "--gamma_source", "snr"] + FG),
    # group 3/4: beta granularity, curriculum off
    ("g3_beta_perdomain_nocur",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.0"] + FG),
    # group 5: reversed curriculum
    ("g5_reverse_curriculum",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "-0.5"] + FG),
    # group 6: orthogonality (comp3 arms)
    ("g6_comp3",
     ["--arch_grad_ema", "1", "--arch_update_every", "5"] + FG),
    ("g6_comp1_comp3",
     C1 + ["--arch_grad_ema", "1", "--arch_update_every", "5"]),
    ("g6_comp2_comp3",
     C2 + ["--arch_grad_ema", "1", "--arch_update_every", "5"]),
    # Adam momentum relation
    ("g8_stacked", ["--momentum_mode", "per_domain_stacked"] + FG),
    # group 7: kappa0 sensitivity
    ("g7_kappa0_0.1",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.1"] + C1),
    ("g7_kappa0_0.25",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.25"] + C1),
    ("g7_kappa0_1.0",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "1.0"] + C1),
    ("g7_kappa0_2.0",
     ["--curriculum_mode", "domain_conditioned", "--kappa0", "2.0"] + C1),
    # group 7: gamma0 sensitivity
    ("g7_gamma0_0.8",
     ["--momentum_mode", "per_domain_only", "--gamma0", "0.8"] + FG),
    ("g7_gamma0_0.85",
     ["--momentum_mode", "per_domain_only", "--gamma0", "0.85"] + FG),
    ("g7_gamma0_0.95",
     ["--momentum_mode", "per_domain_only", "--gamma0", "0.95"] + FG),
    ("g7_gamma0_0.99",
     ["--momentum_mode", "per_domain_only", "--gamma0", "0.99"] + FG),
]

# Fixed config for this orchestrator: ml_r30
CONFIG_FLAGS = ["--downsample_ratio", "30.0"]

RAW_FIELDS = ["variant", "seed", "status", "test_auc_mean", "per_dt",
              "fit_sec", "peak_mem_mb", "wall_sec", "note"]


def done_keys():
    """Set of (variant, seed) already recorded as OK in P6_raw.csv."""
    done = set()
    if os.path.exists(RAW):
        with open(RAW, newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "OK":
                    try:
                        done.add((r["variant"], int(r["seed"])))
                    except (KeyError, ValueError):
                        continue
    return done


def build_tasks():
    done = done_keys()
    tasks = []
    for name, flags in VARIANTS:
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            tasks.append({"variant": name, "seed": seed, "flags": flags})
    return tasks


def task_cmd(t):
    return ([PY, "-u", RUN, "--seed", str(t["seed"]), "--epoch", "50",
             "--tag", f"p6_{t['variant']}", "--outdir", OUTDIR]
            + CONFIG_FLAGS + t["flags"])


def append_row(path, fields, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def finalize(t, rc, log_path, wall):
    out = open(log_path, encoding="utf-8", errors="replace").read()
    note = ""
    if rc != 0 and out.strip():
        note = out.strip().splitlines()[-1][:150]
    auc = per_dt = fit = peak = ""
    m = re.search(r"fit wall time: ([\d.]+)s \| peak cuda mem: ([\d.]+)", out)
    if m:
        fit, peak = m.group(1), m.group(2)
    m = re.search(r"FINAL test auc: ([\d.]+)", out)
    if m:
        auc = m.group(1)
    m = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", out)
    if m:
        per_dt = m.group(1)
    status = "OK" if rc == 0 and auc else "FAILED"
    append_row(RAW, RAW_FIELDS, {
        "variant": t["variant"], "seed": t["seed"], "status": status,
        "test_auc_mean": auc, "per_dt": per_dt, "fit_sec": fit,
        "peak_mem_mb": peak, "wall_sec": f"{wall:.0f}", "note": note})
    return f"p6 {t['variant']} {t['seed']}: {status} auc={auc or 'NA'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max_runs", type=int, default=10**9)
    args = ap.parse_args()
    assert args.concurrency <= 2, "hard cap: 8GB VRAM laptop, max 2 processes"

    queue = build_tasks()[:args.max_runs]
    print(f"[p6_dual] queued {len(queue)} tasks (of "
          f"{len(VARIANTS) * len(SEEDS)} total, {len(done_keys())} already OK)",
          flush=True)
    if not queue:
        print("P6_DUAL_ORCHESTRATOR_DONE", flush=True)
        return

    slots = []           # list of dicts: task, proc, fh, log, t0
    last_launch = 0.0
    launched = 0
    while queue or slots:
        while queue and len(slots) < args.concurrency:
            wait = STAGGER_SEC - (time.time() - last_launch)
            if launched > 0 and wait > 0:
                time.sleep(wait)
            t = queue.pop(0)
            cmd = task_cmd(t)
            log_path = os.path.join(LOGDIR,
                                    f"p6_{t['variant']}_{t['seed']}.log")
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=CWD)
            slots.append({"task": t, "proc": proc, "fh": fh,
                          "log": log_path, "t0": time.time()})
            last_launch = time.time()
            launched += 1
            print(f"[launch] p6_{t['variant']}_{t['seed']} (pid {proc.pid})",
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
    print("P6_DUAL_ORCHESTRATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
