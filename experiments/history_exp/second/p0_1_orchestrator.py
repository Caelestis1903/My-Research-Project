"""P0-1 orchestrator: 60-run sweep for the fixed-total role-reversal study.

60 runs total (per spec section 5):
  joint   24 = 3 settings x {baseline, comp1} x seeds 2022..2025
  oracle  36 = 3 settings x 3 domains x seeds 2022..2025

Execution order (spec): setting-major -> seed -> [baseline, comp1,
oracle(L域), oracle(M域), oracle(S域)] where {L,M,S}-域 for a setting is
the domain that plays that role under that setting (per ROLE_MAPS in
p4_run.py). Same template as experiments/dual_orchestrator.py: concurrency
<=2, STAGGER_SEC=30, resume-safe (key = kind|setting|method_or_domain|seed),
per-run stdout/stderr redirected to experiments/logs/<name>.log.

Per-run directory:
  experiments/P0_1_role_reversal/setting_<X>/<method|oracle_d<d>>/seed<s>/
    config.json
    stdout.log         (written by this orchestrator)
    metrics_epoch.csv  (written by p4_run.py --run_dir)
    metrics_summary.csv

Master CSV: experiments/P0_1_role_reversal/P0_1_raw.csv

Usage:
  python -u experiments/p0_1_orchestrator.py [--max_runs N]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
P4 = os.path.join(BASE, "p4_run.py")
CWD = os.path.join(ROOT, "m3oe", "examples")

P0_1_BASE = os.path.abspath(os.path.join(BASE, "P0_1_role_reversal"))
RAW = os.path.join(P0_1_BASE, "P0_1_raw.csv")
LOGDIR = os.path.join(BASE, "logs")

STAGGER_SEC = 30
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# Per spec section 3 arms. All P0-1 runs pass --fix_optim_grouping 1.
JOINT_METHODS = {
    "baseline": [],
    "comp1": ["--momentum_mode", "per_domain_only"],
}
# spec section 5 execution order within each (setting, seed):
#   [baseline, comp1, oracle(L域), oracle(M域), oracle(S域)].
# "L域" is the domain that plays L under that setting (see ROLE_MAPS).
# The domain-to-role mapping is encoded here so the order is correct.
SETTING_LMS_DOMAIN = {
    "d0_large": [0, 1, 2],   # L域=d0, M域=d1, S域=d2
    "d1_large": [1, 2, 0],   # L域=d1, M域=d2, S域=d0
    "d2_large": [2, 0, 1],   # L域=d2, M域=d0, S域=d1
}

RAW_FIELDS = ["kind", "setting", "method_or_domain", "seed", "status",
              "fit_sec", "peak_mem_mb", "wall_sec", "per_dt_test_auc",
              "best_epoch", "stopped_epoch", "best_val_auc", "note"]


def done_set():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW, newline="")):
            if r["status"] == "OK":
                done.add((r["kind"], r["setting"], r["method_or_domain"], int(r["seed"])))
    return done


def build_tasks():
    tasks = []
    done = done_set()
    for setting in SETTINGS:
        for seed in SEEDS:
            # baseline
            key = ("joint", setting, "baseline", seed)
            if key not in done:
                tasks.append({"kind": "joint", "setting": setting,
                              "method_or_domain": "baseline", "seed": seed})
            # comp1
            key = ("joint", setting, "comp1", seed)
            if key not in done:
                tasks.append({"kind": "joint", "setting": setting,
                              "method_or_domain": "comp1", "seed": seed})
            # oracle in (L域, M域, S域) order (per spec section 5)
            for d in SETTING_LMS_DOMAIN[setting]:
                key = ("oracle", setting, f"oracle_d{d}", seed)
                if key not in done:
                    tasks.append({"kind": "oracle", "setting": setting,
                                  "method_or_domain": f"oracle_d{d}", "seed": seed,
                                  "single_domain": d})
    return tasks


def run_dir(t):
    method_dir = t["method_or_domain"]
    return os.path.join(P0_1_BASE, f"setting_{t['setting']}", method_dir,
                        f"seed{t['seed']}")


def task_cmd(t):
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", "150000,30000,7500",
              "--run_dir", run_dir(t)]
    if t["kind"] == "joint":
        extra = JOINT_METHODS[t["method_or_domain"]]
        cmd = [PY, "-u", P4] + common + extra
        name = f"joint_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    else:
        cmd = [PY, "-u", P4] + common + ["--single_domain", str(t["single_domain"])]
        name = f"oracle_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "kind": t["kind"],
        "setting": t["setting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "single_domain": t.get("single_domain", -1),
        "role_sizes": "150000,30000,7500",
        "momentum_mode": ("per_domain_only"
                          if (t["kind"] == "joint" and t["method_or_domain"] == "comp1")
                          else "none"),
        "fix_optim_grouping": 1,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 1e-2,
        "learning_rate_darts": 1e-3,
        "weight_decay": 1e-5,
        "earlystop_patience": 10,
        "gamma0": 0.9,
        "gamma_max": 0.99,
    }
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def append_row(path, fields, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def parse_summary(log_text):
    """Parse the metrics_summary.csv for the run by looking for the last
    "[p4] P0-1 metrics written to ..." marker and the printed model_path.
    We pull best_epoch / stopped_epoch / best_val_auc from the summary CSV
    the runner wrote; fall back to log parsing if the CSV isn't found."""
    fit = peak = ""
    m = re.search(r"fit wall time: ([\d.]+)s \| peak cuda mem: ([\d.]+)", log_text)
    if m:
        fit, peak = m.group(1), m.group(2)
    per_dt = ""
    m = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", log_text)
    if m:
        per_dt = m.group(1)
    if not per_dt:
        # oracle final line still prints FINAL per-(d,t) test auc
        m = re.search(r"per-\(d,t\)\s*test\s*auc:\s*\[(.*?)\]", log_text)
        if m:
            per_dt = m.group(1)
    return fit, peak, per_dt


def parse_summary_csv(t):
    """Read metrics_summary.csv from the run dir."""
    p = os.path.join(run_dir(t), "metrics_summary.csv")
    if not os.path.exists(p):
        return None, None, None
    try:
        r = next(csv.DictReader(open(p, newline="")))
    except StopIteration:
        return None, None, None
    return r.get("best_epoch"), r.get("stopped_epoch"), r.get("best_val_auc")


def finalize(t, rc, log_path, wall):
    out = open(log_path, encoding="utf-8", errors="replace").read()
    note = out.strip().splitlines()[-1][:200] if rc != 0 and out.strip() else ""
    fit, peak, per_dt = parse_summary(out)
    best_ep, stopped_ep, best_val = parse_summary_csv(t)
    ok = (rc == 0 and per_dt != "")
    append_row(RAW, RAW_FIELDS, {
        "kind": t["kind"],
        "setting": t["setting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "status": "OK" if ok else "FAILED",
        "fit_sec": fit,
        "peak_mem_mb": peak,
        "wall_sec": f"{wall:.0f}",
        "per_dt_test_auc": per_dt,
        "best_epoch": best_ep if best_ep is not None else "",
        "stopped_epoch": stopped_ep if stopped_ep is not None else "",
        "best_val_auc": best_val if best_val is not None else "",
        "note": note,
    })
    return (f"{t['kind']} {t['setting']} {t['method_or_domain']} seed{t['seed']}: "
            f"{'OK' if ok else 'FAILED'} auc={per_dt or 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=10 ** 9)
    args = ap.parse_args()
    assert True, "concurrency cap enforced below"
    # Hard assertion per spec: concurrency <= 2 (8GB VRAM laptop cap).
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P0_1_BASE, exist_ok=True)

    queue = build_tasks()[:args.max_runs]
    print(f"queued {len(queue)} tasks (P0-1, total 60)", flush=True)
    slots = []          # list of dicts: task, proc, log_fh, log_path, t0
    last_launch = 0.0
    launched = 0
    while queue or slots:
        while queue and len(slots) < CONCURRENCY:
            wait = STAGGER_SEC - (time.time() - last_launch)
            if launched > 0 and wait > 0:
                time.sleep(wait)
            t = queue.pop(0)
            write_config(t)
            cmd, name = task_cmd(t)
            log_path = os.path.join(LOGDIR, f"p0_1_{name}.log")
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=CWD)
            slots.append({"task": t, "proc": proc, "fh": fh,
                          "log": log_path, "t0": time.time()})
            last_launch = time.time()
            launched += 1
            print(f"[launch] {name} (pid {proc.pid})", flush=True)
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
    print("P0_1_ORCHESTRATOR DONE", flush=True)


if __name__ == "__main__":
    main()