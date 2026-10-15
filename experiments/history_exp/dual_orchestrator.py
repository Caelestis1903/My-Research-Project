"""Dual-process staggered orchestrator for the remaining P5 experiments (re.md).

Two concurrent training subprocesses at most (--concurrency 2), each new
process launch staggered by >=30 s (STAGGER_SEC) to avoid RAM/VRAM spikes
during dataset deserialization + CUDA context init.

Resume-safe: a task is queued only if its target CSV has no row with
status == OK for its key. All stdout/stderr is redirected 100% to an
individual file under experiments/logs/ (never echoed to the parent).

Task phases (in order):
  1. oracle  kuairand        (3 domains x 5 seeds, run_kuaishou.py --domain)
  2. main    kuairand        (comp1 seeds 2025/2026; comp2/comp1_comp2/comp123 x 5)
  3. main    ml_r3, ml_r10   (6 methods x seeds 2022-2024)
  4. oracle  ml_orig/r30/r100 (3 domains x 5 seeds, p4_run.py --single_domain)

Usage:
  python -u experiments/dual_orchestrator.py [--dataset kuairand|ml|all]
      [--concurrency 2] [--max_runs N]
"""
import argparse
import csv
import os
import re
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
P4 = os.path.join(BASE, "p4_run.py")
KUAI_RUN = os.path.join(ROOT, "m3oe", "examples", "run_kuaishou.py")
CWD = os.path.join(ROOT, "m3oe", "examples")
OUTDIR = os.path.join(BASE, "results", "p5")
RAW = os.path.join(OUTDIR, "P5_raw.csv")
ORACLE = os.path.join(OUTDIR, "P5_oracle.csv")
LOGDIR = os.path.join(BASE, "logs")
os.makedirs(LOGDIR, exist_ok=True)

STAGGER_SEC = 30
SEEDS5 = [2022, 2023, 2024, 2025, 2026]
SEEDS3 = [2022, 2023, 2024]

METHODS = {
    "baseline": [],
    "fixed073": ["--fix_arch", "0.7311"],
    "comp1": ["--momentum_mode", "per_domain_only", "--fix_optim_grouping", "1"],
    "comp2": ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.5",
              "--fix_optim_grouping", "1"],
    "comp1_comp2": ["--momentum_mode", "per_domain_only", "--curriculum_mode",
                    "domain_conditioned", "--kappa0", "0.5", "--fix_optim_grouping", "1"],
    "comp123": ["--momentum_mode", "per_domain_only", "--curriculum_mode",
                "domain_conditioned", "--kappa0", "0.5", "--arch_grad_ema", "1",
                "--arch_update_every", "5", "--fix_optim_grouping", "1"],
}

CONFIGS = {
    "ml_orig": ["--downsample_ratio", "0"],
    "ml_r3": ["--downsample_ratio", "3.0"],
    "ml_r10": ["--downsample_ratio", "10.0"],
    "ml_r30": ["--downsample_ratio", "30.0"],
    "ml_r100": ["--downsample_ratio", "100.0"],
    "kuairand": ["--kuairand", "1", "--dataset_path", "D:/CODE/data/kuairand/processed"],
}

RAW_FIELDS = ["config", "method", "seed", "status", "test_auc_mean", "per_dt",
              "final_arch", "fit_sec", "peak_mem_mb", "wall_sec", "note"]
ORACLE_FIELDS = ["config", "domain", "seed", "status", "auc_d0t0", "auc_d0t1",
                 "wall_sec", "note"]


def done_main():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW, newline="")):
            if r["status"] == "OK":
                done.add((r["config"], r["method"], int(r["seed"])))
    return done


def done_oracle():
    done = set()
    if os.path.exists(ORACLE):
        for r in csv.DictReader(open(ORACLE, newline="")):
            if r["status"] == "OK":
                done.add((r["config"], int(r["domain"]), int(r["seed"])))
    return done


def build_tasks(dataset):
    tasks = []
    dm, do = done_main(), done_oracle()

    def add_main(config, method, seed):
        if (config, method, seed) not in dm:
            tasks.append({"kind": "main", "config": config, "method": method,
                          "seed": seed})

    def add_oracle(config, domain, seed):
        if (config, domain, seed) not in do:
            tasks.append({"kind": "oracle", "config": config, "domain": domain,
                          "seed": seed})

    if dataset in ("kuairand", "all"):
        for domain in range(3):                      # phase 1
            for seed in SEEDS5:
                add_oracle("kuairand", domain, seed)
        for method in ["comp1", "comp2", "comp1_comp2", "comp123"]:  # phase 2
            for seed in SEEDS5:
                add_main("kuairand", method, seed)
    if dataset in ("ml", "all"):
        for config in ["ml_r3", "ml_r10"]:           # phase 3
            for method in METHODS:
                for seed in SEEDS3:
                    add_main(config, method, seed)
        for config in ["ml_orig", "ml_r30", "ml_r100"]:
            for domain in range(3):
                for seed in SEEDS5:
                    add_oracle(config, domain, seed)
    return tasks


def task_cmd(t):
    if t["kind"] == "main":
        cmd = [PY, "-u", P4, "--seed", str(t["seed"]), "--epoch", "50",
               "--tag", f"p5_{t['config']}_{t['method']}", "--outdir", OUTDIR] \
            + CONFIGS[t["config"]] + METHODS[t["method"]]
        name = f"{t['config']}_{t['method']}_{t['seed']}"
    elif t["config"] == "kuairand":
        cmd = [PY, "-u", KUAI_RUN, "--dataset_path", "D:/CODE/data/kuairand/processed",
               "--seed", str(t["seed"]), "--epoch", "50", "--domain", str(t["domain"])]
        name = f"oracle_kuairand_d{t['domain']}_{t['seed']}"
    else:
        cmd = [PY, "-u", P4, "--seed", str(t["seed"]), "--epoch", "50",
               "--single_domain", str(t["domain"]),
               "--tag", f"p5_oracle_{t['config']}_d{t['domain']}", "--outdir", OUTDIR] \
            + CONFIGS[t["config"]]
        name = f"oracle_{t['config']}_d{t['domain']}_{t['seed']}"
    return cmd, name


def append_row(path, fields, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def finalize(t, rc, log_path, wall):
    out = open(log_path, encoding="utf-8", errors="replace").read()
    note = out.strip().splitlines()[-1][:200] if rc != 0 and out.strip() else ""
    if t["kind"] == "main":
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
            "config": t["config"], "method": t["method"], "seed": t["seed"],
            "status": status, "test_auc_mean": auc, "per_dt": per_dt,
            "final_arch": "", "fit_sec": fit, "peak_mem_mb": peak,
            "wall_sec": f"{wall:.0f}", "note": note})
        return f"{t['config']} {t['method']} {t['seed']}: {status} auc={auc or 'NA'}"
    dt = re.findall(r"domain:\s*0,\s*task:\s*(\d),\s*auc:\s*([\d.]+)", out)
    vals = ["", ""]
    for tt, aa in dt:
        vals[int(tt)] = aa
    if t["config"] != "kuairand" and not any(vals):
        m = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", out)
        if m:
            parts = [x.replace("np.float64(", "").replace(")", "")
                     for x in m.group(1).split(",")]
            vals = parts[:2] + [""] * (2 - len(parts))
    status = "OK" if rc == 0 and all(vals) else "FAILED"
    append_row(ORACLE, ORACLE_FIELDS, {
        "config": t["config"], "domain": t["domain"], "seed": t["seed"],
        "status": status, "auc_d0t0": vals[0], "auc_d0t1": vals[1],
        "wall_sec": f"{wall:.0f}", "note": note})
    return (f"oracle {t['config']} d{t['domain']} {t['seed']}: {status} "
            f"auc={vals if any(vals) else 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["kuairand", "ml", "all"])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max_runs", type=int, default=10**9)
    args = ap.parse_args()
    assert args.concurrency <= 2, "hard cap: 8GB VRAM laptop, max 2 processes"

    queue = build_tasks(args.dataset)[:args.max_runs]
    print(f"queued {len(queue)} tasks", flush=True)
    slots = []          # list of dicts: task, proc, log_fh, log_path, t0
    last_launch = 0.0
    launched = 0
    while queue or slots:
        while queue and len(slots) < args.concurrency:
            wait = STAGGER_SEC - (time.time() - last_launch)
            if launched > 0 and wait > 0:
                time.sleep(wait)
            t = queue.pop(0)
            cmd, name = task_cmd(t)
            log_path = os.path.join(LOGDIR, f"{name}.log")
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=CWD)
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
    print("DUAL_ORCHESTRATOR DONE", flush=True)


if __name__ == "__main__":
    main()
