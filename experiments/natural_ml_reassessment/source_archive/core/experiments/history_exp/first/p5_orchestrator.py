"""P5 main-experiment orchestrator.

Runs the matrix: {data configs} x {methods} x {seeds} via p4_run.py,
appending one row per (config, method, seed) to
experiments/results/p5/P5_raw.csv (parsed later by parse_p5.py).

Data configs: ml_orig (no downsampling), ml_r1/r3/r10/r30/r100 (downsampled
train, val/test intact), kuairand (pending data availability).
Methods (switch strings for p4_run.py):
  baseline     : (all off) [+ --fix_optim_grouping if main baseline = fixedbug]
  fixed073     : --fix_arch-like via p2? -> here we emulate with fixed arch:
                 implemented as --momentum_mode none --curriculum none and
                 frozen arch via dedicated flag in p4_run (--fix_arch 0.7311)
  comp1        : --momentum_mode per_domain_only --fix_optim_grouping 1
  comp2        : --curriculum_mode domain_conditioned --kappa0 0.5 --fix_optim_grouping 1
  comp1_comp2  : both
  comp123      : comp1+comp2 + --arch_grad_ema 1 --arch_update_every 5

Resume-safe: skips (config, method, seed) rows already present.
Usage: python p5_orchestrator.py [--only CONFIG,METHOD] [--seeds 2022,2023,...]
"""
import argparse
import csv
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
RUN = os.path.join(BASE, "p4_run.py")
OUTDIR = os.path.join(BASE, "results", "p5")
LOGDIR = os.path.join(OUTDIR, "logs")
RAW = os.path.join(OUTDIR, "P5_raw.csv")
os.makedirs(LOGDIR, exist_ok=True)

# kuairand removed from the run plan at the user's instruction (2026-09-06):
# collected kuairand rows remain in P5_raw.csv but no further kuairand runs.
PRIORITY_CONFIGS = ["ml_orig", "ml_r30", "ml_r100"]
OTHER_CONFIGS = ["ml_r1", "ml_r3", "ml_r10"]
PRIORITY_SEEDS = [2022, 2023, 2024, 2025, 2026]
OTHER_SEEDS = [2022, 2023, 2024]

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
    "ml_r1": ["--downsample_ratio", "1.0"],
    "ml_r3": ["--downsample_ratio", "3.0"],
    "ml_r10": ["--downsample_ratio", "10.0"],
    "ml_r30": ["--downsample_ratio", "30.0"],
    "ml_r100": ["--downsample_ratio", "100.0"],
    "kuairand": ["--kuairand", "1", "--dataset_path", "D:/CODE/data/kuairand/processed"],
}


def existing_keys():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW)):
            if r["status"] == "OK":
                done.add((r["config"], r["method"], int(r["seed"])))
    return done


def append_row(row):
    exists = os.path.exists(RAW)
    with open(RAW, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "method", "seed", "status",
                                          "test_auc_mean", "per_dt", "final_arch",
                                          "fit_sec", "peak_mem_mb", "wall_sec", "note"])
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="CONFIG:METHOD filter")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max_runs", type=int, default=10**9)
    args = ap.parse_args()

    done = existing_keys()
    n = 0
    for config in PRIORITY_CONFIGS + OTHER_CONFIGS:
        seeds = PRIORITY_SEEDS if config in PRIORITY_CONFIGS else OTHER_SEEDS
        if args.seeds:
            seeds = [int(s) for s in args.seeds.split(",")]
        for method, flags in METHODS.items():
            if args.only and args.only != f"{config}:{method}":
                continue
            if (config, method, seeds[0]) in done and (config, method, seeds[-1]) in done:
                # crude completeness check: all seeds present?
                if all((config, method, s) in done for s in seeds):
                    continue
            for seed in seeds:
                if (config, method, seed) in done:
                    continue
                if n >= args.max_runs:
                    print("max_runs reached")
                    return
                cmd = [PY, RUN, "--seed", str(seed), "--epoch", "50",
                       "--tag", f"p5_{config}_{method}", "--outdir", OUTDIR] \
                    + CONFIGS[config] + flags
                log = os.path.join(LOGDIR, f"{config}_{method}_{seed}.log")
                t0 = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     cwd=os.path.join(ROOT, "m3oe", "examples"))
                wall = time.time() - t0
                out = res.stdout + res.stderr
                with open(log, "w", encoding="utf-8", errors="replace") as f:
                    f.write(out)
                status = "OK" if res.returncode == 0 else "FAILED"
                note = ""
                fit_sec = peak = ""
                m = __import__("re").search(r"fit wall time: ([\d.]+)s \| peak cuda mem: ([\d.]+)", out)
                if m:
                    fit_sec, peak = m.group(1), m.group(2)
                m = __import__("re").search(r"FINAL test auc: ([\d.]+)", out)
                auc_mean = m.group(1) if m else ""
                m = __import__("re").search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", out)
                per_dt = m.group(1) if m else ""
                if res.returncode != 0:
                    note = out.strip().splitlines()[-1][:200] if out.strip() else "unknown"
                append_row({"config": config, "method": method, "seed": seed,
                            "status": status, "test_auc_mean": auc_mean,
                            "per_dt": per_dt, "final_arch": "", "fit_sec": fit_sec,
                            "peak_mem_mb": peak, "wall_sec": f"{wall:.0f}", "note": note})
                print(f"{config} {method} {seed}: {status} auc={auc_mean} wall={wall:.0f}s",
                      flush=True)
                n += 1
    print("ORCHESTRATOR DONE")


if __name__ == "__main__":
    main()
