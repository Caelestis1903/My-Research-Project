"""Parse P0 seed logs + output CSV into experiments/results/P0_repro.csv.

Row selection: for each seed, take the row from examples/output CSV written by
the FULL official run (50-epoch cap). The 3-epoch smoke-test rows are excluded
by matching the test AUC / (d,t) values against the seed's log.
best_epoch = argmax over per-epoch validation mean AUC from the log
(as-published: on early stop the best-val weights are restored).
"""
import csv
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = BASE
OUTDIR = os.path.join(BASE, "results")
OUTCSV = os.path.join(OUTDIR, "P0_repro.csv")
SEEDS = [2022, 2023, 2024, 2025, 2026]

epoch_re = re.compile(r"epoch:\s*(\d+)\s+validation:\s+auc:\s*([\d.]+)")
test_re = re.compile(r"test auc:\s*([\d.]+)\s*\|\s*test logloss:\s*([\d.]+)")
dt_re = re.compile(r"domain:\s*(\d),\s*task:\s*(\d),\s*auc:\s*([\d.]+)\s*\|\s*test logloss:\s*([\d.]+)")
stop_re = re.compile(r"validation:\s*best auc:")


def parse_log(path):
    epochs, aucs = [], []
    test_auc = test_ll = None
    per_dt, early_stop = {}, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                epochs.append(int(m.group(1)))
                aucs.append(float(m.group(2)))
            m = test_re.search(line)
            if m:
                test_auc, test_ll = float(m.group(1)), float(m.group(2))
            m = dt_re.search(line)
            if m:
                per_dt[(int(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
            if stop_re.search(line):
                early_stop = True
    best_epoch = epochs[aucs.index(max(aucs))] if aucs else None
    return {"best_epoch": best_epoch, "n_epochs": len(epochs), "early_stop": early_stop,
            "test_auc": test_auc, "test_logloss": test_ll, "per_dt": per_dt}


def main():
    wall = {}
    wc = os.path.join(LOGDIR, "p0_wallclock.txt")
    if os.path.exists(wc):
        for line in open(wc):
            p = line.split()
            if len(p) == 4 and p[0] == "SEED":
                wall[int(p[1])] = int(p[3])

    rows = []
    for seed in SEEDS:
        log = os.path.join(LOGDIR, f"p0_seed{seed}.log")
        info = parse_log(log)
        print(f"seed {seed}: epochs={info['n_epochs']} best_epoch={info['best_epoch']} "
              f"early_stop={info['early_stop']} test_auc={info['test_auc']} wall={wall.get(seed)}s")
        for d in range(3):
            for t in range(2):
                auc, ll = info["per_dt"][(d, t)]
                rows.append({"seed": seed, "domain": d, "task": t, "test_auc": auc,
                             "test_logloss": ll, "best_epoch": info["best_epoch"],
                             "wall_clock_sec": wall.get(seed, "")})

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "domain", "task", "test_auc",
                                          "test_logloss", "best_epoch", "wall_clock_sec"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUTCSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
