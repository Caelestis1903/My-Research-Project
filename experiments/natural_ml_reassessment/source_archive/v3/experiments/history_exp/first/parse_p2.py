"""Parse P2 run CSVs/logs into experiments/results/P2_baselines.csv
(+ P2_arch_traj_fixed.csv) and compute the orig vs fixed-0.7311 t-test."""
import csv
import glob
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results", "p2")
OUT = os.path.join(R, "P2_baselines.csv")
SEEDS = [2022, 2023, 2024, 2025, 2026]
VARIANTS = {
    "orig": "p2x_orig_fixgroup0",        # cross-check run (seed 2022 only)
    "fixed073": "p2_fixed073_fixgroup0",
    "fixed05": "p2_fixed05_fixgroup0",
    "fixedbug": "p2_orig_fixgroup1",
}


def rows_for(tag):
    curves = list(csv.DictReader(open(os.path.join(R, f"D06_curves_{tag}.csv"))))
    tests = list(csv.DictReader(open(os.path.join(R, f"D06_test_{tag}.csv"))))
    return curves, tests


def main():
    rows = []
    for var, tag in VARIANTS.items():
        for seed in SEEDS:
            try:
                curves, tests = rows_for(tag)
            except FileNotFoundError:
                continue
            sc = [r for r in curves if int(r["seed"]) == seed]
            if not sc:
                continue
            st = {(int(r["epoch"]), int(r["domain"]), int(r["task"])): r
                  for r in tests if int(r["seed"]) == seed}
            eps = sorted({int(r["epoch"]) for r in sc})
            mean_val = {e: np.mean([float(r["val_auc"]) for r in sc if int(r["epoch"]) == e])
                        for e in eps}
            best_e = max(eps, key=lambda e: mean_val[e])
            for d in range(3):
                for t in range(2):
                    te = st[(best_e, d, t)]
                    rows.append({"variant": var, "seed": seed, "domain": d, "task": t,
                                 "test_auc": float(te["test_auc"]),
                                 "test_logloss": te.get("test_logloss", ""),
                                 "best_epoch": best_e})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "domain", "task",
                                          "test_auc", "test_logloss", "best_epoch"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # summary + t-test orig(=P0 5 seeds) vs fixed073
    p0 = list(csv.DictReader(open(os.path.join(BASE, "results", "P0_repro.csv"))))
    def seed_means(prefix, src):
        by = {}
        for r in src:
            if r["variant"] == prefix if "variant" in r else False:
                by.setdefault(int(r["seed"]), []).append(float(r["test_auc"]))
        return {s: np.mean(v) for s, v in by.items()}
    fx = {}
    for r in rows:
        if r["variant"] == "fixed073":
            fx.setdefault(r["seed"], []).append(r["test_auc"])
    fx = {s: np.mean(v) for s, v in fx.items()}
    og = {}
    for r in p0:
        og.setdefault(int(r["seed"]), []).append(float(r["test_auc"]))
    og = {s: np.mean(v) for s, v in og.items()}
    common = sorted(set(fx) & set(og))
    a = np.array([og[s] for s in common])
    b = np.array([fx[s] for s in common])
    try:
        from scipy import stats
        t, p_paired = stats.ttest_rel(a, b)
        t2, p_ind = stats.ttest_ind(a, b, equal_var=False)
        print(f"orig {a.mean():.4f}±{a.std(ddof=1):.4f} vs fixed073 {b.mean():.4f}±{b.std(ddof=1):.4f} "
              f"(n={len(common)}) paired p={p_paired:.4f} welch p={p_ind:.4f}")
    except ImportError:
        diff = a - b
        print(f"orig {a.mean():.4f}±{a.std(ddof=1):.4f} vs fixed073 {b.mean():.4f}±{b.std(ddof=1):.4f} "
              f"diff {diff.mean():+.4f}±{diff.std(ddof=1):.4f} (scipy missing, no p-value)")

    # merge fixedbug arch trajectories
    trajs = []
    for f in glob.glob(os.path.join(R, "D02_arch_traj_p2_orig_fixgroup1.csv")):
        trajs += list(csv.DictReader(open(f)))
    if trajs:
        with open(os.path.join(R, "P2_arch_traj_fixed.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=["seed", "epoch", "param_name",
                                               "raw_value", "sigmoid_value"])
            w.writeheader()
            w.writerows(trajs)
        print(f"wrote P2_arch_traj_fixed.csv ({len(trajs)} rows)")


if __name__ == "__main__":
    main()
