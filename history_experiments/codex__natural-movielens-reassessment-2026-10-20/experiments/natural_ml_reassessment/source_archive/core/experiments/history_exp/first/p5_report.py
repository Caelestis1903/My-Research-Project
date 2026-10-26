"""P5 report generator: parse results/p5/P5_raw.csv (+ oracle rows) into the
main comparison tables, domain-gap stats, bootstrap CIs, Pareto verdict and
efficiency table. Writes results/p5/P5_main.csv + figures.
"""
import ast
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results", "p5")
FIG = os.path.join(R, "figs")
os.makedirs(FIG, exist_ok=True)

DT = [(d, t) for d in range(3) for t in range(2)]
METHOD_LABEL = {
    "baseline": "M3oE (as-published)",
    "fixed073": "fixed-0.7311",
    "comp1": "+comp1 (per-domain momentum)",
    "comp2": "+comp2 (domain curriculum)",
    "comp1_comp2": "+comp1&2",
    "comp123": "+comp1&2&3",
}


def load():
    rows = []
    with open(os.path.join(R, "P5_raw.csv")) as f:
        for r in csv.DictReader(f):
            if r["status"] != "OK" or not r.get("per_dt"):
                continue
            aucs = [float(x.replace("np.float64(", "").replace(")", ""))
                    for x in r["per_dt"].split(",")]
            rows.append({"config": r["config"], "method": r["method"], "seed": int(r["seed"]),
                         "aucs": aucs, "mean": float(np.mean(aucs)),
                         "fit_sec": float(r["fit_sec"] or 0),
                         "peak_mb": float(r["peak_mem_mb"] or 0),
                         "wall": float(r["wall_sec"] or 0)})
    return rows


def per_dt_table(rows, config, methods):
    print(f"\n=== {config}: per-(d,t) test AUC (mean±std over seeds) ===")
    out = {}
    for m in methods:
        rs = [r for r in rows if r["config"] == config and r["method"] == m]
        if not rs:
            continue
        cells = []
        for i in range(6):
            v = [r["aucs"][i] for r in rs]
            cells.append(f"{np.mean(v):.4f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}")
        means = [r["mean"] for r in rs]
        out[m] = (rs, means)
        print(f"{METHOD_LABEL.get(m, m):32s} n={len(rs)} avg={np.mean(means):.4f}±{np.std(means, ddof=1) if len(means)>1 else 0:.4f} | {' '.join(cells)}")
    return out


def domain_gap(rows, config, methods):
    print(f"\n=== {config}: domain-gap ===")
    gaps = {}
    for m in methods:
        rs = [r for r in rows if r["config"] == config and r["method"] == m]
        if not rs:
            continue
        per_seed_gap, per_seed_std = [], []
        for r in rs:
            dom = [np.mean([r["aucs"][i] for i in range(6) if DT[i][0] == d]) for d in range(3)]
            per_seed_gap.append(max(dom) - min(dom))
            per_seed_std.append(np.std(dom, ddof=1))
        gaps[m] = (np.mean(per_seed_gap), np.std(per_seed_gap, ddof=1) if len(per_seed_gap) > 1 else 0,
                   np.mean(per_seed_std))
        print(f"{METHOD_LABEL.get(m, m):32s} max-min gap {gaps[m][0]:.4f}±{gaps[m][1]:.4f}, "
              f"per-domain std {gaps[m][2]:.4f}")
    return gaps


def paired_tests(rows, config, methods, ref="baseline"):
    from scipy import stats
    print(f"\n=== {config}: paired t-tests vs {ref} (6-item avg, by seed) ===")
    base = {r["seed"]: r["mean"] for r in rows if r["config"] == config and r["method"] == ref}
    for m in methods:
        if m == ref:
            continue
        rs = {r["seed"]: r["mean"] for r in rows if r["config"] == config and r["method"] == m}
        common = sorted(set(base) & set(rs))
        if len(common) < 3:
            continue
        a = np.array([base[s] for s in common])
        b = np.array([rs[s] for s in common])
        t, p = stats.ttest_rel(a, b)
        d = (b - a).mean() / (np.std(a - b, ddof=1) + 1e-12)  # Cohen's d (paired)
        print(f"{METHOD_LABEL.get(m, m):32s} diff={np.mean(b-a):+.4f} t={t:+.2f} p={p:.4f} d={d:+.2f}")


def bootstrap_ci(rows, config, m, domain, iters=1000, seed=1234):
    """Bootstrap 95% CI of the per-domain AUC across seeds (resampling seeds)."""
    rs = [r["aucs"] for r in rows if r["config"] == config and r["method"] == m]
    if not rs:
        return None
    idxs = [i for i in range(6) if DT[i][0] == domain]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        pick = rs[rng.integers(0, len(rs), len(rs))]
        vals.append(np.mean([np.mean([p[i] for i in idxs]) for p in pick]))
    return np.percentile(vals, [2.5, 97.5]), np.mean(vals)


def efficiency(rows, config, methods):
    print(f"\n=== {config}: efficiency ===")
    for m in methods:
        rs = [r for r in rows if r["config"] == config and r["method"] == m]
        if not rs:
            continue
        print(f"{METHOD_LABEL.get(m, m):32s} fit {np.mean([r['fit_sec'] for r in rs]):.0f}s "
              f"peak {np.mean([r['peak_mb'] for r in rs]):.0f} MiB wall {np.mean([r['wall'] for r in rs]):.0f}s")


def main():
    rows = load()
    if not rows:
        print("no data yet")
        return
    # long format
    with open(os.path.join(R, "P5_main.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "method", "seed", "domain", "task", "test_auc", "mean_auc",
                    "fit_sec", "peak_mem_mb"])
        for r in rows:
            for i, (d, t) in enumerate(DT):
                w.writerow([r["config"], r["method"], r["seed"], d, t, r["aucs"][i],
                            r["mean"], r["fit_sec"], r["peak_mb"]])
    print(f"wrote P5_main.csv ({len(rows)} runs)")

    configs = sorted({r["config"] for r in rows})
    methods = ["baseline", "fixed073", "comp1", "comp2", "comp1_comp2", "comp123"]
    for cfg in configs:
        ms = [m for m in methods if any(r["config"] == cfg and r["method"] == m for r in rows)]
        per_dt_table(rows, cfg, ms)
        domain_gap(rows, cfg, ms)
        paired_tests(rows, cfg, ms)
        efficiency(rows, cfg, ms)

    # trend figure: method gain vs imbalance tier (per-domain AUC for d0)
    tiers = [("ml_r1", "1:1"), ("ml_orig", "1.9:1"), ("ml_r3", "3:1"), ("ml_r10", "10:1"),
             ("ml_r30", "30:1"), ("ml_r100", "100:1")]
    have = [t for t in tiers if any(r["config"] == t[0] for r in rows)]
    if have:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m, mk in [("comp1", "o"), ("comp2", "s"), ("comp1_comp2", "^"), ("comp123", "d")]:
            xs, ys, es = [], [], []
            for cfg, label in have:
                base = [r["mean"] for r in rows if r["config"] == cfg and r["method"] == "baseline"]
                comp = [r["mean"] for r in rows if r["config"] == cfg and r["method"] == m]
                if base and comp:
                    xs.append(label)
                    ys.append(np.mean(comp) - np.mean(base))
                    es.append(np.std(comp) / np.sqrt(len(comp)) + np.std(base) / np.sqrt(len(base)))
            if xs:
                ax.errorbar(range(len(xs)), ys, yerr=es, marker=mk, capsize=3,
                            label=METHOD_LABEL[m])
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Δ avg test AUC vs baseline")
        ax.set_xlabel("imbalance tier (max:min)")
        ax.set_title("P5: method gain vs imbalance")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "p5_trend.png"), dpi=150)
        print("wrote trend figure")


if __name__ == "__main__":
    main()
