"""Recover P2 baseline rows from run logs (the shared-tag CSV dumps were
clobbered across seeds; logs retain the full-precision val-AUC stream and the
FINAL per-(d,t) test AUC at 6 decimals)."""
import csv
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results", "p2")
OUT = os.path.join(R, "P2_baselines.csv")

ep_re = re.compile(r"epoch:\s*(\d+)\s+validation:\s+auc:\s*([\d.]+)")
fin_re = re.compile(r"\[p2\] FINAL per-\(d,t\) test auc: \[(.*?)\]")
stop_re = re.compile(r"validation:\s*best auc:")


def parse_log(path):
    eps, aucs, stopped = [], [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ep_re.search(line)
        if m:
            eps.append(int(m.group(1)))
            aucs.append(float(m.group(2)))
        if stop_re.search(line):
            stopped = True
    fin = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = fin_re.search(line)
        if m:
            fin = [float(x.replace("np.float64(", "").replace(")", ""))
                   for x in m.group(1).split(",")]
    best_e = eps[aucs.index(max(aucs))]
    return best_e, len(eps), stopped, fin


def main():
    rows = []
    for var in ["orig", "fixed073", "fixed05", "fixedbug"]:
        for seed in [2022, 2023, 2024, 2025, 2026]:
            log = os.path.join(R, f"{var}_{seed}.log")
            if not os.path.exists(log):
                continue
            best_e, n_ep, stopped, fin = parse_log(log)
            if fin is None:
                continue
            for i, (d, t) in enumerate([(dd, tt) for dd in range(3) for tt in range(2)]):
                rows.append({"variant": var, "seed": seed, "domain": d, "task": t,
                             "test_auc": fin[i], "test_logloss": "", "best_epoch": best_e,
                             "n_epochs": n_ep, "early_stopped": stopped})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "domain", "task", "test_auc",
                                          "test_logloss", "best_epoch", "n_epochs", "early_stopped"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # summaries + tests
    from scipy import stats
    def seed_means(var):
        by = {}
        for r in rows:
            if r["variant"] == var:
                by.setdefault(r["seed"], []).append(r["test_auc"])
        return {s: np.mean(v) for s, v in by.items()}

    p0 = list(csv.DictReader(open(os.path.join(BASE, "results", "P0_repro.csv"))))
    og = {}
    for r in p0:
        og.setdefault(int(r["seed"]), []).append(float(r["test_auc"]))
    og = {s: np.mean(v) for s, v in og.items()}

    summary = {}
    for var in ["orig", "fixed073", "fixed05", "fixedbug"]:
        sm = seed_means(var)
        if not sm:
            continue
        v = np.array(list(sm.values()))
        summary[var] = (v.mean(), v.std(ddof=1), len(v))
        print(f"{var}: {v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)}) seeds={['%.4f' % x for x in sm.values()]}")

    a = np.array([og[s] for s in sorted(og)])
    b = np.array([seed_means("fixed073")[s] for s in sorted(seed_means("fixed073"))])
    common = sorted(set(og) & set(seed_means("fixed073")))
    a = np.array([og[s] for s in common])
    b = np.array([seed_means("fixed073")[s] for s in common])
    t, p_paired = stats.ttest_rel(a, b)
    t2, p_welch = stats.ttest_ind(a, b, equal_var=False)
    print(f"orig vs fixed073 (paired by seed, n={len(common)}): "
          f"diff={np.mean(a-b):+.4f}, paired t={t:.3f} p={p_paired:.4f}, welch p={p_welch:.4f}")

    fb = seed_means("fixedbug")
    if len(fb) >= 3:
        common2 = sorted(set(og) & set(fb))
        a2 = np.array([og[s] for s in common2])
        b2 = np.array([fb[s] for s in common2])
        t3, p3 = stats.ttest_rel(a2, b2)
        print(f"orig vs fixedbug (n={len(common2)}): diff={np.mean(a2-b2):+.4f}, paired t={t3:.3f} p={p3:.4f}")


if __name__ == "__main__":
    main()
