"""Parse historical KuaiRand P5 rows (baseline/fixed073/comp1) from
results/p5/P5_raw.csv and print per-(domain,task) 6-item means per method,
separating d2 (large domain, index 2) from d1/d3.

Per-(d,t) ordering in P5_raw.csv rows follows p4_run.py's FINAL line:
[d0t0, d0t1, d1t0, d1t1, d2t0, d2t1] (domains 0,1,2 = paper d1,d2,d3? see
P3: domain ids 0/1/2 with shares 7.10%/89.18%/3.72% -> local index 1 is the
LARGE domain d2 of the paper). This script prints both raw indices and the
paper-domain mapping.
"""
import csv
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "results", "p5", "P5_raw.csv")

# local domain index -> (paper domain, train share)
DOM_MAP = {0: ("d1(small)", 0.0710), 1: ("d2(LARGE)", 0.8918), 2: ("d3(small)", 0.0372)}


def parse_per_dt(s):
    vals = re.findall(r"np\.float64\(([\d.]+)\)", s or "")
    if len(vals) != 6:
        vals = re.findall(r"([\d.]+)", s or "")
    return [float(v) for v in vals[:6]] if len(vals) >= 6 else None


def main():
    rows = defaultdict(list)  # method -> list of 6-float vectors
    for r in csv.DictReader(open(RAW, newline="")):
        if r["config"] != "kuairand" or r["status"] != "OK":
            continue
        v = parse_per_dt(r["per_dt"])
        if v:
            rows[r["method"]].append((int(r["seed"]), v))
    if not rows:
        print("no kuairand OK rows found")
        sys.exit(1)
    for method in sorted(rows):
        seeds = rows[method]
        n = len(seeds)
        means = [sum(v[i] for _, v in seeds) / n for i in range(6)]
        overall = sum(means) / 6
        print(f"\n== {method} (n={n}, seeds={[s for s, _ in seeds]}) ==")
        print(f"  6-item mean AUC: {overall:.6f}")
        for d in range(3):
            name, share = DOM_MAP[d]
            m0, m1 = means[2 * d], means[2 * d + 1]
            print(f"  local d{d} = paper {name} (share {share:.2%}): "
                  f"t0={m0:.6f} t1={m1:.6f} mean={(m0 + m1) / 2:.6f}")
        # seed detail
        for s, v in seeds:
            print(f"    seed {s}: " + " ".join(f"{x:.6f}" for x in v))
    # pairwise delta vs baseline on 6-item mean
    if "baseline" in rows:
        base = dict(rows["baseline"])
        for method in sorted(rows):
            if method == "baseline":
                continue
            diffs = []
            for s, v in rows[method]:
                if s in base:
                    diffs.append(sum(v) / 6 - sum(base[s]) / 6)
            if diffs:
                m = sum(diffs) / len(diffs)
                print(f"\n{method} - baseline (paired, n={len(diffs)}): "
                      f"6-item mean delta = {m:+.6f}")


if __name__ == "__main__":
    main()
