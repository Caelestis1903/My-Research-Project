"""Parse P5 raw orchestrator output into per-(d,t) rows + domain-gap stats.

Reads results/p5/P5_raw.csv (orchestrator) whose per_dt field holds the six
rounded test AUCs in (d0t0..d2t1) order, and produces results/p5/P5_main.csv
(long format: one row per config/method/seed/(d,t)).
Domain-gap and Pareto analysis happen in the P5 report step.
"""
import ast
import csv
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results", "p5")
RAW = os.path.join(R, "P5_raw.csv")
OUT = os.path.join(R, "P5_main.csv")

DT = [(d, t) for d in range(3) for t in range(2)]  # order of FINAL print


def main():
    if not os.path.exists(RAW):
        print("no raw csv yet")
        return
    rows = []
    for r in csv.DictReader(open(RAW)):
        if r["status"] != "OK" or not r.get("per_dt"):
            rows.append({"config": r["config"], "method": r["method"], "seed": r["seed"],
                         "domain": "", "task": "", "test_auc": "", "status": r["status"],
                         "fit_sec": r.get("fit_sec", ""), "peak_mem_mb": r.get("peak_mem_mb", "")})
            continue
        aucs = [float(x) for x in ast.literal_eval(f"[{r['per_dt']}]")]
        for (d, t), a in zip(DT, aucs):
            rows.append({"config": r["config"], "method": r["method"], "seed": r["seed"],
                         "domain": d, "task": t, "test_auc": a, "status": "OK",
                         "fit_sec": r.get("fit_sec", ""),
                         "peak_mem_mb": r.get("peak_mem_mb", "")})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "method", "seed", "domain", "task",
                                          "test_auc", "status", "fit_sec", "peak_mem_mb"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # quick domain-gap summary
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "OK" and r["domain"] != "":
            by[(r["config"], r["method"])][int(r["domain"])].append(float(r["test_auc"]))
    for cfg in sorted({c for c, _ in by}):
        print(f"== {cfg} ==")
        for method in sorted({m for _, m in by if (_, m) in by or True}):
            if (cfg, method) not in by:
                continue
            dom_means = []
            line = []
            for d in range(3):
                v = by[(cfg, method)].get(d, [])
                if v:
                    dom_means.append(sum(v) / len(v))
                    line.append(f"d{d}={dom_means[-1]:.4f}")
                else:
                    line.append(f"d{d}=NA")
            if len(dom_means) == 3:
                gap = max(dom_means) - min(dom_means)
                print(f"  {method}: {' '.join(line)}  gap={gap:.4f}")


if __name__ == "__main__":
    main()
