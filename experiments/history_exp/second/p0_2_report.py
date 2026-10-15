"""P0-2 report: pure read-CSV aggregator over the 140 fresh P0-2 runs plus
the 60 P0-1 20:1 rows (referenced, NOT re-run).

Inputs:
  experiments/P0_2_ratio_sweep/P0_2_raw.csv           (P0-2 fresh runs)
  experiments/P0_1_role_reversal/P0_1_raw_dedup.csv   (P0-1 20:1 reuse)

Outputs (all under experiments/P0_2_ratio_sweep/):
  P0_2_all.csv         200 rows (140 + 60), with `ratio` and `source` columns
  P0_2_cells.csv       long: per (ratio, setting, domain, task, seed) cell
                       pairs baseline / comp1 / oracle
  P0_2_summary_ratio.csv  per (ratio, method): mean AUC, worst-domain AUC,
                          comp1 gain (overall + per role)

per_dt_test_auc is a `np.float64(...)` repr string. The 6 elements are in
task-major order [d0t0, d1t0, d2t0, d0t1, d1t1, d2t1] (index = t*3+d), per
spec; oracle rows have 2 elements [t0, t1] for the single oracle domain.

The report is fully self-contained: it parses CSVs, never touches models,
and never runs training. The synthetic-self-test mode (--self_test)
constructs a small in-memory dataframe so the parser can be unit-checked
without real runs.

Usage:
  python -u experiments/p0_2_report.py
  python -u experiments/p0_2_report.py --self_test
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")
P0_1_BASE = os.path.join(BASE, "P0_1_role_reversal")

P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")
P0_1_DEDUP = os.path.join(P0_1_BASE, "P0_1_raw_dedup.csv")

OUT_ALL = os.path.join(P0_2_BASE, "P0_2_all.csv")
OUT_CELLS = os.path.join(P0_2_BASE, "P0_2_cells.csv")
OUT_SUMMARY = os.path.join(P0_2_BASE, "P0_2_summary_ratio.csv")

# All fresh ratios in P0-2; 20:1 is reused from P0-1.
P0_2_RATIOS = (1, 5, 50)
P0_1_RATIO = 20
P0_1_ROLE_SIZES = "150000,30000,7500"
SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
DOMAINS = (0, 1, 2)
TASKS = (0, 1)


# -- parsing ------------------------------------------------------------

FLOAT_RE = re.compile(r"np\.float64\(([-\d.eE+]+)\)")

# ROLE_MAPS: d0 plays L in d0_large, S in d1_large, M in d2_large.
ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}


def parse_per_dt(s):
    """Parse a 'np.float64(x), np.float64(y), ...' string into a list of
    floats. Returns [] for an empty / missing string."""
    if not s:
        return []
    return [float(m.group(1)) for m in FLOAT_RE.finditer(s)]


# per_dt layout: task-major [d0t0, d1t0, d2t0, d0t1, d1t1, d2t1]
# so for a joint run, idx -> (d, t) is (idx % 3, idx // 3).
def cell_from_joint_idx(i):
    d = i % 3
    t = i // 3
    return d, t


# -- per-cell pair model -------------------------------------------------

def expand_per_dt(row):
    """Return list of (domain, task, value) for a row's per_dt_test_auc.
    For joint rows, the list has 6 entries (d,t) per spec. For oracle rows
    the list has 2 entries (t0, t1) for the single oracle domain.
    """
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if row["kind"] == "joint":
        if len(vals) != 6:
            return []
        return [(d, t, vals[t * 3 + d]) for t in TASKS for d in DOMAINS]
    elif row["kind"] == "oracle":
        if len(vals) != 2:
            return []
        d = int(row["method_or_domain"].replace("oracle_d", ""))
        return [(d, 0, vals[0]), (d, 1, vals[1])]
    return []


# -- inputs --------------------------------------------------------------

def load_p0_2_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        r["ratio"] = int(r["ratio"])
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def load_p0_1_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        r["ratio"] = P0_1_RATIO
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def validate_p0_1_role_sizes(rows):
    """Spot-check that P0-1's role_sizes matches the 20:1 contract. The dedup
    CSV doesn't carry role_sizes, so we sample a few run dirs and read
    config.json. This is best-effort: if the per-run config.json is missing
    (e.g. results dir trimmed) we fall back to a warning instead of
    raising, since the report contract only requires role_sizes for the
    140 fresh runs (P0-2's per-row config.json).
    """
    sampled = 0
    for r in rows:
        d = os.path.join(P0_1_BASE, f"setting_{r['setting']}",
                         r["method_or_domain"], f"seed{r['seed']}")
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        sampled += 1
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rs = cfg.get("role_sizes", "")
        if rs != P0_1_ROLE_SIZES:
            raise AssertionError(
                f"P0-1 row {r['setting']}/{r['method_or_domain']}/seed"
                f"{r['seed']} has role_sizes={rs!r}, expected {P0_1_ROLE_SIZES!r}"
            )
        if sampled >= 3:
            break
    if sampled == 0:
        print(f"[warn] no P0-1 config.json sampled; skipping role_sizes check",
              flush=True)


# -- outputs -------------------------------------------------------------

def write_all_csv(p0_2_rows, p0_1_rows, path):
    fields = ["source", "kind", "ratio", "setting", "method_or_domain", "seed",
              "fit_sec", "peak_mem_mb", "wall_sec", "per_dt_test_auc",
              "best_epoch", "stopped_epoch", "best_val_auc", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in p0_2_rows:
            row = {k: r.get(k, "") for k in fields}
            row["source"] = "P0-2"
            w.writerow(row)
        for r in p0_1_rows:
            row = {k: r.get(k, "") for k in fields}
            row["source"] = "P0-1_reuse"
            w.writerow(row)


def build_cells(p0_2_rows, p0_1_rows):
    """Yield dict rows for the long cells CSV. Schema: ratio, setting, domain,
    task, seed, role (L/M/S under that setting), baseline_auc, comp1_auc,
    oracle_auc, delta_transfer, comp1_gain.
    """
    # group all rows (P0-2 + P0-1) by (ratio, setting, kind, method_or_domain, seed)
    pool = []
    for r in p0_2_rows:
        row = dict(r); row["source"] = "P0-2"; pool.append(row)
    for r in p0_1_rows:
        row = dict(r); row["source"] = "P0-1_reuse"; pool.append(row)

    # index by (ratio, setting, kind, method_or_domain, seed)
    idx = {}
    for r in pool:
        key = (r["ratio"], r["setting"], r["kind"],
               r["method_or_domain"], r["seed"])
        idx[key] = r

    out_rows = []
    seen_ratios = set()
    for r in pool:
        seen_ratios.add(r["ratio"])
    for ratio in sorted(seen_ratios):
        # Joint rows are 3 settings (or 1 for ratio_1) per (seed, method).
        for setting in SETTINGS:
            for seed in SEEDS:
                # does this (ratio, setting) have joint data? for ratio=1 only
                # d0_large is actually run, others should be absent.
                key_b = (ratio, setting, "joint", "baseline", seed)
                key_c = (ratio, setting, "joint", "comp1", seed)
                if key_b not in idx or key_c not in idx:
                    continue
                b_cells = expand_per_dt(idx[key_b])
                c_cells = expand_per_dt(idx[key_c])
                if not b_cells or not c_cells:
                    continue
                b_map = {(d, t): v for (d, t, v) in b_cells}
                c_map = {(d, t): v for (d, t, v) in c_cells}
                for (d, t) in sorted(b_map):
                    key_o = (ratio, setting, "oracle", f"oracle_d{d}", seed)
                    if key_o not in idx:
                        o_val = ""
                    else:
                        o_cells = expand_per_dt(idx[key_o])
                        o_map = {(dd, tt): v for (dd, tt, v) in o_cells}
                        o_val = o_map.get((d, t), "")
                    baseline = b_map[(d, t)]
                    comp1 = c_map[(d, t)]
                    try:
                        oracle = float(o_val) if o_val != "" else float("nan")
                        delta = baseline - oracle
                        comp1_gain = comp1 - baseline
                    except (TypeError, ValueError):
                        delta = ""
                        comp1_gain = ""
                    role = ROLE_MAPS[setting][d]
                    out_rows.append({
                        "ratio": ratio,
                        "setting": setting,
                        "domain": d,
                        "task": t,
                        "seed": seed,
                        "role": role,
                        "baseline_auc": f"{baseline:.6f}",
                        "comp1_auc": f"{comp1:.6f}",
                        "oracle_auc": (f"{oracle:.6f}"
                                       if isinstance(oracle, float)
                                       and oracle == oracle else ""),
                        "delta_transfer": (f"{delta:.6f}"
                                           if isinstance(delta, float) else ""),
                        "comp1_gain": (f"{comp1_gain:.6f}"
                                       if isinstance(comp1_gain, float) else ""),
                    })
    return out_rows


def write_cells_csv(rows, path):
    fields = ["ratio", "setting", "domain", "task", "seed", "role",
              "baseline_auc", "comp1_auc", "oracle_auc",
              "delta_transfer", "comp1_gain"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_summary(cells):
    """Per (ratio, method): mean AUC, worst-domain AUC, comp1 gain (overall
    and per role). ratio_1 -> role column 'degenerate' (all three role
    columns are equal in expectation; we mark them so)."""
    # cells are dicts with keys ratio, setting, domain, seed, role,
    # baseline_auc, comp1_auc, oracle_auc, delta_transfer, comp1_gain.
    # mean AUC per (ratio, method) = mean over all cell×seed values.
    # worst-domain AUC = per cell, mean over setting & seed, then take
    # the minimum across the 6 cells.
    # comp1 gain per role = mean over (cell, seed) for cells whose role
    # in that setting matches.

    def parsef(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    summary_rows = []
    by_ratio = defaultdict(list)
    for c in cells:
        by_ratio[int(c["ratio"])].append(c)

    for ratio in sorted(by_ratio):
        rcells = by_ratio[ratio]
        for method in ("baseline", "comp1", "oracle"):
            # Collect (cell_value, domain, role) per (setting, seed)
            cell_avgs = defaultdict(list)  # (d, t) -> [v over seed,setting]
            for c in rcells:
                v = parsef(c[f"{method}_auc"])
                if v is None:
                    continue
                cell_avgs[(int(c["domain"]), int(c["task"]))].append(v)
            cell_means = {k: mean(vs) for k, vs in cell_avgs.items() if vs}
            mean_auc = mean(list(cell_means.values())) if cell_means else float("nan")
            worst = min(cell_means.values()) if cell_means else float("nan")
            # comp1 gain
            if method == "comp1":
                gain_all = []
                gain_by_role = defaultdict(list)
                for c in rcells:
                    g = parsef(c["comp1_gain"])
                    if g is None:
                        continue
                    gain_all.append(g)
                    gain_by_role[c["role"]].append(g)
                if ratio == 1:
                    role_means = {"degenerate": mean(gain_all) if gain_all
                                  else float("nan")}
                else:
                    role_means = {r: mean(vs) for r, vs in gain_by_role.items()}
                summary_rows.append({
                    "ratio": ratio,
                    "method": method,
                    "mean_auc": f"{mean_auc:.6f}",
                    "worst_domain_auc": f"{worst:.6f}",
                    "comp1_gain_overall": f"{mean(gain_all):.6f}",
                    "comp1_gain_role_L": (f"{role_means.get('L', float('nan')):.6f}"
                                           if 'L' in role_means else ""),
                    "comp1_gain_role_M": (f"{role_means.get('M', float('nan')):.6f}"
                                           if 'M' in role_means else ""),
                    "comp1_gain_role_S": (f"{role_means.get('S', float('nan')):.6f}"
                                           if 'S' in role_means else ""),
                })
            else:
                summary_rows.append({
                    "ratio": ratio,
                    "method": method,
                    "mean_auc": f"{mean_auc:.6f}",
                    "worst_domain_auc": f"{worst:.6f}",
                    "comp1_gain_overall": "",
                    "comp1_gain_role_L": "",
                    "comp1_gain_role_M": "",
                    "comp1_gain_role_S": "",
                })
    return summary_rows


def write_summary_csv(rows, path):
    fields = ["ratio", "method", "mean_auc", "worst_domain_auc",
              "comp1_gain_overall",
              "comp1_gain_role_L", "comp1_gain_role_M", "comp1_gain_role_S"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- self-test -----------------------------------------------------------

def self_test():
    """In-memory round-trip: build fake P0-2 + P0-1 rows, run the parsers,
    and verify the report produces the expected columns / counts. Uses no
    real CSVs and no real training."""
    print("=== self-test (synthetic) ===", flush=True)
    p0_2_rows = []
    # ratio_1: only d0_large, all 4 seeds, 2 joint + 3 oracle
    for s in SEEDS:
        p0_2_rows.append({"kind": "joint", "ratio": 1, "setting": "d0_large",
                          "method_or_domain": "baseline", "seed": s,
                          "status": "OK", "fit_sec": "100", "peak_mem_mb": "500",
                          "wall_sec": "110",
                          "per_dt_test_auc": ",".join(
                              f"np.float64({0.5 + 0.01 * (3 * t + d):.6f})"
                              for t in TASKS for d in DOMAINS),
                          "best_epoch": "5", "stopped_epoch": "10",
                          "best_val_auc": "0.7", "note": ""})
        p0_2_rows.append({"kind": "joint", "ratio": 1, "setting": "d0_large",
                          "method_or_domain": "comp1", "seed": s,
                          "status": "OK", "fit_sec": "100", "peak_mem_mb": "500",
                          "wall_sec": "110",
                          "per_dt_test_auc": ",".join(
                              f"np.float64({0.5 + 0.01 * (3 * t + d) + 0.005:.6f})"
                              for t in TASKS for d in DOMAINS),
                          "best_epoch": "5", "stopped_epoch": "10",
                          "best_val_auc": "0.7", "note": ""})
        for d in (0, 1, 2):
            p0_2_rows.append({"kind": "oracle", "ratio": 1, "setting": "d0_large",
                              "method_or_domain": f"oracle_d{d}", "seed": s,
                              "status": "OK", "fit_sec": "50", "peak_mem_mb": "300",
                              "wall_sec": "55",
                              "per_dt_test_auc": "np.float64(0.5), np.float64(0.55)",
                              "best_epoch": "5", "stopped_epoch": "10",
                              "best_val_auc": "0.5", "note": ""})
    # ratio_5 and ratio_50: 3 settings each
    for ratio in (5, 50):
        for setting in SETTINGS:
            for s in SEEDS:
                p0_2_rows.append({"kind": "joint", "ratio": ratio,
                                  "setting": setting,
                                  "method_or_domain": "baseline", "seed": s,
                                  "status": "OK", "fit_sec": "100",
                                  "peak_mem_mb": "500", "wall_sec": "110",
                                  "per_dt_test_auc": ",".join(
                                      f"np.float64({0.5 + 0.01 * (3 * t + d):.6f})"
                                      for t in TASKS for d in DOMAINS),
                                  "best_epoch": "5", "stopped_epoch": "10",
                                  "best_val_auc": "0.7", "note": ""})
                p0_2_rows.append({"kind": "joint", "ratio": ratio,
                                  "setting": setting,
                                  "method_or_domain": "comp1", "seed": s,
                                  "status": "OK", "fit_sec": "100",
                                  "peak_mem_mb": "500", "wall_sec": "110",
                                  "per_dt_test_auc": ",".join(
                                      f"np.float64({0.5 + 0.01 * (3 * t + d) + 0.005:.6f})"
                                      for t in TASKS for d in DOMAINS),
                                  "best_epoch": "5", "stopped_epoch": "10",
                                  "best_val_auc": "0.7", "note": ""})
                for d in (0, 1, 2):
                    p0_2_rows.append({"kind": "oracle", "ratio": ratio,
                                      "setting": setting,
                                      "method_or_domain": f"oracle_d{d}",
                                      "seed": s,
                                      "status": "OK", "fit_sec": "50",
                                      "peak_mem_mb": "300", "wall_sec": "55",
                                      "per_dt_test_auc": "np.float64(0.5), np.float64(0.55)",
                                      "best_epoch": "5", "stopped_epoch": "10",
                                      "best_val_auc": "0.5", "note": ""})

    p0_1_rows = []
    for setting in SETTINGS:
        for s in SEEDS:
            p0_1_rows.append({"kind": "joint", "ratio": 20, "setting": setting,
                              "method_or_domain": "baseline", "seed": s,
                              "status": "OK", "fit_sec": "100",
                              "peak_mem_mb": "500", "wall_sec": "110",
                              "per_dt_test_auc": ",".join(
                                  f"np.float64({0.5 + 0.01 * (3 * t + d):.6f})"
                                  for t in TASKS for d in DOMAINS),
                              "best_epoch": "5", "stopped_epoch": "10",
                              "best_val_auc": "0.7", "note": ""})
            p0_1_rows.append({"kind": "joint", "ratio": 20, "setting": setting,
                              "method_or_domain": "comp1", "seed": s,
                              "status": "OK", "fit_sec": "100",
                              "peak_mem_mb": "500", "wall_sec": "110",
                              "per_dt_test_auc": ",".join(
                                  f"np.float64({0.5 + 0.01 * (3 * t + d) + 0.005:.6f})"
                                  for t in TASKS for d in DOMAINS),
                              "best_epoch": "5", "stopped_epoch": "10",
                              "best_val_auc": "0.7", "note": ""})
            for d in (0, 1, 2):
                p0_1_rows.append({"kind": "oracle", "ratio": 20,
                                  "setting": setting,
                                  "method_or_domain": f"oracle_d{d}", "seed": s,
                                  "status": "OK", "fit_sec": "50",
                                  "peak_mem_mb": "300", "wall_sec": "55",
                                  "per_dt_test_auc": "np.float64(0.5), np.float64(0.55)",
                                  "best_epoch": "5", "stopped_epoch": "10",
                                  "best_val_auc": "0.5", "note": ""})

    cells = build_cells(p0_2_rows, p0_1_rows)
    summary = build_summary(cells)
    print(f"  synthetic p0_2_rows={len(p0_2_rows)} p0_1_rows={len(p0_1_rows)}",
          flush=True)
    print(f"  cells rows={len(cells)}", flush=True)
    print(f"  summary rows={len(summary)}", flush=True)
    # Quick sanity: ratio=1 cells = 1 setting * 6 cells * 4 seeds = 24
    r1 = [c for c in cells if int(c["ratio"]) == 1]
    r5 = [c for c in cells if int(c["ratio"]) == 5]
    r20 = [c for c in cells if int(c["ratio"]) == 20]
    r50 = [c for c in cells if int(c["ratio"]) == 50]
    assert len(r1) == 24, f"ratio_1 cells should be 24, got {len(r1)}"
    assert len(r5) == 72, f"ratio_5 cells should be 72, got {len(r5)}"
    assert len(r20) == 72, f"ratio_20 cells should be 72, got {len(r20)}"
    assert len(r50) == 72, f"ratio_50 cells should be 72, got {len(r50)}"
    # comp1 gain: +0.005 everywhere -> summary mean should be 0.005
    s_comp1 = [s for s in summary if s["method"] == "comp1" and int(s["ratio"]) == 5]
    assert abs(float(s_comp1[0]["comp1_gain_overall"]) - 0.005) < 1e-6
    # ratio_1 comp1 gain roles -> degenerate
    s_r1 = [s for s in summary if s["method"] == "comp1" and int(s["ratio"]) == 1]
    assert s_r1[0]["comp1_gain_role_L"] == ""  # ratio_1 has no per-role split
    print("  self-test OK", flush=True)


# -- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true",
                    help="run synthetic parser round-trip and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    os.makedirs(P0_2_BASE, exist_ok=True)
    p0_2_rows = load_p0_2_rows(P0_2_RAW)
    p0_1_rows = load_p0_1_rows(P0_1_DEDUP)
    print(f"loaded {len(p0_2_rows)} P0-2 rows from {P0_2_RAW}", flush=True)
    print(f"loaded {len(p0_1_rows)} P0-1 rows from {P0_1_DEDUP}", flush=True)
    validate_p0_1_role_sizes(p0_1_rows)
    write_all_csv(p0_2_rows, p0_1_rows, OUT_ALL)
    print(f"wrote {OUT_ALL}", flush=True)
    cells = build_cells(p0_2_rows, p0_1_rows)
    write_cells_csv(cells, OUT_CELLS)
    print(f"wrote {OUT_CELLS} ({len(cells)} rows)", flush=True)
    summary = build_summary(cells)
    write_summary_csv(summary, OUT_SUMMARY)
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)", flush=True)


if __name__ == "__main__":
    main()
