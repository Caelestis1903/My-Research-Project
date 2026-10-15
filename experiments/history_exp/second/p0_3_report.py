"""P0-3 report: pair MMoE (P0-3) and M3OE MDMTRec (P0-2 ratio_1 / ratio_50)
rows cell-by-cell and emit the backbone transfer matrix.

Inputs:
  experiments/P0_3_mmoe_transfer/P0_3_raw.csv       (MMoE, fresh)
  experiments/P0_2_ratio_sweep/P0_2_raw.csv         (M3OE MDMTRec, fresh, filtered
                                                     to ratio in {1, 50} + status OK)

Outputs (all under experiments/P0_3_mmoe_transfer/):
  P0_3_cells.csv
      long table: backbone x tier x setting x domain x task x seed x
                  {baseline_auc, comp1_auc, oracle_auc,
                   delta_transfer, comp1_gain}
                  (delta_transfer = joint - oracle at the same cell;
                   comp1_gain     = comp1 - baseline at the same cell)
  P0_3_transfer_matrix.csv
      per (backbone, tier) 6-cell mean +/- std of delta_transfer, plus
      per-domain and overall means. Strong tier n=12 (3 settings x 4 seeds),
      plus a per-setting strong breakdown (n=4 each).
  P0_3_summary.csv
      per (backbone, tier): mean AUC, worst-domain AUC, comp1 gain
      (overall + per role; balanced tier is degenerate so role columns
      are reported with the "degenerate" marker per the spec).

per_dt_test_auc string layout (np.float64 repr) is parsed the same way as
p0_2_report.py: joint -> 6 elements task-major [d0t0,d1t0,d2t0,d0t1,d1t1,d2t1],
idx -> (d, t) = (idx % 3, idx // 3). Oracle -> 2 elements [t0, t1] for the
single oracle domain.

The balanced tier collapses the role distinction (L=M=S in expectation), so
we mark role aggregates with `degenerate` per the spec.

The report is fully self-contained: it parses CSVs, never touches models,
and never runs training. A synthetic --self_test mode verifies the parser
end-to-end without real run CSVs.

Usage:
  python -u experiments/p0_3_report.py
  python -u experiments/p0_3_report.py --self_test
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
P0_3_BASE = os.path.join(BASE, "P0_3_mmoe_transfer")
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")

P0_3_RAW = os.path.join(P0_3_BASE, "P0_3_raw.csv")
P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")

OUT_CELLS = os.path.join(P0_3_BASE, "P0_3_cells.csv")
OUT_MATRIX = os.path.join(P0_3_BASE, "P0_3_transfer_matrix.csv")
OUT_SUMMARY = os.path.join(P0_3_BASE, "P0_3_summary.csv")

# tiers carried over from P0-3 orchestrator; P0-2 used `ratio` as the tier
# label, so the cross-report mapping is ratio in {1, 50} -> tier.
P0_3_TIERS = ("balanced", "strong")
P0_2_TO_TIER = {1: "balanced", 50: "strong"}
P0_3_TO_RATIO = {"balanced": 1, "strong": 50}

SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
DOMAINS = (0, 1, 2)
TASKS = (0, 1)

FLOAT_RE = re.compile(r"np\.float64\(([-\d.eE+]+)\)")

# Same as p0_2_report.py: d0 plays L in d0_large, S in d1_large, M in d2_large.
ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}


# -- parsing --------------------------------------------------------------

def parse_per_dt(s):
    """Parse 'np.float64(x), np.float64(y), ...' to a list of floats."""
    if not s:
        return []
    return [float(m.group(1)) for m in FLOAT_RE.finditer(s)]


# idx -> (d, t) for joint rows: task-major [d0t0,d1t0,d2t0,d0t1,d1t1,d2t1].
def cell_from_joint_idx(i):
    return i % 3, i // 3


def expand_per_dt(row):
    """(domain, task, value) tuples for one row. Joint -> 6 cells; oracle -> 2
    cells (t0, t1) on the single oracle domain."""
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if row["kind"] == "joint":
        if len(vals) != 6:
            return []
        return [(d, t, vals[t * 3 + d]) for t in TASKS for d in DOMAINS]
    if row["kind"] == "oracle":
        if len(vals) != 2:
            return []
        d = int(row["method_or_domain"].replace("oracle_d", ""))
        return [(d, 0, vals[0]), (d, 1, vals[1])]
    return []


# -- inputs ---------------------------------------------------------------

def load_p0_3_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        r["tier"] = r["tier"]
        r["ratio"] = int(r["ratio"])
        r["seed"] = int(r["seed"])
        r["backbone"] = "mmoe"
        r["kind"] = ("oracle" if r["method_or_domain"].startswith("oracle_d")
                     else "joint")
        rows.append(r)
    return rows


def load_p0_2_rows(path, ratios=(1, 50)):
    """Read P0-2 raw and mark the rows as M3OE MDMTRec with backbone='mdmoe'.
    The tier is derived from the ratio column. status=OK + ratio in the
    allowed set are the only rows we keep."""
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        try:
            ratio = int(r["ratio"])
        except (TypeError, ValueError):
            continue
        if ratio not in ratios:
            continue
        r["ratio"] = ratio
        r["seed"] = int(r["seed"])
        r["backbone"] = "mdmoe"
        r["tier"] = P0_2_TO_TIER[ratio]
        r["kind"] = ("oracle" if r["method_or_domain"].startswith("oracle_d")
                     else "joint")
        rows.append(r)
    return rows


# -- per-cell pairing -----------------------------------------------------

def build_cells(p0_3_rows, p0_2_rows):
    """Long cells table. For every (tier, setting, domain, task, seed, backbone)
    triple we emit one row with baseline / comp1 / oracle AUCs and the
    derived deltas. Pairing is exact: both backbones share the same cached
    data (sampling rule is identical), so we just look up by
    (backbone, tier, setting, kind, method_or_domain, seed)."""
    pool = []
    for r in p0_3_rows:
        pool.append(r)
    for r in p0_2_rows:
        pool.append(r)

    idx = {}
    for r in pool:
        key = (r["backbone"], r["tier"], r["setting"], r["kind"],
               r["method_or_domain"], r["seed"])
        idx[key] = r

    out_rows = []
    for backbone in ("mdmoe", "mmoe"):
        for tier in P0_3_TIERS:
            for setting in SETTINGS:
                for seed in SEEDS:
                    bk = (backbone, tier, setting, "joint", "baseline", seed)
                    ck = (backbone, tier, setting, "joint", "comp1", seed)
                    if bk not in idx or ck not in idx:
                        continue
                    b_cells = expand_per_dt(idx[bk])
                    c_cells = expand_per_dt(idx[ck])
                    if not b_cells or not c_cells:
                        continue
                    b_map = {(d, t): v for (d, t, v) in b_cells}
                    c_map = {(d, t): v for (d, t, v) in c_cells}
                    for (d, t) in sorted(b_map):
                        ok = (backbone, tier, setting, "oracle",
                              f"oracle_d{d}", seed)
                        if ok in idx:
                            o_cells = expand_per_dt(idx[ok])
                            o_map = {(dd, tt): v for (dd, tt, v) in o_cells}
                            o_val = o_map.get((d, t), "")
                        else:
                            o_val = ""
                        baseline = b_map[(d, t)]
                        comp1 = c_map[(d, t)]
                        try:
                            oracle = float(o_val) if o_val != "" else float("nan")
                            delta = baseline - oracle
                            gain = comp1 - baseline
                        except (TypeError, ValueError):
                            oracle = float("nan")
                            delta = float("nan")
                            gain = float("nan")
                        role = ROLE_MAPS[setting][d]
                        out_rows.append({
                            "backbone": backbone,
                            "tier": tier,
                            "setting": setting,
                            "domain": d,
                            "task": t,
                            "seed": seed,
                            "role": role,
                            "baseline_auc": f"{baseline:.6f}",
                            "comp1_auc": f"{comp1:.6f}",
                            "oracle_auc": _fmt_or_blank(oracle),
                            "delta_transfer": _fmt_or_blank(delta),
                            "comp1_gain": _fmt_or_blank(gain),
                        })
    return out_rows


def _fmt_or_blank(x):
    if isinstance(x, float) and x != x:   # NaN check (NaN != NaN)
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    if x == "":
        return ""
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return ""


def write_cells_csv(rows, path):
    fields = ["backbone", "tier", "setting", "domain", "task", "seed", "role",
              "baseline_auc", "comp1_auc", "oracle_auc",
              "delta_transfer", "comp1_gain"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- transfer matrix ------------------------------------------------------

def build_transfer_matrix(cells):
    """Per (backbone, tier): 6-cell mean+/-std of delta_transfer, plus
    per-domain and overall mean. For the strong tier we also emit a
    per-setting breakdown (n=4 each) for diagnostic purposes.

    The matrix rows are: backbone x tier x (cell | domain_mean | overall | setting)."""
    by_key = defaultdict(list)
    for c in cells:
        try:
            v = float(c["delta_transfer"])
        except (TypeError, ValueError):
            continue
        if v != v:    # NaN
            continue
        by_key[(c["backbone"], c["tier"])].append(c)

    def stat(xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return ("", "")
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return (f"{m:.6f}", "")
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        std = var ** 0.5
        return (f"{m:.6f}", f"{std:.6f}")

    out = []
    for backbone in ("mdmoe", "mmoe"):
        for tier in P0_3_TIERS:
            rcells = by_key.get((backbone, tier), [])
            # 6-cell breakdown
            by_cell = defaultdict(list)
            for c in rcells:
                by_cell[(int(c["domain"]), int(c["task"]))].append(
                    float(c["delta_transfer"]))
            for d in DOMAINS:
                for t in TASKS:
                    m, s = stat(by_cell.get((d, t), []))
                    out.append({
                        "backbone": backbone, "tier": tier, "scope": f"cell_d{d}t{t}",
                        "n": len(by_cell.get((d, t), [])),
                        "mean_delta_transfer": m, "std_delta_transfer": s,
                    })
            # per-domain mean
            by_dom = defaultdict(list)
            for c in rcells:
                by_dom[int(c["domain"])].append(float(c["delta_transfer"]))
            for d in DOMAINS:
                m, s = stat(by_dom.get(d, []))
                out.append({
                    "backbone": backbone, "tier": tier, "scope": f"domain_mean_d{d}",
                    "n": len(by_dom.get(d, [])),
                    "mean_delta_transfer": m, "std_delta_transfer": s,
                })
            # overall
            m, s = stat([float(c["delta_transfer"]) for c in rcells])
            out.append({
                "backbone": backbone, "tier": tier, "scope": "overall",
                "n": len(rcells),
                "mean_delta_transfer": m, "std_delta_transfer": s,
            })
            # strong tier: per-setting breakdown (n=4 each)
            if tier == "strong":
                by_set = defaultdict(list)
                for c in rcells:
                    by_set[c["setting"]].append(float(c["delta_transfer"]))
                for setting in SETTINGS:
                    m, s = stat(by_set.get(setting, []))
                    out.append({
                        "backbone": backbone, "tier": tier,
                        "scope": f"setting_{setting}",
                        "n": len(by_set.get(setting, [])),
                        "mean_delta_transfer": m, "std_delta_transfer": s,
                    })
    return out


def write_matrix_csv(rows, path):
    fields = ["backbone", "tier", "scope", "n",
              "mean_delta_transfer", "std_delta_transfer"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- summary (mean / worst-domain / comp1 gain) ---------------------------

def build_summary(cells):
    """Per (backbone, tier): mean AUC across all method-cell-seed values,
    worst-domain AUC (per cell, mean over settings+seeds -> min over 6 cells),
    comp1 gain overall + per role. balanced tier -> role marked degenerate."""

    def parsef(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else float("nan")

    summary_rows = []
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["backbone"], c["tier"])].append(c)

    for backbone in ("mdmoe", "mmoe"):
        for tier in P0_3_TIERS:
            rcells = by_key[(backbone, tier)]
            for method in ("baseline", "comp1", "oracle"):
                cell_avgs = defaultdict(list)   # (d, t) -> [v over seed,setting]
                for c in rcells:
                    v = parsef(c[f"{method}_auc"])
                    if v is None:
                        continue
                    cell_avgs[(int(c["domain"]), int(c["task"]))].append(v)
                cell_means = {k: mean(vs) for k, vs in cell_avgs.items() if vs}
                mean_auc = mean(list(cell_means.values())) if cell_means else float("nan")
                worst = (min(cell_means.values()) if cell_means else float("nan"))
                if method == "comp1":
                    gain_all = []
                    gain_by_role = defaultdict(list)
                    for c in rcells:
                        g = parsef(c["comp1_gain"])
                        if g is None:
                            continue
                        gain_all.append(g)
                        gain_by_role[c["role"]].append(g)
                    if tier == "balanced":
                        # role is degenerate at 1:1 (L=M=S in expectation).
                        role_l = role_m = role_s = "degenerate"
                    else:
                        role_l = f"{mean(gain_by_role.get('L', [])):.6f}" if gain_by_role.get("L") else ""
                        role_m = f"{mean(gain_by_role.get('M', [])):.6f}" if gain_by_role.get("M") else ""
                        role_s = f"{mean(gain_by_role.get('S', [])):.6f}" if gain_by_role.get("S") else ""
                    summary_rows.append({
                        "backbone": backbone, "tier": tier, "method": method,
                        "mean_auc": f"{mean_auc:.6f}",
                        "worst_domain_auc": f"{worst:.6f}",
                        "comp1_gain_overall": f"{mean(gain_all):.6f}",
                        "comp1_gain_role_L": role_l,
                        "comp1_gain_role_M": role_m,
                        "comp1_gain_role_S": role_s,
                    })
                else:
                    summary_rows.append({
                        "backbone": backbone, "tier": tier, "method": method,
                        "mean_auc": f"{mean_auc:.6f}",
                        "worst_domain_auc": f"{worst:.6f}",
                        "comp1_gain_overall": "",
                        "comp1_gain_role_L": "",
                        "comp1_gain_role_M": "",
                        "comp1_gain_role_S": "",
                    })
    return summary_rows


def write_summary_csv(rows, path):
    fields = ["backbone", "tier", "method", "mean_auc", "worst_domain_auc",
              "comp1_gain_overall",
              "comp1_gain_role_L", "comp1_gain_role_M", "comp1_gain_role_S"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- self-test ------------------------------------------------------------

def self_test():
    """In-memory round-trip: build fake MMoE + MDMTRec rows, run the parsers,
    and verify the report produces the expected columns / counts."""
    print("=== self-test (synthetic) ===", flush=True)

    def row(backbone, tier, ratio, setting, kind, mod, seed, base_off=0.0,
            comp1_off=0.0):
        if kind == "joint":
            vals = ",".join(
                f"np.float64({0.7 + 0.01 * (3 * t + d) + base_off + comp1_off:.6f})"
                for t in TASKS for d in DOMAINS)
        else:
            vals = "np.float64(0.65), np.float64(0.66)"
        d = int(mod.replace("oracle_d", "")) if "oracle_d" in mod else 0
        return {
            "backbone": backbone, "tier": tier, "ratio": ratio,
            "setting": setting, "kind": kind, "method_or_domain": mod,
            "seed": seed, "status": "OK",
            "per_dt_test_auc": vals,
        }

    p0_3_rows = []
    p0_2_rows = []
    for backbone in ("mmoe", "mdmoe"):
        for tier, ratio, settings in [
            ("balanced", 1, ["d0_large"]),
            ("strong",   50, list(SETTINGS)),
        ]:
            for setting in settings:
                for s in SEEDS:
                    for kind, mod in [("joint", "baseline"),
                                      ("joint", "comp1")]:
                        r = row(backbone, tier, ratio, setting, kind, mod, s,
                                comp1_off=0.005 if mod == "comp1" else 0.0)
                        if backbone == "mmoe":
                            p0_3_rows.append(r)
                        else:
                            p0_2_rows.append(r)
                    for d in DOMAINS:
                        r = row(backbone, tier, ratio, setting, "oracle",
                                f"oracle_d{d}", s)
                        if backbone == "mmoe":
                            p0_3_rows.append(r)
                        else:
                            p0_2_rows.append(r)

    cells = build_cells(p0_3_rows, p0_2_rows)
    matrix = build_transfer_matrix(cells)
    summary = build_summary(cells)
    print(f"  cells={len(cells)} matrix={len(matrix)} summary={len(summary)}",
          flush=True)

    # Cell count: per backbone, balanced=1*6*4=24, strong=3*6*4=72 -> 192
    expected = 2 * (1 * 6 * 4 + 3 * 6 * 4)
    assert len(cells) == expected, f"cells should be {expected}, got {len(cells)}"
    # comp1 gain is +0.005 everywhere
    for c in cells:
        assert abs(float(c["comp1_gain"]) - 0.005) < 1e-6
    # balanced role -> one of (L/M/S) (the role depends on (setting, d));
    # but gain_by_role is collapsed to "degenerate" in the summary.
    bal = [c for c in cells if c["tier"] == "balanced"]
    assert len(bal) == 2 * 24
    # matrix: 6 cells + 3 domain + 1 overall = 10 scopes; strong gets 3 more.
    assert len(matrix) == 2 * 2 * 10 + 2 * 3  # 40 + 6 = 46 (strong gets 3 extra per backbone)
    # summary: per (backbone, tier) * 3 methods = 2*2*3 = 12
    assert len(summary) == 12
    # comp1 row for balanced tier: role columns say "degenerate"
    bal_comp1 = [s for s in summary
                 if s["tier"] == "balanced" and s["method"] == "comp1"]
    for r in bal_comp1:
        assert r["comp1_gain_role_L"] == "degenerate"
        assert r["comp1_gain_role_M"] == "degenerate"
        assert r["comp1_gain_role_S"] == "degenerate"
    print("  self-test OK", flush=True)


# -- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true",
                    help="run synthetic parser round-trip and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    os.makedirs(P0_3_BASE, exist_ok=True)
    p0_3_rows = load_p0_3_rows(P0_3_RAW)
    p0_2_rows = load_p0_2_rows(P0_2_RAW)
    print(f"loaded {len(p0_3_rows)} P0-3 MMoE rows from {P0_3_RAW}", flush=True)
    print(f"loaded {len(p0_2_rows)} P0-2 M3OE rows from {P0_2_RAW} "
          f"(ratio in {{1, 50}})", flush=True)

    cells = build_cells(p0_3_rows, p0_2_rows)
    write_cells_csv(cells, OUT_CELLS)
    print(f"wrote {OUT_CELLS} ({len(cells)} rows)", flush=True)

    matrix = build_transfer_matrix(cells)
    write_matrix_csv(matrix, OUT_MATRIX)
    print(f"wrote {OUT_MATRIX} ({len(matrix)} rows)", flush=True)

    summary = build_summary(cells)
    write_summary_csv(summary, OUT_SUMMARY)
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)", flush=True)


if __name__ == "__main__":
    main()
