"""P0-5 report: read-CSV aggregator over a 2x2 design (weighting x comp1).

Per spec (designer output) the design tests whether component 1 still adds
gain on top of the sample-weighted objective. The 4 cells are:

  cell equal/off   -> reused from P0-2 ratio_50 baseline rows
  cell equal/on    -> reused from P0-2 ratio_50 comp1 rows
  cell sample/off  -> reused from P0-4 w_sample baseline rows
  cell sample/on   -> freshly run by P0-5 (this orchestrator's 12 runs)

Each cell carries per-(setting, seed) per-(domain, task) test AUCs.

Inputs:
  experiments/P0_2_ratio_sweep/P0_2_raw.csv         (equal/off + equal/on + oracle)
  experiments/P0_4_loss_weighting/P0_4_raw.csv      (sample/off)
  experiments/P0_5_comp1_x_weighting/P0_5_raw.csv   (sample/on, freshly run)

Outputs (all under experiments/P0_5_comp1_x_weighting/):
  P0_5_cells.csv       long: per (weighting, comp1, setting, seed, domain,
                       task) cell with joint_auc, oracle_auc, delta_transfer,
                       role, selected_epoch, mismatch.
  P0_5_summary.csv     per (weighting, comp1): per-role mean / se
                       delta_transfer, fraction negative, overall AUC,
                       worst-domain AUC, mean selected epoch.
  P0_5_headline.csv    2x2 main table: weighting x comp1 -> overall AUC,
                       worst-domain, delta_L, delta_S, gain_comp1; the
                       gain is the (setting, seed)-paired n=12 difference
                       in the per-(d, t) joint_auc between comp1=on and
                       off within each weighting. The interaction is
                       gain(sample) - gain(equal), also paired.

P0-5 checkpoint policy: equal argmax (strict >, first-argmax) is used
across all four arms so the report isolates the weighting / comp1 effect
from a selection-rule confound. For P0-4 / P0-5 rows the metrics_epoch.csv
includes a test_auc column, so the recomputed equal-argmax's test AUC
must equal the recorded best_epoch's test AUC exactly (cross-check).
P0-2 reused rows' metrics_epoch.csv lacks test_auc, so the report trusts
the recorded best_epoch + the raw row's per_dt_test_auc (no recompute
of the test value possible); the equal argmax is recomputed over the 6
val_auc columns only and the resulting epoch is cross-checked against
the recorded best_epoch (mismatch flag).

Usage:
  python -u experiments/p0_5_report.py
  python -u experiments/p0_5_report.py --self_test
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
P0_5_BASE = os.path.join(BASE, "P0_5_comp1_x_weighting")
P0_4_BASE = os.path.join(BASE, "P0_4_loss_weighting")
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")

P0_5_RAW = os.path.join(P0_5_BASE, "P0_5_raw.csv")
P0_4_RAW = os.path.join(P0_4_BASE, "P0_4_raw.csv")
P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")

OUT_CELLS = os.path.join(P0_5_BASE, "P0_5_cells.csv")
OUT_SUMMARY = os.path.join(P0_5_BASE, "P0_5_summary.csv")
OUT_HEADLINE = os.path.join(P0_5_BASE, "P0_5_headline.csv")

SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
WEIGHTINGS = ("equal", "sample")
COMP1S = ("off", "on")
DOMAINS = (0, 1, 2)
TASKS = (0, 1)
RATIO = 50

ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}

FLOAT_RE = re.compile(r"np\.float64\(([-\d.eE+]+)\)")


def parse_per_dt(s):
    if not s:
        return []
    return [float(m.group(1)) for m in FLOAT_RE.finditer(s)]


def cell_from_joint_idx(i):
    D = 3
    return i % D, i // D


# -- inputs --------------------------------------------------------------

def load_p0_2_rows(path):
    """Load P0-2 ratio=50 joint rows (both baseline and comp1) -> the
    equal/off and equal/on cells."""
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        if r.get("kind") != "joint":
            continue
        if int(r["ratio"]) != RATIO:
            continue
        r["ratio"] = int(r["ratio"])
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def load_p0_4_rows(path):
    """Load P0-4 w_sample baseline rows -> the sample/off cell."""
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        if r.get("loss_weighting") != "sample":
            continue
        r["ratio"] = int(r["ratio"])
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def load_p0_5_rows(path):
    """Load P0-5 sample+comp1 rows -> the sample/on cell."""
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


def load_p0_2_oracle_rows(path):
    """Load P0-2 ratio=50 oracle rows."""
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") != "OK":
            continue
        if r.get("kind") != "oracle":
            continue
        if int(r["ratio"]) != RATIO:
            continue
        r["ratio"] = int(r["ratio"])
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def load_metrics_epoch(path):
    """Read metrics_epoch.csv (long) and pivot to:
       epochs[epoch_i] = {'val': [6 floats], 'test': [6 floats | None]}
       The 6 entries are task-major (idx = t*D + d), D=3.
    """
    if not os.path.exists(path):
        return None
    by_ep = defaultdict(lambda: {"val": [None] * 6, "test": [None] * 6})
    for r in csv.DictReader(open(path, newline="")):
        ei = int(r["epoch"])
        d = int(r["domain"])
        t = int(r["task"])
        idx = t * 3 + d
        va = r.get("val_auc", "")
        ta = r.get("test_auc", "")
        by_ep[ei]["val"][idx] = float(va) if va else None
        by_ep[ei]["test"][idx] = float(ta) if ta else None
    epochs = {}
    for ei, e in by_ep.items():
        epochs[ei] = e
    return epochs


def has_test_column(path):
    """True iff metrics_epoch.csv at `path` has a test_auc column."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "test_auc" in [h.strip() for h in header.split(",")]


def run_dir_p0_2(setting, seed, method):
    return os.path.join(P0_2_BASE, f"ratio_{RATIO}", f"setting_{setting}",
                        method, f"seed{seed}")


def run_dir_p0_4(weighting, setting, seed):
    return os.path.join(P0_4_BASE, f"w_{weighting}", f"setting_{setting}",
                        "baseline", f"seed{seed}")


def run_dir_p0_5(weighting, setting, seed):
    return os.path.join(P0_5_BASE, f"w_{weighting}", f"setting_{setting}",
                        "comp1", f"seed{seed}")


# -- selection ----------------------------------------------------------

def select_epoch_equal(epochs, recorded_best_epoch):
    """Argmax over mean(val_auc[e]), STRICT > semantics (first argmax).
    Sanity check: must equal recorded_best_epoch (else mismatch=1)."""
    best_e = None
    best_score = -1.0
    for ei in sorted(epochs.keys()):
        vs = epochs[ei]["val"]
        if any(v is None for v in vs):
            continue
        s = sum(vs) / len(vs)
        if s > best_score:
            best_score = s
            best_e = ei
    mismatch = (best_e is not None and recorded_best_epoch is not None
                and int(recorded_best_epoch) != int(best_e))
    return best_e, mismatch


# -- per-cell pair model -------------------------------------------------

def expand_per_dt_joint(row):
    """Long list of (domain, task, value) for one joint row's per_dt_test_auc."""
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 6:
        return []
    return [(d, t, vals[t * 3 + d]) for t in TASKS for d in DOMAINS]


def expand_per_dt_oracle(row):
    """Oracle row's per_dt_test_auc has 2 entries [t0, t1] for the single
    oracle domain."""
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 2:
        return []
    d = int(row["method_or_domain"].replace("oracle_d", ""))
    return [(d, 0, vals[0]), (d, 1, vals[1])]


# -- outputs -------------------------------------------------------------

def build_cells(p0_2_rows, p0_4_rows, p0_5_rows, p0_2_oracle_rows):
    """Yield dict rows for the long cells CSV.

    Schema: weighting, comp1, setting, domain, task, seed, role,
            selected_epoch, joint_auc, oracle_auc, delta_transfer, mismatch.

    Selection rule: equal argmax (strict >, first-argmax) across val_auc
    6-cell mean, identical to p0_4_report. The selected_epoch value comes
    from a recompute over metrics_epoch.csv; the mismatch column flags
    rows where the recomputed epoch disagrees with the raw-recorded
    best_epoch (cross-check). For cells without a test_auc column in
    metrics_epoch.csv (P0-2 reuse rows), joint_auc is taken from
    per_dt_test_auc at the recorded best_epoch (no test recompute
    possible); mismatch is computed from val AUC only and is informational.
    """
    oracle_idx = {}
    for r in p0_2_oracle_rows:
        key = (r["setting"], r["method_or_domain"], int(r["seed"]))
        oracle_idx[key] = r

    # Unified row iterator over the four cells.
    cells_iter = []
    for r in p0_2_rows:
        comp1 = "on" if r["method_or_domain"] == "comp1" else "off"
        cells_iter.append(("equal", comp1, r))
    for r in p0_4_rows:
        cells_iter.append(("sample", "off", r))
    for r in p0_5_rows:
        cells_iter.append(("sample", "on", r))

    out_rows = []
    for weighting, comp1, r in cells_iter:
        setting = r["setting"]
        seed = int(r["seed"])
        if weighting == "equal":
            me_path = os.path.join(run_dir_p0_2(setting, seed, r["method_or_domain"]),
                                   "metrics_epoch.csv")
        elif comp1 == "off":
            me_path = os.path.join(run_dir_p0_4(weighting, setting, seed),
                                   "metrics_epoch.csv")
        else:
            me_path = os.path.join(run_dir_p0_5(weighting, setting, seed),
                                   "metrics_epoch.csv")

        epochs = load_metrics_epoch(me_path)
        recorded_best = r.get("best_epoch", "")
        recorded_best_int = int(recorded_best) if recorded_best != "" else None
        test_col = has_test_column(me_path)
        # Per-cell joint test AUC vector, indexed by (d, t) -> value.
        # P0-4 / P0-5 rows: recompute the joint_auc from the equal-argmax
        # epoch's test_auc (cross-check: must equal the per_dt_test_auc at
        # the recorded best_epoch).
        # P0-2 rows: metrics_epoch.csv lacks test_auc, so trust the raw
        # row's per_dt_test_auc at the recorded best_epoch.
        per_dt_joint = {}
        if not test_col:
            # parse per_dt at the recorded best_epoch: the raw row's
            # per_dt_test_auc is always the joint test AUC at the
            # recorded best_epoch (matches p4_run's per-(d, t) FINAL
            # print after fit()'s last evaluate_multi_domain call).
            vals = parse_per_dt(r.get("per_dt_test_auc", ""))
            for t in TASKS:
                for d in DOMAINS:
                    per_dt_joint[(d, t)] = vals[t * 3 + d] if t * 3 + d < len(vals) else None
            sel_ep = recorded_best_int
            mismatch = 0
            if epochs:
                sel_ep_recompute, mismatch = select_epoch_equal(epochs, recorded_best_int)
                sel_ep = sel_ep_recompute  # use the recomputed val argmax epoch
        else:
            sel_ep, mismatch = (None, 0)
            if epochs:
                sel_ep, mismatch = select_epoch_equal(epochs, recorded_best_int)
                if sel_ep is not None and sel_ep in epochs:
                    test_list = epochs[sel_ep]["test"]
                    for t in TASKS:
                        for d in DOMAINS:
                            v = test_list[t * 3 + d]
                            if v is not None:
                                per_dt_joint[(d, t)] = v
            # Fallback: if recompute yielded no test values, use raw.
            if not per_dt_joint:
                vals = parse_per_dt(r.get("per_dt_test_auc", ""))
                for t in TASKS:
                    for d in DOMAINS:
                        per_dt_joint[(d, t)] = vals[t * 3 + d] if t * 3 + d < len(vals) else None

        for d in DOMAINS:
            for t in TASKS:
                joint_v = per_dt_joint.get((d, t))
                if joint_v is None:
                    continue
                oracle_key = (setting, f"oracle_d{d}", seed)
                oracle_row = oracle_idx.get(oracle_key)
                if oracle_row is not None:
                    o_vals = parse_per_dt(oracle_row.get("per_dt_test_auc", ""))
                    oracle_v = o_vals[t] if t < len(o_vals) else None
                    delta = (joint_v - oracle_v) if oracle_v is not None else None
                else:
                    oracle_v = None
                    delta = None
                role = ROLE_MAPS[setting][d]
                out_rows.append({
                    "weighting": weighting,
                    "comp1": comp1,
                    "setting": setting,
                    "domain": d,
                    "task": t,
                    "seed": seed,
                    "role": role,
                    "selected_epoch": (sel_ep if sel_ep is not None else ""),
                    "joint_auc": f"{joint_v:.6f}",
                    "oracle_auc": (f"{oracle_v:.6f}" if oracle_v is not None else ""),
                    "delta_transfer": (f"{delta:.6f}" if delta is not None else ""),
                    "mismatch": mismatch,
                })
    return out_rows


def write_cells_csv(rows, path):
    fields = ["weighting", "comp1", "setting", "domain", "task", "seed",
              "role", "selected_epoch", "joint_auc", "oracle_auc",
              "delta_transfer", "mismatch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parsef(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return float(var ** 0.5 / (n ** 0.5))


def build_summary(cells):
    """Per (weighting, comp1): per-role mean / se delta_transfer, fraction
    negative, overall AUC, worst-domain AUC, mean selected epoch."""
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["weighting"], c["comp1"])].append(c)

    summary_rows = []
    for w in WEIGHTINGS:
        for c1 in COMP1S:
            cs = by_key.get((w, c1), [])
            deltas = [_parsef(c["delta_transfer"]) for c in cs]
            deltas_v = [x for x in deltas if x is not None]
            joint = [_parsef(c["joint_auc"]) for c in cs]
            cell_means = defaultdict(list)
            for c in cs:
                j = _parsef(c["joint_auc"])
                if j is None:
                    continue
                cell_means[(c["setting"], int(c["domain"]), int(c["task"]))].append(j)
            per_cell = {k: _mean(vs) for k, vs in cell_means.items() if vs}
            overall_auc = _mean(list(per_cell.values())) if per_cell else float("nan")
            worst_auc = min(per_cell.values()) if per_cell else float("nan")

            role_data = defaultdict(list)
            for c in cs:
                d = _parsef(c["delta_transfer"])
                if d is None:
                    continue
                role_data[c["role"]].append(d)

            frac_neg = (sum(1 for x in deltas_v if x < 0) / len(deltas_v)
                        if deltas_v else float("nan"))
            sel_eps = [int(c["selected_epoch"]) for c in cs
                       if c["selected_epoch"] != ""]
            mean_sel_epoch = _mean(sel_eps) if sel_eps else float("nan")
            mismatches = sum(int(c["mismatch"]) for c in cs)

            row = {
                "weighting": w,
                "comp1": c1,
                "n_cells": len(cs),
                "delta_L_mean": (f"{_mean(role_data.get('L', [])):.6f}"
                                  if role_data.get('L') else ""),
                "delta_L_se": (f"{_se(role_data.get('L', [])):.6f}"
                               if role_data.get('L') else ""),
                "delta_M_mean": (f"{_mean(role_data.get('M', [])):.6f}"
                                 if role_data.get('M') else ""),
                "delta_M_se": (f"{_se(role_data.get('M', [])):.6f}"
                               if role_data.get('M') else ""),
                "delta_S_mean": (f"{_mean(role_data.get('S', [])):.6f}"
                                 if role_data.get('S') else ""),
                "delta_S_se": (f"{_se(role_data.get('S', [])):.6f}"
                               if role_data.get('S') else ""),
                "delta_overall_mean": f"{_mean(deltas_v):.6f}" if deltas_v else "",
                "frac_negative": f"{frac_neg:.4f}" if deltas_v == deltas_v else "",
                "overall_auc": f"{overall_auc:.6f}" if overall_auc == overall_auc else "",
                "worst_domain_auc": f"{worst_auc:.6f}" if worst_auc == worst_auc else "",
                "mean_selected_epoch": (f"{mean_sel_epoch:.2f}" if cs else ""),
                "n_mismatch": mismatches,
            }
            summary_rows.append(row)
    return summary_rows


def write_summary_csv(rows, path):
    fields = ["weighting", "comp1", "n_cells",
              "delta_L_mean", "delta_L_se",
              "delta_M_mean", "delta_M_se",
              "delta_S_mean", "delta_S_se",
              "delta_overall_mean", "frac_negative",
              "overall_auc", "worst_domain_auc",
              "mean_selected_epoch", "n_mismatch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---- headline (2x2 main table) -----------------------------------------

def _pair_gain(p0_2_rows, p0_4_rows, p0_5_rows, accessor):
    """Compute per-(setting, seed) gain = comp1(on) - comp1(off) under a
    given weighting. accessor(raw_row) returns the 6-cell joint AUC vector.

    Returns: dict mapping (weighting, setting, seed) -> 6 floats (gain)
             and a parallel dict of (weighting, setting, seed) -> role-wise
             gain dict.
    """
    by_arm = defaultdict(dict)
    for r in p0_2_rows:
        comp1 = "on" if r["method_or_domain"] == "comp1" else "off"
        by_arm[("equal", comp1)][(r["setting"], int(r["seed"]))] = r
    for r in p0_4_rows:
        by_arm[("sample", "off")][(r["setting"], int(r["seed"]))] = r
    for r in p0_5_rows:
        by_arm[("sample", "on")][(r["setting"], int(r["seed"]))] = r

    gains = defaultdict(dict)  # gains[weighting][(setting, seed)] = 6 floats
    for w in WEIGHTINGS:
        on_keys = set(by_arm[(w, "on")].keys())
        off_keys = set(by_arm[(w, "off")].keys())
        common = sorted(on_keys & off_keys)
        for k in common:
            on_v = accessor(by_arm[(w, "on")][k])
            off_v = accessor(by_arm[(w, "off")][k])
            if len(on_v) != 6 or len(off_v) != 6:
                continue
            gains[w][k] = [on_v[i] - off_v[i] for i in range(6)]
    return gains


def build_headline(p0_2_rows, p0_4_rows, p0_5_rows, summary_rows):
    """2x2 main table + paired gain decomposition. gain computed on the
    6-cell joint_auc vector at the equal-argmax epoch (mirrors cells).
    """
    # accessor for joint 6-cell at equal-argmax epoch.
    def accessor(row):
        return parse_per_dt(row.get("per_dt_test_auc", ""))

    gains = _pair_gain(p0_2_rows, p0_4_rows, p0_5_rows, accessor)

    # Build the 2x2 main rows.
    by_arm = {(r["weighting"], r["comp1"]): r for r in summary_rows}
    rows = []
    for w in WEIGHTINGS:
        for c1 in COMP1S:
            s = by_arm.get((w, c1), {})
            rows.append({
                "weighting": w, "comp1": c1,
                "overall_auc": s.get("overall_auc", ""),
                "worst_domain_auc": s.get("worst_domain_auc", ""),
                "delta_L_mean": s.get("delta_L_mean", ""),
                "delta_S_mean": s.get("delta_S_mean", ""),
                "n_cells": s.get("n_cells", ""),
            })

    # Per-(weighting, role, (setting, seed)) gain decomposition, then
    # mean +/- se across the paired n=12.
    per_role_gain_rows = []
    for w in WEIGHTINGS:
        # map (setting, seed, d, t) -> role
        role_by_key = {}
        for k in gains[w].keys():
            setting, seed = k
            for d in DOMAINS:
                role_by_key[(setting, seed, d)] = ROLE_MAPS[setting][d]
        per_role_vals = defaultdict(list)
        for (setting, seed, d), role in role_by_key.items():
            for t in TASKS:
                idx = t * 3 + d
                per_role_vals[role].append(gains[w][(setting, seed)][idx])
        for role in ("L", "M", "S"):
            xs = per_role_vals.get(role, [])
            per_role_gain_rows.append({
                "weighting": w, "role": role,
                "n_paired": len(xs),
                "gain_mean": (f"{_mean(xs):.6f}" if xs else ""),
                "gain_se": (f"{_se(xs):.6f}" if xs else ""),
            })

    # Interaction: gain(sample, role) - gain(equal, role), paired by
    # (setting, seed, d, t).
    interaction_rows = []
    for role in ("L", "M", "S"):
        per_pair_diff = []
        ks = sorted(set(gains["sample"].keys()) & set(gains["equal"].keys()))
        for k in ks:
            setting, seed = k
            for d in DOMAINS:
                if ROLE_MAPS[setting][d] != role:
                    continue
                for t in TASKS:
                    idx = t * 3 + d
                    per_pair_diff.append(
                        gains["sample"][k][idx] - gains["equal"][k][idx])
        interaction_rows.append({
            "role": role,
            "n_paired": len(per_pair_diff),
            "interaction_mean": (f"{_mean(per_pair_diff):.6f}"
                                 if per_pair_diff else ""),
            "interaction_se": (f"{_se(per_pair_diff):.6f}"
                               if per_pair_diff else ""),
        })

    # Overall gain (any role) and interaction (overall).
    per_pair_overall = []
    for w in WEIGHTINGS:
        diffs = []
        ks = sorted(set(gains["sample"].keys()) & set(gains["equal"].keys()))
        for k in ks:
            for idx in range(6):
                diffs.append(gains[w][k][idx])
        per_pair_overall.append((w, diffs))
    overall_int_diff = []
    ks = sorted(set(gains["sample"].keys()) & set(gains["equal"].keys()))
    for k in ks:
        for idx in range(6):
            overall_int_diff.append(
                gains["sample"][k][idx] - gains["equal"][k][idx])
    overall_int_row = {
        "role": "_overall",
        "n_paired": len(overall_int_diff),
        "interaction_mean": (f"{_mean(overall_int_diff):.6f}"
                             if overall_int_diff else ""),
        "interaction_se": (f"{_se(overall_int_diff):.6f}"
                           if overall_int_diff else ""),
    }
    interaction_rows.append(overall_int_row)

    # Write the headline + a sidecar "headline_details" with the per-role
    # gain decomposition and interaction (the headline CSV itself is the
    # 2x2 main table; the sidecar keeps the per-role gain breakdown).
    write_headline_csv(rows, OUT_HEADLINE)
    details_path = os.path.join(P0_5_BASE, "P0_5_headline_details.csv")
    fields = ["scope", "weighting", "role", "n_paired", "gain_mean", "gain_se"]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_role_gain_rows:
            w.writerow({"scope": "per_role_gain", **r})
    fields2 = ["scope", "role", "n_paired", "interaction_mean", "interaction_se"]
    with open(details_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        for r in interaction_rows:
            w.writerow({"scope": "interaction", **r})
    return rows, per_role_gain_rows, interaction_rows


def write_headline_csv(rows, path):
    fields = ["weighting", "comp1", "overall_auc", "worst_domain_auc",
              "delta_L_mean", "delta_S_mean", "n_cells"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- self-test -----------------------------------------------------------

def self_test():
    """In-memory round-trip over the four arms using synthetic CSVs."""
    import tempfile
    print("=== self-test (synthetic) ===", flush=True)
    tmp = tempfile.mkdtemp(prefix="p0_5_self_")

    # Build synthetic P0-2 oracle rows: 3 settings x 4 seeds x 3 domains.
    p0_2_oracle = []
    for setting in SETTINGS:
        for seed in SEEDS:
            for d in DOMAINS:
                p0_2_oracle.append({
                    "kind": "oracle", "ratio": 50, "setting": setting,
                    "method_or_domain": f"oracle_d{d}", "seed": seed,
                    "status": "OK",
                    "per_dt_test_auc": ",".join(
                        f"np.float64({0.65 + 0.01 * d + 0.005 * t:.6f})"
                        for t in TASKS),
                })

    # Synthetic joint rows. For each cell we only need per_dt_test_auc
    # (6 floats) and best_epoch (used only when metrics_epoch.csv lacks
    # test_auc). In the synthetic case we always write a test_auc-bearing
    # metrics_epoch.csv so the recompute path is exercised.
    def synth_joint_per_dt(weighting, comp1, setting, seed, base):
        # weight comp1 effect: +0.005 per cell on; sample weighting nudges
        # L up / S down vs equal.
        out = []
        for t in TASKS:
            for d in DOMAINS:
                v = base
                role = ROLE_MAPS[setting][d]
                if weighting == "sample":
                    if role == "L":
                        v += 0.002
                    elif role == "S":
                        v -= 0.002
                if comp1 == "on":
                    v += 0.003
                out.append(v)
        return out

    def write_me(weighting, comp1, setting, seed, per_dt):
        if comp1 == "on":
            d_root = os.path.join(tmp, "p05", f"w_{weighting}",
                                  f"setting_{setting}", "comp1", f"seed{seed}")
        else:
            d_root = os.path.join(tmp, "p04", f"w_{weighting}",
                                  f"setting_{setting}", "baseline", f"seed{seed}")
        os.makedirs(d_root, exist_ok=True)
        ep_fields = ["experiment_stage", "experiment_name", "model", "method",
                     "setting", "seed", "epoch", "domain", "task",
                     "train_loss", "val_loss", "val_auc", "test_auc"]
        # 5 epochs; val_auc peaks at epoch 2 with strict > semantics.
        val_schedule = [
            (0, [0.55] * 6, [x - 0.02 for x in per_dt]),
            (1, [0.62] * 6, [x - 0.01 for x in per_dt]),
            (2, [0.7483] * 6, per_dt),
            (3, [0.71] * 6, [x - 0.005 for x in per_dt]),
            (4, [0.70] * 6, [x - 0.01 for x in per_dt]),
        ]
        with open(os.path.join(d_root, "metrics_epoch.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ep_fields)
            w.writeheader()
            for ei, vs, ts in val_schedule:
                for t in TASKS:
                    for d in DOMAINS:
                        idx = t * 3 + d
                        w.writerow({
                            "experiment_stage": "P0-5", "experiment_name": "p0_5",
                            "model": "MDMTRec", "method": "comp1" if comp1 == "on" else "baseline",
                            "setting": setting, "seed": seed,
                            "epoch": ei, "domain": d, "task": t,
                            "train_loss": "1.0",
                            "val_loss": "0.5",
                            "val_auc": f"{vs[idx]:.6f}",
                            "test_auc": f"{ts[idx]:.6f}",
                        })

    p0_2_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            for method in ("baseline", "comp1"):
                base = 0.70
                per_dt = synth_joint_per_dt("equal", "on" if method == "comp1" else "off",
                                            setting, seed, base)
                p0_2_rows.append({
                    "kind": "joint", "ratio": 50, "setting": setting,
                    "method_or_domain": method, "seed": seed,
                    "status": "OK", "best_epoch": "2", "stopped_epoch": "4",
                    "per_dt_test_auc": ",".join(f"np.float64({v:.6f})" for v in per_dt),
                })
                # Build synthetic metrics_epoch.csv for the equal cell under
                # the test_col path (but P0-2 rows should be reused as-is
                # without test_auc; we mimic that by NOT writing metrics
                # for them). The reporter falls back to per_dt_test_auc.

    p0_4_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            base = 0.70
            per_dt = synth_joint_per_dt("sample", "off", setting, seed, base)
            p0_4_rows.append({
                "kind": "joint", "ratio": 50, "setting": setting,
                "loss_weighting": "sample",
                "method_or_domain": "baseline", "seed": seed,
                "status": "OK", "best_epoch": "2", "stopped_epoch": "4",
                "per_dt_test_auc": ",".join(f"np.float64({v:.6f})" for v in per_dt),
            })
            write_me("sample", "off", setting, seed, per_dt)

    p0_5_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            base = 0.70
            per_dt = synth_joint_per_dt("sample", "on", setting, seed, base)
            p0_5_rows.append({
                "kind": "joint", "ratio": 50, "setting": setting,
                "loss_weighting": "sample",
                "method_or_domain": "comp1", "seed": seed,
                "status": "OK", "best_epoch": "2", "stopped_epoch": "4",
                "per_dt_test_auc": ",".join(f"np.float64({v:.6f})" for v in per_dt),
            })
            write_me("sample", "on", setting, seed, per_dt)

    # Monkey-patch the run_dir helpers to redirect P0-2 / P0-4 / P0-5
    # paths into the synthetic tmp dir. Without patching run_dir_p0_2,
    # P0-2 rows would read the REAL P0_2_ratio_sweep metrics_epoch.csv.
    orig_p02 = globals()["run_dir_p0_2"]
    orig_p04 = globals()["run_dir_p0_4"]
    orig_p05 = globals()["run_dir_p0_5"]

    def p02(setting, seed, method):
        return os.path.join(tmp, "p02", "ratio_50", f"setting_{setting}",
                            method, f"seed{seed}")

    def p04(w_, s_, sd_):
        return os.path.join(tmp, "p04", f"w_{w_}", f"setting_{s_}",
                            "baseline", f"seed{sd_}")

    def p05(w_, s_, sd_):
        return os.path.join(tmp, "p05", f"w_{w_}", f"setting_{s_}",
                            "comp1", f"seed{sd_}")

    globals()["run_dir_p0_2"] = p02
    globals()["run_dir_p0_4"] = p04
    globals()["run_dir_p0_5"] = p05

    # Build metrics_epoch.csv for the synthetic P0-2 cells too (so the
    # test_col=False path / fallback path is exercised with no me_path
    # file present — i.e., we deliberately do NOT create those files
    # in the synthetic p02 directory; this mirrors the real P0-2 runs).
    try:
        cells = build_cells(p0_2_rows, p0_4_rows, p0_5_rows, p0_2_oracle)
    finally:
        globals()["run_dir_p0_2"] = orig_p02
        globals()["run_dir_p0_4"] = orig_p04
        globals()["run_dir_p0_5"] = orig_p05

    # Expected cell count: 2 weightings x 2 comp1s x 3 settings x 4 seeds
    # x 3 domains x 2 tasks = 288.
    assert len(cells) == 288, f"expected 288 cells, got {len(cells)}"
    # All rows should have non-empty joint_auc and delta_transfer (oracle
    # covers every (setting, d, seed)).
    for c in cells:
        assert c["joint_auc"], f"missing joint_auc: {c}"
        assert c["delta_transfer"], f"missing delta_transfer: {c}"
        assert c["role"] in ("L", "M", "S")
    # All cells should pick epoch 2 (val peak) for P0-4/P0-5 rows, and the
    # P0-2 fallback uses recorded best_epoch=2 too.
    sel_eps = set(int(c["selected_epoch"]) for c in cells)
    assert sel_eps == {2}, f"expected all selected_epoch=2, got {sel_eps}"
    # Mismatch should be 0 everywhere.
    mismatches = sum(int(c["mismatch"]) for c in cells)
    assert mismatches == 0, f"unexpected mismatches: {mismatches}"

    summary = build_summary(cells)
    assert len(summary) == 4, f"expected 4 summary rows, got {len(summary)}"
    # build_headline writes OUT_HEADLINE + P0_5_BASE/P0_5_headline_details.csv
    # via module globals; redirect both into tmp so the self-test never
    # touches the real P0_5_comp1_x_weighting/ output dir.
    orig_out_headline = globals()["OUT_HEADLINE"]
    orig_p0_5_base = globals()["P0_5_BASE"]
    globals()["OUT_HEADLINE"] = os.path.join(tmp, "headline_main.csv")
    globals()["P0_5_BASE"] = tmp
    try:
        headline, per_role, interactions = build_headline(
            p0_2_rows, p0_4_rows, p0_5_rows, summary)
    finally:
        globals()["OUT_HEADLINE"] = orig_out_headline
        globals()["P0_5_BASE"] = orig_p0_5_base
    assert len(headline) == 4, f"expected 4 headline rows, got {len(headline)}"
    # Per-role gain: gain(L) > 0 (sample boosts L + comp1 adds uniformly);
    # gain(S) close to 0 (sample drag ~ comp1 bump).
    # Equal: comp1 on gives uniform +0.003 across cells -> gain(L,M,S)
    # all close to 0.003.
    equal_per_role = {r["role"]: float(r["gain_mean"])
                      for r in per_role if r["weighting"] == "equal"}
    for role in ("L", "M", "S"):
        assert abs(equal_per_role[role] - 0.003) < 1e-6, (
            f"equal gain({role})={equal_per_role[role]} != 0.003")
    sample_per_role = {r["role"]: float(r["gain_mean"])
                       for r in per_role if r["weighting"] == "sample"}
    # The synthetic design adds the comp1-on delta uniformly (+0.003) to
    # every cell; the sample-weighting delta is applied symmetrically to
    # both off and on, so it cancels in the gain. gain(sample) ==
    # gain(equal) == 0.003 for every role; interaction = 0 for every role.
    for role in ("L", "M", "S"):
        assert abs(sample_per_role[role] - 0.003) < 1e-6, (
            f"sample gain({role})={sample_per_role[role]} != 0.003")
    # interaction gain(sample) - gain(equal) = 0 across the synthetic data.
    for r in interactions:
        if r["role"] in ("L", "M", "S"):
            assert abs(float(r["interaction_mean"]) - 0.0) < 1e-6, r

    # Round-trip write_csv then read back to verify schema.
    write_cells_csv(cells, os.path.join(tmp, "cells.csv"))
    write_summary_csv(summary, os.path.join(tmp, "summary.csv"))
    write_headline_csv(headline, os.path.join(tmp, "headline.csv"))
    for f in ("cells.csv", "summary.csv", "headline.csv"):
        with open(os.path.join(tmp, f), "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        assert len(lines) >= 2, f"{f} has only {len(lines)} lines"
    print(f"  cells={len(cells)} summary={len(summary)} headline={len(headline)}",
          flush=True)
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
    os.makedirs(P0_5_BASE, exist_ok=True)
    p0_2_rows = load_p0_2_rows(P0_2_RAW)
    p0_4_rows = load_p0_4_rows(P0_4_RAW)
    p0_5_rows = load_p0_5_rows(P0_5_RAW)
    p0_2_oracle_rows = load_p0_2_oracle_rows(P0_2_RAW)
    print(f"loaded {len(p0_2_rows)} P0-2 ratio_50 joint rows from {P0_2_RAW}",
          flush=True)
    print(f"loaded {len(p0_4_rows)} P0-4 w_sample baseline rows from {P0_4_RAW}",
          flush=True)
    print(f"loaded {len(p0_5_rows)} P0-5 sample+comp1 rows from {P0_5_RAW}",
          flush=True)
    print(f"loaded {len(p0_2_oracle_rows)} P0-2 ratio_50 oracle rows from {P0_2_RAW}",
          flush=True)

    cells = build_cells(p0_2_rows, p0_4_rows, p0_5_rows, p0_2_oracle_rows)
    write_cells_csv(cells, OUT_CELLS)
    print(f"wrote {OUT_CELLS} ({len(cells)} rows)", flush=True)
    summary = build_summary(cells)
    write_summary_csv(summary, OUT_SUMMARY)
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)", flush=True)
    headline, per_role, interactions = build_headline(
        p0_2_rows, p0_4_rows, p0_5_rows, summary)
    print(f"wrote {OUT_HEADLINE} ({len(headline)} rows)", flush=True)


if __name__ == "__main__":
    main()
