"""P1-3 report: read-CSV aggregator over the Matched Momentum 4-arm design.

Four arms (n=12=3 setting x 4 seed, strong ratio_50, 150000/34500/3000,
seeds 2022-2025, all --loss_weighting sample):

  A: baseline w_sample     -> reused from P0-4 w_sample/baseline
  B: comp1 (γ_d domain-conditioned 0.9/0.977/0.99)
                           -> reused from P0-5 w_sample/comp1
  C: comp1 uniform γ*=0.9432 -> freshly run by P1-3 orchestrator (12 runs)
  D: Adam β1*=0.9432       -> freshly run by P1-3 orchestrator (12 runs;
                              STRUCTURAL CONTROL — does not participate in
                              the H1/H2 verdict)

Inputs:
  experiments/P1_3_matched_momentum/P1_3_raw.csv
    (reused arms show status=OK_REUSED; fresh arms show status=OK)
  experiments/P0_2_ratio_sweep/P0_2_raw.csv  (ratio_50 oracle rows for
                                              Δ_transfer pairing)
  Reuse cells under P1_3_matched_momentum/{A,B}/<rest>/metrics_epoch.csv
  Fresh cells under P1_3_matched_momentum/{C,D}/<rest>/metrics_epoch.csv

Outputs (all under experiments/P1_3_matched_momentum/):
  P1_3_cells.csv        long: per (arm, setting, domain, task, seed)
                         cell with joint_auc, oracle_auc, delta_transfer,
                         role, selected_epoch, mismatch.
  P1_3_summary.csv      per arm: overall / worst-domain / Δ_L/Δ_M/Δ_S
                         mean+/-std, mean selected epoch, n_cells.
  P1_3_headline.csv     paired gains Δ_B/Δ_C/Δ_D vs A (n=12); the verdict
                         Δ_B - Δ_C (mean +/- 95% CI / paired-t p / direction
                         count); D-arm row annotated "structural control";
                         verdict row H1 / H2 / inconclusive.

P0-4/P0-5 reuse arms ship with metrics_epoch.csv that contains test_auc,
so the report recomputes the equal-argmax epoch's joint AUC exactly the
same way p0_5_report.py does (strict > semantics over the 6-cell
val_auc mean). P0-2 oracle rows are reused without recompute.

Usage:
  python -u experiments/p1_3_report.py
  python -u experiments/p1_3_report.py --self_test
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
P1_3_BASE = os.path.join(BASE, "P1_3_matched_momentum")
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")

P1_3_RAW = os.path.join(P1_3_BASE, "P1_3_raw.csv")
P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")

OUT_CELLS = os.path.join(P1_3_BASE, "P1_3_cells.csv")
OUT_SUMMARY = os.path.join(P1_3_BASE, "P1_3_summary.csv")
OUT_HEADLINE = os.path.join(P1_3_BASE, "P1_3_headline.csv")

ARMS = ("A", "B", "C", "D")
SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
DOMAINS = (0, 1, 2)
TASKS = (0, 1)
RATIO = 50

# P1-3 matched-momentum constants (mirror p4_run + p1_3_orchestrator).
GAMMA_STAR = 1.0 - 1.0 / 17.6   # ≈ 0.9431818182
BETA1_STAR = GAMMA_STAR
TAU_STAR = 17.6

# Verdict thresholds from spec.
H1_THRESHOLD = 0.002    # |Δ_B - Δ_C| < 0.002 same sign → H1
H2_THRESHOLD = 0.003    # Δ_B - Δ_C ≥ 0.003, ≥10/12 same direction → H2
H2_MIN_SAME_DIR = 10    # ≥ 10/12 paired-same-direction count

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


def run_dir(arm, setting, seed):
    return os.path.join(P1_3_BASE, arm, f"setting_{setting}", f"seed{seed}")


# -- inputs --------------------------------------------------------------

def load_p1_3_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") not in ("OK", "OK_REUSED"):
            continue
        r["seed"] = int(r["seed"])
        rows.append(r)
    return rows


def load_p0_2_oracle_rows(path):
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
    """Long metrics_epoch.csv -> epochs[ei] = {'val': [6 floats], 'test': [6]}."""
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
    return {ei: e for ei, e in by_ep.items()}


def has_test_column(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "test_auc" in [h.strip() for h in header.split(",")]


def select_epoch_equal(epochs, recorded_best_epoch):
    """Argmax over mean(val_auc[e]), STRICT > semantics (first argmax).
    Mirrors p0_5_report.py::select_epoch_equal."""
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

def build_cells(p1_3_rows, p0_2_oracle_rows, missing_cells):
    """Yield dict rows for the long cells CSV.

    Schema: arm, setting, domain, task, seed, role, selected_epoch,
            joint_auc, oracle_auc, delta_transfer, mismatch.
    """
    oracle_idx = {}
    for r in p0_2_oracle_rows:
        key = (r["setting"], r["method_or_domain"], int(r["seed"]))
        oracle_idx[key] = r

    out_rows = []
    for r in p1_3_rows:
        arm, setting, seed = r["arm"], r["setting"], int(r["seed"])
        me_path = os.path.join(run_dir(arm, setting, seed), "metrics_epoch.csv")
        epochs = load_metrics_epoch(me_path)
        recorded_best = r.get("best_epoch", "")
        recorded_best_int = int(recorded_best) if recorded_best != "" else None
        test_col = has_test_column(me_path)

        per_dt_joint = {}
        sel_ep = None
        mismatch = 0
        if epochs:
            sel_ep, mismatch = select_epoch_equal(epochs, recorded_best_int)
            if test_col and sel_ep is not None and sel_ep in epochs:
                test_list = epochs[sel_ep]["test"]
                for t in TASKS:
                    for d in DOMAINS:
                        v = test_list[t * 3 + d]
                        if v is not None:
                            per_dt_joint[(d, t)] = v
            if not per_dt_joint:
                # Fallback: raw per_dt at recorded best_epoch
                vals = parse_per_dt(r.get("per_dt_test_auc", ""))
                for t in TASKS:
                    for d in DOMAINS:
                        per_dt_joint[(d, t)] = (
                            vals[t * 3 + d] if t * 3 + d < len(vals) else None)

        for d in DOMAINS:
            for t in TASKS:
                joint_v = per_dt_joint.get((d, t))
                if joint_v is None:
                    missing_cells.append((arm, setting, seed, d, t))
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
                    "arm": arm,
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
    fields = ["arm", "setting", "domain", "task", "seed",
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


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var ** 0.5)


def _se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return float(var ** 0.5 / (n ** 0.5))


def build_summary(cells):
    """Per arm: per-role mean / std delta_transfer, overall AUC, worst-
    domain AUC, mean selected epoch."""
    by_arm = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)

    summary_rows = []
    for arm in ARMS:
        cs = by_arm.get(arm, [])
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

        sel_eps = [int(c["selected_epoch"]) for c in cs
                   if c["selected_epoch"] != ""]
        mean_sel_epoch = _mean(sel_eps) if sel_eps else float("nan")
        mismatches = sum(int(c["mismatch"]) for c in cs)

        row = {
            "arm": arm,
            "n_cells": len(cs),
            "delta_L_mean": (f"{_mean(role_data.get('L', [])):.6f}"
                              if role_data.get('L') else ""),
            "delta_L_std": (f"{_std(role_data.get('L', [])):.6f}"
                             if role_data.get('L') else ""),
            "delta_M_mean": (f"{_mean(role_data.get('M', [])):.6f}"
                              if role_data.get('M') else ""),
            "delta_M_std": (f"{_std(role_data.get('M', [])):.6f}"
                             if role_data.get('M') else ""),
            "delta_S_mean": (f"{_mean(role_data.get('S', [])):.6f}"
                              if role_data.get('S') else ""),
            "delta_S_std": (f"{_std(role_data.get('S', [])):.6f}"
                             if role_data.get('S') else ""),
            "delta_overall_mean": f"{_mean(deltas_v):.6f}" if deltas_v else "",
            "overall_auc": f"{overall_auc:.6f}" if overall_auc == overall_auc else "",
            "worst_domain_auc": f"{worst_auc:.6f}" if worst_auc == worst_auc else "",
            "mean_selected_epoch": (f"{mean_sel_epoch:.2f}" if cs else ""),
            "n_mismatch": mismatches,
        }
        summary_rows.append(row)
    return summary_rows


def write_summary_csv(rows, path):
    fields = ["arm", "n_cells",
              "delta_L_mean", "delta_L_std",
              "delta_M_mean", "delta_M_std",
              "delta_S_mean", "delta_S_std",
              "delta_overall_mean",
              "overall_auc", "worst_domain_auc",
              "mean_selected_epoch", "n_mismatch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- headline ------------------------------------------------------------

def _pair_gain(p1_3_rows, accessor):
    """Compute per-(setting, seed) gain = arm X - arm A on the 6-cell
    joint AUC vector. accessor(raw_row) returns the 6-cell joint AUC
    vector. Returns gains[arm][(setting, seed)] = 6 floats (gain)."""
    by_arm = defaultdict(dict)
    for r in p1_3_rows:
        by_arm[r["arm"]][(r["setting"], int(r["seed"]))] = r

    gains = {}
    common = sorted(set(by_arm["A"].keys()) & set(by_arm["B"].keys())
                    & set(by_arm["C"].keys()) & set(by_arm["D"].keys()))
    for arm in ("B", "C", "D"):
        gains[arm] = {}
        for k in common:
            on_v = accessor(by_arm[arm][k])
            off_v = accessor(by_arm["A"][k])
            if len(on_v) != 6 or len(off_v) != 6:
                continue
            gains[arm][k] = [on_v[i] - off_v[i] for i in range(6)]
    return gains, common


def _flatten(gains, arm, common):
    """Flatten gains[arm] across (setting, seed) and (d, t) into a single
    list of scalars (paired-n12 x 6 cells)."""
    return [g for k in sorted(gains[arm].keys()) for g in gains[arm][k]]


def _paired_t_pvalue(diffs):
    """Two-sided paired-t p-value of `diffs` against 0. Returns nan if
    fewer than 2 entries."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    if var <= 0:
        # degenerate: all diffs identical. Zero mean -> no evidence (p=1);
        # non-zero constant -> t = inf -> p=0.
        return 1.0 if m == 0 else 0.0
    se = (var / n) ** 0.5
    t = m / se
    if t == 0:
        return 1.0
    # Student's t survival function via regularized incomplete beta.
    df = n - 1
    x = df / (df + t * t)
    # scipy is not guaranteed; use a Python fallback implementing the exact
    # two-sided identity p = I_{df/(df+t^2)}(df/2, 1/2).
    try:
        # I_{x}(a,b) via continued fraction (Lentz)
        def betacf(a, b, x, max_iter=200, eps=3e-7):
            qab = a + b
            qap = a + 1.0
            qam = a - 1.0
            c = 1.0
            d = 1.0 - qab * x / qap
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            h = d
            for m_ in range(1, max_iter + 1):
                m2 = 2 * m_
                aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                h *= d * c
                aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h

        a = df / 2.0
        b = 0.5
        # Two-sided p for Student's t with df: p = I_x(df/2, 1/2) with
        # x = df/(df+t^2) (exact identity). Evaluate the regularized
        # incomplete beta I_x with the Numerical Recipes symmetry branch.
        log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
        bt = math.exp(log_bt)
        if x < (a + 1.0) / (a + b + 2.0):
            ix = bt * betacf(a, b, x) / a
        else:
            ix = 1.0 - bt * betacf(b, a, 1.0 - x) / b
        return float(min(max(ix, 0.0), 1.0))
    except Exception:
        return float("nan")


def _ci95(diffs):
    """95% CI half-width for the mean of diffs (paired, n samples)."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    se = (var / n) ** 0.5
    # t critical values for 95% CI at common small df.
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201}.get(n - 1, 1.96)
    return t_crit * se


def _same_direction_count(diffs):
    return sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)


def build_headline(p1_3_rows, summary_rows):
    """Paired gains Δ_B / Δ_C / Δ_D vs A (n=12); verdict Δ_B - Δ_C and
    the H1 / H2 / inconclusive row."""
    def accessor(row):
        return parse_per_dt(row.get("per_dt_test_auc", ""))

    gains, common = _pair_gain(p1_3_rows, accessor)
    n_paired = len(common)

    def stats_for_arm(arm):
        diffs = _flatten(gains, arm, common)
        n = len(diffs)
        m = _mean(diffs) if diffs else float("nan")
        std = _std(diffs) if diffs else float("nan")
        ci = _ci95(diffs)
        return n, m, std, ci, diffs

    n_b, mean_b, std_b, ci_b, diffs_b = stats_for_arm("B")
    n_c, mean_c, std_c, ci_c, diffs_c = stats_for_arm("C")
    n_d, mean_d, std_d, ci_d, diffs_d = stats_for_arm("D")

    # Δ_B - Δ_C, paired by (setting, seed) — the spec's pairing unit (n=12).
    # Per-pair diff = mean over the 6 (d, t) cells of (B-A) - (C-A). The
    # mean over pairs equals the cell-level mean, but std/CI/paired-t and
    # the same-direction count must use the 12 independent pairs: counting
    # 72 cells would be pseudoreplication and make ">=10/12" nearly vacuous.
    pair_keys_bc = sorted(set(gains["B"].keys()) & set(gains["C"].keys()))
    diffs_bc = [sum(gains["B"][k][idx] - gains["C"][k][idx] for idx in range(6)) / 6.0
                for k in pair_keys_bc]
    mean_bc = _mean(diffs_bc)
    std_bc = _std(diffs_bc)
    ci_bc = _ci95(diffs_bc)
    p_bc = _paired_t_pvalue(diffs_bc)
    pos_bc, neg_bc = _same_direction_count(diffs_bc)
    n_bc_same_dir = max(pos_bc, neg_bc)
    sign_bc = "+" if pos_bc >= neg_bc else "-"

    # Verdict (spec): H1 if |Δ_B - Δ_C| < 0.002 with Δ_B and Δ_C of the same
    # sign; H2 if Δ_B - Δ_C >= 0.003 with >= 10/12 pairs in the direction of
    # the mean effect (B > C); otherwise inconclusive.
    same_sign_gains = (mean_b >= 0) == (mean_c >= 0)
    if abs(mean_bc) < H1_THRESHOLD and same_sign_gains:
        verdict = "H1"
    elif mean_bc >= H2_THRESHOLD and pos_bc >= H2_MIN_SAME_DIR:
        verdict = "H2"
    else:
        verdict = "inconclusive"

    # Per-arm gain summary rows.
    by_arm_summary = {r["arm"]: r for r in summary_rows}
    rows = []
    for arm in ("A", "B", "C", "D"):
        s = by_arm_summary.get(arm, {})
        rows.append({
            "arm": arm,
            "overall_auc": s.get("overall_auc", ""),
            "worst_domain_auc": s.get("worst_domain_auc", ""),
            "delta_L_mean": s.get("delta_L_mean", ""),
            "delta_M_mean": s.get("delta_M_mean", ""),
            "delta_S_mean": s.get("delta_S_mean", ""),
            "n_cells": s.get("n_cells", ""),
        })

    # Paired-gain decomposition (mean +/- std / 95% CI / n).
    gain_rows = []
    for arm, n_, m_, std_, ci_ in (("B", n_b, mean_b, std_b, ci_b),
                                   ("C", n_c, mean_c, std_c, ci_c),
                                   ("D", n_d, mean_d, std_d, ci_d)):
        row = {
            "scope": f"paired_gain_{arm}_minus_A",
            "n_paired": n_paired,
            "n_obs": n_,
            "gain_mean": f"{m_:.6f}",
            "gain_std": f"{std_:.6f}",
            "gain_ci95": f"{ci_:.6f}",
        }
        gain_rows.append(row)

    # Verdict row: Δ_B - Δ_C.
    verdict_row = {
        "scope": "verdict_BC_minus_BA_minus_C_minus_A",
        "n_paired": n_paired,
        "n_obs": len(diffs_bc),
        "diff_mean": f"{mean_bc:.6f}",
        "diff_std": f"{std_bc:.6f}",
        "diff_ci95": f"{ci_bc:.6f}",
        "paired_t_p": f"{p_bc:.6f}" if p_bc == p_bc else "",
        "pos_count": pos_bc,
        "neg_count": neg_bc,
        "n_same_dir": n_bc_same_dir,
        "dominant_sign": sign_bc,
        "verdict": verdict,
        "h1_threshold": H1_THRESHOLD,
        "h2_threshold": H2_THRESHOLD,
        "h2_min_same_dir": H2_MIN_SAME_DIR,
        "d_arm_role": "structural control (does not participate in verdict)",
    }

    # Annotate D row.
    d_annot = dict(rows[3])
    d_annot["note"] = "structural control (does not participate in H1/H2 verdict)"
    rows[3] = d_annot

    write_headline_csv(rows, gain_rows, verdict_row, OUT_HEADLINE)


def write_headline_csv(rows, gain_rows, verdict_row, path):
    fields_rows = ["arm", "overall_auc", "worst_domain_auc",
                   "delta_L_mean", "delta_M_mean", "delta_S_mean",
                   "n_cells", "note"]
    fields_gain = ["scope", "n_paired", "n_obs", "gain_mean", "gain_std", "gain_ci95"]
    fields_verdict = ["scope", "n_paired", "n_obs", "diff_mean", "diff_std",
                      "diff_ci95", "paired_t_p", "pos_count", "neg_count",
                      "n_same_dir", "dominant_sign", "verdict",
                      "h1_threshold", "h2_threshold", "h2_min_same_dir",
                      "d_arm_role"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_rows)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Sidecar verdict+gain file: write the paired-gain decomposition and
    # the verdict row. The main headline is the 4-arm table above.
    details_path = os.path.join(P1_3_BASE, "P1_3_headline_details.csv")
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_gain)
        w.writeheader()
        for r in gain_rows:
            w.writerow(r)
        w = csv.DictWriter(f, fieldnames=fields_verdict)
        w.writeheader()
        w.writerow(verdict_row)


# -- self-test -----------------------------------------------------------

def self_test():
    import tempfile
    print("=== self-test (synthetic) ===", flush=True)
    tmp = tempfile.mkdtemp(prefix="p1_3_self_")

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

    # Synthetic per_dt_test_auc generator:
    #   A (baseline):          base = 0.70
    #   B (comp1 domain):      base + 0.005 (the residual H1/H2 gain)
    #   C (comp1 uniform γ*):  base + 0.003 (smaller gain, H1 verdict)
    #   D (Adam β1*):          base + 0.001 (structural control)
    def synth_per_dt(arm, setting, seed, base=0.70):
        delta = {"A": 0.000, "B": 0.005, "C": 0.003, "D": 0.001}[arm]
        return [base + delta + 0.001 * d + 0.0005 * t
                for t in TASKS for d in DOMAINS]

    def write_me(arm, setting, seed, per_dt):
        d_root = os.path.join(tmp, "p13", arm,
                              f"setting_{setting}", f"seed{seed}")
        os.makedirs(d_root, exist_ok=True)
        ep_fields = ["experiment_stage", "experiment_name", "model", "method",
                     "setting", "seed", "epoch", "domain", "task",
                     "train_loss", "val_loss", "val_auc", "test_auc"]
        # 5 epochs; val_auc peaks at epoch 2 (strict >).
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
                            "experiment_stage": "P1-3",
                            "experiment_name": "p1_3",
                            "model": "MDMTRec",
                            "method": arm,
                            "setting": setting, "seed": seed,
                            "epoch": ei, "domain": d, "task": t,
                            "train_loss": "1.0", "val_loss": "0.5",
                            "val_auc": f"{vs[idx]:.6f}",
                            "test_auc": f"{ts[idx]:.6f}",
                        })

    p1_3_rows = []
    for arm in ARMS:
        for setting in SETTINGS:
            for seed in SEEDS:
                per_dt = synth_per_dt(arm, setting, seed)
                p1_3_rows.append({
                    "arm": arm, "ratio": 50, "setting": setting,
                    "loss_weighting": "sample",
                    "method_or_domain": ("comp1" if arm in ("B", "C") else "baseline"),
                    "seed": seed, "status": "OK",
                    "best_epoch": "2", "stopped_epoch": "4",
                    "per_dt_test_auc": ",".join(f"np.float64({v:.6f})" for v in per_dt),
                })
                write_me(arm, setting, seed, per_dt)

    # Redirect run_dir and OUT_* into the synthetic tmp.
    orig_run_dir = globals()["run_dir"]
    orig_out_cells = globals()["OUT_CELLS"]
    orig_out_summary = globals()["OUT_SUMMARY"]
    orig_out_headline = globals()["OUT_HEADLINE"]
    orig_p1_3_base = globals()["P1_3_BASE"]

    def p13_run_dir(arm, setting, seed):
        return os.path.join(tmp, "p13", arm, f"setting_{setting}", f"seed{seed}")

    globals()["run_dir"] = p13_run_dir
    globals()["OUT_CELLS"] = os.path.join(tmp, "cells.csv")
    globals()["OUT_SUMMARY"] = os.path.join(tmp, "summary.csv")
    globals()["OUT_HEADLINE"] = os.path.join(tmp, "headline.csv")
    globals()["P1_3_BASE"] = tmp
    missing = []
    patched_cells = patched_summary = patched_headline = None
    try:
        cells = build_cells(p1_3_rows, p0_2_oracle, missing)
        patched_cells = OUT_CELLS
        write_cells_csv(cells, patched_cells)
        summary = build_summary(cells)
        patched_summary = OUT_SUMMARY
        write_summary_csv(summary, patched_summary)
        patched_headline = OUT_HEADLINE
        build_headline(p1_3_rows, summary)
    finally:
        globals()["run_dir"] = orig_run_dir
        globals()["OUT_CELLS"] = orig_out_cells
        globals()["OUT_SUMMARY"] = orig_out_summary
        globals()["OUT_HEADLINE"] = orig_out_headline
        globals()["P1_3_BASE"] = orig_p1_3_base

    assert not missing, f"unexpected missing cells: {missing}"
    # Expected cell count: 4 arms x 3 settings x 4 seeds x 3 domains x 2 tasks
    # = 288.
    assert len(cells) == 288, f"expected 288 cells, got {len(cells)}"
    for c in cells:
        assert c["joint_auc"], f"missing joint_auc: {c}"
        assert c["delta_transfer"], f"missing delta_transfer: {c}"
        assert c["role"] in ("L", "M", "S")
    sel_eps = set(int(c["selected_epoch"]) for c in cells)
    assert sel_eps == {2}, f"expected all selected_epoch=2, got {sel_eps}"
    mismatches = sum(int(c["mismatch"]) for c in cells)
    assert mismatches == 0, f"unexpected mismatches: {mismatches}"

    # Per-arm gain synthetic: B = 0.005, C = 0.003, D = 0.001. Δ_B - Δ_C ≈ 0.002
    # per (setting, seed) pair (n=12), which is right at the H1 threshold;
    # since abs < H1_THRESHOLD=0.002 would fail (mean ≈ 0.002 = H1_THRESHOLD),
    # and >= H2_THRESHOLD=0.003 also fails, the verdict should be
    # "inconclusive".
    bc_diffs = [0.005 - 0.003] * 12   # one value per (setting, seed) pair
    assert abs(_mean(bc_diffs) - 0.002) < 1e-9
    # Check that the headline details CSV records this.
    details_path = os.path.join(tmp, "p1_3_self_xxxx", "P1_3_headline_details.csv")
    # OUT_HEADLINE is in tmp/; we wrote P1_3_headline_details.csv under
    # the patched P1_3_BASE (= tmp).
    details_actual = os.path.join(tmp, "P1_3_headline_details.csv")
    assert os.path.exists(details_actual), (
        f"expected headline_details at {details_actual}")
    with open(details_actual, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Last line is the verdict row.
    last = lines[-1]
    assert "verdict" in last, last
    assert "inconclusive" in last, last

    # Round-trip write_csv then read back to verify schema.
    for f in (patched_cells, patched_summary, patched_headline):
        with open(f, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        assert len(lines) >= 2, f"{f} has only {len(lines)} lines"

    print(f"  cells={len(cells)} summary={len(summary)} headline=4 "
          f"+ paired-gain + verdict rows", flush=True)
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
    os.makedirs(P1_3_BASE, exist_ok=True)
    p1_3_rows = load_p1_3_rows(P1_3_RAW)
    p0_2_oracle_rows = load_p0_2_oracle_rows(P0_2_RAW)
    print(f"loaded {len(p1_3_rows)} P1-3 rows from {P1_3_RAW}", flush=True)
    print(f"loaded {len(p0_2_oracle_rows)} P0-2 ratio_50 oracle rows from "
          f"{P0_2_RAW}", flush=True)
    missing = []
    cells = build_cells(p1_3_rows, p0_2_oracle_rows, missing)
    if missing:
        print(f"[WARN] {len(missing)} cells had no test AUC; listing first 10:",
              flush=True)
        for m in missing[:10]:
            print(f"    {m}", flush=True)
    write_cells_csv(cells, OUT_CELLS)
    print(f"wrote {OUT_CELLS} ({len(cells)} rows)", flush=True)
    summary = build_summary(cells)
    write_summary_csv(summary, OUT_SUMMARY)
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)", flush=True)
    build_headline(p1_3_rows, summary)
    print(f"wrote {OUT_HEADLINE}", flush=True)


if __name__ == "__main__":
    main()
