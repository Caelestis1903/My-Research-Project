"""P1-2 report: read-CSV aggregator over the 6-arm curriculum sweep.

Six arms (curriculum x comp1, all under --loss_weighting sample):
  A0: curriculum none, comp1 off   <- reused from P0-4 w_sample/baseline
  B0: curriculum domain_cond kappa0=0.5, comp1 off
  C0: curriculum domain_cond kappa0=-0.5, comp1 off
  A1: curriculum none, comp1 on    <- reused from P0-5 w_sample/comp1
  B1: curriculum domain_cond kappa0=0.5, comp1 on
  C1: curriculum domain_cond kappa0=-0.5, comp1 on

Inputs:
  experiments/P1_2_curriculum/P1_2_raw.csv               (master rows)
  experiments/P1_2_curriculum/{arm}/setting_<X>/seed<s>/metrics_epoch.csv
  experiments/P1_2_curriculum/{arm}/setting_<X>/seed<s>/grad_norm_epoch.csv
                                                          (may be missing
                                                          for A0/A1 when
                                                          --include_A_grad_norm=0)
  experiments/P0_2_ratio_sweep/P0_2_raw.csv               (for ratio_50 oracle,
                                                          Δ_transfer pairing)

Outputs (all under experiments/P1_2_curriculum/):
  P1_2_paired.csv        per (setting, seed) overall/worst-domain test AUC
                         paired diffs: B0-A0, C0-A0, C0-B0, B1-A1, C1-A1,
                         (B1-A1)-(B0-A0). 12 cells per pair, per-pair
                         median/IQR/sign-agreement rate.
  P1_2_per_domain.csv    per (arm, setting, domain-role, epoch) mean+/-min-max
                         of val AUC across 4 seeds, plus the train loss mean.
  P1_2_peak_epoch.csv    per-run per-domain argmax epoch (val AUC), per
                         (arm, role) distribution + paired diffs.
  P1_2_grad_norm.csv     shared + per-domain L2 norms across epochs (mean
                         and band), with a half-vs-half summary that
                         compares the rho<1 phase to rho=1.
  P1_2_rho_sanity.csv    one row per arm: rho schedule min/median/max and
                         the analytic kappa_d range, so the reader can
                         verify the B/C arms really did offset the small
                         domains and the A arms stayed pinned at rho=1.

Usage:
  python -u experiments/p1_2_report.py
  python -u experiments/p1_2_report.py --self_test
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
P1_2_BASE = os.path.join(BASE, "P1_2_curriculum")
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")

P1_2_RAW = os.path.join(P1_2_BASE, "P1_2_raw.csv")
P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")

OUT_PAIRED = os.path.join(P1_2_BASE, "P1_2_paired.csv")
OUT_PER_DOMAIN = os.path.join(P1_2_BASE, "P1_2_per_domain.csv")
OUT_PEAK = os.path.join(P1_2_BASE, "P1_2_peak_epoch.csv")
OUT_GRAD = os.path.join(P1_2_BASE, "P1_2_grad_norm.csv")
OUT_RHO = os.path.join(P1_2_BASE, "P1_2_rho_sanity.csv")

ARMS = ("A0", "B0", "C0", "A1", "B1", "C1")
SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
DOMAINS = (0, 1, 2)
TASKS = (0, 1)
RATIO = 50

ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}

FLOAT_RE = re.compile(r"np\.float64\(([-\d.eE+]+)\)")

PAIRS = (
    ("B0_minus_A0", "B0", "A0"),
    ("C0_minus_A0", "C0", "A0"),
    ("C0_minus_B0", "C0", "B0"),
    ("B1_minus_A1", "B1", "A1"),
    ("C1_minus_A1", "C1", "A1"),
    ("(B1_minus_A1)_minus_(B0_minus_A0)", "B1_minus_A1", "B0_minus_A0"),
)


def parse_per_dt(s):
    if not s:
        return []
    return [float(m.group(1)) for m in FLOAT_RE.finditer(s)]


def run_dir(arm, setting, seed):
    return os.path.join(P1_2_BASE, arm, f"setting_{setting}", f"seed{seed}")


def load_metrics_epoch(path):
    """Long metrics_epoch.csv -> epochs[e] = {'val': [6 floats], 'test': [6]}.
    Task-major idx = t*D + d (D=3)."""
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


def load_grad_norm(path):
    """Long grad_norm_epoch.csv -> rows[domain][epoch] = (grad_norm, rho).
    domain in {shared, d0, d1, d2}."""
    if not os.path.exists(path):
        return None
    rows = defaultdict(dict)
    for r in csv.DictReader(open(path, newline="")):
        dom = r["domain"]
        ep = int(r["epoch"])
        gn = float(r["grad_norm"])
        rho = float(r["curriculum_rho"])
        rows[dom][ep] = (gn, rho)
    return rows


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------

def load_p1_2_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path, newline="")):
        if r.get("status") not in ("OK", "OK_REUSED"):
            continue
        r["seed"] = int(r["seed"])
        r["kappa0"] = float(r["kappa0"])
        rows.append(r)
    return rows


def load_p0_2_oracle_rows(path):
    """P0-2 ratio=50 oracle rows for Δ_transfer pairing (re-used, not re-run)."""
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


def expand_per_dt_joint(row):
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 6:
        return []
    return [(d, t, vals[t * 3 + d]) for t in TASKS for d in DOMAINS]


def expand_per_dt_oracle(row):
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 2:
        return []
    d = int(row["method_or_domain"].replace("oracle_d", ""))
    return [(d, 0, vals[0]), (d, 1, vals[1])]


# ----------------------------------------------------------------------------
# output 1: paired comparison
# ----------------------------------------------------------------------------

def per_run_metrics(rows, oracle_idx):
    """For each (arm, setting, seed), compute overall test AUC (mean of 6
    per-(d,t) cells) and worst-domain test AUC (min over d of mean over t).
    Test AUC source: prefer recomputed equal-argmax test AUC from
    metrics_epoch.csv; fall back to per_dt_test_auc at recorded best_epoch
    (matches the P0-5 fallback). Δ_transfer is joint - oracle at the
    selected epoch (cell-level, used by P0-4 report-style summaries)."""
    out = {}
    for r in rows:
        arm, setting, seed = r["arm"], r["setting"], int(r["seed"])
        path = os.path.join(run_dir(arm, setting, seed), "metrics_epoch.csv")
        epochs = load_metrics_epoch(path)
        if epochs is None:
            # No metrics file -> skip. Reused A arms should always have one.
            continue
        # recompute equal argmax over val AUC (mean of 6 cells), strict >
        best_e = None
        best_score = -1.0
        for ei in sorted(epochs.keys()):
            vs = epochs[ei]["val"]
            if any(v is None for v in vs):
                continue
            s = sum(vs) / len(vs)
            if s > best_score:
                best_score, best_e = s, ei
        if best_e is None:
            continue
        test_at_best = epochs[best_e]["test"]
        if any(v is None for v in test_at_best):
            # fall back to per_dt_test_auc at recorded best_epoch
            recorded_best = r.get("best_epoch", "")
            if recorded_best == "":
                continue
            try:
                rb = int(recorded_best)
            except ValueError:
                continue
            test_at_best = epochs[rb]["test"] if rb in epochs else test_at_best
            if any(v is None for v in test_at_best):
                continue
        # overall and worst-domain
        cell_means = [(test_at_best[t * 3 + d]) for d in DOMAINS for t in TASKS]
        per_domain = [np.mean([test_at_best[t * 3 + d] for t in TASKS]) for d in DOMAINS]
        overall = float(np.mean(cell_means))
        worst = float(np.min(per_domain))
        # delta_transfer per (d, t) vs P0-2 ratio_50 oracle
        deltas = {}
        for d in DOMAINS:
            for t in TASKS:
                idx = t * 3 + d
                oracle_row = oracle_idx.get((setting, f"oracle_d{d}", seed))
                if oracle_row is None:
                    deltas[(d, t)] = None
                    continue
                ov = parse_per_dt(oracle_row.get("per_dt_test_auc", ""))
                if len(ov) != 2:
                    deltas[(d, t)] = None
                else:
                    deltas[(d, t)] = test_at_best[idx] - ov[t]
        out[(arm, setting, seed)] = {
            "overall": overall, "worst_domain": worst,
            "test_per_cell": list(test_at_best),
            "delta_per_cell": deltas,
            "selected_epoch": best_e,
        }
    return out


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(median="", iqr_lo="", iqr_hi="", sign_rate="", n=0)
    arr = np.array(vals, dtype=np.float64)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    sign_rate = float(np.mean([1 if v > 0 else (0.5 if v == 0 else 0) for v in vals]))
    return dict(median=f"{med:.6f}", iqr_lo=f"{q1:.6f}", iqr_hi=f"{q3:.6f}",
                sign_rate=f"{sign_rate:.4f}", n=len(vals))


def build_paired(metrics):
    rows = []
    for pair_name, a, b in PAIRS:
        # Second-order interaction: (B1-A1) - (B0-A0). Detect by checking
        # the canonical interaction pair_name; the simple-pair check below
        # would misfire on this name because "B1_minus_A1" and "B0_minus_A0"
        # are both substrings of it.
        if pair_name == "(B1_minus_A1)_minus_(B0_minus_A0)":
            for setting in SETTINGS:
                for seed in SEEDS:
                    b1a1 = (metrics.get(("B1", setting, seed)),
                            metrics.get(("A1", setting, seed)))
                    b0a0 = (metrics.get(("B0", setting, seed)),
                            metrics.get(("A0", setting, seed)))
                    if None in (b1a1 + b0a0):
                        continue
                    do_b1a1 = b1a1[0]["overall"] - b1a1[1]["overall"]
                    do_b0a0 = b0a0[0]["overall"] - b0a0[1]["overall"]
                    dw_b1a1 = b1a1[0]["worst_domain"] - b1a1[1]["worst_domain"]
                    dw_b0a0 = b0a0[0]["worst_domain"] - b0a0[1]["worst_domain"]
                    rows.append({
                        "pair": pair_name, "setting": setting, "seed": seed,
                        "delta_overall": f"{do_b1a1 - do_b0a0:.6f}",
                        "delta_worst": f"{dw_b1a1 - dw_b0a0:.6f}",
                    })
            continue
        # All other pairs are simple (a - b) where a, b are arm names.
        for setting in SETTINGS:
            for seed in SEEDS:
                ka = metrics.get((a, setting, seed))
                kb = metrics.get((b, setting, seed))
                if ka is None or kb is None:
                    continue
                rows.append({
                    "pair": pair_name, "setting": setting, "seed": seed,
                    "delta_overall": f"{ka['overall'] - kb['overall']:.6f}",
                    "delta_worst": f"{ka['worst_domain'] - kb['worst_domain']:.6f}",
                })
    # append per-pair stats rows
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    for pair_name in sorted({r["pair"] for r in rows}):
        s_overall = _stats([float(r["delta_overall"]) for r in by_pair[pair_name]])
        s_worst = _stats([float(r["delta_worst"]) for r in by_pair[pair_name]])
        rows.append({
            "pair": pair_name, "setting": "median", "seed": "",
            "delta_overall": s_overall["median"],
            "delta_worst": s_worst["median"],
        })
        rows.append({
            "pair": pair_name, "setting": "iqr", "seed": "",
            "delta_overall": f"[{s_overall['iqr_lo']}, {s_overall['iqr_hi']}]",
            "delta_worst": f"[{s_worst['iqr_lo']}, {s_worst['iqr_hi']}]",
        })
        rows.append({
            "pair": pair_name, "setting": "sign", "seed": "",
            "delta_overall": s_overall["sign_rate"],
            "delta_worst": s_worst["sign_rate"],
        })
        rows.append({
            "pair": pair_name, "setting": "n", "seed": "",
            "delta_overall": str(s_overall["n"]),
            "delta_worst": str(s_worst["n"]),
        })
    return rows


def write_paired_csv(rows, path):
    fields = ["pair", "setting", "seed", "delta_overall", "delta_worst"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# output 2: per-domain val AUC trajectory
# ----------------------------------------------------------------------------

def build_per_domain(rows):
    """Per (arm, setting, role, epoch): mean +/- min-max of val AUC across
    seeds, plus the train_loss mean. role = L/M/S from the setting's
    rotation. Only d0/d1/d2 are valid (no oracle)."""
    per_arm_setting_role_epoch = defaultdict(list)
    train_loss_per_arm_setting_epoch = defaultdict(list)
    for r in rows:
        arm, setting, seed = r["arm"], r["setting"], int(r["seed"])
        path = os.path.join(run_dir(arm, setting, seed), "metrics_epoch.csv")
        epochs = load_metrics_epoch(path)
        if epochs is None:
            continue
        for ei, e in epochs.items():
            for d in DOMAINS:
                for t in TASKS:
                    idx = t * 3 + d
                    v = e["val"][idx]
                    if v is None:
                        continue
                    role = ROLE_MAPS[setting][d]
                    per_arm_setting_role_epoch[(arm, setting, role, ei)].append(v)
            tl_path = os.path.join(run_dir(arm, setting, seed), "metrics_epoch.csv")
            # train_loss lives only in metrics_epoch as a string (per-row),
            # duplicated across (d, t) rows; we average over those duplicates.
            tloss_vals = []
            for row in csv.DictReader(open(tl_path, newline="")):
                if int(row["epoch"]) == ei and row["domain"] == "0" and row["task"] == "0":
                    try:
                        tloss_vals.append(float(row["train_loss"]))
                    except (TypeError, ValueError):
                        pass
            if tloss_vals:
                train_loss_per_arm_setting_epoch[(arm, setting, ei)].append(
                    float(np.mean(tloss_vals)))
    # build long output rows
    out_rows = []
    for (arm, setting, role, ei), vals in per_arm_setting_role_epoch.items():
        arr = np.array(vals, dtype=np.float64)
        mean = float(arr.mean())
        mn = float(arr.min())
        mx = float(arr.max())
        n = int(arr.size)
        out_rows.append({
            "arm": arm, "setting": setting, "role": role,
            "epoch": ei, "metric": "val_auc",
            "mean": f"{mean:.6f}", "min": f"{mn:.6f}", "max": f"{mx:.6f}",
            "n_seeds": n,
        })
    for (arm, setting, ei), vals in train_loss_per_arm_setting_epoch.items():
        arr = np.array(vals, dtype=np.float64)
        out_rows.append({
            "arm": arm, "setting": setting, "role": "ALL",
            "epoch": ei, "metric": "train_loss",
            "mean": f"{arr.mean():.6f}", "min": f"{arr.min():.6f}",
            "max": f"{arr.max():.6f}", "n_seeds": int(arr.size),
        })
    return out_rows


def write_per_domain_csv(rows, path):
    fields = ["arm", "setting", "role", "epoch", "metric", "mean", "min", "max", "n_seeds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# output 3: peak epoch
# ----------------------------------------------------------------------------

def build_peak(rows):
    """Per run, per (domain-role): argmax epoch over val AUC. Per (arm, role):
    mean / median / min / max over the 12 runs (3 settings x 4 seeds), and
    paired diffs for B-A / C-A / C-B within each comp1 setting."""
    per_arm_role = defaultdict(list)  # (arm, role) -> list of peak epochs
    per_cell = {}  # (arm, role, setting, seed) -> peak epoch (paired key)
    per_run_peak = []
    for r in rows:
        arm, setting, seed = r["arm"], r["setting"], int(r["seed"])
        path = os.path.join(run_dir(arm, setting, seed), "metrics_epoch.csv")
        epochs = load_metrics_epoch(path)
        if epochs is None:
            continue
        for d in DOMAINS:
            role = ROLE_MAPS[setting][d]
            best_e, best_v = None, -1.0
            for ei in sorted(epochs.keys()):
                v = epochs[ei]["val"][d]  # use task-0 val AUC as the per-domain signal
                if v is None:
                    continue
                if v > best_v:
                    best_v, best_e = v, ei
            if best_e is None:
                continue
            per_arm_role[(arm, role)].append(best_e)
            per_cell[(arm, role, setting, seed)] = best_e
            per_run_peak.append({
                "arm": arm, "setting": setting, "seed": seed, "domain": d,
                "role": role, "peak_epoch": best_e,
            })
    out_rows = list(per_run_peak)
    # Per (arm, role) summary
    for (arm, role), vals in per_arm_role.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.int64)
        out_rows.append({
            "arm": arm, "setting": "summary", "seed": "",
            "domain": "", "role": role,
            "peak_epoch": (f"mean={arr.mean():.2f} median={float(np.median(arr))} "
                           f"min={arr.min()} max={arr.max()}"),
        })
    # Paired diffs per (role): B0-A0, C0-A0, C0-B0, B1-A1, C1-A1, C1-B1.
    # Paired on (setting, seed) via per_cell — the raw CSV is ordered by
    # completion time (CONCURRENCY=2), so positional zip would mis-pair.
    paired_specs = (
        ("B0_minus_A0", "B0", "A0"),
        ("C0_minus_A0", "C0", "A0"),
        ("C0_minus_B0", "C0", "B0"),
        ("B1_minus_A1", "B1", "A1"),
        ("C1_minus_A1", "C1", "A1"),
        ("C1_minus_B1", "C1", "B1"),
    )
    for pair_name, a, b in paired_specs:
        for role in ("L", "M", "S"):
            diffs = []
            for setting in SETTINGS:
                for seed in SEEDS:
                    va = per_cell.get((a, role, setting, seed))
                    vb = per_cell.get((b, role, setting, seed))
                    if va is None or vb is None:
                        continue
                    diffs.append(va - vb)
            if not diffs:
                continue
            diff = np.array(diffs, dtype=np.float64)
            out_rows.append({
                "arm": pair_name, "setting": "peak_diff", "seed": "",
                "domain": "", "role": role,
                "peak_epoch": (f"mean_diff={diff.mean():.2f} "
                               f"median_diff={float(np.median(diff))} n={len(diffs)}"),
            })
    return out_rows


def write_peak_csv(rows, path):
    fields = ["arm", "setting", "seed", "domain", "role", "peak_epoch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# output 4: grad norm trajectory
# ----------------------------------------------------------------------------

def build_grad_norm(rows):
    """Per (arm, setting, domain-label): mean +/- band across seeds and epochs.
    Plus a half-vs-half summary (epochs with rho<1 vs rho=1)."""
    out_rows = []
    per_arm_setting_dom_ep = defaultdict(list)  # (arm, setting, dom, ep) -> [(gn, rho), ...]
    for r in rows:
        arm, setting, seed = r["arm"], r["setting"], int(r["seed"])
        path = os.path.join(run_dir(arm, setting, seed), "grad_norm_epoch.csv")
        gn = load_grad_norm(path)
        if gn is None:
            continue
        for dom in ("shared", "d0", "d1", "d2"):
            for ep, (val, rho) in gn[dom].items():
                per_arm_setting_dom_ep[(arm, setting, dom, ep)].append((val, rho))
    # Per (arm, setting, dom, epoch): mean over seeds (the band)
    for (arm, setting, dom, ep), vals in per_arm_setting_dom_ep.items():
        arr = np.array([v[0] for v in vals], dtype=np.float64)
        rhos = np.array([v[1] for v in vals], dtype=np.float64)
        out_rows.append({
            "arm": arm, "setting": setting, "domain": dom, "epoch": ep,
            "phase": "rho<1" if float(rhos.mean()) < 0.999 else "rho=1",
            "mean_grad_norm": f"{arr.mean():.6e}",
            "min_grad_norm": f"{arr.min():.6e}",
            "max_grad_norm": f"{arr.max():.6e}",
            "mean_rho": f"{rhos.mean():.6f}",
            "n_seeds": int(arr.size),
        })
    # Per (arm, setting, dom): half-vs-half summary
    by_asm_d = defaultdict(list)
    for r in out_rows:
        if r["setting"] in ("d0_large", "d1_large", "d2_large") and isinstance(r["epoch"], int):
            by_asm_d[(r["arm"], r["setting"], r["domain"], r["phase"])].append(
                float(r["mean_grad_norm"]))
    for (arm, setting, dom, phase), vals in by_asm_d.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        out_rows.append({
            "arm": arm, "setting": setting, "domain": dom, "epoch": "summary",
            "phase": phase,
            "mean_grad_norm": f"{arr.mean():.6e}",
            "min_grad_norm": f"{arr.min():.6e}",
            "max_grad_norm": f"{arr.max():.6e}",
            "mean_rho": "", "n_seeds": int(arr.size),
        })
    return out_rows


def write_grad_norm_csv(rows, path):
    fields = ["arm", "setting", "domain", "epoch", "phase",
              "mean_grad_norm", "min_grad_norm", "max_grad_norm",
              "mean_rho", "n_seeds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# output 5: rho sanity
# ----------------------------------------------------------------------------

def build_rho_sanity(rows):
    """One row per arm: rho schedule min/median/max across all settings
    seeds, plus the analytic kappa_d range at rho=0. The reader uses this
    to verify A arms are pinned at rho=1 and B/C arms really offset the
    small domains at the start of training."""
    out_rows = []
    for arm in ARMS:
        rhos = []
        for r in rows:
            if r["arm"] != arm:
                continue
            path = os.path.join(run_dir(r["arm"], r["setting"], int(r["seed"])),
                                "grad_norm_epoch.csv")
            gn = load_grad_norm(path)
            if gn is None:
                continue
            for dom in gn.values():
                for _, rho in dom.values():
                    rhos.append(rho)
        # analytic kappa_d range at rho=0 (T_c = total/2, rho(0)=0)
        kappa_d_range = "n/a"
        if arm in ("B0", "B1", "C0", "C1"):
            k0 = 0.5 if arm in ("B0", "B1") else -0.5
            # Use d0_large domain_p=(0.8, 0.184, 0.016); max:p_min ratio same
            # across settings (just rotated), so log_ratio values are identical.
            pmax, pmin = 0.8, 0.016
            kappa_d_pmin = k0 * (np.log(pmax / pmin) / np.log(pmax / pmin))
            kappa_d_pmax = k0 * (np.log(pmax / pmax) / np.log(pmax / pmin))
            kappa_d_range = (f"kappa0={k0} kappa_d[pmax]={kappa_d_pmax:.3f} "
                             f"kappa_d[pmin]={kappa_d_pmin:.3f} "
                             f"sign_at_rho0={'+' if kappa_d_pmin > 0 else '-'}")
        out_rows.append({
            "arm": arm,
            "n_rho_obs": len(rhos),
            "rho_min": f"{min(rhos):.6f}" if rhos else "",
            "rho_median": f"{float(np.median(rhos)):.6f}" if rhos else "",
            "rho_max": f"{max(rhos):.6f}" if rhos else "",
            "kappa_d_range": kappa_d_range,
        })
    return out_rows


def write_rho_sanity_csv(rows, path):
    fields = ["arm", "n_rho_obs", "rho_min", "rho_median", "rho_max", "kappa_d_range"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    """Synthesize 6 arms x 3 settings x 4 seeds of fake metrics_epoch and
    grad_norm_epoch CSVs in a tmp dir, point all helpers at that tmp, and
    verify each output's schema and at least one diagnostic number."""
    import tempfile
    print("=== self-test (synthetic) ===", flush=True)
    tmp = tempfile.mkdtemp(prefix="p1_2_self_")
    base = os.path.join(tmp, "P1_2_curriculum")
    os.makedirs(base, exist_ok=True)
    raw_path = os.path.join(base, "P1_2_raw.csv")
    raw_fields = ["arm", "setting", "seed", "kappa0", "curriculum_mode",
                  "loss_weighting", "method_or_domain", "comp1",
                  "kind", "ratio", "status", "source", "reuse_check",
                  "fit_sec", "peak_mem_mb", "wall_sec", "per_dt_test_auc",
                  "best_epoch", "stopped_epoch", "best_val_auc", "note"]
    raw_rows = []
    ep_fields = ["experiment_stage", "experiment_name", "model", "method",
                 "setting", "seed", "epoch", "domain", "task",
                 "train_loss", "val_loss", "val_auc", "test_auc"]
    gn_fields = ["epoch", "domain", "grad_norm", "curriculum_rho"]
    # Synthesize a deterministic schedule:
    #   val_auc(d, epoch) = 0.60 + 0.01*d + 0.005*epoch + role-bump
    # where the "role-bump" depends on (arm, setting, d):
    #   B arm: L gets +0.02, S gets -0.01 (curriculum penalizes small dom)
    #   C arm: L gets -0.01, S gets +0.02 (reverse)
    #   A arm: no role bump
    settings = list(SETTINGS)
    seeds = list(SEEDS)
    for arm in ARMS:
        bump = {}
        if arm.startswith("B"):
            bump = {"L": 0.02, "M": 0.0, "S": -0.01}
        elif arm.startswith("C"):
            bump = {"L": -0.01, "M": 0.0, "S": 0.02}
        for setting in settings:
            for seed in seeds:
                d = os.path.join(base, arm, f"setting_{setting}", f"seed{seed}")
                os.makedirs(d, exist_ok=True)
                ep_rows = []
                gn_rows = []
                for ep in range(5):
                    for t in TASKS:
                        for dom in DOMAINS:
                            role = ROLE_MAPS[setting][dom]
                            v = 0.60 + 0.01 * dom + 0.005 * ep + bump.get(role, 0.0)
                            ta = v + 0.001 * (t - 0.5)
                            ep_rows.append({
                                "experiment_stage": "P1-2",
                                "experiment_name": "curriculum_lod",
                                "model": "MDMTRec",
                                "method": "comp1" if arm.endswith("1") else "baseline",
                                "setting": setting, "seed": seed,
                                "epoch": ep, "domain": dom, "task": t,
                                "train_loss": "0.5",
                                "val_loss": "0.5",
                                "val_auc": f"{v:.6f}",
                                "test_auc": f"{ta:.6f}",
                            })
                    # Synthesize rho across the full range: 0, 0.5, 1, 1, 1
                    # so both rho<1 and rho=1 phases are exercised in the
                    # self-test (the all-off A arm stays at 1 throughout;
                    # B/C arms traverse the ramp).
                    if arm.startswith("A"):
                        rho = 1.0
                    else:
                        rho = min(1.0, ep / 2.0)
                    for k_dom in ("shared", "d0", "d1", "d2"):
                        gn_rows.append({
                            "epoch": ep, "domain": k_dom,
                            "grad_norm": f"{0.01 + 0.001 * ep:.6e}",
                            "curriculum_rho": f"{rho:.6f}",
                        })
                with open(os.path.join(d, "metrics_epoch.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=ep_fields)
                    w.writeheader()
                    for r in ep_rows:
                        w.writerow(r)
                with open(os.path.join(d, "grad_norm_epoch.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=gn_fields)
                    w.writeheader()
                    for r in gn_rows:
                        w.writerow(r)
                raw_rows.append({
                    "arm": arm, "setting": setting, "seed": seed,
                    "kappa0": 0.5 if arm.startswith("B") else (-0.5 if arm.startswith("C") else 0.0),
                    "curriculum_mode": ("domain_conditioned" if not arm.startswith("A")
                                        else "none"),
                    "loss_weighting": "sample",
                    "method_or_domain": ("comp1" if arm.endswith("1") else "baseline"),
                    "comp1": ("per_domain_only" if arm.endswith("1") else "none"),
                    "kind": "joint", "ratio": 50, "status": "OK",
                    "source": "fresh", "reuse_check": "",
                    "fit_sec": "100", "peak_mem_mb": "500", "wall_sec": "110",
                    "per_dt_test_auc": ",".join(
                        f"np.float64({0.65 + 0.01 * dom + 0.005 * 2 + bump.get(ROLE_MAPS[setting][dom], 0.0):.6f})"
                        for t in TASKS for dom in DOMAINS),
                    "best_epoch": "2", "stopped_epoch": "4",
                    "best_val_auc": "0.72", "note": "",
                })
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    # Run all builders against tmp dir.
    rows = load_p1_2_rows(raw_path)
    oracle_idx = {}  # no oracle -> all deltas None
    # Monkey-patch P1_2_BASE / run_dir so per_run_metrics / build_per_domain /
    # build_peak / build_grad_norm / build_rho_sanity read from tmp.
    orig_p1_2_base = P1_2_BASE
    globals()["P1_2_BASE"] = base

    def _tmp_run_dir(arm, setting, seed):
        return os.path.join(base, arm, f"setting_{setting}", f"seed{seed}")
    globals()["run_dir"] = _tmp_run_dir
    try:
        metrics = per_run_metrics(rows, oracle_idx)
        assert len(metrics) == 72, f"expected 72 (arm,setting,seed), got {len(metrics)}"
        paired = build_paired(metrics)
        assert any(r["pair"] == "B0_minus_A0" for r in paired), "missing B0_minus_A0 row"
        assert any(r["pair"] == "(B1_minus_A1)_minus_(B0_minus_A0)" for r in paired), \
            "missing interaction pair"
        # check median / iqr / sign summary rows exist
        for pair_name, _, _ in PAIRS:
            assert any(r["pair"] == pair_name and r["setting"] == "median" for r in paired), \
                f"missing median row for {pair_name}"
            assert any(r["pair"] == pair_name and r["setting"] == "sign" for r in paired), \
                f"missing sign row for {pair_name}"
        pd = build_per_domain(rows)
        assert any(r["metric"] == "val_auc" for r in pd)
        assert any(r["metric"] == "train_loss" for r in pd)
        pk = build_peak(rows)
        assert any(r["setting"] == "summary" for r in pk)
        assert any(r["arm"] == "B0_minus_A0" and r["setting"] == "peak_diff" for r in pk)
        gn = build_grad_norm(rows)
        assert any(r["phase"] == "rho<1" for r in gn)
        assert any(r["phase"] == "rho=1" for r in gn)
        rho = build_rho_sanity(rows)
        assert any(r["arm"] == "A0" and r["rho_min"] == "1.000000" for r in rho), \
            f"A0 rho should be pinned at 1, got {rho}"
        assert any(r["arm"] == "B0" and float(r["rho_min"]) < 0.5 for r in rho), \
            f"B0 rho should start < 1, got {rho}"
        # Write everything to tmp (round-trip)
        write_paired_csv(paired, os.path.join(tmp, "paired.csv"))
        write_per_domain_csv(pd, os.path.join(tmp, "per_domain.csv"))
        write_peak_csv(pk, os.path.join(tmp, "peak.csv"))
        write_grad_norm_csv(gn, os.path.join(tmp, "grad.csv"))
        write_rho_sanity_csv(rho, os.path.join(tmp, "rho.csv"))
        print(f"  metrics={len(metrics)} paired={len(paired)} per_domain={len(pd)} "
              f"peak={len(pk)} grad_norm={len(gn)} rho={len(rho)}", flush=True)
    finally:
        globals()["P1_2_BASE"] = orig_p1_2_base
    print("  self-test OK", flush=True)
    print("  self-test OK", flush=True)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true",
                    help="run synthetic round-trip and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    os.makedirs(P1_2_BASE, exist_ok=True)
    rows = load_p1_2_rows(P1_2_RAW)
    oracle_rows = load_p0_2_oracle_rows(P0_2_RAW)
    oracle_idx = {(r["setting"], r["method_or_domain"], int(r["seed"])): r
                  for r in oracle_rows}
    print(f"loaded {len(rows)} P1-2 rows from {P1_2_RAW}", flush=True)
    print(f"loaded {len(oracle_rows)} P0-2 ratio_50 oracle rows from {P0_2_RAW}",
          flush=True)
    metrics = per_run_metrics(rows, oracle_idx)
    print(f"computed per-run metrics for {len(metrics)} (arm,setting,seed)",
          flush=True)
    paired = build_paired(metrics)
    write_paired_csv(paired, OUT_PAIRED)
    print(f"wrote {OUT_PAIRED} ({len(paired)} rows)", flush=True)
    pd = build_per_domain(rows)
    write_per_domain_csv(pd, OUT_PER_DOMAIN)
    print(f"wrote {OUT_PER_DOMAIN} ({len(pd)} rows)", flush=True)
    pk = build_peak(rows)
    write_peak_csv(pk, OUT_PEAK)
    print(f"wrote {OUT_PEAK} ({len(pk)} rows)", flush=True)
    gn = build_grad_norm(rows)
    write_grad_norm_csv(gn, OUT_GRAD)
    print(f"wrote {OUT_GRAD} ({len(gn)} rows)", flush=True)
    rho = build_rho_sanity(rows)
    write_rho_sanity_csv(rho, OUT_RHO)
    print(f"wrote {OUT_RHO} ({len(rho)} rows)", flush=True)


if __name__ == "__main__":
    main()