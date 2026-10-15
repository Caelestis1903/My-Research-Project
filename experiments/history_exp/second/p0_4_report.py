"""P0-4 report: pure read-CSV aggregator over the 36 fresh P0-4 joint baseline
runs plus the 36 P0-2 ratio_50 oracle rows (referenced, NOT re-run), with
checkpoint post-hoc reselection.

Inputs:
  experiments/P0_4_loss_weighting/P0_4_raw.csv          (P0-4 fresh runs)
  experiments/P0_2_ratio_sweep/P0_2_raw.csv             (P0-2 ratio_50 oracle rows)
  <run_dir>/metrics_epoch.csv per joint run              (for checkpoint reselect)

Outputs (all under experiments/P0_4_loss_weighting/):
  P0_4_cells.csv       long: per (weighting, selection, setting, seed, domain,
                       task) cell with selected_epoch, joint_auc, oracle_auc,
                       delta_transfer, role, censored.
  P0_4_summary.csv     per (weighting, selection): per-role mean / se
                       Δ_transfer, fraction negative, overall AUC,
                       worst-domain AUC, mean selected_epoch,
                       fraction censored.
  P0_4_headline.csv    3 weightings x 2 selections, L 角色 / S 角色
                       mean Δ_transfer.

P0-4 core new logic: post-hoc checkpoint reselection. The orchestrator's
best_epoch is the argmax of mean(val_auc[e]) over epochs. We recompute it
under TWO selection rules and report per-rule Δ_transfer:

  selection='equal': score(e) = mean of 6-cell val_auc[e]. Argmax with
                     STRICT-greater (first-argmax) — must match the
                     recorded best_epoch; otherwise mark MISMATCH.
  selection='sample': score(e) = sum_d w_d_sample * val_auc[e][d],
                      with w_d_sample = p_d / p_mean and p_d the sample
                      share of the ACTUAL domain d under this setting's
                      role rotation (domain_p(setting), derived from
                      ROLE_SIZES + ROLE_MAPS — NOT the fixed d0_large
                      order, since the L/M/S roles rotate across settings).

The sample argmax landing on the last logged epoch is flagged censored=1
(EarlyStopper may have terminated without a clear winner downstream).

Usage:
  python -u experiments/p0_4_report.py
  python -u experiments/p0_4_report.py --self_test
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
P0_4_BASE = os.path.join(BASE, "P0_4_loss_weighting")
P0_2_BASE = os.path.join(BASE, "P0_2_ratio_sweep")

P0_4_RAW = os.path.join(P0_4_BASE, "P0_4_raw.csv")
P0_2_RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")

OUT_CELLS = os.path.join(P0_4_BASE, "P0_4_cells.csv")
OUT_SUMMARY = os.path.join(P0_4_BASE, "P0_4_summary.csv")
OUT_HEADLINE = os.path.join(P0_4_BASE, "P0_4_headline.csv")

SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
WEIGHTINGS = ("equal", "sample", "sqrt")
SELECTIONS = ("equal", "sample")
DOMAINS = (0, 1, 2)
TASKS = (0, 1)
RATIO = 50
ROLE_SIZES = "150000,34500,3000"

# ROLE_MAPS: per setting -> {domain: role}. Identical to p0_2_report.py.
ROLE_MAPS = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
}

# Per-ROLE sample share from ROLE_SIZES (150000/187500=0.8, 34500/187500=
# 0.184, 3000/187500=0.016). The per-DOMAIN share rotates with the setting
# (see ROLE_MAPS), so sample-weighted selection must resolve p through the
# setting's role map — using the fixed d0_large order for d1_large/d2_large
# would attach the L weight to the wrong domain.
_SZ = [float(x) for x in ROLE_SIZES.split(",")]
_TOT = sum(_SZ)
ROLE_P = {"L": _SZ[0] / _TOT, "M": _SZ[1] / _TOT, "S": _SZ[2] / _TOT}


def domain_p(setting):
    """Per-domain sample share tuple (p_0, p_1, p_2) under this setting's
    role rotation. For d0_large this is (0.8, 0.184, 0.016)."""
    return tuple(ROLE_P[ROLE_MAPS[setting][d]] for d in DOMAINS)

# per_dt_test_auc list layout (joint run, 6 entries): task-major
# [d0t0, d1t0, d2t0, d0t1, d1t1, d2t1]. So idx = t * D + d. Same as
# p0_2_report.py — reused here for parity.
FLOAT_RE = re.compile(r"np\.float64\(([-\d.eE+]+)\)")


def parse_per_dt(s):
    if not s:
        return []
    return [float(m.group(1)) for m in FLOAT_RE.finditer(s)]


def cell_from_joint_idx(i):
    """idx = t * D + d -> (d, t)."""
    D = 3
    return i % D, i // D


# -- inputs --------------------------------------------------------------

def load_p0_4_rows(path):
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
    """Load only P0-2 ratio=50 oracle rows (referenced, not re-run)."""
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


def run_dir(weighting, setting, seed):
    return os.path.join(P0_4_BASE, f"w_{weighting}", f"setting_{setting}",
                        "baseline", f"seed{seed}")


def load_config(weighting, setting, seed):
    p = os.path.join(run_dir(weighting, setting, seed), "config.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -- selection ----------------------------------------------------------

def select_epoch_equal(epochs, recorded_best_epoch):
    """Argmax over mean(val_auc[e]), STRICT > semantics (first argmax).
    Sanity check: must equal recorded_best_epoch (else MISMATCH).
    """
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


def sample_weights(p):
    """w_d = p_d / p_mean, sum = D=3. task-major flat via tile."""
    w = np.array(p, dtype=np.float64) / np.mean(p)
    return w


def select_epoch_sample(epochs, p, last_epoch):
    """Argmax over sum_d w_d * val_auc[e][d] (the task-major 6-element list
    covers both tasks for the same domain; we average across tasks so the
    score is purely domain-weighted). Strict > semantics. `p` is the
    per-DOMAIN sample share tuple for this run's setting (domain_p()).

    If argmax == last_epoch: censored=1 (model might keep improving past).
    """
    w = sample_weights(p)
    best_e = None
    best_score = -1.0
    for ei in sorted(epochs.keys()):
        vs = epochs[ei]["val"]
        if any(v is None for v in vs):
            continue
        # vs layout = task-major [d0t0, d1t0, d2t0, d0t1, d1t1, d2t1].
        # Per-domain val AUC = mean over the 2 tasks.
        d_means = []
        for d in range(3):
            d_means.append((vs[d] + vs[3 + d]) / 2.0)
        s = float(np.sum(np.asarray(d_means) * w))
        if s > best_score:
            best_score = s
            best_e = ei
    censored = (best_e is not None and last_epoch is not None
                and int(best_e) == int(last_epoch))
    return best_e, censored


# -- per-cell pair model -------------------------------------------------

def expand_per_dt_joint(row):
    """Long list of (domain, task, value) for one joint row's per_dt_test_auc."""
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 6:
        return []
    return [(d, t, vals[t * 3 + d]) for t in TASKS for d in DOMAINS]


def expand_per_dt_oracle(row):
    """Oracle row's per_dt_test_auc has 2 entries [t0, t1] for the single
    oracle domain. Return [(d, 0, vals[0]), (d, 1, vals[1])]."""
    vals = parse_per_dt(row.get("per_dt_test_auc", ""))
    if len(vals) != 2:
        return []
    d = int(row["method_or_domain"].replace("oracle_d", ""))
    return [(d, 0, vals[0]), (d, 1, vals[1])]


# -- outputs -------------------------------------------------------------

def build_cells(p0_4_rows, p0_2_oracle_rows):
    """Yield dict rows for the long cells CSV.

    Schema: weighting, selection, setting, domain, task, seed, role,
            selected_epoch, joint_auc (test AUC at selected epoch),
            oracle_auc (from P0-2 ratio_50 oracle, ratio==50 only),
            delta_transfer (joint - oracle), censored, mismatch.
    """
    # Index P0-2 oracle rows by (setting, oracle_d, seed) -> row
    oracle_idx = {}
    for r in p0_2_oracle_rows:
        key = (r["setting"], r["method_or_domain"], int(r["seed"]))
        oracle_idx[key] = r

    out_rows = []
    for r in p0_4_rows:
        weighting = r["loss_weighting"]
        setting = r["setting"]
        seed = int(r["seed"])
        # 1. Load per-epoch metrics for checkpoint reselect
        me_path = os.path.join(run_dir(weighting, setting, seed), "metrics_epoch.csv")
        epochs = load_metrics_epoch(me_path)
        if not epochs:
            continue
        recorded_best = r.get("best_epoch", "")
        recorded_best_int = int(recorded_best) if recorded_best != "" else None
        stopped = r.get("stopped_epoch", "")
        stopped_int = int(stopped) if stopped != "" else None
        last_epoch = max(epochs.keys()) if epochs else None

        # 2. Selection = equal
        sel_eq, mismatch = select_epoch_equal(epochs, recorded_best_int)
        # 3. Selection = sample (per-domain p resolved through the setting's
        #    role rotation; identical to the p used by the training-side
        #    loss_weighting=sample arm, which reads actual train counts).
        sel_sa, censored_sa = select_epoch_sample(epochs, domain_p(setting), last_epoch)

        # 4. Per-selection, build the per-(d, t) cell row by reading
        #    the joint 6-cell test_auc at the selected epoch.
        for sel_name, sel_ep in (("equal", sel_eq), ("sample", sel_sa)):
            if sel_ep is None or sel_ep not in epochs:
                continue
            cell_test = epochs[sel_ep]["test"]
            for d in DOMAINS:
                for t in TASKS:
                    idx = t * 3 + d
                    joint_v = cell_test[idx]
                    if joint_v is None:
                        continue
                    # oracle lookup: P0-2 ratio_50 oracle_d{d} for this setting/seed
                    oracle_key = (setting, f"oracle_d{d}", seed)
                    oracle_row = oracle_idx.get(oracle_key)
                    if oracle_row is None:
                        oracle_v = None
                        delta = None
                    else:
                        o_vals = parse_per_dt(oracle_row.get("per_dt_test_auc", ""))
                        if len(o_vals) != 2:
                            oracle_v = None
                            delta = None
                        else:
                            oracle_v = o_vals[t]
                            delta = joint_v - oracle_v
                    role = ROLE_MAPS[setting][d]
                    out_rows.append({
                        "weighting": weighting,
                        "selection": sel_name,
                        "setting": setting,
                        "domain": d,
                        "task": t,
                        "seed": seed,
                        "role": role,
                        "selected_epoch": sel_ep,
                        "joint_auc": f"{joint_v:.6f}",
                        "oracle_auc": (f"{oracle_v:.6f}" if oracle_v is not None else ""),
                        "delta_transfer": (f"{delta:.6f}" if delta is not None else ""),
                        "censored": (1 if (sel_name == "sample" and censored_sa) else 0),
                        "mismatch": (1 if (sel_name == "equal" and mismatch) else 0),
                    })
    return out_rows


def write_cells_csv(rows, path):
    fields = ["weighting", "selection", "setting", "domain", "task", "seed",
              "role", "selected_epoch", "joint_auc", "oracle_auc",
              "delta_transfer", "censored", "mismatch"]
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
    """Per (weighting, selection): per-role mean / se Δ_transfer, fraction
    negative, overall AUC, worst-domain AUC, mean selected_epoch,
    fraction censored."""
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["weighting"], c["selection"])].append(c)

    summary_rows = []
    for w in WEIGHTINGS:
        for sel in SELECTIONS:
            cs = by_key.get((w, sel), [])
            deltas = [_parsef(c["delta_transfer"]) for c in cs]
            deltas_v = [x for x in deltas if x is not None]
            joint = [_parsef(c["joint_auc"]) for c in cs]
            # per-cell (domain x task) means across seeds (the test AUC at the
            # selected epoch): one mean per (setting, d, t)
            cell_means = defaultdict(list)
            for c in cs:
                j = _parsef(c["joint_auc"])
                if j is None:
                    continue
                cell_means[(c["setting"], int(c["domain"]), int(c["task"]))].append(j)
            per_cell = {k: _mean(vs) for k, vs in cell_means.items() if vs}
            overall_auc = _mean(list(per_cell.values())) if per_cell else float("nan")
            worst_auc = min(per_cell.values()) if per_cell else float("nan")

            # per-role Δ_transfer mean / se
            role_data = defaultdict(list)
            for c in cs:
                d = _parsef(c["delta_transfer"])
                if d is None:
                    continue
                role_data[c["role"]].append(d)

            frac_neg = (sum(1 for x in deltas_v if x < 0) / len(deltas_v)
                        if deltas_v else float("nan"))
            mean_sel_epoch = _mean([int(c["selected_epoch"]) for c in cs]) if cs else float("nan")
            frac_censored = (sum(c["censored"] for c in cs) / len(cs)
                             if cs else float("nan"))

            row = {
                "weighting": w,
                "selection": sel,
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
                "frac_censored": (f"{frac_censored:.4f}" if cs else ""),
            }
            summary_rows.append(row)
    return summary_rows


def write_summary_csv(rows, path):
    fields = ["weighting", "selection", "n_cells",
              "delta_L_mean", "delta_L_se",
              "delta_M_mean", "delta_M_se",
              "delta_S_mean", "delta_S_se",
              "delta_overall_mean", "frac_negative",
              "overall_auc", "worst_domain_auc",
              "mean_selected_epoch", "frac_censored"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_headline(summary_rows):
    """3 weightings x 2 selections, L 角色 / S 角色 mean Δ_transfer."""
    rows = []
    by_key = {(r["weighting"], r["selection"]): r for r in summary_rows}
    for w in WEIGHTINGS:
        for sel in SELECTIONS:
            row = by_key.get((w, sel), {})
            rows.append({
                "weighting": w,
                "selection": sel,
                "delta_L_mean": row.get("delta_L_mean", ""),
                "delta_S_mean": row.get("delta_S_mean", ""),
            })
    return rows


def write_headline_csv(rows, path):
    fields = ["weighting", "selection", "delta_L_mean", "delta_S_mean"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -- self-test -----------------------------------------------------------

def self_test():
    """In-memory round-trip: synthesize fake metrics_epoch rows for a single
    (weighting, setting, seed) and verify the parser, selection, and summary
    paths produce the expected shapes. Uses no real CSVs and no training."""
    import numpy as np
    print("=== self-test (synthetic) ===", flush=True)
    # Build a synthetic P0-2 oracle row set (36 entries: 3 settings x 4 seeds
    # x 3 oracle domains) and P0-4 joint rows.
    p0_2_oracle = []
    for setting in SETTINGS:
        for seed in SEEDS:
            for d in DOMAINS:
                # oracle per_dt: 2 entries [t0, t1] for the single oracle domain.
                p0_2_oracle.append({
                    "kind": "oracle", "ratio": 50, "setting": setting,
                    "method_or_domain": f"oracle_d{d}", "seed": seed,
                    "status": "OK",
                    "per_dt_test_auc": ",".join(
                        f"np.float64({0.65 + 0.01 * d + 0.005 * t:.6f})"
                        for t in TASKS),
                })

    # Synthetic P0-4 joint row: 1 weighting, 1 setting, 1 seed, 6-cell test_auc
    # drawn from a deterministic schedule; build metrics_epoch.csv in a tmp
    # directory and feed the real load_metrics_epoch path.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="p0_4_self_")
    weighting, setting, seed = "sample", "d0_large", 2022
    d_root = os.path.join(tmp, f"w_{weighting}", f"setting_{setting}",
                          "baseline", f"seed{seed}")
    os.makedirs(d_root, exist_ok=True)
    ep_fields = ["experiment_stage", "experiment_name", "model", "method",
                 "setting", "seed", "epoch", "domain", "task",
                 "train_loss", "val_loss", "val_auc", "test_auc"]
    # Synthesize 5 epochs: val_auc peaks at epoch 2 (mean 0.7483 > epoch 3
    # mean 0.71, so equal-selection's strict-argmax lands on epoch 2).
    # Mean val AUC score: epoch 0 (0.55), epoch 1 (0.62), epoch 2 (0.7483),
    # epoch 3 (0.71), epoch 4 (0.70). For task-major idx = t*3 + d.
    val_schedule = [
        # (epoch, val[6], test[6])
        (0, [0.55] * 6, [0.54] * 6),
        (1, [0.62] * 6, [0.60] * 6),
        (2, [0.78, 0.74, 0.70, 0.79, 0.75, 0.72],
            [0.77, 0.73, 0.69, 0.78, 0.74, 0.71]),
        (3, [0.71] * 6, [0.70] * 6),
        (4, [0.70] * 6, [0.69] * 6),
    ]
    with open(os.path.join(d_root, "metrics_epoch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields)
        w.writeheader()
        for ei, vs, ts in val_schedule:
            for t in TASKS:
                for d in DOMAINS:
                    idx = t * 3 + d
                    w.writerow({
                        "experiment_stage": "P0-4", "experiment_name": "loss_weighting",
                        "model": "MDMTRec", "method": "baseline",
                        "setting": setting, "seed": seed,
                        "epoch": ei, "domain": d, "task": t,
                        "train_loss": "1.0",
                        "val_loss": "0.5",
                        "val_auc": f"{vs[idx]:.6f}",
                        "test_auc": f"{ts[idx]:.6f}",
                    })
    # Build a fake P0_4_raw row + override run_dir to point at tmp.
    p0_4_rows = [{
        "kind": "joint", "ratio": 50, "setting": setting,
        "loss_weighting": weighting,
        "method_or_domain": "baseline", "seed": seed,
        "status": "OK", "best_epoch": "2", "stopped_epoch": "4",
        "per_dt_test_auc": ",".join(f"np.float64({0.69:.6f})" for _ in range(6)),
        "fit_sec": "100", "peak_mem_mb": "500", "wall_sec": "110",
        "best_val_auc": "0.72", "note": "",
    }]
    # Monkey-patch the module-level run_dir to redirect to our tmp dir.
    orig_run_dir = globals()["run_dir"]

    def patched_run_dir(w_, s_, sd_):
        return os.path.join(tmp, f"w_{w_}", f"setting_{s_}",
                            "baseline", f"seed{sd_}")

    globals()["run_dir"] = patched_run_dir
    try:
        cells = build_cells(p0_4_rows, p0_2_oracle)
    finally:
        globals()["run_dir"] = orig_run_dir

    # Cell count: 2 selections x 3 domains x 2 tasks x 1 setting x 1 seed = 12
    assert len(cells) == 12, f"expected 12 cells, got {len(cells)}"
    # Both selections should pick epoch 2 (val_auc peaks at 2 with strict >).
    epochs_seen = {(c["weighting"], c["selection"]): c["selected_epoch"]
                   for c in cells[:6]}
    for k, ep in epochs_seen.items():
        assert ep == 2, f"{k} -> {ep}, expected 2"
    # mismatch column should be 0 (recompute matches recorded best_epoch=2).
    mismatches = sum(c["mismatch"] for c in cells)
    assert mismatches == 0, f"unexpected mismatches: {mismatches}"
    # Censored: argmax != last_epoch (last=4) -> 0
    censored = sum(c["censored"] for c in cells)
    assert censored == 0, f"unexpected censored: {censored}"
    # delta_transfer format should be a 6-decimal string when oracle present.
    dt = cells[0]["delta_transfer"]
    assert "." in dt and len(dt.split(".")[-1]) == 6, f"bad delta format: {dt}"

    summary = build_summary(cells)
    assert len(summary) == 6, f"expected 6 summary rows, got {len(summary)}"
    headline = build_headline(summary)
    assert len(headline) == 6, f"expected 6 headline rows, got {len(headline)}"

    # Role-rotation regression: domain_p must follow ROLE_MAPS, not the fixed
    # d0_large order. d1_large puts role S on domain 0 and role L on domain 1.
    p1 = domain_p("d1_large")
    assert abs(p1[0] - 0.016) < 1e-12 and abs(p1[1] - 0.8) < 1e-12, p1
    p2 = domain_p("d2_large")
    assert abs(p2[0] - 0.184) < 1e-12 and abs(p2[2] - 0.8) < 1e-12, p2
    # Selection under rotation: two epochs where domain 0 wins epoch A and
    # domain 1 wins epoch B (per-domain val, both tasks equal). Under
    # d0_large p (d0=L) argmax = A; under d1_large p (d0=S, d1=L) argmax = B.
    rot_epochs = {
        0: {"val": [0.90, 0.60, 0.60, 0.90, 0.60, 0.60], "test": [None] * 6},
        1: {"val": [0.60, 0.89, 0.60, 0.60, 0.89, 0.60], "test": [None] * 6},
    }
    e_d0, _ = select_epoch_sample(rot_epochs, domain_p("d0_large"), 1)
    e_d1, _ = select_epoch_sample(rot_epochs, domain_p("d1_large"), 1)
    assert e_d0 == 0, f"d0_large sample-select -> {e_d0}, expected 0"
    assert e_d1 == 1, f"d1_large sample-select -> {e_d1}, expected 1"
    print("  rotation regression OK", flush=True)

    # Round-trip write_csv then read back to verify schema
    write_cells_csv(cells, os.path.join(tmp, "cells.csv"))
    write_summary_csv(summary, os.path.join(tmp, "summary.csv"))
    write_headline_csv(headline, os.path.join(tmp, "headline.csv"))
    for f in ("cells.csv", "summary.csv", "headline.csv"):
        with open(os.path.join(tmp, f), "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        assert len(lines) >= 2, f"{f} has only {len(lines)} lines"
    print(f"  cells={len(cells)} summary={len(summary)} headline={len(headline)}", flush=True)
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
    os.makedirs(P0_4_BASE, exist_ok=True)
    p0_4_rows = load_p0_4_rows(P0_4_RAW)
    p0_2_oracle = load_p0_2_oracle_rows(P0_2_RAW)
    print(f"loaded {len(p0_4_rows)} P0-4 rows from {P0_4_RAW}", flush=True)
    print(f"loaded {len(p0_2_oracle)} P0-2 ratio_50 oracle rows from {P0_2_RAW}",
          flush=True)
    cells = build_cells(p0_4_rows, p0_2_oracle)
    write_cells_csv(cells, OUT_CELLS)
    print(f"wrote {OUT_CELLS} ({len(cells)} rows)", flush=True)
    summary = build_summary(cells)
    write_summary_csv(summary, OUT_SUMMARY)
    print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)", flush=True)
    headline = build_headline(summary)
    write_headline_csv(headline, OUT_HEADLINE)
    print(f"wrote {OUT_HEADLINE} ({len(headline)} rows)", flush=True)


if __name__ == "__main__":
    main()
