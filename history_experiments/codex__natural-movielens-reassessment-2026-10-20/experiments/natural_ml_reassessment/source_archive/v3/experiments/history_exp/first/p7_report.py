"""P7 report.

Inputs:
  - experiments/results/p7/P7_backbones.csv (p7_run.py auto-appends; may
    be missing or empty while the batch is still running)
  - experiments/results/p5/P5_raw.csv       (mdmtoe in-repo reference:
                                             config=ml_orig, method=baseline)

Outputs:
  - experiments/results/P7_summary.csv  (long table, 6-decimal)
  - stdout summary, three sections:
      [A] backbone reproduction (D=3, no momentum) vs M3oE Table 2
      [B] component-1 port paired-t (per_domain_only vs none) per backbone
      [C] D=5 extension: per-arm stats + per-backbone paired-t + per-(d,t)
          AUC mean matrix (5 x 2) with sanity warnings

Idempotent: P7_summary.csv is overwritten on every run.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy import stats
import matplotlib  # noqa: F401  (kept so a future plot addition is one-liner)
matplotlib.use("Agg")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RES = os.path.join(BASE, "results")
P7_CSV = os.path.join(RES, "p7", "P7_backbones.csv")
P5_CSV = os.path.join(RES, "p5", "P5_raw.csv")
OUT_CSV = os.path.join(RES, "P7_summary.csv")

# M3oE Table 2 (MovieLens, AUC*100, MDMT setup)
M3OE_PAPER = {
    "mmoe":        76.35,
    "ple":         75.68,
    "star":        76.08,
    "sharebottom": 76.26,
    "mlp":         75.62,   # listed in paper; we do not run MLP, context only
}

FIELDS = ["section", "backbone", "domain_version", "momentum", "n", "mean",
          "std", "ref", "diff", "t_stat", "p_value", "cohens_d"]


# -------------------------- helpers --------------------------

def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def fmt6(x):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.6f}"


def load_p7():
    """{ (backbone, domain_version, momentum): {seed: row_dict} }.

    The CSV is written by p7_run.DictWriter using the first row's fieldnames,
    so D=5 rows have 15 physical fields while the header only declares 11
    (auc_d3t0..auc_d4t1 are headless extras). We read with csv.reader to
    capture all physical fields, then synthesize the extra auc_d* keys so
    per_domain_matrix() can find them for D=5.
    """
    arms = {}
    if not os.path.exists(P7_CSV):
        return arms
    with open(P7_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return arms
        n_header = len(header)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                bb, dv, mom, sd = row[0], row[1], row[2], int(row[3])
            except (ValueError, IndexError):
                continue
            r = {h: row[i] if i < len(row) else ""
                 for i, h in enumerate(header)}
            # Map any physical fields beyond the declared header to the
            # expected auc_d* names so per_domain_matrix() picks them up.
            # Expected physical layout:
            #   idx 0..4   : backbone, domain_version, momentum, seed, test_auc_mean
            #   idx 5..10  : auc_d0t0..auc_d2t1
            #   idx 11..14 : auc_d3t0..auc_d4t1 (D=5 only; declared as headless)
            extra_dt = [(3, 0), (3, 1), (4, 0), (4, 1)]
            for k, (d, t) in enumerate(extra_dt):
                phys = 5 + 6 + k
                if phys < len(row) and row[phys] != "":
                    r[f"auc_d{d}t{t}"] = row[phys]
            arms.setdefault((bb, dv, mom), {})[sd] = r
    return arms


def load_mdmtoe_ref():
    """test_auc_mean list from P5 ml_orig baseline OK rows."""
    if not os.path.exists(P5_CSV):
        return []
    out = []
    with open(P5_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("config") == "ml_orig" and
                    r.get("method") == "baseline" and
                    r.get("status") == "OK"):
                v = _f(r.get("test_auc_mean", ""))
                if v is not None:
                    out.append(v)
    return out


def arm_arr(arm_dict):
    """Extract per-seed test_auc_mean from {seed: row_dict}."""
    return {s: _f(r.get("test_auc_mean", "")) for s, r in arm_dict.items()
            if _f(r.get("test_auc_mean", "")) is not None}


def paired_stats(a, b):
    """a, b: {seed: auc}. Paired by seed; two-sided scipy ttest_rel.
    Returns dict with n, mean_a, std_a, mean_b, std_b, diff, t, p, d.
    n<2 -> t/p/d are None (sd=0 -> d undefined for a single pair)."""
    common = sorted(set(a.keys()) & set(b.keys()))
    n = len(common)
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mean_a": nan, "std_a": nan,
                "mean_b": nan, "std_b": nan,
                "diff": nan, "t": None, "p": None, "d": None}
    av = np.array([a[s] for s in common], dtype=float)
    bv = np.array([b[s] for s in common], dtype=float)
    mean_a, std_a = float(av.mean()), float(av.std(ddof=1))
    mean_b, std_b = float(bv.mean()), float(bv.std(ddof=1))
    diff = av - bv
    diff_mean = float(diff.mean())
    if n < 2:
        return {"n": n, "mean_a": mean_a, "std_a": std_a,
                "mean_b": mean_b, "std_b": std_b,
                "diff": diff_mean, "t": None, "p": None, "d": None}
    sd = float(diff.std(ddof=1))
    tres = stats.ttest_rel(av, bv)
    d_val = diff_mean / sd if sd > 0 else nan
    return {"n": n, "mean_a": mean_a, "std_a": std_a,
            "mean_b": mean_b, "std_b": std_b,
            "diff": diff_mean, "t": float(tres.statistic),
            "p": float(tres.pvalue), "d": d_val}


def per_domain_matrix(arm_dict, D=None, T=None):
    """Mean across seeds of auc_d{d}t{t} for each (d,t) cell.
    Returns (means, bad, D, T). D and T are auto-detected from the row keys
    (max d, max t present) if not given.
    """
    # Auto-detect (D, T) from the union of auc_d* keys across all rows.
    if D is None or T is None:
        max_d, max_t = -1, -1
        for _seed, row in arm_dict.items():
            for k in row:
                if k.startswith("auc_d") and k.endswith(("t0", "t1")):
                    try:
                        # auc_d{d}t{t}
                        d = int(k[len("auc_d"):-len("t1")].split("t")[0])
                        t = int(k.rsplit("t", 1)[1])
                        if d > max_d:
                            max_d = d
                        if t > max_t:
                            max_t = t
                    except (ValueError, IndexError):
                        continue
        D = max_d + 1 if max_d >= 0 else (D or 0)
        T = max_t + 1 if max_t >= 0 else (T or 0)
    cells = {(d, t): [] for d in range(D) for t in range(T)}
    for _seed, row in arm_dict.items():
        for d in range(D):
            for t in range(T):
                v = _f(row.get(f"auc_d{d}t{t}", ""))
                if v is not None and not math.isnan(v):
                    cells[(d, t)].append(v)
    means = {k: (float(np.mean(vs)) if vs else float("nan"))
             for k, vs in cells.items()}
    bad = []
    for (d, t), v in means.items():
        if math.isnan(v):
            bad.append(f"d{d}t{t}=NaN")
        elif v < 0.55:                     # "near 0.5" with a 5-pt tolerance
            bad.append(f"d{d}t{t}={v:.4f}")
    return means, bad, D, T


# -------------------------- sections --------------------------

def section_A(p7_arms, mdmtoe_ref, csv_rows):
    print("\n[A] backbone reproduction  (domain=three, momentum=none)")
    header = (f"  {'backbone':<14}{'n':>4}{'mean':>10}{'std':>10}  "
              f"{'paper*100':>10}{'diff*100':>10}  "
              f"{'P5_mdmtoe':>10}{'P5_std':>10}{'delta*100':>10}")
    print(header)
    for bb in ["mmoe", "ple", "star", "sharebottom"]:
        arm = arm_arr(p7_arms.get((bb, "three", "none"), {}))
        n = len(arm)
        if n:
            arr = np.array(list(arm.values()), dtype=float)
            mean, std = float(arr.mean()), float(arr.std(ddof=1))
        else:
            mean, std = float("nan"), float("nan")
        paper = M3OE_PAPER[bb]
        ours100 = mean * 100 if n else float("nan")
        diff100 = (ours100 - paper) if n else float("nan")
        if mdmtoe_ref:
            r = np.array(mdmtoe_ref, dtype=float)
            rmean = float(r.mean())
            rstd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        else:
            rmean, rstd = float("nan"), float("nan")
        delta100 = (mean - rmean) * 100 if n and not math.isnan(rmean) else float("nan")
        print(f"  {bb:<14}{n:>4}{fmt6(mean):>10}{fmt6(std):>10}  "
              f"{paper:>10.2f}{fmt6(diff100):>10}  "
              f"{fmt6(rmean):>10}{fmt6(rstd):>10}{fmt6(delta100):>10}")
        if n:
            csv_rows.append({
                "section": "A", "backbone": bb, "domain_version": "three",
                "momentum": "none", "n": n, "mean": mean, "std": std,
                "ref": "M3oE_Table2", "diff": (ours100 - paper) / 100.0,
                "t_stat": "", "p_value": "", "cohens_d": ""})
    # MLP context (no run, paper only)
    print(f"  {'mlp (paper)':<14}{'':>4}{'':>10}{'':>10}  "
          f"{M3OE_PAPER['mlp']:>10.2f}{' (no run)':>10}")
    # mdmtoe in-repo reference (P5 ml_orig baseline)
    if mdmtoe_ref:
        r = np.array(mdmtoe_ref, dtype=float)
        rmean = float(r.mean())
        rstd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        print(f"  {'mdmtoe (P5)':<14}{len(mdmtoe_ref):>4}"
              f"{fmt6(rmean):>10}{fmt6(rstd):>10}  "
              f"{'---':>10}{'---':>10}  "
              f"{fmt6(rmean):>10}{fmt6(rstd):>10}{'---':>10}")
        csv_rows.append({
            "section": "A", "backbone": "mdmtoe", "domain_version": "three",
            "momentum": "none", "n": len(mdmtoe_ref), "mean": rmean,
            "std": rstd, "ref": "P5_ml_orig_baseline", "diff": "",
            "t_stat": "", "p_value": "", "cohens_d": ""})


def section_B(p7_arms, csv_rows):
    print("\n[B] component-1 port  (domain=three, paired-t: "
          "per_domain_only vs none)")
    print(f"  {'backbone':<14}{'n':>4}{'diff':>10}{'t':>10}"
          f"{'p':>10}{'d':>10}")
    for bb in ["mmoe", "ple", "star", "sharebottom"]:
        a = arm_arr(p7_arms.get((bb, "three", "per_domain_only"), {}))
        b = arm_arr(p7_arms.get((bb, "three", "none"), {}))
        s = paired_stats(a, b)
        print(f"  {bb:<14}{s['n']:>4}{fmt6(s['diff']):>10}"
              f"{fmt6(s['t']):>10}{fmt6(s['p']):>10}{fmt6(s['d']):>10}")
        if s["n"] < 2:
            print(f"    [warn] {bb}: only {s['n']} paired seed(s); "
                  f"t/p/d left blank", file=sys.stderr, flush=True)
        if s["n"] == 0:
            print(f"    [warn] {bb}: arm missing entirely",
                  file=sys.stderr, flush=True)
        if s["n"]:
            csv_rows.append({
                "section": "B", "backbone": bb, "domain_version": "three",
                "momentum": "per_domain_only", "n": s["n"],
                "mean": s["mean_a"], "std": s["std_a"],
                "ref": "none", "diff": s["diff"],
                "t_stat": s["t"], "p_value": s["p"], "cohens_d": s["d"]})


def section_C(p7_arms, csv_rows):
    print("\n[C] D/T extension  (domain=five, D=5, T=2)")
    print(f"  {'backbone':<10}{'momentum':<18}{'n':>4}"
          f"{'mean':>10}{'std':>10}")
    arms_c = {}          # (bb, mom) -> full row dict {seed: row}
    arms_c_arr = {}      # (bb, mom) -> {seed: auc_float}
    for bb in ["mdmtoe", "mmoe"]:
        for mom in ["none", "per_domain_only"]:
            full = p7_arms.get((bb, "five", mom), {})
            arr = arm_arr(full)
            arms_c[(bb, mom)] = full
            arms_c_arr[(bb, mom)] = arr
            n = len(arr)
            if n:
                a = np.array(list(arr.values()), dtype=float)
                mean, std = float(a.mean()), float(a.std(ddof=1))
            else:
                mean, std = float("nan"), float("nan")
            print(f"  {bb:<10}{mom:<18}{n:>4}"
                  f"{fmt6(mean):>10}{fmt6(std):>10}")
            if n:
                csv_rows.append({
                    "section": "C", "backbone": bb, "domain_version": "five",
                    "momentum": mom, "n": n, "mean": mean, "std": std,
                    "ref": "", "diff": "", "t_stat": "",
                    "p_value": "", "cohens_d": ""})
            else:
                print(f"    [warn] {bb} {mom}: no OK seed in P7_backbones.csv",
                      file=sys.stderr, flush=True)

    print(f"\n  paired-t (per_domain_only vs none)")
    print(f"  {'backbone':<10}{'n':>4}{'diff':>10}{'t':>10}"
          f"{'p':>10}{'d':>10}")
    for bb in ["mdmtoe", "mmoe"]:
        a = arms_c_arr[(bb, "per_domain_only")]
        b = arms_c_arr[(bb, "none")]
        s = paired_stats(a, b)
        print(f"  {bb:<10}{s['n']:>4}{fmt6(s['diff']):>10}"
              f"{fmt6(s['t']):>10}{fmt6(s['p']):>10}{fmt6(s['d']):>10}")
        if s["n"] < 2 and s["n"] > 0:
            print(f"    [warn] {bb}: only {s['n']} paired seed(s); "
                  f"t/p/d left blank", file=sys.stderr, flush=True)
        if s["n"]:
            csv_rows.append({
                "section": "C", "backbone": bb, "domain_version": "five",
                "momentum": "per_domain_only", "n": s["n"],
                "mean": s["mean_a"], "std": s["std_a"],
                "ref": "none", "diff": s["diff"],
                "t_stat": s["t"], "p_value": s["p"], "cohens_d": s["d"]})

    print(f"\n  per-(d,t) test AUC mean across seeds (D and T auto-detected):")
    cell_header = "".join(f"{f' d{d}t{t}':>9}" for d in range(5) for t in range(2))
    print(f"  {'backbone':<10}{'momentum':<18}{cell_header}")
    for (bb, mom), full in arms_c.items():
        if not full:
            continue
        means, bad, D_det, T_det = per_domain_matrix(full)
        # Only print up to the actually-detected (D, T); pad with empty if smaller.
        cells_str = "".join(
            f"{fmt6(means.get((d, t), float('nan'))):>9}"
            for d in range(5) for t in range(2))
        print(f"  {bb:<10}{mom:<18}{cells_str}")
        if bad:
            print(f"    [warn] {bb} {mom}: {', '.join(bad)}",
                  file=sys.stderr, flush=True)


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet-empty", action="store_true",
                    help="suppress the 'nothing to report' banner")
    args = ap.parse_args()

    p7_arms = load_p7()
    mdmtoe_ref = load_mdmtoe_ref()

    if not p7_arms:
        msg = (f"[p7_report] {P7_CSV} not found or empty; "
               f"nothing to report. (mdmtoe P5 ref rows = {len(mdmtoe_ref)})")
        if not args.quiet_empty:
            print(msg, flush=True)
        return 0

    total = sum(len(v) for v in p7_arms.values())
    print(f"[load] {len(p7_arms)} (backbone,domain_version,momentum) cells, "
          f"{total} (cell,seed) rows from P7_backbones.csv", flush=True)
    print(f"[load] {len(mdmtoe_ref)} mdmtoe P5 ml_orig baseline rows",
          flush=True)

    csv_rows = []
    section_A(p7_arms, mdmtoe_ref, csv_rows)
    section_B(p7_arms, csv_rows)
    section_C(p7_arms, csv_rows)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: (fmt6(v) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\n[csv] wrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
