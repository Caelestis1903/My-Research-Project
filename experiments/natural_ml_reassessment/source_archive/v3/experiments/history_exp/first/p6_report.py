"""P6 ablation report.

Reads:
  - experiments/results/p6/P6_raw.csv   (variant, seed, status, test_auc_mean, ...)
  - experiments/results/p5/P5_raw.csv   (config == ml_r30, status == OK,
                                         method in {baseline, fixed073, comp1,
                                                    comp2, comp1_comp2, comp123})

Writes:
  - experiments/results/P6_ablation.csv  (long table, paired t-test per seed)
  - experiments/results/p6/figs/p6_sensitivity.png  (kappa0 + gamma0 sweeps)

Robust to partial completion: arms with <2 OK seeds get blank p/d and a
stderr warning; groups where all arms are missing are skipped silently
(no error exit). P6_raw.csv may be missing or empty during a live run -
the report still produces the P5-only rows (g1 comp1, g6 comp1/comp2/
comp1_comp2/comp123) and an empty plot.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RES = os.path.join(BASE, "results")
P5_CSV = os.path.join(RES, "p5", "P5_raw.csv")
P6_CSV = os.path.join(RES, "p6", "P6_raw.csv")
P6_OUT = os.path.join(RES, "P6_ablation.csv")
P6_FIG_DIR = os.path.join(RES, "p6", "figs")
P6_FIG = os.path.join(P6_FIG_DIR, "p6_sensitivity.png")
os.makedirs(P6_FIG_DIR, exist_ok=True)

P5_METHODS = ["baseline", "fixed073", "comp1", "comp2",
              "comp1_comp2", "comp123"]
# alias for stdout display only; CSV keeps the original name for traceability
DISPLAY_ALIAS = {"comp1_comp2": "comp1&2"}


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def display(name):
    return DISPLAY_ALIAS.get(name, name)


def load_arms():
    """dict {variant_name: {seed: auc}} from P5 (ml_r30) and P6 (any variant)."""
    arms = {}

    if os.path.exists(P5_CSV):
        with open(P5_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("config") != "ml_r30":
                    continue
                if r.get("status") != "OK":
                    continue
                m = r.get("method")
                if m not in P5_METHODS:
                    continue
                auc = _f(r.get("test_auc_mean", ""))
                seed = _f(r.get("seed", ""))
                if auc is None or seed is None:
                    continue
                arms.setdefault(m, {})[int(seed)] = auc

    if os.path.exists(P6_CSV):
        with open(P6_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") != "OK":
                    continue
                v = r.get("variant")
                if not v:
                    continue
                auc = _f(r.get("test_auc_mean", ""))
                seed = _f(r.get("seed", ""))
                if auc is None or seed is None:
                    continue
                arms.setdefault(v, {})[int(seed)] = auc
    return arms


def paired_stats(arm_a, arm_b):
    """Paired t-test summary for two arm dicts {seed: auc}.

    Returns dict with n, mean_a, std_a, mean_b, std_b, diff, t, p, d.
    n = size of the seed intersection. mean/std use ddof=1. t/p/d are
    None when n < 2 (single-pair case has sd=0 -> d undefined).
    """
    common = sorted(set(arm_a.keys()) & set(arm_b.keys()))
    n = len(common)
    nan = float("nan")
    a = np.array([arm_a[s] for s in common], dtype=float) if n else np.zeros(0)
    b = np.array([arm_b[s] for s in common], dtype=float) if n else np.zeros(0)
    if n == 0:
        return {"n": 0, "mean_a": nan, "std_a": nan,
                "mean_b": nan, "std_b": nan,
                "diff": nan, "t": None, "p": None, "d": None}
    mean_a, std_a = float(a.mean()), float(a.std(ddof=1))
    mean_b, std_b = float(b.mean()), float(b.std(ddof=1))
    diff = a - b
    diff_mean = float(diff.mean())
    if n < 2:
        return {"n": n, "mean_a": mean_a, "std_a": std_a,
                "mean_b": mean_b, "std_b": std_b,
                "diff": diff_mean, "t": None, "p": None, "d": None}
    sd = float(diff.std(ddof=1))
    tres = stats.ttest_rel(a, b)
    t_val = float(tres.statistic)
    p_val = float(tres.pvalue)
    d_val = diff_mean / sd if sd > 0 else nan
    return {"n": n, "mean_a": mean_a, "std_a": std_a,
            "mean_b": mean_b, "std_b": std_b,
            "diff": diff_mean, "t": t_val, "p": p_val, "d": d_val}


def fmt6(x):
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    return f"{x:.6f}"


# (group label, list of (variant, ref) pairs)
GROUPS = [
    ("g1 momentum_tiering",
     [("g1_uniform_gamma", "comp1"),
      ("g1_uniform_gamma", "baseline"),
      ("comp1", "baseline")]),
    ("g2 gamma_source",
     [("g2_manual_gamma", "comp1"),
      ("g2_snr_gamma", "comp1")]),
    ("g3 beta_granularity",
     [("g3_beta_perdomain_nocur", "baseline")]),
    ("g4 curriculum_vs_fixed",
     [("comp2", "g3_beta_perdomain_nocur")]),
    ("g5 curriculum_direction",
     [("g5_reverse_curriculum", "comp2")]),
    ("g6 component_orthogonality",
     [("comp1", "baseline"),
      ("comp2", "baseline"),
      ("g6_comp3", "baseline"),
      ("comp1_comp2", "baseline"),
      ("g6_comp1_comp3", "baseline"),
      ("g6_comp2_comp3", "baseline"),
      ("comp123", "baseline")]),
    ("g7 kappa0_sensitivity",
     [("g7_kappa0_0.1", "comp1_comp2"),
      ("g7_kappa0_0.25", "comp1_comp2"),
      ("g7_kappa0_1.0", "comp1_comp2"),
      ("g7_kappa0_2.0", "comp1_comp2")]),
    ("g7 gamma0_sensitivity",
     [("g7_gamma0_0.8", "comp1"),
      ("g7_gamma0_0.85", "comp1"),
      ("g7_gamma0_0.95", "comp1"),
      ("g7_gamma0_0.99", "comp1")]),
    ("g8 adam_momentum_relation",
     [("comp1", "baseline"),
      ("g8_stacked", "baseline")]),
]


def build_rows(arms):
    """Return list of (label, per_group) for groups with at least one OK row."""
    out = []
    for label, pairs in GROUPS:
        per_group = []
        any_ok = False
        for variant, ref in pairs:
            a = arms.get(variant, {})
            b = arms.get(ref, {})
            s = paired_stats(a, b)
            row = {"group": label, "variant": variant, "n": s["n"],
                   "mean": s["mean_a"], "std": s["std_a"],
                   "ref": ref, "diff": s["diff"],
                   "t_stat": s["t"], "p_value": s["p"], "cohens_d": s["d"]}
            if s["n"] == 0:
                print(f"  [warn] {label} {variant} vs {ref}: missing "
                      f"(n_v={len(a)}, n_r={len(b)})",
                      file=sys.stderr, flush=True)
            else:
                any_ok = True
                if s["n"] < 2:
                    print(f"  [warn] {label} {variant} vs {ref}: only "
                          f"{s['n']} paired seed(s); p/d left blank",
                          file=sys.stderr, flush=True)
            per_group.append((row, a, b))
        if not any_ok:
            print(f"  [skip] {label}: no arm has any OK seed",
                  file=sys.stderr, flush=True)
            continue
        out.append((label, per_group))
    return out


def write_csv(rows, path):
    fields = ["group", "variant", "n", "mean", "std", "ref",
              "diff", "t_stat", "p_value", "cohens_d"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for _, per_group in rows:
            for row, _, _ in per_group:
                w.writerow({k: (fmt6(v) if isinstance(v, float) else v)
                            for k, v in row.items()})


def _arm_table_row(name, arm_dict):
    n = len(arm_dict)
    if n == 0:
        return display(name), 0, "", ""
    arr = np.array(list(arm_dict.values()), dtype=float)
    m = f"{arr.mean():.6f}"
    s = f"{arr.std(ddof=1):.6f}" if n > 1 else ""
    return display(name), n, m, s


def print_summary(arms, rows):
    print("=" * 92)
    print("P6 ablation summary  (config = ml_r30, seeds 2022..2026, paired t)")
    print("=" * 92)
    for label, per_group in rows:
        print(f"\n[{label}]")
        arms_set = set()
        for row, _, _ in per_group:
            arms_set.add(row["variant"])
            arms_set.add(row["ref"])
        print(f"  {'arm':<32}{'n':>4}{'mean':>14}{'std':>14}")
        for a in sorted(arms_set):
            name, n, m, s = _arm_table_row(a, arms.get(a, {}))
            print(f"  {name:<32}{n:>4}{m:>14}{s:>14}")
        print(f"  {'pair':<40}{'n':>4}{'diff':>12}{'t':>10}"
              f"{'p':>12}{'d':>10}")
        for row, _, _ in per_group:
            n = row["n"]
            d = fmt6(row["diff"])
            t = fmt6(row["t_stat"])
            p = fmt6(row["p_value"])
            cd = fmt6(row["cohens_d"])
            pair = f"{display(row['variant'])} vs {display(row['ref'])}"
            print(f"  {pair:<40}{n:>4}{d:>12}{t:>10}{p:>12}{cd:>10}")

    # g6 pairwise diff matrix (stdout only)
    g6_arms = ["comp1", "comp2", "g6_comp3", "comp1_comp2",
               "g6_comp1_comp3", "g6_comp2_comp3", "comp123"]
    have_any = any(a in arms and arms[a] for a in g6_arms)
    if have_any:
        print(f"\n[g6 pairwise diffs (mean(variant) - mean(ref), paired)]")
        col_w = 14
        header = " " * 24 + "".join(f"{a:>{col_w}}" for a in g6_arms)
        print(header)
        for ai in g6_arms:
            cells = [f"{display(ai):<24}"]
            for aj in g6_arms:
                if ai == aj:
                    cells.append(f"{'---':>{col_w}}")
                    continue
                if ai not in arms or not arms[ai] or \
                        aj not in arms or not arms[aj]:
                    cells.append(f"{'NA':>{col_w}}")
                    continue
                s = paired_stats(arms[ai], arms[aj])
                cells.append(f"{fmt6(s['diff']):>{col_w}}")
            print("".join(cells))


def plot_sensitivity(arms, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    baseline_mean = None
    if "baseline" in arms and arms["baseline"]:
        b = np.array(list(arms["baseline"].values()), dtype=float)
        baseline_mean = float(b.mean())

    # left: kappa0 sweep
    kappa_points = [("g7_kappa0_0.1", 0.1), ("g7_kappa0_0.25", 0.25),
                    ("comp1_comp2", 0.5), ("g7_kappa0_1.0", 1.0),
                    ("g7_kappa0_2.0", 2.0)]
    xs, ys, es = [], [], []
    for name, x in kappa_points:
        d = arms.get(name, {})
        if not d:
            continue
        arr = np.array(list(d.values()), dtype=float)
        xs.append(x)
        ys.append(arr.mean())
        es.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    if xs:
        ax1.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="C0",
                     label="kappa0 sweep (comp1 + curriculum)")
    if baseline_mean is not None:
        ax1.axhline(baseline_mean, color="grey", linestyle="--",
                    label=f"baseline = {baseline_mean:.4f}")
    ax1.set_xscale("log")
    ax1.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0])
    ax1.set_xticklabels(["0.1", "0.25", "0.5", "1.0", "2.0"])
    ax1.set_xlabel("kappa0")
    ax1.set_ylabel("test_auc_mean")
    ax1.set_title("kappa0 sensitivity (D=3, comp1 + curriculum)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # right: gamma0 sweep
    gamma_points = [("g7_gamma0_0.8", 0.8), ("g7_gamma0_0.85", 0.85),
                    ("comp1", 0.9), ("g7_gamma0_0.95", 0.95),
                    ("g7_gamma0_0.99", 0.99)]
    xs, ys, es = [], [], []
    for name, x in gamma_points:
        d = arms.get(name, {})
        if not d:
            continue
        arr = np.array(list(d.values()), dtype=float)
        xs.append(x)
        ys.append(arr.mean())
        es.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    if xs:
        ax2.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="C1",
                     label="gamma0 sweep (per_domain_only)")
    if baseline_mean is not None:
        ax2.axhline(baseline_mean, color="grey", linestyle="--",
                    label=f"baseline = {baseline_mean:.4f}")
    ax2.set_xlabel("gamma0")
    ax2.set_ylabel("test_auc_mean")
    ax2.set_title("gamma0 sensitivity (D=3, per_domain_only)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true",
                    help="skip figure generation (CSV + stdout only)")
    args = ap.parse_args()

    arms = load_arms()
    n_pairs = sum(len(v) for v in arms.values())
    print(f"[load] {len(arms)} arms, {n_pairs} (variant,seed) pairs",
          flush=True)
    if not arms:
        print("[abort] no usable arms", file=sys.stderr, flush=True)
        return 1

    rows = build_rows(arms)
    if not rows:
        print("[abort] no group has any usable data", file=sys.stderr,
              flush=True)
        return 1

    write_csv(rows, P6_OUT)
    print(f"[csv] wrote {P6_OUT}", flush=True)
    print_summary(arms, rows)

    if not args.no_plot:
        plot_sensitivity(arms, P6_FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
