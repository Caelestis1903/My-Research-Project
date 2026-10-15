"""P1 post-processing: verdicts + tables + figures from the D02/D04/D06 CSVs."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results", "p1")
FIG = os.path.join(BASE, "results", "p1", "figs")
os.makedirs(FIG, exist_ok=True)
SEEDS = [2022, 2023, 2024, 2025, 2026]
PARAMS = ["_weight_exp_d", "_weight_exp_t", "_weight_bal_d", "_weight_bal_t"]


def rd(name):
    return list(csv.DictReader(open(os.path.join(R, name))))


# ---------------- D0.2 ----------------
def d02():
    rows = []
    for s in SEEDS:
        rows += rd(f"D02_arch_traj_s{s}_b40960.csv")
    summ = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEEDS:
        rs = [r for r in rows if int(r["seed"]) == s]
        by_param = defaultdict(dict)
        for r in rs:
            by_param[r["param_name"]][int(r["epoch"])] = float(r["sigmoid_value"])
        for p in PARAMS:
            d = by_param[p]
            if -1 not in d:
                continue
            init = d[-1]
            train_ep = [e for e in d if e >= 0]
            fin = d[max(train_ep)]
            dev = max(abs(v - init) for e, v in d.items() if e >= 0)
            summ.append({"seed": s, "param": p, "init": init, "final": fin,
                         "max_abs_dev": dev, "min": min(v for e, v in d.items() if e >= 0),
                         "max": max(v for e, v in d.items() if e >= 0)})
    for p, mk in zip(PARAMS, ["o", "s", "^", "d"]):
        for s in SEEDS:
            d = {int(r["epoch"]): float(r["sigmoid_value"]) for r in rows
                 if int(r["seed"]) == s and r["param_name"] == p}
            eps = sorted(e for e in d if e >= 0)
            ax.plot(eps, [d[e] for e in eps], marker=mk, ms=3, lw=1, alpha=0.8)
    ax.axhline(0.7311, color="k", ls="--", lw=1, label="sigmoid(1)=0.7311")
    ax.axhspan(0.715, 0.752, color="green", alpha=0.12, label="0.715–0.752 band")
    ax.set_xlabel("epoch")
    ax.set_ylabel("architecture param (sigmoid)")
    ax.set_title("D0.2 architecture parameter trajectories (batch=40960, 5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d02_arch_traj.png"), dpi=150)

    inband = sum(1 for r in summ if 0.715 <= r["final"] <= 0.752)
    print(f"D0.2: {len(summ)} seed-param series; final in [0.715,0.752]: {inband}/{len(summ)}")
    for p in PARAMS:
        fin = [r["final"] for r in summ if r["param"] == p]
        dev = [r["max_abs_dev"] for r in summ if r["param"] == p]
        print(f"  {p}: final {np.mean(fin):.4f}±{np.std(fin):.4f}, max|dev| {np.mean(dev):.4f}")
    return summ


# ---------------- D0.4 ----------------
def d04():
    for bs in ["40960", "4096"]:
        rows = []
        for s in SEEDS:
            try:
                rows += rd(f"D04_grad_stats_s{s}_b{bs}.csv")
            except FileNotFoundError:
                pass
        if not rows:
            print(f"D0.4 bs={bs}: NO DATA")
            continue
        per_dom = defaultdict(list)
        nsamp = defaultdict(list)
        for r in rows:
            per_dom[int(r["domain"])].append(float(r["snr"]))
            nsamp[int(r["domain"])].append(float(r["n_samples_avg"]))
        print(f"D0.4 bs={bs}  (per-epoch mean over {len(SEEDS)} seeds):")
        for d in sorted(per_dom):
            v = np.array(per_dom[d])
            print(f"  d{d}: SNR {v.mean():.4f}±{v.std():.4f}  "
                  f"n_samples_avg {np.mean(nsamp[d]):.0f}")
        # grad norms
        norms = defaultdict(list)
        for r in rows:
            norms[int(r["domain"])].append(float(r["mean_grad_norm"]))
        for d in sorted(norms):
            print(f"  d{d}: mean_grad_norm {np.mean(norms[d]):.5f}")
        cosr = []
        for s in SEEDS:
            try:
                cosr += rd(f"D04_grad_cosine_s{s}_b{bs}.csv")
            except FileNotFoundError:
                pass
        if cosr:
            by_pair = defaultdict(list)
            for r in cosr:
                by_pair[r["domain_pair"]].append(float(r["cos_mean"]))
            print("  cross-domain cosine (mean of per-epoch means ± std):")
            for p, v in sorted(by_pair.items()):
                print(f"    {p}: {np.mean(v):.4f}±{np.std(v):.4f}")
    # figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = np.arange(3)
    for k, (bs, col) in enumerate([("40960", "#1f77b4"), ("4096", "#ff7f0e")]):
        vals, errs = [], []
        for d in range(3):
            vv = []
            for s in SEEDS:
                try:
                    vv += [float(r["snr"]) for r in rd(f"D04_grad_stats_s{s}_b{bs}.csv")
                           if int(r["domain"]) == d]
                except FileNotFoundError:
                    pass
            vals.append(np.mean(vv) if vv else 0)
            errs.append(np.std(vv) / max(1, np.sqrt(len(vv))) if vv else 0)
        ax.bar(x + (k - 0.5) * width, vals, width, yerr=errs, capsize=3, label=f"batch={bs}", color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(["d0 (21%)", "d1 (40%)", "d2 (39%)"])
    ax.set_ylabel("gradient SNR  ||E[g]||²/E[||g||²]")
    ax.set_title("D0.4 per-domain gradient SNR by batch size (mean±SEM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d04_snr.png"), dpi=150)


# ---------------- D0.6 ----------------
def d06():
    rows = []
    for s in SEEDS:
        rows += rd(f"D06_curves_s{s}_b40960.csv")
    peaks = defaultdict(list)
    fig, ax = plt.subplots(figsize=(8, 5))
    shades = {0: ["#4c72b0", "#85b0d6"], 1: ["#55a868", "#9fd6a9"], 2: ["#c44e52", "#e39a9d"]}
    for d in range(3):
        for t in range(2):
            for s in SEEDS:
                rs = sorted([r for r in rows if int(r["seed"]) == s and int(r["domain"]) == d
                             and int(r["task"]) == t], key=lambda r: int(r["epoch"]))
                if not rs:
                    continue
                eps = [int(r["epoch"]) for r in rs]
                auc = [float(r["val_auc"]) for r in rs]
                peaks[(d, t)].append(eps[int(np.argmax(auc))])
                if s == SEEDS[0]:
                    ax.plot(eps, auc, color=shades[d][t], lw=1.2,
                            label=f"d{d}t{t}" + (" (seed2022)" if True else ""))
                else:
                    ax.plot(eps, auc, color=shades[d][t], lw=0.6, alpha=0.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC")
    ax.set_title("D0.6 per-(d,t) validation curves (seed 2022 bold, others thin)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d06_val_curves.png"), dpi=150)
    print("D0.6 peak-val-AUC epoch per (d,t), mean±std over seeds:")
    for k in sorted(peaks):
        v = peaks[k]
        print(f"  d{k[0]}t{k[1]}: {np.mean(v):.1f}±{np.std(v):.1f}  ({v})")
    allv = [e for v in peaks.values() for e in v]
    print(f"  dispersion over all 18 (d,t,seed) series: max-min={max(allv)-min(allv)}, std={np.std(allv):.2f}")
    return peaks


# ---------------- D0.7 ----------------
def d07():
    out = []
    rows = rd("D06_curves") if os.path.exists(os.path.join(R, "D06_curves.csv")) else None
    for s in SEEDS:
        curves = [r for r in rd(f"D06_curves_s{s}_b40960.csv")]
        tests = rd(f"D06_test_s{s}_b40960.csv")
        eps = sorted({int(r["epoch"]) for r in curves})
        mean_val = {}
        for e in eps:
            vs = [float(r["val_auc"]) for r in curves if int(r["epoch"]) == e]
            mean_val[e] = np.mean(vs)
        best_global = max(eps, key=lambda e: mean_val[e])
        # early stop epoch (patience=10): first epoch with 10 consecutive non-improvements
        best_so_far, stop_e = -1, None
        for e in eps:
            if mean_val[e] > best_so_far:
                best_so_far = mean_val[e]
            elif e - max(x for x in eps if x <= e and mean_val[x] == best_so_far) >= 10:
                stop_e = e
                break
        for d in range(3):
            for t in range(2):
                rs = {int(r["epoch"]): float(r["val_auc"]) for r in curves
                      if int(r["domain"]) == d and int(r["task"]) == t}
                te = {int(r["epoch"]): float(r["test_auc"]) for r in tests
                      if int(r["domain"]) == d and int(r["task"]) == t}
                own = max(eps, key=lambda e: rs[e])
                out.append({"seed": s, "domain": d, "task": t,
                            "best_epoch_own": own, "best_epoch_global": best_global,
                            "test_auc_at_own": te[own], "test_auc_at_global": te[best_global],
                            "delta": te[best_global] - te[own]})
    with open(os.path.join(R, "D07_earlystop.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "domain", "task", "best_epoch_own",
                                          "best_epoch_global", "test_auc_at_own",
                                          "test_auc_at_global", "delta"])
        w.writeheader()
        w.writerows(out)
    by = defaultdict(list)
    for r in out:
        by[(int(r["domain"]), int(r["task"]))].append(r)
    print("D0.7 (batch=40960): own-best vs global early stop")
    for k in sorted(by):
        rs = by[k]
        own_e = [int(r["best_epoch_own"]) for r in rs]
        glob_e = [int(r["best_epoch_global"]) for r in rs]
        dl = [float(r["delta"]) for r in rs]
        print(f"  d{k[0]}t{k[1]}: own epoch {np.mean(own_e):.1f}±{np.std(own_e):.1f} "
              f"(global {np.mean(glob_e):.1f}), delta(test) {np.mean(dl):+.4f}±{np.std(dl):.4f}")
    # figure: best-epoch dispersion
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = sorted(by)
    own_m = [np.mean([int(r["best_epoch_own"]) for r in by[k]]) for k in ks]
    own_s = [np.std([int(r["best_epoch_own"]) for r in by[k]]) for k in ks]
    glob_m = [np.mean([int(r["best_epoch_global"]) for r in by[k]]) for k in ks]
    ax.errorbar(range(len(ks)), own_m, yerr=own_s, fmt="o", capsize=4, label="per-(d,t) own best epoch")
    ax.plot(range(len(ks)), glob_m, "s--", color="red", label="global best epoch")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"d{k[0]}t{k[1]}" for k in ks])
    ax.set_ylabel("epoch of peak validation AUC")
    ax.set_title("D0.7 best-epoch dispersion (batch=40960, 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "d07_dispersion.png"), dpi=150)
    return out


if __name__ == "__main__":
    print("==== D0.2 ====")
    d02()
    print("==== D0.4 ====")
    d04()
    print("==== D0.6 ====")
    d06()
    print("==== D0.7 ====")
    d07()
