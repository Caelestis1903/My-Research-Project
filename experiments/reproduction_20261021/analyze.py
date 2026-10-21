#!/usr/bin/env python
"""Orchestrator-owned descriptive analysis; never selects a model using test data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SEEDS = [2022, 2023, 2024, 2025]
DATASETS = ["movielens", "kuairand"]
CONTRASTS = [("B", "O"), ("D", "B"), ("G", "B"), ("G", "D"),
             ("D", "O"), ("G", "O"), ("G", "Gtau0"),
             ("Gtau0", "D"), ("Dconst01", "D"), ("Dconst01", "B")]


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    mean = float(a.mean()) if n else np.nan
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    half = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return dict(n=n, mean=mean, sd=sd, ci95_low=mean-half,
                ci95_high=mean+half, positive=int((a > 0).sum()),
                negative=int((a < 0).sum()), minimum=float(a.min()) if n else np.nan,
                maximum=float(a.max()) if n else np.nan)


def collect(root, include_stage2=False):
    records, epochs, mechanisms = [], [], []
    paths = list((root / "formal").rglob("completion.json"))
    if include_stage2:
        paths += list((root / "stage2").rglob("completion.json"))
    for p in sorted(paths):
        c = json.loads(p.read_text())
        if not (c.get("valid") and c.get("formal")):
            continue
        identity = {k: c[k] for k in ("dataset", "arm", "seed")}
        row = dict(identity, attempt=str(p.parent), **c["test_metrics"])
        manifest = json.loads((p.parent / "data_manifest.json").read_text())
        for d in range(3):
            row[f"train_domain{d}_count"] = manifest["train_domain_counts"][str(d)]
        for cell, stats in manifest["cell_stats"]["train"].items():
            row[f"train_{cell}_positive_rate"] = stats["positive_rate"]
        for k in ["selected_epoch", "stopped_epoch", "steps", "wall_seconds",
                  "peak_memory_bytes", "best_validation_macro_auc", "checkpoint_sha256"]:
            row[k] = c.get(k)
        ep = pd.read_csv(p.parent / "epoch_metrics.csv")
        selected = ep.loc[ep.epoch == c["selected_epoch"]].iloc[0]
        for k, v in selected.items():
            if k.startswith("val_"):
                row[k] = v
        for k, v in identity.items():
            ep[k] = v
        epochs.append(ep)
        records.append(row)
        mp = p.parent / "mechanism_metrics.csv"
        if mp.exists():
            m = pd.read_csv(mp)
            for k, v in identity.items():
                m[k] = v
            mechanisms.append(m)
    if not records:
        raise RuntimeError("No completed formal runs")
    runs = pd.DataFrame(records)
    if runs.duplicated(["dataset", "arm", "seed"]).any():
        raise RuntimeError("Multiple valid attempts for one run; disposition must be explicit")
    for dataset in DATASETS:
        for arm in ["O", "B", "D", "G"] + (["Gtau0", "Dconst01"] if include_stage2 else []):
            actual = sorted(runs.loc[(runs.dataset == dataset) & (runs.arm == arm), "seed"])
            if actual != SEEDS:
                raise RuntimeError(f"Incomplete required matrix: {dataset}/{arm}: {actual}")
    return runs, pd.concat(epochs, ignore_index=True), pd.concat(mechanisms, ignore_index=True) if mechanisms else pd.DataFrame()


def summarize(runs, columns, contrasts):
    aggregate, paired = [], []
    for (dataset, arm), g in runs.groupby(["dataset", "arm"]):
        for metric in columns:
            if metric in g:
                aggregate.append(dict(dataset=dataset, arm=arm, metric=metric, **describe(g[metric])))
    for dataset, g in runs.groupby("dataset"):
        for treatment, reference in contrasts:
            a = g[g.arm == treatment].set_index("seed")
            b = g[g.arm == reference].set_index("seed")
            if sorted(a.index) != SEEDS or sorted(b.index) != SEEDS:
                continue
            for metric in columns:
                if metric not in a or metric not in b:
                    continue
                delta = a.loc[SEEDS, metric] - b.loc[SEEDS, metric]
                row = dict(dataset=dataset, contrast=f"{treatment}-{reference}",
                           metric=metric, **describe(delta))
                row.update({f"seed{s}": float(delta.loc[s]) for s in SEEDS})
                paired.append(row)
    return pd.DataFrame(aggregate), pd.DataFrame(paired)


def figures(runs, epochs, paired, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    arms = [a for a in ["O", "B", "D", "G", "Gtau0", "Dconst01"] if a in set(runs.arm)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(["macro_auc", "macro_logloss"]):
            ax = axes[i, j]
            g = runs[runs.dataset == dataset]
            means = [g.loc[g.arm == arm, metric].mean() for arm in arms]
            stds = [g.loc[g.arm == arm, metric].std(ddof=1) for arm in arms]
            ax.errorbar(range(len(arms)), means, yerr=stds, fmt="o", capsize=4, color="black")
            for seed in SEEDS:
                vals = g[g.seed == seed].set_index("arm").reindex(arms)[metric]
                ax.plot(range(len(arms)), vals, alpha=.4, linewidth=.8, label=str(seed))
            ax.set_xticks(range(len(arms)), arms, rotation=25 if len(arms)>4 else 0)
            ax.set_title(f"{dataset}: {metric} (mean ± seed SD)")
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"test_endpoints.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, contrast in enumerate(["D-B", "G-B", "G-D"]):
            ax = axes[i, j]
            values = np.array([[paired.loc[(paired.dataset == dataset) &
                (paired.contrast == contrast) & (paired.metric == f"d{d}t{t}_auc"), "mean"].iloc[0]
                for d in range(3)] for t in range(2)])
            limit = max(.001, abs(values).max())
            im = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for t in range(2):
                for d in range(3):
                    ax.text(d, t, f"{values[t, d]:+.4f}", ha="center", va="center",
                            color="white" if abs(values[t,d]) > limit*.65 else "black")
            ax.set_xticks(range(3), ["d0", "d1", "d2"])
            ax.set_yticks(range(2), ["task0", "task1"])
            ax.set_title(f"{dataset} {contrast}: mean ΔAUC")
            fig.colorbar(im, ax=ax, fraction=.04)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"per_cell_auc_deltas.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    curve_rows = []
    for ax, dataset in zip(axes, DATASETS):
        for arm in arms:
            g = epochs[(epochs.dataset == dataset) & (epochs.arm == arm)]
            stats = g.groupby("epoch").val_macro_auc.agg(["mean", "std", "count"])
            for epoch, r in stats.iterrows():
                curve_rows.append(dict(dataset=dataset, arm=arm, epoch=epoch, **r.to_dict()))
            # Only show points supported by all four seeds; stopping is reported separately.
            stats = stats[stats["count"] == 4]
            ax.plot(stats.index, stats["mean"], label=arm)
            ax.fill_between(stats.index, stats["mean"]-stats["std"], stats["mean"]+stats["std"], alpha=.12)
        ax.set_title(f"{dataset}: validation macro AUC")
        ax.set_xlabel("Epoch (only epochs with four seeds shown)")
        ax.grid(alpha=.2)
        ax.legend()
    pd.DataFrame(curve_rows).to_csv(out / "validation_curve_support.csv", index=False)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"validation_trajectories.{ext}", dpi=180)
    plt.close(fig)
    controls = ["G-Gtau0", "Gtau0-D", "Dconst01-D", "Dconst01-B"]
    if all(c in set(paired.contrast) for c in controls):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for i, dataset in enumerate(DATASETS):
            for j, metric in enumerate(["macro_auc", "macro_logloss"]):
                ax = axes[i,j]
                q = paired[(paired.dataset == dataset) & (paired.metric == metric)].set_index("contrast")
                for y, c in enumerate(controls):
                    r = q.loc[c]
                    ax.plot([r.ci95_low,r.ci95_high],[y,y],color="black",linewidth=1.4)
                    ax.scatter([r['mean']],[y],marker="D",s=28,color="black",zorder=3)
                    ax.scatter([r[f"seed{s}"] for s in SEEDS],[y+.12]*4,s=15,alpha=.65,color="tab:blue")
                ax.axvline(0,color="grey",linewidth=.8)
                if j == 0:
                    ax.axvspan(-.001,.001,color="grey",alpha=.12)
                ax.set_yticks(range(4),controls)
                ax.invert_yaxis()
                ax.set_title(f"{dataset}: Δ {metric}")
                ax.set_xlabel("Paired seed differences; diamond mean, line 95% t interval")
                ax.grid(axis="x",alpha=.2)
        for ext in ["png", "svg"]:
            fig.savefig(out / f"paired_control_contrasts.{ext}",dpi=180)
        plt.close(fig)


def supplemental(runs, epochs, mechanisms, paired, out):
    """Prespecified control checks and descriptive training/mechanism views."""
    rows = []
    for dataset in DATASETS:
        a = paired[(paired.dataset == dataset) & (paired.contrast == "Dconst01-D")]
        if not a.empty:
            auc = a[a.metric == "macro_auc"].iloc[0]
            loss = a[a.metric == "macro_logloss"].iloc[0]
            rows.append(dict(dataset=dataset, contrast="Dconst01-D",
                auc_ci95_low=auc.ci95_low, auc_margin=-.001,
                loss_ci95_high=loss.ci95_high, loss_margin=.002,
                descriptive_noninferiority=bool(auc.ci95_low > -.001 and loss.ci95_high < .002)))
    pd.DataFrame(rows).to_csv(out / "constant_weight_noninferiority.csv", index=False)
    # Performance-contrast identity; not additive causal effects on trajectories.
    decomposition = []
    for dataset in DATASETS:
        for metric in ["macro_auc", "macro_logloss", "pooled_task_mean_auc", "worst_cell_auc"]:
            q = paired[(paired.dataset == dataset) & (paired.metric == metric)].set_index("contrast")
            if all(k in q.index for k in ["G-D", "G-Gtau0", "Gtau0-D"]):
                for seed in SEEDS:
                    k = f"seed{seed}"
                    total, amplitude, path = (float(q.loc[c,k]) for c in ["G-D", "G-Gtau0", "Gtau0-D"])
                    assert abs(total-amplitude-path) < 1e-12
                    decomposition.append(dict(dataset=dataset, metric=metric, seed=seed,
                        G_minus_D=total, G_minus_Gtau0=amplitude, Gtau0_minus_D=path))
    pd.DataFrame(decomposition).to_csv(out / "gradient_contrast_decomposition.csv", index=False)
    real_columns = [f"train_real_d{d}t{t}_bce" for d in range(3) for t in range(2)]
    ep = epochs.copy()
    ep["train_real_macro_bce"] = ep[real_columns].mean(axis=1)
    stages = []
    contrasts = [(a,b) for a,b in CONTRASTS if a in set(runs.arm) and b in set(runs.arm)]
    for epoch in [1,5,10]:
        q = ep[ep.epoch == epoch]
        _, delta = summarize(q, ["val_macro_auc", "val_macro_logloss", "train_real_macro_bce"], contrasts)
        delta["epoch"] = epoch
        stages.append(delta)
    pd.concat(stages, ignore_index=True).to_csv(out / "paired_training_stage.csv", index=False)
    selected = ep.merge(runs[["dataset","arm","seed","selected_epoch"]], on=["dataset","arm","seed"])
    selected[selected.epoch == selected.selected_epoch].to_csv(out / "selected_epoch_details.csv", index=False)
    probes = []
    if not mechanisms.empty:
        for _, r in mechanisms.iterrows():
            w = json.loads(r.w)
            specificity = json.loads(r.specificity) if isinstance(r.specificity, str) else {"mean": np.nan}
            scales = np.asarray(json.loads(r.scales))
            norms = np.asarray(json.loads(r.norm_all))
            probes.append(dict(dataset=r.dataset, arm=r.arm, seed=r.seed, epoch=r.epoch, step=r.step,
                weight_mean=w["mean"], weight_sd=w["std"], weight_min=w["min"], weight_max=w["max"],
                specificity_mean=specificity["mean"], scale_min=float(scales.min()), scale_max=float(scales.max()),
                raw_norm_min=float(norms.min()), raw_norm_max=float(norms.max()),
                raw_norm_ratio=float(norms.max()/norms.min()) if norms.min()>0 else np.nan,
                auxiliary_gradient_norm=r.auxiliary_gradient_norm,
                adam_delta_norm=r.adam_delta_norm, adam_delta_raw_cosine=r.adam_delta_raw_cosine,
                conflict_rate=r.conflict_rate, real_generated_cosine_all=r.real_generated_cosine_all,
                real_generated_cosine_shared=r.real_generated_cosine_shared))
    pd.DataFrame(probes).to_csv(out / "mechanism_probe_summary.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=Path("/home_new/xzf_m3oe_reproduction_20261021"))
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "analysis")
    p.add_argument("--include-stage2", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, epochs, mechanisms = collect(args.input_root, args.include_stage2)
    arms = sorted(set(runs.arm))
    contrasts = [(a,b) for a,b in CONTRASTS if a in arms and b in arms]
    metric_cols = [k for k in runs if not k.startswith("val_") and k != "best_validation_macro_auc" and
                   (k.endswith("_auc") or k.endswith("_logloss"))]
    aggregate, paired = summarize(runs, metric_cols, contrasts)
    runs.to_csv(args.output / "all_seed_results.csv", index=False)
    epochs.to_csv(args.output / "all_epoch_metrics.csv", index=False)
    aggregate.to_csv(args.output / "aggregate_test_metrics.csv", index=False)
    paired.to_csv(args.output / "paired_test_deltas.csv", index=False)
    if not mechanisms.empty:
        mechanisms.to_csv(args.output / "all_mechanism_metrics.csv", index=False)
    _, validation = summarize(runs, ["val_macro_auc", "val_macro_logloss",
        "val_pooled_task_mean_auc", "val_worst_cell_auc"], contrasts)
    validation.to_csv(args.output / "paired_validation_deltas.csv", index=False)
    # Continuation screening is validation-only; the label is descriptive, not a p-value.
    screen = validation[validation.metric == "val_macro_auc"].copy()
    seed_columns = [f"seed{s}" for s in SEEDS]
    screen["practical_positive_seeds"] = (screen[seed_columns] >= .001).sum(axis=1)
    screen["practical_negative_seeds"] = (screen[seed_columns] <= -.001).sum(axis=1)
    screen["repeatable_practical_direction"] = (
        (screen.practical_positive_seeds >= 3) | (screen.practical_negative_seeds >= 3))
    screen.to_csv(args.output / "validation_screen.csv", index=False)
    training = runs[["dataset", "arm", "seed", "selected_epoch", "stopped_epoch",
        "steps", "wall_seconds", "peak_memory_bytes", "best_validation_macro_auc", "checkpoint_sha256", "attempt"]]
    training.to_csv(args.output / "training_receipts.csv", index=False)
    domains = []
    for dataset in DATASETS:
        g = runs[runs.dataset == dataset]
        for d in range(3):
            row = dict(dataset=dataset, domain=d,
                       training_n=int(g[f"train_domain{d}_count"].iloc[0]))
            for task in range(2):
                row[f"train_task{task}_positive_rate"] = g[f"train_d{d}t{task}_positive_rate"].iloc[0]
            for arm in arms:
                a = g[g.arm == arm]
                for metric in ["auc", "logloss"]:
                    row[f"{arm}_{metric}"] = a[f"domain{d}_macro_{metric}"].mean()
            for treatment, reference in contrasts:
                for metric in ["auc", "logloss"]:
                    if f"{treatment}_{metric}" in row and f"{reference}_{metric}" in row:
                        row[f"{treatment}-{reference}_{metric}"] = row[f"{treatment}_{metric}"]-row[f"{reference}_{metric}"]
            domains.append(row)
    pd.DataFrame(domains).to_csv(args.output / "domain_size_performance.csv", index=False)
    supplemental(runs, epochs, mechanisms, paired, args.output)
    figures(runs, epochs, paired, args.output)
    receipt = dict(completed_formal_runs=len(runs), seeds=SEEDS, datasets=DATASETS,
                   arms=arms, input_root=str(args.input_root),
                   statistics="seed mean/sample SD; paired t 95% descriptive interval, df=3 for n=4; no multiplicity correction",
                   interpretation="Raw signed deltas: higher AUC and lower LogLoss are favorable. Cell comparisons are exploratory.")
    (args.output / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
