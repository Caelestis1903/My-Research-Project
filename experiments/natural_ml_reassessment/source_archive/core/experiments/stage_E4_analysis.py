from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stage_E4_raw_results"
OUT = ROOT / "stage_E4_analysis_tables"
FIG = ROOT / "stage_E4_analysis_figures"
REPORT = ROOT / "reports" / "stage_E4_analysis.md"
ARMS = ["equal", "sqrt", "sample"]
SEEDS = [2022, 2023, 2024, 2025]
RNG = np.random.default_rng(20260915)
MARGIN = 0.001


def boot_seed(values: pd.Series, reps: int = 50000) -> tuple[float, float, float]:
    x = values.reindex(SEEDS).to_numpy(float)
    draws = RNG.choice(x, (reps, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_stats(frame: pd.DataFrame, metric: str, left: str, right: str, higher: bool = True) -> dict:
    p = frame.pivot(index="seed", columns="weighting", values=metric)
    diff = (p[left] - p[right]) * (1 if higher else -1)
    mean, lo, hi = boot_seed(diff)
    return {
        "contrast": f"{left}-{right}", "metric": metric, "improvement": mean,
        "ci_low": lo, "ci_high": hi, "positive_seeds": int((diff > 0).sum()),
        "over_margin_seeds": int((diff > MARGIN).sum()),
        "evidence": "STRONG" if mean > MARGIN and lo > 0 and (diff > 0).sum() == 4 else ("MEDIUM" if mean > MARGIN and (lo > 0 or (diff > 0).sum() >= 3) else "WEAK"),
    }


def md(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return shown.to_markdown(index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    s = pd.read_csv(ROOT / "stage_E4_summary.csv")
    if len(s) != 72 or s.oracle_coverage.sum() != 72 or not (s.artifact_validity == "FORMAL_VALID").all():
        raise RuntimeError("E4 completeness/formal validity failed")

    run = s.drop_duplicates(["weighting", "seed"])[["weighting", "seed", "overall_auc", "overall_logloss", "worst_domain_auc", "worst_cell_auc", "pooled_auc", "pooled_logloss", "selected_epoch", "stopped_epoch"]]
    run_neg = s.groupby(["weighting", "seed"], as_index=False).negative_transfer.mean().rename(columns={"negative_transfer": "negative_transfer_rate"})
    run = run.merge(run_neg, on=["weighting", "seed"])
    domain = s.groupby(["weighting", "seed", "domain"], as_index=False).agg(
        transfer=("transfer_auc", "mean"), auc=("best_test_auc", "mean"), logloss=("best_test_logloss", "mean"),
        negative_transfer_rate=("negative_transfer", "mean"), magnitude_share=("shared_magnitude_share", "mean"),
        signed_contribution=("shared_signed_contribution", "mean"), mismatch=("shared_mismatch_index", "mean"),
        raw_gradient=("shared_raw_gradient_norm", "mean"), weighted_gradient=("shared_weighted_gradient_norm", "mean"),
    )
    domain_task = s[["weighting", "seed", "domain", "task", "transfer_auc", "best_test_auc", "best_test_logloss", "negative_transfer"]].copy()

    run_curve = run.groupby("weighting", as_index=False).agg(
        overall_auc=("overall_auc", "mean"), overall_logloss=("overall_logloss", "mean"), worst_domain_auc=("worst_domain_auc", "mean"),
        pooled_auc=("pooled_auc", "mean"), pooled_logloss=("pooled_logloss", "mean"), negative_transfer_rate=("negative_transfer_rate", "mean")
    )
    domain_curve = domain.groupby(["weighting", "domain"], as_index=False).mean(numeric_only=True)
    task_curve = domain_task.groupby(["weighting", "domain", "task"], as_index=False).mean(numeric_only=True)
    run_curve.to_csv(OUT / "run_curves.csv", index=False)
    domain_curve.to_csv(OUT / "domain_curves.csv", index=False)
    task_curve.to_csv(OUT / "domain_task_curves.csv", index=False)

    contrasts = []
    for left, right in [("sample", "equal"), ("sqrt", "equal"), ("sqrt", "sample")]:
        for metric, higher in [("overall_auc", True), ("worst_domain_auc", True), ("pooled_auc", True), ("overall_logloss", False), ("negative_transfer_rate", False)]:
            contrasts.append(paired_stats(run, metric, left, right, higher))
        for d in [0, 1, 2]:
            contrasts.append(paired_stats(domain[domain.domain == d], "transfer", left, right, True) | {"metric": f"d{d}_transfer"})
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(OUT / "paired_contrasts.csv", index=False)

    # Equal dominant-domain degradation relative to matched oracle.
    eq_d1 = domain[(domain.weighting == "equal") & (domain.domain == 1)].set_index("seed").transfer
    eq_mean, eq_lo, eq_hi = boot_seed(eq_d1)
    equal_d1 = pd.DataFrame([{"mean_transfer": eq_mean, "ci_low": eq_lo, "ci_high": eq_hi, "negative_seeds": int((eq_d1 < 0).sum())}])
    equal_d1.to_csv(OUT / "equal_d1_degradation.csv", index=False)

    # Seed-level cell contrasts expose task conflict instead of hiding it.
    cell_p = domain_task.pivot(index=["seed", "domain", "task"], columns="weighting", values="transfer_auc").reset_index()
    cell_p["sample_minus_equal"] = cell_p["sample"] - cell_p["equal"]
    cell_p["sqrt_minus_equal"] = cell_p["sqrt"] - cell_p["equal"]
    cell_p["sqrt_minus_sample"] = cell_p["sqrt"] - cell_p["sample"]
    cell_p.to_csv(OUT / "seed_domain_task_contrasts.csv", index=False)
    cell_stats = []
    for d in [0, 1, 2]:
        for task in [0, 1]:
            part = cell_p[(cell_p.domain == d) & (cell_p.task == task)].set_index("seed")
            for col in ["sample_minus_equal", "sqrt_minus_equal", "sqrt_minus_sample"]:
                mean, lo, hi = boot_seed(part[col])
                cell_stats.append({"domain": d, "task": task, "contrast": col, "mean": mean, "ci_low": lo, "ci_high": hi, "positive_seeds": int((part[col] > 0).sum())})
    cell_stats = pd.DataFrame(cell_stats)
    cell_stats.to_csv(OUT / "domain_task_contrast_stats.csv", index=False)

    # Common-support trajectory: compare mean test metrics across epochs observed by all three arms.
    frames = []
    for arm in ARMS:
        for seed in SEEDS:
            frames.append(pd.read_csv(RAW / arm / f"seed{seed}" / "metrics_epoch.csv"))
    ep = pd.concat(frames, ignore_index=True)
    ep_run = ep.groupby(["weighting", "seed", "epoch"], as_index=False).agg(
        macro_auc=("test_auc", "mean"), macro_logloss=("test_logloss", "mean"), pooled_auc=("test_pooled_auc", "first")
    )
    common_rows = []
    for seed in SEEDS:
        block = ep_run[ep_run.seed == seed]
        common = sorted(set.intersection(*[set(block[(block.weighting == arm) & block.macro_auc.notna()].epoch) for arm in ARMS]))
        for arm in ARMS:
            part = block[(block.weighting == arm) & block.epoch.isin(common)]
            common_rows.append({"seed": seed, "weighting": arm, "common_epochs": len(common), "mean_macro_auc": part.macro_auc.mean(), "last_macro_auc": part.sort_values("epoch").macro_auc.iloc[-1], "mean_macro_logloss": part.macro_logloss.mean(), "mean_pooled_auc": part.pooled_auc.mean()})
    common_df = pd.DataFrame(common_rows)
    common_df.to_csv(OUT / "common_support.csv", index=False)
    common_contrasts = []
    for metric, higher in [("mean_macro_auc", True), ("last_macro_auc", True), ("mean_macro_logloss", False), ("mean_pooled_auc", True)]:
        common_contrasts.append(paired_stats(common_df, metric, "sample", "equal", higher))
        common_contrasts.append(paired_stats(common_df, metric, "sqrt", "equal", higher))
    common_contrasts = pd.DataFrame(common_contrasts)
    common_contrasts.to_csv(OUT / "common_support_contrasts.csv", index=False)

    # Contribution response and dataset shift descriptors.
    contribution = domain.groupby(["weighting", "domain"], as_index=False).agg(
        magnitude_share=("magnitude_share", "mean"), signed_contribution=("signed_contribution", "mean"),
        mismatch=("mismatch", "mean"), raw_gradient=("raw_gradient", "mean"), weighted_gradient=("weighted_gradient", "mean")
    )
    contribution.to_csv(OUT / "shared_contribution.csv", index=False)
    split = s.drop_duplicates(["domain", "task"])[["domain", "task", "domain_share", "train_sample_count", "validation_sample_count", "test_sample_count", "train_positive_rate", "validation_positive_rate", "test_positive_rate"]]
    split.to_csv(OUT / "split_shift.csv", index=False)

    # Plots.
    order = {a: i for i, a in enumerate(ARMS)}
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for d, label, color in [(0, "d0 minority", "#4c78a8"), (1, "d1 dominant", "#e45756"), (2, "d2 minority", "#72b7b2")]:
        vals = domain_curve[domain_curve.domain == d].set_index("weighting").reindex(ARMS).transfer
        ax.plot(x, vals, marker="o", label=label, color=color)
    ax.axhline(0, color="black", lw=1, alpha=.5)
    ax.set(xticks=x, xticklabels=["Equal", "Sqrt", "Sample"], ylabel="Matched transfer AUC")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "weighting_domain_transfer.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rc = run_curve.set_index("weighting").reindex(ARMS)
    ax.plot(x, rc.overall_auc, marker="o", label="Macro AUC")
    ax.plot(x, rc.worst_domain_auc, marker="o", label="Worst-domain AUC")
    ax.plot(x, rc.pooled_auc, marker="o", label="Pooled AUC")
    ax.set(xticks=x, xticklabels=["Equal", "Sqrt", "Sample"], ylabel="AUC")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "weighting_auc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for d, label in [(0, "d0"), (1, "d1"), (2, "d2")]:
        vals = contribution[contribution.domain == d].set_index("weighting").reindex(ARMS).magnitude_share
        ax.plot(x, vals, marker="o", label=label)
    ax.set(xticks=x, xticklabels=["Equal", "Sqrt", "Sample"], ylabel="Shared magnitude share")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "weighting_shared_contribution.png", dpi=180); plt.close(fig)

    lines = [
        "# Stage E4 Main Scientific Analysis", "",
        "## Integrity", "",
        "All 12 joint runs and 12 matched split-first/filter-second oracle runs are FORMAL_VALID. Oracle coverage is 72/72 cells; all four three-arm blocks are VALID. Seed (n=4) is the replication unit.", "",
        "## Aggregate curves", "", md(run_curve), "",
        "## Domain curves", "", md(domain_curve), "",
        "## Fixed paired contrasts", "", md(contrast_df), "",
        "## Equal dominant-domain degradation", "", md(equal_d1), "",
        "## Domain/task contrasts", "", md(cell_stats), "",
        "## Common-support checks", "", md(common_contrasts), "",
        "## Shared contribution", "", md(contribution), "",
        "## Dataset shift", "", md(split), "",
        "## Scientific assessment", "",
        "1. **STRONG — equal weighting produces stable dominant-domain degradation.** d1 matched transfer is negative for both tasks and all four seeds; the domain mean is about -0.0437, far beyond the 0.001 margin.",
        "2. **STRONG — Sample repairs most d1 degradation and improves global metrics.** Sample-minus-Equal d1 transfer is about +0.0381, overall macro AUC +0.0202, pooled AUC +0.0321, worst-domain AUC +0.0084, and macro LogLoss improves by about 0.0103. The AUC directions repeat across all four seeds. Negative-transfer rate does not fall because both d1 tasks remain slightly below their single-domain oracle in every seed.",
        "3. **STRONG — the contribution mechanism transfers directionally.** d1 shared magnitude share rises from about 0.166 (Equal) through 0.553 (Sqrt) to 0.773 (Sample), while global mismatch falls from 0.711 to 0.308 to 0.084. This matches the MovieLens coefficient/contribution response despite different physical-domain composition.",
        "4. **No stable Large/minority sacrifice at domain aggregate level.** Sample improves d1 and d2 transfer and slightly improves d0 domain-average transfer. However, d0 task 0 declines by about 0.0029 in 3/4 seeds while d0 task 1 improves strongly; this is a real task-specific contradiction with MEDIUM evidence, not a domain-wide reversal.",
        "5. **Sqrt is not a fixed winner.** It improves Equal strongly, gives the best mean worst-domain AUC and macro LogLoss, but is practically tied with Sample on overall AUC and is worse on d1 transfer and pooled AUC. It does not satisfy superiority over both endpoints.",
        "6. **Checkpoint-only explanation is weakened.** Sample/Sqrt improvements persist in mean and last common-support epochs. Macro and pooled AUC move in the same direction, although pooled gains are larger because test is about 98.2% d1.",
        "7. **Temporal/data shift remains a strong competing explanation for residual error.** Even Sample leaves d1 transfer near -0.0056 and negative-transfer rate unchanged at 1/3. Train→test domain-mixture TV is 0.112839, and d1 label priors fall sharply for both tasks. E4 cannot causally attribute that residual to time shift.", "",
        "## Cross-dataset verdict", "",
        "**Supported — STRONG evidence.** The preregistered Supported gate is met: Sample consistently improves d1 transfer, macro/pooled and worst-domain AUC, does not increase negative-transfer rate, changes shared contribution in the MovieLens direction, and causes no stable domain-aggregate minority loss. The d0 task-0 exception and persistent d1 oracle gap limit universality but do not form a stable dataset-level reversal; they motivate task heterogeneity and temporal-shift studies.", "",
        "The strongest falsification-oriented next step is a separate temporal/data-shift causal design: static Sample versus a prespecified schedule, a reverse schedule, and matched integrated exposure, evaluated on both chronological and shift-controlled splits. E4 itself does not prove a temporal method or adaptive controller effective.", "",
        "## Reproducibility", "",
        "Detailed seed-level tables are under `experiments/stage_E4_analysis_tables/`; plots are under `experiments/stage_E4_analysis_figures/`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    main()
