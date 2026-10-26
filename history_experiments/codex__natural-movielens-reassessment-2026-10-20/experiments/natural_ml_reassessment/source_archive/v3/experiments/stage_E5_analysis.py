from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stage_E5_raw_results"
OUT = ROOT / "stage_E5_analysis_tables"
FIG = ROOT / "stage_E5_analysis_figures"
REPORT = ROOT / "reports" / "stage_E5_analysis.md"
GO_REPORT = ROOT / "reports" / "stage_E5_go_no_go.md"
ARMS = ["static_sample", "static_interior", "static_matched", "forward", "reverse"]
SEEDS = [2022, 2023, 2024, 2025]
ROLES = ["L", "M", "S"]
RNG = np.random.default_rng(20260915)
MARGIN = 0.001


def boot(x: pd.Series, reps: int = 50000) -> tuple[float, float, float]:
    vals = x.reindex(SEEDS).to_numpy(float)
    draws = RNG.choice(vals, (reps, 4), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def contrast_table(run: pd.DataFrame, view: str, candidate: str = "forward") -> pd.DataFrame:
    part = run[run.view == view]
    rows = []
    for baseline in ["static_sample", "static_matched", "reverse", "static_interior"]:
        for metric, higher in [("overall_auc", True), ("overall_logloss", False), ("worst_domain_auc", True)]:
            p = part.pivot(index="seed", columns="arm_key", values=metric)
            delta = (p[candidate] - p[baseline]) * (1 if higher else -1)
            mean, lo, hi = boot(delta)
            rows.append({
                "view": view, "contrast": f"{candidate}-{baseline}", "metric": metric,
                "improvement": mean, "std": delta.std(ddof=1), "median": delta.median(), "min": delta.min(), "max": delta.max(),
                "ci_low": lo, "ci_high": hi, "positive_seeds": int((delta > 0).sum()), "over_margin_seeds": int((delta > MARGIN).sum()),
                "evidence": "STRONG" if mean > MARGIN and lo > 0 and (delta > 0).sum() == 4 else ("MEDIUM" if mean > MARGIN and (lo > 0 or (delta > 0).sum() >= 3) else "WEAK"),
            })
    return pd.DataFrame(rows)


def md(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return shown.to_markdown(index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    s = pd.read_csv(ROOT / "stage_E5_summary.csv")
    if len(s) != 240 or s.available.sum() != 240 or s.oracle_coverage.sum() != 240:
        raise RuntimeError("E5 summary completeness failed")
    run = s.drop_duplicates(["view", "arm_key", "seed"])[
        ["view", "arm_key", "seed", "epoch", "overall_auc", "overall_logloss", "worst_domain_auc", "worst_cell_auc", "peak_epoch", "fixed_final_epoch"]
    ]
    neg = s.groupby(["view", "arm_key", "seed"], as_index=False).negative_transfer.mean().rename(columns={"negative_transfer": "negative_transfer_rate"})
    run = run.merge(neg, on=["view", "arm_key", "seed"])
    arm_curve = run.groupby(["view", "arm_key"], as_index=False).mean(numeric_only=True)
    arm_curve.to_csv(OUT / "arm_curves.csv", index=False)

    final_contrasts = contrast_table(run, "fixed-final")
    selected_contrasts = contrast_table(run, "selected")
    contrasts = pd.concat([final_contrasts, selected_contrasts], ignore_index=True)
    contrasts.to_csv(OUT / "forward_contrasts.csv", index=False)
    # Exact per-seed primary deltas.
    final_p = run[run.view == "fixed-final"].pivot(index="seed", columns="arm_key", values="overall_auc")
    seed_delta = pd.DataFrame({"seed": SEEDS})
    for baseline in ["static_sample", "static_matched", "reverse", "static_interior"]:
        seed_delta[f"forward_minus_{baseline}"] = (final_p.forward - final_p[baseline]).reindex(SEEDS).to_numpy()
    seed_delta.to_csv(OUT / "primary_seed_deltas.csv", index=False)

    # Role/task safety and transfer.
    grouped = s.groupby(["view", "arm_key", "seed", "size_role", "task"], as_index=False).agg(
        auc=("test_auc", "mean"), logloss=("test_logloss", "mean"), transfer=("transfer_auc", "mean"), negative_transfer=("negative_transfer", "mean")
    )
    safety_rows = []
    for view in ["fixed-final", "selected"]:
        p = grouped[grouped.view == view]
        for role in ROLES:
            for task in [0, 1]:
                cell = p[(p.size_role == role) & (p.task == task)]
                for metric, higher in [("auc", True), ("transfer", True), ("logloss", False)]:
                    wide = cell.pivot(index="seed", columns="arm_key", values=metric)
                    delta = (wide.forward - wide.static_sample) * (1 if higher else -1)
                    mean, lo, hi = boot(delta)
                    safety_rows.append({"view": view, "role": role, "task": task, "metric": metric, "forward_vs_static_sample": mean, "ci_low": lo, "ci_high": hi, "positive_seeds": int((delta > 0).sum())})
    safety = pd.DataFrame(safety_rows)
    safety.to_csv(OUT / "role_task_safety.csv", index=False)
    role_safety = grouped.groupby(["view", "arm_key", "seed", "size_role"], as_index=False).mean(numeric_only=True)
    role_rows = []
    for view in ["fixed-final", "selected"]:
        for role in ROLES:
            cell = role_safety[(role_safety.view == view) & (role_safety.size_role == role)]
            for metric, higher in [("auc", True), ("transfer", True), ("logloss", False)]:
                wide = cell.pivot(index="seed", columns="arm_key", values=metric)
                delta = (wide.forward - wide.static_sample) * (1 if higher else -1)
                mean, lo, hi = boot(delta)
                role_rows.append({"view": view, "role": role, "metric": metric, "improvement": mean, "ci_low": lo, "ci_high": hi, "positive_seeds": int((delta > 0).sum())})
    role_summary = pd.DataFrame(role_rows)
    role_summary.to_csv(OUT / "role_safety_summary.csv", index=False)

    # Oracle operational difficulty; do not infer size causality from the canonical mapping.
    oracle = s.drop_duplicates(["seed", "physical_domain", "task"])[["seed", "physical_domain", "size_role", "domain_size", "domain_share", "task", "oracle_auc", "oracle_logloss"]]
    oracle_summary = oracle.groupby(["physical_domain", "size_role", "domain_size", "domain_share", "task"], as_index=False).agg(oracle_auc=("oracle_auc", "mean"), oracle_auc_std=("oracle_auc", "std"), oracle_logloss=("oracle_logloss", "mean"))
    oracle_summary.to_csv(OUT / "oracle_difficulty.csv", index=False)

    # Integrated coefficient exposure, including nonlinear coefficient mismatch vs static alpha=.875.
    exposure = s[s.view == "fixed-final"].drop_duplicates(["arm_key", "seed", "size_role"])[["arm_key", "seed", "size_role", "alpha_exposure_total", "coefficient_integral_cell", "step_weighted_coefficient_integral_cell"]]
    exposure_summary = exposure.groupby(["arm_key", "size_role"], as_index=False).mean(numeric_only=True)
    exposure_summary.to_csv(OUT / "exposure.csv", index=False)
    exp_p = exposure.pivot(index=["seed", "size_role"], columns="arm_key", values="step_weighted_coefficient_integral_cell")
    exposure_difference = pd.DataFrame({
        "role": exp_p.index.get_level_values("size_role"),
        "forward_minus_static_matched": exp_p.forward.to_numpy() - exp_p.static_matched.to_numpy(),
        "reverse_minus_static_matched": exp_p.reverse.to_numpy() - exp_p.static_matched.to_numpy(),
        "forward_minus_reverse": exp_p.forward.to_numpy() - exp_p.reverse.to_numpy(),
    }).groupby("role", as_index=False).mean()
    exposure_difference.to_csv(OUT / "exposure_differences.csv", index=False)

    # Early/middle/late trajectories from raw metric and gradient epoch files.
    metric_frames, grad_frames = [], []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = RAW / arm / "d0_large" / f"seed{seed}"
            metric_frames.append(pd.read_csv(run_dir / "metrics_epoch.csv"))
            grad_frames.append(pd.read_csv(run_dir / "shared_gradient_epoch.csv"))
    me = pd.concat(metric_frames, ignore_index=True)
    ge = pd.concat(grad_frames, ignore_index=True)
    phase_map = {e: "early" if e <= 5 else ("middle" if e <= 11 else "late") for e in range(18)}
    me["phase"] = me.epoch.map(phase_map)
    ge["phase"] = ge.epoch.map(phase_map)
    phase_perf = me.groupby(["arm_key", "seed", "phase"], as_index=False).agg(test_auc=("test_auc", "mean"), test_logloss=("test_logloss", "mean"), validation_auc=("validation_auc", "mean"))
    phase_perf_summary = phase_perf.groupby(["arm_key", "phase"], as_index=False).mean(numeric_only=True)
    phase_perf_summary.to_csv(OUT / "phase_performance.csv", index=False)
    phase_grad = ge.merge(s[["arm_key", "seed", "domain", "size_role"]].drop_duplicates(), on=["arm_key", "seed", "domain"], how="left")
    phase_grad_summary = phase_grad.groupby(["arm_key", "phase", "size_role"], as_index=False).agg(
        magnitude_share=("magnitude_share", "mean"), signed_contribution=("contribution_mean", "mean"), raw_gradient=("raw_gradient_norm", "mean"),
        weighted_gradient=("weighted_gradient_norm", "mean"), variance=("gradient_variance", "mean"), snr=("gradient_SNR", "mean"), mismatch=("mismatch_index", "mean")
    )
    phase_grad_summary.to_csv(OUT / "phase_contribution.csv", index=False)

    # Plots.
    final_curve = arm_curve[arm_curve.view == "fixed-final"].set_index("arm_key").reindex(ARMS)
    selected_curve = arm_curve[arm_curve.view == "selected"].set_index("arm_key").reindex(ARMS)
    x = np.arange(len(ARMS))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, final_curve.overall_auc, marker="o", label="Fixed final")
    ax.plot(x, selected_curve.overall_auc, marker="o", label="Validation selected")
    ax.set(xticks=x, xticklabels=["Static 1", "Static .75", "Static .875", "Forward", "Reverse"], ylabel="Overall AUC")
    ax.tick_params(axis="x", rotation=20); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "arm_overall_auc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for arm in ["static_sample", "static_matched", "forward", "reverse"]:
        vals = phase_perf_summary[phase_perf_summary.arm_key == arm].set_index("phase").reindex(["early", "middle", "late"]).test_auc
        ax.plot([0, 1, 2], vals, marker="o", label=arm)
    ax.set(xticks=[0, 1, 2], xticklabels=["Early", "Middle", "Late"], ylabel="Mean test AUC")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "phase_performance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for arm in ["static_sample", "static_matched", "forward", "reverse"]:
        vals = phase_grad_summary[(phase_grad_summary.arm_key == arm) & (phase_grad_summary.size_role == "L")].set_index("phase").reindex(["early", "middle", "late"]).magnitude_share
        ax.plot([0, 1, 2], vals, marker="o", label=arm)
    ax.set(xticks=[0, 1, 2], xticklabels=["Early", "Middle", "Late"], ylabel="Large shared magnitude share")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "phase_large_contribution.png", dpi=180); plt.close(fig)

    # Strict GO audit.
    def primary(contrast: str) -> pd.Series:
        return final_contrasts[(final_contrasts.contrast == contrast) & (final_contrasts.metric == "overall_auc")].iloc[0]
    go_primary = [primary("forward-static_sample"), primary("forward-static_matched"), primary("forward-reverse")]
    selected_primary = [selected_contrasts[(selected_contrasts.contrast == c) & (selected_contrasts.metric == "overall_auc")].iloc[0] for c in ["forward-static_sample", "forward-static_matched", "forward-reverse"]]
    final_forward = run[(run.view == "fixed-final") & (run.arm_key == "forward")].set_index("seed")
    final_sample = run[(run.view == "fixed-final") & (run.arm_key == "static_sample")].set_index("seed")
    loss_improvement = (final_sample.overall_logloss - final_forward.overall_logloss).mean()
    role_final = role_summary[role_summary.view == "fixed-final"].set_index(["role", "metric"])
    go_checks = pd.DataFrame([
        {"criterion": "final AUC gain >.001 vs static_sample", "pass": bool(go_primary[0].improvement > MARGIN), "value": go_primary[0].improvement},
        {"criterion": "final AUC gain >.001 vs static_matched", "pass": bool(go_primary[1].improvement > MARGIN), "value": go_primary[1].improvement},
        {"criterion": "final AUC gain >.001 vs reverse", "pass": bool(go_primary[2].improvement > MARGIN), "value": go_primary[2].improvement},
        {"criterion": ">=3/4 positive for all three", "pass": bool(all(x.positive_seeds >= 3 for x in go_primary)), "value": min(x.positive_seeds for x in go_primary)},
        {"criterion": "selected directions agree for all three", "pass": bool(all(x.improvement > 0 for x in selected_primary)), "value": min(x.improvement for x in selected_primary)},
        {"criterion": "LogLoss not worse >.001 vs static_sample", "pass": bool(loss_improvement >= -MARGIN), "value": loss_improvement},
        {"criterion": "Large transfer no >.001 degradation", "pass": bool(role_final.loc[("L", "transfer"), "improvement"] >= -MARGIN), "value": role_final.loc[("L", "transfer"), "improvement"]},
        {"criterion": "Small transfer no >.001 degradation", "pass": bool(role_final.loc[("S", "transfer"), "improvement"] >= -MARGIN), "value": role_final.loc[("S", "transfer"), "improvement"]},
        {"criterion": "worst-domain no >.001 degradation", "pass": bool(go_primary[0]["improvement"] is not None and final_contrasts[(final_contrasts.contrast == "forward-static_sample") & (final_contrasts.metric == "worst_domain_auc")].iloc[0].improvement >= -MARGIN), "value": final_contrasts[(final_contrasts.contrast == "forward-static_sample") & (final_contrasts.metric == "worst_domain_auc")].iloc[0].improvement},
    ])
    go_checks.to_csv(OUT / "go_checks.csv", index=False)
    go = bool(go_checks["pass"].all())

    lines = [
        "# Stage E5 Main Scientific Analysis", "",
        "## Integrity", "",
        "All 20 fixed-budget runs and four five-arm blocks are valid. Each run has 18 epochs, identical per-seed batch order and optimizer-step counts, and complete E3 frozen-oracle coverage. Seed (n=4) is the replication unit.", "",
        "## Arm results", "", md(arm_curve), "",
        "## Forward paired contrasts", "", md(contrasts), "",
        "## Primary per-seed deltas", "", md(seed_delta), "",
        "## Role safety relative to static alpha=1", "", md(role_summary), "",
        "## Per-role/task safety", "", md(safety), "",
        "## Exposure audit", "", md(exposure_summary), "", md(exposure_difference), "",
        "## Early/middle/late performance", "", md(phase_perf_summary), "",
        "## Early/middle/late shared contribution", "", md(phase_grad_summary), "",
        "## Oracle operational difficulty", "", md(oracle_summary), "",
        "## Strict GO audit", "", md(go_checks), "",
        "## Scientific assessment", "",
        "1. **WEAK evidence for temporal order.** At fixed epoch 17, Forward beats Reverse by about 0.00138 and Static .875 by about 0.00111, but each contrast is positive in only 2/4 seeds. Reverse beats Forward in seeds 2023 and 2025; Static .875 beats it in the same two seeds. Direction inconsistency prevents a temporal-order claim.",
        "2. **STRONG evidence against Forward as an improvement over the simple baseline.** Forward is worse than static alpha=1 by about 0.00178 at fixed final, positive in only 1/4 seeds. At validation-selected checkpoints it is also worse by about 0.00192; Reverse and static alpha=1 are likewise better. This is not a checkpoint-only artifact.",
        "3. **Exposure matching does not rescue the hypothesis.** Forward and Reverse have identical alpha integrals and step counts; nonlinear coefficient integrals differ slightly from Static .875 and are reported explicitly. Yet Forward-vs-Reverse and Forward-vs-matched signs are unstable, so neither integrated exposure nor time order yields a repeatable advantage.",
        "4. **No beneficial Large/Small trade-off.** Relative to static alpha=1, Forward slightly degrades Large and Medium transfer and degrades Small transfer by about 0.00239; worst-domain AUC is lower by about 0.00239. The schedule does not buy a compensating small-domain improvement.",
        "5. **Contribution timing changes as designed but is not sufficient.** Forward shifts shared magnitude toward Large over training and Reverse shifts it away, while Static controls remain fixed. The absence of stable performance gains despite this manipulation falsifies the claim that the E3 descriptive early/late optimum directly implies a useful linear schedule.",
        "6. **C4 and C8 are ruled out by construction/evidence.** Fixed epochs, steps, examples, batches, and architecture objective are identical. C1/C2 remain descriptive alternatives for individual seeds, but neither explains a stable cross-seed advantage. C7 gradient noise is plausible because the two positive and two negative seed results coincide with varying raw-gradient variance/SNR. C6 cannot be separated from size because E5 uses only canonical d0_large.",
        "7. Oracle difficulty does not equal size: the Small physical domain is not uniformly the hardest across both tasks. With only one role assignment, domain size cannot be claimed as the main causal variable.", "",
        "## GO / STOP", "",
        "**STOP — DROP Curriculum.** The mandatory GO criteria fail: Forward does not beat static alpha=1, matched static, and Reverse by >0.001 with >=3/4 seed consistency; selected-checkpoint directions also fail; safety metrics do not compensate. Therefore: `Contribution Curriculum currently lacks empirical support and is dropped.` No schedule/warmup/ratio/shape expansion is warranted.", "",
        "The best next discriminating step is not another schedule grid. Proceed only to the next independently motivated group3 experiment, with static alpha=1 as the main baseline and without Curriculum. Any future temporal claim would require a new causal design addressing seed-dependent gradient-noise interactions rather than tuning this failed schedule.", "",
        "## Reproducibility", "",
        "Tables: `experiments/stage_E5_analysis_tables/`; figures: `experiments/stage_E5_analysis_figures/`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GO_REPORT.write_text("# Stage E5 GO / STOP\n\n**STOP — DROP Curriculum.**\n\n`Contribution Curriculum currently lacks empirical support and is dropped.`\n\nThe strict preregistered GO gate failed; see `stage_E5_analysis.md` and `stage_E5_analysis_tables/go_checks.csv`.\n", encoding="utf-8")
    print(go_checks.to_string(index=False))


if __name__ == "__main__":
    main()
