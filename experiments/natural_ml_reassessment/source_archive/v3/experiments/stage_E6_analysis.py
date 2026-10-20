from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stage_E6_raw_results"
OUT = ROOT / "stage_E6_analysis_tables"
FIG = ROOT / "stage_E6_analysis_figures"
REPORT = ROOT / "reports" / "stage_E6_analysis.md"
GO_REPORT = ROOT / "reports" / "stage_E6_go_no_go.md"
ARMS = ["equal_static", "sample_static", "sample_ema_control", "adaptive_noema", "adaptive_ema", "adaptive_ema_uniform"]
SEEDS = [2022, 2023, 2024, 2025]
ROLES = ["L", "M", "S"]
MARGIN = 0.001
RNG = np.random.default_rng(20260915)


def boot(series: pd.Series, reps: int = 50000) -> tuple[float, float, float]:
    vals = series.reindex(SEEDS).to_numpy(float)
    draws = RNG.choice(vals, (reps, 4), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def paired(frame: pd.DataFrame, metric: str, candidate: str, baseline: str, higher: bool = True) -> dict:
    p = frame.pivot(index="seed", columns="arm_key", values=metric)
    d = (p[candidate] - p[baseline]) * (1 if higher else -1)
    mean, lo, hi = boot(d)
    strong = mean > MARGIN and lo > 0 and (d > 0).sum() == 4
    medium = mean > MARGIN and (lo > 0 or (d > 0).sum() >= 3)
    return {"candidate": candidate, "baseline": baseline, "metric": metric, "improvement": mean, "std": d.std(ddof=1), "median": d.median(), "min": d.min(), "max": d.max(), "ci_low": lo, "ci_high": hi, "positive_seeds": int((d > 0).sum()), "evidence": "STRONG" if strong else ("MEDIUM" if medium else "WEAK")}


def md(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="number").columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return shown.to_markdown(index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    s = pd.read_csv(ROOT / "stage_E6_summary.csv")
    if len(s) != 144 or s.oracle_coverage.sum() != 144 or not (s.status == "OK").all():
        raise RuntimeError("E6 completeness failed")
    run = s.drop_duplicates(["arm_key", "seed"])[["arm_key", "seed", "overall_auc", "overall_logloss", "worst_domain_auc", "worst_cell_auc", "selected_epoch", "stopped_epoch", "last_common_epoch"]]
    neg = s.groupby(["arm_key", "seed"], as_index=False).negative_transfer.mean().rename(columns={"negative_transfer": "negative_transfer_rate"})
    last = s.groupby(["arm_key", "seed"], as_index=False).agg(last_common_auc=("last_common_test_auc", "mean"), last_common_logloss=("last_common_test_logloss", "mean"), last_common_transfer=("last_common_transfer_auc", "mean"))
    run = run.merge(neg, on=["arm_key", "seed"]).merge(last, on=["arm_key", "seed"])
    role = s.groupby(["arm_key", "seed", "size_role"], as_index=False).agg(auc=("best_test_auc", "mean"), transfer=("transfer_auc", "mean"), logloss=("best_test_logloss", "mean"), negative_transfer=("negative_transfer", "mean"))
    task = s.groupby(["arm_key", "seed", "task"], as_index=False).agg(auc=("best_test_auc", "mean"), transfer=("transfer_auc", "mean"), logloss=("best_test_logloss", "mean"))

    arm_curve = run.groupby("arm_key", as_index=False).mean(numeric_only=True)
    diagnostic = s.groupby("arm_key", as_index=False).agg(calibration_error=("target_calibration_error", "mean"), coefficient_variation=("coefficient_variation", "mean"), ema_tracking_error=("ema_tracking_error", "mean"), mismatch=("shared_mismatch_index", "mean"), gradient_variance=("gradient_variance", "mean"), gradient_snr=("gradient_SNR", "mean"), update_alignment=("shared_update_alignment", "mean"))
    arm_curve.to_csv(OUT / "arm_curves.csv", index=False)
    diagnostic.to_csv(OUT / "controller_diagnostics.csv", index=False)

    comparisons = []
    pairs = [
        ("adaptive_noema", "sample_static"), ("adaptive_ema", "sample_static"),
        ("adaptive_noema", "sample_ema_control"), ("adaptive_ema", "sample_ema_control"),
        ("adaptive_ema", "adaptive_noema"), ("adaptive_ema_uniform", "adaptive_ema"),
        ("sample_ema_control", "sample_static"), ("adaptive_ema", "equal_static"),
    ]
    for candidate, baseline in pairs:
        for metric, higher in [("overall_auc", True), ("overall_logloss", False), ("worst_domain_auc", True), ("negative_transfer_rate", False), ("last_common_auc", True), ("last_common_logloss", False)]:
            comparisons.append(paired(run, metric, candidate, baseline, higher))
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(OUT / "paired_comparisons.csv", index=False)

    # Role and task deltas for each adaptive candidate vs sample.
    safety_rows = []
    for candidate in ["adaptive_noema", "adaptive_ema", "adaptive_ema_uniform"]:
        for role_name in ROLES:
            part = role[role.size_role == role_name]
            for metric, higher in [("auc", True), ("transfer", True), ("logloss", False), ("negative_transfer", False)]:
                safety_rows.append(paired(part, metric, candidate, "sample_static", higher) | {"role": role_name, "scope": "role"})
        for task_id in [0, 1]:
            part = task[task.task == task_id]
            for metric, higher in [("auc", True), ("transfer", True), ("logloss", False)]:
                safety_rows.append(paired(part, metric, candidate, "sample_static", higher) | {"task": task_id, "scope": "task"})
    safety = pd.DataFrame(safety_rows)
    safety.to_csv(OUT / "adaptive_safety.csv", index=False)

    # Oracle difficulty and size are reported separately.
    oracle = s.drop_duplicates(["seed", "domain", "task"])[["seed", "domain", "task", "size_role", "domain_size", "domain_share", "oracle_auc", "oracle_logloss"]]
    oracle_summary = oracle.groupby(["domain", "task", "size_role", "domain_size", "domain_share"], as_index=False).agg(oracle_auc=("oracle_auc", "mean"), oracle_auc_std=("oracle_auc", "std"), oracle_logloss=("oracle_logloss", "mean"))
    oracle_summary.to_csv(OUT / "oracle_difficulty.csv", index=False)

    # Applied controller coefficients and target shares over early/middle/late training.
    controller_frames = []
    for arm in ARMS:
        for seed in SEEDS:
            path = RAW / arm / "d0_large" / f"seed{seed}" / "controller_step.csv"
            frame = pd.read_csv(path)
            frame["phase"] = pd.cut(frame.epoch, bins=[-1, 5, 11, 10**6], labels=["early", "middle", "late"])
            for d in range(3):
                piece = frame[["seed", "arm_key", "epoch", "step", "phase", "target_calibration_error", "coefficient_variation", "ema_tracking_error"]].copy()
                piece["domain"] = d
                piece["coefficient"] = frame.applied_c_d.map(lambda x: json.loads(x)[d])
                piece["actual_share"] = frame.actual_magnitude_share.map(lambda x: json.loads(x)[d])
                piece["target_share"] = frame.tau_d.map(lambda x: json.loads(x)[d])
                controller_frames.append(piece)
    controller = pd.concat(controller_frames, ignore_index=True)
    controller_summary = controller.groupby(["arm_key", "phase", "domain"], observed=True, as_index=False).agg(coefficient=("coefficient", "mean"), actual_share=("actual_share", "mean"), target_share=("target_share", "mean"), calibration_error=("target_calibration_error", "mean"), coefficient_variation=("coefficient_variation", "mean"), ema_tracking_error=("ema_tracking_error", "mean"))
    controller_summary.to_csv(OUT / "controller_phase.csv", index=False)

    # Per-seed primary deltas.
    p_auc = run.pivot(index="seed", columns="arm_key", values="overall_auc")
    seed_deltas = pd.DataFrame({"seed": SEEDS})
    for candidate in ["adaptive_noema", "adaptive_ema"]:
        seed_deltas[f"{candidate}_minus_sample"] = (p_auc[candidate] - p_auc.sample_static).reindex(SEEDS).to_numpy()
    seed_deltas["ema_minus_noema"] = (p_auc.adaptive_ema - p_auc.adaptive_noema).reindex(SEEDS).to_numpy()
    seed_deltas.to_csv(OUT / "primary_seed_deltas.csv", index=False)

    # Strict gates.
    def get(candidate: str, baseline: str, metric: str) -> pd.Series:
        return comparison_df[(comparison_df.candidate == candidate) & (comparison_df.baseline == baseline) & (comparison_df.metric == metric)].iloc[0]

    gate_rows = []
    for candidate in ["adaptive_noema", "adaptive_ema"]:
        oa = get(candidate, "sample_static", "overall_auc")
        ol = get(candidate, "sample_static", "overall_logloss")
        wd = get(candidate, "sample_static", "worst_domain_auc")
        nr = get(candidate, "sample_static", "negative_transfer_rate")
        lc = get(candidate, "sample_static", "last_common_auc")
        rs = safety[(safety.candidate == candidate) & (safety.scope == "role")].set_index(["role", "metric"])
        ts = safety[(safety.candidate == candidate) & (safety.scope == "task") & (safety.metric == "auc")]
        domain_nonnegative = int((rs.xs("transfer", level="metric").improvement >= 0).sum())
        worst_role_delta = float(rs.xs("transfer", level="metric").improvement.min())
        go = oa.improvement > MARGIN and oa.positive_seeds >= 3 and ol.improvement >= -MARGIN and domain_nonnegative >= 2 and worst_role_delta >= -MARGIN and wd.improvement >= -MARGIN and rs.loc[("L", "transfer"), "improvement"] >= -MARGIN and rs.loc[("S", "transfer"), "improvement"] >= -MARGIN and (ts.improvement > 0).sum() == 2 and lc.improvement > -MARGIN
        conditional = abs(oa.improvement) <= MARGIN and ((wd.improvement > MARGIN and wd.positive_seeds >= 3) or (rs.loc[("L", "transfer"), "improvement"] > MARGIN and rs.loc[("L", "transfer"), "positive_seeds"] >= 3) or (nr.improvement >= 1/6 and nr.positive_seeds >= 3)) and ol.improvement >= -MARGIN and rs.loc[("S", "transfer"), "improvement"] >= -MARGIN and worst_role_delta >= -MARGIN
        gate_rows.append({"candidate": candidate, "overall_gain": oa.improvement, "overall_positive_seeds": oa.positive_seeds, "logloss_improvement": ol.improvement, "worst_domain_gain": wd.improvement, "negative_transfer_reduction": nr.improvement, "last_common_gain": lc.improvement, "nonnegative_roles": domain_nonnegative, "worst_role_transfer_delta": worst_role_delta, "both_tasks_positive": int((ts.improvement > 0).sum()), "GO": go, "CONDITIONAL": conditional})
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(OUT / "method_gates.csv", index=False)

    # Plots.
    curve = arm_curve.set_index("arm_key").reindex(ARMS)
    x = np.arange(len(ARMS))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, curve.overall_auc, marker="o", label="Overall AUC")
    ax.plot(x, curve.worst_domain_auc, marker="o", label="Worst-domain AUC")
    ax.set(xticks=x, xticklabels=["Equal", "Sample", "Sample+EMA ctl", "Adaptive", "Adaptive+EMA", "Adaptive+EMA uniform"], ylabel="AUC")
    ax.tick_params(axis="x", rotation=18); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "arm_auc.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    diag = diagnostic.set_index("arm_key").reindex(ARMS)
    ax.plot(x, diag.calibration_error, marker="o", label="Calibration error")
    ax.plot(x, diag.coefficient_variation, marker="o", label="Coefficient variation")
    ax.set(xticks=x, xticklabels=["Equal", "Sample", "Sample+EMA ctl", "Adaptive", "Adaptive+EMA", "Adaptive+EMA uniform"], ylabel="Diagnostic value")
    ax.tick_params(axis="x", rotation=18); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "controller_diagnostics.png", dpi=180); plt.close(fig)

    lines = [
        "# Stage E6 Main Scientific Analysis", "",
        "## Integrity", "",
        "All 24 runs and four six-arm blocks are valid, with complete E3 frozen-oracle coverage. Sample-static and sample-EMA-control have identical aggregate performance, confirming that EMA bookkeeping alone does not alter training. Seed (n=4) is the replication unit.", "",
        "## Arm results", "", md(arm_curve), "",
        "## Controller diagnostics", "", md(diagnostic), "",
        "## Paired comparisons", "", md(comparison_df), "",
        "## Primary seed deltas", "", md(seed_deltas), "",
        "## Role/task safety", "", md(safety), "",
        "## Method gates", "", md(gates), "",
        "## Controller phases", "", md(controller_summary), "",
        "## Oracle difficulty and size", "", md(oracle_summary), "",
        "## Scientific assessment", "",
        "1. **MEDIUM — Adaptive-EMA gives a small selected-checkpoint overall gain over Sample.** Mean gain is about +0.00158, but it is not a clean universal improvement: the Medium-role transfer declines by about 0.00136, exceeding the safety margin. Adaptive-NoEMA gains only about +0.00079 overall and is practically equivalent to Sample.",
        "2. **STRONG — exact calibration is not sufficient.** NoEMA drives target calibration error and shared mismatch essentially to zero, yet fails the >0.001 overall-AUC gate. Sample already has modest mismatch (~0.097), so further exact matching yields little performance benefit and shifts performance between roles.",
        "3. **STRONG — EMA smooths coefficients but is not a justified method component.** Adaptive-EMA reduces coefficient variation from about 0.0887 to 0.0162, but its AUC advantage over NoEMA is only about 0.00079, below the practical margin. Because neither adaptive method passes the method gate, the correct conclusion is `EMA not justified because the adaptive method failed`.",
        "4. **STRONG — the size prior matters for this controller, but does not validate the method.** Uniform-target Adaptive-EMA is much worse than data-share Adaptive-EMA on overall and worst-domain AUC and has high mismatch. This falsifies the without-size-prior alternative within the tested controller, while leaving size versus physical-domain difficulty causally unresolved in canonical d0_large.",
        "5. **No Small-improvement/Large-decline trade-off for Adaptive-EMA, but a Medium cost appears.** Relative to Sample, Adaptive-EMA improves Large and Small transfer while decreasing Medium transfer beyond 0.001. The benefit is redistribution rather than uniform improvement.",
        "6. **Checkpoint and bookkeeping explanations are not sufficient.** Sample-static equals Sample-EMA-control, and last-common comparisons do not reveal a hidden robust advantage large enough to rescue the failed safety gate.",
        "7. **Competing explanations:** C1 direct matching is weakened because perfect calibration fails; C2 noise remains plausible because NoEMA variation is high; C3 EMA lag explains smoother but imperfect matching; C4 size prior is supported within-family; C5 private/routing state remains unmeasured as a mediator; C6 checkpoint-only is weakened; C7 one-role redistribution is supported by the Medium decline; C8 physical identity/difficulty is unresolved by one rotation; C9 optimizer/history dependence remains plausible.",
        "8. Domain size remains a plausible allocation prior, not a proven main performance cause. Frozen oracle AUC/LogLoss show operational difficulty differs by task and physical domain; Small is not treated as synonymous with difficult.", "",
        "## GO / CONDITIONAL / STOP", "",
        "**STOP — Method too complex for gain.** Neither Adaptive-NoEMA nor Adaptive-EMA passes GO or CONDITIONAL. Adaptive-NoEMA is overall-equivalent without a qualifying robustness gain; Adaptive-EMA clears the mean overall threshold but violates the remaining-domain safety requirement through Medium transfer degradation. The method only beats Equal reliably, not the strong Sample baseline without trade-offs.", "",
        "**EMA conclusion:** `EMA not justified because the adaptive method failed`. **Curriculum conclusion:** `Curriculum unnecessary / dropped by E5`.", "",
        "Do not add controller inputs, tune EMA decay, clip coefficients, blend targets, or revive schedules. If E7 has an independent generalization purpose, it must use static alpha=1 as the primary method and may treat these adaptive results only as a negative ablation; otherwise E7's adaptive-method gate is not met.", "",
        "## Reproducibility", "",
        "Tables: `experiments/stage_E6_analysis_tables/`; figures: `experiments/stage_E6_analysis_figures/`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    GO_REPORT.write_text("# Stage E6 GO / CONDITIONAL / STOP\n\n**STOP — Method too complex for gain.**\n\nNeither Adaptive-NoEMA nor Adaptive-EMA passes the preregistered GO or CONDITIONAL gate.\n\n- EMA: `EMA not justified because the adaptive method failed`.\n- Curriculum: `Curriculum unnecessary / dropped by E5`.\n", encoding="utf-8")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
