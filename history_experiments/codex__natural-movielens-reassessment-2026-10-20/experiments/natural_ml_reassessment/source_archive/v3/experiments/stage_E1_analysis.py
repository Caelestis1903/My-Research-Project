from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stage_E1_raw_results"
OUT = ROOT / "stage_E1_analysis_tables"
REPORT = ROOT / "reports" / "stage_E1_analysis.md"
WEIGHTINGS = ("equal", "sample")
SETTINGS = ("d0_large", "d1_large", "d2_large")
SEEDS = (2022, 2023, 2024, 2025)
ROLE_ORDER = ["L", "M", "S"]


def load_all(name: str) -> pd.DataFrame:
    frames = []
    for weighting in WEIGHTINGS:
        for setting in SETTINGS:
            for seed in SEEDS:
                path = RAW / weighting / setting / f"seed{seed}" / name
                frame = pd.read_csv(path)
                if "weighting" not in frame.columns:
                    frame["weighting"] = weighting
                if "setting" not in frame.columns:
                    frame["setting"] = setting
                if "seed" not in frame.columns:
                    frame["seed"] = seed
                frame["source_path"] = str(path)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def describe_delta(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    def summarize(series: pd.Series) -> pd.Series:
        x = series.dropna().astype(float)
        return pd.Series(
            {
                "n": len(x),
                "mean": x.mean(),
                "std": x.std(ddof=1),
                "median": x.median(),
                "min": x.min(),
                "max": x.max(),
                "improved_n": int((x > 0).sum()),
                "declined_n": int((x < 0).sum()),
            }
        )

    return frame.groupby(groups, dropna=False)[value].apply(summarize).unstack().reset_index()


def md_table(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return shown.to_markdown(index=False)


def sustained_onset(values: pd.Series, threshold: float) -> float:
    arr = values.to_numpy(dtype=float)
    for idx in range(max(0, len(arr) - 1)):
        if abs(arr[idx]) >= threshold and abs(arr[idx + 1]) >= threshold and np.sign(arr[idx]) == np.sign(arr[idx + 1]):
            return float(values.index[idx])
    return np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_all("metrics_summary.csv")
    metrics = load_all("metrics_epoch.csv")
    steps = load_all("gradient_step.csv")
    grad_epoch = load_all("gradient_epoch.csv")
    gates = load_all("gate_epoch.csv")
    cosines = load_all("gradient_cosine_epoch.csv")

    identity = ["setting", "seed", "domain", "task", "size_role"]
    eq = summary[summary.weighting == "equal"].copy()
    sa = summary[summary.weighting == "sample"].copy()
    paired = eq.merge(sa, on=identity, suffixes=("_equal", "_sample"), validate="one_to_one")
    for metric in ["best_test_auc", "best_test_logloss", "best_val_auc", "best_val_logloss", "best_train_auc", "best_train_logloss", "best_epoch", "stopped_epoch"]:
        paired[f"delta_{metric}"] = paired[f"{metric}_sample"] - paired[f"{metric}_equal"]
    paired.to_csv(OUT / "paired_checkpoint_cells.csv", index=False)

    role_auc = describe_delta(paired, ["size_role"], "delta_best_test_auc")
    role_loss = describe_delta(paired.assign(delta_loss_improvement=-paired.delta_best_test_logloss), ["size_role"], "delta_loss_improvement")
    task_role_auc = describe_delta(paired, ["size_role", "task"], "delta_best_test_auc")
    physical_auc = describe_delta(paired, ["domain", "size_role"], "delta_best_test_auc")
    for frame, name in [(role_auc, "checkpoint_auc_by_role.csv"), (role_loss, "checkpoint_logloss_by_role.csv"), (task_role_auc, "checkpoint_auc_by_role_task.csv"), (physical_auc, "checkpoint_auc_by_physical_domain_role.csv")]:
        frame.to_csv(OUT / name, index=False)

    run_role = paired.groupby(["setting", "seed", "size_role"], as_index=False).agg(
        delta_auc=("delta_best_test_auc", "mean"),
        delta_logloss=("delta_best_test_logloss", "mean"),
    )
    run_role.to_csv(OUT / "paired_seed_role.csv", index=False)
    run_role_stats = describe_delta(run_role, ["size_role"], "delta_auc")
    run_role_stats.to_csv(OUT / "paired_seed_role_stats.csv", index=False)
    run_overall = paired.groupby(["setting", "seed"], as_index=False).agg(
        delta_auc=("delta_best_test_auc", "mean"),
        delta_logloss=("delta_best_test_logloss", "mean"),
    )
    overall_stats = describe_delta(run_overall, [], "delta_auc") if False else pd.DataFrame(
        [{
            "n": len(run_overall), "mean": run_overall.delta_auc.mean(), "std": run_overall.delta_auc.std(ddof=1),
            "median": run_overall.delta_auc.median(), "min": run_overall.delta_auc.min(), "max": run_overall.delta_auc.max(),
            "improved_n": int((run_overall.delta_auc > 0).sum()), "declined_n": int((run_overall.delta_auc < 0).sum())
        }]
    )
    overall_stats.to_csv(OUT / "checkpoint_auc_overall.csv", index=False)

    role_map = summary[["setting", "domain", "size_role", "domain_share"]].drop_duplicates()
    steps = steps.merge(role_map, on=["setting", "domain"], how="left", validate="many_to_one")
    shared = steps[steps.parameter_group == "shared"].copy()
    common_keys = ["setting", "seed", "epoch", "step", "domain", "task", "parameter_group"]
    se = shared[shared.weighting == "equal"]
    ss = shared[shared.weighting == "sample"]
    common = se.merge(ss, on=common_keys, suffixes=("_equal", "_sample"), validate="one_to_one")
    common["size_role"] = common["size_role_equal"]
    common["actual_contribution_delta"] = common.sample_cf_contribution_sample - common.equal_cf_contribution_equal
    common["equal_ref_trajectory_delta"] = common.equal_cf_contribution_sample - common.equal_cf_contribution_equal
    common["sample_ref_trajectory_delta"] = common.sample_cf_contribution_sample - common.sample_cf_contribution_equal
    common["direct_delta_equal_state"] = common.sample_cf_contribution_equal - common.equal_cf_contribution_equal
    common["direct_delta_sample_state"] = common.sample_cf_contribution_sample - common.equal_cf_contribution_sample
    common["update_alignment_delta"] = common.actual_update_alignment_sample - common.actual_update_alignment_equal
    common.to_csv(OUT / "shared_common_support_cells.csv", index=False)

    contribution_stats = common.groupby("size_role", as_index=False).agg(
        actual_equal=("equal_cf_contribution_equal", "mean"),
        actual_sample=("sample_cf_contribution_sample", "mean"),
        total_actual_delta=("actual_contribution_delta", "mean"),
        direct_delta_equal_state=("direct_delta_equal_state", "mean"),
        direct_delta_sample_state=("direct_delta_sample_state", "mean"),
        equal_ref_trajectory_delta=("equal_ref_trajectory_delta", "mean"),
        sample_ref_trajectory_delta=("sample_ref_trajectory_delta", "mean"),
        update_alignment_equal=("actual_update_alignment_equal", "mean"),
        update_alignment_sample=("actual_update_alignment_sample", "mean"),
        update_alignment_delta=("update_alignment_delta", "mean"),
    )
    contribution_stats.to_csv(OUT / "shared_contribution_decomposition_by_role.csv", index=False)

    # Domain magnitude shares/mismatch are repeated for both tasks, so deduplicate task.
    dom = common.sort_values("task").drop_duplicates(["setting", "seed", "epoch", "step", "domain"])
    mag_stats = dom.groupby("size_role", as_index=False).agg(
        equal_actual_share=("equal_cf_magnitude_share_equal", "mean"),
        sample_actual_share=("sample_cf_magnitude_share_sample", "mean"),
        equal_state_sample_cf_share=("sample_cf_magnitude_share_equal", "mean"),
        sample_state_equal_cf_share=("equal_cf_magnitude_share_sample", "mean"),
        equal_actual_overrep=("overrepresentation_ratio_equal", "mean"),
        sample_actual_overrep=("overrepresentation_ratio_sample", "mean"),
    )
    mag_stats.to_csv(OUT / "shared_magnitude_share_by_role.csv", index=False)

    def mismatch_by_probe(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rows = frame.groupby(["setting", "seed", "epoch", "step"], as_index=False).agg(
            equal_state_equal_cf=(f"equal_cf_magnitude_share_equal", lambda x: np.nan),
        )
        return rows

    mismatch = dom.groupby(["setting", "seed", "epoch", "step"], as_index=False).apply(
        lambda g: pd.Series({
            "equal_state_equal_cf": 0.5 * np.abs(g.equal_cf_magnitude_share_equal.to_numpy() - g.domain_share_equal.to_numpy()).sum(),
            "equal_state_sample_cf": 0.5 * np.abs(g.sample_cf_magnitude_share_equal.to_numpy() - g.domain_share_equal.to_numpy()).sum(),
            "sample_state_equal_cf": 0.5 * np.abs(g.equal_cf_magnitude_share_sample.to_numpy() - g.domain_share_sample.to_numpy()).sum(),
            "sample_state_sample_cf": 0.5 * np.abs(g.sample_cf_magnitude_share_sample.to_numpy() - g.domain_share_sample.to_numpy()).sum(),
        }), include_groups=False
    ).reset_index(drop=True)
    mismatch["direct_equal_state"] = mismatch.equal_state_sample_cf - mismatch.equal_state_equal_cf
    mismatch["direct_sample_state"] = mismatch.sample_state_sample_cf - mismatch.sample_state_equal_cf
    mismatch["trajectory_equal_ref"] = mismatch.sample_state_equal_cf - mismatch.equal_state_equal_cf
    mismatch["trajectory_sample_ref"] = mismatch.sample_state_sample_cf - mismatch.equal_state_sample_cf
    mismatch["total_actual"] = mismatch.sample_state_sample_cf - mismatch.equal_state_equal_cf
    mismatch.to_csv(OUT / "shared_mismatch_decomposition.csv", index=False)

    # Repeat the actual magnitude-mismatch calculation for every parameter location.
    pg_dom = steps.sort_values("task").drop_duplicates(["weighting", "setting", "seed", "epoch", "step", "parameter_group", "domain"])
    pg_mismatch = pg_dom.groupby(["weighting", "setting", "seed", "epoch", "step", "parameter_group"], as_index=False).apply(
        lambda g: pd.Series({"actual_mismatch": 0.5 * np.abs(g.magnitude_share.to_numpy() - g.domain_share.to_numpy()).sum()}),
        include_groups=False,
    ).reset_index(drop=True)
    pg_pair = pg_mismatch[pg_mismatch.weighting == "equal"].merge(
        pg_mismatch[pg_mismatch.weighting == "sample"],
        on=["setting", "seed", "epoch", "step", "parameter_group"], suffixes=("_equal", "_sample")
    )
    pg_pair["delta"] = pg_pair.actual_mismatch_sample - pg_pair.actual_mismatch_equal
    pg_mismatch_stats = pg_pair.groupby("parameter_group", as_index=False).agg(
        equal=("actual_mismatch_equal", "mean"), sample=("actual_mismatch_sample", "mean"), delta=("delta", "mean")
    )
    pg_mismatch_stats.to_csv(OUT / "mismatch_by_parameter_location.csv", index=False)

    ge = grad_epoch.merge(role_map, on=["setting", "domain"], how="left", validate="many_to_one")
    ge_shared = ge[ge.parameter_group == "shared"]
    gradient_role = ge_shared.groupby(["weighting", "size_role"], as_index=False).agg(
        raw_gradient_norm=("gradient_norm", "mean"), weighted_gradient_norm=("weighted_gradient_norm", "mean"),
        gradient_variance=("gradient_variance", "mean"), gradient_SNR=("gradient_SNR", "mean"),
    )
    gradient_role.to_csv(OUT / "shared_gradient_by_role.csv", index=False)
    parameter_location = ge.groupby(["weighting", "parameter_group"], as_index=False).agg(
        gradient_norm=("gradient_norm", "mean"), gradient_SNR=("gradient_SNR", "mean"), contribution=("contribution_mean", "mean")
    )
    parameter_location.to_csv(OUT / "parameter_location.csv", index=False)

    gate_common = gates[gates.weighting == "equal"].merge(
        gates[gates.weighting == "sample"],
        on=["setting", "seed", "epoch", "domain", "task", "cell", "routing_scope"],
        suffixes=("_equal", "_sample"), validate="one_to_one"
    )
    gate_common = gate_common.merge(role_map[["setting", "domain", "size_role"]], on=["setting", "domain"], how="left")
    for metric in ["gate_entropy", "effective_expert_count", "top1_expert_frequency"]:
        gate_common[f"delta_{metric}"] = gate_common[f"{metric}_sample"] - gate_common[f"{metric}_equal"]
    gate_stats = gate_common.groupby("size_role", as_index=False).agg(
        entropy_equal=("gate_entropy_equal", "mean"), entropy_sample=("gate_entropy_sample", "mean"), entropy_delta=("delta_gate_entropy", "mean"),
        experts_delta=("delta_effective_expert_count", "mean"), top1_delta=("delta_top1_expert_frequency", "mean")
    )
    gate_stats.to_csv(OUT / "routing_by_role.csv", index=False)

    cos_common = cosines[cosines.weighting == "equal"].merge(
        cosines[cosines.weighting == "sample"], on=["setting", "seed", "epoch", "domain_pair"], suffixes=("_equal", "_sample")
    )
    cos_common["delta"] = cos_common.cosine_mean_sample - cos_common.cosine_mean_equal
    cosine_stats = cos_common.groupby("domain_pair", as_index=False).agg(equal=("cosine_mean_equal", "mean"), sample=("cosine_mean_sample", "mean"), delta=("delta", "mean"))
    cosine_stats.to_csv(OUT / "cosine_common_support.csv", index=False)

    # Common-support epoch trajectory and post-hoc descriptive onsets.
    me = metrics[metrics.weighting == "equal"]
    ms = metrics[metrics.weighting == "sample"]
    mt = me.merge(ms, on=["setting", "seed", "epoch", "domain", "task", "size_role"], suffixes=("_equal", "_sample"))
    mt["delta_test_auc"] = mt.test_auc_sample - mt.test_auc_equal
    large_traj = mt[mt.size_role == "L"].groupby(["setting", "seed", "epoch"], as_index=False).delta_test_auc.mean()
    mm_epoch = mismatch.groupby(["setting", "seed", "epoch"], as_index=False).total_actual.mean()
    trajectory = large_traj.merge(mm_epoch, on=["setting", "seed", "epoch"], how="inner")
    trajectory.to_csv(OUT / "large_auc_mismatch_trajectory.csv", index=False)
    onset_rows = []
    for (setting, seed), group in trajectory.sort_values("epoch").groupby(["setting", "seed"]):
        auc_s = group.set_index("epoch").delta_test_auc
        mis_s = group.set_index("epoch").total_actual
        onset_rows.append({"setting": setting, "seed": seed, "auc_onset_abs_0.001": sustained_onset(auc_s, 0.001), "mismatch_onset_abs_0.01": sustained_onset(mis_s, 0.01)})
    onsets = pd.DataFrame(onset_rows)
    onsets.to_csv(OUT / "posthoc_onsets.csv", index=False)
    corr = trajectory[["delta_test_auc", "total_actual"]].corr(method="spearman").iloc[0, 1]

    gate_large = gate_common[gate_common.size_role == "L"].groupby(["setting", "seed", "epoch"], as_index=False).agg(
        entropy_delta=("delta_gate_entropy", "mean"), top1_delta=("delta_top1_expert_frequency", "mean")
    )
    routing_traj = trajectory.merge(gate_large, on=["setting", "seed", "epoch"], how="inner")
    routing_entropy_corr = routing_traj[["delta_test_auc", "entropy_delta"]].corr(method="spearman").iloc[0, 1]
    routing_top1_corr = routing_traj[["delta_test_auc", "top1_delta"]].corr(method="spearman").iloc[0, 1]

    # Checkpoint-selection alternative: common-support mean, last common epoch, and the equal arm's selected epoch.
    stop_stats = paired[["setting", "seed", "best_epoch_equal", "best_epoch_sample", "stopped_epoch_equal", "stopped_epoch_sample"]].drop_duplicates()
    mt_observed = mt[mt.test_auc_equal.notna() & mt.test_auc_sample.notna()].copy()
    role_epoch = mt_observed.groupby(["setting", "seed", "epoch", "size_role"], as_index=False).agg(delta_auc=("delta_test_auc", "mean"))
    common_mean = role_epoch.groupby(["setting", "seed", "size_role"], as_index=False).delta_auc.mean().rename(columns={"delta_auc": "common_epoch_mean_delta"})
    last_common = role_epoch.sort_values("epoch").groupby(["setting", "seed", "size_role"], as_index=False).tail(1)[["setting", "seed", "size_role", "epoch", "delta_auc"]].rename(columns={"epoch": "last_common_epoch", "delta_auc": "last_common_delta"})
    equal_best = stop_stats[["setting", "seed", "best_epoch_equal"]]
    at_equal_best = role_epoch.merge(equal_best, on=["setting", "seed"], how="left")
    at_equal_best = at_equal_best[at_equal_best.epoch == at_equal_best.best_epoch_equal][["setting", "seed", "size_role", "delta_auc"]].rename(columns={"delta_auc": "equal_checkpoint_epoch_delta"})
    selection_check = common_mean.merge(last_common, on=["setting", "seed", "size_role"]).merge(at_equal_best, on=["setting", "seed", "size_role"], how="left")
    selection_stats = selection_check.groupby("size_role", as_index=False).agg(
        common_mean=("common_epoch_mean_delta", "mean"),
        common_positive=("common_epoch_mean_delta", lambda x: int((x > 0).sum())),
        last_common_mean=("last_common_delta", "mean"),
        last_common_positive=("last_common_delta", lambda x: int((x > 0).sum())),
        equal_checkpoint_mean=("equal_checkpoint_epoch_delta", "mean"),
        equal_checkpoint_positive=("equal_checkpoint_epoch_delta", lambda x: int((x > 0).sum())),
    )
    selection_check.to_csv(OUT / "checkpoint_selection_seed_role.csv", index=False)
    selection_stats.to_csv(OUT / "checkpoint_selection_stats.csv", index=False)

    stop_stats.to_csv(OUT / "checkpoint_timing.csv", index=False)

    oracle_available = int((summary["oracle_source"].fillna("missing") != "missing").sum())
    role_order = pd.CategoricalDtype(ROLE_ORDER, ordered=True)
    for frame in [role_auc, role_loss, task_role_auc, contribution_stats, mag_stats, gradient_role, gate_stats]:
        if "size_role" in frame:
            frame["size_role"] = frame.size_role.astype(role_order)
            frame.sort_values("size_role", inplace=True)

    lines = [
        "# Stage E1 Main Scientific Analysis",
        "",
        "## Scope and data integrity",
        "",
        "All 24 formal runs completed and all equal/sample pairs were VALID. Analysis uses seed-level paired comparisons; positive AUC delta means sample weighting is better, while positive LogLoss improvement means sample weighting has lower loss. Oracle fields were not available, so transfer and independent operational-difficulty claims are not made.",
        "",
        "## Selected-checkpoint performance",
        "",
        md_table(role_auc),
        "",
        "LogLoss improvement (equal minus sample):",
        "",
        md_table(role_loss),
        "",
        "Per role and task AUC deltas:",
        "",
        md_table(task_role_auc),
        "",
        "Overall six-cell paired AUC delta (12 setting×seed pairs):",
        "",
        md_table(overall_stats),
        "",
        "Seed-level role aggregates (12 setting×seed pairs per role):",
        "",
        md_table(run_role_stats),
        "",
        "## Shared optimization diagnostics",
        "",
        "Contribution decomposition (sample minus equal; trajectory columns compare model states under a common reference weighting):",
        "",
        md_table(contribution_stats),
        "",
        "Magnitude allocation by size role:",
        "",
        md_table(mag_stats),
        "",
        "Mismatch decomposition across matched probes:",
        "",
        md_table(pd.DataFrame([mismatch[["equal_state_equal_cf", "equal_state_sample_cf", "sample_state_equal_cf", "sample_state_sample_cf", "direct_equal_state", "direct_sample_state", "trajectory_equal_ref", "trajectory_sample_ref", "total_actual"]].mean()])),
        "",
        "Shared gradient norm, variance, and SNR:",
        "",
        md_table(gradient_role),
        "",
        "## Competing diagnostics",
        "",
        "Routing:",
        "",
        md_table(gate_stats),
        "",
        "Raw shared-gradient pairwise cosine on common epochs:",
        "",
        md_table(cosine_stats),
        "",
        "Parameter-location summary (cross-group L2 norms are descriptive only; RMS is required for strict cross-group scale comparison):",
        "",
        md_table(parameter_location),
        "",
        "Actual contribution mismatch by parameter location:",
        "",
        md_table(pg_mismatch_stats),
        "",
        "Checkpoint/convergence diagnostic (12 pairs per role):",
        "",
        md_table(selection_stats),
        "",
        "## Timing and association",
        "",
        f"Across common-support epoch observations, Spearman correlation between sample-minus-equal Large test AUC and actual shared mismatch delta was {corr:.4f}; corresponding correlations with Large routing entropy delta and top-1-frequency delta were {routing_entropy_corr:.4f} and {routing_top1_corr:.4f}. Post-hoc onset tables use two-consecutive-epoch thresholds |AUC delta|>=0.001 and |mismatch delta|>=0.01; these thresholds were not preregistered and are descriptive only.",
        "",
        "## Scientific assessment",
        "",
        "### Stable repeated phenomena",
        "",
        "1. **STRONG — sample weighting improves performance under this protocol.** The six-cell test-AUC mean improved in 12/12 setting×seed pairs (mean +0.021085). Seed-level role means improved for Large in 12/12 (+0.014335), Medium in 12/12 (+0.011430), and Small in 11/12 (+0.037488). Both tasks show the same aggregate direction. Mean LogLoss also improved for every role, although individual cells contain exceptions.",
        "2. **STRONG — equal weighting produces severe size-share mismatch, and sample weighting reduces it.** Shared mismatch fell from 0.639906 to 0.132993 (delta -0.506913) in all 12 paired runs. Equal weighting allocated mean shared magnitude shares L/M/S=0.1768/0.2366/0.5866 against data shares 0.800/0.184/0.016; sample weighting changed them to 0.6965/0.2415/0.0621. Small remained overrepresented (3.88x), but far less than under equal weighting (36.66x).",
        "3. **STRONG — actual shared update alignment moves toward Large and away from Small.** Large alignment increased in 12/12 seed-setting pairs (mean +0.0427); Small decreased in 12/12 (mean -0.0199). Pairwise raw-gradient cosine also became more positive for all three physical-domain pairs.",
        "4. **STRONG — the result is not a small-domain-improvement/large-domain-decline trade-off.** Large, Medium, and Small all improve on average; Small improves the most. This directly contradicts a claimed trade-off for the tested sample-weighting intervention.",
        "",
        "### Contradictions and limits",
        "",
        "- Cell-level AUC is not unanimous: each role has 23/24 improving cells, with one decline in each. At the seed-role level, Large and Medium are unanimous, while Small has one decline (d0_large, seed 2024).",
        "- LogLoss has more exceptions than AUC: 4/24 Large, 3/24 Medium, and 2/24 Small cells worsen despite positive mean improvements.",
        "- The mismatch reduction is dominated by the deterministic coefficient change: at the equal-trained state, the direct counterfactual change is -0.499176 versus total actual -0.506913. Common-reference trajectory effects are much smaller and inconsistent (equal reference negative in 8/12 pairs; sample reference negative in 6/12).",
        "- The epoch-level association between Large AUC improvement and mismatch reduction is directionally compatible but weak (Spearman -0.1825). Routing associations are of comparable magnitude, so routing cannot be dismissed as a competing mediator.",
        "- Oracle/baseline rows are entirely missing. Transfer, independent operational difficulty, and a clean size-versus-difficulty decomposition cannot be claimed.",
        "",
        "### Competing explanations",
        "",
        "- **H2 raw gradient/noise — MEDIUM evidence against as the main explanation.** Small raw gradients and variance fall, but Small SNR is essentially unchanged; Large SNR worsens while Large AUC improves. Noise/SNR alone does not match the performance pattern.",
        "- **H3 private parameters — MEDIUM evidence for a shared component, not exclusivity.** Mismatch falls in every parameter group, with the largest absolute reduction in shared (-0.5069), but private and architecture groups also change substantially (-0.3964 to -0.4523).",
        "- **H4 routing — unresolved (WEAK evidence either way).** Routing changes are modest but appear from the same first observed epoch, and their epoch-level correlations with Large AUC are comparable to the mismatch correlation.",
        "- **H5 checkpoint/convergence — STRONG evidence against a selection-only explanation.** Large improvement persists across common epochs (12/12 positive mean), at the equal arm's checkpoint (12/12), and at the last common epoch (11/12). Best-epoch shift averages only +0.25 epoch and varies from -3 to +5.",
        "- **H6 fixed physical domain/difficulty — MEDIUM evidence against a fixed-domain-only explanation.** Large and Medium role improvements hold for all four seeds in all three rotations; physical-domain magnitudes differ, but no oracle difficulty data are available.",
        "- **H7 instrumentation artifact — STRONG evidence against.** Review-stage ON/OFF checks had identical batches, losses, metrics, and final parameters, and all formal pairs passed identity/fingerprint checks.",
        "",
        "### Overall hypothesis judgment",
        "",
        "**H1 evidence strength: MEDIUM.** The predicted contribution distortion, role rotation, actual-update alignment, and performance improvement are highly repeatable, and shared mismatch changes more than any private group. However, E1 is observational, the dominant mismatch change is the mathematically forced coefficient effect, common-reference trajectory effects are inconsistent, the performance–mismatch association is weak, and routing remains a viable mediator. Therefore E1 supports shared contribution distortion as a mechanism candidate but does not establish it as the primary causal explanation.",
        "",
        "**Domain size as the main explanatory variable: MEDIUM for optimization allocation, WEAK for performance causality.** Role rotation makes size a plausible driver of contribution allocation, but absent oracle difficulty data and simultaneous routing changes prevent a main-cause claim for AUC.",
        "",
        "### Best next falsification experiment",
        "",
        "Proceed to the preregistered E2 objective × sampling factorial. Cross target objective (equal-cell versus sample-proportional) with batch sampling (natural versus domain-balanced), using importance correction so each target objective is unbiased under either sampler. Keep expected loss scale, examples processed, initialization, optimizer, checkpoint rule, and seeds fixed. If AUC follows the target objective across samplers, the coefficient/objective explanation gains causal support; if it follows sampling/exposure or gradient variance instead, H1 is weakened. This is more discriminating than adding another weighting heuristic.",
        "",
        "## Availability",
        "",
        f"Non-missing oracle rows: {oracle_available}. Detailed seed-level tables are in `{OUT.name}/`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "runs": 24,
        "pairs": 12,
        "oracle_rows": oracle_available,
        "overall_auc_delta": float(overall_stats.loc[0, "mean"]),
        "large_auc_delta": float(role_auc.loc[role_auc.size_role == "L", "mean"].iloc[0]),
        "medium_auc_delta": float(role_auc.loc[role_auc.size_role == "M", "mean"].iloc[0]),
        "small_auc_delta": float(role_auc.loc[role_auc.size_role == "S", "mean"].iloc[0]),
        "mismatch_total_delta": float(mismatch.total_actual.mean()),
        "spearman_large_auc_vs_mismatch_delta": float(corr),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
