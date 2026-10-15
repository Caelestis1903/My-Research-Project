"""Stage S7-A mechanical aggregation analysis.

Reads only:
  experiments/stage_S7_outputs/stage_S7_gradient_raw.csv
  experiments/stage_S7_outputs/stage_S7_performance_raw.csv

Writes tables to experiments/stage_S7_analysis_tables/.

The script is purely deterministic aggregation. It does not modify any
S1-S7 source file.

Data duplication facts (verified against the raw CSV before writing tables):
  - 6 cells per (dataset, seed, checkpoint, module) -> 36 rows.
  - For fixed (dataset, seed, checkpoint, source_cell, module, target_cell)
    gradient_norm_mean, sample_alignment_mean, source_weight_mean are
    repeated across target_cell (i.e. they are properties of the source).
    They are deduped by source_cell for T2 and T4.
  - For fixed (dataset, seed, checkpoint, target_cell, module) the
    validation_dot_product_mean and validation_gradient_norm are
    repeated across source_cell (i.e. they are properties of the target).
    They are deduped by target_cell for T6.
  - conflict_contribution_mean, pairwise_cosine_mean and
    weighted_source_target_contribution_mean vary along (source, target).
  - performance_raw aggregates (validation_macro_auc,
    validation_target_weighted_auc, validation_target_domain_auc,
    validation_non_target_weighted_auc, validation_macro_logloss,
    validation_target_weighted_logloss, test_macro_auc,
    test_target_domain_auc, test_target_weighted_auc, test_macro_logloss)
    are the same for all 6 cells of a (dataset, seed, checkpoint) row.
    Deduped by (dataset, seed, checkpoint) for T1.
  - pair_class depends on (source, target) and is the relationship between
    source.domain/task and target.domain/task.
  - Identity check (T6): for ALL module,
        sum_i H_{i->k} == validation_dot_product_mean (U_k) within 1e-6.
    This was verified during script development and re-asserted at the
    end of this script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "stage_S7_outputs"
OUT = ROOT / "stage_S7_analysis_tables"
OUT.mkdir(parents=True, exist_ok=True)

GRAD_CSV = SRC / "stage_S7_gradient_raw.csv"
PERF_CSV = SRC / "stage_S7_performance_raw.csv"

SEEDS = (2022, 2023, 2024, 2025)
DATASETS = ("kuairand", "movielens")
EPOCH_ORDER = ("epoch_9", "epoch_19", "epoch_29", "epoch_39", "epoch_49", "selected")
EARLY = ("epoch_9", "epoch_19")
LATE = ("epoch_39", "epoch_49")
PERF_COLS = [
    "validation_macro_auc",
    "validation_target_weighted_auc",
    "validation_target_domain_auc",
    "validation_non_target_weighted_auc",
    "validation_macro_logloss",
    "validation_target_weighted_logloss",
    "validation_target_domain_logloss",
    "macro_auc",
    "target_domain_auc",
    "target_weighted_auc",
    "macro_logloss",
]
PERF_OUTPUT_NAMES = {
    "macro_auc": "test_macro_auc",
    "target_domain_auc": "test_target_domain_auc",
    "target_weighted_auc": "test_target_weighted_auc",
    "macro_logloss": "test_macro_logloss",
}


def _sign_agreement(diff_series: pd.Series) -> dict:
    pos = int((diff_series > 0).sum())
    neg = int((diff_series < 0).sum())
    zero = int((diff_series == 0).sum())
    return {
        "n_pos": pos,
        "n_neg": neg,
        "n_zero": zero,
        "sign_consistent_3of4": max(pos, neg) >= 3,
        "sign_consistent_4of4": max(pos, neg) == 4,
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    grad = pd.read_csv(GRAD_CSV)
    perf = pd.read_csv(PERF_CSV)
    return grad, perf


def table_t1(perf: pd.DataFrame) -> pd.DataFrame:
    """T1 performance trajectory: per (dataset, seed, checkpoint) aggregates
    plus delta vs epoch_9 of the same seed."""
    cols = ["dataset", "seed", "checkpoint"] + PERF_COLS
    base = perf[cols].drop_duplicates(subset=["dataset", "seed", "checkpoint"]).copy()
    base = base.sort_values(["dataset", "seed", "checkpoint"]).reset_index(drop=True)
    out_rows = []
    for (ds, sd), grp in base.groupby(["dataset", "seed"], sort=False):
        epoch9 = grp[grp["checkpoint"] == "epoch_9"]
        if epoch9.empty:
            continue
        ref = epoch9.iloc[0]
        for _, row in grp.iterrows():
            out = {
                "dataset": ds,
                "seed": sd,
                "checkpoint": row["checkpoint"],
            }
            for c in PERF_COLS:
                out_c = PERF_OUTPUT_NAMES.get(c, c)
                out[out_c] = row[c]
                out[f"delta_{out_c}_vs_epoch9"] = row[c] - ref[c]
            out_rows.append(out)
    df = pd.DataFrame(out_rows)
    return df


def _grad_norm_one_source(g: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, seed, checkpoint, source_cell) for module=ALL."""
    sub = g[(g["shared_module"] == "ALL") & (g["pair_class"] == "self")]
    sub = sub[["dataset", "seed", "checkpoint", "source_cell", "gradient_norm_mean"]].copy()
    sub = sub.drop_duplicates(subset=["dataset", "seed", "checkpoint", "source_cell"])
    return sub


def table_t2(grad: pd.DataFrame) -> pd.DataFrame:
    """T2 gradient norm trajectory per source_cell with rank and ratio vs
    epoch_9."""
    base = _grad_norm_one_source(grad)
    base = base.sort_values(["dataset", "seed", "checkpoint", "source_cell"]).reset_index(drop=True)
    base["rank_within_checkpoint"] = base.groupby(["dataset", "seed", "checkpoint"])["gradient_norm_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    # ratio_vs_epoch9 is computed relative to the same (ds, sd, source_cell)
    # at epoch_9.
    ref = (
        base[base["checkpoint"] == "epoch_9"]
        .set_index(["dataset", "seed", "source_cell"])["gradient_norm_mean"]
    )
    base["ratio_vs_epoch9"] = base.apply(
        lambda r: (
            r["gradient_norm_mean"] / ref.loc[(r["dataset"], r["seed"], r["source_cell"])]
            if (r["dataset"], r["seed"], r["source_cell"]) in ref.index
            and ref.loc[(r["dataset"], r["seed"], r["source_cell"])] != 0
            else float("nan")
        ),
        axis=1,
    )
    return base.reset_index(drop=True)


def _cosine_matrix(grad: pd.DataFrame, module: str = "ALL") -> pd.DataFrame:
    sub = grad[grad["shared_module"] == module][
        ["dataset", "seed", "checkpoint", "source_cell", "target_cell", "pairwise_cosine_mean", "pair_class"]
    ].copy()
    return sub


def table_t3(grad: pd.DataFrame) -> pd.DataFrame:
    return _cosine_matrix(grad, "ALL")


def table_t3b(grad: pd.DataFrame) -> pd.DataFrame:
    sub = _cosine_matrix(grad, "ALL")
    sub = sub[sub["pair_class"] != "self"]
    g = (
        sub.groupby(["dataset", "seed", "checkpoint", "pair_class"], as_index=False)["pairwise_cosine_mean"]
        .mean()
        .rename(columns={"pairwise_cosine_mean": "pairwise_cosine_mean_over_cells"})
    )
    return g


def _alignment(grad: pd.DataFrame) -> pd.DataFrame:
    """sample_alignment_mean per source_cell for module=ALL.

    Verified that sample_alignment_mean repeats across target_cell for fixed
    source_cell; dedup by source_cell.
    """
    sub = grad[(grad["shared_module"] == "ALL") & (grad["target_cell"] == 0)][
        ["dataset", "seed", "checkpoint", "source_cell", "sample_alignment_mean", "sample_alignment_sd"]
    ].copy()
    return sub.drop_duplicates(subset=["dataset", "seed", "checkpoint", "source_cell"])


def table_t4(grad: pd.DataFrame) -> pd.DataFrame:
    return _alignment(grad).sort_values(["dataset", "seed", "checkpoint", "source_cell"]).reset_index(drop=True)


def _conflict_matrix(grad: pd.DataFrame, module: str = "ALL") -> pd.DataFrame:
    return grad[grad["shared_module"] == module][
        ["dataset", "seed", "checkpoint", "source_cell", "target_cell", "pair_class", "conflict_contribution_mean"]
    ].copy()


def table_t5(grad: pd.DataFrame) -> pd.DataFrame:
    """ConflictMass = sum_{i!=j} conflict_contribution_mean (ALL module).

    Verified that the (i,j) and (j,i) cells are numerically equal up to
    float precision; we therefore sum over the full 6x6 matrix excluding
    the diagonal (i==j) which is 0. The result is reported as a single
    scalar per (dataset, seed, checkpoint).
    """
    sub = _conflict_matrix(grad, "ALL")
    sub = sub[sub["source_cell"] != sub["target_cell"]]
    g = (
        sub.groupby(["dataset", "seed", "checkpoint"], as_index=False)["conflict_contribution_mean"]
        .sum()
        .rename(columns={"conflict_contribution_mean": "conflict_mass"})
    )
    return g


def _validation_dedup(grad: pd.DataFrame) -> pd.DataFrame:
    """Dedup validation_dot_product_mean and validation_gradient_norm by
    target_cell (repeated across source_cell)."""
    sub = grad[grad["shared_module"] == "ALL"][
        [
            "dataset",
            "seed",
            "checkpoint",
            "target_cell",
            "validation_dot_product_mean",
            "validation_gradient_norm",
        ]
    ].copy()
    sub = sub.drop_duplicates(subset=["dataset", "seed", "checkpoint", "target_cell"])
    return sub


def _source_norm_and_weight(grad: pd.DataFrame) -> pd.DataFrame:
    """Per source_cell: gradient_norm_mean, source_weight_mean (dedup)."""
    sub = grad[grad["shared_module"] == "ALL"][
        ["dataset", "seed", "checkpoint", "source_cell", "gradient_norm_mean", "source_weight_mean"]
    ].copy()
    sub = sub.drop_duplicates(subset=["dataset", "seed", "checkpoint", "source_cell"])
    return sub


def table_t6(grad: pd.DataFrame) -> pd.DataFrame:
    """T6 validation utility U_k per target_cell, plus two normalisations
    and a list of cells with U_k<0.

    Notes on normalisation:
      g_S is the per-source weighted gradient g_S = sum_i w_i g_i. Its norm
      cannot be reconstructed from the per-row fields because we only have
      scalar aggregates (gradient_norm_mean, source_weight_mean,
      pairwise_cosine_mean) per (source, target) pair, not the full
      vectors. We therefore provide two normalisations:
        1) U_k / ||g_k^val||         (always defined; equals U_k/||g_k||)
        2) U_k / (||g_k^val|| * gS_norm_proxy)
           where gS_norm_proxy = sum_i w_i * gradient_norm_mean (an upper
           bound on ||g_S|| via triangle inequality with a sign-cosine
           sign, included as a diagnostic). Documented as proxy only.
    """
    val = _validation_dedup(grad)
    src = _source_norm_and_weight(grad)
    gS = (
        src.assign(_wN=lambda d: d["source_weight_mean"] * d["gradient_norm_mean"])
        .groupby(["dataset", "seed", "checkpoint"], as_index=False)["_wN"]
        .sum()
        .rename(columns={"_wN": "gS_norm_proxy"})
    )
    df = val.merge(gS, on=["dataset", "seed", "checkpoint"], how="left")
    df["U_k_div_gk_norm"] = df["validation_dot_product_mean"] / df["validation_gradient_norm"]
    df["U_k_div_gk_div_gS_proxy"] = df["validation_dot_product_mean"] / (
        df["validation_gradient_norm"] * df["gS_norm_proxy"]
    )
    df = df.sort_values(["dataset", "seed", "checkpoint", "target_cell"]).reset_index(drop=True)

    # Per (dataset, seed, checkpoint) list of cells with U_k<0
    neg_list = (
        df[df["validation_dot_product_mean"] < 0]
        .groupby(["dataset", "seed", "checkpoint"])["target_cell"]
        .apply(lambda s: ";".join(str(int(x)) for x in sorted(s)))
        .reset_index()
        .rename(columns={"target_cell": "U_k_neg_cells"})
    )
    df = df.merge(neg_list, on=["dataset", "seed", "checkpoint"], how="left")
    df["U_k_neg_cells"] = df["U_k_neg_cells"].fillna("")
    return df


def _influence(grad: pd.DataFrame) -> pd.DataFrame:
    return grad[grad["shared_module"] == "ALL"][
        [
            "dataset",
            "seed",
            "checkpoint",
            "source_cell",
            "target_cell",
            "pair_class",
            "weighted_source_target_contribution_mean",
            "weighted_source_target_contribution_sd",
        ]
    ].copy()


def table_t7(grad: pd.DataFrame) -> pd.DataFrame:
    return _influence(grad).sort_values(
        ["dataset", "seed", "checkpoint", "source_cell", "target_cell"]
    ).reset_index(drop=True)


def table_t7b(grad: pd.DataFrame) -> pd.DataFrame:
    sub = _influence(grad)
    g = (
        sub.groupby(["dataset", "seed", "checkpoint", "pair_class"], as_index=False)
        .agg(
            H_sum=("weighted_source_target_contribution_mean", "sum"),
            H_mean=("weighted_source_target_contribution_mean", "mean"),
            n_pairs=("weighted_source_target_contribution_mean", "size"),
        )
    )
    return g


def _domain_task_meta(grad: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, source_cell) giving source.domain/task labels.

    We additionally pull (target_domain, target_task) per (dataset, target_cell).
    """
    src = grad[["dataset", "source_cell", "source_domain", "source_task"]].drop_duplicates()
    src = src.rename(columns={"source_domain": "src_domain_label", "source_task": "src_task_label"})
    tgt = grad[["dataset", "target_cell", "target_domain", "target_task"]].drop_duplicates()
    tgt = tgt.rename(columns={"target_domain": "tgt_domain_label", "target_task": "tgt_task_label"})
    return src, tgt


def table_t7c(grad: pd.DataFrame) -> pd.DataFrame:
    sub = _influence(grad)
    src_meta, tgt_meta = _domain_task_meta(grad)
    sub = sub.merge(src_meta, on=["dataset", "source_cell"], how="left").merge(
        tgt_meta, on=["dataset", "target_cell"], how="left"
    )
    sub = sub.rename(columns={"weighted_source_target_contribution_mean": "H_value"})

    out = (
        sub.sort_values("H_value", ascending=True)
        .groupby(["dataset", "seed", "checkpoint"], as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    # share = |H_value| / sum_of_all_negative_H_abs, computed per
    # (dataset, seed, checkpoint) over the full 6x6 matrix.
    full_neg_total = (
        sub[sub["H_value"] < 0]
        .assign(_abs=lambda d: d["H_value"].abs())
        .groupby(["dataset", "seed", "checkpoint"], as_index=False)["_abs"]
        .sum()
        .rename(columns={"_abs": "_neg_total_abs"})
    )
    out = out.merge(full_neg_total, on=["dataset", "seed", "checkpoint"], how="left")
    out["share_of_negative_sum"] = out["H_value"].abs() / out["_neg_total_abs"].replace(0, np.nan)
    out["share_of_negative_sum"] = out["share_of_negative_sum"].fillna(0.0)
    # Share only makes sense for negative H values; zero out for non-negative.
    out.loc[out["H_value"] >= 0, "share_of_negative_sum"] = 0.0
    out = out.drop(columns=["_neg_total_abs"])
    return out[
        [
            "dataset",
            "seed",
            "checkpoint",
            "source_cell",
            "target_cell",
            "pair_class",
            "src_domain_label",
            "src_task_label",
            "tgt_domain_label",
            "tgt_task_label",
            "H_value",
            "share_of_negative_sum",
        ]
    ]


def table_t8(grad: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """T8 early/late contrast across all required quantities."""
    rows: list[dict] = []

    # --- (1) ConflictMass from T5 ---
    t5 = table_t5(grad)
    for (ds, sd), _ in t5.groupby(["dataset", "seed"], sort=False):
        e = t5[
            (t5["dataset"] == ds) & (t5["seed"] == sd) & (t5["checkpoint"].isin(EARLY))
        ]["conflict_mass"].mean()
        l = t5[
            (t5["dataset"] == ds) & (t5["seed"] == sd) & (t5["checkpoint"].isin(LATE))
        ]["conflict_mass"].mean()
        rows.append(
            {
                "dataset": ds,
                "seed": sd,
                "metric": "ConflictMass",
                "group": "global",
                "early": e,
                "late": l,
                "diff": l - e,
            }
        )

    # --- (2) U_k per target_cell (T6) ---
    t6 = table_t6(grad)
    for (ds, sd, tc), grp in t6.groupby(["dataset", "seed", "target_cell"]):
        e = grp[grp["checkpoint"].isin(EARLY)]["validation_dot_product_mean"].mean()
        l = grp[grp["checkpoint"].isin(LATE)]["validation_dot_product_mean"].mean()
        rows.append(
            {
                "dataset": ds,
                "seed": sd,
                "metric": "U_k",
                "group": f"target_cell={tc}",
                "early": e,
                "late": l,
                "diff": l - e,
            }
        )

    # --- (3) H per source->target pair (T7) ---
    t7 = table_t7(grad)
    for (ds, sd, sc, tc), grp in t7.groupby(["dataset", "seed", "source_cell", "target_cell"]):
        e = grp[grp["checkpoint"].isin(EARLY)]["weighted_source_target_contribution_mean"].mean()
        l = grp[grp["checkpoint"].isin(LATE)]["weighted_source_target_contribution_mean"].mean()
        rows.append(
            {
                "dataset": ds,
                "seed": sd,
                "metric": "H_pair",
                "group": f"{sc}->{tc}",
                "early": e,
                "late": l,
                "diff": l - e,
            }
        )

    # --- (4) N per cell (T2) ---
    t2 = table_t2(grad)
    for (ds, sd, sc), grp in t2.groupby(["dataset", "seed", "source_cell"]):
        e = grp[grp["checkpoint"].isin(EARLY)]["gradient_norm_mean"].mean()
        l = grp[grp["checkpoint"].isin(LATE)]["gradient_norm_mean"].mean()
        rows.append(
            {
                "dataset": ds,
                "seed": sd,
                "metric": "N_cell",
                "group": f"source_cell={sc}",
                "early": e,
                "late": l,
                "diff": l - e,
            }
        )

    # --- (5) validation_target_domain_auc / logloss from performance ---
    t1 = table_t1(perf)
    for c_src, c_out in (
        ("validation_target_domain_auc", "validation_target_domain_auc"),
        ("validation_target_domain_logloss", "validation_target_domain_logloss"),
    ):
        for (ds, sd), grp in t1.groupby(["dataset", "seed"]):
            e = grp[grp["checkpoint"].isin(EARLY)][c_out].mean()
            l = grp[grp["checkpoint"].isin(LATE)][c_out].mean()
            rows.append(
                {
                    "dataset": ds,
                    "seed": sd,
                    "metric": c_out,
                    "group": "global",
                    "early": e,
                    "late": l,
                    "diff": l - e,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _aggregate_across_seeds(per_seed: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = per_seed.groupby(group_cols, as_index=False).agg(
        mean_early=("early", "mean"),
        mean_late=("late", "mean"),
        mean_diff=("diff", "mean"),
        seed_sd_diff=("diff", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        n_pos=("diff", lambda s: int((s > 0).sum())),
        n_neg=("diff", lambda s: int((s < 0).sum())),
        n_zero=("diff", lambda s: int((s == 0).sum())),
    )
    return g


def table_t8_summary(t8: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_across_seeds(t8, ["dataset", "metric", "group"])


def table_t9(grad: pd.DataFrame) -> pd.DataFrame:
    """T9 module breakdown for (a) per-dataset most-negative H pair at late
    checkpoints and (b) ConflictMass, by the 7 non-ALL modules.

    Most-negative H pair at late checkpoints is identified per (dataset,
    seed) and at each late checkpoint separately, then the late-pair's
    H_value is broken down by module.
    """
    rows: list[dict] = []
    inf_all = _influence(grad)
    inf_all_late = inf_all[inf_all["checkpoint"].isin(LATE)]

    for (ds, sd, ck), grp in inf_all_late.groupby(["dataset", "seed", "checkpoint"]):
        worst = grp.sort_values("weighted_source_target_contribution_mean").iloc[0]
        sc, tc = int(worst["source_cell"]), int(worst["target_cell"])
        pc = worst["pair_class"]
        sub_mod = grad[
            (grad["dataset"] == ds)
            & (grad["seed"] == sd)
            & (grad["checkpoint"] == ck)
            & (grad["source_cell"] == sc)
            & (grad["target_cell"] == tc)
            & (grad["shared_module"] != "ALL")
        ][["shared_module", "weighted_source_target_contribution_mean"]]
        for _, r in sub_mod.iterrows():
            rows.append(
                {
                    "kind": "worst_H_pair",
                    "dataset": ds,
                    "seed": sd,
                    "checkpoint": ck,
                    "source_cell": sc,
                    "target_cell": tc,
                    "pair_class": pc,
                    "shared_module": r["shared_module"],
                    "value": r["weighted_source_target_contribution_mean"],
                }
            )

    cm_mod = grad[
        (grad["shared_module"] != "ALL") & (grad["source_cell"] != grad["target_cell"])
    ]
    cm_g = (
        cm_mod.groupby(["dataset", "seed", "checkpoint", "shared_module"], as_index=False)[
            "conflict_contribution_mean"
        ]
        .sum()
        .rename(columns={"conflict_contribution_mean": "value"})
    )
    cm_g["kind"] = "ConflictMass"
    cm_g["source_cell"] = -1
    cm_g["target_cell"] = -1
    cm_g["pair_class"] = "ALL_PAIRS"

    parts = [pd.DataFrame(rows), cm_g[["kind", "dataset", "seed", "checkpoint", "source_cell",
                                       "target_cell", "pair_class", "shared_module", "value"]]]
    return pd.concat(parts, ignore_index=True).sort_values(
        ["kind", "dataset", "seed", "checkpoint", "shared_module"]
    ).reset_index(drop=True)


def table_t10(grad: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """T10 dataset-by-dataset contrast of trajectory means/SDs across
    checkpoints."""
    cm = table_t5(grad)
    cm = cm.rename(columns={"conflict_mass": "ConflictMass_mean"})
    cm_agg = cm.groupby(["dataset", "checkpoint"], as_index=False)["ConflictMass_mean"].agg(
        mean="mean", sd=lambda s: float(np.std(s, ddof=1))
    ).rename(columns={"mean": "ConflictMass_mean", "sd": "ConflictMass_sd"})

    t6 = table_t6(grad)
    minU = (
        t6.groupby(["dataset", "seed", "checkpoint"], as_index=False)["validation_dot_product_mean"]
        .min()
        .rename(columns={"validation_dot_product_mean": "min_U_k"})
    )
    minU_agg = minU.groupby(["dataset", "checkpoint"], as_index=False)["min_U_k"].agg(
        mean="mean", sd=lambda s: float(np.std(s, ddof=1))
    ).rename(columns={"mean": "min_U_k_mean", "sd": "min_U_k_sd"})

    cos = _cosine_matrix(grad, "ALL")
    cos = cos[cos["pair_class"] != "self"]
    mean_cos = (
        cos.groupby(["dataset", "seed", "checkpoint"], as_index=False)["pairwise_cosine_mean"]
        .mean()
        .rename(columns={"pairwise_cosine_mean": "mean_cos_offdiag"})
    )
    mean_cos_agg = mean_cos.groupby(["dataset", "checkpoint"], as_index=False)["mean_cos_offdiag"].agg(
        mean="mean", sd=lambda s: float(np.std(s, ddof=1))
    ).rename(columns={"mean": "mean_cos_offdiag_mean", "sd": "mean_cos_offdiag_sd"})

    t1 = table_t1(perf)
    val_agg = t1.groupby(["dataset", "checkpoint"], as_index=False).agg(
        validation_target_domain_auc_mean=("validation_target_domain_auc", "mean"),
        validation_target_domain_auc_sd=("validation_target_domain_auc", lambda s: float(np.std(s, ddof=1))),
    )

    out = (
        cm_agg.merge(minU_agg, on=["dataset", "checkpoint"], how="outer")
        .merge(mean_cos_agg, on=["dataset", "checkpoint"], how="outer")
        .merge(val_agg, on=["dataset", "checkpoint"], how="outer")
    )
    return out.sort_values(["checkpoint", "dataset"]).reset_index(drop=True)


def table_t11(t8_summary: pd.DataFrame) -> pd.DataFrame:
    """T11 seed-consistency classification per metric / group."""
    out = t8_summary.copy()
    out["n_seeds_same_sign"] = out[["n_pos", "n_neg"]].max(axis=1)
    out["robust_flag"] = out["mean_diff"].abs() > out["seed_sd_diff"]
    return out[
        [
            "dataset",
            "metric",
            "group",
            "mean_early",
            "mean_late",
            "mean_diff",
            "seed_sd_diff",
            "n_pos",
            "n_neg",
            "n_zero",
            "n_seeds_same_sign",
            "robust_flag",
        ]
    ]


def table_t12(grad: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """T12 norm-confound check: late vs early for several norm-aware
    quantities.

    For each (dataset, seed) and per-cell/per-pair identifier, we report
    the early-period mean, late-period mean and their difference.
    """
    rows: list[dict] = []

    src = _source_norm_and_weight(grad)
    src["wN"] = src["source_weight_mean"] * src["gradient_norm_mean"]
    sum_wN = src.groupby(["dataset", "seed", "checkpoint"], as_index=False)["wN"].sum().rename(
        columns={"wN": "sum_wN"}
    )

    val = _validation_dedup(grad)
    val["U_div_gk"] = val["validation_dot_product_mean"] / val["validation_gradient_norm"]

    inf = _influence(grad)
    src2 = src[["dataset", "seed", "checkpoint", "source_cell", "gradient_norm_mean"]].rename(
        columns={"gradient_norm_mean": "N_i"}
    )
    val2 = val[["dataset", "seed", "checkpoint", "target_cell", "validation_gradient_norm"]].rename(
        columns={"validation_gradient_norm": "gk_norm"}
    )
    merged = inf.merge(src2, on=["dataset", "seed", "checkpoint", "source_cell"], how="left").merge(
        val2, on=["dataset", "seed", "checkpoint", "target_cell"], how="left"
    )
    merged["H_div_Ni_gk"] = merged["weighted_source_target_contribution_mean"] / (
        merged["N_i"] * merged["gk_norm"]
    )

    for kind, df, valcol, key_cols in [
        ("sum_wN", sum_wN, "sum_wN", []),
        ("U_k_div_gk_norm", val, "U_div_gk", ["target_cell"]),
        ("H_div_Ni_gk", merged, "H_div_Ni_gk", ["source_cell", "target_cell"]),
    ]:
        for key_vals, sub in df.groupby(["dataset", "seed"] + key_cols, sort=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            ds, sd = key_vals[0], key_vals[1]
            extra = key_vals[2:] if len(key_vals) > 2 else ()
            early = sub[sub["checkpoint"].isin(EARLY)][valcol].mean()
            late = sub[sub["checkpoint"].isin(LATE)][valcol].mean()
            row = {
                "metric": kind,
                "dataset": ds,
                "seed": sd,
                "early": early,
                "late": late,
                "diff": late - early,
            }
            for k, v in zip(key_cols, extra):
                row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    grad, perf = load_data()
    print(
        f"[load] grad rows={len(grad)} perf rows={len(perf)} "
        f"datasets={sorted(grad['dataset'].unique())} seeds={sorted(grad['seed'].unique())}"
    )

    # Sanity assertions
    assert len(grad) == 13824, f"unexpected grad rows: {len(grad)}"
    assert len(perf) == 288, f"unexpected perf rows: {len(perf)}"
    grad_mod = grad[grad["shared_module"] == "ALL"]
    assert len(grad_mod) == 13824 // 8, f"unexpected ALL rows: {len(grad_mod)}"
    # 5 epoch_* checkpoints x 2 datasets x 4 seeds x 6 cells x 6 cells = 1440
    grad_epoch_all = grad_mod[grad_mod["checkpoint"].isin(EPOCH_ORDER[:5])]
    assert len(grad_epoch_all) == 2 * 4 * 5 * 36, f"unexpected epoch ALL rows: {len(grad_epoch_all)}"
    # 'selected' is also a checkpoint in data; 2*4*1*36 = 288 additional rows
    grad_sel_all = grad_mod[grad_mod["checkpoint"] == "selected"]
    assert len(grad_sel_all) == 2 * 4 * 36, f"unexpected selected ALL rows: {len(grad_sel_all)}"

    # T1
    t1 = table_t1(perf)
    t1.to_csv(OUT / "T1_performance_trajectory.csv", index=False)

    # T2
    t2 = table_t2(grad)
    t2.to_csv(OUT / "T2_gradient_norm_trajectory.csv", index=False)

    # T3
    t3 = table_t3(grad)
    t3.to_csv(OUT / "T3_cosine_matrix.csv", index=False)
    t3b = table_t3b(grad)
    t3b.to_csv(OUT / "T3b_cosine_pairclass_summary.csv", index=False)

    # T4
    t4 = table_t4(grad)
    t4.to_csv(OUT / "T4_sample_alignment.csv", index=False)

    # T5
    t5 = table_t5(grad)
    t5.to_csv(OUT / "T5_conflict_mass.csv", index=False)

    # T6
    t6 = table_t6(grad)
    t6.to_csv(OUT / "T6_validation_utility.csv", index=False)

    # T7
    t7 = table_t7(grad)
    t7.to_csv(OUT / "T7_influence_matrix.csv", index=False)
    t7b = table_t7b(grad)
    t7b.to_csv(OUT / "T7b_influence_pairclass_summary.csv", index=False)
    t7c = table_t7c(grad)
    t7c.to_csv(OUT / "T7c_top_harmful_pairs.csv", index=False)

    # T8 + summary
    t8 = table_t8(grad, perf)
    t8.to_csv(OUT / "T8_early_late_contrast_per_seed.csv", index=False)
    t8s = table_t8_summary(t8)
    t8s.to_csv(OUT / "T8_early_late_contrast_summary.csv", index=False)

    # T9
    t9 = table_t9(grad)
    t9.to_csv(OUT / "T9_module_breakdown.csv", index=False)

    # T10
    t10 = table_t10(grad, perf)
    t10.to_csv(OUT / "T10_movie_vs_kuai_pattern.csv", index=False)

    # T11
    t11 = table_t11(t8s)
    t11.to_csv(OUT / "T11_seed_consistency.csv", index=False)

    # T12
    t12 = table_t12(grad, perf)
    t12.to_csv(OUT / "T12_norm_confound_check.csv", index=False)

    # Identity check: sum_i H_{i->k} == U_k  (ALL module)
    val = _validation_dedup(grad)
    inf = _influence(grad)
    sum_H = inf.groupby(["dataset", "seed", "checkpoint", "target_cell"], as_index=False)[
        "weighted_source_target_contribution_mean"
    ].sum().rename(columns={"weighted_source_target_contribution_mean": "sum_H_ik"})
    chk = val.merge(sum_H, on=["dataset", "seed", "checkpoint", "target_cell"], how="left")
    chk["abs_err"] = (chk["sum_H_ik"] - chk["validation_dot_product_mean"]).abs()
    max_err = float(chk["abs_err"].max())
    print(f"[identity] max|sum_i H_ik - U_k| over ALL = {max_err:.3e}")
    print(f"[identity] rows checked = {len(chk)}")
    assert max_err < 1e-6, f"identity violation: max_err={max_err}"

    # Conflict matrix symmetry check (within float precision). Exclude
    # the diagonal because it is 0 (or absent) by construction.
    cm = _conflict_matrix(grad, "ALL")
    sym_errs = []
    for (ds, sd, ck), grp in cm.groupby(["dataset", "seed", "checkpoint"]):
        mat = np.zeros((6, 6))
        for _, r in grp.iterrows():
            i, j = int(r["source_cell"]), int(r["target_cell"])
            mat[i, j] = r["conflict_contribution_mean"]
        mask = ~np.eye(6, dtype=bool)
        sym_errs.append(np.abs(mat[mask] - mat.T[mask]).max())
    sym_err = float(max(sym_errs))
    print(f"[symmetry] max|conflict(i,j)-conflict(j,i)| offdiag = {sym_err:.3e}")
    assert sym_err < 1e-3, f"conflict matrix asymmetry: {sym_err}"

    # Cosine matrix symmetry check (off-diagonal only; diagonal is "self")
    cos_sub = grad[
        (grad["shared_module"] == "ALL")
        & (grad["source_cell"] != grad["target_cell"])
    ][["dataset", "seed", "checkpoint", "source_cell", "target_cell", "pairwise_cosine_mean"]]
    cos_sym_errs = []
    for (ds, sd, ck), grp in cos_sub.groupby(["dataset", "seed", "checkpoint"]):
        mat = np.zeros((6, 6))
        for _, r in grp.iterrows():
            i, j = int(r["source_cell"]), int(r["target_cell"])
            mat[i, j] = r["pairwise_cosine_mean"]
        cos_sym_errs.append(np.abs(mat - mat.T).max())
    cos_sym_err = float(max(cos_sym_errs))
    print(f"[symmetry] max|cos(i,j)-cos(j,i)| offdiag = {cos_sym_err:.3e}")
    assert cos_sym_err < 1e-3, f"cosine matrix asymmetry: {cos_sym_err}"

    # H matrix symmetry check (H_{i->k} vs H_{k->i} not expected equal,
    # but reported in T7). Here we check uniqueness: T7 should have 6*6=36
    # rows per (ds, sd, ck).
    h_rows = t7.groupby(["dataset", "seed", "checkpoint"]).size()
    assert (h_rows == 36).all(), f"T7 expected 36 rows per (ds,sd,ck): {h_rows.unique()}"
    print(f"[T7] H rows per (ds,sd,ck) consistent = True")

    # Report shapes
    print("\nTable row counts:")
    for name, df in [
        ("T1", t1),
        ("T2", t2),
        ("T3", t3),
        ("T3b", t3b),
        ("T4", t4),
        ("T5", t5),
        ("T6", t6),
        ("T7", t7),
        ("T7b", t7b),
        ("T7c", t7c),
        ("T8_per_seed", t8),
        ("T8_summary", t8s),
        ("T9", t9),
        ("T10", t10),
        ("T11", t11),
        ("T12", t12),
    ]:
        print(f"  {name:14s} rows={len(df):>6d}  cols={len(df.columns):>3d}")

    # Quick U_k<0 listing count
    neg_cells = t6[t6["validation_dot_product_mean"] < 0]
    print(f"\n[T6] rows with U_k<0: {len(neg_cells)} ({(neg_cells.shape[0] / len(t6) * 100):.2f}% of rows)")

    # Quick negative H share
    neg_h = t7[t7["weighted_source_target_contribution_mean"] < 0]
    print(
        f"[T7] H<0 cells: {len(neg_h)} / {len(t7)} "
        f"({(neg_h.shape[0] / len(t7) * 100):.2f}%); mean H={t7['weighted_source_target_contribution_mean'].mean():.3e}"
    )

    print("\nAll tables written to:", OUT)


if __name__ == "__main__":
    main()
