"""
Stage S7-B 纯机械统计分析脚本。

输入（只读）：
  - experiments/stage_S7B_outputs/stage_S7B_steps_raw.csv
  - experiments/stage_S7B_outputs/stage_S7B_geometry_raw.csv
  - experiments/stage_S7B_outputs/stage_S7B_validation_raw.csv

输出（写到 experiments/stage_S7B_analysis_tables/）：
  - B0_step0_identity_check.csv
  - B1_branch_contrast_utility.csv
  - B1b_seed_summary.csv
  - B2_harmful_pairs_contrast.csv
  - B2b_harmful_pairs_seed_summary.csv
  - B3_gradient_source_divergence.csv
  - B4_early_late_specificity.csv
  - B5_timing_sequence.csv
  - B6_kuairand_control.csv

口径说明：
  - cell = task * 3 + domain  （cell 索引 task-major）
  - kuairand target domain = 1  → target cells ∈ {1, 4}
  - movielens target domain = 2  → target cells ∈ {2, 5}
  - step 0..4 来自 steps/geometry；step 5 = "final" 仅 validation 有
  - U_k = validation_dot_product（沿 source_cell 重复 6 次，先按 (target_domain, target_task) 去重）
  - H = weighted_source_target_contribution（每个 (source_cell, target_cell) 一个值）
  - diff 统一定义 T − F
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("experiments/stage_S7B_outputs")
OUTPUT_DIR = Path("experiments/stage_S7B_analysis_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS_CSV = INPUT_DIR / "stage_S7B_steps_raw.csv"
GEOM_CSV = INPUT_DIR / "stage_S7B_geometry_raw.csv"
VAL_CSV = INPUT_DIR / "stage_S7B_validation_raw.csv"

# 数据集 → target domain 索引
TARGET_DOMAIN = {"kuairand": 1, "movielens": 2}


def cell_to_domain_task(cell: int) -> tuple[int, int]:
    """cell = task*3 + domain → (domain, task)"""
    return (cell % 3, cell // 3)


def domain_task_to_cell(domain: int, task: int) -> int:
    return task * 3 + domain


def main() -> None:
    print(f"=== Stage S7-B 纯机械统计 ===")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    steps = pd.read_csv(STEPS_CSV)
    geom = pd.read_csv(GEOM_CSV)
    val = pd.read_csv(VAL_CSV)

    print(f"steps   : {steps.shape}")
    print(f"geometry: {geom.shape}")
    print(f"valid.  : {val.shape}")

    # ---------- 派生列 ----------
    # cell（geometry/validation 已有；steps 不需要 cell 字段）
    # domain, task 派生
    for df in (geom, val):
        # cell 已有的话；也允许在 cell=-1 的 scope 聚合行（无 cell）保留
        if "cell" in df.columns:
            mask_cell = df["cell"] >= 0
            df.loc[mask_cell, "domain_calc"] = df.loc[mask_cell, "cell"] % 3
            df.loc[mask_cell, "task_calc"] = df.loc[mask_cell, "cell"] // 3

    # geometry 按 target_cell 去重得到 U_k（一个 target_cell 一条）
    # 同一 (target_domain, target_task) 在同一 (ds, seed, anchor, branch, step, module) 下
    # 6 个 source cell 的 validation_dot_product 完全相等（已验证 nunique=1）。
    geom_uk = (
        geom.groupby(
            [
                "dataset", "seed", "anchor_epoch", "branch", "step",
                "shared_module", "target_domain", "target_task",
            ],
            as_index=False,
        )["validation_dot_product"].first()
    )
    geom_uk["cell"] = geom_uk["target_task"] * 3 + geom_uk["target_domain"]
    # 重复模式注释：每个 (target_cell) 在同一 module 下有 6 行（沿 source_cell 重复），
    # validation_dot_product 在这 6 行上完全相等（U_k），故可直接 .first()。

    # geometry 自身（保留 source_cell 维度以查询 H pair）
    geom_h = geom.copy()  # 唯一键: (ds, seed, anchor, branch, step, module, src_d, src_t, tgt_d, tgt_t)
    # 检查 H 是否沿 source_cell 唯一 —— 每个 source-target 只有一个 H 值
    nunique_check = geom_h.groupby(
        ["dataset", "seed", "anchor_epoch", "branch", "step", "shared_module",
         "source_domain", "source_task", "target_domain", "target_task"]
    )["weighted_source_target_contribution"].nunique().max()
    if nunique_check != 1:
        print(f"WARNING: H 在 (source, target) 组合下不唯一 (max nunique={nunique_check})")

    # ============================================================
    # B0: step0 identity check
    # ============================================================
    b0_rows = []
    b0_warnings = []

    def add_warn(tag: str, ds: str, seed: int, anchor: int, col: str, val: float):
        b0_warnings.append((tag, ds, seed, anchor, col, val))

    s0 = steps[steps["step"] == 0]
    g0 = geom[geom["step"] == 0]
    v0 = val[val["step"] == 0]

    geom_num_cols = [
        "gradient_norm", "pairwise_cosine", "sample_alignment",
        "validation_dot_product", "weighted_source_target_contribution",
        "validation_gradient_norm", "source_weight",
    ]
    val_num_cols = ["val_bce_raw", "auc", "logloss"]
    steps_num_cols = ["L_full", "g_full_norm", "g_target_norm"]

    for (ds, seed, anchor), _ in s0.groupby(["dataset", "seed", "anchor_epoch"]):
        # geometry
        gm = g0[(g0["dataset"] == ds) & (g0["seed"] == seed) & (g0["anchor_epoch"] == anchor)]
        F = gm[gm["branch"] == "F"].set_index(
            ["source_domain", "source_task", "target_domain", "target_task", "shared_module"]
        )[geom_num_cols]
        T = gm[gm["branch"] == "T"].set_index(
            ["source_domain", "source_task", "target_domain", "target_task", "shared_module"]
        )[geom_num_cols]
        m = F.join(T, lsuffix="_F", rsuffix="_T")
        geom_max = max(float((m[c + "_F"] - m[c + "_T"]).abs().max()) for c in geom_num_cols)
        for c in geom_num_cols:
            d = float((m[c + "_F"] - m[c + "_T"]).abs().max())
            if d > 1e-6:
                add_warn("geom", ds, seed, anchor, c, d)

        # validation
        vm = v0[(v0["dataset"] == ds) & (v0["seed"] == seed) & (v0["anchor_epoch"] == anchor)]
        Fv = vm[vm["branch"] == "F"].set_index(["scope", "domain", "task", "cell"])[val_num_cols]
        Tv = vm[vm["branch"] == "T"].set_index(["scope", "domain", "task", "cell"])[val_num_cols]
        mv = Fv.join(Tv, lsuffix="_F", rsuffix="_T")
        val_max = max(float((mv[c + "_F"] - mv[c + "_T"]).abs().max()) for c in val_num_cols)
        for c in val_num_cols:
            d = float((mv[c + "_F"] - mv[c + "_T"]).abs().max())
            if d > 1e-6:
                add_warn("val", ds, seed, anchor, c, d)

        # steps
        sm = s0[(s0["dataset"] == ds) & (s0["seed"] == seed) & (s0["anchor_epoch"] == anchor)]
        Fs = sm[sm["branch"] == "F"].iloc[0]
        Ts = sm[sm["branch"] == "T"].iloc[0]
        steps_max = max(abs(Fs[c] - Ts[c]) for c in steps_num_cols)
        L_full_diff = float(Fs["L_full"] - Ts["L_full"])
        g_full_diff = float(Fs["g_full_norm"] - Ts["g_full_norm"])
        for c in steps_num_cols:
            d = abs(Fs[c] - Ts[c])
            if d > 1e-6:
                add_warn("steps", ds, seed, anchor, c, d)

        b0_rows.append({
            "dataset": ds, "seed": seed, "anchor": anchor,
            "geom_max_abs_diff": geom_max,
            "val_max_abs_diff": val_max,
            "steps_max_abs_diff": float(steps_max),
            "L_full_F_minus_T": L_full_diff,
            "g_full_norm_F_minus_T": g_full_diff,
        })

    b0 = pd.DataFrame(b0_rows)
    b0_path = OUTPUT_DIR / "B0_step0_identity_check.csv"
    b0.to_csv(b0_path, index=False)
    print(f"[B0] {len(b0)} rows → {b0_path.name}")

    # ============================================================
    # 准备 B1-B6 共享数据
    # ============================================================
    # target cells
    target_cells = {
        "kuairand": [domain_task_to_cell(1, 0), domain_task_to_cell(1, 1)],   # 1, 4
        "movielens": [domain_task_to_cell(2, 0), domain_task_to_cell(2, 1)],  # 2, 5
    }

    # ALL module 下 U_k (target_cell 去重后)
    uk_all = geom_uk[geom_uk["shared_module"] == "ALL"].copy()

    # H（ALL module）
    h_all = geom_h[geom_h["shared_module"] == "ALL"].copy()
    h_all["src_cell"] = h_all["source_task"] * 3 + h_all["source_domain"]
    h_all["tgt_cell"] = h_all["target_task"] * 3 + h_all["target_domain"]

    # validation scope 行
    val_scope = val[val["scope"].isin(["macro", "target_domain", "non_target_weighted"])].copy()
    # validation cell 行（target cells 用）
    val_cell = val[val["scope"] == "cell"].copy()

    # steps 表（用于 theta_distance_shared, g_full_norm, g_target_norm, delta_*, cosine_*）
    steps_num = steps[
        ["dataset", "seed", "anchor_epoch", "branch", "step",
         "L_full", "g_full_norm", "g_target_norm", "cosine_gf_gt",
         "delta_full_norm", "delta_target_norm", "delta_applied_norm",
         "delta_cosine", "theta_distance_shared"]
    ].copy()

    # ============================================================
    # B1: branch contrast utility
    # ============================================================
    rows = []
    for ds in steps["dataset"].unique():
        for seed in sorted(steps[steps["dataset"] == ds]["seed"].unique()):
            for anchor in [9, 39]:
                for step in [1, 2, 3, 4, 5]:
                    # U_k (target cells), per branch
                    row = {"dataset": ds, "seed": seed, "anchor": anchor, "step": step}

                    for tc in target_cells[ds]:
                        # 取 step 对应行：step∈{1..4} 直接；step=5(final) 无 geometry → 标 NaN
                        if step in (1, 2, 3, 4):
                            F_uk = uk_all[(uk_all["dataset"] == ds) & (uk_all["seed"] == seed)
                                          & (uk_all["anchor_epoch"] == anchor) & (uk_all["step"] == step)
                                          & (uk_all["branch"] == "F") & (uk_all["cell"] == tc)]
                            T_uk = uk_all[(uk_all["dataset"] == ds) & (uk_all["seed"] == seed)
                                          & (uk_all["anchor_epoch"] == anchor) & (uk_all["step"] == step)
                                          & (uk_all["branch"] == "T") & (uk_all["cell"] == tc)]
                            Fv = float(F_uk["validation_dot_product"].iloc[0]) if len(F_uk) else np.nan
                            Tv = float(T_uk["validation_dot_product"].iloc[0]) if len(T_uk) else np.nan
                        else:
                            Fv = Tv = np.nan
                        row[f"U_F_cell{tc}"] = Fv
                        row[f"U_T_cell{tc}"] = Tv
                        row[f"U_diff_cell{tc}"] = Tv - Fv

                    # min_U_k（6 cell 最小值；final NaN）
                    if step in (1, 2, 3, 4):
                        F_all6 = uk_all[(uk_all["dataset"] == ds) & (uk_all["seed"] == seed)
                                        & (uk_all["anchor_epoch"] == anchor) & (uk_all["step"] == step)
                                        & (uk_all["branch"] == "F")]["validation_dot_product"]
                        T_all6 = uk_all[(uk_all["dataset"] == ds) & (uk_all["seed"] == seed)
                                        & (uk_all["anchor_epoch"] == anchor) & (uk_all["step"] == step)
                                        & (uk_all["branch"] == "T")]["validation_dot_product"]
                        min_F = float(F_all6.min()) if len(F_all6) else np.nan
                        min_T = float(T_all6.min()) if len(T_all6) else np.nan
                    else:
                        min_F = min_T = np.nan
                    row["min_U_F"] = min_F
                    row["min_U_T"] = min_T
                    row["min_U_diff"] = min_T - min_F

                    # scope 指标
                    for scope in ["target_domain", "non_target_weighted", "macro"]:
                        for metric in ["val_bce_raw", "auc", "logloss"]:
                            Fv = val_scope[(val_scope["dataset"] == ds) & (val_scope["seed"] == seed)
                                           & (val_scope["anchor_epoch"] == anchor)
                                           & (val_scope["branch"] == "F")
                                           & (val_scope["step"] == step) & (val_scope["scope"] == scope)]
                            Tv = val_scope[(val_scope["dataset"] == ds) & (val_scope["seed"] == seed)
                                           & (val_scope["anchor_epoch"] == anchor)
                                           & (val_scope["branch"] == "T")
                                           & (val_scope["step"] == step) & (val_scope["scope"] == scope)]
                            F_val = float(Fv[metric].iloc[0]) if len(Fv) else np.nan
                            T_val = float(Tv[metric].iloc[0]) if len(Tv) else np.nan
                            row[f"{scope}_{metric}_F"] = F_val
                            row[f"{scope}_{metric}_T"] = T_val
                            row[f"{scope}_{metric}_diff"] = T_val - F_val

                    # theta_distance_shared（final 步用 step4 行 + 注明）
                    if step in (1, 2, 3, 4):
                        sF = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "F") & (steps_num["step"] == step)]
                        sT = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "T") & (steps_num["step"] == step)]
                        theta_F = float(sF["theta_distance_shared"].iloc[0]) if len(sF) else np.nan
                        theta_T = float(sT["theta_distance_shared"].iloc[0]) if len(sT) else np.nan
                        note = "native"
                    else:
                        sF = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "F") & (steps_num["step"] == 4)]
                        sT = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "T") & (steps_num["step"] == 4)]
                        theta_F = float(sF["theta_distance_shared"].iloc[0]) if len(sF) else np.nan
                        theta_T = float(sT["theta_distance_shared"].iloc[0]) if len(sT) else np.nan
                        note = "from_step4_geometry_has_no_final"
                    row["theta_distance_shared_F"] = theta_F
                    row["theta_distance_shared_T"] = theta_T
                    row["theta_distance_shared_diff"] = theta_T - theta_F
                    row["theta_note"] = note
                    rows.append(row)

    b1 = pd.DataFrame(rows)
    b1_path = OUTPUT_DIR / "B1_branch_contrast_utility.csv"
    b1.to_csv(b1_path, index=False)
    print(f"[B1] {len(b1)} rows → {b1_path.name}")

    # ============================================================
    # B1b: 跨 seed mean/SD/n_pos/n_neg
    # ============================================================
    metric_cols = [c for c in b1.columns if c.endswith("_diff")]
    b1b_rows = []
    for ds in b1["dataset"].unique():
        for anchor in [9, 39]:
            for step in sorted(b1["step"].unique()):
                sub = b1[(b1["dataset"] == ds) & (b1["anchor"] == anchor) & (b1["step"] == step)]
                for mc in metric_cols:
                    vals = sub[mc].dropna()
                    if len(vals) == 0:
                        b1b_rows.append({
                            "dataset": ds, "anchor": anchor, "step": step, "metric": mc,
                            "mean": np.nan, "sd": np.nan, "n_seeds": 0,
                            "n_pos": 0, "n_neg": 0, "n_zero": 0,
                        })
                        continue
                    b1b_rows.append({
                        "dataset": ds, "anchor": anchor, "step": step, "metric": mc,
                        "mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0,
                        "n_seeds": int(len(vals)),
                        "n_pos": int((vals > 0).sum()),
                        "n_neg": int((vals < 0).sum()),
                        "n_zero": int((vals == 0).sum()),
                    })
    b1b = pd.DataFrame(b1b_rows)
    b1b_path = OUTPUT_DIR / "B1b_seed_summary.csv"
    b1b.to_csv(b1b_path, index=False)
    print(f"[B1b] {len(b1b)} rows → {b1b_path.name}")

    # ============================================================
    # B2: harmful pairs contrast
    # ============================================================
    PAIRS = {
        "movielens": [(5, 2), (2, 3), (2, 4), (2, 5), (0, 0), (0, 3), (0, 5)],
        "kuairand": [(4, 1), (1, 1), (4, 4)],
    }
    b2_rows = []
    for ds in ["kuairand", "movielens"]:
        pairs = PAIRS[ds]
        for seed in sorted(h_all["seed"].unique()):
            for anchor in [9, 39]:
                for step in [1, 2, 3, 4, 5]:
                    for (sc, tc) in pairs:
                        row = {
                            "dataset": ds, "seed": seed, "anchor": anchor, "step": step,
                            "src_cell": sc, "tgt_cell": tc, "pair": f"{sc}->{tc}",
                        }
                        if step in (1, 2, 3, 4):
                            for br in ["F", "T"]:
                                sel = h_all[(h_all["dataset"] == ds) & (h_all["seed"] == seed)
                                            & (h_all["anchor_epoch"] == anchor) & (h_all["step"] == step)
                                            & (h_all["branch"] == br)
                                            & (h_all["src_cell"] == sc) & (h_all["tgt_cell"] == tc)]
                                v = float(sel["weighted_source_target_contribution"].iloc[0]) if len(sel) else np.nan
                                row[f"H_{br}"] = v
                            row["H_diff"] = row["H_T"] - row["H_F"]
                        else:
                            row["H_F"] = np.nan
                            row["H_T"] = np.nan
                            row["H_diff"] = np.nan
                        b2_rows.append(row)

    b2 = pd.DataFrame(b2_rows)
    b2_path = OUTPUT_DIR / "B2_harmful_pairs_contrast.csv"
    b2.to_csv(b2_path, index=False)
    print(f"[B2] {len(b2)} rows → {b2_path.name}")

    # B2b: 跨 seed mean/SD/符号 + 整体 Σ_{H<0} 双 branch
    b2b_rows = []
    for ds in ["kuairand", "movielens"]:
        pairs = PAIRS[ds]
        for anchor in [9, 39]:
            for (sc, tc) in pairs:
                for step in [1, 2, 3, 4, 5]:
                    sub = b2[(b2["dataset"] == ds) & (b2["anchor"] == anchor)
                             & (b2["src_cell"] == sc) & (b2["tgt_cell"] == tc)
                             & (b2["step"] == step)]
                    row = {"dataset": ds, "anchor": anchor, "src_cell": sc, "tgt_cell": tc,
                           "pair": f"{sc}->{tc}", "step": step}
                    for vname, c in [("H_F_mean", "H_F"), ("H_T_mean", "H_T"), ("H_diff_mean", "H_diff")]:
                        vals = sub[c].dropna()
                        if len(vals):
                            row[vname] = float(vals.mean())
                            row[vname.replace("_mean", "_sd")] = float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0
                        else:
                            row[vname] = np.nan
                            row[vname.replace("_mean", "_sd")] = np.nan
                    dvals = sub["H_diff"].dropna()
                    row["H_diff_n_pos"] = int((dvals > 0).sum())
                    row["H_diff_n_neg"] = int((dvals < 0).sum())
                    row["H_diff_n_zero"] = int((dvals == 0).sum())
                    b2b_rows.append(row)

    # 整体 Σ_{H<0}（所有 36 个 source→target pair 的负 H 之和；ALL module）
    for ds in ["kuairand", "movielens"]:
        for seed in sorted(h_all["seed"].unique()):
            for anchor in [9, 39]:
                for step in [1, 2, 3, 4, 5]:
                    row = {"dataset": ds, "seed": seed, "anchor": anchor, "step": step}
                    if step in (1, 2, 3, 4):
                        for br in ["F", "T"]:
                            sub = h_all[(h_all["dataset"] == ds) & (h_all["seed"] == seed)
                                        & (h_all["anchor_epoch"] == anchor) & (h_all["step"] == step)
                                        & (h_all["branch"] == br)]
                            neg_sum = float(sub[sub["weighted_source_target_contribution"] < 0]
                                           ["weighted_source_target_contribution"].sum())
                            row[f"sum_neg_H_{br}"] = neg_sum
                        row["sum_neg_H_diff"] = row["sum_neg_H_T"] - row["sum_neg_H_F"]
                    else:
                        row["sum_neg_H_F"] = np.nan
                        row["sum_neg_H_T"] = np.nan
                        row["sum_neg_H_diff"] = np.nan
                    b2b_rows.append(row)

    b2b = pd.DataFrame(b2b_rows)
    b2b_path = OUTPUT_DIR / "B2b_harmful_pairs_seed_summary.csv"
    b2b.to_csv(b2b_path, index=False)
    print(f"[B2b] {len(b2b)} rows → {b2b_path.name}")

    # ============================================================
    # B3: gradient source divergence
    # ============================================================
    b3_cols_pairs = [
        "cosine_gf_gt", "g_full_norm", "g_target_norm",
        "delta_cosine", "delta_full_norm", "delta_target_norm",
        "delta_applied_norm",
    ]
    b3_rows = []
    for ds in steps["dataset"].unique():
        for seed in sorted(steps[steps["dataset"] == ds]["seed"].unique()):
            for anchor in [9, 39]:
                for step in [1, 2, 3, 4]:
                    row = {"dataset": ds, "seed": seed, "anchor": anchor, "step": step}
                    for col in b3_cols_pairs:
                        sF = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "F") & (steps_num["step"] == step)]
                        sT = steps_num[(steps_num["dataset"] == ds) & (steps_num["seed"] == seed)
                                       & (steps_num["anchor_epoch"] == anchor)
                                       & (steps_num["branch"] == "T") & (steps_num["step"] == step)]
                        Fv = float(sF[col].iloc[0]) if len(sF) else np.nan
                        Tv = float(sT[col].iloc[0]) if len(sT) else np.nan
                        row[f"{col}_F"] = Fv
                        row[f"{col}_T"] = Tv
                        row[f"{col}_diff"] = Tv - Fv
                    b3_rows.append(row)
    b3 = pd.DataFrame(b3_rows)
    b3_path = OUTPUT_DIR / "B3_gradient_source_divergence.csv"
    b3.to_csv(b3_path, index=False)
    print(f"[B3] {len(b3)} rows → {b3_path.name}")

    # ============================================================
    # B4: early/late specificity
    # ============================================================
    # 指标 = target cells U_k diff (cell1+cell2 各自的 diff)；min_U_k diff；
    #        target_domain val_bce diff / auc diff；sum_neg_H diff
    # anchor9 vs anchor39 每 (dataset, seed)：step1..4 平均；两者差 = late−early
    # 4-seed 方向一致性：count(seeds: |diff(anchor39)| > |diff(anchor9)|)
    b4_target_diff_cols = []
    for ds in ["kuairand", "movielens"]:
        for tc in target_cells[ds]:
            b4_target_diff_cols.append((ds, tc, f"U_diff_cell{tc}"))

    b4_rows = []
    # 汇总 step1..4 平均
    def early_late_avg(df, dataset, seed, anchor, col, step_min=1, step_max=4):
        sub = df[(df["dataset"] == dataset) & (df["seed"] == seed)
                 & (df["anchor"] == anchor)
                 & (df["step"] >= step_min) & (df["step"] <= step_max)]
        vals = sub[col].dropna()
        return float(vals.mean()) if len(vals) else np.nan

    # U_k diff 指标
    u_diff_specs = [
        ("U_diff_cell1", "U_diff_cell1"),
        ("U_diff_cell4", "U_diff_cell4"),
        ("U_diff_cell2", "U_diff_cell2"),
        ("U_diff_cell5", "U_diff_cell5"),
        ("min_U_diff", "min_U_diff"),
        ("target_domain_val_bce_diff", "target_domain_val_bce_raw_diff"),
        ("target_domain_auc_diff", "target_domain_auc_diff"),
        ("sum_neg_H_diff", "sum_neg_H_diff"),
    ]
    for ds in ["kuairand", "movielens"]:
        for seed in sorted(b1["seed"].unique()):
            for label, col in u_diff_specs:
                # 仅适用该 dataset 存在的列
                if "cell1" in col and ds != "kuairand":
                    continue
                if "cell4" in col and ds != "kuairand":
                    continue
                if "cell2" in col and ds != "movielens":
                    continue
                if "cell5" in col and ds != "movielens":
                    continue
                if col == "sum_neg_H_diff":
                    # 来自 b2b
                    avg9 = early_late_avg(b2b, ds, seed, 9, col)
                    avg39 = early_late_avg(b2b, ds, seed, 39, col)
                else:
                    avg9 = early_late_avg(b1, ds, seed, 9, col)
                    avg39 = early_late_avg(b1, ds, seed, 39, col)
                b4_rows.append({
                    "dataset": ds, "seed": seed, "metric": label,
                    "anchor9_step1to4_mean": avg9,
                    "anchor39_step1to4_mean": avg39,
                    "late_minus_early": avg39 - avg9,
                })

    b4 = pd.DataFrame(b4_rows)
    # 4-seed 方向一致性：|anchor39| > |anchor9| 的 seed 数
    summary_rows = []
    for ds in b4["dataset"].unique():
        for metric in b4["metric"].unique():
            sub = b4[(b4["dataset"] == ds) & (b4["metric"] == metric)]
            n_bigger = int(((sub["anchor39_step1to4_mean"].abs()
                             > sub["anchor9_step1to4_mean"].abs())).sum())
            n_total = int(len(sub))
            summary_rows.append({
                "dataset": ds, "metric": metric,
                "n_seeds_anchor39_abs_gt_anchor9": n_bigger,
                "n_seeds_total": n_total,
            })
    summary = pd.DataFrame(summary_rows)
    # 在主表后追加 summary 段（用 sentinel row 标识），并保存为单独的两张表
    b4_path = OUTPUT_DIR / "B4_early_late_specificity.csv"
    b4.to_csv(b4_path, index=False)
    b4s_path = OUTPUT_DIR / "B4_early_late_specificity_summary.csv"
    summary.to_csv(b4s_path, index=False)
    print(f"[B4] {len(b4)} rows → {b4_path.name}; summary {len(summary)} rows → {b4s_path.name}")

    # ============================================================
    # B5: timing sequence (anchor39, step1..final)
    # ============================================================
    b5_rows = []
    for ds in ["kuairand", "movielens"]:
        for seed in sorted(b1["seed"].unique()):
            sub = b1[(b1["dataset"] == ds) & (b1["seed"] == seed) & (b1["anchor"] == 39)]
            sub = sub.sort_values("step")
            for _, r in sub.iterrows():
                step = int(r["step"])
                row = {
                    "dataset": ds, "seed": seed, "anchor": 39, "step": step,
                }
                # U_k diff（每个 target cell）
                for tc in target_cells[ds]:
                    row[f"U_diff_cell{tc}"] = r.get(f"U_diff_cell{tc}", np.nan)
                # target_domain 三项 diff
                for metric in ["val_bce_raw", "auc", "logloss"]:
                    row[f"target_domain_{metric}_diff"] = r.get(f"target_domain_{metric}_diff", np.nan)
                b5_rows.append(row)
    b5 = pd.DataFrame(b5_rows)
    b5_path = OUTPUT_DIR / "B5_timing_sequence.csv"
    b5.to_csv(b5_path, index=False)
    print(f"[B5] {len(b5)} rows → {b5_path.name}")

    # ============================================================
    # B6: kuairand control
    # ============================================================
    b6_cols_specs = [
        ("min_U_diff", b1, "min_U_diff"),
        ("target_domain_auc_diff", b1, "target_domain_auc_diff"),
        ("target_domain_val_bce_diff", b1, "target_domain_val_bce_raw_diff"),
    ]
    b6_rows = []
    for anchor in [9, 39]:
        for step in [1, 2, 3, 4, 5]:
            for label, df_src, col in b6_cols_specs:
                for ds in ["kuairand", "movielens"]:
                    sub = df_src[(df_src["dataset"] == ds) & (df_src["anchor"] == anchor)
                                 & (df_src["step"] == step)]
                    vals = sub[col].dropna()
                    row = {
                        "dataset": ds, "anchor": anchor, "step": step, "metric": label,
                        "n_seeds": int(len(vals)),
                        "mean": float(vals.mean()) if len(vals) else np.nan,
                        "sd": float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0,
                    }
                    b6_rows.append(row)
    b6 = pd.DataFrame(b6_rows)
    b6_path = OUTPUT_DIR / "B6_kuairand_control.csv"
    b6.to_csv(b6_path, index=False)
    print(f"[B6] {len(b6)} rows → {b6_path.name}")

    # ============================================================
    # 收尾：打印 B0 检查结果
    # ============================================================
    print()
    print("=" * 60)
    print("B0 identity check 汇总:")
    print(f"  geom_max_abs_diff: max = {b0['geom_max_abs_diff'].max():.3e}")
    print(f"  val_max_abs_diff : max = {b0['val_max_abs_diff'].max():.3e}")
    print(f"  steps_max_abs_diff: max = {b0['steps_max_abs_diff'].max():.3e}")
    print(f"  B0 warnings (>1e-6) count: {len(b0_warnings)}")
    if b0_warnings:
        for w in b0_warnings:
            print(f"   WARN: {w}")
    else:
        print("  B0 PASS: all step-0 branch values bit-identical (max abs diff <= 1e-6).")
    print("=" * 60)
    print()
    print("各表行数:")
    for name in [
        "B0_step0_identity_check.csv",
        "B1_branch_contrast_utility.csv",
        "B1b_seed_summary.csv",
        "B2_harmful_pairs_contrast.csv",
        "B2b_harmful_pairs_seed_summary.csv",
        "B3_gradient_source_divergence.csv",
        "B4_early_late_specificity.csv",
        "B4_early_late_specificity_summary.csv",
        "B5_timing_sequence.csv",
        "B6_kuairand_control.csv",
    ]:
        p = OUTPUT_DIR / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            print(f"  {name:48s}  {n:>6d} rows")
        else:
            print(f"  {name:48s}  MISSING")


if __name__ == "__main__":
    main()
