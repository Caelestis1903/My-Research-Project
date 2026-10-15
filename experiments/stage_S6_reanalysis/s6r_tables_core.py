"""s6r_tables_core.py

Generate tables 1, 1b, 2, 2b, 3, 3b, 4, 4b, 5, 6 for one dataset.

Usage:
    venv/Scripts/python.exe experiments/stage_S6_reanalysis/s6r_tables_core.py \
        --dataset {kuairand|movielens}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from s6r_common import (
    CHECKPOINT_TO_EPOCH,
    DatasetSpec,
    E_COMMON,
    SEEDS,
    TAU_PRAC,
    TAU_STRONG,
    cell_label,
    delta_pair,
    load_raw,
    load_val_epoch,
    non_target_domains,
    parse_dataset,
    regime_label,
    table_path,
    target_cells,
    write_csv,
)


def _summary_frame(
    raw_with_seeds: pd.DataFrame,
    key_cols: Sequence[str],
    val_col: str,
    prefix: str,
    mean_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build summary row with mean/sd/n_pos/per-seed/flags.

    mean_col_name: override the column name for the summary mean (defaults
    to "mean"). The cell-trajectory summary tables use mean_delta_auc per §7.
    """
    g = raw_with_seeds.groupby(list(key_cols))
    rows = []
    for keys, sub in g:
        if isinstance(keys, tuple):
            d = dict(zip(key_cols, keys))
        else:
            d = {key_cols[0]: keys}
        per_seed = sub.groupby("seed")[val_col].mean()
        mean = float(per_seed.mean())
        sd = float(per_seed.std(ddof=1)) if per_seed.size >= 2 else float("nan")
        n_pos = int((per_seed > 0).sum())
        row = {**d, (mean_col_name or "mean"): mean, "sd": sd, "n_pos": n_pos, "n": int(per_seed.size)}
        for s in SEEDS:
            row[f"{prefix}_seed{int(s)}"] = float(per_seed.get(int(s), float("nan")))
        row["ge_practical"] = bool(mean >= TAU_PRAC)
        row["ge_strong"] = bool(mean >= TAU_STRONG)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 1 & 2 (cell trajectory, test & val).
# ---------------------------------------------------------------------------
def build_cell_trajectory(
    spec: DatasetSpec, raw: pd.DataFrame, layer: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """layer in {"test", "val"}."""
    if layer == "test":
        keep = raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy()
        val_col = "auc"
        loss_col = "logloss"
    else:  # val
        rows = []
        for arm in ("F", "DH"):
            for seed in SEEDS:
                v = load_val_epoch(spec, arm, int(seed))
                rows.append(v)
        keep = pd.concat(rows, ignore_index=True)
        val_col = "validation_auc"
        loss_col = "validation_logloss"

    delt = delta_pair(keep, "DH", "F", val_col).rename(columns={"delta": "delta_auc"})
    delt_loss = delta_pair(keep, "DH", "F", loss_col).rename(columns={"delta": "delta_logloss"})
    delt = delt.merge(
        delt_loss[["epoch", "seed", "domain", "task", "delta_logloss"]],
        on=["epoch", "seed", "domain", "task"],
        how="left",
    )
    delt["cell_label"] = [cell_label(int(d), int(t)) for d, t in zip(delt["domain"], delt["task"])]
    delt["is_target"] = delt["domain"].astype(int) == spec.target_domain
    if layer == "test":
        keep_epochs = sorted(set(E_COMMON))
    else:
        keep_epochs = sorted(delt["epoch"].unique().tolist())
    delt = delt[delt["epoch"].isin(keep_epochs)].copy()
    long = delt[
        ["dataset", "seed", "epoch", "cell_label", "domain", "task", "is_target", "delta_auc", "delta_logloss"]
    ].rename(columns={"cell_label": "cell"}).copy()
    # sort
    long = long.sort_values(["seed", "epoch", "cell"]).reset_index(drop=True)
    summary = _summary_frame(
        long,
        key_cols=["epoch", "cell"],
        val_col="delta_auc",
        prefix="delta",
        mean_col_name="mean_delta_auc",
    ).sort_values(["epoch", "cell"]).reset_index(drop=True)
    return long, summary


# ---------------------------------------------------------------------------
# Table 3 & 4 (core metrics).
# ---------------------------------------------------------------------------
def _per_seed_aggregates(
    delt: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    """Compute per-seed core aggregates (TargetCost, NonTargetGain, MacroGain, per-non-target-domain)."""
    target = target_cells(spec)
    nt_domains = non_target_domains(spec)

    rows = []
    for (epoch, seed), sub in delt.groupby(["epoch", "seed"]):
        d: Dict[str, float] = {"dataset": spec.name, "epoch": int(epoch), "seed": int(seed)}
        target_mask = sub["domain"].astype(int).isin([t[0] for t in target])
        nontarget_mask = ~target_mask
        d["target_cost"] = float(sub.loc[target_mask, "delta"].mean())
        d["nontarget_gain"] = float(sub.loc[nontarget_mask, "delta"].mean())
        d["macro_gain"] = float(sub["delta"].mean())
        for nd in nt_domains:
            mask = sub["domain"].astype(int) == int(nd)
            d[f"nontarget_domain_gain_d{int(nd)}"] = float(sub.loc[mask, "delta"].mean())
        rows.append(d)
    return pd.DataFrame(rows)


def build_core_metrics(
    spec: DatasetSpec, raw: pd.DataFrame, layer: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if layer == "test":
        keep = raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy()
        val_col = "auc"
    else:
        rows = []
        for arm in ("F", "DH"):
            for seed in SEEDS:
                v = load_val_epoch(spec, arm, int(seed))
                rows.append(v)
        keep = pd.concat(rows, ignore_index=True)
        val_col = "validation_auc"
    delt = delta_pair(keep, "DH", "F", val_col)
    per_seed = _per_seed_aggregates(delt, spec)
    # Order columns: dataset, seed, epoch, target_cost, nontarget_gain, macro_gain, d{a}_gain, d{b}_gain
    nt_domains = non_target_domains(spec)
    ordered = ["dataset", "seed", "epoch", "target_cost", "nontarget_gain", "macro_gain"] + [
        f"nontarget_domain_gain_d{int(d)}" for d in nt_domains
    ]
    per_seed = per_seed[ordered].sort_values(["epoch", "seed"]).reset_index(drop=True)
    # Build summary with each metric
    summary_rows = []
    for epoch in sorted(per_seed["epoch"].unique()):
        sub = per_seed[per_seed["epoch"] == epoch]
        row = {"dataset": spec.name, "epoch": int(epoch)}
        for metric in [
            "target_cost", "nontarget_gain", "macro_gain",
        ] + [f"nontarget_domain_gain_d{int(d)}" for d in nt_domains]:
            per_seed_metric = sub.groupby("seed")[metric].mean()
            row[f"{metric}_mean"] = float(per_seed_metric.mean())
            row[f"{metric}_sd"] = float(per_seed_metric.std(ddof=1)) if per_seed_metric.size >= 2 else float("nan")
            row[f"{metric}_n_pos"] = int((per_seed_metric > 0).sum())
            for s in SEEDS:
                row[f"{metric}_seed{int(s)}"] = float(per_seed_metric.get(int(s), float("nan")))
            row[f"{metric}_ge_practical"] = bool(row[f"{metric}_mean"] >= TAU_PRAC)
            row[f"{metric}_ge_strong"] = bool(row[f"{metric}_mean"] >= TAU_STRONG)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    return per_seed, summary_df


# ---------------------------------------------------------------------------
# Table 5 (domain decomposition).
# ---------------------------------------------------------------------------
def build_domain_decomposition(
    spec: DatasetSpec, raw: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    for layer, src_df, val_col in [
        ("test", raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy(), "auc"),
    ]:
        d_pivot = delta_pair(src_df, "DH", "F", val_col)
        _append_domain_decomp_rows(rows, spec, d_pivot, layer)

    # val: 0..49
    val_rows = []
    for arm in ("F", "DH"):
        for seed in SEEDS:
            v = load_val_epoch(spec, arm, int(seed))
            val_rows.append(v)
    val_df = pd.concat(val_rows, ignore_index=True)
    d_val = delta_pair(val_df, "DH", "F", "validation_auc")
    _append_domain_decomp_rows(rows, spec, d_val, "val")

    out = pd.DataFrame(rows)
    return out


def _append_domain_decomp_rows(
    rows: List[Dict],
    spec: DatasetSpec,
    d_pivot: pd.DataFrame,
    layer: str,
) -> None:
    nt_domains = non_target_domains(spec)
    for (epoch, seed), grp in d_pivot.groupby(["epoch", "seed"]):
        row: Dict[str, object] = {
            "dataset": spec.name, "layer": layer, "seed": int(seed), "epoch": int(epoch),
        }
        for d in nt_domains:
            mask_d = grp["domain"].astype(int) == int(d)
            row[f"d{int(d)}_gain"] = float(grp.loc[mask_d, "delta"].mean())
            t0 = grp[(grp["domain"].astype(int) == int(d)) & (grp["task"] == 0)]
            t1 = grp[(grp["domain"].astype(int) == int(d)) & (grp["task"] == 1)]
            row[f"d{int(d)}_t0"] = float(t0["delta"].iloc[0]) if len(t0) else float("nan")
            row[f"d{int(d)}_t1"] = float(t1["delta"].iloc[0]) if len(t1) else float("nan")
        # target cells
        target_d = spec.target_domain
        mask_t = grp["domain"].astype(int) == int(target_d)
        row[f"d{int(target_d)}_gain"] = float(grp.loc[mask_t, "delta"].mean())
        t0 = grp[(grp["domain"].astype(int) == int(target_d)) & (grp["task"] == 0)]
        t1 = grp[(grp["domain"].astype(int) == int(target_d)) & (grp["task"] == 1)]
        row[f"d{int(target_d)}_t0"] = float(t0["delta"].iloc[0]) if len(t0) else float("nan")
        row[f"d{int(target_d)}_t1"] = float(t1["delta"].iloc[0]) if len(t1) else float("nan")
        rows.append(row)


# ---------------------------------------------------------------------------
# Table 6 (tradeoff points).
# ---------------------------------------------------------------------------
def build_tradeoff_points(
    spec: DatasetSpec, raw: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    test_df = raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy()
    d_test = delta_pair(test_df, "DH", "F", "auc")
    _append_tradeoff_rows(rows, spec, d_test, "test")

    val_rows = []
    for arm in ("F", "DH"):
        for seed in SEEDS:
            v = load_val_epoch(spec, arm, int(seed))
            val_rows.append(v)
    val_df = pd.concat(val_rows, ignore_index=True)
    d_val = delta_pair(val_df, "DH", "F", "validation_auc")
    _append_tradeoff_rows(rows, spec, d_val, "val")
    return pd.DataFrame(rows).sort_values(["layer", "seed", "epoch"]).reset_index(drop=True)


def _append_tradeoff_rows(
    rows: List[Dict],
    spec: DatasetSpec,
    d_pivot: pd.DataFrame,
    layer: str,
) -> None:
    target = target_cells(spec)
    for (epoch, seed), grp in d_pivot.groupby(["epoch", "seed"]):
        mask_t = grp["domain"].astype(int).isin([t[0] for t in target])
        tc = float(grp.loc[mask_t, "delta"].mean())
        nt = float(grp.loc[~mask_t, "delta"].mean())
        rows.append({
            "dataset": spec.name, "layer": layer, "seed": int(seed), "epoch": int(epoch),
            "target_cost": tc, "nontarget_gain": nt, "regime": regime_label(tc, nt),
        })


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    spec = parse_dataset()
    raw = load_raw(spec)
    # Tables 1/1b
    t1_long, t1_sum = build_cell_trajectory(spec, raw, "test")
    write_csv(t1_long, table_path(spec, "s6r_cell_trajectory_test"))
    write_csv(t1_sum, table_path(spec, "s6r_cell_trajectory_test_summary"))
    # Tables 2/2b
    t2_long, t2_sum = build_cell_trajectory(spec, raw, "val")
    write_csv(t2_long, table_path(spec, "s6r_cell_trajectory_val"))
    write_csv(t2_sum, table_path(spec, "s6r_cell_trajectory_val_summary"))
    # Tables 3/3b
    t3_long, t3_sum = build_core_metrics(spec, raw, "test")
    write_csv(t3_long, table_path(spec, "s6r_core_metrics_test"))
    write_csv(t3_sum, table_path(spec, "s6r_core_metrics_test_summary"))
    # Tables 4/4b
    t4_long, t4_sum = build_core_metrics(spec, raw, "val")
    write_csv(t4_long, table_path(spec, "s6r_core_metrics_val"))
    write_csv(t4_sum, table_path(spec, "s6r_core_metrics_val_summary"))
    # Table 5
    t5 = build_domain_decomposition(spec, raw)
    write_csv(t5, table_path(spec, "s6r_domain_decomposition"))
    # Table 6
    t6 = build_tradeoff_points(spec, raw)
    write_csv(t6, table_path(spec, "s6r_tradeoff_points"))
    print(f"{spec.name}: tables 1-6 written ({len(t1_long)}+{len(t1_sum)}+{len(t2_long)}+{len(t2_sum)}+{len(t3_long)}+{len(t3_sum)}+{len(t4_long)}+{len(t4_sum)}+{len(t5)}+{len(t6)})")


if __name__ == "__main__":
    main()