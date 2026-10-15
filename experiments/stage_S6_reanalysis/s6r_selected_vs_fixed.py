"""s6r_selected_vs_fixed.py

Generate tables 9 (selected vs fixed) and 10 (peak width) for one dataset.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from s6r_common import (
    CHECKPOINT_TO_EPOCH,
    DatasetSpec,
    E_COMMON,
    SEEDS,
    TAU_PRAC,
    cell_label,
    delta_pair,
    load_raw,
    load_val_epoch,
    non_target_cells,
    parse_dataset,
    smooth_moving,
    table_path,
    target_cells,
    write_csv,
)


def _select_macro_either(
    spec: DatasetSpec, raw: pd.DataFrame, arm: str, seed: int, layer: str
) -> float:
    """Return per-cell mean AUC at the selected checkpoint (test) or val AUC at
    the selected_epoch (val)."""
    sub = raw[(raw["arm"] == arm) & (raw["seed"] == int(seed)) & (raw["checkpoint"] == "selected")]
    if layer == "test":
        return float(sub["auc"].mean())
    # val
    sub_v = load_val_epoch(spec, arm, int(seed))
    sel_epoch = int(sub["stage_epoch"].iloc[0])
    return float(sub_v[sub_v["epoch"] == sel_epoch]["validation_macro_auc"].iloc[0])


def _macro_at_epoch(
    spec: DatasetSpec, raw: pd.DataFrame, arm: str, seed: int, epoch: int, layer: str
) -> float:
    sub = raw[
        (raw["arm"] == arm)
        & (raw["seed"] == int(seed))
        & (raw["epoch"] == int(epoch))
    ]
    if layer == "test":
        return float(sub["auc"].mean())
    sub_v = load_val_epoch(spec, arm, int(seed))
    return float(sub_v[sub_v["epoch"] == int(epoch)]["validation_macro_auc"].iloc[0])


def build_table9(spec: DatasetSpec, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = target_cells(spec)

    for seed in SEEDS:
        sub = raw[(raw["seed"] == int(seed)) & (raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys())))]
        sel_F = int(raw[(raw["arm"] == "F") & (raw["seed"] == int(seed)) & (raw["checkpoint"] == "selected")]["stage_epoch"].iloc[0])
        sel_DH = int(raw[(raw["arm"] == "DH") & (raw["seed"] == int(seed)) & (raw["checkpoint"] == "selected")]["stage_epoch"].iloc[0])
        # ΔSel macro test = DH sel macro - F sel macro (over 6 cells)
        sel_F_macro = _select_macro_either(spec, raw, "F", seed, "test")
        sel_DH_macro = _select_macro_either(spec, raw, "DH", seed, "test")
        # val versions
        sel_F_macro_val = _select_macro_either(spec, raw, "F", seed, "val")
        sel_DH_macro_val = _select_macro_either(spec, raw, "DH", seed, "val")

        # Per-cell ΔSel at test
        per_cell = {}
        for (d, t) in [(d, t) for d in (0, 1, 2) for t in (0, 1)]:
            f_val = float(raw[(raw["arm"] == "F") & (raw["seed"] == int(seed))
                              & (raw["checkpoint"] == "selected")
                              & (raw["domain"] == d) & (raw["task"] == t)]["auc"].iloc[0])
            dh_val = float(raw[(raw["arm"] == "DH") & (raw["seed"] == int(seed))
                              & (raw["checkpoint"] == "selected")
                              & (raw["domain"] == d) & (raw["task"] == t)]["auc"].iloc[0])
            per_cell[cell_label(d, t)] = dh_val - f_val

        # Timing decomposition on val layer (per §2)
        m_DH_sel = sel_DH_macro_val
        m_DH_sel_F = _macro_at_epoch(spec, raw, "DH", seed, sel_F, "val")
        m_F_sel_F = sel_F_macro_val
        # Anchor at sel_DH: symmetric
        m_F_sel_DH = _macro_at_epoch(spec, raw, "F", seed, sel_DH, "val")

        g_time_val = m_DH_sel - m_DH_sel_F  # timing in DH
        g_epoch_val = m_DH_sel_F - m_F_sel_F  # epoch in DH vs F
        # Symmetric decomposition anchored at sel_DH:
        # epoch effect measured at sel_DH, timing effect measured within F
        g_epoch_sym_val = m_DH_sel - m_F_sel_DH
        g_time_sym_val = m_F_sel_DH - m_F_sel_F

        # Macro gain trajectory test (DH-F) across E_common — per seed mean over 6 cells
        macro_gains = []
        nontarget_gains = []
        for e in E_COMMON:
            e_sub = sub[sub["epoch"] == e]
            piv = e_sub.pivot_table(index=["arm"], values="auc", aggfunc="mean")
            mg = float(piv.loc["DH"].mean() - piv.loc["F"].mean()) if "DH" in piv.index and "F" in piv.index else float("nan")
            macro_gains.append(mg)
            nt_piv = e_sub[~e_sub["domain"].astype(int).isin([t[0] for t in target])]
            nt_piv = nt_piv.pivot_table(index=["arm"], values="auc", aggfunc="mean")
            ntg = float(nt_piv.loc["DH"].mean() - nt_piv.loc["F"].mean()) if "DH" in nt_piv.index and "F" in nt_piv.index else float("nan")
            nontarget_gains.append(ntg)

        # selected-macro Δ test:
        delta_sel_macro_test = sel_DH_macro - sel_F_macro
        # non-target Δ at selected (raw CSV):
        sub_sel = raw[(raw["arm"].isin(["F", "DH"])) & (raw["seed"] == int(seed))
                      & (raw["checkpoint"] == "selected")]
        piv = sub_sel.pivot_table(index=["domain", "task"], columns="arm", values="auc", aggfunc="first").reset_index()
        piv["delta"] = piv["DH"] - piv["F"]
        nt_piv = piv[~piv["domain"].astype(int).isin([t[0] for t in target])]
        sel_macro_gain_nontarget_test = float(nt_piv["delta"].mean())

        # Peak-derived metrics
        max_macro_common = float(np.nanmax(macro_gains))
        median_macro_common = float(np.nanmedian(macro_gains))
        ratio_sel_over_max = delta_sel_macro_test / max_macro_common if max_macro_common > 0 else float("nan")
        ratio_sel_over_median = delta_sel_macro_test / median_macro_common if median_macro_common > 0 else float("nan")

        row = {
            "dataset": spec.name, "seed": int(seed),
            "sel_epoch_F": sel_F, "sel_epoch_DH": sel_DH,
            "delta_sel_macro_test": delta_sel_macro_test,
            "delta_sel_macro_val": sel_DH_macro_val - sel_F_macro_val,
            "G_time_val": g_time_val,
            "G_epoch_val": g_epoch_val,
            "G_time_sym_val": g_time_sym_val,
            "G_epoch_sym_val": g_epoch_sym_val,
            "max_macro_gain_common_test": max_macro_common,
            "median_macro_gain_common_test": median_macro_common,
            "ratio_sel_over_max": ratio_sel_over_max,
            "ratio_sel_over_median": ratio_sel_over_median,
            "sel_macro_gain_nontarget_test": sel_macro_gain_nontarget_test,
        }
        for (d, t) in [(d, t) for d in (0, 1, 2) for t in (0, 1)]:
            row[f"delta_sel_{cell_label(d, t)}_test"] = per_cell[cell_label(d, t)]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed"]).reset_index(drop=True)


def build_table10(spec: DatasetSpec, raw: pd.DataFrame) -> pd.DataFrame:
    """Peak width analysis per (seed, layer, metric).

    For test layer: sequence is the 5-point E_common series.
    For val layer: 0..49 smoothed with w=10.
    """
    rows = []
    nt = non_target_cells(spec)

    # Pre-compute val layer smoothed sequences per seed & metric
    val_rows: List[pd.DataFrame] = []
    for arm in ("F", "DH"):
        for seed in SEEDS:
            v = load_val_epoch(spec, arm, int(seed))
            val_rows.append(v)
    val_df = pd.concat(val_rows, ignore_index=True)
    d_val = delta_pair(val_df, "DH", "F", "validation_auc")

    for layer in ("test", "val"):
        if layer == "test":
            keep = raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy()
            d = delta_pair(keep, "DH", "F", "auc")
            epochs = list(E_COMMON)
        else:
            d = d_val
            epochs = list(range(0, 50))

        for metric in ("nontarget_gain", "macro_gain"):
            for seed in SEEDS:
                sub = d[d["seed"] == int(seed)]
                if metric == "nontarget_gain":
                    sub = sub[sub["domain"].astype(int).isin([t[0] for t in nt])]
                if layer == "test":
                    seq = [float(sub[sub["epoch"] == e]["delta"].mean()) for e in epochs]
                else:
                    base_seq = [float(sub[sub["epoch"] == e]["delta"].mean()) for e in epochs]
                    seq = smooth_moving(base_seq, w=10).tolist()
                arr = np.asarray(seq, dtype=float)
                peak_val = float(arr.max())
                peak_idx = int(arr.argmax())
                peak_epoch = epochs[peak_idx]
                # width_prac: contiguous indices ≥ TAU_PRAC containing peak
                ge = arr >= TAU_PRAC
                # expand around peak
                lo_i = peak_idx
                hi_i = peak_idx
                while lo_i - 1 >= 0 and ge[lo_i - 1]:
                    lo_i -= 1
                while hi_i + 1 < len(ge) and ge[hi_i + 1]:
                    hi_i += 1
                width_prac = hi_i - lo_i + 1
                # width_half_peak: contiguous indices ≥ peak_val/2 containing peak
                half = peak_val / 2.0
                ge2 = arr >= half
                lo_i2 = peak_idx
                hi_i2 = peak_idx
                while lo_i2 - 1 >= 0 and ge2[lo_i2 - 1]:
                    lo_i2 -= 1
                while hi_i2 + 1 < len(ge2) and ge2[hi_i2 + 1]:
                    hi_i2 += 1
                width_half_peak = hi_i2 - lo_i2 + 1
                typical_gain = float(np.nanmedian(arr))
                if layer == "test":
                    is_narrow = bool(width_prac <= 1 and peak_val >= 2 * typical_gain if typical_gain > 0 else width_prac <= 1 and peak_val >= 2 * max(typical_gain, 0.001))
                else:
                    is_narrow = bool(width_prac <= 5 and peak_val >= 2 * typical_gain if typical_gain > 0 else width_prac <= 5 and peak_val >= 2 * max(typical_gain, 0.001))
                rows.append({
                    "dataset": spec.name, "layer": layer, "metric": metric,
                    "seed": int(seed),
                    "peak_epoch": peak_epoch, "peak_val": peak_val,
                    "width_prac": int(width_prac),
                    "width_half_peak": int(width_half_peak),
                    "typical_gain": typical_gain,
                    "is_narrow_peak": is_narrow,
                })
    return pd.DataFrame(rows).sort_values(["layer", "metric", "seed"]).reset_index(drop=True)


def main() -> None:
    spec = parse_dataset()
    raw = load_raw(spec)
    t9 = build_table9(spec, raw)
    t10 = build_table10(spec, raw)
    write_csv(t9, table_path(spec, "s6r_selected_vs_fixed"))
    write_csv(t10, table_path(spec, "s6r_peak_width"))
    print(f"{spec.name}: tables 9/10 written ({len(t9)}+{len(t10)})")


if __name__ == "__main__":
    main()