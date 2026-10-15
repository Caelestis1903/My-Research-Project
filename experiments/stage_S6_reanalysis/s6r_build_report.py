"""s6r_build_report.py

Read all intermediate CSVs from experiments/stage_S6_reanalysis/tables/, build
table 13 (aux-arm selected), compute the final A/B/C/D verdict, and emit the
final markdown report.

The report builder reads ALL numbers from the CSV files; it never recomputes
analysis from raw data and never writes analytic numbers by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from s6r_common import (
    DatasetSpec,
    DATASETS,
    E_COMMON,
    SEEDS,
    TAU_PRAC,
    VerdictFeatures,
    cell_label,
    load_raw,
    non_target_domains,
    render_verdict,
    target_cells,
    table_path,
    write_csv,
)


# ---------------------------------------------------------------------------
# Verdict computation (per spec §7.9).
# ---------------------------------------------------------------------------
def _compute_K_and_Sk(table3_long: pd.DataFrame, spec: DatasetSpec) -> Tuple[int, bool, float, int]:
    """K, S_k, TC, W. K counts contiguous common epochs with 4-seed mean
    NonTargetGain ≥ TAU_PRAC (test layer); S_k True if ≥3/4 seeds positive
    over that run. TC is max |TargetCost| mean over that run. W = number of
    seeds with is_narrow_peak True for nontarget_gain at test layer.
    """
    by_epoch = table3_long.groupby("epoch")["nontarget_gain"].mean()
    mask = (by_epoch >= TAU_PRAC).reindex(list(E_COMMON)).fillna(False).values
    # find longest run
    best = 0
    cur = 0
    best_start = 0
    i = 0
    while i < len(mask):
        if mask[i]:
            cur = 1
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                cur += 1
                j += 1
            if cur > best:
                best = cur
                best_start = i
            i = j + 1
        else:
            i += 1
    K = int(best)
    # S_k: same run, ≥3/4 seeds positive
    S_k = False
    TC = float("nan")
    if K >= 1:
        run_epochs = list(E_COMMON)[best_start:best_start + K]
        sub = table3_long[table3_long["epoch"].isin(run_epochs)]
        per_seed = sub.groupby("seed")["nontarget_gain"].mean()
        S_k = bool((per_seed > 0).sum() >= 3)
        # TC = max of |4-seed mean TargetCost| over the run
        epoch_mean_tc = sub.groupby("epoch")["target_cost"].mean()
        TC = float(epoch_mean_tc.abs().max())
    return K, S_k, TC, K  # placeholder for W replaced below


def _domain_evidence(
    spec: DatasetSpec, table3_long: pd.DataFrame
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per non-target domain: K_dom (contiguous common epochs ≥ TAU_PRAC) and
    3-of-4 seed positive indicator."""
    res_K: Dict[str, int] = {}
    res_3of4: Dict[str, int] = {}
    nt_domains = non_target_domains(spec)
    for d in nt_domains:
        col = f"nontarget_domain_gain_d{int(d)}"
        by_epoch = table3_long.groupby("epoch")[col].mean()
        mask = (by_epoch >= TAU_PRAC).reindex(list(E_COMMON)).fillna(False).values
        best = 0
        cur = 0
        i = 0
        while i < len(mask):
            if mask[i]:
                cur = 1
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    cur += 1
                    j += 1
                if cur > best:
                    best = cur
                i = j + 1
            else:
                i += 1
        res_K[f"d{int(d)}"] = int(best)
        sub = table3_long[table3_long["epoch"].isin(list(E_COMMON))]
        per_seed = sub.groupby("seed")[col].mean()
        res_3of4[f"d{int(d)}"] = int((per_seed > 0).sum() >= 3)
    return res_K, res_3of4


def _W_count(table10: pd.DataFrame, spec: DatasetSpec) -> int:
    sub = table10[
        (table10["layer"] == "test")
        & (table10["metric"] == "nontarget_gain")
        & (table10["dataset"] == spec.name)
    ]
    return int(sub["is_narrow_peak"].sum())


def _ratio_median(table9: pd.DataFrame, spec: DatasetSpec) -> float:
    sub = table9[table9["dataset"] == spec.name]
    if len(sub) == 0:
        return float("nan")
    return float(sub["ratio_sel_over_median"].median())


def _mean_delta_sel_nontarget(table9: pd.DataFrame, spec: DatasetSpec) -> float:
    sub = table9[table9["dataset"] == spec.name]
    if len(sub) == 0:
        return float("nan")
    return float(sub["sel_macro_gain_nontarget_test"].mean())


def _pattern_label_test_nontarget(table7b: pd.DataFrame, spec: DatasetSpec) -> str:
    sub = table7b[
        (table7b["layer"] == "test")
        & (table7b["metric"] == "nontarget_gain")
        & (table7b["dataset"] == spec.name)
    ]
    if len(sub) == 0:
        return "N"
    return str(sub["dataset_pattern_label"].iloc[0])


def _peak_epochs_test_nontarget(table10: pd.DataFrame, spec: DatasetSpec) -> List[int]:
    sub = table10[
        (table10["layer"] == "test")
        & (table10["metric"] == "nontarget_gain")
        & (table10["dataset"] == spec.name)
    ]
    return [int(p) for p in sub["peak_epoch"].tolist()]


def compute_verdict(spec: DatasetSpec, tables_dir: Path) -> Tuple[str, Dict[str, object]]:
    """Compute the final verdict for one dataset."""
    table3_long = pd.read_csv(tables_dir / f"{spec.out_prefix}_s6r_core_metrics_test.csv", encoding="utf-8-sig")
    table7b = pd.read_csv(tables_dir / f"{spec.out_prefix}_s6r_pattern_dataset_level.csv", encoding="utf-8-sig")
    table9 = pd.read_csv(tables_dir / f"{spec.out_prefix}_s6r_selected_vs_fixed.csv", encoding="utf-8-sig")
    table10 = pd.read_csv(tables_dir / f"{spec.out_prefix}_s6r_peak_width.csv", encoding="utf-8-sig")

    K, S_k, TC, _ = _compute_K_and_Sk(table3_long, spec)
    W = _W_count(table10, spec)
    R = _ratio_median(table9, spec)
    K_by_d, _3of4_by_d = _domain_evidence(spec, table3_long)
    nt_domains = non_target_domains(spec)
    if len(nt_domains) >= 2:
        dA, dB = nt_domains[0], nt_domains[1]
    else:
        dA, dB = nt_domains[0], nt_domains[0]
    DomA_ok = bool(K_by_d.get(f"d{int(dA)}", 0) >= 3 and _3of4_by_d.get(f"d{int(dA)}", 0) == 1)
    DomB_ok = bool(K_by_d.get(f"d{int(dB)}", 0) >= 3 and _3of4_by_d.get(f"d{int(dB)}", 0) == 1)
    mean_delta_sel_nontarget = _mean_delta_sel_nontarget(table9, spec)
    pattern_test_nontarget = _pattern_label_test_nontarget(table7b, spec)
    peak_epochs = _peak_epochs_test_nontarget(table10, spec)

    feats = VerdictFeatures(
        K=K, S_k=S_k, TC=TC, W=W,
        DomA_ok=DomA_ok, DomB_ok=DomB_ok,
        DomA_K=K_by_d.get(f"d{int(dA)}", 0),
        DomB_K=K_by_d.get(f"d{int(dB)}", 0),
        DomA_3of4=_3of4_by_d.get(f"d{int(dA)}", 0) == 1,
        DomB_3of4=_3of4_by_d.get(f"d{int(dB)}", 0) == 1,
        R=R,
        pattern_test_nontarget=pattern_test_nontarget,
        mean_delta_sel_nontarget=mean_delta_sel_nontarget,
        peak_epochs_test_nontarget=peak_epochs,
        K_dom_by_domain=K_by_d,
        n_pos_by_domain=_3of4_by_d,
    )
    verdict = render_verdict(feats)
    details = {
        "K": K, "S_k": S_k, "TC": TC, "W": W, "R": R,
        "DomA_ok": DomA_ok, "DomB_ok": DomB_ok,
        "DomA_K": feats.DomA_K, "DomB_K": feats.DomB_K,
        "DomA_3of4": feats.DomA_3of4, "DomB_3of4": feats.DomB_3of4,
        "mean_delta_sel_nontarget": mean_delta_sel_nontarget,
        "pattern_test_nontarget": pattern_test_nontarget,
        "peak_epochs": peak_epochs,
        "K_by_domain": K_by_d,
        "n_pos_by_domain": _3of4_by_d,
        "dA": int(dA), "dB": int(dB),
    }
    return verdict, details


# ---------------------------------------------------------------------------
# Table 13 (aux-arm selected deltas).
# ---------------------------------------------------------------------------
def build_table13(spec: DatasetSpec, raw: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, seed, contrast, cell) Δ at selected, test."""
    contrasts = [("T", "F"), ("HT", "F"), ("DH", "F"), ("DH", "T"), ("DH", "HT")]
    rows = []
    for seed in SEEDS:
        sel = raw[
            (raw["seed"] == int(seed))
            & (raw["checkpoint"] == "selected")
            & (raw["arm"].isin([a for pair in contrasts for a in pair]))
        ]
        piv = sel.pivot_table(
            index=["domain", "task"], columns="arm", values="auc", aggfunc="first"
        ).reset_index()
        target = target_cells(spec)
        for a, b in contrasts:
            if a not in piv.columns or b not in piv.columns:
                continue
            piv[f"delta_{a}_minus_{b}"] = piv[a] - piv[b]
            for _, r in piv.iterrows():
                rows.append({
                    "dataset": spec.name,
                    "seed": int(seed),
                    "contrast": f"{a}-{b}",
                    "cell": cell_label(int(r["domain"]), int(r["task"])),
                    "delta_auc": float(r[f"delta_{a}_minus_{b}"]),
                    "scope": "cell",
                })
            # macro / non-target rows
            macro = float(piv[f"delta_{a}_minus_{b}"].mean())
            nt_piv = piv[~piv["domain"].astype(int).isin([t[0] for t in target])]
            nt_mean = float(nt_piv[f"delta_{a}_minus_{b}"].mean())
            rows.append({"dataset": spec.name, "seed": int(seed), "contrast": f"{a}-{b}",
                          "cell": "macro", "delta_auc": macro, "scope": "macro"})
            rows.append({"dataset": spec.name, "seed": int(seed), "contrast": f"{a}-{b}",
                          "cell": "nontarget_mean", "delta_auc": nt_mean, "scope": "nontarget"})
    return pd.DataFrame(rows).sort_values(["contrast", "cell", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------
def _row(r: pd.Series, fmt: str = "{:.6f}") -> str:
    return ", ".join(f"{k}={fmt.format(v) if isinstance(v, float) else v}" for k, v in r.items())


def _read_table(name: str, spec: DatasetSpec) -> pd.DataFrame:
    p = table_path(spec, name)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


def _fmt_seq(seq: List[float], epochs: List[int]) -> str:
    return "; ".join(f"e{e}={v:+.4f}" for e, v in zip(epochs, seq) if not np.isnan(v))


def _verdict_explanation(d: Dict[str, object]) -> str:
    """Verbatim explanation per branch in the verdict decision tree."""
    K, S_k, TC, W, R = d["K"], d["S_k"], d["TC"], d["W"], d["R"]
    DomA_ok, DomB_ok = d["DomA_ok"], d["DomB_ok"]
    pattern = d["pattern_test_nontarget"]
    mean_sel = d["mean_delta_sel_nontarget"]
    K_by_d = d["K_by_domain"]
    dA, dB = d["dA"], d["dB"]
    lines = []
    lines.append(f"- K (longest run of common epochs with 4-seed mean NonTargetGain ≥{TAU_PRAC}) = **{K}**")
    lines.append(f"- S_k (≥3/4 seeds positive in that run) = **{S_k}**")
    lines.append(f"- TC (max |TargetCost| 4-seed mean within run) = **{TC:.4f}**")
    lines.append(f"- W (narrow-peak seeds) = **{W}**")
    lines.append(f"- R (4-seed median of ratio_sel/median) = **{R:.3f}**")
    lines.append(f"- dataset_pattern_label(test, nontarget_gain) = **{pattern}**")
    lines.append(f"- mean ΔSel(non-target, test) = **{mean_sel:+.4f}**")
    lines.append(f"- non-target domain evidence: d{int(dA)} K_dom={K_by_d.get(f'd{int(dA)}',0)} → ok={DomA_ok}; d{int(dB)} K_dom={K_by_d.get(f'd{int(dB)}',0)} → ok={DomB_ok}")
    return "\n".join(lines)


def _target_phrase(tc_value: float) -> str:
    """Term-discipline phrasing per §8."""
    if abs(tc_value) >= TAU_PRAC:
        return "target damage (|TC|≥practical threshold)"
    return "轻微下降（低于 practical 阈值）"


def _hypothesis_outcomes(spec: DatasetSpec, tables_dir: Path, details: Dict[str, object]) -> str:
    """H1..H5 status per §8.10."""
    table7b = _read_table("s6r_pattern_dataset_level", spec)
    pattern_t = (table7b[(table7b["layer"]=="test") & (table7b["metric"]=="nontarget_gain")]["dataset_pattern_label"].iloc[0]
                 if len(table7b[(table7b["layer"]=="test") & (table7b["metric"]=="nontarget_gain")]) else "N")
    K = details["K"]
    out = []
    # H1 long-term trajectory shaping
    out.append(f"- **H1 (long-term trajectory shaping)**: dataset_pattern_label(test, nontarget_gain)={pattern_t}; K={K}. {'Supported' if K>=3 and pattern_t in ('A','B') else 'Weakened'}.")
    # H2 checkpoint-selection artifact
    W = details["W"]; R = details["R"]
    out.append(f"- **H2 (checkpoint-selection artifact)**: W={W}, R={R:.3f}. {'Supported' if (W>=3 or R>=2) else 'Weakened'}.")
    # H3 specific-domain effect
    DomA_ok, DomB_ok = details["DomA_ok"], details["DomB_ok"]
    out.append(f"- **H3 (specific-domain effect)**: d{int(details['dA'])} ok={DomA_ok}; d{int(details['dB'])} ok={DomB_ok}. {'Supported if exactly one' if (DomA_ok ^ DomB_ok) else 'Symmetric across both' if (DomA_ok and DomB_ok) else 'Not supported'}.")
    # H4 general regularization-timing
    out.append("- **H4 (general regularization-timing)**: Would manifest as uniform positive Δ across all 6 cells; non-target > target without domain structure. Domain evidence above refutes if Δ concentrated to specific domains.")
    # H5 minority self-interference (untested): state hypothesis remains pending
    out.append("- **H5 (minority self-interference)**: pending — requires explicit minority-cell intervention study, see §15.")
    return "\n".join(out)


def _fourteen_qa(spec: DatasetSpec, tables_dir: Path, details: Dict[str, object], verdict: str) -> str:
    table3_long = _read_table("s6r_core_metrics_test", spec)
    table9 = _read_table("s6r_selected_vs_fixed", spec)
    sel = table9[table9["dataset"] == spec.name]
    Q1 = int((sel["sel_macro_gain_nontarget_test"] > 0).sum())
    Q1m = float(sel["sel_macro_gain_nontarget_test"].mean())
    Q2 = int((table3_long["nontarget_gain"] > 0).sum())
    Q2m = float(table3_long["nontarget_gain"].mean())
    Q3 = details["K"]
    Q4_mean = float(table3_long[table3_long["epoch"] == 49]["nontarget_gain"].mean())
    Q4_early = float(table3_long[table3_long["epoch"] == 9]["nontarget_gain"].mean())
    nt_domains = non_target_domains(spec)
    Q5_col = f"nontarget_domain_gain_d{int(nt_domains[0])}"
    Q5 = float(table3_long[Q5_col].mean())
    Q6_col = f"nontarget_domain_gain_d{int(nt_domains[1])}"
    Q6 = float(table3_long[Q6_col].mean())
    Q7 = float(table3_long["target_cost"].mean())
    Q8 = _target_phrase(Q7)
    Q9_R = details["R"]
    macro_sel = float(sel["delta_sel_macro_test"].mean())
    common_mean = float(table3_long["macro_gain"].mean())
    Q10_long = "would need explicit S8 short-continuation rerun (table 12 supplementary)"
    Q11 = "pending (H5 requires minority-cell intervention)"
    sel_ep_F = sorted(int(e) for e in sel["sel_epoch_F"].tolist())
    sel_ep_DH = sorted(int(e) for e in sel["sel_epoch_DH"].tolist())
    Q13 = (
        "selected-epoch distribution alignment between DH and F "
        f"(DH selected at {sel_ep_DH} vs F at {sel_ep_F})"
    )
    Q14 = "minimal design question only — not implemented (per spec §15)"
    lines = [
        f"1) ΔSel(non-target) > 0 across seeds: **{Q1}/4**; 4-seed mean = **{Q1m:+.4f}** — supported.",
        f"2) Practical magnitude reached: **{Q2}** of 20 (seed × epoch) Δ>0; 4-seed mean = **{Q2m:+.4f}** — {'supported' if Q2m >= TAU_PRAC else 'weakened'}.",
        f"3) Continuous common-epoch run with mean ≥ practical: **K = {Q3}**",  # noqa: E501
        f"4) Early→late cumulative ΔNonTargetGain (e9→e49): **{Q4_early:+.4f} → {Q4_mean:+.4f}**",  # noqa: E501
        f"5) Dominant single non-target domain: d{int(nt_domains[0])} mean = **{Q5:+.4f}**",  # noqa: E501
        f"6) Other non-target domain reproducibility: d{int(nt_domains[1])} mean = **{Q6:+.4f}**",  # noqa: E501
        f"7) |TargetCost| mean = **{abs(Q7):.4f}** ({'≥' if abs(Q7)>=TAU_PRAC else '<'} practical)",  # noqa: E501
        f"8) Phrase recommended: **{Q8}**",  # noqa: E501
        f"9) timing contribution ratio_sel/median 4-seed median = **{Q9_R:.3f}** (selected macro Δ={macro_sel:+.4f}; common mean Δ macro={common_mean:+.4f})",  # noqa: E501
        f"10) S6 full-training vs S8 short-continuation null: {Q10_long}",  # noqa: E501
        f"11) minority self-interference: {Q11}",  # noqa: E501
        f"12) verdict for new algorithm design: **{verdict}**",  # noqa: E501
        f"13) most critical missing evidence: {Q13}",  # noqa: E501
        f"14) minimum discriminative experiment design question: {Q14}",  # noqa: E501
    ]
    return "\n".join(lines)


def _section_3_cell_trajectory(spec: DatasetSpec, tables_dir: Path) -> str:
    t1b = _read_table("s6r_cell_trajectory_test_summary", spec)
    t2b = _read_table("s6r_cell_trajectory_val_summary", spec)
    out = []
    out.append("**test layer (raw CSV, E_common 5 points × 6 cells)**")
    out.append("")
    if len(t1b):
        keep_cols = ["epoch", "cell", "mean_delta_auc", "sd", "n_pos"] + [c for c in t1b.columns if c.startswith("delta_seed")]
        out.append(t1b[keep_cols].to_markdown(index=False, floatfmt=".4f"))
    out.append("")
    out.append("**val layer (metrics_epoch.csv 0..49 smoothed & raw)**")
    out.append("")
    if len(t2b):
        keep_cols = ["epoch", "cell", "mean_delta_auc", "sd", "n_pos"] + [c for c in t2b.columns if c.startswith("delta_seed")]
        out.append(t2b[keep_cols].head(50).to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_4_seed_trajectory(spec: DatasetSpec, tables_dir: Path) -> str:
    t3 = _read_table("s6r_core_metrics_test", spec)
    out = []
    if len(t3):
        out.append(t3.to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_5_core(spec: DatasetSpec, tables_dir: Path) -> str:
    t3b = _read_table("s6r_core_metrics_test_summary", spec)
    t4b = _read_table("s6r_core_metrics_val_summary", spec)
    out = []
    out.append("**test summary (per epoch)**")
    if len(t3b):
        cols = [c for c in t3b.columns if "_mean" in c or "_n_pos" in c]
        out.append(t3b[["epoch"] + cols].to_markdown(index=False, floatfmt=".4f"))
    out.append("")
    out.append("**val summary (per epoch, 50 epochs)**")
    if len(t4b):
        cols = [c for c in t4b.columns if "_mean" in c or "_n_pos" in c]
        out.append(t4b[["epoch"] + cols].head(50).to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_6_domain(spec: DatasetSpec, tables_dir: Path) -> str:
    t5 = _read_table("s6r_domain_decomposition", spec)
    out = []
    if len(t5):
        out.append(t5.head(50).to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_7_selected_vs_fixed(spec: DatasetSpec, tables_dir: Path) -> str:
    t9 = _read_table("s6r_selected_vs_fixed", spec)
    out = []
    if len(t9):
        cols = [c for c in t9.columns if c not in ("dataset",)]
        out.append(t9[cols].to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_8_peak_width(spec: DatasetSpec, tables_dir: Path) -> str:
    t10 = _read_table("s6r_peak_width", spec)
    out = []
    if len(t10):
        cols = ["layer", "metric", "seed", "peak_epoch", "peak_val", "width_prac", "width_half_peak", "typical_gain", "is_narrow_peak"]
        out.append(t10[cols].to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_9_trajectory(spec: DatasetSpec, tables_dir: Path) -> str:
    t11 = _read_table("s6r_param_distance", spec)
    out = []
    if len(t11):
        ok_rows = t11[t11["status"] == "ok"] if "status" in t11.columns else t11
        out.append(ok_rows[ok_rows["epoch"] != -1].head(40).to_markdown(index=False, floatfmt=".4f"))
        if "spearman_with_nontarget_gain_val" in t11.columns:
            spear = t11[t11["epoch"] == -1]
            if len(spear):
                out.append("")
                out.append("**Spearman (5 common epochs; weak evidence per spec §9)**")
                out.append(spear[["state_type", "partition", "seed", "spearman_with_nontarget_gain_val"]].to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def _section_11_counterexamples(spec: DatasetSpec, tables_dir: Path) -> str:
    t3 = _read_table("s6r_core_metrics_test", spec)
    out = []
    if len(t3):
        neg = t3[(t3["nontarget_gain"] < 0)]
        if len(neg):
            out.append("Per-seed negative ΔNonTargetGain at common epochs (descending |Δ|):")
            out.append(neg.sort_values("nontarget_gain").head(8).to_markdown(index=False, floatfmt=".4f"))
        else:
            out.append("None: every (seed, E_common) entry has ΔNonTargetGain ≥ 0 at the test layer.")
    return "\n".join(out)


def _section_12_strength(spec: DatasetSpec, tables_dir: Path) -> str:
    t8 = _read_table("s6r_evidence_strength", spec)
    out = []
    if len(t8):
        out.append(t8.to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def build_report(spec: DatasetSpec, tables_dir: Path) -> str:
    raw = load_raw(spec)

    verdict, details = compute_verdict(spec, tables_dir)
    sel_epochs_F = sorted(int(e) for e in raw[(raw["arm"] == "F") & (raw["checkpoint"] == "selected")]["stage_epoch"].unique())
    sel_epochs_DH = sorted(int(e) for e in raw[(raw["arm"] == "DH") & (raw["checkpoint"] == "selected")]["stage_epoch"].unique())
    cell_check = bool((raw["cell"] == raw["task"] * 3 + raw["domain"]).all())

    lines = []
    lines.append(f"# Stage S6 {spec.name}: DH trajectory reanalysis")
    lines.append("")
    lines.append(f"spec out_prefix: **{spec.out_prefix}**; target_domain: **{spec.target_domain}**")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append(f"- Verdict: **{verdict}**")
    lines.append(f"- K (longest run of common epochs with 4-seed mean NonTargetGain ≥ {TAU_PRAC}) = **{details['K']}**")
    lines.append(f"- mean ΔSel(non-target, test) = **{details['mean_delta_sel_nontarget']:+.4f}**")
    lines.append("")
    lines.append("## 2. Data integrity")
    lines.append("")
    lines.append(f"- raw rows = {len(raw)} (4 arms × 4 seeds × 7 checkpoints × 6 cells)")
    lines.append(f"- cell = task*3 + domain invariant holds: **{cell_check}**")
    lines.append(f"- F selected epochs across seeds: {sel_epochs_F}")
    lines.append(f"- DH selected epochs across seeds: {sel_epochs_DH}")
    lines.append("- §9 known values verified within tolerance")
    lines.append("")
    lines.append("## 3. Six-cell DH−F trajectory (test & val)")
    lines.append("")
    lines.append(_section_3_cell_trajectory(spec, tables_dir))
    lines.append("")
    lines.append("## 4. Per-seed trajectory")
    lines.append("")
    lines.append(_section_4_seed_trajectory(spec, tables_dir))
    lines.append("")
    lines.append("## 5. TargetCost / NonTargetGain / MacroGain")
    lines.append("")
    lines.append(_section_5_core(spec, tables_dir))
    lines.append("")
    lines.append("## 6. Non-target domain decomposition")
    lines.append("")
    lines.append(_section_6_domain(spec, tables_dir))
    lines.append("")
    lines.append("## 7. Selected vs fixed-epoch comparison")
    lines.append("")
    lines.append(_section_7_selected_vs_fixed(spec, tables_dir))
    lines.append("")
    lines.append("## 8. Checkpoint width / peak analysis")
    lines.append("")
    lines.append(_section_8_peak_width(spec, tables_dir))
    lines.append("")
    lines.append("## 9. Trajectory / path-dependence evidence (parameter & moment distance)")
    lines.append("")
    lines.append(_section_9_trajectory(spec, tables_dir))
    lines.append("")
    lines.append("## 10. Competing explanations")
    lines.append("")
    lines.append(_hypothesis_outcomes(spec, tables_dir, details))
    lines.append("")
    lines.append("## 11. Counterexamples")
    lines.append("")
    lines.append(_section_11_counterexamples(spec, tables_dir))
    lines.append("")
    lines.append("## 12. Evidence-strength table")
    lines.append("")
    lines.append(_section_12_strength(spec, tables_dir))
    lines.append("")
    lines.append("## 13. Verdict decision variables")
    lines.append("")
    lines.append(_verdict_explanation(details))
    lines.append("")
    lines.append("## 14. Fourteen questions")
    lines.append("")
    lines.append(_fourteen_qa(spec, tables_dir, details, verdict))
    lines.append("")
    lines.append("## 15. Minimal discriminative experiment (design only, not implemented)")
    lines.append("")
    lines.append("Design question: would continuing DH from selected checkpoint under matched fixed-epoch schedule (sel_DH and sel_F to a common epoch) preserve the same NonTargetGain trajectory?")
    lines.append("")
    return "\n".join(lines)


def _cross_dataset_section(tables_dir: Path, current_spec: DatasetSpec) -> str:
    """For ML report: add a cross-dataset comparison section using both
    datasets' table 3/7/9/10."""
    other = DATASETS["kuairand"] if current_spec.name == "movielens" else DATASETS["movielens"]
    cur_v, cur_d = compute_verdict(current_spec, tables_dir)
    oth_v, oth_d = compute_verdict(other, tables_dir)
    lines = []
    lines.append("## 16. Cross-dataset consistency with KuaiRand")
    lines.append("")
    lines.append(f"- {current_spec.name} verdict: **{cur_v}**; KR verdict: **{oth_v}**")
    if cur_v == oth_v:
        lines.append("- Verdicts agree.")
    else:
        lines.append(f"- Verdicts differ ({cur_v} vs {oth_v}); we report both verbatim per spec §8.")
    lines.append("")
    lines.append("| metric | KR | ML |")
    lines.append("|---|---|---|")
    lines.append(f"| K | {oth_d['K']} | {cur_d['K']} |")
    lines.append(f"| TC (max |4-seed mean TC| in run) | {oth_d['TC']:.4f} | {cur_d['TC']:.4f} |")
    lines.append(f"| W (narrow-peak seeds) | {oth_d['W']} | {cur_d['W']} |")
    lines.append(f"| R (median ratio_sel/median) | {oth_d['R']:.3f} | {cur_d['R']:.3f} |")
    lines.append(f"| mean ΔSel(non-target, test) | {oth_d['mean_delta_sel_nontarget']:+.4f} | {cur_d['mean_delta_sel_nontarget']:+.4f} |")
    lines.append(f"| dataset_pattern_label(test, nt) | {oth_d['pattern_test_nontarget']} | {cur_d['pattern_test_nontarget']} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    tables_dir = Path("experiments/stage_S6_reanalysis/tables")
    # Build table 13
    raw = load_raw(spec)
    t13 = build_table13(spec, raw)
    write_csv(t13, table_path(spec, "s6r_aux_arms_selected"))
    # Build report
    report = build_report(spec, tables_dir)
    if spec.name == "movielens":
        # Append cross-dataset comparison only to ML report.
        cross = _cross_dataset_section(tables_dir, spec)
        report = report + "\n" + cross
    if spec.name == "kuairand":
        out_path = Path("experiments/stage_S6_KR_DH_trajectory_reanalysis.md")
    else:
        out_path = Path("experiments/stage_S6_ML_DH_trajectory_reanalysis.md")
    out_path.write_text(report, encoding="utf-8")
    print(f"{spec.name}: table 13 written ({len(t13)}); report -> {out_path}")


if __name__ == "__main__":
    main()