"""s6r_pattern_verdict.py

Generate tables 7 (per-seed pattern classification) and 8 (evidence-strength)
for one dataset. These do not directly drive the final verdict — that lives
in s6r_build_report.py — but they provide the per-row inputs that the report
combines with tables 9/10/11.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from s6r_common import (
    CHECKPOINT_TO_EPOCH,
    DatasetSpec,
    E_COMMON,
    SEEDS,
    TAU_PRAC,
    classify_pattern,
    delta_pair,
    evidence_strength,
    load_raw,
    load_val_epoch,
    n_ge_practical,
    non_target_cells,
    non_target_domains,
    parse_dataset,
    sign_changes,
    smooth_moving,
    table_path,
    total_rise,
    write_csv,
)


def _val_sequence(
    spec: DatasetSpec, layer: str, metric: str, seed: int, smooth: bool = False
) -> List[float]:
    """Return 1-D sequence across epochs (test: 5-pt common; val: 0..49)."""
    if layer == "test":
        raw = load_raw(spec)
        keep = raw[raw["checkpoint"].isin(list(CHECKPOINT_TO_EPOCH.keys()))].copy()
        d = delta_pair(keep, "DH", "F", "auc")
        d = d[d["seed"] == int(seed)]
    else:
        rows = []
        for arm in ("F", "DH"):
            v = load_val_epoch(spec, arm, int(seed))
            rows.append(v)
        keep = pd.concat(rows, ignore_index=True)
        d = delta_pair(keep, "DH", "F", "validation_auc")
    if metric == "nontarget_gain":
        nt = non_target_cells(spec)
        mask = d["domain"].astype(int).isin([t[0] for t in nt])
        agg = d[mask].groupby("epoch")["delta"].mean()
    elif metric == "macro_gain":
        agg = d.groupby("epoch")["delta"].mean()
    else:
        raise ValueError(metric)
    if layer == "test":
        seq = [float(agg.get(e, float("nan"))) for e in E_COMMON]
    else:
        full = list(range(0, 50))
        seq = [float(agg.get(e, float("nan"))) for e in full]
    if smooth and layer == "val":
        seq = smooth_moving(seq, w=10).tolist()
    return seq


def build_table7(spec: DatasetSpec, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in ("test", "val"):
        smooth = layer == "val"
        for metric in ("nontarget_gain", "macro_gain"):
            for seed in SEEDS:
                seq = _val_sequence(spec, layer, metric, int(seed), smooth=smooth)
                pat = classify_pattern(seq)
                row = {
                    "dataset": spec.name, "layer": layer, "metric": metric,
                    "seed": int(seed), "pattern": pat,
                    "n_ge_practical": n_ge_practical(seq),
                    "total_rise": total_rise(seq),
                    "max_val": max(seq) if seq else float("nan"),
                    "argmax_epoch": (E_COMMON if layer == "test" else list(range(0, 50)))[seq.index(max(seq))] if seq else -1,
                    "sign_changes": sign_changes(seq),
                }
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "metric", "seed"]).reset_index(drop=True)


def build_table7b(spec: DatasetSpec, table7: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in ("test", "val"):
        for metric in ("nontarget_gain", "macro_gain"):
            sub = table7[(table7["layer"] == layer) & (table7["metric"] == metric)]
            patterns = sub["pattern"].tolist()
            n_seeds_ge = int((sub["max_val"] >= TAU_PRAC).sum())
            patterns_str = ",".join(patterns)
            # heterogeneity: ≥TAU_PRAC seeds<=2 yet 4-mean≥TAU_PRAC OR {A/B} ⊕ {C/N} both≥1
            mean_max = float(sub["max_val"].mean())
            mean_total = float(sub["total_rise"].mean())
            hetero = False
            if n_seeds_ge <= 2 and (mean_max >= TAU_PRAC or mean_total >= TAU_PRAC):
                hetero = True
            ab_count = sum(1 for p in patterns if p in ("A", "B"))
            cn_count = sum(1 for p in patterns if p in ("C", "N"))
            if ab_count >= 1 and cn_count >= 1:
                hetero = True
            # dataset_pattern_label per spec priority:
            #   1. all four seeds identical -> take that label
            #   2. >=3 seeds share the same pattern in {A,B} and no C -> majority, annotated
            #   3. heterogeneity (D) condition met -> D
            #   4. otherwise mixed
            label = "mixed"
            note = ""
            c_seeds = sum(1 for p in patterns if p == "C")
            if len(set(patterns)) == 1:
                label = patterns[0]
            else:
                maj = next((p for p in ("A", "B") if patterns.count(p) >= 3), None)
                if maj is not None and c_seeds == 0:
                    label = maj
                    note = f"{patterns.count(maj)}/4 {maj}, no C"
                elif hetero:
                    label = "D"
            rows.append({
                "dataset": spec.name, "layer": layer, "metric": metric,
                "patterns_per_seed": patterns_str,
                "n_seeds_ge_practical": n_seeds_ge,
                "seed_heterogeneity": bool(hetero),
                "dataset_pattern_label": label,
                "note": note,
            })
    return pd.DataFrame(rows).sort_values(["layer", "metric"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 8 — evidence strength across the P1..P10 fixed propositions.
# ---------------------------------------------------------------------------
def build_table8(
    spec: DatasetSpec,
    raw: pd.DataFrame,
    table3_long: pd.DataFrame,
    table3_sum: pd.DataFrame,
    table7: pd.DataFrame,
    table9: pd.DataFrame,
    table5: pd.DataFrame,
    table4_long: pd.DataFrame,
) -> pd.DataFrame:
    """Compute evidence strength per proposition.

    Most propositions reference downstream tables that are computed by other
    scripts. We re-derive what is needed from raw + table3/4 + table5 here so
    this script can be developed and re-run independently; the report builder
    uses the table8 we produce for citation.

    ``both_layers_agree`` is evaluated for real: the val-layer (table 4)
    4-seed mean of the relevant metric must agree in direction with the
    test-layer claim; propositions with no val analog are single-layer.
    """
    nt_domains = non_target_domains(spec)
    target_d = spec.target_domain

    # val-layer agreement references (4-seed means over all 50 epochs)
    val_ok = len(table4_long) > 0
    val_nt_mean = float(table4_long["nontarget_gain"].mean()) if val_ok else float("nan")
    val_macro_mean = float(table4_long["macro_gain"].mean()) if val_ok else float("nan")
    val_tc_abs_mean = float(table4_long["target_cost"].abs().mean()) if val_ok else float("nan")
    val_dom_mean = {
        int(d): float(table4_long[f"nontarget_domain_gain_d{int(d)}"].mean()) if val_ok else float("nan")
        for d in nt_domains
    }

    # helper: 4-seed mean at common epochs
    def mean_at(layer_metric: str, epoch: int, mask_col: Optional[str] = None) -> float:
        sub = table3_long[(table3_long["epoch"] == epoch)]
        if mask_col:
            return float(sub[mask_col].mean())
        return float(sub[layer_metric].mean())

    def npos_at(col: str, epoch: int) -> int:
        sub = table3_long[table3_long["epoch"] == epoch]
        return int((sub[col] > 0).sum())

    # P1: selected non-target gain exists across 4 seeds (use raw CSV)
    sel = raw[(raw["checkpoint"] == "selected") & (raw["arm"].isin(["F", "DH"]))].copy()
    sel_piv = sel.pivot_table(
        index=["seed", "domain", "task"], columns="arm", values="auc"
    ).reset_index()
    sel_piv["delta"] = sel_piv["DH"] - sel_piv["F"]
    nt_sel = sel_piv[~sel_piv["domain"].isin([target_d])]
    nt_per_seed = nt_sel.groupby("seed")["delta"].mean()
    p1_mag = float(nt_per_seed.mean())
    p1_n = int((nt_per_seed > 0).sum())
    p1_strength = evidence_strength(p1_n, p1_mag, val_ok and val_nt_mean > 0, False)

    # P2: practical magnitude at selected (also at >=3 common epochs)
    p2_n = int((table3_long["nontarget_gain"] > 0).sum())  # per (seed,epoch) over 5 common
    p2_mag = float(table3_long["nontarget_gain"].mean())
    p2_strength = evidence_strength(min(4, p2_n // 5), p2_mag, val_ok and val_nt_mean > 0, False)

    # P3: span >= 3 consecutive common epochs at practical
    epochs_seq = table3_long.groupby("epoch")["nontarget_gain"].mean().reindex(list(E_COMMON))
    pos_mask = (epochs_seq >= TAU_PRAC).values
    max_run = 0
    cur = 0
    for v in pos_mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    p3_mag = float(epochs_seq.max())
    p3_strength = evidence_strength(
        int((epochs_seq >= TAU_PRAC).sum()) if max_run >= 3 else 0,
        p3_mag, val_ok and val_nt_mean > 0, max_run < 3,
    )

    # P4: early -> late cumulative (use table3_sum, test layer)
    early = float(table3_long[table3_long["epoch"] == 9]["nontarget_gain"].mean())
    late = float(table3_long[table3_long["epoch"] == 49]["nontarget_gain"].mean())
    p4_mag = late - early
    p4_n = int((table3_long[table3_long["epoch"] == 49]["nontarget_gain"] > 0).sum())
    p4_strength = evidence_strength(p4_n, p4_mag, val_ok and val_nt_mean > 0, late < early)

    # P5: dominant single non-target domain (KR d0 / ML d0?)
    # Take the domain with higher mean over common epochs
    by_domain = {}
    for nd in nt_domains:
        col = f"nontarget_domain_gain_d{int(nd)}"
        by_domain[int(nd)] = float(table3_long[col].mean())
    primary = max(by_domain, key=by_domain.get) if by_domain else None
    primary_col = f"nontarget_domain_gain_d{int(primary)}" if primary is not None else None
    primary_n = int((table3_long[primary_col] > 0).sum()) if primary_col else 0
    p5_mag = by_domain.get(int(primary), 0.0)
    # Counterexample: other non-target domain < 0.003 mean
    counter_p5 = False
    for d, m in by_domain.items():
        if int(d) != int(primary) and m < TAU_PRAC:
            counter_p5 = True
    p5_strength = evidence_strength(
        primary_n, p5_mag,
        val_ok and val_dom_mean.get(int(primary), float("nan")) > 0,
        counter_p5,
    )

    # P6: another non-target domain reproduces
    secondary_n_pos = 0
    secondary_mag = 0.0
    secondary_counter = False
    secondary_dom = None
    for d, m in by_domain.items():
        if int(d) != int(primary):
            col = f"nontarget_domain_gain_d{int(d)}"
            secondary_n_pos = int((table3_long[col] > 0).sum())
            secondary_mag = m
            secondary_counter = (m < TAU_PRAC)
            secondary_dom = int(d)
    p6_strength = evidence_strength(
        secondary_n_pos, secondary_mag,
        val_ok and secondary_dom is not None and val_dom_mean.get(secondary_dom, float("nan")) > 0,
        secondary_counter,
    )

    # P7: target cost < practical across 5 epochs (4-seed mean abs)
    tc_abs = table3_long["target_cost"].abs()
    p7_mag = float(table3_long["target_cost"].mean())
    p7_n = int((tc_abs < TAU_PRAC).sum())
    p7_counter = bool((tc_abs >= TAU_PRAC).sum() >= 4)
    p7_strength = evidence_strength(
        p7_n, abs(p7_mag), val_ok and val_tc_abs_mean < TAU_PRAC, p7_counter
    )

    # P8: macro gain (also nontarget) wide window — use test common epochs mean
    p8_mag = float(table3_long["macro_gain"].mean())
    p8_n = int((table3_long["macro_gain"] > 0).sum())
    p8_strength = evidence_strength(p8_n, p8_mag, val_ok and val_macro_mean > 0, False)

    # P9: timing contribution small (use table9 ratio_sel_over_median);
    # no val-layer analog -> single-layer evidence (MEDIUM ceiling)
    if len(table9) > 0:
        rs = table9["ratio_sel_over_median"].dropna()
        p9_mag = float(rs.median()) if len(rs) else float("nan")
        p9_n = int((rs < 2).sum())
        p9_counter = int((rs >= 2).sum()) >= 2
        p9_strength = evidence_strength(p9_n, p9_mag, False, p9_counter)
    else:
        p9_mag, p9_n, p9_counter, p9_strength = float("nan"), 0, True, "WEAK"

    # P10: long-term trajectory dependence — K>=3, S_k=True, see §7.9 A path
    # Use info we can compute here: max_run & n_pos
    p10_mag = float(epochs_seq.mean())
    p10_counter = max_run < 3
    p10_strength = evidence_strength(
        min(4, max_run), p10_mag, val_ok and val_nt_mean > 0, p10_counter
    )

    rows = [
        {"proposition": "P1", "description": "selected non-target gain real in 4 seeds", "strength": p1_strength, "n_seeds_supporting": p1_n, "magnitude": p1_mag, "checkpoint_consistency": "", "counterexample": "", "basis": "raw CSV selected; §9 known values"},
        {"proposition": "P2", "description": "reaches practical magnitude", "strength": p2_strength, "n_seeds_supporting": p2_n, "magnitude": p2_mag, "checkpoint_consistency": f"{int((epochs_seq>=TAU_PRAC).sum())}/5 common", "counterexample": "", "basis": "table 3/3b"},
        {"proposition": "P3", "description": "spans multiple consecutive common checkpoints", "strength": p3_strength, "n_seeds_supporting": int((epochs_seq>=TAU_PRAC).sum()) if max_run>=3 else 0, "magnitude": p3_mag, "checkpoint_consistency": f"run={max_run}/5", "counterexample": "" if max_run>=3 else "max run < 3", "basis": "table 3"},
        {"proposition": "P4", "description": "early->late cumulative rise", "strength": p4_strength, "n_seeds_supporting": p4_n, "magnitude": p4_mag, "checkpoint_consistency": f"e9={early:.4f}→e49={late:.4f}", "counterexample": "", "basis": "table 3"},
        {"proposition": "P5", "description": f"driven by single non-target domain (d{int(primary) if primary is not None else '?'})", "strength": p5_strength, "n_seeds_supporting": primary_n, "magnitude": p5_mag, "checkpoint_consistency": "", "counterexample": "" if not counter_p5 else "other domain < practical", "basis": "table 3"},
        {"proposition": "P6", "description": "another non-target domain reproduces", "strength": p6_strength, "n_seeds_supporting": secondary_n_pos, "magnitude": secondary_mag, "checkpoint_consistency": "", "counterexample": "" if not secondary_counter else "secondary < practical", "basis": "table 3"},
        {"proposition": "P7", "description": "target cost below practical", "strength": p7_strength, "n_seeds_supporting": p7_n, "magnitude": p7_mag, "checkpoint_consistency": "", "counterexample": "" if not p7_counter else "≥1 seed |TC|>=0.003", "basis": "table 3"},
        {"proposition": "P8", "description": "wide-window macro gain", "strength": p8_strength, "n_seeds_supporting": p8_n, "magnitude": p8_mag, "checkpoint_consistency": "", "counterexample": "", "basis": "table 3"},
        {"proposition": "P9", "description": "timing contribution small (ratio_sel/median)", "strength": p9_strength, "n_seeds_supporting": p9_n, "magnitude": p9_mag, "checkpoint_consistency": "", "counterexample": "" if not p9_counter else "≥2 seeds ratio>=2", "basis": "table 9"},
        {"proposition": "P10", "description": "consistent with long-term trajectory dependence", "strength": p10_strength, "n_seeds_supporting": min(4, max_run), "magnitude": p10_mag, "checkpoint_consistency": f"run={max_run}/5", "counterexample": "" if not p10_counter else "max_run<3", "basis": "table 7 + table 3"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    spec = parse_dataset()
    raw = load_raw(spec)
    # Need table3 long and table3 sum and table9 and table5 to build table8.
    table3 = pd.read_csv(table_path(spec, "s6r_core_metrics_test"), encoding="utf-8-sig")
    table3_sum = pd.read_csv(table_path(spec, "s6r_core_metrics_test_summary"), encoding="utf-8-sig")
    table4 = pd.read_csv(table_path(spec, "s6r_core_metrics_val"), encoding="utf-8-sig")
    table5 = pd.read_csv(table_path(spec, "s6r_domain_decomposition"), encoding="utf-8-sig")
    table9_path = table_path(spec, "s6r_selected_vs_fixed")
    table9 = pd.read_csv(table9_path, encoding="utf-8-sig") if table9_path.exists() else pd.DataFrame()
    t7 = build_table7(spec, raw)
    t7b = build_table7b(spec, t7)
    t8 = build_table8(spec, raw, table3, table3_sum, t7, table9, table5, table4)
    write_csv(t7, table_path(spec, "s6r_pattern_classification"))
    write_csv(t7b, table_path(spec, "s6r_pattern_dataset_level"))
    write_csv(t8, table_path(spec, "s6r_evidence_strength"))
    print(f"{spec.name}: tables 7/7b/8 written ({len(t7)}+{len(t7b)}+{len(t8)})")


if __name__ == "__main__":
    main()