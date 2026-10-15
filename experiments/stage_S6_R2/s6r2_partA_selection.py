"""s6r2_partA_selection.py

Part A: pre-registered selection-rule robustness.

R0..R4 are computed independently per (dataset, arm, seed) using
**validation layer only** (spec §3.1, §9.2). Tie-break = earliest epoch.

Outputs (per dataset, into `experiments/stage_S6_R2_raw_results/{dataset}/`):

    s6r2_partA_epochs_{dataset}.csv
        rule x arm x seed -> selected_epoch, selection_objective, canonical_match
    s6r2_partA_val_metrics_{dataset}.csv
        rule x arm x seed -> 6-cell AUCs + T + N + M + WD + WC at the
        selected epoch (val layer, always present).
    s6r2_partA_test_metrics_{dataset}.csv
        rule x arm x seed -> same 10 metrics at test layer, with explicit
        NA + test_available flag for non-saved epochs.
    s6r2_partA_grid_epochs_{dataset}.csv
        A-grid variant (search restricted to GRID ∪ {R0}).
    s6r2_partA_grid_test_metrics_{dataset}.csv
        A-grid variant: every rule has a test value (within the grid).
    s6r2_partA_gate_summary_{dataset}.csv
        For each (rule, metric∈{N,T,M}, layer∈{val, test, test_grid}):
            4-seed mean, SD(ddof=1), per-seed, n_pos, n_ge_003, n_ge_005.

Run via:
    python -m experiments.stage_S6_R2.s6r2_partA_selection --dataset kuairand
    python -m experiments.stage_S6_R2.s6r2_partA_selection --dataset movielens
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402


RULES = ("R0", "R1", "R2", "R3", "R4")
LAYERS = ("val",)
TEST_LAYERS = ("test", "test_grid")


def _agg_to_row_dict(rule: str, arm: str, seed: int, ep: int, agg: c.CellAgg) -> Dict[str, object]:
    """Convert CellAgg to a flat metric row (val layer)."""
    out = {
        "rule": rule, "arm": arm, "seed": int(seed),
        "selected_epoch": int(ep),
        "M": agg.M, "T": agg.T, "N": agg.N,
        "D0": agg.D0, "D1": agg.D1, "D2": agg.D2,
        "WD": agg.WD, "WC": agg.WC,
        "d0t0": agg.cell_aucs["d0t0"],
        "d0t1": agg.cell_aucs["d0t1"],
        "d1t0": agg.cell_aucs["d1t0"],
        "d1t1": agg.cell_aucs["d1t1"],
        "d2t0": agg.cell_aucs["d2t0"],
        "d2t1": agg.cell_aucs["d2t1"],
    }
    return out


def _test_dict_to_row_dict(
    rule: str, arm: str, seed: int, ep: int, m_dict: Dict[str, float],
    source: str, test_available: int,
) -> Dict[str, object]:
    out = {
        "rule": rule, "arm": arm, "seed": int(seed),
        "selected_epoch": int(ep),
        "checkpoint_source": source,
        "test_available": int(test_available),
        "M": m_dict["M"], "T": m_dict["T"], "N": m_dict["N"],
        "D0": m_dict["D0"], "D1": m_dict["D1"], "D2": m_dict["D2"],
        "WD": m_dict["WD"], "WC": m_dict["WC"],
        "d0t0": m_dict["d0t0"], "d0t1": m_dict["d0t1"],
        "d1t0": m_dict["d1t0"], "d1t1": m_dict["d1t1"],
        "d2t0": m_dict["d2t0"], "d2t1": m_dict["d2t1"],
    }
    return out


def _test_na_row(
    rule: str, arm: str, seed: int, ep: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "rule": rule, "arm": arm, "seed": int(seed),
        "selected_epoch": int(ep),
        "checkpoint_source": "NA",
        "test_available": 0,
        "M": float("nan"), "T": float("nan"), "N": float("nan"),
        "D0": float("nan"), "D1": float("nan"), "D2": float("nan"),
        "WD": float("nan"), "WC": float("nan"),
        "d0t0": float("nan"), "d0t1": float("nan"),
        "d1t0": float("nan"), "d1t1": float("nan"),
        "d2t0": float("nan"), "d2t1": float("nan"),
    }
    return out


def run_for_dataset(spec: c.DatasetSpec) -> None:
    out_dir = c.ensure_out_dir(spec)
    print(f"[PartA] dataset={spec.name} -> {out_dir}")

    # Pre-load all aggregations to avoid duplicating work between rules.
    per_arm_seed_agg: Dict[Tuple[str, int], Dict[int, c.CellAgg]] = {}
    for arm in c.ARMS:
        for seed in c.SEEDS:
            per_arm_seed_agg[(arm, seed)] = c.aggregate_per_epoch(spec, arm, seed)

    # ---- A-full: select epoch across all 50 epochs ----
    epochs_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for arm in c.ARMS:
        for seed in c.SEEDS:
            agg_map = per_arm_seed_agg[(arm, seed)]
            e0, v0 = c.select_rule_R0(spec, arm, seed)
            e1, v1 = c.select_rule_R1(spec, arm, seed)
            e2, v2 = c.select_rule_R2(spec, arm, seed)
            e3, v3 = c.select_rule_R3(spec, arm, seed)
            e4, _, weights = c.select_rule_R4(spec, arm, seed)
            v4 = float(
                weights[0] * agg_map[e4].D0
                + weights[1] * agg_map[e4].D1
                + weights[2] * agg_map[e4].D2
            )
            e_canon = c.canonical_selected_epoch(spec, arm, seed)
            assert e_canon == e0, (
                f"R0 mismatch with canonical for {spec.name}/{arm}/seed{seed}: "
                f"R0={e0} canonical={e_canon}"
            )
            sel = {"R0": e0, "R1": e1, "R2": e2, "R3": e3, "R4": e4}
            obj = {"R0": v0, "R1": v1, "R2": v2, "R3": v3, "R4": v4}
            for R in RULES:
                ep = sel[R]
                epochs_rows.append({
                    "rule": R, "arm": arm, "seed": int(seed),
                    "selected_epoch": int(ep),
                    "objective_value": float(obj[R]),
                    "canonical_match": int(ep == e_canon),
                })
                agg = agg_map[ep]
                val_rows.append(_agg_to_row_dict(R, arm, seed, ep, agg))

                # Test lookup per §4.3.
                m, src, avail = c.aggregate_test_one_epoch(spec, arm, seed, ep)
                if avail and m is not None:
                    test_rows.append(_test_dict_to_row_dict(R, arm, seed, ep, m, src, 1))
                else:
                    test_rows.append(_test_na_row(R, arm, seed, ep))

    df_epochs = pd.DataFrame(epochs_rows).sort_values(
        ["rule", "arm", "seed"]).reset_index(drop=True)
    df_val = pd.DataFrame(val_rows).sort_values(
        ["rule", "arm", "seed"]).reset_index(drop=True)
    df_test = pd.DataFrame(test_rows).sort_values(
        ["rule", "arm", "seed"]).reset_index(drop=True)
    c.write_csv(df_epochs, out_dir / f"s6r2_partA_epochs_{spec.name}.csv")
    c.write_csv(df_val, out_dir / f"s6r2_partA_val_metrics_{spec.name}.csv")
    c.write_csv(df_test, out_dir / f"s6r2_partA_test_metrics_{spec.name}.csv")

    # ---- A-grid: select epoch from G(arm,seed) ∪ {e_R0(arm,seed)} ----
    grid_epochs_rows: List[Dict[str, object]] = []
    grid_test_rows: List[Dict[str, object]] = []

    def _select_grid(series_full: Dict[int, float], allowed: List[int]) -> int:
        allowed_set = set(allowed)
        sub = {e: v for e, v in series_full.items() if e in allowed_set}
        if not sub:
            raise AssertionError(
                "A-grid allowed epoch set is empty; cannot select "
                f"(allowed={sorted(allowed_set)})"
            )
        return c.select_epoch_argmax(sub)

    for arm in c.ARMS:
        for seed in c.SEEDS:
            agg_map = per_arm_seed_agg[(arm, seed)]
            # Build series per rule from full agg.
            sR0 = {ep: a.M for ep, a in agg_map.items()}
            sR1 = {ep: a.T for ep, a in agg_map.items()}
            sR2 = {ep: a.N for ep, a in agg_map.items()}
            sR3 = {ep: a.WD for ep, a in agg_map.items()}
            _, _, weights = c.select_rule_R4(spec, arm, seed)
            sR4 = {ep: weights[0] * a.D0 + weights[1] * a.D1 + weights[2] * a.D2
                   for ep, a in agg_map.items()}
            e0_full = c.select_epoch_argmax(sR0)
            allowed = sorted(set(c.GRID_EPOCHS) | {e0_full})

            gR0 = _select_grid(sR0, allowed)
            gR1 = _select_grid(sR1, allowed)
            gR2 = _select_grid(sR2, allowed)
            gR3 = _select_grid(sR3, allowed)
            gR4 = _select_grid(sR4, allowed)

            e_canon = c.canonical_selected_epoch(spec, arm, seed)
            grid_sel = {"R0": gR0, "R1": gR1, "R2": gR2, "R3": gR3, "R4": gR4}
            for R in RULES:
                ep_grid = grid_sel[R]
                grid_epochs_rows.append({
                    "rule": R, "arm": arm, "seed": int(seed),
                    "selected_epoch": int(ep_grid),
                    "objective_value": float(
                        {"R0": sR0[ep_grid], "R1": sR1[ep_grid],
                         "R2": sR2[ep_grid], "R3": sR3[ep_grid],
                         "R4": sR4[ep_grid]}[R]
                    ),
                    "canonical_match": int(ep_grid == e_canon),
                })
                # Test: every chosen epoch is in {common_epoch_*} ∪ {selected_epoch}
                # so test is always available.
                m, src, avail = c.aggregate_test_one_epoch(spec, arm, seed, ep_grid)
                if avail and m is not None:
                    grid_test_rows.append(_test_dict_to_row_dict(R, arm, seed, ep_grid, m, src, 1))
                else:
                    # Although A-grid guarantees every chosen epoch is in
                    # {selected, common_epoch_*} per spec §4.3, mark unavailable
                    # if the lookup returned None to surface an error.
                    raise AssertionError(
                        f"A-grid unexpectedly missing test for "
                        f"{spec.name}/{arm}/seed{seed} R={R} ep={ep_grid}"
                    )

    df_grid_epochs = pd.DataFrame(grid_epochs_rows).sort_values(
        ["rule", "arm", "seed"]).reset_index(drop=True)
    df_grid_test = pd.DataFrame(grid_test_rows).sort_values(
        ["rule", "arm", "seed"]).reset_index(drop=True)
    c.write_csv(df_grid_epochs, out_dir / f"s6r2_partA_grid_epochs_{spec.name}.csv")
    c.write_csv(df_grid_test, out_dir / f"s6r2_partA_grid_test_metrics_{spec.name}.csv")

    # ---- Gate summary ----
    val_full = df_val.set_index(["rule", "arm", "seed"])
    test_full = df_test.set_index(["rule", "arm", "seed"])
    grid_full = df_grid_test.set_index(["rule", "arm", "seed"])

    summary_rows: List[Dict[str, object]] = []
    for R in RULES:
        for metric in ("N", "T", "M"):
            # val layer
            deltas_val: List[float] = []
            deltas_test: List[float] = []
            deltas_grid: List[float] = []
            for seed in c.SEEDS:
                # The metric on DH - F at the chosen epoch (paired by seed).
                v_dh = float(val_full.loc[(R, "DH", seed), metric])
                v_f = float(val_full.loc[(R, "F", seed), metric])
                deltas_val.append(v_dh - v_f)

                t_dh_row = test_full.loc[(R, "DH", seed)]
                t_f_row = test_full.loc[(R, "F", seed)]
                if int(t_dh_row["test_available"]) == 1 and int(t_f_row["test_available"]) == 1:
                    deltas_test.append(float(t_dh_row[metric]) - float(t_f_row[metric]))

                g_dh_row = grid_full.loc[(R, "DH", seed)]
                g_f_row = grid_full.loc[(R, "F", seed)]
                deltas_grid.append(float(g_dh_row[metric]) - float(g_f_row[metric]))

            stats_val = c.gate_stats(deltas_val) if len(deltas_val) >= 1 else None
            stats_test = c.gate_stats(deltas_test) if len(deltas_test) >= 1 else {
                "mean": float("nan"), "sd": float("nan"),
                "per_seed": [], "n_pos": 0, "n_ge_003": 0,
                "n_ge_005": 0, "n_seeds": 0,
            }
            stats_grid = c.gate_stats(deltas_grid) if len(deltas_grid) >= 1 else None

            for layer_name, stats in (("val", stats_val),
                                      ("test", stats_test),
                                      ("test_grid", stats_grid)):
                if stats is None:
                    continue
                row = {
                    "rule": R, "metric": metric, "layer": layer_name,
                    "n_seeds": stats["n_seeds"],
                    "delta_mean": stats["mean"],
                    "delta_sd": stats["sd"],
                    "n_pos": stats["n_pos"],
                    "n_ge_003": stats["n_ge_003"],
                    "n_ge_005": stats["n_ge_005"],
                }
                # Per-seed values for the 4 paired deltas.
                for i, sd_seed in enumerate(c.SEEDS):
                    if i < len(stats["per_seed"]):
                        row[f"delta_seed{int(sd_seed)}"] = stats["per_seed"][i]
                    else:
                        row[f"delta_seed{int(sd_seed)}"] = float("nan")
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["rule", "metric", "layer"]).reset_index(drop=True)
    c.write_csv(df_summary, out_dir / f"s6r2_partA_gate_summary_{spec.name}.csv")

    print(f"[PartA] {spec.name}: epochs={len(df_epochs)} "
          f"val={len(df_val)} test={len(df_test)} "
          f"grid_epochs={len(df_grid_epochs)} grid_test={len(df_grid_test)} "
          f"gate_rows={len(df_summary)}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(c.SPEC.keys()), required=True)
    args = parser.parse_args(argv)
    run_for_dataset(c.SPEC[args.dataset])


if __name__ == "__main__":
    main()
