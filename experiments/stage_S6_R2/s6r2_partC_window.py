"""s6r2_partC_window.py

Part C: selection-window robustness (spec §3.4).

For each (dataset, arm, seed), inspect the W = {e*−2, ..., e*+2} ∩ [0,49]
window around the canonical selected epoch e*. Compute the val layer
macro AUC drops at ±1 and ±2 offsets and classify the local optimum as:
    broad plateau      := max_drop < 0.001
    narrow spike       := drop_1 >= 0.002 AND drop_2 >= 0.004
    moderate           := otherwise

Test layer is reported for any window epoch that lands on a saved
checkpoint (and for canonical selected itself, marked `selected`).

Outputs (per dataset):

    s6r2_partC_window_{dataset}.csv
        arm x seed -> e*, V(e*), 5 val points {e*−2,...,e*+2},
        drop_1 (mean of V(e*)-V(e*±1) over available side),
        drop_2 (mean over available ±2 side),
        max_drop, classification, edge flags, test cells for window.

Run via:
    python -m experiments.stage_S6_R2.s6r2_partC_window --dataset kuairand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402


TAU_BROAD = 0.001
TAU_NARROW_DROP1 = 0.002
TAU_NARROW_DROP2 = 0.004


def _window(e_star: int) -> List[Tuple[int, str]]:
    """Return [(epoch, tag)] for {e*-2..e*+2} ∩ [0,49] with tag indicating
    offset and edge flags."""
    items: List[Tuple[int, str]] = []
    for offset in (-2, -1, 0, 1, 2):
        ep = e_star + offset
        if 0 <= ep <= 49:
            tag = f"d{offset:+d}"
            items.append((ep, tag))
    return items


def _safe_mean_drops(per_epoch: Dict[int, float], e_star: int) -> Tuple[float, float, float]:
    """Compute drop_1 and drop_2 (mean over available side of the offset),
    and max_drop = max_{e∈W, e≠e*} (V(e*) - V(e))."""
    V_star = float(per_epoch[e_star])
    drops_1: List[float] = []
    drops_2: List[float] = []
    other_drops: List[float] = []
    for ep, v in per_epoch.items():
        if ep == e_star:
            continue
        d = V_star - v
        other_drops.append(d)
        if ep == e_star - 1 or ep == e_star + 1:
            drops_1.append(d)
        elif ep == e_star - 2 or ep == e_star + 2:
            drops_2.append(d)
    drop_1 = float(np.mean(drops_1)) if drops_1 else float("nan")
    drop_2 = float(np.mean(drops_2)) if drops_2 else float("nan")
    max_drop = float(max(other_drops)) if other_drops else 0.0
    return drop_1, drop_2, max_drop


def _classify(max_drop: float, drop_1: float, drop_2: float) -> str:
    """Apply §3.4 classification rules."""
    if max_drop < TAU_BROAD:
        return "broad_plateau"
    has1 = not np.isnan(drop_1)
    has2 = not np.isnan(drop_2)
    if (
        has1 and has2
        and drop_1 >= TAU_NARROW_DROP1
        and drop_2 >= TAU_NARROW_DROP2
    ):
        return "narrow_spike"
    return "moderate"


def run_for_dataset(spec: c.DatasetSpec) -> None:
    out_dir = c.ensure_out_dir(spec)
    print(f"[PartC] dataset={spec.name}")

    rows: List[Dict[str, object]] = []
    for arm in c.ARMS:
        for seed in c.SEEDS:
            agg_map = c.aggregate_per_epoch(spec, arm, seed)
            e_star = c.canonical_selected_epoch(spec, arm, seed)
            V_star = float(agg_map[e_star].M)
            window = _window(e_star)
            win_map = {ep: float(agg_map[ep].M) for ep, _ in window}
            drop_1, drop_2, max_drop = _safe_mean_drops(win_map, e_star)
            klass = _classify(max_drop, drop_1, drop_2)
            row: Dict[str, object] = {
                "arm": arm, "seed": int(seed),
                "canonical_epoch": int(e_star),
                "val_M_at_canonical": V_star,
            }
            # Add val_M for each window epoch, including edge flags.
            edge_clip = int(e_star < 2 or e_star > 47)
            row["window_edge_clipped"] = int(edge_clip)
            for ep, tag in window:
                row[f"val_M_{tag}"] = float(win_map[ep])
            row["drop_1"] = drop_1
            row["drop_2"] = drop_2
            row["max_drop"] = max_drop
            row["classification"] = klass
            # Test lookup for window points that are saved.
            row["test_available_count"] = 0
            test_cells: Dict[str, object] = {}
            for ep, tag in window:
                m, src, avail = c.aggregate_test_one_epoch(spec, arm, seed, ep)
                if avail and m is not None:
                    row["test_available_count"] = int(row["test_available_count"]) + 1
                    test_cells[f"test_{tag}_M"] = float(m["M"])
                    test_cells[f"test_{tag}_N"] = float(m["N"])
                    test_cells[f"test_{tag}_T"] = float(m["T"])
                    test_cells[f"test_{tag}_source"] = src
                    row[f"test_{tag}_M"] = float(m["M"])
                    row[f"test_{tag}_N"] = float(m["N"])
                    row[f"test_{tag}_T"] = float(m["T"])
                    row[f"test_{tag}_source"] = src
                else:
                    row[f"test_{tag}_M"] = float("nan")
                    row[f"test_{tag}_N"] = float("nan")
                    row[f"test_{tag}_T"] = float("nan")
                    row[f"test_{tag}_source"] = "NA"
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    c.write_csv(df, out_dir / f"s6r2_partC_window_{spec.name}.csv")
    # Print classification summary.
    cls_counts = df["classification"].value_counts().to_dict()
    print(f"[PartC] {spec.name}: rows={len(df)} classifications={cls_counts}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(c.SPEC.keys()), required=True)
    args = parser.parse_args(argv)
    run_for_dataset(c.SPEC[args.dataset])


if __name__ == "__main__":
    main()
