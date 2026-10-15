"""s6r2_partB_decomposition.py

Part B: algebraic decomposition of the canonical selected gap.

For each (dataset, seed) we compute, for each of the 10 metrics
(M, T, N, D0, D2, plus the 6 single-cell AUCs) and for both layers
(val, test):

    ΔSel     = M_DH(e_DH) - M_F(e_F)
    F-anchor: PolicyAtF  = M_DH(e_F) - M_F(e_F)
              TimingAlongDH = M_DH(e_DH) - M_DH(e_F)
    DH-anchor: PolicyAtDH = M_DH(e_DH) - M_F(e_DH)
              TimingAlongF  = M_F(e_DH) - M_F(e_F)

Identity |ΔSel - (Policy + Timing)| < 1e-9 is asserted for **val** layer
(all epochs available). For **test** layer both e_F and e_DH are required
to be in the saved checkpoint set; per spec §2.6 none of the 16 (arm,seed)
combinations has canonical selected on the grid {9,19,29,39,49}. Therefore
the test-layer decomposition is annotated
`Existing artifacts are insufficient for test-layer cross-epoch decomposition`
and we instead emit a same-epoch grid trajectory of Δ(e)=X_DH(e)−X_F(e)
for e ∈ {9,19,29,39,49}.

Outputs (per dataset, into `experiments/stage_S6_R2_raw_results/{dataset}/`):

    s6r2_partB_decomposition_{dataset}.csv
        seed x metric x layer x anchor -> ΔSel, Policy, Timing, residual, availability
    s6r2_partB_grid_trajectory_{dataset}.csv
        seed x metric x grid_epoch -> Δ_DH_minus_F at this epoch (test)

Run via:
    python -m experiments.stage_S6_R2.s6r2_partB_decomposition --dataset kuairand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from stage_S6_R2 import s6r2_common as c  # noqa: E402


# The 10 metrics required for decomposition.
DECOMP_METRICS: Tuple[str, ...] = (
    "M", "T", "N", "D0", "D2",
    "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1",
)


def _val_metric(spec: c.DatasetSpec, arm: str, seed: int, ep: int, metric: str) -> float:
    agg = c.aggregate_per_epoch(spec, arm, seed)[ep]
    if metric in ("M", "T", "N", "D0", "D1", "D2", "WD", "WC"):
        return float(getattr(agg, metric))
    # cell_label form
    return float(agg.cell_aucs[metric])


def _test_metric(
    spec: c.DatasetSpec, arm: str, seed: int, ep: int, metric: str
) -> Tuple[Optional[float], str, int]:
    """Return (value or None, source, available)."""
    m, src, avail = c.aggregate_test_one_epoch(spec, arm, seed, ep)
    if not avail or m is None:
        return None, src, 0
    if metric in ("M", "T", "N", "D0", "D2"):
        return float(m[metric]), src, 1
    # cell_label form
    return float(m[metric]), src, 1


def _row(
    seed: int, metric: str, layer: str, anchor: str,
    delta_sel: Optional[float],
    policy: Optional[float],
    timing: Optional[float],
    availability: str,
    residual: Optional[float],
) -> Dict[str, object]:
    return {
        "seed": int(seed),
        "metric": metric,
        "layer": layer,
        "anchor": anchor,
        "delta_sel": delta_sel,
        "policy": policy,
        "timing": timing,
        "residual": residual,
        "availability": availability,
    }


def _decompose_val(spec: c.DatasetSpec, seed: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    eF = c.canonical_selected_epoch(spec, "F", seed)
    eD = c.canonical_selected_epoch(spec, "DH", seed)
    for metric in DECOMP_METRICS:
        mF_eF = _val_metric(spec, "F", seed, eF, metric)
        mD_eD = _val_metric(spec, "DH", seed, eD, metric)
        mD_eF = _val_metric(spec, "DH", seed, eF, metric)
        mF_eD = _val_metric(spec, "F", seed, eD, metric)
        delta_sel = mD_eD - mF_eF

        # F-anchor
        pol_F = mD_eF - mF_eF
        tim_D = mD_eD - mD_eF
        res_F = delta_sel - (pol_F + tim_D)
        assert abs(res_F) < 1e-9, (
            f"{spec.name}/seed{seed}/{metric}/F-anchor: residual {res_F}"
        )
        rows.append(_row(seed, metric, "val", "F", delta_sel, pol_F, tim_D,
                          "available", res_F))

        # DH-anchor
        pol_D = mD_eD - mF_eD
        tim_F = mF_eD - mF_eF
        res_D = delta_sel - (pol_D + tim_F)
        assert abs(res_D) < 1e-9, (
            f"{spec.name}/seed{seed}/{metric}/DH-anchor: residual {res_D}"
        )
        rows.append(_row(seed, metric, "val", "DH", delta_sel, pol_D, tim_F,
                          "available", res_D))
    return rows


def _decompose_test(spec: c.DatasetSpec, seed: int) -> List[Dict[str, object]]:
    """Cross-epoch test decomposition requires F and DH at *both* canonical
    epochs to be on the saved checkpoint grid. Per spec §2.6, none of the
    8 (arm,seed) combinations sits on {9,19,29,39,49}; we therefore emit
    availability='insufficient_cross_epoch' rows for every (metric, anchor)
    and let the per-endpoint ΔSel values be derived from the test-side of
    Part A's gate summary (or the grid_trajectory output below).

    Defensive branches remain in case a future run produces a coincidence.
    """
    rows: List[Dict[str, object]] = []
    eF = c.canonical_selected_epoch(spec, "F", seed)
    eD = c.canonical_selected_epoch(spec, "DH", seed)
    # Test availability at each requested endpoint:
    avail_F_eF = _test_metric(spec, "F", seed, eF, "M")[2] == 1
    avail_D_eD = _test_metric(spec, "DH", seed, eD, "M")[2] == 1
    avail_D_eF = _test_metric(spec, "DH", seed, eF, "M")[2] == 1
    avail_F_eD = _test_metric(spec, "F", seed, eD, "M")[2] == 1
    # All four endpoint lookups are required.
    cross_full = avail_F_eF and avail_D_eD and avail_D_eF and avail_F_eD
    avail_endpoint = avail_F_eF and avail_D_eD and not (avail_D_eF and avail_F_eD)
    for metric in DECOMP_METRICS:
        for anchor in ("F", "DH"):
            if cross_full:
                # Defensive: write the full decomposition if every endpoint
                # is reachable.
                mF_eF, _, _ = _test_metric(spec, "F", seed, eF, metric)
                mD_eD, _, _ = _test_metric(spec, "DH", seed, eD, metric)
                mD_eF, _, _ = _test_metric(spec, "DH", seed, eF, metric)
                mF_eD, _, _ = _test_metric(spec, "F", seed, eD, metric)
                if anchor == "F":
                    pol = mD_eF - mF_eF
                    tim = mD_eD - mD_eF
                else:
                    pol = mD_eD - mF_eD
                    tim = mF_eD - mF_eF
                delta_sel = mD_eD - mF_eF
                res = delta_sel - (pol + tim)
                assert abs(res) < 1e-9, (
                    f"{spec.name}/seed{seed}/{metric}/{anchor}-anchor(test): "
                    f"residual {res}"
                )
                rows.append(_row(seed, metric, "test", anchor,
                                  delta_sel, pol, tim, "available", res))
            elif avail_endpoint:
                # Both selected endpoints are available but only at their
                # respective selected test rows; cross-epoch terms are NA.
                mF_eF, _, _ = _test_metric(spec, "F", seed, eF, metric)
                mD_eD, _, _ = _test_metric(spec, "DH", seed, eD, metric)
                delta_sel = mD_eD - mF_eF
                rows.append(_row(seed, metric, "test", anchor,
                                  delta_sel, None, None,
                                  "endpoints_only_no_cross", None))
            else:
                rows.append(_row(seed, metric, "test", anchor,
                                  None, None, None,
                                  "insufficient_cross_epoch", None))
    return rows


def _grid_trajectory(spec: c.DatasetSpec) -> pd.DataFrame:
    """Same-epoch Δ(e) = X_DH(e) - X_F(e) at common epochs {9,19,29,39,49}
    on the TEST layer. Useful proxy for `PolicyAtF` since e_F is unlikely
    to coincide exactly with a grid point. Caches per-cell test metrics
    so each (arm,seed,ep) is computed once."""
    rows: List[Dict[str, object]] = []
    # Cache: (arm, seed, ep) -> 11-tuple test dict (or None).
    cache: Dict[Tuple[str, int, int], Optional[Dict[str, float]]] = {}

    def get(arm: str, seed: int, ep: int) -> Optional[Dict[str, float]]:
        key = (arm, seed, ep)
        if key not in cache:
            m, _, _ = c.aggregate_test_one_epoch(spec, arm, seed, ep)
            cache[key] = m
        return cache[key]

    for seed in c.SEEDS:
        for ep in c.GRID_EPOCHS:
            for metric in DECOMP_METRICS:
                mF = get("F", seed, ep)
                mD = get("DH", seed, ep)
                if mF is None or mD is None:
                    rows.append({
                        "seed": int(seed), "epoch": int(ep),
                        "metric": metric,
                        "F": float("nan"), "DH": float("nan"),
                        "delta_DH_minus_F": float("nan"),
                        "availability": "missing_endpoint",
                    })
                    continue
                f_v = float(mF[metric])
                d_v = float(mD[metric])
                rows.append({
                    "seed": int(seed), "epoch": int(ep),
                    "metric": metric,
                    "F": f_v, "DH": d_v,
                    "delta_DH_minus_F": d_v - f_v,
                    "availability": "available",
                })
    return pd.DataFrame(rows)


def run_for_dataset(spec: c.DatasetSpec) -> None:
    out_dir = c.ensure_out_dir(spec)
    print(f"[PartB] dataset={spec.name} -> {out_dir}")

    all_rows: List[Dict[str, object]] = []
    for seed in c.SEEDS:
        all_rows.extend(_decompose_val(spec, seed))
        all_rows.extend(_decompose_test(spec, seed))

    df = pd.DataFrame(all_rows).sort_values(
        ["seed", "metric", "layer", "anchor"]).reset_index(drop=True)
    c.write_csv(df, out_dir / f"s6r2_partB_decomposition_{spec.name}.csv")

    # Grid trajectory
    df_grid = _grid_trajectory(spec)
    c.write_csv(df_grid, out_dir / f"s6r2_partB_grid_trajectory_{spec.name}.csv")

    # Annotate the run-level summary of why test-layer decomposition is blocked.
    test_df = df[df["layer"] == "test"]
    n_cross_full = int((test_df["availability"] == "available").sum())
    n_endpoints = int((test_df["availability"] == "endpoints_only_no_cross").sum())
    n_insufficient = int(
        (test_df["availability"] == "insufficient_cross_epoch").sum()
    )
    print(f"[PartB] {spec.name}: val rows={int((df['layer']=='val').sum())}, "
          f"test rows={len(test_df)} "
          f"(cross_full={n_cross_full}, endpoints_only={n_endpoints}, "
          f"insufficient={n_insufficient})")
    # Spec §3.3: canonical epochs are NOT on the {9,19,29,39,49} grid.
    # A small `cross_full` count can only arise when two (arm,seed) pairs
    # happen to share the same canonical epoch (e.g. ML seed 2024 both =8),
    # which makes the cross-epoch terms degenerate (0) but the identity
    # still holds. Surface this so the main agent is aware.
    if n_cross_full > 0:
        print(f"[PartB] NOTE: {n_cross_full} test-layer decomposition rows are "
              "non-trivially available (cross-epoch terms computed). "
              "Per spec §2.6 all canonical epochs are off-grid; these "
              "rows exist only when two arms share an epoch by chance.")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(c.SPEC.keys()), required=True)
    args = parser.parse_args(argv)
    run_for_dataset(c.SPEC[args.dataset])


if __name__ == "__main__":
    main()
