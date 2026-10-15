"""s6r_probe_aux.py

Generate table 12 — descriptive-only summary of probe.csv + gradient_step.csv
per (arm, seed, epoch). Missing columns are left blank.

Per spec, these rows are descriptive and DO NOT enter the verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from s6r_common import (
    DatasetSpec,
    SEEDS,
    parse_dataset,
    table_path,
    write_csv,
)


PROBE_COLS = [
    ("g_target_norm_mean", "g_target_norm"),
    ("g_non_norm_mean", "g_non_norm"),
    ("gradient_cosine_mean", "gradient_cosine"),
    ("delta_target_norm_mean", "delta_target_norm"),
    ("private_delta_norm_mean", "private_delta_norm"),
    ("applied_cosine_target_mean", "applied_cosine_target"),
]


def _load_and_aggregate(
    spec: DatasetSpec, arm: str, seed: int, kind_filter: str = "main"
) -> pd.DataFrame:
    """Load probe.csv + gradient_step.csv, filter to `kind`, and aggregate
    over steps within (epoch). Returns columns per spec.
    """
    base = Path(spec.run_root) / arm / f"seed{int(seed)}" / "joint"
    out_rows = []
    for fname in ("probe.csv", "gradient_step.csv"):
        p = base / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "kind" in df.columns:
            df = df[df["kind"].astype(str) == kind_filter]
        if len(df) == 0:
            continue
        df["epoch"] = df["epoch"].astype(int)
        for epoch, sub in df.groupby("epoch"):
            agg = {"arm": arm, "seed": int(seed), "epoch": int(epoch)}
            for out_name, src in PROBE_COLS:
                if src in sub.columns:
                    agg[out_name] = float(sub[src].astype(float).mean())
                else:
                    agg[out_name] = float("nan")
            agg["_source"] = fname
            out_rows.append(agg)
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def build_table12(spec: DatasetSpec) -> pd.DataFrame:
    rows: List[Dict] = []
    for arm in ("F", "DH"):
        for seed in SEEDS:
            agg = _load_and_aggregate(spec, arm, int(seed), kind_filter="main")
            if len(agg) == 0:
                # still emit 50 epochs with NaN
                for epoch in range(0, 50):
                    row = {"dataset": spec.name, "arm": arm, "seed": int(seed), "epoch": epoch}
                    for out_name, _src in PROBE_COLS:
                        row[out_name] = float("nan")
                    rows.append(row)
                continue
            # Average over sources within same (arm,seed,epoch)
            grp_cols = ["arm", "seed", "epoch"]
            agg_mean = agg.groupby(grp_cols)[[c for c, _ in PROBE_COLS]].mean().reset_index()
            for _, r in agg_mean.iterrows():
                row = {"dataset": spec.name, "arm": r["arm"], "seed": int(r["seed"]), "epoch": int(r["epoch"])}
                for out_name, _src in PROBE_COLS:
                    row[out_name] = float(r[out_name]) if not pd.isna(r[out_name]) else float("nan")
                rows.append(row)
            # epochs with no source: pad with NaN
            seen = set(int(e) for e in agg_mean["epoch"].unique())
            for epoch in range(0, 50):
                if epoch in seen:
                    continue
                row = {"dataset": spec.name, "arm": arm, "seed": int(seed), "epoch": int(epoch)}
                for out_name, _src in PROBE_COLS:
                    row[out_name] = float("nan")
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(["arm", "seed", "epoch"]).reset_index(drop=True)
    return out


def main() -> None:
    spec = parse_dataset()
    t12 = build_table12(spec)
    write_csv(t12, table_path(spec, "s6r_probe_aux"))
    print(f"{spec.name}: table 12 written ({len(t12)})")


if __name__ == "__main__":
    main()