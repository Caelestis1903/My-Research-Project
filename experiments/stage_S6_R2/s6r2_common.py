"""s6r2_common.py

Shared utilities for the Stage S6-R2 retrospective robustness /
decomposition analysis. Import-only (no CLI side effects).

Responsibilities:
    - dataset spec registry (`kuairand`, `movielens`);
    - per-run path helpers (run_root, joint dir);
    - loader for metrics_epoch (val layer) and metrics_test (test layer);
    - aggregation of 6-cell validation rows into M/T/N/WD/WC/D0/D1/D2;
    - precomputed-column consistency assertions (1e-9);
    - per-rule selection with earliest-epoch tie-break (val-only);
    - per-epoch lookup of test metrics from saved checkpoints (7 candidates),
      with explicit NA semantics for the rest;
    - Δ / gate summary statistics.

Important design contract (from design doc §3, §4.3, §9):
    - Selection logic only ever receives a val-layer per-epoch series;
      test data is consulted solely for reporting `test_lookup`.
    - Test predictions exist only for {selected, final, common_epoch_*}
      (= 7 distinct epochs per (arm,seed)); all other epochs must be
      reported `NA` with `test_available=0`.
    - Pre-aggregated epoch columns (`validation_macro_auc`,
      `target_weighted_validation_auc`, ...) are *cross-checked* against
      the 6-cell recomputation but never used as the source for R0-R4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments"

# Spec ranges (from design doc §2.6, §3).
SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
ARMS: Tuple[str, ...] = ("F", "DH")
EPOCH_RANGE: Tuple[int, ...] = tuple(range(50))
GRID_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)  # common_epoch_* set
SAVED_TEST_NAMES: Tuple[str, ...] = (
    "selected", "final",
    "common_epoch_9", "common_epoch_19", "common_epoch_29",
    "common_epoch_39", "common_epoch_49",
)
NAME_TO_EPOCH: Dict[str, int] = {
    "common_epoch_9": 9,
    "common_epoch_19": 19,
    "common_epoch_29": 29,
    "common_epoch_39": 39,
    "common_epoch_49": 49,
    "final": 49,
}

# Self-test tolerances.
TOL_EXACT = 1e-9  # intra-pipeline consistency, identity asserts.
TOL_ANCHOR = 1e-4  # Part D anchor reproduction across CPU/GPU.
TOL_VAL_COUNTS = 1e-9  # validation_sample_count across epoch 0..49.

# Thresholds (spec §4 / §10). Both reused for plotting/grading.
TAU_PRAC = 0.003
TAU_STRONG = 0.005


# ---------------------------------------------------------------------------
# Dataset spec registry.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_root: Path
    target_domain: int
    setting: str  # canonical setting string used by load_stage_s1_bundle.


SPEC: Dict[str, DatasetSpec] = {
    "kuairand": DatasetSpec(
        name="kuairand",
        raw_root=EXP_DIR / "stage_S6_raw_results",
        target_domain=1,
        setting="natural",
    ),
    "movielens": DatasetSpec(
        name="movielens",
        raw_root=EXP_DIR / "stage_S6ML_raw_results",
        target_domain=2,
        setting="d2_large",
    ),
}


def parse_dataset(argv: Optional[Sequence[str]] = None) -> DatasetSpec:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=list(SPEC.keys()), required=True)
    args, _unknown = parser.parse_known_args(argv)
    return SPEC[args.dataset]


# ---------------------------------------------------------------------------
# Output directory.
# ---------------------------------------------------------------------------
OUT_ROOT: Path = EXP_DIR / "stage_S6_R2_raw_results"


def dataset_out_dir(spec: DatasetSpec) -> Path:
    return OUT_ROOT / spec.name


def ensure_out_dir(spec: DatasetSpec) -> Path:
    p = dataset_out_dir(spec)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Per-run filesystem access.
# ---------------------------------------------------------------------------
def run_joint_dir(spec: DatasetSpec, arm: str, seed: int) -> Path:
    return spec.raw_root / arm / f"seed{int(seed)}" / "joint"


def load_config(spec: DatasetSpec, arm: str, seed: int) -> Dict[str, object]:
    p = run_joint_dir(spec, arm, seed) / "config.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_parameter_manifest(spec: DatasetSpec, arm: str, seed: int) -> Dict[str, object]:
    p = run_joint_dir(spec, arm, seed) / "parameter_manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_state_dict(spec: DatasetSpec, arm: str, seed: int) -> Dict[str, "torch.Tensor"]:
    import torch  # local import to avoid torch overhead in scripts that don't need it.

    p = run_joint_dir(spec, arm, seed) / "selected_checkpoint.pth"
    sd = torch.load(p, map_location="cpu")
    if not isinstance(sd, Mapping):
        raise TypeError(f"{p}: expected state_dict mapping, got {type(sd).__name__}")
    return {str(k): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Cell layout / label helpers (mirror s6r_common.cell_label convention).
# ---------------------------------------------------------------------------
def cell_label(domain: int, task: int) -> str:
    return f"d{int(domain)}t{int(task)}"


def target_cells(spec: DatasetSpec) -> Tuple[Tuple[int, int], ...]:
    d = spec.target_domain
    return ((d, 0), (d, 1))


def non_target_cells(spec: DatasetSpec) -> Tuple[Tuple[int, int], ...]:
    return tuple((d, t) for d in range(3) for t in range(2) if d != spec.target_domain)


def non_target_domains(spec: DatasetSpec) -> Tuple[int, ...]:
    return tuple(d for d in range(3) if d != spec.target_domain)


# ---------------------------------------------------------------------------
# metrics_epoch loader (per epoch x 6 cells).
# ---------------------------------------------------------------------------
def load_epoch_table(spec: DatasetSpec, arm: str, seed: int) -> pd.DataFrame:
    """Load the long-format per-epoch x per-cell validation frame for one
    (dataset, arm, seed) joint run. Adds helper columns and re-derives cell
    from (domain, task) to enforce `cell == task*3 + domain`.
    """
    p = run_joint_dir(spec, arm, seed) / "metrics_epoch.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    # 300 rows per run; spec §2.2.
    if len(df) != 300:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: expected 300 epoch rows, got {len(df)}"
        )
    # Cast integer columns.
    for col in ("epoch", "domain", "task", "cell"):
        df[col] = df[col].astype(int)
    expected_cell = df["task"] * 3 + df["domain"]
    if not (df["cell"] == expected_cell).all():
        bad = df.loc[~(df["cell"] == expected_cell)].head()
        raise AssertionError(f"{spec.name}/{arm}/seed{seed}: cell != task*3+domain:\n{bad}")
    # Sort canonical order (epoch, then task-major cell).
    df = df.sort_values(["epoch", "cell"]).reset_index(drop=True)
    df["arm"] = arm
    df["seed"] = int(seed)
    df["dataset"] = spec.name
    df["cell_label"] = [
        cell_label(int(d), int(t)) for d, t in zip(df["domain"], df["task"])
    ]
    return df


def assert_val_counts_constant(spec: DatasetSpec, arm: str, seed: int) -> None:
    df = load_epoch_table(spec, arm, seed)
    counts = df.groupby("cell_label", sort=False)["validation_sample_count"].agg(
        lambda s: s.nunique()
    )
    if (counts > 1).any():
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: validation_sample_count is not constant "
            f"across epochs (cells={int((counts>1).sum())})"
        )


# ---------------------------------------------------------------------------
# Aggregation across 6 cells (always recompute; never trust precomputed cols).
# ---------------------------------------------------------------------------
@dataclass
class CellAgg:
    M: float        # macro (6 unweighted mean)
    T: float        # target mean (2 target cells)
    N: float        # non-target mean (4 non-target cells)
    WD: float       # worst-domain mean
    WC: float       # worst cell
    D0: float
    D1: float
    D2: float
    cell_aucs: Dict[str, float]  # cell_label -> AUC


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def aggregate_cells(
    spec: DatasetSpec,
    df_epoch: pd.DataFrame,  # exactly 6 rows (one per cell) at a single epoch
) -> CellAgg:
    if len(df_epoch) != 6:
        raise AssertionError(f"aggregate_cells expects 6 rows, got {len(df_epoch)}")
    by_cell = {row.cell_label: float(row.validation_auc)
               for row in df_epoch.itertuples(index=False)}
    by_domain: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    for row in df_epoch.itertuples(index=False):
        by_domain[int(row.domain)].append(float(row.validation_auc))
    cells_list = [by_cell[cell_label(d, t)] for d in range(3) for t in range(2)]
    target_d = spec.target_domain
    target_vals = [by_cell[cell_label(target_d, 0)],
                   by_cell[cell_label(target_d, 1)]]
    non_target_vals = [
        by_cell[cell_label(d, t)]
        for d in range(3) for t in range(2) if d != target_d
    ]
    domain_means = [_safe_mean(v) for v in (by_domain[0], by_domain[1], by_domain[2])]
    return CellAgg(
        M=float(np.mean(cells_list)),
        T=_safe_mean(target_vals),
        N=_safe_mean(non_target_vals),
        WD=float(np.nanmin(domain_means)),
        WC=float(np.nanmin(cells_list)),
        D0=domain_means[0], D1=domain_means[1], D2=domain_means[2],
        cell_aucs=by_cell,
    )


def aggregate_per_epoch(
    spec: DatasetSpec, arm: str, seed: int
) -> Dict[int, CellAgg]:
    """Return {epoch -> CellAgg} recomputed from the 6 per-cell AUCs."""
    df = load_epoch_table(spec, arm, seed)
    out: Dict[int, CellAgg] = {}
    for ep in EPOCH_RANGE:
        sub = df[df["epoch"] == ep]
        if len(sub) != 6:
            raise AssertionError(
                f"{spec.name}/{arm}/seed{seed}/epoch{ep}: expected 6 rows, got {len(sub)}"
            )
        out[ep] = aggregate_cells(spec, sub)
    return out


def assert_epoch_aggregations_match_precomputed(
    spec: DatasetSpec, arm: str, seed: int,
) -> None:
    """Cross-check recomputed M/WD/WC against the precomputed epoch-level
    columns in metrics_epoch.csv. Tolerance = TOL_EXACT (1e-9)."""
    df = load_epoch_table(spec, arm, seed)
    per_epoch = aggregate_per_epoch(spec, arm, seed)
    fail: List[str] = []
    for ep, agg in per_epoch.items():
        sub = df[df["epoch"] == ep]
        macro_pre = float(sub["validation_macro_auc"].iloc[0])
        wd_pre = float(sub["worst_domain_validation_auc"].iloc[0])
        wc_pre = float(sub["worst_cell_validation_auc"].iloc[0])
        if abs(macro_pre - agg.M) > TOL_EXACT:
            fail.append(f"epoch{ep} macro {macro_pre} vs {agg.M}")
        if abs(wd_pre - agg.WD) > TOL_EXACT:
            fail.append(f"epoch{ep} WD {wd_pre} vs {agg.WD}")
        if abs(wc_pre - agg.WC) > TOL_EXACT:
            fail.append(f"epoch{ep} WC {wc_pre} vs {agg.WC}")
    if fail:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: precomputed vs recomputed mismatch:\n  "
            + "\n  ".join(fail[:10])
            + ("\n  ..." if len(fail) > 10 else "")
        )


def assert_canonical_selected_epoch(spec: DatasetSpec, arm: str, seed: int,
                                    expected_epoch: int) -> None:
    """Selected row in metrics_epoch must agree with the explicit
    `selected_checkpoint=1` flag (which equals the canonical R0 selection
    by design — see stage_S1_run.py:157)."""
    df = load_epoch_table(spec, arm, seed)
    flag_rows = df[df["selected_checkpoint"] == 1]
    if len(flag_rows) != 6:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: selected_checkpoint=1 must cover exactly "
            f"6 rows (one per cell), got {len(flag_rows)}"
        )
    flagged_epoch = int(flag_rows["epoch"].iloc[0])
    if flagged_epoch != expected_epoch:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: selected_checkpoint flag epoch "
            f"{flagged_epoch} != R0 selection {expected_epoch}"
        )


# ---------------------------------------------------------------------------
# Selection logic (val only) + tie-break.
# ---------------------------------------------------------------------------
def select_epoch_argmax(values: Mapping[int, float]) -> int:
    """Return argmax with earliest-epoch tie-break. `values` keys are epochs
    in `EPOCH_RANGE`. Raises if empty."""
    if not values:
        raise ValueError("select_epoch_argmax: empty value map")
    best_val = max(values.values())
    best_epochs = [e for e, v in values.items() if v == best_val]
    return int(min(best_epochs))


def select_rule_R0(spec: DatasetSpec, arm: str, seed: int) -> Tuple[int, float]:
    agg = aggregate_per_epoch(spec, arm, seed)
    series = {ep: a.M for ep, a in agg.items()}
    e = select_epoch_argmax(series)
    return e, series[e]


def select_rule_R1(spec: DatasetSpec, arm: str, seed: int) -> Tuple[int, float]:
    agg = aggregate_per_epoch(spec, arm, seed)
    series = {ep: a.T for ep, a in agg.items()}
    e = select_epoch_argmax(series)
    return e, series[e]


def select_rule_R2(spec: DatasetSpec, arm: str, seed: int) -> Tuple[int, float]:
    agg = aggregate_per_epoch(spec, arm, seed)
    series = {ep: a.N for ep, a in agg.items()}
    e = select_epoch_argmax(series)
    return e, series[e]


def select_rule_R3(spec: DatasetSpec, arm: str, seed: int) -> Tuple[int, float]:
    agg = aggregate_per_epoch(spec, arm, seed)
    series = {ep: a.WD for ep, a in agg.items()}
    e = select_epoch_argmax(series)
    return e, series[e]


def select_rule_R4(
    spec: DatasetSpec, arm: str, seed: int
) -> Tuple[int, float, Dict[int, float]]:
    """Sample-weighted R4 (domain share weights). Returns (selected_epoch,
    objective_value_at_selection, domain_weights)."""
    # Spec §3.1: weights come from `validation_sample_count`, asserted constant
    # across epochs *before* epoch-0 values are taken.
    assert_val_counts_constant(spec, arm, seed)
    df = load_epoch_table(spec, arm, seed)
    # Use epoch 0's per-cell counts to derive domain weights; verified constant.
    e0 = df[df["epoch"] == 0]
    domain_w: Dict[int, float] = {}
    for d in range(3):
        cells_d = e0[e0["domain"] == d]
        domain_w[d] = float(cells_d["validation_sample_count"].sum())
    total = sum(domain_w.values())
    if total <= 0:
        raise AssertionError(f"{spec.name}/{arm}/seed{seed}: zero total val count")
    weights = {d: domain_w[d] / total for d in domain_w}
    agg = aggregate_per_epoch(spec, arm, seed)
    series: Dict[int, float] = {
        ep: weights[0] * a.D0 + weights[1] * a.D1 + weights[2] * a.D2
        for ep, a in agg.items()
    }
    e = select_epoch_argmax(series)
    return e, series[e], weights


# ---------------------------------------------------------------------------
# Test-layer lookup. Returns per-cell AUCs computed from
# predictions_test.npz (consistent with metrics_test.csv) when the requested
# epoch matches one of the 7 saved checkpoints; otherwise returns None
# (= explicit NA). final and common_epoch_49 are both tested epoch 49; the
# caller is expected to record both and to assert consistency.
# ---------------------------------------------------------------------------
_TEST_NPZ_NAME_MAP = {
    "selected": "pred_selected",
    "final": "pred_final",
    "common_epoch_9": "pred_common_epoch_9",
    "common_epoch_19": "pred_common_epoch_19",
    "common_epoch_29": "pred_common_epoch_29",
    "common_epoch_39": "pred_common_epoch_39",
    "common_epoch_49": "pred_common_epoch_49",
}


def _safe_auc_np(y: np.ndarray, s: np.ndarray) -> float:
    """Local minimal safe_auc (mirrors stage_S1_metrics.safe_auc)."""
    if y.size == 0 or not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("safe_auc_npz: non-finite inputs")
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    left = 0
    while left < len(s):
        right = left + 1
        while right < len(s) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2.0
        left = right
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _per_cell_auc_from_npz(npz: np.ndarray, key: str) -> Dict[str, Dict[str, float]]:
    """Return {cell_label: {auc, sample_count, positive_rate}} for one
    prediction array. Labels are taken from the saved npz `labels` (N, 2).
    Domains from `physical_domain` (N,)."""
    pred = np.asarray(npz[key], dtype=np.float64)  # (N, 2)
    labels = np.asarray(npz["labels"], dtype=np.float64)
    domains = np.asarray(npz["physical_domain"], dtype=np.int64)
    out: Dict[str, Dict[str, float]] = {}
    for d in range(3):
        for t in range(2):
            mask = domains == d
            y = labels[mask, t]
            s = pred[mask, t]
            n = int(y.size)
            pos = int((y == 1).sum())
            rate = float(y.mean()) if n else float("nan")
            if n == 0:
                auc = float("nan")
            else:
                auc = _safe_auc_np(y, s) if pos and pos < n else float("nan")
            out[cell_label(d, t)] = {
                "auc": float(auc),
                "sample_count": n,
                "positive_count": pos,
                "positive_rate": rate,
            }
    return out


def load_test_checkpoint(
    spec: DatasetSpec, arm: str, seed: int, checkpoint_name: str
) -> Dict[str, Dict[str, float]]:
    """Read predictions_test.npz and compute per-cell AUC for the given
    checkpoint name. `checkpoint_name` must be in `SAVED_TEST_NAMES`."""
    if checkpoint_name not in SAVED_TEST_NAMES:
        raise ValueError(f"load_test_checkpoint: unknown name {checkpoint_name}")
    npz_key = _TEST_NPZ_NAME_MAP[checkpoint_name]
    p = run_joint_dir(spec, arm, seed) / "predictions_test.npz"
    npz = np.load(p)
    for k in (npz_key, "labels", "physical_domain"):
        if k not in npz.files:
            raise AssertionError(f"{p}: missing npz key {k}")
    return _per_cell_auc_from_npz(npz, npz_key)


def assert_test_checkpoint_matches_metrics_csv(
    spec: DatasetSpec, arm: str, seed: int, checkpoint_name: str,
) -> None:
    """Compare npz-derived per-cell AUC against the saved metrics_test.csv
    row for the same checkpoint. Tolerance = TOL_EXACT (1e-9)."""
    predicted = load_test_checkpoint(spec, arm, seed, checkpoint_name)
    p = run_joint_dir(spec, arm, seed) / "metrics_test.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    sub = df[df["checkpoint"] == checkpoint_name]
    if len(sub) != 6:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}/{checkpoint_name}: expected 6 rows in "
            f"metrics_test.csv, got {len(sub)}"
        )
    failures: List[str] = []
    for _, row in sub.iterrows():
        label = cell_label(int(row["domain"]), int(row["task"]))
        ref = float(row["auc"])
        got = predicted[label]["auc"]
        if not (ref == ref):  # NaN guard
            continue
        if abs(ref - got) > TOL_EXACT:
            failures.append(f"{label} csv={ref} npz={got}")
    if failures:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}/{checkpoint_name}: "
            f"metrics_test.csv vs npz AUC mismatch:\n  " + "\n  ".join(failures)
        )


def test_lookup_for_epoch(
    spec: DatasetSpec, arm: str, seed: int, epoch: int
) -> Optional[Dict[str, Dict[str, float]]]:
    """Return the test-layer per-cell metrics for `epoch` if it corresponds
    to a `common_epoch_*` grid checkpoint; otherwise None (= NA).
    For the canonical selected epoch, use `test_metrics_for_epoch` which
    resolves to `selected` automatically."""
    if epoch in GRID_EPOCHS:
        return load_test_checkpoint(spec, arm, seed, f"common_epoch_{epoch}")
    return None


def test_metrics_for_epoch(
    spec: DatasetSpec, arm: str, seed: int, epoch: int
) -> Tuple[Optional[Dict[str, Dict[str, float]]], str, int]:
    """Returns (metrics_or_None, source_name, test_available)."""
    if epoch == _load_canonical_selected_epoch(spec, arm, seed):
        m = load_test_checkpoint(spec, arm, seed, "selected")
        return m, "selected", 1
    if epoch in GRID_EPOCHS:
        name = f"common_epoch_{epoch}"
        m = load_test_checkpoint(spec, arm, seed, name)
        return m, name, 1
    return None, "NA", 0


def _load_canonical_selected_epoch(spec: DatasetSpec, arm: str, seed: int) -> int:
    df = load_epoch_table(spec, arm, seed)
    rows = df[df["selected_checkpoint"] == 1]
    if len(rows) != 6:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: no canonical selected epoch flag"
        )
    return int(rows["epoch"].iloc[0])


def canonical_selected_epoch(spec: DatasetSpec, arm: str, seed: int) -> int:
    return _load_canonical_selected_epoch(spec, arm, seed)


# ---------------------------------------------------------------------------
# Aggregator at one (layer, epoch) returning the 10 metrics required by Part B.
# Test layer only valid when test_metrics returned a non-None dict.
# ---------------------------------------------------------------------------
TEST_METRIC_NAMES: Tuple[str, ...] = (
    "M", "T", "N", "D0", "D2",
    "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1",
)


def aggregate_test_one_epoch(
    spec: DatasetSpec, arm: str, seed: int, epoch: int
) -> Tuple[Optional[Dict[str, float]], str, int]:
    """Return (metric_dict or None, checkpoint_name, test_available).
    metric_dict keys = TEST_METRIC_NAMES (M, T, N, D0, D2, plus 6 cell AUCs)."""
    m_dict, src, avail = test_metrics_for_epoch(spec, arm, seed, epoch)
    if not avail or m_dict is None:
        return None, "NA", 0
    target_d = spec.target_domain
    target_vals = [
        m_dict[cell_label(target_d, 0)]["auc"],
        m_dict[cell_label(target_d, 1)]["auc"],
    ]
    non_target_vals = [
        m_dict[cell_label(d, t)]["auc"]
        for d in range(3) for t in range(2) if d != target_d
    ]
    cell_aucs = {cell_label(d, t): m_dict[cell_label(d, t)]["auc"]
                 for d in range(3) for t in range(2)}
    out = {
        "M": float(np.mean(list(cell_aucs.values()))),
        "T": float(np.mean(target_vals)),
        "N": float(np.mean(non_target_vals)),
        "D0": float(np.mean([m_dict[cell_label(0, 0)]["auc"],
                              m_dict[cell_label(0, 1)]["auc"]])),
        "D1": float(np.mean([m_dict[cell_label(1, 0)]["auc"],
                              m_dict[cell_label(1, 1)]["auc"]])),
        "D2": float(np.mean([m_dict[cell_label(2, 0)]["auc"],
                              m_dict[cell_label(2, 1)]["auc"]])),
    }
    out["WD"] = float(np.nanmin([out["D0"], out["D1"], out["D2"]]))
    out["WC"] = float(np.nanmin(list(cell_aucs.values())))
    for c, v in cell_aucs.items():
        out[c] = v
    return out, src, 1


# ---------------------------------------------------------------------------
# Gate statistics on 4-seed paired deltas.
# ---------------------------------------------------------------------------
def gate_stats(deltas: Sequence[float]) -> Dict[str, object]:
    """Paired-bydesign 4-seed delta summary. Returns mean, sd(ddof=1),
    per-seed list, positive_count (>0), n_ge_003 (|d|>=0.003), n_ge_005
    (|d|>=0.005), n_seeds.
    All keys are JSON-serialisable."""
    arr = np.asarray(list(deltas), dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("gate_stats: empty deltas")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    n_pos = int(np.sum(arr > 0))
    n_ge_003 = int(np.sum(np.abs(arr) >= TAU_PRAC))
    n_ge_005 = int(np.sum(np.abs(arr) >= TAU_STRONG))
    return {
        "mean": mean,
        "sd": sd,
        "per_seed": [float(v) for v in arr],
        "n_pos": n_pos,
        "n_ge_003": n_ge_003,
        "n_ge_005": n_ge_005,
        "n_seeds": n,
    }


# ---------------------------------------------------------------------------
# CSV writer (UTF-8 BOM kept for Excel consistency with project convention).
# ---------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Smoke check.
# ---------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> None:
    spec = parse_dataset(argv)
    df = load_epoch_table(spec, "F", SEEDS[0])
    agg = aggregate_per_epoch(spec, "F", SEEDS[0])
    canonical = canonical_selected_epoch(spec, "F", SEEDS[0])
    print(
        f"{spec.name}: cells={len(df)//50} x 50 epochs, "
        f"canonical F seed{SEEDS[0]} epoch={canonical}, "
        f"M@canonical={agg[canonical].M:.6f}"
    )


if __name__ == "__main__":
    _main()
