"""s6r_common.py

Public utilities for the Stage S6 reanalysis. Loads spec, applies assertions,
computes per-seed deltas, summary statistics, and the pattern/verdict
classifiers consumed by every other s6r_*.py module.

All paths are relative to the repository root (D:\\CODE\\phase0-3). The script
is library-only: no CSV outputs are written here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants from the Algorithm Specification.
# ---------------------------------------------------------------------------
TAU_PRAC = 0.003
TAU_STRONG = 0.005
EPS = 1e-3  # used in the spec as the literal 0.001 (kept as EPS per spec)

SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
ARMS: Tuple[str, ...] = ("F", "T", "HT", "DH")
E_COMMON: Tuple[int, ...] = (9, 19, 29, 39, 49)
CHECKPOINT_TO_EPOCH: Dict[str, int] = {
    "common_epoch_9": 9,
    "common_epoch_19": 19,
    "common_epoch_29": 29,
    "common_epoch_39": 39,
    "common_epoch_49": 49,
}
COMMON_CHECKPOINTS: Tuple[str, ...] = tuple(CHECKPOINT_TO_EPOCH.keys())

# self-test tolerances (spec §9)
TOL_MEAN = 5e-4
TOL_EXACT = 1e-9
TOL_GENERIC = 1e-6


# ---------------------------------------------------------------------------
# Spec registry. target_domain appears once per dataset; downstream code MUST
# reference spec.target_domain (never a literal).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_csv: str
    run_root: str
    target_domain: int
    out_prefix: str
    formal_run_key_prefix: str
    assert_n_rows: int = 672
    assert_n_cells: int = 6
    assert_epochs_per_run: int = 300


DATASETS: Dict[str, DatasetSpec] = {
    "kuairand": DatasetSpec(
        name="kuairand",
        raw_csv="experiments/stage_S6_outputs/stage_S6_raw_results.csv",
        run_root="experiments/stage_S6_raw_results",
        target_domain=1,
        out_prefix="stage_S6",
        formal_run_key_prefix="s6|sample|",
    ),
    "movielens": DatasetSpec(
        name="movielens",
        raw_csv="experiments/stage_S6ML_outputs/stage_S6ML_raw_results.csv",
        run_root="experiments/stage_S6ML_raw_results",
        target_domain=2,
        out_prefix="stage_S6ML",
        formal_run_key_prefix="s6ml|sample|",
    ),
}


# ---------------------------------------------------------------------------
# Tables directory & helpers
# ---------------------------------------------------------------------------
TABLES_DIR = Path("experiments/stage_S6_reanalysis/tables")


def table_path(spec: DatasetSpec, name: str) -> Path:
    return TABLES_DIR / f"{spec.out_prefix}_{name}.csv"


# ---------------------------------------------------------------------------
# CLI plumbing shared by every script.
# ---------------------------------------------------------------------------
def parse_dataset(argv: Optional[Sequence[str]] = None) -> DatasetSpec:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    args, _unknown = parser.parse_known_args(argv)
    return DATASETS[args.dataset]


# ---------------------------------------------------------------------------
# Cell-label utilities (cell = task*3 + domain, per spec; always rebuilt from
# (domain,task), never read from the cell column).
# ---------------------------------------------------------------------------
def cell_label(domain: int, task: int) -> str:
    return f"d{domain}t{task}"


def assert_cell_invariant(df: pd.DataFrame) -> None:
    expected = df["task"].astype(int) * 3 + df["domain"].astype(int)
    diff_mask = ~(df["cell"].astype(int) == expected)
    if diff_mask.any():
        bad = df.loc[diff_mask, ["domain", "task", "cell"]].head()
        raise AssertionError(f"cell != task*3+domain for some rows: \n{bad}")


def target_cells(spec: DatasetSpec) -> Tuple[Tuple[int, int], ...]:
    d = spec.target_domain
    return ((d, 0), (d, 1))


def non_target_cells(spec: DatasetSpec) -> Tuple[Tuple[int, int], ...]:
    return tuple((d, t) for d in (0, 1, 2) for t in (0, 1) if d != spec.target_domain)


def non_target_domains(spec: DatasetSpec) -> Tuple[int, ...]:
    return tuple(d for d in (0, 1, 2) if d != spec.target_domain)


# ---------------------------------------------------------------------------
# Raw loader.
# ---------------------------------------------------------------------------
def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.raw_csv, encoding="utf-8-sig")
    # Filter by formal_run_key prefix (s6|sample|... for kr, s6ml|sample|... for ml)
    df = df[df["formal_run_key"].astype(str).str.startswith(spec.formal_run_key_prefix)].copy()
    if len(df) != spec.assert_n_rows:
        raise AssertionError(
            f"{spec.name}: expected {spec.assert_n_rows} rows, got {len(df)}"
        )
    if df["arm"].nunique() != len(ARMS):
        raise AssertionError(
            f"{spec.name}: expected {len(ARMS)} arms, got {df['arm'].nunique()}"
        )
    if df["seed"].nunique() != len(SEEDS):
        raise AssertionError(f"{spec.name}: expected {len(SEEDS)} seeds")
    if df["checkpoint"].nunique() != 7:
        raise AssertionError(f"{spec.name}: expected 7 checkpoints")
    if df["cell"].nunique() != spec.assert_n_cells:
        raise AssertionError(
            f"{spec.name}: expected {spec.assert_n_cells} cells, got {df['cell'].nunique()}"
        )
    counts = (
        df.groupby(["arm", "seed", "checkpoint"]).size().unique()
    )
    if not (len(counts) == 1 and counts[0] == spec.assert_n_cells):
        raise AssertionError(
            f"{spec.name}: each (arm,seed,checkpoint) must have {spec.assert_n_cells} rows"
        )
    assert_cell_invariant(df)
    # Add epoch column.
    df = checkpoint_to_epoch(df)
    df["dataset"] = spec.name
    df["cell_label"] = [
        cell_label(int(d), int(t)) for d, t in zip(df["domain"], df["task"])
    ]
    df["is_target"] = df["domain"].astype(int) == spec.target_domain
    return df


def checkpoint_to_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `epoch` integer column without collapsing selected/final.

    final and common_epoch_49 are distinct evaluation rows; we keep them
    separate as instructed.
    """
    df = df.copy()

    def _map(row: pd.Series) -> int:
        v = row["checkpoint"]
        if v in CHECKPOINT_TO_EPOCH:
            return CHECKPOINT_TO_EPOCH[v]
        if v == "selected":
            return int(row["stage_epoch"])
        if v == "final":
            return 49
        raise ValueError(f"Unknown checkpoint string: {v}")

    df["epoch"] = df.apply(_map, axis=1)
    return df


# ---------------------------------------------------------------------------
# Self-test assertions.
# ---------------------------------------------------------------------------
def assert_selected_epoch_consistency(raw: pd.DataFrame, spec: DatasetSpec) -> None:
    """Per spec §9: 4/4 seeds of selected non-target/macro must be positive.

    Asserted per-seed sign of ΔSel (DH - F) at the selected row, plus the
    4-seed mean of macro & non-target Δwithin tolerance.
    """
    if spec.name != "kuairand":
        return
    sel = raw[(raw["checkpoint"] == "selected") & (raw["arm"].isin(["F", "DH"]))].copy()
    piv = sel.pivot_table(
        index=["seed", "domain", "task"], columns="arm", values="auc"
    ).reset_index()
    piv["delta"] = piv["DH"] - piv["F"]
    piv["cell"] = [
        cell_label(int(d), int(t)) for d, t in zip(piv["domain"], piv["task"])
    ]
    by_cell = piv.groupby("cell")["delta"].mean()
    expected = {
        "d1t0": -0.000157,
        "d1t1": -0.000675,
        "d0t0": 0.009319,
        "d0t1": 0.013688,
        "d2t0": 0.004391,
        "d2t1": -0.000371,
    }
    for c, exp in expected.items():
        got = float(by_cell[c])
        if abs(got - exp) > TOL_MEAN:
            raise AssertionError(
                f"{spec.name}: {c} 4-seed mean ΔSel {got} not within {TOL_MEAN} of {exp}"
            )
    macro = piv["delta"].mean()
    if abs(macro - 0.004366) > TOL_MEAN:
        raise AssertionError(
            f"{spec.name}: macro ΔSel {macro} not within {TOL_MEAN} of 0.004366"
        )
    nt = piv[~piv["domain"].isin([spec.target_domain])]
    nt_mean = nt["delta"].mean()
    if abs(nt_mean - 0.006757) > TOL_MEAN:
        raise AssertionError(
            f"{spec.name}: nontarget ΔSel {nt_mean} not within {TOL_MEAN} of 0.006757"
        )
    # 4/4 positive per-seed for non-target & macro.
    per_seed = piv.groupby("seed")["delta"].mean()
    if not (per_seed > 0).all():
        raise AssertionError(
            f"{spec.name}: not all seeds have positive macro ΔSel at selected"
        )
    nt_per_seed = nt.groupby("seed")["delta"].mean()
    if not (nt_per_seed > 0).all():
        raise AssertionError(
            f"{spec.name}: not all seeds have positive nontarget ΔSel at selected"
        )


# ---------------------------------------------------------------------------
# Validation (epoch-level) loader.
# ---------------------------------------------------------------------------
def load_val_epoch(
    spec: DatasetSpec, arm: str, seed: int
) -> pd.DataFrame:
    p = Path(spec.run_root) / arm / f"seed{int(seed)}" / "joint" / "metrics_epoch.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    if len(df) != spec.assert_epochs_per_run:
        raise AssertionError(
            f"{spec.name}/{arm}/seed{seed}: expected {spec.assert_epochs_per_run} rows, got {len(df)}"
        )
    df["dataset"] = spec.name
    df["arm"] = arm
    df["seed"] = int(seed)
    df["cell_label"] = [
        cell_label(int(d), int(t)) for d, t in zip(df["domain"], df["task"])
    ]
    return df


def assert_val_macro_consistency(
    spec: DatasetSpec, raw: pd.DataFrame
) -> None:
    """Spot-check that epoch=9 validation_macro_auc matches the raw CSV
    common_epoch_9 row's validation_macro_auc_at_stage (within TOL_EXACT).
    """
    for arm in ("F", "DH"):
        for seed in SEEDS:
            v = load_val_epoch(spec, arm, int(seed))
            e9 = v[v["epoch"] == 9]
            ref = float(e9["validation_macro_auc"].iloc[0])
            raw_row = raw[
                (raw["arm"] == arm)
                & (raw["seed"] == seed)
                & (raw["checkpoint"] == "common_epoch_9")
            ]
            raw_ref = float(raw_row["validation_macro_auc_at_stage"].iloc[0])
            if abs(ref - raw_ref) > TOL_EXACT:
                raise AssertionError(
                    f"{spec.name}/{arm}/seed{seed}: val epoch=9 macro {ref} != raw {raw_ref}"
                )
            # Also assert selected_checkpoint=1 row's epoch == raw selected_epoch.
            sc_rows = v[v["selected_checkpoint"] == 1]
            if len(sc_rows) == 0:
                raise AssertionError(
                    f"{spec.name}/{arm}/seed{seed}: selected_checkpoint=1 row missing"
                )
            sc_epoch = int(sc_rows["epoch"].iloc[0])
            sel_raw = raw[
                (raw["arm"] == arm)
                & (raw["seed"] == seed)
                & (raw["checkpoint"] == "selected")
            ]
            raw_sel_epoch = int(sel_raw["stage_epoch"].iloc[0])
            if sc_epoch != raw_sel_epoch:
                raise AssertionError(
                    f"{spec.name}/{arm}/seed{seed}: selected epoch mismatch "
                    f"metrics={sc_epoch} raw={raw_sel_epoch}"
                )


# ---------------------------------------------------------------------------
# Δ pair (DH - F per seed per cell per epoch).
# ---------------------------------------------------------------------------
def delta_pair(
    df: pd.DataFrame,
    arm_a: str = "DH",
    arm_b: str = "F",
    value_col: str = "auc",
) -> pd.DataFrame:
    """Return per-seed, per-cell, per-epoch Δ = arm_a - arm_b.

    The input frame must contain both arms.
    """
    sub = df[df["arm"].isin([arm_a, arm_b])].copy()
    piv = sub.pivot_table(
        index=["epoch", "seed", "domain", "task"],
        columns="arm",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    piv["delta"] = piv[arm_a] - piv[arm_b]
    piv["dataset"] = sub["dataset"].iloc[0]
    piv["cell_label"] = [
        cell_label(int(d), int(t)) for d, t in zip(piv["domain"], piv["task"])
    ]
    return piv


# ---------------------------------------------------------------------------
# Aggregation across seeds.
# ---------------------------------------------------------------------------
def summarize_seeds(
    df: pd.DataFrame,
    value_col: str,
    seed_col: str = "seed",
    pos_threshold: float = 0.0,
) -> Dict[str, float]:
    """Mean, sd(ddof=1), n_pos(of n_seeds) where Δ>pos_threshold strictly."""
    g = df.groupby(seed_col)[value_col]
    mean = float(g.mean().mean())
    sd = float(g.mean().std(ddof=1)) if g.mean().size >= 4 else float("nan")
    per_seed = g.mean().to_dict()
    n_pos = int((g.mean() > pos_threshold).sum())
    return {
        "mean": mean,
        "sd": sd,
        "n_pos": n_pos,
        "n": int(g.mean().size),
        "per_seed": per_seed,
    }


def with_seed_pivot(
    df: pd.DataFrame,
    seed_col: str,
    val_col: str,
    out_cols: List[str],
    n_seeds: int = 4,
) -> Dict[str, float]:
    per_seed = df.groupby(seed_col)[val_col].mean()
    res = {}
    for s in SEEDS:
        res[f"{out_cols[0]}_seed{int(s)}"] = float(per_seed.get(int(s), float("nan")))
    return res


# ---------------------------------------------------------------------------
# Pattern classifier (spec §7 table 7).
# ---------------------------------------------------------------------------
def classify_pattern(x: Sequence[float]) -> str:
    """Apply the deterministic A/B/C/N rules over a 1-D sequence.

    Rules (first match wins, exactly per §7):
      A: (i) monotone-non-decreasing within EPS for all adjacent points,
         (ii) total rise >= TAU_PRAC, (iii) final value >= TAU_PRAC.
      B: not A; >=3 points >= TAU_PRAC, their max-min <= TAU_PRAC, all > -EPS.
      C: not A/B; max >= TAU_PRAC, at most 2 points >= TAU_PRAC,
        and if exactly 2 they must be adjacent.
      N: max < TAU_PRAC.
    """
    if len(x) == 0:
        return "N"
    arr = np.asarray(x, dtype=float)
    n = arr.size
    ge_mask = arr >= TAU_PRAC
    n_ge = int(ge_mask.sum())
    total_rise = float(arr[-1] - arr[0])
    max_val = float(arr.max())
    # A condition
    if n >= 2:
        diffs = np.diff(arr)
        monotone = bool(np.all(diffs >= -EPS))
    else:
        monotone = True
    is_A = (
        (total_rise >= TAU_PRAC)
        and monotone
        and (arr[-1] >= TAU_PRAC)
    )
    if is_A:
        return "A"
    # B condition: at least 3 points >= TAU_PRAC with max-min<=TAU_PRAC, all > -EPS
    if n_ge >= 3:
        idxs = np.where(ge_mask)[0]
        vals = arr[idxs]
        if (vals.max() - vals.min()) <= TAU_PRAC and bool(np.all(arr > -EPS)):
            return "B"
    # C condition
    if max_val >= TAU_PRAC and n_ge <= 2:
        if n_ge == 2:
            idxs = np.where(ge_mask)[0]
            if not (idxs[1] - idxs[0] == 1):
                pass  # adjacency requirement fails -> still C by spec wording
        return "C"
    return "N"


def n_ge_practical(x: Sequence[float]) -> int:
    return int(np.sum(np.asarray(x) >= TAU_PRAC))


def total_rise(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1] - arr[0])


def sign_changes(x: Sequence[float]) -> int:
    arr = np.asarray(x, dtype=float)
    if arr.size < 3:
        return 0
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    # a sign change happens between consecutive diffs i, i+1; both must exceed EPS
    mask = np.abs(diffs) > EPS
    both_large = mask[:-1] & mask[1:]
    flip = signs[:-1] * signs[1:] < 0
    return int(np.sum(both_large & flip))


# ---------------------------------------------------------------------------
# Verdict (spec §7.9).
# ---------------------------------------------------------------------------
@dataclass
class VerdictFeatures:
    K: int
    S_k: bool
    TC: float
    W: int
    DomA_ok: bool
    DomB_ok: bool
    DomA_K: int
    DomB_K: int
    DomA_3of4: bool
    DomB_3of4: bool
    R: float
    pattern_test_nontarget: str
    mean_delta_sel_nontarget: float
    peak_epochs_test_nontarget: List[int]
    K_dom_by_domain: Dict[int, int] = field(default_factory=dict)
    n_pos_by_domain: Dict[int, int] = field(default_factory=dict)


def render_verdict(features: VerdictFeatures) -> str:
    """Apply A → B → C → D priority."""
    f = features
    pattern_ok = f.pattern_test_nontarget in ("A", "B")
    cond_A = (
        f.K >= 3
        and f.S_k
        and f.TC < TAU_PRAC
        and f.W <= 1
        and (not math.isnan(f.R)) and f.R < 2
        and pattern_ok
    )
    if cond_A:
        return "A"
    dom_A_full = f.DomA_ok and f.DomA_K >= 3 and f.DomA_3of4
    dom_B_full = f.DomB_ok and f.DomB_K >= 3 and f.DomB_3of4
    exactly_one_dom = (dom_A_full and not dom_B_full) or (dom_B_full and not dom_A_full)
    cond_B = (
        not cond_A
        and exactly_one_dom
        and f.TC < TAU_PRAC
    )
    if cond_B:
        return "B"
    distinct_peaks = len(set(f.peak_epochs_test_nontarget))
    cond_C = (
        f.mean_delta_sel_nontarget >= TAU_PRAC
        and (f.K <= 2 or f.W >= 3 or (not math.isnan(f.R) and f.R >= 2)
             or distinct_peaks >= 3)
    )
    if cond_C:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Evidence-strength classifier (spec §7 table 8).
# ---------------------------------------------------------------------------
def evidence_strength(
    n_seeds_supporting: int,
    magnitude: float,
    both_layers_agree: bool,
    has_counterexample: bool,
) -> str:
    """STRONG = 4/4 seeds AND mean>=TAU_PRAC AND test/val agree AND no
    counterexample. MEDIUM = >=3/4 seeds OR evidence from a single layer only.
    Otherwise WEAK. (Spearman is never an input here, per spec.)"""
    if (
        n_seeds_supporting == 4
        and magnitude >= TAU_PRAC
        and both_layers_agree
        and not has_counterexample
    ):
        return "STRONG"
    if n_seeds_supporting >= 3 or not both_layers_agree:
        return "MEDIUM"
    return "WEAK"


# ---------------------------------------------------------------------------
# Smooth a window function (spec §7 table 7, val layer).
# ---------------------------------------------------------------------------
def smooth_moving(x: Sequence[float], w: int = 10) -> np.ndarray:
    """Centered moving average with window w; edge windows use the available
    points (min_periods=1) so boundary values are not attenuated toward 0."""
    arr = np.asarray(x, dtype=float)
    if w <= 1 or arr.size == 0:
        return arr.copy()
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def regime_label(target_cost: float, nontarget_gain: float) -> str:
    """Deterministic priority order from §7 table 6."""
    if target_cost >= TAU_PRAC and nontarget_gain >= TAU_PRAC:
        return "case0_winwin"
    if abs(target_cost) < TAU_PRAC and nontarget_gain >= TAU_PRAC:
        return "case1_free_gain"
    if target_cost <= -TAU_PRAC and nontarget_gain >= TAU_PRAC:
        return "case2_tradeoff"
    if target_cost <= -TAU_PRAC and nontarget_gain < TAU_PRAC:
        return "case3_no_value"
    return "other"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Lightweight smoke-check (no analysis): run via `python s6r_common.py --dataset ...`
# ---------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> None:
    spec = parse_dataset(argv)
    raw = load_raw(spec)
    assert_selected_epoch_consistency(raw, spec)
    assert_val_macro_consistency(spec, raw)
    print(f"{spec.name}: raw rows={len(raw)} selected epochs={sorted(raw.loc[raw['checkpoint']=='selected','stage_epoch'].unique().tolist())}")


if __name__ == "__main__":
    _main()