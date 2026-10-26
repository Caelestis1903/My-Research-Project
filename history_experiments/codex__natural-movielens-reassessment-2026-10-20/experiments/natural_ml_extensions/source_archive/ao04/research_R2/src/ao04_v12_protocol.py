"""Frozen Stage R2 (AO-04 v1.2) protocol constants.

This module contains no training code and no model code.  Every numeric and
string fact that must not be redefined by the runner lives here so that all
R2 components (model, runner, decision, report) share one source of truth.

Authoritative source: ``research_R2/R2_Design_Specification.md`` (2026-10-11).
Hard-coded deviations from the charter that were resolved by the parent
agent are documented in ``R2_Implementation_Report.md`` (LogLoss clip epsilon).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Arm / dataset / seed vocabulary (Section 3.1 of the design spec).
# ---------------------------------------------------------------------------

ARMS: Tuple[str, ...] = ("B0", "B1", "B2", "B3", "B4")
R2A_ARMS: Tuple[str, ...] = ("B0", "B1", "B2")
R2B_ARMS: Tuple[str, ...] = ("B3", "B4")
ARM_SET: FrozenSet[str] = frozenset(ARMS)

DATASETS: Tuple[str, ...] = ("kuairand", "movielens")
DATASET_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "kuairand": ("natural",),
    "movielens": ("d0_large",),
}
SEEDS: Tuple[int, ...] = (2022, 2023)


# ---------------------------------------------------------------------------
# 2. B3 fixed rho table.  Frozen constants; computed once, validated by SHA.
# ---------------------------------------------------------------------------

# Section 2 of the design specification fixes this numeric array.
B3_RHO_TABLE: Tuple[float, ...] = (
    0.7502780865025745,
    0.8537642404691582,
    0.44953240522281324,
    0.2525338669196546,
    0.23434256306102627,
    0.765217885741144,
)


def _json_array_hash(values: Sequence[float]) -> str:
    """Match the spec's hash value (no spaces, full precision, default flags)."""

    return hashlib.sha256(
        json.dumps([float(v) for v in values], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


B3_TABLE_SHA256: str = _json_array_hash(B3_RHO_TABLE)


# ---------------------------------------------------------------------------
# 3. Architecture raw-scalar initial values and grouped name set.
# ---------------------------------------------------------------------------

ARCHITECTURE_RAW_INIT: float = 1.0
ARCHITECTURE_PARAMETER_NAMES: Tuple[str, ...] = (
    "_weight_exp_d.deep_weights",
    "_weight_exp_t.deep_weights",
    "_weight_bal_d.deep_weights",
    "_weight_bal_t.deep_weights",
)
NEW_ARM_PARAMETER_NAMES: Dict[str, Tuple[str, ...]] = {
    "B0": (),
    "B1": ("q_rho",),
    "B2": ("q_rho",),
    "B3": (),  # rho_fixed is a buffer, not a parameter
    "B4": ("delta_bal_d",),
}


# ---------------------------------------------------------------------------
# 4. Architecture of cell/grid.
# ---------------------------------------------------------------------------

DOMAIN_NUM: int = 3
TASK_NUM: int = 2
EXPERT_NUM: int = 4
EMBEDDING_DIM: int = 16
FCN_DIMS: Tuple[int, ...] = (512, 256, 256, 64)
N_CELLS: int = DOMAIN_NUM * TASK_NUM

# j/k neighbour lookup for the (j,k) source-domain order, used by R_dt.
JK_PARTNER: Dict[int, Tuple[int, int]] = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


# ---------------------------------------------------------------------------
# 5. Optimization and training knobs.
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 40_960
EPOCHS: int = 50
NUM_WORKERS: int = 0
SHUFFLE_TRAIN: bool = True
DROP_LAST: bool = False
LEARNING_RATE_MAIN: float = 0.01
LEARNING_RATE_ARCH: float = 0.001
WEIGHT_DECAY: float = 1e-5
BETAS: Tuple[float, float] = (0.9, 0.999)
EPS_ADAM: float = 1e-8
SHADOW_PATIENCE: int = 10
COMMON_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)

# Row-major order over (device, deterministic flags).
DETERMINISTIC_FLAGS: Dict[str, object] = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
DEFAULT_DEVICE: str = "cuda:0"


# ---------------------------------------------------------------------------
# 6. Selection rule (Section 5 and 10.2 of the design spec).
# ---------------------------------------------------------------------------

SELECTION_RULE: Dict[str, str] = {
    "metric": "validation_macro_auc",
    "rule": "strict_greater",
    "tie_break": "earliest_epoch",
}


# ---------------------------------------------------------------------------
# 7. Domain-cell ordering, validation-only decision thresholds and guards.
# ---------------------------------------------------------------------------

# D_s = M(B1) - M(B0); practical mean >= 0.003, exploratory mean in [0.001, 0.003).
PRACTICAL_D_MEAN: float = 0.003
EXPLORATORY_D_MEAN: float = 0.001

# T_s = M(B1) - M(B2); task_gain mean >= 0.001.
TASK_GAIN_MEAN: float = 0.001

# Guards (Section 10.3 of the design spec).
GUARD_G_A: float = -0.001       # per-task macro AUC Δ per seed mean
GUARD_G_B: float = -0.001       # W Δ mean
GUARD_G_C: float = -0.003       # per-cell AUC Δ per seed mean
GUARD_G_D: float = -0.005       # per-cell AUC Δ single seed
GUARD_G_E: float = 0.010        # per-cell LogLoss Δ mean
GUARD_G_F: float = 0.005        # macro LogLoss Δ mean


# ---------------------------------------------------------------------------
# 8. Loss / objective settings.  LOGLOSS_EPS_ACTUAL is the value used by every
# Stage S1 metric (1e-12).  LOGLOSS_EPS_CHARTEXT is the value written in the
# Charter text (1e-7).  This module and the runner always use the actual value
# because every guard is a paired difference and the delta between the two eps
# values is far smaller than the G-e / G-f budgets.  See Implementation
# Report for the explicit recorded deviation.
# ---------------------------------------------------------------------------

LOGLOSS_EPS_ACTUAL: float = 1e-12
LOGLOSS_EPS_CHARTEXT: float = 1e-7


# ---------------------------------------------------------------------------
# 9. Per-dataset expected counts and param totals (Section 11 of the spec).
# ---------------------------------------------------------------------------

EXPECTED_COUNTS: Dict[str, Dict[str, Dict[str, int]]] = {
    "kuairand": {
        "natural": {
            "train": {"0": 172_782, "1": 1_743_898, "2": 89_452},
            "validation": {"0": 2692, "1": 246_232, "2": 1843},
            "test": {"0": 2613, "1": 246_284, "2": 1870},
        },
    },
    "movielens": {
        "d0_large": {
            "train": {"0": 150_000, "1": 34_500, "2": 3_000},
            "validation": {"0": 27_111, "1": 35_523, "2": 37_387},
            "test": {"0": 27_687, "1": 41_434, "2": 30_900},
        },
    },
}

# Train domain shares (used by W and other shared weighting).
EXPECTED_TRAIN_SHARES: Dict[str, Dict[int, float]] = {
    "kuairand": {
        0: 172_782 / (172_782 + 1_743_898 + 89_452),
        1: 1_743_898 / (172_782 + 1_743_898 + 89_452),
        2: 89_452 / (172_782 + 1_743_898 + 89_452),
    },
    "movielens": {
        0: 150_000 / 187_500,
        1: 34_500 / 187_500,
        2: 3_000 / 187_500,
    },
}

EXPECTED_PARAM_TOTAL: Dict[str, Dict[str, int]] = {
    "kuairand": {"B0": 1_614_258, "B1": 1_614_264, "B2": 1_614_261, "B3": 1_614_258, "B4": 1_614_264},
    "movielens": {"B0": 773_474,   "B1": 773_480,   "B2": 773_477,   "B3": 773_474,   "B4": 773_480},
}


# ---------------------------------------------------------------------------
# 10. Historical fingerprint anchors (Section 11.6 of the design spec).
# ---------------------------------------------------------------------------

# Anchors are asserted by the runner (G0-2) to surface silent drift.
# Full 64-hex values verified against the historical Stage S1 manifests
# experiments/stage_S1_raw_results/<ds>/<setting>/m3oe/seed2022/joint/run_manifest.json
# (closes review item R-8).
FINGERPRINT_ANCHORS: Dict[str, Dict[str, Dict[str, str]]] = {
    "kuairand": {
        "natural": {
            "encoder_hash": "58ff742e14eabc33ed001ad6ec2e4f74c6be375ab4b2a68028277b7b9bb3e33a",
            "scaler_hash": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            "base_split_train": "d2855a27bbfb88e6e391e6b235b47dc3e5f01c51018cc0324f43c4464ac9c479",
            "base_split_validation": "0e8f49a972a0c102b89329aa3b232b2a92f2b793c33a7459baa1d92a2dbe515d",
            "base_split_test": "6f5c25cfa2e38d8842c0a6c00131e12d2bbeaef89dfbb7d834d6b81736238057",
        },
    },
    "movielens": {
        "d0_large": {
            "encoder_hash": "05ab58ef6b9baf0589a3a476789006d8cc1b5c10eba53f1934c0c42d755f9e6b",
            "scaler_hash": "b73c4a08a40b2059f892c72edc6f659baf2acdd430b6ca1a50068bcd0dde1c1e",
            "base_split_train": "3524e8f34c5ad86cae8ea298e95454e5d64c75f378ee9e03259e46f0ec93f154",
            "base_split_validation": "c2910f696310bbffaed6b3e921a5c54f26be9135f9052abdffb208195b60b2e6",
            "base_split_test": "c29ac98bc6cba18d459fe2c91e2b9252091908e05d9e11f369c31d3a220144e7",
        },
    },
}


# ---------------------------------------------------------------------------
# 11. Guarded budget.
# ---------------------------------------------------------------------------

# The two datasets are completely independent for budget purposes.  Each
# dataset has its own attempt and resume counter.
BUDGET_PER_DATASET: Dict[str, int] = {
    "r2a_attempts": 6,
    "r2b_attempts": 4,
    "epochs_per_attempt": 50,
    "resume_per_run": 1,
    "resume_per_dataset_queue": 2,
    "smoke_batches_per_arm": 3,
    "smoke_batches_per_dataset": 30,
}


# ---------------------------------------------------------------------------
# 12. Hpost master switch.
# ---------------------------------------------------------------------------

HPOST_ENABLED: bool = False


# ---------------------------------------------------------------------------
# 13. Helper predicates.
# ---------------------------------------------------------------------------

def dataset_supported(dataset: str) -> bool:
    return dataset in DATASETS


def setting_supported(dataset: str, setting: str) -> bool:
    return setting in DATASET_SETTINGS.get(dataset, ())


def arm_supported(arm: str) -> bool:
    return arm in ARM_SET


def expected_param_total(dataset: str, arm: str) -> int:
    if dataset not in EXPECTED_PARAM_TOTAL:
        raise ValueError("no expected param total for dataset {}".format(dataset))
    if arm not in EXPECTED_PARAM_TOTAL[dataset]:
        raise ValueError("no expected param total for arm {}".format(arm))
    return EXPECTED_PARAM_TOTAL[dataset][arm]


def r2a_queue(dataset: str) -> List[Tuple[str, int]]:
    """Ordered (arm, seed) pairs for the fixed R2-A queue on one dataset."""

    pairs: List[Tuple[str, int]] = []
    for arm in R2A_ARMS:
        for seed in SEEDS:
            pairs.append((arm, seed))
    return pairs


__all__ = [
    "ARMS", "R2A_ARMS", "R2B_ARMS", "ARM_SET",
    "DATASETS", "DATASET_SETTINGS", "SEEDS",
    "B3_RHO_TABLE", "B3_TABLE_SHA256",
    "ARCHITECTURE_RAW_INIT", "ARCHITECTURE_PARAMETER_NAMES",
    "NEW_ARM_PARAMETER_NAMES",
    "DOMAIN_NUM", "TASK_NUM", "EXPERT_NUM",
    "EMBEDDING_DIM", "FCN_DIMS", "N_CELLS", "JK_PARTNER",
    "BATCH_SIZE", "EPOCHS", "NUM_WORKERS", "SHUFFLE_TRAIN", "DROP_LAST",
    "LEARNING_RATE_MAIN", "LEARNING_RATE_ARCH",
    "WEIGHT_DECAY", "BETAS", "EPS_ADAM",
    "SHADOW_PATIENCE", "COMMON_EPOCHS",
    "DETERMINISTIC_FLAGS", "DEFAULT_DEVICE",
    "SELECTION_RULE",
    "PRACTICAL_D_MEAN", "EXPLORATORY_D_MEAN", "TASK_GAIN_MEAN",
    "GUARD_G_A", "GUARD_G_B", "GUARD_G_C", "GUARD_G_D",
    "GUARD_G_E", "GUARD_G_F",
    "LOGLOSS_EPS_ACTUAL", "LOGLOSS_EPS_CHARTEXT",
    "EXPECTED_COUNTS", "EXPECTED_TRAIN_SHARES",
    "EXPECTED_PARAM_TOTAL",
    "FINGERPRINT_ANCHORS",
    "BUDGET_PER_DATASET",
    "HPOST_ENABLED",
    "dataset_supported", "setting_supported", "arm_supported",
    "expected_param_total", "r2a_queue",
]
