"""Frozen Stage S8 protocol -- Optimizer/Trajectory Final Kill Test.

Stage S8 tests whether the late-stage benefit observed for the DH/T arms in
Stage S6 can be stabilized through a longer same-state continuation (the
"optimizer/trajectory mechanism") or whether it merely rescues an
already-overfit late checkpoint ("late rescue").  Per Algorithm Specification
``stage_S8_design.md``.

This module is data-only.  It does not load datasets, models, or result files.

Independent variables:

- ``branch`` in {F, T, DH} (HT is not run; see design §0.c).
- ``anchor_class`` in {selected, middle, late}.
  ``selected`` resolves per seed via ``ANCHORS[dataset]["selected"][seed]``;
  ``middle`` and ``late`` are pinned to 19 and 39 respectively.
- ``horizon`` is derived from a single trajectory prefix (H_max epochs =
  ``H_MAX_EPOCHS_CORE``); nested horizons share the same batch sequence by
  construction.
- ``dataset`` in {kuairand, movielens}, seed in {2022, 2023, 2024, 2025}.

Frozen identity fields:

- Sample coefficient table hashes (KR ``SAMPLE_TABLE_SHA256``,
  ML ``SAMPLE_TABLE_SHA256``) are inherited from ``stage_S6_protocol`` and
  ``stage_S6ML_protocol`` respectively.
- ``INITIAL_PARAMETER_SHA256`` per seed is inherited from the corresponding
  S6 protocol.
- Dataset file hashes: ``KUAIRAND_DATA_SHA256`` and ``MOVIELENS_DATA_SHA256``.
- Objective weighting = ``sample`` (inherited).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, target_for
from stage_S4_protocol import SEEDS as S4_SEEDS
from stage_S4_protocol import (INITIAL_PARAMETER_SHA256 as S4_INITIAL_PARAMETER_SHA256,
                                KUAIRAND_DATA_SHA256)
from stage_S6ML_protocol import (INITIAL_PARAMETER_SHA256 as S6ML_INITIAL_PARAMETER_SHA256,
                                  MOVIELENS_DATA_SHA256, OBJECTIVE_WEIGHTING as S6ML_OBJECTIVE,
                                  SAMPLE_TABLE_SHA256 as S6ML_SAMPLE_TABLE_SHA256,
                                  SETTING as S6ML_SETTING, DATASET as S6ML_DATASET)
from stage_S6_protocol import SAMPLE_TABLE_SHA256 as S6_SAMPLE_TABLE_SHA256


STAGE = "S8"

# Anchor selection pins are PER-SEED for "selected" (from design §0.a):
# ML seed 2022/2023/2024/2025 -> 7/8/8/12; KR seed 2022/2023/2024/2025 -> 30/47/41/24.
ANCHORS: Dict[str, Dict[str, object]] = {
    "movielens": {
        "selected": {2022: 7, 2023: 8, 2024: 8, 2025: 12},
        "middle": 19,
        "late": 39,
    },
    "kuairand": {
        "selected": {2022: 30, 2023: 47, 2024: 41, 2025: 24},
        "middle": 19,
        "late": 39,
    },
}
ANCHOR_CLASSES: Tuple[str, ...] = ("selected", "middle", "late")

# Three-branch continuation matrix (no HT per design §0.c/§2.1).
BRANCHES: Tuple[str, ...] = ("F", "T", "DH")
BRANCH_HISTORY: Dict[str, str] = {"F": "FULL", "T": "TARGET", "DH": "FULL"}
BRANCH_CURRENT: Dict[str, str] = {"F": "FULL", "T": "TARGET", "DH": "TARGET"}

# Batches per epoch (KR 49, ML 5). Inherited from S7B (verified against S6 sources).
BATCHES_PER_EPOCH: Dict[str, int] = {"kuairand": 49, "movielens": 5}

# Frozen seeds, pinned to the S4 matrix.
SEEDS: Tuple[int, ...] = tuple(int(seed) for seed in S4_SEEDS)
if set(SEEDS) != set(S4_SEEDS):
    raise AssertionError("Stage S8 seeds must equal Stage S4 seeds")

# Per-dataset identity (re-exported from S6 protocols; not redefined here).
KUAIRAND_SAMPLE_TABLE_SHA256 = S6_SAMPLE_TABLE_SHA256
MOVIELENS_SAMPLE_TABLE_SHA256 = S6ML_SAMPLE_TABLE_SHA256

# Initial-parameter state hashes per seed (per-dataset).  ``KUAIRAND`` is
# inherited from ``stage_S4_protocol`` (the S4 KR Equal canonical hashes);
# ``MOVIELENS`` is inherited from ``stage_S6ML_protocol``.
INITIAL_PARAMETER_SHA256: Dict[str, Dict[int, str]] = {
    "kuairand": dict(S4_INITIAL_PARAMETER_SHA256),
    "movielens": dict(S6ML_INITIAL_PARAMETER_SHA256),
}

# Two cores: 2-epoch horizons (always) and 5-epoch (optional Phase 3).
H_MAX_EPOCHS_CORE: int = 2
H_MAX_EPOCHS_EXTENDED: int = 5

# Per-dataset dataset layout pin.  Phase 1 base runs are retrained under
# identical S6 F-arm conditions (sample / joint / m3oe) but with full-state
# dumps; their run dirs live under BASE_RAW_ROOT.
DATASETS: Dict[str, Dict[str, str]] = {
    "kuairand": {
        "setting": "natural",
        "base_raw_root_subpath": "stage_S8_base_raw_results/kuairand/seed{N}/joint",
        "source_s6_relpath": "stage_S6_raw_results/F/seed{N}/joint",
        "data_sha256": KUAIRAND_DATA_SHA256,
        "sample_table_sha256": S6_SAMPLE_TABLE_SHA256,
    },
    "movielens": {
        "setting": "d2_large",
        "base_raw_root_subpath": "stage_S8_base_raw_results/movielens/seed{N}/joint",
        "source_s6_relpath": "stage_S6ML_raw_results/F/seed{N}/joint",
        "data_sha256": MOVIELENS_DATA_SHA256,
        "sample_table_sha256": S6ML_SAMPLE_TABLE_SHA256,
    },
}

# ---------------------------------------------------------------------------
# Validity thresholds (frozen from S7-B + design §7).
#
# V0 (step 0 hard gate, continuation): F-branch first step ``L_full`` and
#     ``delta_observed_step_norm`` diff vs S8-BASE source within tolerance;
#     step-0 validation macro AUC diff vs ``metrics_epoch.csv`` epoch=anchor
#     <= 1e-6.
# V1 (per-step per-branch hard gate): applied==delta_applied and
#     private==delta_P via ``assert_vector_match`` (S4 tolerances);
#     None-gradient fail-closed; per-step RNG digest invariant.
# V2 (report-only + tripwire): F-branch per-step ``L_full`` diff vs source,
#     per-epoch post-arch macro AUC diff vs source; tripwire:
#     |L_full diff| > 0.3 OR single-epoch |macro AUC diff| > 1e-3 -> flag.
# V3 (pre-step bit-equality hard gate, all three branches at step 0):
#     parameters + main optimizer state (incl. private m/v/step) + arch
#     optimizer state must be pairwise ``torch.equal``.
# V4 (hard gate, all batches): every continuation main batch and every arch
#     batch must match S8-BASE ``batch_trace.csv`` bytewise.
# V5 (state-recovery hard gate): shared step == bpe*(anchor+1),
#     arch step == anchor+1, private state covers 100% of private params,
#     model param hash == dump hash, initial hash == S8-BASE manifest.
# VBASE (Phase 1 hard gate, run vs source S6 canonical):
#     1. all train batch fingerprints byte-equal;
#     2. per-epoch validation macro AUC diff <= 1e-6;
#     3. selected_epoch == source; selected_macro_auc diff <= 1e-6;
#     4. initial_parameter_state_hash == source;
#     5. final param hash drift recorded report-only.
# ---------------------------------------------------------------------------

V0_L_FULL_DIFF_ABS: float = 1e-5
V0_DELTA_NORM_REL_DIFF: float = 1e-3
V0_VALIDATION_MACRO_AUC_ABS: float = 1e-6
V2_REVIEW_TRIPWIRE_L_FULL_DIFF: float = 0.3
V2_REVIEW_TRIPWIRE_MACRO_AUC: float = 1e-3
VBASE_MACRO_AUC_ABS: float = 1e-6
S4_TOLERANCE_ABS: float = 1e-6
S4_TOLERANCE_REL: float = 1e-4

# File-system layout (frozen from design §4.1 and §8).
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "stage_S8_raw_results")
BASE_RAW_ROOT = os.path.join(HERE, "stage_S8_base_raw_results")
OUTPUT_ROOT = os.path.join(HERE, "stage_S8_outputs")
STATE_PATH = os.path.join(HERE, "stage_S8_state.json")
STATE_PATH_BASE = os.path.join(HERE, "stage_S8_base_state.json")
STATE_PATH_CONT = os.path.join(HERE, "stage_S8_cont_state.json")
STATE_PATH_CONT5 = os.path.join(HERE, "stage_S8_cont5_state.json")
RUN_REPORT_PATH = os.path.join(OUTPUT_ROOT, "stage_S8_run_report.md")
SUMMARY_MANIFEST_PATH = os.path.join(OUTPUT_ROOT, "stage_S8_summary_manifest.json")

# Continuation output filenames (frozen from design §6 / §8).
STEPS_CSV_NAME = "continuation_steps.csv"
GEOMETRY_CSV_NAME = "continuation_geometry.csv"
VALIDATION_CSV_NAME = "continuation_validation.csv"
BEST_CSV_NAME = "continuation_best.csv"
BEST_PTH_PREFIX = "continuation_best_"
AUDIT_JSON_NAME = "replay_audit.json"
PROCESS_INTERVAL_NAME = "process_interval.json"
STDOUT_LOG_NAME = "stdout.log"

# Base-run dump filename pattern (one per anchor class member).
FULL_STATE_DUMP_PREFIX = "full_state_epoch"
FULL_STATE_DUMP_SUFFIX = ".pth"

# The five S8 source files for the inventory hash (added on top of S6
# inventory, which itself extends S4 inventory).
S8_SOURCE_FILES: Tuple[str, ...] = (
    "experiments/stage_S8_protocol.py",
    "experiments/stage_S8_base_run.py",
    "experiments/stage_S8_run.py",
    "experiments/stage_S8_orchestrator.py",
    "experiments/stage_S8_report.py",
)


def key_base(dataset: str, seed: int) -> str:
    """Return the formal_run_key for one Phase 1 base retrain run."""
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if int(seed) not in SEEDS:
        raise ValueError("Stage S8 seed outside matrix: {}".format(seed))
    return "s8base|sample|{dataset}|{setting}|m3oe|seed{seed}|F".format(
        dataset=dataset, setting=DATASETS[dataset]["setting"], seed=int(seed))


def key_cont(dataset: str, seed: int, anchor_class: str) -> str:
    """Return the formal_run_key for one Phase 2 continuation run."""
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if int(seed) not in SEEDS:
        raise ValueError("Stage S8 seed outside matrix: {}".format(seed))
    if anchor_class not in ANCHOR_CLASSES:
        raise ValueError("Stage S8 anchor_class outside matrix: {}".format(anchor_class))
    anchor_epoch = anchor_epoch_for(dataset, int(seed), anchor_class)
    return "s8|{dataset}|seed{seed}|{anchor_class}{epoch}".format(
        dataset=dataset, seed=int(seed), anchor_class=anchor_class,
        epoch=int(anchor_epoch))


def anchor_epoch_for(dataset: str, seed: int, anchor_class: str) -> int:
    """Resolve the concrete anchor epoch for ``(dataset, seed, anchor_class)``.

    For ``selected``, the value is the per-seed pinned number; for ``middle``
    and ``late`` the pinned 19/39 are returned regardless of seed (design
    §2.2 / §9.9).
    """
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if int(seed) not in SEEDS:
        raise ValueError("Stage S8 seed outside matrix: {}".format(seed))
    if anchor_class not in ANCHOR_CLASSES:
        raise ValueError("Stage S8 anchor_class outside matrix: {}".format(anchor_class))
    spec = ANCHORS[dataset]
    if anchor_class == "selected":
        return int(spec["selected"][int(seed)])
    if anchor_class == "middle":
        return int(spec["middle"])
    return int(spec["late"])


def full_state_dump_epochs(dataset: str, seed: int) -> List[int]:
    """Return the sorted list of epoch values for which ``_dump_full_state``
    must emit a snapshot.

    Per design §2.2 / §4.1 / §9.9, this is exactly ``{selected[seed], 19, 39}``
    (deduped, sorted).  These are per-seed -- ``selected`` anchor is pinned
    per (dataset, seed); ``middle`` and ``late`` are global 19 and 39.
    """
    dataset = str(dataset).lower()
    seed = int(seed)
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if seed not in SEEDS:
        raise ValueError("Stage S8 seed outside matrix: {}".format(seed))
    selected = anchor_epoch_for(dataset, seed, "selected")
    middle = anchor_epoch_for(dataset, seed, "middle")
    late = anchor_epoch_for(dataset, seed, "late")
    return sorted({int(selected), int(middle), int(late)})


def full_state_filename(epoch: int) -> str:
    return "{prefix}{epoch}{suffix}".format(
        prefix=FULL_STATE_DUMP_PREFIX, epoch=int(epoch), suffix=FULL_STATE_DUMP_SUFFIX)


def horizon_steps(dataset: str, h_max_epochs: int) -> Dict[str, int]:
    """Return the per-horizon step counts for one continuation run.

    Nested-prefix semantics (design §2.3): ``h_5steps = 5``; ``h_half =
    ceil(bpe/2)``; ``h_1ep = bpe``; ``h_2ep = 2*bpe``; ``h_5ep = 5*bpe``.
    """
    dataset = str(dataset).lower()
    bpe = BATCHES_PER_EPOCH.get(dataset)
    if bpe is None:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if int(h_max_epochs) < 1:
        raise ValueError("h_max_epochs must be >= 1")
    h_max_steps = int(h_max_epochs) * int(bpe)
    out = {
        "5steps": min(5, h_max_steps),
        "half": int(math.ceil(float(bpe) / 2.0)),
        "1ep": int(bpe),
        "2ep": min(int(2 * bpe), h_max_steps),
    }
    if int(h_max_epochs) >= 5:
        out["5ep"] = min(int(5 * bpe), h_max_steps)
    return out


def expected_shared_step(dataset: str, anchor: int) -> int:
    """bpe * (anchor + 1): the shared Adam step after epoch ``anchor``."""
    return int(BATCHES_PER_EPOCH[str(dataset).lower()] * (int(anchor) + 1))


def expected_arch_step(anchor: int) -> int:
    """anchor + 1: the arch Adam step after epoch ``anchor``."""
    return int(int(anchor) + 1)


def base_run_dir(dataset: str, seed: int) -> str:
    """Absolute path of the Phase 1 base retrain run directory."""
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    subpath = DATASETS[dataset]["base_raw_root_subpath"].format(N=int(seed))
    return os.path.abspath(os.path.join(HERE, subpath))


def source_run_dir(dataset: str, seed: int) -> str:
    """Absolute path of the original S6 canonical F-arm run (reference)."""
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    rel = DATASETS[dataset]["source_s6_relpath"].format(N=int(seed))
    return os.path.abspath(os.path.join(HERE, rel))


def cont_run_dir(dataset: str, seed: int, anchor_class: str) -> str:
    """Absolute path of the Phase 2 continuation run directory."""
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    if anchor_class not in ANCHOR_CLASSES:
        raise ValueError("Stage S8 anchor_class outside matrix: {}".format(anchor_class))
    anchor_epoch = anchor_epoch_for(dataset, int(seed), anchor_class)
    rel = "{dataset}/seed{seed}/{anchor_class}{epoch}".format(
        dataset=dataset, seed=int(seed), anchor_class=anchor_class,
        epoch=int(anchor_epoch))
    return os.path.abspath(os.path.join(RAW_ROOT, rel))


def target_setting(dataset: str) -> str:
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S8 dataset must be kuairand or movielens")
    return str(DATASETS[dataset]["setting"])


def target_local_domain(dataset: str) -> int:
    return int(target_for(str(dataset).lower(), target_setting(dataset))["local_domain"])


def data_sha256(dataset: str) -> str:
    return str(DATASETS[str(dataset).lower()]["data_sha256"])


def sample_table_sha256(dataset: str) -> str:
    return str(DATASETS[str(dataset).lower()]["sample_table_sha256"])


def inventory(root: str) -> Dict[str, str]:
    """S6 (KR) + S6ML (ML) + S8 source-file inventory.

    The S8 base runner must reproduce the S6/S6ML F-arm training trajectory;
    its source-drift check uses this inventory.  Concretely we hash the
    S8 source files on top of whatever the S4/S6 base inventory would
    produce -- but since the S8 base runner is a thin subclass of S6 (or
    S6ML), we follow the S6 inventory pattern (S4 inventory + the S6 file
    set) and add the S8 file set on top.

    The implementation below mirrors ``stage_S6_protocol.inventory`` and
    ``stage_S6ML_protocol.inventory`` semantics: start from the S4 inventory
    (``stage_S4_protocol.inventory``) and add the relevant source files.
    We add the S6/S6ML source files first (because they are imported by the
    S8 base runner), then the S8 source files.
    """
    from stage_S4_protocol import inventory as _s4_inventory
    result = _s4_inventory(root)
    extra_files: Tuple[str, ...] = (
        "experiments/stage_S6_protocol.py",
        "experiments/stage_S6_run.py",
        "experiments/stage_S6_orchestrator.py",
        "experiments/stage_S6_report.py",
        "experiments/stage_S6ML_protocol.py",
        "experiments/stage_S6ML_run.py",
        "experiments/stage_S6ML_orchestrator.py",
        "experiments/stage_S6ML_report.py",
        "experiments/stage_S7B_protocol.py",
        "experiments/stage_S7B_run.py",
        "experiments/stage_S7B_orchestrator.py",
        "experiments/stage_S7B_report.py",
    ) + S8_SOURCE_FILES
    for relative in extra_files:
        path = os.path.join(root, *relative.split("/"))
        if not os.path.isfile(path):
            raise FileNotFoundError("S8 inventory missing file: " + path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def inventory_hash(value: Mapping[str, str]) -> str:
    """Stable hash of an inventory dict (sorted-keys canonical form)."""
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validity gradient-sampling rule for the continuation (design §6.3).
# Steps where the validation-gradient (U_k, H, eff_align) geometry must be
# recorded: step 1..5 unconditionally, then every 5th step, plus the final
# step K, plus every epoch boundary (i.e. step == epoch*bpe).
# ---------------------------------------------------------------------------

def valgrad_sample_steps(h_max_steps: int, bpe: int) -> List[int]:
    """Return the sorted 1-based step indices at which val-gradient sampling
    is required (1..H_max).  Final step (``H_max``) and every 5th step are
    unconditional; step 1..5 always included; epoch boundaries also
    included for transparency."""
    steps: List[int] = []
    if int(h_max_steps) < 1:
        return steps
    # Steps 1..5 (capped at H_max).
    for s in range(1, min(5, int(h_max_steps)) + 1):
        if s not in steps:
            steps.append(int(s))
    # Every 5th step.
    for s in range(5, int(h_max_steps) + 1, 5):
        if s not in steps:
            steps.append(int(s))
    # Final step.
    if int(h_max_steps) not in steps:
        steps.append(int(h_max_steps))
    # Epoch boundaries (each epoch's final step).
    for boundary in range(int(bpe), int(h_max_steps) + 1, int(bpe)):
        if boundary not in steps:
            steps.append(int(boundary))
    return sorted(steps)


# ---------------------------------------------------------------------------
# Queue planning (orchestrator uses these helpers to materialise its
# sequential task list).
# ---------------------------------------------------------------------------

def base_tasks() -> List[Dict[str, object]]:
    """Return the ordered Phase 1 base-task list.

    Order: all MovieLens tasks first (fast smoke), then KuaiRand (design §10).
    """
    tasks: List[Dict[str, object]] = []
    for dataset in ("movielens", "kuairand"):
        for seed in SEEDS:
            tasks.append({"phase": "base", "dataset": dataset, "seed": int(seed),
                          "anchor_class": "", "anchor_epoch": 0,
                          "formal_run_key": key_base(dataset, int(seed)),
                          "run_dir": base_run_dir(dataset, int(seed))})
    return tasks


def cont_tasks(h_max_epochs: int = H_MAX_EPOCHS_CORE,
               datasets: Sequence[str] = ("movielens", "kuairand"),
               anchor_classes: Sequence[str] = ("selected", "middle", "late")
               ) -> List[Dict[str, object]]:
    """Return the ordered Phase 2 continuation task list.

    Order: dataset-major (ML first, then KR), seed-major, anchor-class in
    (selected, middle, late).  Tasks with ``anchor + h_max_epochs > 49`` are
    excluded by default (the 5-epoch extension task helper uses the same
    scaffolding but with a relaxed bound; for ``h_max_epochs == 2`` the bound
    is always satisfied because 47 + 2 == 49 < 50).
    """
    tasks: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in SEEDS:
            for anchor_class in anchor_classes:
                anchor_epoch = anchor_epoch_for(dataset, int(seed), anchor_class)
                if int(anchor_epoch) + int(h_max_epochs) > 49:
                    continue
                tasks.append({"phase": "cont", "dataset": dataset, "seed": int(seed),
                              "anchor_class": anchor_class,
                              "anchor_epoch": int(anchor_epoch),
                              "h_max_epochs": int(h_max_epochs),
                              "formal_run_key": key_cont(dataset, int(seed), anchor_class),
                              "run_dir": cont_run_dir(dataset, int(seed), anchor_class)})
    return tasks


def cont5_tasks() -> List[Dict[str, object]]:
    """Phase 3 5-epoch extension task helper (only ``a + 5 <= 49``)."""
    return cont_tasks(h_max_epochs=H_MAX_EPOCHS_EXTENDED,
                       datasets=("movielens", "kuairand"),
                       anchor_classes=ANCHOR_CLASSES)


__all__ = [
    "STAGE", "ANCHORS", "ANCHOR_CLASSES", "BRANCHES", "BRANCH_HISTORY",
    "BRANCH_CURRENT", "BATCHES_PER_EPOCH", "SEEDS",
    "INITIAL_PARAMETER_SHA256", "KUAIRAND_SAMPLE_TABLE_SHA256",
    "MOVIELENS_SAMPLE_TABLE_SHA256",
    "H_MAX_EPOCHS_CORE", "H_MAX_EPOCHS_EXTENDED",
    "DATASETS", "V0_L_FULL_DIFF_ABS", "V0_DELTA_NORM_REL_DIFF",
    "V0_VALIDATION_MACRO_AUC_ABS", "V2_REVIEW_TRIPWIRE_L_FULL_DIFF",
    "V2_REVIEW_TRIPWIRE_MACRO_AUC", "VBASE_MACRO_AUC_ABS",
    "S4_TOLERANCE_ABS", "S4_TOLERANCE_REL",
    "HERE", "RAW_ROOT", "BASE_RAW_ROOT", "OUTPUT_ROOT",
    "STATE_PATH", "STATE_PATH_BASE", "STATE_PATH_CONT", "STATE_PATH_CONT5",
    "RUN_REPORT_PATH", "SUMMARY_MANIFEST_PATH",
    "STEPS_CSV_NAME", "GEOMETRY_CSV_NAME", "VALIDATION_CSV_NAME",
    "BEST_CSV_NAME", "BEST_PTH_PREFIX",
    "AUDIT_JSON_NAME", "PROCESS_INTERVAL_NAME", "STDOUT_LOG_NAME",
    "FULL_STATE_DUMP_PREFIX", "FULL_STATE_DUMP_SUFFIX",
    "S8_SOURCE_FILES", "MODEL_SOURCE_KEY", "MODEL_SOURCE_SHA256",
    "key_base", "key_cont", "anchor_epoch_for", "full_state_dump_epochs",
    "full_state_filename", "horizon_steps", "expected_shared_step",
    "expected_arch_step", "base_run_dir", "source_run_dir", "cont_run_dir",
    "target_setting", "target_local_domain", "data_sha256",
    "sample_table_sha256", "inventory", "inventory_hash",
    "valgrad_sample_steps",
    "base_tasks", "cont_tasks", "cont5_tasks",
]
