"""Frozen Stage S7-B protocol.

This module is data-only: it does not load datasets, models, or result files.
The continuation contract is bit-faithful to the S6 F arm with the same
optimizer state, the same replayed minibatch sequence, and the same anchor
checkpoints (epoch 9 = early; epoch 39 = late).  Branch semantics:

- F branch: bit-for-bit S6 F arm (``shared <- g_F``, ``private <- g_F``,
  ``moments <- g_F``).
- T branch: bit-for-bit S6 DH arm current manipulation (``moments <- g_F``,
  shared parameter step is overwritten by the read-only counterfactual
  ``Delta_T`` calculated from the pre-step persistent state; ``private <- g_F``).
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, target_for
from stage_S4_protocol import SEEDS as S4_SEEDS


STAGE = "S7B"

ANCHORS: Tuple[int, ...] = (9, 39)
K_STEPS: int = 5

DATASETS: Dict[str, Dict[str, str]] = {
    "kuairand": {"setting": "natural",
                 "raw_root": "experiments/stage_S6_raw_results",
                 "source_arm": "F"},
    "movielens": {"setting": "d2_large",
                  "raw_root": "experiments/stage_S6ML_raw_results",
                  "source_arm": "F"},
}

SEEDS: Tuple[int, ...] = tuple(int(seed) for seed in S4_SEEDS)
if set(SEEDS) != set(S4_SEEDS):
    raise AssertionError("Stage S7-B seeds must equal Stage S4 seeds")

# ---------------------------------------------------------------------------
# Validity thresholds (Algorithm Specification section 9 + designer's
# layered V0/V1/V2/V3 contract).
#
#   V0 (hard gate, step 0 only, failure aborts the run): the step-0
#       reconstruction is bit-correct.  The fingerprint equality is the
#       primary contract; the L_full and shared-delta sanity checks below
#       are redundant guards on numerical integrity of the load.
#   V1 (hard gate, per step, both branches, failure aborts the run): the
#       in-step assertions in ``controlled_continuation_step``
#       (``assert_vector_match`` on shared, installed-cosine history,
#       private FULL-residual).  These are bit-enforced inside the run
#       and raise on any violation; the runner never produces a CSV row
#       for a failed V1 step.
#   V2 (report-only, step >= 1): the F branch's divergence from the S6
#       source trajectory is EXPECTED after step 0 (private-moments
#       fresh-Adam approximation; per Algorithm Spec §4) and is recorded
#       in the per-step ``validity_L_full_diff`` /
#       ``validity_delta_norm_rel_diff`` columns.  No abort.  A
#       non-blocking tripwire flags any single step with
#       ``|L_full diff| > REVIEW_TRIPWIRE_L_FULL_DIFF`` for human review
#       in the run report.
#   V3 (hard gate, step 0 only, pre-step, both branches): the two
#       branches' private parameters must be bit-equal at the start of
#       step 0 (both were deepcopied from the same anchor state).  Any
#       violation raises inside the loop.  At step >= 1 the private
#       trajectories necessarily diverge: the T branch overwrote shared
#       with Δ_T at step 0, so the two branches' pre-step shared thetas
#       differ, hence g_P and the private Adam updates differ.  That
#       divergence is the structural consequence of the T branch's
#       semantics — NOT a V3 violation — and is recorded report-only in
#       ``v3_per_step``.
# ---------------------------------------------------------------------------

# V0 (step 0 only).
V0_L_FULL_DIFF_ABS = 1e-5
V0_DELTA_NORM_REL_DIFF = 1e-3

# V2 (step >= 1, report-only).
V2_REVIEW_TRIPWIRE_L_FULL_DIFF = 0.3

# V3 (step 0 only, both branches; step >= 1 divergence is structural).
PRIVATE_BRANCH_EQUAL_TOL = 0.0  # strict torch.equal
COS_DF_DT_CROSS_BRANCH_TOL = 1e-6  # step-0 cross-branch cos(Δ_F, Δ_T)

# Backwards-compat aliases for older references; the new V0/V2 names are
# the authoritative ones.
L_FULL_DIFF_ABORT_THRESHOLD = V0_L_FULL_DIFF_ABS
DELTA_NORM_REL_DIFF_ABORT_THRESHOLD = V0_DELTA_NORM_REL_DIFF
S4_TOLERANCE_ABS = 1e-6
S4_TOLERANCE_REL = 1e-4

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "stage_S7B_raw_results")
OUTPUT_ROOT = os.path.join(HERE, "stage_S7B_outputs")
STATE_PATH = os.path.join(HERE, "stage_S7B_state.json")
RUN_REPORT_PATH = os.path.join(OUTPUT_ROOT, "stage_S7B_run_report.md")

STEPS_CSV_NAME = "continuation_steps.csv"
GEOMETRY_CSV_NAME = "continuation_geometry.csv"
VALIDATION_CSV_NAME = "continuation_validation.csv"
AUDIT_JSON_NAME = "replay_audit.json"
PROCESS_INTERVAL_NAME = "process_interval.json"
STDOUT_LOG_NAME = "stdout.log"

BRANCH_F = "F"
BRANCH_T = "T"
BRANCHES: Tuple[str, ...] = (BRANCH_F, BRANCH_T)


def key(dataset: str, seed: int, anchor: int) -> str:
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S7-B dataset must be kuairand or movielens")
    if int(seed) not in SEEDS:
        raise ValueError("Stage S7-B seed outside matrix: {}".format(seed))
    if int(anchor) not in ANCHORS:
        raise ValueError("Stage S7-B anchor outside matrix: {}".format(anchor))
    return "s7b|{}|seed{}|anchor{}".format(dataset, int(seed), int(anchor))


def source_run_dir(dataset: str, seed: int) -> str:
    dataset = str(dataset).lower()
    spec = DATASETS[dataset]
    raw_root = spec["raw_root"]
    arm = spec["source_arm"]
    return os.path.abspath(os.path.join(
        raw_root, arm, "seed{}".format(int(seed)), "joint"))


def out_run_dir(dataset: str, seed: int, anchor: int) -> str:
    dataset = str(dataset).lower()
    if dataset not in DATASETS:
        raise ValueError("Stage S7-B dataset must be kuairand or movielens")
    return os.path.abspath(os.path.join(
        RAW_ROOT, dataset, "seed{}".format(int(seed)), "anchor{}".format(int(anchor))))


def target_setting(dataset: str) -> str:
    dataset = str(dataset).lower()
    return str(DATASETS[dataset]["setting"])


def target_local_domain(dataset: str) -> int:
    return int(target_for(str(dataset).lower(), target_setting(dataset))["local_domain"])


def source_checkpoint_filename(anchor: int) -> str:
    return "common_epoch_{}.pth".format(int(anchor))


def source_moment_filename(anchor: int) -> str:
    return "moment_state_epoch{}.pth".format(int(anchor))


# KuaiRand has 49 batches/epoch; MovieLens has 5 batches/epoch.
BATCHES_PER_EPOCH = {"kuairand": 49, "movielens": 5}


def expected_shared_step(dataset: str, anchor: int) -> int:
    return int(BATCHES_PER_EPOCH[str(dataset).lower()] * (int(anchor) + 1))


def expected_arch_step(anchor: int) -> int:
    return int(int(anchor) + 1)


__all__ = [
    "STAGE", "ANCHORS", "K_STEPS", "DATASETS", "SEEDS", "BRANCHES", "BRANCH_F", "BRANCH_T",
    "V0_L_FULL_DIFF_ABS", "V0_DELTA_NORM_REL_DIFF",
    "V2_REVIEW_TRIPWIRE_L_FULL_DIFF",
    "PRIVATE_BRANCH_EQUAL_TOL", "COS_DF_DT_CROSS_BRANCH_TOL",
    "L_FULL_DIFF_ABORT_THRESHOLD", "DELTA_NORM_REL_DIFF_ABORT_THRESHOLD",
    "S4_TOLERANCE_ABS", "S4_TOLERANCE_REL",
    "HERE", "RAW_ROOT", "OUTPUT_ROOT", "STATE_PATH", "RUN_REPORT_PATH",
    "STEPS_CSV_NAME", "GEOMETRY_CSV_NAME", "VALIDATION_CSV_NAME",
    "AUDIT_JSON_NAME", "PROCESS_INTERVAL_NAME", "STDOUT_LOG_NAME",
    "MODEL_SOURCE_KEY", "MODEL_SOURCE_SHA256",
    "BATCHES_PER_EPOCH",
    "key", "source_run_dir", "out_run_dir", "target_setting", "target_local_domain",
    "source_checkpoint_filename", "source_moment_filename",
    "expected_shared_step", "expected_arch_step",
]
