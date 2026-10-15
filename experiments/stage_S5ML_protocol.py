"""Frozen Stage S5-ML MovieLens Equal four-arm matrix.

This is the MovieLens cross-dataset replication extension of Stage S5. The
mechanism factor graph (persistent Adam history source ``H`` × current applied
update source ``D``) and the four arms F/T/HT/DH are bit-identical to S5's
KuaiRand definition; only ``dataset`` and ``setting`` change.

See ``promt/S5_add.md`` for the scientific specification and
``experiments/reports/stage_S5_design.md`` for the formal arm definition.

F and T are bit-identical to the S3/S4 F/T update paths.
HT installs ``g_T`` then overwrites shared params with ``theta_before + Delta_F``.
DH installs ``g_F`` then overwrites shared params with ``theta_before + Delta_T``.

The matrix generator accepts a ``phase`` argument.  Phase A materialises the
F/T replication arm pair (8 tasks / 4 waves); Phase B materialises the HT/DH
factorisation arm pair (8 tasks / 4 waves).  The two phases are run against
the same code identity — Phase A's F/T canonical products are not re-run in
Phase B.  The ``waves()`` helper only emits a pair when both arms of the pair
are present in the matrix, so a Phase B matrix yields only ``s5ml_HTDH_seed{N}``
waves, never ``s5ml_FT_seed{N}`` re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, source_inventory as s3_inventory
from stage_S4_protocol import (EQUAL_TABLE_SHA256 as _S4_EQUAL_TABLE_SHA256,
                               INITIAL_PARAMETER_SHA256 as _S4_INITIAL_PARAMETER_SHA256,
                               KUAIRAND_DATA_SHA256 as _S4_KUAIRAND_DATA_SHA256,
                               SEEDS, inventory as _s4_inventory, inventory_hash)

STAGE = "S5ML"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "stage_S5ML_raw_results")
QUEUE_PATH = os.path.join(HERE, "stage_S5ML_queue_manifest.json")
QUEUE_PATH_PHASE_B = os.path.join(HERE, "stage_S5ML_queue_manifest_phaseB.json")
OUTPUT_ROOT = os.path.join(HERE, "stage_S5ML_outputs")


def queue_path_for_phase(phase):
    """Default queue-manifest path for a given phase.

    Phase A uses ``stage_S5ML_queue_manifest.json`` (the originally-frozen
    default); Phase B uses ``stage_S5ML_queue_manifest_phaseB.json`` so the
    two phases keep independent append-only state files.  The caller can
    always override with ``--queue-manifest``.
    """
    if str(phase).upper() == "B":
        return QUEUE_PATH_PHASE_B
    return QUEUE_PATH

ARMS = ("F", "T", "HT", "DH")
ARM_HISTORY = {"F": "FULL", "T": "TARGET", "HT": "TARGET", "DH": "FULL"}
ARM_CURRENT = {"F": "FULL", "T": "TARGET", "HT": "FULL", "DH": "TARGET"}
COMMON_EPOCHS = (9, 19, 29, 39, 49)
MOMENT_DUMP_EPOCHS = (0, 1, 9, 19, 29, 39, 49)
LABELS = ("selected", "final", "common9", "common19", "common29", "common39", "common49")

# Phase A gates the F→T replication; Phase B (HT/DH) is reserved for after the
# replication gate has been ruled on by the Main Agent.
PHASE_A_ARMS = ("F", "T")
PHASE_B_ARMS = ("HT", "DH")

# Dataset identity — frozen by stage_S2_protocol / stage_S3_protocol and the
# accepted equal-MovieLens-d2_large formal run artifacts under
# ``experiments/stage_S2_raw_results/equal/movielens/d2_large/m3oe/``.
DATASET = "movielens"
SETTING = "d2_large"
BACKBONE = "m3oe"
OBJECTIVE_WEIGHTING = "equal"
TARGET_LOCAL_DOMAIN = 2
TARGET_PHYSICAL_DOMAIN = "ages_35_45_50_56"
TASKS = {0: "rating>3", 1: "rating>4"}

# Frozen MovieLens identities (taken from the canonical S2/S3 formal artifacts).
MOVIELENS_DATA_SHA256 = "f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4"
EQUAL_TABLE_SHA256 = "6e7c81773d649f17a53602944dd6d7351c72f0d5d1f22efe58b58308e2eb34bb"

# Initial-parameter state hashes per seed — taken verbatim from the canonical
# S2 Equal MovieLens d2_large formal run artifacts (and re-verified against the
# S3 target-only shared MovieLens d2_large artifacts).  These are the hashes
# the runner is required to reproduce via deterministic seeding.
INITIAL_PARAMETER_SHA256 = {
    2022: "bfa86eb450c5b99f1cc1f7829b731ad2c3eab8ea995100f07f153d6118baf76b",
    2023: "43cecc5a3ac64032542231a6812528e5b1d9c6b50c939a9ad7a8be2eb233bcaf",
    2024: "7908892468504a72ef18f2851a83b9bac1ba20ebb6da45739ebc7d4ebf3a7dcc",
    2025: "1d5048bf76ca079d47a57254248448f3b83959528a5167794cdf2c6d09f10f2b",
}

# The four S5ML source files (this module + the run/orchestrator/report).
S5ML_SOURCE_FILES = (
    "experiments/stage_S5ML_protocol.py",
    "experiments/stage_S5ML_run.py",
    "experiments/stage_S5ML_orchestrator.py",
    "experiments/stage_S5ML_report.py",
)


def key(seed, arm):
    if int(seed) not in SEEDS or arm not in ARMS:
        raise ValueError("outside S5ML matrix")
    return "s5ml|equal|{}|{}|{}|seed{}|{}".format(
        DATASET, SETTING, BACKBONE, int(seed), arm)


def inventory(root):
    """S3 inventory + the four S5ML source files (hash-pinned)."""
    result = _s4_inventory(root)
    for relative in S5ML_SOURCE_FILES:
        path = os.path.join(root, *relative.split("/"))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def matrix(raw_root, *, phase="A"):
    phase_upper = str(phase).upper()
    if phase_upper == "A":
        arms_for_phase = PHASE_A_ARMS
    elif phase_upper == "B":
        arms_for_phase = PHASE_B_ARMS
    else:
        raise ValueError("S5ML phase must be 'A' or 'B' (got {!r})".format(phase))
    return [{"stage": STAGE, "seed": seed, "arm": arm,
             "formal_run_key": key(seed, arm), "dataset": DATASET,
             "setting": SETTING, "backbone": BACKBONE,
             "objective_weighting": OBJECTIVE_WEIGHTING,
             "run_dir": os.path.abspath(os.path.join(raw_root, arm, "seed{}".format(seed), "joint"))}
            for seed in SEEDS for arm in arms_for_phase]


def waves(tasks):
    by_key = {task["formal_run_key"]: task for task in tasks}
    arms_in_plan = sorted({task["arm"] for task in tasks}, key=lambda a: ARMS.index(a))
    pairs = []
    if "F" in arms_in_plan and "T" in arms_in_plan:
        pairs.append(("F", "T"))
    if "HT" in arms_in_plan and "DH" in arms_in_plan:
        pairs.append(("HT", "DH"))
    if not pairs:
        raise ValueError("S5ML wave plan needs at least one arm pair")
    for pair in pairs:
        for seed in SEEDS:
            keys = [key(seed, arm) for arm in pair]
            if any(k not in by_key for k in keys):
                raise ValueError("S5ML wave references unknown key: " + str(keys))
    return [{"wave_id": "s5ml_{}_seed{}".format("".join(pair), seed),
             "seed": seed, "arms": list(pair),
             "formal_run_keys": [key(seed, arm) for arm in pair]}
            for pair in pairs for seed in SEEDS]