"""Frozen Stage S6 KuaiRand Sample four-arm matrix.

Arms factor the persistent Adam history source (H) from the current applied
update source (D).  See ``stage_S6_design.md`` for the formal definition.
This is the Sample counterpart of Stage S5: every mechanism (F/T/HT/DH dispatch,
I1–I9 invariants, current/history manipulation) is bit-identical; the only
allowed change vs S5 is the training objective (``equal`` → ``sample``).

F and T are bit-identical to the S3/S4 F/T update paths (with ``sample``
weighting instead of ``equal``).
HT installs ``g_T`` then overwrites shared params with ``theta_before + Delta_F``.
DH installs ``g_F`` then overwrites shared params with ``theta_before + Delta_T``.

The matrix generator accepts a ``phase`` argument.  Phase A materialises the
F/T replication arm pair (8 tasks / 4 waves); Phase B materialises the HT/DH
factorisation arm pair (8 tasks / 4 waves).  The two phases are run against
the same code identity — Phase A's F/T canonical products are not re-run in
Phase B.  The ``waves()`` helper only emits a pair when both arms of the pair
are present in the matrix, so a Phase B matrix yields only ``s6_HTDH_seed{N}``
waves, never ``s6_FT_seed{N}`` re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, source_inventory as s3_inventory
from stage_S4_protocol import (INITIAL_PARAMETER_SHA256, KUAIRAND_DATA_SHA256,
                               SEEDS, SOURCE_FILES as _S4_SOURCE_FILES,
                               inventory as _s4_inventory, inventory_hash)

STAGE = "S6"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "stage_S6_raw_results")
QUEUE_PATH = os.path.join(HERE, "stage_S6_queue_manifest.json")
QUEUE_PATH_PHASE_B = os.path.join(HERE, "stage_S6_queue_manifest_phaseB.json")
OUTPUT_ROOT = os.path.join(HERE, "stage_S6_outputs")


def queue_path_for_phase(phase):
    """Default queue-manifest path for a given phase.

    Phase A uses ``stage_S6_queue_manifest.json`` (the originally-frozen
    default); Phase B uses ``stage_S6_queue_manifest_phaseB.json`` so the
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

# Phase A gates the F→T replication under Sample; Phase B (HT/DH) is reserved
# for after the Sample replication gate has been ruled on by the Main Agent.
PHASE_A_ARMS = ("F", "T")
PHASE_B_ARMS = ("HT", "DH")

# Frozen Sample KuaiRand/natural coefficient table hash (Stage S6 §3.4).
# train_counts (172782, 1743898, 89452); p_d = (0.08612693..., 0.86928377..., 0.04458928...);
# w_d = 1; effective domain coefficient a_d = p_d; cell = a_d/2;
# population_mean_weight = 1.0; double_weighting = false;
# sampling = ordinary_shuffled_concatenated_loader.
SAMPLE_TABLE_SHA256 = "30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7"

# The four S6 source files (this module + the run/orchestrator/report).
S6_SOURCE_FILES = (
    "experiments/stage_S6_protocol.py",
    "experiments/stage_S6_run.py",
    "experiments/stage_S6_orchestrator.py",
    "experiments/stage_S6_report.py",
)


def key(seed, arm):
    if int(seed) not in SEEDS or arm not in ARMS:
        raise ValueError("outside S6 matrix")
    return "s6|sample|kuairand|natural|m3oe|seed{}|{}".format(seed, arm)


def inventory(root):
    """S4 inventory + the four S6 source files (hash-pinned)."""
    result = _s4_inventory(root)
    for relative in S6_SOURCE_FILES:
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
        raise ValueError("S6 phase must be 'A' or 'B' (got {!r})".format(phase))
    return [{"stage": STAGE, "seed": seed, "arm": arm,
             "formal_run_key": key(seed, arm), "dataset": "kuairand",
             "setting": "natural", "backbone": "m3oe", "objective_weighting": "sample",
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
        raise ValueError("S6 wave plan needs at least one arm pair")
    for pair in pairs:
        for seed in SEEDS:
            keys = [key(seed, arm) for arm in pair]
            if any(k not in by_key for k in keys):
                raise ValueError("S6 wave references unknown key: " + str(keys))
    return [{"wave_id": "s6_{}_seed{}".format("".join(pair), seed),
             "seed": seed, "arms": list(pair),
             "formal_run_keys": [key(seed, arm) for arm in pair]}
            for pair in pairs for seed in SEEDS]