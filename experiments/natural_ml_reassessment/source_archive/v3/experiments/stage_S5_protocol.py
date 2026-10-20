"""Frozen Stage S5 KuaiRand Equal four-arm matrix.

Arms factor the persistent Adam history source (H) from the current applied
update source (D).  See ``stage_S5_design.md`` for the formal definition.

F and T are bit-identical to the S3/S4 F/T update paths.
HT installs ``g_T`` then overwrites shared params with ``theta_before + Delta_F``.
DH installs ``g_F`` then overwrites shared params with ``theta_before + Delta_T``.
"""

from __future__ import annotations

import hashlib
import json
import os

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, source_inventory as s3_inventory
from stage_S4_protocol import (
    ARMS as _S4_ARMS, EQUAL_TABLE_SHA256, INITIAL_PARAMETER_SHA256,
    KUAIRAND_DATA_SHA256, SEEDS, SOURCE_FILES as _S4_SOURCE_FILES,
    inventory as _s4_inventory, inventory_hash,
)

STAGE = "S5"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(HERE, "stage_S5_raw_results")
QUEUE_PATH = os.path.join(HERE, "stage_S5_queue_manifest.json")
OUTPUT_ROOT = os.path.join(HERE, "stage_S5_outputs")

ARMS = ("F", "T", "HT", "DH")
ARM_HISTORY = {"F": "FULL", "T": "TARGET", "HT": "TARGET", "DH": "FULL"}
ARM_CURRENT = {"F": "FULL", "T": "TARGET", "HT": "FULL", "DH": "TARGET"}
COMMON_EPOCHS = (9, 19, 29, 39, 49)
MOMENT_DUMP_EPOCHS = (0, 1, 9, 19, 29, 39, 49)
LABELS = ("selected", "final", "common9", "common19", "common29", "common39", "common49")

S5_SOURCE_FILES = (
    "experiments/stage_S5_protocol.py",
    "experiments/stage_S5_run.py",
    "experiments/stage_S5_orchestrator.py",
    "experiments/stage_S5_report.py",
)


def key(seed, arm):
    if int(seed) not in SEEDS or arm not in ARMS:
        raise ValueError("outside S5 matrix")
    return "s5|equal|kuairand|natural|m3oe|seed{}|{}".format(seed, arm)


def inventory(root):
    """S3 inventory + S4 source files (unchanged, hash-pinned) + the four S5 files."""
    result = _s4_inventory(root)
    for relative in S5_SOURCE_FILES:
        path = os.path.join(root, *relative.split("/"))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def matrix(raw_root):
    return [{"stage": STAGE, "seed": seed, "arm": arm,
             "formal_run_key": key(seed, arm), "dataset": "kuairand",
             "setting": "natural", "backbone": "m3oe", "objective_weighting": "equal",
             "run_dir": os.path.abspath(os.path.join(raw_root, arm, "seed{}".format(seed), "joint"))}
            for seed in SEEDS for arm in ARMS]


def waves(tasks):
    by_key = {task["formal_run_key"]: task for task in tasks}
    if len(by_key) != 16 or set(by_key) != {key(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("S5 requires exactly 16 unique keys")
    return [{"wave_id": "s5_{}_seed{}".format("".join(pair), seed),
             "seed": seed, "arms": list(pair),
             "formal_run_keys": [key(seed, arm) for arm in pair]}
            for pair in (("F", "T"), ("HT", "DH")) for seed in SEEDS]