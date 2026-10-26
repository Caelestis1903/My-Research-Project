"""Frozen Stage S4 KuaiRand Equal four-arm matrix."""

from __future__ import annotations

import hashlib
import json
import os

from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256, source_inventory as s3_inventory

STAGE = "S4"
AMENDMENT = "A1"
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_RAW_ROOT = os.path.join(HERE, "stage_S4_raw_results")
OLD_QUEUE_PATH = os.path.join(HERE, "stage_S4_queue_manifest.json")
OLD_QUEUE_SHA256 = "693b46697ba40bf6e9cd03c6e0d4688a3ee7aa80a5d6ce2a442bed5c846fb221"
OLD_STATE_SHA256 = "8ebff1bc35dd7fa5a182cfb3d56b88f7e336bedb66cd93fba588c5bc188c622e"
A1_RAW_ROOT = os.path.join(HERE, "stage_S4_A1_raw_results")
A1_QUEUE_PATH = os.path.join(HERE, "stage_S4_queue_manifest_A1.json")
A1_OUTPUT_ROOT = os.path.join(HERE, "stage_S4_A1_outputs")
A1_QUEUE_SHA256 = "c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66"
A1_STATE_SHA256 = "540aeb37729a749585dd018adb3e242caf05c83a5d88dfb647a51675a5f2d554"
A2_RAW_ROOT = os.path.join(HERE, "stage_S4_A2_raw_results")
A2_QUEUE_PATH = os.path.join(HERE, "stage_S4_queue_manifest_A2.json")
A2_OUTPUT_ROOT = os.path.join(HERE, "stage_S4_A2_outputs")
OLD_ABORT_REPORT_PATH = os.path.join(HERE, "reports", "stage_S4_run_report.md")
OLD_ABORT_REPORT_SHA256 = "52b6d5924fb60c6453e669d6318a11cc064cf8858c770b77440ff30e33a5d30e"
SEEDS = (2022, 2023, 2024, 2025)
ARMS = ("F", "T", "N", "U")
COMMON_EPOCHS = (9, 19, 29, 39, 49)
KUAIRAND_DATA_SHA256 = "a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f"
EQUAL_TABLE_SHA256 = "6dd0ce338307e7aa24c600cfe4a05397944c6e3a19a88dd0851b269f2925dd7d"
INITIAL_PARAMETER_SHA256 = {
    2022: "381a86db191ec55764195800b70e2fea10d23f9b055dee7f5d50ce951d8f2ae4",
    2023: "2280adf3e4dbda491275b476b2816b968b67e00676c0ac50a0aebd0209fcabd5",
    2024: "7bb9447c093232a4b656a6d32362963c5022023f94dc66379463f2dcad6155ae",
    2025: "90f375b17f9a56f58499edd1fecd99a703a09eb5d880b7ca402a45581e7c93b4",
}
LABELS = ("selected", "final", "common9", "common19", "common29", "common39", "common49")
SOURCE_FILES = ("experiments/stage_S4_protocol.py", "experiments/stage_S4_gradient_controls.py",
                "experiments/stage_S4_run.py", "experiments/stage_S4_orchestrator.py",
                "experiments/stage_S4_report.py")


def key(seed, arm):
    if int(seed) not in SEEDS or arm not in ARMS:
        raise ValueError("outside S4 matrix")
    return "s4|equal|kuairand|natural|m3oe|seed{}|{}".format(seed, arm)


def inventory(root):
    result = s3_inventory(root)
    for relative in SOURCE_FILES:
        path = os.path.join(root, *relative.split("/"))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def inventory_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matrix(raw_root):
    return [{"stage": STAGE, "seed": seed, "arm": arm,
             "formal_run_key": key(seed, arm), "dataset": "kuairand",
             "setting": "natural", "backbone": "m3oe", "objective_weighting": "equal",
             "run_dir": os.path.abspath(os.path.join(raw_root, arm, "seed{}".format(seed), "joint"))}
            for seed in SEEDS for arm in ARMS]


def waves(tasks):
    by_key = {task["formal_run_key"]: task for task in tasks}
    if len(by_key) != 16 or set(by_key) != {key(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("S4 requires exactly 16 unique keys")
    return [{"wave_id": "s4_{}_seed{}".format(pair, seed),
             "seed": seed, "arms": list(pair),
             "formal_run_keys": [key(seed, arm) for arm in pair]}
            for pair in ("FT", "NU") for seed in SEEDS]
