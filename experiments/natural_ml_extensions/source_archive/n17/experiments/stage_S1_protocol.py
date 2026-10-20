"""Immutable protocol definitions for Stage S1.

Stage S1 is intentionally a small, boring protocol module.  All formal
factors live here so that the runner and the queue builder cannot silently
drift apart.  This module contains no training code and no scientific
interpretation.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
BACKBONES: Tuple[str, ...] = ("m3oe", "mmoe", "ple")
DATASETS: Tuple[str, ...] = ("movielens", "kuairand")
MOVIELENS_SETTINGS: Tuple[str, ...] = ("d0_large", "d1_large", "d2_large")
KUAIRAND_SETTINGS: Tuple[str, ...] = ("natural",)
COMMON_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)
EPOCHS: int = 50
TASK_NAMES: Tuple[str, ...] = ("task0", "task1")
REFERENCE_SCOPE: str = "single_domain_mtl_reference"
BACKBONE_LABELS: Tuple[str, ...] = ("M3OE", "MMoE", "PLE")
DATASET_LABELS: Tuple[str, ...] = ("MovieLens-1M", "KuaiRand-Pure")
EXPECTED_JOINT_RUNS: int = 48
EXPECTED_REFERENCE_RUNS: int = 144
EXPECTED_RUN_COUNT: int = 192

ROLE_SIZES: Dict[str, int] = {"L": 150_000, "M": 34_500, "S": 3_000}
ROLE_MAPS: Dict[str, Dict[int, str]] = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
    # KuaiRand has a natural size order; this is descriptive, not a rotation.
    "natural": {0: "M", 1: "L", 2: "S"},
}
PHYSICAL_DOMAINS: Dict[str, Tuple[object, ...]] = {
    "movielens": (0, 1, 2),
    "kuairand": ("tab0", "tab1", "tab4"),
}

THRESHOLDS: Dict[str, float] = {
    "practical_auc": -0.003,
    "strong_auc": -0.005,
    "practical_logloss": 0.005,
}

# The values in this dictionary are deliberately JSON-compatible.  A runner
# must copy them into its manifest and must not replace them with command-line
# defaults supplied by a caller.
FORMAL_PROTOCOL: Dict[str, object] = {
    "stage": "S1",
    "device": "cuda:0",
    "batch_size": 40_960,
    "epoch": EPOCHS,
    "learning_rate": 0.01,
    "learning_rate_darts": 0.001,
    "weight_decay": 1e-5,
    "betas": [0.9, 0.999],
    "architecture_betas": [0.9, 0.999],
    "num_workers": 0,
    "shuffle_train": True,
    "drop_last": False,
    "embedding_dim": 16,
    "fcn_dims": [512, 256, 256, 64],
    "expert_num": 4,
    "mmoe_ple_expert_hidden_units": [256, 64],
    "mmoe_ple_gate_hidden_units": [64],
    "mmoe_ple_tower": [64, "LayerNorm", "ReLU", 1],
    "m3oe_exp_d": 1,
    "m3oe_exp_t": 1,
    "m3oe_bal_d": 1,
    "m3oe_bal_t": 1,
    "shadow_patience": 10,
    "scheduler": None,
    "objective": "direct_per_example_task_mean_bce",
    "checkpoint_metric": "unweighted_validation_macro_auc",
    "checkpoint_tie_break": "earliest_epoch",
    "common_epochs": list(COMMON_EPOCHS),
    "test_evaluation": "deferred_after_selection_freeze",
    "movie_lens_role_sizes": dict(ROLE_SIZES),
    "movie_lens_rotation_rule": "RandomState(10*seed+physical_domain)",
    "kuairand_split": "global_row_order_80_10_10_then_filter",
    "forbidden_methods": [
        "momentum", "ema", "curriculum", "schedule", "safe_mtl",
        "balanced_sampler", "importance_correction", "adaptive_weighting",
    ],
}


def dataset_label(dataset: str) -> str:
    key = str(dataset).lower()
    if key in ("movielens", "movie_lens", "movie-lens", "movielens-1m", "ml"):
        return "MovieLens-1M"
    if key in ("kuairand", "kuai_rand", "kuai-rand", "kuairand-pure", "kuai"):
        return "KuaiRand-Pure"
    raise ValueError("unknown Stage S1 dataset: {}".format(dataset))


def canonical_dataset(dataset: str) -> str:
    label = dataset_label(dataset)
    return "movielens" if label == "MovieLens-1M" else "kuairand"


def settings_for_dataset(dataset: str) -> Tuple[str, ...]:
    return MOVIELENS_SETTINGS if canonical_dataset(dataset) == "movielens" else KUAIRAND_SETTINGS


def physical_domains_for_dataset(dataset: str) -> Tuple[object, ...]:
    return PHYSICAL_DOMAINS[canonical_dataset(dataset)]


def is_reference_key(key: str) -> bool:
    return str(key).startswith("single_domain|")


def formal_key(run_type: str, dataset: str, setting: str, backbone: str,
               seed: int, physical_domain: int | None = None) -> str:
    """Return the collision-free key prescribed by the design."""

    dataset = canonical_dataset(dataset)
    if run_type == "joint":
        return "joint|{}|{}|{}|{}".format(dataset, setting, backbone, int(seed))
    if run_type == "single_domain":
        if physical_domain is None or int(physical_domain) not in (0, 1, 2):
            raise ValueError("single_domain key requires physical_domain in {0,1,2}")
        return "single_domain|{}|{}|{}|{}|{}".format(
            dataset, setting, backbone, int(physical_domain), int(seed))
    raise ValueError("unknown Stage S1 run_type: {}".format(run_type))


def _run_dir(raw_root: str, dataset: str, setting: str, backbone: str,
             seed: int, run_type: str, physical_domain: int | None) -> str:
    dataset = canonical_dataset(dataset)
    leaf = "joint" if run_type == "joint" else "single_domain_d{}".format(int(physical_domain))
    return os.path.abspath(os.path.join(raw_root, dataset, setting, backbone,
                                        "seed{}".format(int(seed)), leaf))


def build_matrix(raw_root: str = "experiments/stage_S1_raw_results") -> List[Dict[str, object]]:
    """Build the complete 192-run matrix with explicit, unique keys.

    References are emitted before the corresponding joint run in every
    block.  That ordering is part of the fail-fast contract: a joint result
    must never be reported while its three references are absent.
    """

    tasks: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for setting in settings_for_dataset(dataset):
            for backbone in BACKBONES:
                for seed in SEEDS:
                    for physical_domain in (0, 1, 2):
                        tasks.append({
                            "run_type": "single_domain",
                            "dataset": dataset,
                            "setting": setting,
                            "backbone": backbone,
                            "physical_domain": physical_domain,
                            "seed": seed,
                            "formal_run_key": formal_key(
                                "single_domain", dataset, setting, backbone, seed,
                                physical_domain),
                            "run_dir": _run_dir(raw_root, dataset, setting, backbone,
                                                 seed, "single_domain", physical_domain),
                        })
                    tasks.append({
                        "run_type": "joint",
                        "dataset": dataset,
                        "setting": setting,
                        "backbone": backbone,
                        "physical_domain": None,
                        "seed": seed,
                        "formal_run_key": formal_key("joint", dataset, setting,
                                                      backbone, seed),
                        "run_dir": _run_dir(raw_root, dataset, setting, backbone,
                                             seed, "joint", None),
                    })
    validate_matrix(tasks)
    return tasks


# Historical stage modules call this operation ``build_tasks``.  Keeping the
# alias makes the S1 queue interchangeable with the existing test harness.
build_tasks = build_matrix


def expected_run_count() -> int:
    return len(build_matrix(os.path.join("experiments", "_stage_s1_count_only")))


def validate_matrix(tasks: Sequence[Mapping[str, object]]) -> None:
    """Fail closed if the formal matrix is not exactly 192 unique tasks."""

    expected = 2 * 3 * 3 * 4 * 4  # ML: 3 settings, KR: 1 setting; refs + joint
    # Expand explicitly to avoid the arithmetic comment becoming a hidden
    # assumption if a factor is edited in the future.
    expected = EXPECTED_RUN_COUNT
    if len(tasks) != expected:
        raise ValueError("Stage S1 matrix has {} tasks; expected {}".format(len(tasks), expected))
    keys = [str(item.get("formal_run_key", "")) for item in tasks]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Stage S1 formal keys are missing or duplicated")
    for item in tasks:
        dataset = canonical_dataset(str(item.get("dataset")))
        setting = str(item.get("setting"))
        backbone = str(item.get("backbone"))
        seed = int(item.get("seed"))
        run_type = str(item.get("run_type"))
        if setting not in settings_for_dataset(dataset):
            raise ValueError("invalid setting in Stage S1 matrix: {}".format(setting))
        if backbone not in BACKBONES or seed not in SEEDS:
            raise ValueError("invalid backbone or seed in Stage S1 matrix")
        if run_type not in ("joint", "single_domain"):
            raise ValueError("invalid Stage S1 run type")
        if run_type == "single_domain" and int(item.get("physical_domain")) not in (0, 1, 2):
            raise ValueError("single-domain task has invalid physical domain")
        expected_key = formal_key(run_type, dataset, setting, backbone, seed,
                                  item.get("physical_domain"))
        if str(item.get("formal_run_key")) != expected_key:
            raise ValueError("formal key mismatch: {} != {}".format(
                item.get("formal_run_key"), expected_key))


def protocol_mismatches(values: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Return exact critical-field mismatches without coercive tolerance."""

    mismatches: Dict[str, Dict[str, object]] = {}
    for field, expected in FORMAL_PROTOCOL.items():
        if field in ("stage", "forbidden_methods"):
            continue
        observed = values.get(field)
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    return mismatches


def assert_formal_values(values: Mapping[str, object]) -> None:
    mismatches = protocol_mismatches(values)
    if mismatches:
        raise ValueError("Stage S1 formal protocol mismatch: {}".format(mismatches))


def common_stage_labels() -> Dict[int, str]:
    return {int(epoch): "common_epoch_{}".format(int(epoch)) for epoch in COMMON_EPOCHS}


__all__ = [
    "SEEDS", "BACKBONES", "DATASETS", "MOVIELENS_SETTINGS", "KUAIRAND_SETTINGS",
    "COMMON_EPOCHS", "EPOCHS", "TASK_NAMES", "REFERENCE_SCOPE", "BACKBONE_LABELS", "DATASET_LABELS",
    "EXPECTED_JOINT_RUNS", "EXPECTED_REFERENCE_RUNS", "EXPECTED_RUN_COUNT",
    "ROLE_SIZES", "ROLE_MAPS",
    "PHYSICAL_DOMAINS", "THRESHOLDS", "FORMAL_PROTOCOL", "dataset_label",
    "canonical_dataset", "settings_for_dataset", "physical_domains_for_dataset",
    "is_reference_key", "formal_key", "build_matrix", "build_tasks", "expected_run_count",
    "validate_matrix", "protocol_mismatches", "assert_formal_values",
    "common_stage_labels",
]
