"""Frozen protocol and matrix contracts for Stage S2.

Stage S2 is deliberately an isolated layer on top of the read-only Stage S1
implementation.  This module contains only protocol values, deterministic
keys, and the exact fresh/reuse matrices.  It does not load data or run a
model.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


STAGE = "S2"
SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
DATASETS: Tuple[str, ...] = ("movielens", "kuairand")
MOVIELENS_SETTINGS: Tuple[str, ...] = ("d0_large", "d1_large", "d2_large")
KUAIRAND_SETTINGS: Tuple[str, ...] = ("natural",)
BACKBONE = "m3oe"
COMMON_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)
EPOCHS = 50
TASKS: Tuple[int, ...] = (0, 1)
PHYSICAL_DOMAINS: Tuple[int, ...] = (0, 1, 2)
ROLE_SIZES: Dict[str, int] = {"L": 150_000, "M": 34_500, "S": 3_000}
ROLE_MAPS: Dict[str, Dict[int, str]] = {
    "d0_large": {0: "L", 1: "M", 2: "S"},
    "d1_large": {0: "S", 1: "L", 2: "M"},
    "d2_large": {0: "M", 1: "S", 2: "L"},
    # Descriptive natural order only; this is not a KuaiRand rotation.
    "natural": {0: "M", 1: "L", 2: "S"},
}
KUAIRAND_TRAIN_COUNTS: Dict[int, int] = {
    0: 172_782, 1: 1_743_898, 2: 89_452,
}
THRESHOLDS: Dict[str, float] = {
    "practical_auc": -0.003,
    "strong_auc": -0.005,
    "practical_logloss": 0.005,
}

# The values are intentionally JSON-compatible and mirror S1's held-constant
# training contract, with the one allowed change being the Equal objective.
FORMAL_PROTOCOL: Dict[str, object] = {
    "stage": STAGE,
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
    "m3oe_exp_d": 1,
    "m3oe_exp_t": 1,
    "m3oe_bal_d": 1,
    "m3oe_bal_t": 1,
    "shadow_patience": 10,
    "scheduler": None,
    "objective": "equal_domain_task_mean_bce",
    "objective_formula": "mean_i( w[d_i] * mean_t(BCE(p_it,y_it)) )",
    "objective_weight_formula": "w_d=1/(3*p_d)=N/(3*n_d)",
    "objective_sampling": "ordinary_shuffled_concatenated_loader",
    "checkpoint_metric": "unweighted_validation_macro_auc",
    "checkpoint_tie_break": "earliest_epoch",
    "common_epochs": list(COMMON_EPOCHS),
    "test_evaluation": "deferred_after_selection_freeze",
    "movie_lens_role_sizes": dict(ROLE_SIZES),
    "movie_lens_rotation_rule": "RandomState(10*seed+physical_domain)",
    "kuairand_split": "global_row_order_80_10_10_then_filter",
    "formal_backbone": BACKBONE,
    "forbidden_methods": [
        "momentum", "ema", "curriculum", "schedule", "safe_mtl",
        "balanced_sampler", "importance_correction", "adaptive_weighting",
    ],
}


def canonical_dataset(dataset: str) -> str:
    value = str(dataset).lower()
    if value in {"movielens", "movie_lens", "movie-lens", "movielens-1m", "ml"}:
        return "movielens"
    if value in {"kuairand", "kuai_rand", "kuai-rand", "kuairand-pure", "kuai"}:
        return "kuairand"
    raise ValueError("unknown Stage S2 dataset: {}".format(dataset))


def settings_for_dataset(dataset: str) -> Tuple[str, ...]:
    return MOVIELENS_SETTINGS if canonical_dataset(dataset) == "movielens" else KUAIRAND_SETTINGS


def dataset_label(dataset: str) -> str:
    return "MovieLens-1M" if canonical_dataset(dataset) == "movielens" else "KuaiRand-Pure"


def physical_domains_for_dataset(dataset: str) -> Tuple[int, ...]:
    # S1 uses local domains 0,1,2 for both datasets.  Physical identities are
    # retained separately in the data/manifest layer.
    canonical_dataset(dataset)
    return PHYSICAL_DOMAINS


def expected_train_counts(dataset: str, setting: str) -> Dict[int, int]:
    dataset = canonical_dataset(dataset)
    if setting not in settings_for_dataset(dataset):
        raise ValueError("invalid Stage S2 setting {} for {}".format(setting, dataset))
    if dataset == "kuairand":
        return dict(KUAIRAND_TRAIN_COUNTS)
    return {d: int(ROLE_SIZES[ROLE_MAPS[setting][d]]) for d in PHYSICAL_DOMAINS}


def domain_shares(dataset: str, setting: str) -> Dict[int, float]:
    counts = expected_train_counts(dataset, setting)
    total = float(sum(counts.values()))
    return {int(domain): float(count) / total for domain, count in counts.items()}


def equal_formal_key(dataset: str, setting: str, seed: int) -> str:
    dataset = canonical_dataset(dataset)
    if setting not in settings_for_dataset(dataset):
        raise ValueError("invalid Stage S2 setting")
    if int(seed) not in SEEDS:
        raise ValueError("invalid Stage S2 seed")
    return "equal|{}|{}|{}|{}".format(dataset, setting, BACKBONE, int(seed))


def reuse_source_key(run_type: str, dataset: str, setting: str,
                     seed: int, physical_domain: int | None = None) -> str:
    """Return the exact frozen S1 key used by an S2 reuse record."""

    dataset = canonical_dataset(dataset)
    if run_type == "joint":
        return "joint|{}|{}|{}|{}".format(dataset, setting, BACKBONE, int(seed))
    if run_type == "single_domain" and physical_domain in PHYSICAL_DOMAINS:
        return "single_domain|{}|{}|{}|{}|{}".format(
            dataset, setting, BACKBONE, int(physical_domain), int(seed))
    raise ValueError("invalid S2 reuse source key")


def equal_run_dir(raw_root: str, dataset: str, setting: str, seed: int) -> str:
    return os.path.abspath(os.path.join(
        str(raw_root), "equal", canonical_dataset(dataset), setting, BACKBONE,
        "seed{}".format(int(seed)), "joint"))


def build_equal_matrix(raw_root: str = "experiments/stage_S2_raw_results") -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for setting in settings_for_dataset(dataset):
            for seed in SEEDS:
                tasks.append({
                    "stage": STAGE, "method": "equal", "run_type": "joint",
                    "dataset": dataset, "setting": setting, "backbone": BACKBONE,
                    "seed": int(seed), "physical_domain": None,
                    "formal_run_key": equal_formal_key(dataset, setting, seed),
                    "run_dir": equal_run_dir(raw_root, dataset, setting, seed),
                    "source_stage": "S2", "provenance_type": "fresh",
                })
    validate_equal_matrix(tasks)
    return tasks


def build_reuse_matrix(s1_raw_root: str = "experiments/stage_S1_raw_results") -> List[Dict[str, object]]:
    """Enumerate exactly 16 Sample joint + 48 single-domain S1 keys."""

    tasks: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for setting in settings_for_dataset(dataset):
            for seed in SEEDS:
                tasks.append({
                    "stage": STAGE, "method": "sample", "run_type": "joint",
                    "dataset": dataset, "setting": setting, "backbone": BACKBONE,
                    "seed": int(seed), "physical_domain": None,
                    "source_stage": "S1", "provenance_type": "reused",
                    "source_formal_run_key": reuse_source_key("joint", dataset, setting, seed),
                    "source_run_dir": os.path.abspath(os.path.join(
                        str(s1_raw_root), dataset, setting, BACKBONE,
                        "seed{}".format(int(seed)), "joint")),
                })
                for physical_domain in PHYSICAL_DOMAINS:
                    tasks.append({
                        "stage": STAGE, "method": "single_domain_mtl_reference",
                        "run_type": "single_domain", "dataset": dataset,
                        "setting": setting, "backbone": BACKBONE, "seed": int(seed),
                        "physical_domain": int(physical_domain),
                        "source_stage": "S1", "provenance_type": "reused",
                        "source_formal_run_key": reuse_source_key(
                            "single_domain", dataset, setting, seed, physical_domain),
                        "source_run_dir": os.path.abspath(os.path.join(
                            str(s1_raw_root), dataset, setting, BACKBONE,
                            "seed{}".format(int(seed)),
                            "single_domain_d{}".format(int(physical_domain)))),
                    })
    validate_reuse_matrix(tasks)
    return tasks


def validate_equal_matrix(tasks: Sequence[Mapping[str, object]]) -> None:
    expected = 4 * 4
    if len(tasks) != expected:
        raise ValueError("Stage S2 Equal matrix has {} tasks; expected {}".format(len(tasks), expected))
    keys = [str(item.get("formal_run_key", "")) for item in tasks]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Stage S2 Equal formal keys are missing or duplicated")
    expected_keys = {equal_formal_key(dataset, setting, seed)
                     for dataset in DATASETS
                     for setting in settings_for_dataset(dataset)
                     for seed in SEEDS}
    if set(keys) != expected_keys:
        raise ValueError("Stage S2 Equal matrix key support mismatch")
    for item in tasks:
        if item.get("method") != "equal" or item.get("run_type") != "joint":
            raise ValueError("Stage S2 fresh matrix contains a non-Equal task")
        if str(item.get("backbone")) != BACKBONE:
            raise ValueError("Stage S2 fresh matrix is M3OE-only")
        expected_key = equal_formal_key(str(item.get("dataset")), str(item.get("setting")), int(item.get("seed")))
        if str(item.get("formal_run_key")) != expected_key:
            raise ValueError("Stage S2 Equal formal key mismatch")


def validate_reuse_matrix(tasks: Sequence[Mapping[str, object]]) -> None:
    if len(tasks) != 64:
        raise ValueError("Stage S2 reuse matrix has {} tasks; expected 64".format(len(tasks)))
    keys = [str(item.get("source_formal_run_key", "")) for item in tasks]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Stage S2 reuse source keys are missing or duplicated")
    expected = {reuse_source_key("joint", dataset, setting, seed)
                for dataset in DATASETS
                for setting in settings_for_dataset(dataset)
                for seed in SEEDS}
    expected.update({reuse_source_key("single_domain", dataset, setting, seed, domain)
                     for dataset in DATASETS
                     for setting in settings_for_dataset(dataset)
                     for seed in SEEDS
                     for domain in PHYSICAL_DOMAINS})
    if set(keys) != expected:
        raise ValueError("Stage S2 reuse source key support mismatch")
    if sum(item.get("run_type") == "joint" for item in tasks) != 16:
        raise ValueError("Stage S2 reuse joint cardinality mismatch")
    if sum(item.get("run_type") == "single_domain" for item in tasks) != 48:
        raise ValueError("Stage S2 reuse reference cardinality mismatch")


def protocol_mismatches(values: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    mismatches: Dict[str, Dict[str, object]] = {}
    for field, expected in FORMAL_PROTOCOL.items():
        observed = values.get(field)
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    return mismatches


def assert_formal_values(values: Mapping[str, object]) -> None:
    mismatches = protocol_mismatches(values)
    if mismatches:
        raise ValueError("Stage S2 formal protocol mismatch: {}".format(mismatches))


def coefficient_table_hash(payload: Mapping[str, object]) -> str:
    """Hash a coefficient table independently of its presentation order."""

    body = {key: payload[key] for key in sorted(payload) if key != "coefficient_table_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode("utf-8")).hexdigest()


__all__ = [
    "STAGE", "SEEDS", "DATASETS", "MOVIELENS_SETTINGS", "KUAIRAND_SETTINGS",
    "BACKBONE", "COMMON_EPOCHS", "EPOCHS", "TASKS", "PHYSICAL_DOMAINS",
    "ROLE_SIZES", "ROLE_MAPS", "KUAIRAND_TRAIN_COUNTS", "THRESHOLDS",
    "FORMAL_PROTOCOL", "canonical_dataset", "dataset_label", "settings_for_dataset",
    "physical_domains_for_dataset", "expected_train_counts", "domain_shares",
    "equal_formal_key", "reuse_source_key", "equal_run_dir", "build_equal_matrix",
    "build_reuse_matrix", "validate_equal_matrix", "validate_reuse_matrix",
    "protocol_mismatches", "assert_formal_values", "coefficient_table_hash",
]
