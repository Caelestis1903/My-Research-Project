"""Frozen Stage S3 protocol, run matrix, and pairing contracts.

This module is data-only: it does not load datasets, models, or result files.
The amendment A1 matrix is the sole supported formal plan.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Mapping, Sequence, Tuple

from stage_S2_protocol import (
    BACKBONE, COMMON_EPOCHS, DATASETS, ROLE_MAPS, ROLE_SIZES, SEEDS,
    canonical_dataset, reuse_source_key, settings_for_dataset,
)

STAGE = "S3"
FULL = "FULL"
ACCESS = "TARGET_ONLY_SHARED"
TARGET_ONLY_SHARED = ACCESS
OBJECTIVES: Tuple[str, ...] = ("sample", "equal")
S3_DATASETS: Tuple[Tuple[str, str], ...] = (("kuairand", "natural"), ("movielens", "d2_large"))
SETTINGS: Tuple[Tuple[str, str], ...] = S3_DATASETS
S3_SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
TARGETS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("kuairand", "natural"): {
        "local_domain": 1,
        "physical_domain": "tab1",
        "size_role": "L",
        "tasks": {0: "click", 1: "long_view"},
    },
    ("movielens", "d2_large"): {
        "local_domain": 2,
        "physical_domain": "ages_35_45_50_56",
        "size_role": "L",
        "tasks": {0: "rating>3", 1: "rating>4"},
    },
}
CHECKPOINTS: Tuple[str, ...] = (
    "selected", "final", "common_epoch_9", "common_epoch_19",
    "common_epoch_29", "common_epoch_39", "common_epoch_49",
)
COMMON_EPOCHS_S3: Tuple[int, ...] = tuple(COMMON_EPOCHS)
EPOCHS = 50
PRACTICAL_AUC = 0.003
STRONG_AUC = 0.005
MODEL_SOURCE_KEY = "m3oe/MDMTRec/models/multi_domain/model.py"
MODEL_SOURCE_SHA256 = "a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1"
LEGACY_QUEUE_SHA256 = "79c649abb1994ee81d4937920009aa7460cc32543dad074af75dc88ddc12623d"
FRESH_TARGET_ROLE = "fresh_target_only"
FRESH_SAMPLE_FULL_ROLE = "fresh_sample_full_replacement"
LEGACY_CONTEXT_ROLE = "legacy_sd_context"
EXPECTED_TARGET_ONLY_RUNS = 16
EXPECTED_SAMPLE_FULL_REPLACEMENTS = 8
EXPECTED_FRESH_RUNS = 24
EXPECTED_REUSED_EQUAL_RUNS = 8
EXPECTED_LEGACY_CONTEXT_RUNS = 24
EXPECTED_INVENTORY_ROWS = 56
EXPECTED_RAW_ROWS = 1680
EXPECTED_PAIRED_ROWS = 2688
EXPECTED_STRICT_PAIRED_ROWS = 1344
EXPECTED_LEGACY_CONTEXT_PAIRED_ROWS = 1344
EXPECTED_INTERACTION_ROWS = 336
EXPECTED_LAUNCH_GROUPS = 12
EXPECTED_TARGET_ONLY_RUNS = 16

S2_FORMAL_PROTOCOL: Dict[str, object] = {
    "stage": "S2", "device": "cuda:0", "batch_size": 40960, "epoch": 50,
    "learning_rate": 0.01, "learning_rate_darts": 0.001, "weight_decay": 1e-5,
    "betas": [0.9, 0.999], "architecture_betas": [0.9, 0.999],
    "num_workers": 0, "shuffle_train": True, "drop_last": False,
    "embedding_dim": 16, "fcn_dims": [512, 256, 256, 64], "expert_num": 4,
    "m3oe_exp_d": 1, "m3oe_exp_t": 1, "m3oe_bal_d": 1, "m3oe_bal_t": 1,
    "shadow_patience": 10, "scheduler": None,
    "objective": "equal_domain_task_mean_bce",
    "objective_formula": "mean_i( w[d_i] * mean_t(BCE(p_it,y_it)) )",
    "objective_weight_formula": "w_d=1/(3*p_d)=N/(3*n_d)",
    "objective_sampling": "ordinary_shuffled_concatenated_loader",
    "checkpoint_metric": "unweighted_validation_macro_auc",
    "checkpoint_tie_break": "earliest_epoch", "common_epochs": list(COMMON_EPOCHS),
    "test_evaluation": "deferred_after_selection_freeze",
    "movie_lens_role_sizes": dict(ROLE_SIZES),
    "movie_lens_rotation_rule": "RandomState(10*seed+physical_domain)",
    "kuairand_split": "global_row_order_80_10_10_then_filter",
    "formal_backbone": BACKBONE,
    "forbidden_methods": ["momentum", "ema", "curriculum", "schedule", "safe_mtl",
                          "balanced_sampler", "importance_correction", "adaptive_weighting"],
}
FORMAL_PROTOCOL: Dict[str, object] = {
    **S2_FORMAL_PROTOCOL,
    "stage": STAGE,
    "objective": "sample_or_equal_factor",
    "shared_access_by_fresh_role": {
        FRESH_TARGET_ROLE: ACCESS, FRESH_SAMPLE_FULL_ROLE: FULL,
    },
    "fresh_matrix": "16_target_only_shared_plus_8_fresh_sample_full_replacements",
    "full_source": "fresh_S3_sample_replacement_and_accepted_S2_equal_reuse",
    "amendment": "A1_source_identity_repair",
    "target_local_domains": {"kuairand|natural": 1, "movielens|d2_large": 2},
    "formal_conditions": ["sample_full", "equal_full", "sample_target_only_shared",
                          "equal_target_only_shared"],
    "epochs": EPOCHS,
}
REVIEW_REPORT_NAME = "stage_S3_amendment_review.md"
AMENDMENT_REUSE_AUDIT_BASENAME = "stage_S3_amendment_reuse_audit"
RELEVANT_SOURCE_FILES: Tuple[str, ...] = (
    "experiments/stage_E1_instrumentation.py",
    "experiments/stage_S1_data.py", "experiments/stage_S1_metrics.py",
    "experiments/stage_S1_models.py", "experiments/stage_S1_protocol.py",
    "experiments/stage_S1_run.py", "experiments/stage_S2_objectives.py",
    "experiments/stage_S2_protocol.py", "experiments/stage_S2_run.py",
    "experiments/stage_S3_protocol.py", "experiments/stage_S3_gradient_access.py",
    "experiments/stage_S3_run.py", "experiments/stage_S3_orchestrator.py",
    "experiments/stage_S3_report.py", "experiments/stage_S3_amendment_recovery.py",
    MODEL_SOURCE_KEY,
)

CONTRASTS: Tuple[str, ...] = (
    "equal_full_minus_sd", "sample_full_minus_sd",
    "equal_target_only_shared_minus_sd", "sample_target_only_shared_minus_sd",
    "equal_full_minus_sample_full",
    "equal_target_only_shared_minus_sample_target_only_shared",
    "equal_target_only_shared_minus_equal_full",
    "sample_target_only_shared_minus_sample_full",
)


def target_for(dataset: str, setting: str) -> Dict[str, object]:
    key = (canonical_dataset(dataset), str(setting))
    try:
        return dict(TARGETS[key])
    except KeyError as exc:
        raise ValueError("setting is outside the frozen Stage S3 matrix: {}".format(key)) from exc


def settings_for_dataset(dataset: str) -> Tuple[str, ...]:
    return tuple(setting for candidate, setting in S3_DATASETS
                 if candidate == canonical_dataset(dataset))


def formal_protocol_for(fresh_role: str) -> Dict[str, object]:
    if fresh_role not in (FRESH_TARGET_ROLE, FRESH_SAMPLE_FULL_ROLE):
        raise ValueError("unknown Stage S3 fresh role")
    result = dict(FORMAL_PROTOCOL)
    result["fresh_role"] = fresh_role
    result["shared_access"] = ACCESS if fresh_role == FRESH_TARGET_ROLE else FULL
    return result


def source_inventory(root: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    base = os.path.abspath(str(root))
    for relative in RELEVANT_SOURCE_FILES:
        path = os.path.join(base, *relative.split("/"))
        if not os.path.isfile(path):
            raise FileNotFoundError("Stage S3 amendment source is missing: {}".format(relative))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[relative] = digest.hexdigest()
    return result


def source_inventory_digest(inventory: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted((str(k), str(v)) for k, v in inventory.items())),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fresh_formal_key(objective: str, dataset: str, setting: str, seed: int) -> str:
    objective = str(objective).lower()
    dataset = canonical_dataset(dataset)
    if objective not in OBJECTIVES:
        raise ValueError("Stage S3 objective must be equal or sample")
    if (dataset, str(setting)) not in TARGETS or int(seed) not in S3_SEEDS:
        raise ValueError("run is outside the frozen Stage S3 target-only matrix")
    return "target_only_shared|{}|{}|{}|{}|{}".format(
        objective, dataset, setting, BACKBONE, int(seed))


def sample_full_replacement_key(dataset: str, setting: str, seed: int) -> str:
    dataset = canonical_dataset(dataset)
    if (dataset, str(setting)) not in TARGETS or int(seed) not in S3_SEEDS:
        raise ValueError("run is outside the frozen Stage S3 Sample FULL matrix")
    return "sample_full_replacement|sample|{}|{}|{}|{}".format(
        dataset, setting, BACKBONE, int(seed))


def sample_target_pending_key(dataset: str, setting: str, seed: int) -> str:
    """Return the Sample target-only key that remains pending until its FULL source passes."""
    return fresh_formal_key("sample", dataset, setting, seed)


def fresh_run_dir(raw_root: str, objective: str, dataset: str, setting: str,
                  seed: int) -> str:
    root = os.path.abspath(str(raw_root))
    dataset = canonical_dataset(dataset)
    seed = int(seed)
    objective = str(objective).lower()
    if objective not in OBJECTIVES:
        raise ValueError("invalid Stage S3 target-only objective")
    relative = os.path.join("target_only_shared", objective, dataset, setting,
                            BACKBONE, "seed{}".format(seed), "joint")
    if (dataset, str(setting)) not in TARGETS or seed not in S3_SEEDS:
        raise ValueError("run is outside the frozen Stage S3 matrix")
    return os.path.abspath(os.path.join(root, relative))


def sample_full_replacement_run_dir(raw_root: str, dataset: str, setting: str, seed: int) -> str:
    root = os.path.abspath(str(raw_root))
    dataset = canonical_dataset(dataset)
    if (dataset, str(setting)) not in TARGETS or int(seed) not in S3_SEEDS:
        raise ValueError("run is outside the frozen Stage S3 Sample FULL matrix")
    return os.path.abspath(os.path.join(root, "full_replacement", "sample", dataset,
                                        setting, BACKBONE, "seed{}".format(int(seed)), "joint"))


def build_fresh_matrix(raw_root: str = "experiments/stage_S3_raw_results") -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            target = target_for(dataset, setting)
            for objective in OBJECTIVES:
                full_key = (sample_full_replacement_key(dataset, setting, seed)
                            if objective == "sample" else
                            "equal|{}|{}|{}|{}".format(dataset, setting, BACKBONE, seed))
                tasks.append({
                    "stage": STAGE,
                    "method": objective,
                    "objective_weighting": objective,
                    "shared_access": ACCESS,
                    "condition": "{}_target_only_shared".format(objective),
                    "fresh_role": FRESH_TARGET_ROLE,
                    "run_type": "joint", "dataset": dataset, "setting": setting,
                    "backbone": BACKBONE, "seed": seed,
                    "formal_run_key": fresh_formal_key(objective, dataset, setting, seed),
                    "source_formal_run_key": full_key,
                    "replacement_run_dir": (sample_full_replacement_run_dir(
                        raw_root, dataset, setting, seed) if objective == "sample" else ""),
                    "run_dir": fresh_run_dir(raw_root, objective, dataset, setting, seed),
                    "source_stage": STAGE, "provenance_type": FRESH_TARGET_ROLE,
                    "target_local_domain": int(target["local_domain"]),
                    "target_physical_domain": str(target["physical_domain"]),
                    **target,
                    "physical_domain": None,
                })
            tasks.append({
                "stage": STAGE, "method": "sample_full", "objective_weighting": "sample",
                "shared_access": FULL, "condition": "sample_full",
                "fresh_role": FRESH_SAMPLE_FULL_ROLE,
                "run_type": "joint", "dataset": dataset, "setting": setting,
                "backbone": BACKBONE, "seed": seed,
                "formal_run_key": sample_full_replacement_key(dataset, setting, seed),
                "source_formal_run_key": fresh_formal_key("sample", dataset, setting, seed),
                "target_only_run_dir": fresh_run_dir(raw_root, "sample", dataset, setting, seed),
                "run_dir": sample_full_replacement_run_dir(raw_root, dataset, setting, seed),
                "source_stage": STAGE, "provenance_type": FRESH_SAMPLE_FULL_ROLE,
                "target_local_domain": int(target["local_domain"]),
                "target_physical_domain": str(target["physical_domain"]),
                **target,
                "physical_domain": None,
            })
    tasks.sort(key=lambda task: (S3_DATASETS.index((str(task["dataset"]), str(task["setting"]))),
                                 S3_SEEDS.index(int(task["seed"])),
                                 0 if task["fresh_role"] == FRESH_TARGET_ROLE else 1,
                                 OBJECTIVES.index(str(task["objective_weighting"]))))
    validate_fresh_matrix(tasks)
    return tasks


def build_reuse_matrix(s1_raw_root: str = "experiments/stage_S1_raw_results",
                       s2_raw_root: str = "experiments/stage_S2_raw_results") -> List[Dict[str, object]]:
    """Return 8 source-proven S2 Equal FULL runs plus 24 S1 D=1 references."""
    tasks: List[Dict[str, object]] = []
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            target = target_for(dataset, setting)
            tasks.append({
                "stage": STAGE, "method": "equal_full", "objective_weighting": "equal",
                "shared_access": FULL, "fresh_role": "reused",
                "reuse_role": "controlled_full_source",
                "run_type": "joint", "dataset": dataset, "setting": setting,
                "backbone": BACKBONE, "seed": seed, "physical_domain": None,
                "source_stage": "S2", "provenance_type": "reused",
                "source_formal_run_key": "equal|{}|{}|{}|{}".format(dataset, setting, BACKBONE, seed),
                "source_run_dir": os.path.abspath(os.path.join(
                    str(s2_raw_root), "equal", dataset, setting, BACKBONE,
                    "seed{}".format(seed), "joint")),
                "formal_run_key": "equal_full|{}|{}|{}|{}".format(dataset, setting, BACKBONE, seed),
                "block": [dataset, setting, seed],
                "target_local_domain": int(target["local_domain"]),
                "target_physical_domain": str(target["physical_domain"]),
                **target,
                "physical_domain": None,
            })
            for physical_domain in range(3):
                tasks.append({
                    "stage": STAGE, "method": "single_domain_mtl_reference",
                    "objective_weighting": "sample", "shared_access": "NA",
                    "fresh_role": "reused", "reuse_role": LEGACY_CONTEXT_ROLE,
                    "run_type": "single_domain", "dataset": dataset, "setting": setting,
                    "backbone": BACKBONE, "seed": seed,
                    "physical_domain": physical_domain,
                    "source_stage": "S1", "provenance_type": "reused",
                    "source_formal_run_key": reuse_source_key(
                        "single_domain", dataset, setting, seed, physical_domain),
                    "source_run_dir": os.path.abspath(os.path.join(
                        str(s1_raw_root), dataset, setting, BACKBONE,
                        "seed{}".format(seed), "single_domain_d{}".format(physical_domain))),
                    "formal_run_key": reuse_source_key(
                        "single_domain", dataset, setting, seed, physical_domain),
                    "block": [dataset, setting, seed],
                    "target_local_domain": int(target["local_domain"]),
                    "target_physical_domain": str(target["physical_domain"]),
                    **target,
                    "physical_domain": int(physical_domain),
                })
    return tasks


def build_historical_audit_matrix(s1_raw_root: str = "experiments/stage_S1_raw_results",
                                  s2_raw_root: str = "experiments/stage_S2_raw_results") -> List[Dict[str, object]]:
    """Return all 40 historical artifacts, preserving rejected Sample FULL keys."""
    tasks = build_reuse_matrix(s1_raw_root, s2_raw_root)
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            target = target_for(dataset, setting)
            tasks.append({
                "stage": STAGE, "method": "sample_full", "objective_weighting": "sample",
                "shared_access": FULL, "condition": "sample_full",
                "fresh_role": "reused", "reuse_role": "rejected_controlled_source",
                "run_type": "joint", "dataset": dataset, "setting": setting,
                "backbone": BACKBONE, "seed": seed, "physical_domain": None,
                "source_stage": "S1", "provenance_type": "reused",
                "source_formal_run_key": reuse_source_key("joint", dataset, setting, seed),
                "source_run_dir": os.path.abspath(os.path.join(
                    str(s1_raw_root), dataset, setting, BACKBONE,
                    "seed{}".format(seed), "joint")),
                "formal_run_key": "sample_full|{}|{}|{}|{}".format(dataset, setting, BACKBONE, seed),
                "block": [dataset, setting, seed],
                "target_local_domain": int(target["local_domain"]),
                "target_physical_domain": str(target["physical_domain"]),
                **target,
                "physical_domain": None,
            })
    validate_reuse_matrix(tasks)
    return tasks


def validate_fresh_matrix(tasks: Sequence[Mapping[str, object]]) -> None:
    if len(tasks) != EXPECTED_FRESH_RUNS:
        raise ValueError("Stage S3 fresh matrix must contain exactly 24 keys")
    keys = [str(task.get("formal_run_key", "")) for task in tasks]
    if any(not key for key in keys) or len(set(keys)) != EXPECTED_FRESH_RUNS:
        raise ValueError("Stage S3 fresh matrix has missing or duplicate formal keys")
    expected = {fresh_formal_key(objective, dataset, setting, seed)
                for dataset, setting in S3_DATASETS for seed in S3_SEEDS
                for objective in OBJECTIVES}
    expected.update({sample_full_replacement_key(dataset, setting, seed)
                     for dataset, setting in S3_DATASETS for seed in S3_SEEDS})
    if set(keys) != expected:
        raise ValueError("Stage S3 fresh matrix key support mismatch")
    replacements = [task for task in tasks if task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE]
    targets = [task for task in tasks if task.get("fresh_role") == FRESH_TARGET_ROLE]
    if len(replacements) != EXPECTED_SAMPLE_FULL_REPLACEMENTS or len(targets) != EXPECTED_TARGET_ONLY_RUNS:
        raise ValueError("Stage S3 fresh matrix must contain 16 target-only and 8 Sample FULL keys")
    for task in tasks:
        dataset, setting, seed = str(task["dataset"]), str(task["setting"]), int(task["seed"])
        if task.get("backbone") != BACKBONE or task.get("run_type") != "joint":
            raise ValueError("Stage S3 is M3OE joint-only")
        if task.get("fresh_role") == FRESH_TARGET_ROLE:
            objective = str(task.get("objective_weighting", ""))
            if task.get("shared_access") != ACCESS or str(task["formal_run_key"]) != fresh_formal_key(
                    objective, dataset, setting, seed):
                raise ValueError("invalid Stage S3 target-only key")
            expected_source = (sample_full_replacement_key(dataset, setting, seed)
                               if objective == "sample" else
                               "equal|{}|{}|{}|{}".format(dataset, setting, BACKBONE, seed))
            if str(task.get("source_formal_run_key", "")) != expected_source:
                raise ValueError("target-only task points to an invalid FULL source")
        elif task.get("fresh_role") == FRESH_SAMPLE_FULL_ROLE:
            if (task.get("objective_weighting") != "sample" or task.get("shared_access") != FULL
                    or str(task["formal_run_key"]) != sample_full_replacement_key(dataset, setting, seed)
                    or str(task.get("source_formal_run_key", "")) != fresh_formal_key(
                        "sample", dataset, setting, seed)):
                raise ValueError("replacement task must point to the exact Sample target run")
        else:
            raise ValueError("unknown Stage S3 fresh role")


def validate_reuse_matrix(tasks: Sequence[Mapping[str, object]]) -> None:
    if len(tasks) != 40:
        raise ValueError("Stage S3 historical audit matrix must contain 40 keys")
    joint = [task for task in tasks if task.get("run_type") == "joint"]
    single = [task for task in tasks if task.get("run_type") == "single_domain"]
    if len(joint) != 16 or len(single) != EXPECTED_LEGACY_CONTEXT_RUNS:
        raise ValueError("Stage S3 historical audit must have 16 joint and 24 D=1 references")
    keys = [str(task.get("source_formal_run_key", "")) for task in tasks]
    if any(not key for key in keys) or len(set(keys)) != 40:
        raise ValueError("Stage S3 historical source keys missing or duplicated")
    if sum(task.get("method") == "equal_full" for task in tasks) != EXPECTED_REUSED_EQUAL_RUNS:
        raise ValueError("Stage S3 expects exactly 8 Equal FULL source controls")
    if sum(task.get("method") == "sample_full" for task in tasks) != 8:
        raise ValueError("Stage S3 audit must retain exactly 8 rejected Sample FULL keys")
    if sum(task.get("reuse_role") == LEGACY_CONTEXT_ROLE for task in tasks) != 24:
        raise ValueError("Stage S3 expects exactly 24 legacy D=1 reference keys")


def build_co_run_plan(tasks: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    validate_fresh_matrix(tasks)
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    pairs: List[Dict[str, object]] = []
    for dataset, setting in S3_DATASETS:
        for seed in S3_SEEDS:
            keys = [fresh_formal_key("equal", dataset, setting, seed),
                    fresh_formal_key("sample", dataset, setting, seed)]
            pairs.append({
                "launch_group_id": "s3_{}_{}_seed{}".format(dataset, setting, seed),
                "dataset": dataset, "setting": setting, "seed": seed,
                "formal_run_keys": keys,
                "backbone": BACKBONE,
                "group_kind": "TARGET_ONLY_OBJECTIVE_PAIR",
                "workers": [
                    {"formal_run_key": keys[0], "planned_peer_run_key": keys[1],
                     "worker_id": "worker-1"},
                    {"formal_run_key": keys[1], "planned_peer_run_key": keys[0],
                     "worker_id": "worker-2"},
                ],
            })
            if any(key not in by_key for key in keys):
                raise ValueError("Stage S3 co-run plan references an absent key")
    return pairs


def build_launch_plan(tasks: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    validate_fresh_matrix(tasks)
    by_key = {str(task["formal_run_key"]): task for task in tasks}
    launch = build_co_run_plan(tasks)
    for dataset, setting in S3_DATASETS:
        for seeds in ((2022, 2023), (2024, 2025)):
            keys = [sample_full_replacement_key(dataset, setting, seed) for seed in seeds]
            if any(key not in by_key for key in keys):
                raise ValueError("Stage S3 Sample FULL resource pair is incomplete")
            launch.append({
                "launch_group_id": "s3_sample_full_replacement_{}_{}_seeds{}_{}".format(
                    dataset, setting, seeds[0], seeds[1]),
                "dataset": dataset, "setting": setting, "seed_pair": list(seeds),
                "backbone": BACKBONE,
                "formal_run_keys": keys, "group_kind": "SAMPLE_FULL_RESOURCE_PAIR",
                "workers": [
                    {"formal_run_key": keys[0], "planned_peer_run_key": keys[1],
                     "worker_id": "worker-1"},
                    {"formal_run_key": keys[1], "planned_peer_run_key": keys[0],
                     "worker_id": "worker-2"},
                ],
            })
    if len(launch) != 12:
        raise AssertionError("Stage S3 launch plan must contain exactly 12 groups")
    return launch


def co_run_plan_digest(plan: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(list(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
