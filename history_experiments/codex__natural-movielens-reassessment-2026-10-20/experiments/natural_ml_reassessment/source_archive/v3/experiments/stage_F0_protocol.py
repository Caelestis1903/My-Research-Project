"""Frozen Stage F0-TRAIN-ML-v1 protocol and queue identities.

This module contains protocol constants and pure helpers only.  It deliberately
keeps the larger F0 matrix visible while exposing executable support for the
first MovieLens/natural MDMT + SDMT block only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROTOCOL_VERSION = "F0-TRAIN-ML-v1"
DATASET = "movielens"
SETTING = "natural"
OBJECTIVE = "sample"
CAPACITY = "Q"
BACKBONES: Tuple[str, ...] = ("m3oe", "mmoe", "cgc1")
SCOPES: Tuple[str, ...] = ("mdmt", "sdmt-d0", "sdmt-d1", "sdmt-d2")
DEVELOPMENT_SEEDS: Tuple[int, ...] = (2018, 2019)
FORMAL_SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
ADDON_SEEDS: Tuple[int, ...] = (2026, 2027, 2028, 2029)
EPOCHS = 50
FIXED_CHECKPOINT_EPOCHS: Tuple[int, ...] = (9, 19, 29, 39, 49)
LEARNING_RATES: Tuple[float, ...] = (0.001, 0.003, 0.01)
BATCH_SIZES: Tuple[int, ...] = (4096, 40960)
WEIGHT_DECAYS: Tuple[float, ...] = (0.0, 1e-5)
ADAM_BETAS: Tuple[float, float] = (0.9, 0.999)
ADAM_EPS = 1e-8
ARCH_LR_RATIO = 0.1
EMBEDDING_DIM = 16
EXPERT_NUM = 4
P_ML = 113
Q = 576_482

# These are the exact c values calculated from the frozen formulas in the
# design addendum.  Runtime construction re-counts actual parameters and fails
# closed if an implementation drifts from the table/formula.
FROZEN_WIDTHS: Dict[Tuple[str, str], Dict[str, int]] = {
    ("m3oe", "mdmt"): {"c": 64, "b": 256, "a": 512, "non_embedding": 576_482},
    ("mmoe", "mdmt"): {"c": 113, "b": 452, "a": 0, "non_embedding": 576_104},
    ("cgc1", "mdmt"): {"c": 68, "b": 272, "a": 0, "non_embedding": 581_300},
    ("m3oe", "sdmt"): {"c": 79, "b": 316, "a": 632, "non_embedding": 573_474},
    ("mmoe", "sdmt"): {"c": 130, "b": 520, "a": 0, "non_embedding": 579_290},
    ("cgc1", "sdmt"): {"c": 101, "b": 404, "a": 0, "non_embedding": 573_490},
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def protocol_hash() -> str:
    """Hash the executable protocol constants and frozen width table."""

    return sha256_json({
        "version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "setting": SETTING,
        "backbones": BACKBONES,
        "scopes": SCOPES,
        "development_seeds": DEVELOPMENT_SEEDS,
        "formal_seeds": FORMAL_SEEDS,
        "addon_seeds": ADDON_SEEDS,
        "epochs": EPOCHS,
        "fixed_checkpoint_epochs": FIXED_CHECKPOINT_EPOCHS,
        "learning_rates": LEARNING_RATES,
        "batch_sizes": BATCH_SIZES,
        "weight_decays": WEIGHT_DECAYS,
        "adam_betas": ADAM_BETAS,
        "adam_eps": ADAM_EPS,
        "arch_lr_ratio": ARCH_LR_RATIO,
        "embedding_dim": EMBEDDING_DIM,
        "expert_num": EXPERT_NUM,
        "p": P_ML,
        "q": Q,
        "widths": {"|".join(k): v for k, v in sorted(FROZEN_WIDTHS.items())},
    })


def scope_instance(scope: str) -> Optional[int]:
    key = str(scope).lower()
    if key == "mdmt":
        return None
    if key.startswith("sdmt-d") and key[-1] in "012":
        return int(key[-1])
    raise ValueError("unsupported F0-B1 scope: {}".format(scope))


def width_for(backbone: str, scope: str) -> Dict[str, int]:
    model = str(backbone).lower()
    if model not in BACKBONES:
        raise ValueError("unsupported Stage F0 backbone: {}".format(backbone))
    scope_key = "mdmt" if str(scope).lower() == "mdmt" else "sdmt"
    try:
        return dict(FROZEN_WIDTHS[(model, scope_key)])
    except KeyError as exc:
        raise ValueError("unsupported Stage F0 scope: {}".format(scope)) from exc


def config_grid() -> List[Dict[str, object]]:
    result = []
    for lr in LEARNING_RATES:
        for batch in BATCH_SIZES:
            for wd in WEIGHT_DECAYS:
                result.append({"lr": lr, "batch_size": batch, "weight_decay": wd})
    return result


def config_id(config: Mapping[str, object]) -> str:
    required = {"lr", "batch_size", "weight_decay"}
    if set(config) != required:
        raise ValueError("F0 config must contain exactly lr, batch_size, weight_decay")
    normalized = {"lr": float(config["lr"]), "batch_size": int(config["batch_size"]),
                  "weight_decay": float(config["weight_decay"])}
    if normalized not in config_grid():
        raise ValueError("configuration is outside the frozen F0 search grid")
    return sha256_json(normalized)[:16]


def unit_key(backbone: str, scope: str) -> str:
    model = str(backbone).lower()
    scope = str(scope).lower()
    if model not in BACKBONES or scope not in SCOPES:
        raise ValueError("unit is outside implemented F0-B1-ML matrix")
    return "|".join((PROTOCOL_VERSION, DATASET, SETTING, model, scope,
                     OBJECTIVE, CAPACITY))


def all_b1_units() -> List[Dict[str, object]]:
    return [{"unit_key": unit_key(model, scope), "dataset": DATASET,
             "setting": SETTING, "backbone": model, "scope": scope,
             "objective": OBJECTIVE, "capacity": CAPACITY,
             "width": width_for(model, scope)}
            for model in BACKBONES for scope in SCOPES]


def canonical_queue_entry(*, phase: str, intent: str, model: str,
                          scope: str, seed_role: str, seed: Optional[int],
                          width: Mapping[str, int], split_hashes: Mapping[str, str],
                          protocol_digest: str, code_digest: str,
                          config: Optional[Mapping[str, object]] = None,
                          attempt_id: str = "attempt-00") -> Dict[str, object]:
    """Build a collision-resistant queue record with stable retry identity.

    Attempt identity is stored beside, not inside, the scientific canonical
    key, so a retry retains the same run_id and writes to a new child folder.
    """

    if phase not in ("search", "formal", "resource"):
        raise ValueError("phase must be search, formal, or resource")
    if intent not in ("search", "formal", "resource"):
        raise ValueError("intent must be search, formal, or resource")
    if phase != intent:
        raise ValueError("phase and intent must agree for an executable F0-B1 item")
    model = str(model).lower()
    scope = str(scope).lower()
    key = unit_key(model, scope)
    cfg = None if config is None else {
        "lr": float(config["lr"]), "batch_size": int(config["batch_size"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if phase == "search" and cfg is None:
        raise ValueError("search queue records require a config")
    if phase == "formal" and cfg is None:
        raise ValueError("formal queue records require a selected config")
    if cfg is not None:
        config_id(cfg)
    width_value = {str(k): int(v) for k, v in width.items()}
    key_obj = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "intent": intent,
        "unit_key": key,
        "config_id": config_id(cfg) if cfg is not None else "",
        "config": cfg,
        "seed_role": str(seed_role),
        "seed": None if seed is None else int(seed),
        "model_width": width_value,
        "split_hashes": dict(sorted((str(k), str(v)) for k, v in split_hashes.items())),
        "protocol_hash": str(protocol_digest),
        "code_hash": str(code_digest),
    }
    canonical = _canonical_json(key_obj)
    full_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        **key_obj,
        "dataset": DATASET,
        "setting": SETTING,
        "backbone": model,
        "scope": scope,
        "attempt_id": str(attempt_id),
        "canonical_key_json": canonical,
        "run_id": full_hash[:20],
        "run_id_full_sha256": full_hash,
        "status": "QUEUED",
    }


def matrix_ledger() -> Dict[str, object]:
    """Expose required and deferred matrix counts without silently dropping work."""

    rows = [
        {"block": "F0-B1-ML", "units": 12, "search": 288, "formal": 48,
         "status": "IMPLEMENTED_NOT_RUN_APPROVAL_PENDING"},
        {"block": "F0-B2-ML", "units": 24, "search": 576, "formal": 96,
         "status": "BLOCKED_NOT_IMPLEMENTED"},
        {"block": "F0-C-ML", "units": 36, "search": 864, "formal": 144,
         "status": "BLOCKED_NOT_IMPLEMENTED"},
        {"block": "F0-D-ML", "units": 6, "search": 144, "formal": 24,
         "status": "BLOCKED_NOT_IMPLEMENTED"},
        {"block": "F0-E-ML", "units": 1, "search": 24, "formal": 4,
         "status": "BLOCKED_NOT_IMPLEMENTED"},
        {"block": "F0-B-KR", "units": 36, "search": 864, "formal": 144,
         "status": "BLOCKED_NOT_IMPLEMENTED_DATA_PROVENANCE"},
        {"block": "F0-D-KR", "units": 6, "search": 144, "formal": 24,
         "status": "BLOCKED_NOT_IMPLEMENTED_DATA_PROVENANCE"},
        {"block": "F0-E-KR", "units": 1, "search": 24, "formal": 4,
         "status": "BLOCKED_NOT_IMPLEMENTED_DATA_PROVENANCE"},
    ]
    return {"protocol_version": PROTOCOL_VERSION, "required_blocks": rows,
            "required_totals": {"units": 122, "search": 2928, "formal": 488,
                                "trajectories": 3416},
            "implemented_b1": {"units": 12, "search": 288, "formal": 48},
            "optional_p": {"ml_units_max": 9, "kr_units_max": 9,
                            "status": "BLOCKED_NOT_IMPLEMENTED"},
            "not_in_matrix": ["Ali-CCP", "optional full-structure P",
                              "gradient/mechanism interventions",
                              "inference-cost microbenchmarks"]}


def expected_matrix_totals() -> Dict[str, int]:
    ledger = matrix_ledger()["required_totals"]
    return {k: int(v) for k, v in ledger.items()}


T_CRITICAL_95_DF3 = 3.182446305284263


def paired_delta_t_interval(deltas: Sequence[float]) -> Tuple[float, float]:
    """Two-sided 95% paired-t interval for the frozen four-seed group."""

    if len(deltas) != 4:
        raise ValueError("F0 first-four interval requires exactly four paired deltas")
    values = [float(value) for value in deltas]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("paired deltas must all be finite")
    mean = sum(values) / 4.0
    sample_variance = sum((value - mean) ** 2 for value in values) / 3.0
    half_width = T_CRITICAL_95_DF3 * math.sqrt(sample_variance) / 2.0
    return mean - half_width, mean + half_width


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values in original order."""

    values = [float(p) for p in p_values]
    if any(not (0.0 <= p <= 1.0) for p in values):
        raise ValueError("p-values must lie in [0,1]")
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    n = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def primary_search_expansion_trigger(validation_primary_deltas: Sequence[Sequence[float]],
                                     epsilon: float = 0.003) -> bool:
    """Return the registered validation-only all-block seed trigger.

    Inputs are five rows (primary comparisons), each containing the first-four
    paired deltas. The calculation intentionally contains no test-data API.
    """

    if len(validation_primary_deltas) != 5:
        raise ValueError("F0-B1 seed trigger requires the five primary comparisons")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("seed-trigger epsilon must be finite and positive")
    # Validate the entire registered family before evaluating the trigger.
    # A triggered early comparison must not hide malformed later input.
    intervals = [paired_delta_t_interval(deltas)
                 for deltas in validation_primary_deltas]
    return any(lower <= 0.0 <= upper and
               (lower <= -epsilon or upper >= epsilon)
               for lower, upper in intervals)


__all__ = [
    "PROTOCOL_VERSION", "DATASET", "SETTING", "OBJECTIVE", "CAPACITY",
    "BACKBONES", "SCOPES", "DEVELOPMENT_SEEDS", "FORMAL_SEEDS", "ADDON_SEEDS",
    "EPOCHS", "FIXED_CHECKPOINT_EPOCHS", "LEARNING_RATES", "BATCH_SIZES",
    "WEIGHT_DECAYS", "ADAM_BETAS", "ADAM_EPS", "ARCH_LR_RATIO", "EMBEDDING_DIM",
    "EXPERT_NUM", "P_ML", "Q", "FROZEN_WIDTHS", "sha256_json", "protocol_hash",
    "scope_instance", "width_for", "config_grid", "config_id", "unit_key",
    "all_b1_units", "canonical_queue_entry", "matrix_ledger", "expected_matrix_totals",
    "holm_adjust", "T_CRITICAL_95_DF3", "paired_delta_t_interval",
    "primary_search_expansion_trigger",
]
