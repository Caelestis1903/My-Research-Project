"""Small strict loader for the frozen baseline configuration."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")
METHODS = ("M3OE_FIXED1", "HYPERGATE", "AUTOIFS")
DATASETS = ("movielens", "kuairand")
SEEDS = (2022, 2023, 2024, 2025, 2026)
PROTOCOL = "MDMT-SOTA-2526-v1"


def _pairs_no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate configuration key: {key}")
        out[key] = value
    return out


def _same_shape(value, template, path="config"):
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            raise ValueError(f"Unknown or missing configuration keys at {path}")
        for key, expected in template.items():
            _same_shape(value[key], expected, f"{path}.{key}")
    elif isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise ValueError(f"Invalid list at {path}")
        for index, (item, expected) in enumerate(zip(value, template)):
            _same_shape(item, expected, f"{path}[{index}]")
    elif type(value) is not type(template):
        raise ValueError(f"Invalid type at {path}: expected {type(template).__name__}")


def load_config(path=None):
    source = Path(path) if path else DEFAULT_CONFIG
    config = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_pairs_no_duplicates)
    reference = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"), object_pairs_hook=_pairs_no_duplicates)
    _same_shape(config, reference)
    if config["protocol"] != PROTOCOL:
        raise ValueError("Unsupported experiment protocol")
    if tuple(config["methods"]) != METHODS or tuple(config["datasets"]) != DATASETS:
        raise ValueError("Method and dataset identities are frozen")
    if tuple(config["formal_seeds"]) != SEEDS:
        raise ValueError("Formal seeds must be 2022–2026")
    if config["roles"]["train"] != ["T0", "F"]:
        raise ValueError("Training role permission is frozen to T0 ∪ F")
    if config["roles"]["preprocess_fit"] != "T0":
        raise ValueError("Preprocessing must fit on T0 only")
    if config["run_status"] != "PENDING_APPROVAL":
        raise ValueError("Real-data run status must remain pending approval")
    if config["models"]["HYPERGATE"]["gates"] != {"embedding": True, "expert_bias": True, "fusion": True}:
        raise ValueError("All HyperGate mechanisms must remain enabled")
    if not config["models"]["HYPERGATE"]["configuration_provenance"]["unverified_proposal"]:
        raise ValueError("HyperGate unverified parameters must be disclosed")
    if config["smoke_only"]:
        raise ValueError("The default research config cannot be smoke-only")
    return config

