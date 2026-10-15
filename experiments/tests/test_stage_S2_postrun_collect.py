"""Regression tests for the post-run dataset-label compatibility adapter."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S2_postrun_collect as postrun  # noqa: E402


ROOT = Path(HERE).parent
S2_RAW_ROOT = ROOT / "experiments" / "stage_S2_raw_results"
S1_RAW_ROOT = ROOT / "experiments" / "stage_S1_raw_results"


def _pair(dataset: str, setting: str, seed: int):
    equal_path = S2_RAW_ROOT / "equal" / dataset / setting / "m3oe" / f"seed{seed}" / "joint" / "run_manifest.json"
    sample_path = S1_RAW_ROOT / dataset / setting / "m3oe" / f"seed{seed}" / "joint" / "run_manifest.json"
    return postrun._read_json(str(equal_path)), postrun._read_json(str(sample_path))


@pytest.mark.parametrize(
    ("dataset", "setting", "seed", "expected", "fresh_label"),
    [
        ("movielens", "d0_large", 2022, "movielens", "MovieLens-1M"),
        ("kuairand", "natural", 2022, "kuairand", "KuaiRand-Pure"),
    ],
)
def test_legal_dataset_aliases_are_normalized_only_in_memory(
    dataset, setting, seed, expected, fresh_label
):
    equal, sample = _pair(dataset, setting, seed)
    equal_before = copy.deepcopy(equal)
    sample_before = copy.deepcopy(sample)

    reasons, audit = postrun.compare_normalized_manifests(equal, sample)

    assert reasons == []
    assert audit["normalization_status"] == "OK"
    assert audit["canonical_dataset"] == expected
    assert audit["equal_raw_dataset"] == fresh_label
    assert audit["sample_raw_dataset"] == expected
    assert audit["non_dataset_bypass_count"] == 0
    assert equal == equal_before
    assert sample == sample_before

    normalized_equal, normalized_sample, _ = postrun.normalize_dataset_pair(equal, sample)
    assert normalized_equal["dataset"] == expected
    assert normalized_sample["dataset"] == expected
    assert equal["dataset"] == fresh_label
    assert sample["dataset"] == expected


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.update(dataset="unknown-dataset"),
        lambda manifest: manifest.update(dataset_key="kuairand"),
        lambda manifest: manifest.update(dataset_label=""),
        lambda manifest: manifest.update(dataset_label="KuaiRand-Pure"),
    ],
)
def test_unknown_or_invalid_dataset_identity_fails_closed(mutator):
    equal, sample = _pair("movielens", "d0_large", 2022)
    mutated = copy.deepcopy(equal)
    mutator(mutated)

    reasons, audit = postrun.compare_normalized_manifests(mutated, sample)

    assert reasons
    assert reasons[0].startswith("dataset_normalization_failed:")
    assert audit["normalization_status"] == "FAIL"
    assert audit["non_dataset_bypass_count"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "dataset_file_hash",
        "base_split_row_id_hashes",
        "physical_view_fingerprints",
        "features",
        "preprocessing_metadata",
        "initial_parameter_state_hash",
        "batch_order_hash",
    ],
)
def test_non_dataset_mismatch_still_uses_strict_comparator(field):
    equal, sample = _pair("movielens", "d0_large", 2022)
    mutated = copy.deepcopy(equal)
    value = mutated[field]
    if isinstance(value, dict):
        mutated[field] = copy.deepcopy(value)
        if field == "physical_view_fingerprints":
            first_domain = next(iter(mutated[field]))
            mutated[field][first_domain] = copy.deepcopy(mutated[field][first_domain])
            mutated[field][first_domain]["test"] = "corrupt-physical-view"
        else:
            first_key = next(iter(mutated[field]))
            mutated[field][first_key] = "corrupt-field"
    else:
        mutated[field] = "corrupt-field"

    reasons, audit = postrun.compare_normalized_manifests(mutated, sample)

    assert any(reason.endswith(":" + field) for reason in reasons), (field, reasons)
    assert audit["normalization_status"] == "OK"
    assert audit["non_dataset_mismatch_reasons"]
    assert audit["non_dataset_bypass_count"] == 0


def test_normalization_audit_is_deterministic_and_does_not_mutate_sources():
    equal, sample = _pair("kuairand", "natural", 2022)
    equal_before = copy.deepcopy(equal)
    sample_before = copy.deepcopy(sample)

    first = postrun.compare_normalized_manifests(equal, sample)
    second = postrun.compare_normalized_manifests(equal, sample)

    assert first == second
    assert equal == equal_before
    assert sample == sample_before


def test_real_s2_pair_audit_has_twelve_movielens_and_four_kuairand_mappings():
    records = postrun.inspect_formal_normalizations(
        s2_raw_root=str(S2_RAW_ROOT), s1_raw_root=str(S1_RAW_ROOT))

    assert len(records) == 16
    assert all(record["strict_comparison_reasons"] == [] for record in records)
    assert {record["non_dataset_bypass_count"] for record in records} == {0}
    assert {record["canonical_dataset"] for record in records} == {"movielens", "kuairand"}
    assert sum(record["canonical_dataset"] == "movielens" for record in records) == 12
    assert sum(record["canonical_dataset"] == "kuairand" for record in records) == 4
    assert sum(
        record["equal_raw_dataset"] != record["sample_raw_dataset"]
        for record in records
    ) == 16
