"""Leakage-controlled MovieLens natural-pool loader for Stage F0.

Unlike Stage S1, this loader uses every row in the global base-training prefix
and constructs SDMT views from that same split without changing global feature
vocabularies or embedding-table shapes.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S1_data import (  # noqa: E402
    DataBundle,
    FeatureSchema,
    MOVIELENS_COLUMNS,
    MOVIELENS_DOMAIN_IDENTITIES,
    MOVIELENS_SPARSE_COLUMNS,
    _apply_vocab,
    _feature_schema,
    _fit_vocab,
    _json_hash,
    row_ids_hash,
    split_row_fingerprint,
)


DEFAULT_MOVIELENS_PATH = "/home/xzf/data/ml-1m/ml-1m.csv"
AGE_GROUPS = ({1, 18}, {25}, {35, 45, 50, 56})


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_path(path: Optional[str]) -> str:
    candidate = path or DEFAULT_MOVIELENS_PATH
    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, "ml-1m.csv")
    return os.path.abspath(candidate)


def _split_indices(length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(length * 0.8)
    validation_end = int(length * 0.9)
    return (np.arange(0, train_end, dtype=np.int64),
            np.arange(train_end, validation_end, dtype=np.int64),
            np.arange(validation_end, length, dtype=np.int64))


def load_movielens_natural(dataset_path: Optional[str] = None,
                           seed: int = 2022) -> DataBundle:
    """Create the one MovieLens natural bundle (no size-role sampling)."""

    csv_path = _csv_path(dataset_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("MovieLens source CSV does not exist: {}".format(csv_path))
    raw_hash = file_sha256(csv_path)
    frame = pd.read_csv(csv_path)
    # cate_id and domain_indicator are derived below and need not exist in the
    # raw MovieLens table.
    required = {"user_id", "movie_id", "gender", "age", "occupation", "zip",
                "genres", "rating"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("MovieLens source lacks required columns: {}".format(missing))
    if len(frame) < 20:
        raise ValueError("MovieLens source needs at least 20 ordered rows for 80/10/10 split")

    frame = frame.copy()
    frame["cate_id"] = frame["genres"].astype(str).str.split("|").str[0]
    domain_lookup = {age: index for index, group in enumerate(AGE_GROUPS) for age in group}
    frame["domain_indicator"] = pd.to_numeric(frame["age"], errors="raise").map(domain_lookup)
    if frame["domain_indicator"].isna().any():
        invalid = sorted(frame.loc[frame["domain_indicator"].isna(), "age"].unique().tolist())
        raise ValueError("MovieLens contains age values outside frozen domain groups: {}".format(invalid))
    rating = pd.to_numeric(frame["rating"], errors="raise")
    if not np.all(np.equal(rating.to_numpy(), np.floor(rating.to_numpy()))):
        raise ValueError("MovieLens rating must be integer-valued")
    frame["click"] = (rating > 3).astype(np.float32)
    frame["like"] = (rating > 4).astype(np.float32)
    train_idx, validation_idx, test_idx = _split_indices(len(frame))

    # Fit categorical vocabularies and the age scaler on the global base-train
    # prefix only.  No scope-specific transform is permitted.
    vocabularies: Dict[str, Dict[str, int]] = {}
    for column in MOVIELENS_SPARSE_COLUMNS:
        if column != "domain_indicator":
            vocabularies[column] = _fit_vocab(frame.iloc[train_idx][column])
    ages = pd.to_numeric(frame["age"], errors="raise").astype(np.float64)
    age_min = float(ages.iloc[train_idx].min())
    age_max = float(ages.iloc[train_idx].max())
    age_span = age_max - age_min

    encoded = np.zeros((len(frame), len(MOVIELENS_COLUMNS)), dtype=np.float32)
    for column_index, column in enumerate(MOVIELENS_COLUMNS):
        if column == "age":
            encoded[:, column_index] = (0.0 if age_span == 0 else
                                        ((ages.to_numpy() - age_min) / age_span).astype(np.float32))
        elif column == "domain_indicator":
            encoded[:, column_index] = frame[column].to_numpy(dtype=np.int64).astype(np.float32)
        else:
            encoded[:, column_index] = _apply_vocab(
                frame[column], vocabularies[column]).astype(np.float32)
    labels = frame[["click", "like"]].to_numpy(dtype=np.float32)
    schema = _feature_schema(
        "movielens", MOVIELENS_COLUMNS, ("age",), MOVIELENS_SPARSE_COLUMNS,
        {column: (3 if column == "domain_indicator" else len(vocabularies[column]) + 1)
         for column in MOVIELENS_SPARSE_COLUMNS},
        vocabularies, {"age_min": age_min, "age_max": age_max})

    def take(index: np.ndarray):
        return (np.asarray(encoded[index], dtype=np.float32),
                np.asarray(labels[index], dtype=np.float32), index.copy())

    train_x, train_y, train_ids = take(train_idx)
    val_x, val_y, val_ids = take(validation_idx)
    test_x, test_y, test_ids = take(test_idx)
    base_split_fingerprints = {
        "train": split_row_fingerprint(train_x, train_y, train_ids),
        "validation": split_row_fingerprint(val_x, val_y, val_ids),
        "test": split_row_fingerprint(test_x, test_y, test_ids),
    }
    bundle = DataBundle(
        dataset="movielens", setting="natural", seed=int(seed), features=schema,
        train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y, train_row_ids=train_ids,
        val_row_ids=val_ids, test_row_ids=test_ids, domain_num=3, task_num=2,
        local_to_physical=(0, 1, 2), role_by_domain={},
        base_split_fingerprints=base_split_fingerprints,
        split_row_id_hashes={"train": row_ids_hash(train_ids),
                             "validation": row_ids_hash(val_ids),
                             "test": row_ids_hash(test_ids)},
        preprocessing_metadata={
            "fit_scope": "global_base_training_prefix_only",
            "split_rule": "global_row_order_80_10_10",
            "training_pool": "complete_base_train_prefix_natural",
            "label_definitions": {"task0": "1[rating>3]", "task1": "1[rating>4]"},
            "physical_domains": list(MOVIELENS_DOMAIN_IDENTITIES),
            "age_groups": [sorted(values) for values in AGE_GROUPS],
            "raw_file": csv_path,
            "raw_file_size": int(os.path.getsize(csv_path)),
            "raw_file_sha256": raw_hash,
            "base_train_row_ids_hash": row_ids_hash(train_ids),
            "global_base_train_size": int(len(train_ids)),
            "user_ids_present": True,
        },
    )
    expected_domain_ids = {0, 1, 2}
    if set(np.unique(train_x[:, -1].astype(np.int64)).tolist()) != expected_domain_ids:
        raise ValueError("natural MovieLens base train must contain all three physical domains")
    return bundle


def make_sdmt_view(joint: DataBundle, physical_domain: int) -> DataBundle:
    """Filter one physical domain and remap routing while keeping global schema."""

    physical_domain = int(physical_domain)
    if joint.dataset != "movielens" or joint.setting != "natural":
        raise ValueError("F0-B1 SDMT views require a MovieLens natural joint bundle")
    if joint.domain_num != 3 or joint.local_to_physical != (0, 1, 2):
        raise ValueError("SDMT source must be the full three-domain joint bundle")
    if physical_domain not in (0, 1, 2):
        raise ValueError("physical domain must be one of 0, 1, 2")

    def filter_split(x_values: np.ndarray, y_values: np.ndarray, row_ids: np.ndarray):
        mask = x_values[:, -1].astype(np.int64) == physical_domain
        view_x = np.array(x_values[mask], copy=True)
        view_x[:, -1] = 0
        return view_x, np.array(y_values[mask], copy=True), np.array(row_ids[mask], copy=True)

    train_x, train_y, train_ids = filter_split(joint.train_x, joint.train_y, joint.train_row_ids)
    val_x, val_y, val_ids = filter_split(joint.val_x, joint.val_y, joint.val_row_ids)
    test_x, test_y, test_ids = filter_split(joint.test_x, joint.test_y, joint.test_row_ids)
    if not len(train_x) or not len(val_x) or not len(test_x):
        raise ValueError("physical domain {} is empty in a split".format(physical_domain))
    # Preserve the entire global feature schema, especially the domain
    # embedding's 3-row shape.  Only the routing values are local zero.
    schema = copy.deepcopy(joint.features)
    view = DataBundle(
        dataset=joint.dataset, setting=joint.setting, seed=joint.seed, features=schema,
        train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y, train_row_ids=train_ids,
        val_row_ids=val_ids, test_row_ids=test_ids, domain_num=1, task_num=2,
        local_to_physical=(physical_domain,), role_by_domain={0: "natural"},
        base_split_fingerprints=dict(joint.base_split_fingerprints),
        split_row_id_hashes={"train": row_ids_hash(train_ids),
                             "validation": row_ids_hash(val_ids),
                             "test": row_ids_hash(test_ids)},
        preprocessing_metadata={**joint.preprocessing_metadata,
                                "scope": "SDMT-d{}".format(physical_domain),
                                "physical_domain": physical_domain,
                                "routing_domain_remap": {physical_domain: 0},
                                "global_embedding_schema_preserved": True,
                                "selected_training_row_ids_hash": row_ids_hash(train_ids)},
    )
    assert_sdmt_pairing(joint, view, physical_domain)
    return view


def assert_sdmt_pairing(joint: DataBundle, view: DataBundle,
                        physical_domain: int) -> Dict[str, str]:
    """Assert IDs, labels, features, and routing agree with a joint-domain slice."""

    hashes: Dict[str, str] = {}
    for split_name, joint_x, joint_y, joint_ids, view_x, view_y, view_ids in (
            ("train", joint.train_x, joint.train_y, joint.train_row_ids,
             view.train_x, view.train_y, view.train_row_ids),
            ("validation", joint.val_x, joint.val_y, joint.val_row_ids,
             view.val_x, view.val_y, view.val_row_ids),
            ("test", joint.test_x, joint.test_y, joint.test_row_ids,
             view.test_x, view.test_y, view.test_row_ids)):
        mask = joint_x[:, -1].astype(np.int64) == int(physical_domain)
        expected_x = np.array(joint_x[mask], copy=True)
        expected_x[:, -1] = 0
        if not (np.array_equal(expected_x, view_x) and
                np.array_equal(joint_y[mask], view_y) and
                np.array_equal(joint_ids[mask], view_ids)):
            raise AssertionError("SDMT {} pairing differs from physical domain {}".format(
                split_name, physical_domain))
        hashes[split_name] = split_row_fingerprint(expected_x, joint_y[mask], joint_ids[mask])
    if joint.features.vocab_sizes != view.features.vocab_sizes:
        raise AssertionError("SDMT feature vocabularies must retain global embedding shape")
    return hashes


def load_b1_bundles(dataset_path: Optional[str] = None, seed: int = 2022):
    joint = load_movielens_natural(dataset_path, seed)
    return {"mdmt": joint,
            **{"sdmt-d{}".format(d): make_sdmt_view(joint, d) for d in (0, 1, 2)}}


def split_hashes(bundle: DataBundle) -> Dict[str, str]:
    return {"train": bundle.split_fingerprints()["train"],
            "validation": bundle.split_fingerprints()["validation"],
            "test": bundle.split_fingerprints()["test"],
            "train_row_ids": bundle.row_id_fingerprints()["train"],
            "validation_row_ids": bundle.row_id_fingerprints()["validation"],
            "test_row_ids": bundle.row_id_fingerprints()["test"]}


__all__ = ["DEFAULT_MOVIELENS_PATH", "file_sha256", "load_movielens_natural",
           "make_sdmt_view", "assert_sdmt_pairing", "load_b1_bundles", "split_hashes"]
