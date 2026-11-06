"""Leakage-free Stage S1 data construction and pairing utilities.

The historical example loaders fit encoders/scalers before splitting and
construct single-domain data by loading a different file.  Stage S1 keeps a
single, immutable base split and derives every rotation/reference from it.
This file owns that contract; it deliberately has no model or training code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .source_protocol import (
    KUAIRAND_SETTINGS,
    ROLE_MAPS,
    ROLE_SIZES,
    canonical_dataset,
    dataset_label,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MOVIELENS_PATH = "/home/xzf/data/ml-1m"
DEFAULT_KUAIRAND_PATH = "/home/xzf/data/kuairand/processed"

MOVIELENS_COLUMNS = (
    "user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id",
    "domain_indicator",
)
MOVIELENS_SPARSE_COLUMNS = (
    "user_id", "movie_id", "gender", "occupation", "zip", "cate_id",
    "domain_indicator",
)
KUAIRAND_SOURCE_COLUMNS = (
    "tab", "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
    "register_days_range", "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9", "onehot_feat10",
    "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
    "onehot_feat15", "onehot_feat16", "onehot_feat17", "video_type",
    "upload_dt", "upload_type", "server_width", "server_height", "music_type",
    "tag", "counts",
)
KUAIRAND_FEATURE_COLUMNS = tuple(c for c in KUAIRAND_SOURCE_COLUMNS if c != "tab") + (
    "domain_indicator",
)
KUAIRAND_SPARSE_COLUMNS = KUAIRAND_FEATURE_COLUMNS
MOVIELENS_DOMAIN_IDENTITIES = ("ages_1_18", "age_25", "ages_35_45_50_56")


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def split_row_fingerprint(x: np.ndarray, y: np.ndarray, row_ids: Optional[np.ndarray] = None) -> str:
    """Hash ordered rows, labels, and (when supplied) original row IDs."""

    h = hashlib.sha256()
    for value in (x, y, row_ids if row_ids is not None else np.asarray([], dtype=np.int64)):
        arr = np.ascontiguousarray(np.asarray(value))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def row_ids_hash(row_ids: np.ndarray) -> str:
    return array_hash(np.asarray(row_ids, dtype=np.int64))


def role_reversal_sample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    setting: str,
    seed: int,
    sizes: Mapping[str, int] = ROLE_SIZES,
    row_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen MovieLens role rotation exactly as Stage S0 did.

    If ``row_ids`` is supplied it is returned alongside the selected arrays;
    this is useful for audits and does not alter the selected rows.
    """

    if setting not in ("d0_large", "d1_large", "d2_large"):
        raise ValueError("MovieLens role setting must be d0_large/d1_large/d2_large")
    x_values = np.asarray(x_train)
    y_values = np.asarray(y_train)
    ids = x_values[:, -1].astype(np.int64)
    keep = np.zeros(len(x_values), dtype=bool)
    for domain, role in ROLE_MAPS[setting].items():
        pool = np.flatnonzero(ids == int(domain))
        requested = int(sizes[role])
        if requested > len(pool):
            raise ValueError("requested {} rows from domain {} but only {} are available".format(
                requested, domain, len(pool)))
        order = np.random.RandomState(int(seed) * 10 + int(domain)).permutation(len(pool))
        keep[pool[order[:requested]]] = True
    selected = (x_values[keep], y_values[keep])
    if row_ids is None:
        return selected
    return selected + (np.asarray(row_ids)[keep],)


def _safe_value(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<NA>"
    return str(value)


def _fit_vocab(values: pd.Series) -> Dict[str, int]:
    # Sorting makes the encoder independent of pandas' hash iteration order.
    unique = sorted({_safe_value(value) for value in values.tolist()})
    return {value: index + 1 for index, value in enumerate(unique)}


def _apply_vocab(values: pd.Series, vocabulary: Mapping[str, int]) -> np.ndarray:
    # Index zero is reserved for values not observed in the global base train.
    return np.asarray([int(vocabulary.get(_safe_value(value), 0)) for value in values], dtype=np.int64)


def _entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    _, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(np.float64) / float(arr.size)
    return float(-(probs * np.log2(probs)).sum())


def _cell_stats(x: np.ndarray, y: np.ndarray, domain_num: int, task_num: int) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    ids = np.asarray(x[:, -1]).astype(np.int64)
    for task in range(task_num):
        for domain in range(domain_num):
            key = "d{}t{}".format(domain, task)
            mask = ids == domain
            labels = np.asarray(y[mask, task], dtype=np.float64)
            result[key] = {
                "count": int(mask.sum()),
                "positive_count": int(labels.sum()) if labels.size else 0,
                "positive_rate": float(labels.mean()) if labels.size else float("nan"),
                "entropy": _entropy(labels),
            }
    return result


@dataclass
class FeatureSchema:
    """Serializable schema consumed by the three Stage S1 constructors."""

    dataset: str
    columns: Tuple[str, ...]
    dense_columns: Tuple[str, ...]
    sparse_columns: Tuple[str, ...]
    vocab_sizes: Dict[str, int]
    embedding_dim: int = 16
    domain_column: str = "domain_indicator"
    encoder_hash: str = ""
    scaler_hash: str = ""
    oov_index: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "columns": list(self.columns),
            "dense_columns": list(self.dense_columns),
            "sparse_columns": list(self.sparse_columns),
            "vocab_sizes": dict(self.vocab_sizes),
            "embedding_dim": int(self.embedding_dim),
            "domain_column": self.domain_column,
            "encoder_hash": self.encoder_hash,
            "scaler_hash": self.scaler_hash,
            "oov_index": int(self.oov_index),
        }


@dataclass
class DataBundle:
    """One deterministic joint view plus its derived single-domain views."""

    dataset: str
    setting: str
    seed: int
    features: FeatureSchema
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_row_ids: np.ndarray
    val_row_ids: np.ndarray
    test_row_ids: np.ndarray
    domain_num: int = 3
    task_num: int = 2
    local_to_physical: Tuple[int, ...] = (0, 1, 2)
    role_by_domain: Dict[int, str] = field(default_factory=dict)
    base_split_fingerprints: Dict[str, str] = field(default_factory=dict)
    split_row_id_hashes: Dict[str, str] = field(default_factory=dict)
    preprocessing_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_joint(self) -> bool:
        return self.domain_num == 3 and self.local_to_physical == (0, 1, 2)

    @property
    def train_counts(self) -> Dict[int, int]:
        ids = self.train_x[:, -1].astype(np.int64)
        return {int(d): int((ids == d).sum()) for d in range(self.domain_num)}

    @property
    def domain_shares(self) -> Dict[int, float]:
        counts = self.train_counts
        total = float(sum(counts.values()))
        return {int(d): (counts[d] / total if total else float("nan")) for d in counts}

    @property
    def physical_domain(self) -> Optional[int]:
        return None if self.is_joint else int(self.local_to_physical[0])

    def _view_fingerprint(self, x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> str:
        return split_row_fingerprint(x, y, ids)

    def split_fingerprints(self) -> Dict[str, str]:
        return {
            "train": self._view_fingerprint(self.train_x, self.train_y, self.train_row_ids),
            "validation": self._view_fingerprint(self.val_x, self.val_y, self.val_row_ids),
            "test": self._view_fingerprint(self.test_x, self.test_y, self.test_row_ids),
        }

    def row_id_fingerprints(self) -> Dict[str, str]:
        return {
            "train": row_ids_hash(self.train_row_ids),
            "validation": row_ids_hash(self.val_row_ids),
            "test": row_ids_hash(self.test_row_ids),
        }

    def cell_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            "train": _cell_stats(self.train_x, self.train_y, self.domain_num, self.task_num),
            "validation": _cell_stats(self.val_x, self.val_y, self.domain_num, self.task_num),
            "test": _cell_stats(self.test_x, self.test_y, self.domain_num, self.task_num),
        }

    def reference_view(self, physical_domain: int) -> "DataBundle":
        """Filter an existing joint view and remap only the routing column."""

        physical_domain = int(physical_domain)
        if physical_domain not in (0, 1, 2):
            raise ValueError("physical_domain must be 0, 1, or 2")
        if self.domain_num != 3:
            raise ValueError("reference_view must be called on the joint bundle")

        def one_split(x: np.ndarray, y: np.ndarray, ids: np.ndarray):
            mask = x[:, -1].astype(np.int64) == physical_domain
            view_x = np.array(x[mask], copy=True)
            view_x[:, -1] = 0
            return view_x, np.array(y[mask], copy=True), np.array(ids[mask], copy=True)

        train_x, train_y, train_ids = one_split(self.train_x, self.train_y, self.train_row_ids)
        val_x, val_y, val_ids = one_split(self.val_x, self.val_y, self.val_row_ids)
        test_x, test_y, test_ids = one_split(self.test_x, self.test_y, self.test_row_ids)
        schema = copy.deepcopy(self.features)
        # The local routing feature has one used value (row 0).  Other feature
        # vocabularies stay global so reference construction cannot change the
        # representation of an example.
        schema.vocab_sizes = dict(schema.vocab_sizes)
        schema.vocab_sizes[schema.domain_column] = 1
        schema.encoder_hash = _json_hash({"base": self.features.encoder_hash,
                                          "domain_remap": physical_domain})
        result = DataBundle(
            dataset=self.dataset, setting=self.setting, seed=self.seed,
            features=schema, train_x=train_x, train_y=train_y,
            val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y,
            train_row_ids=train_ids, val_row_ids=val_ids, test_row_ids=test_ids,
            domain_num=1, task_num=self.task_num, local_to_physical=(physical_domain,),
            role_by_domain={0: self.role_by_domain.get(physical_domain, "")},
            base_split_fingerprints=dict(self.base_split_fingerprints),
            split_row_id_hashes={
                "train": row_ids_hash(train_ids), "validation": row_ids_hash(val_ids),
                "test": row_ids_hash(test_ids),
            },
            preprocessing_metadata={**self.preprocessing_metadata,
                                    "reference_physical_domain": physical_domain,
                                    "routing_domain_remap": {physical_domain: 0}},
        )
        result.preprocessing_metadata["joint_view_fingerprints"] = {
            "train": split_row_fingerprint(
                self.train_x[self.train_x[:, -1].astype(np.int64) == physical_domain],
                self.train_y[self.train_x[:, -1].astype(np.int64) == physical_domain],
                self.train_row_ids[self.train_x[:, -1].astype(np.int64) == physical_domain]),
            "validation": split_row_fingerprint(
                self.val_x[self.val_x[:, -1].astype(np.int64) == physical_domain],
                self.val_y[self.val_x[:, -1].astype(np.int64) == physical_domain],
                self.val_row_ids[self.val_x[:, -1].astype(np.int64) == physical_domain]),
            "test": split_row_fingerprint(
                self.test_x[self.test_x[:, -1].astype(np.int64) == physical_domain],
                self.test_y[self.test_x[:, -1].astype(np.int64) == physical_domain],
                self.test_row_ids[self.test_x[:, -1].astype(np.int64) == physical_domain]),
        }
        result.preprocessing_metadata["remapped_view_fingerprints"] = result.split_fingerprints()
        # Equality is asserted on views with the routing column remapped to
        # local zero.  Keeping both hashes makes the physical-domain source
        # and the reference artifact independently auditable.
        pair_fingerprints: Dict[str, Dict[str, str]] = {}
        for split_name, joint_x, joint_y, joint_ids, ref_x, ref_y, ref_ids in (
                ("train", self.train_x, self.train_y, self.train_row_ids,
                 result.train_x, result.train_y, result.train_row_ids),
                ("validation", self.val_x, self.val_y, self.val_row_ids,
                 result.val_x, result.val_y, result.val_row_ids),
                ("test", self.test_x, self.test_y, self.test_row_ids,
                 result.test_x, result.test_y, result.test_row_ids)):
            mask = joint_x[:, -1].astype(np.int64) == physical_domain
            remapped_joint = np.array(joint_x[mask], copy=True)
            remapped_joint[:, -1] = 0
            pair_fingerprints[split_name] = {
                "joint_remapped": split_row_fingerprint(
                    remapped_joint, joint_y[mask], joint_ids[mask]),
                "reference": split_row_fingerprint(ref_x, ref_y, ref_ids),
            }
            if pair_fingerprints[split_name]["joint_remapped"] != \
                    pair_fingerprints[split_name]["reference"]:
                raise AssertionError("reference pairing fingerprint mismatch in {}".format(split_name))
        result.preprocessing_metadata["pairing_fingerprints"] = pair_fingerprints
        return result

    def manifest_fields(self) -> Dict[str, object]:
        physical_views: Dict[str, Dict[str, str]] = {}
        for local_domain, physical_domain in enumerate(self.local_to_physical):
            split_views: Dict[str, str] = {}
            for split_name, x_values, y_values, row_ids in (
                    ("train", self.train_x, self.train_y, self.train_row_ids),
                    ("validation", self.val_x, self.val_y, self.val_row_ids),
                    ("test", self.test_x, self.test_y, self.test_row_ids)):
                mask = x_values[:, -1].astype(np.int64) == int(local_domain)
                view_x = np.array(x_values[mask], copy=True)
                view_x[:, -1] = 0
                split_views[split_name] = split_row_fingerprint(
                    view_x, y_values[mask], row_ids[mask])
            physical_views[str(physical_domain)] = split_views
        return {
            "dataset": dataset_label(self.dataset),
            "dataset_key": self.dataset,
            "setting": self.setting,
            "seed": int(self.seed),
            "domain_num": int(self.domain_num),
            "task_num": int(self.task_num),
            "local_to_physical": list(self.local_to_physical),
            "physical_domain": self.physical_domain,
            "physical_domain_identity": [
                (("tab0", "tab1", "tab4")[int(value)] if self.dataset == "kuairand"
                 else MOVIELENS_DOMAIN_IDENTITIES[int(value)])
                for value in self.local_to_physical
            ],
            "role_by_domain": {str(k): v for k, v in self.role_by_domain.items()},
            "features": self.features.as_dict(),
            "split_row_fingerprints": self.split_fingerprints(),
            "split_row_id_hashes": self.row_id_fingerprints(),
            "base_split_fingerprints": dict(self.base_split_fingerprints),
            "base_split_row_id_hashes": dict(self.split_row_id_hashes),
            "train_domain_counts": {str(k): v for k, v in self.train_counts.items()},
            "domain_shares": {str(k): v for k, v in self.domain_shares.items()},
            "cell_stats": self.cell_stats(),
            "preprocessing_metadata": self.preprocessing_metadata,
            "physical_view_fingerprints": physical_views,
        }


def _feature_schema(dataset: str, columns: Sequence[str], dense: Sequence[str],
                    sparse: Sequence[str], vocab_sizes: Mapping[str, int],
                    encoder_payload: object, scaler_payload: object) -> FeatureSchema:
    return FeatureSchema(
        dataset=dataset, columns=tuple(columns), dense_columns=tuple(dense),
        sparse_columns=tuple(sparse), vocab_sizes={str(k): int(v) for k, v in vocab_sizes.items()},
        encoder_hash=_json_hash(encoder_payload), scaler_hash=_json_hash(scaler_payload),
    )


def _split_indices(length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(length * 0.8)
    val_end = int(length * 0.9)
    ids = np.arange(length, dtype=np.int64)
    return ids[:train_end], ids[train_end:val_end], ids[val_end:]


def _load_movielens_base(path: str, setting: str, seed: int) -> DataBundle:
    csv_path = path if str(path).lower().endswith(".csv") else os.path.join(path, "ml-1m.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("MovieLens file not found: {}".format(csv_path))
    data = pd.read_csv(csv_path)
    if "genres" not in data or "rating" not in data or "age" not in data:
        raise ValueError("MovieLens input lacks required genres/rating/age columns")
    data = data.copy()
    data["cate_id"] = data["genres"].astype(str).str.split("|").str[0]
    groups = ({1, 18}, {25}, {35, 45, 50, 56})
    group_lookup = {age: index for index, group in enumerate(groups) for age in group}
    data["domain_indicator"] = data["age"].map(group_lookup)
    if data["domain_indicator"].isna().any():
        raise ValueError("MovieLens contains an age outside the frozen physical-domain groups")
    data["click"] = (data["rating"].astype(int) > 3).astype(np.float32)
    data["like"] = (data["rating"].astype(int) > 4).astype(np.float32)
    train_idx, val_idx, test_idx = _split_indices(len(data))

    dense = ("age",)
    sparse = tuple(MOVIELENS_SPARSE_COLUMNS)
    vocabularies: Dict[str, Dict[str, int]] = {}
    for column in sparse:
        if column == "domain_indicator":
            continue
        vocabularies[column] = _fit_vocab(data.iloc[train_idx][column])
    age_values = pd.to_numeric(data["age"], errors="raise").astype(np.float64)
    age_min = float(age_values.iloc[train_idx].min())
    age_max = float(age_values.iloc[train_idx].max())
    age_span = age_max - age_min
    scaler_payload = {"age_min": age_min, "age_max": age_max}

    encoded = np.zeros((len(data), len(MOVIELENS_COLUMNS)), dtype=np.float32)
    for column_index, column in enumerate(MOVIELENS_COLUMNS):
        if column == "age":
            if age_span == 0:
                encoded[:, column_index] = 0.0
            else:
                encoded[:, column_index] = ((age_values.to_numpy() - age_min) / age_span).astype(np.float32)
        elif column == "domain_indicator":
            encoded[:, column_index] = data[column].astype(np.int64).to_numpy().astype(np.float32)
        else:
            encoded[:, column_index] = _apply_vocab(data[column], vocabularies[column]).astype(np.float32)
    labels = data[["click", "like"]].to_numpy(dtype=np.float32)
    schema = _feature_schema(
        "movielens", MOVIELENS_COLUMNS, dense, sparse,
        {column: (3 if column == "domain_indicator" else len(vocabularies[column]) + 1)
         for column in MOVIELENS_SPARSE_COLUMNS},
        vocabularies, scaler_payload)

    def take(indices: np.ndarray):
        return encoded[indices], labels[indices], indices.copy()

    base_train_x, base_train_y, base_train_ids = take(train_idx)
    base_val_x, base_val_y, base_val_ids = take(val_idx)
    base_test_x, base_test_y, base_test_ids = take(test_idx)
    sampled_x, sampled_y, sampled_ids = base_train_x, base_train_y, base_train_ids
    assert isinstance(sampled_ids, np.ndarray)
    role_by_domain = {d: "natural" for d in range(3)}
    bundle = DataBundle(
        dataset="movielens", setting=setting, seed=int(seed), features=schema,
        train_x=np.asarray(sampled_x, dtype=np.float32), train_y=np.asarray(sampled_y, dtype=np.float32),
        val_x=np.asarray(base_val_x, dtype=np.float32), val_y=np.asarray(base_val_y, dtype=np.float32),
        test_x=np.asarray(base_test_x, dtype=np.float32), test_y=np.asarray(base_test_y, dtype=np.float32),
        train_row_ids=np.asarray(sampled_ids, dtype=np.int64),
        val_row_ids=np.asarray(base_val_ids, dtype=np.int64), test_row_ids=np.asarray(base_test_ids, dtype=np.int64),
        domain_num=3, task_num=2, local_to_physical=(0, 1, 2), role_by_domain=role_by_domain,
        base_split_fingerprints={
            "train": split_row_fingerprint(base_train_x, base_train_y, base_train_ids),
            "validation": split_row_fingerprint(base_val_x, base_val_y, base_val_ids),
            "test": split_row_fingerprint(base_test_x, base_test_y, base_test_ids),
        },
        split_row_id_hashes={"train": row_ids_hash(sampled_ids),
                             "validation": row_ids_hash(base_val_ids), "test": row_ids_hash(base_test_ids)},
        preprocessing_metadata={
            "fit_scope": "global_base_training_split",
            "split_rule": "global_row_order_80_10_10",
            "label_definitions": {"task0": "1[rating>3]", "task1": "1[rating>4]"},
            "oov_index": 0,
            "selected_training_row_ids_hash": row_ids_hash(sampled_ids),
            "base_train_row_ids_hash": row_ids_hash(base_train_ids),
            "raw_file": os.path.abspath(csv_path),
            "raw_file_size": int(os.path.getsize(csv_path)),
        },
    )
    return bundle


def _load_kuairand_base(path: str, setting: str, seed: int) -> DataBundle:
    if setting != "natural":
        raise ValueError("KuaiRand Stage S1 has only the natural setting")
    csv_path = path if str(path).lower().endswith(".csv") else os.path.join(path, "kuai_rand.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("KuaiRand file not found: {}".format(csv_path))
    required = list(KUAIRAND_SOURCE_COLUMNS) + ["is_click", "long_view"]
    # KuaiRand is large enough that reading the complete CSV as an object
    # DataFrame can exceed the memory budget on Windows.  Two bounded chunk
    # passes preserve row order while keeping the peak DataFrame small: first
    # fit vocabularies on the global base-train prefix, then encode into the
    # three split arrays.
    tab_values = pd.read_csv(csv_path, usecols=["tab"], dtype="int8")["tab"].to_numpy()
    if not set(np.unique(tab_values).tolist()).issubset({0, 1, 4}):
        raise ValueError("KuaiRand contains tabs outside the frozen {0,1,4} domains")
    total_rows = int(len(tab_values))
    train_end = int(total_rows * 0.8)
    val_end = int(total_rows * 0.9)
    columns = KUAIRAND_FEATURE_COLUMNS
    sparse = tuple(KUAIRAND_SPARSE_COLUMNS)
    categorical_columns = [column for column in KUAIRAND_SOURCE_COLUMNS if column != "tab"]
    dtype_map: Dict[str, object] = {"tab": "int8", "is_click": "int8", "long_view": "int8"}
    # Strings are retained only for the bounded chunk; numeric columns are
    # parsed compactly and then represented by the same deterministic vocab.
    string_columns = {
        "user_active_degree", "follow_user_num_range", "fans_user_num_range",
        "friend_user_num_range", "register_days_range", "video_type",
        "upload_dt", "upload_type", "tag",
    }
    for column in categorical_columns:
        if column in string_columns:
            dtype_map[column] = "string"
        elif column.startswith("onehot_feat") or column in ("server_width", "server_height", "music_type"):
            dtype_map[column] = "float32"
        else:
            dtype_map[column] = "int32"
    vocab_sets: Dict[str, set] = {column: set() for column in categorical_columns}
    offset = 0
    for chunk in pd.read_csv(csv_path, usecols=required, dtype=dtype_map, chunksize=250_000):
        end = offset + len(chunk)
        limit = min(end, train_end)
        if offset < limit:
            local = chunk.iloc[:limit - offset]
            for column in categorical_columns:
                vocab_sets[column].update(_safe_value(value) for value in local[column].tolist())
        offset = end
    vocabularies = {
        column: {value: index + 1 for index, value in enumerate(sorted(values))}
        for column, values in vocab_sets.items()
    }
    split_lengths = (train_end, val_end - train_end, total_rows - val_end)
    split_arrays = [np.zeros((length, len(columns)), dtype=np.float32) for length in split_lengths]
    split_labels = [np.zeros((length, 2), dtype=np.float32) for length in split_lengths]
    offset = 0
    domain_lookup = {0: 0, 1: 1, 4: 2}
    for chunk in pd.read_csv(csv_path, usecols=required, dtype=dtype_map, chunksize=250_000):
        end = offset + len(chunk)
        for split_index, (start, stop) in enumerate(((0, train_end), (train_end, val_end),
                                                       (val_end, total_rows))):
            left, right = max(offset, start), min(end, stop)
            if left >= right:
                continue
            local_left, local_right = left - offset, right - offset
            target_left = left - start
            target = split_arrays[split_index][target_left:target_left + (right - left)]
            for column_index, column in enumerate(columns):
                if column == "domain_indicator":
                    target[:, column_index] = chunk.iloc[local_left:local_right]["tab"].map(domain_lookup).to_numpy()
                else:
                    target[:, column_index] = _apply_vocab(
                        chunk.iloc[local_left:local_right][column], vocabularies[column]).astype(np.float32)
            split_labels[split_index][target_left:target_left + (right - left), :] = \
                chunk.iloc[local_left:local_right][["is_click", "long_view"]].to_numpy(dtype=np.float32)
        offset = end
    del tab_values
    schema = _feature_schema(
        "kuairand", columns, (), sparse,
        {column: (3 if column == "domain_indicator" else len(vocabularies[column]) + 1)
         for column in sparse},
        vocabularies, {})

    # Domain filtering is applied only after the global split has been formed.
    split_values = []
    for split_array, split_label, ids in zip(
            split_arrays, split_labels,
            (np.arange(train_end, dtype=np.int64),
             np.arange(train_end, val_end, dtype=np.int64),
             np.arange(val_end, total_rows, dtype=np.int64))):
        mask = np.isin(split_array[:, -1].astype(np.int64), (0, 1, 2))
        split_values.append((split_array[mask], split_label[mask], ids[mask]))
    train_x, train_y, train_ids = split_values[0]
    val_x, val_y, val_ids = split_values[1]
    test_x, test_y, test_ids = split_values[2]
    role_by_domain = dict(ROLE_MAPS["natural"])
    bundle = DataBundle(
        dataset="kuairand", setting="natural", seed=int(seed), features=schema,
        train_x=np.asarray(train_x, dtype=np.float32), train_y=np.asarray(train_y, dtype=np.float32),
        val_x=np.asarray(val_x, dtype=np.float32), val_y=np.asarray(val_y, dtype=np.float32),
        test_x=np.asarray(test_x, dtype=np.float32), test_y=np.asarray(test_y, dtype=np.float32),
        train_row_ids=np.asarray(train_ids, dtype=np.int64),
        val_row_ids=np.asarray(val_ids, dtype=np.int64), test_row_ids=np.asarray(test_ids, dtype=np.int64),
        domain_num=3, task_num=2, local_to_physical=(0, 1, 2), role_by_domain=role_by_domain,
        base_split_fingerprints={
            "train": split_row_fingerprint(train_x, train_y, train_ids),
            "validation": split_row_fingerprint(val_x, val_y, val_ids),
            "test": split_row_fingerprint(test_x, test_y, test_ids),
        },
        split_row_id_hashes={"train": row_ids_hash(train_ids),
                             "validation": row_ids_hash(val_ids), "test": row_ids_hash(test_ids)},
        preprocessing_metadata={
            "fit_scope": "global_base_training_split",
            "split_rule": "global_row_order_80_10_10_then_filter",
            "label_definitions": {"task0": "stored_is_click", "task1": "stored_long_view"},
            "oov_index": 0,
            "selected_training_row_ids_hash": row_ids_hash(train_ids),
            "raw_file": os.path.abspath(csv_path),
            "raw_file_size": int(os.path.getsize(csv_path)),
            "physical_domain_mapping": {"0": "tab0", "1": "tab1", "2": "tab4"},
        },
    )
    expected = {0: 172_782, 1: 1_743_898, 2: 89_452}
    if bundle.train_counts != expected:
        raise ValueError("KuaiRand natural counts mismatch: expected {} observed {}".format(
            expected, bundle.train_counts))
    return bundle


def load_stage_s1_bundle(dataset: str, dataset_path: Optional[str], setting: str,
                         seed: int) -> DataBundle:
    """Load one deterministic joint bundle; references derive from it."""

    dataset = canonical_dataset(dataset)
    path = dataset_path or (DEFAULT_MOVIELENS_PATH if dataset == "movielens"
                            else DEFAULT_KUAIRAND_PATH)
    if dataset == "movielens":
        return _load_movielens_base(path, setting, int(seed))
    return _load_kuairand_base(path, setting, int(seed))


def load_stage_s1_view(dataset: str, dataset_path: Optional[str], setting: str,
                       seed: int, physical_domain: Optional[int] = None) -> DataBundle:
    joint = load_stage_s1_bundle(dataset, dataset_path, setting, seed)
    return joint if physical_domain is None else joint.reference_view(int(physical_domain))


# Short aliases used by focused protocol tests and review scripts.
build_reference_view = DataBundle.reference_view


def load_kuairand_natural(dataset_path: Optional[str] = None, seed: int = 2022) -> DataBundle:
    return load_stage_s1_bundle("kuairand", dataset_path, "natural", seed)


def expected_kuairand_train_counts() -> Dict[int, int]:
    return {0: 172_782, 1: 1_743_898, 2: 89_452}


def expected_movielens_role_counts() -> Dict[str, int]:
    return dict(ROLE_SIZES)


__all__ = [
    "MOVIELENS_COLUMNS", "MOVIELENS_SPARSE_COLUMNS", "KUAIRAND_SOURCE_COLUMNS",
    "KUAIRAND_FEATURE_COLUMNS", "KUAIRAND_SPARSE_COLUMNS", "MOVIELENS_DOMAIN_IDENTITIES", "FeatureSchema",
    "DataBundle", "array_hash", "split_row_fingerprint", "row_ids_hash",
    "role_reversal_sample", "load_stage_s1_bundle", "load_stage_s1_view",
    "build_reference_view", "load_kuairand_natural",
    "expected_kuairand_train_counts", "expected_movielens_role_counts",
]
