"""Canonical fixed1 adapters and Train/Validation-only source reproduction."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import forward_logits
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02 import protocol as ema02
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.data import joint_counts, sha_file
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31 import audited_data as raw

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "artifacts/conditional_history_shared_update_20261122"
DATASETS = ("movielens", "kuairand")
setup = v3.setup
update_ema = v3.update_ema
state_digest = v3.state_digest
rng_state = v3.rng_state
restore_rng = v3.restore_rng


@dataclass(frozen=True)
class TrainValidation:
    features: raw.FeatureSchema
    train_x: np.ndarray
    train_y: np.ndarray
    train_row_ids: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_row_ids: np.ndarray


def config(dataset):
    return v3.config("A0", dataset)


def make_model(bundle, dataset):
    return v3.make_model(bundle, v3.load_protocol()["datasets"][dataset]["source"])


def optimizer(model, cfg):
    return ema02.make_optimizer(model, cfg)


def _movielens(path, train_end, val_end):
    data = pd.read_csv(path, nrows=val_end)
    if len(data) != val_end:
        raise ValueError("MovieLens ended before the frozen Validation cutoff")
    data["cate_id"] = data["genres"].astype(str).str.split("|").str[0]
    groups = ({1, 18}, {25}, {35, 45, 50, 56})
    domains = data["age"].map({age: d for d, ages in enumerate(groups) for age in ages})
    if domains.isna().any():
        raise ValueError("Unexpected MovieLens age")
    columns = raw.MOVIELENS_COLUMNS
    sparse = raw.MOVIELENS_SPARSE_COLUMNS
    vocabs = {c: raw._fit_vocab(data[c].iloc[:train_end]) for c in sparse if c != "domain_indicator"}
    age = pd.to_numeric(data["age"], errors="raise").astype(np.float64)
    age_min = float(age.iloc[:train_end].min())
    age_max = float(age.iloc[:train_end].max())
    span = age_max - age_min
    x = np.zeros((val_end, len(columns)), dtype=np.float32)
    for index, name in enumerate(columns):
        if name == "age":
            x[:, index] = (age.to_numpy() - age_min) / span if span else 0.0
        elif name == "domain_indicator":
            x[:, index] = domains.to_numpy()
        else:
            x[:, index] = raw._apply_vocab(data[name], vocabs[name])
    rating = data["rating"].astype(int)
    y = np.column_stack((rating > 3, rating > 4)).astype(np.float32)
    features = raw._feature_schema(
        "movielens", columns, ("age",), sparse,
        {c: 3 if c == "domain_indicator" else len(vocabs[c]) + 1 for c in sparse},
        vocabs, {"age_min": age_min, "age_max": age_max},
    )
    ids = np.arange(val_end, dtype=np.int64)
    return TrainValidation(features, x[:train_end], y[:train_end], ids[:train_end],
                           x[train_end:], y[train_end:], ids[train_end:])


def _kuairand(path, train_end, val_end):
    categorical = [name for name in raw.KUAIRAND_SOURCE_COLUMNS if name != "tab"]
    required = list(raw.KUAIRAND_SOURCE_COLUMNS) + ["is_click", "long_view"]
    strings = {
        "user_active_degree", "follow_user_num_range", "fans_user_num_range",
        "friend_user_num_range", "register_days_range", "video_type",
        "upload_dt", "upload_type", "tag",
    }
    types = {"tab": "int8", "is_click": "int8", "long_view": "int8"}
    for name in categorical:
        if name in strings:
            types[name] = "string"
        elif name.startswith("onehot_feat") or name in ("server_width", "server_height", "music_type"):
            types[name] = "float32"
        else:
            types[name] = "int32"
    options = dict(usecols=required, dtype=types, chunksize=250_000)
    values = {name: set() for name in categorical}
    with pd.read_csv(path, nrows=train_end, **options) as reader:
        for chunk in reader:
            for name in categorical:
                values[name].update(raw._safe_value(value) for value in chunk[name].tolist())
    vocabs = {name: {value: i + 1 for i, value in enumerate(sorted(pool))}
              for name, pool in values.items()}

    columns = raw.KUAIRAND_FEATURE_COLUMNS
    x = np.empty((val_end, len(columns)), dtype=np.float32)
    y = np.empty((val_end, 2), dtype=np.float32)
    offset = 0
    with pd.read_csv(path, nrows=val_end, **options) as reader:
        for chunk in reader:
            end = offset + len(chunk)
            domains = chunk["tab"].map({0: 0, 1: 1, 4: 2})
            if domains.isna().any():
                raise ValueError("Unexpected KuaiRand tab")
            for index, name in enumerate(columns):
                x[offset:end, index] = (domains.to_numpy() if name == "domain_indicator"
                                        else raw._apply_vocab(chunk[name], vocabs[name]))
            y[offset:end] = chunk[["is_click", "long_view"]].to_numpy(dtype=np.float32)
            offset = end
    if offset != val_end:
        raise ValueError("KuaiRand ended before the frozen Validation cutoff")
    # The frozen raw SHA identifies a file whose source tab check admits every row.
    # The positional cutoff still precedes source domain validation and mapping.
    features = raw._feature_schema(
        "kuairand", columns, (), raw.KUAIRAND_SPARSE_COLUMNS,
        {c: 3 if c == "domain_indicator" else len(vocabs[c]) + 1 for c in columns},
        vocabs, {},
    )
    ids = np.arange(val_end, dtype=np.int64)
    return TrainValidation(features, x[:train_end], y[:train_end], ids[:train_end],
                           x[train_end:], y[train_end:], ids[train_end:])


def _dataset_identity(dataset, path, bundle, raw_hash):
    return dict(
        dataset=dataset,
        split_id="original",
        raw_path=str(path.resolve()),
        raw_sha256=raw_hash,
        array_hashes={name: raw.array_hash(getattr(bundle, name))
                      for name in ("train_x", "train_y", "val_x", "val_y")},
        split_row_fingerprints={
            "train": raw.split_row_fingerprint(bundle.train_x, bundle.train_y, bundle.train_row_ids),
            "validation": raw.split_row_fingerprint(bundle.val_x, bundle.val_y, bundle.val_row_ids),
        },
        split_row_id_hashes={"train": raw.row_ids_hash(bundle.train_row_ids),
                             "validation": raw.row_ids_hash(bundle.val_row_ids)},
        encoder_hash=bundle.features.encoder_hash,
        scaler_hash=bundle.features.scaler_hash,
        features=bundle.features.as_dict(),
        train_domain_counts=np.bincount(bundle.train_x[:, -1].astype(np.int64), minlength=3).tolist(),
        split_rows=dict(train=len(bundle.train_y), validation=len(bundle.val_y)),
        train_joint_labels=joint_counts(bundle.train_x, bundle.train_y),
        preprocessing_fit="train_only",
    )


def load_data(dataset):
    conf = v3.load_protocol()["datasets"][dataset]
    frozen = conf["source"]
    path = Path(conf["csv"])
    raw_hash = sha_file(path)
    if raw_hash != frozen["raw_sha256"]:
        raise ValueError(f"{dataset}: frozen raw SHA-256 mismatch")
    receipt_path = RECEIPTS / f"{dataset}_data_identity.json"
    receipt = json.loads(receipt_path.read_bytes())
    total_rows = sum(frozen["split_rows"].values())
    train_end, val_end = int(total_rows * .8), int(total_rows * .9)
    loader = {"movielens": _movielens, "kuairand": _kuairand}[dataset]
    bundle = loader(path, train_end, val_end)
    if not all(np.isin(labels, (0, 1)).all() for labels in (bundle.train_y, bundle.val_y)):
        raise ValueError(f"{dataset}: expected complete binary source labels")
    identity = _dataset_identity(dataset, path, bundle, raw_hash)
    for expected in (frozen, receipt):
        for name, actual in identity.items():
            if name not in expected:
                continue
            value = expected[name]
            if isinstance(actual, dict):
                value = {key: value[key] for key in actual}
            if actual != value:
                raise ValueError(f"{dataset}: frozen Train/Validation mismatch in {name}")
    identity["raw_boundaries"] = dict(total_rows=total_rows, train_end=train_end, validation_end=val_end)
    identity["sealed_test_provenance"] = dict(
        status="SEALED", decoded=False,
        receipt=str(receipt_path.relative_to(ROOT)), receipt_sha256=sha_file(receipt_path),
        rows=receipt["split_rows"]["test"],
        array_hashes={key: receipt["array_hashes"][key] for key in ("test_x", "test_y")},
        split_row_fingerprint=receipt["split_row_fingerprints"]["test"],
        row_ids_hash=receipt["split_row_id_hashes"]["test"],
    )
    identity["data_hash"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return bundle, identity
