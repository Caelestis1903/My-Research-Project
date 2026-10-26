"""Stage E4 KuaiRand weighting and dataset primitives.

This module is the small, isolated contract shared by the E4 joint and
single-domain runners.  It owns the Equal/Sqrt/Sample coefficient formula,
the processed-file loader, the split-first/filter-second oracle view, and the
low-cost shared-gradient observer helpers.  It deliberately contains no
adaptive, temporal, curriculum, EMA, or momentum mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
if M3OE_ROOT not in sys.path:
    sys.path.insert(0, M3OE_ROOT)

from stage_E1_instrumentation import (  # noqa: E402
    EPS,
    CELL_ORDER_NOTE,
    WelfordVector,
    batch_fingerprint,
    build_parameter_partition,
    contribution_views,
    flatten_parameter_tensors,
    iter_cells,
    masked_bce_terms,
    parameter_state_hash,
    preserve_rng_state,
    relative_l2_error,
    rng_state_digest,
    safe_auc,
    safe_logloss,
    write_json,
    write_rows_csv,
)


E4_DATASET_PATH = "/xzf/data/kuairand/processed/kuai_rand.csv"
EXPECTED_SEEDS: Tuple[int, ...] = (2022, 2023, 2024, 2025)
DOMAIN_NUM = 3
TASK_NUM = 2
DOMAIN_TABS = (0, 1, 4)
DOMAIN_NAMES = ("d0", "d1", "d2")
TASK_NAMES = ("is_click", "long_view")

# The path names are stable lower-case keys; the display names are retained in
# every manifest so a report cannot silently conflate a condition with alpha.
WEIGHTING_KEYS: Tuple[str, ...] = ("equal", "sqrt", "sample")
WEIGHTING_DISPLAY: Dict[str, str] = {
    "equal": "Equal", "sqrt": "Sqrt", "sample": "Sample",
}
WEIGHTING_ALPHA: Dict[str, float] = {
    "equal": 0.0, "sqrt": 0.5, "sample": 1.0,
}
ALPHA_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0)

RAW_FEATURE_COLUMNS: Tuple[str, ...] = (
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
SPARSE_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    name for name in RAW_FEATURE_COLUMNS if name != "tab"
) + ("domain_indicator",)
LABEL_COLUMNS: Tuple[str, ...] = ("is_click", "long_view")

# Exact audit facts.  Dictionaries use strings at the JSON boundary but the
# in-memory helpers use integer domain/task ids.
EXPECTED_TOTAL_ROWS = 2_507_666
EXPECTED_SPLIT_COUNTS: Dict[str, Dict[int, int]] = {
    "train": {0: 172_782, 1: 1_743_898, 2: 89_452},
    "validation": {0: 2_692, 1: 246_232, 2: 1_843},
    "test": {0: 2_613, 1: 246_284, 2: 1_870},
}
EXPECTED_POSITIVE_COUNTS: Dict[str, Dict[int, Dict[int, int]]] = {
    "train": {
        0: {0: 15_704, 1: 7_121},
        1: {0: 658_901, 1: 446_583},
        2: {0: 55_180, 1: 43_367},
    },
    "validation": {
        0: {0: 245, 1: 101},
        1: {0: 51_644, 1: 27_460},
        2: {0: 1_063, 1: 829},
    },
    "test": {
        0: {0: 228, 1: 120},
        1: {0: 51_847, 1: 27_415},
        2: {0: 1_100, 1: 827},
    },
}
TRAIN_DOMAIN_SHARES = np.asarray(
    [EXPECTED_SPLIT_COUNTS["train"][d] / float(sum(EXPECTED_SPLIT_COUNTS["train"].values()))
     for d in range(DOMAIN_NUM)], dtype=np.float64)

E4_FORMAL_PROTOCOL: Dict[str, object] = {
    "epoch": 50,
    "batch_size": 40960,
    "learning_rate": 1.0e-2,
    "learning_rate_darts": 1.0e-3,
    "weight_decay": 1.0e-5,
    "expert_num": 4,
    "earlystop_patience": 10,
    "probe_interval": 10,
    "instrument": 1,
    "domain_num": 3,
    "task_num": 2,
    "backbone": "mdmoe",
    "model": "MDMTRec",
    "fcn_dims": [512, 256, 256, 64],
    "exp_d": 1,
    "exp_t": 1,
    "bal_d": 1,
    "bal_t": 1,
    "scheduler": None,
    "sampling": "DataLoader(shuffle=True,num_workers=0,drop_last=False)",
    "split": "global_chronological_row_order_80_10_10",
    "dataset": "KuaiRand-Pure",
    "dataset_path": E4_DATASET_PATH,
    "cell_order": CELL_ORDER_NOTE,
    "weightings": list(WEIGHTING_KEYS),
    "seeds": list(EXPECTED_SEEDS),
    "prohibited_mechanisms": [
        "EMA", "Momentum", "Comp1", "Curriculum", "adaptive_controller",
        "temporal_schedule", "balanced_sampler", "importance_correction",
    ],
}

E4_ORACLE_PROTOCOL: Dict[str, object] = dict(E4_FORMAL_PROTOCOL)
E4_ORACLE_PROTOCOL.update({
    "oracle": True,
    "domain_num": 1,
    "instrument": 0,
    "probe_interval": 0,
    "weighting": "equal_tasks_no_domain_weighting",
    "split": "global_chronological_row_order_80_10_10_then_filter_original_domain",
})


def _normalise_shares(domain_shares: Sequence[float]) -> np.ndarray:
    shares = np.asarray(domain_shares, dtype=np.float64)
    if shares.ndim != 1 or len(shares) != DOMAIN_NUM:
        raise ValueError("domain_shares must contain exactly three values")
    if not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("domain_shares must be finite and strictly positive")
    total = float(shares.sum())
    if total <= 0:
        raise ValueError("domain_shares must have a positive sum")
    return shares / total


def canonical_weighting(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "equal": "equal", "a": "equal", "0": "equal",
        "sqrt": "sqrt", "b": "sqrt", "0.5": "sqrt", "0p5": "sqrt",
        "sample": "sample", "c": "sample", "1": "sample",
    }
    if text not in aliases:
        raise ValueError("weighting must be one of Equal/Sqrt/Sample")
    return aliases[text]


def canonical_alpha(value: object) -> float:
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric") from exc
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    for candidate in ALPHA_GRID:
        if abs(alpha - candidate) <= 1.0e-10:
            return candidate
    raise ValueError("alpha must be one of 0, 0.5, 1")


def weighting_alpha(weighting: object) -> float:
    return WEIGHTING_ALPHA[canonical_weighting(weighting)]


def weighting_display(weighting: object) -> str:
    return WEIGHTING_DISPLAY[canonical_weighting(weighting)]


def weighting_coefficients(
    weighting: object = None,
    domain_shares: Optional[Sequence[float]] = None,
    task_num: int = TASK_NUM,
    *,
    alpha: object = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the sole E4 task-major objective coefficient implementation.

    For ``D`` domains, ``c_d = D p_d**alpha / sum_j p_j**alpha`` and the
    same domain coefficient is copied for every task.  No task-normalization
    or label-prior correction is applied.
    """

    if alpha is not None:
        alpha_value = canonical_alpha(alpha)
        key = next(key for key, value in WEIGHTING_ALPHA.items()
                   if abs(value - alpha_value) <= 1.0e-10)
    else:
        key = canonical_weighting(weighting)
    if domain_shares is None:
        raise ValueError("domain_shares is required")
    alpha = WEIGHTING_ALPHA[key]
    shares = _normalise_shares(domain_shares)
    task_num = int(task_num)
    if task_num <= 0:
        raise ValueError("task_num must be positive")
    powers = np.power(shares, alpha)
    per_domain = len(shares) * powers / float(powers.sum())
    return torch.as_tensor(np.tile(per_domain, task_num), dtype=dtype, device=device)


def weighting_coefficients_numpy(
    weighting: object = None, domain_shares: Optional[Sequence[float]] = None,
    task_num: int = TASK_NUM, *, alpha: object = None,
) -> np.ndarray:
    return weighting_coefficients(weighting, domain_shares, task_num, alpha=alpha,
                                  dtype=torch.float64).detach().cpu().numpy()


def coefficient_table(domain_shares: Sequence[float] = TRAIN_DOMAIN_SHARES) -> Dict[str, List[float]]:
    return {key: weighting_coefficients_numpy(key, domain_shares).tolist()
            for key in WEIGHTING_KEYS}


def validate_weighting_coefficients(
    weighting: object, domain_shares: Sequence[float], coefficients: Sequence[float],
    *, atol: float = 1.0e-8,
) -> Dict[str, object]:
    expected = weighting_coefficients_numpy(weighting, domain_shares)
    actual = (coefficients.detach().cpu().numpy().reshape(-1)
              if isinstance(coefficients, torch.Tensor)
              else np.asarray(coefficients, dtype=np.float64).reshape(-1))
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError("E4 weighting coefficients do not match the registered formula")
    if not np.all(actual > 0):
        raise AssertionError("E4 coefficients must be positive")
    if not np.isclose(float(actual[:DOMAIN_NUM].sum()), DOMAIN_NUM, atol=1.0e-7):
        raise AssertionError("E4 domain coefficient sum must equal three")
    if not np.isclose(float(actual.sum()), DOMAIN_NUM * TASK_NUM, atol=1.0e-7):
        raise AssertionError("E4 six-cell coefficient sum must equal six")
    if not np.allclose(actual[:DOMAIN_NUM], actual[DOMAIN_NUM:], atol=atol, rtol=0):
        raise AssertionError("E4 task-major coefficients must repeat by task")
    return {
        "weighting": canonical_weighting(weighting),
        "display": weighting_display(weighting),
        "alpha": weighting_alpha(weighting),
        "expected": expected.tolist(), "actual": actual.tolist(),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "domain_sum": float(actual[:DOMAIN_NUM].sum()),
        "cell_sum": float(actual.sum()),
    }


def weighted_loss_from_terms(
    loss_terms: Sequence[torch.Tensor], coefficients: torch.Tensor,
) -> torch.Tensor:
    terms = list(loss_terms)
    coeff = coefficients.reshape(-1)
    if len(terms) != len(coeff) or not terms:
        raise ValueError("loss term/coefficient vectors must have the same non-zero length")
    return torch.stack(terms).to(device=coeff.device, dtype=coeff.dtype).dot(coeff)


# Compatibility aliases make the mathematical primitive discoverable under
# the terminology used in the design specification without introducing a
# second coefficient implementation.
alpha_to_coefficients = weighting_coefficients
loss_from_terms = weighted_loss_from_terms


def cell_index(task: int, domain: int) -> int:
    return int(task) * DOMAIN_NUM + int(domain)


def split_row_fingerprint(x_values: object, y_values: object) -> str:
    """Hash ordered feature/label rows, including dtype and shape."""

    hasher = hashlib.sha256()
    for value in (x_values, y_values):
        array = np.ascontiguousarray(np.asarray(value))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        if array.dtype.hasobject:
            hasher.update(repr(array.tolist()).encode("utf-8"))
        else:
            hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def normalise_dataset_path(path: object) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def file_identity(path: str) -> Dict[str, object]:
    resolved = normalise_dataset_path(path)
    stat = os.stat(resolved)
    return {
        "path": resolved,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _positive_rate(values: np.ndarray, task: int) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.asarray(values[:, task], dtype=np.float64).mean())


def _split_views(
    x: np.ndarray, y: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n = int(len(x))
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": (x[:train_end], y[:train_end]),
        "validation": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }


def _split_metadata(
    splits: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    split_counts: Dict[str, Dict[int, int]] = {}
    positive_counts: Dict[str, Dict[int, Dict[int, int]]] = {}
    positive_rates: Dict[str, Dict[int, Dict[int, float]]] = {}
    fingerprints: Dict[str, str] = {}
    for split, (x_values, y_values) in splits.items():
        ids = np.asarray(x_values[:, -1], dtype=np.int64)
        split_counts[split] = {d: int((ids == d).sum()) for d in range(DOMAIN_NUM)}
        positive_counts[split] = {}
        positive_rates[split] = {}
        for d in range(DOMAIN_NUM):
            mask = ids == d
            positive_counts[split][d] = {
                t: int(np.asarray(y_values[mask, t], dtype=np.int64).sum())
                for t in range(TASK_NUM)
            }
            positive_rates[split][d] = {
                t: _positive_rate(y_values[mask], t) for t in range(TASK_NUM)
            }
        fingerprints[split] = split_row_fingerprint(x_values, y_values)
    return {
        "split_counts": split_counts,
        "positive_counts": positive_counts,
        "positive_rates": positive_rates,
        "split_row_fingerprints": fingerprints,
    }


def audit_dataset_bundle(bundle: "DatasetBundle", *, strict: bool = True) -> Dict[str, object]:
    """Check the immutable audit counts and return an auditable summary."""

    observed = bundle.metadata
    reasons: List[str] = []
    if int(bundle.total_rows) != EXPECTED_TOTAL_ROWS:
        reasons.append(f"total_rows:{bundle.total_rows}")
    for split in ("train", "validation", "test"):
        for d in range(DOMAIN_NUM):
            got = int(observed["split_counts"][split][d])
            want = EXPECTED_SPLIT_COUNTS[split][d]
            if got != want:
                reasons.append(f"{split}.d{d}.count:{got}!={want}")
            for t in range(TASK_NUM):
                got_pos = int(observed["positive_counts"][split][d][t])
                want_pos = EXPECTED_POSITIVE_COUNTS[split][d][t]
                if got_pos != want_pos:
                    reasons.append(f"{split}.d{d}.t{t}.positive:{got_pos}!={want_pos}")
    result = {
        "valid": not reasons,
        "reasons": reasons,
        "total_rows": int(bundle.total_rows),
        "split_counts": observed["split_counts"],
        "positive_counts": observed["positive_counts"],
        "positive_rates": observed["positive_rates"],
        "split_row_fingerprints": observed["split_row_fingerprints"],
    }
    if strict and reasons:
        raise AssertionError("KuaiRand E4 dataset audit failed: " + "; ".join(reasons[:12]))
    return result


@dataclass
class DatasetBundle:
    dense_features: Sequence[object]
    sparse_features: Sequence[object]
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    total_rows: int
    dataset_path: str
    dataset_identity: str
    dataset_file_identity: Mapping[str, object]
    metadata: Mapping[str, object]

    @property
    def splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            "train": (self.x_train, self.y_train),
            "validation": (self.x_validation, self.y_validation),
            "test": (self.x_test, self.y_test),
        }


def load_processed_dataset(
    dataset_path: str = E4_DATASET_PATH,
    *,
    validate: bool = True,
) -> DatasetBundle:
    """Load the already processed CSV without reordering or re-splitting it."""

    path = os.path.abspath(os.fspath(dataset_path))
    columns = list(RAW_FEATURE_COLUMNS) + list(LABEL_COLUMNS)
    frame = pd.read_csv(path, usecols=columns)
    if not frame.columns.tolist() == columns:
        raise AssertionError("processed CSV schema/order differs from E4 contract")
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise AssertionError(f"processed CSV row count {len(frame)} != {EXPECTED_TOTAL_ROWS}")

    raw_tab = pd.to_numeric(frame["tab"], errors="raise").astype(np.int64)
    if not set(raw_tab.unique().tolist()).issubset(set(DOMAIN_TABS)):
        raise AssertionError("processed CSV contains a tab outside {0,1,4}")
    domain_values = raw_tab.map({tab: idx for idx, tab in enumerate(DOMAIN_TABS)}).to_numpy(dtype=np.int64)
    feature_frame = frame.drop(columns=["tab"]).copy()
    feature_frame["domain_indicator"] = domain_values

    # Match run_kuaishou's global LabelEncoder semantics: each feature is
    # encoded once on the complete processed table, and +1 reserves index 0.
    encoded_columns: List[np.ndarray] = []
    vocab_sizes: List[int] = []
    for name in SPARSE_FEATURE_COLUMNS:
        values = feature_frame[name].where(feature_frame[name].notna(), -1).astype(str)
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(values) + 1
        # ``run_kuaishou.py`` first reserves index 0 for every sparse feature,
        # then explicitly remaps domain_indicator back to 0,1,2 because the
        # MDMTRec forward path compares it with local domain ids.
        if name == "domain_indicator":
            encoded = domain_values
        encoded_columns.append(np.asarray(encoded, dtype=np.int64))
        vocab_sizes.append(int(len(encoder.classes_) + 1))
    x = np.column_stack(encoded_columns).astype(np.int64, copy=False)
    y = frame.loc[:, list(LABEL_COLUMNS)].fillna(-1).to_numpy(dtype=np.float32)
    if not np.isin(y, [0.0, 1.0]).all():
        raise AssertionError("E4 labels must be binary 0/1")
    splits = _split_views(x, y)
    metadata = _split_metadata(splits)
    metadata = dict(metadata)
    metadata.update({
        "domain_mapping": {"tab0": 0, "tab1": 1, "tab4": 2},
        "task_mapping": {"is_click": 0, "long_view": 1},
        "domain_num": DOMAIN_NUM, "task_num": TASK_NUM,
        "train_domain_shares": {
            d: float(metadata["split_counts"]["train"][d] / len(splits["train"][0]))
            for d in range(DOMAIN_NUM)
        },
    })
    from MDMTRec.basic.features import SparseFeature  # local import keeps unit helpers light
    sparse_features = [
        SparseFeature(index, vocab_size=vocab_sizes[index], embed_dim=16)
        for index in range(len(SPARSE_FEATURE_COLUMNS))
    ]
    bundle = DatasetBundle(
        dense_features=[], sparse_features=sparse_features,
        x_train=splits["train"][0], y_train=splits["train"][1],
        x_validation=splits["validation"][0], y_validation=splits["validation"][1],
        x_test=splits["test"][0], y_test=splits["test"][1],
        total_rows=len(x), dataset_path=path,
        dataset_identity=normalise_dataset_path(path),
        dataset_file_identity=file_identity(path), metadata=metadata,
    )
    if validate:
        audit_dataset_bundle(bundle, strict=True)
    return bundle


def filter_split_first_domain(
    bundle: DatasetBundle, original_domain: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    """Filter each already-created global split, then remap domain to local 0."""

    domain = int(original_domain)
    if domain not in range(DOMAIN_NUM):
        raise ValueError("original_domain must be 0, 1, or 2")
    filtered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    counts: Dict[str, Dict[int, int]] = {}
    positives: Dict[str, Dict[int, int]] = {}
    fingerprints: Dict[str, str] = {}
    for split, (x_values, y_values) in bundle.splits.items():
        mask = np.asarray(x_values[:, -1], dtype=np.int64) == domain
        x_view = np.array(x_values[mask], copy=True)
        y_view = np.array(y_values[mask], copy=True)
        x_view[:, -1] = 0
        filtered[split] = (x_view, y_view)
        counts[split] = {t: int(len(y_view)) for t in range(TASK_NUM)}
        positives[split] = {
            t: int(np.asarray(y_view[:, t], dtype=np.int64).sum()) for t in range(TASK_NUM)
        }
        fingerprints[split] = split_row_fingerprint(x_view, y_view)
    info = {
        "original_physical_domain": domain,
        "local_domain_mapping": {str(domain): 0},
        "filter_after_global_split": True,
        "filter_method": "global_chronological_split_then_filter_original_domain",
        "split_counts": counts,
        "positive_counts": positives,
        "split_row_fingerprints": fingerprints,
        "global_split_row_fingerprints": dict(bundle.metadata["split_row_fingerprints"]),
    }
    return filtered, info


filter_domain_after_split = filter_split_first_domain


def strict_validation_macro_auc(
    stats: Mapping[int, Mapping[str, object]], expected_cells: Iterable[int],
) -> float:
    expected = [int(cell) for cell in expected_cells]
    values: List[float] = []
    invalid: List[str] = []
    for cell in expected:
        record = stats.get(int(cell))
        if not isinstance(record, Mapping):
            invalid.append(f"{cell}:missing")
            continue
        try:
            auc = float(record.get("auc"))
            ll = float(record.get("logloss"))
            if not np.isfinite(auc) or not np.isfinite(ll):
                raise ValueError
        except (TypeError, ValueError):
            invalid.append(f"{cell}:nonfinite")
            continue
        values.append(auc)
    if invalid or len(values) != len(expected):
        raise RuntimeError("validation cells incomplete/nonfinite: " + ",".join(invalid))
    return float(np.mean(values))


def metric_records(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
    device: torch.device, domain_num: int, task_num: int,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, float]]:
    """Collect cell metrics plus separately labelled pooled metrics."""

    truth: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    pred_values: Dict[int, List[float]] = {idx: [] for idx, _, _ in iter_cells(domain_num, task_num)}
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_device = x_batch.transpose(0, 1).to(device)
            y_device = y_batch.to(device).float()
            prediction = model(x_device).to(device)
            domain_ids = x_device[-1, :].detach()
            for idx, domain, task in iter_cells(domain_num, task_num):
                mask = domain_ids == domain
                if bool(mask.any()):
                    truth[idx].extend(y_device[mask, task].detach().cpu().tolist())
                    pred_values[idx].extend(prediction[mask, task].detach().cpu().tolist())
    rows: Dict[int, Dict[str, object]] = {}
    for idx, _domain, _task in iter_cells(domain_num, task_num):
        rows[idx] = {
            "sample_count": len(truth[idx]),
            "positive_rate": float(np.mean(truth[idx])) if truth[idx] else float("nan"),
            "auc": safe_auc(truth[idx], pred_values[idx]),
            "logloss": safe_logloss(truth[idx], pred_values[idx]),
        }
    flat_truth = [value for idx in sorted(truth) for value in truth[idx]]
    flat_pred = [value for idx in sorted(pred_values) for value in pred_values[idx]]
    pooled = {
        "auc": safe_auc(flat_truth, flat_pred),
        "logloss": safe_logloss(flat_truth, flat_pred),
    }
    return rows, pooled


def setup_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True


SHARED_STEP_FIELDS = [
    "seed", "weighting", "alpha", "epoch", "step", "domain", "task", "cell",
    "batch_fingerprint", "cell_sample_count", "loss_coefficient",
    "raw_gradient_norm", "raw_gradient_rms", "weighted_gradient_norm",
    "weighted_gradient_rms", "effective_projection", "normalized_contribution",
    "signed_contribution", "domain_signed_contribution", "magnitude_share",
    "overrepresentation_ratio", "mismatch_index", "equal_cf_contribution",
    "sqrt_cf_contribution", "sample_cf_contribution", "equal_cf_magnitude_share",
    "sqrt_cf_magnitude_share", "sample_cf_magnitude_share", "unused_param_count",
    "actual_update_norm", "actual_update_alignment", "actual_update_projection",
    "gradient_reconstruction_error",
]
SHARED_EPOCH_FIELDS = [
    "seed", "weighting", "alpha", "epoch", "domain", "task", "cell",
    "raw_gradient_norm", "weighted_gradient_norm", "gradient_variance",
    "weighted_gradient_variance", "gradient_SNR", "norm_of_mean_gradient",
    "probe_count", "contribution_mean", "contribution_std", "magnitude_share",
    "mismatch_index", "variance_scope",
]


def collect_shared_raw_gradients(
    loss_terms: Sequence[torch.Tensor], model: torch.nn.Module, partition: object,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Collect only shared-parameter raw gradients from one forward graph."""

    named = dict(model.named_parameters())
    names = [name for name in partition.names("shared") if named[name].requires_grad]
    params = [named[name] for name in names]
    raw: Dict[int, torch.Tensor] = {}
    unused: Dict[int, int] = {}
    for idx, term in enumerate(loss_terms):
        if params:
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            vector, count = flatten_parameter_tensors(
                named, names, grads, device=term.device)
        else:
            vector = torch.zeros(0, dtype=term.dtype, device=term.device)
            count = 0
        raw[idx] = vector.detach()
        unused[idx] = int(count)
    return raw, unused


def _vector_norm(value: Optional[torch.Tensor]) -> float:
    if value is None or not value.numel():
        return 0.0
    return float(torch.linalg.vector_norm(value.detach()).cpu())


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or not right.numel():
        return 0.0
    return float(torch.dot(left.reshape(-1), right.reshape(-1)).detach().cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.numel() or not right.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.detach().cpu()) <= EPS:
        return float("nan")
    return float(torch.clamp(_dot_tensor(left, right) / denom, -1.0, 1.0).detach().cpu())


def _dot_tensor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.dot(left.reshape(-1), right.reshape(-1))


def make_shared_gradient_step_rows(
    raw_by_cell: Mapping[int, torch.Tensor], unused_by_cell: Mapping[int, int],
    *, weighting: object, coefficients: torch.Tensor,
    equal_coefficients: torch.Tensor, sqrt_coefficients: torch.Tensor,
    sample_coefficients: torch.Tensor, domain_shares: Sequence[float],
    shared_param_count: int, seed: int, epoch: int, step: int,
    batch_fingerprint_value: str, cell_sample_counts: Sequence[int],
    actual_update: Optional[torch.Tensor] = None,
    reconstruction_error: float = float("nan"),
) -> List[Dict[str, object]]:
    key = canonical_weighting(weighting)
    actual = contribution_views(raw_by_cell, coefficients, DOMAIN_NUM, TASK_NUM)
    equal = contribution_views(raw_by_cell, equal_coefficients, DOMAIN_NUM, TASK_NUM)
    sqrt = contribution_views(raw_by_cell, sqrt_coefficients, DOMAIN_NUM, TASK_NUM)
    sample = contribution_views(raw_by_cell, sample_coefficients, DOMAIN_NUM, TASK_NUM)
    shares = _normalise_shares(domain_shares)
    domain_mag = {
        d: sum(float(actual["magnitude"].get(cell_index(t, d), 0.0)) for t in range(TASK_NUM))
        for d in range(DOMAIN_NUM)
    }
    domain_signed = {
        d: sum(float(actual["normalized"].get(cell_index(t, d), 0.0)) for t in range(TASK_NUM))
        for d in range(DOMAIN_NUM)
    }
    mismatch = 0.5 * sum(abs(domain_mag[d] - float(shares[d])) for d in range(DOMAIN_NUM))
    update_norm = _vector_norm(actual_update)
    rows: List[Dict[str, object]] = []
    for idx, domain, task in iter_cells(DOMAIN_NUM, TASK_NUM):
        raw = raw_by_cell[idx]
        coefficient = float(coefficients[idx].detach().cpu())
        weighted = raw * coefficients[idx].to(device=raw.device)
        raw_norm = _vector_norm(raw)
        weighted_norm = abs(coefficient) * raw_norm
        rows.append({
            "seed": int(seed), "weighting": key, "alpha": weighting_alpha(key),
            "epoch": int(epoch), "step": int(step), "domain": int(domain),
            "task": int(task), "cell": int(idx),
            "batch_fingerprint": str(batch_fingerprint_value),
            "cell_sample_count": int(cell_sample_counts[idx]),
            "loss_coefficient": coefficient, "raw_gradient_norm": raw_norm,
            "raw_gradient_rms": raw_norm / math.sqrt(shared_param_count)
            if shared_param_count else 0.0,
            "weighted_gradient_norm": weighted_norm,
            "weighted_gradient_rms": weighted_norm / math.sqrt(shared_param_count)
            if shared_param_count else 0.0,
            "effective_projection": float(actual["projection"].get(idx, 0.0)),
            "normalized_contribution": float(actual["normalized"].get(idx, 0.0)),
            "signed_contribution": float(actual["normalized"].get(idx, 0.0)),
            "domain_signed_contribution": domain_signed[domain],
            "magnitude_share": domain_mag[domain],
            "overrepresentation_ratio": domain_mag[domain] / (float(shares[domain]) + EPS),
            "mismatch_index": mismatch,
            "equal_cf_contribution": float(equal["normalized"].get(idx, 0.0)),
            "sqrt_cf_contribution": float(sqrt["normalized"].get(idx, 0.0)),
            "sample_cf_contribution": float(sample["normalized"].get(idx, 0.0)),
            "equal_cf_magnitude_share": sum(float(equal["magnitude"].get(cell_index(t, domain), 0.0))
                                             for t in range(TASK_NUM)),
            "sqrt_cf_magnitude_share": sum(float(sqrt["magnitude"].get(cell_index(t, domain), 0.0))
                                            for t in range(TASK_NUM)),
            "sample_cf_magnitude_share": sum(float(sample["magnitude"].get(cell_index(t, domain), 0.0))
                                              for t in range(TASK_NUM)),
            "unused_param_count": int(unused_by_cell.get(idx, 0)),
            "actual_update_norm": update_norm,
            "actual_update_alignment": (_cosine(weighted, actual_update)
                                         if actual_update is not None else float("nan")),
            "actual_update_projection": (_dot(weighted, actual_update) / (update_norm + EPS)
                                         if actual_update is not None else float("nan")),
            "gradient_reconstruction_error": float(reconstruction_error),
        })
    return rows


class SharedGradientEpochAccumulator:
    """Full-vector Welford statistics for the shared raw gradients."""

    def __init__(self) -> None:
        self.count = 0
        self.stats = {idx: WelfordVector() for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        self.contributions = {idx: [] for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        self.magnitudes = {idx: [] for idx, _, _ in iter_cells(DOMAIN_NUM, TASK_NUM)}
        self.mismatches: List[float] = []

    def update(
        self, raw_by_cell: Mapping[int, torch.Tensor], views: Mapping[str, object],
        mismatch: float,
    ) -> None:
        self.count += 1
        for idx, _domain, _task in iter_cells(DOMAIN_NUM, TASK_NUM):
            self.stats[idx].update(raw_by_cell[idx])
            self.contributions[idx].append(float(views["normalized"].get(idx, 0.0)))
            self.magnitudes[idx].append(float(views["magnitude"].get(idx, 0.0)))
        self.mismatches.append(float(mismatch))

    def rows(self, *, weighting: object, coefficients: torch.Tensor,
             seed: int, epoch: int) -> List[Dict[str, object]]:
        key = canonical_weighting(weighting)
        coeff = coefficients.detach().cpu().numpy().reshape(-1)
        mismatch = float(np.mean(self.mismatches)) if self.mismatches else float("nan")
        result: List[Dict[str, object]] = []
        for idx, domain, task in iter_cells(DOMAIN_NUM, TASK_NUM):
            stat = self.stats[idx]
            values = np.asarray(self.contributions[idx], dtype=np.float64)
            mags = np.asarray(self.magnitudes[idx], dtype=np.float64)
            mean_norm = stat.mean_norm
            variance = stat.variance_trace
            result.append({
                "seed": int(seed), "weighting": key, "alpha": weighting_alpha(key),
                "epoch": int(epoch), "domain": int(domain), "task": int(task),
                "cell": int(idx), "raw_gradient_norm": mean_norm,
                "weighted_gradient_norm": mean_norm * abs(float(coeff[idx]))
                if np.isfinite(mean_norm) else float("nan"),
                "gradient_variance": variance,
                "weighted_gradient_variance": variance * float(coeff[idx] ** 2)
                if np.isfinite(variance) else float("nan"),
                "gradient_SNR": stat.snr,
                "norm_of_mean_gradient": stat.norm_of_mean,
                "probe_count": int(stat.count),
                "contribution_mean": float(values.mean()) if len(values) else float("nan"),
                "contribution_std": float(values.std()) if len(values) else float("nan"),
                "magnitude_share": float(mags.mean()) if len(mags) else float("nan"),
                "mismatch_index": mismatch,
                "variance_scope": "shared_vector",
            })
        return result


def json_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = [
    "E4_DATASET_PATH", "EXPECTED_SEEDS", "DOMAIN_NUM", "TASK_NUM", "DOMAIN_TABS",
    "DOMAIN_NAMES", "TASK_NAMES", "WEIGHTING_KEYS", "WEIGHTING_DISPLAY", "WEIGHTING_ALPHA",
    "ALPHA_GRID", "RAW_FEATURE_COLUMNS", "SPARSE_FEATURE_COLUMNS", "LABEL_COLUMNS",
    "EXPECTED_TOTAL_ROWS", "EXPECTED_SPLIT_COUNTS", "EXPECTED_POSITIVE_COUNTS",
    "TRAIN_DOMAIN_SHARES", "E4_FORMAL_PROTOCOL", "E4_ORACLE_PROTOCOL", "DatasetBundle",
    "canonical_weighting", "canonical_alpha", "weighting_alpha", "weighting_display",
    "weighting_coefficients", "weighting_coefficients_numpy", "coefficient_table",
    "validate_weighting_coefficients", "weighted_loss_from_terms", "alpha_to_coefficients",
    "loss_from_terms", "cell_index",
    "split_row_fingerprint", "normalise_dataset_path", "file_identity",
    "load_processed_dataset", "audit_dataset_bundle", "filter_split_first_domain",
    "filter_domain_after_split",
    "strict_validation_macro_auc", "metric_records", "setup_seed", "SHARED_STEP_FIELDS",
    "SHARED_EPOCH_FIELDS", "collect_shared_raw_gradients", "make_shared_gradient_step_rows",
    "SharedGradientEpochAccumulator", "json_hash", "build_parameter_partition",
    "flatten_parameter_tensors", "parameter_state_hash", "preserve_rng_state",
    "relative_l2_error", "rng_state_digest", "safe_auc", "safe_logloss", "write_json",
    "write_rows_csv", "masked_bce_terms", "batch_fingerprint", "iter_cells",
]
