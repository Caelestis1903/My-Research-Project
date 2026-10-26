"""Single frozen metric implementation used by every Stage S1 path.

This module is deliberately NumPy-only.  The runner, deferred-artifact
validator, report collector, and acceptance fixtures all import these
functions so that a score cannot silently acquire a different clipping rule
or cell-index convention in a later stage.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# This value is part of the Stage S1 artifact contract.  Do not duplicate it
# in a caller; the raw runner and all mechanical reports must agree exactly.
LOGLOSS_EPS: float = 1e-12


def safe_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Return tie-aware binary ROC-AUC using stable average ranks."""

    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or y.size == 0:
        raise ValueError("AUC requires equally shaped non-empty vectors")
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        raise ValueError("AUC requires both classes")
    if not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("AUC inputs must be finite")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    left = 0
    while left < len(s):
        right = left + 1
        while right < len(s) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        # Ranks are one-based; the mean rank for [left, right) is
        # ((left+1) + right) / 2.
        ranks[order[left:right]] = (left + 1 + right) / 2.0
        left = right
    n_pos = float(positive.sum())
    n_neg = float(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1.0) / 2.0)
                 / (n_pos * n_neg))


def safe_logloss(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Return binary LogLoss with the frozen ``LOGLOSS_EPS`` clip."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("LogLoss requires equally shaped non-empty vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("LogLoss inputs must be finite")
    p = np.clip(p, LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def cell_metrics(labels: Sequence[float], scores: Sequence[float]) -> dict:
    """Compute all raw binary cell metrics with one shared definition."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("cell metric vectors must be equally shaped and non-empty")
    if len(np.unique(y)) < 2:
        raise ValueError("cell metric requires both classes")
    return {
        "sample_count": int(y.size),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auc": safe_auc(y, p),
        "logloss": safe_logloss(y, p),
    }


def task_domain_cell_index(task: int, domain: int, domain_num: int) -> int:
    """Return the design's task-major cell index ``t * D + d``."""

    task_i, domain_i, domain_count = int(task), int(domain), int(domain_num)
    if task_i < 0 or task_i >= 2 or domain_i < 0 or domain_i >= domain_count:
        raise ValueError("task/domain outside Stage S1 cell support")
    return task_i * domain_count + domain_i


__all__ = ["LOGLOSS_EPS", "safe_auc", "safe_logloss", "cell_metrics",
           "task_domain_cell_index"]
