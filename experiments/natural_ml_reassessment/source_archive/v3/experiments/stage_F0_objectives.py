"""Loss and validation aggregation fixed for the F0 Sample/Q block."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from stage_S1_metrics import safe_auc, safe_logloss


def sample_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean BCE over the current rows and all observed task labels."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] < 1:
        raise ValueError("Sample BCE expects equal [batch, tasks] prediction/target tensors")
    if prediction.shape[0] < 1:
        raise ValueError("Sample BCE rejects an empty batch")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("Sample BCE inputs must be finite")
    return F.binary_cross_entropy(prediction, target.to(dtype=prediction.dtype), reduction="mean")


def equal_diagnostic_bce(prediction: torch.Tensor, target: torch.Tensor,
                         local_domain_ids: torch.Tensor,
                         train_domain_shares: Mapping[int, float],
                         domain_num: int) -> torch.Tensor:
    """Training-share Equal diagnostic, normalized by batch rows and tasks."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Equal diagnostic expects equal [batch, tasks] tensors")
    if local_domain_ids.ndim != 1 or len(local_domain_ids) != len(target):
        raise ValueError("Equal diagnostic domain IDs must have one value per row")
    if int(domain_num) < 1:
        raise ValueError("domain_num must be positive")
    if len(target) == 0:
        raise ValueError("Equal diagnostic rejects an empty batch")
    if set(int(k) for k in train_domain_shares) != set(range(int(domain_num))):
        raise ValueError("training shares must cover every local domain")
    shares = torch.as_tensor([float(train_domain_shares[d]) for d in range(int(domain_num))],
                             device=prediction.device, dtype=prediction.dtype)
    if not bool(torch.isfinite(shares).all()) or bool(torch.any(shares <= 0)):
        raise ValueError("all train domain shares must be finite and positive")
    ids = local_domain_ids.to(device=prediction.device, dtype=torch.long)
    if bool(torch.any((ids < 0) | (ids >= int(domain_num)))):
        raise ValueError("local domain ID is outside Equal scope")
    per_entry = F.binary_cross_entropy(prediction, target.to(dtype=prediction.dtype),
                                       reduction="none")
    row_weights = 1.0 / (float(domain_num) * shares[ids])
    return (per_entry * row_weights.unsqueeze(1)).sum() / float(len(target) * target.shape[1])


def split_weighted_sample_loss(losses: Sequence[torch.Tensor], batch_sizes: Sequence[int]) -> torch.Tensor:
    """Combine batch means with the exact full-pool row denominator."""

    if len(losses) == 0 or len(losses) != len(batch_sizes):
        raise ValueError("losses and batch_sizes must be non-empty and aligned")
    sizes = [int(size) for size in batch_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive")
    if any(loss.ndim != 0 for loss in losses):
        raise ValueError("each batch loss must be a scalar mean")
    total = sum((loss * size for loss, size in zip(losses, sizes)), losses[0].new_zeros(()))
    return total / float(sum(sizes))


def evaluate_cells(labels: np.ndarray, predictions: np.ndarray,
                   local_domain_ids: np.ndarray, physical_domains: Sequence[int],
                   task_num: int = 2) -> Dict[str, object]:
    """Compute each local domain/task cell and equal-cell selection summaries."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    domains = np.asarray(local_domain_ids, dtype=np.int64)
    physical = tuple(int(v) for v in physical_domains)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != int(task_num):
        raise ValueError("evaluation labels and predictions must be [rows, task_num]")
    if domains.shape != (len(y),) or len(physical) == 0:
        raise ValueError("evaluation domain IDs/physical map have invalid shape")
    cells: Dict[str, Dict[str, float]] = {}
    for task in range(int(task_num)):
        for local_domain, physical_domain in enumerate(physical):
            mask = domains == local_domain
            cell_y = y[mask, task]
            cell_p = p[mask, task]
            if not len(cell_y) or len(np.unique(cell_y)) != 2:
                raise ValueError("evaluation cell d{}t{} is empty or single-class".format(
                    physical_domain, task))
            cells["d{}t{}".format(physical_domain, task)] = {
                "physical_domain": physical_domain,
                "task": task,
                "sample_count": int(len(cell_y)),
                "positive_count": int(cell_y.sum()),
                "positive_rate": float(cell_y.mean()),
                "auc": safe_auc(cell_y, cell_p),
                "logloss": safe_logloss(cell_y, cell_p),
            }
    auc_values = [float(value["auc"]) for value in cells.values()]
    ll_values = [float(value["logloss"]) for value in cells.values()]
    return {"cells": cells, "macro_auc": float(np.mean(auc_values)),
            "macro_logloss": float(np.mean(ll_values)),
            "n_cells": len(cells)}


__all__ = ["sample_bce", "equal_diagnostic_bce", "split_weighted_sample_loss",
           "evaluate_cells"]
