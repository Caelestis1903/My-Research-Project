"""Natural valid-label BCE; invalid labels are excluded before arithmetic."""
import torch
from torch.nn import functional as F


def masked_bce(logits, labels, mask=None):
    mask = torch.ones_like(labels, dtype=torch.bool) if mask is None else mask.bool()
    if mask.shape != logits.shape or labels.shape != logits.shape:
        raise ValueError('Label/mask shape mismatch')
    if not bool(mask.any()):
        return None
    valid_labels = labels[mask]
    if not bool(torch.isfinite(valid_labels).all()) or not bool(((valid_labels == 0) | (valid_labels == 1)).all()):
        raise ValueError('Valid labels must be finite binary values')
    return F.binary_cross_entropy_with_logits(logits[mask], valid_labels, reduction='mean')

