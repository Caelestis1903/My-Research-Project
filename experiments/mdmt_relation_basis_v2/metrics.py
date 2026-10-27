"""FP64 logit LogLoss and logit-ranked AUC, task-major/domain-minor cells."""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(labels, logits, domain, mask=None):
    labels, logits = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    domain = np.asarray(domain)
    if labels.shape != logits.shape or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError('Expected [N,2] labels and logits')
    mask = np.ones_like(labels, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != labels.shape or not np.isin(domain, [0, 1, 2]).all():
        raise ValueError('Invalid metric mask/domain')
    if not np.isfinite(logits[mask]).all() or not np.isin(labels[mask], [0, 1]).all():
        raise ValueError('Invalid metric labels/logits')
    losses = np.zeros_like(logits)
    losses[mask] = np.logaddexp(0., logits[mask]) - labels[mask] * logits[mask]
    cells = []
    for task in range(2):
        for d in range(3):
            chosen = (domain == d) & mask[:, task]
            y, z = labels[chosen, task], logits[chosen, task]
            n, positive = len(y), int(y.sum())
            cells.append(dict(cell=3*task+d, domain=d, task=task, n=n,
                              positive=positive, negative=n-positive,
                              logloss=float(losses[chosen, task].mean()) if n else None,
                              auc=float(roc_auc_score(y, z)) if 0 < positive < n else None))
    def mean(field):
        values = [c[field] for c in cells if c[field] is not None]
        return float(np.mean(values)) if values else None
    pooled = []
    for task in range(2):
        chosen = mask[:, task]
        y, z = labels[chosen, task], logits[chosen, task]
        pooled.append(float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else None)
    return dict(cells=cells, macro_logloss=mean('logloss'), macro_auc=mean('auc'),
                natural_logloss=float(losses[mask].mean()) if mask.any() else None,
                pooled_auc_by_task=pooled, valid_requests=int(mask.sum()))

