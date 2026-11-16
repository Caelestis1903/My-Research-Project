"""Unperturbed six-cell metrics, preserving undefined cells."""
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .vendor.gradient_aug_20261031_v2.vendor_ga31.metrics import metrics as source_metrics


def metrics(y, pred, domain):
    y = np.asarray(y); pred = np.asarray(pred, np.float64); domain = np.asarray(domain, np.int64)
    defined = all(np.unique(y[domain == d, t]).size == 2 for d in range(3) for t in range(2))
    if defined:
        out = source_metrics(y, pred, domain)
    else:
        with np.errstate(divide='ignore'):
            loss = -(y * np.maximum(np.log(pred), -100) +
                     (1-y) * np.maximum(np.log1p(-pred), -100))
        out = {}
        for d in range(3):
            mask = domain == d
            for t in range(2):
                labels = y[mask, t]
                out[f'd{d}t{t}_auc'] = float(roc_auc_score(labels, pred[mask, t])) if np.unique(labels).size == 2 else None
                out[f'd{d}t{t}_logloss'] = float(loss[mask, t].mean()) if mask.any() else None
                out[f'd{d}t{t}_count'] = int(mask.sum())
                out[f'd{d}t{t}_positive'] = int(labels.sum())
        for name in ('auc', 'logloss'):
            vals = [out[f'd{d}t{t}_{name}'] for d in range(3) for t in range(2)]
            out['macro_'+name] = None if any(v is None for v in vals) else float(np.mean(vals))
            out['sample_weighted_cell_'+name] = None if any(v is None for v in vals) else float(sum(
                (domain == d).mean()*out[f'd{d}t{t}_{name}']/2 for d in range(3) for t in range(2)))
            for d in range(3):
                v = [out[f'd{d}t{t}_{name}'] for t in range(2)]
                out[f'domain{d}_macro_{name}'] = None if None in v else float(np.mean(v))
            for t in range(2):
                v = [out[f'd{d}t{t}_{name}'] for d in range(3)]
                out[f'task{t}_macro_{name}'] = None if None in v else float(np.mean(v))
        for t in range(2):
            out[f'task{t}_pooled_auc'] = float(roc_auc_score(y[:, t], pred[:, t])) if np.unique(y[:, t]).size == 2 else None
            out[f'task{t}_pooled_logloss'] = float(loss[:, t].mean())
        v = [out[f'task{t}_pooled_auc'] for t in range(2)]
        out['pooled_task_mean_auc'] = None if None in v else float(np.mean(v))
        out['pooled_logloss'] = float(loss.mean())
        out['worst_cell_auc'] = None if not defined else min(out[f'd{d}t{t}_auc'] for d in range(3) for t in range(2))
    for d in range(3):
        for t in range(2):
            out[f'd{d}t{t}_negative'] = out[f'd{d}t{t}_count'] - out[f'd{d}t{t}_positive']
    return out


def evaluate(model, x, y, device, batch_size=40960):
    was = model.training; model.eval(); predictions = []; start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(y), batch_size):
            bx = torch.as_tensor(x[offset:offset+batch_size], device=device).T
            predictions.append(model(bx).cpu().numpy())
    model.train(was)
    out = metrics(y, np.concatenate(predictions), x[:, -1])
    return out, dict(seconds=time.perf_counter()-start, forwards=len(predictions), rows=len(y))
