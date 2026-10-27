"""FP64 logit risk and logit-ranked AUC, preserving task and domain axes."""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(labels, logits, domain, mask=None, domains=3):
    y, z = np.asarray(labels, np.float64), np.asarray(logits, np.float64)
    d = np.asarray(domain)
    if y.shape != z.shape or z.ndim != 2 or d.shape != (len(y),):
        raise ValueError('Expected equal [N,T] labels/logits and [N] domain')
    valid = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != y.shape or not np.isfinite(z[valid]).all() or not np.isin(y[valid],[0,1]).all():
        raise ValueError('Invalid metric input')
    ll = np.logaddexp(0., z)-y*z
    cells = []
    for task in range(y.shape[1]):
        for dom in range(domains):
            chosen = (d == dom) & valid[:,task]
            yy, zz = y[chosen,task], z[chosen,task]
            n, positive = len(yy), int(yy.sum())
            cells.append(dict(cell=task*domains+dom, domain=dom, task=task, n=n,
                              positive=positive, negative=n-positive,
                              logloss=float(ll[chosen,task].mean()) if n else None,
                              auc=float(roc_auc_score(yy,zz)) if 0<positive<n else None))
    def average(key):
        values = [c[key] for c in cells if c[key] is not None]
        return float(np.mean(values)) if values else None
    pooled = []
    for task in range(y.shape[1]):
        chosen = valid[:,task]
        yy, zz = y[chosen,task], z[chosen,task]
        pooled.append(float(roc_auc_score(yy,zz)) if len(np.unique(yy)) == 2 else None)
    return dict(cells=cells, macro_logloss=average('logloss'), macro_auc=average('auc'),
                macro_logloss_denominator=sum(c['logloss'] is not None for c in cells),
                macro_auc_denominator=sum(c['auc'] is not None for c in cells),
                natural_logloss=float(ll[valid].mean()) if valid.any() else None,
                pooled_auc_by_task=pooled, valid_labels=int(valid.sum()))


def weighted_auc(y,z,w):
    y,z,w = map(np.asarray,(y,z,w))
    positive, negative = w[y==1].sum(), w[y==0].sum()
    return float(roc_auc_score(y,z,sample_weight=w)) if positive>0 and negative>0 else None


def prototype_metrics(y,z,domain,responsibility):
    rows=[]
    for d in np.unique(domain):
        pick=np.asarray(domain)==d
        for k in range(responsibility.shape[1]):
            w=responsibility[pick,k].astype(np.float64)
            mass=w.sum()
            for t in range(y.shape[1]):
                yy,zz=np.asarray(y[pick,t],np.float64),np.asarray(z[pick,t],np.float64)
                rows.append(dict(domain=int(d),task=t,prototype=k,n=int(pick.sum()),
                                 hard_n=int((responsibility[pick].argmax(1)==k).sum()),
                                 mass=float(mass),ess=float(mass*mass/(w@w)) if w@w>0 else 0.,
                                 auc=weighted_auc(yy,zz,w),
                                 logloss=float(w@(np.logaddexp(0.,zz)-yy*zz)/mass) if mass>0 else None))
    return rows
