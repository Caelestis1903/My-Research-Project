"""Matched training and validation populations and bounded score banks for B."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit

from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import forward_logits
from .b_build_panels import build_panels
from .b_pairs import exact_transition_table, transition_metrics
from .b_panel_identity import HASHES, SPEC, digest


@dataclass
class Panel:
    x: np.ndarray
    y: np.ndarray
    rank: list
    pairs: list
    natural: list


def prepare(bundle, seed):
    raw = build_panels(bundle, seed, SPEC)
    hashes = {name: p['panel_hash'] for name, p in raw.items()}
    if hashes != HASHES[bundle.dataset][seed]:
        raise ValueError('Fixed row/pair identities differ from frozen source')
    panels = {}
    for name in ('P_A', 'P_B'):
        p = raw[name]
        ids = np.unique(np.concatenate((p['rank_indices'].ravel(), p['natural_indices'].ravel())))
        rank = [np.searchsorted(ids, rows) for rows in p['rank_indices']]
        pairs = [rank[q][pp] for q, pp in enumerate(p['rank_pairs'])]
        natural = [np.searchsorted(ids, rows) for rows in p['natural_indices']]
        panels[name] = Panel(bundle.train_x[ids], bundle.train_y[ids], rank, pairs, natural)
    p = raw['V_mech']
    for side, name in enumerate(('V_A', 'V_B')):
        ids = np.sort(p['indices'][p['validation_row_side'] == side])
        natural = [np.flatnonzero(bundle.val_x[ids, -1] == d) for d in range(3)]
        rank, pairs = [], []
        for q in range(6):
            d = q // 2
            local = p['pairs'][q, :p['pair_counts'][q]]
            local = local[(p['validation_row_side'][d, local] == side).all(1)]
            pairs.append(np.searchsorted(ids, p['indices'][d, local]))
            rank.append(natural[d])
        panels[name] = Panel(bundle.val_x[ids], bundle.val_y[ids], rank, pairs, natural)
    return panels, dict(source_hashes=hashes, spec=SPEC,
        V_mech_support=(p['indices'] >= 0).sum(1).tolist(),
        panel_support={k: dict(rows=len(v.x), rank=[len(r) for r in v.rank],
            pairs=[len(pp) for pp in v.pairs], natural=[len(r) for r in v.natural],
            fixed_pair_hash=digest([pp.tolist() for pp in v.pairs])) for k,v in panels.items()})


@torch.no_grad()
def predict(model, panels, names, counts):
    """Preserve per-module mode; panel models have no stateful eval operators."""
    modes = [m.training for m in model.modules()]
    device = next(model.parameters()).device
    model.eval()
    out = {}
    try:
        for name in names:
            x = panels[name].x
            p, z = forward_logits(model, torch.as_tensor(x, device=device).T)
            out[name] = {'p': p.cpu().numpy(), 'z': z.cpu().numpy()}
            counts['panel_forwards'] += 1
            counts['panel_forward_rows'] += len(x)
    finally:
        for module, mode in zip(model.modules(), modes):
            module.training = mode
    return out


def _ll(y, p):
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide='ignore'):
        return float(-(y*np.maximum(np.log(p), -100)+(1-y)*np.maximum(np.log1p(-p), -100)).mean())


def compare(panels, base, changed):
    result = {}
    for name, bank in changed.items():
        ref = base[name]['p']
        val = bank['p']
        panel = panels[name]
        cells = []
        for q in range(6):
            t, d = q % 2, q // 2
            rank = panel.rank[q]
            table = exact_transition_table(panel.y[rank, t], ref[rank, t], val[rank, t])
            all_stats = transition_metrics(table)
            pp = panel.pairs[q]
            def ordering(scores):
                a, b = scores[pp[:, 0], t], scores[pp[:, 1], t]
                return (a > b).astype(np.int64)*2 + (a == b)
            stat = np.bincount(3*ordering(ref)+ordering(val), minlength=9).reshape(3, 3)
            pair = transition_metrics(stat)
            if not all_stats['pairs'] or not pair['pairs']:
                raise ValueError(f'Missing measurable comparison in {name}/cell{q}')
            natural = panel.natural[d]
            ll_old = _ll(panel.y[natural, t], ref[natural, t])
            ll_new = _ll(panel.y[natural, t], val[natural, t])
            cells.append(dict(q=q, population_transition=table.tolist(),
                population=all_stats, sampled_transition=stat.tolist(), sampled=pair,
                logloss_reference=ll_old, logloss_changed=ll_new, delta_logloss=ll_new-ll_old))
        result[name] = dict(cells=cells,
            macro_pair_delta=float(np.mean([c['sampled']['delta_auc'] for c in cells])),
            macro_auc_delta=float(np.mean([c['population']['delta_auc'] for c in cells])),
            macro_ll_delta=float(np.mean([c['delta_logloss'] for c in cells])))
    return result


def selection_score(online):
    """The section 04 §7 source score uses population probability AUC on P_A."""
    return online['P_A']['macro_auc_delta']


def advance_average(window, scores, rho):
    for name, bank in scores.items():
        for key, value in bank.items():
            window[name][key] = rho*window[name][key]+(1-rho)*value


def window_banks(window):
    return {'WINDOW_PROB_EMA': {k: {'p': v['p']} for k,v in window.items()},
            'WINDOW_LOGIT_EMA': {k: {'p': expit(v['z'])} for k,v in window.items()}}
