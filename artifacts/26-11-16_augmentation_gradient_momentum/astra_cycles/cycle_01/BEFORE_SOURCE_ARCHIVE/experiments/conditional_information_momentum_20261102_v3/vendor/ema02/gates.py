"""Mechanical preregistered predicates; the orchestrator approves each dispatch."""
import copy
import itertools
import numpy as np
from scipy.stats import t as student_t

DEV = (2022, 2024)
HELD = (2023, 2025, 2026)
SIMPLE = ('J0', 'F0', 'RG', 'AG', 'J_SUM', 'F_PR')
ORDER = ('J0', 'F0', 'RG', 'AG', 'J_SUM', 'F_PR', 'J1', 'F1', 'TG')


def paired(candidate, reference, metric='macro_auc'):
    seeds = sorted(set(candidate) & set(reference))
    values = [candidate[s][metric]-reference[s][metric] for s in seeds]
    mean = float(np.mean(values)) if values else None
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    width = float(student_t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values))) if sd is not None else None
    return dict(seeds=seeds, values=values, n=len(values), mean=mean, sample_sd=sd,
        ci95=None if width is None else [mean-width, mean+width], positive=sum(v>0 for v in values))


def protection(candidate, reference, adverse_required):
    bad = []
    for metric in ['macro_logloss']+[f'd{d}t{t}_auc' for d in range(3) for t in range(2)]:
        p = paired(candidate, reference, metric)
        if p['mean'] is None: continue
        adverse = sum(v > 0 if metric == 'macro_logloss' else v < 0 for v in p['values'])
        if adverse >= adverse_required and (p['mean'] >= .002 if metric == 'macro_logloss' else p['mean'] <= -.002):
            bad.append(dict(metric=metric, **p))
    return bad


def table(results, dataset, arm, kappa=None, seeds=DEV, split_id='original'):
    selected = [r for r in results if r['job']['dataset'] == dataset and r['job']['config']['arm'] == arm
        and r['job']['split_id'] == split_id and r['job']['seed'] in seeds
        and (kappa is None or r['job']['config']['kappa'] == kappa)]
    assert len({r['job']['seed'] for r in selected}) == len(selected), 'Ambiguous arm/config comparison'
    return {r['job']['seed']: r['selected_validation']['fixed_ema'] for r in selected}


def strongest(results, dataset, arms, kappa, costs=None):
    choices = []
    for arm in arms:
        values = table(results, dataset, arm, kappa)
        if set(values) == set(DEV):
            choices.append((arm, float(np.mean([v['macro_auc'] for v in values.values()]))))
    if not choices: return None
    maximum = max(v for _,v in choices)
    close = [a for a,v in choices if maximum-v < .0005]
    return min(close, key=lambda a: ((costs or {}).get((dataset,a), float('inf')), ORDER.index(a) if a in ORDER else 99))


def development(results, dataset, arm, kappa, reference=None):
    c = table(results, dataset, arm, kappa); a0 = table(results, dataset, 'A0')
    a1 = table(results, dataset, 'A1')
    needed = ['A0', 'A1'] + ([reference] if reference else [])
    refs = dict(A0=a0, A1=a1)
    if reference: refs[reference] = table(results, dataset, reference, kappa)
    if set(c) != set(DEV) or any(set(refs[a]) != set(DEV) for a in needed):
        return dict(status='BUDGET_STOP_UNMATCHED', arm=arm, missing_controls=needed)
    effects = {a: paired(c, r) for a,r in refs.items()}
    guards = {a: protection(c, refs[a], 2) for a in ['A0']+([reference] if reference else [])}
    harm = any(guards.values())
    full = (effects['A0']['mean'] >= .001 and effects['A0']['positive'] == 2 and
            all(effects[a]['mean'] >= .0005 and effects[a]['positive'] == 2 for a in needed if a != 'A0'))
    boundary = (arm in ('J1','F1') and .0005 <= effects['A0']['mean'] < .001 and
        effects['A0']['positive'] == 2 and reference and effects[reference]['mean'] > 0)
    status = 'TRADEOFF' if harm else 'GO_CONFIRM' if full else 'CALIBRATE_ONCE' if boundary else 'STOP_NO_INCREMENT'
    return dict(status=status, arm=arm, kappa=kappa, reference=reference, effects=effects, guards=guards)


def held_gate(results, dataset, winner, kappa, reference):
    refs = ['A0','A1']+([reference] if reference not in ('A0','A1',None) else [])
    effects = {}; guards = {}
    for seeds, label, positive, adverse in [(HELD,'held',2,2), (tuple(range(2022,2027)),'combined',4,4)]:
        c = table(results, dataset, winner, kappa, seeds)
        for ref in refs:
            r = table(results, dataset, ref, None if ref in ('A0','A1') else kappa, seeds)
            if set(c) != set(seeds) or set(r) != set(seeds): return dict(status='BUDGET_STOP_UNMATCHED')
            e = paired(c,r); effects[label+'_'+ref] = e
            if ref in ('A0',reference): guards[label+'_'+ref] = protection(c,r,adverse)
    passed = all(e['positive'] >= (2 if name.startswith('held') else 4) and
        (not name.startswith('held') or e['mean'] >= (.001 if name.endswith('_A0') else .0005))
        for name,e in effects.items())
    return dict(status='TRADEOFF' if any(guards.values()) else 'GO_WINDOW' if passed else 'STOP_NO_INCREMENT',
                effects=effects,guards=guards)


def s1r_recommendation(results, datasets):
    out = {}
    for dataset in datasets:
        f = table(results,dataset,'F1'); a0 = table(results,dataset,'A0'); f0 = table(results,dataset,'F0')
        if not all(set(v) == set(DEV) for v in (f,a0,f0)): continue
        a = paired(f,a0); b = paired(f,f0)
        guards = protection(f,a0,2)+protection(f,f0,2)
        out[dataset] = dict(triggered=a['mean'] >= .0005 and a['positive'] == 2 and b['mean'] > 0 and not guards,
                            vs_A0=a,vs_F0=b,guards=guards)
    return out


def development_summary(results, p, costs=None):
    out = {}
    for dataset, conf in p['datasets'].items():
        k = conf['kappa']; rows = []
        for arm, pool in [('J0',()),('F0',()),('J1',('J0','J_SUM')),('F1',('F0','FR','F_PR')),
                          ('RG',()),('AG',()),('TG',('AG','RG')),('J_SUM',('J0',)),('F_PR',('F0','FR'))]:
            if set(table(results,dataset,arm,k)) != set(DEV): continue
            ref = strongest(results,dataset,pool,k,costs) if pool else None
            row = development(results,dataset,arm,k,ref)
            if arm == 'F1' and row['status'] in ('GO_CONFIRM','CALIBRATE_ONCE') and set(table(results,dataset,'FR',k)) != set(DEV):
                row['status'] = 'BUDGET_STOP_UNMATCHED'
            if arm == 'TG' and any(set(table(results,dataset,a,k)) != set(DEV) for a in ('AG','RG')):
                row['status'] = 'BUDGET_STOP_UNMATCHED'
            rows.append(row)
        out[dataset] = rows
    return out


def select_winner(rows, results, dataset, costs=None):
    eligible = [r for r in rows if r['status'] == 'GO_CONFIRM' and r['arm'] != 'FR']
    if not eligible: return None
    means = {r['arm']: float(np.mean([v['macro_auc'] for v in table(results,dataset,r['arm'],r['kappa']).values()])) for r in eligible}
    top = max(means.values()); close = [r for r in eligible if top-means[r['arm']] < .0005]
    return min(close,key=lambda r: ((costs or {}).get((dataset,r['arm']),float('inf')), ORDER.index(r['arm'])))
