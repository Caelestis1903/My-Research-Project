"""Five-seed B gates. Gate inputs are completed, independently committed raw files."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .b_panels import selection_score

from .b_observer import write_result
from .b_panel_identity import HASHES, digest

SEEDS = tuple(range(2022,2027))
EPOCHS = (5,20,35,50)


def _input(path):
    payload = Path(path).read_bytes()
    return json.loads(payload),dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(payload).hexdigest())


def _choice(values):
    values = np.asarray(values,dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all():
        raise ValueError('Need all six finite P_A candidate scores')
    q = int(values.argmax())
    return q if values[q]>0 else None


def _cells(row):
    return np.array([c['population']['delta_auc'] for c in row['cells']],dtype=float)


def _stats(values):
    a = np.array(values,dtype=float)
    if len(a)!=len(SEEDS) or not np.isfinite(a).all():
        raise ValueError('Incomplete five-seed effects')
    sd = a.std(axis=0,ddof=1)
    mu = a.mean(axis=0)
    half = sd*2.7764451051977987/math.sqrt(5)
    return dict(seed_order=SEEDS,by_seed=a.tolist(),mean=mu.tolist(),sample_sd=sd.tolist(),
                ci95=[(mu-half).tolist(),(mu+half).tolist()])


def core_gate(dataset,anchor_paths,output_dir):
    anchor_paths = list(anchor_paths)
    data, receipts, witnesses = {},{},[]
    for path in anchor_paths:
        raw,receipt = _input(path)
        i = raw['identity']
        seed,epoch = k = (i['seed'],i['epoch'])
        K = {'movielens':20,'kuairand':49}[dataset]
        if (i['dataset']!=dataset or k in data or seed not in SEEDS or epoch not in EPOCHS
            or i['panel_hashes']!=HASHES[dataset][seed] or i['global_step']!=epoch*K
            or i['virtual_joint']!=i['actual_joint'] or raw['counts']['virtual_adam']!=52
            or raw['counts']['source_sweeps']!=18 or len(raw['effects'])!=51):
            raise ValueError('Unusable B anchor identity/menu')
        families = {}
        for family in ('true','pseudo0','pseudo1'):
            rows = sorted((r for r in raw['effects'] if r['family']==family and r['scope']=='EMBED'
                           and r['mode']=='LOO'),key=lambda r:r['q'])
            if [r['q'] for r in rows]!=list(range(6)):
                raise ValueError('Incomplete source menu')
            scores = [selection_score(r['online']) for r in rows]
            choice = _choice(scores)
            if raw['selected'][family]['q']!=choice:
                raise ValueError('Stored choice differs from P_A-only deterministic choice')
            families[family] = dict(q=choice,scores=scores,confirm={
                panel:[_cells(r['online'][panel]) for r in rows] for panel in ('V_B','P_B')})
            if family=='true' and choice is not None:
                # For D = LOO-JOINT, adding the source benefits self when D_self < 0.
                for panel in ('P_B','V_B'):
                    D = families[family]['confirm'][panel][choice]
                    if D[choice]<0 and any(D[c]>0 for c in range(6) if c!=choice):
                        witnesses.append(dict(seed=seed,epoch=epoch,panel=panel,q=choice,
                                              deletion_delta_auc=D.tolist()))
        data[k] = families
        receipts[k] = receipt
    if set(data)!=set(itertools.product(SEEDS,EPOCHS)):
        raise ValueError('Core gate requires all 20 dataset anchors')
    from .b_tables import core_tables
    core_tables(dataset,anchor_paths,output_dir)
    folds = {}
    for seed in SEEDS:
        training = [(s,e) for s in SEEDS if s!=seed for e in (5,20)]
        means = np.mean([data[k]['true']['scores'] for k in training],axis=0)
        static = _choice(means)
        holdout = []
        for epoch in (35,50):
            f = data[(seed,epoch)]['true']
            vals = {}
            for panel in ('P_B','V_B'):
                def effect(q):
                    return np.zeros(6) if q is None else f['confirm'][panel][q]
                vals[panel]=dict(selected_minus_static=(effect(f['q'])-effect(static)).tolist(),
                    selected=effect(f['q']).tolist(),static=effect(static).tolist())
            holdout.append(dict(epoch=epoch,selected_q=f['q'],panels=vals))
        sources = [dict(seed=s,epoch=e,**receipts[(s,e)]) for s,e in training]
        folds[str(seed)] = dict(q_static=static,train_sources=sources,train_source_hash=digest(sources),
            P_A_mean_scores=means.tolist(),heldout=holdout)
    fixed = dict(dataset=dataset,folds=folds)
    write_result(Path(output_dir)/f'B_STATIC_LOSO_{dataset}.json',fixed)
    # Full core population for both controls, including each zero-action seed/anchor.
    selections = [dict(seed=s,epoch=e,choices={k: dict(q=f['q'],scores=f['scores'],
        V_B_selected_delta=(f['confirm']['V_B'][f['q']].tolist() if f['q'] is not None else [0.]*6))
        for k,f in data[(s,e)].items()}) for s,e in itertools.product(SEEDS,EPOCHS)]
    vs_static = [np.mean([r['panels']['V_B']['selected_minus_static']
                         for r in folds[str(s)]['heldout']]) for s in SEEDS]
    gate = dict(dataset=dataset,anchors=[receipts[k] for k in sorted(data)],
                B_PULSE='TRIGGERED' if witnesses else 'NOT_TRIGGERED',
                B_SEARCH='AWAITING_PULSE' if witnesses else 'NOT_TRIGGERED',
                all_choices=selections,independent_observed_harm=witnesses,
                heldout_V_B_selected_minus_static=_stats(vs_static),static_hash=digest(fixed))
    write_result(Path(output_dir)/f'B_GATE_{dataset}.json',gate)
    return gate


def search_gate(gate,pulse_paths,output_dir):
    dataset = gate['dataset']
    if gate['B_PULSE']!='TRIGGERED':
        if pulse_paths:
            raise ValueError('Pulse inputs outside a dataset B_PULSE trigger')
        return gate
    K = {'movielens':20,'kuairand':49}[dataset]
    choices = {(r['seed'],r['epoch']):r['choices'] for r in gate['all_choices']}
    runs,inputs = {},[]
    for path in pulse_paths:
        r,receipt = _input(path)
        k = (r['seed'],r['epoch'],r['family'])
        if (r['dataset']!=dataset or k in runs or r['q']!=choices[k[:2]][k[2]]['q'] or
                r['parent_panel_identity']!=HASHES[dataset][k[0]] or
                r['counts']['logical_steps']!=K or r['counts']['first_step_reused']!=1 or
                r['counts']['counterfactual_suffix_adam']!=(0 if r['q'] is None else K-1)):
            raise ValueError('Pulse report identity or cost mismatch')
        end = r['milestones'][str(K)]
        runs[k] = {name:_cells(end[name]['V_B']) for name in ('online','fixed_ema')}
        inputs.append(receipt)
    if set(runs)!=set(itertools.product(SEEDS,(20,35),('true','pseudo0'))):
        raise ValueError('Need all 20 pulse slots including NO_ACTION')
    summary = {}
    for readout in ('online','fixed_ema'):
        def by_family(family):
            return np.array([np.mean([runs[(s,e,family)][readout] for e in (20,35)],axis=0) for s in SEEDS])
        true,pseudo = by_family('true'),by_family('pseudo0')
        summary[readout] = dict(true_vs_joint_cells=_stats(true),pseudo_vs_joint_cells=_stats(pseudo),
            true_vs_joint_macro=_stats(true.mean(1)),true_vs_pseudo_macro=_stats((true-pseudo).mean(1)))
    online,F = summary['online'],summary['fixed_ema']
    controls = dict(actual_cross_cell_harm=bool(gate['independent_observed_harm']),
        future_true_beats_joint=online['true_vs_joint_macro']['mean']>0,
        future_true_beats_pseudo=online['true_vs_pseudo_macro']['mean']>0,
        heldout_true_beats_static=gate['heldout_V_B_selected_minus_static']['mean']>0,
        fixed_ema_residual=F['true_vs_joint_macro']['mean']>0,
        six_cell_protection=min(F['true_vs_joint_cells']['mean'])>=-.001,
        action_value=F['true_vs_joint_macro']['mean']>=.0001)
    if not all(controls[k] for k in list(controls)[:4]):
        reason='MECHANISM_WITHOUT_TRAIN_SIGNAL'
    elif not controls['fixed_ema_residual']:
        reason='FIXED_EMA_ALREADY_SUFFICIENT_FOR_THIS_ACTION'
    elif not controls['six_cell_protection']:
        reason='MACRO_GAIN_WITH_CELL_TRADEOFF'
    elif not controls['action_value']:
        reason='LOW_ACTION_VALUE'
    else:
        reason='TRIGGERED'
    result = dict(gate,B_SEARCH=reason,pulse_inputs=inputs,
                  pulse_seed_reductions=summary,search_conditions=controls)
    write_result(Path(output_dir)/f'B_GATE_{dataset}.json',result)
    return result
