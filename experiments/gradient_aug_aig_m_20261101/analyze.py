"""Numerical pairing, selection, guards and finite-contract verdicts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import DEV_SEEDS, SHADOWS, TRIAL_ORDER, write_json


def summarize(values):
    x = np.asarray(values,dtype=float)
    n = len(x); assert n in (2,5)
    sd = float(x.std(ddof=1)); half = {2:12.706204736432095,5:2.7764451051977987}[n]*sd/np.sqrt(n)
    average = float(x.mean())
    return dict(n=n,mean=average,sample_sd=sd,ci95_low=average-half,ci95_high=average+half,
                positive=int((x>0).sum()),negative=int((x<0).sum()))


def compare(candidate, reference, dataset):
    assert set(candidate) == set(reference) and len(candidate) in (2,5)
    metrics = [k for k in next(iter(candidate.values())) if k.endswith(('_auc','_logloss'))]
    deltas = {k:[candidate[s][k]-reference[s][k] for s in sorted(candidate)] for k in metrics}
    adverse = 2 if len(candidate)==2 else 4
    largest = 2 if dataset=='movielens' else 1
    guards = []
    for key in ['worst_cell_auc',f'd{largest}t0_auc',f'd{largest}t1_auc']:
        x = np.asarray(deltas[key])
        if x.mean() <= -.002 and (x<0).sum() >= adverse:
            guards.append(key)
    x = np.asarray(deltas['macro_logloss'])
    if x.mean() >= .002 and (x>0).sum() >= adverse:
        guards.append('macro_logloss')
    auc = np.asarray(deltas['macro_auc'])
    gain = bool(auc.mean()>=.001 and (auc>0).sum()>=adverse)
    exchanges = [k for k in ('pooled_task_mean_auc','sample_weighted_cell_auc')
        if auc.mean()>0 and np.mean(deltas[k])<0]
    return dict(gain_gate=gain,guards=guards,exchanges=exchanges,passed=gain and not guards,
                deltas=deltas,summary={k:summarize(v) for k,v in deltas.items()})


def selected_values(results, dataset, trial, kind):
    return {r['seed']:r['selected_validation'][kind] for r in results
            if r['dataset']==dataset and r['trial']==trial}


def choose(stage, results, dataset, allowed=None):
    choices = []
    for trial in TRIAL_ORDER[stage]:
        for kind in SHADOWS:
            if allowed is not None and (trial,kind) not in allowed:
                continue
            values = selected_values(results,dataset,trial,kind)
            assert sorted(values)==DEV_SEEDS
            choices.append(dict(trial=trial,kind=kind,mean=float(np.mean([v['macro_auc'] for v in values.values()]))))
    if not choices:
        return None
    highest = max(c['mean'] for c in choices)
    # The generated ordering implements the frozen equal-cost simplicity tie rule.
    return next(c for c in choices if highest-c['mean']<.0005)


def build_tables(run_dir):
    out = Path(run_dir)
    paired, summaries, arms = [], [], []
    for split in ('validation','test'):
        frame = pd.read_csv(out/f'{split}_metrics.csv')
        assert not frame.duplicated(['dataset','arm','seed','checkpoint']).any()
        metrics = [k for k in frame if k.endswith(('_auc','_logloss'))]
        for (dataset,checkpoint), table in frame.groupby(['dataset','checkpoint']):
            index = table.set_index(['arm','seed'])
            contrasts = [('EMA_A0','A0_EMA','A0'),('augmentation','A1_EMA','A0_EMA'),
                         ('A1_EMA_minus_A0','A1_EMA','A0')]
            if 'AIG_EMA' in set(table.arm):
                contrasts += [('AIG_independent','AIG_EMA','A1_EMA'),('AIG_minus_A0_EMA','AIG_EMA','A0_EMA')]
            for metric in metrics:
                for arm, values in table.groupby('arm'):
                    arms.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,arm=arm,metric=metric,
                                     **summarize(values.sort_values('seed')[metric])))
                for effect,left,right in contrasts:
                    deltas = []
                    for seed in range(2022,2027):
                        a,b = float(index.loc[(left,seed),metric]),float(index.loc[(right,seed),metric])
                        deltas.append(a-b)
                        paired.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,
                            metric=metric,seed=seed,candidate=a,reference=b,delta=a-b))
                    summaries.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,
                                          metric=metric,**summarize(deltas)))
    pd.DataFrame(paired).to_csv(out/'paired_effects.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'effect_summary.csv',index=False)
    pd.DataFrame(arms).to_csv(out/'arm_summary.csv',index=False)
    validation = pd.read_csv(out/'validation_metrics.csv')
    test = pd.read_csv(out/'test_metrics.csv')
    phase = json.loads((out/'S4_decision.json').read_text())
    decision = {}
    for dataset in ('movielens','kuairand'):
        decision[dataset] = {}
        for method,left,right in [('EMA','A0_EMA','A0'),('augmentation','A1_EMA','A0_EMA'),
                                  ('AIG','AIG_EMA','A1_EMA')]:
            if left not in set(test[test.dataset==dataset].arm):
                decision[dataset][method] = dict(status='STOP_DEVELOPMENT',development=phase['datasets'][dataset])
                continue
            record = {}
            for split,frame in [('validation',validation),('test',test)]:
                part = frame[(frame.dataset==dataset)&(frame.checkpoint=='selected')]
                values = {a:{int(r['seed']):r for r in part[part.arm==a].to_dict('records')} for a in (left,right)}
                record[split] = compare(values[left],values[right],dataset)
            development = phase['datasets'][dataset]['promoted'] if method=='AIG' else record['validation']['gain_gate']
            guard_splits = {s:r['guards'] for s,r in record.items() if r['guards']}
            stable_exchange = any(np.mean(record['test']['deltas'][k])<0 and
                sum(x<0 for x in record['test']['deltas'][k])>=4 for k in record['test']['exchanges'])
            status = ('TRADEOFF' if guard_splits or stable_exchange else
                      'GO' if development and record['test']['gain_gate'] else 'NO_PROMOTION')
            decision[dataset][method] = dict(status=status,development_gate_passed=development,
                guard_splits=guard_splits,stable_pooled_or_traffic_exchange=stable_exchange,**record)
    write_json(out/'decision.json',dict(study_id='AIGM01',datasets=decision,stop_after_finite_budget=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    build_tables(parser.parse_args().run_dir)
