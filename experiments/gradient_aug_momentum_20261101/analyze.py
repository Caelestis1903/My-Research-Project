"""Seed-paired numerical tables and prespecified gates; no scientific interpretation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import protocol, write_json


def summarize(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = 2.7764451051977987*sd/np.sqrt(5)
    return dict(n=5, mean=mean, std=sd, ci_low=mean-half, ci_high=mean+half,
                positive=int((x > 0).sum()), negative=int((x < 0).sum()))


def gain_gate(values):
    x = np.asarray(values)
    return bool(x.mean() >= .001 and (x > 0).sum() >= 4)


def phase1_gate(results, p):
    decisions = {}
    for dataset in p['datasets']:
        lookup = {(r['arm'], r['seed']):r for r in results if r['dataset'] == dataset}
        changes, seed_means, stage_changes, misaligned = [], [], [], []
        for seed in p['seeds']:
            a, b = lookup['A3', seed], lookup['A1', seed]
            changes.append(a['selected_validation']['online']['macro_auc']-b['selected_validation']['online']['macro_auc'])
            table = pd.DataFrame(a['diagnostics'])
            assert len(table) == 50 and table.epoch.tolist() == list(range(1, 51))
            injection = table.injection_ratio.to_numpy(float)
            seed_means.append(float(injection.mean()))
            stages = [table[table.epoch.between(lo, hi)] for lo, hi in p['decision']['p2']['stages']]
            means = [float(s.injection_ratio.mean()) for s in stages]
            stage_changes.append(bool(min(means) > 1e-12 and max(means)/min(means) >= 2))
            early, late = stages[0], stages[-1]
            c0, c1 = float(early.q_delta_cosine.mean()), float(late.q_delta_cosine.mean())
            misaligned.append(bool(c0 >= .1 and c1 <= -.1 and means[0] >= .05 and means[-1] >= .05))
        cv = float(np.std(seed_means, ddof=1)/np.mean(seed_means)) if np.mean(seed_means) > 1e-12 else None
        passed = gain_gate(changes)
        unstable = bool((cv is not None and cv >= .5) or sum(stage_changes) >= 4)
        positive_gate = bool(np.mean(changes) > 0 and sum(v > 0 for v in changes) >= 3 and not passed and unstable)
        mismatch_gate = bool(not passed and sum(misaligned) >= 4)
        decisions[dataset] = dict(triggered=positive_gate or mismatch_gate,
            status='TRIGGERED' if positive_gate or mismatch_gate else 'NOT_TRIGGERED',
            validation_A3_minus_A1=changes, validation_gain_gate=passed,
            seed_injection_means=seed_means, seed_injection_cv=cv,
            repeated_stage_change=stage_changes, repeated_misalignment=misaligned,
            conditional_positive_unstable=positive_gate, repeated_mismatch=mismatch_gate)
    return dict(study_id=p['study_id'], input='selected validation and last_actual_batch D1 only',
                test_accessed=False, datasets=decisions, triggered=any(d['triggered'] for d in decisions.values()))


def guards(deltas, dataset):
    largest = 2 if dataset == 'movielens' else 1
    bad = []
    for metric in [f'd{largest}t0_auc', f'd{largest}t1_auc', 'pooled_task_mean_auc', 'worst_cell_auc']:
        x = np.asarray(deltas[metric])
        if x.mean() <= -.002 and (x < 0).sum() >= 4:
            bad.append(metric)
    x = np.asarray(deltas['macro_logloss'])
    if x.mean() >= .002 and (x > 0).sum() >= 4:
        bad.append('macro_logloss')
    return bad


def method_verdict(method, record):
    guard_splits = {split:record[split]['guards'] for split in ('validation','test')
                    if record[split]['guards']}
    development_passed = (record['validation']['gain_gate'] if method == 'P0' else
                          all(c['macro_auc']['mean'] > 0 for c in record['validation']['comparisons']))
    status = ('TRADEOFF' if guard_splits else
              'GO' if development_passed and record['test']['gain_gate'] else 'NO_PROMOTION')
    return dict(status=status, development_gate_passed=development_passed, guard_splits=guard_splits)


def build_tables(run_dir, p):
    root = Path(run_dir)
    pairs, summaries, arm_rows = [], [], []
    contrasts = [('P0_A0_shadow-online', 'A0', 'shadow', 'A0', 'online')]
    for kind in ('online', 'shadow'):
        for left, right in [('A1','A0'), ('A2','A0'), ('A3','A0'), ('A2','A1'), ('A3','A1'), ('A3','A2')]:
            contrasts.append((f'{left}-{right}_{kind}', left, kind, right, kind))
    contrasts.extend((f'{a}_shadow-online', a, 'shadow', a, 'online') for a in ('A1','A2','A3'))
    for split in ('validation', 'test'):
        frame = pd.read_csv(root/f'{split}_metrics.csv')
        assert not frame.duplicated(['dataset','arm','seed','model_kind','checkpoint']).any()
        metric_names = [k for k in frame if k.endswith(('_auc','_logloss'))]
        for dataset in p['datasets']:
            for checkpoint in ('selected','E10','E50'):
                table = frame[(frame.dataset == dataset)&(frame.checkpoint == checkpoint)].set_index(['arm','model_kind','seed'])
                for metric in metric_names:
                    for (arm, kind), values in table.groupby(level=[0,1]):
                        arm_rows.append(dict(split=split, dataset=dataset, checkpoint=checkpoint,
                            arm=arm, model_kind=kind, metric=metric, **summarize(values.sort_index()[metric])))
                    for effect, left, lk, right, rk in contrasts:
                        changes = []
                        for seed in p['seeds']:
                            av, bv = float(table.loc[(left,lk,seed),metric]), float(table.loc[(right,rk,seed),metric])
                            changes.append(av-bv)
                            pairs.append(dict(split=split, dataset=dataset, checkpoint=checkpoint, effect=effect,
                                arm=left, model_kind=lk, reference_arm=right, reference_kind=rk, metric=metric,
                                seed=seed, arm_value=av, reference_value=bv, value=av-bv))
                        summaries.append(dict(split=split, dataset=dataset, checkpoint=checkpoint,
                                              effect=effect, metric=metric, **summarize(changes)))
    paired = pd.DataFrame(pairs)
    paired.to_csv(root/'paired_effects.csv', index=False)
    pd.DataFrame(summaries).to_csv(root/'effect_summary.csv', index=False)
    pd.DataFrame(arm_rows).to_csv(root/'arm_summary.csv', index=False)
    decisions = {}
    for dataset in p['datasets']:
        decisions[dataset] = {}
        for method, effects in [('P0',['P0_A0_shadow-online']), ('P1_A2',['A2-A0_online','A2-A1_online']),
                                ('P1_A3',['A3-A0_online','A3-A1_online'])]:
            record = {}
            for split in ('validation','test'):
                checks = []
                for effect in effects:
                    rows = paired[(paired.dataset == dataset)&(paired.checkpoint == 'selected')&
                                  (paired.effect == effect)&(paired.split == split)]
                    deltas = {metric:g.sort_values('seed').value.to_numpy() for metric,g in rows.groupby('metric')}
                    checks.append(dict(effect=effect, gain_gate=gain_gate(deltas['macro_auc']),
                        macro_auc=summarize(deltas['macro_auc']), guards=guards(deltas,dataset)))
                record[split] = dict(gain_gate=all(c['gain_gate'] for c in checks),
                                     guards=sorted({m for c in checks for m in c['guards']}), comparisons=checks)
            record.update(method_verdict(method,record))
            if method == 'P1_A2':
                record['interpretation_role'] = 'algebraically_equivalent_implementation_control'
            decisions[dataset][method] = record
    p2 = json.loads((root/'phase1_validation_decision.json').read_text())
    write_json(root/'decision.json', dict(study_id=p['study_id'], methods=decisions, P2=p2,
                                         stop_after_authorized_round=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); build_tables(args.run_dir, protocol())
