"""Apply the frozen S1 validation rule to complete paired trajectories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.conditional_information_momentum_20261102_v3.run import job


def decide(attempt, report):
    records = [json.loads(p.read_text()) for p in (attempt/'runs').glob('*/result.json')]
    records = [r for r in records if r['job']['stage'] == 'S1']
    assert len(records) == 24 and all(r['epochs'] == 50 for r in records)
    index = {(r['job']['dataset'], r['job']['config']['arm'], r['job']['config']['lambda_pair'], r['job']['seed']): r['selected_validation']['fixed_ema'] for r in records}
    assert len(index) == 24
    decisions = {}; jobs = []
    for dataset in ('movielens', 'kuairand'):
        comparisons = []
        for arm in ('P00', 'P11'):
            for lam in (.05, .2):
                values = [index[dataset, arm, lam, s] for s in (2022, 2024)]
                base = [index[dataset, 'A0', 0., s] for s in (2022, 2024)]
                differences = {k: [v[k]-b[k] for v, b in zip(values, base)] for k in values[0] if k.endswith(('_auc', '_logloss'))}
                protection = []
                for metric in ['macro_logloss'] + [f'd{d}t{t}_auc' for d in range(3) for t in range(2)] + ['sample_weighted_cell_auc', 'pooled_task_mean_auc']:
                    x = np.array(differences[metric]); adverse = x if metric.endswith('logloss') else -x
                    if adverse.mean() >= .002 and (adverse > 0).all():
                        protection.append(metric)
                comparisons.append(dict(arm=arm, lambda_pair=lam,
                    validation_macro_auc_mean=float(np.mean([v['macro_auc'] for v in values])),
                    mean_delta_auc=float(np.mean(differences['macro_auc'])),
                    positive_seeds=sum(x > 0 for x in differences['macro_auc']),
                    protection=protection, paired_differences=differences))
        selected = {}
        for arm in ('P00', 'P11'):
            candidates = [c for c in comparisons if c['arm'] == arm]
            safe = [c for c in candidates if not c['protection']]
            selected[arm] = max(safe or candidates, key=lambda c: (c['validation_macro_auc_mean'], -c['lambda_pair']))['lambda_pair']
        proceed = any(c['mean_delta_auc'] > 0 for c in comparisons)
        decisions[dataset] = dict(S1_complete=12, S2='TRIGGERED' if proceed else 'NOT_TRIGGERED',
            reason='At least one positive mean validation trend' if proceed else 'All candidate mean validation AUC differences nonpositive; none 2/2 positive',
            selected_lambda=selected, comparisons=comparisons)
        if proceed:
            jobs.extend(job(dataset, arm, seed, 'S2', lambda_pair=selected['P11']) for arm in ('P10', 'P01', 'PCV', 'PCOLL') for seed in (2022, 2024))
    decision = dict(approved_by='orchestrator', stage='S1', test_accessed=False, datasets=decisions)
    (report/'DEVELOPMENT_DECISION_S1.json').write_text(json.dumps(decision, indent=2)+'\n')
    (attempt/'S2_DISPATCH.json').write_text(json.dumps(dict(stage='S2', approved_by='orchestrator', jobs=jobs), indent=2)+'\n')
    print(json.dumps({d: {k: v for k, v in x.items() if k != 'comparisons'} for d, x in decisions.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--attempt', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    a = p.parse_args(); decide(a.attempt, a.report)
