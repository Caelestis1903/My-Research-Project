"""Summarize existing scalar traces; no model or dataset is loaded."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


def summarize(attempt, out):
    rows = []
    donors = {}
    for path in sorted((attempt / 'runs').glob('*/result.json')):
        result = json.loads(path.read_text())
        job = result['job']
        config = job['config']
        trace = {}
        for source in sorted(path.parent.glob('attempt*/batch_trace.jsonl')):
            for line in source.read_text().splitlines():
                record = json.loads(line)
                trace[record['step']] = record
        assert len(trace) == result['global_step'], (job['key'], len(trace))
        sums = {}
        maxima = {}
        teacher_sum = teacher_rows = cosine_sum = cosine_rows = 0
        for step, record in sorted(trace.items()):
            diagnostic = record['diagnostics']
            n = record['counts']['target_rows']
            for name, value in diagnostic.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    if name.endswith('_max'):
                        maxima[name] = max(maxima.get(name, -math.inf), value)
                    else:
                        sums[name] = sums.get(name, 0.) + value
            response = diagnostic.get('teacher_logit_change_abs_mean')
            if response is not None:
                teacher_sum += response * n
                teacher_rows += n
            cosine = diagnostic.get('task_direction_cosine_mean')
            if cosine is not None:
                cosine_sum += cosine * n
                cosine_rows += n
            if 'donor_identity_hash' in diagnostic:
                donors[(job['dataset'], job['seed'], config['arm'], step)] = diagnostic['donor_identity_hash']
        count = result['counts']
        target = count['target_rows']
        views = count['extra_view_rows']
        fields = 6 if job['dataset'] == 'movielens' else 33
        row = dict(dataset=job['dataset'], split_id=job['split_id'], arm=config['arm'],
                   kappa=config['kappa'], seed=job['seed'], key=job['key'],
                   worker_seconds=result['worker_seconds'], **count, **maxima)
        for name in ('donor_covered', 'double_donor_rows', 'no_donor_rows'):
            row[name + '_rate'] = sums[name] / target if name in sums and target else None
        field_rows = sums.get('field_rows', 0)
        row['fallback_rate'] = sums.get('fallback_rows', 0) / field_rows if field_rows else None
        row['mean_projection_ratio'] = (sums['projection_ratio_sum'] / field_rows
            if field_rows and config['arm'] in ('F1', 'FR') else None)
        row['cap_rate'] = sums.get('capped_rows', 0) / field_rows if field_rows else None
        for rank in range(3):
            row[f'rank{rank}_rate'] = sums.get(f'rank{rank}_rows', 0) / field_rows if field_rows and config['arm'] != 'F0' else None
        row['selected_response_mean'] = (sums['selected_score_sum'] / sums['score_rows']
            if config['arm'] in ('J1', 'J_SUM') and sums.get('score_rows') else None)
        row['perturbation_energy'] = sums.get('perturbation_energy', 0.)
        row['nonzero_view_rate'] = sums.get('nonzero_view_rows', 0.) / views if views else None
        row['actual_perturbation_rms'] = math.sqrt(row['perturbation_energy'] / (views * fields * 16)) if views else None
        row['teacher_finite_response_mean'] = teacher_sum / teacher_rows if teacher_rows else None
        row['teacher_response_diagnostic_rows'] = teacher_rows
        row['task_direction_cosine_mean'] = cosine_sum / cosine_rows if cosine_rows else None
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'diagnostics_by_run.csv', index=False)
    pairs = []
    for dataset, seed in sorted({(d, s) for d, s, _, _ in donors}):
        a = {step: value for (d, s, arm, step), value in donors.items()
             if (d, s, arm) == (dataset, seed, 'J0')}
        b = {step: value for (d, s, arm, step), value in donors.items()
             if (d, s, arm) == (dataset, seed, 'J1')}
        common = sorted(a.keys() & b.keys())
        pairs.append(dict(dataset=dataset, seed=seed, J0_steps=len(a), J1_steps=len(b),
                          matched_steps=len(common), equal_donor_hashes=sum(a[s] == b[s] for s in common)))
    (out / 'donor_pairing.json').write_text(json.dumps(pairs, indent=2) + '\n')
    print(json.dumps(dict(completed_runs=len(rows), donor_pairs=pairs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'tables')
    args = parser.parse_args()
    summarize(args.attempt, args.out)
