"""Seed/stage summaries and exact-step joins of the frozen training diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean
from scipy.stats import spearmanr

CELLS = ('d0t0', 'd0t1', 'd1t0', 'd1t1', 'd2t0', 'd2t1')
STAGES = ('1-10', '11-30', '31-50')


def stage(epoch):
    if not 1 <= epoch <= 50:
        raise ValueError(f'Epoch outside frozen budget: {epoch}')
    return STAGES[0 if epoch <= 10 else 1 if epoch <= 30 else 2]


def finite(value):
    return value is not None and math.isfinite(value)


def average(values):
    values = [v for v in values if finite(v)]
    return mean(values) if values else None


def sign_fields(values, prefix):
    values = [v for v in values if finite(v)]
    n = len(values)
    return {prefix + '_count': n,
            prefix + '_positive_fraction': sum(v > 0 for v in values) / n if n else None,
            prefix + '_negative_fraction': sum(v < 0 for v in values) / n if n else None,
            prefix + '_zero_fraction': sum(v == 0 for v in values) / n if n else None}


def summarize_diagnostics(records):
    grouped = defaultdict(list)
    for record in records:
        for group, value in record['groups'].items():
            grouped[(group, stage(record['epoch']))].append(value)
    result = []
    for (group, phase), rows in sorted(grouped.items()):
        solves = [r['solve'] for r in rows if r.get('solve') is not None]
        residuals = [s['raw_residual'] for s in solves if finite(s.get('raw_residual'))]
        alphas = [r['alpha'] for r in rows if r.get('alpha') is not None]
        components = [v for a in alphas for v in a]
        cosines = [matrix[i][j] for r in rows for matrix in [r['cosines']]
                   for i in range(len(matrix)) for j in range(i + 1, len(matrix))]
        row = dict(group=group, stage=phase, diag_count=len(rows),
                   refresh_solve_count=len(solves), solver_failure_count=sum(not s['success'] for s in solves),
                   fallback_diag_count=sum(bool(r['fallback']) for r in rows),
                   raw_residual_count=len(residuals), raw_residual_max=max(residuals) if residuals else None,
                   alpha_diag_count=len(alphas), alpha_component_count=len(components),
                   alpha_sum_mean=average([sum(a) for a in alphas]),
                   alpha_min=min(components) if components else None,
                   alpha_max=max(components) if components else None, alpha_mean=average(components),
                   candidate_norm_mean=average([v for r in rows for v in r['candidate_norms']]),
                   candidate_cosine_offdiag_mean=average(cosines),
                   update_norm_mean=average([r['update_norm'] for r in rows]),
                   natural_adam_norm_mean=average([r['natural_adam_norm'] for r in rows]),
                   update_norm_ratio_mean=average([r['update_norm_ratio'] for r in rows]))
        for name in ('actual_utility', 'data_utility'):
            row.update(sign_fields([v for r in rows for v in r.get(name, [])], name))
        result.append(row)
    return result


def indexed(records):
    result = {}
    for record in records:
        key = (record['epoch'], record['step'])
        if key in result:
            raise ValueError(f'Duplicate epoch/step observation: {key}')
        result[key] = record
    return result


def utility_relation(utility, improvement):
    result = dict(spearman=None, sign_agreement=None, sign_compared=0,
                  utility_zero_count=None, improvement_zero_count=None, zero_pair_count=None)
    if utility is None or len(utility) != 6 or not all(finite(v) for v in utility):
        return dict(result, status='missing_six_cell_utility')
    if improvement is None or len(improvement) != 6 or not all(finite(v) for v in improvement):
        return dict(result, status='missing_cell_loss')
    nonzero = [(u, v) for u, v in zip(utility, improvement) if u != 0 and v != 0]
    result.update(sign_compared=len(nonzero),
                  sign_agreement=mean((u > 0) == (v > 0) for u, v in nonzero) if nonzero else None,
                  utility_zero_count=sum(v == 0 for v in utility),
                  improvement_zero_count=sum(v == 0 for v in improvement),
                  zero_pair_count=sum(u == 0 or v == 0 for u, v in zip(utility, improvement)))
    constant = len(set(utility)) < 2 or len(set(improvement)) < 2
    if not constant:
        result['spearman'] = float(spearmanr(utility, improvement).statistic)
    return dict(result, status='constant_rank_vector' if constant else 'complete')


def pair_observations(diag_records, short_records, arm):
    diag = {k: r for k, r in indexed(diag_records).items() if 'S' in r['groups']}
    short = indexed(short_records)
    pairs = []
    for key in sorted(diag.keys() & short.keys()):
        observations = short[key]['observations']
        combined = [o for o in observations if o['candidate'] == 'combined']
        if len(combined) > 1:
            raise ValueError(f'Duplicate combined short step: {key}')
        if not combined:
            continue
        value = diag[key]['groups']['S']
        labels = [o['candidate'] for o in observations if o['candidate'] != 'combined']
        members = sorted(int(v) for v in labels) if all(str(v).isdigit() for v in labels) else []
        six_members = arm in ('C1', 'C2') and members == list(range(6))
        loss = combined[0].get('loss_change')
        improvement = [-v if finite(v) else None for v in loss] if loss is not None else None
        row = dict(epoch=key[0], step=key[1], stage=stage(key[0]),
                   candidate_members=json.dumps(members), six_members=six_members,
                   loss_change=json.dumps(loss), actual_utility=json.dumps(value.get('actual_utility')),
                   data_utility=json.dumps(value.get('data_utility')))
        for i, cell in enumerate(CELLS):
            row['loss_change_' + cell] = loss[i] if loss is not None and len(loss) == 6 else None
        for name in ('actual', 'data'):
            relation = utility_relation(value.get(name + '_utility'), improvement) if six_members else dict(
                status='two_task_vector' if arm == 'C0' else 'incomplete_member_set',
                spearman=None, sign_agreement=None, sign_compared=0,
                utility_zero_count=None, improvement_zero_count=None, zero_pair_count=None)
            row.update({name + '_' + field: v for field, v in relation.items()})
        pairs.append(row)
    paired_keys = {(r['epoch'], r['step']) for r in pairs}
    coverage = dict(short_step_windows=len(short), diag_s_windows=len(diag), paired_windows=len(pairs),
                    unmatched_short_windows=len(short.keys() - paired_keys),
                    six_member_complete_loss_windows=sum(r['six_members'] and all(finite(r['loss_change_' + c]) for c in CELLS) for r in pairs),
                    actual_rank_windows=sum(r['actual_spearman'] is not None for r in pairs),
                    data_rank_windows=sum(r['data_spearman'] is not None for r in pairs),
                    short_step_epochs=json.dumps(sorted({k[0] for k in short})),
                    paired_epochs=json.dumps(sorted({r['epoch'] for r in pairs})),
                    paired_epoch_steps=json.dumps(sorted(paired_keys)),
                    unmatched_short_epoch_steps=json.dumps(sorted(short.keys() - paired_keys)))
    return pairs, coverage


def read_records(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, records):
    fields = list(dict.fromkeys(field for row in records for field in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def summarize(out):
    completion = json.loads((out / 'EXPERIMENT_COMPLETE.json').read_text())
    if completion['status'] != 'COMPLETE':
        raise ValueError('EXPERIMENT_COMPLETE status must be COMPLETE')
    diagnostics, pairs, coverage = [], [], []
    for dataset in ('movielens', 'kuairand'):
        for arm in ('C0', 'C1', 'C2'):
            for seed in range(2022, 2027):
                identity = dict(dataset=dataset, arm=arm, seed=seed)
                folder = out / 'formal' / f'{dataset}_{arm}_seed{seed}'
                diag = read_records(folder / 'DIAGNOSTICS.jsonl')
                short = read_records(folder / 'SHORT_STEP.jsonl')
                diagnostics.extend(dict(**identity, **r) for r in summarize_diagnostics(diag))
                joined, counts = pair_observations(diag, short, arm)
                pairs.extend(dict(**identity, **r) for r in joined)
                coverage.append(dict(**identity, **counts))
    tables = out / 'tables'
    tables.mkdir(exist_ok=True)
    for name, values in [('DIAGNOSTIC_SEEDS', diagnostics), ('MECHANISM_PAIRS', pairs), ('PAIR_COVERAGE', coverage)]:
        write_csv(tables / (name + '.csv'), values)
    return dict(status='COMPLETE',seed_stage_group_rows=len(diagnostics), paired_windows=len(pairs), seed_coverage_rows=len(coverage))


def self_check():
    value = dict(alpha=[1, 2, 3, 4, 5, 6], fallback=False, candidate_norms=[1, 2],
                 cosines=[[1, .25], [.25, 1]], update_norm=2, natural_adam_norm=4,
                 update_norm_ratio=.5, actual_utility=[0, 1, -1, 2, -2, 3],
                 data_utility=[1, 1, 2, 2, 3, 3], solve={'success':True, 'raw_residual':3e-5})
    diag = [dict(epoch=e, step=s, groups={'S':dict(value)}) for e, s in [(1, 1), (2, 12), (3, 21)]]
    diag[1]['groups']['S'].update(solve=None, raw_residual=999)
    diag[2]['groups']['S']['solve'] = {'success':False}
    def observed(epoch, step, loss):
        return dict(epoch=epoch,step=step, observations=[dict(candidate='combined',loss_change=loss)] + [dict(candidate=str(i)) for i in range(6)])
    short = [observed(1,1,[0,-1,1,-2,2,-3]), observed(2,11,[-1]*6), observed(3,21,[None,-1,1,-2,2,-3])]
    pairs, counts = pair_observations(diag, short, 'C1')
    assert [(r['epoch'],r['step']) for r in pairs] == [(1,1),(3,21)]
    assert counts['paired_windows']==2 and counts['unmatched_short_windows']==1
    assert json.loads(counts['paired_epochs'])==[1,3]
    assert pairs[0]['actual_spearman']==1 and pairs[0]['actual_sign_agreement']==1
    assert pairs[0]['actual_sign_compared']==5 and pairs[0]['actual_zero_pair_count']==1
    assert pairs[1]['actual_spearman'] is None and pairs[1]['actual_status']=='missing_cell_loss'
    tied = utility_relation([1,1,2,2,3,3],[1,1,2,2,3,3])
    assert tied['spearman']==1 and utility_relation([0]*6,[1]*6)['spearman'] is None
    two = [dict(epoch=1,step=1,observations=[dict(candidate='combined',loss_change=[-1]*6),dict(candidate='0'),dict(candidate='1')])]
    c0, _ = pair_observations([dict(epoch=1,step=1,groups={'S':dict(value,actual_utility=[1,2],data_utility=[2,3])})],two,'C0')
    assert json.loads(c0[0]['actual_utility'])==[1,2] and c0[0]['actual_spearman'] is None
    row = summarize_diagnostics(diag)[0]
    assert row['refresh_solve_count']==2 and row['solver_failure_count']==1
    assert row['raw_residual_count']==1 and row['raw_residual_max']==3e-5
    assert row['candidate_cosine_offdiag_mean']==.25 and row['diag_count']==3
    assert stage(10)=='1-10' and stage(11)=='11-30' and stage(30)=='11-30' and stage(31)=='31-50'
    return dict(status='PASS',checks=['exact_join','six_cell_rank','ties_zeros_missing','two_task_vector','actual_solve_residual','offdiagonal_cosine','stage_boundaries'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--out', type=Path, help='Completed experiment artifact root')
    choice.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_check() if args.self_check else summarize(args.out), sort_keys=True))
