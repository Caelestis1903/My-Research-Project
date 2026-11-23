"""Scalar-only S1 tables and the preregistered Validation decision."""

import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from experiments.conditional_history_shared_update_20261122.protocol import sha, write
from experiments.conditional_history_shared_update_20261122.results import paired_stats, write_csv

from .diagnostics import ANCHORS, CELLS, DATASETS, FORKS, IDENTITY, PANELS, READOUTS, SCOPES, SEEDS


CONTROLS = ('U_ONLY', 'P→U', 'J_ACC', 'J2')
SELECTION = CONTROLS[:3]
CONFIRMATION = CONTROLS[1:3]
MECHANISM = ('shared_update', 'post_u_recovery')
GUARD = -.001
AUC_METRICS = ('macro_auc', *(cell + '_auc' for cell in CELLS))
EQUAL_DATASET = 'equal_dataset'
AGGREGATE = 'anchor_mean'


def _mean(values):
    values = tuple(values)
    return None if None in values else fmean(values)


def _delta(left, right):
    return None if left is None or right is None else left - right


def _summary(deltas):
    if any(delta is None for delta in deltas.values()):
        return dict(n=sum(delta is not None for delta in deltas.values()), mean_delta=None, sd_delta=None,
                    ci95_low=None, ci95_high=None, positive=None, negative=None, zero=None,
                    **{f'delta_{seed}': delta for seed, delta in deltas.items()})
    out = paired_stats(deltas, dict.fromkeys(SEEDS, 0.0))
    return {key: value for key, value in out.items() if not key.endswith(('_left', '_right'))}


def _expected_endpoints():
    return {
        (arm, scope, orientation, phase, readout, panel)
        for arm, scope, orientation, phase, readouts in (
            ('ANCHOR', None, None, 'anchor', READOUTS),
            *((arm, scope, orientation, 'end', READOUTS) for arm, scope, orientation in FORKS),
            *((arm, scope, orientation, 'after_u', ('online',)) for arm, scope, orientation in FORKS if arm == 'U→P'),
        ) for readout in readouts for panel in PANELS
    }


def _read_json(path):
    return json.loads(path.read_text())


def _index_endpoints(artifact_dir):
    manifest = _read_json(artifact_dir / 'S1_RUN_MANIFEST.json')
    status = _read_json(artifact_dir / 'S1_RUN.json')
    if status['status'] != 'COMPLETE':
        raise ValueError('All ten S1 trajectories must finish before scope selection')
    jobs = {(job['dataset'], job['seed']): job for job in manifest['jobs']}
    if (len(manifest['jobs']) != len(jobs) or set(jobs) != {(d, s) for d in DATASETS for s in SEEDS}
            or any(job['arm'] != 'J_ORIG' for job in jobs.values())
            or len({job['logical_id'] for job in jobs.values()}) != len(jobs)):
        raise ValueError('S1 manifest requires one canonical trajectory for each dataset and seed')
    tables = {'endpoints': [], 'transitions': [], 'responses': []}
    index, panel_hashes = {}, {}
    for (dataset, seed), job in sorted(jobs.items()):
        run_dir = artifact_dir / 'runs' / job['logical_id']
        result = _read_json(run_dir / 'result.json')
        if result['job'] != job or result['epochs'] != 50 or result['test_evaluations'] != 0:
            raise ValueError(f'S1 requires a complete and sealed result for {dataset}/{seed}')
        for epoch in ANCHORS:
            path = run_dir / f'forks_E{epoch}.json'
            if result['fork_hashes'][path.name] != sha(path):
                raise ValueError(f'S1 fork identity differs from completion for {dataset}/{seed}/E{epoch}')
            receipt = _read_json(path)
            meta = receipt['metadata']
            expected = {key: job[key] for key in IDENTITY if key not in ('epoch', 'attempt_id')}
            expected.update(epoch=epoch, attempt_id=meta['attempt_id'])
            if not meta['attempt_id'] or any(meta.get(key) != value for key, value in expected.items()):
                raise ValueError(f'S1 fork metadata differs for {dataset}/{seed}/E{epoch}')
            panel_hash = receipt['panel_hash']
            if not panel_hash or panel_hash != panel_hashes.setdefault(dataset, panel_hash):
                raise ValueError(f'S1 panel changed for {dataset}')
            purity = receipt['purity']
            fork_states = purity['fork_states']
            fork_keys = {(row['arm'], row['scope'], row['orientation']) for row in fork_states}
            if (len(fork_states) != len(FORKS) or fork_keys != set(FORKS)
                    or purity['live_before'] != purity['live_after']
                    or any(row['live_after'] != purity['live_before'] for row in fork_states)):
                raise ValueError(f'S1 fork purity proof differs for {dataset}/{seed}/E{epoch}')
            for table, rows in tables.items():
                for row in receipt[table]:
                    if any(row.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'Wrong S1 {table} row identity for {dataset}/{seed}/E{epoch}')
                    rows.append(row)
            endpoints = {}
            for row in receipt['endpoints']:
                key = tuple(row[k] for k in ('arm', 'scope', 'orientation', 'phase', 'readout', 'panel'))
                if key in endpoints or row['panel_hash'] != panel_hash:
                    raise ValueError(f'S1 duplicate endpoint or wrong panel for {dataset}/{seed}/E{epoch}')
                for metric in AUC_METRICS:
                    value = row[metric]
                    if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                        raise ValueError(f'Invalid S1 AUC for {dataset}/{seed}/E{epoch}/{key}/{metric}')
                endpoints[key] = row
            if endpoints.keys() != _expected_endpoints():
                raise ValueError(f'Incomplete or unregistered endpoint arms for {dataset}/{seed}/E{epoch}')
            index[dataset, seed, epoch] = endpoints
    return index, jobs, tables


def _paired(index, dataset, seed, epoch, panel, readout, scope, control, metric):
    rows = index[dataset, seed, epoch]
    orders = ('AB', 'BA')
    after_u = [('U→P', scope, order, 'after_u') for order in orders]
    candidate = [('U→P', scope, order, 'end') for order in orders]
    if control == 'shared_update':
        candidate, controls = after_u, [('ANCHOR', None, None, 'anchor')]
    elif control == 'post_u_recovery':
        controls = after_u
    elif control in ('U_ONLY', 'J_ACC'):
        controls = [(control, None, 'ACC', 'end')]
    else:
        controls = [(control, scope if control == 'P→U' else None, order, 'end') for order in orders]
    left = _mean(rows[(*key, readout, panel)][metric] for key in candidate)
    right = _mean(rows[(*key, readout, panel)][metric] for key in controls)
    return _delta(left, right)


def _guard(stats):
    means = [stats[cell]['mean_delta'] for cell in AUC_METRICS[1:]]
    return all(mean is not None and mean >= GUARD for mean in means)


def _clock_disparity(scope, control):
    if control != 'J_ACC' or scope == 'P_FULL':
        return None
    return 'P_AXIS' if scope == 'P_CELL' else 'P_CELL'


def scope_decision(contrasts):
    """Use online V_A only to freeze a scope, then test that scope on online V_B."""
    def get(dataset, epoch, panel, scope, control):
        return contrasts[dataset, epoch, panel, 'online', scope, control]

    eligible = {}
    va = {}
    for scope in SCOPES:
        qualifying = []
        for dataset in DATASETS:
            estimates = [get(dataset, AGGREGATE, 'V_A', scope, control) for control in SELECTION]
            if all(d['macro_auc']['mean_delta'] is not None and d['macro_auc']['mean_delta'] > 0 and _guard(d)
                   for d in estimates):
                qualifying.append(dataset)
        equal = {control: get(EQUAL_DATASET, AGGREGATE, 'V_A', scope, control)['macro_auc']['mean_delta']
                 for control in SELECTION}
        score = min(equal.values()) if all(value is not None for value in equal.values()) else None
        va[scope] = dict(qualified_datasets=qualifying, equal_dataset_macro_auc=equal, rank_score=score)
        if qualifying and score is not None:
            eligible[scope] = score
    scope = max(eligible, key=eligible.get) if eligible else None
    decision = dict(status='COMPLETE_NO_GO_S1', s1_gate='NO_GO', frozen_scope=scope, selection_panel='V_A',
                    confirmation_panel='V_B', primary_readout='online', seed_count=len(SEEDS),
                    cell_guard=GUARD, v_a=va, v_b=None, confirmed_datasets=[], s2_clock_blocker=None,
                    reason='NO_ELIGIBLE_V_A_SCOPE')
    if scope is None:
        return decision

    vb, confirmed = {}, []
    for dataset in DATASETS:
        estimates = {control: get(dataset, AGGREGATE, 'V_B', scope, control) for control in CONFIRMATION}
        means = {control: d['macro_auc']['mean_delta'] for control, d in estimates.items()}
        upper = {control: d['macro_auc']['ci95_high'] for control, d in estimates.items()}
        significant_negative = [control for control, value in upper.items() if value is not None and value < 0]
        anchor_means = {f'E{epoch}': {
            control: get(dataset, epoch, 'V_B', scope, control)['macro_auc']['mean_delta']
            for control in CONFIRMATION} for epoch in ANCHORS}
        replicated = [epoch for epoch, ds in anchor_means.items()
                      if all(value is not None and value > 0 for value in ds.values())]
        guard = all(_guard(d) for d in estimates.values())
        passed = (dataset in va[scope]['qualified_datasets'] and guard and len(replicated) >= 2
                  and all(value is not None and value > 0 for value in means.values()))
        if passed:
            confirmed.append(dataset)
        vb[dataset] = dict(mean_macro_auc=means, ci95_upper=upper, significantly_negative=significant_negative,
                           unavailable_intervals=[control for control, value in upper.items() if value is None],
                           anchor_mean_macro_auc=anchor_means, replicated_anchors=replicated,
                           cell_guard_pass=guard, passed=passed)
    vb[EQUAL_DATASET] = {control: get(EQUAL_DATASET, AGGREGATE, 'V_B', scope, control)['macro_auc']['mean_delta']
                         for control in CONFIRMATION}
    decision.update(v_b=vb, confirmed_datasets=confirmed, reason='INDEPENDENT_V_B_DID_NOT_CONFIRM')
    if not confirmed:
        return decision
    if any(vb[d]['significantly_negative'] for d in DATASETS if d not in confirmed):
        decision['reason'] = 'SIGNIFICANTLY_NEGATIVE_ON_OTHER_DATASET'
        return decision
    clock_blocker = _clock_disparity(scope, 'J_ACC')
    decision.update(s1_gate='GO', status='TECHNICAL_BLOCKED' if clock_blocker else 'GO',
                    reason='S2_PARAMETER_CLOCK_MISMATCH' if clock_blocker else 'V_B_CONFIRMED',
                    s2_clock_blocker=clock_blocker, dataset_conditional=len(confirmed) == 1)
    return decision


def _comparisons(index, jobs):
    metric_columns = tuple(key for key in next(iter(next(iter(index.values())).values()))
                           if key.endswith(('_auc', '_logloss')) and '_fallback_' not in key)
    summaries, seed_rows = {}, []
    for panel in PANELS:
        for readout in READOUTS:
            for scope in SCOPES:
                for control in ((*CONTROLS, *MECHANISM) if readout == 'online' else CONTROLS):
                    differences = {}
                    for dataset in DATASETS:
                        for seed in SEEDS:
                            for metric in metric_columns:
                                for epoch in ANCHORS:
                                    differences[dataset, seed, metric, epoch] = _paired(
                                        index, dataset, seed, epoch, panel, readout, scope, control, metric)
                                differences[dataset, seed, metric, AGGREGATE] = _mean(
                                    differences[dataset, seed, metric, epoch] for epoch in ANCHORS)
                    for dataset in (*DATASETS, EQUAL_DATASET):
                        for epoch in (*ANCHORS, AGGREGATE):
                            estimates = {}
                            for metric in metric_columns:
                                deltas = {seed: _mean(differences[d, seed, metric, epoch]
                                                     for d in (DATASETS if dataset == EQUAL_DATASET else (dataset,)))
                                          for seed in SEEDS}
                                result = _summary(deltas)
                                estimates[metric] = result
                                for seed, delta in deltas.items():
                                    provenance = [jobs[d, seed] for d in
                                                  (DATASETS if dataset == EQUAL_DATASET else (dataset,))]
                                    seed_rows.append(dict(dataset=dataset, arm='U→P', scope=scope, control=control,
                                                          seed=seed, readout=readout, epoch=epoch, panel=panel,
                                                          metric=metric, delta=delta,
                                                          excluded_private_group_clock=_clock_disparity(scope, control),
                                                          leave_seed_out_mean=_mean(d for s, d in deltas.items() if s != seed),
                                                          **{key: '|'.join(dict.fromkeys(j[key] for j in provenance))
                                                             for key in ('source_hash', 'data_hash', 'config_hash')},
                                                          **{key: result[key] for key in ('n', 'mean_delta', 'sd_delta',
                                                                                         'ci95_low', 'ci95_high',
                                                                                         'positive', 'negative', 'zero')}))
                            summaries[dataset, epoch, panel, readout, scope, control] = estimates
    return summaries, seed_rows


def analyze(artifact_dir):
    """Validate completed S1 endpoint receipts and reproduce all registered scalar contrasts."""
    artifact_dir = Path(artifact_dir)
    index, jobs, tables = _index_endpoints(artifact_dir)
    summaries, seed_rows = _comparisons(index, jobs)
    decision = scope_decision(summaries)
    for filename, rows in (('FORK_ENDPOINTS.csv', tables['endpoints']), ('PAIR_TRANSITIONS.csv', tables['transitions']),
                           ('CELL_RESPONSE.csv', tables['responses']), ('PAIRED_SEEDS.csv', seed_rows)):
        write_csv(artifact_dir / filename, rows, list(IDENTITY) + ['arm', 'readout'])
    write(artifact_dir / 'SCOPE_DECISION.json', decision)
    return decision
