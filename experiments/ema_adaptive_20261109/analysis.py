"""Predeclared S-only, training-panel-only forward prediction audit."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from experiments.ema_mechanism_20261106.analysis import feature_matrices, ridge_predict, seed_statistics, write_table
from .audit import Conflict, Incomplete, SEEDS, read_json, write_json

CELL_KEYS = ['dataset', 'objective', 'seed', 'epoch', 'panel_id', 'domain', 'task']
SIGNALS = ['g_fast', 'g_slow', 'D', 'l_fast', 'l_slow']
FIT_ORIGINS = [5, 10, 15]
SCORE_ORIGINS = [25, 30, 35, 40, 45]
ACTIONS = ('F', 'fast', 'slow')


def unique_rows(frame, keys):
    """Only identical duplicate aliases can collapse to one observation."""
    dedup = frame.drop_duplicates()
    if dedup.duplicated(keys).any():
        raise Incomplete('Conflicting duplicate keys: ' + ','.join(keys))
    return dedup


def contrasts(frame):
    keys = ['job_id', 'epoch', 'panel_id', 'physical_readout_id', 'domain', 'task']
    frame = unique_rows(frame, keys)
    frames = []
    for variant in ACTIONS:
        match = ((frame.variant == 'F') & frame.group_id.isna()) if variant == 'F' else (
            (frame.variant == variant) & (frame.group_id == 'S'))
        part = frame.loc[match].copy()
        cols = CELL_KEYS + ['auc', 'logloss', 'state_hash', 'actual_count', 'missing_reason',
                           'job_id', 'init_hash', 'order_hash', 'layout_hash', 'group_hash', 'identity_id']
        part = unique_rows(part[cols], CELL_KEYS)
        frames.append(part.rename(columns={k: f'{variant}_{k}' for k in cols if k not in CELL_KEYS}))
    paired = frames[0]
    for part in frames[1:]:
        paired = paired.merge(part, on=CELL_KEYS, validate='one_to_one', how='outer')
    for variant in ACTIONS:
        if paired[f'{variant}_auc'].isna().any() or paired[f'{variant}_logloss'].isna().any():
            raise Incomplete('Missing F/fast_S/slow_S AUC/LL observation')
    for identity in ('job_id', 'init_hash', 'order_hash', 'layout_hash', 'group_hash', 'identity_id', 'actual_count'):
        if not (paired[f'F_{identity}'].eq(paired[f'fast_{identity}']) &
                paired[f'F_{identity}'].eq(paired[f'slow_{identity}'])).all():
            raise Incomplete('Pulse trajectory identity differs: ' + identity)
    paired['g_fast'] = paired.fast_auc - paired.F_auc
    paired['g_slow'] = paired.slow_auc - paired.F_auc
    paired['D'] = paired.g_fast - paired.g_slow
    paired['l_fast'] = paired.F_logloss - paired.fast_logloss
    paired['l_slow'] = paired.F_logloss - paired.slow_logloss
    paired['group_id'] = 'S'
    return paired


def validate_coverage(frame):
    expected = pd.MultiIndex.from_product([['movielens', 'kuairand'], ['clean', 'gaussian_d0'], SEEDS,
        range(5, 51, 5), ['P_A', 'P_B', 'V_mech', 'V_A', 'V_B'], range(3), range(2)], names=CELL_KEYS)
    actual = pd.MultiIndex.from_frame(frame[CELL_KEYS])
    if len(frame) != len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise Incomplete('S-only full twenty-trajectory, five-panel, six-cell matrix incomplete')


def scopes(frame, values, time='epoch', extra=()):
    keys = ['dataset', 'objective', 'seed', time, *extra]
    rows = frame.copy()
    rows['scope'] = 'd' + rows.domain.astype(str) + 't' + rows.task.astype(str)
    parts = [rows[keys + ['scope'] + values]]
    for task in (0, 1):
        part = rows[rows.task == task].groupby(keys, dropna=False)[values].mean().reset_index()
        part['scope'] = f'TASK{task}'
        parts.append(part)
    part = rows.groupby(keys, dropna=False)[values].mean().reset_index()
    part['scope'] = 'GLOBAL'
    parts.append(part)
    return pd.concat(parts, ignore_index=True)


def correlation(x, y, kind):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float({'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}[kind](x, y).statistic)


def sequence_stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    nonzero = valid & (x != 0) & (y != 0)
    return {'sign_agreement': float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else np.nan,
            'sign_agreement_including_zero': float(np.mean(np.sign(x[valid]) == np.sign(y[valid]))) if valid.any() else np.nan,
            'zero_pair_fraction': float(np.mean(~nonzero[valid])) if valid.any() else np.nan,
            'valid_pairs': int(valid.sum()), 'nonzero_pairs': int(nonzero.sum()),
            **{kind: correlation(x, y, kind) for kind in ('pearson', 'spearman', 'kendall')}}


def persistence(frame):
    agg = scopes(frame, SIGNALS, extra=['panel_id'])
    keys = ['dataset', 'objective', 'seed', 'scope']
    directions = [('P_A', 'P_A'), ('P_B', 'P_B'), ('V_A', 'V_A'), ('V_B', 'V_B'), ('V_mech', 'V_mech'),
                  ('P_A', 'P_B'), ('P_A', 'V_A'), ('P_A', 'V_B'), ('P_B', 'P_A')]
    records, stats = [], []
    for first, second in directions:
        a = agg[agg.panel_id == first].drop(columns='panel_id').rename(columns={'epoch': 'origin_epoch'})
        b = agg[agg.panel_id == second].drop(columns='panel_id').rename(columns={'epoch': 'target_epoch'})
        for horizon in (0, 5):
            if horizon == 0 and first == second:
                continue
            a = a.copy()
            a['target_epoch'] = a.origin_epoch + horizon
            pairs = a.merge(b, on=keys + ['target_epoch'], suffixes=('_current', '_future'), validate='one_to_one')
            pairs['feature_panel'], pairs['score_panel'], pairs['horizon_epochs'] = first, second, horizon
            records.append(pairs)
            for phase, origins in [('ALL', list(range(5, 46, 5))), ('EARLY', FIT_ORIGINS), ('LATE', SCORE_ORIGINS)]:
                for key, group in pairs[pairs.origin_epoch.isin(origins)].groupby(keys):
                    for signal in SIGNALS:
                        result = sequence_stats(group[signal + '_current'], group[signal + '_future'])
                        stats.append(dict(zip(keys, key), feature_panel=first, score_panel=second, horizon_epochs=horizon,
                                          phase=phase, signal=signal, **result))
    return pd.concat(records, ignore_index=True), pd.DataFrame(stats)


def predictive_frame(contrasts_frame, metrics, panel):
    a = contrasts_frame[contrasts_frame.panel_id == panel].copy()
    a = a.rename(columns={'epoch': 'origin_epoch', 'g_fast': 'current_fast', 'g_slow': 'current_slow'})
    a['target_epoch'] = a.origin_epoch + 5
    y = contrasts_frame[contrasts_frame.panel_id == panel][CELL_KEYS + ['g_fast', 'g_slow']].rename(
        columns={'epoch': 'target_epoch', 'g_fast': 'y_fastF', 'g_slow': 'y_slowF'})
    keys = ['dataset', 'objective', 'seed', 'panel_id', 'domain', 'task', 'target_epoch']
    result = a.merge(y, on=keys, validate='one_to_one')
    if panel == 'P_A':
        metric = unique_rows(metrics[(metrics.group_id == 'S') & (metrics.panel_id == 'P_A')],
                             ['job_id', 'epoch', 'domain', 'task', 'group_id', 'panel_id'])
        metric = metric.rename(columns={'epoch': 'origin_epoch'})
        join = ['dataset', 'objective', 'seed', 'origin_epoch', 'domain', 'task', 'panel_id', 'group_id']
        needed = [c for c in metric.columns if c not in result.columns] + join
        before = len(result)
        result = result.merge(metric[needed], on=join, validate='one_to_one')
        if len(result) != before:
            raise Incomplete('S/P_A feature rows are missing')
        for identity in ('job_id', 'init_hash', 'order_hash', 'layout_hash', 'group_hash', 'identity_id'):
            if not result[identity].eq(result['F_' + identity]).all():
                raise Incomplete('Feature/readout trajectory identity differs: ' + identity)
    return result


def split(frame, dataset, heldout):
    subset = frame[frame.dataset == dataset]
    train = subset[(subset.seed != heldout) & subset.origin_epoch.isin(FIT_ORIGINS)].copy()
    score = subset[(subset.seed == heldout) & subset.origin_epoch.isin(SCORE_ORIGINS)].copy()
    if len(train) != 144 or len(score) != 60:
        raise Incomplete(f'S-only forward fold coverage: {len(train)}/{len(score)}')
    if heldout in set(train.seed) or train.target_epoch.max() >= score.origin_epoch.min():
        raise ValueError('Seed/time leakage')
    return train, score


def fit_predictions(train, score, spec, models):
    results, transforms = {}, {}
    for model in models:
        if model == 'C_carry':
            results[model] = score[['current_fast', 'current_slow']].to_numpy()
        else:
            x, z, transform = feature_matrices(train, score, model, spec)
            results[model] = ridge_predict(x, train[['y_fastF', 'y_slowF']].to_numpy(), z)
            transforms[model] = transform
    return results, transforms


def predict(frame, metrics, source_spec):
    spec = copy.deepcopy(source_spec)
    models = ['C_design', 'C_simple', 'C_persist', 'C_carry', 'M0', 'M2_full', 'M2_detach', 'M3_direct']
    spec['feature_sets']['C_persist'] = ['design', 'persistence']
    spec['feature_blocks']['persistence'] = ['current_fast', 'current_slow']
    oof, folds = [], []
    for panel in ('P_A', 'P_B'):
        data = predictive_frame(frame, metrics, panel)
        chosen_models = models if panel == 'P_A' else ['C_design', 'C_persist', 'C_carry']
        for dataset in ('movielens', 'kuairand'):
            for heldout in SEEDS:
                train, score = split(data, dataset, heldout)
                predictions, transforms = fit_predictions(train, score, spec, chosen_models)
                folds.append({'dataset': dataset, 'heldout_seed': heldout, 'feature_panel': panel,
                              'fit_seeds': sorted(train.seed.unique().tolist()), 'train_origins': FIT_ORIGINS,
                              'score_origins': SCORE_ORIGINS, 'train_rows': len(train), 'score_rows': len(score),
                              'fit_target_panel': panel, 'score_only_panels': ['P_B', 'V_A', 'V_B'] if panel == 'P_A' else ['P_A', 'V_A', 'V_B'],
                              'transforms': transforms})
                basecols = ['dataset', 'objective', 'seed', 'origin_epoch', 'target_epoch', 'domain', 'task', 'group_id',
                            'F_job_id', 'F_init_hash', 'F_order_hash', 'F_layout_hash', 'F_group_hash', 'F_identity_id']
                for model, prediction in predictions.items():
                    part = score[basecols].copy()
                    part['feature_panel'], part['model_id'] = panel, model
                    part['yhat_fast'], part['yhat_slow'] = prediction[:, 0], prediction[:, 1]
                    oof.append(part)
    return pd.concat(oof, ignore_index=True), folds


def static_policies(frame):
    rows = []
    for panel in ('P_A', 'P_B'):
        for dataset in ('movielens', 'kuairand'):
            for objective in ('clean', 'gaussian_d0'):
                for seed in SEEDS:
                    fit = frame[(frame.panel_id == panel) & (frame.dataset == dataset) & (frame.objective == objective) &
                                (frame.seed != seed) & frame.epoch.isin(FIT_ORIGINS)]
                    for scope in ('GLOBAL', 'TASK0', 'TASK1'):
                        values = fit if scope == 'GLOBAL' else fit[fit.task == int(scope[-1])]
                        scores = [0., values.g_fast.mean(), values.g_slow.mean()]
                        rows.append({'dataset': dataset, 'objective': objective, 'seed': seed, 'feature_panel': panel,
                                     'scope': scope, 'static_action': int(np.argmax(scores)),
                                     'fit_mean_fast': scores[1], 'fit_mean_slow': scores[2]})
    return pd.DataFrame(rows)


def evaluate_predictions(oof, frame, policies):
    target = frame.rename(columns={'epoch': 'target_epoch', 'panel_id': 'score_panel'})
    keys = ['dataset', 'objective', 'seed', 'target_epoch', 'domain', 'task']
    values = ['g_fast', 'g_slow', 'l_fast', 'l_slow', 'F_auc', 'fast_auc', 'slow_auc',
              'F_logloss', 'fast_logloss', 'slow_logloss', 'F_state_hash', 'fast_state_hash', 'slow_state_hash',
              'F_actual_count', 'fast_actual_count', 'slow_actual_count']
    merged = oof.merge(target[keys + ['score_panel'] + values], on=keys, validate='many_to_many')
    merged = merged[((merged.feature_panel == 'P_A') & merged.score_panel.isin(['P_A', 'P_B', 'V_A', 'V_B'])) |
                    ((merged.feature_panel == 'P_B') & merged.score_panel.isin(['P_A', 'P_B', 'V_A', 'V_B']))].copy()
    pred_values = ['yhat_fast', 'yhat_slow', 'g_fast', 'g_slow', 'l_fast', 'l_slow']
    extra = ['feature_panel', 'score_panel', 'model_id']
    agg = scopes(merged, pred_values, time='origin_epoch', extra=extra)
    agg['target_epoch'] = agg.origin_epoch + 5
    policy = policies[policies.scope == 'GLOBAL'].drop(columns=['scope', 'fit_mean_fast', 'fit_mean_slow']).rename(columns={'static_action': 'uniform_action'})
    agg = agg.merge(policy, on=['dataset', 'objective', 'seed', 'feature_panel'], validate='many_to_one')
    scores = np.column_stack([np.zeros(len(agg)), agg.yhat_fast, agg.yhat_slow])
    truth = np.column_stack([np.zeros(len(agg)), agg.g_fast, agg.g_slow])
    loss = np.column_stack([np.zeros(len(agg)), agg.l_fast, agg.l_slow])
    action = np.argmax(scores, axis=1)
    idx = np.arange(len(agg))
    agg['action'], agg['chosen_minus_F'] = action, truth[idx, action]
    agg['chosen_LL_utility'] = loss[idx, action]
    agg['uniform_minus_F'] = truth[idx, agg.uniform_action.astype(int)]
    agg['uniform_LL_utility'] = loss[idx, agg.uniform_action.astype(int)]
    agg['chosen_minus_uniform'] = agg.chosen_minus_F - agg.uniform_minus_F
    agg['regret'] = truth.max(axis=1) - agg.chosen_minus_F
    agg['D_true'], agg['D_pred'] = agg.g_fast - agg.g_slow, agg.yhat_fast - agg.yhat_slow
    for name, observed, predicted in [('D', agg.D_true, agg.D_pred), ('fast', agg.g_fast, agg.yhat_fast), ('slow', agg.g_slow, agg.yhat_slow)]:
        agg['abs_error_' + name] = np.abs(observed - predicted)
        agg['sq_error_' + name] = (observed - predicted) ** 2
    valid = agg.D_true != 0
    agg['direction_correct'] = np.where(valid, (np.sign(agg.D_true) == np.sign(agg.D_pred)).astype(float), np.nan)
    for reference in ('C_design', 'C_simple'):
        key = ['dataset', 'objective', 'seed', 'origin_epoch', 'feature_panel', 'score_panel', 'scope']
        ref = agg[agg.model_id == reference][key + ['chosen_minus_F']].rename(columns={'chosen_minus_F': 'reference_value'})
        agg = agg.merge(ref, on=key, how='left', validate='many_to_one')
        agg['chosen_minus_' + reference] = agg.chosen_minus_F - agg.reference_value
        agg = agg.drop(columns='reference_value')
    return merged, agg


def prediction_seed(agg):
    keys = ['dataset', 'objective', 'seed', 'feature_panel', 'score_panel', 'scope', 'model_id']
    rows = []
    for key, group in agg.groupby(keys, sort=True):
        group = group.sort_values('origin_epoch')
        action = group.action.to_numpy()
        runs = np.split(action, np.flatnonzero(action[1:] != action[:-1]) + 1)
        value = dict(zip(keys, key))
        for metric in ('chosen_minus_F', 'chosen_LL_utility', 'uniform_minus_F', 'chosen_minus_uniform',
                       'chosen_minus_C_design', 'chosen_minus_C_simple', 'regret', 'direction_correct'):
            value[metric] = group[metric].mean()
        for target in ('D', 'fast', 'slow'):
            value['MAE_' + target] = group['abs_error_' + target].mean()
            value['RMSE_' + target] = np.sqrt(group['sq_error_' + target].mean())
        value.update({kind: correlation(group.D_pred, group.D_true, kind) for kind in ('pearson', 'spearman', 'kendall')})
        value.update(n_windows=len(group), action_missing_rate=float(group.action.isna().mean()),
                     action_switches=int(np.count_nonzero(action[1:] != action[:-1])),
                     mean_action_run_windows=float(np.mean([len(run) for run in runs])),
                     mean_observed_run_span_epochs=float(np.mean([(len(run)-1)*5 for run in runs])), feedback_delay_epochs=5)
        rows.append(value)
    return pd.DataFrame(rows)


def static_scores(frame, policies):
    target = frame[frame.epoch.isin([30, 35, 40, 45, 50]) & frame.panel_id.isin(['P_A', 'P_B', 'V_A', 'V_B'])].copy()
    target = target.rename(columns={'panel_id': 'score_panel'})
    parts = []
    for panel in ('P_A', 'P_B'):
        policy = policies[policies.feature_panel == panel]
        task = policy[policy.scope != 'GLOBAL'].copy()
        task['task'] = task.scope.str[-1].astype(int)
        task = task.rename(columns={'static_action': 'task_action'})
        uniform = policy[policy.scope == 'GLOBAL'].rename(columns={'static_action': 'uniform_action'})
        keys = ['dataset', 'objective', 'seed']
        part = target.merge(task[keys + ['task', 'task_action']], on=keys + ['task'], validate='many_to_one')
        part = part.merge(uniform[keys + ['uniform_action']], on=keys, validate='many_to_one')
        part['feature_panel'] = panel
        idx = np.arange(len(part))
        truth = np.column_stack([np.zeros(len(part)), part.g_fast, part.g_slow])
        loss = np.column_stack([np.zeros(len(part)), part.l_fast, part.l_slow])
        part['task_minus_F'] = truth[idx, part.task_action.astype(int)]
        part['task_LL_utility'] = loss[idx, part.task_action.astype(int)]
        part['uniform_minus_F'] = truth[idx, part.uniform_action.astype(int)]
        part['uniform_LL_utility'] = loss[idx, part.uniform_action.astype(int)]
        part['task_minus_uniform'] = part.task_minus_F - part.uniform_minus_F
        part['task_minus_uniform_LL'] = part.task_LL_utility - part.uniform_LL_utility
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    metrics = ['task_minus_F', 'task_LL_utility', 'uniform_minus_F', 'uniform_LL_utility', 'task_minus_uniform',
               'task_minus_uniform_LL', 'g_fast', 'g_slow', 'l_fast', 'l_slow']
    agg = scopes(raw, metrics, extra=['feature_panel', 'score_panel'])
    keys = ['dataset', 'objective', 'seed', 'feature_panel', 'score_panel', 'scope']
    return raw, agg.groupby(keys, as_index=False)[metrics].mean()


def stable(values):
    values = np.asarray(values, float)
    good = len(values) == 5 and np.isfinite(values).all()
    leave_one = [(values.sum() - v) / 4 for v in values] if good else []
    return {'pass': bool(good and values.mean() > 0 and min(leave_one) >= 0),
            'seed_values': dict(zip(map(str, SEEDS), values.tolist())), **seed_statistics(values),
            'leave_one_seed_out_means': leave_one}


def gate(prediction, static, source_complete=True):
    if not source_complete:
        return {'state': 'SOURCE_INCOMPLETE', 'dynamic': [], 'static': []}
    dynamic_checks, static_checks = [], []
    pred = prediction[(prediction.dataset == 'kuairand') & (prediction.objective == 'gaussian_d0')]
    stat = static[(static.dataset == 'kuairand') & (static.objective == 'gaussian_d0')]

    def series(data, panel, score, scope, metric, model=None):
        sub = data[(data.feature_panel == panel) & data.score_panel.isin(['V_A', 'V_B'] if score == 'V_AVG' else [score]) & (data.scope == scope)]
        if model is not None:
            sub = sub[sub.model_id == model]
        return sub.groupby('seed')[metric].mean().reindex(SEEDS).to_numpy()

    def check(label, vals, require_stable=True):
        result = stable(vals)
        if not require_stable:
            result['pass'] = bool(len(vals) == 5 and np.isfinite(vals).all() and np.mean(vals) > 0)
        return {'check': label, **result}

    for scope in ('GLOBAL', 'TASK0', 'TASK1'):
        checks = []
        for score in ('P_B', 'V_AVG'):
            for reference in ('C_design', 'C_simple'):
                vals = series(pred, 'P_A', score, scope, 'MAE_D', reference) - series(pred, 'P_A', score, scope, 'MAE_D', 'C_persist')
                checks.append(check(f'{score}:MAE_{reference}-persist', vals))
            for metric in ('chosen_minus_F', 'chosen_minus_uniform', 'chosen_minus_C_design', 'chosen_minus_C_simple'):
                checks.append(check(f'{score}:{metric}', series(pred, 'P_A', score, scope, metric, 'C_persist')))
            # Every forecast uses the same nonzero-target denominator; predicted ties are incorrect.
            vals = series(pred, 'P_A', score, scope, 'direction_correct', 'C_carry') - series(pred, 'P_A', score, scope, 'direction_correct', 'C_design')
            checks.append(check(f'{score}:raw_direction_minus_identity', vals))
        checks.append(check('P_B_to_P_A:direction', series(pred, 'P_B', 'P_A', scope, 'direction_correct', 'C_carry'), False))
        checks.append(check('P_B_to_P_A:action_minus_F', series(pred, 'P_B', 'P_A', scope, 'chosen_minus_F', 'C_persist'), False))
        dynamic_checks.append({'scope': scope, 'pass': all(c['pass'] for c in checks), 'checks': checks})
    for scope in ('GLOBAL', 'TASK0', 'TASK1'):
        forward_checks, reverse_checks = [], []
        for score in ('P_B', 'V_AVG'):
            for metric in ('task_minus_F', 'task_minus_uniform'):
                forward_checks.append(check(f'P_A_to_{score}:{metric}', series(stat, 'P_A', score, scope, metric)))
        for score in ('P_A', 'V_AVG'):
            for metric in ('task_minus_F', 'task_minus_uniform'):
                reverse_checks.append(check(f'P_B_to_{score}:{metric}', series(stat, 'P_B', score, scope, metric), False))
        required = forward_checks + (reverse_checks if scope == 'GLOBAL' else [])
        static_checks.append({'scope': scope, 'pass': all(c['pass'] for c in required),
                              'reverse_role': 'required' if scope == 'GLOBAL' else 'diagnostic_only',
                              'checks': forward_checks + reverse_checks})
    static_pass = static_checks[0]['pass'] and any(c['pass'] for c in static_checks[1:])
    state = 'PERSISTENCE_SUPPORTED' if any(c['pass'] for c in dynamic_checks) else (
        'STATIC_PREFERENCE_ONLY' if static_pass else 'NO_ACTIONABLE_SIGNAL')
    return {'state': state, 'dynamic': dynamic_checks, 'static': static_checks,
            'new_training_triggered': state in ('PERSISTENCE_SUPPORTED', 'STATIC_PREFERENCE_ONLY'),
            'requires_reviewed_A1_seed_amendment': state in ('PERSISTENCE_SUPPORTED', 'STATIC_PREFERENCE_ONLY'),
            'evidence_files': ['A0_PREDICTION_SEED.csv', 'A0_STATIC_SEED.csv'],
            'scientific_unit': 'seed', 'seeds': list(SEEDS), 'pulse_duration_epochs': 1, 'observation_spacing_epochs': 5}


def summary(frame, keys, metrics):
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            values = group.groupby('seed')[metric].mean().reindex(SEEDS)
            rows.append(dict(zip(keys, key), metric=metric, seed_values=values.to_dict(), **seed_statistics(values)))
    return pd.DataFrame(rows)


def historical_comparison(source, out):
    old = pd.read_csv(source / 'PREDICTIVE_OOF.csv', float_precision='round_trip')
    old = old[old.group_id == 'S'].copy()
    keys = ['dataset', 'objective', 'seed', 'origin_epoch', 'target_epoch', 'cell', 'group_id', 'model_id', 'label_panel', 'row_fold_direction']
    old = unique_rows(old, keys)
    old['comparison_protocol'] = 'HISTORICAL_VALIDATION_FIT'
    write_table(out / 'A0_HISTORICAL_VALIDATION_FIT.csv', old)
    decisions = pd.read_csv(source / 'ACTION_DECISIONS.csv', float_precision='round_trip')
    decisions = unique_rows(decisions, ['dataset', 'scope', 'objective', 'cell', 'group_id', 'model_id'])
    decisions = decisions[decisions.group_id == 'S'].copy()
    decisions['comparison_protocol'] = 'HISTORICAL_VALIDATION_FIT_NESTED_SCOPES'
    write_table(out / 'A0_HISTORICAL_ACTION_SCOPES.csv', decisions)


def analyze(study, out):
    audit = read_json(out / 'AUDIT_INVENTORY.json')
    if audit['status'] != 'VERIFIED':
        raise Incomplete('Source audit has not passed')
    previous = read_json(out / 'RESOLVED_PROTOCOL.json')
    if any(previous[k] != study[k] for k in ('manifest_sha256', 'design_sha256', 'implementation')):
        raise Conflict('Manifest/design/implementation changed after audit')
    source = Path(study['source'])
    paths = audit['archive']['recovered_tables']
    raw = pd.read_csv(paths.get('PULSE_READOUTS.csv', source / 'PULSE_READOUTS.csv'), float_precision='round_trip')
    metrics = pd.read_csv(paths.get('PULSE_METRICS.csv', source / 'PULSE_METRICS.csv'), float_precision='round_trip')
    frame = contrasts(raw)
    validate_coverage(frame)
    write_table(out / 'A0_S_CONTRASTS.csv', frame)
    rows, seed = persistence(frame)
    write_table(out / 'A0_PERSISTENCE_ROWS.csv', rows)
    write_table(out / 'A0_SEED.csv', seed)
    pkeys = ['dataset', 'objective', 'feature_panel', 'score_panel', 'scope', 'horizon_epochs', 'phase', 'signal']
    write_table(out / 'A0_SUMMARY.csv', summary(seed, pkeys, ['sign_agreement', 'sign_agreement_including_zero',
        'zero_pair_fraction', 'pearson', 'spearman', 'kendall', 'valid_pairs', 'nonzero_pairs']))
    source_spec = read_json(source / 'RESOLVED_PROTOCOL.json')['manifest']['predictive']
    oof, folds = predict(frame, metrics, source_spec)
    policies = static_policies(frame)
    merged, actions = evaluate_predictions(oof, frame, policies)
    predicted_seed = prediction_seed(actions)
    write_table(out / 'A0_PREDICTIVE_OOF.csv', merged)
    write_json(out / 'A0_PREDICTIVE_FOLDS.json', {'folds': folds, 'fit_labels': 'training feature panel only',
               'fixed_ridge_alpha': 1, 'hyperparameter_search': False})
    write_table(out / 'A0_ACTIONS.csv', actions)
    write_table(out / 'A0_PREDICTION_SEED.csv', predicted_seed)
    qkeys = ['dataset', 'objective', 'feature_panel', 'score_panel', 'scope', 'model_id']
    write_table(out / 'A0_PREDICTION_SUMMARY.csv', summary(predicted_seed, qkeys,
        [c for c in predicted_seed.columns if c not in qkeys + ['seed']]))
    raw_static, static_seed = static_scores(frame, policies)
    write_table(out / 'A0_STATIC_POLICIES.csv', policies)
    write_table(out / 'A0_STATIC_ACTIONS.csv', raw_static)
    write_table(out / 'A0_STATIC_SEED.csv', static_seed)
    skeys = ['dataset', 'objective', 'feature_panel', 'score_panel', 'scope']
    write_table(out / 'A0_STATIC_SUMMARY.csv', summary(static_seed, skeys,
        [c for c in static_seed.columns if c not in skeys + ['seed']]))
    historical_comparison(source, out)
    decision = gate(predicted_seed, static_seed)
    write_json(out / 'A0_DECISION.json', decision)
    return decision
