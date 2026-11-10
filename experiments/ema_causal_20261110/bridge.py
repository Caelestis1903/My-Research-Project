"""Frozen action-level LOSO bridge; inputs contain real pulse readouts only."""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

SEEDS = tuple(range(2022, 2027))
ORIGINS = (5, 10, 15)
EVALUATION_ORIGINS = (25, 30, 35, 40, 45)
MENUS = {'rho7': ('fast', 'slow'), 'mechanism7': ('drop_recent', 'drop_old')}
PREDICTORS = ('C_design_adapted', 'C_simple_adapted', 'C_carry_direct')
EXTRA_NAMES = ('log1p_r_rms', 'log1p_delta_rms', 'cosine', 'is_zero_R',
               'is_zero_delta', 'log1p_mean_cell_rms_zdiff',
               'log1p_group_rank_gradient_rms', 'current_macro_auc_utility')


def action_order(menu):
    return ('F',) + tuple(f'{scope}/{variant}' for scope in ('all', 'S', 'N')
                         for variant in MENUS[menu])


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _number(value):
    return value is not None and bool(np.isfinite(value))


def _mean(values):
    return float(np.mean(values)) if values and all(_number(v) for v in values) else None


def _cells(record, population, metric):
    rows = record.get('metrics', {}).get(population, []) if record else []
    return [row.get(metric) for row in rows] if len(rows) == 6 else [None] * 6


def _difference(record, reference, population, metric):
    a, b = _cells(record, population, metric), _cells(reference, population, metric)
    return [float(x - y) if _number(x) and _number(y) else None for x, y in zip(a, b)]


def _utility(record, reference, population):
    return _mean(_difference(record, reference, population, 'auc'))


def choose(values, order):
    """Strict maximum, F first; an incomplete action menu is unestimable."""
    if any(not _number(values.get(action)) for action in order):
        return None
    return max(order, key=lambda action: values[action])


def _features(record, reference, menu):
    action = record['action_id']; index = action_order(menu)[1:].index(action)
    base = [float(i == index) for i in range(6)] + [record['epoch'] / 50.]
    g = record.get('geometry', {})
    names = ('r_rms', 'delta_rms', 'cosine', 'is_zero_R', 'is_zero_delta',
             'mean_cell_rms_zdiff', 'group_rank_gradient_rms')
    raw = [g.get(name) for name in names]
    for i in (0, 1, 5, 6):
        raw[i] = float(np.log1p(raw[i])) if _number(raw[i]) and raw[i] >= 0 else None
    raw += [_utility(record, reference, 'P_A')]
    raw = [float(v) if _number(v) else None for v in raw]
    return base, base + raw


def _ridge(x, y, continuous):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mean = np.zeros(x.shape[1]); scale = np.ones(x.shape[1])
    mean[continuous] = x[:, continuous].mean(axis=0)
    std = x[:, continuous].std(axis=0)
    scale[continuous] = np.where(std > 0, std, 1.)
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0.
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                constant_columns=[int(i) for i, s in zip(continuous, std) if s == 0])


def _predict(model, x):
    if model is None or any(not _number(v) for v in x):
        return None
    z = (np.asarray(x) - model['mean']) / model['scale']
    return float(model['beta'][0] + z @ np.asarray(model['beta'][1:]))


def seed_summary(values):
    """Only the five independent seeds form a complete inferential unit."""
    values = {int(k): v for k, v in values.items()}
    complete = set(values) == set(SEEDS) and all(_number(v) for v in values.values())
    result = dict(seed_values=values, status='COMPLETE' if complete else 'INCOMPLETE',
                  n=sum(_number(v) for v in values.values()))
    if not complete:
        return dict(result, mean=None, sd=None, ci95=None)
    a = np.asarray([values[s] for s in SEEDS]); mean = float(a.mean()); sd = float(a.std(ddof=1))
    half = float(2.7764451052 * sd / np.sqrt(5.))
    return dict(result, mean=mean, sd=sd, ci95=[mean-half, mean+half],
                positive=int((a > 0).sum()), negative=int((a < 0).sum()), zero=int((a == 0).sum()))


def feature_manifest():
    names = [f'action_onehot_{i}' for i in range(6)] + ['epoch_over_50'] + list(EXTRA_NAMES)
    return dict(adapted_definition=True, feature_order=names,
                definitions='research_plans/26-11-10_ema_causal_attribution/08_ANALYSIS_AND_DECISION.md#4',
                source_code_path=__file__, label_access='P_A only; no validation labels in predictors',
                source_clock='origin epoch', gradient_definition='six-cell mean of all 256x256 softplus pairs; full functional graph',
                continuous_columns=[6, 7, 8, 9, 12, 13, 14],
                binary_columns=[0, 1, 2, 3, 4, 5, 10, 11],
                penalty='sum_squared_error + 1 * nonintercept_beta_squared',
                training_origins=list(ORIGINS), training_targets=[e+5 for e in ORIGINS],
                evaluation_origins=list(EVALUATION_ORIGINS),
                evaluation_targets=[e+5 for e in EVALUATION_ORIGINS],
                statistical_unit='seed', tie_order={m: action_order(m) for m in MENUS})


def _evaluation_row(index, seed, origin, menu, selector, population, selected, static, predictions):
    target = origin + 5; order = action_order(menu)
    records = {a: index.get((seed, target, a)) for a in order}; reference = records['F']
    utilities = {a: _utility(records[a], reference, population) for a in order}
    oracle = choose(utilities, order)
    value = utilities.get(selected); static_value = utilities.get(static)
    predicted_values = [predictions.get(a) for a in order[1:]] if predictions else []
    actual_values = [utilities[a] for a in order[1:]]
    mae = _mean([abs(p-v) if _number(p) and _number(v) else None
                 for p, v in zip(predicted_values, actual_values)])
    selected_record = records.get(selected)
    return dict(seed=seed, origin_epoch=origin, target_epoch=target, menu=menu,
                selector=selector, population=population, selected_action=selected,
                static_action=static, oracle_action=oracle, utility=value,
                static_utility=static_value,
                delta_static=float(value-static_value) if _number(value) and _number(static_value) else None,
                regret=float(utilities[oracle]-value) if oracle and _number(value) else None,
                mae=mae, cell_auc_delta=_difference(selected_record, reference, population, 'auc'),
                cell_logloss_delta=_difference(selected_record, reference, population, 'logloss'),
                macro_logloss_delta=_mean(_difference(selected_record, reference, population, 'logloss')),
                state_id=selected_record.get('state_id') if selected_record else None,
                status='ESTIMABLE' if selected and _number(value) else 'UNESTIMABLE')


def run_bridge(records):
    """Return scalar bridge tables from all ten pulse epochs of one or more conditions.

    Every input action has metrics[population] as six {auc, logloss} rows in
    domain-major/task-minor order and seven raw geometry fields (see manifest).
    """
    groups = defaultdict(list)
    for row in records:
        if row['menu'] not in MENUS or row['action_id'] not in action_order(row['menu']):
            raise ValueError('Unknown frozen bridge menu/action')
        if row['seed'] not in SEEDS or row['epoch'] not in range(5, 51, 5):
            raise ValueError('Bridge seed or clock outside frozen schedule')
        groups[(row['dataset'], row['objective'], row['menu'])].append(row)
    output = dict(FEATURE_MANIFEST=feature_manifest(), BRIDGE_FEATURES=[], BRIDGE_FOLDS=[],
                  BRIDGE_PREDICTIONS=[], BRIDGE_VALUE=[], BRIDGE_SUMMARY=[],
                  CURRENT_CHOICES=[], H2_TRIGGER=[])
    for (dataset, objective, menu), rows in sorted(groups.items()):
        identity = dict(dataset=dataset, objective=objective, menu=menu)
        index = {(r['seed'], r['epoch'], r['action_id']): r for r in rows}
        if len(index) != len(rows):
            raise ValueError('Duplicate bridge action identity')
        order = action_order(menu); populations = sorted(set(p for r in rows for p in r.get('metrics', {})))
        features = {}
        for (seed, epoch, action), row in sorted(index.items()):
            if action == 'F':
                continue
            base, full = _features(row, index.get((seed, epoch, 'F')), menu)
            features[(seed, epoch, action)] = (base, full)
            output['BRIDGE_FEATURES'].append(dict(identity, seed=seed, origin_epoch=epoch,
                action_id=action, source_epoch=epoch, label_access='current P_A',
                design=base, simple=full, status='ESTIMABLE' if all(_number(v) for v in full) else 'UNESTIMABLE'))
        evaluation = []
        for holdout in SEEDS:
            train = []
            for seed in SEEDS:
                if seed == holdout:
                    continue
                for origin in ORIGINS:
                    for action in order[1:]:
                        feature = features.get((seed, origin, action))
                        target = _utility(index.get((seed, origin+5, action)), index.get((seed, origin+5, 'F')), 'P_A')
                        train.append(dict(seed=seed, origin=origin, action=action,
                                          features=feature, target=target))
            static_means = {'F': 0.}
            for action in order[1:]:
                static_means[action] = _mean([r['target'] for r in train if r['action'] == action])
            static = choose(static_means, order)
            models = {}
            for kind, column, continuous in (('C_design_adapted', 0, [6]),
                                              ('C_simple_adapted', 1, [6, 7, 8, 9, 12, 13, 14])):
                valid = [r for r in train if r['features'] is not None and _number(r['target'])
                         and all(_number(v) for v in r['features'][column])]
                # Missing rows remain explicit; no imputed features or labels.
                model = _ridge([r['features'][column] for r in valid], [r['target'] for r in valid], continuous) if valid else None
                models[kind] = model
                output['BRIDGE_FOLDS'].append(dict(identity, holdout_seed=holdout, predictor=kind,
                    train_origins=list(ORIGINS), train_targets=[e+5 for e in ORIGINS],
                    evaluation_origins=list(EVALUATION_ORIGINS), expected_training_rows=72,
                    training_rows=len(valid), temporal_units=len({(r['seed'], r['origin']) for r in valid}),
                    training_row_hash=_hash(valid), excluded_rows=[dict(seed=r['seed'], origin=r['origin'], action=r['action']) for r in train if r not in valid],
                    feature_hash=_hash(output['FEATURE_MANIFEST']), menu_hash=_hash(order),
                    scaler=model, static_action=static, static_P_A_means=static_means,
                    status='COMPLETE' if len(valid)==72 else 'INCOMPLETE'))
            for origin in EVALUATION_ORIGINS:
                predictions = {kind: {'F': 0.} for kind in PREDICTORS}
                for action in order[1:]:
                    pair = features.get((holdout, origin, action))
                    for kind, column in (('C_design_adapted', 0), ('C_simple_adapted', 1)):
                        predictions[kind][action] = _predict(models[kind], pair[column]) if pair else None
                    predictions['C_carry_direct'][action] = _utility(index.get((holdout, origin, action)), index.get((holdout, origin, 'F')), 'P_A')
                selections = {kind: choose(pred, order) for kind, pred in predictions.items()}
                selections.update(F='F', static=static)
                for population in populations:
                    target_utils = {a: _utility(index.get((holdout, origin+5, a)), index.get((holdout, origin+5, 'F')), population) for a in order}
                    selections['oracle'] = choose(target_utils, order)
                    for selector, selected in selections.items():
                        evaluation.append(dict(identity, **_evaluation_row(index, holdout, origin, menu,
                            selector, population, selected, static, predictions.get(selector))))
                    for kind, pred in predictions.items():
                        for action in order:
                            output['BRIDGE_PREDICTIONS'].append(dict(identity, holdout_seed=holdout,
                                origin_epoch=origin, target_epoch=origin+5, predictor=kind,
                                action_id=action, population=population, predicted_utility=pred[action],
                                actual_utility=target_utils[action], selected_action=selections[kind],
                                static_action=static, oracle_action=selections['oracle'],
                                label_access='origin P_A; other-seed early target P_A for ridge',
                                status='ESTIMABLE' if _number(pred[action]) else 'UNESTIMABLE'))
        for seed in SEEDS:
            for epoch in range(5, 51, 5):
                for chooser, selection_population in (('validation_split_choice', 'V_A'), ('train_direct_choice', 'P_A')):
                    utility = {a: _utility(index.get((seed, epoch, a)), index.get((seed, epoch, 'F')), selection_population) for a in order}
                    selected = choose(utility, order)
                    for population in populations:
                        row = _evaluation_row(index, seed, epoch-5, menu, chooser, population, selected, 'F', None)
                        row.update(identity, epoch=epoch, selection_population=selection_population)
                        output['CURRENT_CHOICES'].append(row)
        buckets = defaultdict(list)
        for row in evaluation:
            buckets[(row['seed'], row['selector'], row['population'])].append(row)
        for (seed, selector, population), values in sorted(buckets.items()):
            counts = Counter(v['selected_action'] for v in values if v['selected_action'] is not None)
            probabilities = np.asarray(list(counts.values()), dtype=float) / len(values)
            result = dict(identity, seed=seed, selector=selector, population=population,
                windows=len(values), estimable_windows=sum(v['status']=='ESTIMABLE' for v in values),
                action_counts=dict(counts), action_entropy=float(-(probabilities*np.log(probabilities)).sum()),
                fully_static=len(counts)==1 and sum(counts.values())==len(EVALUATION_ORIGINS),
                status='COMPLETE' if all(v['status']=='ESTIMABLE' for v in values) else 'INCOMPLETE')
            for name in ('utility', 'delta_static', 'regret', 'mae', 'macro_logloss_delta'):
                result[name] = _mean([v[name] for v in values])
            for name in ('cell_auc_delta', 'cell_logloss_delta'):
                result[name] = [_mean([v[name][cell] for v in values]) for cell in range(6)]
            output['BRIDGE_VALUE'].append(result)
    buckets = defaultdict(dict)
    for row in output['BRIDGE_VALUE']:
        for metric in ('utility', 'delta_static', 'regret', 'mae', 'macro_logloss_delta'):
            key = tuple(row[k] for k in ('dataset', 'objective', 'menu', 'selector', 'population')) + (metric,)
            buckets[key][row['seed']] = row[metric]
        for metric in ('cell_auc_delta', 'cell_logloss_delta'):
            for cell in range(6):
                key = tuple(row[k] for k in ('dataset', 'objective', 'menu', 'selector', 'population')) + (f'{metric}_{cell}',)
                buckets[key][row['seed']] = row[metric][cell]
    for key, values in sorted(buckets.items()):
        output['BRIDGE_SUMMARY'].append(dict(zip(('dataset','objective','menu','selector','population','metric'),key), **seed_summary(values)))
    for dataset, objective in sorted({(r['dataset'],r['objective']) for r in records}):
        output['H2_TRIGGER'].append(h2_trigger(output, dataset, objective))
    return output


def h2_trigger(tables, dataset, objective):
    """08 §6.1 pre-registered budget trigger; no scientific acceptance gate."""
    current = [r for r in tables['CURRENT_CHOICES'] if r['dataset']==dataset and r['objective']==objective
               and r['menu']=='mechanism7' and r['selector']=='validation_split_choice'
               and r['population']=='V_B' and r['epoch'] in (35,50)]
    space = seed_summary({s: _mean([r['utility'] for r in current if r['seed']==s])
                          if sum(r['seed']==s for r in current)==2 else None for s in SEEDS})
    summaries = tables['BRIDGE_SUMMARY']
    def get(menu, selector, metric):
        return next((r for r in summaries if r['dataset']==dataset and r['objective']==objective
                     and r['menu']==menu and r['selector']==selector
                     and r['population']=='V_mech' and r['metric']==metric), {})
    rules = []
    for selector in PREDICTORS:
        mechanism = get('mechanism7', selector, 'utility'); delta = get('mechanism7', selector, 'delta_static'); rho = get('rho7', selector, 'utility')
        complete = all(r.get('status')=='COMPLETE' for r in (mechanism,delta,rho))
        rules.append(dict(selector=selector, mechanism=mechanism, delta_static=delta, rho=rho,
                          evaluable=complete, positive_vs_F_and_static=complete and mechanism['mean']>0 and delta['mean']>0,
                          higher_than_rho=complete and mechanism['mean']>rho['mean']))
    complete = space['status']=='COMPLETE' and all(r['evaluable'] for r in rules)
    triggered = complete and space['mean']>.0002 and any(r['positive_vs_F_and_static'] and r['higher_than_rho'] for r in rules)
    return dict(dataset=dataset, objective=objective, trigger='H2_MATCH', space=space, rules=rules,
                status='TRIGGERED' if triggered else ('NOT_TRIGGERED' if complete else 'DEPENDENCY_INCOMPLETE'),
                triggered=triggered, input_hash=_hash(dict(current=current,rules=rules)), epsilon=.0002)
