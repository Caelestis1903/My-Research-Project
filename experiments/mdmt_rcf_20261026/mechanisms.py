"""Registered RCF functional profiles and standard fixed-space controls."""
from __future__ import annotations

import itertools
import time
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from .protocol import FeatureBank, derive_rng, record_access
from .vendor_snapshot.mdmt_dfd_20261026.state import array_hash, digest
from .demand import outer_requests
from .memory_data import persist,release_data_pages


def _projection_bank(bundle, projections, method, *, metadata=None):
    values, columns, bounds = {}, {}, {}
    for target in range(3):
        for task in range(2):
            owners = [(s, projections[s, task]) for s in range(3) if s != target
                      and (s, task) in projections]
            columns[target, task] = [dict(function_id=f'{method}_s{s}_q{task}_k{k}',
                source=s, task=task, slot=k) for s, matrix in owners for k in range(matrix.shape[1])]
            bounds[target, task] = [(-1., 1.)]*len(columns[target, task])
            for role, view in bundle.views.items():
                mask = view.domain == target
                pieces = [bundle.pilots.hidden(role, s, task, standardized=True)[mask] @ matrix
                          for s, matrix in owners]
                values[role, target, task] = np.concatenate(pieces, axis=1) if pieces else np.empty((int(mask.sum()), 0))
    bank = FeatureBank(values=values, columns=columns, bounds=bounds,
        identity=dict(method=method, dataset=bundle.dataset, seed=bundle.seed,
                      projection_hashes={f'{s}_{q}':array_hash(v) for (s,q),v in projections.items()},
                      **(metadata or {})))
    # RAM-only reconstruction factors; never serialize this object as model state.
    bank.projections = projections
    return bank


def spectral_directions(demand):
    directions = {}
    for key, unit in demand.units.items():
        if unit.get('metric') is None or not unit.get('groups'):
            continue
        matrix = sum((g['pi']*np.outer(g['score'], g['score']) for g in unit['groups']),
                     np.zeros_like(unit['metric']))
        matrix = (matrix+matrix.T)/2
        eigenvalues, vectors = eigh(matrix, unit['metric'])
        directions[key] = vectors[:, np.argsort(eigenvalues)[::-1][:2]]
    return directions


def spectral_features(bundle, demand, config):
    return _projection_bank(bundle, spectral_directions(demand), 'SPECTRAL_K2',
        metadata=dict(demand=demand.identity, interpretation='common_metric_quadratic_proxy'))


def fit_rrr(bundle, demand, config, *, ungrouped=False):
    """Two genuine rank-two GLM fits; downstream reader selection remains separate."""
    pf = bundle.views['P_form']
    device = config.get('_device', getattr(bundle, 'device', 'cpu'))
    method = 'UNGROUPED_RRR' if ungrouped else 'RRR_K2'
    raw_phi = np.stack([np.stack([bundle.pilots.hidden('P_form', s, q, standardized=True)
                                 for q in range(2)], axis=1) for s in range(3)], axis=1)
    # [N,source,task,phi]
    phi = torch.as_tensor(raw_phi, dtype=torch.float32, device=device)
    b = torch.as_tensor(np.array(bundle.base_oof), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.array(pf.y), dtype=torch.float32, device=device)
    domain = torch.as_tensor(pf.domain, dtype=torch.long, device=device)
    groups = np.zeros_like(pf.group) if ungrouped else pf.group
    group_tensor = torch.as_tensor(groups, dtype=torch.long, device=device)
    ng = 1 if ungrouped else 2
    group_dim = torch.zeros((3, 2, ng), device=device)
    counts = np.bincount(pf.domain, minlength=3)
    row_weights = torch.as_tensor(len(pf.row_id)/(3*counts[pf.domain]), dtype=torch.float32, device=device)
    spectral = spectral_directions(demand)
    banks = []
    for start_id, start_name in enumerate(('RANDOM', 'SPECTRAL')):
        v0 = np.empty((3, 2, 32, 2), np.float32)
        for s in range(3):
            for q in range(2):
                rng = derive_rng(bundle.seed, bundle.dataset, 'rrr_init', s, q,
                                 unit=(100 if ungrouped else 0)+start_id)
                random = rng.normal(0, .1, (32, 2)).astype(np.float32)
                v0[s, q] = spectral.get((s, q), random) if start_name == 'SPECTRAL' else random
        v = torch.nn.Parameter(torch.tensor(v0, device=device))
        alpha = torch.nn.Parameter(torch.ones_like(group_dim))
        beta = torch.nn.Parameter(torch.zeros_like(group_dim))
        coeff = torch.nn.Parameter(torch.zeros((3, 2, ng, 3, 2), device=device))
        optimizer = torch.optim.Adam([v, alpha, beta, coeff], lr=.001,
                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
        records, started = [], time.monotonic()
        for pass_id in range(1, 21):
            order = derive_rng(bundle.seed, bundle.dataset, 'rrr_plan',
                unit=(100 if ungrouped else 0)+start_id, pass_id=pass_id).permutation(len(pf.row_id))
            loss_sum, rows = 0.0, 0
            for begin in range(0, len(order), 512):
                idx = order[begin:begin+512]
                optimizer.zero_grad(set_to_none=True)
                projected = torch.einsum('nsqf,sqfk->nsqk', phi[idx], v)
                combined = torch.zeros((len(idx), 2), device=device)
                calibrated = torch.zeros_like(combined)
                for q in range(2):
                    d, g = domain[idx], group_tensor[idx, q]
                    chosen = coeff[d, q, g]
                    # source == target is excluded from the predictive model.
                    mask = (torch.arange(3, device=device)[None, :] != d[:, None]).to(phi.dtype)
                    combined[:, q] = (projected[:, :, q]*chosen*mask[:, :, None]).sum((1, 2))
                    calibrated[:, q] = alpha[d, q, g]*b[idx, q]+beta[d, q, g]
                target = (row_weights[idx, None]*F.binary_cross_entropy_with_logits(
                    calibrated+combined, y[idx], reduction='none')).mean()
                correction_rms = (row_weights[idx, None]*combined.square()).mean()
                objective = target+.1*correction_rms/2+1e-6*v.square().sum()/2
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite RRR objective')
                objective.backward()
                optimizer.step()
                with torch.no_grad():
                    alpha.clamp_(.25, 4)
                    beta.clamp_(-4, 4)
                    coeff.clamp_(-1, 1)
                record_access(bundle, method, 'P_form', pf.row_id[idx],
                    purpose='formation', start=start_name, pass_id=pass_id)
                loss_sum += float(objective.detach())*len(idx)
                rows += len(idx)
            records.append(dict(pass_id=pass_id, objective=loss_sum/rows,
                                rows=rows, row_plan_hash=array_hash(pf.row_id[order])))
        projections = {(s, q):v[s, q].detach().cpu().numpy().astype(np.float64)
                       for s in range(3) for q in range(2)}
        bank = _projection_bank(bundle, projections, method, metadata=dict(
            start=start_name, start_id=start_id, passes=20, records=records,
            elapsed_seconds=time.monotonic()-started, group_formation=not ungrouped,
            temporary_read_parameters_discarded=True, reader_candidate_lambdas=[.001, .1],
            extra_start_cost_disclosed=True))
        banks.append(bank)
    return banks


def _group_metric(view, probabilities, target, task, group):
    mask = (view.domain == target) & (view.group[:, task] == group)
    y = np.asarray(view.y[mask, task], np.float64)
    p = np.asarray(probabilities[mask, task], np.float64)
    p = np.clip(p, 1e-7, 1-1e-7)
    return dict(n=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))) if len(y) else None,
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None)


def _subset_bank(bank, function_id):
    values, columns, bounds = {}, {}, {}
    for key, cols in bank.columns.items():
        selected = [i for i, col in enumerate(cols) if col['function_id'] == function_id]
        columns[key] = [cols[i] for i in selected]
        bounds[key] = [bank.bounds[key][i] for i in selected]
        for (role, target, task), a in bank.values.items():
            if (target, task) == key:
                values[role, target, task] = a[:, selected]
    subset = FeatureBank(values=values, columns=columns, bounds=bounds,
                        identity=dict(method='SINGLE_FUNCTION', parent=bank.identity,
                                      function_id=function_id))
    if hasattr(bank, 'functions'):
        subset.functions = {function_id:bank.functions[function_id]}
    return subset


def _frozen_reader_logits(bundle, predictions, role, bank=None):
    """Evaluate the actual frozen logit function, including saturated predictions.

    A probability (even before clipping) cannot recover a logit after sigmoid
    saturation. Mechanism interventions therefore use the resident coefficients
    and fixed features, exactly as the selected reader does.
    """
    coefficients = predictions.diagnostics.get('selected_coefficients')
    predictor = predictions.diagnostics.get('predictor')
    if coefficients is None and predictor is not None:
        if getattr(predictor, 'off', False):
            return np.asarray(bundle.base_logits[role], np.float64).copy()
        coefficients = getattr(predictor, 'coefficients', None)
    if coefficients is None:
        raise RuntimeError('Frozen coefficients are required to reconstruct actual reader logits')
    if bank is None and predictor is not None:
        bank = getattr(predictor, 'bank', None)
    view = bundle.views[role]
    base = np.asarray(bundle.base_logits[role], np.float64)
    result = np.empty_like(base)
    for target in range(3):
        pos = np.flatnonzero(view.domain == target)
        for task in range(2):
            for group in (0, 1):
                policy = coefficients.get((target, task, group))
                if policy is None:
                    policy = coefficients.get(f'{target}_{task}_{group}')
                if policy is None:
                    raise RuntimeError('Missing frozen group coefficients for mechanism intervention')
                own = view.group[pos, task] == group
                rows = pos[own]
                weights = np.asarray(policy.get('weights', policy.get('w')), np.float64)
                if len(weights):
                    if bank is None:
                        raise RuntimeError('Frozen feature bank missing for nonempty reader coefficients')
                    x = np.asarray(bank.values[role, target, task][own], np.float64)
                    correction = x @ weights
                else:
                    correction = 0.0
                result[rows, task] = policy['alpha']*base[rows, task]+policy['beta']+correction
    return result


def _frozen_delete_predictions(bundle, formed, predictions, function_id, replacement_id=None):
    """Uses the actual frozen reader; no retraining, outer selection, or sign choice."""
    coefficients = predictions.diagnostics.get('selected_coefficients')
    if coefficients is None:
        predictor = predictions.diagnostics.get('predictor')
        if callable(predictor) or hasattr(predictor, 'predict'):
            replacement = formed.functions.get(replacement_id) if replacement_id else None
            altered = copy_feature_bank_without(formed.features, function_id, replacement, bundle=bundle)
            predict = predictor.predict if hasattr(predictor,'predict') else predictor
            return {role:predict(role, altered) for role in bundle.views}
        return None
    replacement = formed.functions.get(replacement_id) if replacement_id else None
    altered = copy_feature_bank_without(formed.features, function_id, replacement, bundle=bundle)
    # Re-evaluate the edited fixed feature expression directly rather than
    # subtracting from a sigmoid-inverted probability or a rounded huge sum.
    return {role:expit(_frozen_reader_logits(bundle, predictions, role, altered))
            for role in bundle.views}


def copy_feature_bank_without(bank, function_id, replacement=None, *, bundle=None):
    """Preserve columns and zero just the selected function for frozen prediction."""
    values = {key:np.array(value, copy=True) for key,value in bank.values.items()}
    for (target, task), columns in bank.columns.items():
        for i, col in enumerate(columns):
            if col['function_id'] == function_id:
                for (role, d, q), value in values.items():
                    if (d,q) == (target,task):
                        if replacement is None:
                            value[:, i] = 0
                        else:
                            # The replacement shares source/task and is consumed by the
                            # same target rows in the original frozen reader column.
                            if bundle is None:
                                raise RuntimeError('Replacement predictor requires target row positions')
                            original = np.flatnonzero(bundle.views[role].domain == target)
                            value[:, i] = replacement.outputs[role][original]
    return FeatureBank(values=values, columns=bank.columns, bounds=bank.bounds,
                        identity=dict(bank.identity, deleted_function=function_id))


def diagnose_functions(bundle, formed, predictions, config):
    from .baselines import fixed_feature_bank
    from .readers import fit_reader
    if not hasattr(bundle, '_mechanism_group_cal'):
        bundle._mechanism_group_cal = fit_reader(bundle,
            fixed_feature_bank(bundle, 'GROUP_CAL', config), 'GROUP_CAL', config)
    group_cal = bundle._mechanism_group_cal
    roles = ['P_form', 'P_cal', 'Q_explore', 'validation']
    if config.get('_outer_authorized', False):
        roles += ['test_exposed', 'Q_legacy_confirm_exposed']
    profiles, retention, deletion, crossovers, replacements, utility_drift = [], [], [], [], [], []
    if not hasattr(bundle, '_mechanism_single_functions'):
        bundle._mechanism_single_functions = {}
    single_results = {}
    for function_id, function in formed.functions.items():
        cache_key = (formed.identity['functions_hash'], function_id)
        if cache_key not in bundle._mechanism_single_functions:
            bundle._mechanism_single_functions[cache_key] = fit_reader(bundle,
                _subset_bank(formed.features, function_id),
                f'{formed.method}_SINGLE_{function_id}', config)
        single = bundle._mechanism_single_functions[cache_key]
        single_results[function_id] = single
        removed = _frozen_delete_predictions(bundle, formed, predictions, function_id)
        if removed is None:
            raise RuntimeError('Frozen reader coefficients/predictor required for deletion profiles')
        for role in roles:
            view = bundle.views[role]
            record_access(bundle, formed.method+'_DIAGNOSE', role, view.row_id,
                          purpose='diagnose',
                          function_id=function_id)
            delta = function.outputs.get(role)
            if delta is None:
                delta = function.values(view.h)
            targets = [function.target] if function.target is not None else [t for t in range(3) if t != function.source]
            base_p = group_cal.probabilities[role]
            base_logit = _frozen_reader_logits(bundle, group_cal, role)
            plus, minus = base_p.copy(), base_p.copy()
            active = delta != 0
            plus[active, function.task] = expit(base_logit[active, function.task]+delta[active])
            minus[active, function.task] = expit(base_logit[active, function.task]-delta[active])
            for target in targets:
                for group in (0, 1):
                    common = dict(method=formed.method, role=role, function_id=function_id,
                        source=function.source, task=function.task, slot=function.slot,
                        target=target, group=group)
                    ref = _group_metric(view, base_p, target, function.task, group)
                    for action, p in [('FIXED_PLUS', plus), ('FIXED_MINUS', minus),
                                      ('SINGLE_READER', single.probabilities[role])]:
                        metric = _group_metric(view, p, target, function.task, group)
                        profiles.append(dict(common, action=action, **metric,
                            delta_ll_vs_group_cal=None if metric['logloss'] is None else
                                metric['logloss']-ref['logloss'],
                            delta_auc_vs_group_cal=None if metric['auc'] is None else
                                metric['auc']-ref['auc']))
                    original = _group_metric(view, predictions.probabilities[role], target, function.task, group)
                    changed = _group_metric(view, removed[role], target, function.task, group)
                    deletion.append(dict(common, action='FROZEN_DELETE', **changed,
                        delta_ll_vs_full=None if changed['logloss'] is None else changed['logloss']-original['logloss']))
                    mask = (view.domain == target) & (view.group[:, function.task] == group)
                    label = np.asarray(view.y[mask,function.task],np.float64)
                    d = np.asarray(delta[mask],np.float64)
                    refit = np.asarray(bundle.base_logits[role][mask,function.task],np.float64)
                    reference_ll = np.logaddexp(0,refit)-label*refit
                    new_ll = np.logaddexp(0,refit+d)-label*(refit+d)
                    drift_record = dict(common, refit_delta_ll=float(np.mean(new_ll-reference_ll)))
                    if role == 'P_form':
                        old = np.asarray(bundle.base_oof[mask,function.task],np.float64)
                        old_change = np.logaddexp(0,old+d)-np.logaddexp(0,old)-label*d
                        drift_record.update(oof_delta_ll=float(np.mean(old_change)),
                            sign_changed=bool(np.mean(old_change)*np.mean(new_ll-reference_ll)<0))
                    utility_drift.append(drift_record)
            source_pos = np.flatnonzero(view.domain == function.source)
            if function.target is None:
                anchor = function.anchor_pilots or bundle.pilots
                a = np.asarray(anchor.logits(role, function.source, function.task)[source_pos], np.float64)
                d = np.asarray(delta[source_pos], np.float64)
                y = view.y[source_pos, function.task]
                s = float(np.mean(np.logaddexp(0,a+d)-np.logaddexp(0,a)-y*d))
                k = float(np.mean(np.logaddexp(0,a+d)-np.logaddexp(0,a)-expit(a)*d))
                foreign = [np.flatnonzero(view.domain == t) for t in range(3) if t != function.source]
                rms = .5*sum(float(np.mean(np.asarray(delta[pos], np.float64)**2)) for pos in foreign)
                retention.append(dict(function_id=function_id, role=role, source=function.source,
                    task=function.task, S=s, K=k, R=rms, c=function.scale,
                    source_budget_exceeded=s>config['formation']['source_ll_budget'],
                    rms_budget_exceeded=rms>config['formation']['foreign_delta_mean_square_budget']))
    # Every predeclared target-group pair is reported; no best outer pair or sign is chosen.
    for source in range(3):
        for task in range(2):
            functions = [f for f in formed.functions.values() if f.source == source and f.task == task]
            for first, second in itertools.combinations(functions, 2):
                for removed_fn, replacement_fn in ((first,second),(second,first)):
                    changed_predictions = _frozen_delete_predictions(bundle,formed,predictions,
                        removed_fn.function_id,replacement_fn.function_id)
                    if changed_predictions is None:
                        raise RuntimeError('Frozen replacement prediction unavailable')
                    for role in roles:
                        view = bundle.views[role]
                        targets = [first.target] if first.target is not None else [t for t in range(3) if t != source]
                        for target in targets:
                            for group in (0,1):
                                old = _group_metric(view,predictions.probabilities[role],target,task,group)
                                new = _group_metric(view,changed_predictions[role],target,task,group)
                                replacements.append(dict(role=role,source=source,task=task,target=target,group=group,
                                    removed_function=removed_fn.function_id,replacement_function=replacement_fn.function_id,
                                    **new,delta_ll_vs_full=None if new['logloss'] is None else new['logloss']-old['logloss']))
                for role in roles:
                    lookup = {(x['function_id'], x['target'], x['group']):x for x in profiles
                              if x['role'] == role and x['action'] == 'FIXED_PLUS' and x['task'] == task}
                    group_ids = [(t,g) for t in range(3) if t != source for g in (0,1)]
                    for ga, gb in itertools.combinations(group_ids, 2):
                        rows = [lookup.get((f.function_id, *group)) for f,group in
                                [(first,ga),(second,ga),(first,gb),(second,gb)]]
                        if any(x is None or x['logloss'] is None for x in rows):
                            continue
                        a0, a1, b0, b1 = rows
                        sufficient = all(x['n'] >= config['groups']['min_rows'] and
                            x['positive'] >= config['groups']['min_positive'] and
                            x['negative'] >= config['groups']['min_negative'] for x in rows)
                        da, db = a0['logloss']-a1['logloss'], b0['logloss']-b1['logloss']
                        crossovers.append(dict(role=role, source=source, task=task,
                            function_a=first.function_id, function_b=second.function_id,
                            target_a=ga[0], group_a=ga[1], target_b=gb[0], group_b=gb[1],
                            delta_a=da, delta_b=db, cross_advantage=da*db<0,
                            sufficient_support=sufficient,
                            action='FIXED_PLUS_NO_OUTER_SIGN_SELECTION'))
    pf = bundle.views['P_form']
    refit = bundle.base_logits['P_form']
    drift = dict(mean_abs=float(np.mean(np.abs(bundle.base_oof-refit))),
                 rms=float(np.sqrt(np.mean((bundle.base_oof-refit)**2))), cells=[])
    for task in range(2):
        for target in range(3):
            mask = pf.domain == target
            old = np.asarray(bundle.base_oof[mask,task], np.float64)
            new = np.asarray(refit[mask,task], np.float64)
            drift['cells'].append(dict(target=target, task=task,
                mean_abs=float(np.mean(np.abs(old-new))),
                oof_ll=float(np.mean(np.logaddexp(0,old)-pf.y[mask,task]*old)),
                refit_training_ll=float(np.mean(np.logaddexp(0,new)-pf.y[mask,task]*new))))
    result = dict(method=formed.method, group_profiles=profiles, source_retention=retention,
        deletion_profiles=deletion, crossovers=crossovers, oof_refit_drift=drift,
        replacement_profiles=replacements, single_function_oof_refit_utility=utility_drift,
        single_function_predictions=single_results,
        old_eta_mu='NOT_APPLICABLE', roles=roles,
        requests={role:outer_requests(getattr(formed,'request_bundle',bundle), formed.requests, role) for role in roles}
                 if hasattr(formed.requests, 'units') else {'status':'NOT_APPLICABLE'})
    for role,request_data in result['requests'].items():
        if isinstance(request_data,dict):
            for name in ('request_projection','requested_slot','supported_request'):
                if name in request_data:
                    request_data[name]=persist(bundle,('diagnostic_request',formed.method,role,name),request_data[name])
    release_data_pages(bundle)
    return result
