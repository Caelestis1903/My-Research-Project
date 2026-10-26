"""RCF-v1 profiled target risk responses and frozen finite assignments."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.special import expit

from .protocol import DemandResult, record_access
from .vendor_snapshot.mdmt_dfd_20261026.state import digest, array_hash


def profiled_moments(phi, b, y, config):
    """All moments are conditional on this group's rows; pi is NOT applied here."""
    phi = np.asarray(phi, dtype=np.float64)
    b, y = np.asarray(b, dtype=np.float64), np.asarray(y, dtype=np.float64)
    p = expit(b)
    e, v = y-p, p*(1-p)
    n = np.column_stack((np.ones(len(b)), b))
    bp = np.mean(e[:, None]*phi, axis=0)
    bn = np.mean(e[:, None]*n, axis=0)
    hpp = (phi.T * v) @ phi / len(b)
    hpn = (phi.T * v) @ n / len(b)
    hnn = (n.T * v) @ n / len(b)
    dc = config['demand']
    rho = max(dc['nuisance_ridge_floor'],
              dc['nuisance_ridge_relative']*float(np.trace(hnn))/2)
    score = bp-hpn @ np.linalg.solve(hnn+rho*np.eye(2), bn)
    return dict(score=score, b_phi=bp, b_n=bn, H_phi_phi=hpp,
                H_phi_n=hpn, H_nn=hnn, nuisance_ridge=rho)


def symmetric_metric(hessians, pi, source_phi, config):
    source_phi = np.asarray(source_phi, dtype=np.float64)
    centered = source_phi-source_phi.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    c0 = sum((w*h for w, h in zip(pi, hessians)), np.zeros_like(covariance))
    c0 += config['demand']['source_covariance_weight']*covariance
    c0 = (c0+c0.T)/2
    raw_eig = np.linalg.eigvalsh(c0)
    relative = max(float(np.max(np.abs(raw_eig))), np.finfo(float).tiny)
    if float(raw_eig.min()) < -1e-8*relative:
        raise FloatingPointError('Materially negative target/source metric eigenvalue')
    dc = config['demand']
    rho = max(dc['metric_ridge_floor'], dc['metric_ridge_relative']*np.trace(c0)/len(c0))
    metric = c0+rho*np.eye(len(c0))
    eigenvalues, vectors = np.linalg.eigh(metric)
    condition = float(eigenvalues.max()/eigenvalues.min())
    if eigenvalues.min() <= 0 or condition > dc['max_condition_number']:
        raise FloatingPointError(f'Invalid RCF metric condition={condition}')
    inverse_root = (vectors/np.sqrt(eigenvalues)) @ vectors.T
    return metric, inverse_root, dict(ridge=float(rho), condition=condition,
                                    eigenvalues=eigenvalues, source_covariance=covariance)


def exhaustive_assignment(responses, pi, group_ids):
    """Minimum weighted two-means over all nonempty bipartitions; fixed tie rule."""
    r, pi = np.asarray(responses, np.float64), np.asarray(pi, np.float64)
    n = len(r)
    if n == 1:
        return np.ones((1, 1)), 0.0
    best = None
    for tail in itertools.product((0, 1), repeat=n-1):
        labels = np.asarray((0,)+tail)
        if not np.any(labels == 1):
            continue
        objective = 0.0
        for k in (0, 1):
            mask = labels == k
            center = np.average(r[mask], axis=0, weights=pi[mask])
            objective += float(np.sum(pi[mask, None]*(r[mask]-center)**2))
        members = tuple(tuple(group_ids[i] for i in range(n) if labels[i] == k)
                        for k in (0, 1))
        key = (objective, members)
        if best is None or key < best[0]:
            best = (key, labels)
    return np.eye(2)[best[1]], float(best[0][0])


def compute_demand(bundle, config):
    view = bundle.views['P_form']
    fit = bundle.views['fit']
    if bundle.base_oof is None:
        raise RuntimeError('B OOF must be fitted before demand')
    units = {}
    gc = config['groups']
    for source in range(3):
        for task in range(2):
            phi = bundle.pilots.hidden('P_form', source, task, standardized=True)
            groups, unsupported = [], []
            for target in range(3):
                if target == source:
                    continue
                nt = int(np.sum(view.domain == target))
                for group in (0, 1):
                    pos = np.flatnonzero((view.domain == target) & (view.group[:, task] == group))
                    labels = view.y[pos, task]
                    positive, negative = int(labels.sum()), int(len(pos)-labels.sum())
                    record_access(bundle, 'RCF_DEMAND', 'P_form', view.row_id[pos], task,
                                  purpose='demand', target=target, group=group)
                    item = dict(target=target, group=group, rows=view.row_id[pos],
                                positions=pos, n=len(pos), positive=positive, negative=negative,
                                pi_raw=0.5*len(pos)/nt if nt else 0.0)
                    if (len(pos) < gc['min_rows'] or positive < gc['min_positive']
                            or negative < gc['min_negative']):
                        item['status'] = 'LOW_SUPPORT'
                        unsupported.append(item)
                        continue
                    item.update(profiled_moments(phi[pos], bundle.base_oof[pos, task], labels, config))
                    item['status'] = 'ELIGIBLE'
                    groups.append(item)
            if not groups:
                units[source, task] = dict(source=source, task=task, effective_k=0,
                                           groups=[], unsupported=unsupported,
                                           assignment=np.empty((0, 0)), metric=None,
                                           status='NO_ESTIMABLE_REQUEST',
                                           reason='NO_SUPPORTED_TARGET_GROUP')
                continue
            mass = sum(g['pi_raw'] for g in groups)
            for g in groups:
                g['pi'] = g['pi_raw']/mass
            pi = np.asarray([g['pi'] for g in groups])
            source_pos = np.flatnonzero(fit.domain == source)
            source_phi = bundle.pilots.hidden('fit', source, task, standardized=True)[source_pos]
            record_access(bundle, 'RCF_DEMAND', 'fit', fit.row_id[source_pos], task,
                          purpose='normalizer', source=source, labels=False)
            metric, inverse_root, details = symmetric_metric(
                [g['H_phi_phi'] for g in groups], pi, source_phi, config)
            responses = np.stack([inverse_root @ g['score'] for g in groups])
            for g, r in zip(groups, responses):
                g['r'] = r
            if np.all(np.linalg.norm(responses, axis=1) <= config['demand']['zero_request_norm']):
                assignment, objective, status = np.empty((len(groups), 0)), None, 'NO_ESTIMABLE_REQUEST'
            else:
                assignment, objective = exhaustive_assignment(
                    responses, pi, [(g['target'], g['group']) for g in groups])
                status = 'REQUEST_ESTIMATED'
                if assignment.shape[1] == 2:
                    for k in range(2):
                        n = sum(g['n'] for i, g in enumerate(groups) if assignment[i, k] > 0)
                        p = sum(g['positive'] for i, g in enumerate(groups) if assignment[i, k] > 0)
                        if n < gc['min_rows'] or p < gc['min_positive'] or n-p < gc['min_negative']:
                            assignment = np.ones((len(groups), 1))
                            status = 'MERGED_LOW_SLOT_SUPPORT'
                            break
            norm = bundle.pilots.phi_normalizers.get((source, task), {})
            units[source, task] = dict(source=source, task=task, effective_k=assignment.shape[1],
                groups=groups, unsupported=unsupported, assignment=assignment,
                metric=metric, inverse_root=inverse_root, metric_details=details,
                phi_normalizer=norm, clustering_objective=objective, status=status,
                training_rows_hash=array_hash(view.row_id))
    identity = dict(method='RCF_DEMAND', dataset=bundle.dataset, seed=bundle.seed,
                    prefix=bundle.identity, objective='profiled_true_target_B_OOF_score',
                    units_hash=digest(units))
    return DemandResult(units=units, identity=identity)


def outer_requests(bundle, demand, role):
    """Actual frozen group-request mapping on any role; uses no outer labels.

    Training r and A remain in DemandResult. The scalar response projection is
    phi(x)^T C^{-1} b_tilde for the row's frozen (target,group); it is an input
    function, not a claimed outer conditional probability or unobserved old mu.
    """
    view = bundle.views[role]
    request = np.full((len(view.row_id), 3, 2), np.nan, dtype=np.float32)
    slot = np.full((len(view.row_id), 3, 2), -1, dtype=np.int8)
    supported = np.zeros((len(view.row_id), 3, 2), dtype=bool)
    for (source, task), unit in demand.units.items():
        if not unit.get('groups'):
            continue
        phi = bundle.pilots.hidden(role, source, task, standardized=True)
        for j, group in enumerate(unit['groups']):
            mask = (view.domain == group['target']) & (view.group[:, task] == group['group'])
            request[mask, source, task] = phi[mask] @ np.linalg.solve(unit['metric'], group['score'])
            supported[mask, source, task] = True
            if unit['effective_k']:
                slot[mask, source, task] = int(np.argmax(unit['assignment'][j]))
    return dict(row_id=view.row_id, group=view.group, request_projection=request,
                requested_slot=slot, supported_request=supported,
                definition='frozen_group_phi_times_C_inverse_profiled_training_score',
                old_eta_mu='NOT_APPLICABLE')
