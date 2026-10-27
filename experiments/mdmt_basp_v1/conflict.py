"""F-only, two-fold conditional response diagnostics and split proposals."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from .model import BASPModel
from .benefit import measure_benefits


def balanced_agreement(first, second):
    first, second = np.asarray(first), np.asarray(second)
    def score(a, b):
        values = [np.mean(b[a == k] == k) for k in (0, 1) if np.any(a == k)]
        return float(np.mean(values)) if len(values) == 2 else 0.
    return max(score(first, second), score(first, 1-second))


def ridge_fit(features, targets, weights, ridge=.01):
    """Standardization is fitted on this training fold only."""
    x, y, w = np.asarray(features, np.float64), np.asarray(targets, np.float64), np.asarray(weights, np.float64)
    w = w/max(w.sum(), 1e-30)
    mean = (x*w[:, None]).sum(0)
    std = np.sqrt((((x-mean)**2)*w[:, None]).sum(0))
    std = np.maximum(std, 1e-6)
    center = (y*w[:, None]).sum(0)
    z = (x-mean)/std
    gram = z.T @ (w[:, None]*z)
    coef = np.linalg.solve(gram + ridge*np.eye(gram.shape[0]), z.T @ (w[:, None]*(y-center)))
    return dict(mean=mean, std=std, center=center, coef=coef)


def ridge_predict(fit, features):
    return (features-fit['mean'])/fit['std'] @ fit['coef'] + fit['center']


def conditional_partition(features, responses, mass, folds, seed, cfg):
    pred = np.zeros_like(responses, dtype=np.float64)
    baseline = np.zeros_like(pred)
    for fold in (0, 1):
        train, test = folds != fold, folds == fold
        if train.sum() < 2 or not test.any() or mass[train].sum() <= 0:
            return None, dict(status='INSUFFICIENT_FOLD_SUPPORT')
        fit = ridge_fit(features[train], responses[train], mass[train], cfg['ridge'])
        pred[test] = ridge_predict(fit, features[test])
        baseline[test] = fit['center']
    denominator = float((((responses-baseline)**2)*mass[:, None]).sum())
    numerator = float((((responses-pred)**2)*mass[:, None]).sum())
    skill = 1-numerator/denominator if denominator > 1e-20 else None
    diagnostic = dict(status='NO_PREDICTABLE_PARTITION', oof_skill=skill,
                      response_variance=denominator/max(float(mass.sum()), 1e-30))
    if skill is None or skill <= cfg['minimum_prediction_skill'] or np.ptp(pred, axis=0).max(initial=0) <= 1e-10:
        return None, diagnostic
    labels = KMeans(2, init='k-means++', n_init=8, max_iter=100, tol=1e-4,
                    random_state=int(seed)).fit_predict(pred, sample_weight=mass)
    fit = ridge_fit(features, responses, mass, cfg['ridge'])
    centers = np.stack([np.average(pred[labels == k], axis=0, weights=mass[labels == k]) for k in (0, 1)])
    # Distance difference is affine in input, so it can be deployed without labels.
    difference = centers[1]-centers[0]
    weight = -2*(fit['coef'] @ difference)/fit['std']
    bias = float(np.square(centers[1]).sum()-np.square(centers[0]).sum()
                 -2*fit['center']@difference-weight@fit['mean'])
    logit = features@weight+bias
    deployed = (logit < 0).astype(int)
    agreement = balanced_agreement(labels, deployed)
    diagnostic.update(deployment_balanced_agreement=agreement,
                      covariance=np.cov(pred.T, aweights=mass).tolist(),
                      predicted_mean=pred.mean(0).tolist())
    if agreement < cfg['min_deployment_balanced_agreement']:
        diagnostic['status'] = 'UNDEPLOYABLE_PARTITION'
        return None, diagnostic
    return dict(labels=labels, weight=weight, bias=bias, prediction=pred, deployed=deployed), diagnostic


def _support(partition, mass, cfg, parent_hard=None):
    x = np.clip(partition['weight_logits'], -60, 60)
    responsibility = 1/(1+np.exp(-x))
    masses = [mass*responsibility, mass*(1-responsibility)]
    eligible = np.ones(len(mass), bool) if parent_hard is None else parent_hard
    counts = [int(np.sum((partition['labels'] == k) & eligible)) for k in (0, 1)]
    ess = [float(v.sum()**2/max(float((v*v).sum()), 1e-30)) for v in masses]
    shares = [float(v.sum()/max(float(mass.sum()), 1e-30)) for v in masses]
    valid = min(counts) >= cfg['min_child_hard_support'] and min(ess) >= cfg['min_child_ess'] and min(shares) >= cfg['min_child_parent_mass']
    return valid, dict(child_hard_support=counts, child_ess=ess, child_parent_mass=shares)


def _risk_reversal(gain, labels, folds, tasks, weights):
    profiles = []
    preferred = []
    for fold in (0, 1):
        means = []
        for k in (0, 1):
            mask = (folds == fold) & (labels == k)
            means.append(np.average(gain[mask], axis=0, weights=weights[mask]) if weights[mask].sum() > 0 else None)
        if any(value is None for value in means):
            return False, 0., []
        means = np.stack(means)
        profiles.append(means.tolist())
        with_off = np.concatenate((means, np.zeros((*means.shape[:-1], 1))), -1)
        preferred.append(with_off.argmax(-1))
    reversal = any(preferred[0][0, task] != preferred[0][1, task] and
                   preferred[0][0, task] == preferred[1][0, task] and
                   preferred[0][1, task] == preferred[1][1, task] for task in range(tasks))
    improvement = 0.
    for fold in (0, 1):
        mask = folds == fold
        total = weights[mask].sum()
        if total <= 0:
            return False, 0., profiles
        parent_mean = np.average(gain[mask], axis=0, weights=weights[mask])
        parent = np.concatenate((parent_mean, np.zeros((tasks, 1))), -1).max(-1).mean()
        children = 0.
        for k in (0, 1):
            group = mask & (labels == k)
            if weights[group].sum() > 0:
                group_mean = np.average(gain[group], axis=0, weights=weights[group])
                best = np.concatenate((group_mean, np.zeros((tasks, 1))), -1).max(-1).mean()
                children += weights[group].sum()/total*best
        improvement += (children-parent)/2
    return reversal, float(improvement), profiles


@torch.no_grad()
def source_gradient_sketch(model, output, packet, domain, y, source, parent, seed):
    """Exact per-task BCE gradient of one rank-r U, followed by fixed J."""
    tree = model.sources[source]
    leaf = tree.leaves[parent]
    phi = packet['phi'][:, source]
    hidden = F.gelu(F.linear(phi, tree.blocks[leaf].V))
    gradients = y.new_zeros(len(y), model.T, model.P*tree.rank)
    mass = y.new_zeros(len(y))
    for d, part in output.details.items():
        if d == source:
            continue
        a = model.action_maps[d].index((source, leaf))
        coefficient = torch.einsum('bk,btk->bt', part['q'], part['alpha'][..., a])
        error = part['logits'].sigmoid()-y[part['rows']]
        w = model.readers[f'd{d}s{source}']
        gradient = ((error*coefficient)[..., None, None] * w[None, :, :, None] * hidden[part['rows'], None, None])
        gradients[part['rows']] = gradient.flatten(-2)
        mass[part['rows']] = coefficient.abs().mean(-1)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    sketch = (torch.randint(0, 2, (16, model.P*tree.rank), generator=generator)*2-1).to(gradients)/4
    return (gradients @ sketch.T).flatten(1).cpu().numpy(), mass.cpu().numpy()


def propose(model, cache, folds, previous, cfg, epoch, excluded_nodes=None):
    if not isinstance(model, BASPModel) or cache is None:
        return None, dict(status='NO_PROBE', nodes=[])
    model.eval()
    measured = measure_benefits(model, cache.packet, cache.domain, cache.y)
    output = measured['output']
    features = torch.cat((cache.packet.get('h0', cache.packet['h']).detach(),
                          cache.packet.get('z0', cache.packet['z']).detach()), -1).cpu().numpy().astype(np.float64)
    allgain = measured['benefits'].cpu().numpy()
    folds = np.asarray(folds)
    domain = cache.domain.cpu().numpy()
    nodes, candidates = [], []
    oldnodes = {node['id']: node for node in (previous or {}).get('nodes', [])}
    seed = model.seed
    splitcfg = cfg['split']
    excluded_nodes = set(excluded_nodes or ())
    for kind in ('NEED_SPLIT', 'SOURCE_SPLIT'):
        if kind == 'SOURCE_SPLIT' and not model.foreign:
            continue
        for d in range(model.D):
            if kind == 'SOURCE_SPLIT' and bool(model.growth_blocked[d]):
                continue
            tree = model.needs[d] if kind == 'NEED_SPLIT' else model.sources[d]
            if len(tree.leaves) >= (cfg['model']['need_leaves_max_per_domain'] if kind == 'NEED_SPLIT' else cfg['model']['source_leaves_max_per_domain']):
                continue
            for k, leaf in enumerate(tree.leaves):
                key = f'{kind}/d{d}/{leaf}'
                if key in excluded_nodes:
                    nodes.append(dict(id=key, kind=kind, domain=d, parent=k, epoch=epoch, status='COOLDOWN'))
                    continue
                rows = domain == d if kind == 'NEED_SPLIT' else domain != d
                if kind == 'NEED_SPLIT':
                    mass = output.need_mass[cache.domain == d, k].detach().cpu().numpy()
                    na = len(model.action_maps[d])
                    response = allgain[rows, :, :na]/float(model.benefit_scale)
                    response = response.reshape(rows.sum(), -1)*np.tile(np.sqrt(model.action_weight(d).cpu().numpy()/model.T), model.T)
                else:
                    sketch, weight = source_gradient_sketch(model, output, cache.packet, cache.domain, cache.y, d, k, seed)
                    response, mass = sketch[rows], weight[rows]
                diag = dict(id=key, kind=kind, domain=d, parent=k, epoch=epoch)
                if mass.sum() <= 0 or np.count_nonzero(mass) < 2*splitcfg['min_child_hard_support']:
                    diag['status'] = 'INSUFFICIENT_SUPPORT'
                    nodes.append(diag)
                    continue
                partition, details = conditional_partition(features[rows], response, mass, folds[rows], seed, splitcfg)
                diag.update(details)
                if partition is None:
                    nodes.append(diag)
                    continue
                partition['weight_logits'] = features[rows]@partition['weight']+partition['bias']
                centers = None
                if kind == 'NEED_SPLIT':
                    mu = (cache.packet.get('h0', cache.packet['h'])[cache.domain == d] @ model.fixed_projection
                          if tree.fixed else output.details[d]['mu']).detach()
                    labels = partition['labels']
                    centers = torch.stack([mu[torch.as_tensor(labels == i, device=mu.device)].mean(0) for i in (0, 1)])
                    delta = torch.cdist(mu, centers).square() if tree.fixed else ((mu[:, None]-centers[None]).square()*model.action_weight(d)).sum((-1, -2))/model.T
                    deployed = delta.argmin(-1).cpu().numpy()
                    agreement = balanced_agreement(labels, deployed)
                    diag['deployment_balanced_agreement'] = agreement
                    if agreement < splitcfg['min_deployment_balanced_agreement']:
                        diag['status'] = 'UNDEPLOYABLE_GAIN_ROUTER'
                        nodes.append(diag)
                        continue
                    # Keep the actual distance gate's responsibility for support.
                    partition['weight_logits'] = (delta[:, 1]-delta[:, 0]).cpu().numpy()
                    partition['deployed'] = deployed
                parent_hard = output.details[d]['q'].argmax(-1).cpu().numpy() == k if kind == 'NEED_SPLIT' else None
                supported, support = _support(partition, mass, splitcfg, parent_hard)
                diag.update(support)
                if not supported:
                    diag['status'] = 'INSUFFICIENT_SUPPORT'
                    nodes.append(diag)
                    continue
                if kind == 'NEED_SPLIT':
                    gains = allgain[rows, :, :len(model.action_maps[d])]
                    reversal, improvement, profiles = _risk_reversal(gains, partition['labels'], folds[rows], model.T, mass)
                else:
                    # A column is meaningful only within its target/task action library.
                    reversal, improvement, profiles = False, 0., {}
                    for target in range(model.D):
                        if target == d:
                            continue
                        subset = domain[rows] == target
                        if not subset.any():
                            continue
                        value, gain_estimate, profile = _risk_reversal(
                            allgain[rows][subset, :, :len(model.action_maps[target])],
                            partition['labels'][subset], folds[rows][subset], model.T, mass[subset])
                        reversal = reversal or value
                        improvement += gain_estimate*mass[subset].sum()/max(mass.sum(), 1e-30)
                        profiles[str(target)] = dict(actions=model.action_maps[target], profiles=profile,
                                                      risk_reversal=value, estimated_gain=gain_estimate)
                previous_labels = oldnodes.get(key, {}).get('deployed_labels')
                temporal = balanced_agreement(partition['deployed'], previous_labels) if previous_labels is not None and len(previous_labels) == rows.sum() else None
                diag.update(temporal_balanced_agreement=temporal, panel_risk_reversal=reversal,
                            estimated_local_gain=improvement, panel_gain_profiles=profiles,
                            deployed_labels=partition['deployed'].tolist())
                stable = temporal is not None and temporal >= splitcfg['min_temporal_balanced_agreement']
                if not stable and not (splitcfg['risk_reversal_alternative_to_temporal_agreement'] and reversal):
                    diag['status'] = 'WAIT_TEMPORAL_OR_RISK_REVERSAL'
                    nodes.append(diag)
                    continue
                heterogeneity = float(np.var(partition['prediction'], axis=0).sum())
                score = heterogeneity*float(mass.mean())*max(improvement, 1e-12)
                diag.update(status='QUALIFIED', score=score)
                proposal = dict(kind=kind, domain=d, parent=k, node_id=key, epoch=epoch,
                                centers=centers, weight=torch.as_tensor(partition['weight'], device=cache.domain.device, dtype=cache.packet['h'].dtype),
                                bias=torch.as_tensor(partition['bias'], device=cache.domain.device, dtype=cache.packet['h'].dtype),
                                coverage=dict(target_domain=d if kind == 'NEED_SPLIT' else None,
                                              excluded_domain=None if kind == 'NEED_SPLIT' else d,
                                              weight=partition['weight'].tolist(), bias=partition['bias'], group=0),
                                score=score, diagnostics=diag)
                candidates.append(proposal)
                nodes.append(diag)
    proposal = max(candidates, key=lambda value: value['score']) if candidates else None
    return proposal, dict(status='PROPOSAL' if proposal else 'NO_SPLIT_PROPOSAL', epoch=epoch,
                          nodes=nodes, proposal_count=len(candidates), basis='BCE_ONLY_RISK_AND_GRADIENT')
