"""BASP losses, reader-only exploration and complete-support source projection."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .model import BASPModel, packet_values
from .benefit import logit_bce


def masked_bce(logits, y, mask=None):
    valid = torch.isfinite(y) if mask is None else mask.bool() & torch.isfinite(y)
    if not valid.any():
        return logits.sum()*0
    return F.binary_cross_entropy_with_logits(logits[valid], y[valid])


def bernoulli_kl(reference_logits, current_logits):
    p = reference_logits.detach().sigmoid()
    return F.softplus(current_logits)-p*current_logits-(F.softplus(reference_logits.detach())-p*reference_logits.detach())


def _contrast(model, parts, probe):
    positive, negative = [], []
    limit_pos = model.cfg['contrast']['max_positive_pairs']
    limit_neg = model.cfg['contrast']['max_negative_pairs']
    teacher = probe.get('teacher_prediction')
    tq = probe.get('teacher_q')
    if teacher is None or tq is None:
        model.last_contrast_diagnostic = dict(positive_pairs=0, negative_pairs=0,
            eligible_negative_group_pairs=sum(len(v) for v in probe.get('pairs', {}).values()))
        return next(model.parameters()).new_zeros(())
    for d, p in parts.items():
        if not len(p['mu']):
            continue
        oldmu = teacher[p['rows'], :, :p['mu'].shape[-1]].detach()
        hard = tq[p['rows'], :len(model.needs[d].leaves)].argmax(-1)
        weight = model.action_weight(d)
        for k in hard.unique().tolist():
            ids = (hard == k).nonzero().flatten()
            # Adjacent pairs cost O(B), and never manufacture negative labels.
            for i, j in zip(ids[:-1], ids[1:]):
                distance = ((oldmu[i]-oldmu[j]).square()*weight).sum().div(model.T).sqrt()
                if distance <= model.cfg['contrast']['positive_teacher_distance_max'] and len(positive) < limit_pos:
                    positive.append(((p['mu'][i]-p['mu'][j]).square()*weight).sum()/model.T)
        for k, j, margin in probe.get('pairs', {}).get(d, []):
            first, second = (hard == k).nonzero().flatten(), (hard == j).nonzero().flatten()
            for i0, i1 in zip(first, second):
                if len(negative) >= limit_neg:
                    break
                distance = ((p['mu'][i0]-p['mu'][i1]).square()*weight).sum().div(model.T).clamp_min(1e-20).sqrt()
                negative.append((distance.new_tensor(margin)-distance).clamp_min(0).square())
    zero = next(model.parameters()).new_zeros(())
    model.last_contrast_diagnostic = dict(positive_pairs=len(positive), negative_pairs=len(negative),
        eligible_negative_group_pairs=sum(len(v) for v in probe.get('pairs', {}).values()))
    return (torch.stack(positive).mean() if positive else zero) + (torch.stack(negative).mean() if negative else zero)


def training_losses(model, packet, domain, y, probe=None, owner=None, epoch=1, shadow_action=0):
    model.set_epoch(epoch)
    output = model(packet, domain)
    zero = output.logits.new_zeros(())
    result = dict(rec=masked_bce(output.logits, y), gain=zero, cluster=zero,
                  contrast=zero, owner=zero, anchor=zero, shadow=zero)
    if not isinstance(model, BASPModel):
        return result
    if probe is not None and model.has_gain and epoch >= 2:
        pp = probe['packet']
        pdomain = probe['domain']
        parts, _ = model.predict_components(pp, pdomain, auxiliary=True)
        gain_losses, cluster_losses = [], []
        for d, part in parts.items():
            if not len(part['mu']):
                continue
            w = model.action_weight(d)
            target = probe['target'][part['rows'], :, :part['mu'].shape[-1]].detach()
            gain_losses.append(((part['mu']-target).square()*w).sum(-1).mean())
            if model.flags['GAIN_CLUSTER'] and epoch >= 3:
                tree = model.needs[d]
                root_centers = tree.centers()
                root_mass = torch.stack([part['node_mass'][node] for node in tree.roots], -1)
                root_dist = ((part['mu'][:, None]-root_centers[None]).square()*w).sum((-1, -2))/model.T
                numerator = (root_mass*root_dist).sum(-1)
                denominator = root_mass.sum(-1)
                for parent, children in tree.splits.items():
                    centers = tree.split_centers[parent]()
                    mass = torch.stack([part['node_mass'][child] for child in children], -1)
                    dist = ((part['mu'][:, None]-centers[None]).square()*w).sum((-1, -2))/model.T
                    numerator = numerator+(mass*dist).sum(-1)
                    denominator = denominator+mass.sum(-1)
                cluster_losses.append((numerator/denominator.clamp_min(1e-30)).mean())
        if gain_losses:
            result['gain'] = torch.stack(gain_losses).mean()
        if cluster_losses:
            result['cluster'] = torch.stack(cluster_losses).mean()
        if model.flags['CONTRASTIVE'] and epoch >= model.cfg['training']['contrast_start_epoch']:
            result['contrast'] = _contrast(model, parts, probe)
    if owner is not None and epoch >= 3:
        opacket, odomain, oy = owner['packet'], owner['domain'], owner['y']
        weight = torch.as_tensor(owner.get('weights', torch.ones(model.D)/model.D), device=y.device, dtype=y.dtype)
        if weight.ndim != 1 or len(weight) != model.D:
            raise ValueError('Owner weights must be the full T0 domain proportions')
        for source in range(model.D):
            rows = odomain == source
            if not rows.any():
                continue
            part = {k: value[rows].detach() for k, value in opacket.items()}
            own = model.owner_logits(part, source)
            ref = part['z'] + F.linear(part['phi'][:, source], model.teacher_w[source], model.teacher_b[source])
            result['owner'] = result['owner'] + weight[source]*masked_bce(own, oy[rows])
            result['anchor'] = result['anchor'] + weight[source]*bernoulli_kl(ref, own).mean()
    if len(y):
        # The action clock includes all (target, source, root) entries, even OFF paths.
        actions = [(d, s, root) for d in range(model.D) for s in range(model.D) if s != d
                   for root in model.sources[s].roots]
        d, s, root = actions[int(shadow_action) % len(actions)]
        rows = domain == d
        if rows.any():
            zref = output.reference_logits[rows].detach()
            phi = packet['phi'][rows, s].detach()
            rootvalue = phi+model.sources[s].blocks[root](phi).detach()
            correction = F.linear(rootvalue.detach(), model.readers[f'd{d}s{s}'])
            result['shadow'] = masked_bce(zref+correction, y[rows])
    return result


@torch.no_grad()
def project_sources(model, owner_factory, foreign_factory, cfg=None):
    """Factories stream *all* T0/F rows; return records after projection.

    The only retained arrays are logits/labels, never model state on disk.
    Owner routing and V stay fixed while all U in one source share gamma.
    """
    if not isinstance(model, BASPModel):
        return []
    cfg = cfg or model.cfg
    protection = cfg['source_protection']
    if not callable(owner_factory):
        owner_batches = owner_factory
        owner_factory = lambda: iter(owner_batches)
    if not callable(foreign_factory):
        foreign_batches = foreign_factory
        foreign_factory = lambda: iter(foreign_batches)
    stats = [dict(reference=[], change=[], labels=[], residual_square=None, residual_n=None,
                  foreign_square={}, foreign_n={}) for _ in range(model.D)]
    for batch in owner_factory():
        packet, domain, y = batch['packet'], batch['domain'], batch['y']
        for s in range(model.D):
            rows = domain == s
            if not rows.any():
                continue
            p = {key: value[rows].detach() for key, value in packet.items()}
            own, u, values = model.owner_logits(p, s, return_values=True)
            reference = p['z']+F.linear(p['phi'][:, s], model.teacher_w[s], model.teacher_b[s])
            st = stats[s]
            st['reference'].append(reference.double().cpu().numpy())
            st['change'].append((own-reference).double().cpu().numpy())
            st['labels'].append(y[rows].double().cpu().numpy())
            sq = (values-p['phi'][:, s, None]).double().square().sum((0, 2)).cpu().numpy()
            st['residual_square'] = sq if st['residual_square'] is None else st['residual_square']+sq
            st['residual_n'] = (st['residual_n'] or 0)+len(values)*model.P
    for batch in foreign_factory():
        packet, domain = batch['packet'], batch['domain']
        for s in range(model.D):
            for d in range(model.D):
                if d == s:
                    continue
                rows = domain == d
                if not rows.any():
                    continue
                phi = packet['phi'][rows, s].detach()
                value = model.sources[s].values(phi)
                sq = (value-phi[:, None]).double().square().sum((0, 2)).cpu().numpy()
                st = stats[s]
                st['foreign_square'][d] = st['foreign_square'].get(d, np.zeros_like(sq))+sq
                st['foreign_n'][d] = st['foreign_n'].get(d, 0)+len(phi)*model.P
    records = []
    for s, st in enumerate(stats):
        if not st['reference']:
            raise ValueError('Full source owner support is required for projection')
        ref, change, y = (np.concatenate(st[key]) for key in ('reference', 'change', 'labels'))
        def ll(z):
            return (np.logaddexp(0., z)-y*z).mean(0)
        baseline = ll(ref)
        limit = baseline+protection['owner_task_ll_tolerance']
        def feasible(gamma):
            return bool(np.all(ll(ref+gamma*change) <= limit+protection['feasibility_tolerance']))
        lo, hi = 0., 1.
        if feasible(1.):
            lo = 1.
        else:
            for _ in range(protection['bisection_steps']):
                mid = (lo+hi)/2
                if feasible(mid):
                    lo = mid
                else:
                    hi = mid
        owner_rms = np.sqrt(st['residual_square']/st['residual_n'])
        foreign_rms = {d: np.sqrt(v/st['foreign_n'][d]) for d, v in st['foreign_square'].items()}
        maximum = max([float(owner_rms.max(initial=0))]+[float(v.max(initial=0)) for v in foreign_rms.values()])
        gamma_rms = min(1., protection['residual_rms_limit']/max(maximum, 1e-30))
        gamma = min(lo, gamma_rms)
        for block in model.sources[s].blocks.values():
            block.U.mul_(gamma)
        model.constraint_streak[s] = model.constraint_streak[s]+1 if gamma < protection['growth_pause_gamma_below'] else 0
        if model.constraint_streak[s] >= protection['growth_pause_consecutive_checks']:
            model.growth_blocked[s] = True
        records.append(dict(source=s, gamma=gamma, risk_gamma=lo, rms_gamma=gamma_rms,
                            baseline_logloss=baseline.tolist(), projected_logloss=ll(ref+gamma*change).tolist(),
                            owner_rms_before=owner_rms.tolist(),
                            foreign_rms_before={str(d): v.tolist() for d, v in foreign_rms.items()},
                            maximum_residual_rms_after=gamma*maximum,
                            binding='SOURCE_CONSTRAINT_BINDING' if bool(model.growth_blocked[s]) else
                            'RISK' if lo < gamma_rms else 'RMS' if gamma_rms < 1 else 'NONE'))
    return records
