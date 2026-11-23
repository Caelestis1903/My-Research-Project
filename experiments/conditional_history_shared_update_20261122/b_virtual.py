"""Frozen native Adam interventions. Source groups are parameter-coordinate identities."""
import copy
import hashlib
from collections import Counter

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.optimizers import shared_name
from experiments.conditional_information_momentum_20261102_v3.protocol import state_digest


def registry(model, bundle):
    domain_name = f"embedding.embed_dict.{bundle.features.columns.index('domain_indicator')}.weight"
    offset = 0
    counts = Counter()
    groups, records = {}, []
    for name, p in model.named_parameters():
        root, *tail = name.split('.')
        if name == domain_name:
            if tuple(p.shape) != (3, 16):
                raise ValueError('Domain embedding shape changed')
            parts = [(f'D{d}', 16) for d in range(3)]
        elif shared_name(name):
            parts = [('EXPERT' if root == 'expert' else 'STAR_SKIP', p.numel())]
        elif root == 'embedding':
            parts = [('EMBED', p.numel())]
        elif root in ('slot_weight', 'slot_bias', 'domain_expert'):
            parts = [('D'+tail[0], p.numel())]
        elif root == 'task_expert':
            parts = [('T'+tail[0], p.numel())]
        elif root in ('tower', 'gate'):
            i = int(tail[0])
            parts = [(f'DT{i%3}{i//3}', p.numel())]
        else:
            raise ValueError('Unmapped trainable name '+name)
        for group, size in parts:
            scope = 'S_DENSE' if shared_name(name) else 'EMBED' if group=='EMBED' else 'N'
            groups[name] = scope
            records.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype),
                                group=group, scope=scope, start=offset, end=offset+size))
            counts[group] += size
            offset += size
    ml = bundle.dataset == 'movielens'
    if (domain_name not in groups or offset != (773470 if ml else 1614254)
        or sum(counts[k] for k in ('EMBED','STAR_SKIP','EXPERT')) != (483152 if ml else 661920)
        or any(counts[f'D{d}'] != (74960 if ml else 295632) for d in range(3))
        or any(counts[f'T{t}'] != 16576 for t in range(2))
        or any(counts[f'DT{d}{t}'] != 5381 for d in range(3) for t in range(2))):
        raise ValueError('Frozen graph coordinates changed')
    return groups, dict(records=records, counts=dict(counts), group_hash=state_digest(records))


def partition(labels, domain, context):
    y = labels.detach().cpu().numpy().astype(np.int64).ravel()
    truth = (2*domain.detach().cpu().numpy().astype(np.int64)[:,None]+np.arange(2)).ravel()
    capacities = np.bincount(2*truth+y, minlength=12).reshape(6,2)
    masks = {'true': truth}
    for partition_id in range(2):
        key = f"conditional_history_shared_update_20261122|{context['dataset']}|{context['seed']}|{context['global_step']}|{partition_id}"
        generator = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
        member = np.full_like(truth,-1)
        for label in (0,1):
            entries = generator.permutation(np.flatnonzero(y == label))
            member[entries] = np.repeat(np.arange(6), capacities[:,label])
        if not np.array_equal(np.bincount(2*member+y,minlength=12).reshape(6,2), capacities):
            raise AssertionError('Pseudo loss-entry membership failed')
        masks['pseudo'+str(partition_id)] = member
    return masks, dict(capacities=capacities.tolist(), membership_hashes={k:state_digest(v) for k,v in masks.items()})


def decompose(model, real, labels, domains, context):
    masks, receipt = partition(labels, domains, context)
    pars = tuple(model.parameters())
    gradients, values = {}, {}
    for family, member in masks.items():
        member = torch.as_tensor(member, device=real.device)
        gradients[family], values[family] = [], []
        for q in range(6):
            loss = real.flatten()[member==q].sum()/real.numel()
            gs = torch.autograd.grad(loss, pars, allow_unused=True, retain_graph=True)
            gradients[family].append(gs)
            values[family].append(float(loss.detach()))
    receipt['losses'] = values
    return gradients, receipt


def native(opt):
    adam = getattr(opt,'adam',opt)
    if not isinstance(adam,torch.optim.Adam) or getattr(opt,'mode','adam') != 'adam':
        raise TypeError('B_CORE requires an actual clean native Adam')
    return adam


def virtual_step(model, optimizer, scopes, scope=None, mode='JOINT', gs=None):
    opt = native(optimizer)
    branch = copy.deepcopy(model)
    named = dict(model.named_parameters())
    bnamed = dict(branch.named_parameters())
    lookup = {id(p): bnamed[n] for n,p in named.items()}
    param_groups = [dict(params=[lookup[id(p)] for p in g['params']]) for g in opt.param_groups]
    clone = torch.optim.Adam(param_groups, **opt.defaults)
    clone.load_state_dict(copy.deepcopy(opt.state_dict()))
    for i,(n,p) in enumerate(named.items()):
        g = p.grad
        if g is not None and scope==scopes[n]:
            src = gs[i] if gs is not None else None
            if mode == 'ZERO_DATA' or (mode == 'ONLY' and src is None):
                g = torch.zeros_like(g)
            elif mode == 'ONLY':
                g = src
            elif mode == 'LOO' and src is not None:
                g = g-src
        bnamed[n].grad = None if g is None else g.detach().clone()
    clone.step()
    return branch, clone


def gradient_audit(model, gradients, scopes, losses, target):
    named = list(model.named_parameters())
    receipt = {}
    for family, sources in gradients.items():
        errors, refs = [], []
        norms = {k: [] for k in ('EMBED','S_DENSE','N','STAR_SKIP','EXPERT')}
        for q in range(6):
            terms = {k: [] for k in norms}
            for (name,p),g in zip(named, sources[q]):
                if g is None:
                    continue
                energy = g.detach().double().square().sum()
                terms[scopes[name]].append(energy)
                if scopes[name]=='S_DENSE':
                    terms['EXPERT' if name.startswith('expert.') else 'STAR_SKIP'].append(energy)
            for k,vals in terms.items():
                norms[k].append(float(torch.stack(vals).sum().sqrt()) if vals else 0.)
        for i,(_,p) in enumerate(named):
            terms = [s[i] for s in sources if s[i] is not None]
            if not terms:
                if p.grad is not None:
                    errors.append(p.grad.abs().max())
                continue
            g = sum(terms[1:], terms[0])
            ref = p.grad if p.grad is not None else torch.zeros_like(g)
            errors.append((g-ref).abs().max())
            refs.append(ref.abs().max())
        err = float(torch.stack(errors).max()) if errors else 0.
        mx = float(torch.stack(refs).max()) if refs else 0.
        loss_error = abs(sum(losses[family])-target)
        if err > 2e-5+2e-4*mx or loss_error > 1e-6+1e-5*abs(target):
            raise ArithmeticError(f'{family} source-sum residual {err}, loss {loss_error}')
        receipt[family] = dict(max_gradient_error=err, max_gradient_reference=mx,
                               loss_sum_error=loss_error, gradient_norms=norms)
    return receipt


def steps(model, opt):
    return {n: int(opt.state.get(p, {}).get('step',0)) for n,p in model.named_parameters()}


def norm(base, changed, names):
    ref, alt = dict(base.named_parameters()),dict(changed.named_parameters())
    return float(torch.stack([(alt[n].detach().double()-ref[n].detach().double()).square().sum()
                              for n in names]).sum().sqrt())


@torch.no_grad()
def interpolate(out, origin, other, scale, names):
    # Original buffers and all coordinates outside the destination remain intact.
    dest, ref, alt = dict(out.named_parameters()),dict(origin.named_parameters()),dict(other.named_parameters())
    for n in names:
        dest[n].copy_(ref[n]+(alt[n]-ref[n])*scale)
