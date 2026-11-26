"""Fixed diagnostics on the live trajectory's exact gradient and state identity."""
import copy
import hashlib
import heapq
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from experiments.conditional_history_shared_update_20261122.a_localization import replay
from experiments.conditional_history_shared_update_20261122.protocol import digest as legacy_digest
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import forward_logits
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.evaluate import metrics
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import (
    rng_state, restore_rng, state_digest, update_ema,
)
from .optimizers import Update, norm_matched_steps

STUDY = 'history_information_20261124'
CELLS = tuple((d, t) for d in range(3) for t in range(2))


def panels(dataset, seed, bundle):
    result = {}
    for split, x, y, ids, names in (
            ('train', bundle.train_x, bundle.train_y, bundle.train_ids, ('P_TRAIN',)),
            ('validation', bundle.val_x, bundle.val_y, bundle.val_ids, ('V_A','V_B'))):
        prefix = f'{STUDY}|panel|{dataset}|{seed}|{split}|'.encode()
        heaps, totals = {}, Counter()
        strata = (4*x[:, -1]+2*y[:, 0]+y[:, 1]).astype(np.int64)
        for idx, (row, stratum) in enumerate(zip(ids.tolist(), strata.tolist())):
            hashed = hashlib.sha256(prefix+str(row).encode()).digest()
            half = hashed[0] & 1 if len(names) == 2 else 0
            key = half, stratum
            totals[key] += 1
            pool = heaps.setdefault(key, [])
            rank = -int.from_bytes(hashed, 'big')
            if len(pool) < 512:
                heapq.heappush(pool, (rank, idx))
            elif rank > pool[0][0]:
                heapq.heapreplace(pool, (rank, idx))
        for half, name in enumerate(names):
            picks, inclusion, counts = [], [], []
            for stratum in range(12):
                pool = sorted(heaps.get((half, stratum), ()), reverse=True)
                total = totals[half, stratum]
                pi = len(pool)/total if total else None
                counts.append(dict(domain=stratum//4,joint_label=stratum%4,population=total,
                                   selected=len(pool),inclusion_probability=pi))
                picks.extend(idx for _, idx in pool)
                inclusion.extend([pi]*len(pool))
            indices = np.asarray(picks, dtype=np.int64)
            selected = ids[indices]
            result[name] = dict(x=x[indices], y=y[indices], row_ids=selected, inclusion=np.asarray(inclusion),
                                panel_hash=state_digest(selected), strata=counts, split=split,
                                population='full_train' if split=='train' else f'validation_hash_half_{half}',
                                half=half if split=='validation' else None)
    return result


def fixed_pairs(dataset, seed, panel, labels, domain, purpose, count, inclusion=None):
    result = {}
    for d,t in CELLS:
        cell = f'd{d}t{t}'
        pos = np.flatnonzero((domain==d) & (labels[:,t]==1))
        neg = np.flatnonzero((domain==d) & (labels[:,t]==0))
        if not len(pos) or not len(neg):
            result[cell] = None
            continue
        key = f'{STUDY}|pair|{dataset}|{seed}|{panel}|{cell}|{purpose}'
        gen = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
        weights = [None, None] if inclusion is None else [1/inclusion[v] for v in (pos,neg)]
        result[cell] = tuple(gen.choice(indices, count, replace=True,
                                        p=None if p is None else p/p.sum())
                             for indices,p in zip((pos,neg), weights))
    return result


def pair_scores(predictions, cell, indices):
    if indices is None:
        return None
    t = int(cell[-1])
    pos, neg = (predictions[side, t] for side in indices)
    return (pos>neg).astype(np.uint8)+(pos>=neg).astype(np.uint8)


def pair_transition(before, after):
    if before is None or after is None:
        return dict(null_reason='NO_LABEL_SUPPORT')
    return dict(corrected=int((after>before).sum()), broken=int((after<before).sum()),
                before_ties=int((before==1).sum()), after_ties=int((after==1).sum()),
                maintained_correct=int(((before==2)&(after==2)).sum()),
                lost_correct=int(((before==2)&(after<2)).sum()),
                repaired_incorrect=int(((before==0)&(after>0)).sum()),
                net=float((after.astype(float)-before).mean()/2), pair_count=len(before))


def pair_overlap(left_before, left_after, right_before, right_after):
    if any(side is None for side in (left_before,left_after,right_before,right_after)):
        return dict(null_reason='NO_LABEL_SUPPORT')
    left, right = left_after>left_before, right_after>right_before
    shared, union = int((left&right).sum()), int((left|right).sum())
    return dict(common_correction=shared, correction_union=union,
                correction_jaccard=shared/union if union else None,
                correction_jaccard_null_reason=None if union else 'EMPTY_CORRECTION_UNION',
                left_only_correction=int((left&~right).sum()), right_only_correction=int((right&~left).sum()),
                common_destruction=int(((left_after<left_before)&(right_after<right_before)).sum()))


def _norm(vectors):
    return float(torch.stack([p.double().square().sum() for p in vectors]).sum().sqrt())


def _d1_probe(pre, panel, observer, counts, directions, output, device):
    pairs = panel['d1_pairs']
    labels, domains = panel['y'], panel['x'][:,-1]
    pre.eval()
    with torch.enable_grad():
        _, logits = forward_logits(pre, torch.as_tensor(panel['x'],device=device).T)
        counts['diagnostic_probe_forwards'] += 1
        active = [cell for cell in pairs if pairs[cell] is not None]
        params = dict(pre.named_parameters())
        probe_params = [params[n] for n in observer.positions]
        for cell in pairs:
            d,t = int(cell[1]),int(cell[-1])
            mask = domains == d
            npos, nneg = int(labels[mask,t].sum()), int(mask.sum()-labels[mask,t].sum())
            mass = panel['train_domain_counts'][d]/(2*panel['train_rows'])
            identity = dict(diagnostic='D1_SOURCE', panel='P_TRAIN', split='train_probe', source_id=cell,
                            cell=cell, n_positive=npos, n_negative=nneg, train_cell_mass=mass)
            if pairs[cell] is None:
                output.append(dict(identity, null_reason='NO_LABEL_SUPPORT'))
                continue
            p,n = (torch.as_tensor(side,device=device) for side in pairs[cell])
            loss = F.softplus(-(logits[p,t]-logits[n,t])).mean()
            raw = torch.autograd.grad(loss, probe_params, retain_graph=cell!=active[-1], allow_unused=True)
            by_name = dict(zip(observer.positions,raw))
            counts['diagnostic_probe_autograd_sweeps'] += 1
            for group, items in observer.names.items():
                names = [name for _,name,_ in items]
                grad = [by_name[name] if by_name[name] is not None else torch.zeros_like(params[name]) for name in names]
                norm = _norm(grad)
                step = [directions['H0'][name].fp64_step(observer.spec.lr) for name in names]
                projected = float(torch.stack([(a.double()*b.double()).sum() for a,b in zip(grad,step)]).sum())
                def alignment(vectors):
                    radius = _norm(vectors)*norm
                    return float(torch.stack([(a.double()*b.double()).sum() for a,b in zip(grad,vectors)]).sum())/radius if radius else None
                q = [observer.vector(n,'q') for n in names]
                d = [(observer.vector(n,'vmix')/(1-.999**observer.step)).sqrt().add_(1e-8).reciprocal_()
                     for n in names]
                output.append(dict(identity, parameter_group=group, raw_l2=norm, mass_scaled_l2=mass*norm,
                                   gradient_hash=state_digest(grad), pair_hash=state_digest(panel['row_ids'][np.r_[pairs[cell][0],pairs[cell][1]]]),
                                   probe_loss=float(loss.detach()), h0_fitted_step_dot=projected,
                                   q_alignment=alignment(q),current_d_q_alignment=alignment([a*b for a,b in zip(q,d)]),
                                   u_alignment=alignment([observer.vector(n,'u0') for n in names]),null_reason=None))


@torch.no_grad()
def _directions(pre, live, optimizer, observer, packet):
    lr, lam = optimizer.config['lr'], optimizer.config['lambda_pair']
    pre_params = dict(pre.named_parameters())
    is_pair = packet['pair'] is not None
    directions = {key:{} for key in ('H0','C1_Z050','C1_Z100','C2_A050','C2_A100','M9_V0','M9_V1')}
    base = {key:{} for key in directions}
    for i, (name, actual) in enumerate(optimizer.shared):
        theta = pre_params[name]
        m = optimizer.moments[name]
        tau = m['step']
        bc = 1-.9**tau
        v0, sep = m['v'], observer.vector(name,'vsep')
        d0 = (v0/(1-.999**tau)).sqrt().add_(1e-8)
        b = m['m']/bc
        m9 = (m['m']+optimizer.shadow[name]['M9'].sum(0))/bc
        d = d0.reciprocal()
        if is_pair:
            k_bias = 1-.99**optimizer.pair_clock
            q = optimizer.history[-1][i]
            u = observer.vector(name,'u0')
        full = b.clone()
        if is_pair:
            full.add_(q,alpha=lam/k_bias)
        torch.testing.assert_close(theta.addcdiv(full,d0,value=-lr), actual, rtol=0, atol=0)
        directions['H0'][name] = Update(full,d0)
        base['H0'][name] = Update(b,d0)
        for branch,zeta in (('C1_Z050',.5),('C1_Z100',1.)):
            sep_v = v0.mul(1-zeta).add_(sep,alpha=zeta) if zeta<1 else sep
            denom = (sep_v/(1-.999**tau)).sqrt().add_(1e-8)
            directions[branch][name] = Update(full,denom)
            base[branch][name] = Update(b,denom)
        for branch,alpha in (('C2_A050',.5),('C2_A100',1.)):
            if is_pair:
                direction = b*d
                direction.add_(q*d,alpha=lam*(1-alpha)/k_bias)
                direction.add_(u,alpha=lam*alpha/k_bias)
                directions[branch][name] = Update(direction)
                base[branch][name] = Update(b*d)
            else:
                directions[branch][name] = Update(b,d0)
                base[branch][name] = directions[branch][name]
        for branch,v in (('M9_V0',v0),('M9_V1',sep)):
            denom = (v/(1-.999**tau)).sqrt().add_(1e-8)
            directions[branch][name] = Update(m9,denom)
            base[branch][name] = Update(b,denom)
    return directions, base


@torch.no_grad()
def _panel_predict(model, panel, device, count, batch):
    was = model.training
    model.eval()
    probabilities, logits = [], []
    try:
        for i in range(0,len(panel['y']),batch):
            p,z = forward_logits(model,torch.as_tensor(panel['x'][i:i+batch],device=device).T)
            probabilities.append(p.cpu().numpy())
            logits.append(z.cpu().numpy())
            count['diagnostic_panel_forwards'] += 1
    finally:
        model.train(was)
    count['diagnostic_panel_evaluations'] += 1
    count['diagnostic_panel_forward_rows'] += len(panel['y'])
    return np.concatenate(probabilities), np.concatenate(logits)


def _panel_rows(panel, preds, logits, ref, branch, readout, panel_id):
    y,x = panel['y'],panel['x']
    domain = x[:,-1]
    scores = metrics(y,preds,domain)
    records = []
    for d,t in CELLS:
        cell = f'd{d}t{t}'
        mask = domain==d
        pairs = panel['d2_pairs'][cell]
        row = dict(diagnostic='D2_PANEL', branch_id=branch, reference_branch='H0', panel=panel_id, readout=readout,
                   split='train_probe' if panel_id=='P_TRAIN' else 'validation', population=panel['population'],
                   panel_hash=panel['panel_hash'], cell=cell, count=scores[f'{cell}_count'],
                   positive=scores[f'{cell}_positive'], negative=scores[f'{cell}_negative'],
                   auc=scores[f'{cell}_auc'], logloss=scores[f'{cell}_logloss'], macro_auc=scores['macro_auc'])
        sf = pair_scores(preds,cell,pairs)
        row.update(pair_transition(pair_scores(ref,cell,pairs),sf))
        if pairs is not None:
            row['weighted_auc'] = float(roc_auc_score(y[mask,t],preds[mask,t],sample_weight=1/panel['inclusion'][mask]))
            pos,neg = pairs
            p_pos,p_neg = preds[pos,t],preds[neg,t]
            margin = logits[pos,t]-logits[neg,t]
            row.update(pair_hash=state_digest(panel['row_ids'][np.r_[pos,neg]]), mean_logit_margin=float(margin.mean()),
                       sigmoid_saturation_ties=int(((p_pos==p_neg)&((p_pos==0)|(p_pos==1))&(margin!=0)).sum()),
                       null_reason=None)
        else:
            row['weighted_auc'] = None
        records.append(row)
    return records


def attach_pairs(bank, dataset, seed, bundle):
    for name, panel in bank.items():
        d, y = panel['x'][:,-1],panel['y']
        panel['d2_pairs'] = fixed_pairs(dataset,seed,name,y,d,'D2',8192)
        if name=='P_TRAIN':
            panel['d1_pairs'] = fixed_pairs(dataset,seed,name,y,d,'D1_SOURCE_PAIR',2048,panel['inclusion'])
            panel['train_rows'] = len(bundle.train_y)
            panel['train_domain_counts'] = np.bincount(bundle.train_x[:,-1].astype(int),minlength=3)


def anchor_diagnostics(model, optimizer, observer, packet, bank, device, eval_batch):
    start_rng = rng_state(device)
    counts = Counter()
    try:
        legacy = dict(job=dict(config=optimizer.config), model=packet['online'], model_mode=packet['model_mode'],
                      ema=packet['ema'], ema_mode=packet['ema_mode'], optimizer=packet['optimizer'],
                      group_hash=legacy_digest(optimizer.parameter_manifest), raw_pair=packet['raw_pair'],
                      mixed_grad=packet['native_grad'], pair=packet['pair'], actual_post=packet['actual_post'])
        pre, live, ema_pre, ema_h0, replayed, audit = replay(model,legacy,device)
        counts['diagnostic_reference_steps'] += 1
        outputs = [dict(diagnostic='D2_REPLAY', branch_id='H0', anchor_hash=state_digest(packet), **audit)]
        directions, base = _directions(pre,live,replayed,observer,packet)
        _d1_probe(pre,bank['P_TRAIN'],observer,counts,directions,outputs,device)
        theta = dict(pre.named_parameters())
        names_by_group = {g:[n for _,n,_ in items] for g,items in observer.names.items()}
        lr = optimizer.config['lr']
        reference = directions['H0']
        reference_radii = {g:_norm([reference[n].fp64_step(lr) for n in names])
                           for g,names in names_by_group.items()}
        for group,names in names_by_group.items():
            p = dict(live.named_parameters())
            actual = [p[n].detach().double()-theta[n].detach().double() for n in names]
            ideal = [reference[n].fp64_step(lr) for n in names]
            auxiliary = [step-base['H0'][n].fp64_step(lr) for n,step in zip(names,ideal)]
            outputs.append(dict(diagnostic='D2_GEOMETRY',branch_id='H0',reference_branch='H0',parameter_group=group,
                                reference_radius=reference_radii[group],candidate_radius=reference_radii[group],matched_factor=None,
                                actual_radius=_norm(actual),rounding_residual=_norm([a-i for a,i in zip(actual,ideal)]),
                                auxiliary_step_l2=_norm(auxiliary),status='OK'))
        bases, ref = (('online',live),('fixed_ema',ema_h0)), {}
        for readout, ref_model in bases:
            for name,panel in bank.items():
                p,z = _panel_predict(ref_model,panel,device,counts,eval_batch)
                ref[readout,name] = p
                outputs.extend(_panel_rows(panel,p,z,p,'H0',readout,name))
        for key,direct in directions.items():
            if key=='H0':
                continue
            match = key.startswith(('C1_','C2_'))
            for matched in (False,True) if match else (False,):
                branch_id = f'M_{key}' if matched else key
                branch = copy.deepcopy(live)
                target = dict(branch.named_parameters())
                with torch.no_grad():
                    values, matched_groups = norm_matched_steps(
                        [(n,theta[n],reference[n],direct[n]) for n in reference],lr) if matched else ({},{})
                    for group,names in names_by_group.items():
                        r0 = reference_radii[group]
                        r1 = _norm([direct[n].fp64_step(lr) for n in names])
                        fit = matched_groups.get(group)
                        factor = fit['scale'] if fit else None
                        status = fit['status'] if fit else 'OK'
                        wanted, actual, auxiliary = [], [], []
                        for n in names:
                            expected = reference[n].fp64_step(lr)*factor if matched else direct[n].fp64_step(lr)
                            target[n].copy_(theta[n])
                            if matched:
                                target[n].copy_(values[n])
                            else:
                                direct[n].apply_(target[n],lr)
                            wanted.append(expected)
                            actual.append(target[n].double()-theta[n].double())
                            base_direction = base['H0'][n].fp64_step(lr)*factor if matched else base[key][n].fp64_step(lr)
                            auxiliary.append(expected-base_direction)
                        outputs.append(dict(diagnostic='D2_GEOMETRY', branch_id=branch_id, reference_branch='H0', parameter_group=group,
                                            reference_radius=r0, candidate_radius=r1, matched_factor=factor,
                                            actual_radius=_norm(actual), rounding_residual=_norm([a-w for a,w in zip(actual,wanted)]),
                                            auxiliary_step_l2=_norm(auxiliary), status=status))
                    ema_branch = copy.deepcopy(ema_pre)
                    update_ema(ema_branch,branch,optimizer.config['rho'])
                counts['diagnostic_virtual_steps'] += 1
                hashes = dict(branch_model_hash=state_digest(branch.state_dict()), branch_ema_hash=state_digest(ema_branch.state_dict()))
                for readout, estimator in (('online',branch),('fixed_ema',ema_branch)):
                    for name,panel in bank.items():
                        p,z = _panel_predict(estimator,panel,device,counts,eval_batch)
                        outputs.extend(dict(row,**hashes) for row in _panel_rows(panel,p,z,ref[readout,name],branch_id,readout,name))
                del branch, ema_branch
        return outputs, dict(counts)
    finally:
        restore_rng(start_rng,device)
