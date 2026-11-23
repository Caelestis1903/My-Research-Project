"""One-step history interventions on a captured, post-forward A21 optimizer input."""
import copy
from collections import Counter

import numpy as np
import torch

from .b_observer import ALL
from .b_panels import compare, predict
from .b_virtual import interpolate, norm
from .history_controls import make_history_optimizer
from .protocol import digest, v3
from .results import CELLS

GROUPS = ('STAR_SKIP', 'EXPERT')
SOURCES = ('ALL', *CELLS)


@torch.no_grad()
def replay(template, packet, device):
    cfg = packet['job']['config']
    pre = copy.deepcopy(template)
    pre.load_state_dict(packet['model'])
    pre.train(packet['model_mode'])
    ema = copy.deepcopy(template).requires_grad_(False)
    ema.load_state_dict(packet['ema'])
    ema.train(packet['ema_mode'])
    live = copy.deepcopy(pre)
    opt = make_history_optimizer(live, cfg)
    if opt.mode != 'collapsed' or cfg['dynamic'] or digest(opt.parameter_manifest) != packet['group_hash']:
        raise ValueError('Expected the frozen static collapsed-history parameter dispatch')
    # Native Adam can reuse the input packet's CPU step tensors when loading.
    opt.load_state_dict(copy.deepcopy(packet['optimizer']))
    for name, p in live.named_parameters():
        g = packet['mixed_grad'][name]
        p.grad = g.to(device) if g is not None else None
    before_steps = parameter_steps(opt)
    pair, raw = packet['pair'], packet['raw_pair']
    opt.step(pair, None if raw is None else [g.to(device) for g in raw])
    fixed = copy.deepcopy(ema)
    v3.update_ema(fixed, live, cfg['rho'])
    actual_post = dict(model=v3.state_digest(live.state_dict()), optimizer=v3.state_digest(opt.state_dict()),
                       ema=v3.state_digest(fixed.state_dict()))
    if actual_post != packet['actual_post']:
        raise RuntimeError('Restored H step differs from the actual trajectory')
    after_steps = parameter_steps(opt)
    for n, g in packet['mixed_grad'].items():
        if after_steps[n] != before_steps[n] + int(g is not None):
            raise ArithmeticError('Unexpected parameter clock at '+n)
    old_clock = packet['optimizer']['pair_clock']
    if opt.pair_clock != old_clock + int(pair is not None):
        raise ArithmeticError('Pair clock mismatch')
    audit = dict(actual_post=actual_post, pre_model_hash=v3.state_digest(pre.state_dict()),
                 pre_optimizer_hash=v3.state_digest(packet['optimizer']), pre_ema_hash=v3.state_digest(ema.state_dict()),
                 pre_steps=before_steps, post_steps=after_steps, pair_clock_before=old_clock,
                 pair_clock_after=opt.pair_clock, native_none=[n for n,g in packet['mixed_grad'].items() if g is None],
                 native_none_hash=digest([n for n,g in packet['mixed_grad'].items() if g is None]),
                 mixed_gradient_hash=v3.state_digest(packet['mixed_grad']), raw_pair_gradient_hash=v3.state_digest(raw),
                 Q_reconstruction_max_abs=opt.reconstruction_error())
    return pre, live, ema, fixed, opt, audit


def parameter_steps(opt):
    return {n: int(opt.moments[n]['step']) if n in opt.moments else int(opt.adam.state.get(p, {}).get('step', 0))
            for n,p in opt.named}


@torch.no_grad()
def alternatives(pre, live, opt, pair):
    """Whole .9 uses the independent M9 shadow. Individual sources remove E_c - M9_c / bc."""
    start = dict(pre.named_parameters())
    lr, lam = opt.config['lr'], opt.config['lambda_pair']
    changes = {}
    largest = 0.
    for i, (n, actual) in enumerate(opt.shared):
        theta = start[n]
        m, s = opt.moments[n], opt.shadow[n]
        step = m['step']
        b9 = 1-.9**step
        denom = (m['v']/(1-.999**step)).sqrt().add_(1e-8)
        n_history = m['m']/b9
        delta_n = -s['M9']/b9
        if pair is not None:
            coeff = lam/(1-.99**opt.pair_clock)
            n_history.add_(opt.history[-1][i], alpha=coeff)
            delta_n.add_(s['Q'], alpha=coeff)
        n_shadow = (m['m']+s['M9'].sum(0))/b9
        torch.testing.assert_close(n_history-n_shadow, delta_n.sum(0), atol=2e-6, rtol=2e-4)
        ref = theta.addcdiv(n_history, denom, value=-lr)
        if not torch.equal(ref, actual):
            raise ArithmeticError('Reconstructed H parameter differs at '+n)
        shadow = theta.addcdiv(n_shadow, denom, value=-lr)
        rebuilt_shadow = theta.addcdiv(n_history-delta_n.sum(0), denom, value=-lr)
        torch.testing.assert_close(rebuilt_shadow, shadow, atol=2e-6, rtol=2e-4)
        largest = max(largest, float((rebuilt_shadow-shadow).abs().max()))
        changes[n] = {'ALL':shadow, **{cell:theta.addcdiv(n_history-delta_n[c], denom, value=-lr)
                                       for c,cell in enumerate(CELLS)}}
    return changes, largest


@torch.no_grad()
def group_cosine(pre, ref, alt, names, nr, na):
    if nr == 0 or na == 0:
        return None
    p,r,a = (dict(m.named_parameters()) for m in (pre, ref, alt))
    dots = [((r[n].double()-p[n].double())*(a[n].double()-p[n].double())).sum() for n in names]
    return float(torch.stack(dots).sum()/(nr*na))


def match_factor(ref_norm, alt_norm):
    if alt_norm:
        return ref_norm/alt_norm
    return 1. if ref_norm == 0 else None


def reliability(panels):
    # Pair sampling and the two row-disjoint folds provide separate readout checks.
    values = {p:np.array([c['population']['delta_auc'] for c in bank['cells']]) for p,bank in panels.items()}
    return dict(validation_fold_same_sign=(np.sign(values['V_A'])==np.sign(values['V_B'])).tolist(),
                train_fold_same_sign=(np.sign(values['P_A'])==np.sign(values['P_B'])).tolist(),
                fixed_pair_minus_population={p:[c['sampled']['delta_auc']-c['population']['delta_auc']
                                                for c in bank['cells']] for p,bank in panels.items()},
                max_transition_closure_error=max(c[k]['closure_error'] for bank in panels.values()
                                                 for c in bank['cells'] for k in ('population','sampled')))


@torch.no_grad()
def observe(template, panels, packet, device):
    pre, live, ema, fixed, opt, audit = replay(template, packet, device)
    changes, audit['shadow_update_reconstruction_max_abs'] = alternatives(pre, live, opt, packet['pair'])
    groups = {g:[n for n in changes if n.startswith('expert.') == (g=='EXPERT')] for g in GROUPS}
    if any(not names for names in groups.values()):
        raise ValueError('A history parameter group is empty')
    counts = Counter(panel_forwards=0, panel_forward_rows=0, replayed_optimizer_steps=1,
                     scientific_training_steps=0, autograd_sweeps=0, test_evaluations=0)
    ref = {'online':predict(live, panels, ALL, counts), 'fixed_ema':predict(fixed, panels, ALL, counts)}
    rows = []
    def measure(branch, group, source, scale, ref_norm, alt_norm, cosine, factor):
        names = groups[group] if group in groups else list(changes)
        row = dict(group=group, source=source, scale=scale, reference='H_ACTUAL', ref_update_norm=ref_norm,
                   raw_alt_update_norm=alt_norm, raw_update_cosine=cosine, scale_factor=factor)
        row['action'] = ('H_SHADOW' if group=='S_DENSE' else
                         'whole_group_to_adam' if source=='ALL' else 'single_source_to_adam')
        if factor is None:
            rows.append(dict(row,status='UNMATCHABLE_ZERO_ALTERNATIVE'))
            return
        branch_f = copy.deepcopy(ema)
        v3.update_ema(branch_f, branch, opt.config['rho'])
        actual_norm = norm(pre, branch, names)
        row.update(status='OK', applied_update_norm=actual_norm,
                   model_hash=v3.state_digest(branch.state_dict()), ema_hash=v3.state_digest(branch_f.state_dict()))
        if scale=='norm_matched':
            row['matched_norm_error'] = abs(actual_norm-ref_norm)
            # One rounding per stored FP32 parameter bounds achievable group-L2 precision.
            precision = torch.finfo(next(branch.parameters()).dtype).eps
            p = dict(pre.named_parameters())
            theta_norm = float(torch.stack([p[n].double().square().sum() for n in names]).sum().sqrt())
            if row['matched_norm_error']>4*precision*(theta_norm+ref_norm):
                raise ArithmeticError('Group norm matching exceeds floating-point storage precision')
        row['reliability'] = {}
        for readout, model in [('online',branch), ('fixed_ema',branch_f)]:
            bank = predict(model, panels, ALL, counts)
            row[readout] = compare(panels, ref[readout], bank)
            row['reliability'][readout] = reliability(row[readout])
        rows.append(row)

    for group, names in [*groups.items(), ('S_DENSE',list(changes))]:
        nr = norm(pre, live, names)
        audit.setdefault('mixed_gradient_l2',{})[group] = float(torch.stack([
            packet['mixed_grad'][n].double().square().sum() for n in names]).sum().sqrt())
        for source in SOURCES if group in GROUPS else ('ALL',):
            alt = copy.deepcopy(live)
            params = dict(alt.named_parameters())
            for n in names:
                params[n].copy_(changes[n][source])
            na = norm(pre, alt, names)
            cosine = group_cosine(pre,live,alt,names,nr,na)
            measure(alt, group, source, 'raw', nr, na, cosine, 1.)
            if group in GROUPS:
                factor = match_factor(nr, na)
                if factor is not None and factor != 1:
                    interpolate(alt, pre, alt, factor, names)
                measure(alt, group, source, 'norm_matched', nr, na, cosine, factor)
    return dict(audit=audit, effects=rows, counts=dict(counts))


def gate_rows(record, source_ref):
    """Use the fixed independent V_B population AUC for the preregistered whole-group gate."""
    output = []
    for effect in record['effects']:
        if effect['source']!='ALL' or effect['group'] not in GROUPS:
            continue
        row = {k:record[k] for k in ('dataset','seed','arm','history_form','anchor_epoch')}
        row.update(group=effect['group'],scale=effect['scale'],split='validation',readout='fixed_ema',
                   panel='V_B',action='whole_group_to_adam',status=effect['status'],source_ref=source_ref)
        if row['status']=='OK':
            scored = effect['fixed_ema']['V_B']
            row.update({CELLS[c['q']]+'_delta_auc':c['population']['delta_auc'] for c in scored['cells']})
            row['macro_delta_auc'] = scored['macro_auc_delta']
        output.append(row)
    return output
