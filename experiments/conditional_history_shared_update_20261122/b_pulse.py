"""One gated pulse from a committed E20/E35 RAM packet. No new first-step Adam."""
import copy
import time

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.protocol import (
    rng_state, restore_rng, state_digest)
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.objectives import train_step
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import make_optimizer
from .b_observer import CONFIRM, displaced, ema_residual, ensemble_overlap, write_result
from .b_panels import advance_average, compare, predict, window_banks
from .b_virtual import norm


def run_pulse(observer, model_template, config, epoch, family, *, gate):
    """Run one admitted slot in a supervised worker, restoring caller RNG on exit."""
    if (gate.get('dataset')!=observer.dataset or gate.get('B_PULSE')!='TRIGGERED'
            or family not in ('true','pseudo0') or epoch not in (20,35) or config['arm']!='A0'):
        raise ValueError('Pulse slot has no matching B_PULSE gate')
    win = observer.ram['windows'][str(epoch)]
    K = observer.K
    if len(win['rows'])!=K or 'terminal_models' not in win:
        raise ValueError('Pulse requires the complete parent window and selected post')
    branch = win['branches'][family]
    device = next(model_template.parameters()).device
    saved = rng_state(device)
    try:
        return _run(observer,model_template,config,win,K,branch,epoch,family,device)
    finally:
        restore_rng(saved,device)


def _run(observer,template,c,win,K,branch,epoch,family,device):
    tick = time.perf_counter()
    panels = observer.panels
    marks = (1,(K+1)//2,K)
    counts = dict(counterfactual_suffix_adam=0,panel_forwards=0,panel_forward_rows=0,
                  logical_steps=K,first_step_reused=1,branch_step_panel_reads=0)
    report = dict(epoch=epoch,dataset=observer.dataset,seed=observer.seed,
                  family=family,q=branch['q'],counts=counts,milestones={},
                  first_row_hash=win['first_step_hash'],start_step=win['start_step'],
                  parent_panel_identity=observer.ram['panels'])
    if branch['q'] is None:
        report.update(status='COMPLETE/NO_ACTION',parent_window_reused=True,
                      delta_ema_recurrence_residual_upper=[0.]*K,
                      terminal_common_radius=dict(status='NOT_INFORMATIVE',radius=0.))
        for j in marks:
            report['milestones'][str(j)] = {k:compare(panels,v,v) for k,v in win['parents'][str(j)].items()}
        end = win['parents'][str(K)]
        banks = win['ensembles']
    else:
        model = copy.deepcopy(template)
        model.load_state_dict(branch['model'])
        model.train(branch['mode'])
        ema = copy.deepcopy(model).requires_grad_(False)
        ema.load_state_dict(branch['ema'])
        opt = make_optimizer(model,c)
        opt.load_state_dict(copy.deepcopy(branch['adam']))
        restore_rng(win['start_rng'],device)
        loader = torch.Generator()
        loader.set_state(win['loader'])
        order = torch.randperm(len(observer.bundle.train_y),generator=loader).numpy()
        avg = copy.deepcopy(win['init_scores'])
        # The parent/branch post-F each used the identical native FP32 update.
        # The parent and branch individual FP64 residual norms bound the residual of their difference.
        residuals = [win['residuals'][0]+branch['first_ema_residual']]
        for j in range(1,K+1):
            if j==1:
                p,f = branch['scores'],branch['ema_scores']
            else:
                offset = (j-2)*40960
                ids = order[offset:offset+40960]
                bundle = observer.bundle
                row_ids = bundle.train_row_ids[ids]
                if state_digest(row_ids)!=win['rows'][j-1]:
                    raise ValueError('Pulse suffix order differs from parent')
                prev = {n:p.detach().clone() for n,p in ema.named_parameters()}
                x = torch.as_tensor(bundle.train_x[ids],device=device).T
                y = torch.as_tensor(bundle.train_y[ids],device=device)
                context = dict(dataset=observer.dataset,seed=observer.seed,epoch=epoch+1,
                               global_step=win['start_step']+j-1,split_id='original')
                train_step(model,opt,ema,x,y,row_ids,c,context)
                counts['counterfactual_suffix_adam'] += 1
                p = predict(model,panels,CONFIRM,counts)
                if j in marks:
                    f = predict(ema,panels,CONFIRM,counts)
                residuals.append(ema_residual(prev,model,ema,c['rho'])+win['residuals'][j-1])
            if state_digest(rng_state(device))!=win['step_rng'][j-1]:
                raise RuntimeError('Pulse stochastic stream differs from parent')
            counts['branch_step_panel_reads'] += 1
            advance_average(avg,p,c['rho'])
            if j in marks:
                ref = win['parents'][str(j)]
                report['milestones'][str(j)] = dict(online=compare(panels,ref['online'],p),
                    fixed_ema=compare(panels,ref['fixed_ema'],f))
        end = dict(online=p,fixed_ema=f)
        banks = window_banks(avg)
        parent = copy.deepcopy(model)
        parent.load_state_dict(win['terminal_models']['online'])
        parent_f = copy.deepcopy(ema)
        parent_f.load_state_dict(win['terminal_models']['fixed_ema'])
        names = list(dict(parent.named_parameters()))
        rn,rf = norm(parent,model,names),norm(parent_f,ema,names)
        r = min(rn,rf)
        matched = dict(radius=r,online_norm=rn,ema_norm=rf,background='JOINT_ONLINE')
        if r:
            matched['status'] = 'COMPLETE'
            for name,old,new,scale in [('online',parent,model,r/rn),('fixed_ema',parent_f,ema,r/rf)]:
                tmp = displaced(parent,old,new,scale,names)
                matched[name] = compare(panels,win['parents'][str(K)]['online'],predict(tmp,panels,CONFIRM,counts))
        else:
            matched['status'] = 'NOT_INFORMATIVE'
        report.update(status='COMPLETE',delta_ema_recurrence_residual_upper=residuals,terminal_common_radius=matched)
    report['function_vs_parameter'] = {k:compare(panels,end['fixed_ema'],v) for k,v in banks.items()}
    report['function_vs_parent'] = {k:compare(panels,win['ensembles'][k],v) for k,v in banks.items()}
    report['pair_overlap'] = ensemble_overlap(panels,end['online'],end['fixed_ema'],banks)
    report['worker_seconds'] = time.perf_counter()-tick
    write_result(observer.out/f'B_PULSE_E{epoch}_{family}.json',report)
    return report
