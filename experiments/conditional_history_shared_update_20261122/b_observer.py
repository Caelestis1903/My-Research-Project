"""B_CORE attachment to the clean worker. model/Adam/F remain owned by the caller."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.protocol import (
    cpu_copy, rng_state, state_digest, update_ema)
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.objectives import losses, train_step
from .b_panels import advance_average, compare, predict, prepare, selection_score, window_banks
from .b_virtual import decompose, gradient_audit, interpolate, native, norm, registry, steps, virtual_step

ALL = ('P_A', 'P_B', 'V_A', 'V_B')
CONFIRM = ('P_B', 'V_B')


def write_result(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(obj,stream,separators=(',',':'),allow_nan=False)
        stream.write('\n')
    os.replace(tmp,path)


def identity(model, opt, ema):
    return dict(model=state_digest(model.state_dict()), adam=state_digest(native(opt).state_dict()),
                ema=state_digest(ema.state_dict()), modes=[m.training for m in model.modules()])


def _rng_loader(device,loader):
    return state_digest(rng_state(device)),state_digest(loader.get_state())


def _save_post(branch, opt, ema, scores, ema_scores, q):
    return dict(q=q,model=cpu_copy(branch.state_dict()), adam=cpu_copy(opt.state_dict()),
        ema=cpu_copy(ema.state_dict()), mode=branch.training,
        scores={k:scores[k] for k in CONFIRM}, ema_scores={k:ema_scores[k] for k in CONFIRM})


@torch.no_grad()
def displaced(post, reference, changed, scale, names):
    out = copy.deepcopy(post)
    base, alt, target = dict(reference.named_parameters()),dict(changed.named_parameters()),dict(out.named_parameters())
    for n in names:
        target[n].add_(alt[n]-base[n],alpha=scale)
    return out


def _menu():
    for scope in ('EMBED','S_DENSE','N'):
        yield 'true', scope, 'ZERO_DATA', None
        for mode in ('ONLY','LOO'):
            for q in range(6):
                yield 'true',scope,mode,q
    for family in ('pseudo0','pseudo1'):
        for q in range(6):
            yield family,'EMBED','LOO',q


class BObserver:
    def __init__(self,bundle,dataset,seed,output_dir):
        if dataset != bundle.dataset or seed not in range(2022,2027):
            raise ValueError('Unregistered B replay identity')
        self.bundle,self.dataset,self.seed = bundle,dataset,seed
        self.out = Path(output_dir)
        self.K = {'movielens':20,'kuairand':49}[dataset]
        self.panels,self.panel_manifest = prepare(bundle,seed)
        self.scopes = None
        self.ram = dict(dataset=dataset,seed=seed,panels=self.panel_manifest['source_hashes'],
                        anchors=[],windows={},active=None)
        write_result(self.out/'B_PANEL_MANIFEST.json',self.panel_manifest)

    def state_dict(self):
        return self.ram

    def load_state_dict(self,state):
        if any(self.ram[k] != state[k] for k in ('dataset','seed','panels')):
            raise ValueError('B RAM packet identity mismatch')
        self.ram = state

    def step(self,model,optimizer,ema,x,y,row_ids,config,context,loader):
        if (config['arm']!='A0' or config['dataset']!=self.dataset or context['seed']!=self.seed
                or context['dataset']!=self.dataset):
            raise ValueError('Observer attached to a different clean trajectory')
        epoch,step = context['epoch'],context['global_step']
        anchor = epoch in (5,20,35,50) and step==epoch*self.K
        prev = None
        if self.ram['active'] is not None:
            prev = {n:p.detach().clone() for n,p in ema.named_parameters()}
        if anchor:
            expected = (21927 if self.dataset=='movielens' else 40052)
            if len(row_ids)!=expected or len(y)!=expected:
                raise ValueError('Anchor must use the real final batch')
            if self.scopes is None:
                self.scopes, record = registry(model,self.bundle)
                self.group_hash = record['group_hash']
                write_result(self.out/'B_GROUP_MANIFEST.json',record)
            if epoch in (20,35):
                prev = {n:p.detach().clone() for n,p in ema.named_parameters()}
            value = self._anchor(model,optimizer,ema,x,y,row_ids,config,context,loader)
        else:
            value = train_step(model,optimizer,ema,x,y,row_ids,config,context)
        if self.ram['active'] is not None:
            rng_before = _rng_loader(x.device,loader)
            self._window_step(model,ema,row_ids,config,context,prev,value['counts'])
            if _rng_loader(x.device,loader)!=rng_before:
                raise RuntimeError('Window read changed RNG or loader state')
        return value

    def _anchor(self,model,optimizer,ema,x,y,row_ids,c,context,loader):
        tick = time.perf_counter()
        epoch,step = context['epoch'], context['global_step']
        pulse = epoch in (20,35)
        optimizer.zero_grad(set_to_none=True)
        scalar,source,diagnostic,work = losses(model,ema,x,y,row_ids,c,context)
        if not bool(torch.isfinite(scalar)):
            raise FloatingPointError('B anchor nonfinite loss')
        rng_before = _rng_loader(x.device,loader)
        gradients,parts = decompose(model,source['real'],y,x[-1],context)
        scalar.backward()
        adam = native(optimizer)
        pre = identity(model,adam,ema)
        pre_steps = steps(model,adam)
        parts['sum_audit'] = gradient_audit(model,gradients,self.scopes,parts['losses'],float(scalar.detach()))
        none = [n for n,p in model.named_parameters() if p.grad is None]
        zero = [n for n,p in model.named_parameters() if p.grad is not None and not bool(p.grad.any())]
        joint,jopt = virtual_step(model,adam,self.scopes)
        jf = copy.deepcopy(ema)
        update_ema(jf,joint,c['rho'])
        expected = identity(joint,jopt,jf)
        post_steps = steps(joint,jopt)
        counts = dict(panel_forwards=0,panel_forward_rows=0,virtual_adam=1,source_sweeps=18)
        jbank = predict(joint,self.panels,ALL,counts)
        fbank = predict(jf,self.panels,ALL,counts)
        rows = []
        selected = {s:dict(q=None,score=0.) for s in gradients}
        candidates = {}
        if pulse:
            initial = predict(ema,self.panels,CONFIRM,counts)
        del jopt
        true_embed = {}
        for family,scope,mode,q in _menu():
            names = [n for n,s in self.scopes.items() if s==scope]
            gs = None if q is None else gradients[family][q]
            branch,altopt = virtual_step(model,adam,self.scopes,scope,mode,gs)
            bf = copy.deepcopy(ema)
            update_ema(bf,branch,c['rho'])
            counts['virtual_adam'] += 1
            obank = predict(branch,self.panels,ALL,counts)
            ebank = predict(bf,self.panels,ALL,counts)
            row = dict(family=family,scope=scope,mode=mode,q=q,
                relative_to='ZERO_DATA' if mode=='ONLY' else 'JOINT')
            if mode=='ZERO_DATA':
                zero_o,zero_f = obank,ebank
            row['online'] = compare(self.panels,zero_o if mode=='ONLY' else jbank,obank)
            row['fixed_ema'] = compare(self.panels,zero_f if mode=='ONLY' else fbank,ebank)
            row['applied_update_norm'] = norm(model,branch,names)
            if scope=='S_DENSE':
                row['group_update_norm'] = {k:norm(model,branch,[n for n in names if n.startswith('expert.')==expert])
                                            for k,expert in [('STAR_SKIP',False),('EXPERT',True)]}
            if mode=='LOO':
                with torch.no_grad():
                    bp,jp,fe,fj = map(lambda m:dict(m.named_parameters()),(branch,joint,bf,jf))
                    row['one_step_F_rounding_max'] = float(torch.stack([
                        ((fe[n].double()-fj[n].double())-
                         (1-c['rho'])*(bp[n].double()-jp[n].double())).abs().max() for n in names]).max())
                self._controlled(row,model,joint,jf,branch,jbank,fbank,counts,c['rho'],names)
                if scope=='EMBED':
                    # Keep just the six true EMBED displacements while matching each pseudo q.
                    truth = true_embed.get(q)
                    if family=='true':
                        true_embed[q] = {n:p.detach().cpu().clone() for n,p in branch.named_parameters() if n in names}
                    elif truth is not None:
                        other = copy.deepcopy(joint)
                        for n,p in other.named_parameters():
                            if n in truth:
                                p.data.copy_(truth[n])
                        row['true_pseudo_matched'] = self._matched(model,joint,other,branch,names,counts)
                        del other
                    score = selection_score(row['online'])
                    if score > selected[family]['score']:
                        selected[family] = dict(q=q,score=score)
                        if pulse and family in ('true','pseudo0'):
                            candidates[family] = _save_post(branch,altopt,bf,obank,ebank,q)
                            candidates[family]['first_ema_residual'] = ema_residual(dict(ema.named_parameters()),branch,bf,c['rho'])
            rows.append(row)
            del branch,altopt,bf
        if counts['virtual_adam']!=52 or _rng_loader(x.device,loader)!=rng_before or identity(model,adam,ema)!=pre:
            raise RuntimeError('Anchor count or diagnostic isolation failed')
        optimizer.step()
        update_ema(ema,model,c['rho'])
        actual = identity(model,adam,ema)
        if actual!=expected or steps(model,adam)!=post_steps:
            raise RuntimeError('Virtual JOINT differs from the live Adam/EMA commit')
        if pulse:
            for family in ('true','pseudo0'):
                if family not in candidates:
                    candidates[family] = dict(q=None)
            self.ram['active'] = str(epoch)
            self.ram['windows'][str(epoch)] = dict(start_step=step,epoch=epoch,
                init_scores=initial,parents={},branches=candidates,rows=[],residuals=[],step_rng=[],
                start_rng=cpu_copy(rng_state(x.device)),loader=loader.get_state().clone(),
                first_step_hash=state_digest(np.asarray(row_ids)),rho=c['rho'],
                avg=copy.deepcopy(initial))
        anchor_id = dict(dataset=self.dataset,seed=self.seed,epoch=epoch,global_step=step,
            batch=self.K,batch_offset=(self.K-1)*40960,B=len(y),row_hash=state_digest(np.asarray(row_ids)),
            panel_hashes=self.ram['panels'],group_hash=self.group_hash,
            rng_loader_hash=list(rng_before),pre=pre,virtual_joint=expected,actual_joint=actual,
            native_none=none,native_tensor_zero=zero,pre_steps=pre_steps,post_steps=post_steps,
            pre_steps_hash=state_digest(pre_steps),post_steps_hash=state_digest(post_steps))
        record = dict(identity=anchor_id,partitions=parts,selected=selected,effects=rows,
                      counts=counts,observer_seconds=time.perf_counter()-tick)
        write_result(self.out/f'B_ANCHOR_E{epoch:02d}.json',record)
        self.ram['anchors'].append(anchor_id)
        work.update(counts)
        work['B_anchor_seconds'] = record['observer_seconds']
        return dict(loss=float(scalar.detach()),real_loss=float(source['real'].detach().mean()),
                    pred=source['pred'].detach(),diagnostics=diagnostic,counts=work)

    def _matched(self,pre,joint,left,right,names,counts):
        ln,rn = norm(pre,left,names),norm(pre,right,names)
        radius = min(ln,rn)
        out = dict(radius=radius,left_norm=ln,right_norm=rn)
        if radius==0:
            return dict(out,status='NOT_INFORMATIVE')
        tmp = copy.deepcopy(joint)
        interpolate(tmp,pre,left,radius/ln,names)
        a = predict(tmp,self.panels,ALL,counts)
        interpolate(tmp,pre,right,radius/rn,names)
        b = predict(tmp,self.panels,ALL,counts)
        return dict(out,status='COMPLETE',effects=compare(self.panels,a,b))

    def _controlled(self,row,pre,joint,jf,branch,jbank,fbank,counts,rho,names):
        row['matched'] = self._matched(pre,joint,joint,branch,names,counts)
        for key,background,ref,scale in [('online_small',joint,jbank,1-rho),
                                         ('F_same_small',jf,fbank,1-rho),
                                         ('F_full_delta',jf,fbank,1.)]:
            tmp = displaced(background,joint,branch,scale,names)
            row[key] = compare(self.panels,ref,predict(tmp,self.panels,ALL,counts))
            del tmp

    def _window_step(self,model,ema,row_ids,c,context,prev,cost):
        tick = time.perf_counter()
        key = self.ram['active']
        win = self.ram['windows'][key]
        j = context['global_step']-win['start_step']+1
        if j!=len(win['rows'])+1:
            raise RuntimeError('Noncontiguous B parent window')
        score_cost = dict(panel_forwards=0,panel_forward_rows=0)
        p = predict(model,self.panels,CONFIRM,score_cost)
        advance_average(win['avg'],p,c['rho'])
        win['rows'].append(state_digest(np.asarray(row_ids)))
        win['step_rng'].append(state_digest(rng_state(next(model.parameters()).device)))
        win['residuals'].append(ema_residual(prev,model,ema,c['rho']))
        if j in (1,(self.K+1)//2,self.K):
            f = predict(ema,self.panels,CONFIRM,score_cost)
            win['parents'][str(j)] = dict(online=p,fixed_ema=f)
        if j==self.K:
            banks = window_banks(win.pop('avg'))
            win['terminal_models'] = dict(online=cpu_copy(model.state_dict()),fixed_ema=cpu_copy(ema.state_dict()))
            win['ensembles'] = banks
            summary = {k:compare(self.panels,f,b) for k,b in banks.items()}
            write_result(self.out/f'B_WINDOW_E{key}.json',dict(reference='F_K',start='F_pre',
                start_step=win['start_step'],K=j,rho=c['rho'],readouts=summary,
                pair_overlap=ensemble_overlap(self.panels,p,f,banks),
                parameter_ema_residual_max=max(win['residuals']),row_hashes=win['rows']))
            self.ram['active'] = None
        for k,v in score_cost.items():
            cost[k] = cost.get(k,0)+v
        cost['parent_window_step_panel_reads'] = 1
        cost['parent_window_seconds'] = time.perf_counter()-tick


@torch.no_grad()
def ema_residual(prev, model, ema, rho):
    m = dict(model.named_parameters())
    return float(torch.stack([(p.double()-rho*prev[n].double()-(1-rho)*m[n].double()).abs().max()
                              for n,p in ema.named_parameters()]).max())


def ensemble_overlap(panels,online,ema,banks):
    result = {}
    for label,bank in banks.items():
        cells = {}
        for name in CONFIRM:
            rows = []
            for q,pp in enumerate(panels[name].pairs):
                def ranks(read):
                    a,b = read[name]['p'][pp[:,0],q%2],read[name]['p'][pp[:,1],q%2]
                    return (a>b).astype(float)+.5*(a==b)
                start = ranks(online)
                f,z = ranks(ema)-start,ranks(bank)-start
                rows.append(dict(corrected_overlap=float(np.minimum(f.clip(0),z.clip(0)).mean()),
                                 broken_overlap=float(np.minimum((-f).clip(0),(-z).clip(0)).mean())))
            cells[name] = rows
        result[label] = cells
    return result
