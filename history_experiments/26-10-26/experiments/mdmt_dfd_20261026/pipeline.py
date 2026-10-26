"""Conditional P2/P3 execution on a verified, immutable own-seed prefix.

This module contains no quality-driven stopping inside an admitted stage.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

from .functions import PilotBank,FunctionBank,ResidualMLP,freeze,optimizer_owner_receipt
from .readers import Normalizer,ProbabilityHead,ReaderGroup,stable_mixture_bce,solve_simplex,select_whole_policy
from .plans import rng,torch_seed,batches
from .state import (atomic_json,append_jsonl,digest,array_hash,state_hash,sha_file,
                    save_checkpoint,load_checkpoint,capture_rng,restore_rng)
from .metrics import evaluate_probabilities,save_predictions
from .costs import CostLedger,measured_budget

P2_ARMS=('NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD')
P3_ARMS=('U_BUDGET','POOL')


def source_identity():
    files=('pipeline.py','functions.py','readers.py','state.py','plans.py','metrics.py','costs.py','demand.py','prefix.py')
    return {n:sha_file(Path(__file__).with_name(n)) for n in files}


def _seedfn(seed,dataset,namespace,component_offset=0):
    return lambda s,q,k:torch_seed(seed,dataset,namespace,source=s,task=q,component=component_offset+k)


def _chunks(ids,size=4096):
    for i in range(0,len(ids),size):yield np.asarray(ids[i:i+size],dtype=np.int64)


def _cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _identity(data,seed,arm,stage,prefix_hash):
    return dict(spec='DFD-v4.0',dataset=data.dataset,seed=int(seed),method=arm,stage=stage,
                data_hash=data.data_hash,prefix_hash=prefix_hash,source_hash=digest(source_identity()),
                config_hash=digest(dict(arm=arm,stage=stage,batch=40960,physical_formation_batch=40960,formation_passes=12,reader_form=60 if arm=='U_BUDGET' else 20,reader_read=10)),
                plan_hash=digest({'rng':'SeedSequence-v4','dataset':data.dataset,'seed':seed,'arm_paired_plans':True}))


def _save(path,identity,stage,model=None,opt=None,**extra):
    payload=dict(identity=identity,stage=stage,models={} if model is None else {'main':_cpu_state(model)},
                 optimizers={} if opt is None else {'main':copy.deepcopy(opt.state_dict())},rng=capture_rng(),
                 modes={} if model is None else {n:m.training for n,m in model.named_modules()},
                 requires_grad={} if model is None else {n:p.requires_grad for n,p in model.named_parameters()},
                 gradients={} if model is None else {n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()},**extra)
    save_checkpoint(path,payload)
    return payload


def _restore(payload,model,opt=None):
    model.load_state_dict(payload['models']['main'])
    if opt is not None:opt.load_state_dict(payload['optimizers']['main'])
    for n,m in model.named_modules():m.train(payload.get('modes',{}).get(n,False))
    for n,p in model.named_parameters():
        p.requires_grad_(payload.get('requires_grad',{}).get(n,True))
        grad=payload.get('gradients',{}).get(n)
        p.grad=None if grad is None else grad.to(p.device)
    restore_rng(payload['rng'])


def _feature_matrix(cache,ids,pilot=None,task=None,device='cuda:0'):
    out=[]
    for part in _chunks(ids):
        c=cache.get(part,device)
        with torch.no_grad():
            x=torch.cat((c['h'],c['z']),-1)
            if pilot is not None:
                r=pilot.residuals(c['h'])[:,:,task]
                x=torch.cat((x,r),-1)
        out.append(x.cpu().numpy())
    return np.concatenate(out)


def geometry_weights(data,cache,seed,device='cuda:0'):
    fit=data.ids('fit');w=np.ones((len(fit),2,2),np.float64);receipt={}
    for s in range(3):
        positions=np.flatnonzero(data.domain[fit]==s);ids=fit[positions]
        selected=rng(seed,data.dataset,'geometry_subset',source=s).permutation(ids)[:min(20000,len(ids))]
        subset=np.concatenate([cache.get(p,device)['h'].cpu().numpy() for p in _chunks(selected)])
        model=KMeans(n_clusters=2,init='k-means++',n_init=5,max_iter=300,tol=1e-4,
                     random_state=torch_seed(seed,data.dataset,'geometry_fit',source=s)).fit(subset)
        centers=model.cluster_centers_.astype(np.float64)
        def distance(x):return np.maximum(((x[:,None,:].astype(np.float64)-centers[None,:,:])**2).sum(-1),0.)
        distances=distance(subset);nz=distances[distances>0]
        base=max(float(np.median(nz)) if len(nz) else 1.,1e-6)
        def responsibilities(d,temp):
            logits=-d/temp;logits-=logits.max(-1,keepdims=True)
            e=np.exp(logits);return .05+.9*e/e.sum(-1,keepdims=True)
        def entropy(temp):
            p=responsibilities(distances,temp);return float(-(p*np.log(p)).sum(-1).mean())
        target=.5*np.log(2);lo,hi=base*1e-4,base*1e4
        el,eh=entropy(lo),entropy(hi)
        reachable=el<=target<=eh
        if reachable:
            for _ in range(60):
                mid=float(np.exp((np.log(lo)+np.log(hi))/2))
                if entropy(mid)<target:lo=mid
                else:hi=mid
            temp=float(np.exp((np.log(lo)+np.log(hi))/2))
        else:temp=min((lo,hi),key=lambda t:(abs(entropy(t)-target),t))
        offset=0
        for part in _chunks(ids):
            h=cache.get(part,device)['h'].cpu().numpy();val=2*responsibilities(distance(h),temp)
            pos=positions[offset:offset+len(part)];w[pos,0,:]=val;w[pos,1,:]=val;offset+=len(part)
        receipt[str(s)]=dict(subset_rows=selected,centers=centers,center_order_intended_targets=[t for t in range(3) if t!=s],
                             base=base,tau=temp,entropy=entropy(temp),target_entropy=target,reachable=reachable,
                             kmeans_iterations=int(model.n_iter_),random_state=torch_seed(seed,data.dataset,'geometry_fit',source=s))
    return w,receipt


def weights_from_frozen_geometry(data,cache,geometry,device='cuda:0'):
    """Resume a completed center/temperature choice without refitting KMeans."""
    fit=data.ids('fit');weights=np.ones((len(fit),2,2),np.float64)
    for s in range(3):
        receipt=geometry[str(s)];centers=np.asarray(receipt['centers'],dtype=np.float64);temperature=float(receipt['tau'])
        if centers.shape!=(2,256) or not temperature>0:raise ValueError('Frozen GEOM state malformed')
        if receipt['center_order_intended_targets']!=[t for t in range(3) if t!=s]:raise ValueError('Frozen GEOM target identity mismatch')
        positions=np.flatnonzero(data.domain[fit]==s)
        for start in range(0,len(positions),4096):
            pos=positions[start:start+4096];h=cache.get(fit[pos],device)['h'].cpu().numpy().astype(np.float64)
            distances=np.maximum(((h[:,None,:]-centers[None,:,:])**2).sum(-1),0.)
            logits=-distances/temperature;logits-=logits.max(-1,keepdims=True)
            e=np.exp(logits);value=2*(.05+.9*e/e.sum(-1,keepdims=True))
            weights[pos,0]=value;weights[pos,1]=value
    return weights


def shuffled_weights(data,fit_rows,weights,local_prob,seed):
    """One fixed joint-vector permutation within source/label/confidence strata."""
    fit_rows=np.asarray(fit_rows,dtype=np.int64);weights=np.asarray(weights,dtype=np.float64)
    y=data.batch(fit_rows,'fit','cpu',purpose='formation').y.numpy()
    result=weights.copy();index=np.arange(len(fit_rows));receipt={}
    for s in range(3):
        pos=np.flatnonzero(data.domain[fit_rows]==s)
        for q in range(2):
            cuts=np.quantile(local_prob[pos,q],np.arange(1,10)/10,method='linear')
            bins=np.searchsorted(cuts,local_prob[pos,q],side='right')
            stream=rng(seed,data.dataset,'shuffle_control',source=s,task=q)
            strata=[]
            for label in (0,1):
                for box in range(10):
                    target=pos[(y[pos,q]==label)&(bins==box)]
                    target=target[np.argsort(fit_rows[target],kind='stable')]
                    source=stream.permutation(target)
                    result[target,q,:]=weights[source,q,:]
                    # Separate task permutations; both intent coordinates stay joint.
                    strata.append(dict(label=label,bin=box,n=len(target),singleton=len(target)==1,
                                       row_hash=array_hash(fit_rows[target]),assignment_hash=array_hash(fit_rows[source])))
                    if len(target) and not np.array_equal(np.sort(result[target,q],axis=0),np.sort(weights[target,q],axis=0)):
                        raise AssertionError('SHUFFLE marginal changed')
            receipt[f's{s}_q{q}']=dict(cuts=cuts,strata=strata,changed_rows=float(np.any(result[pos,q]!=weights[pos,q],axis=1).mean()))
    return result,receipt


def weight_statistics(rows,weights,domain):
    out={}
    for s in range(3):
        for q in range(2):
            v=weights[np.asarray(domain)==s,q,:]
            out[f's{s}_q{q}']=dict(n=len(v),mean=v.mean(0),std=v.std(0),min=v.min(0),max=v.max(0),
                                   mass=v.sum(0),ess=v.sum(0)**2/(v*v).sum(0),
                                   correlation=float(np.corrcoef(v.T)[0,1]) if np.std(v[:,0])>0 else None)
    return dict(fit_row_hash=array_hash(rows),shape=list(weights.shape),row_sum_max_error=float(np.max(np.abs(weights.sum(-1)-2))),owners=out)


def train_formation(data,cache,pilot,seed,arm,weights,run_dir,heavy_dir,identity,device='cuda:0',resume=True,pooled=None,stop_after_pass=None):
    run_dir,heavy_dir=Path(run_dir),Path(heavy_dir);heavy_dir.mkdir(parents=True,exist_ok=True)
    kind='LOCAL_CAP' if arm=='LOCAL_CAP' else 'POOL' if arm=='POOL' else 'SOURCE'
    bank=FunctionBank(pilot,_seedfn(seed,data.dataset,'formation_perturb'),kind=kind,pooled=pooled).to(device)
    opt=torch.optim.Adam(bank.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    path=heavy_dir/'formation.pt';ledger=CostLedger(run_dir/'costs.jsonl',identity)
    current_pass,next_step=1,0
    if resume and path.exists():
        st=load_checkpoint(path,identity);_restore(st,bank,opt)
        if st.get('complete'):return freeze(bank),st
        current_pass,next_step=st['pass_id'],st['next_logical_step']
    fit=data.ids('fit');inverse=np.full(len(data.x),-1,dtype=np.int64);inverse[fit]=np.arange(len(fit))
    if weights is not None and np.asarray(weights).shape!=(len(fit),2,2):raise ValueError('Responsibility layout must be fit-position/task/intent')
    for pass_id in range(current_pass,13):
        plan=batches(fit,seed,data.dataset,'formation_plan',pass_id,batch_size=40960)
        with ledger.measure('formation_pass',arm=arm,pass_id=pass_id,begin_step=next_step):
            for step in range(next_step,len(plan)):
                ids=plan[step];opt.zero_grad(set_to_none=True);loss_value=0.
                for part in _chunks(ids,40960):
                    c=cache.get(part,device);batch=data.batch(part,'fit',device,purpose='formation')
                    w=None if weights is None else torch.as_tensor(weights[inverse[part]],device=device,dtype=torch.float32)
                    loss=bank.formation_loss(c['h'],c['z'],batch.y,batch.domain,w,denominator=len(ids))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite formation loss')
                    loss.backward();loss_value+=float(loss.detach())
                opt.step()
                _save(path,identity,'formation',bank,opt,pass_id=pass_id,next_logical_step=step+1,
                      row_plan_hash=array_hash(np.concatenate(plan)),owner_map=bank.owner_map,complete=False)
                append_jsonl(run_dir/'formation_steps.jsonl',dict(pass_id=pass_id,logical_step=step,rows=len(ids),loss=loss_value))
        ledger.access('formation','fit',len(fit),passes=1,arm=arm,pass_id=pass_id)
        next_step=0
        st=_save(path,identity,'formation',bank,opt,pass_id=pass_id+1,next_logical_step=0,owner_map=bank.owner_map,complete=pass_id==12)
        if pass_id==1:measured_budget(run_dir,'formation',{'formation_pass':11})
        if stop_after_pass is not None and pass_id==stop_after_pass and pass_id<12:
            st['retained_boundary']=f'formation_pass{pass_id}'
            return freeze(bank),st
    atomic_json(run_dir/'formation_owner_optimizer.json',optimizer_owner_receipt(bank,opt))
    return freeze(bank),st


def train_pooled_pilot(data,cache,seed,run_dir,heavy_dir,identity,device='cuda:0',resume=True):
    models=nn.ModuleList([ResidualMLP(seed=torch_seed(seed,data.dataset,'pilot_init',task=q,component=20)) for q in range(2)]).to(device)
    opt=torch.optim.Adam(models.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    path=Path(heavy_dir)/'pooled_pilot.pt';ledger=CostLedger(Path(run_dir)/'costs.jsonl',identity)
    start,skip=1,0
    if resume and path.exists():
        st=load_checkpoint(path,identity);_restore(st,models,opt)
        if st.get('complete'):return [freeze(h) for h in models]
        start,skip=st['pass_id'],st['next_logical_step']
    for pass_id in range(start,13):
        plan=batches(data.ids('fit'),seed,data.dataset,'pilot_plan',pass_id,component=20,batch_size=40960)
        with ledger.measure('pooled_pilot_pass',pass_id=pass_id,begin_step=skip):
            for step in range(skip,len(plan)):
                ids=plan[step];opt.zero_grad(set_to_none=True)
                for part in _chunks(ids,40960):
                    c=cache.get(part,device);b=data.batch(part,'fit',device,purpose='pilot')
                    a=torch.stack([c['z'][:,q]+models[q](c['h']) for q in range(2)],-1)
                    loss=F.binary_cross_entropy_with_logits(a,b.y,reduction='sum')/(2*len(ids));loss.backward()
                opt.step();_save(path,identity,'pooled_pilot',models,opt,pass_id=pass_id,next_logical_step=step+1,complete=False)
        ledger.access('pooled_pilot','fit',len(data.ids('fit')),passes=1,pass_id=pass_id)
        skip=0;_save(path,identity,'pooled_pilot',models,opt,pass_id=pass_id+1,next_logical_step=0,complete=pass_id==12)
    return [freeze(h) for h in models]


class CandidateCache:
    def __init__(self,data,cache,pilot,bank,root,device='cuda:0'):
        self.data,self.cache,self.pilot,self.bank,self.root,self.device=data,cache,pilot,bank,Path(root),device
        self.root.mkdir(parents=True,exist_ok=True);self.maps={}
        self.identity=dict(data_hash=data.data_hash,pilot_hash=state_hash(pilot.state_dict()),bank_hash=state_hash(bank.state_dict()),kind=bank.kind)
    def role(self,role):
        if role in self.maps:return self.maps[role]
        rows=self.data.ids(role);path=self.root/f'{role}_candidates.npy';meta=path.with_suffix('.json')
        expected=dict(self.identity,row_hash=array_hash(rows),shape=[len(rows),2,5],dtype='float32')
        if path.exists() and meta.exists() and json.loads(meta.read_text())==expected:
            out=np.load(path,mmap_mode='r');self.maps[role]=out;return out
        out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),2,5))
        offset=0
        with torch.no_grad():
            for part in _chunks(rows):
                c=self.cache.get(part,self.device)
                out[offset:offset+len(part)]=self.bank.candidates(c['h'],c['z'],c['domain'],self.pilot).cpu().numpy();offset+=len(part)
        out.flush();atomic_json(meta,expected);self.maps[role]=np.load(path,mmap_mode='r');return self.maps[role]
    def get(self,ids,role,device=None):
        rows=self.data.ids(role);pos=np.searchsorted(rows,ids)
        if not np.array_equal(rows[pos],ids):raise ValueError('Candidate role membership mismatch')
        return torch.as_tensor(np.array(self.role(role)[pos]),device=device or self.device)


def fit_reader_normalizers(data,cache,candidates,reader,device):
    # Streaming population moments over actual target fit rows, no target labels.
    for t in range(3):
        ids=data.ids('fit',t)
        for q in range(2):
            n=0;mean=np.zeros(263,np.float64);m2=np.zeros(263,np.float64)
            for part in _chunks(ids):
                c=cache.get(part,device);a=candidates.get(part,'fit',device)
                values=ReaderGroup.features(c['h'],c['z'],a,q).cpu().numpy().astype(np.float64)
                bn=len(values);bm=values.mean(0);bv=((values-bm)**2).sum(0);delta=bm-mean
                mean+=delta*bn/(n+bn);m2+=bv+delta**2*n*bn/(n+bn);n+=bn
            reader.normalizers[t,q]=Normalizer(mean,np.maximum(np.sqrt(m2/n),1e-3))


def _predict_reader(data,cache,candidates,reader,role,device,authorization=None):
    ids=data.ids(role);out=[];alpha=[];labels=[];domains=[]
    for part in _chunks(ids):
        b=data.batch(part,role,'cpu',purpose='select' if role=='P_cal' else 'evaluate',authorization=authorization)
        c=cache.get(part,device);a=candidates.get(part,role,device)
        p,w=reader.predict(c['h'],c['z'],c['domain'],a)
        out.append(p.cpu().numpy());alpha.append(w.cpu().numpy());labels.append(b.y.numpy());domains.append(b.domain.numpy())
    return dict(rows=ids,p=np.concatenate(out),alpha=np.concatenate(alpha),y=np.concatenate(labels),domain=np.concatenate(domains))


def train_reader(data,cache,pilot,bank,seed,arm,run_dir,heavy_dir,identity,device='cuda:0',resume=True,uniform_prefit=None,uniform_identity=None,stop_after_pass=None):
    run_dir,heavy_dir=Path(run_dir),Path(heavy_dir);heavy_dir.mkdir(parents=True,exist_ok=True)
    reader=ReaderGroup(_seedfn(seed,data.dataset,'read_init')).to(device)
    opt=torch.optim.Adam(reader.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    candidates=CandidateCache(data,cache,pilot,bank,heavy_dir/'candidate_cache',device)
    path=heavy_dir/'reader_progress.pt';ledger=CostLedger(run_dir/'costs.jsonl',identity)
    form_passes=60 if arm=='U_BUDGET' else 20;phase,pass_id,next_head,next_step='P_form',1,0,0;choices=[]
    if resume and path.exists():
        st=load_checkpoint(path,identity);_restore(st,reader,opt);reader.load_extra_state(st['extra'])
        if st.get('complete'):
            selected=load_checkpoint(heavy_dir/'reader_selected.pt',identity);_restore(selected,reader,opt);reader.load_extra_state(selected['extra'])
            return freeze(reader),candidates,selected
        phase,pass_id,next_head,next_step=st['phase'],st['pass_id'],st['next_head'],st['next_logical_step'];choices=st.get('choices',[])
    else:
        fit_reader_normalizers(data,cache,candidates,reader,device)
        if uniform_prefit is not None:
            if uniform_identity is None:raise ValueError('UNIFORM prefit requires complete declared dependency identity')
            st=load_checkpoint(uniform_prefit,uniform_identity)
            if st['phase']!='P_form' or st['pass_id']!=21 or st['next_head']!=0 or st['next_logical_step']!=0:
                raise ValueError('UNIFORM dependency is not the complete prefit20 boundary')
            _restore(st,reader,opt);reader.load_extra_state(st['extra']);phase,pass_id='P_form',21
    def snapshot(name,phase,pass_id,next_head=0,next_step=0,complete=False):
        return _save(heavy_dir/name,identity,'reader',reader,opt,phase=phase,pass_id=pass_id,next_head=next_head,
                     next_logical_step=next_step,extra=reader.extra_state(),choices=choices,complete=complete)
    if not choices and phase=='P_form':
        reader.off=True;raw=_predict_reader(data,cache,candidates,reader,'P_cal',device);reader.off=False
        metric=evaluate_probabilities(raw['y'],raw['p'],raw['domain'])
        choices.append(dict(candidate='OFF',macro_logloss=metric['macro_logloss']))
        initial=snapshot('reader_selected.pt','P_form',0)
        initial['selection']=choices[0];initial['extra']['off']=True
        save_checkpoint(heavy_dir/'reader_selected.pt',initial)
        ledger.access('reader_select','P_cal',len(raw['rows']),tasks=2,candidate='OFF')
    while True:
        last=form_passes if phase=='P_form' else 10
        for pid in range(pass_id,last+1):
            with ledger.measure('reader_form_pass' if phase=='P_form' else 'reader_read_pass',arm=arm,pass_id=pid,begin_head=next_head,begin_step=next_step):
                for head_index in range(next_head,6):
                    q,t=divmod(head_index,3);ids=data.ids(phase,t)
                    plan=batches(ids,seed,data.dataset,'read_plan',pid,source=t,task=q,component=0 if phase=='P_form' else 100,batch_size=512)
                    head=reader.head(t,q)
                    for step in range(next_step,len(plan)):
                        part=plan[step];c=cache.get(part,device);a=candidates.get(part,phase,device)
                        b=data.batch(part,phase,device,purpose='reader');x=reader.features(c['h'],c['z'],a,q)
                        opt.zero_grad(set_to_none=True);g=head(reader.normalizers[t,q].transform(x))
                        loss=stable_mixture_bce(a[:,q],g,b.y[:,q]);
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reader risk')
                        loss.backward();opt.step()
                        snapshot('reader_progress.pt',phase,pid,head_index,step+1)
                    next_step=0
                next_head=0
            ledger.access('reader','P_form' if phase=='P_form' else 'P_read',len(data.ids(phase)),tasks=2,passes=1,pass_id=pid,arm=arm)
            if phase=='P_form' and pid==20:snapshot('reader_prefit20.pt',phase,pid+1)
            if phase=='P_form' and pid==form_passes or phase=='P_read':
                candidate_id=0 if phase=='P_form' else pid
                with ledger.measure('reader_calibration',candidate=candidate_id):
                    raw=_predict_reader(data,cache,candidates,reader,'P_cal',device)
                    metric=evaluate_probabilities(raw['y'],raw['p'],raw['domain'])
                choice=dict(candidate=candidate_id,macro_logloss=metric['macro_logloss'],metrics=metric)
                previous_best=min(c['macro_logloss'] for c in choices)
                choices.append(choice)
                if choice['macro_logloss']<previous_best:
                    selected=snapshot('reader_selected.pt',phase,pid+1)
                    selected['selection']=choice;save_checkpoint(heavy_dir/'reader_selected.pt',selected)
                ledger.access('reader_select','P_cal',len(raw['rows']),tasks=2,candidate=candidate_id)
            snapshot('reader_progress.pt',phase,pid+1)
            if pid==1:measured_budget(run_dir,'reader',{'reader_form_pass':max(form_passes-pid,0),'reader_read_pass':10})
            if phase=='P_form' and stop_after_pass is not None and pid==stop_after_pass:
                retained=load_checkpoint(path,identity);retained['retained_boundary']=f'reader_form_pass{pid}'
                return freeze(reader),candidates,retained
        if phase=='P_form':phase,pass_id,next_head,next_step='P_read',1,0,0
        else:break
    best=select_whole_policy(choices);selected=load_checkpoint(heavy_dir/'reader_selected.pt',identity)
    _restore(selected,reader,opt);reader.load_extra_state(selected['extra']);reader.off=best['candidate']=='OFF'
    selected['extra']=reader.extra_state();selected['selection']=best;selected['choices']=choices;selected['complete']=True
    save_checkpoint(heavy_dir/'reader_selected.pt',selected)
    # Preserve trained pass10 separately; progress is a completion receipt, not a
    # replacement of the selected complete six-head state.
    progress=load_checkpoint(path,identity);progress['complete']=True;save_checkpoint(path,progress)
    atomic_json(run_dir/'reader_selection.json',dict(selected=best,candidates=choices,whole_six_heads=True))
    return freeze(reader),candidates,selected


def fit_constant(data,candidates,run_dir,heavy_dir,identity,device='cuda:0',resume=True):
    path=Path(heavy_dir)/'constant.pt';ledger=CostLedger(Path(run_dir)/'costs.jsonl',identity)
    if resume and path.exists():return load_checkpoint(path,identity)
    alphas={};receipts={}
    with ledger.measure('constant_solver'):
        for q in range(2):
            for t in range(3):
                ps=[];ys=[]
                for role in ('P_form','P_read'):
                    ids=data.ids(role,t)
                    for part in _chunks(ids):
                        a=candidates.get(part,role,device).double().sigmoid().cpu().numpy()
                        ps.append(a[:,q]);ys.append(data.batch(part,role,'cpu',purpose='constant').y.numpy()[:,q])
                alpha,receipt=solve_simplex(np.concatenate(ps),np.concatenate(ys));alphas[t,q]=alpha;receipts[f't{t}_q{q}']=receipt
    cal=data.ids('P_cal');p=[];off=[];ys=[];ds=[]
    for part in _chunks(cal):
        a=candidates.get(part,'P_cal',device).double().sigmoid().cpu().numpy();d=data.domain[part]
        result=np.empty((len(part),2),np.float64)
        for t in range(3):
            mask=d==t
            for q in range(2):result[mask,q]=a[mask,q]@alphas[t,q]
        p.append(result);off.append(a[...,0]);b=data.batch(part,'P_cal','cpu',purpose='select');ys.append(b.y.numpy());ds.append(d)
    y,d=np.concatenate(ys),np.concatenate(ds);m=evaluate_probabilities(y,np.concatenate(p),d);m0=evaluate_probabilities(y,np.concatenate(off),d)
    selected='CONST' if m['macro_logloss']<m0['macro_logloss'] else 'OFF'
    payload=dict(identity=identity,stage='constant',alphas=alphas,solver_receipts=receipts,selection=selected,
                 P_cal={'CONST':m,'OFF':m0},rng=capture_rng(),complete=True)
    save_checkpoint(path,payload);atomic_json(Path(run_dir)/'constant_solver_receipts.json',dict(receipts=receipts,selection=selected,P_cal=payload['P_cal']))
    ledger.access('constant_solver','P_form',len(data.ids('P_form')),tasks=2)
    ledger.access('constant_solver','P_read',len(data.ids('P_read')),tasks=2)
    ledger.access('constant_select','P_cal',len(cal),tasks=2)
    return payload


class DirectGroup(nn.Module):
    def __init__(self,input_dim,seed,dataset,positive_rates=None):
        super().__init__();self.input_dim=input_dim
        self.heads=nn.ModuleDict({f's{t}_q{q}':ProbabilityHead(input_dim,.5 if positive_rates is None else positive_rates[t,q],
                          torch_seed(seed,dataset,'read_init',source=t,task=q,component=10)) for q in range(2) for t in range(3)})
        self.normalizers={};self.optimizers={}
    def predict(self,features,domain):
        out=torch.empty((len(domain),2),device=domain.device,dtype=torch.float64)
        with torch.no_grad():
            for t in range(3):
                mask=domain==t
                if not bool(mask.any()):continue
                for q in range(2):
                    key=f's{t}_q{q}';x=features if self.input_dim==258 else features[q]
                    out[mask,q]=self.heads[key](self.normalizers[key].transform(x[mask])).double().sigmoid()
        return out


def _direct_features(c,pilot,input_dim):
    base=torch.cat((c['h'],c['z']),-1)
    if input_dim==258:return base
    r=pilot.residuals(c['h']).detach()
    return [torch.cat((base,r[:,:,q]),-1) for q in range(2)]


def _direct_predict(data,cache,pilot,group,role,device,authorization=None):
    ps=[];ys=[];ds=[];ids=data.ids(role)
    for part in _chunks(ids):
        c=cache.get(part,device);features=_direct_features(c,pilot,group.input_dim)
        ps.append(group.predict(features,c['domain']).cpu().numpy())
        b=data.batch(part,role,'cpu',purpose='select' if role=='P_cal' else 'evaluate',authorization=authorization)
        ys.append(b.y.numpy());ds.append(b.domain.numpy())
    return dict(rows=ids,p=np.concatenate(ps),y=np.concatenate(ys),domain=np.concatenate(ds))


def train_direct(data,cache,pilot,eta,seed,arm,run_dir,heavy_dir,identity,device='cuda:0',resume=True):
    heavy_dir,run_dir=Path(heavy_dir),Path(run_dir);heavy_dir.mkdir(parents=True,exist_ok=True)
    group=DirectGroup(258 if arm=='H_DIRECT' else 261,seed,data.dataset).to(device)
    if arm=='ETA_DIRECT':
        group.heads=copy.deepcopy(eta.heads).to(device);group.normalizers=copy.deepcopy(eta.normalizers)
        group.requires_grad_(True)
    optimizers={key:torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0) for key,head in group.heads.items()}
    if arm=='ETA_DIRECT':
        for key,opt in optimizers.items():opt.load_state_dict(copy.deepcopy(eta.optimizers[key].state_dict()))
    path=heavy_dir/'direct_progress.pt';ledger=CostLedger(run_dir/'costs.jsonl',identity)
    phase,pass_id,head_idx,next_step='P_read' if arm=='ETA_DIRECT' else 'P_form',1,0,0;choices=[]
    def restore(st):
        group.load_state_dict(st['models']['main']);group.normalizers={k:Normalizer.from_state_dict(v) for k,v in st['normalizers'].items()}
        for key,opt in optimizers.items():opt.load_state_dict(st['optimizers'][key])
        restore_rng(st['rng'])
    def snap(name,phase,pid,head=0,step=0,complete=False):
        st=dict(identity=identity,stage='direct',models={'main':_cpu_state(group)},optimizers={k:copy.deepcopy(v.state_dict()) for k,v in optimizers.items()},
                normalizers={k:v.state_dict() for k,v in group.normalizers.items()},phase=phase,pass_id=pid,next_head=head,next_logical_step=step,
                choices=choices,rng=capture_rng(),input_dim=group.input_dim,complete=complete)
        save_checkpoint(heavy_dir/name,st);return st
    if resume and path.exists():
        st=load_checkpoint(path,identity);restore(st)
        if st.get('complete'):
            st=load_checkpoint(heavy_dir/'direct_selected.pt',identity);restore(st);return freeze(group),st
        phase,pass_id,head_idx,next_step=st['phase'],st['pass_id'],st['next_head'],st['next_logical_step'];choices=st['choices']
    elif arm=='H_DIRECT':
        for t in range(3):
            values=_feature_matrix(cache,data.ids('fit',t),device=device)
            for q in range(2):
                key=f's{t}_q{q}';group.normalizers[key]=Normalizer.from_array(values)
                rate=float(data.batch(data.ids('P_form',t),'P_form','cpu',purpose='reader').y[:,q].mean())
                group.heads[key].output.bias.data.fill_(float(np.log(np.clip(rate,1e-4,1-1e-4)/(1-np.clip(rate,1e-4,1-1e-4)))))
    def score(candidate):
        raw=_direct_predict(data,cache,pilot,group,'P_cal',device);m=evaluate_probabilities(raw['y'],raw['p'],raw['domain'])
        previous_best=min((c['macro_logloss'] for c in choices),default=float('inf'))
        choice=dict(candidate=candidate,macro_logloss=m['macro_logloss'],metrics=m)
        choices.append(choice)
        if choice['macro_logloss']<previous_best:
            selected=snap('direct_selected.pt',phase,pass_id);selected['selection']=choice
            save_checkpoint(heavy_dir/'direct_selected.pt',selected)
        ledger.access('direct_select','P_cal',len(raw['rows']),tasks=2,candidate=candidate)
    if arm=='ETA_DIRECT' and not choices:score(0)
    while True:
        last=20 if phase=='P_form' else 10
        for pid in range(pass_id,last+1):
            with ledger.measure('direct_form_pass' if phase=='P_form' else 'direct_read_pass',pass_id=pid,begin_head=head_idx,begin_step=next_step):
                for hi in range(head_idx,6):
                    q,t=divmod(hi,3);key=f's{t}_q{q}';head=group.heads[key];opt=optimizers[key]
                    plan=batches(data.ids(phase,t),seed,data.dataset,'read_plan',pid,source=t,task=q,
                                 component=(10 if arm=='H_DIRECT' else 11)+(100 if phase=='P_read' else 0),batch_size=512)
                    for step in range(next_step,len(plan)):
                        part=plan[step];c=cache.get(part,device);x=_direct_features(c,pilot,group.input_dim)
                        if group.input_dim==261:x=x[q]
                        b=data.batch(part,phase,device,purpose='reader');opt.zero_grad(set_to_none=True)
                        logit=head(group.normalizers[key].transform(x));loss=F.binary_cross_entropy_with_logits(logit,b.y[:,q]);loss.backward();opt.step()
                        snap('direct_progress.pt',phase,pid,hi,step+1)
                    next_step=0
                head_idx=0
            ledger.access('direct_train',phase,len(data.ids(phase)),tasks=2,passes=1,pass_id=pid)
            pass_id=pid+1
            if phase=='P_read' or phase=='P_form' and pid==20:score(pid if phase=='P_read' else 0)
            snap('direct_progress.pt',phase,pid+1)
        if phase=='P_form':phase,pass_id='P_read',1
        else:break
    selected=min(enumerate(choices),key=lambda v:(v[1]['macro_logloss'],v[0]))[1]
    st=load_checkpoint(heavy_dir/'direct_selected.pt',identity);restore(st);st['selection']=selected;st['choices']=choices;st['complete']=True
    save_checkpoint(heavy_dir/'direct_selected.pt',st);progress=load_checkpoint(path,identity);progress['complete']=True;save_checkpoint(path,progress)
    atomic_json(run_dir/'direct_selection.json',dict(selected=selected,candidates=choices,whole_six_heads=True,has_OFF=False))
    return freeze(group),st


def _export_raw(data,cache,pilot,arm,role,device,authorization=None):
    ps=[];ys=[];ds=[];ids=data.ids(role)
    for part in _chunks(ids):
        c=cache.get(part,device)
        with torch.no_grad():a=c['z'] if arm=='NATIVE' else pilot.local_logits(c['h'],c['z'],c['domain'])
        purpose='formation' if role=='fit' else 'evaluate'
        ps.append(a.double().sigmoid().cpu().numpy());b=data.batch(part,role,'cpu',purpose=purpose,authorization=authorization)
        ys.append(b.y.numpy());ds.append(b.domain.numpy())
    return dict(rows=ids,p=np.concatenate(ps),y=np.concatenate(ys),domain=np.concatenate(ds))


def _constant_predict(data,candidates,state,role,device,authorization=None):
    ps=[];ys=[];ds=[];ids=data.ids(role)
    for part in _chunks(ids):
        a=candidates.get(part,role,device).double().sigmoid().cpu().numpy();d=data.domain[part];p=a[...,0].copy()
        if state['selection']!='OFF':
            for t in range(3):
                mask=d==t
                for q in range(2):p[mask,q]=a[mask,q]@state['alphas'][t,q]
        ps.append(p);b=data.batch(part,role,'cpu',purpose='evaluate',authorization=authorization);ys.append(b.y.numpy());ds.append(d)
    return dict(rows=ids,p=np.concatenate(ps),y=np.concatenate(ys),domain=np.concatenate(ds))


def _bank_from_state(pilot,state,device):
    kind=state.get('kind','SOURCE')
    bank=FunctionBank(pilot,kind=kind,pooled=[pilot.head(0,q) for q in range(2)] if kind=='POOL' else None)
    bank.load_state_dict(state['models']['main']);return freeze(bank.to(device))


def run_pipeline(data,seed,artifact_root,device='cuda:0',stage='p2',arms=None,resume=True,stop_after=None):
    from .prefix import load_prefix,paths as prefix_paths
    from .demand import refit_eta,make_responsibilities
    root=Path(artifact_root);prefix_dir=root/'prefix'/data.dataset/f'seed{seed}'
    cache,pilot=load_prefix(data,seed,root,device)
    freeze(pilot);prefix_hash=digest(dict(pilot=state_hash(pilot.state_dict()),cache_hash=cache.hash,native_hash=cache.native_hash))
    arms=tuple(arms or (P3_ARMS if stage=='p3' else P2_ARMS))
    if any(a not in P2_ARMS+P3_ARMS for a in arms):raise ValueError('Unknown pre-registered arm')
    if stop_after not in (None,'formation1','reader1'):raise ValueError('Unknown retained component boundary')
    # ETA is refitted only once after the dataset-level P1 gate, never for a failed P1.
    eta=refit_eta(data,cache,pilot,seed,prefix_dir,device)
    responsibilities=make_responsibilities(data,cache,pilot,eta,seed,prefix_dir,device)
    fit=np.asarray(responsibilities['fit_row_id']);expected=data.ids('fit')
    if not np.array_equal(fit,expected):raise ValueError('Responsibility global row map mismatch')
    dfd_w=np.asarray(responsibilities['w'],dtype=np.float64)
    local_fit=None;weights_cache={};uniform_bank=None;results={}
    # UNIFORM must precede CONST and U_BUDGET dependencies, without discarding any arm.
    ordered=sorted(arms,key=lambda a:0 if a=='UNIFORM' else 2 if a=='CONST_UNIFORM' else 1)
    for arm in ordered:
        run_dir=root/'runs'/data.dataset/arm/f'seed{seed}';heavy_dir=run_dir/'transient'
        run_dir.mkdir(parents=True,exist_ok=True);heavy_dir.mkdir(parents=True,exist_ok=True)
        identity=_identity(data,seed,arm,stage,prefix_hash);atomic_json(run_dir/'config.json',identity)
        atomic_json(run_dir/'data_manifest.json',data.manifest);ledger=CostLedger(run_dir/'costs.jsonl',identity)
        if (run_dir/'COMPLETE.json').exists() and resume:
            done=json.loads((run_dir/'COMPLETE.json').read_text())
            if done['identity']!=identity:raise ValueError('Completed arm identity mismatch')
            results[arm]=done;continue
        bank,reader,candidates,direct,constant=None,None,None,None,None
        final_state=None
        if arm not in ('NATIVE','LOCAL','H_DIRECT','ETA_DIRECT'):
            if arm in ('CONST_UNIFORM','U_BUDGET'):
                uniform_path=root/'runs'/data.dataset/'UNIFORM'/f'seed{seed}'/'transient'/'formation.pt'
                uniform_identity=_identity(data,seed,'UNIFORM','p2' if stage=='p3' else stage,prefix_hash)
                uniform_st=load_checkpoint(uniform_path,uniform_identity)
                if not uniform_st.get('complete'):raise ValueError('UNIFORM dependency incomplete')
                bank=_bank_from_state(pilot,uniform_st,device)
            else:
                weight_path=heavy_dir/'weights.pt'
                if resume and weight_path.exists():
                    saved_weights=load_checkpoint(weight_path,identity)
                    if not np.array_equal(saved_weights['fit_row_id'],fit):raise ValueError('Frozen weight row identity mismatch')
                    weights=np.asarray(saved_weights['w'],dtype=np.float64)
                    if weights.shape!=(len(fit),2,2):raise ValueError('Frozen weight shape mismatch')
                else:
                    if arm=='DFD':weights=dfd_w
                    elif arm=='UNIFORM':weights=np.ones_like(dfd_w)
                    elif arm=='GEOM':
                        geometry_path=heavy_dir/'geometry.pt'
                        with ledger.measure('geometry'):
                            if resume and geometry_path.exists():
                                saved_geometry=load_checkpoint(geometry_path,identity)
                                if saved_geometry['fit_row_id_hash']!=array_hash(fit):raise ValueError('Frozen GEOM row identity mismatch')
                                geometry=saved_geometry['geometry'];weights=weights_from_frozen_geometry(data,cache,geometry,device)
                            else:
                                weights,geometry=geometry_weights(data,cache,seed,device)
                                save_checkpoint(geometry_path,dict(identity=identity,geometry=geometry,fit_row_id_hash=array_hash(fit)))
                    elif arm=='SHUFFLE':
                        if local_fit is None:local_fit=_export_raw(data,cache,pilot,'LOCAL','fit',device)['p']
                        with ledger.measure('shuffle'):
                            weights,shuffle=shuffled_weights(data,fit,dfd_w,local_fit,seed)
                        atomic_json(run_dir/'shuffle_receipt.json',shuffle)
                    else:weights=None
                if weights is not None:
                    if not weight_path.exists():save_checkpoint(weight_path,dict(identity=identity,fit_row_id=fit,w=weights))
                    atomic_json(run_dir/'weight_statistics.json',weight_statistics(fit,weights,data.domain[fit]))
                pooled=train_pooled_pilot(data,cache,seed,run_dir,heavy_dir,identity,device,resume) if arm=='POOL' else None
                bank,formation=train_formation(data,cache,pilot,seed,arm,weights,run_dir,heavy_dir,identity,device,resume,pooled,
                                               stop_after_pass=1 if stop_after=='formation1' else None)
                if formation.get('retained_boundary'):
                    return dict(status='RETAINED_BOUNDARY',component=formation['retained_boundary'],dataset=data.dataset,
                                seed=seed,arm=arm,run_dir=str(run_dir),checkpoint=str(heavy_dir/'formation.pt'))
                formation['kind']=bank.kind;save_checkpoint(heavy_dir/'formation.pt',formation)
            if arm=='CONST_UNIFORM':
                candidates=CandidateCache(data,cache,pilot,bank,heavy_dir/'candidate_cache',device)
                constant=fit_constant(data,candidates,run_dir,heavy_dir,identity,device,resume);final_state=constant
            else:
                prefit=root/'runs'/data.dataset/'UNIFORM'/f'seed{seed}'/'transient'/'reader_prefit20.pt' if arm=='U_BUDGET' else None
                dependency_identity=uniform_identity if arm=='U_BUDGET' else None
                reader,candidates,final_state=train_reader(data,cache,pilot,bank,seed,arm,run_dir,heavy_dir,identity,device,resume,prefit,dependency_identity,
                                                           stop_after_pass=1 if stop_after=='reader1' else None)
                if final_state.get('retained_boundary'):
                    return dict(status='RETAINED_BOUNDARY',component=final_state['retained_boundary'],dataset=data.dataset,
                                seed=seed,arm=arm,run_dir=str(run_dir),checkpoint=str(heavy_dir/'reader_progress.pt'))
        elif arm in ('H_DIRECT','ETA_DIRECT'):
            direct,final_state=train_direct(data,cache,pilot,eta,seed,arm,run_dir,heavy_dir,identity,device,resume)
        component_hashes=dict(native_hash=cache.native_hash,pilot_hash=state_hash(pilot.state_dict()),
                              bank_hash=None if bank is None else state_hash(bank.state_dict()),
                              readout_state_hash=None if final_state is None else state_hash(final_state),
                              exact_policy='NATIVE' if arm=='NATIVE' else 'LOCAL' if arm=='LOCAL' else None)
        prediction_identity=digest(dict(identity=identity,components=component_hashes))
        atomic_json(run_dir/'PREDICTOR_COMPONENT_IDENTITIES.json',dict(predictor_hash=prediction_identity,components=component_hashes))
        metrics={}
        for role in ('validation','Q_explore'):
            with ledger.measure('evaluation',role=role):
                if arm in ('NATIVE','LOCAL'):raw=_export_raw(data,cache,pilot,arm,role,device)
                elif direct is not None:raw=_direct_predict(data,cache,pilot,direct,role,device)
                elif constant is not None:raw=_constant_predict(data,candidates,constant,role,device)
                else:raw=_predict_reader(data,cache,candidates,reader,role,device)
                metrics[role]=save_predictions(run_dir/f'{role}_selected.npz',raw['rows'],raw['y'],raw['p'],raw['domain'],role,prediction_identity,
                                               **({'alpha':raw['alpha']} if 'alpha' in raw else {}))
            ledger.access('evaluation',role,len(raw['rows']),tasks=2)
        # Dependency paths plus exact hashes are enough to reconstruct; all
        # learned final components are also embedded in one predictor capsule.
        native_selected=load_checkpoint(prefix_paths(data,seed,root)[1]/'native_selected.pt')
        capsule=dict(identity=identity,predictor_hash=prediction_identity,component_hashes=component_hashes,arm=arm,pilot=_cpu_state(pilot),
                     native_state=native_selected['models']['native']['model'],schema=data.schema,
                     bank=None if bank is None else {'kind':bank.kind,'state':_cpu_state(bank),'owner_map':bank.owner_map},
                     reader=None if reader is None else {'state':_cpu_state(reader),'extra':reader.extra_state()},
                     direct=None if direct is None else {'state':_cpu_state(direct),'input_dim':direct.input_dim,
                                  'normalizers':{k:v.state_dict() for k,v in direct.normalizers.items()}},
                     constant=constant,prefix_dir=str(prefix_dir),cache_identity=getattr(cache,'identity',None),
                     native_dependency='own seed validation-selected frozen native in prefix',complete=True)
        save_checkpoint(heavy_dir/'predictor.pt',capsule)
        # Keep the fixed P_read pass10 endpoint distinct from P_cal-selected.
        if reader is not None or direct is not None:
            if reader is not None:
                selected_state=copy.deepcopy(reader.state_dict());selected_extra=copy.deepcopy(reader.extra_state())
                terminal=load_checkpoint(heavy_dir/'reader_progress.pt',identity)
                reader.load_state_dict(terminal['models']['main']);reader.load_extra_state(terminal['extra'])
            else:
                selected_state=copy.deepcopy(direct.state_dict())
                terminal=load_checkpoint(heavy_dir/'direct_progress.pt',identity)
                direct.load_state_dict(terminal['models']['main'])
            for role in ('validation','Q_explore'):
                with ledger.measure('endpoint_evaluation',role=role):
                    raw=_predict_reader(data,cache,candidates,reader,role,device) if reader is not None else _direct_predict(data,cache,pilot,direct,role,device)
                    endpoint_identity=digest(dict(identity=identity,components=dict(component_hashes,readout_state_hash=state_hash(terminal)),endpoint='P_read_pass10'))
                    save_predictions(run_dir/f'{role}_read_pass10.npz',raw['rows'],raw['y'],raw['p'],raw['domain'],role,
                                     endpoint_identity,**({'alpha':raw['alpha']} if 'alpha' in raw else {}))
                ledger.access('endpoint_evaluation',role,len(raw['rows']),tasks=2)
            if reader is not None:reader.load_state_dict(selected_state);reader.load_extra_state(selected_extra)
            else:direct.load_state_dict(selected_state)
        # Rebuild the complete predictor from the capsule and original raw
        # encoded inputs; this check does not reuse its feature cache.
        check_rows=data.ids('Q_explore')[:min(96,len(data.ids('Q_explore')))]
        rebuilt=CapsulePredictor(capsule,device)
        with torch.no_grad():
            check=rebuilt(torch.as_tensor(np.array(data.x[check_rows]),device=device)).cpu().numpy()
        saved=np.load(run_dir/'Q_explore_selected.npz')['p'][:len(check_rows)]
        reconstruction_error=float(np.max(np.abs(check-saved)))
        if reconstruction_error>1e-6:raise ValueError('Final predictor reconstruction mismatch')
        atomic_json(run_dir/'PREDICTOR_RECONSTRUCTION.json',dict(rows=check_rows,max_abs_error=reconstruction_error,cache_reused=False))
        done=dict(identity=identity,predictor_hash=prediction_identity,component_hashes=component_hashes,predictor_path=str(heavy_dir/'predictor.pt'),metrics=metrics,status='COMPLETE')
        atomic_json(run_dir/'COMPLETE.json',done);results[arm]=done
    return results


class CapsulePredictor(nn.Module):
    """Complete inference-only predictor: encoded X in, probabilities out, no Y."""
    def __init__(self,capsule,device='cuda:0'):
        super().__init__()
        from .native_adapter import make_native
        self.arm=capsule['arm'];self.constant=capsule['constant'];self.device=device
        self.native,_=make_native(capsule['schema'],capsule['identity']['seed'],capsule['identity']['dataset'],device)
        self.native.load_state_dict(capsule['native_state']);freeze(self.native)
        self.pilot=None
        if self.arm not in ('NATIVE','H_DIRECT'):
            self.pilot=PilotBank().to(device);self.pilot.load_state_dict(capsule['pilot']);freeze(self.pilot)
        self.bank=self.reader=self.direct=None
        if capsule['bank'] is not None:
            kind=capsule['bank']['kind']
            self.bank=FunctionBank(self.pilot,kind=kind,pooled=[self.pilot.head(0,q) for q in range(2)] if kind=='POOL' else None).to(device)
            self.bank.load_state_dict(capsule['bank']['state']);freeze(self.bank)
        if capsule['reader'] is not None:
            self.reader=ReaderGroup().to(device);self.reader.load_state_dict(capsule['reader']['state']);self.reader.load_extra_state(capsule['reader']['extra']);freeze(self.reader)
        if capsule['direct'] is not None:
            st=capsule['direct'];self.direct=DirectGroup(st['input_dim'],capsule['identity']['seed'],capsule['identity']['dataset']).to(device)
            self.direct.load_state_dict(st['state']);self.direct.normalizers={k:Normalizer.from_state_dict(v) for k,v in st['normalizers'].items()};freeze(self.direct)
        self.eval()

    def forward(self,x):
        with torch.no_grad():
            c=self.native.encode(x);domain=x[:,-1].long()
            if self.arm=='NATIVE':return c.logits.double().sigmoid()
            if self.arm=='LOCAL':return self.pilot.local_logits(c.h,c.logits,domain).double().sigmoid()
            if self.direct is not None:
                return self.direct.predict(_direct_features({'h':c.h,'z':c.logits},self.pilot,self.direct.input_dim),domain)
            a=self.bank.candidates(c.h,c.logits,domain,self.pilot)
            if self.reader is not None:return self.reader.predict(c.h,c.logits,domain,a)[0]
            probabilities=a.double().sigmoid();out=probabilities[...,0].clone()
            if self.constant['selection']!='OFF':
                for t in range(3):
                    mask=domain==t
                    for q in range(2):out[mask,q]=(probabilities[mask,q]*torch.as_tensor(self.constant['alphas'][t,q],device=x.device)).sum(-1)
            return out


def evaluate_saved(data,seed,artifact_root,arms,roles,device='cuda:0',authorization=None):
    from .prefix import load_prefix
    root=Path(artifact_root);cache,pilot=load_prefix(data,seed,root,device);freeze(pilot);out={}
    for arm in arms:
        run_dir=root/'runs'/data.dataset/arm/f'seed{seed}';heavy_dir=run_dir/'transient'
        capsule=load_checkpoint(heavy_dir/'predictor.pt');pilot.load_state_dict(capsule['pilot'])
        bank=reader=direct=candidates=None
        if capsule['bank'] is not None:
            kind=capsule['bank']['kind']
            bank=FunctionBank(pilot,kind=kind,pooled=[pilot.head(0,q) for q in range(2)] if kind=='POOL' else None);bank.load_state_dict(capsule['bank']['state']);freeze(bank.to(device))
            candidates=CandidateCache(data,cache,pilot,bank,heavy_dir/'candidate_cache',device)
        if capsule['reader'] is not None:
            reader=ReaderGroup().to(device);reader.load_state_dict(capsule['reader']['state']);reader.load_extra_state(capsule['reader']['extra']);freeze(reader)
        if capsule['direct'] is not None:
            st=capsule['direct'];direct=DirectGroup(st['input_dim'],seed,data.dataset).to(device);direct.load_state_dict(st['state'])
            direct.normalizers={k:Normalizer.from_state_dict(v) for k,v in st['normalizers'].items()};freeze(direct)
        out[arm]={}
        for role in roles:
            if arm in ('NATIVE','LOCAL'):raw=_export_raw(data,cache,pilot,arm,role,device,authorization)
            elif direct is not None:raw=_direct_predict(data,cache,pilot,direct,role,device,authorization)
            elif capsule['constant'] is not None:raw=_constant_predict(data,candidates,capsule['constant'],role,device,authorization)
            else:raw=_predict_reader(data,cache,candidates,reader,role,device,authorization)
            out[arm][role]=save_predictions(run_dir/f'{role}_selected.npz',raw['rows'],raw['y'],raw['p'],raw['domain'],role,capsule['predictor_hash'],**({'alpha':raw['alpha']} if 'alpha' in raw else {}))
    return out


def benchmark_saved(data,seed,artifact_root,arms,device='cuda:0'):
    """Fixed 4096-row complete-inference timing; repeats are not research seeds."""
    root=Path(artifact_root);rows=data.ids('fit')[:4096]
    if len(rows)!=4096:raise ValueError('Inference benchmark requires 4096 natural rows')
    x=torch.as_tensor(np.array(data.x[rows]),device=device);records={}
    for arm in arms:
        run_dir=root/'runs'/data.dataset/arm/f'seed{seed}'
        capsule=load_checkpoint(run_dir/'transient'/'predictor.pt');model=CapsulePredictor(capsule,device)
        sync=lambda:torch.cuda.synchronize() if str(device).startswith('cuda') else None
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):model(x)
            samples=[]
            for _ in range(30):
                sync();start=time.perf_counter();model(x);sync();samples.append(time.perf_counter()-start)
        native_count=sum(p.numel() for p in model.native.parameters());total=sum(p.numel() for p in model.parameters())
        record=dict(batch_size=4096,warmup=5,repetitions=30,median_seconds=float(np.median(samples)),
                    p95_seconds=float(np.quantile(samples,.95)),samples_seconds=samples,precision='FP32 model/FP64 probability mixture',
                    parameters=total,native_parameters=native_count,added_deployment_parameters=total-native_count,
                    candidate_functions_per_task=0 if arm in ('NATIVE','H_DIRECT') else 3 if arm=='ETA_DIRECT' else 1 if arm=='LOCAL' else 5,
                    readout_heads_per_task=0 if arm in ('NATIVE','LOCAL','CONST_UNIFORM') else 1,
                    cuda_device=str(device),device_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu',
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                    predictor_hash=capsule['predictor_hash'],interpretation='30 latency repetitions, not independent training seeds; concurrent load retained in scheduler logs')
        atomic_json(run_dir/'INFERENCE_BENCHMARK.json',record);records[arm]=record
        del model
    return records


def decide_stage(root,stage,dataset,seeds=(5101,5102)):
    root=Path(root);refs=[a for a in P2_ARMS if a!='DFD'] if stage=='p2' else list(P3_ARMS)
    rows=[];all_pass=True
    for ref in refs:
        gains=[];aucs=[];cells=[]
        for seed in seeds:
            path=root/'runs'/dataset/ref/f'seed{seed}'/'COMPLETE.json';dp=root/'runs'/dataset/'DFD'/f'seed{seed}'/'COMPLETE.json'
            if not path.exists() or not dp.exists():raise RuntimeError('Stage decision requires every triggered arm complete')
            a=json.loads(dp.read_text())['metrics']['Q_explore'];b=json.loads(path.read_text())['metrics']['Q_explore']
            gains.append(b['macro_logloss']-a['macro_logloss']);aucs.append(a['macro_auc']-b['macro_auc'])
            cells.append([ac['logloss']-bc['logloss'] for ac,bc in zip(a['cells'],b['cells'])])
        flags=dict(each_seed_gain_positive=all(g>0 for g in gains),mean_gain=float(np.mean(gains))>=.0005,
                   auc_protection=float(np.mean(aucs))>=-.0005,cell_protection=float(np.max(np.mean(cells,axis=0)))<=.003)
        ok=all(flags.values());all_pass&=ok
        rows.append(dict(reference=ref,ll_gains=gains,mean_ll_gain=float(np.mean(gains)),auc_deltas=aucs,
                         cell_ll_deltas=cells,worst_mean_cell_ll_delta=float(np.max(np.mean(cells,axis=0))),flags=flags,PASS=ok))
    nondegenerate=True
    if stage=='p2':
        for seed in seeds:
            stats=json.loads((root/'runs'/dataset/'DFD'/f'seed{seed}'/'weight_statistics.json').read_text())
            variable=any(max(v['std'])>1e-6 for v in stats['owners'].values())
            a=np.load(root/'runs'/dataset/'DFD'/f'seed{seed}'/'Q_explore_selected.npz')['p'];b=np.load(root/'runs'/dataset/'LOCAL'/f'seed{seed}'/'Q_explore_selected.npz')['p']
            nondegenerate&=variable and bool(np.max(np.abs(a-b))>1e-6)
        all_pass&=nondegenerate
    result=dict(stage=stage,dataset=dataset,seeds=list(seeds),comparisons=rows,nondegenerate=nondegenerate,PASS=all_pass,
                next_stage='p3' if stage=='p2' else 'p4',next_status='READY' if all_pass else 'NO_START')
    atomic_json(root/f'DECISION_{stage.upper()}_{dataset}.json',result);return result


def fixed_content_interventions(data,seed,artifact_root,device='cuda:0',role='Q_explore',authorization=None):
    """At most eight changes per cell, for both fixed final readers."""
    from .prefix import load_prefix
    from .demand import refit_eta,pilot_residual_array,candidates as pilot_candidates,analytic_mu
    from .metrics import losses,auc
    root=Path(artifact_root);cache,pilot=load_prefix(data,seed,root,device);freeze(pilot)
    capsules={arm:load_checkpoint(root/'runs'/data.dataset/arm/f'seed{seed}'/'transient'/'predictor.pt') for arm in ('DFD','UNIFORM')}
    banks={};readers={}
    for arm,cap in capsules.items():
        bank=FunctionBank(pilot,kind=cap['bank']['kind']).to(device);bank.load_state_dict(cap['bank']['state']);banks[arm]=freeze(bank)
        reader=ReaderGroup().to(device);reader.load_state_dict(cap['reader']['state']);reader.load_extra_state(cap['reader']['extra']);readers[arm]=freeze(reader)
    collected={};components={arm:[] for arm in capsules};alphas={arm:[] for arm in capsules};rows=data.ids(role);ys=[];domains=[]
    for part in _chunks(rows):
        c=cache.get(part,device)
        with torch.no_grad():
            logits={arm:bank.candidates(c['h'],c['z'],c['domain'],pilot) for arm,bank in banks.items()}
            uniform=logits['UNIFORM'].double().sigmoid()
            for arm,reader in readers.items():
                a=logits[arm];base,alpha=reader.predict(c['h'],c['z'],c['domain'],a);p=a.double().sigmoid()
                components[arm].append(p.cpu().numpy());alphas[arm].append(alpha.cpu().numpy())
                variations={'ORIGINAL':base,'LOCAL':p[...,0]}
                for source_slot in range(2):
                    changed=p.clone();start=1+2*source_slot;changed[...,start:start+2]=p[...,:1]
                    variations[f'DELETE_SOURCE_SLOT{source_slot}']=(alpha*changed).sum(-1)
                    changed=p.clone();changed[...,start]=p[...,start+1];changed[...,start+1]=p[...,start]
                    variations[f'SWAP_SOURCE_SLOT{source_slot}']=(alpha*changed).sum(-1)
                for slot in range(1,5):
                    changed=p.clone();changed[...,slot]=uniform[...,slot]
                    variations[f'UNIFORM_FUNCTION_SLOT{slot}']=(alpha*changed).sum(-1)
                for key,value in variations.items():collected.setdefault(arm+'_'+key,[]).append(value.cpu().numpy())
        b=data.batch(part,role,'cpu',purpose='evaluate',authorization=authorization);ys.append(b.y.numpy());domains.append(b.domain.numpy())
    y,d=np.concatenate(ys),np.concatenate(domains);result={};predictions={}
    dest=root/'diagnostics'/data.dataset/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    function_correlations={}
    for arm in components:
        component=np.concatenate(components[arm]);alpha=np.concatenate(alphas[arm])
        np.savez_compressed(dest/f'{role}_{arm}_components.npz',row_id=rows,domain=d,probabilities=component,alpha=alpha)
        function_correlations[arm]={}
        for t in range(3):
            for q in range(2):
                value=component[d==t,q]
                function_correlations[arm][f't{t}_q{q}']=dict(correlation=np.corrcoef(value.T).tolist(),
                    same_source_mean_absolute_differences=[float(np.mean(np.abs(value[:,1]-value[:,2]))),float(np.mean(np.abs(value[:,3]-value[:,4])))])
    for key,values in collected.items():
        p=np.concatenate(values);predictions[key]=p
        result[key]=save_predictions(dest/f'{role}_{key}.npz',rows,y,p,d,role,capsules[key.split('_')[0]]['predictor_hash'],intervention=np.asarray(key),fixed_original_alpha=np.asarray(True))
    # Apply only cutpoints frozen from the P1 OOF predictions. These scores do
    # not select interventions or change any alpha, function, or model state.
    prefix=root/'prefix'/data.dataset/f'seed{seed}'
    cuts=json.loads((prefix/'P1_OOF_RECEIPTS.json').read_text())['demand_quartile_cuts']
    eta=refit_eta(data,cache,pilot,seed,prefix,device)
    residual=pilot_residual_array(data,cache,pilot,prefix,device)
    support=data.supports(rows);groups=[];mu_rows=np.zeros((len(rows),2,2),np.float64)
    def score_group(mask,t,q,description):
        n=int(mask.sum());labels=y[mask,q]
        metrics={}
        for name,pred in predictions.items():
            arm=name.split('_')[0];base=predictions[arm+'_ORIGINAL'][mask,q]
            value=pred[mask,q]
            ll=float(losses(labels,value).mean()) if n else None
            baseline=float(losses(labels,base).mean()) if n else None
            av=auc(labels,value);ab=auc(labels,base)
            metrics[name]=dict(logloss=ll,auc=av,delta_logloss=None if n==0 else ll-baseline,
                               delta_auc=None if av is None or ab is None else av-ab)
        groups.append(dict(domain=t,task=q,n=n,positive=int(labels.sum()),negative=n-int(labels.sum()),
                           raw_users=None if data.user_ids is None else int(len(np.unique(data.user_ids[rows[mask]]))),
                           description=description,metrics=metrics))
    with torch.no_grad():
        for t in range(3):
            pos=np.flatnonzero(d==t);ids=rows[pos]
            for q in range(2):
                values=_feature_matrix(cache,ids,pilot,q,device)
                eta_p=eta.predict(values,t,q).double().cpu().numpy()
                _,aa=pilot_candidates(cache,residual,ids,t,q)
                mm=analytic_mu(aa[:,0,None],aa[:,1:],eta_p[:,None]);mu_rows[pos,q]=mm
                for k,s in enumerate(s for s in range(3) if s!=t):
                    threshold=np.asarray(cuts[f's{s}_t{t}_q{q}']);assigned=np.searchsorted(threshold,mm[:,k],side='right')
                    for group in range(4):
                        mask=np.zeros(len(rows),bool);mask[pos[assigned==group]]=True
                        score_group(mask,t,q,dict(kind='P1_OOF_DEMAND_QUARTILE',source=s,group=group,cuts=threshold))
                fit_ids=data.ids('fit',t);pfit,_=pilot_candidates(cache,residual,fit_ids,t,q)
                confidence=np.quantile(pfit[:,0],np.arange(1,10)/10,method='linear')
                peval,_=pilot_candidates(cache,residual,ids,t,q);assigned=np.searchsorted(confidence,peval[:,0],side='right')
                for group in range(10):
                    mask=np.zeros(len(rows),bool);mask[pos[assigned==group]]=True
                    score_group(mask,t,q,dict(kind='FIT_LOCAL_PILOT_CONFIDENCE_DECILE',group=group,cuts=confidence))
                if support is not None:
                    for user_known in (False,True):
                        for item_known in (False,True):
                            mask=(d==t)&(support[:,0]==user_known)&(support[:,1]==item_known)
                            score_group(mask,t,q,dict(kind='FIT_USER_ITEM_SUPPORT',user_known=user_known,item_known=item_known))
    np.savez_compressed(dest/f'{role}_demand_groups.npz',row_id=rows,domain=d,mu=mu_rows,
                        support=np.asarray([]) if support is None else support,role=np.asarray(role))
    atomic_json(dest/f'{role}_interventions.json',dict(overall=result,groups=groups,function_correlations=function_correlations,
                support_status=data.support_reason or 'raw user/item available',fixed_original_alpha=True,maximum_changes_per_cell=8))
    return result


def cleanup_intermediates(artifact_root,dataset=None,seed=None):
    """Remove this campaign's temporary models/caches after locked evaluation.

    Internal metadata-directory symlinks are unlinked; their targets are removed
    only when they resolve under this campaign. No external link is traversed.
    """
    import os,shutil
    root=Path(artifact_root).resolve()
    if root.name!='v4_0' or root.parent.name!='mdmt_dfd_v4' or Path('/home_new') in root.parents:
        raise ValueError('Cleanup is restricted to the explicit DFD-v4 campaign root')
    complete_path=root/'EXPOSED_EVALUATION_COMPLETE.json';freeze_path=root/'FROZEN_EVALUATION_SPEC.json'
    if not complete_path.is_file() or not freeze_path.is_file():
        raise RuntimeError('Cleanup requires completed dependent stages and locked exposed evaluation')
    completion=json.loads(complete_path.read_text());frozen=json.loads(freeze_path.read_text());payload=dict(frozen)
    frozen_hash=payload.pop('freeze_hash')
    if digest(payload)!=frozen_hash or completion.get('freeze_hash')!=frozen_hash:
        raise ValueError('Cleanup evaluation/freeze identity mismatch')
    for output in frozen.get('development_prefix_outputs',[]):
        if dataset is not None and output['dataset']!=dataset:continue
        if seed is not None and output['seed']!=int(seed):continue
        path=root/'prefix'/output['dataset']/f"seed{output['seed']}"/'PREFIX_RECONSTRUCTION_RECEIPT.json'
        if not path.is_file() or not json.loads(path.read_text()).get('PASS'):
            raise RuntimeError('Cleanup waits for raw-X prefix reconstruction receipt')
    def matches(path):
        rel=path.relative_to(root)
        return (dataset is None or dataset in rel.parts) and (seed is None or f'seed{int(seed)}' in rel.parts)
    targets=[];aliases=[];extra_files=[];metadata=[]
    temporary=root/'_temporary'
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink():raise ValueError('Campaign _temporary must be a local directory')
        if dataset is None and seed is None:targets.append(temporary)
        else:
            for path in temporary.rglob('seed*'):
                if path.is_dir() and not path.is_symlink() and matches(path):targets.append(path)
    for branch in ('prefix','runs'):
        base=root/branch
        if not base.exists():continue
        for current,dirs,files in os.walk(base,followlinks=False):
            current=Path(current)
            if not matches(current):continue
            for name in dirs+files:
                path=current/name
                if path.is_symlink():
                    target=path.resolve(strict=False)
                    if root in target.parents:
                        # Only aliases into selected temporary state are removed.
                        if any(target==candidate or candidate in target.parents for candidate in targets) or name in ('transient','states','cache'):
                            aliases.append(path)
                            if target.exists() and target.is_dir() and not any(target==candidate or candidate in target.parents for candidate in targets):targets.append(target)
                    continue
                if path.is_dir() and name=='transient':targets.append(path)
                elif path.is_file() and (path.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.bin') or name in ('responsibilities.npz','pilot_residuals.npy','h.npy','z.npy')):
                    extra_files.append(path)
    # Collapse nested targets and reject external paths before any mutation.
    unique=[]
    for path in sorted(set(targets),key=lambda p:len(p.parts)):
        resolved=path.resolve()
        if path.is_symlink() or root not in resolved.parents:raise ValueError('External cleanup target')
        if not any(path==p or p in path.parents for p in unique):unique.append(path)
    targets=unique
    removed=[]
    for target in targets:
        size=0
        for current,dirs,files in os.walk(target,followlinks=False):
            for name in files:
                file=Path(current)/name
                if file.is_symlink():continue
                size+=file.stat().st_size
                if file.suffix=='.json':metadata.append(dict(path=str(file.relative_to(root)),content=json.loads(file.read_text())))
        removed.append(dict(path=str(target.relative_to(root)),bytes=size,kind='temporary_directory'))
    for path in extra_files:
        if any(path==p or p in path.parents for p in targets):continue
        removed.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,kind='temporary_model_or_cache_file'))
    # Preserve cache/source identity metadata before removing the temporary bytes.
    atomic_json(root/'TRANSIENT_IDENTITY_INDEX.json',dict(freeze_hash=frozen_hash,temporary_metadata=metadata))
    for path in set(aliases):
        if path.is_symlink():
            removed.append(dict(path=str(path.relative_to(root)),bytes=0,kind='internal_temporary_alias'));path.unlink()
    for path in extra_files:
        if path.exists() and not any(path==p or p in path.parents for p in targets):path.unlink()
    for path in targets:
        if path.exists():shutil.rmtree(path)
    remaining=[]
    for current,dirs,files in os.walk(root,followlinks=False):
        for name in files:
            path=Path(current)/name
            if matches(path) and not path.is_symlink() and path.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.bin'):
                remaining.append(str(path.relative_to(root)))
    atomic_json(root/'INTERMEDIATE_CLEANUP.json',dict(freeze_hash=frozen_hash,removed=removed,
                models_remaining=remaining,final_models_retained=bool(remaining),
                scope='Only this campaign temporary targets and internal aliases; logs/predictions/config/data identities retained; external directories not followed'))
    if remaining:raise RuntimeError('Unexpected model files remain inside this campaign: '+str(remaining))
    return removed
