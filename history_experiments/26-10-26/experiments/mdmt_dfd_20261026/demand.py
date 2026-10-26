"""Fold-isolated conditional probabilities, analytic demand, and fixed weights."""
from __future__ import annotations
from pathlib import Path
import copy,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .plans import folds,batches,torch_seed
from .state import atomic_json,array_hash,digest,state_hash,save_checkpoint,load_checkpoint,capture_rng,restore_rng,module_state,restore_module,sha_file
from .readers import Normalizer,ProbabilityHead,MixtureHead,solve_simplex,stable_mixture_bce
from .metrics import evaluate_probabilities,losses,save_predictions
from .costs import CostLedger


def source_identity():
    root=Path(__file__).resolve().parent
    files={name:sha_file(root/name) for name in ('demand.py','readers.py','functions.py','plans.py','state.py','metrics.py')}
    return digest(files)


def analytic_mu(a0,a1,eta):
    if torch.is_tensor(a0):return F.softplus(a0.double())-F.softplus(a1.double())-eta.double()*(a0.double()-a1.double())
    a0=np.asarray(a0,dtype=np.float64);a1=np.asarray(a1,dtype=np.float64)
    return np.logaddexp(0,a0)-np.logaddexp(0,a1)-np.asarray(eta,dtype=np.float64)*(a0-a1)

def diagnostic_logits(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7);return np.log(p)-np.log1p(-p)

def pilot_residual_array(data,cache,pilot,run_dir,device):
    path=Path(run_dir)/'transient'/'pilot_residuals.npy';path.parent.mkdir(parents=True,exist_ok=True);identity_path=Path(run_dir)/'pilot_residuals.json';key=digest(dict(cache_hash=cache.hash,pilot_hash=state_hash(pilot.state_dict())))
    if identity_path.exists():
        if json.loads(identity_path.read_text())['key']!=key:raise ValueError('Pilot cache identity mismatch')
        return np.load(path,mmap_mode='r')
    a=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(data.x),3,2))
    with torch.no_grad():
        for i in range(0,len(data.x),4096):
            ids=np.arange(i,min(i+4096,len(data.x)));a[ids]=pilot.residuals(cache.get(ids,device)['h']).cpu().numpy()
    a.flush();atomic_json(identity_path,dict(key=key,layout='global_row/source/task'));return np.load(path,mmap_mode='r')

def features(cache,residuals,ids,q,logit_only=False):
    ids=np.asarray(ids,dtype=np.int64);a=np.concatenate((np.array(cache.z[ids]),np.array(residuals[ids,:,q])),axis=1)
    return a if logit_only else np.concatenate((np.array(cache.h[ids]),a),axis=1)

def candidates(cache,residuals,ids,target,q):
    ids=np.asarray(ids,dtype=np.int64);z=np.asarray(cache.z[ids,q],dtype=np.float64);res=np.asarray(residuals[ids,:,q],dtype=np.float64)
    probs=1/(1+np.exp(-np.clip(z[:,None]+res,-745,709)))
    own=probs[:,target];sources=[s for s in range(3) if s!=target]
    p=np.stack((own,.5*(own+probs[:,sources[0]]),.5*(own+probs[:,sources[1]])),axis=1)
    return p,diagnostic_logits(p)


def training_candidate_logits(cache,residuals,ids,target,q):
    # Exact mixture logits without diagnostic probability clipping.
    z=np.asarray(cache.z[ids,q],dtype=np.float64)
    rr=np.asarray(residuals[ids,:,q],dtype=np.float64)
    a=z[:,None]+rr;own=a[:,target]
    result=[own]
    for source in range(3):
        if source==target:continue
        lp=np.logaddexp(-np.logaddexp(0,-own),-np.logaddexp(0,-a[:,source]))-np.log(2.)
        ln=np.logaddexp(-np.logaddexp(0,own),-np.logaddexp(0,a[:,source]))-np.log(2.)
        result.append(lp-ln)
    return np.stack(result,axis=1)


def _normalizer(cache,residuals,data,target,q):return Normalizer.from_array(features(cache,residuals,data.ids('fit',target),q))

def _head_train(data,cache,residuals,target,q,train_ids,seed,component,directory,device,normalizer,kind='eta',passes=20,initial=None,ledger=None,resume=True,stop_after_pass=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);file=directory/'last.pt'
    logit_only=kind=='logit';input_dim=5 if logit_only else 261
    rate=float(np.asarray(data.y[train_ids,q]).mean())
    head=(MixtureHead(input_dim,3,torch_seed(seed,data.dataset,40,target,q,component)) if kind=='direct' else ProbabilityHead(input_dim,rate,torch_seed(seed,data.dataset,40,target,q,component))).to(device)
    opt=torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    ident=dict(dataset=data.dataset,seed=int(seed),data_hash=data.data_hash,cache_hash=cache.hash,train_row_hash=array_hash(np.sort(train_ids)),target=target,task=q,kind=kind,component=component,normalizer_hash=digest(normalizer.state_dict()),source_hash=source_identity(),pilot_feature_key=next(json.loads((p/'pilot_residuals.json').read_text())['key'] for p in Path(directory).parents if (p/'pilot_residuals.json').exists()))
    pass_id=1;cursor=0
    if file.exists():
        if not resume:raise FileExistsError('Existing demand state requires resume')
        s=load_checkpoint(file,ident);restore_module(head,opt,s['models']['head']);restore_rng(s['rng']);pass_id=s['pass'];cursor=s['next_logical_step']
    elif initial is not None:restore_module(head,opt,initial)
    role='P_form';purpose='eta' if kind!='direct' else 'reader'
    while pass_id<=passes:
        plan=batches(train_ids,seed,data.dataset,41,pass_id,target,q,component,batch_size=512)
        cm=ledger.measure('eta_pass' if kind!='direct' else 'p1_direct_pass',kind=kind,target=target,task=q,rng_component=component,pass_id=pass_id) if ledger else __import__('contextlib').nullcontext()
        with cm:
            for step in range(cursor,len(plan)):
                ids=plan[step];b=data.batch(ids,role,device,purpose=purpose);x=features(cache,residuals,ids,q,logit_only)
                xx=torch.as_tensor(normalizer.transform(x),device=device,dtype=torch.float32)
                opt.zero_grad(set_to_none=True);out=head(xx)
                if kind=='direct':
                    aa=training_candidate_logits(cache,residuals,ids,target,q);loss=stable_mixture_bce(torch.as_tensor(aa,device=device),out,b.y[:,q])
                else:loss=F.binary_cross_entropy_with_logits(out.reshape(-1),b.y[:,q])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite eta loss')
                loss.backward();opt.step()
                if ledger:ledger.access('eta' if kind!='direct' else 'p1_direct','P_form',len(ids),tasks=1,target=target,task=q,rng_component=component,pass_id=pass_id)
                save_checkpoint(file,dict(identity=ident,stage=kind,**{'pass':pass_id},next_logical_step=step+1,models={'head':module_state(head,opt)},optimizers={'head':opt.state_dict()},rng=capture_rng(),normalizer=normalizer.state_dict(),row_plan_hash=array_hash(np.concatenate(plan))))
        save_checkpoint(file,dict(identity=ident,stage=kind,**{'pass':pass_id+1},next_logical_step=0,models={'head':module_state(head,opt)},optimizers={'head':opt.state_dict()},rng=capture_rng(),normalizer=normalizer.state_dict(),complete=pass_id==passes))
        if stop_after_pass and pass_id>=stop_after_pass:break
        pass_id+=1;cursor=0
    head.eval();return head,opt,ident


def _predict_head(head,normalizer,cache,residuals,ids,q,device,kind='eta'):
    values=[]
    with torch.no_grad():
        for i in range(0,len(ids),4096):
            subset=ids[i:i+4096];x=features(cache,residuals,subset,q,kind=='logit');out=head(torch.as_tensor(normalizer.transform(x),device=device,dtype=torch.float32))
            values.append((out if kind=='direct' else out.sigmoid()).cpu().numpy())
    return np.concatenate(values)


def train_eta_and_oof(data,cache,pilot,seed,run_dir,device='cuda:0',resume=True,stop_after=None):
    run_dir=Path(run_dir);file=run_dir/'p1_oof.npz'
    oof_identity=dict(source_hash=source_identity(),cache_hash=cache.hash,pilot_hash=state_hash(pilot.state_dict()),data_hash=data.data_hash,seed=int(seed))
    completion=run_dir/'P1_OOF_COMPLETE.json'
    if file.exists() and completion.exists():
        try:saved=json.loads(completion.read_text())
        except (OSError,json.JSONDecodeError):saved=None
        if saved is not None:
            if saved['identity']!=oof_identity:raise ValueError('OOF complete identity mismatch')
            if saved['npz_sha256']==sha_file(file):return dict(status='COMPLETE',path=str(file))
    # A partial/corrupt export is rebuilt from existing complete head states.
    # No completed fold training is repeated; _head_train resumes each same identity.
    residuals=pilot_residual_array(data,cache,pilot,run_dir,device);rows=data.ids('P_form');fold=folds(data);n=len(rows)
    eta=np.zeros((n,2));mu=np.zeros((n,2,2));cand=np.zeros((n,2,3));actions=np.zeros((n,2),dtype=np.int8);logit_actions=actions.copy();eta_logit=eta.copy();mu_logit=mu.copy()
    pred={k:np.zeros((n,2)) for k in ('CONST','LOGIT','DEMAND','DIRECT')};receipts=[];head_receipts=[];quartiles={};d_stats={};clip_stats={};fold_const_d=np.zeros((n,2,2))
    ledger=CostLedger(run_dir/'costs.jsonl',dict(dataset=data.dataset,seed=int(seed),stage='p1'))
    for t in range(3):
        pos=np.flatnonzero(data.domain[rows]==t);target_ids=rows[pos]
        for q in range(2):
            norm=_normalizer(cache,residuals,data,t,q);ln=Normalizer.from_state_dict({k:(np.asarray(v)[256:] if k in ('mean','std') else v) for k,v in norm.state_dict().items()})
            atomic_json(run_dir/f'eta_normalizer_s{t}_q{q}.json',norm.state_dict())
            for heldout in (0,1):
                train=target_ids[fold[pos]!=heldout];held=target_ids[fold[pos]==heldout];outpos=np.searchsorted(rows,held);tc,ta=candidates(cache,residuals,train,t,q);hc,ha=candidates(cache,residuals,held,t,q)
                data._check(train,'P_form','constant');alpha,rec=solve_simplex(tc,np.asarray(data.y[train,q]));pred['CONST'][outpos,q]=hc@alpha
                rr=dict(target=t,task=q,heldout_fold=heldout,train_row_hash=array_hash(train),heldout_row_hash=array_hash(held),intersection=0 if not np.intersect1d(train,held).size else 'ERROR');rr.update(rec);receipts.append(rr)
                train_d=analytic_mu(ta[:,0,None],ta[:,1:],np.asarray(data.y[train,q])[:,None]);fold_const_d[outpos,q]=train_d.mean(0)
                for kind,component,nrm in [('eta',heldout,norm),('logit',10+heldout,ln),('direct',20+heldout,norm)]:
                    head,opt,ident=_head_train(data,cache,residuals,t,q,train,seed,component,run_dir/'transient'/'oof_states'/f's{t}_q{q}_{kind}_hold{heldout}',device,nrm,kind=kind,ledger=ledger,resume=resume,stop_after_pass=1 if stop_after=='eta1' else None)
                    if stop_after=='eta1':return dict(status='RETAINED_BOUNDARY',component='eta_first_fold_pass1')
                    head_receipts.append(dict(identity=ident,state_hash=state_hash(head.state_dict()),optimizer_hash=state_hash(opt.state_dict()),train_row_hash=array_hash(train),heldout_row_hash=array_hash(held),heldout_fold=heldout))
                    value=_predict_head(head,nrm,cache,residuals,held,q,device,kind)
                    if kind=='direct':
                        weights=torch.softmax(torch.as_tensor(value,dtype=torch.float64),dim=1).numpy();pred['DIRECT'][outpos,q]=(weights*hc).sum(1)
                    else:
                        value=value.reshape(-1);mm=analytic_mu(ha[:,0,None],ha[:,1:],value[:,None]);act=np.argmax(np.column_stack((np.zeros(len(held)),mm)),axis=1);pred['DEMAND' if kind=='eta' else 'LOGIT'][outpos,q]=hc[np.arange(len(held)),act]
                        if kind=='eta':eta[outpos,q]=value;mu[outpos,q]=mm;actions[outpos,q]=act
                        else:eta_logit[outpos,q]=value;mu_logit[outpos,q]=mm;logit_actions[outpos,q]=act
                cand[outpos,q]=hc
            cc,aa=candidates(cache,residuals,target_ids,t,q);d=analytic_mu(aa[:,0,None],aa[:,1:],np.asarray(data.y[target_ids,q])[:,None]);src=[s for s in range(3) if s!=t]
            for k,s in enumerate(src):
                key=f's{s}_t{t}_q{q}';quartiles[key]=np.quantile(mu[pos,q,k],[.25,.5,.75]).tolist();d_stats[key]=dict(mean=float(d[:,k].mean()),variance=float(d[:,k].var()),oof_mse=float(np.mean((mu[pos,q,k]-d[:,k])**2)),train_fold_constant_mse=float(np.mean((fold_const_d[pos,q,k]-d[:,k])**2)))
            values=features(cache,residuals,target_ids,q);trans=(values-np.asarray(norm.mean))/np.asarray(norm.std);clip_stats[f's{t}_q{q}']=dict(per_dimension=(np.abs(trans)>10).mean(0).tolist(),mean=float((np.abs(trans)>10).mean()))
    y=np.asarray(data.y[rows]);domain=data.domain[rows];metrics={k:evaluate_probabilities(y,v,domain) for k,v in pred.items()}
    checkpoint_hash=digest(dict(cache_hash=cache.hash,pilot_hash=state_hash(pilot.state_dict()),fold_hash=array_hash(fold),receipts=receipts,head_receipts=head_receipts))
    np.savez_compressed(file,row_id=rows,rows=rows,fold=fold,eta=eta,mu=mu,eta_logit=eta_logit,mu_logit=mu_logit,candidates=cand,actions=actions,logit_actions=logit_actions,y=y,domain=domain,role=np.asarray('P_form_OOF'),checkpoint_hash=np.asarray(checkpoint_hash),train_fold_constant_d=fold_const_d,**{'p_'+k:v for k,v in pred.items()},**{'loss_'+k:losses(y,v) for k,v in pred.items()})
    atomic_json(run_dir/'P1_METRICS.json',dict(dataset=data.dataset,seed=int(seed),metrics=metrics,checkpoint_hash=checkpoint_hash,role='P_form_OOF'))
    atomic_json(run_dir/'P1_OOF_RECEIPTS.json',dict(train_fold_simplex=receipts,demand_quartile_cuts=quartiles,difference_statistics=d_stats,normalizer_clip_rates=clip_stats,fold_hash=array_hash(fold),head_receipts=head_receipts))
    atomic_json(run_dir/'P1_OOF_COMPLETE.json',dict(identity=oof_identity,npz_sha256=sha_file(file),checkpoint_hash=checkpoint_hash))
    return dict(status='COMPLETE',path=str(file),metrics=metrics)

class DemandModel(nn.Module):
    def __init__(self):
        super().__init__();self.heads=nn.ModuleDict();self.normalizers={};self.optimizers={}
    def predict(self,x,target,task):
        key=f's{target}_q{task}';head=self.heads[key];device=next(head.parameters()).device
        return head(torch.as_tensor(self.normalizers[key].transform(x),device=device,dtype=torch.float32)).sigmoid().reshape(-1)

def refit_eta(data,cache,pilot,seed,run_dir,device='cuda:0'):
    run_dir=Path(run_dir);residuals=pilot_residual_array(data,cache,pilot,run_dir,device);model=DemandModel();ledger=CostLedger(run_dir/'costs.jsonl',dict(dataset=data.dataset,seed=int(seed),stage='eta_refit'))
    for t in range(3):
        for q in range(2):
            key=f's{t}_q{q}';norm=_normalizer(cache,residuals,data,t,q)
            head,opt,_=_head_train(data,cache,residuals,t,q,data.ids('P_form',t),seed,2,run_dir/'transient'/'eta_refit'/key,device,norm,ledger=ledger)
            model.heads[key]=head;model.normalizers[key]=norm;model.optimizers[key]=opt
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    return model

def make_responsibilities(data,cache,pilot,eta,seed,run_dir,device='cuda:0'):
    run_dir=Path(run_dir);path=run_dir/'responsibilities.npz'
    responsibility_identity=dict(source_hash=source_identity(),cache_hash=cache.hash,pilot_hash=state_hash(pilot.state_dict()),eta_hash=state_hash(eta.state_dict()),normalizers_hash=digest({k:v.state_dict() for k,v in eta.normalizers.items()}),oof_sha256=sha_file(run_dir/'p1_oof.npz'),fit_row_hash=array_hash(data.ids('fit')))
    if path.exists():
        saved=json.loads((run_dir/'RESPONSIBILITY_COMPLETE.json').read_text())
        if saved['identity']!=responsibility_identity or saved['npz_sha256']!=sha_file(path):raise ValueError('Responsibility cache dependency mismatch')
        with np.load(path) as a:return {k:a[k] for k in a.files}
    oof=np.load(run_dir/'p1_oof.npz');residuals=pilot_residual_array(data,cache,pilot,run_dir,device);m=np.zeros((3,3,2));tau=np.zeros((3,2));bs={}
    for t in range(3):
        ids=data.ids('P_form',t);pos=np.searchsorted(oof['rows'],ids)
        for q in range(2):
            pp,aa=candidates(cache,residuals,ids,t,q);dd=analytic_mu(aa[:,0,None],aa[:,1:],np.asarray(data.y[ids,q])[:,None])
            for k,s in enumerate(s for s in range(3) if s!=t):m[s,t,q]=dd[:,k].mean();bs[s,t,q]=oof['mu'][pos,q,k]-m[s,t,q]
    for s in range(3):
        targets=[t for t in range(3) if t!=s]
        for q in range(2):
            arrays=[bs[s,t,q] for t in targets];mean=.5*sum(a.mean() for a in arrays);variance=.5*sum(np.mean((a-mean)**2) for a in arrays);tau[s,q]=max(.001,float(np.sqrt(variance)))
    fit=data.ids('fit');w=np.empty((len(fit),2,2),dtype=np.float32);stats={}
    with torch.no_grad():
        for s in range(3):
            pos=np.flatnonzero(data.domain[fit]==s);ids=fit[pos];targets=[t for t in range(3) if t!=s]
            for q in range(2):
                bvals=np.empty((len(ids),2));clips={}
                for k,t in enumerate(targets):
                    mus=[];clip_count=np.zeros(261);total=0
                    for i in range(0,len(ids),4096):
                        sub=ids[i:i+4096];x=features(cache,residuals,sub,q);key=f's{t}_q{q}';norm=eta.normalizers[key];raw=(x-norm.mean)/norm.std;clip_count+=(np.abs(raw)>10).sum(0);total+=len(sub)
                        probability=eta.predict(x,t,q).double().cpu().numpy();_,aa=candidates(cache,residuals,sub,t,q);sourceidx=[j for j in range(3) if j!=t].index(s)
                        mus.append(analytic_mu(aa[:,0],aa[:,sourceidx+1],probability))
                    values=np.concatenate(mus);bvals[:,k]=values-m[s,t,q];clips[str(t)]=dict(mean=float((clip_count/total).mean()),per_dimension=(clip_count/total).tolist(),mu_mean=float(values.mean()),mu_sd=float(values.std()),mu_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist())
                scaled=bvals/tau[s,q];scaled-=scaled.max(1,keepdims=True);e=np.exp(scaled);ww=.1+1.8*e/e.sum(1,keepdims=True);w[pos,q]=ww
                ownp=1/(1+np.exp(-np.asarray(cache.z[ids,q],dtype=np.float64)-np.asarray(residuals[ids,s,q],dtype=np.float64)));bounds=np.quantile(ownp,np.arange(1,10)/10);bins=np.searchsorted(bounds,ownp,side='right');strata=[]
                for label in (0,1):
                    for bin_id in range(10):
                        mask=(np.asarray(data.y[ids,q])==label)&(bins==bin_id);part=ww[mask];strata.append(dict(label=label,confidence_bin=bin_id,n=int(mask.sum()),mean=part.mean(0).tolist() if len(part) else None,mass=part.sum(0).tolist(),ess=(part.sum(0)**2/(part**2).sum(0)).tolist() if len(part) else None))
                stats[f's{s}_q{q}']=dict(targets=targets,weight_mean=ww.mean(0).tolist(),weight_sd=ww.std(0).tolist(),mass=ww.sum(0).tolist(),ess=(ww.sum(0)**2/(ww**2).sum(0)).tolist(),weight_correlation=float(np.corrcoef(ww.T)[0,1]) if ww[:,0].std()>0 and ww[:,1].std()>0 else None,centered_mu_quantiles=np.quantile(bvals,[0,.25,.5,.75,1],axis=0).tolist(),source_support_clips=clips,confidence_decile_bounds=bounds.tolist(),label_confidence_strata=strata)
    if np.max(np.abs(w.sum(2)-2))>1e-6:raise ValueError('Responsibility conservation error')
    result=dict(fit_row_id=fit,w=w,m=m,tau=tau);np.savez_compressed(path,**result);atomic_json(run_dir/'RESPONSIBILITY_SUPPORT.json',stats);atomic_json(run_dir/'RESPONSIBILITY_COMPLETE.json',dict(identity=responsibility_identity,npz_sha256=sha_file(path)));return result
