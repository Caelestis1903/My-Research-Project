"""Strong direct models and fixed-content controls on the common prefix."""
from __future__ import annotations
import copy,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from .protocol import FeatureBank,PredictionBundle,derive_rng,torch_seed,folds,evaluate,record_access,OUTER
from .state import freeze,state_hash,digest,array_hash
from .memory_data import persist,map_prediction,release_data_pages

class DirectHead(nn.Module):
    def __init__(self,mean,std,seed,positive_rate):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed));self.hidden=nn.Linear(len(mean),32);self.output=nn.Linear(32,1)
            nn.init.xavier_uniform_(self.hidden.weight);nn.init.zeros_(self.hidden.bias);nn.init.zeros_(self.output.weight)
            p=float(np.clip(positive_rate,1e-4,1-1e-4));nn.init.constant_(self.output.bias,np.log(p/(1-p)))
    def forward(self,x):return self.output(F.relu(self.hidden(((x-self.mean)/self.std).clamp(-10,10)))).squeeze(-1)

def input_matrix(bundle,role,task,eta=False,h=None,z=None):
    v=bundle.views[role];h=v.h if h is None else h;z=v.z if z is None else z
    pieces=[h,z]
    if eta:pieces.extend(bundle.pilots.residual(role,s,task)[:,None] for s in range(3))
    return np.concatenate(pieces,axis=1).astype(np.float32)

def _new_head(bundle,target,task,eta,training_y,unit=2):
    raw=input_matrix(bundle,'fit',task,eta);x=raw[bundle.views['fit'].domain==target]
    mean=x.mean(0,dtype=np.float64);std=np.maximum(x.std(0,dtype=np.float64),1e-3)
    model=DirectHead(mean,std,torch_seed(bundle.seed,bundle.dataset,'base_init',target,task,unit),np.mean(training_y)).to(bundle.device)
    return model,torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)

def _predict_head(model,raw,chunk=40960):
    dev=next(model.parameters()).device;out=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(raw),chunk):out.append(model(torch.as_tensor(np.array(raw[i:i+chunk]),device=dev)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0,dtype=np.float32)

def _fit_head(bundle,model,opt,role,target,task,passes,method,unit,eta=False,train_positions=None,stage='base_plan',snapshots=()):
    view=bundle.views[role];positions=np.flatnonzero(view.domain==target) if train_positions is None else np.asarray(train_positions)
    raw=input_matrix(bundle,role,task,eta);records=[];saved={}
    if 0 in snapshots:saved[0]=freeze(copy.deepcopy(model).cpu())
    for pass_id in range(1,passes+1):
        order=derive_rng(bundle.seed,bundle.dataset,stage,target,task,unit,pass_id).permutation(positions);loss_sum=0.
        model.train()
        for begin in range(0,len(order),512):
            idx=order[begin:begin+512];record_access(bundle,method,role,view.row_id[idx],task,purpose='base' if role=='P_form' and method.startswith('B_') else 'direct',pass_id=pass_id)
            opt.zero_grad(set_to_none=True);x=torch.as_tensor(np.array(raw[idx]),device=bundle.device);y=torch.as_tensor(np.array(view.y[idx,task]),device=bundle.device)
            loss=F.binary_cross_entropy_with_logits(model(x),y);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(idx)
        records.append(dict(pass_id=pass_id,rows=len(order),steps=int(np.ceil(len(order)/512)),loss=loss_sum/max(1,len(order)),row_plan_hash=array_hash(view.row_id[order])))
        if pass_id in snapshots:saved[pass_id]=freeze(copy.deepcopy(model).cpu())
    return records,saved

def fit_base(bundle,config):
    if bundle.base_logits:return
    pf=bundle.views['P_form'];ff=folds(bundle.data);oof=np.empty_like(pf.z);models={};records=[];started=time.monotonic()
    for t in range(3):
        record_access(bundle,'B_NORMALIZER','fit',bundle.views['fit'].row_id[bundle.views['fit'].domain==t],purpose='normalizer')
        for q in range(2):
            own=np.flatnonzero(pf.domain==t)
            for fold in (0,1):
                train=own[ff[own]!=fold];held=own[ff[own]==fold]
                model,opt=_new_head(bundle,t,q,False,pf.y[train,q],unit=fold)
                rec,_=_fit_head(bundle,model,opt,'P_form',t,q,20,'B_OOF',fold,train_positions=train);records.append(dict(target=t,task=q,fold=fold,passes=rec))
                oof[held,q]=_predict_head(model,input_matrix(bundle,'P_form',q)[held]);del model,opt
            model,opt=_new_head(bundle,t,q,False,pf.y[own,q],unit=2)
            rec,_=_fit_head(bundle,model,opt,'P_form',t,q,20,'B_REFIT',2);records.append(dict(target=t,task=q,fold='REFIT',passes=rec))
            freeze(model);models[t,q]=(model,opt)
    bundle.base_oof=persist(bundle,('base_oof','P_form'),oof);bundle.base_models=models
    bundle.normalizers['base']={f'{t}_{q}':dict(mean=m.mean.detach().cpu().numpy(),std=m.std.detach().cpu().numpy()) for (t,q),(m,o) in models.items()}
    bundle.normalizers['pilot_phi']=copy.deepcopy(bundle.pilots.phi_normalizers);bundle.normalizers['local_phi']=copy.deepcopy(bundle.pilots.local_normalizers)
    bundle.base_logits={role:persist(bundle,('base_refit',role),predict_direct_models(bundle,models,role)) for role in bundle.views}
    bundle.diagnostics['base_records']=records;bundle.diagnostics['base_seconds']=time.monotonic()-started
    bundle.identity['base_hash']=digest({f'{t}_{q}':state_hash(m.state_dict()) for (t,q),(m,o) in models.items()})

def predict_direct_models(bundle,models,role,eta=False,raw_h=None,raw_z=None,raw_source=None):
    view=bundle.views[role];h=view.h if raw_h is None else raw_h;z=view.z if raw_z is None else raw_z;out=np.empty((len(h),2),np.float32)
    for t in range(3):
        positions=np.flatnonzero(view.domain==t)
        for q in range(2):
            model=models[t,q][0] if isinstance(models[t,q],tuple) else models[t,q]
            if eta and raw_source is None:residuals=[bundle.pilots.residual(role,s,q) for s in range(3)]
            for begin in range(0,len(positions),40960):
                pos=positions[begin:begin+40960];pieces=[h[pos],z[pos]]
                if eta:pieces.extend((raw_source[pos,s,q] if raw_source is not None else residuals[s][pos])[:,None] for s in range(3))
                out[pos,q]=_predict_head(model,np.concatenate(pieces,axis=1).astype(np.float32))
    return out

def _raw_hidden(head,h):
    dev=next(head.parameters()).device;parts=[]
    with torch.no_grad():
        for begin in range(0,len(h),40960):parts.append(F.relu(head.hidden(torch.as_tensor(np.array(h[begin:begin+40960]),device=dev))).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0,32),dtype=np.float32)

def _raw_source(bundle,h,initial=False,local_replica=None,standardized=False):
    """Canonical observer residual reconstruction; training paths are unchanged."""
    from .raw_reconstruction_observer import raw_source
    return raw_source(bundle,h,initial=initial,local_replica=local_replica,standardized=standardized)

def _select_family(bundle,method,candidates,terminal,models,eta=False,records=None):
    pc=bundle.views['P_cal'];record_access(bundle,method,'P_cal',pc.row_id,purpose='select')
    scores={key:evaluate(pc.y,expit(pred['P_cal']),pc.domain)['macro_logloss'] for key,pred in candidates.items()}
    chosen=min(scores,key=lambda k:(scores[k],k));p={r:expit(z) for r,z in candidates[chosen].items()};term={r:expit(z) for r,z in terminal.items()}
    selected_models=models[chosen]
    bundle.normalizers[method]={f'{t}_{q}':dict(mean=m.mean.detach().cpu().numpy(),std=m.std.detach().cpu().numpy()) for (t,q),m in selected_models.items()}
    result=PredictionBundle(method,p,term,dict(selected_pass=int(chosen),candidates=[dict(pass_id=int(k),p_cal_macro_ll=v) for k,v in scores.items()],selection_role='P_cal',selection_unit='whole_six_heads'),dict(records=records or [],models=selected_models,terminal_models=models[max(models)],eta=eta),dict(bundle.identity,method=method,selected_pass=int(chosen)))
    def validate(b):return validate_raw_predictor(b,result,lambda role,h,z,source:expit(predict_direct_models(b,selected_models,role,eta,raw_h=h,raw_z=z,raw_source=source)))
    result.diagnostics['_raw_x_validate']=validate;return map_prediction(bundle,result)

def fit_direct_baselines(bundle,config):
    fit_base(bundle,config);results={};started=time.monotonic()
    for method,eta,effort in [('H_DIRECT',False,False),('ETA_DIRECT_MATCHED',True,False),('H_DIRECT_EFFORT',False,True)]:
        family={0:{},5:{},10:{}};records=[]
        for t in range(3):
            for q in range(2):
                if not eta and not effort:
                    old,oldopt=bundle.base_models[t,q];model=copy.deepcopy(old).to(bundle.device).requires_grad_(True);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0);opt.load_state_dict(copy.deepcopy(oldopt.state_dict()))
                else:
                    pf=bundle.views['P_form'];own=pf.domain==t;model,opt=_new_head(bundle,t,q,eta,pf.y[own,q],unit=2)
                    rec,_=_fit_head(bundle,model,opt,'P_form',t,q,60 if effort else 20,method,2,eta);records.append(dict(target=t,task=q,stage='form',passes=rec))
                rec,saved=_fit_head(bundle,model,opt,'P_read',t,q,10,method,2,eta,stage='direct_plan',snapshots=(0,5,10));records.append(dict(target=t,task=q,stage='read',passes=rec))
                for k,m in saved.items():family[k][t,q]=m
                del model,opt
        predictions={k:{r:predict_direct_models(bundle,mods,r,eta) for r in bundle.views} for k,mods in family.items()}
        results[method]=_select_family(bundle,method,predictions,predictions[10],family,eta,records)
    bundle.diagnostics['direct_seconds']=time.monotonic()-started;return results

def fixed_feature_bank(bundle,method,config):
    fit_base(bundle,config);values={};columns={};bounds={}
    for t in range(3):
        for q in range(2):
            sources=[s for s in range(3) if s!=t]
            if method=='GROUP_CAL':cols=[]
            elif method=='PILOT_LOGIT':cols=[dict(function_id=f'pilot_s{s}_minus_t{t}_q{q}',source=s,task=q,slot=0) for s in sources]
            else:cols=[dict(function_id=f'{method}_s{s}_q{q}_j{j}',source=s,task=q,slot=j) for s in (range(2) if method=='FULL_TARGET' else sources) for j in range(32)]
            columns[t,q]=cols;bounds[t,q]=[(-1.,1.) if method=='PILOT_LOGIT' else (None,None) for c in cols]
            for role,v in bundle.views.items():
                mask=v.domain==t
                if method=='GROUP_CAL':a=np.empty((int(mask.sum()),0))
                elif method=='PILOT_LOGIT':a=np.stack([bundle.pilots.residual(role,s,q)[mask]-bundle.pilots.residual(role,t,q)[mask] for s in sources],1)
                elif method=='FULL_TARGET':a=np.concatenate([bundle.pilots.local_hidden(role,t,q,k)[mask] for k in range(2)],1)
                elif method in ('FULL_SOURCE','FULL_SOURCE_ALL','RANDOM_FEATURE'):a=np.concatenate([bundle.pilots.hidden(role,s,q,initial=method=='RANDOM_FEATURE')[mask] for s in sources],1)
                else:raise ValueError('Unknown fixed content '+method)
                values[role,t,q]=a
    bank=FeatureBank(values,columns,bounds,dict(bundle.identity,method=method));bank.fixed_method=method;return bank

def raw_features(bundle,bank,role,h,z,domain):
    """Canonical full-role observer from independent raw-X H/Z."""
    from .raw_reconstruction_observer import raw_features as reconstruct
    return reconstruct(bundle,bank,role,h,z,domain)

def _independent_raw_packets(bundle,role,chunk=4096):
    from pathlib import Path
    cache=getattr(bundle,'_raw_validation_packets',None)
    native_hash=bundle.identity.get('native_hash')
    if native_hash is None:native_hash=state_hash(bundle.native.state_dict())
    if cache is None:cache={};bundle._raw_validation_packets=cache
    key=(native_hash,role)
    if key in cache:return cache[key]
    # Only input-derived data arrays are written. This is never a training state.
    root=Path(bundle.output_root)/'_raw_x_verification'/native_hash[:16];root.mkdir(parents=True,exist_ok=True)
    view=bundle.views[role];hpath=root/(role+'_h.npy');zpath=root/(role+'_z.npy')
    h=np.lib.format.open_memmap(hpath,mode='w+',dtype=np.float32,shape=(len(view.row_id),256));z=np.lib.format.open_memmap(zpath,mode='w+',dtype=np.float32,shape=(len(view.row_id),2))
    native=bundle.native;device=next(native.parameters()).device
    with torch.no_grad():
        for start in range(0,len(view.row_id),chunk):
            ids=view.row_id[start:start+chunk];packet=native.encode(torch.as_tensor(np.array(bundle.data.x[ids]),device=device))
            h[start:start+len(ids)]=packet.h.cpu().numpy();z[start:start+len(ids)]=packet.logits.cpu().numpy()
    h.flush();z.flush();cache[key]=(h,z);return h,z

def release_raw_validation_cache(bundle):
    from pathlib import Path
    cache=getattr(bundle,'_raw_validation_packets',{})
    paths=[]
    for arrays in cache.values():
        for a in arrays:
            paths.append(Path(a.filename));a.flush()
            if getattr(a,'_mmap',None) is not None:a._mmap.close()
    cache.clear()
    for path in paths:
        if '_raw_x_verification' not in path.parts:raise ValueError('verification cleanup outside owned data root')
        path.unlink(missing_ok=True)
    return dict(released='independent raw-X verification data only',files=len(paths),model_files=0)

def validate_raw_predictor(bundle,result,predictor,chunk=4096):
    """Full-row raw-X→native→frozen downstream reconstruction; no labels used."""
    maximum=0.;rows=0;per_role={}
    for role,view in bundle.views.items():
        h,z=_independent_raw_packets(bundle,role,chunk)
        source=_raw_source(bundle,h) if result.diagnostics.get('eta') else None
        p=predictor(role,h,z,source);error=float(np.max(np.abs(p-result.probabilities[role])));per_role[role]=error;maximum=max(maximum,error);rows+=len(p)
    if maximum>1e-6:raise ValueError(f'Raw-X complete predictor mismatch {result.method}: {maximum}')
    return dict(status='PASS',rows=rows,max_probability_error=maximum,per_role=per_role,scope='independent raw-X native and frozen downstream model evaluation',raw_native_sweeps='once per role/native hash then reuse independent reconstruction data',labels_used=False)

class ProbabilityGate(nn.Module):
    def __init__(self,mean,std,seed):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed));self.hidden=nn.Linear(261,32);self.output=nn.Linear(32,3)
            nn.init.xavier_uniform_(self.hidden.weight);nn.init.zeros_(self.hidden.bias);nn.init.zeros_(self.output.weight)
            self.output.bias.data.copy_(torch.log(torch.tensor([.9,.05,.05])))
    def forward(self,x):return self.output(F.relu(self.hidden(((x-self.mean)/self.std).clamp(-10,10))))

def mixture_inputs(bundle,role,target,task,*,raw_h=None,raw_z=None,raw_source=None):
    view=bundle.views[role];mask=view.domain==target;h=view.h if raw_h is None else raw_h;z=view.z if raw_z is None else raw_z
    if raw_h is None:base=bundle.base_logits[role]
    else:base=predict_direct_models(bundle,bundle.base_models,role,raw_h=h,raw_z=z)
    sources=[s for s in range(3) if s!=target]
    candidates=np.column_stack([base[mask,task]]+[z[mask,task]+(bundle.pilots.residual(role,s,task)[mask] if raw_source is None else raw_source[mask,s,task]) for s in sources]).astype(np.float32)
    features=np.concatenate([h[mask],z[mask],expit(candidates)],1).astype(np.float32)
    return features,candidates

def predict_mixture_models(bundle,models,role,*,raw_h=None,raw_z=None,raw_source=None):
    from .vendor_snapshot.mdmt_dfd_20261026.readers import mixture_probability
    view=bundle.views[role];out=np.empty((len(view.row_id),2),dtype=np.float64)
    for target in range(3):
        pos=np.flatnonzero(view.domain==target)
        for task in range(2):
            x,a=mixture_inputs(bundle,role,target,task,raw_h=raw_h,raw_z=raw_z,raw_source=raw_source)
            model=models[target,task];device=next(model.parameters()).device;values=[]
            with torch.no_grad():
                for begin in range(0,len(x),40960):
                    xx=torch.as_tensor(np.array(x[begin:begin+40960]),device=device);aa=torch.as_tensor(a[begin:begin+40960],device=device)
                    values.append(mixture_probability(aa,model(xx)).cpu().numpy())
            out[pos,task]=np.concatenate(values) if values else []
    return out

def fit_probability_reader(bundle,config):
    from .vendor_snapshot.mdmt_dfd_20261026.readers import stable_mixture_bce
    fit_base(bundle,config);families={0:{},5:{},10:{}};records=[];started=time.monotonic()
    view=bundle.views['P_read']
    for t in range(3):
        own=np.flatnonzero(view.domain==t)
        for q in range(2):
            fitx,_=mixture_inputs(bundle,'fit',t,q);mean=fitx.mean(0,dtype=np.float64);std=np.maximum(fitx.std(0,dtype=np.float64),1e-3)
            record_access(bundle,'PILOT_PROB_NORMALIZER','fit',bundle.views['fit'].row_id[bundle.views['fit'].domain==t],q,purpose='normalizer')
            model=ProbabilityGate(mean,std,torch_seed(bundle.seed,bundle.dataset,'prob_init',t,q)).to(bundle.device);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
            raw,candidates=mixture_inputs(bundle,'P_read',t,q);families[0][t,q]=freeze(copy.deepcopy(model).cpu());head_records=[]
            for pass_id in range(1,11):
                order=derive_rng(bundle.seed,bundle.dataset,'prob_plan',t,q,pass_id=pass_id).permutation(len(own));loss_sum=0.
                for begin in range(0,len(order),512):
                    loc=order[begin:begin+512];rows=own[loc]
                    record_access(bundle,'PILOT_PROB','P_read',view.row_id[rows],q,purpose='reader',pass_id=pass_id)
                    optimizer.zero_grad(set_to_none=True);x=torch.as_tensor(np.array(raw[loc]),device=bundle.device);a=torch.as_tensor(candidates[loc],device=bundle.device);y=torch.as_tensor(np.array(view.y[rows,q]),device=bundle.device)
                    loss=stable_mixture_bce(a,model(x),y);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(loc)
                head_records.append(dict(pass_id=pass_id,loss=loss_sum/max(1,len(own)),rows=len(own),steps=int(np.ceil(len(own)/512)),row_hash=array_hash(view.row_id[own[order]])))
                if pass_id in (5,10):families[pass_id][t,q]=freeze(copy.deepcopy(model).cpu())
            records.append(dict(target=t,task=q,passes=head_records));del model,optimizer
    allp={k:{role:predict_mixture_models(bundle,models,role) for role in bundle.views} for k,models in families.items()}
    pc=bundle.views['P_cal'];record_access(bundle,'PILOT_PROB','P_cal',pc.row_id,purpose='select')
    scores={k:evaluate(pc.y,p['P_cal'],pc.domain)['macro_logloss'] for k,p in allp.items()};selected=min(scores,key=lambda k:(scores[k],k));models=families[selected]
    result=PredictionBundle('PILOT_PROB',allp[selected],allp[10],dict(selected_pass=int(selected),unit='complete_six_head_policy',role='P_cal',candidate_history=[dict(pass_id=k,p_cal_macro_ll=v) for k,v in scores.items()]),dict(_models=models,_terminal_models=families[10],records=records,elapsed_seconds=time.monotonic()-started,eta=True),dict(bundle.identity,method='PILOT_PROB',selected_pass=selected))
    def validator(b):return validate_raw_predictor(b,result,lambda role,h,z,source:predict_mixture_models(b,models,role,raw_h=h,raw_z=z,raw_source=source))
    result.diagnostics['_raw_x_validate']=validator;return map_prediction(bundle,result)
