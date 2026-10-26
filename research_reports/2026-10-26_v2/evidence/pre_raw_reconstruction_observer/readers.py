"""FP64 convex readers and complete six-head validation policies."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .protocol import FeatureBank,PredictionBundle,evaluate,record_access
from .state import digest
from .memory_data import persist,map_prediction,release_data_pages

def objective_gradient(theta,b,x,y,lam,ridge=1e-6):
    alpha,beta=theta[:2];w=theta[2:];correction=x@w;logits=alpha*b+beta+correction
    error=expit(logits)-y;n=len(y)
    value=np.mean(np.logaddexp(0,logits)-y*logits)+lam*np.mean(correction**2)/2+ridge*np.dot(w,w)/2
    grad=np.r_[np.dot(error,b)/n,np.mean(error),x.T@error/n+lam*x.T@correction/n+ridge*w]
    return float(value),grad

def kkt_residual(theta,gradient,bounds):
    projected=np.array(gradient,copy=True)
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None and theta[j]<=lo+1e-9:projected[j]=min(projected[j],0)
        if hi is not None and theta[j]>=hi-1e-9:projected[j]=max(projected[j],0)
    return float(np.max(np.abs(projected))) if len(projected) else 0.

def fit_group(b,x,y,lam,bounds,config):
    b=np.asarray(b,np.float64);x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    box=[(.25,4.),(-4.,4.)]+list(bounds);start=np.r_[1.,0.,np.zeros(x.shape[1])]
    if len(y)==0:return start,dict(status='EMPTY_GROUP',rows=0,kkt_residual=0.)
    if x.shape[1]>0 and all(lo is None and hi is None for lo,hi in bounds):
        # Approved equivalent numerical coordinates apply only to wholly free
        # FULL content coefficients. Return original alpha/beta/w unchanged.
        from .numerical_reader import solve_free_full
        theta,receipt=solve_free_full(b,x,y,lam,bounds,config,objective_gradient=objective_gradient)
        if not receipt['accepted']:
            raise RuntimeError('Whitened FULL reader failed registered numerical contract: '+str(receipt))
        return theta,receipt
    fn=lambda theta:objective_gradient(theta,b,x,y,lam,config['reader']['coefficient_ridge'])
    result=minimize(fn,start,jac=True,method='L-BFGS-B',bounds=box,options=dict(maxiter=config['reader']['max_iter'],gtol=config['reader']['gtol'],ftol=1e-12,maxls=40))
    retried=False
    if not result.success:
        retried=True;result=minimize(fn,start,jac=True,method='L-BFGS-B',bounds=box,options=dict(maxiter=config['reader']['numerical_retry_max_iter'],gtol=config['reader']['gtol'],ftol=1e-12,maxls=40))
    value,gradient=fn(result.x);residual=kkt_residual(result.x,gradient,box)
    free=[j for j,(lo,hi) in enumerate(box) if (lo is None or result.x[j]>lo+1e-9) and (hi is None or result.x[j]<hi-1e-9)]
    if not np.isfinite(result.x).all() or not np.isfinite(value):raise FloatingPointError('Reader nonfinite optimum')
    if not result.success and residual>1e-5:raise RuntimeError('Reader did not converge after registered numerical retry: '+str(result.message))
    return result.x,dict(status='CONVERGED' if result.success else 'KKT_CONVERGED',rows=len(y),objective=value,iterations=int(result.nit),kkt_residual=residual,free_gradient_max=float(np.max(np.abs(gradient[free]))) if free else 0.,retry=retried,message=str(result.message),bounds=box)

class LinearPredictor:
    def __init__(self,bundle,bank,coefficients,off=False):self.bundle=bundle;self.bank=bank;self.coefficients=coefficients;self.off=off
    def predict(self,role,features=None,base=None,groups=None):
        v=self.bundle.views[role];base=self.bundle.base_logits[role] if base is None else base
        if self.off:return expit(base)
        bank=self.bank if features is None else features;group=v.group if groups is None else groups
        out=np.empty_like(base,dtype=np.float64)
        for t in range(3):
            mask=v.domain==t;pos=np.flatnonzero(mask)
            for q in range(2):
                x=bank.values[role,t,q] if isinstance(bank,FeatureBank) else bank[t,q]
                for g in (0,1):
                    selected=group[pos,q]==g;rows=pos[selected];c=self.coefficients[t,q,g]
                    out[rows,q]=c['alpha']*base[rows,q]+c['beta']+np.asarray(x[selected],np.float64)@c['weights']
        return expit(out)
    def __call__(self,role,features=None):return self.predict(role,features)

def fit_reader_candidates(bundle,features,method,config,*,train_roles=('P_read',)):
    banks=features if isinstance(features,list) else [features];candidates=[]
    for bank_index,bank in enumerate(banks):
        # Fixed-content banks are already released after selection. Retained
        # delta/single-function banks use the same explicit data mapping.
        if not hasattr(bank,'fixed_method'):
            for key in bank.values:
                bank.values[key]=persist(bundle,('reader_features',method,bank_index,key),bank.values[key])
    for start_id,bank in enumerate(banks):
        for lam in config['reader']['lambda_grid']:
            coefficients={};checks=[]
            for t in range(3):
                for q in range(2):
                    for g in (0,1):
                        bs=[];xs=[];ys=[]
                        for role in train_roles:
                            view=bundle.views[role];own=view.domain==t;sub=view.group[own,q]==g;rows=np.flatnonzero(own)[sub]
                            record_access(bundle,method,role,view.row_id[rows],q,purpose='reader',lambda_value=lam,start_id=start_id,target=t,group=g)
                            bs.append(bundle.base_logits[role][rows,q]);xs.append(bank.values[role,t,q][sub]);ys.append(view.y[rows,q])
                        b=np.concatenate(bs);x=np.concatenate(xs);y=np.concatenate(ys)
                        theta,check=fit_group(b,x,y,lam,bank.bounds[t,q],config);coefficients[t,q,g]=dict(alpha=float(theta[0]),beta=float(theta[1]),weights=theta[2:].copy())
                        checks.append(dict(target=t,task=q,group=g,**check))
            predictor=LinearPredictor(bundle,bank,coefficients);pc=bundle.views['P_cal'];record_access(bundle,method,'P_cal',pc.row_id,purpose='select',lambda_value=lam,start_id=start_id)
            p=predictor.predict('P_cal');score=evaluate(pc.y,p,pc.domain)['macro_logloss']
            candidates.append(dict(start_id=start_id,lambda_value=lam,p_cal_macro_ll=score,coefficients=coefficients,predictor=predictor,checks=checks,bank=bank,off=False))
    off=LinearPredictor(bundle,banks[0],{},True);pc=bundle.views['P_cal'];record_access(bundle,method,'P_cal',pc.row_id,purpose='select',candidate='OFF')
    score=evaluate(pc.y,off.predict('P_cal'),pc.domain)['macro_logloss'];candidates.append(dict(start_id=-1,lambda_value=None,p_cal_macro_ll=score,coefficients={},predictor=off,checks=[],bank=banks[0],off=True))
    return candidates

def fit_reader(bundle,features,method,config,*,train_roles=('P_read',)):
    started=time.monotonic();candidates=fit_reader_candidates(bundle,features,method,config,train_roles=train_roles)
    # Exact risk ties: OFF first, then larger lambda, finally deterministic start.
    best=min(candidates,key=lambda c:(c['p_cal_macro_ll'],not c['off'],-(c['lambda_value'] or 0),c['start_id']))
    predictor=best['predictor'];probabilities={role:predictor.predict(role) for role in bundle.views}
    coefficients=best['coefficients']
    if best['off']:
        coefficients={(t,q,g):dict(alpha=1.,beta=0.,weights=np.zeros(len(best['bank'].columns[t,q]))) for t in range(3) for q in range(2) for g in (0,1)}
    selection=dict(selected='OFF' if best['off'] else 'READER',selected_lambda=best['lambda_value'],selected_start=best['start_id'],p_cal_macro_ll=best['p_cal_macro_ll'],unit='complete_six_head_policy',role='P_cal',candidate_history=[{k:c[k] for k in ('start_id','lambda_value','p_cal_macro_ll','off')} for c in candidates])
    result=PredictionBundle(method,probabilities,{k:v.copy() for k,v in probabilities.items()},selection,dict(selected_coefficients=coefficients,policies=coefficients,predictor=predictor,_candidates=candidates,checks=best['checks'],elapsed_seconds=time.monotonic()-started),dict(bundle.identity,method=method,reader_policy_hash=digest(dict(selection=selection,coefficients=coefficients)),train_roles=list(train_roles)))
    def validator(b):
        from .baselines import validate_raw_predictor,predict_direct_models,raw_features
        def calculate(role,h,z,source):
            base=predict_direct_models(b,b.base_models,role,raw_h=h,raw_z=z)
            if predictor.off:return expit(base)
            groups=np.zeros((len(z),2),dtype=np.int64);view=b.views[role]
            for t in range(3):groups[view.domain==t]=(z[view.domain==t]>b.normalizers['group_cut'][t]).astype(np.int64)
            content=raw_features(b,best['bank'],role,h,z,view.domain)
            return predictor.predict(role,content,base=base,groups=groups)
        return validate_raw_predictor(b,result,calculate)
    result.diagnostics['_raw_x_validate']=validator
    result.diagnostics['_candidates']=[]  # unused fits retain no duplicate large FeatureBank
    # Fixed-content values are reproducible from retained pilots and do not need
    # duplicate N×64 copies for each completed control. Predictions/policies stay.
    if all(hasattr(c['bank'],'fixed_method') for c in candidates):
        for c in candidates:c['bank'].values.clear()
        result.diagnostics['_candidates']=[]
        result.diagnostics['fixed_feature_storage']='released after fitting; raw-X reconstructed from frozen pilot metadata'
    result=map_prediction(bundle,result)
    release_data_pages(bundle)
    return result


# A reviewed normal reload of this module also refreshes the already loaded
# observational collector. The sole target is hardcoded to this package; fresh
# imports do not import delivery, run benchmarks, fit models or touch state.
import importlib as _observer_importlib
import sys as _observer_sys
_observer_delivery_name = __package__ + '.delivery'
if _observer_delivery_name in _observer_sys.modules:
    _observer_importlib.reload(_observer_sys.modules[_observer_delivery_name])

# Fixed, independently authorized operational hook. Workers/clients are NO_OP;
# the sole target is the existing registered controller's Campaign.phase.
from .phase_scheduler import install_on_reader_import as _install_reviewed_phase_scheduler
_reviewed_phase_scheduler_install = _install_reviewed_phase_scheduler()
