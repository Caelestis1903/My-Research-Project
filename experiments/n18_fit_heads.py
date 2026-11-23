"""Mechanical P3 frozen-head distillation training. Records raw metrics only."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,log_loss
from n18_common import OUT,dump

def design_basis(cell=None,center=False):
    d,t=np.indices((3,2));A=np.c_[np.ones(6),(d.ravel()==1),(d.ravel()==2),(t.ravel()==1)].astype(float)
    Q=np.linalg.qr(A)[0]
    if cell is None:return Q
    e=np.eye(6)[:,cell];r=(np.eye(6)-Q@Q.T)@e if center else e
    return np.linalg.qr(np.c_[Q,r])[0]

def matrices(data,mean,std):
    z=(data['z'].astype(float)-mean)/std
    return np.concatenate([z,np.ones((*z.shape[:-1],1))],axis=-1)

def solve(Z,teacher,domains,basis):
    cells=(domains[:,None]*2+np.arange(2)).ravel();X=Z.reshape(-1,Z.shape[-1]);y=logit(np.clip(teacher,1e-6,1-1e-6)).ravel()
    # Every domain contributes the same number of rows; rows carry two tasks.
    H=(basis[cells,:,None]*X[:,None,:]).reshape(len(X),-1)
    cov=H.T@H/len(H)+np.eye(H.shape[1])*1e-6
    coef=np.linalg.solve(cov,H.T@y/len(H));W=basis@coef.reshape(basis.shape[1],-1)
    error=float(np.mean((np.sum(X*W[cells],axis=1)-y)**2))
    return W,coef,error

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--measurement',type=Path)
    p.add_argument('--out',type=Path);a=p.parse_args();source=a.measurement or OUT/'measurements'/f'seed{a.seed}'
    out=a.out or OUT/'P3_training'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False);started=time.time()
    data={key:dict(np.load(source/f'features_{key}.npz')) for key in ['fit','selection','diagnostic']}
    mean=data['fit']['z'].mean(axis=(0,1));std=data['fit']['z'].std(axis=(0,1)).clip(1e-6)
    Z={key:matrices(v,mean,std) for key,v in data.items()};K=Z['fit'].shape[-1]
    models={};coeffs={};metadata={}
    for name,basis in [('flat',np.eye(6)),('additive',design_basis())]:
        W,coef,err=solve(Z['fit'],data['fit']['teacher'],data['fit']['domains'],basis)
        models[name]=W;coeffs[name]=coef;coeffs[name+'_basis']=basis
        metadata[name]=dict(coefficient_dof=len(coef),stored_head_scalars=len(coef)+basis.size,train_logit_mse=err)
    candidates=[]
    for cell in range(6):
        basis=design_basis(cell);W,coef,err=solve(Z['fit'],data['fit']['teacher'],data['fit']['domains'],basis)
        candidates.append((err,cell,W,coef,basis))
    err,cell,W,coef,basis=min(candidates,key=lambda x:(x[0],x[1]))
    models['ordinary_one_cell']=W;coeffs['ordinary_one_cell']=coef;coeffs['ordinary_one_cell_basis']=basis
    # Canonicalizing both equivalent classes prevents parameter-ridge inequivalence.
    centered=design_basis(cell,True)
    projector_error=float(abs(basis@basis.T-centered@centered.T).max())
    assert projector_error<1e-12
    models['centered_repair']=W.copy();coeffs['centered_repair']=coef.copy();coeffs['centered_repair_basis']=basis.copy()
    for name in ['ordinary_one_cell','centered_repair']:
        metadata[name]=dict(coefficient_dof=len(coef),stored_head_scalars=len(coef)+basis.size,train_logit_mse=err,selected_cell=cell)
    u,s,v=np.linalg.svd(models['flat'],full_matrices=False);r=4
    models['rank4_svd']=(u[:,:r]*s[:r])@v[:r]
    coeffs['rank4_left']=u[:,:r]*s[:r];coeffs['rank4_right']=v[:r]
    metadata['rank4_svd']=dict(coefficient_dof=r*(6+K-r),stored_head_scalars=r*(6+K))
    rows=[];pred_archive={}
    for split in ['selection','diagnostic']:
        dd=data[split];domains=dd['domains'];n=len(domains)
        for name in ['teacher',*models]:
            pred=dd['teacher'] if name=='teacher' else expit(np.einsum('ntk,ntk->nt',Z[split],models[name][domains[:,None]*2+np.arange(2)]))
            pred_archive[f'{split}_{name}']=pred
            for d in range(3):
                mask=domains==d
                for t in range(2):
                    yy=dd['y'][mask,t];pp=pred[mask,t]
                    rows.append(dict(seed=a.seed,split=split,model=name,domain=d,task=t,n=int(mask.sum()),
                        positives=int(yy.sum()),auc=float(roc_auc_score(yy,pp)),logloss=float(log_loss(yy,pp,labels=[0,1])),
                        teacher_logit_mse=float(np.mean((logit(pp.clip(1e-6,1-1e-6))-logit(dd['teacher'][mask,t].clip(1e-6,1-1e-6)))**2))))
        for field in ['domains','y','row_ids']:pred_archive[f'{split}_{field}']=dd[field]
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'head_coefficients.npz',**coeffs,feature_mean=mean,feature_std=std)
    np.savez_compressed(out/'predictions.npz',**pred_archive)
    dump(out/'config.json',dict(seed=a.seed,fit_rows=len(data['fit']['domains']),teacher_epoch=10,
         method='closed-form frozen-feature logit ridge distillation',ridge=1e-6,bias_regularized=True,
         selection='residual cell minimizes train logit MSE; no epoch/LR search',models=metadata,
         equivalence_projector_error=projector_error,mean_std_storage=mean.size+std.size,
         labels_used_for_fitting=False,optimizer_count=0,test_access=False))
    dump(out/'receipt.json',dict(status='COMPLETE',seed=a.seed,wall_seconds=time.time()-started,
         fit_count=8,training_type='analytic head fitting; backbone frozen',test_access=False))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
