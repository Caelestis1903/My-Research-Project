"""AO04 seed2022 validation paired bootstrap after embedded-state row binding."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
BASE=ROOT/'research_R2/formal_runs_20261012/kuairand/natural/m3oe'
RECOVERY=OUT/'ao04_recovery_attempt3'
inventory=json.loads((RECOVERY/'state_recovery_inventory.json').read_text())
for arm in ['B0','B1']:
    r=next(x for x in inventory if x['case']==f'kuairand_{arm}_2022')
    assert r['validation_binding']=='RECONSTRUCTED_LOADER_AND_FULL_PREDICTIONS_MATCH'
    path=BASE/f'{arm}/seed2022/attempt1/validation_predictions_selected.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['validation_npz_sha256']
with np.load(BASE/'B1/seed2022/attempt1/validation_predictions_selected.npz',allow_pickle=False) as z:
    a=z['predictions'];labels=z['labels']
with np.load(BASE/'B0/seed2022/attempt1/validation_predictions_selected.npz',allow_pickle=False) as z:
    b=z['predictions'];assert np.array_equal(labels,z['labels'])
with np.load(RECOVERY/'kuairand_B1_2022_validation_row_ids.npz',allow_pickle=False) as z:
    ids=z['row_ids'];domains=z['domains']
with np.load(RECOVERY/'kuairand_B0_2022_validation_row_ids.npz',allow_pickle=False) as z:
    assert np.array_equal(ids,z['row_ids']) and np.array_equal(domains,z['domains'])

# Same tie-aware weighted-AUC calculation verified by the earlier 100-case oracle.
def setup(y,p):
    order=np.argsort(p,kind='stable');s=p[order]
    return order,np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1],y[order]
def aucw(w,state,np_,nn):
    order,starts,y=state;ww=w[order]
    pos=np.add.reduceat(ww*y,starts);neg=np.add.reduceat(ww*(1-y),starts)
    return np.sum(pos*(np.cumsum(neg)-.5*neg))/(np_*nn)
rng=np.random.default_rng(20261016);rows=[];draws=[]
for d in range(3):
    for t in range(2):
        mask=domains==d;y=labels[mask,t].astype(float);aa=a[mask,t];bb=b[mask,t]
        ip=np.flatnonzero(y==1);ineg=np.flatnonzero(y==0);na=len(ip);nb=len(ineg)
        sa=setup(y,aa);sb=setup(y,bb);vals=[]
        ll_a=-(y*np.log(np.clip(aa.astype(float),1e-12,1-1e-12))+(1-y)*np.log1p(-np.clip(aa.astype(float),1e-12,1-1e-12)))
        ll_b=-(y*np.log(np.clip(bb.astype(float),1e-12,1-1e-12))+(1-y)*np.log1p(-np.clip(bb.astype(float),1e-12,1-1e-12)))
        loss=ll_a-ll_b;loss_vals=[]
        for i in range(1000):
            index=np.r_[ip[rng.integers(na,size=na)],ineg[rng.integers(nb,size=nb)]]
            w=np.bincount(index,minlength=len(y))
            vals.append(aucw(w,sa,na,nb)-aucw(w,sb,na,nb));loss_vals.append(float(w@loss/len(y)))
        lo,hi=np.quantile(vals,[.025,.975]);lllo,llhi=np.quantile(loss_vals,[.025,.975])
        rows.append(dict(dataset='kuairand',seed=2022,split='validation',contrast='B1-B0',domain=d,task=t,
            n=len(y),n_pos=na,auc_B0=float(roc_auc_score(y,bb)),auc_B1=float(roc_auc_score(y,aa)),
            delta_auc=float(roc_auc_score(y,aa)-roc_auc_score(y,bb)),ci95_low=float(lo),ci95_high=float(hi),
            delta_logloss=float(loss.mean()),logloss_ci95_low=float(lllo),logloss_ci95_high=float(llhi),
            replicates=1000,analysis_rng=20261016,scope='conditional on selected models; per-cell label-stratified rows'))
        draws.append(vals)
        print(d,t,rows[-1]['delta_auc'],lo,hi,flush=True)
pd.DataFrame(rows).to_csv(OUT/'ao04_validation_bootstrap.csv',index=False)
np.savez_compressed(OUT/'ao04_validation_bootstrap_draws.npz',draws=np.array(draws))
(OUT/'ao04_validation_bootstrap_receipt.json').write_text(json.dumps({'status':'COMPLETE_ONE_SEED_CONDITIONAL_FORENSIC',
    'training_runs':0,'independent_review':False,'cells':6,'replicates':1000,
    'method':'paired rows within domain/task and class; tasks separately stratified; no synthesized macro CI',
    'limitations':['conditional on validation-selected fitted models','not a post-selection confidence guarantee',
        'no user/time clustering; no multiplicity correction; one complete paired seed'],
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
