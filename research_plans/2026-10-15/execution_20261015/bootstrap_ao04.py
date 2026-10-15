"""Same-row paired class-stratified bootstrap for existing selected predictions.
Fixed retrospective analysis seed 20261015; 1000 replicates; no model access.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'research_R2/formal_runs_20261012/kuairand/natural/m3oe'
with np.load(BASE/'B0/_final_test_evaluated/predictions_test.npz',allow_pickle=False) as z:
 ids=z['row_ids'];y=z['labels'];dom=z['physical_domain'];a=z['pred_B1_selected_seed2022'];b=z['pred_B0_selected_seed2022']
for arm,pred in [('B1',a),('B0',b)]:
 with np.load(BASE/f'{arm}/seed2022/attempt1/predictions_test.npz',allow_pickle=False) as z:
  assert np.array_equal(ids,z['row_ids']) and np.array_equal(y,z['labels']) and np.array_equal(pred,z['pred_selected'])
def setup(y,p):
 order=np.argsort(p,kind='stable');s=p[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
 return order,starts,y[order]
def aucw(w,state,np_,nn):
 order,starts,ys=state;ww=w[order];pos=np.add.reduceat(ww*ys,starts);neg=np.add.reduceat(ww*(1-ys),starts)
 return np.sum(pos*(np.cumsum(neg)-.5*neg))/(np_*nn)
rng=np.random.default_rng(20261015);rows=[];dist=[]
for d in range(3):
 for t in range(2):
  yy=y[dom==d,t].astype(np.float64);aa=a[dom==d,t];bb=b[dom==d,t]
  ip=np.flatnonzero(yy==1);ineg=np.flatnonzero(yy==0);na=len(ip);nb=len(ineg);sa=setup(yy,aa);sb=setup(yy,bb);vals=[]
  for i in range(1000):
   idx=np.r_[ip[rng.integers(na,size=na)],ineg[rng.integers(nb,size=nb)]];w=np.bincount(idx,minlength=len(yy));vals.append(aucw(w,sa,na,nb)-aucw(w,sb,na,nb))
  delta=float(roc_auc_score(yy,aa)-roc_auc_score(yy,bb));lo,hi=np.quantile(vals,[.025,.975]);dist.append(vals)
  rows.append(dict(dataset='kuairand',seed=2022,contrast='AO04_B1-B0',checkpoint='selected',domain=d,task=t,n=len(yy),n_pos=na,n_neg=nb,delta_auc=delta,ci95_low=lo,ci95_high=hi,replicates=1000,analysis_rng=20261015,method='same-row paired within cell and label; cellwise marginal CI'))
  print(d,t,delta,lo,hi,flush=True)
pd.DataFrame(rows).to_csv(OUT/'ao04_paired_bootstrap.csv',index=False)
# Tasks share rows but were stratified separately: never synthesize a macro CI.
