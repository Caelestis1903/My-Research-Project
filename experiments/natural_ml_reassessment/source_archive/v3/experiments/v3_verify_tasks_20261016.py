"""Primary numerical cross-check of stored complete cell-task gradients."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_audit_20261016 as audit
base=audit.base;OUT=audit.OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[];hashes=[];checks=[]
 for opt in ('sgd','adam'):
  for seed in audit.SEEDS:
   source=OUT/'runs'/f'V1tasks__kuairand__{opt}__epoch10__seed{seed}';receipt=json.loads((source/'receipt.json').read_text());assert receipt['status']=='PASS'
   assert base.sha(source/'task_gradients.npz')==receipt['task_gradient_sha'];assert base.sha(Path(receipt['source_checkpoint']))==receipt['checkpoint_sha']
   reference=Path(receipt['v1_reference']);old=np.load(reference/'exact.npz');raw=np.load(source/'task_gradients.npz');g=raw['gradients'];weighted=raw['overall_task_gradients']
   b=audit.bundle('kuairand',seed);model=base.fixed_model(b,seed);groups,_=audit.block_indices(model);groups['ALL']=np.arange(sum(p.numel() for p in model.parameters()))
   p_train=np.array([b.train_counts[d] for d in range(3)],dtype=float);p_train/=p_train.sum();assert np.allclose(weighted,np.einsum('d,dtk->tk',p_train,g),atol=1e-13,rtol=1e-13)
   errors=np.linalg.norm(g.mean(1)-old['domain_gradients'],axis=1);bounds=1e-6+1e-4*np.linalg.norm(old['domain_gradients'],axis=1);assert np.all(errors<=bounds)
   reported=pd.read_csv(source/'cross_task_geometry.csv',dtype={'domain':str});actual=[]
   for domain,values in [(str(d),g[d]) for d in range(3)]+[('ALL',weighted)]:
    for block,ix in groups.items():
     u,v=values[:,ix];den=np.linalg.norm(u)*np.linalg.norm(v);row=dict(domain=domain,block=block,task0_norm=float(np.linalg.norm(u)),task1_norm=float(np.linalg.norm(v)),cosine=float(u@v/den) if den else np.nan)
     match=reported[(reported.domain==domain)&(reported.block==block)];assert len(match)==1
     for key in ('task0_norm','task1_norm','cosine'):assert np.isclose(row[key],match.iloc[0][key],atol=1e-12,rtol=1e-10,equal_nan=True)
     row.update(seed=seed,optimizer=opt,source_checkpoint_epoch=10,source_run=str(source));actual.append(row)
   rows.extend(actual);checks.append(dict(seed=seed,optimizer=opt,status='PASS',max_domain_half_sum_residual=float(errors.max()),max_domain_bound=float(bounds.max())))
   hashes.append(dict(seed=seed,optimizer=opt,path=str(source/'task_gradients.npz'),verified_sha=receipt['task_gradient_sha'],checkpoint_verified_sha=receipt['checkpoint_sha']))
 pd.DataFrame(rows).to_csv(a.out/'complete_cross_task_geometry.csv',index=False);pd.DataFrame(checks).to_csv(a.out/'half_sum_identity.csv',index=False);base.dump(a.out/'verified_raw_hashes.json',hashes)
 base.dump(a.out/'receipt.json',dict(status='PASS',tasks_verified=len(checks),coordinate_scope='full model, all training rows per domain×task, float64 accumulation of actual frozen autograd field',source_sha=base.sha(__file__),no_new_AUC_inference=True))
if __name__=='__main__':main()
