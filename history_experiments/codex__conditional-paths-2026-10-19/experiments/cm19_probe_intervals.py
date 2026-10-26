"""Paired within-domain row bootstrap for frozen functional sensitivity probes."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from cm19_train import OUT,dump

def auc(y,p):
    n1=int(y.sum());n0=len(y)-n1
    if n0==0 or n1==0:return np.nan
    return (rankdata(p,method='average')[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

def bce(y,p):
    # NumPy float64 scoring for paired risk, distinct from training float32 BCE.
    p=np.clip(p.astype(np.float64),1e-7,1-1e-7);return -(y*np.log(p)+(1-y)*np.log1p(-p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=200);a=p.parse_args();manifest=json.loads((OUT/'run_manifest.json').read_text());records=[]
    for c in manifest['runs']:
        if c['stage'] not in ['G0','G0M','G0MC']:continue
        out=OUT/'runs'/c['run_id'];cfg=json.loads((out/'config.json').read_text());art=Path(cfg['state_directory'])
        for label in ['selected','final']:
            for split in ['meta','validation']:
                with np.load(art/f'probe_{label}_{split}.npz') as f:data={k:f[k] for k in f.files}
                for d in range(3):
                    ix=data['domains']==d;pp=data['p'][ix];yy=data['labels'][ix];n=len(pp);rng=np.random.default_rng(191019+d);draws=rng.integers(0,n,(a.replicates,n))
                    for key,value in data.items():
                        if not key.startswith('neutral_'):continue
                        qq=value[ix];kind=key.removeprefix('neutral_')
                        for j,t in enumerate(cfg['task_ids']):
                            delta=np.abs(pp[:,j]-qq[:,j]);risk=bce(yy[:,t],qq[:,j])-bce(yy[:,t],pp[:,j])
                            pboot=delta[draws].mean(1);rboot=risk[draws].mean(1)
                            if np.array_equal(pp[:,j],qq[:,j]):aboot=np.zeros(a.replicates)
                            else:aboot=np.array([auc(yy[idx,t],qq[idx,j])-auc(yy[idx,t],pp[idx,j]) for idx in draws])
                            record=dict(run_id=c['run_id'],dataset=c['dataset'],model=c['model'],optimizer=c['optimizer'],seed=c['seed'],checkpoint=label,probe=split,domain=d,task=t,intervention=kind,n=n,replicates=a.replicates,valid_auc_replicates=int(np.isfinite(aboot).sum()),mean_delta_p=float(delta.mean()),mean_delta_bce=float(risk.mean()),delta_auc=auc(yy[:,t],qq[:,j])-auc(yy[:,t],pp[:,j]))
                            for stat,arr in [('delta_p',pboot),('delta_bce',rboot),('delta_auc',aboot)]:record[stat+'_ci95_low']=float(np.nanquantile(arr,.025));record[stat+'_ci95_high']=float(np.nanquantile(arr,.975))
                            records.append(record)
    pd.DataFrame(records).to_csv(OUT/'probe_conditional_intervals.csv',index=False)
    dump(OUT/'probe_interval_receipt.json',dict(rows=len(records),replicates=a.replicates,scope='paired row bootstrap within physical domain; same indices across tasks/interventions; conditional on one model/probe, not optimization/population confidence',auc_ranking='average ranks for float32 ties',bce='float64 clipped1e-7, distinct from raw float32 training objective'))
    print(json.dumps(dict(status='COMPLETE_PROBE_INTERVALS',rows=len(records))))
if __name__=='__main__':main()
