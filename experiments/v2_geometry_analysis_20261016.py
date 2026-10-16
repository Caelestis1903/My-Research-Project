"""P6 spectra and temporal readouts with explicit measurement scopes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from v2_20261016 import OUT,sha,dump


def main():
    target=OUT/'analysis';temporal=[];geometry=[];cosines=[];spectra=[]
    for receipt in sorted((OUT/'probes').glob('*/*/seed*/receipt.json')):
        r=json.loads(receipt.read_text());directory=receipt.parent;dataset=r['dataset'];opt=r['optimizer'];seed=r['seed']
        config=json.loads((OUT/'runs'/dataset/opt/f'seed{seed}'/'SN/attempt1/config.json').read_text());alpha=np.array(config['alpha'])
        for filename,bag in [('geometry.csv',geometry),('cosines.csv',cosines),('covariance_spectrum.csv',spectra)]:
            frame=pd.read_csv(directory/filename)
            # This legacy producer field scales covariance of a 32-row
            # microbatch estimator. It is not production batch covariance.
            if 'batch40960_trace' in frame:
                frame=frame.rename(columns={'batch40960_trace':'diagnostic_microbatch_trace_divided_by_40960'})
                frame['scaled_trace_scope']='arithmetic diagnostic only; not the covariance of production40960-row batches'
            bag.append(frame.assign(dataset=dataset,optimizer=opt,seed=seed))
        for snap in ('initial','selected','final'):
            with np.load(directory/(snap+'_gradient_pool.npz'),allow_pickle=False) as z:
                pools={block:z[block] for block in ('C','P','R','E')}
            rng=np.random.default_rng(seed*1000+971);uniforms=rng.random((2048,2))
            for policy,q in [('N',alpha),('B',np.ones(3)/3)]:
                d=np.searchsorted(np.cumsum(q),uniforms[:,0],side='right');j=np.floor(uniforms[:,1]*8).astype(int)
                for block,pool in pools.items():
                    expected=sum(alpha[k]*pool[k].mean(0) for k in range(3))
                    values=pool[d,j]*(alpha[d]/q[d])[:,None]-expected
                    for lag in (1,5,10):
                        left=values[lag:];right=values[:-lag];denom=np.sqrt(np.square(left).sum()*np.square(right).sum())
                        corr=float((left*right).sum()/denom) if denom>0 else None
                        temporal.append(dict(dataset=dataset,optimizer=opt,seed=seed,snapshot=snap,policy=policy,block=block,
                            lag=lag,autocorrelation=corr,sequence_length=2048,scope='synthetic IID fixed-theta finite-pool projected gradients; not training autocorrelation'))
    if geometry:
        pd.concat(geometry,ignore_index=True).to_csv(target/'finite_pool_geometry.csv',index=False)
        pd.concat(cosines,ignore_index=True).to_csv(target/'finite_pool_task_domain_cosines.csv',index=False)
        pd.concat(spectra,ignore_index=True).to_csv(target/'finite_pool_covariance_spectra.csv',index=False)
    pd.DataFrame(temporal).to_csv(target/'finite_pool_temporal_autocorrelation.csv',index=False)
    actual=[];normcorr=[]
    for path in sorted((OUT/'runs').glob('*/*/seed*/*/attempt*/run_manifest.json')):
        c=json.loads(path.read_text())['config'];obs=pd.read_csv(path.parent/'optimizer_observations.csv')
        for block,z in obs.groupby('partition'):
            for stage,low,high in [('early',1,10),('middle',11,30),('late',31,50)]:
                zz=z[z.epoch.between(low,high)];pairs=zz[zz.batch==0].merge(zz[zz.batch==1],on='epoch',suffixes=('_0','_1'))
                actual.append(dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],seed=c['seed'],arm=c['arm'],block=block,
                    stage=stage,observed_pairs=len(pairs),mean_actual_lag1_cosine=float(pairs.consecutive_gradient_cosine_1.mean()),
                    scope='actual first two full training batches per epoch; raw gradient cosine, not centered covariance correlation'))
                normcorr.append(dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],seed=c['seed'],arm=c['arm'],block=block,
                    stage=stage,pairs=len(pairs),norm_lag1_pearson=float(pairs.gradient_norm_0.corr(pairs.gradient_norm_1)),
                    scope='gradient norm only across epoch pairs; residual training drift can confound even within stage'))
    pd.DataFrame(actual).to_csv(target/'actual_lag1_gradient_cosine.csv',index=False)
    pd.DataFrame(normcorr).to_csv(target/'actual_lag1_gradient_norm_correlation.csv',index=False)
    dump(target/'geometry_receipt.json',dict(status='COMPLETE',probe_runs=len(geometry),source_sha256=sha(__file__),
        synthetic_temporal_rows=len(temporal),actual_training_stage_rows=len(actual),
        full_coordinate_training_covariance_unmeasured=True,causal_variance_explanation_identified=False))
    print('P6 temporal and spectrum packet complete',len(temporal),len(actual))

if __name__=='__main__':main()
