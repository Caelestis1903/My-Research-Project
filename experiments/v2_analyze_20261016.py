"""Primary-authored numerical summaries. Does not automatically claim causality."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_20261016 import OUT,sha,dump
RAW_CACHE={}


def bind_prediction_to_data(path,dataset,split):
    if dataset not in RAW_CACHE:
        if dataset=='kuairand':
            raw=pd.read_csv('/home/xzf/data/kuairand/processed/kuai_rand.csv',usecols=['tab','is_click','long_view'])
            labels=raw[['is_click','long_view']].to_numpy();domains=raw.tab.map({0:0,1:1,4:2}).to_numpy()
        else:
            raw=pd.read_csv('/home/xzf/data/ml-1m/ml-1m.csv',usecols=['rating','age'])
            labels=np.stack([(raw.rating.to_numpy()>3),(raw.rating.to_numpy()>4)],axis=1)
            mapping={a:d for d,ages in enumerate(({1,18},{25},{35,45,50,56})) for a in ages}
            domains=raw.age.map(mapping).to_numpy()
        RAW_CACHE[dataset]=(labels,domains)
    labels,domains=RAW_CACHE[dataset]
    with np.load(path,allow_pickle=False) as z:
        ids=z['row_ids'];assert np.array_equal(labels[ids],z['labels'])
        assert np.array_equal(domains[ids],z['domains'])
    lo=int(.8*len(labels)) if split=='validation' else int(.9*len(labels))
    hi=int(.9*len(labels)) if split=='validation' else len(labels)
    expected=np.flatnonzero(np.isin(domains[lo:hi],[0,1,2]))+lo
    assert np.array_equal(ids,expected),'missing/extra/reordered evaluation rows'


def summarize(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261016)
    draws=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
    lo,hi=np.quantile(draws,[.025,.975]);pos=int((a>0).sum());neg=int((a<0).sum())
    strength='STRONG' if (pos==len(a) or neg==len(a)) and (lo>0 or hi<0) else ('MEDIUM' if max(pos,neg)>=3 else 'WEAK')
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),median=float(np.median(a)),
        positive=pos,negative=neg,ci_low=float(lo),ci_high=float(hi),evidence=strength,
        **{f'seed{s}':float(x) for s,x in zip((2022,2023,2024,2025),a)})


def cells_from_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        d=z['domains'];y=z['labels'];pred=z['predictions'].astype(float);ids=z['row_ids']
    assert len(np.unique(ids))==len(ids) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    rows=[]
    for domain in range(3):
        m=d==domain
        for task in range(2):
            yy=y[m,task];pp=np.clip(pred[m,task],1e-12,1-1e-12)
            rows.append(dict(domain=domain,task=task,auc=roc_auc_score(yy,pred[m,task]),
                logloss=float(-(yy*np.log(pp)+(1-yy)*np.log1p(-pp)).mean()),n=int(m.sum())))
    return pd.DataFrame(rows)


def aggregates(c):
    out={}
    for field in ('auc','logloss'):
        out[field+'_macro']=float(c[field].mean())
        out[field+'_sample_weighted']=float(np.average(c[field],weights=c.n))
        for d in range(3):out[f'{field}_d{d}']=float(c[c.domain==d][field].mean())
        for t in range(2):out[f'{field}_t{t}']=float(c[c.task==t][field].mean())
        for row in c.itertuples():out[f'{field}_d{row.domain}t{row.task}']=float(getattr(row,field))
    return out


def calibrate():
    records=[]
    for path in sorted((OUT/'calibration').glob('*/attempt1/config.json')):
        r=path.parent;config=json.loads(path.read_text());receipt=r/'run_manifest.json'
        if not receipt.exists():
            assert (r/'failure.json').exists(),'calibration still active'
            records.append(dict(lr=config['lr'],status='FAILED',score=None));continue
        v=pd.read_csv(r/'metrics_epoch.csv');score=v[v.epoch>0].groupby('epoch').auc.mean()
        records.append(dict(lr=config['lr'],status='COMPLETE',score=float(score.max()),epoch=int(score.idxmax()),
            initial_auc=float(v[v.epoch==0].auc.mean()),run=str(r)))
    assert len(records)==3,'expected all three preregistered candidates'
    eligible=[r for r in records if r['status']=='COMPLETE']
    assert eligible,'all calibration candidates failed; primary audit required'
    best=sorted(eligible,key=lambda r:(-r['score'],r['epoch'],r['lr']))[0]
    adequate=any(r['score']-r['initial_auc']>=.005 for r in eligible)
    dump(OUT/'sgd_calibration_decision.json',dict(status='FROZEN_VALIDATION_ONLY',selected_lr=best['lr'],
        optimization_adequate=adequate,grid=[.1,1.,10.],boundary_winner=best['lr'] in (.1,10.),
        test_used=False,criterion='best validation macro AUC epoch1..10; earliest then smaller lr',candidates=records))
    print(json.dumps(dict(selected_lr=best['lr'],optimization_adequate=adequate,candidates=records)))


def audit_run(run):
    manifest=json.loads((run/'run_manifest.json').read_text());c=manifest['config']
    checks=[]
    for filename,h in manifest['artifact_hashes'].items():
        checks.append(dict(run=str(run),file=filename,match=sha(run/filename)==h));assert checks[-1]['match'],checks[-1]
    for source,h in c['source_hashes'].items():assert sha(OUT.parents[2]/source)==h,(source,'source drift')
    v=pd.read_csv(run/'metrics_epoch.csv');assert len(v)==6*(c['epochs']+1)
    assert set(v.epoch)==set(range(c['epochs']+1)) and not v.duplicated(['epoch','domain','task']).any()
    train_count=sum(c['data']['train_domain_counts'].values());steps_per_epoch=math.ceil(train_count/c['batch_size'])
    assert np.array_equal(v.steps.to_numpy(),v.epoch.to_numpy()*steps_per_epoch)
    trace=pd.read_csv(run/'batch_trace.csv');assert len(trace)==c['epochs']*steps_per_epoch
    assert not trace.duplicated(['epoch','batch']).any() and set(trace.epoch)==set(range(1,c['epochs']+1))
    assert (trace[['d0','d1','d2']].sum(axis=1)==trace.n).all()
    assert (trace.groupby('epoch').n.sum()==train_count).all()
    assert c['optimizer_count']==1 and c['learned_fusion'] is False and c['darts'] is False
    assert np.isfinite(trace[['weighted_loss','unweighted_bce']].to_numpy()).all()
    select=json.loads((run/'selection_frozen.json').read_text());epoch=int(v[v.epoch>0].groupby('epoch').auc.mean().idxmax())
    assert epoch==select['selected_epoch'] and select['test_access_started'] is False
    for split,ref in [('validation',v[v.epoch==epoch]),('test',pd.read_csv(run/'metrics_test.csv').query("checkpoint=='selected'"))]:
        bind_prediction_to_data(run/(split+'_selected.npz'),c['dataset'],split)
        calc=cells_from_prediction(run/(split+'_selected.npz')).sort_values(['domain','task']);ref=ref.sort_values(['domain','task'])
        for field in ('auc','logloss'):assert np.max(np.abs(calc[field].to_numpy()-ref[field].to_numpy()))<1e-10
    test=pd.read_csv(run/'metrics_test.csv')
    assert len(test)==6*9 and not test.duplicated(['checkpoint','domain','task']).any()
    return c,v,test,checks


def analyze(phase='P1'):
    target=OUT/'analysis';target.mkdir(exist_ok=True);runs=[];aggregated=[];checks=[];allcells=[];exposures=[]
    for path in sorted((OUT/'runs').glob('*/*/seed*/*/attempt*/run_manifest.json')):
        r=path.parent;c,v,t,cc=audit_run(r);checks.extend(cc)
        info=dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],seed=c['seed'],arm=c['arm'],run=str(r))
        runs.append(info);selected=json.loads((r/'selection_frozen.json').read_text())['selected_epoch']
        for split,cells in [('validation',v),('test',t)]:
            allcells.append(cells.assign(run=str(r),variant=c['sampler']))
            for (epoch,key),group in cells.groupby(['epoch','checkpoint']):
                aggregated.append(dict(**info,split=split,epoch=int(epoch),checkpoint=key,selected_epoch=selected,**aggregates(group)))
            if split=='validation':
                aggregated.append(dict(**info,split=split,epoch=selected,checkpoint='selected',selected_epoch=selected,**aggregates(v[v.epoch==selected])))
        exposures.append(pd.read_csv(r/'exposure.csv').assign(**info))
    a=pd.DataFrame(aggregated);a.to_csv(target/'aggregate_metrics.csv',index=False)
    pd.concat(allcells,ignore_index=True).to_csv(target/'cell_metrics.csv',index=False)
    pd.concat(exposures,ignore_index=True).to_csv(target/'exposure.csv',index=False)
    pd.DataFrame(checks).to_csv(target/'artifact_hash_checks.csv',index=False)
    contrasts=[]
    for keys,z in a.groupby(['dataset','optimizer','variant','split','checkpoint']):
        dataset,opt,variant,split,key=keys
        # Selected epochs differ per arm; fixed epochs must match.
        for balanced,natural in [('SB','SN'),('EB','EN')]:
            left=z[z.arm==balanced];right=z[z.arm==natural]
            merge=['seed'] if key=='selected' else ['seed','epoch']
            pairs=left.merge(right,on=merge,suffixes=('_b','_n'),validate='one_to_one')
            for row in pairs.itertuples():
                common=dict(dataset=dataset,optimizer=opt,variant=variant,split=split,checkpoint=key,
                    contrast=balanced+'-'+natural,seed=row.seed,epoch=-1 if key=='selected' else row.epoch)
                for field in [n for n in a.columns if n.startswith(('auc_','logloss_'))]:
                    contrasts.append(dict(**common,metric=field,delta=getattr(row,field+'_b')-getattr(row,field+'_n')))
    d=pd.DataFrame(contrasts);d.to_csv(target/'paired_differences.csv',index=False)
    summaries=[]
    for keys,z in d.groupby(['dataset','optimizer','variant','split','checkpoint','contrast','epoch','metric']):
        if len(z)!=4:continue
        summaries.append(dict(zip(['dataset','optimizer','variant','split','checkpoint','contrast','epoch','metric'],keys),
            **summarize(z.sort_values('seed').delta)))
    summary=pd.DataFrame(summaries);summary.to_csv(target/'paired_seed_summary.csv',index=False)
    # Strict initialization/data/batch pairing for each optimizer and q arm.
    pairing=[]
    for dataset in a.dataset.unique():
        for seed in (2022,2023,2024,2025):
            refs=[r for r in runs if r['dataset']==dataset and r['seed']==seed]
            if not refs:continue
            cfg=[json.loads((Path(r['run'])/'config.json').read_text()) for r in refs]
            assert len({c['init_sha'] for c in cfg})==1
            assert all(c['data']==cfg[0]['data'] for c in cfg)
            for variant in sorted({r['variant'] for r in refs}):
                for q in ('N','B'):
                    ts=[pd.read_csv(Path(r['run'])/'batch_trace.csv') for r in refs if r['arm'][1]==q and r['variant']==variant]
                    assert all(t.row_ids_sha256.equals(ts[0].row_ids_sha256) for t in ts)
            pairing.append(dict(dataset=dataset,seed=seed,runs=len(refs),passed=True))
    pd.DataFrame(pairing).to_csv(target/'pairing_checks.csv',index=False)
    # Both timing anchors give exact algebra, not causal decomposition.
    timings=[]
    for (dataset,opt,variant,seed),z in a.query("split=='validation' and checkpoint=='trajectory'").groupby(['dataset','optimizer','variant','seed']):
        for b,n in [('SB','SN'),('EB','EN')]:
            bz=z[z.arm==b].set_index('epoch');nz=z[z.arm==n].set_index('epoch')
            if bz.empty or nz.empty:continue
            # Epoch zero is a baseline readout, never a selectable checkpoint.
            be=int(bz.loc[bz.index>0].auc_macro.idxmax());ne=int(nz.loc[nz.index>0].auc_macro.idxmax())
            selected=bz.loc[be,'auc_macro']-nz.loc[ne,'auc_macro']
            fixed_n=bz.loc[ne,'auc_macro']-nz.loc[ne,'auc_macro'];timing_b=bz.loc[be,'auc_macro']-bz.loc[ne,'auc_macro']
            fixed_b=bz.loc[be,'auc_macro']-nz.loc[be,'auc_macro'];timing_n=nz.loc[be,'auc_macro']-nz.loc[ne,'auc_macro']
            assert abs(selected-fixed_n-timing_b)<1e-12 and abs(selected-fixed_b-timing_n)<1e-12
            timings.append(dict(dataset=dataset,optimizer=opt,variant=variant,seed=seed,contrast=b+'-'+n,
                balanced_selected_epoch=be,natural_selected_epoch=ne,selected_effect=selected,
                trajectory_at_natural_selection=fixed_n,balanced_timing=timing_b,
                trajectory_at_balanced_selection=fixed_b,natural_timing=timing_n))
    pd.DataFrame(timings).to_csv(target/'selection_timing.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for opt,z in a.query("dataset=='kuairand' and split=='validation' and checkpoint=='trajectory' and arm in ['SN','SB'] and variant=='iid'").groupby('optimizer'):
        for arm,c in z.groupby('arm'):
            y=c.groupby('epoch').mean(numeric_only=True)
            for axis,metric in zip(axes,('auc_macro','logloss_macro')):axis.plot(y.index,y[metric],label=opt+' '+arm)
    for axis,metric in zip(axes,('Macro AUC','Macro LogLoss')):axis.set(xlabel='Completed epoch',ylabel=metric);axis.legend()
    fig.tight_layout();fig.savefig(target/'p1_trajectories.png',dpi=160);fig.savefig(target/'p1_trajectories.pdf');plt.close(fig)
    dump(target/'receipt.json',dict(status='PASS_NUMERICAL_AUDIT',runs=len(runs),artifact_checks=len(checks),pairing=True,phase=phase))
    print(summary.query("dataset=='kuairand' and variant=='iid' and split=='validation' and checkpoint=='selected' and metric=='auc_macro'").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');p.add_argument('--phase',default='P1');args=p.parse_args()
    if args.calibrate:calibrate()
    else:analyze(args.phase)
