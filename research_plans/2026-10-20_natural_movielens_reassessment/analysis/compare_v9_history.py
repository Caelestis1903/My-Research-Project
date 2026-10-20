"""Prespecified V9 windows; historical CSV evidence and frozen new predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze_core import interval

HERE=Path(__file__).resolve().parent
OUT=HERE/'v9_history';OUT.mkdir(exist_ok=True)
OLD=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016/archive/formal')
ART=Path('/home_new/xzf_natural_movielens_reassessment')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

frames=[];sources=[];selected={}
for arm in ('SN','SB'):
    for seed in (2022,2023,2024,2025):
        run=OLD/f'V9__movielens__sgd__{arm}__seed{seed}/budget050'
        path=run/'metrics_test.csv';f=pd.read_csv(path)
        assert set(f.epoch)==set(range(1,51)) and f.groupby('epoch').size().eq(6).all()
        pop='movielens_artificial_historical'
        f['population']=pop;f['main_steps']=5*f.epoch;frames.append(f)
        selected[(pop,arm,seed)]=json.loads((run/'selection_frozen.json').read_text())['selected_epoch']
        sources.append(dict(population=pop,arm=arm,seed=seed,path=str(path),sha256=sha(path),
                            evidence_grade='historical saved CSV; not old predictions reevaluated'))

lock=ART/'V9_trajectory_readout/frozen_readout.json';new_complete=False
if lock.exists():
    frozen=json.loads(lock.read_text());runs=frozen['runs']
    new_complete=all((ART/'V9_trajectory_readout'/r['run_id']/'receipt.json').exists() for r in runs)
    if new_complete:
        with np.load(ART/'data/movielens_natural.npz') as z:
            ids=z['test_row_ids'];y=z['test_y'];domains=z['test_x'][:,-1].astype(int)
        audits=[]
        for entry in runs:
            run=ART/'V9_trajectory_readout'/entry['run_id'];receipt=json.loads((run/'receipt.json').read_text())
            assert receipt['status']=='COMPLETE' and receipt['optimizer_updates']==0
            assert receipt['freeze_sha256']==sha(lock)
            path=run/'metrics_test_trajectory.csv';f=pd.read_csv(path)
            assert set(f.epoch)==set(range(1,51)) and f.groupby('epoch').size().eq(6).all()
            for epoch,g in f.groupby('epoch'):
                prediction=run/f'test_epoch{epoch:03d}.npz'
                with np.load(prediction) as z:
                    assert np.array_equal(z['row_ids'],ids) and np.array_equal(z['labels'],y)
                    assert np.array_equal(z['domains'],domains)
                    p=z['predictions'].astype(float)
                for row in g.itertuples():
                    mask=domains==row.domain;yy=y[mask,row.task];pp=p[mask,row.task]
                    aa=roc_auc_score(yy,pp);pp=np.clip(pp,1e-12,1-1e-12)
                    ll=np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp))
                    assert abs(aa-row.auc)<1e-12 and abs(ll-row.logloss)<1e-12
                    audits.append(dict(run_id=entry['run_id'],epoch=epoch,domain=row.domain,task=row.task,
                        auc_residual=aa-row.auc,logloss_residual=ll-row.logloss,prediction_sha256=sha(prediction)))
            # The new readout must reproduce all previously frozen test points.
            existing=pd.read_csv(Path(entry['run_dir'])/'metrics_test.csv')
            for row in existing.itertuples():
                match=f[(f.epoch==row.epoch+1)&(f.domain==row.domain)&(f.task==row.task)]
                assert len(match)==1
                assert abs(match.auc.iloc[0]-row.auc)<1e-12
                assert abs(match.logloss.iloc[0]-row.logloss)<1e-12
            pop='movielens_natural';f['population']=pop;frames.append(f)
            selected[(pop,entry['arm'],entry['seed'])]=json.loads((Path(entry['run_dir'])/'selection_frozen_natural.json').read_text())['selected_epoch']
            sources.append(dict(population=pop,arm=entry['arm'],seed=entry['seed'],path=str(path),sha256=sha(path),
                                evidence_grade='new frozen predictions recomputed and compared to original common points'))
        pd.DataFrame(audits).to_csv(OUT/'new_prediction_recalculation.csv',index=False)

cell=pd.concat(frames,ignore_index=True);cell.to_csv(OUT/'all_epoch_cell_metrics.csv',index=False)
pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
readouts=[]
for (pop,arm,seed),g in cell.groupby(['population','arm','seed']):
    for name,gg in [('AULC_1_'+str(n),g[g.epoch<=n]) for n in (10,16,20,50)]+[
        ('E'+str(n),g[g.epoch==n]) for n in (10,20,30,40,50)]+[
        ('selected',g[g.epoch==selected[(pop,arm,seed)]])]:
        for (domain,task),cc in gg.groupby(['domain','task']):
            readouts.append(dict(population=pop,arm=arm,seed=seed,readout=name,domain=domain,task=task,
                auc=cc.auc.mean(),logloss=cc.logloss.mean(),epochs_averaged=cc.epoch.nunique(),
                first_epoch=int(cc.epoch.min()),last_epoch=int(cc.epoch.max())))
readouts=pd.DataFrame(readouts);readouts.to_csv(OUT/'seed_window_cell_metrics.csv',index=False)
macro=readouts.groupby(['population','arm','seed','readout'])[['auc','logloss']].mean().reset_index()
macro.to_csv(OUT/'seed_window_macro_metrics.csv',index=False)
pairs=[];summaries=[];paircells=[]
for (pop,readout),g in macro.groupby(['population','readout']):
    a=g[g.arm=='SB'].set_index('seed');b=g[g.arm=='SN'].set_index('seed')
    assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
    for seed in a.index:
        pairs.append(dict(population=pop,readout=readout,seed=seed,auc=a.loc[seed,'auc']-b.loc[seed,'auc'],logloss=a.loc[seed,'logloss']-b.loc[seed,'logloss']))
    for metric in ('auc','logloss'):
        summaries.append(dict(population=pop,readout=readout,metric=metric,reference_mean=b[metric].mean(),
            candidate_mean=a[metric].mean(),**interval(a[metric]-b[metric])))
    cc=readouts[(readouts.population==pop)&(readouts.readout==readout)]
    aa=cc[cc.arm=='SB'].set_index(['seed','domain','task']);bb=cc[cc.arm=='SN'].set_index(['seed','domain','task'])
    for ix in aa.index:
        paircells.append(dict(population=pop,readout=readout,seed=ix[0],domain=ix[1],task=ix[2],
            auc=aa.loc[ix,'auc']-bb.loc[ix,'auc'],logloss=aa.loc[ix,'logloss']-bb.loc[ix,'logloss'],baseline_auc=bb.loc[ix,'auc']))
pd.DataFrame(pairs).to_csv(OUT/'paired_seed_deltas.csv',index=False)
pd.DataFrame(paircells).to_csv(OUT/'paired_cell_deltas.csv',index=False)
summary=pd.DataFrame(summaries);summary.to_csv(OUT/'paired_summary.csv',index=False)
(OUT/'analysis_status.json').write_text(json.dumps(dict(new_complete=new_complete,historical_only=not new_complete,
    windows=[10,16,20,50],auc_window_is='arithmetic mean of epoch AUC',
    same_epoch_is_not_same_update_budget=True,untouched_test=False),indent=2)+'\n')
print(summary[summary.readout.isin(['AULC_1_16','E50','selected'])].to_string(index=False))
