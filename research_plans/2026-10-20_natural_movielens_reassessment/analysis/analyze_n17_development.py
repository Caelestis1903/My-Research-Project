"""Primary development-only summary; never opens test artifacts or chooses new recipes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_core import interval

ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_development'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
configs={}
for name in ('base_manifest.json','post_pilot_manifest.json','H200_manifest.json'):
    path=ART/name
    if not path.exists():continue
    for c in json.loads(path.read_text())['configurations']:
        if c['group']!='pilot' and c.get('run_dir'):configs[c['id']]=c
planned=pd.DataFrame(configs.values());admitted=[];missing=[]
for key,g in planned.groupby(['group','lr']):
    ready=[]
    for c in g.to_dict('records'):
        path=Path(c['run_dir'])/'receipt.json'
        if not path.exists():missing.append(c['id']);continue
        receipt=json.loads(path.read_text())
        if receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED' and receipt['epochs']==c['epochs']:ready.append(c)
        else:missing.append(c['id'])
    if len(ready)==len(g):
        assert len(g)==8 and set(g.seed)=={2022,2023,2024,2025} and set(g.arm)=={'SN','SB'}
        admitted.extend(ready)
rows=[];sources=[];traces=[]
for c in admitted:
    run=Path(c['run_dir']);r=json.loads((run/'receipt.json').read_text())
    selection=json.loads((run/'selection_frozen.json').read_text())
    f=pd.read_csv(run/'trajectory.csv');cells=pd.read_csv(run/'metrics_selection.csv')
    assert list(f.epoch)==list(range(1,c['epochs']+1)) and np.isfinite(f.selection_macro_auc).all()
    assert cells.groupby('epoch').size().eq(6).all()
    macro=cells.groupby('epoch')[['auc','logloss']].mean()
    assert len(macro)==c['epochs'] and np.max(np.abs(macro.auc.to_numpy()-f.selection_macro_auc.to_numpy()))<1e-12
    selected=int(f.loc[f.selection_macro_auc.idxmax(),'epoch'])
    assert selected==selection['selected_epoch']==r['selected_epoch']
    assert int(f.iloc[-1].steps)==r['steps'] and int(f.iloc[-1].examples)==r['examples']
    for label,epoch in [('selected',selected),('final',c['epochs'])]:
        t=f[f.epoch.eq(epoch)].iloc[0]
        rows.append(dict(group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],readout=label,
            epoch=epoch,epochs_budget=c['epochs'],main_steps=r['steps'],actual_examples=r['examples'],
            selection_auc=macro.loc[epoch,'auc'],selection_logloss=macro.loc[epoch,'logloss'],
            train_empirical_objective=t.train_empirical_objective,online_objective=t.online_objective,
            run_dir=str(run)))
    f['group']=c['group'];f['lr']=c['lr'];f['arm']=c['arm'];f['seed']=c['seed'];traces.append(f)
    for name in ('receipt.json','config.json','trajectory.csv','metrics_selection.csv','selection_frozen.json'):
        sources.append(dict(group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],path=str(run/name),sha256=sha(run/name)))
raw=pd.DataFrame(rows);raw.to_csv(OUT/'seed_selection_readouts.csv',index=False)
pd.concat(traces).to_csv(OUT/'training_and_selection_trajectories.csv',index=False)
summary=[];paired=[]
for key,g in raw.groupby(['group','lr','arm','readout']):
    for metric in ['selection_auc','selection_logloss','train_empirical_objective','online_objective','epoch']:
        summary.append(dict(group=key[0],lr=key[1],arm=key[2],readout=key[3],metric=metric,**interval(g[metric])))
for key,g in raw.groupby(['group','lr','readout']):
    a=g[g.arm=='SB'].set_index('seed');b=g[g.arm=='SN'].set_index('seed')
    assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
    for metric in ['selection_auc','selection_logloss','train_empirical_objective','online_objective']:
        paired.append(dict(group=key[0],lr=key[1],readout=key[2],metric=metric,contrast='SB-SN',**interval(a[metric]-b[metric])))
pd.DataFrame(summary).to_csv(OUT/'selection_readout_summary.csv',index=False)
pd.DataFrame(paired).to_csv(OUT/'selection_q_contrast_summary.csv',index=False)
pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
status=dict(status='DEVELOPMENT_ONLY',admitted_runs=len(admitted),planned_formal_now=len(configs),missing_run_ids=missing,
    test_read=False,recipes_changed=False,partial_seed_means_used=False,
    interpretation='Fixed selection population; selected is earliest validation max; not heldout generalization.')
(OUT/'analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(status)
print(pd.DataFrame(paired)[lambda x:x.metric.eq('selection_auc')].round(7).to_string(index=False))
