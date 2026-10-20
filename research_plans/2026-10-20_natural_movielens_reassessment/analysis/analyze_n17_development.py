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
rows=[];sources=[];traces=[];cellrows=[]
for c in admitted:
    run=Path(c['run_dir']);r=json.loads((run/'receipt.json').read_text())
    selection=json.loads((run/'selection_frozen.json').read_text())
    f=pd.read_csv(run/'trajectory.csv');cells=pd.read_csv(run/'metrics_selection.csv')
    assert list(f.epoch)==list(range(1,c['epochs']+1)) and np.isfinite(f.selection_macro_auc).all()
    assert cells.groupby('epoch').size().eq(6).all()
    data_path=ART.parent/'data'/(c['population']+'.json')
    data=json.loads(data_path.read_text())['data'];counts=data['train_domain_counts']
    cells['train_p_weight']=cells.domain.map({int(d):n/sum(counts.values())/2 for d,n in counts.items()})
    macro=cells.groupby('epoch')[['auc','logloss']].mean()
    for epoch,gg in cells.groupby('epoch'):
        for metric in ('auc','logloss'):
            macro.loc[epoch,'train_p_cell_'+metric]=np.average(gg[metric],weights=gg.train_p_weight)
            macro.loc[epoch,'eval_p_cell_'+metric]=np.average(gg[metric],weights=gg.n)
    assert len(macro)==c['epochs'] and np.max(np.abs(macro.auc.to_numpy()-f.selection_macro_auc.to_numpy()))<1e-12
    selected=int(f.loc[f.selection_macro_auc.idxmax(),'epoch'])
    assert selected==selection['selected_epoch']==r['selected_epoch']
    assert int(f.iloc[-1].steps)==r['steps'] and int(f.iloc[-1].examples)==r['examples']
    for label,epoch in [('selected',selected),('final',c['epochs'])]:
        t=f[f.epoch.eq(epoch)].iloc[0]
        for cell in cells[cells.epoch.eq(epoch)].to_dict('records'):
            cellrows.append(dict(cell,group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],readout=label))
        rows.append(dict(group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],readout=label,
            epoch=epoch,epochs_budget=c['epochs'],main_steps=r['steps'],actual_examples=r['examples'],
            selection_auc=macro.loc[epoch,'auc'],selection_logloss=macro.loc[epoch,'logloss'],
            selection_train_p_cell_auc=macro.loc[epoch,'train_p_cell_auc'],
            selection_eval_p_cell_auc=macro.loc[epoch,'eval_p_cell_auc'],
            selection_train_p_cell_logloss=macro.loc[epoch,'train_p_cell_logloss'],
            selection_eval_p_cell_logloss=macro.loc[epoch,'eval_p_cell_logloss'],
            train_empirical_objective=t.train_empirical_objective,online_objective=t.online_objective,
            run_dir=str(run)))
    f['group']=c['group'];f['lr']=c['lr'];f['arm']=c['arm'];f['seed']=c['seed'];traces.append(f)
    sources.append(dict(group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],path=str(data_path),sha256=sha(data_path)))
    for name in ('receipt.json','config.json','trajectory.csv','metrics_selection.csv','selection_frozen.json'):
        sources.append(dict(group=c['group'],lr=c['lr'],arm=c['arm'],seed=c['seed'],path=str(run/name),sha256=sha(run/name)))
prefix_rows=[]
for c in admitted:
    if c['group']!='H200':continue
    run=Path(c['run_dir']);parent=Path(c['resume_from'])
    binding=json.loads((run/'H100_parent_frozen.json').read_text())
    assert binding['receipt_sha256']==sha(parent/'receipt.json')
    assert binding['selection_sha256']==sha(parent/'selection_frozen.json')
    assert binding['gate_sha256']==sha(ART/'plateau_frozen.json')
    a=pd.read_csv(run/'trajectory.csv');b=pd.read_csv(parent/'trajectory.csv')
    columns=['epoch','train_empirical_objective','online_objective','selection_macro_auc','steps','examples']
    assert len(b)==100 and len(a)==200
    residual=float(np.abs(a.iloc[:100][columns].to_numpy()-b[columns].to_numpy()).max())
    assert residual<1e-12
    ma=json.loads((run/'all_model_index.json').read_text());mb=json.loads((parent/'all_model_index.json').read_text())
    assert len(ma)==200 and len(mb)==100
    assert [(x['epoch'],x['sha256']) for x in ma[:100]]==[(x['epoch'],x['sha256']) for x in mb]
    prefix_rows.append(dict(run_dir=str(run),parent_dir=str(parent),trajectory_prefix_maxabs=residual,
        first_100_model_index_hashes_equal=True,parent_receipt_sha256=sha(parent/'receipt.json'),
        parent_selection_sha256=sha(parent/'selection_frozen.json'),gate_sha256=sha(ART/'plateau_frozen.json'),
        evidence='producer model indexes and immutable parent binding; later test freeze independently verifies model bytes'))
if prefix_rows:pd.DataFrame(prefix_rows).to_csv(OUT/'H200_prefix_identity.csv',index=False)
raw=pd.DataFrame(rows);raw.to_csv(OUT/'seed_selection_readouts.csv',index=False)
cell=pd.DataFrame(cellrows);cell.to_csv(OUT/'seed_selection_cell_readouts.csv',index=False)
cellpairs=[]
keys=['seed','readout','domain','task']
for (group,lr),g in cell.groupby(['group','lr']):
    a=g[g.arm.eq('SB')].set_index(keys);b=g[g.arm.eq('SN')].set_index(keys)
    assert set(a.index)==set(b.index)
    for ix in a.index:
        cellpairs.append(dict(group=group,lr=lr,contrast='SB-SN',**dict(zip(keys,ix)),
            auc=float(a.loc[ix,'auc']-b.loc[ix,'auc']),logloss=float(a.loc[ix,'logloss']-b.loc[ix,'logloss']),
            baseline_auc=float(b.loc[ix,'auc']),baseline_logloss=float(b.loc[ix,'logloss'])))
pd.DataFrame(cellpairs).to_csv(OUT/'paired_selection_cell_deltas.csv',index=False)
pd.concat(traces).to_csv(OUT/'training_and_selection_trajectories.csv',index=False)
summary=[];paired=[]
for key,g in raw.groupby(['group','lr','arm','readout']):
    for metric in ['selection_auc','selection_logloss','selection_train_p_cell_auc','selection_eval_p_cell_auc','selection_train_p_cell_logloss','selection_eval_p_cell_logloss','train_empirical_objective','online_objective','epoch']:
        summary.append(dict(group=key[0],lr=key[1],arm=key[2],readout=key[3],metric=metric,**interval(g[metric])))
for key,g in raw.groupby(['group','lr','readout']):
    a=g[g.arm=='SB'].set_index('seed');b=g[g.arm=='SN'].set_index('seed')
    assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
    for metric in ['selection_auc','selection_logloss','selection_train_p_cell_auc','selection_eval_p_cell_auc','selection_train_p_cell_logloss','selection_eval_p_cell_logloss','train_empirical_objective','online_objective']:
        paired.append(dict(group=key[0],lr=key[1],readout=key[2],metric=metric,contrast='SB-SN',**interval(a[metric]-b[metric])))
pd.DataFrame(summary).to_csv(OUT/'selection_readout_summary.csv',index=False)
pd.DataFrame(paired).to_csv(OUT/'selection_q_contrast_summary.csv',index=False)
continuations=[]
if any(c['group']=='H200' for c in admitted):
    final=raw[raw.readout.eq('final')]
    child=final[final.group.eq('H200')].set_index(['lr','arm','seed'])
    parent=final[final.group.eq('formal')].set_index(['lr','arm','seed'])
    metrics=['selection_auc','selection_logloss','selection_train_p_cell_logloss',
        'selection_eval_p_cell_logloss','train_empirical_objective']
    delta=child[metrics]-parent.loc[child.index,metrics]
    delta.to_csv(OUT/'H200_minus_H100_seed_readouts.csv')
    for (lr,arm),g in delta.groupby(['lr','arm']):
        for metric in metrics:
            continuations.append(dict(lr=lr,arm=arm,metric=metric,contrast='E200-E100',**interval(g[metric])))
    pd.DataFrame(continuations).to_csv(OUT/'H200_minus_H100_summary.csv',index=False)
    finalcell=cell[cell.readout.eq('final')]
    a=finalcell[finalcell.group.eq('H200')].set_index(['lr','arm','seed','domain','task'])
    b=finalcell[finalcell.group.eq('formal')].set_index(['lr','arm','seed','domain','task'])
    z=(a[['auc','logloss']]-b.loc[a.index,['auc','logloss']]).reset_index()
    z.to_csv(OUT/'H200_minus_H100_cell_deltas.csv',index=False)
    marginal=[]
    for axis in ['domain','task']:
        byseed=z.groupby(['arm','seed',axis])[['auc','logloss']].mean().reset_index()
        for (arm,index),gg in byseed.groupby(['arm',axis]):
            for metric in ['auc','logloss']:
                marginal.append(dict(arm=arm,axis=axis,index=index,metric=metric,**interval(gg[metric])))
    pd.DataFrame(marginal).to_csv(OUT/'H200_minus_H100_marginal_summary.csv',index=False)
pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
status=dict(status='DEVELOPMENT_ONLY',admitted_runs=len(admitted),planned_formal_now=len(configs),missing_run_ids=missing,
    test_read=False,recipes_changed=False,partial_seed_means_used=False,
    interpretation='Fixed selection population; selected is earliest validation max; not heldout generalization.')
(OUT/'analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(status)
print(pd.DataFrame(paired)[lambda x:x.metric.eq('selection_auc')].round(7).to_string(index=False))
