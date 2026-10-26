"""Primary selection-only analysis of the complete frozen 16-pilot response surface."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_pilots'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
response=pd.read_csv(ART/'pilot_response.csv');assert len(response)==16
rows=[];sources=[]
for row in response.itertuples():
    run=ART/'pilot'/('lr'+str(row.lr))/row.arm/('seed'+str(row.seed))
    receipt=json.loads((run/'receipt.json').read_text());selection=json.loads((run/'selection_frozen.json').read_text())
    assert sha(run/'receipt.json')==row.receipt_sha256
    assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
    assert (receipt['epochs'],receipt['steps'],receipt['examples'])==(100,2000,80016700)
    assert selection['test_read'] is False
    trajectory=pd.read_csv(run/'trajectory.csv');assert list(trajectory.epoch)==list(range(1,101))
    assert np.isfinite(trajectory.selection_macro_auc).all()
    best=float(trajectory.selection_macro_auc.max());epoch=int(trajectory.loc[trajectory.selection_macro_auc.idxmax(),'epoch'])
    assert abs(best-row.selected_score)<1e-12 and epoch==selection['selected_epoch']==receipt['selected_epoch']
    cell=pd.read_csv(run/'metrics_selection.csv');assert cell.groupby('epoch').size().eq(6).all()
    macro=cell.groupby('epoch')[['auc','logloss']].mean();assert len(macro)==100
    assert np.max(np.abs(macro.auc.to_numpy()-trajectory.selection_macro_auc.to_numpy()))<1e-12
    for name in ('receipt.json','selection_frozen.json','trajectory.csv','metrics_selection.csv'):
        sources.append(dict(run_id=run.name,lr=row.lr,arm=row.arm,seed=row.seed,path=str(run/name),sha256=sha(run/name)))
    rows.append(dict(lr=row.lr,arm=row.arm,seed=row.seed,selected_epoch=epoch,
        selected_auc=best,selected_logloss=macro.loc[epoch,'logloss'],final_auc=macro.loc[100,'auc'],
        final_logloss=macro.loc[100,'logloss'],epochs=100,steps=2000,examples=80016700,
        run_dir=str(run),receipt_sha256=sha(run/'receipt.json')))
f=pd.DataFrame(rows);f.to_csv(OUT/'selection_only_pilot_runs.csv',index=False)
summary=f.groupby(['lr','arm'])[['selected_auc','selected_logloss','final_auc','final_logloss','selected_epoch']].agg(['mean','std']).reset_index()
summary.columns=['_'.join(x).rstrip('_') for x in summary.columns];summary.to_csv(OUT/'response_surface.csv',index=False)
common=f.groupby('lr').selected_auc.mean().sort_index();winner=float(common.idxmax())
own={arm:float(g.groupby('lr').selected_auc.mean().sort_index().idxmax()) for arm,g in f.groupby('arm')}
frozen=json.loads((ART/'winners_frozen.json').read_text());assert frozen['common_lr']==winner and frozen['own_lr']==own
assert frozen['test_read'] is False and frozen['response_sha256']==sha(ART/'pilot_response.csv')
for name in ('pilot_response.csv','winners_frozen.json','winner_run_usage.json','post_pilot_manifest.json'):
    raw=(ART/name).read_bytes();(OUT/name).write_bytes(raw)
    sources.append(dict(path=str(ART/name),sha256=sha(ART/name)))
(OUT/'source_files.json').write_text(json.dumps(sources,indent=2)+'\n')
(OUT/'analysis_status.json').write_text(json.dumps(dict(pilots_complete=16,formal_results_read=False,test_read=False,
    common_lr=winner,own_lr=own,scope='selection records and full budget receipts; no new prediction evaluation'),indent=2)+'\n')
print(summary.round(7).to_string(index=False));print('COMMON',common.to_dict(),'WINNER',winner,'OWN',own)
