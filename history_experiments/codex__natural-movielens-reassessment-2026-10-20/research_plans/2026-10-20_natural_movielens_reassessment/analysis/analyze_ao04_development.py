"""Primary pre-test AO04 validation closure; all twelve full runs required."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze_core import interval
ART=Path('/home_new/xzf_natural_movielens_reassessment');ROOT=ART/'AO04'
OUT=Path(__file__).resolve().parent/'ao04_development'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze():
 configs=json.loads((ROOT/'manifest.json').read_text())['configurations'];assert len(configs)==12
 with np.load(ART/'data/movielens_natural.npz') as z:ids=z['val_row_ids'];y=z['val_y'];domains=z['val_x'][:,-1].astype(int)
 OUT.mkdir(exist_ok=True);rows=[];sources=[];audit=[]
 for cfg in configs:
  run=Path(cfg['run_dir']);r=json.loads((run/'receipt.json').read_text());sel=json.loads((run/'selection.json').read_text())
  assert r['status']=='COMPLETE_TRAINING_TEST_DEFERRED' and r['epochs']==50 and r['steps']==1000 and r['arch_steps']==50
  assert r['examples']==40008350 and r['test_evaluated'] is False
  f=pd.read_csv(run/'metrics_epoch.csv');assert set(f.epoch)==set(range(50)) and not f.duplicated(['epoch','domain','task']).any()
  assert f.groupby('epoch').size().eq(6).all()
  curve=f.groupby('epoch').validation_auc.mean();selected=int(curve.idxmax());assert selected==sel['selected_epoch']==r['selected_epoch_zero_based']
  assert abs(curve.loc[selected]-max(sel['macro_auc_by_epoch'].values()))<1e-12
  with np.load(run/'validation_predictions_selected.npz') as z:
   assert np.array_equal(z['labels'],y)  # Original stash has labels/predictions only; fixed unshuffled validation order.
   pred=z['predictions']
   for row in f[f.epoch==selected].itertuples():
    ix=domains==row.domain;yy=y[ix,row.task];pp=pred[ix,row.task].astype(float)
    auc=roc_auc_score(yy,pp);pp=np.clip(pp,1e-12,1-1e-12);ll=-np.mean(yy*np.log(pp)+(1-yy)*np.log1p(-pp))
    assert abs(auc-row.validation_auc)<1e-12 and abs(ll-row.validation_logloss)<1e-12
    audit.append(dict(run_id=cfg['id'],domain=row.domain,task=row.task,auc_residual=auc-row.validation_auc,LL_residual=ll-row.validation_logloss))
  for stage,ep in [('selected',selected),('final',49)]:
   g=f[f.epoch==ep]
   for row in g.itertuples():rows.append(dict(run_id=cfg['id'],arm=cfg['arm'],seed=cfg['seed'],stage=stage,epoch=ep+1,domain=row.domain,task=row.task,auc=row.validation_auc,logloss=row.validation_logloss,train_objective=row.train_objective))
  sources.append(dict(run_id=cfg['id'],run_dir=str(run),receipt_sha256=sha(run/'receipt.json'),selection_sha256=sha(run/'selection.json'),metrics_sha256=sha(run/'metrics_epoch.csv'),predictions_sha256=sha(run/'validation_predictions_selected.npz'),paired_init_hash=r['paired_initial_state_sha256']))
 assert pd.DataFrame(sources).assign(seed=[c['seed'] for c in configs]).groupby('seed').paired_init_hash.nunique().eq(1).all()
 cell=pd.DataFrame(rows);cell.to_csv(OUT/'seed_cell_readouts.csv',index=False)
 macro=cell.groupby(['arm','seed','stage'],as_index=False)[['auc','logloss']].mean();macro.to_csv(OUT/'seed_macro_readouts.csv',index=False)
 summary=[]
 for stage,g in macro.groupby('stage'):
  for candidate,reference in [('B1','B0'),('B2','B0'),('B1','B2')]:
   a=g[g.arm==candidate].set_index('seed');b=g[g.arm==reference].set_index('seed');assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
   for metric in ['auc','logloss']:summary.append(dict(stage=stage,contrast=candidate+'-'+reference,metric=metric,candidate_mean=float(a[metric].mean()),reference_mean=float(b[metric].mean()),**interval(a[metric]-b[metric])))
 pd.DataFrame(summary).to_csv(OUT/'four_seed_summary.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'selected_prediction_recalculation.csv',index=False)
 (OUT/'analysis_status.json').write_text(json.dumps(dict(complete=True,runs=12,paired_seeds=4,selected_prediction_cells_recomputed=len(audit),test_read=False),indent=2)+'\n')
 print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':analyze()
