"""Read historical bytes; write only audit outputs. No torch/model imports."""
from pathlib import Path
import csv,json,hashlib,sys,platform,subprocess
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(name,rows):
 if rows: pd.DataFrame(rows).to_csv(OUT/name,index=False)
def read(p):return json.loads(p.read_text())

if sys.argv[-1]=='inventory':
 rows=[]
 for top in ['experiments','research_R2','research_R1_R3','m3oe','research_context','research_plans','exp_ana','promt']:
  candidates=sorted((ROOT/top).rglob('*'))
  if top=='experiments': candidates+=sorted(ROOT.glob('*.md'))
  for p in candidates:
   if not p.is_file() or OUT in p.parents or '__pycache__' in p.parts:continue
   rows.append(dict(source=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p),source_kind='local_existing_bytes',immutable_path=None,generation_command='unknown_historical; inspect source manifest'))
 dump('artifact_inventory.json',rows)
 missing=[]
 for stage in ['S4_A2','S5','S6','S8']:
  files=[r['source'] for r in rows if f'stage_{stage}_' in r['source']]
  for name in ['config.json','run_manifest.json','predictions_test.npz','selected_checkpoint.pth','parameter_manifest.json']:
   found=[s for s in files if s.endswith('/'+name)]
   if not found:missing.append(dict(stage=stage,artifact=name,status='INCOMPLETE',search_scope='all inventoried local stage paths'))
 for ds,setting,arms in [('kuairand','natural',['B0','B1','B2']),('movielens','d0_large',['B0','B1','B2'])]:
  for arm in arms:
   for seed in [2022,2023]:
    p=ROOT/f'research_R2/formal_runs_20261012/{ds}/{setting}/m3oe/{arm}/seed{seed}/attempt1'
    for name in ['run_manifest.json','config.json','selection.json','metrics_epoch.csv','metrics_test.csv','validation_predictions_selected.npz','predictions_test.npz','selected_checkpoint.pth','batch_trace.csv','parameter_manifest.json']:
     if not (p/name).is_file():missing.append(dict(stage='AO04',dataset=ds,arm=arm,seed=seed,artifact=name,status='INCOMPLETE',expected_path=str((p/name).relative_to(ROOT))))
 for stage,arms in [('S5',['F','T','HT','DH']),('S6',['F','DH']),('S6ML',['F','DH'])]:
  for arm in arms:
   for seed in [2022,2023,2024,2025]:
    p=ROOT/f'experiments/stage_{stage}_raw_results/{arm}/seed{seed}/joint'
    for name in ['config.json','run_manifest.json','metrics_epoch.csv','metrics_test.csv','predictions_test.npz','selected_checkpoint.pth','optimizer_state.pth','architecture_optimizer_state.pth','parameter_manifest.json']:
     if not (p/name).is_file():missing.append(dict(stage=stage,arm=arm,seed=seed,artifact=name,status='INCOMPLETE',expected_path=str((p/name).relative_to(ROOT)),note='canonical path absent; inventory searched alternative stage archives separately'))
 dump('missing_artifacts.json',missing)
 print('inventory',len(rows),'missing',len(missing));sys.exit()

if sys.argv[-1]=='metrics':
 base=ROOT/'research_R2/formal_runs_20261012'
 allz=base/'kuairand/natural/m3oe/B0/_final_test_evaluated/predictions_test.npz'
 with np.load(allz,allow_pickle=False) as z: ids=z['row_ids']; labels=z['labels']; domains=z['physical_domain']
 source=Path('/home/xzf/data/kuairand/processed/kuai_rand.csv')
 raw=pd.read_csv(source,usecols=['tab','is_click','long_view'])
 # Historical 'physical_domain' actually stores local IDs (S1 local_to_physical).
 # Verify the tab-to-local mapping against raw rows; do not infer from counts.
 source_checks={'source_sha256':sha(source),'source_rows':len(raw),'saved_domain_encoding':'local IDs despite physical_domain field name; 0=tab0,1=tab1,2=tab4','test_row_ids_unique':len(np.unique(ids))==len(ids),'test_row_ids_in_bounds':bool(((ids>=0)&(ids<len(raw))).all()),'labels_equal_source':bool(np.array_equal(raw.iloc[ids][['is_click','long_view']].to_numpy(),labels)),'domains_equal_source':bool(np.array_equal(raw.iloc[ids]['tab'].map({0:0,1:1,4:2}).to_numpy(),domains))}
 dump('dataset_prediction_binding.json',source_checks)
 assert source_checks['labels_equal_source'] and source_checks['domains_equal_source']
 splitstats=[]
 n=len(raw); cuts=[0,int(n*.8),int(n*.9),n]
 for sp,a,b in zip(['train','validation','test'],cuts[:-1],cuts[1:]):
  v=raw.iloc[a:b]
  for d in [0,1,4]:
   for t,col in enumerate(['is_click','long_view']):
    y=v.loc[v.tab==d,col].to_numpy();splitstats.append(dict(split=sp,physical_domain=d,task=t,n=len(y),n_pos=int(y.sum()),n_neg=int(len(y)-y.sum()),positive_rate=float(y.mean())))
 save('split_cell_counts.csv',splitstats)
 metricrows=[]; checks=[]; selection=[]; codematches=[]; identity=[]
 for mp in sorted(base.rglob('run_manifest.json')):
  p=mp.parent;m=read(mp)
  identity.append({k:m.get(k) for k in ['dataset','arm','seed','status','epochs_completed','initial_parameter_state_hash','batch_order_hash','architecture_batch_order_hash','selected_epoch']})
  if (p/'config.json').exists():
   cfg=read(p/'config.json')
   for name,expected in cfg.get('code',{}).items():
    cp=ROOT/'research_R2/src'/name
    codematches.append(dict(run=str(p.relative_to(ROOT)),source=str(cp.relative_to(ROOT)),expected=expected,actual=sha(cp) if cp.exists() else None,match=cp.exists() and sha(cp)==expected))
  if (p/'metrics_epoch.csv').exists() and m.get('selected_epoch') is not None:
   df=pd.read_csv(p/'metrics_epoch.csv'); means=df.groupby('epoch').validation_auc.mean(); epoch=int(means.idxmax())
   selection.append(dict(run=str(p.relative_to(ROOT)),recomputed_epoch=epoch,manifest_epoch=m['selected_epoch'],match=epoch==m['selected_epoch']))
  pred=p/'predictions_test.npz'
  if not pred.exists() or m.get('dataset')!='kuairand':continue
  with np.load(pred,allow_pickle=False) as z:
   assert np.array_equal(z['row_ids'],ids) and np.array_equal(z['labels'],labels)
   csvrows=pd.read_csv(p/'metrics_test.csv')
   for key in z.files:
    if not key.startswith('pred_'):continue
    predarr=z[key].astype(np.float64); checkpoint=key[5:]
    for d,physical in enumerate([0,1,4]):
     mask=domains==d
     for t in [0,1]:
      y=labels[mask,t].astype(np.float64); pr=predarr[mask,t]
      assert np.isfinite(pr).all() and ((pr>=0)&(pr<=1)).all()
      auc=float(roc_auc_score(y,pr)); ll={}
      for eps in [1e-12,1e-7]:
       pp=np.clip(pr,eps,1-eps);ll[eps]=float(-(y*np.log(pp)+(1-y)*np.log1p(-pp)).mean())
      ref=csvrows[(csvrows.checkpoint==checkpoint)&(csvrows.domain==d)&(csvrows.task==t)]
      assert len(ref)==1
      row=dict(dataset='kuairand',arm=m['arm'],seed=m['seed'],checkpoint=checkpoint,domain=d,task=t,n=len(y),n_pos=int(y.sum()),auc=auc,logloss=ll[1e-12],logloss_eps1e7=ll[1e-7],auc_residual=auc-float(ref.iloc[0].test_auc),logloss_residual=ll[1e-12]-float(ref.iloc[0].test_logloss),source=str(pred.relative_to(ROOT)))
      metricrows.append(row)
   vp=p/'validation_predictions_selected.npz'
   if vp.exists():
    with np.load(vp,allow_pickle=False) as v:
     checks.append(dict(run=str(p.relative_to(ROOT)),npz_sha256=sha(vp),manifest_sha256=m.get('selected_validation_predictions_sha256'),hash_matches=sha(vp)==m.get('selected_validation_predictions_sha256'),row_ids_present='row_ids' in v.files,rows=len(v['labels']),status='IDENTITY_INCOMPLETE_NO_ROW_IDS'))
 save('ao04_test_metric_recalc.csv',metricrows);save('selection_checks.csv',selection);save('source_hash_checks.csv',codematches);dump('ao04_validation_identity.json',checks);dump('run_identity.json',identity)
 # Existing selected aggregate CSV is a separate L2 source, including unavailable predictions.
 ag=pd.read_csv(ROOT/'research_R2/analysis/selected_aggregate_metrics.csv'); pairs=[]
 for (ds,seed,split),g in ag.groupby(['dataset','seed','split']):
  for a,b in [('B1','B0'),('B1','B2')]:
   aa=g[g.arm==a];bb=g[g.arm==b]
   if len(aa)==len(bb)==1:
    r=dict(dataset=ds,seed=int(seed),split=split,contrast=a+'-'+b,grade='L2')
    for col in ag.columns:
     if col.endswith('_auc') or col.endswith('_logloss'):r['delta_'+col]=float(aa.iloc[0][col])-float(bb.iloc[0][col])
    pairs.append(r)
 save('ao04_selected_paired_l2.csv',pairs)
 print('metrics',len(metricrows),'selection',selection,'max_auc_residual',max(abs(r['auc_residual']) for r in metricrows),'max_ll_residual',max(abs(r['logloss_residual']) for r in metricrows))
