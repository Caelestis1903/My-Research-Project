"""Primary AO04 local validation-probe analysis; no test or optimizer updates."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path('/home_new/xzf_natural_movielens_reassessment/AO04')
OUT=Path(__file__).resolve().parent/'ao04_development'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 configs=json.loads((ROOT/'manifest.json').read_text())['configurations'];assert len(configs)==12
 rows=[];risks=[];sources=[]
 for cfg in configs:
  folder=Path(cfg['run_dir'])/'local_fusion_diagnostics';r=json.loads((folder/'receipt.json').read_text())
  assert r['status']=='COMPLETE_RAW_DIAGNOSTIC' and r['training_updates']==0 and not r['test_read']
  for cp in r['checkpoints']:
   assert sha(cp['source'])==cp['sha256'];path=folder/(cp['checkpoint']+'_sensitivity.npz')
   with np.load(path) as z:
    assert len(z['row_ids'])==24 and np.bincount(z['domains'].astype(int)).tolist()==[8,8,8]
    rows.append(dict(arm=cfg['arm'],seed=cfg['seed'],checkpoint=cp['checkpoint'],
     q_jacobian_RMS=float(np.sqrt(np.mean(z['logit_jacobian_wrt_q']**2))),rho_jacobian_RMS=float(np.sqrt(np.mean(z['logit_jacobian_wrt_rho']**2))),
     other_expert_difference_RMS=float(np.sqrt(np.mean(z['other_expert_differences']**2))),coefficient=cp['a_D_times_one_minus_b_D'],
     q_dimension=int(z['logit_jacobian_wrt_q'].shape[-1]),probe_role=cp['B0_probe'] or 'trained arm'))
   sources.append(dict(run_id=cfg['id'],path=str(path),sha256=sha(path)))
  f=pd.read_csv(folder/'probe_cell_risks.csv');baseline=f[f.case=='baseline'].set_index(['checkpoint','domain','task'])
  f['delta_logloss']=[v.logloss-baseline.loc[(v.checkpoint,v.domain,v.task),'logloss'] for v in f.itertuples()]
  f['delta_auc']=[v.auc-baseline.loc[(v.checkpoint,v.domain,v.task),'auc'] if v.auc_defined else np.nan for v in f.itertuples()]
  f['arm']=cfg['arm'];f['seed']=cfg['seed'];risks.append(f)
  for name in ['receipt.json','probe_cell_risks.csv']:sources.append(dict(run_id=cfg['id'],path=str(folder/name),sha256=sha(folder/name)))
 OUT.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(OUT/'local_geometry.csv',index=False);pd.concat(risks,ignore_index=True).to_csv(OUT/'local_probe_risk_deltas.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'local_source_index.csv',index=False)
 (OUT/'local_analysis_status.json').write_text(json.dumps(dict(complete=True,runs=12,checkpoints=36,probe_rows_per_domain=8,test_read=False,
  scope='finite probe sensitivity, not global identifiability or population performance'),indent=2)+'\n')
 print('AO04 local development complete')
if __name__=='__main__':main()
