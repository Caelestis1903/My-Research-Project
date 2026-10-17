"""Primary verification and descriptive four-seed geometry summaries."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())

def main():
 manifest=O/'V11_shadow_geometry_supplement_manifest.json';m=load(manifest);q=load(manifest.with_suffix('.status.json'));frames=[];records=[];max_triangle=0.;max_partition=0.;max_raw_cosine_error=0.
 for task in m['tasks']:
  actual=q['tasks'][task['name']];assert actual['status']=='COMPLETE' and actual['exit_code']==0
  cmd=task['command'];d=Path(cmd[cmd.index('--out')+1]);rc=load(d/'receipt.json');parent=Path(rc['parent']);pr=load(parent/'receipt.json')
  assert rc['status']=='COMPLETE' and rc['performance_recomputed'] is False and rc['test_access'] is False
  assert sha(R/'experiments/v3_shadow_geometry_20261016.py')==rc['source_sha']
  assert sha(R/'experiments/v3_state_shadow_20261016.py')==rc['replay_source_sha']==pr['source_sha']
  assert rc['checkpoint_checks']==pr['checkpoint_checks'] and rc['draws']==pr['draws']
  for name,h in rc['parent_hashes'].items():assert sha(parent/name)==h
  for source,h in zip(rc['source_runs'],rc['source_receipt_sha']):assert sha(Path(source)/'receipt.json')==h
  for check in rc['checkpoint_checks']:
   assert all(x['l2_error']==0. for x in check['comparisons'])
   assert sha(Path(rc['source_runs'][0])/f'epoch{check["epoch"]:03d}.pt')==check['checkpoint_sha']
  for name,h in rc['hashes'].items():assert sha(d/name)==h
  f=pd.read_csv(d/'update_geometry.csv');pf=pd.read_csv(parent/'summary.csv');assert len(f)==rc['rows']==435==5*len(pf)
  keys=['seed','epoch','current_arm','condition','norm_control','norm_informative']
  assert set(f[keys].itertuples(index=False,name=None))==set(pf[keys].itertuples(index=False,name=None))
  assert not f.duplicated(keys+['block']).any()
  good=f.comparison_informative.astype(bool);assert np.array_equal(good.to_numpy(),(f[['delta_norm','native_norm']].min(axis=1)>=1e-12).to_numpy())
  assert f.loc[~good,'cosine_to_native'].isna().all()
  g=f[good];assert (abs(g.cosine_to_native)<=1+1e-12).all()
  dn,nn,dd,cos=[g[n].to_numpy() for n in ['delta_norm','native_norm','delta_minus_native_norm','cosine_to_native']]
  residual=abs(dd*dd-(dn*dn+nn*nn-2*dn*nn*cos))/np.maximum(dn*dn+nn*nn+dd*dd,1e-30);assert residual.max()<1e-12;max_triangle=max(max_triangle,float(residual.max()))
  native=f[(f.condition=='native')&good];assert np.max(abs(native.cosine_to_native-1))<1e-12
  nonzero=f[f.cosine_to_same_condition_raw.notna()];e=float(abs(nonzero.cosine_to_same_condition_raw-1).max());assert e<2e-6;max_raw_cosine_error=max(max_raw_cosine_error,e)
  for _,z in f.groupby(keys):
   assert set(z.block)=={'ALL','C','P','R','E'}
   allrow=z[z.block=='ALL'].iloc[0];blocks=z[z.block!='ALL']
   for col in ['delta_norm','native_norm','delta_minus_native_norm']:
    squared=float((blocks[col]**2).sum());total=float(allrow[col]**2);e=abs(squared-total)/max(squared+total,1e-30);assert e<1e-12;max_partition=max(max_partition,e)
  matched=f[(f.block=='ALL')&(f.norm_control=='match_native_global_norm')];e=float((abs(matched.delta_norm-matched.native_norm)/matched.native_norm).max());assert e<=2e-6 and abs(e-rc['norm_match_max_relative_error'])<1e-12
  f['delta_norm_relative_native']=np.where(f.native_norm>=1e-12,f.delta_norm/f.native_norm,np.nan)
  frames.append(f);records.append(dict(seed=rc['seed'],rows=len(f),receipt_sha=sha(d/'receipt.json'),parent_receipt_sha=sha(parent/'receipt.json'),norm_match_max_relative_error=e))
 allrows=pd.concat(frames,ignore_index=True);assert sorted(allrows.seed.unique())==[2022,2023,2024,2025]
 summary=[];metrics=['delta_norm','native_norm','delta_minus_native_norm','cosine_to_native','cosine_to_same_condition_raw','delta_norm_relative_native']
 for key,z in allrows.groupby(['epoch','current_arm','condition','norm_control','block']):
  assert sorted(z.seed)==[2022,2023,2024,2025]
  for metric in metrics:
   v=z.sort_values('seed')[metric].to_numpy();finite=v[np.isfinite(v)];summary.append(dict(zip(['epoch','current_arm','condition','norm_control','block'],key))|dict(metric=metric,finite_seeds=len(finite),mean=float(finite.mean()) if len(finite) else None,minimum=float(finite.min()) if len(finite) else None,maximum=float(finite.max()) if len(finite) else None,seed_values=json.dumps([None if not np.isfinite(x) else float(x) for x in v])))
 out=O/'analysis/V11_shadow_geometry_all4';out.mkdir(exist_ok=False);allrows.to_csv(out/'all4_geometry.csv',index=False);pd.DataFrame(summary).to_csv(out/'descriptive_summary.csv',index=False)
 result=dict(status='PASS',time=time.time(),source_sha=sha(Path(__file__)),independent_review=False,records=records,rows=len(allrows),descriptive_rows=len(summary),max_squared_triangle_relative_residual=max_triangle,max_block_partition_relative_residual=max_partition,max_cosine_to_same_condition_raw_error=max_raw_cosine_error,test_access=False,scope='Same frozen parent theta/m/v replay and batches; all source/parent/output hashes/current producers and12physical checkpoints verified. Scalar triangle and disjoint block norm identities checked; same-condition global normalization preserves direction. Four-seed means/ranges are descriptive, no confidence or causal mediation claim from angles alone.',hashes={name:sha(out/name) for name in ['all4_geometry.csv','descriptive_summary.csv']})
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(O/'V11_shadow_geometry_all4_primary_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','hashes','scope']}))
if __name__=='__main__':main()
