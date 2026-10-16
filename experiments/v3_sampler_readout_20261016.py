"""Exact sampler-intervention identity checks and exposure measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_audit_20261016 as audit
base=audit.base

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads(a.manifest.read_text());identities=[];exposure=[];first=[];observations=[]
 directories=set()
 for comparison in manifest['comparisons']:
  baseline,variant=map(Path,(comparison['baseline'],comparison['variant']));kind=comparison['intervention'];assert kind in ('ORDER','PERM')
  configs=[json.loads((d/'config.json').read_text()) for d in (baseline,variant)]
  for key in ('dataset','seed','arm','optimizer','lr','batch_size','init_sha256','split_sha256','corruption','duplication','data'):assert configs[0][key]==configs[1][key],key
  producers=[json.loads((d/'receipt.json').read_text()) for d in (baseline,variant)]
  b=audit.bundle(configs[0]['dataset'],configs[0]['seed']);files=sorted(baseline.glob('batches_epoch*.npz'));others=sorted(variant.glob('batches_epoch*.npz'));assert len(files)==len(others)
  for file,other in zip(files,others):
   assert file.name==other.name
   for path,producer in zip((file,other),producers):assert base.sha(path)==producer['artifact_hashes'][path.name],f'Batch hash mismatch {path}'
   x=np.load(file);y=np.load(other);assert np.array_equal(x['sizes'],y['sizes'])
   changed=not np.array_equal(x['order'],y['order'])
   if kind=='ORDER':
    assert np.array_equal(x['rows'],y['rows'])
    assert np.array_equal(np.sort(x['order']),np.sort(y['order']))
   else:
    assert np.array_equal(x['order'],y['order'])
    assert np.array_equal(b.train_x[x['rows'],-1],b.train_x[y['rows'],-1])
   identities.append(dict(intervention=kind,baseline=str(baseline),variant=str(variant),epoch=int(file.stem.split('epoch')[1]),row_count=len(x['rows']),batch_count=len(x['sizes']),order_changed=changed,identity='PASS',baseline_batch_sha=base.sha(file),variant_batch_sha=base.sha(other)))
  directories.update((baseline,variant))
 for directory in sorted(directories):
  c=json.loads((directory/'config.json').read_text());b=audit.bundle(c['dataset'],c['seed']);t=pd.read_csv(directory/'trajectory.csv');e=np.load(directory/'exposure_final.npz');n=len(b.train_x)
  for row in t.to_dict('records'):
   for d in range(3):
    count=b.train_counts[d];draws=row[f'exposure_d{d}'];unique=row[f'unique_epoch_d{d}']
    exposure.append(dict(run=str(directory),seed=c['seed'],optimizer=c['optimizer'],arm=c['arm'],replacement=c['replacement'],order=c['order'],epoch=row['epoch'],domain=d,pool_support=count,draws=draws,unique=unique,
      repeated_draws=draws-unique,min_repeated_draws_at_fixed_quota=max(0,draws-count),coverage_epoch=unique/count,coverage_cumulative=row[f'coverage_d{d}'],repeat_histogram=row[f'repeat_histogram_d{d}']))
  for d in range(3):
   pool=np.flatnonzero(b.train_x[:,-1]==d);seen=e['first_seen_step'][pool];observed=np.sort(seen[seen>=0]);target=int(np.ceil(.9*len(pool)))
   first.append(dict(run=str(directory),seed=c['seed'],arm=c['arm'],domain=d,seen_rows=len(observed),pool_support=len(pool),first_seen_median_step=float(np.median(observed)) if len(observed) else None,
     time_to90pct_step=int(observed[target-1]) if len(observed)>=target else None,never_seen=len(pool)-len(observed),total_draws=int(e['counts'][pool].sum())))
  frame=pd.read_csv(directory/'optimizer_observations.csv');frame['run']=str(directory);frame['seed']=c['seed'];frame['arm']=c['arm'];frame['optimizer']=c['optimizer'];observations.extend(frame.to_dict('records'))
 pd.DataFrame(identities).to_csv(a.out/'intervention_identity.csv',index=False);pd.DataFrame(exposure).to_csv(a.out/'coverage_repetition.csv',index=False)
 pd.DataFrame(first).to_csv(a.out/'first_seen_summary.csv',index=False);pd.DataFrame(observations).to_csv(a.out/'moving_theta_optimizer_observations.csv',index=False)
 base.dump(a.out/'receipt.json',dict(status='PASS',source_sha=base.sha(__file__),comparisons=manifest['comparisons'],epochs_checked=len(identities),
  scope='ORDER identical exact batches; PERM identical per-row physical-domain sequence and exact quotas; no zero-repetition assertion',
  identifiability='For fixed draw count E and realized unique count U, repeated draws R=E-U. Cumulative coverage and repeated draws cannot be independently varied under fixed exposure merely by relabeling the same row multiset.'))
if __name__=='__main__':main()
