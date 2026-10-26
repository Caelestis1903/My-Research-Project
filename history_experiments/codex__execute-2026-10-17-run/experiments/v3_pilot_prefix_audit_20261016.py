"""Audit calibration/finalist seed overlap using actual saved draw prefixes."""
import argparse
import json
from pathlib import Path
import numpy as np
import v3_audit_20261016 as audit
base=audit.base;OUT=audit.OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 choices=json.loads((OUT/'calibration_frozen.json').read_text())['selected_lr'];records=[]
 for opt in ('adam','sgd'):
  for seed in (2022,2023):
   matches=[]
   for directory in (OUT/'runs').glob('V4pilot*'):
    if not directory.is_dir():continue
    c=json.loads((directory/'config.json').read_text())
    if c['optimizer']==opt and c['seed']==seed and c['lr']==choices[opt]:matches.append(directory)
   assert len(matches)==1;pilot=matches[0];formal=OUT/'runs'/f'V4final__kuairand__{opt}__SN__seed{seed}'
   c0=json.loads((pilot/'config.json').read_text());c1=json.loads((formal/'config.json').read_text())
   assert json.loads((formal/'receipt.json').read_text())['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
   for key in ('seed','init_sha256','source_hashes','lr','optimizer','p','q','data','split_sha256'):assert c0[key]==c1[key],key
   checks=[]
   for epoch in range(1,21):
    paths=[d/f'batches_epoch{epoch:03d}.npz' for d in (pilot,formal)];x,y=map(np.load,paths)
    assert all(np.array_equal(x[key],y[key]) for key in ('rows','sizes','order'))
    checks.append(dict(epoch=epoch,pilot_file_sha=base.sha(paths[0]),formal_file_sha=base.sha(paths[1]),array_identity=True))
   records.append(dict(optimizer=opt,seed=seed,pilot=str(pilot),formal=str(formal),initial_hash=c0['init_sha256'],epochs_checked=checks))
 base.dump(a.out/'receipt.json',dict(status='PASS',source_sha=base.sha(__file__),records=records,
  inference='Pilot seeds overlap formal cohort; fresh process directories are not additional independent training seeds. Report seed bootstrap conditional on chosen LR and acknowledge no outer LR-selection uncertainty. Fixed-LR reference contrasts are retained separately.',
  scientifically_distinct_seeds_in_formal_cohort=4,additional_seeds_created_by_rerunning_pilots=0))
if __name__=='__main__':main()
