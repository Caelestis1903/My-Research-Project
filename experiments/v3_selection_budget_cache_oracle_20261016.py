from pathlib import Path
import json,sys,time
import numpy as np
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sys.path.insert(0,str(R/'experiments'));import v3_readout_20261016 as r
D=O/'analysis/V2readout__sgd__seed2022__until050';receipt=json.loads((D/'receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['source_sha']==r.base.sha(r.__file__);source=Path(receipt['source_runs'][0]);indices=[0,7,8,9];producer=json.loads((source/'receipt.json').read_text());assert r.base.sha(source/'receipt.json')==receipt['source_run_receipt_sha'][0]
predictions=[]
for i in indices:
 p=source/f'validation_epoch{i+1:03d}.npz';assert r.base.sha(p)==producer['artifact_hashes'][p.name];predictions.append(np.load(p)['predictions'])
b=r.audit.bundle('kuairand',2022);split=np.load(O/'splits/kuairand_seed2022.npz')['selection'];got=r.bootstrap_curves(b.val_y,b.val_x[:,-1].astype(int),np.asarray([predictions]),split,1000,2022+20261016)[:,0];expected=np.load(D/'bootstrap_curves.npz')['selection'][:,0,indices];error=float(abs(got-expected).max());assert error<1e-12
x=dict(status='PASS_ACTUAL_FULL_CPU50_TO_SINGLE_ARM4EPOCH_ALL1000_BOOTSTRAP',time=time.time(),independent_review=False,source_receipt_sha=r.base.sha(D/'receipt.json'),source_run=str(source),epochs=[i+1 for i in indices],replicates=1000,bitexact=bool(np.array_equal(got,expected)),max_error=error,algorithm='Samefrozen655multinomialRNG; actualfull1000pairedCPU50 reference vs onearmfouractualepochs. No newtraining/test/selectiondefinition.',role='Actualdataarithmeticacceptancebeforesparsebudgetresultcanbeused; toyverificationprecededformaldispatch.')
(O/'selection_budget_sparse_actual_full_cache_oracle.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
