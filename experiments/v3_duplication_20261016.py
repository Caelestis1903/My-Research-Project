"""Exact coupled alias duplication: raw finite-pool invariance receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import v3_audit_20261016 as audit
import v3_local_20261016 as local
base=audit.base;OUT=audit.OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads(a.manifest.read_text());records=[]
    for directory in map(Path,manifest['baseline_runs']):
        c=json.loads((directory/'config.json').read_text());receipt=json.loads((directory/'receipt.json').read_text())
        assert c['replacement']=='WR' and c['duplication']==1 and c['corruption']==0 and c['optimizer']=='sgd'
        assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        b=audit.bundle(c['dataset'],c['seed']);small=int(np.argmin(c['p']));pool=np.flatnonzero(b.train_x[:,-1]==small)
        inverse=np.full(len(b.train_x),-1,dtype=np.int64);inverse[pool]=np.arange(len(pool))
        model=base.fixed_model(b,c['seed']).to(a.device);checkpoint=directory/'epoch010.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);theta=local.vector(model).clone()
        for factor in (2,4):
            rng=np.random.default_rng(c['seed']+factor*100000+20261016);virtual=np.tile(pool,factor);chain=hashlib.sha256();count=0;checked=0
            for file in sorted(directory.glob('batches_epoch*.npz')):
                batches=np.load(file);rows=batches['rows'];chosen=rows[b.train_x[rows,-1]==small]
                alias=rng.integers(0,factor,size=len(chosen));indices=inverse[chosen]+alias*len(pool)
                mapped=virtual[indices];assert np.array_equal(mapped,chosen)
                chain.update(indices.astype(np.int64).tobytes());count+=len(chosen);checked+=len(rows)
            assert checked==receipt['examples']
            # Actual production autograd and optimizer update oracle at common theta/state.
            sampler=base.Sampler(b,np.array(c['q']),c['seed']+20261016,'iid');rows=sampler.sample(c['batch_size']);mapped=rows.copy()
            mask=b.train_x[rows,-1]==small;alias=rng.integers(0,factor,size=int(mask.sum()))
            mapped[mask]=virtual[inverse[rows[mask]]+alias*len(pool)]
            gradients=[];deltas=[];losses=[]
            for draw in (rows,mapped):
                local.load_vector(model,theta);opt=local.optimizer(model,'sgd',c['lr']);opt.load_state_dict(state['optimizer'])
                gradient,loss=audit.grad(model,b,draw,torch.tensor(np.array(c['p'])/np.array(c['q']),device=a.device,dtype=torch.float32),a.device)
                gradients.append(gradient.clone());losses.append(loss);opt.step();deltas.append((local.vector(model)-theta).clone())
            assert torch.equal(gradients[0],gradients[1]) and torch.equal(deltas[0],deltas[1]) and losses[0]==losses[1]
            records.append(dict(seed=c['seed'],arm=c['arm'],factor=factor,epochs=receipt['epochs'],original_support=len(pool),nominal_pool=factor*len(pool),
                total_examples_checked=checked,small_draws_checked=count,virtual_index_chain_sha256=chain.hexdigest(),max_gradient_difference=float((gradients[0]-gradients[1]).abs().max()),
                max_update_difference=float((deltas[0]-deltas[1]).abs().max()),loss_difference=losses[1]-losses[0],baseline_run=str(directory),checkpoint_sha=base.sha(checkpoint)))
    pd.DataFrame(records).to_csv(a.run_dir/'coupled_identity.csv',index=False)
    base.dump(a.run_dir/'receipt.json',dict(status='PASS',source_sha=base.sha(__file__),records=records,
        implementation='uniform original row plus independent uniform alias; virtual np.tile pool, mapped rows identical',
        full_training_reuse='exact coupled identical-input optimizer-state induction; not separate independent reruns',
        inferential_scope='WR empirical distribution unchanged; duplicates add zero independent support; no PERM invariance claim'))
if __name__=='__main__':main()
