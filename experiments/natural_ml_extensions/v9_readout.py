"""Read all already-trained AC-V9 models; no training or selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.natural_ml_reassessment.bootstrap import ROOT, activate, dump, sha
from experiments.natural_ml_reassessment.data import load_population

activate('v3')
import v3_train_20261016 as original

OUT = ROOT/'V9_trajectory_readout'
LOCK = OUT/'frozen_readout.json'


def freeze():
    runs = [r for r in json.loads((ROOT/'core_manifest.json').read_text())['runs'] if r['stage']=='AC-V9']
    assert len(runs)==8
    assert {(r['arm'],r['seed']) for r in runs}=={(a,s) for a in ('SN','SB') for s in (2022,2023,2024,2025)}
    entries=[]
    for cfg in runs:
        run=Path(cfg['run_dir'])
        assert json.loads((run/'run_manifest.json').read_text())['status']=='COMPLETE'
        names=['config.json','receipt.json','selection_frozen_natural.json']+[f'epoch{e:03d}_model.pt' for e in range(1,51)]
        entries.append(dict(run_id=cfg['run_id'],run_dir=str(run),arm=cfg['arm'],seed=cfg['seed'],
                            hashes={name:sha(run/name) for name in names}))
    frozen=dict(study='AC-V9 frozen whole trajectory readout',all_training_complete=True,
                optimizer_updates=0,epochs=list(range(1,51)),windows=[10,16,20,50],
                independent_review=False,reviewer_verdict='PRIMARY_PASS_UNDER_START_MD',
                source_sha256=sha(__file__),runs=entries)
    if LOCK.exists():
        assert json.loads(LOCK.read_text())==frozen
    else:
        dump(LOCK,frozen)
    return frozen


def run_readout(run_id,device):
    frozen=json.loads(LOCK.read_text())
    assert frozen['all_training_complete'] and frozen['source_sha256']==sha(__file__)
    entry=next(r for r in frozen['runs'] if r['run_id']==run_id)
    directory=Path(entry['run_dir']);out=OUT/run_id;out.mkdir(parents=True,exist_ok=True)
    for name,value in entry['hashes'].items():
        assert sha(directory/name)==value, name
    torch.set_num_threads(2)
    target=torch.device(device)
    if target.type=='cuda':torch.cuda.set_device(target)
    original.base.setup_seed(entry['seed'])
    bundle=load_population('movielens_natural')
    model=original.base.fixed_model(bundle,entry['seed']).to(target)
    metrics=[]
    with torch.no_grad():
        for epoch in frozen['epochs']:
            checkpoint=directory/f'epoch{epoch:03d}_model.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['epoch']==epoch
            model.load_state_dict(state['model'])
            predictions=original.predict(model,bundle,'test',target)
            np.savez_compressed(out/f'test_epoch{epoch:03d}.npz',row_ids=bundle.test_row_ids,
                domains=bundle.test_x[:,-1].astype(int),labels=bundle.test_y,predictions=predictions)
            for row in original.metrics(bundle,predictions,'test'):
                metrics.append(dict(run_id=run_id,arm=entry['arm'],seed=entry['seed'],epoch=epoch,
                                    main_steps=20*epoch,split='test',**row))
    path=out/'metrics_test_trajectory.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    dump(out/'receipt.json',dict(status='COMPLETE',run_id=run_id,epochs_evaluated=50,optimizer_updates=0,
        freeze_sha256=sha(LOCK),source_sha256=sha(__file__),
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}))
    print(run_id,'COMPLETE 50 frozen checkpoints; optimizer updates=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--run-id');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.action=='freeze':
        print('frozen',len(freeze()['runs']),'runs',flush=True)
    else:
        assert args.run_id
        run_readout(args.run_id,args.device)
