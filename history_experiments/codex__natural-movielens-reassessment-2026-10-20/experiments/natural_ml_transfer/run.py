"""Original fit/step implementations with data/output and exposure injection."""
import argparse
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from .common import ART,source_hashes,configure,ExposureRNG,metadata,dump,sha

def check(cfg):
    gate=json.loads((ART/'transfer_review_gate.json').read_text())
    assert gate['verdict']=='PASS' and not gate['blocking_issues']
    assert cfg['study'] in gate['stages']
    actual=source_hashes(cfg['study'])
    assert actual==cfg['source_hashes']==gate['source_hashes'][cfg['study']]

def counts_snapshot(runner,name):
    folder=runner.state_dir if hasattr(runner,'state_dir') else runner.art
    np.savez_compressed(folder/('exposure_'+name+'.npz'),counts=runner.reassessment_counts)

def observe_rows(runner,rows,append):
    b=runner.data if hasattr(runner,'data') else runner.b
    np.add.at(runner.reassessment_counts,rows,1)
    counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3)
    append('reassessment_batch_exposure.csv',[dict(step=runner.steps,n=len(rows),
        d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
        row_ids_sha256=hashlib.sha256(b.train_ids[rows].astype(np.int64).tobytes()).hexdigest())])

def n19_runner(cfg,device,dm):
    old=importlib.import_module('n19_train')
    old.OUT=dm.OUT;old.ART=dm.ART;old.population=dm.population
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset');parser.add_argument('--suite',default='natural');parser.add_argument('--method')
    parser.add_argument('--seed',type=int);parser.add_argument('--device');parser.add_argument('--backbone',default=None)
    parser.add_argument('--noise-interaction',type=float,default=0.);parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--batch-size',type=int,default=40960);parser.add_argument('--tag',default='')
    a=parser.parse_args(cfg['flags']);a.device=device;a.resume=False;a.engineering=False
    assert a.epochs==100 and a.batch_size==40960 and a.lr==.001
    class Runner(old.Runner):
        def __init__(self):
            super().__init__(a)
            self.reassessment_counts=np.zeros(len(self.data.train_x),dtype=np.int64)
            if cfg['budget']=='main178125':self.train_rng=ExposureRNG(self.train_rng,self.data.main,178125)
            self.config.update(metadata(cfg,self.data));dump(self.out/'config.json',self.config)
        def step(self,rows):
            value=super().step(rows)
            observe_rows(self,rows,self.append)
            return value
        def append(self,file,rows):
            if file=='trajectory.csv':
                rows=[dict(r,examples=int(self.reassessment_counts.sum())) for r in rows]
            return super().append(file,rows)
        def save(self,epoch,name):
            super().save(epoch,name);counts_snapshot(self,name)
        def prune_rank(self):
            if not getattr(self.model,'rank_mode',False):return super().prune_rank()
            import torch
            self.save(10,'pre_prune_epoch010.pt')
            with torch.no_grad():before=torch.linalg.svdvals(self.model.slots().detach().double()).cpu().tolist()
            result=super().prune_rank()
            with torch.no_grad():after=torch.linalg.svdvals(self.model.slots().detach().double()).cpu().tolist()
            self.save(10,'post_prune_epoch010.pt')
            dump(self.out/'pruning_effective_spectra.json',dict(epoch=10,before=before,after=after,extra_training_updates=0))
            return result
    return Runner(),old

def cm19_runner(cfg,device,dm):
    old=importlib.import_module('cm19_train');old.OUT=dm.OUT;old.ART=dm.ART;old.population=dm.population
    c=dict(cfg['historical_config'])
    class Runner(old.Runner):
        def __init__(self):
            super().__init__(c,device,smoke=False)
            self.reassessment_counts=np.zeros(len(self.b.train_x),dtype=np.int64)
            if cfg['budget']=='main178125':self.rng=ExposureRNG(self.rng,self.main,178125)
            self.config.update(metadata(cfg,self.b));dump(self.out/'config.json',self.config)
            self.config.update(effective_ordinary_support_rows=len(self.main),
                ordinary_epoch_draws=178125 if cfg['budget']=='main178125' else len(self.b.main) if c.get('schedule')=='matched_updates' else len(self.main))
            dump(self.out/'config.json',self.config)
        def update(self,rows):
            value=super().update(rows)
            observe_rows(self,rows,lambda file,rs:old.append(self.out/file,rs))
            return value
        def save(self,epoch,name):
            super().save(epoch,name);counts_snapshot(self,name)
    return Runner(),old

def execute(cfg,device):
    check(cfg)
    if Path(cfg['run_dir']).exists():raise FileExistsError('Preserve existing attempt; never overwrite or automatically resume: '+cfg['run_dir'])
    dm=configure(cfg)
    runner,old=(n19_runner if cfg['study']=='N19' else cm19_runner)(cfg,device,dm)
    if cfg['study']=='N19':runner.save(0,'initial.pt')
    runner.fit()
    assert source_hashes(cfg['study'])==cfg['source_hashes']
    receipt_path=runner.out/'receipt.json';receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
    if cfg['study']=='N19':
        draws=178125 if cfg['budget']=='main178125' else len(runner.data.main)
        assert receipt['epochs']==100 and receipt['steps']==100*math.ceil(draws/40960)
        assert int(runner.reassessment_counts.sum())==100*draws
    else:
        total_steps=cfg['historical_config'].get('max_steps')
        if total_steps:assert receipt['steps']==total_steps
        else:assert receipt['epochs']==100
        assert int(runner.reassessment_counts.sum())==receipt['examples']
        if cfg['budget']=='main178125':assert receipt['steps']==500 and receipt['examples']==17812500
    b=runner.data if cfg['study']=='N19' else runner.b
    main_counts=np.bincount(b.train_x[:,-1].astype(int),weights=runner.reassessment_counts,minlength=3).astype(np.int64)
    receipt.update(global_run_id=cfg['run_id'],actual_main_examples=int(runner.reassessment_counts.sum()),
        actual_main_domain_exposures=main_counts.tolist(),independent_training_support=int((runner.reassessment_counts>0).sum()),
        dataset_config=cfg['population'],budget_mode=cfg['budget'],exploratory_replication=True)
    dump(receipt_path,receipt)
    dump(runner.out/'progress.json',dict(receipt,status='TRAIN_COMPLETE_TEST_DEFERRED'))
    states=runner.state_dir if cfg['study']=='N19' else runner.art
    dump(runner.out/'checkpoint_index.json',{p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in states.glob('*.pt')})
    dump(runner.out/'reassessment_receipt.json',dict(status='COMPLETE_TRAINING_TEST_DEFERRED',
        global_run_id=cfg['run_id'],receipt_sha256=sha(receipt_path),source_hashes=cfg['source_hashes'],
        checkpoint_index_sha256=sha(runner.out/'checkpoint_index.json'),test_evaluated=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();execute(json.loads(a.config.read_text()),a.device)
if __name__=='__main__':main()
