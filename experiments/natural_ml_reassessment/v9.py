"""Natural AC-V9 adapter to the actual archived v3 fixed-model SGD producer."""
import csv
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from .bootstrap import activate, ROOT, dump, sha, source_hashes, HERE
# Load the data bundle first, then put v3's own module tree first. Core and v3
# S1 model/metrics files are byte-identical in the archive provenance.
from .data import load_population
activate('v3')
import v3_train_20261016 as original
from stage_S3_gradient_access import partition_parameters
from .telemetry import append, observe, aggregates

class Runner(original.Runner):
    def __init__(self,cfg,device='cuda:0'):
        self.cfg=cfg;self.frozen_sources=source_hashes(('v9.py',))
        b=load_population(cfg['population'])
        original.audit.bundle=lambda *a,**k:b
        protocol=ROOT/'v9_protocol';protocol.mkdir(parents=True,exist_ok=True)
        original.OUT=protocol;original.audit.OUT=protocol
        # Preserve the v3 SHA-ordered, domain x joint-label stratified half-val
        # selection rule. Full validation metrics are additionally emitted.
        original.audit.frozen_split(b,'movielens',cfg['seed'])
        self.full_partition=partition_parameters(original.base.fixed_model(b,cfg['seed']))
        original_sources=original.base.sources
        def frozen_sources():
            values=original_sources()
            # Original fit checks ROOT/name; add this immutable adapter identity
            # with its correct relative path into the actual source inventory.
            for name in ('v9.py',):
                path=HERE/name;values[os.path.relpath(path,original.ROOT)]=sha(path)
            return values
        original.base.sources=frozen_sources
        directory=Path(cfg['run_dir']);resume=(directory/'last_resume.pt').exists()
        args=SimpleNamespace(dataset='movielens',seed=cfg['seed'],arm=cfg['arm'],optimizer='sgd',lr=.3,batch_size=40960,
            epochs=50,replacement='WR',order='ORIGINAL',duplication=1,corruption=0.,device=device,run_dir=directory,smoke=False,resume=resume)
        super().__init__(args)
        self.group_of={n:g for g,names in self.full_partition['groups'].items() for n in names}
        self.config.update(stage='AC-V9',dataset_config=cfg['population'],setting='natural',budget=cfg['budget'],
            natural_source_hashes=self.frozen_sources,independent_review=True,reviewer_verdict='REQUIRED_EXTERNAL_PASS',
            protocol_identity='natural transfer of original fixed-fusion v3 SGD .3 momentum0; no DARTS',
            validation_selection_rule='original v3 fixed SHA256 row_id order within domain x joint label; half validation; strict earliest maximum',
            optimizer_parameter_partition=self.full_partition)
        if not resume:
            dump(self.dir/'config.json',self.config)
            self.save(0,'last_resume.pt')

    def step(self,rows,epoch,index):
        before={n:p.detach().clone() for n,p in self.named.items()}
        result=super().step(rows,epoch,index)
        append(self.dir/'optimizer_full_observations.csv',observe(self.named,list(self.named),self.group_of,self.opt,before,epoch,index,'main'))
        return result

    def fit(self):
        super().fit()
        if source_hashes(('v9.py',))!=self.frozen_sources:raise RuntimeError('V9 study producer source drift')
        assert self.steps==1000 and self.a.epochs==50
        # The original v3 fit defers all test access. Its .3 LR, selection rule,
        # and fixed E50 rule are already prespecified here; persist that lock
        # before loading any test data into predictions.
        dump(self.dir/'selection_frozen_natural.json',dict(selected_epoch=self.selected,selected_validation_macro_auc=self.best,
             history=self.history,lr=.3,momentum=0.,lr_or_rule_selection_using_test=False,test_access_started=False))
        (self.dir/'metrics_epoch.csv').unlink(missing_ok=True)
        (self.dir/'metrics_test.csv').unlink(missing_ok=True)
        (self.dir/'metrics_aggregates.csv').unlink(missing_ok=True)
        # Reuse the original full-validation predictions; no model updates.
        for epoch in range(1,51):
            predictions=np.load(self.dir/f'validation_epoch{epoch:03d}.npz')['predictions']
            rows=original.metrics(self.b,predictions,'val');records={r['task']*3+r['domain']:r for r in rows}
            for r in rows:r.update(epoch=epoch-1,stage='AC-V9',seed=self.a.seed,arm=self.a.arm,split='validation',checkpoint='trajectory',step_main=20*epoch,step_arch=0)
            append(self.dir/'metrics_epoch.csv',rows)
            append(self.dir/'metrics_aggregates.csv',aggregates(records,self.b,'validation',epoch-1,'trajectory',20*epoch,0))
        keys={'selected':self.dir/'selected_state.pt','final':self.dir/'epoch050.pt',
              **{f'common{e-1:02d}':self.dir/f'epoch{e:03d}.pt' for e in (10,20,30,40,50)}}
        for label,path in keys.items():
            state=torch.load(path,map_location='cpu',weights_only=False);self.model.load_state_dict(state['model'])
            predictions=original.predict(self.model,self.b,'test',self.device)
            rows=original.metrics(self.b,predictions,'test');records={r['task']*3+r['domain']:r for r in rows}
            for r in rows:r.update(epoch=state['epoch']-1,stage='AC-V9',seed=self.a.seed,arm=self.a.arm,split='test',checkpoint=label,step_main=state['steps'],step_arch=0)
            append(self.dir/'metrics_test.csv',rows)
            append(self.dir/'metrics_aggregates.csv',aggregates(records,self.b,'test',state['epoch']-1,label,state['steps'],0))
            np.savez_compressed(self.dir/f'test_{label}.npz',row_ids=self.b.test_row_ids,domains=self.b.test_x[:,-1].astype(int),labels=self.b.test_y,predictions=predictions)
        dump(self.dir/'progress.json',dict(status='COMPLETE',epochs_completed=50,step_main=self.steps,step_arch=0))
        # Refresh the native deferred-training receipt after this adapter's
        # prespecified deferred test evaluation, so its hashes describe the
        # final stable artifacts rather than the old TRAINING progress file.
        receipt=json.loads((self.dir/'receipt.json').read_text())
        receipt.update(status='COMPLETE',test_evaluated_after_prespecified_rule_freeze=True,
            artifact_hashes={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('receipt.json','run_manifest.json','sha_index.json','stdout.log')})
        dump(self.dir/'receipt.json',receipt)
        dump(self.dir/'run_manifest.json',dict(self.config,status='COMPLETE',epochs_completed=50,step_main=self.steps,step_arch=0,
            selected_epoch=self.selected-1,selected_validation_macro_auc=self.best,state_complete=True,wall_seconds=time.time()-self.started,
            artifact_hashes={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('run_manifest.json','sha_index.json','stdout.log')}))
        dump(self.dir/'sha_index.json',{p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('sha_index.json','stdout.log')})
