"""AC-2D adapter. The archived Stage2D loss, sampler and Adam step are inherited."""
from pathlib import Path
import hashlib
import json
import math
import random
import csv
import shutil
import tempfile
import time
import numpy as np
import torch
from .bootstrap import activate, dump, sha, source_hashes
activate()
import stage2d_20261016 as original
from stage_S1_models import initial_state_hash
from stage_S3_gradient_access import assert_expected_parameter_counts
from .data import load_population
from .telemetry import append, observe, aggregates

class Runner(original.Runner):
    def __init__(self, cfg, device='cuda:0'):
        self.cfg=cfg
        requested_dir=Path(cfg['run_dir'])
        resumable=(requested_dir/'last_resume.pt').exists()
        initialization_dir=requested_dir
        if resumable:
            initialization_dir=Path(tempfile.mkdtemp(prefix='resume_init_',dir=requested_dir.parent))/'new'
        b=load_population(cfg['population'])
        # Constructor adapters change only the data/provenance entry points.
        original.DATA=Path(b.preprocessing_metadata['raw_file'])
        original.DATA_SHA=b.preprocessing_metadata['raw_file_sha256']
        original.load_stage_s1_bundle=lambda *a,**k:b
        original.assert_expected_parameter_counts=lambda p,d:assert_expected_parameter_counts(p,'movielens')
        original.source_hashes=source_hashes
        original.STUDY='NATURAL_ML_AC_2D_20261020'
        original.ROOT=Path(__file__).resolve().parents[2]
        super().__init__(cfg['arm'],cfg['seed'],initialization_dir,device=device,code_commit='natural_reassessment_frozen_source_archive')
        self.draw_counts=np.zeros(len(b.train_x),dtype=np.int64)
        self.start_epoch=0
        self.config.update(stage='AC-2D',dataset='movielens',setting='natural',dataset_config=cfg['population'],
                           budget=cfg['budget'],main_examples_per_epoch=cfg['expected_main_examples_per_epoch'],
                           independent_review=True,reviewer_verdict='REQUIRED_EXTERNAL_PASS',
                           expected_main_steps=cfg['main_steps'],expected_arch_steps=cfg['architecture_steps'],
                           protocol_identity='natural transfer; not historical exact initialization replay')
        dump(self.dir/'config.json',self.config)
        payload=self.save_state(-1)
        torch.save(payload,self.dir/'initial_state.pt')
        if resumable:
            rebuilt=self.config
            self.dir=requested_dir
            self.config=json.loads((self.dir/'config.json').read_text())
            for key in ('source_hashes','init_sha','data','alpha','q','budget','dataset_config'):
                if self.config[key]!=rebuilt[key]:raise RuntimeError('resume identity differs: '+key)
            state=torch.load(self.dir/'last_resume.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(state['model']);self.main.load_state_dict(state['optimizer']);self.arch.load_state_dict(state['arch_optimizer'])
            modules=dict(self.model.named_modules())
            for name,tau in state['model_tau'].items():modules[name].tau=tau
            self.step_main=state['step_main'];self.step_arch=state['step_arch'];self.seen=state['main_seen'];self.draw_counts=state['draw_counts']
            self.best=state['selected_val_auc'];self.selected_epoch=state['selected_epoch'];self.val_history=state['val_history'];self.start_epoch=state['epoch']+1
            n=cfg['expected_main_examples_per_epoch']
            for epoch in range(self.start_epoch):
                for offset in range(0,n,40960):
                    rows=self.main_sampler.sample(min(40960,n-offset));self.batch_chains['main'].update(self.bundle.train_row_ids[rows].astype(np.int64).tobytes())
                rows=self.arch_sampler.sample(40960);self.batch_chains['arch'].update(self.bundle.train_row_ids[rows].astype(np.int64).tobytes())
            assert self.batch_chains['main'].hexdigest()==state['main_batch_chain']
            assert self.batch_chains['arch'].hexdigest()==state['arch_batch_chain']
            self.main_sampler.rng.bit_generator.state=state['main_sampler_rng'];self.arch_sampler.rng.bit_generator.state=state['arch_sampler_rng']
            rng=state['rng'];torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
            if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
            # Discard only observations from the incomplete next epoch.
            for path in self.dir.glob('*.csv'):
                with path.open(newline='') as handle:reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
                if fields and 'epoch' in fields:
                    rows=[r for r in rows if int(r['epoch'])<=state['epoch']]
                    with path.open('w',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            if self.selected_epoch==state['epoch']:torch.save(state,self.dir/'selected_state.pt')
            if state['epoch'] in original.COMMON:torch.save(state,self.dir/f"epoch{state['epoch']:02d}.pt")
            shutil.rmtree(initialization_dir.parent)

    def check_peers(self):
        # Row/schema equality is asserted in the independently frozen data manifest.
        assert self.config['data']['preprocessing_metadata']['optimizer_seed_changes_population'] is False

    def step(self,rows,kind,epoch,batch_index,telemetry):
        names=self.main_names if kind=='main' else self.arch_names
        before={n:self.named[n].detach().clone() for n in names}
        result=super().step(rows,kind,epoch,batch_index,True)
        opt=self.main if kind=='main' else self.arch
        append(self.dir/'optimizer_full_observations.csv',observe(self.named,names,self.group_of,opt,before,epoch,batch_index,kind))
        if kind=='main':np.add.at(self.draw_counts,rows,1)
        return result

    def save_state(self,epoch):
        payload=super().save_state(epoch)
        payload.update(draw_counts=self.draw_counts.copy(),population=self.cfg['population'],budget=self.cfg['budget'],
                       config=self.config,model_class='archived StageS1M3OEAdapter')
        torch.save(payload,self.dir/'last_resume.pt')
        return payload

    def metrics(self,records,split,epoch,checkpoint='trajectory',state=None):
        ms=self.step_main if state is None else state['step_main'];a=self.step_arch if state is None else state['step_arch']
        append(self.dir/'metrics_aggregates.csv',aggregates(records,self.bundle,split,epoch,checkpoint,ms,a))
        return super().metrics(records,split,epoch,checkpoint,state)

    def fit(self):
        # Original producer's complete per-stage loop, with its exact step() and
        # sampler implementations. Hooks add persistence and measured exposures.
        n=self.cfg['expected_main_examples_per_epoch']
        self._fit_epoch_loop(50,n,40960)
        actual={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('stdout.log','sha_index.json')}
        dump(self.dir/'sha_index.json',actual)

    def _fit_epoch_loop(self,epochs,n,batch_size):
        # Lifecycle is intentionally specific to Stage2D; it does not replace
        # the controlled S5/S6 or fixed-model V9 training algorithms.
        for epoch in range(self.start_epoch,epochs):
            before=self.draw_counts.copy();weighted_loss=0.;exposure=np.zeros(3,dtype=np.int64)
            for step,offset in enumerate(range(0,n,batch_size)):
                rows=self.main_sampler.sample(min(batch_size,n-offset))
                loss,count=self.step(rows,'main',epoch,step,True)
                exposure+=count;weighted_loss+=loss*len(rows)
            self.step(self.arch_sampler.sample(batch_size),'arch',epoch,0,True)
            records,ids,domains,labels,predictions=self.evaluate('val');score=original._macro(records,'auc')
            if not math.isfinite(score):raise FloatingPointError('nonfinite validation score')
            self.val_history[epoch]=score;improved=score>self.best
            if improved:self.best=score;self.selected_epoch=epoch
            append(self.dir/'metrics_epoch.csv',self.metrics(records,'validation',epoch))
            stats=dict(epoch=epoch,objective=weighted_loss/n,main_examples=n,validation_macro_auc=score,
                       selected_epoch_so_far=self.selected_epoch,step_main=self.step_main,step_arch=self.step_arch)
            for d,pool in enumerate(self.main_sampler.pools):
                increment=(self.draw_counts-before)[pool];counts=self.draw_counts[pool]
                values,freq=np.unique(increment,return_counts=True)
                stats.update({f'exposure_d{d}':int(exposure[d]),f'unique_epoch_d{d}':int(np.count_nonzero(increment)),
                              f'unique_cumulative_d{d}':int(np.count_nonzero(counts)),f'repeats_epoch_d{d}':int(increment.sum()-np.count_nonzero(increment)),
                              f'repeat_histogram_d{d}':';'.join(f'{i}:{j}' for i,j in zip(values,freq))})
            append(self.dir/'epoch_observations.csv',[stats]);payload=self.save_state(epoch)
            if improved:
                torch.save(payload,self.dir/'selected_state.pt')
                np.savez_compressed(self.dir/'validation_selected.npz',row_ids=ids,domains=domains,labels=labels,predictions=predictions)
            if epoch in original.COMMON or epoch==epochs-1:torch.save(payload,self.dir/f'epoch{epoch:02d}.pt')
            assert source_hashes()==self.sources,'source drift'
            dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,selected_epoch=self.selected_epoch,
                                              step_main=self.step_main,step_arch=self.step_arch,elapsed_seconds=time.time()-self.started))
            print(json.dumps(dict(epoch=epoch,val_macro_auc=score,selected_epoch=self.selected_epoch,elapsed=time.time()-self.started)),flush=True)
        assert self.step_main==self.cfg['main_steps'] and self.step_arch==50
        assert self.selected_epoch==max(self.val_history,key=lambda e:(self.val_history[e],-e))
        dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,
             val_history=self.val_history,test_access_started=False,rule='validation_macro_auc_strict_greater_earliest'))
        keys={'selected':self.dir/'selected_state.pt','final':self.dir/'epoch49.pt'}
        keys.update({f'common{e:02d}':self.dir/f'epoch{e:02d}.pt' for e in original.COMMON})
        (self.dir/'metrics_test.csv').unlink(missing_ok=True)
        aggregate_path=self.dir/'metrics_aggregates.csv'
        if aggregate_path.exists():
            with aggregate_path.open(newline='') as handle:reader=csv.DictReader(handle);fields=reader.fieldnames;rows=[r for r in reader if r['split']!='test']
            with aggregate_path.open('w',newline='') as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for key,path in keys.items():
            state=torch.load(path,map_location='cpu',weights_only=False);self.model.load_state_dict(state['model'],strict=True)
            modules=dict(self.model.named_modules())
            for name,tau in state['model_tau'].items():modules[name].tau=tau
            records,ids,domains,labels,pred=self.evaluate('test')
            append(self.dir/'metrics_test.csv',self.metrics(records,'test',state['epoch'],key,state=state))
            np.savez_compressed(self.dir/f'test_{key}.npz',row_ids=ids,domains=domains,labels=labels,predictions=pred)
        dump(self.dir/'progress.json',dict(status='COMPLETE',epochs_completed=epochs,step_main=self.step_main,step_arch=self.step_arch))
        manifest=dict(self.config,status='COMPLETE',epochs_completed=epochs,step_main=self.step_main,step_arch=self.step_arch,
                      selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,state_complete=True,
                      main_batch_chain=self.batch_chains['main'].hexdigest(),arch_batch_chain=self.batch_chains['arch'].hexdigest(),
                      wall_seconds=time.time()-self.started,artifact_hashes={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('run_manifest.json','stdout.log')})
        dump(self.dir/'run_manifest.json',manifest)
