"""Natural AC-S5/S6: original controlled_step and original epoch/arch methods.

Only lifecycle/provenance/persistence and the prespecified main exposure cap
are adapted. S5 and S6 retain separate archived producer method dispatch.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .bootstrap import activate, dump, sha, source_hashes
activate()
import stage_S1_run as s1
import stage_S3_run as s3
from stage_S5ML_run import S5MLRunner, build_argparser
from stage_S6ML_run import S6MLRunner
from stage_S4_gradient_controls import validate_adam_delta_against_clone
from stage_E1_instrumentation import preserve_rng_state
from stage2d_20261016 import cpu
from .data import load_population
from .telemetry import append, aggregates, observe

HISTORY={'F':'FULL','T':'TARGET','HT':'TARGET','DH':'FULL'}
CURRENT={'F':'FULL','T':'TARGET','HT':'FULL','DH':'TARGET'}
COMMON=(9,19,29,39,49)

class RecordedDataset(Dataset):
    def __init__(self,original):self.original=original;self.recorder=None
    def __len__(self):return len(self.original)
    def __getitem__(self,index):
        if self.recorder is not None:self.recorder.append(int(index))
        return self.original[index]

class RecordedIterator:
    def __init__(self,loader,limit=None):
        self.loader=loader;self.limit=limit;self.offset=0;self.consumed=0
        self.creation_rng=torch.get_rng_state().clone();self.first_rng=None
        self.iterator=iter(loader.base)
    def __iter__(self):return self
    def __next__(self):
        if self.limit is not None and self.offset>=self.limit:raise StopIteration
        if self.first_rng is None:self.first_rng=torch.get_rng_state().clone()
        rows=[];self.loader.dataset.recorder=rows
        try:x,y=next(self.iterator)
        finally:self.loader.dataset.recorder=None
        if self.limit is not None:
            n=min(len(x),self.limit-self.offset);x=x[:n];y=y[:n];rows=rows[:n]
        self.offset+=len(x);self.consumed+=1;self.loader.last_indices=np.asarray(rows,dtype=np.int64)
        return x,y
    def state(self):
        return dict(creation_rng=self.creation_rng,first_rng=self.first_rng,consumed=self.consumed,limit=self.limit,offset=self.offset)

class RecordedLoader:
    def __init__(self,loader):
        self.dataset=RecordedDataset(loader.dataset);self.limit=None;self.last_indices=None
        self.base=DataLoader(self.dataset,batch_size=loader.batch_size,shuffle=True,num_workers=0,drop_last=False)
    def __iter__(self):return RecordedIterator(self,self.limit)
    def __len__(self):return len(self.base) if self.limit is None else math.ceil(self.limit/self.base.batch_size)
    def restore_iterator(self,state):
        if state is None:return None
        with preserve_rng_state():
            torch.set_rng_state(state['creation_rng']);iterator=RecordedIterator(self,state['limit'])
            if state['first_rng'] is not None:torch.set_rng_state(state['first_rng'])
            for _ in range(state['consumed']):next(iterator)
        assert iterator.offset==state['offset']
        return iterator

def args_for(cfg,device):
    args=SimpleNamespace(**{a.dest:a.default for a in build_argparser()._actions if a.dest!='help'})
    values=dict(dataset='movielens',setting='natural',backbone='m3oe',run_type='joint',seed=cfg['seed'],arm=cfg['arm'],
        run_dir=cfg['run_dir'],device=device,epoch=50,batch_size=40960,learning_rate=.01,learning_rate_darts=.001,
        weight_decay=1e-5,expert_num=4,fcn_dims=[512,256,256,64],shadow_patience=10,num_workers=0,
        objective_weighting='equal' if cfg['stage']=='AC-S5' else 'sample',
        shared_access='TARGET_ONLY_SHARED' if cfg['arm']=='T' else 'FULL',fresh_role='natural_reassessment',
        source_formal_run_key='',formal_run_key=cfg['run_id'],co_run_pair_id='natural_seed'+str(cfg['seed']),
        planned_peer_run_key='manifest_paired_arm',worker_id='independently_reviewed_luna_runner',attempt_id='natural_fresh',
        smoke_only=False,smoke_representative=False,allow_protocol_deviation=0,allow_nonformal_seed=0,dataset_path='')
    vars(args).update(values);return args

class ControlledMixin:
    def __init__(self,cfg,device='cuda:0'):
        self.cfg=cfg;self.dir=Path(cfg['run_dir']);self.started=time.time();self.arm=cfg['arm']
        requested_dir=self.dir;resumable=(self.dir/'last_resume.pt').exists()
        if resumable:self.dir=Path(tempfile.mkdtemp(prefix='controlled_resume_init_',dir=self.dir.parent))/'new'
        self.history_source=HISTORY[self.arm];self.current_source=CURRENT[self.arm]
        self.sources=source_hashes(('controlled.py',))
        bundle=load_population(cfg['population']);s1.load_stage_s1_bundle=lambda *a,**k:bundle
        prior_target_for=s3.target_for
        s3.target_for=lambda dataset,setting:dict(local_domain=2,physical_domain='ages_35_45_50_56',size_role='natural',tasks={0:'rating>3',1:'rating>4'}) if setting=='natural' else prior_target_for(dataset,setting)
        args=args_for(dict(cfg,run_dir=str(self.dir)),device)
        # S3 -> S2 -> S1 setup preserves objective dispatch while bypassing the
        # historical hardcoded dataset/hash guard. In particular S6 never calls
        # S4.__init__, which would clobber Sample with Equal.
        validate_adam_delta_against_clone(device)
        s3.S3Runner.__init__(self,args)
        self.probe_rows=[];self.zero_target_main_steps=0;self.main_steps_seen=0
        self._moment_dump_paths=[];self._none_count_full=0;self._none_count_target=0
        self.train_loader=RecordedLoader(self.train_loader)
        self.draw_counts=np.zeros(len(bundle.train_x),dtype=np.int64)
        self.start_epoch=0;self.best=-math.inf;self.bad_epochs=0;self.val_history={};self.architecture_iterator=None
        self.named=self.parameter_by_name
        self.group_of={n:g for g,names in self.partition['groups'].items() for n in names}
        self.main_names=[n for n in self.named if self.group_of[n]!='A_arch'];self.arch_names=self.partition['groups']['A_arch']
        self.p=np.asarray([bundle.train_counts[d] for d in range(3)],dtype=float);self.p/=self.p.sum()
        self.alpha=np.ones(3)/3 if cfg['stage']=='AC-S5' else self.p.copy()
        self.config=dict(cfg,dataset='movielens',setting='natural',dataset_config=cfg['population'],model='M3OE',
            objective_weighting=args.objective_weighting,alpha=self.alpha.tolist(),q=self.p.tolist(),importance_weight=(self.alpha/self.p).tolist(),
            data=bundle.manifest_fields(),source_hashes=self.sources,init_sha=self.initial_parameter_state_hash,
            main_learning_rate=.01,arch_learning_rate=.001,betas=[.9,.999],eps=1e-8,weight_decay=1e-5,
            learned_fusion=True,darts=True,history_source=self.history_source,current_source=self.current_source,
            coefficient_table=self.objective_table,selected_rule='validation six cell macro AUC earliest strict greater',
            sampling='ordinary shuffled concatenated loader; without replacement per epoch',
            full_support=True,main_exposure_per_epoch=cfg['expected_main_examples_per_epoch'],independent_review=True,
            protocol_identity='natural transfer; preserved historical control formula; not historical exact initialization replay')
        dump(self.dir/'config.json',self.config);dump(self.dir/'partition.json',self.partition)
        torch.save(self._save(-1),self.dir/'initial_state.pt')
        if resumable:
            temporary=self.dir.parent;self.dir=requested_dir;self.run_dir=str(requested_dir);self.args.run_dir=self.run_dir
            rebuilt=self.config;self.config=json.loads((self.dir/'config.json').read_text())
            for key in ('source_hashes','init_sha','data','alpha','q','budget','dataset_config','objective_weighting'):
                if self.config[key]!=rebuilt[key]:raise RuntimeError('controlled resume mismatch: '+key)
            state=torch.load(self.dir/'last_resume.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer']);self.arch_optimizer.load_state_dict(state['arch_optimizer'])
            modules=dict(self.model.named_modules())
            for name,tau in state['model_tau'].items():modules[name].tau=tau
            self.optimizer_steps=state['step_main'];self.architecture_optimizer_steps=state['step_arch'];self.draw_counts=state['draw_counts']
            self.start_epoch=state['epoch']+1;self.best=state['selected_val_auc'];self.selected_epoch=state['selected_epoch'];self.val_history=state['val_history'];self.bad_epochs=state['bad_epochs']
            self._none_count_full=state['none_count_full'];self._none_count_target=state['none_count_target'];self.zero_target_main_steps=state['zero_target_main_steps'];self.main_steps_seen=state['main_steps_seen']
            self.architecture_iterator=self.train_loader.restore_iterator(state['arch_iterator'])
            rng=state['rng'];torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
            if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
            for path in self.dir.glob('*.csv'):
                with path.open(newline='') as handle:reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
                if fields and 'epoch' in fields:
                    rows=[r for r in rows if int(r['epoch'])<=state['epoch']]
                    with path.open('w',newline='') as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            if self.selected_epoch==state['epoch']:torch.save(state,self.dir/'selected_state.pt')
            if state['epoch'] in COMMON:torch.save(state,self.dir/f"epoch{state['epoch']:02d}.pt")
            shutil.rmtree(temporary)

    def _validate_args(self):
        a=self.args
        if a.dataset!='movielens' or a.setting!='natural' or a.backbone!='m3oe' or a.run_type!='joint':raise ValueError('natural MovieLens joint M3OE required')
        if a.seed not in (2022,2023,2024,2025) or a.arm not in HISTORY:raise ValueError('outside frozen matrix')
        for key,value in dict(epoch=50,batch_size=40960,learning_rate=.01,learning_rate_darts=.001,weight_decay=1e-5,expert_num=4,fcn_dims=[512,256,256,64]).items():
            if getattr(a,key)!=value:raise ValueError('protocol mismatch: '+key)

    def _check_source_drift(self):
        if source_hashes(('controlled.py',))!=self.sources:raise RuntimeError('controlled producer source drift')

    def _controlled_step(self,losses,fingerprint,x_batch,epoch,step,kind):
        names=self.arch_names if kind=='architecture' else self.main_names
        optimizer=self.arch_optimizer if kind=='architecture' else self.optimizer
        before={n:self.named[n].detach().clone() for n in names}
        old_probes=len(self.probe_rows)
        # Dispatches to the separately archived S5ML or S6ML implementation.
        super()._controlled_step(losses,fingerprint,x_batch,epoch,step,kind)
        append(self.dir/'optimizer_full_observations.csv',observe(self.named,names,self.group_of,optimizer,before,epoch,step,kind))
        append(self.dir/'gradient_step.csv',[self.gradient_rows[-1]])
        if len(self.probe_rows)>old_probes:append(self.dir/'probe.csv',[self.probe_rows[-1]])
        rows=self.train_loader.last_indices
        if rows is None or len(rows)!=len(x_batch):raise AssertionError('recorded batch row identities lost')
        if kind=='main':np.add.at(self.draw_counts,rows,1)
        counts=np.bincount(self.bundle.train_x[rows,-1].astype(int),minlength=3)
        append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=step,kind=kind,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
            row_ids_sha256=hashlib.sha256(self.bundle.train_row_ids[rows].astype(np.int64).tobytes()).hexdigest(),objective=float(losses['full'].detach()))])

    def _train_epoch(self,epoch):
        self.train_loader.limit=187500 if self.cfg['budget']=='matched250' else None
        try:return super()._train_epoch(epoch)
        finally:self.train_loader.limit=None

    def state(self,epoch):
        attrs={n:m.tau for n,m in self.model.named_modules() if hasattr(m,'tau')}
        return dict(epoch=epoch,model=cpu(self.model.state_dict()),model_tau=attrs,optimizer=cpu(self.optimizer.state_dict()),
            arch_optimizer=cpu(self.arch_optimizer.state_dict()),optimizer_parameter_names=self.main_names,arch_optimizer_parameter_names=self.arch_names,
            step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps,rng=s1._raw_rng_state(),config=self.config,
            draw_counts=self.draw_counts,selected_epoch=self.selected_epoch,selected_val_auc=self.best,val_history=self.val_history,
            bad_epochs=self.bad_epochs,arch_iterator=self.architecture_iterator.state() if self.architecture_iterator is not None else None,
            none_count_full=self._none_count_full,none_count_target=self._none_count_target,zero_target_main_steps=self.zero_target_main_steps,
            main_steps_seen=self.main_steps_seen,state_complete=True,model_class='archived StageS1M3OEAdapter',config_sha=sha(self.dir/'config.json'))

    def _save(self,epoch):
        payload=self.state(epoch);tmp=self.dir/'resume.tmp.pt';torch.save(payload,tmp);tmp.replace(self.dir/'last_resume.pt')
        return payload

    def _metrics(self,records,split,epoch,label,state):
        rows=[dict(seed=self.seed,arm=self.arm,stage=self.cfg['stage'],split=split,epoch=epoch,checkpoint=label,
                  domain=k%3,task=k//3,auc=float(v['auc']),logloss=float(v['logloss']),n=int(v['sample_count']),n_pos=int(v['positive_count']),
                  step_main=state['step_main'],step_arch=state['step_arch']) for k,v in records.items()]
        append(self.dir/('metrics_epoch.csv' if split=='validation' else 'metrics_test.csv'),rows)
        append(self.dir/'metrics_aggregates.csv',aggregates(records,self.bundle,split,epoch,label,state['step_main'],state['step_arch']))

    def fit(self):
        # Exact S1 outer order: original controlled train epoch, original arch
        # update, validation, earliest strict selection, then deferred tests.
        for epoch in range(self.start_epoch,50):
            before=self.draw_counts.copy();objective,train_records=self._train_epoch(epoch)
            self.architecture_iterator=self._architecture_update(self.architecture_iterator,epoch)
            with preserve_rng_state():records=s1._cell_records(self.bundle,self.model,self.val_loader,self.device)
            score=s1._macro(records,'auc')
            if not math.isfinite(score):raise FloatingPointError('nonfinite validation')
            self.val_history[epoch]=score;improved=score>self.best
            if improved:self.best=score;self.selected_epoch=epoch;self.bad_epochs=0
            else:self.bad_epochs+=1
            stats=dict(epoch=epoch,objective=objective,step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps,
                       selected_epoch_so_far=self.selected_epoch,validation_macro_auc=score)
            for d in range(3):
                pool=np.flatnonzero(self.bundle.train_x[:,-1]==d);inc=(self.draw_counts-before)[pool];c=self.draw_counts[pool]
                values,freq=np.unique(inc,return_counts=True)
                stats.update({f'exposure_d{d}':int(inc.sum()),f'unique_epoch_d{d}':int(np.count_nonzero(inc)),
                              f'unique_cumulative_d{d}':int(np.count_nonzero(c)),f'repeat_epoch_d{d}':int(inc.sum()-np.count_nonzero(inc)),
                              f'repeat_histogram_d{d}':';'.join(f'{i}:{j}' for i,j in zip(values,freq))})
            append(self.dir/'epoch_observations.csv',[stats])
            self._metrics(records,'validation',epoch,'trajectory',dict(step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps))
            payload=self._save(epoch)
            if improved:torch.save(payload,self.dir/'selected_state.pt')
            if epoch in COMMON:torch.save(payload,self.dir/f'epoch{epoch:02d}.pt')
            self._check_source_drift();dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps))
            print(json.dumps(dict(epoch=epoch,val_macro_auc=score,selected_epoch=self.selected_epoch,elapsed=time.time()-self.started)),flush=True)
        assert self.optimizer_steps==self.cfg['main_steps'] and self.architecture_optimizer_steps==50
        assert self.selected_epoch==max(self.val_history,key=lambda e:(self.val_history[e],-e))
        dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,val_history=self.val_history,test_access_started=False))
        keys={'selected':self.dir/'selected_state.pt','final':self.dir/'epoch49.pt',**{f'common{e:02d}':self.dir/f'epoch{e:02d}.pt' for e in COMMON}}
        (self.dir/'metrics_test.csv').unlink(missing_ok=True)
        path=self.dir/'metrics_aggregates.csv'
        if path.exists():
            with path.open(newline='') as handle:reader=csv.DictReader(handle);fields=reader.fieldnames;rows=[r for r in reader if r['split']!='test']
            with path.open('w',newline='') as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for label,path in keys.items():
            state=torch.load(path,map_location='cpu',weights_only=False);self.model.load_state_dict(state['model']);modules=dict(self.model.named_modules())
            for name,tau in state['model_tau'].items():modules[name].tau=tau
            with preserve_rng_state():records,ids,domains,labels,predictions=s1._prediction_records(self.bundle,self.model,self.test_loader,self.device,self.bundle.test_row_ids)
            self._metrics(records,'test',state['epoch'],label,state)
            np.savez_compressed(self.dir/f'test_{label}.npz',row_ids=ids,domains=domains,labels=labels,predictions=predictions)
        dump(self.dir/'progress.json',dict(status='COMPLETE',epochs_completed=50,step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps))
        dump(self.dir/'run_manifest.json',dict(self.config,status='COMPLETE',epochs_completed=50,step_main=self.optimizer_steps,step_arch=self.architecture_optimizer_steps,
            selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,state_complete=True,wall_seconds=time.time()-self.started,
            artifact_hashes={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('stdout.log','run_manifest.json','sha_index.json')}))
        dump(self.dir/'sha_index.json',{p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('stdout.log','sha_index.json')})

class S5Runner(ControlledMixin,S5MLRunner):pass
class S6Runner(ControlledMixin,S6MLRunner):pass

def Runner(cfg,device='cuda:0'):
    return (S5Runner if cfg['stage']=='AC-S5' else S6Runner)(cfg,device)
