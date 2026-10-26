"""Original N17 model/steps/state/capture, with prespecified population/budget injection."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from .n17_bootstrap import OUT,prepare,sources,sha,dump

def build_runner(config,device='cuda:0',resume=False):
    b,sp,c,root=prepare(config['population'])
    from n17_ml import Runner as HistoricalRunner
    from n17_quota import QuotaSampler
    from v3_train_20261016 import predict,metrics
    class NaturalRunner(HistoricalRunner):
        def __init__(self,a):
            super().__init__(a)
            self.exposure_budget=config.get('epoch_examples')
            self.quota=config.get('sampling')=='quota'
            if self.quota:
                rng=copy.deepcopy(self.sampler.rng.bit_generator.state)
                self.sampler=QuotaSampler(self.b,self.q,a.seed)
                self.sampler.rng.bit_generator.state=rng
            self.config.update(population=config['population'],population_seed=2022,
                phase=config['group'],epoch_examples=self.exposure_budget,
                epoch_definition='187500 IID-WR draws on full natural support' if self.exposure_budget else 'complete population exposure count with replacement',
                sampling='systematic_quota_WR' if self.quota else 'IID_WR',
                actual_weight='B_t*p_d/n_d' if self.quota else 'p_d/q_d',
                data_cache=str(OUT.parent/'data'/(config['population']+'.npz')),
                independent_review=False,review="primary under explicitly invoked start.md",exploratory_replication=True,
                actual_module_paths=__import__('experiments.natural_ml_extensions.bootstrap',fromlist=['actual_module_paths']).actual_module_paths('n17'),
                epochs=a.epochs,selection=f'macro six-cell selection AUC; earliest strict max epochs1..{a.epochs}',
                selection_row_ids_sha256=hashlib.sha256(self.b.val_row_ids[self.selection].astype(np.int64).tobytes()).hexdigest(),
                diagnostic_row_ids_sha256=hashlib.sha256(self.b.val_row_ids[self.diagnostic].astype(np.int64).tobytes()).hexdigest(),
                parameter_partitions={g:[name for name in self.named if c.base.group(name)==g] for g in ('C','P','R','E')},
                initialization_and_optimizer_seed=a.seed,formal_seed=a.seed in (2022,2023,2024,2025),
                pilot_seed=a.seed in (2030,2031),gradient_probe_scope='every epoch first two actual updates; ALL coordinates, native m/v, no altered update',
                extension_config=config)
            if not a.resume or config.get('resume_from'):dump(self.dir/'config.json',self.config)
            self._probe_position=None
            original_step=self.opt.step
            def observed_step(*args,**kwargs):
                position=self._probe_position
                if position is None or position[1]>1:return original_step(*args,**kwargs)
                before={n:p.detach().cpu().clone() for n,p in self.named.items()}
                gradient={n:(torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().clone() for n,p in self.named.items()}
                native_before=c.base.cpu(copy.deepcopy(self.opt.state_dict()))
                result=original_step(*args,**kwargs)
                folder=self.dir/'optimizer_probes';folder.mkdir(exist_ok=True)
                torch.save(dict(epoch=position[0],batch=position[1],step=self.steps+1,
                    parameter_order=list(self.named),gradient=gradient,native_optimizer_before=native_before,
                    native_optimizer_after=c.base.cpu(self.opt.state_dict()),
                    update={n:p.detach().cpu()-before[n] for n,p in self.named.items()},
                    derivative_semantics='original training graph including routing detach'),
                    folder/f'epoch{position[0]:03d}_batch{position[1]:02d}.pt')
                return result
            self.opt.step=observed_step
        def step(self,rows,epoch,index):
            self._probe_position=(epoch,index)
            if self.quota:
                counts=np.bincount(self.b.train_x[rows,-1].astype(int),minlength=3)
                assert np.all(counts>0)
                self.weight=torch.tensor(len(rows)*self.p/counts,device=self.device,dtype=torch.float32)
                c.base.csv_append(self.dir/'quota_weights.csv',[dict(epoch=epoch,batch=index,n=len(rows),
                    **{f'n{d}':int(counts[d]) for d in range(3)},
                    **{f'w{d}':float(len(rows)*self.p[d]/counts[d]) for d in range(3)})])
            return super().step(rows,epoch,index)
        def fit(self):
            # Original n17_train.fit loop and capture save semantics, with only n
            # changed for the explicitly registered exposure/update control.
            a=self.a;n=self.exposure_budget or len(self.b.train_x)
            if not a.resume:self.save(0,'epoch000.pt')
            for epoch in range(self.start_epoch+1,a.epochs+1):
                start=time.time();weighted=0.;sampler_state=copy.deepcopy(self.sampler.rng.bit_generator.state)
                for index,offset in enumerate(range(0,n,a.batch_size)):
                    rows=self.sampler.sample(min(a.batch_size,n-offset));weighted+=self.step(rows,epoch,index)
                assert all(torch.isfinite(p).all() for p in self.model.parameters()),'nonfinite parameter'
                pred=predict(self.model,self.b,'val',self.device,batch_size=a.batch_size)
                np.savez_compressed(self.dir/f'validation_epoch{epoch:03d}.npz',predictions=pred)
                cells=metrics(self.b,pred,'val',self.selection);score=float(np.mean([r['auc'] for r in cells]));assert math.isfinite(score)
                for row in cells:row.update(epoch=epoch,seed=a.seed,arm=a.arm)
                c.base.csv_append(self.dir/'metrics_selection.csv',cells);self.history[epoch]=score
                improved=score>self.best
                if improved:self.best=score;self.selected=epoch
                objective=self.objective();assert math.isfinite(objective),'nonfinite J'
                stats=dict(epoch=epoch,train_empirical_objective=objective,online_objective=weighted/n,
                    selection_macro_auc=score,steps=self.steps,examples=epoch*n,
                    epoch_seconds=time.time()-start,elapsed_seconds=time.time()-self.started)
                c.base.csv_append(self.dir/'trajectory.csv',[stats]);dump(self.dir/f'sampler_epoch{epoch:03d}.json',sampler_state)
                if improved:self.save(epoch,'selected_state.pt')
                if epoch in (10,40,100,200):self.save(epoch,f'epoch{epoch:03d}.pt')
                elif epoch in (20,50):torch.save(dict(epoch=epoch,model=c.base.cpu(self.model.state_dict())),self.dir/f'epoch{epoch:03d}_model.pt')
                self.save(epoch,'last_resume.pt')
                dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,selected_epoch=self.selected,steps=self.steps,
                    elapsed_seconds=time.time()-self.started,epoch_seconds=stats['epoch_seconds'],
                    peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if self.device.type=='cuda' else 0))
                print(json.dumps(stats),flush=True)
                assert self.source_hashes==c.sources(),'frozen source drift'
            dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected,history=self.history,budget=a.epochs,test_read=False))
            index=[]
            for epoch in range(1,a.epochs+1):
                path=self.dir/'all_models'/f'epoch{epoch:03d}_model.pt'
                assert path.is_file();index.append(dict(epoch=epoch,path=str(path),sha256=sha(path)))
            dump(self.dir/'all_model_index.json',index)
            state_index=[]
            for path in sorted(self.dir.glob('*.pt')):
                if path.name.endswith('_model.pt'):continue
                state_index.append(dict(path=str(path),sha256=sha(path),complete_native_state=True))
            dump(self.dir/'native_state_index.json',state_index)
            dump(self.dir/'receipt.json',dict(status='COMPLETE_TRAINING_TEST_DEFERRED',epochs=a.epochs,steps=self.steps,
                examples=n*a.epochs,selected_epoch=self.selected,best_selection_auc=self.best,
                wall_seconds=time.time()-self.started,config_sha256=sha(self.dir/'config.json'),
                source_hashes=self.source_hashes,all_epoch_models_retained=True,
                model_capture_source_sha256=self.capture_sha,ml_source_sha256=self.ml_sha,
                all_model_index_sha256=sha(self.dir/'all_model_index.json'),test_read=False))
    a=SimpleNamespace(seed=config['seed'],arm=config['arm'],optimizer='adam',lr=config['lr'],
        epochs=config['epochs'],batch_size=40960,device=device,run_dir=Path(config['run_dir']),
        phase=config['group'],subset_seed=2022,resume=resume,smoke=False)
    return NaturalRunner(a)

def execute(config,device='cuda:0',resume=False):
    gate=json.loads((OUT/'review_gate.json').read_text())
    assert gate['verdict']=='PASS' and not gate['blocking_issues']
    assert gate['source_hashes']==sources()
    if config.get('resume_from') and not resume:
        from .n17_manifest import stage_continuation
        stage_continuation(config);resume=True
    build_runner(config,device,resume).fit()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true');a=p.parse_args()
    execute(json.loads(a.config.read_text()),a.device,a.resume)
if __name__=='__main__':main()
