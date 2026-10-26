"""Stage 2D: independently identified same-objective/different-q training."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16/stage2d_execution_20261016'
STUDY='STAGE2D_KR_IID_20261016'
DATA=Path('/home/xzf/data/kuairand/processed/kuai_rand.csv')
DATA_SHA='a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f'
ARMS=('SN','SB','EN','EB');SEEDS=(2022,2023,2024,2025)
BATCH=40960;EPOCHS=50;COMMON=(9,19,29,39,49)

from stage_S1_data import load_stage_s1_bundle
from stage_S1_models import build_model,initial_state_hash,parameter_manifest
from stage_S3_gradient_access import partition_parameters,assert_expected_parameter_counts
from stage_S1_run import _prediction_records,_macro,_raw_rng_state,setup_seed,TorchDataset
from stage_E1_instrumentation import preserve_rng_state


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def csv_append(path,rows):
    if not rows:return
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        if not exists:w.writeheader()
        w.writerows(rows)


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def source_hashes():
    files=[Path(__file__),ROOT/'experiments/stage_S1_data.py',ROOT/'experiments/stage_S1_models.py',
       ROOT/'experiments/stage_S1_run.py',ROOT/'experiments/stage_S1_metrics.py',
       ROOT/'experiments/stage_E1_instrumentation.py',ROOT/'experiments/stage_S3_gradient_access.py',
       ROOT/'m3oe/MDMTRec/models/multi_domain/model.py',ROOT/'m3oe/MDMTRec/basic/layers.py',
       ROOT/'m3oe/MDMTRec/basic/features.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


class Sampler:
    def __init__(self,bundle,q,seed,stream):
        self.pools=[np.flatnonzero(bundle.train_x[:,-1]==d) for d in range(3)]
        self.q=np.asarray(q);self.rng=np.random.default_rng(seed*1000+stream)
    def sample(self,n):
        u=self.rng.random((n,2));d=np.searchsorted(np.cumsum(self.q),u[:,0],side='right')
        rows=np.empty(n,dtype=np.int64)
        for k,pool in enumerate(self.pools):
            mask=d==k;rows[mask]=pool[np.floor(u[mask,1]*len(pool)).astype(np.int64)]
        return rows


class Runner:
    def __init__(self,arm,seed,directory,device='cuda:0',engineering=False,code_commit='UNCOMMITTED_ENGINEERING'):
        self.arm=arm;self.seed=seed;self.dir=Path(directory);self.dir.mkdir(parents=True,exist_ok=False)
        self.device=torch.device(device);self.engineering=engineering
        setup_seed(seed);torch.set_num_threads(2)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        assert sha(DATA)==DATA_SHA
        self.bundle=load_stage_s1_bundle('kuairand',str(DATA.parent),'natural',seed)
        p=np.array([self.bundle.train_counts[d] for d in range(3)],dtype=float);p/=p.sum()
        self.p=p;self.alpha=p if arm[0]=='S' else np.ones(3)/3
        self.q=p if arm[1]=='N' else np.ones(3)/3
        self.weight=torch.tensor(self.alpha/self.q,dtype=torch.float32,device=self.device)
        self.main_sampler=Sampler(self.bundle,self.q,seed,17);self.arch_sampler=Sampler(self.bundle,self.q,seed,29)
        self.model=build_model(self.bundle.features,3,2,'m3oe',seed=seed).to(self.device)
        assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or
            (isinstance(m,torch.nn.Dropout) and m.p>0) for m in self.model.modules())
        self.partition=partition_parameters(self.model);assert_expected_parameter_counts(self.partition,'kuairand')
        self.named=dict(self.model.named_parameters());self.group_of={n:g for g,ns in self.partition['groups'].items() for n in ns}
        self.main_names=[n for n in self.named if self.group_of[n]!='A_arch']
        self.arch_names=self.partition['groups']['A_arch']
        assert not set(self.main_names)&set(self.arch_names) and set(self.main_names+self.arch_names)==set(self.named)
        self.main=torch.optim.Adam([self.named[n] for n in self.main_names],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        self.arch=torch.optim.Adam([self.named[n] for n in self.arch_names],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        self.init=initial_state_hash(self.model);self.sources=source_hashes();self.started=time.time()
        self.seen=np.zeros(len(self.bundle.train_x),dtype=bool);self.step_main=0;self.step_arch=0
        self.best=-math.inf;self.selected_epoch=None;self.val_history={};self.batch_chains={'main':hashlib.sha256(),'arch':hashlib.sha256()}
        self.attrs=lambda:{n:m.tau for n,m in self.model.named_modules() if hasattr(m,'tau')}
        self.config=dict(study_id=STUDY,stage='2D',dataset='kuairand',setting='natural',model='M3OE',arm=arm,seed=seed,
            alpha=self.alpha.tolist(),q=self.q.tolist(),importance_weight=(self.alpha/self.q).tolist(),
            sampler='IID domain q then uniform within domain WITH replacement; common uniform coupling',
            batch_size=BATCH,epochs=EPOCHS,main_examples_per_epoch=len(self.bundle.train_x),arch_batch_size=BATCH,
            main_learning_rate=.01,arch_learning_rate=.001,betas=[.9,.999],eps=1e-8,weight_decay=1e-5,
            clipping=False,scheduler=None,early_stop=False,selection='validation six-cell macro AUC; strict >; earliest tie',
            code_commit=code_commit,source_hashes=self.sources,dataset_sha256=DATA_SHA,
            init_sha=self.init,data=self.bundle.manifest_fields(),reviewer_verdict='USER_DIRECTED_PRIMARY_SELF_REVIEW',
            independent_review=False,engineering_only=engineering,
            thresholds=dict(practical_auc=.003,strong_auc=.005,same_sign_seeds=3,
                cell_mean_auc_floor=-.003,individual_cell_auc_floor=-.005,cell_mean_ll_ceiling=.010))
        dump(self.dir/'config.json',self.config);dump(self.dir/'parameter_manifest.json',parameter_manifest(self.model))
        dump(self.dir/'partition.json',self.partition)
        self.check_peers()

    def check_peers(self):
        if self.engineering:return
        for arm in ARMS:
            file=self.dir.parent.parent/arm/self.dir.name/'config.json'
            if file.exists():
                other=json.loads(file.read_text());assert other['init_sha']==self.init
                assert other['data']==self.config['data']

    def loader(self,split):
        b=self.bundle
        return DataLoader(TorchDataset(getattr(b,split+'_x'),getattr(b,split+'_y')),
                          batch_size=BATCH,shuffle=False,num_workers=0,drop_last=False)

    def evaluate(self,split):
        with preserve_rng_state():
            return _prediction_records(self.bundle,self.model,self.loader(split),self.device,
                                       getattr(self.bundle,split+'_row_ids'))

    def metrics(self,records,split,epoch,checkpoint='trajectory',state=None):
        main_step=self.step_main if state is None else state['step_main']
        arch_step=self.step_arch if state is None else state['step_arch']
        return [dict(study_id=STUDY,seed=self.seed,arm=self.arm,split=split,epoch=epoch,checkpoint=checkpoint,
             domain=k%3,task=k//3,auc=float(r['auc']),logloss=float(r['logloss']),n=int(r['sample_count']),
             n_pos=int(r['positive_count']),step_main=main_step,step_arch=arch_step) for k,r in records.items()]

    def step(self,rows,kind,epoch,batch_index,telemetry):
        b=self.bundle;x=torch.tensor(b.train_x[rows],dtype=torch.float32,device=self.device).T
        y=torch.tensor(b.train_y[rows],dtype=torch.float32,device=self.device)
        d=x[-1].long();self.model.train();self.model.zero_grad(set_to_none=True)
        prediction=self.model(x);terms=F.binary_cross_entropy(prediction,y,reduction='none').mean(1)
        loss=(terms*self.weight[d]).mean()
        assert torch.isfinite(loss).item(),'nonfinite loss'
        loss.backward();opt=self.main if kind=='main' else self.arch
        names=self.main_names if kind=='main' else self.arch_names
        for n in names:
            g=self.named[n].grad
            if g is not None and not torch.isfinite(g).all().item():raise FloatingPointError('nonfinite gradient '+n)
        before={n:self.named[n].detach().clone() for n in names} if telemetry else None
        norms={g:dict(gradient=0.,m=0.,v=0.,delta=0.) for g in ['C','P','A_arch']}
        if telemetry:
            for n in names:
                g=self.named[n].grad
                if g is not None:norms[self.group_of[n]]['gradient']+=float(g.detach().double().square().sum().item())
        opt.step()
        if telemetry:
            for n in names:
                p=self.named[n];s=opt.state.get(p,{})
                if not torch.isfinite(p).all().item():raise FloatingPointError('nonfinite parameter '+n)
                g=norms[self.group_of[n]]
                g['delta']+=float((p.detach()-before[n]).double().square().sum().item())
                for key,field in [('m','exp_avg'),('v','exp_avg_sq')]:
                    if field in s:g[key]+=float(s[field].detach().double().square().sum().item())
            csv_append(self.dir/'optimizer_observations.csv',[dict(epoch=epoch,batch=batch_index,kind=kind,partition=g,
                gradient_norm=math.sqrt(v['gradient']),m_norm=math.sqrt(v['m']),v_norm=math.sqrt(v['v']),
                applied_delta_norm=math.sqrt(v['delta'])) for g,v in norms.items() if (kind=='arch')==(g=='A_arch')])
        ids=np.asarray(b.train_row_ids[rows],dtype=np.int64)
        batch_hash=hashlib.sha256(ids.tobytes()).hexdigest();self.batch_chains[kind].update(ids.tobytes())
        counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3)
        csv_append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=batch_index,kind=kind,
            n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),row_ids_sha256=batch_hash,
            objective=float(loss.detach().cpu()))])
        if kind=='main':self.step_main+=1;self.seen[rows]=True
        else:self.step_arch+=1
        return float(loss.detach().cpu()),counts

    def save_state(self,epoch):
        payload=dict(study_id=STUDY,epoch=epoch,arm=self.arm,seed=self.seed,model=cpu(self.model.state_dict()),
            model_tau=self.attrs(),optimizer=cpu(self.main.state_dict()),arch_optimizer=cpu(self.arch.state_dict()),
            optimizer_parameter_names=self.main_names,arch_optimizer_parameter_names=self.arch_names,
            step_main=self.step_main,step_arch=self.step_arch,rng=_raw_rng_state(),
            main_sampler_rng=self.main_sampler.rng.bit_generator.state,arch_sampler_rng=self.arch_sampler.rng.bit_generator.state,
            main_seen=self.seen,selected_epoch=self.selected_epoch,selected_val_auc=self.best,
            val_history=self.val_history,main_batch_chain=self.batch_chains['main'].hexdigest(),
            arch_batch_chain=self.batch_chains['arch'].hexdigest(),config_sha=sha(self.dir/'config.json'),
            batch_position='end of epoch; main next epoch offset0; arch next draw',state_complete=True)
        tmp=self.dir/'resume.tmp.pt';torch.save(payload,tmp);tmp.replace(self.dir/'last_resume.pt')
        return payload

    def fit(self,epochs=EPOCHS,epoch_examples=None,batch_size=BATCH):
        n=len(self.bundle.train_x) if epoch_examples is None else epoch_examples
        for epoch in range(epochs):
            exposure=np.zeros(3,dtype=np.int64);epoch_seen=np.zeros(len(self.seen),dtype=bool);weighted_loss=0.
            for step,offset in enumerate(range(0,n,batch_size)):
                rows=self.main_sampler.sample(min(batch_size,n-offset));epoch_seen[rows]=True
                loss,count=self.step(rows,'main',epoch,step,step in (0,math.ceil(n/batch_size)-1))
                exposure+=count;weighted_loss+=loss*len(rows)
            rows=self.arch_sampler.sample(batch_size);self.step(rows,'arch',epoch,0,True)
            records,ids,domains,labels,predictions=self.evaluate('val');score=_macro(records,'auc')
            assert math.isfinite(score)
            self.val_history[epoch]=score;improved=score>self.best
            if improved:self.best=score;self.selected_epoch=epoch
            csv_append(self.dir/'metrics_epoch.csv',self.metrics(records,'validation',epoch))
            stats=dict(epoch=epoch,objective=weighted_loss/n,main_examples=n,
                d0=int(exposure[0]),d1=int(exposure[1]),d2=int(exposure[2]),
                unique_main_rows_this_epoch=int(epoch_seen.sum()),unique_main_rows_cumulative=int(self.seen.sum()),
                **{f'unique_cumulative_d{d}':int(self.seen[pool].sum()) for d,pool in enumerate(self.main_sampler.pools)},
                validation_macro_auc=score,selected_epoch_so_far=self.selected_epoch,
                step_main=self.step_main,step_arch=self.step_arch,
                main_batch_chain=self.batch_chains['main'].hexdigest(),arch_batch_chain=self.batch_chains['arch'].hexdigest())
            csv_append(self.dir/'epoch_observations.csv',[stats]);payload=self.save_state(epoch)
            if improved:
                torch.save(payload,self.dir/'selected_state.pt')
                np.savez_compressed(self.dir/'validation_selected.npz',row_ids=ids,domains=domains,labels=labels,predictions=predictions)
            if epoch in COMMON or epoch==epochs-1:torch.save(payload,self.dir/f'epoch{epoch:02d}.pt')
            assert source_hashes()==self.sources,'source drift'
            dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,epochs=epochs,
                selected_epoch=self.selected_epoch,step_main=self.step_main,step_arch=self.step_arch,
                elapsed_seconds=time.time()-self.started,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if self.device.type=='cuda' else 0))
            print(json.dumps(dict(epoch=epoch,val_macro_auc=score,selected_epoch=self.selected_epoch,elapsed=time.time()-self.started)),flush=True)
        assert self.selected_epoch==max(self.val_history,key=lambda e:(self.val_history[e],-e))
        dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,
            val_history=self.val_history,test_access_started=False,rule='validation_macro_auc_strict_greater_earliest'))
        keys={'selected':self.dir/'selected_state.pt','final':self.dir/f'epoch{epochs-1:02d}.pt'}
        keys.update({f'common{e:02d}':self.dir/f'epoch{e:02d}.pt' for e in COMMON if e<epochs})
        for key,path in keys.items():
            state=torch.load(path,map_location='cpu',weights_only=False);self.model.load_state_dict(state['model'],strict=True)
            modules=dict(self.model.named_modules())
            for name,tau in state['model_tau'].items():modules[name].tau=tau
            records,ids,domains,labels,pred=self.evaluate('test')
            csv_append(self.dir/'metrics_test.csv',self.metrics(records,'test',state['epoch'],key,state=state))
            np.savez_compressed(self.dir/f'test_{key}.npz',row_ids=ids,domains=domains,labels=labels,predictions=pred)
        complete_progress=dict(status='COMPLETE',epochs_completed=epochs,step_main=self.step_main,step_arch=self.step_arch,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if self.device.type=='cuda' else 0,
            wall_seconds=time.time()-self.started)
        dump(self.dir/'progress.json',complete_progress)
        manifest=dict(self.config,status='COMPLETE',epochs_completed=epochs,step_main=self.step_main,step_arch=self.step_arch,
            selected_epoch=self.selected_epoch,selected_validation_macro_auc=self.best,wall_seconds=time.time()-self.started,
            main_batch_chain=self.batch_chains['main'].hexdigest(),arch_batch_chain=self.batch_chains['arch'].hexdigest(),
            state_complete=True,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if self.device.type=='cuda' else 0,
            artifact_hashes={p.name:sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ['stdout.log','run_manifest.json']})
        dump(self.dir/'run_manifest.json',manifest)


def check(output):
    runner=Runner('SN',2022,output,device='cpu',engineering=True)
    # Same-q streams must match exactly, independent of objective.
    for a,b in [('SN','EN'),('SB','EB')]:
        q=runner.p if a[1]=='N' else np.ones(3)/3
        one=Sampler(runner.bundle,q,2022,17);two=Sampler(runner.bundle,q,2022,17)
        assert np.array_equal(one.sample(40960),two.sample(40960))
    rows=np.concatenate([pool[:4] for pool in runner.main_sampler.pools])
    x=torch.tensor(runner.bundle.train_x[rows],dtype=torch.float32).T
    y=torch.tensor(runner.bundle.train_y[rows],dtype=torch.float32);domain=x[-1].long()
    pred=runner.model(x);terms=F.binary_cross_entropy(pred,y,reduction='none').mean(1)
    for alpha in [runner.p,np.ones(3)/3]:
        for q in [runner.p,np.ones(3)/3]:
            w=torch.tensor(alpha/q,dtype=torch.float32)
            actual=(terms*w[domain]).mean()
            expected=sum(terms[domain==d].sum()*float(alpha[d]/q[d])/len(rows) for d in range(3))
            assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
    # Limit only engineering view sizes; formal dimensions/schema remain exact.
    b=runner.bundle
    for split in ['val','test']:
        xx=getattr(b,split+'_x');yy=getattr(b,split+'_y');ids=getattr(b,split+'_row_ids');chosen=[]
        for d in range(3):
            candidates=np.flatnonzero(xx[:,-1]==d);selected=list(candidates[:24])
            for t in range(2):
                for label in [0,1]:selected.append(int(candidates[np.flatnonzero(yy[candidates,t]==label)[0]]))
            chosen.extend(selected)
        chosen=np.unique(chosen);setattr(b,split+'_x',xx[chosen]);setattr(b,split+'_y',yy[chosen]);setattr(b,split+'_row_ids',ids[chosen])
    runner.fit(epochs=1,epoch_examples=96,batch_size=32)
    state=torch.load(runner.dir/'last_resume.pt',map_location='cpu',weights_only=False)
    assert state['state_complete'] and state['step_main']==3 and state['step_arch']==1
    saved=copy.deepcopy(runner.model.state_dict());runner.model.load_state_dict(state['model'],strict=True)
    assert all(torch.equal(saved[k].cpu(),v) for k,v in state['model'].items())
    dump(OUT/'engineering_verification.json',dict(status='PASS_ENGINEERING_ONLY',worker_sha256=sha(__file__),
        complete_pipeline=True,loss_oracle=True,same_q_pairing=True,main_steps=3,arch_steps=1,formal_runs=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['check','profile','train'],required=True)
    parser.add_argument('--arm',choices=ARMS,default='SN');parser.add_argument('--seed',type=int,default=2022)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--code-commit',default='UNCOMMITTED_ENGINEERING')
    args=parser.parse_args();assert args.seed in SEEDS
    try:
        if args.mode=='check':check(args.run_dir)
        elif args.mode=='profile':
            r=Runner('EN',2022,args.run_dir,device=args.device,engineering=True)
            for step in range(3):r.step(r.main_sampler.sample(BATCH),'main',0,step,True)
            r.step(r.arch_sampler.sample(BATCH),'arch',0,0,True);torch.cuda.synchronize()
            peak=torch.cuda.max_memory_reserved()/1024**2
            dump(OUT/'cuda_profile.json',dict(status='PASS',worker_sha256=sha(__file__),batch_size=BATCH,
                peak_reserved_mib=peak,reservation_mib=peak*1.25+768,main_steps=3,arch_steps=1))
        else:
            r=Runner(args.arm,args.seed,args.run_dir,device=args.device,code_commit=args.code_commit)
            assert not r.engineering and len(r.bundle.train_x)==2006132;r.fit()
    except BaseException as e:
        if args.run_dir.exists():dump(args.run_dir/'failure.json',dict(exception=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
