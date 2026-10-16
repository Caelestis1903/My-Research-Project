"""Fixed-fusion, single-optimizer v2 experiments. Raw readouts only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import stage2d_20261016 as old
from stage_S1_data import load_stage_s1_bundle
from stage_S1_models import build_model,initial_state_hash
from stage_S1_run import setup_seed,TorchDataset,_prediction_records,_macro,_raw_rng_state
from stage_E1_instrumentation import preserve_rng_state

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16_v2/execution_20261016'
COMMON=(1,5,10,20,30,40,50)
SEEDS=(2022,2023,2024,2025)
BATCH=40960
sha=old.sha;dump=old.dump;csv_append=old.csv_append;cpu=old.cpu


class ConstantFusion(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.register_buffer('value',torch.tensor([value],dtype=torch.float32))
    def forward(self):return self.value


def fixed_model(bundle,seed):
    model=build_model(bundle.features,3,2,'m3oe',seed=seed)
    for name in ('_weight_exp_d','_weight_exp_t','_weight_bal_d','_weight_bal_t'):
        value=float(torch.sigmoid(getattr(model,name).deep_weights.detach()).item())
        setattr(model,name,ConstantFusion(value))
    assert not any(n.startswith('_weight_') for n,p in model.named_parameters())
    return model


def group(name):
    if name.startswith('gate.'):return 'R'
    if name.startswith(('slot_weight.','slot_bias.','tower.')):return 'P'
    if name.startswith(('domain_expert.','task_expert.')):return 'E'
    return 'C'


def sources():
    return {**old.source_hashes(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}


class Sampler(old.Sampler):
    def __init__(self,bundle,q,seed,variant='iid'):
        super().__init__(bundle,q,seed,17);self.variant=variant
        self.orders=[np.arange(len(p)) for p in self.pools];self.positions=[0]*3
        self.cycle_rng=np.random.default_rng(seed*1000+71)
        for order in self.orders:self.cycle_rng.shuffle(order)
    def sample(self,n):
        if self.variant=='iid':return super().sample(n)
        if self.variant!='cyclic':raise ValueError(self.variant)
        u=self.rng.random((n,2));domains=np.searchsorted(np.cumsum(self.q),u[:,0],side='right')
        rows=np.empty(n,dtype=np.int64)
        for d,pool in enumerate(self.pools):
            where=np.flatnonzero(domains==d);offset=0
            while offset<len(where):
                pos=self.positions[d];take=min(len(where)-offset,len(pool)-pos)
                rows[where[offset:offset+take]]=pool[self.orders[d][pos:pos+take]]
                offset+=take;self.positions[d]+=take
                if self.positions[d]==len(pool):
                    self.positions[d]=0;self.cycle_rng.shuffle(self.orders[d])
        return rows


class Runner:
    def __init__(self,arm,seed,directory,optimizer='adam',lr=.01,device='cuda:0',dataset='kuairand',
                 epochs=50,calibration=False,engineering=False,variant='iid',code_commit=None):
        self.dir=Path(directory);self.dir.mkdir(parents=True,exist_ok=False)
        self.arm=arm;self.seed=seed;self.epochs=epochs;self.calibration=calibration;self.engineering=engineering
        self.device=torch.device(device);self.started=time.time();self.dataset=dataset
        setup_seed(seed);torch.set_num_threads(2)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        self.bundle=load_stage_s1_bundle(dataset,None,'natural' if dataset=='kuairand' else 'd2_large',seed)
        self.p=np.array([self.bundle.train_counts[d] for d in range(3)],dtype=float);self.p/=self.p.sum()
        self.alpha=self.p if arm[0]=='S' else np.ones(3)/3
        self.q=self.p if arm[1]=='N' else np.ones(3)/3
        self.weight=torch.tensor(self.alpha/self.q,dtype=torch.float32,device=self.device)
        self.sampler=Sampler(self.bundle,self.q,seed,variant)
        self.model=fixed_model(self.bundle,seed).to(self.device)
        self.named=dict(self.model.named_parameters());self.groups={g:[n for n in self.named if group(n)==g] for g in ('C','P','R','E')}
        if optimizer=='adam':self.opt=torch.optim.Adam(self.model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        elif optimizer=='sgd':self.opt=torch.optim.SGD(self.model.parameters(),lr=lr,momentum=0,weight_decay=1e-5)
        else:raise ValueError(optimizer)
        self.init=initial_state_hash(self.model);self.sources=sources();self.step_count=0
        self.counts=np.zeros(len(self.bundle.train_x),dtype=np.int32);self.best=-math.inf;self.selected=None
        self.history={};self.batch_chain=hashlib.sha256();self.previous_gradient={}
        self.config=dict(study_id='MDMT_V2_FIXED_SINGLE_20261016',arm=arm,seed=seed,dataset=dataset,
            optimizer=optimizer,lr=lr,momentum=0 if optimizer=='sgd' else None,weight_decay=1e-5,
            betas=[.9,.999] if optimizer=='adam' else None,eps=1e-8 if optimizer=='adam' else None,
            optimizer_count=1,learned_fusion=False,darts=False,fusion_effective=.7310585975646973,
            gradient_balancing=False,dbo=False,augmentation=False,alpha=self.alpha.tolist(),q=self.q.tolist(),
            importance_weight=(self.alpha/self.q).tolist(),sampler=variant,batch_size=BATCH,epochs=epochs,
            data=self.bundle.manifest_fields(),init_sha=self.init,source_hashes=self.sources,
            code_commit=code_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            independent_review=False,review='PRIMARY_SELF_REVIEW_per_start_section11',
            calibration=calibration,engineering=engineering,epoch_index='completed epochs; 0 is initialization',
            selection='validation macro six-cell AUC; strict >; earliest tie; epochs>=1',
            loss='mean_i(alpha[d_i]/q[d_i] * mean_t(BCE))',partitions=self.groups)
        dump(self.dir/'config.json',self.config)

    def loader(self,split):
        b=self.bundle
        return DataLoader(TorchDataset(getattr(b,split+'_x'),getattr(b,split+'_y')),batch_size=BATCH,
                          shuffle=False,num_workers=0,drop_last=False)

    def evaluate(self,split,epoch,key):
        with preserve_rng_state():
            records,ids,domains,labels,pred=_prediction_records(self.bundle,self.model,self.loader(split),self.device,
                                                            getattr(self.bundle,split+'_row_ids'))
        rows=[dict(seed=self.seed,arm=self.arm,optimizer=self.config['optimizer'],dataset=self.dataset,
            split=split,epoch=epoch,checkpoint=key,domain=k%3,task=k//3,auc=float(r['auc']),
            logloss=float(r['logloss']),n=int(r['sample_count']),n_pos=int(r['positive_count']),steps=self.step_count)
            for k,r in records.items()]
        return _macro(records,'auc'),rows,dict(row_ids=ids,domains=domains,labels=labels,predictions=pred)

    def state(self,epoch):
        return dict(epoch=epoch,model=cpu(self.model.state_dict()),optimizer=cpu(self.opt.state_dict()),
            optimizer_parameter_names=list(self.named),rng=_raw_rng_state(),sampler_rng=self.sampler.rng.bit_generator.state,
            sampler_cycle_rng=self.sampler.cycle_rng.bit_generator.state,sampler_orders=self.sampler.orders,
            sampler_positions=self.sampler.positions,exposure_counts=self.counts,steps=self.step_count,
            best=self.best,selected=self.selected,history=self.history,config_sha=sha(self.dir/'config.json'),
            batch_chain=self.batch_chain.hexdigest(),epoch_boundary=True)

    def save(self,epoch,name):
        tmp=self.dir/'state.tmp.pt';torch.save(self.state(epoch),tmp);tmp.replace(self.dir/name)

    def step(self,rows,epoch,index,telemetry=False):
        b=self.bundle;x=torch.tensor(b.train_x[rows],dtype=torch.float32,device=self.device).T
        y=torch.tensor(b.train_y[rows],dtype=torch.float32,device=self.device);d=x[-1].long()
        self.model.train();self.opt.zero_grad(set_to_none=True)
        pred=self.model(x);terms=F.binary_cross_entropy(pred,y,reduction='none').mean(1)
        loss=(terms*self.weight[d]).mean();assert torch.isfinite(loss).item(),'nonfinite loss'
        loss.backward()
        before={n:p.detach().clone() for n,p in self.named.items()} if telemetry else None
        grads={g:torch.cat([(self.named[n].grad if self.named[n].grad is not None else torch.zeros_like(self.named[n])).detach().flatten() for n in ns]) for g,ns in self.groups.items()} if telemetry else None
        self.opt.step();self.step_count+=1
        if telemetry:
            observations=[]
            for g,ns in self.groups.items():
                grad=grads[g];assert torch.isfinite(grad).all().item()
                delta=sum(float((self.named[n].detach()-before[n]).double().square().sum()) for n in ns)
                moments={k:math.sqrt(sum(float(self.opt.state.get(self.named[n],{}).get(k,torch.tensor(0.)).double().square().sum()) for n in ns)) for k in ('exp_avg','exp_avg_sq')}
                prior=self.previous_gradient.get(g)
                cosine=float(F.cosine_similarity(grad,prior,dim=0)) if prior is not None else None
                observations.append(dict(epoch=epoch,batch=index,steps=self.step_count,partition=g,
                    gradient_norm=float(grad.double().norm()),update_norm=math.sqrt(delta),m_norm=moments['exp_avg'],v_norm=moments['exp_avg_sq'],
                    consecutive_gradient_cosine=cosine,lag_steps=1 if prior is not None else None))
                self.previous_gradient[g]=grad.clone()
            csv_append(self.dir/'optimizer_observations.csv',observations)
        # Lag observations are consecutive only when sampled at first two steps.
        ids=np.asarray(b.train_row_ids[rows],dtype=np.int64);self.batch_chain.update(ids.tobytes())
        counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3)
        csv_append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=index,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
            row_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),weighted_loss=float(loss.detach()),unweighted_bce=float(terms.detach().mean()))])
        np.add.at(self.counts,rows,1)
        return float(loss.detach()),float(terms.detach().mean()),counts

    def fit(self,epoch_examples=None,batch_size=BATCH):
        score,metrics,pred=self.evaluate('val',0,'initial');self.initial_score=score
        csv_append(self.dir/'metrics_epoch.csv',metrics);np.savez_compressed(self.dir/'validation_epoch00.npz',**pred)
        self.save(0,'epoch00.pt')
        n=len(self.bundle.train_x) if epoch_examples is None else epoch_examples
        for epoch in range(1,self.epochs+1):
            self.previous_gradient={};start_counts=self.counts.copy();exposure=np.zeros(3,dtype=int);weighted=0.;plain=0.
            for index,offset in enumerate(range(0,n,batch_size)):
                rows=self.sampler.sample(min(batch_size,n-offset))
                w,u,c=self.step(rows,epoch,index,telemetry=index in (0,1))
                exposure+=c;weighted+=w*len(rows);plain+=u*len(rows)
            score,metrics,pred=self.evaluate('val',epoch,'trajectory');assert math.isfinite(score)
            self.history[epoch]=score;improved=score>self.best
            if improved:self.best=score;self.selected=epoch
            csv_append(self.dir/'metrics_epoch.csv',metrics)
            stats=dict(epoch=epoch,weighted_loss=weighted/n,unweighted_bce=plain/n,validation_macro_auc=score,
                unique_epoch=int(np.count_nonzero(self.counts-start_counts)),unique_cumulative=int(np.count_nonzero(self.counts)),steps=self.step_count)
            for d,pool in enumerate(self.sampler.pools):
                vals=self.counts[pool];inc=vals-start_counts[pool]
                stats.update({f'exposure_d{d}':int(exposure[d]),f'unique_epoch_d{d}':int(np.count_nonzero(inc)),
                    f'unique_cumulative_d{d}':int(np.count_nonzero(vals)),f'repeat_median_d{d}':float(np.median(vals)),
                    f'repeat_p95_d{d}':float(np.quantile(vals,.95)),f'repeat_max_d{d}':int(vals.max())})
            csv_append(self.dir/'exposure.csv',[stats]);self.save(epoch,'last_resume.pt')
            if improved:
                self.save(epoch,'selected_state.pt');np.savez_compressed(self.dir/'validation_selected.npz',**pred)
            if epoch in COMMON or epoch==self.epochs:
                self.save(epoch,f'epoch{epoch:02d}.pt');np.savez_compressed(self.dir/f'validation_epoch{epoch:02d}.npz',**pred)
            assert self.sources==sources(),'source drift'
            dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,selected_epoch=self.selected,
                elapsed_seconds=time.time()-self.started,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if self.device.type=='cuda' else 0))
            print(json.dumps(dict(epoch=epoch,val_macro_auc=score,best=self.best,elapsed=time.time()-self.started)),flush=True)
        dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected,validation_macro_auc=self.best,
            initial_validation_auc=self.initial_score,history=self.history,test_access_started=False,calibration=self.calibration))
        if not self.calibration:
            keys={'selected':self.dir/'selected_state.pt'}
            keys.update({f'epoch{e:02d}':self.dir/f'epoch{e:02d}.pt' for e in (0,)+COMMON if (self.dir/f'epoch{e:02d}.pt').exists()})
            for key,path in keys.items():
                state=torch.load(path,map_location='cpu',weights_only=False);self.model.load_state_dict(state['model'],strict=True)
                self.step_count=state['steps'];score,metrics,pred=self.evaluate('test',state['epoch'],key)
                csv_append(self.dir/'metrics_test.csv',metrics);np.savez_compressed(self.dir/f'test_{key}.npz',**pred)
        progress=dict(status='COMPLETE',epochs_completed=self.epochs,selected_epoch=self.selected,
            best_validation_macro_auc=self.best,initial_validation_macro_auc=self.initial_score,
            wall_seconds=time.time()-self.started,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if self.device.type=='cuda' else 0)
        dump(self.dir/'progress.json',progress)
        dump(self.dir/'run_manifest.json',dict(**progress,config=self.config,artifact_hashes={p.name:sha(p) for p in self.dir.iterdir()
            if p.is_file() and p.name not in ('stdout.log','run_manifest.json')}))


def smoke(directory):
    r=Runner('SB',2022,directory,device='cpu',engineering=True,epochs=1)
    b=r.bundle
    for split in ('val','test'):
        xx=getattr(b,split+'_x');yy=getattr(b,split+'_y');ids=getattr(b,split+'_row_ids');chosen=[]
        for d in range(3):
            indices=np.flatnonzero(xx[:,-1]==d);chosen.extend(indices[:16])
            for t in range(2):
                for label in (0,1):chosen.append(indices[np.flatnonzero(yy[indices,t]==label)[0]])
        chosen=np.unique(chosen)
        for suffix,arr in (('_x',xx),('_y',yy),('_row_ids',ids)):setattr(b,split+suffix,arr[chosen])
    rows=np.concatenate([p[:4] for p in r.sampler.pools]);x=torch.tensor(b.train_x[rows]).T;y=torch.tensor(b.train_y[rows]);domain=x[-1].long()
    terms=F.binary_cross_entropy(r.model(x),y,reduction='none').mean(1)
    actual=(terms*r.weight[domain]).mean();expected=sum(terms[domain==d].sum()*float(r.alpha[d]/r.q[d])/len(rows) for d in range(3))
    assert torch.allclose(actual,expected,atol=1e-7)
    assert len(r.opt.param_groups)==1 and not any(n.startswith('_weight') for n in r.named)
    one=Sampler(b,r.q,2022);two=Sampler(b,r.q,2022);assert np.array_equal(one.sample(40960),two.sample(40960))
    r.fit(epoch_examples=96,batch_size=32)
    state=torch.load(r.dir/'selected_state.pt',map_location='cpu',weights_only=False)
    assert state['steps']==3 and state['optimizer']['state']
    model=fixed_model(b,2022);model.load_state_dict(state['model'],strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state['model'].items())
    dump(OUT/'engineering_verification.json',dict(status='PASS',loss_oracle=True,single_optimizer=True,
        learned_fusion=False,same_q_pairing=True,full_pipeline=True,worker_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','check','profile'),default='train')
    p.add_argument('--arm',choices=('SN','SB','EN','EB'),default='SN');p.add_argument('--seed',type=int,default=2022)
    p.add_argument('--optimizer',choices=('adam','sgd'),default='adam');p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--dataset',choices=('kuairand','movielens'),default='kuairand');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--calibration',action='store_true');p.add_argument('--variant',choices=('iid','cyclic'),default='iid')
    p.add_argument('--code-commit');args=p.parse_args()
    try:
        if args.mode=='check':smoke(args.run_dir);return
        r=Runner(args.arm,args.seed,args.run_dir,optimizer=args.optimizer,lr=args.lr,device=args.device,
            dataset=args.dataset,epochs=args.epochs,calibration=args.calibration,variant=args.variant,code_commit=args.code_commit)
        if args.mode=='profile':
            for i in range(3):r.step(r.sampler.sample(BATCH),1,i,telemetry=i<2)
            torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/1024**2
            dump(OUT/'cuda_profile.json',dict(status='PASS',peak_reserved_mib=peak,reservation_mib=peak*1.2+512,worker_sha256=sha(__file__)))
        else:r.fit()
    except BaseException as exc:
        if args.run_dir.exists():dump(args.run_dir/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
