"""Frozen v3 objective-preserving training. Produces raw data, no conclusions."""
from __future__ import annotations
import argparse
import hashlib
import math
import random
import shutil
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import v2_20261016 as base
import v3_audit_20261016 as audit
from stage_S1_metrics import safe_auc,safe_logloss

ROOT=audit.ROOT;OUT=audit.OUT;COMMON=(1,5,10,15,20,30,40,50)

def metrics(b,pred,split,subset=None):
    y=getattr(b,split+'_y');d=getattr(b,split+'_x')[:,-1].astype(int)
    ix=np.arange(len(y)) if subset is None else subset
    result=[]
    for domain in range(3):
        chosen=ix[d[ix]==domain]
        for task in range(2):
            yy=y[chosen,task];pp=pred[chosen,task]
            result.append(dict(domain=domain,task=task,n=len(yy),n_pos=int(yy.sum()),auc=safe_auc(yy,pp),logloss=safe_logloss(yy,pp)))
    return result

@torch.no_grad()
def predict(model,b,split,device,batch_size=40960):
    model.eval();xx=getattr(b,split+'_x');pred=[]
    for offset in range(0,len(xx),batch_size):
        x=torch.tensor(xx[offset:offset+batch_size],device=device,dtype=torch.float32).T
        pred.append(model(x).cpu().numpy())
    return np.concatenate(pred)

def corruption(b,prob,domain):
    if not prob:return
    for split in ('train','val','test'):
        x=getattr(b,split+'_x');ids=getattr(b,split+'_row_ids')
        rng=np.random.default_rng(20261016+{'train':1,'val':2,'test':3}[split])
        # Coupled clean/mild/strong: identical uniform masks before thresholding.
        chosen=np.flatnonzero(x[:,-1]==domain)
        mask=rng.random((len(chosen),x.shape[1]-1))<prob
        values=x[chosen,:-1].copy();values[mask]=0;x[chosen,:-1]=values

class Runner:
    def __init__(self,args):
        self.a=args;self.dir=args.run_dir;self.dir.mkdir(parents=True,exist_ok=args.resume)
        self.device=torch.device(args.device);torch.set_num_threads(2);base.setup_seed(args.seed)
        if self.device.type=='cuda':torch.cuda.set_device(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        self.b=audit.bundle(args.dataset,args.seed)
        split_file=OUT/'splits'/f'{args.dataset}_seed{args.seed}.npz'
        frozen=np.load(split_file);assert np.array_equal(frozen['row_ids'],self.b.val_row_ids)
        self.selection=frozen['selection'];self.diagnostic=frozen['diagnostic']
        self.data_manifest=self.b.manifest_fields();self.model=base.fixed_model(self.b,args.seed).to(self.device)
        self.named=dict(self.model.named_parameters());self.initial=base.initial_state_hash(self.model)
        self.p=np.array([self.b.train_counts[d] for d in range(3)],dtype=float);self.p/=self.p.sum()
        self.q=self.p if args.arm=='SN' else np.ones(3)/3
        self.weight=torch.tensor(self.p/self.q,device=self.device,dtype=torch.float32)
        self.sampler=base.Sampler(self.b,self.q,args.seed,'iid' if args.replacement=='WR' else 'cyclic')
        self.small=int(np.argmin(self.p));corruption(self.b,args.corruption,self.small)
        if args.duplication>1:
            self.sampler.pools[self.small]=np.tile(self.sampler.pools[self.small],args.duplication)
            self.sampler.orders[self.small]=np.arange(len(self.sampler.pools[self.small]));self.sampler.cycle_rng.shuffle(self.sampler.orders[self.small])
        if args.optimizer=='adam':self.opt=torch.optim.Adam(self.model.parameters(),lr=args.lr,weight_decay=1e-5)
        elif args.optimizer=='sgd':self.opt=torch.optim.SGD(self.model.parameters(),lr=args.lr,momentum=0,weight_decay=1e-5)
        else:
            from v2_adatask_20261016 import TaskAdam
            self.opt=TaskAdam(list(self.named.items()),lr=args.lr)
        self.sources={**base.sources(),'experiments/v3_train_20261016.py':base.sha(__file__),'experiments/v3_audit_20261016.py':base.sha(audit.__file__)}
        self.config=dict(study='V3_OBJECTIVE_EQUIVALENT_20261016',**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            p=self.p.tolist(),q=self.q.tolist(),importance_weight=(self.p/self.q).tolist(),data=self.data_manifest,
            init_sha256=self.initial,source_hashes=self.sources,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            learned_fusion=False,darts=False,optimizer_count=1,weight_decay=1e-5,loss='mean_i[p_d/q_d * mean_task BCE]',
            selection='selection validation macro AUC, earliest strict max epochs>=1',diagnostic_used_for_selection=False,test_access=False,
            independent_review=False,split_sha256=base.sha(OUT/'splits'/f'{args.dataset}_seed{args.seed}.npz'))
        if args.corruption:
            self.config['transformed_feature_sha256']={split:hashlib.sha256(np.ascontiguousarray(getattr(self.b,split+'_x')).view(np.uint8)).hexdigest() for split in ('train','val','test')}
        if not args.resume:base.dump(self.dir/'config.json',self.config)
        self.started=time.time();self.steps=0;self.best=-math.inf;self.selected=0;self.history={};self.start_epoch=0
        self.counts=np.zeros(len(self.b.train_x),dtype=np.int32);self.first_seen=np.full(len(self.counts),-1,dtype=np.int32)
        self.chain=hashlib.sha256();self.previous={}
        if args.resume:
            original=__import__('json').loads((self.dir/'config.json').read_text())
            for key in ('dataset','seed','arm','optimizer','lr','batch_size','replacement','order','duplication','corruption','source_hashes','init_sha256'):
                assert original[key]==self.config[key],f'Resume mismatch {key}'
            self.config=original;state=torch.load(self.dir/'last_resume.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer'])
            self.sampler.rng.bit_generator.state=state['sampler_rng'];self.sampler.cycle_rng.bit_generator.state=state['sampler_cycle_rng']
            self.sampler.orders=state['sampler_orders'];self.sampler.positions=state['sampler_positions']
            self.counts=state['counts'];self.first_seen=state['first_seen'];self.steps=state['steps'];self.best=state['best']
            self.selected=state['selected'];self.history=state['history'];self.start_epoch=state['epoch']
            rng=state['rng'];torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
            if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
            # Reconstruct hashlib internal state from immutable exact batch row lists.
            for epoch in range(1,self.start_epoch+1):
                batches=np.load(self.dir/f'batches_epoch{epoch:03d}.npz');starts=np.r_[0,np.cumsum(batches['sizes'])]
                for index in batches['order']:
                    rows=batches['rows'][starts[index]:starts[index+1]];self.chain.update(self.b.train_row_ids[rows].astype(np.int64).tobytes())
            assert self.chain.hexdigest()==state['batch_chain']
            for filename in ('receipt.json','selection_frozen.json'):
                current=self.dir/filename
                if current.exists():
                    record=__import__('json').loads(current.read_text())
                    prior_epoch=record.get('epochs',max(map(int,record.get('history',{'0':0}))))
                    snapshot=self.dir/f'{current.stem}_until{prior_epoch:03d}.json'
                    if not snapshot.exists():shutil.copyfile(current,snapshot)
            base.dump(self.dir/f'extension_to{args.epochs}.json',dict(start_epoch=self.start_epoch,new_total_epochs=args.epochs,original_config_sha=base.sha(self.dir/'config.json'),test_read=False))

    def state(self,epoch):
        return dict(epoch=epoch,model=base.cpu(self.model.state_dict()),optimizer=base.cpu(self.opt.state_dict()),
            optimizer_parameter_names=list(self.named),rng=base._raw_rng_state(),sampler_rng=self.sampler.rng.bit_generator.state,
            sampler_cycle_rng=self.sampler.cycle_rng.bit_generator.state,sampler_orders=self.sampler.orders,sampler_positions=self.sampler.positions,
            counts=self.counts,first_seen=self.first_seen,steps=self.steps,best=self.best,selected=self.selected,history=self.history,
            batch_chain=self.chain.hexdigest(),config_sha=base.sha(self.dir/'config.json'),state_complete=True)

    def save(self,epoch,name):
        temp=self.dir/'save.tmp.pt';torch.save(self.state(epoch),temp);temp.replace(self.dir/name)

    def step(self,rows,epoch,index):
        b=self.b;x=torch.tensor(b.train_x[rows],device=self.device,dtype=torch.float32).T
        y=torch.tensor(b.train_y[rows],device=self.device,dtype=torch.float32)
        self.model.train();self.opt.zero_grad(set_to_none=True)
        terms=F.binary_cross_entropy(self.model(x),y,reduction='none');task_losses=(terms*self.weight[x[-1].long(),None]).mean(0)/2
        loss=task_losses.sum() if self.a.optimizer=='adatask' else (terms.mean(1)*self.weight[x[-1].long()]).mean()
        assert torch.isfinite(loss)
        if self.a.optimizer=='adatask':
            params=list(self.named.values());gs=[torch.autograd.grad(task_losses[t],params,retain_graph=t==0,allow_unused=True) for t in range(2)]
            self.opt.task_gradients={}
            for i,p in enumerate(params):
                gradients=[gs[t][i] if gs[t][i] is not None else torch.zeros_like(p) for t in range(2)]
                p.grad=gradients[0]+gradients[1];self.opt.task_gradients[p]=gradients
        else:loss.backward()
        telemetry=index<6
        if telemetry:
            before={n:p.detach().clone() for n,p in self.named.items()}
            gradients={g:torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for n,p in self.named.items() if base.group(n)==g]) for g in 'CPRE'}
        self.opt.step();self.steps+=1
        if telemetry:
            observations=[]
            for g,gradient in gradients.items():
                names=[n for n in self.named if base.group(n)==g];history=self.previous.setdefault(g,[])
                moments={key:math.sqrt(sum(float(v.double().square().sum()) for n in names for k,v in self.opt.state[self.named[n]].items() if k==key or k.startswith(key+'_'))) for key in ('exp_avg','exp_avg_sq')}
                observations.append(dict(epoch=epoch,batch=index,steps=self.steps,block=g,gradient_norm=float(gradient.double().norm()),
                    update_norm=math.sqrt(sum(float((self.named[n].detach()-before[n]).double().square().sum()) for n in names)),
                    lag1_cosine=float(F.cosine_similarity(gradient,history[-1],dim=0)) if len(history)>=1 else None,
                    lag2_cosine=float(F.cosine_similarity(gradient,history[-2],dim=0)) if len(history)>=2 else None,
                    lag5_cosine=float(F.cosine_similarity(gradient,history[-5],dim=0)) if len(history)>=5 else None,**moments))
                history.append(gradient.clone())
                if len(history)>5:history.pop(0)
            base.csv_append(self.dir/'optimizer_observations.csv',observations)
        ids=b.train_row_ids[rows].astype(np.int64);fingerprint=hashlib.sha256(ids.tobytes()).hexdigest();self.chain.update(ids.tobytes())
        domains=b.train_x[rows,-1].astype(int);counts=np.bincount(domains,minlength=3);ww=self.p[domains]/self.q[domains]
        new=rows[self.first_seen[rows]<0];self.first_seen[new]=self.steps
        np.add.at(self.counts,rows,1)
        base.csv_append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=index,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
            row_ids_sha256=fingerprint,weighted_loss=float(loss.detach()),unweighted_bce=float(terms.mean().detach()),
            ess=float(ww.sum()**2/np.square(ww).sum()),weight_variance=float(ww.var()),max_weight=float(ww.max()))])
        return float(loss.detach())*len(rows),float(terms.mean().detach())*len(rows)

    @torch.no_grad()
    def empirical_objective(self):
        self.model.eval();total=0.;n=len(self.b.train_x)
        for offset in range(0,n,self.a.batch_size):
            x=torch.tensor(self.b.train_x[offset:offset+self.a.batch_size],device=self.device,dtype=torch.float32).T
            y=torch.tensor(self.b.train_y[offset:offset+self.a.batch_size],device=self.device,dtype=torch.float32)
            total+=float(F.binary_cross_entropy(self.model(x),y,reduction='sum'))/2
        return total/n

    def fit(self):
        a=self.a;n=len(self.b.train_x) if not a.smoke else 3*min(a.batch_size,64)
        if not a.resume:self.save(0,'epoch00.pt')
        for epoch in range(self.start_epoch+1,a.epochs+1):
            before=self.counts.copy();self.previous={}
            batches=[self.sampler.sample(min(a.batch_size,n-offset)) for offset in range(0,n,a.batch_size)]
            order=np.arange(len(batches))
            if a.order=='PERMUTED':np.random.default_rng(a.seed*100000+epoch+20261016).shuffle(order)
            sizes=np.array([len(x) for x in batches],dtype=np.int32)
            base_rows=np.concatenate(batches).astype(np.int32)
            np.savez_compressed(self.dir/f'batches_epoch{epoch:03d}.npz',rows=base_rows,sizes=sizes,order=order)
            weighted=plain=0.
            for index,j in enumerate(order):
                w,u=self.step(batches[j],epoch,index);weighted+=w;plain+=u
            predictions=predict(self.model,self.b,'val',self.device)
            np.savez_compressed(self.dir/f'validation_epoch{epoch:03d}.npz',predictions=predictions)
            rows=metrics(self.b,predictions,'val',self.selection);score=float(np.mean([r['auc'] for r in rows]));assert math.isfinite(score)
            for r in rows:r.update(epoch=epoch,split='selection',seed=a.seed,arm=a.arm)
            base.csv_append(self.dir/'metrics_selection.csv',rows);self.history[epoch]=score
            improved=score>self.best
            if improved:self.best=score;self.selected=epoch
            # Full empirical target loss evaluated at fixed theta, separately from online stochastic loss.
            objective=self.empirical_objective()
            stats=dict(epoch=epoch,online_objective=weighted/n,online_unweighted_bce=plain/n,train_empirical_objective=objective,
                selection_macro_auc=score,steps=self.steps,examples=epoch*n)
            increments=self.counts-before
            for d,pool in enumerate([np.flatnonzero(self.b.train_x[:,-1]==d) for d in range(3)]):
                inc=increments[pool];cumulative=self.counts[pool];values,freq=np.unique(inc,return_counts=True)
                stats.update({f'exposure_d{d}':int(inc.sum()),f'unique_epoch_d{d}':int(np.count_nonzero(inc)),
                    f'unique_cumulative_d{d}':int(np.count_nonzero(cumulative)),f'coverage_d{d}':float(np.count_nonzero(cumulative)/len(pool)),
                    f'repeat_max_d{d}':int(inc.max()),f'repeat_histogram_d{d}':';'.join(f'{i}:{j}' for i,j in zip(values,freq))})
            base.csv_append(self.dir/'trajectory.csv',[stats]);self.save(epoch,'last_resume.pt')
            if improved:self.save(epoch,'selected_state.pt')
            if epoch in COMMON or epoch==a.epochs:self.save(epoch,f'epoch{epoch:03d}.pt')
            torch.save(dict(epoch=epoch,model=base.cpu(self.model.state_dict())),self.dir/f'epoch{epoch:03d}_model.pt')
            base.dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,selected_epoch=self.selected,
                elapsed_seconds=time.time()-self.started,peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if self.device.type=='cuda' else 0))
            print(epoch,score,'elapsed',time.time()-self.started,flush=True)
            assert all(base.sha(ROOT/name)==fingerprint for name,fingerprint in self.sources.items()),'Frozen source drift'
        base.dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected,history=self.history,rules=['macro_auc','macro_logloss','early_auc_1_10','fixed50'],test_read=False))
        np.savez_compressed(self.dir/'exposure_final.npz',counts=self.counts,first_seen_step=self.first_seen)
        # Diagnostic/test are deliberately deferred to orchestrator's global LR/rule freeze.
        base.dump(self.dir/'receipt.json',dict(status='COMPLETE_TRAINING_TEST_DEFERRED',epochs=a.epochs,selected_epoch=self.selected,
            best_selection_auc=self.best,wall_seconds=time.time()-self.started,steps=self.steps,examples=n*a.epochs,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if self.device.type=='cuda' else 0,
            artifact_hashes={p.name:base.sha(p) for p in self.dir.iterdir() if p.is_file() and p.name not in ('receipt.json','stdout.log')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['kuairand','movielens'],default='kuairand')
    p.add_argument('--seed',type=int,default=2022);p.add_argument('--arm',choices=['SN','SB'],default='SN')
    p.add_argument('--optimizer',choices=['adam','sgd','adatask'],default='sgd');p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--batch-size',type=int,default=40960);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--replacement',choices=['WR','PERM'],default='WR');p.add_argument('--order',choices=['ORIGINAL','PERMUTED'],default='ORIGINAL')
    p.add_argument('--duplication',type=int,default=1);p.add_argument('--corruption',type=float,default=0.)
    p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:Runner(args).fit()
    except Exception:
        args.run_dir.mkdir(parents=True,exist_ok=True);base.dump(args.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
