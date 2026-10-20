"""N1/N2 frozen objective training. Raw outputs only; no research decisions."""
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
import n17_common as c
from v3_train_20261016 import predict,metrics

class Runner:
    def __init__(self,a):
        self.a=a;self.dir=a.run_dir;self.dir.mkdir(parents=True,exist_ok=a.resume)
        self.device=torch.device(a.device);torch.set_num_threads(2);c.base.setup_seed(a.seed)
        if self.device.type=='cuda':torch.cuda.set_device(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        self.b=c.bundle(seed=a.seed);self.sp=c.split();assert np.array_equal(self.sp['row_ids'],self.b.val_row_ids)
        self.selection=self.sp['selection'];self.diagnostic=self.sp['diagnostic']
        self.model=c.base.fixed_model(self.b,a.seed).to(self.device);self.named=dict(self.model.named_parameters())
        self.p=np.array([self.b.train_counts[d] for d in range(3)],dtype=float);self.p/=self.p.sum()
        self.q=self.p.copy() if a.arm=='SN' else np.ones(3)/3
        self.weight=torch.tensor(self.p/self.q,dtype=torch.float32,device=self.device)
        self.sampler=c.base.Sampler(self.b,self.q,a.seed,'iid')
        if a.optimizer=='adam':self.opt=torch.optim.Adam(self.model.parameters(),lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        else:self.opt=torch.optim.SGD(self.model.parameters(),lr=a.lr,momentum=0,weight_decay=1e-5)
        self.source_hashes=c.sources();self.best=-math.inf;self.selected=0;self.history={};self.steps=0;self.start_epoch=0
        self.config=dict(study='N17',phase=a.phase,seed=a.seed,population_seed=2022,arm=a.arm,optimizer=a.optimizer,lr=a.lr,
            epochs=a.epochs,batch_size=a.batch_size,weight_decay=1e-5,p=self.p.tolist(),q=self.q.tolist(),importance_weight=(self.p/self.q).tolist(),
            init_sha256=c.base.initial_state_hash(self.model),data=self.b.manifest_fields(),source_hashes=self.source_hashes,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
            data_cache=str(c.HISTORY/'local_cache/kuairand_2022.pkl'),split_sha256=c.sha(c.HISTORY/'splits/kuairand_seed2022.npz'),
            sampler_seed=a.seed*1000+17,initialization_seed=a.seed,learned_fusion=False,fusion_effective=.7310585975646973,
            darts=False,optimizer_count=1,dbo=False,gradient_balancing=False,augmentation=False,routing_input_detached=True,
            forward='forward(x,test_flag=False)',loss='mean_i[p_d/q_d * mean_t BCE]',
            selection='macro six-cell AUC on selection half, earliest strict max epochs1..100',test_access=False,
            independent_review=False,review='primary self review under start.md section11',engineering=a.smoke)
        if a.resume:
            original=json.loads((self.dir/'config.json').read_text())
            for key in ('seed','arm','optimizer','lr','batch_size','p','q','source_hashes','init_sha256','split_sha256'):
                assert original[key]==self.config[key],f'resume mismatch {key}'
            self.config=original;state=torch.load(self.dir/'last_resume.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer'])
            self.sampler.rng.bit_generator.state=state['sampler_rng'];self.steps=state['steps'];self.best=state['best'];self.selected=state['selected'];self.history=state['history'];self.start_epoch=state['epoch']
            rng=state['rng'];torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
            if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
            # Remove only uncommitted epoch rows from this new run after an interruption.
            import pandas as pd
            for file in ('trajectory.csv','metrics_selection.csv','batch_trace.csv'):
                path=self.dir/file
                if path.exists():
                    df=pd.read_csv(path);df[df.epoch<=self.start_epoch].to_csv(path,index=False)
        else:c.dump(self.dir/'config.json',self.config)
        self.started=time.time()

    def state(self,epoch):
        return dict(epoch=epoch,model=c.base.cpu(self.model.state_dict()),optimizer=c.base.cpu(self.opt.state_dict()),
            optimizer_parameter_names=list(self.named),rng=c.base._raw_rng_state(),sampler_rng=copy.deepcopy(self.sampler.rng.bit_generator.state),
            steps=self.steps,best=self.best,selected=self.selected,history=self.history,source_hashes=self.source_hashes,state_complete=True)

    def save(self,epoch,name):
        tmp=self.dir/(name+'.tmp');torch.save(self.state(epoch),tmp);tmp.replace(self.dir/name)

    def step(self,rows,epoch,index):
        x=torch.tensor(self.b.train_x[rows],device=self.device,dtype=torch.float32).T
        y=torch.tensor(self.b.train_y[rows],device=self.device,dtype=torch.float32)
        self.model.train();self.opt.zero_grad(set_to_none=True)
        terms=F.binary_cross_entropy(self.model(x),y,reduction='none').mean(1)
        loss=(terms*self.weight[x[-1].long()]).mean();assert torch.isfinite(loss),'nonfinite loss'
        loss.backward();self.opt.step();self.steps+=1
        counts=np.bincount(self.b.train_x[rows,-1].astype(int),minlength=3)
        c.base.csv_append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=index,step=self.steps,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),row_ids_sha256=hashlib.sha256(self.b.train_row_ids[rows].astype(np.int64).tobytes()).hexdigest(),objective=float(loss.detach()))])
        return float(loss.detach())*len(rows)

    @torch.no_grad()
    def objective(self):
        self.model.eval();total=0.;n=len(self.b.train_x)
        for start in range(0,n,self.a.batch_size):
            x=torch.tensor(self.b.train_x[start:start+self.a.batch_size],device=self.device,dtype=torch.float32).T
            y=torch.tensor(self.b.train_y[start:start+self.a.batch_size],device=self.device,dtype=torch.float32)
            total+=float(F.binary_cross_entropy(self.model(x),y,reduction='sum'))/2
        return total/n

    def fit(self):
        a=self.a;n=192 if a.smoke else len(self.b.train_x)
        if not a.resume:self.save(0,'epoch000.pt')
        for epoch in range(self.start_epoch+1,a.epochs+1):
            start=time.time();weighted=0.;sampler_state=copy.deepcopy(self.sampler.rng.bit_generator.state)
            for index,offset in enumerate(range(0,n,a.batch_size)):
                rows=self.sampler.sample(min(a.batch_size,n-offset));weighted+=self.step(rows,epoch,index)
            assert all(torch.isfinite(p).all() for p in self.model.parameters()),'nonfinite parameter'
            pred=predict(self.model,self.b,'val',self.device,batch_size=a.batch_size)
            np.savez_compressed(self.dir/f'validation_epoch{epoch:03d}.npz',predictions=pred)
            cells=metrics(self.b,pred,'val',self.selection);score=float(np.mean([r['auc'] for r in cells]));assert math.isfinite(score)
            for r in cells:r.update(epoch=epoch,seed=a.seed,arm=a.arm)
            c.base.csv_append(self.dir/'metrics_selection.csv',cells);self.history[epoch]=score
            improved=score>self.best
            if improved:self.best=score;self.selected=epoch
            objective=self.objective();assert math.isfinite(objective),'nonfinite J'
            stats=dict(epoch=epoch,train_empirical_objective=objective,online_objective=weighted/n,selection_macro_auc=score,
                steps=self.steps,examples=epoch*n,epoch_seconds=time.time()-start,elapsed_seconds=time.time()-self.started)
            c.base.csv_append(self.dir/'trajectory.csv',[stats]);c.dump(self.dir/f'sampler_epoch{epoch:03d}.json',sampler_state)
            if improved:self.save(epoch,'selected_state.pt')
            if epoch in (10,40,100,200):self.save(epoch,f'epoch{epoch:03d}.pt')
            elif epoch in (20,50):torch.save(dict(epoch=epoch,model=c.base.cpu(self.model.state_dict())),self.dir/f'epoch{epoch:03d}_model.pt')
            self.save(epoch,'last_resume.pt')
            c.dump(self.dir/'progress.json',dict(status='TRAINING',epoch_completed=epoch,selected_epoch=self.selected,steps=self.steps,
                elapsed_seconds=time.time()-self.started,epoch_seconds=stats['epoch_seconds'],peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if self.device.type=='cuda' else 0))
            print(json.dumps(stats),flush=True)
            assert self.source_hashes==c.sources(),'frozen source drift'
        c.dump(self.dir/'selection_frozen.json',dict(selected_epoch=self.selected,history=self.history,budget=a.epochs,test_read=False))
        c.dump(self.dir/'receipt.json',dict(status='COMPLETE_TRAINING_TEST_DEFERRED',epochs=a.epochs,steps=self.steps,examples=n*a.epochs,
            selected_epoch=self.selected,best_selection_auc=self.best,wall_seconds=time.time()-self.started,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if self.device.type=='cuda' else 0,
            config_sha256=c.sha(self.dir/'config.json'),source_hashes=self.source_hashes))

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['SN','SB'],required=True)
    p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--phase',default='N1_pilot');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=40960)
    p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:Runner(a).fit()
    except Exception:
        c.dump(a.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
