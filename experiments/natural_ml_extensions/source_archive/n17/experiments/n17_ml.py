"""Conditional fixed-population MovieLens runner; optimizer seed is independent."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import torch
import n17_common as c
from n17_capture import Runner as CaptureRunner


class Runner(CaptureRunner):
    # Explicit environment initialization; every training/evaluation/update/save
    # method is inherited unchanged from the frozen N1 implementation.
    def __init__(self, a):
        self.a=a;self.dir=a.run_dir;self.dir.mkdir(parents=True,exist_ok=a.resume)
        self.device=torch.device(a.device);torch.set_num_threads(2);c.base.setup_seed(a.seed)
        if self.device.type=='cuda':torch.cuda.set_device(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        self.b=c.bundle('movielens',a.subset_seed);self.b.seed=a.seed
        self.sp=c.split('movielens',a.subset_seed);assert np.array_equal(self.sp['row_ids'],self.b.val_row_ids)
        self.selection=self.sp['selection'];self.diagnostic=self.sp['diagnostic']
        self.model=c.base.fixed_model(self.b,a.seed).to(self.device);self.named=dict(self.model.named_parameters())
        self.p=np.array([self.b.train_counts[d] for d in range(3)],dtype=float);self.p/=self.p.sum()
        self.q=self.p.copy() if a.arm=='SN' else np.ones(3)/3
        self.weight=torch.tensor(self.p/self.q,dtype=torch.float32,device=self.device)
        self.sampler=c.base.Sampler(self.b,self.q,a.seed,'iid')
        self.opt=(torch.optim.Adam(self.model.parameters(),lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
                  if a.optimizer=='adam' else torch.optim.SGD(self.model.parameters(),lr=a.lr,momentum=0,weight_decay=1e-5))
        self.source_hashes=c.sources();self.best=-math.inf;self.selected=0;self.history={};self.steps=0;self.start_epoch=0
        self.capture_sha=c.sha(Path(__file__).with_name('n17_capture.py'));self.ml_sha=c.sha(__file__)
        self.config=dict(study='N17',phase=a.phase,dataset='movielens',subset_seed=a.subset_seed,population_seed=a.subset_seed,
            seed=a.seed,arm=a.arm,optimizer=a.optimizer,lr=a.lr,epochs=a.epochs,batch_size=a.batch_size,weight_decay=1e-5,
            p=self.p.tolist(),q=self.q.tolist(),importance_weight=(self.p/self.q).tolist(),
            init_sha256=c.base.initial_state_hash(self.model),data=self.b.manifest_fields(),source_hashes=self.source_hashes,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),
            data_cache=str(c.HISTORY/'local_cache'/f'movielens_{a.subset_seed}.pkl'),
            split_sha256=c.sha(c.HISTORY/'splits'/f'movielens_seed{a.subset_seed}.npz'),
            sampler_seed=a.seed*1000+17,initialization_seed=a.seed,learned_fusion=False,fusion_effective=.7310585975646973,
            darts=False,optimizer_count=1,dbo=False,gradient_balancing=False,augmentation=False,routing_input_detached=True,
            forward='forward(x,test_flag=False)',loss='mean_i[p_d/q_d * mean_t BCE]',
            selection='macro six-cell AUC on selection half, earliest strict max epochs1..100',test_access=False,
            independent_review=False,review='primary self review under start.md section11',engineering=a.smoke,
            model_capture_source_sha256=self.capture_sha,ml_source_sha256=self.ml_sha,all_epoch_models_retained=True,test_prediction_deferred=True)
        if a.resume:
            original=json.loads((self.dir/'config.json').read_text())
            for key in ('seed','arm','optimizer','lr','batch_size','p','q','source_hashes','init_sha256','split_sha256','subset_seed','ml_source_sha256','model_capture_source_sha256'):
                assert original[key]==self.config[key],f'resume mismatch {key}'
            self.config=original;state=torch.load(self.dir/'last_resume.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer'])
            self.sampler.rng.bit_generator.state=state['sampler_rng'];self.steps=state['steps'];self.best=state['best'];self.selected=state['selected'];self.history=state['history'];self.start_epoch=state['epoch']
            rng=state['rng'];torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
            if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
            import pandas as pd
            for file in ('trajectory.csv','metrics_selection.csv','batch_trace.csv'):
                path=self.dir/file
                if path.exists():
                    frame=pd.read_csv(path);frame[frame.epoch<=self.start_epoch].to_csv(path,index=False)
        else:c.dump(self.dir/'config.json',self.config)
        self.started=time.time()

    def fit(self):
        super().fit();assert self.ml_sha==c.sha(__file__)
        receipt=json.loads((self.dir/'receipt.json').read_text());receipt['ml_source_sha256']=self.ml_sha;c.dump(self.dir/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--subset-seed',type=int,default=2022)
    p.add_argument('--arm',choices=['SN','SB'],required=True);p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--phase',default='N5_confirmation');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=40960)
    p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:Runner(a).fit()
    except Exception:
        c.dump(a.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
