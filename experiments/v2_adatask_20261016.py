"""Published AdaTask-Adam mechanism on the frozen v2 prediction/objective."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import v2_20261016 as base

original_sources=base.sources
def sources():return {**original_sources(),'experiments/v2_adatask_20261016.py':base.sha(__file__)}
base.sources=sources
OUT=base.OUT


class TaskAdam(torch.optim.Optimizer):
    def __init__(self,named,lr=.01,wd=1e-5):
        self.named=dict(named);self.shared={p for n,p in named if not n.startswith(('gate.','tower.'))}
        self.task_gradients={}
        super().__init__(list(self.named.values()),dict(lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=wd))

    @torch.no_grad()
    def step(self,closure=None):
        for cfg in self.param_groups:
            b1,b2=cfg['betas'];lr=cfg['lr'];eps=cfg['eps'];wd=cfg['weight_decay']
            for p in cfg['params']:
                if p.grad is None:continue
                s=self.state[p];s['step']=s.get('step',0)+1;step=s['step'];denom_bias=math.sqrt(1-b2**step)
                if p in self.shared:
                    # Both task gradients and L2 terms are evaluated before updating theta.
                    gs=[g.detach()+wd/2*p.detach() for g in self.task_gradients[p]]
                    update=torch.zeros_like(p)
                    for task,g in enumerate(gs):
                        mk='exp_avg_'+str(task);vk='exp_avg_sq_'+str(task)
                        if mk not in s:s[mk]=torch.zeros_like(p);s[vk]=torch.zeros_like(p)
                        s[mk].mul_(b1).add_(g,alpha=1-b1);s[vk].mul_(b2).addcmul_(g,g,value=1-b2)
                        update.addcdiv_(s[mk],s[vk].sqrt().div(denom_bias).add_(eps),value=lr/(1-b1**step))
                    p.sub_(update)
                else:
                    g=p.grad.detach()+wd*p.detach()
                    if 'exp_avg' not in s:s['exp_avg']=torch.zeros_like(p);s['exp_avg_sq']=torch.zeros_like(p)
                    s['exp_avg'].mul_(b1).add_(g,alpha=1-b1);s['exp_avg_sq'].mul_(b2).addcmul_(g,g,value=1-b2)
                    p.addcdiv_(s['exp_avg'],s['exp_avg_sq'].sqrt().div(denom_bias).add_(eps),value=-lr/(1-b1**step))


class Runner(base.Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.opt=TaskAdam(list(self.named.items()),lr=self.config['lr'])
        self.config.update(optimizer='adatask',task_state_shared='all except gate/tower',
            l2_allocation='wd/2 theta in each task gradient; total raw regularized gradient unchanged',
            upstream_commit='84853de15dedda72d65629c8cedfcd55cbb720cf',
            upstream_source='https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py',
            source_hashes=self.sources)
        base.dump(self.dir/'config.json',self.config)

    def step(self,rows,epoch,index,telemetry=False):
        b=self.bundle;x=torch.tensor(b.train_x[rows],dtype=torch.float32,device=self.device).T
        y=torch.tensor(b.train_y[rows],dtype=torch.float32,device=self.device);d=x[-1].long()
        self.model.train();self.opt.zero_grad(set_to_none=True)
        pred=self.model(x);terms=F.binary_cross_entropy(pred,y,reduction='none')
        losses=(terms*self.weight[d,None]).mean(0)/2;loss=losses.sum();assert torch.isfinite(loss).item()
        parameters=list(self.named.values())
        gradients=[torch.autograd.grad(losses[t],parameters,retain_graph=t==0,allow_unused=True) for t in range(2)]
        self.opt.task_gradients={}
        for i,p in enumerate(parameters):
            gs=[gradients[t][i] if gradients[t][i] is not None else torch.zeros_like(p) for t in range(2)]
            p.grad=gs[0]+gs[1];self.opt.task_gradients[p]=gs
        before={n:p.detach().clone() for n,p in self.named.items()} if telemetry else None
        raw={g:torch.cat([self.named[n].grad.detach().flatten() for n in ns]) for g,ns in self.groups.items()} if telemetry else None
        self.opt.step();self.step_count+=1
        if telemetry:
            observations=[]
            for g,ns in self.groups.items():
                grad=raw[g];assert torch.isfinite(grad).all().item()
                delta=sum(float((self.named[n].detach()-before[n]).double().square().sum()) for n in ns)
                moments={}
                for moment in ('exp_avg','exp_avg_sq'):
                    moments[moment]=math.sqrt(sum(float(v.double().square().sum()) for n in ns for key,v in self.opt.state[self.named[n]].items() if key in (moment,moment+'_0',moment+'_1')))
                prior=self.previous_gradient.get(g)
                observations.append(dict(epoch=epoch,batch=index,steps=self.step_count,partition=g,
                    gradient_norm=float(grad.double().norm()),update_norm=math.sqrt(delta),m_norm=moments['exp_avg'],v_norm=moments['exp_avg_sq'],
                    consecutive_gradient_cosine=float(F.cosine_similarity(grad,prior,dim=0)) if prior is not None else None,
                    lag_steps=1 if prior is not None else None,moment_summary='sqrt sum task-state squared norms'))
                self.previous_gradient[g]=grad.clone()
            base.csv_append(self.dir/'optimizer_observations.csv',observations)
        ids=np.asarray(b.train_row_ids[rows],dtype=np.int64);self.batch_chain.update(ids.tobytes())
        counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3)
        base.csv_append(self.dir/'batch_trace.csv',[dict(epoch=epoch,batch=index,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
            row_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),weighted_loss=float(loss.detach()),unweighted_bce=float(terms.detach().mean()))])
        np.add.at(self.counts,rows,1)
        return float(loss.detach()),float(terms.detach().mean()),counts


def check():
    source=Path('/tmp/v2_adatask_20261016/adatask.py')
    spec=importlib.util.spec_from_file_location('official_adatask',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(20261016)
    one=torch.nn.Parameter(torch.randn(7,dtype=torch.float64));two=torch.nn.Parameter(one.detach().clone())
    private=torch.nn.Parameter(torch.randn(5,dtype=torch.float64));private_ref=torch.nn.Parameter(private.detach().clone())
    ours=TaskAdam([('shared',one),('tower.0.weight',private)],lr=.01)
    official=module.Adam_with_AdaTask([two],lr=.01,n_tasks=2,task_weight=[1,1]);vanilla=torch.optim.Adam([private_ref],lr=.01,weight_decay=1e-5)
    max_error=0.
    for step in range(8):
        g0=torch.randn_like(one);g1=torch.randn_like(one);gp=torch.randn_like(private)
        regularized=[g0+1e-5/2*one.detach(),g1+1e-5/2*one.detach()]
        one.grad=g0+g1;private.grad=gp;private_ref.grad=gp.clone();ours.task_gradients={one:[g0,g1]}
        ours.step();official.step([two],[],[[regularized[0]],[regularized[1]]],[]);vanilla.step()
        error=max(float((one-two).abs().max()),float((private-private_ref).abs().max()));max_error=max(max_error,error)
        assert error<1e-12,(step,error)
    state=ours.state_dict();restored=TaskAdam([('shared',one),('tower.0.weight',private)]);restored.load_state_dict(state)
    assert restored.state[one]['step']==8 and torch.equal(restored.state[one]['exp_avg_0'],ours.state[one]['exp_avg_0'])
    # Raw two-task objective gradient identity on a genuine nonlinear toy output.
    theta=torch.randn(4,dtype=torch.float64,requires_grad=True);pred=torch.sigmoid(theta[:2]);target=torch.tensor([0.,1.],dtype=torch.float64)
    losses=F.binary_cross_entropy(pred,target,reduction='none')/2
    total=torch.autograd.grad(losses.sum(),theta,retain_graph=True)[0]
    separate=sum(torch.autograd.grad(losses[t],theta,retain_graph=True)[0] for t in range(2));assert torch.allclose(total,separate,atol=1e-14)
    base.dump(OUT/'adatask_engineering.json',dict(status='PASS',official_commit='84853de15dedda72d65629c8cedfcd55cbb720cf',
        official_source_sha256=base.sha(source),max_absolute_update_error=max_error,steps=8,private_torch_adam=True,
        task_gradient_sum_identity=True,optimizer_restore=True,worker_sha256=base.sha(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('check','profile','train'),default='train');p.add_argument('--run-dir',type=Path)
    p.add_argument('--arm',choices=('SN','SB'),default='SN');p.add_argument('--seed',type=int,default=2022);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--device',default='cuda:0');p.add_argument('--code-commit');p.add_argument('--calibration',action='store_true');args=p.parse_args()
    if args.mode=='check':check();return
    try:
        r=Runner(args.arm,args.seed,args.run_dir,lr=args.lr,device=args.device,epochs=args.epochs,calibration=args.calibration,code_commit=args.code_commit)
        if args.mode=='profile':
            for i in range(3):r.step(r.sampler.sample(base.BATCH),1,i,telemetry=i<2)
            torch.cuda.synchronize(r.device);peak=torch.cuda.max_memory_reserved(r.device)/1024**2
            base.dump(OUT/'adatask_cuda_profile.json',dict(status='PASS',peak_reserved_mib=peak,reservation_mib=peak*1.2+512,worker_sha256=base.sha(__file__)))
        else:r.fit()
    except BaseException as exc:
        if args.run_dir and args.run_dir.exists():base.dump(args.run_dir/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
