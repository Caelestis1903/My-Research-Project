"""Raw local counterfactual and complete task-gradient diagnostics."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import v3_audit_20261016 as audit
import v3_train_20261016 as train
base=audit.base;OUT=audit.OUT

def vector(model):return torch.cat([p.detach().flatten() for p in model.parameters()])
def load_vector(model,value):
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(value[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    assert offset==len(value)

def optimizer(model,kind,lr):
    return (torch.optim.SGD(model.parameters(),lr=lr,momentum=0,weight_decay=1e-5) if kind=='sgd' else
        torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5))

@torch.no_grad()
def objective(model,b,device):
    model.eval();total=0.
    for start in range(0,len(b.train_x),40960):
        x=torch.tensor(b.train_x[start:start+40960],device=device,dtype=torch.float32).T
        y=torch.tensor(b.train_y[start:start+40960],device=device,dtype=torch.float32)
        total+=float(F.binary_cross_entropy(model(x),y,reduction='sum'))/2
    return total/len(b.train_x)

def local(a):
    freeze=json.loads((OUT/'analysis_rules_frozen.json').read_text());assert freeze['status']=='FROZEN'
    config=json.loads((a.source_run/'config.json').read_text());assert config['arm']=='SN'
    b=audit.bundle(config['dataset'],a.seed);model=base.fixed_model(b,a.seed).to(a.device)
    checkpoint=a.source_run/f'epoch{a.epoch:03d}.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);theta=vector(model).clone();groups,_=audit.block_indices(model)
    p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum()
    deltas={};draws={}
    for index,(arm,q) in enumerate((('SN',p),('SB',np.ones(3)/3))):
        load_vector(model,theta);opt=optimizer(model,config['optimizer'],config['lr']);opt.load_state_dict(copy.deepcopy(state['optimizer']))
        sampler=base.Sampler(b,q,a.seed+1000000+a.epoch*100+index,'iid');rows=sampler.sample(config['batch_size'])
        model.train();audit.grad(model,b,rows,torch.tensor(p/q,device=a.device,dtype=torch.float32),a.device);opt.step()
        deltas[arm]=(vector(model)-theta).clone();draws[arm]=dict(row_hash=hashlib.sha256(rows.tobytes()).hexdigest(),domain_counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3).tolist())
    conditions={'BASE':torch.zeros_like(theta),**deltas}
    for group,ix in groups.items():
        value=deltas['SB'].clone();value[ix]=deltas['SN'][ix];conditions[f'SB_replace_{group}']=value
    split=np.load(OUT/'splits'/f'{config["dataset"]}_seed{a.seed}.npz');cells=[];summary=[]
    for condition,delta in conditions.items():
        load_vector(model,theta+delta);pred=train.predict(model,b,'val',a.device)
        row=dict(condition=condition,seed=a.seed,epoch=a.epoch,train_empirical_objective=objective(model,b,a.device))
        for name in ('selection','diagnostic'):
            result=train.metrics(b,pred,'val',split[name])
            for cell in result:cell.update(condition=condition,split=name,seed=a.seed,epoch=a.epoch);cells.append(cell)
            for metric in ('auc','logloss'):row[f'{name}_{metric}']=float(np.mean([c[metric] for c in result]))
        for group,ix in groups.items():row[f'displacement_{group}']=float(delta[ix].double().norm())
        summary.append(row)
    pd.DataFrame(cells).to_csv(a.run_dir/'cells.csv',index=False);pd.DataFrame(summary).to_csv(a.run_dir/'summary.csv',index=False)
    base.dump(a.run_dir/'receipt.json',dict(status='COMPLETE',mode='local',seed=a.seed,epoch=a.epoch,draws=draws,
        checkpoint=str(checkpoint),checkpoint_sha=base.sha(checkpoint),config=config,source_sha=base.sha(__file__),freeze_sha=base.sha(OUT/'analysis_rules_frozen.json')))

def tasks(a):
    b=audit.bundle('kuairand',a.seed);model=base.fixed_model(b,a.seed).to(a.device)
    source=audit.OLD/f'runs/kuairand/{a.optimizer}/seed{a.seed}/SN/attempt1/epoch10.pt'
    state=torch.load(source,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.train()
    groups,_=audit.block_indices(model);groups['ALL']=np.arange(sum(p.numel() for p in model.parameters()))
    gradients=np.zeros((3,2,len(groups['ALL'])),dtype=np.float64);params=list(model.parameters())
    for d in range(3):
        pool=np.flatnonzero(b.train_x[:,-1]==d)
        for start in range(0,len(pool),40960):
            rows=pool[start:start+40960];x=torch.tensor(b.train_x[rows],device=a.device,dtype=torch.float32).T
            y=torch.tensor(b.train_y[rows],device=a.device,dtype=torch.float32)
            terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(0)
            for task in range(2):
                gs=torch.autograd.grad(terms[task],params,retain_graph=task==0,allow_unused=True)
                gradient=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(params,gs)])
                gradients[d,task]+=gradient.cpu().numpy().astype(float)*(len(rows)/len(pool))
    p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum();total=np.einsum('d,dtk->tk',p,gradients)
    records=[]
    for domain,values in [(str(d),gradients[d]) for d in range(3)]+[('ALL',total)]:
        for block,ix in groups.items():
            u,v=values[:,ix];den=np.linalg.norm(u)*np.linalg.norm(v)
            records.append(dict(domain=domain,block=block,task0_norm=float(np.linalg.norm(u)),task1_norm=float(np.linalg.norm(v)),cosine=float(u@v/den) if den else None))
    np.savez_compressed(a.run_dir/'task_gradients.npz',gradients=gradients,overall_task_gradients=total)
    pd.DataFrame(records).to_csv(a.run_dir/'cross_task_geometry.csv',index=False)
    # V1 path discovery is based on stored source config rather than a guessed name.
    matches=[]
    for path in (OUT/'audits').rglob('config.json'):
        c=json.loads(path.read_text())
        if c.get('seed')==a.seed and c.get('optimizer')==a.optimizer and c.get('epoch')==10 and c.get('k')==512:matches.append(path.parent)
    assert len(matches)==1,matches
    expected=np.load(matches[0]/'exact.npz')['domain_gradients'];actual=gradients.mean(1)
    residuals=np.linalg.norm(actual-expected,axis=1);bounds=1e-6+1e-4*np.linalg.norm(expected,axis=1)
    assert np.all(residuals<=bounds)
    base.dump(a.run_dir/'receipt.json',dict(status='PASS',mode='tasks',seed=a.seed,optimizer=a.optimizer,source_checkpoint=str(source),
        source_sha=base.sha(__file__),checkpoint_sha=base.sha(source),domain_decomposition_residuals=residuals.tolist(),bounds=bounds.tolist(),
        task_gradient_sha=base.sha(a.run_dir/'task_gradients.npz'),v1_reference=str(matches[0])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['local','tasks'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--optimizer',choices=['sgd','adam'],default='sgd');p.add_argument('--epoch',type=int,default=10)
    p.add_argument('--source-run',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);base.setup_seed(a.seed)
    if a.device.startswith('cuda'):torch.cuda.set_device(a.device)
    if a.mode=='local':local(a)
    else:tasks(a)
if __name__=='__main__':main()
