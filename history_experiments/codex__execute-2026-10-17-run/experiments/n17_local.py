"""N3 raw local norm/direction counterfactuals at a fixed native state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import n17_common as c
from n17_geometry import flat_grad,indices
from v3_train_20261016 import predict,metrics

@torch.no_grad()
def install(model,vector):
    start=0
    for p in model.parameters():p.copy_(vector[start:start+p.numel()].reshape_as(p));start+=p.numel()
    assert start==len(vector)

@torch.no_grad()
def objective(model,b,device,batch):
    model.eval();total=0.
    for start in range(0,len(b.train_x),batch):
        x=torch.tensor(b.train_x[start:start+batch],device=device,dtype=torch.float32).T;y=torch.tensor(b.train_y[start:start+batch],device=device,dtype=torch.float32)
        total+=float(F.binary_cross_entropy(model(x),y,reduction='sum'))/2
    return total/len(b.train_x)

def native_update(worker,opt,state,gradient):
    worker.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));start=0
    for p in worker.parameters():p.grad=gradient[start:start+p.numel()].reshape_as(p).clone();start+=p.numel()
    opt.step();return torch.cat([p.detach().flatten() for p in worker.parameters()]).clone()

def evaluate(worker,vector,b,sp,device,batch):
    install(worker,vector);pred=predict(worker,b,'val',device,batch_size=batch);j=objective(worker,b,device,batch);rows=[]
    for split in ('selection','diagnostic'):
        cells=metrics(b,pred,'val',sp[split]);rows.extend([dict(split=split,**row) for row in cells])
    return rows,j,hashlib.sha256(pred.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--anchor-arm',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--out',type=Path,required=True);p.add_argument('--draws',type=int,default=32);p.add_argument('--start-draw',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=40960);p.add_argument('--only-block');p.add_argument('--engineering',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(2);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    b=c.bundle(seed=a.seed);sp=c.split();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if a.engineering:
        ix=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:n] for d,n in enumerate((96,64,32))])
        for suffix in ('x','y','row_ids'):setattr(b,'train_'+suffix,getattr(b,'train_'+suffix)[ix])
        chosen=[]
        for d in range(3):
            ds=np.flatnonzero(b.val_x[:,-1]==d);chosen.extend(ds[:16])
            for task in range(2):
                for label in (0,1):chosen.append(ds[np.flatnonzero(b.val_y[ds,task]==label)[0]])
        chosen=np.unique(chosen)
        for suffix in ('x','y','row_ids'):setattr(b,'val_'+suffix,getattr(b,'val_'+suffix)[chosen])
        sp={'selection':np.arange(len(chosen)),'diagnostic':np.arange(len(chosen))}
    anchor=c.base.fixed_model(b,a.seed).to(device);anchor.load_state_dict(state['model']);anchor.train();initial=c.base.initial_state_hash(anchor)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or (isinstance(m,torch.nn.Dropout) and m.p>0) for m in anchor.modules())
    worker=c.base.fixed_model(b,a.seed).to(device)
    opt=torch.optim.Adam(worker.parameters(),lr=a.lr,weight_decay=1e-5) if a.optimizer=='adam' else torch.optim.SGD(worker.parameters(),lr=a.lr,weight_decay=1e-5)
    assert float(state['optimizer']['param_groups'][0]['lr'])==a.lr
    groups,order=indices(anchor,device);theta=torch.cat([p.detach().flatten() for p in anchor.parameters()]).clone()
    pdom=np.array([b.train_counts[d] for d in range(3)],dtype=float);pdom/=pdom.sum();epoch=state['epoch']
    stream=a.seed+epoch*100+900000
    samplers={arm:c.base.Sampler(b,q,stream,'iid') for arm,q in [('SN',pdom),('SB',np.ones(3)/3)]}
    config={**vars(a),'checkpoint':str(a.checkpoint),'out':str(a.out),'checkpoint_sha':c.sha(a.checkpoint),'epoch':epoch,'p':pdom.tolist(),'parameter_order':order,'source_sha':c.sha(__file__),'geometry_source_sha':c.sha(Path(__file__).with_name('n17_geometry.py')),'base_sources':c.sources(),'draw_stream':stream,'test_read':False}
    c.dump(a.out/'config.json',config)
    blocks=[a.only_block] if a.only_block else ['C','C_embedding','C_shared_STAR','C_skip','C_star_mlp','C_experts','P']
    assert all(block in groups for block in blocks)
    def record(draw,case,vector,block='ALL',target=None,direction_undefined=False):
        cells,j,fingerprint=evaluate(worker,vector,b,sp,device,a.batch_size);ix=groups[block];applied=float((vector[ix]-theta[ix]).double().norm())
        info=dict(draw=draw,case=case,seed=a.seed,optimizer=a.optimizer,anchor_arm=a.anchor_arm,epoch=epoch,block=block,train_J=j,predictions_sha=fingerprint,applied_block_update_norm=applied,target_block_update_norm=target,direction_undefined=direction_undefined)
        for cell in cells:c.base.csv_append(a.out/'cell_readouts.csv',[dict(**info,**cell)])
        for split in ('selection','diagnostic'):
            chosen=[x for x in cells if x['split']==split];c.base.csv_append(a.out/'macro_readouts.csv',[dict(**info,split=split,auc=float(np.mean([x['auc'] for x in chosen])),logloss=float(np.mean([x['logloss'] for x in chosen])))])
    record(-1,'unupdated_anchor',theta)
    for draw in range(a.start_draw+a.draws):
        sampled={arm:sampler.sample(a.batch_size) for arm,sampler in samplers.items()}
        if draw<a.start_draw:continue
        native={}
        for arm,q in [('SN',pdom),('SB',np.ones(3)/3)]:
            rows=sampled[arm];weights=torch.tensor(pdom/q,device=device,dtype=torch.float32);gradient=flat_grad(anchor,b,rows,device,weights)
            native[arm]=native_update(worker,opt,state,gradient)
            c.base.csv_append(a.out/'draws.csv',[dict(draw=draw,arm=arm,row_ids_sha=hashlib.sha256(b.train_row_ids[rows].astype(np.int64).tobytes()).hexdigest(),gradient_norm=float(gradient.double().norm()))])
            record(draw,'native_'+arm,native[arm])
        for block in blocks:
            ix=groups[block];cn=native['SN'][ix]-theta[ix];cb=native['SB'][ix]-theta[ix];rn=float(cn.double().norm());rb=float(cb.double().norm())
            hybrid=native['SB'].clone();hybrid[ix]=native['SN'][ix];record(draw,block+'_full',hybrid,block,rn,rn<1e-12 or rb<1e-12)
            if block=='P':continue
            if rn<1e-12 or rb<1e-12:continue
            hybrid=native['SB'].clone();hybrid[ix]=theta[ix]+cb*(rn/rb);record(draw,block+'_norm',hybrid,block,rn)
            hybrid=native['SB'].clone();hybrid[ix]=theta[ix]+cn*(rb/rn);record(draw,block+'_direction',hybrid,block,rb)
        c.dump(a.out/'progress.json',dict(status='RUNNING',draw_completed=draw+1,elapsed_seconds=time.time()-started));print('draw',draw+1,'elapsed',time.time()-started,flush=True)
    assert c.base.initial_state_hash(anchor)==initial and config['source_sha']==c.sha(__file__) and config['base_sources']==c.sources()
    assert config['geometry_source_sha']==c.sha(Path(__file__).with_name('n17_geometry.py'))
    c.dump(a.out/'receipt.json',dict(status='COMPLETE_ENGINEERING' if a.engineering else 'COMPLETE',draws=a.draws,start_draw=a.start_draw,epoch=epoch,anchor_unchanged=True,wall_seconds=time.time()-started,source_sha=c.sha(__file__),config_sha=c.sha(a.out/'config.json')))

if __name__=='__main__':main()
