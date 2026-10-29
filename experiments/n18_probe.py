"""P3 feature extraction, P4/P6 fixed-point raw measurements. No model updates."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from n18_common import OUT,bundle,dump,model_and_state,flat_grad,stratified,validation_pools,timer

def cell_gradients(model,b,rows,split,device,functional=False):
    # Remove ONLY the routing stop-gradient in the diagnostic Jacobian. Forward
    # values remain identical; the historical model and its training are untouched.
    capture={};handles=[]
    if functional:
        handles.append(model.skip_conn.register_forward_hook(lambda m,a,o:capture.__setitem__('skip',o)))
        handles.append(model.star_mlp.register_forward_hook(lambda m,a,o:capture.__setitem__('star',o)))
        def reconnect(module,args):
            value=capture['skip']+capture['star']
            assert torch.equal(value.detach(),args[0])
            return (value,)
        handles.extend(gate.register_forward_pre_hook(reconnect) for gate in model.gate)
    x=torch.tensor(getattr(b,split+'_x')[rows],device=device,dtype=torch.float32).T
    y=torch.tensor(getattr(b,split+'_y')[rows],device=device,dtype=torch.float32)
    loss=F.binary_cross_entropy(model(x),y,reduction='none')
    terms=[loss[x[-1].long()==d,t].mean() for d in range(3) for t in range(2)]
    result=torch.stack([flat_grad(v,model,i<5) for i,v in enumerate(terms)])
    for handle in handles:handle.remove()
    return result

def extraction(model,b,rows,split,device):
    captured={}
    handles=[tower.register_forward_pre_hook(lambda module,args,index=i:captured.__setitem__(index,args[0].detach()))
             for i,tower in enumerate(model.tower)]
    allz=[];allp=[]
    with torch.no_grad():
        for offset in range(0,len(rows),512):
            rr=rows[offset:offset+512];xx=getattr(b,split+'_x')[rr]
            x=torch.tensor(xx,device=device,dtype=torch.float32).T;pred=model(x)
            z=torch.stack([captured[t*3+d] for d in range(3) for t in range(2)],dim=1)
            ids=torch.tensor(xx[:,-1],device=device,dtype=torch.long)*2
            actual=torch.stack([z[torch.arange(len(rr),device=device),ids+t] for t in range(2)],dim=1)
            allz.append(actual.cpu().numpy());allp.append(pred.cpu().numpy())
    for h in handles:h.remove()
    return dict(z=np.concatenate(allz),teacher=np.concatenate(allp),y=getattr(b,split+'_y')[rows],
                domains=getattr(b,split+'_x')[rows,-1].astype(int),row_ids=getattr(b,split+'_row_ids')[rows])

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',required=True)
    p.add_argument('--out',type=Path);p.add_argument('--skip-features',action='store_true')
    a=p.parse_args();out=a.out or OUT/'measurements'/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(2);device=torch.device(a.device)
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    b=bundle();model,state,path=model_and_state(b,a.seed,device);del state
    rng=np.random.default_rng(a.seed*1000+18);sel,diag=validation_pools(b)
    params=list(model.named_parameters());count=sum(p.numel() for _,p in params)
    mask=torch.cat([torch.full((p.numel(),),name.startswith(('embedding.','shared_weight','shared_bias','skip_conn.','star_mlp.','expert.')),device=device,dtype=torch.bool) for name,p in params])
    costs={}
    for name,param in params:
        key=name.split('.')[0];costs[key]=costs.get(key,0)+param.numel()
    config=dict(seed=a.seed,device=a.device,checkpoint=str(path),epoch=10,dataset='kuairand',population_seed=2022,
        fixed_fusion=True,optimizer_count=0,new_training=False,loss='unweighted cell BCE for P4; mean-task row BCE for P6',
        cell_order=[[d,t] for d in range(3) for t in range(2)],parameter_count=count,module_costs=costs,
        parameter_order=[dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in params],test_access=False)
    dump(out/'config.json',config)
    # P3 exact row IDs and cached frozen features for subsequent fitting stage.
    start=timer(device)
    for key,split,n,pools in [('fit','train',2048,None),('selection','val',512,sel),('diagnostic','val',512,diag)]:
        rows=stratified(getattr(b,split+'_x'),n,rng,pools)
        if not a.skip_features:
            data=extraction(model,b,rows,split,device)
            np.savez_compressed(out/f'features_{key}.npz',**data)
    extraction_seconds=timer(device)-start
    # P4: genuinely non-overlapping train draws and a disjoint validation draw.
    trainpools=[rng.permutation(np.flatnonzero(b.train_x[:,-1]==d)) for d in range(3)]
    rowsA=np.concatenate([pool[:256] for pool in trainpools]);rowsB=np.concatenate([pool[256:512] for pool in trainpools])
    rowsC=stratified(b.val_x,256,rng,diag)
    start=timer(device)
    grads=[cell_gradients(model,b,rowsA,'train',device),cell_gradients(model,b,rowsB,'train',device),cell_gradients(model,b,rowsC,'val',device)]
    # Fourth block is the actual prediction-function Jacobian at C; first three
    # retain the training derivative for interaction/noise comparisons.
    functional=cell_gradients(model,b,rowsC,'val',device,functional=True)
    elapsed=timer(device)-start;stack=torch.cat([*grads,functional]).double()
    np.savez_compressed(out/'P4_gram.npz',all_gram=(stack@stack.T).cpu().numpy(),
        shared_gram=(stack[:,mask]@stack[:,mask].T).cpu().numpy(),
        A_row_ids=b.train_row_ids[rowsA],B_row_ids=b.train_row_ids[rowsB],C_row_ids=b.val_row_ids[rowsC])
    h=functional/functional.norm(dim=1,keepdim=True).clamp_min(1e-30)
    # P6 holdout moments: per-row full gradient, no optimizer state intervention.
    pdom=np.array([b.train_counts[d] for d in range(3)],float);pdom/=pdom.sum()
    samples={};summaries={};start=timer(device)
    for block,start_index in [('design',512),('heldout',560)]:
        records=[];sums=torch.zeros((3,count),device=device,dtype=torch.float64)
        for d in range(3):
            rows=trainpools[d][start_index:start_index+48]
            for row in rows:
                x=torch.tensor(b.train_x[row:row+1],device=device,dtype=torch.float32).T
                y=torch.tensor(b.train_y[row:row+1],device=device,dtype=torch.float32)
                pred=model(x);loss=F.binary_cross_entropy(pred,y,reduction='mean');g=flat_grad(loss,model)
                sums[d]+=g.double()
                residual=(pred-y).detach().cpu().numpy()[0]
                records.append(dict(domain=d,row_id=int(b.train_row_ids[row]),norm2=float(g.double().square().sum()),
                                    projections=(h.double()@g.double()).cpu().numpy().tolist(),
                                    logit_norm2=float(np.square(residual).sum()/4),task_residual2=np.square(residual).tolist()))
        means=sums/48;population_mean=torch.tensor(pdom,device=device,dtype=torch.float64)@means
        summaries[block]=dict(domain_mean_gram=(means@means.T).cpu().numpy().tolist(),
                              population_mean_norm2=float(population_mean.square().sum()),
                              population_mean_projections=(h.double()@population_mean).cpu().numpy().tolist())
        samples[block]=records
        print(block,'gradients complete',flush=True)
    sampling_seconds=timer(device)-start
    dump(out/'P6_samples.json',samples);dump(out/'P6_moments.json',dict(p=pdom.tolist(),blocks=summaries,n_per_domain=48))
    dump(out/'receipt.json',dict(status='COMPLETE',seed=a.seed,gradient_parameters=count,
        feature_extraction_seconds=extraction_seconds,features_skipped=a.skip_features,
        P4_24_cell_gradients_seconds=elapsed,P4_gradient_calls=24,
        P6_288_row_gradients_seconds=sampling_seconds,P6_gradient_calls=288,
        wall_seconds=time.time()-started,peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2,
        optimizer_steps=0,test_access=False,target_derivative='complete prediction-function Jacobian; routing reconnected with identical values'))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
