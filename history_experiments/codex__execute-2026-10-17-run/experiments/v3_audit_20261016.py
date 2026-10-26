"""V0/V1 raw exact empirical-gradient and production-batch estimator audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import v2_20261016 as base

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16_v3/execution_20261016'
OLD=Path('/home/xzf/Code/phase0-3-20261016-git/research_plans/2026-10-16_v2/execution_20261016')
SEEDS=(2022,2023,2024,2025)

def bundle(dataset,seed):
    path=OUT/'local_cache'/f'{dataset}_{seed}.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    b=base.load_stage_s1_bundle(dataset,None,'natural' if dataset=='kuairand' else 'd2_large',seed)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(b,f,protocol=5)
    tmp.replace(path)
    return b

def frozen_split(b,dataset,seed):
    # Joint-label stratification prevents avoidable empty cells, with row-id hash ordering.
    chosen=[]
    domains=b.val_x[:,-1].astype(int);joint=b.val_y[:,0].astype(int)*2+b.val_y[:,1].astype(int)
    ids=b.val_row_ids.astype(np.int64)
    hashes=np.array([int.from_bytes(hashlib.sha256(f'20261016:{int(i)}'.encode()).digest()[:8],'little') for i in ids],dtype=np.uint64)
    for d in range(3):
        for label in range(4):
            rows=np.flatnonzero((domains==d)&(joint==label))
            chosen.extend(rows[np.argsort(hashes[rows],kind='stable')[:(len(rows)+1)//2]])
    selection=np.sort(chosen);diagnostic=np.setdiff1d(np.arange(len(ids)),selection)
    path=OUT/'splits'/f'{dataset}_seed{seed}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        old=np.load(path);assert np.array_equal(old['selection'],selection) and np.array_equal(old['diagnostic'],diagnostic)
    else:np.savez_compressed(path,selection=selection,diagnostic=diagnostic,row_ids=ids)
    base.dump(path.with_suffix('.json'),dict(dataset=dataset,seed=seed,file_sha256=base.sha(path),
        selection_count=len(selection),diagnostic_count=len(diagnostic),history_exposed=True,
        split_rule='fixed sha256 row_id ordering within domain x joint label, half selection',
        data=b.manifest_fields(),frozen_before_v3_training=True))
    return selection,diagnostic

def grad(model,b,rows,weights,device):
    model.zero_grad(set_to_none=True)
    x=torch.as_tensor(b.train_x[rows],device=device,dtype=torch.float32).T
    y=torch.as_tensor(b.train_y[rows],device=device,dtype=torch.float32)
    terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(1)
    loss=(terms*weights[x[-1].long()]).mean()
    loss.backward()
    values=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in model.parameters()])
    assert torch.isfinite(values).all() and torch.isfinite(loss)
    return values,float(loss.detach())

def exact(model,b,pools,device,batch_size):
    size=sum(p.numel() for p in model.parameters());out=np.zeros((3,size),dtype=np.float64)
    losses=[];ones=torch.ones(3,device=device)
    for d,pool in enumerate(pools):
        total=0.
        for start in range(0,len(pool),batch_size):
            rows=pool[start:start+batch_size];g,l=grad(model,b,rows,ones,device)
            out[d]+=g.cpu().numpy().astype(np.float64)*(len(rows)/len(pool));total+=l*len(rows)/len(pool)
        losses.append(total)
    return out,losses

def block_indices(model):
    groups={k:[] for k in 'CPRE'};start=0;order=[]
    for n,p in model.named_parameters():
        end=start+p.numel();groups[base.group(n)].extend(range(start,end));order.append(dict(name=n,shape=list(p.shape),start=start,end=end,block=base.group(n)));start=end
    return {k:np.asarray(v) for k,v in groups.items()},order

def geometry(matrix,reference,groups,device,seed):
    k,size=matrix.shape;mean=np.mean(matrix,axis=0,dtype=np.float64)
    rng=np.random.default_rng(seed);counts=rng.multinomial(k,np.ones(k)/k,size=2000)/k
    allgram=np.zeros((k,k),dtype=np.float64);rows=[]
    for block,ix in groups.items():
        gram=torch.zeros((k,k),device=device,dtype=torch.float64)
        for start in range(0,len(ix),8192):
            cols=ix[start:start+8192]
            x=torch.tensor(np.asarray(matrix[:,cols]),device=device,dtype=torch.float64)
            x-=torch.as_tensor(mean[cols],device=device)
            gram.addmm_(x,x.T)
        gc=gram.cpu().numpy();allgram+=gc
        rows.append(geometry_row(block,gc,mean[ix],reference[ix],counts))
    rows.append(geometry_row('ALL',allgram,mean,reference,counts))
    for row in rows:row.update(k=k,coordinates=size if row['block']=='ALL' else len(groups[row['block']]),rank_upper_bound=k-1,covariance_scope='full coordinates, empirical K-batch covariance; spectrum from exact Gram')
    return rows,mean,allgram

def geometry_row(block,gram,mean,reference,counts):
    k=len(gram);eigs=np.maximum(np.linalg.eigvalsh(gram/(k-1)),0)[::-1]
    trace=float(np.trace(gram)/(k-1));bias=float(np.linalg.norm(mean-reference));se=math.sqrt(trace/k)
    # Distribution-free bound (estimated covariance); conservative, family-wise scope preregistered.
    alpha=.001/(16*2*5);bound=se/math.sqrt(alpha)+1e-6*max(1,float(np.linalg.norm(reference)))
    probs=eigs[eigs>0]/max(trace,1e-300)
    boot=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',counts,gram,counts,optimize=True),0))
    norm_product=np.sqrt(np.maximum(np.diag(gram)[:,None]*np.diag(gram)[None,:],0))
    row=dict(block=block,bias_norm=bias,reference_norm=float(np.linalg.norm(reference)),mean_norm=float(np.linalg.norm(mean)),
        mean_reference_cosine=float(np.dot(mean,reference)/(max(np.linalg.norm(mean)*np.linalg.norm(reference),1e-300))),
        covariance_trace=trace,lambda_max=float(eigs[0]),effective_rank=float(np.exp(-np.sum(probs*np.log(probs)))) if len(probs) else 0.,
        anisotropy=float(eigs[0]/trace) if trace else 0.,top10_explained=float(eigs[:10].sum()/trace) if trace else 0.,
        mc_rms_se=se,mc_conservative_bound=bound,mc_alpha_per_comparison=alpha,mc_norm_bootstrap95=float(np.quantile(boot,.95)),
        bias_within_uncertainty=bool(bias<=bound),eigenvalues=eigs.tolist())
    for lag in (1,2,5):
        den=np.diag(norm_product,k=lag);row[f'centered_lag{lag}_cosine']=float(np.mean(np.diag(gram,k=lag)/np.maximum(den,1e-300))) if lag<k else None
    return row

def audit(args):
    directory=args.run_dir;directory.mkdir(parents=True,exist_ok=False);started=time.time()
    device=torch.device(args.device);torch.set_num_threads(2)
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    b=bundle('kuairand',args.seed);model=base.fixed_model(b,args.seed).to(device);model.train()
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or (isinstance(m,torch.nn.Dropout) and m.p>0) for m in model.modules())
    source=OLD/f'runs/kuairand/{args.optimizer}/seed{args.seed}/SN/attempt1/epoch{args.epoch:02d}.pt'
    state=torch.load(source,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
    init=base.initial_state_hash(model);groups,order=block_indices(model)
    pools=[np.flatnonzero(b.train_x[:,-1]==d) for d in range(3)]
    p=np.array([len(x) for x in pools],dtype=float);p/=p.sum()
    base.dump(directory/'config.json',dict(seed=args.seed,optimizer=args.optimizer,epoch=args.epoch,k=args.k,batch_size=args.batch_size,
        source_checkpoint=str(source),source_checkpoint_sha=base.sha(source),source_code=base.sources(),audit_code_sha=base.sha(__file__),
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),p=p.tolist(),data=b.manifest_fields(),parameter_order=order))
    gd,losses=exact(model,b,pools,device,args.batch_size);ref=p@gd
    np.savez(directory/'exact.npz',domain_gradients=gd,reference=ref,domain_losses=losses)
    # Independent global-row sweep verifies decomposition in the production autograd path.
    direct=np.zeros_like(ref);n=len(b.train_x);ones=torch.ones(3,device=device)
    for start in range(0,n,args.batch_size):
        rows=np.arange(start,min(start+args.batch_size,n));g,_=grad(model,b,rows,ones,device);direct+=g.cpu().numpy().astype(float)*len(rows)/n
    residual=float(np.linalg.norm(direct-ref));identity_ok=residual<=1e-6+1e-4*np.linalg.norm(ref)
    numerical=[]
    for arm,q in [('SN',p),('SB',np.ones(3)/3)]:
        sampler=base.Sampler(b,q,args.seed+args.epoch*100+100000,'iid')
        weights=torch.as_tensor(p/q,device=device,dtype=torch.float32)
        assert np.allclose(q*(p/q),p,atol=1e-15,rtol=0)
        matrix=np.lib.format.open_memmap(directory/f'{arm}_gradients.npy',mode='w+',dtype=np.float32,shape=(args.k,len(ref)))
        for index in range(args.k):
            rows=sampler.sample(args.batch_size);g,loss=grad(model,b,rows,weights,device);matrix[index]=g.cpu().numpy()
            counts=np.bincount(b.train_x[rows,-1].astype(int),minlength=3)
            norm=float(g.double().norm());cos=float(torch.dot(g.double(),torch.as_tensor(ref,device=device))/(g.double().norm()*np.linalg.norm(ref)+1e-300))
            base.csv_append(directory/'batches.csv',[dict(arm=arm,batch=index,n=len(rows),d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),
                row_hash=hashlib.sha256(np.asarray(b.train_row_ids[rows],dtype=np.int64).tobytes()).hexdigest(),loss=loss,norm=norm,reference_cosine=cos)])
            if index%32==0:
                matrix.flush();base.dump(directory/'progress.json',dict(status='RUNNING',arm=arm,batches_completed=index+1,elapsed_seconds=time.time()-started))
        matrix.flush();result,mean,gram=geometry(matrix,ref,groups,device,args.seed+args.epoch)
        for row in result:row.update(arm=arm,seed=args.seed,optimizer=args.optimizer,epoch=args.epoch)
        numerical.extend(result);np.savez(directory/f'{arm}_summary.npz',mean=mean,gram=gram)
    assert base.initial_state_hash(model)==init
    base.dump(directory/'geometry.json',numerical)
    status='PASS' if identity_ok and all(x['bias_within_uncertainty'] for x in numerical) else 'FAIL_OBJECTIVE_AUDIT'
    base.dump(directory/'receipt.json',dict(status=status,exact_identity_residual=residual,exact_identity_pass=bool(identity_ok),
        elapsed_seconds=time.time()-started,model_unchanged=True,k=args.k,full_coordinates=len(ref),
        peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else 0,
        artifact_hashes={x.name:base.sha(x) for x in directory.iterdir() if x.is_file() and x.name!='receipt.json'}))
    base.dump(directory/'progress.json',dict(status=status,elapsed_seconds=time.time()-started))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['freeze','audit'],default='audit')
    parser.add_argument('--seed',type=int,default=2022);parser.add_argument('--optimizer',default='adam');parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--k',type=int,default=512);parser.add_argument('--batch-size',type=int,default=40960)
    parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    try:
        if args.mode=='freeze':
            for dataset in ('kuairand','movielens'):
                for seed in SEEDS:
                    b=bundle(dataset,seed);frozen_split(b,dataset,seed);print(dataset,seed,'FROZEN',flush=True)
        else:audit(args)
    except Exception:
        if args.run_dir:
            args.run_dir.mkdir(parents=True,exist_ok=True);base.dump(args.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
