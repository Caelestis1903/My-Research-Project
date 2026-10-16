"""P3/P4/P6 immutable-state readouts; no model optimization."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from v2_20261016 import OUT,COMMON,sha,dump,fixed_model,group
from stage_S1_data import load_stage_s1_bundle


def differences():
    records=[]
    for configfile in sorted((OUT/'runs').glob('*/*/seed*/SN/attempt1/config.json')):
        nr=configfile.parent;br=nr.parent.parent/'SB'/'attempt1'
        if not (nr/'run_manifest.json').exists() or not (br/'run_manifest.json').exists():continue
        c=json.loads(configfile.read_text());names=[n for ns in c['partitions'].values() for n in ns]
        initial=torch.load(nr/'epoch00.pt',map_location='cpu',weights_only=False)
        for key in ['epoch00']+[f'epoch{e:02d}' for e in COMMON]+['selected_state']:
            files=[r/(key+'.pt') for r in (nr,br)]
            if not all(p.exists() for p in files):continue
            one,two=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
            assert one['optimizer_parameter_names']==two['optimizer_parameter_names']
            oname={n:i for i,n in enumerate(one['optimizer_parameter_names'])}
            for block,ns in c['partitions'].items():
                num=sum(one['model'][n].numel() for n in ns)
                distance=math.sqrt(sum(float((one['model'][n].double()-two['model'][n].double()).square().sum()) for n in ns))
                disp=[math.sqrt(sum(float((s['model'][n].double()-initial['model'][n].double()).square().sum()) for n in ns)) for s in (one,two)]
                row=dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],seed=c['seed'],snapshot=key,
                    epoch_n=one['epoch'],epoch_b=two['epoch'],block=block,dimension=num,theta_distance=distance,
                    distance_per_sqrt_dimension=distance/math.sqrt(num),natural_displacement=disp[0],balanced_displacement=disp[1],
                    distance_over_mean_displacement=distance/(sum(disp)/2) if sum(disp)>0 else 0)
                for moment in ('exp_avg','exp_avg_sq'):
                    if c['optimizer']=='adam':
                        row[moment+'_distance']=math.sqrt(sum(float((one['optimizer']['state'][oname[n]][moment].double()-two['optimizer']['state'][oname[n]][moment].double()).square().sum()) for n in ns)) if key!='epoch00' else 0
                if key=='epoch00':assert distance==0
                records.append(row)
    target=OUT/'analysis';target.mkdir(exist_ok=True);pd.DataFrame(records).to_csv(target/'block_state_differences.csv',index=False)
    print('state difference rows',len(records))


def probe(run,device='cpu'):
    config=json.loads((run/'config.json').read_text());seed=config['seed'];dataset=config['dataset']
    target=OUT/'probes'/dataset/config['optimizer']/f'seed{seed}';target.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(971)
    b=load_stage_s1_bundle(dataset,None,'natural' if dataset=='kuairand' else 'd2_large',seed)
    model=fixed_model(b,seed).to(device);names=list(dict(model.named_parameters()));params=list(model.parameters())
    p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum()
    rng=np.random.default_rng(seed*1000+811);rows=[]
    for d in range(3):
        pool=np.flatnonzero(b.train_x[:,-1]==d);rows.append(rng.choice(pool,8*32,replace=False).reshape(8,32))
        assert len(np.unique(rows[-1]))==256
    pools=np.asarray(rows);np.savez_compressed(target/'pool_rows.npz',indices=pools,row_ids=b.train_row_ids[pools])
    blocks={g:[i for i,n in enumerate(names) if group(n)==g] for g in ('C','P','R','E')}
    projections={};selectrng=np.random.default_rng(971)
    for g,ii in blocks.items():
        dim=sum(params[i].numel() for i in ii);projections[g]=np.sort(selectrng.choice(dim,min(256,dim),replace=False))
    np.savez_compressed(target/'projection_coordinates.npz',**projections)
    geometry=[];reach=[];cosines=[];spectrum=[]
    snapshots=[('initial',run/'epoch00.pt'),('selected',run/'selected_state.pt'),('final',run/'epoch50.pt')]
    for key,path in snapshots:
        state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);model.eval()
        vectors={g:np.zeros((3,8,len(projections[g])),dtype=float) for g in blocks};task_vectors={g:np.zeros((3,2,len(projections[g])),dtype=float) for g in blocks}
        for d in range(3):
            for j in range(8):
                chosen=pools[d,j];x=torch.tensor(b.train_x[chosen],device=device).T;y=torch.tensor(b.train_y[chosen],device=device)
                losses=F.binary_cross_entropy(model(x),y,reduction='none').mean(0)
                taskgrads=[]
                for task in range(2):
                    grad=torch.autograd.grad(losses[task],params,retain_graph=task==0,allow_unused=True)
                    projected={}
                    for g,ii in blocks.items():
                        vec=torch.cat([(grad[i] if grad[i] is not None else torch.zeros_like(params[i])).detach().flatten() for i in ii])
                        proj=vec[torch.tensor(projections[g],device=device)].double().cpu().numpy();projected[g]=proj
                        task_vectors[g][d,task]+=proj/8
                        if j==0:
                            for i in ii:
                                reach.append(dict(snapshot=key,domain=d,task=task,block=g,parameter=names[i],
                                    autograd_used=grad[i] is not None,norm=float(grad[i].double().norm()) if grad[i] is not None else 0,
                                    parameter_count=params[i].numel()))
                    taskgrads.append(projected)
                for g in blocks:vectors[g][d,j]=(taskgrads[0][g]+taskgrads[1][g])/2
        np.savez_compressed(target/(key+'_gradient_pool.npz'),**vectors)
        for g,pool in vectors.items():
            alpha=p;true_mean=sum(alpha[d]*pool[d].mean(0) for d in range(3))
            for policy,q in [('N',p),('B',np.ones(3)/3)]:
                values=np.concatenate([pool[d]*(alpha[d]/q[d]) for d in range(3)])
                probs=np.repeat(q/8,8);mean=probs@values
                error=float(np.max(np.abs(mean-true_mean)));assert error<1e-10
                centered=values-mean;cov=(centered.T*probs)@centered
                eig=np.linalg.eigvalsh(cov);assert eig.min()>-1e-10;positive=np.maximum(eig,0);trace=positive.sum()
                weights=positive[positive>0]/trace if trace>0 else np.array([1.])
                rank=float(np.exp(-(weights*np.log(weights)).sum()))
                geometry.append(dict(snapshot=key,epoch=state['epoch'],block=g,policy=policy,projected_dimension=len(mean),
                    mean_error=error,mean_norm=float(np.linalg.norm(mean)),covariance_trace=float(trace),
                    batch40960_trace=float(trace/40960),effective_rank=rank,top_eigenvalue_share=float(positive[-1]/trace) if trace>0 else 0))
                spectrum.extend(dict(snapshot=key,block=g,policy=policy,index=i,eigenvalue=float(e)) for i,e in enumerate(eig))
            tasks=task_vectors[g].reshape(6,-1)
            for i in range(6):
                for j in range(6):
                    denom=np.linalg.norm(tasks[i])*np.linalg.norm(tasks[j]);value=float(tasks[i]@tasks[j]/denom) if denom else None
                    cosines.append(dict(snapshot=key,block=g,domain_i=i//2,task_i=i%2,domain_j=j//2,task_j=j%2,cosine=value))
        dump(target/(key+'_receipt.json'),dict(state_sha256=sha(path),state_path=str(path),epoch=state['epoch'],
            matched_theta=True,finite_pool=True,units='32-row domain microbatch gradients; not individual example covariance',
            inference_limit='coordinate projection and finite pool, no full-data covariance or AUC mediation claim'))
    for name,data in [('geometry',geometry),('reachability',reach),('cosines',cosines),('covariance_spectrum',spectrum)]:
        pd.DataFrame(data).to_csv(target/(name+'.csv'),index=False)
    dump(target/'receipt.json',dict(status='COMPLETE',dataset=dataset,seed=seed,optimizer=config['optimizer'],
        source_worker_sha256=config['source_hashes']['experiments/v2_20261016.py'],probe_sha256=sha(__file__),
        pool_rows_sha256=sha(target/'pool_rows.npz'),all_six_cells=True,mean_identity=True,psd=True))
    print('probe complete',str(target))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--differences',action='store_true');args=p.parse_args()
    if args.differences:differences()
    else:probe(args.run,args.device)
