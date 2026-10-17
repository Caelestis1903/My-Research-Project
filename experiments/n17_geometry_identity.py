"""Raw full-coordinate N2 fixed-state moments; no scientific conclusion or selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import n17_common as c
from n17_quota import QuotaSampler

def flat_grad(model,b,rows,device,weights=None):
    x=torch.tensor(b.train_x[rows],device=device,dtype=torch.float32).T;y=torch.tensor(b.train_y[rows],device=device,dtype=torch.float32)
    loss=F.binary_cross_entropy(model(x),y,reduction='none').mean(1)
    if weights is not None:loss=loss*weights[x[-1].long()]
    params=list(model.parameters());gg=torch.autograd.grad(loss.mean(),params,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,gg)])

def indices(model,device):
    out={x:[] for x in ('C','P','R','E','C_embedding','C_shared_STAR','C_skip','C_star_mlp','C_experts')};order=[];start=0
    for name,p in model.named_parameters():
        block=c.base.group(name);ix=list(range(start,start+p.numel()));out[block].extend(ix)
        if block=='C':
            sub='C_embedding' if name.startswith('embedding.') else 'C_shared_STAR' if name in ('shared_weight','shared_bias') else 'C_skip' if name.startswith('skip_conn.') else 'C_star_mlp' if name.startswith('star_mlp.') else 'C_experts' if name.startswith('expert.') else None
            assert sub is not None,f'unknown C coordinate {name}';out[sub].extend(ix)
        order.append(dict(name=name,start=start,end=start+p.numel(),block=block));start+=p.numel()
    out={k:torch.tensor(v,device=device,dtype=torch.long) for k,v in out.items()};out['ALL']=torch.arange(start,device=device)
    return out,order

def blocknorm(x,ix):return float(x[ix].double().square().sum())

class Moments:
    def __init__(self,size,device):
        self.n=0;self.mean=torch.zeros(size,device=device,dtype=torch.float64);self.m2=torch.zeros_like(self.mean)
    def add(self,x):
        self.n+=1;delta=x.double()-self.mean;self.mean+=delta/self.n;self.m2+=delta*(x.double()-self.mean)

def frozen_map(gradient,theta,state,optimizer,lr):
    """Use PyTorch's float32 equations on a fixed serialized optimizer state."""
    if optimizer=='sgd':return -lr*(gradient+1e-5*theta)
    # Concatenated state has one common native step; all coordinates are supported.
    g=gradient+1e-5*theta;step=state['step']+1
    m=torch.lerp(state['m'],g,1-.9);v=state['v'].clone().mul_(.999).addcmul_(g,g,value=1-.999)
    # Same native Adam denominator order; this is the pre-parameter-rounding update.
    denom=v.sqrt()/np.sqrt(1-.999**step)+1e-8
    return (-lr/(1-.9**step))*m/denom

def same_graph_identity(model,b,rows,device,weights,coeff,partition_gradient,groups):
    """Check aggregation on one graph; record cross-forward roundoff separately."""
    x=torch.tensor(b.train_x[rows],device=device,dtype=torch.float32).T
    y=torch.tensor(b.train_y[rows],device=device,dtype=torch.float32)
    terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(1)
    params=list(model.parameters());domains=x[-1].long()
    direct_loss=(terms*weights[domains]).mean()
    composed_loss=sum(float(coeff[d])*terms[domains==d].mean() for d in range(3))
    def vector(loss,retain=False):
        gg=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,gg)]).double()
    direct=vector(direct_loss,True);composed=vector(composed_loss)
    residual=float((direct-composed).norm());tolerance=1e-6+1e-4*float(partition_gradient.norm())
    assert residual<=tolerance,'same-graph aggregation identity failed'
    cross=direct-partition_gradient
    return dict(status='PASS_SAME_GRAPH_AGGREGATION',residual=residual,tolerance=tolerance,
        cross_forward_residual_norm=float(cross.norm()),cross_forward_exceeds_original_tolerance=bool(float(cross.norm())>tolerance),
        cross_forward_block_residual_norm={key:math_sqrt(blocknorm(cross,ix)) for key,ix in groups.items()},
        cross_forward_scope='single-draw numerical comparison, not a uniform bank error bound; estimator unchanged')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--anchor-arm',required=True)
    p.add_argument('--device',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--k',type=int,default=512);p.add_argument('--batch-size',type=int,default=40960)
    p.add_argument('--adam-map',action='store_true');p.add_argument('--engineering',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    torch.set_num_threads(2);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    b=c.bundle(seed=a.seed)
    if a.engineering:
        ix=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:n] for d,n in enumerate((96,64,32))])
        for suffix in ('x','y','row_ids'):setattr(b,'train_'+suffix,getattr(b,'train_'+suffix)[ix])
    model=c.base.fixed_model(b,a.seed).to(device)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.train();initial=c.base.initial_state_hash(model)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or (isinstance(m,torch.nn.Dropout) and m.p>0) for m in model.modules())
    groups,order=indices(model,device);size=sum(p.numel() for p in model.parameters());pdom=np.array([b.train_counts[d] for d in range(3)],dtype=float);pdom/=pdom.sum();pools=[np.flatnonzero(b.train_x[:,-1]==d) for d in range(3)]
    exact=torch.zeros((3,size),device=device,dtype=torch.float64)
    for d,pool in enumerate(pools):
        for offset in range(0,len(pool),a.batch_size):
            rows=pool[offset:offset+a.batch_size];exact[d]+=flat_grad(model,b,rows,device).double()*(len(rows)/len(pool))
    pref=torch.tensor(pdom,device=device,dtype=torch.float64);reference=pref@exact
    theta=torch.cat([p.detach().flatten() for p in model.parameters()]);mapstate=None
    if a.adam_map:
        assert a.optimizer=='adam';states=state['optimizer']['state'];ids=state['optimizer']['param_groups'][0]['params'];times=[float(states[i]['step']) for i in ids];assert len(set(times))==1
        mapstate=dict(m=torch.cat([states[i]['exp_avg'].flatten() for i in ids]).to(device),v=torch.cat([states[i]['exp_avg_sq'].flatten() for i in ids]).to(device),step=times[0])
    config={**vars(a),'checkpoint':str(a.checkpoint),'out':str(a.out),'checkpoint_sha':c.sha(a.checkpoint),'epoch':state['epoch'],'p':pdom.tolist(),'parameter_order':order,'source_sha':c.sha(__file__),'quota_source_sha':c.sha(Path(__file__).with_name('n17_quota.py')),'base_sources':c.sources(),'test_read':False}
    config['optimizer_mapping_scope']='float32 native Adam equations before parameter storage rounding; fixed previous-v denominator independent of current draw'
    config['identity_validation_version']='same_graph_v1';config['original_geometry_source_sha']=c.sha(Path(__file__).with_name('n17_geometry.py'))
    c.dump(a.out/'config.json',config);allrows=[];maps=[];identity=None
    for arm,q in [('SN',pdom),('SB',np.ones(3)/3)]:
        for variant in ('IID','quota'):
            # Independent batches and separate variant streams; paired q uses common random source.
            stream=a.seed+state['epoch']*100+500000+(0 if variant=='IID' else 10000)
            sampler=c.base.Sampler(b,q,stream,'iid') if variant=='IID' else QuotaSampler(b,q,stream)
            moments=Moments(size,device);domain_stats={key:[] for key in groups};energies={key:[] for key in groups};inverses=[]
            umom=Moments(size,device) if a.adam_map else None;lmom=Moments(size,device) if a.adam_map else None
            for draw in range(a.k):
                rows=sampler.sample(a.batch_size);ds=b.train_x[rows,-1].astype(int);counts=np.bincount(ds,minlength=3);assert np.all(counts>0)
                parts=torch.stack([flat_grad(model,b,rows[ds==d],device) for d in range(3)])
                coeff=pdom if variant=='quota' else counts/a.batch_size*pdom/q
                gradient=torch.tensor(coeff,device=device,dtype=torch.float64)@parts.double();moments.add(gradient)
                if identity is None:
                    actual_weights=torch.tensor(pdom/q if variant=='IID' else a.batch_size*pdom/counts,device=device,dtype=torch.float32)
                    identity=same_graph_identity(model,b,rows,device,actual_weights,coeff,gradient,groups)
                inverses.append((1/counts).tolist())
                centered=parts.double()-exact
                for block,ix in groups.items():
                    domain_stats[block].append([int(counts[d])*blocknorm(centered[d],ix) for d in range(3)])
                    energies[block].append(blocknorm(gradient-reference,ix))
                if a.adam_map:
                    gf=gradient.float();u=frozen_map(gf,theta,mapstate,'adam',a.lr);umom.add(u)
                    # D depends only on native previous v/clock, never on current draw.
                    g=gf+1e-5*theta;denom=torch.sqrt(mapstate['v']/(1-.999**max(mapstate['step'],1)))+1e-8
                    lin=-a.lr*torch.lerp(mapstate['m'],g,.1)/(1-.9**(mapstate['step']+1))/denom;lmom.add(lin)
                c.base.csv_append(a.out/'draw_counts.csv',[dict(arm=arm,variant=variant,draw=draw,n=a.batch_size,d0=int(counts[0]),d1=int(counts[1]),d2=int(counts[2]),row_ids_sha=__import__('hashlib').sha256(b.train_row_ids[rows].astype(np.int64).tobytes()).hexdigest(),**{'energy_'+block:energies[block][-1] for block in groups})])
                if draw%32==0:c.dump(a.out/'progress.json',dict(status='RUNNING',arm=arm,variant=variant,draw=draw+1,elapsed_seconds=time.time()-started))
            inv=np.mean(inverses,axis=0)
            for block,ix in groups.items():
                observations=np.asarray(domain_stats[block]);within_domain=observations.mean(0);stderr=observations.std(0,ddof=1)/np.sqrt(a.k)
                between=float((sum((pdom[d]**2/q[d])*exact[d,ix].square().sum() for d in range(3))-reference[ix].square().sum())/a.batch_size) if variant=='IID' else 0.
                within=float(np.dot(pdom**2/q,within_domain)/a.batch_size) if variant=='IID' else float(np.dot(pdom**2*inv,within_domain))
                trace=float(moments.m2[ix].sum()/(a.k-1));bias=float((moments.mean[ix]-reference[ix]).norm())
                allrows.append(dict(seed=a.seed,optimizer=a.optimizer,anchor_arm=a.anchor_arm,epoch=state['epoch'],arm=arm,variant=variant,block=block,coordinates=len(ix),k=a.k,batch_size=a.batch_size,empirical_covariance_trace=trace,reference_centered_energy_trace=float(np.mean(energies[block])),reference_energy_trace_mc_se=float(np.std(energies[block],ddof=1)/np.sqrt(a.k)),within_trace_prediction=within,between_trace_prediction=between,sum_trace_prediction=within+between,mean_bias_norm=bias,mc_rms_se=math_sqrt(trace/a.k),domain_covariance_trace_estimates=within_domain.tolist(),domain_trace_mc_se=stderr.tolist(),mean_inverse_counts=inv.tolist()))
                if a.adam_map:
                    uofmean=frozen_map(moments.mean.float(),theta,mapstate,'adam',a.lr)
                    maps.append(dict(seed=a.seed,anchor_arm=a.anchor_arm,epoch=state['epoch'],arm=arm,variant=variant,block=block,mean_gradient_norm=float(moments.mean[ix].norm()),mean_update_norm=float(umom.mean[ix].norm()),jensen_residual_norm=float((umom.mean[ix]-uofmean[ix]).norm()),mean_linear_update_norm=float(lmom.mean[ix].norm()),update_mc_rms_se=math_sqrt(float(umom.m2[ix].sum()/(a.k-1))/a.k)))
            np.savez_compressed(a.out/f'{arm}_{variant}_moments.npz',mean=moments.mean.cpu().numpy(),m2=moments.m2.cpu().numpy(),mean_update=umom.mean.cpu().numpy() if umom is not None else np.array([]),mean_linear_update=lmom.mean.cpu().numpy() if lmom is not None else np.array([]))
            print(arm,variant,'complete',a.k,'draws',time.time()-started,flush=True)
    assert c.base.initial_state_hash(model)==initial
    assert config['source_sha']==c.sha(__file__) and config['base_sources']==c.sources()
    assert config['quota_source_sha']==c.sha(Path(__file__).with_name('n17_quota.py'))
    np.savez_compressed(a.out/'exact_domain_means.npz',domain_gradients=exact.cpu().numpy(),reference=reference.cpu().numpy())
    c.dump(a.out/'geometry.json',allrows);c.dump(a.out/'adam_mapping.json',maps);c.dump(a.out/'receipt.json',dict(status='COMPLETE_ENGINEERING' if a.engineering else 'COMPLETE',identity=identity,k=a.k,epoch=state['epoch'],model_unchanged=True,wall_seconds=time.time()-started,source_sha=c.sha(__file__),config_sha=c.sha(a.out/'config.json')))

def math_sqrt(x):return float(np.sqrt(max(0.,x)))

if __name__=='__main__':main()
