"""Whole natural epochs, RAM-only selections, coefficient replay and scalar records."""
import copy
import hashlib
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .data import load_bundle,model_for,DEFAULT_CACHE
from .group_registry import build_path_groups
from .optimizer import PUBAdam,losses,gradients,candidate,CELLS
from .solver import CCPSolver
from .metrics import metrics

WD={'movielens':1e-4,'kuairand':1e-5}
RHO={'movielens':.9828205985452511,'kuairand':.995}

def evaluate(model,bundle,split):
    if split=='test' and not hasattr(bundle,'test_y'): raise ValueError('Test not loaded before selection freeze')
    x,y=getattr(bundle,split+'_x'),getattr(bundle,split+'_y')
    rng=rng_state(); was=model.training; model.eval(); pred=[]
    try:
        with torch.no_grad():
            for i in range(0,len(x),BATCH): pred.append(model(torch.as_tensor(np.array(x[i:i+BATCH]),device=next(model.parameters()).device).T).cpu().numpy())
    finally: model.train(was); restore_rng(rng)
    return metrics(y,np.concatenate(pred),x[:,-1])

def consider(selection,epoch,values,state):
    if selection is None or values['macro_auc']>selection['metrics']['macro_auc']:
        return dict(epoch=epoch,metrics=values,state={k:v.clone() for k,v in state.items()},state_hash=state_hash(state))
    return selection

def job_path(out,phase,dataset,arm,seed,frequency=10):
    suffix=f'_f{frequency}' if phase=='pilot' else ''
    return Path(out)/phase/f'{dataset}_{arm}_seed{seed}{suffix}'

def coefficient_path(out,job):
    return job_path(out,'formal',job['dataset'],'C'+job['arm'][1],job['seed'])/'ALPHA.jsonl'

def short_step_probe(model,optimizer,x,y,grads,active,updates):
    """Use only existing natural training rows; diagnostics do not alter the trajectory."""
    idx=optimizer.indices['S']; old=optimizer.layout.flatten(model.parameters()); old_state=cpu_state(model); rng=rng_state()
    members=optimizer.members('S',active); scratch=copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        def cell_loss(net):
            b=F.binary_cross_entropy(net(x),y,reduction='none')
            return [float(b[x[-1]==d,t].mean()) if bool((x[-1]==d).any()) else None for d,t in CELLS]
        before=cell_loss(model); rows=[]
        for identity,direction in [('combined',updates['S'])]+[(str(i),candidate(grads[i,idx],optimizer.m[idx],optimizer.v[idx],optimizer.step+1)) for i in members]:
            vector=old.clone(); vector[idx]-=optimizer.lr*direction
            optimizer.layout.copy_to(scratch,vector); after=cell_loss(scratch)
            rows.append(dict(candidate=identity,before=before,after=after,loss_change=[a-b if a is not None else None for a,b in zip(after,before)]))
    model.load_state_dict(old_state); restore_rng(rng)
    return rows

def run_job(job,out,device='cuda:0',cache_root=DEFAULT_CACHE):
    setup(job['seed'],device); directory=job_path(out,job['phase'],job['dataset'],job['arm'],job['seed'],job.get('frequency',10)); directory.mkdir(parents=True,exist_ok=True)
    started=time.time(); identity=dict(job=job,implementation_hash=implementation_hash()); key=digest(identity)
    if (directory/'RESULT.json').exists():
        result=read_json(directory/'RESULT.json')
        if result['identity_hash']!=key: raise ValueError('Completed job identity differs')
        return result
    # Restart an interrupted job from the seed; no training state is written to disk.
    attempt=len(list(directory.glob('ATTEMPT*.json')))+1
    if attempt>1:
        for p in directory.glob('*.jsonl'): p.rename(p.with_suffix(f'.attempt{attempt-1}.jsonl'))
    write_json(directory/f'ATTEMPT{attempt}.json',dict(started=started,identity=identity,device=device))
    bundle=load_bundle(job['dataset'],cache_root,False); model=model_for(bundle).to(device); registry=build_path_groups(model,bundle); layout=registry.layout
    init_hash=state_hash(model.state_dict()); loader=torch.Generator().manual_seed(job['seed']+11000)
    arm=job['arm']; topology=int(arm[-1]) if arm[0] in ('C','G','E') else 1
    native=arm=='A'; ema_enabled=native or arm.startswith('E')
    if native: opt=torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=WD[job['dataset']],foreach=False,fused=False,amsgrad=False)
    else: opt=PUBAdam(model,registry,topology,'G' if arm.startswith('G') else 'PUB',weight_decay=WD[job['dataset']],frequency=job['frequency'],solver=CCPSolver(**job['solver']))
    ema=copy.deepcopy(model).requires_grad_(False) if ema_enabled else None
    kinds=['A','F'] if native else [arm]+([arm+'F'] if ema_enabled else [])
    selections={k:None for k in kinds}; anchors={k:{} for k in kinds}; step=0; samples=0; train_seconds=0.; validation_seconds=0.; fallback_count=0; group_count=0
    replay=[]
    if arm.startswith('G'):
        replay=[__import__('json').loads(line) for line in coefficient_path(out,job).read_text().splitlines()]
    alpha_hash=hashlib.sha256(); order_hash=hashlib.sha256()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,job['epochs']+1):
        model.train(); order=torch.randperm(len(bundle.train_y),generator=loader); epoch_order=hashlib.sha256(order.numpy().tobytes()).hexdigest(); order_hash.update(order.numpy().tobytes())
        weighted_loss=0.; epoch_start=time.perf_counter()
        for offset in range(0,len(order),BATCH):
            indices=order[offset:offset+BATCH].numpy(); x=torch.as_tensor(np.array(bundle.train_x[indices]),device=device).T; y=torch.as_tensor(np.array(bundle.train_y[indices]),device=device)
            step+=1; samples+=len(y)
            if native:
                opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy(model(x),y,reduction='none').mean(1).mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); opt.step()
            else:
                terms,loss,active=losses(model,x,y,topology)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                gs=gradients(terms,model)
                do_probe=offset==0
                old_flat=layout.flatten(model.parameters()) if do_probe else None
                old_opt=opt.state_dict() if do_probe else None
                coefficients,diags,updates=opt.apply(gs,active,replay=replay[step-1]['groups'] if replay else None,diagnostics=step%10==1,retain_updates=do_probe)
                if replay and (replay[step-1]['step']!=step or replay[step-1]['active_cells']!=active): raise ValueError('Replay step identity changed')
                alpha_hash.update(digest(coefficients).encode()); append_json(directory/'ALPHA.jsonl',coefficients)
                fallback_count+=sum(v['fallback'] for v in coefficients['groups'].values()); group_count+=sum(len(v['members'])>1 for v in coefficients['groups'].values())
                if diags: append_json(directory/'DIAGNOSTICS.jsonl',dict(epoch=epoch,step=step,extra_autograd=len(terms)-1,groups=diags))
                if do_probe:
                    after_flat=layout.flatten(model.parameters()); after_opt=opt.state_dict(); layout.copy_to(model,old_flat); opt.load_state_dict(old_opt)
                    probe=short_step_probe(model,opt,x[:,:1024],y[:1024],gs,active,updates)
                    layout.copy_to(model,after_flat); opt.load_state_dict(after_opt)
                    append_json(directory/'SHORT_STEP.jsonl',dict(epoch=epoch,step=step,groups='S',observations=probe))
            weighted_loss+=float(loss.detach())*len(y)
            if ema is not None:
                with torch.no_grad():
                    for p,q in zip(ema.parameters(),model.parameters()): p.mul_(RHO[job['dataset']]).add_(q,alpha=1-RHO[job['dataset']])
                    for p,q in zip(ema.buffers(),model.buffers()): p.copy_(q)
        torch.cuda.synchronize(device); epoch_train=time.perf_counter()-epoch_start; train_seconds+=epoch_train; eval_start=time.perf_counter()
        for kind in kinds:
            net=ema if kind=='F' or kind.endswith('F') else model
            values=evaluate(net,bundle,'val'); state=cpu_state(net)
            selections[kind]=consider(selections[kind],epoch,values,state)
            if epoch in (10,50): anchors[kind]['E'+str(epoch)]=dict(epoch=epoch,metrics=values,state=state,state_hash=state_hash(state))
            append_json(directory/'VALIDATION.jsonl',dict(kind=kind,epoch=epoch,step=step,samples=samples,init_hash=init_hash,order_hash=epoch_order,train_loss=weighted_loss/len(order),train_seconds=epoch_train,**values))
        validation_seconds+=time.perf_counter()-eval_start
        print(__import__('json').dumps(dict(dataset=job['dataset'],arm=arm,seed=job['seed'],epoch=epoch,step=step,loss=weighted_loss/len(order))),flush=True)
    if replay and len(replay)!=step: raise ValueError('Replay length differs from training')
    if replay:
        source=read_json(coefficient_path(out,job).parent/'RESULT.json')
        if source['alpha_trace_hash']!=alpha_hash.hexdigest() or source['order_hash']!=order_hash.hexdigest() or source['init_hash']!=init_hash: raise ValueError('Control trace/input identity differs')
    selected_metadata={kind:{k:v for k,v in row.items() if k!='state'} for kind,row in selections.items()}
    frozen=dict(identity_hash=key,implementation_hash=identity['implementation_hash'],job=job,init_hash=init_hash,order_hash=order_hash.hexdigest(),selections=selected_metadata,alpha_trace_hash=alpha_hash.hexdigest(),steps=step,samples=samples)
    write_json(directory/'SELECTION_FROZEN.json',frozen)
    endpoints=[]
    for kind,row in selections.items():
        for anchor,record in [('selected',row)]+list(anchors[kind].items()):
            endpoints.append(dict(dataset=job['dataset'],seed=job['seed'],kind=kind,anchor=anchor,epoch=record['epoch'],state_hash=record['state_hash'],split='validation',metrics=record['metrics']))
    if job['phase']!='pilot':
        testbundle=load_bundle(job['dataset'],cache_root,True)
        for kind,row in selections.items():
            for anchor,record in [('selected',row)]+list(anchors[kind].items()):
                model.load_state_dict(record['state'])
                endpoints.append(dict(dataset=job['dataset'],seed=job['seed'],kind=kind,anchor=anchor,epoch=record['epoch'],state_hash=record['state_hash'],split='test',metrics=evaluate(model,testbundle,'test')))
    cost=dict(wall_seconds=time.time()-started,train_seconds=train_seconds,validation_seconds=validation_seconds,peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,steps=step,samples=samples,rows_per_training_second=samples/train_seconds,fallback_groups=fallback_count,balanced_group_steps=group_count)
    result=dict(**frozen,status='COMPLETE',endpoints=endpoints,cost=cost,last_validation={k:read_last_validation(directory,k) for k in kinds})
    write_json(directory/'RESULT.json',result); return result

def read_last_validation(directory,kind):
    rows=[__import__('json').loads(line) for line in (directory/'VALIDATION.jsonl').read_text().splitlines()]
    return next(r for r in reversed(rows) if r['kind']==kind)
