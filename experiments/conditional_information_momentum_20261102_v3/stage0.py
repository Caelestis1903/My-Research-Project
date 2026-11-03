"""Required real-data acceptance; no formal trajectory or test evaluation."""
import copy
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import (load_protocol,load_dataset,make_model,config,make_optimizer,setup,
                       state_digest,cpu_copy,equal,write_json)
from .boundary_sampler import forward_logits
from .objectives import train_step
from .vendor.ema02.objectives import train_step as source_step
from .vendor.ema02.protocol import make_optimizer as source_optimizer
from .guardian import ensure_guardian,request


def stage0(attempt,device):
    start=time.perf_counter();attempt=Path(attempt);attempt.mkdir(parents=True,exist_ok=True)
    p=load_protocol();records=[];manifests={};peak=0
    ensure_guardian(attempt)
    for dataset in p['datasets']:
        bundle,manifest=load_dataset(dataset,p=p);manifests[dataset]=manifest
        write_json(attempt/'data'/f'{dataset}.json',manifest)
        ids=torch.randperm(len(bundle.train_y),generator=torch.Generator().manual_seed(2022+11000)).numpy()[:40960]
        x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device);rows=bundle.train_row_ids[ids]
        context=dict(study='EMA02V2',dataset=dataset,split_id='original',seed=2022,epoch=1,global_step=1)
        setup(2022,device);m=make_model(bundle,p['datasets'][dataset]['source']).to(device)
        a=copy.deepcopy(m);e=copy.deepcopy(m).requires_grad_(False);ae=copy.deepcopy(e)
        c=config('P11',dataset,0.);opt=make_optimizer(m,c);ao=source_optimizer(a,c)
        prob,z=forward_logits(m,x);maxerr=float((prob-z.sigmoid()).abs().max().detach());assert maxerr<=1e-6
        del prob,z
        for step in (1,2):
            context['global_step']=step
            va=train_step(m,opt,e,x,y,rows,c,context)
            vb=source_step(a,ao,ae,x,y,rows,dict(c,arm='A0'),context)
            assert va['loss']==vb['loss'] and equal(m.state_dict(),a.state_dict()) and equal(e.state_dict(),ae.state_dict())
            assert all(equal(pp.grad,qq.grad) for pp,qq in zip(m.parameters(),a.parameters()))
            if step==1:
                # Real RAM IPC restores all baseline state for the second step.
                snap=request(attempt,'roundtrip',value=cpu_copy(dict(m=m.state_dict(),e=e.state_dict(),o=opt.state_dict())))['value']
                m.load_state_dict(snap['m']);e.load_state_dict(snap['e']);opt.load_state_dict(snap['o'])
        records.append(dict(dataset=dataset,arm='lambda0',rows=len(y),steps=2,source_equal=True,probability_max_abs_error=maxerr))
        del m,a,e,ae,opt,ao,snap,va,vb;gc.collect();torch.cuda.empty_cache()
        for arm in ('A0','A1','P11','PCV','PCOLL'):
            setup(2022,device);m=make_model(bundle,p['datasets'][dataset]['source']).to(device)
            e=copy.deepcopy(m).requires_grad_(False);c=config(arm,dataset,0. if arm in ('A0','A1') else .2);opt=make_optimizer(m,c)
            torch.cuda.reset_peak_memory_stats(device);tick=time.perf_counter();counts={}
            for step in (1,2):
                context['global_step']=step
                result=train_step(m,opt,e,x,y,rows,c,context,diagnostic=True)
                for k,v in result['counts'].items():counts[k]=counts.get(k,0)+v
            torch.cuda.synchronize(device);reserved=torch.cuda.max_memory_reserved(device);peak=max(peak,reserved)
            assert counts['optimizer_steps']==2 and counts['ema_updates']==2
            records.append(dict(dataset=dataset,arm=arm,rows=len(y),steps=2,seconds=time.perf_counter()-tick,
                counts=counts,last_loss=result['loss'],diagnostics=result['diagnostics'],parameter_manifest=opt.parameter_manifest,
                peak_reserved_bytes=reserved,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
            del m,e,opt,result;gc.collect();torch.cuda.empty_cache()
        del bundle,x,y;gc.collect()
    out=dict(status='PASS',device=device,data=manifests,measurements=records,peak_reserved_bytes=peak,
             worker_seconds=time.perf_counter()-start,test_accessed=False,formal_trajectories=0)
    write_json(attempt/'STAGE0_REPORT.json',out)
    return out
