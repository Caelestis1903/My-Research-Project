"""Only source identity, view semantics and trajectory noninterference checks."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .protocol import setup,arm_config,optimizer_for,cpu_copy,equal,rng_state
from .views import make_model
from .objectives import losses
from .probe import temporal_observation


def trajectory(bundle,dataset,p,device,enabled,out):
    setup(2022,device);config=arm_config(p,'B-ML' if dataset=='movielens' else 'B-KR',dataset)
    model=make_model(bundle,p['datasets'][dataset]).to(device);opt=optimizer_for(model,p,config)
    loaders={k:torch.Generator().manual_seed(2022+offset) for k,offset in p['rng']['loader_offsets'].items()}
    order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy();previous=None
    for batch in range(1,5):
        ids=order[(batch-1)*40960:batch*40960]
        x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
        opt.zero_grad(set_to_none=True)
        if enabled:previous=temporal_observation(model,opt,loaders,bundle,dataset,2022,1,batch,batch,device,out,previous)
        loss,_=losses(model,x,y,p['datasets'][dataset]['augmented_domain'],config,dataset,2022,batch)
        (config['loss_scale']*loss).backward();opt.step()
    return cpu_copy((model.state_dict(),opt.state_dict(),[param.grad for param in model.parameters()],
                     rng_state(device),{k:g.get_state() for k,g in loaders.items()}))


def run_checks(bundles,p,device,out):
    records=[]
    for dataset,bundle in bundles.items():
        setup(2022,device);conf=p['datasets'][dataset]
        base='B-ML' if dataset=='movielens' else 'B-KR';config=arm_config(p,base,dataset)
        model=make_model(bundle,conf).to(device);reference=make_model(bundle,conf,reference=True).to(device)
        reference.load_state_dict(model.state_dict())
        assert sum(param.numel() for param in model.parameters())=={'movielens':773470,'kuairand':1614254}[dataset]
        x=torch.as_tensor(bundle.train_x[:40960],device=device).T;y=torch.as_tensor(bundle.train_y[:40960],device=device)
        oa=optimizer_for(model,p,config);ob=optimizer_for(reference,p,config)
        value,info=losses(model,x,y,conf['augmented_domain'],config,dataset,2022,1)
        pred=reference(x);oracle=F.binary_cross_entropy(pred,y,reduction='none').mean(1).mean()
        torch.testing.assert_close(info['pred'],pred,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(value,oracle,atol=1e-6,rtol=1e-5)
        (config['loss_scale']*value).backward();(config['loss_scale']*oracle).backward()
        for a,b in zip(model.parameters(),reference.parameters()):
            assert (a.grad is None)==(b.grad is None)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
        oa.step();ob.step()
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        peaks=[]
        for arm in ('M1','G-IID2','G-ANTI2'):
            setup(2022,device);candidate=make_model(bundle,conf).to(device)
            ac=arm_config(p,arm,dataset,{dataset:base});opt=optimizer_for(candidate,p,ac)
            torch.cuda.reset_peak_memory_stats(device)
            loss,info=losses(candidate,x,y,conf['augmented_domain'],ac,dataset,2022,1)
            (ac['loss_scale']*loss).backward();opt.step()
            peaks.append(dict(arm=arm,peak_memory_bytes=torch.cuda.max_memory_allocated(device),extra_views=info['extra_views']))
            del candidate,opt,loss,info
        del model,reference,oa,ob,value,pred,oracle
        without=trajectory(bundle,dataset,p,device,False,None)
        with_probe=trajectory(bundle,dataset,p,device,True,Path(out)/'trajectory'/dataset)
        assert equal(without,with_probe),'Probe on/off changed the four-step real-data training trajectory'
        records.append(dict(dataset=dataset,source_identity=True,parameter_count=True,actual_batch_size=40960,
            view_memory_checks=peaks,temporal_on_off_equal=True,real_development_optimizer_steps=13,
            maximum_development_optimizer_steps=conf['steps_per_epoch'],test_evaluated=False))
        del without,with_probe
    return dict(status='PASS',datasets=records,seed=2022,development_test_access=False)
