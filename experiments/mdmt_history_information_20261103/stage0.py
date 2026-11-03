"""Real-model wiring and full-batch resource measurements before formal runs."""
import copy
import gc
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from .protocol import (ROOT,REPORT,load_protocol,load_dataset,make_model,config,make_optimizer,
                       setup,cpu_copy,write_json,state_digest)
from .objectives import train_step
from .vendor.v3.vendor.ema02.objectives import train_step as source_step

def check_cpu():
    start=time.perf_counter()
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).parent/'tests')],
                          cwd=ROOT,capture_output=True,text=True)
    record=dict(status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,
                stdout=result.stdout,stderr=result.stderr,seconds=time.perf_counter()-start)
    write_json(REPORT/'CPU_TESTS.json',record)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return record

def _close(a,b):
    for key in a:assert torch.allclose(a[key],b[key],atol=1e-6,rtol=1e-5),key

def wiring(model,x,y,rows,context,dataset,attempt):
    from .guardian import request
    records=[]
    for arm,overrides,baseline in [('A0f',{},'A0'),('A1',{},'A1'),('H_MIX',{'kappa':0},'A0'),
           ('T0',{'kappa':0},'A0'),('M',{'lambda_mix':0},'A0'),('M',{'tau':0},'A0'),('TM',{'kappa':0},'M')]:
        m=copy.deepcopy(model);reference=copy.deepcopy(model);ema=copy.deepcopy(model).requires_grad_(False)
        reference_ema=copy.deepcopy(ema);c=config(arm,dataset,**overrides);opt=make_optimizer(m,c)
        ref_config=config('M' if baseline=='M' else 'A1' if baseline=='A1' else 'A0f',dataset)
        ref_opt=make_optimizer(reference,ref_config)
        for step in (1,2):
            ctx=dict(context,global_step=step);v=train_step(m,opt,ema,x,y,rows,c,ctx)
            if baseline=='M':r=train_step(reference,ref_opt,reference_ema,x,y,rows,ref_config,ctx)
            else:
                sc=dict(ref_config,arm=baseline,target_domain=0 if dataset=='movielens' else 2,
                        kappa=.05 if dataset=='movielens' else .1,lambda_aug=.5)
                r=source_step(reference,ref_opt,reference_ema,x,y,rows,sc,dict(ctx,study='EMA02V2',split_id='original'))
            assert abs(v['loss']-r['loss'])<=1e-6
            _close(m.state_dict(),reference.state_dict());_close(ema.state_dict(),reference_ema.state_dict())
            for p,q in zip(m.parameters(),reference.parameters()):
                if p.grad is None:assert q.grad is None
                else:assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)
            if step==1:
                saved=request(attempt,'roundtrip',value=cpu_copy(dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict())))['value']
                m.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer'])
                del saved
        records.append(dict(arm=arm,overrides=overrides,baseline=baseline,steps=2,loss_gradient_update_ema_restore='PASS'))
        del m,reference,ema,reference_ema,opt,ref_opt,v,r;gc.collect();torch.cuda.empty_cache()
    return records

def stage0(attempt,device):
    from .guardian import ensure_guardian
    attempt=Path(attempt);attempt.mkdir(parents=True,exist_ok=True);ensure_guardian(attempt)
    p=load_protocol();records=[];manifests={};wiring_records={};peak=0;start=time.perf_counter()
    for dataset in p['datasets']:
        bundle,manifest=load_dataset(dataset,p=p);manifests[dataset]=manifest
        write_json(attempt/'data'/f'{dataset}.json',manifest)
        ids=torch.randperm(len(bundle.train_y),generator=torch.Generator().manual_seed(2022+11000)).numpy()[:40960]
        x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device);rows=bundle.train_row_ids[ids]
        assert x.shape[0]==len(bundle.features.columns) and y.shape==(40960,2)
        setup(2022,device);model=make_model(bundle,p['datasets'][dataset]['source']).to(device)
        assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for m in model.modules())
        context=dict(dataset=dataset,seed=2022,epoch=1,global_step=1)
        # Real-data wiring uses all three domains; the cost smoke below uses the full unchanged batch.
        subset=torch.cat([torch.where(x[-1]==d)[0][:64] for d in range(3)])
        wiring_records[dataset]=wiring(model,x[:,subset],y[subset],rows[subset.cpu().numpy()],context,dataset,attempt)
        del model;gc.collect();torch.cuda.empty_cache()
        for arm in ('A0f','M','H_MIX','TM'):
            setup(2022,device);m=make_model(bundle,p['datasets'][dataset]['source']).to(device)
            ema=copy.deepcopy(m).requires_grad_(False);c=config(arm,dataset);opt=make_optimizer(m,c)
            torch.cuda.reset_peak_memory_stats(device);times=[];counts={};diagnostics=[]
            for step in range(1,6):
                torch.cuda.synchronize(device);tick=time.perf_counter()
                result=train_step(m,opt,ema,x,y,rows,c,dict(context,global_step=step),diagnostic=True)
                torch.cuda.synchronize(device);elapsed=time.perf_counter()-tick
                if step>2:times.append(elapsed)
                for k,v in result['counts'].items():counts[k]=counts.get(k,0)+v
                diagnostics.append(result['diagnostics'])
            reserved=torch.cuda.max_memory_reserved(device);peak=max(peak,reserved)
            assert counts['optimizer_steps']==5 and counts['ema_updates']==5
            record=dict(dataset=dataset,arm=arm,rows=len(y),warmup_steps=2,measured_steps=3,
                measured_seconds=times,median_step_seconds=statistics.median(times),counts=counts,
                diagnostics=diagnostics,last_loss=result['loss'],parameter_manifest=opt.parameter_manifest,
                peak_reserved_bytes=reserved,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            records.append(record);write_json(REPORT/f'STAGE0_{dataset}_{arm}.json',record)
            del m,ema,opt,result;gc.collect();torch.cuda.empty_cache()
        base=next(r['median_step_seconds'] for r in records if r['dataset']==dataset and r['arm']=='A0f')
        for r in records:
            if r['dataset']==dataset:
                r['cost_ratio_vs_A0']=r['median_step_seconds']/base
                r['status']='COST_BLOCKED' if r['cost_ratio_vs_A0']>4 else 'PASS'
        del bundle,x,y;gc.collect()
    result=dict(status='PASS',device=device,measurements=records,wiring=wiring_records,data=manifests,
        peak_reserved_bytes=peak,worker_seconds=time.perf_counter()-start,test_accessed=False,formal_trajectories=0)
    write_json(attempt/'STAGE0_REPORT.json',result);write_json(REPORT/'STAGE0.json',result)
    write_json(REPORT/'DATA_MANIFEST.json',manifests)
    write_json(REPORT/'PARAMETER_MANIFEST.json',{r['dataset']:r['parameter_manifest'] for r in records if r['arm']=='A0f'})
    return result
