"""Short real-data CUDA acceptance and measured dispatch reservations."""
import copy
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import (load_protocol, config, make_optimizer, setup, cpu_copy,
                       state_digest, write_json, rng_state)
from .vendor.gradient_aug_20261031_v2.views import make_model
from .data import load_dataset
from .evaluate import evaluate
from .objectives import train_step
from .input_probe import embedding, probe_logits
from .guardian import ensure_guardian, request

CORE_ARMS = ('A0','A1','J0','J1','F0','F1')


def probe_check(model, x, device):
    start=time.perf_counter()
    x = x[:, :min(x.shape[1], 128)]
    before = state_digest(model.state_dict()); random_before = state_digest(rng_state(device))
    h = embedding(model, x).requires_grad_(True)
    logits = probe_logits(model, x, embedding_leaf=h)
    with torch.no_grad(): expected = model(x)
    probability_error = float((logits.sigmoid()-expected).abs().max())
    assert probability_error <= 2e-6
    # A deterministic sparse direction avoids consuming the training RNG.
    direction = torch.zeros_like(h)
    for lo, hi in model.perturb_slices:
        direction[:, lo:hi] = torch.sin(torch.arange(hi-lo, device=device, dtype=h.dtype)+.31)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    direction *= h.norm(dim=1, keepdim=True).clamp_min(1e-3)
    derivative = torch.autograd.grad(logits.sum(), h)[0]
    analytic = float((derivative*direction).sum())
    differences = []
    for eps in (1e-3, 1e-4):
        with torch.no_grad():
            plus = probe_logits(model,x,embedding_leaf=h.detach()+eps*direction).sum()
            minus = probe_logits(model,x,embedding_leaf=h.detach()-eps*direction).sum()
        numerical = float((plus-minus)/(2*eps))
        differences.append(dict(step=eps, analytic=analytic, numerical=numerical,
            absolute_error=abs(numerical-analytic), relative_error=abs(numerical-analytic)/max(abs(analytic),1e-6)))
    assert min(v['relative_error'] for v in differences) <= .01, differences
    assert before == state_digest(model.state_dict())
    assert random_before == state_digest(rng_state(device))
    assert all(p.grad is None for p in model.parameters())
    return dict(probability_max_abs_error=probability_error, directional_checks=differences,
                parameter_gradients_none=True, state_rng_unchanged=True,seconds=time.perf_counter()-start,
                work=dict(probe_forwards=5,probe_forward_rows=5*x.shape[1],source_forwards=1,
                          source_forward_rows=x.shape[1],input_gradient_calls=1))


def execute(attempt, device='cuda:0', arms=CORE_ARMS, startup_seconds=0.):
    attempt=Path(attempt); p=load_protocol(); ensure_guardian(attempt)
    measurements={}; data_records={}; forecasts={}; physical=[]
    if (attempt/'STAGE0_REPORT.json').exists():
        import json
        prior=json.loads((attempt/'STAGE0_REPORT.json').read_text())
        measurements.update(prior['measurements']);data_records.update(prior['data'])
        forecasts.update(prior['cost_forecast']);physical.extend(prior['physical_runs'])
    for dataset_index,(dataset, conf) in enumerate(p['datasets'].items()):
        load_start=time.perf_counter();bundle,manifest=load_dataset(dataset,p=p)
        data_seconds=time.perf_counter()-load_start;data_records[dataset]=manifest
        write_json(attempt/'data'/f'{dataset}_original.json',manifest)
        order=torch.randperm(len(bundle.train_y),generator=torch.Generator().manual_seed(2022+11000)).numpy()
        eval_cost=None
        for arm_index,arm in enumerate(arms):
            c=config(arm,dataset);setup(2022,device);init_start=time.perf_counter()
            model=make_model(bundle,conf['source']).to(device);ema=copy.deepcopy(model).requires_grad_(False)
            opt=make_optimizer(model,c);torch.cuda.synchronize(device)
            init_seconds=time.perf_counter()-init_start
            assert sum(v.numel() for v in model.parameters()) == (773470 if dataset=='movielens' else 1614254)
            torch.cuda.reset_peak_memory_stats(device)
            check=None;step_times=[];diagnostics=[];counts={};start=time.perf_counter()
            for step in (1,2):
                ids=order[(step-1)*40960:step*40960]
                x=torch.as_tensor(bundle.train_x[ids],device=device).T
                y=torch.as_tensor(bundle.train_y[ids],device=device)
                if step==1:
                    target=x[:,x[-1]==c['target_domain']]
                    check=probe_check(model,target,device)
                torch.cuda.synchronize(device);tick=time.perf_counter()
                result=train_step(model,opt,ema,x,y,bundle.train_row_ids[ids],c,
                    dict(study='EMA02V2',dataset=dataset,split_id='original',seed=2022,epoch=1,global_step=step),diagnostic=True)
                torch.cuda.synchronize(device);step_times.append(time.perf_counter()-tick)
                assert np.isfinite(result['loss']) and all(q.grad is None for q in ema.parameters())
                diagnostics.append(result['diagnostics'])
                for k,v in result['counts'].items():counts[k]=counts.get(k,0)+v
                del x,y,result
            if eval_cost is None:
                _,online_cost=evaluate(model,bundle.val_x,bundle.val_y,device)
                _,ema_cost=evaluate(ema,bundle.val_x,bundle.val_y,device)
                eval_cost=dict(online=online_cost,fixed_ema=ema_cost)
            sample=cpu_copy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                selected={f'{kind}_{point}':model.state_dict() for kind in ('online','fixed_ema') for point in ('selected','E50')}))
            ipc_start=time.perf_counter();echo=request(attempt,'roundtrip',value=sample)
            ipc_seconds=time.perf_counter()-ipc_start
            assert state_digest(echo['value'])==state_digest(sample)
            peak=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
            elapsed=time.perf_counter()-start
            key=dataset+'/'+arm
            measurements[key]=dict(dataset=dataset,arm=arm,steps=2,batch_size=40960,step_seconds=step_times,
                data_seconds=data_seconds,init_seconds=init_seconds,validation_pair=eval_cost,
                validation_pair_measurement=dataset+'/'+arms[0],validation_measurement_reused=arm_index!=0,
                ipc_roundtrip_seconds=ipc_seconds,peak_memory_bytes=peak,peak_reserved_bytes=reserved,
                probe=check,diagnostics=diagnostics,counts=counts)
            epochs_steps=int(np.ceil(len(bundle.train_y)/40960))*50
            final_forecast=(2*sum(v['seconds'] for v in eval_cost.values())+data_seconds+
                4*init_seconds+2*ipc_seconds)
            forecast=(startup_seconds+data_seconds+init_seconds+max(step_times)*epochs_steps+
                50*sum(v['seconds'] for v in eval_cost.values())+50*ipc_seconds+final_forecast)
            forecasts[key]=dict(expected_worker_seconds=forecast,peak_memory_bytes=peak,
                peak_reserved_bytes=reserved,measurement=key,train_steps=epochs_steps,
                train_step_seconds=max(step_times),validation_passes=100,final_evaluation_passes=4,
                final_evaluation_seconds=final_forecast,
                final_evaluation_forecast='four_forwards_plus_standalone_data_init_and_IPC_from_Stage0',
                process_startup_seconds=startup_seconds,data_seconds=data_seconds,
                initialization_seconds=init_seconds,epoch_ipc_seconds=ipc_seconds)
            physical.append(dict(dataset=dataset,arm=arm,kind='S0',worker_seconds=elapsed+init_seconds+
                (data_seconds if arm_index==0 else 0.)+(startup_seconds if arm_index==dataset_index==0 else 0.),
                measured_steps=2,optimizer_steps=counts['optimizer_steps'],acceptance_probe_work=check['work'],
                measured_validation_forwards=sum(v['forwards'] for v in eval_cost.values()) if arm_index==0 else 0,
                measured_validation_rows=sum(v['rows'] for v in eval_cost.values()) if arm_index==0 else 0,
                test_accessed=False))
            write_json(attempt/'STAGE0_REPORT.json',dict(status='IN_PROGRESS',measurements=measurements,
                data=data_records,cost_forecast=forecasts,physical_runs=physical,test_accessed=False))
            del model,ema,opt,sample,echo;gc.collect();torch.cuda.empty_cache()
        del bundle;gc.collect()
    report=dict(status='PASS',measurements=measurements,data=data_records,cost_forecast=forecasts,
                physical_runs=physical,test_accessed=False)
    write_json(attempt/'STAGE0_REPORT.json',report);write_json(attempt/'cost_forecast.json',forecasts)
    return report
