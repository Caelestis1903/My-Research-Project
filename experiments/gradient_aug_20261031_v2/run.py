"""GA31-R3: one RAM coordinator through baseline selection, probes and test."""
import argparse
import contextlib
import gc
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .protocol import (DEFAULT_PROTOCOL,ROOT,protocol,setup,optimizer_for,snapshot,restore,cpu_copy,equal,
    tensor_bytes,send_memory,receive_memory,write_json,append_csv,arm_config,stage_a_jobs,stage_b_jobs,
    choose_baselines,anti_gate)
from .views import make_model,digest_tensor
from .objectives import losses
from .probe import temporal_observation,run_offline
from .vendor_ga31.data import load_dataset
from .vendor_ga31.audited_data import row_ids_hash
from .vendor_ga31.metrics import metrics,evaluate


def state_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for key,item in v.items():h.update(str(key).encode());visit(item)
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
        else:h.update(str(v).encode())
    visit(value)
    return h.hexdigest()


def optimizer_diagnostic(model,opt,before,grad_norm_sq,config,epoch,step,out):
    fractions=[];energies=[];theta_sq=0.
    for parameter,prior in zip(model.parameters(),before):
        if parameter.grad is None:continue
        state=opt.state[parameter];count=float(state['step'])
        vhat=state['exp_avg_sq']/(1-.999**count)
        fraction=config['eps']/(vhat.sqrt()+config['eps'])
        fractions.append(fraction.detach().reshape(-1).cpu().numpy())
        energies.append((parameter.detach()-prior).square().reshape(-1).cpu().numpy())
        theta_sq+=float(prior.double().square().sum())
    f=np.concatenate(fractions).astype(np.float64);e=np.concatenate(energies).astype(np.float64)
    append_csv(Path(out)/'optimizer_scale.csv',dict(epoch=epoch,global_step=step,measurement='last_actual_batch',
        decay_to_data_gradient=config['weight_decay']*np.sqrt(theta_sq/grad_norm_sq) if grad_norm_sq else None,
        epsilon_fraction_min=float(f.min()),epsilon_fraction_median=float(np.median(f)),
        epsilon_fraction_p90=float(np.quantile(f,.9)),epsilon_fraction_p99=float(np.quantile(f,.99)),
        epsilon_fraction_max=float(f.max()),epsilon_fraction_update_energy_weighted=float((f*e).sum()/e.sum()) if e.sum() else None,
        update_energy=float(e.sum())))
def train_one(bundle,dataset,p,arm,seed,device,out,connection=None,epochs=50,probes=True,baselines=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    setup(seed,device);conf=p['datasets'][dataset];a=arm_config(p,arm,dataset,baselines)
    model=make_model(bundle,conf).to(device);opt=optimizer_for(model,p,a)
    assert sum(v.numel() for v in model.parameters())=={'movielens':773470,'kuairand':1614254}[dataset]
    init=cpu_copy(model.state_dict())
    loaders={k:torch.Generator().manual_seed(seed+offset) for k,offset in p['rng']['loader_offsets'].items()}
    write_json(out/'resolved_config.json',dict(dataset=dataset,arm=arm,seed=seed,device=str(device),config=a,optimizer=dict(p['optimizer'],weight_decay=a['weight_decay'],eps=a['eps']),epochs=epochs,batch_size=40960,source_commit=p['source_commit'],state_storage='RAM_ONLY'))
    states={};selected_epoch=None;best=-float('inf');step=0;trace=[];start=time.perf_counter()
    forwards=0;backwards=0;extra_views=0;ready_sent=False;temporal_count=0;temporal_seconds=0.
    temporal_previous=None
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,epochs+1):
        epoch_start=time.perf_counter();model.train();temporal_previous=None
        order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        order_hash=row_ids_hash(bundle.train_row_ids[order]);trace.append(order_hash)
        preds=[];ys=[];ds=[];optimized_total=0.;real_total=0.;epoch_views=0
        for batch_index,start_row in enumerate(range(0,len(order),40960),1):
            indices=order[start_row:start_row+40960];step+=1
            x=torch.as_tensor(bundle.train_x[indices],device=device).T
            y=torch.as_tensor(bundle.train_y[indices],device=device)
            opt.zero_grad(set_to_none=True)
            if a['stage']=='A' and probes and epoch in p['probe']['temporal_online']['windows_epochs'][dataset] and batch_index<=4:
                observation=temporal_observation(model,opt,loaders,bundle,dataset,seed,epoch,batch_index,step,device,out,temporal_previous)
                temporal_count+=1;temporal_seconds+=observation['row']['seconds'];temporal_previous=observation if batch_index<4 else None
                del observation
            loss,info=losses(model,x,y,conf['augmented_domain'],a,dataset,seed,step)
            scaled=a['loss_scale']*loss
            if not bool(torch.isfinite(scaled)): raise FloatingPointError(f'nonfinite loss epoch{epoch} step{step}')
            scaled.backward()
            diagnostic=a['stage']=='A' and batch_index==conf['steps_per_epoch']
            prior=[param.detach().clone() for param in model.parameters()] if diagnostic else None
            grad_norm_sq=sum(float(param.grad.detach().double().square().sum()) for param in model.parameters() if param.grad is not None) if diagnostic else None
            opt.step()
            if diagnostic:optimizer_diagnostic(model,opt,prior,grad_norm_sq,a,epoch,step,out)
            for v,view in enumerate(info['views'],1):
                append_csv(out/'view_stats.csv',dict(epoch=epoch,batch=batch_index,global_step=step,view=v,family=view['family'],key=view['key'],noise_seed=view['seed'],view_hash=view['hash'],target_rows=int(info['mask'].sum())))
            n=len(indices);optimized_total+=float(scaled.detach())*n;real_total+=float(info['real'].detach().mean())*n
            extra_views+=info['extra_views'];epoch_views+=info['extra_views'];forwards+=info['forwards'];backwards+=1
            preds.append(info['pred'].detach().cpu().numpy());ys.append(y.cpu().numpy());ds.append(x[-1].cpu().numpy())
            append_csv(out/'batch_trace.csv',dict(epoch=epoch,batch=batch_index,global_step=step,rows=n,
                       original_row_hash=row_ids_hash(bundle.train_row_ids[indices]),epoch_order_hash=order_hash,
                       extra_views=info['extra_views']))
            if not ready_sent and connection is not None:
                torch.cuda.synchronize(device)
                send_memory(connection,dict(kind='allocated',peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
                ready_sent=True
        train_metrics=metrics(np.concatenate(ys),np.concatenate(preds),np.concatenate(ds))
        val=evaluate(model,bundle.val_x,bundle.val_y,device)
        forwards+=(len(bundle.val_y)+40959)//40960
        selected=val['macro_auc']>best
        if selected:
            if selected_epoch is not None and selected_epoch not in p['training']['save_epochs']:
                del states[selected_epoch]
            selected_epoch=epoch;best=val['macro_auc']
        if epoch in p['training']['save_epochs'] or selected:
            states[epoch]=snapshot(model,opt,loaders,epoch,step,val,device)
        append_csv(out/'epoch_metrics.csv',dict(epoch=epoch,global_step=step,original_rows=len(order),
            epoch_order_hash=order_hash,training_online_optimized_loss=optimized_total/len(order),
            training_online_real_bce=real_total/len(order),extra_views=epoch_views,
            epoch_seconds=time.perf_counter()-epoch_start,is_selected=int(selected),
            **{'train_online_'+k:v for k,v in train_metrics.items()},**{'val_'+k:v for k,v in val.items()}))
        for split,record in [('train_online',train_metrics),('validation',val)]:
            for d in range(3):
                for t in range(2):
                    append_csv(out/'per_cell_metrics.csv',dict(epoch=epoch,split=split,domain=d,task=t,
                        auc=record[f'd{d}t{t}_auc'],logloss=record[f'd{d}t{t}_logloss'],
                        count=record[f'd{d}t{t}_count'],positive=record[f'd{d}t{t}_positive']))
        append_csv(out/'augmentation_stats.csv',dict(epoch=epoch,original_rows=len(order),extra_views=epoch_views,target_domain=conf['augmented_domain'],mix_lambda=.5,relative_std=.1,weight_sum_per_original=1.,num_views=a['num_views']))
        print(f'{dataset} {arm} seed{seed} epoch{epoch}/{epochs} step{step} selected={selected}',flush=True)
    if epochs==50:assert step==conf['total_optimizer_steps']
    training_seconds=time.perf_counter()-start
    probe_result=dict(observed_states=temporal_count,gradients=3*temporal_count,extra_optimizer_steps=0,seconds=temporal_seconds)
    result=dict(dataset=dataset,arm=arm,seed=seed,epochs=epochs,global_step=step,selected_epoch=selected_epoch,
        selected_validation=states[selected_epoch]['validation'],epoch_order_hashes=trace,
        init=init,states=states,config=a,training_seconds=training_seconds,probe=probe_result,
        original_exposures=len(bundle.train_y)*epochs,extra_views=extra_views,
        training_backwards=backwards,training_forwards_including_validation=forwards,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),snapshot_tensor_bytes=tensor_bytes(states),
        output_dir=str(out))
    write_json(out/'state_summary.json',dict(init_hash=state_digest(init),snapshots=[dict(epoch=e,global_step=state['global_step'],model_hash=state_digest(state['model']),optimizer_hash=state_digest(state['optimizer'])) for e,state in states.items()]))
    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k not in ('init','states')})
    return result



def worker(connection,bundles,dataset,p,arm,seed,device,out,epochs,probes,baselines,task,source_results):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:
                if task=='train':
                    result=train_one(bundles[dataset],dataset,p,arm,seed,device,out,connection,epochs,probes,baselines)
                else:
                    setup(seed,device)
                    send_memory(connection,dict(kind='allocated',peak_memory_bytes=1500*1024**2))
                    source=source_results[(dataset,arm,seed)]
                    result=run_offline(bundles[dataset],dataset,p,source,device,out)
                    result['output_dir']=str(out)
                    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k!='observations'})
                send_memory(connection,dict(kind='complete',result=result))
            except Exception as exc:
                traceback.print_exc()
                send_memory(connection,dict(kind='error',error=repr(exc),traceback=traceback.format_exc(),oom=isinstance(exc,torch.OutOfMemoryError)))
    connection.close()
def gpu_status():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit',
                                    '--format=csv,noheader,nounits'],text=True)
    result=[]
    for line in output.splitlines():
        index,memory,total,power,limit=[float(part.strip()) for part in line.split(',')]
        result.append(dict(index=int(index),memory=memory,total=total,power=power,limit=limit))
    return result


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024


def schedule(bundles,p,run_dir,jobs,epochs=50,probes=True,devices=None,baselines=None,task="train",source_results=None,completed_before=0):
    ctx=mp.get_context('fork');pending=list(jobs);active=[];completed=[];blocked=[];peak={}
    allocation_pending=False;attempts={};serial_only=set()
    while pending or active:
        for record in list(active):
            connection=record['connection'];process=record['process']
            if connection.poll():
                message=receive_memory(connection)
                if message['kind']=='allocated':
                    record['allocated']=True;allocation_pending=False
                    peak[record['gpu']]=max(peak.get(record['gpu'],0),message['peak_memory_bytes'])
                elif message['kind'] in ('complete','error'):
                    process.join();active.remove(record);connection.close()
                    if not record['allocated']:allocation_pending=False
                    if message['kind']=='complete':
                        completed.append(message['result'])
                        write_json(Path(run_dir)/'progress.json',dict(stage=task,completed=completed_before+len(completed),stage_completed=len(completed),stage_total=len(jobs),active=len(active),pending=len(pending)))
                    else:
                        failure=dict(job=record['job'],device=record['gpu'],attempt=record['attempt'],**message)
                        append_csv(Path(run_dir)/'failure_ledger.csv',dict(dataset=record['job'][0],arm=record['job'][1],seed=record['job'][2],
                                   device=record['gpu'],attempt=record['attempt'],oom=message['oom'],error=message['error']))
                        if message['oom'] and record['attempt']==1:
                            serial_only.add(record['job']);pending.append(record['job'])
                        else:blocked.append(failure)
            elif not process.is_alive():
                process.join();active.remove(record);connection.close();allocation_pending=False
                blocked.append(dict(job=record['job'],error=f'worker exited {process.exitcode} without RAM state'))
        if pending and not allocation_pending and available_ram()>8*1024**3:
            for status in gpu_status():
                gpu=status['index'];free=(status['total']-status['memory'])*1024**2
                if devices is not None and gpu not in devices:continue
                if status['memory']/status['total']>=.85 or status['power']/status['limit']>=.85:continue
                if free<peak.get(gpu,1500*1024**2)*1.15:continue
                job_index=next((i for i,job in enumerate(pending) if job not in serial_only or not active),None)
                if job_index is None:continue
                job=pending.pop(job_index);attempts[job]=attempts.get(job,0)+1
                dataset,arm,seed=job;out=Path(run_dir)/('probes' if task=='probe' else 'runs')/dataset/arm/f'seed{seed}'/f'attempt{attempts[job]}'
                parent,child=ctx.Pipe(duplex=False)
                process=ctx.Process(target=worker,args=(child,bundles,dataset,p,arm,seed,f'cuda:{gpu}',out,epochs,probes,baselines,task,source_results))
                process.start();child.close()
                active.append(dict(connection=parent,process=process,job=job,gpu=gpu,allocated=False,attempt=attempts[job]))
                append_csv(Path(run_dir)/'launch_ledger.csv',dict(time=time.time(),dataset=dataset,arm=arm,seed=seed,
                    attempt=attempts[job],gpu=gpu,memory_fraction=status['memory']/status['total'],
                    power_fraction=status['power']/status['limit'],active_workers=len(active)))
                allocation_pending=True;break
        time.sleep(.5)
    return completed,blocked



def freeze_selection(results,p,run_dir,baseline,candidates,expected_jobs):
    expected=set(expected_jobs);actual={(r['dataset'],r['arm'],r['seed']) for r in results}
    assert len(results)==len(expected)==len(actual) and expected==actual
    rows=[]
    for r in results:
        assert r['epochs']==50 and r['global_step']==p['training']['optimizer_steps_per_run'][r['dataset']]
        rows.append(dict(**{k:r[k] for k in ('dataset','arm','seed','selected_epoch','selected_validation','output_dir')},
                         global_step=r['states'][r['selected_epoch']]['global_step']))
    frozen=dict(study=p['study_id'],source_commit=p['source_commit'],seeds=p['seeds'],baseline=baseline,
        candidates=candidates,selected=rows,common_epochs=[10,50],test_started=False,
        effective_run_keys=[list(k) for k in sorted(expected)])
    write_json(Path(run_dir)/'selection_frozen.json',frozen)
    return frozen


def test_worker(connection,bundle,p,result,device):
    try:
        setup(result['seed'],device);model=make_model(bundle,p['datasets'][result['dataset']]).to(device)
        records={}
        for epoch in dict.fromkeys([result['selected_epoch'],10,50]):
            model.load_state_dict(result['states'][epoch]['model'])
            records[epoch]=evaluate(model,bundle.test_x,bundle.test_y,device)
        send_memory(connection,dict(kind='test_complete',metrics=records))
    except Exception:send_memory(connection,dict(kind='error',error=traceback.format_exc()))


def evaluate_all(bundles,p,results,frozen,run_dir,devices):
    assert len(frozen['selected'])==len(results) and Path(run_dir,'selection_frozen.json').exists()
    ctx=mp.get_context('fork')
    for r in results:
        choices=[s for s in gpu_status() if s['index'] in devices]
        gpu=min(choices,key=lambda s:s['memory']/s['total'])['index']
        parent,child=ctx.Pipe(duplex=False)
        proc=ctx.Process(target=test_worker,args=(child,bundles[r['dataset']],p,r,f'cuda:{gpu}'))
        proc.start();child.close();message=receive_memory(parent);proc.join();parent.close()
        if message['kind']=='error':raise RuntimeError(message['error'])
        for checkpoint,epoch in [('selected',r['selected_epoch']),('E10',10),('E50',50)]:
            key=dict(dataset=r['dataset'],arm=r['arm'],seed=r['seed'],checkpoint=checkpoint,
                epoch=epoch,global_step=r['states'][epoch]['global_step'])
            append_csv(Path(run_dir)/'test_metrics.csv',dict(**key,**message['metrics'][epoch]))
            append_csv(Path(r['output_dir'])/'test_metrics.csv',dict(**key,**message['metrics'][epoch]))
            append_csv(Path(run_dir)/'validation_metrics.csv',dict(**key,**r['states'][epoch]['validation']))
        print(f"test complete {r['dataset']} {r['arm']} seed{r['seed']}",flush=True)


def all_stages(bundles,p,out,devices):
    jobs_a=stage_a_jobs(p)
    results,blocked=schedule(bundles,p,out,jobs_a,devices=devices)
    if blocked:raise RuntimeError(f'Stage A blocked: {blocked}')
    baseline=choose_baselines(results,p);write_json(out/'baseline_frozen.json',baseline)
    lookup={(r['dataset'],r['arm'],r['seed']):r for r in results}
    selected_a=[key for key in lookup if key[1]==baseline['baselines'][key[0]]]
    probes,blocked=schedule(bundles,p,out,selected_a,devices=devices,task='probe',source_results=lookup,completed_before=15)
    if blocked:raise RuntimeError(f'Offline probe blocked: {blocked}')
    observations=[row for probe in probes for row in probe['observations']]
    gates=anti_gate(observations,p);jobs_b=stage_b_jobs(p,gates)
    candidates=dict(gates=gates,jobs=[list(j) for j in jobs_b],mandatory_arms=['M1','G-IID2'],
        expected_formal_runs=15+len(jobs_b),frozen_before_stage_b=True)
    write_json(out/'candidate_frozen.json',candidates)
    completed_b,blocked=schedule(bundles,p,out,jobs_b,devices=devices,baselines=baseline['baselines'],completed_before=15)
    results.extend(completed_b)
    if blocked:raise RuntimeError(f'Stage B blocked: {blocked}')
    for dataset in bundles:
        for seed in p['seeds']:
            paired=[r for r in results if r['dataset']==dataset and r['seed']==seed]
            assert all(equal(paired[0]['init'],r['init']) and paired[0]['epoch_order_hashes']==r['epoch_order_hashes'] for r in paired)
    frozen=freeze_selection(results,p,out,baseline,candidates,jobs_a+jobs_b)
    evaluate_all(bundles,p,results,frozen,out,devices)
    for r in results:
        append_csv(out/'run_ledger.csv',{k:v for k,v in r.items() if k not in ('init','states','test_metrics','selected_validation','epoch_order_hashes','probe','config')})
    online=sum(r['probe']['observed_states'] for r in results);pg=sum(r['gradients'] for r in probes)
    steps=sum(r['counterfactual_steps'] for r in probes)
    assert online==120 and pg<=840 and steps<=300
    write_json(out/'probe_ledger.json',dict(online_observed_states=online,online_gradients=online*3,
        offline_batches=sum(r['batches'] for r in probes),offline_gradients=pg,offline_counterfactual_steps=steps,
        offline_worker_seconds=sum(r['seconds'] for r in probes),offline=[{k:v for k,v in r.items() if k!='observations'} for r in probes]))
    write_json(out/'completion.json',dict(status='COMPLETE',completed=len(results),expected=len(jobs_a+jobs_b),epochs_each=50,
        test_evaluated=True,original_optimizer_steps=sum(r['global_step'] for r in results),
        snapshot_tensor_bytes=sum(r['snapshot_tensor_bytes'] for r in results),online_observed_states=online,
        offline_probe_batches=sum(r['batches'] for r in probes)))
    from .analyze import build_tables
    build_tables(out,p)
    for item in results:
        item.pop('states');item.pop('init')
    results.clear();lookup.clear();gc.collect()
    write_json(out/'state_release.json',dict(released=True,storage='RAM_ONLY',after='unified test and numeric report tables'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['audit','check','all'],required=True)
    parser.add_argument('--protocol',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--devices',default='0,1')
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();p=protocol(args.protocol);out=Path(args.run_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new attempt: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    (out/'failure_ledger.csv').write_text('dataset,arm,seed,device,attempt,oom,error\n')
    write_json(out/'resolved_protocol.json',p)
    manifest=json.loads((Path(__file__).parent/'SOURCE_MANIFEST.json').read_text())
    manifest['plan_commit']='a08d1045390f55b3047017ed8e0487d7198c1ea4'
    manifest['execution_code_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.rglob('*.py')}
    write_json(out/'source_manifest.json',manifest)
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
        gpus=gpu_status(),tf32=False,cudnn_benchmark=False,cpu_threads=1,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    bundles={}
    for dataset in p['datasets']:
        data_out=out/'data'/dataset;data_out.mkdir(parents=True,exist_ok=True)
        bundles[dataset],_=load_dataset(dataset,p,out=data_out)
        print(f'data identity PASS {dataset}',flush=True)
    if args.mode=='audit':return
    if args.mode=='check':
        from .checks import run_checks
        write_json(out/'checks.json',run_checks(bundles,p,args.device,out));return
    try:all_stages(bundles,p,out,[int(v) for v in args.devices.split(',')])
    except Exception as exc:
        write_json(out/'completion.json',dict(status='BLOCKED',error=repr(exc),test_evaluated=False))
        raise


if __name__=='__main__':main()
