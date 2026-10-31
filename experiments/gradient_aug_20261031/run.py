"""GA31-v2: RAM-only training coordinator and deferred unified test phase.

Run as python -m experiments.gradient_aug_20261031.run --mode train --run-dir <new-attempt>.
"""
import argparse
import contextlib
import gc
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .protocol import (ARMS,DEFAULT_PROTOCOL,ROOT,protocol,setup,optimizer,snapshot,restore,cpu_copy,
                       equal,tensor_bytes,send_memory,receive_memory,write_json,append_csv,arm_config)
from .model import make_model
from .data import load_dataset
from .audited_data import row_ids_hash
from .objectives import losses
from .metrics import metrics,evaluate
from .probe import run_probes


def train_one(bundle,dataset,p,arm,seed,device,out,connection=None,epochs=50,probes=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    setup(seed,device);conf=p['datasets'][dataset];a=arm_config(p,arm)
    model=make_model(bundle,conf).to(device);opt=optimizer(model,p)
    assert sum(v.numel() for v in model.parameters())=={'movielens':773470,'kuairand':1614254}[dataset]
    init=cpu_copy(model.state_dict())
    loaders={k:torch.Generator().manual_seed(seed+offset) for k,offset in p['rng']['loader_offsets'].items()}
    write_json(out/'resolved_config.json',dict(dataset=dataset,arm=arm,seed=seed,device=str(device),
        augmentation_enabled=a['augmentation_enabled'],loss_scale=a['loss_scale'],
        aug_relative_std=.1,aug_mix_lambda=.5,gradient_probe_enabled=probes,
        unsupported_flags=p['unsupported_flags'],optimizer=p['optimizer'],epochs=epochs,batch_size=40960,
        source_commit=p['source_commit'],model_module=model.__class__.__module__,
        forward='(x,test_flag=False,*,relative_std=None,generator=None)',fixed_fusion=[1,1,1,1],
        optimizer_count=1,loss='mean_i,t weighted real/augmented probability BCE',state_storage='RAM only'))
    states={};selected_epoch=None;best=-float('inf');step=0;trace=[];start=time.perf_counter()
    forwards=0;backwards=0;extra_views=0;ready_sent=False
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,epochs+1):
        epoch_start=time.perf_counter();model.train()
        order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        order_hash=row_ids_hash(bundle.train_row_ids[order]);trace.append(order_hash)
        preds=[];ys=[];ds=[];optimized_total=0.;real_total=0.;epoch_views=0
        for batch_index,start_row in enumerate(range(0,len(order),40960),1):
            indices=order[start_row:start_row+40960];step+=1
            x=torch.as_tensor(bundle.train_x[indices],device=device).T
            y=torch.as_tensor(bundle.train_y[indices],device=device)
            noise_seed=seed*1000000+step
            opt.zero_grad(set_to_none=True)
            loss,info=losses(model,x,y,conf['augmented_domain'],noise_seed,enabled=a['augmentation_enabled'])
            scaled=a['loss_scale']*loss
            if not bool(torch.isfinite(scaled)): raise FloatingPointError(f'nonfinite loss epoch{epoch} step{step}')
            scaled.backward();opt.step()
            n=len(indices);optimized_total+=float(scaled.detach())*n;real_total+=float(info['real'].detach().mean())*n
            extra_views+=info['extra_views'];epoch_views+=info['extra_views'];forwards+=info['forwards'];backwards+=1
            preds.append(info['pred'].detach().cpu().numpy());ys.append(y.cpu().numpy());ds.append(x[-1].cpu().numpy())
            append_csv(out/'batch_trace.csv',dict(epoch=epoch,batch=batch_index,global_step=step,rows=n,
                       original_row_hash=row_ids_hash(bundle.train_row_ids[indices]),epoch_order_hash=order_hash,
                       noise_seed=noise_seed if a['augmentation_enabled'] else None,extra_views=info['extra_views']))
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
        append_csv(out/'augmentation_stats.csv',dict(epoch=epoch,original_rows=len(order),extra_views=epoch_views,
                   target_domain=conf['augmented_domain'],mix_lambda=.5,relative_std=.1,
                   weight_sum_per_original=1.,noise_seed_first=seed*1000000+step-conf['steps_per_epoch']+1,
                   noise_seed_last=seed*1000000+step))
        print(f'{dataset} {arm} seed{seed} epoch{epoch}/{epochs} step{step} selected={selected}',flush=True)
    if epochs==50:assert step==conf['total_optimizer_steps']
    training_seconds=time.perf_counter()-start
    probe_result=run_probes(bundle,dataset,p,arm,seed,states,selected_epoch,device,out) if probes else None
    result=dict(dataset=dataset,arm=arm,seed=seed,epochs=epochs,global_step=step,selected_epoch=selected_epoch,
        selected_validation=states[selected_epoch]['validation'],epoch_order_hashes=trace,
        init=init,states=states,training_seconds=training_seconds,probe=probe_result,
        original_exposures=len(bundle.train_y)*epochs,extra_views=extra_views,
        training_backwards=backwards,training_forwards_including_validation=forwards,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),snapshot_tensor_bytes=tensor_bytes(states),
        output_dir=str(out))
    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k not in ('init','states')})
    return result


def worker_train(connection,bundles,dataset,p,arm,seed,device,out,epochs,probes):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:
                result=train_one(bundles[dataset],dataset,p,arm,seed,device,out,connection,epochs,probes)
                send_memory(connection,dict(kind='complete',result=result))
            except Exception as exc:
                traceback.print_exc()
                send_memory(connection,dict(kind='error',error=repr(exc),traceback=traceback.format_exc(),
                    oom=isinstance(exc,torch.OutOfMemoryError)))
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


def schedule(bundles,p,run_dir,jobs,epochs=50,probes=True,devices=None):
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
                        write_json(Path(run_dir)/'progress.json',dict(completed=len(completed),total=len(jobs),active=len(active),pending=len(pending)))
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
                dataset,arm,seed=job;out=Path(run_dir)/'runs'/dataset/arm/f'seed{seed}'/f'attempt{attempts[job]}'
                parent,child=ctx.Pipe(duplex=False)
                process=ctx.Process(target=worker_train,args=(child,bundles,dataset,p,arm,seed,f'cuda:{gpu}',out,epochs,probes))
                process.start();child.close()
                active.append(dict(connection=parent,process=process,job=job,gpu=gpu,allocated=False,attempt=attempts[job]))
                append_csv(Path(run_dir)/'launch_ledger.csv',dict(time=time.time(),dataset=dataset,arm=arm,seed=seed,
                    attempt=attempts[job],gpu=gpu,memory_fraction=status['memory']/status['total'],
                    power_fraction=status['power']/status['limit'],active_workers=len(active)))
                allocation_pending=True;break
        time.sleep(.5)
    return completed,blocked


def freeze_selection(results,p,run_dir):
    assert len(results)==40 and len({(r['dataset'],r['arm'],r['seed']) for r in results})==40
    rows=[];scales={};sensitivity_scales={}
    for dataset in p['datasets']:
        baseline={arm:np.mean([r['selected_validation']['macro_auc'] for r in results if r['dataset']==dataset and r['arm']==arm])
                  for arm in ('P0','P1')}
        scales[dataset]=1 if baseline['P0']>=baseline['P1'] else .1
        old_means={arm:np.mean([r['selected_validation']['macro_auc'] for r in results if r['dataset']==dataset and r['arm']==arm and r['seed']<=2025]) for arm in ('P0','P1')}
        sensitivity_scales[dataset]=1 if old_means['P0']>=old_means['P1'] else .1
    for r in results:
        rows.append(dict(**{k:r[k] for k in ('dataset','arm','seed','selected_epoch','selected_validation','output_dir')},
                         global_step=r['states'][r['selected_epoch']]['global_step']))
    frozen=dict(study=p['study_id'],source_commit=p['source_commit'],seeds=p['seeds'],
                working_scale=scales,four_seed_sensitivity_working_scale=sensitivity_scales,selected=rows,common_epochs=[10,50],test_started=False)
    write_json(Path(run_dir)/'selection_frozen.json',frozen)
    return frozen


def test_worker(connection,bundle,dataset,p,result,device):
    try:
        setup(result['seed'],device);model=make_model(bundle,p['datasets'][dataset]).to(device)
        records={}
        for epoch in dict.fromkeys([result['selected_epoch'],10,50]):
            model.load_state_dict(result['states'][epoch]['model'])
            records[epoch]=evaluate(model,bundle.test_x,bundle.test_y,device)
        send_memory(connection,dict(kind='test_complete',metrics=records))
    except Exception:
        send_memory(connection,dict(kind='error',error=traceback.format_exc()))


def evaluate_all(bundles,p,results,frozen,run_dir,devices=None):
    assert len(frozen['selected'])==40
    ctx=mp.get_context('fork')
    # Test has no backward; one worker at a time leaves other processes' allocation unchanged.
    for result in results:
        choices=[s for s in gpu_status() if devices is None or s['index'] in devices]
        gpu=min(choices,key=lambda s:s['memory']/s['total'])['index']
        parent,child=ctx.Pipe(duplex=False)
        process=ctx.Process(target=test_worker,args=(child,bundles[result['dataset']],result['dataset'],p,result,f'cuda:{gpu}'))
        process.start();child.close();message=receive_memory(parent);process.join();parent.close()
        if message['kind']=='error':raise RuntimeError(message['error'])
        result['test_metrics']=message['metrics']
        rows=[]
        for checkpoint,epoch in [('selected',result['selected_epoch']),('E10',10),('E50',50)]:
            row=dict(dataset=result['dataset'],arm=result['arm'],seed=result['seed'],checkpoint=checkpoint,
                     epoch=epoch,global_step=result['states'][epoch]['global_step'],**result['test_metrics'][epoch])
            append_csv(Path(result['output_dir'])/'test_metrics.csv',row)
            append_csv(Path(run_dir)/'test_metrics.csv',row)
            val=dict(dataset=result['dataset'],arm=result['arm'],seed=result['seed'],checkpoint=checkpoint,
                     epoch=epoch,global_step=result['states'][epoch]['global_step'],**result['states'][epoch]['validation'])
            append_csv(Path(run_dir)/'validation_metrics.csv',val)
        print(f"test complete {result['dataset']} {result['arm']} seed{result['seed']}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['audit','check','dev','train','evaluate'],required=True)
    parser.add_argument('--dataset',choices=['movielens','kuairand'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=range(2022,2027),default=2022)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--devices',default='0,1')
    parser.add_argument('--protocol',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--data-path')
    args=parser.parse_args();p=protocol(args.protocol);out=Path(args.run_dir)
    if args.mode=='evaluate':raise RuntimeError('evaluate requires live RAM states in the training coordinator; cold-start state files are prohibited')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new attempt directory: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    write_json(out/'resolved_protocol.json',p)
    import json
    write_json(out/'source_manifest.json',json.loads((ROOT/'research_reports/2026-10-31/SOURCE_FETCH_RECEIPT.json').read_text()))
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
        gpus=gpu_status(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        tf32=False,cudnn_benchmark=False,cpu_threads=1,working_tree='exported source; no local Git metadata'))
    datasets=[args.dataset] if args.dataset else list(p['datasets'])
    bundles={}
    for dataset in datasets:
        data_out=out/'data'/dataset;data_out.mkdir(parents=True,exist_ok=True)
        bundles[dataset],_=load_dataset(dataset,p,args.data_path,data_out)
        print(f'data identity PASS {dataset}',flush=True)
    if args.mode=='audit':return
    if args.mode=='check':
        from .checks import run_checks
        write_json(out/'checks.json',run_checks(bundles,p,args.device,out));return
    if args.mode=='dev':
        jobs=[(d,a,2022) for d in datasets for a in ([args.arm] if args.arm else ARMS)]
        results,blocked=schedule(bundles,p,out,jobs,epochs=1,probes=False,devices=[int(args.device.split(':')[-1])])
        write_json(out/'completion.json',dict(completed=len(results),expected=len(jobs),blocked=blocked,mode='dev',test_evaluated=False))
        return
    assert len(bundles)==2 and args.arm is None and args.dataset is None, 'Formal coordinator runs all 40 paired jobs'
    jobs=[(dataset,arm,seed) for seed in p['seeds'] for dataset in datasets for arm in ARMS]
    results,blocked=schedule(bundles,p,out,jobs,devices=[int(v) for v in args.devices.split(',')])
    if blocked:
        write_json(out/'completion.json',dict(completed=len(results),expected=40,blocked=blocked,status='BLOCKED',test_evaluated=False));return
    for dataset in datasets:
        for seed in p['seeds']:
            paired=[r for r in results if r['dataset']==dataset and r['seed']==seed]
            assert all(equal(paired[0]['init'],r['init']) and paired[0]['epoch_order_hashes']==r['epoch_order_hashes'] for r in paired)
    frozen=freeze_selection(results,p,out)
    evaluate_all(bundles,p,results,frozen,out,devices=[int(v) for v in args.devices.split(',')])
    for result in results:
        append_csv(out/'run_ledger.csv',{k:v for k,v in result.items() if k not in ('init','states','test_metrics','selected_validation','epoch_order_hashes','probe')})
    write_json(out/'completion.json',dict(status='COMPLETE',completed=40,epochs_each=50,test_evaluated=True,
               probes_completed=sum(r['probe']['batches'] for r in results),snapshot_tensor_bytes=sum(r['snapshot_tensor_bytes'] for r in results)))
    from .analyze import build_tables
    build_tables(out,p)


if __name__=='__main__':main()
