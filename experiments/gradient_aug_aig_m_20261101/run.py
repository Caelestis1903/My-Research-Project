"""Finite staged search, pure-RAM caching, and one globally frozen test opening."""
import argparse
import contextlib
import gc
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from experiments.gradient_aug_20261031_v2.views import make_model
from experiments.gradient_aug_20261031_v2.vendor_ga31.data import load_dataset
from experiments.gradient_aug_20261031_v2.vendor_ga31.audited_data import row_ids_hash
from experiments.gradient_aug_20261031_v2.vendor_ga31.metrics import metrics
from experiments.gradient_aug_20261031_v2.run import gpu_status, available_ram
from .protocol import (HERE, ROOT, DEV_SEEDS, SHADOWS, TRIAL_ORDER, protocol, candidates, digest,
    make_optimizer, make_shadows, update_shadows, snapshot, model_state, checkpoint_rows, source_manifest,
    setup, state_digest, tensor_bytes, send_memory, receive_memory, write_json, append_csv, selected_update)
from .optimizer import parameter_groups
from .objectives import train_step
from .analyze import choose, compare, selected_values, build_tables


def evaluate_timed(model, x, y, device):
    was = model.training; model.eval(); predictions = []; events = []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0,len(y),40960):
            bx = torch.as_tensor(x[offset:offset+40960],device=device).T
            left,right = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            left.record(); pred = model(bx); right.record()
            predictions.append(pred.cpu().numpy()); events.append((left,right))
    torch.cuda.synchronize(device)
    gpu_ms = sum(left.elapsed_time(right) for left,right in events)
    values = metrics(y,np.concatenate(predictions),x[:,-1])
    model.train(was)
    return values,dict(forwards=len(events),wall_seconds=time.perf_counter()-start,gpu_event_ms=gpu_ms)


def run_identity(dataset, seed, c, p):
    return dict(dataset=dataset,seed=seed,config_hash=digest(c),data_hash=digest(p['datasets'][dataset]),
        source_hash=p['source_manifest_hash'],execution_commit=p['execution_commit'],rng=p['rng'],
        epochs=50,batch_size=40960,precision='FP32',metrics='GA31_R3_six_cell_macro_v1')


def train_one(bundle, dataset, p, stage, trial, c, seed, device, out, connection=None, attempt=1):
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    setup(seed,device);conf=p['datasets'][dataset]
    model=make_model(bundle,conf).to(device);optimizer=make_optimizer(model,p,c,dataset)
    shadows=make_shadows(model,c);kinds=['online']+list(shadows)
    init_hash=state_digest(model.state_dict());groups=parameter_groups(model,conf['augmented_domain'])
    assert sum(v.numel() for v in model.parameters())=={'movielens':773470,'kuairand':1614254}[dataset]
    loaders={k:torch.Generator().manual_seed(seed+v) for k,v in p['rng']['loader_offsets'].items()}
    selected={};best={k:-float('inf') for k in kinds};states={};order_hashes=[];calibration=[]
    identity=run_identity(dataset,seed,c,p);run_hash=digest(identity)
    write_json(out/'resolved_config.json',dict(config=c,**identity,run_hash=run_hash,stage=stage,trial=trial))
    write_json(out/'parameter_groups.json',dict(zip((n for n,_ in model.named_parameters()),groups)))
    write_json(out/'model_identity.json',dict(parameter_count=sum(v.numel() for v in model.parameters()),
        init_hash=init_hash,field_slices=model.field_slices,perturb_slices=model.perturb_slices))
    counters=dict(original_exposures=0,extra_views=0,view_draws=0,training_forwards=0,gradient_calls=0,
        optimizer_steps=0,shadow_updates=0,validation_forwards=0,validation_gpu_event_ms=0.,
        validation_wall_seconds=0.,training_gpu_event_ms=0.,selection_checks=0)
    validation_costs={k:dict(forwards=0,wall_seconds=0.,gpu_event_ms=0.) for k in kinds}
    torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter();ready=False
    step=0
    for epoch in range(1,51):
        epoch_start=time.perf_counter();model.train()
        order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        order_hash=row_ids_hash(bundle.train_row_ids[order]);order_hashes.append(order_hash)
        predictions=[];labels=[];domains=[];real_sum=reference_sum=0.;gate_epoch=[]
        for batch,offset in enumerate(range(0,len(order),40960),1):
            ids=order[offset:offset+40960];step+=1
            left,right=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            left.record()
            x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
            scalar,info,diagnostics,calls=train_step(model,optimizer,x,y,conf['augmented_domain'],c,
                dataset,seed,step,diagnose=batch==conf['steps_per_epoch'],calibrate=trial=='S95' and epoch==1)
            update_shadows(shadows,model,c);right.record();right.synchronize()
            counters['training_gpu_event_ms']+=left.elapsed_time(right)
            for record in diagnostics:
                row=dict(epoch=epoch,batch=batch,global_step=step,group=record['group'],
                    u=record.get('u'),gate=record.get('gate'),lambda_t=record.get('lambda_t'),
                    q_norm=record.get('q_norm'),m_norm=record.get('m_norm'),injection_ratio=record.get('injection_ratio'),
                    measurement='last_actual_batch' if batch==conf['steps_per_epoch'] else 'training_step')
                append_csv(out/'gate_stats.csv',row);gate_epoch.append(row)
                if trial=='S95' and epoch==1 and row['group'] in ('E','P'):
                    calibration.append(dict(group=row['group'],u=row['u'],seed=seed,epoch=epoch,batch=batch,global_step=step))
            n=len(ids);counters['original_exposures']+=n;counters['extra_views']+=info['extra_views']
            counters['view_draws']+=len(info['views']);counters['training_forwards']+=info['forwards']
            counters['gradient_calls']+=calls;counters['optimizer_steps']+=1;counters['shadow_updates']+=len(shadows)
            real_sum+=float(info['real'].detach().mean(1).mean())*n;reference_sum+=float(scalar)*n
            predictions.append(info['pred'].detach().cpu().numpy());labels.append(y.cpu().numpy());domains.append(x[-1].cpu().numpy())
            append_csv(out/'batch_trace.csv',dict(epoch=epoch,batch=batch,global_step=step,rows=n,
                original_row_hash=row_ids_hash(bundle.train_row_ids[ids]),epoch_order_hash=order_hash,
                extra_views=info['extra_views'],gradient_calls=calls,optimizer_steps=1))
            for view in info['views']:
                append_csv(out/'view_stats.csv',dict(epoch=epoch,batch=batch,global_step=step,
                    view_key=view['key'],view_seed=view['seed'],view_hash=view['hash'],target_rows=int(info['mask'].sum())))
            if not ready and connection is not None:
                send_memory(connection,dict(kind='allocated',peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
                ready=True
            del info,scalar,x,y
        training=metrics(np.concatenate(labels),np.concatenate(predictions),np.concatenate(domains));validation={}
        for kind,current in [('online',model)]+list(shadows.items()):
            validation[kind],cost=evaluate_timed(current,bundle.val_x,bundle.val_y,device)
            for key,value in cost.items():validation_costs[kind][key]+=value
            counters['validation_forwards']+=cost['forwards'];counters['validation_gpu_event_ms']+=cost['gpu_event_ms']
            counters['validation_wall_seconds']+=cost['wall_seconds'];counters['selection_checks']+=1
            selected_update(selected,best,kind,epoch,validation[kind]['macro_auc'])
            append_csv(out/'validation_epoch_metrics.csv',dict(epoch=epoch,global_step=step,model_kind=kind,
                **validation[kind],validation_wall_seconds=cost['wall_seconds'],validation_gpu_event_ms=cost['gpu_event_ms']))
        if epoch in (10,50) or epoch in selected.values():
            states[epoch]=snapshot(model,optimizer,shadows,loaders,epoch,step,validation,selected,best,c,device,counters)
        for old in list(states):
            if old not in (10,50) and old not in selected.values():del states[old]
        append_csv(out/'epoch_metrics.csv',dict(epoch=epoch,global_step=step,epoch_order_hash=order_hash,
            train_real_bce=real_sum/len(order),reference_fixed_mixed_loss=reference_sum/len(order),
            epoch_seconds=time.perf_counter()-epoch_start,**{'train_'+k:v for k,v in training.items()}))
        for group in ('E','P'):
            rows=[r for r in gate_epoch if r['group']==group]
            if rows:
                append_csv(out/'gate_epoch_summary.csv',dict(epoch=epoch,group=group,observations=len(rows),
                    **{key:float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None
                       for key in ('u','gate','lambda_t','q_norm','m_norm','injection_ratio')}))
        print(f'{stage} {dataset} {trial} seed{seed} E{epoch}/50 step{step} selected={selected}',flush=True)
    torch.cuda.synchronize(device);assert step==conf['total_optimizer_steps']
    result=dict(dataset=dataset,stage=stage,trial=trial,seed=seed,attempt=attempt,epochs=50,global_step=step,
        selected_epochs=selected,selected_validation={k:states[e]['validation'][k] for k,e in selected.items()},
        config=c,config_hash=digest(c),run_hash=run_hash,run_identity=identity,init_hash=init_hash,
        epoch_order_hashes=order_hashes,states=states,calibration=calibration,output_dir=str(out),
        training_seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),snapshot_tensor_bytes=tensor_bytes(states),
        validation_costs=validation_costs,**counters)
    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k!='states'})
    write_json(out/'state_summary.json',dict(checkpoints=[row for k in kinds for row in checkpoint_rows(result,k)]))
    return result


def worker(connection,bundles,p,job,device,out,attempt):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:
                result=train_one(bundles[job['dataset']],job['dataset'],p,job['stage'],job['trial'],job['config'],
                    job['seed'],device,out,connection,attempt)
                send_memory(connection,dict(kind='complete',result=result))
            except Exception as exc:
                traceback.print_exc();send_memory(connection,dict(kind='error',error=repr(exc),
                    traceback=traceback.format_exc(),oom=isinstance(exc,torch.OutOfMemoryError)))
    connection.close()


def ledger_row(p,job,status,attempt=0,result=None,cache_source=''):
    identity=run_identity(job['dataset'],job['seed'],job['config'],p);r=result or {}
    data=p['datasets'][job['dataset']]
    return dict(stage=job['stage'],dataset=job['dataset'],trial=job['trial'],seed=job['seed'],attempt=attempt,
        status=status,execution_commit=p['execution_commit'],source_manifest_hash=p['source_manifest_hash'],
        data_hash=identity['data_hash'],raw_sha256=data['raw_sha256'],
        split_row_id_hashes=json.dumps(data['split_row_id_hashes'],sort_keys=True),
        encoder_hash=data['encoder_hash'],scaler_hash=data['scaler_hash'],
        config_hash=identity['config_hash'],run_hash=digest(identity),
        resolved_config=json.dumps(job['config'],sort_keys=True),planned_epochs=50,
        planned_optimizer_steps=p['datasets'][job['dataset']]['total_optimizer_steps'],
        completed_epochs=r.get('epochs',0),completed_optimizer_steps=r.get('global_step',0),
        init_hash=r.get('init_hash',''),epoch_order_hash=digest(r.get('epoch_order_hashes',[])),
        selected_epochs=json.dumps(r.get('selected_epochs',{}),sort_keys=True),
        selected_model_hashes=json.dumps({k:state_digest(model_state(r['states'][e],k))
            for k,e in r.get('selected_epochs',{}).items()},sort_keys=True),
        cache_source=cache_source,output_dir=r.get('output_dir',''))


def schedule(bundles,p,out,jobs,devices):
    ctx=mp.get_context('fork');pending=list(jobs);active=[];completed=[];blocked=[]
    peaks={};attempts={};serial_only=set();allocation_pending=False
    for job in jobs:append_csv(out/'trial_ledger.csv',ledger_row(p,job,'PLANNED'))
    while pending or active:
        for record in list(active):
            connection,proc=record['connection'],record['process'];message=None
            if connection.poll():
                try:message=receive_memory(connection)
                except EOFError:message=dict(kind='error',error=f'worker exited {proc.exitcode} without RAM state',oom=False)
            elif not proc.is_alive():message=dict(kind='error',error=f'worker exited {proc.exitcode} without RAM state',oom=False)
            if message is None:continue
            if message['kind']=='allocated':
                record['allocated']=True;allocation_pending=False
                peaks[record['gpu']]=max(peaks.get(record['gpu'],0),message['peak_memory_bytes']);continue
            proc.join();connection.close();active.remove(record)
            if not record['allocated']:allocation_pending=False
            job=record['job']
            if message['kind']=='complete':
                completed.append(message['result'])
                append_csv(out/'trial_ledger.csv',ledger_row(p,job,'COMPLETE',record['attempt'],message['result']))
            else:
                failure=dict(stage=job['stage'],dataset=job['dataset'],trial=job['trial'],seed=job['seed'],
                    device=record['gpu'],attempt=record['attempt'],oom=message['oom'],error=message['error'])
                append_csv(out/'failure_ledger.csv',failure);append_csv(out/'trial_ledger.csv',ledger_row(p,job,'FAILED',record['attempt']))
                if message['oom'] and record['attempt']==1:
                    serial_only.add(digest(job));pending.append(job)
                else:blocked.append(failure)
            write_json(out/'progress.json',dict(stage=jobs[0]['stage'],completed=len(completed),total=len(jobs),
                active=len(active),pending=len(pending),failed=len(blocked),test_accessed=False))
        if pending and not allocation_pending and available_ram()>8*1024**3:
            for status in gpu_status():
                gpu=status['index']
                if gpu not in devices or status['memory']/status['total']>=.85 or status['power']/status['limit']>=.85:continue
                if (status['total']-status['memory'])*1024**2<peaks.get(gpu,1600*1024**2)*1.15:continue
                index=next((i for i,j in enumerate(pending) if digest(j) not in serial_only or not active),None)
                if index is None:continue
                job=pending.pop(index);key=digest(job);attempts[key]=attempts.get(key,0)+1
                destination=out/'runs'/job['stage']/job['dataset']/job['trial']/f"seed{job['seed']}"/f'attempt{attempts[key]}'
                parent,child=ctx.Pipe(duplex=False)
                proc=ctx.Process(target=worker,args=(child,bundles,p,job,f'cuda:{gpu}',destination,attempts[key]))
                proc.start();child.close();active.append(dict(connection=parent,process=proc,job=job,gpu=gpu,allocated=False,attempt=attempts[key]))
                append_csv(out/'launch_ledger.csv',dict(time=time.time(),stage=job['stage'],dataset=job['dataset'],
                    trial=job['trial'],seed=job['seed'],attempt=attempts[key],gpu=gpu,
                    memory_fraction=status['memory']/status['total'],power_fraction=status['power']/status['limit'],active_workers=len(active)))
                allocation_pending=True;break
        time.sleep(.5)
    if blocked:raise RuntimeError(f'Incomplete stage: {blocked}')
    return completed


def make_jobs(stage,configs,seeds):
    return [dict(stage=stage,dataset=d,trial=trial,config=c,seed=seed)
        for seed in seeds for d,choices in configs.items() for trial,c in choices.items()]


def stage_decision(stage,results,p,reference=None):
    answer={}
    for dataset in p['datasets']:
        best=choose(stage,results,dataset);record=dict(selected=best)
        if reference is not None:
            ref=reference[dataset];comparisons={};allowed=set()
            for trial in TRIAL_ORDER[stage]:
                for kind in SHADOWS:
                    item=compare(selected_values(results,dataset,trial,kind),ref['values'],dataset)
                    comparisons[f'{trial}/{kind}']=item
                    if item['passed']:allowed.add((trial,kind))
            if stage=='S4':
                best=choose(stage,results,dataset,allowed=allowed)
                record.update(selected=best,promoted=best is not None)
            else:record['promoted']=comparisons[f"{best['trial']}/{best['kind']}"]['passed']
            record['comparisons']=comparisons
        answer[dataset]=record
    return dict(stage=stage,datasets=answer,test_accessed=False)


def test_worker(connection,bundle,p,result,kinds,device):
    try:
        setup(result['seed'],device);model=make_model(bundle,p['datasets'][result['dataset']]).to(device)
        records={};cost=dict(forwards=0,wall_seconds=0.,gpu_event_ms=0.)
        for arm,kind in kinds:
            for row in checkpoint_rows(result,kind,arm):
                key=(kind,row['epoch'])
                if key in records:continue
                state=result['states'][row['epoch']];model.load_state_dict(model_state(state,kind))
                records[key],item=evaluate_timed(model,bundle.test_x,bundle.test_y,device)
                for name,value in item.items():cost[name]+=value
        send_memory(connection,dict(kind='complete',metrics=records,cost=cost))
    except Exception:send_memory(connection,dict(kind='error',error=traceback.format_exc()))


def unified_test(bundles,p,out,formal):
    frozen=[]
    for item in formal:
        for arm,kind in item['kinds']:frozen.extend(checkpoint_rows(item['result'],kind,arm))
    write_json(out/'selection_frozen.json',dict(study_id=p['study_id'],execution_commit=p['execution_commit'],
        code_hash=p['source_manifest_hash'],checkpoints=frozen,test_started=False))
    ctx=mp.get_context('fork')
    for item in formal:
        result=item['result'];gpu=min((s for s in gpu_status() if s['index'] in p['devices']),key=lambda s:s['memory']/s['total'])['index']
        parent,child=ctx.Pipe(duplex=False);proc=ctx.Process(target=test_worker,
            args=(child,bundles[result['dataset']],p,result,item['kinds'],f'cuda:{gpu}'))
        proc.start();child.close();message=receive_memory(parent);proc.join();parent.close()
        if message['kind']=='error':raise RuntimeError(message['error'])
        append_csv(out/'test_costs.csv',dict(dataset=result['dataset'],trial=result['trial'],seed=result['seed'],**message['cost']))
        for arm,kind in item['kinds']:
            for row in checkpoint_rows(result,kind,arm):
                validation=row.pop('validation');test=message['metrics'][kind,row['epoch']]
                for split,values in [('validation',validation),('test',test)]:append_csv(out/f'{split}_metrics.csv',dict(**row,**values))
                append_csv(out/'selected_metrics.csv',dict(**row,val_macro_auc=validation['macro_auc'],
                    val_macro_logloss=validation['macro_logloss'],test_macro_auc=test['macro_auc'],test_macro_logloss=test['macro_logloss']))
        print(f"test complete {result['dataset']} {result['trial']} seed{result['seed']}",flush=True)


def aggregate(results,out):
    for filename in ('epoch_metrics.csv','validation_epoch_metrics.csv','gate_stats.csv','gate_epoch_summary.csv'):
        chunks=[]
        for result in results:
            path=Path(result['output_dir'])/filename
            if not path.exists():continue
            table=pd.read_csv(path)
            for key in ('dataset','stage','trial','seed','attempt','config_hash','run_hash'):table[key]=result[key]
            chunks.append(table)
        if chunks:pd.concat(chunks,ignore_index=True).to_csv(out/filename,index=False)
    keys=['dataset','stage','trial','seed','attempt','epochs','global_step','training_seconds',
        'original_exposures','extra_views','view_draws','training_forwards','gradient_calls','optimizer_steps',
        'shadow_updates','validation_forwards','validation_wall_seconds','validation_gpu_event_ms','training_gpu_event_ms',
        'selection_checks','peak_memory_bytes','peak_reserved_bytes','snapshot_tensor_bytes','config_hash','run_hash','output_dir']
    for result in results:append_csv(out/'costs.csv',{k:result[k] for k in keys})
    rows=[]
    for result in results:
        for kind in result['selected_epochs']:rows.extend(checkpoint_rows(result,kind))
    write_json(out/'development_selected_checkpoints.json',dict(checkpoints=rows))


def all_stages(bundles,p,out):
    start=time.perf_counter();results=[];configs={};base={};reference={};decisions={}
    for stage in ('S1','S2','S3','S4'):
        configs[stage]={d:candidates(stage,d,p,base.get(d),p.get('thresholds',{}).get(d)) for d in bundles}
        write_json(out/f'{stage}_candidates.json',configs[stage])
        current=schedule(bundles,p,out,make_jobs(stage,configs[stage],DEV_SEEDS),p['devices']);results.extend(current)
        decisions[stage]=stage_decision(stage,current,p,reference if stage!='S1' else None)
        write_json(out/f'{stage}_decision.json',decisions[stage])
        if stage=='S1':
            for d in bundles:
                selected=decisions[stage]['datasets'][d]['selected'];base[d]=configs[stage][d][selected['trial']]
                reference[d]=dict(**selected,values=selected_values(current,d,selected['trial'],selected['kind']))
            historical={d:compare(reference[d]['values'],selected_values(current,d,'C','ema995'),d) for d in bundles}
            write_json(out/'S1_historical_anchor_comparison.json',historical)
        if stage=='S2':
            for d in bundles:
                selected=decisions[stage]['datasets'][d]['selected']
                reference[d]=dict(**selected,values=selected_values(current,d,selected['trial'],selected['kind']))
        if stage=='S3':
            calibration=[dict(dataset=r['dataset'],**row) for r in current if r['trial']=='S95' for row in r['calibration']]
            p['thresholds']={}
            for d in bundles:
                p['thresholds'][d]={f'q{q}':{g:float(np.quantile([row['u'] for row in calibration if row['dataset']==d and row['group']==g],q/100,method='linear'))
                    for g in ('E','P')} for q in (50,75)}
                assert sum(row['dataset']==d for row in calibration)==2*2*p['datasets'][d]['steps_per_epoch']
            write_json(out/'thresholds_frozen.json',dict(source='S95 E1 training only; seeds 2022,2024',
                statistics=calibration,thresholds=p['thresholds'],test_accessed=False))
        print(f'{stage} COMPLETE '+json.dumps({d:{k:v for k,v in record.items() if k in ("selected","promoted")}
            for d,record in decisions[stage]['datasets'].items()},ensure_ascii=False),flush=True)
    frozen={};formal_jobs=[]
    for d in bundles:
        a0=decisions['S1']['datasets'][d]['selected'];a1=decisions['S2']['datasets'][d]['selected']
        frozen[d]={'A0':dict(config=configs['S1'][d][a0['trial']],kind=a0['kind']),
                   'A1':dict(config=configs['S2'][d][a1['trial']],kind=a1['kind'])}
        aig=decisions['S4']['datasets'][d]['selected']
        if aig:frozen[d]['AIG']=dict(config=configs['S4'][d][aig['trial']],kind=aig['kind'])
        for trial,item in frozen[d].items():
            for seed in p['seeds']:formal_jobs.append(dict(stage='S5',dataset=d,trial=trial,config=item['config'],seed=seed))
    write_json(out/'formal_configuration_frozen.json',dict(configurations=frozen,execution_commit=p['execution_commit'],
        code_hash=p['source_manifest_hash'],data_hashes={d:digest(p['datasets'][d]) for d in bundles},
        jobs=formal_jobs,thresholds=p['thresholds'],test_accessed=False))
    cache={r['run_hash']:r for r in results};pending=[];formal=[];cache_count=0
    for job in formal_jobs:
        key=digest(run_identity(job['dataset'],job['seed'],job['config'],p))
        if key not in cache:pending.append(job);continue
        r=cache[key];assert r['epochs']==50 and r['run_identity']==run_identity(job['dataset'],job['seed'],job['config'],p)
        cache_count+=1;append_csv(out/'trial_ledger.csv',ledger_row(p,job,'CACHE_REUSED',r['attempt'],r,r['output_dir']))
        formal.append(dict(result=r,kinds=([('A0','online'),('A0_EMA',frozen[job['dataset']]['A0']['kind'])] if job['trial']=='A0'
            else [(job['trial']+'_EMA',frozen[job['dataset']][job['trial']]['kind'])])))
    additional=schedule(bundles,p,out,pending,p['devices']) if pending else [];results.extend(additional)
    for r in additional:
        formal.append(dict(result=r,kinds=([('A0','online'),('A0_EMA',frozen[r['dataset']]['A0']['kind'])] if r['trial']=='A0'
            else [(r['trial']+'_EMA',frozen[r['dataset']][r['trial']]['kind'])])))
    for d in bundles:
        for seed in p['seeds']:
            paired=[r for r in results if r['dataset']==d and r['seed']==seed]
            assert all(r['init_hash']==paired[0]['init_hash'] and r['epoch_order_hashes']==paired[0]['epoch_order_hashes'] for r in paired)
    assert len(formal)==len(formal_jobs) and len(results)==44+len(formal_jobs)-cache_count
    unified_test(bundles,p,out,formal);aggregate(results,out);build_tables(out)
    write_json(out/'completion.json',dict(status='COMPLETE',development_unique_runs=44,formal_online_outputs=len(formal),
        formal_cache_reuses=cache_count,unique_training_runs=len(results),nominal_without_cache=44+len(formal),
        total_epochs=sum(r['epochs'] for r in results),optimizer_steps=sum(r['global_step'] for r in results),
        logical_formal_model_seed_outputs=sum(len(x['kinds']) for x in formal),test_evaluated=True,
        wall_seconds=time.perf_counter()-start,snapshot_tensor_bytes=sum(r['snapshot_tensor_bytes'] for r in results)))
    for r in results:r.pop('states')
    formal.clear();results.clear();gc.collect()
    write_json(out/'state_release.json',dict(released=True,storage='RAM_ONLY',after='unified test and tables'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['audit','check','all'],required=True)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--devices',default='0,1');parser.add_argument('--code-commit');parser.add_argument('--stage0-receipt')
    args=parser.parse_args();p=protocol();out=Path(args.run_dir);manifest=source_manifest()
    if args.mode in ('check','all') and not args.code_commit:parser.error('Provide the reviewed immutable --code-commit')
    if args.mode=='all':
        if not args.stage0_receipt:parser.error('Provide the PASS --stage0-receipt from this code manifest')
        receipt=json.loads(Path(args.stage0_receipt).read_text())
        assert receipt['status']=='PASS' and receipt['source_manifest_hash']==manifest['hash']
        assert receipt['execution_commit']==args.code_commit
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new attempt: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    p.update(execution_commit=args.code_commit,source_manifest_hash=manifest['hash'],devices=[int(v) for v in args.devices.split(',')])
    write_json(out/'source_manifest.json',dict(execution_commit=args.code_commit,**manifest));write_json(out/'resolved_config.json',p)
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,gpus=gpu_status(),
        tf32=False,cudnn_benchmark=False,cpu_threads=1,training_gpu_event_boundary='per batch input transfer through online and shadow updates; includes host gaps',
        evaluation_gpu_event_boundary='sum of per-batch model forward event intervals, excludes host metric calculation'))
    templates={}
    for stage in TRIAL_ORDER:
        templates[stage]={}
        for d in p['datasets']:
            threshold_recipe={f'q{q}':{g:f'S95.E1.train.seeds2022_2024.{g}.q{q}' for g in ('E','P')} for q in (50,75)}
            templates[stage][d]=candidates(stage,d,p,thresholds=threshold_recipe)
            if stage!='S1':
                for c in templates[stage][d].values():c.update(lr='S1.selected.lr',weight_decay='S1.selected.weight_decay')
    write_json(out/'trial_plan.json',dict(development_seeds=DEV_SEEDS,seeds=p['seeds'],epochs=50,batch_size=40960,
        stages=TRIAL_ORDER,base_configs={d:candidates('S1',d,p) for d in p['datasets']},
        finite_candidate_templates=templates,template_hash=digest(templates),
        resolution='template_hash identifies the recipe; actual config_hash is written only after S1/threshold resolution',
        schema=list(ledger_row(p,dict(stage='S1',dataset='movielens',trial='C',seed=2022,config=candidates('S1','movielens',p)['C']),'PLANNED'))))
    bundles={};manifests={}
    for d in p['datasets']:
        destination=out/'data'/d;destination.mkdir(parents=True,exist_ok=True)
        bundles[d],manifests[d]=load_dataset(d,p,out=destination);print(f'data identity PASS {d}',flush=True)
    write_json(out/'data_manifest.json',manifests)
    if args.mode=='audit':return
    if args.mode=='check':
        from .checks import run_checks
        result=run_checks(bundles,p,args.device,out)
        write_json(out/'stage0_checks.json',dict(**result,source_manifest_hash=manifest['hash'],execution_commit=args.code_commit));return
    try:all_stages(bundles,p,out)
    except Exception as exc:
        write_json(out/'completion.json',dict(status='BLOCKED',error=repr(exc)));raise


if __name__=='__main__':main()
