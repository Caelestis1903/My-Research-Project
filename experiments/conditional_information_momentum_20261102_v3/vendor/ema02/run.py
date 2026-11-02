"""Explicit reviewed dispatch, detached workers and one final frozen evaluation."""
import time
ENTRY_START=time.perf_counter()
import argparse
import copy
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
from .protocol import (load_protocol, config, make_optimizer, setup, cpu_copy, state_digest,
    rng_state, restore_rng, write_json)
from .vendor.gradient_aug_20261031_v2.views import make_model
from .data import load_dataset
from .evaluate import evaluate
from .guardian import request, ensure_guardian, alive
from .objectives import train_step

HERE=Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_manifest():
    rows=[dict(path=str(f.relative_to(HERE)),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
          for f in sorted(HERE.rglob('*')) if f.suffix in ('.py','.json')]
    return dict(files=rows,sha256=digest(rows))


def review_check(path):
    r=json.loads(Path(path).read_text())
    assert r['verdict'] in ('PASS','PASS WITH FIXES') and not r.get('blocking_issues'), 'Independent review not passed'
    assert r['source_hash']==source_manifest()['sha256'], 'Reviewed source identity changed'
    return r


def job(dataset,arm,seed,stage,split_id='original',kappa=None):
    assert seed in range(2022,2027)
    c=config(arm,dataset,kappa=kappa)
    record=dict(dataset=dataset,seed=seed,stage=stage,split_id=split_id,config=c,
                source_hash=source_manifest()['sha256'])
    record['key']=digest({k:v for k,v in record.items() if k!='stage'})
    return record


def initial_dispatch():
    return dict(stage='S1',approved_by='orchestrator',path_choice=None,jobs=[job(d,a,s,'S1')
        for s in (2022,2024) for d in ('movielens','kuairand') for a in ('A0','A1','J0','J1','F0','F1')])


def gpu_status():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [dict(zip(('index','memory','total','power','limit'),[float(v.strip()) for v in line.split(',')]))
            for line in p.stdout.strip().splitlines()]


def choose_device(status,devices,peak_bytes):
    for r in status:
        if int(r['index']) in devices and r['memory']/r['total']<.85 and r['power']/r['limit']<.85:
            if (r['total']-r['memory'])*1024**2 >= peak_bytes:
                return int(r['index'])
    return None


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0


def _jsonl(path,record):
    with Path(path).open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')


def worker(attempt,key,device):
    attempt=Path(attempt);j=json.loads((attempt/'jobs'/f'{key}.json').read_text())
    assert j['source_hash']==source_manifest()['sha256']
    registration=request(attempt,'register',key=key,job=j,pid=os.getpid())
    number=registration['attempt'];state=registration['state'];out=attempt/'runs'/key/f'attempt{number}'
    out.mkdir(parents=True,exist_ok=True);start=ENTRY_START;committed_seconds=state['physical_worker_seconds'] if state else 0.
    try:
        faulthandler.enable(all_threads=True)
        p=load_protocol();bundle,manifest=load_dataset(j['dataset'],j['split_id'],p)
        write_json(out/'data_manifest.json',manifest)
        c=j['config'];setup(j['seed'],device)
        model=make_model(bundle,p['datasets'][j['dataset']]['source']).to(device)
        ema=copy.deepcopy(model).requires_grad_(False);opt=make_optimizer(model,c)
        loader=torch.Generator().manual_seed(j['seed']+11000)
        selected={};selected_epochs={k:None for k in ('online','fixed_ema')}
        selected_validation={};best={k:-float('inf') for k in selected_epochs};counts={};step=0;epoch0=0
        if state:
            model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer'])
            loader.set_state(state['loader']);restore_rng(state['rng'],device)
            epoch0=state['epoch'];step=state['global_step'];selected=state['selected_models']
            selected_epochs=state['selected_epochs'];selected_validation=state['selected_validation']
            best=state['best'];counts=state['counts'];validation=state['validation']
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(epoch0+1,51):
            tick=time.perf_counter();model.train();order=torch.randperm(len(bundle.train_y),generator=loader).numpy()
            loss_sum=real_sum=0.;epoch_counts={};epoch_diagnostics=[]
            for batch,offset in enumerate(range(0,len(order),40960),1):
                ids=order[offset:offset+40960];step+=1
                x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
                left,right=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);left.record()
                value=train_step(model,opt,ema,x,y,bundle.train_row_ids[ids],c,
                    dict(study='EMA02V2',dataset=j['dataset'],split_id=j['split_id'],seed=j['seed'],epoch=epoch,global_step=step),
                    diagnostic=batch<=2)
                right.record();right.synchronize();n=len(ids)
                loss_sum+=value['loss']*n;real_sum+=value['real_loss']*n
                for name,v in value['counts'].items():
                    counts[name]=counts.get(name,0)+v;epoch_counts[name]=epoch_counts.get(name,0)+v
                counts['train_gpu_event_ms']=counts.get('train_gpu_event_ms',0.)+left.elapsed_time(right)
                record=dict(epoch=epoch,batch=batch,step=step,rows=n,loss=value['loss'],real_loss=value['real_loss'],
                    worker_seconds=time.perf_counter()-start,counts=value['counts'],diagnostics=value['diagnostics'])
                _jsonl(out/'batch_trace.jsonl',record)
                if batch<=2:epoch_diagnostics.append(value['diagnostics'])
                if epoch==epoch0+1 and batch==1:
                    request(attempt,'allocated',key=key,peak_memory_bytes=torch.cuda.max_memory_reserved(device))
                del x,y,value
            validation={};validation_cost={}
            for kind,estimator in [('online',model),('fixed_ema',ema)]:
                values,cost=evaluate(estimator,bundle.val_x,bundle.val_y,device)
                validation[kind]=values;validation_cost[kind]=cost
                auc=values['macro_auc']
                if auc is not None and auc>best[kind]:
                    best[kind]=auc;selected_epochs[kind]=epoch
                    selected[kind]=cpu_copy(estimator.state_dict());selected_validation[kind]=values
                counts['validation_forwards']=counts.get('validation_forwards',0)+cost['forwards']
                counts['validation_rows']=counts.get('validation_rows',0)+cost['rows']
                counts['validation_seconds']=counts.get('validation_seconds',0.)+cost['seconds']
            epoch_record=dict(key=key,epoch=epoch,global_step=step,loss=loss_sum/len(order),real_loss=real_sum/len(order),
                training_and_validation_seconds=time.perf_counter()-tick,validation=validation,validation_cost=validation_cost,
                selected_epochs=selected_epochs,model_hash=state_digest(model.state_dict()),ema_hash=state_digest(ema.state_dict()),
                optimizer_hash=state_digest(opt.state_dict()),epoch_order_hash=state_digest(bundle.train_row_ids[order]),
                counts=epoch_counts,diagnostics=epoch_diagnostics)
            epoch_file=out/f'epoch_{epoch:02d}.json';write_json(epoch_file,epoch_record)
            snapshot=cpu_copy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),rng=rng_state(device),
                loader=loader.get_state(),epoch=epoch,global_step=step,selected_models=selected,selected_epochs=selected_epochs,
                selected_validation=selected_validation,best=best,counts=counts,validation=validation,
                physical_worker_seconds=committed_seconds+time.perf_counter()-start,
                job_key=key,scalar_epoch_file=str(epoch_file.relative_to(attempt))))
            ack=request(attempt,'commit',key=key,state=snapshot,epoch_file=str(epoch_file.relative_to(attempt)))
            assert ack==dict(status='RAM_COMMITTED',epoch=epoch)
            write_json(out/'progress.json',dict(epoch=epoch,global_step=step,status='RAM_COMMITTED',key=key))
            del snapshot
        result=dict(job=j,epochs=50,global_step=step,selected_epochs=selected_epochs,
            selected_validation=selected_validation,E50_validation=validation,
            selected_hashes={k:state_digest(v) for k,v in selected.items()},
            E50_hashes=dict(online=state_digest(model.state_dict()),fixed_ema=state_digest(ema.state_dict())),
            optimizer_hash=state_digest(opt.state_dict()),counts=counts,
            worker_seconds=committed_seconds+time.perf_counter()-start,
            attempt_worker_seconds=time.perf_counter()-start,attempt=number,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        request(attempt,'complete',key=key,result=result)
        write_json(out/'physical_cost.json',dict(key=key,attempt=number,status='COMPLETE',
            worker_seconds=time.perf_counter()-start,committed_epoch=50,counts=counts))
    except BaseException as error:
        failure=dict(key=key,attempt=number,error=repr(error),traceback=traceback.format_exc(),
                     worker_seconds=time.perf_counter()-start,oom=isinstance(error,torch.OutOfMemoryError))
        write_json(out/'failure.json',failure)
        request(attempt,'failed',key=key,failure=failure)
        raise


def results(attempt):
    return [json.loads(f.read_text()) for f in sorted((Path(attempt)/'runs').glob('*/result.json'))]


def forecast_for(j,forecasts):
    return forecasts[j['dataset']+'/'+j['config']['arm']]


def budget_admission(cap,spent,active_remaining,next_remaining,final_remaining=0.):
    required=spent+sum(active_remaining)+next_remaining+final_remaining
    return dict(admitted=required<=cap,required_seconds=required,cap_seconds=cap,
                spent_seconds=spent,remaining_active_seconds=sum(active_remaining),
                next_remaining_seconds=next_remaining,remaining_final_seconds=final_remaining)


def remaining_forecast(j,record,forecasts,active=False):
    f=forecast_for(j,forecasts);final=f['final_evaluation_seconds']
    credit=record.get('resume_worker_seconds',0.)+max(0.,time.time()-record['started']) if active else record.get('worker_seconds',0.)
    restart=(f['process_startup_seconds']+f['data_seconds']+f['initialization_seconds']) if record and not active else 0.
    return max(0.,f['expected_worker_seconds']-final-credit)+restart+final


def dispatch_check(attempt,dispatch):
    assert dispatch['approved_by']=='orchestrator'
    jobs=dispatch['jobs'];assert len({j['key'] for j in jobs})==len(jobs)
    p=load_protocol();stage=dispatch['stage'];assert stage in p['stages'] and stage!='S0'
    assert len(jobs)<=p['stages'][stage]['max_trajectories']
    for j in jobs:
        assert j==job(j['dataset'],j['config']['arm'],j['seed'],stage,j['split_id'],j['config']['kappa'])
        assert j['stage']==stage
        assert j['seed'] in p['stages'][stage]['seeds']
    if stage=='S1':assert {j['key'] for j in jobs}=={j['key'] for j in initial_dispatch()['jobs']}
    else:assert len([r for r in results(attempt) if r['job']['stage']=='S1'])==24, 'Initial matrix incomplete'
    ledger=Path(attempt)/'dispatches'
    prior=[json.loads(f.read_text()) for f in ledger.glob('*.json')] if ledger.exists() else []
    choices={d['path_choice'] for d in prior if d.get('path_choice')}
    if choices:assert dispatch.get('path_choice') in choices and len(choices)==1
    if stage in ('S2M','S3M'):assert dispatch['path_choice']=='main'
    if stage=='S2B':assert dispatch['path_choice']=='backup'
    alljobs={j['key'] for d in prior for j in d['jobs']}|{j['key'] for j in jobs}
    assert len(alljobs)<=86


def physical_cost(attempt,status=None):
    attempt=Path(attempt);total=0.
    for f in (attempt/'runs').glob('*/attempt*/physical_cost.json'):
        total+=json.loads(f.read_text())['worker_seconds']
    for f in (attempt/'runs').glob('*/attempt*/failure.json'):
        total+=json.loads(f.read_text())['worker_seconds']
    if status:
        for r in status['jobs'].values():
            if r['status']=='RUNNING' and alive(r.get('worker_pid')):total+=time.time()-r['started']
    for f in (attempt/'test_endpoints').glob('*.json'):
        total+=json.loads(f.read_text()).get('cost',{}).get('seconds',0.)
    for f in (attempt/'evaluation_failures').glob('*.json'):
        total+=json.loads(f.read_text())['worker_seconds']
    return total


def schedule(attempt,dispatch,devices):
    attempt=Path(attempt);ensure_guardian(attempt);dispatch_check(attempt,dispatch)
    forecasts=json.loads((attempt/'cost_forecast.json').read_text())
    stage0=json.loads((attempt/'STAGE0_REPORT.json').read_text());assert stage0['status']=='PASS'
    for j in dispatch['jobs']:forecast_for(j,forecasts)
    dfile=attempt/'dispatches'/f"{dispatch['stage']}_{digest(dispatch)[:12]}.json"
    if not dfile.exists():
        write_json(dfile,dispatch);_jsonl(attempt/'decision_ledger.jsonl',dict(time=time.time(),dispatch_file=str(dfile),**dispatch))
    for j in dispatch['jobs']:write_json(attempt/'jobs'/f"{j['key']}.json",j)
    approved={j['key']:j for f in (attempt/'dispatches').glob('*.json') for j in json.loads(f.read_text())['jobs']}
    reserved=sum(forecast_for(j,forecasts)['expected_worker_seconds'] for j in approved.values())
    s0_cost=sum(r['worker_seconds'] for r in stage0['physical_runs'])
    write_json(attempt/'cost_reservation.json',dict(expected_queue_worker_seconds=reserved,
        physical_cap_seconds=1.20*reserved,stage0_worker_seconds=s0_cost,approved_unique=len(approved)))
    processes={};serial=set();start=time.perf_counter()
    while True:
        status=request(attempt,'status');records=status['jobs']
        for key,proc in list(processes.items()):
            code=proc.poll()
            if code is not None:
                if records.get(key,{}).get('status')=='RUNNING':
                    r=records[key];failure=dict(key=key,attempt=r['attempts'],error=f'Native worker exit {code}',
                        worker_seconds=time.time()-r['started'],exitcode=code,oom=False)
                    write_json(attempt/'runs'/key/f"attempt{r['attempts']}"/'failure.json',failure)
                    request(attempt,'failed',key=key,failure=failure)
                del processes[key]
        status=request(attempt,'status');records=status['jobs']
        pending=[];active=[];failed=[]
        for j in dispatch['jobs']:
            r=records.get(j['key'])
            if r and r['status']=='COMPLETE':
                if j['key'] in processes and processes[j['key']].poll() is None:active.append(j)
                continue
            if r and (alive(r.get('worker_pid')) or (j['key'] in processes and processes[j['key']].poll() is None)):
                active.append(j);continue
            if r and r['status']=='RUNNING':
                failure=dict(key=j['key'],attempt=r['attempts'],error='Worker disappeared during controller absence',
                    worker_seconds=time.time()-r['started'],oom=False)
                write_json(attempt/'runs'/j['key']/f"attempt{r['attempts']}"/'failure.json',failure)
                request(attempt,'failed',key=j['key'],failure=failure)
            if r and r['attempts']>=2:failed.append(j);continue
            if r:serial.add(j['key'])
            pending.append(j)
        spent=s0_cost+physical_cost(attempt,status);cap=1.20*reserved
        write_json(attempt/'progress.json',dict(stage=dispatch['stage'],complete=sum(
            records.get(j['key'],{}).get('status')=='COMPLETE' for j in dispatch['jobs']),total=len(dispatch['jobs']),
            active=[dict(key=j['key'],epoch=records[j['key']]['epoch']) for j in active],pending=len(pending),
            failed=len(failed),physical_worker_seconds=spent,physical_cap_seconds=1.20*reserved,
            queue_wall_seconds=time.perf_counter()-start,test_accessed=False,
            budget_status='ACTIVE_OVERRUN' if spent>cap and active else 'OVER_CAP' if spent>cap else 'WITHIN_CAP'))
        if not pending and not active:
            if failed:raise RuntimeError('BUDGET_STOP_UNMATCHED: required jobs exhausted retry')
            if spent>cap:
                export_tables(attempt)
                raise RuntimeError('BUDGET_STOP_UNMATCHED: completed active work exceeded frozen cap')
            break
        if spent>=1.20*reserved and pending:
            if not active:raise RuntimeError('BUDGET_STOP_UNMATCHED: physical cost cap reached')
            time.sleep(1);continue
        allocating=any(not records[j['key']].get('allocated') for j in active)
        if pending and not allocating and available_ram()>6*1024**3:
            chosen=pending[0]
            final_remaining=sum(forecast_for(approved[k],forecasts)['final_evaluation_seconds']
                for k,r in records.items() if k in approved and r['status']=='COMPLETE')
            final_remaining=max(0.,final_remaining-sum(json.loads(f.read_text()).get('cost',{}).get('seconds',0.)
                for f in (attempt/'test_endpoints').glob('*.json')))
            admission=budget_admission(cap,spent,
                [remaining_forecast(j,records[j['key']],forecasts,True) for j in active],
                remaining_forecast(chosen,records.get(chosen['key'],{}),forecasts),final_remaining)
            write_json(attempt/'budget_admission.json',dict(time=time.time(),key=chosen['key'],**admission))
            if not admission['admitted']:
                if not active:
                    export_tables(attempt)
                    raise RuntimeError('BUDGET_STOP_UNMATCHED: frozen remaining-work reservation does not fit')
                time.sleep(.5);continue
            if chosen['key'] in serial and active:time.sleep(.5);continue
            forecast=forecast_for(chosen,forecasts)
            gpu=choose_device(gpu_status(),devices,forecast['peak_reserved_bytes'])
            if gpu is not None:
                logdir=attempt/'launch_logs';logdir.mkdir(exist_ok=True)
                with (logdir/(chosen['key']+'.log')).open('ab') as log:
                    proc=subprocess.Popen([sys.executable,'-m','experiments.ema_view_design_20261102_v2.run','_worker','--attempt',str(attempt),
                        '--key',chosen['key'],'--device',f'cuda:{gpu}'],stdin=subprocess.DEVNULL,
                        cwd=HERE.parents[1],stdout=log,stderr=log,start_new_session=True)
                processes[chosen['key']]=proc
                _jsonl(attempt/'launch_ledger.jsonl',dict(key=chosen['key'],pid=proc.pid,gpu=gpu,time=time.time(),
                    gpu_status=gpu_status(),serial_retry=chosen['key'] in serial))
                # Wait until this worker registers, so the next poll cannot dispatch it twice.
                while proc.poll() is None:
                    registered=request(attempt,'status')['jobs'].get(chosen['key'],{})
                    if registered.get('worker_pid')==proc.pid:break
                    time.sleep(.2)
                if proc.poll() is not None and not registered.get('worker_pid')==proc.pid:
                    raise RuntimeError('Worker failed before registration; see launch_logs/'+chosen['key']+'.log')
        time.sleep(.5)
    export_tables(attempt)
    return results(attempt)


def freeze(attempt,decision):
    attempt=Path(attempt);status=request(attempt,'status');completed=results(attempt)
    assert not any(alive(r.get('worker_pid')) for r in status['jobs'].values()), 'Training still active'
    assert decision['approved_by']=='orchestrator' and decision['training_complete']
    assert not (attempt/'selection_frozen.json').exists(), 'Final freeze already exists'
    entries=[]
    for r in completed:
        state=request(attempt,'get',key=r['job']['key'])['state']
        assert state is not None, 'EVAL_STATE_LOST'
        for kind in ('online','fixed_ema'):
            for checkpoint in ('selected','E50'):
                h=r['selected_hashes'].get(kind) if checkpoint=='selected' else r['E50_hashes'][kind]
                epoch=r['selected_epochs'][kind] if checkpoint=='selected' else 50
                entries.append(dict(key=r['job']['key'],job=r['job'],model_kind=kind,checkpoint=checkpoint,epoch=epoch,model_hash=h))
    frozen=dict(source_hash=source_manifest()['sha256'],time=time.time(),decision=decision,endpoints=entries,
                validation_only_selection=True,test_opened=False)
    write_json(attempt/'selection_frozen.json',frozen)
    return frozen


def final_evaluate(attempt,devices):
    attempt=Path(attempt);frozen=json.loads((attempt/'selection_frozen.json').read_text())
    assert frozen['source_hash']==source_manifest()['sha256']
    p=load_protocol();rows=[];cache={};device=None
    reservation=json.loads((attempt/'cost_reservation.json').read_text())
    forecasts=json.loads((attempt/'cost_forecast.json').read_text())
    for entry in frozen['endpoints']:
        output=attempt/'test_endpoints'/f"{entry['key']}_{entry['model_kind']}_{entry['checkpoint']}.json"
        if output.exists():rows.append(json.loads(output.read_text()));continue
        endpoint_start=time.perf_counter()
        s0_cost=reservation['stage0_worker_seconds'];spent=s0_cost+physical_cost(attempt)
        remaining=[e for e in frozen['endpoints'] if e['model_hash'] is not None and not
            (attempt/'test_endpoints'/f"{e['key']}_{e['model_kind']}_{e['checkpoint']}.json").exists()]
        predicted=sum(forecast_for(e['job'],forecasts)['final_evaluation_seconds']/4 for e in remaining)
        if not budget_admission(reservation['physical_cap_seconds'],spent,[],predicted)['admitted']:
            write_json(attempt/'evaluation_budget_stop.json',dict(status='BUDGET_STOP_UNMATCHED',spent_seconds=spent,
                remaining_evaluation_forecast_seconds=predicted,cap_seconds=reservation['physical_cap_seconds']))
            export_tables(attempt)
            raise RuntimeError('BUDGET_STOP_UNMATCHED: final evaluation reservation does not fit')
        if entry['model_hash'] is None:
            row=dict(**entry,status='INCONCLUSIVE_WINDOW',metrics=None)
        else:
            j=entry['job'];ident=(j['dataset'],j['split_id'])
            if ident not in cache:cache[ident]=load_dataset(*ident,p=p)[0]
            bundle=cache[ident]
            state=request(attempt,'get',key=entry['key'])['state'];assert state is not None,'EVAL_STATE_LOST'
            model_state=(state['selected_models'][entry['model_kind']] if entry['checkpoint']=='selected'
                else state['model' if entry['model_kind']=='online' else 'ema'])
            assert state_digest(model_state)==entry['model_hash']
            while device is None:
                gpu=choose_device(gpu_status(),devices,2*1024**3)
                if gpu is not None:device=f'cuda:{gpu}'
                else:time.sleep(1)
            setup(j['seed'],device);model=make_model(bundle,p['datasets'][j['dataset']]['source']).to(device)
            model.load_state_dict(model_state);values,cost=evaluate(model,bundle.test_x,bundle.test_y,device)
            cost['metric_evaluation_seconds']=cost['seconds']
            cost['seconds']=time.perf_counter()-endpoint_start
            cost['timing_scope']='endpoint_data_hash_load_IPC_setup_model_and_metrics'
            row=dict(**entry,status='COMPLETE',role='test' if j['split_id']=='original' else 'confirm_W1',metrics=values,cost=cost)
            del model;torch.cuda.empty_cache()
        write_json(output,row);rows.append(row)
    write_json(attempt/'final_evaluation.json',dict(status='COMPLETE',endpoints=len(rows),rows=rows))
    export_tables(attempt)
    final_spent=reservation['stage0_worker_seconds']+physical_cost(attempt)
    if final_spent>reservation['physical_cap_seconds']:
        write_json(attempt/'evaluation_budget_stop.json',dict(status='BUDGET_STOP_UNMATCHED',
            spent_seconds=final_spent,cap_seconds=reservation['physical_cap_seconds'],all_endpoint_metrics_retained=True))
        write_json(attempt/'final_evaluation.json',dict(status='BUDGET_STOP_UNMATCHED',evaluation_complete=True,
            endpoints=len(rows),rows=rows,physical_spent_seconds=final_spent))
        raise RuntimeError('BUDGET_STOP_UNMATCHED: active final evaluation exceeded frozen cap')
    for key in {r['key'] for r in rows}:request(attempt,'release',key=key)
    return rows


def export_tables(attempt):
    import csv
    import pandas as pd
    from .gates import paired
    attempt=Path(attempt);complete=results(attempt);epochs=[];endpoints=[];costs=[]
    commits=attempt/'epoch_commit_ledger.jsonl'
    for line in commits.read_text().splitlines() if commits.exists() else []:
        c=json.loads(line);record=json.loads((attempt/c['epoch_file']).read_text())
        j=json.loads((attempt/'jobs'/f"{c['key']}.json").read_text())
        for kind,values in record['validation'].items():
            epochs.append(dict(key=c['key'],dataset=j['dataset'],split_id=j['split_id'],arm=j['config']['arm'],
                kappa=j['config']['kappa'],seed=j['seed'],epoch=record['epoch'],model_kind=kind,
                loss=record['loss'],real_loss=record['real_loss'],**values))
    for r in complete:
        j=r['job']
        for kind in ('online','fixed_ema'):
            for point,values in [('selected',r['selected_validation'].get(kind)),('E50',r['E50_validation'][kind])]:
                if values is None:continue
                endpoints.append(dict(key=j['key'],dataset=j['dataset'],split_id=j['split_id'],arm=j['config']['arm'],
                    kappa=j['config']['kappa'],seed=j['seed'],model_kind=kind,checkpoint=point,
                    epoch=r['selected_epochs'][kind] if point=='selected' else 50,
                    model_hash=r['selected_hashes'][kind] if point=='selected' else r['E50_hashes'][kind],**values))
    tests=[]
    for f in (attempt/'test_endpoints').glob('*.json'):
        r=json.loads(f.read_text());j=r['job']
        tests.append(dict(key=r['key'],dataset=j['dataset'],split_id=j['split_id'],arm=j['config']['arm'],
            kappa=j['config']['kappa'],seed=j['seed'],model_kind=r['model_kind'],checkpoint=r['checkpoint'],
            epoch=r['epoch'],model_hash=r['model_hash'],status=r['status'],**(r['metrics'] or {})))
    s0=attempt/'STAGE0_REPORT.json'
    if s0.exists():costs.extend(json.loads(s0.read_text())['physical_runs'])
    for f in (attempt/'runs').glob('*/attempt*/*.json'):
        if f.name not in ('failure.json','physical_cost.json'):continue
        r=json.loads(f.read_text());j=json.loads((attempt/'jobs'/f"{r['key']}.json").read_text())
        costs.append(dict(key=r['key'],dataset=j['dataset'],arm=j['config']['arm'],seed=j['seed'],
            attempt=r['attempt'],kind='training',status=r.get('status','FAILED'),worker_seconds=r['worker_seconds'],
            committed_epoch=r.get('committed_epoch'),error=r.get('error')))
    final_cost=sum(json.loads(f.read_text()).get('cost',{}).get('seconds',0.) for f in (attempt/'test_endpoints').glob('*.json'))
    if final_cost:costs.append(dict(kind='final_evaluation',worker_seconds=final_cost))
    for f in (attempt/'evaluation_failures').glob('*.json'):
        costs.append(dict(kind='final_evaluation_partial',**json.loads(f.read_text())))
    for name,rows in [('epoch_metrics.csv',epochs),('validation_metrics.csv',endpoints),('test_metrics.csv',tests),
                      ('physical_cost_ledger.csv',costs)]:
        pd.DataFrame(rows).to_csv(attempt/name,index=False)
    runs=[dict(key=r['job']['key'],dataset=r['job']['dataset'],split_id=r['job']['split_id'],arm=r['job']['config']['arm'],
        kappa=r['job']['config']['kappa'],seed=r['job']['seed'],stage=r['job']['stage'],status='COMPLETE',epochs=50,
        selected_online_epoch=r['selected_epochs']['online'],selected_ema_epoch=r['selected_epochs']['fixed_ema'],
        worker_seconds=r['worker_seconds'],**r['counts']) for r in complete]
    pd.DataFrame(runs).to_csv(attempt/'run_ledger.csv',index=False)
    trials=[]
    for f in (attempt/'dispatches').glob('*.json'):
        for j in json.loads(f.read_text())['jobs']:
            trials.append(dict(key=j['key'],dataset=j['dataset'],split_id=j['split_id'],stage=j['stage'],
                arm=j['config']['arm'],kappa=j['config']['kappa'],seed=j['seed'],
                status='COMPLETE' if any(r['key']==j['key'] for r in runs) else 'INCOMPLETE'))
    pd.DataFrame(trials).drop_duplicates('key').to_csv(attempt/'trial_ledger.csv',index=False) if trials else None
    pd.DataFrame([json.loads(f.read_text()) for f in (attempt/'runs').glob('*/attempt*/failure.json')]).to_csv(
        attempt/'failure_ledger.csv',index=False)
    pairs=[]
    for role,records in [('validation',endpoints),('test',tests)]:
        frame=pd.DataFrame(records)
        if not len(frame):continue
        dimensions=['dataset','split_id','model_kind','checkpoint']
        for labels,group in frame.groupby(dimensions):
            for (arm,kappa),c in group.groupby(['arm','kappa']):
                for ref in group['arm'].unique():
                    if ref==arm:continue
                    r=group[group['arm']==ref]
                    if ref not in ('A0','A1'):r=r[r['kappa']==kappa]
                    if len(r)!=r.seed.nunique():continue
                    for seeds,phase in [((2022,2024),'development'),((2023,2025,2026),'held'),(tuple(range(2022,2027)),'combined')]:
                        a={int(v['seed']):v for v in c.to_dict('records') if v['seed'] in seeds}
                        b={int(v['seed']):v for v in r.to_dict('records') if v['seed'] in seeds}
                        for metric in [n for n in group.columns if n.endswith(('_auc','_logloss'))]:
                            if not a or not b:continue
                            if any(v.get(metric) is None or pd.isna(v.get(metric)) for v in list(a.values())+list(b.values())):continue
                            effect=paired(a,b,metric)
                            if not effect['n']:continue
                            pairs.append(dict(role=role,**dict(zip(dimensions,labels)),arm=arm,kappa=kappa,reference=ref,
                                phase=phase,metric=metric,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in effect.items()}))
    pd.DataFrame(pairs).to_csv(attempt/'paired_effects.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=[
        'stage0','run','attach','status','initial-dispatch','recommend-next','freeze','final-evaluate','shutdown','source-manifest','_guardian','_worker'])
    parser.add_argument('--attempt',type=Path);parser.add_argument('--review',type=Path)
    parser.add_argument('--dispatch',type=Path);parser.add_argument('--decision',type=Path)
    parser.add_argument('--devices',default='0,1');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--arms',default='A0,A1,J0,J1,F0,F1');parser.add_argument('--key')
    args=parser.parse_args();devices=[int(v) for v in args.devices.split(',')]
    if args.mode=='source-manifest':print(json.dumps(source_manifest(),indent=2));return
    if args.mode=='initial-dispatch':print(json.dumps(initial_dispatch(),indent=2));return
    assert args.attempt is not None
    if args.mode=='_guardian':
        from .guardian import serve
        serve(args.attempt);return
    if args.mode=='_worker':worker(args.attempt,args.key,args.device);return
    if args.mode in ('stage0','run','attach','final-evaluate'):
        assert args.review is not None;review_check(args.review)
    if args.mode=='stage0':
        from .stage0 import execute
        report_path=args.attempt/'STAGE0_REPORT.json'
        prior=json.loads(report_path.read_text()) if report_path.exists() else dict(physical_runs=[])
        prior_seconds=sum(r['worker_seconds'] for r in prior['physical_runs'])
        try:
            report=execute(args.attempt,args.device,tuple(args.arms.split(',')),time.perf_counter()-ENTRY_START)
        except BaseException as error:
            current=json.loads(report_path.read_text()) if report_path.exists() else dict(
                measurements={},data={},cost_forecast={},physical_runs=[])
            completed_seconds=sum(r['worker_seconds'] for r in current['physical_runs'])-prior_seconds
            elapsed=time.perf_counter()-ENTRY_START
            extra=max(0.,elapsed-completed_seconds)
            current['physical_runs'].append(dict(kind='S0_FAILED',worker_seconds=extra,error=repr(error)))
            current['status']='FAIL';write_json(report_path,current)
            _jsonl(args.attempt/'stage0_invocations.jsonl',dict(time=time.time(),status='FAIL',
                elapsed_worker_seconds=elapsed,completed_arm_seconds=completed_seconds,
                additional_charged_seconds=extra,error=repr(error)))
            raise
        elapsed=time.perf_counter()-ENTRY_START
        completed_seconds=sum(r['worker_seconds'] for r in report['physical_runs'])-prior_seconds
        extra=max(0.,elapsed-completed_seconds)
        report['physical_runs'].append(dict(kind='S0_INVOCATION_OVERHEAD',worker_seconds=extra))
        write_json(report_path,report)
        _jsonl(args.attempt/'stage0_invocations.jsonl',dict(time=time.time(),status='PASS',
            elapsed_worker_seconds=elapsed,completed_arm_seconds=completed_seconds,additional_charged_seconds=extra))
        print(json.dumps(dict(status=report['status'],measurements=len(report['measurements']))));return
    if args.mode in ('run','attach'):
        assert args.dispatch is not None
        schedule(args.attempt,json.loads(args.dispatch.read_text()),devices);return
    if args.mode=='status':
        status=request(args.attempt,'status')
        print(json.dumps(dict(guardian_pid=status['pid'],jobs={k:{n:r.get(n) for n in
            ('status','epoch','worker_pid','attempts','allocated','worker_seconds')} for k,r in status['jobs'].items()}),indent=2));return
    if args.mode=='recommend-next':
        from .gates import development_summary,s1r_recommendation
        r=results(args.attempt);p=load_protocol()
        print(json.dumps(dict(S1R=s1r_recommendation(r,p['datasets']),development=development_summary(r,p)),indent=2));return
    if args.mode=='freeze':
        assert args.decision is not None
        print(json.dumps(freeze(args.attempt,json.loads(args.decision.read_text())),indent=2));return
    if args.mode=='final-evaluate':
        before=physical_cost(args.attempt);started=time.perf_counter()
        try:final_evaluate(args.attempt,devices)
        except BaseException as error:
            charged=physical_cost(args.attempt)-before
            write_json(args.attempt/'evaluation_failures'/f'{time.time_ns()}.json',dict(
                worker_seconds=max(0.,time.perf_counter()-started-charged),error=repr(error)))
            export_tables(args.attempt)
            raise
        return
    if args.mode=='shutdown':print(request(args.attempt,'shutdown'))


if __name__=='__main__':main()
