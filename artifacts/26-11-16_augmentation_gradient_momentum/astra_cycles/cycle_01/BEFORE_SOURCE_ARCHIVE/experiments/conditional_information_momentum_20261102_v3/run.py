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
from .protocol import make_model
from .protocol import load_dataset
from .protocol import evaluate
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


def job(dataset,arm,seed,stage,split_id='original',lambda_pair=0.):
    assert seed in range(2022,2027)
    c=config(arm,dataset,lambda_pair=lambda_pair)
    record=dict(dataset=dataset,seed=seed,stage=stage,split_id=split_id,config=c,
                source_hash=source_manifest()['sha256'])
    record['key']=digest({k:v for k,v in record.items() if k!='stage'})
    return record


def initial_dispatch():
    return dict(stage='S1',approved_by='orchestrator',jobs=[job(d,a,s,'S1',lambda_pair=lam)
        for s in (2022,2024) for d in ('movielens','kuairand')
        for a,lam in [('A0',0.),('A1',0.),('P00',.05),('P00',.2),('P11',.05),('P11',.2)]])


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
        e10={};e10_validation={};selected={};selected_epochs={k:None for k in ('online','fixed_ema')}
        selected_validation={};best={k:-float('inf') for k in selected_epochs};counts={};step=0;epoch0=0
        if state:
            model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer'])
            loader.set_state(state['loader']);restore_rng(state['rng'],device)
            epoch0=state['epoch'];step=state['global_step'];selected=state['selected_models']
            selected_epochs=state['selected_epochs'];selected_validation=state['selected_validation']
            e10=state['E10_models'];e10_validation=state['E10_validation'];best=state['best'];counts=state['counts'];validation=state['validation']
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
                if epoch==10:e10[kind]=cpu_copy(estimator.state_dict());e10_validation[kind]=values
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
                E10_models=e10,E10_validation=e10_validation,parameter_manifest=opt.parameter_manifest,source_hash=j['source_hash'],data_manifest=manifest,
                physical_worker_seconds=committed_seconds+time.perf_counter()-start,
                job_key=key,scalar_epoch_file=str(epoch_file.relative_to(attempt))))
            ack=request(attempt,'commit',key=key,state=snapshot,epoch_file=str(epoch_file.relative_to(attempt)))
            assert ack==dict(status='RAM_COMMITTED',epoch=epoch)
            write_json(out/'progress.json',dict(epoch=epoch,global_step=step,status='RAM_COMMITTED',key=key))
            del snapshot
        result=dict(job=j,epochs=50,global_step=step,selected_epochs=selected_epochs,
            selected_validation=selected_validation,E50_validation=validation,E10_validation=e10_validation,
            E10_hashes={k:state_digest(v) for k,v in e10.items()},parameter_manifest=opt.parameter_manifest,
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


def schedule(attempt,dispatch,devices):
    attempt=Path(attempt);ensure_guardian(attempt)
    assert dispatch['approved_by']=='orchestrator'
    assert json.loads((attempt/'STAGE0_REPORT.json').read_text())['status']=='PASS'
    assert not (attempt/'selection_frozen.json').exists()
    assert dispatch['stage'] in ('S1','S2','S2_EXTRA','S3')
    if dispatch['stage']=='S1':assert dispatch==initial_dispatch()
    for j in dispatch['jobs']:
        assert j['stage']==dispatch['stage']
        assert j['config']['lambda_pair'] in ((0.,) if j['config']['arm'] in ('A0','A1') else (.05,.2))
        if dispatch['stage']!='S3':assert j['seed'] in (2022,2024)
        if dispatch['stage']=='S2':assert j['config']['arm'] in ('P10','P01','PCV','PCOLL')
        if dispatch['stage']=='S2_EXTRA':assert j['config']['arm'] in ('PCV','PCOLL')
    existing=[json.loads(f.read_text()) for f in (attempt/'dispatches').glob('*.json')]
    all_dispatches=existing+[dispatch]
    development={j['key'] for ds in all_dispatches if ds['stage']!='S3' for j in ds['jobs']}
    assert len(development)<=48
    for d in ('movielens','kuairand'):
        confirmation=[j for ds in all_dispatches if ds['stage']=='S3' for j in ds['jobs'] if j['dataset']==d]
        assert len({(j['config']['arm'],j['config']['lambda_pair']) for j in confirmation})<=4
        assert len({j['key'] for j in confirmation})<=20
    for j in dispatch['jobs']:
        assert j==job(j['dataset'],j['config']['arm'],j['seed'],j['stage'],lambda_pair=j['config']['lambda_pair'])
        write_json(attempt/'jobs'/f"{j['key']}.json",j)
    write_json(attempt/'dispatches'/f"{dispatch['stage']}_{digest(dispatch)[:12]}.json",dispatch)
    processes={}; start=time.perf_counter()
    peak=json.loads((attempt/'STAGE0_REPORT.json').read_text())['peak_reserved_bytes']
    while True:
        records=request(attempt,'status')['jobs'];pending=[];active=[];failed=[]
        for j in dispatch['jobs']:
            k=j['key'];r=records.get(k,{})
            proc=processes.get(k)
            if r.get('status')=='COMPLETE':continue
            if alive(r.get('worker_pid')) or (proc and proc.poll() is None):active.append(k);continue
            if r or proc:
                failed.append(k);continue
            pending.append(j)
        write_json(attempt/'progress.json',dict(stage=dispatch['stage'],complete=sum(records.get(j['key'],{}).get('status')=='COMPLETE'
            for j in dispatch['jobs']),total=len(dispatch['jobs']),active=[dict(key=k,epoch=records.get(k,{}).get('epoch',0)) for k in active],
            pending=len(pending),failed=failed,queue_wall_seconds=time.perf_counter()-start,test_accessed=False))
        if failed:raise RuntimeError('Worker failure; inspect launch log and failure record: '+str(failed))
        if not pending and not active:break
        allocating=any(not records.get(k,{}).get('allocated') for k in active)
        if pending and not allocating and available_ram()>6*1024**3:
            gpu=choose_device(gpu_status(),devices,peak)
            if gpu is not None:
                j=pending[0];logdir=attempt/'launch_logs';logdir.mkdir(exist_ok=True)
                with (logdir/(j['key']+'.log')).open('ab') as log:
                    proc=subprocess.Popen([sys.executable,'-m',__package__+'.run','_worker','--attempt',str(attempt),
                        '--key',j['key'],'--device',f'cuda:{gpu}'],cwd=HERE.parents[1],stdin=subprocess.DEVNULL,
                        stdout=log,stderr=log,start_new_session=True)
                processes[j['key']]=proc
                _jsonl(attempt/'launch_ledger.jsonl',dict(key=j['key'],pid=proc.pid,gpu=gpu,time=time.time(),gpu_status=gpu_status()))
                while proc.poll() is None:
                    if request(attempt,'status')['jobs'].get(j['key'],{}).get('worker_pid')==proc.pid:break
                    time.sleep(.2)
        time.sleep(1)
    return results(attempt)


def freeze(attempt,decision):
    attempt=Path(attempt);records=request(attempt,'status')['jobs']
    assert decision['approved_by']=='orchestrator' and decision['training_complete']
    assert all(r['status']=='COMPLETE' for r in records.values())
    assert not (attempt/'selection_frozen.json').exists()
    entries=[]
    for r in results(attempt):
        state=request(attempt,'get',key=r['job']['key'])['state'];assert state is not None
        for kind in ('online','fixed_ema'):
            for checkpoint in ('selected','E10','E50'):
                h=r[{'selected':'selected_hashes','E10':'E10_hashes','E50':'E50_hashes'}[checkpoint]][kind]
                epoch=r['selected_epochs'][kind] if checkpoint=='selected' else int(checkpoint[1:])
                entries.append(dict(key=r['job']['key'],job=r['job'],model_kind=kind,checkpoint=checkpoint,epoch=epoch,model_hash=h))
    frozen=dict(source_hash=source_manifest()['sha256'],time=time.time(),decision=decision,endpoints=entries,
        validation_only_selection=True,test_opened=False)
    write_json(attempt/'selection_frozen.json',frozen)
    return frozen


def final_evaluate(attempt,devices):
    attempt=Path(attempt);frozen=json.loads((attempt/'selection_frozen.json').read_text())
    assert frozen['source_hash']==source_manifest()['sha256'];p=load_protocol();rows=[];cache={}
    for entry in frozen['endpoints']:
        output=attempt/'test_endpoints'/f"{entry['key']}_{entry['model_kind']}_{entry['checkpoint']}.json"
        if output.exists():rows.append(json.loads(output.read_text()));continue
        tick=time.perf_counter();j=entry['job'];d=j['dataset']
        if d not in cache:cache[d]=load_dataset(d,p=p)[0]
        bundle=cache[d];state=request(attempt,'get',key=entry['key'])['state'];assert state is not None
        if entry['checkpoint']=='selected':weights=state['selected_models'][entry['model_kind']]
        elif entry['checkpoint']=='E10':weights=state['E10_models'][entry['model_kind']]
        else:weights=state['model' if entry['model_kind']=='online' else 'ema']
        assert state_digest(weights)==entry['model_hash']
        gpu=None
        while gpu is None:
            gpu=choose_device(gpu_status(),devices,2*1024**3)
            if gpu is None:time.sleep(1)
        device=f'cuda:{gpu}';setup(j['seed'],device)
        model=make_model(bundle,p['datasets'][d]['source']).to(device);model.load_state_dict(weights)
        values,cost=evaluate(model,bundle.test_x,bundle.test_y,device)
        cost['total_endpoint_seconds']=time.perf_counter()-tick
        row=dict(**entry,status='COMPLETE',role='test',metrics=values,cost=cost)
        write_json(output,row);rows.append(row);del model;torch.cuda.empty_cache()
    write_json(attempt/'final_evaluation.json',dict(status='COMPLETE',endpoints=len(rows),rows=rows))
    for key in {r['key'] for r in rows}:request(attempt,'release',key=key)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['manifest','initial','stage0','schedule','freeze','evaluate','status','_worker','_guardian','shutdown'])
    parser.add_argument('--attempt',type=Path);parser.add_argument('--review',type=Path);parser.add_argument('--dispatch',type=Path)
    parser.add_argument('--decision',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--key')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--devices',default='0,1')
    a=parser.parse_args();devices=[int(x) for x in a.devices.split(',')]
    if a.command=='manifest':out=source_manifest()
    elif a.command=='initial':out=initial_dispatch()
    elif a.command=='_guardian':
        from .guardian import serve
        serve(a.attempt);return
    elif a.command=='_worker':worker(a.attempt,a.key,a.device);return
    elif a.command=='status':out=request(a.attempt,'status')
    elif a.command=='shutdown':out=request(a.attempt,'shutdown')
    else:
        review_check(a.review)
        if a.command=='stage0':
            from .stage0 import stage0
            out=stage0(a.attempt,a.device)
        elif a.command=='schedule':out=schedule(a.attempt,json.loads(a.dispatch.read_text()),devices)
        elif a.command=='freeze':out=freeze(a.attempt,json.loads(a.decision.read_text()))
        else:out=final_evaluate(a.attempt,devices)
    if a.output:write_json(a.output,out)
    else:print(json.dumps(out,indent=2))

if __name__=='__main__':main()
