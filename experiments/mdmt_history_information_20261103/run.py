"""Explicit frozen-stage dispatch and RAM-only training/evaluation lifecycle."""
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
import torch
from .protocol import (ROOT, HERE, REPORT, PLAN, DEFAULT_ATTEMPT, load_protocol, config,
    make_optimizer, setup, cpu_copy, state_digest, rng_state, restore_rng, write_json,
    make_model, load_dataset, evaluate, source_manifest, job, digest)
from .guardian import request, ensure_guardian, alive
from .objectives import train_step
from . import decisions

HISTORY = ROOT/'artifacts/conditional_information_momentum_20261102_v3/attempt1'


def review_check(path):
    r=json.loads(Path(path).read_text())
    assert r['verdict'] in ('PASS','PASS WITH FIXES') and not r.get('blocking_issues'), 'Independent review has blocking issues'
    assert r['source_hash']==source_manifest()['sha256'], 'Source changed since independent review'
    return r


def gpu_status():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [dict(zip(('index','memory','total','power','limit'),[float(v.strip()) for v in line.split(',')])) for line in p.stdout.strip().splitlines()]


def choose_device(status,devices,peak_bytes):
    for r in status:
        if int(r['index']) in devices and r['memory']/r['total']<.85 and r['power']/r['limit']<.85 and (r['total']-r['memory'])*1024**2>=peak_bytes:return int(r['index'])
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
        resume_counts=dict(counts)
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
                    diagnostic=step==1 or step%20==0)
                right.record();right.synchronize();n=len(ids)
                loss_sum+=value['loss']*n;real_sum+=value['real_loss']*n
                for name,v in value['counts'].items():
                    counts[name]=counts.get(name,0)+v;epoch_counts[name]=epoch_counts.get(name,0)+v
                counts['train_gpu_event_ms']=counts.get('train_gpu_event_ms',0.)+left.elapsed_time(right)
                record=dict(epoch=epoch,batch=batch,step=step,rows=n,loss=value['loss'],real_loss=value['real_loss'],
                    worker_seconds=time.perf_counter()-start,counts=value['counts'],diagnostics=value['diagnostics'])
                _jsonl(out/'batch_trace.jsonl',record)
                if step==1 or step%20==0:epoch_diagnostics.append(value['diagnostics'])
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
            optimizer_hash=state_digest(opt.state_dict()),counts=counts,physical_counts={k:v-resume_counts.get(k,0) for k,v in counts.items()},
            worker_seconds=committed_seconds+time.perf_counter()-start,
            attempt_worker_seconds=time.perf_counter()-start,attempt=number,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        request(attempt,'complete',key=key,result=result)
        write_json(out/'physical_cost.json',dict(key=key,attempt=number,status='COMPLETE',
            worker_seconds=time.perf_counter()-start,committed_epoch=50,counts=counts))
    except BaseException as error:
        failure_counts=dict(locals().get('counts',{}));failure_resume=dict(locals().get('resume_counts',{}))
        failure=dict(key=key,attempt=number,error=repr(error),traceback=traceback.format_exc(),
                     worker_seconds=time.perf_counter()-start,oom=isinstance(error,torch.OutOfMemoryError),
                     completed_step_counts={k:v-failure_resume.get(k,0) for k,v in failure_counts.items()},failed_step_counts_complete=False)
        write_json(out/'failure.json',failure)
        request(attempt,'failed',key=key,failure=failure)
        raise


def results(attempt):
    return [json.loads(f.read_text()) for f in sorted((Path(attempt)/'runs').glob('*/result.json'))]


def import_history(attempt):
    """Import exact validation scalars; historical test files stay unopened until freeze."""
    receipt=REPORT/'HISTORICAL_RAW_VERIFICATION.json'
    assert receipt.exists(), 'Pinned historical archive verification is required'
    rows=[];p=load_protocol()
    for file in sorted(HISTORY.glob('runs/*/result.json')):
        r=json.loads(file.read_text());old=r['job'];c=old['config'];a=c['arm']
        if a not in ('A0','A1') or old['seed'] not in (2022,2024):continue
        assert r['epochs']==50 and old['split_id']=='original' and c['lambda_pair']==0
        d=old['dataset'];new=job(d,'A0f' if a=='A0' else a,old['seed'],'HISTORICAL_REFERENCE');nc=new['config']
        for field in ('dataset','rho','weight_decay','lr','epochs','batch_size'):assert c[field]==nc[field],field
        if a=='A1':
            for field in ('lambda_aug','kappa','target_domain'):assert c[field]==nc[field],field
        manifests=list(file.parent.glob('attempt*/data_manifest.json'));assert manifests
        manifest=json.loads(manifests[-1].read_text());expected=p['datasets'][d]['source']
        for field in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','encoder_hash','scaler_hash','train_domain_counts','split_rows'):assert manifest[field]==expected[field],(d,field)
        for endpoint in ('selected','E10','E50'):
            assert set(r[endpoint+'_validation'])=={'online','fixed_ema'}
            assert set(r[endpoint+'_hashes'])=={'online','fixed_ema'}
        provenance=dict(commit='a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288',old_key=old['key'],old_job=old,result_path=str(file),result_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),verification_receipt=str(receipt),epoch_directory=str(manifests[-1].parent),numerical_reference=True)
        r=dict(r,job=new,provenance=provenance);write_json(Path(attempt)/'runs'/new['key']/'result.json',r);rows.append(r)
    assert len(rows)==8, ('Expected A0/A1 x two datasets x two seeds',len(rows))
    write_json(Path(attempt)/'HISTORICAL_REFERENCES.json',[dict(key=r['job']['key'],**r['provenance']) for r in rows])
    return rows


def prepare(attempt,plan=PLAN):
    p=load_protocol(plan);assert digest(p)==digest(load_protocol()),'Only the frozen plan is accepted'
    attempt=Path(attempt);attempt.mkdir(parents=True,exist_ok=True)
    existing=attempt/'PREPARED.json'
    if existing.exists():
        saved=json.loads(existing.read_text());assert saved['source_hash']==source_manifest()['sha256'];return saved
    manifests={}
    for d in p['datasets']:
        bundle,manifest=load_dataset(d,p=p);manifests[d]=manifest
        model=make_model(bundle,p['datasets'][d]['source']);opt=make_optimizer(model,config('A0f',d))
        write_json(REPORT/f'PARAMETER_MANIFEST_{d}.json',opt.parameter_manifest)
        del model,opt,bundle
    write_json(REPORT/'DATA_MANIFEST.json',manifests);write_json(attempt/'DATA_MANIFEST.json',manifests)
    import_history(attempt)
    out=dict(status='PREPARED',source_hash=source_manifest()['sha256'],plan=str(PLAN),test_accessed=False)
    write_json(existing,out);write_json(REPORT/'IMPLEMENTATION_SOURCE_MANIFEST.json',source_manifest());return out


def failed_jobs(attempt):
    statuses=request(attempt,'status')['jobs'];out={}
    for key,r in statuses.items():
        if r['status']=='COMPLETE' or alive(r.get('worker_pid')):continue
        failure=r.get('failure',{})
        infrastructure=failure.get('oom') or r['status']=='RUNNING'
        status=('NEEDS_INFRASTRUCTURE_REPLAY' if r['attempts']<2 else 'COST_BLOCKED') if infrastructure else 'IMPLEMENTATION_BLOCKED'
        out[key]=dict(status=status,job=r['job'],attempts=r['attempts'],failure=failure)
    return out


def cost_blocked(attempt):
    s=json.loads((Path(attempt)/'STAGE0_REPORT.json').read_text())
    # stage0 publishes explicit per-dataset/arm statuses; only those measured identities are blocked.
    blocked={(r['dataset'],r['arm']) for r in s.get('measurements',[]) if r.get('status')=='COST_BLOCKED'}
    for r in failed_jobs(attempt).values():
        if r['status']!='NEEDS_INFRASTRUCTURE_REPLAY':blocked.add((r['job']['dataset'],digest(r['job']['config']),r['job']['seed']))
    return blocked


def schedule(attempt,stage,devices):
    attempt=Path(attempt);ensure_guardian(attempt)
    assert json.loads((attempt/'PREPARED.json').read_text())['source_hash']==source_manifest()['sha256']
    s0=json.loads((attempt/'STAGE0_REPORT.json').read_text());assert s0['status']=='PASS'
    assert not (attempt/'SELECTION_FROZEN.json').exists()
    blocked=cost_blocked(attempt);jobs=[j for j in decisions.stage_jobs(attempt,stage) if not decisions.blocked_recipe(j['dataset'],decisions.recipe(j['config']),blocked,j['seed'])]
    old_files=list((attempt/'dispatches').glob('*.json'));all_jobs={}
    for file in old_files:
        for j in json.loads(file.read_text())['jobs']:all_jobs[j['key']]=j
    all_jobs.update({j['key']:j for j in jobs});assert len(all_jobs)<=96
    limits=dict(D0=18,D1=24,REPLICATE=6,D2=24,CONFIRM=24);assert len(jobs)<=limits[stage]
    dispatch=dict(stage=stage,approved_by='orchestrator',jobs=jobs,blocked=[list(x) for x in sorted(blocked)])
    target=attempt/'dispatches'/f'{stage}.json'
    if target.exists():
        dispatch=json.loads(target.read_text());jobs=dispatch['jobs']
        assert all(j['source_hash']==source_manifest()['sha256'] for j in jobs)
    else:write_json(target,dispatch)
    for j in jobs:write_json(attempt/'jobs'/f"{j['key']}.json",j)
    processes={};start=time.perf_counter();peak=s0['peak_reserved_bytes']
    while True:
        records=request(attempt,'status')['jobs'];done={r['job']['key'] for r in results(attempt)};pending=[];active=[];failed=[]
        for j in jobs:
            k=j['key'];r=records.get(k,{});proc=processes.get(k)
            if k in done:continue
            if alive(r.get('worker_pid')) or (proc and proc.poll() is None):active.append(k)
            elif r or proc:failed.append(k)
            else:pending.append(j)
        write_json(attempt/'progress.json',dict(stage=stage,complete=sum(j['key'] in done for j in jobs),total=len(jobs),active=[dict(key=k,epoch=records.get(k,{}).get('epoch',0)) for k in active],pending=len(pending),failed=failed,queue_wall_seconds=time.perf_counter()-start,test_accessed=False))
        if failed:write_json(attempt/'BLOCKED_RUNS.json',failed_jobs(attempt))
        if not pending and not active:break
        allocating=any(not records.get(k,{}).get('allocated') for k in active)
        if pending and not allocating and available_ram()>6*1024**3:
            gpu=None;j=None
            status=gpu_status()
            for candidate in pending:
                arm=candidate['config']['arm'];dataset=candidate['dataset']
                measured={'T0':'TM','T_U':'TM','T_NORM':'TM','H_REAL':'H_MIX','H_U':'H_MIX','H_NORM':'H_MIX','C_MIX':'H_MIX','U_MIX':'M','SCALE':'A0f','A1':'M'}.get(arm,arm)
                needed=next((r['peak_reserved_bytes'] for r in s0['measurements'] if r['dataset']==dataset and r['arm']==measured),peak)
                gpu=choose_device(status,devices,needed)
                if gpu is not None:j=candidate;break
            if gpu is not None:
                logdir=attempt/'launch_logs';logdir.mkdir(exist_ok=True)
                with (logdir/(j['key']+'.log')).open('ab') as log:
                    proc=subprocess.Popen([sys.executable,'-m',__package__+'.run','_worker','--attempt',str(attempt),'--key',j['key'],'--device',f'cuda:{gpu}'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                processes[j['key']]=proc
                _jsonl(attempt/'launch_ledger.jsonl',dict(key=j['key'],pid=proc.pid,gpu=gpu,time=time.time(),gpu_status=gpu_status()))
                while proc.poll() is None:
                    if request(attempt,'status')['jobs'].get(j['key'],{}).get('worker_pid')==proc.pid:break
                    time.sleep(.2)
        time.sleep(1)
    failures=failed_jobs(attempt)
    return dict(stage=stage,status='NEEDS_INFRASTRUCTURE_REPLAY' if any(r['status']=='NEEDS_INFRASTRUCTURE_REPLAY' for r in failures.values()) else 'COMPLETE',jobs=len(jobs),blocked=failures,queue_wall_seconds=time.perf_counter()-start)


def decide(attempt,stage):
    attempt=Path(attempt);required=json.loads((attempt/'dispatches'/f'{stage}.json').read_text())['jobs'];rs=results(attempt)
    failures=failed_jobs(attempt);assert not any(r['status']=='NEEDS_INFRASTRUCTURE_REPLAY' for r in failures.values()),'Infrastructure replay decision pending'
    assert {j['key'] for j in required}<={r['job']['key'] for r in rs}|set(failures),'Stage incomplete'
    decision=decisions.decide(attempt,stage,rs,cost_blocked(attempt));decision['approved_by']='orchestrator';decision['blocked_runs']=failures
    path=attempt/'decisions'/f'{stage}.json'
    if path.exists():assert json.loads(path.read_text())==decision
    else:write_json(path,decision)
    write_json(REPORT/f'DECISION_{stage}.json',decision);return decision


def freeze(attempt):
    attempt=Path(attempt);paths=[attempt/'decisions'/f'{s}.json' for s in decisions.STAGES];existing=[p for p in paths if p.exists()];assert existing
    last=json.loads(existing[-1].read_text());assert all(v.get('next_stage') is None for v in last['datasets'].values()),'Triggered stage remains unfinished'
    rs=results(attempt);failures=failed_jobs(attempt);assert not any(r['status']=='NEEDS_INFRASTRUCTURE_REPLAY' for r in failures.values())
    done={r['job']['key'] for r in rs}|set(failures)
    for p in (attempt/'dispatches').glob('*.json'):assert {j['key'] for j in json.loads(p.read_text())['jobs']}<=done
    records=request(attempt,'status')['jobs'];assert all(r['status']=='COMPLETE' or k in failures for k,r in records.items())
    entries=[]
    for r in rs:
        if not r.get('provenance'):assert request(attempt,'get',key=r['job']['key'])['state'] is not None
        for kind in ('online','fixed_ema'):
            for checkpoint in ('selected','E10','E50'):
                epoch=r['selected_epochs'][kind] if checkpoint=='selected' else int(checkpoint[1:])
                entries.append(dict(key=r['job']['key'],job=r['job'],model_kind=kind,checkpoint=checkpoint,epoch=epoch,model_hash=r[checkpoint+'_hashes'][kind],provenance=r.get('provenance')))
    configuration=dict(source_hash=source_manifest()['sha256'],decision=last,training_complete=True,blocked_runs=failures,jobs=[r['job'] for r in rs])
    frozen=dict(source_hash=source_manifest()['sha256'],decision=last,endpoints=entries,validation_only_selection=True,test_opened=False)
    for filename,value in [('CONFIGURATION_FROZEN.json',configuration),('SELECTION_FROZEN.json',frozen)]:
        file=attempt/filename
        if file.exists():assert json.loads(file.read_text())==value
        else:write_json(file,value)
        write_json(REPORT/filename,value)
    return frozen


def final_evaluate(attempt,devices):
    attempt=Path(attempt);frozen=json.loads((attempt/'SELECTION_FROZEN.json').read_text());assert (attempt/'CONFIGURATION_FROZEN.json').exists()
    assert frozen['source_hash']==source_manifest()['sha256'];p=load_protocol();rows=[];cache={};historical=None
    for entry in frozen['endpoints']:
        output=attempt/'test_endpoints'/f"{entry['key']}_{entry['model_kind']}_{entry['checkpoint']}.json"
        if output.exists():rows.append(json.loads(output.read_text()));continue
        if entry.get('provenance'):
            if historical is None:historical=json.loads((HISTORY/'final_evaluation.json').read_text())['rows']
            matches=[r for r in historical if r['key']==entry['provenance']['old_key'] and r['model_kind']==entry['model_kind'] and r['checkpoint']==entry['checkpoint']];assert len(matches)==1
            old=matches[0];assert old['model_hash']==entry['model_hash'] and old['epoch']==entry['epoch']
            row=dict(**entry,status='COMPLETE',role='test',metrics=old['metrics'],cost=dict(numerical_reference=True,new_evaluation_seconds=0.))
        else:
            tick=time.perf_counter();j=entry['job'];d=j['dataset']
            if d not in cache:cache[d]=load_dataset(d,p=p)[0]
            bundle=cache[d];state=request(attempt,'get',key=entry['key'])['state'];assert state is not None
            if entry['checkpoint']=='selected':weights=state['selected_models'][entry['model_kind']]
            elif entry['checkpoint']=='E10':weights=state['E10_models'][entry['model_kind']]
            else:weights=state['model' if entry['model_kind']=='online' else 'ema']
            assert state_digest(weights)==entry['model_hash'];gpu=None
            while gpu is None:
                gpu=choose_device(gpu_status(),devices,2*1024**3)
                if gpu is None:time.sleep(1)
            device=f'cuda:{gpu}';setup(j['seed'],device);model=make_model(bundle,p['datasets'][d]['source']).to(device);model.load_state_dict(weights)
            values,cost=evaluate(model,bundle.test_x,bundle.test_y,device);cost['total_endpoint_seconds']=time.perf_counter()-tick
            row=dict(**entry,status='COMPLETE',role='test',metrics=values,cost=cost)
            del model;torch.cuda.empty_cache()
        write_json(output,row);rows.append(row)
    write_json(attempt/'final_evaluation.json',dict(status='COMPLETE',endpoints=len(rows),rows=rows))
    for key in {r['key'] for r in rows if not r.get('provenance')}:request(attempt,'release',key=key)
    return dict(status='COMPLETE',endpoints=len(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','stage0','dispatch','decide','freeze','evaluate','report','retry','status','_worker','_guardian','shutdown'])
    parser.add_argument('--attempt',type=Path,default=DEFAULT_ATTEMPT);parser.add_argument('--review',type=Path,default=REPORT/'INDEPENDENT_REVIEW_RECEIPT.json')
    parser.add_argument('--plan',type=Path,default=PLAN);parser.add_argument('--stage',choices=decisions.STAGES);parser.add_argument('--cpu',action='store_true');parser.add_argument('--split',choices=['test'],default='test')
    parser.add_argument('--reason',choices=['oom','worker_terminated']);parser.add_argument('--output',type=Path);parser.add_argument('--key');parser.add_argument('--device',default='cuda:0');parser.add_argument('--devices',default='0,1')
    a=parser.parse_args();devices=[int(x) for x in a.devices.split(',')]
    if a.command=='_guardian':
        from .guardian import serve
        serve(a.attempt);return
    if a.command=='_worker':worker(a.attempt,a.key,a.device);return
    if a.command=='prepare':out=prepare(a.attempt,a.plan)
    elif a.command=='check':
        result=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q'],cwd=ROOT)
        out=dict(status='PASS' if result.returncode==0 else 'FAIL',returncode=result.returncode);write_json(REPORT/'CPU_TESTS.json',out)
        if result.returncode:raise SystemExit(result.returncode)
    elif a.command=='status':out=request(a.attempt,'status')
    elif a.command=='shutdown':out=request(a.attempt,'shutdown')
    elif a.command=='report':
        from .reporting import report
        out=report(a.attempt)
    else:
        review_check(a.review)
        if a.command=='stage0':
            from .stage0 import stage0
            out=stage0(a.attempt,a.device)
        elif a.command=='dispatch':assert a.stage;out=schedule(a.attempt,a.stage,devices)
        elif a.command=='decide':assert a.stage;out=decide(a.attempt,a.stage)
        elif a.command=='retry':
            assert a.key and a.reason and not (a.attempt/'SELECTION_FROZEN.json').exists()
            record=request(a.attempt,'status')['jobs'][a.key]
            assert record['status']!='COMPLETE' and not alive(record.get('worker_pid')) and record['attempts']<2
            if a.reason=='oom':assert record.get('failure',{}).get('oom')
            else:assert record.get('status')=='RUNNING', 'Numerical/application failures are not infrastructure termination'
            _jsonl(a.attempt/'replay_ledger.jsonl',dict(key=a.key,reason=a.reason,time=time.time(),previous_attempts=record['attempts']))
            worker(a.attempt,a.key,a.device);out=dict(status='COMPLETE',key=a.key)
        elif a.command=='freeze':out=freeze(a.attempt)
        else:out=final_evaluate(a.attempt,devices)
    if a.output:write_json(a.output,out)
    else:print(json.dumps(out,indent=2))

if __name__=='__main__':main()
