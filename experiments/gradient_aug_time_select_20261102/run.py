"""Finite RAM-only queue with independent shadow replay and one frozen test opening."""
import argparse
import contextlib
import copy
import faulthandler
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
from experiments.gradient_aug_20261031_v2.run import gpu_status,available_ram
from experiments.gradient_aug_aig_m_20261101.run import evaluate_timed
from .protocol import (ROOT,HISTORY,DEV_SEEDS,CONFIRM_SEEDS,ARMS,protocol,config,lambda_for,optimizer,
    source_manifest,identity,digest,setup,cpu_copy,equal,state_digest,tensor_bytes,rng_state,
    snapshot,restore,model_state,checkpoint_rows,parameter_groups,send_memory,receive_memory,
    write_json,append_csv,selected_update)
from .objectives import train_step
from .shadows import ShadowController,replay_sequence
from .analyze import compare,values,build_tables
from .recovery import (COST_KEYS,historical_endpoint_core,attempt_endpoint_rows,task_clock,
    recovery_context,verify_recomputed_results,execution_costs,scientific_job_key)


def counters():
    return dict(original_exposures=0,extra_views=0,view_draws=0,training_forwards=0,gradient_calls=0,
        optimizer_steps=0,shadow_updates=0,validation_forwards=0,validation_wall_seconds=0.,
        validation_gpu_event_ms=0.,training_gpu_event_ms=0.,selection_checks=0,mask_fields=0,
        perturbation_energy=0.,lambda_steps=0.,lambda_sample_weight=0.,training_passes=1)


def train_pass(bundle,dataset,p,job,device,out,connection=None,replay=None,expected=None,attempt=1):
    c=job['config'];seed=job['seed'];conf=p['datasets'][dataset];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    setup(seed,device);model=make_model(bundle,conf).to(device);opt=optimizer(model,c)
    ctl=ShadowController(model,c,replay);shadows=ctl.shadows
    assert sum(v.numel() for v in model.parameters())=={'movielens':773470,'kuairand':1614254}[dataset]
    loaders={k:torch.Generator().manual_seed(seed+v) for k,v in p['rng']['loader_offsets'].items()}
    selected={};best={};states={};order_hashes=[];step_hashes=[];view_hashes=[];mask_records=[];projection_sums={g:[0.,0.] for g in ('E','P')}
    run_identity=identity(dataset,seed,c,p,job.get('control'));run_hash=digest(run_identity)
    write_json(out/'resolved_config.json',dict(config=c,**run_identity,run_hash=run_hash,replay_pass=replay is not None))
    groups=parameter_groups(model,conf['augmented_domain'])
    write_json(out/'parameter_groups.json',dict(zip((n for n,_ in model.named_parameters()),groups)))
    count=counters();init_hash=state_digest(model.state_dict());step=0;start=time.perf_counter();ready=False
    validation_costs={};torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,51):
        epoch_start=time.perf_counter();model.train();order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        order_hash=row_ids_hash(bundle.train_row_ids[order]);order_hashes.append(order_hash)
        predictions=[];labels=[];domains=[];real_sum=reference_sum=0.;probability_log=[];row_log=[];step_log=[]
        lam=lambda_for(c,epoch,p['random_off_epochs'])
        for batch,offset in enumerate(range(0,len(order),40960),1):
            ids=order[offset:offset+40960];step+=1
            left,right=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);left.record()
            x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
            control=None
            if c['random_mask_enabled']:control=job['control']['mask_records'][str(seed)][step-1]
            scalar,info,projection,counts=train_step(model,opt,x,y,conf['augmented_domain'],c,dataset,seed,step,lam,control)
            updated=ctl.step(model,counts['gradnorm'],step,epoch)
            right.record();right.synchronize();count['training_gpu_event_ms']+=left.elapsed_time(right)
            model_hash=state_digest(model.state_dict());step_hashes.append(model_hash)
            if expected is not None:assert model_hash==expected['step_hashes'][step-1],f'Replay online differs at step{step}'
            for record in projection:
                projection_sums[record['group']][0]+=record['retained_norm'];projection_sums[record['group']][1]+=record['delta_norm']
                append_csv(out/'projection_stats.csv',dict(epoch=epoch,batch=batch,global_step=step,**record))
            record=info.get('mask_record')
            if record is not None:
                mask_records.append(record);count['mask_fields']+=sum(record['counts']);count['perturbation_energy']+=sum(record['energies'])
                append_csv(out/'mask_stats.csv',dict(epoch=epoch,batch=batch,global_step=step,**{
                    k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in record.items()}))
                if info.get('probabilities') is not None and replay is None:
                    row_log.append(bundle.train_row_ids[ids][info['mask'].cpu().numpy()]);probability_log.append(info['probabilities'].cpu().numpy())
                    step_log.append(np.full(len(row_log[-1]),step,dtype=np.int32))
            vhash=digest([v['hash'] for v in info['views']]);view_hashes.append(vhash)
            n=len(ids);count['original_exposures']+=n;count['extra_views']+=info['extra_views'];count['view_draws']+=len(info['views'])
            count['training_forwards']+=info['forwards'];count['gradient_calls']+=counts['gradient_calls'];count['optimizer_steps']+=1
            count['shadow_updates']+=updated;count['lambda_steps']+=lam;count['lambda_sample_weight']+=lam*n
            append_csv(out/'batch_trace.csv',dict(epoch=epoch,batch=batch,global_step=step,rows=n,
                original_row_hash=row_ids_hash(bundle.train_row_ids[ids]),epoch_order_hash=order_hash,online_model_hash=model_hash,
                lambda_aug=lam,gradnorm=counts['gradnorm'],cv=ctl.cv,rho=ctl.rho,
                replay_rho=None if replay is None else replay[step-1],view_hash=vhash,gradient_calls=counts['gradient_calls']))
            for view in info['views']:append_csv(out/'view_stats.csv',dict(epoch=epoch,batch=batch,global_step=step,
                view_key=view['key'],view_seed=view['seed'],view_hash=view['hash'],target_rows=int(info['mask'].sum())))
            real_sum+=float(info['real'].detach().mean(1).mean())*n;reference_sum+=float(scalar)*n
            if replay is None:
                predictions.append(info['pred'].detach().cpu().numpy());labels.append(y.cpu().numpy());domains.append(x[-1].cpu().numpy())
            if not ready and connection is not None:
                send_memory(connection,dict(kind='allocated',peak_memory_bytes=torch.cuda.max_memory_allocated(device)));ready=True
            del info,scalar,x,y
        count['shadow_updates']+=ctl.epoch_end(model,epoch);validation={}
        for kind in ctl.valid(epoch,replay is not None):
            current=model if kind=='online' else shadows[kind]
            validation[kind],cost=evaluate_timed(current,bundle.val_x,bundle.val_y,device)
            target=validation_costs.setdefault(kind,dict(forwards=0,wall_seconds=0.,gpu_event_ms=0.))
            for k,v in cost.items():target[k]+=v
            count['validation_forwards']+=cost['forwards'];count['validation_wall_seconds']+=cost['wall_seconds']
            count['validation_gpu_event_ms']+=cost['gpu_event_ms'];count['selection_checks']+=1
            if kind not in best:best[kind]=-float('inf')
            selected_update(selected,best,kind,epoch,validation[kind]['macro_auc'])
            append_csv(out/'validation_epoch_metrics.csv',dict(epoch=epoch,global_step=step,model_kind=kind,**validation[kind],**{
                'validation_'+k:v for k,v in cost.items()}))
        if epoch in (10,50) or epoch in selected.values():
            states[epoch]=snapshot(model,opt,shadows,ctl,loaders,epoch,step,validation,selected,best,c,device,count)
        for old in list(states):
            if old not in (10,50) and old not in selected.values():del states[old]
        if probability_log:
            np.savez_compressed(out/f'mask_probabilities_E{epoch:02d}.npz',row_id=np.concatenate(row_log),
                global_step=np.concatenate(step_log),probability=np.concatenate(probability_log))
        train={} if replay is not None else metrics(np.concatenate(labels),np.concatenate(predictions),np.concatenate(domains))
        append_csv(out/'epoch_metrics.csv',dict(epoch=epoch,global_step=step,epoch_order_hash=order_hash,
            train_real_bce=real_sum/len(order),reference_fixed_mixed_loss=reference_sum/len(order),
            lambda_aug=lam,epoch_seconds=time.perf_counter()-epoch_start,**{'train_'+k:v for k,v in train.items()}))
        print(f"{job['stage']} {dataset} {job['trial']} seed{seed} {'replay' if replay is not None else 'primary'} E{epoch}/50 step{step} selected={selected}",flush=True)
    assert step==conf['total_optimizer_steps']
    if expected is not None:
        assert order_hashes==expected['epoch_order_hashes'] and view_hashes==expected['view_hashes']
        assert state_digest(opt.state_dict())==expected['final_optimizer_hash']
    result=dict(dataset=dataset,stage=job['stage'],trial=job['trial'],seed=seed,attempt=attempt,epochs=50,global_step=step,
        selected_epochs=selected,selected_validation={k:states[e]['validation'][k] for k,e in selected.items()},states=states,
        config=c,config_hash=digest(c),run_hash=run_hash,run_identity=run_identity,init_hash=init_hash,
        epoch_order_hashes=order_hashes,step_hashes=step_hashes,view_hashes=view_hashes,mask_records=mask_records,
        projection_sums=projection_sums,rhos=ctl.rhos,final_optimizer_hash=state_digest(opt.state_dict()),output_dir=str(out),
        training_seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),snapshot_tensor_bytes=tensor_bytes(states),
        validation_costs=validation_costs,**count)
    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k not in ('states','mask_records','step_hashes','view_hashes','rhos')})
    write_json(out/'state_summary.json',dict(checkpoints=[row for k in selected for row in checkpoint_rows(result,k)]))
    del model,opt,ctl,shadows;gc.collect();torch.cuda.empty_cache()
    return result


def train_one(bundle,dataset,p,job,device,out,connection=None,attempt=1):
    result=train_pass(bundle,dataset,p,job,device,Path(out)/'primary',connection,attempt=attempt)
    if not job['config']['replay_enabled']:return result
    sequence=replay_sequence(result['rhos'])
    write_json(Path(out)/'rho_frozen.json',dict(adaptive=result['rhos'],replay=sequence,seed=1729,
        adaptive_hash=digest(result['rhos']),replay_hash=digest(sequence),multiset_equal=True,test_accessed=False))
    extra=train_pass(bundle,dataset,p,job,device,Path(out)/'replay',replay=sequence,expected=result,attempt=attempt)
    for epoch,state in extra['states'].items():
        if epoch not in result['states']:result['states'][epoch]=state
        else:
            existing=result['states'][epoch];assert state_digest(existing['model'])==state_digest(state['model'])
            existing['shadows']['replay_rho']=state['shadows']['replay_rho'];existing['validation'].update(state['validation'])
    result['selected_epochs'].update(extra['selected_epochs']);result['selected_validation'].update(extra['selected_validation'])
    result['validation_costs'].update(extra['validation_costs']);result['training_seconds']+=extra['training_seconds']
    for k in counters():
        result[k]+=extra[k]
    result['peak_memory_bytes']=max(result['peak_memory_bytes'],extra['peak_memory_bytes'])
    result['peak_reserved_bytes']=max(result['peak_reserved_bytes'],extra['peak_reserved_bytes'])
    result['snapshot_tensor_bytes']=tensor_bytes(result['states']);result['replay_verified']=True
    result['pass_dirs']=[str(Path(out)/'primary'),str(Path(out)/'replay')];result['output_dir']=str(out)
    write_json(Path(out)/'runtime_stats.json',{k:v for k,v in result.items() if k not in ('states','mask_records','step_hashes','view_hashes','rhos')})
    write_json(Path(out)/'state_summary.json',dict(checkpoints=[row for k in result['selected_epochs'] for row in checkpoint_rows(result,k)]))
    return result


def worker(connection,bundles,p,job,device,out,attempt):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            faulthandler.enable(file=stderr,all_threads=True)
            try:
                result=train_one(bundles[job['dataset']],job['dataset'],p,job,device,out,connection,attempt)
                send_memory(connection,dict(kind='complete',result=result))
            except Exception as exc:
                traceback.print_exc();send_memory(connection,dict(kind='error',error=repr(exc),
                    traceback=traceback.format_exc(),oom=isinstance(exc,torch.OutOfMemoryError),
                    recoverable=isinstance(exc,(torch.OutOfMemoryError,OSError))))
            finally:faulthandler.disable()
    connection.close()


def ledger_row(p,job,status,attempt=0,result=None):
    r=result or {};ident=identity(job['dataset'],job['seed'],job['config'],p,job.get('control'))
    return dict(stage=job['stage'],dataset=job['dataset'],trial=job['trial'],seed=job['seed'],attempt=attempt,
        status=status,run_id=digest(ident),config_hash=ident['config_hash'],data_hash=ident['data_hash'],
        source_hash=ident['source_hash'],execution_commit=p['execution_commit'],planned_epochs=50,
        planned_steps=p['datasets'][job['dataset']]['total_optimizer_steps'],completed_epochs=r.get('epochs',0),
        completed_steps=r.get('global_step',0),time=time.time(),output_dir=r.get('output_dir',''),
        selected_epochs=json.dumps(r.get('selected_epochs',{}),sort_keys=True),resolved_config=json.dumps(job['config'],sort_keys=True))


def settled_native_failure(process):
    process.join();code=process.exitcode
    return dict(kind='error',error=f'worker exited {code} without RAM state',oom=False,
        recoverable=code==-11,exitcode=code,signal=-code if code is not None and code<0 else None,reaped=True)


def next_pending_index(pending,active,serial,p):
    if any(scientific_job_key(record['job'],p) in serial for record in active):return None
    retry=next((i for i,job in enumerate(pending) if scientific_job_key(job,p) in serial),None)
    if retry is not None:return None if active else retry
    return 0 if pending else None


def schedule(bundles,p,out,jobs,devices):
    if not jobs:return []
    ctx=mp.get_context('fork');pending=list(jobs);active=[];completed=[];blocked=[];peaks={};attempts={};serial=set();allocating=False
    run_ids=[digest(identity(j['dataset'],j['seed'],j['config'],p,j.get('control'))) for j in jobs]
    assert len(set(run_ids))==len(run_ids)
    history=p.get('recovery',{}).get('native_failure_history',{}).get('failures_by_job',{})
    for job in jobs:
        key=scientific_job_key(job,p);attempts[key]=history.get(key,0)
        if attempts[key]:serial.add(key)
    for job in jobs:append_csv(out/'trial_ledger.csv',ledger_row(p,job,'PLANNED'))
    while pending or active:
        for record in list(active):
            connection,proc=record['connection'],record['process'];message=None
            if connection.poll():
                try:message=receive_memory(connection)
                except EOFError:message=settled_native_failure(proc)
            elif not proc.is_alive():message=settled_native_failure(proc)
            if message is None:continue
            if message['kind']=='allocated':
                record['allocated']=True;allocating=False
                peaks[record['gpu']]=max(peaks.get(record['gpu'],0),message['peak_memory_bytes']);continue
            if not message.get('reaped'):proc.join()
            connection.close();active.remove(record)
            if not record['allocated']:allocating=False
            job=record['job']
            if message['kind']=='complete':
                completed.append(message['result']);append_csv(out/'trial_ledger.csv',ledger_row(p,job,'COMPLETE',record['attempt'],message['result']))
            else:
                failure=dict(stage=job['stage'],dataset=job['dataset'],trial=job['trial'],seed=job['seed'],
                    device=record['gpu'],attempt=record['attempt'],oom=message['oom'],error=message['error'],time=time.time(),
                    exitcode=message.get('exitcode',proc.exitcode),signal=message.get('signal'),
                    recoverable=message.get('recoverable',message['oom']),retry_index=record['attempt']-1,
                    failure_count=record['attempt'],
                    prior_native_failures=history.get(scientific_job_key(job,p),0),run_id=digest(identity(
                        job['dataset'],job['seed'],job['config'],p,job.get('control'))))
                append_csv(out/'failure_ledger.csv',failure);append_csv(out/'trial_ledger.csv',ledger_row(p,job,'FAILED',record['attempt']))
                if record['attempt']<=p['limits']['retries'] and message.get('recoverable',message['oom']):
                    if message['oom'] or message.get('exitcode')==-11:serial.add(scientific_job_key(job,p))
                    pending.append(job)
                else:blocked.append(failure)
            write_json(out/'progress.json',dict(stage=jobs[0]['stage'],completed=len(completed),total=len(jobs),
                active=len(active),pending=len(pending),failed=len(blocked),test_accessed=False))
        if pending and time.time()>=p.get('dispatch_deadline',float('inf')):
            p['budget_stopped']=True
            for job in pending:append_csv(out/'trial_ledger.csv',ledger_row(p,job,'NOT_DISPATCHED_HARD_LIMIT'))
            pending.clear()
        if pending and not allocating and available_ram()>8*1024**3:
            for status in gpu_status():
                gpu=status['index']
                if gpu not in devices or status['memory']/status['total']>=.85 or status['power']/status['limit']>=.85:continue
                if (status['total']-status['memory'])*1024**2<peaks.get(gpu,1800*1024**2)*1.15:continue
                index=next_pending_index(pending,active,serial,p)
                if index is None:continue
                job=pending.pop(index);key=scientific_job_key(job,p)
                if attempts[key]>p['limits']['retries']:
                    blocked.append(dict(dataset=job['dataset'],trial=job['trial'],seed=job['seed'],
                        error='Native retry budget exhausted in prior raw failure ledgers'))
                    append_csv(out/'trial_ledger.csv',ledger_row(p,job,'RETRY_EXHAUSTED',attempts[key]));break
                attempts[key]+=1
                destination=out/'runs'/job['stage']/job['dataset']/job['trial']/f"seed{job['seed']}"/f'attempt{attempts[key]}'
                parent,child=ctx.Pipe(duplex=False);proc=ctx.Process(target=worker,args=(child,bundles,p,job,f'cuda:{gpu}',destination,attempts[key]))
                proc.start();child.close();active.append(dict(connection=parent,process=proc,job=job,gpu=gpu,allocated=False,attempt=attempts[key]))
                append_csv(out/'trial_ledger.csv',ledger_row(p,job,'RUNNING',attempts[key]))
                append_csv(out/'launch_ledger.csv',dict(time=time.time(),stage=job['stage'],dataset=job['dataset'],trial=job['trial'],
                    seed=job['seed'],attempt=attempts[key],gpu=gpu,memory_fraction=status['memory']/status['total'],
                    power_fraction=status['power']/status['limit'],active_workers=len(active),
                    serial_retry=key in serial,retry_index=attempts[key]-1,prior_native_failures=history.get(key,0)))
                allocating=True;break
        time.sleep(.5)
    if blocked:raise RuntimeError(f'Incomplete stage: {blocked}')
    return completed


def make_jobs(stage,arm,datasets,seeds,changes=None,control=None):
    return [dict(stage=stage,dataset=d,trial=arm,seed=s,config=config(arm,d,**(changes or {})),
        **({} if control is None else {'control':control[d]})) for s in seeds for d in datasets]


def confirmation_budget_reason(existing,planned,group):
    if planned+group>12:return 'confirmation_stage_cap_12_including_new_baselines'
    if existing+planned+group>40:return 'global_unique_trajectory_cap_40'
    return None


def historical_baseline_verification(results,p,out):
    rows=[row for result in results if result['trial'] in ('A0','A1') and result['seed'] in DEV_SEEDS
        for kind in ('online','fixed_ema') for row in checkpoint_rows(result,kind)]
    return historical_endpoint_core(rows,p,out)['verified']


def matched_control(results,dataset,arm):
    if arm.startswith('B2'):
        sums={g:[sum(r['projection_sums'][g][i] for r in results if r['dataset']==dataset and r['trial']==arm)
            for i in range(2)] for g in ('E','P')}
        ratios={g:(a/b if b else 1.) for g,(a,b) in sums.items()}
        assert all(0<=v<=1+1e-6 for v in ratios.values())
        shrink={g:min(v,1.) for g,v in ratios.items()}
        return dict(shrink=shrink),dict(source='B2 all training steps/tasks, both development seeds',sums=sums,shrink=shrink)
    if arm=='B1':
        records={str(r['seed']):r['mask_records'] for r in results if r['dataset']==dataset and r['trial']=='B1'}
        assert sorted(map(int,records))==DEV_SEEDS
        return {},dict(source='B1 training per-step per-field counts and energies',mask_records=records)
    return {},dict(source='frozen random five off-epochs, independent seed1729')


def candidate_decisions(results,p,matched=None,effective=None):
    answer={};matched=matched or {}
    for d in p['datasets']:
        reference=values(results,d,'A1',seeds=DEV_SEEDS);answer[d]={}
        for arm in ('B2','B1','B3'):
            trial=(effective or {}).get((d,arm),arm)
            primary=compare(values(results,d,trial,seeds=DEV_SEEDS),reference,d)
            control=matched.get((d,arm));item=dict(primary=primary,matched=control,
                trial=trial,status='NO_PROMOTION' if control is None else 'STOP')
            if control is not None and primary['passed'] and control['matched_exceeded']:item['status']='ELIGIBLE'
            elif control is None and primary['control_needed']:item['status']='MATCH_REQUIRED'
            elif primary['tradeoff']:item['status']='TRADEOFF'
            answer[d][arm]=item
        adaptive=values(results,d,'A1','adaptive_ema',DEV_SEEDS);replay=values(results,d,'A1','replay_rho',DEV_SEEDS)
        fixed_result=compare(adaptive,reference,d);replay_result=compare(adaptive,replay,d)
        answer[d]['C2']=dict(fixed=fixed_result,replay=replay_result,
            status='ELIGIBLE' if fixed_result['passed'] and replay_result['passed'] else 'NO_PROMOTION')
    return answer


def wait_for_design(out,request,p,start):
    path=out/'conditional_design_request.json';resolution=out/'conditional_design_resolution.json'
    write_json(path,dict(**request,status='WAITING_FOR_REVIEWED_DESIGN',test_accessed=False))
    while not resolution.exists():
        if time.perf_counter()-start>=p['limits']['hard_hours']*3600:
            return dict(action='STOP',reason='12h dispatch limit reached while awaiting registered conditional design')
        time.sleep(1.)
    decision=json.loads(resolution.read_text());assert decision.get('reviewed') is True
    if decision['action']!='STOP':raise RuntimeError('New conditional formula requires reviewed implementation/source freeze before dispatch')
    return decision


def test_worker(connection,bundle,p,result,device):
    try:
        setup(result['seed'],device);model=make_model(bundle,p['datasets'][result['dataset']]).to(device)
        records={};cost=dict(forwards=0,wall_seconds=0.,gpu_event_ms=0.)
        for kind in result['selected_epochs']:
            for row in checkpoint_rows(result,kind):
                key=(kind,row['epoch'])
                if key in records:continue
                model.load_state_dict(model_state(result['states'][row['epoch']],kind))
                records[key],item=evaluate_timed(model,bundle.test_x,bundle.test_y,device)
                for name,value in item.items():cost[name]+=value
        send_memory(connection,dict(kind='complete',metrics=records,cost=cost))
    except Exception:send_memory(connection,dict(kind='error',error=traceback.format_exc()))


def history_rows(p,datasets,seeds,split):
    frame=pd.read_csv(Path(p['history_dir'])/f'{split}_metrics.csv');rows=[]
    for r in frame.to_dict('records'):
        if r['dataset'] not in datasets or int(r['seed']) not in seeds:continue
        if r['arm'] not in ('A0','A0_EMA','A1_EMA'):continue
        r['history_arm']=r['arm'];r['history_model_kind']=r['model_kind']
        r['model_kind']='online' if r['arm']=='A0' else 'fixed_ema'
        r['arm']='A0' if r['arm'] in ('A0','A0_EMA') else 'A1'
        r['stage']='NUMERICAL_BASELINE_REUSED';r['source_hash']='historical_AIGM01'
        r['execution_commit']='2cec25ab5b32411855a13ab3143678a5e980ea3f'
        r['history_ref']='26-11-01_v3@6f5742c2085c1e49d1767b0a7079bb9f28160eda'
        rows.append(r)
    return rows


def unified_test(bundles,p,out,results,history_datasets):
    frozen=[row for r in results for k in r['selected_epochs'] for row in checkpoint_rows(r,k)]
    history_validation=history_rows(p,history_datasets,CONFIRM_SEEDS,'validation') if history_datasets else []
    current={(r['dataset'],r['trial'],r['seed']) for r in results}
    history_validation=[r for r in history_validation if (r['dataset'],r['arm'],r['seed']) not in current]
    write_json(out/'selection_frozen.json',dict(study_id=p['study_id'],execution_commit=p['execution_commit'],
        code_hash=p['source_manifest_hash'],checkpoints=frozen,historical_baselines=history_validation,
        unavailable=[dict(dataset=r['dataset'],arm=r['trial'],seed=r['seed'],kind='swa20',checkpoint='E10',reason='SWA starts at E31')
            for r in results if 'swa20' in r['selected_epochs']],test_started=False))
    validation_rows=[];test_rows=[];selected=[];ctx=mp.get_context('fork')
    for result in results:
        gpu=min((s for s in gpu_status() if s['index'] in p['devices']),key=lambda s:s['memory']/s['total'])['index']
        parent,child=ctx.Pipe(duplex=False);proc=ctx.Process(target=test_worker,
            args=(child,bundles[result['dataset']],p,result,f'cuda:{gpu}'))
        proc.start();child.close();message=receive_memory(parent);proc.join();parent.close()
        if message['kind']=='error':raise RuntimeError(message['error'])
        append_csv(out/'test_costs.csv',dict(dataset=result['dataset'],trial=result['trial'],seed=result['seed'],**message['cost']))
        for kind in result['selected_epochs']:
            for row in checkpoint_rows(result,kind):
                val=row.pop('validation');test=message['metrics'][kind,row['epoch']]
                validation_rows.append(dict(**row,**val));test_rows.append(dict(**row,**test))
                selected.append(dict(**row,val_macro_auc=val['macro_auc'],val_macro_logloss=val['macro_logloss'],
                    test_macro_auc=test['macro_auc'],test_macro_logloss=test['macro_logloss']))
        print(f"test complete {result['dataset']} {result['trial']} seed{result['seed']}",flush=True)
    validation_rows.extend(history_validation)
    if history_datasets:test_rows.extend(r for r in history_rows(p,history_datasets,CONFIRM_SEEDS,'test')
        if (r['dataset'],r['arm'],r['seed']) not in current)
    pd.DataFrame(validation_rows).to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame(test_rows).to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'selected_metrics.csv',index=False)


def aggregate(results,out):
    for filename in ('epoch_metrics.csv','validation_epoch_metrics.csv','projection_stats.csv','mask_stats.csv','batch_trace.csv'):
        chunks=[]
        for result in results:
            dirs=result.get('pass_dirs',[result['output_dir']])
            for i,directory in enumerate(dirs):
                path=Path(directory)/filename
                if not path.exists():continue
                table=pd.read_csv(path)
                for key in ('dataset','stage','trial','seed','attempt','config_hash','run_hash'):table[key]=result[key]
                table['pass']='replay' if i else 'primary';chunks.append(table)
        if chunks:pd.concat(chunks,ignore_index=True).to_csv(out/filename,index=False)
    keys=['dataset','stage','trial','seed','attempt','epochs','global_step','training_seconds','peak_memory_bytes',
        'peak_reserved_bytes','snapshot_tensor_bytes','config_hash','run_hash','output_dir']+list(counters())
    pd.DataFrame([{k:r[k] for k in keys} for r in results]).to_csv(out/'costs.csv',index=False)
    write_json(out/'development_selected_checkpoints.json',dict(checkpoints=[row for r in results for k in r['selected_epochs'] for row in checkpoint_rows(r,k)]))


def finish_queue(bundles,p,out,results,start,additional,history_datasets=()):
    verify_recomputed_results(results,p,out)
    unified_test(bundles,p,out,results,history_datasets);aggregate(results,out);build_tables(out)
    costs=execution_costs(results,p,out);write_json(out/'execution_cost_continuity.json',costs)
    write_json(out/'completion.json',dict(status='EARLY_STOP_BUDGET' if p.get('budget_stopped') else 'COMPLETE',
        development_unique_runs=sum(r['stage'] in ('calibration','development') for r in results),additional_runs=additional,
        unique_training_runs=costs['unique_scientific_trajectories'],current_attempt_training_runs=len(results),
        full_training_passes=costs['combined']['training_passes'],current_attempt_passes=costs['current_attempt']['training_passes'],
        partial_training_passes=costs['partial_passes'],cost_lower_bound_fields=costs['lower_bound_fields'],
        total_epochs=costs['combined']['training_passes']*50,optimizer_steps=costs['combined']['optimizer_steps'],
        training_worker_seconds=costs['combined']['training_seconds'],task_start_unix=p['task_start_unix'],
        wall_seconds=time.perf_counter()-start,test_evaluated=True,snapshot_tensor_bytes=sum(r['snapshot_tensor_bytes'] for r in results)))
    for result in results:result.pop('states')
    results.clear();gc.collect();write_json(out/'state_release.json',dict(released=True,storage='RAM_ONLY',after='frozen unified test and tables'))


def finish_if_budget(bundles,p,out,results,start,additional):
    if not p.get('budget_stopped'):return False
    write_json(out/'decision.json',dict(status='EARLY_STOP_BUDGET',reason='12h hard dispatch limit',
        complete_unique_trajectories=len(results),test_accessed=False))
    if results:finish_queue(bundles,p,out,results,start,additional)
    else:write_json(out/'completion.json',dict(status='EARLY_STOP_BUDGET',current_attempt_training_runs=0,
        test_evaluated=False,task_start_unix=p['task_start_unix'],reason='Original 12h dispatch limit already reached'))
    return True


def all_stages(bundles,p,out):
    start=time.perf_counter()-(time.time()-p['task_start_unix']);results=[];additional=0;matched={};trial_decisions=[]
    calibration=[]
    for dataset in bundles:
        job=make_jobs('calibration','A0',[dataset],[2022])
        current=schedule(bundles,p,out,job,p['devices']);results.extend(current)
        if finish_if_budget(bundles,p,out,results,start,additional):return
        calibration.append(dict(dataset=dataset,seconds=current[0]['training_seconds'],peak=current[0]['peak_memory_bytes']))
    fresh_forecast=sum(x['seconds'] for x in calibration)*24
    expected_worker=p.get('recovery',{}).get('original_development_forecast_seconds',fresh_forecast)
    write_json(out/'throughput_calibration.json',dict(runs=calibration,complete_50_epochs=True,
        projected_development_worker_seconds=expected_worker,
        fresh_attempt_forecast_seconds=fresh_forecast,original_forecast_retained=bool(p.get('recovery')),
        calculation='2 A0 passes plus 4 passes each A1/B1/B2/B3, factors 1/1.25/2.2/1.05 per dataset'))
    results.extend(schedule(bundles,p,out,make_jobs('development','A0',bundles,[2024]),p['devices']))
    if finish_if_budget(bundles,p,out,results,start,additional):return
    for arm in ARMS[1:]:
        results.extend(schedule(bundles,p,out,make_jobs('development',arm,bundles,DEV_SEEDS),p['devices']))
        if finish_if_budget(bundles,p,out,results,start,additional):return
        if arm=='A1':
            write_json(out/'A_decision.json',{d:dict(ema=compare(values(results,d,'A0'),values(results,d,'A0','online'),d),
                augmentation=compare(values(results,d,'A1'),values(results,d,'A0'),d),
                adaptive_vs_fixed=compare(values(results,d,'A1','adaptive_ema'),values(results,d,'A1'),d),
                adaptive_vs_replay=compare(values(results,d,'A1','adaptive_ema'),values(results,d,'A1','replay_rho'),d)) for d in bundles})
        else:
            write_json(out/f'{arm}_development_decision.json',{d:compare(values(results,d,arm),values(results,d,'A1'),d) for d in bundles})
    assert len(results)==20
    verify_recomputed_results(results,p,out)
    physical=execution_costs(results,p,out)
    prior_worker=physical['prior_attempt']['training_seconds'];current_worker=physical['current_attempt']['training_seconds']
    cost_ratio=(prior_worker+current_worker)/expected_worker
    write_json(out/'budget_calibration.json',dict(development_worker_seconds=prior_worker+current_worker,
        prior_attempt_worker_seconds=prior_worker,current_attempt_worker_seconds=current_worker,
        worker_seconds_lower_bound='training_seconds' in physical['lower_bound_fields'],
        predicted_worker_seconds=expected_worker,cost_ratio=cost_ratio,cancel_tuning=cost_ratio>1.5,
        cancel_second_confirmation=cost_ratio>2.,test_accessed=False))
    historical=historical_baseline_verification(results,p,out)
    preliminary=candidate_decisions(results,p)
    for arm,control_arm in [('B2','R2'),('B1','R1'),('B3','R3')]:
        for dataset in bundles:
            gate=preliminary[dataset][arm]['primary'];item=dict(dataset=dataset,arm=arm,control=control_arm)
            if not gate['control_needed']:
                item.update(status='NOT_TRIGGERED',reason='No practical/exploratory gate or stable registered LL/worst-cell trade-off')
            elif additional+2>p['limits']['additional'] or time.perf_counter()-start>=p['limits']['stop_hours']*3600:
                item.update(status='BUDGET_STOP_UNMATCHED')
            else:
                changes,control=matched_control(results,dataset,arm)
                write_json(out/f'{dataset}_{control_arm}_frozen.json',dict(config_changes=changes,control=control,hash=digest(control),test_accessed=False))
                jobs=make_jobs('matched_control',control_arm,[dataset],DEV_SEEDS,changes,{dataset:control})
                current=schedule(bundles,p,out,jobs,p['devices']);results.extend(current);additional+=2
                if finish_if_budget(bundles,p,out,results,start,additional):return
                matched[(dataset,arm)]=compare(values(results,dataset,arm,seeds=DEV_SEEDS),values(results,dataset,control_arm,seeds=DEV_SEEDS),dataset)
                item.update(status='COMPLETE',comparison=matched[(dataset,arm)])
            trial_decisions.append(item)
            write_json(out/'matched_control_decisions.json',dict(additional_used=additional,decisions=trial_decisions,test_accessed=False))
    decisions=candidate_decisions(results,p,matched);effective={};tuning=[];tuned=set()
    for dataset in bundles:
        for arm in ('B2','B1','B3'):
            item=decisions[dataset][arm];primary=item['primary'];control=item['matched']
            trigger=bool(control and control['matched_exceeded'] and primary['exploratory'])
            record=dict(dataset=dataset,arm=arm,trigger=trigger,status='NOT_TRIGGERED')
            if trigger:
                if primary['passed']:
                    record.update(status='NOT_NEEDED',reason='Default and matched control settle practical promotion; confirmation takes priority')
                elif additional+4>8 or len(tuned-{arm})>=2 or cost_ratio>1.5 or time.perf_counter()-start>=p['limits']['stop_hours']*3600:
                    record.update(status='BUDGET_STOP',reason='Control priority exhausted registered additional/time budget')
                elif arm=='B1':
                    resolution=wait_for_design(out,dict(dataset=dataset,arm=arm,trial='tau2',additional_used=additional,
                        remaining_additional=8-additional),p,start)
                    record.update(status='STOP_BY_COORDINATOR',resolution=resolution)
                else:
                    trial=arm+('_eta05' if arm=='B2' else '_delay10');control_arm='R2_eta05' if arm=='B2' else 'R3_delay10'
                    changes={'eta':.5} if arm=='B2' else {'delay_epochs':10}
                    current=schedule(bundles,p,out,make_jobs('additional_trial',trial,[dataset],DEV_SEEDS,changes),p['devices'])
                    results.extend(current);additional+=2;tuned.add(arm)
                    if finish_if_budget(bundles,p,out,results,start,additional):return
                    if arm=='B2':control_changes,control_data=matched_control(results,dataset,trial)
                    else:
                        off=sorted(np.random.default_rng(1729).choice(np.arange(1,51),10,replace=False).tolist())
                        control_changes={'off_epochs':off};control_data=dict(source='registered delay10 matched random ten-off schedule',off_epochs=off)
                    write_json(out/f'{dataset}_{control_arm}_frozen.json',dict(control=control_data,changes=control_changes,
                        hash=digest(control_data),test_accessed=False))
                    current=schedule(bundles,p,out,make_jobs('additional_control',control_arm,[dataset],DEV_SEEDS,
                        control_changes,{dataset:control_data}),p['devices'])
                    results.extend(current);additional+=2
                    if finish_if_budget(bundles,p,out,results,start,additional):return
                    new_primary=compare(values(results,dataset,trial,seeds=DEV_SEEDS),values(results,dataset,'A1',seeds=DEV_SEEDS),dataset)
                    new_matched=compare(values(results,dataset,trial,seeds=DEV_SEEDS),values(results,dataset,control_arm,seeds=DEV_SEEDS),dataset)
                    default_mean=np.mean([v['macro_auc'] for v in values(results,dataset,arm,seeds=DEV_SEEDS).values()])
                    trial_mean=np.mean([v['macro_auc'] for v in values(results,dataset,trial,seeds=DEV_SEEDS).values()])
                    selected=bool(new_primary['passed'] and new_matched['matched_exceeded'] and trial_mean-default_mean>=.0005)
                    if selected:effective[(dataset,arm)]=trial;matched[(dataset,arm)]=new_matched
                    record.update(status='COMPLETE',trial=trial,primary=new_primary,matched=new_matched,selected=selected)
            tuning.append(record)
    write_json(out/'additional_trial_decisions.json',dict(trials=tuning,additional_used=additional,test_accessed=False))
    decisions=candidate_decisions(results,p,matched,effective)
    candidates=[]
    for arm in ('B2','B1','B3'):
        datasets=[d for d in bundles if decisions[d][arm]['status']=='ELIGIBLE']
        if datasets:
            candidates.append(dict(candidate='C1',arm=arm,kind='fixed_ema',datasets=datasets,
                trials={d:effective.get((d,arm),arm) for d in datasets}));break
    datasets=[d for d in bundles if decisions[d]['C2']['status']=='ELIGIBLE']
    if datasets:candidates.append(dict(candidate='C2',arm='A1',kind='adaptive_ema',datasets=datasets))
    candidates=candidates[:2];confirm_jobs=[];history_datasets=set();reserved=set();confirmation=[]
    if cost_ratio>2.:candidates=candidates[:1]
    for candidate in candidates:
        for dataset in candidate['datasets']:
            trial=candidate.get('trials',{}).get(dataset,candidate['arm'])
            same=any(j['dataset']==dataset and j['trial']==trial for j in confirm_jobs)
            needed=0 if same else 3
            baselines=[]
            if not historical[dataset]:
                for arm in ('A0','A1'):
                    if (dataset,arm) not in reserved and arm!=trial:
                        baselines.extend(make_jobs('confirmation_baseline',arm,[dataset],CONFIRM_SEEDS))
            group_size=len(baselines)+needed
            reason=confirmation_budget_reason(len(results),len(confirm_jobs),group_size)
            if time.perf_counter()-start>=p['limits']['stop_hours']*3600:reason='stop_dispatch_hours_reached'
            if reason:
                confirmation.append(dict(**candidate,dataset=dataset,status='BUDGET_STOP',reason=reason,
                    planned_confirmation_runs=len(confirm_jobs),required_group_runs=group_size,
                    required_new_baselines=len(baselines),required_candidate_runs=needed));continue
            if historical[dataset]:history_datasets.add(dataset)
            confirm_jobs.extend(baselines)
            for job in baselines:reserved.add((dataset,job['trial']))
            if not same:
                changes={'eta':.5} if trial.endswith('_eta05') else {'delay_epochs':10} if trial.endswith('_delay10') else {}
                confirm_jobs.extend(make_jobs('confirmation',trial,[dataset],CONFIRM_SEEDS,changes))
            reserved.add((dataset,trial));confirmation.append(dict(**candidate,dataset=dataset,status='PLANNED',trial=trial))
    assert len(confirm_jobs)<=12 and len(results)+len(confirm_jobs)<=40
    write_json(out/'configuration_frozen.json',dict(decisions=decisions,candidates=candidates,confirmation=confirmation,
        confirmation_jobs=confirm_jobs,additional_used=additional,history_datasets=sorted(history_datasets),
        execution_commit=p['execution_commit'],source_hash=p['source_manifest_hash'],test_accessed=False))
    if confirm_jobs:results.extend(schedule(bundles,p,out,confirm_jobs,p['devices']))
    if finish_if_budget(bundles,p,out,results,start,additional):return
    assert len(results)<=40 and additional<=8
    write_json(out/'decision.json',dict(study_id='GATS02',development=decisions,confirmation=confirmation,
        candidate_limit=2,additional_used=additional,unique_training_runs=len(results),test_accessed=False))
    finish_queue(bundles,p,out,results,start,additional,history_datasets)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['audit','preflight','check','all'],required=True)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--device',default='cuda:1');parser.add_argument('--devices',default='0,1')
    parser.add_argument('--code-commit');parser.add_argument('--stage0-receipt');parser.add_argument('--history-dir',default=str(HISTORY))
    parser.add_argument('--recovery-attempt');parser.add_argument('--task-start-unix',type=float)
    parser.add_argument('--additional-cost-attempt',action='append',default=[])
    args=parser.parse_args();p=protocol();out=Path(args.run_dir);manifest=source_manifest()
    if args.recovery_attempt and args.task_start_unix is None:parser.error('Recovery requires the original --task-start-unix')
    if args.additional_cost_attempt and not args.recovery_attempt:parser.error('Additional raw cost sources require --recovery-attempt as identity reference')
    if args.mode=='preflight' and not args.recovery_attempt:parser.error('Preflight requires --recovery-attempt with original numerical endpoints')
    if args.mode in ('check','all') and not args.code_commit:parser.error('Provide reviewed immutable --code-commit')
    if args.mode=='all':
        if not args.stage0_receipt:parser.error('Provide --stage0-receipt')
        receipt=json.loads(Path(args.stage0_receipt).read_text())
        assert receipt['status']=='PASS' and receipt['source_manifest_hash']==manifest['hash'] and receipt['execution_commit']==args.code_commit
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new attempt: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    p.update(execution_commit=args.code_commit,source_manifest_hash=manifest['hash'],devices=[int(v) for v in args.devices.split(',')],history_dir=args.history_dir)
    p.update(task_clock(args.task_start_unix,p['limits']))
    if args.recovery_attempt:
        p['recovery']=recovery_context(args.recovery_attempt,args.task_start_unix,p,args.additional_cost_attempt)
        preflight=historical_endpoint_core(attempt_endpoint_rows(args.recovery_attempt),p,out,'historical_preflight.json')
        write_json(out/'recovery_execution_receipt.json',dict(task_start_unix=p['task_start_unix'],
            recovery_start_unix=p['recovery_start_unix'],elapsed_before_recovery_seconds=p['elapsed_before_recovery_seconds'],
            stop_dispatch_deadline=p['stop_dispatch_deadline'],dispatch_deadline=p['dispatch_deadline'],
            prior_attempt=p['recovery']['prior_attempt'],prior_costs=p['recovery']['prior_costs'],
            raw_cost_source_paths=[source['source_attempt'] for source in p['recovery']['cost_sources']],
            prior_lower_bound_fields=p['recovery']['prior_lower_bound_fields'],
            prior_complete_passes=sum(source['complete_passes'] for source in p['recovery']['cost_sources']),
            prior_partial_passes=sum(source['partial_passes'] for source in p['recovery']['cost_sources']),
            native_failure_history=p['recovery']['native_failure_history'],
            original_development_forecast_seconds=p['recovery']['original_development_forecast_seconds'],
            original_execution_commit=p['recovery']['original_execution_commit'],
            original_source_hash=p['recovery']['original_source_hash'],execution_commit=args.code_commit,
            source_hash=manifest['hash'],historical_preflight_verified=preflight['verified'],
            historical_test_accessed_for_decision=False,prior_RAM_weights_restored=False))
        oldcost=[{k:r['stats'].get(k) for k in ('dataset','trial','seed','run_hash')+COST_KEYS}
            for r in p['recovery']['runs'].values()]
        pd.DataFrame(oldcost).to_csv(out/'prior_attempt_costs.csv',index=False)
        write_json(out/'prior_physical_cost_sources.json',dict(sources=p['recovery']['cost_sources'],
            source_paths_deduplicated=True,inherited_prior_receipts_added=False))
        pd.DataFrame([row for source in p['recovery']['cost_sources'] for row in source['items']]).to_csv(out/'prior_physical_cost_ledger.csv',index=False)
    write_json(out/'source_manifest.json',dict(execution_commit=args.code_commit,**manifest));write_json(out/'resolved_config.json',p)
    if args.mode=='preflight':
        print(f"historical CPU preflight: {preflight['verified']}; {preflight['matched_endpoints']}/{preflight['endpoint_count']} endpoint matches",flush=True)
        return
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,gpus=gpu_status(),
        tf32=False,cudnn_benchmark=False,cpu_threads=1,training_gpu_event_boundary='batch input through online/shadow update, host gaps included',
        evaluation_gpu_event_boundary='sum of model forward event intervals; CPU metrics excluded'))
    write_json(out/'trial_plan.json',dict(base_runs=20,max_additional=8,max_total=40,seeds=p['seeds'],
        arms={d:{arm:config(arm,d) for arm in ARMS} for d in p['datasets']},random_off_epochs=p['random_off_epochs'],
        off_schedule_seed=1729,numpy_version=np.__version__,state_storage='RAM_ONLY'))
    (out/'failure_ledger.csv').write_text('stage,dataset,trial,seed,device,attempt,oom,error,time,exitcode,signal,recoverable,retry_index,failure_count,prior_native_failures,run_id\n')
    bundles={};manifests={}
    for d in p['datasets']:
        destination=out/'data'/d;destination.mkdir(parents=True,exist_ok=True)
        bundles[d],manifests[d]=load_dataset(d,p,raw_path=p['datasets'][d]['raw_path_default'],out=destination)
        print(f'data identity PASS {d}',flush=True)
    write_json(out/'data_manifest.json',manifests)
    if args.mode=='audit':return
    if args.mode=='check':
        from .checks import run_checks
        answer=run_checks(bundles,p,args.device,out)
        write_json(out/'stage0_checks.json',dict(**answer,source_manifest_hash=manifest['hash'],execution_commit=args.code_commit));return
    try:all_stages(bundles,p,out)
    except Exception as exc:
        costs=execution_costs([],p,out);write_json(out/'execution_cost_continuity.json',costs)
        write_json(out/'completion.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            physical_cost_ledger='physical_cost_ledger.csv',complete_passes=costs['complete_passes'],partial_passes=costs['partial_passes']));raise


if __name__=='__main__':main()
