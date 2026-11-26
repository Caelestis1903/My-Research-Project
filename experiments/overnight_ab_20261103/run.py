"""Prepare, audit, run epoch quanta, and archive all frozen A/B jobs."""
import argparse
import importlib
import json
import os
import platform
import signal
import secrets
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from . import config,guardian
from .state import read_json,write_json,append_json,sha_file,digest,state_digest
from .data_bridge import prepare_cache,cached_bundle,training_view
from .inventory import scan
from .reporting import write_csv,report
from . import queue

PACKAGE='experiments.overnight_ab_20261103'
def family_module(job):
    return importlib.import_module('.published_training' if job['arm'] in ('HYPERGATE','AUTOIFS') else '.worker_b',PACKAGE)
def identity(job):
    return {k:job[k] for k in ('job_id','trajectory_id','arm','config_hash','source_hash','data_hash')}
def prepare(plan,history_root,attempt):
    attempt=Path(attempt).resolve(); attempt.mkdir(parents=True,exist_ok=True)
    p=config.load_plan(plan); plan_hash=digest(p)
    if (attempt/'PREPARED.json').exists():
        old=read_json(attempt/'PREPARED.json')
        if old['plan_hash']!=plan_hash or old['history_root']!=str(Path(history_root).resolve()):
            raise ValueError('Attempt belongs to a different frozen configuration')
        return old
    manifests=prepare_cache(attempt/'data_cache',p)
    inventory=scan(history_root,p); expanded=config.plan_module().compile_plan(p)
    config.plan_module().materialize(p,attempt/'plan')
    jobs=[]; states={}; audits=inventory['audits']
    for row in expanded['jobs']:
        manifest=manifests[row['dataset']]
        data_identity={key:manifest[key] for key in ('raw_sha256','array_hashes','split_row_id_hashes',
            'encoder_hash','scaler_hash','train_domain_counts','split_rows')}
        job=config.make_job(p,row,digest(data_identity))
        imported=any((r['dataset'],r['arm'],r['seed'],r['estimator'],r['checkpoint'],r['split'])==
            (job['dataset'],job['arm'],job['seed'],'fixed_ema','selected','test') for r in inventory['endpoints'])
        status='CONDITIONAL_NOT_ENABLED' if not row['core'] else 'REUSE_DONE' if imported else 'READY'
        if row['arm'] in ('A0','C_MIX','P01') and not imported: status='WAIT_EXTERNAL'
        job['initial_status']=status; jobs.append(job); states[job['job_id']]=dict(status=status,commit_index=0)
        if row['arm'] in ('HYPERGATE','AUTOIFS','C_MIX_P01_SERIAL_V1'):
            audits.append(dict(dataset=row['dataset'],method=row['arm'],seed=row['seed'],legacy=False,
                eligible_main=True,action='REPLAY_MISSING',reason='FULL_NATURAL_PROTOCOL_NEW_TRAJECTORY',
                trajectory_id=job['trajectory_id'],source_hash=job['source_hash']))
    report_root=config.ROOT/'research_reports/2026-11-03_overnight_AB'
    prepared=dict(protocol=p,plan_hash=plan_hash,history_root=str(Path(history_root).resolve()),
        execution_root=str(config.ROOT),report_root=str(report_root),attempt=str(attempt),
        jobs=jobs,source_refs=p['source_refs'],fallback_enabled=False,created=time.time())
    write_json(attempt/'REUSED_ENDPOINTS.json',inventory['endpoints'])
    write_json(attempt/'REUSED_VALIDATION_CURVES.json',inventory['validation_curves'])
    write_json(attempt/'IDENTITY_ALIASES.json',inventory['aliases'])
    a_audits=[r for r in audits if r.get('legacy') or r['method'] in ('HYPERGATE','AUTOIFS')]
    for row in inventory['endpoints']:
        if row['arm']=='A0' and row['estimator']=='online' and row['checkpoint']=='selected' and row['split']=='test':
            a_audits.append(dict(dataset=row['dataset'],method='M3OE_FIXED1_ADAPTED',seed=row['seed'],
                eligible_main=True,legacy=False,action='REUSE_DONE',reason='A0_ONLINE_IDENTITY_ALIAS',
                **{k:row[k] for k in ('trajectory_id','selection_id','epoch','model_hash','source_path','source_sha256')}))
    b_audits=[r for r in audits if not r.get('legacy') and r['method'] in ('A0','C_MIX','P01','C_MIX_P01_SERIAL_V1')]
    fields=('source_commit','source_path','source_sha256','trajectory_id','selection_id','selected_epoch','model_hash',
            'population_match','preprocess_match','budget_match','metric_match','state_available','external_owner')
    for row in a_audits+b_audits:
        for field in fields: row.setdefault(field,None)
        row['selected_epoch']=row.get('selected_epoch') or row.get('epoch')
        row.update(population_match=not row.get('legacy'),preprocess_match=not row.get('legacy'))
    write_csv(attempt/'A_REUSE_AUDIT.csv',a_audits)
    write_csv(attempt/'B_REUSE_AUDIT.csv',b_audits)
    write_json(attempt/'DATA_MANIFEST.json',manifests)
    write_json(attempt/'SOURCE_LOCK.json',dict(source_refs=p['source_refs'],reused_files=inventory['source_files'],
        actual_dependency_closure=config.source_manifest()))
    write_json(attempt/'ENVIRONMENT.json',dict(python=sys.version,python_executable=sys.executable,
        pytorch=torch.__version__,cuda=torch.version.cuda,gpus=subprocess.check_output(['nvidia-smi','-L'],text=True),
        platform=platform.platform()))
    start=Path(history_root)/'start.md'
    write_json(attempt/'RUNTIME_OVERRIDES.json',dict(source_path=str(start),source_sha256=sha_file(start),
        rule='Each GPU independently adds workers below 85% memory and power with measured memory/RAM admission',
        original_max_workers_per_device=1,max_workers_per_device=None,
        shared_compute_owners=[row for row in queue.compute_processes() if row['owner'] and
            row['owner']['pid'] in (762513,784445,810368) and row['command'] and 'flir_jepa_20261102' in row['command']],
        shared_compute_authorization='User start.md per-GPU concurrency; orchestrator confirmed these three existing mwt/flir_jepa jobs'))
    write_json(attempt/'CONFIGURATION_FROZEN.json',dict(protocol=p,plan_hash=plan_hash,data=manifests,
        matrices=expanded,created=time.time(),test_policy=p['common']['test_policy']))
    write_json(attempt/'queue_state.json',dict(jobs=states,next_lane='A',created=time.time()))
    write_json(attempt/'PREPARED.json',prepared)
    write_csv(attempt/'JOBS.csv',[{k:v for k,v in j.items() if not isinstance(v,(dict,list))} for j in jobs])
    report(attempt)
    return prepared
def smoke(attempt,devices):
    attempt=Path(attempt); prepared=read_json(attempt/'PREPARED.json'); records={}
    for job in prepared['jobs']:
        if job['seed']!=2022 or job['arm'] not in ('HYPERGATE','AUTOIFS','C_MIX_P01_SERIAL_V1'): continue
        unknown=queue.unknown_compute(attempt); reserved=queue.external_reservations(attempt)
        dev=next((d for d in devices if d not in reserved and queue.gpu_resources(d)['uuid'] not in unknown
                  and queue.resources_allow(queue.gpu_resources(d),0)),None)
        if dev is None: raise RuntimeError('WAIT_RESOURCE: no GPU fits original-batch smoke')
        resource=queue.gpu_resources(dev)
        with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
            resource=queue.gpu_resources(dev)
            if dev in queue.external_reservations(attempt) or resource['uuid'] in queue.unknown_compute(attempt) or not queue.resources_allow(resource,0):
                raise RuntimeError('WAIT_RESOURCE: smoke admission changed')
            records[job['dataset']+'.'+job['arm']]=smoke_batch(attempt,job,f'cuda:{dev}')
        write_json(attempt/'SMOKE.json',records)
    return records
def smoke_batch(attempt,job,device):
    view=cached_bundle(Path(attempt)/'data_cache',job['dataset'],False)
    module=family_module(job); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device); start=time.time()
    session=module.create_session(job,view,device)
    size=job['config']['batch_size']; x=torch.as_tensor(np.array(view.train_x[:size]),device=device)
    y=torch.as_tensor(np.array(view.train_y[:size]),device=device); ids=np.array(view.train_row_ids[:size])
    if job['arm'] in ('HYPERGATE','AUTOIFS'):
        loss=session['model'].objective(x,y,torch.ones_like(y,dtype=torch.bool))['loss']
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite smoke objective')
        loss.backward()
        for optimizer in session['optimizers'].values(): optimizer.step()
    else:
        from .objectives import train_step
        train_step(session['model'],session['optimizers']['main'],session['ema'],x.T,y,ids,
            job['config'],dict(dataset=job['dataset'],seed=job['seed'],epoch=1,global_step=1),diagnostic=True)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in session['model'].parameters()):
        raise FloatingPointError('Nonfinite smoke gradient')
    torch.cuda.synchronize(device)
    result=dict(status='PASS',batch_size=size,peak_memory_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.time()-start,device=device,source_hash=job['source_hash'])
    del session,x,y; torch.cuda.empty_cache()
    return result
def rebuild_model(job,bundle,device,snapshot):
    if job['arm'] in ('HYPERGATE','AUTOIFS'):
        from .published_models import build_published,restore_model_meta
        model=build_published(job['method'],bundle,job['seed'],job['config']).to(device)
        restore_model_meta(model,snapshot.get('model_meta',{}))
    else:
        from experiments.mdmt_history_information_20261103 import protocol as history
        model=history.make_model(bundle,history.load_protocol()['datasets'][job['dataset']]['source']).to(device)
    model.load_state_dict(snapshot['model_state'])
    if state_digest(model.state_dict())!=snapshot['model_hash']: raise ValueError('Endpoint model hash mismatch')
    return model
def archive(attempt,job,snapshot,device):
    from .evaluation import evaluate_model
    attempt=Path(attempt); root=attempt/'runs'/job['job_id']; root.mkdir(parents=True,exist_ok=True)
    if snapshot['completed_final']!=50 or snapshot['completed_search']!=(10 if job['arm']=='AUTOIFS' else 0):
        raise ValueError('Incomplete training cannot be evaluated')
    frozen={kind:{name:{k:v for k,v in state.items() if k!='model_state'}
        for name,state in points.items()} for kind,points in snapshot['snapshots'].items()}
    commits={row['commit_index']:row['epoch_record_sha'] for row in
        (json.loads(line) for line in (attempt/'epoch_commit_ledger.jsonl').read_text().splitlines())
        if row['key']==job['job_id']}
    files={sha_file(path):path for path in root.glob('launch_*/epoch_*.json')}
    curve_paths=[files[commits[index]] for index in sorted(commits)]
    records=[read_json(path) for path in curve_paths]
    from .selection import score
    for kind,points in snapshot['snapshots'].items():
        final=[row for row in records if row['stage']==points['selected']['stage']]
        if {row['epoch'] for row in final}!=set(range(1,51)): raise ValueError('Complete legal validation window required')
        def values(row): return row['validation'] if 'macro_auc' in row['validation'] else row['validation'][kind]
        winner=max(sorted(final,key=lambda row:row['epoch']),key=lambda row:score(values(row)))
        winner_hash=winner['model_hash'] if isinstance(winner['model_hash'],str) else winner['model_hash'][kind]
        if (winner['epoch'],winner_hash)!=(points['selected']['epoch'],points['selected']['model_hash']):
            raise ValueError('Selected endpoint differs from committed strict validation argmax')
    write_json(root/'selection_frozen.json',dict(configuration_frozen_sha256=sha_file(attempt/'CONFIGURATION_FROZEN.json'),
        trajectory_id=job['trajectory_id'],endpoints=frozen,validation_files={str(f.relative_to(root)):sha_file(f) for f in curve_paths}))
    bundle=cached_bundle(attempt/'data_cache',job['dataset'],True); endpoints=[]
    for kind,points in snapshot['snapshots'].items():
        if set(points)!=set(('selected','E10','E50')): raise ValueError('Required endpoint missing')
        for checkpoint,state in points.items():
            model=rebuild_model(job,bundle,device,state)
            for split in ('validation','test'):
                values=evaluate_model(model,bundle,split,job['method'],job['config'].get('eval_batch_size',40960))
                if split=='validation' and values!=state['validation']: raise ValueError('Frozen validation changed')
                selection_id=digest(dict(trajectory_id=job['trajectory_id'],estimator=kind,selection=job['protocol']['selection']))
                endpoint_id=digest(dict(selection_id=selection_id,checkpoint=checkpoint,stage=state['stage'],
                    epoch=state['epoch'],model_hash=state['model_hash'],split=split))
                endpoints.append(dict(protocol_id=job['protocol']['protocol_id'],data_protocol=job['protocol']['data_protocol'],
                    config_hash=job['config_hash'],source_hash=job['source_hash'],trajectory_id=job['trajectory_id'],
                    selection_id=selection_id,endpoint_id=endpoint_id,dataset=job['dataset'],method=job['method'],
                    arm=job['arm'],role='PUBLISHED_BASELINE' if job['lane']=='A' else 'COMBINED_METHOD',
                    seed=job['seed'],estimator=kind,checkpoint=checkpoint,stage=state['stage'],epoch=state['epoch'],
                    model_hash=state['model_hash'],split=split,source_path=str(root/'selection_frozen.json'),
                    source_sha256=sha_file(root/'selection_frozen.json'),reused=False,independent_run_id=job['trajectory_id'],**values))
            del model; torch.cuda.empty_cache()
    write_json(root/'endpoints.json',endpoints); write_json(root/'cost.json',snapshot['cost'])
    write_json(root/'DONE.json',dict(job_id=job['job_id'],trajectory_id=job['trajectory_id'],completed=time.time(),
        schema_version=1,files={name:sha_file(root/name) for name in ('selection_frozen.json','endpoints.json','cost.json')}))
    guardian.request(attempt,'release',key=job['job_id'])
def run_job(attempt,job_id,device,quantum_epochs=1,dispatch_token=None):
    attempt=Path(attempt); prepared=read_json(attempt/'PREPARED.json')
    job=next(j for j in prepared['jobs'] if j['job_id']==job_id)
    claim=read_json(attempt/'claims'/(job_id+'.json'))
    if not dispatch_token or claim['token']!=dispatch_token or claim['device']!=device or not guardian.alive(claim['coordinator']):
        raise RuntimeError('run-job requires an active coordinator claim')
    if not queue.accepted(attempt,job): raise RuntimeError('Family awaits independent engineering PASS')
    if config.source_manifest(job['family'])['sha256']!=job['source_hash']: raise ValueError('Frozen implementation changed')
    lockfile=attempt/('IMPLEMENTATION_LOCK_'+job['family']+'.json')
    if not lockfile.exists(): write_json(lockfile,dict(family=job['family'],source=config.source_manifest(job['family']),created=time.time()))
    with queue.lock('trajectory-'+job['trajectory_id']+'.lock'):
        gpu_index=int(device.rsplit(':',1)[-1]); resource=queue.gpu_resources(gpu_index)
        with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
            resource=queue.gpu_resources(gpu_index)
            if gpu_index in queue.external_reservations(attempt) or resource['uuid'] in queue.unknown_compute(attempt):
                raise RuntimeError('WAIT_RESOURCE: GPU has an unapproved external owner')
            smoke_row=read_json(attempt/'SMOKE.json')[job['dataset']+'.'+job['arm']]
            remote=guardian.request(attempt,'status')['jobs'].get(job_id,{})
            if not queue.resources_allow(resource,smoke_row['peak_memory_bytes'],remote.get('snapshot_bytes',smoke_row['peak_memory_bytes'])):
                raise RuntimeError('WAIT_RESOURCE: original configuration does not fit admission gates')
        registered=guardian.request(attempt,'register',key=job_id,job=identity(job),pid=os.getpid(),device=device)
        launch=registered['launch_index']; parent=attempt/'runs'/job_id/f'launch_{launch}'
        parent.mkdir(parents=True,exist_ok=True); started=time.time()
        try:
            bundle=cached_bundle(attempt/'data_cache',job['dataset'],False); module=family_module(job)
            state=registered['state']; session=module.restore_session(job,bundle,device,state) if state else module.create_session(job,bundle,device)
            guardian.request(attempt,'allocated',key=job_id,peak_memory_bytes=torch.cuda.memory_reserved(device))
            for _ in range(quantum_epochs):
                if session['complete']: break
                row=module.advance_epoch(session); traces=row.pop('batch_trace',[])
                for trace in traces: append_json(parent/'batch_trace.jsonl',trace)
                epoch_file=parent/f"epoch_{row['commit_index']:03d}.json"; write_json(epoch_file,row)
                state=module.export_session(session)
                guardian.request(attempt,'commit',key=job_id,state=state,identity=identity(job),epoch_record_sha=sha_file(epoch_file))
            if session['complete']:
                guardian.request(attempt,'train_complete',key=job_id)
                archive(attempt,job,state,device)
            else: guardian.request(attempt,'yield',key=job_id)
            write_json(parent/'physical_cost.json',dict(job_id=job_id,launch_index=launch,
                physical_worker_seconds=time.time()-started,commit_index=session['commit_index']))
        except Exception as error:
            failure=dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                seconds=time.time()-started,infrastructure=isinstance(error,(OSError,torch.cuda.OutOfMemoryError)))
            write_json(parent/'failure.json',failure); append_json(attempt/'failures.jsonl',dict(job_id=job_id,**failure))
            guardian.request(attempt,'failed',key=job_id,failure=failure)
            raise
def reconcile(attempt,state,guard):
    for key,row in state['jobs'].items():
        if row['status'] in ('REUSE_DONE','CONDITIONAL_NOT_ENABLED','WAIT_EXTERNAL'): continue
        if (Path(attempt)/'runs'/key/'DONE.json').exists(): row.update(status='DONE'); continue
        remote=guard['jobs'].get(key)
        if remote:
            row.update(status=remote['status'],commit_index=remote['commit_index'],
                worker=remote.get('worker'),failure_count=remote['failure_count'])
            if row['status'] in ('RUNNING','TRAIN_COMPLETE') and not guardian.alive(row.get('worker')):
                failure=dict(error='Worker process exited before completed scalar archive',infrastructure=True)
                answer=guardian.request(attempt,'failed',key=key,failure=failure); row['status']=answer['status']
        elif row.get('commit_index',0)>0:
            row.update(status='STATE_LOST_REPLAY',commit_index=0)
        elif row['status'] in ('CLAIMED','RUNNING') and not guardian.alive(row.get('worker')): row['status']='READY'
def run(attempt,devices,resume=False):
    attempt=Path(attempt); prepared=read_json(attempt/'PREPARED.json')
    with queue.lock('coordinator-'+digest(str(attempt))+'.lock'):
        guardian.ensure_guardian(attempt); processes={}; last_report=0
        def stopping(signum,frame): (attempt/'STOP_AFTER_EPOCH').touch()
        signal.signal(signal.SIGTERM,stopping); signal.signal(signal.SIGINT,stopping)
        while True:
            state=read_json(attempt/'queue_state.json'); guard=guardian.request(attempt,'status')
            reconcile(attempt,state,guard)
            for key,process in list(processes.items()):
                if process.poll() is not None: del processes[key]
            active={key for key,row in state['jobs'].items() if guardian.alive(row.get('worker'))}
            active.update(processes)
            stop=(attempt/'STOP_AFTER_EPOCH').exists()
            remaining=[row for row in state['jobs'].values() if row['status'] not in
                ('DONE','REUSE_DONE','CONDITIONAL_NOT_ENABLED','BLOCKED_ENGINEERING','WAIT_EXTERNAL')]
            if not stop:
                reservations=queue.external_reservations(attempt)
                unknown=queue.unknown_compute(attempt)
                eligible=[job for job in prepared['jobs'] if queue.accepted(attempt,job)]
                for device in devices:
                    if device in reservations: continue
                    resource=queue.gpu_resources(device)
                    if resource['uuid'] in unknown: continue
                    with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
                        resource=queue.gpu_resources(device)
                        smoke_rows=read_json(attempt/'SMOKE.json') if (attempt/'SMOKE.json').exists() else {}
                        job=queue.choose_job(eligible,state['jobs'],state['next_lane'],active)
                        if job is None: continue
                        smoke_row=smoke_rows.get(job['dataset']+'.'+job['arm'])
                        if not smoke_row or smoke_row['status']!='PASS': continue
                        # Wait until each newly launched worker has completed its first allocation.
                        allocation_pending=any(r.get('device')==f'cuda:{device}' and r.get('worker') and
                            guardian.alive(r.get('worker')) and not r.get('allocated')
                            for r in guard['jobs'].values())
                        registering=any(state['jobs'][key].get('device')==device and
                            (key not in guard['jobs'] or (guard['jobs'][key].get('worker') or {}).get('pid')!=proc.pid)
                            for key,proc in processes.items())
                        if allocation_pending or registering: continue
                        job_bytes=guard['jobs'].get(job['job_id'],{}).get('snapshot_bytes',smoke_row['peak_memory_bytes'])
                        if not queue.resources_allow(resource,smoke_row['peak_memory_bytes'],job_bytes): continue
                        key=job['job_id']; state['jobs'][key].update(status='CLAIMED',device=device)
                        token=secrets.token_hex(24); claim_path=attempt/'claims'/(key+'.json')
                        write_json(claim_path,dict(token=token,device=f'cuda:{device}',coordinator=guardian.process_identity(os.getpid())))
                        os.chmod(claim_path,0o600)
                        logroot=attempt/'runs'/key; logroot.mkdir(parents=True,exist_ok=True)
                        with (logroot/'worker.log').open('ab') as log:
                            process=subprocess.Popen([sys.executable,'-m',PACKAGE+'.run','run-job','--attempt',str(attempt),
                                '--job-id',key,'--device',f'cuda:{device}','--quantum-epochs','1','--dispatch-token',token],cwd=config.ROOT,
                                stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,
                                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                        processes[key]=process; active.add(key); state['jobs'][key]['worker']=guardian.process_identity(process.pid)
                        state['next_lane']='B' if job['lane']=='A' else 'A'
                        append_json(attempt/'events.jsonl',dict(event='DISPATCH',job_id=key,pid=process.pid,
                            device=device,time=time.time(),lane=job['lane']))
            write_json(attempt/'queue_state.json',state)
            if time.time()-last_report>=60 or not remaining or (stop and not active):
                report(attempt); last_report=time.time()
            if (not remaining and not active) or (stop and not active): break
            time.sleep(2 if active else 15)
        return report(attempt)
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=
        ('prepare','smoke','run','run-job','status','pause','resume','retry','report','_guardian'))
    parser.add_argument('--attempt',type=Path,required=True); parser.add_argument('--plan',type=Path)
    parser.add_argument('--history-root',type=Path); parser.add_argument('--devices',default='0,1')
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--job-id'); parser.add_argument('--quantum-epochs',type=int,default=1)
    parser.add_argument('--dispatch-token',help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    devices=[int(d) for d in args.devices.split(',')]
    if args.command=='prepare':
        if args.plan is None or args.history_root is None: parser.error('prepare requires --plan and --history-root')
        value=prepare(args.plan,args.history_root,args.attempt); print({'prepared':str(args.attempt),'jobs':len(value['jobs'])})
    elif args.command=='_guardian': guardian.serve(args.attempt)
    elif args.command=='smoke': print(smoke(args.attempt,devices))
    elif args.command=='run-job':
        if not args.job_id: parser.error('run-job requires --job-id')
        run_job(args.attempt,args.job_id,args.device,args.quantum_epochs,args.dispatch_token)
    elif args.command in ('run','resume'):
        if args.command=='resume': (args.attempt/'STOP_AFTER_EPOCH').unlink(missing_ok=True)
        print(run(args.attempt,devices,args.resume))
    elif args.command=='pause': (args.attempt/'STOP_AFTER_EPOCH').touch(); print('STOP_AFTER_EPOCH')
    elif args.command=='retry':
        state=read_json(args.attempt/'queue_state.json'); row=state['jobs'][args.job_id]
        if row['status'] not in ('BLOCKED_ENGINEERING','RETRY_PENDING','WAIT_RESOURCE'): raise ValueError('Job is not retryable')
        guard=guardian.request(args.attempt,'status')
        if args.job_id in guard['jobs']: guardian.request(args.attempt,'retry',key=args.job_id)
        row['status']='READY'; write_json(args.attempt/'queue_state.json',state)
        append_json(args.attempt/'events.jsonl',dict(event='MANUAL_RETRY',job_id=args.job_id,time=time.time()))
    elif args.command=='status': print(read_json(args.attempt/'queue_state.json'))
    elif args.command=='report': print(report(args.attempt))
if __name__=='__main__': main()

