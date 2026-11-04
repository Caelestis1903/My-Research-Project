"""Review-gated ten-run adapter using the existing epoch RAM guardian."""
import argparse
import hashlib
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiments.overnight_ab_20261103 import guardian, queue
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.evaluation import evaluate_model
from experiments.overnight_ab_20261103.inventory import read_csv, scalar_row
from experiments.overnight_ab_20261103.selection import score
from experiments.overnight_ab_20261103.state import read_json, write_json, append_json, sha_file, digest, state_digest
from . import config, worker
from .reporting import summarize, committed_chain

PACKAGE = 'experiments.gaussian_domain_coverage_20261104'


def identity(job):
    return {k:job[k] for k in ('job_id','trajectory_id','arm','config_hash','source_hash','data_hash')}


def prepare(attempt, history_root=config.HISTORY, cache_root=config.CACHE):
    attempt, history_root, cache_root = Path(attempt).resolve(), Path(history_root).resolve(), Path(cache_root).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    source = config.source_manifest()
    if (attempt/'PREPARED.json').exists():
        old = read_json(attempt/'PREPARED.json')
        if old['source_hash']!=source['sha256'] or old['history_root']!=str(history_root) or old['cache_root']!=str(cache_root):
            raise ValueError('Attempt belongs to a different frozen source or input')
        return old
    manifest_file = cache_root/'kuairand/manifest.json'
    if sha_file(manifest_file)!='33a9b0f2866a0c2440481df4bd3315b041b4f85042a06c6a4652972be44fb08d':
        raise ValueError('Historical data manifest changed')
    manifest = read_json(manifest_file)
    expected = read_json(config.SOURCE/'vendor_ga31/resolved_protocol.json')['datasets']['kuairand']
    if sha_file(manifest['raw_path'])!=expected['raw_sha256']:
        raise ValueError('Raw CSV identity changed')
    for field in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes',
                  'encoder_hash','scaler_hash','train_domain_counts','split_rows'):
        if manifest[field]!=expected[field]:
            raise ValueError('Data identity differs: '+field)
    for name, expected_sha in manifest['cache_sha256'].items():
        if sha_file(cache_root/'kuairand'/(name+'.npy'))!=expected_sha:
            raise ValueError('Cached data changed: '+name)
    if manifest['train_domain_counts']!=[172782,1743898,89452] or manifest['split_rows']['train']!=2006132:
        raise ValueError('Domain mapping or train population changed')
    frozen_path = history_root/'artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720/formal_configuration_frozen.json'
    if sha_file(frozen_path)!='86ef0e2313887ccd91aa712b81d6585e4123fc7f8ece80720695de1c3a1b5fbd':
        raise ValueError('Original Gaussian configuration changed')
    original = read_json(frozen_path)['configurations']['kuairand']['A1']
    c = original['config']
    if not (original['kind']=='ema995' and c['use_augmentation'] and c['use_shadow_ema']
            and not c['use_sad'] and not c['use_aig_m'] and c['lr']==.01 and c['weight_decay']==1e-5
            and c['noise_scale']==.1 and c['lambda_aug']==.5 and c['shadow_rates']['ema995']==.995):
        raise ValueError('Historical G-D2 optimizer/Gaussian/EMA identity differs')
    endpoint_file = config.ROOT/'research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv'
    if sha_file(endpoint_file)!='eb260009b1172f3fb4ed25a12cc1836c06a1ffb24bf583cea1e7e08de4943fce':
        raise ValueError('Signed AB endpoint table changed')
    aliases = {('GAUSSIAN_A1','fixed_ema'):'G-D2', ('A0','fixed_ema'):'FIXED_EMA',
               ('A0','online'):'M3OE', ('HYPERGATE','online'):'HYPERGATE', ('AUTOIFS','online'):'AUTOIFS'}
    reused, audit, verified = [], [], {}
    for row in map(scalar_row, read_csv(endpoint_file)):
        alias = aliases.get((row['arm'],row['estimator']))
        if row['dataset']!='kuairand' or alias is None:
            continue
        path = Path(row['source_path'])
        if not path.is_absolute():
            path = history_root/path
        if path not in verified:
            verified[path] = sha_file(path)
        if verified[path]!=row['source_sha256']:
            raise ValueError('Reused endpoint source changed: '+str(path))
        original_arm = row['arm']
        row.update(arm=alias, source_arm=original_arm, reused=True, new_training=False)
        reused.append(row)
        audit.append({k:row[k] for k in ('arm','seed','estimator','checkpoint','split','epoch',
                                       'model_hash','trajectory_id','source_path','source_sha256')})
    for arm in aliases.values():
        current = [r for r in reused if r['arm']==arm]
        if len(current)!=30 or {int(r['seed']) for r in current}!=set(config.SEEDS):
            raise ValueError('Historical five-seed endpoint matrix is incomplete: '+arm)
        for seed in config.SEEDS:
            for checkpoint in ('selected','E10','E50'):
                peers = [r for r in current if r['seed']==seed and r['checkpoint']==checkpoint]
                if len(peers)!=2 or len({(r['epoch'],r['model_hash']) for r in peers})!=1:
                    raise ValueError('Reused validation/test identity mismatch')
    data_identity = {k:manifest[k] for k in ('raw_sha256','array_hashes','split_row_id_hashes',
                    'encoder_hash','scaler_hash','train_domain_counts','split_rows')}
    jobs = [config.make_job(arm,seed,digest(data_identity),source['sha256'])
            for seed in config.SEEDS for arm in config.ARMS]
    (attempt/'data_cache').symlink_to(cache_root, target_is_directory=True)
    prepared = dict(protocol=config.protocol(), jobs=jobs, source_hash=source['sha256'],
        history_root=str(history_root), cache_root=str(cache_root), data=manifest,
        endpoint_table=dict(path=str(endpoint_file),sha256=sha_file(endpoint_file)),
        original_gaussian_configuration=dict(path=str(frozen_path),sha256=sha_file(frozen_path),config=original),
        created=time.time(), report_root=str(config.REPORT), execution_root=str(config.ROOT))
    write_json(attempt/'REUSED_ENDPOINTS.json', reused)
    write_json(attempt/'REUSE_AUDIT.json', dict(endpoints=audit, source_files={str(p):h for p,h in verified.items()},
        data_match=True, optimizer_match=True, validation_selected_protocol='AUC_SELECTED_V1',
        gaussian_source='original_GA31_R3', original_target_domain=2))
    write_json(attempt/'SOURCE_LOCK.json', source)
    write_json(attempt/'DATA_MANIFEST.json', manifest)
    write_json(attempt/'CONFIGURATION_FROZEN.json', prepared)
    write_json(attempt/'RUNTIME_OVERRIDES.json', dict(shared_compute_owners=queue.compute_processes(),
        shared_compute_authorization='User start.md per-GPU concurrency below 85% memory and power; preserve existing tasks',
        max_workers_per_device=None))
    write_json(attempt/'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable,
        pytorch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
        gpus=subprocess.check_output(['nvidia-smi','-L'],text=True)))
    write_json(attempt/'queue_state.json', dict(jobs={j['job_id']:dict(status='READY',commit_index=0) for j in jobs}))
    write_json(attempt/'PREPARED.json', prepared)
    return prepared


def unknown_compute(attempt):
    approved = read_json(Path(attempt)/'RUNTIME_OVERRIDES.json')['shared_compute_owners']
    unknown = {}
    for row in queue.compute_processes():
        if row['owner'] and row['owner']['pid']==os.getpid():
            continue
        if row['command'] and PACKAGE in row['command'] and str(Path(attempt)) in row['command']:
            continue
        if any(row['owner']==old['owner'] and row['command']==old['command'] and
               row['gpu_uuid']==old['gpu_uuid'] for old in approved):
            continue
        unknown.setdefault(row['gpu_uuid'],[]).append(row)
    return unknown


def smoke(attempt, devices):
    config.review_check()
    attempt = Path(attempt)
    prepared = read_json(attempt/'PREPARED.json')
    bundle = cached_bundle(attempt/'data_cache','kuairand',False)
    records = {}
    for arm in config.ARMS:
        job = next(j for j in prepared['jobs'] if j['arm']==arm and j['seed']==2022)
        while True:
            unknown = unknown_compute(attempt)
            reservations = queue.external_reservations(attempt)
            device = next((d for d in devices if d not in reservations and
                queue.gpu_resources(d)['uuid'] not in unknown and queue.resources_allow(queue.gpu_resources(d),2<<30)),None)
            if device is not None:
                break
            write_json(attempt/'smoke_wait.json',dict(status='WAIT_RESOURCE',time=time.time()))
            time.sleep(15)
        torch.cuda.set_device(device)
        resource = queue.gpu_resources(device)
        with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
            if not queue.resources_allow(queue.gpu_resources(device),2<<30):
                raise RuntimeError('WAIT_RESOURCE: smoke admission changed')
            session = worker.create_session(job,bundle,f'cuda:{device}')
            torch.cuda.reset_peak_memory_stats(device)
            # Include all three natural domains while retaining the source batch size.
            ids = np.random.default_rng(2022).permutation(len(bundle.train_y))[:40960]
            x = torch.as_tensor(np.array(bundle.train_x[ids]),device=f'cuda:{device}').T
            y = torch.as_tensor(np.array(bundle.train_y[ids]),device=f'cuda:{device}')
            value = worker.train_step(session['model'],session['optimizers']['main'],session['ema'],
                x,y,bundle.train_row_ids[ids],job['config'],2022,1)
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in session['model'].parameters()):
                raise FloatingPointError('Nonfinite smoke gradient')
            snapshot = worker.export_session(session)
            restored = worker.restore_session(job,bundle,f'cuda:{device}',snapshot)
            if state_digest(worker.export_session(restored))!=state_digest(snapshot):
                raise ValueError('Real-batch RAM state roundtrip changed')
            torch.cuda.synchronize(device)
            records[arm] = dict(status='PASS',batch_size=40960,device=f'cuda:{device}',
                peak_memory_bytes=torch.cuda.max_memory_reserved(device),source_hash=job['source_hash'],
                loss=value['loss'],views=value['views'],optimizer_steps=1,ema_updates=1,
                init_hash=session['init_hash'])
            del restored, snapshot, session, x, y
            torch.cuda.empty_cache()
        write_json(attempt/'SMOKE.json',records)
    return records


def archive(attempt, job, snapshot, device):
    attempt = Path(attempt)
    root = attempt/'runs'/job['job_id']
    if snapshot['completed_final']!=50 or snapshot['global_step']!=2450:
        raise ValueError('Incomplete trajectory cannot be evaluated')
    ledger = committed_chain(attempt,job['job_id'])
    if [r['commit_index'] for r in ledger]!=list(range(1,51)):
        raise ValueError('The committed 50-epoch validation window is incomplete')
    files = {sha_file(path):path for path in root.glob('launch_*/epoch_*.json')}
    paths = [files[r['epoch_record_sha']] for r in ledger]
    records = [read_json(path) for path in paths]
    for kind, points in snapshot['snapshots'].items():
        winner = max(records,key=lambda row:score(row['validation'][kind]))
        if (winner['epoch'],winner['model_hash'][kind])!=(points['selected']['epoch'],points['selected']['model_hash']):
            raise ValueError('Selected state differs from committed validation argmax')
    frozen = {kind:{name:{k:v for k,v in state.items() if k!='model_state'}
                   for name,state in points.items()} for kind,points in snapshot['snapshots'].items()}
    write_json(root/'selection_frozen.json',dict(trajectory_id=job['trajectory_id'],endpoints=frozen,
        configuration_sha256=sha_file(attempt/'CONFIGURATION_FROZEN.json'),
        validation_files={str(p.relative_to(root)):sha_file(p) for p in paths}))
    bundle = cached_bundle(attempt/'data_cache','kuairand',True)
    session = worker.create_session(job,bundle,device)
    model = session['model']
    endpoints = []
    for kind, points in snapshot['snapshots'].items():
        if set(points)!=set(('selected','E10','E50')):
            raise ValueError('A required diagnostic endpoint is missing')
        for checkpoint, endpoint in points.items():
            model.load_state_dict(endpoint['model_state'])
            if state_digest(model.state_dict())!=endpoint['model_hash']:
                raise ValueError('Endpoint model hash mismatch')
            selection_id = digest(dict(trajectory_id=job['trajectory_id'],estimator=kind,selection=job['protocol']['selection']))
            for split in ('validation','test'):
                values = evaluate_model(model,bundle,split,job['arm'],job['config']['eval_batch_size'])
                if split=='validation' and values!=endpoint['validation']:
                    raise ValueError('Frozen validation endpoint changed')
                endpoints.append(dict(dataset='kuairand',arm=job['arm'],method=job['arm'],role='RESEARCH_METHOD',
                    seed=job['seed'],estimator=kind,checkpoint=checkpoint,split=split,stage='train',epoch=endpoint['epoch'],
                    model_hash=endpoint['model_hash'],config_hash=job['config_hash'],source_hash=job['source_hash'],
                    protocol_id=job['protocol']['protocol_id'],data_protocol=job['protocol']['data_protocol'],
                    trajectory_id=job['trajectory_id'],selection_id=selection_id,
                    endpoint_id=digest(dict(selection_id=selection_id,checkpoint=checkpoint,split=split,
                        epoch=endpoint['epoch'],model_hash=endpoint['model_hash'])),
                    source_path=str(root/'selection_frozen.json'),source_sha256=sha_file(root/'selection_frozen.json'),
                    reused=False,new_training=True,independent_run_id=job['trajectory_id'],**values))
    write_json(root/'endpoints.json',endpoints)
    write_json(root/'cost.json',snapshot['cost'])
    write_json(root/'DONE.json',dict(job_id=job['job_id'],trajectory_id=job['trajectory_id'],completed=time.time(),
        files={name:sha_file(root/name) for name in ('selection_frozen.json','endpoints.json','cost.json')}))
    guardian.request(attempt,'release',key=job['job_id'])


def run_job(attempt, job_id, device, token, quantum_epochs=1):
    config.review_check()
    attempt = Path(attempt)
    prepared = read_json(attempt/'PREPARED.json')
    job = next(j for j in prepared['jobs'] if j['job_id']==job_id)
    if config.source_manifest()['sha256']!=job['source_hash']:
        raise ValueError('Frozen implementation changed')
    claim = read_json(attempt/'claims'/(job_id+'.json'))
    if not token or token!=claim['token'] or device!=claim['device'] or not guardian.alive(claim['coordinator']):
        raise RuntimeError('A worker requires its active coordinator claim')
    with queue.lock('trajectory-'+job['trajectory_id']+'.lock'):
        gpu = int(device.rsplit(':',1)[-1])
        resource = queue.gpu_resources(gpu)
        with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
            resource = queue.gpu_resources(gpu)
            if gpu in queue.external_reservations(attempt) or resource['uuid'] in unknown_compute(attempt):
                raise RuntimeError('WAIT_RESOURCE: external owner changed')
            measured = read_json(attempt/'SMOKE.json')[job['arm']]
            remote = guardian.request(attempt,'status')['jobs'].get(job_id,{})
            if not queue.resources_allow(resource,measured['peak_memory_bytes'],remote.get('snapshot_bytes',measured['peak_memory_bytes'])):
                raise RuntimeError('WAIT_RESOURCE: original configuration does not fit')
        registered = guardian.request(attempt,'register',key=job_id,job=identity(job),pid=os.getpid(),device=device)
        launch = registered['launch_index']
        parent = attempt/'runs'/job_id/f'launch_{launch}'
        parent.mkdir(parents=True,exist_ok=True)
        start = time.time()
        try:
            bundle = cached_bundle(attempt/'data_cache','kuairand',False)
            snapshot = registered['state']
            session = worker.restore_session(job,bundle,device,snapshot) if snapshot else worker.create_session(job,bundle,device)
            guardian.request(attempt,'allocated',key=job_id,peak_memory_bytes=torch.cuda.memory_reserved(device))
            for _ in range(quantum_epochs):
                if session['complete']:
                    break
                row = worker.advance_epoch(session)
                traces = row.pop('batch_trace')
                for trace in traces:
                    append_json(parent/'batch_trace.jsonl',trace)
                path = parent/f"epoch_{row['commit_index']:03d}.json"
                write_json(path,row)
                snapshot = worker.export_session(session)
                guardian.request(attempt,'commit',key=job_id,state=snapshot,identity=identity(job),epoch_record_sha=sha_file(path))
            if session['complete']:
                guardian.request(attempt,'train_complete',key=job_id)
                archive(attempt,job,snapshot,device)
            else:
                guardian.request(attempt,'yield',key=job_id)
            write_json(parent/'physical_cost.json',dict(job_id=job_id,launch_index=launch,
                physical_worker_seconds=time.time()-start,commit_index=session['commit_index']))
        except Exception as error:
            failure = dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                seconds=time.time()-start,infrastructure=isinstance(error,(OSError,torch.cuda.OutOfMemoryError)))
            write_json(parent/'failure.json',failure)
            append_json(attempt/'failures.jsonl',dict(job_id=job_id,**failure))
            guardian.request(attempt,'failed',key=job_id,failure=failure)
            raise


def preserve_lost_run(attempt, key):
    """Keep failed physical work while a fresh guardian restarts its launch numbering."""
    attempt = Path(attempt)
    parent = attempt/'runs'/key
    history = attempt/'replay_history'/key
    history.mkdir(parents=True,exist_ok=True)
    destination = history/f'replay_{len(list(history.glob("replay_*")))+1}'
    chain = committed_chain(attempt,key)
    parent.rename(destination)
    write_json(destination/'STATE_LOST_REPLAY.json',dict(job_id=key,reason='GUARDIAN_RAM_STATE_LOST',
        discarded_commits=chain,scientific_job_identity_unchanged=True,time=time.time()))
    append_json(attempt/'events.jsonl',dict(event='STATE_LOST_REPLAY',job_id=key,
        preserved_scalar_path=str(destination),discarded_epochs=len(chain),time=time.time()))


def reconcile(attempt, state, remote):
    for key, row in state['jobs'].items():
        parent = Path(attempt)/'runs'/key
        if (parent/'DONE.json').exists():
            done = read_json(parent/'DONE.json')
            if done['job_id']!=key or any(sha_file(parent/name)!=expected for name,expected in done['files'].items()):
                raise ValueError('Completed scalar archive changed: '+key)
            if key in remote['jobs'] and remote['jobs'][key]['status']!='DONE':
                guardian.request(attempt,'release',key=key)
            row['status']='DONE'
            continue
        other = remote['jobs'].get(key)
        if other:
            row.update(status=other['status'],commit_index=other['commit_index'],
                       worker=other.get('worker'),failure_count=other['failure_count'])
            if row['status'] in ('RUNNING','TRAIN_COMPLETE') and not guardian.alive(row.get('worker')):
                answer=guardian.request(attempt,'failed',key=key,
                    failure=dict(error='Worker exited before scalar archive completed',infrastructure=True))
                row['status']=answer['status']
        elif not guardian.alive(row.get('worker')) and (row.get('commit_index',0)>0 or list(parent.glob('launch_*'))):
            preserve_lost_run(attempt,key)
            row.update(status='STATE_LOST_REPLAY',commit_index=0)
        elif row['status'] in ('CLAIMED','RUNNING') and not guardian.alive(row.get('worker')):
            row['status']='READY'


def run(attempt, devices):
    config.review_check()
    attempt = Path(attempt)
    prepared = read_json(attempt/'PREPARED.json')
    if prepared['source_hash']!=config.source_manifest()['sha256']:
        raise ValueError('Prepared source changed')
    with queue.lock('coordinator-'+digest(str(attempt))+'.lock'):
        guardian.ensure_guardian(attempt)
        processes = {}
        def stopping(signum, frame):
            (attempt/'STOP_AFTER_EPOCH').touch()
        signal.signal(signal.SIGTERM,stopping)
        signal.signal(signal.SIGINT,stopping)
        while True:
            state = read_json(attempt/'queue_state.json')
            remote = guardian.request(attempt,'status')
            reconcile(attempt,state,remote)
            for key, process in list(processes.items()):
                if process.poll() is not None:
                    del processes[key]
            active = {key for key,row in state['jobs'].items() if guardian.alive(row.get('worker'))}
            active.update(processes)
            remaining = [r for r in state['jobs'].values() if r['status']!='DONE']
            stop = (attempt/'STOP_AFTER_EPOCH').exists()
            blocked = [key for key,r in state['jobs'].items() if r['status']=='BLOCKED_ENGINEERING']
            if blocked:
                raise RuntimeError('Worker requires an engineering fix: '+str(blocked))
            if not stop:
                unknown, reservations = unknown_compute(attempt), queue.external_reservations(attempt)
                for device in devices:
                    resource = queue.gpu_resources(device)
                    if device in reservations or resource['uuid'] in unknown:
                        continue
                    with queue.lock(f"dispatch-{platform.node()}-{resource['uuid']}.lock",blocking=True):
                        resource = queue.gpu_resources(device)
                        job = queue.choose_job(prepared['jobs'],state['jobs'],'B',active)
                        if job is None:
                            continue
                        measured = read_json(attempt/'SMOKE.json')[job['arm']]
                        allocating = any(r.get('device')==f'cuda:{device}' and guardian.alive(r.get('worker'))
                            and not r.get('allocated') for r in remote['jobs'].values())
                        registering = any(state['jobs'][key].get('device')==device and
                            (key not in remote['jobs'] or (remote['jobs'][key].get('worker') or {}).get('pid')!=p.pid)
                            for key,p in processes.items())
                        if allocating or registering:
                            continue
                        snapshot_bytes = remote['jobs'].get(job['job_id'],{}).get('snapshot_bytes',measured['peak_memory_bytes'])
                        if not queue.resources_allow(resource,measured['peak_memory_bytes'],snapshot_bytes):
                            continue
                        key, token = job['job_id'], secrets.token_hex(24)
                        state['jobs'][key].update(status='CLAIMED',device=device)
                        path = attempt/'claims'/(key+'.json')
                        write_json(path,dict(token=token,device=f'cuda:{device}',coordinator=guardian.process_identity(os.getpid())))
                        os.chmod(path,0o600)
                        root = attempt/'runs'/key
                        root.mkdir(parents=True,exist_ok=True)
                        with (root/'worker.log').open('ab') as log:
                            process = subprocess.Popen([sys.executable,'-m',PACKAGE+'.run','run-job',
                                '--attempt',str(attempt),'--job-id',key,'--device',f'cuda:{device}',
                                '--dispatch-token',token],cwd=config.ROOT,stdout=log,stderr=log,
                                stdin=subprocess.DEVNULL,start_new_session=True,
                                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                        processes[key]=process
                        active.add(key)
                        state['jobs'][key]['worker']=guardian.process_identity(process.pid)
                        append_json(attempt/'events.jsonl',dict(event='DISPATCH',job_id=key,pid=process.pid,
                            device=device,time=time.time(),gpu_resources=resource))
            write_json(attempt/'queue_state.json',state)
            write_json(attempt/'progress.json',dict(status='RUNNING',
                completed=sum(r['status']=='DONE' for r in state['jobs'].values()),total=10,
                committed_epochs=sum(r.get('commit_index',0) for r in state['jobs'].values()),
                active=sorted(active),time=time.time()))
            if not remaining and not active:
                value = summarize(attempt)
                write_json(attempt/'COMPLETE.json',dict(**value,finished=time.time()))
                write_json(attempt/'SHUTDOWN.json',guardian.request(attempt,'shutdown'))
                return value
            if stop and not active:
                return dict(status='PAUSED_RAM')
            time.sleep(2 if active else 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','smoke','run','resume','status','summarize','manifest','run-job'))
    parser.add_argument('--attempt',type=Path,default=config.ATTEMPT)
    parser.add_argument('--history-root',type=Path,default=config.HISTORY)
    parser.add_argument('--cache-root',type=Path,default=config.CACHE)
    parser.add_argument('--devices',default='0,1')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--job-id')
    parser.add_argument('--dispatch-token',help=argparse.SUPPRESS)
    args = parser.parse_args()
    devices = [int(v) for v in args.devices.split(',')]
    if args.command=='prepare':
        value=prepare(args.attempt,args.history_root,args.cache_root)
        print(dict(jobs=len(value['jobs']),source_hash=value['source_hash']))
    elif args.command=='smoke':
        print(smoke(args.attempt,devices))
    elif args.command in ('run','resume'):
        if args.command=='resume':
            (args.attempt/'STOP_AFTER_EPOCH').unlink(missing_ok=True)
        print(run(args.attempt,devices))
    elif args.command=='status':
        print(read_json(args.attempt/'progress.json') if (args.attempt/'progress.json').exists() else read_json(args.attempt/'queue_state.json'))
    elif args.command=='summarize':
        print(summarize(args.attempt))
    elif args.command=='manifest':
        print(json.dumps(config.source_manifest(),indent=2))
    elif args.command=='run-job':
        run_job(args.attempt,args.job_id,args.device,args.dispatch_token)


if __name__=='__main__':
    main()
