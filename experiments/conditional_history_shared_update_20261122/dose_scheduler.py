"""Foreground admission of A_DOSE workers using the frozen core telemetry and fits."""
import fcntl
import os
import signal
import subprocess
import sys
import time

from .dose_stage import jobs, stopping
from .protocol import ROOT, append, digest, read, write
from .scheduler import admit, sample, verify_selected
from .state import request

MODULE = 'experiments.conditional_history_shared_update_20261122.dose_stage'

def supervise(art, stage, reviewed):
    art = art.resolve()
    attempt = art/'attempt1'
    lock = (art/f'{stage}_SUPERVISION.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen = jobs(stage,reviewed['source_hash'])
    status = request(attempt,'status')['jobs']
    queue = []
    for job in frozen:
        key = job['key']
        job_dir = attempt/'runs'/key
        rec = status.get(key)
        if rec and rec['status']=='COMPLETE':
            verify_selected(attempt,job)
        elif (job_dir/'COMPLETE_VERIFIED.json').is_file() and not rec:
            # Lost owner. Accepted scalars can be reused; reconstruct selected only when needed.
            assert read(job_dir/'COMPLETE_VERIFIED.json')['result_hash']==digest(read(job_dir/'result.json'))
        else:
            if rec and rec['status']=='RUNNING':
                pid = rec.get('worker_pid')
                from experiments.conditional_information_momentum_20261102_v3.guardian import alive
                if alive(pid):
                    raise RuntimeError(f'Existing worker {pid} for {key}')
                request(attempt,'paused',key=key,reason='DEAD_WORKER_RESUME_LAST_ACK')
            queue.append(job)
    live = {}
    failed = []
    policy = read(art/f'{stage}_CHECK.json')
    usable = policy['cuda_usable_total_bytes']
    profiles = dict(policy['full_process_bytes'])
    events = art/f'{stage}_admissions.jsonl'
    if events.exists():
        import json
        for line in events.read_text().splitlines():
            record = json.loads(line)
            for dataset,observed in record.get('full_process_bytes',{}).items():
                profiles[dataset] = max(profiles[dataset],observed)
    pause_path = art/f'{stage}_PAUSE'
    def stop_signal(_sig,_frame):
        write(pause_path,dict(reason='SUPERVISOR_INTERRUPTED',time=time.time()))
    old = {s:signal.signal(s,stop_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        while queue or live:
            measurements = sample()
            jobs_status = request(attempt,'status')['jobs']
            for key,(proc,job,gpu,log) in list(live.items()):
                measured = [p['bytes'] for p in measurements['processes'] if p['pid']==proc.pid]
                if measured:
                    role = job['dataset']
                    profiles[role] = max(profiles[role],max(measured))
                rc = proc.poll()
                if rc is not None:
                    log.close()
                    del live[key]
                    append(events,dict(event='worker_exit',key=key,returncode=rc,device=gpu,time=time.time(),
                                       full_process_bytes=profiles))
                    if rc or (not stopping(art,stage) and jobs_status.get(key,{}).get('status') != 'COMPLETE' and request(attempt,'status')['jobs'].get(key,{}).get('status') != 'COMPLETE'):
                        failed.append(key)
                        write(pause_path,dict(reason='FAILED_WORKER',key=key,returncode=rc))
            if stopping(art,stage):
                if not live:
                    break
                time.sleep(1)
                continue
            waiting = {gpu for key,(proc,job,gpu,log) in live.items()
                       if not (jobs_status.get(key,{}).get('worker_pid')==proc.pid and jobs_status[key].get('allocated'))}
            for row in measurements['devices']:
                gpu = row['index']
                if gpu in waiting:
                    continue
                choice = next((j for j in queue if admit(row,measurements['host_ram_used'],
                              usable[str(gpu)],profiles[j['dataset']])),None)
                if choice is None:
                    continue
                queue.remove(choice)
                log = (art/f"{choice['key']}.log").open('ab')
                proc = subprocess.Popen([sys.executable,'-m',MODULE,'_worker','--artifacts',str(art),
                    '--stage',stage,'--key',choice['key'],'--gpu',str(gpu),'--parent',str(os.getpid())],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                live[choice['key']] = (proc,choice,gpu,log)
                append(events,dict(event='admitted',dataset=choice['dataset'],key=choice['key'],pid=proc.pid,device=gpu,
                                   telemetry=measurements,physical_fit_bytes=profiles[choice['dataset']],
                                   cuda_usable_total=usable[str(gpu)]))
                break  # Resample all cards after every startup.
            time.sleep(1)
    finally:
        if live:
            write(pause_path,dict(reason='SUPERVISOR_EXIT',time=time.time()))
        for proc,job,gpu,log in live.values():
            proc.wait()
            log.close()
        for s,handler in old.items():
            signal.signal(s,handler)
        lock.close()
    write(art/f'{stage}_RUN.json',dict(status='PAUSED' if stopping(art,stage) else 'COMPLETE',
            failed=failed,pending=[j['key'] for j in queue],full_process_bytes=profiles,time=time.time()))
    if failed:
        raise RuntimeError(f'Failed jobs {failed}; see worker logs and last RAM ACKs')
