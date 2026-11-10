"""External resource-admitted supervisor; scientific package remains unchanged."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.ema_causal_20261110.protocol import ROOT, CONDITIONS, SEEDS, ANCHORS, WINDOWS, schedule, peak_active_bytes, implementation_identity
from functools import lru_cache
from experiments.ema_causal_20261110.state import request, alive, process_identity, disk_admit
from experiments.ema_mechanism_20261106.run import resource_admission
from experiments.overnight_ab_20261103.state import read_json, write_json, append_json

SCRIPT=Path(__file__).resolve()
SCIENCE='experiments.ema_causal_20261110.run'
MODULE='experiments.ema_causal_scheduler_20261110'
BLOCKS=[d+'__'+o for d,o in CONDITIONS]
CAP=7<<30
GUARDIAN_CAP=4<<30


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argv(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
    except (OSError,UnicodeError):return []


def option(args,name):
    return args[args.index(name)+1] if name in args and args.index(name)+1<len(args) else None


def same_attempt(pid,args,out):
    value=option(args,'--out')
    if value is None:return False
    try:return (Path(f'/proc/{pid}/cwd').resolve()/value).resolve()==out
    except OSError:return False


def process_kind(args):
    if '-m' not in args:return None
    i=args.index('-m')
    return tuple(args[i+1:i+3])


def scan(out):
    workers={};finalizers=[];controllers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        pid=int(p.name);args=argv(pid);identity=process_identity(pid)
        if not identity or not same_attempt(pid,args,out):continue
        kind=process_kind(args)
        item=dict(identity=identity,argv=args)
        if kind==(SCIENCE,'_worker'):
            key=option(args,'--key')
            if key in workers:raise RuntimeError('Duplicate live workers: '+str(key))
            workers[key]=item
        elif kind==(MODULE,'_finalize'):finalizers.append(dict(item,block=option(args,'--block')))
        elif kind in ((SCIENCE,'run'),(SCIENCE,'resume')):controllers.append(item)
    if len(finalizers)>1:raise RuntimeError('Multiple attribution children')
    return workers,finalizers,controllers


def rss(identity):
    if not alive(identity):return 0
    try:return int(next(s.split()[1] for s in Path(f'/proc/{identity["pid"]}/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
    except (OSError,StopIteration):return 0


def bounds(job):
    p=job['config']['parameter_count']
    # Six retained packets: online/F/R/four old-R vectors, buffers and predictions.
    # All three panels contain 294,912 FP32 prediction bytes; packet <= 7 copies + buffers.
    packet=(16+8*len(WINDOWS[job['dataset']]))*p+(4<<20)
    retained=(len(ANCHORS)+1)*packet+(12+32+16)*p
    active=peak_active_bytes(job['dataset'],p)
    # Full future row: retained anchors, three selected states, two source and matched packets.
    return dict(packet=packet,retained=retained,active=active,scratch=100*p+(32<<20))


@lru_cache(maxsize=4)
def phases(dataset,objective,p):
    S=schedule(dataset)['epoch_end_steps'][0];users={};prediction=294912
    for t in schedule(dataset)['epoch_end_steps']:
        for h in set(min(h,t) for h in WINDOWS[dataset]):users.setdefault(t-h,set()).add(t)
    def packet(e):return (16+8*len(set(min(h,e*S) for h in WINDOWS[dataset])))*p+(4<<20)
    def snapshot(e,events,pulse):
        source=16*sum(e>=v for v in (20,35)) if objective=='clean' else 0
        matched=8*sum(e>=v for v in (35,50))
        selected=16+8*len(set(min(h,max(1,e)*S) for h in WINDOWS[dataset]))
        return (24+12+selected+source+matched+pulse+8*events)*p+(events+10)*prediction+(1<<20)
    result=[]
    for epoch in range(1,51):
        start=(epoch-1)*S;end=epoch*S
        old_events=sum(o<=start<max(v) for o,v in users.items())
        events=sum(o<=end<max(v) for o,v in users.items())
        peak_events=sum(o<=end and start<max(v) for o,v in users.items())
        old=snapshot(epoch-1,old_events,8*int(epoch%5==0)) if epoch>1 else 0
        new=snapshot(epoch,events,8*int((epoch+1)%5==0 and epoch<50))
        retained=sum(packet(e) for e in ANCHORS if e<epoch)
        variants=[dict(stage='commit',future_retained=retained,worker_state=new+32*p,
            ipc_and_old_commit=2*new+old,pending_anchor=(2 if epoch in ANCHORS else 1)*packet(epoch)),
            dict(stage='training',future_retained=retained,
            worker_state=max(peak_active_bytes(dataset,p),snapshot(epoch,peak_events,16)+32*p),
            ipc_and_old_commit=old,pending_anchor=packet(epoch))]
        for variant in variants:
            stage=variant.pop('stage')
            result.append(dict(epoch=epoch,stage=stage,history_events=events,terms=variant,
                total=sum(variant.values()),serialized_host_bytes=2*(new+packet(epoch)),
                guardian_resident_and_worker=2*old+retained,
                guardian_incoming=new+(packet(epoch) if epoch in ANCHORS else 0)))
    return result


def guardian_lower(job,epoch):
    if not epoch:return 0
    p=job['config']['parameter_count']
    # Only guaranteed parameter tensors are subtracted; all arrays/buffers remain double-counted.
    source=16*sum(epoch>=e for e in (20,35)) if job['objective']=='clean' else 0
    matched=8*sum(epoch>=e for e in (35,50))
    return (24+12+24+source+matched+sum(16+8*len(set(min(h,e*job['config']['steps_per_epoch']) for h in WINDOWS[job['dataset']])) for e in ANCHORS if e<=epoch))*p


def forecast(status,jobs,active,proposed=None,finalizer_block=None):
    bykey={j['job_id']:j for j in jobs};keys=set(active)
    if proposed:keys.add(proposed)
    terms=dict(guardian_actual=status['payload_bytes'],replaced_guardian_parameter_lower_bound=0,future_retained=0,worker_state=0,
               ipc_and_old_commit=0,pending_anchor=0,finalizer_row=0,finalizer_scratch=0)
    peaks={};guardian_stages={}
    for key in keys:
        job=bykey[key];epoch=status['jobs'].get(key,{}).get('epoch',0)
        remaining=[v for v in phases(job['dataset'],job['objective'],job['config']['parameter_count']) if v['epoch']>=min(epoch+1,50)]
        peak=max(remaining,key=lambda r:r['total'])
        guardian_stages[key]=[v for v in remaining if v['stage']=='commit']
        peaks[key]=peak
        for name,value in peak['terms'].items():terms[name]+=value
        # A discovered worker may still be exiting after COMPLETE already stripped its state.
        if status['jobs'].get(key,{}).get('status') not in ('COMPLETE','FINALIZED'):
            terms['replaced_guardian_parameter_lower_bound']-=guardian_lower(job,epoch)
    if finalizer_block:
        b=max((bounds(j) for j in jobs if block_of(j)==finalizer_block),key=lambda v:v['retained'])
        # Generator advancement can briefly overlap previous and incoming fetched row.
        terms['finalizer_row']=2*b['retained'];terms['finalizer_scratch']=b['scratch']
    base=status['payload_bytes']+terms['replaced_guardian_parameter_lower_bound']
    residents={k:max(v['guardian_resident_and_worker'] for v in stages) for k,stages in guardian_stages.items()}
    commits={k:base+sum(v for key,v in residents.items() if key!=k)+max(v['guardian_resident_and_worker']+v['guardian_incoming'] for v in stages)
             for k,stages in guardian_stages.items()}
    return dict(terms=terms,total=sum(terms.values()),cap=CAP,stage_peaks=peaks,
                guardian_predicate=dict(total=max(commits.values(),default=status['payload_bytes']),cap=GUARDIAN_CAP,
                    base=base,other_worker_resident_bounds=residents,commit_totals=commits),
                existing_peak_active_bytes={key:bounds(bykey[key])['active'] for key in keys})


def block_of(job):return job['dataset']+'__'+job['objective']


def candidate(jobs,status,active):
    pending=[b for b in BLOCKS if b not in status['blocks']]
    for block in pending[:2]:
        for j in jobs:
            key=j['job_id'];row=status['jobs'].get(key,{})
            if block_of(j)==block and key not in active and row.get('status') not in ('COMPLETE','FINALIZED'):
                if row.get('launches',0)>=2:raise RuntimeError('Restart budget exhausted: '+key)
                return key
    return None


def complete(out,block,jobs,status,finalized=False):
    path=out/'blocks'/block/'BLOCK_COMPLETE.json'
    if not path.exists():return False
    r=read_json(path);selected=[j for j in jobs if block_of(j)==block]
    expected='FINALIZED' if finalized else 'COMPLETE'
    return (r.get('status')=='COMPLETE' and r.get('block')==block and r.get('seeds')==SEEDS
            and set(r.get('job_ids',[]))=={j['job_id'] for j in selected}
            and r.get('logical_epochs')==250
            and r.get('logical_adam_steps')==250*selected[0]['config']['steps_per_epoch']
            and all(status['jobs'].get(j['job_id'],{}).get('status')==expected
                    and status['jobs'][j['job_id']]['epoch']==50 for j in selected))


def admit(eligible,memory,extra,amount,finalizer_live=False,launch_finalizer=False):
    if launch_finalizer and finalizer_live:return None,'attribution already active'
    if amount['total']>CAP:return None,'physical tensor forecast exceeds 7 GiB'
    if amount['guardian_predicate']['total']>GUARDIAN_CAP:return None,'guardian commit predicate exceeds 4 GiB'
    if memory['used']+extra>=.9*memory['total']:return None,'host RSS reservation exceeds 90%'
    return (eligible[0],'admitted') if eligible else (None,'no GPU below memory/power 85%')


def snapshot(out,jobs,devices,old=None):
    status=request(out,'status');workers,finalizers,controllers=scan(out)
    known={j['job_id'] for j in jobs}
    if set(workers)-known:raise RuntimeError('Unknown attempt worker')
    active=set(workers)
    for key,row in status['jobs'].items():
        if alive(row.get('worker')):
            if key not in workers or workers[key]['identity']!=row['worker']:raise RuntimeError('Worker identity cannot be reconciled: '+key)
            active.add(key)
    pending=[b for b in BLOCKS if b not in status['blocks']]
    first=pending[0] if pending else None
    ready=first and all(status['jobs'].get(j['job_id'],{}).get('status')=='COMPLETE' for j in jobs if block_of(j)==first)
    action='finalize' if ready and not finalizers else 'worker'
    key=None if action=='finalize' else candidate(jobs,status,active)
    block=finalizers[0]['block'] if finalizers else (first if action=='finalize' else None)
    if finalizers and finalizers[0]['block']!=first and finalizers[0]['block'] not in status['blocks']:raise RuntimeError('Attribution order mismatch')
    # Finalization is separately admitted: it cannot overlap workers without its own reservation.
    amount=forecast(status,jobs,active,key,block)
    raw={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    memory=dict(total=raw['MemTotal'],used=raw['MemTotal']-raw['MemAvailable'])
    observed=max([rss(v['identity']) for v in workers.values()]+[rss(v['identity']) for v in finalizers]+[rss(old),0])
    # Calibration has no RSS field: reserve observed process RSS plus full payload growth and 2 GiB Python/data overhead.
    process_reserve=max(observed,2<<30)+amount['total']-status['payload_bytes']
    serialized=sum(v['serialized_host_bytes'] for v in amount['stage_peaks'].values())+amount['terms']['finalizer_row']
    extra=process_reserve*int(bool(key) or action=='finalize')+serialized
    eligible=resource_admission(devices)
    device,reason=admit(eligible,memory,extra,amount,bool(finalizers),action=='finalize')
    return dict(status=status,workers=workers,finalizers=finalizers,controllers=controllers,active=sorted(active),
                pending_registration=sorted(k for k in workers if not alive(status['jobs'].get(k,{}).get('worker'))),
                action=action,key=key,block=first,device=device,reason=reason,eligible=eligible,
                payload=amount,host=memory,serialized_host_reservation=serialized,extra_host_rss_reservation=extra,observed_process_rss=observed)


def spawn(out,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        child=subprocess.Popen([sys.executable,'-m',*args,'--out',str(out)],cwd=ROOT,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    return child


def run(args):
    out=Path(args.out).resolve();prepared=read_json(out/'PREPARED.json');jobs=prepared['jobs']
    identity=implementation_identity()['sha256']
    if identity!=prepared['identity']['implementation_hash']:raise RuntimeError('Scientific hash changed')
    for ds in ('MOVIELENS','KUAIRAND'):
        r=read_json(out/f'SMOKE_{ds}.json')
        if r['status']!='PASS' or r['implementation_hash']!=identity:raise RuntimeError('Scientific smoke identity mismatch')
    guardian=read_json(out/'guardian.json')['owner']
    if not alive(guardian):raise RuntimeError('Existing guardian required')
    old=process_identity(args.controller_pid) if args.controller_pid else None
    if old and args.controller_start and old['start']!=args.controller_start:raise RuntimeError('Controller creation identity mismatch')
    provenance=dict(scientific_hash=identity,prepared_sha256=sha(out/'PREPARED.json'),supervisor_sha256=sha(SCRIPT),
        amendment_sha256=sha(ROOT/'research_plans/26-11-10_ema_causal_attribution/EXECUTION_SCHEDULER_AMENDMENT.md'))
    if args.mode=='dry':
        snap=snapshot(out,jobs,args.devices,old)
        print(json.dumps(dict(provenance=provenance,old_controller=old,guardian=guardian,
            boundary_ready=complete(out,BLOCKS[0],jobs,snap['status'],True) and BLOCKS[0] in snap['status']['blocks'],snapshot=snap),indent=2));return
    review=read_json(out/'SCHEDULER_REVIEW.json')
    if review.get('verdict')!='PASS' or review.get('supervisor_sha256')!=sha(SCRIPT):raise RuntimeError('Scheduler reviewer PASS bound to current hash required')
    with (out/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=out/'SCHEDULER_TAKEOVER.json'
        if not receipt.exists():
            if not old or not args.controller_start:raise RuntimeError('Bind controller PID and start identity from dry output')
            old_args=argv(old['pid'])
            if process_kind(old_args) not in ((SCIENCE,'run'),(SCIENCE,'resume')) or not same_attempt(old['pid'],old_args,out):raise RuntimeError('Controller command mismatch')
            while True:
                if not alive(guardian):raise RuntimeError('Guardian identity lost')
                status=request(out,'status')
                if BLOCKS[0] in status['blocks'] and complete(out,BLOCKS[0],jobs,status,True):break
                print('Waiting for ML-C complete receipt and guardian FINALIZED rows',flush=True);time.sleep(3)
            if alive(old):
                if argv(old['pid'])!=old_args:raise RuntimeError('Controller argv changed')
                os.kill(old['pid'],signal.SIGTERM)
            while alive(old):time.sleep(.1)
            snap=snapshot(out,jobs,args.devices,old)
            if snap['controllers']:raise RuntimeError('Another controller remains; no dispatch')
            write_json(receipt,dict(provenance,old_controller=old,old_argv=old_args,new_controller=process_identity(os.getpid()),
                guardian=guardian,boundary_sha256=sha(out/'blocks'/BLOCKS[0]/'BLOCK_COMPLETE.json'),snapshot=snap,time=time.time()))
        else:
            prior=read_json(receipt)
            if prior['guardian']!=guardian or prior['scientific_hash']!=identity:raise RuntimeError('Takeover guardian/science identity changed')
        children=[]
        launch_file=out/'SCHEDULER_LAUNCHES.json'
        launches=read_json(launch_file) if launch_file.exists() else {}
        while True:
            for child in children[:]:
                if child.poll() is not None:children.remove(child)
            if not alive(guardian):raise RuntimeError('Guardian identity lost')
            snap=snapshot(out,jobs,args.devices,old)
            if snap['controllers']:raise RuntimeError('Competing original controller; no dispatch')
            status=snap['status']
            old_launches=launches.copy()
            for key,worker in snap['workers'].items():
                row=status['jobs'].get(key,{})
                if row.get('status') not in ('COMPLETE','FINALIZED'):
                    launches[key]=max(launches.get(key,0),row.get('launches',0)+(not alive(row.get('worker'))))
            if launches!=old_launches:write_json(launch_file,launches)
            if set(status['blocks'])==set(BLOCKS):
                if len(status['jobs'])!=20 or any(r['status']!='FINALIZED' or r['epoch']!=50 for r in status['jobs'].values()):raise RuntimeError('Final job totals mismatch')
                if not all(complete(out,b,jobs,status,True) for b in BLOCKS):raise RuntimeError('Final receipt totals mismatch')
                write_json(out/'TRAINING_COMPLETE.json',dict(status='COMPLETE',jobs=20,epochs=1000,real_adam_steps=sum(50*j['config']['steps_per_epoch'] for j in jobs),time=time.time()));return
            block=snap['block'];keys=[j['job_id'] for j in jobs if block_of(j)==block]
            if not snap['finalizers'] and complete(out,block,jobs,status):
                request(out,'finalize',block=block,keys=keys);continue
            meta=out/'SCHEDULER_FINALIZER.json'
            if not snap['finalizers'] and meta.exists():
                last=read_json(meta)
                if last['block'] not in status['blocks']:
                    raise RuntimeError('Attribution child exited before complete receipt; anchors retained; inspect '+str(out/'scheduler_finalizer.log'))
            decision={k:v for k,v in snap.items() if k not in ('status','workers','controllers')}
            disk_admit(out,16384)
            append_json(out/'SCHEDULER_DECISIONS.jsonl',dict(decision,time=time.time()))
            if snap['device'] is not None:
                if snap['action']=='finalize':
                    child=spawn(out,[MODULE,'_finalize','--block',block,'--device',f'cuda:{snap["device"]}'],out/'scheduler_finalizer.log')
                    write_json(meta,dict(block=block,identity=process_identity(child.pid),supervisor_sha256=sha(SCRIPT)))
                    children.append(child)
                elif snap['key']:
                    key=snap['key'];count=max(launches.get(key,0),status['jobs'].get(key,{}).get('launches',0))
                    if count>=2:raise RuntimeError('Worker launch budget exhausted, including pre-registration failures: '+key)
                    launches[key]=count+1;write_json(launch_file,launches)
                    children.append(spawn(out,[SCIENCE,'_worker','--key',key,'--device',f'cuda:{snap["device"]}'],out/'runs'/key/'worker.log'))
            time.sleep(3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['dry','run','_finalize']);parser.add_argument('--out',required=True)
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1]);parser.add_argument('--controller-pid',type=int,default=0)
    parser.add_argument('--controller-start');parser.add_argument('--block',choices=BLOCKS);parser.add_argument('--device')
    args=parser.parse_args()
    if args.mode=='_finalize':
        from experiments.ema_causal_20261110.block import finalize_block
        out=Path(args.out).resolve();jobs=[j for j in read_json(out/'PREPARED.json')['jobs'] if block_of(j)==args.block]
        with (out/'scheduler_attribution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finalize_block(out,jobs,args.device)
            if not complete(out,args.block,jobs,request(out,'status')):raise RuntimeError('Attribution receipt mismatch')
            request(out,'finalize',block=args.block,keys=[j['job_id'] for j in jobs])
    else:run(args)


if __name__=='__main__':main()
