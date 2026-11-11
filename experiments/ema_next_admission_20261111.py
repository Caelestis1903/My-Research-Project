"""Admission-only adapter for the active EMA study; scientific package stays pinned."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.ema_next_20261111 import scheduler
from experiments.ema_next_20261111.protocol import implementation_identity
from experiments.ema_next_20261111.state import request,process_identity,alive
from experiments.overnight_ab_20261103.state import read_json,write_json,sha_file,digest

ORIGINAL_ADMIT=scheduler.admit
ORIGINAL_GPU_STATE=scheduler.gpu_state
MODULE='experiments.ema_next_20261111.run'
BASELINE_PATH=Path(__file__).resolve().parents[1]/'research_reports/2026-11-11_ema_next/scheduler_revisions/KR_PARENT_BASELINE.json'


def gate_write(out,dataset,name,value):
    path=Path(out)/'decisions'/dataset/(name+'.json')
    canonical=json.loads(json.dumps(value,allow_nan=False))
    if path.exists():
        if read_json(path)!=canonical:raise ValueError('Frozen decision changed '+dataset+'/'+name)
        return
    write_json(path,value)


def option(args,name):
    return args[args.index(name)+1] if name in args and args.index(name)+1<len(args) else None


def worker_inventory(out):
    out=Path(out).resolve();workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().decode().split('\0')[:-1]
            if '-m' not in args:continue
            i=args.index('-m')
            if args[i+1:i+3]!=[MODULE,'_worker']:continue
            value=option(args,'--out')
            if value is None or (p/'cwd').resolve().joinpath(value).resolve()!=out:continue
            identity=process_identity(int(p.name))
            if not identity:continue
            key=option(args,'--key');device=option(args,'--device')
            if not key or not device or not device.startswith('cuda:'):raise ValueError('Attempt worker has no job/device identity')
            job=read_json(out/'runs'/key/'JOB.json')
            rss=int(next(s.split()[1] for s in (p/'status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
            epochs=sorted((out/'runs'/key/'epochs').glob('E*.json'));peak=0
            if epochs:peak=read_json(epochs[-1]).get('cost',{}).get('gpu_peak_allocated',0)
            workers.append(dict(identity=identity,key=key,device=int(device.split(':')[1]),dataset=job['dataset'],family=job['family'],objective=job['objective'],job=job,rss_bytes=rss,allocated_peak_bytes=peak))
        except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):continue
    return workers


def gpu_inventory(devices):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
    gpus=[];uuids={}
    for line in raw.splitlines():
        v=[x.strip() for x in line.split(',')];index=int(v[0]);uuids[v[1]]=index
        if index in devices:gpus.append(dict(index=index,total=float(v[2])*2**20,used=float(v[3])*2**20,power=float(v[4]),power_limit=float(v[5])))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes={}
    for line in raw.splitlines():
        v=[x.strip() for x in line.split(',')]
        if len(v)!=3:continue
        try:processes[(int(v[0]),uuids[v[1]])]=float(v[2])*2**20
        except (KeyError,ValueError):continue
    return gpus,processes


def reservations(prepared,workers,gpus,process_memory,highwater):
    highwater=copy.deepcopy(highwater);highwater.setdefault('process_bytes',{});highwater.setdefault('device_process_bytes',{});highwater.setdefault('nonallocated_overhead_bytes',0)
    for w in workers:
        used=process_memory.get((w['identity']['pid'],w['device']))
        if used is None:continue
        ds=w['dataset'];dk=f'{w["device"]}:{ds}'
        highwater['process_bytes'][ds]=max(highwater['process_bytes'].get(ds,0),used)
        highwater['device_process_bytes'][dk]=max(highwater['device_process_bytes'].get(dk,0),used)
        if w['allocated_peak_bytes']:
            highwater['nonallocated_overhead_bytes']=max(highwater['nonallocated_overhead_bytes'],used-w['allocated_peak_bytes'])
    reserve={ds:max(c['gpu_peak_bytes']+highwater['nonallocated_overhead_bytes'],highwater['process_bytes'].get(ds,0)) for ds,c in prepared['calibration'].items()}
    adjusted=[dict(g,pending_worker_bytes=0.) for g in gpus];by_device={g['index']:g for g in adjusted};entries=[]
    for w in workers:
        used=process_memory.get((w['identity']['pid'],w['device']))
        deficit=max(reserve[w['dataset']]-(used or 0),0)
        if w['device'] in by_device:by_device[w['device']]['pending_worker_bytes']+=deficit
        entries.append(dict(w,observed_gpu_bytes=used,reserved_gpu_bytes=reserve[w['dataset']],unallocated_reservation_bytes=deficit))
    for g in adjusted:g['used_observed']=g['used'];g['used']+=g['pending_worker_bytes']
    return adjusted,reserve,highwater,entries


def verified_parent_baseline(out,prepared,status,job,workers):
    if job.get('dataset')!='kuairand' or job.get('family')!='PARENT' or job.get('objective')!='clean':return None
    if not BASELINE_PATH.exists():return None
    proof=read_json(BASELINE_PATH);out=Path(out).resolve()
    if str(out)!=proof['attempt'] or prepared['identity']['implementation_hash']!=proof['scientific_hash']:return None
    if status.get('pid')!=proof['guardian_owner']['pid'] or process_identity(status['pid'])!=proof['guardian_owner']:return None
    complete=out/'decisions'/proof['previous_dataset']/'DATASET_COMPLETE.json'
    if not complete.exists() or sha_file(complete)!=proof['previous_complete_sha256']:return None
    if any(w['dataset']!='kuairand' or w.get('family')!='PARENT' or w.get('objective')!='clean' for w in workers):return None
    worker_keys={w['key'] for w in workers}
    for key,row in status.get('jobs',{}).items():
        j=row['job']
        if j['dataset']=='kuairand':
            if j['family']!='PARENT' or j['objective']!='clean' or row['status'] not in ('PENDING','RUNNING','COMPLETE'):return None
            if row['status']!='COMPLETE' and key not in worker_keys and key!=job['job_id']:return None
        elif j['dataset']!=proof['previous_dataset'] or row['status'] not in ('COMPLETE','RELEASED'):return None
    with (out/'RESOURCE_ADMISSION.jsonl').open() as stream:
        first=next((r for line in stream if (r:=json.loads(line))['job_id'].startswith('P.C.kuairand.')),None)
    if first is None or digest(first)!=proof['first_admission_sha256']:return None
    if first['job_id']!=proof['first_job_id'] or first['time']!=proof['first_admission_time']:return None
    if first['reason']!='ADMITTED' or first['worker_reservations'] or first['discovered_worker_count']!=0:return None
    if first['parent_cohort']['retained_payload_bytes']!=proof['payload_bytes']:return None
    return dict(proof,receipt_sha256=sha_file(BASELINE_PATH))


def parent_cohort_forecast(prepared,status,job,workers,active_count,baseline=None):
    ds=job['dataset']
    if job.get('family')!='PARENT' or active_count>len(workers):return None
    if any(w.get('family')!='PARENT' or w['dataset']!=ds for w in workers):return None
    layout=prepared['resource_estimate']['datasets'][ds]['layout_elements']
    P=job['config']['parameter_count'];S=prepared['manifest']['datasets'][ds]['shared_parameter_count']
    C=4*sum(sum(layout[k].values()) for k in ('best','prefix','anchors'))
    L=4*layout['active'];metadata=(1<<20)+200*prepared['groups'][ds]['runtime_buffer_bytes']
    I=L+4*(12*P+3*S)+metadata;B=status['payload_bytes'];n=len(workers)+1
    done=0
    if baseline is not None:
        B=baseline['payload_bytes'];worker_keys={w['key'] for w in workers}
        done=sum(r['job']['dataset']==ds and r['status']=='COMPLETE' and key not in worker_keys for key,r in status['jobs'].items())
    guardian=B+done*(C+metadata)+n*(C+2*L+metadata)+I
    physical=guardian+n*(C+L+5*4*P+metadata)+I
    return dict(dataset=ds,cohort_count=n,completed_parent_count=done,retained_payload_bytes=B,actual_payload_bytes=status['payload_bytes'],baseline=baseline,immutable_closure_bytes=C,live_bytes=L,
        metadata_bytes=metadata,incoming_bytes=I,guardian_predicate=guardian,physical_tensor_bytes=physical,
        guardian_formula='B+d*(C+metadata)+n*(C+2L+metadata)+I',physical_formula='G+n*(C+L+5*FP32P+metadata)+I')


def a_cohort_forecast(prepared,status,job,workers,active_count):
    ds=job['dataset']
    if job.get('family') not in ('A_MATCH','A_RAW'):return None
    if active_count>len(workers):return None
    resource=prepared['resource_estimate']['datasets'][ds]
    if resource['single_window_history_peak']>6:return None
    P=prepared['manifest']['datasets'][ds]['parameter_count']
    for current in [job]+[w.get('job',{}) for w in workers]:
        if current.get('dataset')!=ds or current.get('family') not in ('A_MATCH','A_RAW') or current.get('objective')!='clean':return None
        if current.get('readout_menu') is not None or current.get('config',{}).get('parameter_count')!=P:return None
        if any(current['config'].get(k)!=prepared['manifest']['datasets'][ds][k] for k in ('steps_per_epoch','rho')):return None
        start=current.get('start_epoch');end=current.get('end_epoch')
        if current.get('continuation'):
            if (start,end)!=(26,50):return None
        elif (start,end) not in ((20,26),(35,41)) or current.get('anchor',start)!=start:return None
        b=current.get('frozen_B')
        if b is not None and b not in prepared['readouts'][ds]:return None
        if b is not None and b.get('family')!='B_SPLIT':return None
    layout=resource['layout_elements'];D=4*(sum(sum(layout[k].values()) for k in ('best','prefix','anchors'))+layout['active'])
    E=20*4*P+(32<<20);B=status['payload_bytes'];n=len(workers)+1
    return dict(dataset=ds,cohort_count=n,actual_payload_bytes=B,per_worker_extra_bytes=E,parent_fetch_closure_bytes=D,
        guardian_predicate=B+(2*n+1)*E,physical_tensor_bytes=B+(4*n+1)*E+n*D,
        guardian_formula='B+(2n+1)E',physical_formula='B+(4n+1)E+nD')


class Admission:
    def __init__(self,out,persist=True):
        self.out=Path(out).resolve();self.path=self.out/'SCHEDULER_ADMISSION_HIGHWATER.json';self.persist=persist
        self.highwater=read_json(self.path) if self.path.exists() else {}

    def __call__(self,prepared,status,job,devices,active_count):
        workers=worker_inventory(self.out);gpus,processes=gpu_inventory(devices)
        adjusted,reserve,updated,entries=reservations(prepared,workers,gpus,processes,self.highwater)
        if self.persist and updated!=self.highwater:write_json(self.path,updated)
        self.highwater=updated
        details=dict(worker_reservations=entries,gpu_adjusted=adjusted,candidate_gpu_reserve=reserve[job['dataset']],adapter_sha256=sha_file(__file__))
        if any(w['key']==job['job_id'] for w in workers):return None,dict(details,job_id=job['job_id'],reason='same_job_pending_or_running')
        if not updated.get('nonallocated_overhead_bytes'):
            return None,dict(details,job_id=job['job_id'],reason='waiting_for_measured_CUDA_overhead')
        # Count process identities before guardian registration and before CUDA allocation.
        count=max(active_count,len(workers));P=job['config']['parameter_count'];host_each=(2<<30)+4*((20*P*4)+(32<<20))
        pending_host=sum(max(host_each-w['rss_bytes'],0) for w in workers)
        mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemTotal']-mem['MemAvailable']+pending_host+host_each>=.9*mem['MemTotal']:
            return None,dict(details,job_id=job['job_id'],reason='host_90pct_with_pending_processes',pending_host_bytes=pending_host)
        resolved=dict(prepared,calibration={d:dict(c,gpu_peak_bytes=reserve[d]) for d,c in prepared['calibration'].items()})
        baseline=verified_parent_baseline(self.out,prepared,status,job,workers)
        cohort=parent_cohort_forecast(prepared,status,job,workers,active_count,baseline)
        a_cohort=a_cohort_forecast(prepared,status,job,workers,active_count)
        if a_cohort is not None:cohort=a_cohort
        if cohort is not None:
            ds=job['dataset'];estimate=prepared['resource_estimate'];datasets=dict(estimate['datasets'])
            datasets[ds]=dict(datasets[ds],peak=cohort)
            resolved['resource_estimate']=dict(estimate,datasets=datasets)
            count=0  # Cohort already includes every discovered process and the proposed worker.
            details['a_cohort' if a_cohort is not None else 'parent_cohort']=cohort
        scheduler.gpu_state=lambda _:adjusted
        try:device,receipt=ORIGINAL_ADMIT(resolved,status,job,devices,count)
        finally:scheduler.gpu_state=ORIGINAL_GPU_STATE
        receipt.update(details,discovered_worker_count=len(workers),pending_host_bytes=pending_host)
        return device,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--devices',type=int,nargs='+',default=[0,1]);parser.add_argument('--inspect',action='store_true');parser.add_argument('--previous-controller',type=int)
    args=parser.parse_args();out=args.out.resolve();prepared=read_json(out/'PREPARED.json')
    expected=prepared['identity']['implementation_hash']
    if implementation_identity()['sha256']!=expected:raise ValueError('Scientific package differs from frozen implementation')
    for ds in prepared['manifest']['dataset_order']:prepared.setdefault('calibration',{})[ds]=read_json(out/f'SMOKE_{ds.upper()}.json')
    admission=Admission(out,persist=not args.inspect);workers=worker_inventory(out)
    if args.inspect:
        gpus,processes=gpu_inventory(args.devices);adjusted,reserve,highwater,entries=reservations(prepared,workers,gpus,processes,admission.highwater)
        print(json.dumps(dict(scientific_hash=expected,adapter_sha256=sha_file(__file__),workers=entries,gpus=adjusted,reserve=reserve,highwater=highwater),indent=2));return
    review=read_json(out/'SCHEDULER_ADMISSION_REVIEW.json')
    if review.get('verdict')!='PASS' or review.get('adapter_sha256')!=sha_file(__file__) or review.get('scientific_hash')!=expected:raise ValueError('Hash-bound admission review PASS required')
    receipt=dict(kind='SCHEDULER_ONLY_REVISION',scientific_hash=expected,adapter_path=str(Path(__file__).resolve()),adapter_sha256=sha_file(__file__),
        previous_controller=(process_identity(args.previous_controller) or dict(pid=args.previous_controller)) if args.previous_controller else read_json(out/'SCHEDULER_PREVIOUS_CONTROLLER.json'),
        new_controller=process_identity(os.getpid()),adopted_workers=workers,time=time.time(),
        cause='Pending CUDA reservations, audited parent baseline, bounded Clean A cohort forecasts, and JSON-equivalent decision reentry; scientific package unchanged.',
        baseline_receipt_sha256=sha_file(BASELINE_PATH),
        initial_observation=dict(worker_process_MiB=1346,worker_allocated_peak_MiB=934.89404296875,ML_smoke_allocated_peak_bytes=1048753664))
    gpus,processes=gpu_inventory(args.devices)
    _,_,admission.highwater,_=reservations(prepared,workers,gpus,processes,admission.highwater)
    write_json(admission.path,admission.highwater)
    if (out/'SCHEDULER_REVISION.json').exists():
        previous=read_json(out/'SCHEDULER_REVISION.json');write_json(out/f'SCHEDULER_REVISION_{previous["new_controller"]["pid"]}.json',previous)
    write_json(out/'SCHEDULER_REVISION.json',receipt)
    scheduler.admit=admission
    scheduler.gate_write=gate_write
    return scheduler.execute(out,args.devices)

if __name__=='__main__':main()
