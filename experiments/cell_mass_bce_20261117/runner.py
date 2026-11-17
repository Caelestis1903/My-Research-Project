"""Foreground-supervised Cycle05 mechanics; no AI lifecycle or disk tensor states."""
import argparse
import copy
import ctypes
import gzip
import io
import json
import math
import os
import resource
import subprocess
import threading
import time
import traceback
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from . import protocol as p
from .core import train_step
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step

MODULE = 'experiments.cell_mass_bce_20261117.runner'
CASES = tuple((dev,d,p.ARMS[0]) for dev in (0,1) for d in p.DATASETS)


def host_fraction():
    m={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
       if line.startswith(('MemTotal:','MemAvailable:'))}
    return (m['MemTotal']-m['MemAvailable'])/m['MemTotal']


class _Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class _Process(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
              ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]


QUERY_NAMES = ('nvmlInit_v2','nvmlDeviceGetHandleByIndex_v2','nvmlDeviceGetMemoryInfo',
               'nvmlDeviceGetPowerUsage','nvmlDeviceGetEnforcedPowerLimit','nvmlDeviceGetComputeRunningProcesses_v2')


class TelemetryLog:
    def __init__(self,path,identity):
        self.path=p.new_path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.records=0; self.closed=False
        self.stream=gzip.open(self.path,'xt',compresslevel=3)
        self.emit(dict(event='DICTIONARY',query_names=list(QUERY_NAMES),identity=identity))

    def emit(self,row):
        with self.lock:
            p.json_safe(row); self.stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            self.stream.flush(); self.records+=1

    def close(self):
        with self.lock:
            if not self.closed: self.stream.close(); self.closed=True

    def reference(self):
        with self.lock:
            if not self.closed: self.stream.flush()
            return dict(path=str(self.path),sha256=p.sha_file(self.path),bytes=self.path.stat().st_size,
                        record_start=1,record_end=self.records)


class NVML:
    """Three complete fresh attempts, named v2 query facts, no cached sample."""
    def __init__(self,device,*,emit,identity,phase=lambda:'controller',lib=None,host=host_fraction,sleep=time.sleep):
        self.device=int(device); self.emit=emit; self.identity=identity; self.phase=phase
        self.lib=lib if lib is not None else ctypes.CDLL('libnvidia-ml.so.1')
        self.host=host; self.sleep=sleep; self.incidents=0; self.last_error=None

    def _query(self,name,args,incident,attempt,*,capacity=None):
        code=int(getattr(self.lib,name)(*args))
        row=dict(event='QUERY',query_id=QUERY_NAMES.index(name),return_code=code,device=self.device,calling_pid=os.getpid(),
            phase=self.phase(),observed_pid=os.getpid(),incident_id=incident,sample_attempt=attempt,time=time.time(),monotonic=time.monotonic(),
            arguments=dict(device=self.device) if name=='nvmlDeviceGetHandleByIndex_v2' else {},
            requested_capacity=256 if capacity is not None else None,returned_capacity=capacity.value if capacity is not None else None)
        self.emit(row)
        if code!=0 or (capacity is not None and capacity.value>256):
            self.last_error=dict(row,query_name=name)
            raise p.TechnicalPause(f'{name} failed code={code} device={self.device} phase={row["phase"]} sample_attempt={attempt}')

    def sample(self):
        self.incidents+=1; incident=f'{os.getpid()}:{self.device}:{self.incidents}'
        for attempt in range(1,4):
            start=time.monotonic()
            try:
                handle=ctypes.c_void_p()
                self._query('nvmlInit_v2',(),incident,attempt)
                self._query('nvmlDeviceGetHandleByIndex_v2',(ctypes.c_uint(self.device),ctypes.byref(handle)),incident,attempt)
                mem=_Memory(); power=ctypes.c_uint(); limit=ctypes.c_uint()
                self._query('nvmlDeviceGetMemoryInfo',(handle,ctypes.byref(mem)),incident,attempt)
                self._query('nvmlDeviceGetPowerUsage',(handle,ctypes.byref(power)),incident,attempt)
                self._query('nvmlDeviceGetEnforcedPowerLimit',(handle,ctypes.byref(limit)),incident,attempt)
                capacity=ctypes.c_uint(256); processes=(_Process*256)()
                self._query('nvmlDeviceGetComputeRunningProcesses_v2',(handle,ctypes.byref(capacity),processes),incident,attempt,capacity=capacity)
                rows=[dict(pid=int(x.pid),used_bytes=int(x.usedGpuMemory)) for x in processes[:capacity.value]]
                host=self.host()
                p.require(mem.total>0 and 0<=mem.used<=mem.total and power.value>=0 and limit.value>0 and
                          math.isfinite(host) and 0<=host<=1 and all(0<=r['used_bytes']<2**63 for r in rows),'Unknown/invalid complete NVML sample')
                row=dict(event='SAMPLE',sample_id=incident+f':{attempt}',incident_id=incident,sample_attempt=attempt,
                    time=time.time(),monotonic_start=start,monotonic_end=time.monotonic(),device=self.device,phase=self.phase(),
                    calling_pid=os.getpid(),used_bytes=int(mem.used),nvml_total_bytes=int(mem.total),power_watts=power.value/1000,
                    power_limit_watts=limit.value/1000,host_fraction=host,processes=rows,
                    own_process_bytes=sum(r['used_bytes'] for r in rows if r['pid']==os.getpid()),known=True)
                self.emit(row); return row
            except Exception as error:
                self.emit(dict(event='SAMPLE_FAILED',incident_id=incident,sample_attempt=attempt,device=self.device,
                    phase=self.phase(),calling_pid=os.getpid(),time=time.time(),monotonic_end=time.monotonic(),
                    error_type=type(error).__name__,error=str(error),known=False))
                if attempt==3: raise
                self.sleep(.05)


class Profile:
    def __init__(self,device,j,number):
        self.device=int(device); self.job=j; self.number=number; self.phase='context_initialization'
        self.phase_summaries={}; self.peaks={}; self.lock=threading.RLock(); self.cuda_ready=False
        self.error=None; self.done=threading.Event(); self.closed=False; self.witness_refs=[]
        self.raw=TelemetryLog(p.ATTEMPT/'telemetry'/f'worker_{os.getpid()}_{number}.jsonl.gz',
            dict(pid=os.getpid(),job_key=j['key'],registration_attempt=number,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH))
        self.nvml=NVML(device,emit=self.raw.emit,identity=j,phase=lambda:self.phase)
        self.thread=threading.Thread(target=self._loop,daemon=False); self.thread.start()

    def initialize_cuda(self):
        self._sample(); torch.cuda.set_device(self.device); torch.cuda.init()
        with self.lock: self.cuda_ready=True; self._sample()

    def _sample(self,force=False):
        with self.lock:
            row=self.nvml.sample()
            row.update(torch_allocated_bytes=torch.cuda.memory_allocated(self.device) if self.cuda_ready else 0,
                torch_reserved_bytes=torch.cuda.memory_reserved(self.device) if self.cuda_ready else 0,
                host_rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
            row['non_torch_bytes']=max(0,row['own_process_bytes']-row['torch_reserved_bytes'])
            self.raw.emit(dict(event='TORCH_PHASE',sample_id=row['sample_id'],phase=self.phase,
                torch_allocated_bytes=row['torch_allocated_bytes'],torch_reserved_bytes=row['torch_reserved_bytes'],
                non_torch_bytes=row['non_torch_bytes'],host_rss_bytes=row['host_rss_bytes']))
            phase=self.phase_summaries.setdefault(row['phase'],dict(samples=0,first_time=row['time'],last_time=row['time']))
            phase['samples']+=1; phase['last_time']=row['time']
            for name in ('own_process_bytes','torch_allocated_bytes','torch_reserved_bytes','non_torch_bytes','host_rss_bytes'):
                phase[name]=max(phase.get(name,0),row[name]); self.peaks[name]=max(self.peaks.get(name,0),row[name])
            return row

    def _loop(self):
        while not self.done.is_set():
            try: self._sample()
            except Exception as error: self.error=error; return
            self.done.wait(.005)

    def mark(self,name):
        with self.lock:
            self.phase=name
            if self.cuda_ready: torch.cuda.synchronize(self.device)
            if self.error is not None: raise self.error
            row=self._sample()
            if self.cuda_ready: p.require(row['used_bytes']<=torch.cuda.get_device_properties(self.device).total_memory,'Physical capacity exceeded at '+name)
            return row

    def close(self):
        self.done.set(); self.thread.join(); self.raw.close(); self.closed=True
        if self.error is not None: raise self.error

    def receipt(self):
        props=torch.cuda.get_device_properties(self.device)
        return dict(dataset=self.job['dataset'],arm=self.job['arm'],device=self.device,key=self.job['key'],producer_pid=os.getpid(),
            attempt=self.number,purpose=self.job['purpose'],source_hash=self.job['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
            environment=p.environment(),hardware=dict(name=props.name,compute_capability=[props.major,props.minor],cuda_usable_total_bytes=props.total_memory),
            full_process_peak_bytes=self.peaks['own_process_bytes'],torch_allocated_peak_bytes=self.peaks['torch_allocated_bytes'],
            torch_reserved_peak_bytes=self.peaks['torch_reserved_bytes'],non_torch_bytes=self.peaks['non_torch_bytes'],
            peak_host_rss_bytes=self.peaks['host_rss_bytes'],phases=sorted(self.phase_summaries),phase_summaries=self.phase_summaries,
            raw_log=self.raw.reference(),witness_refs=self.witness_refs,test_accessed=False)


def admission(current,incoming_measured_full_process_bytes=0,*,pending=False,already_resident_bytes=0):
    required=('used_bytes','nvml_total_bytes','power_watts','power_limit_watts','host_fraction','cuda_usable_total_bytes','monotonic_end')
    known=current.get('known') is True and all(isinstance(current.get(k),(int,float)) and math.isfinite(current[k]) for k in required)
    known=known and current['nvml_total_bytes']>0 and current['power_limit_watts']>0 and current['cuda_usable_total_bytes']>0 and 0<=current['used_bytes']<=current['nvml_total_bytes'] and current['power_watts']>=0 and 0<=current['host_fraction']<=1
    fresh=known and 0<=time.monotonic()-current['monotonic_end']<=1.
    memory=current['used_bytes']/current['nvml_total_bytes'] if known else None
    power=current['power_watts']/current['power_limit_watts'] if known else None
    remaining=max(0,incoming_measured_full_process_bytes-already_resident_bytes)
    predicates=dict(fresh_known=bool(fresh),current_memory_lt85=bool(known and memory<.85),current_power_lt85=bool(known and power<.85),
        current_host_lt90=bool(known and current['host_fraction']<.90),no_pending=not bool(pending),
        physical_fit=bool(known and current['used_bytes']+remaining<=current['cuda_usable_total_bytes']))
    return dict(eligible=all(predicates.values()),predicates=predicates,reasons=[k for k,v in predicates.items() if not v],
        sample_id=current.get('sample_id'),current_memory_fraction=memory,current_power_fraction=power,
        incoming_measured_full_process_bytes=incoming_measured_full_process_bytes,already_resident_bytes=already_resident_bytes,
        remaining_measured_bytes=remaining,test_accessed=False)


def admission_row(j,queue_index,device,decision,pending_identity,envelope):
    return dict(time=time.time(),purpose=j['purpose'],queue_index=queue_index,key=j['key'],device=device,
        pending_identity=pending_identity,envelope=envelope,**decision)


def pending_matches(pending,record,*,alive):
    return bool(alive and record.get('worker_pid')==pending['pid'] and record.get('attempts')==pending['expected_attempt'] and
        record.get('status')=='RUNNING' and record.get('allocated') is True and record.get('job',{}).get('key')==pending['key'])


def witness_payload(profile):
    return dict(producer_pid=profile['producer_pid'],job_key=profile['key'],attempt=profile['attempt'],purpose=profile['purpose'],
        dataset=profile['dataset'],arm=profile['arm'],source_hash=profile['source_hash'],design_bundle_sha256=profile['design_bundle_sha256'],
        environment=profile['environment'],hardware=profile['hardware'],device=profile['device'],
        full_process_peak_bytes=profile['full_process_peak_bytes'],reserved_peak_bytes=profile['torch_reserved_peak_bytes'],
        non_torch_peak_bytes=profile['non_torch_bytes'],allocated_peak_bytes=profile['torch_allocated_peak_bytes'],
        host_rss_bytes=profile['peak_host_rss_bytes'],phase_coverage=profile['phases'],raw_log=profile['raw_log'])


def registry_prefix(a):
    path=a/'profiles/witness_registry.jsonl'; data=path.read_bytes() if path.exists() else b''
    rows=[json.loads(line) for line in data.splitlines()]
    return rows,dict(path=str(path),bytes=len(data),records=len(rows),sha256=hashlib.sha256(data).hexdigest())


def collect_witnesses(a):
    rows,_=registry_prefix(a); known={r['witness_id'] for r in rows}
    for journal in sorted((a/'profiles').glob('producer_*.jsonl')):
        for line in journal.read_text().splitlines():
            row=json.loads(line); wid=row['witness_id']; payload={k:v for k,v in row.items() if k!='witness_id'}
            p.require(p.digest(payload)==wid,'Witness identity differs')
            raw=row['raw_log']; path=p.new_path(raw['path']); data=path.read_bytes()[:raw['bytes']]
            p.require(len(data)==raw['bytes'] and hashlib.sha256(data).hexdigest()==raw['sha256'],'Witness raw prefix differs')
            if wid not in known: p.append_json(a/'profiles/witness_registry.jsonl',row); known.add(wid)
    return registry_prefix(a)


def measured_envelope(profiles,dataset,device,hardware,source_hash,environment=None):
    compatible=[r for r in profiles if r['dataset']==dataset and r['arm']==p.ARMS[0] and r['source_hash']==source_hash and
                r['design_bundle_sha256']==p.DESIGN_HASH and r['hardware']==hardware and (environment is None or r['environment']==environment)]
    p.require(compatible and any(r['device']==device for r in compatible),'Missing compatible measured target-card envelope')
    peak=max(max(r['full_process_peak_bytes'],r['reserved_peak_bytes']+r['non_torch_peak_bytes']) for r in compatible)
    return peak,compatible


def envelope_version(a,dataset,device,hardware,manifest,*,engineering=False):
    rows,prefix=collect_witnesses(a)
    compatible=[r for r in rows if r['dataset']==dataset and r['source_hash']==manifest['source_hash'] and
        r['design_bundle_sha256']==p.DESIGN_HASH and r['environment']==manifest['environment'] and r['hardware']==hardware and r['arm']==p.ARMS[0]]
    if not engineering:
        signoff=p.read_json(a/'ENGINEERING_SIGNOFF.json')
        reviewed={r['witness_id'] for r in signoff['witnesses']}
        p.require(any(r['device']==device and r['witness_id'] in reviewed for r in compatible),'No reviewed C05 target-card witness')
    bootstrap=None
    if engineering:
        bootstrap=bootstrap_profiles(hardware)
    p.require(compatible or bootstrap,'No measured envelope')
    maxima={}
    for name in ('full_process_peak_bytes','reserved_peak_bytes','non_torch_peak_bytes'):
        value=max((r[name] for r in compatible),default=0)
        maxima[name]=dict(value=value,witness_ids=[r['witness_id'] for r in compatible if r[name]==value])
    peak=max((max(r['full_process_peak_bytes'],r['reserved_peak_bytes']+r['non_torch_peak_bytes']) for r in compatible),default=0)
    winners=[r['witness_id'] for r in compatible if max(r['full_process_peak_bytes'],r['reserved_peak_bytes']+r['non_torch_peak_bytes'])==peak]
    value=dict(scope=dict(dataset=dataset,arm=p.ARMS[0],source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        environment=manifest['environment'],hardware=hardware,target_device=device),registry_prefix=prefix,maxima=maxima,
        conservative_bytes=max(peak,bootstrap['incoming_bytes'] if bootstrap else 0),max_witness_ids=winners,
        historical_bootstrap=bootstrap)
    eid=p.digest(value); path=a/'profiles/envelopes'/f'{eid}.json'; p.immutable(path,dict(envelope_id=eid,**value))
    return value['conservative_bytes'],dict(envelope_id=eid,path=str(path),sha256=p.sha_file(path))


def bootstrap_profiles(hardware):
    p.require(hardware==dict(name='NVIDIA GeForce RTX 3090',compute_capability=[8,6],cuda_usable_total_bytes=25307578368),'Bootstrap hardware differs')
    signoff=p.A4/'attempt2/ENGINEERING_SIGNOFF.json'
    p.require(p.sha_file(signoff)=='ee7dc227bdcfc73798e5920da6eee441574378f25fe894cbf6d0c4c1b22868ac','Historical bootstrap signoff changed')
    expected=('1f196d597c2dbadf70eb6c748d46d8f80e5c1d1f4275fca337922627dc074450',
        '9d841b078b6b2ded6a9b5e022630a0e35040d3533722118c0347262e16929081',
        '793856cbdb3c6d3063165bf83b91bc7d28f0d2c987a305867cc687964a296f18',
        'a92798765283423c60099955420ceedfc79ac2db250b477b0df48dfab34ca895')
    profiles=p.read_json(signoff)['profiles']; p.require(len(profiles)==4 and tuple(r['sha256'] for r in profiles)==expected,'Bootstrap profile pointers differ')
    values=[p.receipt_json(r) for r in profiles]
    p.require(all(r['hardware']==hardware for r in values),'Bootstrap profile hardware differs')
    return dict(scope='historical_bootstrap',signoff=_receipt(signoff),profiles=profiles,
        incoming_bytes=max(max(r['full_process_peak_bytes'],r['torch_reserved_peak_bytes']+r['non_torch_bytes']) for r in values))


def restore(model,ema,opt,loader,state,device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    tick=time.perf_counter()
    model.load_state_dict(p.cpu_copy(state['model'])); ema.load_state_dict(p.cpu_copy(state['ema']))
    opt.load_state_dict(p.cpu_copy(state['optimizer']))
    loader.set_state(p.cpu_copy(state['loader']))
    for root,name in ((model,'model'),(ema,'ema')):
        for n,module in root.named_modules():
            module.training=state['source_modes'][name][n]
    p.restore_rng(p.cpu_copy(state['rng']),device)
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    return dict(seconds=time.perf_counter()-tick,seconds_semantics='actual_state_restore_RNG_last')


def roundtrip(state):
    b=io.BytesIO(); tick=time.perf_counter(); torch.save(state,b); size=b.tell(); b.seek(0)
    result=torch.load(b,map_location='cpu',weights_only=False)
    return result,dict(bytes=size,seconds=time.perf_counter()-tick)


def _verify_commit(key,snapshot,record,epoch_file,attempt_number,*,send,journal,epoch_sha,now=time.time,transaction=None,reconcile=False):
    """The transcript uses only actual serve fields; no owner-generated hashes are assumed."""
    p.validate_epoch_state(snapshot,record)
    events=dict(commit_requested=transaction['commit_requested'] if transaction else now()); warning=None
    if reconcile:
        warning='transport_warning_reconciled'; events['ram_ack_received']=None
    else:
        try:
            answer=send('commit',key=key,state=snapshot,epoch_file=epoch_file)
            p.require(answer==dict(status='RAM_COMMITTED',epoch=snapshot['epoch']), 'Unexpected commit ACK')
            events['ram_ack_received']=now()
        except (OSError,EOFError,ConnectionError):
            warning='transport_warning_reconciled'; events['ram_ack_received']=None
    status=send('status'); events['status_received']=now()
    got=send('get',key=key); events['get_received']=now()
    saved=got['state']; live=status['jobs'][key]
    p.require(saved is not None and live['epoch']==snapshot['epoch'], 'Commit not acknowledged at candidate epoch; technical PAUSE/replay required')
    p.validate_epoch_state(saved,record)
    p.require(p.state_digest(saved)==p.state_digest(snapshot) and live['job']==snapshot['job'] and
              live['counts']==snapshot['counts'] and
              (live['attempts']>=attempt_number if reconcile else live['attempts']==attempt_number), 'Owner returned conflicting commit')
    rows=[r for r in journal() if r['key']==key and r['epoch']==snapshot['epoch']]
    p.require(len(rows)==1,'Missing/ambiguous own server commit journal')
    jr=rows[0]
    p.require(jr['attempt']==attempt_number and jr['epoch_file']==epoch_file and jr['status']=='RAM_COMMITTED', 'Own journal differs')
    events['hash_verified']=now()
    p.require(jr['time']>=events['commit_requested'] and jr['time']<=events['get_received'], 'Journal chronology differs')
    return dict(key=key,attempt=attempt_number,epoch=snapshot['epoch'],epoch_file=epoch_file,
        epoch_file_sha256=epoch_sha,batch_log_sha256=record['batch_log_sha256'],source_hash=snapshot['source_hash'],
        design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.controls_hash(),owner_endpoint_hash=snapshot['owner_endpoint_hash'],
        continuation_hash=record['continuation_hash'],scientific_identity_hash=record['scientific_identity_hash'],
        ram_commit_status='RAM_COMMITTED',ram_get_verified=True,server_journal_row=jr,events=events,
        transport_warning_reconciled=warning is not None,test_accessed=False)


def commit_snapshot(attempt,key,snapshot,record,path,*,attempt_number):
    a=p.new_root(attempt); path=p.new_path(path)
    p.validate_epoch_state(snapshot,record); p.immutable(path,record)
    # This separate measured preflight does not insert its duration into the already hashed snapshot.
    cloned,serialization=roundtrip(snapshot)
    p.require(p.state_digest(cloned)==record['continuation_hash'],'CPU BytesIO preflight differs'); del cloned
    transaction=dict(key=key,purpose=snapshot['purpose'],epoch=snapshot['epoch'],attempt=attempt_number,epoch_file=str(path.relative_to(a)),
        epoch_file_sha256=p.sha_file(path),commit_requested=time.time(),continuation_hash=record['continuation_hash'],
        source_hash=snapshot['source_hash'],design_bundle_sha256=p.DESIGN_HASH,owner_endpoint_hash=snapshot['owner_endpoint_hash'],serialization=serialization,test_accessed=False)
    transaction_path=a/'transactions'/key/f'attempt{attempt_number}'/f'epoch_{snapshot["epoch"]:02d}.json'
    p.immutable(transaction_path,transaction)
    def journal():
        return [json.loads(line) for line in (a/'epoch_commit_ledger.jsonl').read_text().splitlines()]
    tick=time.perf_counter()
    ack=_verify_commit(key,snapshot,record,str(path.relative_to(a)),attempt_number,
        send=lambda command,**payload:p.request(a,command,**payload),journal=journal,epoch_sha=p.sha_file(path),transaction=transaction)
    ack.update(transaction_file=str(transaction_path.relative_to(a)),transaction_sha256=p.sha_file(transaction_path),
               serialization=serialization,commit_status_get_seconds=time.perf_counter()-tick)
    ack['events']['numeric_ack_written']=time.time()
    p.immutable(a/'acks'/key/f'epoch_{snapshot["epoch"]:02d}.json',ack)
    return ack


def reconcile_ack(attempt,key,snapshot):
    a=p.new_root(attempt); epoch=snapshot['epoch']; target=a/'acks'/key/f'epoch_{epoch:02d}.json'
    if target.exists(): return p.read_json(target)
    rows=[json.loads(line) for line in (a/'epoch_commit_ledger.jsonl').read_text().splitlines()]
    committed=[r for r in rows if r['key']==key and r['epoch']==epoch]
    p.require(len(committed)==1,'Missing/ambiguous committed transaction journal')
    transaction_path=a/'transactions'/key/f'attempt{committed[0]["attempt"]}'/f'epoch_{epoch:02d}.json'
    transaction=p.read_json(transaction_path)
    p.require(transaction['key']==key and transaction['epoch']==epoch and
              transaction['epoch_file']==committed[0]['epoch_file'] and transaction['attempt']==committed[0]['attempt'] and
              transaction['source_hash']==snapshot['source_hash'] and transaction['design_bundle_sha256']==p.DESIGN_HASH,
              'Committed transaction identity differs')
    path=p.new_path(a/transaction['epoch_file']); record=p.read_json(path)
    p.require(p.sha_file(path)==transaction['epoch_file_sha256'] and transaction['continuation_hash']==p.state_digest(snapshot),
              'Lost-reply pending transaction conflicts with owner state')
    rows=[json.loads(line) for line in (a/'epoch_commit_ledger.jsonl').read_text().splitlines()]
    ack=_verify_commit(key,snapshot,record,transaction['epoch_file'],transaction['attempt'],
        send=lambda command,**payload:p.request(a,command,**payload),journal=lambda:rows,
        epoch_sha=transaction['epoch_file_sha256'],transaction=transaction,reconcile=True)
    ack.update(transaction_file=str(transaction_path.relative_to(a)),transaction_sha256=p.sha_file(transaction_path),
        serialization=transaction['serialization'],commit_status_get_seconds=None,
        transfer_cost_null_reason='reply_lost_before_cost_record',events={**ack['events'],'numeric_ack_written':time.time()})
    p.immutable(target,ack); return ack


def register(attempt,j):
    try:
        return p.request(attempt,'register',key=j['key'],job=j,pid=os.getpid())
    except (OSError,EOFError,ConnectionError):
        status=p.request(attempt,'status'); r=status['jobs'].get(j['key'])
        p.require(r is not None and r['job']==j and r['worker_pid']==os.getpid() and r['status']=='RUNNING', 'Lost register reply unresolved')
        got=p.request(attempt,'get',key=j['key'])
        return dict(attempt=r['attempts'],state=got['state'],transport_warning_reconciled=True)


def initial_state(j,model,ema,opt,loader,device,owner):
    return dict(job=j,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.controls_hash(),
        owner_endpoint_hash=owner['endpoint_hash'],model=p.cpu_copy(model.state_dict()),ema=p.cpu_copy(ema.state_dict()),
        optimizer=p.cpu_copy(opt.state_dict()),rng=p.cpu_copy(p.rng_state(device)),loader=loader.get_state().clone(),
        source_modes={name:{n:m.training for n,m in root.named_modules()} for name,root in (('model',model),('ema',ema))},
        selected_models={},selected_epochs={},selected_validation={},best={k:-1. for k in p.ESTIMATORS},common_validation={},
        initial_hash=p.state_digest(model.state_dict()),initial_ema_hash=p.state_digest(ema.state_dict()),initial_optimizer_hash=p.state_digest(opt.state_dict()),
        purpose=j['purpose'],clock_kind='formal' if j['purpose']=='formal' else 'engineering',epoch=0,global_step=0,
        next_epoch=1 if j['purpose']=='formal' else None,scientific_epochs_completed=0,
        counts=dict(optimizer_updates=0,ema_updates=0),operations={n:0 for n in p.OPERATIONS},
        adam_step_counters={n:None for n,_ in model.named_parameters()},last_parameter_activity={},
        branch_counts=dict(ordinary=0,all_single_class=0,lambda_zero=0),costs={})


def capture(state,model,ema,opt,loader,device):
    tick=time.perf_counter(); result=p.cpu_copy(state)
    result.update(model=p.cpu_copy(model.state_dict()),ema=p.cpu_copy(ema.state_dict()),optimizer=p.cpu_copy(opt.state_dict()),
                  loader=loader.get_state().clone(),rng=p.cpu_copy(p.rng_state(device)),
                  source_modes={name:{n:m.training for n,m in root.named_modules()} for name,root in (('model',model),('ema',ema))})
    result['costs']['snapshot_capture_seconds']=time.perf_counter()-tick
    return result


def advance(state,value,rows):
    before=state['adam_step_counters']; after=value['adam_step_counters']; activity=value['parameter_activity']
    p.require(set(before)==set(after)==set(activity), 'Private optimizer parameter coverage differs')
    for name,old in before.items():
        expected=old if activity[name]=='none' else (old or 0)+1
        p.require(after[name]==expected, f'Source None/zero private step differs: {name}')
    state['adam_step_counters']=copy.deepcopy(after); state['last_parameter_activity']=copy.deepcopy(activity)
    for name in state['counts']:
        p.require(value[name]==1,'One live optimizer/EMA update required'); state['counts'][name]+=1
    o=state['operations']; diag=int(value['extra_autograd_sweeps']==3)
    increments=dict(training_rows=rows,task_entries=2*rows,ordinary_forwards=1,scalar_backwards=1,
                    live_adam_steps=1,live_ema_steps=1,diagnostic_points=diag,extra_autograd_sweeps=3*diag,
                    virtual_adam_steps=4*diag,probe_forwards=0,view_forwards=0,algorithm_rng_draws=0)
    for name,inc in increments.items():
        o[name]+=inc
    for name in ('ordinary_forwards','scalar_backwards','probe_forwards','view_forwards','algorithm_rng_draws','extra_autograd_sweeps','virtual_adam_steps'):
        p.require(value[name]==increments[name], f'Core operation differs: {name}')
    state['branch_counts'][value['branch']]+=1; state['global_step']+=1


def gradients(model):
    return {n:None if x.grad is None else x.grad.detach().cpu().clone() for n,x in model.named_parameters()}


def observation(model,ema,opt,loader,device):
    return dict(model=p.state_digest(model.state_dict()),ema=p.state_digest(ema.state_dict()),
                optimizer=p.state_digest(opt.state_dict()),rng=p.state_digest(p.rng_state(device)),
                loader=p.state_digest(loader.get_state()),gradients=p.state_digest(gradients(model)),
                activity={n:'none' if x.grad is None else ('tensor_zero' if torch.count_nonzero(x.grad).item()==0 else 'nonzero') for n,x in model.named_parameters()})


def validation(model,ema,bundle,device,state,ordinal,engineering=False):
    rng=p.cpu_copy(p.rng_state(device)); modes={name:{n:m.training for n,m in root.named_modules()} for name,root in (('model',model),('ema',ema))}
    values={}; costs={}
    try:
        for name,root in (('online',model),('fixed_ema',ema)):
            v,c=p.evaluate(root,bundle.val_x,bundle.val_y,device,batch_size=40960)
            p.json_safe(v)
            p.require(all(v[k+'_auc'] is not None and math.isfinite(v[k+'_auc']) for k in p.CELLS) and
                      v['macro_auc'] is not None, 'Undefined required validation AUC')
            values[name]=v; costs[name]=c
            if v['macro_auc']>state['best'][name]:
                state['best'][name]=v['macro_auc']; state['selected_epochs'][name]=ordinal
                state['selected_validation'][name]=copy.deepcopy(v); state['selected_models'][name]=p.cpu_copy(root.state_dict())
        if not engineering and ordinal in p.DIAGNOSTIC_EPOCHS:
            state['common_validation'][str(ordinal)]=copy.deepcopy(values)
        if engineering:
            state['selection_clock']='engineering_commit_ordinal'
    finally:
        for root,name in ((model,'model'),(ema,'ema')):
            for n,m in root.named_modules(): m.training=modes[name][n]
        p.restore_rng(rng,device)
    return values,costs


def epoch_record(snapshot,*,validation,validation_cost,batch_path,batch_count,epoch_order_hash,seconds=0.):
    r={k:copy.deepcopy(snapshot[k]) for k in ('job','purpose','clock_kind','epoch','global_step','next_epoch',
        'scientific_epochs_completed','counts','operations','branch_counts','source_hash','design_bundle_sha256',
        'controls_index_hash','owner_endpoint_hash','selected_epochs','selected_validation','common_validation',
        'adam_step_counters','last_parameter_activity','costs')}
    r['job_key']=r.pop('job')['key']
    for k in ('engineering_commit_ordinal','engineering_state_step','sample_epoch','sample_batches','selection_clock'):
        if k in snapshot: r[k]=snapshot[k]
    dictionary=next((json.loads(path.read_text()) for path in (p.ATTEMPT/'dictionaries').glob('parameters_*.json')
        if [v['name'] for v in json.loads(path.read_text())['parameters']]==list(snapshot['adam_step_counters'])),None)
    p.require(dictionary is not None,'Epoch parameter dictionary missing')
    r['parameter_dictionary_id']=dictionary['dictionary_id']
    for field in ('adam_step_counters','last_parameter_activity'):
        r[field]=[r[field].get(v['name']) for v in dictionary['parameters']]
    r.update(p.component_hashes(snapshot)); r.update(continuation_hash=p.state_digest(snapshot),
        scientific_identity_hash=p.scientific_identity(snapshot),validation=validation,validation_cost=validation_cost,
        batch_log=str(Path(batch_path).relative_to(p.ATTEMPT)),batch_log_sha256=p.sha_file(batch_path),
        batch_count=batch_count,epoch_order_hash=epoch_order_hash,seconds=seconds,test_accessed=False)
    return r


def allocation_ack(a,j,profile,number):
    profile.number=number; row=profile.mark('first_FULL_lazy_Adam_allocation_ACK')
    peak=max(profile.peaks['own_process_bytes'],profile.peaks['torch_reserved_bytes']+profile.peaks['non_torch_bytes'])
    resource_observation(a,profile,number)
    reply=p.request(a,'allocated',key=j['key'],peak_memory_bytes=peak)
    p.require(reply==dict(status='ALLOCATED'),'Allocation ACK differs')
    current=profile.nvml.sample()
    p.append_json(a/'allocation_acks.jsonl',dict(key=j['key'],attempt=number,worker_pid=os.getpid(),device=profile.device,time=time.time(),
        source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,peak_memory_bytes=peak,reply=reply,
        sample_id=current['sample_id'],after_real_FULL=True,test_accessed=False))


def resource_observation(a,profile,number):
    profile.number=number; receipt=profile.receipt(); payload=witness_payload(receipt); payload['attempt']=number
    wid=p.digest(payload); row=dict(witness_id=wid,**payload)
    p.append_json(a/'profiles'/f'producer_{os.getpid()}.jsonl',row)
    ref=dict(witness_id=wid,producer_journal=str(a/'profiles'/f'producer_{os.getpid()}.jsonl'))
    profile.witness_refs.append(ref); return ref


def parameter_dictionary(a,model):
    names=[dict(name=n,shape=list(x.shape),dtype=str(x.dtype),numel=x.numel()) for n,x in model.named_parameters()]
    dictionary_id=p.digest(names)
    groups={'ALL':list(range(len(names)))}
    for prefix in sorted({row['name'].split('.',1)[0] for row in names}):
        groups[prefix]=[i for i,row in enumerate(names) if row['name'].split('.',1)[0]==prefix]
    p.immutable(a/'dictionaries'/f'parameters_{dictionary_id}.json',dict(dictionary_id=dictionary_id,parameters=names,groups=groups))
    return dictionary_id,[x['name'] for x in names]


def normalize_value(a,value,model,context,key,number):
    result=copy.deepcopy(value); did,names=parameter_dictionary(a,model)
    result['parameter_dictionary_id']=did
    for field in ('parameter_activity','adam_step_counters'):
        result[field]=[result[field][n] for n in names]
    diag=result.get('diagnostics')
    if diag is not None:
        point_payload={k:v for k,v in diag.items() if k not in ('parameter_groups','point_id')}
        point_payload.update(key=key,job_id=key,physical_attempt_id=f'{os.getpid()}:{number}',attempt=number,
            context=context,parameter_dictionary_id=did,core_point_id=diag['point_id'])
        point_id=p.digest(point_payload)
        point=dict(point_id=point_id,**point_payload,parameter_groups={g:{k:v for k,v in fields.items() if k!='parameter_names'}
            for g,fields in diag['parameter_groups'].items()})
        p.immutable(a/'diagnostics'/f'{point_id}.json',point)
        result['diagnostics']=dict(point_id=point_id,actual_virtual_exact=diag['actual_virtual_exact'])
    return result

def _new_objects(j,bundle,device):
    p.setup(j['seed'],device)
    model=p.make_model(bundle,p.load_protocol()['datasets'][j['dataset']]['source']).to(device)
    ema=copy.deepcopy(model).requires_grad_(False); opt=p.optimizer(model,j['config'])
    loader=torch.Generator(device='cpu').manual_seed(j['seed']+11000)
    return model,ema,opt,loader


def _engineering_context(j,call,step,sample_batch,role,loader):
    return dict(purpose='engineering_smoke',clock_kind='engineering',dataset=j['dataset'],seed=j['seed'],
        case_id=j['case_id'],engineering_call_ordinal=call,engineering_state_step=step,sample_epoch=1,
        sample_batch=sample_batch,branch_role=role,pre_loader_hash=p.state_digest(loader.get_state()))


def _receipt(path):
    return dict(path=str(path),sha256=p.sha_file(path))


def _engineering_proof(a,j,saved):
    ack=reconcile_ack(a,j['key'],saved); record=p.read_json(a/ack['epoch_file'])
    p.validate_epoch_state(saved,record)
    proof=p.receipt_json(record['engineering_proof'])
    p.require(proof['job']==j and proof['owner_endpoint_hash']==saved['owner_endpoint_hash'] and
              proof['suffix_scientific_identity_hash']==p.scientific_identity(saved), 'Engineering proof suffix differs')
    prefix=p.receipt_json(proof['prefix_ack'])
    p.require(prefix==p.read_json(a/'acks'/j['key']/'epoch_01.json') and prefix['key']==j['key'] and
              prefix['continuation_hash']==proof['prefix_hash'] and prefix['ram_get_verified'], 'Engineering proof prefix differs')
    calls=[]
    for receipt in proof['call_logs']:
        p.require(p.sha_file(receipt['path'])==receipt['sha256'],'Engineering call log changed')
        calls.extend(json.loads(line) for line in Path(receipt['path']).read_text().splitlines())
    p.require([x['call'] for x in calls]==list(range(1,9)), 'Engineering eight-call proof incomplete')
    roles=('prefix','prefix','source_CLEAN_fork','lambda_zero_fork','suffix_expected','suffix_expected','restored_suffix','restored_suffix')
    for i,x in enumerate(calls,1):
        p.require(x['key']==j['key'] and x['source_hash']==j['source_hash'] and x['design_bundle_sha256']==p.DESIGN_HASH and
                  x['context']==dict(purpose='engineering_smoke',clock_kind='engineering',dataset=j['dataset'],seed=j['seed'],
                      case_id=j['case_id'],engineering_call_ordinal=i,engineering_state_step=(1,2,3,3,3,4,3,4)[i-1],sample_epoch=1,
                      sample_batch=1 if i in (1,3,4,5,7) else p.STEPS[j['dataset']],branch_role=roles[i-1],
                      pre_loader_hash=calls[0]['context']['pre_loader_hash']) and
                  x['diagnose']==(i in (1,6)) and x['B']==(40960 if i in (1,3,4,5,7) else p.TAILS[j['dataset']]),
                  'Engineering call identity/plan differs')
    p.require(dict(physical_train_calls=len(calls),diagnostic_points=sum(x['diagnose'] for x in calls),
                   extra_autograd_sweeps=sum(x.get('value',{}).get('extra_autograd_sweeps',0) for x in calls),
                   virtual_adam_steps=sum(x.get('value',{}).get('virtual_adam_steps',0) for x in calls))==p.ENGINEERING_PLAN and
              all(calls[i-1]['value']['diagnostics']['actual_virtual_exact'] for i in (1,6)),
              'Engineering lazy/populated diagnostic proof missing')
    p.require(calls[2]['observation']==calls[3]['observation'] and calls[5]['observation']==calls[7]['observation'] and
              proof['source_lambda_zero_exact'] and proof['prefix_immutable'] and proof['diagnostics_neutral_exact'] and
              proof['validation']==record['validation']==saved['selected_validation'] and
              proof['expected_scientific_identity_hash']==proof['suffix_scientific_identity_hash'], 'Engineering exact proof differs')
    p.require({x['row_order_hash'] for x in calls if x['call'] in (1,3,4,5,7)}=={calls[0]['row_order_hash']} and
              {x['row_order_hash'] for x in calls if x['call'] in (2,6,8)}=={calls[1]['row_order_hash']}, 'Engineering source sample order differs')
    return proof,calls


def _engineering_finish(a,j,saved,out,device):
    proof,calls=_engineering_proof(a,j,saved)
    receipts=[_receipt(path) for path in sorted((a/'smoke'/j['case_id']).glob('attempt*/resource_profile.json'))]
    p.require(receipts,'Actual engineering profiles missing'); profiles=[p.receipt_json(r) for r in receipts]
    prof=copy.deepcopy(profiles[-1])
    for r in profiles:
        p.require(all(r[k]==prof[k] for k in ('key','dataset','arm','device','source_hash','design_bundle_sha256','hardware','environment')) and
                  r['raw_log'] and r['witness_refs'],'Engineering profile provenance differs')
    for field in ('full_process_peak_bytes','torch_allocated_peak_bytes','torch_reserved_peak_bytes','non_torch_bytes','peak_host_rss_bytes'):
        prof[field]=max(r[field] for r in profiles)
    prof['profile_refs']=receipts; prof['phases']=sorted({phase for r in profiles for phase in r['phases']})
    prof['witness_refs']=[x for r in profiles for x in r['witness_refs']]
    profile_path=out/'case_profile.json'; p.immutable(profile_path,prof)
    call_logs=[_receipt(path) for path in sorted((a/'smoke'/j['case_id']).glob('attempt*/batches.jsonl'))]
    counts=p.engineering_physical_counts(j,call_logs)
    p.require(all(counts[k]>=v for k,v in p.ENGINEERING_PLAN.items()),'Physical work omits verified plan')
    result=dict(status='ENGINEERING_VERIFIED_PENDING_REVIEW',key=j['key'],job=j,device=int(device),dataset=j['dataset'],arm=j['arm'],
        **counts,physical_train_calls_semantics='returned_calls; unreturned work is null in terminal core',verified_plan_counts=dict(p.ENGINEERING_PLAN),
        engineering_state_step=4,engineering_commit_ordinal=2,scientific_epochs_completed=0,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        prefix_hash=proof['prefix_hash'],suffix_scientific_identity_hash=proof['suffix_scientific_identity_hash'],source_lambda_zero_exact=True,
        diagnostics_neutral_exact=True,actual_virtual_exact=True,actual_gradients_activity_exact=True,prefix_immutable=True,
        validation=proof['validation'],validation_costs=proof['validation_costs'],restore_bytesio=proof['restore_bytesio'],
        proof=_receipt(Path(proof['path'])),profile=_receipt(profile_path),physical_call_logs=call_logs,
        physical_attempts_path=str(a/'physical_attempts.jsonl'),test_accessed=False)
    p.immutable(out/'case.json',result); return result


class LocalAttempt:
    def __init__(self,a,key,device,purpose,source_hash):
        self.a=a; self.key=key; self.device=int(device); self.purpose=purpose; self.source_hash=source_hash
        self.pid=os.getpid(); self.start=time.time(); self.mono=time.monotonic(); self.number=None; self.profile=None; self.log=None
        self.returned=0; self.operations={n:0 for n in p.OPERATIONS}; self.pending=None; self.last=None; self.last_ack=None
        self.out=None; self.error=None; self.status='PAUSED'; self.cleanup_errors=[]

    def facts(self,status):
        return dict(status=status,key=self.key,purpose=self.purpose,pid=self.pid,device=self.device,attempt=self.number,
            start_time=self.start,start_monotonic=self.mono,end_time=time.time(),end_monotonic=time.monotonic(),
            seconds=time.monotonic()-self.mono,source_hash=self.source_hash,design_bundle_sha256=p.DESIGN_HASH,
            returned_train_calls=self.returned,returned_operations=self.operations,last_completed_operation=self.last,
            current_pending_call=self.pending,unknown_partial_call=self.pending is not None,
            unreturned_call_operations=None if self.pending is not None else 0,
            unknown_null_reason='train_step_did_not_return_receipt' if self.pending is not None else None,
            last_observed_ack=None if self.last_ack is None else {k:self.last_ack[k] for k in ('key','attempt','epoch','continuation_hash','epoch_file','epoch_file_sha256')},original_exception=self.error,test_accessed=False)

    def call_started(self,ordinal,context):
        self.pending=dict(ordinal=ordinal,context=context,start_monotonic=time.monotonic())

    def call_returned(self,value,rows):
        self.returned+=1; self.last=dict(operation='train_step',ordinal=self.pending['ordinal'])
        self.operations['training_rows']+=rows; self.operations['task_entries']+=2*rows
        mapping=dict(ordinary_forwards='ordinary_forwards',scalar_backwards='scalar_backwards',live_adam_steps='optimizer_updates',
            live_ema_steps='ema_updates',extra_autograd_sweeps='extra_autograd_sweeps',virtual_adam_steps='virtual_adam_steps',
            probe_forwards='probe_forwards',view_forwards='view_forwards',algorithm_rng_draws='algorithm_rng_draws')
        for name,field in mapping.items(): self.operations[name]+=int(value[field]) if value is not None else int(name in ('ordinary_forwards','scalar_backwards','live_adam_steps','live_ema_steps'))
        self.operations['diagnostic_points']+=int(value is not None and value['extra_autograd_sweeps']==3)
        self.pending=None

    def isolated(self,name,call):
        try: return call()
        except BaseException as error:
            self.cleanup_errors.append(dict(cleanup=name,error_type=type(error).__name__,error=str(error)))
            return None

    def finish(self,*,owner_cleanup=None):
        # Core facts are written before sampler, owner IPC, or any other cleanup.
        p.append_json(self.a/'physical_attempts.jsonl',self.facts('TERMINAL_CORE'))
        if self.log is not None: self.isolated('batch_log_close',self.log.close)
        if self.profile is not None:
            if not self.profile.closed: self.isolated('profile_close',self.profile.close)
            if self.profile.cuda_ready and self.out is not None:
                self.isolated('profile_receipt',lambda:p.write_json(self.out/'resource_profile.json',self.profile.receipt()))
        if owner_cleanup is not None: self.isolated('owner_status',owner_cleanup)
        if self.cleanup_errors:
            p.append_json(self.a/'cleanup_errors.jsonl',dict(key=self.key,pid=self.pid,attempt=self.number,errors=self.cleanup_errors,
                original_exception=self.error,time=time.time(),test_accessed=False))
        p.append_json(self.a/'physical_attempts.jsonl',dict(self.facts('CLEANUP_COMPLETE'),physical_status=self.status,
            cleanup_errors=len(self.cleanup_errors)))


def _execute(attempt,key,device,purpose,body):
    a=p.new_root(attempt)
    # Local STARTED/TERMINAL_CORE do not require registration, Profile, or live storage.
    source=None
    try: source=p.read_json(a/'run_manifest.json')['source_hash']
    except Exception: pass
    local=LocalAttempt(a,key,device,purpose,source)
    p.append_json(a/'physical_attempts.jsonl',local.facts('STARTED'))
    try:
        result=body(local); local.status=result.get('status','RETURNED') if isinstance(result,dict) else 'RETURNED'
        return result
    except BaseException as error:
        local.error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        local.isolated('technical_pause',lambda:p.write_json(a/'TECHNICAL_PAUSE.json',dict(key=key,source_hash=source,
            design_bundle_sha256=p.DESIGN_HASH,reason=local.error,time=time.time(),test_accessed=False)))
        local.isolated('pause_latch',lambda:p.new_path(a/'PAUSE_TRAINING').touch())
        raise
    finally:
        def owner_cleanup():
            if local.error is not None and local.number is not None:
                local.isolated('owner_failed',lambda:p.request(a,'failed',key=local.key,failure=dict(type=local.error['type'],time=time.time())))
            return p.request(a,'status')
        local.finish(owner_cleanup=owner_cleanup)


def smoke_case(attempt,case_id,device,dataset,arm,recovery_review=None):
    return _execute(attempt,None,device,'engineering_smoke',lambda local:_smoke_body(attempt,case_id,device,dataset,arm,recovery_review,local))


def _smoke_body(attempt,case_id,device,dataset,arm,recovery_review,local):
    a=p.new_root(attempt); m=p.validate_attempt(a); p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'Stop/pause before engineering')
    p.require((int(device),dataset,arm) in CASES,'Unregistered engineering case')
    case_number=CASES.index((int(device),dataset,arm))+1
    p.require(case_id==f'case{case_number}_gpu{device}_{dataset}','Engineering output case identity differs')
    j=p.job(dataset,arm,2022,m['source_hash'],'engineering_smoke',case_id); local.key=j['key']; p.immutable(a/'jobs'/f'{j["key"]}.json',j)
    previous=p.request(a,'status')['jobs'].get(j['key'])
    if previous is not None:
        p.require(recovery_review is not None,'Existing engineering case requires fresh reviewed recovery')
        _engineering_recovery_review(a,recovery_review,j['key'])
        saved=p.request(a,'get',key=j['key'])['state']
        if saved is not None:
            _verify_saved_ack(a,j,saved)
            if saved['epoch']==2:
                # All eight calls were verified before commit; finalize from the exact ACK-bound proof.
                p.request(a,'paused',key=j['key'],reason='ENGINEERING_VERIFIED')
                out=a/'smoke'/case_id/f'attempt{previous["attempts"]}'
                return _engineering_finish(a,j,saved,out,device)
    registration=register(a,j); number=registration['attempt']; saved=registration['state']
    out=a/'smoke'/case_id/f'attempt{number}'; out.mkdir(parents=True,exist_ok=True)
    local.number=number; profile=Profile(device,j,number); local.profile=profile; local.out=out; dev=f'cuda:{device}'; started=time.perf_counter(); physical_calls=0; physical=[]; log=None
    terminal='PAUSED'; profile_closed=False; call_pending=False; segment_started=started
    p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
        status='STARTED',resume_ack_epoch=saved['epoch'] if saved else 0,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
    try:
        bundle,_=p.dataset_cache(dataset); profile.mark('data_loaded'); profile.initialize_cuda()
        model,ema,opt,loader=_new_objects(j,bundle,dev); parameter_dictionary(a,model); profile.mark('staged_model_EMA_lazy_Adam')
        def record_cost(kind,role,cost):
            p.append_json(a/'costs.jsonl',dict(key=j['key'],attempt=number,purpose=j['purpose'],cost_kind=kind,role=role,
                source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,**cost,test_accessed=False))
        def load(state,role):
            try: cost=restore(model,ema,opt,loader,state,dev)
            except BaseException:
                record_cost('restore',role,dict(seconds=None,null_reason='restore_did_not_return')); raise
            record_cost('restore',role,cost)
        if saved is None:
            state=initial_state(j,model,ema,opt,loader,dev,p.owner_binding(a)); order=torch.randperm(p.ROWS[dataset],generator=loader).numpy()
            state.update(sample_order=order.copy(),sample_epoch=1,sample_batches=[1,p.STEPS[dataset]])
        else:
            p.require(saved['epoch']==1,'Only engineering prefix can resume calls')
            load(saved,'reviewed_prefix_restore'); state=p.cpu_copy(saved); order=state['sample_order']
        samples=(order[:40960],order[-p.TAILS[dataset]:]); batch_path=out/'batches.jsonl'; log=batch_path.open('x'); local.log=log
        def call(ordinal,step,which,diagnose,role,delegate=False,lambda_zero=False):
            nonlocal physical_calls,call_pending
            p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'USER_STOPPED/PAUSE engineering atomic boundary')
            ids=samples[which]; x=torch.as_tensor(bundle.train_x[ids],device=dev).T; y=torch.as_tensor(bundle.train_y[ids],device=dev)
            context=_engineering_context(j,ordinal,step,1 if which==0 else p.STEPS[dataset],role,loader)
            profile.mark(f'call{ordinal}_{"FULL" if which==0 else "tail"}_{"diagnosed" if diagnose else "ordinary"}_before')
            tick=time.perf_counter(); c=j['config'] if not lambda_zero else dict(j['config'],mixture_lambda=0.)
            call_pending=True; local.call_started(ordinal,context)
            if delegate:
                loss=source_clean_step(model,opt,ema,x,y,c); v=None
            else:
                v=train_step(model,opt,ema,x,y,c,context,row_ids=bundle.train_row_ids[ids],diagnose=diagnose)
            torch.cuda.synchronize(dev); physical_calls+=1
            observed=observation(model,ema,opt,loader,dev); did,names=parameter_dictionary(a,model)
            observed['activity']=[observed['activity'][n] for n in names]; observed['parameter_dictionary_id']=did
            receipt=dict(key=j['key'],attempt=number,pid=os.getpid(),source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                call=ordinal,state_step=step,role=role,diagnose=diagnose,B=len(ids),seconds=time.perf_counter()-tick,
                row_order_hash=p.state_digest(np.asarray(bundle.train_row_ids[ids],dtype=np.int64)),
                observation=observed,context=context,test_accessed=False)
            if v is not None: receipt['value']=normalize_value(a,v,model,context,j['key'],number)
            else: receipt['source_loss']=loss
            physical.append(receipt); log.write(json.dumps(receipt,allow_nan=False)+'\n'); log.flush(); call_pending=False; local.call_returned(v,len(ids))
            profile.mark(f'call{ordinal}_after'); return v
        model.train()
        if saved is None:
            v=call(1,1,0,True,'prefix'); advance(state,v,len(samples[0])); allocation_ack(a,j,profile,number)
            v=call(2,2,1,False,'prefix'); advance(state,v,len(samples[1])); state.update(epoch=1,engineering_commit_ordinal=1,engineering_state_step=2)
            prefix=capture(state,model,ema,opt,loader,dev)
            record_cost('CPU_snapshot_capture','prefix',dict(seconds=prefix['costs']['snapshot_capture_seconds']))
            prefix_batch=out/'prefix_batches.jsonl'; prefix_batch.write_text(batch_path.read_text())
            r=epoch_record(prefix,validation={},validation_cost={},batch_path=prefix_batch,batch_count=2,
                           epoch_order_hash=p.state_digest(bundle.train_row_ids[order]))
            profile.mark('prefix_CPU_copy_serialization_commit_get')
            local.last_ack=commit_snapshot(a,j['key'],prefix,r,out/'engineering_commit_01.json',attempt_number=number)
        else:
            prefix=p.cpu_copy(saved); ack=p.read_json(a/'acks'/j['key']/'epoch_01.json')
            prefix_record=p.read_json(a/ack['epoch_file']); prefix_batch=a/prefix_record['batch_log']
            p.require(p.sha_file(prefix_batch)==prefix_record['batch_log_sha256']==ack['batch_log_sha256'],'Prefix call log changed')
        prefix_hash=p.state_digest(prefix)
        load(prefix,'source_CLEAN_fork'); call(3,3,0,False,'source_CLEAN_fork',delegate=True); direct=physical[-1]['observation']
        if saved is not None: allocation_ack(a,j,profile,number)
        load(prefix,'lambda_zero_fork'); call(4,3,0,False,'lambda_zero_fork',lambda_zero=True)
        p.require(direct==physical[-1]['observation'],'Populated source CLEAN versus lambda0 differs')
        p.require(p.state_digest(prefix)==prefix_hash,'Saved prefix mutated by forks')
        load(prefix,'suffix_expected'); state=p.cpu_copy(prefix)
        v=call(5,3,0,False,'suffix_expected'); advance(state,v,len(samples[0]))
        v=call(6,4,1,True,'suffix_expected'); advance(state,v,len(samples[1])); expected_gradient=physical[-1]['observation']
        profile.mark('expected_full_validation_online_EMA'); vals,cost=validation(model,ema,bundle,dev,state,2,True)
        for kind,value in cost.items(): record_cost('validation','expected_'+kind,value)
        state.update(epoch=2,engineering_commit_ordinal=2,engineering_state_step=4)
        expected=capture(state,model,ema,opt,loader,dev); expected_identity=p.scientific_identity(expected)
        record_cost('CPU_snapshot_capture','expected_suffix',dict(seconds=expected['costs']['snapshot_capture_seconds']))
        p.request(a,'paused',key=j['key'],reason='ENGINEERING_RESTORE_SUFFIX')
        p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
            status='PAUSED_FOR_VERIFIED_REPLAY',returned_train_calls=sum(x['attempt']==number for x in physical),unknown_partial_call=False,
            seconds=time.perf_counter()-segment_started,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
        second=register(a,j); p.require(second['attempt']==number+1,'Engineering re-register attempt differs'); number=second['attempt']; local.number=number; profile.number=number; profile.raw.emit(dict(event='REGISTRATION',attempt=number,pid=os.getpid(),key=j['key']))
        segment_started=time.perf_counter()
        p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
            status='STARTED',resume_ack_epoch=1,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
        get_started=time.perf_counter(); got=p.request(a,'get',key=j['key'])
        record_cost('RAM_get','prefix_replay',dict(seconds=time.perf_counter()-get_started))
        p.require(p.state_digest(got['state'])==prefix_hash and p.state_digest(second['state'])==prefix_hash,'Owner prefix changed')
        saved_copy,bytes_cost=roundtrip(got['state']); record_cost('BytesIO','prefix_replay',bytes_cost)
        profile.mark('BytesIO_deep_restore_RNG_last')
        load(saved_copy,'restored_suffix'); state=p.cpu_copy(saved_copy)
        v=call(7,3,0,False,'restored_suffix'); advance(state,v,len(samples[0])); allocation_ack(a,j,profile,number)
        v=call(8,4,1,False,'restored_suffix'); advance(state,v,len(samples[1]))
        p.require(physical[-1]['observation']==expected_gradient,'Diagnose-on/off restored suffix gradients/state differs')
        profile.mark('replayed_full_validation_online_EMA'); vals2,cost2=validation(model,ema,bundle,dev,state,2,True)
        for kind,value in cost2.items(): record_cost('validation','replayed_'+kind,value)
        state.update(epoch=2,engineering_commit_ordinal=2,engineering_state_step=4)
        replay=capture(state,model,ema,opt,loader,dev)
        record_cost('CPU_snapshot_capture','replayed_suffix',dict(seconds=replay['costs']['snapshot_capture_seconds']))
        p.require(vals2==vals and p.scientific_identity(replay)==expected_identity,'Exact restored suffix identity/validation differs')
        p.require(p.state_digest(prefix)==prefix_hash and p.state_digest(saved_copy)==prefix_hash,'Saved prefix optimizer mutated')
        log.close(); log=None; suffix_batch=out/'restored_suffix_batches.jsonl'
        suffix_batch.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in physical if x['call'] in (7,8)))
        # Preserve real proof observations before committing the verified suffix.
        remaining=out/'proof_suffix_calls.jsonl'
        remaining.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in physical if x['call']>=3))
        proof_path=out/'verified_plan.json'
        proof=dict(job=j,owner_endpoint_hash=replay['owner_endpoint_hash'],path=str(proof_path),prefix_hash=prefix_hash,
            prefix_ack=_receipt(a/'acks'/j['key']/'epoch_01.json'),call_logs=[_receipt(prefix_batch),_receipt(remaining)],
            source_lambda_zero_exact=True,prefix_immutable=True,diagnostics_neutral_exact=True,
            expected_scientific_identity_hash=expected_identity,suffix_scientific_identity_hash=p.scientific_identity(replay),
            validation=vals2,validation_costs=[cost,cost2],restore_bytesio=bytes_cost,test_accessed=False)
        p.immutable(proof_path,proof)
        r=epoch_record(replay,validation=vals2,validation_cost=cost2,batch_path=suffix_batch,batch_count=2,
                       epoch_order_hash=p.state_digest(bundle.train_row_ids[order]))
        r['engineering_proof']=_receipt(proof_path)
        profile.mark('suffix_CPU_copy_serialization_commit_get')
        local.last_ack=commit_snapshot(a,j['key'],replay,r,out/'engineering_commit_02.json',attempt_number=number)
        profile.mark('suffix_commit_get_verified')
        p.request(a,'paused',key=j['key'],reason='ENGINEERING_VERIFIED'); terminal='ENGINEERING_VERIFIED'
        resource_observation(a,profile,number); profile.close(); profile_closed=True; p.immutable(out/'resource_profile.json',profile.receipt())
        return _engineering_finish(a,j,replay,out,device)
    finally:
        # The outer local-finally owns cleanup and preserves primary failures.
        local.log=log


def _complete(a,j,result):
    try:
        reply=p.request(a,'complete',key=j['key'],result=result)
        p.require(reply==dict(status='COMPLETE'),'Completion ACK differs')
    except (OSError,EOFError,ConnectionError):
        p.require(p.request(a,'status')['jobs'][j['key']]['status']=='COMPLETE','Lost completion reply unresolved')
    got=p.request(a,'get',key=j['key'])
    p.require(got['result']==result and set(got['state'])=={'selected_models','selected_epochs','selected_validation'},'Compacted selected RAM differs')
    p.require({k:p.state_digest(v) for k,v in got['state']['selected_models'].items()}==result['selected_hashes'],'Compacted selected hashes differ')


def worker(attempt,key,device):
    return _execute(attempt,key,device,'formal',lambda local:_worker_body(attempt,key,device,local))


def _worker_body(attempt,key,device,local):
    a=p.new_root(attempt); manifest=p.validate_attempt(a,True); j=p.read_json(a/'jobs'/f'{key}.json')
    p.require(j['key']==key and j==p.job(j['dataset'],j['arm'],j['seed'],manifest['source_hash']), 'Formal job/path identity differs')
    p.require(j['purpose']=='formal' and not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'Worker stopped/paused')
    registration=register(a,j); number=registration['attempt']; saved=registration['state']
    out=a/'runs'/key/f'attempt{number}'; out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); returned=0; local.number=number; profile=Profile(device,j,number); local.profile=profile; local.out=out; dev=f'cuda:{device}'; state=None
    p.append_json(a/'physical_attempts.jsonl',dict(key=key,attempt=number,status='STARTED',pid=os.getpid(),device=int(device),
        resume_ack_epoch=saved['epoch'] if saved else 0,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        purpose='formal',time=time.time(),test_accessed=False))
    try:
        bundle,_=p.dataset_cache(j['dataset'],manifest['cache_dir']); profile.initialize_cuda()
        model,ema,opt,loader=_new_objects(j,bundle,dev); parameter_dictionary(a,model)
        profile.mark('staged_model_EMA_Adam')
        state=initial_state(j,model,ema,opt,loader,dev,p.owner_binding(a))
        references=p.load_reference_index()
        p.require(state['initial_hash']==references['initial_hashes'][f'{j["dataset"]}/{j["seed"]}'],'Source initialization differs')
        if saved is not None:
            _verify_saved_ack(a,j,saved); cost=dict(seconds=None,null_reason='restore_did_not_return')
            try: cost=restore(model,ema,opt,loader,saved,dev)
            finally:
                p.append_json(a/'costs.jsonl',dict(key=key,attempt=number,purpose='formal',cost_kind='restore',
                    resume_ack_epoch=saved['epoch'],source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                    **cost,test_accessed=False))
            state=p.cpu_copy(saved)
        initial_epoch=state['epoch']; del saved,registration
        allocated=False
        for epoch in range(initial_epoch+1,51):
            if p.stop_requested() or (a/'PAUSE_TRAINING').exists():
                p.request(a,'paused',key=key,reason='USER_STOPPED' if p.stop_requested() else 'TECHNICAL_PAUSE'); return
            tick=time.perf_counter(); model.train(); order=torch.randperm(len(bundle.train_y),generator=loader).numpy()
            order_hash=p.state_digest(np.asarray(bundle.train_row_ids[order],dtype=np.int64))
            p.require(order_hash==references['epoch_order_hashes'][f'{j["dataset"]}/{j["seed"]}/{epoch}'],'Source epoch order differs')
            batch_path=out/f'batches_epoch_{epoch:02d}.jsonl.gz'; batch_count=0; sums={n:0. for n in ('loss','clean_loss','same_class_loss','completion_loss','scaled_loss','cell_mass_loss')}; null_sums=set()
            from .tables import mass_add,mass_rows,statistic_add,statistic_rows
            mass_acc=defaultdict(lambda:dict(update_sum=0.,row_weighted_sum=0.,batches=0,rows=0))
            stat_acc=defaultdict(lambda:dict(observed_batches=0,null_batches=0,sum=0.,count_weighted_sum=0.,observed_count=0,null_reasons=defaultdict(int)))
            cell_counts={cell:dict(n=0,n0=0,n1=0) for cell in p.CELLS}
            with gzip.open(p.new_path(batch_path),'xt',compresslevel=3) as stream:
                for b,offset in enumerate(range(0,len(order),40960),1):
                    ids=order[offset:offset+40960]; context=dict(purpose='formal',clock_kind='formal',dataset=j['dataset'],seed=j['seed'],
                        epoch=epoch,batch=b,global_step=state['global_step']+1,pre_loader_hash=p.state_digest(loader.get_state()))
                    diag=b==1 and epoch in p.DIAGNOSTIC_EPOCHS
                    profile.mark(f'E{epoch}_B{b}_{"diagnosed" if diag else "ordinary"}_before')
                    x=torch.as_tensor(bundle.train_x[ids],device=dev).T; y=torch.as_tensor(bundle.train_y[ids],device=dev)
                    local.call_started(returned+1,context)
                    v=train_step(model,opt,ema,x,y,j['config'],context,row_ids=bundle.train_row_ids[ids],diagnose=diag)
                    advance(state,v,len(ids)); returned+=1; batch_count+=1
                    mass_add(mass_acc,v['ledger'],len(ids)); statistic_add(stat_acc,v['ledger'])
                    for cell,values in v['ledger']['cells'].items():
                        for name in ('n','n0','n1'): cell_counts[cell][name]+=values[name]
                    for n in sums:
                        if v[n] is None: null_sums.add(n)
                        else: sums[n]+=v[n]*len(ids)
                    row=dict(key=key,attempt=number,context=context,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                        B=len(ids),entries=2*len(ids),counts=copy.deepcopy(state['counts']),operations=copy.deepcopy(state['operations']),**normalize_value(a,v,model,context,key,number),test_accessed=False)
                    stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'); stream.flush(); local.call_returned(v,len(ids))
                    del x,y,v; profile.mark(f'E{epoch}_B{b}_after')
                    if diag: resource_observation(a,profile,number)
                    if not allocated:
                        allocation_ack(a,j,profile,number); allocated=True
                    # Stop after the atomic call. A full epoch may still be validated/committed.
                    if b<p.STEPS[j['dataset']] and (p.stop_requested() or (a/'PAUSE_TRAINING').exists()):
                        p.request(a,'paused',key=key,reason='USER_STOPPED' if p.stop_requested() else 'TECHNICAL_PAUSE'); return
            profile.mark('full_validation_online_EMA'); vals,cost=validation(model,ema,bundle,dev,state,epoch)
            state.update(epoch=epoch,next_epoch=epoch+1,scientific_epochs_completed=epoch)
            state['costs']['worker_seconds_since_attempt_start']=time.perf_counter()-started
            profile.mark('CPU_snapshot_capture'); snapshot=capture(state,model,ema,opt,loader,dev)
            record=epoch_record(snapshot,validation=vals,validation_cost=cost,batch_path=batch_path,batch_count=batch_count,
                                epoch_order_hash=order_hash,seconds=time.perf_counter()-tick)
            record.update({n:None if n in null_sums else value/len(order) for n,value in sums.items()})
            record['same_class_loss_null_reason']='not_computed_delegate' if 'same_class_loss' in null_sums else None
            record['objective_ledger']=dict(batches=batch_count,rows=len(order),cells=cell_counts,
                mass_summaries=mass_rows({},mass_acc),diagnostic_statistics=statistic_rows({},stat_acc))
            profile.mark('CPU_BytesIO_commit_get'); local.last_ack=commit_snapshot(a,key,snapshot,record,out/f'epoch_{epoch:02d}.json',attempt_number=number)
            p.disk_observation(a,phase='epoch_commit',remaining_transactions=max(0,508-len(list((a/'acks').glob('*/epoch_*.json')))))
            resource_observation(a,profile,number)
            state=snapshot
        epochs=p.epoch_files(a,dict(job=j)); p.require(len(epochs)==50,'All50 ACKs required before source complete')
        last=epochs[50][1]; result=dict(job=j,epochs=50,global_step=state['global_step'],initial_hash=state['initial_hash'],
            selected_epochs=copy.deepcopy(state['selected_epochs']),selected_validation=copy.deepcopy(state['selected_validation']),
            selected_hashes=p.component_hashes(state)['selected_hashes'],common_validation=copy.deepcopy(state['common_validation']),
            final_hashes=dict(online=last['model_hash'],fixed_ema=last['ema_hash']),optimizer_hash=last['optimizer_hash'],
            counts=state['counts'],operations=state['operations'],source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
            controls_index_hash=p.controls_hash(),owner_endpoint_hash=state['owner_endpoint_hash'],worker_seconds=time.perf_counter()-started,
            epoch_receipts=[dict(epoch=e,path=str(path),sha256=p.sha_file(path)) for e,(path,r) in epochs.items()],test_accessed=False)
        p.verify_selection(result,epochs); _complete(a,j,result)
        return dict(status='COMPLETE',key=key)
    finally:
        pass


def _verify_saved_ack(a,j,saved):
    e=saved['epoch']; ack=reconcile_ack(a,j['key'],saved)
    record=p.read_json(a/ack['epoch_file']); p.validate_epoch_state(saved,record)
    p.require(p.state_digest(saved)==ack['continuation_hash'] and ack['ram_get_verified'] and
              p.sha_file(a/ack['epoch_file'])==ack['epoch_file_sha256'],'Last acknowledged RAM state differs')


def _engineering_recovery_review(a,review_path,key):
    m=p.validate_attempt(a); r=p.check_review(review_path,'recovery',m['source_hash']); owner=p.owner_binding(a)
    pause=p.read_json(a/'TECHNICAL_PAUSE.json')
    allowed={p.job(d,arm,2022,m['source_hash'],'engineering_smoke',f'case{i+1}_gpu{dev}_{d}')['key']
             for i,(dev,d,arm) in enumerate(CASES)}
    p.require(key in allowed and set(r['last_acks'])=={key},'Recovery requires one of the four fixed engineering keys')
    p.require(r['scope']=='engineering' and r['case_keys']==[key] and
              pause['source_hash']==m['source_hash'] and pause['design_bundle_sha256']==p.DESIGN_HASH and
              r['pause_sha256']==p.sha_file(a/'TECHNICAL_PAUSE.json') and r['owner_endpoint_hash']==owner['endpoint_hash'],
              'Engineering recovery review scope/pause/owner differs')
    return r


def recover(attempt,review_path,*,purpose='formal',case_key=None):
    a=p.new_root(attempt); p.require(not p.stop_requested(),'USER_STOPPED prohibits recovery')
    engineering=purpose=='engineering_smoke'; m=p.validate_attempt(a,not engineering)
    if engineering:
        r=_engineering_recovery_review(a,review_path,case_key)
    else:
        r=p.check_review(review_path,'recovery',m['source_hash']); owner=p.owner_binding(a)
        p.require(r['pause_sha256']==p.sha_file(a/'TECHNICAL_PAUSE.json') and r['owner_endpoint_hash']==owner['endpoint_hash'], 'Recovery review pause/owner differs')
        p.require(bool(r['resource_profiles']),'Recovery review requires measured resource evidence')
    for receipt in r.get('resource_profiles',[]): p.receipt_json(receipt)
    status=p.request(a,'status')
    p.require(not any(_alive(x.get('worker_pid')) for x in status['jobs'].values()),'Wait for every registered live worker')
    if engineering:
        p.require(case_key in status['jobs'] and status['jobs'][case_key]['job']['purpose']==purpose,
                  'Reviewed engineering key missing')
    for key,x in status['jobs'].items():
        if x['job']['purpose']!=purpose or x['status']=='COMPLETE' or (engineering and key!=case_key): continue
        got=p.request(a,'get',key=key)
        if x['epoch']:
            p.require(got['state'] is not None,'Acknowledged RAM lost'); _verify_saved_ack(a,x['job'],got['state'])
        p.require(r['last_acks'].get(key)==x['epoch'],'Recovery review last ACK differs')
        if x['status'] in ('FAILED','RUNNING'):
            reply=p.request(a,'recover',key=key); p.require(reply['status']=='RECOVERY_READY','Recovery API differs')
    p.immutable(a/'recovery_reviews'/f'{p.sha_file(review_path)}.json',dict(review_path=str(Path(review_path).resolve()),
        review_sha256=p.sha_file(review_path),pause_sha256=p.sha_file(a/'TECHNICAL_PAUSE.json'),source_hash=m['source_hash'],
        design_bundle_sha256=p.DESIGN_HASH,purpose=purpose,case_key=case_key,test_accessed=False))
    (a/'PAUSE_TRAINING').unlink(missing_ok=True)
    return dict(status='RECOVERY_READY',test_accessed=False)


def _alive(pid):
    if pid is None: return False
    stat=Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z'


def _launch(args,out):
    p.new_path(out).parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    log=out.open('ab')
    proc=subprocess.Popen([p.PYTHON,'-B','-u','-m',MODULE,*args],cwd=p.ROOT,env=env,stdout=log,stderr=log)
    return proc,log


def _supervisor_exit(a,x,code):
    local_facts=[json.loads(line) for line in (a/'physical_attempts.jsonl').read_text().splitlines()]
    terminal=[r for r in local_facts if r.get('pid')==x['proc'].pid and r['status']=='TERMINAL_CORE']
    p.append_json(a/'physical_attempts.jsonl',dict(key=x['key'],case_id=x.get('case'),pid=x['proc'].pid,device=x['device'],
        status='SUPERVISOR_OBSERVED_EXIT',exit_code=code,signal=-code if code<0 else None,launch_time=x['launched_at'],observed_exit_time=time.time(),
        elapsed_wall_seconds=time.monotonic()-x['launch_monotonic'],seconds_semantics='process_wall_including_load_and_exit',
        returned_train_calls=terminal[-1]['returned_train_calls'] if terminal else None,
        unknown_partial_call=terminal[-1]['unknown_partial_call'] if terminal else None,
        worker_terminal_exists=bool(terminal),last_observed_owner=x.get('owner_facts'),last_observed_ack=x.get('last_ack'),
        absent_terminal_null_reason=None if terminal else 'worker_exit_before_local_finally',time=time.time(),test_accessed=False))


def controller_hardware(device):
    # CUDA driver device queries need no context or worker allocation.
    lib=ctypes.CDLL('libcuda.so.1'); dev=ctypes.c_int(); name=ctypes.create_string_buffer(256)
    major=ctypes.c_int(); minor=ctypes.c_int(); total=ctypes.c_size_t()
    def query(name,*args): p.require(getattr(lib,name)(*args)==0,'CUDA hardware query failed: '+name)
    query('cuInit',ctypes.c_uint(0)); query('cuDeviceGet',ctypes.byref(dev),ctypes.c_int(device))
    query('cuDeviceGetName',name,ctypes.c_int(len(name)),dev)
    query('cuDeviceComputeCapability',ctypes.byref(major),ctypes.byref(minor),dev)
    query('cuDeviceTotalMem_v2',ctypes.byref(total),dev)
    p.require(total.value>0,'Unknown CUDA-usable memory')
    return dict(name=name.value.decode(),compute_capability=[major.value,minor.value],cuda_usable_total_bytes=total.value)


def _schedule(a,manifest,jobs,devices,*,engineering,recovery_review=None):
    active={}; pending={}; wall=time.monotonic(); completed=set(); controller_logs={}; samplers={}; hardware={}
    for dev in devices:
        hardware[dev]=controller_hardware(dev)
        log=TelemetryLog(a/'telemetry'/f'controller_{os.getpid()}_gpu{dev}.jsonl.gz',
            dict(pid=os.getpid(),source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,purpose='controller',hardware=hardware[dev]))
        controller_logs[dev]=log
        samplers[dev]=NVML(dev,emit=log.emit,identity=manifest,phase=lambda:'admission')
    def sample(dev):
        row=samplers[dev].sample(); row['cuda_usable_total_bytes']=hardware[dev]['cuda_usable_total_bytes']; return row
    try:
        while True:
            status=p.request(a,'status'); own=status['jobs']
            for pid,x in list(active.items()):
                record=own.get(x['key']); x['owner_facts']=None if record is None else {k:record.get(k) for k in ('status','worker_pid','attempts','allocated','epoch','counts','last_commit')}
                ack_paths=sorted((a/'acks'/x['key']).glob('epoch_*.json'))
                x['last_ack']=_receipt(ack_paths[-1]) if ack_paths else None
                code=x['proc'].poll()
                if code is None: continue
                _supervisor_exit(a,x,code); x['log'].close(); del active[pid]
                held=pending.get(pid)
                if held:
                    p.append_json(a/'admission.jsonl',dict(event='DEAD_STARTUP_RESOLVED',pending=held,exit_code=code,
                        process_alive=_alive(pid),owner_record=x['owner_facts'],time=time.time()))
                    p.require(not _alive(pid),'Exited startup still alive'); pending.pop(pid)
                if code or (record and record['status']=='FAILED'):
                    (a/'PAUSE_TRAINING').touch()
                    if not (a/'TECHNICAL_PAUSE.json').exists():
                        p.write_json(a/'TECHNICAL_PAUSE.json',dict(key=x['key'],source_hash=manifest['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                            reason='SUPERVISOR_WORKER_EXIT',exit_code=code,time=time.time(),test_accessed=False))
                elif engineering: completed.add(x['key'])
            own=p.request(a,'status')['jobs']
            if any(r['status']=='FAILED' or (r['status']=='RUNNING' and r.get('worker_pid') not in active) for r in own.values()):
                (a/'PAUSE_TRAINING').touch()
            for held_pid,held in list(pending.items()):
                dev=held['device']; record=own.get(held['key'],{}); x=active.get(held['pid'])
                if pending_matches(held,record,alive=x is not None and x['proc'].poll() is None and _alive(held['pid'])):
                    # Verify this PID/attempt's explicit FULL allocation receipt in addition to status.
                    receipts=[]
                    path=a/'allocation_acks.jsonl'
                    if path.exists(): receipts=[json.loads(line) for line in path.read_text().splitlines()]
                    matched=[r for r in receipts if r['key']==held['key'] and r['attempt']==held['expected_attempt'] and
                        r['worker_pid']==held['pid'] and r['after_real_FULL'] and r['device']==dev]
                    if not matched: continue
                    current=sample(dev)
                    p.append_json(a/'admission.jsonl',dict(event='PENDING_ACK_RELEASE_RESAMPLE',pending=held,
                        sample_id=current['sample_id'],allocation_ack_id=p.digest(matched[-1]),time=time.time()))
                    pending.pop(held_pid)
                # Engineering re-registration resets allocation and is protected until call7 FULL.
            for pid,x in active.items():
                record=own.get(x['key'],{})
                if record.get('status')=='RUNNING' and record.get('worker_pid')==pid and not record.get('allocated'):
                    pending[pid]=dict(key=x['key'],pid=pid,device=x['device'],expected_attempt=record['attempts'])
            unfinished=[]
            for i,j in enumerate(jobs):
                if engineering:
                    done=bool(list((a/'smoke'/j['case_id']).glob('attempt*/case.json')))
                else: done=own.get(j['key'],{}).get('status')=='COMPLETE'
                if not done and j['key'] not in {x['key'] for x in active.values()}: unfinished.append((i,j))
            stopped=p.stop_requested(); paused=(a/'PAUSE_TRAINING').exists()
            if not unfinished and not active: break
            if (stopped or paused) and not active: break
            if not stopped and not paused:
                for dev in devices:
                    for i,j in unfinished:
                        if engineering and int(j['case_id'].split('_')[1][3:])!=dev: continue
                        if j['key'] in {x['key'] for x in active.values()}: continue
                        p.require(not p.stop_requested(),'USER_STOPPED before admission')
                        incoming,envelope=envelope_version(a,j['dataset'],dev,hardware[dev],manifest,engineering=engineering)
                        current=sample(dev)
                        decision=admission(current,incoming,pending=any(v['device']==dev for v in pending.values()))
                        if not decision['predicates']['fresh_known']:
                            current=sample(dev); decision=admission(current,incoming,pending=any(v['device']==dev for v in pending.values()))
                        row=admission_row(j,i,dev,decision,[v for v in pending.values() if v['device']==dev],envelope)
                        row['decision_id']=p.digest(row); p.append_json(a/'admission.jsonl',row)
                        if not decision['eligible']:
                            if any(x!='physical_fit' for x in decision['reasons']): break
                            continue
                        expected=own.get(j['key'],{}).get('attempts',0)+1
                        if engineering:
                            args=['_case','--attempt',str(a),'--case-id',j['case_id'],'--device',str(dev),'--dataset',j['dataset'],'--arm',j['arm']]
                            if recovery_review: args+=['--recovery-review',str(recovery_review)]
                            path=a/'smoke'/j['case_id']/'foreground.log'
                        else:
                            args=['_worker','--attempt',str(a),'--key',j['key'],'--device',str(dev)]; path=a/'runs'/j['key']/'foreground.log'
                        proc,stream=_launch(args,path)
                        active[proc.pid]=dict(proc=proc,log=stream,key=j['key'],job=j,device=dev,case=j.get('case_id'),
                            launched_at=time.time(),launch_monotonic=time.monotonic())
                        pending[proc.pid]=dict(key=j['key'],pid=proc.pid,device=dev,expected_attempt=expected)
                        current=sample(dev)
                        p.append_json(a/'admission.jsonl',dict(event='STARTUP_RESAMPLE',pending=pending[proc.pid],device=dev,
                            sample_id=current['sample_id'],time=time.time()))
                        break
            time.sleep(.5)
        result=dict(status='USER_STOPPED' if p.stop_requested() else ('PAUSED' if (a/'PAUSE_TRAINING').exists() else
            ('ENGINEERING_PENDING_INDEPENDENT_REVIEW' if engineering else 'FORMAL_COMPLETE')),
            foreground_wall_seconds=time.monotonic()-wall,test_accessed=False)
        p.append_json(a/'foreground_runs.jsonl',result); return result
    except BaseException as error:
        p.write_json(a/'TECHNICAL_PAUSE.json',dict(reason=type(error).__name__,message=str(error),source_hash=manifest['source_hash'],
            design_bundle_sha256=p.DESIGN_HASH,time=time.time(),test_accessed=False)); (a/'PAUSE_TRAINING').touch(); raise
    finally:
        # Supervise mechanical children until they have safely left their current atomic work.
        while active:
            for pid,x in list(active.items()):
                code=x['proc'].poll()
                if code is not None: _supervisor_exit(a,x,code); x['log'].close(); del active[pid]
            if active: time.sleep(.5)
        for log in controller_logs.values(): log.close()


def smoke(attempt=p.ATTEMPT,review_path=None,devices=(0,1),recovery=False,recovery_review=None,case_id=None):
    a=p.new_root(attempt); m=p.validate_attempt(a); p.check_review(review_path,'code',m['source_hash'])
    jobs=[p.job(d,arm,2022,m['source_hash'],'engineering_smoke',f'case{i+1}_gpu{dev}_{d}') for i,(dev,d,arm) in enumerate(CASES)]
    if recovery:
        j=next((j for j in jobs if j['case_id']==case_id),None); p.require(j is not None,'Unknown recovery case')
        recover(a,recovery_review,purpose='engineering_smoke',case_key=j['key']); jobs=[j]
    result=_schedule(a,m,jobs,devices,engineering=True,recovery_review=recovery_review)
    p.disk_observation(a,phase='after_engineering',remaining_transactions=500); return result


def run(attempt=p.ATTEMPT,review_path=None,devices=(0,1),recovery=False,recovery_review=None):
    a=p.new_root(attempt); m=p.validate_attempt(a,True); p.check_review(review_path,'engineering',m['source_hash'])
    if recovery: recover(a,recovery_review)
    p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'USER_STOPPED/PAUSE')
    jobs=[p.job(d,p.ARMS[0],seed,m['source_hash']) for d in ('kuairand','movielens') for seed in p.SEEDS]
    return _schedule(a,m,jobs,devices,engineering=False)


def handoff(attempt=p.ATTEMPT):
    a=p.new_root(attempt); m=p.validate_attempt(a,True); status=p.request(a,'status'); owner=p.owner_binding(a)
    rows=[]; total=0; useful={n:0 for n in p.OPERATIONS}; all_acks=[]; results=[]
    for d in p.DATASETS:
        for seed in p.SEEDS:
            for arm in p.ARMS:
                j=p.job(d,arm,seed,m['source_hash']); r=status['jobs'][j['key']]
                p.require(r['status']=='COMPLETE','Formal handoff requires every completed job')
                got=p.request(a,'get',key=j['key']); result=got['result']; s=got['state']
                p.require(s['selected_epochs']==result['selected_epochs'] and s['selected_validation']==result['selected_validation'], 'Selected compact state differs')
                hashes={k:p.state_digest(v) for k,v in s['selected_models'].items()}
                p.require(hashes==result['selected_hashes'],'Selected hashes differ')
                size=sum(t.numel()*t.element_size() for state in s['selected_models'].values() for t in state.values())
                total+=size; epochs=p.epoch_files(a,result); p.verify_selection(result,epochs)
                p.require(len(epochs)==50,'Handoff ACK coverage')
                for name in useful: useful[name]+=result['operations'][name]
                all_acks.extend(_receipt(a/'acks'/j['key']/f'epoch_{e:02d}.json') for e in range(1,51))
                results.append(_receipt(a/'runs'/j['key']/'result.json'))
                rows.append(dict(key=j['key'],job=j,selected_epochs=result['selected_epochs'],selected_hashes=hashes,
                                 selected_validation=result['selected_validation'],ram_get_verified=True,tensor_bytes=size))
    p.require(total>0 and len(rows)==10,'Actual20 selected model coverage differs')
    expected=dict(training_rows=701574750,task_entries=1403149500,ordinary_forwards=17250,scalar_backwards=17250,
        live_adam_steps=17250,live_ema_steps=17250,diagnostic_points=40,extra_autograd_sweeps=120,virtual_adam_steps=160,
        probe_forwards=0,view_forwards=0,algorithm_rng_draws=0)
    p.require(useful==expected and len(all_acks)==500,'Handoff useful work differs')
    from .tables import physical_cost_rows,_jsonl
    costs=physical_cost_rows(a); calls=[r for r in costs if r['cost_kind']=='returned_train_call']
    physical=dict(returned_train_calls=len(calls),training_rows=sum(r['B'] for r in calls),task_entries=sum(r['entries'] for r in calls),
        diagnostic_points=sum(r['diagnostic_points'] for r in calls),extra_autograd_sweeps=sum(r['extra_autograd_sweeps'] for r in calls),
        virtual_adam_steps=sum(r['virtual_adam_steps'] for r in calls))
    physical['unknown_partial_calls']=sum(r.get('current_pending_call') is not None for r in _jsonl(a/'physical_attempts.jsonl') if r['status']=='TERMINAL_CORE')
    physical['unreturned_operations']=None if physical['unknown_partial_calls'] else 0
    p.require(physical['returned_train_calls']>=17250 and physical['diagnostic_points']>=40,'Physical work undercounts useful work')
    table=_receipt(a/'tables/TABLE_RECEIPT.json'); controls=_receipt(a/'CONTROLS_VERIFIED.json')
    receipt=dict(status='READY_FOR_PARENT_SELECTED_RAM_ACCEPTANCE',source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        research_bundle_sha256=p.RESEARCH_HASH,external_runtime=p.read_json(a/'source_receipt.json')['external_runtime'],
        controls_index_hash=p.controls_hash(),environment=m['environment'],data=m['data'],selection_hash=m['selection_hash'],
        config_hashes=sorted({r['job']['config_hash'] for r in rows}),owner=owner,physical_root=str(a),logical_root=str(p.LOGICAL),
        code_review=_receipt(Path(m['code_review_path'])),resource_review=_receipt(a/'ENGINEERING_SIGNOFF.json'),
        recovery_reviews=[_receipt(path) for path in sorted((a/'recovery_reviews').glob('*.json'))],
        selected_pairs=rows,retained_tensor_bytes=total,committed_epochs=500,ack_receipts=all_acks,result_receipts=results,
        controls_receipt=controls,table_receipt=table,useful_operations=useful,physical_operations=physical,
        raw_resource_journals=[_receipt(a/name) for name in ('admission.jsonl','allocation_acks.jsonl','profiles/witness_registry.jsonl',
            'physical_attempts.jsonl','disk_observations.jsonl')],
        raw_only_runner_statement='Numeric results, source/state identities and operation/cost facts only; no research conclusions.',test_accessed=False)
    p.write_json(a/'RUNNER_HANDOFF.json',receipt); return receipt



def admission_row_check():
    # Required fields from controller_2930478_gpu0 SAMPLE; only fixture freshness is renewed.
    current=dict(sample_id='2930478:0:1:1',known=True,device=0,used_bytes=20985675776,
        nvml_total_bytes=25769803776,power_watts=192.963,power_limit_watts=350.,host_fraction=.5195776199158447,
        cuda_usable_total_bytes=25307578368,monotonic_end=time.monotonic())
    incoming=2027945984
    j=dict(purpose='engineering_smoke',key='fixture-startup')
    envelope=dict(path='fixture-envelope.json',sha256='fixture-envelope')
    held=dict(pid=33,key=j['key'],device=0,expected_attempt=1)
    for pending in ([],[held]):
        decision=admission(current,incoming,pending=bool(pending)); raw=copy.deepcopy(decision)
        row=admission_row(j,0,0,decision,pending,envelope)
        assert row['sample_id']==current['sample_id'] and all(row[k]==v for k,v in raw.items())
        assert decision==raw and row['eligible']==(not bool(pending))
        assert row['incoming_measured_full_process_bytes']==row['remaining_measured_bytes']==incoming
        assert row['already_resident_bytes']==0 and row['predicates']['no_pending']==(not bool(pending))
        assert row['envelope'] is envelope and row['pending_identity'] is pending
        assert row['purpose']==j['purpose'] and row['key']==j['key'] and row['queue_index']==row['device']==0
        assert set(row)==set(raw)|{'time','purpose','queue_index','key','device','pending_identity','envelope'}
    return dict(status='PASS',sample_id=current['sample_id'],admission_rows=2,raw_decision_preserved=True,
        incoming_envelope_pending_preserved=True,no_gpu_queries=True,test_accessed=False)


def finite_checks():
    startup_row=admission_row_check()
    from unittest.mock import patch
    from types import SimpleNamespace
    import sys
    module=sys.modules[__name__]
    # All telemetry is injected at the real named query boundary; no GPU query occurs.
    class Library:
        def __init__(self,failures): self.failures=failures; self.full_attempts=0
        def nvmlInit_v2(self): self.full_attempts+=1; return 0
        def nvmlDeviceGetHandleByIndex_v2(self,device,pointer): pointer._obj.value=17; return 0
        def nvmlDeviceGetMemoryInfo(self,handle,pointer):
            pointer._obj.total=100; pointer._obj.used=80; pointer._obj.free=20; return 0
        def nvmlDeviceGetPowerUsage(self,handle,pointer):
            if self.full_attempts<=self.failures: return 999
            pointer._obj.value=80000; return 0
        def nvmlDeviceGetEnforcedPowerLimit(self,handle,pointer): pointer._obj.value=100000; return 0
        def nvmlDeviceGetComputeRunningProcesses_v2(self,handle,count,rows):
            count._obj.value=1; rows[0].pid=os.getpid(); rows[0].usedGpuMemory=20; return 0
    rows=[]; lib=Library(1)
    telemetry=NVML(0,emit=rows.append,identity=dict(key='fixture',attempt=7),phase=lambda:'finite_cpu',lib=lib,host=lambda:.89,sleep=lambda seconds:None)
    current=telemetry.sample(); current['cuda_usable_total_bytes']=100
    assert lib.full_attempts==2 and current['sample_attempt']==2 and current['own_process_bytes']==20
    failed=[r for r in rows if r['event']=='QUERY' and r['return_code']]
    assert len(failed)==1 and QUERY_NAMES[failed[0]['query_id']]=='nvmlDeviceGetPowerUsage'
    assert failed[0]['phase']=='finite_cpu' and failed[0]['sample_attempt']==1
    bad=Library(3); rejected=[]
    try: NVML(1,emit=rejected.append,identity={},lib=bad,host=lambda:.1,sleep=lambda seconds:None).sample()
    except p.TechnicalPause: pass
    else: raise AssertionError('Persistent named failure admitted')
    assert bad.full_attempts==3 and len([r for r in rejected if r['event']=='SAMPLE_FAILED'])==3
    now=lambda **changes:dict(current,monotonic_end=time.monotonic(),**changes)
    assert admission(now(),20)['eligible'] and not admission(now(),21)['eligible']
    assert admission(now(used_bytes=84),16)['eligible']
    assert not admission(now(used_bytes=85),0)['eligible']
    assert not admission(now(power_watts=85),0)['eligible']
    assert not admission(now(host_fraction=.9),0)['eligible']
    assert not admission(dict(current,monotonic_end=time.monotonic()-2),0)['eligible']
    assert not admission(now(known=False),0)['eligible']
    assert not admission(now(),1,pending=True)['eligible']
    assert admission(now(),25,already_resident_bytes=5)['eligible']
    assert admission(now(device=1,used_bytes=0),90)['eligible']
    held=dict(pid=33,key='fixture',expected_attempt=2)
    record=dict(worker_pid=33,attempts=1,status='RUNNING',allocated=True,job=dict(key='fixture'))
    assert not pending_matches(held,record,alive=True)
    assert pending_matches(held,dict(record,attempts=2),alive=True)
    assert not pending_matches(held,dict(record,attempts=2),alive=False)
    # Fixed numeric prefix reproduces full-list conservative maxima without copied histories.
    hw=dict(name='fixture',compute_capability=[8,6],cuda_usable_total_bytes=100)
    profiles=[dict(dataset='movielens',arm=p.ARMS[0],device=i,source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
        environment={},hardware=hw,full_process_peak_bytes=10+i,reserved_peak_bytes=8+i,non_torch_peak_bytes=3,witness_id=str(i)) for i in (0,1)]
    assert measured_envelope(profiles,'movielens',0,hw,'fixture',{})[0]==12
    raw=b''.join((json.dumps(r,separators=(',',':'))+'\n').encode() for r in profiles)
    prefix=dict(bytes=len(raw),records=2,sha256=hashlib.sha256(raw).hexdigest())
    extended=raw+b'future_unbound_record\n'
    assert hashlib.sha256(extended[:prefix['bytes']]).hexdigest()==prefix['sha256']
    assert measured_envelope([json.loads(line) for line in extended[:prefix['bytes']].splitlines()],'movielens',1,hw,'fixture',{})[0]==12
    # Terminal core precedes both failing cleanup paths and preserves unreturned-work unknowns.
    journal=[]
    class BrokenProfile:
        closed=False; cuda_ready=False
        def close(self): raise RuntimeError('injected_sampler_close')
    local=LocalAttempt(Path('/tmp/c05-finite-virtual'),'fixture',0,'engineering_check','fixture')
    local.profile=BrokenProfile(); local.error=dict(type='ValueError',message='original_training_error',traceback='fixture')
    local.call_started(1,dict(fixture=True))
    def broken_owner(): raise OSError('injected_owner_status')
    with patch.object(p,'append_json',lambda path,row:journal.append(copy.deepcopy(row))): local.finish(owner_cleanup=broken_owner)
    assert journal[0]['status']=='TERMINAL_CORE' and journal[0]['original_exception']==local.error
    assert journal[0]['unreturned_call_operations'] is None and journal[0]['returned_train_calls']==0
    assert {e['cleanup'] for e in local.cleanup_errors}=={'profile_close','owner_status'}
    # Populated installed source Adam, RAM BytesIO and actual storage transcript.
    with patch.object(p,'controls_hash',lambda:'fixture-controls'):
        torch.set_num_threads(1); torch.manual_seed(2022)
        model=torch.nn.Linear(2,2); ema=copy.deepcopy(model).requires_grad_(False); opt=p.optimizer(model,dict(lr=.01,weight_decay=.0001))
        loader=torch.Generator().manual_seed(13222); x=torch.ones(2,2)
        opt.zero_grad(set_to_none=True); model(x).square().mean().backward(); opt.step()
        j=p.job('movielens',p.ARMS[0],2022,'fixture','engineering_smoke','fixture')
        state=initial_state(j,model,ema,opt,loader,'cpu',dict(endpoint_hash='fixture'))
        state.update(epoch=1,engineering_commit_ordinal=1,engineering_state_step=2,global_step=2,counts=dict(optimizer_updates=2,ema_updates=2))
        saved=p.cpu_copy(state); h=p.state_digest(saved); restored,cost=roundtrip(saved)
        assert p.state_digest(restored)==h and cost['bytes']>0
        restore(model,ema,opt,loader,restored,'cpu')
        for index,values in opt.state_dict()['state'].items():
            for key in ('step','exp_avg','exp_avg_sq'): assert values[key].data_ptr()!=saved['optimizer']['state'][index][key].data_ptr()
        expected=torch.randperm(17,generator=loader); restore(model,ema,opt,loader,saved,'cpu')
        assert torch.equal(expected,torch.randperm(17,generator=loader)) and p.state_digest(saved)==h
        record=dict(job_key=j['key'],epoch=1,global_step=2,counts=state['counts'],operations=state['operations'],source_hash='fixture',
            design_bundle_sha256=p.DESIGN_HASH,controls_index_hash='fixture-controls',owner_endpoint_hash='fixture',
            **p.component_hashes(state),continuation_hash=p.state_digest(state),scientific_identity_hash=p.scientific_identity(state),
            batch_log_sha256='fixture-batch',test_accessed=False)
        for lost in (False,True):
            calls=[]; clock=iter(range(10,30)); commits=[]
            def send(command,**kw):
                calls.append(command)
                if command=='commit':
                    commits.append(dict(key=j['key'],epoch=1,attempt=1,epoch_file='smoke/epoch.json',time=10.5,status='RAM_COMMITTED'))
                    if lost: raise EOFError('injected_lost_reply')
                    return dict(status='RAM_COMMITTED',epoch=1)
                if command=='get': return dict(state=p.cpu_copy(state),result=None)
                if command=='status': return dict(pid=99,retained_states=1,retained_bytes=1,jobs={j['key']:dict(epoch=1,job=j,counts=state['counts'],attempts=1)})
                raise AssertionError(command)
            ack=_verify_commit(j['key'],state,record,'smoke/epoch.json',1,send=send,journal=lambda:commits,epoch_sha='fixture',now=lambda:float(next(clock)))
            assert calls==['commit','status','get'] and ack['ram_get_verified'] and ack['transport_warning_reconciled']==lost
    import tempfile
    from .tables import diagnostic_rows,weights_rows
    class Fixture(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear=torch.nn.Linear(1,2)
        def forward(self,x): return self.linear(x[:-1].T).sigmoid()
    with tempfile.TemporaryDirectory(prefix='c05-lovelace-integration-') as directory:
        root=Path(directory)
        with patch.object(p,'ATTEMPT',root),patch.object(p,'new_path',lambda path:p.contained(path,root)):
            model=Fixture(); ema=copy.deepcopy(model).requires_grad_(False); config=p.config('movielens',p.ARMS[0]); opt=p.optimizer(model,config)
            loader=torch.Generator().manual_seed(13222)
            bx=torch.tensor([[.1,.2,.3,.4,.5,.6],[0.,0.,1.,1.,2.,2.]])
            by=torch.tensor([[0.,1.],[1.,0.],[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
            context=dict(purpose='engineering_check',clock_kind='engineering',dataset='movielens',seed=2022,case_id='finite',
                engineering_call_ordinal=1,engineering_state_step=1,sample_epoch=1,sample_batch=1,branch_role='runtime_integration',
                fixture_rows=6,pre_loader_hash=p.state_digest(loader.get_state()))
            value=train_step(model,opt,ema,bx,by,config,context,row_ids=np.arange(6,dtype=np.int64),diagnose=True)
            compact=normalize_value(root,value,model,context,'finite',1)
            assert isinstance(compact['parameter_activity'],list) and compact['diagnostics']['actual_virtual_exact']
            formal_context=dict(epoch=1,batch=1,global_step=1)
            batch=dict(compact,context=formal_context,attempt=1)
            flattened=diagnostic_rows(dict(key='finite',dataset='movielens',seed=2022),batch)
            assert flattened and len(weights_rows({},batch))==6
            assert all('parameter_names' not in row and 'virtual_post' not in row for row in flattened)
            # Actual registry ingestion and immutable prefix versions use a numeric raw witness.
            raw_path=root/'telemetry/finite.jsonl'; raw_path.parent.mkdir(); raw_path.write_text('{"event":"finite","used_bytes":20}\n')
            raw_ref=dict(path=str(raw_path),sha256=p.sha_file(raw_path),bytes=raw_path.stat().st_size,record_start=1,record_end=1)
            journal=root/'profiles/producer_1.jsonl'; journal.parent.mkdir()
            for i in (0,1):
                payload=dict(profiles[i],producer_pid=1,job_key='finite',attempt=1,purpose='engineering_smoke',
                    allocated_peak_bytes=3,host_rss_bytes=4,phase_coverage=['finite'],raw_log=raw_ref)
                payload.pop('witness_id'); p.append_json(journal,dict(witness_id=p.digest(payload),**payload))
            rows,prefix=collect_witnesses(root); assert len(rows)==2
            assert collect_witnesses(root)[1]==prefix
            with patch.object(module,'bootstrap_profiles',lambda hardware:dict(scope='historical_bootstrap',incoming_bytes=9)):
                peak,ref=envelope_version(root,'movielens',0,hw,dict(source_hash='fixture',environment={}),engineering=True)
                assert peak==12 and p.read_json(ref['path'])['registry_prefix']==prefix
    return dict(status='PASS',groups=7,named_query_retry_attempts=[2,3],terminal_cleanup_failures=2,
        stale_pending_rejected=True,ram_only_restore=True,storage_transcripts=2,actual_core_integration_calls=1,actual_core_diagnostic_points=1,actual_core_virtual_steps=4,
        numeric_registry_prefix_fixture=True,startup_row=startup_row,no_gpu_queries=True,test_accessed=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','smoke','run','handoff','_worker','_case'))
    parser.add_argument('--root',default=str(p.ROOT)); parser.add_argument('--attempt',default=str(p.ATTEMPT))
    parser.add_argument('--review'); parser.add_argument('--recovery-review'); parser.add_argument('--recover',action='store_true')
    parser.add_argument('--devices',default='0,1'); parser.add_argument('--device',type=int); parser.add_argument('--key')
    parser.add_argument('--case-id'); parser.add_argument('--dataset'); parser.add_argument('--arm')
    args=parser.parse_args(); p.require(Path(args.root)==p.ROOT,'ROOT differs'); a=p.new_root(args.attempt)
    if args.command=='prepare': result=p.prepare(a,args.review)
    elif args.command=='smoke': result=smoke(a,args.review,tuple(map(int,args.devices.split(','))),args.recover,args.recovery_review,args.case_id)
    elif args.command=='run': result=run(a,args.review,tuple(map(int,args.devices.split(','))),args.recover,args.recovery_review)
    elif args.command=='handoff': result=handoff(a)
    elif args.command=='_worker': worker(a,args.key,args.device); result=dict(status='WORKER_RETURNED')
    else: result=smoke_case(a,args.case_id,args.device,args.dataset,args.arm,args.recovery_review)
    # Handoff is numeric but large; print only its public status.
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected_pairs','validation')},allow_nan=False))


if __name__=='__main__':
    main()
