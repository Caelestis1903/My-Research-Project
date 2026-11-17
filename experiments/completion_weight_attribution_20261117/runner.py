"""Foreground-supervised Cycle04 mechanics; no AI lifecycle or disk tensor states."""
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from . import protocol as p
from .core import train_step
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step

MODULE = 'experiments.completion_weight_attribution_20261117.runner'
CASES = ((0,'movielens',p.ARMS[0]),(0,'kuairand',p.ARMS[1]),
         (1,'movielens',p.ARMS[1]),(1,'kuairand',p.ARMS[0]))


def host_fraction():
    m={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
       if line.startswith(('MemTotal:','MemAvailable:'))}
    return (m['MemTotal']-m['MemAvailable'])/m['MemTotal']


class _Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class _Process(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
              ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]


class NVML:
    """Read board telemetry and own process residency through installed NVML v2 ABI."""
    def __init__(self, device):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1'); self.device=int(device)
        self._ok(self.lib.nvmlInit_v2()); self.handle=ctypes.c_void_p()
        self._ok(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(self.device),ctypes.byref(self.handle)))

    @staticmethod
    def _ok(code):
        p.require(code==0, f'NVML measurement failed code={code}')

    def sample(self):
        mem=_Memory(); power=ctypes.c_uint(); limit=ctypes.c_uint()
        self._ok(self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(mem)))
        self._ok(self.lib.nvmlDeviceGetPowerUsage(self.handle,ctypes.byref(power)))
        self._ok(self.lib.nvmlDeviceGetEnforcedPowerLimit(self.handle,ctypes.byref(limit)))
        capacity=ctypes.c_uint(256); processes=(_Process*256)()
        self._ok(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,ctypes.byref(capacity),processes))
        own=sum(int(x.usedGpuMemory) for x in processes[:capacity.value] if x.pid==os.getpid())
        p.require(own < 2**63, 'NVML process memory unavailable')
        return dict(time=time.time(),device=self.device,used_bytes=int(mem.used),nvml_total_bytes=int(mem.total),
                    power_watts=power.value/1000,power_limit_watts=limit.value/1000,
                    host_fraction=host_fraction(),own_process_bytes=own)


class Profile:
    def __init__(self,device,j):
        self.device=int(device); self.job=j; self.nvml=NVML(device); self.rows=[]; self.phase='context_initialization'
        self.phase_summaries={}; self.peaks={}; self.lock=threading.RLock()
        self.error=None; self.done=threading.Event(); self.cuda_ready=False
        self.thread=threading.Thread(target=self._loop,daemon=False); self.thread.start()

    def initialize_cuda(self):
        # Only the worker's main thread creates the context, before model allocation.
        self._sample(force=True)
        torch.cuda.set_device(self.device); torch.cuda.init()
        with self.lock:
            self.cuda_ready=True
            self._sample(force=True)

    def _sample(self,force=False):
        with self.lock:
            return self._sample_locked(force)

    def _sample_locked(self,force=False):
        row=self.nvml.sample(); row['phase']=self.phase
        if self.cuda_ready:
            row.update(torch_allocated_bytes=torch.cuda.memory_allocated(self.device),
                       torch_reserved_bytes=torch.cuda.memory_reserved(self.device))
        else:
            row.update(torch_allocated_bytes=0,torch_reserved_bytes=0)
        row['non_torch_bytes']=max(0,row['own_process_bytes']-row['torch_reserved_bytes'])
        with self.lock:
            phase=self.phase_summaries.setdefault(row['phase'],dict(samples=0,first_time=row['time'],last_time=row['time']))
            phase['samples']+=1; phase['last_time']=row['time']; grew=False
            for name in ('own_process_bytes','torch_allocated_bytes','torch_reserved_bytes','non_torch_bytes'):
                phase[name]=max(phase.get(name,0),row[name])
                if name not in self.peaks or row[name]>self.peaks[name]:
                    self.peaks[name]=row[name]; grew=True
            if grew or force: self.rows.append(row)
        return row

    def _loop(self):
        while not self.done.is_set():
            try: self._sample()
            except Exception as error:
                self.error=str(error); return
            self.done.wait(.005)

    def mark(self,name):
        self.phase=name
        if self.cuda_ready:
            torch.cuda.synchronize(self.device)
        row=self._sample(force=True); p.require(self.error is None, self.error or 'Resource sampler failed')
        if self.cuda_ready:
            usable=torch.cuda.get_device_properties(self.device).total_memory
            p.require(row['used_bytes']<=usable, f'Calibration physical capacity exceeded at {name}')
        return row

    def close(self):
        self.done.set(); self.thread.join(); p.require(self.error is None,self.error or 'Resource sampler failed')

    def receipt(self):
        props=torch.cuda.get_device_properties(self.device)
        return dict(dataset=self.job['dataset'],arm=self.job['arm'],device=self.device,key=self.job['key'],
            source_hash=self.job['source_hash'],design_bundle_sha256=p.DESIGN_HASH,environment=p.environment(),
            hardware=dict(name=props.name,compute_capability=[props.major,props.minor],cuda_usable_total_bytes=props.total_memory),
            full_process_peak_bytes=max(r['own_process_bytes'] for r in self.rows),
            torch_allocated_peak_bytes=torch.cuda.max_memory_allocated(self.device),
            torch_reserved_peak_bytes=torch.cuda.max_memory_reserved(self.device),
            non_torch_bytes=max(r['non_torch_bytes'] for r in self.rows),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            phases=sorted(self.phase_summaries),phase_summaries=self.phase_summaries,samples=self.rows,test_accessed=False)


def admission(current, incoming_measured_full_process_bytes=0, *, pending=False, already_resident_bytes=0, cold=False):
    memory=current['used_bytes']/current['nvml_total_bytes']; power=current['power_watts']/current['power_limit_watts']
    remaining=max(0,incoming_measured_full_process_bytes-already_resident_bytes)
    fit=cold or current['used_bytes']+remaining<=current['cuda_usable_total_bytes']
    reasons=[]
    if memory>=.85: reasons.append('CURRENT_MEMORY_85')
    if power>=.85: reasons.append('CURRENT_POWER_85')
    if current['host_fraction']>=.90: reasons.append('CURRENT_HOST_90')
    if pending: reasons.append('PENDING_ALLOCATION_ACK')
    if not fit: reasons.append('CUDA_USABLE_PHYSICAL_FIT')
    return dict(eligible=not reasons,reasons=reasons,current=current,current_memory_fraction=memory,
                current_power_fraction=power,pending_ack=pending,cold_calibration=cold,
                incoming_measured_full_process_bytes=incoming_measured_full_process_bytes,
                already_resident_bytes=already_resident_bytes,remaining_measured_bytes=remaining,
                physical_fit=fit,test_accessed=False)


def measured_envelope(profiles,dataset,device,hardware,source_hash):
    compatible=[r for r in profiles if r['dataset']==dataset and r['source_hash']==source_hash and
                r['design_bundle_sha256']==p.DESIGN_HASH and r['hardware']==hardware]
    p.require({r['arm'] for r in compatible}==set(p.ARMS) and any(r['device']==device for r in compatible),
              'Missing compatible measured dataset/arm/target-card envelope')
    return max(max(r['full_process_peak_bytes'],r['torch_reserved_peak_bytes']+r['non_torch_bytes']) for r in compatible), compatible


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
        design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.CONTROLS_HASH,owner_endpoint_hash=snapshot['owner_endpoint_hash'],
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
    return dict(job=j,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,controls_index_hash=p.CONTROLS_HASH,
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
    for k in ('engineering_commit_ordinal','engineering_state_step','sample_epoch','sample_batches','selection_clock'):
        if k in snapshot: r[k]=snapshot[k]
    r.update(p.component_hashes(snapshot)); r.update(continuation_hash=p.state_digest(snapshot),
        scientific_identity_hash=p.scientific_identity(snapshot),validation=validation,validation_cost=validation_cost,
        batch_log=str(Path(batch_path).relative_to(p.ATTEMPT)),batch_log_sha256=p.sha_file(batch_path),
        batch_count=batch_count,epoch_order_hash=epoch_order_hash,seconds=seconds,test_accessed=False)
    return r


def allocation_ack(a,j,profile,number):
    row=profile.mark('first_FULL_lazy_Adam_allocation_ACK'); peak=max(
        max(r['own_process_bytes'] for r in profile.rows),torch.cuda.max_memory_reserved(profile.device)+max(r['non_torch_bytes'] for r in profile.rows))
    resource_observation(a,profile,number)
    reply=p.request(a,'allocated',key=j['key'],peak_memory_bytes=peak)
    p.require(reply==dict(status='ALLOCATED'), 'Allocation ACK differs')
    current=profile.nvml.sample(); current['cuda_usable_total_bytes']=torch.cuda.get_device_properties(profile.device).total_memory
    p.append_json(a/'allocation_acks.jsonl',dict(key=j['key'],attempt=number,worker_pid=os.getpid(),device=profile.device,time=time.time(),
        source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,peak_memory_bytes=peak,
        reply=reply,resampled_current=current,test_accessed=False))


def resource_observation(a,profile,number):
    props=torch.cuda.get_device_properties(profile.device)
    p.append_json(a/'resource_observations.jsonl',dict(dataset=profile.job['dataset'],arm=profile.job['arm'],
        key=profile.job['key'],device=profile.device,attempt=number,time=time.time(),
        source_hash=profile.job['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        hardware=dict(name=props.name,compute_capability=[props.major,props.minor],cuda_usable_total_bytes=props.total_memory),
        full_process_peak_bytes=max(r['own_process_bytes'] for r in profile.rows),
        torch_reserved_peak_bytes=torch.cuda.max_memory_reserved(profile.device),
        non_torch_bytes=max(r['non_torch_bytes'] for r in profile.rows),test_accessed=False))


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
    p.require(bool(receipts),'Actual engineering profiles missing')
    profiles=[p.receipt_json(r) for r in receipts]; prof=copy.deepcopy(profiles[-1])
    for r in profiles:
        p.require(all(r[k]==prof[k] for k in ('key','dataset','arm','device','source_hash','design_bundle_sha256','hardware','environment')) and
                  r['key']==j['key'] and r['source_hash']==j['source_hash'] and r['device']==int(device) and r['samples'],
                  'Engineering profile provenance differs')
    for field in ('full_process_peak_bytes','torch_allocated_peak_bytes','torch_reserved_peak_bytes','non_torch_bytes','peak_host_rss_bytes'):
        prof[field]=max(r[field] for r in profiles)
    prof['samples']=[dict(row,physical_profile=receipt) for r,receipt in zip(profiles,receipts) for row in r['samples']]
    prof['phases']=sorted({phase for r in profiles for phase in r['phases']})
    prof['phase_summaries']={}
    for phase in prof['phases']:
        observed=[r['phase_summaries'][phase] for r in profiles if phase in r['phase_summaries']]
        prof['phase_summaries'][phase]=dict(samples=sum(x['samples'] for x in observed),
            first_time=min(x['first_time'] for x in observed),last_time=max(x['last_time'] for x in observed),
            **{name:max(x[name] for x in observed) for name in ('own_process_bytes','torch_allocated_bytes','torch_reserved_bytes','non_torch_bytes')})
    prof['contributing_profiles']=receipts
    profile_path=out/'case_profile.json'; p.immutable(profile_path,prof)
    call_logs=[_receipt(path) for path in sorted((a/'smoke'/j['case_id']).glob('attempt*/batches.jsonl'))]
    counts=p.engineering_physical_counts(j,call_logs)
    p.require(all(counts[k]>=v for k,v in p.ENGINEERING_PLAN.items()),'Actual physical work cannot omit verified plan')
    result=dict(status='ENGINEERING_VERIFIED_PENDING_REVIEW',key=j['key'],job=j,device=int(device),dataset=j['dataset'],arm=j['arm'],
        **counts,physical_train_calls_semantics='returned_calls; unreturned partial work recorded separately',
        verified_plan_counts=dict(p.ENGINEERING_PLAN),engineering_state_step=4,engineering_commit_ordinal=2,scientific_epochs_completed=0,
        source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,prefix_hash=proof['prefix_hash'],
        suffix_scientific_identity_hash=proof['suffix_scientific_identity_hash'],source_lambda_zero_exact=True,diagnostics_neutral_exact=True,
        actual_gradients_activity_exact=True,prefix_immutable=True,validation=proof['validation'],validation_costs=proof['validation_costs'],
        restore_bytesio=proof['restore_bytesio'],proof=_receipt(Path(proof['path'])),profile=_receipt(profile_path),
        physical_call_logs=call_logs,physical_attempts_path=str(a/'physical_attempts.jsonl'),test_accessed=False)
    p.immutable(out/'case.json',result); return result


def smoke_case(attempt,case_id,device,dataset,arm,recovery_review=None):
    a=p.new_root(attempt); m=p.validate_attempt(a); p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'Stop/pause before engineering')
    p.require((int(device),dataset,arm) in CASES,'Unregistered engineering case')
    case_number=CASES.index((int(device),dataset,arm))+1
    p.require(case_id==f'case{case_number}_gpu{device}_{dataset}','Engineering output case identity differs')
    j=p.job(dataset,arm,2022,m['source_hash'],'engineering_smoke',case_id); p.immutable(a/'jobs'/f'{j["key"]}.json',j)
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
    profile=Profile(device,j); dev=f'cuda:{device}'; started=time.perf_counter(); physical_calls=0; physical=[]; log=None
    terminal='PAUSED'; profile_closed=False; call_pending=False; segment_started=started
    p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
        status='STARTED',resume_ack_epoch=saved['epoch'] if saved else 0,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
    try:
        bundle,_=p.dataset_cache(dataset); profile.mark('data_loaded'); profile.initialize_cuda()
        model,ema,opt,loader=_new_objects(j,bundle,dev); profile.mark('staged_model_EMA_lazy_Adam')
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
        samples=(order[:40960],order[-p.TAILS[dataset]:]); batch_path=out/'batches.jsonl'; log=batch_path.open('x')
        def call(ordinal,step,which,diagnose,role,delegate=False,lambda_zero=False):
            nonlocal physical_calls,call_pending
            p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'USER_STOPPED/PAUSE engineering atomic boundary')
            ids=samples[which]; x=torch.as_tensor(bundle.train_x[ids],device=dev).T; y=torch.as_tensor(bundle.train_y[ids],device=dev)
            context=_engineering_context(j,ordinal,step,1 if which==0 else p.STEPS[dataset],role,loader)
            profile.mark(f'call{ordinal}_{"FULL" if which==0 else "tail"}_{"diagnosed" if diagnose else "ordinary"}_before')
            tick=time.perf_counter(); c=j['config'] if not lambda_zero else dict(j['config'],mixture_lambda=0.)
            call_pending=True
            if delegate:
                loss=source_clean_step(model,opt,ema,x,y,c); v=None
            else:
                v=train_step(model,opt,ema,x,y,c,context,row_ids=bundle.train_row_ids[ids],diagnose=diagnose)
            torch.cuda.synchronize(dev); physical_calls+=1
            receipt=dict(key=j['key'],attempt=number,pid=os.getpid(),source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                call=ordinal,state_step=step,role=role,diagnose=diagnose,B=len(ids),seconds=time.perf_counter()-tick,
                row_order_hash=p.state_digest(np.asarray(bundle.train_row_ids[ids],dtype=np.int64)),
                observation=observation(model,ema,opt,loader,dev),context=context,test_accessed=False)
            if v is not None: receipt['value']=v
            else: receipt['source_loss']=loss
            physical.append(receipt); log.write(json.dumps(receipt,allow_nan=False)+'\n'); log.flush(); call_pending=False
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
            commit_snapshot(a,j['key'],prefix,r,out/'engineering_commit_01.json',attempt_number=number)
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
        v=call(6,4,1,True,'suffix_expected'); advance(state,v,len(samples[1])); expected_gradient=observation(model,ema,opt,loader,dev)
        profile.mark('expected_full_validation_online_EMA'); vals,cost=validation(model,ema,bundle,dev,state,2,True)
        for kind,value in cost.items(): record_cost('validation','expected_'+kind,value)
        state.update(epoch=2,engineering_commit_ordinal=2,engineering_state_step=4)
        expected=capture(state,model,ema,opt,loader,dev); expected_identity=p.scientific_identity(expected)
        record_cost('CPU_snapshot_capture','expected_suffix',dict(seconds=expected['costs']['snapshot_capture_seconds']))
        p.request(a,'paused',key=j['key'],reason='ENGINEERING_RESTORE_SUFFIX')
        p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
            status='PAUSED_FOR_VERIFIED_REPLAY',returned_train_calls=sum(x['attempt']==number for x in physical),unknown_partial_call=False,
            seconds=time.perf_counter()-segment_started,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
        second=register(a,j); p.require(second['attempt']==number+1,'Engineering re-register attempt differs'); number=second['attempt']
        segment_started=time.perf_counter()
        p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
            status='STARTED',resume_ack_epoch=1,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))
        get_started=time.perf_counter(); got=p.request(a,'get',key=j['key'])
        record_cost('RAM_get','prefix_replay',dict(seconds=time.perf_counter()-get_started))
        p.require(p.state_digest(got['state'])==prefix_hash and p.state_digest(second['state'])==prefix_hash,'Owner prefix changed')
        saved_copy,bytes_cost=roundtrip(got['state']); record_cost('BytesIO','prefix_replay',bytes_cost)
        profile.mark('BytesIO_deep_restore_RNG_last')
        load(saved_copy,'restored_suffix'); state=p.cpu_copy(saved_copy)
        v=call(7,3,0,False,'restored_suffix'); advance(state,v,len(samples[0]))
        v=call(8,4,1,False,'restored_suffix'); advance(state,v,len(samples[1]))
        p.require(observation(model,ema,opt,loader,dev)==expected_gradient,'Diagnose-on/off restored suffix gradients/state differs')
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
        commit_snapshot(a,j['key'],replay,r,out/'engineering_commit_02.json',attempt_number=number)
        profile.mark('suffix_commit_get_verified')
        p.request(a,'paused',key=j['key'],reason='ENGINEERING_VERIFIED'); terminal='ENGINEERING_VERIFIED'
        profile.close(); profile_closed=True; p.immutable(out/'resource_profile.json',profile.receipt())
        return _engineering_finish(a,j,replay,out,device)
    except BaseException as error:
        stopped=p.stop_requested(); terminal='USER_STOPPED' if stopped else 'PAUSED'
        if stopped:
            p.request(a,'paused',key=j['key'],reason='USER_STOPPED')
            return dict(status='USER_STOPPED',key=j['key'],physical_train_calls=physical_calls,test_accessed=False)
        p.write_json(a/'TECHNICAL_PAUSE.json',dict(key=j['key'],case_id=case_id,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                     reason=type(error).__name__,time=time.time(),test_accessed=False))
        (a/'PAUSE_TRAINING').touch(); p.request(a,'failed',key=j['key'],failure=dict(type=type(error).__name__,time=time.time()))
        raise
    finally:
        if log is not None: log.close()
        if not profile_closed:
            profile.close()
            if profile.cuda_ready: p.immutable(out/'resource_profile.json',profile.receipt())
        p.append_json(a/'physical_attempts.jsonl',dict(key=j['key'],purpose=j['purpose'],attempt=number,pid=os.getpid(),device=int(device),
            status=terminal,returned_train_calls=sum(x['attempt']==number for x in physical),unknown_partial_call=call_pending,
            partial_call_seconds=None,seconds=time.perf_counter()-segment_started,
            source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,test_accessed=False))


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
    a=p.new_root(attempt); manifest=p.validate_attempt(a,True); j=p.read_json(a/'jobs'/f'{key}.json')
    p.require(j['key']==key and j==p.job(j['dataset'],j['arm'],j['seed'],manifest['source_hash']), 'Formal job/path identity differs')
    p.require(j['purpose']=='formal' and not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'Worker stopped/paused')
    registration=register(a,j); number=registration['attempt']; saved=registration['state']
    out=a/'runs'/key/f'attempt{number}'; out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); returned=0; profile=Profile(device,j); dev=f'cuda:{device}'; state=None
    p.append_json(a/'physical_attempts.jsonl',dict(key=key,attempt=number,status='STARTED',pid=os.getpid(),device=int(device),
        resume_ack_epoch=saved['epoch'] if saved else 0,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        purpose='formal',time=time.time(),test_accessed=False))
    try:
        bundle,_=p.dataset_cache(j['dataset'],manifest['cache_dir']); profile.initialize_cuda()
        model,ema,opt,loader=_new_objects(j,bundle,dev)
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
            batch_path=out/f'batches_epoch_{epoch:02d}.jsonl.gz'; batch_count=0; sums={n:0. for n in ('loss','clean_loss','same_class_loss','completion_loss','scaled_loss')}
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
                    v=train_step(model,opt,ema,x,y,j['config'],context,row_ids=bundle.train_row_ids[ids],diagnose=diag)
                    advance(state,v,len(ids)); returned+=1; batch_count+=1
                    mass_add(mass_acc,v['ledger'],len(ids)); statistic_add(stat_acc,v['ledger'])
                    for cell,values in v['ledger']['cells'].items():
                        for name in ('n','n0','n1'): cell_counts[cell][name]+=values[name]
                    for n in sums: sums[n]+=v[n]*len(ids)
                    row=dict(key=key,attempt=number,job=j,context=context,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                        B=len(ids),entries=2*len(ids),counts=copy.deepcopy(state['counts']),operations=copy.deepcopy(state['operations']),**v,test_accessed=False)
                    stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'); stream.flush()
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
            record.update({n:value/len(order) for n,value in sums.items()})
            record['objective_ledger']=dict(batches=batch_count,rows=len(order),cells=cell_counts,
                mass_summaries=mass_rows({},mass_acc),diagnostic_statistics=statistic_rows({},stat_acc))
            profile.mark('CPU_BytesIO_commit_get'); commit_snapshot(a,key,snapshot,record,out/f'epoch_{epoch:02d}.json',attempt_number=number)
            resource_observation(a,profile,number)
            state=snapshot
        epochs=p.epoch_files(a,dict(job=j)); p.require(len(epochs)==50,'All50 ACKs required before source complete')
        last=epochs[50][1]; result=dict(job=j,epochs=50,global_step=state['global_step'],initial_hash=state['initial_hash'],
            selected_epochs=copy.deepcopy(state['selected_epochs']),selected_validation=copy.deepcopy(state['selected_validation']),
            selected_hashes=p.component_hashes(state)['selected_hashes'],common_validation=copy.deepcopy(state['common_validation']),
            final_hashes=dict(online=last['model_hash'],fixed_ema=last['ema_hash']),optimizer_hash=last['optimizer_hash'],
            counts=state['counts'],operations=state['operations'],source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
            controls_index_hash=p.CONTROLS_HASH,owner_endpoint_hash=state['owner_endpoint_hash'],worker_seconds=time.perf_counter()-started,
            epoch_receipts=[dict(epoch=e,path=str(path),sha256=p.sha_file(path)) for e,(path,r) in epochs.items()],test_accessed=False)
        p.verify_selection(result,epochs); _complete(a,j,result)
    except BaseException as error:
        p.write_json(a/'TECHNICAL_PAUSE.json',dict(key=key,source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
            reason=type(error).__name__,time=time.time(),test_accessed=False)); (a/'PAUSE_TRAINING').touch()
        p.request(a,'failed',key=key,failure=dict(type=type(error).__name__,time=time.time()))
        raise
    finally:
        profile.close()
        if profile.cuda_ready: p.write_json(out/'resource_profile.json',profile.receipt())
        status=p.request(a,'status')['jobs'][key]['status']
        p.append_json(a/'physical_attempts.jsonl',dict(key=key,attempt=number,purpose='formal',status=status,pid=os.getpid(),device=int(device),
            returned_train_calls=returned,unknown_partial_call=status=='FAILED',seconds=time.perf_counter()-started,
            source_hash=j['source_hash'],design_bundle_sha256=p.DESIGN_HASH,time=time.time(),test_accessed=False))


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
    proc=subprocess.Popen([p.PYTHON,'-u','-m',MODULE,*args],cwd=p.ROOT,env=env,stdout=log,stderr=log)
    return proc,log


def _supervisor_exit(a,x,code):
    p.append_json(a/'physical_attempts.jsonl',dict(key=x.get('key'),case_id=x.get('case'),pid=x['proc'].pid,
        device=x['device'],status='SUPERVISOR_OBSERVED_EXIT',exit_code=code,
        elapsed_wall_seconds=time.time()-x['launched_at'],seconds_semantics='process_wall_including_load_and_exit',
        returned_train_calls=None,unknown_partial_call=code!=0,time=time.time(),test_accessed=False))


def smoke(attempt=p.ATTEMPT,review_path=None,devices=(0,1),recovery=False,recovery_review=None,case_id=None):
    a=p.new_root(attempt); m=p.validate_attempt(a); p.check_review(review_path,'code',m['source_hash'])
    pending=list(enumerate(CASES)); active={}; completed=[]
    if recovery:
        match=next(((i,c) for i,c in pending if case_id==f'case{i+1}_gpu{c[0]}_{c[1]}'),None)
        p.require(match is not None,'Engineering recovery requires one registered --case-id')
        i,c=match; j=p.job(c[1],c[2],2022,m['source_hash'],'engineering_smoke',case_id)
        recover(a,recovery_review,purpose='engineering_smoke',case_key=j['key']); pending=[match]
    for i,c in enumerate(CASES):
        case=f'case{i+1}_gpu{c[0]}_{c[1]}'
        receipts=list((a/'smoke'/case).glob('attempt*/case.json'))
        if receipts:
            p.require(all(p.read_json(path)['source_hash']==m['source_hash'] for path in receipts),'Completed engineering source differs')
            completed.append(case)
            if (i,c) in pending: pending.remove((i,c))
    try:
        while pending or active:
            if p.stop_requested() or (a/'PAUSE_TRAINING').exists():
                if not active: break
            for device in devices:
                if any(x['device']==device for x in active.values()): continue
                match=next(((i,c) for i,c in pending if c[0]==device),None)
                if match is None or p.stop_requested() or (a/'PAUSE_TRAINING').exists(): continue
                i,c=match; current=NVML(device).sample(); decision=admission(current,cold=True)
                p.append_json(a/'admission.jsonl',dict(purpose='engineering_smoke',queue_index=i,decision=decision,time=time.time()))
                if not decision['eligible']: continue
                case=f'case{i+1}_gpu{device}_{c[1]}'; log=a/'smoke'/case/'foreground.log'
                args=['_case','--attempt',str(a),'--case-id',case,'--device',str(device),'--dataset',c[1],'--arm',c[2]]
                if recovery: args+=['--recovery-review',str(recovery_review)]
                proc,stream=_launch(args,log)
                active[proc.pid]=dict(proc=proc,log=stream,device=device,case=case,launched_at=time.time()); pending.remove(match)
            for pid,x in list(active.items()):
                code=x['proc'].poll()
                if code is None: continue
                _supervisor_exit(a,x,code)
                x['log'].close(); del active[pid]
                if code:
                    (a/'PAUSE_TRAINING').touch()
                    if not (a/'TECHNICAL_PAUSE.json').exists():
                        p.write_json(a/'TECHNICAL_PAUSE.json',dict(case_id=x['case'],source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                            reason='ENGINEERING_PROCESS_EXIT',exit_code=code,time=time.time(),test_accessed=False))
                else: completed.append(x['case'])
            if active or pending: time.sleep(.5)
            if (a/'PAUSE_TRAINING').exists() and not active: break
        return dict(status='USER_STOPPED' if p.stop_requested() else ('ENGINEERING_PENDING_INDEPENDENT_REVIEW' if len(completed)==4 else
            ('ENGINEERING_CASE_VERIFIED_PENDING_OTHER_CASES' if recovery and case_id in completed else 'PAUSED')),cases=completed,test_accessed=False)
    finally:
        for x in active.values():
            x['proc'].wait(); x['log'].close()


def _profiles(a):
    s=p.read_json(a/'ENGINEERING_SIGNOFF.json')
    return [p.receipt_json(r) for r in s['profiles']]


def run(attempt=p.ATTEMPT,review_path=None,devices=(0,1),recovery=False,recovery_review=None):
    a=p.new_root(attempt); m=p.validate_attempt(a,True); p.check_review(review_path,'engineering',m['source_hash'])
    if recovery: recover(a,recovery_review)
    p.require(not p.stop_requested() and not (a/'PAUSE_TRAINING').exists(),'USER_STOPPED/technical PAUSE')
    profiles=_profiles(a); active={}; pending_allocations={}; wall=time.perf_counter()
    jobs=[p.job(d,arm,s,m['source_hash']) for d in ('kuairand','movielens') for s in p.SEEDS for arm in p.ARMS]
    try:
        while True:
            status=p.request(a,'status'); own=status['jobs']
            failed=[k for k,r in own.items() if r['job']['purpose']=='formal' and r['status']=='FAILED']
            failed += [k for k,r in own.items() if r['job']['purpose']=='formal' and r['status']=='RUNNING' and
                       k not in {x['key'] for x in active.values()} and not _alive(r.get('worker_pid'))]
            if failed: (a/'PAUSE_TRAINING').touch()
            for pid,x in list(active.items()):
                code=x['proc'].poll()
                if code is None: continue
                _supervisor_exit(a,x,code)
                x['log'].close(); del active[pid]
                pending=pending_allocations.get(x['device'])
                if pending and pending['key']==x['key'] and pending['pid']==pid:
                    pending_allocations.pop(x['device'])
                if code:
                    (a/'PAUSE_TRAINING').touch()
                    if not (a/'TECHNICAL_PAUSE.json').exists():
                        p.write_json(a/'TECHNICAL_PAUSE.json',dict(key=x['key'],source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                            reason='WORKER_EXIT',exit_code=code,time=time.time(),test_accessed=False))
            for device,pending in list(pending_allocations.items()):
                x=active.get(pending['pid']); record=own.get(pending['key'],{})
                if (x is not None and x['key']==pending['key'] and x['device']==device and x['proc'].poll() is None and
                    record.get('job')==x['job'] and record.get('worker_pid')==pending['pid'] and
                    record.get('attempts')==pending['expected_attempt'] and record.get('status')=='RUNNING' and record.get('allocated') is True):
                    # Register resets allocated; require this launch's allocation ACK, never a stale flag.
                    current=NVML(device).sample()
                    p.append_json(a/'admission.jsonl',dict(event='PENDING_ACK_RELEASE_RESAMPLE',**pending,device=device,
                        worker_pid=record['worker_pid'],observed_attempt=record['attempts'],current=current,time=time.time(),
                        source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH))
                    pending_allocations.pop(device)
            unfinished=[(i,j) for i,j in enumerate(jobs) if own.get(j['key'],{}).get('status')!='COMPLETE' and
                        j['key'] not in {x['key'] for x in active.values()}]
            if not unfinished and not active: break
            stopped=p.stop_requested(); paused=(a/'PAUSE_TRAINING').exists()
            if (stopped or paused) and not active: break
            if not stopped and not paused:
                for device in devices:
                    props=torch.cuda.get_device_properties(device); current=NVML(device).sample()
                    current['cuda_usable_total_bytes']=props.total_memory
                    hardware=dict(name=props.name,compute_capability=[props.major,props.minor],cuda_usable_total_bytes=props.total_memory)
                    for i,j in unfinished:
                        if j['key'] in {x['key'] for x in active.values()}: continue
                        observations=[json.loads(line) for line in (a/'resource_observations.jsonl').read_text().splitlines()] if (a/'resource_observations.jsonl').exists() else []
                        peak,contributors=measured_envelope(profiles+observations,j['dataset'],device,hardware,m['source_hash'])
                        decision=admission(current,peak,pending=device in pending_allocations)
                        p.append_json(a/'admission.jsonl',dict(queue_index=i,key=j['key'],dataset=j['dataset'],arm=j['arm'],device=device,
                            source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,decision=decision,
                            measured_envelope_profiles=[dict(key=r['key'],arm=r['arm'],device=r['device'],profile_hash=p.digest(r)) for r in contributors],time=time.time()))
                        if not decision['eligible']:
                            if any(r!='CUDA_USABLE_PHYSICAL_FIT' for r in decision['reasons']): break
                            continue
                        expected_attempt=own.get(j['key'],{}).get('attempts',0)+1
                        proc,stream=_launch(['_worker','--attempt',str(a),'--key',j['key'],'--device',str(device)],a/'runs'/j['key']/'foreground.log')
                        active[proc.pid]=dict(proc=proc,log=stream,key=j['key'],job=j,device=device,launched_at=time.time())
                        pending_allocations[device]=dict(key=j['key'],pid=proc.pid,expected_attempt=expected_attempt)
                        # Resample after each startup; pending protects the unallocated process, no fixed cap.
                        p.append_json(a/'admission.jsonl',dict(event='STARTUP_RESAMPLE',**pending_allocations[device],device=device,
                            current=NVML(device).sample(),time=time.time(),source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH))
                        break
            time.sleep(.5)
        result=dict(status='USER_STOPPED' if p.stop_requested() else ('PAUSED' if (a/'PAUSE_TRAINING').exists() else 'FORMAL_COMPLETE'),
                    foreground_wall_seconds=time.perf_counter()-wall,test_accessed=False)
        p.append_json(a/'foreground_runs.jsonl',result); return result
    finally:
        for x in active.values(): x['proc'].wait(); x['log'].close()


def handoff(attempt=p.ATTEMPT):
    a=p.new_root(attempt); m=p.validate_attempt(a,True); status=p.request(a,'status'); owner=p.owner_binding(a)
    rows=[]; total=0
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
                rows.append(dict(key=j['key'],job=j,selected_epochs=result['selected_epochs'],selected_hashes=hashes,
                                 selected_validation=result['selected_validation'],ram_get_verified=True,tensor_bytes=size))
    p.require(total==191017920,'Actual40 selected state bytes differ')
    receipt=dict(status='READY_FOR_PARENT_SELECTED_RAM_ACCEPTANCE',source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
                 owner=owner,selected_pairs=rows,retained_tensor_bytes=total,committed_epochs=1000,test_accessed=False)
    p.write_json(a/'RUNNER_HANDOFF.json',receipt); return receipt


def finite_checks():
    current=dict(time=1.,device=0,used_bytes=80,nvml_total_bytes=100,power_watts=80,power_limit_watts=100,host_fraction=.89,cuda_usable_total_bytes=100)
    assert admission(current,20)['eligible']
    assert not admission(current,21)['eligible']
    assert admission({**current,'used_bytes':84},16)['eligible']
    assert not admission({**current,'used_bytes':85},1)['eligible']
    assert not admission({**current,'power_watts':85},1)['eligible']
    assert not admission({**current,'host_fraction':.9},1)['eligible']
    assert not admission(current,1,pending=True)['eligible']
    assert admission(current,25,already_resident_bytes=5)['eligible']
    assert admission({**current,'device':1,'used_bytes':0},90)['eligible']
    hardware=dict(name='fixture',compute_capability=[8,6],cuda_usable_total_bytes=100)
    profiles=[dict(dataset='movielens',arm=arm,device=i,source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,hardware=hardware,
                   full_process_peak_bytes=10+i,torch_reserved_peak_bytes=8+i,non_torch_bytes=3) for i,arm in enumerate(p.ARMS)]
    assert measured_envelope(profiles,'movielens',0,hardware,'fixture')[0]==12
    # Populated CPU Adam restore with mutable-state storage isolation and next permutation equality.
    torch.set_num_threads(1); model=torch.nn.Linear(2,2); ema=copy.deepcopy(model).requires_grad_(False)
    c=dict(lr=.01,weight_decay=.0001); opt=p.optimizer(model,c); loader=torch.Generator().manual_seed(13222)
    x=torch.ones(2,2); y=torch.zeros(2,2); opt.zero_grad(set_to_none=True); (model(x)-y).square().mean().backward(); opt.step()
    j=p.job('movielens',p.ARMS[0],2022,'fixture','engineering_smoke','fixture')
    s=initial_state(j,model,ema,opt,loader,'cpu',dict(endpoint_hash='fixture'))
    s.update(epoch=1,engineering_commit_ordinal=1,engineering_state_step=2,global_step=2,counts=dict(optimizer_updates=2,ema_updates=2))
    saved=p.cpu_copy(s); h=p.state_digest(saved); restore(model,ema,opt,loader,saved,'cpu')
    for n,t in opt.state_dict()['state'].items():
        for key in ('step','exp_avg','exp_avg_sq'):
            assert t[key].data_ptr()!=saved['optimizer']['state'][n][key].data_ptr()
    expected=torch.randperm(17,generator=loader); restore(model,ema,opt,loader,saved,'cpu')
    assert torch.equal(expected,torch.randperm(17,generator=loader)) and p.state_digest(saved)==h
    moved,byte_cost=roundtrip(saved); assert p.state_digest(moved)==h and byte_cost['bytes']>0
    # Actual serve transcript; commit reply may be lost but get/status/journal prove its exact payload.
    record=dict(job=j,epoch=1,global_step=2,counts=s['counts'],operations=s['operations'],source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
                controls_index_hash=p.CONTROLS_HASH,owner_endpoint_hash='fixture',**p.component_hashes(s),continuation_hash=p.state_digest(s),
                scientific_identity_hash=p.scientific_identity(s),batch_log_sha256='batch',test_accessed=False)
    for lost in (False,True):
        calls=[]; clock=iter(range(10,30)); journal=[]
        def send(command,**kw):
            calls.append(command)
            if command=='commit':
                journal.append(dict(key=j['key'],epoch=1,attempt=1,epoch_file='smoke/epoch.json',time=10.5,status='RAM_COMMITTED'))
                if lost: raise EOFError()
                return dict(status='RAM_COMMITTED',epoch=1)
            if command=='get': return dict(state=p.cpu_copy(s),result=None)
            if command=='status': return dict(pid=99,retained_states=1,retained_bytes=1,jobs={j['key']:dict(epoch=1,job=j,counts=s['counts'],attempts=1)})
            raise AssertionError(command)
        ack=_verify_commit(j['key'],s,record,'smoke/epoch.json',1,send=send,journal=lambda:journal,epoch_sha='epoch',now=lambda:float(next(clock)))
        assert calls==['commit','status','get'] and ack['ram_get_verified'] and ack['transport_warning_reconciled']==lost
        assert ack['events']['hash_verified']>=ack['events']['get_received']
    original=s['source_hash']; s['source_hash']='wrong'
    try: p.validate_epoch_state(s,record)
    except p.TechnicalPause: pass
    else: raise AssertionError('Source mismatch accepted')
    s['source_hash']=original
    corrupted=p.cpu_copy(s); corrupted['epoch']=2
    try: p.validate_epoch_state(corrupted,record)
    except p.TechnicalPause: pass
    else: raise AssertionError('Skipped engineering commit accepted')
    # One finite integration call uses the actual core API and actual diagnostic schema.
    class Fixture(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear=torch.nn.Linear(1,2)
        def forward(self,x): return self.linear(x[:-1].T).sigmoid()
    model=Fixture(); ema=copy.deepcopy(model).requires_grad_(False); config=p.config('movielens',p.ARMS[0])
    opt=p.optimizer(model,config); loader=torch.Generator().manual_seed(13222)
    state=initial_state(j,model,ema,opt,loader,'cpu',dict(endpoint_hash='fixture'))
    bx=torch.tensor([[.1,.2,.3,.4,.5,.6],[0.,0.,1.,1.,2.,2.]])
    by=torch.tensor([[0.,1.],[1.,0.],[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
    ctx=dict(purpose='engineering_check',clock_kind='engineering',dataset='movielens',seed=2022,
        case_id='fixture',engineering_call_ordinal=1,engineering_state_step=1,sample_epoch=1,sample_batch=1,
        branch_role='runtime_integration',fixture_rows=6,pre_loader_hash=p.state_digest(loader.get_state()))
    value=train_step(model,opt,ema,bx,by,config,ctx,row_ids=np.arange(6,dtype=np.int64),diagnose=True)
    advance(state,value,6); assert state['counts']==dict(optimizer_updates=1,ema_updates=1)
    from .tables import diagnostic_rows,weights_rows
    # Tables receives a formal-shaped numeric context; the finite training call above keeps its engineering clock.
    batch=dict(value,context=dict(epoch=1,batch=1,global_step=1),attempt=1)
    flattened=diagnostic_rows(dict(key='fixture',dataset='movielens',seed=2022),batch)
    assert flattened and len(weights_rows({},batch))==6
    assert all(r['counterfactual_at_host_state'] for r in flattened)
    # Staged context/model/EMA telemetry uses the real sampler and envelope arithmetic.
    from unittest.mock import patch
    from contextlib import ExitStack
    from pathlib import PurePosixPath
    from types import SimpleNamespace
    import hashlib
    import sys
    telemetry=dict(own=0,reserved=0)
    class Telemetry:
        def __init__(self,device): self.device=int(device)
        def sample(self):
            return dict(time=time.time(),device=self.device,used_bytes=telemetry['own'],nvml_total_bytes=10000,
                power_watts=10,power_limit_watts=100,host_fraction=.1,own_process_bytes=telemetry['own'])
    props=SimpleNamespace(name='mocked_CPU_fixture',major=8,minor=6,total_memory=10000)
    def initialize():
        assert threading.current_thread() is threading.main_thread()
        telemetry.update(own=400,reserved=0)
    cuda_mocks=dict(init=initialize,set_device=lambda device:None,synchronize=lambda device:None,
        memory_allocated=lambda device:telemetry['reserved'],memory_reserved=lambda device:telemetry['reserved'],
        max_memory_allocated=lambda device:telemetry['reserved'],max_memory_reserved=lambda device:telemetry['reserved'],
        get_device_properties=lambda device:props)
    module=sys.modules[__name__]
    with patch.object(module,'NVML',Telemetry),patch.multiple(torch.cuda,**cuda_mocks):
        sampler=Profile(0,j)
        try:
            sampler.initialize_cuda(); telemetry.update(own=900,reserved=500)
            row=sampler.mark('model_allocated')
            assert (row['own_process_bytes'],row['torch_reserved_bytes'],row['non_torch_bytes'])==(900,500,400)
            receipts=[dict(dataset='movielens',arm=arm,device=0,source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
                hardware=hardware,full_process_peak_bytes=900,torch_reserved_peak_bytes=500,non_torch_bytes=400) for arm in p.ARMS]
            assert measured_envelope(receipts,'movielens',0,hardware,'fixture')[0]==900
            telemetry.update(own=1100,reserved=700); sampler.mark('EMA_allocated')
            assert sampler.peaks['non_torch_bytes']==400 and sampler.peaks['own_process_bytes']==1100
        finally: sampler.close()
    # All filesystem effects of the actual runtime entrypoints below stay in strings/BytesIO.
    actual_environment=p.environment()
    files={}; dirs=set()
    class MemoryPath(PurePosixPath):
        def resolve(self): return self
        def exists(self): return str(self) in files or str(self) in dirs
        def is_file(self): return str(self) in files
        def mkdir(self,**kw): dirs.add(str(self))
        def touch(self): files.setdefault(str(self),'')
        def unlink(self,missing_ok=False):
            if not missing_ok: assert self.exists()
            files.pop(str(self),None)
        def read_text(self): return files[str(self)]
        def write_text(self,value): files[str(self)]=value; return len(value)
        def glob(self,pattern):
            n=len(PurePosixPath(pattern).parts)
            return iter(MemoryPath(k) for k in sorted(files) if MemoryPath(k).is_relative_to(self) and
                len(MemoryPath(k).relative_to(self).parts)==n and MemoryPath(k).relative_to(self).match(pattern))
        def open(self,mode='r',**kw):
            name=str(self); assert not ('x' in mode and name in files)
            class Stream(io.StringIO):
                def flush(self):
                    if any(c in mode for c in 'wax'): files[name]=self.getvalue()
                def close(self):
                    if not self.closed: self.flush()
                    super().close()
            stream=Stream(files.get(name,'') if 'a' in mode or 'r' in mode else '')
            if 'a' in mode: stream.seek(0,2)
            return stream
    root=MemoryPath(p.ATTEMPT); real_rng=p.rng_state; real_restore_rng=p.restore_rng
    real_as_tensor=torch.as_tensor; real_immutable=p.immutable; real_mark=Profile.mark
    owner=dict(endpoint_hash='fixture-owner'); records={}; sent=[]; injection={}
    def sha(path): return hashlib.sha256(files[str(path)].encode()).hexdigest()
    def boundary(path):
        path=MemoryPath(path); assert path.is_relative_to(root); return path
    def manifest(attempt,require_engineering=False):
        assert MemoryPath(attempt)==root
        if injection.get('engineering'): assert not require_engineering, 'Signoff must not be required during engineering recovery'
        return dict(source_hash='fixture',cache_dir='unused',code_review_path=str(root/'CODE_REVIEW.json'))
    def send(attempt,command,**kw):
        assert MemoryPath(attempt)==root; sent.append(command); key=kw.get('key')
        if command=='status':
            return dict(pid=99,jobs={k:{n:v for n,v in r.items() if n!='state'} for k,r in records.items()})
        if command=='register':
            r=records.setdefault(key,dict(job=kw['job'],epoch=0,attempts=0,status='REGISTERED',worker_pid=None))
            assert r['job']==kw['job'] and r['status'] in ('REGISTERED','PAUSED') and r['worker_pid'] is None
            r.update(attempts=r['attempts']+1,status='RUNNING',worker_pid=kw['pid'],allocated=False)
            return dict(attempt=r['attempts'],state=p.cpu_copy(r.get('state')))
        r=records[key]
        if command=='get': return dict(state=p.cpu_copy(r.get('state')),result=None)
        if command=='commit':
            if injection.pop('before_commit',False): raise OSError('finite uncommitted transport failure')
            state=kw['state']; assert state['epoch']==r['epoch']+1
            p.validate_epoch_state(state,p.read_json(root/kw['epoch_file']))
            r.update(state=p.cpu_copy(state),epoch=state['epoch'],counts=state['counts'])
            p.append_json(root/'epoch_commit_ledger.jsonl',dict(key=key,epoch=r['epoch'],attempt=r['attempts'],
                epoch_file=kw['epoch_file'],time=time.time(),status='RAM_COMMITTED'))
            return dict(status='RAM_COMMITTED',epoch=r['epoch'])
        if command=='allocated': r['allocated']=True; return dict(status='ALLOCATED')
        if command in ('paused','failed'):
            r.update(status=command.upper(),worker_pid=None); return dict(status=r['status'])
        if command=='recover':
            assert r['status'] in ('FAILED','RUNNING') and not any(x.get('worker_pid') for x in records.values())
            r.update(status='REGISTERED',worker_pid=None); return dict(status='RECOVERY_READY',epoch=r['epoch'])
        raise AssertionError('Unsupported/fake engineering API: '+command)
    def immutable(path,value):
        path=MemoryPath(path)
        if ('acks' in path.parts and injection.pop('lost_local_ack',False)) or (path.name=='case.json' and injection.pop('suffix_receipt',False)):
            raise OSError('finite lost receipt after RAM commitment')
        return real_immutable(path,value)
    def mark(profile,name):
        result=real_mark(profile,name)
        if name=='call6_after' and injection.pop('prefix_failure',False): raise RuntimeError('finite failure while retaining prefix ACK')
        return result
    def setup_files():
        files.clear(); dirs.clear(); records.clear(); sent.clear(); injection.clear(); telemetry.update(own=0,reserved=0)
        p.immutable(root/'CODE_REVIEW.json',dict(verdict='PASS',blocking_issues=[],source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
            coder_ids=['core-coder','runtime-coder','repair-coder'],reviewer_id='code-reviewer',test_accessed=False))
        p.immutable(root/'run_manifest.json',manifest(root))
    def recovery_review(key,engineering=True):
        r=records[key]
        p.write_json(root/'TECHNICAL_PAUSE.json',dict(key=key,source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,reason='FINITE_FAILURE'))
        (root/'PAUSE_TRAINING').touch()
        review=dict(verdict='PASS',blocking_issues=[],reviewer_id='fresh-recovery-reviewer',source_hash='fixture',
            design_bundle_sha256=p.DESIGN_HASH,owner_endpoint_hash=owner['endpoint_hash'],pause_sha256=sha(root/'TECHNICAL_PAUSE.json'),
            last_acks={key:r['epoch']},resource_profiles=[],test_accessed=False)
        if engineering: review.update(scope='engineering',case_keys=[key])
        else:
            p.immutable(root/'finite_resource.json',dict(finite=True)); review['resource_profiles']=[_receipt(root/'finite_resource.json')]
        p.write_json(root/'RECOVERY_REVIEW.json',review); return root/'RECOVERY_REVIEW.json'
    with ExitStack() as stack:
        for obj,name,value in ((module,'Path',MemoryPath),(p,'Path',MemoryPath),(p,'ATTEMPT',root),
            (p,'new_root',lambda attempt=root:root if MemoryPath(attempt)==root else boundary('/wrong')),
            (p,'new_path',boundary),(p,'load_reference_index',lambda:dict(entries=[])),(p,'sha_file',sha),(p,'validate_attempt',manifest),(p,'stop_requested',lambda:False),
            (p,'owner_binding',lambda attempt=root:owner),(p,'environment',lambda:actual_environment),(p,'request',send),(p,'immutable',immutable),
            (p,'rng_state',lambda device=None:real_rng('cpu')),(p,'restore_rng',lambda state,device=None:real_restore_rng(state,'cpu')),
            (module,'NVML',Telemetry),(Profile,'mark',mark)):
            stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.multiple(torch.cuda,**cuda_mocks))
        # Actual run() with one finite owner/launch transcript; no subprocess, IPC or GPU.
        setup_files(); scheduler_cases={}; scheduler_clock=dict(step=0); launches=[]; chronology=[]
        queue=[p.job(d,arm,seed,'fixture') for d in ('kuairand','movielens') for seed in p.SEEDS for arm in p.ARMS]
        first,other,next_job=queue[:3]
        review=root/'ENGINEERING_REVIEW.json'
        p.immutable(review,dict(verdict='PASS',blocking_issues=[],source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
            reviewer_id='resource-reviewer',test_accessed=False))
        scheduler_profiles=[dict(dataset=d,arm=arm,device=device,source_hash='fixture',design_bundle_sha256=p.DESIGN_HASH,
            hardware=dict(name=props.name,compute_capability=[props.major,props.minor],cuda_usable_total_bytes=props.total_memory),
            full_process_peak_bytes=100,torch_reserved_peak_bytes=80,non_torch_bytes=20,key=f'{d}/{arm}/{device}')
            for d in p.DATASETS for arm in p.ARMS for device in (0,1)]
        class SchedulerTelemetry(Telemetry):
            def sample(self):
                chronology.append(('sample',self.device,scheduler_clock['step']))
                return super().sample()
        def scheduler_status(attempt,command,**kw):
            assert attempt==root and command=='status' and not kw
            step=scheduler_clock['step']
            own={j['key']:dict(job=j,status='COMPLETE',attempts=1,worker_pid=None) for j in queue[3:]}
            if step!=1:
                row=dict(job=first,status='RUNNING',attempts=2,worker_pid=1001,allocated=True)
                if step==0: row.update(status='PAUSED',attempts=1,worker_pid=None)
                elif step==2: row.update(attempts=1,worker_pid=999)
                elif step==3: row['attempts']=3
                elif step==4: row['worker_pid']=999
                elif step==5: row['allocated']=False
                own[first['key']]=row
            if step>=1: own[other['key']]=dict(job=other,status='RUNNING',attempts=1,worker_pid=1002,allocated=False)
            if step>=7: own[next_job['key']]=dict(job=next_job,status='RUNNING',attempts=1,worker_pid=1003,allocated=True)
            return dict(jobs=own)
        def scheduler_launch(args,out):
            key=args[args.index('--key')+1]; device=int(args[args.index('--device')+1]); step=scheduler_clock['step']
            chronology.append(('launch',device,key,step)); launches.append(dict(key=key,device=device,step=step))
            code=lambda: (9 if key==next_job['key'] and scheduler_clock['step']>=7 else
                          (0 if scheduler_clock['step']>=8 else None))
            return SimpleNamespace(pid=1000+len(launches),poll=code,wait=lambda:0),io.StringIO()
        real_append=p.append_json
        def scheduler_append(path,value):
            if value.get('event')=='PENDING_ACK_RELEASE_RESAMPLE': chronology.append(('release',value['device'],value['key']))
            real_append(path,value)
        retained_cases=('absent_status','stale_pid_and_attempt','right_pid_wrong_attempt','wrong_pid_right_attempt','allocated_false')
        def scheduler_tick(seconds):
            assert seconds==.5
            step=scheduler_clock['step']; rows=[json.loads(line) for line in (root/'admission.jsonl').read_text().splitlines()]
            released=[r for r in rows if r.get('event')=='PENDING_ACK_RELEASE_RESAMPLE']
            if step==0:
                assert [(r['key'],r['device']) for r in launches]==[(first['key'],0),(other['key'],1)]
                assert [r['expected_attempt'] for r in rows if r.get('event')=='STARTUP_RESAMPLE']==[2,1]
                scheduler_cases['independent_gpu1']=dict(status='PASS',launch_devices=[0,1],new_key_registration=1)
            elif 1<=step<=5:
                assert len(launches)==2 and not released
                decision=next(r['decision'] for r in reversed(rows) if r.get('device')==0 and 'decision' in r)
                assert decision['reasons']==['PENDING_ALLOCATION_ACK']
                scheduler_cases[retained_cases[step-1]]=dict(status='PASS',pending_retained=True)
            elif step==6:
                assert len(released)==1 and released[0]['key']==first['key']
                assert (released[0]['pid'],released[0]['worker_pid'],released[0]['expected_attempt'],released[0]['observed_attempt'])==(1001,1001,2,2)
                assert launches[-1]==dict(key=next_job['key'],device=0,step=6)
                release_index=chronology.index(('release',0,first['key'])); launch_index=chronology.index(('launch',0,next_job['key'],6))
                assert chronology[release_index-1]==('sample',0,6) and ('sample',0,6) in chronology[release_index+1:launch_index]
                startup=next(r for r in rows if r.get('event')=='STARTUP_RESAMPLE' and r['key']==next_job['key'])
                assert startup['expected_attempt']==1
                scheduler_cases['qualified_release']=dict(status='PASS',release_count=1,resampled_before_next_launch=True,new_key_registration=startup['expected_attempt'])
            elif step==7:
                assert (root/'PAUSE_TRAINING').exists() and p.read_json(root/'TECHNICAL_PAUSE.json')['exit_code']==9
                assert len(released)==1 and released[0]['key']!=next_job['key'] and len(launches)==3
                assert any(json.loads(line).get('exit_code')==9 for line in (root/'physical_attempts.jsonl').read_text().splitlines())
                scheduler_cases['nonzero_exit']=dict(status='PASS',exit_code=9,allocation_release=False)
            else: raise AssertionError('Unexpected scheduler iteration')
            scheduler_clock['step']+=1
        with patch.object(module,'_profiles',lambda attempt:scheduler_profiles),patch.object(module,'_launch',scheduler_launch),\
             patch.object(module,'NVML',SchedulerTelemetry),patch.object(p,'request',scheduler_status),\
             patch.object(p,'append_json',scheduler_append),patch.object(time,'sleep',scheduler_tick):
            scheduler_result=run(root,review)
        assert scheduler_result['status']=='PAUSED' and len(scheduler_cases)==8
        # Failed physical epoch -> real recovery entrypoint -> next attempt -> canonical ACK.
        for purpose in ('formal','engineering_smoke'):
            setup_files(); ej=p.job('movielens',p.ARMS[0],2022,'fixture',purpose,'case1_gpu0_movielens' if purpose!='formal' else None)
            p.immutable(root/'jobs'/f'{ej['key']}.json',ej); reg=register(root,ej)
            snapshot=p.cpu_copy(s); snapshot.update(job=ej,source_hash='fixture',owner_endpoint_hash=owner['endpoint_hash'])
            if purpose=='formal':
                for name in ('engineering_commit_ordinal','engineering_state_step'): snapshot.pop(name,None)
                k=p.STEPS['movielens']; snapshot.update(purpose='formal',clock_kind='formal',epoch=1,global_step=k,next_epoch=2,
                    scientific_epochs_completed=1,counts=dict(optimizer_updates=k,ema_updates=k),operations=dict(training_rows=p.ROWS['movielens'],
                    task_entries=2*p.ROWS['movielens'],ordinary_forwards=k,scalar_backwards=k,live_adam_steps=k,live_ema_steps=k,
                    diagnostic_points=1,extra_autograd_sweeps=3,virtual_adam_steps=4,probe_forwards=0,view_forwards=0,algorithm_rng_draws=0))
            path=root/'finite_batches.jsonl'; path.write_text('{}\n')
            rec=epoch_record(snapshot,validation={},validation_cost={},batch_path=path,batch_count=2,epoch_order_hash='fixture')
            injection['before_commit']=True
            try: commit_snapshot(root,ej['key'],snapshot,rec,root/'runs'/'attempt1'/'epoch_01.json',attempt_number=1)
            except p.TechnicalPause: pass
            else: raise AssertionError('Uncommitted candidate accepted')
            failed=dict(files); send(root,'failed',key=ej['key'])
            review=recovery_review(ej['key'],purpose!='formal')
            recover(root,review,purpose=purpose,case_key=ej['key'] if purpose!='formal' else None)
            reg=register(root,ej); snapshot['costs']['finite_replay_seconds']=.125
            rec=epoch_record(snapshot,validation={},validation_cost={},batch_path=path,batch_count=2,epoch_order_hash='fixture')
            ack=commit_snapshot(root,ej['key'],snapshot,rec,root/'runs'/'attempt2'/'epoch_01.json',attempt_number=reg['attempt'])
            assert ack['attempt']==2 and 'attempt2' in ack['transaction_file']
            failed_record=p.read_json(root/'runs'/'attempt1'/'epoch_01.json')
            assert failed_record['scientific_identity_hash']==rec['scientific_identity_hash'] and failed_record['continuation_hash']!=rec['continuation_hash']
            old=root/'transactions'/ej['key']/'attempt1'/'epoch_01.json'
            assert old.read_text()==failed[str(old)] and len(list((root/'transactions'/ej['key']).glob('attempt*/epoch_*.json')))==2
            # Lose only the local ACK after an actual second epoch commit, then reconcile without a resend.
            snapshot=p.cpu_copy(snapshot)
            if purpose=='formal':
                snapshot.update(epoch=2,global_step=40,next_epoch=3,scientific_epochs_completed=2,counts=dict(optimizer_updates=40,ema_updates=40))
                for name in p.OPERATIONS:
                    if name not in p.DIAG_OPERATIONS: snapshot['operations'][name]*=2
            else: snapshot.update(epoch=2,global_step=4,engineering_commit_ordinal=2,engineering_state_step=4,counts=dict(optimizer_updates=4,ema_updates=4))
            rec=epoch_record(snapshot,validation={},validation_cost={},batch_path=path,batch_count=2,epoch_order_hash='fixture')
            injection['lost_local_ack']=True
            try: commit_snapshot(root,ej['key'],snapshot,rec,root/'runs'/'attempt2'/'epoch_02.json',attempt_number=2)
            except OSError: pass
            else: raise AssertionError('Lost local ACK fixture did not fail')
            commits=sent.count('commit'); wrong=p.cpu_copy(records[ej['key']]['state']); wrong['costs']['changed_actual_payload']=True
            try: reconcile_ack(root,ej['key'],wrong)
            except p.TechnicalPause: pass
            else: raise AssertionError('Lost ACK accepted a different full continuation')
            ack=reconcile_ack(root,ej['key'],p.cpu_copy(records[ej['key']]['state']))
            assert sent.count('commit')==commits and ack['ram_get_verified'] and ack['transport_warning_reconciled']
            accepted=p.epoch_files(root,dict(job=ej))
            assert list(accepted)==[1,2] and 'attempt2' in str(accepted[1][0])
        # The actual engineering entrypoint runs the finite real-core model on CPU, with real proof comparisons.
        class RuntimeFixture(torch.nn.Module):
            def __init__(self): super().__init__(); self.linear=torch.nn.Linear(1,2)
            def forward(self,x): return self.linear(x[:-1].T).sigmoid()
        rows=p.ROWS['movielens']; ids=np.arange(rows,dtype=np.int64)
        bundle=SimpleNamespace(train_x=np.column_stack((ids%17/17,ids%3)).astype(np.float32),
            train_y=np.column_stack((ids%2,(ids//3)%2)).astype(np.float32),train_row_ids=ids,
            val_x=np.column_stack((np.arange(120)%17/17,np.arange(120)%3)).astype(np.float32),
            val_y=np.column_stack((np.arange(120)%2,(np.arange(120)//3)%2)).astype(np.float32))
        def objects(j,bundle,device):
            p.setup(j['seed'],'cpu'); model=RuntimeFixture(); ema=copy.deepcopy(model).requires_grad_(False)
            telemetry.update(own=900,reserved=500)
            return model,ema,p.optimizer(model,j['config']),torch.Generator().manual_seed(j['seed']+11000)
        def tensor(data,*args,**kw):
            if str(kw.get('device','')).startswith('cuda'): kw['device']='cpu'
            return real_as_tensor(data,*args,**kw)
        stack.enter_context(patch.object(p,'dataset_cache',lambda *args:(bundle,{})))
        stack.enter_context(patch.object(module,'_new_objects',objects))
        stack.enter_context(patch.object(torch,'as_tensor',tensor))
        case='case1_gpu0_movielens'; outcomes=[]
        for failure in ('prefix_failure','suffix_receipt'):
            setup_files(); injection.update(engineering=True,**{failure:True})
            try: smoke_case(root,case,0,'movielens',p.ARMS[0])
            except (RuntimeError,OSError): pass
            else: raise AssertionError('Engineering negative fixture did not fail')
            ej=next(iter(records.values()))['job']; key=ej['key']; retained=p.state_digest(records[key]['state'])
            review=recovery_review(key)
            # Exact fresh review boundary: stale pause, foreign owner and source fail before recovery mutation.
            valid=p.read_json(review)
            for field in ('pause_sha256','owner_endpoint_hash','source_hash','design_bundle_sha256'):
                p.write_json(review,{**valid,field:'wrong'})
                try: recover(root,review,purpose='engineering_smoke',case_key=key)
                except p.TechnicalPause: pass
                else: raise AssertionError('Unbound recovery accepted: '+field)
                assert p.state_digest(records[key]['state'])==retained
            p.write_json(review,valid); recover(root,review,purpose='engineering_smoke',case_key=key)
            before=sum(len(MemoryPath(path).read_text().splitlines()) for path in files if MemoryPath(path).name=='batches.jsonl')
            telemetry.update(own=0,reserved=0)  # Recovery runs in a fresh worker process.
            result=smoke_case(root,case,0,'movielens',p.ARMS[0],review)
            assert p.receipt_json(result['profile'])['non_torch_bytes']==400
            assert result['verified_plan_counts']==p.ENGINEERING_PLAN and result['scientific_epochs_completed']==0
            assert result['physical_train_calls']==(12 if failure=='prefix_failure' else 8)
            assert result['diagnostic_points']==(3 if failure=='prefix_failure' else 2)
            assert 'complete' not in sent and 'release' not in sent and len(records)==1
            assert not (root/'ENGINEERING_SIGNOFF.json').exists()
            if failure=='suffix_receipt': assert before==result['physical_train_calls'] and records[key]['epoch']==2
            proof=p.receipt_json(result['proof']); original=MemoryPath(proof['path']).read_text()
            MemoryPath(proof['path']).write_text(original+' ')
            try: _engineering_proof(root,ej,records[key]['state'])
            except p.TechnicalPause: pass
            else: raise AssertionError('Changed/missing engineering proof accepted')
            MemoryPath(proof['path']).write_text(original)
            restore_costs=[json.loads(line) for line in (root/'costs.jsonl').read_text().splitlines() if json.loads(line)['cost_kind']=='restore']
            assert restore_costs and all(x['cost_kind']=='restore' and x['seconds']>=0 and x['attempt']>0 for x in restore_costs)
            from . import tables
            with patch.object(tables,'open',lambda path,mode:MemoryPath(path).open(mode),create=True):
                costs=tables.physical_cost_rows(root)
                exported=[r for r in costs if r['cost_kind']=='diagnostic']
                assert sum(r['extra_autograd_sweeps'] for r in exported)==result['extra_autograd_sweeps']
                assert len([r for r in costs if r['cost_kind']=='restore'])==len(restore_costs)
                assert len({r['attempt'] for r in exported})>1
                incomplete=root/'smoke'/case/'attempt99'/'batches.jsonl'; incomplete.write_text('{')
                partial=tables.physical_cost_rows(root)
                assert any(r.get('null_reason')=='incomplete_physical_batch_log' and r['seconds'] is None for r in partial)
                incomplete.unlink()
            outcomes.append(dict(failure=failure,returned_physical_calls=result['physical_train_calls'],diagnostic_points=result['diagnostic_points'],
                extra_autograd_sweeps=result['extra_autograd_sweeps'],virtual_adam_steps=result['virtual_adam_steps'],
                recovered_added_calls=result['physical_train_calls']-before,restore_records=len(restore_costs),
                profiles=len(p.receipt_json(result['profile'])['contributing_profiles'])))
    return dict(status='PASS',groups=11,pending_generation=scheduler_cases,storage_transcripts=2,actual_commit_recovery_cases=4,
                staged_telemetry=dict(own=900,reserved=500,non_torch=400,envelope=900),engineering_recovery=outcomes,
                core_integration_calls=1,integration_diagnostic_rows=len(flattened),
                counts_keys=['optimizer_updates','ema_updates'],tensor_storage='CPU RAM/BytesIO only',test_accessed=False)



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
