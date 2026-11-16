"""Foreground Cycle03 mechanical workers; the parent owns the frozen RAM server."""
import argparse
import copy
import ctypes
import gzip
import io
import json
import os
import resource
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from .core import train_step, schedule
from .protocol import (ROOT, ATTEMPT, DEFAULT_CACHE, ARMS, DATASETS, SEEDS, KINDS, CELLS,
    PYTHON, PROTECTED, config, job, digest, immutable, sha_file, read_json, design_bundle,
    cycle_path, local_path, stop_requested, validate_attempt, prepare, dataset_cache,
    load_protocol, make_model, setup, optimizer, cpu_copy, state_digest, rng_state,
    restore_rng, evaluate, write_json, request, alive, commits, validate_epoch_state,
    assert_old_unchanged, epoch_files, verify_selection, empty_phase_counters, validate_formal_clock,
    prefix_check, PHASE_FIELDS)
from experiments.natural_quadratic_ranking_20261116.core import train_step as frozen_train_step

MODULE = 'experiments.nqr_temporal_exposure_20261116.runner'
BOOTSTRAP = {'movielens': 1384120320, 'kuairand': 1658847232}


def numeric(path, value):
    path = local_path(ATTEMPT, path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    write_json(temp, value); temp.replace(path)


def append(path, value):
    path = local_path(ATTEMPT, path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')


def admission(row, host_fraction, incoming_bytes, usable_bytes, pending=False):
    """Current utilization and physical 100% capacity are separate predicates."""
    assert incoming_bytes > 0 and usable_bytes > 0 and row['total_bytes'] > 0 and row['power_limit'] > 0
    reasons = []
    if row['used_bytes']/row['total_bytes'] >= .85: reasons.append('CURRENT_MEMORY')
    if row['power']/row['power_limit'] >= .85: reasons.append('CURRENT_POWER')
    if host_fraction >= .9: reasons.append('CURRENT_HOST_RAM')
    if pending: reasons.append('PENDING_ALLOCATION_ACK')
    if row['used_bytes']+incoming_bytes > usable_bytes: reasons.append('PHYSICAL_CAPACITY')
    return dict(eligible=not reasons, reasons=reasons, current=row, host_fraction=host_fraction,
                incoming_full_process_bytes=incoming_bytes, cuda_usable_bytes=usable_bytes,
                pending_allocation_ack=pending)


def runtime_admission_checks():
    mib = 1024**2
    row = dict(index=0, used_bytes=19118*mib, total_bytes=24576*mib, power=280., power_limit=350.)
    capacity = 25307578368
    assert admission(row,.6,4306*mib,capacity)['eligible']
    assert not admission(row,.6,5284*mib,capacity)['eligible']
    assert (19118+4306)/24576 > .85
    assert admission(dict(row,used_bytes=capacity-4306*mib),.6,4306*mib,capacity)['eligible']
    for change,host,reason in (({'used_bytes':.85*row['total_bytes']},.6,'CURRENT_MEMORY'),
                               ({'power':.85*350},.6,'CURRENT_POWER'),({},.9,'CURRENT_HOST_RAM')):
        assert reason in admission(dict(row,**change),host,1,capacity)['reasons']
    assert admission(row,.899,4306*mib,capacity)['eligible']
    assert admission(row,.6,4306*mib,capacity,True)['reasons'] == ['PENDING_ALLOCATION_ACK']
    other = dict(row,index=1,used_bytes=1000*mib)
    assert admission(other,.6,5284*mib,capacity)['eligible']
    assert not admission(dict(row,power=350),.6,1,capacity)['eligible']
    assert admission(row,.6,1,capacity)['eligible']
    p = dict(full_process_peak_bytes=100,torch_reserved_peak_bytes=90,non_torch_bytes=20)
    assert incoming_profile(p) == 110
    assert incoming_profile(dict(p,full_process_peak_bytes=130)) == 130
    assert incoming_profile(dict(p,torch_reserved_peak_bytes=120)) == 140
    return dict(status='PASS', fixtures=13, gpu_queries=0, ipc_calls=0,
                semantics='independent_current85_host90_physical100_pending_ack_resample')


def host_ram():
    values = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 1-values['MemAvailable']/values['MemTotal']


class NVML:
    """Read-only NVML bytes and per-PID residency; constructed only during RUN."""
    class Memory(ctypes.Structure):
        _fields_ = [('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]

    class Process(ctypes.Structure):
        _fields_ = [('pid',ctypes.c_uint),('used',ctypes.c_ulonglong),
                    ('gpu_instance',ctypes.c_uint),('compute_instance',ctypes.c_uint)]

    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1'); self.call('nvmlInit_v2')

    def call(self, name, *args):
        code = getattr(self.lib,name)(*args)
        if code: raise RuntimeError(f'{name}: NVML status {code}')

    def handle(self, gpu):
        h = ctypes.c_void_p(); self.call('nvmlDeviceGetHandleByIndex_v2',ctypes.c_uint(gpu),ctypes.byref(h)); return h

    def card(self, gpu):
        h = self.handle(gpu); m = self.Memory(); power = ctypes.c_uint(); limit = ctypes.c_uint()
        self.call('nvmlDeviceGetMemoryInfo',h,ctypes.byref(m))
        self.call('nvmlDeviceGetPowerUsage',h,ctypes.byref(power))
        self.call('nvmlDeviceGetEnforcedPowerLimit',h,ctypes.byref(limit))
        return dict(index=gpu,used_bytes=m.used,total_bytes=m.total,power=power.value/1000,power_limit=limit.value/1000)

    def pid_bytes(self, gpu, pid):
        n = ctypes.c_uint(256); rows = (self.Process*256)()
        self.call('nvmlDeviceGetComputeRunningProcesses_v2',self.handle(gpu),ctypes.byref(n),rows)
        return sum(r.used for r in rows[:n.value] if r.pid == pid)


def device_identity(gpu):
    device = torch.device(f'cuda:{gpu}'); p = torch.cuda.get_device_properties(device)
    _, usable = torch.cuda.mem_get_info(device)
    return dict(name=p.name,major=p.major,minor=p.minor,cuda_usable_bytes=usable)


def incoming_profile(profile):
    return max(profile['full_process_peak_bytes'],profile['torch_reserved_peak_bytes']+profile['non_torch_bytes'])


class Profile:
    def __init__(self, gpu, path, j):
        self.gpu = gpu; self.device = torch.device(f'cuda:{gpu}'); self.path = path
        self.nvml = NVML(); self.phase = 'allocation'; self.stop = threading.Event(); self.error = None
        self.lock = threading.Lock()
        self.value = dict(key=j['key'],dataset=j['dataset'],arm=j['arm'],device=gpu,pid=os.getpid(),
            source_hash=j['source_hash'],design_bundle_sha256=j['design_bundle_sha256'],
            hardware=device_identity(gpu),full_process_peak_bytes=0,torch_reserved_peak_bytes=0,
            torch_allocated_peak_bytes=0,non_torch_bytes=0,context_measured=False,
            host_peak_fraction=0.,peak_host_rss_bytes=0,phases={},measured=True)
        self.thread = threading.Thread(target=self.monitor)

    def sample(self):
        full = self.nvml.pid_bytes(self.gpu,os.getpid())
        reserved = torch.cuda.memory_reserved(self.device); allocated = torch.cuda.memory_allocated(self.device)
        row = dict(time=time.time(),full_process_bytes=full,torch_reserved_bytes=reserved,
                   torch_allocated_bytes=allocated,host_ram_fraction=host_ram(),
                   host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   current_card=self.nvml.card(self.gpu),cuda_usable_bytes=self.value['hardware']['cuda_usable_bytes'])
        with self.lock:
            v = self.value
            v['full_process_peak_bytes'] = max(v['full_process_peak_bytes'],full)
            v['torch_reserved_peak_bytes'] = max(v['torch_reserved_peak_bytes'],torch.cuda.max_memory_reserved(self.device))
            v['torch_allocated_peak_bytes'] = max(v['torch_allocated_peak_bytes'],torch.cuda.max_memory_allocated(self.device))
            if full > 0 and reserved > 0 and full >= reserved:
                overhead = full-reserved
                v['non_torch_bytes'] = max(v['non_torch_bytes'] if v['context_measured'] else 0,overhead)
                v['context_measured'] = True
            v['host_peak_fraction'] = max(v['host_peak_fraction'],row['host_ram_fraction'])
            v['peak_host_rss_bytes'] = max(v['peak_host_rss_bytes'],row['host_rss_bytes'])
            phase = v['phases'].setdefault(self.phase,dict(samples=0,full_process_peak_bytes=0,torch_reserved_peak_bytes=0,torch_allocated_peak_bytes=0))
            phase['samples'] += 1
            for key,val in (('full_process_peak_bytes',full),('torch_reserved_peak_bytes',reserved),('torch_allocated_peak_bytes',allocated)):
                phase[key] = max(phase[key],val)
            phase.setdefault('first_sample_time',row['time']); phase['last_sample_time']=row['time']
            phase['host_peak_fraction']=max(phase.get('host_peak_fraction',0.),row['host_ram_fraction'])
            phase['peak_host_rss_bytes']=max(phase.get('peak_host_rss_bytes',0),row['host_rss_bytes'])
            phase['non_torch_bytes']=max(phase.get('non_torch_bytes',0),max(0,full-reserved))
            phase['latest']=row
            v['latest'] = row; v['incoming_bytes'] = incoming_profile(v)
            append(self.path.with_name('resource_samples.jsonl'),dict(key=v['key'],pid=v['pid'],phase=self.phase,
                source_hash=v['source_hash'],design_bundle_sha256=v['design_bundle_sha256'],**row))
            numeric(self.path,v)

    def monitor(self):
        while not self.stop.wait(.05):
            try: self.sample()
            except BaseException as error: self.error = error; return

    def start(self):
        self.sample(); self.thread.start(); return self

    def mark(self, phase):
        if self.error: raise RuntimeError('Residency profiling failed') from self.error
        self.phase = phase; self.sample()

    def close(self):
        self.stop.set(); self.thread.join(); self.sample()
        if self.error: raise RuntimeError('Residency profiling failed') from self.error
        return copy.deepcopy(self.value)


def require_storage(attempt, manifest):
    attempt = cycle_path(attempt); owner = read_json(attempt/'STORAGE_OWNER.json'); info = read_json(attempt/'guardian.json')
    server = ROOT/'experiments/task_view_gradient_residual_momentum_20261116/runner.py'
    assert owner['root'] == str(attempt) and owner['pid'] == info['pid'] and owner['socket'] == info['address']
    assert owner['session_id'] and owner['server_sha256'] == sha_file(server)
    assert owner['source_hash'] == manifest['source_hash'] and owner['design_bundle_sha256'] == manifest['design_bundle_sha256']
    assert owner['endpoint_hash'] == digest([str(attempt),info['pid'],info['address']])
    assert (attempt/'guardian.json').stat().st_mode & 0o777 == 0o600
    assert alive(info['pid']) and request(attempt,'status')['pid'] == info['pid']
    return {k:owner[k] for k in ('root','pid','socket','session_id','server_sha256','source_hash','design_bundle_sha256','endpoint_hash')}


def ack_receipt(attempt, key, ack, record, transfer=None):
    path = Path(ack['epoch_file']); target = path.with_name(f'ack_{ack["epoch"]:02d}.json')
    value = dict(key=key,attempt=ack['attempt'],epoch=ack['epoch'],epoch_file=str(path),
        epoch_file_sha256=sha_file(path),continuation_hash=record['continuation_hash'],
        source_hash=record['job']['source_hash'],design_bundle_sha256=record['job']['design_bundle_sha256'],
        owner_endpoint_hash=read_json(attempt/'STORAGE_OWNER.json')['endpoint_hash'],test_accessed=False)
    if target.exists():
        old = read_json(target); assert all(old[k] == v for k,v in value.items()); return old
    value['transfer'] = transfer
    immutable(target,value); return value


def validate_continuation(attempt, key, state, owner_record):
    rows = commits(attempt,key)
    if owner_record['epoch'] == 0:
        assert state is None and not rows
        return None
    assert rows and rows[-1]['epoch'] == owner_record['epoch']
    assert state is not None and state['job'] == owner_record['job']
    assert state['epoch'] == owner_record['epoch'] and state['counts'] == owner_record['counts']
    record = read_json(rows[-1]['epoch_file']); validate_epoch_state(state,record)
    return ack_receipt(attempt,key,rows[-1],record)


def commit_snapshot(attempt, key, snapshot, record, epoch_file, profile=None):
    validate_epoch_state(snapshot,record)
    immutable(epoch_file,record)
    if profile: profile.mark('cpu_snapshot_serialization')
    tick = time.perf_counter(); buffer = io.BytesIO(); torch.save(snapshot,buffer)
    transfer = dict(serialization_seconds=time.perf_counter()-tick,serialization_bytes=buffer.getbuffer().nbytes)
    buffer.close(); tick = time.perf_counter(); warning = None
    if profile: profile.mark('ram_roundtrip')
    try:
        answer = request(attempt,'commit',key=key,state=snapshot,epoch_file=str(epoch_file))
        assert answer == dict(status='RAM_COMMITTED',epoch=snapshot['epoch'])
    except (OSError,EOFError,ConnectionError) as error:
        # A lost reply is reconciled against retained RAM and the journal, never blindly resent.
        warning = repr(error)
    owner = request(attempt,'status')['jobs'][key]
    actual = request(attempt,'get',key=key)['state']
    assert owner['epoch'] == snapshot['epoch'], 'Uncommitted transaction: PAUSE at prior ACK'
    validate_epoch_state(actual,record)
    ack = validate_continuation(attempt,key,actual,owner)
    transfer['ipc_get_verify_seconds'] = time.perf_counter()-tick
    transfer['transport_warning_reconciled'] = warning
    # The first local ACK may have been reconstructed while checking the RAM journal.
    p = epoch_file.with_name(f'ack_{snapshot["epoch"]:02d}.json')
    if ack.get('transfer') is None:
        numeric(p,dict(ack,transfer=transfer))
    return transfer


def register(attempt, j):
    try:
        return request(attempt,'register',key=j['key'],job=j,pid=os.getpid())
    except (OSError,EOFError,ConnectionError):
        r = request(attempt,'status')['jobs'].get(j['key'])
        assert r and r['job'] == j and r['status'] == 'RUNNING' and r['worker_pid'] == os.getpid()
        return dict(attempt=r['attempts'],state=request(attempt,'get',key=j['key'])['state'])


def restore(model, ema, opt, loader, state, device):
    model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); opt.load_state_dict(cpu_copy(state['optimizer']))
    loader.set_state(state['loader']); restore_rng(state['rng'],device)


def adam_steps(model, opt):
    return {name: int(opt.state[p]['step'].item()) if 'step' in opt.state.get(p, {}) else None
            for name, p in model.named_parameters()}


def observe_step(state, model, opt, before, value):
    after = adam_steps(model, opt)
    activity = {name: p.grad is not None for name, p in model.named_parameters()}
    for name, active in activity.items():
        expected = (before[name] or 0)+1 if active else before[name]
        assert after[name] == expected, ('ADAM_ACTIVITY', name, before[name], after[name])
    assert value['optimizer_updates'] == value['ema_updates'] == value['ordinary_forwards'] == value['scalar_backwards'] == 1
    assert value['probe_forwards'] == value['view_forwards'] == value['algorithm_rng_draws'] == 0
    state['global_step'] += 1
    for name in state['counts']: state['counts'][name] += value[name]
    assert set(value['phase_increments']) == set(PHASE_FIELDS)
    for name in PHASE_FIELDS: state['phase_counters'][name] += value['phase_increments'][name]
    state['adam_step_counters'] = after; state['last_parameter_activity'] = activity


def new_costs():
    return dict(training_seconds=0.,evaluation_seconds=0.,diagnostic_seconds=0.,diagnostic_steps=0,
        diagnostic_seconds_semantics='inclusive_diagnosed_step_subset_of_training_seconds',
        serialization_seconds=0.,transfer_seconds=0.,serialization_bytes=0,
        ordinary_forwards=0,scalar_backwards=0,evaluation_forwards=0,evaluation_rows=0,
        probe_forwards=0,view_forwards=0,algorithm_rng_draws=0,
        optimizer_updates=0,ema_updates=0,attempted_training_batches=0,failed_training_seconds=0.,
        failed_training_batches=0,unmeasured_transfer_transactions=0,
        ordinary_forwards_semantics='successful_step_returns_failed_batch_calls_unmeasured')


def cost_difference(costs, before):
    return {k:v-before.get(k,0) if isinstance(v,(int,float)) else v for k,v in costs.items()}


def step_cost(costs, value, diagnose):
    costs['training_seconds'] += value['seconds']
    for name in ('ordinary_forwards','scalar_backwards','optimizer_updates','ema_updates','probe_forwards','view_forwards','algorithm_rng_draws'):
        costs[name] += value[name]
    if diagnose:
        costs['diagnostic_seconds'] += value['seconds']; costs['diagnostic_steps'] += 1


def transfer_cost(costs, transfer):
    if transfer is None:
        costs['unmeasured_transfer_transactions'] += 1
        return
    costs['serialization_seconds'] += transfer['serialization_seconds']
    costs['serialization_bytes'] += transfer['serialization_bytes']
    costs['transfer_seconds'] += transfer['ipc_get_verify_seconds']


def ledger_add(summary, ledger):
    B = ledger['B']; summary['batches'] += 1; summary['rows'] += B
    for key in ('total_mass','bce_mass','opposite_mass'):
        s = summary['masses'].setdefault(key,dict(update_sum=0.,row_weighted_sum=0.))
        s['update_sum'] += ledger[key]; s['row_weighted_sum'] += B*ledger[key]
    for task,value in ledger['task_mass'].items():
        key = 'task'+task
        s = summary['masses'].setdefault(key,dict(update_sum=0.,row_weighted_sum=0.))
        s['update_sum'] += value; s['row_weighted_sum'] += B*value
    for cell,r in ledger['cells'].items():
        c = summary['cells'].setdefault(cell,dict(n=0,n0=0,n1=0,statistics={}))
        for key in ('n','n0','n1'): c[key] += r[key]
        for key in ('mu0','mu1','var0','var1','bbar0','bbar1','S','Q_R','Q_B','Q_M','risk'):
            s = c['statistics'].setdefault(key,dict(observed_batches=0,null_batches=0,sum=0.,count_weighted_sum=0.,observed_count=0))
            if r[key] is None:
                s['null_batches'] += 1; continue
            denominator = r['n0'] if key in ('mu0','var0','bbar0') else r['n1'] if key in ('mu1','var1','bbar1') else r['n']
            s['observed_batches'] += 1; s['sum'] += r[key]
            s['count_weighted_sum'] += denominator*r[key]; s['observed_count'] += denominator


def finish_ledger(summary):
    for s in summary['masses'].values():
        s['update_mean'] = s['update_sum']/summary['batches']
        s['row_weighted_mean'] = s['row_weighted_sum']/summary['rows']
    return summary


def validation(model, ema, bundle, device, costs, profile):
    values = {}; raw_costs = {}
    for kind,estimator in (('online',model),('fixed_ema',ema)):
        profile.mark('validation_'+kind)
        mode = estimator.training; before_rng = state_digest(rng_state(device))
        v,cost = evaluate(estimator,bundle.val_x,bundle.val_y,device)
        estimator.train(mode)
        assert state_digest(rng_state(device)) == before_rng
        assert v['macro_auc'] is not None and all(v[c+'_auc'] is not None and v[c+'_logloss'] is not None for c in CELLS)
        values[kind] = v; raw_costs[kind] = cost
        costs['evaluation_seconds'] += cost['seconds']; costs['evaluation_forwards'] += cost['forwards']; costs['evaluation_rows'] += cost['rows']
        profile.sample()
    return values,raw_costs


def snapshot_record(j, model, ema, opt, loader, state, profile, physical_seconds):
    profile.mark('cpu_snapshot_serialization')
    snapshot = cpu_copy(dict(job=j,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        rng=rng_state(next(model.parameters()).device),loader=loader.get_state(),**state,
        physical_worker_seconds=physical_seconds,
        peak_allocated_bytes=profile.value['torch_allocated_peak_bytes'],
        peak_reserved_bytes=profile.value['torch_reserved_peak_bytes'],peak_full_process_bytes=profile.value['full_process_peak_bytes']))
    profile.sample()
    # Snapshot costs end at capture; its own serialization/IPC cost is in the ACK.
    record = {name:snapshot[name] for name in ('job','epoch','global_step','counts','selected_epochs','costs',
        'physical_worker_seconds','peak_allocated_bytes','peak_reserved_bytes','peak_full_process_bytes',
        'clock_kind','next_epoch','schedule_epoch','schedule','next_schedule','phase_counters',
        'adam_step_counters','last_parameter_activity')}
    for name in ('engineering_steps','engineering_commits','scientific_epochs_completed','sample_epoch','sample_batches'):
        if name in snapshot: record[name] = snapshot[name]
    record.update(source_hash=j['source_hash'],design_bundle_sha256=j['design_bundle_sha256'],
        continuation_hash=state_digest(snapshot),selected_hashes={k:state_digest(v) for k,v in snapshot['selected_models'].items()},
        **{name+'_hash':state_digest(snapshot[name]) for name in ('model','ema','optimizer','rng','loader')},test_accessed=False)
    return snapshot,record


def complete(attempt, j, state, out, physical_attempts):
    assert state['epoch'] == 50
    expected = 50*j['config']['steps_per_epoch']
    assert state['global_step'] == expected and state['counts'] == dict(optimizer_updates=expected,ema_updates=expected)
    assert j.get('purpose', 'formal') == 'formal' and state['clock_kind'] == 'formal'
    validate_formal_clock(j, 50, j['config']['steps_per_epoch'], expected, state['counts'], state['phase_counters'])
    assert state['next_epoch'] == 51 and state['next_schedule'] is None
    result = dict(job=j,epochs=50,global_step=state['global_step'],counts=state['counts'],
        initial_hash=state['initial_hash'],selected_epochs=state['selected_epochs'],selected_validation=state['selected_validation'],
        selected_hashes={k:state_digest(v) for k,v in state['selected_models'].items()},common_validation=state['common_validation'],
        final_hashes=dict(online=state_digest(state['model']),fixed_ema=state_digest(state['ema'])),
        optimizer_hash=state_digest(state['optimizer']),costs=state['costs'],
        peak_allocated_bytes=state['peak_allocated_bytes'],peak_reserved_bytes=state['peak_reserved_bytes'],
        peak_full_process_bytes=state['peak_full_process_bytes'],physical_attempts=physical_attempts,
        worker_seconds=state['physical_worker_seconds'],physical_cost_provenance='all_runs_key_attempt_physical_attempt_json',test_accessed=False)
    result.update({n: cpu_copy(state[n]) for n in ('clock_kind','next_epoch','schedule_epoch','schedule','next_schedule',
        'phase_counters','adam_step_counters','last_parameter_activity')})
    epochs = {}
    for ack in commits(attempt,j['key']):
        path = Path(ack['epoch_file']); epochs[ack['epoch']] = (path,read_json(path))
    verify_selection(result,epochs)
    try:
        assert request(attempt,'complete',key=j['key'],result=result)['status'] == 'COMPLETE'
    except (OSError,EOFError,ConnectionError):
        retained = request(attempt,'get',key=j['key'])
        assert retained['result'] == result, 'Completion unconfirmed: PAUSE and reconcile epoch50'
    retained = request(attempt,'get',key=j['key'])
    assert retained['result'] == result and set(retained['state']) == {'selected_models','selected_epochs','selected_validation'}
    assert {k:state_digest(v) for k,v in retained['state']['selected_models'].items()} == result['selected_hashes']
    assert all(retained['state'][name] == result[name] for name in ('selected_epochs','selected_validation'))
    return result


def worker(attempt, key, gpu):
    attempt = cycle_path(attempt); manifest = validate_attempt(attempt,require_smoke=True)
    require_storage(attempt,manifest)
    j = read_json(attempt/'jobs'/f'{key}.json'); assert j in manifest['jobs']
    if stop_requested() or (attempt/'PAUSE_TRAINING').exists(): return
    registration = register(attempt,j); number = registration['attempt']; previous = registration['state']
    out = attempt/'runs'/key/f'attempt{number}'; out.mkdir(parents=True,exist_ok=True)
    owner = request(attempt,'status')['jobs'][key]
    preceding_ack = validate_continuation(attempt,key,previous,owner)
    tick = time.perf_counter(); costs = new_costs(); cost_start = copy.deepcopy(costs)
    profile = None; status = 'FAILED'; error_record = None
    physical_start_step = previous['global_step'] if previous else 0
    numeric(out/'physical_attempt.json',dict(key=key,attempt=number,pid=os.getpid(),gpu=gpu,status='RUNNING',
        resume_epoch=owner['epoch'],resume_step=physical_start_step,started=time.time(),test_accessed=False))
    try:
        if previous is not None and previous['epoch'] == 50:
            transfer_cost(previous['costs'],preceding_ack.get('transfer'))
            complete(attempt,j,previous,out,number); status = 'COMPLETE'; return
        device = torch.device(f'cuda:{gpu}'); setup(j['seed'],device)
        bundle,data = dataset_cache(j['dataset'],manifest['cache_dir']); immutable(out/'data_manifest.json',data)
        model = make_model(bundle,load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model,j['config'])
        loader = torch.Generator().manual_seed(j['seed']+11000)
        index = read_json(attempt/'reused_reference_index.json'); reference_key = f'{j["dataset"]}/{j["seed"]}'
        initial_hash = state_digest(model.state_dict()); assert initial_hash == index['initial_hashes'][reference_key]
        state = dict(epoch=0,global_step=0,counts=dict(optimizer_updates=0,ema_updates=0),
            selected_models={},selected_epochs={},selected_validation={},best={k:-1. for k in KINDS},
            common_validation={},initial_hash=initial_hash,costs=costs,
            clock_kind='formal',next_epoch=1,schedule_epoch=None,schedule=None,
            next_schedule=dict(schedule(j['arm'],1),clock_kind='formal'),phase_counters=empty_phase_counters(),
            adam_step_counters=adam_steps(model,opt),last_parameter_activity={n:False for n,_ in model.named_parameters()})
        prior_attempts = [read_json(p) for p in sorted((attempt/'runs'/key).glob('attempt*/physical_attempt.json')) if p.parent != out]
        assert all(r['status'] != 'RUNNING' for r in prior_attempts), 'Unreconciled physical attempt: PAUSE'
        previous_seconds = sum(r['seconds'] for r in prior_attempts)
        if previous:
            restore(model,ema,opt,loader,previous,device)
            state = {k:cpu_copy(previous[k]) for k in state}; costs = state['costs']
            assert adam_steps(model,opt) == state['adam_step_counters']
            transfer_cost(costs,preceding_ack.get('transfer'))
        costs['physical_attempts_before_current'] = dict(count=len(prior_attempts),seconds=previous_seconds,
            failed=sum(r['status']=='FAILED' for r in prior_attempts),
            numeric_costs={k:sum(r['costs'][k] for r in prior_attempts) if all(r['costs'].get(k) is not None for r in prior_attempts) else None
                for k,v in new_costs().items() if isinstance(v,(int,float))})
        cost_start = copy.deepcopy(costs)
        previous_peaks = {name:previous[name] for name in ('peak_allocated_bytes','peak_reserved_bytes','peak_full_process_bytes')} if previous else {}
        del previous,registration
        torch.cuda.reset_peak_memory_stats(device)
        profile = Profile(gpu,out/'resource_profile.json',j).start()
        for old,new in (('peak_allocated_bytes','torch_allocated_peak_bytes'),('peak_reserved_bytes','torch_reserved_peak_bytes'),('peak_full_process_bytes','full_process_peak_bytes')):
            profile.value[new] = max(profile.value[new],previous_peaks.get(old,0))
        epoch0 = state['epoch']
        for epoch in range(epoch0+1,51):
            if stop_requested() or (attempt/'PAUSE_TRAINING').exists():
                request(attempt,'paused',key=key,reason='USER_STOPPED' if stop_requested() else 'TECHNICAL_PAUSE'); status='PAUSED'; return
            epoch_tick = time.perf_counter(); model.train()
            order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
            order_hash = state_digest(np.asarray(bundle.train_row_ids[order],dtype=np.int64))
            assert order_hash == index['epoch_order_hashes'][f'{reference_key}/{epoch}']
            sums = dict(loss=0.,clean_loss=0.,completed_loss=0.)
            summary = dict(batches=0,rows=0,masses={},cells={})
            phase_summary = {phase:dict(batches=0,rows=0,masses={},cells={}) for phase in ('on','off')}
            phase_losses = {phase:dict(rows=0,updates=0,loss_sum=0.,clean_loss_sum=0.,completed_loss_sum=0.) for phase in ('on','off')}
            batch_path = out/f'batches_epoch_{epoch:02d}.jsonl.gz'
            with gzip.open(batch_path,'xt',compresslevel=3) as log:
                for batch,offset in enumerate(range(0,len(order),40960),1):
                    ids = order[offset:offset+40960]; diagnose = batch == 1 and epoch in (1,10,11,40,41,50)
                    profile.mark('diagnostic' if diagnose else 'tail' if len(ids)<40960 else 'training')
                    x = torch.as_tensor(bundle.train_x[ids],device=device).T
                    y = torch.as_tensor(bundle.train_y[ids],device=device,dtype=torch.float32)
                    validate_formal_clock(j,epoch,batch-1,state['global_step'],state['counts'],state['phase_counters'])
                    assert j.get('purpose','formal') == 'formal' and state['clock_kind'] == 'formal'
                    context = dict(dataset=j['dataset'],seed=j['seed'],epoch=epoch,batch=batch,global_step=state['global_step']+1,
                        purpose='formal',clock_kind='formal')
                    before = adam_steps(model,opt)
                    costs['attempted_training_batches'] += 1
                    batch_tick = time.perf_counter()
                    try:
                        value = train_step(model,opt,ema,x,y,j['config'],context,bundle.train_row_ids[ids],diagnose=diagnose)
                    except BaseException:
                        costs['failed_training_seconds'] += time.perf_counter()-batch_tick
                        costs['failed_training_batches'] += 1
                        raise
                    step_cost(costs,value,diagnose); observe_step(state,model,opt,before,value)
                    validate_formal_clock(j,epoch,batch,state['global_step'],state['counts'],state['phase_counters'])
                    for name in sums: sums[name] += len(ids)*value[name]
                    ledger_add(summary,value['ledger'])
                    phase=value['schedule']['schedule_phase']; ledger_add(phase_summary[phase],value['ledger'])
                    phase_losses[phase]['rows'] += len(ids); phase_losses[phase]['updates'] += 1
                    for name in sums: phase_losses[phase][name+'_sum'] += len(ids)*value[name]
                    log.write(json.dumps(dict(job_key=key,**context,**value,counts=state['counts'],phase_counters=state['phase_counters'],
                        adam_step_counters=state['adam_step_counters'],last_parameter_activity=state['last_parameter_activity']),
                        separators=(',', ':'),allow_nan=False)+'\n')
                    profile.sample()
                    if epoch == epoch0+1 and batch == 1:
                        ack = request(attempt,'allocated',key=key,peak_memory_bytes=profile.value['incoming_bytes'])
                        assert ack['status'] == 'ALLOCATED'
                        numeric(out/'allocation_ack.json',dict(key=key,attempt=number,gpu=gpu,pid=os.getpid(),
                            full_process_bytes=profile.value['full_process_peak_bytes'],ack=ack,time=time.time()))
                    del x,y,value
            assert summary['batches'] == j['config']['steps_per_epoch'] and summary['rows'] == len(order)
            values,evaluation_cost = validation(model,ema,bundle,device,costs,profile)
            for kind,estimator in (('online',model),('fixed_ema',ema)):
                if values[kind]['macro_auc'] > state['best'][kind]:
                    state['best'][kind] = values[kind]['macro_auc']; state['selected_epochs'][kind] = epoch
                    state['selected_models'][kind] = cpu_copy(estimator.state_dict()); state['selected_validation'][kind] = values[kind]
            if epoch in (10,30,40,50): state['common_validation'][str(epoch)] = values
            state['epoch'] = epoch; state['costs'] = costs
            state.update(next_epoch=epoch+1,schedule_epoch=epoch,schedule=dict(schedule(j['arm'],epoch),clock_kind='formal'),
                next_schedule=dict(schedule(j['arm'],epoch+1),clock_kind='formal') if epoch<50 else None)
            snapshot,record = snapshot_record(j,model,ema,opt,loader,state,profile,previous_seconds+time.perf_counter()-tick)
            record.update(validation=values,validation_cost=evaluation_cost,epoch_order_hash=order_hash,
                initial_hash=initial_hash,objective_ledger=finish_ledger(summary),
                **{name:v/len(order) for name,v in sums.items()},epoch_seconds=time.perf_counter()-epoch_tick,
                batch_log=str(batch_path),batch_log_sha256=sha_file(batch_path),batch_count=summary['batches'])
            record['phase_ledgers'] = {phase:finish_ledger(v) if v['batches'] else v for phase,v in phase_summary.items()}
            record['phase_losses'] = {phase:dict(v,**{n: v[n+'_sum']/v['rows'] if v['rows'] else None for n in sums}) for phase,v in phase_losses.items()}
            record['prefix_check'] = prefix_check(attempt,record)
            transfer = commit_snapshot(attempt,key,snapshot,record,out/f'epoch_{epoch:02d}.json',profile)
            transfer_cost(costs,transfer); del snapshot
            if epoch < 50 and (stop_requested() or (attempt/'PAUSE_TRAINING').exists()):
                request(attempt,'paused',key=key,reason='USER_STOPPED' if stop_requested() else 'TECHNICAL_PAUSE'); status='PAUSED'; return
        retained = request(attempt,'get',key=key)['state']
        validate_continuation(attempt,key,retained,request(attempt,'status')['jobs'][key])
        retained['costs'] = costs
        complete(attempt,j,retained,out,number); status='COMPLETE'
    except BaseException as error:
        error_record = dict(error=repr(error),traceback=traceback.format_exc(),oom=isinstance(error,torch.OutOfMemoryError))
        numeric(out/'failure.json',error_record); numeric(attempt/'PAUSE_TRAINING',dict(reason='TECHNICAL_PAUSE',key=key,**error_record))
        try: request(attempt,'failed',key=key,failure=error_record)
        except (OSError,EOFError,ConnectionError): pass
        raise
    finally:
        try:
            measured = profile.close() if profile else None
        except BaseException as error:
            measured = copy.deepcopy(profile.value)
            measured['profiling_error'] = repr(error)
            numeric(attempt/'PAUSE_TRAINING',dict(reason='PROFILE_TECHNICAL_PAUSE',key=key,error=repr(error)))
        numeric(out/'physical_attempt.json',dict(key=key,attempt=number,pid=os.getpid(),gpu=gpu,status=status,
            resume_step=physical_start_step,seconds=time.perf_counter()-tick,costs=cost_difference(costs,cost_start),profile=measured,
            failure=error_record,test_accessed=False))


def identity_state(model, opt, ema, loader, device):
    return cpu_copy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        rng=rng_state(device),loader=loader.get_state(),activity=[p.grad is not None for p in model.parameters()]))


def assert_restore_isolated(opt, snapshot):
    saved = snapshot['optimizer']['state']
    actual = opt.state_dict()['state']
    for key, values in saved.items():
        for name in ('step', 'exp_avg', 'exp_avg_sq'):
            if name in values:
                assert values[name].data_ptr() != actual[key][name].data_ptr(), ('RESTORE_ALIAS', key, name)


def smoke_case(attempt, j, gpu, manifest):
    assert j['purpose'] == 'engineering_smoke' and j['fixture_id'] == 'temporal_populated_boundary_v1'
    assert j['source_hash'] == manifest['source_hash'] and j['design_bundle_sha256'] == manifest['design_bundle_sha256']
    key = j['key']; registration = register(attempt,j); assert registration['state'] is None
    number = registration['attempt']; out = attempt/'runs'/key/f'attempt{number}'
    out.mkdir(parents=True,exist_ok=True); tick=time.perf_counter(); costs=new_costs(); profile=None
    status='FAILED'; failure=None; final_profile=None; physical_calls=0
    attempt_tick=tick; attempt_cost_start=copy.deepcopy(costs); attempt_call_start=0
    try:
        device=torch.device(f'cuda:{gpu}'); setup(2022,device)
        bundle,_=dataset_cache(j['dataset'],manifest['cache_dir'])
        model=make_model(bundle,load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema=copy.deepcopy(model).requires_grad_(False); opt=optimizer(model,j['config'])
        loader=torch.Generator().manual_seed(2022+11000); initial_hash=state_digest(model.state_dict())
        index=read_json(attempt/'reused_reference_index.json'); assert initial_hash==index['initial_hashes'][j['dataset']+'/2022']
        order=torch.randperm(len(bundle.train_y),generator=loader).numpy()
        assert state_digest(np.asarray(bundle.train_row_ids[order],dtype=np.int64))==index['epoch_order_hashes'][j['dataset']+'/2022/1']
        samples=(order[:40960],order[(len(order)//40960)*40960:])
        assert [len(ids) for ids in samples] == [40960,j['config']['tail_rows']]
        frozen_loader_hash=state_digest(loader.get_state())
        torch.cuda.reset_peak_memory_stats(device); profile=Profile(gpu,out/'resource_profile.json',j).start()
        before_virtual=10 if j['arm']==ARMS[0] else 40
        state=dict(epoch=0,global_step=0,counts=dict(optimizer_updates=0,ema_updates=0),
            selected_models={},selected_epochs={},selected_validation={},best={},common_validation={},
            initial_hash=initial_hash,costs=costs,clock_kind='engineering_virtual',next_epoch=None,
            schedule_epoch=None,schedule=None,next_schedule=dict(schedule(j['arm'],before_virtual),clock_kind='engineering_virtual'),
            phase_counters=empty_phase_counters(),adam_step_counters=adam_steps(model,opt),
            last_parameter_activity={n:False for n,_ in model.named_parameters()},engineering_steps=0,engineering_commits=0,
            scientific_epochs_completed=0,sample_epoch=1,sample_batches=[1,j['config']['steps_per_epoch']])
        batches=[]; references=[]; transfers=[]; baseline=None; baseline_hash=None; suffix_state=None; suffix_values=None

        def execute(step, direct=False, diagnose=False):
            nonlocal physical_calls
            ids=samples[(step-1)%2]; virtual=before_virtual+(step>2)
            context=dict(dataset=j['dataset'],seed=2022,epoch=virtual,batch=step,global_step=step,
                purpose='engineering_smoke',clock_kind='engineering_virtual',sample_epoch=1,
                sample_batch=1 if step%2 else j['config']['steps_per_epoch'])
            phase=schedule(j['arm'],virtual)['schedule_phase']
            tag=('direct_c2_' if direct else '')+phase+('_full' if step%2 else '_tail')+('_diagnostic' if diagnose else '')
            profile.mark(tag)
            x=torch.as_tensor(bundle.train_x[ids],device=device).T
            y=torch.as_tensor(bundle.train_y[ids],device=device,dtype=torch.float32)
            costs['attempted_training_batches']+=1; physical_calls+=1; start=time.perf_counter()
            try:
                if direct:
                    inner=dict(j['config'],arm='NQR-025',mixture_lambda=schedule(j['arm'],virtual)['effective_lambda'],
                        objective='natural_quadratic_ranking_v1')
                    value=frozen_train_step(model,opt,ema,x,y,inner,context,bundle.train_row_ids[ids],diagnose=False)
                else:
                    value=train_step(model,opt,ema,x,y,j['config'],context,bundle.train_row_ids[ids],diagnose=diagnose)
                step_cost(costs,value,diagnose); profile.sample()
                return value,context
            except BaseException:
                costs['failed_training_seconds']+=time.perf_counter()-start; costs['failed_training_batches']+=1; raise
            finally:
                del x,y

        for step in range(1,5):
            if step==3:
                if stop_requested():
                    request(attempt,'paused',key=key,reason='USER_STOPPED')
                    status='PAUSED'
                    return dict(status='USER_STOPPED',key=key,engineering_steps=2,test_accessed=False)
                # Sequential reference routes use the populated CPU baseline and actual step2.
                baseline_hash=state_digest(baseline)
                restore(model,ema,opt,loader,baseline,device); assert_restore_isolated(opt,baseline)
                suffix_state={n:cpu_copy(baseline[n]) for n in state}
                for replay in (3,4):
                    before=adam_steps(model,opt); value,_=execute(replay,diagnose=False)
                    observe_step(suffix_state,model,opt,before,value)
                suffix_state.update(epoch=2,engineering_steps=4,engineering_commits=2,schedule_epoch=before_virtual+1,
                    schedule=dict(schedule(j['arm'],before_virtual+1),clock_kind='engineering_virtual'),next_schedule=None)
                suffix_identity=identity_state(model,opt,ema,loader,device)
                suffix_values,_=validation(model,ema,bundle,device,costs,profile)
                assert state_digest(baseline)==baseline_hash
                request(attempt,'paused',key=key,reason='ENGINEERING_RECOVERY_CHECK')
                numeric(out/'physical_attempt.json',dict(key=key,purpose='engineering_smoke',attempt=number,
                    pid=os.getpid(),gpu=gpu,status='PAUSED',seconds=time.perf_counter()-attempt_tick,
                    costs=cost_difference(costs,attempt_cost_start),physical_train_calls=physical_calls-attempt_call_start,
                    profile=copy.deepcopy(profile.value),failure=None,test_accessed=False))
                profile.mark('recovery'); recovered=register(attempt,j)['state']
                number=request(attempt,'status')['jobs'][key]['attempts']
                out=attempt/'runs'/key/f'attempt{number}'; out.mkdir(parents=True,exist_ok=True)
                profile.path=out/'resource_profile.json'
                attempt_tick=time.perf_counter(); attempt_cost_start=copy.deepcopy(costs); attempt_call_start=physical_calls
                validate_continuation(attempt,key,recovered,request(attempt,'status')['jobs'][key])
                assert state_digest(recovered)==baseline_hash
                restore(model,ema,opt,loader,recovered,device); assert_restore_isolated(opt,baseline)
                state={n:cpu_copy(recovered[n]) for n in state}; state['costs']=costs
                del recovered
            base=identity_state(model,opt,ema,loader,device); base_hash=state_digest(base)
            before=adam_steps(model,opt); value,context=execute(step,diagnose=True)
            observe_step(state,model,opt,before,value)
            main=identity_state(model,opt,ema,loader,device)
            if step==1:
                ack=request(attempt,'allocated',key=key,peak_memory_bytes=profile.value['incoming_bytes'])
                assert ack['status']=='ALLOCATED'
                numeric(out/'allocation_ack.json',dict(key=key,gpu=gpu,pid=os.getpid(),actual_step=1,
                    full_rows=40960,ack=ack,full_process_bytes=profile.value['incoming_bytes'],time=time.time()))
            restore(model,ema,opt,loader,base,device); assert_restore_isolated(opt,base)
            direct,_=execute(step,direct=True)
            assert state_digest(identity_state(model,opt,ema,loader,device))==state_digest(main)
            for field in ('loss','clean_loss','completed_loss','branch','ledger','row_order_hash',
                          'optimizer_updates','ema_updates','ordinary_forwards','scalar_backwards',
                          'probe_forwards','view_forwards','algorithm_rng_draws'):
                assert value[field]==direct[field], ('C2_EQUIVALENCE',step,field)
            assert state_digest(base)==base_hash
            restore(model,ema,opt,loader,main,device); assert_restore_isolated(opt,main)
            assert state_digest(loader.get_state())==frozen_loader_hash
            batches.append(dict(context,**value,counts=cpu_copy(state['counts']),phase_counters=cpu_copy(state['phase_counters'])))
            references.append(dict(actual_step=step,source_equivalence=True,diagnostic_neutrality=True,
                                   base_unchanged=True,optimizer_restore_noalias=True))
            if step%2:
                continue
            values,eval_cost=validation(model,ema,bundle,device,costs,profile)
            commit=step//2; virtual=before_virtual+commit-1
            state.update(epoch=commit,engineering_steps=step,engineering_commits=commit,schedule_epoch=virtual,
                schedule=dict(schedule(j['arm'],virtual),clock_kind='engineering_virtual'),
                next_schedule=dict(schedule(j['arm'],before_virtual+1),clock_kind='engineering_virtual') if commit==1 else None)
            if commit==2:
                assert state_digest(identity_state(model,opt,ema,loader,device))==state_digest(suffix_identity)
                for field in ('global_step','counts','phase_counters','adam_step_counters','last_parameter_activity',
                              'epoch','clock_kind','next_epoch','schedule_epoch','schedule','next_schedule',
                              'engineering_steps','engineering_commits','scientific_epochs_completed','sample_epoch','sample_batches'):
                    assert state[field]==suffix_state[field], ('RECOVERY_SUFFIX',field)
                assert values==suffix_values and state_digest(baseline)==baseline_hash
            snapshot,record=snapshot_record(j,model,ema,opt,loader,state,profile,time.perf_counter()-tick)
            record.update(validation=values,validation_cost=eval_cost,purpose='engineering_smoke',batches=cpu_copy(batches),
                          physical_train_calls=physical_calls)
            profile.mark('ram_roundtrip'); returned=request(attempt,'roundtrip',value=snapshot)['value']
            assert state_digest(returned)==record['continuation_hash']; del returned
            transfer=commit_snapshot(attempt,key,snapshot,record,out/f'epoch_{commit:02d}.json',profile)
            transfers.append(transfer); transfer_cost(costs,transfer)
            if commit==1: baseline=snapshot
            else: del snapshot
        assert physical_calls==10 and costs['optimizer_updates']==10 and state['global_step']==4
        assert state_digest(baseline)==baseline_hash
        request(attempt,'paused',key=key,reason='ENGINEERING_RECOVERY_VERIFIED')
        request(attempt,'release',key=key); assert request(attempt,'get',key=key)['state'] is None
        measured=profile.close(); final_profile=measured; profile=None; status='PASS'
        result=dict(status='PASS',purpose='engineering_smoke',clock_kind='engineering_virtual',key=key,
            dataset=j['dataset'],arm=j['arm'],seed=2022,device=gpu,full_rows=40960,tail_rows=j['config']['tail_rows'],precision='float32',
            references=references,recovery_verified=True,source_equivalence_verified=True,
            effective_lambdas=[0.,.25],physical_train_calls=10,engineering_steps=4,engineering_commits=2,
            scientific_epochs_completed=0,continuation_hash=record['continuation_hash'],baseline_unchanged=True,
            source_hash=j['source_hash'],design_bundle_sha256=j['design_bundle_sha256'],profile=measured,costs=costs,
            transfers=transfers,seconds=time.perf_counter()-tick,test_accessed=False)
        numeric(attempt/'smoke'/key/'case.json',result); return result
    except BaseException as error:
        failure=dict(error=repr(error),traceback=traceback.format_exc(),oom=isinstance(error,torch.OutOfMemoryError))
        numeric(attempt/'PAUSE_TRAINING',dict(reason='SMOKE_TECHNICAL_PAUSE',key=key,**failure))
        try: request(attempt,'failed',key=key,failure=failure)
        except (OSError,EOFError,ConnectionError): pass
        raise
    finally:
        measured=profile.close() if profile else final_profile
        numeric(out/'physical_attempt.json',dict(key=key,purpose='engineering_smoke',attempt=number,pid=os.getpid(),gpu=gpu,
            status=status,physical_train_calls=physical_calls-attempt_call_start,seconds=time.perf_counter()-attempt_tick,
            costs=cost_difference(costs,attempt_cost_start),profile=measured,failure=failure,test_accessed=False))


def measured_profile(attempt, j, gpu, hardware, smoke=None):
    profiles=[]
    if smoke:
        assert any(r['dataset']==j['dataset'] and r['device']==gpu and r['profile']['hardware']==hardware for r in smoke['cases']), 'Device/profile changed: PAUSE'
        profiles += [r['profile'] for r in smoke['cases'] if r['dataset']==j['dataset']]
    for path in (attempt/'runs').glob('*/attempt*/resource_profile.json'):
        try:
            p = read_json(path)
            if p['dataset'] == j['dataset']: profiles.append(p)
        except json.JSONDecodeError: continue
    profiles=[p for p in profiles if p['hardware']==hardware and p['source_hash']==j['source_hash']
              and p['design_bundle_sha256']==j['design_bundle_sha256'] and p['measured'] and p['context_measured']]
    assert profiles, 'No compatible Cycle03 full-process profile: PAUSE'
    return max(incoming_profile(p) for p in profiles)


def finish_dead_attempt(attempt, key, owner_record, returncode=None):
    """A killed worker leaves numeric evidence and its preceding RAM ACK intact."""
    assert not alive(owner_record.get('worker_pid'))
    out=attempt/'runs'/key/f'attempt{owner_record["attempts"]}'; path=out/'physical_attempt.json'
    previous=read_json(path) if path.exists() else {}
    if previous.get('status') in ('FAILED','PAUSED','COMPLETE'): return previous
    launches=[r for r in map(json.loads,(attempt/'launch_ledger.jsonl').read_text().splitlines()) if r['key']==key]
    launch=launches[-1]; costs=new_costs()
    for log_path in sorted(out.glob('batches_epoch_*.jsonl.gz')):
        try:
            with gzip.open(log_path,'rt') as stream:
                for line in stream:
                    value=json.loads(line); costs['attempted_training_batches']+=1
                    step_cost(costs,value,value['diagnostics'] is not None)
        except (EOFError,json.JSONDecodeError):
            pass
    for ack_path in out.glob('ack_*.json'):
        transfer_cost(costs,read_json(ack_path).get('transfer'))
    # A terminated partial operation has no returned counters/timers; preserve nulls.
    costs.update(evaluation_seconds=None,evaluation_forwards=None,evaluation_rows=None,
                 failed_training_seconds=None,failed_training_batches=None)
    profile_path=out/'resource_profile.json'
    value=dict(key=key,attempt=owner_record['attempts'],pid=launch['pid'],gpu=launch['gpu'],status='FAILED',
        resume_step=previous.get('resume_step'),seconds=time.time()-launch['time'],
        seconds_semantics='launch_to_observed_death_upper_bound',costs=costs,
        profile=read_json(profile_path) if profile_path.exists() else None,
        failure=dict(reason='WORKER_DIED',returncode=returncode,unreturned_operation_costs=None),
        raw_cost_provenance='complete_returned_batch_lines_and_ACK_receipts_partial_operation_unknown',test_accessed=False)
    numeric(path,value); return value


def recover(attempt, manifest):
    assert not stop_requested()
    pause=attempt/'PAUSE_TRAINING'
    if pause.exists():
        correction=read_json(attempt/'RECOVERY_REVIEW.json')
        assert correction['verdict'] in ('PASS','PASS WITH FIXES') and not correction.get('blocking_issues')
        assert correction['source_hash']==manifest['source_hash'] and correction['design_bundle_sha256']==manifest['design_bundle_sha256']
        assert correction['pause_sha256']==sha_file(pause) and correction['same_protocol'] is True
        assert correction['reviewer_id'] not in read_json(manifest['review_path'])['coder_ids']
    status=request(attempt,'status'); assert not any(alive(r.get('worker_pid')) for r in status['jobs'].values())
    formal={j['key']:j for j in manifest['jobs']}
    for key,r in status['jobs'].items():
        if key not in formal: continue
        assert r['job']==formal[key]
        if r['status'] in ('RUNNING','FAILED'):
            finish_dead_attempt(attempt,key,r)
            state=request(attempt,'get',key=key)['state']; validate_continuation(attempt,key,state,r); del state
            physical = read_json(attempt/'runs'/key/f'attempt{r["attempts"]}'/'physical_attempt.json')
            assert physical['status'] in ('FAILED','PAUSED')
            if r['epoch'] == 0:
                assert physical['status'] == 'FAILED', 'Epoch-zero reinitialization requires failed-attempt evidence'
            request(attempt,'recover',key=key)
    assert not (attempt/'PAUSE_TRAINING').exists(), 'Unresolved technical pause requires parent correction'


def run(attempt, devices, review_path, recovery=False):
    attempt=cycle_path(attempt); manifest=validate_attempt(attempt,review_path,require_smoke=True)
    require_storage(attempt,manifest); assert not stop_requested()
    if recovery: recover(attempt,manifest)
    existing = request(attempt,'status')['jobs']
    assert not any(alive(r.get('worker_pid')) for r in existing.values()), 'Existing workers need their current foreground supervisor'
    formal_keys = {j['key'] for j in manifest['jobs']}
    assert not any(r['status'] in ('FAILED','RUNNING') for k,r in existing.items() if k in formal_keys), 'Use --recover after technical correction'
    assert not (attempt/'PAUSE_TRAINING').exists()
    nvml=NVML(); hardware={g:device_identity(g) for g in devices}; smoke=read_json(attempt/'SMOKE_REPORT.json')
    processes={}; pending={}; started=time.perf_counter(); reason='RUNNING'
    try:
        while True:
            stopped=stop_requested(); status=request(attempt,'status'); records=status['jobs']
            for key,p in list(processes.items()):
                if p.poll() is not None:
                    p.wait(); del processes[key]
                    if records.get(key,{}).get('status') not in ('COMPLETE','PAUSED'):
                        if key in records: finish_dead_attempt(attempt,key,records[key],p.returncode)
                        numeric(attempt/'PAUSE_TRAINING',dict(reason='WORKER_EXIT',key=key,returncode=p.returncode))
            for gpu,key in list(pending.items()):
                r=records.get(key,{})
                if r.get('allocated') or (r.get('status') in ('FAILED','PAUSED','COMPLETE')):
                    # Read fresh measurements after this card's real allocation ACK.
                    append(attempt/'admission.jsonl',dict(event='ALLOCATION_ACK_RESAMPLE',gpu=gpu,key=key,
                        current=nvml.card(gpu),host_fraction=host_ram(),allocated=r.get('allocated',False),time=time.time()))
                    del pending[gpu]
            failed=any(r.get('status')=='FAILED' for k,r in records.items() if any(j['key']==k for j in manifest['jobs']))
            paused=(attempt/'PAUSE_TRAINING').exists() or failed
            finished=all(records.get(j['key'],{}).get('status')=='COMPLETE' for j in manifest['jobs'])
            if finished and not processes:
                reason='TECHNICAL_PAUSE' if paused else 'USER_STOPPED' if stopped else 'COMPLETE'; break
            if (stopped or paused) and not processes:
                reason='USER_STOPPED' if stopped else 'TECHNICAL_PAUSE'; break
            if not stopped and not paused:
                for gpu in devices:
                    # Each decision uses a new card and host observation; no card coupling.
                    if stop_requested(): break
                    candidates=[j for j in manifest['jobs'] if j['key'] not in processes and records.get(j['key'],{}).get('status','REGISTERED') in ('REGISTERED','PAUSED')]
                    candidates.sort(key=lambda j:(j['dataset']!='kuairand',j['seed'],ARMS.index(j['arm'])))
                    for j in candidates:
                        incoming=measured_profile(attempt,j,gpu,hardware[gpu],smoke)
                        decision=admission(nvml.card(gpu),host_ram(),incoming,hardware[gpu]['cuda_usable_bytes'],gpu in pending)
                        append(attempt/'admission.jsonl',dict(event='ADMISSION',key=j['key'],gpu=gpu,time=time.time(),**decision))
                        if not decision['eligible']: continue
                        if stop_requested(): break
                        command=[PYTHON,'-m',MODULE,'worker','--attempt',str(attempt),'--key',j['key'],'--gpu',str(gpu)]
                        log_path=attempt/'worker_logs'/f'{j["key"]}_{time.time_ns()}.log'; log_path.parent.mkdir(parents=True,exist_ok=True)
                        with log_path.open('xb') as log:
                            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=log)
                        processes[j['key']]=p; pending[gpu]=j['key']
                        append(attempt/'launch_ledger.jsonl',dict(key=j['key'],gpu=gpu,pid=p.pid,command=command,time=time.time(),admission=decision))
                        append(attempt/'admission.jsonl',dict(event='STARTUP_RESAMPLE',gpu=gpu,key=j['key'],
                            current=nvml.card(gpu),host_fraction=host_ram(),time=time.time()))
                        break
            time.sleep(.5)
    finally:
        # Children stay attached and finish the current atomic epoch when supervision stops.
        if processes:
            numeric(attempt/'PAUSE_TRAINING',dict(reason='SUPERVISION_EXIT',time=time.time()))
            for p in processes.values(): p.wait()
        append(attempt/'foreground_runs.jsonl',dict(status=reason,seconds=time.perf_counter()-started,test_accessed=False,time=time.time()))
    if reason=='COMPLETE':
        handoff(attempt)
    return dict(status=reason,test_accessed=False)


def smoke(attempt, devices, review_path):
    attempt=cycle_path(attempt); manifest=validate_attempt(attempt,review_path)
    require_storage(attempt,manifest); assert not stop_requested() and not (attempt/'PAUSE_TRAINING').exists()
    assert len(devices)==2 and len(set(devices))==2
    assert not (attempt/'SMOKE_REPORT.json').exists(), 'Completed smoke is reused, never rerun'
    nvml=NVML(); hardware={g:device_identity(g) for g in devices}; cases=[]
    for di,dataset in enumerate(DATASETS):
        for ai,arm in enumerate(ARMS):
            gpu=devices[(di+ai)%2]
            j=job(dataset,arm,2022,manifest['source_hash']); j.pop('key')
            j.update(purpose='engineering_smoke',fixture_id='temporal_populated_boundary_v1',
                run_manifest_sha256=sha_file(attempt/'run_manifest.json')); j['key']=digest(j)
            immutable(attempt/'smoke'/j['key']/'job.json',j)
            path=attempt/'smoke'/j['key']/'case.json'
            if path.exists():
                case=read_json(path)
                assert case['status']=='PASS' and case['source_hash']==manifest['source_hash']
                assert case['design_bundle_sha256']==manifest['design_bundle_sha256'] and case['device']==gpu
                cases.append(case); continue
            while True:
                assert not stop_requested() and not (attempt/'PAUSE_TRAINING').exists()
                decision=admission(nvml.card(gpu),host_ram(),BOOTSTRAP[dataset],hardware[gpu]['cuda_usable_bytes'])
                append(attempt/'smoke_admission.jsonl',dict(dataset=dataset,arm=arm,gpu=gpu,time=time.time(),
                    bootstrap=dict(c2_formal_full_process_bytes=BOOTSTRAP[dataset],own_profile=False),**decision))
                if decision['eligible']: break
                time.sleep(.5)
            command=[PYTHON,'-m',MODULE,'worker','--attempt',str(attempt),'--key',j['key'],'--gpu',str(gpu),'--smoke-case']
            log_path=attempt/'smoke'/j['key']/'worker.log'
            with log_path.open('ab') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=log)
            append(attempt/'smoke_admission.jsonl',dict(event='STARTUP_RESAMPLE',gpu=gpu,key=j['key'],
                current=nvml.card(gpu),host_fraction=host_ram(),time=time.time()))
            allocated_seen=False
            while process.poll() is None:
                owner=request(attempt,'status')['jobs'].get(j['key'],{})
                if owner.get('allocated') and not allocated_seen:
                    append(attempt/'smoke_admission.jsonl',dict(event='ALLOCATION_ACK_RESAMPLE',gpu=gpu,key=j['key'],
                        current=nvml.card(gpu),host_fraction=host_ram(),time=time.time()))
                    allocated_seen=True
                time.sleep(.1)
            assert process.wait()==0, f'Engineering case failed: {log_path}'
            if stop_requested() and not path.exists():
                return dict(status='USER_STOPPED',cases=len(cases),test_accessed=False)
            cases.append(read_json(path))
            append(attempt/'smoke_admission.jsonl',dict(event='CASE_EXIT_RESAMPLE',gpu=gpu,
                current=nvml.card(gpu),host_fraction=host_ram(),key=j['key'],time=time.time()))
    report=dict(status='PASS',purpose='engineering_smoke',cases=cases,source_hash=manifest['source_hash'],
        design_bundle_sha256=manifest['design_bundle_sha256'],run_manifest_sha256=sha_file(attempt/'run_manifest.json'),
        owner_endpoint_hash=read_json(attempt/'STORAGE_OWNER.json')['endpoint_hash'],hardware=hardware,test_accessed=False)
    immutable(attempt/'SMOKE_REPORT.json',report)
    numeric(attempt/'OLD_INVENTORY_AFTER_SMOKE.json',assert_old_unchanged(attempt))
    return dict(status='PASS',cases=4,physical_train_calls=40,test_accessed=False)


def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(map(tensor_bytes,value))
    return 0


def handoff(attempt):
    attempt=cycle_path(attempt); manifest=validate_attempt(attempt,require_smoke=True)
    owner=require_storage(attempt,manifest); status=request(attempt,'status')
    assert not any(alive(r.get('worker_pid')) for r in status['jobs'].values())
    rows=[]; epochs=[]
    for j in manifest['jobs']:
        r=status['jobs'][j['key']]; assert r['status']=='COMPLETE' and r['epoch']==50
        retained=request(attempt,'get',key=j['key']); result=retained['result']; state=retained['state']
        assert result['job']==j and result['test_accessed'] is False
        assert set(state)=={'selected_models','selected_epochs','selected_validation'}
        hashes={k:state_digest(v) for k,v in state['selected_models'].items()}; assert hashes==result['selected_hashes']
        assert all(state[name] == result[name] for name in ('selected_epochs','selected_validation'))
        committed=epoch_files(attempt,result); assert len(committed)==50
        for epoch,(path,record) in sorted(committed.items()):
            epochs.append(dict(key=j['key'],epoch=epoch,path=str(path),sha256=sha_file(path),continuation_hash=record['continuation_hash']))
        rows.append(dict(key=j['key'],epoch=50,selected_epochs=result['selected_epochs'],selected_hashes=hashes,retained_bytes=tensor_bytes(state)))
        del retained,state
    assert len(rows)==20 and len(epochs)==1000
    report=dict(status='AWAITING_PARENT_STORAGE_ACCEPTANCE',root=str(attempt),owner=owner,jobs=rows,epochs=epochs,
        retained_bytes=sum(r['retained_bytes'] for r in rows),source_hash=manifest['source_hash'],
        design_bundle_sha256=manifest['design_bundle_sha256'],test_accessed=False)
    numeric(attempt/'STORAGE_HANDOFF.json',report)
    numeric(attempt/'OLD_INVENTORY_AFTER_RUN.json',assert_old_unchanged(attempt))
    return dict(status=report['status'],jobs=20,epochs=1000,test_accessed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','smoke','run','worker','handoff','tables'))
    parser.add_argument('--attempt',default=str(ATTEMPT)); parser.add_argument('--cache',default=str(DEFAULT_CACHE))
    parser.add_argument('--review'); parser.add_argument('--devices',default='0,1'); parser.add_argument('--gpu',type=int)
    parser.add_argument('--key'); parser.add_argument('--recover',action='store_true'); parser.add_argument('--smoke-case',action='store_true')
    args=parser.parse_args(); devices=[int(x) for x in args.devices.split(',')]; attempt=cycle_path(args.attempt)
    if args.command=='prepare': result=prepare(attempt,args.cache,args.review)
    elif args.command=='smoke': result=smoke(attempt,devices,args.review)
    elif args.command=='run': result=run(attempt,devices,args.review,args.recover)
    elif args.command=='handoff': result=handoff(attempt)
    elif args.command=='tables':
        from .tables import export
        result=export(attempt)
    elif args.smoke_case:
        manifest=validate_attempt(attempt); require_storage(attempt,manifest)
        assert not stop_requested() and not (attempt/'PAUSE_TRAINING').exists()
        result=smoke_case(attempt,read_json(attempt/'smoke'/args.key/'job.json'),args.gpu,manifest)
    else: result=worker(attempt,args.key,args.gpu)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
