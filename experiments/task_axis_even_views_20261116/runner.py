"""Foreground mechanical workers; frozen cycle-local RAM server, no automatic starts."""
import argparse
import copy
import gzip
import importlib.util
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from .core import train_step, synchronize
from .protocol import (ROOT, CYCLE, OLD, ATTEMPT, DEFAULT_CACHE, ARMS, DATASETS, SEEDS,
    BUNDLE, design, config, job, cycle_path, source_manifest, prepare, validate_attempt,
    stop_requested, request, alive, assert_old_unchanged, immutable, sha_file, digest,
    dataset_cache, load_protocol, make_model, setup, optimizer, cpu_copy, state_digest,
    rng_state, restore_rng, evaluate, write_json, validate_epoch_state, validate_continuation,
    effective_run_manifest, reviewed_resource_1_receipt, environment_identity,
    conservative_board_power_budgets, RESOURCE_1_RECEIPT, RESOURCE_1_POWER_SEMANTICS)
from experiments.auxiliary_objective_attribution_20261115.stage2 import core as ndp
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.protocol import tensor_bytes

MODULE = 'experiments.task_axis_even_views_20261116.runner'
_interrupted = False


def ram_usage():
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 1-values['MemAvailable']/values['MemTotal']


def gpu_status():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,power.draw,power.limit',
        '--format=csv,noheader,nounits'], text=True)
    return [dict(zip(('index', 'memory', 'total', 'power', 'limit'), map(float, row.split(','))))
            for row in raw.strip().splitlines()]


def interrupted(signum, frame):
    global _interrupted
    _interrupted = True


def install_signals():
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(name, interrupted)


def parent_alive(pid):
    return os.getppid() == pid and alive(pid)


def pipe_lost():
    # The scheduler holds this pipe open. Its death closes the last writer.
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready and os.read(sys.stdin.fileno(), 1) == b'')


def storage_status(attempt):
    attempt = cycle_path(attempt)
    info = json.loads((attempt/'guardian.json').read_text())
    old = json.loads((OLD/'guardian.json').read_text())
    assert info['pid'] != old['pid'] and info['address'] != old['address']
    assert alive(info['pid']), 'Cycle RAM process lost: PAUSE; no automatic restart'
    status = request(attempt, 'status'); assert status['pid'] == info['pid']
    return status, dict(root=str(attempt), pid=info['pid'], socket=info['address'],
                       endpoint_hash=digest([str(attempt), info['pid'], info['address']]))


def attach(attempt, owner_session):
    """Parent records its persistent foreground tool-session handle after launch."""
    assert owner_session, 'Parent-owned foreground tool-session handle is required'
    manifest = validate_attempt(attempt); status, endpoint = storage_status(attempt)
    for key, record in status['jobs'].items():
        if record['status'] != 'COMPLETE':
            validate_continuation(attempt, key, request(attempt, 'get', key=key)['state'], record)
    receipt = dict(**endpoint, owner_tool_session=str(owner_session), source_hash=manifest['source_hash'],
                   design_bundle_sha256=BUNDLE, ownership='FOREGROUND_PARENT_CURRENT_CONVERSATION', test_accessed=False)
    immutable(cycle_path(attempt)/'STORAGE_OWNER.json', receipt)
    return dict(status='ATTACHED', **receipt, retained_bytes=status['retained_bytes'])


def require_storage(attempt):
    attempt = cycle_path(attempt); status, endpoint = storage_status(attempt)
    owner = json.loads((attempt/'STORAGE_OWNER.json').read_text())
    assert all(owner[k] == v for k, v in endpoint.items()), 'RAM owner identity changed'
    manifest = effective_run_manifest(attempt)
    assert owner['source_hash'] == manifest.get('storage_bootstrap_source_hash', manifest['source_hash'])
    assert owner['design_bundle_sha256'] == BUNDLE
    for key, record in status['jobs'].items():
        j = record['job']
        assert j == job(j['dataset'], j['config']['arm'], j['seed'], manifest['source_hash']) and j['key'] == key
    return status, owner


def live_worker_pids(attempt, status):
    pids = {r.get('worker_pid') for r in status['jobs'].values()}
    ledger = cycle_path(attempt)/'launch_ledger.jsonl'
    if ledger.exists():
        pids.update(json.loads(line)['pid'] for line in ledger.read_text().splitlines())
    return sorted(pid for pid in pids if alive(pid))


def adopt_resource_1(attempt, review_path, owner_session):
    """Parent-only one-time pre-formal adoption; never starts or replaces storage."""
    attempt = cycle_path(attempt); path = attempt/RESOURCE_1_RECEIPT
    assert not path.exists(), 'RESOURCE-1 permits exactly one supersession'
    assert not stop_requested(), 'Manual stop prohibits source adoption'
    receipt = reviewed_resource_1_receipt(attempt, review_path)
    assert receipt['run_manifest']['environment'] == environment_identity()
    assert_old_unchanged(attempt)
    for name in ('dispatch.json', 'progress.json', 'launch_ledger.jsonl', 'epoch_commit_ledger.jsonl',
                 'worker_exit_ledger.jsonl', 'COMPLETE_RUNNER_REPORT.json'):
        assert not (attempt/name).exists(), ('RESOURCE-1 requires no formal activity', name)
    for name in ('jobs', 'runs', 'launch_logs'):
        assert not (attempt/name).exists() or not any((attempt/name).iterdir()), 'Formal files already exist'
    status, endpoint = storage_status(attempt); owner = receipt['evidence_reuse']['owner']
    assert owner_session == owner['owner_tool_session'], 'Original parent session is required'
    assert all(owner[k] == v for k, v in endpoint.items()), 'RAM owner identity changed'
    assert not status['jobs'] and not status['retained_states'] and not status['retained_bytes']
    assert not live_worker_pids(attempt, status), 'Live workers prohibit adoption'
    assert not stop_requested(), 'Manual stop prohibits source adoption'
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    return dict(status='ADOPTED', receipt_path=str(path), receipt_sha256=sha_file(path),
        owner=owner, effective_source_hash=receipt['run_manifest']['source_hash'], test_accessed=False)


def handoff(attempt, owner_session, accept=False):
    """Numeric live-RAM receipt; parent explicitly accepts before runner closure."""
    manifest = validate_attempt(attempt); status, owner = require_storage(attempt)
    assert owner_session == owner['owner_tool_session']
    assert not live_worker_pids(attempt, status)
    inventory = []
    for key, r in sorted(status['jobs'].items()):
        state = request(attempt, 'get', key=key)['state']
        if r['status'] != 'COMPLETE':
            validate_continuation(attempt, key, state, r)
        selected_hashes = {kind: state_digest(v) for kind, v in state.get('selected_models', {}).items()} if state else {}
        if r['status'] == 'COMPLETE' and not r.get('state_released'):
            assert selected_hashes == r['result']['selected_hashes']
            assert set(state) == {'selected_models', 'selected_epochs', 'selected_validation'}
        inventory.append(dict(key=key, status=r['status'], epoch=r['epoch'], counts=r.get('counts'),
            selected_hashes=selected_hashes, state_present=state is not None, retained_bytes=tensor_bytes(state)))
        del state
    receipt = dict(**owner, jobs=inventory, retained_states=status['retained_states'],
        retained_bytes=status['retained_bytes'], source_hash_verified=manifest['source_hash'],
        observed_at=time.time(), parent_accepted=bool(accept), old_inventories=assert_old_unchanged(attempt))
    name = 'PARENT_STORAGE_ACCEPTANCE.json' if accept else 'STORAGE_HANDOFF.json'
    if (cycle_path(attempt)/RESOURCE_1_RECEIPT).exists():
        receipt.update(effective_run_manifest_sha256=digest(manifest),
                       supersession_receipt_sha256=sha_file(cycle_path(attempt)/RESOURCE_1_RECEIPT))
        name = 'RESOURCE_1_'+name
    write_json(cycle_path(attempt)/name, receipt)
    return receipt


def serve(attempt):
    attempt = cycle_path(attempt); validate_attempt(attempt)
    assert not stop_requested() and not (attempt/'PAUSE_TRAINING').exists()
    assert not (attempt/'guardian.json').exists(), 'Existing owner receipt: attach or PAUSE; never replace'
    path = ROOT/'experiments/task_view_gradient_residual_momentum_20261116/runner.py'
    assert sha_file(path) == '5ddcb68c206c6d78ad6c3cecf59441081aa9a9ddde3233ade4af55d158b32cfa'
    from experiments.task_view_gradient_residual_momentum_20261116.runner import serve as frozen_serve
    frozen_serve(attempt)


def host_admissible(extra_bytes):
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 1-(values['MemAvailable']-extra_bytes)/values['MemTotal'] < .90


def choose_device(devices, peak_bytes, power_watts, pending=(), host_bytes=0, rows=None):
    if not host_admissible(host_bytes):
        return None
    for row in gpu_status() if rows is None else rows:
        gpu = int(row['index'])
        if gpu not in devices or gpu in pending:
            continue
        if ((row['memory']*1024**2+peak_bytes)/(row['total']*1024**2) < .85 and
                (row['power']+power_watts)/row['limit'] < .85):
            return gpu
    return None


def add_ledger(total, value):
    """Sum batch facts while preserving each actual-B denominator in batch logs."""
    total['batches'] += 1
    for name in ('B', 'entries', 'clean_entries'):
        total[name] = total.get(name, 0)+value[name]
    for cell, fields in value['cells'].items():
        target = total['cells'].setdefault(cell, {})
        for name, number in fields.items():
            if isinstance(number, list):
                target[name] = [a+b for a, b in zip(target.get(name, [0., 0.]), number)]
            else:
                target[name] = target.get(name, 0)+number


def worker(attempt, key, device, scheduler_pid):
    attempt = cycle_path(attempt); manifest = validate_attempt(attempt, require_smoke=True)
    j = json.loads((attempt/'jobs'/f'{key}.json').read_text())
    assert j == job(j['dataset'], j['config']['arm'], j['seed'], manifest['source_hash'])
    assert parent_alive(scheduler_pid) and not pipe_lost(), 'Worker requires its owning scheduler pipe/PID'
    storage, _ = require_storage(attempt)
    if stop_requested() or (attempt/'PAUSE_TRAINING').exists():
        return
    install_signals(); registered = request(attempt, 'register', key=key, job=j, pid=os.getpid())
    number = registered['attempt']; state = registered['state']; out = attempt/'runs'/key/f'attempt{number}'
    out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    previous_seconds = 0.
    lost_supervision = False; cost_status = 'FAILED'
    try:
        validate_continuation(attempt, key, state, storage['jobs'].get(key, dict(job=j, epoch=0)))
        previous_seconds = state['physical_worker_seconds'] if state else 0.
        load_started = time.perf_counter(); bundle, data = dataset_cache(j['dataset'], manifest['cache_dir'])
        data_seconds = time.perf_counter()-load_started; write_json(out/'data_manifest.json', data)
        c = j['config']; setup(j['seed'], device)
        model = make_model(bundle, load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
        loader = torch.Generator().manual_seed(j['seed']+11000); initial_hash = state_digest(model.state_dict())
        reference = json.loads((attempt/'reused_reference_index.json').read_text())
        assert initial_hash == reference['initial_hashes'][f'{j["dataset"]}/{j["seed"]}']
        selected = {}; selected_epochs = {}; selected_validation = {}; common_validation = {}
        best = {k: -float('inf') for k in ('online', 'fixed_ema')}
        counts = dict(optimizer_updates=0, ema_updates=0); step = 0; epoch0 = 0
        peak_allocated = peak_reserved = 0
        if state:
            assert state['job'] == j and state['initial_hash'] == initial_hash
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); opt.load_state_dict(state['optimizer'])
            loader.set_state(state['loader']); restore_rng(state['rng'], device)
            epoch0 = state['epoch']; step = state['global_step']; selected = state['selected_models']
            selected_epochs = state['selected_epochs']; selected_validation = state['selected_validation']; best = state['best']
            counts = state['counts']; common_validation = state['common_validation']
            peak_allocated = state['peak_allocated_bytes']; peak_reserved = state['peak_reserved_bytes']
        del state, registered
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(epoch0+1, 51):
            lost_supervision |= not parent_alive(scheduler_pid) or pipe_lost() or _interrupted
            if stop_requested() or (attempt/'PAUSE_TRAINING').exists() or lost_supervision:
                cost_status = 'PAUSED'
                request(attempt, 'paused', key=key, reason='MANUAL_STOP' if stop_requested() else 'SUPERVISION_LOST' if lost_supervision else 'TECHNICAL_PAUSE')
                return
            tick = time.perf_counter(); model.train(); order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            order_hash = state_digest(bundle.train_row_ids[order])
            assert order_hash == reference['epoch_order_hashes'][f'{j["dataset"]}/{j["seed"]}/{epoch}']
            sums = {k: 0. for k in ('loss', 'clean_loss', 'view_loss', 'probe_seconds', 'construction_seconds', 'training_seconds')}
            ledger = dict(batches=0, cells={}); batch_path = out/f'batches_epoch_{epoch:02d}.jsonl.gz'
            with gzip.open(batch_path, 'xt', compresslevel=3) as stream:
                for batch, offset in enumerate(range(0, len(order), 40960), 1):
                    ids = order[offset:offset+40960]; step += 1
                    x = torch.as_tensor(bundle.train_x[ids], device=device).T
                    y = torch.as_tensor(bundle.train_y[ids], device=device)
                    context = dict(dataset=j['dataset'], seed=j['seed'], epoch=epoch, batch=batch,
                        global_step=step, row_order_hash=state_digest(bundle.train_row_ids[ids]),
                        diagnose=batch == 1 and epoch in (1, 10, 25, 40, 50))
                    value = train_step(model, opt, ema, x, y, c, context)
                    assert value['optimizer_updates'] == value['ema_updates'] == 1
                    for name in sums:
                        sums[name] += value[name]*(len(ids) if name in ('loss', 'clean_loss', 'view_loss') else 1)
                    for name in counts:
                        counts[name] += value[name]
                    add_ledger(ledger, value['ledger'])
                    stream.write(json.dumps(dict(dataset=j['dataset'], arm=c['arm'], seed=j['seed'],
                        epoch=epoch, batch=batch, global_step=step, **value), separators=(',', ':'), allow_nan=False)+'\n')
                    if batch == 1:
                        request(attempt, 'allocated', key=key, peak_memory_bytes=torch.cuda.max_memory_reserved(device))
                    lost_supervision |= not parent_alive(scheduler_pid) or pipe_lost() or _interrupted
                    del x, y, value
            validation_started = time.perf_counter(); validation = {}; validation_cost = {}
            for kind, estimator in (('online', model), ('fixed_ema', ema)):
                values, cost = evaluate(estimator, bundle.val_x, bundle.val_y, device)
                validation[kind] = values; validation_cost[kind] = cost
                score = values['macro_auc']; assert score is not None
                if score > best[kind]:
                    best[kind] = score; selected_epochs[kind] = epoch
                    selected[kind] = cpu_copy(estimator.state_dict()); selected_validation[kind] = values
            if epoch in (10, 30, 50):
                common_validation[str(epoch)] = validation
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            record = dict(epoch=epoch, global_step=step, seconds=time.perf_counter()-tick,
                **{k: v/len(order) if k in ('loss', 'clean_loss', 'view_loss') else v for k, v in sums.items()},
                validation_seconds=time.perf_counter()-validation_started, validation_cost=validation_cost,
                validation=validation, objective_ledger=ledger, selected_epochs=selected_epochs.copy(),
                selected_hashes={k: state_digest(v) for k, v in selected.items()}, counts=counts.copy(),
                model_hash=state_digest(model.state_dict()), ema_hash=state_digest(ema.state_dict()),
                optimizer_hash=state_digest(opt.state_dict()), rng_hash=state_digest(rng_state(device)),
                loader_hash=state_digest(loader.get_state()), epoch_order_hash=order_hash,
                support_log=str(batch_path.relative_to(attempt)), support_log_sha256=sha_file(batch_path),
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                ram_usage=ram_usage(), source_hash=manifest['source_hash'], design_bundle_sha256=BUNDLE,
                test_accessed=False)
            epoch_file = out/f'epoch_{epoch:02d}.json'
            snapshot = cpu_copy(dict(job=j, model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                rng=rng_state(device), loader=loader.get_state(), epoch=epoch, global_step=step,
                selected_models=selected, selected_epochs=selected_epochs, selected_validation=selected_validation,
                common_validation=common_validation, best=best, counts=counts, initial_hash=initial_hash,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                physical_worker_seconds=previous_seconds+time.perf_counter()-started))
            # Hash only the payload sent to RAM; the fingerprint stays in the numerical receipt.
            record['continuation_hash'] = state_digest(snapshot)
            validate_epoch_state(snapshot, record)
            write_json(epoch_file, record)
            ack = request(attempt, 'commit', key=key, state=snapshot, epoch_file=str(epoch_file.resolve()))
            assert ack['status'] == 'RAM_COMMITTED' and ack['epoch'] == epoch
            del snapshot
            write_json(out/'progress.json', dict(epoch=epoch, global_step=step, key=key, status='RAM_COMMITTED',
                attempt_worker_seconds=time.perf_counter()-started, test_accessed=False))
            request(attempt, 'allocated', key=key, peak_memory_bytes=peak_reserved)
            lost_supervision |= not parent_alive(scheduler_pid) or pipe_lost() or _interrupted
            if epoch < 50 and (stop_requested() or (attempt/'PAUSE_TRAINING').exists() or lost_supervision):
                cost_status = 'PAUSED'
                request(attempt, 'paused', key=key, reason='MANUAL_STOP' if stop_requested() else 'SUPERVISION_LOST' if lost_supervision else 'TECHNICAL_PAUSE')
                return
        expected = 50*c['steps_per_epoch']; assert step == expected == counts['optimizer_updates'] == counts['ema_updates']
        result = dict(job=j, epochs=50, global_step=step, initial_hash=initial_hash,
            selected_epochs=selected_epochs, selected_validation=selected_validation,
            selected_hashes={k: state_digest(v) for k, v in selected.items()}, common_validation=common_validation,
            final_hashes=dict(online=state_digest(model.state_dict()), fixed_ema=state_digest(ema.state_dict())),
            optimizer_hash=state_digest(opt.state_dict()), counts=counts,
            worker_seconds=previous_seconds+time.perf_counter()-started, data_load_seconds=data_seconds,
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, test_accessed=False)
        assert request(attempt, 'complete', key=key, result=result)['status'] == 'COMPLETE'
        cost_status = 'COMPLETE'
    except BaseException as error:
        failure = dict(key=key, attempt=number, error=repr(error), traceback=traceback.format_exc(),
            oom=isinstance(error, torch.OutOfMemoryError), seconds=time.perf_counter()-started, test_accessed=False)
        write_json(out/'failure.json', failure); (attempt/'PAUSE_TRAINING').touch(exist_ok=True)
        request(attempt, 'failed', key=key, failure=failure)
        raise
    finally:
        write_json(out/'worker_cost.json', dict(key=key, attempt=number, status=cost_status,
            seconds=time.perf_counter()-started, test_accessed=False))


def schedule(attempt, devices, review_path=None):
    attempt = cycle_path(attempt); manifest = validate_attempt(attempt, review_path, require_smoke=True)
    require_storage(attempt); install_signals(); supervisor_pid = os.getppid()
    smoke_record = json.loads((attempt/'SMOKE_REPORT.json').read_text())
    jobs = [job(d, a, s, manifest['source_hash']) for s in SEEDS for d in DATASETS for a in ARMS]
    immutable(attempt/'dispatch.json', dict(jobs=jobs, trajectories=30, test_accessed=False))
    for j in jobs:
        immutable(attempt/'jobs'/f'{j["key"]}.json', j)
    initial, _ = require_storage(attempt)
    assert not live_worker_pids(attempt, initial), 'A live worker belongs to an existing foreground scheduler'
    processes = {}; known_gpus = {}; process_started = {}; exited = set(); jobs_by_key = {j['key']: j for j in jobs}
    power_budgets = conservative_board_power_budgets(smoke_record)
    reservations = {d: dict(bytes=r['peak_reserved_bytes'], power=power_budgets[d],
                           power_semantics=RESOURCE_1_POWER_SEMANTICS,
                           host=r['host_admission_bytes'], transient=r['transient_snapshot_bytes'])
                    for d, r in smoke_record['datasets'].items()}
    started = time.perf_counter(); manual = False; technical = False; failure_keys = set()
    try:
        while True:
            manual |= stop_requested()
            technical |= _interrupted or not parent_alive(supervisor_pid) or (attempt/'PAUSE_TRAINING').exists()
            status, _ = require_storage(attempt); records = status['jobs']; active = []; pending = []
            for j in jobs:
                key = j['key']; record = records.get(key, {}); process = processes.get(key)
                if process:
                    code = process.poll()
                    if code is not None and key not in exited:
                        with (attempt/'worker_exit_ledger.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(key=key, pid=process.pid, returncode=code,
                                seconds=time.perf_counter()-process_started[key], observed_at=time.time()))+'\n')
                        exited.add(key)
                    if code is not None and process.stdin and not process.stdin.closed:
                        process.stdin.close()
                if record.get('status') == 'COMPLETE':
                    continue
                if process and process.poll() is None:
                    active.append(key)
                elif record.get('status') in ('FAILED', 'RUNNING') or (process and process.returncode != 0):
                    failure_keys.add(key); technical = True
                elif process:  # A paused/zero-exit worker is never silently retried in this invocation.
                    technical = True
                else:
                    pending.append(j)
            if technical:
                (attempt/'PAUSE_TRAINING').touch(exist_ok=True)
            for key in active:
                actual = records.get(key, {}).get('peak_memory_bytes', 0)
                reservation = reservations[jobs_by_key[key]['dataset']]
                if actual > reservation['bytes']:
                    reservation['bytes'] = actual
            rows = gpu_status()
            # Actual resource excursions stop admission until safe; each card remains independent.
            pending_cards = {known_gpus[k] for k in active if not records.get(k, {}).get('allocated')}
            host_pending = sum(reservations[jobs_by_key[k]['dataset']]['host'] for k in active
                               if not records.get(k, {}).get('allocated'))
            host_transient = sum(reservations[jobs_by_key[k]['dataset']]['transient'] for k in active)
            complete = sum(r.get('status') == 'COMPLETE' for r in records.values())
            progress = dict(status='MANUAL_STOPPING' if manual else 'PAUSING' if technical else 'RUNNING',
                complete=complete, total=30, pending=len(pending), failed=sorted(failure_keys),
                active=[dict(key=k, pid=processes[k].pid, gpu=known_gpus[k], epoch=records.get(k, {}).get('epoch', 0)) for k in active],
                seconds=time.perf_counter()-started, ram_usage=ram_usage(), gpu_status=rows,
                reservations=reservations, retained_states=status['retained_states'], retained_bytes=status['retained_bytes'],
                supervisor_pid=supervisor_pid, scheduler_pid=os.getpid(), test_accessed=False)
            write_json(attempt/'progress.json', progress)
            if not active and (manual or technical or complete == 30):
                progress['status'] = 'USER_STOPPED' if manual else 'PAUSED' if technical else 'COMPLETE'
                progress['old_inventories'] = assert_old_unchanged(attempt)
                write_json(attempt/'progress.json', progress)
                if complete == 30:
                    assert sum(r['result']['global_step'] for r in records.values()) == 51750
                    report = dict(progress, coverage=jobs, source_hash=manifest['source_hash'], design_bundle_sha256=BUNDLE,
                        worker_seconds_sum=sum(r['result']['worker_seconds'] for r in records.values()),
                        state_retention_keys=sorted(records), smoke_receipt_sha256=sha_file(attempt/'SMOKE_REPORT.json'),
                        storage_handoff_pending=True, owner=require_storage(attempt)[1])
                    write_json(attempt/'COMPLETE_RUNNER_REPORT.json', report)
                return progress
            if not manual and not technical:
                for j in pending:
                    reserve = reservations[j['dataset']]
                    gpu = choose_device(devices, reserve['bytes'], reserve['power'], pending_cards,
                        reserve['host']+host_pending+host_transient, rows)
                    if gpu is None:
                        continue
                    if stop_requested() or _interrupted or not parent_alive(supervisor_pid):
                        break
                    require_storage(attempt)
                    logs = attempt/'launch_logs'; logs.mkdir(exist_ok=True)
                    with (logs/f'{j["key"]}.log').open('ab') as log:
                        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, '_worker',
                            '--attempt', str(attempt), '--key', j['key'], '--device', f'cuda:{gpu}',
                            '--scheduler-pid', str(os.getpid())], cwd=ROOT, stdin=subprocess.PIPE,
                            stdout=log, stderr=log)  # Same session; owned pipe closes if this scheduler dies.
                    processes[j['key']] = process; known_gpus[j['key']] = gpu; pending_cards.add(gpu)
                    process_started[j['key']] = time.perf_counter()
                    host_pending += reserve['host']
                    with (attempt/'launch_ledger.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(key=j['key'], pid=process.pid, gpu=gpu,
                            scheduler_pid=os.getpid(), time=time.time(), reservation=reserve.copy(),
                            stop_controls_checked=True, detached=False))+'\n')
                    # Re-sample actual load before any additional admission; pending reserves cover startup.
                    rows = gpu_status()
            time.sleep(2)
    finally:
        # EOF requests a safe atomic pause, and this foreground scheduler joins every owned worker.
        for process in processes.values():
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        for process in processes.values():
            process.wait()


def step_signature(model, opt, ema, device, loader):
    """All parameters/buffers, moments/counters, supplied gradients and RNG identity."""
    return dict(model=state_digest(model.state_dict()), optimizer=state_digest(opt.state_dict()),
        ema=state_digest(ema.state_dict()), gradients=state_digest({n: p.grad for n, p in model.named_parameters()}),
        none_mask={n: p.grad is None for n, p in model.named_parameters()},
        adam_steps={n: int(opt.state[p]['step'].item()) if p in opt.state else None for n, p in model.named_parameters()},
        rng=state_digest(rng_state(device)), loader=state_digest(loader.get_state()))


def historical_ndp_step():
    from .protocol import REFERENCE
    path = REFERENCE/'experiments/auxiliary_objective_attribution_20261115/stage2/core.py'
    assert sha_file(path) == sha_file(Path(ndp.__file__))
    spec = importlib.util.spec_from_file_location(
        'experiments.auxiliary_objective_attribution_20261115.stage2._historical_identity', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.train_step


def one_sided_source_step(model, opt, ema, x, y, c, context, row_ids):
    # Identity-only adapter, never a new arm or a rewritten objective.
    return ndp.train_step(model, opt, ema, x, y, c, context, row_ids)


def source_identities(bundle, dataset, device, attempt, peak, power_increment):
    supervisor_pid = os.getppid()
    loader = torch.Generator().manual_seed(13022)
    order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
    tail = len(order) % 40960; assert tail
    reference_step = historical_ndp_step(); observations = []
    for name, ids in (('full', order[:40960]), ('tail', order[-tail:])):
        signatures = {}; source_loss = None
        for mode in ('SOURCE-CLEAN',)+tuple('NULL/'+a for a in ARMS)+('SOURCE-NDP-A', 'DELEGATED-NDP-A'):
            gpu = int(str(device).split(':')[-1])
            while choose_device([gpu], peak, power_increment) is None:
                assert not stop_requested() and not _interrupted and parent_alive(supervisor_pid)
                time.sleep(2)
            assert not stop_requested() and not _interrupted and parent_alive(supervisor_pid)
            require_storage(attempt)
            setup(2022, device); c = config(dataset, mode[5:] if mode.startswith('NULL/') else ARMS[0])
            model = make_model(bundle, load_protocol()['datasets'][dataset]['source']).to(device)
            ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
            x = torch.as_tensor(bundle.train_x[ids], device=device).T
            y = torch.as_tensor(bundle.train_y[ids], device=device)
            context = dict(dataset=dataset, seed=2022, epoch=1,
                batch=1 if name == 'full' else (len(order)+40959)//40960,
                row_order_hash=state_digest(bundle.train_row_ids[ids]), diagnose=False)
            with PowerSamples(gpu) as samples:
                if mode == 'SOURCE-CLEAN':
                    source_loss = source_clean_step(model, opt, ema, x, y, c)
                elif mode.startswith('NULL/'):
                    def zero_views(m, bx, by, alpha):
                        return ndp.make_views(m, bx, by, alpha, force_zero=True)
                    with patch('experiments.task_axis_even_views_20261116.core.make_views', side_effect=zero_views):
                        value = train_step(model, opt, ema, x, y, c, context)
                    assert value['zero_delta_canonical'] and value['loss'] == source_loss
                    assert value['ledger']['denominator'] == 2*len(ids)
                elif mode == 'SOURCE-NDP-A':
                    value = reference_step(model, opt, ema, x, y, c, context, bundle.train_row_ids[ids])
                else:
                    value = one_sided_source_step(model, opt, ema, x, y, c, context, bundle.train_row_ids[ids])
            assert all(r['power']/r['limit'] < .85 and r['memory']/r['total'] < .85 for r in samples.rows)
            signatures[mode] = step_signature(model, opt, ema, device, loader)
            del model, ema, opt, x, y
            torch.cuda.empty_cache()
        assert all(signatures['SOURCE-CLEAN'] == signatures['NULL/'+a] for a in ARMS)
        assert signatures['SOURCE-NDP-A'] == signatures['DELEGATED-NDP-A']
        observations.append(dict(slice=name, rows=len(ids), row_order_hash=state_digest(bundle.train_row_ids[ids]),
            null_source_exact=True, one_sided_source_exact=True, signatures=signatures))
    return dict(status='PASS', full_rows=40960, tail_rows=tail, observations=observations)


class PowerSamples:
    """Sample actual board draw around a measured training step; no training delegation."""
    def __init__(self, gpu):
        self.gpu = gpu; self.rows = []; self.done = threading.Event(); self.error = None

    def sample(self):
        self.rows.append(next(r for r in gpu_status() if int(r['index']) == self.gpu))

    def loop(self):
        try:
            while not self.done.wait(.1):
                self.sample()
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.sample(); self.thread = threading.Thread(target=self.loop)
        self.thread.start(); return self

    def __exit__(self, *args):
        self.done.set(); self.thread.join(); self.sample()
        if self.error:
            raise self.error


def smoke(attempt, device, review_path=None):
    attempt = cycle_path(attempt); manifest = validate_attempt(attempt, review_path)
    require_storage(attempt); install_signals(); supervisor_pid = os.getppid()
    assert str(device).startswith('cuda:') and torch.cuda.is_available()
    assert not (attempt/'SMOKE_REPORT.json').exists(), 'No automatic smoke retry or receipt overwrite'
    gpu = int(str(device).split(':')[-1]); output = dict(status='RUNNING', source_hash=manifest['source_hash'],
        design_bundle_sha256=BUNDLE, datasets={}, test_accessed=False, old_before=assert_old_unchanged(attempt))
    write_json(attempt/'SMOKE_REPORT.json', output)
    try:
        for dataset in DATASETS:
            bundle, _ = dataset_cache(dataset, manifest['cache_dir']); arms = []; snapshot_bytes = 0; peak_power_increment = 0.
            planning = int((4.5 if dataset == 'movielens' else 5.5)*1024**3)
            for arm in ARMS:
                while choose_device([gpu], max([planning]+[a['peak_reserved_bytes'] for a in arms]), peak_power_increment) is None:
                    assert not stop_requested() and not _interrupted and parent_alive(supervisor_pid), 'Smoke supervision/stop'
                    require_storage(attempt); time.sleep(2)
                assert not stop_requested() and not _interrupted and parent_alive(supervisor_pid)
                setup(2022, device); c = config(dataset, arm)
                model = make_model(bundle, load_protocol()['datasets'][dataset]['source']).to(device)
                historical = json.loads((attempt/'reused_reference_index.json').read_text())
                assert state_digest(model.state_dict()) == historical['initial_hashes'][f'{dataset}/2022']
                ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
                loader = torch.Generator().manual_seed(13022)
                ids = torch.randperm(len(bundle.train_y), generator=loader).numpy()[:40960]
                x = torch.as_tensor(bundle.train_x[ids], device=device).T; y = torch.as_tensor(bundle.train_y[ids], device=device)
                before_rng = state_digest(rng_state(device)); before_loader = state_digest(loader.get_state())
                torch.cuda.reset_peak_memory_stats(device); synchronize(device); started = time.perf_counter()
                with PowerSamples(gpu) as power:
                    value = train_step(model, opt, ema, x, y, c, dict(dataset=dataset, seed=2022,
                        epoch=1, batch=1, global_step=1, diagnose=True, row_order_hash=state_digest(bundle.train_row_ids[ids])))
                synchronize(device)
                assert before_rng == state_digest(rng_state(device)) and before_loader == state_digest(loader.get_state())
                assert value['optimizer_updates'] == value['ema_updates'] == 1
                assert value['ledger']['B'] == 40960 and value['ledger']['denominator'] == 81920
                assert all(torch.isfinite(p).all() for p in model.parameters())
                assert all(torch.isfinite(p).all() for p in ema.parameters())
                maximum_power = max(r['power'] for r in power.rows)
                increment = max(0., maximum_power-power.rows[0]['power']); peak_power_increment = max(peak_power_increment, increment)
                assert maximum_power/power.rows[0]['limit'] < .85, 'Measured board power exceeded admission gate'
                value.update(arm=arm, rows=len(ids), precision='float32', seconds=time.perf_counter()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    power_before_watts=power.rows[0]['power'], peak_board_power_watts=maximum_power,
                    power_increment_watts=increment, power_samples=len(power.rows), ram_usage=ram_usage())
                assert all(r['memory']/r['total'] < .85 for r in power.rows) and host_admissible(0)
                # Measure a realistic continuation plus two selected states through frozen BytesIO IPC.
                assert host_admissible(4*6*tensor_bytes(model.state_dict())), 'Host RAM cannot hold transient smoke snapshots'
                snapshot = cpu_copy(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                    selected_models=dict(online=model.state_dict(), fixed_ema=ema.state_dict()),
                    loader=loader.get_state(), rng=rng_state(device)))
                snapshot_bytes = max(snapshot_bytes, tensor_bytes(snapshot))
                response = request(attempt, 'roundtrip', value=snapshot)
                assert state_digest(response['value']) == state_digest(snapshot)
                del snapshot, response
                arms.append(value); del model, ema, opt, x, y; torch.cuda.empty_cache()
            peak = max(a['peak_reserved_bytes'] for a in arms)
            identities = source_identities(bundle, dataset, device, attempt, peak, peak_power_increment)
            peak = max(peak, torch.cuda.max_memory_reserved(device))
            # Host occupancy already includes old/new owner states; reserve pending worker RSS and transient copies.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            output['datasets'][dataset] = dict(arms=arms, identities=identities, peak_reserved_bytes=peak,
                power_increment_watts=peak_power_increment, peak_host_rss_bytes=rss,
                snapshot_bytes=snapshot_bytes, transient_snapshot_bytes=3*snapshot_bytes,
                host_admission_bytes=rss+3*snapshot_bytes)
            write_json(attempt/'SMOKE_REPORT.json', output); del bundle
        output.update(status='PASS', old_after=assert_old_unchanged(attempt))
    except BaseException as error:
        output.update(status='USER_STOPPED' if stop_requested() else 'PAUSED', error=repr(error), traceback=traceback.format_exc())
        (attempt/'PAUSE_TRAINING').touch(exist_ok=True)
        raise
    finally:
        write_json(attempt/'SMOKE_REPORT.json', output)
    return output


def recover(attempt, key, review_path=None):
    attempt = cycle_path(attempt); validate_attempt(attempt, review_path, require_smoke=True)
    assert key and not stop_requested()
    status, _ = require_storage(attempt)
    assert not live_worker_pids(attempt, status)
    record = status['jobs'][key]; assert record['status'] in ('FAILED', 'RUNNING', 'PAUSED')
    state = request(attempt, 'get', key=key)['state']
    validate_continuation(attempt, key, state, record)
    del state
    # This is the explicit recovery API. No resets, changed keys, source substitution or retry loop.
    if record['status'] in ('FAILED', 'RUNNING'):
        answer = request(attempt, 'recover', key=key)
    else:
        assert not any(r['status'] in ('FAILED', 'RUNNING') for r in status['jobs'].values())
        answer = dict(status='RECOVERY_READY', key=key, epoch=record['epoch'])
        (attempt/'PAUSE_TRAINING').unlink(missing_ok=True)
    return dict(answer, test_accessed=False)


def release(attempt, keys):
    assert keys, 'Release requires explicit parent-designated new-cycle keys'
    validate_attempt(attempt); status, _ = require_storage(attempt)
    assert not live_worker_pids(attempt, status)
    for key in keys:
        assert key in status['jobs'] and status['jobs'][key]['status'] == 'COMPLETE'
        assert request(attempt, 'release', key=key)['status'] == 'RELEASED'
    receipt = dict(keys=keys, time=time.time(), test_accessed=False)
    with (cycle_path(attempt)/'release_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt)+'\n')
    return dict(status='RELEASED', **receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', '_guardian', 'attach', 'smoke', 'run', '_worker',
        'status', 'handoff', 'accept-handoff', 'tables', 'recover', 'release', 'shutdown', 'adopt-resource-1'))
    parser.add_argument('--attempt', type=Path, default=ATTEMPT)
    parser.add_argument('--review', type=Path); parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--devices', default='0,1')
    parser.add_argument('--key', action='append'); parser.add_argument('--owner-session')
    parser.add_argument('--scheduler-pid', type=int)
    args = parser.parse_args(); attempt = cycle_path(args.attempt)
    if args.command == '_guardian':
        serve(attempt); return
    if args.command == '_worker':
        assert len(args.key or []) == 1 and args.scheduler_pid
        worker(attempt, args.key[0], args.device, args.scheduler_pid); return
    if args.command == 'prepare':
        assert args.review; value = prepare(attempt, args.cache_dir, args.review)
    elif args.command == 'attach':
        value = attach(attempt, args.owner_session)
    elif args.command == 'adopt-resource-1':
        assert args.review and args.owner_session
        value = adopt_resource_1(attempt, args.review, args.owner_session)
    elif args.command == 'smoke':
        value = smoke(attempt, args.device, args.review)
    elif args.command == 'run':
        value = schedule(attempt, [int(v) for v in args.devices.split(',')], args.review)
    elif args.command == 'tables':
        from .tables import export
        value = export(attempt)
    elif args.command in ('handoff', 'accept-handoff'):
        value = handoff(attempt, args.owner_session, accept=args.command == 'accept-handoff')
    elif args.command == 'recover':
        assert len(args.key or []) == 1; value = recover(attempt, args.key[0], args.review)
    elif args.command == 'release':
        value = release(attempt, args.key)
    elif args.command == 'shutdown':
        status, _ = require_storage(attempt)
        assert not status['retained_states'] and not live_worker_pids(attempt, status)
        value = request(attempt, 'shutdown')
    else:
        status, owner = require_storage(attempt)
        value = dict(owner=owner, pid=status['pid'], retained_bytes=status['retained_bytes'],
            retained_states=status['retained_states'], jobs={k: {n: r.get(n) for n in
                ('epoch', 'status', 'worker_pid', 'counts', 'state_released')} for k, r in status['jobs'].items()}, test_accessed=False)
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
