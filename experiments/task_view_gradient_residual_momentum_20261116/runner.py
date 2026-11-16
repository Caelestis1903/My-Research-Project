"""Reviewed full50 workers with epoch-atomic RAM continuation and resource admission."""
import argparse
import copy
import gzip
import json
import os
import resource
import secrets
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
import torch
from .core import ResidualChannel, source_clean_step, train_step, synchronize
from .protocol import (ROOT, ARTIFACT, DEFAULT_CACHE, REFERENCE, SEEDS, DATASETS, ARMS,
    config, job, source_manifest, immutable, dataset_cache, prepare, validate_attempt, stop_requested,
    load_protocol, make_model, setup, cpu_copy, state_digest, rng_state, restore_rng,
    evaluate, write_json, optimizer, request, alive, sha_file)
from experiments.auxiliary_objective_attribution_20261115.runner import (
    choose_device as audited_choose_device, ram_usage, gpu_status)
from experiments.conditional_information_momentum_20261102_v3.protocol import send_memory, receive_memory
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.protocol import tensor_bytes

MODULE = 'experiments.task_view_gradient_residual_momentum_20261116.runner'


def choose_device(devices, peak, allocating=(), host_bytes=0):
    info = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:'))}
    if 1-(info['MemAvailable']-host_bytes)/info['MemTotal'] >= .9:
        return None
    for row in gpu_status():
        gpu = int(row['index'])
        if gpu in devices and (row['memory']*1024**2+peak)/(row['total']*1024**2) < .85:
            if audited_choose_device([gpu], peak, allocating) == gpu:
                return gpu
    return None


def ensure_guardian(attempt):
    """Same audited RAM IPC; local adapter adds pause and selected-state compaction."""
    attempt = Path(attempt).resolve(); attempt.mkdir(parents=True, exist_ok=True)
    if (attempt/'guardian.json').exists():
        try:
            return request(attempt, 'status')
        except (OSError, EOFError) as error:
            raise RuntimeError('RAM guardian lost; PAUSED. Rerun affected frozen trajectories in a new attempt.') from error
    with (attempt/'guardian.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, '-m', MODULE, '_guardian', '--attempt', str(attempt)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(600):
        if (attempt/'guardian.json').exists():
            return request(attempt, 'status')
        if process.poll() is not None:
            break
        time.sleep(.1)
    raise RuntimeError('RAM guardian startup failed; see guardian.log')


def serve(attempt):
    attempt = Path(attempt).resolve(); auth = secrets.token_bytes(32)
    address = f'/tmp/tvgrm-{os.getuid()}-{secrets.token_hex(8)}.sock'
    records = {}; shutdown = False
    with Listener(address, family='AF_UNIX', authkey=auth) as server:
        write_json(attempt/'guardian.json', dict(address=address, authkey=auth.hex(), pid=os.getpid()))
        os.chmod(attempt/'guardian.json', 0o600)
        while not shutdown:
            with server.accept() as connection:
                try:
                    msg = receive_memory(connection); command = msg['command']; key = msg.get('key')
                    if command == 'status':
                        answer = dict(pid=os.getpid(), retained_states=sum('state' in r for r in records.values()),
                            retained_bytes=sum(tensor_bytes(r.get('state')) for r in records.values()),
                            jobs={k: {n: v for n, v in r.items() if n != 'state'} for k, r in records.items()})
                    elif command == 'roundtrip':
                        answer = dict(value=msg['value'])
                    elif command == 'register':
                        r = records.setdefault(key, dict(job=msg['job'], epoch=0, attempts=0, status='REGISTERED'))
                        assert r['job'] == msg['job'] and r['status'] in ('REGISTERED', 'PAUSED') and not alive(r.get('worker_pid'))
                        assert r['epoch'] == 0 or 'state' in r, 'Committed RAM state was released; frozen rerun required'
                        r.update(worker_pid=msg['pid'], status='RUNNING', attempts=r['attempts']+1, allocated=False)
                        answer = dict(attempt=r['attempts'], state=r.get('state'))
                    elif command == 'recover':
                        assert not stop_requested(), 'Manual stop prohibits recovery'
                        assert not any(alive(r.get('worker_pid')) for r in records.values()), 'Wait for live workers to commit and stop'
                        r = records[key]
                        assert r['status'] in ('FAILED', 'RUNNING'), 'Only failed/dead RUNNING jobs need recovery'
                        assert r['epoch'] == 0 or ('state' in r and r['state']['epoch'] == r['epoch']), 'Committed RAM state lost; frozen rerun required'
                        event = dict(key=key, previous_status=r['status'], epoch=r['epoch'], attempts=r['attempts'], time=time.time())
                        r.update(status='REGISTERED', worker_pid=None, allocated=False)
                        unresolved = [k for k, v in records.items() if v['status'] in ('FAILED', 'RUNNING')]
                        if not unresolved:
                            (attempt/'PAUSE_TRAINING').unlink(missing_ok=True)
                        with (attempt/'recovery_ledger.jsonl').open('a') as stream:
                            stream.write(json.dumps(event)+'\n')
                        answer = dict(status='RECOVERY_READY', **event, unresolved=unresolved, test_accessed=False)
                    elif command == 'allocated':
                        records[key].update(allocated=True, peak_memory_bytes=msg['peak_memory_bytes'])
                        answer = dict(status='ALLOCATED')
                    elif command == 'commit':
                        r = records[key]; state = msg['state']
                        assert state['epoch'] == r['epoch']+1
                        r.update(state=state, epoch=state['epoch'], counts=state['counts'], last_commit=time.time())
                        with (attempt/'epoch_commit_ledger.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(key=key, epoch=r['epoch'], attempt=r['attempts'],
                                epoch_file=msg['epoch_file'], time=time.time(), status='RAM_COMMITTED'))+'\n')
                        answer = dict(status='RAM_COMMITTED', epoch=r['epoch'])
                    elif command == 'complete':
                        r = records[key]; assert r['epoch'] == 50
                        r.update(status='COMPLETE', result=msg['result'], worker_pid=None)
                        # Completed trajectories need selected states, not final/optimizer/RNG duplicates.
                        r['state'] = {n: r['state'][n] for n in ('selected_models', 'selected_epochs', 'selected_validation')}
                        write_json(attempt/'runs'/key/'result.json', msg['result'])
                        answer = dict(status='COMPLETE')
                    elif command == 'paused':
                        records[key].update(status='PAUSED', worker_pid=None, pause_reason=msg['reason'])
                        answer = dict(status='PAUSED')
                    elif command == 'failed':
                        records[key].update(status='FAILED', worker_pid=None, failure=msg['failure'])
                        answer = dict(status='FAILED')
                    elif command == 'get':
                        answer = dict(state=records[key].get('state'), result=records[key].get('result'))
                    elif command == 'release':
                        assert not alive(records[key].get('worker_pid'))
                        records[key].pop('state', None); records[key]['state_released'] = True
                        answer = dict(status='RELEASED')
                    elif command == 'shutdown':
                        assert not any(alive(r.get('worker_pid')) or 'state' in r for r in records.values())
                        shutdown = True; answer = dict(status='SHUTDOWN')
                    else:
                        raise ValueError(command)
                    send_memory(connection, answer)
                except (EOFError, BrokenPipeError, ConnectionResetError):
                    continue
                except Exception as error:
                    try:
                        send_memory(connection, dict(error=f'{type(error).__name__}: {error}'))
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        pass
    Path(address).unlink(missing_ok=True)


def worker(attempt, key, device):
    attempt = Path(attempt); j = json.loads((attempt/'jobs'/f'{key}.json').read_text())
    manifest = validate_attempt(attempt, None, require_smoke=True)
    assert j['source_hash'] == manifest['source_hash']
    if stop_requested() or (attempt/'PAUSE_TRAINING').exists():
        return
    registration = request(attempt, 'register', key=key, job=j, pid=os.getpid())
    number = registration['attempt']; state = registration['state']
    out = attempt/'runs'/key/f'attempt{number}'; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); previous_seconds = state['physical_worker_seconds'] if state else 0.
    try:
        load_started = time.perf_counter()
        bundle, data = dataset_cache(j['dataset'], manifest['cache_dir'])
        data_seconds = time.perf_counter()-load_started; write_json(out/'data_manifest.json', data)
        c = j['config']; setup(j['seed'], device)
        model = make_model(bundle, load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
        channel = ResidualChannel(model, opt, c); loader = torch.Generator().manual_seed(j['seed']+11000)
        initial_hash = state_digest(model.state_dict())
        selected = {}; selected_epochs = {}; selected_validation = {}; common_validation = {}
        best = {kind: -float('inf') for kind in ('online', 'fixed_ema')}
        counts = dict(optimizer_updates=0, ema_updates=0); step = 0; epoch0 = 0
        if state:
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); opt.load_state_dict(state['optimizer'])
            channel.load_state_dict(state['residual']); loader.set_state(state['loader']); restore_rng(state['rng'], device)
            epoch0 = state['epoch']; step = state['global_step']; selected = state['selected_models']
            selected_epochs = state['selected_epochs']; selected_validation = state['selected_validation']; best = state['best']
            counts = state['counts']; common_validation = state['common_validation']; initial_hash = state['initial_hash']
        del state, registration
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(epoch0+1, 51):
            if stop_requested() or (attempt/'PAUSE_TRAINING').exists():
                request(attempt, 'paused', key=key, reason='MANUAL_STOP' if stop_requested() else 'TECHNICAL_PAUSE')
                return
            tick = time.perf_counter(); model.train()
            order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            sums = dict(clean_loss=0., view_loss=0., seconds=0., probe_seconds=0., construction_seconds=0.)
            ledger = dict(batches=0, B=0, clean_entries=0, view_entries=0, cells={})
            cap_batches = 0; batch_path = out/f'batches_epoch_{epoch:02d}.jsonl.gz'
            with gzip.open(batch_path, 'xt', compresslevel=3) as batch_log:
                for batch, offset in enumerate(range(0, len(order), 40960), 1):
                    ids = order[offset:offset+40960]; step += 1
                    x = torch.as_tensor(bundle.train_x[ids], device=device).T
                    y = torch.as_tensor(bundle.train_y[ids], device=device)
                    value = train_step(model, opt, channel, ema, x, y, c,
                        dict(dataset=j['dataset'], seed=j['seed'], epoch=epoch, batch=batch, global_step=step),
                        bundle.train_row_ids[ids], diagnose=batch == 1 and epoch in (1, 10, 25, 40, 50))
                    for name in ('clean_loss', 'view_loss'):
                        sums[name] += value[name]*len(ids)
                    for name in ('seconds', 'probe_seconds', 'construction_seconds'):
                        sums[name] += value[name]
                    cap_batches += value['cap_active']; weights = value['ledger']; ledger['batches'] += 1
                    for name in ('B', 'clean_entries', 'view_entries'):
                        ledger[name] += weights[name]
                    for cell, count in weights['cells'].items():
                        ledger['cells'][cell] = ledger['cells'].get(cell, 0)+count
                    for name in counts:
                        counts[name] += value[name]
                    batch_log.write(json.dumps(dict(epoch=epoch, batch=batch, global_step=step, **value),
                                               separators=(',', ':'), allow_nan=False)+'\n')
                    if batch == 1:
                        request(attempt, 'allocated', key=key, peak_memory_bytes=torch.cuda.max_memory_reserved(device))
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
            record = dict(epoch=epoch, global_step=step, seconds=time.perf_counter()-tick,
                clean_loss=sums['clean_loss']/len(order), view_loss=sums['view_loss']/len(order),
                probe_seconds=sums['probe_seconds'], construction_seconds=sums['construction_seconds'],
                training_seconds=sums['seconds'], validation_seconds=time.perf_counter()-validation_started,
                validation_cost=validation_cost, validation=validation, objective_ledger=ledger,
                cap_batches=cap_batches, cap_fraction=cap_batches/ledger['batches'], selected_epochs=selected_epochs.copy(),
                model_hash=state_digest(model.state_dict()), ema_hash=state_digest(ema.state_dict()),
                optimizer_hash=state_digest(opt.state_dict()), residual_hash=state_digest(channel.state_dict()),
                epoch_order_hash=state_digest(bundle.train_row_ids[order]),
                support_log=str(batch_path.relative_to(attempt)), support_log_sha256=sha_file(batch_path),
                ram_usage=ram_usage(), examples_per_second=len(order)/(time.perf_counter()-tick), test_accessed=False)
            epoch_file = out/f'epoch_{epoch:02d}.json'; write_json(epoch_file, record)
            snapshot = cpu_copy(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                residual=channel.state_dict(), rng=rng_state(device), loader=loader.get_state(), epoch=epoch, global_step=step,
                selected_models=selected, selected_epochs=selected_epochs, selected_validation=selected_validation,
                common_validation=common_validation, best=best, counts=counts, initial_hash=initial_hash,
                physical_worker_seconds=previous_seconds+time.perf_counter()-started))
            request(attempt, 'commit', key=key, state=snapshot, epoch_file=str(epoch_file.relative_to(attempt)))
            del snapshot
            # Manual stop completes this epoch's updates, validation and guardian ACK.
            stopping = stop_requested()
            write_json(out/'progress.json', dict(epoch=epoch, global_step=step, key=key, status='RAM_COMMITTED',
                manual_stop_observed=stopping, test_accessed=False))
            if epoch < 50 and (stopping or (attempt/'PAUSE_TRAINING').exists()):
                request(attempt, 'paused', key=key, reason='MANUAL_STOP' if stopping else 'TECHNICAL_PAUSE')
                return
        expected = 50*((len(bundle.train_y)+40959)//40960)
        assert step == expected == counts['optimizer_updates'] == counts['ema_updates']
        result = dict(job=j, epochs=50, global_step=step, initial_hash=initial_hash,
            selected_epochs=selected_epochs, selected_validation=selected_validation,
            selected_hashes={k: state_digest(v) for k, v in selected.items()}, common_validation=common_validation,
            final_hashes=dict(online=state_digest(model.state_dict()), fixed_ema=state_digest(ema.state_dict())),
            optimizer_hash=state_digest(opt.state_dict()), residual_hash=state_digest(channel.state_dict()), counts=counts,
            worker_seconds=previous_seconds+time.perf_counter()-started, data_load_seconds=data_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, test_accessed=False)
        request(attempt, 'complete', key=key, result=result)
    except BaseException as error:
        failure = dict(key=key, attempt=number, error=repr(error), traceback=traceback.format_exc(),
                       oom=isinstance(error, torch.OutOfMemoryError), seconds=time.perf_counter()-started)
        write_json(out/'failure.json', failure); request(attempt, 'failed', key=key, failure=failure)
        raise


def schedule(attempt, devices, review_path):
    attempt = Path(attempt).resolve(); manifest = validate_attempt(attempt, review_path, require_smoke=True)
    ensure_guardian(attempt)
    smoke_record = json.loads((attempt/'SMOKE_REPORT.json').read_text())
    jobs = [job(d, a, s, manifest['source_hash']) for s in SEEDS for d in DATASETS for a in ARMS]
    immutable(attempt/'dispatch.json', dict(jobs=jobs, trajectories=50, test_accessed=False))
    for j in jobs:
        immutable(attempt/'jobs'/f"{j['key']}.json", j)
    processes = {}; known_gpus = {}; jobs_by_key = {j['key']: j for j in jobs}
    started = time.perf_counter(); launch_path = attempt/'launch_ledger.jsonl'; manual = False
    if launch_path.exists():
        known_gpus.update({r['key']: r['gpu'] for r in map(json.loads, launch_path.read_text().splitlines())})
    while True:
        manual = manual or stop_requested()
        status = request(attempt, 'status'); records = status['jobs']; pending = []; active = []; failed = []
        for j in jobs:
            key = j['key']; record = records.get(key, {}); process = processes.get(key)
            if record.get('status') == 'COMPLETE':
                continue
            if alive(record.get('worker_pid')) or (process and process.poll() is None):
                active.append(key)
            elif record.get('status') == 'FAILED' or (process and process.returncode not in (None, 0)):
                failed.append(key)
            elif record.get('status') == 'RUNNING':
                failed.append(key)
            else:
                pending.append(j)
        if failed:
            (attempt/'PAUSE_TRAINING').touch(exist_ok=True)
        progress = dict(status='MANUAL_STOPPING' if manual else 'PAUSING' if failed else 'RUNNING',
            complete=len(jobs)-len(active)-len(pending)-len(failed), total=50, pending=len(pending), failed=failed,
            active=[dict(key=k, dataset=jobs_by_key[k]['dataset'], arm=jobs_by_key[k]['config']['arm'],
                seed=jobs_by_key[k]['seed'], epoch=records.get(k, {}).get('epoch', 0), gpu=known_gpus.get(k)) for k in active],
            seconds=time.perf_counter()-started, ram_usage=ram_usage(), retained_states=status['retained_states'],
            retained_bytes=status['retained_bytes'], test_accessed=False)
        write_json(attempt/'progress.json', progress)
        if not active and (manual or failed or (attempt/'PAUSE_TRAINING').exists()):
            progress['status'] = 'MANUAL_STOPPED' if manual else 'PAUSED'; write_json(attempt/'progress.json', progress)
            return progress
        if not pending and not active:
            progress['status'] = 'COMPLETE'; write_json(attempt/'progress.json', progress); return progress
        allocating = {known_gpus.get(k) for k in active if not records.get(k, {}).get('allocated')}
        allocating_host = sum(smoke_record['datasets'][jobs_by_key[k]['dataset']]['host_admission_bytes']
                              for k in active if not records.get(k, {}).get('allocated'))
        if pending and not manual and not failed and not (attempt/'PAUSE_TRAINING').exists():
            # Admission is per device with no job-count limit; bootstrap one measured allocation at a time.
            for j in pending:
                peak = smoke_record['datasets'][j['dataset']]['peak_reserved_bytes']
                host = smoke_record['datasets'][j['dataset']]['host_admission_bytes']
                gpu = choose_device(devices, peak, allocating, host+allocating_host)
                if gpu is None:
                    continue
                if stop_requested():
                    manual = True; break
                logs = attempt/'launch_logs'; logs.mkdir(exist_ok=True)
                with (logs/f"{j['key']}.log").open('ab') as log:
                    process = subprocess.Popen([sys.executable, '-m', MODULE, '_worker',
                        '--attempt', str(attempt), '--key', j['key'], '--device', f'cuda:{gpu}'],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                processes[j['key']] = process; known_gpus[j['key']] = gpu; allocating.add(gpu)
                allocating_host += host
                with launch_path.open('a') as stream:
                    stream.write(json.dumps(dict(key=j['key'], pid=process.pid, gpu=gpu, time=time.time(),
                                                manual_stop_checked=True))+'\n')
        time.sleep(2)


def smoke(attempt, device, review_path):
    attempt = Path(attempt); manifest = validate_attempt(attempt, review_path)
    assert str(device).startswith('cuda') and torch.cuda.is_available()
    out = dict(status='RUNNING', source_hash=manifest['source_hash'], design_sha256=manifest['design_sha256'],
               datasets={}, test_accessed=False)
    write_json(attempt/'SMOKE_REPORT.json', out)
    for dataset in DATASETS:
        bundle, _ = dataset_cache(dataset, manifest['cache_dir']); arms = []; peak = 0; signatures = {}; snapshot_bytes = 0
        for arm in ('SOURCE-CLEAN', 'GAMMA0', 'TV-GRM-010', 'TV-GRM-SHUFFLE-010', 'TV-GRM-INSTANT-010'):
            if stop_requested():
                out['status'] = 'MANUAL_STOPPED'; write_json(attempt/'SMOKE_REPORT.json', out); return out
            while choose_device([int(str(device).split(':')[-1])], peak) is None:
                if stop_requested():
                    out['status'] = 'MANUAL_STOPPED'; write_json(attempt/'SMOKE_REPORT.json', out); return out
                time.sleep(2)
            setup(2022, device); c = config(dataset, 'CLEAN' if arm in ('SOURCE-CLEAN', 'GAMMA0') else arm)
            model = make_model(bundle, load_protocol()['datasets'][dataset]['source']).to(device)
            # Seven parameter copies: online, EMA, Adam m/v, q, selected online/EMA.
            snapshot_bytes = max(snapshot_bytes, 7*tensor_bytes(model.state_dict()))
            ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c); channel = ResidualChannel(model, opt, c)
            loader = torch.Generator().manual_seed(13022)
            ids = torch.randperm(len(bundle.train_y), generator=loader).numpy()[:40960]
            x = torch.as_tensor(bundle.train_x[ids], device=device).T; y = torch.as_tensor(bundle.train_y[ids], device=device)
            torch.cuda.reset_peak_memory_stats(device); synchronize(device); started = time.perf_counter()
            if arm == 'SOURCE-CLEAN':
                value = dict(clean_loss=source_clean_step(model, opt, ema, x, y, c), optimizer_updates=1, ema_updates=1)
            else:
                value = train_step(model, opt, channel, ema, x, y, c,
                    dict(dataset=dataset, seed=2022, epoch=1, batch=1, global_step=1), bundle.train_row_ids[ids], diagnose=True)
            synchronize(device)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(torch.isfinite(p).all() for p in ema.parameters())
            assert value['optimizer_updates'] == value['ema_updates'] == 1
            if arm in ('SOURCE-CLEAN', 'GAMMA0'):
                signatures[arm] = dict(model=state_digest(model.state_dict()), optimizer=state_digest(opt.state_dict()),
                    ema=state_digest(ema.state_dict()), rng=state_digest(rng_state(device)), loader=state_digest(loader.get_state()))
                assert not channel.q
            value.update(arm=arm, seconds=time.perf_counter()-started, rows=len(ids),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device), ram_usage=ram_usage())
            peak = max(peak, value['peak_reserved_bytes']); arms.append(value)
            del model, ema, opt, channel, x, y; torch.cuda.empty_cache()
        assert signatures['SOURCE-CLEAN'] == signatures['GAMMA0']
        out['datasets'][dataset] = dict(arms=arms, peak_reserved_bytes=peak,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            host_admission_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+3*snapshot_bytes,
            estimated_recovery_snapshot_bytes=snapshot_bytes,
            gamma0_equivalence=dict(status='PASS', **signatures))
        write_json(attempt/'SMOKE_REPORT.json', out); del bundle
    out['status'] = 'PASS'; write_json(attempt/'SMOKE_REPORT.json', out)
    return out


def retain(attempt, arm):
    """Keep one complete candidate configuration; called after parent candidate decision."""
    assert arm in ARMS[:3]
    status = request(attempt, 'status'); records = status['jobs']
    assert not any(alive(r.get('worker_pid')) for r in records.values())
    keep = {k for k, r in records.items() if r['job']['config']['arm'] == arm and r['status'] == 'COMPLETE'}
    assert len(keep) == 10 and {(records[k]['job']['dataset'], records[k]['job']['seed']) for k in keep} == {
        (d, s) for d in DATASETS for s in SEEDS}
    for k in keep:
        state = request(attempt, 'get', key=k)['state']; result = records[k]['result']
        assert state is not None, 'Selected RAM state lost; exact frozen rerun required'
        assert {kind: state_digest(value) for kind, value in state['selected_models'].items()} == result['selected_hashes']
    released = []
    for key in records:
        if key not in keep:
            request(attempt, 'release', key=key); released.append(key)
    receipt = dict(arm=arm, guardian_attempt=str(Path(attempt).resolve()), keys=sorted(keep),
                   released_keys=released, test_accessed=False)
    write_json(Path(attempt)/'retained_candidate.json', receipt)
    return receipt


def recover(attempt, key, review_path):
    """Explicitly rearm one repaired technical failure under its unchanged frozen identity."""
    assert key, 'recover requires --key'
    validate_attempt(attempt, review_path, require_smoke=True)
    assert not stop_requested(), 'Manual stop prohibits recovery'
    ensure_guardian(attempt)
    return request(attempt, 'recover', key=key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'smoke', 'run', 'tables', 'status', 'retain', 'release', 'recover', '_worker', '_guardian'))
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--review', type=Path); parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--devices', default='0,1')
    parser.add_argument('--key'); parser.add_argument('--arm', choices=ARMS[:3])
    args = parser.parse_args()
    if args.command == '_guardian':
        serve(args.attempt); return
    if args.command == '_worker':
        worker(args.attempt, args.key, args.device); return
    if args.command == 'prepare':
        assert args.review, 'prepare requires independent static/CPU review'
        value = prepare(args.attempt, args.cache_dir, args.reference_root, args.review)
    elif args.command == 'smoke':
        value = smoke(args.attempt, args.device, args.review)
    elif args.command == 'run':
        value = schedule(args.attempt, [int(x) for x in args.devices.split(',')], args.review)
    elif args.command == 'tables':
        from .tables import export
        value = export(args.attempt)
    elif args.command == 'retain':
        value = retain(args.attempt, args.arm)
    elif args.command == 'recover':
        value = recover(args.attempt, args.key, args.review)
    elif args.command == 'release':
        records = request(args.attempt, 'status')['jobs']
        keys = [args.key] if args.key else list(records)
        for key in keys:
            request(args.attempt, 'release', key=key)
        value = dict(status='RELEASED', keys=keys, test_accessed=False)
    else:
        value = request(args.attempt, 'status')
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
