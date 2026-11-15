"""Reviewed full-budget workers; weights and continuation states stay in RAM."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from .core import train_step, sensitivity, forward
from .protocol import (ROOT, SEEDS, DATASETS, ARMS, config, job, digest, source_manifest,
    immutable, dataset_cache, check_review, write_manifests, jsonl, load_protocol, make_model,
    setup, cpu_copy, state_digest, rng_state, restore_rng, evaluate, write_json,
    optimizer, request, ensure_guardian, alive)


def worker(attempt, key, device):
    attempt = Path(attempt); j = json.loads((attempt/'jobs'/f'{key}.json').read_text())
    assert j['source_hash'] == source_manifest()['sha256']
    registration = request(attempt, 'register', key=key, job=j, pid=os.getpid())
    number = registration['attempt']; state = registration['state']
    out = attempt/'runs'/key/f'attempt{number}'; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); previous_seconds = state['physical_worker_seconds'] if state else 0.
    try:
        bundle, manifest = dataset_cache(j['dataset'], attempt)
        write_json(out/'data_manifest.json', manifest)
        c = j['config']; setup(j['seed'], device)
        model = make_model(bundle, load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
        loader = torch.Generator().manual_seed(j['seed']+11000)
        initial_hash = state_digest(model.state_dict())
        selected = {}; selected_epochs = {}; selected_validation = {}
        best = {kind: -float('inf') for kind in ('online', 'fixed_ema')}
        common_validation = {}; counts = dict(optimizer_updates=0, ema_updates=0)
        step = 0; epoch0 = 0
        if state:
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
            opt.load_state_dict(state['optimizer']); loader.set_state(state['loader'])
            restore_rng(state['rng'], device)
            epoch0 = state['epoch']; step = state['global_step']
            selected = state['selected_models']; selected_epochs = state['selected_epochs']
            selected_validation = state['selected_validation']; best = state['best']
            counts = state['counts']; common_validation = state['common_validation']
            initial_hash = state['initial_hash']
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(epoch0+1, 51):
            tick = time.perf_counter(); model.train()
            order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            sums = dict(loss=0., clean_loss=0., auxiliary_loss=0.)
            for batch, offset in enumerate(range(0, len(order), 40960), 1):
                ids = order[offset:offset+40960]; step += 1
                x = torch.as_tensor(bundle.train_x[ids], device=device).T
                y = torch.as_tensor(bundle.train_y[ids], device=device)
                value = train_step(model, opt, ema, x, y, c,
                    dict(dataset=j['dataset'], seed=j['seed'], epoch=epoch, global_step=step))
                for name in sums:
                    sums[name] += value[name]*len(ids)
                counts['optimizer_updates'] += value['optimizer_updates']; counts['ema_updates'] += value['ema_updates']
                jsonl(out/'batches.jsonl', dict(epoch=epoch, batch=batch, global_step=step, rows=len(ids), **value))
                if epoch == epoch0+1 and batch == 1:
                    request(attempt, 'allocated', key=key, peak_memory_bytes=torch.cuda.max_memory_reserved(device))
                del x, y, value
            validation = {}
            for kind, estimator in (('online', model), ('fixed_ema', ema)):
                values, cost = evaluate(estimator, bundle.val_x, bundle.val_y, device)
                validation[kind] = values
                score = values['macro_auc']; assert score is not None
                if score > best[kind]:
                    best[kind] = score; selected_epochs[kind] = epoch
                    selected[kind] = cpu_copy(estimator.state_dict()); selected_validation[kind] = values
            if epoch in (10, 50):
                common_validation[str(epoch)] = validation
            record = dict(epoch=epoch, global_step=step, seconds=time.perf_counter()-tick,
                **{k:v/len(order) for k,v in sums.items()}, validation=validation,
                selected_epochs=selected_epochs.copy(), model_hash=state_digest(model.state_dict()),
                ema_hash=state_digest(ema.state_dict()), optimizer_hash=state_digest(opt.state_dict()),
                epoch_order_hash=state_digest(bundle.train_row_ids[order]))
            epoch_file = out/f'epoch_{epoch:02d}.json'; write_json(epoch_file, record)
            snapshot = cpu_copy(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                rng=rng_state(device), loader=loader.get_state(), epoch=epoch, global_step=step,
                selected_models=selected, selected_epochs=selected_epochs, selected_validation=selected_validation,
                common_validation=common_validation, best=best, counts=counts, initial_hash=initial_hash,
                physical_worker_seconds=previous_seconds+time.perf_counter()-started))
            request(attempt, 'commit', key=key, state=snapshot, epoch_file=str(epoch_file.relative_to(attempt)))
            write_json(out/'progress.json', dict(epoch=epoch, global_step=step, key=key, status='RAM_COMMITTED'))
            del snapshot
        expected_steps = 50*((len(bundle.train_y)+40959)//40960)
        assert step == expected_steps == counts['optimizer_updates'] == counts['ema_updates']
        result = dict(job=j, epochs=50, global_step=step, initial_hash=initial_hash,
            selected_epochs=selected_epochs, selected_validation=selected_validation,
            selected_hashes={k:state_digest(v) for k,v in selected.items()}, common_validation=common_validation,
            final_hashes=dict(online=state_digest(model.state_dict()), fixed_ema=state_digest(ema.state_dict())),
            optimizer_hash=state_digest(opt.state_dict()), counts=counts,
            worker_seconds=previous_seconds+time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), test_accessed=False)
        request(attempt, 'complete', key=key, result=result)
    except BaseException as error:
        failure = dict(key=key, attempt=number, error=repr(error), traceback=traceback.format_exc(),
                       oom=isinstance(error, torch.OutOfMemoryError), seconds=time.perf_counter()-started)
        write_json(out/'failure.json', failure); request(attempt, 'failed', key=key, failure=failure)
        raise


def gpu_status():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,power.draw,power.limit',
                                   '--format=csv,noheader,nounits'], text=True)
    return [dict(zip(('index', 'memory', 'total', 'power', 'limit'), map(float, line.split(','))))
            for line in raw.strip().splitlines()]


def ram_usage():
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 1-values['MemAvailable']/values['MemTotal']


def choose_device(devices, peak):
    if ram_usage() >= .9:
        return None
    for row in gpu_status():
        if int(row['index']) in devices and row['memory']/row['total'] < .85 and row['power']/row['limit'] < .85:
            if (row['total']-row['memory'])*1024**2 >= peak:
                return int(row['index'])
    return None


def results(attempt):
    return [json.loads(path.read_text()) for path in sorted((Path(attempt)/'runs').glob('*/result.json'))]


def schedule(attempt, jobs, devices, stage):
    attempt = Path(attempt).resolve(); ensure_guardian(attempt)
    assert not (attempt/'selection_frozen.json').exists(), 'Training configuration already frozen'
    smoke_record = json.loads((attempt/'SMOKE_REPORT.json').read_text())
    assert smoke_record['status'] == 'PASS' and smoke_record['source_hash'] == source_manifest()['sha256']
    immutable(attempt/'dispatches'/f'{stage}.json', dict(stage=stage, jobs=jobs))
    for j in jobs:
        assert j == job(j['dataset'], j['config']['arm'], j['seed'], j['config']['alpha'])
        immutable(attempt/'jobs'/f"{j['key']}.json", j)
    processes = {}; started = time.perf_counter()
    while True:
        records = request(attempt, 'status')['jobs']; pending = []; active = []; failed = []
        for j in jobs:
            key = j['key']; record = records.get(key, {}); process = processes.get(key)
            if record.get('status') == 'COMPLETE':
                continue
            if alive(record.get('worker_pid')) or (process and process.poll() is None):
                active.append(key)
            elif record.get('status') == 'FAILED':
                failed.append(key)
            elif record and record.get('attempts', 0) >= 2:
                failed.append(key)
            else:
                pending.append(j)
        write_json(attempt/'progress.json', dict(stage=stage, complete=len(jobs)-len(active)-len(pending)-len(failed),
            total=len(jobs), active=[dict(key=k, epoch=records.get(k, {}).get('epoch', 0)) for k in active],
            pending=len(pending), failed=failed, seconds=time.perf_counter()-started, test_accessed=False))
        if failed:
            raise RuntimeError('Worker failure; inspect failure.json: '+str(failed))
        if not pending and not active:
            break
        allocating = any(not records.get(k, {}).get('allocated') for k in active)
        if pending and not allocating:
            j = pending[0]
            peak = smoke_record['datasets'][j['dataset']]['peak_reserved_bytes']
            gpu = choose_device(devices, peak)
            if gpu is not None:
                logs = attempt/'launch_logs'; logs.mkdir(exist_ok=True)
                with (logs/f"{j['key']}.log").open('ab') as log:
                    process = subprocess.Popen([sys.executable, '-m', __package__+'.run', '_worker',
                        '--attempt', str(attempt), '--key', j['key'], '--device', f'cuda:{gpu}'],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                processes[j['key']] = process
                jsonl(attempt/'launch_ledger.jsonl', dict(key=j['key'], pid=process.pid, gpu=gpu, time=time.time()))
                # Let the allocating worker report its full first-batch peak before further admissions.
                while process.poll() is None:
                    current = request(attempt, 'status')['jobs'].get(j['key'], {})
                    if current.get('allocated') or current.get('status') == 'FAILED':
                        break
                    time.sleep(1)
        time.sleep(2)
    return dict(status='COMPLETE', stage=stage, jobs=len(jobs))


def check_sensitivity(model, small):
    embedding32, logits32, gradient32 = sensitivity(model, small)
    with torch.no_grad():
        torch.testing.assert_close(forward(model, small)[1], logits32, rtol=0, atol=0)
    # FP64 avoids cancellation when differencing logits at tiny initial embeddings.
    probe = copy.deepcopy(model).double()
    embedding64, _, gradient64 = sensitivity(probe, small.double())
    torch.testing.assert_close(gradient32.double(), gradient64, rtol=1e-2, atol=1e-4)
    direction = torch.ones_like(embedding64)/embedding64.shape[1]**.5
    with torch.no_grad():
        finite = (forward(probe, small.double(), 1e-5*direction)[1]-
                  forward(probe, small.double(), -1e-5*direction)[1])/(2e-5)
    derivative = (gradient64*direction[:, None]).sum(2)
    torch.testing.assert_close(finite, derivative, rtol=1e-3, atol=1e-7)
    return dict(probe_dtype='float64', step=1e-5, fd_rtol=1e-3, fd_atol=1e-7,
                fp32_rtol=1e-2, fp32_atol=1e-4,
                fd_max_absolute_error=float((finite-derivative).abs().max()),
                fp32_gradient_max_absolute_error=float((gradient32.double()-gradient64).abs().max()))


def smoke(attempt, device):
    attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    out = dict(status='RUNNING', source_hash=source_manifest()['sha256'], datasets={}, test_accessed=False)
    write_json(attempt/'SMOKE_REPORT.json', out)
    for dataset in DATASETS:
        bundle, manifest = dataset_cache(dataset, attempt)
        write_json(attempt/f'{dataset}_data_manifest.json', manifest)
        arms = []; peak = 0
        for arm in ARMS:
            while choose_device([int(str(device).split(':')[-1])], 0) is None:
                time.sleep(2)
            setup(2022, device); c = config(dataset, arm)
            model = make_model(bundle, load_protocol()['datasets'][dataset]['source']).to(device)
            ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
            loader = torch.Generator().manual_seed(2022+11000)
            ids = torch.randperm(len(bundle.train_y), generator=loader).numpy()[:40960]
            x = torch.as_tensor(bundle.train_x[ids], device=device).T
            y = torch.as_tensor(bundle.train_y[ids], device=device)
            torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
            if arm == 'CLEAN':
                sensitivity_check = check_sensitivity(model, x[:, :4])
            value = train_step(model, opt, ema, x, y, c,
                               dict(dataset=dataset, seed=2022, epoch=1, global_step=1))
            torch.cuda.synchronize(device)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            value.update(arm=arm, seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device), rows=len(ids))
            if arm == 'CLEAN':
                value['sensitivity_check'] = sensitivity_check
            peak = max(peak, value['peak_reserved_bytes']); arms.append(value)
            del model, ema, opt, x, y
            torch.cuda.empty_cache()
        out['datasets'][dataset] = dict(arms=arms, peak_reserved_bytes=peak)
        write_json(attempt/'SMOKE_REPORT.json', out)
    out['status'] = 'PASS'
    write_json(attempt/'SMOKE_REPORT.json', out)
    return out


def freeze(attempt, configurations):
    attempt = Path(attempt); entries = []
    records = {r['job']['key']:r for r in results(attempt)}
    for dataset, arm, alpha in configurations:
        for seed in SEEDS:
            j = job(dataset, arm, seed, alpha); result = records[j['key']]
            assert result['epochs'] == 50
            state = request(attempt, 'get', key=j['key'])['state']; assert state is not None
            for kind in ('online', 'fixed_ema'):
                assert state_digest(state['selected_models'][kind]) == result['selected_hashes'][kind]
                entries.append(dict(job=j, key=j['key'], kind=kind,
                    epoch=result['selected_epochs'][kind], model_hash=result['selected_hashes'][kind]))
    frozen = dict(source_hash=source_manifest()['sha256'], configurations=configurations,
        endpoints=entries, test_opened=False, seed_pool='2022-2026 reused per user override')
    immutable(attempt/'selection_frozen.json', frozen)
    return frozen


def test_evaluate(attempt, devices):
    attempt = Path(attempt); frozen = json.loads((attempt/'selection_frozen.json').read_text())
    assert frozen['source_hash'] == source_manifest()['sha256']
    out = []; bundle_cache = {}
    for endpoint in frozen['endpoints']:
        path = attempt/'test_endpoints'/f"{endpoint['key']}_{endpoint['kind']}.json"
        if path.exists():
            out.append(json.loads(path.read_text())); continue
        j = endpoint['job']; dataset = j['dataset']
        if dataset not in bundle_cache:
            bundle_cache[dataset] = dataset_cache(dataset, attempt)[0]
        bundle = bundle_cache[dataset]
        state = request(attempt, 'get', key=j['key'])['state']; assert state is not None
        weights = state['selected_models'][endpoint['kind']]
        assert state_digest(weights) == endpoint['model_hash']
        gpu = None
        while gpu is None:
            gpu = choose_device(devices, 2*1024**3)
            if gpu is None:
                time.sleep(2)
        device = f'cuda:{gpu}'; setup(j['seed'], device)
        model = make_model(bundle, load_protocol()['datasets'][dataset]['source']).to(device)
        model.load_state_dict(weights)
        values, cost = evaluate(model, bundle.test_x, bundle.test_y, device)
        record = dict(**endpoint, metrics=values, cost=cost, test_accessed=True)
        immutable(path, record); out.append(record)
        del model; torch.cuda.empty_cache()
    write_json(attempt/'TEST_COMPLETION.json', dict(status='COMPLETE', endpoints=len(out)))
    return dict(status='COMPLETE', endpoints=len(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('manifest', 'smoke', 'pilot', 'hpo', 'decision', 'select', 'confirm', 'freeze', 'evaluate', 'tables',
                                            'status', 'release', '_worker'))
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--review', type=Path); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='0,1'); parser.add_argument('--key')
    parser.add_argument('--configurations', type=Path)
    args = parser.parse_args(); devices = list(map(int, args.devices.split(',')))
    if args.command == '_worker':
        worker(args.attempt, args.key, args.device); return
    if args.command == 'status':
        print(json.dumps(request(args.attempt, 'status'), indent=2)); return
    if args.command == 'release':
        for r in results(args.attempt):
            request(args.attempt, 'release', key=r['job']['key'])
        print(json.dumps(request(args.attempt, 'shutdown'))); return
    check_review(args.review)
    if args.command == 'manifest':
        write_manifests(args.attempt); value = dict(status='COMPLETE', source_hash=source_manifest()['sha256'])
    elif args.command == 'smoke':
        try:
            value = smoke(args.attempt, args.device)
        except BaseException as error:
            write_json(args.attempt/'SMOKE_FAILURE.json', dict(status='ENGINEERING_BLOCKED',
                error=repr(error), traceback=traceback.format_exc(), test_accessed=False))
            raise
    elif args.command == 'pilot':
        value = schedule(args.attempt, [job(d,a,s) for s in SEEDS for d in DATASETS for a in ARMS], devices, 'PILOT')
    elif args.command in ('decision', 'hpo'):
        from .selection import pilot_decision
        decision = pilot_decision(results(args.attempt))
        immutable(args.attempt/'PILOT_DECISION.json', decision)
        if args.command == 'hpo' and decision['routes']:
            value = schedule(args.attempt, [job(d,a,s,alpha) for a in decision['routes']
                for alpha in (.01,.03,.05) for s in SEEDS for d in DATASETS], devices, 'HPO')
        else:
            value = decision
    elif args.command == 'freeze':
        value = freeze(args.attempt, json.loads(args.configurations.read_text()))
    elif args.command == 'select':
        from .selection import hpo_selection
        routes = json.loads((args.attempt/'PILOT_DECISION.json').read_text())['routes']
        value = hpo_selection(results(args.attempt), routes)
        immutable(args.attempt/'HPO_SELECTION.json', value)
        immutable(args.attempt/'confirmation_configurations.json', value['configurations'])
    elif args.command == 'confirm':
        configurations = json.loads(args.configurations.read_text())
        value = schedule(args.attempt, [job(d,a,s,alpha) for d,a,alpha in configurations for s in SEEDS],
                         devices, 'CONFIRM_'+digest(configurations)[:12])
    elif args.command == 'tables':
        from .tables import export
        value = export(args.attempt, results(args.attempt))
    else:
        value = test_evaluate(args.attempt, devices)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
