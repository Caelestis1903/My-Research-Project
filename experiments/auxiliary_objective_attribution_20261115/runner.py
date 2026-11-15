"""Stage 1 only: full-budget workers, resource admission and RAM continuation."""
import argparse
import copy
import gzip
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from .core import train_step
from .protocol import (ROOT, SEEDS, DATASETS, ARMS, config, job, source_manifest, immutable,
    dataset_cache, check_review, prepare, results, load_protocol, make_model, setup, cpu_copy,
    state_digest, rng_state, restore_rng, evaluate, write_json, optimizer, request,
    ensure_guardian, alive, sha_file)


def ram_usage():
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return 1-values['MemAvailable']/values['MemTotal']


def gpu_status():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,power.draw,power.limit',
                                   '--format=csv,noheader,nounits'], text=True)
    return [dict(zip(('index','memory','total','power','limit'), map(float, line.split(','))))
            for line in raw.strip().splitlines()]


def choose_device(devices, peak, allocating=()):
    if ram_usage() >= .9:
        return None
    for row in gpu_status():
        gpu = int(row['index'])
        if gpu in devices and gpu not in allocating and row['memory']/row['total'] < .85 and row['power']/row['limit'] < .85:
            if (row['total']-row['memory'])*1024**2 >= peak:
                return gpu
    return None


def worker(attempt, key, device):
    attempt = Path(attempt); j = json.loads((attempt/'jobs'/f'{key}.json').read_text())
    assert j['source_hash'] == source_manifest()['sha256']
    manifest = json.loads((attempt/'run_manifest.json').read_text())
    registration = request(attempt, 'register', key=key, job=j, pid=os.getpid())
    number = registration['attempt']; state = registration['state']
    out = attempt/'runs'/key/f'attempt{number}'; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); previous_seconds = state['physical_worker_seconds'] if state else 0.
    try:
        load_started = time.perf_counter()
        bundle, data_manifest = dataset_cache(j['dataset'], manifest['cache_dir'])
        data_seconds = time.perf_counter()-load_started
        write_json(out/'data_manifest.json', data_manifest)
        c = j['config']; setup(j['seed'], device)
        model = make_model(bundle, load_protocol()['datasets'][j['dataset']]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model, c)
        loader = torch.Generator().manual_seed(j['seed']+11000)
        initial_hash = state_digest(model.state_dict())
        selected = {}; selected_epochs = {}; selected_validation = {}
        best = {kind:-float('inf') for kind in ('online','fixed_ema')}
        common_validation = {}; counts = dict(optimizer_updates=0, ema_updates=0)
        step = 0; epoch0 = 0
        if state:
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
            opt.load_state_dict(state['optimizer']); loader.set_state(state['loader'])
            restore_rng(state['rng'], device)
            epoch0 = state['epoch']; step = state['global_step']
            selected = state['selected_models']; selected_epochs = state['selected_epochs']
            selected_validation = state['selected_validation']; best = state['best']
            counts = state['counts']; common_validation = state['common_validation']; initial_hash = state['initial_hash']
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(epoch0+1, 51):
            tick = time.perf_counter(); model.train()
            order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            sums = dict(loss=0., clean_loss=0., auxiliary_loss=0., probe_seconds=0., training_seconds=0.)
            ledger = dict(batches=0, B=0, M=0, total_weight_mass=0., cells={})
            batch_path = out/f'batches_epoch_{epoch:02d}.jsonl.gz'
            with gzip.open(batch_path, 'xt', compresslevel=3) as batch_log:
                for batch, offset in enumerate(range(0, len(order), 40960), 1):
                    ids = order[offset:offset+40960]; step += 1
                    x = torch.as_tensor(bundle.train_x[ids], device=device).T
                    y = torch.as_tensor(bundle.train_y[ids], device=device)
                    value = train_step(model, opt, ema, x, y, c,
                        dict(dataset=j['dataset'], seed=j['seed'], epoch=epoch, global_step=step),
                        bundle.train_row_ids[ids], diagnose=batch == 1)
                    for name in ('loss','clean_loss','auxiliary_loss'):
                        sums[name] += value[name]*len(ids)
                    for name in ('probe_seconds','training_seconds'):
                        sums[name] += value[name]
                    weights = value['ledger']; ledger['batches'] += 1
                    for name in ('B','M','total_weight_mass'):
                        ledger[name] += weights[name]
                    for cell, fields in weights['cells'].items():
                        target = ledger['cells'].setdefault(cell, {})
                        for name, v in fields.items():
                            target[name] = target.get(name, 0)+v
                    counts['optimizer_updates'] += value['optimizer_updates']; counts['ema_updates'] += value['ema_updates']
                    batch_log.write(json.dumps(dict(epoch=epoch, batch=batch, global_step=step, rows=len(ids), **value),
                        separators=(',', ':'), allow_nan=False)+'\n')
                    if epoch == epoch0+1 and batch == 1:
                        request(attempt, 'allocated', key=key, peak_memory_bytes=torch.cuda.max_memory_reserved(device))
                    del x, y, value
            validation = {}
            for kind, estimator in (('online',model),('fixed_ema',ema)):
                values, cost = evaluate(estimator, bundle.val_x, bundle.val_y, device)
                validation[kind] = values
                score = values['macro_auc']; assert score is not None
                if score > best[kind]:
                    best[kind] = score; selected_epochs[kind] = epoch
                    selected[kind] = cpu_copy(estimator.state_dict()); selected_validation[kind] = values
            if epoch in (10,30,50):
                common_validation[str(epoch)] = validation
            record = dict(epoch=epoch, global_step=step, seconds=time.perf_counter()-tick,
                **{k:v/len(order) for k,v in sums.items() if k.endswith('loss')},
                probe_seconds=sums['probe_seconds'], training_seconds=sums['training_seconds'],
                validation=validation, objective_ledger=ledger, selected_epochs=selected_epochs.copy(),
                model_hash=state_digest(model.state_dict()), ema_hash=state_digest(ema.state_dict()),
                optimizer_hash=state_digest(opt.state_dict()), epoch_order_hash=state_digest(bundle.train_row_ids[order]),
                support_log=str(batch_path.relative_to(attempt)), support_log_sha256=sha_file(batch_path),
                ram_usage=ram_usage(), examples_per_second=len(order)/(time.perf_counter()-tick), test_accessed=False)
            epoch_file = out/f'epoch_{epoch:02d}.json'; write_json(epoch_file, record)
            snapshot = cpu_copy(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                rng=rng_state(device), loader=loader.get_state(), epoch=epoch, global_step=step,
                selected_models=selected, selected_epochs=selected_epochs, selected_validation=selected_validation,
                common_validation=common_validation, best=best, counts=counts, initial_hash=initial_hash,
                physical_worker_seconds=previous_seconds+time.perf_counter()-started))
            request(attempt, 'commit', key=key, state=snapshot, epoch_file=str(epoch_file.relative_to(attempt)))
            write_json(out/'progress.json', dict(epoch=epoch, global_step=step, key=key, status='RAM_COMMITTED'))
            del snapshot
        expected = 50*((len(bundle.train_y)+40959)//40960)
        assert step == expected == counts['optimizer_updates'] == counts['ema_updates']
        result = dict(job=j, epochs=50, global_step=step, initial_hash=initial_hash,
            selected_epochs=selected_epochs, selected_validation=selected_validation,
            selected_hashes={k:state_digest(v) for k,v in selected.items()}, common_validation=common_validation,
            final_hashes=dict(online=state_digest(model.state_dict()), fixed_ema=state_digest(ema.state_dict())),
            optimizer_hash=state_digest(opt.state_dict()), counts=counts,
            worker_seconds=previous_seconds+time.perf_counter()-started, data_load_seconds=data_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), test_accessed=False)
        request(attempt, 'complete', key=key, result=result)
    except BaseException as error:
        failure = dict(key=key, attempt=number, error=repr(error), traceback=traceback.format_exc(),
                       oom=isinstance(error, torch.OutOfMemoryError), seconds=time.perf_counter()-started)
        write_json(out/'failure.json', failure); request(attempt, 'failed', key=key, failure=failure)
        raise


def schedule(attempt, devices):
    attempt = Path(attempt).resolve(); ensure_guardian(attempt)
    smoke_record = json.loads((attempt/'SMOKE_REPORT.json').read_text())
    assert smoke_record['status'] == 'PASS' and smoke_record['source_hash'] == source_manifest()['sha256']
    jobs = [job(d,a,s) for s in SEEDS for d in DATASETS for a in ARMS]
    immutable(attempt/'dispatch.json', dict(stage='STAGE1', jobs=jobs))
    for j in jobs:
        immutable(attempt/'jobs'/f"{j['key']}.json", j)
    processes = {}; known_gpus = {}; jobs_by_key = {j['key']:j for j in jobs}; started = time.perf_counter()
    launch_path = attempt/'launch_ledger.jsonl'
    if launch_path.exists():
        known_gpus.update({r['key']:r['gpu'] for r in map(json.loads, launch_path.read_text().splitlines())})
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
            elif record and record.get('attempts',0) >= 2:
                failed.append(key)
            else:
                pending.append(j)
        progress = dict(stage='STAGE1', complete=len(jobs)-len(active)-len(pending)-len(failed), total=len(jobs),
            active=[dict(key=k, dataset=jobs_by_key[k]['dataset'], arm=jobs_by_key[k]['config']['arm'],
                seed=jobs_by_key[k]['seed'], epoch=records.get(k,{}).get('epoch',0), gpu=known_gpus.get(k)) for k in active],
            pending=len(pending), failed=failed, seconds=time.perf_counter()-started, ram_usage=ram_usage(), test_accessed=False)
        write_json(attempt/'progress.json', progress)
        if failed:
            raise RuntimeError('Worker failure; inspect failure.json: '+str(failed))
        if not pending and not active:
            break
        allocating = {known_gpus[k] for k in active if not records.get(k,{}).get('allocated')}
        if pending:
            j = pending[0]; peak = smoke_record['datasets'][j['dataset']]['peak_reserved_bytes']
            gpu = choose_device(devices, peak, allocating)
            if gpu is not None:
                logs = attempt/'launch_logs'; logs.mkdir(exist_ok=True)
                with (logs/f"{j['key']}.log").open('ab') as log:
                    process = subprocess.Popen([sys.executable,'-m',__package__+'.runner','_worker',
                        '--attempt',str(attempt),'--key',j['key'],'--device',f'cuda:{gpu}'],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                processes[j['key']] = process; known_gpus[j['key']] = gpu
                with launch_path.open('a') as f:
                    f.write(json.dumps(dict(key=j['key'],pid=process.pid,gpu=gpu,time=time.time()))+'\n')
        time.sleep(2)
    return dict(status='COMPLETE', stage='STAGE1', jobs=len(jobs), seconds=time.perf_counter()-started)


def smoke(attempt, device):
    from .checks import check_sensitivity
    attempt = Path(attempt); manifest = json.loads((attempt/'run_manifest.json').read_text())
    out = dict(status='RUNNING', source_hash=source_manifest()['sha256'], datasets={}, test_accessed=False)
    write_json(attempt/'SMOKE_REPORT.json', out)
    for dataset in DATASETS:
        bundle, _ = dataset_cache(dataset, manifest['cache_dir']); arms = []; peak = 0
        for arm in ARMS:
            while choose_device([int(str(device).split(':')[-1])], 0) is None:
                time.sleep(2)
            setup(2022, device); c = config(dataset,arm)
            model = make_model(bundle,load_protocol()['datasets'][dataset]['source']).to(device)
            ema = copy.deepcopy(model).requires_grad_(False); opt = optimizer(model,c)
            loader = torch.Generator().manual_seed(13022)
            ids = torch.randperm(len(bundle.train_y),generator=loader).numpy()[:40960]
            x = torch.as_tensor(bundle.train_x[ids],device=device).T
            y = torch.as_tensor(bundle.train_y[ids],device=device)
            torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
            sensitivity_check = check_sensitivity(model,x[:,:4]) if arm == 'B' else None
            value = train_step(model,opt,ema,x,y,c,dict(dataset=dataset,seed=2022,epoch=1,global_step=1),
                               bundle.train_row_ids[ids],diagnose=True)
            torch.cuda.synchronize(device)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            value.pop('support')
            value.update(arm=arm,seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),rows=len(ids),sensitivity_check=sensitivity_check)
            assert value['optimizer_updates'] == value['ema_updates'] == 1
            peak = max(peak,value['peak_reserved_bytes']); arms.append(value)
            del model,ema,opt,x,y; torch.cuda.empty_cache()
        out['datasets'][dataset] = dict(arms=arms,peak_reserved_bytes=peak)
        write_json(attempt/'SMOKE_REPORT.json',out)
    out['status'] = 'PASS'; write_json(attempt/'SMOKE_REPORT.json',out)
    return dict(status='PASS',source_hash=out['source_hash'],datasets=list(out['datasets']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','smoke','stage1','tables','status','release','_worker'))
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--review',type=Path); parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--pilot-root',type=Path); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--devices',default='0,1'); parser.add_argument('--key')
    args = parser.parse_args()
    if args.command == '_worker':
        worker(args.attempt,args.key,args.device); return
    if args.command == 'prepare':
        value = prepare(args.attempt,args.cache_dir,args.pilot_root)
    elif args.command == 'status':
        value = request(args.attempt,'status')
    elif args.command == 'release':
        for r in results(args.attempt):
            if r['job']['config']['arm'] != 'CLEAN':
                request(args.attempt,'release',key=r['job']['key'])
        value = request(args.attempt,'shutdown')
    elif args.command == 'tables':
        from .tables import export
        value = export(args.attempt)
    else:
        check_review(args.review)
        value = smoke(args.attempt,args.device) if args.command == 'smoke' else schedule(args.attempt,list(map(int,args.devices.split(','))))
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
