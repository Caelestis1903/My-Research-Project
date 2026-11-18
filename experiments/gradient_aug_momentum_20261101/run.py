"""One RAM coordinator from the frozen forty-run matrix to unified testing."""
import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from experiments.gradient_aug_20261031_v2.views import make_model
from experiments.gradient_aug_20261031_v2.vendor_ga31.data import load_dataset
from experiments.gradient_aug_20261031_v2.vendor_ga31.audited_data import row_ids_hash
from experiments.gradient_aug_20261031_v2.vendor_ga31.metrics import metrics, evaluate
from experiments.gradient_aug_20261031_v2.run import gpu_status, available_ram
from .protocol import (ROOT, HERE, DEFAULT_PROTOCOL, protocol, arm_config, make_optimizer,
    setup, cpu_copy, equal, tensor_bytes, send_memory, receive_memory, write_json, append_csv,
    state_digest, make_shadow, update_shadow, snapshot, checkpoint_rows, selected_update)
from .objectives import train_step
from .analyze import phase1_gate, build_tables


def train_one(bundle, dataset, p, arm, seed, device, out, connection=None, epochs=50, attempt=1):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    setup(seed, device); conf = p['datasets'][dataset]; c = arm_config(p,arm,dataset)
    model = make_model(bundle,conf).to(device); optimizer = make_optimizer(model,p,c)
    shadow = make_shadow(model,c)
    assert sum(v.numel() for v in model.parameters()) == {'movielens':773470,'kuairand':1614254}[dataset]
    init_hash = state_digest(model.state_dict())
    loaders = {k:torch.Generator().manual_seed(seed+offset) for k,offset in p['rng']['loader_offsets'].items()}
    kinds = ['online']+(['shadow'] if shadow is not None else [])
    selected = {}; best = {k:-float('inf') for k in kinds}; states = {}; order_hashes = []
    diagnostics = []; step = 0; ready_sent = False
    start = time.perf_counter(); forward_count = gradient_calls = extra_views = 0
    validation_forwards = shadow_validation_forwards = original_exposures = 0
    write_json(out/'resolved_config.json',dict(study_id=p['study_id'],dataset=dataset,arm=arm,seed=seed,
        device=device,config=c,optimizer=p['optimizer'],epochs=epochs,batch_size=40960,
        execution_commit=p.get('execution_commit'),source=p['source'],state_storage='RAM_ONLY'))
    write_json(out/'model_identity.json',dict(parameter_count=sum(v.numel() for v in model.parameters()),
        state_dict_keys=list(model.state_dict()),parameter_names=[k for k,_ in model.named_parameters()],
        init_hash=init_hash,field_slices=model.field_slices,perturb_slices=model.perturb_slices))
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,epochs+1):
        epoch_start = time.perf_counter(); model.train()
        order = torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        order_hash = row_ids_hash(bundle.train_row_ids[order]); order_hashes.append(order_hash)
        predictions, labels, domains = [], [], []; optimized_sum = real_sum = 0.; epoch_views = 0
        for batch, start_row in enumerate(range(0,len(order),40960),1):
            ids = order[start_row:start_row+40960]; step += 1
            x = torch.as_tensor(bundle.train_x[ids],device=device).T
            y = torch.as_tensor(bundle.train_y[ids],device=device)
            scalar, info, diagnostic, calls = train_step(model,optimizer,x,y,conf['augmented_domain'],
                c,dataset,seed,step,diagnose=batch == conf['steps_per_epoch'])
            update_shadow(shadow,model,c['rho_shadow'])
            if diagnostic is not None:
                row = dict(epoch=epoch,global_step=step,**diagnostic); diagnostics.append(row)
                append_csv(out/'optimizer_state_diagnostics.csv',row)
            forward_count += info['forwards']; gradient_calls += calls
            epoch_views += info['extra_views']; extra_views += info['extra_views']
            n = len(ids); original_exposures += n
            optimized_sum += float(scalar)*n; real_sum += float(info['real'].detach().mean(1).mean())*n
            predictions.append(info['pred'].detach().cpu().numpy()); labels.append(y.cpu().numpy()); domains.append(x[-1].cpu().numpy())
            append_csv(out/'batch_trace.csv',dict(epoch=epoch,batch=batch,global_step=step,rows=n,
                original_row_hash=row_ids_hash(bundle.train_row_ids[ids]),epoch_order_hash=order_hash,
                extra_views=info['extra_views'],gradient_calls=calls,optimizer_steps=1))
            for view in info['views']:
                append_csv(out/'view_stats.csv',dict(epoch=epoch,batch=batch,global_step=step,
                    view_key=view['key'],view_seed=view['seed'],view_hash=view['hash'],target_rows=int(info['mask'].sum())))
            if not ready_sent and connection is not None:
                torch.cuda.synchronize(device)
                send_memory(connection,dict(kind='allocated',peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
                ready_sent = True
            del info, scalar, x, y
        training = metrics(np.concatenate(labels),np.concatenate(predictions),np.concatenate(domains))
        validation = {'online':evaluate(model,bundle.val_x,bundle.val_y,device)}
        validation_forwards += (len(bundle.val_y)+40959)//40960
        if shadow is not None:
            validation['shadow'] = evaluate(shadow,bundle.val_x,bundle.val_y,device)
            shadow_validation_forwards += (len(bundle.val_y)+40959)//40960
        for kind in kinds:
            selected_update(selected,best,kind,epoch,validation[kind]['macro_auc'])
        if epoch in p['training']['save_epochs'] or epoch in selected.values():
            states[epoch] = snapshot(model,optimizer,shadow,loaders,epoch,step,validation,device,
                dict(dataset=dataset,arm=arm,seed=seed,epoch_order_hash=order_hash,
                     raw_sha256=conf['raw_sha256'],split_row_id_hashes=conf['split_row_id_hashes']))
        for old in list(states):
            if old not in p['training']['save_epochs'] and old not in selected.values():
                del states[old]
        append_csv(out/'epoch_metrics.csv',dict(epoch=epoch,global_step=step,original_rows=len(order),
            epoch_order_hash=order_hash,training_online_optimized_loss=optimized_sum/len(order),
            training_online_real_bce=real_sum/len(order),extra_views=epoch_views,
            epoch_seconds=time.perf_counter()-epoch_start,**{'train_online_'+k:v for k,v in training.items()},
            **{f'val_{kind}_{k}':v for kind,values in validation.items() for k,v in values.items()}))
        if shadow is not None:
            append_csv(out/'shadow_metrics.csv',dict(epoch=epoch,global_step=step,
                shadow_updates=step,online_auc=validation['online']['macro_auc'],shadow_auc=validation['shadow']['macro_auc'],
                delta_auc=validation['shadow']['macro_auc']-validation['online']['macro_auc'],
                online_logloss=validation['online']['macro_logloss'],shadow_logloss=validation['shadow']['macro_logloss']))
        for split, kind, values in [('train_online','online',training)]+[('validation',k,v) for k,v in validation.items()]:
            for d in range(3):
                for t in range(2):
                    append_csv(out/'per_cell_metrics.csv',dict(epoch=epoch,split=split,model_kind=kind,domain=d,task=t,
                        auc=values[f'd{d}t{t}_auc'],logloss=values[f'd{d}t{t}_logloss'],
                        count=values[f'd{d}t{t}_count'],positive=values[f'd{d}t{t}_positive']))
        print(f'{dataset} {arm} seed{seed} epoch{epoch}/{epochs} step{step} selected={selected}',flush=True)
    torch.cuda.synchronize(device)
    if epochs == 50:
        assert step == conf['total_optimizer_steps']
    result = dict(dataset=dataset,arm=arm,seed=seed,attempt=attempt,epochs=epochs,global_step=step,
        selected_epochs=selected,selected_validation={k:states[e]['validation'][k] for k,e in selected.items()},
        init_hash=init_hash,epoch_order_hashes=order_hashes,states=states,config=c,diagnostics=diagnostics,
        training_seconds=time.perf_counter()-start,original_exposures=original_exposures,extra_views=extra_views,
        training_forwards=forward_count,gradient_calls=gradient_calls,optimizer_steps=step,
        online_validation_forwards=validation_forwards,shadow_validation_forwards=shadow_validation_forwards,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),snapshot_tensor_bytes=tensor_bytes(states),output_dir=str(out))
    write_json(out/'runtime_stats.json',{k:v for k,v in result.items() if k not in ('states','diagnostics')})
    write_json(out/'state_summary.json',dict(init_hash=init_hash,snapshots=[dict(epoch=e,global_step=s['global_step'],
        model_hash=state_digest(s['model']),shadow_hash=state_digest(s['shadow']),optimizer_hash=state_digest(s['optimizer']),
        shadow_updates=s['shadow_updates']) for e,s in states.items()]))
    return result


def worker(connection,bundles,dataset,p,arm,seed,device,out,attempt):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:
                result = train_one(bundles[dataset],dataset,p,arm,seed,device,out,connection,attempt=attempt)
                send_memory(connection,dict(kind='complete',result=result))
            except Exception as exc:
                traceback.print_exc()
                send_memory(connection,dict(kind='error',error=repr(exc),traceback=traceback.format_exc(),oom=isinstance(exc,torch.OutOfMemoryError)))
    connection.close()


def schedule(bundles,p,out,jobs,devices):
    ctx = mp.get_context('fork'); pending = list(jobs); active = []; completed = []; blocked = []
    peaks = {}; attempts = {}; serial_only = set(); allocation_pending = False
    while pending or active:
        for record in list(active):
            connection, proc = record['connection'],record['process']
            message = None
            if connection.poll():
                try:
                    message = receive_memory(connection)
                except EOFError:
                    message = dict(kind='error',error=f'worker exited {proc.exitcode} without RAM state',oom=False)
            elif not proc.is_alive():
                message = dict(kind='error',error=f'worker exited {proc.exitcode} without RAM state',oom=False)
            if message is None:
                continue
            if message['kind'] == 'allocated':
                record['allocated'] = True; allocation_pending = False
                peaks[record['gpu']] = max(peaks.get(record['gpu'],0),message['peak_memory_bytes'])
                continue
            proc.join(); connection.close(); active.remove(record)
            if not record['allocated']:
                allocation_pending = False
            if message['kind'] == 'complete':
                completed.append(message['result'])
            else:
                dataset, arm, seed = record['job']
                failure = dict(dataset=dataset,arm=arm,seed=seed,device=record['gpu'],attempt=record['attempt'],
                               oom=message['oom'],error=message['error'])
                append_csv(out/'failure_ledger.csv',failure)
                if message['oom'] and record['attempt'] == 1:
                    serial_only.add(record['job']); pending.append(record['job'])
                else:
                    blocked.append(failure)
            write_json(out/'progress.json',dict(stage='phase1',completed=len(completed),total=len(jobs),active=len(active),pending=len(pending),failed=len(blocked)))
        if pending and not allocation_pending and available_ram() > 8*1024**3:
            for status in gpu_status():
                gpu = status['index']
                if gpu not in devices or status['memory']/status['total'] >= .85 or status['power']/status['limit'] >= .85:
                    continue
                if (status['total']-status['memory'])*1024**2 < peaks.get(gpu,1500*1024**2)*1.15:
                    continue
                index = next((i for i,j in enumerate(pending) if j not in serial_only or not active),None)
                if index is None:
                    continue
                job = pending.pop(index); attempts[job] = attempts.get(job,0)+1
                dataset, arm, seed = job
                destination = out/'runs'/dataset/arm/f'seed{seed}'/f'attempt{attempts[job]}'
                parent, child = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=worker,args=(child,bundles,dataset,p,arm,seed,f'cuda:{gpu}',destination,attempts[job]))
                proc.start(); child.close()
                active.append(dict(connection=parent,process=proc,job=job,gpu=gpu,allocated=False,attempt=attempts[job]))
                append_csv(out/'launch_ledger.csv',dict(time=time.time(),dataset=dataset,arm=arm,seed=seed,attempt=attempts[job],
                    gpu=gpu,memory_fraction=status['memory']/status['total'],power_fraction=status['power']/status['limit'],active_workers=len(active)))
                allocation_pending = True; break
        time.sleep(.5)
    return completed,blocked


def freeze_selection(results,p,out,jobs):
    expected = set(jobs); actual = {(r['dataset'],r['arm'],r['seed']) for r in results}
    assert expected == actual and len(results) == len(expected)
    rows = []
    for result in results:
        assert result['epochs'] == 50 and result['global_step'] == p['datasets'][result['dataset']]['total_optimizer_steps']
        rows.extend(checkpoint_rows(result))
    frozen = dict(study_id=p['study_id'],execution_commit=p['execution_commit'],source=p['source'],
        effective_run_keys=[list(k) for k in sorted(expected)],checkpoints=rows,test_started=False)
    write_json(out/'selection_frozen.json',frozen)
    return frozen


def test_worker(connection,bundle,p,result,device):
    try:
        setup(result['seed'],device); model = make_model(bundle,p['datasets'][result['dataset']]).to(device)
        records = {}; start = time.perf_counter(); forwards = 0
        for row in checkpoint_rows(result):
            key = (row['model_kind'],row['epoch'])
            if key in records:
                continue
            state = result['states'][row['epoch']]
            model.load_state_dict(state['model' if row['model_kind'] == 'online' else 'shadow'])
            records[key] = evaluate(model,bundle.test_x,bundle.test_y,device)
            forwards += (len(bundle.test_y)+40959)//40960
        torch.cuda.synchronize(device)
        send_memory(connection,dict(kind='complete',metrics=records,test_seconds=time.perf_counter()-start,test_forwards=forwards))
    except Exception:
        send_memory(connection,dict(kind='error',error=traceback.format_exc()))


def evaluate_all(bundles,p,results,frozen,out,devices):
    assert len(frozen['effective_run_keys']) == len(results) and (out/'selection_frozen.json').exists()
    ctx = mp.get_context('fork')
    for result in results:
        gpu = min((s for s in gpu_status() if s['index'] in devices),key=lambda s:s['memory']/s['total'])['index']
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=test_worker,args=(child,bundles[result['dataset']],p,result,f'cuda:{gpu}'))
        proc.start(); child.close(); message = receive_memory(parent); proc.join(); parent.close()
        if message['kind'] == 'error':
            raise RuntimeError(message['error'])
        result['test_seconds'] = message['test_seconds']; result['test_forwards'] = message['test_forwards']
        for row in checkpoint_rows(result):
            validation = row.pop('validation'); key = (row['model_kind'],row['epoch'])
            test = message['metrics'][key]
            for split, values in [('validation',validation),('test',test)]:
                record = dict(**row,**values)
                append_csv(out/f'{split}_metrics.csv',record)
                append_csv(Path(result['output_dir'])/f'{split}_metrics.csv',record)
            append_csv(out/'selected_metrics.csv',dict(**row,val_macro_auc=validation['macro_auc'],
                val_macro_logloss=validation['macro_logloss'],test_macro_auc=test['macro_auc'],test_macro_logloss=test['macro_logloss']))
        print(f"test complete {result['dataset']} {result['arm']} seed{result['seed']}",flush=True)


def aggregate(results,out):
    import pandas as pd
    for name in ('epoch_metrics.csv','per_cell_metrics.csv','optimizer_state_diagnostics.csv','shadow_metrics.csv'):
        chunks = []
        for result in results:
            path = Path(result['output_dir'])/name
            if path.exists():
                table = pd.read_csv(path)
                for key in ('dataset','arm','seed','attempt'):
                    table[key] = result[key]
                chunks.append(table)
        if chunks:
            pd.concat(chunks,ignore_index=True).to_csv(out/name,index=False)
    for result in results:
        cost_keys = ['dataset','arm','seed','attempt','epochs','global_step','training_seconds','test_seconds',
            'original_exposures','extra_views','training_forwards','gradient_calls','optimizer_steps',
            'online_validation_forwards','shadow_validation_forwards','test_forwards','peak_memory_bytes','snapshot_tensor_bytes']
        append_csv(out/'costs.csv',{k:result[k] for k in cost_keys})
        append_csv(out/'run_ledger.csv',{k:result[k] for k in cost_keys+['output_dir','init_hash']})


def all_stages(bundles,p,out,devices):
    start = time.perf_counter()
    jobs = [(d,a['id'],s) for s in p['seeds'] for d in p['datasets'] for a in p['arms']]
    results, blocked = schedule(bundles,p,out,jobs,devices)
    if blocked:
        raise RuntimeError(f'Phase1 incomplete: {blocked}')
    assert len(results) == 40
    for dataset in bundles:
        for seed in p['seeds']:
            paired = [r for r in results if r['dataset'] == dataset and r['seed'] == seed]
            assert len(paired) == 4
            assert all(paired[0]['init_hash'] == r['init_hash'] and paired[0]['epoch_order_hashes'] == r['epoch_order_hashes'] for r in paired)
    gate = phase1_gate(results,p); write_json(out/'phase1_validation_decision.json',gate)
    if gate['triggered']:
        write_json(out/'progress.json',dict(stage='WAITING_FOR_P2_DESIGN_REVIEW',phase1_completed=40,
            phase1_snapshots_in_RAM=True,test_accessed=False,ready_file=str(out/'phase2_ready.json')))
        print('P2 triggered; retaining RAM states while waiting for phase2_ready.json',flush=True)
        while not (out/'phase2_ready.json').exists():
            time.sleep(5)
        ready = json.loads((out/'phase2_ready.json').read_text())
        assert ready['review_verdict'] in ('PASS','PASS WITH FIXES') and ready['blocking_fixes_complete']
        assert ready['implementation_commit'] and ready['budget_frozen']
        extension = importlib.import_module('experiments.gradient_aug_momentum_20261101.phase2')
        additional = extension.execute(bundles,p,results,out,devices,ready)
        results.extend(additional); jobs.extend((r['dataset'],r['arm'],r['seed']) for r in additional)
    frozen = freeze_selection(results,p,out,jobs)
    evaluate_all(bundles,p,results,frozen,out,devices)
    aggregate(results,out); build_tables(out,p)
    write_json(out/'completion.json',dict(status='COMPLETE',formal_runs=len(results),phase1_runs=40,
        total_epochs=sum(r['epochs'] for r in results),optimizer_steps=sum(r['global_step'] for r in results),
        test_evaluated=True,wall_seconds=time.perf_counter()-start,
        snapshot_tensor_bytes=sum(r['snapshot_tensor_bytes'] for r in results)))
    for result in results:
        result.pop('states')
    results.clear(); gc.collect()
    write_json(out/'state_release.json',dict(released=True,storage='RAM_ONLY',after='unified test and numerical tables'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['audit','check','all'],required=True)
    parser.add_argument('--protocol',default=str(DEFAULT_PROTOCOL)); parser.add_argument('--run-dir',required=True)
    parser.add_argument('--devices',default='0,1'); parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--code-commit',help='Immutable reviewed GAM01 implementation commit')
    args = parser.parse_args(); p = protocol(args.protocol); out = Path(args.run_dir)
    if args.mode == 'all' and not args.code_commit:
        parser.error('--mode all requires --code-commit after independent review and Stage0')
    p['execution_commit'] = args.code_commit
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new attempt: {out}')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    (out/'failure_ledger.csv').write_text('dataset,arm,seed,device,attempt,oom,error\n')
    write_json(out/'resolved_config.json',p)
    packages = [HERE,ROOT/'experiments/gradient_aug_20261031_v2']
    source_files = sorted(f for directory in packages for f in directory.rglob('*') if f.suffix in ('.py','.json'))
    write_json(out/'source_manifest.json',dict(source=p['source'],execution_commit=args.code_commit,
        files=[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in source_files]))
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
        gpus=gpu_status(),tf32=False,cudnn_benchmark=False,cpu_threads=1,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    bundles = {}; manifests = {}
    for dataset in p['datasets']:
        destination = out/'data'/dataset; destination.mkdir(parents=True,exist_ok=True)
        bundles[dataset],manifests[dataset] = load_dataset(dataset,p,out=destination)
        print(f'data identity PASS {dataset}',flush=True)
    write_json(out/'data_manifest.json',manifests)
    if args.mode == 'audit':
        return
    if args.mode == 'check':
        from .checks import run_checks
        write_json(out/'checks.json',run_checks(bundles,p,args.device,out)); return
    try:
        all_stages(bundles,p,out,[int(v) for v in args.devices.split(',')])
    except Exception as exc:
        write_json(out/'completion.json',dict(status='BLOCKED',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
