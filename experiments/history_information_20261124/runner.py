"""Finite stage execution and live selected-state custody in one foreground process."""
import csv
import hashlib
import json
import multiprocessing
import os
import resource
import signal
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from threading import current_thread, main_thread

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.evaluate import metrics
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import (
    cpu_copy, receive_memory, rng_state, send_memory, state_digest,
)
from . import diagnostics, objectives, scheduler, sources
from .observers import ANCHOR_EPOCHS, COMMON_EPOCHS, HistoryObserver, pair_record, selected_work
from .protocol import RunSpec, STUDY, digest
from .selection import EXTENSIONS, gaussian_paused_jobs
from .state import ANCHOR, EPOCH, SELECTED, RamStore, capture, live_hashes, restore

SOURCE_COMMIT = '358d3a465747eda6af1d55acc937759fa651a10b'
PUBLISHED_COMMIT = 'c8675ae2923c15b6965884b140b5b646690cd95e'
READOUTS = ('online','fixed_ema')
CELLS = tuple(f'd{d}t{t}' for d,t in diagnostics.CELLS)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, allow_nan=False, indent=2)+'\n',encoding='utf-8')
    pending.replace(path)


def _append_rows(path, rows):
    if not rows:
        return
    with path.open('a', encoding='utf-8') as stream:
        for value in rows:
            stream.write(json.dumps(value, allow_nan=False, separators=(',',':'))+'\n')


def _append(path, value):
    _append_rows(path,[value])


@contextmanager
def _append_transaction(paths):
    offsets = {p: p.stat().st_size if p.exists() else None for p in paths}
    try:
        yield
    except BaseException:
        for path,offset in offsets.items():
            if offset is None:
                path.unlink(missing_ok=True)
            else:
                with path.open('r+b') as stream:
                    stream.truncate(offset)
        raise


def _csv(path, rows):
    if not rows:
        return
    with path.open('a+', encoding='utf-8', newline='') as stream:
        stream.seek(0)
        keys = next(csv.reader(stream), None)
        if keys is None:
            keys = list(rows[0])
            writer = csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader()
        else:
            writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writerows(rows)


def _meta(spec, attempt, device, uuid, generation):
    return dict(study_id=STUDY, job_key=spec.job_key, logical_run_id=spec.job_key,
                attempt_id=attempt, dataset=spec.dataset, seed=spec.seed, stage=spec.stage,
                backbone=spec.backbone, arm=spec.arm, config_id=spec.arm, scope=spec.scope,
                algorithm_id=spec.algorithm_id, config_hash=spec.config_hash, source_hash=spec.source_hash,
                data_hash=spec.data_hash, group_hash=spec.group_hash, scope_hash=spec.group_hash,
                source_commit=PUBLISHED_COMMIT if spec.backbone!='m3oe' else SOURCE_COMMIT,
                owner_generation=generation, device=device, gpu_uuid=uuid,
                selection_readout=spec.selection_readout, parent_arm=None,
                representative_id=spec.target_algorithm_id, cost_class='NEW_SCIENTIFIC_TRAIN')


def _model_state(model, spec, at_epoch, value, identity):
    state = cpu_copy(model.state_dict())
    info = dict(identity, epoch=at_epoch, model_hash=state_digest(state),
                buffers_hash=state_digest(dict(model.named_buffers())), mode=model.training,
                model_meta=objectives.inference_meta(model,spec))
    return dict(model_state=state, identity=info, validation=value)


def _endpoint(row):
    metric = {k:v for k,v in row.items() if k not in ('state_hash','evaluation_seconds','n','n_positive','n_negative',
                                                       'metric_name','value','cell')}
    return [dict(metric, cell=c, auc=metric[f'{c}_auc'], logloss=metric[f'{c}_logloss'],
                 cell_auc=metric[f'{c}_auc'], cell_logloss=metric[f'{c}_logloss'],
                 count=metric[f'{c}_count'], positive=metric[f'{c}_positive'], negative=metric[f'{c}_negative'],
                 traffic_auc=metric['sample_weighted_cell_auc'], pooled_auc=metric['pooled_task_mean_auc'],
                 traffic_logloss=metric['sample_weighted_cell_logloss'], pooled_logloss=metric['pooled_logloss'],
                 source_type='NEW_TRAIN') for c in CELLS]


def _curve(identity, checkpoint, epoch, steps, events, state, readout, scores, cost, losses):
    model_hash, buffers_hash = state['model_hash'], state['buffers_hash']
    common = dict(identity, checkpoint=checkpoint, endpoint=checkpoint, epoch=epoch, selected_epoch=epoch,
                  global_step=steps, native_steps=steps, pair_events=events, readout=readout, split='validation',
                  model_hash=model_hash, buffers_hash=buffers_hash, state_hash=model_hash,
                  evaluation_seconds=cost, status='OK', **scores, **losses)
    rows = []
    for name in ('auc', 'logloss'):
        for cell in CELLS:
            rows.append(dict(common, metric_name=name, cell=cell, value=scores[f'{cell}_{name}'],
                             n=scores[f'{cell}_count'], n_positive=scores[f'{cell}_positive'], n_negative=scores[f'{cell}_negative']))
        for level in ('macro', 'sample_weighted_cell'):
            key = f'{level}_{name}'
            rows.append(dict(common,metric_name=key,cell='all',value=scores[key],n=None,n_positive=None,n_negative=None))
        pooled_key = 'pooled_task_mean_auc' if name == 'auc' else 'pooled_logloss'
        rows.append(dict(common,metric_name=pooled_key,cell='all',value=scores[pooled_key],n=None,n_positive=None,n_negative=None))
    return common, rows


def _exchange(connection, value):
    send_memory(connection,value)
    response = receive_memory(connection)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


def _readout_epoch(spec, epoch):
    return epoch-10 if spec.backbone=='autoifs' else epoch


def _prefix_summary(work):
    return {k: (int(v) if isinstance(v,np.integer) else v) for k,v in work.items()}


def _clock_summary(optimizer):
    moments = getattr(optimizer,'moments',{})
    global_k = getattr(optimizer,'pair_clock',0)
    values = [(name,m['step'],m.get('k',global_k)) for name,m in moments.items()]
    return dict(global_pair_clock=global_k, custom_native_range=[min(v[1] for v in values),max(v[1] for v in values)] if values else None,
                custom_pair_range=[min(v[2] for v in values),max(v[2] for v in values)] if values else None,
                custom_clocks_hash=digest(values))


def _worker(spec_data, identity, cache_root, initial, connection, end, artifacts):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['CUDA_VISIBLE_DEVICES'] = identity['gpu_uuid']
    device = torch.device('cuda:0')
    spec = RunSpec.from_dict(spec_data)
    started = time.monotonic()
    baseline = initial['global_step'] if initial else 0
    base_epoch = initial['epoch'] if initial else 0
    accepted = base_epoch
    physical_steps = 0
    current_epoch = base_epoch
    steps = baseline
    allocation, work, window_work, attempts = {}, Counter(), Counter(), Counter()
    submitting = False
    try:
        bundle = sources.load_train_validation(spec.dataset,cache_root=cache_root)
        if bundle.data_hash != spec.data_hash:
            raise ValueError('Worker Train/Validation identity differs from the queue')
        model, optimizer, ema = objectives.build_training(spec,bundle,device)
        torch.cuda.reset_peak_memory_stats(device)
        total = torch.cuda.mem_get_info(device)[1]
        allocation.update(cuda_usable_total_bytes=total, model_initialization_hash=state_digest(model.state_dict()))
        send_memory(connection,dict(kind='initialized', **allocation))
        data_modes = selected_work(spec)
        observer = HistoryObserver(spec,optimizer) if data_modes['d1'] else None
        loader = objectives.loader_generator(spec)
        selected, pair_events = {}, 0
        if initial:
            recovery_id = {k:v for k,v in identity.items() if k not in ('attempt_id','selection_readout','parent_arm','representative_id','cost_class')}
            old_id = initial['identity']
            if recovery_id != {k:old_id[k] for k in recovery_id}:
                raise ValueError('Recovery identity or actual GPU differs from the verified packet')
            restore(initial,old_id,model,optimizer,ema,loader,observer,device,objectives.restore_inference_meta)
            selected, pair_events = initial['selected'],initial['pair_events']
            work.update(initial['work'])
            del initial
        bank = None
        for epoch in range(base_epoch+1,spec.epochs+1):
            current_epoch = epoch
            if end.is_set() or scheduler.stop_requested(artifacts):
                break
            window_work.clear()
            objectives.prepare_epoch(model,optimizer,ema,spec,epoch)
            model.train()
            if ema is not None:
                ema.train()
            order = objectives.training_order(spec,bundle,epoch,loader)
            if (len(order)+spec.batch_size-1)//spec.batch_size != spec.steps_per_epoch:
                raise ValueError('Source epoch batch count differs from RunSpec')
            steps_log, records = [], []
            elapsed_epoch = time.monotonic()
            losses, real_losses = 0., 0.
            at = _readout_epoch(spec,epoch)
            train, diagnostic_time = 0., 0.
            for batch,offset in enumerate(range(0,len(order),spec.batch_size),1):
                start = time.monotonic()
                indices = order[offset:offset+spec.batch_size]
                xs, ys = bundle.train_x[indices], bundle.train_y[indices]
                x = torch.as_tensor(xs,device=device).T
                y = torch.as_tensor(ys,device=device)
                rows = bundle.train_ids[indices]
                steps += 1
                last = offset+len(indices)==len(order)
                anchor_now = data_modes['d2'] and last and epoch in ANCHOR_EPOCHS
                context = dict(study='EMA02V2', dataset=spec.dataset,split_id='original',seed=spec.seed,
                               epoch=epoch,global_step=steps,batch=batch,last_batch=last,role='NEW_SCIENTIFIC_TRAIN')
                pre_rng = rng_state(device) if anchor_now else None
                pre_buffers = cpu_copy(dict(model.named_buffers())) if anchor_now else None
                pre_info, anchor = {}, None

                def pre_step(step_context, pair, raw):
                    nonlocal anchor, submitting
                    pre_info.update(pair_record(pair,xs[:,-1],ys))
                    none = [n for n,p in model.named_parameters() if p.grad is None]
                    pre_info.update(native_none=none,native_none_mask_hash=digest(none))
                    if anchor_now:
                        anchor = capture(ANCHOR,identity,model,optimizer,ema,loader,observer,device,epoch,steps,order,selected,
                                         cursor=offset, model_meta=objectives.inference_meta(model,spec), context=context,
                                         batch_indices=indices.copy(),row_ids=rows.copy(),pre_forward_rng=pre_rng,
                                         pre_forward_buffers=pre_buffers,post_forward_buffers=cpu_copy(dict(model.named_buffers())),
                                         native_grad=cpu_copy({n:p.grad for n,p in model.named_parameters()}),
                                         pair=cpu_copy(pair),raw_pair=cpu_copy(raw),pair_events=pair_events,
                                         global_step_before=steps-1)
                    if observer:
                        observer.observe(pair,raw)
                    submitting = True

                attempts['physical_batches_attempted'] += 1
                value = objectives.step(model,optimizer,ema,x,y,rows,spec,context,pre_step)
                submitting = False
                if not pre_info:
                    pre_info.update(pair_record(None,xs[:,-1],ys))
                    info = value.get('diagnostics',{})
                    if info.get('pair_count'):
                        raise RuntimeError('PAIR provenance was absent from the actual pre-step callback')
                    none = [n for n,p in model.named_parameters() if p.grad is None]
                    pre_info.update(native_none=none,native_none_mask_hash=digest(none))
                with torch.no_grad():
                    finite = torch.stack([torch.isfinite(p).all() for p in model.state_dict().values() if p.is_floating_point()]).all()
                    if not bool(finite) or not np.isfinite(value['loss']):
                        raise FloatingPointError('Nonfinite model or loss after the step')
                physical_steps += 1
                pair_events += int(pre_info['pair_event'])
                group_records = observer.finish() if observer else []
                step_counts = value['counts']
                window_work.update(step_counts)
                work.update(step_counts)
                losses += len(rows)*value['loss']
                real_losses += len(rows)*value['real_loss']
                steps_log.append(dict(identity,epoch=epoch,global_step=steps,batch=batch,rows=len(rows),
                                      row_id_hash=state_digest(rows),loss=value['loss'],real_loss=value['real_loss'],
                                      native_step=steps,pair_events=pair_events,phase='E1_10' if epoch<=10 else 'E11_25' if epoch<=25 else 'E26_50',
                                      groups=group_records,work=_prefix_summary(step_counts),
                                      clocks=_clock_summary(optimizer),
                                      source_diagnostics=value.get('diagnostics',{}),**pre_info))
                if physical_steps==1:
                    torch.cuda.synchronize(device)
                    _exchange(connection,dict(kind='allocated',allocated=torch.cuda.max_memory_allocated(device),
                                              reserved=torch.cuda.max_memory_reserved(device)))
                train += time.monotonic()-start
                if anchor_now:
                    start_diag = time.monotonic()
                    if anchor is None:
                        raise RuntimeError('Actual H0 post-backward anchor was absent')
                    anchor['actual_post'] = live_hashes(model,optimizer,ema)
                    signature = state_digest(anchor)
                    del value,x,y
                    response = _exchange(connection,dict(kind='anchor',packet=anchor,anchor_hash=signature))
                    anchor = response['packet']
                    if state_digest(anchor) != signature:
                        raise RuntimeError('Worker observed a changed anchor after foreground roundtrip')
                    live_before = (live_hashes(model,optimizer,ema),state_digest(rng_state(device)),
                                   state_digest(loader.get_state()),state_digest(selected),state_digest(observer.state_dict()))
                    if bank is None:
                        bank = diagnostics.panels(spec.dataset,spec.seed,bundle)
                        diagnostics.attach_pairs(bank,spec.dataset,spec.seed,bundle)
                        for name,attrs in bank.items():
                            records.append(dict(identity,diagnostic='D2_PANEL_MANIFEST',epoch=epoch,panel=name,
                                                panel_hash=attrs['panel_hash'],split=attrs['split'],population=attrs['population'],
                                                strata=attrs['strata']))
                    out, measured = diagnostics.anchor_diagnostics(model,optimizer,observer,anchor,bank,device,spec.eval_batch_size)
                    live_after = (live_hashes(model,optimizer,ema),state_digest(rng_state(device)),
                                  state_digest(loader.get_state()),state_digest(selected),state_digest(observer.state_dict()))
                    if live_before != live_after:
                        raise RuntimeError('A diagnostic changed live model/optimizer/EMA/RNG/loader/selected/observer state')
                    records.extend(dict(identity,epoch=epoch,global_step=steps,anchor_hash=signature,**{
                        k:v for k,v in row.items() if k!='anchor_hash'}) for row in out)
                    window_work.update(measured)
                    work.update(measured)
                    diagnostic_time += time.monotonic()-start_diag
                    del anchor,response
                else:
                    del value,x,y
            curves, endpoints, probs = [], [], {}
            eval_seconds = 0.
            if at>0:
                for readout in spec.readouts:
                    estimator = model if readout=='online' else ema
                    start_eval = time.monotonic()
                    pred, ev = objectives.predict(spec,estimator,bundle.val_x,device)
                    if not np.isfinite(pred).all():
                        raise FloatingPointError('Full Validation predictions must be finite')
                    scores = metrics(bundle.val_y,pred,bundle.val_x[:,-1])
                    if scores['macro_auc'] is None:
                        raise FloatingPointError('Complete Validation requires finite support in all six cells')
                    eval_time = time.monotonic()-start_eval
                    eval_seconds += eval_time
                    forecast = dict(model_hash=state_digest(estimator.state_dict()),
                                    buffers_hash=state_digest(dict(estimator.named_buffers())))
                    losses_row = dict(loss=losses/len(order), real_loss=real_losses/len(order))
                    summary, long = _curve(identity,f'E{at}',at,steps,pair_events,forecast,readout,scores,eval_time,losses_row)
                    curves.extend(long)
                    if at in COMMON_EPOCHS:
                        endpoints.extend(_endpoint(summary))
                        if data_modes['d4']:
                            probs[f'E{at}',readout] = dict(pred=pred, epoch=at, model_hash=summary['model_hash'])
                    if readout not in selected or scores['macro_auc']>selected[readout]['validation']['macro_auc']:
                        sel = _model_state(estimator,spec,at,scores,dict(identity,readout=readout))
                        sel['summary'] = dict(summary,checkpoint='selected',endpoint='selected')
                        if data_modes['d4']:
                            sel['validation_probabilities'] = pred
                        selected[readout] = sel
                    for counter in (window_work,work):
                        counter['eval_forwards'] += ev['forwards']
                        counter['eval_forward_rows'] += ev['rows']
                    del pred
            objectives.finish_epoch(model,optimizer,ema,spec,epoch)
            # This read sees stop requests received during this epoch before the acknowledged commit.
            boundary_stop = end.is_set() or scheduler.stop_requested(artifacts)
            if epoch == spec.epochs:
                for readout,sel in selected.items():
                    endpoints.extend(_endpoint(sel['summary']))
                    if data_modes['d4']:
                        probs['selected',readout] = dict(pred=sel['validation_probabilities'],epoch=sel['identity']['epoch'],
                                                         model_hash=sel['identity']['model_hash'])
            cost = dict(identity,epoch=epoch,epoch_count=1,global_step=steps,global_batches=len(steps_log),
                        optimizer_calls=window_work.get('optimizer_steps',0),
                        autograd_sweeps=window_work.get('shared_autograd_calls',0), train_forwards=window_work.get('training_forwards',0),
                        eval_forwards=window_work['eval_forwards'],train_seconds=train,eval_seconds=eval_seconds,
                        diagnostic_seconds=diagnostic_time,worker_seconds=time.monotonic()-elapsed_epoch,
                        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),max_process_gpu_bytes=0,
                        peak_host_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        physical_replay_steps=0, physical_batches_attempted=attempts['physical_batches_attempted'],
                        diagnostic_reference_steps=window_work['diagnostic_reference_steps'],
                        diagnostic_virtual_steps=window_work['diagnostic_virtual_steps'],
                        diagnostic_probe_forwards=window_work['diagnostic_probe_forwards'],
                        diagnostic_probe_autograd_sweeps=window_work['diagnostic_probe_autograd_sweeps'],
                        diagnostic_panel_evaluations=window_work['diagnostic_panel_evaluations'],
                        diagnostic_panel_forwards=window_work['diagnostic_panel_forwards'],status='COMPLETE_EPOCH')
            packet = capture(EPOCH,identity,model,optimizer,ema,loader,observer,device,epoch,steps,order,selected,
                             cursor=len(order),model_meta=objectives.inference_meta(model,spec),pair_events=pair_events,
                             transaction=epoch,original_successful_steps=steps,physical_steps=physical_steps,
                             physical_replay_steps=0,ema_rho=spec.rho,work=_prefix_summary(work),
                             boundary_stop=boundary_stop,model_initialization_hash=allocation['model_initialization_hash'],
                             receipts=dict(endpoints=[(r['readout'],r['endpoint'],r['model_hash']) for r in endpoints if r['cell']==CELLS[0]],
                                           anchor=next((r['anchor_hash'] for r in records if r['diagnostic']=='D2_REPLAY'),None),
                                           scope=scheduler.STUDY,owner_generation=identity['owner_generation']))
            _exchange(connection,dict(kind='epoch',packet=packet,curves=curves,endpoints=endpoints,steps=steps_log,
                                      diagnostic_records=records,probabilities=probs,cost=cost))
            accepted = epoch
            attempts.clear()
            del packet,probs,steps_log,records,curves,endpoints
            if boundary_stop:
                break
        if accepted==spec.epochs:
            send_memory(connection,dict(kind='complete',epoch=accepted,worker_seconds=time.monotonic()-started))
        else:
            send_memory(connection,dict(kind='stopped',epoch=accepted,worker_seconds=time.monotonic()-started))
    except BaseException as error:
        oom = isinstance(error,torch.OutOfMemoryError)
        details = dict(kind='failure', reason=type(error).__name__, message=str(error), trace=traceback.format_exc(),
                       oom=oom, failed_allocation_bytes=scheduler.allocation_bytes(error) if oom else None,
                       epoch=current_epoch, last_verified_epoch=accepted, global_step=steps, partial_submission=submitting,
                       physical_steps=physical_steps, physical_batches_attempted=attempts['physical_batches_attempted'],
                       uncommitted_costs=_prefix_summary(window_work),worker_seconds=time.monotonic()-started,
                       worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       cuda_usable_total_bytes=allocation.get('cuda_usable_total_bytes'), batch_size=spec.batch_size,
                       cuda_allocated_bytes=torch.cuda.memory_allocated(device) if torch.cuda.is_initialized() else None,
                       cuda_reserved_bytes=torch.cuda.memory_reserved(device) if torch.cuda.is_initialized() else None)
        try:
            _exchange(connection,details)
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass
    finally:
        connection.close()


class Supervisor:
    """The single RAM owner for all commands accepted by one supervise PTY."""
    def __init__(self, artifact_root, *, emit=None):
        self.artifacts = Path(artifact_root).resolve()
        self.emit = emit or (lambda payload: print(json.dumps(payload,allow_nan=False),flush=True))
        self.store = RamStore()
        self.context = multiprocessing.get_context('spawn')
        self.jobs = {}
        self.live = {}
        self.cache = None
        self.inputs = set()
        self.cuda_totals = {}
        self.footprints = {}
        self.after_oom = False
        self.closed = False
        self.rotor = 0

    def _inputs(self, specs):
        given = self.artifacts/'train_validation_inputs'
        if given.is_dir():
            return given
        if self.cache is None:
            self.cache = tempfile.TemporaryDirectory(prefix=STUDY+'-',dir='/dev/shm')
        root = Path(self.cache.name)
        for name in dict.fromkeys(spec.dataset for spec in specs):
            if name not in self.inputs:
                sources.prepare_train_validation_cache(root,name)
                self.inputs.add(name)
        return root

    @staticmethod
    def _profile(spec):
        return spec.dataset,spec.backbone,spec.config_hash,spec.source_hash,spec.group_hash

    def _measure(self, measurements, worker):
        value = scheduler.residency(measurements,worker['process'].pid,worker['uuid']) if measurements else None
        if measurements:
            worker['host_peak'] = max(worker['host_peak'],measurements['host_ram_used_bytes'])
        if value is not None:
            worker['resident'] = value
            worker['peak'] = max(worker['peak'],value)
        return value

    def _log(self, out, event, **value):
        row = dict(event=event,time=time.time(),**value)
        _append(out/'admissions.jsonl',row)
        self.emit(row)

    def _start(self, out, job, gpu, inputs, end, measurements, admission_mode):
        spec, key = job['spec'],job['spec'].job_key
        identity = _meta(spec, job['attempt']+1, f"cuda:{gpu['ordinal']}", gpu['uuid'], self.store.generation)
        initial = self.store.get(f'{key}:current')[0] if job['epoch'] else None
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker, args=(spec.to_dict(),identity,str(inputs),initial,child,end,str(self.artifacts)))
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        job.update(attempt=identity['attempt_id'],status='RUNNING',reason=None,uuid=gpu['uuid'])
        self.live[key] = dict(process=process,connection=parent,identity=identity,uuid=gpu['uuid'],
                              allocated=False,peak=0,resident=None,started=time.time(),terminal=None,
                              admission_mode=admission_mode,host_peak=measurements['host_ram_used_bytes'])
        self._log(out,'ADMITTED',job_key=key,attempt_id=job['attempt'],pid=process.pid,gpu=gpu,
                  host_ram_used=measurements['host_ram_used'],physical_capacity_active=self.after_oom,
                  physical_fit_mode=admission_mode,
                  cuda_usable_total_bytes=self.cuda_totals.get(gpu['uuid']),
                  full_process_bytes=self.footprints.get(self._profile(spec)))

    def _predictions(self, out, identity, probabilities):
        if not probabilities:
            return
        directory = out/'predictions'
        directory.mkdir(exist_ok=True)
        for (checkpoint,readout),row in probabilities.items():
            path = directory/f"{identity['job_key']}_{checkpoint}_{readout}.npy"
            pending = path.with_suffix('.pending')
            with pending.open('wb') as stream:
                np.save(stream,row['pred'],allow_pickle=False)
            pending.replace(path)
            infos = dict(identity,checkpoint=checkpoint,endpoint=checkpoint,readout=readout,
                         epoch=row['epoch'],model_hash=row['model_hash'],prediction_hash=state_digest(row['pred']),
                         path=str(path),relative_path=str(path.relative_to(self.artifacts)),
                         file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(row['pred']),split='validation')
            _append(out/'predictions.jsonl',infos)

    def _anchor(self, out, message, job, worker):
        packet = message['packet']
        key = f"{job['spec'].job_key}:anchor:{packet['epoch']}"
        old = job['anchors'].get(key)
        if old:
            saved, _ = self.store.get(key)
            # Physical attempt identity may differ, while the live anchor content must replay exactly.
            old_sig, new_sig = (state_digest({k:v for k,v in p.items() if k!='identity'}) for p in (saved,packet))
            if old_sig != new_sig:
                raise RuntimeError('Replayed anchor does not equal its preceding captured state')
        receipt = self.store.commit(key,packet,worker['identity'])
        job['anchors'][key] = receipt
        _append(out/'state_receipts.jsonl',receipt)
        saved, _ = self.store.get(key,worker['identity'])
        if state_digest(saved) != message['anchor_hash']:
            raise RuntimeError('Anchor worker/owner hashes differ')
        return dict(packet=saved)

    def _epoch(self, out, message, job, worker):
        packet = message['packet']
        spec, identity = job['spec'],worker['identity']
        key = spec.job_key
        epoch = packet['epoch']
        if epoch != job['epoch']+1 or packet['global_step'] != epoch*spec.steps_per_epoch:
            raise ValueError('Epoch packet is out of order or skips a registered batch')
        current, pending = f'{key}:current', f'{key}:epoch_pending'
        receipt = dict(self.store.commit(pending,packet,identity),guardian_key=current)
        known = ('state_receipts.jsonl','steps.jsonl','diagnostics.jsonl','curves.csv','endpoints.csv',
                 'predictions.jsonl','costs.csv')
        paths = [out/name for name in known]
        for checkpoint,readout in message['probabilities']:
            prediction = out/'predictions'/f'{key}_{checkpoint}_{readout}.npy'
            if prediction.exists():
                self.store.release(pending)
                raise RuntimeError('An unacknowledged epoch already has a prediction artifact')
            paths.extend((prediction,prediction.with_suffix('.pending')))
        cost = message['cost']
        cost['max_process_gpu_bytes'] = worker['peak'] or None
        cost['worker_peak_rss_bytes'] = cost['peak_host_ram_bytes']
        cost['peak_host_ram_bytes'] = worker['host_peak']
        cost['owner_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        anchor = f'{key}:anchor:{epoch}'
        try:
            with _append_transaction(paths):
                _append(out/'state_receipts.jsonl',receipt)
                _append_rows(out/'steps.jsonl',message['steps'])
                _append_rows(out/'diagnostics.jsonl',message['diagnostic_records'])
                _csv(out/'curves.csv',message['curves'])
                _csv(out/'endpoints.csv',message['endpoints'])
                self._predictions(out,identity,message['probabilities'])
                _csv(out/'costs.csv',[cost])
                if anchor in job['anchors']:
                    _append(out/'state_receipts.jsonl',dict(job['anchors'][anchor],available=False,
                            release_reason='DIAGNOSTICS_COMMITTED',released_at=time.time()))
            self.store.promote(pending,current)
        except BaseException:
            self.store.release(pending)
            raise
        job['epoch'] = epoch
        if worker['peak']:
            profile = self._profile(spec)
            self.footprints[profile] = max(worker['peak'],self.footprints.get(profile,0))
        if anchor in job['anchors']:
            self.store.release(anchor)
            job['anchors'].pop(anchor)
        self.emit(dict(event='EPOCH_COMMITTED',stage=spec.stage,job_key=key,attempt_id=job['attempt'],
                       epoch=epoch,ram_hash=receipt['get_hash']))
        return dict(get_hash=receipt['get_hash'])

    def _finish(self, out, job, worker):
        spec, key = job['spec'],job['spec'].job_key
        packet, _ = self.store.get(f'{key}:current',worker['identity'])
        if packet['epoch'] != spec.epochs or set(packet['selected']) != set(spec.readouts):
            raise RuntimeError('Completed worker lacks a full budget and both registered selected readouts')
        for readout, choice in packet['selected'].items():
            identity = choice['identity']
            if any(identity[k] != worker['identity'][k] for k in ('owner_generation','job_key','dataset','seed',
                                                                 'source_hash','group_hash','data_hash','config_hash','gpu_uuid')):
                raise RuntimeError('Selected model identity differs from the completed worker')
            selected = dict(schema=SELECTED,identity=identity,epoch=identity['epoch'],
                            global_step=choice['summary']['global_step'],model_state=choice['model_state'],
                            validation=choice['validation'])
            receipt = self.store.commit(f'{key}:selected:{readout}',selected,identity)
            _append(out/'state_receipts.jsonl',receipt)
            job['selected'][readout] = identity['model_hash']
        self.store.release(f'{key}:current')
        _append(out/'state_receipts.jsonl',dict(run_id=key,state_kind=EPOCH,guardian_key=f'{key}:current',
                                               available=False,release_reason='FINAL_SELECTED_VERIFIED',released_at=time.time()))
        job.update(status='COMPLETE',reason=None)

    def _oom(self, out, job, worker, error, measurements):
        self.after_oom = True
        profile = self._profile(job['spec'])
        reserved = error['cuda_reserved_bytes']
        allocated = error['cuda_allocated_bytes']
        need = error['failed_allocation_bytes']
        # Full process minus torch reservation supplies observed CUDA/context overhead.
        measurements = error.get('oom_measurements') or measurements
        resident = self._measure(measurements,worker) or worker['resident']
        if resident and allocated is not None and reserved is not None and need:
            footprint = max(worker['peak'],allocated+need+max(resident-reserved,0))
            self.footprints[profile] = max(footprint,self.footprints.get(profile,0))
        else:
            self.footprints.pop(profile,None)
        spec = job['spec']
        maximum = {'movielens':325619,'kuairand':1743898}
        domains = {'movielens':2,'kuairand':1}
        self._log(out,'CUDA_OOM',job_key=spec.job_key,attempt_id=job['attempt'],gpu_uuid=worker['uuid'],
                  current_processes=measurements['processes'] if measurements else None,
                  failed_allocation_bytes=need,allocator_allocated_bytes=allocated,allocator_reserved_bytes=reserved,
                  cuda_usable_total_bytes=error['cuda_usable_total_bytes'],batch_size=spec.batch_size,
                  max_domain=domains[spec.dataset],max_domain_train_rows=maximum[spec.dataset],
                  full_process_bytes=self.footprints.get(profile),observed_peak_process_bytes=worker['peak'] or None)

    def _terminal(self, out, job, worker, measurements, end):
        message = worker['terminal']
        kind = message['kind'] if message else 'lost'
        spec, key = job['spec'],job['spec'].job_key
        if kind in ('lost','failure') and job['epoch']==spec.epochs:
            # A final acknowledgement can be lost after the owner has committed the full packet.
            kind = 'complete'
        if kind=='complete':
            self._finish(out,job,worker)
        elif kind=='stopped':
            job.update(status='TECHNICAL_PAUSED',reason='WORKSPACE_STOP_OR_FOREGROUND_INTERRUPT')
        else:
            error = message if kind=='failure' else dict(reason='WORKER_LOST',message='Worker exited without a terminal receipt')
            _append(out/'failures.jsonl',dict(worker['identity'],**error,process_exit_code=worker['process'].exitcode,
                                             measured_peak_process_bytes=worker['peak'] or None))
            if error.get('oom'):
                self._oom(out,job,worker,error,measurements)
            replayed = error.get('physical_batches_attempted',0)
            uncommitted = error.get('uncommitted_costs',{})
            job['failed_physical_batches'] += replayed
            if replayed or error.get('worker_seconds'):
                _csv(out/'costs.csv',[dict(worker['identity'],epoch=error.get('epoch'),epoch_count=0,
                    global_step=error.get('global_step'),global_batches=0,optimizer_calls=uncommitted.get('optimizer_steps',0),
                    autograd_sweeps=uncommitted.get('shared_autograd_calls',0),train_forwards=uncommitted.get('training_forwards',0),
                    eval_forwards=uncommitted.get('eval_forwards',0),train_seconds=None,eval_seconds=None,diagnostic_seconds=None,
                    worker_seconds=error.get('worker_seconds'),max_gpu_allocated_bytes=error.get('cuda_allocated_bytes'),
                    max_gpu_reserved_bytes=error.get('cuda_reserved_bytes'),max_process_gpu_bytes=worker['peak'] or None,
                    peak_host_ram_bytes=worker['host_peak'],physical_replay_steps=replayed,physical_batches_attempted=replayed,
                    diagnostic_reference_steps=uncommitted.get('diagnostic_reference_steps',0),
                    diagnostic_virtual_steps=uncommitted.get('diagnostic_virtual_steps',0),
                    diagnostic_probe_forwards=uncommitted.get('diagnostic_probe_forwards',0),
                    diagnostic_probe_autograd_sweeps=uncommitted.get('diagnostic_probe_autograd_sweeps',0),
                    diagnostic_panel_evaluations=uncommitted.get('diagnostic_panel_evaluations',0),
                    diagnostic_panel_forwards=uncommitted.get('diagnostic_panel_forwards',0),status='UNCOMMITTED_REPLAY_COST',
                    worker_peak_rss_bytes=error.get('worker_peak_rss_bytes'),
                    owner_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)])
            recoverable = (error.get('oom') or error['reason'] in ('FloatingPointError','WORKER_LOST')
                           or message and message.get('partial_submission'))
            repeated_boundary = job['failed_epoch']==job['epoch'] and (
                not error.get('oom') or worker['admission_mode']=='QUIESCENT_WORKLOAD_MEASUREMENT')
            job.update(status='TECHNICAL_PAUSED',reason=error['reason'],failed_epoch=job['epoch'])
            if recoverable and not repeated_boundary and not end.is_set():
                job.update(status='READY',reason=error['reason'])
        _csv(out/'runs.csv',[dict(worker['identity'],state=job['status'],status=job['status'],
            started_at=worker['started'],ended_at=time.time(),epoch=job['epoch'],physical_replay=job['attempt']>1,
            replay_reason=job['launch_reason'],reason=job['reason'],queue_hash=job['queue_hash'])])
        self._log(out,'WORKER_EXIT',job_key=key,attempt_id=job['attempt'],status=job['status'],epoch=job['epoch'])

    def _drain(self, out, measurements, end):
        resample = False
        for key,worker in list(self.live.items()):
            channel,pid = worker['connection'],worker['process']
            job = self.jobs[key]
            self._measure(measurements,worker)
            try:
                while channel.poll():
                    row = receive_memory(channel)
                    kind = row['kind']
                    if kind=='initialized':
                        self.cuda_totals[worker['uuid']] = row['cuda_usable_total_bytes']
                    elif kind=='allocated':
                        worker['allocated'] = True
                        resample = True
                        send_memory(channel,dict(ok=True))
                    elif kind=='failure':
                        worker['terminal'] = row
                        # Preserve a full-process sample while the failed process still owns CUDA.
                        if row['oom']:
                            try:
                                row['oom_measurements'] = scheduler.sample()
                                self._measure(row['oom_measurements'],worker)
                            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
                                pass
                        send_memory(channel,dict(ok=True))
                    elif kind in ('complete','stopped'):
                        worker['terminal'] = row
                    elif kind in ('epoch','anchor'):
                        try:
                            if kind=='epoch':
                                ack = self._epoch(out,row,job,worker)
                            else:
                                ack = self._anchor(out,row,job,worker)
                        except Exception as exc:
                            ack = dict(error=f'{type(exc).__name__}: {exc}')
                            job['owner_failure'] = ack['error']
                        send_memory(channel,ack)
                    else:
                        raise RuntimeError(f'Invalid worker message {kind}')
            except (EOFError,BrokenPipeError,ConnectionResetError):
                pass
            if not pid.is_alive():
                pid.join()
                channel.close()
                self.live.pop(key)
                if job.get('owner_failure'):
                    job.update(status='TECHNICAL_PAUSED',reason=job['owner_failure'])
                    self._log(out,'WORKER_OWNER_FAILURE',job_key=key,reason=job['reason'])
                else:
                    self._terminal(out,job,worker,measurements,end)
                pid.close()
        return resample

    @staticmethod
    def _visible(devices):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            return [dict(row,ordinal=row['index']) for row in devices]
        result = []
        for ordinal, name in enumerate(visible.split(',')):
            match = [r for r in devices if name==str(r['index']) or (name.startswith('GPU-') and r['uuid'].startswith(name))]
            if len(match)==1:
                result.append(dict(match[0],ordinal=ordinal))
        return result

    def _device_capacity(self, gpu):
        uuid = gpu['uuid']
        if uuid not in self.cuda_totals:
            try:
                capacity = next(p.total_memory for index in range(torch.cuda.device_count())
                                if str((p:=torch.cuda.get_device_properties(index)).uuid).removeprefix('GPU-')
                                == uuid.removeprefix('GPU-'))
                self.cuda_totals[uuid] = capacity
            except (AttributeError,StopIteration,RuntimeError):
                return None
        return self.cuda_totals[uuid]

    def _queue(self, queue):
        if queue['schema_version'] != 1 or queue['study_id'] != STUDY or queue['queue_hash'] != digest({
                k:v for k,v in queue.items() if k!='queue_hash'}):
            raise ValueError('Queue identity did not verify')
        specs = [RunSpec.from_dict(row) for row in queue['jobs']]
        if len({s.job_key for s in specs}) != len(specs) or any(s.stage!=queue['stage'] for s in specs):
            raise ValueError('Queue contains duplicate jobs or a different stage')
        for spec in specs:
            if spec.to_dict() not in queue['jobs'] or any(len(getattr(spec,k)) != 64 for k in ('source_hash','data_hash','group_hash')):
                raise ValueError('Queue must carry the full resolved trajectory and frozen identities')
        for path, expected in queue.get('prerequisite_receipts',{}).items():
            receipt = (self.artifacts/path).resolve()
            if not receipt.is_relative_to(self.artifacts) or hashlib.sha256(receipt.read_bytes()).hexdigest()!=expected:
                raise ValueError('Queue prerequisite receipt differs from its frozen hash')
        return specs

    def run(self, queue, *, conditional_pause=None):
        if self.closed or self.live:
            raise RuntimeError('Foreground owner is closed or still dispatching a command')
        specs = self._queue(queue)
        stage = queue['stage']
        out = self.artifacts/stage
        if conditional_pause is not None:
            blocked = gaussian_paused_jobs(queue, conditional_pause)
            specs = [spec for spec in specs if spec.job_key not in blocked]
        elif any(spec.arm in EXTENSIONS['Gaussian'] for spec in specs):
            proof = json.loads((out/'CONDITIONAL_CHECKS.json').read_text(encoding='utf-8'))
            gaussian = proof.get('modules', {}).get('Gaussian', {})
            if (proof.get('status') != 'PASS' or proof.get('stage') != stage or proof.get('study_id') != STUDY
                    or proof.get('queue_hash') != queue['queue_hash'] or gaussian.get('result') != 'PASS'
                    or not gaussian.get('source', {}).get('gaussian_source_hash')
                    or any(spec.source_hash != proof.get('source_hash') for spec in specs)):
                raise ValueError('Invalid Gaussian acceptance for the immutable queue')
        out.mkdir(parents=True,exist_ok=True)
        path = out/'RUN_RECEIPT.json'
        if path.is_file():
            previous_receipt = json.loads(path.read_text(encoding='utf-8'))
            if previous_receipt['queue_hash']!=queue['queue_hash']:
                raise ValueError('An existing stage receipt binds a different queue')
            if previous_receipt['owner_generation']!=self.store.generation and previous_receipt['jobs']:
                raise RuntimeError('Stage state is no longer held by its acknowledged RAM owner')
            if previous_receipt['status'] in ('COMPLETE','NOT_TRIGGERED'):
                return dict(previous_receipt,receipt_path=str(path))
        end = self.context.Event()
        reason = None
        started = time.time()
        for spec in specs:
            key = spec.job_key
            record = self.jobs.get(key)
            if record is not None and record['queue_hash']!=queue['queue_hash']:
                raise ValueError('Logical trajectory already belongs to a different immutable queue')
            if record is None:
                self.jobs[key] = dict(spec=spec,queue_hash=queue['queue_hash'],status='READY',epoch=0,
                                      attempt=0,anchors={},selected={},uuid=None,reason=None,launch_reason=None,
                                      failed_physical_batches=0, failed_epoch=None)
            elif record['status']=='TECHNICAL_PAUSED' and not record.get('owner_failure'):
                record['status'] = 'READY'
        status = 'NOT_TRIGGERED' if not specs and not queue.get('reuse_slots') else 'COMPLETE'
        if queue.get('reuse_slots'):
            raise RuntimeError('The queue references existing parent RAM with no verified foreground owner handoff')
        old = None
        if current_thread() is main_thread():
            previous = {}
            def request_stop(_sig,_frame):
                nonlocal old
                old = 'FOREGROUND_INTERRUPT'
            for signum in (signal.SIGINT,signal.SIGTERM):
                previous[signum] = signal.signal(signum,request_stop)
        else:
            previous = {}
        try:
            active = [self.jobs[s.job_key] for s in specs if self.jobs[s.job_key]['status']=='READY']
            if active and (controls:=scheduler.stop_status(self.artifacts)):
                old = controls[0]
            inputs = self._inputs([job['spec'] for job in active]) if active and old is None else None
            last_event = None
            reported_telemetry_errors = None
            while active or self.live:
                controls = scheduler.stop_status(self.artifacts)
                if old or controls:
                    reason = old or controls[0]
                    end.set()
                measurements = None
                try:
                    measurements = scheduler.sample()
                except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as error:
                    event = 'RESOURCE_TELEMETRY_UNAVAILABLE'
                    if last_event != event:
                        self._log(out,event,detail=str(error))
                    last_event = event
                if measurements and measurements['errors'] and measurements['errors']!=reported_telemetry_errors:
                    reported_telemetry_errors = measurements['errors']
                    self._log(out,'RESOURCE_TELEMETRY_UNAVAILABLE',detail=reported_telemetry_errors)
                resample = self._drain(out,measurements,end)
                active = [self.jobs[s.job_key] for s in specs if self.jobs[s.job_key]['status']=='READY']
                if resample and not end.is_set():
                    continue
                if end.is_set():
                    if not self.live:
                        break
                elif active and measurements:
                    devices = self._visible(measurements['devices'])
                    pending = {worker['uuid'] for worker in self.live.values() if not worker['allocated']}
                    launched = False
                    feasible = False
                    for delta in range(len(devices)):
                        index = (self.rotor+delta)%len(devices)
                        gpu = devices[index]
                        uuid = gpu['uuid']
                        for job in active:
                            spec = job['spec']
                            if any(self.jobs.get(dep,{}).get('status')!='COMPLETE' for dep in spec.dependencies):
                                continue
                            if job['uuid'] not in (None,uuid):
                                continue
                            usable = self._device_capacity(gpu) if self.after_oom else None
                            footprint = self.footprints.get(self._profile(spec))
                            if self.after_oom and (not usable or (footprint and footprint>usable)):
                                continue
                            feasible = True
                            if uuid in pending or not scheduler.eligible(gpu,measurements['host_ram_used']):
                                continue
                            admission_mode = 'CURRENT_ONLY'
                            if self.after_oom:
                                if footprint and not scheduler.eligible(gpu,measurements['host_ram_used'],physical_fit=True,
                                                                        cuda_usable=usable,full_process=footprint):
                                    continue
                                if not footprint and (not measurements['process_available'] or not scheduler.quiescent_capacity(
                                        gpu,measurements['processes'],usable)):
                                    continue
                                admission_mode = 'VERIFIED_FOOTPRINT' if footprint else 'QUIESCENT_WORKLOAD_MEASUREMENT'
                            if old or scheduler.stop_requested(self.artifacts):
                                end.set()
                                break
                            job['launch_reason'] = job['reason']
                            self._start(out,job,gpu,inputs,end,measurements,admission_mode)
                            self.rotor = (index+1)%len(devices)
                            launched = True
                            last_event = None
                            break
                        if launched or end.is_set():
                            break
                    if launched:
                        continue
                    if not feasible and not self.live:
                        reason = ('PHYSICAL_CAPACITY_UNVERIFIED_OR_INSUFFICIENT' if self.after_oom else
                                  'NO_VISIBLE_GPU_OR_DEPENDENCY_NOT_COMPLETE')
                        break
                    if last_event!='WAIT_RESOURCE':
                        self._log(out,'WAIT_RESOURCE',pending=len(active),running=len(self.live),physical_capacity_active=self.after_oom)
                        last_event = 'WAIT_RESOURCE'
                elif active and not self.live:
                    reason = 'RESOURCE_TELEMETRY_UNAVAILABLE'
                    break
                if self.live or active:
                    time.sleep(.35)
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            self._log(out,'SUPERVISION_ERROR',detail=reason)
        finally:
            end.set()
            while self.live:
                self._drain(out,None,end)
                if self.live:
                    time.sleep(.35)
            for sig,handler in previous.items():
                signal.signal(sig,handler)
        jobs = []
        for spec in specs:
            key = spec.job_key
            job = self.jobs[key]
            if job['status']=='READY':
                job.update(status='TECHNICAL_PAUSED',reason=reason or job['reason'] or 'DEPENDENCY_NOT_COMPLETE')
            jobs.append(dict(job_key=key,dataset=spec.dataset,seed=spec.seed,arm=spec.arm,status=job['status'],
                             attempt_id=job['attempt'],epoch=job['epoch'],reason=job['reason'],
                             failed_physical_batches=job['failed_physical_batches'],selected_hashes=job['selected']))
        if conditional_pause or reason=='USER_STOPPED' or any(row['status']!='COMPLETE' for row in jobs):
            status = 'USER_STOPPED' if reason=='USER_STOPPED' else 'TECHNICAL_PAUSED'
        receipt = dict(stage=stage,queue_hash=queue['queue_hash'],status=status,jobs=jobs,reason=reason,
                       owner_generation=self.store.generation,started_at=started,ended_at=time.time(),
                       artifacts={name:str(out/name) for name in ('runs.csv','curves.csv','endpoints.csv','steps.jsonl',
                                  'diagnostics.jsonl','costs.csv','state_receipts.jsonl','predictions.jsonl','admissions.jsonl')
                                  if (out/name).is_file()})
        if conditional_pause:
            receipt['conditional_pause'] = conditional_pause
        _json(path,receipt)
        return dict(receipt,receipt_path=str(path))

    def selected_get(self, job_key, readout):
        if self.closed or self.jobs[job_key]['status']!='COMPLETE':
            raise RuntimeError('Selected readback requires a live owner and a completed trajectory')
        job = self.jobs[job_key]
        packet, receipt_hash = self.store.get(f'{job_key}:selected:{readout}')
        identity = packet['identity']
        model_hash = state_digest(packet['model_state'])
        if job['selected'][readout]!=model_hash or identity['readout']!=readout:
            raise RuntimeError('Actual selected readback differs from its completed trajectory identity')
        _append(self.artifacts/job['spec'].stage/'state_receipts.jsonl',dict(run_id=job_key,state_kind=SELECTED,
            guardian_key=f'{job_key}:selected:{readout}',schema_version=1,epoch=packet['epoch'],step=packet['global_step'],
            put_hash=receipt_hash,get_hash=receipt_hash,owner=self.store.generation,available=True,verified_at=time.time()))
        return dict(model_state=packet['model_state'],identity=identity,model_hash=model_hash,ram_hash=receipt_hash)

    def release_selected(self, retain):
        retain = set(retain)
        for key,job in self.jobs.items():
            if key not in retain:
                for readout in list(job['selected']):
                    name = f'{key}:selected:{readout}'
                    self.store.release(name)
                    _append(self.artifacts/job['spec'].stage/'state_receipts.jsonl',dict(run_id=key,state_kind=SELECTED,
                        guardian_key=name,available=False,release_reason='OUTSIDE_FROZEN_SET',released_at=time.time()))
                    del job['selected'][readout]

    def close(self):
        if self.live:
            raise RuntimeError('Close the foreground owner only after active atomic work has been harvested')
        self.closed = True
        self.store.close()
        if self.cache is not None:
            self.cache.cleanup()
