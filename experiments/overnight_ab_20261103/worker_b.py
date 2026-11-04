"""One complete frozen M3/combined epoch with RAM-only session export."""
import copy
import hashlib
import resource
import time
import torch
from experiments.mdmt_history_information_20261103 import protocol as source
from . import evaluation, selection, state
from .objectives import make_optimizer, train_step


def create_session(job, bundle, device):
    if torch.device(device).type == 'cuda': torch.cuda.set_device(device)
    source.setup(job['seed'], device)
    model = source.make_model(bundle, source.load_protocol()['datasets'][job['dataset']]['source']).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = make_optimizer(model, job['config'])
    manifest = [dict(item, numel=model.get_parameter(item['name']).numel())
                for item in optimizer.parameter_manifest]
    return dict(job=copy.deepcopy(job), bundle=bundle, device=str(device), stage='train', next_stage_epoch=1,
                completed_search=0, completed_final=0, commit_index=0, global_step=0,
                model=model, ema=ema, optimizers={'main': optimizer},
                loader=torch.Generator().manual_seed(job['seed'] + 11000), rng=state.capture_rng(device),
                selection={kind: dict(final_stage='train', best_score=-float('inf'), selected=None)
                           for kind in ('online', 'fixed_ema')},
                snapshots={kind: {} for kind in ('online', 'fixed_ema')}, cost={}, complete=False,
                parameter_manifest=manifest, parameter_manifest_hash=state.state_digest(manifest))


def _snapshot(model, epoch, validation, method):
    model_state = state.cpu_copy(model.state_dict())
    return dict(model_state=model_state, stage='train', epoch=epoch, model_hash=state.state_digest(model_state),
                validation=copy.deepcopy(validation), model_meta={'method': method})


def advance_epoch(session):
    if session['complete']:
        raise ValueError('Training budget is already complete')
    job = session['job']
    c = job['config']
    bundle = session['bundle']
    model = session['model']
    optimizer = session['optimizers']['main']
    device = session['device']
    epoch = session['next_stage_epoch']
    state.restore_rng(session['rng'],session['device'])
    model.train()
    order = torch.randperm(len(bundle.train_y), generator=session['loader']).numpy()
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    cost, trace = {}, []
    loss_sum = real_sum = 0.
    start = time.perf_counter()
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for batch, offset in enumerate(range(0, len(order), c['batch_size']), 1):
        ids = order[offset:offset + c['batch_size']]
        session['global_step'] += 1
        step = session['global_step']
        x = torch.as_tensor(bundle.train_x[ids], device=device).T
        y = torch.as_tensor(bundle.train_y[ids], device=device)
        if cuda:
            left, right = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            left.record()
        value = train_step(model, optimizer, session['ema'], x, y, bundle.train_row_ids[ids], c,
                           dict(study='EMA02V2', split_id='original', dataset=job['dataset'],
                                seed=job['seed'], epoch=epoch, global_step=step),
                           diagnostic=step == 1 or step % 20 == 0)
        if cuda:
            right.record()
            right.synchronize()
            cost['train_gpu_event_ms'] = cost.get('train_gpu_event_ms', 0.) + left.elapsed_time(right)
        loss_sum += value['loss'] * len(ids)
        real_sum += value['real_loss'] * len(ids)
        for key, number in value['counts'].items():
            cost[key] = cost.get(key, 0) + number
        trace.append(dict(epoch=epoch, batch=batch, step=step, rows=len(ids), loss=value['loss'],
                          real_loss=value['real_loss'], counts=value['counts'], diagnostics=value['diagnostics']))
    cost['training_seconds'] = time.perf_counter() - start
    validation, hashes = {}, {}
    eval_start = time.perf_counter()
    for kind, estimator in (('online', model), ('fixed_ema', session['ema'])):
        values = evaluation.evaluate_model(estimator, bundle, 'validation', job['arm'], c['eval_batch_size'])
        validation[kind] = values
        model_state = estimator.state_dict()
        hashes[kind] = state.state_digest(model_state)
        tracker = session['selection'][kind]
        selection.consider(tracker, 'train', epoch, values, model_state, {'method': job['arm']})
        session['snapshots'][kind]['selected'] = tracker['selected']
        if epoch in (10, c['epochs']):
            session['snapshots'][kind][f'E{epoch}'] = _snapshot(estimator, epoch, values, job['arm'])
    cost.update(validation_seconds=time.perf_counter() - eval_start,
                validation_rows=2 * len(bundle.val_y),
                validation_forwards=2 * ((len(bundle.val_y) + c['eval_batch_size'] - 1) // c['eval_batch_size']),
                worker_seconds=time.perf_counter() - start,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
                rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                label_exposures=2 * len(order))
    for key, value in cost.items():
        if key.startswith('peak_') or key == 'rss_peak_bytes':
            session['cost'][key] = max(session['cost'].get(key, 0), value)
        else:
            session['cost'][key] = session['cost'].get(key, 0) + value
    session['completed_final'] = epoch
    session['next_stage_epoch'] = epoch + 1
    session['commit_index'] += 1
    session['complete'] = epoch == c['epochs']
    session['rng'] = state.capture_rng(session['device'])
    selected = {kind: {'epoch': tracker['selected']['epoch'], 'model_hash': tracker['selected']['model_hash']}
                for kind, tracker in session['selection'].items()}
    return dict(job_id=job['job_id'], stage='train', epoch=epoch, commit_index=session['commit_index'],
                completed_search=0, completed_final=epoch, global_step=session['global_step'],
                validation=validation, model_hash=hashes, selection_hash=state.state_digest(selected),
                train_objective={'loss': loss_sum / len(order), 'real_loss': real_sum / len(order)},
                cost=cost, complete=session['complete'], order_hash=order_hash, batch_trace=trace,
                parameter_manifest=session['parameter_manifest'], parameter_manifest_hash=session['parameter_manifest_hash'])


def export_session(session):
    fields = ('job', 'stage', 'next_stage_epoch', 'completed_search', 'completed_final', 'commit_index',
              'global_step', 'selection', 'snapshots', 'cost', 'complete', 'rng',
              'parameter_manifest', 'parameter_manifest_hash')
    snapshot = {key: session[key] for key in fields}
    snapshot.update(schema_version=1, family='B', model=session['model'].state_dict(),
                    ema=session['ema'].state_dict(), optimizers={'main': session['optimizers']['main'].state_dict()},
                    loader_state=session['loader'].get_state())
    return state.cpu_copy(snapshot)


def restore_session(job, bundle, device, snapshot):
    if snapshot['schema_version'] != 1 or snapshot['family'] != 'B' or snapshot['job'] != job:
        raise ValueError('RAM snapshot does not match this frozen job')
    session = create_session(job, bundle, device)
    session['model'].load_state_dict(snapshot['model'])
    session['ema'].load_state_dict(snapshot['ema'])
    session['optimizers']['main'].load_state_dict(snapshot['optimizers']['main'])
    session['loader'].set_state(snapshot['loader_state'])
    for key in ('stage', 'next_stage_epoch', 'completed_search', 'completed_final', 'commit_index', 'global_step',
                'selection', 'snapshots', 'cost', 'complete', 'rng'):
        session[key] = state.cpu_copy(snapshot[key])
    state.restore_rng(snapshot['rng'],session['device'])
    return session
