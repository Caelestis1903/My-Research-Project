"""A complete original-batch epoch, exported only through RAM IPC."""
import copy
import hashlib
import resource
import time
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import make_model
from experiments.overnight_ab_20261103 import evaluation, selection, state
from .objectives import train_step


def create_session(job, bundle, device):
    history.setup(job['seed'], device)
    source_conf = history.load_protocol()['datasets']['kuairand']['source']
    model = make_model(bundle, source_conf).to(device)
    if sum(p.numel() for p in model.parameters()) != 1614254:
        raise ValueError('Fixed1 KuaiRand model identity changed')
    ema = copy.deepcopy(model).requires_grad_(False)
    c = job['config']
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'], betas=tuple(c['betas']),
        eps=c['eps'], weight_decay=c['weight_decay'], foreach=False, fused=False, amsgrad=False)
    return dict(job=copy.deepcopy(job), bundle=bundle, device=str(device), stage='train',
        next_stage_epoch=1, completed_search=0, completed_final=0, commit_index=0, global_step=0,
        model=model, ema=ema, optimizers={'main': optimizer},
        loader=torch.Generator().manual_seed(job['seed']+11000), rng=state.capture_rng(device),
        selection={kind:dict(final_stage='train', best_score=-float('inf'), selected=None)
                   for kind in ('online', 'fixed_ema')},
        snapshots={kind:{} for kind in ('online', 'fixed_ema')}, cost={}, complete=False,
        init_hash=state.state_digest(model.state_dict()))


def advance_epoch(session):
    if session['complete']:
        raise ValueError('The fixed 50-epoch budget is complete')
    job, bundle, model = session['job'], session['bundle'], session['model']
    c, device, epoch = job['config'], session['device'], session['next_stage_epoch']
    state.restore_rng(session['rng'], device)
    model.train()
    order = torch.randperm(len(bundle.train_y), generator=session['loader']).numpy()
    order_hash = hashlib.sha256(np.asarray(bundle.train_row_ids[order]).tobytes()).hexdigest()
    cost, trace = {}, []
    loss_sum = real_sum = 0.
    start = time.perf_counter()
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for batch, offset in enumerate(range(0, len(order), c['batch_size']), 1):
        ids = order[offset:offset+c['batch_size']]
        session['global_step'] += 1
        x = torch.as_tensor(np.array(bundle.train_x[ids]), device=device).T
        y = torch.as_tensor(np.array(bundle.train_y[ids]), device=device)
        row_ids = np.asarray(bundle.train_row_ids[ids])
        value = train_step(model, session['optimizers']['main'], session['ema'], x, y,
                           row_ids, c, job['seed'], session['global_step'])
        for view in value['views']:
            mask = np.asarray(bundle.train_x[ids, -1]) == view['domain']
            view['row_ids_hash'] = hashlib.sha256(row_ids[mask].tobytes()).hexdigest()
        loss_sum += value['loss']*len(ids)
        real_sum += value['real_loss']*len(ids)
        for name in ('extra_views', 'forwards', 'optimizer_steps', 'ema_updates', 'backward_calls'):
            cost[name] = cost.get(name, 0)+value[name]
        for view in value['views']:
            name = f"d{view['domain']}_extra_views"
            cost[name] = cost.get(name, 0)+view['rows']
        trace.append(dict(epoch=epoch, batch=batch, global_step=session['global_step'], rows=len(ids),
            original_row_hash=hashlib.sha256(row_ids.tobytes()).hexdigest(), **value))
    if cuda:
        torch.cuda.synchronize(device)
    cost['training_seconds'] = time.perf_counter()-start
    validation, hashes = {}, {}
    eval_start = time.perf_counter()
    for kind, estimator in (('online', model), ('fixed_ema', session['ema'])):
        values = evaluation.evaluate_model(estimator, bundle, 'validation', job['arm'], c['eval_batch_size'])
        validation[kind] = values
        hashes[kind] = state.state_digest(estimator.state_dict())
        tracker = session['selection'][kind]
        selection.consider(tracker, 'train', epoch, values, estimator.state_dict(), {'method':job['arm']})
        session['snapshots'][kind]['selected'] = tracker['selected']
        if epoch in (10, c['epochs']):
            model_state = state.cpu_copy(estimator.state_dict())
            session['snapshots'][kind][f'E{epoch}'] = dict(model_state=model_state, stage='train',
                epoch=epoch, model_hash=state.state_digest(model_state), validation=copy.deepcopy(values),
                model_meta={'method':job['arm']})
    cost.update(validation_seconds=time.perf_counter()-eval_start,
        validation_rows=2*len(bundle.val_y), original_rows=len(order), label_exposures=2*len(order),
        worker_seconds=time.perf_counter()-start,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for name, value in cost.items():
        if name.startswith('peak_') or name == 'rss_peak_bytes':
            session['cost'][name] = max(session['cost'].get(name, 0), value)
        else:
            session['cost'][name] = session['cost'].get(name, 0)+value
    session.update(completed_final=epoch, next_stage_epoch=epoch+1,
                   commit_index=session['commit_index']+1, complete=epoch==c['epochs'])
    session['rng'] = state.capture_rng(device)
    if session['complete']:
        if session['global_step'] != 2450 or session['cost']['original_rows'] != 100306600:
            raise ValueError('The original population or training budget changed')
    return dict(job_id=job['job_id'], stage='train', epoch=epoch, commit_index=session['commit_index'],
        completed_search=0, completed_final=epoch, global_step=session['global_step'],
        validation=validation, model_hash=hashes,
        train_objective=dict(loss=loss_sum/len(order), real_loss=real_sum/len(order)),
        cost=cost, complete=session['complete'], order_hash=order_hash,
        init_hash=session['init_hash'], batch_trace=trace)


def export_session(session):
    fields = ('job', 'stage', 'next_stage_epoch', 'completed_search', 'completed_final', 'commit_index',
              'global_step', 'selection', 'snapshots', 'cost', 'complete', 'rng', 'init_hash')
    snapshot = {key:session[key] for key in fields}
    snapshot.update(schema_version=1, family='GAUSSIAN_COVERAGE', model=session['model'].state_dict(),
                    ema=session['ema'].state_dict(), optimizers={'main':session['optimizers']['main'].state_dict()},
                    loader_state=session['loader'].get_state())
    return state.cpu_copy(snapshot)


def restore_session(job, bundle, device, snapshot):
    if snapshot['schema_version'] != 1 or snapshot['family'] != 'GAUSSIAN_COVERAGE' or snapshot['job'] != job:
        raise ValueError('RAM snapshot does not match this job')
    session = create_session(job, bundle, device)
    session['model'].load_state_dict(snapshot['model'])
    session['ema'].load_state_dict(snapshot['ema'])
    session['optimizers']['main'].load_state_dict(snapshot['optimizers']['main'])
    session['loader'].set_state(snapshot['loader_state'])
    for key in ('stage', 'next_stage_epoch', 'completed_search', 'completed_final', 'commit_index',
                'global_step', 'selection', 'snapshots', 'cost', 'complete', 'rng', 'init_hash'):
        session[key] = state.cpu_copy(snapshot[key])
    state.restore_rng(snapshot['rng'], device)
    return session
