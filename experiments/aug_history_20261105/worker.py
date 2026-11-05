"""Source online training, sixteen complete validations, and RAM epoch snapshots."""
import copy
import hashlib
import resource
import time
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.overnight_ab_20261103 import evaluation, selection, state
from .adam_counterfactual import snapshot_optimizer
from .model_adapter import build_model
from .objectives import loss_terms
from .registry import ParameterLayout
from .responses import extract_aug_grads, response_basis
from .shadow_bank import ShadowBank


def create_session(job, bundle, device):
    history.setup(job['seed'], device)
    model = build_model(bundle, job['config']).to(device)
    layout = ParameterLayout.from_model(model)
    scratch = copy.deepcopy(model).requires_grad_(False)
    c = job['config']
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'], betas=tuple(c['betas']),
        eps=c['eps'], weight_decay=c['weight_decay'], foreach=False, fused=False, amsgrad=False)
    ids = ['online']+[row['id'] for row in job['candidates']]
    if job.get('route','B')=='A':
        from .group_registry import build_path_groups
        from .grouped_ema import GroupedEMABank
        registry=build_path_groups(model,bundle)
        if job.get('group_hash',registry.hash)!=registry.hash:
            raise ValueError('A job group identity changed')
        bank=GroupedEMABank(layout.flatten(model.parameters()),job['candidates'],c['rho'],registry,bundle,job['seed'])
    else:
        bank=ShadowBank(layout.flatten(model.parameters()),job['candidates'],c['rho'],job['dataset'],job['seed'])
    return dict(job=copy.deepcopy(job), bundle=bundle, device=str(device), model=model,
        scratch=scratch, optimizer=optimizer, layout=layout,
        bank=bank,
        loader=torch.Generator().manual_seed(job['seed']+11000), rng=history.rng_state(device),
        epoch=0, global_step=0, commit_index=0, batch_position=0,
        selection={kind:dict(final_stage='train', best_score=-float('inf'), selected=None) for kind in ids},
        snapshots={kind:{} for kind in ids}, cost={}, candidate_cost={kind:0. for kind in ids},
        init_hash=state.state_digest(model.state_dict()), complete=False)


def train_batch(session, x, y, row_ids=None):
    model, optimizer, bank = session['model'], session['optimizer'], session['bank']
    config, seed = session['job']['config'], session['job']['seed']
    step = session['global_step']+1
    if session['job'].get('route','B')=='A':
        if row_ids is None:
            raise ValueError('A probes require current raw batch row IDs')
        optimizer.zero_grad(set_to_none=True)
        terms=loss_terms(model,x,y,dict(config,seed=seed,step=step))
        if not bool(torch.isfinite(terms.total)):
            raise FloatingPointError('Nonfinite training loss')
        terms.total.backward(); optimizer.step()
        response=bank.after_step(dict(step=step,theta=session['layout'].flatten(model.parameters()),
            row_ids=row_ids,model=model))
        session['global_step']=step
        return dict(loss=float(terms.total.detach()),clean_loss=float(terms.clean.detach()),
            extra_views=terms.metadata['extra_views'],probe_seconds=response['controller_seconds']+response['sampling_seconds'],
            response=response,views=[{k:v[k] for k in ('key','seed','hash')} for v in terms.metadata['views']])
    optimizer.zero_grad(set_to_none=True)
    refresh = bank.refresh_due(step)
    terms = loss_terms(model,x,y,dict(config,seed=seed,step=step))
    if not bool(torch.isfinite(terms.total)):
        raise FloatingPointError('Nonfinite training loss')
    active_aug = bool(terms.metadata['augmented'])
    response, probe_seconds = None, 0.
    if refresh and active_aug:
        start = time.perf_counter()
        snapshot = snapshot_optimizer(model,optimizer)
        auxiliary = extract_aug_grads(terms,tuple(model.parameters()))
        probe_seconds += time.perf_counter()-start
    terms.total.backward()
    if refresh:
        if active_aug:
            start = time.perf_counter()
            response = response_basis(snapshot,[p.grad for p in model.parameters()],auxiliary)
            probe_seconds += time.perf_counter()-start
        else:
            empty = bank.phi['F'].new_empty((session['layout'].size,0))
            response = dict(D=empty, M=empty, metadata=dict(empty_target=True))
    optimizer.step()
    theta = session['layout'].flatten(model.parameters())
    if refresh and active_aug:
        actual = theta-session['layout'].flatten(snapshot['theta'])
        if not torch.equal(actual,response['full_displacement']):
            raise RuntimeError('Counterfactual full-gradient Adam differs from real update')
    bank.after_step(dict(step=step,theta=theta,response=response))
    session['global_step'] = step
    return dict(loss=float(terms.total.detach()), clean_loss=float(terms.clean.detach()),
        extra_views=terms.metadata['extra_views'], probe_seconds=probe_seconds,
        response=response['metadata'] if response else None,
        views=[{k:v[k] for k in ('key','seed','hash')} for v in terms.metadata['views']])


def advance_epoch(session):
    if session['complete']:
        raise ValueError('The full fifty-epoch budget is complete')
    job, bundle, model = session['job'], session['bundle'], session['model']
    c, device, epoch = job['config'], session['device'], session['epoch']+1
    history.restore_rng(session['rng'],device)
    model.train()
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    order = torch.randperm(len(bundle.train_y),generator=session['loader']).numpy()
    order_hash = hashlib.sha256(np.asarray(bundle.train_row_ids[order]).tobytes()).hexdigest()
    start, loss, clean, probes, extra = time.perf_counter(), 0., 0., 0., 0
    sampling_seconds=0.
    trace = []
    for batch, offset in enumerate(range(0,len(order),c['batch_size']),1):
        ids = order[offset:offset+c['batch_size']]
        x = torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T
        y = torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
        value = train_batch(session,x,y,row_ids=np.asarray(bundle.train_row_ids[ids]))
        loss += value['loss']*len(ids); clean += value['clean_loss']*len(ids)
        probes += value['probe_seconds']; extra += value['extra_views']
        if job.get('route','B')=='A': sampling_seconds+=value['response']['sampling_seconds']
        trace.append(dict(batch=batch,global_step=session['global_step'],rows=len(ids),
            row_ids_hash=hashlib.sha256(np.asarray(bundle.train_row_ids[ids]).tobytes()).hexdigest(),**value))
    if cuda:
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-start
    train_rng = history.rng_state(device)
    validation, hashes = {}, {}
    buffers = {k:v.detach() for k,v in model.named_buffers()}
    for kind in session['snapshots']:
        estimator = model
        if kind != 'online':
            estimator = session['scratch']
            session['layout'].copy_to(estimator,session['bank'].phi[kind],buffers)
        eval_start = time.perf_counter()
        values = evaluation.evaluate_model(estimator,bundle,'validation','G-D0',c['eval_batch_size'])
        elapsed = time.perf_counter()-eval_start
        session['candidate_cost'][kind] += elapsed
        if job.get('route','B')=='A' and kind!='online':
            session['bank'].states[kind].cost['validation_seconds']+=elapsed
        hashes[kind] = state.state_digest(estimator.state_dict())
        validation[kind] = values
        tracker = session['selection'][kind]
        selection.consider(tracker,'train',epoch,values,estimator.state_dict(),{'candidate_id':kind})
        session['snapshots'][kind]['selected'] = tracker['selected']
        if epoch in (10,50):
            session['snapshots'][kind][f'E{epoch}'] = dict(model_state=state.cpu_copy(estimator.state_dict()),
                stage='train',epoch=epoch,model_hash=hashes[kind],validation=copy.deepcopy(values),model_meta={'candidate_id':kind})
    history.restore_rng(train_rng,device)
    total_seconds = time.perf_counter()-start
    cost = dict(training_seconds=train_seconds-probes, probe_seconds=probes,
        validation_seconds=total_seconds-train_seconds, gpu_seconds=total_seconds if cuda else 0.,
        wall_seconds=total_seconds, original_rows=len(order), extra_views=extra,
        real_adam_steps=len(trace), validations=len(validation),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
        peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if job.get('route','B')=='A': cost['sampling_seconds']=sampling_seconds
    for key,value in cost.items():
        session['cost'][key] = max(session['cost'].get(key,0),value) if key.startswith('peak_') else session['cost'].get(key,0)+value
    session.update(epoch=epoch,commit_index=epoch,batch_position=0,rng=train_rng,complete=epoch==50)
    if session['complete'] and session['global_step'] != c['steps_per_epoch']*50:
        raise ValueError('Population/step count differs from frozen protocol')
    result=dict(dataset=job['dataset'],seed=job['seed'],job_id=job['job_id'],epoch=epoch,
        commit_index=epoch,global_step=session['global_step'],order_hash=order_hash,
        init_hash=session['init_hash'],layout_hash=session['layout'].hash,
        validation=validation,model_hash=hashes,cost=cost,
        candidate_cost=session['candidate_cost'].copy(),
        training_loss=loss/len(order),clean_loss=clean/len(order),batch_trace=trace,
        complete=session['complete'])
    if job.get('route','B')=='A':
        result.update(group_hash=session['bank'].registry.hash,
            estimator_costs={cid:state.cost.copy() for cid,state in session['bank'].states.items()})
    return result


def export_session(session):
    fields = ('job','epoch','global_step','commit_index','batch_position','rng','snapshots',
              'cost','candidate_cost','init_hash','complete')
    snapshot = {k:session[k] for k in fields}
    snapshot.update(schema_version=1, route=session['job'].get('route','B'),layout_hash=session['layout'].hash,
        online=session['model'].state_dict(),optimizer=session['optimizer'].state_dict(),
        bank=session['bank'].state_dict(),loader=session['loader'].get_state(),
        best_scores={k:v['best_score'] for k,v in session['selection'].items()})
    return state.cpu_copy(snapshot)


def restore_session(job,bundle,device,snapshot):
    if snapshot['schema_version'] != 1 or snapshot['route'] != job.get('route','B') or snapshot['job'] != job:
        raise ValueError('RAM snapshot identity mismatch')
    session = create_session(job,bundle,device)
    if session['layout'].hash != snapshot['layout_hash']:
        raise ValueError('RAM snapshot parameter layout changed')
    session['model'].load_state_dict(snapshot['online'])
    session['optimizer'].load_state_dict(snapshot['optimizer'])
    session['bank'].load_state_dict(snapshot['bank'],device)
    session['loader'].set_state(snapshot['loader'])
    for key in ('epoch','global_step','commit_index','batch_position','rng','snapshots',
                'cost','candidate_cost','init_hash','complete'):
        session[key] = state.cpu_copy(snapshot[key])
    for key,tracker in session['selection'].items():
        tracker.update(best_score=snapshot['best_scores'][key],selected=session['snapshots'][key].get('selected'))
    history.restore_rng(snapshot['rng'],device)
    return session
