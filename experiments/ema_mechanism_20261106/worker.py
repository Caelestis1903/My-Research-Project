"""Frozen online trajectory with non-feedback observations and RAM export."""
import copy
import hashlib
import resource
import time
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses as source_losses
from experiments.aug_history_20261105.model_adapter import build_model
from experiments.aug_history_20261105.registry import ParameterLayout
from experiments.aug_history_20261105.group_registry import build_path_groups,GROUPS
from experiments.overnight_ab_20261103.state import cpu_copy,state_digest
from experiments.overnight_ab_20261103.selection import score as selection_score
from .shadows import FixedShadowBank,PulseWindow
from .readouts import panel_predictions,evaluate_full_validation,pulse_rows,group_hybrid
from .observers import observe_innovation,observe_pulse,rank_direction_scores


def create_session(job,bundle,device,panels):
    history.setup(job['seed'],device);c=job['config']
    model=build_model(bundle,c).to(device);layout=ParameterLayout.from_model(model)
    scratch=copy.deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=tuple(c['betas']),eps=c['eps'],
        weight_decay=c['weight_decay'],foreach=False,fused=False,amsgrad=False)
    bank=FixedShadowBank(layout.flatten(model.parameters()),c['rho_by_physical_shadow'],c['aliases'])
    return dict(job=copy.deepcopy(job),bundle=bundle,device=str(device),panels=panels,
        model=model,scratch=scratch,layout=layout,groups=build_path_groups(model,bundle),optimizer=optimizer,
        bank=bank,pulse=PulseWindow(c['rho']),loader=torch.Generator().manual_seed(job['seed']+11000),
        rng=history.rng_state(device),epoch=0,global_step=0,commit_index=0,complete=False,
        init_hash=state_digest(model.state_dict()),snapshots={key:{} for key in ('online',*bank.phi)},
        best_scores={key:-float('inf') for key in ('online',*bank.phi)},retained_pulse_states={},
        innovation=[],cost={})


def train_batch(session,x,y,row_ids=None,observe=True):
    job=session['job'];c=job['config'];step=session['global_step']+1
    optimizer=session['optimizer'];optimizer.zero_grad(set_to_none=True)
    loss,info=source_losses(session['model'],x,y,0,
        dict(num_views=int(job['objective']=='gaussian_d0'),view_family='gaussian',pairing='iid'),
        job['dataset'],job['seed'],step,mix_lambda=.5,kappa=c['kappa'])
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite source training loss')
    loss.backward();optimizer.step();theta=session['layout'].flatten(session['model'].parameters())
    if observe: session['innovation'].append(observe_innovation(theta,session['bank'].get('F'),session['groups']))
    session['bank'].after_adam(theta,step);session['pulse'].after_adam(theta,step)
    session['global_step']=step
    return dict(loss=float(loss.detach()),clean_loss=float(info['real'].mean(1).mean().detach()),
        extra_views=info['extra_views'],rows=len(y))


def innovation_summary(rows):
    result=[]
    for b,name in enumerate(GROUPS):
        row=dict(group_id=name,steps=len(rows),actual_count=rows[-1][b]['actual_count'])
        for metric in ('l2_squared','rms','energy_share'):
            values=[r[b][metric] for r in rows if r[b][metric] is not None]
            row[metric+'_mean']=float(np.mean(values)) if values else None
            row[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            row[metric+'_max']=float(np.max(values)) if values else None
            row[metric+'_last']=rows[-1][b][metric];row[metric+'_count']=len(values)
        result.append(row)
    return result


def collect_observations(session,epoch):
    model,scratch,layout,bank=session['model'],session['scratch'],session['layout'],session['bank']
    bundle,panels=session['bundle'],session['panels'];c=session['job']['config']
    buffers={k:v.detach() for k,v in model.named_buffers()};validation={};hashes={};payload=[];by_panel={p:{} for p in panels}
    start=time.perf_counter()
    for kind in session['snapshots']:
        estimator=model
        if kind!='online': layout.copy_to(scratch,bank.get(kind),buffers);estimator=scratch
        is_f=kind==bank.aliases.get('F','F');values,pred=evaluate_full_validation(estimator,bundle,panels['V_mech'],c['eval_batch_size'],include_logits=kind=='online' or is_f)
        validation[kind]=values;hashes[kind]=state_digest(estimator.state_dict())
        slot=session['snapshots'][kind]
        snapshot=None
        if selection_score(values)>session['best_scores'][kind] or epoch in (10,50):
            snapshot=dict(model_state=cpu_copy(estimator.state_dict()),epoch=epoch,model_hash=hashes[kind],validation=copy.deepcopy(values),stage='train')
        if selection_score(values)>session['best_scores'][kind]:
            session['best_scores'][kind]=selection_score(values);slot['selected']=snapshot
        if epoch in (10,50): slot[f'E{epoch}']=snapshot
        payload.append(dict(readout_id=kind,physical_readout_id=kind,panel_id='V_mech',state_hash=hashes[kind],arrays=pred))
        if kind=='online' or is_f:
            canonical='F' if is_f else 'online';by_panel['V_mech'][canonical]=pred
            for name in ('P_A','P_B'):
                pp=panel_predictions(estimator,panels[name]);by_panel[name][canonical]=pp
                payload.append(dict(readout_id=kind,physical_readout_id=kind,panel_id=name,state_hash=hashes[kind],arrays=pp))
    validation_seconds=time.perf_counter()-start;pulse_readouts=[];pulse_metrics=[];pulse_seconds=0.;gradient_seconds=0.
    if session['pulse'].active and session['pulse'].end_epoch==epoch:
        start=time.perf_counter();pulse=session['pulse'].finish(session['global_step'])
        base=bank.get('F');theta=layout.flatten(model.parameters());hybrid=base.new_empty(base.shape)
        fkind=bank.aliases.get('F','F');fhash=hashes[fkind]
        for name,panel in panels.items():
            pulse_readouts.extend(pulse_rows(panel,by_panel[name]['F'],'F',None,'F',fhash))
        descriptors=[(f'{speed}_{g}',g,speed,vector,b) for speed,vector in (('fast',pulse['fast']),('slow',pulse['slow'])) for b,g in enumerate(GROUPS)]
        descriptors.extend([('all_fast',None,'all_fast',pulse['fast'],None),('all_slow',None,'all_slow',pulse['slow'],None)])
        for readout,group,variant,vector,b in descriptors:
            if b is None: hybrid.copy_(vector)
            else: group_hybrid(base,vector,session['groups'].indices(base.device),b,hybrid)
            layout.copy_to(scratch,hybrid,buffers);state_hash=state_digest(scratch.state_dict())
            for name,panel in panels.items():
                pred=panel_predictions(scratch,panel)
                if name=='V_mech': pred.pop('z',None)
                by_panel[name][readout]=pred
                payload.append(dict(readout_id=readout,physical_readout_id=readout,panel_id=name,state_hash=state_hash,arrays=pred))
                pulse_readouts.extend(pulse_rows(panel,pred,readout,group,variant,state_hash))
        pulse_seconds=time.perf_counter()-start;start=time.perf_counter()
        m2=rank_direction_scores(scratch,base,pulse['fast'],pulse['slow'],panels['P_A'],layout,session['groups'])
        gradient_seconds=time.perf_counter()-start
        pulse_metrics=observe_pulse(theta,base,pulse['fast'],pulse['slow'],session['groups'],by_panel,panels,pulse_readouts,m2)
        if epoch in (20,40):
            session['retained_pulse_states'][f'E{epoch}']=cpu_copy(dict(F=base,fast=pulse['fast'],slow=pulse['slow'],buffers=buffers))
        session['pulse'].fast=None;session['pulse'].slow=None
    return dict(validation=validation,model_hash=hashes,pulse_readouts=pulse_readouts,pulse_metrics=pulse_metrics),payload,dict(validation_seconds=validation_seconds,pulse_seconds=pulse_seconds,gradient_seconds=gradient_seconds)


def advance_epoch(session):
    if session['complete']: raise ValueError('Fifty-epoch budget already complete')
    job,bundle,device=session['job'],session['bundle'],session['device'];c=job['config'];epoch=session['epoch']+1
    history.restore_rng(session['rng'],device);session['model'].train();session['innovation']=[]
    cuda=str(device).startswith('cuda')
    if cuda: torch.cuda.reset_peak_memory_stats(device)
    order=torch.randperm(len(bundle.train_y),generator=session['loader']).numpy()
    order_hash=hashlib.sha256(np.asarray(bundle.train_row_ids[order]).tobytes()).hexdigest()
    start=time.perf_counter();loss=clean=0.;extra=0
    for offset in range(0,len(order),c['batch_size']):
        ids=order[offset:offset+c['batch_size']]
        x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T;y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
        value=train_batch(session,x,y,np.asarray(bundle.train_row_ids[ids]));loss+=value['loss']*len(ids);clean+=value['clean_loss']*len(ids);extra+=value['extra_views']
    if cuda: torch.cuda.synchronize(device)
    train_seconds=time.perf_counter()-start;train_rng=history.rng_state(device)
    observed,payload,cost=collect_observations(session,epoch)
    if (epoch+1)%5==0 and epoch<50:
        session['pulse'].begin(session['bank'].get('F'),epoch+1,session['global_step'])
    history.restore_rng(train_rng,device)
    cost.update(training_seconds=train_seconds,probe_seconds=cost['pulse_seconds']+cost['gradient_seconds'],
        original_rows=len(order),extra_views=extra,real_adam_steps=c['steps_per_epoch'],
        validations=len(observed['validation']),pulse_state_readouts=26 if epoch%5==0 else 0,
        gradient_probes=12 if epoch%5==0 else 0,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
        peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        wall_seconds=time.perf_counter()-start)
    for key,value in cost.items(): session['cost'][key]=max(session['cost'].get(key,0),value) if key.startswith('peak_') else session['cost'].get(key,0)+value
    session.update(epoch=epoch,commit_index=epoch,rng=train_rng,complete=epoch==50)
    record=dict(schema_version=1,job_id=job['job_id'],dataset=job['dataset'],objective=job['objective'],seed=job['seed'],
        online_trajectory_id=job['online_trajectory_id'],identity_id=job['job_id'],epoch=epoch,global_step=session['global_step'],
        commit_index=epoch,init_hash=session['init_hash'],order_hash=order_hash,layout_hash=session['layout'].hash,
        group_hash=session['groups'].hash,training_loss=loss/len(order),clean_loss=clean/len(order),
        innovation_stats=innovation_summary(session['innovation']),cost=cost,complete=session['complete'],predictions=[],**observed)
    session['innovation']=[]
    return record,payload


def export_session(session):
    result={k:session[k] for k in ('job','epoch','global_step','commit_index','complete','rng','snapshots','best_scores','retained_pulse_states','cost','init_hash')}
    result.update(schema_version=1,layout_hash=session['layout'].hash,group_hash=session['groups'].hash,
        online=session['model'].state_dict(),optimizer=session['optimizer'].state_dict(),bank=session['bank'].state_dict(),
        pulse=session['pulse'].state_dict(),loader=session['loader'].get_state())
    return cpu_copy(result)


def restore_session(job,bundle,device,panels,snapshot):
    if snapshot['schema_version']!=1 or snapshot['job']!=job: raise ValueError('RAM snapshot identity changed')
    session=create_session(job,bundle,device,panels)
    if session['layout'].hash!=snapshot['layout_hash'] or session['groups'].hash!=snapshot['group_hash']: raise ValueError('RAM layout changed')
    session['model'].load_state_dict(snapshot['online']);session['optimizer'].load_state_dict(snapshot['optimizer'])
    session['bank'].load_state_dict(snapshot['bank'],device);session['pulse'].load_state_dict(snapshot['pulse'],device)
    session['loader'].set_state(snapshot['loader'])
    for k in ('epoch','global_step','commit_index','complete','rng','snapshots','best_scores','retained_pulse_states','cost','init_hash'): session[k]=cpu_copy(snapshot[k])
    history.restore_rng(snapshot['rng'],device)
    return session
