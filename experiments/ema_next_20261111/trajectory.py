"""Source trajectories, real A feedback, and simultaneous registered B/C readouts."""
import copy
import hashlib
import time
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as source
from experiments.ema_mechanism_20261106.worker import create_session as source_session,source_losses
from experiments.ema_mechanism_20261106.panels import load_panels
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import cpu_copy,state_digest
from experiments.ema_causal_20261110.interventions import group_masks
from .history_readout import LaggedFStore,split_history,split_packet,shared_only
from .optimizer_direction import capture_embed_inputs,apply_matched_direction,set_embed_beta1_zero
from .conditional_readout import build_unique_pairs,fit_from_PA,apply_lagged_policy,mix_logits,sign_change_fraction,pair_margins
from .evaluation import (evaluate_full_population,predict_panels,panel_metrics,update_earliest_best,
                         materialize_packet,prediction_hash,sigmoid,combine_predictions)


def put_object(session,value):
    key=state_digest(value)
    if key not in session['objects']:
        session['objects'][key]=value;session['new_objects'][key]=value
    return key


def required_objects(snapshot,objects):
    keep={};pending=list(snapshot.get('object_refs',[]))
    while pending:
        key=pending.pop()
        if key in keep:continue
        keep[key]=objects[key];pending.extend(objects[key].get('refs',[]))
    return keep


def reduce_session(session):
    state=export_session(session,copy_live=False)
    session['objects']=required_objects(state,session['objects']);session['new_objects']={}


def _menu(job,prepared):
    if 'readout_menu' in job:return copy.deepcopy(job['readout_menu'])
    if job['family']=='PARENT':return copy.deepcopy(prepared['readouts'][job['dataset']])
    result=[dict(id='O',family='O',forwards=1),dict(id='F',family='F',forwards=1)]
    if job.get('frozen_B'):result.append(copy.deepcopy(job['frozen_B']))
    return result


def create_parent(job,prepared,device):
    job=copy.deepcopy(job);bundle=cached_bundle(prepared['cache_root'],job['dataset'],include_test=False)
    panels=load_panels(prepared['out'],job['dataset'],job['seed'],bundle)
    session=source_session(job,bundle,device,panels)
    for key in ('pulse','snapshots','best_scores','retained_pulse_states','innovation'):session.pop(key,None)
    session.update(prepared=prepared,objects={},new_objects={},selected={},prefix={},anchors={},pending_c=None,
                   menu=_menu(job,prepared),raw_active=False,training_updates=0)
    session['masks']=group_masks(session['groups'])
    session['context']=dict(scratch=session['scratch'],layout=session['layout'],masks=session['masks'],device=str(device))
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in session['model'].modules()):raise ValueError('Unexpected stateful source layer')
    windows=sorted({int(r['H']) for r in session['menu'] if 'H' in r})
    steps=job['config']['steps_per_epoch']
    session['history']=LaggedFStore([e*steps for e in range(1,51)],windows,job['config']['rho'])
    session['history'].capture(0,session['bank'].get('F'),lambda x:put_object(session,x))
    has_c=any(r['family'].startswith('C_') for r in session['menu'])
    session['pairs']={name:build_unique_pairs(panels[name],job['dataset'],job['seed']) for name in ('P_A','P_B')} if has_c else {}
    return session


def train_one_step(s,x,y,row_indices=None):
    j=s['job'];c=j['config'];optimizer=s['optimizer'];step=s['global_step']+1
    optimizer.zero_grad(set_to_none=True)
    if j['family'] in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):
        from .distillation import student_loss
        loss,info=student_loss(s,x,y,row_indices)
    else:
        loss,info=source_losses(s['model'],x,y,0,dict(num_views=int(j['objective']=='gaussian_d0'),view_family='gaussian',pairing='iid'),
                                j['dataset'],j['seed'],step,mix_lambda=.5,kappa=c['kappa'])
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite source training loss')
    loss.backward()
    active=j['family'] in ('A_MATCH','A_RAW') and s['epoch']>=j.get('start_epoch',20)
    pre=capture_embed_inputs(s['model'],optimizer,s['layout'],s['masks']['EMBED']) if active else None
    if active and j['family']=='A_RAW' and not s['raw_active']:
        set_embed_beta1_zero(optimizer,s['layout'],s['masks']['EMBED'],s['model']);s['raw_active']=True
    optimizer.step();diag=None
    if active and j['family']=='A_MATCH':diag=apply_matched_direction(pre,optimizer,j['direction_alpha'])
    elif active:
        change=torch.cat([r['parameter'][r['mask']].double()-r['before'] for r in pre]) if pre else torch.zeros(1)
        diag=dict(applied_norm=float(change.norm()),native_norm=float(change.norm()),relative_error=0.,cosine=1.,fallback=None)
    theta=s['layout'].flatten(s['model'].parameters());s['bank'].after_adam(theta,step)
    s['global_step']=step;s['training_updates']+=1
    s['history'].capture(step,s['bank'].get('F'),lambda x:put_object(s,x))
    return dict(loss=float(loss.detach()),clean_loss=float(info['real'].mean(1).mean().detach()),
                soft_loss=float(info['soft'].mean().detach()) if 'soft' in info else None,direction=diag)


def _base(s,vector,epoch):
    return put_object(s,dict(kind='single',vector=vector.detach().cpu().clone(),buffers=cpu_copy(dict(s['model'].named_buffers())),refs=[]))


def _descriptor(s,base_id,spec,epoch,parent_F_ref=None,**extra):
    base=s['objects'][base_id]
    descriptor=dict(kind='predictor',base_ref=base_id,config=copy.deepcopy(s['job']['config']),
                    epoch=epoch,step=s['global_step'],spec=copy.deepcopy(spec),refs=[base_id],
                    dataset=s['job']['dataset'],seed=s['job']['seed'],objective=s['job']['objective'],source_job=s['job']['job_id'],**extra)
    if parent_F_ref is not None:descriptor.update(parent_F_ref=parent_F_ref,refs=[base_id,parent_F_ref])
    identity={k:v for k,v in descriptor.items() if k not in ('kind','spec','refs','parent_F_ref','epoch','step','source_job','config')}
    descriptor['state_id']=state_digest(identity)
    return put_object(s,descriptor)


def _mix_predictions(recent,old,a,mode,domains):
    return combine_predictions(recent,old,a,domains,mode)


def _mix_panels(recent,old,a,mode,panels):
    result={}
    for name,panel in panels.items():
        r,o=recent[name],old[name];domains=panel['natural_x'][...,-1].reshape(-1)
        natural=_mix_predictions(dict(p=r['natural_p'].reshape(-1,2),z=r['natural_z'].reshape(-1,2)),dict(p=o['natural_p'].reshape(-1,2),z=o['natural_z'].reshape(-1,2)),a,mode,domains)
        v=dict(natural_p=natural['p'].reshape(r['natural_p'].shape),natural_z=natural['z'].reshape(r['natural_z'].shape))
        if name!='V_mech':
            w=np.asarray(a,dtype=np.float32)
            if w.ndim:w=w[:,None]
            z=((1-w)*r['rank_z']+w*o['rank_z']).astype(np.float32)
            v.update(rank_z=z,rank_p=((1-w)*r['rank_p']+w*o['rank_p']).astype(np.float32) if mode=='probability' else sigmoid(z))
        result[name]=v
    return result


def _registered(s,key,spec,packet_id,epoch,readouts,pred=None,panels=None):
    packet=materialize_packet(s['objects'][packet_id],s['objects'])
    if pred is None:
        values,pred=evaluate_full_population(packet,s['bundle'],context=s['context'],eval_batch_size=s['job']['config']['eval_batch_size'],return_predictions=True)
    else:values=source.metrics(s['bundle'].val_y,pred['p'],s['bundle'].val_x[:,-1])
    readouts[key]=dict(metrics=values,state_id=s['objects'][packet_id]['state_id'],packet_id=packet_id,spec=spec,prediction_hash=prediction_hash(pred))
    update_earliest_best(s['selected'],key,packet_id,epoch,values,spec)
    return pred


def _pair_transitions(panel,changed,reference,online):
    values=[]
    for cell in range(6):
        if panel['panel_id']=='V_mech':
            domain,task=divmod(cell,2);table=panel['pairs'][cell,:int(panel['pair_counts'][cell])]
            cp=changed['natural_p'][domain,:,task];rp=reference['natural_p'][domain,:,task];op=online['natural_p'][domain,:,task]
        else:
            table=panel['rank_pairs'][cell];cp=changed['rank_p'][cell];rp=reference['rank_p'][cell];op=online['rank_p'][cell]
        c=np.sign(pair_margins(cp,table));r=np.sign(pair_margins(rp,table));o=np.sign(pair_margins(op,table))
        corrected=float(np.maximum(c-r,0).sum()/2);broken=float(np.maximum(r-c,0).sum()/2)
        values.append(dict(cell=cell,pairs=len(table),corrected=corrected,broken=broken,net=corrected-broken,
                           overlap_with_F_corrections=int(np.sum((c>r)&(r>o)))))
    return values


def observe_epoch(s,epoch):
    readouts={};diagnostics=dict(B={},C={});c=s['job']['config'];theta=s['layout'].flatten(s['model'].parameters());fixed=s['bank'].get('F')
    menu={r['id']:r for r in s['menu']};buffer=cpu_copy(dict(s['model'].named_buffers()))
    single_ids={key:_base(s,value,epoch) for key,value in (('O',theta),('F',fixed))}
    single_packets={key:_descriptor(s,pid,menu.get(key,dict(id=key,family=key)),epoch) for key,pid in single_ids.items()}
    full={};panel_pred={}
    for key in ('O','F'):
        full[key]=_registered(s,key,menu.get(key,dict(id=key,family=key)),single_packets[key],epoch,readouts)
        if any(r['family']=='B_SPLIT' for r in menu.values()):panel_pred[key]=predict_panels(materialize_packet(s['objects'][single_packets[key]],s['objects']),s['panels'],s['context'],c['eval_batch_size'])
    for key,spec in menu.items():
        if spec['family']=='B_SHARED_ONLY':
            pid=_descriptor(s,_base(s,shared_only(fixed,theta,s['masks']['S']),epoch),spec,epoch)
            _registered(s,key,spec,pid,epoch,readouts)
    needs={(int(r['H']),r.get('scope','S')) for r in menu.values() if 'H'in r}
    origins=s['history'].consume(s['global_step'],s['objects']) if needs else {}
    c_panels={};c_used={};c_pending=None
    for H,scope in sorted(needs):
        h=min(H,s['global_step']);recent,old,audit=split_history(fixed,origins[H],c['rho'],h,s['masks']['S'] if scope=='S' else None)
        base=split_packet(recent,old,fixed,s['masks']['S'],scope,H=H,**audit,buffers=buffer,refs=[])
        base_id=put_object(s,base)
        component=[];component_panels=[]
        for vector in (recent,old):
            packet=dict(kind='single',vector=vector,buffers=buffer,config=c)
            _,pred=evaluate_full_population(packet,s['bundle'],context=s['context'],eval_batch_size=c['eval_batch_size'],return_predictions=True)
            component.append(pred);component_panels.append(predict_panels(packet,s['panels'],s['context'],c['eval_batch_size']))
        for key,spec in menu.items():
            if spec.get('H')!=H or spec.get('scope','S')!=scope or spec['family']!='B_SPLIT':continue
            pid=_descriptor(s,base_id,spec,epoch,parent_F_ref=single_packets['F'],mode=spec['mode'],a=audit['a'])
            pred=_mix_predictions(*component,audit['a'],spec['mode'],s['bundle'].val_x[:,-1])
            pp=_mix_panels(*component_panels,audit['a'],spec['mode'],s['panels'])
            _registered(s,key,spec,pid,epoch,readouts,pred)
            diagnostics['B'][key]=dict(audit,scope=scope,H=H,component_logit_rms=float(np.sqrt(np.mean((component[0]['z'].astype(np.float64)-component[1]['z'])**2))),
                mixture_minus_F_logit_rms=float(np.sqrt(np.mean((pred['z'].astype(np.float64)-full['F']['z'])**2))),
                pair_transitions={name:_pair_transitions(panel,pp[name],panel_pred['F'][name],panel_pred['O'][name]) for name,panel in s['panels'].items()})
        c_specs=[(key,spec) for key,spec in menu.items() if spec.get('H')==H and spec.get('scope','S')==scope and spec['family'].startswith('C_')]
        if c_specs:
            for key,spec in c_specs:
                used=apply_lagged_policy(s['pending_c'],audit['a'],epoch,spec.get('shrinkage',0.));mode='parameter' if spec['family'].endswith('PARAM') else 'logit'
                parent=single_packets['F'] if mode=='logit' else None
                chosen_base=base_id
                if spec['family']=='C_GLOBAL_PARAM':
                    alpha=used['weights'][0];chosen_base=_base(s,((1-alpha)*recent.double()+alpha*old.double()).float(),epoch)
                pid=_descriptor(s,chosen_base,spec,epoch,parent_F_ref=parent,mode=mode,a=audit['a'],weights=used['weights'],policy=used)
                if mode=='logit':
                    pred=_mix_predictions(*component,used['weights'],'logit',s['bundle'].val_x[:,-1]);pp=_mix_panels(*component_panels,used['weights'],'logit',s['panels'])
                    _registered(s,key,spec,pid,epoch,readouts,pred)
                else:
                    _registered(s,key,spec,pid,epoch,readouts)
                    pp=predict_panels(materialize_packet(s['objects'][pid],s['objects']),s['panels'],s['context'],c['eval_batch_size'])
                c_panels[key]=pp;c_used[key]=used
            c_pending=fit_from_PA(component_panels[0]['P_A']['rank_z'],component_panels[1]['P_A']['rank_z'],s['pairs']['P_A'],audit['a'],epoch)
    for key,pp in c_panels.items():
        spec=menu[key];mode='PARAM' if spec['family'].endswith('PARAM') else 'PRED';global_key=f'C_GLOBAL_{mode}'
        raw=panel_metrics(s['panels']['P_B'],pp['P_B'],s['pairs']['P_B'])
        ref=panel_metrics(s['panels']['P_B'],c_panels.get(global_key,pp)['P_B'],s['pairs']['P_B'])
        risk=[a['risk']-b['risk'] if a['risk'] is not None and b['risk'] is not None else None for a,b in zip(raw['cells'],ref['cells'])]
        auc=[a['auc']-b['auc'] if a['auc'] is not None and b['auc'] is not None else None for a,b in zip(raw['cells'],ref['cells'])]
        changes=[sign_change_fraction(pp['P_B']['rank_z'][i],c_panels.get(global_key,pp)['P_B']['rank_z'][i],pairs) for i,pairs in enumerate(s['pairs']['P_B']['tables'])]
        complete=all(v is not None for v in risk+auc+changes)
        diagnostics['C'][key]=dict(used=c_used[key],raw_PB=raw,global_PB=ref,complete=complete,
            alpha_abs_difference=float(np.mean(np.abs(np.asarray(c_used[key]['weights'])-c_used[key]['global_alpha']))),
            pair_rank_change_rate=float(np.mean(changes)) if complete else None,pb_risk_delta=float(np.mean(risk)) if complete else None,
            pb_auc_delta=float(np.mean(auc)) if complete else None,pb_cell_risk_delta=risk,pb_cell_auc_delta=auc)
    if c_pending is not None:s['pending_c']=c_pending;diagnostics['C_fit']=c_pending
    diagnostics['pair_identity']={name:p['identity'] for name,p in s['pairs'].items()} if epoch==1 else {}
    return readouts,diagnostics


def _roots(state):
    refs=[v['packet_id'] for name in ('selected','prefix') for v in state.get(name,{}).values()]
    refs+=list(state.get('anchors',{}).values())+list(state.get('history',{}).get('records',{}).values())
    if state.get('teacher_ref'):refs.append(state['teacher_ref'])
    return list(dict.fromkeys(refs))


def export_session(s,copy_live=True):
    result={k:copy.deepcopy(s[k]) for k in ('job','epoch','global_step','complete','selected','prefix','anchors','pending_c','raw_active','training_updates','cost','init_hash')}
    result.update(rng=s['rng'],online=s['model'].state_dict(),optimizer=s['optimizer'].state_dict(),bank=s['bank'].state_dict(),loader=s['loader'].get_state(),history=s['history'].state_dict())
    if 'teacher_ref'in s:result.update(teacher_ref=s['teacher_ref'],soft_target_hash=s['soft_target_hash'],teacher_row_hash=s['teacher_row_hash'])
    result['object_refs']=_roots(result)
    return cpu_copy(result) if copy_live else result


def _prefix_packet(s,selected):
    result={}
    for key,value in selected.items():
        if key not in ('O','F') and not value['spec']['family']=='B_SPLIT':continue
        packet=s['objects'][value['packet_id']]
        if 'parent_F_ref'in packet:
            packet={k:v for k,v in packet.items() if k!='parent_F_ref'};packet['refs']=[packet['base_ref']]
            value=dict(value,packet_id=put_object(s,packet))
        result[key]=copy.deepcopy(value)
    return result


def _save_anchor(s,epoch):
    snapshot=export_session(s)
    snapshot['anchors']={}
    snapshot['selected']=_prefix_packet(s,s['selected']) if epoch==20 else {}
    snapshot['prefix']=copy.deepcopy(snapshot['selected'])
    snapshot['pending_c']=None;snapshot['object_refs']=_roots(snapshot)
    key=put_object(s,dict(kind='anchor',snapshot=snapshot,refs=snapshot['object_refs']))
    s['anchors'][str(epoch)]=key
    if epoch==20:s['prefix']=copy.deepcopy(snapshot['prefix'])


def advance_epoch(s):
    j=s['job'];c=j['config'];epoch=s['epoch']+1
    source.restore_rng(s['rng'],s['device']);s['model'].train();start=time.perf_counter()
    order=torch.randperm(len(s['bundle'].train_y),generator=s['loader']).numpy();loss=clean=soft=0.;directions=[]
    for offset in range(0,len(order),c['batch_size']):
        ids=order[offset:offset+c['batch_size']]
        x=torch.as_tensor(np.array(s['bundle'].train_x[ids]),device=s['device']).T
        y=torch.as_tensor(np.array(s['bundle'].train_y[ids]),device=s['device'])
        value=train_one_step(s,x,y,ids);loss+=value['loss']*len(ids);clean+=value['clean_loss']*len(ids)
        if value['soft_loss'] is not None:soft+=value['soft_loss']*len(ids)
        if value['direction']:directions.append(value['direction'])
    training_seconds=time.perf_counter()-start;rng=source.rng_state(s['device']);observer_start=time.perf_counter()
    readouts,diagnostics=observe_epoch(s,epoch)
    source.restore_rng(rng,s['device']);s.update(epoch=epoch,rng=rng,complete=epoch>=j['end_epoch'])
    if j['family']=='PARENT' and epoch in (20,35):_save_anchor(s,epoch)
    if directions:
        diagnostics['A']=dict(steps=len(directions),native_norm_mean=float(np.mean([x['native_norm'] for x in directions])),
            applied_norm_mean=float(np.mean([x['applied_norm'] for x in directions])),relative_error_max=max(x['relative_error'] for x in directions),
            cosine_mean=float(np.mean([x['cosine'] for x in directions])),fallback_count=sum(x['fallback'] is not None for x in directions))
    observation_seconds=time.perf_counter()-observer_start;elapsed=time.perf_counter()-start
    s['cost']['wall_seconds']=s['cost'].get('wall_seconds',0.)+elapsed
    record=dict(job_id=j['job_id'],dataset=j['dataset'],seed=j['seed'],objective=j['objective'],family=j['family'],epoch=epoch,global_step=s['global_step'],
                training_loss=loss/len(order),clean_loss=clean/len(order),soft_loss=soft/len(order) if 'teacher_ref'in s else None,
                readouts=readouts,diagnostics=diagnostics,order_hash=hashlib.sha256(s['bundle'].train_row_ids[order].tobytes()).hexdigest(),
                cost=dict(wall_seconds=elapsed,train_seconds=training_seconds,observation_seconds=observation_seconds,real_adam_steps=c['steps_per_epoch'],
                          gpu_peak_allocated=torch.cuda.max_memory_allocated(s['device']) if str(s['device']).startswith('cuda') else 0))
    needed=required_objects(export_session(s,copy_live=False),s['objects'])
    return record,{key:value for key,value in s['new_objects'].items() if key in needed}


def _trim_history(state,windows,current_epoch,steps,rho):
    history=LaggedFStore([e*steps for e in range(current_epoch+1,51)],windows,rho)
    old=state['records']
    history.records={int(k):v for k,v in old.items() if int(k) in history.users}
    missing=[origin for origin in history.users if origin<=current_epoch*steps and origin not in history.records]
    if missing:raise ValueError('Missing fork history '+str(missing))
    return history


def restore_session(job,prepared,device,snapshot,objects):
    if snapshot.get('kind')=='anchor':snapshot=snapshot['snapshot']
    job=copy.deepcopy(job)
    if job['family'] in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):job['config']['lr']=job['student_lr']
    s=create_parent(job,prepared,device)
    s['objects']=dict(objects);s['new_objects']={}
    if snapshot.get('raw_active'):set_embed_beta1_zero(s['optimizer'],s['layout'],s['masks']['EMBED'],s['model'])
    s['model'].load_state_dict(snapshot['online']);s['optimizer'].load_state_dict(cpu_copy(snapshot['optimizer']))
    s['bank'].load_state_dict(snapshot['bank'],device);s['loader'].set_state(snapshot['loader'])
    for key in ('epoch','global_step','rng','selected','prefix','anchors','pending_c','raw_active','training_updates','cost','init_hash'):
        if key in snapshot:s[key]=cpu_copy(snapshot[key])
    s['history']=LaggedFStore.from_state(snapshot['history']);s['complete']=s['epoch']>=job['end_epoch']
    if job['family'] in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):
        from .distillation import attach_teacher
        s['teacher_ref']=snapshot['teacher_ref'];attach_teacher(s,s['objects'][s['teacher_ref']],s['objects'])
        if s['soft_target_hash']!=snapshot['soft_target_hash']:raise ValueError('Teacher recovery target identity changed')
    source.restore_rng(s['rng'],device)
    return s


def fork_at_epoch(job,prepared,device,snapshot,objects):
    if snapshot.get('kind')=='anchor':snapshot=snapshot['snapshot']
    s=restore_session(job,prepared,device,snapshot,objects)
    keep={'O','F'}|({job['frozen_B']['id']} if job.get('frozen_B') else set())
    s['selected']={k:v for k,v in s['selected'].items() if k in keep};s['prefix']={k:v for k,v in s['prefix'].items() if k in keep};s['anchors']={};s['pending_c']=None
    windows=[job['frozen_B']['H']] if job.get('frozen_B') else []
    s['history']=_trim_history(snapshot['history'],windows,s['epoch'],job['config']['steps_per_epoch'],job['config']['rho'])
    s['training_updates']=0
    reduce_session(s)
    return s


def continue_to_epoch(s,end_epoch,commit=None):
    while s['epoch']<end_epoch:
        record,objects=advance_epoch(s)
        if commit:commit(record,export_session(s),objects)
        reduce_session(s)
    return s
