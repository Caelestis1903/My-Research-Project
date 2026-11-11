"""Frozen two-state teachers and source-scaled, paired five-epoch students."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.mdmt_history_information_20261103 import protocol as source
from experiments.ema_mechanism_20261106.shadows import FixedShadowBank
from experiments.overnight_ab_20261103.state import state_digest
from .evaluation import materialize_packet,predict_arrays


def freeze_teacher(packet,objects):
    if packet['kind']!='predictor' or not packet.get('parent_F_ref'):raise ValueError('Teacher lacks its exact same-epoch parent F')
    parent=objects[packet['parent_F_ref']]
    if parent['epoch']!=packet['epoch']:raise ValueError('Teacher and initialization epoch differ')
    return dict(packet)


def cache_train_targets(packet,bundle,context,objects,eval_batch_size=40960):
    teacher=materialize_packet(packet,objects)
    pred=predict_arrays(teacher,bundle.train_x,eval_batch_size,context)['p']
    if pred.shape!=bundle.train_y.shape or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):
        raise ValueError('Invalid train teacher probabilities')
    return dict(kind='soft_targets',probabilities=pred,teacher_hash=packet['state_id'],
                row_hash=hashlib.sha256(bundle.train_row_ids.tobytes()).hexdigest(),refs=[])


def attach_teacher(session,teacher_packet,objects):
    teacher=freeze_teacher(teacher_packet,objects);j=session['job']
    parent=objects[teacher['parent_F_ref']]
    target_packet=parent if j['family']=='KD_FTEACHER' else teacher
    row_hash=hashlib.sha256(session['bundle'].train_row_ids.tobytes()).hexdigest()
    if j['family']=='KD_LABEL':
        session['soft_targets']=None;session['soft_target_hash']='LABEL_ONLY'
    else:
        cached=None
        cache=session['prepared'].get('teacher_target_cache')
        guardian=Path(session['prepared']['out'])/'guardian.json'
        request=None
        if guardian.exists():
            from .state import request
            cached=request(session['prepared']['out'],'teacher_get',key=target_packet['state_id'])['target']
        if cached is None and cache is not None:cached=cache.get(target_packet['state_id'])
        if cached is None:
            cached=cache_train_targets(target_packet,session['bundle'],session['context'],objects,j['config']['eval_batch_size'])
            if cache is not None:cache[target_packet['state_id']]=cached
            if request is not None:request(session['prepared']['out'],'teacher_put',key=target_packet['state_id'],target=cached)
        if cached['row_hash']!=row_hash or cached['teacher_hash']!=target_packet['state_id']:raise ValueError('Teacher cache row/state identity mismatch')
        session['soft_targets']=cached['probabilities'];session['soft_target_hash']=state_digest(cached)
    session['teacher_row_hash']=row_hash


def create_student(job,prepared,device,teacher_packet,objects):
    from .trajectory import create_parent,put_object,observe_epoch,export_session
    teacher=freeze_teacher(teacher_packet,objects)
    job=copy.deepcopy(job);job['config']['lr']=job['student_lr']
    session=create_parent(job,prepared,device);session['objects']=dict(objects);session['new_objects']={}
    session['teacher_ref']=put_object(session,teacher)
    parent=materialize_packet(objects[teacher['parent_F_ref']],objects)
    session['layout'].copy_to(session['model'],parent['vector'].to(device),parent['buffers'])
    initial=session['layout'].flatten(session['model'].parameters())
    session['bank']=FixedShadowBank(initial,{'F':job['config']['rho']},{})
    session['init_hash']=state_digest(session['model'].state_dict())
    if session['optimizer'].state:raise ValueError('Student Adam must start fresh')
    attach_teacher(session,teacher,objects)
    rng=source.rng_state(device);readouts,diagnostics=observe_epoch(session,0);source.restore_rng(rng,device);session['rng']=rng
    session['initial_record']=dict(job_id=job['job_id'],dataset=job['dataset'],seed=job['seed'],objective=job['objective'],family=job['family'],
        epoch=0,global_step=0,readouts=readouts,diagnostics=diagnostics,teacher_state_id=teacher['state_id'],teacher_epoch=teacher['epoch'],
        initializer_state_id=objects[teacher['parent_F_ref']]['state_id'],soft_target_hash=session['soft_target_hash'],train_row_hash=session['teacher_row_hash'],
        cost=dict(wall_seconds=0.,real_adam_steps=0))
    return session


def student_loss(session,x,y,row_indices):
    prediction=session['model'](x);hard=F.binary_cross_entropy(prediction,y,reduction='none')
    weight=float(session['job'].get('distill_weight',0.))
    if session['job']['family']=='KD_LABEL':weight=0.
    if weight:
        if row_indices is None:raise ValueError('Teacher targets require original train row indices')
        q=torch.as_tensor(np.array(session['soft_targets'][row_indices]),device=x.device,dtype=prediction.dtype)
        soft=F.binary_cross_entropy(prediction,q.detach(),reduction='none')
    else:soft=hard.detach()
    loss=(1-weight)*hard.mean(1).mean()+weight*soft.mean(1).mean()
    return loss,dict(real=hard,soft=soft,extra_views=0)


def student_step(session,x,y,row_indices):
    from .trajectory import train_one_step
    return train_one_step(session,x,y,row_indices)
