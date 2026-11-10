"""Coordinate-exact group interventions and FP64 temporal composition."""
import hashlib
import torch
from experiments.aug_history_20261105.group_registry import GROUPS


def vector_hash(vector):
    return hashlib.sha256(vector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def group_masks(registry):
    coordinates=registry.coordinates
    masks={g:coordinates==i for i,g in enumerate(GROUPS)}
    masks['all']=torch.ones_like(coordinates,dtype=torch.bool)
    masks['N']=coordinates!=0
    for name in ('EMBED','STAR_SKIP','EXPERT'):
        masks[name]=torch.zeros_like(coordinates,dtype=torch.bool)
    for record in registry.records:
        if record['group']!='S': continue
        root=record['parameter_name'].split('.')[0]
        group='EMBED' if root=='embedding' else 'EXPERT' if root=='expert' else 'STAR_SKIP'
        for start,end in record['flat_ranges']: masks[group][start:end]=True
    masks['D_ALL']=(coordinates>=1)&(coordinates<=3)
    masks['T_ALL']=(coordinates>=4)&(coordinates<=5)
    masks['DT_ALL']=(coordinates>=6)&(coordinates<=11)
    families=('EMBED','STAR_SKIP','EXPERT','D_ALL','T_ALL','DT_ALL')
    if not bool(torch.stack([masks[g] for g in families]).sum(0).eq(1).all()):
        raise ValueError('Fine families do not partition layout')
    return masks


def build_state(packet, descriptor, masks=None):
    online=packet['online_fp32'];R=packet['R_fp64']
    fixed=packet.get('F_fp32')
    if fixed is None: fixed=(online.double()+R).float()
    background=descriptor.get('background','O');base=online if background=='O' else fixed
    if background not in ('O','F'): raise ValueError('Unknown background')
    group=descriptor.get('group','all');component=descriptor.get('component','full')
    operation=descriptor.get('operation','splice' if component=='full' else 'add')
    coefficient=float(descriptor.get('coefficient',1.))
    if masks is None: masks=packet.get('masks')
    mask=torch.ones_like(online,dtype=torch.bool) if group=='all' else masks[group].to(online.device)
    if component=='full' and coefficient==1 and (operation=='splice' or (operation=='add' and background=='O') or (operation=='delete' and background=='F')):
        donor=fixed if operation!='delete' else online
        value=base.clone();value[mask]=donor[mask]
    else:
        if component=='full': delta=R
        elif component in ('recent','old'):
            old=packet['old_R_fp64_by_effective_h'][int(descriptor['h'])]
            if isinstance(old,dict): old=old['old_R']
            old=torch.zeros_like(R) if old is None else old.to(R.device)
            delta=old if component=='old' else R-old
        else: raise ValueError('Unknown correction component')
        delta=delta.to(base.device)
        direction=-1 if operation=='delete' else 1
        value=base.double().clone();value[mask]+=direction*coefficient*delta[mask];value=value.float()
    metadata=dict(descriptor,state_hash=vector_hash(value),parent_state_id=packet.get('state_id'),
        mask_hash=vector_hash(mask),numerical_dtype='float64_then_float32' if component!='full' else 'coordinate_copy')
    return value,metadata


def evaluate_state(model,layout,packet,descriptor,x,masks=None):
    from .functional import read_predictions
    value,identity=build_state(packet,descriptor,masks)
    layout.copy_to(model,value,packet.get('buffers'))
    with torch.no_grad(): output=read_predictions(model,x)
    return output,identity


def matched_states(fixed, first, second):
    a=first.double()-fixed.double();b=second.double()-fixed.double()
    na=float(a.norm());nb=float(b.norm());radius=min(na,nb)
    info=dict(first_norm=na,second_norm=nb,radius=radius,informative=radius>0)
    if radius==0: return fixed.clone(),fixed.clone(),info
    return (fixed.double()+a*(radius/na)).float(),(fixed.double()+b*(radius/nb)).float(),info


def source_direction_states(online, pretheta, data_grad, mhat, vhat, mask, *, lr, eps, weight_decay):
    """Two virtual source directions matched to the real step within one scope."""
    actual=online.double()-pretheta.double();radius=actual[mask].norm()
    numerator=data_grad.double()+weight_decay*pretheta.double()
    directions={'current_preconditioned':-lr*numerator/(vhat.double().sqrt()+eps),
                'momentum_isotropic':-mhat.double()}
    result={}
    for key,direction in directions.items():
        norm=direction[mask].norm();value=online.double().clone()
        if norm>0: value[mask]=pretheta.double()[mask]+direction[mask]*(radius/norm)
        result[key]=(value.float(),dict(actual_norm=float(radius),direction_norm=float(norm),informative=bool(norm>0)))
    return result
