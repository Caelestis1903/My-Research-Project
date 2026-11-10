"""Actual source-graph cuts, fusion interventions and full-function derivatives."""
from contextlib import contextmanager
import time
import numpy as np
import torch
from experiments.ema_mechanism_20261106.readouts import PredictionBatch


@contextmanager
def _hooks(model, hooks):
    was=model.training;model.eval()
    try: yield hooks
    finally:
        for handle in hooks: handle.remove()
        model.train(was)


def read_predictions(model,x,params=None,full_functional_graph=False):
    outputs={};handles=[]
    with _hooks(model,handles):
        if full_functional_graph:
            values={}
            handles.append(model.skip_conn.register_forward_hook(lambda m,i,o:values.__setitem__('skip',o)))
            handles.append(model.star_mlp.register_forward_hook(lambda m,i,o:values.__setitem__('star',o)))
            for gate in model.gate:
                handles.append(gate.register_forward_pre_hook(lambda m,i:(values['star']+values['skip'],)))
        for k,tower in enumerate(model.tower):
            handles.append(tower.register_forward_hook(lambda m,i,o,k=k:outputs.__setitem__(k,o.squeeze(1))))
        if params is None: p=model(x)
        else:
            buffers={k:v.detach().clone() for k,v in model.named_buffers()}
            p=torch.func.functional_call(model,(params,buffers),(x,),strict=True)
        z=torch.stack([sum(torch.where(x[-1]==d,outputs[3*t+d],torch.zeros_like(outputs[0]))
                           for d in range(3)) for t in range(2)],1)
        return PredictionBatch(p,z)


def capture_cut(model,x,cut):
    values={};handles=[]
    with _hooks(model,handles):
        if cut in ('h','star_plus_skip'):
            handles.append(model.expert[0].register_forward_pre_hook(lambda m,i:values.__setitem__('h',i[0].detach().clone())))
        elif cut in ('fused','fused_tower_input'):
            for k,tower in enumerate(model.tower):
                handles.append(tower.register_forward_pre_hook(lambda m,i,k=k:values.__setitem__(k,i[0].detach().clone())))
        else: raise ValueError('Unknown complete cut')
        with torch.no_grad():read_predictions(model,x)
    return values


def mixed_readout(model,x,cut,donor_values):
    handles=[]
    with _hooks(model,handles):
        if cut in ('h','star_plus_skip'):
            for module in (*model.gate,*model.expert,*model.domain_expert,*model.task_expert):
                handles.append(module.register_forward_pre_hook(lambda m,i:(donor_values['h'],)))
        elif cut in ('fused','fused_tower_input'):
            for k,tower in enumerate(model.tower):
                handles.append(tower.register_forward_pre_hook(lambda m,i,k=k:(donor_values[k],)))
        else: raise ValueError('Unknown complete cut')
        with torch.no_grad():return read_predictions(model,x)


def capture_fusion_parts(model,x):
    for key in ('exp_d','exp_t','bal_d','bal_t'):
        if float(getattr(model,key))!=1.:raise ValueError('Fusion decomposition requires fixed1')
    values={kind:{} for kind in ('gate','shared_experts','domain_output','task_output','actual_fused')};handles=[]
    with _hooks(model,handles):
        for kind,modules in (('gate',model.gate),('shared_experts',model.expert),
                             ('domain_output',model.domain_expert),('task_output',model.task_expert)):
            for k,module in enumerate(modules):
                handles.append(module.register_forward_hook(lambda m,i,o,k=k,kind=kind:values[kind].__setitem__(k,o.detach().clone())))
        for k,module in enumerate(model.tower):
            handles.append(module.register_forward_pre_hook(lambda m,i,k=k:values['actual_fused'].__setitem__(k,i[0].detach().clone())))
        with torch.no_grad():read_predictions(model,x)
    return values


def fusion_components(first,second):
    e0=torch.stack(list(first['shared_experts'].values()),1).double()
    e1=torch.stack(list(second['shared_experts'].values()),1).double();de=e1-e0
    components={key:{} for key in ('content','routing','interaction','domain','task')}
    residuals=[]
    for k in range(6):
        g0=first['gate'][k].double();g1=second['gate'][k].double();dg=g1-g0
        components['content'][k]=torch.bmm(g0[:,None,:],de).squeeze(1)
        components['routing'][k]=torch.bmm(dg[:,None,:],e0).squeeze(1)
        components['interaction'][k]=torch.bmm(dg[:,None,:],de).squeeze(1)
        components['domain'][k]=second['domain_output'][k%3].double()-first['domain_output'][k%3].double()
        components['task'][k]=second['task_output'][k//3].double()-first['task_output'][k//3].double()
        total=sum(part[k] for part in components.values())
        recomputed=(torch.bmm(g1[:,None,:],e1)-torch.bmm(g0[:,None,:],e0)).squeeze(1)+components['domain'][k]+components['task'][k]
        actual=second['actual_fused'][k].double()-first['actual_fused'][k].double()
        residual=actual-total
        algebra=float((total-recomputed).norm()/recomputed.norm().clamp_min(1e-12))
        if algebra>1e-10:raise ArithmeticError('Fusion FP64 identity failed')
        residuals.append(dict(cell_index=k,algebra_relative_l2=algebra,
            source_rounding_max_abs=float(residual.abs().max()),source_rounding_relative_l2=float(residual.norm()/actual.norm().clamp_min(1e-12))))
    return components,residuals


def fusion_part_readout(model,x,parts,component):
    donor={k:(parts['actual_fused'][k].double()-component[k]).float() for k in range(6)}
    return mixed_readout(model,x,'fused',donor)


def _parameters(layout,vector):
    return {name:vector[a:b].reshape(shape) for name,shape,(a,b) in zip(layout.names,layout.shapes,layout.offsets)}


def integrate_path(model,layout,x,theta0,theta1,groups=None,pairs=None,task=None):
    """Gauss-Legendre line integral; groups are disjoint coordinate masks.

    pairs is [K,2] row indices; task is scalar or [K] task indices.
    With no pairs, return per-row/task logit integrals.
    """
    start=time.perf_counter();a=theta0.detach().to(x.device);b=theta1.detach().to(x.device);delta=b-a
    def function(vector):
        z=read_predictions(model,x,params=_parameters(layout,vector),full_functional_graph=True).z
        if pairs is None:return z
        indices=torch.as_tensor(pairs,device=x.device);tasks=torch.as_tensor(0 if task is None else task,device=x.device)
        return z[indices[:,0],tasks]-z[indices[:,1],tasks]
    endpoints={alpha:function(a if alpha==0 else b if alpha==1 else a+alpha*delta).detach().double() for alpha in (0.,.25,.5,.75,1.)}
    actual=endpoints[1.]-endpoints[0.];method='jvp';nodes_used=0
    def derivative(point,direction):
        nonlocal method
        try:return torch.func.jvp(function,(point,),(direction,))[1].detach().double()
        except (NotImplementedError,RuntimeError) as error:
            if 'forward' not in str(error).lower() and 'jvp' not in str(error).lower():raise
            method='reverse_ad_bounded'
            variable=point.detach().requires_grad_(True);result=function(variable)
            values=[]
            for value in result.reshape(-1):
                grad=torch.autograd.grad(value,variable,retain_graph=True)[0]
                values.append((grad*direction).sum().detach().double())
            return torch.stack(values).reshape(result.shape)
    for count in (8,16,32):
        nodes,weights=np.polynomial.legendre.leggauss(count);total=torch.zeros_like(actual)
        for node,weight in zip(nodes,weights):total+=(weight/2)*derivative(a+float((node+1)/2)*delta,delta)
        nodes_used=count;residual=total-actual
        if bool((residual.abs()<=1e-5+.01*actual.abs()).all()):break
    group_integrals={}
    if groups:
        for name,mask in groups.items():
            direction=delta*mask.to(delta.device);value=torch.zeros_like(actual)
            for node,weight in zip(nodes,weights):value+=(weight/2)*derivative(a+float((node+1)/2)*delta,direction)
            group_integrals[name]=value
    return dict(actual_delta=actual.cpu().numpy(),integral=total.cpu().numpy(),
        absolute_residual=residual.abs().cpu().numpy(),relative_residual=(residual.abs()/actual.abs().clamp_min(1e-12)).cpu().numpy(),
        sign_unresolved=(residual.abs()>1e-5+.01*actual.abs()).cpu().numpy(),
        nodes=nodes_used,method=method,seconds=time.perf_counter()-start,
        endpoints={str(k):v.cpu().numpy() for k,v in endpoints.items()},
        group_integrals={k:v.cpu().numpy() for k,v in group_integrals.items()})


def rank_gradient_summary(model,layout,x,cells,masks,vector=None):
    """cells are (positive rows, negative rows, task); every Cartesian pair contributes."""
    if vector is None:vector=layout.flatten(model.parameters())
    vector=vector.detach().requires_grad_(True)
    z=read_predictions(model,x,params=_parameters(layout,vector),full_functional_graph=True).z
    risks=[]
    for positive,negative,task in cells:
        pos=torch.as_tensor(positive,device=x.device);neg=torch.as_tensor(negative,device=x.device)
        if not len(pos) or not len(neg):raise ValueError('Rank gradient cell missing class')
        risks.append(torch.nn.functional.softplus(-(z[pos,task,None]-z[neg,task][None,:])).mean())
    risk=torch.stack(risks).mean();gradient=torch.autograd.grad(risk,vector)[0]
    return dict(rank_risk=float(risk.detach()),gradient_rms={g:float(gradient[m.to(gradient.device)].square().mean().sqrt()) for g,m in masks.items()})


def star_cross_readout(model,layout,x,theta0,theta1,mode='delete',background='F'):
    """Remove/add Δshared_weight*Δslot_weight with the background embedding fixed."""
    if mode not in ('delete','add') or background not in ('O','F'):raise ValueError('Invalid STAR intervention')
    first=_parameters(layout,theta0.to(x.device));second=_parameters(layout,theta1.to(x.device))
    layout.copy_to(model,theta1 if background=='F' else theta0)
    shared=second['shared_weight'].double()-first['shared_weight'].double()
    products=[shared*(second[f'slot_weight.{d}'].double()-first[f'slot_weight.{d}'].double()) for d in range(3)]
    captured={};handles=[]
    def replace(module,inputs):
        emb=captured['embedding'].double();cross=torch.zeros_like(inputs[0],dtype=torch.float64)
        for d in range(3):cross=torch.where((x[-1]==d)[:,None],emb@products[d],cross)
        return ((inputs[0].double()+(-1 if mode=='delete' else 1)*cross).float(),)
    with _hooks(model,handles):
        handles.append(model.embedding.register_forward_hook(lambda m,i,o:captured.__setitem__('embedding',o.detach())))
        handles.append(model.star_mlp.register_forward_pre_hook(replace))
        with torch.no_grad():return read_predictions(model,x)
