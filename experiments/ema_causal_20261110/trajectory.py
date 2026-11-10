"""One source trajectory with read-only causal observations and sparse RAM history."""
import copy
import hashlib
import time
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as source
from experiments.ema_mechanism_20261106.worker import create_session as source_session, train_batch as source_batch
from experiments.ema_mechanism_20261106.readouts import panel_predictions, read_predictions, pulse_rows
from experiments.ema_mechanism_20261106.shadows import PulseWindow
from experiments.overnight_ab_20261103.state import cpu_copy, state_digest, tensor_bytes
from .protocol import WINDOWS, ANCHORS, schedule
from .interventions import group_masks


def predictions(model,panels): return {name:panel_predictions(model,p) for name,p in panels.items()}


def full_prediction(model,bundle,batch_size=40960,split='val',operator=None):
    xall=getattr(bundle,split+'_x'); y=getattr(bundle,split+'_y');p=[];z=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for offset in range(0,len(y),batch_size):
            x=torch.as_tensor(np.array(xall[offset:offset+batch_size]),device=device).T
            value=read_predictions(model,x) if operator is None else operator(x)
            p.append(value.p.cpu().numpy());z.append(value.z.cpu().numpy())
    p=np.concatenate(p);z=np.concatenate(z)
    return source.metrics(y,p,xall[:,-1]),p,z


def _pb_objective(s):
    from experiments.ema_mechanism_20261106.worker import source_losses
    panel=s['panels']['P_B'];x=torch.as_tensor(panel['natural_x'].reshape(-1,panel['natural_x'].shape[-1]).copy(),device=s['device']).T
    y=torch.as_tensor(panel['natural_y'].reshape(-1,2).copy(),device=s['device'])
    j=s['job'];c=j['config']
    with torch.no_grad():
        loss,_=source_losses(s['scratch'],x,y,0,dict(num_views=int(j['objective']=='gaussian_d0'),view_family='gaussian',pairing='iid'),j['dataset'],j['seed'],s['global_step'],mix_lambda=.5,kappa=c['kappa'])
    return float(loss)


def create_session(job,bundle,device,panels):
    s=source_session(job,bundle,device,panels)
    s['snapshots']={};s['best_scores']={'O':-float('inf'),'F':-float('inf')}
    s['selected']={};s['pending_packets']={};s['packet_refs']={};s['matched']=PulseWindow(job['config']['rho']);s['matched_done']={};s['source_packets']={}
    s['events']={};s['schedule']=schedule(job['dataset']);s['history_users']={}
    for t in s['schedule']['epoch_end_steps']:
        for h in set(min(h,t) for h in WINDOWS[job['dataset']]): s['history_users'].setdefault(t-h,set()).add(t)
    theta=s['layout'].flatten(s['model'].parameters());s['ideal_R']=torch.zeros_like(theta,dtype=torch.float64)
    s['q']=predictions(s['scratch'],panels)
    s['q']={p:{k:v.astype(np.float64) for k,v in values.items()} for p,values in s['q'].items()}
    s['last_online']=predictions(s['scratch'],panels)
    capture_history(s)
    return s


def capture_history(s):
    step=s['global_step']
    if step not in s['history_users']: return
    theta=s['layout'].flatten(s['model'].parameters());f=s['bank'].get('F')
    s['layout'].copy_to(s['scratch'],theta)
    s['events'][step]=dict(R=(f.double()-theta.double()).cpu(),predictions=copy.deepcopy(s['last_online']),pb_objective=_pb_objective(s),rounding_max_abs=float((f.double()-theta.double()-s['ideal_R']).abs().max()))


def train_batch(s,x,y,observe=True):
    before=s['layout'].flatten(s['model'].parameters()) if observe else None
    step=s['global_step']+1;j=s['job'];c=j['config'];S=c['steps_per_epoch']
    keep_source=observe and j['objective']=='clean' and step in (20*S,35*S)
    if keep_source:
        # The original step is executed once; its gradients remain available afterwards.
        pretheta=before.detach().cpu().clone()
    value=source_batch(s,x,y,observe=False)
    if not observe: return value
    theta=s['layout'].flatten(s['model'].parameters());rho=c['rho']
    s['ideal_R'].mul_(rho).sub_(theta.double()-before.double(),alpha=rho)
    s['matched'].after_adam(theta,step)
    if keep_source:
        optimizer=s['optimizer'];m=[];v=[];g=[];t=None
        for p in s['model'].parameters():
            st=optimizer.state[p];t=int(st['step']);m.append(st['exp_avg']/(1-c['betas'][0]**t));v.append(st['exp_avg_sq']/(1-c['betas'][1]**t));g.append(p.grad)
        s['source_packets'][step//S]=dict(pretheta=pretheta,data_grad=s['layout'].flatten(g).cpu(),mhat=s['layout'].flatten(m).cpu(),vhat=s['layout'].flatten(v).cpu(),step=t)
    rng=source.rng_state(s['device'])
    s['layout'].copy_to(s['scratch'],theta)
    s['last_online']=predictions(s['scratch'],s['panels'])
    for panel,values in s['q'].items():
        for key,q in values.items(): q*=rho;q+=(1-rho)*s['last_online'][panel][key]
    capture_history(s)
    if step in s['schedule']['matched_start_steps']:
        end=(35 if step==35*S-WINDOWS[j['dataset']][1] else 50)
        s['matched'].begin(s['bank'].get('F'),end,step)
    source.restore_rng(rng,s['device'])
    return value


def rank_gradient(s,f):
    panel=s['panels']['P_A'];rank=panel['rank_x'];device=s['device'];layout=s['layout']
    flat=f.detach().clone().requires_grad_(True)
    params={n:flat[a:b].view(shape) for n,shape,(a,b) in zip(layout.names,layout.shapes,layout.offsets)}
    x=torch.as_tensor(rank.reshape(-1,rank.shape[-1]).copy(),device=device).T
    z=s['scratch'].forward_logits(x,params=params,probe_input_full_graph=True).reshape(6,512,2)
    risk=sum(torch.nn.functional.softplus(z[c,256:,c%2][None,:]-z[c,:256,c%2][:,None]).mean() for c in range(6))/6
    grad=torch.autograd.grad(risk,flat)[0].double();masks=group_masks(s['groups'])
    return {g:float(grad[masks[g].to(device)].square().mean().sqrt()) for g in ('all','S','N')}


def panel_metrics(panels,preds):
    rows=[]
    for name,p in panels.items(): rows+=pulse_rows(p,preds[name],'state',None,'state','')
    return {name:[{k:r[k] for k in ('auc','logloss')} for r in rows if r['panel_id']==name] for name in ('P_A','P_B','V_mech','V_A','V_B')}


def pulse_observations(s,packet):
    epoch=packet['epoch'];f=s['bank'].get('F');layout=s['layout'];masks=group_masks(s['groups']);device=f.device
    pulse=s['pulse'].finish(s['global_step']);g=rank_gradient(s,f);r=packet['R_fp64'].to(device)
    h=WINDOWS[s['job']['dataset']][1];eff=min(h,s['global_step']);old=packet['old_R_fp64_by_effective_h'][eff].to(device);recent=r-old
    layout.copy_to(s['scratch'],f);base=predictions(s['scratch'],s['panels']);rows=[]
    for menu in ('rho7','mechanism7'):
        actions=[('F','all',f)]
        for scope in ('all','S','N'):
            mask=masks[scope].to(device)
            for variant in (('fast','slow') if menu=='rho7' else ('drop_recent','drop_old')):
                value=f.clone()
                if menu=='rho7': value[mask]=pulse[variant][mask]
                else: value[mask]=(f.double()[mask]-(recent if variant=='drop_recent' else old)[mask]).float()
                actions.append((scope+'/'+variant,scope,value))
        for action,scope,value in actions:
            layout.copy_to(s['scratch'],value);pred=base if action=='F' else predictions(s['scratch'],s['panels']);mask=masks[scope].to(device)
            rv=r[mask];dv=(value.double()-f.double())[mask];rn=rv.norm();dn=dv.norm()
            zdiff=pred['P_A']['natural_z'].astype(np.float64)-base['P_A']['natural_z'].astype(np.float64)
            rows.append(dict(dataset=s['job']['dataset'],objective=s['job']['objective'],seed=s['job']['seed'],epoch=epoch,menu=menu,action_id=action,scope=scope,state_id=state_digest(value),metrics=panel_metrics(s['panels'],pred),geometry=dict(r_rms=float(rv.square().mean().sqrt()),delta_rms=float(dv.square().mean().sqrt()),cosine=float(torch.dot(rv,dv)/(rn*dn)) if rn and dn else 0.,is_zero_R=int(rn==0),is_zero_delta=int(dn==0),mean_cell_rms_zdiff=float(np.sqrt((zdiff**2).mean(axis=1)).mean()),group_rank_gradient_rms=g[scope])))
    s['pulse'].fast=None;s['pulse'].slow=None
    return rows


def epoch_packet(s,epoch):
    theta=s['layout'].flatten(s['model'].parameters());f=s['bank'].get('F');step=s['global_step'];rho=s['job']['config']['rho']
    old={};origins={};error=actual_error=(f.double()-theta.double()-s['ideal_R']);max_error=float(error.abs().max());window_bounds={}
    for h in set(min(h,step) for h in WINDOWS[s['job']['dataset']]):
        event=s['events'][step-h];old[h]=event['R']*rho**h;origins[h]=dict(step=step-h,predictions=event['predictions'],pb_objective=event['pb_objective'])
        window_bounds[h]=max_error+rho**h*event['rounding_max_abs']
    actual=f.double()-theta.double()
    return dict(epoch=epoch,step=step,online_fp32=theta.cpu(),F_fp32=f.detach().cpu().clone(),R_fp64=actual.cpu(),old_R_fp64_by_effective_h=old,origins=origins,q=copy.deepcopy(s['q']),online_predictions=copy.deepcopy(s['last_online']),pb_objective=_pb_objective(s),rounding_rms=float(actual_error.square().mean().sqrt()),rounding_max_abs=max_error,rounding_window_max_abs_bounds=window_bounds,buffers=cpu_copy(dict(s['model'].named_buffers())))


def rounding_observation(s):
    from .pairs import exact_transition_table,transition_metrics
    from .block import _view
    theta=s['layout'].flatten(s['model'].parameters());fixed=s['bank'].get('F')
    ideal=(theta.double()+s['ideal_R']).float()
    s['layout'].copy_to(s['scratch'],fixed);actual=predictions(s['scratch'],s['panels'])
    s['layout'].copy_to(s['scratch'],ideal);reconstructed=predictions(s['scratch'],s['panels'])
    rows=[]
    for population,panel in s['panels'].items():
        for d in range(3):
            for t in range(2):
                y,p,z=_view(panel,actual[population],d,t,population);_,q,w=_view(panel,reconstructed[population],d,t,population)
                table=exact_transition_table(y,p,q)
                rows.append(dict(population=population,domain=d,task=t,reference='actual_FP32_F',changed='FP32_cast_online_plus_ideal_FP64_R',probability_max_abs=float(np.max(np.abs(q.astype('float64')-p.astype('float64')))),probability_rms=float(np.sqrt(np.mean((q.astype('float64')-p.astype('float64'))**2))),logit_max_abs=float(np.max(np.abs(w.astype('float64')-z.astype('float64')))),counts=table.tolist(),**transition_metrics(table)))
    s['layout'].copy_to(s['scratch'],fixed)
    return rows


def advance_epoch(s):
    j=s['job'];c=j['config'];epoch=s['epoch']+1;source.restore_rng(s['rng'],s['device']);s['model'].train();start=time.perf_counter()
    order=torch.randperm(len(s['bundle'].train_y),generator=s['loader']).numpy();loss=clean=0.
    for offset in range(0,len(order),c['batch_size']):
        ids=order[offset:offset+c['batch_size']];x=torch.as_tensor(np.array(s['bundle'].train_x[ids]),device=s['device']).T;y=torch.as_tensor(np.array(s['bundle'].train_y[ids]),device=s['device'])
        result=train_batch(s,x,y);loss+=result['loss']*len(ids);clean+=result['clean_loss']*len(ids)
    rng=source.rng_state(s['device']);theta=s['layout'].flatten(s['model'].parameters());s['layout'].copy_to(s['scratch'],theta)
    packet=epoch_packet(s,epoch);validation={};hashes={}
    for kind,vector in (('O',theta),('F',s['bank'].get('F'))):
        s['layout'].copy_to(s['scratch'],vector)
        values,_,_=full_prediction(s['scratch'],s['bundle'],c['eval_batch_size']);validation[kind]=values;hashes[kind]=state_digest(vector)
        if values['macro_auc']>s['best_scores'][kind]:
            s['best_scores'][kind]=values['macro_auc'];s['selected'][kind]=dict(epoch=epoch,step=s['global_step'],state=vector.detach().cpu().clone(),state_hash=hashes[kind],validation=values)
            if kind=='F': s['selected']['OatF']=dict(epoch=epoch,step=s['global_step'],state=theta.detach().cpu().clone(),state_hash=hashes['O'],validation=validation['O']);s['selected_packet']=packet
    if epoch in ANCHORS:
        key=f'E{epoch}';s['pending_packets'][key]=packet;s['packet_refs'][key]=dict(epoch=epoch,step=s['global_step'])
    rounding_audit=rounding_observation(s) if epoch in ANCHORS else []
    pulse=pulse_observations(s,packet) if epoch%5==0 else []
    if s['matched'].active and s['matched'].end_epoch==epoch: s['matched_done'][epoch]=cpu_copy(s['matched'].finish(s['global_step']));s['matched'].fast=None;s['matched'].slow=None
    if (epoch+1)%5==0 and epoch<50: s['pulse'].begin(s['bank'].get('F'),epoch+1,s['global_step'])
    for origin,users in s['history_users'].items():
        if origin in s['events'] and max(users)<=s['global_step']: del s['events'][origin]
    source.restore_rng(rng,s['device']);s.update(epoch=epoch,complete=epoch==50,rng=rng,commit_index=epoch)
    elapsed=time.perf_counter()-start;s['cost']['wall_seconds']=s['cost'].get('wall_seconds',0)+elapsed
    record=dict(job_id=j['job_id'],dataset=j['dataset'],objective=j['objective'],seed=j['seed'],epoch=epoch,global_step=s['global_step'],training_loss=loss/len(order),clean_loss=clean/len(order),validation=validation,state_hashes=hashes,order_hash=hashlib.sha256(s['bundle'].train_row_ids[order].tobytes()).hexdigest(),pulse=pulse,rounding_rms=packet['rounding_rms'],rounding_max_abs=packet['rounding_max_abs'],rounding_window_max_abs_bounds=packet['rounding_window_max_abs_bounds'],rounding_audit=rounding_audit,cost=dict(wall_seconds=elapsed,real_adam_steps=c['steps_per_epoch']))
    return record


def export_session(s):
    keys=('job','epoch','global_step','complete','rng','best_scores','selected','selected_packet','events','history_users','schedule','q','last_online','ideal_R','matched_done','source_packets','packet_refs','cost','init_hash','commit_index')
    result={k:s[k] for k in keys if k in s}
    result.update(online=s['model'].state_dict(),optimizer=s['optimizer'].state_dict(),bank=s['bank'].state_dict(),pulse=s['pulse'].state_dict(),matched=s['matched'].state_dict(),loader=s['loader'].get_state())
    return cpu_copy(result)


def restore_session(job,bundle,device,panels,state):
    s=create_session(job,bundle,device,panels);s['model'].load_state_dict(state['online']);s['optimizer'].load_state_dict(state['optimizer'])
    s['bank'].load_state_dict(state['bank'],device);s['pulse'].load_state_dict(state['pulse'],device);s['matched'].load_state_dict(state['matched'],device);s['loader'].set_state(state['loader'])
    for key,value in state.items():
        if key not in ('online','optimizer','bank','pulse','matched','loader'): s[key]=cpu_copy(value)
    s['ideal_R']=s['ideal_R'].to(device);source.restore_rng(s['rng'],device);return s
