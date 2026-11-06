"""Original probabilities and raw logits from the same read-only forward."""
from dataclasses import dataclass
import numpy as np
import torch
from experiments.mdmt_history_information_20261103.protocol import metrics
from sklearn.metrics import roc_auc_score as auc_score

def log_loss(y,p):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y)
    with np.errstate(divide="ignore"):
        return -(y*np.maximum(np.log(p),-100)+(1-y)*np.maximum(np.log1p(-p),-100)).mean()

@dataclass
class PredictionBatch:
    p: object
    z: object


def read_predictions(scratch,x,*,params=None,full_functional_graph=False):
    outputs={};handles=[];was=scratch.training;scratch.eval()
    try:
        if full_functional_graph:
            captured={}
            handles.append(scratch.skip_conn.register_forward_hook(lambda m,i,o:captured.__setitem__('skip',o)))
            handles.append(scratch.star_mlp.register_forward_hook(lambda m,i,o:captured.__setitem__('star',o)))
            for gate in scratch.gate:
                handles.append(gate.register_forward_pre_hook(lambda m,i:(captured['star']+captured['skip'],)))
        for index,tower in enumerate(scratch.tower):
            handles.append(tower.register_forward_hook(lambda m,i,o,index=index:outputs.__setitem__(index,o.squeeze(1))))
        p=scratch(x) if params is None else torch.func.functional_call(scratch,params,(x,))
        z=[]
        for task in range(2):
            value=torch.zeros(x.shape[1],device=x.device,dtype=p.dtype)
            for domain in range(3): value=torch.where(x[-1]==domain,outputs[3*task+domain],value)
            z.append(value)
        return PredictionBatch(p,torch.stack(z,1))
    finally:
        for hook in handles: hook.remove()
        scratch.train(was)


@torch.no_grad()
def group_hybrid(base,changed,group_index,group_id,out):
    out.copy_(base);mask=group_index==group_id;out[mask]=changed[mask]


def panel_predictions(scratch,panel):
    device=next(scratch.parameters()).device
    with torch.no_grad():
        natural=np.asarray(panel['natural_x']); shape=natural.shape[:2]
        x=torch.as_tensor(np.array(natural.reshape(-1,natural.shape[-1])),device=device).T
        value=read_predictions(scratch,x)
        result=dict(natural_p=value.p.cpu().numpy().reshape(*shape,2),natural_z=value.z.cpu().numpy().reshape(*shape,2))
        if panel['panel_id']!='V_mech':
            rank=np.asarray(panel['rank_x']);x=torch.as_tensor(np.array(rank.reshape(-1,rank.shape[-1])),device=device).T
            value=read_predictions(scratch,x);p=value.p.reshape(6,rank.shape[1],2);z=value.z.reshape_as(p)
            task=torch.tensor([0,1,0,1,0,1],device=device)[:,None,None]
            result.update(rank_p=p.gather(2,task.expand(6,rank.shape[1],1)).squeeze(2).cpu().numpy(),
                          rank_z=z.gather(2,task.expand(6,rank.shape[1],1)).squeeze(2).cpu().numpy())
    return dict(p=result['natural_p'],z=result['natural_z']) if panel['panel_id']=='V_mech' else result


def evaluate_full_validation(model,bundle,panel,batch_size=40960,include_logits=True):
    device=next(model.parameters()).device; predictions=[]; natural_p=np.empty((*panel['natural_indices'].shape,2),np.float32)
    natural_z=np.empty_like(natural_p) if include_logits else None
    indices=panel['natural_indices']; was=model.training;model.eval()
    try:
        with torch.no_grad():
            for offset in range(0,len(bundle.val_y),batch_size):
                end=min(offset+batch_size,len(bundle.val_y));x=torch.as_tensor(np.array(bundle.val_x[offset:end]),device=device).T
                value=read_predictions(model,x);p=value.p.cpu().numpy();predictions.append(p)
                mask=(indices>=offset)&(indices<end);natural_p[mask]=p[indices[mask]-offset]
                if include_logits: natural_z[mask]=value.z.cpu().numpy()[indices[mask]-offset]
    finally: model.train(was)
    result=dict(p=natural_p)
    if include_logits: result['z']=natural_z
    return metrics(bundle.val_y,np.concatenate(predictions),bundle.val_x[:,-1]),result


def cell_metrics(y,p):
    if not len(y): return dict(auc=None,logloss=None,actual_count=0,missing_reason='EMPTY_PANEL')
    return dict(auc=float(auc_score(y,p)) if len(np.unique(y))==2 else None,
        logloss=float(log_loss(y,p)),actual_count=len(y),missing_reason=None if len(np.unique(y))==2 else 'NA_SINGLE_CLASS')


def pulse_rows(panel,pred,readout_id,group_id,variant,state_hash):
    rows=[];is_v=panel['panel_id']=='V_mech'
    for d in range(3):
        for t in range(2):
            c=2*d+t;support=group_id is None or group_id in ('S',f'D{d}',f'T{t}',f'DT{d}{t}')
            for side,name in ((None,'V_mech'),(0,'V_A'),(1,'V_B')) if is_v else ((None,panel['panel_id']),):
                y=np.asarray(panel['natural_y'][d,:,t]);p=pred['p'][d,:,t] if is_v else pred['natural_p'][d,:,t]
                mask=np.ones(len(y),bool) if side is None else panel['validation_row_side'][d]==side
                vals=cell_metrics(y[mask],p[mask]);risk=None
                if not is_v:
                    vals['auc']=cell_metrics(panel['rank_y'][c],pred['rank_p'][c])['auc']
                    pairs=panel['rank_pairs'][c];z=pred['rank_z'][c]
                    risk=float(np.logaddexp(0.,-(z[pairs[:,0]].astype('float64')-z[pairs[:,1]].astype('float64'))).mean())
                rows.append(dict(panel_id=name,readout_id=readout_id,physical_readout_id=readout_id,group_id=group_id,
                    variant=variant,domain=d,task=t,functional_support=support,state_hash=state_hash,rank_risk=risk,**vals))
    return rows
