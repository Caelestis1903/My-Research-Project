"""Original full-population metrics and complete, immutable predictor endpoints."""
import copy
import hashlib
import numpy as np
import torch
from experiments.aug_history_20261105.model_adapter import build_model
from experiments.aug_history_20261105.registry import ParameterLayout
from experiments.aug_history_20261105.group_registry import build_path_groups
from experiments.ema_causal_20261110.interventions import group_masks
from experiments.ema_mechanism_20261106.readouts import read_predictions,cell_metrics
from experiments.mdmt_history_information_20261103.protocol import metrics
from experiments.overnight_ab_20261103.state import state_digest
from .history_readout import expand_split
from .conditional_readout import pair_risk,mix_logits


def make_context(bundle,config,device='cpu'):
    model=build_model(bundle,config).to(device).requires_grad_(False)
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in model.modules()):
        raise ValueError('Source buffer/training structure changed')
    return dict(scratch=model,layout=ParameterLayout.from_model(model),masks=group_masks(build_path_groups(model,bundle)),device=str(device))


def sigmoid(z):
    return torch.sigmoid(torch.from_numpy(np.asarray(z,dtype=np.float32))).numpy()


def combine_predictions(recent,old,weights,domains,mode):
    z=mix_logits(recent['z'],old['z'],domains,weights)
    w=np.asarray(weights,dtype=np.float32)
    if w.ndim:w=w.reshape(3,2)[np.asarray(domains,dtype=np.int64)]
    p=((1-w)*recent['p']+w*old['p']).astype(np.float32) if mode=='probability' else sigmoid(z)
    return dict(p=p,z=z)


@torch.no_grad()
def _vector_prediction(vector,buffers,x,context):
    context['layout'].copy_to(context['scratch'],vector.to(context['device']),buffers)
    v=read_predictions(context['scratch'],x)
    return v.p,v.z


@torch.no_grad()
def predict_batch(packet,x,context):
    if packet['kind']=='single':
        return _vector_prediction(packet['vector'],packet['buffers'],x,context)
    recent,old=expand_split(packet,context['masks']['S']);weights=packet.get('weights',packet['a'])
    w=torch.as_tensor(weights,device=x.device,dtype=torch.float32)
    if packet['mode']=='parameter':
        if w.ndim==0 or bool((w==w.flatten()[0]).all()):
            alpha=float(w.flatten()[0]);vector=((1-alpha)*recent.double()+alpha*old.double()).float()
            return _vector_prediction(vector,packet['buffers'],x,context)
        p=torch.empty((x.shape[1],2),device=x.device);z=torch.empty_like(p)
        for domain in range(3):
            mask=x[-1]==domain
            if not bool(mask.any()):continue
            for task in range(2):
                alpha=float(w.reshape(3,2)[domain,task]);vector=((1-alpha)*recent.double()+alpha*old.double()).float()
                q,v=_vector_prediction(vector,packet['buffers'],x[:,mask],context)
                p[mask,task]=q[:,task];z[mask,task]=v[:,task]
        return p,z
    pr,zr=_vector_prediction(recent,packet['buffers'],x,context)
    po,zo=_vector_prediction(old,packet['buffers'],x,context)
    combined=combine_predictions(dict(p=pr.cpu().numpy(),z=zr.cpu().numpy()),dict(p=po.cpu().numpy(),z=zo.cpu().numpy()),weights,x[-1].cpu().numpy(),packet['mode'])
    return torch.from_numpy(combined['p']).to(x.device),torch.from_numpy(combined['z']).to(x.device)


def predict_arrays(packet,xall,batch_size,context):
    p=[];z=[]
    with torch.no_grad():
        for start in range(0,len(xall),batch_size):
            x=torch.as_tensor(np.array(xall[start:start+batch_size]),device=context['device']).T
            pp,zz=predict_batch(packet,x,context);p.append(pp.cpu().numpy());z.append(zz.cpu().numpy())
    return dict(p=np.concatenate(p).astype(np.float32),z=np.concatenate(z).astype(np.float32))


def materialize_packet(packet,objects):
    if packet['kind']!='predictor': return packet
    base=objects[packet['base_ref']]
    return dict(base,**{k:v for k,v in packet.items() if k!='kind'})


def evaluate_full_population(packet,bundle,split='val',eval_batch_size=40960,context=None,freeze=None,return_predictions=False,objects=None):
    if packet['kind']=='predictor': packet=materialize_packet(packet,objects)
    if split=='test':
        if freeze is None or packet.get('state_id') not in freeze.get('state_ids',[]): raise ValueError('Test endpoint is not globally frozen')
    if split not in ('val','test'):raise ValueError('Full evaluation split must be val/test')
    context=context or make_context(bundle,packet['config'])
    x=getattr(bundle,split+'_x');y=getattr(bundle,split+'_y');pred=predict_arrays(packet,x,eval_batch_size,context)
    values=metrics(y,pred['p'],x[:,-1])
    return (values,pred) if return_predictions else values


def predict_panels(packet,panels,context,batch_size=40960):
    result={}
    for name,panel in panels.items():
        x=np.asarray(panel['natural_x']);valid=np.asarray(panel.get('valid_mask',np.ones(x.shape[:2],bool)))
        pp=np.zeros((*x.shape[:2],2),np.float32);zz=np.zeros_like(pp)
        if valid.any():
            natural=predict_arrays(packet,x[valid],batch_size,context);pp[valid]=natural['p'];zz[valid]=natural['z']
        out=dict(natural_p=pp,natural_z=zz)
        if name!='V_mech':
            rank=panel['rank_x'];pred=predict_arrays(packet,rank.reshape(-1,rank.shape[-1]),batch_size,context)
            p=pred['p'].reshape(6,rank.shape[1],2);z=pred['z'].reshape(p.shape)
            out.update(rank_p=np.stack([p[c,:,c%2] for c in range(6)]),rank_z=np.stack([z[c,:,c%2] for c in range(6)]))
        result[name]=out
    return result


def panel_metrics(panel,pred,pairs=None):
    cells=[]
    for c in range(6):
        d,t=divmod(c,2);valid=np.asarray(panel.get('valid_mask',np.ones(panel['natural_y'].shape[:2],bool)))[d]
        v=cell_metrics(panel['natural_y'][d,valid,t],pred['natural_p'][d,valid,t])
        if panel['panel_id']!='V_mech':
            v['auc']=cell_metrics(panel['rank_y'][c],pred['rank_p'][c])['auc']
            v['risk']=pair_risk(pred['rank_z'][c],pairs['tables'][c]) if pairs else None
        cells.append(v)
    return dict(cells=cells,macro_auc=float(np.mean([c['auc'] for c in cells])) if all(c['auc'] is not None for c in cells) else None,
                macro_logloss=float(np.mean([c['logloss'] for c in cells])) if all(c['logloss'] is not None for c in cells) else None,
                macro_risk=float(np.mean([c['risk'] for c in cells])) if all(c.get('risk') is not None for c in cells) else None)


def update_earliest_best(selected,readout,packet_id,epoch,values,spec):
    if readout not in selected or values['macro_auc']>selected[readout]['validation']['macro_auc']:
        selected[readout]=dict(packet_id=packet_id,epoch=int(epoch),validation=copy.deepcopy(values),spec=copy.deepcopy(spec))
        return True
    return False


def freeze_endpoints(selected,objects):
    result=[]
    for key,value in selected.items():
        packet=objects[value['packet_id']]
        result.append(dict(readout_id=key,packet_id=value['packet_id'],state_id=packet['state_id'],epoch=value['epoch'],validation=value['validation'],spec=value['spec']))
    return result


def test_frozen(packet,bundle,freeze,context=None,eval_batch_size=40960,objects=None):
    return evaluate_full_population(packet,bundle,'test',eval_batch_size,context,freeze,objects=objects)


def prediction_hash(prediction):
    h=hashlib.sha256()
    for key,value in sorted(prediction.items()):h.update(key.encode());h.update(np.asarray(value).tobytes())
    return h.hexdigest()
