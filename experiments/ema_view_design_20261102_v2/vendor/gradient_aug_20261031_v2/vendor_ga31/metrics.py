"""Real-input metrics; pooled labels stay separated by task."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def metrics(y,pred,domain):
    y=np.asarray(y); pred=np.asarray(pred,dtype=np.float64); domain=np.asarray(domain,dtype=np.int64)
    # Match probability BCE's log floor, including exact sigmoid 0/1 predictions.
    with np.errstate(divide='ignore'):
        loss=-(y*np.maximum(np.log(pred),-100)+(1-y)*np.maximum(np.log1p(-pred),-100))
    out={}
    for d in range(3):
        mask=domain==d
        for t in range(2):
            out[f'd{d}t{t}_auc']=float(roc_auc_score(y[mask,t],pred[mask,t]))
            out[f'd{d}t{t}_logloss']=float(loss[mask,t].mean())
            out[f'd{d}t{t}_count']=int(mask.sum())
            out[f'd{d}t{t}_positive']=int(y[mask,t].sum())
    for name in ('auc','logloss'):
        out['macro_'+name]=float(np.mean([out[f'd{d}t{t}_{name}'] for d in range(3) for t in range(2)]))
        out['sample_weighted_cell_'+name]=float(sum((domain==d).mean()*out[f'd{d}t{t}_{name}']/2 for d in range(3) for t in range(2)))
        for d in range(3):out[f'domain{d}_macro_{name}']=float(np.mean([out[f'd{d}t{t}_{name}'] for t in range(2)]))
        for t in range(2):out[f'task{t}_macro_{name}']=float(np.mean([out[f'd{d}t{t}_{name}'] for d in range(3)]))
    for t in range(2):
        out[f'task{t}_pooled_auc']=float(roc_auc_score(y[:,t],pred[:,t]))
        out[f'task{t}_pooled_logloss']=float(loss[:,t].mean())
    out['pooled_task_mean_auc']=float(np.mean([out[f'task{t}_pooled_auc'] for t in range(2)]))
    out['pooled_logloss']=float(loss.mean())
    out['worst_cell_auc']=min(out[f'd{d}t{t}_auc'] for d in range(3) for t in range(2))
    return out


def evaluate(model,x,y,device,batch_size=40960):
    was=model.training; model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(y),batch_size):
            bx=torch.as_tensor(x[i:i+batch_size],device=device).T
            preds.append(model(bx).cpu().numpy())
    model.train(was)
    return metrics(y,np.concatenate(preds),x[:,-1])
