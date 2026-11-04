"""One probability convention for published and M3 model endpoints."""
import numpy as np
import torch
from experiments.mdmt_history_information_20261103.protocol import metrics

def evaluate_model(model,bundle,split,method,batch_size):
    prefix='val' if split in ('val','validation') else split
    x=getattr(bundle,prefix+'_x'); y=getattr(bundle,prefix+'_y')
    device=next(model.parameters()).device
    published=method.upper() in ('HYPERGATE','AUTOIFS')
    was=model.training; model.eval(); predictions=[]
    try:
        with torch.no_grad():
            for offset in range(0,len(y),batch_size):
                bx=torch.as_tensor(np.asarray(x[offset:offset+batch_size]),device=device)
                output=model(bx) if published else model(bx.T)
                if isinstance(output,dict): output=output['logits']
                if isinstance(output,tuple): output=output[0]
                if published: output=output.float().sigmoid()
                predictions.append(output.detach().cpu().numpy())
    finally: model.train(was)
    return metrics(y,np.concatenate(predictions),x[:,-1])

