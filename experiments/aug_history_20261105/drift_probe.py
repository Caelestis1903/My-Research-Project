"""Training-only old-online/old-teacher input drift on shared absolute delta coordinates."""
import hashlib
from collections import OrderedDict
import numpy as np
import torch


def input_drift(online,teacher,x_probe,fields):
    n=x_probe.shape[1]
    delta=torch.zeros(n,fields,16,device=x_probe.device,dtype=next(online.parameters()).dtype,requires_grad=True)
    current=online.forward_logits(x_probe,params=OrderedDict((k,p.detach()) for k,p in online.named_parameters()),
        delta=delta,probe_input_full_graph=True)
    previous=teacher.forward_logits(x_probe,params=OrderedDict((k,p.detach()) for k,p in teacher.named_parameters()),
        delta=delta,probe_input_full_graph=True)
    gradients=[torch.autograd.grad((current[:,t]-previous[:,t]).sum(),delta,retain_graph=t==0)[0] for t in range(2)]
    return torch.stack(gradients,dim=1).detach()


class DriftProbeSampler:
    def __init__(self,bundle):
        self.bundle=bundle
        self.pool=np.flatnonzero(bundle.train_x[:,-1]==0)

    def sample(self,excluded_row_ids,seed,step):
        b=self.bundle
        key=f'AH1105|{b.dataset}|{seed}|{step}|C_PROBE|0|0'
        value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
        pool=self.pool[~np.isin(b.train_row_ids[self.pool],np.asarray(excluded_row_ids))]
        selected=np.random.default_rng(value).choice(pool,size=min(128,len(pool)),replace=False)
        ids=np.asarray(b.train_row_ids[selected])
        return np.asarray(b.train_x[selected]),dict(key=key,seed=value,rows=len(selected),
            row_ids_hash=hashlib.sha256(ids.tobytes()).hexdigest(),reason='OK' if len(selected) else 'EMPTY_TRAIN_D0')
