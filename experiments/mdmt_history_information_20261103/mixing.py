"""Natural within-batch donors and virtual non-routing embedding interpolation."""
from dataclasses import dataclass
import hashlib
import torch
from torch.func import functional_call

@dataclass(frozen=True)
class StepContext:
    dataset: str
    seed: int
    epoch: int
    global_step: int

@dataclass
class MixBatch:
    anchor_idx: torch.Tensor
    donor_idx: torch.Tensor
    target_task: int
    domain: int
    support_level: torch.Tensor
    tau: float
    index_hash: str

def generator(context,stream,domain,task):
    if isinstance(context,dict): context=StepContext(**{k:context[k] for k in StepContext.__annotations__})
    key=f'MDMT1103|{context.dataset}|{context.seed}|{context.epoch}|{context.global_step}|{stream}|{domain}|{task}'
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
    return torch.Generator().manual_seed(seed)

def build_mix_batches(x,y,row_ids,context,mode,tau,max_anchors=256,domains=None):
    labels=y.detach().cpu(); domain=x[-1].detach().cpu(); rows=torch.as_tensor(row_ids).cpu()
    result=[]
    for d in (range(3) if domains is None else domains):
        candidates=torch.where(domain==d)[0]; n=len(candidates)
        if not n: continue
        anchors=candidates[torch.randperm(n,generator=generator(context,'anchor',d,-1))[:min(max_anchors,n)]]
        for task in range(2):
            gen=generator(context,'donor',d,task); donors=[];levels=[]
            # The candidate pools are shared by all anchors with a joint label.
            pools={(a,b):candidates[(labels[candidates,task]==a)&(labels[candidates,1-task]==b)]
                   for a in (0,1) for b in (0,1)}
            for i in anchors:
                a,b=map(int,labels[i,[task,1-task]])
                pool=pools[a,1-b]; pool=pool[rows[pool]!=rows[i]]
                level=1
                if mode=='unconditional' or not len(pool):
                    pool=torch.cat([pools[a,0],pools[a,1]]);pool=pool[rows[pool]!=rows[i]];level=2
                if len(pool): donor=pool[torch.randint(len(pool),(1,),generator=gen)[0]]
                else: donor=i;level=3
                donors.append(donor);levels.append(level)
            donors=torch.stack(donors)
            h=hashlib.sha256(torch.stack([anchors,donors]).numpy().tobytes()).hexdigest()
            result.append(MixBatch(anchors,donors,task,d,torch.tensor(levels),tau,h))
    return result

def model_forward(model,x,functional_state=None):
    return model(x) if functional_state is None else functional_call(model,functional_state,(x,),strict=True)

def forward_mixed(model,x,mix_batch,functional_state=None):
    mb=mix_batch;m=len(mb.anchor_idx)
    indices=torch.cat([mb.anchor_idx,mb.donor_idx]).to(x.device)
    def mix(module,inputs,output):
        result=output.clone()
        for name,(lo,hi) in model.field_slices.items():
            if name!='domain_indicator': result[:m,lo:hi]=(1-mb.tau)*output[:m,lo:hi]+mb.tau*output[m:,lo:hi]
        return result
    hook=model.embedding.register_forward_hook(mix)
    try: return model_forward(model,x[:,indices],functional_state)[:m]
    finally: hook.remove()
