"""Actual within-domain joint-label boundaries, fixed target mass and sampled reads."""
import hashlib
import numpy as np
import torch

CONDITIONS=[(d,t,a,b) for d in range(3) for t in range(2)
            for a in range(4) for b in range(4) if ((a>>(1-t))&1)==1 and ((b>>(1-t))&1)==0]

def rng(context,stream):
    key='|'.join(str(context[k]) for k in ('dataset','seed','epoch','global_step'))+'|'+stream
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))

def boundaries(domain,y,demand,dynamic):
    domain=np.asarray(domain); y=np.asarray(y)
    assert y.shape==(len(domain),2) and np.isin(y,(0,1)).all()
    labels=(2*y[:,0]+y[:,1]).astype(int)
    groups={(d,a):np.flatnonzero((domain==d)&(labels==a)) for d in range(3) for a in range(4)}
    active=[]; weights=[]
    for i,(d,t,a,b) in enumerate(CONDITIONS):
        pos,neg=groups[d,a],groups[d,b]
        if not len(pos) or not len(neg): continue
        mask=domain==d; n=int(mask.sum()); npos=int(y[mask,t].sum()); nneg=n-npos
        active.append(i); weights.append(n/(len(y)*2)*len(pos)/npos*len(neg)/nneg)
    w=np.asarray(weights,np.float64)
    if not active:return active,w,w,groups
    assert len(active)<=24
    z=w.sum(); D=np.asarray([demand.get(i,.5) for i in active]) if dynamic else np.full(len(active),.5)
    q=.25/len(active)+.25*w/z+.5*w*(.05+D)/np.sum(w*(.05+D))
    assert np.isclose(q.sum(),1) and np.all(q>0) and z<=1+1e-12
    assert np.max(w/q)<=4+1e-12 and np.max(.01/q)<=.96+1e-12
    return active,w,q,groups

def sample(domain,y,demand,dynamic,context,row_ids):
    active,w,q,groups=boundaries(domain,y,demand,dynamic)
    if not active:return None
    index=min(int(np.searchsorted(np.cumsum(q),rng(context,'condition').random(),side='right')),len(active)-1)
    selected=active[index];d,t,a,b=CONDITIONS[selected];gen=rng(context,'pair_indices')
    pos=gen.choice(groups[d,a],2048,replace=True); neg=gen.choice(groups[d,b],2048,replace=True)
    ids=np.concatenate((np.asarray(row_ids)[pos],np.asarray(row_ids)[neg]))
    return dict(active=active,w=w,q=q,index=index,selected=selected,task=t,pos=pos,neg=neg,
                unique_pair_rows=int(len(np.unique(ids))),pair_index_hash=hashlib.sha256(ids.tobytes()).hexdigest())

def forward_logits(model,x):
    captured={}; handles=[]
    for i,tower in enumerate(model.tower):
        handles.append(tower.register_forward_hook(lambda module,args,out,i=i:captured.__setitem__(i,out.squeeze(1))))
    try:
        probability=model(x)
        logits=torch.stack([sum(torch.where(x[-1]==d,captured[t*3+d],torch.zeros_like(captured[t*3+d]))
                                for d in range(3)) for t in range(2)],dim=1)
        return probability,logits
    finally:
        for h in handles:h.remove()
