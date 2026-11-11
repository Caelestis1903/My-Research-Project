"""Sparse actual-F history and stable finite two-centroid construction."""
from collections import Counter
import math
import torch


class LaggedFStore:
    def __init__(self, target_steps, windows, rho):
        self.rho=float(rho);self.windows=tuple(windows)
        self.targets={int(t):{int(h):max(0,int(t)-int(h)) for h in windows} for t in target_steps}
        self.users=Counter(origin for needs in self.targets.values() for origin in set(needs.values()))
        self.records={};self.peak_vectors=0

    def capture(self, step, fixed, put):
        if step in self.users and self.users[step]>0:
            if step in self.records: raise ValueError('Duplicate F capture')
            self.records[step]=put(dict(kind='history',vector=fixed.detach().cpu().clone(),step=int(step),refs=[]))
            self.peak_vectors=max(self.peak_vectors,len(self.records))

    def consume(self, step, objects):
        origins=self.targets[int(step)]
        result={h:objects[self.records[origin]]['vector'] for h,origin in origins.items()}
        for origin in set(origins.values()):
            self.users[origin]-=1
            if self.users[origin]==0: del self.records[origin]
        return result

    def state_dict(self):
        return dict(rho=self.rho,windows=self.windows,targets=self.targets,users=dict(self.users),
                    records=dict(self.records),peak_vectors=self.peak_vectors)

    @classmethod
    def from_state(cls,state):
        value=cls([],state['windows'],state['rho'])
        value.targets={int(k):{int(h):int(v) for h,v in x.items()} for k,x in state['targets'].items()}
        value.users=Counter({int(k):int(v) for k,v in state['users'].items()})
        value.records={int(k):v for k,v in state['records'].items()};value.peak_vectors=state['peak_vectors']
        return value

    def schedule(self):
        live=set();counts=self.users.copy();peak=0;events=[]
        for step in sorted(set(counts)|set(self.targets)):
            if counts[step]>0: live.add(step)
            peak=max(peak,len(live))
            for origin in set(self.targets.get(step,{}).values()):
                counts[origin]-=1
                if not counts[origin]: live.remove(origin)
            events.append(dict(step=step,live=len(live)))
        return dict(peak_vectors=peak,capture_steps=sorted(self.users),events=events)


def split_history(fixed, old, rho, h, shared_mask=None):
    if h<=0: raise ValueError('History split is defined only after an optimizer step')
    f=fixed.detach().cpu().double();o=old.detach().cpu().double()
    a=math.exp(h*math.log(rho));den=-math.expm1(h*math.log(rho));r=(f-a*o)/den
    ideal=(den*r+a*o-f).abs().max()
    if shared_mask is not None:
        mask=shared_mask.cpu();r[~mask]=f[~mask];o[~mask]=f[~mask]
    recent=r.float();older=o.float()
    residual=((1-a)*recent.double()+a*older.double()-f).abs().max()
    return recent,older,dict(h=int(h),a=a,fp64_barycenter_max_abs=float(ideal),fp32_barycenter_max_abs=float(residual))


def shared_only(fixed, online, shared_mask):
    result=online.detach().cpu().clone();m=shared_mask.cpu();result[m]=fixed.detach().cpu()[m];return result


def split_packet(recent,old,fixed,mask,scope,**metadata):
    if scope=='S':
        m=mask.cpu()
        return dict(kind='split',scope=scope,recent=recent[m].clone(),old=old[m].clone(),
                    nonshared=fixed.detach().cpu()[~m].clone(),**metadata)
    return dict(kind='split',scope='all',recent=recent,old=old,**metadata)


def expand_split(packet,mask):
    if packet['scope']=='all': return packet['recent'],packet['old']
    m=mask.cpu();recent=torch.empty(len(m),dtype=torch.float32);old=torch.empty_like(recent)
    recent[m]=packet['recent'];old[m]=packet['old'];recent[~m]=packet['nonshared'];old[~m]=packet['nonshared']
    return recent,old
