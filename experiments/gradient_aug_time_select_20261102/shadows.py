"""Non-feedback fixed, delayed, epoch-average, adaptive and replay states."""
import copy
from collections import deque
import numpy as np
import torch
from .protocol import update_shadow


class ShadowController:
    def __init__(self, model, c, replay=None):
        self.c=c; self.norms=deque(maxlen=100); self.rho=c['rho_fixed']; self.cv=None
        self.swa_count=0; self.replay=replay; self.rhos=[]
        names=[]
        if c['ema_enabled']:names.append('fixed_ema')
        if c['delayed_ema_enabled']:names.append('delayed5')
        if c['swa_enabled']:names.append('swa20')
        if c['adaptive_ema_enabled']:names.append('adaptive_ema')
        if replay is not None and c['replay_enabled']:names.append('replay_rho')
        assert len(names)<=4
        self.shadows={k:copy.deepcopy(model).requires_grad_(False) for k in names}

    @torch.no_grad()
    def step(self,model,gradnorm,step,epoch):
        c=self.c; updates=0
        if c['adaptive_ema_enabled']:
            self.norms.append(float(gradnorm))
            if step%100==0:
                x=np.asarray(self.norms);self.cv=float(x.std(ddof=0)/(abs(x.mean())+1e-12))
                self.rho=c['rho_min']+(c['rho_max']-c['rho_min'])*self.cv/(self.cv+c['cv_c'])
            self.rhos.append(self.rho)
        for kind,shadow in self.shadows.items():
            if kind=='swa20' or (kind=='delayed5' and epoch<=5):continue
            rho=(self.rho if kind=='adaptive_ema' else float(self.replay[step-1]) if kind=='replay_rho' else c['rho_fixed'])
            update_shadow(shadow,model,rho);updates+=1
        return updates

    @torch.no_grad()
    def epoch_end(self,model,epoch):
        updates=0
        if 'delayed5' in self.shadows and epoch==5:
            self.shadows['delayed5'].load_state_dict(model.state_dict());updates+=1
        if 'swa20' in self.shadows and epoch>=31:
            self.swa_count+=1;shadow=self.shadows['swa20']
            if self.swa_count==1:shadow.load_state_dict(model.state_dict())
            else:
                for a,b in zip(shadow.parameters(),model.parameters()):a.add_(b-a,alpha=1/self.swa_count)
                for a,b in zip(shadow.buffers(),model.buffers()):a.copy_(b)
            updates+=1
        return updates

    def valid(self,epoch,replaying=False):
        if replaying:return ['replay_rho']
        return ['online']+[k for k in self.shadows if k!='replay_rho' and
            not (k=='delayed5' and epoch<=5) and not (k=='swa20' and epoch<31)]

    def state(self):
        return dict(norms=list(self.norms),rho=self.rho,cv=self.cv,swa_count=self.swa_count,rhos=list(self.rhos),
            replay=None if self.replay is None else list(self.replay))

    def restore(self,state):
        self.norms=deque(state['norms'],maxlen=100);self.rho=state['rho'];self.cv=state['cv']
        self.swa_count=state['swa_count'];self.rhos=list(state['rhos']);self.replay=state['replay']


def replay_sequence(rhos):
    values=np.asarray(rhos,dtype=np.float64)
    shuffled=values[np.random.default_rng(1729).permutation(len(values))]
    assert np.array_equal(np.sort(values),np.sort(shuffled))
    return shuffled.tolist()
