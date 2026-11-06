"""Fixed histories and one-epoch pulses; neither updates the online model."""
import math
import torch

class FixedShadowBank:
    def __init__(self,theta0,rho_by_physical_id,aliases):
        self.rho=dict(rho_by_physical_id);self.aliases=dict(aliases)
        self.phi={key:theta0.detach().clone() for key in self.rho};self.updates=0
    def get(self,readout_id): return self.phi[self.aliases.get(readout_id,readout_id)]
    @torch.no_grad()
    def after_adam(self,theta_after,step):
        if step!=self.updates+1: raise ValueError('Nonconsecutive EMA clock')
        for key,phi in self.phi.items(): phi.mul_(self.rho[key]).add_(theta_after,alpha=1-self.rho[key])
        self.updates=step
    def state_dict(self): return dict(phi=self.phi,rho=self.rho,aliases=self.aliases,updates=self.updates)
    def load_state_dict(self,value,device):
        if value['rho']!=self.rho or value['aliases']!=self.aliases: raise ValueError('Shadow identity changed')
        self.phi={k:v.to(device).clone() for k,v in value['phi'].items()};self.updates=value['updates']

class PulseWindow:
    def __init__(self,rho_F):
        self.rho_F=rho_F;self.active=False;self.fast=None;self.slow=None
        self.start_step=0;self.last_step=0;self.end_epoch=None
    def begin(self,phi_f,end_epoch,start_step):
        if self.active: raise ValueError('Pulse window already active')
        self.fast=phi_f.detach().clone();self.slow=phi_f.detach().clone()
        self.active=True;self.end_epoch=end_epoch;self.start_step=start_step;self.last_step=start_step
    @torch.no_grad()
    def after_adam(self,theta_after,step):
        if not self.active: return
        if step!=self.last_step+1: raise ValueError('Nonconsecutive pulse clock')
        for value,rho in ((self.fast,self.rho_F**2),(self.slow,math.sqrt(self.rho_F))):
            value.mul_(rho).add_(theta_after,alpha=1-rho)
        self.last_step=step
    def finish(self,expected_end_step):
        if not self.active or self.last_step!=expected_end_step: raise ValueError('Incomplete pulse window')
        self.active=False
        return dict(fast=self.fast,slow=self.slow,start_step=self.start_step,end_step=self.last_step,end_epoch=self.end_epoch)
    def state_dict(self): return vars(self).copy()
    def load_state_dict(self,value,device):
        self.__dict__.update({k:v.to(device).clone() if torch.is_tensor(v) else v for k,v in value.items()})
