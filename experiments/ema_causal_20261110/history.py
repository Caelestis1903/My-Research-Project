"""RAM-only sparse history and exact, all-step score averages."""
from collections import Counter, defaultdict
import copy
import numpy as np
import torch


def _clone(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return {k:_clone(v) for k,v in value.items()}
    return copy.deepcopy(value)


class ScoreEMA:
    def __init__(self, rho, initial):
        self.rho = float(rho)
        self.value = {k:np.asarray(v,dtype=np.float64).copy() for k,v in initial.items()}
        self.step = 0

    def update(self, predictions):
        if predictions.keys() != self.value.keys(): raise ValueError('Score fields changed')
        for key, value in self.value.items():
            incoming=np.asarray(predictions[key],dtype=np.float64)
            if value.shape != incoming.shape: raise ValueError('Score population changed')
            value *= self.rho
            value += (1-self.rho)*incoming
        self.step += 1
        return self.value


class SparseHistory:
    def __init__(self, target_steps, windows, rho):
        self.rho=float(rho);self.windows=tuple(windows)
        self.needs={int(t):sorted({min(int(h),int(t)) for h in windows}) for t in target_steps}
        self.references=Counter(t-h for t,hs in self.needs.items() for h in hs if t-h>0)
        self.capture_steps=set(self.references);self.records={};self.initial=None
        self.peak_vectors=0

    def capture(self, step, R, predictions=None, stats=None):
        if step==0:
            if bool(torch.count_nonzero(R)): raise ValueError('R0 must be zero')
            self.initial=dict(predictions=_clone(predictions),stats=_clone(stats))
        elif step in self.capture_steps:
            if step in self.records: raise ValueError('History source captured twice')
            self.records[step]=dict(R=R.detach().double().clone(),predictions=_clone(predictions),stats=_clone(stats))
            self.peak_vectors=max(self.peak_vectors,len(self.records))

    def consume(self, step):
        result={}
        for h in self.needs[step]:
            origin=step-h
            item=self.initial if origin==0 else self.records[origin]
            if item is None: raise ValueError('Missing initialization predictions')
            result[h]=dict(old_R=None if origin==0 else self.rho**h*item['R'],
                origin_step=origin,rho_power=self.rho**h,predictions=item['predictions'],stats=item['stats'])
            if origin:
                self.references[origin]-=1
                if self.references[origin]==0: del self.records[origin]
        return result

    def schedule(self):
        live=set();peak=0;refs=self.references.copy();events=[]
        for step in sorted(set(self.capture_steps)|set(self.needs)):
            if step in self.capture_steps: live.add(step)
            peak=max(peak,len(live))
            for h in self.needs.get(step,[]):
                origin=step-h
                events.append(dict(target_step=step,h_eff=h,origin_step=origin,rho_power=self.rho**h))
                if origin:
                    refs[origin]-=1
                    if not refs[origin]: live.remove(origin)
        return dict(events=events,capture_steps=sorted(self.capture_steps),peak_vectors=peak)


def update_ideal_R(ideal_R, before, after, rho):
    return rho*ideal_R-rho*(after.double()-before.double())


def rounding_audit(online, fixed, ideal_R):
    actual=fixed.double()-online.double();error=actual-ideal_R
    return dict(max_abs=float(error.abs().max()),rms=float(error.square().mean().sqrt()),
                relative_l2=float(error.norm()/actual.norm().clamp_min(1e-12)))
