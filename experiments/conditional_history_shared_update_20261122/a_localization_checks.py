"""Synthetic exact-shadow and fixed-panel wiring checks. No research data or RAM I/O."""
import copy
from itertools import product

import numpy as np
import torch

from .a_localization import GROUPS, SOURCES, alternatives, gate_rows, match_factor, observe
from .b_observer import ALL
from .b_panels import Panel
from .history_controls import history_config, make_history_optimizer
from .protocol import digest, v3
from .results import CELLS, SEEDS, localization_evidence


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = torch.nn.Parameter(torch.ones(1))
        self.expert = torch.nn.Linear(1,1,bias=False)
        self.tower = torch.nn.ModuleList(torch.nn.Linear(1,1) for _ in range(6))

    def forward(self,x):
        z = self.expert(x[:1].T)*self.shared_weight
        dom = x[-1].long()
        logits = torch.cat([t(z) for t in self.tower],dim=1)
        return torch.sigmoid(torch.stack([logits.gather(1,(dom+3*t)[:,None]).flatten() for t in range(2)],dim=1))


def check():
    v3.setup(2022)
    model = Tiny()
    ema = copy.deepcopy(model).requires_grad_(False)
    cfg = history_config('A21','movielens')
    opt = make_history_optimizer(model,cfg)
    shadow = {n:torch.zeros_like(p) for n,p in opt.shared}
    for i,selected in enumerate((0,7,12,None,17),1):
        pre = copy.deepcopy(model)
        pair = dict(w=np.array([.4]),q=np.array([.5]),index=0,selected=selected) if selected is not None else None
        raw = [torch.randn_like(p) for _,p in opt.shared] if pair else None
        for j,(n,p) in enumerate(model.named_parameters()):
            p.grad = torch.randn_like(p) if j<4 else None
            if n in shadow:
                shadow[n].mul_(.9).add_(p.grad.add(p.detach(),alpha=cfg['weight_decay']),alpha=.1)
        packet = v3.cpu_copy(dict(job={'config':cfg}, model=model.state_dict(), model_mode=True, ema=ema.state_dict(),
                        ema_mode=False, optimizer=opt.state_dict(),group_hash=digest(opt.parameter_manifest),pair=pair,
                        raw_pair=raw,mixed_grad={n:p.grad for n,p in model.named_parameters()}))
        opt.step(pair,raw)
        v3.update_ema(ema,model,cfg['rho'])
        candidates, _ = alternatives(pre,model,opt,pair)
        for n,p in opt.shared:
            s=opt.moments[n]
            denom=(s['v']/(1-.999**i)).sqrt()+1e-8
            expected=dict(pre.named_parameters())[n].addcdiv(shadow[n]/(1-.9**i),denom,value=-cfg['lr'])
            torch.testing.assert_close(candidates[n]['ALL'],expected)
            residual=sum(p.detach()-candidates[n][c] for c in CELLS)
            torch.testing.assert_close(p.detach()-candidates[n]['ALL'],residual,atol=1e-6,rtol=1e-5)
    packet['actual_post']={k:v3.state_digest(v.state_dict()) for k,v in [('model',model),('optimizer',opt),('ema',ema)]}
    x = np.column_stack((np.arange(12,dtype='float32')/12,np.repeat(np.arange(3),4))).astype('float32')
    y = np.tile(np.array([[0,0],[1,1],[1,0],[0,1]],dtype='float32'),(3,1))
    natural = [np.flatnonzero(x[:,-1]==d) for d in range(3)]
    rank = [natural[q//2] for q in range(6)]
    pairs = [np.array([(a,b) for a in ids if y[a,q%2] for b in ids if not y[b,q%2]]) for q,ids in enumerate(rank)]
    panels = {n:Panel(x,y,rank,pairs,natural) for n in ALL}
    before = v3.state_digest(packet)
    raw = observe(model,panels,packet,torch.device('cpu'))
    assert v3.state_digest(packet)==before
    assert {(r['group'],r['source'],r['scale']) for r in raw['effects']} == (
        set(product(GROUPS,SOURCES,('raw','norm_matched')))|{('S_DENSE','ALL','raw')})
    assert {r['status'] for r in raw['effects']}=={'OK'}
    for e in raw['effects']:
        if e['scale']=='norm_matched':
            assert np.isclose(e['applied_update_norm'],e['ref_update_norm'],rtol=1e-5,atol=1e-7)
    raw.update(dataset='movielens',arm='A21',history_form='A21')
    rows = [r for seed,epoch in product(SEEDS,(5,20,35,50)) for r in
            gate_rows(dict(raw,seed=seed,anchor_epoch=epoch),'synthetic')]
    evidence = localization_evidence(rows,'movielens','A21','A21',CELLS)
    assert len(rows)==80 and all(e[s]['informative'] for e in evidence.values() for s in ('raw','norm_matched'))
    assert match_factor(0,0)==1 and match_factor(0,2)==0 and match_factor(2,0) is None
    return {'status':'PASS','synthetic_steps':5,'action_rows':len(raw['effects']),
            'panel_forwards':raw['counts']['panel_forwards'],'gate_rows':len(rows),
            'scientific_training_steps':0,'test_evaluations':0}


if __name__=='__main__':
    print(check())
