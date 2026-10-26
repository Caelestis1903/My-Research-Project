"""Minimal numerical/permission checks for conditional formation and readout."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

from .functions import PilotBank,FunctionBank,freeze
from .readers import ReaderGroup,Normalizer,stable_mixture_bce,solve_simplex,select_whole_policy
from .state import state_hash,save_checkpoint,load_checkpoint
from .pipeline import shuffled_weights,_save,_restore


def run(device='cpu',artifact_root='/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0'):
    torch.manual_seed(42);np_rng=np.random.default_rng(42)
    pilot=freeze(PilotBank().to(device));bank=FunctionBank(pilot).to(device)
    h=torch.randn(18,256,device=device,requires_grad=True);z=torch.randn(18,2,device=device,requires_grad=True)
    y=torch.randint(2,(18,2),device=device).float();d=torch.arange(18,device=device)%3
    w=torch.as_tensor(np_rng.uniform(.1,1.9,(18,2,1)),device=device,dtype=torch.float32);w=torch.cat((w,2-w),-1)
    actual=bank.formation_loss(h,z,y,d,w)
    reference=0.
    for s in range(3):
        mask=d==s
        for q in range(2):
            for k,t in enumerate(t for t in range(3) if t!=s):
                reference=reference+(w[mask,q,k]*F.binary_cross_entropy_with_logits(z[mask,q].detach()+bank.head(s,t,q)(h[mask].detach()),y[mask,q],reduction='none')).sum()/(4*len(h))
    err=float(abs(actual.detach()-reference.detach()));assert err<=1e-6
    actual.backward();assert h.grad is None and z.grad is None and all(p.grad is None for p in pilot.parameters())
    full_grads={k:p.grad.detach().clone() for k,p in bank.named_parameters()}
    bank.zero_grad(set_to_none=True)
    for sl in (slice(0,7),slice(7,18)):
        bank.formation_loss(h[sl],z[sl],y[sl],d[sl],w[sl],denominator=len(h)).backward()
    grad_error=max(float((p.grad-full_grads[k]).abs().max()) for k,p in bank.named_parameters())
    assert all(torch.allclose(p.grad,full_grads[k],atol=1e-6,rtol=1e-4) for k,p in bank.named_parameters())
    opt=torch.optim.Adam(bank.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    opt.zero_grad(set_to_none=True);bank.formation_loss(h,z,y,torch.full_like(d,2),w).backward();opt.step()
    before={k:dict(parameter=p.detach().clone(),state=copy.deepcopy(opt.state[p])) for k,p in bank.named_parameters() if k.startswith('heads.s2_')}
    opt.zero_grad(set_to_none=True);bank.formation_loss(h,z,y,torch.zeros_like(d),w).backward();opt.step()
    for k,p in bank.named_parameters():
        if k in before:
            assert p.grad is None and torch.equal(p,before[k]['parameter'])
            assert state_hash(opt.state[p])==state_hash(before[k]['state'])
    transient=Path(artifact_root)/'_temporary';transient.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='libraries_semantic_',dir=transient) as td:
        file=Path(td)/'state.pt';identity={'check':'formation_resume'}
        _save(file,identity,'formation',bank,opt,pass_id=1,next_logical_step=2)
        resumed=FunctionBank(pilot).to(device);resumed_opt=torch.optim.Adam(resumed.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        _restore(load_checkpoint(file,identity),resumed,resumed_opt)
        for mod,optim in ((bank,opt),(resumed,resumed_opt)):
            optim.zero_grad(set_to_none=True);mod.formation_loss(h,z,y,d,w).backward();optim.step()
        state_error=max(float((p-q).abs().max()) for p,q in zip(bank.parameters(),resumed.parameters()))
        assert state_error<=1e-6 and state_hash(opt.state_dict())==state_hash(resumed_opt.state_dict())
    candidates=bank.candidates(h,z,d,pilot)
    assert candidates.shape==(18,2,5)
    with torch.no_grad():
        for t in range(3):
            mask=d==t
            for q in range(2):
                ordered=[z[mask,q]+pilot.head(t,q)(h[mask])]
                ordered.extend(z[mask,q]+bank.head(s,k,q)(h[mask]) for s in range(3) if s!=t for k in range(3) if k!=s)
                assert torch.equal(candidates[mask,q],torch.stack(ordered,-1))
    reader=ReaderGroup().to(device)
    reader.normalizers={(t,q):Normalizer.from_array(np_rng.normal(size=(20,263))) for t in range(3) for q in range(2)}
    reader.off=True;p,alpha=reader.predict(h,z,d,candidates)
    assert torch.equal(p,candidates.double().sigmoid()[...,0]) and torch.equal(alpha[...,0],torch.ones_like(alpha[...,0]))
    a=torch.randn(20,5,device=device,dtype=torch.float64);g=torch.randn(20,5,device=device,dtype=torch.float64,requires_grad=True);yy=torch.randint(2,(20,),device=device).double()
    mix=(g.softmax(-1)*a.sigmoid()).sum(-1);stable=stable_mixture_bce(a,g,yy,reduction='none')
    mixerr=float((stable-(-yy*mix.log()-(1-yy)*torch.log1p(-mix))).abs().max().detach());assert mixerr<=1e-10
    simplex=[]
    for k in (3,5):
        probs=np_rng.uniform(.02,.98,(300,k));labels=np_rng.binomial(1,probs[:,1]);alpha,receipt=solve_simplex(probs,labels)
        assert receipt['accepted'];simplex.append({'K':k,'gap':receipt['fw_gap'],'sum_error':receipt['sum_error']})
    rows=np.arange(180,dtype=np.int64);domains=rows%3;labels=np.stack((rows%2,(rows//2)%2),-1).astype(np.float32)
    data=SimpleNamespace(dataset='movielens',domain=domains,batch=lambda ids,role,device,purpose:SimpleNamespace(y=torch.as_tensor(labels[ids])))
    weights=np_rng.uniform(.1,1.9,(180,2,1));weights=np.concatenate((weights,2-weights),-1)
    probs=np_rng.uniform(size=(180,2));shuffled,receipt=shuffled_weights(data,rows,weights,probs,5101)
    assert np.allclose(shuffled.sum(-1),2,atol=1e-15)
    for s in range(3):
        for q in range(2):
            ids=np.flatnonzero(domains==s);cuts=np.asarray(receipt[f's{s}_q{q}']['cuts']);bins=np.searchsorted(cuts,probs[ids,q],side='right')
            for label in (0,1):
                for box in range(10):
                    take=ids[(labels[ids,q]==label)&(bins==box)]
                    left=sorted(tuple(v) for v in weights[take,q]);right=sorted(tuple(v) for v in shuffled[take,q]);assert left==right
    whole=select_whole_policy([{'candidate':'OFF','macro_logloss':.4},{'candidate':0,'macro_logloss':.4},{'candidate':1,'macro_logloss':.41}]);assert whole['candidate']=='OFF'
    local=FunctionBank(pilot,kind='LOCAL_CAP');assert sum(p.numel() for p in local.parameters())==198168
    return dict(status='PASS',device=device,pilot_parameters=49542,source_function_parameters=99084,local_function_parameters=198168,
                formation_loss_error=err,microbatch_gradient_error=grad_error,absent_initialized_Adam_unchanged=True,
                frozen_input_and_pilot_gradients=True,complete_resume_parameter_error=state_error,
                candidate_order='local,foreign-source,intended-target',exact_OFF=True,stable_mixture_fp64_error=mixerr,
                simplex=simplex,shuffle_joint_vectors_preserved=True,whole_policy_tie='OFF')


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    print(json.dumps(run(args.device),indent=2))
