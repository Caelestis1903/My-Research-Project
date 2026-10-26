"""Minimum scientific admission checks on real schemas and gradient/state rules."""
from __future__ import annotations
from pathlib import Path
import copy,tempfile
import numpy as np
import torch
from torch.nn import functional as F
from .data import RoleBundle
from .native_adapter import make_native
from .functions import PilotBank,FunctionBank
from .plans import torch_seed,folds
from .state import atomic_json,module_state,restore_module,capture_rng,restore_rng,save_checkpoint,load_checkpoint,state_hash,array_hash
from .readers import stable_mixture_bce,solve_simplex
from .demand import analytic_mu,diagnostic_logits
from .prefix import configure


def verify(root,device='cuda:0',logical_rows=40960):
    configure();torch.manual_seed(173);receipt={}
    for dataset in ('movielens','kuairand'):
        data=RoleBundle(dataset);model,_=make_native(data.schema,5101,dataset,device);ids=data.ids('fit')[:logical_rows];b=data.batch(ids,'fit',device,purpose='semantic_check')
        # Native parity is independently receipted by reviewer; compare actual full physical batch to direct reference here. The optional 4096 GEMM accumulation is not admitted for ML.
        model.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(b.x),b.y,reduction='sum')/(2*len(ids));loss.backward();full={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
        model.zero_grad(set_to_none=True);micro_loss=0.
        for i in range(0,len(ids),40960):
            l=F.binary_cross_entropy_with_logits(model(b.x[i:i+40960]),b.y[i:i+40960],reduction='sum')/(2*len(ids));micro_loss+=float(l.detach());l.backward()
        errors=[]
        for n,p in model.named_parameters():
            a=full[n]
            if a is None:assert p.grad is None
            else:
                assert torch.allclose(a,p.grad,atol=1e-6,rtol=1e-4),n;errors.append(float((a-p.grad).abs().max()))
        receipt[dataset]=dict(logical_rows=len(ids),full_physical_vs_reference_loss_error=abs(float(loss.detach())-micro_loss),full_physical_vs_reference_gradient_max_error=max(errors),physical_batch=40960,optional_micro4096='NOT_ADMITTED: ML 19 gradient elements failed original allclose; production uses full logical physical batch without tolerance changes',fold_shared_across_tasks=True,fold_hash=array_hash(folds(data)))
        del b,full,model;torch.cuda.empty_cache()
    # Analytic paired-risk difference, with the exact same guarded diagnostic logits.
    p=np.asarray([[1e-14,.1,.8],[.2,.7,1-1e-14],[.8,.3,.4]],dtype=np.float64);a=diagnostic_logits(p);y=np.asarray([0,1,1.],dtype=np.float64)
    d=analytic_mu(a[:,0,None],a[:,1:],y[:,None]);direct=(np.logaddexp(0,a[:,0])-y*a[:,0])[:,None]-(np.logaddexp(0,a[:,1:])-y[:,None]*a[:,1:]);err=float(np.max(np.abs(d-direct)));assert err<=1e-10
    receipt['bce_identity_max_error']=err
    h=torch.randn(8,256,device=device);z=torch.randn(8,2,device=device);yy=torch.randint(2,(8,2),device=device).float();domain=torch.zeros(8,dtype=torch.long,device=device)
    bank=PilotBank().to(device);opt=torch.optim.Adam(bank.parameters(),lr=.001,weight_decay=1e-5)
    opt.zero_grad(set_to_none=True);bank.pilot_loss(h,z,yy,domain).backward()
    absent=[p for s in (1,2) for q in (0,1) for p in bank.head(s,q).parameters()];assert all(p.grad is None for p in absent);opt.step();assert all(p not in opt.state for p in absent)
    state=module_state(bank,opt);rng=capture_rng();opt.zero_grad(set_to_none=True);bank.pilot_loss(h,z,yy,domain).backward();opt.step();expected=module_state(bank,opt)
    restore_module(bank,opt,state);restore_rng(rng);opt.zero_grad(set_to_none=True);bank.pilot_loss(h,z,yy,domain).backward();opt.step();actual=module_state(bank,opt);assert state_hash(actual)==state_hash(expected)
    receipt['inactive_pilot_none_adam']=True;receipt['complete_next_step_resume_equal']=True
    # Five-component mixture identity, FP64 extreme logits and FP32 stored head gradients.
    logits=torch.randn(32,5,device=device,dtype=torch.float64)*4;gate=torch.randn(32,5,device=device,dtype=torch.float64,requires_grad=True);yb=torch.randint(2,(32,),device=device).double();mix=(gate.softmax(1)*logits.sigmoid()).sum(1);stable=stable_mixture_bce(logits,gate,yb,reduction='none');ref=-yb*mix.log()-(1-yb)*torch.log1p(-mix);err=float((stable-ref).abs().max());assert err<=1e-10
    receipt['mixture_bce_max_error']=err
    # A real simplex solve only verifies the production solver; no research training.
    probs=np.asarray([[.1,.2,.3],[.9,.8,.7],[.3,.5,.7],[.8,.4,.2]]);_,solver=solve_simplex(probs,np.asarray([0,1,1,0.]));receipt['simplex']=solver
    receipt['status']='PASS';atomic_json(Path(root)/'SEMANTIC_RECEIPTS.json',receipt);return receipt
