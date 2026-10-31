import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from ..protocol import setup,protocol,optimizer,snapshot,restore,equal,rng_state,cpu_copy
from ..model import make_model,perturb_embedding
from ..objectives import losses
from ..probe import combine


def synthetic():
    columns=('user_id','movie_id','gender','age','occupation','zip','cate_id','domain_indicator')
    schema=SimpleNamespace(columns=columns,dense_columns=('age',),vocab_sizes={name:8 for name in columns})
    return SimpleNamespace(features=schema)


def batch(n=9):
    x=torch.zeros((8,n));x[-1]=torch.arange(n)%3;x[3]=.5
    y=(torch.arange(n*2).reshape(n,2)%3==0).float()
    return x,y


def test_reference_and_zero_noise():
    setup(2022);p=protocol();bundle=synthetic();conf=p['datasets']['movielens']
    source=make_model(bundle,conf,reference=True);initial=copy.deepcopy(source.state_dict())
    for scale in (1.,.1):
        for enabled,kappa in [(False,.1),(True,0.)]:
            model=make_model(bundle,conf);model.load_state_dict(initial)
            reference=make_model(bundle,conf,reference=True);reference.load_state_dict(initial)
            x,y=batch();a=optimizer(model,p);b=optimizer(reference,p)
            loss,_=losses(model,x,y,0,2022000001,enabled=enabled,relative_std=kappa)
            oracle=F.binary_cross_entropy(reference(x),y,reduction='none').mean(1).mean()
            torch.testing.assert_close(loss,oracle,atol=1e-6,rtol=1e-5)
            (loss*scale).backward();(oracle*scale).backward()
            for left,right in zip(model.parameters(),reference.parameters()):
                assert (left.grad is None)==(right.grad is None)
                if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=1e-6,rtol=1e-5)
            a.step();b.step()
            for left,right in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)


def test_budget_slices_and_rng():
    setup(2022);p=protocol();m=make_model(synthetic(),p['datasets']['movielens']);x,y=batch()
    state=torch.get_rng_state().clone();embedding=m.embedding(x,m.features,squeeze_dim=True)
    modified=perturb_embedding(embedding,m.perturb_slices,.1,torch.Generator().manual_seed(99))
    for name in ('age','domain_indicator'):
        lo,hi=m.field_slices[name];assert torch.equal(embedding[:,lo:hi],modified[:,lo:hi])
    assert torch.equal(state,torch.get_rng_state())
    for n,target in [(1,0),(9,2),(4,8)]:
        bx,by=batch(n);value,info=losses(m,bx,by,target,11)
        expected=info['real'].sum()/info['real'].numel()
        if info['augmented'] is not None:
            expected+=.5*(info['augmented'].sum()-info['real'][info['mask']].sum())/info['real'].numel()
        torch.testing.assert_close(value,expected)
        zero,_=losses(m,bx,by,target,11,mix_lambda=0)
        torch.testing.assert_close(zero,F.binary_cross_entropy(m(bx),by))


def test_ram_resume_and_none():
    setup(2022);p=protocol();bundle=synthetic();conf=p['datasets']['movielens'];model=make_model(bundle,conf)
    opt=optimizer(model,p);loaders={'train':torch.Generator().manual_seed(13022)};x,y=batch()
    value,_=losses(model,x,y,0,2022000001);value.backward();opt.step()
    torch.randperm(19,generator=loaders['train'])
    state=snapshot(model,opt,loaders,1,1,{},'cpu')
    next_order=torch.randperm(19,generator=loaders['train']);opt.zero_grad(set_to_none=True)
    value,_=losses(model,x,y,0,2022000002);value.backward();opt.step();expected=cpu_copy((model.state_dict(),opt.state_dict()))
    clone=make_model(bundle,conf);clone_opt=optimizer(clone,p);clone_loaders={'train':torch.Generator()}
    restore(clone,clone_opt,clone_loaders,state,'cpu')
    assert torch.equal(next_order,torch.randperm(19,generator=clone_loaders['train']))
    value,_=losses(clone,x,y,0,2022000002);value.backward();clone_opt.step()
    assert equal(expected,(clone.state_dict(),clone_opt.state_dict()))
    skipped=torch.nn.Parameter(torch.ones(1));adam=torch.optim.Adam([skipped],lr=.01,weight_decay=1.)
    before=skipped.detach().clone();adam.step();assert torch.equal(before,skipped) and not adam.state
    assert combine((None,),(None,),.5)==(None,)
