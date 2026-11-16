import copy
import io
import numpy as np
import pytest
import torch
from ..protocol import config, make_optimizer, cpu_copy, equal, update_ema
from ..boundary_sampler import boundaries, sample, CONDITIONS, forward_logits
from ..objectives import train_step
from ..optimizers import PairOptimizer, shared_name
from ..vendor.ema02.tests.test_core import fixture
from ..vendor.ema02.objectives import train_step as source_step
from ..vendor.ema02.protocol import make_optimizer as source_optimizer


def test_boundaries_support_and_expectations():
    y=np.array([[0,0],[1,0],[1,1]]*3);d=np.repeat(np.arange(3),3)
    active,w,q,g=boundaries(d,y,{},True)
    assert len(active)==12 and all(CONDITIONS[i][2]!=1 and CONDITIONS[i][3]!=1 for i in active)
    assert abs(w.sum()-1)<1e-12
    a2,w2,q2,_=boundaries(np.repeat(np.arange(3),4),np.tile([[0,0],[0,1],[1,0],[1,1]],(3,1)),{},False)
    assert len(a2)==24
    old=np.arange(len(w),dtype=float)/10;grad=np.arange(len(w),dtype=float)/3
    # Enumerate the actual conditional estimator, and one-condition history update.
    cv=[np.dot(w,old)+w[s]/q[s]*(grad[s]-old[s]) for s in range(len(w))]
    assert abs(np.dot(q,cv)-np.dot(w,grad))<1e-12
    for c in range(len(w)):
        expected=sum(q[s]*(old[c]+(.01/q[c]*(grad[c]-old[c]) if s==c else 0)) for s in range(len(w)))
        assert abs(expected-(.99*old[c]+.01*grad[c]))<1e-12
    assert not boundaries(d,np.zeros_like(y),{},True)[0]
    ctx=dict(dataset='movielens',seed=2022,epoch=1,global_step=1)
    p=sample(d,y,{},True,ctx,np.arange(9));c=CONDITIONS[p['selected']]
    assert np.all(d[p['pos']]==c[0]) and np.all(d[p['neg']]==c[0])
    assert np.all(y[p['pos'],c[1]]==1) and np.all(y[p['neg'],c[1]]==0)


def test_hook_grouping_and_source_bypass():
    m,x,y,ids,ctx=fixture();p,z=forward_logits(m,x)
    assert torch.equal(p,z.sigmoid())
    assert shared_name('expert.0.domain_specific.0.weight') and not shared_name('domain_expert.0.weight')
    reference=copy.deepcopy(m);ema=copy.deepcopy(m).requires_grad_(False);other=copy.deepcopy(ema)
    c=config('P11','movielens',0.);opt=make_optimizer(m,c);refopt=source_optimizer(reference,c)
    for step in (1,2):
        ctx['global_step']=step
        a=train_step(m,opt,ema,x,y,ids,c,ctx)
        b=source_step(reference,refopt,other,x,y,ids,dict(c,arm='A0'),ctx)
        assert a['loss']==b['loss'] and equal(m.state_dict(),reference.state_dict()) and equal(ema.state_dict(),other.state_dict())
        assert all(equal(p.grad,q.grad) for p,q in zip(m.parameters(),reference.parameters()))


@pytest.mark.parametrize('arm',['P00','P10','P01','P11','PCV','PCOLL','A0','A1'])
def test_resume_and_no_graph(arm):
    m,x,y,ids,ctx=fixture();e=copy.deepcopy(m).requires_grad_(False);c=config(arm,'movielens',.05 if arm.startswith('P') else 0)
    opt=make_optimizer(m,c);train_step(m,opt,e,x,y,ids,c,ctx)
    # BytesIO exercises complete optimizer/history serialization without a checkpoint file.
    stream=io.BytesIO();torch.save(cpu_copy(dict(m=m.state_dict(),e=e.state_dict(),o=opt.state_dict())),stream);stream.seek(0)
    snap=torch.load(stream,weights_only=False);other=copy.deepcopy(m);oe=copy.deepcopy(e);oo=make_optimizer(other,c)
    other.load_state_dict(snap['m']);oe.load_state_dict(snap['e']);oo.load_state_dict(snap['o'])
    ctx['global_step']=2
    train_step(m,opt,e,x,y,ids,c,ctx);train_step(other,oo,oe,x,y,ids,c,ctx)
    assert equal(m.state_dict(),other.state_dict()) and equal(e.state_dict(),oe.state_dict()) and equal(opt.state_dict(),oo.state_dict())
    assert all(h.grad_fn is None and not h.requires_grad for hs in opt.history.values() for h in hs)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared_weight=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
        self.other=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float64))
        self.absent=torch.nn.Parameter(torch.tensor([4.],dtype=torch.float64))

@pytest.mark.parametrize('mode',['eccm','collapsed','cv'])
def test_hand_update_clocks_none_zero(mode):
    m=Tiny();c=dict(mode=mode,lambda_pair=.2,lr=.01,weight_decay=.1);o=PairOptimizer(m,c)
    pair=dict(selected=0,active=[0,1],w=np.array([.4,.6]),q=np.array([.25,.75]),index=0)
    r=[torch.tensor([.7],dtype=torch.float64)];real=.3;ratio=1.6
    m.shared_weight.grad=torch.tensor([real+.2*ratio*.7],dtype=torch.float64)
    m.other.grad=torch.zeros_like(m.other);o.step(pair,r)
    mixed=real+.2+.2*ratio*.7
    if mode=='eccm':numerator=real+.2+.2*.4*(.01/.25*.7)/.01
    elif mode=='collapsed':numerator=real+.2+.2*ratio*.7
    else:numerator=mixed
    expected=2-.01*numerator/(abs(mixed)+1e-8)
    assert abs(float(m.shared_weight.detach())-expected)<1e-12
    assert float(m.absent.detach())==4 and float(m.other.detach())<3
    assert m.absent not in o.adam.state
    if mode=='eccm':assert o.active=={0:1,1:1} and o.selected=={0:1}
    old=cpu_copy(o.history);clock=o.pair_clock
    m.shared_weight.grad=torch.zeros_like(m.shared_weight);o.step()
    assert equal(old,o.history) and o.pair_clock==clock


def test_cv_reads_old_history():
    m=Tiny();o=PairOptimizer(m,dict(mode='cv',lambda_pair=.2,lr=.01,weight_decay=.1))
    o._table(0)[0].fill_(.8);o._table(1)[0].fill_(.2)
    pair=dict(selected=0,active=[0,1],w=np.array([.4,.6]),q=np.array([.25,.75]),index=0)
    m.shared_weight.grad=torch.tensor([.3+.2*1.6*.7],dtype=torch.float64)
    o.step(pair,[torch.tensor([.7],dtype=torch.float64)])
    expected=.3+.2*1.6*.7+.2*(.4*.8+.6*.2-1.6*.8)
    assert abs(float(m.shared_weight.grad)-expected)<1e-12
    assert abs(float(o.history[0][0])-.79)<1e-12
