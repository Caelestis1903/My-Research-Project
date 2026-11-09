import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.optimize import root
from experiments.pub_mdmt_20261109.common import setup,rng_state,restore_rng,state_hash
from experiments.pub_mdmt_20261109.data import load_bundle,model_for
from experiments.pub_mdmt_20261109.registry import ParameterLayout
from experiments.pub_mdmt_20261109.group_registry import GROUPS,build_path_groups
from experiments.pub_mdmt_20261109.optimizer import PUBAdam,losses,gradients,candidate,CELLS
from experiments.pub_mdmt_20261109.solver import CCPSolver
from experiments.pub_mdmt_20261109.training import consider
from experiments.pub_mdmt_20261109.metrics import metrics

def toy(dtype=torch.float64):
    model=torch.nn.Module(); model.register_parameter('p',torch.nn.Parameter(torch.linspace(-1,1,36,dtype=dtype)))
    layout=ParameterLayout.from_model(model); coords=torch.arange(12).repeat_interleave(3)
    registry=SimpleNamespace(layout=layout,indices=lambda device:coords.to(device))
    return model,registry

def supported_grads(dtype=torch.float64):
    setup(2022); matrix=torch.randn(6,36,dtype=dtype)
    matrix[:,:3]=matrix[:,:3].abs()+.1
    for c,(d,t) in enumerate(CELLS):
        for i,g in enumerate(GROUPS):
            if g not in ('S',f'D{d}',f'T{t}',f'DT{d}{t}'): matrix[c,i*3:(i+1)*3]=0
    return matrix

@pytest.mark.parametrize('dataset',['movielens','kuairand'])
def test_real_loss_graph_partition_empty_domain(dataset):
    setup(2022); b=load_bundle(dataset); model=model_for(b); registry=build_path_groups(model,b)
    ids=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:8] for d in range(3)])
    x=torch.tensor(np.array(b.train_x[ids])).T; y=torch.tensor(np.array(b.train_y[ids]))
    terms,natural,active=losses(model,x,y,1)
    direct=torch.autograd.grad(natural,tuple(model.parameters()),retain_graph=True); gs=gradients(terms,model)
    assert torch.allclose(sum(terms),natural,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(gs.sum(0),registry.layout.flatten(direct),atol=1e-7,rtol=1e-6)
    coords=registry.indices('cpu')
    for c,(d,t) in enumerate(CELLS):
        allowed=[i for i,g in enumerate(GROUPS) if g in ('S',f'D{d}',f'T{t}',f'DT{d}{t}')]
        assert not gs[c,~torch.isin(coords,torch.tensor(allowed))].any()
    assert sum(registry.counts.values())==sum(p.numel() for p in model.parameters())
    terms,natural,active=losses(model,x[:,8:],y[8:],1); empty=gradients(terms,model)
    assert 0 not in active and 1 not in active and not empty[:2].any()


def test_candidate_hand_formula_and_no_mutation():
    g=torch.tensor([[.2,-.3],[.1,.5]],dtype=torch.float64); m=torch.tensor([.1,.2],dtype=torch.float64); v=torch.tensor([.3,.4],dtype=torch.float64)
    before=(m.clone(),v.clone()); result=candidate(g,m,v,7)
    expected=((.9*m+.1*g)/(1-.9**7))/(torch.sqrt((.999*v+.001*g*g)/(1-.999**7))+1e-8)
    torch.testing.assert_close(result,expected,atol=1e-14,rtol=1e-14)
    assert torch.equal(m,before[0]) and torch.equal(v,before[1])

@pytest.mark.parametrize('wd',[0.,1e-4,1e-5])
def test_adam_disabled_and_fallback_match_native(wd):
    model,reg=toy(); control=copy.deepcopy(model); native=torch.optim.Adam(control.parameters(),lr=.01,weight_decay=wd,foreach=False,fused=False)
    custom=PUBAdam(model,reg,mode='ADAM',weight_decay=wd)
    for k in range(4):
        gs=supported_grads()+k*.01; g=gs.sum(0)
        control.p.grad=g.clone(); native.step(); custom.apply(gs,list(range(6)))
        torch.testing.assert_close(model.p,control.p,atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(custom.m,native.state[control.p]['exp_avg'],atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(custom.v,native.state[control.p]['exp_avg_sq'],atol=1e-14,rtol=1e-14)
        assert custom.step==int(native.state[control.p]['step'])
    fallback,reg2=toy(); native2=copy.deepcopy(fallback); opt2=torch.optim.Adam(native2.parameters(),lr=.01,weight_decay=wd,foreach=False,fused=False)
    class Failed:
        def solve(self,*args): return None,dict(success=False,reason='test failure')
    pub=PUBAdam(fallback,reg2,solver=Failed(),weight_decay=wd)
    gs=supported_grads(); trace,_,_=pub.apply(gs,list(range(6))); native2.p.grad=gs.sum(0); opt2.step()
    torch.testing.assert_close(fallback.p,native2.p,atol=1e-14,rtol=1e-14)
    assert trace['groups']['S']['fallback']


def test_solver_diagonal_analytic_nondiagonal_and_permutation():
    solver=CCPSolver(niter=200,tol=1e-4,ridge=1e-12)
    for G in [np.diag([2.,7.]),np.array([[4.,1.],[1.,9.]]),np.eye(6)+.2*np.ones((6,6))]:
        a,diag=solver.solve(G)
        assert diag['success'],diag
        assert np.max(np.abs(a*(G@a)-1))<1e-4
        oracle=root(lambda z:np.exp(z)*(G@np.exp(z))-1,np.log(1/np.sqrt(np.diag(G)))).x
        np.testing.assert_allclose(a,np.exp(oracle),rtol=1e-4,atol=1e-5)
        p=np.arange(len(a))[::-1]; b,_=solver.solve(G[p][:,p]); np.testing.assert_allclose(a,b[p],rtol=1e-4,atol=1e-5)
        scaled,_=solver.solve(G*100); np.testing.assert_allclose(scaled,a/10,rtol=1e-4,atol=1e-5)
    assert solver.solve(np.zeros((2,2)))[0] is None

@pytest.mark.parametrize('topology',[0,1,2])
def test_task_order_replay_ram_restore_and_dt(topology):
    model,reg=toy(); second=copy.deepcopy(model); _,reg2=toy(); control=copy.deepcopy(model); native=torch.optim.Adam(control.parameters(),lr=.01,weight_decay=1e-4,foreach=False,fused=False)
    solver=CCPSolver(niter=200,tol=1e-4,ridge=1e-12)
    a=PUBAdam(model,reg,topology,solver=solver,weight_decay=1e-4); b=PUBAdam(second,reg2,topology,solver=CCPSolver(niter=200,tol=1e-4,ridge=1e-12),weight_decay=1e-4)
    gs=supported_grads(); gs=torch.stack([gs[::2].sum(0),gs[1::2].sum(0)]) if topology==0 else gs
    p=list(reversed(range(len(gs)))); before=a.state_dict(); record,_,_=a.apply(gs,list(range(6))); b.apply(gs[p],list(reversed(range(6))),task_order=p)
    torch.testing.assert_close(model.p,second.p,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a.m,b.m); torch.testing.assert_close(a.v,b.v)
    control.p.grad=gs.sum(0); native.step(); torch.testing.assert_close(model.p[18:],control.p[18:],atol=1e-14,rtol=1e-14)
    if topology==1:
        for g in ('D0','D1','D2','T0','T1'):
            row=record['groups'][g]
            assert row['alpha']==[record['groups']['S']['alpha'][i] for i in row['members']]
    replay_model,reg3=toy(); g=PUBAdam(replay_model,reg3,topology,mode='G',weight_decay=1e-4)
    class NoSolver:
        def solve(self,*args): raise AssertionError('G must never solve')
    g.solver=NoSolver(); replay_record,_,_=g.apply(gs,list(range(6)),replay=record['groups']); assert replay_record==record
    snapshot=copy.deepcopy(a.state_dict()); theta=model.p.detach().clone(); rng=rng_state(); loader=torch.Generator().manual_seed(2022+11000); loader_state=loader.get_state(); expected_order=torch.randperm(100,generator=loader)
    a.apply(gs,list(range(6))); expected=model.p.detach().clone()
    model.p.data.copy_(theta); a.load_state_dict(snapshot); restore_rng(rng); loader.set_state(loader_state); assert torch.equal(expected_order,torch.randperm(100,generator=loader))
    a.apply(gs,list(range(6))); torch.testing.assert_close(model.p,expected,atol=0,rtol=0)
    assert before['step']==0 and not before['m'].any()


def test_selection_tie_and_raw_probability_ll():
    state={'p':torch.tensor([1.])}; selection=consider(None,1,{'macro_auc':.7},state); other=consider(selection,2,{'macro_auc':.7},state)
    assert other is selection and other['epoch']==1
    y=np.tile(np.array([[0,0],[1,1]]),(3,1)); d=np.repeat([0,1,2],2); p=np.tile([[.2,.3],[.8,.7]],(3,1)); values=metrics(y,p,d)
    assert abs(values['macro_logloss']-float(-np.log([.8,.7]).mean()))<1e-14
    assert values['macro_auc']==1 and values['pooled_task_mean_auc']==1

@pytest.mark.parametrize('topology',[0,1,2])
def test_weighted_formula_uses_one_wd_and_old_state(topology):
    model,reg=toy(); opt=PUBAdam(model,reg,topology,weight_decay=1e-4)
    class Fixed:
        def solve(self,G,warm=None): return np.arange(1,len(G)+1)*.13,{'success':True}
    opt.solver=Fixed(); opt.m.fill_(.02); opt.v.fill_(.003); opt.step=3
    old=model.p.detach().clone(); gs=supported_grads()
    if topology==0: gs=torch.stack([gs[::2].sum(0),gs[1::2].sum(0)])
    expected={}
    def U(z): return ((.9*.02+.1*z)/(1-.9**4))/(torch.sqrt((.999*.003+.001*z*z)/(1-.999**4))+1e-8)
    for group in GROUPS:
        idx=opt.indices[group]; members=opt.members(group,list(range(6)))
        if len(members)>1:
            if topology==1 and group!='S': weights=torch.tensor([(i+1)*.13 for i in members],dtype=old.dtype)
            else: weights=torch.arange(1,len(members)+1,dtype=old.dtype)*.13
            gb=(gs[members][:,idx]*weights[:,None]).sum(0); z=gb+1e-4*old[idx]
            direction=(U(gs[members][:,idx])*weights[:,None]).sum(0)+U(z)-U(gb)
            expected[group]=(idx,old[idx]-.01*direction,.9*.02+.1*z,.999*.003+.001*z*z)
    trace,_,_=opt.apply(gs,list(range(6)))
    for group,(idx,p,m,v) in expected.items():
        torch.testing.assert_close(model.p[idx],p,atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(opt.m[idx],m,atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(opt.v[idx],v,atol=1e-14,rtol=1e-14)
    if topology==2:
        assert len(trace['groups']['S']['alpha'])==6 and len(trace['groups']['D0']['alpha'])==2 and len(trace['groups']['T0']['alpha'])==3


def test_zero_single_active_and_test_boundary():
    from experiments.pub_mdmt_20261109.training import evaluate
    for active in ([],[0]):
        model,reg=toy(); control=copy.deepcopy(model); opt=PUBAdam(model,reg,1,weight_decay=1e-4)
        native=torch.optim.Adam(control.parameters(),lr=.01,weight_decay=1e-4,foreach=False,fused=False)
        gs=torch.zeros(6,36,dtype=torch.float64)
        if active: gs[0]=supported_grads()[0]
        for _ in range(2):
            opt.apply(gs,active); control.p.grad=gs.sum(0); native.step()
            torch.testing.assert_close(model.p,control.p,atol=1e-14,rtol=1e-14)
    with pytest.raises(ValueError,match='Test not loaded'): evaluate(model,SimpleNamespace(),'test')


def test_offline_frequency100_staleness():
    from experiments.pub_mdmt_20261109.report import frequency100_staleness
    trace=[{'step':1,'groups':{'S':{'members':[0,1],'alpha':[1.,1.]}}},{'step':2,'groups':{'S':{'members':[0,1],'alpha':[2.,2.]}}}]
    row=frequency100_staleness(trace)[0]
    assert row['frequency100_refreshes']==1 and row['mean_relative_alpha_distance']==.25 and row['mean_age_steps']==.5
