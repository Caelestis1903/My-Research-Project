import copy
import importlib.util
import numpy as np
import torch
from ..protocol import config,make_optimizer,cpu_copy,ROOT,update_ema
from ..mixing import build_mix_batches,forward_mixed,model_forward,MixBatch
from ..objectives import train_step,real_loss,mixed_objective,collect_history_probes
from ..optimizers import project_history,transport_correction,shared_name,norm,HistoryOptimizer
from ..vendor.v3.vendor.ema02.objectives import train_step as source_step
from ..vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import MDMTRec
from ..vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.basic.features import SparseFeature

def fixture():
    torch.set_num_threads(1);torch.manual_seed(2022)
    m=MDMTRec([SparseFeature(i,16 if i<3 else 3,embed_dim=16) for i in range(4)],
        domain_num=3,task_num=2,fcn_dims=[32,16,16,8],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    m.field_slices={**{f'f{i}':(16*i,16*(i+1)) for i in range(3)},'domain_indicator':(48,64)}
    m.perturb_slices=[(0,16),(16,32),(32,48)]
    x=torch.stack([torch.arange(12)%16,(torch.arange(12)+2)%16,(torch.arange(12)+4)%16,torch.arange(12)//4]).float()
    y=torch.tensor([[0,0],[1,0],[1,1],[1,0]]*3).float()
    return m,x,y,np.arange(12),dict(dataset='movielens',seed=2022,epoch=1,global_step=1)

def close(a,b):
    if torch.is_tensor(a):assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        for x,y in zip(a,b):close(x,y)
    else:assert a==b

def test_reference_and_projection_transport():
    path=ROOT/'research_plans/26-11-03/design_reference.py'
    spec=importlib.util.spec_from_file_location('reference1103',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    assert ref.self_test()['test_count']==13
    rng=np.random.default_rng(2022)
    for _ in range(8):
        m=rng.normal(size=6);d=np.exp(rng.normal(size=6));pairs=[(rng.normal(size=6),rng.normal(size=6)) for _ in range(2)]
        result,_=project_history([torch.tensor(m)],[torch.tensor(d)],
             [([torch.tensor(a)],[torch.tensor(b)]) for a,b in pairs],.5)
        np.testing.assert_allclose(result[0].numpy(),ref.repair_history(m,d,pairs,.5),atol=1e-9)
    g=torch.tensor([.6,.5],dtype=torch.float64);old=torch.tensor([.4,.4],dtype=torch.float64)
    correction,_=transport_correction([g],[old],.25,5)
    expected=ref.transport_moment(np.zeros(2),g.numpy(),old.numpy(),5)-.1*g.numpy()
    np.testing.assert_allclose(correction[0],expected,atol=1e-12)

def test_donor_support_rng_and_domain_and_both_embeddings():
    m,x,y,rows,ctx=fixture();before=torch.get_rng_state().clone()
    batches=build_mix_batches(x,y,rows,ctx,'conditioned',.1)
    assert torch.equal(before,torch.get_rng_state())
    for mb in batches:
        assert torch.equal(x[-1,mb.anchor_idx],x[-1,mb.donor_idx])
        assert torch.equal(y[mb.anchor_idx,mb.target_task],y[mb.donor_idx,mb.target_task])
        for a,d,level in zip(mb.anchor_idx,mb.donor_idx,mb.support_level):
            if level==1:assert y[a,1-mb.target_task]!=y[d,1-mb.target_task]
            elif level==2:assert not ((x[-1]==mb.domain)&(y[:,mb.target_task]==y[a,mb.target_task])&(y[:,1-mb.target_task]!=y[a,1-mb.target_task])).any()
            else:assert a==d
    singleton=build_mix_batches(x[:,:1],y[:1],rows[:1],ctx,'conditioned',.1)
    assert all(mb.support_level.tolist()==[3] for mb in singleton)
    # Directly retain the embedding output to inspect both interpolation endpoints.
    observed=[]
    hook=m.embedding.register_forward_hook(lambda mod,args,out: (out.retain_grad(),observed.append(out)) and None)
    mb=MixBatch(torch.tensor([1]),torch.tensor([2]),0,0,torch.tensor([1]),.1,'test')
    forward_mixed(m,x,mb)[:,0].sum().backward();hook.remove()
    assert observed[0].grad[0,:48].abs().sum()>0 and observed[0].grad[1,:48].abs().sum()>0
    assert not m.embedding._forward_hooks

def test_exact_bypasses_source_adam_and_ram_restore():
    m,x,y,rows,ctx=fixture()
    for arm,kw,source_arm in [('A0f',{},'A0'),('H_MIX',{'kappa':0},'A0'),('T0',{'kappa':0},'A0'),
            ('M',{'lambda_mix':0},'A0'),('M',{'tau':0},'A0'),('A1',{},'A1')]:
        a=copy.deepcopy(m);b=copy.deepcopy(m);ea=copy.deepcopy(m);eb=copy.deepcopy(m)
        c=config(arm,'movielens',**kw);oa=make_optimizer(a,c)
        ob=torch.optim.Adam(b.parameters(),lr=.01,weight_decay=1e-4,foreach=False,fused=False)
        sc=dict(c,arm=source_arm,target_domain=0,kappa=.05,lambda_aug=.5)
        for step in (1,2):
            context=dict(ctx,global_step=step);r=train_step(a,oa,ea,x,y,rows,c,context)
            s=source_step(b,ob,eb,x,y,rows,sc,dict(context,study='EMA02V2',split_id='original'))
            assert abs(r['loss']-s['loss'])<1e-6
            close(a.state_dict(),b.state_dict());close(ea.state_dict(),eb.state_dict())
            for p,q in zip(a.parameters(),b.parameters()):
                if p.grad is None:assert q.grad is None
                else:close(p.grad,q.grad)
            if step==1:oa.load_state_dict(cpu_copy(oa.state_dict()))
    a=copy.deepcopy(m);b=copy.deepcopy(m);ca=config('TM','movielens',kappa=0);cb=config('M','movielens')
    train_step(a,make_optimizer(a,ca),copy.deepcopy(a),x,y,rows,ca,ctx)
    train_step(b,make_optimizer(b,cb),copy.deepcopy(b),x,y,rows,cb,ctx)
    close(a.state_dict(),b.state_dict())

def test_functional_complete_state_and_history_restore():
    m,x,y,rows,ctx=fixture();batches=build_mix_batches(x,y,rows,ctx,'conditioned',.1)
    state=({n:p.detach().clone().requires_grad_(shared_name(n)) for n,p in m.named_parameters()},
           {n:b.detach().clone() for n,b in m.named_buffers()})
    close(model_forward(m,x,state),m(x))
    original=mixed_objective(m,x,y,batches,.1)
    previous=mixed_objective(m,x,y,batches,.1,state)
    g=torch.autograd.grad(original,[p for n,p in m.named_parameters() if shared_name(n)])
    old=torch.autograd.grad(previous,[p for n,p in state[0].items() if shared_name(n)])
    for a,b in zip(g,old):close(a,b)
    name=next(n for n in state[0] if 'tower' in n and n.endswith('.3.bias'))
    state[0][name].add_(1.)
    assert not torch.allclose(model_forward(m,x,state),m(x))
    for arm in ('H_MIX','TM','H_NORM','T_NORM','C_MIX'):
        a=copy.deepcopy(m);c=config(arm,'movielens');ea=copy.deepcopy(a);oa=make_optimizer(a,c)
        initial=cpu_copy(a.state_dict());result=train_step(a,oa,ea,x,y,rows,c,ctx)
        assert set(result['diagnostics']['task_support_rates'])=={'0','1'}
        assert all(abs(sum(rates.values())-1)<1e-6 for rates in result['diagnostics']['task_support_rates'].values())
        if arm.startswith('T'):
            for n,p in oa.previous_state[0].items():close(p,initial[n])
        b=copy.deepcopy(a);eb=copy.deepcopy(ea);ob=make_optimizer(b,c);ob.load_state_dict(cpu_copy(oa.state_dict()))
        train_step(a,oa,ea,x,y,rows,c,dict(ctx,global_step=2))
        train_step(b,ob,eb,x,y,rows,c,dict(ctx,global_step=2))
        close(a.state_dict(),b.state_dict());close(ea.state_dict(),eb.state_dict());close(oa.state_dict(),ob.state_dict())

def test_probe_preserves_grad_and_none_clock():
    m,x,y,rows,ctx=fixture();real_loss(m,x,y).backward();before=[None if p.grad is None else p.grad.clone() for p in m.parameters()]
    batches=build_mix_batches(x,y,rows,ctx,'conditioned',.1,domains=[0])
    collect_history_probes(m,x,y,ctx,batches,'conditioned_mix')
    for p,g in zip(m.parameters(),before):
        if g is None:assert p.grad is None
        else:close(p.grad,g)
    c=config('A0f','movielens');o=make_optimizer(m,c);o.zero_grad();p=next(m.parameters());o.step();assert p not in o.adam.state
    p.grad=torch.zeros_like(p);o.step();assert o.adam.state[p]['step']==1

def test_history_controls_and_coupled_decay_once(monkeypatch):
    class TwoParameters(torch.nn.Module):
        def __init__(self):
            super().__init__();self.shared_weight=torch.nn.Parameter(torch.tensor([-.4,.3],dtype=torch.float64))
            self.other=torch.nn.Parameter(torch.tensor([.2],dtype=torch.float64))
    models={};optimizers={}
    probes=[([torch.tensor([1.,0.],dtype=torch.float64)],[torch.tensor([1.,.1],dtype=torch.float64)])]
    for arm in ('H_MIX','H_NORM','C_MIX','TM','T_NORM'):
        m=TwoParameters();o=HistoryOptimizer(m,config(arm,'movielens'))
        o.global_step=1;o.moments={'shared_weight':dict(step=1,m=torch.tensor([-.2,.1],dtype=torch.float64),v=torch.tensor([.03,.02],dtype=torch.float64))}
        m.shared_weight.grad=torch.tensor([.1,-.3],dtype=torch.float64);m.other.grad=torch.tensor([.4],dtype=torch.float64)
        models[arm]=m;optimizers[arm]=o
    old_parameters={'shared_weight':torch.tensor([-.3,.2],dtype=torch.float64)}
    old_grads=[torch.tensor([-.2,.2],dtype=torch.float64)]
    for arm,o in optimizers.items():
        o.step(probes=probes,old_point_data_grads=old_grads,old_parameters=old_parameters)
        g=torch.tensor([.1,-.3],dtype=torch.float64)+1e-4*torch.tensor([-.4,.3],dtype=torch.float64)
        expected_v=.999*torch.tensor([.03,.02],dtype=torch.float64)+.001*g.square()
        close(o.moments['shared_weight']['v'],expected_v)
    s=optimizers['H_MIX'].moments['shared_weight'];d=1/((s['v']/(1-.999**2)).sqrt()+1e-8)
    history={a:(o.moments['shared_weight']['m']-.1*g)/.9 for a,o in optimizers.items() if a.startswith('H')}
    close(norm([d.sqrt()*history['H_MIX']]),norm([d.sqrt()*history['H_NORM']]))
    close(norm([d.sqrt()*optimizers['TM'].moments['shared_weight']['m']]),norm([d.sqrt()*optimizers['T_NORM'].moments['shared_weight']['m']]))
    repaired,_=project_history([g],[d],probes,1.)
    expected=.9*torch.tensor([-.2,.1],dtype=torch.float64)+.1*repaired[0]
    close(optimizers['C_MIX'].moments['shared_weight']['m'],expected)
    # Isolate the NORM small-denominator rule from the projection solver tolerance.
    from .. import optimizers as module
    monkeypatch.setattr(module,'project_history',lambda m,d,p,k:([a*.5 for a in m],{'history_norm_ratio':.5}))
    m=TwoParameters();o=HistoryOptimizer(m,config('H_NORM','movielens'))
    tiny=torch.tensor([1e-13,-1e-13],dtype=torch.float64)
    o.global_step=1;o.moments={'shared_weight':dict(step=1,m=tiny.clone(),v=torch.ones(2,dtype=torch.float64))}
    m.shared_weight.grad=torch.zeros_like(m.shared_weight)
    info=o.step(probes=probes)
    assert info['norm_degenerate']==1 and info['applied_history_norm_ratio']==1.
    torch.testing.assert_close(o.moments['shared_weight']['m'],.9*tiny+.1e-4*torch.tensor([-.4,.3],dtype=torch.float64),atol=1e-18,rtol=1e-12)
