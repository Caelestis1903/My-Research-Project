import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from ..protocol import (protocol,setup,optimizer_for,arm_config,cpu_copy,equal,rng_state,
    snapshot,restore,stage_a_jobs,stage_b_jobs,choose_baselines,anti_gate,derive_seed)
from ..views import make_model,draw_view,apply_view
from ..objectives import losses
from ..probe import temporal_observation,variance_pair,linear


def synthetic(n=12):
    columns=('user_id','movie_id','gender','age','occupation','zip','cate_id','domain_indicator')
    schema=SimpleNamespace(columns=columns,dense_columns=('age',),vocab_sizes={name:8 for name in columns})
    x=np.zeros((n,8),np.float32);x[:,-1]=np.arange(n)%3;x[:,3]=.5
    y=(np.arange(n*2).reshape(n,2)%3==0).astype(np.float32)
    return SimpleNamespace(features=schema,train_x=x,train_y=y,train_row_ids=np.arange(n))


def batch(bundle,device='cpu'):
    return torch.as_tensor(bundle.train_x,device=device).T,torch.as_tensor(bundle.train_y,device=device)


def test_reference_views_budget_and_rng():
    setup(2022);p=protocol();bundle=synthetic();conf=p['datasets']['movielens'];x,y=batch(bundle)
    model=make_model(bundle,conf);reference=make_model(bundle,conf,reference=True);reference.load_state_dict(model.state_dict())
    assert list(model.state_dict())==list(reference.state_dict())
    for config in [arm_config(p,'B-ML','movielens'),dict(view_family='gaussian',num_views=2,pairing='iid')]:
        for kappa in (0.,):
            a=copy.deepcopy(model);b=copy.deepcopy(reference)
            value,info=losses(a,x,y,0,config,'movielens',2022,1,kappa=kappa)
            oracle=F.binary_cross_entropy(b(x),y,reduction='none').mean(1).mean()
            torch.testing.assert_close(value,oracle,atol=1e-6,rtol=1e-5)
            value.backward();oracle.backward()
            for left,right in zip(a.parameters(),b.parameters()):
                assert (left.grad is None)==(right.grad is None)
                if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=1e-6,rtol=1e-5)
            oa=optimizer_for(a,p,arm_config(p,'B-ML','movielens'));ob=optimizer_for(b,p,arm_config(p,'B-ML','movielens'))
            oa.step();ob.step()
            for left,right in zip(a.parameters(),b.parameters()):torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
    state=rng_state();embedding=model.embedding(x,model.features,squeeze_dim=True)
    view=draw_view(model,len(y),'gaussian','identity-key','cpu');negative=draw_view(model,len(y),'gaussian','unused','cpu',negative_of=view)
    assert torch.equal(view['noise'],-negative['noise'])
    assert equal(state,rng_state())
    modified=apply_view(embedding,model.perturb_slices,view)
    masked=draw_view(model,len(y),'fieldmask','mask-key','cpu');removed=apply_view(embedding,model.perturb_slices,masked)
    for name in ('age','domain_indicator'):
        lo,hi=model.field_slices[name]
        assert torch.equal(embedding[:,lo:hi],modified[:,lo:hi]) and torch.equal(embedding[:,lo:hi],removed[:,lo:hi])
    for i,index in enumerate(masked['fields']):
        lo,hi=model.perturb_slices[int(index)];assert torch.count_nonzero(removed[i,lo:hi])==0
    for n,target in [(1,0),(12,2),(12,8)]:
        b=synthetic(n);bx,by=batch(b)
        for arm in ('M1','G-IID2','G-ANTI2'):
            config=arm_config(p,arm,'movielens',{'movielens':'B-ML'})
            value,info=losses(model,bx,by,target,config,'movielens',2022,1)
            expected=info['real'].sum()
            if info['augmented']:expected+=.5*(sum(a.sum() for a in info['augmented'])/len(info['augmented'])-info['real'][info['mask']].sum())
            torch.testing.assert_close(value,expected/by.numel())
            disabled,_=losses(model,bx,by,target,config,'movielens',2022,1,mix_lambda=0)
            torch.testing.assert_close(disabled,F.binary_cross_entropy(model(bx),by))
    assert derive_seed('repeat')==derive_seed('repeat') and derive_seed('repeat')!=derive_seed('different')


def test_scale_mapping_none_and_ram_resume():
    setup(2022)
    a=torch.nn.Parameter(torch.tensor([.2,-.1,.6]));b=torch.nn.Parameter(a.detach().clone());unused_a=torch.nn.Parameter(torch.ones(1));unused_b=torch.nn.Parameter(torch.ones(1))
    oa=torch.optim.Adam([a,unused_a],lr=.01,weight_decay=1e-5,eps=1e-8,foreach=False,fused=False)
    ob=torch.optim.Adam([b,unused_b],lr=.01,weight_decay=1e-4,eps=1e-7,foreach=False,fused=False)
    for step in range(3):
        grad=torch.tensor([.1+step*.01,-.3,.00001]);a.grad=.1*grad;b.grad=grad
        oa.step();ob.step();torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(oa.state[a]['exp_avg']/.1,ob.state[b]['exp_avg'])
        torch.testing.assert_close(oa.state[a]['exp_avg_sq']/.01,ob.state[b]['exp_avg_sq'])
        assert float(oa.state[a]['step'])==float(ob.state[b]['step'])
    assert unused_a not in oa.state and unused_b not in ob.state
    p=protocol();bundle=synthetic();conf=p['datasets']['movielens'];config=arm_config(p,'B-ML','movielens');x,y=batch(bundle)
    model=make_model(bundle,conf);opt=optimizer_for(model,p,config);loaders={'train':torch.Generator().manual_seed(13022)}
    F.binary_cross_entropy(model(x),y).backward();opt.step();torch.randperm(19,generator=loaders['train'])
    state=snapshot(model,opt,loaders,1,1,{},'cpu');order=torch.randperm(19,generator=loaders['train'])
    opt.zero_grad(set_to_none=True);F.binary_cross_entropy(model(x),y).backward();opt.step();expected=cpu_copy((model.state_dict(),opt.state_dict()))
    clone=make_model(bundle,conf);co=optimizer_for(clone,p,config);cl={'train':torch.Generator()};restore(clone,co,cl,state,'cpu')
    assert torch.equal(order,torch.randperm(19,generator=cl['train']))
    F.binary_cross_entropy(clone(x),y).backward();co.step();assert equal(expected,(clone.state_dict(),co.state_dict()))
    assert linear((1,(None,)),(.5,(None,)))==(None,)


def trajectory(bundle,p,device,enabled,out=None):
    setup(2022,device);model=make_model(bundle,p['datasets']['movielens']).to(device)
    config=arm_config(p,'B-ML','movielens');opt=optimizer_for(model,p,config);loader={'train':torch.Generator().manual_seed(13022)}
    previous=None
    for step in range(1,5):
        torch.randperm(len(bundle.train_y),generator=loader['train']);opt.zero_grad(set_to_none=True)
        if enabled:previous=temporal_observation(model,opt,loader,bundle,'movielens',2022,1,step,step,device,out,previous)
        x,y=batch(bundle,device);loss,_=losses(model,x,y,0,config,'movielens',2022,step);(.1*loss).backward();opt.step()
    return cpu_copy((model.state_dict(),opt.state_dict(),[p.grad for p in model.parameters()],rng_state(device),loader['train'].get_state()))


def test_temporal_noninterference_and_variance():
    p=protocol();bundle=synthetic();assert equal(trajectory(bundle,p,'cpu',False),trajectory(bundle,p,'cpu',True))
    plus=[torch.tensor([float(i),2.*i],dtype=torch.float64) for i in range(4)]
    minus=[-v for v in plus];result=variance_pair(plus,minus)
    assert result['V_anti']==0 and result['V_iid']>0


def metric_record(auc):
    return dict(macro_auc=auc,macro_logloss=.5,d2t0_auc=.7,d2t1_auc=.7,pooled_task_mean_auc=.7,worst_cell_auc=.6)


def test_gates_and_budgets():
    p=protocol();rows=[]
    for seed in p['seeds']:
        for d,arm,_ in [j for j in stage_a_jobs(p) if j[2]==seed]:
            record=metric_record(.7 if arm!='L2-ML' else .6998)
            rows.append(dict(dataset=d,arm=arm,seed=seed,selected_validation=record))
    chosen=choose_baselines(rows,p);assert chosen['baselines']['movielens']=='L2-ML' and chosen['ml_practical_substitute']
    for dataset,expected_steps in [('none',56750),('movielens',61750),('kuairand',69000),('both',74000)]:
        gates={d:dict(triggered=dataset in (d,'both')) for d in p['datasets']}
        jobs=stage_a_jobs(p)+stage_b_jobs(p,gates)
        assert len(jobs)==35+5*sum(g['triggered'] for g in gates.values())
        assert sum(p['training']['optimizer_steps_per_run'][d] for d,a,s in jobs)==expected_steps
        assert all(len([j for j in jobs if j[0]==d and j[1]==a])==5 for d in p['datasets'] for a in ('M1','G-IID2'))
    observations=[]
    for d in p['datasets']:
        for s in p['seeds']:
            observations.extend([dict(dataset=d,seed=s,gate_anchor=True,valid_variance=True,V_anti=.1,V_iid=.1),
                                 dict(dataset=d,seed=s,gate_anchor=True,valid_variance=True,V_anti=.1,V_iid=1.)])
    gates=anti_gate(observations,p);assert all(g['triggered'] for g in gates.values())
    assert abs(gates['movielens']['mean_ratio']-2/11)<1e-12
    gates=anti_gate([r for r in observations if r['seed']!=2026],p);assert not any(g['triggered'] for g in gates.values())
