import copy
import numpy as np
import pytest
import torch
from experiments.ema_causal_20261110.history import ScoreEMA,SparseHistory,update_ideal_R
from experiments.ema_causal_20261110.interventions import build_state,matched_states
from experiments.ema_causal_20261110.pairs import exact_transition_table,transition_metrics,sample_pairs
from experiments.ema_causal_20261110.functional import (read_predictions,capture_cut,mixed_readout,
    capture_fusion_parts,fusion_components,fusion_part_readout,integrate_path,rank_gradient_summary)
from experiments.aug_history_20261105.registry import ParameterLayout
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.reference import MDMTRec
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.basic.features import SparseFeature

torch.set_num_threads(1)


def test_history_weights_windows_and_splice():
    rho=.8;theta=torch.zeros(7,dtype=torch.float32);fixed=theta.clone();ideal=theta.double()
    history=SparseHistory([4,8],[2,4,10],rho);history.capture(0,ideal,{'p':np.zeros(3)})
    q=ScoreEMA(rho,{'p':np.zeros(3)});all_states=[ideal.clone()];scores=[np.zeros(3)]
    for step in range(1,9):
        before=theta.clone();theta+=(-1)**step*.15
        fixed.mul_(rho).add_(theta,alpha=1-rho);ideal=update_ideal_R(ideal,before,theta,rho)
        actual=fixed.double()-theta.double();history.capture(step,actual,{'p':np.full(3,step)})
        q.update({'p':np.full(3,step)});scores.append(np.full(3,step));all_states.append(actual)
        if step in (4,8):
            old=history.consume(step)
            for h,item in old.items():
                expected=rho**h*all_states[step-h]
                value=torch.zeros_like(actual) if item['old_R'] is None else item['old_R']
                torch.testing.assert_close(expected,value,rtol=0,atol=0)
                torch.testing.assert_close(actual,(actual-value)+value,rtol=1e-10,atol=1e-14)
            packet=dict(online_fp32=theta,F_fp32=fixed,R_fp64=actual,old_R_fp64_by_effective_h=old)
            mask=torch.arange(7)%2==0
            value,_=build_state(packet,dict(group='S'),{'S':mask})
            assert torch.equal(value[mask],fixed[mask]) and torch.equal(value[~mask],theta[~mask])
    expanded=rho**8*scores[0]+sum((1-rho)*rho**(8-k)*scores[k] for k in range(1,9))
    np.testing.assert_allclose(q.value['p'],expanded,rtol=1e-14)
    assert not history.records


def test_exact_ties_against_bruteforce():
    rng=np.random.default_rng(2022)
    for n in (2,7,50,200):
        for _ in range(10):
            y=rng.integers(0,2,n);a=rng.integers(0,4,n)/3;b=rng.integers(0,4,n)/3
            expected=np.zeros((3,3),np.int64)
            for i in np.flatnonzero(y==1):
                for j in np.flatnonzero(y==0):expected[int(np.sign(a[i]-a[j]))+1,int(np.sign(b[i]-b[j]))+1]+=1
            np.testing.assert_array_equal(exact_transition_table(y,a,b),expected)
    y=np.array([0,0,1,1]);a=y.copy();b=1-y
    table=exact_transition_table(y,a,b);assert table[2,0]==4
    assert transition_metrics(table)['delta_auc']==-1
    assert exact_transition_table(y,a,a).trace()==4
    pairs=sample_pairs(np.tile([0,1],100),2022,1000)
    assert len(pairs)==len(np.unique(pairs,axis=0))==1000
    np.testing.assert_array_equal(pairs,sample_pairs(np.tile([0,1],100),2022,1000))


@pytest.fixture
def graph():
    torch.manual_seed(2022)
    model=MDMTRec([SparseFeature(0,7,embed_dim=4),SparseFeature(1,3,embed_dim=4)],3,2,[16,8,8,4],4,1,1,1,1)
    x=torch.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]])
    return model,x


def test_complete_cuts_fusion_and_derivative(graph):
    model,x=graph;layout=ParameterLayout.from_model(model);a=layout.flatten(model.parameters()).clone()
    b=a+torch.randn_like(a)*.0001
    for vector in (a,b):
        layout.copy_to(model,vector);original=read_predictions(model,x)
        for cut in ('h','fused'):
            captured=capture_cut(model,x,cut);mixed=mixed_readout(model,x,cut,captured)
            assert torch.equal(mixed.p,original.p) and torch.equal(mixed.z,original.z)
    layout.copy_to(model,a);first=capture_fusion_parts(model,x)
    layout.copy_to(model,b);second=capture_fusion_parts(model,x)
    components,residuals=fusion_components(first,second)
    assert max(row['algebra_relative_l2'] for row in residuals)<1e-10
    zero={k:torch.zeros_like(v) for k,v in second['actual_fused'].items()}
    assert torch.equal(fusion_part_readout(model,x,second,zero).p,read_predictions(model,x).p)
    pairs=np.array([[0,1],[2,3],[4,5]])
    result=integrate_path(model,layout,x,a,b,pairs=pairs,task=0)
    assert result['method']=='jvp' and not result['sign_unresolved'].any()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    masks={'all':torch.ones_like(a,dtype=torch.bool)}
    summary=rank_gradient_summary(model,layout,x,[(np.array([0,2,4]),np.array([1,3,5]),0)],masks)
    assert summary['gradient_rms']['all']>0


@pytest.mark.parametrize('dataset',['movielens','kuairand'])
def test_real_source_support_and_complete_cuts(dataset):
    import json
    from pathlib import Path
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.aug_history_20261105.model_adapter import build_model
    from experiments.aug_history_20261105.group_registry import build_path_groups,GROUPS
    from experiments.ema_causal_20261110.interventions import group_masks
    from experiments.ema_causal_20261110.functional import star_cross_readout
    root=Path(__file__).resolve().parents[3]
    config=json.loads((root/'research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json').read_text())['configs'][dataset]
    cache=Path('/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache')
    bundle=cached_bundle(cache,dataset,include_test=False)
    model=build_model(bundle,config);layout=ParameterLayout.from_model(model)
    registry=build_path_groups(model,bundle);masks=group_masks(registry)
    rows=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:4] for d in range(3)])
    x=torch.tensor(np.array(bundle.train_x[rows])).T;theta=layout.flatten(model.parameters()).clone()
    original=read_predictions(model,x)
    for cut in ('h','fused'):
        mixed=mixed_readout(model,x,cut,capture_cut(model,x,cut))
        assert torch.equal(mixed.p,original.p)
    for full in (False,True):
        logits=read_predictions(model,x,full_functional_graph=full).z
        for d in range(3):
            for t in range(2):
                grad=torch.autograd.grad(logits[x[-1]==d,t].sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(model.parameters(),grad)])
                supported=masks['S']|masks[f'D{d}']|masks[f'T{t}']|masks[f'DT{d}{t}']
                assert not bool(flat[~supported].ne(0).any())
    assert torch.equal(star_cross_readout(model,layout,x,theta,theta).p,original.p)
    parts=capture_fusion_parts(model,x)
    assert parts['gate'][0].shape==(len(rows),4)
    components,residuals=fusion_components(parts,parts)
    assert all(row['algebra_relative_l2']==0 for row in residuals)


def test_full_function_directional_derivative(graph):
    model,x=graph;layout=ParameterLayout.from_model(model)
    theta=layout.flatten(model.parameters());direction=torch.randn_like(theta);direction/=direction.norm()
    def f(v):
        params={n:v[a:b].view(shape) for n,shape,(a,b) in zip(layout.names,layout.shapes,layout.offsets)}
        return read_predictions(model,x,params=params,full_functional_graph=True).z
    derivative=torch.func.jvp(f,(theta,),(direction,))[1]
    epsilon=.001;finite=(f(theta+epsilon*direction)-f(theta-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(derivative,finite,atol=5e-4,rtol=.02)
