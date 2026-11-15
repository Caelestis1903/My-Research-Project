"""Compact checks of the interventions, numerical graph and RAM continuation."""
import copy
import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from .core import forward, sensitivity, sample, rng, make_view, train_step, weight_ledger
from .checks import check_sensitivity
from .protocol import config, optimizer, cpu_copy, state_digest, setup, rng_state, restore_rng
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.model import MDMTRec
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.basic.features import SparseFeature


def fixture():
    setup(2022,'cpu')
    model = MDMTRec([SparseFeature(0,20,embed_dim=4),SparseFeature(1,3,embed_dim=4)],
        domain_num=3,task_num=2,fcn_dims=[16,8,8,4],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    x = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12],[0,0,0,0,1,1,1,1,2,2,2,2]],dtype=torch.float32)
    y = torch.tensor([[0,0],[1,0],[0,1],[1,1]]*3,dtype=torch.float32)
    return model,x,y,dict(dataset='movielens',seed=2022,epoch=1,global_step=1)


def hooks(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


def test_sampler_support_and_independent_streams():
    _,x,y,context = fixture(); before = state_digest(rng_state('cpu'))
    rows,tasks,cells = sample(x[-1].numpy(),y.numpy(),context)
    assert np.array_equal(rows,sample(x[-1].numpy(),y.numpy(),context)[0])
    assert len(rows) == 24 and len(cells) == 12
    for i in range(0,len(rows),2):
        assert x[-1,rows[i]] == x[-1,rows[i+1]]
        assert y[rows[i],tasks[i]] == 1 and y[rows[i+1],tasks[i+1]] == 0
    examples,example_tasks,_ = sample(x[-1].numpy(),y.numpy(),context,True)
    assert len(examples) == 24
    assert not np.array_equal(rng(context,'pairs').random(10),rng(context,'examples').random(10))
    assert before == state_digest(rng_state('cpu'))
    assert sample(x[-1].numpy(),np.zeros_like(y.numpy()),context)[0].size == 0


@pytest.mark.parametrize('arm',['Z-PAIR','Z-EXAMPLE'])
def test_zero_view_objective_and_gradient_formula(arm):
    model,x,y,context = fixture(); c = config('movielens',arm)
    view,_,trace = make_view(model,x,y,arm,.03,context,np.arange(100,112))
    assert view is not None and torch.equal(view['delta'],torch.zeros_like(view['delta']))
    assert not view['delta'].requires_grad and hooks(model) == 0
    a = forward(model,view['x'],view['delta'])[0]; b = model(view['x'])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a,model(x)[view['rows']],rtol=1e-6,atol=1e-7)
    ids = torch.arange(len(view['tasks'])); tasks = view['tasks']
    auxiliary = 2*F.binary_cross_entropy(a[ids,tasks],view['y'][ids,tasks])
    expected_aux = 2*F.binary_cross_entropy(b[ids,tasks],y[view['rows'],tasks])
    torch.testing.assert_close(auxiliary,expected_aux,rtol=0,atol=0)
    clean = F.binary_cross_entropy(model(x),y); total = clean+.25*auxiliary
    parameters = tuple(model.parameters())
    actual = torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True)
    cg = torch.autograd.grad(clean,parameters,retain_graph=True,allow_unused=True)
    ag = torch.autograd.grad(auxiliary,parameters,retain_graph=True,allow_unused=True)
    for p,g,h,k in zip(parameters,actual,cg,ag):
        expected = (torch.zeros_like(p) if h is None else h)+.25*(torch.zeros_like(p) if k is None else k)
        if g is not None:
            torch.testing.assert_close(g,expected,rtol=1e-5,atol=1e-6)
    entries = F.binary_cross_entropy(a.detach()[ids,tasks],view['y'][ids,tasks],reduction='none')
    ledger = weight_ledger(x,y,model(x),view,entries,.25)
    assert ledger['M'] == len(tasks) == len(trace['accepted_row_ids'])
    assert ledger['auxiliary_to_clean_ratio'] == len(y)/len(tasks)
    assert sum(v['aux_count'] for v in ledger['cells'].values()) == len(tasks)
    assert sum(v['clean_count'] for v in ledger['cells'].values()) == 2*len(y)
    assert sum(v['aux_weight_mass'] for v in ledger['cells'].values()) == pytest.approx(.5)
    assert sum(v['clean_weight_mass'] for v in ledger['cells'].values()) == pytest.approx(1.)
    assert ledger['clean_numerator']/ledger['clean_denominator'] + .25*2*ledger['auxiliary_numerator']/ledger['M'] == pytest.approx(float(total.detach()))
    matched = 'B' if arm == 'Z-PAIR' else 'C'
    _,_,treated_trace = make_view(model,x,y,matched,.03,context,np.arange(100,112))
    for field in ('candidate_row_ids','candidate_tasks','candidate_labels','keep','cap'):
        assert trace[field] == treated_trace[field]


def test_empty_auxiliary_and_training_graph():
    model,x,y,context = fixture(); y = torch.zeros_like(y); reference = copy.deepcopy(model)
    ema = copy.deepcopy(model); ref_ema = copy.deepcopy(model); c = config('movielens','Z-PAIR')
    opt = optimizer(model,c); ref_opt = optimizer(reference,c)
    record = train_step(model,opt,ema,x,y,c,context)
    ref_opt.zero_grad(set_to_none=True); F.binary_cross_entropy(reference(x),y).backward(); ref_opt.step()
    update_ema(ref_ema,reference,c['rho'])
    for a,b in ((model,reference),(ema,ref_ema)):
        assert state_digest(a.state_dict()) == state_digest(b.state_dict())
    assert state_digest(opt.state_dict()) == state_digest(ref_opt.state_dict())
    assert record['auxiliary_loss'] == 0 and record['loss'] == record['clean_loss']
    assert record['ledger']['M'] == 0 and record['ledger']['total_weight_mass'] == 1
    assert record['optimizer_updates'] == record['ema_updates'] == 1
    for actual,expected in zip(model.parameters(),reference.parameters()):
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad,expected.grad,rtol=0,atol=0)
    plain = copy.deepcopy(model); wrapped = copy.deepcopy(model)
    F.binary_cross_entropy(plain(x),y).backward(); F.binary_cross_entropy(forward(wrapped,x)[0],y).backward()
    for a,b in zip(plain.parameters(),wrapped.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_probe_jacobian_finite_difference_and_hook_cleanup():
    model,x,_,_ = fixture(); check_sensitivity(model,x[:,:4])
    assert hooks(model) == 0 and all(p.grad is None for p in model.parameters())
    with pytest.raises(Exception):
        forward(model,x[:1],probe=True)
    assert hooks(model) == 0


def test_complete_ram_resume_and_selected_identity():
    model,x,y,context = fixture(); c = config('movielens','Z-EXAMPLE')
    ema = copy.deepcopy(model); opt = optimizer(model,c); loader = torch.Generator().manual_seed(13022)
    selected = {}; best = -float('inf'); counts = 0
    def epoch(number):
        nonlocal best,selected,counts
        order = torch.randperm(len(y),generator=loader)
        random.random(); np.random.random(); torch.rand(1)
        train_step(model,opt,ema,x[:,order],y[order],c,dict(context,epoch=number,global_step=number))
        score = float(model(x).mean().detach()); counts += 1
        if score > best:
            best = score; selected = dict(epoch=number,model=cpu_copy(model.state_dict()))
    epoch(1)
    state = cpu_copy(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),
        rng=rng_state('cpu'),loader=loader.get_state(),epoch=1,global_step=1,best=best,selected=selected,counts=counts))
    epoch(2)
    def signature():
        return state_digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),
            rng=rng_state('cpu'),loader=loader.get_state(),best=best,selected=selected,counts=counts))
    expected = signature()
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); ema.load_state_dict(state['ema'])
    loader.set_state(state['loader']); restore_rng(state['rng'],'cpu')
    best = state['best']; selected = state['selected']; counts = state['counts']
    epoch(state['epoch']+1)
    assert signature() == expected


def test_paired_arithmetic_and_empty_support_overlap():
    from .tables import paired, overlap
    assert paired([0.]*5)['sample_sd'] == 0
    assert paired([1.,2.,3.,4.,5.])['mean'] == 3
    trace = dict(candidate_row_ids=[],candidate_tasks=[],candidate_labels=[],candidate_domains=[],
                 accepted_row_ids=[],accepted_tasks=[])
    assert overlap(dict(support=trace),dict(support=trace))['jaccard'] == 1
