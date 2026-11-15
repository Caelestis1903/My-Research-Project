"""Compact tests of the new natural mixture and canonical zero counterpart."""
import copy
import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from . import core
from .core import make_views,losses,weight_ledger,train_step
from .protocol import config,optimizer,cpu_copy,state_digest,rng_state,restore_rng,evaluate
from .runner import zero_equivalence
from ..test_core import fixture
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def test_explicit_natural_mixture_gradients_and_task_views():
    model,x,y,context = fixture(); c = config('movielens')
    before = state_digest(rng_state('cpu'))
    deltas,_,_,_,_ = make_views(model,x,y,c['alpha'])
    assert before == state_digest(rng_state('cpu')) and all(p.grad is None for p in model.parameters())
    assert all(not delta.requires_grad and torch.isfinite(delta).all() for delta in deltas)
    total,clean,view_loss,probability,views,canonical = losses(model,x,y,deltas,.25)
    assert not canonical
    clean_entries = F.binary_cross_entropy(probability,y,reduction='none')
    view_entries = torch.stack([F.binary_cross_entropy(views[t][0][:,t],y[:,t],reduction='none') for t in range(2)],dim=1)
    explicit = (.75*clean_entries+.25*view_entries).sum()/(2*len(y))
    torch.testing.assert_close(total,explicit,rtol=1e-6,atol=1e-7)
    parameters = tuple(model.parameters())
    actual = torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True)
    expected = torch.autograd.grad(explicit,parameters,retain_graph=True,allow_unused=True)
    for a,b in zip(actual,expected):
        if a is not None:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    ledger = weight_ledger(x,y,probability,views,.25)
    assert ledger['entries'] == 2*len(y) and ledger['total_weight_mass'] == 1
    assert ledger['clean_weight_mass'] == .75 and ledger['view_weight_mass'] == .25
    for cell,v in ledger['cells'].items():
        assert v['clean_count'] == v['view_count'] == v['count']
        assert v['total_weight_mass'] == v['count']/(2*len(y))
        assert v['clean_weight_mass']+v['view_weight_mass'] == pytest.approx(v['total_weight_mass'])


def test_zero_explicit_mixture_and_exact_clean_step():
    model,x,y,context = fixture(); c = config('movielens')
    deltas,_,_,_,_ = make_views(model,x,y,.03,force_zero=True)
    _,_,_,probability,_,canonical = losses(model,x,y,deltas,.25)
    assert canonical and all(torch.equal(d,torch.zeros_like(d)) for d in deltas)
    clean = F.binary_cross_entropy(model(x),y)
    p0 = core.forward(model,x,deltas[0])[0]; p1 = core.forward(model,x,deltas[1])[0]
    torch.testing.assert_close(p0,probability,rtol=0,atol=0)
    torch.testing.assert_close(p1,probability,rtol=0,atol=0)
    explicit = .75*F.binary_cross_entropy(probability,y)+.25*.5*(
        F.binary_cross_entropy(p0[:,0],y[:,0])+F.binary_cross_entropy(p1[:,1],y[:,1]))
    torch.testing.assert_close(explicit,clean,rtol=1e-6,atol=1e-7)
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(explicit,parameters,retain_graph=True,allow_unused=True)
    b = torch.autograd.grad(clean,parameters,allow_unused=True)
    for aa,bb in zip(a,b):
        if aa is not None:
            torch.testing.assert_close(aa,bb,rtol=1e-5,atol=1e-6)
    assert zero_equivalence(model,x,y,c,context)['status'] == 'PASS'


def test_short_missing_domain_single_label_and_zero_direction(monkeypatch):
    model,x,y,context = fixture(); c = config('movielens')
    embedding,logits,gradient = core.sensitivity(model,x)
    monkeypatch.setattr(core,'sensitivity',lambda model_,x_: (embedding[:len(x_.T)],logits[:len(x_.T)],torch.zeros_like(gradient[:len(x_.T)])))
    for B in (3,1):
        local_x = x[:,:B]; local_y = torch.zeros_like(y[:B])
        m = copy.deepcopy(model); ema = copy.deepcopy(model); opt = optimizer(m,c)
        value = train_step(m,opt,ema,local_x,local_y,c,context)
        assert value['zero_delta_canonical'] and value['ledger']['entries'] == 2*B
        assert value['ledger']['denominator'] == 2*B and value['ledger']['total_entry_coefficient'] == 1/(2*B)
        assert sum(v['count'] for v in value['ledger']['cells'].values()) == 2*B
        assert value['optimizer_updates'] == value['ema_updates'] == 1
        for task in range(2):
            assert value['diagnostics'][f'task{task}']['zero_count'] == B


@pytest.mark.parametrize('force_zero',[False,True])
def test_ram_continuation_selected_identity_and_zero_source_trajectory(force_zero):
    model,x,y,context = fixture(); c = config('movielens')
    ema = copy.deepcopy(model); opt = optimizer(model,c)
    reference = copy.deepcopy(model); reference_ema = copy.deepcopy(model); ref_opt = optimizer(reference,c)
    loader = torch.Generator().manual_seed(13022); counts = 0
    best = {'online':-float('inf'),'fixed_ema':-float('inf')}; selected = {}; selected_epochs = {}
    reference_best = dict(best); reference_selected = {}; reference_epochs = {}
    def epoch(number):
        nonlocal counts
        order = torch.randperm(len(y),generator=loader)
        random.random(); np.random.random(); torch.rand(1)
        train_step(model,opt,ema,x[:,order],y[order],c,dict(context,epoch=number,global_step=number),force_zero=force_zero)
        counts += 1
        if force_zero:
            ref_opt.zero_grad(set_to_none=True); F.binary_cross_entropy(reference(x[:,order]),y[order]).backward()
            ref_opt.step(); update_ema(reference_ema,reference,c['rho'])
            for a,b in ((model,reference),(ema,reference_ema)):
                assert state_digest(a.state_dict()) == state_digest(b.state_dict())
            assert state_digest(opt.state_dict()) == state_digest(ref_opt.state_dict())
        for kind,estimator in (('online',model),('fixed_ema',ema)):
            values,_ = evaluate(estimator,x.T.numpy(),y.numpy(),'cpu')
            if values['macro_auc'] > best[kind]:
                best[kind] = values['macro_auc']; selected_epochs[kind] = number
                selected[kind] = cpu_copy(estimator.state_dict())
            if force_zero:
                ref = reference if kind == 'online' else reference_ema
                ref_values,_ = evaluate(ref,x.T.numpy(),y.numpy(),'cpu')
                assert values == ref_values
                if ref_values['macro_auc'] > reference_best[kind]:
                    reference_best[kind] = ref_values['macro_auc']; reference_epochs[kind] = number
                    reference_selected[kind] = cpu_copy(ref.state_dict())
    epoch(1)
    state = cpu_copy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),rng=rng_state('cpu'),
        loader=loader.get_state(),epoch=1,global_step=1,best=best,selected=selected,selected_epochs=selected_epochs,counts=counts,
        reference=reference.state_dict(),reference_ema=reference_ema.state_dict(),ref_opt=ref_opt.state_dict(),
        reference_best=reference_best,reference_selected=reference_selected,reference_epochs=reference_epochs))
    epoch(2)
    def signature():
        return state_digest(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            rng=rng_state('cpu'),loader=loader.get_state(),best=best,selected=selected,selected_epochs=selected_epochs,counts=counts))
    expected = signature()
    model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); opt.load_state_dict(state['optimizer'])
    restore_rng(state['rng'],'cpu'); loader.set_state(state['loader'])
    best = state['best']; selected = state['selected']; selected_epochs = state['selected_epochs']; counts = state['counts']
    reference.load_state_dict(state['reference']); reference_ema.load_state_dict(state['reference_ema']); ref_opt.load_state_dict(state['ref_opt'])
    reference_best = state['reference_best']; reference_selected = state['reference_selected']; reference_epochs = state['reference_epochs']
    epoch(state['epoch']+1)
    assert signature() == expected
    if force_zero:
        assert selected_epochs == reference_epochs and state_digest(selected) == state_digest(reference_selected)
