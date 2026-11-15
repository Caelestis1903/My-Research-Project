"""Small correctness checks for the scientific intervention and its lifecycle."""
import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from .core import forward, sensitivity, project, pair_deltas, sample, make_view, train_step
from .protocol import config, optimizer, cpu_copy, state_digest
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.model import MDMTRec
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.basic.features import SparseFeature


def fixture(dtype=torch.float32):
    torch.set_num_threads(1); torch.manual_seed(2022)
    model = MDMTRec([SparseFeature(0, 20, embed_dim=4), SparseFeature(1, 3, embed_dim=4)],
        domain_num=3, task_num=2, fcn_dims=[16,8,8,4], expert_num=4,
        exp_d=1, exp_t=1, bal_d=1, bal_t=1).to(dtype=dtype)
    x = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12], [0,0,0,0,1,1,1,1,2,2,2,2]], dtype=dtype)
    y = torch.tensor([[0,0],[1,0],[0,1],[1,1]]*3, dtype=dtype)
    context = dict(dataset='movielens', seed=2022, epoch=1, global_step=1)
    return model, x, y, context


def hook_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


def test_sampler_projection_and_cap():
    _, x, y, context = fixture()
    state = torch.get_rng_state().clone(); numpy_state = np.random.get_state()
    rows, tasks, cells = sample(x[-1].numpy(), y.numpy(), context)
    assert np.array_equal(rows, sample(x[-1].numpy(), y.numpy(), context)[0])
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert len(rows) == 24 and len(cells) == 12
    for i in range(0, len(rows), 2):
        assert x[-1, rows[i]] == x[-1, rows[i+1]]
        assert y[rows[i], tasks[i]] == 1 and y[rows[i+1], tasks[i+1]] == 0
    examples, target_tasks, _ = sample(x[-1].numpy(), y.numpy(), context, True)
    assert len(examples) == len(rows)
    q = project(torch.tensor([[1.,2.],[1.,2.],[1.,0.]]), torch.tensor([[1.,0.],[0.,0.],[1.,0.]]))
    assert torch.equal(q, torch.tensor([[0.,2.],[1.,2.],[0.,0.]]))
    e = torch.ones(4, 2); g = torch.tensor([[1.,2.]]*4); h = torch.tensor([[1.,0.]]*4)
    z = torch.tensor([.01, 0., 0., 0.])
    delta, valid, info = pair_deltas(e,z,g,h,.05,True)
    change = (g*delta).sum(1)
    assert abs(float(change[0]-change[1])) <= .010001
    assert valid.tolist() == [True, False] and torch.isfinite(delta).all()
    assert info['projection_residual_max'] == 0
    empty, _, _ = sample(x[-1].numpy(), np.ones_like(y.numpy()), context)
    assert len(empty) == 0


def test_full_probe_derivative_independence_and_finite_difference():
    model, x, _, _ = fixture(torch.float64)
    ordinary_p, ordinary_z, _ = forward(model, x)
    probe_p, probe_z, embedding = forward(model, x, probe=True)
    torch.testing.assert_close(probe_p, ordinary_p, rtol=0, atol=0)
    torch.testing.assert_close(probe_z, ordinary_z, rtol=0, atol=0)
    summed = torch.autograd.grad(probe_z[:,0].sum(), embedding, retain_graph=True)[0]
    for row in (0,4,8):
        alone = torch.autograd.grad(probe_z[row,0], embedding, retain_graph=True)[0]
        torch.testing.assert_close(summed[row], alone[row], rtol=1e-10, atol=1e-12)
        _, _, gradient = sensitivity(model, x[:,row:row+1])
        torch.testing.assert_close(summed[row], gradient[0,0], rtol=1e-10, atol=1e-12)
    direction = torch.randn_like(embedding); direction /= direction.norm(dim=1, keepdim=True)
    for t in range(2):
        derivative = torch.autograd.grad(probe_z[:,t].sum(), embedding, retain_graph=True)[0]
        with torch.no_grad():
            plus = forward(model, x, 1e-5*direction)[1][:,t]
            minus = forward(model, x, -1e-5*direction)[1][:,t]
        torch.testing.assert_close((plus-minus)/2e-5, (derivative*direction).sum(1), rtol=1e-3, atol=1e-7)
    assert hook_count(model) == 0 and all(p.grad is None for p in model.parameters())
    with pytest.raises(Exception):
        forward(model, x[:1], probe=True)
    assert hook_count(model) == 0


def test_training_detach_and_zero_lambda_equivalence():
    model, x, y, context = fixture()
    reference = copy.deepcopy(model); ema = copy.deepcopy(model); reference_ema = copy.deepcopy(model)
    c = config('movielens','A'); c['lambda_aug'] = 0.
    opt = optimizer(model,c); reference_opt = optimizer(reference,c)
    result = train_step(model,opt,ema,x,y,c,context)
    reference_opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy(reference(x),y).backward(); reference_opt.step()
    update_ema(reference_ema,reference,c['rho'])
    assert state_digest(model.state_dict()) == state_digest(reference.state_dict())
    assert state_digest(opt.state_dict()) == state_digest(reference_opt.state_dict())
    assert state_digest(ema.state_dict()) == state_digest(reference_ema.state_dict())
    assert result['optimizer_updates'] == result['ema_updates'] == 1
    plain = copy.deepcopy(model); wrapped = copy.deepcopy(model)
    F.binary_cross_entropy(plain(x),y).backward()
    F.binary_cross_entropy(forward(wrapped,x)[0],y).backward()
    for p,q in zip(plain.parameters(),wrapped.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_all_views_and_memory_continuation():
    model,x,y,context = fixture()
    for arm in ('A','B','C','RANDOM','GAUSSIAN','B_MATCH'):
        view, diag = make_view(model,x,y,arm,.03,context)
        assert view is not None and not view['delta'].requires_grad
        assert torch.isfinite(view['delta']).all() and hook_count(model) == 0
        if arm == 'A':
            z = forward(model,view['x'],view['delta'])[1]
            ids = torch.arange(len(view['tasks']))
            assert (z[ids,1-view['tasks']]-view['clean_logits'][ids,1-view['tasks']]).abs().max() <= .001001
    c = config('movielens','A'); ema = copy.deepcopy(model); opt = optimizer(model,c)
    train_step(model,opt,ema,x,y,c,context)
    snapshot = cpu_copy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict()))
    loader = torch.Generator().manual_seed(13022); order_state = loader.get_state().clone()
    first_order = torch.randperm(12,generator=loader)
    context['global_step'] = 2
    train_step(model,opt,ema,x,y,c,context)
    continued = state_digest(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict()))
    model.load_state_dict(snapshot['model']); ema.load_state_dict(snapshot['ema']); opt.load_state_dict(snapshot['optimizer'])
    loader.set_state(order_state); assert torch.equal(first_order,torch.randperm(12,generator=loader))
    train_step(model,opt,ema,x,y,c,context)
    assert state_digest(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict())) == continued


def test_five_seed_validation_gate():
    from .selection import pilot_decision, paired
    records = []
    for dataset in ('movielens','kuairand'):
        for seed in range(2022,2027):
            for arm,score in [('CLEAN',.70),('RANDOM',.701),('GAUSSIAN',.702),('A',.703),('B',.701),('C',.704)]:
                records.append(dict(job=dict(dataset=dataset,seed=seed,config=dict(arm=arm,alpha=.03)),
                    epochs=50,selected_validation=dict(fixed_ema=dict(macro_auc=score))))
    assert pilot_decision(records)['routes'] == ['C','A']
    with pytest.raises(AssertionError):
        pilot_decision(records[:-1])
    assert paired([0.,0.,0.,0.,0.])['ci95'] == [0.,0.]
