import copy
import numpy as np
import torch
from ..protocol import config, make_optimizer, update_ema, cpu_copy, equal, rng_state, state_digest
from ..rng import normal
from ..input_probe import embedding, sparse_fields, probe_logits, task_gradients, gain_gradients
from ..views_joint_label import donor_indices, joint_delta
from ..views_ema_field import null_basis, project, scale_gain, field_delta
from ..objectives import losses, source_losses, source_config, train_step
from ..vendor.gradient_aug_20261031_v2.views import MDMTRec
from ..vendor.gradient_aug_20261031_v2.vendor_ga31.basic.features import SparseFeature


def fixture(dtype=torch.float32):
    torch.set_num_threads(1); torch.manual_seed(2022)
    model = MDMTRec([SparseFeature(i, 16 if i < 3 else 3, embed_dim=16) for i in range(4)],
        domain_num=3, task_num=2, fcn_dims=[32, 16, 16, 8], expert_num=4,
        exp_d=1, exp_t=1, bal_d=1, bal_t=1).to(dtype=dtype)
    model.perturb_slices = [(0, 16), (16, 32), (32, 48)]
    x = torch.stack([torch.arange(12) % 16, (torch.arange(12)+2) % 16,
        (torch.arange(12)+4) % 16, torch.zeros(12)]).to(dtype=dtype)
    y = torch.tensor([[0, 0], [1, 0], [1, 1]]*4, dtype=dtype)
    rows = np.arange(100, 112, dtype=np.int64)
    context = dict(study='EMA02V2', dataset='movielens', split_id='original', seed=2022,
        epoch=1, global_step=1)
    return model, x, y, rows, context


def test_probe_complete_derivative_and_no_state_mutation():
    m, x, y, ids, ctx = fixture(torch.float64)
    original = state_digest(m.state_dict()); before_rng = state_digest(rng_state())
    h = embedding(m, x).requires_grad_(True)
    z = probe_logits(m, x, embedding_leaf=h)
    # Source uses FP32 output allocations; forward agreement is tested at its declared precision.
    m32 = copy.deepcopy(m).float()
    assert (m32(x.float())-probe_logits(m32, x.float()).sigmoid()).abs().max() < 2e-6
    direction = torch.randn_like(h); direction /= direction.norm()
    derivative = torch.autograd.grad(z[:, 0].sum(), h)[0].mul(direction).sum()
    eps = 1e-5
    numerical = (probe_logits(m, x, embedding_leaf=h.detach()+eps*direction)[:, 0].sum()
        -probe_logits(m, x, embedding_leaf=h.detach()-eps*direction)[:, 0].sum())/(2*eps)
    assert torch.allclose(derivative, numerical, rtol=1e-4, atol=1e-8)
    # Every model parameter stays a constant in the probe.
    assert all(p.grad is None for p in m.parameters())
    assert state_digest(m.state_dict()) == original
    after_random_direction = state_digest(rng_state())
    probe_logits(m, x, embedding_leaf=h)
    assert state_digest(rng_state()) == after_random_direction
    assert before_rng != after_random_direction


def test_row_rng_and_donors_and_common_radius():
    m, x, y, ids, ctx = fixture()
    before = state_digest(rng_state())
    noise = normal(ctx, ids, (3, 16), 'test', x)
    reverse = normal(ctx, ids[::-1].copy(), (3, 16), 'test', x)
    assert torch.equal(noise, reverse.flip(0))
    assert state_digest(rng_state()) == before
    donors = donor_indices(y.numpy(), ids, ctx)
    assert np.all(donors >= 0) and np.all(donors[:, 0] != donors[:, 1])
    assert np.all(donors != np.arange(12)[:, None])
    assert np.all(y.numpy()[donors] == y.numpy()[:, None])
    outputs = []
    for arm in ('J0', 'J1', 'J_SUM'):
        delta, info, count = joint_delta(m, x, y, ids, config(arm, 'movielens'), ctx, True)
        assert not delta.requires_grad and delta.grad_fn is None
        assert info['interpolation_max'] <= .200001
        outputs.append(delta.flatten(1).norm(dim=1))
    assert torch.allclose(outputs[0], outputs[1], atol=1e-8, rtol=1e-5)
    assert torch.allclose(outputs[1], outputs[2], atol=1e-8, rtol=1e-5)
    assert np.all(donor_indices(np.array([[0, 0]]), np.array([7]), ctx) == -1)


def test_joint_projection_cap_degeneracy_and_fr_energy():
    j = torch.tensor([[[0., 0., 0.], [0., 0., 0.]],
        [[1., 0., 0.], [2., 0., 0.]], [[1., 0., 0.], [0., 1., 0.]]])
    basis, normalized, ranks = null_basis(j)
    assert ranks.tolist() == [0, 1, 2]
    v = torch.randn(3, 3, dtype=torch.float64)
    projected = project(basis, v)
    assert torch.allclose(project(basis, projected), projected, atol=1e-12)
    assert torch.einsum('ntf,nf->nt', normalized, projected).abs().max() < 1e-12
    h = torch.ones(3, 3, 16)
    a, capped = scale_gain(h, projected, 2.)
    assert a.abs().max() <= .25+1e-14
    assert torch.einsum('ntf,nf->nt', normalized, a).abs().max() < 1e-12
    m, x, y, ids, ctx = fixture()
    ema = copy.deepcopy(m).requires_grad_(False)
    g, _ = gain_gradients(m, x, y); jac, _ = gain_gradients(ema, x)
    b, _, _ = null_basis(jac)
    assert (project(b, g).norm(dim=1)/g.double().norm(dim=1).clamp_min(1e-12)).max() < 1e-3
    f1, info, _ = field_delta(m, ema, x, y, ids, config('F1', 'movielens'), ctx)
    fr, fr_info, _ = field_delta(m, ema, x, y, ids, config('FR', 'movielens'), ctx)
    assert info['fallback_rows'] == len(y)
    assert fr_info['fr_energy_relative_error_max'] < 1e-5
    assert torch.allclose(f1.square().sum((1, 2)), fr.square().sum((1, 2)), rtol=1e-5, atol=1e-10)
    assert all(p.grad is None for p in m.parameters()) and all(p.grad is None for p in ema.parameters())


def test_source_closed_paths_and_task_budget():
    m, x, y, ids, ctx = fixture(); ema = copy.deepcopy(m).requires_grad_(False)
    for arm in ('A0', 'A1'):
        c = config(arm, 'movielens')
        actual, info, _, _ = losses(m, ema, x, y, ids, c, ctx)
        explicit = None
        if arm == 'A1':
            explicit = [dict(family='gaussian', noise=normal(ctx, ids, (3, 16), 'A1|gaussian', x))]
        expected, _ = source_losses(m, x, y, 0, source_config(arm == 'A1'), 'movielens', 2022, 1,
            mix_lambda=.5, kappa=.05, explicit_views=explicit)
        assert torch.equal(actual, expected)
        ga = torch.autograd.grad(actual, tuple(m.parameters()), allow_unused=True)
        gb = torch.autograd.grad(expected, tuple(m.parameters()), allow_unused=True)
        assert all(a is None and b is None or torch.equal(a, b) for a, b in zip(ga, gb))
    for arm in ('TG', 'RG'):
        scalar, info, _, count = losses(m, ema, x, y, ids, config(arm, 'movielens'), ctx)
        expected = (info['real'].sum()+.5*(info['augmented'][0][:, 0].sum()
            +info['augmented'][1][:, 1].sum()-info['real'].sum()))/y.numel()
        assert torch.equal(scalar, expected) and count['extra_view_rows'] == 2*len(y)
    disabled = config('F1', 'movielens'); disabled['kappa'] = 0.
    scalar, _, _, count = losses(m, ema, x, y, ids, disabled, ctx)
    expected, _ = source_losses(m, x, y, 0, source_config(False), 'movielens', 2022, 1)
    assert torch.equal(scalar, expected) and count['input_gradient_calls'] == 0
    empty = x.clone(); empty[-1] = 1
    _, _, _, count = losses(m, ema, empty, y, ids, config('F1', 'movielens'), ctx)
    assert count['target_rows'] == 0 and count['teacher_forwards'] == 0


def test_delta_training_derivative_adam_none_and_ema_order():
    from ..input_probe import add_delta
    m, x, y, ids, ctx = fixture()
    h = embedding(m, x).requires_grad_(True)
    delta = (sparse_fields(m, h)*.1).detach()
    assert torch.equal(torch.autograd.grad(add_delta(m, h, delta).sum(), h)[0], torch.ones_like(h))
    a = torch.nn.Parameter(torch.tensor([1.])); b = torch.nn.Parameter(torch.tensor([1.]))
    o = torch.optim.Adam([a, b], lr=.01, weight_decay=.1, foreach=False, fused=False)
    a.grad = torch.zeros_like(a); b.grad = None; o.step()
    assert a.item() < 1 and b.item() == 1 and a in o.state and b not in o.state
    ema = copy.deepcopy(m).requires_grad_(False); old = cpu_copy(ema.state_dict())
    o = make_optimizer(m, config('A0', 'movielens'))
    train_step(m, o, ema, x, y, ids, config('A0', 'movielens'), ctx)
    rho = config('A0', 'movielens')['rho']
    assert all(torch.equal(ema.state_dict()[k], old[k]*rho+m.state_dict()[k]*(1-rho))
        or torch.allclose(ema.state_dict()[k], old[k]*rho+m.state_dict()[k]*(1-rho), atol=1e-7)
        for k in old)
    # All registered mechanisms must support a single ordinary supervised update.
    for arm in ('A1', 'J0', 'J1', 'J_SUM', 'F0', 'F1', 'FR', 'F_PR', 'TG', 'RG', 'AG'):
        model = copy.deepcopy(m); shadow = copy.deepcopy(ema)
        result = train_step(model, make_optimizer(model, config(arm, 'movielens')), shadow,
            x, y, ids, config(arm, 'movielens'), ctx, diagnostic=True)
        assert np.isfinite(result['loss']) and result['counts']['optimizer_steps'] == 1
        assert result['counts']['ema_updates'] == 1
        assert all(p.grad is None for p in shadow.parameters())
