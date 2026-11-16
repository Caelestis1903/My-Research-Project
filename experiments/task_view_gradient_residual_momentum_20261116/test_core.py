"""Four CPU groups for the new numerical semantics; reuse the audited Jacobian."""
import copy
import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from . import core
from .core import ResidualChannel, train_step, norm, rademacher, support_ledger, view_losses
from .protocol import config, optimizer, cpu_copy, state_digest, rng_state, restore_rng, request
from .runner import ensure_guardian
from experiments.auxiliary_objective_attribution_20261115.test_core import fixture, hooks
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def test_a_natural_gradients_target_selection_detached_views():
    model, x, y, context = fixture()
    before = state_digest(rng_state('cpu'))
    deltas, _, _, _, _ = core.make_views(model, x, y, .03)
    assert before == state_digest(rng_state('cpu')) and hooks(model) == 0
    assert all(p.grad is None for p in model.parameters())
    assert all(not d.requires_grad for d in deltas)
    parameters = tuple(model.parameters())
    clean, view = view_losses(model, x, y, deltas)
    explicit_clean = F.binary_cross_entropy(model(x), y, reduction='sum')/(2*len(y))
    outputs = [core.forward(model, x, d)[0] for d in deltas]
    explicit_view = sum(F.binary_cross_entropy(outputs[t][:, t], y[:, t], reduction='sum')
                        for t in range(2))/(2*len(y))
    for actual, expected in ((clean, explicit_clean), (view, explicit_view)):
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
        eg = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
        for a, e in zip(ag, eg):
            assert (a is None) == (e is None)
            if a is not None:
                torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    assert norm(torch.autograd.grad(view, tuple(model.embedding.parameters()), retain_graph=True)) > 0
    differentiable_delta = torch.ones_like(deltas[0], requires_grad=True)
    value = core.forward(model, x, differentiable_delta)[0].sum()
    assert torch.autograd.grad(value, differentiable_delta, allow_unused=True)[0] is None
    # Tail batch and missing domains keep both tasks and every zero-direction entry.
    for B in (1, 3, len(y)):
        ledger = support_ledger(x[:, :B], torch.zeros_like(y[:B]))
        assert ledger['clean_denominator'] == ledger['view_denominator'] == sum(ledger['cells'].values()) == 2*B
        zeros = [torch.zeros_like(d[:B]) for d in deltas]
        c, v = view_losses(model, x[:, :B], y[:B], zeros)
        torch.testing.assert_close(c, v)


def toy():
    model = torch.nn.Module()
    model.register_parameter('a', torch.nn.Parameter(torch.tensor([1., -2.])))
    model.register_parameter('b', torch.nn.Parameter(torch.tensor([.5, 3.])))
    return model


def test_b_parameter_counters_none_reentry_weight_decay_and_instant():
    for arm in ('TV-GRM-010', 'TV-GRM-INSTANT-010'):
        model = toy(); c = config('movielens', arm); c['weight_decay'] = .2
        opt = optimizer(model, c); channel = ResidualChannel(model, opt, c)
        q = {}; steps = {'a': 0, 'b': 0}; m = {}; v = {}
        inputs = [([torch.tensor([.3, -.4]), None], [torch.tensor([.4, .2]), torch.ones(2)]),
                  ([torch.zeros(2), torch.tensor([.1, .5])], [None, torch.tensor([-.2, .4])]),
                  ([None, torch.tensor([-.3, .2])], [torch.ones(2), None]),
                  ([torch.tensor([.2, .1]), None], [torch.tensor([.7, -.1]), None])]
        for batch, (gc, gv) in enumerate(inputs, 1):
            old = {name: p.detach().clone() for name, p in model.named_parameters()}
            old_q = cpu_copy(channel.q); old_opt = cpu_copy(opt.state_dict())
            for (name, _), g, gview in zip(model.named_parameters(), gc, gv):
                if g is None:
                    continue
                steps[name] += 1; h = g+.2*old[name]
                m[name] = .9*m.get(name, torch.zeros_like(h))+.1*h
                v[name] = .999*v.get(name, torch.zeros_like(h))+.001*h.square()
                r = (torch.zeros_like(g) if gview is None else gview)-g
                q[name] = .9*q.get(name, torch.zeros_like(r))+.1*r
            info = channel.step(gc, gv, dict(dataset='movielens', seed=2022, epoch=1, batch=batch), True)
            assert info['view_only_parameters'] == sum(g is None and vv is not None for g, vv in zip(gc, gv))
            for index, ((name, p), g) in enumerate(zip(model.named_parameters(), gc)):
                if g is None:
                    assert torch.equal(p, old[name])
                    if name in old_q:
                        assert torch.equal(channel.q[name], old_q[name])
                    if index in old_opt['state']:
                        assert state_digest(opt.state[p]) == state_digest(old_opt['state'][index])
                else:
                    state = opt.state[p]
                    assert int(state['step']) == steps[name]
                    torch.testing.assert_close(state['exp_avg'], m[name])
                    torch.testing.assert_close(state['exp_avg_sq'], v[name])
                    if arm == 'TV-GRM-010':
                        torch.testing.assert_close(channel.q[name], q[name])
                    else:
                        assert not channel.q
                        residual = (torch.zeros_like(g) if gv[index] is None else gv[index])-g
                        denominator = (v[name]/(1-.999**steps[name])).sqrt()+1e-8
                        uc = (m[name]/(1-.9**steps[name]))/denominator
                        expected = old[name]-.01*uc-.01*info['cap_scale']*residual/denominator
                        torch.testing.assert_close(p, expected, rtol=1e-5, atol=2e-7)
        assert steps == {'a': 3, 'b': 2}


def test_c_global_cap_shuffle_update_norm_and_independent_rng():
    context = dict(dataset='movielens', seed=2022, epoch=1, batch=1)
    gc = [torch.tensor([1e-3, 5.]), torch.tensor([.4, 1e-2])]
    gv = [torch.tensor([10., -10.]), torch.tensor([-4., 3.])]
    records = []; injected = []
    for arm in ('TV-GRM-010', 'TV-GRM-SHUFFLE-010'):
        model = toy(); c = config('movielens', arm); opt = optimizer(model, c)
        channel = ResidualChannel(model, opt, c)
        plain = copy.deepcopy(model); source = optimizer(plain, c)
        for p, g in zip(plain.parameters(), gc):
            p.grad = g.clone()
        source.step()
        before = state_digest(rng_state('cpu')); info = channel.step(gc, gv, context)
        assert before == state_digest(rng_state('cpu'))
        actual = norm([(a-b)/c['lr'] for a, b in zip(plain.parameters(), model.parameters())])
        assert actual == pytest.approx(info['residual_injection_norm'], rel=2e-4)
        assert info['residual_injection_norm'] <= .1*info['clean_update_norm']*(1+1e-5)
        records.append(info); injected.append(actual)
    assert records[1]['shuffle_relative_error'] <= 1e-5
    assert records[0]['residual_injection_norm'] == pytest.approx(records[1]['residual_injection_norm'], rel=1e-6)
    p = torch.zeros(128)
    assert torch.equal(rademacher(p, context, 'a'), rademacher(p, context, 'a'))
    assert not torch.equal(rademacher(p, context, 'a'), rademacher(p, dict(context, batch=2), 'a'))
    assert not torch.equal(rademacher(p, context, 'a'), rademacher(p, context, 'b'))
    for gradients in ([torch.zeros(2), torch.zeros(2)], gc):
        model = toy(); c = config('movielens', 'TV-GRM-SHUFFLE-010'); c['weight_decay'] = 0.
        info = ResidualChannel(model, optimizer(model, c), c).step(gradients, gradients, context)
        assert info['residual_injection_norm'] == info['shuffle_target_norm'] == 0.
    # C=0 with nonzero residual also injects exact zero.
    model = toy(); c['weight_decay'] = 0.
    info = ResidualChannel(model, optimizer(model, c), c).step([torch.zeros(2)]*2, gv, context)
    assert info['clean_update_norm'] == info['residual_injection_norm'] == 0.


def test_d_gamma0_exact_source_and_ram_restore(tmp_path, monkeypatch):
    initial, x, y, base_context = fixture()
    ensure_guardian(tmp_path)
    try:
        for gamma0 in (True, False):
            key = 'gamma0' if gamma0 else 'residual'
            request(tmp_path, 'register', key=key, job=dict(cpu_fixture=key), pid=0)
            model = copy.deepcopy(initial); ema = copy.deepcopy(initial)
            c = config('movielens', 'CLEAN' if gamma0 else 'TV-GRM-010')
            opt = optimizer(model, c); channel = ResidualChannel(model, opt, c)
            ref = copy.deepcopy(initial); ref_ema = copy.deepcopy(initial); ref_opt = optimizer(ref, c)
            loader = torch.Generator().manual_seed(13022); selected = {}; best = -float('inf'); count = 0
            real_views = core.make_views
            def forbidden(*args, **kwargs):
                raise AssertionError('gamma0 must not probe')
            if gamma0:
                monkeypatch.setattr(core, 'make_views', forbidden)
            def epoch(number):
                nonlocal selected, best, count
                order = torch.randperm(len(y), generator=loader)
                random.random(); np.random.random(); torch.rand(1)
                before_rng = state_digest(rng_state('cpu'))
                train_step(model, opt, channel, ema, x[:, order], y[order], c,
                           dict(base_context, epoch=number, batch=1, global_step=number))
                assert before_rng == state_digest(rng_state('cpu'))
                if gamma0:
                    ref_opt.zero_grad(set_to_none=True)
                    F.binary_cross_entropy(ref(x[:, order]), y[order]).backward()
                    ref_opt.step(); update_ema(ref_ema, ref, c['rho'])
                    assert state_digest(model.state_dict()) == state_digest(ref.state_dict())
                    assert state_digest(ema.state_dict()) == state_digest(ref_ema.state_dict())
                    assert state_digest(opt.state_dict()) == state_digest(ref_opt.state_dict()) and not channel.q
                count += 1; score = float(ema(x).mean().detach())
                if score > best:
                    best = score; selected = dict(epoch=number, model=cpu_copy(ema.state_dict()))
            def snapshot():
                return cpu_copy(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                    residual=channel.state_dict(), loader=loader.get_state(), rng=rng_state('cpu'),
                    selected=selected, best=best, count=count, ref=ref.state_dict(), ref_ema=ref_ema.state_dict(),
                    ref_opt=ref_opt.state_dict(), epoch=count, counts=dict(optimizer_updates=count, ema_updates=count)))
            epoch(1)
            request(tmp_path, 'commit', key=key, state=snapshot(), epoch_file='cpu_fixture_epoch_01')
            epoch(2); epoch(3); expected = state_digest(snapshot())
            saved = request(tmp_path, 'get', key=key)['state']
            model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema']); opt.load_state_dict(saved['optimizer'])
            channel.load_state_dict(saved['residual']); loader.set_state(saved['loader']); restore_rng(saved['rng'], 'cpu')
            selected = saved['selected']; best = saved['best']; count = saved['count']
            ref.load_state_dict(saved['ref']); ref_ema.load_state_dict(saved['ref_ema']); ref_opt.load_state_dict(saved['ref_opt'])
            epoch(2); epoch(3)
            assert state_digest(snapshot()) == expected
            monkeypatch.setattr(core, 'make_views', real_views)
            request(tmp_path, 'paused', key=key, reason='CPU_FIXTURE_COMPLETE')
            request(tmp_path, 'release', key=key)
    finally:
        request(tmp_path, 'shutdown')
    assert not list(tmp_path.rglob('*.pt')) and not list(tmp_path.rglob('*.pkl'))


def test_explicit_recovery_reuses_committed_epoch_and_rejects_missing_state(tmp_path):
    ensure_guardian(tmp_path)
    try:
        payload = dict(epoch=1, counts=dict(optimizer_updates=20, ema_updates=20), value=torch.tensor([1., 2.]))
        for key in ('failed', 'dead'):
            request(tmp_path, 'register', key=key, job=dict(cpu_fixture=key), pid=0)
            request(tmp_path, 'commit', key=key, state=payload, epoch_file='cpu_fixture_epoch_01')
        request(tmp_path, 'failed', key='failed', failure=dict(error='repaired CPU fixture'))
        (tmp_path/'PAUSE_TRAINING').touch()
        with pytest.raises(RuntimeError):
            request(tmp_path, 'register', key='failed', job=dict(cpu_fixture='failed'), pid=0)
        event = request(tmp_path, 'recover', key='failed')
        assert event['unresolved'] == ['dead'] and (tmp_path/'PAUSE_TRAINING').exists()
        event = request(tmp_path, 'recover', key='dead')
        assert not event['unresolved'] and not (tmp_path/'PAUSE_TRAINING').exists()
        for key in ('failed', 'dead'):
            registration = request(tmp_path, 'register', key=key, job=dict(cpu_fixture=key), pid=0)
            assert registration['attempt'] == 2 and state_digest(registration['state']) == state_digest(payload)
            request(tmp_path, 'paused', key=key, reason='CPU_FIXTURE_COMPLETE')
            request(tmp_path, 'release', key=key)
        # A released committed state must never be silently restarted from epoch zero.
        request(tmp_path, 'failed', key='failed', failure=dict(error='missing RAM fixture'))
        with pytest.raises(RuntimeError, match='Committed RAM state lost'):
            request(tmp_path, 'recover', key='failed')
    finally:
        request(tmp_path, 'shutdown')
