import copy
from types import SimpleNamespace
import numpy as np
import torch
from experiments.mdmt_history_information_20261103.protocol import cpu_copy, equal
from experiments.mdmt_history_information_20261103.optimizers import HistoryOptimizer, project_history
from experiments.conditional_information_momentum_20261102_v3.optimizers import PairOptimizer
from experiments.overnight_ab_20261103 import objectives as objective
from experiments.overnight_ab_20261103.optimizers import SerialCPOptimizer


def config(arm=objective.CP):
    return dict(arm=arm, dataset='movielens', lr=.01, weight_decay=.01, rho=.95, epochs=50,
                batch_size=40960, eval_batch_size=40960, betas=[.9, .999], eps=1e-8,
                c_mix=dict(tau=.1, kappa=1., gamma=.2, probe_domains_per_step=1),
                p01=dict(lambda_pair=.05, dynamic=False, pairs_per_step=2048,
                         history_alpha=.01, history_bias_beta=.99, mode='eccm'),
                gaussian=dict(lambda_aug=.5, kappa=.05, target_domain=0))


class Vectors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
        self.private_weight = torch.nn.Parameter(torch.tensor([.2, .9], dtype=torch.float64))


def pair():
    return dict(active=[2, 3], selected=2, index=0, w=np.array([.2, .3]), q=np.array([.4, .6]))


def test_cp_native_clocks_moments_single_update_and_decay():
    model = Vectors()
    opt = SerialCPOptimizer(model, objective.native_config(config()))
    old, private = model.shared_weight.detach().clone(), model.private_weight.detach().clone()
    mixed_gradient = torch.tensor([-.8, .6], dtype=torch.float64)
    private_gradient = torch.tensor([.7, -.2], dtype=torch.float64)
    r = torch.tensor([.4, -.6], dtype=torch.float64)
    model.shared_weight.grad = mixed_gradient.clone()
    model.private_weight.grad = private_gradient.clone()
    control = torch.nn.Parameter(private.clone())
    control.grad = private_gradient.clone()
    ordinary = torch.optim.Adam([control], lr=.01, weight_decay=.01, foreach=False, fused=False)
    ordinary.step()
    b = mixed_gradient + .01 * old
    u = b - .05 * .5 * r
    v = .001 * b.square()
    m = .1 * u
    h = .01 / .4 * r
    H = .05 * .2 * h / (1 - .99)
    d = ((v / (1 - .999)).sqrt() + 1e-8).reciprocal()
    expected = old - .01 * d * (m / (1 - .9) + H)
    info = opt.step(pair(), [r], probes=[], diagnostic=True)
    torch.testing.assert_close(model.shared_weight, expected, rtol=1e-14, atol=1e-14)
    assert torch.equal(model.private_weight, control)
    assert torch.equal(model.shared_weight.grad, mixed_gradient)
    torch.testing.assert_close(opt.moments['shared_weight']['v'], v, rtol=0, atol=0)
    torch.testing.assert_close(opt.moments['shared_weight']['m'], m, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(opt.history[2][0], h)
    assert opt.active == {2: 1, 3: 1} and opt.selected == {2: 1}
    assert torch.count_nonzero(opt.history[3][0]) == 0
    assert opt.global_step == 1 and opt.moments['shared_weight']['step'] == 1
    assert info['constraints'] == 0


def test_cp_projects_real_term_but_v_stays_mixed():
    model = Vectors()
    c = config()
    c['weight_decay'] = 0.
    opt = SerialCPOptimizer(model, objective.native_config(c))
    mixed = torch.tensor([-1., .4], dtype=torch.float64)
    r = torch.tensor([.6, -.3], dtype=torch.float64)
    model.shared_weight.grad = mixed.clone()
    model.private_weight.grad = torch.zeros_like(model.private_weight)
    probes = [([torch.tensor([1., 0.], dtype=torch.float64)],
               [torch.tensor([1., 0.], dtype=torch.float64)])]
    u = mixed - .05 * .5 * r
    d = (mixed.abs() + 1e-8).reciprocal()
    repaired, _ = project_history([u], [d], probes, 1., .2)
    info = opt.step(pair(), [r], probes, diagnostic=True)
    torch.testing.assert_close(opt.moments['shared_weight']['m'], .1 * repaired[0])
    torch.testing.assert_close(opt.moments['shared_weight']['v'], .001 * mixed.square())
    assert info['constraints'] == 2 and info['projection_distance'] > 0


def test_pair_none_ignores_preserved_history_and_restore_is_exact():
    model = Vectors()
    opt = SerialCPOptimizer(model, objective.native_config(config()))
    model.shared_weight.grad = torch.tensor([-.8, .6], dtype=torch.float64)
    model.private_weight.grad = torch.tensor([.7, -.2], dtype=torch.float64)
    opt.step(pair(), [torch.tensor([.4, -.6], dtype=torch.float64)], [])
    saved_model, saved_opt = cpu_copy(model.state_dict()), cpu_copy(opt.state_dict())
    clone = Vectors()
    clone.load_state_dict(saved_model)
    restored = SerialCPOptimizer(clone, objective.native_config(config()))
    restored.load_state_dict(saved_opt)
    for current, current_opt in ((model, opt), (clone, restored)):
        old = current.shared_weight.detach().clone()
        previous = cpu_copy(current_opt.moments['shared_weight'])
        current.shared_weight.grad = torch.tensor([.2, -.5], dtype=torch.float64)
        current.private_weight.grad = torch.tensor([.4, .1], dtype=torch.float64)
        b = current.shared_weight.grad + .01 * old
        expected_m = .9 * previous['m'] + .1 * b
        expected_v = .999 * previous['v'] + .001 * b.square()
        expected = old - .01 * expected_m / (1 - .9 ** 2) / ((expected_v / (1 - .999 ** 2)).sqrt() + 1e-8)
        current_opt.step(pair=None, probes=[])
        torch.testing.assert_close(current.shared_weight, expected, rtol=1e-14, atol=1e-14)
        assert current_opt.active == saved_opt['active']
        assert equal(current_opt.history, saved_opt['history'])
    assert equal(model.state_dict(), clone.state_dict())
    assert equal(opt.state_dict(), restored.state_dict())


def test_native_component_dispatch_and_kappa_isolation(monkeypatch):
    calls = []
    monkeypatch.setattr(objective.history_objectives, 'train_step', lambda *a: calls.append(('history', a[6])) or 'H')
    monkeypatch.setattr(objective.pair_objectives, 'train_step', lambda *a: calls.append(('pair', a[6])) or 'P')
    for arm, expected in (('A0', 'H'), ('C_MIX', 'H'), ('P01', 'P'), ('GAUSSIAN_A1', 'H')):
        assert objective.train_step(None, None, None, None, None, None, config(arm), {}) == expected
    assert [call[0] for call in calls] == ['history', 'history', 'pair', 'history']
    assert [call[1]['arm'] for call in calls] == ['A0f', 'C_MIX', 'P01', 'A1']
    assert objective.native_config(config(objective.CG))['kappa'] == 1.
    assert objective.native_config(config(objective.CG), 'GAUSSIAN_A1')['kappa'] == .05
    assert isinstance(objective.make_optimizer(Vectors(), config('A0')), HistoryOptimizer)
    assert isinstance(objective.make_optimizer(Vectors(), config('C_MIX')), HistoryOptimizer)
    assert isinstance(objective.make_optimizer(Vectors(), config('P01')), PairOptimizer)


class TinyM3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 2)
        self.field_slices = {'feature': (0, 2)}
        self.shared_weight = torch.nn.Parameter(torch.randn(2))
        self.shared_bias = torch.nn.Parameter(torch.randn(2))
        self.tower = torch.nn.ModuleList([torch.nn.Linear(2, 1) for _ in range(6)])

    def forward(self, x):
        h = self.embedding(x[0].long()) * self.shared_weight + self.shared_bias
        logits = [tower(h).squeeze(1) for tower in self.tower]
        return torch.stack([sum(torch.where(x[-1] == d, logits[t * 3 + d], torch.zeros_like(logits[0]))
                                for d in range(3)).sigmoid() for t in range(2)], dim=1)


def fixture_bundle():
    x = np.stack([np.arange(12), np.repeat(np.arange(3), 4)], axis=1).astype('float32')
    y = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32'), (3, 1))
    return SimpleNamespace(train_x=x, train_y=y, train_row_ids=np.arange(12), val_x=x, val_y=y,
                           val_row_ids=np.arange(12))


def test_cp_probe_gradients_leave_p_grad_untouched_and_ema_once(monkeypatch):
    torch.manual_seed(2022)
    model = TinyM3()
    ema = copy.deepcopy(model).requires_grad_(False)
    old_ema = cpu_copy(ema.state_dict())
    c = config()
    opt = objective.make_optimizer(model, c)
    original_probes = objective._probes
    seen = []
    def checked_probes(*args):
        before = [p.grad.clone() for p in model.parameters()]
        result = original_probes(*args)
        assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), before))
        seen.append(True)
        return result
    monkeypatch.setattr(objective, '_probes', checked_probes)
    original_update = objective.update_ema
    updates = []
    def counted_update(*args):
        updates.append(True)
        return original_update(*args)
    monkeypatch.setattr(objective, 'update_ema', counted_update)
    bundle = fixture_bundle()
    value = objective.train_step(model, opt, ema, torch.from_numpy(bundle.train_x).T,
        torch.from_numpy(bundle.train_y), bundle.train_row_ids, c,
        dict(dataset='movielens', seed=2022, epoch=1, global_step=1), diagnostic=True)
    assert len(seen) == len(updates) == opt.global_step == 1
    assert value['counts']['training_forwards'] == 1
    assert value['counts']['probe_forwards'] == 4
    assert value['counts']['shared_autograd_calls'] == 5
    for name, current in model.state_dict().items():
        expected = old_ema[name].mul(c['rho']).add(current, alpha=1 - c['rho'])
        assert torch.equal(ema.state_dict()[name], expected)


def test_worker_ram_resume_matches_continuous_epoch(monkeypatch):
    from experiments.overnight_ab_20261103 import worker_b
    monkeypatch.setattr(worker_b.source, 'make_model', lambda *args: TinyM3())
    monkeypatch.setattr(worker_b.source, 'load_protocol', lambda: {'datasets': {'movielens': {'source': {}}}})
    job = dict(job_id='fixture.CP.s2022', arm=objective.CP, dataset='movielens', seed=2022, config=config())
    bundle = fixture_bundle()
    live = worker_b.create_session(job, bundle, 'cpu')
    worker_b.advance_epoch(live)
    saved = worker_b.export_session(live)
    continuous_record = worker_b.advance_epoch(live)
    continuous = worker_b.export_session(live)
    restored = worker_b.restore_session(job, bundle, 'cpu', saved)
    resumed_record = worker_b.advance_epoch(restored)
    resumed = worker_b.export_session(restored)
    for key in ('model', 'ema', 'optimizers', 'loader_state', 'rng', 'selection', 'snapshots',
                'commit_index', 'global_step', 'next_stage_epoch'):
        assert equal(continuous[key], resumed[key]), key
    for key in ('order_hash', 'model_hash', 'validation', 'selection_hash', 'train_objective', 'batch_trace'):
        assert continuous_record[key] == resumed_record[key], key
    assert continuous['commit_index'] == 2 and not continuous['complete']
