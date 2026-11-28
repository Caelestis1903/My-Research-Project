import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.conditional_history_shared_update_20261122.history_controls import history_config, make_history_optimizer
from experiments.conditional_history_shared_update_20261122.runner import batch_inputs
from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import forward_logits, sample
from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from experiments.three_directions_20261128 import freshness_optimizer as fresh
from experiments.three_directions_20261128.protocol import ROOT, STUDY, load_train_validation, scope_digest

AUDIT = ROOT / "artifacts" / STUDY


class TinyM3oE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor(.7))
        self.private = nn.Parameter(torch.tensor(.2))
        self.tower = nn.ModuleList(nn.Linear(1, 1) for _ in range(6))
        self.register_buffer("counter", torch.tensor(0.))

    def forward(self, x):
        self.counter.add_(1)
        feature = (x[0] * self.shared_weight + self.private + .01 * self.counter
                   + .02 * torch.rand_like(x[0])).unsqueeze(1)
        logits = [tower(feature).squeeze(1) for tower in self.tower]
        return torch.stack([sum(torch.where(x[-1] == d, logits[3*t+d], 0.) for d in range(3))
                            for t in range(2)], dim=1).sigmoid()


def fixtures():
    x = torch.tensor([[-.6, .4, .8, 1.5, -.4, 2.], [0., 0., 1., 1., 2., 2.]])
    y = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 0.], [0., 0.], [1., 0.]])
    return x, y, np.arange(len(y), dtype=np.int64)


def context(step):
    return dict(dataset="movielens", seed=2022, epoch=1, global_step=step)


def trajectory(model, config):
    optimizer = (v3.make_optimizer(model, config) if config["arm"] == "P00"
                 else make_history_optimizer(model, config))
    return model, optimizer, copy.deepcopy(model).requires_grad_(False)


def trace(state, result, device):
    model, optimizer, ema = state
    diagnostics = dict(result["diagnostics"])
    return dict(model=v3.cpu_copy(model.state_dict()), optimizer=v3.cpu_copy(optimizer.state_dict()),
                ema=v3.cpu_copy(ema.state_dict()), gradients={n: v3.cpu_copy(p.grad) for n, p in model.named_parameters()},
                rng=v3.rng_state(device), pred=result["pred"].cpu(), loss=result["loss"],
                real_loss=result["real_loss"], diagnostics=diagnostics, counts=result["counts"])


def assert_stock_equal(stock, wrapped):
    assert set(stock["diagnostics"]).issubset(wrapped["diagnostics"])
    wrapped["diagnostics"] = {key: wrapped["diagnostics"][key] for key in stock["diagnostics"]}
    assert v3.equal(stock, wrapped)


@pytest.mark.parametrize("gamma,observer", [(None, False), (0., False), (0., True)])
def test_stock_equivalence_with_mutating_buffers_and_rng(gamma, observer):
    v3.setup(2022)
    initial = TinyM3oE()
    cfg = v3.config("P00", "movielens", .05) if gamma is None else history_config("A21", "movielens")
    baseline, wrapped = [trajectory(copy.deepcopy(initial), cfg) for _ in range(2)]
    adapter = fresh.FreshnessAdapter(gamma, observer)
    x, y, rows = fixtures()
    for step, labels in enumerate((y, y, torch.zeros_like(y), y), 1):
        cap = wrapped[0].shared_weight.detach().clone()
        before = v3.rng_state(x.device)
        source = trace(baseline, train_step(*baseline, x, labels, rows, cfg, context(step)), x.device)
        v3.restore_rng(before, x.device)
        result = adapter.step(*wrapped, x, labels, rows, cfg, context(step))
        assert_stock_equal(source, trace(wrapped, result, x.device))
        assert result["diagnostics"]["no_event"] == (step == 3)
        if gamma is not None:
            assert torch.equal(adapter.snapshot["shared_weight"], cap)
            assert wrapped[1].on_pre_step is None
        if observer and step == 1:
            assert result["diagnostics"]["correction_norm"] == 0.
            assert result["diagnostics"]["probe_forwards"] == 0
        if observer and step == 2:
            assert result["diagnostics"]["correction_norm"] > 0.
            assert result["diagnostics"]["probe_forwards"] == 1


def independent_loss(model, x, pair):
    _, logits = forward_logits(model, x)
    return F.softplus(-(logits[pair["pos"], pair["task"]] - logits[pair["neg"], pair["task"]])).mean()


def test_correction_matches_reference_and_conserves_probe_state(monkeypatch):
    v3.setup(2022)
    cfg = history_config("A21", "movielens")
    x, y, rows = fixtures()
    model, optimizer, ema = trajectory(TinyM3oE(), cfg)
    adapter = fresh.FreshnessAdapter(.5)
    theta1 = model.shared_weight.detach().clone()
    adapter.step(model, optimizer, ema, x, y, rows, cfg, context(1))
    theta2 = model.shared_weight.detach().clone()
    rng = v3.rng_state(x.device)
    control = copy.deepcopy(model)
    pair = sample(x[-1].numpy(), y.numpy(), optimizer.demand, False, context(2), rows)

    def independent_gradient(theta):
        replica = copy.deepcopy(control)
        replica.shared_weight.data.copy_(theta)
        v3.restore_rng(rng, x.device)
        return torch.autograd.grad(independent_loss(replica, x, pair), replica.shared_weight)[0]

    now, old = independent_gradient(theta2), independent_gradient(theta1)
    eps = .002

    def displaced_loss(delta):
        replica = copy.deepcopy(control)
        replica.shared_weight.data.copy_(theta1 + delta)
        v3.restore_rng(rng, x.device)
        return independent_loss(replica, x, pair).item()

    assert (displaced_loss(eps) - displaced_loss(-eps)) / (2 * eps) == pytest.approx(float(old), abs=1e-4)
    assert abs(float(now - old)) > 1e-8
    reference = trajectory(copy.deepcopy(model), cfg)
    reference[1].load_state_dict(v3.cpu_copy(optimizer.state_dict()))
    received = {}

    def before_stock(pair, raw):
        received.update(mixed=reference[0].shared_weight.grad.clone(), raw=raw[0].clone())

    reference[1].on_pre_step = before_stock
    v3.restore_rng(rng, x.device)
    train_step(*reference, x, y, rows, cfg, context(2))
    assert torch.equal(received["raw"], now)

    probe = fresh.old_pair_gradient

    def check_probe(*args):
        inputs = lambda: dict(model=v3.cpu_copy(model.state_dict()), optimizer=v3.cpu_copy(optimizer.state_dict()),
                              ema=v3.cpu_copy(ema.state_dict()), grads={n: v3.cpu_copy(p.grad) for n, p in model.named_parameters()},
                              rng=v3.rng_state(x.device), mode=model.training, x=x.clone(), rows=rows.copy())
        before = inputs()
        output = probe(*args)
        assert v3.equal(before, inputs())
        torch.testing.assert_close(output[0], old)
        return output

    monkeypatch.setattr(fresh, "old_pair_gradient", check_probe)
    moments = v3.cpu_copy(optimizer.moments["shared_weight"])
    history = v3.cpu_copy(optimizer.history[-1][0])
    v3.restore_rng(rng, x.device)
    result = adapter.step(model, optimizer, ema, x, y, rows, cfg, context(2))
    assert result["diagnostics"]["correction_norm"] > 0.
    ratio = float(pair["w"][pair["index"]] / pair["q"][pair["index"]])
    alpha = .05 * ratio
    correct = now + .5 * (now - old)
    mixed = received["mixed"] + (correct - now) * alpha + cfg["weight_decay"] * theta2
    first = .9 * moments["m"] + .1 * (mixed - alpha * correct)
    second = .999 * moments["v"] + .001 * mixed.square()
    rebuilt = .99 * history + .01 * ratio * correct
    expected = theta2 - .01 * (first / (1 - .9**2) + .05 * rebuilt / (1 - .99**2)) / (
        (second / (1 - .999**2)).sqrt() + 1e-8)
    torch.testing.assert_close(model.shared_weight, expected)
    torch.testing.assert_close(optimizer.history[-1][0], rebuilt)
    torch.testing.assert_close(optimizer.moments["shared_weight"]["m"], first)
    torch.testing.assert_close(optimizer.moments["shared_weight"]["v"], second)
    assert optimizer.pair_clock == 2
    for (name, p), (_, ref) in zip(model.named_parameters(), reference[0].named_parameters()):
        if name != "shared_weight":
            assert torch.equal(p.grad, ref.grad)
            assert torch.equal(p, ref)
    assert torch.equal(adapter.snapshot["shared_weight"], theta2)


def test_zero_clock_and_failed_probe(monkeypatch):
    v3.setup(2022)
    cfg = history_config("A21", "movielens")
    state = trajectory(TinyM3oE(), cfg)
    model, optimizer, _ = state
    x, y, rows = fixtures()
    adapter = fresh.FreshnessAdapter(.5)
    adapter.step(*state, x, y, rows, cfg, context(1))
    q = optimizer.history[-1][0].clone()
    adapter.step(*state, x, torch.zeros_like(y), rows, cfg, context(2))
    assert optimizer.pair_clock == 1 and torch.equal(q, optimizer.history[-1][0])
    zero_x = x.clone()
    zero_x[0].zero_()
    result = adapter.step(*state, zero_x, y, rows, cfg, context(3))
    assert result["diagnostics"]["zero_event"]
    assert optimizer.pair_clock == 2
    torch.testing.assert_close(optimizer.history[-1][0], q * .99)
    for value, error in (([None], RuntimeError), ([torch.tensor(float("nan"))], FloatingPointError)):
        monkeypatch.setattr(fresh, "old_pair_gradient", lambda *args, value=value: value)
        with pytest.raises(error):
            adapter.step(*state, x, y, rows, cfg, context(4))
        assert adapter.progress == 3 and optimizer.pair_clock == 2 and optimizer.on_pre_step is None
        assert all(not tower._forward_hooks for tower in model.tower)


@pytest.mark.parametrize("dataset,tail", [("movielens", 21927), ("kuairand", 40052)])
def test_real_first_and_tail_match_source(dataset, tail, monkeypatch):
    v3.setup(2022)
    opened = []
    original_load = np.load

    def read_array(path, **kwargs):
        assert kwargs == dict(mmap_mode="r", allow_pickle=False)
        opened.append(Path(path).name)
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", read_array)
    bundle = load_train_validation(dataset, AUDIT / "DATA_AUDIT.json")
    assert set(opened) == {f"{split}_{field}.npy" for split in ("train", "validation") for field in ("x", "y", "row_ids")}
    frozen = v3.load_protocol()["datasets"][dataset]["source"]
    init_model = v3.make_model(bundle, frozen)
    generator = torch.Generator().manual_seed(2022 + 11000)
    order = torch.randperm(len(bundle.train_y), generator=generator).numpy()
    cut = len(order) - tail
    assert cut % 40960 == 0 and len(order[cut:]) == tail
    device = torch.device("cpu")
    for arm, gamma, observer in (("P00", None, False), ("A21", 0., True)):
        cfg = v3.config(arm, dataset, .05) if gamma is None else history_config(arm, dataset)
        stock, wrapped = [trajectory(copy.deepcopy(init_model), cfg) for _ in range(2)]
        from experiments.conditional_history_shared_update_20261122.protocol import read
        audit = read(AUDIT / "SOURCE_AUDIT.json")
        audited = audit["canonical_optimizer_and_ema"][dataset]
        first_draw = next(item for item in audit["gaussian_first_draw_archived_replays"]
                          if item["dataset"] == dataset and item["seed"] == 2022)
        assert v3.state_digest(bundle.train_row_ids[order[:40960]]) == first_draw["batch_rows_sha256"]
        assert scope_digest(stock[1].parameter_manifest) == audited["parameter_manifest_hash"]
        adapter = fresh.FreshnessAdapter(gamma, observer)
        last_step = cut // 40960 + 1
        loader_state = generator.get_state().clone()
        for step, indices in ((1, order[:40960]), (last_step, order[cut:])):
            if step > 1:
                adapter.progress = step - 1
            ctx = dict(context(step), dataset=dataset, batch=step, last_batch=step == last_step)
            args = batch_inputs(bundle, indices, device)
            before = v3.rng_state(device)
            source = trace(stock, train_step(*stock, *args, cfg, ctx), device)
            v3.restore_rng(before, device)
            result = adapter.step(*wrapped, *args, cfg, ctx)
            assert_stock_equal(source, trace(wrapped, result, device))
        assert torch.equal(generator.get_state(), loader_state)


def test_b2_adapter_resume_across_epoch():
    v3.setup(2022)
    cfg = history_config("A21", "movielens")
    initial, adapter = trajectory(TinyM3oE(), cfg), fresh.FreshnessAdapter(.5)
    x, y, rows = fixtures()
    adapter.step(*initial, x, y, rows, cfg, context(1))
    saved_adapter = v3.cpu_copy(adapter.state_dict())
    saved_optimizer = v3.cpu_copy(initial[1].state_dict())
    resumed = trajectory(copy.deepcopy(initial[0]), cfg)
    resumed[1].load_state_dict(saved_optimizer)
    resumed[2].load_state_dict(initial[2].state_dict())
    resumed_adapter = fresh.FreshnessAdapter(.5)
    resumed_adapter.load_state_dict(saved_adapter)
    rng = v3.rng_state(x.device)
    tail_ctx = dict(context(2), epoch=2)
    uninterrupted = adapter.step(*initial, x, y, rows, cfg, tail_ctx)
    v3.restore_rng(rng, x.device)
    replay = resumed_adapter.step(*resumed, x, y, rows, cfg, tail_ctx)
    for result in (uninterrupted, replay):
        result["diagnostics"].pop("probe_seconds")
    assert_stock_equal(trace(initial, uninterrupted, x.device), trace(resumed, replay, x.device))
    assert v3.equal(adapter.state_dict(), resumed_adapter.state_dict())
