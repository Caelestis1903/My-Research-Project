"""A0 engineering checks for Train pair exposure, GA31 identity and gradients."""
import copy
import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.conditional_history_shared_update_20261122.runner import batch_inputs
from experiments.conditional_information_momentum_20261102_v3 import boundary_sampler, protocol as v3
from experiments.three_directions_20261128 import a_step
from experiments.three_directions_20261128.macro_pair_sampler import PAIR_COUNT, MacroPairSampler, ordered_ids_hash
from experiments.three_directions_20261128.protocol import load_train_validation

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/three_directions_20261128"
DATASETS = ("movielens", "kuairand")
ARMS = ("B0", "P0", "G0", "GP0", "GP-shared")


class TinyM3oE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = nn.ModuleList([nn.Linear(2, 1) for _ in range(6)])
        self.register_buffer("offset", torch.tensor(.125))
        self.perturb_slices = [(0, 1)]

    def forward(self, x, *, view=None, kappa=.1):
        embedded = x[:2].T + self.offset
        if view is not None:
            embedded = a_step.views.apply_view(embedded, self.perturb_slices, view, kappa)
        logits = [tower(embedded).squeeze(1) for tower in self.tower]
        return torch.stack([sum(torch.where(x[-1] == d, logits[3 * t + d].sigmoid(), 0.)
                                for d in range(3)) for t in range(2)], dim=1)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny():
    domain = np.repeat(np.arange(3), 5)
    y = np.tile(np.array([[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (3, 1))
    x = np.column_stack((np.linspace(-.6, .6, len(y)), y[:, 1], domain)).astype(np.float32)
    ids = np.arange(10, 10 + 2 * len(y), 2, dtype=np.int64)[::-1].copy()
    bundle = types.SimpleNamespace(dataset="movielens", train_x=x, train_y=y, train_row_ids=ids,
                                   identity={"split_row_fingerprints": {"train": "tiny-train"}})
    sampler = MacroPairSampler(bundle.dataset, domain, y, ids, "tiny-train")
    context = dict(dataset=bundle.dataset, seed=2022, epoch=1, global_step=1)
    while sampler.sample_macro6(context)["condition"][0] != 0:
        context["global_step"] += 1
    return bundle, sampler, context


@pytest.fixture(scope="module")
def archive():
    commit = "c8675ae2923c15b6965884b140b5b646690cd95e"
    archive_path = ("experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/"
                    "gradient_aug_20261031_v2/objectives.py")
    assert subprocess.check_output(["git", "rev-parse", f"{commit}:{archive_path}"], cwd=ROOT).decode().strip() == (
        "c4b58d0e230c03bfe69f8822ddac818d332d5ed1")
    reference = types.ModuleType("ga31_archived_objectives")
    reference.__package__ = a_step.ga31.__package__
    source = subprocess.check_output(["git", "show", f"{commit}:{archive_path}"], cwd=ROOT)
    exec(compile(source, archive_path, "exec"), reference.__dict__)
    return reference


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("A0 archived draws and full batches require CUDA")
    return torch.device(os.environ.get("DIRECTION_A_TEST_DEVICE", "cuda:0"))


@pytest.fixture(scope="module", params=DATASETS)
def bundle(request):
    return load_train_validation(request.param, ARTIFACTS / "DATA_AUDIT.json")


def parts(model, arm, bundle):
    config = a_step.a_config(arm, bundle.dataset)
    optimizer, adapter = a_step.make_training_parts(model, types.SimpleNamespace(config=config), bundle)
    return config, optimizer, adapter


def test_macro6_expectation_support_replacement_and_streams(tiny):
    bundle, sampler, context = tiny
    domain, y = bundle.train_x[:, -1], bundle.train_y
    logits = np.column_stack((np.linspace(-1., 1., len(y)), np.linspace(2., -2., len(y))))
    exact = np.mean([
        np.logaddexp(0, -(logits[(domain == d) & (y[:, t] == 1), t, None]
                         - logits[(domain == d) & (y[:, t] == 0), t])).mean()
        for d in range(3) for t in range(2)
    ])
    expected = 0.
    joint = 2 * y[:, 0] + y[:, 1]
    assert sampler.active == tuple(boundary_sampler.CONDITIONS)
    assert sampler.w.dtype == sampler.q.dtype == np.float64
    assert sampler.q.sum() == pytest.approx(1.)
    assert np.max(sampler.w / sampler.q) <= 2.
    for d in range(3):
        for t in range(2):
            assert sum(w for (d_, t_, _, _), w in zip(sampler.active, sampler.w) if (d_, t_) == (d, t)) == (
                pytest.approx(1 / 6))
    for (d, t, a, b), w, q in zip(sampler.active, sampler.w, sampler.q):
        pos = logits[(domain == d) & (joint == a), t]
        neg = logits[(domain == d) & (joint == b), t]
        expected += q * (w / q) * np.logaddexp(0, -(pos[:, None] - neg)).mean()
    assert expected == pytest.approx(exact, abs=1e-15)

    draw = sampler.sample_macro6(context)
    again = sampler.sample_macro6(dict(context, arm="GP0", attempt="fresh"))
    np.testing.assert_array_equal(draw["ids"], again["ids"])
    assert draw["pair_index_hash"] == again["pair_index_hash"] == ordered_ids_hash(draw["ids"])
    assert draw["pair_index_hash"] != sampler.sample_macro6(dict(context, global_step=81))["pair_index_hash"]
    d, t, a, b = draw["condition"]
    pos, neg = draw["offsets"].reshape(2, PAIR_COUNT)
    assert len(draw["ids"]) == 2 * PAIR_COUNT and draw["unique_pairs"] < PAIR_COUNT
    assert ((domain[pos] == d) & (domain[neg] == d)).all()
    assert ((y[pos, t] == 1) & (y[neg, t] == 0) & (joint[pos] == a) & (joint[neg] == b)).all()
    keep = joint != 1
    pruned = MacroPairSampler(bundle.dataset, domain[keep], y[keep], bundle.train_row_ids[keep], "pruned")
    assert 0 < len(pruned.active) < len(sampler.active) and all(1 not in c[2:] for c in pruned.active)
    with pytest.raises(ValueError, match="supports"):
        MacroPairSampler(bundle.dataset, domain, np.zeros_like(y), bundle.train_row_ids, "missing")

    margin = torch.tensor([-2., 2.], requires_grad=True)
    F.softplus(-margin).mean().backward()
    assert torch.isfinite(margin.grad).all() and (margin.grad < 0.).all()


def test_forward_order_grad_mode_coupling_and_no_hooks(tiny, monkeypatch):
    bundle, sampler, context = tiny
    x, y, rows = batch_inputs(bundle, np.arange(len(bundle.train_y)), "cpu")
    original_forward = a_step.forward_logits
    calls = []

    def spy(model, batch, **kwargs):
        pred, logits = original_forward(model, batch, **kwargs)
        torch.testing.assert_close(pred.detach(), logits.detach().sigmoid(), atol=torch.finfo(torch.float32).eps, rtol=0.)
        calls.append((batch.shape[1], logits.requires_grad, kwargs.get("view")))
        return pred, logits

    monkeypatch.setattr(a_step, "forward_logits", spy)
    hashes, captured_views = set(), {}
    for arm in ARMS:
        model = TinyM3oE()
        ema = copy.deepcopy(model)
        config, optimizer, adapter = parts(model, arm, bundle)
        rng = torch.get_rng_state()
        before = model.offset.clone()
        calls.clear()
        result = adapter.step(model, optimizer, ema, x, y, rows, config, context)
        diagnostic, work = result["diagnostics"], result["counts"]
        hashes.add(diagnostic["pair_index_hash"])
        assert torch.equal(before, ema.offset) and torch.equal(rng, torch.get_rng_state())
        assert not result["pred"].requires_grad and not any(tower._forward_hooks for tower in model.tower)
        assert [row[0] for row in calls] == [len(y), 2 * PAIR_COUNT] + ([5] if config["gaussian_bce"] else [])
        assert calls[0][1] is False and calls[1][1] == bool(config["lambda_pair"])
        assert work["training_forwards"] == len(calls) and work["optimizer_steps"] == work["ema_updates"] == 1
        assert work["pair_draws"] == PAIR_COUNT and work["pair_diagnostic_vjp_calls"] == bool(config["lambda_pair"])
        assert all(float(s["step"]) == 1 for s in optimizer.state.values())
        if config["gaussian_bce"]:
            assert calls[2][1] is False
            captured_views[arm] = (calls[1][2], calls[2][2])

    assert len(hashes) == 1
    independent, bce = captured_views["GP0"]
    shared, bce_shared = captured_views["GP-shared"]
    for _, base in captured_views.values():
        assert torch.equal(bce["noise"], base["noise"])
    assert independent["key"] == shared["key"] != bce_shared["key"]
    pos_ind, neg_ind = independent["noise"].chunk(2)
    pos_shared, neg_shared = shared["noise"].chunk(2)
    assert torch.equal(pos_ind, pos_shared) and torch.equal(pos_shared, neg_shared)
    assert torch.all(pos_ind != neg_ind)
    pos_ids = sampler.sample_macro6(context)["ids"][:PAIR_COUNT]
    i, j = np.flatnonzero(pos_ids == pos_ids[0])[:2]
    assert not torch.equal(pos_ind[i], pos_ind[j])

    def fail(*args, **kwargs):
        raise RuntimeError("forward")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward"):
        original_forward(model, x)
    assert not any(tower._forward_hooks for tower in model.tower)


def test_gaussian_empty_d0_and_default_source_identity(tiny, archive):
    bundle, _, context = tiny
    x, y, rows = batch_inputs(bundle, np.flatnonzero(bundle.train_x[:, -1] != 0), "cpu")
    model = TinyM3oE()
    ema = copy.deepcopy(model)
    config, optimizer, adapter = parts(model, "G0", bundle)
    args = (model, x, y, 0, a_step.source_config(True), bundle.dataset, context["seed"], context["global_step"])
    original_loss, original = archive.losses(*args)
    default_loss, default = a_step.ga31.losses(*args)
    assert torch.equal(original_loss, default_loss) and torch.equal(original["pred"], default["pred"])
    result = adapter.step(model, optimizer, ema, x, y, rows, config, context)
    assert result["diagnostics"]["gaussian_bce"] is None and result["counts"]["gaussian_bce_forwards"] == 0
    assert result["loss"] == float(original_loss.detach()) and torch.equal(result["pred"], original["pred"])
    active_x, active_y, _ = batch_inputs(bundle, np.arange(len(bundle.train_y)), "cpu")
    active_args = (model, active_x, active_y, *args[3:])
    original_loss, original = archive.losses(*active_args)
    default_loss, default = a_step.ga31.losses(*active_args)
    assert torch.equal(original_loss, default_loss) and torch.equal(original["pred"], default["pred"])
    assert all(torch.equal(a, b) for a, b in zip(original["augmented"], default["augmented"]))
    assert original["views"][0]["hash"] == default["views"][0]["hash"]
    adapter.load_state_dict(adapter.state_dict())
    state = adapter.state_dict()
    state["sampler"]["pool_hashes"]["0:0"] = "invalid"
    with pytest.raises(ValueError, match="sampler identity"):
        adapter.load_state_dict(state)


def test_component_norms_match_direct_vjps(tiny, archive, monkeypatch):
    bundle, sampler, context = tiny
    x, y, rows = batch_inputs(bundle, np.arange(len(bundle.train_y)), "cpu")
    torch.manual_seed(2022)
    model = TinyM3oE()
    ema = copy.deepcopy(model)
    config, optimizer, adapter = parts(model, "P0", bundle)
    reference = copy.deepcopy(model)
    params = tuple(reference.parameters())
    base, _ = archive.losses(reference, x, y, 0, a_step.source_config(False),
                             bundle.dataset, context["seed"], context["global_step"])
    draw = sampler.sample_macro6(context)
    pair_input = torch.from_numpy(bundle.train_x[draw["offsets"]]).T
    _, logits = boundary_sampler.forward_logits(reference, pair_input)
    pos, neg = logits[:, draw["condition"][1]].chunk(2)
    pair = draw["w"] / draw["q"] * F.softplus(neg - pos).mean()
    grads_base = torch.autograd.grad(base, params, retain_graph=True)
    grads_pair = torch.autograd.grad(pair, params, retain_graph=True)
    mixed = base + .05 * pair
    mixed.backward()

    num_calls = {"backward": 0, "grad": 0}

    def counted(name, original):
        def call(*args, **kwargs):
            num_calls[name] += 1
            return original(*args, **kwargs)
        return call

    for name in num_calls:
        monkeypatch.setattr(torch.autograd, name, counted(name, getattr(torch.autograd, name)))

    actual = adapter.step(model, optimizer, ema, x, y, rows, config, context)
    assert num_calls == {"backward": 1, "grad": 1}
    assert actual["loss"] == float(mixed.detach())
    for p, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, expected.grad, atol=2e-7, rtol=1e-6)

    def norm(values):
        return float(torch.cat([g.detach().flatten().double() for g in values]).norm())

    expected = dict(base=norm(grads_base), pair=norm(grads_pair), applied_pair=norm([.05 * g for g in grads_pair]),
                    total=norm([p.grad for p in reference.parameters()]))
    assert actual["diagnostics"]["grad_norms"] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_cuda_registered_noise_all_five_seeds(bundle, cuda):
    source_audit = json.loads((ARTIFACTS / "SOURCE_AUDIT.json").read_text())
    dataset = bundle.dataset
    d = v3.load_protocol()["datasets"][dataset]
    v3.setup(2022, cuda)
    model = v3.make_model(bundle, d["source"]).to(cuda)
    config, _, adapter = parts(model, "G0", bundle)
    expected_conditions = {"movielens": 12, "kuairand": 20}
    assert len(adapter.sampler.active) == expected_conditions[dataset]
    fields = ("user_id movie_id gender occupation zip cate_id".split() if dataset == "movielens" else
              "user_active_degree is_live_streamer is_video_author follow_user_num_range fans_user_num_range "
              "friend_user_num_range register_days_range".split() + [f"onehot_feat{i}" for i in range(18)] +
              "video_type upload_dt upload_type server_width server_height music_type tag counts".split())
    assert d["source"]["perturbable_sparse_fields"] == fields
    assert model.perturb_slices == [model.field_slices[name] for name in fields]
    assert model.field_slices["domain_indicator"] not in model.perturb_slices
    matching = [row for row in source_audit["gaussian_first_draw_archived_replays"] if row["dataset"] == dataset]
    assert [row["seed"] for row in matching] == list(range(2022, 2027))
    for row in matching:
        seed = row["seed"]
        loader = torch.Generator().manual_seed(seed + 11000)
        order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
        x, _, rows = batch_inputs(bundle, order[:40960], cuda)
        context = dict(dataset=dataset, seed=seed, epoch=1, global_step=1)
        d0 = (x[-1] == 0).cpu().numpy()
        view = a_step.base_gaussian_view(model, x, config, context)
        assert ordered_ids_hash(rows) == row["batch_rows_sha256"]
        assert ordered_ids_hash(rows[d0]) == row["d0_rows_sha256"]
        assert view["noise"].shape == (row["target_rows"], len(fields), 16)
        assert view["key"] == row["key"] and view["hash"] == row["archived_noise_sha256"]


@pytest.mark.parametrize("arm", ["B0", "G0"])
@pytest.mark.parametrize("tail", [False, True])
def test_cuda_actual_first_and_tail_preserve_source(bundle, cuda, archive, arm, tail):
    dataset = bundle.dataset
    v3.setup(2022, cuda)
    model = v3.make_model(bundle, v3.load_protocol()["datasets"][dataset]["source"]).to(cuda)
    reference = copy.deepcopy(model)
    ema, reference_ema = copy.deepcopy(model), copy.deepcopy(reference)
    config, optimizer, adapter = parts(model, arm, bundle)
    reference_optimizer = a_step.make_optimizer(reference, config)
    loader = torch.Generator().manual_seed(2022 + 11000)
    order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
    batches = (len(order) + 40959) // 40960
    step = batches if tail else 1
    x, y, rows = batch_inputs(bundle, order[(step - 1) * 40960:step * 40960], cuda)
    assert len(y) == ({"movielens": 21927, "kuairand": 40052}[dataset] if tail else 40960)
    initial_rng = torch.cuda.get_rng_state(cuda)
    target = config["target_domain"]
    ref_loss, ref_source = archive.losses(
        reference, x, y, target, a_step.source_config(config["gaussian_bce"]), dataset, 2022, step,
        mix_lambda=config["lambda_aug"], kappa=config["kappa"],
    )
    ref_loss.backward()
    reference_optimizer.step()
    a_step.update_ema(reference_ema, reference, config["rho"])
    actual = adapter.step(model, optimizer, ema, x, y, rows, config,
                          dict(dataset=dataset, seed=2022, epoch=1, global_step=step))
    assert torch.equal(initial_rng, torch.cuda.get_rng_state(cuda))
    assert torch.equal(actual["pred"], ref_source["pred"]) and actual["loss"] == float(ref_loss.detach())
    assert actual["counts"]["training_forward_rows"] == len(y) + ref_source["extra_views"] + 2 * PAIR_COUNT
    assert actual["diagnostics"]["grad_norms"]["base"] == actual["diagnostics"]["grad_norms"]["total"]
    if config["gaussian_bce"]:
        assert actual["diagnostics"]["gaussian_bce"]["noise_hash"] == ref_source["views"][0]["hash"]

    max_diff = {"grad": 0., "model": 0., "ema": 0., "moment": 0.}

    def compare(kind, value, expected, *, atol, rtol):
        max_diff[kind] = max(max_diff[kind], float((value.detach() - expected.detach()).abs().max()))
        torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)

    for p, q in zip(model.parameters(), reference.parameters()):
        compare("grad", p.grad, q.grad, atol=2e-7, rtol=2e-5)
        compare("model", p, q, atol=2e-5, rtol=2e-5)
        for key in ("exp_avg", "exp_avg_sq"):
            compare("moment", optimizer.state[p][key], reference_optimizer.state[q][key], atol=2e-8, rtol=4e-5)
        assert optimizer.state[p]["step"] == reference_optimizer.state[q]["step"] == 1
    for p, q in zip(ema.parameters(), reference_ema.parameters()):
        compare("ema", p, q, atol=5e-7, rtol=2e-5)
    assert all(torch.equal(p, q) for p, q in zip(model.buffers(), reference.buffers()))
    assert all(torch.equal(p, q) for p, q in zip(ema.buffers(), reference_ema.buffers()))
    print(dataset, arm, "tail" if tail else "first", max_diff)


def test_cuda_real_pair_smoke(bundle, cuda, monkeypatch):
    dataset = bundle.dataset
    source = v3.load_protocol()["datasets"][dataset]["source"]
    v3.setup(2022, cuda)
    initial = v3.make_model(bundle, source).to(cuda)
    optimizer_config = dict(movielens=(1e-4, .9828205985452511), kuairand=(1e-5, .995))
    order = torch.randperm(len(bundle.train_y), generator=torch.Generator().manual_seed(13022)).numpy()
    original_forward = a_step.forward_logits

    def assert_routing(*args, **kwargs):
        pred, logits = original_forward(*args, **kwargs)
        torch.testing.assert_close(pred.detach(), logits.detach().sigmoid(), atol=torch.finfo(torch.float32).eps, rtol=0.)
        return pred, logits

    monkeypatch.setattr(a_step, "forward_logits", assert_routing)
    matched = []
    for arm in ("P0", "GP0", "GP-shared"):
        model = copy.deepcopy(initial)
        ema = copy.deepcopy(model)
        config, optimizer, adapter = parts(model, arm, bundle)
        wd, rho = optimizer_config[dataset]
        group, = optimizer.param_groups
        assert type(optimizer) is torch.optim.Adam
        assert (config["target_domain"], config["pair_count"], config["lambda_pair"], config["tau"], config["rho"]) == (
            0, 2048, .05, 1., rho)
        assert (group["lr"], group["weight_decay"], group["betas"], group["eps"]) == (.01, wd, (.9, .999), 1e-8)
        assert all(not group[key] for key in ("foreach", "fused", "amsgrad", "decoupled_weight_decay"))
        context = dict(dataset=dataset, seed=2022, epoch=1, global_step=1)
        for domain in (0, 1):
            while adapter.sampler.sample_macro6(context)["condition"][0] != domain:
                context["global_step"] += 1
            start = (context["global_step"] - 1) * 40960
            x, y, rows = batch_inputs(bundle, order[start:start + 512], cuda)
            result = adapter.step(model, optimizer, ema, x, y, rows, config, context)
            diag, work = result["diagnostics"], result["counts"]
            assert work["gaussian_pair_rows"] == (4096 if arm != "P0" and domain == 0 else 0)
            assert work["pair_rows"] == 4096 and work["pair_diagnostic_vjp_calls"] == 1
            assert diag["pair_requires_grad"] and diag["grad_norms"]["pair"] > 0
            assert not any(tower._forward_hooks for tower in model.tower)
            matched.append((arm, domain, diag["pair_index_hash"], diag["gaussian_bce"], diag["gaussian_pair"]))
    for domain in (0, 1):
        clean, independent, shared = [row for row in matched if row[1] == domain]
        assert clean[2] == independent[2] == shared[2] and independent[3] == shared[3]
        if domain == 0:
            assert independent[4]["key"] == shared[4]["key"] and independent[4]["noise_hash"] != shared[4]["noise_hash"]
