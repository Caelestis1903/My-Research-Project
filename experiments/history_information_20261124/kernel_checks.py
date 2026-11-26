"""Finite implementation checks for the history kernel. No experiment dispatch."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from . import objectives

from experiments.conditional_history_shared_update_20261122.history_controls import make_history_optimizer
from experiments.conditional_information_momentum_20261102_v3 import objectives as source_objectives
from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import sample
from experiments.overnight_ab_20261103 import published_training as published

from . import groups, sources
from .optimizers import HistoryUpdate, Update, components, norm_matched_steps, transition
from .protocol import RunSpec


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def _reference(entries, zeta, alpha, beta, lam, dtype):
    mixed = torch.stack([entry[0] for entry in entries])
    b, h, events = [], [], []
    for tick, (g, raw, ratio) in enumerate(entries):
        if ratio is not None:
            events.append(tick)
        h.append(torch.zeros_like(g) if raw is None else raw * ratio)
        b.append(g - lam * h[-1])
    b, h = torch.stack(b), torch.stack(h)

    def average(terms, beta):
        n = len(terms)
        weights = torch.tensor([(1 - beta) * beta ** (n - 1 - i) for i in range(n)], dtype=dtype)
        return (terms * weights[:, None]).sum(0)

    n = len(entries)
    v = average((1 - zeta) * mixed.square() + zeta * (b.square() + (lam * h).square()), .999)
    m_hat = average(b, .9) / (1 - .9 ** n)
    d = 1 / (torch.sqrt(v / (1 - .999 ** n)) + 1e-8)
    candidate = d * m_hat
    if n - 1 in events:
        k = len(events)
        scales = torch.stack([_reference(entries[:tick + 1], zeta, 0., beta, lam, dtype)[1]
                              if tick != n - 1 else d for tick in events])
        q_hat = average(h[events], beta) / (1 - beta ** k)
        u_hat = average(scales * h[events], beta) / (1 - beta ** k)
        candidate = candidate + lam * ((1 - alpha) * d * q_hat + alpha * u_hat)
    return candidate, d, v


def algebra_checks():
    tensor = lambda values: torch.tensor(values, dtype=torch.float64)
    entries = [(tensor([1., -2.]), tensor([.3, .4]), .7),
               (tensor([2., 3.]), None, None),
               (tensor([.4, -.31]), None, .2),
               (tensor([-.15, 2.]), tensor([-.12, .8]), 1.7),
               (tensor([.2, -.08]), tensor([.7, -.2]), .6)]
    specs = [
        RunSpec("movielens", 2022, "H0"),
        RunSpec("movielens", 2022, "C1_Z050"),
        RunSpec("movielens", 2022, "C1_Z100"),
        RunSpec("movielens", 2022, "C2_A050"),
        RunSpec("movielens", 2022, "C2_A100"),
        RunSpec("movielens", 2022, "S_C1", zeta=.5),
        RunSpec("movielens", 2022, "S_C2", alpha=.5),
        RunSpec("movielens", 2022, "C12", zeta=.5, alpha=1.),
    ]
    biggest_direction_error = 0.
    for spec in specs:
        assert RunSpec.from_dict(spec.to_dict()) == spec
        state = {}
        for n, (mixed, raw, ratio) in enumerate(entries, 1):
            pre = v3.state_digest(state)
            next_state, update = transition(state, mixed, raw, ratio, spec)
            assert v3.state_digest(state) == pre
            expected, _, expected_v = _reference(entries[:n], spec.zeta, spec.alpha, spec.beta_pair,
                                                 spec.lambda_pair, torch.float64)
            close(update.direction(), expected)
            close(next_state["v"], expected_v)
            biggest_direction_error = max(biggest_direction_error, float((update.direction() - expected).abs().max()))
            assert next_state["step"] == n
            assert next_state["k"] == sum(r is not None for _, _, r in entries[:n])
            if ratio is None:
                assert all(next_state.get(key) is state.get(key) for key in ("q", "u"))
            state = next_state
        skipped, update = transition(state, None, tensor([.2, .3]), .5, spec)
        assert skipped is state and update is None

    h0, c2 = specs[0], specs[4]
    h0_state, h0_step = transition({}, *entries[0], h0)
    c2_state, c2_step = transition({}, *entries[0], c2)
    close(h0_step.direction(), c2_step.direction())
    _, h0_step = transition(h0_state, *entries[0], h0)
    _, c2_step = transition(c2_state, *entries[0], c2)
    close(h0_step.direction(), c2_step.direction())
    cold, update = transition({}, tensor([0., 1.]), None, None, c2)
    assert cold["k"] == 0 and "q" not in cold and "u" not in cold and torch.isfinite(update.direction()).all()

    native = tensor([.03, -.01])
    theta = tensor([3., -2.])
    raw = tensor([.7, -.3])
    mixed, b, h = components(native, theta, raw, 1.2, .05, .0001)
    close(mixed, native + .0001 * theta)
    close(b, mixed - .05 * h)
    ada = RunSpec("movielens", 2022, "P_C1")
    state, update = transition({}, mixed, raw, 1.2, ada)
    close(state["v"], .001 * b.square())
    close(state["v_p"], .001 * h.square())
    close(update.direction(), b / (b.abs() + 1e-8) + .05 * h / (h.abs() + 1e-8))
    later, _ = transition(state, mixed, None, None, ada)
    close(later["v_p"], .999 * state["v_p"])
    assert later["k"] == state["k"] and later["q"] is state["q"]

    laprop = RunSpec("movielens", 2022, "P_C2")
    state, _ = transition({}, mixed, None, 1.2, laprop)
    later, update = transition(state, mixed * 2, None, None, laprop)
    d = 1 / (torch.sqrt((.999 * .001 * mixed.square() + .001 * (mixed * 2).square()) / (1 - .999**2)) + 1e-8)
    close(update.direction(), (.9 * state["t"] + .1 * d * mixed * 2) / (1 - .9**2))
    assert not set(later) & {"m", "q", "u"}

    zero, one, two = tensor([0., 0.]), tensor([1., -1.]), tensor([2., -2.])
    updated, receipts = norm_matched_steps([
        ("shared_weight", zero.float(), Update(one), Update(two)),
        ("expert.0.weight", zero.float(), Update(zero), Update(one)),
    ], .01)
    assert receipts["EXPERT"]["status"] == "UNMATCHED_ZERO" and not updated["expert.0.weight"].any()
    torch.testing.assert_close(updated["shared_weight"], -.01 * two.float())
    assert torch.equal(one, tensor([1., -1.])) and torch.equal(two, tensor([2., -2.]))
    assert RunSpec("movielens", 2022, "C1_Z050", zeta=0).canonical_arm == "H0"
    assert RunSpec("movielens", 2022, "S_C1", zeta=0).canonical_arm == "S_C1"
    assert RunSpec("movielens", 2022, "G_Cstar", algorithm_id="C1", zeta=0).canonical_arm == "G_H0"
    assert RunSpec("movielens", 2022, "HG_Cstar", algorithm_id="C1", zeta=0).canonical_arm == "HG_H0"
    assert RunSpec("movielens", 2022, "Cstar_N", algorithm_id="C1", zeta=0).canonical_arm == "H0_N"
    assert RunSpec("movielens", 2022, "G_B1", lambda_pair=0).canonical_arm == "G_B0"
    return dict(max_fp64_direction_error=biggest_direction_error, canonical_routes="verified")


def _inputs(bundle, order):
    return torch.as_tensor(bundle.train_x[order]).T, torch.as_tensor(bundle.train_y[order]), bundle.train_ids[order]


def _source_hashes(model, optimizer, ema):
    return [v3.state_digest(item.state_dict()) for item in (model, optimizer, ema)]


def real_batch_checks():
    records = []
    for dataset, candidate in (("movielens", "C1_Z050"), ("kuairand", "C2_A100")):
        bundle = sources.load_train_validation(dataset)
        assert not any(name.startswith("test_") for name in vars(bundle))
        spec = RunSpec(dataset, 2022, "H0")
        order = objectives.training_order(spec, bundle, 1, objectives.loader_generator(spec))
        for batch_ids, tick in ((order[:spec.batch_size], 1),
                                (order[(spec.steps_per_epoch - 1) * spec.batch_size:], spec.steps_per_epoch)):
            x, y, row_ids = _inputs(bundle, batch_ids)
            context = dict(dataset=dataset, seed=spec.seed, epoch=1, global_step=tick)
            originals = objectives.build_training(spec, bundle, "cpu")
            oracle_model = copy.deepcopy(originals[0])
            oracle_optimizer = make_history_optimizer(oracle_model, originals[1].config)
            oracle_ema = copy.deepcopy(originals[2])
            pair_archive = []
            wrapped = objectives.step(*originals, x, y, row_ids, spec, context,
                                      lambda ctx, pair, raw: pair_archive.append((pair, raw)))
            direct = source_objectives.train_step(oracle_model, oracle_optimizer, oracle_ema, x, y, row_ids,
                                                  oracle_optimizer.config, context)
            hashes = _source_hashes(*originals)
            assert hashes == _source_hashes(oracle_model, oracle_optimizer, oracle_ema)
            assert wrapped["loss"] == direct["loss"]
            pair, raw = pair_archive[0]
            expected_pair = sample(x[-1].numpy(), y.numpy(), {}, False, context, row_ids)
            assert pair["pair_index_hash"] == expected_pair["pair_index_hash"]
            assert np.array_equal(pair["q"], expected_pair["q"]) and len(raw) == len(originals[1].shared)
            originals[1].reconstruction_error()
            del oracle_model, oracle_optimizer, oracle_ema, pair_archive, raw

            new_spec = RunSpec(dataset, 2022, candidate)
            adapted = objectives.build_training(new_spec, bundle, "cpu")
            seen = []
            value = objectives.step(*adapted, x, y, row_ids, new_spec, context,
                                    lambda ctx, pair, raw: seen.append(pair["pair_index_hash"]))
            assert seen == [pair["pair_index_hash"]]
            assert value["counts"]["training_forwards"] == value["counts"]["shared_autograd_calls"] == 1
            assert all(state["step"] == state["k"] == 1 for state in adapted[1].moments.values())
            outside = set(dict(adapted[0].named_parameters())) - set(dict(adapted[1].shared))
            for name in outside:
                p = dict(adapted[0].named_parameters())[name]
                q = dict(originals[0].named_parameters())[name]
                assert torch.equal(p, q)
            rec = groups.registry(adapted[0], bundle)[1]
            records.append(dict(dataset=dataset, rows=len(y), tick=tick, data_hash=bundle.data_hash,
                                group_hash=rec["group_hash"], pair_hash=seen[0], exact_h0_hashes=hashes,
                                native_complement_exact=True))
    return records


def gate_check():
    authorization = dict(study_id="history_information_20261124", split="test", dataset="movielens",
                         policy_hash=sources.sha_file(sources.OUTPUT / "TEST_POLICY.json"),
                         actual_selected_ram_get_hash_verified=True, new_test_endpoints=["selected"],
                         freeze_hash="0" * 64)
    with tempfile.TemporaryDirectory() as directory:
        try:
            sources.load_authorized_test("movielens", authorization, artifact_root=Path(directory))
        except PermissionError:
            return "blocked forged claim without freeze receipts"
    raise AssertionError("Unfrozen Test access proceeded")


def _canonical_parent_checks():
    routes = (
        ("Cstar_N", "H0_N", dict(algorithm_id="C1", zeta=0.)),
        ("HG_Cstar", "HG_H0", dict(algorithm_id="C1", zeta=0.)),
        ("G_Cstar", "G_H0", dict(algorithm_id="C1", zeta=0.)),
        ("HG_H0", "HG_B0", dict(lambda_pair=0.)),
        ("G_B1", "G_B0", dict(lambda_pair=0.)),
    )
    pair = dict(selected=0, index=0, w=np.array([.2]), q=np.array([.5]))
    records = []
    for arm, parent, override in routes:
        actual = RunSpec("movielens", 2022, arm, source_hash="a" * 64, group_hash="b" * 64,
                         dependencies=("candidate-dependency",), **override)
        reference = RunSpec("movielens", 2022, parent, source_hash="a" * 64, group_hash="c" * 64,
                            dependencies=("parent-dependency",))
        assert actual.job_key != reference.job_key
        assert actual.base_objective == reference.base_objective
        hashes = []
        for spec in (actual, reference):
            model = torch.nn.Module()
            model.local = torch.nn.Parameter(torch.tensor([1.]))
            model.native = torch.nn.Parameter(torch.tensor([2.]))
            scopes = dict(local=reference.scope, native="NATIVE_ALL")
            optimizer = HistoryUpdate(model, spec, scopes)
            ema = copy.deepcopy(model).requires_grad_(False)
            for grad in (.2, -.1):
                model.local.grad = torch.tensor([grad])
                model.native.grad = torch.tensor([grad * 2])
                if spec.lambda_pair:
                    optimizer.step(pair, [torch.tensor([.3])])
                else:
                    optimizer.step()
                v3.update_ema(ema, model, spec.rho)
            hashes.append(_source_hashes(model, optimizer, ema))
        assert hashes[0] == hashes[1], (arm, parent, hashes)
        records.append(dict(arm=arm, parent=parent, model_optimizer_ema=hashes[0]))
    return records


def _support_checks():
    n_model = torch.nn.Module()
    n_model.native = torch.nn.Parameter(torch.tensor([2.]))
    n_model.local = torch.nn.Parameter(torch.tensor([1.]))
    n_model.unused = torch.nn.Parameter(torch.tensor([3.]))
    n_spec = RunSpec("movielens", 2022, "H0_N")
    n_opt = HistoryUpdate(n_model, n_spec, {"native": "NATIVE_ALL", "local": "N", "unused": "N"})
    pair = dict(selected=0, index=0, w=np.array([.2]), q=np.array([.5]))
    n_model.local.grad = torch.zeros_like(n_model.local)
    n_model.native.grad = torch.ones_like(n_model.native)
    before = v3.state_digest((n_model.state_dict(), n_opt.state_dict()))
    try:
        n_opt.step(pair, [None, torch.ones_like(n_model.unused)])
    except RuntimeError as exc:
        assert str(exc) == "Nonzero raw PAIR gradient with native None: unused"
    else:
        raise AssertionError("Inconsistent PAIR support was silently skipped")
    assert before == v3.state_digest((n_model.state_dict(), n_opt.state_dict()))

    n_opt.step(pair, [None, None])
    assert n_opt.moments["local"]["k"] == n_opt.pair_clock == 1
    assert not n_opt.moments["local"]["q"].any() and "unused" not in n_opt.moments
    assert n_model.unused.item() == 3. and n_model.local.item() < 1.
    n_opt.step(pair, [torch.ones_like(n_model.local), torch.zeros_like(n_model.unused)])
    q = n_opt.moments["local"]["q"].clone()
    n_opt.step(pair, [torch.zeros_like(n_model.local), None])
    assert torch.equal(n_opt.moments["local"]["q"], .99 * q)
    state = n_opt.moments["local"]
    n_opt.step()
    assert (state["step"], state["k"], n_opt.pair_clock) == (3, 3, 3)
    assert n_opt.moments["local"]["step"] == 4 and n_opt.moments["local"]["k"] == 3
    assert n_opt.moments["local"]["q"] is state["q"] and "unused" not in n_opt.moments
    return dict(local_none_and_zero="verified", contradictory_support="rejected without a commit")


def optimizer_checks():
    scaled = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    plain = copy.deepcopy(scaled)
    spec = RunSpec("movielens", 2022, "G_B0")
    a = HistoryUpdate(scaled, spec, {"weight": "NATIVE_ALL"})
    b = HistoryUpdate(plain, spec, {"weight": "NATIVE_ALL"})
    c = .1
    b.adam.param_groups[0].update(weight_decay=spec.weight_decay / c, eps=1e-8 / c)
    for g in (torch.tensor([[.01, -.07]], dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64)):
        scaled.weight.grad = g * c
        plain.weight.grad = g.clone()
        a.step()
        b.step()
        close(scaled.weight, plain.weight)
        a_state, b_state = a.adam.state[scaled.weight], b.adam.state[plain.weight]
        close(a_state["exp_avg"], b_state["exp_avg"] * c)
        close(a_state["exp_avg_sq"], b_state["exp_avg_sq"] * c ** 2)
        assert a_state["step"] == b_state["step"]

    m_model = torch.nn.Module()
    m_model.shared_weight = torch.nn.Parameter(torch.tensor([2., -1.]))
    m_model.native = torch.nn.Parameter(torch.tensor([3., 1.]))
    m_spec = RunSpec("movielens", 2022, "M_C2", target_algorithm_id="SHORT_C2", alpha=.5)
    m_opt = HistoryUpdate(m_model, m_spec, {"shared_weight": "S_DENSE", "native": "NATIVE_ALL"})
    m_model.shared_weight.grad = torch.tensor([-.1, .4])
    m_model.native.grad = torch.tensor([.9, .7])
    pair = dict(selected=0, index=0, w=np.array([.2]), q=np.array([.5]))
    m_opt.step(pair, [torch.tensor([.3, .2])])
    h0 = m_opt.moments["shared_weight"]
    target = m_opt.candidate_moments["shared_weight"]
    torch.testing.assert_close(target["q"], 10 * h0["q"], rtol=1e-5, atol=1e-7)
    assert "u" not in h0 and "u" in target
    assert m_opt.last_update["STAR_SKIP"]["fp32_displacement_error"] < 1e-7
    state = copy.deepcopy(m_opt.state_dict())
    clone_model = copy.deepcopy(m_model)
    clone_opt = HistoryUpdate(clone_model, m_spec, {"shared_weight": "S_DENSE", "native": "NATIVE_ALL"})
    clone_opt.load_state_dict(state)
    assert v3.state_digest(state) == v3.state_digest(clone_opt.state_dict())
    return dict(ordinary_adam_scale_cold_hot="verified", **_support_checks(),
                conditional_parent_hashes=_canonical_parent_checks(),
                norm_match_short_shadow="verified", state_roundtrip="exact")


def _conditional_batch(bundle, spec, order, epoch=1, tick=1):
    x, y, ids = _inputs(bundle, order)
    context = dict(dataset=spec.dataset, seed=spec.seed, epoch=epoch, global_step=tick)
    support = dict(dataset=spec.dataset, rows=len(ids), global_step=tick, data_hash=bundle.data_hash,
                   row_ids_hash=hashlib.sha256(ids.tobytes()).hexdigest(),
                   domain_rows=np.bincount(x[-1].numpy().astype(np.int64), minlength=3).tolist())
    return x, y, ids, context, support


def _n_source_contract(bundle):
    spec = RunSpec(bundle.dataset, 2022, "H0_N")
    order = objectives.training_order(spec, bundle, 1, objectives.loader_generator(spec))[:256]
    x, y, ids, context, support = _conditional_batch(bundle, spec, order)
    variants = {}
    for arm in ("H0_N", "C1_Z050"):
        variant = RunSpec(bundle.dataset, 2022, arm)
        model, opt, ema = objectives.build_training(variant, bundle, "cpu")
        scopes, identity = groups.registry(model, bundle)
        expected = {name for name, scope in scopes.items() if scope == variant.scope}
        assert set(dict(opt.shared)) == expected
        if arm == "H0_N":
            domain_name = f"embedding.embed_dict.{bundle.features.columns.index('domain_indicator')}.weight"
            domain = [record for record in identity["records"] if record["name"] == domain_name]
            assert [r["mask"] for r in domain] == [{"axis": 0, "index": i} for i in range(3)]
            assert domain_name in expected and all(r["group"].startswith(("D", "T"))
                                                  for r in identity["records"] if r["name"] in expected)
        outside = [(name, p) for name, p in model.named_parameters() if name not in expected]
        oracle_params = [torch.nn.Parameter(p.detach().clone()) for _, p in outside]
        oracle = torch.optim.Adam(oracle_params, **opt.adam.defaults)

        def native_oracle(ctx, pair, raw):
            assert pair is not None and len(raw) == len(opt.shared)
            for (_, parameter), clone in zip(outside, oracle_params):
                clone.grad = None if parameter.grad is None else parameter.grad.clone()
            oracle.step()

        value = objectives.step(model, opt, ema, x, y, ids, variant, context, native_oracle)
        assert value["diagnostics"]["pair_count"] == 2048
        assert all(torch.equal(p, clone) for (_, p), clone in zip(outside, oracle_params))
        native_hash = v3.state_digest(oracle.state_dict())
        assert v3.state_digest(opt.adam.state_dict()) == native_hash
        for name, p in opt.shared:
            if p.grad is None:
                assert name not in opt.moments
            else:
                assert (opt.moments[name]["step"], opt.moments[name]["k"]) == (1, 1)
        variants[variant.scope] = dict(group_hash=identity["scope_hash"][variant.scope],
                                       native_state_hash=native_hash, native_parameters=len(outside),
                                       pair_hash=value["diagnostics"]["pair_index_hash"])
    assert variants["N"]["pair_hash"] == variants["S_DENSE"]["pair_hash"]
    return support | dict(scopes=variants)


def _hypergate_source_contract(bundle):
    spec = RunSpec(bundle.dataset, 2022, "HG_B0")
    cfg = objectives._published_config(spec)
    model, opt, ema = objectives.build_training(spec, bundle, "cpu")
    order = objectives.training_order(spec, bundle, 1, objectives.loader_generator(spec))
    assert (spec.batch_size, spec.eval_batch_size, spec.steps_per_epoch) == (
        cfg["batch_size"], cfg["eval_batch_size"], (len(order) + spec.batch_size - 1) // spec.batch_size)
    x, y, ids, context, support = _conditional_batch(bundle, spec, order[:spec.batch_size])
    oracle_model = copy.deepcopy(model)
    oracle_opt = published._optimizers(oracle_model, "HYPERGATE", "train", cfg)["base"]
    assert model.training and any(isinstance(m, torch.nn.Dropout) and m.training and m.p == .3
                                  for m in model.modules())
    seen, forwarded = [], []
    original_base = objectives._published_base

    def record_base(*args):
        value = original_base(*args)
        forwarded.append(value[2].detach())
        return value

    def record_terms(*args, **kwargs):
        value = original_terms(*args, **kwargs)
        seen.append(value[0].detach())
        return value

    rng = torch.get_rng_state()
    original_terms = model._forward_terms
    with patch.object(model, "_forward_terms", side_effect=record_terms), patch.object(
            objectives, "_published_base", side_effect=record_base):
        actual = objectives.step(model, opt, ema, x, y, ids, spec, context)
    assert len(seen) == len(forwarded) == 1 and torch.equal(seen[0], forwarded[0])
    assert torch.equal(actual["pred"], seen[0].sigmoid())
    after = torch.get_rng_state()
    assert not torch.equal(rng, after)
    torch.set_rng_state(rng)
    direct = oracle_model.objective(x.T, y, torch.ones_like(y, dtype=torch.bool))
    direct["loss"].backward()
    oracle_opt.step()
    assert torch.equal(after, torch.get_rng_state())
    assert actual["diagnostics"]["base_components"] == {name: float(value.detach()) for name, value in direct.items()}
    assert v3.state_digest(model.state_dict()) == v3.state_digest(oracle_model.state_dict())
    adam_hash = v3.state_digest(oracle_opt.state_dict())
    assert v3.state_digest(opt.adam.state_dict()) == adam_hash
    flags = {key: opt.adam.defaults[key] for key in ("foreach", "fused", "amsgrad")}
    assert flags == {key: oracle_opt.defaults[key] for key in flags}
    pair_hashes = {}
    for arm in ("HG_B1", "HG_H0"):
        variant = RunSpec(bundle.dataset, 2022, arm)
        model, opt, ema = objectives.build_training(variant, bundle, "cpu")
        value = objectives.step(model, opt, ema, x, y, ids, variant, context)
        assert value["counts"]["training_forwards"] == 1 and value["diagnostics"]["pair_count"] == 2048
        assert value["diagnostics"]["base_components"] == actual["diagnostics"]["base_components"]
        assert value["counts"]["shared_autograd_calls"] == int(arm == "HG_H0")
        pair_hashes[arm] = value["diagnostics"]["pair_index_hash"]
    assert len(set(pair_hashes.values())) == 1
    return support | dict(batch_size=spec.batch_size, tail_rows=len(order) % spec.batch_size,
                          steps_per_epoch=spec.steps_per_epoch, native_components=list(direct),
                          logits_hash=v3.state_digest(seen[0]), native_adam_hash=adam_hash,
                          adam_flags=flags, pair_hashes=pair_hashes, train_dropout_rng_exact=True)


@torch.no_grad()
def _gaussian_source_contract(bundle, coverage):
    from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses

    spec = RunSpec(bundle.dataset, 2022, "G_H0")
    model, _, _ = objectives.build_training(spec, bundle, "cpu")
    order = objectives.training_order(spec, bundle, 1, objectives.loader_generator(spec))
    records = []
    for tick in (1, spec.steps_per_epoch):
        start = (tick - 1) * spec.batch_size
        x, y, ids, context, support = _conditional_batch(bundle, spec, order[start:start + spec.batch_size], tick=tick)
        expected_rows = spec.batch_size if tick == 1 else {"movielens": 21927, "kuairand": 40052}[bundle.dataset]
        assert len(ids) == expected_rows and len(np.unique(ids)) == expected_rows
        base, probability, logits, info, work = objectives._m3oe_base(model, x, y, spec, context)
        kappa = {"movielens": .05, "kuairand": .1}[bundle.dataset]
        if bundle.dataset == "kuairand":
            direct, detail = coverage.losses(model, x, y, (0,), seed=spec.seed, step=tick, kappa=kappa)
        else:
            direct, detail = losses(model, x, y, 0, dict(num_views=1, view_family="gaussian", pairing="iid"),
                                    spec.dataset, spec.seed, tick, kappa=kappa)
        assert torch.equal(base, direct) and torch.equal(probability, detail["pred"])
        torch.testing.assert_close(logits.sigmoid(), probability, atol=1e-7, rtol=1e-5)
        domain_rows = support["domain_rows"][0]
        assert torch.equal(detail["mask"], x[-1] == 0)
        assert detail["extra_views"] == work["extra_view_rows"] == domain_rows
        assert work["training_forwards"] == detail["forwards"] == 2 and len(info["gaussian_views"]) == 1
        key = f"GA31-R3|{spec.dataset}|{spec.seed}|train|{tick}|gaussian|1"
        noise_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1 << 63) - 1)
        generator = torch.Generator(device=x.device).manual_seed(noise_seed)
        fields = 6 if spec.dataset == "movielens" else 33
        assert len(model.perturb_slices) == fields and all(hi - lo == 16 for lo, hi in model.perturb_slices)
        noise = torch.stack([torch.randn((domain_rows, 16), generator=generator) for _ in range(fields)], dim=1)
        noise_hash = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        reference_view, = detail["views"]
        assert torch.equal(reference_view["noise"], noise)
        assert info["gaussian_views"][0] == dict(domain=0, rows=domain_rows, key=key, seed=noise_seed,
                                                  noise_hash=reference_view["hash"])
        assert reference_view["hash"] == noise_hash
        pair = sample(x[-1].numpy(), y.numpy(), {}, False, context, ids)
        assert len(pair["pos"]) == len(pair["neg"]) == 2048
        records.append(support | dict(d0_rows=domain_rows, perturbed_fields=fields, noise_key=key,
                                      noise_seed=noise_seed, noise_hash=noise_hash, pair_hash=pair["pair_index_hash"],
                                      d0_row_ids_hash=hashlib.sha256(ids[detail["mask"].numpy()].tobytes()).hexdigest(),
                                      base_loss=float(base), exact_source_target=True))
    return records


def _autoifs_source_contract(bundle):
    spec = RunSpec(bundle.dataset, 2022, "REPAIR_AUTOIFS")
    cfg = objectives._published_config(spec)
    model, opt, ema = objectives.build_training(spec, bundle, "cpu")
    oracle = published.create_session(dict(dataset=spec.dataset, seed=spec.seed, config=cfg,
                                           method="AUTOIFS"), bundle, "cpu")
    source_model = oracle["model"]
    assert (spec.batch_size, cfg["search_epochs"], cfg["reuse_epochs"], spec.epochs) == (4096, 10, 50, 60)
    for epoch in range(1, 11):
        objectives.prepare_epoch(model, opt, ema, spec, epoch)
        source_epoch = epoch - 1
        if source_epoch == cfg["rewind_epoch_index"]:
            oracle["rewind_state"] = v3.cpu_copy(source_model.state_dict())
        if source_epoch:
            source_model.network.temp *= (cfg["temperature_end"] / cfg["temperature_start"]) ** (
                1. / (cfg["search_epochs"] - 1))
    assert model.network.temp == source_model.network.temp
    rewind_hash = v3.state_digest(oracle["rewind_state"])
    assert v3.state_digest(opt.session["rewind_state"]) == rewind_hash

    def compare_batch(epoch, source_epoch, optimizer_calls):
        order = objectives.training_order(spec, bundle, epoch, objectives.loader_generator(spec))
        assert np.array_equal(order, published._stage_rng(spec.seed, spec.dataset, "AUTOIFS", source_epoch)
                              .permutation(len(bundle.train_y)))
        tick = (epoch - 1) * spec.steps_per_epoch + 1
        x, y, ids, context, support = _conditional_batch(bundle, spec, order[:256], epoch=epoch, tick=tick)
        rng = torch.get_rng_state()
        actual = objectives.step(model, opt, ema, x, y, ids, spec, context)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        for native in oracle["optimizers"].values():
            native.zero_grad(set_to_none=True)
        direct = oracle["model"].objective(x.T, y, torch.ones_like(y, dtype=torch.bool))
        direct["loss"].backward()
        for native in oracle["optimizers"].values():
            native.step()
        assert torch.equal(after, torch.get_rng_state())
        assert actual["diagnostics"]["base_components"] == {name: float(value.detach()) for name, value in direct.items()}
        assert actual["counts"]["optimizer_steps"] == optimizer_calls
        model_hash = v3.state_digest(oracle["model"].state_dict())
        assert v3.state_digest(model.state_dict()) == model_hash
        hashes = {name: v3.state_digest(native.state_dict()) for name, native in oracle["optimizers"].items()}
        assert {name: v3.state_digest(native.state_dict()) for name, native in opt.session["optimizers"].items()} == hashes
        return support | dict(stage=oracle["stage"], source_epoch=source_epoch, optimizer_steps=optimizer_calls,
                              exact_model_hash=model_hash, exact_optimizer_hashes=hashes)

    search = compare_batch(10, 9, 2)
    before_reuse = torch.get_rng_state()
    objectives.finish_epoch(model, opt, ema, spec, 10)
    after_reuse = torch.get_rng_state()
    torch.set_rng_state(before_reuse)
    published._start_reuse(oracle)
    assert torch.equal(after_reuse, torch.get_rng_state())
    assert opt.session["model"] is model and opt.session["stage"] == "reuse"
    assert opt.calls == 1 and not opt.adam.state and published._selector_matches(model, oracle["selector_state"])
    assert objectives.inference_meta(model, spec) == dict(ticket=True, temp=cfg["temperature_end"], gate_frozen=True)
    transition_hash = v3.state_digest(oracle["model"].state_dict())
    assert v3.state_digest(model.state_dict()) == transition_hash
    reuse = compare_batch(11, 101, 1)
    restored = copy.deepcopy(opt.state_dict())
    opt.load_state_dict(restored)
    assert v3.state_digest(opt.state_dict()) == v3.state_digest(restored)
    return dict(dataset=spec.dataset, search=search, reuse=reuse, rewind_hash=rewind_hash,
                exact_ticket_model_hash=transition_hash, fresh_reuse_adam=True, state_roundtrip=True)


def conditional_contracts(module, root):
    """Execute one conditional source oracle using verified Train/Validation inputs."""
    root = Path(root)
    audit = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    source = dict(source_hash=audit["source_hash"], device="cpu", seed=2022)
    if module == "Gaussian":
        gaussian = audit["gaussian_source"]
        current = sources.audit_gaussian_source(sources.ROOT.parent / "phase0-3_git", audit["published_commit"])
        if current["status"] != "VERIFIED" or not (
            gaussian == current or gaussian["status"] == "UNRESOLVED_IDENTITY"
        ):
            raise ValueError(f"Gaussian source does not match S0: {current['blocker']}")
        entry, = (row for row in current["files"] if row["path"] == f"{sources.GAUSSIAN_PACKAGE}/objectives.py")
        loader = importlib.util.spec_from_file_location("_ga31_r3_coverage_oracle", entry["actual_path"])
        coverage = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(coverage)
        source.update(gaussian_source_hash=current["source_hash"], commit=audit["published_commit"])
    else:
        check = {"N": _n_source_contract, "HyperGate": _hypergate_source_contract,
                 "AutoIFS": _autoifs_source_contract}[module]
        source["commit"] = audit["evidence_commit" if module == "N" else "published_commit"]
    support = [_support_checks()] if module == "N" else []
    rng = v3.rng_state("cpu")
    try:
        for dataset in ("movielens", "kuairand"):
            bundle = sources.load_train_validation(dataset, root / "train_validation_inputs")
            assert not bundle.identity["include_test"]
            if module == "Gaussian":
                support.extend(_gaussian_source_contract(bundle, coverage))
            else:
                support.append(check(bundle))
    finally:
        v3.restore_rng(rng, "cpu")
    return dict(source=source, support=support, result="PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    report = dict(algebra=algebra_checks(), optimizer=optimizer_checks(), test_gate=gate_check())
    if args.real:
        report["first_tail"] = real_batch_checks()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
