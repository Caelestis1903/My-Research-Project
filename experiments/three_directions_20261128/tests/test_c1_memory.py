"""C1 bank, objective, prefix and complete-epoch engineering checks."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.conditional_history_shared_update_20261122.protocol import append, write
from experiments.conditional_history_shared_update_20261122.runner import batch_inputs
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import rng
from experiments.three_directions_20261128 import a_step, c0_audit, c1_protocol, c_step, memory_bank, ram_state
from experiments.three_directions_20261128.macro_pair_sampler import PAIR_COUNT, MacroPairSampler, ordered_ids_hash
from experiments.three_directions_20261128.protocol import READOUTS, RunSpec, algorithm_digest, digest, read, sha, v3
from experiments.three_directions_20261128.tests.test_direction_a import TinyM3oE

SEED = 2022
SNAPSHOTS = [dict(label=label, model_hash=str(i) * 64) for i, label in enumerate(c0_audit.SNAPSHOTS, 1)]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bundle_fixture(repeats):
    domain = np.repeat(np.arange(3), 4 * repeats)
    y = np.tile(np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), repeats, axis=0), (3, 1))
    x = np.column_stack((np.linspace(-.6, .6, len(y)), y[:, 1], domain)).astype(np.float32)
    ids = (100 + 2 * np.arange(len(y), dtype=np.int64))[::-1].copy()
    return SimpleNamespace(dataset="movielens", train_x=x, train_y=y, train_row_ids=ids,
                           identity={"split_row_fingerprints": {"train": "fixture"}})


def sampler_for(bundle):
    return MacroPairSampler(bundle.dataset, bundle.train_x[:, -1], bundle.train_y, bundle.train_row_ids, "fixture")


def save_inputs(root, sampler, candidates, margins, *, teacher=None, prefix=None):
    candidate = c0_audit.save_candidates(root, sampler, SEED, candidates)
    candidate.update(sampler=sampler.state_dict(),
                     condition_hashes=[dict(condition=list(k), hash=c["hash"], ordered_ids_hash=c["ordered_ids_hash"])
                                       for k, c in candidates.items()])
    snapshots = copy.deepcopy(SNAPSHOTS)
    if teacher is None:
        teacher = dict(epoch=10, global_step=10, model_hash=snapshots[1]["model_hash"], online_hash="online")
    else:
        snapshots[1]["model_hash"] = teacher["model_hash"]
    raw = np.concatenate([margins[k] for k in candidates])
    margin_receipt = c0_audit.save_arrays(
        root / "margins.npz", E05=np.full(len(raw), -33.), E10=raw, selected=np.full(len(raw), 33.),
        condition_hashes=np.array([c["hash"] for c in candidates.values()]),
        candidate_sha256=np.array(candidate["sha256"]), model_hashes=np.array([s["model_hash"] for s in snapshots]))
    return dict(candidate=candidate, margins=margin_receipt, teacher=teacher, snapshots=snapshots, prefix=prefix)


def reference_indices(context, label, condition, support, size):
    stream = "|".join([label, *(str(v) for v in condition)])
    return np.sort(rng(context, stream).choice(support, size, replace=False))


def test_bank_has_strict_sign_order_capacity_and_random_control(tmp_path):
    sampler = sampler_for(bundle_fixture(70))
    candidates = c0_audit.candidate_set(sampler, SEED)
    special, empty, unit = tuple(candidates)[:3]
    raw = {key: np.r_[np.linspace(.01, 9., 3072), -np.ones(1024)] for key in candidates}
    raw[special].fill(-8.)
    raw[special][:5] = [np.nextafter(0., 1.), -8., 0., 9., -0.]
    raw[empty].fill(0.)
    raw[unit].fill(-1.)
    raw[unit][50] = .1
    inputs = save_inputs(tmp_path, sampler, candidates, raw)
    banks, summaries = memory_bank.prepare_banks(sampler, SEED, inputs)
    repeated, _ = memory_bank.prepare_banks(sampler, SEED, inputs)
    random, correct = (banks[mode]["conditions"] for mode in ("random", "correct"))
    context = dict(dataset=sampler.dataset, seed=SEED, epoch=10, global_step=10)
    for key, candidate in candidates.items():
        selected = np.flatnonzero(raw[key] > 0.)
        capacity = min(2048, len(selected))
        expected_correct = reference_indices(context, "c1_bank_correct", key, selected, capacity)
        expected_random = reference_indices(context, "c1_bank_random", key, np.arange(len(raw[key])), capacity)
        for actual, expected in ((correct[key], expected_correct), (random[key], expected_random)):
            np.testing.assert_array_equal(actual["indices"], expected)
            np.testing.assert_array_equal(actual["offsets"], np.stack((candidate["left"][expected], candidate["right"][expected])))
            np.testing.assert_array_equal(actual["ids"], sampler.row_ids[actual["offsets"]])
            np.testing.assert_array_equal(actual["target_margins"], np.clip(raw[key][expected], -5, 5))
            assert len(np.unique(actual["indices"])) == capacity
            assert not actual["target_margins"].flags.writeable
        assert (correct[key]["target_margins"] > 0.).all()
    assert correct[special]["indices"].tolist() == [0, 3]
    assert correct[special]["target_margins"].tolist() == [np.nextafter(0., 1.), 5.]
    assert raw[special][random[special]["indices"]].min() == -8.
    assert len(correct[empty]["indices"]) == len(random[empty]["indices"]) == 0
    assert correct[unit]["indices"].tolist() == [50]
    assert summaries[0]["raw_tied"] == 2 and summaries[0]["correct"]["clip_saturation_high"] == 1
    assert summaries[0]["overlap"] == len(np.intersect1d(correct[special]["indices"], random[special]["indices"]))
    assert banks["random"]["bank_hash"] != banks["correct"]["bank_hash"]
    for mode in banks:
        memory_bank.validate_bank(banks[mode], repeated[mode])
    broken = copy.deepcopy(banks["correct"])
    broken["conditions"][special]["target_margins"] = broken["conditions"][special]["target_margins"][::-1]
    with pytest.raises(ValueError, match="bank bytes"):
        memory_bank.validate_bank(broken, banks["correct"])


@pytest.mark.parametrize("corruption", ["ordered_ids", "condition_hashes", "model_hashes", "margin_dtype", "nonfinite", "file_sha"])
def test_reject_c0_archive_identity_before_bank_build(tmp_path, corruption):
    sampler = sampler_for(bundle_fixture(3))
    candidates = c0_audit.candidate_set(sampler, SEED)
    raw = {key: np.ones(len(c["ranks"])) for key, c in candidates.items()}
    inputs = save_inputs(tmp_path, sampler, candidates, raw)
    if corruption == "file_sha":
        inputs["margins"]["sha256"] = "changed"
    else:
        candidate_error = corruption in ("ordered_ids", "condition_hashes")
        receipt_name = "candidate" if candidate_error else "margins"
        receipt = inputs[receipt_name]
        with np.load(receipt["path"], allow_pickle=False) as archive:
            fields = dict(archive)
        if corruption == "ordered_ids":
            fields["positive_ids"][0], fields["negative_ids"][0] = fields["negative_ids"][0], fields["positive_ids"][0]
        elif corruption == "condition_hashes":
            fields["condition_hashes"] = fields["condition_hashes"][::-1]
        elif corruption == "model_hashes":
            fields["model_hashes"] = fields["model_hashes"][::-1]
        elif corruption == "margin_dtype":
            fields["E10"] = fields["E10"].astype(np.float32)
        else:
            fields["E10"][0] = np.nan
        receipt.update(c0_audit.save_arrays(Path(receipt["path"]), **fields))
    with pytest.raises(ValueError, match="C0"):
        memory_bank.prepare_banks(sampler, SEED, inputs)


def test_repeated_draws_same_positions_exact_c4_layout_and_empty_fallback(tmp_path):
    sampler = sampler_for(bundle_fixture(3))
    candidates = c0_audit.candidate_set(sampler, SEED)
    context = dict(dataset=sampler.dataset, seed=SEED, epoch=11, global_step=11)
    current = sampler.sample_macro6(context)
    condition = current["condition"]
    raw = {key: np.zeros(len(c["ranks"])) for key, c in candidates.items()}
    raw[condition][[0, 3, 8]] = [.5, 1., 2.]
    inputs = save_inputs(tmp_path, sampler, candidates, raw)
    banks, _ = memory_bank.prepare_banks(sampler, SEED, inputs)
    before = v3.state_digest(v3.rng_state())
    pairs = {arm: memory_bank.draw_pair(banks[mode], current, dict(context, arm=arm, attempt="retry"))
             for arm, mode in memory_bank.MODES.items()}
    assert before == v3.state_digest(v3.rng_state())
    c2, c3, c4 = (pairs[arm] for arm in ("C2", "C3", "C4"))
    assert c2["draw_positions_hash"] == c3["draw_positions_hash"] == c4["draw_positions_hash"]
    assert c3["pair_index_hash"] == c4["pair_index_hash"] == ordered_ids_hash(c3["ids"])
    stream = "c1_memory_draw|" + "|".join(map(str, condition))
    positions = rng(context, stream).choice(3, PAIR_COUNT, replace=True)
    assert c3["draw_positions_hash"] == ordered_ids_hash(positions)
    assert c3["w"] / c3["q"] == current["w"] / current["q"]
    assert len(c3["ids"]) == 2 * PAIR_COUNT and c3["unique_pairs"] == len(set(positions)) == 3
    entry = banks["correct"]["conditions"][condition]
    np.testing.assert_array_equal(c3["ids"].reshape(2, PAIR_COUNT), entry["ids"][:, positions])
    np.testing.assert_array_equal(c3["target_margins"], entry["target_margins"][positions])
    assert memory_bank.draw_pair(banks["correct"], None, context) is None
    while current["condition"] == condition:
        context["global_step"] += 1
        current = sampler.sample_macro6(context)
    fallback = memory_bank.draw_pair(banks["correct"], current, context)
    assert fallback["memory_source"] == "current_fallback" and fallback["memory_miss"] == "empty_bank"
    for key in current:
        assert fallback[key] is current[key]


def test_stable_fp32_soft_and_hard_reference_gradients():
    margin = torch.tensor([-1000., -1., 0., 2., 1000.], requires_grad=True)
    teacher = torch.tensor([-5., -.5, 0., 1.5, 5.], requires_grad=True)
    pair = dict(memory_source="bank", target_margins=teacher)
    config = dict(tau=1.)
    target = teacher.detach().double().sigmoid().numpy()
    raw = margin.detach().double().numpy()
    expected_soft = np.logaddexp(0., raw) - raw * target
    expected_sigmoid = np.exp(-np.logaddexp(0, -raw))
    for arm in ("C2", "C3", "C4"):
        obtained = c_step.pair_loss(margin, pair, config, arm)
        wanted = np.logaddexp(0, -raw) if arm == "C4" else expected_soft
        gradient_target = 1. if arm == "C4" else target
        assert obtained.dtype == torch.float32 and float(obtained.detach()) == pytest.approx(wanted.mean(), abs=2e-5)
        gradient, teacher_gradient = torch.autograd.grad(obtained, (margin, teacher), allow_unused=True)
        assert teacher_gradient is None
        np.testing.assert_allclose(gradient.detach().numpy(), (expected_sigmoid - gradient_target) / len(raw), atol=3e-8)
    fallback = dict(memory_source="current_fallback")
    assert torch.equal(c_step.pair_loss(margin, fallback, config, "C2"), torch.nn.functional.softplus(-margin).mean())


@pytest.fixture
def replay_fixture(tmp_path, monkeypatch):
    sampler = sampler_for(bundle_fixture(2))
    candidates = c0_audit.candidate_set(sampler, SEED)
    context = dict(dataset=sampler.dataset, seed=SEED, epoch=11, global_step=11)
    sample = sampler.sample_macro6
    bank_condition = sample(context)["condition"]
    raw = {condition: np.array([-.1, .2, -.3, .4]) if condition == bank_condition else np.zeros(4)
           for condition in candidates}
    banks, _ = memory_bank.prepare_banks(sampler, SEED, save_inputs(tmp_path, sampler, candidates, raw))
    bank = banks["correct"]
    fallback_step = 12
    while sample(dict(context, global_step=fallback_step))["condition"] == bank_condition:
        fallback_step += 1
    no_event_step = fallback_step + 1
    monkeypatch.setattr(sampler, "sample_macro6",
                        lambda ctx: None if ctx["global_step"] == no_event_step else sample(ctx))
    rows = []
    for step in range(1, no_event_step + 1):
        context = dict(dataset=sampler.dataset, seed=SEED, epoch=min(step, 11), global_step=step)
        current = sampler.sample_macro6(context)
        applied = current if step <= 10 else memory_bank.draw_pair(bank, current, context)
        diagnostics = dict(train_identity=sampler.train_identity, memory_phase="warmup" if step <= 10 else "memory")
        if step > 10:
            diagnostics["bank_hash"] = bank["bank_hash"]
        if applied is None:
            diagnostics.update(condition=None, current_pair_index_hash=None, pair_index_hash=None,
                               memory_source="none", memory_miss="no_event", w=None, q=None, w_over_q=None,
                               unique_pairs=0, unique_endpoints=0)
            counts = dict(pair_draws=0, pair_rows=0, pair_forwards=0, pair_diagnostic_vjp_calls=0)
        else:
            diagnostics.update(condition=list(applied["condition"]), current_pair_index_hash=current["pair_index_hash"],
                               pair_index_hash=applied["pair_index_hash"], w_over_q=applied["w"] / applied["q"],
                               memory_source=applied.get("memory_source", "current"),
                               memory_miss=applied.get("memory_miss"))
            diagnostics.update({key: applied[key] for key in ("w", "q", "unique_pairs", "unique_endpoints")})
            if diagnostics["memory_source"] == "bank":
                diagnostics["draw_positions_hash"] = applied["draw_positions_hash"]
            counts = dict(pair_draws=PAIR_COUNT, pair_rows=2 * PAIR_COUNT, pair_forwards=1, pair_diagnostic_vjp_calls=1)
        rows.append(dict(context=context, diagnostics=diagnostics, counts=counts))
    return sampler, bank, rows


def test_committed_replay_exposure_and_hash_rejection(replay_fixture):
    sampler, bank, rows = replay_fixture
    output = memory_bank.replay_exposure(sampler, bank, rows)
    memory, = [r for r in output if r["phase"] == "memory" and r.get("bank_size")]
    assert memory["events"] == len(rows) - 12 and memory["draws"] == memory["events"] * PAIR_COUNT
    assert memory["bank_size"] == memory["unique_pairs"] == memory["current_overlap"] == memory["uniform_bank_unique"] == 2
    assert memory["expected_draws_per_bank_member"] == memory["draws"] / 2
    fallback, = [r for r in output if r["phase"] == "memory" and r.get("bank_size") == 0]
    assert fallback["events"] == fallback["misses"] == 1 and fallback["draws"] == PAIR_COUNT
    no_event, = [r for r in output if r["condition"] is None]
    assert no_event == dict(phase="memory", condition=None, events=0, draws=0, misses=1)
    rows[10]["diagnostics"]["draw_positions_hash"] = "changed"
    with pytest.raises(ValueError, match="draw positions"):
        memory_bank.replay_exposure(sampler, bank, rows)


@pytest.mark.parametrize("source,group,field,value", [
    ("bank", "counts", "pair_rows", 0),
    ("bank", "counts", "pair_forwards", 0),
    ("bank", "counts", "pair_diagnostic_vjp_calls", 0),
    ("bank", "diagnostics", "w_over_q", 0.),
    ("bank", "diagnostics", "memory_miss", "empty_bank"),
    ("current_fallback", "diagnostics", "memory_miss", None),
    ("none", "counts", "pair_draws", PAIR_COUNT),
    ("none", "counts", "pair_rows", 2 * PAIR_COUNT),
    ("none", "counts", "pair_forwards", 1),
    ("none", "counts", "pair_diagnostic_vjp_calls", 1),
    ("none", "diagnostics", "w", 1.),
    ("none", "diagnostics", "q", 1.),
    ("none", "diagnostics", "w_over_q", 0.),
    ("none", "diagnostics", "unique_pairs", 1),
    ("none", "diagnostics", "unique_endpoints", 2),
    ("none", "diagnostics", "memory_miss", None),
])
def test_committed_replay_rejects_inconsistent_work_or_exposure(replay_fixture, source, group, field, value):
    sampler, bank, rows = replay_fixture
    row = next(row for row in rows if row["diagnostics"]["memory_source"] == source)
    row[group][field] = value
    with pytest.raises(ValueError, match="Committed C1"):
        memory_bank.replay_exposure(sampler, bank, rows)


def _parts(bundle, arm="P0", inputs=None):
    torch.manual_seed(SEED)
    model = TinyM3oE()
    ema = copy.deepcopy(model).requires_grad_(False)
    loader = torch.Generator().manual_seed(SEED + 11000)
    job = SimpleNamespace(config=a_step.a_config("P0", bundle.dataset), arm=arm, seed=SEED)
    if arm == "P0":
        optimizer, adapter = a_step.make_training_parts(model, job, bundle)
    else:
        optimizer, adapter = c_step.make_training_parts(model, job, bundle, inputs, lambda _inputs: None)
        adapter.verify_init(v3.state_digest(model.state_dict()))
    return model, optimizer, ema, loader, adapter, job.config


def _reduced_epoch(bundle, parts, epoch):
    model, optimizer, ema, loader, adapter, config = parts
    order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
    x, y, rows = batch_inputs(bundle, order, "cpu")
    context = dict(dataset=bundle.dataset, seed=SEED, epoch=epoch, global_step=epoch)
    result = adapter.step(model, optimizer, ema, x, y, rows, config, context)
    order_hash = v3.state_digest(rows)
    if isinstance(adapter, c_step.DirectionCAdapter):
        adapter.finish_epoch(epoch, epoch, order_hash, model, optimizer, ema, loader, "cpu")
    state = v3.cpu_copy(dict(epoch=epoch, global_step=epoch, permutation=order, cursor=len(order),
                             epoch_metrics=dict(order_hash=order_hash), online=model.state_dict(), ema=ema.state_dict(),
                             optimizer=optimizer.state_dict(), loader=loader.get_state(), rng=v3.rng_state(),
                             adapter=adapter.state_dict()))
    return state, c_step.prefix_step(order_hash, result["diagnostics"])


def _restore(parts, state):
    model, optimizer, ema, loader, adapter, _ = parts
    adapter.restore_epoch(state)
    model.load_state_dict(state["online"])
    optimizer.load_state_dict(state["optimizer"])
    ema.load_state_dict(state["ema"])
    loader.set_state(state["loader"])
    v3.restore_rng(state["rng"])


def test_stock_prefix_e10_install_ram_rollback_and_deterministic_replay(tmp_path):
    bundle = bundle_fixture(2)
    stock = _parts(bundle)
    init = v3.state_digest(stock[0].state_dict())
    baseline = [_reduced_epoch(bundle, stock, epoch) for epoch in range(1, 11)]
    expected = baseline[-1][0]
    prefix = dict(steps_per_epoch=1, init_model_hash=init, batches=[row for _, row in baseline],
                  order_hashes=[state["epoch_metrics"]["order_hash"] for state, _ in baseline])
    prefix["hash"] = digest(prefix)
    teacher = dict(epoch=10, global_step=10, model_hash=v3.state_digest(expected["ema"]),
                   online_hash=v3.state_digest(expected["online"]))
    sampler = stock[4].sampler
    candidates = c0_audit.candidate_set(sampler, SEED)
    raw = {k: np.array([-2., 0., .5, 7.]) for k in candidates}
    inputs = save_inputs(tmp_path, sampler, candidates, raw, teacher=teacher, prefix=prefix)
    c2, c3 = (_parts(bundle, arm, inputs) for arm in ("C2", "C3"))
    adapter = c3[4]
    journal = ram_state.Journal(tmp_path / "journal", dict(stage="C1", job_key="fixture"))
    try:
        with journal.writer():
            attempt = journal.new_attempt("fixture")
            for epoch in range(1, 11):
                other, _ = _reduced_epoch(bundle, c2, epoch)
                state, _ = _reduced_epoch(bundle, c3, epoch)
                assert all(v3.equal(state[k], baseline[epoch - 1][0][k])
                           for k in ("online", "optimizer", "ema", "loader", "rng"))
                assert state["adapter"]["boundary"] == other["adapter"]["boundary"]
                receipt = journal.commit_epoch(attempt, state)
            installed = adapter.bank["bank_hash"]
            assert installed != c2[4].bank["bank_hash"]
            e10, _ = journal.recover()
            restored = _parts(bundle, "C3", inputs)
            _restore(restored, e10)
            assert restored[4].bank["bank_hash"] == installed
            wrong = copy.deepcopy(e10)
            wrong["online"]["offset"].add_(1)
            with pytest.raises(ValueError, match="E10 model"):
                restored[4].restore_epoch(wrong)
            corrupted = copy.deepcopy(e10)
            corrupted["adapter"]["bank"]["binding"]["epoch"] = 9
            with pytest.raises(ValueError, match="bank identity"):
                restored[4].restore_epoch(corrupted)
            with ram_state._opened(receipt["name"]) as packet:
                packet.buf[0] ^= 1
            rollback, _ = journal.recover()
            assert rollback["epoch"] == 9
            _restore(restored, rollback)
            assert restored[4].bank is None
            retried, _ = _reduced_epoch(bundle, restored, 10)
            journal.commit_epoch(attempt, retried)
            assert v3.equal(retried, e10)
            ctx = dict(dataset=bundle.dataset, seed=SEED, epoch=11, global_step=11)
            current = sampler.sample_macro6(ctx)
            assert v3.equal(memory_bank.draw_pair(adapter.bank, current, ctx),
                            memory_bank.draw_pair(restored[4].bank, current, ctx))
    finally:
        if journal.path.exists():
            for receipt in read(journal.path)["epochs"]:
                ram_state._unlink(receipt["name"])


def test_bad_warmup_pair_or_e10_teacher_cannot_install(tmp_path):
    bundle = bundle_fixture(2)
    stock = _parts(bundle)
    expected = _reduced_epoch(bundle, stock, 1)[1]
    prefix = dict(steps_per_epoch=1, init_model_hash=v3.state_digest(stock[0].state_dict()), batches=[expected],
                  order_hashes=[expected[0]] * 10, hash="prefix")
    parts = _parts(bundle)
    model, optimizer, ema, loader, stock_adapter, config = parts
    prefix["init_model_hash"] = v3.state_digest(model.state_dict())
    inputs = dict(prefix=prefix, teacher=dict(epoch=10, global_step=10, model_hash="incorrect", online_hash="incorrect"))
    adapter = c_step.DirectionCAdapter(stock_adapter, SEED, "C3", inputs,
                                       lambda _inputs: pytest.fail("read before E10 match"))
    order = np.arange(len(bundle.train_y))
    x, y, rows = batch_inputs(bundle, order, "cpu")
    ctx = dict(dataset=bundle.dataset, seed=SEED, epoch=1, global_step=1)
    with pytest.raises(ValueError, match="prefix batch"):
        adapter.step(model, optimizer, ema, x, y, rows, config, ctx)
    with pytest.raises(ValueError, match="row order"):
        adapter.finish_epoch(10, 10, "wrong order", model, optimizer, ema, loader, "cpu")
    with pytest.raises(ValueError, match="teacher"):
        adapter.finish_epoch(10, 10, expected[0], model, optimizer, ema, loader, "cpu")
    with pytest.raises(ValueError, match="verified E10 bank"):
        adapter.step(model, optimizer, ema, x, y, rows, config, dict(ctx, epoch=11, global_step=11))


def test_c1_only_freezes_c2_c3_with_exact_p0_foundation():
    jobs = c1_protocol.c1_jobs()
    assert len(jobs) == len({job.key for job in jobs}) == 20
    assert {job.arm for job in jobs} == {"C2", "C3"}
    for job in jobs:
        assert job.stage == "C1" and dict(job.config) == a_step.a_config("P0", job.dataset)
        assert job.adapter["unique_pair_capacity"] == job.adapter["pair_draws"] == 2048


def test_frozen_scientific_gate_rejects_changed_decisions(tmp_path, monkeypatch):
    c0, reports = tmp_path / "C0", tmp_path / "reports"
    monkeypatch.setattr(c1_protocol, "REPORTS", reports)
    expected = {}
    for directory, names in ((c0, ("STATUS.json", "MANIFEST.json", "SUMMARY.json", "decisions.json")),
                             (reports, ("C0_ANALYSIS.md", "C0_RESEARCH_REVIEW.md"))):
        directory.mkdir()
        for name in names:
            path = directory / name
            path.write_text(name)
            expected[name] = c1_protocol.sha(path)
    assert c1_protocol.verify_gates(tmp_path, expected=expected) == expected
    (c0 / "decisions.json").write_text("changed")
    with pytest.raises(ValueError, match="gate"):
        c1_protocol.verify_gates(tmp_path, expected=expected)


def _historical_control(tmp_path, roster, job, bundle):
    journal = c0_audit.runner.job_journal(tmp_path, roster, job)
    (journal.directory / "JOURNAL.lock").touch()
    binding = journal.binding
    writer = dict(id="committed", gpu="fixture", generation="original")
    owner = dict(binding, attempt=writer["id"], gpu=writer["gpu"], storage_generation=writer["generation"])
    scores = dict(macro_auc=.5, macro_logloss=.4)
    for cell in c0_audit.CELLS:
        scores.update({f"{cell}_auc": .5, f"{cell}_logloss": .4})
    selected = {readout: 1 for readout in READOUTS}
    model_hashes = {readout: digest([job.arm, readout, 1]) for readout in READOUTS}
    model_hashes["fixed_ema"] = SNAPSHOTS[2]["model_hash"]
    receipts = {}
    for readout, checkpoint, epoch, model_hash in (
        ("online", "selected", 1, model_hashes["online"]),
        ("fixed_ema", "selected", 1, model_hashes["fixed_ema"]),
        ("fixed_ema", "E05", 5, SNAPSHOTS[0]["model_hash"]),
        ("fixed_ema", "E10", 10, SNAPSHOTS[1]["model_hash"]),
    ):
        key = f"{readout}-{checkpoint}"
        model = dict(kind="compact_model", readout=readout, epoch=epoch, model_hash=model_hash)
        receipts[key] = dict(identity=dict(owner, **model, global_step=epoch, key=key), sha256=digest(key))
    required = {key: {field: receipt["identity"][field] for field in ("kind", "readout", "epoch", "model_hash")}
                for key, receipt in receipts.items()}
    endpoints = []
    for checkpoint, epoch in (("E10", 10), ("E50", 50), ("selected", 1)):
        for readout in READOUTS:
            model_hash = required.get(f"{readout}-{checkpoint}", {}).get("model_hash", digest([job.arm, readout, epoch]))
            endpoints.append(dict(binding, split="validation", readout=readout, checkpoint=checkpoint,
                                  epoch=epoch, model_hash=model_hash, **scores))
    result = dict(job=job.record(), binding=binding, epoch=50, global_step=50, test_accesses=0,
                  endpoints=endpoints, compact_models=required, selected=selected)
    write(journal.path, dict(binding=binding, generation=writer["generation"], attempts=[writer], acknowledged=50,
                             complete=dict(result=result, required=required), compact=receipts))
    completion = journal.directory / "COMPLETE_VERIFIED.json"
    write(completion, result)
    train_rows = len(bundle.train_y)
    loader = torch.Generator().manual_seed(job.seed + 11000)
    validation = {readout: scores for readout in READOUTS}
    attempt = journal.directory / "attempts" / writer["id"]
    write(attempt / "IDENTITY.json", dict(data=bundle.identity, init_model_hash=digest(job.key)))
    batches, orders = [], []
    for epoch in range(1, 51):
        rows = bundle.train_row_ids[torch.randperm(train_rows, generator=loader).numpy()]
        order_hash = v3.state_digest(rows)
        receipt = dict(identity=dict(owner, kind="epoch", epoch=epoch, global_step=epoch), sha256=digest(epoch))
        write(journal.directory / f"epoch_{epoch:02d}.json",
              dict(epoch=epoch, global_step=epoch, rows=train_rows, order_hash=order_hash, validation=validation,
                   selected_epochs=selected, selected_metrics=validation, selected_model_hashes=model_hashes,
                   ram_receipt=receipt))
        if job.arm == "P0" and epoch <= 10:
            row_hash = ordered_ids_hash(rows)
            ctx = dict(dataset=job.dataset, seed=job.seed, epoch=epoch, global_step=epoch, batch=1,
                       last_batch=True, split_id="original")
            diagnostic = dict(train_identity="fixture", ordered_bce_row_hash=row_hash, condition=[0, 0, 3, 0],
                              w=.25, q=.5, w_over_q=.5, pair_index_hash=digest([epoch, row_hash]))
            append(attempt / "batches.jsonl", dict(context=ctx, diagnostics=diagnostic, row_id_hash=row_hash,
                                                  rows=train_rows, counts=dict(pair_draws=PAIR_COUNT)))
            batches.append([row_hash, [0, 0, 3, 0], .25, .5, diagnostic["pair_index_hash"]])
            orders.append(order_hash)
    ram = {label: dict(receipts[f"fixed_ema-{label}"], read_verify_decode_seconds=1.) for label in c0_audit.SNAPSHOTS}
    selected_rows = {row["readout"]: row for row in endpoints if row["checkpoint"] == "selected"}
    readbacks = {key: value for key, value in receipts.items()
                 if key.endswith("-selected") or (key.endswith("-E10") and job.arm == "P0")}
    link = dict(logical_arm={"B0": "C0", "P0": "C1"}[job.arm], original_job=job.record(), original_binding=binding,
                completion_sha256=sha(completion), journal_sha256=sha(journal.path), final_epoch_receipt=receipt,
                selected=selected_rows, readbacks=readbacks, validation_endpoints=endpoints)
    inputs = {}
    if job.arm == "P0":
        online = next(row for row in endpoints if row["readout"] == "online" and row["checkpoint"] == "E10")
        teacher = dict(original_job_key=job.key, original_ram_sha256=ram["E10"]["sha256"], epoch=10,
                       global_step=10, model_hash=ram["E10"]["identity"]["model_hash"], online_hash=online["model_hash"])
        traces = {writer["id"]: dict(trace_sha256=sha(attempt / "batches.jsonl"),
                                    identity_sha256=sha(attempt / "IDENTITY.json"))}
        prefix = dict(init_model_hash=digest(job.key), steps_per_epoch=1, order_hashes=orders, batches=batches, traces=traces)
        inputs = dict(teacher=teacher, prefix=dict(prefix, hash=digest(prefix)))
    return link, dict(ram=ram, selected_epoch=1, binding=binding), inputs


@pytest.fixture
def roster_fixture(tmp_path, monkeypatch):
    bundle = bundle_fixture(2)
    bundle.identity["data_hash"] = "data"
    sampler = sampler_for(bundle)
    candidates = c0_audit.candidate_set(sampler, SEED)
    raw = {key: np.ones(len(c["ranks"])) for key, c in candidates.items()}
    c0, reports = tmp_path / "C0", tmp_path / "reports"
    inputs = save_inputs(c0, sampler, candidates, raw)
    dataset = sampler.dataset
    key = f"{dataset}_{SEED}"
    original_source = dict(source_hash="source", source_manifest="manifest", ga31={},
                           runtime=dict(a_step="a", macro_pair_sampler="s"))
    datasets = {dataset: dict(data_hash="data", scope_hash="scope")}
    controls = [RunSpec("A1", dataset, SEED, arm, a_step.a_config(arm, dataset), {}) for arm in ("B0", "P0")]
    a_roster = dict(source=original_source, datasets=datasets,
                    jobs=[dict(job.record(), config_hash=job.config_hash,
                               algorithm_hash=algorithm_digest(job, original_source)) for job in controls])
    a_roster_path = tmp_path / "A1" / "ROSTER.json"
    write(a_roster_path, a_roster)
    links, audits, endpoints = {}, {}, []
    for job in controls:
        link, audit, historical = _historical_control(tmp_path, a_roster, job, bundle)
        endpoints.extend(link["validation_endpoints"])
        links[job.key] = dict(link, audit=dict(candidate_sha256=inputs["candidate"]["sha256"],
                                               margins_sha256=inputs["margins"]["sha256"]))
        audits[job.key] = dict(audit, margins=inputs["margins"])
        inputs.update(historical)
    write(tmp_path / "A1" / "ENDPOINTS.json", endpoints)
    cache = dict(split_rows=dict(train=len(bundle.train_y)), split_row_fingerprints=dict(train="fixture"))
    write(c0 / "MANIFEST.json", dict(candidates={key: inputs["candidate"]}, jobs=audits, caches={dataset: cache},
                                    roster=dict(a_roster, sha256=sha(a_roster_path))))
    gates = {"MANIFEST.json": c1_protocol.sha(c0 / "MANIFEST.json")}
    for directory, names in ((c0, ("STATUS.json", "SUMMARY.json", "decisions.json")),
                             (reports, ("C0_ANALYSIS.md", "C0_RESEARCH_REVIEW.md"))):
        directory.mkdir(exist_ok=True)
        for name in names:
            path = directory / name
            path.write_text(name)
            gates[name] = c1_protocol.sha(path)
    write(tmp_path / "B1" / "ROSTER.json", dict(datasets=datasets))
    source = {}
    monkeypatch.setattr(c1_protocol, "REPORTS", reports)
    monkeypatch.setattr(c1_protocol, "DATASETS", (dataset,))
    monkeypatch.setattr(c1_protocol, "SEEDS", (SEED,))
    monkeypatch.setattr(c1_protocol, "_guard", lambda _artifacts: None)
    monkeypatch.setattr(c1_protocol, "_source_snapshot", lambda *_args: {})
    monkeypatch.setattr(c1_protocol, "_current_source", lambda *_args: source)
    monkeypatch.setattr(c1_protocol, "_registered_datasets", lambda _artifacts: datasets)
    monkeypatch.setattr(c1_protocol, "algorithm_digest", lambda _job, _source: "algorithm")
    verify_gates = c1_protocol.verify_gates
    monkeypatch.setattr(c1_protocol, "verify_gates",
                        lambda artifacts, *, expected=gates: verify_gates(artifacts, expected=expected))
    jobs = c1_protocol.c1_jobs()
    roster = dict(study=c1_protocol.STUDY, stage="C1", source=source, datasets=datasets,
                  jobs=[c1_protocol._record(job, source) for job in jobs], gates=gates, test_status="SEALED",
                  C4="NOT_TRIGGERED", controls=links, seed_inputs={key: inputs})
    return sampler, roster, inputs


def test_rehashed_existing_roster_rejects_substituted_e10_targets(tmp_path, roster_fixture):
    sampler, roster, inputs = roster_fixture
    path = tmp_path / "C1" / "ROSTER.json"
    write(path, dict(roster, roster_hash=digest(roster)))
    assert c1_protocol.freeze_roster(tmp_path)[0]["roster_hash"] == digest(roster)

    def verify_inputs(record):
        c1_protocol.verify_seed_inputs(tmp_path, expected=roster["gates"], dataset=sampler.dataset,
                                       seed=SEED, inputs=record)

    adapter = c_step.DirectionCAdapter(SimpleNamespace(sampler=sampler), SEED, "C3", inputs, verify_inputs)
    adapter._prepare()
    with np.load(inputs["margins"]["path"], allow_pickle=False) as archive:
        fields = dict(archive)
    fields["E10"] = np.zeros_like(fields["E10"])
    inputs["margins"] = c0_audit.save_arrays(tmp_path / "replacement.npz", **fields)
    banks, _ = memory_bank.prepare_banks(sampler, SEED, inputs)
    assert all(not len(entry["indices"]) for entry in banks["correct"]["conditions"].values())
    write(path, dict(roster, roster_hash=digest(roster)))
    with pytest.raises(ValueError, match="margin receipt differs from the frozen C0 manifest"):
        c1_protocol.freeze_roster(tmp_path)
    with pytest.raises(ValueError, match="margin receipt differs from the frozen C0 manifest"):
        adapter._prepare()


@pytest.mark.parametrize("identity", ["candidate", "snapshots", "teacher"])
def test_seed_revalidation_rejects_changed_candidate_or_e10_identity(tmp_path, roster_fixture, identity):
    sampler, roster, inputs = roster_fixture
    if identity == "candidate":
        inputs["candidate"]["sha256"] = "other candidate receipt"
    elif identity == "snapshots":
        inputs["snapshots"][1]["model_hash"] = "other E10 snapshot"
    else:
        inputs["teacher"]["global_step"] += 1
    with pytest.raises(ValueError, match="differs from the frozen C0 manifest"):
        c1_protocol.verify_seed_inputs(tmp_path, expected=roster["gates"], dataset=sampler.dataset,
                                       seed=SEED, inputs=inputs)


@pytest.mark.parametrize("mutation", ["null_controls", "selected", "E50", "online_hash", "missing_prefix", "prefix_batch"])
def test_rehashed_roster_requires_exact_historical_controls_and_warmup(tmp_path, roster_fixture, monkeypatch, mutation):
    sampler, roster, inputs = roster_fixture
    path = tmp_path / "C1" / "ROSTER.json"
    write(path, dict(roster, roster_hash=digest(roster)))

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Existing roster verification must use metadata only")

    monkeypatch.setattr(c1_protocol, "load_train_validation", unexpected_load)
    monkeypatch.setattr(ram_state.Journal, "read_model", unexpected_load)
    assert c1_protocol.freeze_roster(tmp_path)[0]["roster_hash"] == digest(roster)

    if mutation == "null_controls":
        roster["controls"] = {job.key: None for job in c1_protocol.c1_jobs()}
    elif mutation in ("selected", "E50"):
        b0 = roster["controls"][f"A1-{sampler.dataset}-B0-seed{SEED}"]
        row = next(r for r in b0["validation_endpoints"] if r["checkpoint"] == mutation and r["readout"] == "online")
        row["macro_logloss"] = .9
    elif mutation == "online_hash":
        inputs["teacher"]["online_hash"] = "f" * 64
    elif mutation == "missing_prefix":
        del inputs["prefix"]
    else:
        prefix = inputs["prefix"]
        prefix["batches"][0][-1] = "f" * 64
        prefix["hash"] = digest({key: value for key, value in prefix.items() if key != "hash"})
    write(path, dict(roster, roster_hash=digest(roster)))

    expectation = "control links differ" if mutation in ("null_controls", "selected", "E50") else "teacher or warmup prefix"
    with pytest.raises(ValueError, match=expectation):
        c1_protocol.freeze_roster(tmp_path)
    if mutation in ("online_hash", "missing_prefix", "prefix_batch"):
        with pytest.raises(ValueError, match=expectation):
            c1_protocol.verify_seed_inputs(tmp_path, expected=roster["gates"], dataset=sampler.dataset,
                                           seed=SEED, inputs=inputs)


@pytest.mark.parametrize(("arm", "checkpoint", "readout", "field"), [
    ("B0", "E10", "online", "macro_logloss"),
    ("B0", "E50", "online", "macro_logloss"),
    ("B0", "E50", "fixed_ema", "macro_auc"),
    ("P0", "E50", "fixed_ema", "macro_logloss"),
    ("P0", "E10", "online", "model_hash"),
])
def test_fresh_control_checks_committed_validation_and_published_model_identity(
    tmp_path, roster_fixture, monkeypatch, arm, checkpoint, readout, field,
):
    sampler, roster, _ = roster_fixture
    bundle = bundle_fixture(2)
    bundle.identity["data_hash"] = "data"
    a_roster = read(tmp_path / "A1" / "ROSTER.json")
    job = RunSpec("A1", sampler.dataset, SEED, arm, a_step.a_config(arm, sampler.dataset), {})
    manifest = read(tmp_path / "C0" / "MANIFEST.json")
    audit, cache = manifest["jobs"][job.key], manifest["caches"][sampler.dataset]
    directory = tmp_path / "A1" / "runs" / job.key
    completion_path, journal_path = directory / "COMPLETE_VERIFIED.json", directory / "RAM_JOURNAL.json"
    result, journal = read(completion_path), read(journal_path)
    published = copy.deepcopy(result["endpoints"])

    def read_model(_journal, key, *, expected):
        assert all(journal["compact"][key]["identity"][k] == v for k, v in expected.items())
        return {"weight": torch.ones(1)}, journal["compact"][key]

    monkeypatch.setattr(ram_state.Journal, "read_model", read_model)
    link, _ = c1_protocol._control(tmp_path, a_roster, job, audit, cache, published, bundle=bundle)
    assert dict(link, audit=roster["controls"][job.key]["audit"]) == roster["controls"][job.key]

    row = next(r for r in result["endpoints"] if r["checkpoint"] == checkpoint and r["readout"] == readout)
    row[field] = "f" * 64 if field == "model_hash" else .125
    if field != "model_hash":
        # Agreement between the two publications still requires the original epoch metrics.
        published = copy.deepcopy(result["endpoints"])
    journal["complete"]["result"] = result
    write(completion_path, result)
    write(journal_path, journal)
    with pytest.raises(ValueError, match=f"{readout}/{checkpoint} differs from committed Validation"):
        c1_protocol._control(tmp_path, a_roster, job, audit, cache, published, bundle=bundle)
