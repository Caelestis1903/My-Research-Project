"""Direct checks of the C1 shared transaction and complete-epoch worker."""
import copy
import json
import math
import subprocess
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.three_directions_20261128 import a_step, c0_audit, c1_protocol, c_step, memory_bank, ram_state, runner
from experiments.three_directions_20261128.protocol import ROOT, STUDY, RunSpec, digest, load_train_validation
from experiments.three_directions_20261128.tests.test_c1_memory import (
    SEED, _parts, _reduced_epoch, bundle_fixture, save_inputs, single_thread,
)
from experiments.three_directions_20261128.tests.test_direction_a import TinyM3oE

ARTIFACTS = ROOT / "artifacts" / STUDY
SOURCE_ARCHIVE = "70455ed0ebc898ae80275d17f14e7be80f8cfd57"


@pytest.mark.parametrize("dataset", ["movielens", "kuairand"])
def test_cuda_seed_2022_first_and_natural_tail_against_archived_p0(dataset, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Real first and tail parity requires CUDA")
    device = torch.device("cuda:0")
    bundle = load_train_validation(dataset, ARTIFACTS / "DATA_AUDIT.json")
    stock_job = RunSpec("A1", dataset, SEED, "P0", a_step.a_config("P0", dataset), {})
    c_job = next(job for job in c1_protocol.c1_jobs() if (job.dataset, job.seed, job.arm) == (dataset, SEED, "C3"))
    archived_roster = runner.read(ARTIFACTS / "A1" / "ROSTER.json")
    prepared = c0_audit.prepare_job(ARTIFACTS, archived_roster, stock_job, bundle)
    journal = prepared["journal"]
    committed = [runner.read(journal.directory / f"epoch_{e:02d}.json") for e in range(1, 11)]
    width = stock_job.config["batch_size"]
    steps = math.ceil(len(bundle.train_y) / width)
    expected_prefix = c1_protocol._prefix(stock_job, journal, committed, steps, len(bundle.train_y),
                                          bundle.identity["split_row_fingerprints"]["train"], bundle=bundle)
    inputs = dict(prefix=expected_prefix)
    roster = dict(seed_inputs={f"{dataset}_{SEED}": inputs})
    archived = ModuleType("archived_a_step")
    archived.__package__ = a_step.__package__
    path = f"experiments/{STUDY}/a_step.py"
    source = subprocess.check_output(["git", "show", f"{SOURCE_ARCHIVE}:{path}"], cwd=ROOT)
    exec(compile(source, path, "exec"), archived.__dict__)
    trace = []

    def observe(original):
        def forward(model, batch, **kwargs):
            result = original(model, batch, **kwargs)
            trace.append((batch.shape[1], c_step.v3.state_digest(batch), result[1].requires_grad))
            return result
        return forward

    for module in (archived, a_step):
        monkeypatch.setattr(module, "forward_logits", observe(module.forward_logits))
    source_model = runner.v3.load_protocol()["datasets"][dataset]["source"]
    for step in (1, steps):
        states, outcomes, traces = [], [], []
        predictions = []
        for job in (stock_job, c_job):
            runner.v3.setup(SEED, device)
            loader = torch.Generator().manual_seed(SEED + 11000)
            order = torch.randperm(len(bundle.train_y), generator=loader).numpy()
            x, y, rows = runner.batch_inputs(bundle, order[(step - 1) * width:step * width], device)
            assert len(rows) == (width if step == 1 else {"movielens": 21927, "kuairand": 40052}[dataset])
            model = runner.v3.make_model(bundle, source_model).to(device)
            ema = copy.deepcopy(model).requires_grad_(False)
            if job.stage == "A1":
                optimizer, adapter = archived.make_training_parts(model, job, bundle)
            else:
                optimizer, adapter = runner.make_training_parts(model, job, bundle, artifacts=ARTIFACTS, roster=roster)
                adapter.verify_init(runner.v3.state_digest(model.state_dict()))
            context = dict(dataset=dataset, seed=SEED, epoch=1, global_step=step, batch=step,
                           last_batch=step == steps, split_id="original")
            before = runner.v3.rng_state(device)
            result = adapter.step(model, optimizer, ema, x, y, rows, job.config, context)
            if job.stage == "C1":
                assert sum(result["diagnostics"][f"pair_margin_{s}"] for s in ("positive", "tied", "negative")) == 2048
            assert runner.v3.equal(runner.v3.rng_state(device), before)
            states.append(runner.v3.state_digest([model.state_dict(), ema.state_dict(), optimizer.state_dict(),
                                                 [p.grad for p in model.parameters()], loader.get_state(), before]))
            outcomes.append((result["loss"], result["real_loss"], result["counts"],
                             result["diagnostics"]["grad_norms"], result["diagnostics"]["pair_index_hash"]))
            traces.append(trace.copy())
            trace.clear()
            predictions.append(result["pred"])
        assert states[0] == states[1] and outcomes[0] == outcomes[1] and traces[0] == traces[1]
        assert [n for n, _, _ in traces[0]] == [len(rows), 4096]
        assert [tracks for _, _, tracks in traces[0]] == [False, True]
        assert outcomes[1][2]["pair_draws"] == 2048 and outcomes[1][2]["pair_diagnostic_vjp_calls"] == 1
        assert torch.equal(*predictions)


@pytest.mark.parametrize("event", ["bank", "current_fallback", "none"])
def test_actual_memory_transaction_and_no_event_bce(tmp_path, monkeypatch, event):
    bundle = bundle_fixture(2)
    model, optimizer, ema, _, stock, config = _parts(bundle)
    ref_model = copy.deepcopy(model)
    ref_ema = copy.deepcopy(ema)
    ref_optimizer = a_step.make_optimizer(ref_model, config)
    context = dict(dataset=bundle.dataset, seed=SEED, epoch=11, global_step=11)
    current = stock.sampler.sample_macro6(context)
    candidates = c0_audit.candidate_set(stock.sampler, SEED)
    margins = {condition: np.array([-.5, 0., .75, 9.]) for condition in candidates}
    if event == "current_fallback":
        margins[current["condition"]].fill(0.)
    inputs = save_inputs(tmp_path, stock.sampler, candidates, margins)
    adapter = c_step.DirectionCAdapter(stock, SEED, "C3", inputs, lambda _inputs: None)
    adapter.bank, _ = adapter._prepare()
    applied = memory_bank.draw_pair(adapter.bank, current, context) if event != "none" else None
    x, y, rows = runner.batch_inputs(bundle, np.arange(len(bundle.train_y)), "cpu")
    if event == "current_fallback":
        a_step.DirectionAAdapter(bundle.train_x, stock.sampler).step(
            ref_model, ref_optimizer, ref_ema, x, y, rows, config, context)
    elif event == "none":
        loss, _ = a_step.ga31.losses(ref_model, x, y, config["target_domain"], a_step.source_config(False),
                                   bundle.dataset, SEED, context["global_step"])
        loss.backward()
        ref_optimizer.step()
        a_step.update_ema(ref_ema, ref_model, config["rho"])

    draws, forwards, calls = [], [], {"backward": 0, "grad": 0}
    original = a_step.forward_logits

    def sample(context):
        draws.append(context)
        return current if event != "none" else None

    def forward(_model, x, **kw):
        result = original(_model, x, **kw)
        forwards.append((x.clone(), result[1].requires_grad))
        return result

    def counted(name, original):
        def call(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return call

    monkeypatch.setattr(stock.sampler, "sample_macro6", sample)
    monkeypatch.setattr(a_step, "forward_logits", forward)
    for name in calls:
        monkeypatch.setattr(torch.autograd, name, counted(name, getattr(torch.autograd, name)))
    actual = adapter.step(model, optimizer, ema, x, y, rows, config, context)
    diagnostic, counts = actual["diagnostics"], actual["counts"]
    pair_forward = int(applied is not None)
    assert draws == [context] and calls == dict(backward=1, grad=pair_forward)
    assert len(forwards) == 1 + pair_forward and torch.equal(forwards[0][0], x) and not forwards[0][1]
    assert counts["optimizer_steps"] == counts["ema_updates"] == 1
    assert (counts["pair_rows"], counts["pair_draws"], counts["pair_forwards"], counts["pair_diagnostic_vjp_calls"]) == (
        4096 * pair_forward, 2048 * pair_forward, pair_forward, pair_forward)
    assert diagnostic["memory_source"] == event and diagnostic["bank_hash"] == adapter.bank["bank_hash"]
    if applied is not None:
        assert forwards[1][1] and torch.equal(forwards[1][0], torch.as_tensor(bundle.train_x[applied["offsets"]]).T)
        assert diagnostic["current_pair_index_hash"] == current["pair_index_hash"]
        assert diagnostic["pair_index_hash"] == applied["pair_index_hash"]
        assert diagnostic["w_over_q"] == current["w"] / current["q"]
        assert diagnostic["unique_pairs"] == applied["unique_pairs"]
        assert diagnostic["unique_endpoints"] == applied["unique_endpoints"]
    if event in ("none", "current_fallback"):
        assert all(runner.v3.equal(a.state_dict(), b.state_dict())
                   for a, b in ((model, ref_model), (optimizer, ref_optimizer), (ema, ref_ema)))
    if event == "none":
        for field in ("condition", "current_pair_index_hash", "pair_index_hash", "w", "q", "w_over_q"):
            assert diagnostic[field] is None
        assert diagnostic["unique_pairs"] == diagnostic["unique_endpoints"] == 0
        assert diagnostic["memory_miss"] == "no_event" and diagnostic["pair_loss"] == 0.
    elif event == "current_fallback":
        assert diagnostic["memory_miss"] == "empty_bank"
    else:
        assert diagnostic["draw_positions_hash"] == applied["draw_positions_hash"]
        assert diagnostic["memory_miss"] is None


def test_worker_ram_recovery_diagnostics_and_committed_exposure(tmp_path, monkeypatch):
    bundle = bundle_fixture(2)
    bundle.data_hash = "fixture-data"
    bundle.features = SimpleNamespace(encoder_hash="encoder", scaler_hash="scaler")
    bundle.val_x, bundle.val_y = bundle.train_x, bundle.train_y
    v3 = runner.v3
    v3.setup(SEED, "cpu")
    baseline = _parts(bundle)
    initial = v3.state_digest(baseline[0].state_dict())
    warmup = [_reduced_epoch(bundle, baseline, epoch) for epoch in range(1, 11)]
    e10 = warmup[-1][0]
    prefix = dict(steps_per_epoch=1, init_model_hash=initial, batches=[row for _, row in warmup],
                  order_hashes=[state["epoch_metrics"]["order_hash"] for state, _ in warmup])
    prefix["hash"] = digest(prefix)
    teacher = dict(epoch=10, global_step=10, model_hash=v3.state_digest(e10["ema"]),
                   online_hash=v3.state_digest(e10["online"]))
    sampler = baseline[4].sampler
    candidates = c0_audit.candidate_set(sampler, SEED)
    margins = {condition: np.array([-.5, 0., .75, 9.]) for condition in candidates}
    inputs = save_inputs(tmp_path, sampler, candidates, margins, teacher=teacher, prefix=prefix)
    source = dict(source_hash="C1-source", source_manifest="manifest", ga31="ga31", a1_warmup={},
                  runtime={name: name for name in c1_protocol.NUMERICAL})
    roster = dict(source=source, datasets={bundle.dataset: dict(data_hash=bundle.data_hash,
                  scope_hash=runner.scope_digest(runner._scope_manifest(baseline[0])))},
                  seed_inputs={f"{bundle.dataset}_{SEED}": inputs}, gates={"fixture": "pass"})
    job = RunSpec("C1", bundle.dataset, SEED, "C3", dict(baseline[-1], epochs=11), {})
    calls = []

    def verify(artifacts, *, expected, dataset, seed, inputs):
        assert expected is roster["gates"] and dataset == bundle.dataset and seed == SEED
        assert inputs is roster["seed_inputs"][f"{bundle.dataset}_{SEED}"]
        calls.append(artifacts)

    monkeypatch.setattr(c1_protocol, "verify_seed_inputs", verify)
    monkeypatch.setattr(runner, "load_train_validation", lambda dataset, _path: bundle)
    monkeypatch.setattr(v3, "make_model", lambda *_: TinyM3oE())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 100)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (900, 1000))
    monkeypatch.setattr(runner.scheduler, "gpu_sample", lambda: {})
    monkeypatch.setattr(runner.scheduler, "residency", lambda *_: 400)
    latest = {}
    commit = ram_state.Journal.commit_epoch

    def recorded(journal, attempt, state):
        latest[journal.directory] = [v3.state_digest(state[key]) for key in ("online", "ema", "optimizer", "loader", "rng")]
        return commit(journal, attempt, state)

    monkeypatch.setattr(ram_state.Journal, "commit_epoch", recorded)
    primary, retry = (tmp_path / name for name in ("primary", "retry"))
    journals = {path: runner.job_journal(path, roster, job) for path in (primary, retry)}

    def run(path, name, *, diagnostic=None, stopped=lambda: False):
        journal = journals[path]
        invocation = path / name
        with journal.writer():
            result = runner._worker(path, journal, roster, job, "fixture", "cpu", invocation, diagnostic, stopped)
        assert result["status"] != "TECHNICAL_FAILURE", result.get("traceback")
        metadata = runner.read(journal.path)
        attempt = journal.directory / "attempts" / metadata["attempts"][-1]["id"]
        sites = {runner.read(p)["site"] for p in (invocation / "allocations").glob("*.json")}
        rows = [json.loads(line) for line in (attempt / "batches.jsonl").read_text().splitlines()]
        return result["status"], metadata, attempt, sites, rows

    try:
        status, _, _, sites, original_trace = run(primary, "continuous")
        assert status == "COMPLETE" and {"e10_memory_install", "first_memory_batch"} <= sites
        retry_journal = journals[retry]
        receipt_file = retry_journal.directory / "epoch_10.json"
        status, pending, _, sites, _ = run(retry, "stop-after-e10", stopped=receipt_file.exists)
        assert status == "BOUNDARY_STOP" and "e10_memory_install" in sites and "first_memory_batch" not in sites
        receipt = pending["epochs"][0]
        status, diagnostic, attempt, sites, diagnostic_trace = run(retry, "measure-from-e10", diagnostic=11)
        assert status == "DIAGNOSTIC_COMPLETE" and diagnostic["epochs"] == pending["epochs"]
        assert not (attempt / "E10_MEMORY.json").exists() and "first_memory_batch" in sites
        assert diagnostic_trace[-1]["diagnostics"]["draw_positions_hash"] == original_trace[-1]["diagnostics"]["draw_positions_hash"]

        with ram_state._opened(receipt["name"]) as packet:
            packet.buf[0] ^= 1
        status, diagnostic, attempt, sites, diagnostic_trace = run(retry, "measure-from-e9", diagnostic=11)
        assert status == "DIAGNOSTIC_COMPLETE" and diagnostic["epochs"] == pending["epochs"][1:]
        assert {"e10_memory_install", "first_memory_batch"} <= sites and (attempt / "E10_MEMORY.json").is_file()
        assert [row["context"]["epoch"] for row in diagnostic_trace] == [10, 11]
        status, _, attempt, _, replay_trace = run(retry, "resume-normal")
        assert status == "COMPLETE"
        assert runner.read(receipt_file)["ram_receipt"]["identity"]["attempt"] == attempt.name
        assert replay_trace[-1]["diagnostics"]["draw_positions_hash"] == original_trace[-1]["diagnostics"]["draw_positions_hash"]
        a, b = (journal.completed() for journal in journals.values())
        for key in ("prefix", "bank", "exposure"):
            assert a["memory"][key] == b["memory"][key]
        assert a["endpoints"] == b["endpoints"] and latest[journals[primary].directory] == latest[retry_journal.directory]
        assert sum(row["events"] for row in a["memory"]["exposure"]) == 11
        assert calls == [primary, retry, retry, retry, retry]

        original = dict(a["endpoints"][-1], stage="A1", arm="P0", job_key="original-p0", source_hash="archived")
        roster["controls"] = {"original-p0": dict(logical_arm="C1", validation_endpoints=[original], readbacks={})}
        runner.publish_tables(primary, "C1", roster, [job], dict(status="COMPLETE", test_accesses=0))
        status = runner.read(primary / "C1" / "STAGE_STATUS.json")
        assert status["completed"] == status["jobs"] == 1
        linked, = [row for row in runner.read(primary / "C1" / "SELECTED.json") if row["linked_control"]]
        assert linked == dict(original, logical_arm="C1", linked_control=True)
        assert runner.read(primary / "C1" / "EXPOSURE.json")[job.key]["rows"] == a["memory"]["exposure"]
    finally:
        for journal in journals.values():
            if journal.path.exists():
                meta = runner.read(journal.path)
                for receipt in [*meta["epochs"], *meta["compact"].values()]:
                    ram_state._unlink(receipt["name"])
