import copy
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mdmt_shared_private_coordination_20261123 import blocks, cli, diagnostics, s0, scheduler, source, training


def test_registered_matrix_and_current_card_admission():
    authority = dict(source_hash="source", datasets=dict.fromkeys(scheduler.DATASETS, "data"),
                     panels=dict.fromkeys(scheduler.DATASETS, "panels"))
    jobs = scheduler.jobs(authority)
    assert len(jobs) == len({job["logical_id"] for job in jobs}) == 10
    assert {(job["dataset"], job["seed"], job["arm"]) for job in jobs} == {
        (dataset, seed, "J_ORIG") for dataset in scheduler.DATASETS for seed in range(2022, 2027)}
    assert all(job["config"]["batch_size"] == 40960 and job["config"]["epochs"] == 50 for job in jobs)
    card = dict(used=84, total=100, watts=84, limit=100)
    assert scheduler.admit(card, .89)
    assert scheduler.admit(card, .89, (98, 14))
    assert not scheduler.admit(card, .89, (98, 15))
    assert not scheduler.admit(dict(card, used=85), .89)
    assert not scheduler.admit(dict(card, watts=85), .89)
    assert not scheduler.admit(card, .90)


def test_loader_preview_and_coverage_for_fifty_epochs():
    for size, count, tail in ((800167, 20, 21927), (2006132, 49, 40052)):
        plan = training.batch_plan(size, 40960)
        assert len(plan) == count
        assert plan[0][0] == 0 and plan[-1][1] == size
        assert all(n == k for (_, n), (k, _) in zip(plan, plan[1:]))
        assert plan[-1][1] - plan[-1][0] == tail
    loader = torch.Generator().manual_seed(13022)
    original = torch.Generator().manual_seed(13022)
    plan = training.batch_plan(53, 10)
    for epoch in range(1, 51):
        if epoch in (11, 26, 41):
            before = loader.get_state().clone()
            preview = training.preview_order(53, loader)
            np.testing.assert_array_equal(preview, training.preview_order(53, loader))
            assert torch.equal(loader.get_state(), before)
        order = training.next_order(53, loader)
        actual = np.concatenate([order[a:b] for a, b in plan])
        np.testing.assert_array_equal(actual, torch.randperm(53, generator=original).numpy())
        np.testing.assert_array_equal(np.sort(actual), np.arange(53))
        if epoch in (11, 26, 41):
            np.testing.assert_array_equal(order, preview)


def test_strict_earliest_independent_validation_selection():
    model = torch.nn.Linear(1, 1)
    selected = {}
    first_hash = training.v3.state_digest(model.state_dict())
    training.select(selected, "online", 1, dict(macro_auc=.6), model)
    with torch.no_grad():
        model.weight.add_(1)
    training.select(selected, "online", 2, dict(macro_auc=.6), model)
    training.select(selected, "fixed_ema", 2, dict(macro_auc=.6), model)
    assert selected["online"]["epoch"] == 1 and selected["fixed_ema"]["epoch"] == 2
    assert training.v3.state_digest(selected["online"]["model"]) == first_hash
    training.select(selected, "online", 3, dict(macro_auc=.61), model)
    assert selected["online"]["epoch"] == 3
    with pytest.raises(RuntimeError):
        training.select(selected, "online", 4, dict(macro_auc=None), model)


def test_ram_verified_intent_recovery_and_same_next_native_update(monkeypatch, tmp_path):
    device = torch.device("cpu")
    training.v3.setup(2022, device)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid())
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-4)
    cfg = dict(rho=.99, weight_decay=1e-4)
    x, y = torch.rand(12, 2), torch.rand(12, 2)
    loader = torch.Generator().manual_seed(13022)
    order = training.next_order(len(x), loader)
    blocks.step(model, opt, ema, ((x[order], y[order]),), cfg, detailed=False)
    job = dict(logical_id="same", config_hash="canonical", data_hash="natural")
    order = training.next_order(len(x), loader)
    original = training.capture(model, opt, ema, loader, order, device, job=job, owner="owner", epoch=1,
                                counts=dict(optimizer_steps=1), next_prediction="expected")
    records = {}

    def request(owner, cmd, **payload):
        if cmd == "status":
            return dict(jobs={k: dict(epoch=value["epoch"]) for k, value in records.items()})
        key = payload["key"]
        if cmd == "get":
            return {"state": copy.deepcopy(records[key])}
        assert cmd == "commit"
        records[key] = copy.deepcopy(payload["state"])
        return dict(status="RAM_COMMITTED", epoch=records[key]["epoch"])

    monkeypatch.setattr(training, "request", request)
    receipt_path = tmp_path / "attempts/physical/RAM_E01.json"
    receipt = training.commit(tmp_path, "ram-key", 1, receipt_path, original, 2.0)
    assert receipt["status"] == "RAM_VERIFIED"
    training.write(receipt_path, dict(receipt, status="RAM_INTENT"))
    expected_order = training.next_order(len(x), loader)
    blocks.step(model, opt, ema, ((x[expected_order], y[expected_order]),), cfg, detailed=False)
    expected = training.capture(model, opt, ema, loader, expected_order, device)
    packet, recovered = training.recover(tmp_path, "owner", tmp_path, job)
    assert recovered == receipt
    assert training.read(receipt_path) == receipt
    assert training.recover(tmp_path, "lost_owner", tmp_path, job) == (None, None)
    training.restore(packet, model, opt, ema, loader, device)
    derived = training.next_order(len(x), loader)
    np.testing.assert_array_equal(derived, expected_order)
    blocks.step(model, opt, ema, ((x[derived], y[derived]),), cfg, detailed=False)
    assert training.v3.equal(training.capture(model, opt, ema, loader, derived, device), expected)
    records["ram-key"]["packet"]["cursor"] += 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        training.recover(tmp_path, "owner", tmp_path, job)


def test_resumed_anchor_residency_does_not_skip_startup_ack(monkeypatch, tmp_path):
    job = dict(logical_id="resume", dataset="movielens", seed=2022, data_hash="data", panel_hash="panel",
               config=dict(epochs=50))
    out = tmp_path / "runs" / job["logical_id"]
    path = out / "attempts/current/ALLOCATION.json"
    packet = dict(epoch=10, selected={}, epochs=[], counts={}, next_prediction="next")
    owner = tmp_path / "ram"
    bundle = SimpleNamespace(train_y=np.zeros(12))
    memory_calls = []
    requests = []
    pid = 42
    card = dict(used=72, total=100, watts=80, limit=100)
    telemetry = dict(time=2, devices=[card], host_ram_used=.6, processes=[dict(pid=pid, bytes=60)])

    def diagnose(*args):
        training.write(out / "forks_E10.json", dict(resources=dict(process_gpu_bytes=90)))
        return {"key": "anchor"}

    def epoch_step(_model, _opt, _ema, _loader, _bundle, _job, epoch, _device, _counts, allocate, _physical):
        assert epoch == 11
        allocate()
        allocate()
        raise RuntimeError("fixture reached the first resumed batch")

    def process_memory():
        memory_calls.append(True)
        return 60, telemetry

    monkeypatch.setattr(training, "owner_identity", lambda _: "owner")
    monkeypatch.setattr(training, "register", lambda *args: None)
    monkeypatch.setattr(training, "request", lambda _owner, cmd, **payload: requests.append((cmd, payload)))
    monkeypatch.setattr(training, "recover", lambda *args: (packet, {"key": "previous"}))
    monkeypatch.setattr(training, "restore", lambda *args: None)
    monkeypatch.setattr(training, "diagnose", diagnose)
    monkeypatch.setattr(training, "epoch_step", epoch_step)
    monkeypatch.setattr(training, "process_memory", process_memory)
    monkeypatch.setattr(training.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(source, "load_data", lambda _: (bundle, dict(data_hash="data")))
    monkeypatch.setattr(source, "make_model", lambda *_: torch.nn.Linear(1, 1))
    monkeypatch.setattr(source, "optimizer", lambda model, _: torch.optim.Adam(model.parameters()))
    monkeypatch.setattr(diagnostics, "prepare", lambda *_: SimpleNamespace(manifest=dict(panel_hash="panel")))

    with pytest.raises(RuntimeError, match="fixture reached the first resumed batch"):
        training.worker(tmp_path, owner, job, "current", "cpu", lambda: False)

    assert len(memory_calls) == 1
    assert requests[0] == ("allocated", dict(key="D2:resume:current:baseline", peak_memory_bytes=60))
    assert training.read(path)["process_gpu_bytes"] == 60
    assert training.read(out / "attempts/current.json")["process_gpu_bytes"] == 90
    assert scheduler.startup_pending(path, pid, dict(telemetry, time=1))
    fresh = dict(telemetry, time=3)
    assert not scheduler.startup_pending(path, pid, fresh)
    assert scheduler.admit(card, fresh["host_ram_used"])


def test_publish_counts_orphaned_diagnostic_journal_after_restart(tmp_path):
    job = dict(dataset="movielens", seed=2022, arm="J_ORIG", logical_id="restart",
               source_hash="source", data_hash="data", config_hash="config")
    out = tmp_path / "runs" / job["logical_id"]
    attempts = out / "attempts"
    update = dict(attempt_id="prior", fork_key="interrupted-E25", epoch=25, arm="J2", phase="update",
                  readout=None, optimizer_steps=1, seconds=.125)
    journal_costs = {"prior": [update], "replacement": []}
    replacement = dict(attempt_id="replacement", status="COMPLETE", last_verified_epoch=50, resumed_epoch=25,
                       wall_seconds=12, max_rss_bytes=20, peak_allocated_bytes=30, peak_reserved_bytes=40,
                       process_gpu_bytes=50, counts={"optimizer_steps": 2})
    training.write(attempts / "replacement.json", replacement)
    result = dict(job=job, anchors={}, fork_hashes={}, selected_epochs=dict.fromkeys(training.KINDS, 50),
                  selected_hashes=dict.fromkeys(training.KINDS, "selected"))
    for epoch in training.ANCHORS:
        attempt = "prior" if epoch == 10 else "replacement"
        key, path = f"accepted-E{epoch}", out / f"forks_E{epoch:02d}.json"
        row = dict(attempt_id=attempt, epoch=epoch, arm="J2", phase="end", readout="online", eval_forwards=1)
        journal_costs[attempt].append(dict(row, fork_key=key))
        training.write(path, dict(metadata=dict(fork_key=key), costs=[row], resources=dict(process_gpu_bytes=1)))
        result["anchors"][str(epoch)] = dict(bytes=1)
        result["fork_hashes"][path.name] = training.sha(path)
    for attempt, costs in journal_costs.items():
        training.write(attempts / attempt / "FORK_COSTS.json",
                       dict(costs=costs, counts=dict(optimizer_steps=sum(row.get("optimizer_steps", 0) for row in costs))))
    for epoch in range(1, 51):
        training.write(out / f"epoch_{epoch:02d}.json",
                       dict(attempt_id="prior" if epoch <= 25 else "replacement",
                            validation=dict.fromkeys(training.KINDS, {})))

    scheduler.publish(tmp_path, [result])

    with (tmp_path / "COSTS.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    assert [row["attempt_id"] for row in costs if row["cost_kind"] == "baseline_attempt"] == ["replacement"]
    assert [int(row["epoch"]) for row in costs if row["cost_kind"] == "fork"] == list(training.ANCHORS)
    retry, = [row for row in costs if row["cost_kind"] == "fork_attempt"]
    assert retry["attempt_id"] == "prior" and retry["status"] == "WORKER_RECEIPT_MISSING"
    assert retry["fork_key"] == update["fork_key"] and retry["optimizer_steps"] == "1"
    assert retry["seconds"] == "0.125" and retry["readout"] == "online+fixed_ema"
    assert retry["wall_seconds"] == retry["max_rss_bytes"] == retry["process_gpu_bytes"] == ""


def test_stop_and_test_seal(tmp_path):
    assert not scheduler.stopping(tmp_path)
    (tmp_path / "USER_STOPPED").touch()
    with pytest.raises(RuntimeError, match="stop or pause"):
        cli.guard_stop(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["evaluate", "--artifacts", str(tmp_path)])
    with pytest.raises(SystemExit, match="2"):
        cli.main(["_worker", "--artifacts", str(tmp_path)])
    with pytest.raises(SystemExit, match="2"):
        s0.main(["gpu-fidelity", "--dataset", "movielens", "--device", "cuda:0", "--output", str(tmp_path)])
