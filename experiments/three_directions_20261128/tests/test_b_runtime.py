from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.three_directions_20261128 import runner, scheduler
from experiments.three_directions_20261128.protocol import ROOT, STUDY, stage_jobs, verify_sources


def test_frozen_initial_rosters_and_source():
    art = ROOT / "artifacts" / STUDY
    for stage, count in (("A1", 40), ("B1", 30)):
        jobs = stage_jobs(stage)
        assert len(jobs) == len({job.key for job in jobs}) == count
        assert {job.seed for job in jobs} == set(range(2022, 2027))
        assert {(job.config["epochs"], job.config["batch_size"]) for job in jobs} == {(50, 40960)}
        assert verify_sources(art, stage)["main"] == "641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc"
    assert {j.config["arm"] for j in stage_jobs("B1") if j.arm == "B0"} == {"P00"}
    assert {j.config["arm"] for j in stage_jobs("B1") if j.arm != "B0"} == {"A21"}
    assert {j.arm for j in stage_jobs("A1") if runner._retention_required(j)} == {"B0", "P0"}
    with pytest.raises(ValueError):
        stage_jobs("A2")


def test_host_ram_fallback(monkeypatch):
    def unavailable():
        raise OSError("fixture memory source unavailable")
    monkeypatch.setattr(scheduler, "_proc_memory", unavailable)
    monkeypatch.setattr(scheduler, "_psutil_memory", lambda: (100, 11))
    monkeypatch.setattr(scheduler, "_free_memory", lambda: (1000, 123))
    reading = scheduler.host_memory()
    assert (reading["source"], reading["used_fraction"]) == ("psutil", .89)
    assert reading["errors"][0]["source"] == "/proc/meminfo"
    monkeypatch.setattr(scheduler, "_psutil_memory", unavailable)
    assert scheduler.host_memory()["source"] == "free -b"
    monkeypatch.setattr(scheduler, "_free_memory", unavailable)
    with pytest.raises(RuntimeError):
        scheduler.host_memory()


def test_independent_card_admission_and_post_oom_measurement():
    gpu0 = dict(uuid="A", used=84, total=100, watts=84, limit=100)
    gpu1 = dict(gpu0, uuid="B", watts=85)
    first = SimpleNamespace(workload="A")
    second = SimpleNamespace(workload="B")
    telemetry = dict(host=dict(used_fraction=.89), processes=[])
    assert scheduler.eligible(gpu0, .89)
    assert not scheduler.eligible(gpu0, .90)
    assert not scheduler.eligible(dict(gpu0, used=85), .89)
    assert not scheduler.eligible(gpu1, .89)
    assert scheduler.launch_plan(gpu0, telemetry, [], [first], {}) == (first, None)
    assert scheduler.launch_plan(gpu1, telemetry, [], [first], {}) is None
    pending = dict(gpu="B", diagnostic=False, allocated=False)
    assert scheduler.launch_plan(gpu0, telemetry, [pending], [first], {}) == (first, None)
    assert scheduler.launch_plan(gpu0, telemetry, [dict(pending, gpu="A")], [first], {}) is None
    assert scheduler.launch_plan(gpu0, telemetry, [dict(pending, gpu="A", allocated=True)], [first], {}) == (first, None)
    base = dict(complete=True, epoch=1, cuda_usable=95)
    capacity = dict(A=dict(oom=True, needed_epoch={}, measurements={
        "A": dict(base, peak_process=12), "B": dict(base, peak_process=11)}))
    assert scheduler.launch_plan(gpu0, telemetry, [], [first, second], capacity) == (second, None)
    capacity["A"]["needed_epoch"] = {"A": 2}
    assert scheduler.launch_plan(gpu0, telemetry, [], [first, second], capacity) == (first, 2)
    occupied = dict(telemetry, processes=[dict(uuid="A")])
    assert scheduler.launch_plan(gpu0, occupied, [], [first, second], capacity) == (second, None)
    assert scheduler.failed_allocation("CUDA out of memory. Tried to allocate 1.50 GiB.") == 3 * 2**29


def test_scoped_stop_order(tmp_path, monkeypatch):
    workspaces = [tmp_path / name for name in ("original", "other")]
    current = tmp_path / "current"
    artifacts = current / "artifacts" / STUDY
    monkeypatch.setattr(scheduler, "ROOT", current)
    monkeypatch.setattr(scheduler, "WORKSPACES", workspaces)

    def sentinel(root, study, disposition):
        target = root / "artifacts" / study / disposition
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    sentinel(workspaces[1], "26-11-16_augmentation_gradient_momentum", "USER_STOPPED")
    assert scheduler.stop_status(artifacts) is None
    pause = sentinel(workspaces[0], scheduler.HISTORY, "PAUSE_TRAINING")
    assert Path(scheduler.stop_status(artifacts)["path"]) == pause
    stop = sentinel(workspaces[1], STUDY, "USER_STOPPED")
    assert scheduler.stop_status(artifacts) == dict(status="USER_STOPPED", path=str(stop))
    assert any(item["path"] == str(artifacts / "USER_STOPPED") for item in scheduler.stop_controls(artifacts))


def test_supervision_samples_after_ack_and_before_each_launch(tmp_path, monkeypatch):
    import torch

    stages = [SimpleNamespace(key=f"job{i}", workload="fixture") for i in range(2)]
    gpu = dict(index=0, uuid="GPU-test", used=1, total=100, watts=1, limit=100)
    sample_count = 0
    launches = []

    def measured():
        nonlocal sample_count
        sample_count += 1
        return dict(devices=[gpu], host=dict(used_fraction=.1), processes=[], sample=sample_count)

    def fake_child(argv, **_kwargs):
        directory = Path(argv[argv.index("--invocation") + 1])
        runner.write(directory / "allocations" / "0001.json", dict(site="setup"))
        runner.write(directory / "allocations" / "0002.json", dict(site="first_completed_batch"))
        runner.write(directory / "OUTCOME.json", dict(status="COMPLETE"))
        launches.append(sample_count)
        return SimpleNamespace(pid=len(launches), poll=lambda: 0, returncode=0)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scheduler, "sample", measured)
    monkeypatch.setattr(scheduler, "stop_status", lambda _: None)
    monkeypatch.setattr(scheduler.subprocess, "Popen", fake_child)
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "job_journal", lambda *args: SimpleNamespace(completed=lambda: None))
    monkeypatch.setattr(runner, "publish_tables", lambda *args: None)
    status = scheduler.supervise(tmp_path, "B1", {}, stages)
    assert status["status"] == "COMPLETE"
    assert launches == [2, 5]
    import json
    receipts = [json.loads(line) for line in (tmp_path / "B1" / "DISPATCH.jsonl").read_text().splitlines()]
    reads = [item["telemetry"]["sample"] for item in receipts if item["event"] != "EXIT"]
    assert reads == list(range(2, 8))


@pytest.mark.parametrize("stop_at_oom", [False, True], ids=["across-stages", "scoped-stop"])
def test_supervision_retains_cuda_oom_capacity(tmp_path, monkeypatch, stop_at_oom):
    import torch

    original = tmp_path / "original"
    stop = original / "artifacts" / STUDY / "USER_STOPPED"
    stop.parent.mkdir(parents=True)
    gpu = dict(index=0, uuid="GPU-test", used=1, total=100, watts=1, limit=100)
    telemetry = dict(devices=[gpu], host=dict(used_fraction=.1), processes=[])
    a_job = SimpleNamespace(key="a-job", workload="A1:fixture")
    b_job = SimpleNamespace(key="b-job", workload="B1:fixture")
    launches = []

    def fake_child(argv, **_kwargs):
        stage = argv[argv.index("--stage") + 1]
        invocation = Path(argv[argv.index("--invocation") + 1])
        target = int(argv[-1]) if "--diagnostic-through" in argv else None
        launches.append((stage, target))
        if len(launches) == 1:
            outcome = dict(status="CUDA_OOM", failed_epoch=3, measurement=dict(complete=False))
            if stop_at_oom:
                stop.touch()
        elif target is not None:
            outcome = dict(status="DIAGNOSTIC_COMPLETE",
                           measurement=dict(complete=True, epoch=target, cuda_usable=95, peak_process=12))
        else:
            outcome = dict(status="COMPLETE")
        runner.write(invocation / "OUTCOME.json", outcome)
        returncode = int(outcome["status"] == "CUDA_OOM")
        return SimpleNamespace(pid=len(launches), poll=lambda: returncode, returncode=returncode)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scheduler, "ROOT", tmp_path / "current")
    monkeypatch.setattr(scheduler, "WORKSPACES", (original, tmp_path / "other"))
    monkeypatch.setattr(scheduler, "sample", lambda: telemetry)
    monkeypatch.setattr(scheduler.subprocess, "Popen", fake_child)
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "job_journal", lambda *args: SimpleNamespace(completed=lambda: None))
    monkeypatch.setattr(runner, "publish_tables", lambda *args: None)

    capacity_path = tmp_path / "CAPACITY.json"
    a_status = scheduler.supervise(tmp_path, "A1", {}, [a_job])
    if stop_at_oom:
        assert a_status == dict(status="USER_STOPPED", path=str(stop), stage="A1", test_accesses=0)
        assert launches == [("A1", None)]
        assert scheduler.read(capacity_path) == {
            gpu["uuid"]: dict(oom=True, needed_epoch={a_job.workload: 3}, measurements={})
        }
        stop.unlink()
        a_status = scheduler.supervise(tmp_path, "A1", {}, [a_job])
    assert a_status["status"] == "COMPLETE"
    assert scheduler.supervise(tmp_path, "B1", {}, [b_job])["status"] == "COMPLETE"
    assert launches == [("A1", None), ("A1", 3), ("A1", None), ("B1", 1), ("B1", None)]
    device = scheduler.read(capacity_path)[gpu["uuid"]]
    assert device["oom"] and device["needed_epoch"] == {a_job.workload: 3}
    assert {workload: row["epoch"] for workload, row in device["measurements"].items()} == {
        a_job.workload: 3, b_job.workload: 1
    }
