"""Resume validator fails closed before any training is attempted."""

import json
import os
import sys
import threading
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stage_S1_orchestrator as orchestrator
from stage_S1_orchestrator import (acquire_stage_lock, archive_failed_attempt,
                                   build_tasks, classify_task_state, release_stage_lock,
                                   validate_run)


def test_missing_artifact_is_pending_then_blocked(tmp_path):
    task = build_tasks(str(tmp_path))[0]
    valid, reasons = validate_run(task)
    assert not valid
    assert "run_dir_missing" in reasons
    valid_count, pending, blocked = classify_task_state([task], rerun_failed=False)
    assert valid_count == 0 and pending == [task] and blocked == []
    os.makedirs(task["run_dir"])
    valid_count, pending, blocked = classify_task_state([task], rerun_failed=False)
    # A pre-created empty canonical is not a failed attempt and must remain
    # directly resumable without creating a failed_attempts directory.
    assert valid_count == 0 and pending == [task] and blocked == []
    assert not os.path.exists(os.path.join(task["run_dir"], "failed_attempts"))


def test_archive_sharing_violation_rolls_back_without_partial_attempt(tmp_path, monkeypatch):
    task = build_tasks(str(tmp_path))[0]
    run_dir = task["run_dir"]
    os.makedirs(run_dir)
    for name in ("run_manifest.json", "stdout.log"):
        with open(os.path.join(run_dir, name), "w", encoding="utf-8") as handle:
            handle.write(name)
    retained = os.path.join(run_dir, "failed_attempts", "attempt_002")
    os.makedirs(retained)
    with open(os.path.join(retained, "stdout.log"), "w", encoding="utf-8") as handle:
        handle.write("preserve-attempt-002")

    original_replace = orchestrator.os.replace
    calls = []

    def sharing_violation_on_second_replace(source, target):
        calls.append((source, target))
        if len(calls) == 2:
            error = OSError(32, "sharing violation")
            error.winerror = 32
            raise error
        return original_replace(source, target)

    monkeypatch.setattr(orchestrator.os, "replace", sharing_violation_on_second_replace)
    with pytest.raises(OSError):
        archive_failed_attempt(run_dir)

    assert os.path.isfile(os.path.join(run_dir, "run_manifest.json"))
    assert os.path.isfile(os.path.join(run_dir, "stdout.log"))
    assert os.path.isfile(os.path.join(retained, "stdout.log"))
    assert not os.path.exists(os.path.join(run_dir, "failed_attempts", "attempt_003"))
    assert sorted(name for name in os.listdir(os.path.join(run_dir, "failed_attempts"))
                  if name.startswith("attempt_")) == ["attempt_002"]
    assert not [name for name in os.listdir(os.path.join(run_dir, "failed_attempts"))
                if name.startswith(".staging_attempt_")]


def test_repeated_resume_does_not_duplicate_archive(tmp_path):
    task = build_tasks(str(tmp_path))[0]
    os.makedirs(task["run_dir"])
    with open(os.path.join(task["run_dir"], "run_manifest.json"), "w", encoding="utf-8") as handle:
        handle.write("invalid-attempt")

    _, first_pending, first_blocked = classify_task_state([task], rerun_failed=True)
    _, second_pending, second_blocked = classify_task_state([task], rerun_failed=True)
    archive_root = os.path.join(task["run_dir"], "failed_attempts")
    assert first_pending and not first_blocked
    assert second_pending and not second_blocked
    assert os.path.isfile(os.path.join(archive_root, "attempt_001", "run_manifest.json"))
    assert not os.path.exists(os.path.join(archive_root, "attempt_002"))


def test_launch_header_without_child_marker_is_reusable_not_failed(tmp_path):
    task = build_tasks(str(tmp_path))[0]
    os.makedirs(task["run_dir"])
    with open(os.path.join(task["run_dir"], "stdout.log"), "w", encoding="utf-8") as handle:
        handle.write("[launch 123] command-not-started\n")
    _, pending, blocked = classify_task_state([task], rerun_failed=True)
    assert pending and not blocked
    assert not os.path.exists(os.path.join(task["run_dir"], "failed_attempts"))


def test_canonical_probe_sharing_violation_is_blocking(tmp_path, monkeypatch):
    task = build_tasks(str(tmp_path))[0]
    os.makedirs(task["run_dir"])
    with open(os.path.join(task["run_dir"], "stdout.log"), "w", encoding="utf-8") as handle:
        handle.write("[stage_S1] child_started pid=1\n")
    original_listdir = orchestrator.os.listdir

    def sharing_violation(path):
        if os.path.abspath(path) == os.path.abspath(task["run_dir"]):
            error = OSError(32, "sharing violation")
            error.winerror = 32
            raise error
        return original_listdir(path)

    monkeypatch.setattr(orchestrator.os, "listdir", sharing_violation)
    _, pending, blocked = classify_task_state([task], rerun_failed=True)
    assert not pending and len(blocked) == 1
    assert "canonical_attempt_probe_busy:32" in blocked[0]["validation_reasons"]
    assert not os.path.exists(os.path.join(task["run_dir"], "failed_attempts"))


def test_concurrent_archive_has_one_winner(tmp_path):
    task = build_tasks(str(tmp_path))[0]
    run_dir = task["run_dir"]
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        handle.write("invalid-attempt")
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        results.append(archive_failed_attempt(run_dir))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(bool(value) for value in results) == 1
    assert os.path.isdir(os.path.join(run_dir, "failed_attempts", "attempt_001"))
    assert not os.path.exists(os.path.join(run_dir, "failed_attempts", "attempt_002"))


def test_stage_lock_is_atomic_under_concurrent_resume(tmp_path):
    raw_root = str(tmp_path / "raw")
    barrier = threading.Barrier(2)
    winners = []
    failures = []

    def worker():
        barrier.wait()
        try:
            path = acquire_stage_lock(raw_root)
            winners.append(path)
        except RuntimeError:
            failures.append(True)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1 and len(failures) == 1
    release_stage_lock(winners[0])
    assert not os.path.exists(winners[0])


def test_stage_lock_fsync_and_owner_token_safe_release(tmp_path, monkeypatch):
    raw_root = str(tmp_path / "raw")
    fsync_calls = []
    original_fsync = orchestrator.os.fsync

    def recording_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(orchestrator.os, "fsync", recording_fsync)
    lock_path = acquire_stage_lock(raw_root)
    assert fsync_calls
    replacement = {"pid": 999999, "stage": "S1", "token": "new-owner-token"}
    Path(lock_path).write_text(json.dumps(replacement), encoding="utf-8")
    release_stage_lock(lock_path)
    assert Path(lock_path).is_file()
    assert json.loads(Path(lock_path).read_text(encoding="utf-8")) == replacement
    os.remove(lock_path)


def test_failed_lock_fsync_does_not_leave_partial_owned_lock(tmp_path, monkeypatch):
    raw_root = str(tmp_path / "raw")

    def failed_fsync(_fd):
        raise OSError(5, "fsync failed")

    monkeypatch.setattr(orchestrator.os, "fsync", failed_fsync)
    with pytest.raises(OSError):
        acquire_stage_lock(raw_root)
    assert not os.path.exists(os.path.join(raw_root, orchestrator.LOCK_FILENAME))


def test_child_process_and_redirect_are_closed_before_archive(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    stdout_path = os.path.join(run_dir, "stdout.log")
    returncode = orchestrator._run_child_process(
        [sys.executable, "-c", "print('child-complete')"], stdout_path)
    assert returncode == 0
    stdout_text = Path(stdout_path).read_text(encoding="utf-8")
    assert "[stage_S1] child_started pid=" in stdout_text
    assert "child-complete" in stdout_text
    archive = archive_failed_attempt(run_dir)
    assert archive and os.path.isfile(os.path.join(archive, "stdout.log"))


def test_child_process_handle_is_explicitly_closed_after_context(tmp_path, monkeypatch):
    events = []

    class Handle:
        def Close(self):
            events.append("handle_closed")

    class Process:
        pid = 123

        def __init__(self, *_args, **_kwargs):
            self._handle = Handle()

        def __enter__(self):
            events.append("entered")
            return self

        def wait(self):
            events.append("waited")
            return 0

        def __exit__(self, *_args):
            events.append("exited")

    monkeypatch.setattr(orchestrator.subprocess, "Popen", Process)
    stdout_path = str(tmp_path / "stdout.log")
    assert orchestrator._run_child_process(["fixture"], stdout_path) == 0
    assert events == ["entered", "waited", "exited", "handle_closed"]


def test_queue_manifest_replace_is_atomic_and_fsynced(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text('{"resume_state":"OLD"}\n', encoding="utf-8")
    fsync_calls = []
    original_fsync = orchestrator.os.fsync
    original_replace = orchestrator.os.replace

    def recording_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    def failed_replace(source, target):
        if os.path.abspath(target) == os.path.abspath(queue_path):
            raise OSError(32, "replace blocked")
        return original_replace(source, target)

    monkeypatch.setattr(orchestrator.os, "fsync", recording_fsync)
    monkeypatch.setattr(orchestrator.os, "replace", failed_replace)
    with pytest.raises(OSError):
        orchestrator._write_queue_manifest(str(queue_path), [], resume_state="RUNNING")
    assert fsync_calls
    assert json.loads(queue_path.read_text(encoding="utf-8"))["resume_state"] == "OLD"
    assert not list(tmp_path.glob("queue.json.tmp-*"))


def test_dry_run_cannot_overwrite_live_queue_state(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    queue_path = tmp_path / "queue.json"
    queue_path.write_text('{"resume_state":"RUNNING"}\n', encoding="utf-8")
    held_lock = acquire_stage_lock(str(raw_root))
    monkeypatch.setattr(orchestrator, "build_tasks", lambda _root: [])
    monkeypatch.setattr(orchestrator, "validate_matrix", lambda _tasks: None)
    monkeypatch.setattr(orchestrator, "assert_formal_launcher", lambda _args: None)
    monkeypatch.setattr(sys, "argv", ["stage_S1_orchestrator.py", "--dry_run",
                                      "--raw_root", str(raw_root),
                                      "--queue_manifest", str(queue_path)])
    try:
        with pytest.raises(RuntimeError):
            orchestrator.main()
        assert json.loads(queue_path.read_text(encoding="utf-8"))["resume_state"] == "RUNNING"
    finally:
        release_stage_lock(held_lock)


def _run_main_fixture(tmp_path, monkeypatch, pending, child_returncode=0, max_runs=192):
    raw_root = tmp_path / "raw"
    queue_path = tmp_path / "queue.json"
    task = {"formal_run_key": "single_domain|fixture", "run_type": "single_domain",
            "run_dir": str(raw_root / "run")}
    monkeypatch.setattr(orchestrator, "build_tasks", lambda _root: [task])
    monkeypatch.setattr(orchestrator, "validate_matrix", lambda _tasks: None)
    monkeypatch.setattr(orchestrator, "assert_formal_launcher", lambda _args: None)
    monkeypatch.setattr(orchestrator, "runtime_preflight", lambda _args: None)

    saw_running_with_lock = []

    def classify(_tasks, _rerun_failed, require_formal=True):
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        saw_running_with_lock.append(payload.get("resume_state") == "RUNNING"
                                      and os.path.isfile(raw_root / orchestrator.LOCK_FILENAME))
        return (0, [task] if pending else [], [])

    monkeypatch.setattr(orchestrator, "classify_task_state", classify)
    monkeypatch.setattr(orchestrator, "command_for", lambda _task, _args: ["fixture-child"])
    monkeypatch.setattr(orchestrator, "_run_child_process",
                        lambda _command, _stdout: child_returncode)
    argv = ["stage_S1_orchestrator.py", "--raw_root", str(raw_root),
            "--queue_manifest", str(queue_path), "--max_runs", str(max_runs)]
    monkeypatch.setattr(sys, "argv", argv)
    return orchestrator.main(), json.loads(queue_path.read_text(encoding="utf-8")), saw_running_with_lock


def test_queue_resume_states_are_accurate(tmp_path, monkeypatch):
    code, payload, saw_running = _run_main_fixture(
        tmp_path, monkeypatch, pending=True, max_runs=0)
    assert code == 0 and saw_running == [True]
    assert payload["resume_state"] == "PARTIAL_MAX_RUNS"
    assert payload["pending_after"] == 1 and payload["launched"] == 0


def test_queue_complete_requires_zero_pending(tmp_path, monkeypatch):
    code, payload, saw_running = _run_main_fixture(
        tmp_path, monkeypatch, pending=False)
    assert code == 0 and saw_running == [True]
    assert payload["resume_state"] == "COMPLETE"
    assert payload["pending_after"] == 0 and payload["launched"] == 0


def test_queue_failed_child_state_is_accurate(tmp_path, monkeypatch):
    code, payload, saw_running = _run_main_fixture(
        tmp_path, monkeypatch, pending=True, child_returncode=7)
    assert code == 7 and saw_running == [True]
    assert payload["resume_state"] == "FAILED_CHILD"
    assert payload["returncode"] == 7 and payload["launched"] == 1


def test_valid_classification_never_archives_or_overwrites(tmp_path, monkeypatch):
    task = build_tasks(str(tmp_path))[0]
    os.makedirs(task["run_dir"])
    sentinel = Path(task["run_dir"]) / "run_manifest.json"
    sentinel.write_text("valid-sentinel", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "validate_run", lambda *_args, **_kwargs: (True, []))
    valid_count, pending, blocked = classify_task_state([task], rerun_failed=True)
    assert valid_count == 1 and not pending and not blocked
    assert sentinel.read_text(encoding="utf-8") == "valid-sentinel"
    assert not (Path(task["run_dir"]) / "failed_attempts").exists()


def test_dead_pid_lock_is_reclaimed_and_audit_recorded(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / orchestrator.LOCK_FILENAME
    lock_path.write_text(json.dumps({"pid": 32028, "stage": "S1",
                                     "token": "dead-orchestrator-token"}),
                         encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_pid_is_deterministically_dead", lambda _pid: True)
    path = acquire_stage_lock(str(raw_root))
    assert path == str(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["stage"] == "S1"
    assert payload["token"] != "dead-orchestrator-token"
    reclaim_log = raw_root / orchestrator.LOCK_RECLAIM_LOG_FILENAME
    assert reclaim_log.is_file()
    events = json.loads(reclaim_log.read_text(encoding="utf-8"))
    assert isinstance(events, list) and len(events) == 1
    assert events[0]["previous_pid"] == 32028
    assert events[0]["previous_token"] == "dead-orchestrator-token"
    assert events[0]["reclaimed_by_pid"] == os.getpid()
    release_stage_lock(path)
    assert not lock_path.exists()


def test_alive_pid_lock_is_not_reclaimed(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / orchestrator.LOCK_FILENAME
    lock_path.write_text(json.dumps({"pid": 42424, "stage": "S1",
                                     "token": "live-orchestrator-token"}),
                         encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_pid_is_deterministically_dead", lambda _pid: False)
    with pytest.raises(RuntimeError):
        acquire_stage_lock(str(raw_root))
    assert lock_path.read_text(encoding="utf-8") == json.dumps(
        {"pid": 42424, "stage": "S1", "token": "live-orchestrator-token"})
    assert not (raw_root / orchestrator.LOCK_RECLAIM_LOG_FILENAME).exists()


def test_corrupt_lock_is_not_reclaimed(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / orchestrator.LOCK_FILENAME
    lock_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_pid_is_deterministically_dead", lambda _pid: True)
    with pytest.raises(RuntimeError):
        acquire_stage_lock(str(raw_root))
    assert lock_path.read_text(encoding="utf-8") == "{not-json"
    assert not (raw_root / orchestrator.LOCK_RECLAIM_LOG_FILENAME).exists()


def test_wrong_stage_lock_is_not_reclaimed(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / orchestrator.LOCK_FILENAME
    lock_path.write_text(json.dumps({"pid": 32028, "stage": "E1",
                                     "token": "wrong-stage-token"}),
                         encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_pid_is_deterministically_dead", lambda _pid: True)
    with pytest.raises(RuntimeError):
        acquire_stage_lock(str(raw_root))
    assert json.loads(lock_path.read_text(encoding="utf-8"))["stage"] == "E1"
    assert not (raw_root / orchestrator.LOCK_RECLAIM_LOG_FILENAME).exists()


def test_self_pid_lock_is_not_reclaimed(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / orchestrator.LOCK_FILENAME
    lock_path.write_text(json.dumps({"pid": os.getpid(), "stage": "S1",
                                     "token": "self-token"}),
                         encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_pid_is_deterministically_dead", lambda _pid: True)
    with pytest.raises(RuntimeError):
        acquire_stage_lock(str(raw_root))
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "self-token"


def test_pid_liveness_check_treats_windows_access_denied_as_alive(monkeypatch):
    """Access-denied on Windows must never be treated as a dead PID."""

    if os.name != "nt":
        monkeypatch.setattr(orchestrator.os, "name", "nt")
        import ctypes
        class _FakeHandle(int):
            def __bool__(self):
                return False

        def _fake_open_process(_flags, _inherit, _pid):
            ctypes.windll.kernel32.GetLastError.restype = None
            return _FakeHandle()

        class _FakeCtypes:
            windll = ctypes.windll
            def GetLastError(self):
                return 5

        monkeypatch.setattr(ctypes, "GetLastError", _FakeCtypes().GetLastError)
        monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", _fake_open_process)
        assert orchestrator._pid_is_deterministically_dead(1234) is False
        return
    import ctypes
    original_open = ctypes.windll.kernel32.OpenProcess
    original_get_error = getattr(ctypes, "GetLastError", lambda: 0)

    def _fake_open_process(_flags, _inherit, _pid):
        return 0

    def _fake_get_error():
        return 5

    monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", _fake_open_process)
    monkeypatch.setattr(ctypes, "GetLastError", _fake_get_error)
    try:
        assert orchestrator._pid_is_deterministically_dead(1234) is False
    finally:
        monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", original_open)
        if original_get_error is not None:
            monkeypatch.setattr(ctypes, "GetLastError", original_get_error)


def test_pid_liveness_check_treats_windows_invalid_parameter_as_dead(monkeypatch):
    """An ERROR_INVALID_PARAMETER Windows return is the dead-PID signal."""

    if os.name != "nt":
        monkeypatch.setattr(orchestrator.os, "name", "nt")
        import ctypes

        def _fake_open_process(_flags, _inherit, _pid):
            return 0

        def _fake_get_error():
            return 87

        monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", _fake_open_process)
        monkeypatch.setattr(ctypes, "GetLastError", _fake_get_error)
        assert orchestrator._pid_is_deterministically_dead(32028) is True
        return
    import ctypes

    def _fake_open_process(_flags, _inherit, _pid):
        return 0

    def _fake_get_error():
        return 87

    monkeypatch.setattr(ctypes.windll.kernel32, "OpenProcess", _fake_open_process)
    monkeypatch.setattr(ctypes, "GetLastError", _fake_get_error)
    assert orchestrator._pid_is_deterministically_dead(32028) is True
