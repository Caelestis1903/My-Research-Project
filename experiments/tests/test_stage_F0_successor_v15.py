from __future__ import annotations

import hashlib
import ast
import fcntl
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
RUN_SCRIPTS = EXPERIMENTS / "stage_F0_outputs/runs"
if str(RUN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUN_SCRIPTS))

import stage_F0_successor_v15 as v15
import run_search_claim_worker as claim_worker


def _write_non75_peer_diagnostic(directory, *, boundary, returncode, stdout,
                                 stderr, marker_paths):
    """Persist the first failure artifact before optional marker harvesting."""
    directory.mkdir(parents=True, exist_ok=False)
    markers = {}
    for name, marker_path in marker_paths.items():
        present = marker_path is not None and marker_path.is_file()
        markers[name] = {
            "status": "present" if present else "missing",
            "path": str(marker_path) if marker_path is not None else None,
            "content": marker_path.read_text(encoding="utf-8") if present else None,
        }
    result = {"boundary": boundary, "returncode": returncode,
              "stdout": stdout, "stderr": stderr, "markers": markers}
    (directory / "supervisor_result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return directory


def _wait_for_peer_rendezvous(*, supervisor_process, marker_paths, timeout_seconds,
                              diagnostic_directory):
    """Wait for the post-snapshot claim and both later deferred-child markers."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    reason = None
    rows = {}
    marker_errors = {}
    missing = list(marker_paths)
    while True:
        rows = {}
        marker_errors = {}
        for name, path in marker_paths.items():
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                value = row.get("monotonic") if isinstance(row, dict) else None
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))):
                    raise ValueError("marker lacks a finite numeric monotonic timestamp")
                rows[name] = row
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                marker_errors[name] = "{}: {}".format(type(exc).__name__, exc)
        missing = [name for name in marker_paths if name not in rows]
        if not missing:
            order = ("snapshot", "peer_wait", "peer_claim", "supervisor_resume", "child_defer")
            monotonic_values = [rows[name]["monotonic"] for name in order]
            if all(left < right for left, right in zip(
                    monotonic_values, monotonic_values[1:])):
                return rows
            reason = "RENDEZVOUS_MARKER_ORDER_INVALID"
            break
        returncode = supervisor_process.poll()
        if returncode is not None:
            reason = "SUPERVISOR_EXITED_BEFORE_COMPLETE_RENDEZVOUS_MARKERS"
            break
        if time.monotonic() >= deadline:
            reason = "RENDEZVOUS_TIMEOUT"
            break
        time.sleep(0.02)

    stdout = stderr = ""
    if supervisor_process.poll() is not None:
        try:
            stdout, stderr = supervisor_process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "<communicate timeout>", "<communicate timeout>"
    marker_state = {}
    for name, path in marker_paths.items():
        try:
            content = path.read_text(encoding="utf-8")
            marker_state[name] = {"status": "complete" if name in rows else "invalid",
                                  "content": content,
                                  "error": marker_errors.get(name)}
        except (OSError, UnicodeError) as exc:
            marker_state[name] = {"status": "missing", "path": str(path),
                                  "error": "{}: {}".format(type(exc).__name__, exc)}
    returncode = supervisor_process.poll()
    try:
        pgid = os.getpgid(supervisor_process.pid) if returncode is None else None
    except ProcessLookupError:
        pgid = None
    diagnostic_directory.mkdir(parents=True, exist_ok=False)
    payload = {"reason": reason, "supervisor_pid": supervisor_process.pid,
        "supervisor_pgid": pgid, "process_alive_at_failure": returncode is None,
        "supervisor_returncode": returncode,
        "missing_markers": missing, "markers": marker_state,
        "marker_errors": marker_errors,
        "stdout": stdout, "stderr": stderr,
        "deadline_seconds": timeout_seconds,
        "elapsed_seconds": time.monotonic() - started}
    (diagnostic_directory / "rendezvous_failure.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    raise AssertionError("peer rendezvous failed; evidence=" + str(diagnostic_directory))


def _embedded_peer_supervisor_wrapper() -> str:
    source_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    target = next(node for node in source_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "test_real_pending_graph_release_reentry_terminal_and_f140_claim")
    for node in ast.walk(target):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            candidate = node.args[0].value
            if "capture_f140_state" in candidate and "pre_spawn_snapshot_captured" in candidate:
                return candidate
    raise AssertionError("embedded peer supervisor wrapper was not found")


def test_non75_diagnostic_persists_output_with_missing_markers(tmp_path):
    out = tmp_path / "non75"
    _write_non75_peer_diagnostic(out, boundary="SERIALIZED", returncode=1,
        stdout="raw stdout", stderr="raw stderr",
        marker_paths={"snapshot": None, "resume": tmp_path / "absent"})
    payload = json.loads((out / "supervisor_result.json").read_text())
    assert payload["returncode"] == 1
    assert payload["stdout"] == "raw stdout"
    assert payload["stderr"] == "raw stderr"
    assert payload["markers"]["snapshot"]["status"] == "missing"
    assert payload["markers"]["resume"]["status"] == "missing"


def test_parent_peer_rendezvous_waits_for_both_later_markers_and_orders_them(tmp_path):
    paths = {name: tmp_path / (name + ".json") for name in (
        "snapshot", "peer_wait", "peer_claim", "supervisor_resume", "child_defer")}
    writer = ("import json,sys,time; from pathlib import Path; "
        "[(time.sleep(.02),Path(p).write_text(json.dumps({'monotonic':time.monotonic()}))) "
        "for p in sys.argv[1:]]")
    supervisor = subprocess.Popen([sys.executable, "-c", writer,
        *[str(path) for path in paths.values()]], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    rows = _wait_for_peer_rendezvous(supervisor_process=supervisor,
        marker_paths=paths, timeout_seconds=2,
        diagnostic_directory=tmp_path / "unexpected-success-diagnostic")
    assert list(rows) == list(paths)
    stdout, stderr = supervisor.communicate(timeout=2)
    assert supervisor.returncode == 0, (stdout, stderr)
    times = [rows[name]["monotonic"] for name in paths]
    assert all(left < right for left, right in zip(times, times[1:]))


def test_parent_peer_rendezvous_archives_early_supervisor_exit(tmp_path):
    paths = {name: tmp_path / (name + ".json") for name in (
        "snapshot", "peer_wait", "peer_claim", "supervisor_resume", "child_defer")}
    writer = ("import json,sys,time; from pathlib import Path; "
        "[Path(p).write_text(json.dumps({'monotonic':time.monotonic()})) "
        "for p in sys.argv[1:4]]; Path(sys.argv[4]).open('w').close(); "
        "print('diagnostic stdout'); "
        "print('diagnostic stderr',file=sys.stderr); sys.exit(17)")
    supervisor = subprocess.Popen([sys.executable, "-c", writer,
        *[str(path) for path in paths.values()]], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    evidence = tmp_path / "early-exit-evidence"
    with pytest.raises(AssertionError, match="peer rendezvous failed"):
        _wait_for_peer_rendezvous(supervisor_process=supervisor,
            marker_paths=paths, timeout_seconds=2, diagnostic_directory=evidence)
    payload = json.loads((evidence / "rendezvous_failure.json").read_text())
    assert payload["reason"] == "SUPERVISOR_EXITED_BEFORE_COMPLETE_RENDEZVOUS_MARKERS"
    assert payload["supervisor_returncode"] == 17
    assert payload["missing_markers"] == ["supervisor_resume", "child_defer"]
    assert payload["markers"]["supervisor_resume"]["status"] == "invalid"
    assert "JSONDecodeError" in payload["marker_errors"]["supervisor_resume"]
    assert payload["stdout"].strip() == "diagnostic stdout"
    assert payload["stderr"].strip() == "diagnostic stderr"


def test_parent_peer_rendezvous_retries_marker_opened_before_write(tmp_path):
    paths = {name: tmp_path / (name + ".json") for name in (
        "snapshot", "peer_wait", "peer_claim", "supervisor_resume", "child_defer")}
    opened = tmp_path / "child-defer-opened"
    script = r"""
import json,sys,time
from pathlib import Path
paths=[Path(value) for value in sys.argv[1:6]]
opened=Path(sys.argv[6])
target=paths[-1]
real_open=Path.open
def pause_after_open(self,*args,**kwargs):
    handle=real_open(self,*args,**kwargs)
    if self == target:
        opened.write_text('open-returned')
        time.sleep(.30)
    return handle
Path.open=pause_after_open
for path in paths[:-1]:
    path.write_text(json.dumps({'monotonic':time.monotonic()}))
    time.sleep(.02)
target.write_text(json.dumps({'monotonic':time.monotonic()}))
"""
    supervisor = subprocess.Popen([sys.executable, "-c", script,
        *[str(path) for path in paths.values()], str(opened)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 2
    while not opened.exists() and supervisor.poll() is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert opened.is_file(), supervisor.poll()
    assert paths["child_defer"].is_file()
    assert paths["child_defer"].stat().st_size == 0
    rows = _wait_for_peer_rendezvous(supervisor_process=supervisor,
        marker_paths=paths, timeout_seconds=2,
        diagnostic_directory=tmp_path / "unexpected-torn-marker-diagnostic")
    stdout, stderr = supervisor.communicate(timeout=2)
    assert supervisor.returncode == 0, (stdout, stderr)
    assert rows["child_defer"]["monotonic"] > rows["supervisor_resume"]["monotonic"]


def test_initial_peer_snapshot_and_resume_are_immutable_across_later_capture(
        tmp_path, monkeypatch):
    wrapper_tree = ast.parse(_embedded_peer_supervisor_wrapper())
    initial_state_node = next(node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "initial_snapshot_recorded"
                for target in node.targets))
    capture_node = next(node for node in ast.walk(wrapper_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "capture_then_signal")
    namespace = {"os": os, "Path": Path, "time": time,
                 "real_capture": lambda _root, _entries: {"state": "initial"}}
    exec(compile(ast.Module(body=[initial_state_node, capture_node], type_ignores=[]),
                 "<peer capture wrapper>", "exec"), namespace)

    snapshot_bytes = tmp_path / "snapshot.json"
    snapshot_done = tmp_path / "snapshot-done.json"
    peer_ready = tmp_path / "peer-claimed.json"
    peer_waiting = tmp_path / "peer-waiting.json"
    resume = tmp_path / "supervisor-resumed.json"
    defer = tmp_path / "child-defer.json"
    for key, value in {
        "V15_SNAPSHOT_BYTES": snapshot_bytes,
        "V15_SNAPSHOT_DONE": snapshot_done,
        "V15_PEER_READY": peer_ready,
        "V15_DEFER_RESUMED": resume,
    }.items():
        monkeypatch.setenv(key, str(value))

    result = []
    failure = []
    snapshot_opened = threading.Event()
    allow_snapshot_write = threading.Event()
    original_path_open = Path.open

    def pause_after_snapshot_open(path, mode="r", *args, **kwargs):
        # Exercise the real Path.write_text open/truncate-before-write window.
        handle = original_path_open(path, mode, *args, **kwargs)
        if path == snapshot_done and "w" in mode and not snapshot_opened.is_set():
            snapshot_opened.set()
            allow_snapshot_write.wait(timeout=2)
        return handle

    monkeypatch.setattr(Path, "open", pause_after_snapshot_open)

    def first_capture():
        try:
            result.append(namespace["capture_then_signal"](tmp_path, []))
        except BaseException as exc:
            failure.append(exc)
    capture_thread = threading.Thread(target=first_capture)
    capture_thread.start()
    try:
        assert snapshot_opened.wait(timeout=2), "snapshot writer did not reach open-before-write"
        assert snapshot_done.is_file()
        assert snapshot_done.stat().st_size == 0, "fixture did not expose the empty-marker window"
        allow_snapshot_write.set()

        # A path's existence is insufficient: keep retrying until the ordinary
        # Path.write_text producer has published a complete, identity-bound row.
        deadline = time.monotonic() + 2
        snapshot_meta = None
        last_error = None
        while time.monotonic() < deadline:
            try:
                candidate = json.loads(snapshot_done.read_text(encoding="utf-8"))
                snapshot_data = snapshot_bytes.read_bytes()
                if (isinstance(candidate, dict)
                        and candidate.get("phase") == "pre_spawn_snapshot_captured"
                        and candidate.get("snapshot_sha256") == hashlib.sha256(
                            snapshot_data).hexdigest()
                        and isinstance(candidate.get("monotonic"), (int, float))
                        and not isinstance(candidate.get("monotonic"), bool)
                        and math.isfinite(float(candidate["monotonic"]))):
                    snapshot_meta = candidate
                    break
                last_error = "snapshot marker incomplete or inconsistent"
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                last_error = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(0.005)
        assert snapshot_meta is not None, "complete snapshot marker not published: {}".format(
            last_error)

        peer_waiting.write_text(json.dumps({"monotonic": time.monotonic()}) + "\n")
        peer_ready.write_text(json.dumps({"monotonic": time.monotonic()}) + "\n")
        capture_thread.join(timeout=2)
        assert not capture_thread.is_alive(), "initial capture failed to resume after peer claim"
        assert not failure, failure
        initial_bytes = snapshot_bytes.read_bytes()
        initial_done = snapshot_done.read_bytes()
        initial_resume = resume.read_bytes()
        resume_meta = json.loads(initial_resume)
        defer.write_text(json.dumps({"monotonic": time.monotonic()}) + "\n")

        namespace["real_capture"] = lambda _root, _entries: {"state": "later-capture"}
        assert namespace["capture_then_signal"](tmp_path, []) == {"state": "later-capture"}
        assert snapshot_bytes.read_bytes() == initial_bytes
        assert snapshot_done.read_bytes() == initial_done
        assert resume.read_bytes() == initial_resume
        assert snapshot_meta["snapshot_sha256"] == hashlib.sha256(initial_bytes).hexdigest()
        assert resume_meta["snapshot_sha256"] == snapshot_meta["snapshot_sha256"]
        marker_times = [snapshot_meta["monotonic"],
            json.loads(peer_waiting.read_text())["monotonic"],
            json.loads(peer_ready.read_text())["monotonic"], resume_meta["monotonic"],
            json.loads(defer.read_text())["monotonic"]]
        assert all(left < right for left, right in zip(marker_times, marker_times[1:]))
    finally:
        # The embedded capture waits for peer_ready for up to ten minutes. Always
        # release it and join before pytest restores monkeypatched Path methods.
        allow_snapshot_write.set()
        if capture_thread.is_alive() and not peer_ready.exists():
            peer_ready.write_text(json.dumps({"monotonic": time.monotonic()}) + "\n")
        capture_thread.join(timeout=2)
        assert not capture_thread.is_alive(), "capture thread leaked after cleanup rendezvous"


def test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral():
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "test_real_pending_graph_release_reentry_terminal_and_f140_claim")
    scripts = []
    for node in ast.walk(target):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            candidate = node.args[0].value
            if "run_search_trajectory_child" in candidate:
                scripts.append(candidate)
            elif "capture_f140_state" in candidate and "pre_spawn_snapshot_captured" in candidate:
                scripts.append(candidate)
    child = next(script for script in scripts if "run_search_trajectory_child" in script)
    supervisor = next(script for script in scripts if "pre_spawn_snapshot_captured" in script)
    compile(child, "<v15 child fixture>", "exec")
    compile(supervisor, "<v15 supervisor fixture>", "exec")
    assert child.index("faulthandler, json") < child.index("def fake_live")
    assert "wait_for_changed_peer_before_runner" not in child
    assert "while not peer_ready.is_file()" not in child
    assert child.index("phase': 'resource_defer_decision") < child.index(
        "raise v15.ResourceDeferred")
    assert supervisor.index("pre_spawn_snapshot_captured") < supervisor.index(
        "while not peer_ready.is_file()") < supervisor.index(
        "supervisor_snapshot_wait_complete") < supervisor.rindex("return state")
    assert "initial_snapshot_recorded['value']" in supervisor
    assert supervisor.index("if initial_snapshot_recorded['value']") < supervisor.index(
        "snapshot_bytes =")
    assert supervisor.index("V15_DEFER_RESUMED") < supervisor.index(
        "initial_snapshot_recorded['value'] = True")


def test_frozen_design_and_bootstrap_remediation_identities_are_exact():
    frozen = v15._assert_frozen_artifacts(v15.ROOT)
    assert frozen[v15.HOLD_REL] == v15.INCIDENT_HOLD_SHA256
    assert frozen[v15.QUEUE_REL] == v15.QUEUE_SHA256
    assert v15.sha256_file(v15.ROOT / v15.APPROVAL_RECORD) == v15.APPROVAL_RECORD_SHA256
    assert v15.sha256_file(v15.ROOT / v15.REMEDIATION_CANDIDATE) == v15.REMEDIATION_CANDIDATE_SHA256


def test_control_path_rejects_final_and_parent_symlink_aliases(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    target = real / "artifact.json"
    target.write_text("{}\n")
    parent_link = tmp_path / "experiments"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlink"):
        v15._repo_path(tmp_path, "experiments/artifact.json")
    (tmp_path / "experiments").unlink()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/real.json").write_text("{}\n")
    (tmp_path / "experiments/alias.json").symlink_to(
        tmp_path / "experiments/real.json")
    with pytest.raises(PermissionError, match="symlink"):
        v15._repo_path(tmp_path, "experiments/alias.json")


def test_transaction_ledger_rejects_skipped_or_foreign_terminal_phases(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"schema": v15.LEDGER_SCHEMA,
                                  "event": "PREPARED", "transaction_id": "tx-a"}) + "\n"
                     + json.dumps({"schema": v15.LEDGER_SCHEMA,
                                   "event": "COMMITTED", "transaction_id": "tx-b"}) + "\n")
    with pytest.raises(PermissionError, match="order/identity|foreign transaction"):
        v15._transaction_rows(ledger, "tx-a")


@pytest.mark.parametrize("verdict", ["PASS", "PASS WITH FIXES"])
def test_amendment_accepts_both_reviewed_verdicts_and_exact_domains(verdict):
    v15._validate_amendment({
        "status": "APPROVED", "reviewer_verdict": verdict,
        "frozen_scientific_code_hash": "s", "runtime_control_code_hash": "r",
        "control_hash_inventory_sha256": "c",
        "approval_record_sha256": v15.APPROVAL_RECORD_SHA256,
        "remediation_candidate_sha256": v15.REMEDIATION_CANDIDATE_SHA256,
    }, scientific_sha256="s", runtime_sha256="r", control_sha256="c")


@pytest.mark.parametrize("field,value,match", [
    ("status", None, "status"), ("status", "PENDING", "status"),
    ("reviewer_verdict", None, "verdict"), ("reviewer_verdict", "FAIL", "verdict"),
    ("frozen_scientific_code_hash", "0" * 64, "scientific"),
    ("runtime_control_code_hash", "0" * 64, "runtime"),
    ("control_hash_inventory_sha256", "0" * 64, "control"),
    ("approval_record_sha256", "0" * 64, "approval"),
    ("remediation_candidate_sha256", "0" * 64, "candidate"),
])
def test_amendment_missing_invalid_and_wrong_digests_fail_closed(field, value, match):
    amendment = {
        "status": "APPROVED", "reviewer_verdict": "PASS",
        "frozen_scientific_code_hash": "s", "runtime_control_code_hash": "r",
        "control_hash_inventory_sha256": "c",
        "approval_record_sha256": v15.APPROVAL_RECORD_SHA256,
        "remediation_candidate_sha256": v15.REMEDIATION_CANDIDATE_SHA256,
    }
    if value is None:
        amendment.pop(field)
    else:
        amendment[field] = value
    with pytest.raises(PermissionError, match=match):
        v15._validate_amendment(amendment, scientific_sha256="s",
                                runtime_sha256="r", control_sha256="c")


def test_gpu_memory_uses_strict_integer_boundary_and_rejects_bool():
    def row(used, total=1000):
        return {"readable": True, "eligible": True, "uuid": "GPU-x",
                "memory_total_bytes": total, "memory_used_bytes": used}
    assert v15.fresh_gpu_eligible(row(899))
    assert not v15.fresh_gpu_eligible(row(900))
    assert not v15.fresh_gpu_eligible(row(True))
    assert not v15.fresh_gpu_eligible({**row(10), "memory_used_fraction": 0.9})


def test_one_target_gpu_passes_when_non_target_is_unreadable_and_uuid_is_bound():
    devices = [
        {"index": 0, "uuid": "GPU-0", "readable": True, "eligible": True,
         "memory_total_bytes": 1000, "memory_used_bytes": 899},
        {"index": 1, "uuid": "GPU-1", "readable": False, "eligible": False,
         "memory_total_bytes": None, "memory_used_bytes": None},
    ]
    assert v15.validate_target_rows(devices, [{"index": 0, "uuid": "GPU-0"}]) == {0: "GPU-0"}
    with pytest.raises(PermissionError, match="identity/memory"):
        v15.validate_target_rows(devices, [{"index": 0, "uuid": "GPU-other"}])
    with pytest.raises(PermissionError, match="ambiguous"):
        v15.validate_target_rows(devices + [dict(devices[0])],
                                 [{"index": 0, "uuid": "GPU-0"}])


def test_target_gpu_rejects_exactly_ninety_percent_and_duplicate_uuid():
    row = {"index": 0, "uuid": "GPU-0", "readable": True, "eligible": True,
           "memory_total_bytes": 1000, "memory_used_bytes": 900}
    with pytest.raises(PermissionError, match="identity/memory"):
        v15.validate_target_rows([row], [{"index": 0, "uuid": "GPU-0"}])
    rows = [dict(row, memory_used_bytes=100),
            {"index": 1, "uuid": "GPU-0", "readable": True, "eligible": True,
             "memory_total_bytes": 1000, "memory_used_bytes": 100}]
    with pytest.raises(PermissionError, match="duplicated"):
        v15.validate_target_rows(rows, [{"index": 0, "uuid": "GPU-0"},
                                        {"index": 1, "uuid": "GPU-0"}])


def test_fresh_only_probe_treats_broken_symlink_as_history(tmp_path):
    canonical = json.dumps({"fixture": "fresh-key"}, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    full_sha = hashlib.sha256(canonical.encode()).hexdigest()
    rid = full_sha[:20]
    entry = {"run_id": rid, "canonical_key_json": canonical,
             "run_id_full_sha256": full_sha}
    assert v15.key_is_untouched(tmp_path, entry)
    path = v15.fresh_member_paths(tmp_path, rid)["status"]
    path.parent.mkdir(parents=True)
    path.symlink_to(path.parent / "missing.json")
    assert not v15.key_is_untouched(tmp_path, entry)
    with pytest.raises(PermissionError, match="history"):
        v15.validate_fresh_entry(tmp_path, entry, [rid])


def test_real_frozen_f140_canonical_key_passes_identity_and_freshness(tmp_path):
    queue = v15._read_json(v15.ROOT / v15.QUEUE_REL)
    f140 = v15.frozen_f140_ids(v15.ROOT)
    entry = next(row for row in queue["entries"] if row["run_id"] in set(f140))
    v15.validate_fresh_entry(tmp_path, entry, f140)


def test_controlled_clock_expires_live_observation_after_sixty_seconds(monkeypatch):
    real_monotonic = time.monotonic
    clock = {"offset": 0.0}
    monkeypatch.setattr(v15.time, "monotonic",
                        lambda: real_monotonic() + clock["offset"])
    started = v15.time.monotonic()
    assert v15.observation_is_fresh(started_monotonic=started)
    clock["offset"] = 61.25
    assert v15.time.monotonic() - started > 60.0
    assert not v15.observation_is_fresh(started_monotonic=started)


def test_fresh_f140_validator_rejects_outside_set_and_wrong_full_digest(tmp_path):
    queue = v15._read_json(v15.ROOT / v15.QUEUE_REL)
    f140 = v15.frozen_f140_ids(v15.ROOT)
    entry = next(row for row in queue["entries"] if row["run_id"] in set(f140))
    with pytest.raises(PermissionError, match="outside frozen F140"):
        v15.validate_fresh_entry(tmp_path, entry, [])
    with pytest.raises(PermissionError, match="identity"):
        v15.validate_fresh_entry(tmp_path, {**entry, "run_id_full_sha256": "0" * 64}, f140)


def test_canonical_mutation_phase_precedes_first_lock_creation_fault(tmp_path, monkeypatch):
    import stage_F0_queue as queue_module

    frozen_queue = json.loads((v15.ROOT / v15.QUEUE_REL).read_text())
    f140 = set(v15.frozen_f140_ids(v15.ROOT))
    entry = next(row for row in frozen_queue["entries"]
                 if row["run_id"] in f140)
    phase_log = tmp_path / "worker_logs/invocation.phases.jsonl"
    phase_log.parent.mkdir(parents=True)
    monkeypatch.setenv("F0_PHASE_LOG", str(phase_log))
    monkeypatch.setenv("F0_WORKER_ID", "0")
    monkeypatch.setenv("F0_INVOCATION_NONCE", "canonical-lock-boundary-nonce")
    original_lock = queue_module.QueueFileLock

    class FaultImmediatelyBeforeCanonicalLock:
        def __init__(self, path, *args, **kwargs):
            if "/canonical_locks/" in str(path):
                raise RuntimeError("injected immediately before canonical lock creation")
            original_lock(path, *args, **kwargs)

    monkeypatch.setattr(queue_module, "QueueFileLock", FaultImmediatelyBeforeCanonicalLock)
    with pytest.raises(RuntimeError, match="immediately before canonical lock creation"):
        queue_module._claim_next_entry_unlocked(
            {"entries": [entry]}, str(tmp_path), gpu_uuid="GPU-fixture",
            gpu_index=0, fresh_only_v15=True, frozen_f140_ids=f140)
    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert [row["phase"] for row in events] == ["CANONICAL_MUTATION_STARTED"]
    assert events[0]["run_id"] == entry["run_id"]
    assert not os.path.lexists(tmp_path / "canonical_locks" / (entry["run_id"] + ".lock"))
    assert not os.path.lexists(tmp_path / "queue_status" / (entry["run_id"] + ".json"))
    assert not os.path.lexists(tmp_path / entry["run_id"])


def _deferral_core():
    return {
        "schema": v15.DEFER_SCHEMA, "outcome": "RESOURCE_DEFERRED_PRECLAIM",
        "exit_code": 75, "boundary": "SERIALIZED_PRECLAIM_MEMORY_CHECK",
        "canonical_claim": None, "claimed_run_id": None,
        "canonical_mutation_started": False, "attempt_created": False,
        "checkpoint_loaded": False,
        "telemetry": {"index": 1, "uuid": "GPU-1", "readable": True,
                       "eligible": False, "memory_total_bytes": 1000,
                       "memory_used_bytes": 900, "memory_used_fraction": 0.9,
                       "observation_started_monotonic": 10.0,
                       "observed_at_utc": "2026-10-13T00:00:00Z"},
        "worker_id": "1", "physical_gpu_index": 1, "gpu_uuid": "GPU-1",
        "frozen_target_pair": {"index": 1, "uuid": "GPU-1"},
        "supervisor_pid": 100, "supervisor_start_ticks": 101,
        "child_pid": 200, "child_start_ticks": 201, "invocation": 1,
        "nonce": "test-invocation-nonce", "expected_receipt_path": "/tmp/receipt.json",
        "execution_gate_sha256": "a" * 64, "release_sha256": "b" * 64,
        "queue_sha256": "c" * 64, "source_incident_hold_sha256": "d" * 64,
        "source_failure_nonce": "e" * 32,
        "decision_utc": "2026-10-13T00:00:01Z", "decision_monotonic": 10.1,
        "forbidden_flags": {"resume": False, "retry": False, "new_attempt": False,
                            "checkpoint_load": False, "formal": False, "test": False},
    }


def test_deferral_receipt_hmac_binds_complete_core_and_rejects_tampering():
    secret = b"x" * 32
    receipt = v15.create_deferred_receipt(secret, _deferral_core())
    v15.verify_deferred_receipt(secret, receipt, expected={"nonce": "test-invocation-nonce",
                                                           "physical_gpu_index": 1})
    changed = dict(receipt, gpu_uuid="GPU-0")
    with pytest.raises(PermissionError, match="HMAC"):
        v15.verify_deferred_receipt(secret, changed, expected={})
    with pytest.raises(PermissionError, match="eligible telemetry"):
        v15.create_deferred_receipt(secret, {
            **_deferral_core(), "telemetry": {"index": 1, "uuid": "GPU-1", "readable": True,
                "eligible": True, "memory_total_bytes": 1000, "memory_used_bytes": 899,
                "observation_started_monotonic": 10.0,
                "observed_at_utc": "2026-10-13T00:00:00Z"}})
    stale = _deferral_core()
    stale["decision_monotonic"] = 80.1
    with pytest.raises(PermissionError, match="stale"):
        v15.create_deferred_receipt(secret, stale)
    inconsistent = _deferral_core()
    inconsistent["telemetry"] = {**inconsistent["telemetry"], "memory_used_fraction": 0.1}
    with pytest.raises(PermissionError, match="fraction"):
        v15.create_deferred_receipt(secret, inconsistent)


def test_deferral_receipt_uses_exclusive_durable_publication(tmp_path):
    path = tmp_path / "nonce.receipt.json"
    receipt = v15.write_deferred_receipt(path, b"s" * 32, _deferral_core())
    assert json.loads(path.read_text()) == receipt
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(PermissionError, match="already exists"):
        v15.write_deferred_receipt(path, b"s" * 32, _deferral_core())


def test_authenticated_live_peer_claim_is_accepted_during_deferral(tmp_path):
    queue = v15._read_json(v15.ROOT / v15.QUEUE_REL)
    f140 = set(v15.frozen_f140_ids())
    entry = next(row for row in queue["entries"] if row["run_id"] in f140)
    rid = entry["run_id"]
    root = tmp_path / "stage_F0_outputs" / "runs"
    root.mkdir(parents=True)
    gate_path = root.parent / "execution_isolation_gate.json"
    gate_path.write_text(json.dumps({"gpu_target_pairs": [{"index": 1, "uuid": "GPU-1"}]}))
    target = {"run_id": rid, "run_id_full_sha256": entry["run_id_full_sha256"],
        "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
        "worker_id": "1", "gpu_index": 1, "gpu_uuid": "GPU-1"}
    canonical_path = v15.fresh_member_paths(root, rid)["lock"]
    worker_path = root / "worker_locks/worker_1.lock"
    gpu_path = root / "gpu_locks/gpu1.lock"
    for path in (canonical_path, gpu_path, worker_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    control_path = root / "worker_locks/worker_1.control"
    supervisor_meta = tmp_path / "supervisor.json"
    ready = tmp_path / "ready"
    supervisor_script = """
import fcntl, json, os, subprocess, sys
sys.path.insert(0, os.environ['F0_TEST_RUN_SCRIPTS'])
sys.path.insert(0, os.environ['F0_TEST_EXPERIMENTS'])
from stage_F0_worker_control import linux_process_identity
worker_path, control_path, canonical_path, gpu_path, ready, meta_path, log_path = sys.argv[1:8]
assert os.getpgrp() == os.getpid()
supervisor = linux_process_identity(os.getpid())
host = __import__('socket').gethostname()
worker_nonce = 'different-supervisor-nonce'
worker_fd = os.open(worker_path, os.O_RDWR)
fcntl.flock(worker_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
worker_owner = {'pid': os.getpid(), 'pgid': os.getpgrp(),
    'start_ticks': supervisor['start_ticks'], 'host': host, 'worker_id': '1',
    'nonce': worker_nonce, 'slot_indexes': [0, 1]}
os.ftruncate(worker_fd, 0)
os.write(worker_fd, (json.dumps(worker_owner) + '\\n').encode())
with open(control_path, 'w') as handle:
    handle.write('{} {} {} {} {} 1\\n'.format(os.getpid(), os.getpgrp(),
        supervisor['start_ticks'], worker_nonce, host))
child_script = "import fcntl,os,sys,time; paths=sys.argv[1:3]; fds=[os.open(p,os.O_RDWR) for p in paths]; [fcntl.flock(fd,fcntl.LOCK_EX) for fd in fds]; open(sys.argv[3],'w').write('ready'); time.sleep(60)"
child = subprocess.Popen([sys.executable, '-c', child_script,
    canonical_path, gpu_path, ready], close_fds=True)
child_identity = linux_process_identity(child.pid)
with open(meta_path, 'w') as handle:
    json.dump({'supervisor_pid': os.getpid(), 'supervisor_pgid': os.getpgrp(),
        'supervisor_start_ticks': supervisor['start_ticks'], 'host': host,
        'worker_nonce': worker_nonce, 'child_pid': child.pid,
        'child_start_ticks': child_identity['start_ticks']}, handle)
with open(log_path, 'a') as handle:
    handle.write(json.dumps({'event': 'CHILD_STARTED', 'worker_id': '1',
        'nonce': 'peer-nonce', 'invocation': 1, 'pid': os.getpid(),
        'host': host, 'pgid': os.getpgrp(),
        'supervisor_start_ticks': supervisor['start_ticks'],
        'child_pid': child.pid,
        'child_start_ticks': child_identity['start_ticks']}) + '\\n')
child.wait()
"""
    logs = root / "worker_logs"
    logs.mkdir()
    log = logs / "audited_worker_1.jsonl"
    supervisor = subprocess.Popen([sys.executable, "-c", supervisor_script,
        str(worker_path), str(control_path), str(canonical_path), str(gpu_path),
        str(ready), str(supervisor_meta), str(log)], env={**os.environ,
            "F0_TEST_RUN_SCRIPTS": str(RUN_SCRIPTS),
            "F0_TEST_EXPERIMENTS": str(EXPERIMENTS)}, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while (not ready.exists() or not supervisor_meta.exists()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        supervisor_data = json.loads(supervisor_meta.read_text(encoding="utf-8"))
        from stage_F0_worker_control import linux_process_identity
        child_identity = linux_process_identity(supervisor_data["child_pid"])
        peer_host = supervisor_data["host"]
        worker_nonce = supervisor_data["worker_nonce"]
        identity = {**target,
            "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
            "status": "CLAIMED", "pid": supervisor_data["child_pid"],
            "start_ticks": child_identity["start_ticks"], "nonce": "peer-nonce",
            "recovery_count": 0, "resume_checkpoint": None}
        paths = v15.fresh_member_paths(root, rid)
        paths["status"].parent.mkdir(parents=True)
        paths["status"].write_text(json.dumps(identity))
        paths["lock"].write_text(json.dumps({key: identity[key] for key in (
            "run_id", "run_id_full_sha256", "canonical_key_json", "attempt_id",
            "pid", "start_ticks", "nonce")}))
        worker_owner = json.loads(worker_path.read_text(encoding="utf-8"))
        gpu_owner = {"pid": supervisor_data["child_pid"], "host": peer_host,
            "gpu_index": 1, "gpu_uuid": "GPU-1",
            "claimed_unix": time.time()}
        gpu_path.write_text(json.dumps(gpu_owner))
        assert "host" not in identity
        assert v15._authenticated_peer_progress(root, rid, identity, entry) is True
        identity["host"] = "wrong-status-host"
        paths["status"].write_text(json.dumps(identity))
        assert v15._authenticated_peer_progress(root, rid, identity, entry) is False
        identity.pop("host")
        paths["status"].write_text(json.dumps(identity))
        gpu_path.write_text(json.dumps({**gpu_owner, "host": "wrong-peer-host"}))
        assert v15._authenticated_peer_progress(root, rid, identity, entry) is False
        gpu_path.write_text(json.dumps(gpu_owner))
        before = {}
        # A distinct live child and its supervisor identity authenticate the
        # peer claim; fictitious PIDs or a text-only event are rejected.
        v15.verify_deferred_no_mutation(root, [entry], before=before,
            child_pid=os.getpid() + 100000, child_start_ticks=child_identity["start_ticks"] + 1,
            nonce="deferring-nonce")
        worker_path.write_text(json.dumps({**worker_owner, "nonce": "forged-supervisor-owner"}))
        with pytest.raises(PermissionError, match="authenticated live peer"):
            v15.verify_deferred_no_mutation(root, [entry], before={},
                child_pid=os.getpid() + 100000, child_start_ticks=child_identity["start_ticks"] + 1,
                nonce="deferring-nonce")
        worker_path.write_text(json.dumps(worker_owner))
        log.unlink()
        with pytest.raises(PermissionError, match="authenticated live peer"):
            v15.verify_deferred_no_mutation(root, [entry], before={},
                child_pid=os.getpid() + 100000, child_start_ticks=child_identity["start_ticks"] + 1,
                nonce="deferring-nonce")
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGTERM)
        try:
            supervisor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=5)


def test_deferred_race_rejects_forged_peer_completion_without_real_receipt(tmp_path):
    queue = v15._read_json(v15.ROOT / v15.QUEUE_REL)
    f140 = set(v15.frozen_f140_ids())
    entry = next(row for row in queue["entries"] if row["run_id"] in f140)
    rid = entry["run_id"]
    paths = v15.fresh_member_paths(tmp_path, rid)
    paths["status"].parent.mkdir(parents=True)
    paths["lock"].parent.mkdir(parents=True)
    status = {"run_id": rid, "run_id_full_sha256": entry["run_id_full_sha256"],
        "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
        "status": "CLAIMED", "pid": 23456, "start_ticks": 777,
        "nonce": "peer-complete-nonce", "recovery_count": 0, "resume_checkpoint": None}
    paths["status"].write_text(json.dumps(status))
    owner_fields = ("run_id", "run_id_full_sha256", "canonical_key_json", "attempt_id",
                    "pid", "start_ticks", "nonce")
    paths["lock"].write_text(json.dumps({key: status[key] for key in owner_fields}))
    before = v15.capture_f140_state(tmp_path, [entry])
    status["status"] = "VALIDATION_COMPLETE"
    paths["status"].write_text(json.dumps(status))
    attempt = tmp_path / rid / "attempts" / "attempt-00"
    attempt.mkdir(parents=True)
    projection = {key: status[key] for key in ("run_id", "run_id_full_sha256",
        "canonical_key_json", "attempt_id", "status")}
    for name in ("run_manifest.json", "trajectory_summary.json"):
        (attempt / name).write_text(json.dumps(projection))
    logs = tmp_path / "worker_logs"
    logs.mkdir(parents=True)
    (logs / "audited_worker_1.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"event": "CHILD_STARTED", "worker_id": "1", "nonce": status["nonce"],
         "invocation": 3, "child_pid": status["pid"],
         "child_start_ticks": status["start_ticks"]},
        {"event": "CHILD_VALIDATION_COMPLETE", "worker_id": "1",
         "nonce": status["nonce"], "invocation": 3, "child_pid": status["pid"],
         "run_id": rid, "receipt_sha256": "a" * 64})) + "\n")
    with pytest.raises(PermissionError, match="authenticated live peer"):
        v15.verify_deferred_no_mutation(tmp_path, [entry], before=before,
            child_pid=34567, child_start_ticks=888, nonce="deferred-child")


def test_deferred_race_accepts_completed_peer_only_with_real_validated_receipt(tmp_path):
    from stage_F0_worker_control import linux_process_identity

    source_queue = v15._read_json(v15.ROOT / v15.QUEUE_REL)
    f140 = set(v15.frozen_f140_ids())
    entry = next(row for row in source_queue["entries"] if row["run_id"] in f140)
    rid = entry["run_id"]
    repo = tmp_path / "isolated_repo"
    output = repo / "experiments/stage_F0_outputs"
    root = output / "runs"
    root.mkdir(parents=True)
    queue_path = repo / v15.QUEUE_REL
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps(source_queue))
    (output / "execution_isolation_gate.json").write_text(json.dumps(
        {"gpu_target_pairs": [{"index": 1, "uuid": "GPU-1"}]}))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        child_identity = linux_process_identity(child.pid)
        supervisor_identity = linux_process_identity(os.getpid())
        nonce, worker_id, gpu_index, gpu_uuid = "completed-peer-real-receipt", "1", 1, "GPU-1"
        status = {"run_id": rid, "run_id_full_sha256": entry["run_id_full_sha256"],
            "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
            "worker_id": worker_id, "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
            "status": "CLAIMED", "pid": child.pid,
            "start_ticks": child_identity["start_ticks"], "nonce": nonce,
            "recovery_count": 0, "resume_checkpoint": None}
        paths = v15.fresh_member_paths(root, rid)
        paths["status"].parent.mkdir(parents=True)
        paths["lock"].parent.mkdir(parents=True)
        paths["status"].write_text(json.dumps(status))
        owner = {key: status[key] for key in ("run_id", "run_id_full_sha256",
            "canonical_key_json", "attempt_id", "pid", "start_ticks", "nonce")}
        paths["lock"].write_text(json.dumps(owner))
        before = v15.capture_f140_state(root, [entry])
        attempt = root / rid / "attempts/attempt-00"
        attempt.mkdir(parents=True)
        status.update({"status": "VALIDATION_COMPLETE"})
        projection = {key: status[key] for key in ("run_id", "run_id_full_sha256",
            "canonical_key_json", "attempt_id", "status")}
        manifest_path = attempt / "run_manifest.json"
        manifest_path.write_text(json.dumps(projection, sort_keys=True))
        (attempt / "trajectory_summary.json").write_text(json.dumps(projection))
        checkpoints = {}
        for name in ("final_checkpoint.pt", "selected_checkpoint.pt"):
            checkpoint = attempt / name
            checkpoint.write_bytes(("fixture:" + name).encode())
            checkpoints[name] = v15.sha256_file(checkpoint)
        paths["status"].write_text(json.dumps(status))
        (root / "canonical_locks").mkdir(parents=True, exist_ok=True)
        (root / "canonical_locks" / (rid + ".lock")).write_text(json.dumps({
            **owner, "gpu_uuid": gpu_uuid}))
        gpu_owner = {"pid": child.pid, "host": __import__("socket").gethostname(),
            "start_ticks": child_identity["start_ticks"], "nonce": nonce,
            "run_id": rid, "run_id_full_sha256": entry["run_id_full_sha256"],
            "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
            "worker_id": worker_id, "gpu_index": gpu_index, "gpu_uuid": gpu_uuid}
        gpu_lock = root / "gpu_locks/gpu1.lock"
        gpu_lock.parent.mkdir(parents=True)
        gpu_lock.write_text(json.dumps(gpu_owner))
        (root / "worker_locks/worker_1.lock").parent.mkdir(parents=True)
        (root / "worker_locks/worker_1.lock").write_text(json.dumps({"pid": child.pid,
            "worker_id": worker_id, "nonce": nonce}))
        receipt_path = root / "worker_logs/peer.receipt.json"
        receipt_path.parent.mkdir(parents=True)
        owner_sha = hashlib.sha256(json.dumps(gpu_owner, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        receipt = {"schema": "F0-CHILD-RECEIPT-v1", "status": "VALIDATION_COMPLETE",
            "pid": child.pid, "ppid": os.getpid(), "pgid": os.getpgid(child.pid),
            "worker_id": worker_id, "nonce": nonce, "run_id": rid,
            "run_id_full_sha256": entry["run_id_full_sha256"],
            "canonical_key_json": entry["canonical_key_json"], "attempt_id": "attempt-00",
            "status_path": str(paths["status"]), "manifest_path": str(manifest_path),
            "output_root": str(root), "manifest_sha256": v15.sha256_file(manifest_path),
            "gpu_lock_owner": gpu_owner, "gpu_lock_path": str(gpu_lock),
            "gpu_lock_owner_sha256": owner_sha, "checkpoints": checkpoints}
        receipt_path.write_text(json.dumps(receipt))
        rows = [{"event": "CHILD_STARTED", "worker_id": worker_id, "nonce": nonce,
            "invocation": 3, "pid": os.getpid(),
            "supervisor_start_ticks": supervisor_identity["start_ticks"],
            "pgid": os.getpgid(child.pid), "child_pid": child.pid,
            "child_start_ticks": child_identity["start_ticks"],
            "receipt_path": str(receipt_path)},
            {"event": "CHILD_VALIDATION_COMPLETE", "worker_id": worker_id,
             "nonce": nonce, "invocation": 3, "child_pid": child.pid,
             "run_id": rid, "receipt_sha256": v15.sha256_file(receipt_path)}]
        log_path = root / "worker_logs/audited_worker_1.jsonl"
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        v15.verify_deferred_no_mutation(root, [entry], before=before,
            child_pid=os.getpid() + 100000, child_start_ticks=1, nonce="deferred-child")
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_truncated_append_only_log_fails_closed_without_repair(tmp_path):
    log = tmp_path / "append.jsonl"
    original = b'{"event":"PREPARED"}\n{"event":"COMMIT'
    log.write_bytes(original)
    with pytest.raises(PermissionError, match="truncated"):
        v15.validate_observation_ledger(log)
    assert log.read_bytes() == original


def test_terminal_durability_fence_runs_inside_existing_lock_without_relock(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text("{}\n")
    runs = tmp_path / "runs"
    with v15.execution_state_lock(runs):
        v15._durability_fence(gate, lock_already_held=True)
    assert gate.read_text() == "{}\n"


def test_runtime_observation_log_does_not_change_release_ledger_digest(tmp_path):
    control = tmp_path / "execution_control"
    control.mkdir()
    ledger = control / "postbootstrap_v15_release.jsonl"
    ledger.write_text(json.dumps({"event": "COMMITTED"}) + "\n")
    before = v15.sha256_file(ledger)
    event_sha = v15.record_runtime_observation(
        tmp_path, transaction_id="tx", stage="LOCKED_RELEASE_DECISION",
        target={"index": 1, "uuid": "GPU-1"},
        telemetry={"index": 1, "uuid": "GPU-1", "readable": True,
                   "eligible": True, "memory_total_bytes": 1000, "memory_used_bytes": 200,
                   "memory_used_fraction": 0.2,
                   "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "observation_started_monotonic": time.monotonic()},
        go=True, source_snapshot_sha256="a" * 64,
        prepared_identity_sha256="b" * 64,
        prelaunch_audit_sha256="c" * 64, reason="ASSIGNED_GPU_BELOW_90_PERCENT")
    assert len(event_sha) == 64
    assert v15.sha256_file(ledger) == before
    assert len(v15.validate_observation_ledger(control / "postbootstrap_v15_runtime_observations.jsonl")) == 1


@pytest.mark.parametrize("mutation", [
    {"go": False}, {"source_snapshot_sha256": "z" * 64},
    {"prelaunch_audit_sha256": "0" * 63}, {"reason": "UNBOUND"},
    {"reason": "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT"},
    {"stage": "FOREIGN_BOUNDARY"},
    {"telemetry": {"memory_used_bytes": 900}},
    {"telemetry": {"memory_used_fraction": 0.9}},
    {"monotonic": float("nan")},
])
def test_runtime_observation_full_schema_rejects_inconsistent_decisions(tmp_path, mutation):
    path = tmp_path / "observation.jsonl"
    row = {"schema": "F0-V15-RUNTIME-OBSERVATION-v2", "transaction_id": "tx",
        "stage": "LOCKED_RELEASE_DECISION", "target": {"index": 1, "uuid": "GPU-1"},
        "telemetry": {"index": 1, "uuid": "GPU-1", "readable": True,
            "eligible": True, "memory_total_bytes": 1000, "memory_used_bytes": 200,
            "observed_at_utc": "2026-10-13T00:00:00+00:00",
            "observation_started_monotonic": 10.0},
        "go": True, "source_snapshot_sha256": "a" * 64,
        "prepared_identity_sha256": "b" * 64,
        "prepared_identity_kind": "CANONICAL_PREPARED_REQUEST",
        "prelaunch_audit_sha256": "c" * 64,
        "reason": "ASSIGNED_GPU_BELOW_90_PERCENT",
        "utc": "2026-10-13T00:00:00+00:00", "monotonic": 11.0}
    for key, value in mutation.items():
        if key == "telemetry":
            row["telemetry"].update(value)
        else:
            row[key] = value
    path.write_text(json.dumps(row, allow_nan=True) + "\n")
    with pytest.raises(PermissionError, match="invalid identity|GO differs|reason/GO|fraction"):
        v15.validate_observation_ledger(path)


def test_observation_history_binds_one_current_prepared_graph(tmp_path):
    path = tmp_path / "observation.jsonl"
    rows = []
    for tx, source, prelaunch, prepared in (
            ("tx", "a" * 64, "c" * 64, "b" * 64),
            ("foreign-tx", "a" * 64, "c" * 64, "b" * 64)):
        rows.append({"schema": "F0-V15-RUNTIME-OBSERVATION-v2",
            "transaction_id": tx, "stage": "LOCKED_RELEASE_DECISION",
            "target": {"index": 1, "uuid": "GPU-1"},
            "telemetry": {"index": 1, "uuid": "GPU-1", "readable": True,
                "eligible": True, "memory_total_bytes": 1000, "memory_used_bytes": 200,
                "memory_used_fraction": 0.2, "observed_at_utc": "2026-10-13T00:00:00+00:00",
                "observation_started_monotonic": 10.0},
            "go": True, "source_snapshot_sha256": source,
            "prepared_identity_sha256": prepared,
            "prepared_identity_kind": "CANONICAL_PREPARED_REQUEST",
            "prelaunch_audit_sha256": prelaunch,
            "reason": "ASSIGNED_GPU_BELOW_90_PERCENT",
            "utc": "2026-10-13T00:00:01+00:00", "monotonic": 11.0})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = path.read_bytes()
    with pytest.raises(PermissionError, match="different immutable transaction graphs"):
        v15.validate_observation_ledger(path)
    assert path.read_bytes() == original
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(PermissionError, match="current PREPARED source graph"):
        v15.validate_observation_ledger(path, expected_identity={
            "transaction_id": "tx", "source_snapshot_sha256": "d" * 64,
            "prelaunch_audit_sha256": "c" * 64,
            "prepared_identity_sha256": "b" * 64})
    assert path.read_bytes() == json.dumps(rows[0]).encode() + b"\n"


def test_release_resource_lock_set_covers_non_target_worker_and_gpu(tmp_path):
    runs = tmp_path / "runs"
    lock = runs / "worker_locks/worker_1.lock"
    lock.parent.mkdir(parents=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(PermissionError, match="resource is currently locked"):
            with v15._admission_resource_locks(runs):
                pass
    finally:
        os.close(fd)


def _copy_frozen_f140_identity_files(repo):
    for rel in (v15.QUEUE_REL,
                "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json"):
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v15.ROOT / rel, target)


def test_release_requires_untouched_f140_and_current_disk_slots(tmp_path, monkeypatch):
    repo = tmp_path / "clone"
    _copy_frozen_f140_identity_files(repo)
    runs = repo / "experiments/stage_F0_outputs/runs"
    runs.mkdir(parents=True, exist_ok=True)
    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    monkeypatch.setenv("F0_PROCESS_SCHEDULER_ROOT", str(scheduler))
    for name in ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                 "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE"):
        monkeypatch.delenv(name, raising=False)
    assert v15._assert_current_release_resources(runs, repo)["f140_untouched_count"] == 140
    rid = v15.frozen_f140_ids(repo)[0]
    paths = v15.fresh_member_paths(runs, rid)
    paths["status"].parent.mkdir(parents=True)
    paths["status"].write_text("{}\n")
    with pytest.raises(PermissionError, match="touched F140"):
        v15._assert_current_release_resources(runs, repo)


def _supervisor_deferral_fixture(tmp_path):
    from stage_F0_postabnormal_v14 import QUEUE_SHA256

    repo = tmp_path / "isolated_repo"
    runs = repo / "experiments/stage_F0_outputs/runs"
    logs = runs / "worker_logs"
    control = runs / "execution_control"
    logs.mkdir(parents=True)
    control.mkdir()
    gate = repo / "experiments/stage_F0_outputs/execution_isolation_gate.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps({"schema": v15.SCHEMA,
        "gpu_target_pairs": [{"index": 1, "uuid": "GPU-1"}]}) + "\n")
    release = control / "global_hold_release.json"
    release.write_text("{\"release\":\"fixture\"}\n")
    queue_path = repo / v15.QUEUE_REL
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v15.ROOT / v15.QUEUE_REL, queue_path)
    snapshot_path = repo / "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json"
    shutil.copy2(v15.ROOT / "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_source_snapshot.json",
                 snapshot_path)
    queue_doc = json.loads(queue_path.read_text())
    phase = logs / "invocation.phases.jsonl"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    from stage_F0_worker_control import linux_process_identity
    child_identity = linux_process_identity(child.pid)
    supervisor_identity = linux_process_identity(os.getpid())
    receipt_path = logs / "invocation.receipt.json"
    secret = v15.new_deferral_secret()
    core = _deferral_core()
    core.update({"worker_id": "1", "physical_gpu_index": 1,
        "supervisor_pid": os.getpid(),
        "supervisor_start_ticks": supervisor_identity["start_ticks"],
        "child_pid": child.pid, "child_start_ticks": child_identity["start_ticks"], "invocation": 1,
        "nonce": "fixture-invocation-nonce", "expected_receipt_path": str(receipt_path.resolve()),
        "execution_gate_sha256": v15.sha256_file(gate),
        "release_sha256": v15.sha256_file(release), "queue_sha256": QUEUE_SHA256,
        "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
        "source_failure_nonce": v15.INCIDENT_NONCE,
        "decision_monotonic": __import__("time").monotonic(),
        "telemetry": {"index": 1, "uuid": "GPU-1", "readable": True,
            "eligible": False, "memory_total_bytes": 1000, "memory_used_bytes": 900,
            "memory_used_fraction": 0.9,
            "observation_started_monotonic": __import__("time").monotonic(),
            "observed_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat()}})
    core["decision_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()
    core["decision_monotonic"] = __import__("time").monotonic()
    receipt = v15.write_deferred_receipt(receipt_path, secret, core)
    phase_events = []
    for name, extra in (
            ("child_preclaim_entry", {"receipt_path": str(receipt_path.resolve())}),
            ("serialized_preclaim_memory_check", {"gpu_index": 1, "gpu_uuid": "GPU-1"}),
            ("resource_deferred_receipt_durable", {"receipt_sha256": v15.sha256_file(receipt_path),
                                                    "exit_code": 75})):
        phase_events.append({"phase": name, "worker_id": "1", "nonce": core["nonce"],
            "pid": child.pid, "ppid": os.getpid(), "pgid": child.pid,
            "timestamp": "fixture", "unix_time": time.time(), **extra})
    phase.write_text("".join(json.dumps(row) + "\n" for row in phase_events))
    supervisor = object.__new__(claim_worker.TrajectorySupervisor)
    supervisor.worker_id = "1"
    supervisor.output_root = runs
    supervisor.queue_path = queue_path
    supervisor.execution_gate_path = gate
    outcome = {"receipt": receipt, "return_code": 75, "receipt_path": receipt_path,
        "child_pid": child.pid, "child_start_ticks": child_identity["start_ticks"], "invocation": 1,
        "nonce": "fixture-invocation-nonce", "phase_path": phase,
        "deferral_secret": secret, "pre_spawn_f140_state": {}, "_fixture_child": child}
    return supervisor, outcome, {"pid": 100, "start_ticks": 101}, {
        "pid": child.pid, "start_ticks": child_identity["start_ticks"]}, queue_doc


def test_real_supervisor_accepts_complete_authenticated_preclaim_receipt(tmp_path):
    supervisor, outcome, identity, child_identity, _queue = _supervisor_deferral_fixture(tmp_path)
    try:
        identity = {"pid": os.getpid(), "start_ticks": __import__(
            "stage_F0_worker_control").linux_process_identity(os.getpid())["start_ticks"]}
        accepted = supervisor._validate_resource_deferral(outcome, identity=identity,
                                                           child_identity=child_identity)
        assert accepted["outcome"] == "RESOURCE_DEFERRED_PRECLAIM"
    finally:
        outcome["_fixture_child"].terminate()
        outcome["_fixture_child"].wait(timeout=5)


@pytest.mark.parametrize("mutation", ["missing", "tamper", "wrong_nonce", "wrong_exit"])
def test_real_supervisor_rejects_exit75_spoofs_and_replay(tmp_path, mutation):
    supervisor, outcome, identity, child_identity, _queue = _supervisor_deferral_fixture(tmp_path)
    identity = {"pid": os.getpid(), "start_ticks": __import__(
        "stage_F0_worker_control").linux_process_identity(os.getpid())["start_ticks"]}
    if mutation == "missing":
        outcome["receipt"] = None
    elif mutation == "tamper":
        outcome["receipt"] = {**outcome["receipt"], "gpu_uuid": "GPU-spoof"}
    elif mutation == "wrong_nonce":
        outcome["nonce"] = "different-invocation-nonce"
    else:
        outcome["return_code"] = 1
    with pytest.raises(PermissionError):
        supervisor._validate_resource_deferral(outcome, identity=identity,
                                                child_identity=child_identity)
    outcome["_fixture_child"].terminate()
    outcome["_fixture_child"].wait(timeout=5)


def _hardlink_tree(source: Path, destination: Path):
    shutil.copytree(source, destination, copy_function=os.link)


def _detach_clone_worker_logs(repo: Path) -> dict[str, str]:
    """Copy-on-write worker logs and locks before exercising a clone."""
    source_dir = v15.RUNS / "worker_logs"
    clone_dir = repo / "experiments/stage_F0_outputs/runs/worker_logs"
    source_hashes: dict[str, str] = {}
    clone_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if source.is_symlink() or not source.is_file():
            continue
        clone = clone_dir / source.name
        if not clone.exists():
            continue
        source_hashes[source.name] = v15.sha256_file(source)
        if os.path.samestat(source.stat(), clone.stat()):
            temporary = clone.with_name(clone.name + ".cow." + str(os.getpid()))
            with temporary.open("wb") as handle:
                handle.write(clone.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, clone.stat().st_mode & 0o777)
            os.replace(temporary, clone)
        if os.path.samestat(source.stat(), clone.stat()):
            raise AssertionError("clone worker log still shares production inode: " + source.name)
    _detach_clone_mutable_locks(repo)
    return source_hashes


def _preserve_and_reset_clone_peer_log(source_log: Path, clone_log: Path,
                                       evidence_root: Path) -> dict:
    """Archive frozen clone bytes, then atomically freshen only its peer log."""
    source_before = source_log.read_bytes()
    clone_before = clone_log.read_bytes()
    source_stat = source_log.stat()
    clone_stat = clone_log.stat()
    if os.path.samestat(source_stat, clone_stat):
        raise AssertionError("peer clone log still aliases outer source inode")
    if source_before != clone_before:
        raise AssertionError("detached clone peer log bytes differ from its frozen source")

    evidence_root.mkdir(parents=True, exist_ok=True)
    archived_bytes = evidence_root / "audited_worker_1.pretest.jsonl"
    archived_bytes.write_bytes(clone_before)
    if archived_bytes.read_bytes() != clone_before:
        raise AssertionError("external peer log archive does not preserve original bytes")

    temporary = clone_log.with_name(clone_log.name + ".fresh." + str(os.getpid()))
    old_mode = stat.S_IMODE(clone_stat.st_mode)
    try:
        with temporary.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, old_mode)
        os.replace(temporary, clone_log)
        dir_fd = os.open(clone_log.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

    fresh_stat = clone_log.stat()
    if (not stat.S_ISREG(fresh_stat.st_mode) or clone_log.is_symlink()
            or fresh_stat.st_size != 0
            or os.path.samestat(source_stat, fresh_stat)
            or (fresh_stat.st_dev, fresh_stat.st_ino) == (clone_stat.st_dev, clone_stat.st_ino)):
        raise AssertionError("fresh peer log is not an empty new regular inode")
    if source_log.read_bytes() != source_before:
        raise AssertionError("outer/source peer log changed during clone reset")
    if v15._read_jsonl_exact(clone_log) != []:
        raise AssertionError("strict JSONL reader rejected fresh empty peer clone log")

    metadata = {
        "source_log_path": str(source_log),
        "source_sha256": hashlib.sha256(source_before).hexdigest(),
        "source_size": len(source_before),
        "source_device": source_stat.st_dev,
        "source_inode": source_stat.st_ino,
        "clone_log_path": str(clone_log),
        "clone_original_sha256": hashlib.sha256(clone_before).hexdigest(),
        "clone_original_size": len(clone_before),
        "clone_original_device": clone_stat.st_dev,
        "clone_original_inode": clone_stat.st_ino,
        "clone_fresh_device": fresh_stat.st_dev,
        "clone_fresh_inode": fresh_stat.st_ino,
        "clone_fresh_size": fresh_stat.st_size,
        "archived_original_path": str(archived_bytes),
        "archived_original_sha256": hashlib.sha256(archived_bytes.read_bytes()).hexdigest(),
        "strict_reader_rows": 0,
    }
    (evidence_root / "audited_worker_1.pretest.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


def _detach_clone_mutable_locks(repo: Path) -> dict[str, str]:
    """COW all regular lock files before a hardlink clone acquires any lock."""
    source_runs = v15.RUNS
    clone_runs = repo / "experiments/stage_F0_outputs/runs"
    source_hashes: dict[str, str] = {}
    lock_files = []
    for source in source_runs.rglob("*"):
        if source.is_symlink() or not source.is_file():
            continue
        rel = source.relative_to(source_runs)
        if source.name.endswith(".lock") or rel.as_posix() == "execution_control/execution_state.lock":
            lock_files.append((source, rel))
    for source, rel in lock_files:
        clone = clone_runs / rel
        if not clone.exists() or clone.is_symlink():
            continue
        source_before = v15.sha256_file(source)
        source_hashes[rel.as_posix()] = source_before
        if os.path.samestat(source.stat(), clone.stat()):
            temporary = clone.with_name(clone.name + ".cow-lock." + str(os.getpid()))
            with temporary.open("wb") as handle:
                handle.write(clone.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, clone.stat().st_mode & 0o777)
            os.replace(temporary, clone)
        if os.path.samestat(source.stat(), clone.stat()):
            raise AssertionError("clone lock still shares production inode: " + rel.as_posix())
        if v15.sha256_file(source) != source_before:
            raise AssertionError("production lock bytes changed during clone COW: " + rel.as_posix())
    return source_hashes


def test_clone_worker_logs_are_detached_and_production_bytes_stay_fixed(tmp_path):
    repo = Path(tempfile.mkdtemp(prefix=".v15-cow-test-", dir=v15.ROOT))
    try:
        _hardlink_tree(v15.ROOT / "experiments", repo / "experiments")
        source_hashes = _detach_clone_worker_logs(repo)
        lock_hashes = _detach_clone_mutable_locks(repo)
        source_dir = v15.RUNS / "worker_logs"
        clone_dir = repo / "experiments/stage_F0_outputs/runs/worker_logs"
        assert source_hashes
        assert "execution_control/execution_state.lock" in lock_hashes
        for name, digest in source_hashes.items():
            assert not os.path.samestat((source_dir / name).stat(), (clone_dir / name).stat())
            assert v15.sha256_file(source_dir / name) == digest
        for rel, digest in lock_hashes.items():
            assert not os.path.samestat((v15.RUNS / rel).stat(),
                                        (repo / "experiments/stage_F0_outputs/runs" / rel).stat())
            assert v15.sha256_file(v15.RUNS / rel) == digest
    finally:
        shutil.rmtree(repo)


def test_peer_log_clone_freshening_archives_history_and_changes_only_clone(tmp_path):
    source = tmp_path / "source/audited_worker_1.jsonl"
    clone = tmp_path / "clone/audited_worker_1.jsonl"
    evidence = tmp_path / "evidence"
    source.parent.mkdir(parents=True)
    clone.parent.mkdir(parents=True)
    historical = b"UserWarning: historical non-JSON diagnostic line\n"
    source.write_bytes(historical)
    os.link(source, clone)
    with pytest.raises(AssertionError, match="aliases outer source inode"):
        _preserve_and_reset_clone_peer_log(source, clone, evidence)
    assert source.read_bytes() == historical
    assert clone.read_bytes() == historical

    detached = clone.with_name(clone.name + ".cow")
    detached.write_bytes(clone.read_bytes())
    os.replace(detached, clone)
    original_source_stat = source.stat()
    metadata = _preserve_and_reset_clone_peer_log(source, clone, evidence)
    assert (source.stat().st_dev, source.stat().st_ino) == (
        original_source_stat.st_dev, original_source_stat.st_ino)
    assert source.read_bytes() == historical
    assert (evidence / "audited_worker_1.pretest.jsonl").read_bytes() == historical
    assert metadata["clone_original_sha256"] == hashlib.sha256(historical).hexdigest()
    assert metadata["archived_original_sha256"] == hashlib.sha256(historical).hexdigest()
    assert metadata["strict_reader_rows"] == 0
    assert clone.stat().st_size == 0


def test_frozen_c97_survives_valid_monotonic_f140_completion(tmp_path):
    from stage_F0_worker_control import linux_process_identity

    clone_root = Path(tempfile.mkdtemp(prefix=".v15-c97-fixture-", dir=v15.ROOT))
    try:
        output = clone_root / "stage_F0_outputs"
        runs = output / "runs"
        _hardlink_tree(v15.RUNS, runs)
        (output / "queues").mkdir(parents=True)
        os.link(v15.ROOT / v15.QUEUE_REL, output / "queues/F0-B1-ML-search-queue.json")
        snapshot = v15._read_json(v15.RUNS / "execution_control/postabnormal_v14_source_snapshot.json")
        expected_h51 = v15._capture_post_migration_h51(runs)
        queue = v15._read_json(output / "queues/F0-B1-ML-search-queue.json")
        f140 = set(snapshot["fresh_ids"])
        row = next(item for item in queue["entries"] if item["run_id"] in f140)
        rid = row["run_id"]
        process = linux_process_identity(os.getpid())
        status = {
            "run_id": rid, "run_id_full_sha256": row["run_id_full_sha256"],
            "canonical_key_json": row["canonical_key_json"], "attempt_id": "attempt-00",
            "status": "VALIDATION_COMPLETE", "pid": os.getpid(),
            "start_ticks": process["start_ticks"], "nonce": "completed-f140-fixture",
            "recovery_count": 0, "resume_checkpoint": None,
        }
        status_dir = runs / "queue_status"
        status_dir.mkdir(exist_ok=True)
        (status_dir / (rid + ".json")).write_text(json.dumps(status), encoding="utf-8")
        lock_dir = runs / "canonical_locks"
        lock_dir.mkdir(exist_ok=True)
        (lock_dir / (rid + ".lock")).write_text(json.dumps({
            key: status[key] for key in ("run_id", "run_id_full_sha256", "canonical_key_json",
                                         "attempt_id", "pid", "start_ticks", "nonce")}),
            encoding="utf-8")
        attempt = runs / rid / "attempts" / "attempt-00"
        attempt.mkdir(parents=True)
        for name in ("run_manifest.json", "trajectory_summary.json"):
            (attempt / name).write_text(json.dumps({
                "run_id": rid, "run_id_full_sha256": row["run_id_full_sha256"],
                "canonical_key_json": row["canonical_key_json"],
                "attempt_id": "attempt-00", "status": "VALIDATION_COMPLETE"}),
                encoding="utf-8")

        v15._validate_protected_snapshot_at_root(
            snapshot, runs, allow_live_fresh_progress=True, expected_h51=expected_h51)
        tampered = dict(status, run_id_full_sha256="0" * 64)
        (status_dir / (rid + ".json")).write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(PermissionError, match="status identity"):
            v15._validate_protected_snapshot_at_root(
                snapshot, runs, allow_live_fresh_progress=True, expected_h51=expected_h51)
    finally:
        shutil.rmtree(clone_root)


def _build_valid_pending_graph(repo: Path, *, include_second_target: bool = False):
    from stage_F0_postabnormal_v14 import H51_RUN_IDS, QUEUE_SHA256

    source_doc = v15.build_source_snapshot(root=repo)
    source_rel = "experiments/stage_F0_outputs/runs/execution_control/postbootstrap_v15_source_snapshot.json"
    source_path = repo / source_rel
    v15.durable_json_create(source_path, source_doc)
    source_file_sha = v15.sha256_file(source_path)
    device = v15.query_gpu_memory(gpu_index=0, expected_uuid="GPU-fixture-0",
        provider=lambda index: {"index": index, "uuid": "GPU-fixture-0",
                                "memory_total_bytes": 1000, "memory_used_bytes": 100})
    device["selected"] = True
    if include_second_target:
        second_device = v15.query_gpu_memory(gpu_index=1, expected_uuid="GPU-fixture-1",
            provider=lambda index: {"index": index, "uuid": "GPU-fixture-1",
                                    "memory_total_bytes": 1000, "memory_used_bytes": 100})
        second_device["selected"] = True
        other_device = second_device
    else:
        other_device = {"index": 1, "uuid": None, "readable": False, "eligible": False,
                        "selected": False,
                        "reason": "isolated fixture marks non-target unreadable"}
    started = time.monotonic()
    prelaunch = v15.build_prelaunch_audit(root=repo, nonce="fixture-prelaunch-nonce-001",
        target_rows=[device, other_device], disk_free_bytes=1024, process_count=1,
        observation_started_monotonic=started)
    prelaunch_rel = ("experiments/stage_F0_outputs/runs/execution_control/"
                     "prelaunch_audit_v15_fresh_search_fixture-prelaunch-nonce-001.json")
    prelaunch_path = repo / prelaunch_rel
    v15.durable_json_create(prelaunch_path, prelaunch)
    prelaunch_file_sha = v15.sha256_file(prelaunch_path)
    target_pairs = [{"index": 0, "uuid": "GPU-fixture-0"}]
    if include_second_target:
        target_pairs.append({"index": 1, "uuid": "GPU-fixture-1"})
    prepared = v15.prepare_release_identity(nonce="fixture-release-nonce-0001",
        principal="isolated principal fixture", approval_utc="2026-10-13T00:00:00Z",
        prelaunch_sha256=prelaunch_file_sha,
        source_snapshot_sha256=source_doc["snapshot_sha256"],
        targets=target_pairs, root=repo)
    runtime = v15.runtime_hash_inventory(repo)
    control = v15.control_hash_inventory_v15(repo)
    runtime_sha, control_sha = v15.sha256_json(runtime), v15.sha256_json(control)
    report_rel = "experiments/reports/stage_F0_v15_isolated_fixture_report.json"
    review_rel = "experiments/reports/stage_F0_v15_isolated_fixture_review.json"
    report_doc = {"schema": "isolated fixture report", "no_training": True}
    report_path = repo / report_rel
    v15.durable_json_create(report_path, report_doc)
    report_sha = v15.sha256_file(report_path)
    amendment = {
        "status": "APPROVED", "reviewer_verdict": "PASS",
        "frozen_scientific_code_hash": v15.SCIENTIFIC_SHA256,
        "runtime_control_code_hash": runtime_sha,
        "control_hash_inventory_sha256": control_sha,
        "approval_record_sha256": v15.APPROVAL_RECORD_SHA256,
        "remediation_candidate_sha256": v15.REMEDIATION_CANDIDATE_SHA256,
    }
    gate = {
        "schema": v15.SCHEMA, "launch_scope": v15.SCOPE,
        "launch_authorized": False,
        "authorization": {"resume": False, "retry": False, "new_attempt": False,
            "checkpoint_load": False, "formal": False, "test": False},
        "prior_search_gate_sha256": v15.SEARCH_GATE_SHA256,
        "search_queue_sha256": QUEUE_SHA256,
        "runtime_hashes": runtime, "runtime_hash_inventory_sha256": runtime_sha,
        "control_hashes": control, "control_hash_inventory_sha256": control_sha,
        "source_frozen_artifacts": v15._assert_frozen_artifacts(repo),
        "preserved_held_run_ids": sorted(H51_RUN_IDS),
        "bootstrap_amendment": amendment,
        "design_path": v15.DESIGN_PATH, "design_sha256": v15.DESIGN_SHA256,
        "design_review_path": v15.DESIGN_REVIEW_PATH,
        "design_review_sha256": v15.DESIGN_REVIEW_SHA256,
        "implementation_report_path": report_rel,
        "implementation_report_sha256": report_sha,
        "implementation_review_path": review_rel,
        "source_snapshot_path": source_rel,
        "source_snapshot_sha256": source_doc["snapshot_sha256"],
        "source_snapshot_file_sha256": source_file_sha,
        "prelaunch_audit_path": prelaunch_rel,
        "prelaunch_audit_sha256": prelaunch_file_sha,
        "prelaunch_target_pairs": target_pairs,
        "gpu_target_pairs": target_pairs,
        "prepared_release_id": prepared["transaction_id"],
    }
    candidate_core_sha = v15.sha256_json(v15._candidate_core_projection(gate))
    review_doc = {
        "reviewer_identity": "isolated fixture reviewer", "verdict": "PASS",
        "design_path": v15.DESIGN_PATH, "design_sha256": v15.DESIGN_SHA256,
        "implementation_report_path": report_rel,
        "implementation_report_sha256": report_sha,
        "approved_candidate_core_sha256": candidate_core_sha,
        "runtime_hashes": runtime, "runtime_hash_inventory_sha256": runtime_sha,
        "control_hashes": control, "control_hash_inventory_sha256": control_sha,
        "changed_source_hashes": control, "unresolved_blockers": [],
        "scientific_preservation_confirmed": True,
    }
    review_path = repo / review_rel
    v15.durable_json_create(review_path, review_doc)
    review_sha = v15.sha256_file(review_path)
    gate["implementation_review_sha256"] = review_sha
    staged_rel = ("experiments/stage_F0_outputs/runs/execution_control/"
                  "staged_go_v15_fixture-release-nonce-0001.json")
    staged_doc = {"schema": "F0-V15-STAGED-GO-v1", "verdict": "GO",
        "production_release_authorized": False,
        "candidate_core_sha256": candidate_core_sha,
        "implementation_review_sha256": review_sha,
        "implementation_report_sha256": report_sha,
        "source_snapshot_sha256": source_doc["snapshot_sha256"],
        "prelaunch_audit_sha256": prelaunch_file_sha,
        "design_sha256": v15.DESIGN_SHA256,
        "reviewer_identity": review_doc["reviewer_identity"]}
    staged_path = repo / staged_rel
    v15.durable_json_create(staged_path, staged_doc)
    gate.update({"approved_candidate_core_sha256": candidate_core_sha,
                 "staged_go_path": staged_rel,
                 "staged_go_sha256": v15.sha256_file(staged_path)})
    return gate, prepared, prelaunch_file_sha, source_doc, device


def test_release_refresh_no_go_and_pre_hold_resource_drift_stay_held(monkeypatch, tmp_path):
    """Real graph validators cover stale inputs, NO-GO and final locked rechecks."""
    import fcntl

    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    repo = Path(tempfile.mkdtemp(prefix=".v15-b1-release-check-", dir=v15.ROOT))
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in (
        "stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
        "stage_F0_protocol", "stage_F0_queue")}
    try:
        _hardlink_tree(v15.ROOT / "experiments", repo / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", repo / "m3oe")
        _detach_clone_worker_logs(repo)
        sys.path.insert(0, str(repo / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        gate, prepared, prelaunch_sha, source_doc, _device = _build_valid_pending_graph(repo)
        runs = repo / "experiments/stage_F0_outputs/runs"
        release = {"schema": v15.RELEASE_SCHEMA, "release_id": prepared["transaction_id"],
            "nonce": prepared["nonce"], "approved_by": prepared["approved_by"],
            "approved_by_role": "principal", "principal_intent": v15.SCOPE,
            "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha, "targets": prepared["targets"],
            "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate))}
        real_query = v15.query_gpu_memory
        used_values = []
        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            used = used_values.pop(0) if used_values else 100
            selected = provider or (lambda index: {"index": index, "uuid": expected_uuid,
                "memory_total_bytes": 1000, "memory_used_bytes": used})
            return real_query(gpu_index=gpu_index, expected_uuid=expected_uuid, provider=selected)
        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        scheduler = tmp_path / "scheduler"
        scheduler.mkdir()
        monkeypatch.setenv("F0_PROCESS_SCHEDULER_ROOT", str(scheduler))
        for key in ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                    "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE"):
            monkeypatch.delenv(key, raising=False)
        stale = {"devices": [], "observation_started_monotonic": time.monotonic() - 61.0}
        used_values[:] = [100]
        with pytest.raises(RuntimeError, match="fault injection after PREPARED"):
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale,
                fault_after="PREPARED")
        hold = runs / "execution_control/global_hold.json"
        active_hold_bytes = hold.read_bytes()
        ledger = runs / "execution_control/postbootstrap_v15_release.jsonl"
        prepared_ledger_bytes = ledger.read_bytes()

        # Partial reentry identity drift fails before it can alter the
        # immutable PREPARED row, release ledger or active hold.
        bad_prepared = dict(prepared)
        bad_prepared["targets"] = [dict(row, uuid="GPU-OTHER")
                                   for row in prepared["targets"]]
        bad_gate = dict(gate)
        bad_gate["gpu_target_pairs"] = bad_prepared["targets"]
        bad_release = dict(release, targets=bad_prepared["targets"],
            pending_gate_sha256=v15.sha256_bytes(v15._canonical_indented_bytes(bad_gate)))
        with pytest.raises(PermissionError):
            v15.commit_principal_release(runs_root=runs, prepared=bad_prepared,
                gate_pending=bad_gate, release_payload=bad_release,
                live_observation=stale)
        assert ledger.read_bytes() == prepared_ledger_bytes
        assert hold.read_bytes() == active_hold_bytes

        source_path = repo / gate["source_snapshot_path"]
        source_bytes = source_path.read_bytes()
        detached_source = source_path.with_name(source_path.name + ".cow-test")
        detached_source.write_bytes(source_bytes)
        os.replace(detached_source, source_path)
        source_path.write_bytes(source_bytes + b" ")
        try:
            with pytest.raises(PermissionError):
                v15.commit_principal_release(runs_root=runs, prepared=prepared,
                    gate_pending=gate, release_payload=release,
                    live_observation=stale)
            assert ledger.read_bytes() == prepared_ledger_bytes
            assert hold.read_bytes() == active_hold_bytes
        finally:
            source_path.write_bytes(source_bytes)
            source_path.chmod(0o444)

        altered_hold = json.loads(active_hold_bytes)
        altered_hold["incident_id"] = "foreign-hold"
        # The initial clone uses hardlinks for immutable protected inputs.
        # Detach this one clone path before the deliberate mutation so the
        # production incident hold can never be written through the link.
        detached_hold = hold.with_name(hold.name + ".cow-test")
        detached_hold.write_bytes(active_hold_bytes)
        os.replace(detached_hold, hold)
        hold.write_text(json.dumps(altered_hold, sort_keys=True) + "\n")
        altered_hold_bytes = hold.read_bytes()
        try:
            with pytest.raises(PermissionError):
                v15.commit_principal_release(runs_root=runs, prepared=prepared,
                    gate_pending=gate, release_payload=release,
                    live_observation=stale)
            assert ledger.read_bytes() == prepared_ledger_bytes
            assert hold.read_bytes() == altered_hold_bytes
        finally:
            hold.write_bytes(active_hold_bytes)
            hold.chmod(0o444)

        used_values[:] = [100, 900]
        with pytest.raises(v15.ResourceDeferred) as deferred:
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale)
        assert deferred.value.boundary == "PRE_HOLD_RELEASE_DECISION"
        assert hold.read_bytes() == active_hold_bytes
        assert json.loads(hold.read_text())["active"] is True
        # A pre-hold NO-GO may have completed transaction steps 1–3, which
        # explicitly permit an immutable release artifact, non-authorizing
        # pending gate and successor pointer. It must not append hold/commit
        # phases or create an admission-capable canonical gate.
        control = runs / "execution_control"
        gate_path = runs.parent / "execution_isolation_gate.json"
        canonical_pending = json.loads(gate_path.read_text())
        assert canonical_pending["launch_authorized"] is False
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [row["event"] for row in events] == ["PREPARED",
            "RELEASE_ARTIFACT_DURABLE", "PENDING_GATE_DURABLE", "RELEASE_POINTER_DURABLE"]
        assert not any(row["event"] in ("HOLD_RELEASED", "COMMITTED") for row in events)
        release_path = control / "global_hold_release_v15.json"
        release_pointer = control / "global_hold_release.json"
        assert release_path.is_file()
        assert release_pointer.read_bytes() == release_path.read_bytes()
        from stage_F0_worker_control import validate_execution_gate
        with pytest.raises(PermissionError):
            validate_execution_gate(root=repo, gate_path=gate_path,
                search_gate_path=repo / v15.SEARCH_GATE_REL)
        observation_path = runs / "execution_control/postbootstrap_v15_runtime_observations.jsonl"
        observations = v15.validate_observation_ledger(observation_path)
        assert [row["go"] for row in observations[-3:]] == [True, True, False]
        assert observations[-1]["reason"] == "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT"

        queue_entries = v15._read_json(repo / v15.QUEUE_REL)["entries"]
        fresh = set(v15.frozen_f140_ids(repo))
        original_assert = v15._assert_current_release_resources
        f140_injected = {"path": None}
        calls = {"count": 0}
        def inject_f140_at_hold_check(runs_root, repo_root):
            calls["count"] += 1
            if calls["count"] == 3:
                path = v15.fresh_member_paths(runs_root, next(iter(fresh)))["status"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                f140_injected["path"] = path
            return original_assert(runs_root, repo_root)
        monkeypatch.setattr(v15, "_assert_current_release_resources", inject_f140_at_hold_check)
        with pytest.raises(PermissionError, match="touched F140"):
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale)
        assert hold.read_bytes() == active_hold_bytes
        f140_injected["path"].unlink()
        monkeypatch.setattr(v15, "_assert_current_release_resources", original_assert)

        slot = scheduler / "slot4.lock"
        slot_state = {"fd": None, "count": 0}
        def inject_slot_at_hold_check(runs_root, repo_root):
            slot_state["count"] += 1
            if slot_state["count"] == 3:
                slot.touch()
                slot_state["fd"] = os.open(slot, os.O_RDWR)
                fcntl.flock(slot_state["fd"], fcntl.LOCK_EX | fcntl.LOCK_NB)
            return original_assert(runs_root, repo_root)
        monkeypatch.setattr(v15, "_assert_current_release_resources", inject_slot_at_hold_check)
        try:
            with pytest.raises(PermissionError, match="another F0 task process"):
                v15.commit_principal_release(runs_root=runs, prepared=prepared,
                    gate_pending=gate, release_payload=release, live_observation=stale)
            assert hold.read_bytes() == active_hold_bytes
        finally:
            if slot_state["fd"] is not None:
                fcntl.flock(slot_state["fd"], fcntl.LOCK_UN)
                os.close(slot_state["fd"])
            if slot.exists():
                slot.unlink()
            monkeypatch.setattr(v15, "_assert_current_release_resources", original_assert)
        final_archive = runs / "execution_control" / (
            "execution_isolation_gate_v15_released_" + prepared["nonce"] + ".json")
        with pytest.raises(RuntimeError, match="fault injection at FINAL_GATE_ARCHIVE"):
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale,
                fault_after="FINAL_GATE_ARCHIVE")
        assert final_archive.is_file()
        assert (runs.parent / "execution_isolation_gate.json").read_bytes() == \
            v15._canonical_indented_bytes(gate)
        assert json.loads((runs.parent / "execution_isolation_gate.json").read_text())[
            "launch_authorized"] is False
        assert json.loads(hold.read_text())["active"] is False
        with pytest.raises(RuntimeError, match=
                           "fault injection at FINAL_GATE_RENAME_BEFORE_DIR_FSYNC"):
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale,
                fault_after="FINAL_GATE_RENAME_BEFORE_DIR_FSYNC")
        assert (runs.parent / "execution_isolation_gate.json").read_bytes() == \
            final_archive.read_bytes()
        assert json.loads((runs.parent / "execution_isolation_gate.json").read_text())[
            "launch_authorized"] is True
        result = v15.commit_principal_release(runs_root=runs, prepared=prepared,
            gate_pending=gate, release_payload=release, live_observation=stale)
        assert result["status"] == "COMMITTED_IDEMPOTENT"
        assert json.loads(hold.read_text())["active"] is False
        assert all(v15.key_is_untouched(runs, row) for row in queue_entries
                   if row["run_id"] in fresh)
    finally:
        shutil.rmtree(repo, ignore_errors=True)
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_each_release_fault_boundary_uses_independent_prepared_clone(monkeypatch, tmp_path):
    """Every publication fault starts from the same PREPARED-only preimage."""
    import stat

    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    base = Path(tempfile.mkdtemp(prefix=".v15-fault-prepared-base-", dir=v15.ROOT))
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in (
        "stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
        "stage_F0_protocol", "stage_F0_queue")}
    try:
        _hardlink_tree(v15.ROOT / "experiments", base / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", base / "m3oe")
        _detach_clone_worker_logs(base)
        sys.path.insert(0, str(base / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        gate, prepared, prelaunch_sha, source_doc, device = _build_valid_pending_graph(base)
        release = {"schema": v15.RELEASE_SCHEMA,
            "release_id": prepared["transaction_id"], "nonce": prepared["nonce"],
            "approved_by": prepared["approved_by"], "approved_by_role": "principal",
            "principal_intent": v15.SCOPE, "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha, "targets": prepared["targets"],
            "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate))}
        real_query = v15.query_gpu_memory
        telemetry_queries = []
        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            telemetry_queries.append((gpu_index, expected_uuid))
            return real_query(gpu_index=gpu_index, expected_uuid=expected_uuid,
                provider=provider or (lambda index: {"index": index, "uuid": expected_uuid,
                    "memory_total_bytes": 1000, "memory_used_bytes": 100}))
        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        base_runs = base / "experiments/stage_F0_outputs/runs"
        stale = {"devices": [], "observation_started_monotonic": -1}
        with pytest.raises(RuntimeError, match="fault injection after PREPARED"):
            v15.commit_principal_release(runs_root=base_runs, prepared=prepared,
                gate_pending=gate, release_payload=release, live_observation=stale,
                fault_after="PREPARED")
        base_ledger = base_runs / v15.RELEASE_LEDGER_REL.split("runs/", 1)[1]
        assert [row["event"] for row in map(json.loads, base_ledger.read_text().splitlines())] == [
            "PREPARED"]

        scheduler = tmp_path / "v15-independent-fault-scheduler"
        scheduler.mkdir()
        monkeypatch.setenv("F0_PROCESS_SCHEDULER_ROOT", str(scheduler))
        for key in ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                    "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE"):
            monkeypatch.delenv(key, raising=False)

        boundaries = [
            "RELEASE_ARTIFACT_RENAME", "RELEASE_ARTIFACT_DURABLE",
            "PENDING_GATE_RENAME", "PENDING_GATE_RENAME_DURABLE", "PENDING_GATE_DURABLE",
            "RELEASE_POINTER_RENAME", "RELEASE_POINTER_DURABLE", "HOLD_RENAME",
            "HOLD_RENAMED_DURABLE", "HOLD_RELEASED", "COMMITTED", "FINAL_GATE_ARCHIVE",
            "FINAL_GATE_RENAME_BEFORE_DIR_FSYNC", "FINAL_GATE_DURABLE",
        ]
        for ordinal, boundary in enumerate(boundaries):
            case = Path(tempfile.mkdtemp(prefix=".v15-fault-case-{}-".format(ordinal),
                                         dir=v15.ROOT))
            try:
                _hardlink_tree(base / "experiments", case / "experiments")
                _hardlink_tree(base / "m3oe", case / "m3oe")
                case_runs = case / "experiments/stage_F0_outputs/runs"
                _detach_clone_worker_logs(case)
                # The baseline's PREPARED and observation ledgers are
                # append-only; detach them before this case appends.
                for rel in (v15.RELEASE_LEDGER_REL, v15.OBSERVATION_LEDGER_REL):
                    path = case_runs / rel.split("runs/", 1)[1]
                    if path.exists():
                        mode = stat.S_IMODE(path.stat().st_mode)
                        temp = path.with_name(path.name + ".cow-test")
                        temp.write_bytes(path.read_bytes())
                        os.chmod(temp, mode)
                        os.replace(temp, path)

                with pytest.raises(RuntimeError, match="fault injection"):
                    v15.commit_principal_release(runs_root=case_runs, prepared=prepared,
                        gate_pending=gate, release_payload=release,
                        live_observation=stale, fault_after=boundary)

                gate_path = case / v15.CANONICAL_GATE_REL
                hold_path = case_runs / v15.HOLD_REL.split("runs/", 1)[1]
                ledger = case_runs / v15.RELEASE_LEDGER_REL.split("runs/", 1)[1]
                pending_path = case_runs / ("execution_control/execution_isolation_gate_v15_pending_"
                                            + prepared["nonce"] + ".json")
                archive_path = case_runs / ("execution_control/execution_isolation_gate_v15_released_"
                                            + prepared["nonce"] + ".json")
                events = [json.loads(line) for line in ledger.read_text().splitlines()]
                assert events[0]["event"] == "PREPARED"
                # A nonce from another identity must never adopt this
                # transaction's partially published graph. Use a correctly
                # derived transaction ID so this exercises identity isolation,
                # rather than malformed-input rejection.
                alternate_nonce = "foreign-v15-transaction-{:02d}-nonce".format(ordinal)
                alternate_prepared = dict(prepared)
                alternate_prepared["nonce"] = alternate_nonce
                import uuid
                alternate_prepared["transaction_id"] = uuid.uuid5(
                    uuid.NAMESPACE_URL, "f0-v15:" + alternate_nonce).hex
                alternate_gate = dict(gate)
                alternate_gate["prepared_release_id"] = alternate_prepared["transaction_id"]
                alternate_release = dict(release,
                    release_id=alternate_prepared["transaction_id"], nonce=alternate_nonce,
                    pending_gate_sha256=v15.sha256_bytes(
                        v15._canonical_indented_bytes(alternate_gate)))
                identity_paths = [ledger, gate_path,
                    case_runs / v15.HOLD_REL.split("runs/", 1)[1],
                    case_runs / "execution_control/global_hold_release_v15.json",
                    case_runs / ("execution_control/execution_isolation_gate_v15_pending_"
                                 + prepared["nonce"] + ".json"),
                    case_runs / "execution_control/global_hold_release.json",
                    case_runs / ("execution_control/execution_isolation_gate_v15_released_"
                                 + prepared["nonce"] + ".json"),
                    case_runs / v15.OBSERVATION_LEDGER_REL.split("runs/", 1)[1]]
                identity_before = [(path.exists(), path.read_bytes() if path.exists() else None)
                                   for path in identity_paths]
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=case_runs,
                        prepared=alternate_prepared, gate_pending=alternate_gate,
                        release_payload=alternate_release, live_observation=stale)
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in identity_paths] == identity_before
                hold_active = json.loads(hold_path.read_text())["active"]
                assert hold_active is (boundary not in {
                    "HOLD_RENAME", "HOLD_RENAMED_DURABLE", "HOLD_RELEASED", "COMMITTED",
                    "FINAL_GATE_ARCHIVE", "FINAL_GATE_RENAME_BEFORE_DIR_FSYNC",
                    "FINAL_GATE_DURABLE"})
                if boundary in {"PENDING_GATE_RENAME", "PENDING_GATE_RENAME_DURABLE",
                                "PENDING_GATE_DURABLE", "RELEASE_POINTER_RENAME",
                                "RELEASE_POINTER_DURABLE", "HOLD_RENAME",
                                "HOLD_RENAMED_DURABLE", "HOLD_RELEASED", "COMMITTED",
                                "FINAL_GATE_ARCHIVE", "FINAL_GATE_RENAME_BEFORE_DIR_FSYNC",
                                "FINAL_GATE_DURABLE"}:
                    assert pending_path.is_file()
                    assert json.loads(gate_path.read_text())["launch_authorized"] is False or \
                        boundary in {"FINAL_GATE_RENAME_BEFORE_DIR_FSYNC", "FINAL_GATE_DURABLE"}
                if boundary in {"FINAL_GATE_ARCHIVE", "FINAL_GATE_RENAME_BEFORE_DIR_FSYNC",
                                "FINAL_GATE_DURABLE"}:
                    assert archive_path.is_file()
                if boundary == "FINAL_GATE_ARCHIVE":
                    assert gate_path.read_bytes() == v15._canonical_indented_bytes(gate)
                    assert json.loads(gate_path.read_text())["launch_authorized"] is False
                elif boundary in {"FINAL_GATE_RENAME_BEFORE_DIR_FSYNC", "FINAL_GATE_DURABLE"}:
                    assert gate_path.read_bytes() == archive_path.read_bytes()
                    assert json.loads(gate_path.read_text())["launch_authorized"] is True
                else:
                    from stage_F0_worker_control import validate_execution_gate
                    with pytest.raises(PermissionError):
                        validate_execution_gate(root=case, gate_path=gate_path,
                            search_gate_path=case / v15.SEARCH_GATE_REL)

                # Inject a distinct later failure hold at every recovery/ACK
                # point. The API must preserve it and leave transaction bytes
                # untouched; then restore only the clone's preimage to test
                # exact same-identity recovery.
                ledger_before_later_hold = ledger.read_bytes()
                gate_before_later_hold = gate_path.read_bytes()
                hold_before_later_hold = hold_path.read_bytes()
                later_hold = json.loads(hold_before_later_hold)
                later_hold.update({"active": True, "incident_id": "later-foreign-hold",
                                   "failure_nonce": "later-hold-for-" + boundary})
                hold_temp = hold_path.with_name(hold_path.name + ".later-hold")
                hold_temp.write_text(json.dumps(later_hold, sort_keys=True) + "\n")
                os.chmod(hold_temp, 0o444)
                os.replace(hold_temp, hold_path)
                injected_hold_bytes = hold_path.read_bytes()
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=case_runs, prepared=prepared,
                        gate_pending=gate, release_payload=release,
                        live_observation=stale)
                assert hold_path.read_bytes() == injected_hold_bytes
                assert ledger.read_bytes() == ledger_before_later_hold
                assert gate_path.read_bytes() == gate_before_later_hold
                restore = hold_path.with_name(hold_path.name + ".restore")
                restore.write_bytes(hold_before_later_hold)
                os.chmod(restore, 0o444)
                os.replace(restore, hold_path)

                if boundary == "FINAL_GATE_RENAME_BEFORE_DIR_FSYNC":
                    # Simulate the alternate restart where the rename did not
                    # survive: only the clone's exact pending bytes return.
                    pending_bytes = pending_path.read_bytes()
                    v15._atomic_replace_bytes(gate_path, pending_bytes)
                    assert json.loads(gate_path.read_text())["launch_authorized"] is False
                recovered = v15.commit_principal_release(runs_root=case_runs,
                    prepared=prepared, gate_pending=gate, release_payload=release,
                    live_observation=stale)
                assert recovered["status"] in ("COMMITTED", "COMMITTED_IDEMPOTENT")
                terminal = json.loads(gate_path.read_text())
                assert terminal["launch_authorized"] is True
                assert json.loads(hold_path.read_text())["active"] is False
                assert sum(row["event"] == "COMMITTED" for row in
                           map(json.loads, ledger.read_text().splitlines())) == 1

                # A still newer hold blocks even the terminal/recovery
                # opportunity. The active hold and every transaction artifact
                # remain byte-exact; after restoring the clone's matching
                # released hold, same-identity acknowledgement is read-only.
                terminal_hold_bytes = hold_path.read_bytes()
                committed_bytes = ledger.read_bytes()
                observation_path = case_runs / v15.OBSERVATION_LEDGER_REL.split(
                    "runs/", 1)[1]
                observation_bytes = (observation_path.read_bytes()
                                     if observation_path.exists() else None)
                terminal_paths = [gate_path, hold_path,
                    case_runs / "execution_control/global_hold_release_v15.json",
                    pending_path, case_runs / "execution_control/global_hold_release.json",
                    archive_path]
                later_terminal_hold = json.loads(terminal_hold_bytes)
                later_terminal_hold.update({"active": True,
                    "incident_id": "later-terminal-hold-" + boundary,
                    "failure_nonce": "terminal-hold-for-" + boundary})
                later_path = hold_path.with_name(hold_path.name + ".terminal-later-hold")
                later_path.write_text(json.dumps(later_terminal_hold, sort_keys=True) + "\n")
                os.chmod(later_path, 0o444)
                os.replace(later_path, hold_path)
                terminal_later_bytes = hold_path.read_bytes()
                rejection_before = [(path.exists(), path.read_bytes() if path.exists() else None)
                                    for path in terminal_paths]
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=case_runs, prepared=prepared,
                        gate_pending=gate, release_payload=release, live_observation=stale)
                assert hold_path.read_bytes() == terminal_later_bytes
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in terminal_paths] == rejection_before
                assert ledger.read_bytes() == committed_bytes
                assert (observation_path.read_bytes() if observation_path.exists()
                        else None) == observation_bytes
                restore_terminal_hold = hold_path.with_name(hold_path.name + ".terminal-restore")
                restore_terminal_hold.write_bytes(terminal_hold_bytes)
                os.chmod(restore_terminal_hold, 0o444)
                os.replace(restore_terminal_hold, hold_path)
                terminal_before_ack = [(path.exists(), path.read_bytes() if path.exists() else None)
                                       for path in terminal_paths]
                ack_query_count = len(telemetry_queries)
                ack = v15.commit_principal_release(runs_root=case_runs, prepared=prepared,
                    gate_pending=gate, release_payload=release, live_observation=stale)
                assert ack["status"] == "COMMITTED_IDEMPOTENT"
                assert len(telemetry_queries) == ack_query_count
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in terminal_paths] == terminal_before_ack
                assert ledger.read_bytes() == committed_bytes
                assert (observation_path.read_bytes() if observation_path.exists()
                        else None) == observation_bytes
            finally:
                shutil.rmtree(case, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity(monkeypatch, tmp_path):
    """Separate archive file-fsync and parent-directory-fsync crash points."""
    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    base = Path(tempfile.mkdtemp(prefix=".v15-preimage-fsync-base-", dir=v15.ROOT))
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in (
        "stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
        "stage_F0_protocol", "stage_F0_queue")}
    try:
        _hardlink_tree(v15.ROOT / "experiments", base / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", base / "m3oe")
        _detach_clone_worker_logs(base)
        sys.path.insert(0, str(base / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        gate, prepared, prelaunch_sha, source_doc, _device = _build_valid_pending_graph(base)
        release = {"schema": v15.RELEASE_SCHEMA,
            "release_id": prepared["transaction_id"], "nonce": prepared["nonce"],
            "approved_by": prepared["approved_by"], "approved_by_role": "principal",
            "principal_intent": v15.SCOPE, "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha,
            "targets": prepared["targets"], "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate))}
        real_query = v15.query_gpu_memory
        telemetry_queries = []
        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            telemetry_queries.append((gpu_index, expected_uuid))
            return real_query(gpu_index=gpu_index, expected_uuid=expected_uuid,
                provider=provider or (lambda index: {"index": index, "uuid": expected_uuid,
                    "memory_total_bytes": 1000, "memory_used_bytes": 100}))
        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        scheduler = tmp_path / "v15-preimage-fsync-scheduler"
        scheduler.mkdir()
        monkeypatch.setenv("F0_PROCESS_SCHEDULER_ROOT", str(scheduler))
        for key in ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                    "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE"):
            monkeypatch.delenv(key, raising=False)

        preimage_case_ids = [(label, boundary)
            for label in ("gate", "hold", "release_pointer", "prepared_ledger")
            for boundary in ("file_fsync", "parent_directory_fsync")]
        assert len(preimage_case_ids) == 8 and len(set(preimage_case_ids)) == 8
        for label, boundary in preimage_case_ids:
                case = Path(tempfile.mkdtemp(prefix=".v15-preimage-{}-{}-".format(
                    label, boundary), dir=v15.ROOT))
                try:
                    _hardlink_tree(base / "experiments", case / "experiments")
                    _hardlink_tree(base / "m3oe", case / "m3oe")
                    _detach_clone_worker_logs(case)
                    case_runs = case / "experiments/stage_F0_outputs/runs"
                    control_dir = case_runs / "execution_control"
                    target = ("postbootstrap_v15_release.jsonl" if label == "prepared_ledger"
                              else "v15_preimage_{}_".format(label))
                    source_path = ({
                        "gate": case / v15.CANONICAL_GATE_REL,
                        "hold": case_runs / v15.HOLD_REL.split("runs/", 1)[1],
                        "release_pointer": control_dir / "global_hold_release.json",
                    }.get(label))
                    source_bytes = source_path.read_bytes() if source_path is not None else None
                    gate_path = case / v15.CANONICAL_GATE_REL
                    hold_path = case_runs / v15.HOLD_REL.split("runs/", 1)[1]
                    pointer_path = control_dir / "global_hold_release.json"
                    source_gate_bytes = gate_path.read_bytes()
                    source_hold_bytes = hold_path.read_bytes()
                    source_pointer_bytes = pointer_path.read_bytes()
                    source_snapshot_path = case / gate["source_snapshot_path"]
                    prelaunch_path = case / gate["prelaunch_audit_path"]
                    source_snapshot_bytes = source_snapshot_path.read_bytes()
                    prelaunch_bytes = prelaunch_path.read_bytes()
                    state = {"target_file_fsynced": False, "injected": False,
                             "target_archive": None, "matches": []}
                    real_fsync = os.fsync

                    def inject_after_archive_fsync(fd):
                        try:
                            fd_path = Path(os.readlink("/proc/self/fd/{}".format(fd)))
                            mode = os.fstat(fd).st_mode
                        except (OSError, ValueError):
                            fd_path, mode = Path(""), 0
                        result = real_fsync(fd)
                        is_target_file = target in str(fd_path)
                        if is_target_file and stat.S_ISREG(mode):
                            state["target_file_fsynced"] = True
                            state["target_archive"] = fd_path
                            state["matches"].append({"stage": "file_fsync",
                                "path": str(fd_path), "fd_kind": "regular"})
                            if boundary == "file_fsync" and not state["injected"]:
                                state["injected"] = True
                                raise RuntimeError("injected after {} file fsync".format(label))
                        elif (boundary == "parent_directory_fsync"
                              and state["target_file_fsynced"] and not state["injected"]
                              and stat.S_ISDIR(mode) and fd_path.resolve() == control_dir.resolve()):
                            state["matches"].append({"stage": "parent_directory_fsync",
                                "path": str(fd_path.resolve()), "fd_kind": "directory"})
                            state["injected"] = True
                            raise RuntimeError("injected after {} parent-directory fsync".format(label))
                        return result

                    with monkeypatch.context() as scoped:
                        scoped.setattr(os, "fsync", inject_after_archive_fsync)
                        with pytest.raises(RuntimeError, match="injected after"):
                            v15.commit_principal_release(runs_root=case_runs,
                                prepared=prepared, gate_pending=gate,
                                release_payload=release,
                                live_observation={"devices": [],
                                    "observation_started_monotonic": -1})
                    assert state["injected"] is True
                    assert state["target_file_fsynced"] is True
                    ledger = control_dir / "postbootstrap_v15_release.jsonl"
                    expected_target_path = (ledger if label == "prepared_ledger" else
                        control_dir / (target + hashlib.sha256(source_bytes).hexdigest() + ".bin"))
                    assert state["target_archive"] == expected_target_path
                    assert [row["stage"] for row in state["matches"]] == (
                        ["file_fsync"] if boundary == "file_fsync" else
                        ["file_fsync", "parent_directory_fsync"])
                    assert state["matches"][0]["fd_kind"] == "regular"
                    assert Path(state["matches"][0]["path"]) == expected_target_path
                    if boundary == "parent_directory_fsync":
                        assert state["matches"][1] == {"stage": "parent_directory_fsync",
                            "path": str(control_dir.resolve()), "fd_kind": "directory"}
                    assert json.loads(hold_path.read_text())["active"] is True
                    assert gate_path.read_bytes() == source_gate_bytes
                    assert hold_path.read_bytes() == source_hold_bytes
                    assert pointer_path.read_bytes() == source_pointer_bytes
                    assert source_snapshot_path.read_bytes() == source_snapshot_bytes
                    assert prelaunch_path.read_bytes() == prelaunch_bytes
                    assert not (control_dir / "global_hold_release_v15.json").exists()
                    assert not any(control_dir.glob("execution_isolation_gate_v15_pending_*.json"))
                    assert not any(control_dir.glob("execution_isolation_gate_v15_released_*.json"))
                    if label == "prepared_ledger":
                        # The append syscall and file fsync completed before
                        # the injected failure; restart must recognize this
                        # exact PREPARED row rather than append another.
                        assert ledger.is_file()
                        rows_after_fault = [json.loads(line)
                            for line in ledger.read_text().splitlines()]
                        assert len(rows_after_fault) == 1
                        assert rows_after_fault[0]["event"] == "PREPARED"
                        assert rows_after_fault[0]["transaction_id"] == prepared["transaction_id"]
                        prepared_bytes_at_fault = ledger.read_bytes().splitlines(keepends=True)[0]
                        assert source_path is None
                    else:
                        archive = control_dir / (target +
                            hashlib.sha256(source_bytes).hexdigest() + ".bin")
                        assert archive.is_file() and not archive.is_symlink()
                        assert archive.read_bytes() == source_bytes
                        # Failure happens while archiving original input
                        # bytes, before PREPARED and pointer mutation.
                        assert source_path.read_bytes() == source_bytes
                        assert not ledger.exists()

                    expected_archive_sources = [
                        ("gate", gate_path.read_bytes()),
                        ("hold", hold_path.read_bytes()),
                        ("release_pointer", pointer_path.read_bytes())]
                    order = {"gate": 0, "hold": 1, "release_pointer": 2}
                    archived_through = (2 if label == "prepared_ledger" else order[label])
                    for archive_label, archive_source in expected_archive_sources:
                        archive = control_dir / ("v15_preimage_" + archive_label + "_" +
                            hashlib.sha256(archive_source).hexdigest() + ".bin")
                        if order[archive_label] <= archived_through:
                            assert archive.is_file() and archive.read_bytes() == archive_source
                        else:
                            assert not archive.exists()

                    # At every incomplete pre-PREPARED or PREPARED boundary,
                    # admission remains closed. A newly active hold also
                    # blocks the original identity without releasing or
                    # rewriting any archived preimage or pointer.
                    from stage_F0_worker_control import validate_execution_gate
                    with pytest.raises(PermissionError):
                        validate_execution_gate(root=case, gate_path=gate_path,
                            search_gate_path=case / v15.SEARCH_GATE_REL)
                    transaction_paths = [gate_path, hold_path, pointer_path,
                        control_dir / "global_hold_release_v15.json",
                        control_dir / ("execution_isolation_gate_v15_pending_" +
                            prepared["nonce"] + ".json"),
                        control_dir / ("execution_isolation_gate_v15_released_" +
                            prepared["nonce"] + ".json"), ledger]
                    before_later_hold = [(path.exists(), path.read_bytes()
                        if path.exists() else None) for path in transaction_paths]
                    later_hold = json.loads(source_hold_bytes)
                    later_hold.update({"active": True,
                        "incident_id": "q3-prepared-later-hold-" + label,
                        "failure_nonce": "q3-prepared-later-nonce-" + label})
                    later_path = hold_path.with_name(hold_path.name + ".q3-prepared-hold")
                    later_path.write_text(json.dumps(later_hold, sort_keys=True) + "\n")
                    os.chmod(later_path, 0o444)
                    os.replace(later_path, hold_path)
                    later_bytes = hold_path.read_bytes()
                    with pytest.raises(PermissionError):
                        v15.commit_principal_release(runs_root=case_runs,
                            prepared=prepared, gate_pending=gate, release_payload=release,
                            live_observation={"devices": [],
                                "observation_started_monotonic": -1})
                    assert hold_path.read_bytes() == later_bytes
                    assert [(path.exists(), path.read_bytes() if path.exists() else None)
                            for path in transaction_paths] == [
                                (exists, raw if path != hold_path else later_bytes)
                                for path, (exists, raw) in zip(transaction_paths,
                                    before_later_hold)]
                    restore = hold_path.with_name(hold_path.name + ".q3-prepared-restore")
                    restore.write_bytes(source_hold_bytes)
                    os.chmod(restore, 0o444)
                    os.replace(restore, hold_path)

                    # A durable PREPARED identity must reject a foreign nonce
                    # before the original transaction is resumed, with no
                    # pointer, archive, ledger, observation, or telemetry
                    # mutation. Pre-PREPARED archive interruptions have no
                    # committed transaction identity to compare yet.
                    if label == "prepared_ledger":
                        import uuid
                        foreign_nonce = "foreign-before-resume-prepared-v15"
                        foreign_prepared = dict(prepared, nonce=foreign_nonce,
                            transaction_id=uuid.uuid5(uuid.NAMESPACE_URL,
                                "f0-v15:" + foreign_nonce).hex)
                        foreign_gate = dict(gate,
                            prepared_release_id=foreign_prepared["transaction_id"])
                        foreign_release = dict(release,
                            release_id=foreign_prepared["transaction_id"],
                            nonce=foreign_nonce,
                            pending_gate_sha256=v15.sha256_bytes(
                                v15._canonical_indented_bytes(foreign_gate)))
                        immediate_paths = [gate_path, hold_path, pointer_path,
                            control_dir / "global_hold_release_v15.json",
                            control_dir / ("execution_isolation_gate_v15_pending_" +
                                prepared["nonce"] + ".json"),
                            control_dir / ("execution_isolation_gate_v15_released_" +
                                prepared["nonce"] + ".json"), ledger,
                            control_dir / v15.OBSERVATION_LEDGER_REL.split("runs/", 1)[1]]
                        before_foreign = [(path.exists(), path.read_bytes()
                            if path.exists() else None) for path in immediate_paths]
                        queries_before_foreign = len(telemetry_queries)
                        with pytest.raises(PermissionError):
                            v15.commit_principal_release(runs_root=case_runs,
                                prepared=foreign_prepared, gate_pending=foreign_gate,
                                release_payload=foreign_release,
                                live_observation={"devices": [],
                                    "observation_started_monotonic": -1})
                        assert [(path.exists(), path.read_bytes() if path.exists() else None)
                                for path in immediate_paths] == before_foreign
                        assert len(telemetry_queries) == queries_before_foreign

                    resumed = v15.commit_principal_release(runs_root=case_runs,
                        prepared=prepared, gate_pending=gate, release_payload=release,
                        live_observation={"devices": [],
                            "observation_started_monotonic": -1})
                    assert resumed["status"] == "COMMITTED"
                    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
                    assert rows[0]["event"] == "PREPARED"
                    assert rows[0]["transaction_id"] == prepared["transaction_id"]
                    prepared_after_recovery = ledger.read_bytes().splitlines(keepends=True)[0]
                    assert prepared_after_recovery == (prepared_bytes_at_fault if
                        label == "prepared_ledger" else prepared_after_recovery)
                    assert sum(row["event"] == "PREPARED" for row in rows) == 1
                    assert [row["event"] for row in rows].count("COMMITTED") == 1
                    assert source_snapshot_path.read_bytes() == source_snapshot_bytes
                    assert prelaunch_path.read_bytes() == prelaunch_bytes

                    # Once PREPARED exists, a different nonce cannot write or
                    # query; after completion a same-identity ACK preserves
                    # every byte and does not consult telemetry.
                    alternate_nonce = "foreign-preimage-v15-" + label + "-nonce"
                    import uuid
                    alternate_prepared = dict(prepared, nonce=alternate_nonce,
                        transaction_id=uuid.uuid5(uuid.NAMESPACE_URL,
                            "f0-v15:" + alternate_nonce).hex)
                    alternate_gate = dict(gate,
                        prepared_release_id=alternate_prepared["transaction_id"])
                    alternate_release = dict(release,
                        release_id=alternate_prepared["transaction_id"], nonce=alternate_nonce,
                        pending_gate_sha256=v15.sha256_bytes(
                            v15._canonical_indented_bytes(alternate_gate)))
                    final_paths = [gate_path, hold_path,
                        control_dir / "global_hold_release_v15.json",
                        control_dir / ("execution_isolation_gate_v15_pending_" +
                            prepared["nonce"] + ".json"), pointer_path,
                        control_dir / ("execution_isolation_gate_v15_released_" +
                            prepared["nonce"] + ".json"), ledger,
                        control_dir / v15.OBSERVATION_LEDGER_REL.split("runs/", 1)[1]]
                    before_wrong_nonce = [(path.exists(), path.read_bytes()
                        if path.exists() else None) for path in final_paths]
                    queries_before_wrong_nonce = len(telemetry_queries)
                    with pytest.raises(PermissionError):
                        v15.commit_principal_release(runs_root=case_runs,
                            prepared=alternate_prepared, gate_pending=alternate_gate,
                            release_payload=alternate_release,
                            live_observation={"devices": [],
                                "observation_started_monotonic": -1})
                    assert [(path.exists(), path.read_bytes() if path.exists() else None)
                            for path in final_paths] == before_wrong_nonce
                    assert len(telemetry_queries) == queries_before_wrong_nonce
                    before_ack = [(path.exists(), path.read_bytes() if path.exists() else None)
                                  for path in final_paths]
                    queries_before_ack = len(telemetry_queries)
                    assert v15.commit_principal_release(runs_root=case_runs,
                        prepared=prepared, gate_pending=gate, release_payload=release,
                        live_observation={"devices": [], "observation_started_monotonic": -1})[
                            "status"] == "COMMITTED_IDEMPOTENT"
                    assert [(path.exists(), path.read_bytes() if path.exists() else None)
                            for path in final_paths] == before_ack
                    assert len(telemetry_queries) == queries_before_ack
                finally:
                    shutil.rmtree(case, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_each_release_syscall_boundary_has_independent_restart_oracle(monkeypatch, tmp_path):
    """Inject each release fsync/append boundary and verify exact recovery."""
    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    base = Path(tempfile.mkdtemp(prefix=".v15-syscall-boundary-base-", dir=v15.ROOT))
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in (
        "stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
        "stage_F0_protocol", "stage_F0_queue")}
    try:
        _hardlink_tree(v15.ROOT / "experiments", base / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", base / "m3oe")
        _detach_clone_worker_logs(base)
        sys.path.insert(0, str(base / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        gate, prepared, prelaunch_sha, source_doc, _device = _build_valid_pending_graph(base)
        release = {"schema": v15.RELEASE_SCHEMA,
            "release_id": prepared["transaction_id"], "nonce": prepared["nonce"],
            "approved_by": prepared["approved_by"], "approved_by_role": "principal",
            "principal_intent": v15.SCOPE, "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha,
            "targets": prepared["targets"], "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate))}
        telemetry_queries = []
        real_query = v15.query_gpu_memory
        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            telemetry_queries.append((gpu_index, expected_uuid))
            return real_query(gpu_index=gpu_index, expected_uuid=expected_uuid,
                provider=provider or (lambda index: {"index": index, "uuid": expected_uuid,
                    "memory_total_bytes": 1000, "memory_used_bytes": 100}))
        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        scheduler = tmp_path / "v15-release-syscall-scheduler"
        scheduler.mkdir()
        monkeypatch.setenv("F0_PROCESS_SCHEDULER_ROOT", str(scheduler))
        for env_key in ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                        "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE"):
            monkeypatch.delenv(env_key, raising=False)

        # The baseline has one immutable PREPARED row and the original
        # canonical gate/hold/release pointer. Every case gets a separate COW
        # clone of precisely that same preimage.
        base_runs = base / "experiments/stage_F0_outputs/runs"
        with pytest.raises(RuntimeError, match="fault injection after PREPARED"):
            v15.commit_principal_release(runs_root=base_runs, prepared=prepared,
                gate_pending=gate, release_payload=release,
                live_observation={"devices": [], "observation_started_monotonic": -1},
                fault_after="PREPARED")
        base_ledger = base_runs / v15.RELEASE_LEDGER_REL.split("runs/", 1)[1]
        prepared_line = base_ledger.read_bytes().splitlines(keepends=True)[0]
        assert json.loads(prepared_line)["event"] == "PREPARED"

        # Each item identifies the exact opened file whose successful fsync
        # will be followed by an injected interruption. Directory entries are
        # separate cases, triggered by the immediately following parent fsync.
        boundary_specs = [
            ("observation:LOCKED_RELEASE_DECISION:file", "PREPARED"),
            ("observation:LOCKED_RELEASE_DECISION:dir", "PREPARED"),
            ("release_archive:file", "PREPARED"),
            ("release_archive:dir", "PREPARED"),
            ("ledger:RELEASE_ARTIFACT_DURABLE:file", "RELEASE_ARTIFACT_DURABLE"),
            ("ledger:RELEASE_ARTIFACT_DURABLE:dir", "RELEASE_ARTIFACT_DURABLE"),
            ("pending_archive:file", "RELEASE_ARTIFACT_DURABLE"),
            ("pending_archive:dir", "RELEASE_ARTIFACT_DURABLE"),
            ("pending_temp:file", "RELEASE_ARTIFACT_DURABLE"),
            ("pending_gate:rename_dir", "RELEASE_ARTIFACT_DURABLE"),
            ("ledger:PENDING_GATE_DURABLE:file", "PENDING_GATE_DURABLE"),
            ("ledger:PENDING_GATE_DURABLE:dir", "PENDING_GATE_DURABLE"),
            ("pointer_temp:file", "PENDING_GATE_DURABLE"),
            ("pointer:rename_dir", "PENDING_GATE_DURABLE"),
            ("ledger:RELEASE_POINTER_DURABLE:file", "RELEASE_POINTER_DURABLE"),
            ("ledger:RELEASE_POINTER_DURABLE:dir", "RELEASE_POINTER_DURABLE"),
            ("observation:PRE_HOLD_RELEASE_DECISION:file", "RELEASE_POINTER_DURABLE"),
            ("observation:PRE_HOLD_RELEASE_DECISION:dir", "RELEASE_POINTER_DURABLE"),
            ("hold_temp:file", "RELEASE_POINTER_DURABLE"),
            ("hold:rename_dir", "RELEASE_POINTER_DURABLE"),
            ("ledger:HOLD_RELEASED:file", "HOLD_RELEASED"),
            ("ledger:HOLD_RELEASED:dir", "HOLD_RELEASED"),
            ("ledger:COMMITTED:file", "COMMITTED"),
            ("ledger:COMMITTED:dir", "COMMITTED"),
            ("final_archive:file", "COMMITTED"),
            ("final_archive:dir", "COMMITTED"),
            ("final_gate_temp:file", "COMMITTED"),
            ("final_gate:rename_dir", "COMMITTED"),
            ("terminal_fence:file", "COMMITTED"),
            ("terminal_fence:dir", "COMMITTED"),
            # Retry-only existing-object durability points. The seed hook
            # leaves the object visible; the second call faults the exact
            # fsync performed by its recovery branch.
            ("recovery:release_archive:file", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:release_archive:dir", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:pending_archive:file", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:pending_archive:dir", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:pending_gate:file", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:pending_gate:dir", "RELEASE_ARTIFACT_DURABLE"),
            ("recovery:release_pointer:file", "PENDING_GATE_DURABLE"),
            ("recovery:release_pointer:dir", "PENDING_GATE_DURABLE"),
            ("recovery:hold:file", "RELEASE_POINTER_DURABLE"),
            ("recovery:hold:dir", "RELEASE_POINTER_DURABLE"),
            ("recovery:final_archive:file", "COMMITTED"),
            ("recovery:final_archive:dir", "COMMITTED"),
        ]
        assert len(boundary_specs) == 42
        assert len({name for name, _ in boundary_specs}) == len(boundary_specs)
        assert sum(name.startswith("recovery:") for name, _ in boundary_specs) == 12
        recovery_seed = {
            "release_archive": "RELEASE_ARTIFACT_DURABLE",
            "pending_archive": "PENDING_GATE_RENAME",
            "pending_gate": "PENDING_GATE_RENAME",
            "release_pointer": "RELEASE_POINTER_RENAME",
            "hold": "HOLD_RENAME",
            "final_archive": "FINAL_GATE_ARCHIVE",
        }
        ordered_events = ["PREPARED", "RELEASE_ARTIFACT_DURABLE",
            "PENDING_GATE_DURABLE", "RELEASE_POINTER_DURABLE", "HOLD_RELEASED", "COMMITTED"]
        event_index = {event: index for index, event in enumerate(ordered_events)}
        for ordinal, (boundary, last_durable_event) in enumerate(boundary_specs):
            case = Path(tempfile.mkdtemp(prefix=".v15-release-syscall-{:02d}-".format(ordinal),
                                         dir=v15.ROOT))
            try:
                _hardlink_tree(base / "experiments", case / "experiments")
                _hardlink_tree(base / "m3oe", case / "m3oe")
                _detach_clone_worker_logs(case)
                _detach_clone_mutable_locks(case)
                runs = case / "experiments/stage_F0_outputs/runs"
                control = runs / "execution_control"
                for rel in (v15.RELEASE_LEDGER_REL, v15.OBSERVATION_LEDGER_REL):
                    path = runs / rel.split("runs/", 1)[1]
                    if path.exists():
                        copy = path.with_name(path.name + ".q3-cow")
                        copy.write_bytes(path.read_bytes())
                        os.chmod(copy, stat.S_IMODE(path.stat().st_mode))
                        os.replace(copy, path)

                canonical = case / v15.CANONICAL_GATE_REL
                hold = runs / v15.HOLD_REL.split("runs/", 1)[1]
                release_artifact = control / "global_hold_release_v15.json"
                pending_archive = control / ("execution_isolation_gate_v15_pending_"
                    + prepared["nonce"] + ".json")
                release_pointer = control / "global_hold_release.json"
                final_archive = control / ("execution_isolation_gate_v15_released_"
                    + prepared["nonce"] + ".json")
                ledger = control / "postbootstrap_v15_release.jsonl"
                observations = control / "postbootstrap_v15_runtime_observations.jsonl"
                source_path = case / gate["source_snapshot_path"]
                prelaunch_path = case / gate["prelaunch_audit_path"]
                immutable_source = source_path.read_bytes()
                immutable_prelaunch = prelaunch_path.read_bytes()
                prepared_before = ledger.read_bytes().splitlines(keepends=True)[0]
                prepared_record = json.loads(prepared_before)
                preimage_paths = [(control / ("v15_preimage_" + label + "_" + digest + ".bin"),
                                   digest)
                    for label, digest in prepared_record["source_preimages"].items()]
                source_gate_bytes = canonical.read_bytes()
                source_hold_bytes = hold.read_bytes()
                source_pointer_bytes = release_pointer.read_bytes()

                def identify_regular_file(path):
                    name = path.name
                    if name == "global_hold_release.json":
                        return "release_pointer"
                    if name == "global_hold.json":
                        return "hold"
                    if name == ledger.name:
                        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
                        return "ledger:" + rows[-1]["event"]
                    if name == observations.name:
                        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
                        return "observation:" + rows[-1]["stage"]
                    if ".tmp." in name:
                        target_name = name.split(".tmp.", 1)[0]
                        payload = json.loads(path.read_bytes())
                        if target_name == "execution_isolation_gate.json":
                            return "final_gate_temp" if payload.get("launch_authorized") is True else "pending_temp"
                        if target_name == "global_hold_release.json":
                            return "pointer_temp"
                        if target_name == "global_hold.json":
                            return "hold_temp"
                    if name == release_artifact.name:
                        return "release_archive"
                    if name == pending_archive.name:
                        return "pending_archive"
                    if name == final_archive.name:
                        return "final_archive"
                    if path == canonical and json.loads(path.read_bytes()).get("launch_authorized") is True:
                        return "terminal_fence"
                    if path == canonical and json.loads(path.read_bytes()).get("schema") == v15.SCHEMA:
                        return "pending_gate"
                    return None

                state = {"target_file_key": None, "injected": False, "fd_path": None,
                         "phase": "forward", "matches": [], "seed_completed": None,
                         "temp_bytes": None, "temp_destination": None,
                         "destination_before_temp": None}
                real_fsync = os.fsync
                # Explicit (file class, phase, boundary id) rows avoid dict
                # key replacement: forward creation and recovery of the same
                # archive are distinct callsites and remain independently
                # selectable.
                directory_boundaries = [
                    ("release_archive", "forward", "release_archive:dir"),
                    ("pending_archive", "forward", "pending_archive:dir"),
                    ("final_archive", "forward", "final_archive:dir"),
                    ("release_archive", "recovery", "recovery:release_archive:dir"),
                    ("pending_archive", "recovery", "recovery:pending_archive:dir"),
                    ("final_archive", "recovery", "recovery:final_archive:dir"),
                    ("release_pointer", "recovery", "recovery:release_pointer:dir"),
                    ("hold", "recovery", "recovery:hold:dir"),
                    ("pending_gate", "recovery", "recovery:pending_gate:dir"),
                    ("ledger:RELEASE_ARTIFACT_DURABLE", "forward", "ledger:RELEASE_ARTIFACT_DURABLE:dir"),
                    ("ledger:PENDING_GATE_DURABLE", "forward", "ledger:PENDING_GATE_DURABLE:dir"),
                    ("ledger:RELEASE_POINTER_DURABLE", "forward", "ledger:RELEASE_POINTER_DURABLE:dir"),
                    ("ledger:HOLD_RELEASED", "forward", "ledger:HOLD_RELEASED:dir"),
                    ("ledger:COMMITTED", "forward", "ledger:COMMITTED:dir"),
                    ("observation:LOCKED_RELEASE_DECISION", "forward", "observation:LOCKED_RELEASE_DECISION:dir"),
                    ("observation:PRE_HOLD_RELEASE_DECISION", "forward", "observation:PRE_HOLD_RELEASE_DECISION:dir"),
                    ("pending_temp", "forward", "pending_gate:rename_dir"),
                    ("pointer_temp", "forward", "pointer:rename_dir"),
                    ("hold_temp", "forward", "hold:rename_dir"),
                    ("final_gate_temp", "forward", "final_gate:rename_dir"),
                    ("terminal_fence", "forward", "terminal_fence:dir"),
                ]
                assert len(directory_boundaries) == len({(key, phase) for key, phase, _ in directory_boundaries})
                assert len(directory_boundaries) == len({label for _, _, label in directory_boundaries})

                def fault_after_selected_fsync(fd):
                    try:
                        fd_path = Path(os.readlink("/proc/self/fd/{}".format(fd)))
                        mode = os.fstat(fd).st_mode
                    except (OSError, ValueError):
                        fd_path, mode = Path(""), 0
                    result = real_fsync(fd)
                    phase = state["phase"]
                    if stat.S_ISREG(mode):
                        file_key = identify_regular_file(fd_path)
                        if file_key is not None:
                            state["fd_path"] = fd_path
                            state["target_file_key"] = file_key
                            file_boundary = ("recovery:" + file_key + ":file"
                                if phase == "recovery" else file_key + ":file")
                            if boundary == file_boundary and not state["injected"]:
                                state["injected"] = True
                                temp_destinations = {
                                    "pending_temp": canonical,
                                    "pointer_temp": release_pointer,
                                    "hold_temp": hold,
                                    "final_gate_temp": canonical,
                                }
                                if file_key in temp_destinations:
                                    destination = temp_destinations[file_key]
                                    state["temp_bytes"] = fd_path.read_bytes()
                                    state["temp_destination"] = destination
                                    state["destination_before_temp"] = destination.read_bytes()
                                state["matches"].append({"boundary": boundary,
                                    "phase": phase, "fd_kind": "regular",
                                    "file_key": file_key, "fd_path": str(fd_path)})
                                raise RuntimeError("Q3 injected after file fsync: " + boundary)
                    elif stat.S_ISDIR(mode):
                        expected_dir_boundary = next((label for key, mode_phase, label
                            in directory_boundaries if key == state["target_file_key"]
                            and mode_phase == phase), None)
                        expected_parent = (canonical.parent if state["target_file_key"] in {
                            "pending_temp", "pending_gate", "final_gate_temp", "terminal_fence"} else control)
                        if (fd_path.resolve() == expected_parent.resolve()
                                and expected_dir_boundary == boundary and not state["injected"]):
                            state["injected"] = True
                            state["matches"].append({"boundary": boundary,
                                "phase": phase, "fd_kind": "directory",
                                "file_key": state["target_file_key"],
                                "fd_path": str(fd_path.resolve())})
                            raise RuntimeError("Q3 injected after parent fsync: " + boundary)
                    return result

                with monkeypatch.context() as scoped:
                    scoped.setattr(os, "fsync", fault_after_selected_fsync)
                    if boundary.startswith("recovery:"):
                        recovery_object = boundary.split(":")[1]
                        state["phase"] = "seed"
                        with pytest.raises(RuntimeError,
                                match="fault injection at " + recovery_seed[recovery_object]) as seed_exc:
                            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                                gate_pending=gate, release_payload=release,
                                live_observation={"devices": [],
                                    "observation_started_monotonic": -1},
                                fault_after=recovery_seed[recovery_object])
                        state["seed_completed"] = str(seed_exc.value)
                        assert state["seed_completed"] == (
                            "fault injection at " + recovery_seed[recovery_object])
                        assert state["matches"] == [], "recovery selector fired during seed phase"
                        assert release_artifact.is_file()
                        if recovery_object == "release_archive":
                            assert release_artifact.read_bytes() == v15._canonical_indented_bytes(release)
                        elif recovery_object in {"pending_archive", "pending_gate"}:
                            assert pending_archive.read_bytes() == v15._canonical_indented_bytes(gate)
                            assert canonical.read_bytes() == v15._canonical_indented_bytes(gate)
                        elif recovery_object == "release_pointer":
                            assert release_pointer.read_bytes() == v15._canonical_indented_bytes(release)
                        elif recovery_object == "hold":
                            assert json.loads(hold.read_text())["active"] is False
                            assert release_pointer.read_bytes() == v15._canonical_indented_bytes(release)
                        elif recovery_object == "final_archive":
                            assert json.loads(final_archive.read_text())["transaction_state"] == "COMMITTED"
                            assert canonical.read_bytes() == v15._canonical_indented_bytes(gate)
                        state["phase"] = "recovery"
                        state.update({"injected": False, "fd_path": None,
                                      "target_file_key": None, "matches": []})
                    with pytest.raises(RuntimeError, match="Q3 injected after"):
                        v15.commit_principal_release(runs_root=runs, prepared=prepared,
                            gate_pending=gate, release_payload=release,
                            live_observation={"devices": [],
                                "observation_started_monotonic": -1})
                assert state["injected"] is True, "requested syscall boundary was not reached: " + boundary
                expected_file_key = {
                    "pending_gate:rename_dir": "pending_temp",
                    "pointer:rename_dir": "pointer_temp",
                    "hold:rename_dir": "hold_temp",
                    "final_gate:rename_dir": "final_gate_temp",
                }.get(boundary, boundary.split(":")[-2]
                    if boundary.startswith("recovery:") else boundary.rsplit(":", 1)[0])
                assert state["target_file_key"] == expected_file_key, boundary
                assert len(state["matches"]) == 1, (boundary, state["matches"])
                match = state["matches"][0]
                expected_phase = "recovery" if boundary.startswith("recovery:") else "forward"
                assert match["boundary"] == boundary and match["phase"] == expected_phase
                expected_fd_kind = ("directory" if boundary.endswith(":dir")
                    or boundary.endswith(":rename_dir") else "regular")
                assert match["fd_kind"] == expected_fd_kind
                assert Path(match["fd_path"]).exists(), (boundary, match)
                expected_fd_path = {
                    "release_archive": release_artifact,
                    "pending_archive": pending_archive,
                    "final_archive": final_archive,
                    "pending_gate": canonical,
                    "release_pointer": release_pointer,
                    "hold": hold,
                    "ledger": ledger,
                    "observation": observations,
                    "pending_temp": canonical,
                    "pointer_temp": release_pointer,
                    "hold_temp": hold,
                    "final_gate_temp": canonical,
                    "terminal_fence": canonical,
                }[expected_file_key.split(":")[0]]
                if expected_fd_kind == "regular":
                    if expected_file_key.endswith("_temp"):
                        temp_path = Path(match["fd_path"])
                        assert temp_path.parent == expected_fd_path.parent, (boundary, match)
                        assert temp_path.name.startswith(expected_fd_path.name + ".tmp."), (boundary, match)
                        suffix = temp_path.name[len(expected_fd_path.name + ".tmp."):]
                        assert re.fullmatch(r"[0-9a-f]{32}", suffix), (boundary, suffix)
                        assert stat.S_ISREG(temp_path.stat().st_mode), (boundary, match)
                        assert state["temp_destination"] == expected_fd_path
                        assert state["temp_bytes"] == temp_path.read_bytes()
                        old_destination = {
                            "pending_temp": source_gate_bytes,
                            "pointer_temp": source_pointer_bytes,
                            "hold_temp": source_hold_bytes,
                            "final_gate_temp": v15._canonical_indented_bytes(gate),
                        }[expected_file_key]
                        assert state["destination_before_temp"] == old_destination, boundary
                        assert expected_fd_path.read_bytes() == old_destination, boundary
                        if expected_file_key == "pending_temp":
                            expected_payload = v15._canonical_indented_bytes(gate)
                        elif expected_file_key == "pointer_temp":
                            expected_payload = v15._canonical_indented_bytes(release)
                        elif expected_file_key == "hold_temp":
                            expected_payload = state["temp_bytes"]
                        elif expected_file_key == "final_gate_temp":
                            expected_payload = final_archive.read_bytes()
                        else:
                            raise AssertionError("unexpected atomic temp class: " + expected_file_key)
                        assert state["temp_bytes"] == expected_payload, boundary
                        if expected_file_key == "hold_temp":
                            released = json.loads(state["temp_bytes"])
                            assert released["active"] is False
                            assert v15.sha256_bytes(state["temp_bytes"]) == prepared_record[
                                "released_hold_sha256"]
                        if expected_file_key == "final_gate_temp":
                            assert json.loads(state["temp_bytes"])["transaction_state"] == "COMMITTED"
                    else:
                        assert Path(match["fd_path"]) == expected_fd_path, (boundary, match)
                else:
                    expected_parent = expected_fd_path.parent.resolve()
                    assert Path(match["fd_path"]).resolve() == expected_parent, (boundary, match)

                rows_at_fault = [json.loads(line) for line in ledger.read_bytes().splitlines()]
                assert [row["event"] for row in rows_at_fault] == ordered_events[:
                    event_index[last_durable_event] + 1]
                assert ledger.read_bytes().splitlines(keepends=True)[0] == prepared_before
                assert source_path.read_bytes() == immutable_source
                assert prelaunch_path.read_bytes() == immutable_prelaunch
                for preimage_path, digest in preimage_paths:
                    assert preimage_path.is_file()
                    assert hashlib.sha256(preimage_path.read_bytes()).hexdigest() == digest
                terminal_gate = json.loads(canonical.read_text())
                is_terminal = (terminal_gate.get("schema") == v15.SCHEMA
                    and terminal_gate.get("transaction_state") == "COMMITTED"
                    and terminal_gate.get("launch_authorized") is True
                    and terminal_gate.get("committed_ledger_sha256") ==
                        hashlib.sha256(ledger.read_bytes()).hexdigest())
                final_publication_boundaries = {
                    "final_gate:rename_dir", "terminal_fence:file", "terminal_fence:dir"}
                pending_publication_boundaries = {
                    "pending_gate:rename_dir", "ledger:PENDING_GATE_DURABLE:file",
                    "ledger:PENDING_GATE_DURABLE:dir", "pointer_temp:file", "pointer:rename_dir",
                    "ledger:RELEASE_POINTER_DURABLE:file", "ledger:RELEASE_POINTER_DURABLE:dir",
                    "observation:PRE_HOLD_RELEASE_DECISION:file",
                    "observation:PRE_HOLD_RELEASE_DECISION:dir", "hold_temp:file", "hold:rename_dir",
                    "ledger:HOLD_RELEASED:file", "ledger:HOLD_RELEASED:dir",
                    "ledger:COMMITTED:file", "ledger:COMMITTED:dir", "final_archive:file",
                    "final_archive:dir", "final_gate_temp:file",
                    "recovery:pending_archive:file", "recovery:pending_archive:dir",
                    "recovery:pending_gate:file", "recovery:pending_gate:dir",
                    "recovery:release_pointer:file", "recovery:release_pointer:dir",
                    "recovery:hold:file", "recovery:hold:dir",
                    "recovery:final_archive:file", "recovery:final_archive:dir",
                    *final_publication_boundaries}
                pending_archive_boundaries = {
                    "pending_archive:file", "pending_archive:dir", *pending_publication_boundaries}
                expected_terminal = boundary in final_publication_boundaries
                assert is_terminal is expected_terminal, boundary
                assert canonical.read_bytes() == (final_archive.read_bytes() if is_terminal else
                    v15._canonical_indented_bytes(gate) if boundary in pending_publication_boundaries
                    else source_gate_bytes)
                pointer_has_been_published = boundary in {
                    "pointer:rename_dir", "ledger:RELEASE_POINTER_DURABLE:file",
                    "ledger:RELEASE_POINTER_DURABLE:dir",
                    "observation:PRE_HOLD_RELEASE_DECISION:file",
                    "observation:PRE_HOLD_RELEASE_DECISION:dir", "hold_temp:file",
                    "hold:rename_dir", "ledger:HOLD_RELEASED:file", "ledger:HOLD_RELEASED:dir",
                    "ledger:COMMITTED:file", "ledger:COMMITTED:dir", "final_archive:file",
                    "final_archive:dir", "final_gate_temp:file",
                    "recovery:release_pointer:file", "recovery:release_pointer:dir",
                    "recovery:hold:file", "recovery:hold:dir",
                    "recovery:final_archive:file", "recovery:final_archive:dir",
                    *final_publication_boundaries}
                assert release_pointer.read_bytes() == (v15._canonical_indented_bytes(release)
                    if pointer_has_been_published else source_pointer_bytes)
                hold_is_released = boundary in {
                    "hold:rename_dir", "ledger:HOLD_RELEASED:file", "ledger:HOLD_RELEASED:dir",
                    "ledger:COMMITTED:file", "ledger:COMMITTED:dir", "final_archive:file",
                    "final_archive:dir", "final_gate_temp:file", "recovery:hold:file",
                    "recovery:hold:dir", "recovery:final_archive:file",
                    "recovery:final_archive:dir", *final_publication_boundaries}
                assert json.loads(hold.read_text())["active"] is (not hold_is_released)
                if boundary != "observation:LOCKED_RELEASE_DECISION:file" and boundary != \
                        "observation:LOCKED_RELEASE_DECISION:dir":
                    assert release_artifact.read_bytes() == v15._canonical_indented_bytes(release)
                if boundary in pending_archive_boundaries:
                    assert pending_archive.read_bytes() == v15._canonical_indented_bytes(gate)
                if boundary in {"final_archive:file", "final_archive:dir",
                                "recovery:final_archive:file", "recovery:final_archive:dir",
                                "final_gate_temp:file",
                                *final_publication_boundaries}:
                    assert json.loads(final_archive.read_text())["launch_authorized"] is True
                if boundary.startswith("observation:"):
                    observation_rows = [json.loads(line)
                        for line in observations.read_bytes().splitlines()]
                    assert observation_rows[-1]["stage"] == boundary.split(":")[1]
                    assert (observation_rows[-1]["target"]["index"],
                            observation_rows[-1]["target"]["uuid"]) == (
                                prepared["targets"][0]["index"],
                                prepared["targets"][0]["uuid"])
                if not is_terminal:
                    from stage_F0_worker_control import validate_execution_gate
                    with pytest.raises(PermissionError):
                        validate_execution_gate(root=case, gate_path=canonical,
                            search_gate_path=case / v15.SEARCH_GATE_REL)
                else:
                    from stage_F0_worker_control import validate_execution_gate
                    accepted = validate_execution_gate(root=case, gate_path=canonical,
                        search_gate_path=case / v15.SEARCH_GATE_REL)
                    assert accepted["execution_gate"]["transaction_state"] == "COMMITTED"

                # A different but well-formed nonce cannot recover or rewrite
                # any interrupted graph; ledger/observation identities reject
                # it before another telemetry query.
                alternate_nonce = "foreign-syscall-v15-{:02d}-nonce".format(ordinal)
                import uuid
                alternate_prepared = dict(prepared, nonce=alternate_nonce,
                    transaction_id=uuid.uuid5(uuid.NAMESPACE_URL,
                        "f0-v15:" + alternate_nonce).hex)
                alternate_gate = dict(gate,
                    prepared_release_id=alternate_prepared["transaction_id"])
                alternate_release = dict(release,
                    release_id=alternate_prepared["transaction_id"], nonce=alternate_nonce,
                    pending_gate_sha256=v15.sha256_bytes(v15._canonical_indented_bytes(alternate_gate)))
                transaction_paths = [canonical, hold, release_artifact, pending_archive,
                    release_pointer, final_archive, ledger, observations]
                transaction_before_wrong_nonce = [(path.exists(), path.read_bytes()
                    if path.exists() else None) for path in transaction_paths]
                query_count_before_wrong_nonce = len(telemetry_queries)
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=runs,
                        prepared=alternate_prepared, gate_pending=alternate_gate,
                        release_payload=alternate_release,
                        live_observation={"devices": [], "observation_started_monotonic": -1})
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in transaction_paths] == transaction_before_wrong_nonce
                assert len(telemetry_queries) == query_count_before_wrong_nonce

                # Each interruption is also retried under a distinct later
                # active hold. Only the clone hold is changed; replay must
                # preserve it and leave every other transaction byte intact.
                prior_hold_bytes = hold.read_bytes()
                foreign_hold = json.loads(prior_hold_bytes)
                foreign_hold.update({"active": True,
                    "incident_id": "q3-later-hold-{:02d}".format(ordinal),
                    "failure_nonce": "q3-later-hold-nonce-{:02d}".format(ordinal)})
                hold_temp = hold.with_name(hold.name + ".q3-later-hold")
                hold_temp.write_text(json.dumps(foreign_hold, sort_keys=True) + "\n")
                os.chmod(hold_temp, 0o444)
                os.replace(hold_temp, hold)
                foreign_hold_bytes = hold.read_bytes()
                graph_paths = [canonical, release_artifact, pending_archive,
                    release_pointer, final_archive, ledger]
                graph_before_later_hold = [(path.exists(), path.read_bytes()
                    if path.exists() else None) for path in graph_paths]
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=runs, prepared=prepared,
                        gate_pending=gate, release_payload=release,
                        live_observation={"devices": [], "observation_started_monotonic": -1})
                assert hold.read_bytes() == foreign_hold_bytes
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in graph_paths] == graph_before_later_hold
                restore_hold = hold.with_name(hold.name + ".q3-restore-hold")
                restore_hold.write_bytes(prior_hold_bytes)
                os.chmod(restore_hold, 0o444)
                os.replace(restore_hold, hold)

                # Recover with the sole original identity. Preserve the exact
                # PREPARED line/audit/source bytes and require one event per
                # release phase, even when the interrupted syscall already
                # made its append or rename visible.
                recovered = v15.commit_principal_release(runs_root=runs,
                    prepared=prepared, gate_pending=gate, release_payload=release,
                    live_observation={"devices": [], "observation_started_monotonic": -1})
                assert recovered["status"] in ("COMMITTED", "COMMITTED_IDEMPOTENT")
                final_rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
                assert [row["event"] for row in final_rows] == ordered_events
                assert sum(line == prepared_before for line in
                    ledger.read_bytes().splitlines(keepends=True)) == 1
                assert source_path.read_bytes() == immutable_source
                assert prelaunch_path.read_bytes() == immutable_prelaunch
                assert canonical.read_bytes() == final_archive.read_bytes()
                assert json.loads(hold.read_text())["active"] is False

                # A fresh incident hold also blocks terminal acknowledgement;
                # after restoring the exact clone projection, ACK is strictly
                # read-only and does not query telemetry or append either log.
                terminal_files = [canonical, hold, release_artifact, pending_archive,
                    release_pointer, final_archive, ledger, observations]
                before_terminal_ack = [(path.exists(), path.read_bytes()
                    if path.exists() else None) for path in terminal_files]
                query_count_before_ack = len(telemetry_queries)
                later_terminal = json.loads(hold.read_text())
                later_terminal.update({"active": True,
                    "incident_id": "q3-terminal-hold-{:02d}".format(ordinal),
                    "failure_nonce": "q3-terminal-hold-nonce-{:02d}".format(ordinal)})
                hold_temp = hold.with_name(hold.name + ".q3-terminal-hold")
                hold_temp.write_text(json.dumps(later_terminal, sort_keys=True) + "\n")
                os.chmod(hold_temp, 0o444)
                os.replace(hold_temp, hold)
                terminal_hold_bytes = hold.read_bytes()
                before_rejected_ack = [(path.exists(), path.read_bytes()
                    if path.exists() else None) for path in terminal_files]
                with pytest.raises(PermissionError):
                    v15.commit_principal_release(runs_root=runs, prepared=prepared,
                        gate_pending=gate, release_payload=release,
                        live_observation={"devices": [], "observation_started_monotonic": -1})
                assert hold.read_bytes() == terminal_hold_bytes
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in terminal_files] == before_rejected_ack
                assert len(telemetry_queries) == query_count_before_ack
                restore_terminal = hold.with_name(hold.name + ".q3-terminal-restore")
                restore_terminal.write_bytes(before_terminal_ack[1][1])
                os.chmod(restore_terminal, 0o444)
                os.replace(restore_terminal, hold)
                stable_terminal = [(path.exists(), path.read_bytes()
                    if path.exists() else None) for path in terminal_files]
                query_count_at_ack = len(telemetry_queries)
                ack = v15.commit_principal_release(runs_root=runs, prepared=prepared,
                    gate_pending=gate, release_payload=release,
                    live_observation={"devices": [], "observation_started_monotonic": -1})
                assert ack["status"] == "COMMITTED_IDEMPOTENT"
                assert len(telemetry_queries) == query_count_at_ack
                assert [(path.exists(), path.read_bytes() if path.exists() else None)
                        for path in terminal_files] == stable_terminal
            finally:
                shutil.rmtree(case, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_real_runner_child_preclaim_smoke(tmp_path, monkeypatch):
    """Low-cost process-chain smoke before the complete fault/race E2E."""
    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    repo = Path(tempfile.mkdtemp(prefix=".v15-child-smoke-", dir=v15.ROOT))
    saved_path = list(sys.path)
    module_names = ("stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
                    "stage_F0_protocol", "stage_F0_queue")
    saved_modules = {name: sys.modules.pop(name, None) for name in module_names}
    try:
        _hardlink_tree(v15.ROOT / "experiments", repo / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", repo / "m3oe")
        source_log_hashes = _detach_clone_worker_logs(repo)
        sys.path.insert(0, str(repo / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        gate, prepared, prelaunch_sha, source_doc, device = _build_valid_pending_graph(repo)
        gate_path = repo / v15.CANONICAL_GATE_REL
        runs = repo / "experiments/stage_F0_outputs/runs"
        queue_doc = json.loads((repo / v15.QUEUE_REL).read_text())
        from stage_F0_postabnormal_v14 import H51_RUN_IDS
        real_query = v15.query_gpu_memory

        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            fake_provider = provider or (lambda index: {
                "index": index, "uuid": expected_uuid,
                "memory_total_bytes": 1000, "memory_used_bytes": 100})
            return real_query(gpu_index=gpu_index, expected_uuid=expected_uuid,
                              provider=fake_provider)

        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        payload = {
            "schema": v15.RELEASE_SCHEMA, "release_id": prepared["transaction_id"],
            "nonce": prepared["nonce"], "approved_by": prepared["approved_by"],
            "approved_by_role": "principal", "principal_intent": v15.SCOPE,
            "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha,
            "targets": prepared["targets"], "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate)),
        }

        def live_observation():
            row = v15.query_gpu_memory(gpu_index=0, expected_uuid=device["uuid"],
                provider=lambda index: {"index": index, "uuid": device["uuid"],
                    "memory_total_bytes": 1000, "memory_used_bytes": 100})
            return {"devices": [{**row, "selected": True}],
                    "observation_started_monotonic": row["observation_started_monotonic"],
                    "observed_at_utc": row["observed_at_utc"]}

        result = v15.commit_principal_release(runs_root=runs, prepared=prepared,
            gate_pending=gate, release_payload=payload, live_observation=live_observation())
        assert result["status"] == "COMMITTED"

        wrapper = tmp_path / "v15_child_smoke.py"
        wrapper.write_text("""\
import os, sys
sys.path.insert(0, os.environ['V15_CLONE_EXPERIMENTS'])
sys.path.insert(0, os.environ['V15_CLONE_RUNS'])
import stage_F0_successor_v15 as v15
def fake_live(*, runs_root, gpu_index, gpu_uuid, boundary, provider=None):
    row = v15.query_gpu_memory(gpu_index=gpu_index, expected_uuid=gpu_uuid,
        provider=lambda index: {'index': index, 'uuid': gpu_uuid,
            'memory_total_bytes': 1000, 'memory_used_bytes': 950})
    if boundary == 'CHILD_PRECLAIM':
        raise v15.ResourceDeferred(row, boundary)
    return row
v15.require_live_gpu_memory = fake_live
import stage_F0_run
stage_F0_run.runtime_preflight = lambda _root: {
    'cuda_available': True, 'device_count': 2,
    'resource_concurrency_policy': {'usable_gpu_indices': [0, 1]}, 'gpus': []}
from run_search_trajectory_child import main
raise SystemExit(main())
""", encoding="utf-8")
        import run_search_claim_worker as worker_module
        from stage_F0_queue import acquire_process_slot
        scheduler = str(worker_module.SCHEDULER_ROOT)
        supervisor_lease = acquire_process_slot(role="F0-supervisor-0", blocking=False,
                                                scheduler_dir=scheduler)
        child_lease = None
        try:
            child_lease = acquire_process_slot(role="F0-trajectory-child-0", blocking=False,
                                               scheduler_dir=scheduler)
            supervisor = worker_module.TrajectorySupervisor(
                worker_id="0", supervisor_slot_fd=supervisor_lease.lock.handle.fileno(),
                child_slot_fd=child_lease.lock.handle.fileno(),
                supervisor_slot_index=supervisor_lease.index,
                child_slot_index=child_lease.index, queue_path=repo / v15.QUEUE_REL,
                search_gate_path=repo / v15.SEARCH_GATE_REL,
                execution_gate_path=gate_path, data_path=worker_module.DATA,
                output_root=runs, python=worker_module.PYTHON, child_entry=wrapper)
            identity = worker_module.linux_process_identity(os.getpid())
            identity["nonce"] = "v15-child-smoke-supervisor"
            before = v15.capture_f140_state(runs, [row for row in queue_doc["entries"]
                if row["run_id"] in set(v15.frozen_f140_ids(repo))])
            env_names = ("F0_PROCESS_SCHEDULER_ROOT", "F0_RESERVED_PROCESS_SLOT_FD",
                "F0_RESERVED_PROCESS_SLOT_INDEX", "F0_RESERVED_PROCESS_SLOT_OWNER_PID",
                "F0_RESERVED_PROCESS_SLOT_ROLE", "V15_CLONE_EXPERIMENTS", "V15_CLONE_RUNS")
            previous = {key: os.environ.get(key) for key in env_names}
            os.environ.update({"F0_PROCESS_SCHEDULER_ROOT": scheduler,
                "F0_RESERVED_PROCESS_SLOT_FD": str(supervisor_lease.lock.handle.fileno()),
                "F0_RESERVED_PROCESS_SLOT_INDEX": str(supervisor_lease.index),
                "F0_RESERVED_PROCESS_SLOT_OWNER_PID": str(os.getpid()),
                "F0_RESERVED_PROCESS_SLOT_ROLE": "F0-supervisor-0",
                "V15_CLONE_EXPERIMENTS": str(repo / "experiments"),
                "V15_CLONE_RUNS": str(repo / "experiments/stage_F0_outputs/runs")})
            try:
                code, outcome, child_identity = supervisor._spawn_one(invocation=1,
                    identity=identity, child_args=["--queue", str(repo / v15.QUEUE_REL),
                    "--claim-next", "--dataset", str(worker_module.DATA),
                    "--output-root", str(runs), "--run-gate", str(repo / v15.SEARCH_GATE_REL),
                    "--device", "cuda:0", "--held-run-ids-json",
                    json.dumps(sorted(H51_RUN_IDS))])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            assert code == 75, Path(outcome["stderr_path"]).read_text()
            receipt = supervisor._validate_resource_deferral(outcome,
                identity=identity, child_identity=child_identity)
            assert receipt["boundary"] == "CHILD_PRECLAIM"
            assert receipt["canonical_claim"] is None
            assert receipt["attempt_created"] is False
            assert receipt["checkpoint_loaded"] is False
            assert v15.capture_f140_state(runs, [row for row in queue_doc["entries"]
                if row["run_id"] in set(v15.frozen_f140_ids(repo))]) == before
            phases = [json.loads(line)["phase"] for line in
                Path(outcome["phase_path"]).read_text().splitlines()]
            assert "child_preclaim_entry" in phases
            assert "resource_deferred_receipt_durable" in phases
            assert "training_before" not in phases
            assert {name: v15.sha256_file(v15.RUNS / "worker_logs" / name)
                    for name in source_log_hashes} == source_log_hashes
        finally:
            if child_lease is not None:
                child_lease.close()
            supervisor_lease.close()
    finally:
        sys.path[:] = saved_path
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        shutil.rmtree(repo)


def test_real_pending_graph_release_reentry_terminal_and_f140_claim(tmp_path, monkeypatch):
    """Real validators, isolated hardlink clone, fake telemetry, no training/data."""
    # The frozen v1.4 incident record stores the original production-root
    # failure-audit path. Validate that immutable historical graph with the
    # real production-path modules; all v1.5 state and dispatch below still
    # comes from the isolated clone. This is a path-identity constraint, not
    # a patched validator or acceptance result.
    production_v14 = importlib.import_module("stage_F0_postabnormal_v14")
    production_migration = importlib.import_module("stage_F0_migration")
    repo = Path(tempfile.mkdtemp(prefix=".v15-qualification-", dir=v15.ROOT))
    peer_supervisor = None
    peer_stop_path = None
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in (
        "stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
        "stage_F0_protocol", "stage_F0_queue")}
    try:
        _hardlink_tree(v15.ROOT / "experiments", repo / "experiments")
        _hardlink_tree(v15.ROOT / "m3oe", repo / "m3oe")
        source_log_hashes = _detach_clone_worker_logs(repo)
        sys.path.insert(0, str(repo / "experiments"))
        sys.modules["stage_F0_postabnormal_v14"] = production_v14
        sys.modules["stage_F0_migration"] = production_migration
        peer_continuity_requested = os.environ.get("V15_TEST_PEER_CONTINUITY") == "1"
        peer_evidence_root = None
        if peer_continuity_requested:
            evidence_value = os.environ.get("V15_TEST_EVIDENCE_ROOT")
            if not evidence_value or not Path(evidence_value).is_absolute():
                raise AssertionError("peer-continuity requires an absolute external evidence root")
            peer_evidence_root = Path(evidence_value)
            peer_evidence_root.mkdir(parents=True, exist_ok=False)
            _preserve_and_reset_clone_peer_log(
                v15.RUNS / "worker_logs/audited_worker_1.jsonl",
                repo / "experiments/stage_F0_outputs/runs/worker_logs/audited_worker_1.jsonl",
                peer_evidence_root / "peer-log-prehistory")
        gate, prepared, prelaunch_sha, source_doc, device = _build_valid_pending_graph(
            repo, include_second_target=peer_continuity_requested)
        v15.validate_current_gate(gate, root=repo, require_release=False)
        for field, reason in (
                ("source_snapshot_sha256", "source snapshot identity mismatch"),
                ("prelaunch_audit_sha256", "immutable prelaunch audit bytes differ"),
                ("approved_candidate_core_sha256", "candidate core digest differs"),
                ("staged_go_sha256", "staged-GO record bytes differ")):
            altered = dict(gate)
            altered[field] = "0" * 64
            with pytest.raises(PermissionError, match=reason):
                v15.validate_current_gate(altered, root=repo, require_release=False)
        gate_path = repo / v15.CANONICAL_GATE_REL
        runs = repo / "experiments/stage_F0_outputs/runs"
        queue_doc = json.loads((repo / v15.QUEUE_REL).read_text())
        from stage_F0_postabnormal_v14 import H51_RUN_IDS
        real_query_gpu_memory = v15.query_gpu_memory
        real_monotonic = time.monotonic
        controlled_clock = {"offset": 0.0}
        def controlled_monotonic():
            return real_monotonic() + controlled_clock["offset"]
        # Model an interruption beyond the 60-second telemetry limit without
        # sleeping. Only the process-local monotonic clock advances; immutable
        # audit/PREPARED bytes are asserted below.
        monkeypatch.setattr(v15.time, "monotonic", controlled_monotonic)
        telemetry_queries = []
        runtime_used = {"default": 100, "sequence": []}
        def isolated_query(*, gpu_index, expected_uuid, provider=None):
            telemetry_queries.append({"index": gpu_index, "uuid": expected_uuid})
            if provider is not None:
                fixture_provider = provider
            else:
                used = (runtime_used["sequence"].pop(0) if runtime_used["sequence"]
                        else runtime_used["default"])
                fixture_provider = lambda index: {"index": index, "uuid": expected_uuid,
                    "memory_total_bytes": 1000, "memory_used_bytes": used}
            return real_query_gpu_memory(gpu_index=gpu_index,
                expected_uuid=expected_uuid, provider=fixture_provider)
        monkeypatch.setattr(v15, "query_gpu_memory", isolated_query)
        release_payload = {
            "schema": v15.RELEASE_SCHEMA, "release_id": prepared["transaction_id"],
            "nonce": prepared["nonce"], "approved_by": prepared["approved_by"],
            "approved_by_role": "principal", "principal_intent": v15.SCOPE,
            "per_device_resource_rule": v15.PER_DEVICE_RULE,
            "approval_utc": prepared["approval_utc"],
            "source_incident_hold_sha256": v15.INCIDENT_HOLD_SHA256,
            "source_failure_nonce": v15.INCIDENT_NONCE,
            "source_snapshot_sha256": source_doc["snapshot_sha256"],
            "prelaunch_audit_sha256": prelaunch_sha,
            "targets": prepared["targets"], "authorization": prepared["authorization"],
            "pending_gate_sha256": v15.sha256_bytes(v15._canonical_indented_bytes(gate)),
        }
        def fresh_live_observation():
            rows = []
            for target in prepared["targets"]:
                telemetry = v15.query_gpu_memory(gpu_index=target["index"],
                    expected_uuid=target["uuid"], provider=lambda index, uuid=target["uuid"]:
                        {"index": index, "uuid": uuid,
                         "memory_total_bytes": 1000, "memory_used_bytes": 100})
                rows.append({**telemetry, "selected": True})
            return {"devices": rows,
                    "observation_started_monotonic": min(row[
                        "observation_started_monotonic"] for row in rows),
                    "observed_at_utc": max(row["observed_at_utc"] for row in rows)}
        stale_caller_observation = {"devices": [], "observation_started_monotonic": -1}
        prelaunch_path = repo / gate["prelaunch_audit_path"]
        source_snapshot_path = repo / gate["source_snapshot_path"]
        immutable_prelaunch_bytes = prelaunch_path.read_bytes()
        immutable_source_snapshot_bytes = source_snapshot_path.read_bytes()
        immutable_prelaunch_sha256 = hashlib.sha256(immutable_prelaunch_bytes).hexdigest()
        immutable_source_snapshot_sha256 = hashlib.sha256(
            immutable_source_snapshot_bytes).hexdigest()
        aged_observation = fresh_live_observation()
        aged_observation_started = min(row["observation_started_monotonic"]
                                       for row in aged_observation["devices"])
        runtime_used["sequence"] = [100]
        with pytest.raises(RuntimeError, match="fault injection after PREPARED"):
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release_payload,
                live_observation=stale_caller_observation, fault_after="PREPARED")
        ledger = runs / v15.RELEASE_LEDGER_REL.split("runs/", 1)[1]
        prepared_ledger_bytes = ledger.read_bytes()
        prepared_row_bytes = prepared_ledger_bytes.splitlines(keepends=True)[0]
        assert json.loads(prepared_row_bytes)["event"] == "PREPARED"
        assert prelaunch_path.read_bytes() == immutable_prelaunch_bytes
        assert source_snapshot_path.read_bytes() == immutable_source_snapshot_bytes
        assert hashlib.sha256(immutable_prelaunch_bytes).hexdigest() == immutable_prelaunch_sha256
        assert hashlib.sha256(immutable_source_snapshot_bytes).hexdigest() == immutable_source_snapshot_sha256
        # Advance the synthetic monotonic clock by more than the design's
        # freshness window. This does not alter immutable record timestamps.
        controlled_clock["offset"] += 61.25
        assert controlled_monotonic() - aged_observation_started > 60.0
        assert not v15.observation_is_fresh(started_monotonic=aged_observation_started)
        target_pairs = {(row["index"], row["uuid"]) for row in prepared["targets"]}
        assert all((row["index"], row["uuid"]) in target_pairs
                   for row in aged_observation["devices"])
        queries_before_aged_reentry = len(telemetry_queries)
        hold_path = runs / v15.HOLD_REL.split("runs/", 1)[1]
        hold_before_no_go = hold_path.read_bytes()
        runtime_used["sequence"] = ([100, 100, 100, 900]
                                     if peer_continuity_requested else [100, 900])
        with pytest.raises(v15.ResourceDeferred) as deferred:
            v15.commit_principal_release(runs_root=runs, prepared=prepared,
                gate_pending=gate, release_payload=release_payload,
                live_observation=aged_observation)
        assert deferred.value.boundary == "PRE_HOLD_RELEASE_DECISION"
        control_dir = runs / "execution_control"
        release_path = control_dir / "global_hold_release_v15.json"
        pending_path = control_dir / ("execution_isolation_gate_v15_pending_"
                                      + prepared["nonce"] + ".json")
        release_pointer = control_dir / "global_hold_release.json"
        final_archive = control_dir / ("execution_isolation_gate_v15_released_"
                                       + prepared["nonce"] + ".json")
        assert [json.loads(line)["event"] for line in ledger.read_bytes().splitlines()] == [
            "PREPARED", "RELEASE_ARTIFACT_DURABLE", "PENDING_GATE_DURABLE",
            "RELEASE_POINTER_DURABLE"]
        assert ledger.read_bytes().startswith(prepared_row_bytes)
        assert release_path.read_bytes() == v15._canonical_indented_bytes(release_payload)
        assert pending_path.read_bytes() == v15._canonical_indented_bytes(gate)
        assert release_pointer.read_bytes() == v15._canonical_indented_bytes(release_payload)
        assert gate_path.read_bytes() == v15._canonical_indented_bytes(gate)
        assert not final_archive.exists()
        assert json.loads(gate_path.read_text())["launch_authorized"] is False
        assert all(event not in [json.loads(line)["event"]
                                for line in ledger.read_bytes().splitlines()]
                   for event in ("HOLD_RELEASED", "COMMITTED"))
        assert prelaunch_path.read_bytes() == immutable_prelaunch_bytes
        assert source_snapshot_path.read_bytes() == immutable_source_snapshot_bytes
        assert hashlib.sha256(prelaunch_path.read_bytes()).hexdigest() == immutable_prelaunch_sha256
        assert hashlib.sha256(source_snapshot_path.read_bytes()).hexdigest() == immutable_source_snapshot_sha256
        assert telemetry_queries
        assert all((row["index"], row["uuid"]) in target_pairs for row in telemetry_queries)
        assert len(telemetry_queries) - queries_before_aged_reentry >= len(target_pairs)
        assert hold_path.read_bytes() == hold_before_no_go
        assert json.loads(hold_path.read_text())["active"] is True
        observation_path = runs / v15.OBSERVATION_LEDGER_REL.split("runs/", 1)[1]
        observations_before_recovery = v15.validate_observation_ledger(observation_path)
        assert observations_before_recovery[-1]["go"] is False
        assert observations_before_recovery[-1]["reason"] == "ASSIGNED_GPU_AT_OR_ABOVE_90_PERCENT"
        runtime_used["sequence"] = []
        queries_before_eligible_reentry = len(telemetry_queries)
        # The same immutable PREPARED identity resumes after the target's
        # no-go observation; the complete syscall restart matrix is exercised
        # independently from PREPARED-only clones above.
        result = v15.commit_principal_release(runs_root=runs, prepared=prepared,
            gate_pending=gate, release_payload=release_payload,
            live_observation=aged_observation)
        assert result["status"] == "COMMITTED"
        assert len(telemetry_queries) - queries_before_eligible_reentry >= len(target_pairs)
        assert all((row["index"], row["uuid"]) in target_pairs
                   for row in telemetry_queries[queries_before_eligible_reentry:])
        committed_bytes = ledger.read_bytes()
        assert committed_bytes.startswith(prepared_row_bytes)
        assert prelaunch_path.read_bytes() == immutable_prelaunch_bytes
        assert source_snapshot_path.read_bytes() == immutable_source_snapshot_bytes
        assert hashlib.sha256(prelaunch_path.read_bytes()).hexdigest() == immutable_prelaunch_sha256
        assert hashlib.sha256(source_snapshot_path.read_bytes()).hexdigest() == immutable_source_snapshot_sha256
        committed_digest = json.loads(gate_path.read_text())["committed_ledger_sha256"]
        observations = runs / v15.OBSERVATION_LEDGER_REL.split("runs/", 1)[1]
        observation_bytes = observations.read_bytes() if observations.exists() else None
        terminal_artifacts = {
            "release": release_path.read_bytes(),
            "pending": pending_path.read_bytes(),
            "release_pointer": release_pointer.read_bytes(),
            "final_archive": final_archive.read_bytes(),
            "hold": hold_path.read_bytes(),
            "canonical_gate": gate_path.read_bytes(),
        }
        telemetry_query_count_at_commit = len(telemetry_queries)
        assert v15.commit_principal_release(runs_root=runs, prepared=prepared,
            gate_pending=gate, release_payload=release_payload,
            live_observation={"devices": [], "observation_started_monotonic": -1})[
                "status"] == "COMMITTED_IDEMPOTENT"
        assert ledger.read_bytes() == committed_bytes
        assert release_path.read_bytes() == terminal_artifacts["release"]
        assert pending_path.read_bytes() == terminal_artifacts["pending"]
        assert release_pointer.read_bytes() == terminal_artifacts["release_pointer"]
        assert final_archive.read_bytes() == terminal_artifacts["final_archive"]
        assert hold_path.read_bytes() == terminal_artifacts["hold"]
        assert gate_path.read_bytes() == terminal_artifacts["canonical_gate"]
        assert json.loads(gate_path.read_text())["committed_ledger_sha256"] == committed_digest
        assert (observations.read_bytes() if observations.exists() else None) == observation_bytes
        assert len(telemetry_queries) == telemetry_query_count_at_commit
        from stage_F0_worker_control import validate_execution_gate
        validated = validate_execution_gate(root=repo, gate_path=gate_path,
            search_gate_path=repo / v15.SEARCH_GATE_REL)
        assert validated["execution_gate"]["launch_authorized"] is True

        # Exercise both resource deferral boundaries through the real
        # supervisor -> trajectory-child -> stage_F0_run path.  The isolated
        # child wrapper injects telemetry only inside this clone; it still
        # delegates bootstrap, inherited-slot validation, phase logging,
        # receipt authentication, and runner execution to the real modules.
        child_wrapper = tmp_path / "v15_race_child.py"
        child_wrapper.write_text("""\
import faulthandler, json, os, sys
from pathlib import Path
dump_seconds = os.environ.get('V15_TEST_FAULTHANDLER_SECONDS')
if dump_seconds:
    faulthandler.dump_traceback_later(float(dump_seconds), repeat=True, file=sys.stderr)
sys.path.insert(0, os.environ['V15_CLONE_EXPERIMENTS'])
sys.path.insert(0, os.environ['V15_CLONE_RUNS'])
import stage_F0_successor_v15 as v15
target_boundary = os.environ['V15_DEFER_BOUNDARY']
def fake_live(*, runs_root, gpu_index, gpu_uuid, boundary, provider=None):
    peer_mode = os.environ.get('V15_TEST_PEER_MODE') == '1'
    used = 950 if not peer_mode and boundary == target_boundary else 100
    telemetry = v15.query_gpu_memory(gpu_index=gpu_index, expected_uuid=gpu_uuid,
        provider=lambda index: {'index': index, 'uuid': gpu_uuid,
            'memory_total_bytes': 1000, 'memory_used_bytes': used})
    if not peer_mode and boundary == target_boundary:
        if os.environ.get('V15_PEER_COORDINATED') == '1':
            import time
            ready = Path(os.environ['V15_DEFER_READY'])
            if not ready.exists():
                ready.write_text(json.dumps({'phase': 'resource_defer_decision',
                    'boundary': boundary, 'pid': os.getpid(),
                    'monotonic': time.monotonic()}) + '\\n')
        raise v15.ResourceDeferred(telemetry, boundary)
    return telemetry
v15.require_live_gpu_memory = fake_live
import stage_F0_run
stage_F0_run.runtime_preflight = lambda _root: {
    'cuda_available': True, 'device_count': 2,
    'resource_concurrency_policy': {'usable_gpu_indices': [0, 1]},
    'gpus': []}
import types
stage_F0_run.load_b1_bundles = lambda _path: {'sentinel': types.SimpleNamespace(
    preprocessing_metadata={'raw_file_sha256': os.environ['V15_DATA_HASH']})}
if os.environ.get('V15_TEST_PEER_MODE') == '1':
    import json, time
    original_phase = stage_F0_run._execution_phase
    def peer_hold_after_real_claim(phase, **fields):
        original_phase(phase, **fields)
        if phase == 'canonical_claim_before':
            waiting = Path(os.environ['V15_PEER_WAITING'])
            waiting.write_text(json.dumps({**fields, 'monotonic': time.monotonic()},
                sort_keys=True) + '\\n')
            snapshot_done = Path(os.environ['V15_SNAPSHOT_DONE'])
            deadline = time.monotonic() + 600
            while not snapshot_done.is_file() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not snapshot_done.is_file():
                raise RuntimeError('deferred supervisor snapshot was not captured')
        if phase == 'canonical_claim_after' and fields.get('claimed') is True:
            marker = Path(os.environ['V15_PEER_READY'])
            marker.write_text(json.dumps({**fields, 'monotonic': time.monotonic()},
                sort_keys=True) + '\\n')
            progress = Path(os.environ['V15_PEER_PROGRESS'])
            stop = Path(os.environ['V15_PEER_STOP'])
            count = 0
            while not stop.exists():
                count += 1
                with progress.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'count': count,
                        'monotonic': time.monotonic()}) + '\\n')
                    handle.flush()
                time.sleep(0.1)
    stage_F0_run._execution_phase = peer_hold_after_real_claim
from run_search_trajectory_child import main
raise SystemExit(main())
""", encoding="utf-8")
        supervisor_wrapper = tmp_path / "v15_supervisor_run.py"
        supervisor_wrapper.write_text("""\
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ['V15_CLONE_EXPERIMENTS'])
sys.path.insert(0, os.environ['V15_CLONE_RUNS'])
import stage_F0_successor_v15 as v15
import run_search_claim_worker as worker
boundary = os.environ['V15_DEFER_BOUNDARY']
worker_id = os.environ.get('V15_TEST_WORKER_ID', '0')
if worker_id == '0' and os.environ.get('V15_PEER_COORDINATED') == '1':
    real_capture = v15.capture_f140_state
    initial_snapshot_recorded = {'value': False}
    def capture_then_signal(root, entries):
        state = real_capture(root, entries)
        if initial_snapshot_recorded['value']:
            return state
        snapshot_bytes = __import__('json').dumps(state, sort_keys=True) + '\\n'
        Path(os.environ['V15_SNAPSHOT_BYTES']).write_text(snapshot_bytes)
        import hashlib, time
        Path(os.environ['V15_SNAPSHOT_DONE']).write_text(__import__('json').dumps({
            'phase': 'pre_spawn_snapshot_captured', 'monotonic': time.monotonic(),
            'snapshot_sha256': hashlib.sha256(snapshot_bytes.encode()).hexdigest(),
            'snapshot_path': os.environ['V15_SNAPSHOT_BYTES']}, sort_keys=True) + '\\n')
        peer_ready = Path(os.environ['V15_PEER_READY'])
        deadline = time.monotonic() + 600
        while not peer_ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not peer_ready.is_file():
            raise RuntimeError('peer did not publish post-snapshot canonical claim')
        Path(os.environ['V15_DEFER_RESUMED']).write_text(__import__('json').dumps({
            'phase': 'supervisor_snapshot_wait_complete',
            'monotonic': time.monotonic(),
            'snapshot_sha256': hashlib.sha256(snapshot_bytes.encode()).hexdigest()},
            sort_keys=True) + '\\n')
        initial_snapshot_recorded['value'] = True
        return state
    v15.capture_f140_state = capture_then_signal
    real_auth = v15._authenticated_peer_progress
    def record_peer_auth(root, run_id, status, entry):
        import sys
        trace_lines = []
        trace_returns = []
        target_code = real_auth.__code__
        def trace_target(frame, event, arg):
            if frame.f_code is target_code:
                if event == 'line':
                    trace_lines.append(frame.f_lineno)
                elif event == 'return':
                    trace_returns.append({'line': frame.f_lineno,
                        'result': bool(arg) if isinstance(arg, bool) else None})
                return trace_target
            return None
        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace_target)
            accepted = real_auth(root, run_id, status, entry)
        finally:
            sys.settrace(previous_trace)
        row = {'run_id': run_id, 'status': status.get('status'),
            'accepted': bool(accepted), 'trace_lines': trace_lines,
            'trace_returns': trace_returns}
        with open(os.environ['V15_PEER_AUTH_DECISIONS'], 'a', encoding='utf-8') as handle:
            handle.write(__import__('json').dumps(row, sort_keys=True) + '\\n')
            handle.flush()
        return accepted
    v15._authenticated_peer_progress = record_peer_auth
    with open(os.environ['V15_PEER_AUTH_DECISIONS'], 'a', encoding='utf-8') as handle:
        handle.write(__import__('json').dumps({'event': 'hook_installed',
            'worker_id': worker_id}, sort_keys=True) + '\\n')
        handle.flush()
def live(*, runs_root, gpu_index, gpu_uuid, boundary, provider=None):
    # Supervisor pre-spawn is GO; the real child wrapper injects the selected
    # race boundary immediately before claim/checkpoint work.
    return v15.query_gpu_memory(gpu_index=gpu_index, expected_uuid=gpu_uuid,
        provider=lambda index: {'index': index, 'uuid': gpu_uuid,
            'memory_total_bytes': 1000, 'memory_used_bytes': 100})
v15.require_live_gpu_memory = live
from stage_F0_queue import acquire_process_slot
scheduler = os.environ['F0_PROCESS_SCHEDULER_ROOT']
supervisor_slot = acquire_process_slot(role='F0-supervisor-' + worker_id, blocking=False,
    scheduler_dir=scheduler)
child_slot = acquire_process_slot(role='F0-trajectory-child-' + worker_id, blocking=False,
    scheduler_dir=scheduler)
repo = Path(os.environ['V15_CLONE_ROOT'])
runs = Path(os.environ['V15_CLONE_RUNS'])
controller = worker.TrajectorySupervisor(worker_id=worker_id,
    supervisor_slot_fd=supervisor_slot.lock.handle.fileno(),
    child_slot_fd=child_slot.lock.handle.fileno(),
    supervisor_slot_index=supervisor_slot.index,
    child_slot_index=child_slot.index,
    queue_path=repo / os.environ['V15_QUEUE_REL'],
    search_gate_path=repo / os.environ['V15_SEARCH_GATE_REL'],
    execution_gate_path=repo / os.environ['V15_GATE_REL'],
    data_path=Path(os.environ['V15_EMPTY_DATA']), output_root=runs,
    python=Path(sys.executable), child_entry=Path(os.environ['V15_CHILD_WRAPPER']))
raise SystemExit(controller.run())
""", encoding="utf-8")
        import run_search_claim_worker as worker_module
        frozen_f140 = set(v15.frozen_f140_ids(repo))
        before_races = v15.capture_f140_state(runs,
            [row for row in queue_doc["entries"]
             if row["run_id"] in frozen_f140])
        race_nonces = set()
        worker_log_path = runs / "worker_logs/audited_worker_0.jsonl"
        worker_log_prefix = worker_log_path.read_bytes()
        empty_data = repo / "experiments/stage_F0_outputs/runs/v15-empty-data-sentinel.csv"
        empty_data.write_bytes(b"")
        scheduler_root = tmp_path / "v15-supervisor-scheduler"
        scheduler_root.mkdir()
        requested_boundary = os.environ.get("V15_TEST_SINGLE_BOUNDARY")
        if requested_boundary not in (None, "CHILD_PRECLAIM", "SERIALIZED_PRECLAIM_MEMORY_CHECK"):
            raise AssertionError("unknown diagnostic boundary: " + str(requested_boundary))
        boundaries = ((requested_boundary,) if requested_boundary else
                      ("CHILD_PRECLAIM", "SERIALIZED_PRECLAIM_MEMORY_CHECK"))
        peer_continuity = os.environ.get("V15_TEST_PEER_CONTINUITY") == "1"
        if peer_continuity and len(boundaries) != 1:
            raise AssertionError("peer-continuity fixture requires one selected deferral boundary")
        supervisor_timeout_seconds = int(os.environ.get(
            "V15_TEST_SUPERVISOR_TIMEOUT_SECONDS", "180"))
        if not 1 <= supervisor_timeout_seconds <= 600:
            raise AssertionError("test supervisor timeout must be between 1 and 600 seconds")
        for boundary in boundaries:
            # Initialize every diagnostic path before spawning either process.
            # A non-75 return must be archived even when a wrapper never reached
            # one of its phase markers.
            peer_ready = peer_progress = peer_stop_path = None
            snapshot_done = snapshot_bytes = defer_ready = peer_waiting = None
            defer_resumed = peer_auth_decisions = None
            env = dict(os.environ)
            env.update({
                "F0_PROCESS_SCHEDULER_ROOT": str(scheduler_root),
                "V15_CLONE_ROOT": str(repo),
                "V15_CLONE_EXPERIMENTS": str(repo / "experiments"),
                "V15_CLONE_RUNS": str(runs),
                "V15_DEFER_BOUNDARY": boundary,
                "V15_DATA_HASH": str(json.loads(
                    (repo / v15.SEARCH_GATE_REL).read_text())["dataset_file_sha256"]),
                "V15_QUEUE_REL": v15.QUEUE_REL,
                "V15_SEARCH_GATE_REL": v15.SEARCH_GATE_REL,
                "V15_GATE_REL": v15.CANONICAL_GATE_REL,
                "V15_EMPTY_DATA": str(empty_data),
                "V15_CHILD_WRAPPER": str(child_wrapper),
            })
            supervisor_command = [sys.executable, str(supervisor_wrapper)]
            supervisor_process = None
            if peer_continuity:
                # Start the deferred supervisor first. Its real pre-spawn
                # capture is the snapshot against which the peer's later
                # CLAIMED mutation must be authenticated.
                peer_ready = tmp_path / "v15-live-peer-ready.json"
                peer_progress = tmp_path / "v15-live-peer-heartbeats.jsonl"
                peer_stop_path = tmp_path / "v15-live-peer-stop"
                snapshot_done = tmp_path / "v15-deferred-snapshot-done"
                snapshot_bytes = tmp_path / "v15-deferred-snapshot.json"
                defer_ready = tmp_path / "v15-deferred-child-waiting.json"
                peer_waiting = tmp_path / "v15-peer-before-claim.json"
                defer_resumed = tmp_path / "v15-deferred-resumed-after-peer.json"
                peer_auth_decisions = tmp_path / "v15-peer-auth-decisions.jsonl"
                env.update({"V15_TEST_WORKER_ID": "0", "V15_PEER_COORDINATED": "1",
                    "V15_PEER_READY": str(peer_ready),
                    "V15_PEER_PROGRESS": str(peer_progress),
                    "V15_PEER_STOP": str(peer_stop_path),
                    "V15_SNAPSHOT_DONE": str(snapshot_done),
                    "V15_SNAPSHOT_BYTES": str(snapshot_bytes),
                    "V15_DEFER_READY": str(defer_ready),
                    "V15_PEER_WAITING": str(peer_waiting),
                    "V15_DEFER_RESUMED": str(defer_resumed),
                    "V15_PEER_AUTH_DECISIONS": str(peer_auth_decisions)})
                supervisor_process = subprocess.Popen(supervisor_command, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                snapshot_deadline = time.monotonic() + supervisor_timeout_seconds
                while (not snapshot_done.is_file() and supervisor_process.poll() is None
                       and time.monotonic() < snapshot_deadline):
                    time.sleep(0.05)
                if not snapshot_done.is_file():
                    stdout, stderr = supervisor_process.communicate(timeout=10)
                    raise AssertionError(("deferred supervisor did not capture its pre-spawn F140 snapshot",
                        supervisor_process.returncode, stdout, stderr))
                expected_snapshot = json.loads(snapshot_bytes.read_text(encoding="utf-8"))
                peer_ready = tmp_path / "v15-live-peer-ready.json"
                peer_progress = tmp_path / "v15-live-peer-heartbeats.jsonl"
                peer_stop_path = tmp_path / "v15-live-peer-stop"
                peer_env = dict(env)
                peer_env.update({"V15_TEST_WORKER_ID": "1",
                    "V15_TEST_PEER_MODE": "1",
                    "V15_PEER_READY": str(peer_ready),
                    "V15_PEER_PROGRESS": str(peer_progress),
                    "V15_PEER_STOP": str(peer_stop_path)})
                peer_log = runs / "worker_logs/audited_worker_1.jsonl"
                peer_log_prefix = peer_log.read_bytes()
                peer_supervisor = subprocess.Popen([sys.executable, str(supervisor_wrapper)],
                    env=peer_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                peer_deadline = time.monotonic() + supervisor_timeout_seconds
                while (not peer_ready.is_file() and peer_supervisor.poll() is None
                       and time.monotonic() < peer_deadline):
                    time.sleep(0.05)
                if not peer_ready.is_file():
                    peer_failure_evidence = tmp_path / "v15-peer-preclaim-failure"
                    peer_failure_evidence.mkdir(exist_ok=True)
                    for candidate in (runs / "worker_logs").glob("*"):
                        if (candidate.is_file() and not candidate.is_symlink()
                                and (candidate.name == "audited_worker_1.jsonl"
                                     or candidate.name.startswith("trajectory_worker1_")
                                     or candidate.name.startswith("worker_1"))):
                            shutil.copy2(candidate, peer_failure_evidence / candidate.name)
                    if peer_supervisor.poll() is None:
                        os.killpg(peer_supervisor.pid, signal.SIGKILL)
                    peer_stdout, peer_stderr = peer_supervisor.communicate(timeout=15)
                    raise AssertionError(("real peer failed to reach canonical_claim_after",
                        peer_supervisor.returncode, peer_stdout, peer_stderr,
                        str(peer_failure_evidence)))
                rendezvous = _wait_for_peer_rendezvous(
                    supervisor_process=supervisor_process,
                    marker_paths={"snapshot": snapshot_done,
                        "peer_wait": peer_waiting, "peer_claim": peer_ready,
                        "supervisor_resume": defer_resumed, "child_defer": defer_ready},
                    timeout_seconds=supervisor_timeout_seconds,
                    diagnostic_directory=tmp_path / "v15-peer-rendezvous-failure")
                snapshot_meta = rendezvous["snapshot"]
                peer_wait_meta = rendezvous["peer_wait"]
                peer_marker = rendezvous["peer_claim"]
                resume_meta = rendezvous["supervisor_resume"]
                defer_wait_meta = rendezvous["child_defer"]
                assert snapshot_meta["snapshot_sha256"] == v15.sha256_file(snapshot_bytes)
                peer_evidence = tmp_path / "v15-live-peer-continuity-evidence"
                peer_evidence.mkdir(exist_ok=True)
                shutil.copy2(peer_ready, peer_evidence / peer_ready.name)
                shutil.copy2(peer_progress, peer_evidence / peer_progress.name)
                shutil.copy2(peer_log, peer_evidence / peer_log.name)
                peer_new_bytes = peer_log.read_bytes()[len(peer_log_prefix):]
                peer_events = [json.loads(line) for line in
                    peer_new_bytes.decode("utf-8").splitlines() if line]
                peer_starts = [row for row in peer_events if row.get("event") == "CHILD_STARTED"]
                assert len(peer_starts) == 1, peer_events
                peer_start = peer_starts[0]
                peer_pid = int(peer_start["child_pid"])
                peer_nonce = str(peer_start["nonce"])
                peer_run_id = str(peer_marker["canonical"]["run_id"])
                assert peer_run_id in frozen_f140
                from stage_F0_worker_control import linux_process_identity
                peer_identity = linux_process_identity(peer_pid)
                assert int(peer_start["child_start_ticks"]) == int(peer_identity["start_ticks"])
                peer_entry = next(row for row in queue_doc["entries"]
                                  if row["run_id"] == peer_run_id)
                peer_paths = v15.fresh_member_paths(runs, peer_run_id)
                peer_status = json.loads(peer_paths["status"].read_text(encoding="utf-8"))
                peer_canonical_path = runs / "canonical_locks" / (peer_run_id + ".lock")
                peer_gpu_path = runs / "gpu_locks/gpu1.lock"
                peer_worker_path = runs / "worker_locks/worker_1.lock"
                peer_worker_control_path = runs / "worker_locks/worker_1.control"
                peer_canonical = json.loads(peer_canonical_path.read_text(encoding="utf-8"))
                peer_gpu = json.loads(peer_gpu_path.read_text(encoding="utf-8"))
                peer_worker = json.loads(peer_worker_path.read_text(encoding="utf-8"))
                peer_worker_control = peer_worker_control_path.read_text(
                    encoding="ascii").strip().split()
                (peer_evidence / "owner_and_locks.json").write_text(json.dumps({
                    "supervisor_pid": peer_supervisor.pid, "child_pid": peer_pid,
                    "child_start_ticks": peer_identity["start_ticks"],
                    "worker_nonce": peer_start["nonce"], "run_id": peer_run_id,
                    "status_owner": peer_status, "canonical_owner": peer_canonical,
                    "gpu_owner": peer_gpu, "worker_owner": peer_worker,
                    "worker_control_fields": peer_worker_control},
                    sort_keys=True, indent=2) + "\n", encoding="utf-8")
                for owner in (peer_status, peer_canonical):
                    assert owner["run_id"] == peer_run_id
                    assert owner["run_id_full_sha256"] == peer_entry["run_id_full_sha256"]
                    assert owner["canonical_key_json"] == peer_entry["canonical_key_json"]
                    assert owner["attempt_id"] == peer_status["attempt_id"]
                    assert int(owner["pid"]) == peer_pid
                assert peer_status["status"] == "CLAIMED"
                assert peer_status["nonce"] == peer_nonce
                assert int(peer_status["start_ticks"]) == int(peer_identity["start_ticks"])
                assert peer_canonical["nonce"] == peer_nonce
                assert peer_gpu["gpu_index"] == 1
                assert peer_gpu["gpu_uuid"] == "GPU-fixture-1"
                assert peer_worker["pid"] == peer_supervisor.pid
                assert peer_start["worker_id"] == "1"
                peer_supervisor_identity = linux_process_identity(peer_supervisor.pid)
                assert peer_worker["pgid"] == peer_supervisor.pid
                assert peer_worker["start_ticks"] == peer_supervisor_identity["start_ticks"]
                assert peer_worker["nonce"] == peer_worker_control[3]
                assert peer_worker_control == [str(peer_supervisor.pid),
                    str(peer_supervisor.pid), str(peer_supervisor_identity["start_ticks"]),
                    peer_worker["nonce"], peer_worker["host"], "1"]
                assert peer_gpu["pid"] == peer_pid
                assert peer_gpu["host"] == peer_status["host"]
                peer_lock_paths = (peer_canonical_path, peer_worker_path, peer_gpu_path)
                lock_checks = []
                for lock_path in peer_lock_paths:
                    fd = os.open(lock_path, os.O_RDWR)
                    try:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            lock_checks.append({"path": str(lock_path), "held": True,
                                "inode": os.fstat(fd).st_ino})
                        else:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                            lock_checks.append({"path": str(lock_path), "held": False,
                                "inode": os.fstat(fd).st_ino})
                    finally:
                        os.close(fd)
                assert all(row["held"] for row in lock_checks), lock_checks
                progress_before = len(peer_progress.read_text(encoding="utf-8").splitlines())
                peer_changed_snapshot = v15.capture_f140_state(runs,
                    [row for row in queue_doc["entries"]
                     if row["run_id"] in frozen_f140])
                assert peer_changed_snapshot != expected_snapshot, (
                    "peer claim did not change the deferred supervisor's frozen F140 snapshot")
                assert json.loads(snapshot_bytes.read_text(encoding="utf-8")) == expected_snapshot
            if supervisor_process is None:
                env["V15_TEST_WORKER_ID"] = "0"
                supervisor_process = subprocess.Popen(supervisor_command, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = supervisor_process.communicate(
                    timeout=supervisor_timeout_seconds)
                completed = subprocess.CompletedProcess(supervisor_command,
                    supervisor_process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                # Preserve clone-only evidence before the outer finally removes
                # the isolated tree. The timeout is deliberately unchanged; this
                # diagnostic copy lets us distinguish lock wait, validation,
                # and runner progress without touching production logs.
                evidence = tmp_path / ("v15-timeout-evidence-" + boundary)
                evidence.mkdir()
                started_rows = []
                audited_log = runs / "worker_logs/audited_worker_0.jsonl"
                if audited_log.is_file():
                    for line in audited_log.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get("event") == "CHILD_STARTED":
                            started_rows.append(event)
                child_pid = (started_rows[-1].get("child_pid")
                             if started_rows else None)
                proc_rows = {"boundary": boundary,
                             "supervisor_pid": supervisor_process.pid,
                             "child_pid": child_pid, "processes": {},
                             "proc_locks": None}
                for role, pid in (("supervisor", supervisor_process.pid),
                                  ("child", child_pid)):
                    if not isinstance(pid, int):
                        continue
                    fields = {}
                    for field in ("status", "wchan", "syscall", "stack"):
                        try:
                            fields[field] = Path("/proc/{}/{}".format(pid, field)).read_text()
                        except (OSError, UnicodeError) as exc:
                            fields[field] = "unavailable: {}".format(exc)
                    proc_rows["processes"][role] = fields
                try:
                    proc_rows["proc_locks"] = Path("/proc/locks").read_text()
                except OSError as exc:
                    proc_rows["proc_locks"] = "unavailable: {}".format(exc)
                (evidence / "live_process_snapshot.json").write_text(
                    json.dumps(proc_rows, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
                # The supervisor and runner child deliberately share a PGID;
                # terminate that isolated group only after capturing /proc.
                try:
                    os.killpg(supervisor_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    supervisor_process.kill()
                try:
                    supervisor_process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    supervisor_process.kill()
                    supervisor_process.communicate()
                clone_logs = runs / "worker_logs"
                evidence_logs = evidence / "worker_logs"
                evidence_logs.mkdir()
                for path in clone_logs.iterdir():
                    if (path.name == "audited_worker_0.jsonl"
                            or path.name.startswith("trajectory_worker0_1_")
                            or path.name.startswith("worker_0")):
                        if path.is_file() and not path.is_symlink():
                            shutil.copy2(path, evidence_logs / path.name)
                for rel in ("execution_control/execution_state.lock",
                            "worker_locks/worker_0.lock", "gpu_locks/gpu0.lock",
                            "gpu_locks/gpu1.lock"):
                    source = runs / rel
                    if source.is_file() and not source.is_symlink():
                        target = evidence / "locks" / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                snapshot_rows = []
                for directory in (runs / "queue_status", runs / "canonical_locks",
                                  runs / "gpu_locks", runs / "worker_locks"):
                    if not directory.is_dir():
                        continue
                    for path in sorted(directory.iterdir()):
                        try:
                            info = path.lstat()
                            snapshot_rows.append({"path": str(path.relative_to(repo)),
                                "mode": stat.S_IMODE(info.st_mode), "inode": info.st_ino,
                                "size": info.st_size, "is_symlink": path.is_symlink()})
                        except FileNotFoundError:
                            snapshot_rows.append({"path": str(path.relative_to(repo)),
                                                  "vanished_during_snapshot": True})
                (evidence / "resource_snapshot.json").write_text(
                    json.dumps({"boundary": boundary, "rows": snapshot_rows},
                               sort_keys=True, indent=2) + "\n", encoding="utf-8")
                raise
            if peer_continuity and completed.returncode != 75:
                # A non-deferral supervisor result is useful only with its
                # contemporaneous clone evidence. Copy it outside pytest's
                # cleanup tree before surfacing the original return code.
                diagnostic = peer_evidence_root / (
                    "worker0-return-{}".format(completed.returncode))
                marker_paths = {
                    "snapshot": snapshot_done,
                    "deferred_wait": defer_ready,
                    "peer_wait": peer_waiting,
                    "peer_claim": peer_ready,
                    "peer_resume": defer_resumed,
                    "peer_auth_decisions": peer_auth_decisions,
                }
                _write_non75_peer_diagnostic(diagnostic, boundary=boundary,
                    returncode=completed.returncode, stdout=completed.stdout,
                    stderr=completed.stderr, marker_paths=marker_paths)
                relevant_logs = [worker_log_path]
                if peer_continuity:
                    relevant_logs.append(runs / "worker_logs/audited_worker_1.jsonl")
                process_summary = []
                for log_path in relevant_logs:
                    worker_summary = {"audit_log": str(log_path),
                                      "status": "present" if log_path.is_file() else "missing",
                                      "children": []}
                    if log_path.is_file() and not log_path.is_symlink():
                        shutil.copy2(log_path, diagnostic / log_path.name)
                        try:
                            rows = [json.loads(line) for line in
                                log_path.read_text(encoding="utf-8").splitlines() if line]
                        except (OSError, ValueError):
                            rows = []
                        for row in rows:
                            if row.get("event") == "CHILD_STARTED":
                                child_summary = {"nonce": row.get("nonce"),
                                                 "pid": row.get("child_pid"),
                                                 "artifacts": {}}
                                for key in ("phase_path", "receipt_path", "stdout_path", "stderr_path"):
                                    candidate = row.get(key)
                                    if isinstance(candidate, str):
                                        source = Path(candidate)
                                        if source.is_file() and not source.is_symlink():
                                            shutil.copy2(source, diagnostic / source.name)
                                            child_summary["artifacts"][key] = {
                                                "status": "present", "path": candidate,
                                                "sha256": v15.sha256_file(source)}
                                            if key == "phase_path":
                                                try:
                                                    phases = [json.loads(line).get("phase")
                                                        for line in source.read_text(
                                                            encoding="utf-8").splitlines() if line]
                                                except (OSError, ValueError):
                                                    phases = None
                                                child_summary["phase_order"] = phases
                                                child_summary["last_phase"] = (
                                                    phases[-1] if phases else None)
                                                child_summary["phase_tail"] = (
                                                    phases[-8:] if phases else [])
                                                child_summary["training_before_seen"] = (
                                                    "training_before" in (phases or []))
                                        else:
                                            child_summary["artifacts"][key] = {
                                                "status": "missing", "path": candidate}
                                worker_summary["children"].append(child_summary)
                            failure_path = row.get("failure_path")
                            if isinstance(failure_path, str):
                                source = Path(failure_path)
                                if source.is_file() and not source.is_symlink():
                                    shutil.copy2(source, diagnostic / source.name)
                                    worker_summary.setdefault("failure_artifacts", []).append({
                                        "status": "present", "path": failure_path,
                                        "sha256": v15.sha256_file(source)})
                                else:
                                    worker_summary.setdefault("failure_artifacts", []).append({
                                        "status": "missing", "path": failure_path})
                    process_summary.append(worker_summary)
                (diagnostic / "process_summary.json").write_text(json.dumps(
                    process_summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                if peer_auth_decisions.is_file():
                    shutil.copy2(peer_auth_decisions, diagnostic / peer_auth_decisions.name)
                    try:
                        auth_rows = [json.loads(line) for line in
                            peer_auth_decisions.read_text(encoding="utf-8").splitlines()
                            if line]
                        import inspect
                        auth_source, auth_first_line = inspect.getsourcelines(
                            v15._authenticated_peer_progress)
                        trace_line_numbers = sorted({number
                            for row in auth_rows
                            for number in row.get("trace_lines", [])
                            if isinstance(number, int)})
                        source_map = {str(number): auth_source[number - auth_first_line].rstrip("\n")
                            for number in trace_line_numbers
                            if auth_first_line <= number < auth_first_line + len(auth_source)}
                        (diagnostic / "peer_auth_source_lines.json").write_text(json.dumps({
                            "source_path": inspect.getsourcefile(v15._authenticated_peer_progress),
                            "source_sha256": v15.sha256_file(Path(inspect.getsourcefile(
                                v15._authenticated_peer_progress))),
                            "line_text": source_map}, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
                    except (OSError, TypeError, ValueError, IndexError) as exc:
                        (diagnostic / "peer_auth_source_lines.unavailable").write_text(
                            "{}: {}\n".format(type(exc).__name__, exc), encoding="utf-8")
                else:
                    (diagnostic / "peer_auth_decisions.missing").write_text(
                        "auth decision log absent before cleanup\n", encoding="utf-8")
                state_inventory = []
                for rel in (v15.HOLD_REL, v15.CANONICAL_GATE_REL,
                            v15.RELEASE_REL, v15.RELEASE_LEDGER_REL,
                            v15.OBSERVATION_LEDGER_REL, v15.QUEUE_REL,
                            v15.SEARCH_GATE_REL,
                            "experiments/stage_F0_outputs/runs/execution_control/postabnormal_v14_release.jsonl"):
                    source = repo / rel
                    if source.is_file() and not source.is_symlink():
                        target = diagnostic / "state" / Path(rel).name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        state_inventory.append({"path": rel, "status": "present",
                            "size": source.stat().st_size,
                            "sha256": v15.sha256_file(source)})
                    else:
                        state_inventory.append({"path": rel, "status": "missing"})
                (diagnostic / "protected_state_inventory.json").write_text(json.dumps(
                    state_inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                resource_rows = []
                for directory in (runs / "queue_status", runs / "canonical_locks",
                                  runs / "gpu_locks", runs / "worker_locks"):
                    if not directory.is_dir():
                        continue
                    for source in sorted(directory.iterdir()):
                        try:
                            info = source.lstat()
                            if source.is_file() and not source.is_symlink():
                                target = diagnostic / "locks" / str(source.relative_to(runs))
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(source, target)
                            resource_rows.append({"path": str(source.relative_to(repo)),
                                "mode": stat.S_IMODE(info.st_mode), "inode": info.st_ino,
                                "size": info.st_size, "is_symlink": source.is_symlink(),
                                "sha256": (v15.sha256_file(source)
                                    if source.is_file() and not source.is_symlink() else None)})
                        except FileNotFoundError:
                            resource_rows.append({"path": str(source.relative_to(repo)),
                                "vanished_during_snapshot": True})
                (diagnostic / "resource_snapshot.json").write_text(json.dumps(
                    resource_rows, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                proc_snapshot = {"supervisors": [], "children": [], "live_peer_child": None,
                                 "lock_probe": [], "proc_locks": None}
                for log_path in relevant_logs:
                    if not log_path.is_file() or log_path.is_symlink():
                        continue
                    try:
                        rows = [json.loads(line) for line in
                            log_path.read_text(encoding="utf-8").splitlines() if line]
                    except (OSError, ValueError):
                        continue
                    for row in rows:
                        if row.get("event") != "CHILD_STARTED":
                            continue
                        for role, pid in (("supervisor", row.get("pid")),
                                          ("child", row.get("child_pid"))):
                            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                                continue
                            identity = {"role": role, "pid": pid,
                                "worker_id": row.get("worker_id"), "status": "unavailable"}
                            for field in ("status", "wchan", "syscall", "comm"):
                                try:
                                    identity[field] = Path("/proc/{}/{}".format(pid, field)).read_text()
                                    identity["status"] = "present"
                                except (OSError, UnicodeError) as exc:
                                    identity.setdefault("unavailable_fields", {})[field] = str(exc)
                            proc_snapshot["supervisors" if role == "supervisor"
                                          else "children"].append(identity)
                for rel in ("execution_control/execution_state.lock",
                            "worker_locks/worker_0.lock", "worker_locks/worker_1.lock",
                            "gpu_locks/gpu0.lock", "gpu_locks/gpu1.lock"):
                    lock_path = runs / rel
                    result = {"path": str(lock_path.relative_to(repo)),
                              "status": "missing"}
                    if lock_path.is_file() and not lock_path.is_symlink():
                        try:
                            fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                            try:
                                result.update({"status": "present",
                                    "inode": os.fstat(fd).st_ino,
                                    "sha256": v15.sha256_file(lock_path)})
                                try:
                                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    result["held_by_other"] = True
                                else:
                                    result["held_by_other"] = False
                                    fcntl.flock(fd, fcntl.LOCK_UN)
                            finally:
                                os.close(fd)
                        except OSError as exc:
                            result.update({"status": "probe_error", "error": str(exc)})
                    proc_snapshot["lock_probe"].append(result)
                try:
                    proc_snapshot["proc_locks"] = Path("/proc/locks").read_text()
                except OSError as exc:
                    proc_snapshot["proc_locks"] = "unavailable: {}".format(exc)
                (diagnostic / "process_and_lock_snapshot.json").write_text(json.dumps(
                    proc_snapshot, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                manifest = []
                for source in sorted(diagnostic.rglob("*")):
                    if source.is_file() and source.name != "SHA256SUMS.json":
                        manifest.append({"path": str(source.relative_to(diagnostic)),
                            "size": source.stat().st_size,
                            "sha256": v15.sha256_file(source)})
                (diagnostic / "SHA256SUMS.json").write_text(json.dumps(
                    manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            assert completed.returncode == 75, (boundary, completed.returncode,
                completed.stdout, completed.stderr)
            new_events = [json.loads(line) for line in
                worker_log_path.read_text().splitlines()[len(worker_log_prefix.decode().splitlines()):]]
            starts = [event for event in new_events if event.get("event") == "CHILD_STARTED"]
            authenticated = [event for event in new_events if event.get("event") ==
                "RESOURCE_DEFERRED_PRECLAIM_AUTHENTICATED"]
            assert len(starts) == len(authenticated) == 1, new_events
            assert not any(event.get("event") in ("CHILD_VALIDATION_COMPLETE",
                "WORKER_HALTED_ON_CHILD_FAILURE") for event in new_events)
            nonce = authenticated[0]["nonce"]
            race_nonces.add(nonce)
            assert authenticated[0]["automatic_respawn"] is False
            assert authenticated[0]["peer_stop"] is False
            assert starts[0]["nonce"] == nonce
            receipt_path = Path(starts[0]["receipt_path"])
            assert receipt_path.is_file()
            phases = [json.loads(line)["phase"] for line in
                Path(starts[0]["phase_path"]).read_text().splitlines()]
            assert "child_preclaim_entry" in phases
            if boundary == "SERIALIZED_PRECLAIM_MEMORY_CHECK":
                assert phases.index("child_preclaim_entry") < phases.index(
                    "serialized_preclaim_memory_check")
            post_deferral_snapshot = v15.capture_f140_state(runs,
                [row for row in queue_doc["entries"]
                 if row["run_id"] in frozen_f140])
            if peer_continuity:
                assert post_deferral_snapshot == peer_changed_snapshot
                resume_meta = json.loads(defer_resumed.read_text(encoding="utf-8"))
                assert resume_meta["monotonic"] > peer_marker["monotonic"]
                assert resume_meta["snapshot_sha256"] == snapshot_meta["snapshot_sha256"]
                auth_decisions = [json.loads(line) for line in
                    peer_auth_decisions.read_text(encoding="utf-8").splitlines() if line]
                accepted_peer_rows = [row for row in auth_decisions
                    if row.get("run_id") == peer_run_id and row.get("accepted") is True]
                assert accepted_peer_rows, auth_decisions
                peer_status_path = v15.fresh_member_paths(runs, peer_run_id)["status"]
                peer_canonical_path = v15.fresh_member_paths(runs, peer_run_id)["lock"]
                forged_before = {str(path): path.read_bytes() for path in (
                    peer_status_path, peer_canonical_path, peer_worker_path,
                    peer_worker_control_path, peer_gpu_path, peer_log)}
                forged_status = {**peer_status, "pid": peer_pid + 100000}
                assert v15._authenticated_peer_progress(
                    runs, peer_run_id, forged_status, peer_entry) is False
                forged_after = {str(path): path.read_bytes() for path in (
                    peer_status_path, peer_canonical_path, peer_worker_path,
                    peer_worker_control_path, peer_gpu_path, peer_log)}
                assert forged_after == forged_before
                (peer_evidence / "forged_identity_rejection.json").write_text(json.dumps({
                    "result": "rejected", "forged_pid": forged_status["pid"],
                    "mutation": False}, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
                (peer_evidence / "authenticated_peer_decisions.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in auth_decisions), encoding="utf-8")
                (peer_evidence / "snapshot_before_peer.json").write_text(
                    json.dumps(expected_snapshot, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
                (peer_evidence / "snapshot_after_peer.json").write_text(
                    json.dumps(peer_changed_snapshot, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
                (peer_evidence / "race_order.json").write_text(json.dumps({
                    "snapshot": snapshot_meta, "deferred_child_wait": defer_wait_meta,
                    "peer_before_claim": peer_wait_meta, "peer_claimed": peer_marker,
                    "deferred_child_resumed": resume_meta}, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
            else:
                assert post_deferral_snapshot == before_races
            hold_state = json.loads((runs / "execution_control/global_hold.json").read_text())
            assert hold_state.get("active") is False
            if peer_continuity:
                assert peer_supervisor is not None and peer_supervisor.poll() is None
                assert linux_process_identity(peer_pid)["start_ticks"] == peer_identity["start_ticks"]
                progress_after = len(peer_progress.read_text(encoding="utf-8").splitlines())
                assert progress_after > progress_before, (progress_before, progress_after)
                peer_events_after = [json.loads(line) for line in
                    peer_log.read_bytes()[len(peer_log_prefix):].decode("utf-8").splitlines()
                    if line]
                assert len([row for row in peer_events_after
                            if row.get("event") == "CHILD_STARTED"]) == 1
                assert not any(row.get("event") in ("SIGNAL_FORWARDED_TO_CHILD",
                    "WORKER_HALTED_ON_CHILD_FAILURE", "WORKER_STOPPED_BEFORE_NEXT_CHILD")
                    for row in peer_events_after)
                assert peer_log.read_text(encoding="utf-8").count(peer_nonce) >= 1
                shutil.copy2(peer_ready, peer_evidence / peer_ready.name)
                shutil.copy2(peer_progress, peer_evidence / peer_progress.name)
                shutil.copy2(peer_log, peer_evidence / peer_log.name)
                shutil.copy2(Path(peer_start["phase_path"]),
                             peer_evidence / Path(peer_start["phase_path"]).name)
                peer_phase_rows = [json.loads(line) for line in
                    Path(peer_start["phase_path"]).read_text(encoding="utf-8").splitlines()
                    if line]
                assert "training_before" not in [row.get("phase") for row in peer_phase_rows]
                (peer_evidence / "owner_and_locks.json").write_text(json.dumps({
                    "supervisor_pid": peer_supervisor.pid, "child_pid": peer_pid,
                    "child_start_ticks": peer_identity["start_ticks"],
                    "worker_nonce": peer_start["nonce"], "run_id": peer_run_id,
                    "run_id_full_sha256": peer_entry["run_id_full_sha256"],
                    "canonical_key_json": peer_entry["canonical_key_json"],
                    "attempt_id": peer_status["attempt_id"],
                    "worker_owner": peer_worker, "gpu_owner": peer_gpu,
                    "canonical_owner": peer_canonical, "held_locks": lock_checks,
                    "heartbeat_before": progress_before, "heartbeat_after": progress_after,
                    "deferred_supervisor_return_code": completed.returncode,
                    "deferred_event": authenticated[0]}, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
            branch_evidence = tmp_path / ("v15-boundary-evidence-" + boundary)
            branch_evidence.mkdir(exist_ok=True)
            for key in ("phase_path", "receipt_path", "stdout_path", "stderr_path"):
                source = Path(starts[0][key])
                if source.is_file() and not source.is_symlink():
                    shutil.copy2(source, branch_evidence / source.name)
            shutil.copy2(worker_log_path, branch_evidence / worker_log_path.name)
            if peer_continuity:
                for source in peer_evidence.iterdir():
                    if source.is_file() and not source.is_symlink():
                        shutil.copy2(source, peer_evidence_root / source.name)
                for source in branch_evidence.iterdir():
                    if source.is_file() and not source.is_symlink():
                        shutil.copy2(source, peer_evidence_root / source.name)
                evidence_manifest = []
                for source in sorted(peer_evidence_root.iterdir()):
                    if source.is_file() and not source.is_symlink():
                        evidence_manifest.append({"name": source.name,
                            "size": source.stat().st_size,
                            "sha256": v15.sha256_file(source)})
                (peer_evidence_root / "manifest.json").write_text(
                    json.dumps(evidence_manifest, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
            worker_log_prefix = worker_log_path.read_bytes()
        child_events = [json.loads(line) for line in
            worker_log_path.read_bytes()[len(worker_log_prefix):].decode("utf-8").splitlines()]
        assert len(race_nonces) == len(boundaries)
        assert {name: v15.sha256_file(v15.RUNS / "worker_logs" / name)
                for name in source_log_hashes} == source_log_hashes
        if requested_boundary:
            if peer_continuity:
                assert peer_stop_path is not None
                os.killpg(peer_supervisor.pid, signal.SIGKILL)
                peer_supervisor.communicate(timeout=15)
                peer_supervisor = None
            return

        from stage_F0_queue import claim_next_entry
        frozen_f140 = set(v15.frozen_f140_ids(repo))
        f140_candidate = next(row for row in queue_doc["entries"]
                              if row["run_id"] in frozen_f140)
        out_of_f140 = next(row for row in queue_doc["entries"]
                           if row["run_id"] not in frozen_f140)
        v15.validate_fresh_entry(runs, f140_candidate, sorted(frozen_f140))
        with pytest.raises(PermissionError, match="outside frozen F140"):
            v15.validate_fresh_entry(runs, out_of_f140, sorted(frozen_f140))
        untouched_before = {rid: (os.path.lexists(runs / "queue_status" / (rid + ".json")),
                                  os.path.lexists(runs / "canonical_locks" / (rid + ".lock")),
                                  os.path.lexists(runs / rid)) for rid in frozen_f140}
        hold_path = runs / "execution_control" / "global_hold.json"
        hold_backup = runs / "execution_control" / "global_hold.test-backup"
        os.replace(hold_path, hold_backup)
        try:
            with pytest.raises(PermissionError, match="missing"):
                claim_next_entry(queue_doc, str(runs), gpu_uuid=device["uuid"], gpu_index=0,
                    held_run_ids=H51_RUN_IDS,
                    gpu_telemetry_provider=lambda index: {"index": index,
                        "uuid": device["uuid"], "memory_total_bytes": 1000,
                        "memory_used_bytes": 100})
            untouched_after = {rid: (os.path.lexists(runs / "queue_status" / (rid + ".json")),
                                     os.path.lexists(runs / "canonical_locks" / (rid + ".lock")),
                                     os.path.lexists(runs / rid)) for rid in frozen_f140}
            assert untouched_after == untouched_before
        finally:
            os.replace(hold_backup, hold_path)
        claim, canonical_lock, _resume = claim_next_entry(queue_doc, str(runs),
            gpu_uuid=device["uuid"], gpu_index=0, held_run_ids=H51_RUN_IDS,
            gpu_telemetry_provider=lambda index: {"index": index, "uuid": device["uuid"],
                "memory_total_bytes": 1000, "memory_used_bytes": 100})
        try:
            assert claim["run_id"] in v15.frozen_f140_ids(repo)
            status = json.loads((runs / "queue_status" / (claim["run_id"] + ".json")).read_text())
            assert status["status"] == "CLAIMED"
        finally:
            canonical_lock.close()
    finally:
        if peer_supervisor is not None and peer_supervisor.poll() is None:
            try:
                os.killpg(peer_supervisor.pid, signal.SIGKILL)
                peer_supervisor.communicate(timeout=15)
            except ProcessLookupError:
                pass
        sys.path[:] = saved_path
        for name in ("stage_F0_postabnormal_v14", "stage_F0_migration", "stage_F0_data",
                     "stage_F0_protocol", "stage_F0_queue"):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        shutil.rmtree(repo)

