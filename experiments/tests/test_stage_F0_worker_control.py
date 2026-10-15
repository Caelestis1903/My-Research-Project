import json
import fcntl
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/home/xzf/miniconda3/envs/phase0-3-env/bin/python')
CHILD = ROOT / 'experiments/stage_F0_outputs/runs/run_search_trajectory_child.py'


def _reserve_two_test_slots():
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot

    return [acquire_process_slot(role='F0-control-process-test', blocking=False)
            for _ in range(2)]


def _identity(pid):
    raw = Path('/proc/{}/stat'.format(pid)).read_text(encoding='ascii')
    tail = raw[raw.rfind(')') + 2:].split()
    return {'state': tail[0], 'ppid': int(tail[1]), 'pgid': int(tail[2]),
            'start_ticks': int(tail[19])}


def _wait_for(predicate, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except (OSError, ValueError):
            pass
        time.sleep(0.025)
    return False


def _parent_helper_source():
    return r'''import json, os, socket, subprocess, sys, time
from pathlib import Path

mode, child_entry, scheduler, ready, receipt, phase, slot0, slot1, fd0, fd1, index0, index1, marker = sys.argv[1:]
fd0, fd1, index0, index1 = int(fd0), int(fd1), int(index0), int(index1)
os.setpgid(0, 0)
pgid = os.getpgrp()
fds = [(slot0, fd0, index0, "test-supervisor"),
       (slot1, fd1, index1, "test-child")]
for path, fd, index, role in fds:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "role": role, "slot": index,
                   "host": socket.gethostname()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
raw = Path("/proc/{}/stat".format(os.getpid())).read_text(encoding="ascii")
parent_ticks = int(raw[raw.rfind(")") + 2:].split()[19])
cmd = [sys.executable, child_entry,
       "--expected-parent-pid", str(os.getpid()),
       "--expected-parent-start-ticks", str(parent_ticks),
       "--expected-pgid", str(pgid), "--worker-id", "0", "--nonce", "control-test",
       "--phase-log", phase, "--receipt", receipt,
       "--child-slot-fd", str(fd1), "--child-slot-index", str(index1),
       "--scheduler-root", scheduler, "--child-slot-role", "test-child"]
cmd.extend(["--self-test-runner-import-marker", marker])
if mode == "guarded":
    cmd.extend(["--self-test-parent-guard-ready", ready])
else:
    cmd.extend(["--self-test-pre-pdeath-ready", ready,
                "--self-test-pre-pdeath-delay", "0.35"])
child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, pass_fds=(fd1,))
deadline = time.monotonic() + 5
while not Path(ready).is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
if not Path(ready).is_file():
    raise RuntimeError("child bootstrap readiness timed out")
print(json.dumps({"pid": os.getpid(), "pgid": pgid, "child_pid": child.pid}), flush=True)
if mode == "guarded":
    child.wait()
'''


def _supervisor_helper_source():
    return r'''import importlib.util, json, os, resource, signal, socket, sys, threading, time
from pathlib import Path

(runs, result_path, queue_path, search_gate, execution_gate, data_path,
 child_entry, scheduler, slot0, slot1, fd0, fd1, index0, index1,
 mode, signal_no) = sys.argv[1:]
fd0, fd1, index0, index1, signal_no = map(int, (fd0, fd1, index0, index1, signal_no))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.path.insert(0, os.environ["F0_RUNS_DIR"])
sys.path.insert(0, os.environ["F0_EXPERIMENTS_DIR"])
import run_search_claim_worker as controller
from stage_F0_worker_control import PRESERVED_H51_RUN_IDS
os.environ["F0_PROCESS_SCHEDULER_ROOT"] = scheduler
os.setpgid(0, 0)
for path, fd, index, role in ((slot0, fd0, index0, "F0-supervisor-0"),
                              (slot1, fd1, index1, "F0-trajectory-child-0")):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "role": role, "slot": index,
                   "host": socket.gethostname()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
Path(queue_path).write_text("{}\n", encoding="utf-8")
Path(search_gate).write_text("{}\n", encoding="utf-8")
Path(execution_gate).write_text("{}\n", encoding="utf-8")
Path(data_path).write_text("fixture\n", encoding="utf-8")
if mode == "malformed_status":
    status_dir = Path(runs) / "queue_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "malformed.json").write_text("{broken-status", encoding="utf-8")
if mode == "invalid_active_claim":
    def invalid_claim(**kwargs):
        raise PermissionError("fixture stale RUNNING claim")
    controller.validate_active_claims = invalid_claim
elif mode != "malformed_status":
    controller.validate_active_claims = lambda **kwargs: []
if mode == "ignore_term":
    def durable_peer_stop(worker_id, *, output_root):
        path = Path(output_root) / "worker_logs" / "peer-stop-request.json"
        controller._atomic_json(path, {"worker_id": "1", "requested_by": worker_id,
                                       "signal": "SIGTERM", "durable": True})
        return {"worker_id": "1", "signal": "SIGTERM", "sent": True,
                "request_path": str(path)}
    controller._stop_peer_worker = durable_peer_stop
controller._validate_completion_receipt = lambda *args, **kwargs: None
worker = controller.TrajectorySupervisor(
    worker_id="0", supervisor_slot_fd=fd0, child_slot_fd=fd1,
    supervisor_slot_index=index0, child_slot_index=index1,
    queue_path=Path(queue_path), search_gate_path=Path(search_gate),
    execution_gate_path=Path(execution_gate), data_path=Path(data_path),
    output_root=Path(runs), python=Path(sys.executable),
    child_entry=Path(child_entry),
    termination_timeout=(5.0 if mode == "duplicate_interrupt" else 0.5))
def frozen_gates():
    if mode == "stop_before_child" and signal_no:
        controller._on_signal(signal_no, None)
    return {
        "search_gate": {"blocked_search_keys": []},
        "execution_gate": {"preserved_held_run_ids": list(
            PRESERVED_H51_RUN_IDS)}}
worker._check_pre_child_gates = frozen_gates
if mode == "spawnfail":
    worker.python = Path(str(Path(runs) / "missing-python"))
if signal_no and mode != "stop_before_child":
    def trigger_after_child_handler_ready():
        deadline = time.monotonic() + 5
        marker = Path(os.environ["F0_TEST_CHILD_READY"])
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        controller._on_signal(signal_no, None)
    timer = threading.Thread(target=trigger_after_child_handler_ready, daemon=True)
    timer.start()
else:
    timer = None
code = worker.run()
if timer is not None:
    timer.join(timeout=2)
Path(result_path).write_text(json.dumps({"exit_code": code}), encoding="utf-8")
sys.exit(code)
'''


def _run_supervisor_fixture(tmp_path, mode, *, signal_no=0):
    leases = _reserve_two_test_slots()
    try:
        helper = tmp_path / 'supervisor_helper.py'
        fixture = tmp_path / 'child_fixture.py'
        helper.write_text(_supervisor_helper_source(), encoding='utf-8')
        fixture.write_text(
            'import os, signal, sys, time\n'
            'mode = os.environ.get("F0_TEST_CHILD_MODE", "empty")\n'
            'if mode == "abrt": os.write(2, b"native fixture abort\\n"); os.abort()\n'
            'if mode == "nonzero": os.write(2, b"native fixture nonzero\\n"); sys.exit(17)\n'
            'if mode == "signal_sleep":\n'
            '  def stop(sig, _frame): sys.exit(128 + sig)\n'
            '  for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, stop)\n'
            '  signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})\n'
            '  from pathlib import Path; Path(os.environ["F0_TEST_CHILD_READY"]).write_text("ready")\n'
            '  time.sleep(60)\n'
            'if mode == "ignore_term":\n'
            '  signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
            '  signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})\n'
            '  from pathlib import Path; Path(os.environ["F0_TEST_CHILD_READY"]).write_text("ready")\n'
            '  time.sleep(60)\n', encoding='utf-8')
        runs = tmp_path / 'runs'
        result_path = tmp_path / 'supervisor-result.json'
        queue_path = tmp_path / 'queue.json'
        search_gate = tmp_path / 'search-gate.json'
        execution_gate = tmp_path / 'execution-gate.json'
        data_path = tmp_path / 'fixture-data.csv'
        scheduler = Path(leases[0].lock.path).parent
        env = dict(os.environ)
        env.update({'F0_PROCESS_SCHEDULER_ROOT': str(scheduler),
                    'F0_TEST_CHILD_MODE': mode,
                    'F0_TEST_CHILD_READY': str(tmp_path / 'child-handler-ready'),
                    'F0_RUNS_DIR': str(ROOT / 'experiments/stage_F0_outputs/runs'),
                    'F0_EXPERIMENTS_DIR': str(ROOT / 'experiments')})
        proc = subprocess.run(
            [str(PYTHON), str(helper), str(runs), str(result_path), str(queue_path),
             str(search_gate), str(execution_gate), str(data_path), str(fixture),
             str(scheduler), leases[0].lock.path, leases[1].lock.path,
             str(leases[0].lock.handle.fileno()), str(leases[1].lock.handle.fileno()),
             str(leases[0].index), str(leases[1].index), mode, str(signal_no)],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
            close_fds=True,
            pass_fds=(leases[0].lock.handle.fileno(), leases[1].lock.handle.fileno()))
        if not result_path.is_file() and mode == 'ignore_term':
            return leases, runs, proc, proc.returncode
        if not result_path.is_file():
            raise AssertionError('supervisor fixture exited without result: {}'.format(
                proc.stderr[-4000:]))
        return_code = json.loads(result_path.read_text(encoding='utf-8'))['exit_code']
        return leases, runs, proc, return_code
    except BaseException:
        for lease in leases:
            lease.close()
        raise


def _start_duplicate_signal_fixture(tmp_path, *, fail_receipt=False):
    leases = _reserve_two_test_slots()
    try:
        helper = tmp_path / 'duplicate_signal_supervisor.py'
        fixture = tmp_path / 'duplicate_signal_child.py'
        helper.write_text(_supervisor_helper_source(), encoding='utf-8')
        fixture.write_text(r'''import json, os, signal, sys, time
from pathlib import Path

ROOT = Path('/home/xzf/Code/phase0-3')
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/tests'))
sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
import run_search_trajectory_child as child_control
import stage_F0_run as runner_module
from stage_F0_queue import (QueueFileLock, _atomic_json, adopt_reserved_process_slot,
                            queue_status_path)
from stage_F0_worker_control import append_phase_event
from test_stage_F0_run import _entry, _runner, _synthetic_bundle

def option(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]

supervisor_pid = int(option('--expected-parent-pid'))
nonce = option('--nonce')
worker_id = option('--worker-id')
phase_log = option('--phase-log')
receipt_path = option('--receipt')
child_slot_fd = int(option('--child-slot-fd'))
child_slot_index = int(option('--child-slot-index'))
scheduler_root = option('--scheduler-root')
child_slot_role = option('--child-slot-role')
output_root = Path(option('--output-root')).resolve()
slot_lease = adopt_reserved_process_slot(
    fd=child_slot_fd, index=child_slot_index, owner_pid=supervisor_pid,
    role=child_slot_role, scheduler_dir=scheduler_root)

bundle = _synthetic_bundle()
entry = _entry(bundle, max_epochs=2)
attempt_dir = output_root / entry['run_id'] / 'attempts' / entry['attempt_id']
status_file = Path(queue_status_path(str(output_root), entry['run_id']))
_atomic_json(str(status_file), {
    'run_id': entry['run_id'], 'run_id_full_sha256': entry['run_id_full_sha256'],
    'canonical_key_json': entry['canonical_key_json'], 'attempt_id': entry['attempt_id'],
    'pid': os.getpid(), 'host': __import__('socket').gethostname(),
    'gpu_index': 0, 'status': 'CLAIMED', 'recovery_count': 0,
})
runner = _runner(bundle, entry, attempt_dir)
original_train_epoch = runner._train_epoch
ready_path = Path(os.environ['F0_DUPLICATE_TRAINING_READY'])
manifest_barrier = Path(os.environ['F0_DUPLICATE_MANIFEST_BARRIER'])
manifest_release = Path(os.environ['F0_DUPLICATE_MANIFEST_RELEASE'])
receipt_pre_barrier = Path(os.environ['F0_DUPLICATE_RECEIPT_PRE_BARRIER'])
receipt_pre_release = Path(os.environ['F0_DUPLICATE_RECEIPT_PRE_RELEASE'])
receipt_post_barrier = Path(os.environ['F0_DUPLICATE_RECEIPT_POST_BARRIER'])
receipt_post_release = Path(os.environ['F0_DUPLICATE_RECEIPT_POST_RELEASE'])
receipt_committed = Path(os.environ['F0_DUPLICATE_RECEIPT_COMMITTED'])
repeat_marker_root = Path(os.environ['F0_DUPLICATE_REPEAT_MARKERS'])
receipt_write_failure_marker = Path(os.environ['F0_DUPLICATE_RECEIPT_WRITE_FAILURE'])

def cancellable_epoch(epoch):
    if epoch == 1:
        checkpoint = attempt_dir / 'resume_epoch_001.pt'
        ready_path.write_text(json.dumps({
            'pid': os.getpid(), 'run_id': entry['run_id'],
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': runner_module._sha_file(str(checkpoint)),
            'completed_epochs': 1,
        }) + '\n', encoding='utf-8')
        while True:
            time.sleep(0.01)
    return original_train_epoch(epoch)

runner._train_epoch = cancellable_epoch
def receipt_barrier(phase, payload):
    if phase == 'before_receipt_commit':
        receipt_pre_barrier.write_text(json.dumps({
            'pid': os.getpid(), 'phase': phase, 'receipt_exists': Path(receipt_path).exists(),
            'receipt_path': str(Path(receipt_path).resolve()),
            'status_path': payload['status_path'], 'manifest_path': payload['manifest_path'],
            'canonical_lock_path': payload['canonical_lock_path'],
            'gpu_lock_path': payload['gpu_lock_path'],
            'canonical_lock_owner': payload['canonical_lock_owner'],
            'gpu_lock_owner': payload['gpu_lock_owner'],
        }, sort_keys=True) + '\n', encoding='utf-8')
        while not receipt_pre_release.exists():
            time.sleep(0.01)
    elif phase == 'after_receipt_commit':
        receipt_committed.write_text(json.dumps({
            'pid': os.getpid(), 'status': payload['status'],
            'signal_repeat_count': payload['signal_repeat_count'],
        }, sort_keys=True) + '\n', encoding='utf-8')
        receipt_post_barrier.write_text(json.dumps({
            'pid': os.getpid(), 'phase': phase,
            'receipt_sha256': runner_module._sha_file(receipt_path),
        }, sort_keys=True) + '\n', encoding='utf-8')
        while not receipt_post_release.exists():
            time.sleep(0.01)

runner.interruption_receipt_hook = receipt_barrier
if os.environ.get('F0_DUPLICATE_FAIL_RECEIPT') == '1':
    original_receipt_writer = runner_module._write_invocation_receipt
    def fail_interrupted_receipt(path, payload):
        if payload.get('status') == 'INTERRUPTED':
            receipt_write_failure_marker.write_text(json.dumps({
                'path': str(Path(path).resolve()), 'status': payload.get('status'),
                'pid': os.getpid(), 'phase': 'injected_before_atomic_commit',
            }) + '\n', encoding='utf-8')
            raise OSError('injected interrupted-receipt commit failure')
        return original_receipt_writer(path, payload)
    runner_module._write_invocation_receipt = fail_interrupted_receipt

original_write_json = runner_module._write_json

def barrier_write_json(path, payload):
    if (Path(path).resolve() == (attempt_dir / 'run_manifest.json').resolve()
            and payload.get('status') == 'INTERRUPTED'):
        manifest_barrier.write_text(json.dumps({
            'pid': os.getpid(), 'phase': 'manifest_interrupt_commit',
            'status_file': str(status_file),
        }) + '\n', encoding='utf-8')
        while not manifest_release.exists():
            time.sleep(0.01)
    return original_write_json(path, payload)

runner_module._write_json = barrier_write_json
repeat_signals = {signal.SIGTERM: 'term', signal.SIGINT: 'int', signal.SIGHUP: 'hup'}

def observed_handler(signum, frame):
    child_control._signal_handler(signum, frame)
    if child_control._CHILD_SIGNAL is not None:
        (repeat_marker_root / (repeat_signals[signum] + '.received')).write_text(
            json.dumps({'signal': int(signum),
                        'first_signal': child_control._CHILD_SIGNAL,
                        'repeat_count': child_control._CHILD_SIGNAL_REPEAT_COUNT}) + '\n',
            encoding='utf-8')

for sig in repeat_signals:
    signal.signal(sig, observed_handler)
signal.pthread_sigmask(signal.SIG_UNBLOCK, set(repeat_signals))

gpu_lock = QueueFileLock(str(output_root / 'gpu_locks/gpu0.lock'))
try:
    runner.run(max_epochs=2, status_file=str(status_file), gpu_lock=gpu_lock)
except KeyboardInterrupt:
    slot_lease.close()
    signal_no = int(child_control._CHILD_SIGNAL or signal.SIGTERM)
    payload = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    assert payload['status'] == 'INTERRUPTED'
    assert payload['signal'] == signal_no
    append_phase_event(
        phase_log, phase='child_interrupted_receipt_verified_after_commit',
        worker_id=worker_id, nonce=nonce, signal=signal_no,
        signal_repeat_count=child_control._CHILD_SIGNAL_REPEAT_COUNT,
        run_id=payload['run_id'], attempt_id=payload['attempt_id'],
        completed_epochs=payload['completed_epochs'],
        manifest_sha256=payload['manifest_sha256'],
        last_full_checkpoint=payload['last_full_checkpoint'],
        receipt_path=str(Path(receipt_path).resolve()))
    raise SystemExit(payload['exit_code'])
else:
    slot_lease.close()
''', encoding='utf-8')
        runs = tmp_path / 'duplicate_signal_runs'
        result_path = tmp_path / 'duplicate-signal-supervisor-result.json'
        queue_path = tmp_path / 'duplicate-signal-queue.json'
        search_gate = tmp_path / 'duplicate-signal-search-gate.json'
        execution_gate = tmp_path / 'duplicate-signal-execution-gate.json'
        data_path = tmp_path / 'duplicate-signal-data.csv'
        ready = tmp_path / 'training-ready.json'
        manifest_barrier = tmp_path / 'manifest-write-barrier.json'
        manifest_release = tmp_path / 'manifest-write-release'
        receipt_pre_barrier = tmp_path / 'receipt-precommit-barrier.json'
        receipt_pre_release = tmp_path / 'receipt-precommit-release'
        receipt_post_barrier = tmp_path / 'receipt-postcommit-barrier.json'
        receipt_post_release = tmp_path / 'receipt-postcommit-release'
        receipt_committed = tmp_path / 'receipt-committed.json'
        receipt_write_failure = tmp_path / 'receipt-write-failure.json'
        repeat_markers = tmp_path / 'repeat-signals'
        repeat_markers.mkdir()
        scheduler = Path(leases[0].lock.path).parent
        env = dict(os.environ)
        env.update({
            'F0_PROCESS_SCHEDULER_ROOT': str(scheduler),
            'F0_RUNS_DIR': str(ROOT / 'experiments/stage_F0_outputs/runs'),
            'F0_EXPERIMENTS_DIR': str(ROOT / 'experiments'),
            'F0_DUPLICATE_TRAINING_READY': str(ready),
            'F0_DUPLICATE_MANIFEST_BARRIER': str(manifest_barrier),
            'F0_DUPLICATE_MANIFEST_RELEASE': str(manifest_release),
            'F0_DUPLICATE_RECEIPT_PRE_BARRIER': str(receipt_pre_barrier),
            'F0_DUPLICATE_RECEIPT_PRE_RELEASE': str(receipt_pre_release),
            'F0_DUPLICATE_RECEIPT_POST_BARRIER': str(receipt_post_barrier),
            'F0_DUPLICATE_RECEIPT_POST_RELEASE': str(receipt_post_release),
            'F0_DUPLICATE_RECEIPT_COMMITTED': str(receipt_committed),
            'F0_DUPLICATE_RECEIPT_WRITE_FAILURE': str(receipt_write_failure),
            'F0_DUPLICATE_REPEAT_MARKERS': str(repeat_markers),
        })
        if fail_receipt:
            env['F0_DUPLICATE_FAIL_RECEIPT'] = '1'
        proc = subprocess.Popen(
            [str(PYTHON), str(helper), str(runs), str(result_path), str(queue_path),
             str(search_gate), str(execution_gate), str(data_path), str(fixture),
             str(scheduler), leases[0].lock.path, leases[1].lock.path,
             str(leases[0].lock.handle.fileno()), str(leases[1].lock.handle.fileno()),
             str(leases[0].index), str(leases[1].index), 'duplicate_interrupt', '0'],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            close_fds=True,
            pass_fds=(leases[0].lock.handle.fileno(), leases[1].lock.handle.fileno()))
        return {
            'leases': leases, 'runs': runs, 'proc': proc, 'result_path': result_path,
            'ready': ready, 'manifest_barrier': manifest_barrier,
            'manifest_release': manifest_release, 'receipt_committed': receipt_committed,
            'receipt_pre_barrier': receipt_pre_barrier,
            'receipt_pre_release': receipt_pre_release,
            'receipt_post_barrier': receipt_post_barrier,
            'receipt_post_release': receipt_post_release,
            'repeat_markers': repeat_markers,
            'receipt_write_failure': receipt_write_failure,
            'fail_receipt': fail_receipt,
        }
    except BaseException:
        for lease in leases:
            lease.close()
        raise


def _start_parent_guard_fixture(tmp_path, mode):
    leases = _reserve_two_test_slots()
    try:
        helper = tmp_path / 'parent_helper.py'
        helper.write_text(_parent_helper_source(), encoding='utf-8')
        ready = tmp_path / (mode + '.ready.json')
        receipt = tmp_path / (mode + '.receipt.json')
        phase = tmp_path / (mode + '.phases.jsonl')
        scheduler = Path(leases[0].lock.path).parent
        env = dict(os.environ)
        env['F0_PROCESS_SCHEDULER_ROOT'] = str(scheduler)
        proc = subprocess.Popen(
            [str(PYTHON), str(helper), mode, str(CHILD), str(scheduler), str(ready),
             str(receipt), str(phase), leases[0].lock.path, leases[1].lock.path,
             str(leases[0].lock.handle.fileno()), str(leases[1].lock.handle.fileno()),
             str(leases[0].index), str(leases[1].index),
             str(tmp_path / 'runner-import-marker')],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, close_fds=True,
            pass_fds=(leases[0].lock.handle.fileno(), leases[1].lock.handle.fileno()))
        return leases, proc, ready, receipt, phase
    except BaseException:
        for lease in leases:
            lease.close()
        raise


def test_process_group_stop_and_continue_cover_supervisor_and_child(tmp_path):
    leases = _reserve_two_test_slots()
    helper = tmp_path / 'group_helper.py'
    child_code = 'import time; time.sleep(60)'
    helper.write_text(
        'import json, os, subprocess, sys, time\n'
        'os.setpgid(0, 0)\n'
        'child = subprocess.Popen([sys.executable, "-c", ' + repr(child_code) + '])\n'
        'print(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(), '
        '"child_pid": child.pid}), flush=True)\n'
        'child.wait()\n', encoding='utf-8')
    proc = None
    pgid = None
    child_pid = None
    try:
        proc = subprocess.Popen([str(PYTHON), str(helper)], cwd=str(ROOT),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        row = json.loads(proc.stdout.readline())
        pgid, child_pid = int(row['pgid']), int(row['child_pid'])
        assert row['pid'] == pgid
        assert _wait_for(lambda: _identity(proc.pid)['state'] not in ('Z', 'X')
                         and _identity(child_pid)['state'] not in ('Z', 'X'))

        os.killpg(pgid, signal.SIGSTOP)
        assert _wait_for(lambda: _identity(proc.pid)['state'] == 'T'
                         and _identity(child_pid)['state'] == 'T')
        os.killpg(pgid, signal.SIGCONT)
        assert _wait_for(lambda: _identity(proc.pid)['state'] != 'T'
                         and _identity(child_pid)['state'] != 'T')
    finally:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for lease in leases:
            lease.close()
    assert _wait_for(lambda: _identity(child_pid)['state'] in ('Z', 'X')
                     if Path('/proc/{}/stat'.format(child_pid)).exists() else True)


def test_pdeathsig_kills_guarded_child_and_parent_race_fails_before_runner_import(tmp_path):
    guarded = _start_parent_guard_fixture(tmp_path, 'guarded')
    leases, parent, ready, receipt, _phase = guarded
    try:
        assert _wait_for(lambda: ready.is_file())
        ready_payload = json.loads(ready.read_text(encoding='utf-8'))
        assert ready_payload['guard_installed'] is True
        child_pid = int(ready_payload['pid'])
        assert child_pid != parent.pid
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=3)
        assert _wait_for(lambda: not Path('/proc/{}/stat'.format(child_pid)).exists()
                         or _identity(child_pid)['state'] in ('Z', 'X'))
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        for lease in leases:
            lease.close()

    racing = _start_parent_guard_fixture(tmp_path, 'race')
    leases, parent, ready, receipt, _phase = racing
    try:
        assert _wait_for(lambda: ready.is_file())
        ready_payload = json.loads(ready.read_text(encoding='utf-8'))
        assert ready_payload['guard_installed'] is False
        child_pid = int(ready_payload['pid'])
        assert parent.wait(timeout=3) == 0
        assert _wait_for(lambda: receipt.is_file())
        failure = json.loads(receipt.read_text(encoding='utf-8'))
        assert failure['status'] == 'BOOTSTRAP_OR_RUNNER_FAILURE'
        assert failure['pid'] == child_pid
        assert not Path(tmp_path / 'runner-import-marker').exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        for lease in leases:
            lease.close()


@pytest.mark.parametrize('mode, expected_signal', [
    ('int', signal.SIGINT), ('term', signal.SIGTERM), ('hup', signal.SIGHUP),
])
def test_supervisor_forwards_int_term_hup_to_current_child(tmp_path, mode, expected_signal):
    leases, runs, proc, return_code = _run_supervisor_fixture(
        tmp_path, 'signal_sleep', signal_no=int(expected_signal))
    try:
        assert proc.returncode != 0
        assert return_code != 0
        worker_log = runs / 'worker_logs/audited_worker_0.jsonl'
        events = [json.loads(line) for line in worker_log.read_text().splitlines()]
        assert sum(row['event'] == 'CHILD_STARTED' for row in events) == 1
        failure = json.loads(next(runs.glob('worker_logs/worker_0_failure_*.json')).read_text())
        assert failure['signal'] == int(expected_signal)
        assert failure['automatic_retry'] is False
    finally:
        for lease in leases:
            lease.close()


def test_real_duplicate_signals_preserve_interrupted_checkpoint_and_cleanup(tmp_path):
    fixture = _start_duplicate_signal_fixture(tmp_path)
    leases = fixture['leases']
    runs = fixture['runs']
    proc = fixture['proc']
    supervisor_pid = proc.pid
    paths = [fixture['manifest_release'], fixture['receipt_pre_release'],
             fixture['receipt_post_release']]
    queue_locks = []
    worker_lock = None
    scheduler_locks = []
    try:
        assert _wait_for(lambda: fixture['ready'].is_file(), timeout=20)
        ready = json.loads(fixture['ready'].read_text(encoding='utf-8'))
        child_pid = int(ready['pid'])
        supervisor_identity = _identity(supervisor_pid)
        child_identity = _identity(child_pid)
        assert supervisor_identity['pgid'] == supervisor_pid
        assert child_identity['pgid'] == supervisor_pid

        sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
        import run_search_claim_worker as controller
        first = controller._stop_peer_worker('1', output_root=runs)
        assert first['sent'] is True, first
        assert first['target'] == 'supervisor_pid'
        assert first['send_method'] == 'pidfd'
        assert first['pid'] == supervisor_pid
        assert first['pgid'] == supervisor_pid
        assert first['start_ticks'] == supervisor_identity['start_ticks']

        assert _wait_for(lambda: fixture['manifest_barrier'].is_file(), timeout=20)
        barrier = json.loads(fixture['manifest_barrier'].read_text(encoding='utf-8'))
        assert barrier['pid'] == child_pid
        # A repeat peer-stop is a real supervisor-PID signal. The child must
        # receive no second production TERM while its manifest commit is held.
        repeated = controller._stop_peer_worker('1', output_root=runs)
        assert repeated['sent'] is True and repeated['pid'] == supervisor_pid
        controller._atomic_json(runs / 'execution_control/global_hold.json', {
            'schema': 'F0-EXECUTION-GLOBAL-HOLD-v1', 'active': True,
            'hold_id': 'duplicate-signal-fixture',
            'events': [{'event': 'DUPLICATE_SIGNAL_FIXTURE_HOLD',
                        'worker_id': '0', 'child_pid': child_pid,
                        'supervisor_pid': supervisor_pid}],
        })
        worker_log = runs / 'worker_logs/audited_worker_0.jsonl'
        assert _wait_for(lambda: worker_log.is_file() and any(
            json.loads(line).get('event') == 'SUPERVISOR_STOP_SIGNAL_REPEATED'
            for line in worker_log.read_text(encoding='utf-8').splitlines()), timeout=5)

        # Direct child signals are test fault injection, deliberately inside
        # the runner's manifest cleanup barrier. The actual child handler must
        # acknowledge every repeat without another exception.
        for sig, name in ((signal.SIGTERM, 'term'), (signal.SIGINT, 'int'),
                          (signal.SIGHUP, 'hup')):
            os.kill(child_pid, sig)
            marker = fixture['repeat_markers'] / (name + '.received')
            assert _wait_for(lambda marker=marker: marker.is_file(), timeout=5)
            observed = json.loads(marker.read_text(encoding='utf-8'))
            assert observed['first_signal'] == signal.SIGTERM

        fixture['manifest_release'].touch()
        assert _wait_for(lambda: fixture['receipt_pre_barrier'].is_file(), timeout=20)
        pre = json.loads(fixture['receipt_pre_barrier'].read_text(encoding='utf-8'))
        assert pre['pid'] == child_pid and pre['phase'] == 'before_receipt_commit'
        assert pre['receipt_exists'] is False
        receipt_path = Path(pre['receipt_path'])
        assert not receipt_path.exists()
        status_pre = json.loads(Path(pre['status_path']).read_text(encoding='utf-8'))
        manifest_pre = json.loads(Path(pre['manifest_path']).read_text(encoding='utf-8'))
        assert status_pre['status'] == manifest_pre['status'] == 'INTERRUPTED'
        assert status_pre['completed_epochs'] == manifest_pre['epochs_completed'] == 1

        # Barrier A: status/manifest are visible, receipt is not yet committed,
        # and an independent reserved pytest process cannot take either lock.
        from stage_F0_queue import QueueFileLock
        for lock_path in (pre['canonical_lock_path'], pre['gpu_lock_path']):
            with pytest.raises(BlockingIOError):
                QueueFileLock(lock_path)

        # Deliver another real set of duplicate signals in the receipt
        # preparation window. They must remain idempotent and be reflected in
        # the final locked status/manifest/receipt transaction.
        for sig, name in ((signal.SIGTERM, 'term'), (signal.SIGINT, 'int'),
                          (signal.SIGHUP, 'hup')):
            os.kill(child_pid, sig)
            marker = fixture['repeat_markers'] / (name + '.received')
            assert _wait_for(lambda marker=marker: marker.is_file()
                             and json.loads(marker.read_text(encoding='utf-8'))[
                                 'repeat_count'] >= 4, timeout=5)
        fixture['receipt_pre_release'].touch()

        assert _wait_for(lambda: fixture['receipt_committed'].is_file(), timeout=20)
        assert _wait_for(lambda: fixture['receipt_post_barrier'].is_file(), timeout=20)
        post = json.loads(fixture['receipt_post_barrier'].read_text(encoding='utf-8'))
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert post['pid'] == child_pid and post['phase'] == 'after_receipt_commit'
        assert post['receipt_sha256'] == __import__('hashlib').sha256(
            receipt_path.read_bytes()).hexdigest()
        assert receipt['status'] == 'INTERRUPTED'
        assert receipt['transaction_commit'] == 'DURABLE_BEFORE_LOCK_RELEASE'
        assert receipt['signal'] == signal.SIGTERM
        assert receipt['signal_repeat_count'] == 6
        assert receipt['pid'] == child_pid and receipt['ppid'] == supervisor_pid
        assert receipt['pgid'] == supervisor_pid
        assert receipt['child_start_ticks'] == child_identity['start_ticks']
        assert receipt['supervisor_start_ticks'] == supervisor_identity['start_ticks']

        status = json.loads(Path(receipt['status_path']).read_text(encoding='utf-8'))
        manifest = json.loads(Path(receipt['manifest_path']).read_text(encoding='utf-8'))
        assert status['status'] == manifest['status'] == 'INTERRUPTED'
        assert status['completed_epochs'] == manifest['epochs_completed'] == 1
        assert status['last_full_checkpoint'] == manifest['last_full_checkpoint']
        assert receipt['last_full_checkpoint'] == status['last_full_checkpoint']
        assert status['interruption_identity'] == manifest['interruption_identity']
        assert status['interruption_identity'] == receipt['interruption_identity']
        assert receipt['canonical_lock_owner']['pid'] == child_pid
        assert receipt['gpu_lock_owner']['pid'] == child_pid
        assert receipt['interruption_identity']['signal_repeat_count'] == 6
        checkpoint = Path(receipt['manifest_path']).parent / receipt['last_full_checkpoint']['name']
        digest = __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == ready['checkpoint_sha256']
        assert digest == receipt['last_full_checkpoint']['sha256']
        assert receipt['last_full_checkpoint']['completed_epochs'] == 1
        assert manifest['run_id'] == status['run_id'] == receipt['run_id']

        # Barrier B: receipt is durable but F0Runner still holds both resource
        # locks. They become available only after the child leaves run()'s
        # finally block.
        for lock_path in (pre['canonical_lock_path'], pre['gpu_lock_path']):
            with pytest.raises(BlockingIOError):
                QueueFileLock(lock_path)

        from run_search_claim_worker import _ExclusiveLock
        with pytest.raises(BlockingIOError):
            _ExclusiveLock(runs / 'worker_locks/worker_0.lock')
        scheduler_root = Path(leases[0].lock.path).parent
        for lease in leases:
            # Close only this process's inherited descriptor copy. Do not
            # issue LOCK_UN: the supervisor must retain the reservation.
            lease.lock.handle.close()
        assert _wait_for(lambda: all(_cannot_lock(scheduler_root / 'slot{}.lock'.format(i))
                                    for i in (leases[0].index, leases[1].index)))

        fixture['receipt_post_release'].touch()
        assert proc.wait(timeout=15) == 143
        stdout, stderr = proc.communicate(timeout=2)
        assert fixture['result_path'].is_file(), (stdout, stderr[-4000:])
        assert json.loads(fixture['result_path'].read_text(encoding='utf-8'))[
            'exit_code'] == 143

        worker_events = [json.loads(line) for line in worker_log.read_text().splitlines()]
        assert sum(row['event'] == 'CHILD_STARTED' for row in worker_events) == 1
        assert sum(row['event'] == 'SIGNAL_FORWARDED_TO_CHILD' for row in worker_events) == 1
        repeat_events = [row for row in worker_events
                         if row['event'] == 'SUPERVISOR_STOP_SIGNAL_REPEATED']
        assert len(repeat_events) == 1
        assert repeat_events[0]['repeat_count'] == 1
        forward = next(row for row in worker_events
                       if row['event'] == 'SIGNAL_FORWARDED_TO_CHILD')
        assert repeat_events[0]['termination_deadline_monotonic'] == \
            forward['termination_deadline_monotonic']
        validated = [row for row in worker_events
                     if row['event'] == 'CHILD_INTERRUPTED_RECEIPT_VALIDATED']
        assert len(validated) == 1
        assert validated[0]['next_child_started'] is False
        assert validated[0]['signal_repeat_count'] == 6
        assert not any(row['event'] == 'CHILD_VALIDATION_COMPLETE'
                       for row in worker_events)
        assert json.loads((runs / 'execution_control/global_hold.json').read_text())[
            'active'] is True
        phase_path = Path(next(runs.glob('worker_logs/*.receipt.json')).with_name(
            next(runs.glob('worker_logs/*.receipt.json')).name.replace(
                '.receipt.json', '.phases.jsonl')))
        phase_events = [json.loads(line) for line in phase_path.read_text().splitlines()]
        assert any(row.get('phase') == 'signal_forwarded'
                   for row in phase_events)
        assert any(row.get('phase') == 'supervisor_signal_repeated'
                   for row in phase_events)
        assert any(row.get('phase') == 'interrupted_receipt_precommit'
                   for row in phase_events)
        assert any(row.get('phase') == 'interrupted_receipt_committed'
                   for row in phase_events)
        interruption_phase = next(row for row in phase_events
                                  if row.get('phase') ==
                                  'child_interrupted_receipt_verified_after_commit')
        assert interruption_phase['signal_repeat_count'] == 6
        assert interruption_phase['last_full_checkpoint'] == receipt['last_full_checkpoint']

        # After the worker exits, an independent process can take back every
        # released canonical, GPU, worker-ID, and reserved process-slot lock.
        worker_lock = _ExclusiveLock(runs / 'worker_locks/worker_0.lock')
        worker_lock.close()
        worker_lock = None
        queue_locks.append(QueueFileLock(str(runs / 'canonical_locks' /
                                             (receipt['run_id'] + '.lock'))))
        queue_locks.append(QueueFileLock(str(runs / 'gpu_locks/gpu0.lock')))
        for index in (leases[0].index, leases[1].index):
            scheduler_locks.append(QueueFileLock(str(
                scheduler_root / 'slot{}.lock'.format(index))))
    finally:
        for path in paths:
            path.touch(exist_ok=True)
        for lock in [*queue_locks, *scheduler_locks]:
            lock.close()
        if worker_lock is not None:
            worker_lock.close()
        for lease in leases:
            lease.close()
        if proc.poll() is None:
            try:
                os.killpg(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def test_interrupted_receipt_commit_failure_holds_and_never_fabricates_success(tmp_path):
    fixture = _start_duplicate_signal_fixture(tmp_path, fail_receipt=True)
    proc = fixture['proc']
    leases = fixture['leases']
    runs = fixture['runs']
    try:
        assert _wait_for(lambda: fixture['ready'].is_file(), timeout=20)
        ready = json.loads(fixture['ready'].read_text(encoding='utf-8'))
        child_pid = int(ready['pid'])

        sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
        import run_search_claim_worker as controller
        sent = controller._stop_peer_worker('1', output_root=runs)
        assert sent['sent'] is True and sent['send_method'] == 'pidfd'
        assert sent['pid'] == proc.pid

        assert _wait_for(lambda: fixture['manifest_barrier'].is_file(), timeout=20)
        fixture['manifest_release'].touch()
        assert _wait_for(lambda: fixture['receipt_pre_barrier'].is_file(), timeout=20)
        pre = json.loads(fixture['receipt_pre_barrier'].read_text(encoding='utf-8'))
        receipt_path = Path(pre['receipt_path'])
        assert pre['receipt_exists'] is False and not receipt_path.exists()
        from stage_F0_queue import QueueFileLock
        for lock_path in (pre['canonical_lock_path'], pre['gpu_lock_path']):
            with pytest.raises(BlockingIOError):
                QueueFileLock(lock_path)

        fixture['receipt_pre_release'].touch()
        assert proc.wait(timeout=15) != 0
        stdout, stderr = proc.communicate(timeout=2)
        assert fixture['receipt_write_failure'].is_file(), (stdout, stderr[-4000:])
        injected = json.loads(fixture['receipt_write_failure'].read_text(encoding='utf-8'))
        assert injected['pid'] == child_pid
        assert injected['phase'] == 'injected_before_atomic_commit'
        assert injected['status'] == 'INTERRUPTED'
        assert not receipt_path.exists()
        assert not fixture['receipt_committed'].exists()
        assert not fixture['receipt_post_barrier'].exists()

        status = json.loads(Path(pre['status_path']).read_text(encoding='utf-8'))
        manifest = json.loads(Path(pre['manifest_path']).read_text(encoding='utf-8'))
        assert status['status'] == manifest['status'] == 'INTERRUPTED'
        assert status['completed_epochs'] == manifest['epochs_completed'] == 1
        checkpoint = Path(pre['manifest_path']).parent / 'resume_epoch_001.pt'
        assert __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest() == \
            ready['checkpoint_sha256']

        worker_log = runs / 'worker_logs/audited_worker_0.jsonl'
        events = [json.loads(line) for line in worker_log.read_text().splitlines()]
        assert sum(row['event'] == 'CHILD_STARTED' for row in events) == 1
        assert sum(row['event'] == 'SIGNAL_FORWARDED_TO_CHILD' for row in events) == 1
        assert not any(row['event'] == 'CHILD_INTERRUPTED_RECEIPT_VALIDATED'
                       for row in events)
        failure_paths = list((runs / 'worker_logs').glob('worker_0_failure_*.json'))
        assert len(failure_paths) == 1
        failure = json.loads(failure_paths[0].read_text(encoding='utf-8'))
        assert failure['automatic_retry'] is False
        assert failure['canonical_claim']['run_id'] == status['run_id']
        assert failure['canonical_claim']['attempt_id'] == status['attempt_id']
        hold = json.loads((runs / 'execution_control/global_hold.json').read_text())
        assert hold['active'] is True

        # An incomplete receipt transaction releases locks only when the
        # runner unwinds, and remains held for explicit audit/recovery.
        canonical = QueueFileLock(pre['canonical_lock_path'])
        gpu = QueueFileLock(pre['gpu_lock_path'])
        canonical.close()
        gpu.close()
        for lease in leases:
            lease.lock.handle.close()
        scheduler_root = Path(leases[0].lock.path).parent
        slots = [QueueFileLock(str(scheduler_root / 'slot{}.lock'.format(lease.index)))
                 for lease in leases]
        for slot in slots:
            slot.close()
    finally:
        for path in (fixture['manifest_release'], fixture['receipt_pre_release'],
                     fixture['receipt_post_release']):
            path.touch(exist_ok=True)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        for lease in leases:
            lease.close()


def _cannot_lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        try:
            os.close(fd)
        except UnboundLocalError:
            pass


@pytest.mark.parametrize('expected_signal', [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_stop_before_child_start_does_not_claim_or_hold(tmp_path, expected_signal):
    leases, runs, proc, return_code = _run_supervisor_fixture(
        tmp_path, 'stop_before_child', signal_no=int(expected_signal))
    try:
        assert proc.returncode == 128 + int(expected_signal)
        assert return_code == 128 + int(expected_signal)
        events = [json.loads(line) for line in
                  (runs / 'worker_logs/audited_worker_0.jsonl').read_text().splitlines()]
        assert not any(row['event'] == 'CHILD_STARTED' for row in events)
        assert any(row['event'] == 'WORKER_STOPPED_BEFORE_NEXT_CHILD' for row in events)
        assert not (runs / 'execution_control/global_hold.json').exists()
    finally:
        for lease in leases:
            lease.close()


@pytest.mark.parametrize('mode, expected_signal, started', [
    ('spawnfail', None, 0),
    ('nonzero', None, 1),
    ('abrt', signal.SIGABRT, 1),
    ('empty', None, 1),
    ('invalid_active_claim', None, 0),
    ('malformed_status', None, 0),
])
def test_failure_modes_hold_without_retry_or_second_child(
        tmp_path, mode, expected_signal, started):
    leases, runs, proc, return_code = _run_supervisor_fixture(tmp_path, mode)
    try:
        assert proc.returncode != 0
        assert return_code != 0
        hold_path = runs / 'execution_control/global_hold.json'
        hold = json.loads(hold_path.read_text(encoding='utf-8'))
        assert hold['active'] is True
        events = [json.loads(line) for line in
                  (runs / 'worker_logs/audited_worker_0.jsonl').read_text().splitlines()]
        assert sum(row['event'] == 'CHILD_STARTED' for row in events) == started
        failure_files = list((runs / 'worker_logs').glob('worker_0_failure_*.json'))
        assert len(failure_files) == 1
        failure = json.loads(failure_files[0].read_text(encoding='utf-8'))
        assert failure['automatic_retry'] is False
        assert failure['automatic_recovery'] is False
        assert failure['signal'] == expected_signal
        assert failure['canonical_claim'] is None
        if started:
            assert Path(failure['native_stderr_path']).is_file()
            assert failure['native_stderr_sha256']
            assert Path(failure['phase_events_path']).is_file()
            assert failure['phase_events_sha256']
    finally:
        for lease in leases:
            lease.close()


def test_termination_timeout_persists_hold_and_peer_stop_before_group_sigkill(tmp_path):
    leases, runs, proc, return_code = _run_supervisor_fixture(
        tmp_path, 'ignore_term', signal_no=int(signal.SIGTERM))
    try:
        assert proc.returncode == -signal.SIGKILL
        assert return_code == -signal.SIGKILL
        hold = json.loads((runs / 'execution_control/global_hold.json').read_text())
        assert hold['active'] is True
        failure_files = list((runs / 'worker_logs').glob('worker_0_failure_*.json'))
        assert len(failure_files) == 1
        failure = json.loads(failure_files[0].read_text(encoding='utf-8'))
        assert failure['phase'] == 'cleanup_escalation'
        assert failure['child_pid'] is not None
        peer_request = runs / 'worker_logs/peer-stop-request.json'
        assert json.loads(peer_request.read_text(encoding='utf-8'))['durable'] is True
        assert any(event.get('failure_path') == str(failure_files[0])
                   for event in hold['events'])
        worker_events = [json.loads(line) for line in
                         (runs / 'worker_logs/audited_worker_0.jsonl').read_text().splitlines()]
        escalation = [row for row in worker_events
                      if row['event'] == 'CLEANUP_ESCALATION_GROUP_SIGKILL']
        assert len(escalation) == 1 and escalation[0]['failure_durable'] is True
        phase_path = Path(failure['phase_events_path'])
        phase_events = [json.loads(line) for line in phase_path.read_text().splitlines()]
        assert any(row['phase'] == 'cleanup_escalation_failure_durable'
                   for row in phase_events)
    finally:
        for lease in leases:
            lease.close()


def test_production_termination_timeout_remains_30_seconds():
    sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
    import run_search_claim_worker as controller

    assert inspect.signature(controller.TrajectorySupervisor.__init__).parameters[
        'termination_timeout'].default == 30.0


def test_active_claim_reader_waits_for_locked_owner_and_status_rewrite(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
    import run_search_claim_worker as controller

    output = tmp_path / 'runs'
    run_id = 'concurrent-owner-rewrite'
    owner_path = output / 'canonical_locks' / (run_id + '.lock')
    status_path = output / 'queue_status' / (run_id + '.json')
    control_path = output / 'worker_locks' / 'worker_0.control'
    state_lock_path = output / 'execution_control' / 'execution_state.lock'
    owner_path.parent.mkdir(parents=True)
    status_path.parent.mkdir(parents=True)
    control_path.parent.mkdir(parents=True)
    identity = controller.linux_process_identity(os.getpid())
    control_path.write_text('{} {} {} fixture\n'.format(
        identity['pid'], identity['pgid'], identity['start_ticks']), encoding='ascii')
    owner = {'pid': os.getpid(), 'host': __import__('socket').gethostname(),
             'run_id': run_id}
    status = {'pid': os.getpid(), 'host': __import__('socket').gethostname(),
              'run_id': run_id, 'status': 'RUNNING', 'gpu_index': None}
    owner_path.write_text(json.dumps(owner), encoding='utf-8')
    status_path.write_text(json.dumps(status), encoding='utf-8')
    rewriting = threading.Event()
    reader_started = threading.Event()
    read_result = []
    read_seconds = []

    def rewrite_owner_and_status():
        state_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with state_lock_path.open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            owner_path.write_text('{"pid":', encoding='utf-8')
            status_path.write_text('{"status":', encoding='utf-8')
            rewriting.set()
            time.sleep(0.25)
            owner_path.write_text(json.dumps(owner), encoding='utf-8')
            status_path.write_text(json.dumps(status), encoding='utf-8')
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read_active_claims():
        reader_started.set()
        started_at = time.monotonic()
        try:
            read_result.extend(controller.validate_active_claims(
                output_root=output, worker_id='0', worker_identity=identity))
        except BaseException as exc:
            read_result.append(exc)
        read_seconds.append(time.monotonic() - started_at)

    writer = threading.Thread(target=rewrite_owner_and_status)
    writer.start()
    assert rewriting.wait(timeout=2)
    reader = threading.Thread(target=read_active_claims)
    reader.start()
    assert reader_started.wait(timeout=2)
    writer.join(timeout=3)
    reader.join(timeout=3)
    assert not writer.is_alive() and not reader.is_alive()
    assert read_result == [{'run_id': run_id, 'pid': os.getpid(), 'status': 'RUNNING'}]
    assert read_seconds[0] >= 0.15


@pytest.mark.parametrize('missing_key', [
    '0d2e6555b049c012c0e5', 'a80a232a3c0068a48a08',
])
def test_frozen_held_run_ids_rejects_caller_omitting_new_incident_key(missing_key):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import (PRESERVED_H51_RUN_IDS,
                                         frozen_held_run_ids)

    complete = list(PRESERVED_H51_RUN_IDS)
    execution_gate = {'preserved_held_run_ids': complete}
    search_gate = {'blocked_search_keys': [complete[0]]}
    assert frozen_held_run_ids(execution_gate, search_gate, complete) == sorted(complete)
    caller = [run_id for run_id in complete if run_id != missing_key]
    with pytest.raises(PermissionError, match='caller-held IDs differ'):
        frozen_held_run_ids(execution_gate, search_gate, caller)


@pytest.mark.parametrize("rows, expected", [
    ("GPU-0, 1000, 899\nGPU-1, 1000, 899\n", True),
    ("GPU-0, 1000, 899\nGPU-1, 1000, 900\n", False),
    ("GPU-0, 1000, 899\nGPU-1, 1000, N/A\n", False),
])
def test_supervisor_rechecks_assigned_gpu_memory_before_child_spawn(
        tmp_path, monkeypatch, rows, expected):
    sys.path.insert(0, str(CHILD.parent))
    import run_search_claim_worker as supervisor
    monkeypatch.setattr(supervisor, "RUNS", tmp_path)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **kw:
                        type("Result", (), {"returncode": 0, "stdout": rows,
                                             "stderr": ""})())
    prelaunch_path = tmp_path / "execution_control" / "prelaunch_audit_v14_fresh_search.json"
    prelaunch_path.parent.mkdir(parents=True, exist_ok=True)
    prelaunch_path.write_text(json.dumps({"gpu_memory": {"devices": [
        {"index": 0, "uuid": "GPU-0", "readable": True, "eligible": True},
        {"index": 1, "uuid": "GPU-1", "readable": True, "eligible": True},
    ]}}), encoding="utf-8")
    fixture_gate_path = tmp_path / "execution_isolation_gate.json"
    fixture_gate_path.write_text(json.dumps({"schema": "F0-EXECUTION-ISOLATION-v1.4"}),
                                 encoding="utf-8")

    class Fixture:
        output_root = tmp_path
        execution_gate_path = fixture_gate_path
        worker_id = "1"

    if expected:
        record = supervisor.TrajectorySupervisor._prechild_gpu_memory_gate(Fixture())
        assert record["go"] is True
        assert record["gpu_uuid"] == "GPU-1"
        assert record["memory_used_fraction"] == pytest.approx(0.899)
    else:
        with pytest.raises(PermissionError, match="pre-child GPU"):
            supervisor.TrajectorySupervisor._prechild_gpu_memory_gate(Fixture())


@pytest.mark.parametrize('malformation', [
    'omit_sigsegv', 'omit_peer', 'old_h47', 'same_count_substitution',
])
def test_execution_gate_validation_rejects_non_h49_sets(tmp_path, malformation):
    sys.path.insert(0, str(ROOT / 'experiments'))
    import stage_F0_worker_control as control

    for relative in control.CONTROL_HASH_FILES:
        fixture_file = tmp_path / relative
        fixture_file.parent.mkdir(parents=True, exist_ok=True)
        fixture_file.write_text('frozen fixture input\n', encoding='utf-8')
    held = list(control.PRESERVED_H49_RUN_IDS)
    if malformation == 'omit_sigsegv':
        held.remove('7fd2006bdc8f1817a6f7')
    elif malformation == 'omit_peer':
        held.remove('1f04c7fb623d18f940ac')
    elif malformation == 'old_h47':
        held = list(control.PRESERVED_V12_H47_RUN_IDS)
    else:
        held[held.index('7fd2006bdc8f1817a6f7')] = 'ffffffffffffffffffff'
    gate_path = tmp_path / 'execution-gate.json'
    gate_path.write_text(json.dumps({
        'schema': 'F0-EXECUTION-ISOLATION-v1',
        'control_hashes': control.control_hash_inventory(tmp_path),
        'control_hash_inventory_sha256': control._sha256_json(
            control.control_hash_inventory(tmp_path)),
        'preserved_held_run_ids': held,
        'reviewer_verdict': 'PENDING',
    }), encoding='utf-8')

    with pytest.raises(PermissionError, match='exact frozen H49 hold set'):
        control.validate_execution_gate(
            root=tmp_path, gate_path=gate_path,
            search_gate_path=tmp_path / 'search-gate.json')


def test_global_hold_release_rejects_45_keys_and_same_count_substitution(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import (PRESERVED_SIGNAL_CLEANUP_KEYS,
                                         _hold_is_released,
                                         global_hold_path,
                                         global_hold_release_path,
                                         sha256_file)

    monkeypatch.setattr('stage_F0_worker_control._validate_released_v13_runtime',
                        lambda *_args, **_kwargs: True)
    hold = {'schema': 'F0-GLOBAL-HOLD-v1', 'active': False, 'hold_id': 'fixture-hold',
            'preserved_held_run_ids': list(PRESERVED_SIGNAL_CLEANUP_KEYS),
            'preserved_held_run_ids_sha256': '60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e'}
    hold_path = global_hold_path(tmp_path)
    hold_path.parent.mkdir(parents=True, exist_ok=True)
    hold_path.write_text(json.dumps(hold), encoding='utf-8')
    release = {
        'reviewer_verdict': 'PASS', 'hold_id': hold['hold_id'],
        'approved_by': 'fixture principal', 'approved_by_role': 'principal',
        'release_id': 'fixture-release', 'hold_sha256': sha256_file(hold_path),
        'preserved_held_run_ids_sha256': '60ff1cee2c6c77c65ee9fc7597bce7794d19ada480eecebe9b8742dc4261658e',
    }
    release_path = global_hold_release_path(tmp_path)
    previous_45 = [key for key in PRESERVED_SIGNAL_CLEANUP_KEYS
                   if key not in ('b3a55740aadd0e3377a9', 'cfbf6cc0e541e76cfb92',
                                  '7fd2006bdc8f1817a6f7', '1f04c7fb623d18f940ac')]
    assert len(previous_45) == 45
    release['preserved_held_run_ids'] = previous_45
    release_path.write_text(json.dumps(release), encoding='utf-8')
    assert _hold_is_released(hold, tmp_path) is False

    release['preserved_held_run_ids'] = list(PRESERVED_SIGNAL_CLEANUP_KEYS[:-1])
    release_path.write_text(json.dumps(release), encoding='utf-8')
    assert _hold_is_released(hold, tmp_path) is False

    substituted = list(PRESERVED_SIGNAL_CLEANUP_KEYS)
    substituted[substituted.index('cfbf6cc0e541e76cfb92')] = 'ffffffffffffffffffff'
    release['preserved_held_run_ids'] = substituted
    release_path.write_text(json.dumps(release), encoding='utf-8')
    assert _hold_is_released(hold, tmp_path) is False

    release['preserved_held_run_ids'] = list(PRESERVED_SIGNAL_CLEANUP_KEYS)
    release_path.write_text(json.dumps(release), encoding='utf-8')
    monkeypatch.setattr('stage_F0_worker_control._validate_released_v13_runtime',
                        lambda *_args, **_kwargs: False)
    assert _hold_is_released(hold, tmp_path) is False
    monkeypatch.setattr('stage_F0_worker_control._validate_released_v13_runtime',
                        lambda *_args, **_kwargs: True)
    assert _hold_is_released(hold, tmp_path) is True


def test_released_hold_requires_v13_runtime_status_and_provenance(monkeypatch, tmp_path):
    import stage_F0_worker_control as control

    monkeypatch.setattr(control, '_validate_released_v13_runtime', lambda *_args, **_kwargs: False)
    assert control._validate_released_v13_runtime(tmp_path) is False

