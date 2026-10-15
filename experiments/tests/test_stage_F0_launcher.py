import ast
import importlib.util
import json
import os
import signal
import socket
import time
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/home/xzf/miniconda3/envs/phase0-3-env/bin/python')
LAUNCHER = ROOT / 'experiments/stage_F0_outputs/runs/run_search_claim_worker.py'
SHELL_WRAPPER = ROOT / 'experiments/stage_F0_outputs/runs/run_search_worker.sh'
PYTHON_TASK_WRAPPER = ROOT / 'experiments/stage_F0_outputs/runs/run_f0_python_task.sh'


def _load_controller_module():
    name = 'f0_claim_worker_under_test'
    spec = importlib.util.spec_from_file_location(name, LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_execution_gate_tree(root: Path, *, verdict='PENDING', launch=False,
                              amendment_status='APPROVED',
                              execution_status='READY'):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import (CONTROL_HASH_FILES,
                                         PRESERVED_SIGNAL_CLEANUP_KEYS,
                                         control_hash_inventory, sha256_file,
                                         _sha256_json)
    from stage_F0_queue import code_hash

    for name in CONTROL_HASH_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('test-only frozen input\n', encoding='utf-8')
    report = root / 'review' / 'new-review.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"verdict":"PASS"}\n', encoding='utf-8')
    search = {
        'phase': 'search', 'protocol_version': 'F0-TRAIN-ML-v1',
        'data_audit_verdict': 'PASS', 'gpu_preflight_status': 'PASS',
        'formal_or_test_started': False, 'execution_status': execution_status,
        'code_hash': 'frozen-scientific-code-sha256',
        'blocked_search_keys': [PRESERVED_SIGNAL_CLEANUP_KEYS[0]],
    }
    search_path = root / 'search_gate.json'
    search_path.write_text(json.dumps(search), encoding='utf-8')
    control_hashes = control_hash_inventory(root)
    gate = {
        'schema': 'F0-EXECUTION-ISOLATION-v1',
        'control_hashes': control_hashes,
        'control_hash_inventory_sha256': _sha256_json(control_hashes),
        'reviewer_verdict': verdict,
        'reviewer_report_path': 'review/new-review.json',
        'reviewer_report_sha256': sha256_file(report),
        'launch_authorized': launch,
        'launch_scope': 'F0-B1-ML search only; fresh UNCLAIMED canonical keys only',
        'preserved_scientific_code_hash': search['code_hash'],
        'scientific_code_hash_amendment': {
            'status': amendment_status, 'reviewer_verdict': verdict,
            'frozen_scientific_code_hash': search['code_hash'],
            'runtime_control_code_hash': code_hash(require_tests=False),
            'control_hash_inventory_sha256': _sha256_json(control_hashes),
        },
        'preserved_held_run_ids': list(PRESERVED_SIGNAL_CLEANUP_KEYS),
        'prior_search_gate_sha256': sha256_file(search_path),
    }
    if execution_status == 'PAUSED_ABNORMAL_WORKER_EXIT':
        gate['prior_incident_release'] = {
            'reviewer_verdict': verdict, 'release_id': 'reviewed-release',
            'approved_by': 'principal identity', 'approved_by_role': 'principal',
            'preserved_held_run_ids': gate['preserved_held_run_ids'],
        }
    gate_path = root / 'execution_gate.json'
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    return gate_path, search_path, gate


def test_wrapper_pre_reserves_two_slots_and_refuses_pid_only_stop():
    syntax = subprocess.run(['bash', '-n', str(SHELL_WRAPPER)], capture_output=True,
                            text=True, timeout=10)
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(['bash', str(SHELL_WRAPPER), '0', '--self-test-reservation'],
                            cwd=str(ROOT), check=True, capture_output=True, text=True,
                            timeout=15)
    payload = json.loads(result.stdout.strip())
    assert payload['pid'] == payload['pgid']
    assert len(set(payload['reserved_slots'])) == 2
    assert payload['trajectory_started'] is False
    assert payload['max_concurrent_python_processes'] == 5
    assert payload['max_gpu_training_workers'] == 2

    rejected = subprocess.run(['bash', str(SHELL_WRAPPER), '--pid-stop'],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=10)
    assert rejected.returncode == 64
    assert 'PID-only STOP is rejected' in rejected.stderr


def test_wrapper_does_not_start_python_without_two_free_slots():
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot

    leases = [acquire_process_slot(role='launcher-cap-test', blocking=False)
              for _ in range(3)]
    try:
        result = subprocess.run(
            ['bash', str(SHELL_WRAPPER), '0', '--self-test-reservation'],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        assert result.returncode == 75
        assert 'could not pre-reserve two' in result.stderr
        assert result.stdout == ''
    finally:
        for lease in leases:
            lease.close()


def _start_pidfd_supervisor_fixture(tmp_path, output_root, controller, lease):
    ready = tmp_path / 'pidfd-target-ready.json'
    source = (
        'import json, os, signal, sys, time\n'
        'from pathlib import Path\n'
        'ready = Path(sys.argv[1])\n'
        'def stop(_sig, _frame):\n'
        ' print("received-term", flush=True); raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\n'
        'ready.write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}))\n'
        'while True: time.sleep(0.02)\n'
    )
    proc = subprocess.Popen([str(PYTHON), '-c', source, str(ready)],
                            cwd=str(ROOT), start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), 'pidfd supervisor fixture did not start'
    identity = controller.linux_process_identity(proc.pid)
    assert identity['pgid'] == proc.pid
    lease.lock.write_owner({
        'claimed_unix': time.time(), 'host': socket.gethostname(),
        'pid': proc.pid, 'role': 'operator-target-fixture', 'slot': lease.index,
    })
    controller._write_control_owner('0', {
        'pid': proc.pid, 'pgid': identity['pgid'],
        'start_ticks': identity['start_ticks'], 'nonce': 'pidfd-fixture-nonce',
    }, output_root=output_root)
    return proc, identity


def test_operator_signal_wrapper_uses_pidfd_and_refuses_stale_identity(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot
    import run_search_claim_worker as controller

    output_root = tmp_path / 'runs'
    scheduler = tmp_path / 'scheduler'
    monkeypatch.setenv('F0_RUNS_DIR', str(output_root))
    monkeypatch.setenv('F0_PROCESS_SCHEDULER_ROOT', str(scheduler))
    lease = acquire_process_slot(role='operator-target-reservation', blocking=False)
    target = None
    try:
        target, identity = _start_pidfd_supervisor_fixture(
            tmp_path, output_root, controller, lease)
        owner = controller._read_worker_control_identity('0', output_root=output_root)
        controller._write_control_owner('0', {
            **owner, 'start_ticks': identity['start_ticks'] + 1,
        }, output_root=output_root)
        rejected = subprocess.run(
            ['bash', str(SHELL_WRAPPER), 'signal', '0', 'TERM'], cwd=str(ROOT),
            env=dict(os.environ), capture_output=True, text=True, timeout=15)
        assert rejected.returncode != 0
        assert ('start-tick' in rejected.stderr or 'start_ticks' in rejected.stderr
                or 'identity' in rejected.stderr)
        assert target.poll() is None

        controller._write_control_owner('0', owner, output_root=output_root)
        sent = subprocess.run(
            ['bash', str(SHELL_WRAPPER), 'signal', '0', 'TERM'], cwd=str(ROOT),
            env=dict(os.environ), capture_output=True, text=True, timeout=15)
        assert sent.returncode == 0, sent.stderr
        payload = json.loads(sent.stdout.strip().splitlines()[-1])
        assert payload['status'] == 'SENT'
        assert payload['send_method'] == 'pidfd'
        assert payload['identity']['pid'] == target.pid
        assert target.wait(timeout=5) == 0
    finally:
        if target is not None and target.poll() is None:
            target.kill()
            target.wait(timeout=3)
        lease.close()


def test_pidfd_signal_fails_closed_if_unavailable_or_target_exits_after_validation(
        tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'experiments/stage_F0_outputs/runs'))
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot
    import run_search_claim_worker as controller

    output_root = tmp_path / 'runs'
    scheduler = tmp_path / 'scheduler'
    monkeypatch.setenv('F0_PROCESS_SCHEDULER_ROOT', str(scheduler))
    lease = acquire_process_slot(role='pidfd-exit-race-fixture', blocking=False)
    target = None
    try:
        target, _identity = _start_pidfd_supervisor_fixture(
            tmp_path, output_root, controller, lease)
        original_open = controller._pidfd_open

        def unavailable(_pid):
            raise PermissionError('pidfd unavailable fixture')

        monkeypatch.setattr(controller, '_pidfd_open', unavailable)
        with pytest.raises(PermissionError, match='pidfd unavailable'):
            controller._send_verified_supervisor_signal(
                '0', signal.SIGTERM, output_root=output_root)
        assert target.poll() is None

        monkeypatch.setattr(controller, '_pidfd_open', original_open)
        original_send = controller._pidfd_send_signal

        def exit_before_pidfd_send(pidfd, signum):
            target.kill()
            target.wait(timeout=3)
            return original_send(pidfd, signum)

        monkeypatch.setattr(controller, '_pidfd_send_signal', exit_before_pidfd_send)
        with pytest.raises(ProcessLookupError):
            controller._send_verified_supervisor_signal(
                '0', signal.SIGTERM, output_root=output_root)
        assert target.poll() is not None
    finally:
        if target is not None and target.poll() is None:
            target.kill()
            target.wait(timeout=3)
        lease.close()


def test_python_task_wrapper_reserves_before_exec_and_refuses_sixth_process(tmp_path,
                                                                            request):
    syntax = subprocess.run(['bash', '-n', str(PYTHON_TASK_WRAPPER)], capture_output=True,
                            text=True, timeout=10)
    assert syntax.returncode == 0, syntax.stderr
    pytest_lease = request.config._f0_process_slot
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import process_scheduler_root
    scheduler = Path(process_scheduler_root())
    env = dict(os.environ, F0_PROCESS_SCHEDULER_ROOT=str(scheduler))
    marker = tmp_path / 'started.json'
    script = ('import json, os; open(os.environ["F0_TEST_MARKER"], "w").write(json.dumps('
             '{"pid":os.getpid(),"owner":int(os.environ["F0_RESERVED_PROCESS_SLOT_OWNER_PID"]),'
             '"slot":int(os.environ["F0_RESERVED_PROCESS_SLOT_INDEX"])}))')
    env['F0_TEST_MARKER'] = str(marker)
    result = subprocess.run([str(PYTHON_TASK_WRAPPER), 'pytest-fixture', str(PYTHON),
                             '-c', script], cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    payload = json.loads(marker.read_text(encoding='utf-8'))
    assert payload['pid'] == payload['owner']
    assert payload['slot'] != pytest_lease.index

    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot
    leases = [acquire_process_slot(role='worker-fixture-{}'.format(i), blocking=False,
                                   scheduler_dir=str(scheduler)) for i in range(4)]
    sixth_marker = tmp_path / 'sixth-started'
    env['F0_TEST_MARKER'] = str(sixth_marker)
    try:
        refused = subprocess.run([str(PYTHON_TASK_WRAPPER), 'sixth-test', str(PYTHON),
                                  '-c', 'open(__import__("os").environ["F0_TEST_MARKER"],"w").write("bad")'],
                                 cwd=str(ROOT), env=env, capture_output=True,
                                 text=True, timeout=10)
        assert refused.returncode == 75
        assert not sixth_marker.exists()
    finally:
        for lease in leases:
            lease.close()


def test_supervisor_is_standard_library_only_and_starts_fresh_child_per_call(tmp_path, request):
    tree = ast.parse(LAUNCHER.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert not imported.intersection({'torch', 'stage_F0_run', 'stage_F0_queue',
                                     'stage_F0_data'})

    controller = _load_controller_module()
    fixture = tmp_path / 'one_call_child.py'
    fixture.write_text(
        'import os\n'
        'with open(os.environ["F0_TEST_PID_FILE"], "a", encoding="ascii") as f:\n'
        '    f.write(str(os.getpid()) + "\\n")\n', encoding='utf-8')
    pid_file = tmp_path / 'child-pids.txt'
    os.environ['F0_TEST_PID_FILE'] = str(pid_file)
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import acquire_process_slot, process_scheduler_root
    supervisor_lease = request.config._f0_process_slot
    child_lease = acquire_process_slot(
        role='F0-trajectory-child-0', blocking=False,
        scheduler_dir=process_scheduler_root())
    try:
        supervisor = controller.TrajectorySupervisor(
            worker_id='0', supervisor_slot_fd=supervisor_lease.fd,
            child_slot_fd=child_lease.lock.handle.fileno(),
            supervisor_slot_index=supervisor_lease.index,
            child_slot_index=child_lease.index,
            python=PYTHON, child_entry=fixture, output_root=tmp_path / 'runs')
        identity = controller.linux_process_identity(os.getpid())
        identity['nonce'] = 'fixture-supervisor'
        rows = [supervisor._spawn_one(invocation=n, identity=identity,
                                      child_args=['--fixture-only'])
                for n in (1, 2)]
    finally:
        child_lease.close()
        os.environ.pop('F0_TEST_PID_FILE', None)
    pids = [int(row[1]['child_pid']) for row in rows]
    recorded = [int(value) for value in pid_file.read_text().splitlines()]
    assert [row[0] for row in rows] == [0, 0]
    assert len(pids) == len(set(pids)) == 2
    assert recorded == pids
    assert all(row[2]['pgid'] == identity['pgid'] for row in rows)


def test_execution_gate_rejects_pending_old_or_stale_review_and_amendment(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import (control_hash_inventory,
                                         validate_execution_gate)

    gate_path, search_path, gate = _make_execution_gate_tree(tmp_path)
    with pytest.raises(PermissionError, match='reviewer verdict'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)
    gate['reviewer_verdict'] = 'PASS WITH FIXES'
    gate['scientific_code_hash_amendment']['reviewer_verdict'] = 'PASS WITH FIXES'
    gate['launch_authorized'] = True
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    assert validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                   search_gate_path=search_path)['execution_gate'] == gate

    gate['scientific_code_hash_amendment']['status'] = 'PENDING'
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(PermissionError, match='amendment is unreviewed'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)

    gate['scientific_code_hash_amendment']['status'] = 'APPROVED'
    frozen_search_bytes = search_path.read_bytes()
    changed_search = json.loads(frozen_search_bytes)
    changed_search['blocked_search_keys'] = ['changed-only-old-hold']
    search_path.write_text(json.dumps(changed_search), encoding='utf-8')
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(PermissionError, match='prior search gate hash'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)
    search_path.write_bytes(frozen_search_bytes)
    gate['control_hashes']['experiments/stage_F0_outputs/runs/run_search_worker.sh'] = 'stale'
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(PermissionError, match='hash inventory is stale'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)


@pytest.mark.parametrize('mutation', [
    'replace_new_key', 'truncate_to_previous_47', 'truncate_to_previous_45',
])
def test_execution_gate_rejects_noncanonical_preserved_hold_set(tmp_path, mutation):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import (PRESERVED_SIGNAL_CLEANUP_KEYS,
                                         validate_execution_gate)

    gate_path, search_path, gate = _make_execution_gate_tree(
        tmp_path, verdict='PASS', launch=True)
    held = list(gate['preserved_held_run_ids'])
    assert len(held) == 49
    if mutation == 'replace_new_key':
        held[held.index('7fd2006bdc8f1817a6f7')] = 'ffffffffffffffffffff'
    elif mutation == 'truncate_to_previous_47':
        held = [key for key in held
                if key not in ('7fd2006bdc8f1817a6f7', '1f04c7fb623d18f940ac')]
    else:
        held = [key for key in held
                if key not in ('b3a55740aadd0e3377a9', 'cfbf6cc0e541e76cfb92',
                               '7fd2006bdc8f1817a6f7', '1f04c7fb623d18f940ac')]
    gate['preserved_held_run_ids'] = sorted(held)
    assert len(set(gate['preserved_held_run_ids'])) == (
        49 if mutation == 'replace_new_key' else
        47 if mutation == 'truncate_to_previous_47' else 45)
    assert set(gate['preserved_held_run_ids']) != set(PRESERVED_SIGNAL_CLEANUP_KEYS)
    gate_path.write_text(json.dumps(gate), encoding='utf-8')

    with pytest.raises(PermissionError, match='exact frozen H49 hold set'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)


def test_pre_child_gate_rejects_before_dataset_probe(tmp_path):
    controller = _load_controller_module()
    execution_gate = tmp_path / 'invalid-execution-gate.json'
    execution_gate.write_text('{"schema":"old"}\n', encoding='utf-8')
    supervisor = controller.TrajectorySupervisor(
        worker_id='0', supervisor_slot_fd=0, child_slot_fd=0,
        supervisor_slot_index=0, child_slot_index=1,
        queue_path=tmp_path / 'missing-queue.json',
        search_gate_path=tmp_path / 'missing-search-gate.json',
        execution_gate_path=execution_gate,
        data_path=tmp_path / 'missing-dataset.csv',
        output_root=tmp_path / 'runs')
    with pytest.raises(PermissionError, match='execution-gate schema'):
        supervisor._check_pre_child_gates()


def test_queue_empty_receipt_allows_explicitly_held_unmaterialized_keys(tmp_path):
    controller = _load_controller_module()
    queue_path = tmp_path / 'queue.json'
    receipt_path = tmp_path / 'receipt.json'
    queue = {'queue_hash': 'frozen-queue-hash',
             'entries': [{'run_id': 'held-key'}, {'run_id': 'complete-key'}]}
    (tmp_path / 'queue_status').mkdir()
    (tmp_path / 'queue_status/complete-key.json').write_text(
        json.dumps({'run_id': 'complete-key', 'status': 'VALIDATION_COMPLETE'}),
        encoding='utf-8')
    queue_path.write_text(json.dumps(queue), encoding='utf-8')
    receipt_path.write_text(json.dumps({
        'schema': 'F0-CHILD-RECEIPT-v1', 'status': 'QUEUE_EMPTY',
        'queue_hash': queue['queue_hash'], 'held_run_ids': ['held-key'],
        'pid': 42002, 'ppid': 42001, 'pgid': 42001, 'worker_id': '0',
        'nonce': 'child-nonce', 'output_root': str(tmp_path.resolve()),
    }), encoding='utf-8')
    assert controller._validate_queue_empty_receipt(receipt_path, queue_path,
        tmp_path, child_pid=42002, supervisor_pid=42001, worker_pgid=42001,
        worker_id='0', nonce='child-nonce',
        expected_held_run_ids=['held-key'])['status'] == 'QUEUE_EMPTY'
    for bad in ({'pid': 999}, {'ppid': 999}, {'pgid': 999},
                {'worker_id': '1'}, {'nonce': 'forged-nonce'},
                {'held_run_ids': ['held-key', 'complete-key']}):
        receipt_path.write_text(json.dumps({
            'schema': 'F0-CHILD-RECEIPT-v1', 'status': 'QUEUE_EMPTY',
            'queue_hash': queue['queue_hash'], 'held_run_ids': ['held-key'],
            'pid': 42002, 'ppid': 42001, 'pgid': 42001, 'worker_id': '0',
            'nonce': 'child-nonce', 'output_root': str(tmp_path.resolve()), **bad,
        }), encoding='utf-8')
        with pytest.raises(PermissionError):
            controller._validate_queue_empty_receipt(
                receipt_path, queue_path, tmp_path, child_pid=42002,
                supervisor_pid=42001, worker_pgid=42001, worker_id='0',
                nonce='child-nonce', expected_held_run_ids=['held-key'])


def test_old_abnormal_worker_gate_requires_principal_release_and_keeps_holds(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_worker_control import validate_execution_gate

    gate_path, search_path, gate = _make_execution_gate_tree(
        tmp_path, verdict='PASS', launch=True, execution_status='PAUSED_ABNORMAL_WORKER_EXIT')
    gate['scientific_code_hash_amendment']['reviewer_verdict'] = 'PASS'
    gate['prior_incident_release']['approved_by_role'] = 'reviewer'
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(PermissionError, match='invalid or drops a protected key'):
        validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                search_gate_path=search_path)
    gate['prior_incident_release']['approved_by_role'] = 'principal'
    gate_path.write_text(json.dumps(gate), encoding='utf-8')
    assert validate_execution_gate(root=tmp_path, gate_path=gate_path,
                                   search_gate_path=search_path)


def test_completion_receipt_binds_frozen_row_locks_and_checkpoints(tmp_path, monkeypatch):
    controller = _load_controller_module()
    sys.path.insert(0, str(ROOT / 'experiments'))
    from stage_F0_queue import QueueFileLock
    from stage_F0_run import _write_completion_receipt

    output = tmp_path / 'runs'
    run_id = 'canonical-test-id'
    attempt_id = 'attempt-00'
    full_hash = 'a' * 64
    canonical_key = '{"dataset":"ml-1m","seed":2022}'
    queue_path = tmp_path / 'queue.json'
    queue_path.write_text(json.dumps({
        'entries': [{
            'run_id': run_id, 'run_id_full_sha256': full_hash,
            'canonical_key_json': canonical_key, 'attempt_id': attempt_id,
        }],
    }), encoding='utf-8')
    attempt = output / run_id / 'attempts' / attempt_id
    attempt.mkdir(parents=True)
    status_path = output / 'queue_status' / (run_id + '.json')
    status_path.parent.mkdir(parents=True)
    manifest_path = attempt / 'run_manifest.json'
    child_pid, supervisor_pid, worker_pgid = os.getpid(), os.getppid(), os.getpgrp()
    common = {
        'status': 'VALIDATION_COMPLETE', 'run_id': run_id,
        'run_id_full_sha256': full_hash,
        'canonical_key_json': canonical_key, 'attempt_id': attempt_id,
    }
    status_path.write_text(json.dumps({**common, 'pid': child_pid,
                                       'host': socket.gethostname(), 'gpu_index': 0}),
                           encoding='utf-8')
    manifest_path.write_text(json.dumps(common), encoding='utf-8')
    (attempt / 'final_checkpoint.pt').write_bytes(b'final checkpoint fixture')
    (attempt / 'selected_checkpoint.pt').write_bytes(b'selected checkpoint fixture')
    (output / 'canonical_locks').mkdir(parents=True)
    gpu_owner = {
        'pid': child_pid, 'host': socket.gethostname(), 'gpu_index': 0,
        'gpu_uuid': 'gpu-fixture-uuid', 'run_id': run_id,
        'run_id_full_sha256': full_hash, 'attempt_id': attempt_id,
    }
    (output / 'canonical_locks' / (run_id + '.lock')).write_text(json.dumps({
        'pid': child_pid, 'run_id': run_id, 'run_id_full_sha256': full_hash,
        'canonical_key_json': canonical_key, 'attempt_id': attempt_id,
        'gpu_uuid': 'gpu-fixture-uuid', 'host': socket.gethostname(),
    }), encoding='utf-8')
    (output / 'gpu_locks').mkdir(parents=True)
    nonce = 'fixture-child-nonce'
    receipt_path = tmp_path / 'invocation-receipt.json'
    receipt_path_value = os.environ.get('F0_INVOCATION_RECEIPT')
    worker_id_value = os.environ.get('F0_WORKER_ID')
    nonce_value = os.environ.get('F0_INVOCATION_NONCE')
    monkeypatch.setenv('F0_INVOCATION_RECEIPT', str(receipt_path))
    monkeypatch.setenv('F0_WORKER_ID', '0')
    monkeypatch.setenv('F0_INVOCATION_NONCE', nonce)
    gpu_lock = QueueFileLock(str(output / 'gpu_locks/gpu0.lock'))
    gpu_lock.write_owner(gpu_owner)
    _write_completion_receipt(
        str(receipt_path), entry={**common, 'queue_hash': 'frozen-queue-hash'},
        output_root=str(output), status_file=str(status_path), attempt_dir=str(attempt),
        summary={'status': 'VALIDATION_COMPLETE'}, gpu_lock=gpu_lock)
    gpu_lock.close()
    peer_lock = QueueFileLock(str(output / 'gpu_locks/gpu0.lock'))
    peer_lock.write_owner({
        'pid': 42099, 'host': socket.gethostname(), 'gpu_index': 0,
        'gpu_uuid': 'gpu-fixture-uuid', 'run_id': 'peer-run',
        'run_id_full_sha256': 'b' * 64, 'attempt_id': 'attempt-peer',
    })
    peer_lock.close()
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

    # A peer may legally take the released GPU lock before supervisor receipt
    # validation; the committed receipt must continue to validate its frozen
    # owner snapshot rather than the mutable current owner file.
    validated = controller._validate_completion_receipt(
        receipt_path, child_pid=child_pid, nonce=nonce, worker_id='0',
        supervisor_pid=supervisor_pid, worker_pgid=worker_pgid,
        queue_path=queue_path, output_root=output)
    assert validated['run_id'] == run_id

    receipt_path.write_text(json.dumps({**receipt, 'pid': child_pid + 1}),
                            encoding='utf-8')
    with pytest.raises(PermissionError, match='invalid identity/status'):
        controller._validate_completion_receipt(
            receipt_path, child_pid=child_pid, nonce=nonce, worker_id='0',
            supervisor_pid=supervisor_pid, worker_pgid=worker_pgid,
            queue_path=queue_path, output_root=output)

    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    (attempt / 'selected_checkpoint.pt').write_bytes(b'changed checkpoint')
    with pytest.raises(PermissionError, match='checkpoint transaction validation failed'):
        controller._validate_completion_receipt(
            receipt_path, child_pid=child_pid, nonce=nonce, worker_id='0',
            supervisor_pid=supervisor_pid, worker_pgid=worker_pgid,
            queue_path=queue_path, output_root=output)
    for name, value in (('F0_INVOCATION_RECEIPT', receipt_path_value),
                        ('F0_WORKER_ID', worker_id_value),
                        ('F0_INVOCATION_NONCE', nonce_value)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
