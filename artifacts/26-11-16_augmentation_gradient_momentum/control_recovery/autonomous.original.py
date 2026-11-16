"""Durable outer loop for one native Codex parent; no training or analysis here."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path('/home/xzf/Code/phase0-3')
REL = Path('artifacts/26-11-16_augmentation_gradient_momentum')
MODEL = 'gpt-6.1-sol'
PROMPT = Path(__file__).with_name('AUTONOMOUS_MAIN_PROMPT.md')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_state(workspace):
    path = Path(workspace) / REL / 'AUTONOMOUS_STATE.json'
    return json.loads(path.read_text()) if path.exists() else {}


def stop_requested(workspace=ROOT, control_workspace=CONTROL):
    for root in {Path(workspace), Path(control_workspace)}:
        directory = root / REL
        if (directory / 'STOP_AUTONOMOUS').exists():
            return True
        state = read_state(root)
        if state.get('manual_stop_requested') or state.get('termination_reason') == 'USER_STOPPED':
            return True
    return False


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_state(workspace=ROOT, control_workspace=CONTROL, **changes):
    """Cooperating writers use this merge so a stale snapshot cannot clear stop."""
    directory = Path(workspace) / REL
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state(workspace)
        stopped = state.get('manual_stop_requested', False) or stop_requested(workspace, control_workspace)
        state.update(changes)
        state['manual_stop_requested'] = bool(stopped or state.get('manual_stop_requested'))
        state['updated_at_utc'] = now()
        atomic_text(directory / 'AUTONOMOUS_STATE.json', json.dumps(state, indent=2) + '\n')
    return state


def request_stop(workspace=ROOT, control_workspace=CONTROL):
    # The sentinel also protects against writers outside the state lock protocol.
    atomic_text(Path(workspace) / REL / 'STOP_AUTONOMOUS', now() + '\n')
    return update_state(workspace, control_workspace, manual_stop_requested=True)


def test_allowed(state):
    freeze = state.get('final_freeze') or {}
    selected = freeze.get('selected_states') or []
    expected = {(d, s) for d in ('movielens', 'kuairand') for s in range(2022, 2027)}
    identities = {(x.get('dataset'), x.get('seed')) for x in selected}
    return bool(
        state.get('manual_stop_requested')
        and state.get('termination_reason') == 'USER_STOPPED'
        and state.get('status') == 'USER_STOPPED'
        and state.get('final_candidate_eligible')
        and freeze.get('complete')
        and freeze.get('selected_ram_valid')
        and freeze.get('selected_state_receipt')
        and identities == expected and len(selected) == len(expected)
        and all(x.get('selected_sha256') and x.get('guardian_ref') for x in selected)
    )


# Prevent pytest collecting the policy predicate as a test.
test_allowed.__test__ = False


def codex_command(workspace, session_id=None):
    command = ['codex', 'exec']
    if session_id:
        command += ['resume', session_id]
    command += ['--json', '--model', MODEL, '-c', 'model_reasoning_effort="xhigh"',
                '--dangerously-bypass-approvals-and-sandbox']
    if not session_id:
        command += ['--cd', str(workspace)]
    return command + ['-']


def consume_event(line, workspace, control_workspace):
    event = json.loads(line)
    kind = event['type']
    changes = {'last_event_type': kind}
    if event.get('id') is not None:
        changes['last_event_id'] = event['id']
    elif isinstance(event.get('item'), dict) and event['item'].get('id') is not None:
        changes['last_event_id'] = event['item']['id']
    if kind == 'thread.started':
        session_id = event['thread_id']
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('thread.started has no valid thread_id')
        old = read_state(workspace).get('parent_session_id')
        if old and old != session_id:
            raise RuntimeError(f'CLI returned thread {session_id}, expected {old}')
        changes['parent_session_id'] = session_id
        changes['parent_thread_started_event'] = event
    if kind in ('turn.failed', 'error'):
        changes['last_cli_error'] = event
    # Item text is untrusted log content; it never sets stage/candidate completion.
    update_state(workspace, control_workspace, **changes)
    return kind


def run_turn(workspace, control_workspace, prompt):
    state = read_state(workspace)
    turn = state.get('driver_turns_started', 0) + 1
    directory = Path(workspace) / REL / 'driver_logs'
    directory.mkdir(parents=True, exist_ok=True)
    events = directory / f'turn_{turn:06d}.jsonl'
    errors = directory / f'turn_{turn:06d}.stderr.log'
    command = codex_command(workspace, state.get('parent_session_id'))
    update_state(workspace, control_workspace, driver_turns_started=turn,
                 last_cli_command=command, last_cli_events=str(events),
                 last_cli_stderr=str(errors), last_cli_error=None, last_cli_exit_code=None,
                 last_cli_launched=False)
    completed = False
    failed = None
    with events.open('ab', buffering=0) as output, errors.open('ab', buffering=0) as stderr:
        process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr)
        update_state(workspace, control_workspace, parent_cli_pid=process.pid, last_cli_launched=True)
        try:
            process.stdin.write(prompt.encode())
            process.stdin.close()
            buffer = b''
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if stop_requested(workspace, control_workspace):
                        update_state(workspace, control_workspace, manual_stop_requested=True)
                    for key, _ in selector.select(timeout=.5):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        output.write(chunk)
                        buffer += chunk
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            if not line.strip():
                                continue
                            try:
                                kind = consume_event(line, workspace, control_workspace)
                                completed |= kind == 'turn.completed'
                                if kind in ('turn.failed', 'error'):
                                    failed = line.decode(errors='replace')
                            except (ValueError, KeyError, RuntimeError) as error:
                                failed = str(error)
                if buffer.strip():
                    try:
                        kind = consume_event(buffer, workspace, control_workspace)
                        completed |= kind == 'turn.completed'
                        if kind in ('turn.failed', 'error'):
                            failed = buffer.decode(errors='replace')
                    except (ValueError, KeyError, RuntimeError) as error:
                        failed = str(error)
            code = process.wait()
        finally:
            process.stdout.close()
        os.fsync(output.fileno())
    update_state(workspace, control_workspace, parent_cli_pid=None, last_cli_exit_code=code)
    if code or failed or not completed or not read_state(workspace).get('parent_session_id'):
        tail = errors.read_text(errors='replace')[-4000:]
        raise RuntimeError(f'Codex CLI exit={code}; turn.completed={completed}; error={failed}; stderr={tail}')


def stop_summary(workspace, control_workspace):
    state = update_state(workspace, control_workspace, manual_stop_requested=True)
    path = Path(workspace) / REL / 'AUTONOMOUS_STOP_SUMMARY.md'
    fields = ('current_cycle', 'current_stage', 'current_candidate', 'candidate_decision',
              'completed_trajectories', 'planned_trajectories', 'report_path',
              'parent_session_id', 'pause_reason', 'next_action', 'stop_finalization',
              'final_freeze', 'champion')
    atomic_text(path, '# Manual stop summary\n\n' + now() + '\n\n' +
                '\n'.join(f'- {key}: {json.dumps(state.get(key), ensure_ascii=False)}' for key in fields) +
                '\n\nFull and partial research results: follow the recorded report and raw evidence paths.\n')
    update_state(workspace, control_workspace, stop_summary_path=str(path))


def stop_finalized(state):
    final = state.get('stop_finalization') or {}
    return bool(final.get('complete') and final.get('atomic_operation_complete') and final.get('summary_path'))


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def resume_prompt(workspace, control_workspace, finalize=False):
    mode = ('FINALIZE ONLY. Manual stop is latched. Dispatch no new research or training. '
            'Wait for existing workers to commit a safe atomic state; collect partial results, '
            'write the final research summary and stop_finalization receipt. '
            'Set complete and atomic_operation_complete only after workers are quiescent. '
            'Record summary_path. Keep eligible selected RAM states alive for a final freeze. '
            'Do not open Test in this turn.' if finalize else
            'Continue the exact recorded next_action. Inspect existing stage receipts first. '
            'Complete remaining work and record the next concrete action before yielding. '
            'A candidate NO-GO or Astra STOP requires fresh native Astra research, not termination.')
    return (mode + f'\nWorkspace: {workspace}\nControl workspace: {control_workspace}\n' +
            f'Reread {PROMPT} and {Path(workspace) / REL / "AUTONOMOUS_STATE.json"}. '
            'Check both stop controls before every child dispatch. Use update_state for all state merges.\n')


def drive(workspace, control_workspace, recover=False):
    directory = Path(workspace) / REL
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_state(workspace)
        if state.get('status') == 'USER_STOPPED' and stop_finalized(state):
            stop_summary(workspace, control_workspace)
            return
        try:
            if state.get('status') == 'PAUSED' and not recover:
                raise RuntimeError('PAUSED: repair recorded cause, then start --recover to resume exact thread')
            if process_alive(state.get('parent_cli_pid')):
                raise RuntimeError('Previous parent CLI is still alive; wait/reconcile its actual log before resuming')
            if (state.get('driver_turns_started', 0) and state.get('last_cli_launched', True)
                    and not state.get('parent_session_id')):
                raise RuntimeError('Missing parent session ID after dispatch; recover from actual JSONL before resuming')
            update_state(workspace, control_workspace, status='ACTIVE', driver_pid=os.getpid(),
                         driver_status='RUNNING', pause_reason=None)
            while True:
                state = read_state(workspace)
                if stop_requested(workspace, control_workspace):
                    update_state(workspace, control_workspace, manual_stop_requested=True)
                    stop_summary(workspace, control_workspace)
                    if state.get('parent_session_id') and not stop_finalized(state):
                        run_turn(workspace, control_workspace, resume_prompt(workspace, control_workspace, True))
                        state = read_state(workspace)
                        if not stop_finalized(state):
                            raise RuntimeError('Stop finalizer did not confirm atomic completion and summary')
                    elif not state.get('parent_session_id'):
                        update_state(workspace, control_workspace, stop_finalization=dict(
                            complete=True, atomic_operation_complete=True,
                            summary_path=str(directory / 'AUTONOMOUS_STOP_SUMMARY.md')))
                    update_state(workspace, control_workspace, status='USER_STOPPED',
                                 driver_status='USER_STOPPED',
                                 termination_reason='USER_STOPPED', next_action='FINAL_FREEZE_OR_NO_ELIGIBLE_CANDIDATE')
                    stop_summary(workspace, control_workspace)
                    return
                if state.get('status') == 'PAUSED':
                    update_state(workspace, control_workspace, driver_status='PAUSED')
                    return
                if state.get('status') != 'ACTIVE' or state.get('termination_reason'):
                    raise RuntimeError('Only manual stop may terminate research; repair parent state')
                if not state.get('next_action'):
                    raise RuntimeError('Completed parent turn has no concrete next_action')
                prompt = (resume_prompt(workspace, control_workspace) if state.get('parent_session_id')
                          else PROMPT.read_text() + f'\nEXECUTION_WORKSPACE={workspace}\nCONTROL_WORKSPACE={control_workspace}\n')
                # Recheck immediately at dispatch; the parent checks again before native children.
                if stop_requested(workspace, control_workspace):
                    continue
                run_turn(workspace, control_workspace, prompt)
        except Exception as error:
            state = read_state(workspace)
            update_state(workspace, control_workspace, status='PAUSED', termination_reason=None,
                         driver_status='PAUSED', pause_reason=str(error),
                         last_cli_error=state.get('last_cli_error') or str(error),
                         recovery_next_action=state.get('next_action'),
                         next_action='Repair recorded technical error; start --recover with exact parent_session_id')
            if stop_requested(workspace, control_workspace):
                stop_summary(workspace, control_workspace)
            raise
        finally:
            update_state(workspace, control_workspace, driver_pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'status', 'stop'))
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--control-workspace', type=Path, default=CONTROL)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    control = args.control_workspace.resolve()
    if args.action == 'stop':
        state = request_stop(workspace, control)
        print(json.dumps(state, indent=2))
    elif args.action == 'status':
        print(json.dumps(read_state(workspace), indent=2))
    elif args.action == 'start':
        directory = workspace / REL
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), 'run', '--workspace', str(workspace),
                   '--control-workspace', str(control)] + (['--recover'] if args.recover else [])
        with (directory / 'driver.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=workspace, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        print(json.dumps({'launched_pid': process.pid, 'state_path': str(directory / 'AUTONOMOUS_STATE.json')}))
    else:
        drive(workspace, control, args.recover)


if __name__ == '__main__':
    main()
