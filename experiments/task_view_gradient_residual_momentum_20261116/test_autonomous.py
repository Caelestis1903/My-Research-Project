"""Small CLI-event and stop-policy fixtures; no research/training dispatch."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import autonomous as a


@pytest.fixture
def roots(tmp_path):
    workspace, control = tmp_path / 'execution', tmp_path / 'control'
    a.update_state(workspace, control, status='ACTIVE', next_action='independent review',
                   manual_stop_requested=False, completed_trajectories=7)
    return workspace, control


def test_actual_thread_event_and_exact_resume(roots):
    workspace, control = roots
    event = {'type': 'thread.started', 'thread_id': '01a1dfb8-605b-7430-8462-5fd3da0b048f'}
    a.consume_event(json.dumps(event), workspace, control)
    a.consume_event(json.dumps({'type': 'item.completed',
                                'item': {'id': 'item_4', 'type': 'agent_message', 'text': 'DONE'}}), workspace, control)
    state = a.read_state(workspace)
    assert state['parent_thread_started_event'] == event
    assert state['last_event_id'] == 'item_4'
    assert state['completed_trajectories'] == 7
    assert a.codex_command(workspace, event['thread_id'])[:4] == ['codex', 'exec', 'resume', event['thread_id']]
    initial = a.codex_command(workspace)
    assert '--json' in initial and '--cd' in initial and 'gpt-6.1-sol' in initial
    assert 'model_reasoning_effort="xhigh"' in initial
    assert '--last' not in initial and 'gpt-6-astra' not in initial
    with pytest.raises(RuntimeError, match='expected'):
        a.consume_event('{"type":"thread.started","thread_id":"different"}', workspace, control)
    with pytest.raises(ValueError):
        a.consume_event('invalid event', workspace, control)


@pytest.mark.parametrize('location', ['execution_flag', 'control_flag', 'execution_sentinel', 'control_sentinel'])
def test_stop_is_sticky_across_state_updates(roots, location):
    workspace, control = roots
    root = control if location.startswith('control') else workspace
    if location.endswith('flag'):
        a.update_state(root, root, manual_stop_requested=True)
    else:
        a.atomic_text(root / a.REL / 'STOP_AUTONOMOUS', 'stop\n')
    assert a.stop_requested(workspace, control)
    a.update_state(workspace, control, manual_stop_requested=False, current_stage='REVIEW')
    assert a.read_state(workspace)['manual_stop_requested'] is True
    assert a.read_state(workspace)['completed_trajectories'] == 7


@pytest.mark.parametrize('failure', [False, True])
def test_cli_event_fixture_records_completion_or_failure(roots, monkeypatch, failure):
    workspace, control = roots
    events = [{'type': 'thread.started', 'thread_id': 'fixture-session'}, {'type': 'turn.started'}]
    events.append({'type': 'turn.failed', 'error': {'message': 'fixture CLI failure'}} if failure
                  else {'type': 'turn.completed', 'usage': {}})
    script = ('import sys; sys.stdin.read(); '
              f'print({repr("\n".join(json.dumps(x) for x in events))}); '
              f'print("fixture stderr", file=sys.stderr); sys.exit({int(failure)})')
    real_popen = subprocess.Popen
    commands = []

    def fixture_popen(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, '-c', script], **kwargs)

    monkeypatch.setattr(a.subprocess, 'Popen', fixture_popen)
    if failure:
        with pytest.raises(RuntimeError, match='fixture CLI failure'):
            a.run_turn(workspace, control, 'CLI parser fixture only')
    else:
        a.run_turn(workspace, control, 'CLI parser fixture only')
        a.run_turn(workspace, control, 'resume parser fixture only')
        assert commands[1][:4] == ['codex', 'exec', 'resume', 'fixture-session']
    state = a.read_state(workspace)
    assert state['last_cli_exit_code'] == int(failure)
    assert Path(state['last_cli_events']).read_text().count('thread.started') == 1
    assert 'fixture stderr' in Path(state['last_cli_stderr']).read_text()
    assert state['completed_trajectories'] == 7
    if failure:
        assert state['last_cli_error']['error']['message'] == 'fixture CLI failure'


def test_stop_before_first_dispatch_produces_summary(roots, monkeypatch):
    workspace, control = roots
    a.request_stop(control, control)
    monkeypatch.setattr(a, 'run_turn', lambda *args: pytest.fail('must not dispatch'))
    a.drive(workspace, control)
    state = a.read_state(workspace)
    assert state['status'] == state['termination_reason'] == 'USER_STOPPED'
    assert Path(state['stop_summary_path']).is_file()
    assert state['completed_trajectories'] == 7
    assert not a.test_allowed(state)


def test_incomplete_session_recovery_pauses_without_new_dispatch(roots, monkeypatch):
    workspace, control = roots
    a.update_state(workspace, control, driver_turns_started=1)
    a.request_stop(control, control)
    monkeypatch.setattr(a, 'run_turn', lambda *args: pytest.fail('must not replace session'))
    with pytest.raises(RuntimeError, match='Missing parent session ID'):
        a.drive(workspace, control, recover=True)
    state = a.read_state(workspace)
    assert state['status'] == 'PAUSED' and state['termination_reason'] is None
    assert state['manual_stop_requested'] is True
    assert 'Missing parent session ID' in state['pause_reason']
    assert Path(state['stop_summary_path']).is_file()


def test_cli_spawn_failure_is_recorded_and_can_finalize_on_recovery(roots, monkeypatch):
    workspace, control = roots

    def unavailable(*args, **kwargs):
        raise FileNotFoundError('fixture codex executable unavailable')

    monkeypatch.setattr(a.subprocess, 'Popen', unavailable)
    with pytest.raises(FileNotFoundError, match='unavailable'):
        a.drive(workspace, control)
    state = a.read_state(workspace)
    assert state['status'] == 'PAUSED' and state['termination_reason'] is None
    assert state['last_cli_launched'] is False and state.get('parent_session_id') is None
    assert 'unavailable' in state['last_cli_error']
    a.request_stop(control, control)
    a.drive(workspace, control, recover=True)
    assert a.read_state(workspace)['status'] == 'USER_STOPPED'


def test_test_gate_needs_manual_stop_freeze_and_selected_ram():
    state = dict(status='USER_STOPPED', termination_reason='USER_STOPPED', manual_stop_requested=True,
                 final_candidate_eligible=True, final_freeze=dict(
                     complete=True, selected_ram_valid=True, selected_state_receipt='live guardian check',
                     selected_states=[dict(dataset=d, seed=s, selected_sha256=f'hash-{d}-{s}', guardian_ref='live-ref')
                                      for d in ('movielens', 'kuairand') for s in range(2022, 2027)]))
    assert a.test_allowed(state)
    for key, value in [('status', 'ACTIVE'), ('status', 'PAUSED'), ('termination_reason', None),
                       ('manual_stop_requested', False), ('final_candidate_eligible', False)]:
        changed = dict(state, **{key: value})
        assert not a.test_allowed(changed)
    for key, value in [('complete', False), ('selected_ram_valid', False), ('selected_state_receipt', None),
                       ('selected_states', state['final_freeze']['selected_states'][:-1])]:
        changed = copy.deepcopy(state)
        changed['final_freeze'][key] = value
        assert not a.test_allowed(changed)
    changed = copy.deepcopy(state)
    changed['final_freeze']['selected_states'][0]['selected_sha256'] = None
    assert not a.test_allowed(changed)
