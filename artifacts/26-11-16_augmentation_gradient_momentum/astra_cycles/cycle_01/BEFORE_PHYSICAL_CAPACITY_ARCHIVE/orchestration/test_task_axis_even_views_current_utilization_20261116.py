"""Focused CPU fixtures; all hardware, storage and scheduler effects are mocked."""
import json

import pytest

from orchestration import task_axis_even_views_current_utilization_20261116 as overlay


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Hardware/IPC/worker launch is outside this CPU fixture')
    monkeypatch.setattr(overlay.frozen, 'ram_usage', lambda: .15)
    monkeypatch.setattr(overlay.frozen, 'gpu_status', forbidden)
    monkeypatch.setattr(overlay.frozen, 'request', forbidden)
    monkeypatch.setattr(overlay.p, 'request', forbidden)
    monkeypatch.setattr(overlay.frozen.subprocess, 'Popen', forbidden)


def row(gpu=1, **changes):
    return dict(dict(index=gpu, memory=8522, total=24576, power=265.76, limit=350), **changes)


def test_current_measurements_ignore_all_projected_budgets(monkeypatch):
    monkeypatch.setattr(overlay.frozen, 'host_admissible', lambda _: True)
    assert overlay.frozen.choose_device([1], 5*1024**3, 195.61, rows=[row()]) is None
    for power in (195.61, 206.31, 10**20):
        assert overlay.choose_device([1], 10**20, power, host_bytes=10**20, rows=[row()]) == 1


@pytest.mark.parametrize('change', [dict(memory=24576*.85), dict(memory=24576*.86),
    dict(power=350*.85), dict(power=350*.86)])
def test_either_current_gpu_resource_at_or_above_85_rejects(change):
    assert overlay.choose_device([1], 0, 0, rows=[row(**change)]) is None


@pytest.mark.parametrize('ram, expected', [(.899, 1), (.90, None), (.91, None)])
def test_current_host_ram_gate(monkeypatch, ram, expected):
    monkeypatch.setattr(overlay.frozen, 'ram_usage', lambda: ram)
    assert overlay.choose_device([1], 0, 0, rows=[row()]) == expected


def test_cards_independent_and_pending_allocation_ack_respected():
    choose = overlay.choose_device
    assert choose([0, 1], 0, 0, rows=[row(0, power=300), row(1)]) == 1
    assert choose([0, 1], 0, 0, rows=[row(0), row(1, memory=24000)]) == 0
    assert choose([0, 1], 0, 0, pending={0}, rows=[row(0), row(1)]) == 1
    assert choose([0, 1], 0, 0, pending={0, 1}, rows=[row(0), row(1)]) is None
    assert choose([0], 0, 0, rows=[row(1)]) is None


def test_no_active_process_count_cap_and_current_resampling(monkeypatch):
    samples = iter([row()]*100+[row(power=300), row()])
    monkeypatch.setattr(overlay.frozen, 'gpu_status', lambda: [next(samples)])
    for _ in range(100):
        assert overlay.choose_device([1], 0, 0) == 1
    assert overlay.choose_device([1], 0, 0) is None
    assert overlay.choose_device([1], 0, 0) == 1


@pytest.fixture
def reviewed(tmp_path):
    binding = overlay.policy_binding()
    review = dict(verdict='PASS', blocking_issues=[], cpu_checks_passed=True,
        role='NEW_DISTINCT_CURRENT_UTILIZATION_REVIEWER', independent_of_coder=True,
        reviewer_agent_id='fixture-distinct-reviewer', execution_policy=binding)
    path = tmp_path/'review.json'
    path.write_text(json.dumps(review))
    return path, review


@pytest.mark.parametrize('change', ['old_review', 'overlay', 'policy', 'worker', 'blocking', 'not_independent'])
def test_overlay_review_rejects_wrong_binding(reviewed, change):
    path, review = reviewed
    if change in ('overlay', 'policy', 'worker'):
        field = dict(overlay='overlay_sha256', policy='policy_sha256', worker='worker_source_hash')[change]
        review['execution_policy'][field] = '0'*64
    elif change == 'old_review':
        review['role'] = 'NEW_DISTINCT_RESOURCE_1_REVIEWER'
    elif change == 'blocking':
        review['blocking_issues'] = ['unresolved']
    else:
        review['independent_of_coder'] = False
    path.write_text(json.dumps(review))
    with pytest.raises(AssertionError):
        overlay.check_overlay_review(path)


@pytest.mark.parametrize('gate, message', [('stop', 'User stop'), ('pause', 'explicit ACK'),
    ('live_worker', 'quiesce and hand off')])
def test_run_preflight_leaves_no_policy_receipt(monkeypatch, tmp_path, reviewed, gate, message):
    review_path, _ = reviewed
    monkeypatch.setattr(overlay.p, 'cycle_path', lambda _: tmp_path)
    monkeypatch.setattr(overlay.p, 'stop_requested', lambda: gate == 'stop')
    monkeypatch.setattr(overlay.frozen, 'validate_attempt', lambda *a, **kw: dict(source_hash=overlay.FROZEN_SOURCE))
    monkeypatch.setattr(overlay.frozen, 'require_storage', lambda _: ({}, {}))
    monkeypatch.setattr(overlay.frozen, 'live_worker_pids', lambda *a: [123])
    if gate == 'pause':
        (tmp_path/'PAUSE_TRAINING').touch()
    with pytest.raises(AssertionError, match=message):
        overlay.run(tmp_path, [0, 1], review_path, tmp_path/'unused_worker_review.json')
    assert not (tmp_path/overlay.RECEIPT).exists()


def test_run_appends_separate_receipt_before_schedule_and_labels_outputs(monkeypatch, tmp_path, reviewed):
    review_path, review = reviewed
    worker_review = tmp_path/'worker_review.json'
    worker_review.write_text('{"verdict":"PASS"}')
    monkeypatch.setattr(overlay.p, 'cycle_path', lambda _: tmp_path)
    monkeypatch.setattr(overlay.p, 'stop_requested', lambda: False)
    monkeypatch.setattr(overlay.frozen, 'validate_attempt', lambda *a, **kw: dict(source_hash=overlay.FROZEN_SOURCE))
    monkeypatch.setattr(overlay.frozen, 'require_storage', lambda _: ({}, dict(pid=1570727, owner_tool_session='59365')))
    monkeypatch.setattr(overlay.frozen, 'live_worker_pids', lambda *a: [])
    original_choose = overlay.frozen.choose_device
    original_write = overlay.frozen.write_json

    def schedule(attempt, devices, worker_review_path):
        records = [json.loads(line) for line in (tmp_path/overlay.RECEIPT).read_text().splitlines()]
        assert records[-1]['overlay_sha256'] == review['execution_policy']['overlay_sha256']
        assert records[-1]['overlay_review_sha256'] == overlay.p.sha_file(review_path)
        assert records[-1]['worker_source_hash'] == overlay.FROZEN_SOURCE
        assert devices == [0, 1] and worker_review_path == worker_review
        assert overlay.frozen.choose_device is overlay.choose_device
        result = dict(status='COMPLETE', reservations={'kuairand': dict(power=206.31)})
        for name in ('progress.json', 'COMPLETE_RUNNER_REPORT.json'):
            overlay.frozen.write_json(attempt/name, result)
            data = json.loads((attempt/name).read_text())
            assert data['admission_policy'] == overlay.ADMISSION
            assert 'reservations' not in data
            assert data['historical_non_admission_reservations'] == result['reservations']
            assert data['execution_policy_receipt']['record_sha256'] == overlay.p.digest(records[-1])
        return result

    monkeypatch.setattr(overlay.frozen, 'schedule', schedule)
    for _ in range(2):
        result = overlay.run(tmp_path, [0, 1], review_path, worker_review)
        assert result['admission_policy']['name'] == 'CURRENT_UTILIZATION' and 'reservations' not in result
    assert len((tmp_path/overlay.RECEIPT).read_text().splitlines()) == 2
    assert overlay.frozen.choose_device is original_choose and overlay.frozen.write_json is original_write
