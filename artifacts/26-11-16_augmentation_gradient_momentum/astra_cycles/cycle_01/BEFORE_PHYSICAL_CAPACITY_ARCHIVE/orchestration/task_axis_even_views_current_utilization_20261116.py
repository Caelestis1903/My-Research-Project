"""Separately reviewed foreground admission overlay; scientific workers stay frozen."""
import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from experiments.task_axis_even_views_20261116 import protocol as p
from experiments.task_axis_even_views_20261116 import runner as frozen


FROZEN_SOURCE = '6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1'
OVERLAY = Path(__file__).resolve()
POLICY = p.CYCLE/'CURRENT_UTILIZATION_POLICY.md'
RECEIPT = 'CURRENT_UTILIZATION_EXECUTION_POLICY.jsonl'
ADMISSION = dict(name='CURRENT_UTILIZATION',
    predicate='current_gpu_memory/total < 0.85 AND current_gpu_power/limit < 0.85 AND current_host_ram < 0.90',
    independent_cards=True, pending_allocation_ack=True, resample_after_launch=True,
    predictive_reservations=False, active_worker_cap=None)


def choose_device(devices, peak_bytes, power_watts, pending=(), host_bytes=0, rows=None):
    # The frozen call signature carries historical budgets; none are admission inputs.
    if not frozen.ram_usage() < .90:
        return None
    for row in frozen.gpu_status() if rows is None else rows:
        gpu = int(row['index'])
        if gpu in devices and gpu not in pending:
            if row['memory']/row['total'] < .85 and row['power']/row['limit'] < .85:
                return gpu
    return None


def policy_binding():
    source = p.source_manifest()
    assert source['sha256'] == FROZEN_SOURCE, 'Frozen scientific worker source changed'
    return dict(overlay_path=str(OVERLAY), overlay_sha256=p.sha_file(OVERLAY),
        policy_path=str(POLICY), policy_sha256=p.sha_file(POLICY),
        worker_source_hash=FROZEN_SOURCE, design_bundle_sha256=p.BUNDLE, admission=ADMISSION)


def check_overlay_review(review_path):
    review = json.loads(Path(review_path).read_text())
    assert review['verdict'] in ('PASS', 'PASS WITH FIXES') and not review.get('blocking_issues')
    assert review['role'] == 'NEW_DISTINCT_CURRENT_UTILIZATION_REVIEWER'
    assert review['independent_of_coder'] is True and review['reviewer_agent_id']
    assert review['cpu_checks_passed'] is True
    binding = policy_binding()
    assert review['execution_policy'] == binding, 'Overlay source/policy/worker review binding changed'
    return binding, review


def label_progress(value, receipt_reference):
    value = dict(value, admission_policy=ADMISSION, execution_policy_receipt=receipt_reference)
    if 'reservations' in value:
        value['historical_non_admission_reservations'] = value.pop('reservations')
    return value


def run(attempt, devices, review_path, worker_review_path):
    attempt = p.cycle_path(attempt)
    assert not p.stop_requested(), 'User stop prohibits dispatch'
    assert not (attempt/'PAUSE_TRAINING').exists(), 'Use existing explicit ACK recovery before RUN'
    binding, review = check_overlay_review(review_path)
    manifest = frozen.validate_attempt(attempt, worker_review_path, require_smoke=True)
    assert manifest['source_hash'] == FROZEN_SOURCE
    status, owner = frozen.require_storage(attempt)
    assert not frozen.live_worker_pids(attempt, status), 'Current runner must quiesce and hand off first'
    assert not p.stop_requested() and not (attempt/'PAUSE_TRAINING').exists()
    receipt = dict(binding, overlay_review_path=str(Path(review_path).resolve()),
        overlay_review_sha256=p.sha_file(review_path), reviewer_agent_id=review['reviewer_agent_id'],
        worker_review_path=str(Path(worker_review_path).resolve()),
        worker_review_sha256=p.sha_file(worker_review_path), owner=owner,
        scheduler_pid=os.getpid(), observed_at=time.time(), test_accessed=False)
    receipt_path = attempt/RECEIPT
    with receipt_path.open('a') as stream:
        stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n')
    reference = dict(path=str(receipt_path), record_sha256=p.digest(receipt))
    original_write_json = frozen.write_json

    def write_progress(path, value):
        if Path(path) in (attempt/'progress.json', attempt/'COMPLETE_RUNNER_REPORT.json'):
            value = label_progress(value, reference)
        return original_write_json(path, value)

    # Workers still execute frozen.MODULE in fresh processes with unchanged job keys.
    with patch.object(frozen, 'choose_device', choose_device), patch.object(frozen, 'write_json', write_progress):
        return label_progress(frozen.schedule(attempt, devices, worker_review_path), reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run',))
    parser.add_argument('--attempt', type=Path, default=p.ATTEMPT)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--review', type=Path, required=True, help='Fresh independent overlay review JSON')
    parser.add_argument('--worker-review', type=Path, required=True, help='Existing frozen scientific review JSON')
    args = parser.parse_args()
    print(json.dumps(run(args.attempt, [int(v) for v in args.devices.split(',')],
        args.review, args.worker_review), allow_nan=False))


if __name__ == '__main__':
    main()
