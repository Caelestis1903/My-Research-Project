#!/usr/bin/env python3
"""Restore AC-V9 training attempts lost to the old ledger upsert-key collision.

Run only while the published coordinator is stopped at an empty-child stage
barrier. All row values are taken from immutable coordinator events and the
published core manifest; examples remain the event-reported value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ART = Path('/home_new/xzf_natural_movielens_reassessment')
LEDGER = Path('/home/xzf/Code/phase0-3-natural-git/research_plans/2026-10-20_natural_movielens_reassessment/RUN_LEDGER.csv')
EVENTS = ART / 'runner/core_execution_events_adopt.jsonl'
MANIFEST = ART / 'core_manifest.json'
FROZEN_READOUT = ART / 'V9_trajectory_readout/frozen_readout.json'
RUNNER = ART / 'runner'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def require_stopped_empty_coordinator(pid: int) -> None:
    status = Path(f'/proc/{pid}/status')
    children = Path(f'/proc/{pid}/task/{pid}/children')
    if not status.is_file() or not children.is_file():
        raise RuntimeError(f'coordinator PID {pid} is not present')
    state = next((line.split()[1] for line in status.read_text().splitlines()
                  if line.startswith('State:')), '')
    child_pids = children.read_text().strip()
    if state not in ('T', 't'):
        raise RuntimeError(f'coordinator PID {pid} must be stopped at a barrier; state={state!r}')
    if child_pids:
        raise RuntimeError(f'coordinator PID {pid} still has child processes: {child_pids}')


def read_events() -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for line in EVENTS.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        run_id = event.get('run_id', '')
        attempt_id = event.get('attempt_id', '')
        if event.get('stage') == 'AC-V9' or run_id.startswith('AC-V9__'):
            if attempt_id:
                grouped.setdefault(attempt_id, {})[event.get('event', '')] = event
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator-pid', required=True, type=int)
    args = parser.parse_args()
    require_stopped_empty_coordinator(args.coordinator_pid)

    with LEDGER.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or 'execution_kind' not in fields or 'attempt_id' not in fields:
        raise RuntimeError('RUN_LEDGER.csv has an unexpected header')

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    configs = {item['run_id']: item for item in manifest['runs'] if item.get('stage') == 'AC-V9'}
    if len(configs) != 8:
        raise RuntimeError(f'expected 8 AC-V9 configs, got {len(configs)}')
    frozen = json.loads(FROZEN_READOUT.read_text(encoding='utf-8'))
    frozen_entries = {item['run_id']: item for item in frozen.get('runs', [])}
    if not frozen.get('all_training_complete') or set(frozen_entries) != set(configs):
        raise RuntimeError('V9 frozen readout manifest does not certify the complete eight-run training set')
    grouped = read_events()
    event_rows = []
    restored = []
    for run_id, config in sorted(configs.items()):
        attempt_id = f'{run_id}_attempt01'
        pair = grouped.get(attempt_id, {})
        start = pair.get('ATTEMPT_START')
        end = pair.get('ATTEMPT_EXIT')
        if not start:
            raise RuntimeError(f'missing start source event for {attempt_id}')
        if start.get('run_id') != run_id or (end and end.get('run_id') != run_id):
            raise RuntimeError(f'event run_id mismatch for {attempt_id}')
        if end and start.get('pid') != end.get('pid'):
            raise RuntimeError(f'event PID mismatch for {attempt_id}')
        existing = [r for r in rows if r.get('attempt_id') == attempt_id
                    and r.get('execution_kind', 'training') == 'training']
        if existing:
            raise RuntimeError(f'training row already exists for {attempt_id}; refusing to overwrite')

        run_dir = Path(config['run_dir'])
        receipt_path = run_dir / 'receipt.json'
        progress_path = run_dir / 'progress.json'
        receipt = {}
        progress = {}
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                receipt = {}
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                progress = {}
        receipt_complete = (receipt.get('status') == 'COMPLETE'
                            and int(receipt.get('epochs', -1)) == int(config.get('epochs', -2))
                            and int(receipt.get('steps', -1)) == int(config.get('main_steps', -2)))
        frozen_item = frozen_entries[run_id]
        frozen_hashes = frozen_item.get('hashes', {})
        frozen_hashes_valid = (frozen_item.get('run_dir') == str(run_dir)
                               and bool(frozen_hashes)
                               and all((run_dir / name).is_file() and sha(run_dir / name) == digest
                                       for name, digest in frozen_hashes.items()))
        if end:
            success = end.get('status') == 'COMPLETE' and end.get('receipt_valid') is True
            if success and (not receipt_complete or not frozen_hashes_valid):
                raise RuntimeError(f'receipt/frozen-hash validation failed for completed attempt {attempt_id}')
            row_status = 'COMPLETE' if success else 'FAILED'
            epochs_completed = end.get('epochs', '')
            main_steps = end.get('steps', '')
            examples = receipt.get('examples', '') if success else end.get('examples', '')
            exitcode = end.get('exitcode', '')
            completed_utc = end.get('utc', '')
            event_examples_value = end.get('examples', '')
        else:
            # SN2022's attempt01 has a start event but no exit event. Recover its
            # full-budget state only from its own receipt/progress plus the
            # frozen readout's per-artifact hashes; leave exit time unavailable.
            if not (receipt_complete and frozen_hashes_valid
                    and progress.get('status') == 'COMPLETE'
                    and int(progress.get('epochs_completed', -1)) == int(config.get('epochs', -2))
                    and int(progress.get('step_main', -1)) == int(config.get('main_steps', -2))):
                raise RuntimeError(f'missing exit event and no valid full-budget artifact evidence for {attempt_id}')
            success = True
            row_status = 'COMPLETE'
            epochs_completed = receipt.get('epochs', '')
            main_steps = receipt.get('steps', '')
            examples = receipt.get('examples', '')
            exitcode = 'UNKNOWN_MONITOR_LOST'
            completed_utc = ''
            event_examples_value = ''
        row = {key: '' for key in fields}
        row.update({
            'attempt_id': attempt_id,
            'attempt_no': '1',
            'run_id': run_id,
            'study': 'NATURAL_ML_AC_V9_20261020',
            'population': config.get('population', 'movielens_natural'),
            'seed': str(config.get('seed', '')),
            'batch': str(config.get('batch_size', '')),
            'epochs_planned': str(config.get('epochs', '')),
            'epochs_completed': str(epochs_completed),
            'main_steps': str(main_steps),
            'examples': str(examples),
            'status': row_status,
            'PID': str(start.get('pid', '')),
            'started_utc': start.get('utc', ''),
            'completed_utc': completed_utc,
            'exitcode': str(exitcode),
            'output_dir': str(run_dir),
            'attempt_artifacts_dir': str(run_dir) if success else '',
            'stdout_log': start.get('log', ''),
            'receipt': str(receipt_path) if success else '',
            'selected': str(run_dir / 'selected_state.pt') if success and (run_dir / 'selected_state.pt').is_file() else '',
            'final': str(run_dir / 'epoch050.pt') if success and (run_dir / 'epoch050.pt').is_file() else '',
            'checksum_index': str(run_dir / 'sha_index.json') if success and (run_dir / 'sha_index.json').is_file() else '',
            'gpu': str(start.get('gpu', '')),
            'failure_detail': '',
            'stage': 'AC-V9',
            'arm': config.get('arm', ''),
            'budget': config.get('budget', ''),
            'architecture_steps': str(config.get('architecture_steps', '')),
            'progress_json': str(run_dir / 'progress.json') if success and (run_dir / 'progress.json').is_file() else '',
            'run_manifest': str(run_dir / 'run_manifest.json') if success and (run_dir / 'run_manifest.json').is_file() else '',
            'execution_kind': 'training',
            'command': json.dumps(start.get('command', config.get('command', [])), ensure_ascii=False),
            'source_manifest': str(MANIFEST),
            'source_manifest_sha256': sha(MANIFEST),
            'expected_steps': str(config.get('main_steps', '')),
        })
        if success and (not end or end.get('exitcode') == 'UNKNOWN_MONITOR_LOST'):
            if end:
                row['failure_detail'] = ('Restored from coordinator adoption events and run receipt; OS exitcode unavailable; '
                                         f'adopt event examples={event_examples_value}, receipt examples={examples}.')
            else:
                row['failure_detail'] = ('Restored from start event plus COMPLETE run receipt/progress and frozen V9 artifact hashes; '
                                         'ATTEMPT_EXIT event, completion time, and OS exitcode were unavailable.')
        elif not success:
            trace = 'trace unavailable'
            log = start.get('log', '')
            if log and Path(log).is_file():
                tail = Path(log).read_text(errors='replace').splitlines()[-1:]
                if tail:
                    trace = tail[0]
            row['failure_detail'] = (f'Restored from coordinator events; zero-epoch failure; {trace}; '
                                     'later attempt02 reused the configured run directory, so no attempt01 artifacts are claimed.')
        event_rows.append(row)
        run_manifest_path = run_dir / 'run_manifest.json'
        restored.append({
            'run_id': run_id, 'attempt_id': attempt_id, 'status': row['status'],
            'pid': row['PID'], 'started_utc': row['started_utc'], 'completed_utc': row['completed_utc'],
            'exitcode': row['exitcode'], 'epochs_completed': row['epochs_completed'],
            'main_steps': row['main_steps'], 'event_examples_value': event_examples_value,
            'receipt_examples_value': receipt.get('examples', '') if success else '',
            'receipt_valid': bool(receipt_complete and frozen_hashes_valid) if success else bool(end and end.get('receipt_valid')),
            'stdout_log': row['stdout_log'],
            'receipt': str(receipt_path) if success else '',
            'receipt_sha256': sha(receipt_path) if success and receipt_path.is_file() else '',
            'progress_json': str(progress_path) if success and progress_path.is_file() else '',
            'frozen_readout_manifest': str(FROZEN_READOUT) if success else '',
            'frozen_readout_manifest_sha256': sha(FROZEN_READOUT) if success else '',
            'frozen_artifact_hashes_verified': bool(frozen_hashes_valid) if success else False,
            'run_manifest': str(run_manifest_path) if success and run_manifest_path.is_file() else '',
            'run_manifest_sha256': sha(run_manifest_path) if success and run_manifest_path.is_file() else '',
        })

    required_retries = {
        f'{run_id}_attempt02' for run_id in configs
        if run_id.endswith(('SN__2024', 'SB__2023'))
    }
    present_training_ids = {r.get('attempt_id') for r in rows
                            if r.get('stage') == 'AC-V9' and r.get('execution_kind', 'training') == 'training'}
    if not required_retries.issubset(present_training_ids):
        raise RuntimeError(f'expected attempt02 rows were not present before repair: {sorted(required_retries - present_training_ids)}')

    stamp = utc_stamp()
    backup = LEDGER.with_name(f'RUN_LEDGER.pre_v9_ledger_repair_{stamp}.csv')
    shutil.copy2(LEDGER, backup)
    before_sha = sha(LEDGER)
    if sha(backup) != before_sha:
        raise RuntimeError('ledger backup hash does not match the pre-repair file')

    updated = rows + event_rows
    fd, tmp_name = tempfile.mkstemp(prefix='RUN_LEDGER.repair.', suffix='.csv.tmp', dir=LEDGER.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, '') for key in fields} for row in updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, LEDGER)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    final_rows = list(csv.DictReader(LEDGER.open(newline='', encoding='utf-8')))
    keys = [(r.get('execution_kind', 'training'), r.get('attempt_id')) for r in final_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('duplicate (execution_kind, attempt_id) remains after repair')
    v9_training = [r for r in final_rows if r.get('stage') == 'AC-V9' and r.get('execution_kind', 'training') == 'training']
    v9_diagnostics = [r for r in final_rows if r.get('stage') == 'V9_TRAJECTORY_READOUT' and r.get('execution_kind') == 'diagnostic']
    if len(v9_training) != 10 or len(v9_diagnostics) != 8:
        raise RuntimeError(f'bad post-repair V9 counts: training={len(v9_training)} diagnostics={len(v9_diagnostics)}')

    metadata = {
        'repair': 'AC-V9 attempt01 ledger row restoration after upsert collision',
        'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'coordinator_pid_stopped_empty_barrier': args.coordinator_pid,
        'ledger': str(LEDGER), 'backup': str(backup), 'before_sha256': before_sha,
        'after_sha256': sha(LEDGER), 'event_source': str(EVENTS), 'event_source_sha256': sha(EVENTS),
        'manifest_source': str(MANIFEST), 'manifest_source_sha256': sha(MANIFEST),
        'frozen_readout_source': str(FROZEN_READOUT), 'frozen_readout_source_sha256': sha(FROZEN_READOUT),
        'recovery_rows': restored,
        'preserved_attempt02_training_ids': sorted(required_retries),
        'preserved_v9_readout_diagnostic_rows': 8,
        'post_repair_v9_training_row_count_including_retries': len(v9_training),
    }
    metadata_path = RUNNER / f'v9_ledger_attempt_recovery_{stamp}.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'REPAIRED', 'rows_added': len(event_rows),
                      'v9_training_rows': len(v9_training), 'v9_readout_rows': len(v9_diagnostics),
                      'ledger_before_sha256': before_sha, 'ledger_after_sha256': metadata['after_sha256'],
                      'backup': str(backup), 'recovery_record': str(metadata_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
