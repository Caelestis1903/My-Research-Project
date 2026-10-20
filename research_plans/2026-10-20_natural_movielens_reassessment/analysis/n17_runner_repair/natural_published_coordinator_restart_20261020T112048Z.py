#!/usr/bin/env python3
"""Mechanical scheduler for the reviewed post-core MovieLens queues.

This runner waits for the already-running V2/core coordinators, then executes
only published, reviewed training and raw diagnostic commands. It never reads
test outcomes or makes scientific decisions. N17's frozen manifest actions are
invoked only at their preregistered queue barriers.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path('/home/xzf/Code/phase0-3-natural-git')
ART = Path('/home_new/xzf_natural_movielens_reassessment')
RUNNER = ART / 'runner'
PLAN = RUNNER / 'published_queue_plan.json'
N17_PREPARE_LOCK_WRAPPER = RUNNER / 'n17_prepare_lock_runner.py'
N17_PREPARE_LOCK_PATH = ART / 'N17/history/splits/movielens_seed2022.prepare.lock'
N17_PREPARE_LOCK_MODULES = {
    'experiments.natural_ml_extensions.n17_train',
    'experiments.natural_ml_extensions.n17_diagnostics',
    'experiments.natural_ml_extensions.n17_readout',
    'experiments.natural_ml_extensions.n17_basiccheck',
}
TEST_RELEASE_ROOT = RUNNER / 'released_test_queues'
TEST_RELEASE_SCHEMA = 'natural_movielens_test_release_v1'
CORE_STATE = RUNNER / 'core_coordinator_adopt_state.json'
LEDGER = REPO / 'research_plans/2026-10-20_natural_movielens_reassessment/RUN_LEDGER.csv'
PYTHON = '/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
STATE = RUNNER / 'published_coordinator_state.json'
EVENTS = RUNNER / 'published_execution_events.jsonl'
SNAPSHOTS = RUNNER / 'published_gpu_snapshots.csv'
EXTRA_FIELDS = [
    'stage', 'arm', 'budget', 'architecture_steps', 'progress_json', 'run_manifest',
    'execution_kind', 'diagnostic_id', 'checkpoint', 'output_json', 'command',
    'original_command', 'prepare_lock_path', 'wrapper_sha256',
    'test_release_manifest', 'test_release_sha256', 'parent_freeze_path', 'parent_freeze_sha256',
    'config_path', 'source_manifest', 'source_manifest_sha256', 'source_hashes',
    'state_dir', 'paired_init_dir', 'expected_steps', 'expected_examples',
    'task_scope', 'task_group', 'memory_profile_group',
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def serialize_n17_prepare(command: list[str]) -> tuple[list[str], dict[str, str]]:
    """Wrap an N17 module so only its shared split preparation is locked."""
    cmd = list(command)
    try:
        module_flag = cmd.index('-m')
        module = cmd[module_flag + 1]
    except (ValueError, IndexError):
        return cmd, {}
    if module not in N17_PREPARE_LOCK_MODULES:
        return cmd, {}
    if not N17_PREPARE_LOCK_WRAPPER.is_file():
        raise FileNotFoundError(f'N17 prepare-lock wrapper missing: {N17_PREPARE_LOCK_WRAPPER}')
    wrapped = cmd[:module_flag] + [str(N17_PREPARE_LOCK_WRAPPER), module] + cmd[module_flag + 2:]
    metadata = {'prepare_lock_path': str(N17_PREPARE_LOCK_PATH),
                'wrapper_sha256': sha(N17_PREPARE_LOCK_WRAPPER)}
    return wrapped, metadata


def json_read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def event(value: dict[str, Any]) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    with EVENTS.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utc(), **value}, ensure_ascii=False) + '\n')


def ensure_ledger() -> list[str]:
    if not LEDGER.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=EXTRA_FIELDS).writeheader()
    with LEDGER.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)
    fields = old_fields[:]
    for key in EXTRA_FIELDS:
        if key not in fields:
            fields.append(key)
    if fields != old_fields:
        tmp = LEDGER.with_suffix('.csv.tmp')
        with tmp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, '') for key in fields})
        os.replace(tmp, LEDGER)
    return fields


def ledger_rows() -> list[dict[str, str]]:
    ensure_ledger()
    with LEDGER.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def ledger_rows_readonly() -> list[dict[str, str]]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, Any]]) -> None:
    fields = ensure_ledger()
    tmp = LEDGER.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    os.replace(tmp, LEDGER)


def save_row(row: dict[str, Any]) -> None:
    rows = ledger_rows()
    row_kind = row.get('execution_kind', 'training')
    for i, old in enumerate(rows):
        if (old.get('execution_kind', 'training') == row_kind
                and old.get('attempt_id') == row['attempt_id']):
            rows[i] = row
            break
    else:
        rows.append(row)
    write_rows(rows)


def attempt_id_for(execution_kind: str, run_id: str, attempt_no: int) -> str:
    """Give non-training executions explicit ID namespaces in the shared ledger."""
    prefix = {'diagnostic': 'diagnostic__', 'control': 'control__', 'test': 'test__'}.get(execution_kind, '')
    return f"{prefix}{run_id}_attempt{attempt_no:02d}"


def next_attempt_no(run_id: str, execution_kind: str) -> int:
    numbers = []
    for row in ledger_rows():
        if row.get('run_id') == run_id and row.get('execution_kind', 'training') == execution_kind:
            try:
                numbers.append(int(row.get('attempt_no') or 0))
            except ValueError:
                pass
    return max(numbers, default=0) + 1


def gpu_memory() -> dict[int, dict[str, int]]:
    try:
        raw = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=5, stderr=subprocess.STDOUT)
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4:
            continue
        try:
            result[int(parts[0])] = {'used': int(parts[1]), 'total': int(parts[2]), 'util': int(parts[3])}
        except ValueError:
            continue
    return result


def write_snapshot(mem: dict[int, dict[str, int]], active: dict[str, dict[str, Any]]) -> None:
    exists = SNAPSHOTS.exists()
    with SNAPSHOTS.open('a', newline='', encoding='utf-8') as f:
        fields = ['utc', 'gpu', 'memory_used_mib', 'memory_total_mib', 'utilization_pct',
                  'study_process_count', 'study_pids', 'ceiling_pct']
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for gpu, values in sorted(mem.items()):
            entries = [entry for entry in active.values() if entry['gpu'] == gpu]
            writer.writerow({'utc': utc(), 'gpu': gpu, 'memory_used_mib': values['used'],
                'memory_total_mib': values['total'], 'utilization_pct': values['util'],
                'study_process_count': len(entries), 'study_pids': ';'.join(str(e['pid']) for e in entries),
                'ceiling_pct': 90})


def set_stage_state(status: str, **values: Any) -> None:
    old = {}
    if STATE.exists():
        try:
            old = json_read(STATE)
        except Exception:
            old = {}
    atomic_json(STATE, {**old, 'status': status, 'utc': utc(), **values})


def verify_core_and_v2_complete() -> tuple[bool, dict[str, Any]]:
    # The core coordinator still owns this CSV until the barrier passes. Read it
    # without changing its header or serializing rows while another writer runs.
    rows = ledger_rows_readonly()
    unique_v2 = {r.get('run_id') for r in rows if r.get('study') == 'NATURAL_ML_D_V2' and r.get('status') == 'COMPLETE'}
    core_state = json_read(CORE_STATE) if CORE_STATE.exists() else {}
    core_unique = {r.get('run_id') for r in rows if r.get('stage') in ('AC-2D', 'AC-S5', 'AC-S6', 'AC-V9') and r.get('status') == 'COMPLETE' and r.get('execution_kind', 'training') == 'training'}
    ready = (core_state.get('status') == 'ALL_COMPLETE' and not core_state.get('active')
             and not core_state.get('terminal_failures') and len(core_unique) == 120 and len(unique_v2) == 16)
    return ready, {'core_state': core_state.get('status'), 'core_complete_unique': len(core_unique),
                   'v2_complete_unique': len(unique_v2), 'core_pending': core_state.get('pending'),
                   'core_failures': core_state.get('terminal_failures', [])}


def wait_for_core(poll: float) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    set_stage_state('WAITING_FOR_V2_AND_CORE', phase='core_barrier')
    while True:
        ready, counts = verify_core_and_v2_complete()
        set_stage_state('WAITING_FOR_V2_AND_CORE' if not ready else 'CORE_BARRIER_PASSED',
                        phase='core_barrier', barrier=counts)
        if ready:
            # The core controller writes its terminal state just before exit.
            time.sleep(max(10.0, poll * 2))
            ready, counts = verify_core_and_v2_complete()
            if ready:
                event({'event': 'CORE_BARRIER_PASSED', 'barrier': counts})
                return
        if counts.get('core_failures'):
            set_stage_state('CORE_TERMINAL_FAILURES', phase='core_barrier', barrier=counts)
            raise RuntimeError('core coordinator reported terminal failures; preserve all attempts and notify parent')
        time.sleep(poll)


def population_train_count(population: str) -> int:
    candidates = [ART / 'data' / f'{population}.json',
                  ART / 'transfer_population_cache' / population / 'manifest.json']
    for path in candidates:
        if not path.exists():
            continue
        manifest = json_read(path)
        payload = manifest.get('data', manifest)
        counts = payload.get('train_domain_counts')
        if counts:
            return sum(int(v) for v in counts.values())
        preprocessing = payload.get('preprocessing_metadata', {})
        if preprocessing.get('selected_train_size'):
            return int(preprocessing['selected_train_size'])
    raise FileNotFoundError(f'no population train counts for {population}')


def population_train_domain_counts(population: str) -> dict[int, int]:
    candidates = [ART / 'data' / f'{population}.json',
                  ART / 'transfer_population_cache' / population / 'manifest.json']
    for path in candidates:
        if not path.exists():
            continue
        manifest = json_read(path)
        payload = manifest.get('data', manifest)
        counts = payload.get('train_domain_counts')
        if counts:
            return {int(k): int(v) for k, v in counts.items()}
    raise FileNotFoundError(f'no population per-domain train counts for {population}')


def population_main_domain_counts(population: str) -> dict[int, int]:
    """Reproduce the frozen 5% per-domain meta split from original training pools."""
    base = population_train_domain_counts(population)
    return {domain: count - max(1, int(count * .05)) for domain, count in base.items()}


def config_for(task: dict[str, Any]) -> dict[str, Any]:
    path = task.get('config_path')
    return json_read(path) if path and Path(path).is_file() else {}


def expected_budget(task: dict[str, Any]) -> dict[str, Any]:
    cfg = config_for(task)
    study = task.get('study')
    if study == 'AO04':
        return {'epochs': 50, 'steps': 1000, 'examples': 40008350, 'arch_steps': 50}
    if study == 'N17':
        epochs = int(cfg.get('epochs', task.get('epochs_planned', 100)))
        batch = int(cfg.get('batch_size', task.get('batch_size', 40960)))
        population = cfg.get('population') or task.get('population')
        if not population:
            raise KeyError(f"N17 task has no population: {task.get('run_id')}")
        exposure = int(cfg.get('epoch_examples') or population_train_count(population))
        return {'epochs': epochs, 'steps': epochs * math.ceil(exposure / batch), 'examples': epochs * exposure}
    if study in ('N19', 'CM19'):
        hcfg = task.get('historical_config') or {}
        if not hcfg:
            hcfg = cfg.get('historical_config', {})
        budget = task.get('budget')
        if budget == 'main178125':
            return {'epochs': 100, 'steps': 500, 'examples': 17812500}
        max_steps = hcfg.get('max_steps')
        if max_steps:
            return {'epochs': None, 'steps': int(max_steps), 'examples': None}
        population = task.get('population') or cfg.get('population', 'movielens_natural')
        main_counts = population_main_domain_counts(population)
        full_main = sum(main_counts.values())
        epochs = int(task.get('epochs', hcfg.get('epochs', 100)))
        batch = int(task.get('batch_size', hcfg.get('batch_size', 40960)))
        if study == 'CM19':
            fraction = float(hcfg.get('fraction', 1.0))
            target_domain = 1 if hcfg.get('dataset') == 'kuairand' else 2
            if fraction < 1.0:
                if target_domain not in main_counts:
                    raise KeyError(f'fraction target domain {target_domain} missing for {population}')
                sampled = dict(main_counts)
                sampled[target_domain] = int(sampled[target_domain] * fraction)
                ordinary_main = sum(sampled.values())
            else:
                ordinary_main = full_main
            exposure = full_main if hcfg.get('schedule') == 'matched_updates' else ordinary_main
        else:
            # N19's ordinary epoch permutes only the split main rows; its 5%
            # per-domain meta rows are excluded before batches are formed.
            exposure = full_main
        return {'epochs': epochs, 'steps': epochs * math.ceil(exposure / batch), 'examples': epochs * exposure}
    return {'epochs': int(task.get('epochs_planned', task.get('epochs', 0)) or 0),
            'steps': task.get('main_steps_planned'), 'examples': task.get('examples_planned')}


def training_paths(task: dict[str, Any]) -> dict[str, Path]:
    run = Path(task['run_dir'])
    study = task['study']
    if study in ('N19', 'CM19'):
        return {'run': run, 'state': Path(task['state_dir']), 'receipt': run / 'receipt.json',
                'audit': run / 'reassessment_receipt.json', 'progress': run / 'progress.json'}
    if study == 'AO04':
        return {'run': run, 'init': Path(task['paired_init_dir']), 'receipt': run / 'receipt.json',
                'index': run / 'checkpoint_index.json'}
    return {'run': run, 'receipt': run / 'receipt.json', 'index': run / 'all_model_index.json',
            'progress': run / 'progress.json'}


def audit_training(task: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
    paths = training_paths(task)
    try:
        receipt = json_read(paths['receipt'])
    except (OSError, ValueError):
        return False, None, 'missing_or_invalid_training_receipt'
    budget = expected_budget(task)
    status_ok = receipt.get('status') == 'COMPLETE_TRAINING_TEST_DEFERRED'
    checks = [status_ok]
    if budget.get('epochs') is not None:
        checks.append(int(receipt.get('epochs', -1)) == budget['epochs'])
    if budget.get('steps') is not None:
        checks.append(int(receipt.get('steps', -1)) == budget['steps'])
    if budget.get('examples') is not None:
        examples = receipt.get('examples', receipt.get('actual_main_examples'))
        checks.append(int(examples or -1) == budget['examples'])
    if budget.get('arch_steps') is not None:
        checks.append(int(receipt.get('arch_steps', -1)) == budget['arch_steps'])
    if task['study'] in ('N19', 'CM19'):
        try:
            audit = json_read(paths['audit'])
            checks.extend([audit.get('status') == 'COMPLETE_TRAINING_TEST_DEFERRED',
                           audit.get('global_run_id') == task['run_id'],
                           audit.get('receipt_sha256') == sha(paths['receipt']),
                           Path(paths['run'] / 'checkpoint_index.json').is_file()])
        except (OSError, ValueError):
            checks.append(False)
    elif task['study'] == 'AO04':
        checks.append(paths['index'].is_file())
        checks.append(receipt.get('test_evaluated') is False)
    else:
        checks.append(paths['index'].is_file())
        checks.append(receipt.get('test_read') is False)
    if task['study'] == 'AO04' and receipt.get('paired_initial_state_sha256'):
        pass  # paired same-seed integrity is audited after the full AO04 family.
    reason = 'valid_full_budget_receipt' if all(checks) else 'training_receipt_budget_or_artifact_audit_failed'
    return all(checks), receipt, reason


def read_progress(task: dict[str, Any], row: dict[str, Any], memo: dict[str, Any]) -> dict[str, int]:
    result = {'epochs': 0, 'steps': 0, 'examples': 0, 'arch_steps': 0}
    paths = training_paths(task)
    for path in (paths.get('progress'), paths['run'] / 'progress.json'):
        if path and path.is_file():
            try:
                data = json_read(path)
                result['epochs'] = int(data.get('epochs_completed', data.get('epoch_completed', data.get('epoch', 0))) or 0)
                result['steps'] = int(data.get('main_steps', data.get('steps', data.get('step_main', 0))) or 0)
                result['examples'] = int(data.get('examples', data.get('main_examples_seen', data.get('actual_main_examples', 0))) or 0)
                result['arch_steps'] = int(data.get('arch_steps', data.get('step_arch', 0)) or 0)
                if result['epochs'] or result['steps'] or result['examples']:
                    return result
            except (OSError, ValueError, TypeError):
                pass
    trajectory = paths['run'] / 'trajectory.csv'
    if trajectory.is_file():
        try:
            with trajectory.open(newline='', encoding='utf-8') as f:
                records = list(csv.DictReader(f))
            if records:
                item = records[-1]
                result['epochs'] = int(float(item.get('epoch', 0) or 0))
                result['steps'] = int(float(item.get('steps', 0) or 0))
                result['examples'] = int(float(item.get('examples', 0) or 0))
                result['arch_steps'] = int(float(item.get('arch_steps', 0) or 0))
                return result
        except (OSError, ValueError, TypeError, csv.Error):
            pass
    if task['study'] == 'AO04':
        checkpoint = paths['run'] / 'last_resume.pt'
        if checkpoint.is_file():
            try:
                stat = checkpoint.stat()
                key = task['run_id']
                previous = memo.get(key, {})
                if stat.st_mtime_ns != previous.get('mtime_ns'):
                    memo[key] = {'mtime_ns': stat.st_mtime_ns, 'stable_since': time.monotonic(), 'progress': previous.get('progress', result)}
                elif time.monotonic() - previous.get('stable_since', time.monotonic()) >= 10 and time.monotonic() - previous.get('last_read', 0) >= 30:
                    import torch
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    result = {'epochs': int(state.get('epoch', 0)), 'steps': int(state.get('main_steps', 0)),
                              'examples': int(state.get('examples_processed', 0)), 'arch_steps': int(state.get('arch_steps', 0))}
                    memo[key] = {'mtime_ns': stat.st_mtime_ns, 'stable_since': previous.get('stable_since', time.monotonic()),
                                 'last_read': time.monotonic(), 'progress': result}
                else:
                    result = previous.get('progress', result)
            except Exception as exc:
                memo.setdefault(task['run_id'], {})['progress_read_error'] = repr(exc)
    return result


def verify_gate(task: dict[str, Any]) -> None:
    gate_path = Path(task['gate'])
    gate = json_read(gate_path)
    if gate.get('verdict') not in ('PASS', 'PASS WITH FIXES') or gate.get('blocking_issues'):
        raise RuntimeError(f'review gate no longer passes: {gate_path}')
    expected_sha = task.get('gate_sha256')
    if expected_sha and sha(gate_path) != expected_sha:
        raise RuntimeError(f'review gate hash changed: {gate_path}')
    manifest = task.get('source_manifest')
    source_hash = task.get('source_manifest_sha256')
    if manifest and source_hash and sha(manifest) != source_hash:
        raise RuntimeError(f'published manifest hash changed: {manifest}')


def verify_test_release(task: dict[str, Any]) -> None:
    release_path = Path(task['test_release_manifest'])
    if not release_path.is_file() or sha(release_path) != task['test_release_sha256']:
        raise RuntimeError(f'test release manifest missing or changed: {release_path}')
    release = json_read(release_path)
    if release.get('status') != 'RELEASED' or release.get('test_authorized') is not True:
        raise RuntimeError(f'test release is not explicitly authorized: {release_path}')
    if (release.get('family') != task.get('test_release_family')
            or release.get('release_id') != task.get('test_release_id')):
        raise RuntimeError(f'test release identity changed: {release_path}')
    parent = Path(task['parent_freeze_path'])
    if not parent.is_file() or sha(parent) != task['parent_freeze_sha256']:
        raise RuntimeError(f'parent test freeze missing or changed: {parent}')


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a nonempty absolute path string')
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f'{label} must be absolute: {value!r}')
    return path


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def known_test_release_hashes() -> dict[str, str]:
    known: dict[str, str] = {}
    if not EVENTS.is_file():
        return known
    with EVENTS.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get('event') == 'TEST_RELEASE_DISCOVERED':
                known[str(entry['release_path'])] = str(entry['release_sha256'])
    return known


def load_test_release(path: Path, family: str) -> tuple[list[dict[str, Any]], str]:
    """Validate one immutable, parent-authorized test release and adapt its tasks."""
    if path.is_symlink() or not _inside(path, TEST_RELEASE_ROOT):
        raise ValueError(f'test release must be a regular file under {TEST_RELEASE_ROOT}: {path}')
    digest = sha(path)
    release = json_read(path)
    required_top = {'schema', 'family', 'release_id', 'status', 'test_authorized',
                    'parent_freeze_path', 'parent_freeze_sha256', 'tasks'}
    optional_top = {'study', 'scope'}
    if set(release) - required_top - optional_top or not required_top.issubset(release):
        raise ValueError(f'test release top-level schema mismatch: {path}')
    if (release['schema'] != TEST_RELEASE_SCHEMA or release['family'] != family
            or release['status'] != 'RELEASED' or release['test_authorized'] is not True):
        raise ValueError(f'test release is not explicitly released for {family}: {path}')
    release_id = release['release_id']
    if (not isinstance(release_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', release_id)
            or release_id in {'.', '..'} or path.stem != release_id):
        raise ValueError(f'test release_id must match its safe filename stem: {path}')
    parent_path = _absolute_path(release['parent_freeze_path'], 'parent_freeze_path')
    parent_hash = release['parent_freeze_sha256']
    if (not isinstance(parent_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', parent_hash)
            or not parent_path.is_file() or sha(parent_path) != parent_hash):
        raise ValueError(f'test parent freeze path/hash mismatch: {parent_path}')
    tasks = release['tasks']
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f'test release must contain a nonempty tasks list: {path}')
    if family == 'transfer':
        if not all(isinstance(release.get(k), str) and release[k] for k in ('study', 'scope')):
            raise ValueError('transfer test releases must bind one study and one scope')
    elif 'study' in release or 'scope' in release:
        raise ValueError(f'{family} release cannot declare transfer study/scope fields')

    allowed_task = {'run_id', 'study', 'scope', 'config_path', 'run_dir', 'command', 'receipt',
                    'receipt_checks', 'required_files', 'artifacts', 'profile_group', 'artifact_sha256'}
    seen: set[str] = set()
    adapted: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict) or set(item) - allowed_task:
            raise ValueError(f'test task has unsupported or malformed fields: {path}')
        required_task = {'run_id', 'study', 'scope', 'config_path', 'run_dir', 'command',
                         'receipt', 'receipt_checks', 'required_files', 'artifacts'}
        if not required_task.issubset(item):
            raise ValueError(f'test task is missing required fields: {path}')
        run_id = item['run_id']
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise ValueError(f'test run_id must be nonempty and unique in release: {path}')
        seen.add(run_id)
        if (not isinstance(item['study'], str) or not item['study']
                or not isinstance(item['scope'], str) or not item['scope']):
            raise ValueError(f'test task study/scope is malformed: {run_id}')
        if family in ('N17', 'AO04') and item['study'] != family:
            raise ValueError(f'{family} test release contains mismatched study: {run_id}')
        if family == 'transfer' and (item['study'] != release['study'] or item['scope'] != release['scope']):
            raise ValueError(f'transfer release contains a task outside its bound scope: {run_id}')
        config = _absolute_path(item['config_path'], f'{run_id}.config_path')
        run_dir = _absolute_path(item['run_dir'], f'{run_id}.run_dir')
        receipt_path = _absolute_path(item['receipt'], f'{run_id}.receipt')
        if not config.is_file() or not run_dir.is_dir() or not _inside(receipt_path, run_dir):
            raise ValueError(f'test config/run/receipt paths are invalid for {run_id}')
        command = item['command']
        if (not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command)
                or command[0] != PYTHON or '--device' not in command):
            raise ValueError(f'test command must be an argv list using the frozen Python and --device: {run_id}')
        device_index = command.index('--device')
        if device_index + 1 >= len(command) or command[device_index + 1] != 'cuda:0':
            raise ValueError(f'test command must freeze --device cuda:0 for scheduler remapping: {run_id}')
        if command.count('--device') != 1:
            raise ValueError(f'test command must contain exactly one --device option: {run_id}')
        for flag, expected_path in (('--config', config), ('--run-dir', run_dir)):
            if flag in command:
                index = command.index(flag)
                if index + 1 >= len(command) or Path(command[index + 1]) != expected_path:
                    raise ValueError(f'{flag} disagrees with explicit task path for {run_id}')
        checks = item['receipt_checks']
        if not isinstance(checks, dict) or 'status' not in checks or not checks:
            raise ValueError(f'test receipt_checks must include an exact status value: {run_id}')
        artifacts = item['artifacts']
        required_files = item['required_files']
        if (not isinstance(artifacts, list) or not artifacts
                or not isinstance(required_files, list)):
            raise ValueError(f'test artifacts/required_files must be lists: {run_id}')
        artifact_paths = [_absolute_path(v, f'{run_id}.artifacts[]') for v in artifacts]
        if (len(set(artifact_paths)) != len(artifact_paths)
                or any(not _inside(p, run_dir) or p == receipt_path for p in artifact_paths)):
            raise ValueError(f'test artifacts must be distinct files inside run_dir and exclude receipt: {run_id}')
        required_paths = [_absolute_path(v, f'{run_id}.required_files[]') for v in required_files]
        if len(set(required_paths)) != len(required_paths):
            raise ValueError(f'duplicate required input path for {run_id}')
        artifact_hashes = item.get('artifact_sha256', {})
        if (not isinstance(artifact_hashes, dict)
                or set(artifact_hashes) - {str(p) for p in artifact_paths}
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(v)) for v in artifact_hashes.values())):
            raise ValueError(f'artifact_sha256 must map declared output paths to SHA256 digests: {run_id}')
        profile_group = item.get('profile_group', f'TEST_{family.upper()}')
        if not isinstance(profile_group, str) or not profile_group:
            raise ValueError(f'profile_group must be a nonempty string: {run_id}')
        task = {
            'run_id': run_id, 'diagnostic_id': f"test:{release_id}:{run_id}",
            'study': item['study'], 'scope': item['scope'], 'kind': 'released_test',
            'stage': f"TEST_{family.upper()}_{release_id}", 'diagnostic_type': 'released_test',
            'execution_kind': 'test', 'config_path': str(config), 'run_dir': str(run_dir),
            'output_dir': str(run_dir), 'receipt': str(receipt_path),
            'required_files': [str(p) for p in required_paths],
            'artifacts': [str(p) for p in artifact_paths], 'artifact_sha256': artifact_hashes,
            'receipt_checks': checks, 'command': command,
            'source_manifest': str(path), 'source_manifest_sha256': digest,
            'test_release_manifest': str(path), 'test_release_sha256': digest,
            'test_release_family': family, 'test_release_id': release_id,
            'parent_freeze_path': str(parent_path), 'parent_freeze_sha256': parent_hash,
            'profile_group': profile_group,
        }
        adapted.append(task)
    return adapted, digest


def released_test_tasks(boundary: str) -> list[dict[str, Any]]:
    """Scan every family at every safe boundary; never infer authorization."""
    known = known_test_release_hashes()
    all_tasks: list[dict[str, Any]] = []
    release_summary: list[dict[str, Any]] = []
    for family in ('N17', 'AO04', 'transfer'):
        directory = TEST_RELEASE_ROOT / family
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.json')):
            tasks, digest = load_test_release(path, family)
            old_hash = known.get(str(path))
            if old_hash and old_hash != digest:
                raise RuntimeError(f'immutable test release was modified after discovery: {path}')
            if not old_hash:
                event({'event': 'TEST_RELEASE_DISCOVERED', 'release_path': str(path),
                       'release_sha256': digest, 'family': family,
                       'release_id': json_read(path)['release_id'], 'task_count': len(tasks)})
                known[str(path)] = digest
            release_summary.append({'family': family, 'path': str(path), 'sha256': digest,
                                    'task_count': len(tasks)})
            all_tasks.extend(tasks)
    by_run_id: dict[str, tuple[str, str]] = {}
    rows = ledger_rows_readonly()
    for task in all_tasks:
        identity = (task['test_release_manifest'], task['test_release_sha256'])
        previous = by_run_id.get(task['run_id'])
        if previous and previous != identity:
            raise RuntimeError(f'test run_id appears in multiple immutable releases: {task["run_id"]}')
        by_run_id[task['run_id']] = identity
        for row in rows:
            if (row.get('run_id') == task['run_id'] and row.get('execution_kind') == 'test'
                    and row.get('test_release_manifest')
                    and (row.get('test_release_manifest') != task['test_release_manifest']
                         or row.get('test_release_sha256') != task['test_release_sha256'])):
                raise RuntimeError(f'test ledger identity conflicts with immutable release: {task["run_id"]}')
    event({'event': 'TEST_QUEUE_SCAN', 'boundary': boundary, 'release_count': len(release_summary),
           'task_count': len(all_tasks), 'releases': release_summary})
    return all_tasks


def run_released_test_queues(boundary: str, args: argparse.Namespace, fields: list[str],
                             profiles: dict[str, dict[str, Any]], retries: dict[str, int]) -> None:
    tasks = released_test_tasks(boundary)
    for family in ('N17', 'AO04', 'transfer'):
        family_tasks = [task for task in tasks if task['test_release_family'] == family]
        release_ids = sorted({task['test_release_id'] for task in family_tasks})
        for release_id in release_ids:
            release_tasks = [task for task in family_tasks if task['test_release_id'] == release_id]
            stage = release_tasks[0]['stage']
            if any(task['stage'] != stage for task in release_tasks):
                raise RuntimeError(f'inconsistent release task stage: {family}/{release_id}')
            run_gpu_diagnostic_stage(stage, release_tasks, args, fields, profiles, retries)
            event({'event': 'TEST_RELEASE_CLOSED', 'boundary': boundary, 'family': family,
                   'release_id': release_id, 'release_path': release_tasks[0]['test_release_manifest'],
                   'release_sha256': release_tasks[0]['test_release_sha256'],
                   'task_count': len(release_tasks)})


def make_row(task: dict[str, Any], attempt_id: str, attempt_no: int, pid: int | str,
             gpu: int | str, log: Path, fields: list[str]) -> dict[str, Any]:
    row = {key: '' for key in fields}
    budget = expected_budget(task)
    cfg = config_for(task)
    row.update({
        'attempt_id': attempt_id, 'attempt_no': attempt_no, 'run_id': task['run_id'],
        'study': f"NATURAL_ML_{task['study'].replace('-', '_')}_20261020",
        'population': task.get('population', cfg.get('population', 'movielens_natural')),
        'seed': task.get('seed', cfg.get('seed', '')), 'optimizer': task.get('optimizer', cfg.get('optimizer', '')),
        'lr': task.get('lr', cfg.get('lr', '')), 'batch': task.get('batch_size', cfg.get('batch_size', '')),
        'epochs_planned': budget.get('epochs', ''), 'epochs_completed': 0, 'main_steps': 0, 'examples': 0,
        'status': 'RUNNING', 'PID': pid, 'started_utc': utc(), 'completed_utc': '', 'exitcode': '',
        'output_dir': task.get('run_dir', ''), 'attempt_artifacts_dir': task.get('run_dir', ''),
        'stdout_log': str(log), 'receipt': str(training_paths(task)['receipt']),
        'selected': '', 'final': '', 'checksum_index': '', 'gpu': gpu, 'failure_detail': '',
        'stage': task.get('stage', task.get('study', '')), 'arm': task.get('arm', cfg.get('arm', '')),
        'budget': task.get('budget', ''), 'architecture_steps': 0, 'progress_json': str(training_paths(task).get('progress', '')),
        'run_manifest': '', 'execution_kind': 'training', 'diagnostic_id': '', 'checkpoint': '', 'output_json': '',
        'command': json.dumps(task['command'], ensure_ascii=False), 'config_path': task.get('config_path', ''),
        'source_manifest': task.get('source_manifest', ''), 'source_manifest_sha256': task.get('source_manifest_sha256', ''),
        'source_hashes': json.dumps(cfg.get('sources', cfg.get('source_hashes', {})), sort_keys=True),
        'state_dir': task.get('state_dir', ''), 'paired_init_dir': task.get('paired_init_dir', ''),
        'expected_steps': budget.get('steps', ''), 'expected_examples': budget.get('examples', ''),
        'task_scope': task.get('scope', ''), 'task_group': task.get('group', ''),
        'memory_profile_group': task.get('profile_group', ''),
    })
    return row


def archive_incomplete(task: dict[str, Any], next_no: int) -> str:
    paths = training_paths(task)
    present = {key: path for key, path in paths.items() if key in ('run', 'state', 'init') and path.exists()}
    if not present:
        return ''
    dest = RUNNER / 'failed_attempts' / f"{task['run_id']}_attempt{max(1, next_no - 1):02d}"
    n = 2
    while dest.exists():
        dest = dest.with_name(f"{task['run_id']}_attempt{max(1, next_no - 1):02d}_{n}")
        n += 1
    dest.mkdir(parents=True)
    moved = []
    for key, path in present.items():
        target = dest / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append({'kind': key, 'from': str(path), 'to': str(target)})
    atomic_json(dest / 'attempt_artifacts.json', {'run_id': task['run_id'], 'archived_utc': utc(), 'artifacts': moved})
    event({'event': 'ARCHIVED_INCOMPLETE_TRAINING_ATTEMPT', 'run_id': task['run_id'], 'archive': str(dest), 'artifacts': moved})
    rows = ledger_rows()
    old = [r for r in rows if r.get('run_id') == task['run_id'] and r.get('execution_kind', 'training') == 'training']
    if old:
        latest = max(old, key=lambda x: int(x.get('attempt_no') or 0))
        latest['attempt_artifacts_dir'] = str(dest)
        if latest.get('status') != 'COMPLETE':
            latest['failure_detail'] = (latest.get('failure_detail', '') + '; archived before fresh full-budget retry').strip('; ')
        write_rows(rows)
    return str(dest)


def build_n17_task(config: dict[str, Any], source_manifest: Path, gate_path: Path,
                   profile_group: str = 'N17') -> dict[str, Any]:
    config_path = Path(config['config_path'])
    task = dict(config)
    run_id = str(config.get('id', config.get('run_id', config_path.stem)))
    return {
        'queue': 'N17', 'study': 'N17', 'scope': config.get('group', ''), 'group': config.get('group', ''),
        'run_id': run_id, 'arm': config.get('arm', ''), 'seed': config.get('seed', ''),
        'population': config['population'], 'budget': 'epoch_examples' if config.get('epoch_examples') else 'population_epoch',
        'epochs_planned': int(config['epochs']), 'epoch_examples': config.get('epoch_examples'),
        'batch_size': int(config.get('batch_size', 40960)), 'lr': config.get('lr'), 'optimizer': config.get('optimizer', 'adam'),
        'run_dir': config['run_dir'], 'state_dir': None, 'config_path': str(config_path),
        'command': [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_train', '--config', str(config_path), '--device', 'cuda:0'],
        'source_manifest': str(source_manifest), 'source_manifest_sha256': sha(source_manifest),
        'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'gate_verdict': 'PASS',
        'profile_group': profile_group, 'status': 'QUEUED_AFTER_CORE', 'config_snapshot': task,
    }


def build_core_diagnostic_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(plan['core_manifest'])
    gate_path = ART / 'review_gate.json'
    manifest = json_read(manifest_path)
    tasks = []
    for item in manifest.get('offline_diagnostics', []):
        checkpoint = Path(item['run_dir']) / item['checkpoint']
        run_id = f"{item['run_id']}__{item['checkpoint']}"
        tasks.append({
            'run_id': run_id, 'diagnostic_id': f"{item['run_id']}:{item['checkpoint']}",
            'study': 'AC-2D', 'scope': item['run_id'], 'kind': 'mechanism', 'stage': 'CORE_OFFLINE_DIAGNOSTICS',
            'diagnostic_type': 'core_mechanism', 'checkpoint': str(checkpoint),
            'output_json': item['output'], 'output_dir': str(Path(item['output']).parent),
            'receipt': item['output'], 'required_files': [str(checkpoint)],
            'command': [PYTHON, '-m', 'experiments.natural_ml_reassessment.mechanism',
                        '--run-dir', item['run_dir'], '--checkpoint', item['checkpoint'],
                        '--output', item['output'], '--device', 'cuda:0'],
            'source_manifest': str(manifest_path), 'source_manifest_sha256': plan['core_manifest_sha256'],
            'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'profile_group': 'CORE_MECHANISM',
        })
    return tasks


def build_n17_diagnostic_tasks(manifest_path: Path, gate_path: Path) -> list[dict[str, Any]]:
    manifest = json_read(manifest_path)
    tasks = []
    for config in manifest.get('configurations', []):
        config_path = ART / 'N17' / 'configs' / f"{config['id']}.json"
        output_dir = Path(config['out'])
        tasks.append({
            'run_id': config['id'], 'diagnostic_id': config['id'], 'study': 'N17',
            'scope': config.get('checkpoint', ''), 'kind': config['kind'],
            'stage': 'N17_CHECKPOINT_DIAGNOSTICS', 'diagnostic_type': 'n17',
            'checkpoint': config['checkpoint'], 'config_path': str(config_path),
            'output_dir': str(output_dir), 'receipt': str(output_dir / 'extension_receipt.json'),
            'required_files': [config['checkpoint'], str(config_path)],
            'command': [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_diagnostics',
                        '--config', str(config_path), '--device', 'cuda:0'],
            'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
            'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'profile_group': 'N17_DIAGNOSTIC',
        })
    return tasks


def build_ao04_diagnostic_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    queue = plan['diagnostic_queues']['AO04']
    manifest_path = Path(queue['manifest']); gate_path = ART / 'AO04/review_gate.json'
    tasks = []
    for item in queue['per_run_tasks']:
        run_dir = Path(item['run_dir'])
        tasks.append({
            'run_id': item['run_id'], 'diagnostic_id': item['run_id'], 'study': 'AO04',
            'scope': 'movielens_natural', 'kind': 'local_fusion', 'stage': 'AO04_DIAGNOSTICS',
            'diagnostic_type': 'ao04', 'output_dir': str(run_dir / 'local_fusion_diagnostics'),
            'receipt': str(run_dir / 'local_fusion_diagnostics' / 'receipt.json'),
            'required_files': [str(run_dir / 'receipt.json'), str(run_dir / 'run_manifest.json')],
            'run_dir': str(run_dir),
            'command': [PYTHON, '-m', 'experiments.natural_ml_extensions.ao04_diagnostics',
                        '--run-dir', str(run_dir), '--device', 'cuda:0'],
            'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
            'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'profile_group': 'AO04_DIAGNOSTIC',
        })
    return tasks


def build_v9_trajectory_readout_tasks() -> list[dict[str, Any]]:
    manifest_path = ART / 'V9_trajectory_readout/frozen_readout.json'
    source_path = REPO / 'experiments/natural_ml_extensions/v9_readout.py'
    gate_path = ART / 'review_gate_v9.json'
    manifest = json_read(manifest_path)
    if (manifest.get('all_training_complete') is not True or manifest.get('optimizer_updates') != 0
            or manifest.get('reviewer_verdict') != 'PRIMARY_PASS_UNDER_START_MD'
            or manifest.get('independent_review') is not False
            or manifest.get('epochs') != list(range(1, 51))
            or manifest.get('windows') != [10, 16, 20, 50]
            or sha(source_path) != manifest.get('source_sha256')):
        raise RuntimeError('V9 trajectory readout frozen manifest/source audit failed')
    runs = manifest.get('runs', [])
    expected = {(arm, seed) for arm in ('SN', 'SB') for seed in (2022, 2023, 2024, 2025)}
    if len(runs) != 8 or {(r.get('arm'), r.get('seed')) for r in runs} != expected:
        raise RuntimeError(f'V9 trajectory readout run set mismatch: {len(runs)}')
    tasks = []
    for item in runs:
        run_id = item['run_id']
        output_dir = ART / 'V9_trajectory_readout' / run_id
        tasks.append({
            'run_id': run_id, 'diagnostic_id': f'V9_TRAJECTORY_READOUT:{run_id}',
            'study': 'V9_TRAJECTORY_READOUT', 'scope': 'E1-E50', 'kind': 'trajectory_readout',
            'stage': 'V9_TRAJECTORY_READOUT', 'diagnostic_type': 'v9_readout',
            'output_dir': str(output_dir), 'receipt': str(output_dir / 'receipt.json'),
            'required_files': [str(Path(item['run_dir']) / name) for name in item.get('hashes', {})],
            'source_run_dir': item['run_dir'], 'frozen_hashes': item.get('hashes', {}),
            'command': [PYTHON, '-m', 'experiments.natural_ml_extensions.v9_readout', 'run',
                        '--run-id', run_id, '--device', 'cuda:0'],
            'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
            'gate': str(gate_path), 'gate_sha256': sha(gate_path),
            'profile_group': 'V9_TRAJECTORY_READOUT',
        })
    return tasks


def transfer_diagnostic_output_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(plan['diagnostic_queues']['transfer']['manifest'])
    gate_path = ART / 'transfer_review_gate.json'
    manifest = json_read(manifest_path)
    tasks = []
    for item in manifest.get('tasks', []):
        cfg_path = Path(item['config_path'])
        cfg = json_read(cfg_path)
        rid = item['id']
        receipt = ART / cfg['study'] / cfg['scope'] / 'diagnostic_receipts' / f"{cfg['historical_run_id']}__{item['kind']}.json"
        tasks.append({
            'run_id': rid, 'diagnostic_id': rid, 'study': cfg['study'], 'scope': cfg['scope'],
            'kind': item['kind'], 'stage': 'TRANSFER_DIAGNOSTICS', 'diagnostic_type': 'transfer',
            'config_path': str(cfg_path), 'receipt': str(receipt),
            'output_dir': str(receipt.parent), 'command': list(item['command']),
            'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
            'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'profile_group': 'TRANSFER_CPU',
        })
    return tasks


def n17_training_diagnostics_complete() -> bool:
    manifest_path = ART / 'N17/diagnostic_manifest.json'
    if not manifest_path.is_file():
        return False
    for config in json_read(manifest_path).get('configurations', []):
        receipt = Path(config['checkpoint']).parent / 'receipt.json'
        try:
            if json_read(receipt).get('status') != 'COMPLETE_TRAINING_TEST_DEFERRED':
                return False
        except (OSError, ValueError):
            return False
    return True


def run_n17_diagnostic_queue(args: argparse.Namespace, fields: list[str],
                             profiles: dict[str, dict[str, Any]], retries: dict[str, int]) -> dict[str, int]:
    n17_diag_manifest = ART / 'N17/diagnostic_manifest.json'
    if not n17_diag_manifest.is_file():
        run_control('N17_diagnostic_manifest', [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_diagnostics', '--manifest'],
                    [n17_diag_manifest], fields)
    if not n17_training_diagnostics_complete():
        raise RuntimeError('N17 diagnostic manifest has checkpoints without completed full-budget training receipts')
    n17_gate = ART / 'N17/review_gate.json'
    n17_tasks = build_n17_diagnostic_tasks(n17_diag_manifest, n17_gate)
    if len(n17_tasks) != 48:
        raise RuntimeError(f'expected 48 base N17 diagnostics, got {len(n17_tasks)}')
    run_gpu_diagnostic_stage('N17_CHECKPOINT_DIAGNOSTICS', n17_tasks, args, fields, profiles, retries)

    precision_decision = ART / 'N17/precision_decision.json'
    extra_precision_manifest = ART / 'N17/extra_precision_manifest.json'
    run_control('N17_precision_gate', [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_diagnostics', '--precision-gate'],
                [precision_decision, extra_precision_manifest], fields)
    extra = json_read(extra_precision_manifest).get('configurations', [])
    if extra:
        extra_tasks = build_n17_diagnostic_tasks(extra_precision_manifest, n17_gate)
        run_gpu_diagnostic_stage('N17_EXTRA_PRECISION_DIAGNOSTICS', extra_tasks, args, fields, profiles, retries)
    event({'event': 'N17_DIAGNOSTICS_CLOSED', 'base_diagnostics': len(n17_tasks),
           'extra_precision_diagnostics': len(extra), 'precision_decision': str(precision_decision),
           'test_status': 'READY_FOR_PARENT_CLOSURE'})
    run_released_test_queues('N17_DIAGNOSTICS_CLOSED', args, fields, profiles, retries)
    return {'base': len(n17_tasks), 'extra_precision': len(extra)}


def run_ao04_diagnostic_queue(plan: dict[str, Any], args: argparse.Namespace, fields: list[str],
                              profiles: dict[str, dict[str, Any]], retries: dict[str, int]) -> int:
    ao04_tasks = build_ao04_diagnostic_tasks(plan)
    if len(ao04_tasks) != 12:
        raise RuntimeError(f'expected 12 AO04 diagnostics, got {len(ao04_tasks)}')
    for task in ao04_tasks:
        training_task = next((t for t in plan['published_queues']['AO04']['tasks'] if t['run_id'] == task['run_id']), None)
        if training_task is None or not audit_training(training_task)[0]:
            raise RuntimeError(f"AO04 diagnostic not eligible before valid training receipt: {task['run_id']}")
    run_gpu_diagnostic_stage('AO04_DIAGNOSTICS', ao04_tasks, args, fields, profiles, retries)
    event({'event': 'AO04_FAMILY_CLOSED', 'training_configs': 12, 'diagnostics': len(ao04_tasks),
           'paired_init_audit': str(RUNNER / 'AO04_paired_init_hash_audit.json'),
           'test_status': 'READY_FOR_PARENT_CLOSURE'})
    run_released_test_queues('AO04_DIAGNOSTICS_CLOSED', args, fields, profiles, retries)
    return len(ao04_tasks)


def run_core_diagnostic_queue(plan: dict[str, Any], args: argparse.Namespace, fields: list[str],
                              profiles: dict[str, dict[str, Any]], retries: dict[str, int]) -> int:
    core_tasks = build_core_diagnostic_tasks(plan)
    if len(core_tasks) != 160:
        raise RuntimeError(f'expected 160 core offline diagnostics, got {len(core_tasks)}')
    run_gpu_diagnostic_stage('CORE_OFFLINE_DIAGNOSTICS', core_tasks, args, fields, profiles, retries)
    event({'event': 'CORE_OFFLINE_DIAGNOSTICS_CLOSED', 'diagnostics': len(core_tasks)})
    return len(core_tasks)


def run_transfer_diagnostics_for_study(plan: dict[str, Any], study: str, args: argparse.Namespace,
                                       fields: list[str], profiles: dict[str, dict[str, Any]],
                                       retries: dict[str, int]) -> dict[str, int]:
    tasks = transfer_diagnostic_output_tasks(plan)
    if len(tasks) != 143:
        raise RuntimeError(f'expected 143 transfer diagnostics, got {len(tasks)}')
    selected = [task for task in tasks if task['study'] == study]
    expected = 8 if study == 'N19' else 135 if study == 'CM19' else None
    if expected is None or len(selected) != expected:
        raise RuntimeError(f'{study} transfer diagnostic count mismatch: {len(selected)} != {expected}')
    completed_by_scope = {}
    scopes = sorted({task['scope'] for task in selected})
    for scope in scopes:
        scope_tasks = [task for task in selected if task['scope'] == scope]
        run_cpu_diagnostic_stage(f'TRANSFER_DIAGNOSTICS_{study}_{scope}', scope_tasks, args, fields, retries)
        completed_by_scope[scope] = len(scope_tasks)
        event({'event': 'TRANSFER_SCOPE_DIAGNOSTICS_CLOSED', 'study': study, 'scope': scope,
               'diagnostics': len(scope_tasks), 'test_status': 'DEFERRED_PENDING_PARENT_CLOSURE'})
        run_released_test_queues(f'TRANSFER_SCOPE_CLOSED:{study}:{scope}', args, fields, profiles, retries)
    return completed_by_scope


def update_state_for_stage(stage: str, active: dict[str, dict[str, Any]], pending: deque,
                           profiles: dict[str, dict[str, Any]], terminal: list[str], status: str = 'RUNNING') -> None:
    atomic_json(STATE, {
        'status': status, 'phase': stage, 'utc': utc(), 'pending': len(pending),
        'active': [{'run_id': e['task']['run_id'], 'attempt_id': aid, 'pid': e['pid'], 'gpu': e['gpu'],
                    'profile_group': e['task'].get('profile_group'), 'progress': e.get('progress', {})}
                   for aid, e in active.items()],
        'profiles': profiles, 'terminal_failures': terminal,
    })


def run_gpu_stage(stage: str, tasks: list[dict[str, Any]], args: argparse.Namespace,
                  fields: list[str], profiles: dict[str, dict[str, Any]], retries: dict[str, int]) -> None:
    pending = deque(tasks)
    active: dict[str, dict[str, Any]] = {}
    terminal: list[str] = []
    progress_memo: dict[str, Any] = {}
    baseline: dict[tuple[str, int], int] = {}
    peak: dict[tuple[str, int], int] = {}
    while pending or active:
        mem = gpu_memory()
        if not mem:
            update_state_for_stage(stage, active, pending, profiles, terminal, 'WAITING_FOR_GPU_SNAPSHOT')
            time.sleep(args.poll_seconds)
            continue
        write_snapshot(mem, active)
        # Audit every live attempt and capture mechanical progress only.
        for aid, entry in list(active.items()):
            task = entry['task']
            pr = read_progress(task, entry['row'], progress_memo)
            entry['progress'] = pr
            entry['row'].update({'epochs_completed': pr['epochs'], 'main_steps': pr['steps'],
                                 'examples': pr['examples'], 'architecture_steps': pr['arch_steps']})
            save_row(entry['row'])
            key = (str(task.get('profile_group', stage)), entry['gpu'])
            if key not in [(k, int(g)) for k, p in profiles.items() for g in p]:
                peak[key] = max(peak.get(key, baseline.get(key, 0)), mem.get(entry['gpu'], {}).get('used', 0))
                if pr['epochs'] >= 1:
                    base = baseline.get(key, mem[entry['gpu']]['used'])
                    delta = max(256, peak[key] - base)
                    total = mem[entry['gpu']]['total']; ceiling = int(total * .90)
                    usable = max(0, ceiling - base - args.memory_margin_mib)
                    slots = min(args.max_slots_per_gpu, max(1, usable // math.ceil(delta * args.profile_safety_factor)))
                    profiles.setdefault(key[0], {})[str(key[1])] = {
                        'baseline_mib': base, 'peak_single_mib': peak[key], 'delta_mib': delta,
                        'safe_slots': int(slots), 'ceiling_mib': ceiling,
                        'safety_factor': args.profile_safety_factor, 'margin_mib': args.memory_margin_mib,
                        'profile_source': 'first complete epoch of this profile group on this GPU',
                    }
                    event({'event': 'GPU_STAGE_PROFILED', 'stage': stage, 'profile_group': key[0], 'gpu': key[1],
                           **profiles[key[0]][str(key[1])]})
            proc = entry['proc']
            code = proc.poll()
            if code is None:
                continue
            ok, receipt, audit = audit_training(task)
            row = entry['row']
            if receipt:
                pr['epochs'] = int(receipt.get('epochs', pr['epochs']) or 0)
                pr['steps'] = int(receipt.get('steps', pr['steps']) or 0)
                pr['examples'] = int(receipt.get('examples', receipt.get('actual_main_examples', pr['examples'])) or 0)
                pr['arch_steps'] = int(receipt.get('arch_steps', pr['arch_steps']) or 0)
            row.update({'epochs_completed': pr['epochs'], 'main_steps': pr['steps'], 'examples': pr['examples'],
                        'architecture_steps': pr['arch_steps'], 'completed_utc': utc(), 'exitcode': str(code)})
            if ok:
                row.update({'status': 'COMPLETE', 'failure_detail': '' if code == 0 else f'valid full-budget receipt; process exit={code}'})
            else:
                try:
                    tail = '\n'.join(Path(row['stdout_log']).read_text(errors='replace').splitlines()[-20:])
                except OSError:
                    tail = 'stdout log unavailable'
                row.update({'status': 'FAILED', 'failure_detail': f'{audit}; exit={code}; tail={tail[-2000:]}'})
            save_row(row)
            event({'event': 'ATTEMPT_EXIT', 'run_id': task['run_id'], 'attempt_id': aid, 'pid': entry['pid'],
                   'gpu': entry['gpu'], 'exitcode': code, 'status': row['status'], 'budget_audit': audit,
                   'epochs': pr['epochs'], 'steps': pr['steps'], 'examples': pr['examples'],
                   'architecture_steps': pr['arch_steps']})
            active.pop(aid)
            if row['status'] != 'COMPLETE':
                n = retries.get(task['run_id'], 0) + 1
                retries[task['run_id']] = n
                if n < args.retry_limit:
                    pending.appendleft(task)
                else:
                    terminal.append(task['run_id'])
        if terminal:
            update_state_for_stage(stage, active, pending, profiles, terminal, 'TERMINAL_FAILURES')
            raise RuntimeError(f'{stage}: retry limit exhausted for {terminal}')
        # Fill eligible per-card slots. Profiles are maintained separately by task type and physical GPU.
        for gpu, values in sorted(mem.items()):
            card_active = [e for e in active.values() if e['gpu'] == gpu]
            total_active_count = len(card_active)
            if total_active_count >= args.max_slots_per_gpu:
                continue
            for group in dict.fromkeys(t.get('profile_group', stage) for t in pending):
                group_active = [e for e in card_active if e['task'].get('profile_group', stage) == group]
                card_tasks = [t for t in pending if t.get('profile_group', stage) == group]
                if not card_tasks:
                    continue
                profile = profiles.get(str(group), {}).get(str(gpu))
                ceiling = int(values['total'] * .90)
                if profile is None:
                    if group_active:
                        continue
                    if values['used'] + 8192 + args.memory_margin_mib >= ceiling:
                        continue
                    chosen = card_tasks[0]
                    pending.remove(chosen)
                    baseline[(str(group), gpu)] = values['used']
                    peak[(str(group), gpu)] = values['used']
                    launch_task(chosen, gpu, active, fields, 'training', progress_memo, args)
                    total_active_count += 1
                    card_active = [e for e in active.values() if e['gpu'] == gpu]
                    continue
                slot_limit = min(args.max_slots_per_gpu, int(profile['safe_slots']))
                if total_active_count >= slot_limit:
                    continue
                d = int(profile['delta_mib']); base = int(profile['baseline_mib'])
                while total_active_count < slot_limit and card_tasks:
                    current_group_count = sum(1 for e in active.values() if e['gpu'] == gpu and e['task'].get('profile_group', stage) == group)
                    external = max(base, int(values['used']) - current_group_count * d)
                    predicted = external + math.ceil((current_group_count + 1) * d * args.profile_safety_factor) + args.memory_margin_mib
                    if int(values['used']) >= ceiling or predicted >= ceiling:
                        break
                    chosen = card_tasks.pop(0)
                    pending.remove(chosen)
                    launch_task(chosen, gpu, active, fields, 'training', progress_memo, args)
                    total_active_count += 1
                    values = gpu_memory().get(gpu, values)
                    if values['used'] >= ceiling:
                        break
        update_state_for_stage(stage, active, pending, profiles, terminal)
        if pending or active:
            time.sleep(args.poll_seconds)
    set_stage_state('STAGE_COMPLETE', phase=stage, completed=len(tasks), profiles=profiles)
    event({'event': 'STAGE_COMPLETE', 'stage': stage, 'count': len(tasks)})


def launch_task(task: dict[str, Any], gpu: int, active: dict[str, dict[str, Any]], fields: list[str],
                kind: str, progress_memo: dict[str, Any], args: argparse.Namespace) -> None:
    verify_gate(task)
    ok, receipt, _ = audit_training(task)
    rows = ledger_rows()
    if ok:
        existing = [r for r in rows if r.get('run_id') == task['run_id'] and r.get('execution_kind', 'training') == 'training']
        if not any(r.get('status') == 'COMPLETE' for r in existing):
            no = next_attempt_no(task['run_id'], 'training')
            aid = f"{task['run_id']}_attempt{no:02d}"
            row = make_row(task, aid, no, '', gpu, RUNNER / 'logs' / 'training' / f'{aid}.log', fields)
            row.update({'status': 'COMPLETE', 'completed_utc': utc(), 'exitcode': 'UNKNOWN_MONITOR_LOST',
                        'epochs_completed': receipt.get('epochs', 0), 'main_steps': receipt.get('steps', 0),
                        'examples': receipt.get('examples', receipt.get('actual_main_examples', 0)),
                        'failure_detail': 'valid full-budget receipt audited before scheduler launch; OS exit unavailable'})
            save_row(row)
        return
    no = next_attempt_no(task['run_id'], 'training')
    archive = archive_incomplete(task, no)
    paths = training_paths(task)
    # The frozen commands own output directory creation. Never pre-create a scientific run directory.
    if paths['run'].exists() or paths.get('state', Path('/nonexistent')).exists():
        raise FileExistsError(f'run/state directory remains after attempt archive: {task["run_id"]}')
    logdir = RUNNER / 'logs' / 'training'; logdir.mkdir(parents=True, exist_ok=True)
    aid = f"{task['run_id']}_attempt{no:02d}"
    log = logdir / f'{aid}.log'
    cmd = list(task['command'])
    if '--device' in cmd:
        cmd[cmd.index('--device') + 1] = 'cuda:0'
    else:
        cmd.extend(['--device', 'cuda:0'])
    producer_cmd = list(cmd)
    cmd, prepare_lock = serialize_n17_prepare(cmd)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with log.open('w', encoding='utf-8') as out:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    row = make_row(task, aid, no, proc.pid, gpu, log, fields)
    row.update({'command': json.dumps(cmd, ensure_ascii=False),
                'original_command': json.dumps(producer_cmd, ensure_ascii=False),
                'prepare_lock_path': prepare_lock.get('prepare_lock_path', ''),
                'wrapper_sha256': prepare_lock.get('wrapper_sha256', '')})
    if archive:
        row['failure_detail'] = f'prior incomplete artifacts archived at {archive}'
    save_row(row)
    entry = {'task': task, 'gpu': gpu, 'pid': proc.pid, 'proc': proc, 'row': row,
             'progress': {'epochs': 0, 'steps': 0, 'examples': 0, 'arch_steps': 0},
             'adopted': False, 'started_monotonic': time.monotonic()}
    active[aid] = entry
    event({'event': 'ATTEMPT_START', 'run_id': task['run_id'], 'attempt_id': aid, 'pid': proc.pid,
           'gpu': gpu, 'profile_group': task.get('profile_group'), 'command': cmd, 'log': str(log),
           'original_command': producer_cmd, **prepare_lock,
           'source_manifest': task.get('source_manifest'), 'source_manifest_sha256': task.get('source_manifest_sha256'),
           'expected_budget': expected_budget(task)})
    print(json.dumps({'event': 'START', 'run_id': task['run_id'], 'attempt_id': aid, 'pid': proc.pid, 'gpu': gpu}), flush=True)


def run_control(action_id: str, command: list[str], outputs: list[Path], fields: list[str]) -> None:
    present = [path for path in outputs if path.exists()]
    if present and len(present) == len(outputs):
        try:
            for path in outputs:
                if path.suffix == '.json':
                    json_read(path)
        except (OSError, ValueError):
            pass
        else:
            existing = [r for r in ledger_rows() if r.get('run_id') == action_id
                        and r.get('execution_kind') == 'control' and r.get('status') == 'COMPLETE']
            if not existing:
                no = next_attempt_no(action_id, 'control')
                aid = attempt_id_for('control', action_id, no)
                row = {k: '' for k in fields}
                row.update({'attempt_id': aid, 'attempt_no': no, 'run_id': action_id,
                            'study': 'NATURAL_ML_N17_20261020', 'status': 'COMPLETE',
                            'started_utc': '', 'completed_utc': utc(), 'exitcode': 'UNKNOWN_MONITOR_LOST',
                            'execution_kind': 'control', 'receipt': ';'.join(str(p) for p in outputs),
                            'command': json.dumps(command),
                            'failure_detail': 'control outputs already present and valid; OS exit unavailable'})
                save_row(row)
            event({'event': 'CONTROL_ACTION_REUSED', 'run_id': action_id,
                   'outputs': [str(p) for p in outputs]})
            return
    if present:
        dest = RUNNER / 'failed_attempts' / 'controls' / f'{action_id}_partial_{int(time.time())}'
        dest.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(present):
            shutil.move(str(path), str(dest / f'{index:02d}_{path.name}'))
        atomic_json(dest / 'attempt_artifacts.json', {'run_id': action_id, 'archived_utc': utc(),
                    'artifacts': [str(path) for path in present]})
        event({'event': 'ARCHIVED_PARTIAL_CONTROL_OUTPUTS', 'run_id': action_id, 'archive': str(dest)})
    logdir = RUNNER / 'logs' / 'control'; logdir.mkdir(parents=True, exist_ok=True)
    no = next_attempt_no(action_id, 'control')
    aid = attempt_id_for('control', action_id, no)
    log = logdir / f'{aid}.log'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = utc()
    with log.open('w', encoding='utf-8') as out:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    success = result.returncode == 0 and all(path.exists() for path in outputs)
    row = {k: '' for k in fields}
    row.update({'attempt_id': aid, 'attempt_no': no, 'run_id': action_id, 'study': 'NATURAL_ML_N17_20261020',
                'status': 'COMPLETE' if success else 'FAILED', 'started_utc': start, 'completed_utc': utc(),
                'exitcode': str(result.returncode), 'stdout_log': str(log), 'execution_kind': 'control',
                'receipt': ';'.join(str(p) for p in outputs), 'command': json.dumps(command),
                'failure_detail': '' if success else 'control action exit/output audit failed'})
    save_row(row)
    event({'event': 'CONTROL_ACTION', 'run_id': action_id, 'attempt_id': aid, 'command': command,
           'exitcode': result.returncode, 'status': row['status'], 'outputs': [str(p) for p in outputs]})
    if not success:
        raise RuntimeError(f'control action failed: {action_id}; inspect {log}')


def diagnostic_artifacts(task: dict[str, Any]) -> list[Path]:
    """Return only this task's raw output paths, never a shared study tree."""
    kind = task['diagnostic_type']
    if kind == 'core_mechanism':
        return [Path(task['output_json'])]
    if kind == 'n17':
        return [Path(task['output_dir'])]
    if kind == 'ao04':
        return [Path(task['output_dir'])]
    if kind == 'v9_readout':
        return [Path(task['output_dir'])]
    if kind == 'released_test':
        return [Path(task['receipt']), *(Path(path) for path in task['artifacts'])]
    if kind == 'transfer':
        cfg = json_read(task['config_path'])
        study, scope, diagnostic = cfg['study'], cfg['scope'], task['kind']
        base = ART / study / scope
        run_id = cfg['historical_run_id']
        paths = [base / 'diagnostic_receipts' / f'{run_id}__{diagnostic}.json']
        if study == 'N19':
            files = {
                'rank': ['rank_finite_removal.csv', 'rank_finite_removal_scope.json'],
                'routing': ['router_weights.csv', 'router_subspace_margins.csv', 'routing_scope.json'],
                'credit': ['credit_decisions.csv', 'credit_seed_summaries.csv',
                           'credit_decision_summary.csv', 'credit_decision_scope.json'],
                'dynamics': ['cell_trajectories.csv', 'training_trajectories.csv',
                             'mean_cell_trajectories.csv', 'mean_training_trajectories.csv',
                             'stage_changes.csv', 'rank_allocations.csv', 'reliability_updates.csv', 'scope.json'],
            }
            paths.extend(base / 'dynamics' / name for name in files[diagnostic])
        else:
            run_dir = Path(cfg['run_dir'])
            if diagnostic == 'local':
                paths.extend(run_dir / name for name in ('offline_local_counterfactual.csv',
                                                          'offline_moment_statistics.csv',
                                                          'local_mechanism_receipt.json'))
            elif diagnostic == 'cost':
                paths.append(run_dir / 'cost_profile.json')
        return paths
    raise ValueError(f'unknown diagnostic type {kind}')


def audit_diagnostic(task: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
    path = Path(task['receipt'])
    try:
        receipt = json_read(path)
    except (OSError, ValueError):
        return False, None, 'missing_or_invalid_diagnostic_receipt'
    if not isinstance(receipt, dict):
        return False, None, 'diagnostic_receipt_not_an_object'
    kind = task['diagnostic_type']
    if kind == 'core_mechanism':
        checkpoint = Path(task['checkpoint'])
        valid = (checkpoint.is_file() and receipt.get('checkpoint') == checkpoint.name
                 and receipt.get('checkpoint_sha256') == sha(checkpoint)
                 and receipt.get('batch_size') == 40960
                 and receipt.get('mc_independent_batches') == 512)
    elif kind == 'n17':
        valid = receipt.get('status') == 'COMPLETE' and receipt.get('test_read') is False
    elif kind == 'ao04':
        valid = (receipt.get('status') == 'COMPLETE_RAW_DIAGNOSTIC'
                 and receipt.get('training_updates') == 0 and receipt.get('test_read') is False)
    elif kind == 'transfer':
        cfg = json_read(task['config_path'])
        valid = (receipt.get('status') == 'COMPLETE_RAW_DIAGNOSTIC'
                 and receipt.get('kind') == task['kind']
                 and receipt.get('global_run_id') == cfg['run_id']
                 and receipt.get('training_updates') == 0 and receipt.get('test_read') is False)
    elif kind == 'v9_readout':
        expected = {f'test_epoch{epoch:03d}.npz' for epoch in range(1, 51)} | {'metrics_test_trajectory.csv'}
        artifacts = receipt.get('artifacts', {}) if receipt else {}
        output = Path(task['output_dir'])
        valid = (receipt.get('status') == 'COMPLETE' and receipt.get('run_id') == task['run_id']
                 and receipt.get('epochs_evaluated') == 50 and receipt.get('optimizer_updates') == 0
                 and receipt.get('freeze_sha256') == task.get('source_manifest_sha256')
                 and receipt.get('source_sha256') == json_read(task['source_manifest']).get('source_sha256')
                 and set(artifacts) == expected
                 and all((output / name).is_file() and sha(output / name) == digest
                         for name, digest in artifacts.items()))
    elif kind == 'released_test':
        try:
            verify_test_release(task)
            checks = task['receipt_checks']
            artifacts = [Path(p) for p in task['artifacts']]
            digest_checks = task.get('artifact_sha256', {})
            valid = (all(receipt.get(key) == value for key, value in checks.items())
                     and all(path.is_file() for path in artifacts)
                     and all(sha(Path(path)) == digest for path, digest in digest_checks.items()))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            valid = False
    else:
        return False, receipt, f'unknown_diagnostic_type:{kind}'
    if valid:
        return True, receipt, 'valid_raw_diagnostic_receipt'
    return False, receipt, 'diagnostic_receipt_audit_failed'


def archive_incomplete_diagnostic(task: dict[str, Any], next_no: int) -> str:
    present = [path for path in diagnostic_artifacts(task) if path.exists()]
    if not present:
        return ''
    execution_kind = task.get('execution_kind', 'diagnostic')
    dest = RUNNER / 'failed_attempts' / ('tests' if execution_kind == 'test' else 'diagnostics') / attempt_id_for(
        execution_kind, task['run_id'], max(1, next_no - 1))
    suffix = 2
    while dest.exists():
        dest = dest.with_name(attempt_id_for(
            execution_kind, task['run_id'], max(1, next_no - 1)) + f'_{suffix}')
        suffix += 1
    dest.mkdir(parents=True)
    moved = []
    for index, path in enumerate(present):
        target = dest / f'{index:02d}_{path.name}'
        shutil.move(str(path), str(target))
        moved.append({'from': str(path), 'to': str(target)})
    atomic_json(dest / 'attempt_artifacts.json', {'run_id': task['run_id'], 'archived_utc': utc(), 'artifacts': moved})
    rows = ledger_rows()
    old = [r for r in rows if r.get('run_id') == task['run_id']
           and r.get('execution_kind', 'diagnostic') == execution_kind]
    if old:
        latest = max(old, key=lambda x: int(x.get('attempt_no') or 0))
        latest['attempt_artifacts_dir'] = str(dest)
        if latest.get('status') != 'COMPLETE':
            latest['failure_detail'] = (latest.get('failure_detail', '') + '; partial diagnostic outputs archived').strip('; ')
        write_rows(rows)
    event({'event': 'ARCHIVED_INCOMPLETE_TEST_ATTEMPT' if execution_kind == 'test'
           else 'ARCHIVED_INCOMPLETE_DIAGNOSTIC_ATTEMPT', 'run_id': task['run_id'],
           'archive': str(dest), 'artifacts': moved})
    return str(dest)


def make_diagnostic_row(task: dict[str, Any], attempt_id: str, attempt_no: int,
                        pid: int | str, gpu: int | str, log: Path, fields: list[str]) -> dict[str, Any]:
    row = {key: '' for key in fields}
    row.update({
        'attempt_id': attempt_id, 'attempt_no': attempt_no, 'run_id': task['run_id'],
        'study': f"NATURAL_ML_{task.get('study', 'DIAGNOSTIC')}_20261020",
        'status': 'RUNNING', 'PID': pid, 'started_utc': utc(), 'completed_utc': '', 'exitcode': '',
        'output_dir': task.get('output_dir', ''), 'attempt_artifacts_dir': task.get('output_dir', ''),
        'stdout_log': str(log), 'receipt': str(task['receipt']), 'gpu': gpu,
        'failure_detail': '', 'stage': task.get('stage', ''),
        'execution_kind': task.get('execution_kind', 'diagnostic'),
        'diagnostic_id': task.get('diagnostic_id', task['run_id']), 'checkpoint': task.get('checkpoint', ''),
        'output_json': task.get('output_json', ''), 'command': json.dumps(task['command'], ensure_ascii=False),
        'config_path': task.get('config_path', ''), 'source_manifest': task.get('source_manifest', ''),
        'source_manifest_sha256': task.get('source_manifest_sha256', ''),
        'run_manifest': task.get('run_manifest', ''), 'task_scope': task.get('scope', ''),
        'task_group': task.get('kind', ''), 'memory_profile_group': task.get('profile_group', ''),
        'test_release_manifest': task.get('test_release_manifest', ''),
        'test_release_sha256': task.get('test_release_sha256', ''),
        'parent_freeze_path': task.get('parent_freeze_path', ''),
        'parent_freeze_sha256': task.get('parent_freeze_sha256', ''),
    })
    return row


def diagnostic_ready(task: dict[str, Any]) -> None:
    if task['diagnostic_type'] == 'released_test':
        verify_test_release(task)
    for key in task.get('required_files', []):
        if not Path(key).is_file():
            raise FileNotFoundError(f"required diagnostic input missing for {task['run_id']}: {key}")
    if task['diagnostic_type'] == 'transfer':
        manifest = json_read(ART / 'transfer_manifest.json')
        cfg = json_read(task['config_path'])
        scope_runs = [r for r in manifest['runs'] if r['study'] == cfg['study'] and r['scope'] == cfg['scope']]
        if not scope_runs:
            raise RuntimeError(f"empty required transfer scope: {cfg['study']} {cfg['scope']}")
        for item in scope_runs:
            receipt = json_read(Path(item['run_dir']) / 'reassessment_receipt.json')
            if receipt.get('status') != 'COMPLETE_TRAINING_TEST_DEFERRED':
                raise RuntimeError(f"transfer diagnostic scope not fully trained: {item['run_id']}")
    elif task['diagnostic_type'] == 'v9_readout':
        manifest = json_read(task['source_manifest'])
        if sha(Path(task['source_manifest'])) != task.get('source_manifest_sha256'):
            raise RuntimeError('V9 frozen readout manifest hash changed')
        frozen = next((item for item in manifest.get('runs', [])
                       if item.get('run_id') == task['run_id']), None)
        if (frozen is None or frozen.get('run_dir') != task.get('source_run_dir')
                or frozen.get('hashes') != task.get('frozen_hashes')):
            raise RuntimeError(f"V9 frozen readout source entry changed: {task['run_id']}")
        for name, digest in task.get('frozen_hashes', {}).items():
            source = Path(task['source_run_dir']) / name
            if not source.is_file() or sha(source) != digest:
                raise RuntimeError(f"V9 frozen training input hash mismatch: {task['run_id']} {name}")


def launch_diagnostic(task: dict[str, Any], gpu: int | None, active: dict[str, dict[str, Any]],
                      fields: list[str], kind: str, args: argparse.Namespace) -> None:
    execution_kind = task.get('execution_kind', kind)
    if execution_kind == 'test':
        verify_test_release(task)
    else:
        verify_gate(task)
    diagnostic_ready(task)
    ok, receipt, _ = audit_diagnostic(task)
    rows = ledger_rows()
    if ok:
        existing = [r for r in rows if r.get('run_id') == task['run_id']
                    and r.get('execution_kind', 'diagnostic') == execution_kind]
        if not any(r.get('status') == 'COMPLETE' for r in existing):
            no = next_attempt_no(task['run_id'], execution_kind)
            aid = attempt_id_for(execution_kind, task['run_id'], no)
            row = make_diagnostic_row(task, aid, no, '', '' if gpu is None else gpu,
                                     RUNNER / 'logs' / ('tests' if execution_kind == 'test' else 'diagnostics')
                                     / f'{aid}.log', fields)
            row.update({'status': 'COMPLETE', 'completed_utc': utc(), 'exitcode': 'UNKNOWN_MONITOR_LOST',
                        'failure_detail': 'valid released test receipt audited before launch; OS exit unavailable'
                        if execution_kind == 'test' else
                        'valid raw diagnostic receipt audited before launch; OS exit unavailable'})
            save_row(row)
        return
    no = next_attempt_no(task['run_id'], execution_kind)
    archive = archive_incomplete_diagnostic(task, no)
    # A completed output cannot reach here: audit_diagnostic has already excluded it.
    for path in diagnostic_artifacts(task):
        if path.exists():
            raise FileExistsError(f"diagnostic output remained after archive for {task['run_id']}: {path}")
    logdir = RUNNER / 'logs' / ('tests' if execution_kind == 'test' else 'diagnostics')
    logdir.mkdir(parents=True, exist_ok=True)
    aid = attempt_id_for(execution_kind, task['run_id'], no)
    log = logdir / f'{aid}.log'
    cmd = list(task['command'])
    env_values = {'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    if gpu is not None:
        if '--device' in cmd:
            cmd[cmd.index('--device') + 1] = 'cuda:0'
        else:
            cmd.extend(['--device', 'cuda:0'])
        env_values['CUDA_VISIBLE_DEVICES'] = str(gpu)
    else:
        env_values['OMP_NUM_THREADS'] = '1'
        env_values['MKL_NUM_THREADS'] = '1'
    producer_cmd = list(cmd)
    cmd, prepare_lock = serialize_n17_prepare(cmd)
    env = dict(os.environ, **env_values)
    with log.open('w', encoding='utf-8') as out:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    row = make_diagnostic_row(task, aid, no, proc.pid, '' if gpu is None else gpu, log, fields)
    row.update({'command': json.dumps(cmd, ensure_ascii=False),
                'original_command': json.dumps(producer_cmd, ensure_ascii=False),
                'prepare_lock_path': prepare_lock.get('prepare_lock_path', ''),
                'wrapper_sha256': prepare_lock.get('wrapper_sha256', '')})
    if archive:
        row['failure_detail'] = f'prior partial outputs archived at {archive}'
    save_row(row)
    if gpu is None:
        entry_gpu = -1
    else:
        entry_gpu = gpu
    active[aid] = {'task': task, 'gpu': entry_gpu, 'pid': proc.pid, 'proc': proc, 'row': row,
                   'adopted': False, 'started_monotonic': time.monotonic(), 'peak_mib': 0}
    event({'event': 'TEST_ATTEMPT_START' if execution_kind == 'test' else 'DIAGNOSTIC_ATTEMPT_START',
           'run_id': task['run_id'], 'attempt_id': aid,
           'pid': proc.pid, 'gpu': gpu, 'diagnostic_type': task['diagnostic_type'],
           'command': cmd, 'log': str(log), 'source_manifest': task.get('source_manifest'),
           'original_command': producer_cmd, **prepare_lock,
           'source_manifest_sha256': task.get('source_manifest_sha256'),
           'test_release_manifest': task.get('test_release_manifest'),
           'test_release_sha256': task.get('test_release_sha256'),
           'parent_freeze_path': task.get('parent_freeze_path'),
           'parent_freeze_sha256': task.get('parent_freeze_sha256')})
    print(json.dumps({'event': 'TEST_START' if execution_kind == 'test' else 'DIAGNOSTIC_START',
                      'run_id': task['run_id'], 'attempt_id': aid,
                      'pid': proc.pid, 'gpu': gpu}), flush=True)


def run_gpu_diagnostic_stage(stage: str, tasks: list[dict[str, Any]], args: argparse.Namespace,
                             fields: list[str], profiles: dict[str, dict[str, Any]],
                             retries: dict[str, int]) -> None:
    pending = deque(tasks)
    active: dict[str, dict[str, Any]] = {}
    terminal: list[str] = []
    baselines: dict[tuple[str, int], int] = {}
    peaks: dict[tuple[str, int], int] = {}
    while pending or active:
        mem = gpu_memory()
        if not mem:
            update_state_for_stage(stage, active, pending, profiles, terminal, 'WAITING_FOR_GPU_SNAPSHOT')
            time.sleep(args.poll_seconds)
            continue
        write_snapshot(mem, active)
        for aid, entry in list(active.items()):
            gpu = entry['gpu']; group = str(entry['task'].get('profile_group', stage)); key = (group, gpu)
            entry['peak_mib'] = max(entry.get('peak_mib', 0), mem.get(gpu, {}).get('used', 0))
            code = entry['proc'].poll()
            if code is None:
                continue
            task = entry['task']
            if group not in profiles or str(gpu) not in profiles[group]:
                base = baselines.get(key, mem[gpu]['used'])
                delta = max(256, entry['peak_mib'] - base)
                ceiling = int(mem[gpu]['total'] * .90)
                safe = min(args.max_slots_per_gpu, max(1, (ceiling - base - args.memory_margin_mib) //
                           math.ceil(delta * args.profile_safety_factor)))
                profiles.setdefault(group, {})[str(gpu)] = {
                    'baseline_mib': base, 'peak_single_mib': entry['peak_mib'], 'delta_mib': delta,
                    'safe_slots': int(safe), 'ceiling_mib': ceiling,
                    'safety_factor': args.profile_safety_factor, 'margin_mib': args.memory_margin_mib,
                    'profile_source': 'first completed diagnostic on this GPU',
                }
                event({'event': 'GPU_DIAGNOSTIC_PROFILED', 'stage': stage, 'profile_group': group,
                       'gpu': gpu, **profiles[group][str(gpu)]})
            ok, receipt, audit = audit_diagnostic(task)
            row = entry['row']; row.update({'completed_utc': utc(), 'exitcode': str(code)})
            if ok:
                row.update({'status': 'COMPLETE', 'failure_detail': '' if code == 0 else f'valid receipt; process exit={code}'})
            else:
                try:
                    tail = '\n'.join(Path(row['stdout_log']).read_text(errors='replace').splitlines()[-20:])
                except OSError:
                    tail = 'stdout log unavailable'
                row.update({'status': 'FAILED', 'failure_detail': f'{audit}; exit={code}; tail={tail[-2000:]}'})
            save_row(row)
            event({'event': 'DIAGNOSTIC_ATTEMPT_EXIT', 'run_id': task['run_id'], 'attempt_id': aid,
                   'pid': entry['pid'], 'gpu': gpu, 'exitcode': code, 'status': row['status'],
                   'receipt_audit': audit})
            active.pop(aid)
            if row['status'] != 'COMPLETE':
                n = retries.get(task['run_id'], 0) + 1; retries[task['run_id']] = n
                if n < args.retry_limit:
                    pending.appendleft(task)
                else:
                    terminal.append(task['run_id'])
        if terminal:
            update_state_for_stage(stage, active, pending, profiles, terminal, 'TERMINAL_FAILURES')
            raise RuntimeError(f'{stage}: retry limit exhausted for {terminal}')
        for gpu, values in sorted(mem.items()):
            card_active = [e for e in active.values() if e['gpu'] == gpu]
            active_count = len(card_active)
            if active_count >= args.max_slots_per_gpu:
                continue
            for group in dict.fromkeys(t.get('profile_group', stage) for t in pending):
                group = str(group)
                same_group = [e for e in card_active if e['task'].get('profile_group', stage) == group]
                candidates = [t for t in pending if t.get('profile_group', stage) == group]
                if not candidates:
                    continue
                profile = profiles.get(group, {}).get(str(gpu))
                ceiling = int(values['total'] * .90)
                if profile is None:
                    if same_group or values['used'] + 8192 + args.memory_margin_mib >= ceiling:
                        continue
                    task = candidates[0]; pending.remove(task)
                    baselines[(group, gpu)] = values['used']; peaks[(group, gpu)] = values['used']
                    launch_diagnostic(task, gpu, active, fields, 'diagnostic', args)
                    card_active = [e for e in active.values() if e['gpu'] == gpu]
                    active_count = len(card_active)
                    continue
                slot_limit = min(args.max_slots_per_gpu, int(profile['safe_slots']))
                if active_count >= slot_limit:
                    continue
                base = int(profile['baseline_mib']); delta = int(profile['delta_mib'])
                while active_count < slot_limit and candidates:
                    group_active_count = sum(1 for e in active.values() if e['gpu'] == gpu and
                                             e['task'].get('profile_group', stage) == group)
                    external = max(base, int(values['used']) - group_active_count * delta)
                    predicted = external + math.ceil((group_active_count + 1) * delta * args.profile_safety_factor) + args.memory_margin_mib
                    if int(values['used']) >= ceiling or predicted >= ceiling:
                        break
                    task = candidates.pop(0); pending.remove(task)
                    launch_diagnostic(task, gpu, active, fields, 'diagnostic', args)
                    active_count += 1
                    values = gpu_memory().get(gpu, values)
                    if values['used'] >= ceiling:
                        break
        update_state_for_stage(stage, active, pending, profiles, terminal)
        if pending or active:
            time.sleep(args.poll_seconds)
    set_stage_state('STAGE_COMPLETE', phase=stage, completed=len(tasks), profiles=profiles)
    event({'event': 'DIAGNOSTIC_STAGE_COMPLETE', 'stage': stage, 'count': len(tasks)})


def run_cpu_diagnostic_stage(stage: str, tasks: list[dict[str, Any]], args: argparse.Namespace,
                             fields: list[str], retries: dict[str, int]) -> None:
    pending = deque(tasks); active: dict[str, dict[str, Any]] = {}; terminal: list[str] = []
    while pending or active:
        while pending and len(active) < args.cpu_workers:
            task = pending.popleft()
            launch_diagnostic(task, None, active, fields, 'diagnostic', args)
        for aid, entry in list(active.items()):
            code = entry['proc'].poll()
            if code is None:
                continue
            task = entry['task']; ok, receipt, audit = audit_diagnostic(task)
            row = entry['row']; row.update({'completed_utc': utc(), 'exitcode': str(code)})
            if ok:
                row.update({'status': 'COMPLETE', 'failure_detail': '' if code == 0 else f'valid receipt; process exit={code}'})
            else:
                try:
                    tail = '\n'.join(Path(row['stdout_log']).read_text(errors='replace').splitlines()[-20:])
                except OSError:
                    tail = 'stdout log unavailable'
                row.update({'status': 'FAILED', 'failure_detail': f'{audit}; exit={code}; tail={tail[-2000:]}'})
            save_row(row)
            event({'event': 'DIAGNOSTIC_ATTEMPT_EXIT', 'run_id': task['run_id'], 'attempt_id': aid,
                   'pid': entry['pid'], 'gpu': None, 'exitcode': code, 'status': row['status'], 'receipt_audit': audit})
            active.pop(aid)
            if row['status'] != 'COMPLETE':
                n = retries.get(task['run_id'], 0) + 1; retries[task['run_id']] = n
                if n < args.retry_limit:
                    pending.appendleft(task)
                else:
                    terminal.append(task['run_id'])
        if terminal:
            update_state_for_stage(stage, active, pending, {}, terminal, 'TERMINAL_FAILURES')
            raise RuntimeError(f'{stage}: retry limit exhausted for {terminal}')
        update_state_for_stage(stage, active, pending, {}, terminal)
        if pending or active:
            time.sleep(args.poll_seconds)
    set_stage_state('STAGE_COMPLETE', phase=stage, completed=len(tasks))
    event({'event': 'DIAGNOSTIC_STAGE_COMPLETE', 'stage': stage, 'count': len(tasks)})


def append_dynamic_queue(plan: dict[str, Any], name: str, manifest_path: Path,
                         configs: list[dict[str, Any]], gate_path: Path) -> list[dict[str, Any]]:
    tasks = [build_n17_task(cfg, manifest_path, gate_path) for cfg in configs]
    existing = plan['published_queues'].get(name, {}).get('tasks', [])
    ids = {item['run_id'] for item in existing}
    tasks = [item for item in tasks if item['run_id'] not in ids]
    plan['published_queues'][name] = {
        'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'gate': str(gate_path), 'gate_sha256': sha(gate_path), 'gate_verdict': 'PASS',
        'count': len(existing) + len(tasks), 'tasks': existing + tasks,
    }
    plan['counts'][name] = len(existing) + len(tasks)
    atomic_json(PLAN, plan)
    event({'event': 'DYNAMIC_QUEUE_ADDED', 'queue': name, 'source_manifest': str(manifest_path),
           'source_manifest_sha256': sha(manifest_path), 'added': len(tasks), 'total': len(existing) + len(tasks)})
    return existing + tasks


def n17_stage_tasks(plan: dict[str, Any], group: str | None = None) -> list[dict[str, Any]]:
    tasks = plan['published_queues']['N17_initial_executable']['tasks']
    if group == 'pilot':
        return [t for t in tasks if t.get('group') == 'pilot']
    if group == 'nonpilot':
        return [t for t in tasks if t.get('group') != 'pilot']
    return list(tasks)


def run_training_queues(plan: dict[str, Any], args: argparse.Namespace, fields: list[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    retries: dict[str, int] = {}
    n17_gate = ART / 'N17/review_gate.json'
    # N17's 16 prespecified pilots run first to unlock the frozen winner manifest.
    pilots = n17_stage_tasks(plan, 'pilot')
    if len(pilots) != 16:
        raise RuntimeError(f'expected 16 N17 pilots, got {len(pilots)}')
    run_gpu_stage('N17_PILOT', pilots, args, fields, profiles, retries)
    freeze_file = ART / 'N17/winners_frozen.json'
    post_file = ART / 'N17/post_pilot_manifest.json'
    run_control('N17_freeze_winners', [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_manifest', 'freeze-winners'],
                [freeze_file, post_file], fields)
    post = json_read(post_file)
    dynamic = append_dynamic_queue(plan, 'N17_post_pilot', post_file, post['configurations'], n17_gate)
    # The frozen V9 readout has no optimizer updates and depends on completed
    # checkpoints. Run after N17 pilots unlock their queue and before long training.
    v9_readout_tasks = build_v9_trajectory_readout_tasks()
    if len(v9_readout_tasks) != 8:
        raise RuntimeError(f'expected 8 frozen V9 trajectory readouts, got {len(v9_readout_tasks)}')
    run_gpu_diagnostic_stage('V9_TRAJECTORY_READOUT', v9_readout_tasks, args, fields, profiles, retries)
    base_nonpilot = n17_stage_tasks(plan, 'nonpilot')
    n17_remaining = base_nonpilot + dynamic
    run_gpu_stage('N17_BASE_AND_POST_PILOT', n17_remaining, args, fields, profiles, retries)
    # The plateau gate is a frozen N17 mechanical action after all common-LR formal runs finish.
    plateau_file = ART / 'N17/plateau_frozen.json'
    run_control('N17_plateau_gate', [PYTHON, '-m', 'experiments.natural_ml_extensions.n17_manifest', 'plateau'],
                [plateau_file], fields)
    h200_manifest = ART / 'N17/H200_manifest.json'
    if json_read(plateau_file).get('trigger_H200'):
        if not h200_manifest.is_file():
            raise RuntimeError('N17 plateau gate triggered H200 but no H200 manifest was produced')
        h200 = json_read(h200_manifest)
        tasks = [build_n17_task(cfg, h200_manifest, n17_gate) for cfg in h200['configurations']]
        append_dynamic_queue(plan, 'N17_H200_conditional', h200_manifest, h200['configurations'], n17_gate)
        run_gpu_stage('N17_H200_CONDITIONAL', tasks, args, fields, profiles, retries)
    n17_diag_counts = run_n17_diagnostic_queue(args, fields, profiles, retries)
    set_stage_state('N17_FAMILY_CLOSED_TEST_READY', phase='N17_closure', profiles=profiles,
                    diagnostics=n17_diag_counts, plateau=json_read(plateau_file))
    event({'event': 'N17_FAMILY_CLOSED_TEST_READY', 'diagnostics': n17_diag_counts,
           'plateau': json_read(plateau_file), 'test_status': 'READY_FOR_PARENT_CLOSURE'})

    ao04 = plan['published_queues']['AO04']['tasks']
    if len(ao04) != 12:
        raise RuntimeError(f'expected 12 AO04 tasks, got {len(ao04)}')
    run_gpu_stage('AO04', ao04, args, fields, profiles, retries)
    # Mechanically compare paired initialization hashes for each AO04 seed across arms.
    paired = {}
    for task in ao04:
        ok, receipt, why = audit_training(task)
        if not ok or not receipt:
            raise RuntimeError(f'AO04 completion audit changed before paired-hash check: {task["run_id"]} {why}')
        seed = str(task['seed'])
        paired.setdefault(seed, {})[task['arm']] = receipt.get('paired_initial_state_sha256')
    pair_ok = all(len(arms) == 3 and len(set(arms.values())) == 1 and None not in arms.values() for arms in paired.values())
    atomic_json(RUNNER / 'AO04_paired_init_hash_audit.json', {'status': 'PASS' if pair_ok else 'FAIL',
        'raw_hashes_by_seed_arm': paired, 'utc': utc()})
    if not pair_ok:
        raise RuntimeError('AO04 paired initialization tensor hashes differ; raw hashes preserved')

    ao04_diag_count = run_ao04_diagnostic_queue(plan, args, fields, profiles, retries)
    set_stage_state('AO04_FAMILY_CLOSED_TEST_READY', phase='AO04_closure', profiles=profiles,
                    diagnostics=ao04_diag_count, paired_init_audit=str(RUNNER / 'AO04_paired_init_hash_audit.json'))
    event({'event': 'AO04_FAMILY_CLOSED_TEST_READY', 'training_configs': len(ao04),
           'diagnostics': ao04_diag_count, 'paired_init_audit': str(RUNNER / 'AO04_paired_init_hash_audit.json'),
           'test_status': 'READY_FOR_PARENT_CLOSURE'})

    run_core_diagnostic_queue(plan, args, fields, profiles, retries)
    run_released_test_queues('CORE_OFFLINE_DIAGNOSTICS_CLOSED', args, fields, profiles, retries)

    # Transfer stages retain exact published scopes, budgets, and commands. Close
    # each study's diagnostics as soon as every training config in that study is done.
    transfer = plan['published_queues']['transfer']['tasks']
    for study in ('N19', 'CM19'):
        tasks = [t for t in transfer if t['study'] == study]
        expected = 316 if study == 'N19' else 508
        if len(tasks) != expected:
            raise RuntimeError(f'{study} transfer queue count mismatch: {len(tasks)} != {expected}')
        run_gpu_stage(f'TRANSFER_{study}', tasks, args, fields, profiles, retries)
        scope_counts = run_transfer_diagnostics_for_study(plan, study, args, fields, profiles, retries)
        event({'event': 'TRANSFER_STUDY_CLOSED', 'study': study, 'training_configs': len(tasks),
               'diagnostics_by_scope': scope_counts, 'test_status': 'DEFERRED_PENDING_PARENT_CLOSURE'})
    return profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-slots-per-gpu', type=int, default=5)
    ap.add_argument('--poll-seconds', type=float, default=5)
    ap.add_argument('--retry-limit', type=int, default=3)
    ap.add_argument('--memory-margin-mib', type=int, default=512)
    ap.add_argument('--profile-safety-factor', type=float, default=1.20)
    ap.add_argument('--cpu-workers', type=int, default=4)
    ap.add_argument('--wait-core-poll-seconds', type=float, default=20)
    a = ap.parse_args()
    if not 1 <= a.max_slots_per_gpu <= 5:
        raise SystemExit('max-slots-per-gpu must be 1..5')
    RUNNER.mkdir(parents=True, exist_ok=True)
    lock_handle = (RUNNER / 'published_coordinator.lock').open('a+')
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another published coordinator holds the exclusive runner lock')
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f'pid={os.getpid()} utc={utc()}\n')
    lock_handle.flush()
    wait_for_core(a.wait_core_poll_seconds)
    fields = ensure_ledger()
    plan = json_read(PLAN)
    ao_manifest = ART / 'AO04/manifest.json'
    if len(plan['published_queues'].get('AO04', {}).get('tasks', [])) != 12:
        raise RuntimeError('AO04 tasks missing from published queue plan')
    if sha(ao_manifest) != plan['published_queues']['AO04']['manifest_sha256']:
        raise RuntimeError('AO04 source manifest changed after queue publication')
    set_stage_state('POST_CORE_QUEUE_RUNNING', phase='training')
    profiles = run_training_queues(plan, a, fields)
    set_stage_state('ALL_RELEASED_TRAINING_AND_DIAGNOSTICS_COMPLETE_TESTS_DEFERRED',
                    phase='closed', profiles=profiles)
    event({'event': 'RELEASED_QUEUES_COMPLETE', 'profiles': profiles,
           'test_status': 'DEFERRED_PENDING_PARENT_CLOSURE'})


if __name__ == '__main__':
    main()
