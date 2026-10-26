"""Mechanical launch with content-derived provenance for primary-approved pairs."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/home/xzf/Code/phase0-3-20261016-v3-git')
OUT = ROOT / 'research_plans/2026-10-16_v3/execution_20261016'
PYTHON = '/home/xzf/miniconda3/envs/phase0-3-env/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    pair = int(sys.argv[1])
    assert pair in range(2, 6)  # pair1 already has its own actual launched queue.
    stem = f'V5V8_seed2025_continuation250_pair{pair}'
    manifest = OUT / (stem + '_manifest.json')
    review = OUT / (stem + '_activation_review.json')
    predecessors = OUT / (stem + '_predecessors.json')
    record = OUT / (stem + '_queue_launch.json')
    status = manifest.with_suffix('.status.json')
    assert not record.exists() and not status.exists()
    approval = load(review)
    assert approval['status'] == 'PASS' and approval['actual_training_authorized'] is True
    assert approval['manifest_sha'] == sha(manifest)
    tasks = load(manifest)['tasks']
    assert len(tasks) == 2
    trainer = ROOT / tasks[0]['command'][1]
    assert all(ROOT / t['command'][1] == trainer for t in tasks)
    pin = load(OUT / 'V5V8_seed2025_continuation250_copy_launch.json')['trainer_source_sha']
    actual_trainer_sha = sha(trainer)
    assert actual_trainer_sha == pin
    for task, snapshot in zip(tasks, approval['records']):
        command = task['command']
        target = Path(command[command.index('--run-dir') + 1])
        assert str(target) == snapshot['target']
        assert command[command.index('--epochs') + 1] == '250' and '--resume' in command
        assert sha(target / 'receipt.json') == snapshot['pretraining_receipt_sha']
        assert sha(target / 'continuation_source_to250.json') == snapshot['provenance_sha']
    prior = load(predecessors)
    checkpoint = load(OUT / 'orchestrator_checkpoint.json')
    current = checkpoint['active_gpu_status_files']
    assert current[:len(prior)] == prior and current[len(prior)] == status.name
    assert all((OUT / name).exists() for name in prior)
    queue = ROOT / 'experiments/v3_queue_20261016.py'
    command = [PYTHON, str(queue), '--manifest', str(manifest)]
    for name in prior:
        command.extend(['--preceding-status', str(OUT / name)])
    log = OUT / 'logs' / (stem + '_queue.log')
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    result = dict(pid=process.pid, command=command, command_shlex=shlex.join(command), cwd=str(ROOT),
        code_commit_at_queue_launch=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        manifest=str(manifest), manifest_sha256=sha(manifest), activation_review=str(review),
        activation_review_sha256=sha(review), predecessors_file=str(predecessors),
        predecessors_sha256=sha(predecessors), predecessor_count=len(prior),
        queue_source=str(queue), queue_source_sha256=sha(queue), training_source=str(trainer),
        training_source_sha256=actual_trainer_sha, launcher_source_sha256=sha(Path(__file__)),
        started_unix=time.time(), stdout_log=str(log), status=str(status),
        start_new_session=True, returncode=None, provenance='All hashes computed from physical bytes inside launcher; no supplied hash strings.')
    record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['pid', 'started_unix', 'code_commit_at_queue_launch',
        'manifest_sha256', 'activation_review_sha256', 'predecessor_count', 'queue_source_sha256',
        'training_source_sha256', 'launcher_source_sha256', 'status']}))


if __name__ == '__main__':
    main()
