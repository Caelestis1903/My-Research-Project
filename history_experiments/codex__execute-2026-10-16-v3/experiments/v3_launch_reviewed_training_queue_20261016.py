"""Fixed byte-derived launcher for primary-verified continuation training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_plans/2026-10-16_v3/execution_20261016'
    stem = sys.argv[1]
    assert Path(stem).name == stem
    manifest = out / (stem + '_manifest.json')
    review = out / (stem + '_activation_review.json')
    predecessor_file = out / (stem + '_predecessors.json')
    status = manifest.with_suffix('.status.json')
    launch = out / (stem + '_queue_launch.json')
    assert not status.exists() and not launch.exists()
    approval = load(review)
    assert approval['status'] == 'PASS' and approval['actual_training_authorized'] is True
    assert approval.get('training_manifest_sha', approval.get('manifest_sha')) == sha(manifest)
    spec = load(manifest)
    assert len(spec['tasks']) == len(approval['records'])
    sources = {}
    for task, record in zip(spec['tasks'], approval['records']):
        command = task['command']
        source = root / command[1]
        digest = sha(source)
        expected = {'v3_train_20261016.py': 'af6ca583e8224063472c635a5b2d3c53809dbc99864f82319418d687ed006c7b',
                    'v3_state_train_20261016.py': '1b30c54e430d429ed1bdab5c87a3b451994916f1760101e9142cb39364e2470f'}
        assert digest == expected[source.name]
        sources[str(source)] = digest
        target = Path(command[command.index('--run-dir') + 1])
        assert str(target) == record['target'] and '--resume' in command
        receipt = target / 'receipt.json'
        assert sha(receipt) == record.get('source_receipt_sha', record.get('pretraining_receipt_sha'))
        assert load(receipt)['epochs'] == record.get('actual_prefix_epochs', record.get('current_target_epochs_before_training'))
        assert int(command[command.index('--epochs') + 1]) > load(receipt)['epochs']
    predecessors = load(predecessor_file)
    registered = load(out / 'orchestrator_checkpoint.json')['active_gpu_status_files']
    assert registered[:len(predecessors)] == predecessors and registered[len(predecessors)] == status.name
    assert all((out / name).exists() for name in predecessors)
    queue = root / 'experiments/v3_queue_20261016.py'
    command = [sys.executable, str(queue), '--manifest', str(manifest)]
    for name in predecessors:
        command.extend(['--preceding-status', str(out / name)])
    log = out / 'logs' / (stem + '_queue.log')
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    result = dict(pid=process.pid, command=command, cwd=str(root), started_unix=time.time(),
        code_commit_at_queue_launch=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        manifest=str(manifest), manifest_sha256=sha(manifest), activation_review=str(review),
        activation_review_sha256=sha(review), predecessors_file=str(predecessor_file),
        predecessors_sha256=sha(predecessor_file), predecessor_count=len(predecessors),
        queue_source_sha256=sha(queue), launcher_source_sha256=sha(__file__),
        training_source_sha256=sources, stdout_log=str(log), status=str(status),
        start_new_session=True, returncode=None)
    with launch.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'command'}))


if __name__ == '__main__':
    main()
