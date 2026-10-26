"""Mechanical native-test launch after the complete, immutable global freeze."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_plans/2026-10-16_v3/execution_20261016'
    stem = sys.argv[1]
    assert Path(stem).name == stem and '..' not in stem
    manifest = out / (stem + '_manifest.json')
    review = out / (stem + '_activation_review.json')
    predecessors_file = out / (stem + '_predecessors.json')
    status = manifest.with_suffix('.status.json')
    launch = out / (stem + '_queue_launch.json')
    assert not status.exists() and not launch.exists()
    freeze_file = out / 'test_readout_frozen.json'
    freeze = load(freeze_file)
    assert freeze['status'] == 'FROZEN' and freeze['all_condition_definitions_frozen'] is True
    assert freeze['analysis_rules_sha'] == sha(out / 'analysis_rules_frozen.json')
    assert sha(manifest) in freeze['test_native_manifest_hashes']
    inputs_file = out / freeze['test_native_parent_inputs_file']
    assert sha(inputs_file) == freeze['test_native_parent_inputs_sha']
    inputs = load(inputs_file)['records']
    allowed = {item['run']: item for item in inputs}
    assert len(allowed) == len(inputs)
    approval = load(review)
    assert approval['status'] == 'PASS' and approval['native_tests_authorized'] is True
    assert approval['manifest_sha'] == sha(manifest)
    assert approval['test_global_freeze_sha'] == sha(freeze_file)
    engine = root / 'experiments/v3_readout_20261016.py'
    queue = root / 'experiments/v3_queue_20261016.py'
    assert sha(engine) == approval['readout_sha'] == '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'
    assert sha(queue) == '5b42100b36fc70cbdd9e3e84b38ea0718996a8326aa73404ce36eef76ce92418'
    spec = load(manifest)
    assert spec['cwd'] == str(root) and spec['tasks']
    seen = set()
    for task in spec['tasks']:
        command = task['command']
        assert len(command) == 8
        directory = Path(command[5])
        assert command == [sys.executable, 'experiments/v3_readout_20261016.py',
                           '--mode', 'test', '--natural', str(directory), '--device', '{device}']
        assert str(directory) in allowed and str(directory) not in seen
        seen.add(str(directory))
        item = allowed[str(directory)]
        assert sha(directory / 'receipt.json') == item['producer_receipt_sha']
        assert sha(directory / 'config.json') == item['config_sha']
        producer = load(directory / 'receipt.json')
        assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert producer['epochs'] == item['epochs']
        assert not (directory / 'metrics_test.csv').exists()
        assert not (directory / 'test_readout_receipt.json').exists()
        assert not any(directory.glob('test_epoch*.npz'))
    predecessors = load(predecessors_file)
    registry = load(out / 'orchestrator_checkpoint.json')['active_gpu_status_files']
    assert registry[:len(predecessors)] == predecessors
    assert registry[len(predecessors)] == status.name
    assert all((out / name).exists() for name in predecessors)
    command = [sys.executable, str(queue), '--manifest', str(manifest)]
    for name in predecessors:
        command.extend(['--preceding-status', str(out / name)])
    log = out / 'logs' / (stem + '_queue.log')
    started = time.time()
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    result = dict(pid=process.pid, command=command, cwd=str(root), started_unix=started,
        code_commit_at_queue_launch=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        manifest=str(manifest), manifest_sha256=sha(manifest),
        activation_review=str(review), activation_review_sha256=sha(review),
        predecessors_file=str(predecessors_file), predecessors_sha256=sha(predecessors_file),
        predecessor_count=len(predecessors), test_global_freeze_sha256=sha(freeze_file),
        native_inputs_sha256=sha(inputs_file), source_sha256={
            'readout': sha(engine), 'queue': sha(queue), 'launcher': sha(Path(__file__))},
        stdout_log=str(log), status=str(status), start_new_session=True, returncode=None)
    with launch.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'command'}))


if __name__ == '__main__':
    main()
