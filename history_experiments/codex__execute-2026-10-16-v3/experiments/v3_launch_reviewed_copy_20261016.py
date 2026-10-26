"""Launch an approved immutable prefix copy with computed provenance."""
import hashlib
import json
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
    manifest = out / (stem + '_copy_manifest.json')
    training = out / (stem + '_manifest.json')
    review = out / (stem + '_copy_activation_review.json')
    receipt = out / (stem + '_copy_receipt.json')
    launch = out / (stem + '_copy_launch.json')
    assert not launch.exists() and not receipt.exists()
    approval = load(review)
    assert approval['status'] == 'PASS' and approval['actual_copy_authorized'] is True
    assert sha(manifest) == approval['copy_manifest_sha']
    assert sha(training) == approval['training_manifest_sha']
    copier = root / 'experiments/v3_prepare_continuation_20261016.py'
    assert sha(copier) == approval['copier_source_sha']
    sources = set()
    for item, task in zip(load(manifest)['continuations'], load(training)['tasks'], strict=True):
        source, target = Path(item['source']), Path(item['target'])
        producer = source / 'receipt.json'
        assert sha(producer) == approval['source_receipts'][str(source)]
        assert load(producer)['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert load(producer)['epochs'] == item['from_epochs'] < item['to_epochs']
        assert not target.exists()
        sources.add(sha(root / task['command'][1]))
    assert sources == {approval['trainer_source_sha']}
    active = []
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = cmdline.read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if len(args) > 1 and Path(args[1].decode(errors='replace')).name == copier.name:
            active.append(int(cmdline.parent.name))
    assert len(active) < 2, active
    command = [sys.executable, str(copier), '--manifest', str(manifest), '--out', str(receipt)]
    log = out / 'logs' / (stem + '_copy.log')
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(pid=process.pid, command=command, cwd=str(root), started_unix=time.time(),
        code_commit_at_queue_launch=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        manifest=str(manifest), manifest_sha256=sha(manifest), training_manifest_sha256=sha(training),
        activation_review=str(review), activation_review_sha256=sha(review),
        copier_source_sha=sha(copier), trainer_source_sha=next(iter(sources)),
        launcher_source_sha256=sha(__file__), other_active_copier_pids=active,
        stdout_log=str(log), receipt=str(receipt), start_new_session=True, returncode=None)
    with launch.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
