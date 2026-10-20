"""Launch a primary-approved CPU manifest with byte-derived provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_plans/2026-10-16_v3/execution_20261016'
    stem = sys.argv[1]
    assert '/' not in stem and '..' not in stem
    manifest = out / f'{stem}_CPU_manifest.json'
    review = out / f'{stem}_activation_review.json'
    launch = out / f'{stem}_CPU_queue_launch.json'
    status = manifest.with_suffix('.status.json')
    assert not launch.exists() and not status.exists()
    approval = json.loads(review.read_text())
    assert approval['status'].startswith('PASS')
    assert approval['manifest_sha'] == sha(manifest)
    spec = json.loads(manifest.read_text())
    assert Path(spec['cwd']) == root
    assert 1 <= spec['max_cpu_processes'] <= 12
    sources = {}
    for task in spec['tasks']:
        source = Path(task['command'][1])
        if not source.is_absolute():
            source = root / source
        assert source.name in ('v3_selection_budget_confirmation_20261016.py',
                               'v3_project_pair_prefix_20261016.py')
        digest = sha(source)
        assert digest == approval['source_sha']
        sources[str(source)] = digest
    queue = root / 'experiments/v3_cpu_queue_20261016.py'
    command = [sys.executable, str(queue), '--manifest', str(manifest)]
    log = out / 'logs' / f'{stem}_CPU_queue.log'
    with log.open('ab') as handle:
        process = subprocess.Popen(command, cwd=root, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    record = dict(pid=process.pid, command=command, cwd=str(root),
        started_unix=time.time(), code_commit_at_queue_launch=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        manifest=str(manifest), manifest_sha256=sha(manifest),
        activation_review=str(review), activation_review_sha256=sha(review),
        queue_source_sha256=sha(queue), launcher_source_sha256=sha(__file__),
        task_source_sha256=sources, status_path=str(status), stdout_log=str(log),
        returncode=None, start_new_session=True)
    with launch.open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
