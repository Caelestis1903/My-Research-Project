"""Archive this study's scalar records and the exact historical inputs it reused."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
HISTORY = Path('/home/xzf/Code/phase0-3')
ATTEMPT = HISTORY/'artifacts/mdmt_overnight_ab_20261103/attempt1'
ARMS = {'HYPERGATE', 'AUTOIFS', 'C_MIX_P01_SERIAL_V1'}
ALLOWED = {'.json', '.jsonl', '.csv', '.log', '.txt', '.yaml', '.yml'}


def main():
    jobs = json.loads((ATTEMPT/'queue_state.json').read_text())['jobs']
    new = [s for key, s in jobs.items() if key.split('.')[1] in ARMS]
    assert len(new) == 30 and all(s['status'] == 'DONE' for s in new), 'Training incomplete'
    files = {}
    def add(path):
        path = path.resolve()
        relative = str(path.relative_to(HISTORY))
        assert path.suffix in ALLOWED, relative
        if relative not in files:
            data = path.read_bytes()
            files[relative] = dict(path=relative, bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest())
        return files[relative]

    names = ('CONFIGURATION_FROZEN.json', 'PREPARED.json', 'SOURCE_LOCK.json',
        'DATA_MANIFEST.json', 'ENVIRONMENT.json', 'RUNTIME_OVERRIDES.json',
        'JOBS.csv', 'IDENTITY_ALIASES.json', 'A_REUSE_AUDIT.csv', 'B_REUSE_AUDIT.csv',
        'progress.json', 'queue_state.json', 'events.jsonl', 'failures.jsonl',
        'resource_usage.csv', 'SMOKE.json', 'coordinator.log', 'guardian.log',
        'epoch_commit_ledger.jsonl', 'COMPLETE.json')
    for name in names:
        if (ATTEMPT/name).is_file():
            add(ATTEMPT/name)
    for path in ATTEMPT.glob('IMPLEMENTATION_LOCK_*.json'):
        add(path)
    for directory in ('runs', 'jobs'):
        for path in (ATTEMPT/directory).rglob('*'):
            if path.is_file() and path.suffix in ALLOWED:
                add(path)
    for row in csv.DictReader((HERE/'ENDPOINTS.csv').open()):
        source = row['source_path'] or row.get('source_result')
        assert source
        path = Path(source)
        if not path.is_absolute():
            path = HISTORY/path
        entry = add(path)
        expected = row['source_sha256'] or row.get('source_result_sha256')
        assert expected and entry['sha256'] == expected, f'Source checksum: {source}'
    old = HISTORY/'research_reports/2026-11-03_mdmt_five_seed_completion'
    for name in ('endpoints.csv', 'validation_epochs.csv', 'RUN_MANIFEST.json', 'DECISIONS.json'):
        add(old/name)
    previous = HISTORY/'artifacts/mdmt_five_seed_completion_20261103/attempt1'
    for name in ('PREPARED.json', 'COMPLETE.json', 'history/final_evaluation.json',
                 'pair/final_evaluation.json'):
        add(previous/name)

    archive = HERE/'raw_training_scalars.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name in sorted(files):
            output.add(HISTORY/name, arcname=name, recursive=False)
    manifest = dict(archive=archive.name, archive_bytes=archive.stat().st_size,
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    file_count=len(files), files=[files[n] for n in sorted(files)])
    parts = []
    with archive.open('rb') as source:
        while chunk := source.read(2*1024*1024):
            path = HERE/f'{archive.name}.part{len(parts):03d}'
            path.write_bytes(chunk)
            parts.append(dict(path=path.name, bytes=len(chunk),
                              sha256=hashlib.sha256(chunk).hexdigest()))
    manifest['parts'] = parts
    (HERE/'RAW_ARTIFACTS_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))


if __name__ == '__main__':
    main()
