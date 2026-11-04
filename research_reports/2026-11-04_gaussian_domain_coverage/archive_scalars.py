"""Archive the completed study's scalar records and exact reused inputs."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATTEMPT = ROOT / 'artifacts/gaussian_domain_coverage_20261104/attempt1'
ALLOWED = {'.json', '.jsonl', '.csv', '.log', '.txt', '.yaml', '.yml'}


def main():
    prepared = json.loads((ATTEMPT / 'PREPARED.json').read_text())
    history = Path(prepared['history_root'])
    jobs = json.loads((ATTEMPT / 'queue_state.json').read_text())['jobs']
    assert len(jobs) == 10 and all(row['status'] == 'DONE' for row in jobs.values())
    assert json.loads((ATTEMPT / 'COMPLETE.json').read_text())['committed_epochs'] == 500
    files = {}

    def add(path, expected=None):
        path = path.resolve()
        assert path.suffix in ALLOWED, path
        if path.is_relative_to(ROOT):
            name = 'execution/' + str(path.relative_to(ROOT))
        else:
            name = 'history/' + str(path.relative_to(history))
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        assert expected is None or checksum == expected, path
        files[name] = dict(path=name, bytes=len(data), sha256=checksum, source=path)

    for path in ATTEMPT.iterdir():
        if path.is_file() and path.suffix in ALLOWED and path.name != 'guardian.json':
            add(path)
    for directory in ('runs', 'replay_history'):
        for path in (ATTEMPT / directory).rglob('*'):
            if path.is_file() and path.suffix in ALLOWED:
                add(path)
    for row in csv.DictReader((HERE / 'ENDPOINTS.csv').open()):
        add(Path(row['source_path']), row['source_sha256'])
    endpoint_table = prepared['endpoint_table']
    add(Path(endpoint_table['path']), endpoint_table['sha256'])
    original = prepared['original_gaussian_configuration']
    add(Path(original['path']), original['sha256'])

    archive = HERE / 'raw_training_scalars.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name, row in sorted(files.items()):
            output.add(row['source'], arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as check:
        assert set(check.getnames()) == set(files)
        for member in check:
            assert hashlib.sha256(check.extractfile(member).read()).hexdigest() == files[member.name]['sha256']
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = dict(archive=archive.name, archive_bytes=archive.stat().st_size,
                    archive_sha256=checksum, file_count=len(files),
                    roots=dict(execution=str(ROOT), history=str(history)),
                    files=[{k: v for k, v in row.items() if k != 'source'}
                           for _, row in sorted(files.items())])
    (HERE / 'RAW_ARTIFACTS_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (HERE / 'raw_training_scalars.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))


if __name__ == '__main__':
    main()
