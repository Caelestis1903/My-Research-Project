"""Preserve complete numerical run artifacts as ordinary <=80 MiB Git files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'raw_archive'
CHUNK = 80 * 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    files = set()
    for path in args.paths:
        files.update(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else files.add(path)
    files = sorted(files)
    assert files and not list(OUT.glob(args.name + '.tar.gz.part*'))
    records = []
    for path in files:
        assert path.is_file() and not path.is_symlink()
        assert path.suffix.lower() not in ('.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.pickle'), path
        relative = path.resolve().relative_to(ROOT)
        records.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha(path)))
    archive = Path('/tmp') / (args.name + '.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for path, record in zip(files, records):
            tar.add(path, arcname=record['path'], recursive=False)
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(CHUNK):
            path = OUT / f'{args.name}.tar.gz.part{index:03d}'
            path.write_bytes(block)
            parts.append(dict(file=path.name, bytes=len(block), sha256=sha(path)))
            index += 1
    # Read back tar headers and SHA each payload once, so restore completeness is checked.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == [r['path'] for r in records]
        for member, record in zip(members, records):
            digest = hashlib.sha256()
            payload = tar.extractfile(member)
            for block in iter(lambda: payload.read(1024 * 1024), b''):
                digest.update(block)
            assert digest.hexdigest() == record['sha256'], member.name
    manifest = dict(archive=args.name + '.tar.gz', archive_bytes=archive.stat().st_size,
                    archive_sha256=sha(archive), split_bytes=CHUNK, parts=parts,
                    file_count=len(records), files=records, payload_verification='all SHA256 match',
                    excluded_model_state_extensions=True)
    (OUT / (args.name + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / (args.name + '.SHA256SUMS')).write_text(''.join(p['sha256'] + '  ' + p['file'] + '\n' for p in parts))
    print(json.dumps({k: manifest[k] for k in ('archive', 'archive_bytes', 'archive_sha256', 'file_count')}))


if __name__ == '__main__':
    main()
