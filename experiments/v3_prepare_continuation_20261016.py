"""Copy and verify an immutable training prefix for a separate continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    manifest = json.loads(args.manifest.read_text())
    records = []
    for item in manifest['continuations']:
        source, target = Path(item['source']), Path(item['target'])
        assert not target.exists(), target
        receipt_sha = sha(source / 'receipt.json')
        receipt = json.loads((source / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert receipt['epochs'] == item['from_epochs'] < item['to_epochs']
        assert all(path.is_file() and not path.is_symlink() for path in source.iterdir())
        started = time.time()
        subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(target)], check=True)
        for original in source.iterdir():
            original_stat, clone_stat = original.stat(), (target / original.name).stat()
            assert (original_stat.st_dev, original_stat.st_ino) != (clone_stat.st_dev, clone_stat.st_ino), original
        checked, size = 0, 0
        for name, digest in receipt['artifact_hashes'].items():
            clone = target / name
            assert sha(clone) == digest, clone
            original_stat, clone_stat = (source / name).stat(), clone.stat()
            assert (original_stat.st_dev, original_stat.st_ino) != (clone_stat.st_dev, clone_stat.st_ino), name
            checked += 1
            size += clone_stat.st_size
        assert sha(target / 'receipt.json') == receipt_sha == sha(source / 'receipt.json')
        provenance = dict(source=str(source), target=str(target), source_receipt_sha=receipt_sha,
                          from_epochs=item['from_epochs'], requested_epochs=item['to_epochs'],
                          copied_artifacts_checked=checked, copied_bytes_checked=size,
                          prefix_identity='ALL_ORIGINAL_PRODUCER_HASHES_MATCH',
                          independent_training_replicate=False,
                          original_inputs_mutated=False, preparation_seconds=time.time() - started)
        (target / 'continuation_source.json').write_text(json.dumps(provenance, indent=2) + '\n')
        records.append(provenance)
    args.out.write_text(json.dumps(dict(status='PASS', source_sha=sha(Path(__file__)),
                                        manifest_sha=sha(args.manifest), continuations=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
