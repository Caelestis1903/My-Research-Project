"""Primary read-only verification of an already reviewed exact continuation copy."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/xzf/Code/phase0-3-20261016-v3-git')
OUT = ROOT / 'research_plans/2026-10-16_v3/execution_20261016'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    stem = sys.argv[1]
    manifest_path = OUT / (stem + '_copy_manifest.json')
    receipt_path = OUT / (stem + '_copy_receipt.json')
    train_path = OUT / (stem + '_manifest.json')
    launch = load(OUT / (stem + '_copy_launch.json'))
    copied = load(receipt_path)
    manifest, training = load(manifest_path), load(train_path)
    assert copied['status'] == 'PASS'
    assert sha(ROOT / 'experiments/v3_prepare_continuation_20261016.py') == copied['source_sha'] == launch['copier_source_sha']
    assert sha(manifest_path) == copied['manifest_sha'] == launch['manifest_sha256']
    assert len(copied['continuations']) == len(manifest['continuations']) == len(training['tasks'])
    records = []
    for item, copy, task in zip(manifest['continuations'], copied['continuations'], training['tasks']):
        source, target = Path(item['source']), Path(item['target'])
        assert copy['source'] == str(source) and copy['target'] == str(target)
        assert sha(source / 'receipt.json') == sha(target / 'receipt.json') == copy['source_receipt_sha']
        producer = load(source / 'receipt.json')
        assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert producer['epochs'] == item['from_epochs'] < item['to_epochs'] == copy['requested_epochs']
        assert copy['from_epochs'] == item['from_epochs']
        assert copy['copied_artifacts_checked'] == len(producer['artifact_hashes'])
        assert copy['copied_bytes_checked'] == sum((target / name).stat().st_size for name in producer['artifact_hashes'])
        assert copy['prefix_identity'] == 'ALL_ORIGINAL_PRODUCER_HASHES_MATCH'
        assert copy['independent_training_replicate'] is False and copy['original_inputs_mutated'] is False
        names = ['config.json', 'trajectory.csv', 'metrics_selection.csv',
                 f'epoch{producer["epochs"]:03d}.pt',
                 f'validation_epoch{producer["epochs"]:03d}.npz',
                 f'batches_epoch{producer["epochs"]:03d}.npz']
        for name in names:
            assert sha(source / name) == sha(target / name) == producer['artifact_hashes'][name]
        # Every inherited file must remain a distinct inode, including metadata
        # absent from the producer's own artifact inventory.
        inherited_metadata = []
        for path in source.iterdir():
            clone = target / path.name
            assert path.is_file() and clone.is_file() and not path.is_symlink() and not clone.is_symlink()
            assert (path.stat().st_dev, path.stat().st_ino) != (clone.stat().st_dev, clone.stat().st_ino)
            if path.name.startswith('continuation_source'):
                assert sha(path) == sha(clone)
                inherited_metadata.append(path.name)
        metadata = target / copy['provenance_artifact']
        assert load(metadata) == copy
        assert not (source / metadata.name).exists()
        assert set(p.name for p in target.iterdir()) == set(p.name for p in source.iterdir()) | {metadata.name}
        command = task['command']
        assert Path(command[command.index('--run-dir') + 1]) == target
        assert int(command[command.index('--epochs') + 1]) == item['to_epochs'] and '--resume' in command
        trainer = ROOT / command[1]
        assert sha(trainer) == launch['trainer_source_sha']
        records.append(dict(source=str(source), target=str(target), source_receipt_sha=copy['source_receipt_sha'],
                            actual_prefix_epochs=producer['epochs'], requested_epochs=item['to_epochs'],
                            primary_critical_hash_checks=len(names), all_artifacts_copier_verified=len(producer['artifact_hashes']),
                            all_original_inodes_distinct=True, inherited_metadata_verified=inherited_metadata,
                            provenance_artifact=metadata.name))
    result = dict(status='PASS', time=time.time(), independent_review=False,
                  source_sha=sha(Path(__file__)), copy_receipt_sha=sha(receipt_path),
                  copy_manifest_sha=sha(manifest_path), training_manifest_sha=sha(train_path),
                  records=records, actual_training_authorized=True, test_access=False,
                  scope='Exact continuation only; all producer artifacts checked by frozen copier; primary independently rehashes six critical artifacts per run and all inherited continuation metadata, checks every inherited inode, honest old-duration receipts, new provenance, unchanged trainer and requested resume commands. No scientific stopping decision or new algorithm.')
    path = OUT / (stem + '_activation_review.json')
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), status=result['status'], runs=len(records),
                          critical_hash_checks=sum(x['primary_critical_hash_checks'] for x in records))))


if __name__ == '__main__':
    main()
