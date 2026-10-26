"""Primary pretraining review of a completed independent pair in a serial copier."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/xzf/Code/phase0-3-20261016-v3-git')
OUT = ROOT / 'research_plans/2026-10-16_v3/execution_20261016'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while b := stream.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(name, data):
    path = OUT / name
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2) + '\n')
    return path


def main():
    pair = int(sys.argv[1])
    assert pair in range(1, 6)
    stem = 'V5V8_seed2025_continuation250'
    design_path = OUT / (stem + '_streamed_pair_activation_design.json')
    design = load(design_path)
    assert design['status'] == 'PASS_SCHEDULING_AMENDMENT_BEFORE_ANY250_TRAINING'
    copy_manifest = OUT / (stem + '_copy_manifest.json')
    template = OUT / (stem + '_manifest.json')
    launch = load(OUT / (stem + '_copy_launch.json'))
    assert sha(copy_manifest) == design['copy_manifest_sha'] == launch['manifest_sha256']
    assert sha(template) == design['training_template_sha']
    assert sha(ROOT / 'experiments/v3_prepare_continuation_20261016.py') == design['copy_source_sha'] == launch['copier_source_sha']
    assert sha(ROOT / 'experiments/v3_train_20261016.py') == launch['trainer_source_sha']
    items = load(copy_manifest)['continuations'][(pair - 1) * 2: pair * 2]
    train = load(template)
    tasks = train['tasks'][(pair - 1) * 2:pair * 2]
    records = []
    for item, task in zip(items, tasks):
        source, target = Path(item['source']), Path(item['target'])
        assert item['from_epochs'] == 200 and item['to_epochs'] == 250
        metadata = target / 'continuation_source_to250.json'
        copy = load(metadata)
        assert copy['source'] == str(source) and copy['target'] == str(target)
        assert copy['from_epochs'] == 200 and copy['requested_epochs'] == 250
        assert copy['provenance_artifact'] == metadata.name
        assert copy['prefix_identity'] == 'ALL_ORIGINAL_PRODUCER_HASHES_MATCH'
        assert copy['independent_training_replicate'] is False and copy['original_inputs_mutated'] is False
        assert sha(source / 'receipt.json') == sha(target / 'receipt.json') == copy['source_receipt_sha']
        producer = load(source / 'receipt.json')
        assert producer['epochs'] == 200 and producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert copy['copied_artifacts_checked'] == len(producer['artifact_hashes'])
        assert copy['copied_bytes_checked'] == sum((target / name).stat().st_size for name in producer['artifact_hashes'])
        critical = {}
        for name in ['config.json', 'trajectory.csv', 'metrics_selection.csv', 'epoch200.pt', 'validation_epoch200.npz', 'batches_epoch200.npz']:
            digest = sha(source / name)
            assert digest == sha(target / name) == producer['artifact_hashes'][name]
            critical[name] = digest
        inherited = {}
        count = 0
        for path in source.iterdir():
            clone = target / path.name
            assert path.is_file() and clone.is_file() and not path.is_symlink() and not clone.is_symlink()
            assert (path.stat().st_dev, path.stat().st_ino) != (clone.stat().st_dev, clone.stat().st_ino)
            count += 1
            if path.name.startswith('continuation_source'):
                assert sha(path) == sha(clone)
                inherited[path.name] = sha(path)
        assert set(p.name for p in target.iterdir()) == set(p.name for p in source.iterdir()) | {metadata.name}
        assert not (source / metadata.name).exists()
        command = task['command']
        assert Path(command[command.index('--run-dir') + 1]) == target and '--resume' in command
        assert command[command.index('--epochs') + 1] == '250'
        records.append(dict(source=str(source), target=str(target), pretraining_receipt_sha=copy['source_receipt_sha'],
                            critical_artifact_hashes=critical, all_inherited_inodes_distinct=count,
                            inherited_metadata_hashes=inherited, provenance_sha=sha(metadata),
                            actual_copier_completed_record=copy, current_target_epochs_before_training=200))
    newstem = stem + f'_pair{pair}'
    manifest_path = write(newstem + '_manifest.json', {**train, 'tasks': tasks})
    review_path = write(newstem + '_activation_review.json', dict(status='PASS', time=time.time(), independent_review=False,
        source_sha=sha(Path(__file__)), design_sha=sha(design_path), copy_manifest_sha=sha(copy_manifest),
        training_template_sha=sha(template), manifest_sha=sha(manifest_path), records=records,
        actual_training_authorized=True, scientific_budget='All five pairs/all10 conditions mandatory matched250, subject unchanged repeated stopping rule. No partial-factor conclusion.',
        global_copier_may_continue_other_targets=True, test_access=False))
    checkpoint = load(OUT / 'orchestrator_checkpoint.json')
    predecessors = checkpoint['active_gpu_status_files'][:]
    status_name = newstem + '_manifest.status.json'
    assert status_name not in predecessors
    write(newstem + '_predecessors.json', predecessors)
    checkpoint['active_gpu_status_files'].append(status_name)
    checkpoint['active'][newstem] = 'Primary pretraining prefix PASS; exact original two commands authorized with all registered predecessors. Actual launch pending.'
    checkpoint['last_primary_update'] = time.time()
    (OUT / 'orchestrator_checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', pair=pair, runs=2, critical_physical_hashes=12,
        predecessor_count=len(predecessors), manifest=str(manifest_path), activation_review=str(review_path))))


if __name__ == '__main__':
    main()
