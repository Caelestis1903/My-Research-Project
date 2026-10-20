"""Verify prescribed completion and frozen zero-update readout, without scoring."""
import hashlib
import json
import zipfile
from pathlib import Path
import pandas as pd
from cm19_train import ROOT, OUT, dump


def main():
    manifest = json.loads((OUT / 'run_manifest.json').read_text())
    freeze = json.loads((OUT / 'TEST_FREEZE.json').read_text())
    assert len(manifest['runs']) == 856 and len(manifest['aliases']) == 72
    assert hashlib.sha256((OUT / 'run_manifest.json').read_bytes()).hexdigest() == freeze['manifest_sha256']
    assert hashlib.sha256((OUT / 'DEVELOPMENT_DISPOSITION.md').read_bytes()).hexdigest() == freeze['development_sha256']
    for name, digest in freeze['training_source_hashes'].items():
        assert hashlib.sha256((ROOT / 'experiments' / name).read_bytes()).hexdigest() == digest
    metric_rows = 0
    prediction_archives = 0
    for c in manifest['runs']:
        rr = OUT / 'runs' / c['run_id']
        train = json.loads((rr / 'receipt.json').read_text())
        test = json.loads((rr / 'test_receipt.json').read_text())
        cfg = json.loads((rr / 'config.json').read_text())
        expected = freeze['runs'][c['run_id']]
        assert train['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED' and train['from_scratch']
        assert train['steps'] == c['max_steps'] if c.get('max_steps') else train['epochs'] == c['epochs']
        for field, value in c.items():
            assert cfg[field] == value, (c['run_id'], field)
        for file, key in [('config.json', 'config_sha256'), ('receipt.json', 'receipt_sha256'),
                          ('selection_frozen.json', 'selection_sha256')]:
            assert hashlib.sha256((rr / file).read_bytes()).hexdigest() == expected[key]
        assert test['status'] == 'COMPLETE_FROZEN_TEST'
        assert test['optimizer_steps_during_evaluation'] == 0 and test['selected_and_final']
        assert test['checkpoints'] == expected['checkpoint_sha256']
        for label in ['selected', 'final']:
            file = Path(cfg['state_directory']) / f'test_{label}.npz'
            assert file.stat().st_size > 0, file
            with zipfile.ZipFile(file) as archive:
                assert {'predictions.npy', 'labels.npy', 'domains.npy', 'row_ids.npy'}.issubset(archive.namelist()), file
                assert archive.testzip() is None, file
            prediction_archives += 1
        metrics = pd.read_csv(rr / 'metrics_test.csv')
        assert len(metrics) == test['records']
        keys = ['checkpoint', 'intervention', 'domain', 'task']
        assert not metrics.duplicated(keys).any(), c['run_id']
        assert set(metrics.checkpoint) == {'selected', 'final'}
        target = {(d, t) for d in range(3) for t in cfg['task_ids']}
        for (checkpoint, intervention), block in metrics.groupby(['checkpoint', 'intervention']):
            assert set(zip(block.domain, block.task)) == target
            natural = metrics[(metrics.checkpoint == checkpoint) & (metrics.intervention == 'none')]
            joined = block.merge(natural[['domain', 'task', 'n']], on=['domain', 'task'], suffixes=('', '_natural'))
            assert len(joined) == len(target) and joined.n.eq(joined.n_natural).all()
        metric_rows += len(metrics)
    dump(OUT / 'FINAL_AUDIT.json', dict(
        status='PASS', independent_full_training=856, frozen_zero_update_tests=856,
        aliases=72, nominal_configurations=928, unique_test_metric_rows=metric_rows,
        config_receipt_selection_hashes_match=True, frozen_core_and_disposition_match=True,
        no_duplicate_test_cells=True,
        validated_test_prediction_archives=prediction_archives,
        prediction_validation='Nonempty ZIP/NPZ, required array members and every-member CRC; no rescoring.',
        checkpoint_content_hashes='Independently checked by cm19_artifacts.py final pass against TEST_FREEZE.',
        scope='Completion/freeze/cell-key integrity only; no new metric scoring or training.',
    ))
    print(f'PASS:856 full training +856 frozen tests; {metric_rows} unique test metric rows.')


if __name__ == '__main__':
    main()
