"""Reference exact native-test predictions for an immutable model prefix."""
import argparse
import json
from pathlib import Path

import pandas as pd
import v3_readout_20261016 as r

FROZEN_READOUT_SHA = '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    assert r.base.sha(r.__file__) == FROZEN_READOUT_SHA
    freeze_path = r.OUT / 'test_readout_frozen.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['status'] == 'FROZEN' and freeze['all_condition_definitions_frozen'] is True
    assert freeze['analysis_rules_sha'] == r.base.sha(r.OUT / 'analysis_rules_frozen.json')
    assert freeze['test_reuse_manifest_sha'] == r.base.sha(args.manifest)
    item = json.loads(args.manifest.read_text())['pairs'][args.index]
    parent, target = Path(item['parent']).resolve(), Path(item['target']).resolve()
    assert parent != target
    for directory, key in ((parent, 'parent_producer_receipt_sha'),
                           (target, 'target_producer_receipt_sha')):
        assert r.base.sha(directory / 'receipt.json') == item[key]
        assert r.base.sha(directory / 'config.json') == item['config_sha']
    producers = [json.loads((d / 'receipt.json').read_text()) for d in (parent, target)]
    assert all(p['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED' for p in producers)
    count = producers[1]['epochs']
    assert count == item['epochs'] <= producers[0]['epochs']
    expected = {f'epoch{e:03d}_model.pt' for e in range(1, count + 1)}
    assert {x.name for x in target.glob('epoch*_model.pt')} == expected
    native_path = parent / 'test_readout_receipt.json'
    native = json.loads(native_path.read_text())
    assert native['status'] == 'COMPLETE' and native['source_sha'] == FROZEN_READOUT_SHA
    assert native['test_global_freeze_sha'] == r.base.sha(freeze_path)
    assert native['freeze_sha'] == freeze['analysis_rules_sha']
    assert sorted(native['epochs']) == list(range(1, producers[0]['epochs'] + 1))
    for name in ('metrics_test.csv', 'test_readout_receipt.json',
                 'test_prediction_reuse_provenance.json'):
        assert not (target / name).exists(), name
    assert not list(target.glob('test_epoch*.npz'))
    references = {}
    for epoch in range(1, count + 1):
        model_name = f'epoch{epoch:03d}_model.pt'
        digest = producers[0]['artifact_hashes'][model_name]
        assert producers[1]['artifact_hashes'][model_name] == digest
        assert r.base.sha(parent / model_name) == r.base.sha(target / model_name) == digest
        pred_name = f'test_epoch{epoch:03d}.npz'
        pred_sha = native['hashes'][pred_name]
        assert r.base.sha(parent / pred_name) == pred_sha
        references[str(epoch)] = dict(path=str(parent / pred_name), sha256=pred_sha,
                                      model_sha256=digest)
    parent_metrics = parent / 'metrics_test.csv'
    frame = pd.read_csv(parent_metrics)
    assert set(frame.epoch) == set(range(1, producers[0]['epochs'] + 1))
    assert not frame.duplicated(['epoch', 'domain', 'task']).any()
    assert len(frame) == producers[0]['epochs'] * 6
    frame = frame[frame.epoch <= count].copy()
    config = json.loads((target / 'config.json').read_text())
    assert (frame.arm == config['arm']).all() and (frame.seed == config['seed']).all()
    assert (frame.split == 'test').all() and len(frame) == count * 6
    frame.to_csv(target / 'metrics_test.csv', index=False)
    provenance = dict(status='EXACT_MODEL_PREFIX_TEST_PREDICTION_REUSE',
        parent=str(parent), target=str(target), epochs=count,
        native_test_receipt_sha=r.base.sha(native_path),
        parent_metrics_sha=r.base.sha(parent_metrics),
        parent_producer_receipt_sha=item['parent_producer_receipt_sha'],
        target_producer_receipt_sha=item['target_producer_receipt_sha'],
        exact_configuration_sha=item['config_sha'], prediction_references=references,
        newly_evaluated_models=0, additional_independent_replicates=0,
        no_child_prediction_files_claimed=True)
    r.base.dump(target / 'test_prediction_reuse_provenance.json', provenance)
    r.base.dump(target / 'test_readout_receipt.json', dict(status='COMPLETE',
        epochs=list(range(1, count + 1)), freeze_sha=freeze['analysis_rules_sha'],
        test_global_freeze_sha=r.base.sha(freeze_path), source_sha=r.base.sha(__file__),
        original_native_test_serializer_sha=FROZEN_READOUT_SHA,
        reuse_manifest_sha=r.base.sha(args.manifest), test_previously_exposed=True,
        inference='Exact trained model/config prefix; predictions referenced from native parent',
        prediction_references=references,
        hashes={name: r.base.sha(target / name) for name in
                ('metrics_test.csv', 'test_prediction_reuse_provenance.json')}))


if __name__ == '__main__':
    main()
