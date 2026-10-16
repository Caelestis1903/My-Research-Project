"""Reuse an exactly identical completed bootstrap prefix; never resample rows."""
from __future__ import annotations
import argparse
import ast
import csv
import json
from pathlib import Path
import types
import numpy as np
import pandas as pd
import v3_readout_20261016 as readout

FROZEN_READOUT_SHA = '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'
GPU_READOUT_SHA = '3e2024481c2a5f86a24c34b6a503c370a096f3bc7ec0b1c39694684998fa2900'
base = readout.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--natural', type=Path, required=True)
    parser.add_argument('--balanced', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path(readout.__file__)
    assert base.sha(source) == FROZEN_READOUT_SHA
    parent = args.parent.absolute()
    receipt = json.loads((parent / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    assert receipt['source_sha'] in (FROZEN_READOUT_SHA, GPU_READOUT_SHA)
    if receipt['source_sha'] == GPU_READOUT_SHA:
        assert base.sha(source.with_name('v3_gpu_readout_20261016.py')) == GPU_READOUT_SHA
        assert receipt['original_pair_serializer_sha'] == FROZEN_READOUT_SHA
        assert receipt['compute_backend'] == 'CUDA_float64'
    assert receipt['freeze_sha'] == base.sha(readout.OUT / 'analysis_rules_frozen.json')
    for name in ('bootstrap_curves.npz', 'cell_trajectory.csv'):
        assert base.sha(parent / name) == receipt['hashes'][name], name
    directories = (args.natural.absolute(), args.balanced.absolute())
    configs = [json.loads((d / 'config.json').read_text()) for d in directories]
    parent_directories = list(map(Path, receipt['source_runs']))
    count = args.epochs
    assert count >= 50
    verified = []
    for i, (directory, parent_directory, config) in enumerate(zip(directories, parent_directories, configs)):
        assert config['arm'] == ('SN', 'SB')[i]
        assert config['seed'] == receipt['seed'] and config['dataset'] == receipt['dataset']
        assert config == json.loads((parent_directory / 'config.json').read_text())
        producer = json.loads((directory / 'receipt.json').read_text())
        parent_producer = json.loads((parent_directory / 'receipt.json').read_text())
        assert producer['status'] == parent_producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        assert producer['epochs'] == count <= parent_producer['epochs']
        assert base.sha(parent_directory / 'receipt.json') == receipt['source_run_receipt_sha'][i]
        for name in ('config.json', 'trajectory.csv'):
            assert base.sha(directory / name) == producer['artifact_hashes'][name]
            assert base.sha(parent_directory / name) == parent_producer['artifact_hashes'][name]
        for epoch in range(1, count + 1):
            for name in (f'validation_epoch{epoch:03d}.npz', f'epoch{epoch:03d}_model.pt'):
                expected = producer['artifact_hashes'][name]
                assert expected == parent_producer['artifact_hashes'][name]
                assert base.sha(directory / name) == base.sha(parent_directory / name) == expected
        prefix_training = pd.read_csv(directory / 'trajectory.csv')
        parent_training = pd.read_csv(parent_directory / 'trajectory.csv')
        assert len(prefix_training) == count
        pd.testing.assert_frame_equal(prefix_training, parent_training.iloc[:count].reset_index(drop=True))
        verified.append(dict(prefix=str(directory), parent=str(parent_directory),
            prefix_receipt_sha=base.sha(directory / 'receipt.json'),
            parent_receipt_sha=base.sha(parent_directory / 'receipt.json'),
            identical_prediction_and_model_files=2 * count))
    with np.load(parent / 'bootstrap_curves.npz') as array:
        assert array['selection'].shape == (1000, 2, len(array['epochs']), 2)
        assert array['diagnostic'].shape == (200, 2, len(array['epochs']), 2)
        assert np.array_equal(array['epochs'], np.arange(1, len(array['epochs']) + 1))
        assert count <= len(array['epochs'])
        selection = array['selection'][:, :, :count].copy()
        diagnostic = array['diagnostic'][:, :, :count].copy()
        point = array['point'][:, :, :count].copy()
        epochs = array['epochs'][:count].tolist()
    assert point.shape == (2, 2, count, 2)
    assert all(np.isfinite(x).all() for x in (selection, diagnostic, point))
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    with (parent / 'cell_trajectory.csv').open(newline='') as inp, (out / 'cell_trajectory.csv').open('w', newline='') as dest:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        n_rows = 0
        for row in reader:
            if int(row['epoch']) > count:
                continue
            arm = ('SN', 'SB').index(row['arm'])
            assert row['run'] == str(parent_directories[arm])
            row['run'] = str(directories[arm])
            writer.writerow(row)
            n_rows += 1
        assert n_rows == 24 * count
    np.savez_compressed(out / 'bootstrap_curves.npz', selection=selection,
        diagnostic=diagnostic, point=point, epochs=epochs)
    base.dump(out / 'prefix_provenance.json', dict(status='EXACT_PREFIX_REUSE',
        design='prefix_bootstrap_cache_reuse_design.json', parent=str(parent),
        parent_receipt_sha=base.sha(parent / 'receipt.json'),
        parent_cache_sha=base.sha(parent / 'bootstrap_curves.npz'),
        parent_compute_backend=receipt.get('compute_backend', 'CPU_numpy_float64'),
        serializer_sha=FROZEN_READOUT_SHA, epochs=count, sources=verified,
        bootstrap_weights_depend_on='seed, split, domain rows, replicate count; independent of epoch horizon',
        new_resampling=False, independent_replicates_added=False, test_access=False))
    # Execute the frozen serializer itself, without duplicating its statistical
    # decisions. Only the producer provenance/source_sha names the new utility.
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pair')
    seam = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'indices' for t in n.targets))
    tail = ast.Module(body=function.body[seam:], type_ignores=[])
    namespace = dict(vars(readout))
    namespace.update(out=out, configs=configs, directories=directories, epochs=epochs,
        selection=selection, diagnostic=diagnostic, point=point,
        seed=receipt['seed'], dataset=receipt['dataset'],
        args=types.SimpleNamespace(bootstraps=1000), __file__=__file__)
    exec(compile(tail, str(source), 'exec'), namespace)


if __name__ == '__main__':
    main()
