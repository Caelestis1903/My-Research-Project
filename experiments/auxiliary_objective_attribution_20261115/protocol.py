"""Frozen identities, verified read-only data cache, and CLEAN evidence reuse."""
import gzip
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path
import torch
from experiments.conditional_information_momentum_20261102_v3.protocol import (
    load_protocol, load_dataset, make_model, setup, cpu_copy, state_digest,
    rng_state, restore_rng, evaluate, write_json)
from experiments.conditional_information_momentum_20261102_v3.guardian import request, ensure_guardian, alive
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.data import sha_file
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import array_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT/'research_reports/auxiliary_objective_attribution_20261115'
SEEDS = (2022, 2023, 2024, 2025, 2026)
DATASETS = ('movielens', 'kuairand')
ARMS = ('B', 'C', 'Z-PAIR', 'Z-EXAMPLE')
PILOT_COMMIT = '52b48c9aab066d56fa91a060864c7c9abf12e50c'
SELECTION = 'earliest strict maximum of validation six-cell Macro AUC; independent online and EMA'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_manifest():
    paths = [HERE/name for name in ('__init__.py', 'core.py', 'protocol.py', 'runner.py', 'checks.py')]
    source = ROOT/'experiments/conditional_information_momentum_20261102_v3'
    paths += sorted(p for p in source.rglob('*') if p.suffix in ('.py', '.json'))
    files = [dict(path=str(p.relative_to(ROOT)), sha256=sha_file(p)) for p in paths]
    return dict(files=files, sha256=digest(files))


def immutable(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable identity differs: {path}'
    else:
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def config(dataset, arm):
    assert dataset in DATASETS and arm in ARMS
    p = load_protocol()['datasets'][dataset]
    return dict(dataset=dataset, arm=arm, alpha=.03, pair_budget=128, lambda_aug=.25,
        epochs=50, batch_size=40960, precision='float32', lr=.01, betas=[.9,.999], eps=1e-8,
        weight_decay=p['weight_decay'], rho=p['rho'], gate_training_detached=True,
        probe_gate_full_derivative=True, clean_loss='mean_probability_BCE',
        auxiliary_loss='2_mean_target_probability_BCE', sampler='pilot_sha256_keyed_v1',
        objective='pilot_extra_target_BCE_v1', data_hash=digest(p['source']), selection_hash=digest(SELECTION))


def optimizer(model, c):
    return torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'],
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False, amsgrad=False)


def job(dataset, arm, seed):
    assert seed in SEEDS
    c = config(dataset, arm)
    out = dict(dataset=dataset, seed=seed, config=c, config_hash=digest(c), source_hash=source_manifest()['sha256'])
    out['key'] = digest(out)
    return out


def dataset_cache(dataset, cache_dir=None):
    identity = load_protocol()['datasets'][dataset]
    assert sha_file(identity['csv']) == identity['source']['raw_sha256']
    path = Path(cache_dir)/f'{dataset}.pkl' if cache_dir else None
    if path is None or not path.exists():
        return load_dataset(dataset)
    with path.open('rb') as f:
        bundle, manifest = pickle.load(f)
    expected = identity['source']
    assert manifest['raw_sha256'] == expected['raw_sha256']
    for field, value in expected['array_hashes'].items():
        assert array_hash(getattr(bundle, field)) == value, (dataset, field)
    assert bundle.split_fingerprints() == expected['split_row_fingerprints']
    assert bundle.row_id_fingerprints() == expected['split_row_id_hashes']
    assert bundle.features.encoder_hash == expected['encoder_hash']
    assert bundle.features.scaler_hash == expected['scaler_hash']
    for field in ('array_hashes', 'split_row_fingerprints', 'split_row_id_hashes',
                  'encoder_hash', 'scaler_hash', 'split_rows', 'train_domain_counts'):
        assert manifest[field] == expected[field], (dataset, field)
    return bundle, manifest


def check_review(path):
    review = json.loads(Path(path).read_text())
    assert review['verdict'] in ('PASS', 'PASS WITH FIXES') and not review.get('blocking_issues')
    assert review['source_hash'] == source_manifest()['sha256'], 'Reviewed training source changed'
    return review


def gzip_jsonl(path, value):
    with gzip.open(path, 'at', compresslevel=3) as f:
        f.write(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def copy_reuse(attempt, pilot_root):
    attempt = Path(attempt); pilot_root = Path(pilot_root)
    original = pilot_root/'artifacts/heterogeneity_aware_20261115/attempt1'
    files = [original/name for name in ('source_receipt.json', 'run_manifest.json',
        'movielens_data_manifest.json', 'kuairand_data_manifest.json')]
    files += [pilot_root/'research_reports/2026-11-15_heterogeneity_aware'/name for name in
        ('COMPLETION.json', 'IMPLEMENTATION_REVIEW.json', 'FINAL_REVIEW.json')]
    records = []
    for path in sorted((original/'runs').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result['job']['config']['arm'] != 'CLEAN':
            continue
        assert result['epochs'] == 50 and not result['test_accessed']
        records.append(result)
        files += [path] + sorted(path.parent.glob('attempt*/epoch_*.json'))
        files += sorted(path.parent.glob('attempt*/batches.jsonl'))
        files += sorted(path.parent.glob('attempt*/data_manifest.json'))
    assert len(records) == 10
    assert {(r['job']['dataset'], r['job']['seed']) for r in records} == {(d,s) for d in DATASETS for s in SEEDS}
    receipts = []
    for path in files:
        relative = path.relative_to(pilot_root)
        target = attempt/'reused'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha_file(target) == sha_file(path)
        else:
            shutil.copyfile(path, target)
        receipts.append(dict(source=str(relative), target=str(target.relative_to(attempt)), sha256=sha_file(target)))
    immutable(attempt/'reuse_receipt.json', dict(source_commit=PILOT_COMMIT, trajectories=10, files=receipts))
    return records


def prepare(attempt, cache_dir, pilot_root):
    attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    source = source_manifest()
    immutable(attempt/'source_receipt.json', source)
    immutable(attempt/'run_manifest.json', dict(source_hash=source['sha256'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seeds=list(SEEDS), datasets=list(DATASETS), arms=list(ARMS), trajectories=40,
        reused_clean_trajectories=10, epochs=50, batch_size=40960, optimizer_updates=69000, ema_updates=69000,
        configs={d:{a:config(d,a) for a in ARMS} for d in DATASETS},
        cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
        selection=SELECTION, selection_hash=digest(SELECTION), model_storage='RAM_ONLY', test_accessed=False,
        design_sha256=sha_file(REPORT/'DESIGN_SPECIFICATION.md')))
    copy_reuse(attempt, pilot_root)
    for dataset in DATASETS:
        _, manifest = dataset_cache(dataset, cache_dir)
        immutable(attempt/f'{dataset}_data_manifest.json', manifest)
    return dict(status='PREPARED', source_hash=source['sha256'], new_trajectories=40, reused=10)


def results(attempt):
    attempt = Path(attempt)
    paths = sorted((attempt/'runs').glob('*/result.json'))
    paths += sorted((attempt/'reused/artifacts/heterogeneity_aware_20261115/attempt1/runs').glob('*/result.json'))
    return [json.loads(p.read_text()) for p in paths]


def epoch_files(attempt, result):
    root = Path(attempt)/'runs'/result['job']['key']
    if result['job']['config']['arm'] == 'CLEAN':
        root = Path(attempt)/'reused/artifacts/heterogeneity_aware_20261115/attempt1/runs'/result['job']['key']
    by_epoch = {}
    for path in sorted(root.glob('attempt*/epoch_*.json')):
        record = json.loads(path.read_text()); by_epoch[record['epoch']] = (path, record)
    return by_epoch
