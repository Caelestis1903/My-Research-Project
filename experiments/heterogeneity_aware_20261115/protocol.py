"""Source protocol, immutable records and verified data-only cache."""
import hashlib
import json
import os
import pickle
import fcntl
from pathlib import Path
import torch
from experiments.conditional_information_momentum_20261102_v3.protocol import (
    load_protocol, load_dataset, make_model, setup, cpu_copy, state_digest,
    rng_state, restore_rng, evaluate, write_json)
from experiments.conditional_information_momentum_20261102_v3.guardian import request, ensure_guardian, alive

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (2022, 2023, 2024, 2025, 2026)
DATASETS = ('movielens', 'kuairand')
ARMS = ('CLEAN', 'A', 'B', 'C', 'RANDOM', 'GAUSSIAN')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_manifest():
    files = []
    for directory in (HERE, ROOT/'experiments/conditional_information_momentum_20261102_v3'):
        for path in sorted(directory.rglob('*')):
            if path.suffix in ('.py', '.json'):
                files.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(files=files, sha256=digest(files))


def immutable(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable identity differs: {path}'
    else:
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def config(dataset, arm, alpha=.03):
    assert dataset in DATASETS and arm in ARMS+('B_MATCH',)
    assert alpha in (.01, .03, .05)
    p = load_protocol()['datasets'][dataset]
    return dict(dataset=dataset, arm=arm, alpha=alpha, pair_budget=128, lambda_aug=.25,
                epochs=50, batch_size=40960, lr=.01, weight_decay=p['weight_decay'], rho=p['rho'],
                gate_training_detached=True, probe_gate_full_derivative=True,
                clean_loss='mean_probability_BCE', auxiliary_loss='2_mean_target_probability_BCE')


def optimizer(model, c):
    return torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'],
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False, amsgrad=False)


def job(dataset, arm, seed, alpha=.03):
    assert seed in SEEDS
    out = dict(dataset=dataset, seed=seed, config=config(dataset, arm, alpha), source_hash=source_manifest()['sha256'])
    out['key'] = digest(out)
    return out


def dataset_cache(dataset, attempt):
    path = Path(attempt)/'data_cache'/f'{dataset}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = load_protocol()['datasets'][dataset]
        h = hashlib.sha256()
        with open(identity['csv'], 'rb') as raw:
            for block in iter(lambda: raw.read(8 << 20), b''):
                h.update(block)
        assert h.hexdigest() == identity['source']['raw_sha256']
        if path.exists():
            with path.open('rb') as f:
                bundle, manifest = pickle.load(f)
            assert manifest['raw_sha256'] == h.hexdigest()
        else:
            bundle, manifest = load_dataset(dataset)
            with path.with_suffix('.tmp').open('wb') as f:
                pickle.dump((bundle, manifest), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(path.with_suffix('.tmp'), path)
        from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import array_hash
        for field, expected in identity['source']['array_hashes'].items():
            assert array_hash(getattr(bundle, field)) == expected, (dataset, field)
        return bundle, manifest


def check_review(path):
    review = json.loads(Path(path).read_text())
    assert review['verdict'] in ('PASS', 'PASS WITH FIXES') and not review.get('blocking_issues')
    assert review['source_hash'] == source_manifest()['sha256'], 'Reviewed source changed'
    return review


def write_manifests(attempt):
    import subprocess
    source = source_manifest()
    immutable(Path(attempt)/'source_receipt.json', source)
    immutable(Path(attempt)/'run_manifest.json', dict(
        source_hash=source['sha256'], git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        seeds=list(SEEDS), seed_override='User: only 2022-2026; development and confirmation reuse seeds',
        datasets=list(DATASETS), pilot_arms=list(ARMS), pilot_alpha=.03,
        epochs=50, batch_size=40960, pair_budget=128, lambda_aug=.25,
        hpo_alphas=[.01, .03, .05], max_hpo_routes=2,
        clean_loss='mean over rows and both tasks of probability BCE',
        selection='earliest strict maximum of validation six-cell Macro AUC; independent online and EMA',
        test_access='after frozen configuration only', model_storage='RAM_ONLY',
        design_sha256=hashlib.sha256((ROOT/'research_reports/2026-11-15_heterogeneity_aware/DESIGN_SPECIFICATION.md').read_bytes()).hexdigest(),
        configs={d:config(d,'A') for d in DATASETS}))


def jsonl(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
