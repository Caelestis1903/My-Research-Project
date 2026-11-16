"""Frozen matrix identities and read-only reuse of verified historical controls."""
import json
import subprocess
from pathlib import Path
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SEEDS, DATASETS, SELECTION, digest, immutable, sha_file, dataset_cache, optimizer,
    load_protocol, make_model, setup, cpu_copy, state_digest, rng_state, restore_rng,
    evaluate, write_json)
from experiments.conditional_information_momentum_20261102_v3.guardian import request, alive

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT = ROOT/'artifacts/26-11-16_augmentation_gradient_momentum'
DESIGN = ARTIFACT/'DESIGN_SPECIFICATION.md'
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache')
REFERENCE = Path('/home/xzf/Code/phase0-3_auxiliary_20261115')
ARMS = ('TV-GRM-005', 'TV-GRM-010', 'TV-GRM-020', 'TV-GRM-SHUFFLE-010', 'TV-GRM-INSTANT-010')
GAMMAS = (.05, .10, .20)
NDP_SOURCE = '831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b'


def source_manifest():
    paths = [HERE/n for n in ('__init__.py', 'core.py', 'protocol.py', 'runner.py', 'tables.py', 'test_core.py', 'autonomous.py')]
    aux = ROOT/'experiments/auxiliary_objective_attribution_20261115'
    paths += [aux/n for n in ('__init__.py', 'core.py', 'protocol.py', 'runner.py', 'checks.py', 'test_core.py',
                              'stage2/__init__.py', 'stage2/core.py', 'stage2/tables.py')]
    source = ROOT/'experiments/conditional_information_momentum_20261102_v3'
    paths += sorted(p for p in source.rglob('*') if p.suffix in ('.py', '.json'))
    files = [dict(path=str(p.relative_to(ROOT)), sha256=sha_file(p)) for p in paths]
    return dict(files=files, sha256=digest(files), design_sha256=sha_file(DESIGN),
                selection_hash=digest(SELECTION))


def config(dataset, arm):
    assert dataset in DATASETS and arm in ARMS+('CLEAN',)
    p = load_protocol()['datasets'][dataset]
    mode = 'shuffle' if 'SHUFFLE' in arm else 'instant' if 'INSTANT' in arm else 'momentum'
    gamma = 0. if arm == 'CLEAN' else int(arm.rsplit('-', 1)[1])/100.
    return dict(dataset=dataset, arm=arm, gamma=gamma, residual_mode=mode, alpha=.03,
        epochs=50, batch_size=40960, precision='float32', lr=.01, betas=[.9, .999], eps=1e-8,
        weight_decay=p['weight_decay'], rho=p['rho'], model='M3oE_fixed1',
        optimizer='torch_Adam_clean_coupled_WD', cap='global_preconditioned_residual_update',
        support='natural_all_2B_entries', gate_training_detached=True,
        probe_gate_full_derivative=True, data_hash=digest(p['source']), selection_hash=digest(SELECTION))


def job(dataset, arm, seed, source_hash=None):
    assert seed in SEEDS and arm in ARMS
    c = config(dataset, arm)
    value = dict(dataset=dataset, seed=seed, config=c, config_hash=digest(c),
                 source_hash=source_hash or source_manifest()['sha256'])
    value['key'] = digest(value)
    return value


def check_review(path):
    review = json.loads(Path(path).read_text())
    assert review['verdict'] in ('PASS', 'PASS WITH FIXES') and not review.get('blocking_issues')
    current = source_manifest()
    assert review['source_hash'] == current['sha256'], 'Reviewed algorithm source changed'
    assert review['design_sha256'] == current['design_sha256'], 'Reviewed design changed'
    assert review['cpu_checks_passed'] is True
    return review


def stop_requested():
    # The driver owns shared state writes and latches the user flag.
    from .autonomous import stop_requested as shared_stop_requested
    return shared_stop_requested(ROOT, Path('/home/xzf/Code/phase0-3'))


def reference_index(reference_root):
    root = Path(reference_root).resolve()
    stage = root/'artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1'
    receipt = json.loads((stage/'source_receipt.json').read_text())
    assert receipt['sha256'] == NDP_SOURCE
    data_receipts = []
    for dataset in DATASETS:
        path = stage/f'{dataset}_data_manifest.json'
        actual = json.loads(path.read_text()); expected = load_protocol()['datasets'][dataset]['source']
        for field in ('raw_sha256', 'array_hashes', 'split_row_fingerprints', 'split_row_id_hashes',
                      'encoder_hash', 'scaler_hash', 'split_rows', 'train_domain_counts'):
            assert actual[field] == expected[field], (dataset, field)
        data_receipts.append(dict(dataset=dataset, path=str(path), sha256=sha_file(path)))
    audit = root/'research_reports/auxiliary_objective_attribution_20261115/STAGE2_FINAL_REVIEW.json'
    reviewed = json.loads(audit.read_text())
    assert reviewed['verdict'] == 'PASS' and not reviewed['blocking_issues'] and reviewed['source_hash'] == NDP_SOURCE
    entries = []
    clean = json.loads((stage/'reused_clean_index.json').read_text())
    paths = [(root/r['result_path'], r['result_sha256']) for r in clean['entries']]
    paths += [(p, sha_file(p)) for p in sorted((stage/'runs').glob('*/result.json'))]
    for path, checksum in paths:
        assert sha_file(path) == checksum
        result = json.loads(path.read_text()); j = result['job']; arm = j['config']['arm']
        assert result['epochs'] == 50 and not result['test_accessed'] and arm in ('CLEAN', 'NDP-A')
        expected = config(j['dataset'], 'CLEAN')
        for name in ('epochs', 'batch_size', 'lr', 'weight_decay', 'rho'):
            assert j['config'][name] == expected[name], (path, name)
        if arm == 'NDP-A':
            assert j['source_hash'] == NDP_SOURCE
            assert j['config']['selection_hash'] == digest(SELECTION)
            assert j['config']['data_hash'] == expected['data_hash']
            assert j['config']['alpha'] == .03 and j['config']['mixture_lambda'] == .25
        entries.append(dict(dataset=j['dataset'], arm=arm, seed=j['seed'], key=j['key'],
                            result_path=str(path), result_sha256=checksum))
    assert len(entries) == 20
    assert {(e['dataset'], e['arm'], e['seed']) for e in entries} == {
        (d, a, s) for d in DATASETS for a in ('CLEAN', 'NDP-A') for s in SEEDS}
    preflight = json.loads((ARTIFACT/'PREFLIGHT_RECEIPT.json').read_text())
    assert preflight['status'] == 'PASS'
    for name, field in (('REPORT.md', 'full_report_sha256'),):
        assert sha_file(root/'research_reports/auxiliary_objective_attribution_20261115'/name) == preflight['ndp_a_evidence'][field]
    return dict(entries=entries, trajectories=20, source_hash=NDP_SOURCE,
                selection_hash=digest(SELECTION), data_receipts=data_receipts,
                review_path=str(audit), review_sha256=sha_file(audit), test_accessed=False)


def prepare(attempt, cache_dir, reference_root, review_path):
    check_review(review_path)
    assert not stop_requested(), 'Manual stop prohibits preparation for a new dispatch'
    attempt = Path(attempt).resolve(); attempt.mkdir(parents=True, exist_ok=True)
    source = source_manifest(); immutable(attempt/'source_receipt.json', source)
    immutable(attempt/'reused_reference_index.json', reference_index(reference_root))
    immutable(attempt/'run_manifest.json', dict(source_hash=source['sha256'], design_sha256=source['design_sha256'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        review_path=str(Path(review_path).resolve()), review_sha256=sha_file(review_path),
        seeds=list(SEEDS), datasets=list(DATASETS), arms=list(ARMS), new_trajectories=50,
        reused_clean_trajectories=10, reused_ndp_trajectories=10, epochs=50, batch_size=40960,
        optimizer_updates=86250, ema_updates=86250, steps_per_epoch=dict(movielens=20, kuairand=49),
        configs={d: {a: config(d, a) for a in ARMS} for d in DATASETS},
        cache_dir=str(Path(cache_dir).resolve()), reference_root=str(Path(reference_root).resolve()),
        selection=SELECTION, selection_hash=digest(SELECTION), model_storage='RAM_ONLY',
        stop_boundary='epoch_after_validation_and_RAM_commit', test_accessed=False))
    for dataset in DATASETS:
        bundle, data = dataset_cache(dataset, cache_dir)
        assert (len(bundle.train_y)+40959)//40960 == (20 if dataset == 'movielens' else 49)
        immutable(attempt/f'{dataset}_data_manifest.json', data)
        del bundle
    return dict(status='PREPARED', source_hash=source['sha256'], new_trajectories=50, test_accessed=False)


def validate_attempt(attempt, review_path, require_smoke=False):
    current = source_manifest(); attempt = Path(attempt)
    manifest = json.loads((attempt/'run_manifest.json').read_text())
    assert manifest['source_hash'] == current['sha256'] and manifest['design_sha256'] == current['design_sha256']
    check_review(review_path or manifest['review_path'])
    if require_smoke:
        smoke = json.loads((attempt/'SMOKE_REPORT.json').read_text())
        assert smoke['status'] == 'PASS' and not smoke['test_accessed']
        assert smoke['source_hash'] == current['sha256'] and smoke['design_sha256'] == current['design_sha256']
        assert set(smoke['datasets']) == set(DATASETS)
        assert all(len(d['arms']) == 5 and d['gamma0_equivalence']['status'] == 'PASS'
                   for d in smoke['datasets'].values())
    return manifest


def results(attempt):
    attempt = Path(attempt)
    paths = sorted((attempt/'runs').glob('*/result.json'))
    index = json.loads((attempt/'reused_reference_index.json').read_text())
    for row in index['entries']:
        path = Path(row['result_path']); assert sha_file(path) == row['result_sha256']; paths.append(path)
    return [json.loads(p.read_text()) for p in paths]


def epoch_files(attempt, result):
    attempt = Path(attempt); root = attempt/'runs'/result['job']['key']
    if result['job']['config']['arm'] in ('CLEAN', 'NDP-A'):
        index = json.loads((attempt/'reused_reference_index.json').read_text())
        row = next(r for r in index['entries'] if r['key'] == result['job']['key'])
        root = Path(row['result_path']).parent
    by_epoch = {}
    for path in sorted(root.glob('attempt*/epoch_*.json')):
        value = json.loads(path.read_text()); by_epoch[value['epoch']] = (path, value)
    return by_epoch
