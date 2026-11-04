"""Frozen ten-job matrix and hashes for the implementation and its dependencies."""
from pathlib import Path
from experiments.overnight_ab_20261103.state import digest, read_json, sha_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'research_reports/2026-11-04_gaussian_domain_coverage'
ATTEMPT = ROOT / 'artifacts/gaussian_domain_coverage_20261104/attempt1'
HISTORY = Path('/home/xzf/Code/phase0-3')
CACHE = HISTORY / 'artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache'
SEEDS = tuple(range(2022, 2027))
ARMS = {'G-D0': [0], 'G-D02': [0, 2]}
FAMILY = 'GAUSSIAN_COVERAGE'
SOURCE = ROOT / 'experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2'


def source_manifest():
    paths = list(HERE.rglob('*.py'))
    history = ROOT / 'experiments/mdmt_history_information_20261103'
    paths += [p for p in history.rglob('*') if p.suffix in ('.py', '.json')]
    paths += [ROOT / 'experiments/overnight_ab_20261103' / name for name in
              ('guardian.py', 'state.py', 'selection.py', 'evaluation.py', 'data_bridge.py',
               'queue.py', 'reporting.py', 'inventory.py')]
    paths += [ROOT / 'research_plans/26-11-03/experiment_queue.yaml',
              REPORT / 'DESIGN_SPECIFICATION.md']
    files = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(set(paths))}
    return dict(files=files, sha256=digest(files))


def protocol():
    return dict(study_id='GAUSSIAN_DOMAIN_COVERAGE_20261104', protocol_id='AUC_SELECTED_V1',
        data_protocol='FULLTRAIN_NATURAL80_10_10', seeds=list(SEEDS),
        selection=dict(split='validation', metric='macro_auc', mode='strict_max_earliest_tie',
                       first_epoch=1, last_epoch=50, use_test_for_selection=False),
        decision=dict(strong_auc_gain=.003, candidate_auc_gain=.002, minimum_cell_auc_delta=-.001),
        common=dict(precision='float32', checkpoint_storage='RAM_ONLY', early_stopping=False))


def make_job(arm, seed, data_hash, source_hash):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Job is outside the frozen ten-trajectory matrix')
    conf = dict(target_domains=ARMS[arm], lambda_aug=.5, kappa=.1, rho=.995,
                lr=.01, weight_decay=1e-5, betas=[.9, .999], eps=1e-8,
                epochs=50, batch_size=40960, eval_batch_size=40960,
                foreach=False, fused=False, amsgrad=False, precision='float32')
    identity = dict(dataset='kuairand', seed=seed, arm=arm, config=conf,
                    data_hash=data_hash, source_hash=source_hash, protocol=protocol())
    return dict(job_id=f'kuairand.{arm}.s{seed}', dataset='kuairand', seed=seed, arm=arm,
        method=arm, family=FAMILY, lane='B', config=conf, config_hash=digest(conf),
        source_hash=source_hash, data_hash=data_hash, protocol=protocol(), trajectory_id=digest(identity))


def review_check():
    receipt = read_json(REPORT / 'INDEPENDENT_REVIEW_RECEIPT.json')
    if receipt['verdict'] not in ('PASS', 'PASS WITH FIXES') or receipt.get('blocking_issues'):
        raise RuntimeError('Independent engineering review has blocking issues')
    if receipt['source_hash'] != source_manifest()['sha256']:
        raise RuntimeError('Implementation differs from the reviewed source')
    return receipt
