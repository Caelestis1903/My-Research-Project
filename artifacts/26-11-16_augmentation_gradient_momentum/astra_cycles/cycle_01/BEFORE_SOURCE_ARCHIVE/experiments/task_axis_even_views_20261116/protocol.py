"""Cycle 01 identities, exact historical controls, and cycle-local RAM client."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SEEDS, DATASETS, SELECTION, digest, immutable, sha_file, dataset_cache as source_cache,
    optimizer, load_protocol, make_model, setup, cpu_copy, state_digest, rng_state,
    restore_rng, evaluate, write_json)
from experiments.conditional_information_momentum_20261102_v3.guardian import request as frozen_request, alive

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTROL = Path('/home/xzf/Code/phase0-3')
REL = Path('artifacts/26-11-16_augmentation_gradient_momentum')
CYCLE = ROOT/REL/'astra_cycles/cycle_01'
PLAN = ROOT/'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01'
ATTEMPT = CYCLE/'attempt1'
OLD = ROOT/REL/'attempt1'
REFERENCE = Path('/home/xzf/Code/phase0-3_auxiliary_20261115')
NDP = REFERENCE/'artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1'
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache')
ARMS = ('AXIS-EVEN', 'AXIS-IID2', 'ISOTROPIC-EVEN')
BUNDLE = 'f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754'
DESIGN_FILES = ('ALGORITHM_DESIGNS.md', 'CODEX_EXECUTION_PLAN.md', 'CRITICAL_REVIEW.md',
                'DESIGN_COMPLETENESS_ADDENDUM.md', 'EXPERIMENT_MANIFEST.json',
                'LITERATURE_AND_NOVELTY.md', 'RESEARCH_EXPLORATION.md')


def design():
    pairs = [[n, sha_file(PLAN/n)] for n in sorted(DESIGN_FILES)]
    actual = hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()
    assert actual == BUNDLE, 'Approved seven-file design changed'
    approval = json.loads((CYCLE/'DESIGN_APPROVAL.json').read_text())
    assert approval['verdict'] == 'PASS' and approval['design_bundle_sha256'] == actual
    return json.loads((PLAN/'EXPERIMENT_MANIFEST.json').read_text())


def cycle_path(attempt):
    path = Path(attempt).absolute()
    assert path == ATTEMPT and path.resolve() == ATTEMPT, 'Only isolated cycle_01/attempt1 is authorized'
    for name in ('runs', 'jobs', 'tables'):
        assert (path/name).resolve() == path/name, 'Output directory aliases are prohibited'
    return path


def stop_requested():
    for root in (ROOT, CONTROL):
        state = json.loads((root/REL/'AUTONOMOUS_STATE.json').read_text())
        if any((root/p).exists() for p in (Path('STOP_AUTONOMOUS'), REL/'STOP_AUTONOMOUS')):
            return True
        if state.get('manual_stop_requested') or state.get('termination_reason') == 'USER_STOPPED':
            return True
    return False


def request(attempt, command, **payload):
    # The frozen client is always rooted here, including recover/release/complete.
    return frozen_request(cycle_path(attempt), command, **payload)


def source_manifest():
    """Called by reviewer/prepare only after both coder reports are complete."""
    design()
    paths = list(HERE.glob('*.py')) + [HERE/'README.md', ROOT/'requirements-phase0-3.lock']
    aux = ROOT/'experiments/auxiliary_objective_attribution_20261115'
    paths += list(aux.glob('*.py')) + list((aux/'stage2').glob('*.py'))
    source = ROOT/'experiments/conditional_information_momentum_20261102_v3'
    paths += [p for p in source.rglob('*') if p.suffix in ('.py', '.json')]
    tv = ROOT/'experiments/task_view_gradient_residual_momentum_20261116'
    paths += [tv/n for n in ('__init__.py', 'core.py', 'protocol.py', 'runner.py', 'autonomous.py')]
    assert all((HERE/n).is_file() for n in ('__init__.py', 'core.py', 'test_core.py', 'runner.py', 'tables.py'))
    files = [dict(path=str(p.relative_to(ROOT)), sha256=sha_file(p)) for p in sorted(set(paths))]
    return dict(files=files, sha256=digest(files), design_bundle_sha256=BUNDLE,
                selection_hash=digest(SELECTION))


def config(dataset, arm):
    assert dataset in DATASETS and arm in ARMS
    value = dict(design()['configs'][dataset], dataset=dataset, arm=arm)
    value.update(rho=value['ema_rho'], data_hash=value['source_data_config_hash'],
                 rng_version='axis_even_v1', loader_seed_offset=11000)
    assert value['data_hash'] == digest(load_protocol()['datasets'][dataset]['source'])
    return value


def job(dataset, arm, seed, source_hash):
    assert seed in SEEDS
    c = config(dataset, arm)
    value = dict(cycle_id=design()['cycle_id'], dataset=dataset, arm=arm, seed=seed,
                 config=c, config_hash=digest(c), source_hash=source_hash, design_bundle_sha256=BUNDLE)
    value['key'] = digest(value)
    return value


def check_review(path):
    value = json.loads(Path(path).read_text()); source = source_manifest()
    assert value['verdict'] in ('PASS', 'PASS WITH FIXES') and not value.get('blocking_issues')
    assert value['source_hash'] == source['sha256'], 'Current source is not independently reviewed'
    assert value['design_bundle_sha256'] == BUNDLE and value['cpu_checks_passed'] is True
    return value


def dataset_cache(dataset, cache_dir=DEFAULT_CACHE):
    assert (Path(cache_dir)/f'{dataset}.pkl').is_file(), 'Restore the verified source cache; no reconstruction'
    bundle, manifest = source_cache(dataset, cache_dir)
    expected = design()['datasets'][dataset]['identity']
    for field in ('raw_sha256', 'array_hashes', 'split_row_fingerprints', 'split_row_id_hashes',
                  'encoder_hash', 'scaler_hash', 'split_rows', 'train_domain_counts'):
        assert manifest[field] == expected[field], (dataset, field)
    return bundle, manifest


def old_inventory():
    """Whole journals and complete result collections, never prefix-only protection."""
    inventories = []
    for root in (OLD, NDP):
        journal = root/'epoch_commit_ledger.jsonl'
        inventories.append(dict(root=str(root), journal=dict(path=str(journal),
            bytes=journal.stat().st_size if journal.exists() else None,
            sha256=sha_file(journal) if journal.exists() else None), results=[
                [str(p.relative_to(root)), sha_file(p)] for p in sorted((root/'runs').glob('*/result.json'))]))
    index = OLD/'reused_reference_index.json'
    return dict(roots=inventories, controls_index_sha256=sha_file(index))


def assert_old_unchanged(attempt):
    before = json.loads((cycle_path(attempt)/'OLD_INVENTORY_BEFORE.json').read_text())
    actual = old_inventory()
    assert actual == before, 'Protected old journal/result collection changed: PAUSE'
    return actual


def reference_index():
    manifest = design(); controls = manifest['controls']; receipt = controls['index_receipt']
    assert sha_file(receipt['path']) == receipt['sha256']
    index = json.loads(Path(receipt['path']).read_text())
    assert index['entries'] == controls['entries'] and index['selection_hash'] == digest(SELECTION)
    # Includes original reviews, source/config/data receipts and the null-branch proof.
    for item in manifest['ndp_handoff_evidence']['files']:
        assert sha_file(item['path']) == item['sha256'], item['path']
    historical = json.loads((NDP/'source_receipt.json').read_text())
    assert historical['sha256'] == controls['ndp_source_hash']
    for item in historical['files']:
        base = ROOT if Path(item['path']).parts[0] == 'experiments' else REFERENCE
        assert sha_file(base/item['path']) == item['sha256'], ('Historical dependency differs', item['path'])
    proof = {}; orders = {}; seen = set()
    for entry in index['entries']:
        path = Path(entry['result_path']); assert sha_file(path) == entry['result_sha256']
        r = json.loads(path.read_text()); j = r['job']; dataset = j['dataset']; arm = j['config']['arm']
        assert (dataset, arm, j['seed']) == (entry['dataset'], entry['arm'], entry['seed'])
        seen.add((dataset, arm, j['seed'])); c = config(dataset, 'AXIS-EVEN')
        assert r['epochs'] == 50 and r['test_accessed'] is False
        for field in ('epochs', 'batch_size', 'lr', 'weight_decay', 'rho', 'gate_training_detached', 'probe_gate_full_derivative'):
            assert j['config'][field] == c[field], (path, field)
        assert r['global_step'] == r['counts']['optimizer_updates'] == r['counts']['ema_updates'] == 50*c['steps_per_epoch']
        if arm == 'NDP-A':
            assert j['source_hash'] == controls['ndp_source_hash']
            assert j['config']['data_hash'] == c['data_hash'] and j['config']['selection_hash'] == digest(SELECTION)
            assert j['config']['alpha'] == .03 and j['config']['mixture_lambda'] == .25
        key = f'{dataset}/{j["seed"]}'
        if key in proof:
            assert proof[key] == r['initial_hash'], 'Historical CLEAN/NDP initial states differ'
        proof[key] = r['initial_hash']
        epochs = epoch_files_for_root(path.parent)
        assert set(epochs) == set(range(1, 51))
        for epoch, (_, data) in epochs.items():
            order_key = f'{key}/{epoch}'
            if order_key in orders:
                assert orders[order_key] == data['epoch_order_hash']
            orders[order_key] = data['epoch_order_hash']
    assert seen == {(d, a, s) for d in DATASETS for a in ('CLEAN', 'NDP-A') for s in SEEDS}
    return dict(index, initial_hashes=proof, epoch_order_hashes=orders, index_sha256=receipt['sha256'],
                dependency_identity_verified=True, environment=manifest['environment'])


def environment_identity():
    expected = design()['environment']
    assert Path(sys.executable).resolve() == Path(expected['python_executable']).resolve()
    assert platform.python_version() == expected['python_version'] and torch.__version__ == expected['torch_version']
    lock = expected['package_lock_receipt']; assert sha_file(lock['path']) == lock['sha256']
    packages = {}
    for line in Path(lock['path']).read_text().splitlines():
        if '==' in line and not line.startswith(('#', '--')):
            name, version = line.split('=='); actual = importlib.metadata.version(name)
            assert actual == version, ('Installed package differs from lock', name, actual, version)
            packages[name] = actual
    return dict(python=platform.python_version(), torch=torch.__version__, packages=packages,
                lock_sha256=lock['sha256'], executable=str(Path(sys.executable).resolve()))


def prepare(attempt, cache_dir, review_path):
    assert not stop_requested(), 'Manual stop prohibits preparation'
    check_review(review_path); attempt = cycle_path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    environment = environment_identity()
    source = source_manifest(); controls = reference_index()
    immutable(attempt/'source_receipt.json', source)
    immutable(attempt/'reused_reference_index.json', controls)
    immutable(attempt/'OLD_INVENTORY_BEFORE.json', old_inventory())
    for dataset in DATASETS:
        bundle, data = dataset_cache(dataset, cache_dir)
        assert (len(bundle.train_y)+40959)//40960 == config(dataset, ARMS[0])['steps_per_epoch']
        immutable(attempt/f'{dataset}_data_manifest.json', data); del bundle
    immutable(attempt/'run_manifest.json', dict(source_hash=source['sha256'], design_bundle_sha256=BUNDLE,
        review_path=str(Path(review_path).resolve()), review_sha256=sha_file(review_path),
        controls_index_hash=digest(controls), cache_dir=str(Path(cache_dir).resolve()),
        seeds=list(SEEDS), datasets=list(DATASETS), arms=list(ARMS), new_trajectories=30,
        reused_trajectories=20, epochs=50, batch_size=40960, precision='float32',
        optimizer_updates=51750, ema_updates=51750, selection=SELECTION,
        environment=environment, model_storage='RAM_ONLY_BytesIO', test_accessed=False))
    return dict(status='PREPARED', source_hash=source['sha256'], design_bundle_sha256=BUNDLE, test_accessed=False)


def validate_attempt(attempt, review_path=None, require_smoke=False):
    attempt = cycle_path(attempt); value = json.loads((attempt/'run_manifest.json').read_text())
    assert value['source_hash'] == source_manifest()['sha256'] and value['design_bundle_sha256'] == BUNDLE
    assert value['environment'] == environment_identity()
    assert sha_file(value['review_path']) == value['review_sha256']
    check_review(review_path or value['review_path'])
    index = json.loads((attempt/'reused_reference_index.json').read_text())
    assert digest(index) == value['controls_index_hash']
    for entry in index['entries']:
        assert sha_file(entry['result_path']) == entry['result_sha256']
    assert_old_unchanged(attempt)
    if require_smoke:
        smoke = json.loads((attempt/'SMOKE_REPORT.json').read_text())
        assert smoke['status'] == 'PASS' and smoke['source_hash'] == value['source_hash']
        assert smoke['design_bundle_sha256'] == BUNDLE and smoke['test_accessed'] is False
        assert set(smoke['datasets']) == set(DATASETS)
        for dataset, data in smoke['datasets'].items():
            assert {a['arm'] for a in data['arms']} == set(ARMS)
            assert all(a['rows'] == 40960 and a['precision'] == 'float32' for a in data['arms'])
            assert data['identities']['status'] == 'PASS'
            assert data['identities']['tail_rows'] == design()['datasets'][dataset]['identity']['split_rows']['train'] % 40960
            assert data['peak_reserved_bytes'] > 0 and data['host_admission_bytes'] > 0
    return value


def validate_epoch_state(state, record):
    """Bind the complete CPU continuation to its numerical epoch receipt."""
    assert state['job']['source_hash'] == record['source_hash']
    assert state['job']['design_bundle_sha256'] == record['design_bundle_sha256'] == BUNDLE
    assert 1 <= state['epoch'] <= 50
    expected = state['epoch']*state['job']['config']['steps_per_epoch']
    assert state['global_step'] == expected
    assert state['counts'] == dict(optimizer_updates=expected, ema_updates=expected)
    for name in ('epoch', 'global_step', 'counts', 'selected_epochs', 'peak_allocated_bytes', 'peak_reserved_bytes'):
        assert state[name] == record[name], f'Continuation {name} differs from ACK receipt'
    for name in ('model', 'ema', 'optimizer', 'rng', 'loader'):
        assert state_digest(state[name]) == record[f'{name}_hash'], f'Continuation {name} hash differs'
    assert {k: state_digest(v) for k, v in state['selected_models'].items()} == record['selected_hashes']
    assert state_digest(state) == record['continuation_hash'], 'Full continuation fingerprint differs'


def validate_continuation(attempt, key, state, owner_record):
    """Use only the owner's last acknowledged epoch, never an uncommitted glob."""
    attempt = cycle_path(attempt); j = owner_record['job']
    manifest = json.loads((attempt/'run_manifest.json').read_text())
    assert j == job(j['dataset'], j['config']['arm'], j['seed'], manifest['source_hash']) and j['key'] == key
    ledger = attempt/'epoch_commit_ledger.jsonl'
    commits = [r for r in map(json.loads, ledger.read_text().splitlines())
               if r['key'] == key and r['status'] == 'RAM_COMMITTED'] if ledger.exists() else []
    if owner_record['epoch'] == 0:
        assert state is None and not commits, 'Unacknowledged continuation state'
        return
    assert state is not None and state['job'] == j and state['epoch'] == owner_record['epoch']
    assert state['counts'] == owner_record['counts'], 'Continuation counters differ from RAM owner'
    assert commits and commits[-1]['epoch'] == owner_record['epoch'], 'Missing matching RAM ACK'
    ack = commits[-1]
    expected = attempt/'runs'/key/f'attempt{ack["attempt"]}'/f'epoch_{ack["epoch"]:02d}.json'
    path = Path(ack['epoch_file'])
    assert path == expected and path.resolve() == expected, 'ACK epoch file is not cycle-local'
    validate_epoch_state(state, json.loads(path.read_text()))


def epoch_files_for_root(root):
    by_epoch = {}
    # Recoveries may leave uncommitted numeric files; later attempts replace earlier ones.
    for directory in sorted(Path(root).glob('attempt*'), key=lambda p: int(p.name[7:])):
        for path in sorted(directory.glob('epoch_*.json')):
            data = json.loads(path.read_text()); by_epoch[data['epoch']] = (path, data)
    return by_epoch


def epoch_files(attempt, result):
    attempt = cycle_path(attempt); root = attempt/'runs'/result['job']['key']
    if result['job']['config']['arm'] in ('CLEAN', 'NDP-A'):
        index = json.loads((attempt/'reused_reference_index.json').read_text())
        row = next(r for r in index['entries'] if r['key'] == result['job']['key'])
        root = Path(row['result_path']).parent
    return epoch_files_for_root(root)


def results(attempt):
    attempt = cycle_path(attempt); validate_attempt(attempt, require_smoke=True)
    paths = sorted((attempt/'runs').glob('*/result.json')); assert len(paths) == 30
    index = json.loads((attempt/'reused_reference_index.json').read_text())
    paths += [Path(e['result_path']) for e in index['entries']]
    return [json.loads(p.read_text()) for p in paths]
