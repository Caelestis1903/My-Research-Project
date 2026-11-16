"""Cycle 01 identities, exact historical controls, and cycle-local RAM client."""
import ast
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
RESOURCE_1_SOURCE = '2ee2bdddb67861e65449cb3577c1d835f13395000a90b4ac101508c855f7b1a1'
RESOURCE_1_RECEIPT = 'RESOURCE_1_SUPERSESSION.json'
RESOURCE_1_POWER_SEMANTICS = 'CONSERVATIVE_OBSERVED_BOARD_PEAK_WATTS'
RESOURCE_1_ORIGINALS = {
    'source_receipt.json': '76e8ad95e7a93aa2ad4c891c5200063468978236a592ca8098679e532fb8380a',
    'run_manifest.json': '8b33bb35b9ae0694d12baf6ae5e063bb592050fd4796555ff98d032dc9c9e114',
    'SMOKE_REPORT.json': '4d50a42e5a56e0c2df2392b18ae40cec8768714f4315b63c6148efe7113521f1',
    'STORAGE_OWNER.json': '7970b5519f6f4ba6b481bd7f6f4a460e455d1e3aba0a40b5bda2ef0714e71b0d',
    'STORAGE_HANDOFF.json': '24bfa763720d61be8b0ac745fc09e2c586abd1c53d9a69aae17c2e381cb50053',
    'PARENT_STORAGE_ACCEPTANCE.json': '8494481587b37df8112e39a86b50da0f04f749ff58739b35619a91fa8d176544',
    'guardian.json': '1d842dc91fb5c9557596228eaa6864fc574e8beb078548ee0c4b1bf6d94d26c9',
}


def conservative_board_power_budgets(smoke):
    return {d: max(data['power_increment_watts'],
                   max(arm['peak_board_power_watts'] for arm in data['arms']))
            for d, data in smoke['datasets'].items()}


def resource_1_evidence(attempt):
    """Exact evidence-reuse binding for the distinct reviewer; no IPC or writes."""
    attempt = cycle_path(attempt); current = source_manifest()
    for name, expected in RESOURCE_1_ORIGINALS.items():
        assert sha_file(attempt/name) == expected, ('RESOURCE-1 original changed', name)
    original = json.loads((attempt/'source_receipt.json').read_text())
    run = json.loads((attempt/'run_manifest.json').read_text())
    smoke = json.loads((attempt/'SMOKE_REPORT.json').read_text())
    assert original['sha256'] == digest(original['files']) == run['source_hash'] == RESOURCE_1_SOURCE
    assert original['design_bundle_sha256'] == run['design_bundle_sha256'] == BUNDLE
    assert original['selection_hash'] == current['selection_hash']
    assert current['sha256'] != RESOURCE_1_SOURCE, 'RESOURCE-1 cannot downgrade to the bootstrap source'
    assert sha_file(run['review_path']) == run['review_sha256']
    assert smoke['status'] == 'PASS' and smoke['source_hash'] == RESOURCE_1_SOURCE
    assert smoke['design_bundle_sha256'] == BUNDLE and smoke['test_accessed'] is False
    old_files = {r['path']: r['sha256'] for r in original['files']}
    new_files = {r['path']: r['sha256'] for r in current['files']}
    prefix = str(HERE.relative_to(ROOT))+'/'
    changed = {prefix+'runner.py', prefix+'protocol.py', prefix+'test_recovery.py'}
    assert set(new_files) == set(old_files)
    unchanged = [r for r in original['files'] if r['path'] not in changed]
    assert all(new_files[r['path']] == r['sha256'] for r in unchanged), 'Scientific dependency changed'
    # The two mixed control/science modules retain every other original function byte-for-byte.
    functions = {}
    allowed = {'runner.py': {'require_storage', 'handoff', 'schedule', 'main'},
               'protocol.py': {'validate_attempt', 'validate_continuation'}}
    for name, permitted in allowed.items():
        rel = prefix+name; before = CYCLE/'BEFORE_SOURCE_ARCHIVE'/rel
        assert sha_file(before) == old_files[rel], 'Original source archive changed'
        before_text = before.read_text(); after_text = (ROOT/rel).read_text()
        old_functions = {n.name: ast.get_source_segment(before_text, n)
                         for n in ast.parse(before_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new_functions = {n.name: ast.get_source_segment(after_text, n)
                         for n in ast.parse(after_text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for function, text in old_functions.items():
            if function not in permitted:
                assert new_functions.get(function) == text, ('Scientific/recovery function changed', name, function)
                functions[f'{rel}:{function}'] = hashlib.sha256(text.encode()).hexdigest()
    owner = json.loads((attempt/'STORAGE_OWNER.json').read_text())
    guardian = json.loads((attempt/'guardian.json').read_text())
    assert owner['source_hash'] == RESOURCE_1_SOURCE and owner['design_bundle_sha256'] == BUNDLE
    assert owner['root'] == str(attempt) and owner['pid'] == guardian['pid'] and owner['socket'] == guardian['address']
    assert owner['endpoint_hash'] == digest([str(attempt), owner['pid'], owner['socket']])
    budgets = conservative_board_power_budgets(smoke)
    assert budgets == {'movielens': 195.61, 'kuairand': 206.31}
    return dict(issue='RESOURCE-1', approved=True, scope='ONE_PRE_FORMAL_ADMISSION_SUPERSESSION',
        original_source_hash=RESOURCE_1_SOURCE, original_receipts=RESOURCE_1_ORIGINALS,
        original_smoke_sha256=RESOURCE_1_ORIGINALS['SMOKE_REPORT.json'],
        source_hash=current['sha256'], design_bundle_sha256=BUNDLE,
        unchanged_source_files_sha256=digest(unchanged), unchanged_functions_sha256=digest(functions),
        owner=owner, power_semantics=RESOURCE_1_POWER_SEMANTICS, power_budgets_watts=budgets)


def reviewed_resource_1_receipt(attempt, review_path):
    """Build a prospective receipt; only the parent's adoption command writes it."""
    attempt = cycle_path(attempt); evidence = resource_1_evidence(attempt)
    review = check_review(review_path)
    assert review['verdict'] == 'PASS' and review.get('resource_1_evidence_reuse') == evidence, \
        'Fresh exact-source PASS must explicitly approve the complete RESOURCE-1 evidence binding'
    original = json.loads((attempt/'run_manifest.json').read_text())
    manifest = dict(original, source_hash=evidence['source_hash'],
        review_path=str(Path(review_path).resolve()), review_sha256=sha_file(review_path),
        storage_bootstrap_source_hash=RESOURCE_1_SOURCE, smoke_source_hash=RESOURCE_1_SOURCE,
        power_semantics=evidence['power_semantics'], power_budgets_watts=evidence['power_budgets_watts'])
    return dict(schema_version=1, status='REVIEWED', correction='RESOURCE-1',
        evidence_reuse=evidence, source_manifest=source_manifest(), run_manifest=manifest,
        pre_formal=dict(formal_keys=0, live_workers=[], dispatch=False, progress=False, commits=False),
        test_accessed=False)


def effective_run_manifest(attempt):
    """One exact reviewed supersession, otherwise the original strict source gate."""
    attempt = cycle_path(attempt); path = attempt/RESOURCE_1_RECEIPT
    if path.exists():
        receipt = json.loads(path.read_text())
        expected = reviewed_resource_1_receipt(attempt, receipt['run_manifest']['review_path'])
        assert receipt == expected, 'RESOURCE-1 source/review/evidence/owner binding changed'
        value = receipt['run_manifest']
    else:
        value = json.loads((attempt/'run_manifest.json').read_text())
    assert value['source_hash'] == source_manifest()['sha256'] and value['design_bundle_sha256'] == BUNDLE
    return value


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
    attempt = cycle_path(attempt); value = effective_run_manifest(attempt)
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
        assert smoke['status'] == 'PASS' and smoke['source_hash'] == value.get('smoke_source_hash', value['source_hash'])
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
    manifest = effective_run_manifest(attempt)
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
