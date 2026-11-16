"""Cycle02 identities and numeric receipts; frozen numerical helpers stay unchanged."""
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SEEDS, DATASETS, SELECTION, digest, immutable, sha_file,
    dataset_cache as source_cache, optimizer, load_protocol, make_model, setup,
    cpu_copy, state_digest, rng_state, restore_rng, evaluate, write_json)
from experiments.conditional_information_momentum_20261102_v3.guardian import request as frozen_request

ROOT = Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
CONTROL = Path('/home/xzf/Code/phase0-3')
REL = Path('artifacts/26-11-16_augmentation_gradient_momentum')
PLAN = ROOT/'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02'
CYCLE = ROOT/REL/'astra_cycles/cycle_02'
ATTEMPT = CYCLE/'attempt1'
HERE = ROOT/'experiments/natural_quadratic_ranking_20261116'
C1 = ROOT/REL/'astra_cycles/cycle_01/attempt1'
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache')
ARMS = ('NQR-025', 'BRIER-OPPORTUNITY-025', 'MEAN-GAP-025')
KINDS = ('online', 'fixed_ema')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
PACKAGE_FILES = ('__init__.py', 'core.py', 'checks.py', 'protocol.py', 'runner.py', 'tables.py', 'README.md')
DESIGN_FILES = ('ALGORITHM_DESIGNS.md', 'CODEX_EXECUTION_PLAN.md', 'CRITICAL_REVIEW.md',
                'DESIGN_SPECIFICATION.md', 'EXPERIMENT_MANIFEST.json',
                'LITERATURE_AND_NOVELTY.md', 'RESEARCH_EXPLORATION.md')
PROTECTED = '6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1'
ASTRA_SIX = '98b62580445f29e13fec3c8d58576c0b3a12b83b9e8c2b711d6acd0bd04bd086'
PYTHON = '/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python'


def read_json(path):
    return json.loads(Path(path).read_text())


def design():
    return read_json(PLAN/'EXPERIMENT_MANIFEST.json')


def design_bundle():
    files = [[name, sha_file(PLAN/name)] for name in sorted(DESIGN_FILES)]
    identity = hashlib.sha256(json.dumps(files, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    acceptance = read_json(CYCLE/'DESIGN_ACCEPTANCE.json')
    assert acceptance['verdict'] == 'COMPLETE_APPROVED_FOR_IMPLEMENTATION'
    assert acceptance['files'] == files and acceptance['design_bundle_sha256'] == identity
    handoff = read_json(CYCLE/'ASTRA_HANDOFF.json')
    six = [r for r in files if r[0] != 'DESIGN_SPECIFICATION.md']
    assert hashlib.sha256(json.dumps(six,separators=(',', ':')).encode()).hexdigest() == ASTRA_SIX
    assert handoff['research_bundle_sha256'] == ASTRA_SIX and handoff['context_closed'] is True
    return dict(files=files, sha256=identity, original_astra_six_sha256=ASTRA_SIX,
                astra_handoff_sha256=sha_file(CYCLE/'ASTRA_HANDOFF.json'))


def config(dataset, arm):
    assert dataset in DATASETS and arm in ARMS, (dataset, arm)
    p = load_protocol()['datasets'][dataset]
    d = design()['datasets'][dataset]
    return dict(cycle_id='cycle_02', dataset=dataset, arm=arm,
        objective='natural_quadratic_ranking_v1', mixture_lambda=.25, probability_margin=1.0,
        completed_risk=dict(zip(ARMS, ('Q_R', 'Q_B', 'Q_M')))[arm],
        epochs=50, batch_size=40960, precision='float32', lr=.01, betas=[.9, .999], eps=1e-8,
        weight_decay=p['weight_decay'], rho=p['rho'], steps_per_epoch=d['steps_per_epoch'],
        tail_rows=d['tail_rows'], loader_seed_offset=11000, gate_training_detached=True,
        probe_gate_full_derivative=False, algorithm_rng_draws=0,
        data_hash=digest(p['source']), selection_hash=digest(SELECTION))


def protected_source():
    recorded = read_json(C1/'RESOURCE_1_SUPERSESSION.json')['source_manifest']['files']
    actual = [dict(path=r['path'], sha256=sha_file(ROOT/r['path'])) for r in recorded]
    assert len(actual) == 79 and actual == recorded and digest(actual) == PROTECTED
    return actual


def source_manifest():
    # Byte-only closure discovery; never invoke an older source/design gate.
    frozen = protected_source()
    paths = {ROOT/r['path'] for r in frozen} | {HERE/n for n in PACKAGE_FILES} | {ROOT/'requirements-phase0-3.lock'}
    pending = list(paths)
    while pending:
        path = pending.pop()
        if path.suffix != '.py':
            continue
        parts = path.relative_to(ROOT).with_suffix('').parts
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                base = prefix + (node.module.split('.') if node.module else [])
                modules = ['.'.join(base)] + ['.'.join(base+[a.name]) for a in node.names if a.name != '*']
            for module in modules:
                bits = module.split('.')
                if bits[0] != 'experiments':
                    continue
                candidates = [ROOT.joinpath(*bits).with_suffix('.py')]
                candidates += [ROOT.joinpath(*bits[:n])/'__init__.py' for n in range(1, len(bits)+1)]
                for candidate in candidates:
                    if candidate.is_file() and candidate not in paths:
                        paths.add(candidate); pending.append(candidate)
    files = [dict(path=str(p.relative_to(ROOT)), sha256=sha_file(p)) for p in sorted(paths)]
    return dict(files=files, sha256=digest(files), design_bundle_sha256=design_bundle()['sha256'],
                selection_hash=digest(SELECTION), protected_c1_source_sha256=PROTECTED)


def job(dataset, arm, seed, source_hash):
    assert seed in SEEDS
    c = config(dataset, arm)
    value = dict(cycle_id='cycle_02', dataset=dataset, arm=arm, seed=seed, config=c,
                 config_hash=digest(c), source_hash=source_hash, design_bundle_sha256=design_bundle()['sha256'])
    value['key'] = digest(value)
    return value


def cycle_path(attempt):
    path = Path(attempt)
    assert path.is_absolute() and path == ATTEMPT and path.resolve() == ATTEMPT
    assert all(p.resolve() == p for p in (path, path/'runs', path/'jobs', path/'tables', path/'smoke'))
    return path


def local_path(attempt, path):
    attempt = cycle_path(attempt); path = Path(path)
    assert path.is_absolute() and path.is_relative_to(attempt) and path.resolve() == path
    return path


def stop_requested():
    stopped = False
    for root in (ROOT, CONTROL):
        state = read_json(root/REL/'AUTONOMOUS_STATE.json')
        stopped |= bool(state.get('manual_stop_requested') or state.get('termination_reason') == 'USER_STOPPED'
                        or (root/'STOP_AUTONOMOUS').exists() or (root/REL/'STOP_AUTONOMOUS').exists())
    return stopped


def request(attempt, command, **payload):
    return frozen_request(cycle_path(attempt), command, **payload)


def alive(pid):
    if not pid:
        return False
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def environment_identity():
    expected = design()['environment']
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
    assert sys.dont_write_bytecode and platform.python_version() == expected['python_version']
    assert torch.__version__ == expected['torch_version']
    lock = expected['package_lock_receipt']; assert sha_file(lock['path']) == lock['sha256']
    packages = {}
    for line in Path(lock['path']).read_text().splitlines():
        if '==' in line and not line.startswith(('#', '--')):
            name, version = line.split('==')
            assert importlib.metadata.version(name) == version, name
            packages[name] = version
    return dict(executable=str(Path(sys.executable).resolve()), python=platform.python_version(),
                torch=torch.__version__, lock_sha256=lock['sha256'], packages=packages)


def dataset_cache(dataset, cache_dir=DEFAULT_CACHE):
    assert Path(cache_dir).resolve() == DEFAULT_CACHE and (DEFAULT_CACHE/f'{dataset}.pkl').is_file(), 'Restore immutable cache; no reconstruction'
    bundle, data = source_cache(dataset, DEFAULT_CACHE)
    d = design()['datasets'][dataset]
    for field in ('raw_sha256', 'encoder_hash', 'scaler_hash', 'train_domain_counts'):
        assert data[field] == d[field], (dataset, field)
    for short, actual in (('array_hashes', d['array_hashes_train_validation']),
                          ('split_row_fingerprints', d['split_row_fingerprints_train_validation']),
                          ('split_row_id_hashes', d['split_row_id_hashes_train_validation'])):
        assert all(data[short][k] == v for k, v in actual.items()), (dataset, short)
    assert data['features'] == d['features']
    assert len(bundle.train_y) == d['train_rows'] and len(bundle.val_y) == d['validation_rows']
    return bundle, data


def old_inventory():
    roots = (ROOT/REL/'attempt1', C1)
    records = []
    for root in roots:
        journals = []
        for name in ('epoch_commit_ledger.jsonl', 'recovery_ledger.jsonl', 'guardian.json', 'STORAGE_OWNER.json', 'STORAGE_HANDOFF.json', 'PARENT_STORAGE_ACCEPTANCE.json'):
            p = root/name
            journals.append(dict(path=str(p), bytes=p.stat().st_size if p.exists() else None,
                                 sha256=sha_file(p) if p.exists() else None))
        results = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha_file(p))
                   for p in sorted((root/'runs').glob('*/result.json'))]
        records.append(dict(root=str(root), journals=journals, results=results))
    return dict(roots=records, protected_c1_source_sha256=digest(protected_source()),
                reused_index_sha256=sha_file(C1/'reused_reference_index.json'))


def assert_old_unchanged(attempt):
    actual = old_inventory()
    assert actual == read_json(cycle_path(attempt)/'OLD_INVENTORY_BEFORE.json'), 'Protected inventory changed: PAUSE'
    return actual


def historical_epochs(root):
    epochs = {}
    for directory in sorted(Path(root).glob('attempt*'), key=lambda p: int(p.name[7:])):
        for path in sorted(directory.glob('epoch_*.json')):
            record = read_json(path)
            assert record['epoch'] not in epochs, 'Ambiguous historical epoch requires parent correction'
            epochs[record['epoch']] = (path, record)
    assert set(epochs) == set(range(1, 51))
    return epochs


def verify_selection(result, epochs):
    assert set(epochs) == set(range(1, 51))
    for kind in KINDS:
        for _, record in epochs.values():
            v = record['validation'][kind]
            assert v['macro_auc'] is not None and all(v[c+'_auc'] is not None and v[c+'_logloss'] is not None for c in CELLS)
        selected = max(range(1, 51), key=lambda e: epochs[e][1]['validation'][kind]['macro_auc'])
        assert result['selected_epochs'][kind] == selected
        record = epochs[selected][1]
        assert result['selected_validation'][kind] == record['validation'][kind]
        assert result['selected_hashes'][kind] == record['model_hash' if kind == 'online' else 'ema_hash']


def reference_index():
    controls = design()['controls']; receipt = controls['index_receipt']
    assert sha_file(receipt['path']) == receipt['sha256']
    original = read_json(receipt['path'])
    assert original['selection_hash'] == controls['selection_hash'] == digest(SELECTION)
    assert all(design()['environment'][k] == v for k,v in original['environment'].items())
    assert len(original['initial_hashes']) == 10 and len(original['epoch_order_hashes']) == 500
    for item in original['data_receipts']:
        assert sha_file(item['path']) == item['sha256']
        expected = design()['datasets'][item['dataset']]['recorded_data_manifest']
        assert item['sha256'] == expected['sha256']
    entries = controls['entries']; subset = [r for r in original['entries'] if r['arm'] == 'CLEAN']
    assert sorted(subset, key=lambda r:r['key']) == sorted(entries, key=lambda r:r['key']) and len(entries) == 10
    assert {(r['dataset'], r['seed']) for r in entries} == {(d,s) for d in DATASETS for s in SEEDS}
    pilot = Path('/home/xzf/Code/phase0-3-20261115')
    source_path = pilot/'artifacts/heterogeneity_aware_20261115/attempt1/source_receipt.json'
    historical = read_json(source_path)
    assert historical['sha256'] == digest(historical['files']) == 'a67f446902642f0ce777ac7736474b00a1400902184ccf0d461581ace1b9a491'
    shared = [r for r in historical['files'] if r['path'].startswith('experiments/conditional_information_momentum_20261102_v3/')]
    assert len(shared) == 52 and all(sha_file(ROOT/r['path']) == r['sha256'] for r in shared)
    for name,expected in (('core.py','ec3fcfafc0e3487439b9552204fb14f1514eb18a42a13323648f00796836819d'),
                          ('protocol.py','a72a5eb9345cd9b0889cbe99541a41173e110ec892cc207274eec98757394f5a')):
        assert sha_file(pilot/'experiments/heterogeneity_aware_20261115'/name) == expected
    pilot_run = read_json(pilot/'artifacts/heterogeneity_aware_20261115/attempt1/run_manifest.json')
    assert pilot_run['source_hash'] == historical['sha256'] and pilot_run['seeds'] == list(SEEDS)
    assert pilot_run['selection'] == SELECTION and pilot_run['epochs'] == 50 and pilot_run['batch_size'] == 40960
    bound = []
    for entry in entries:
        path = Path(entry['result_path']); assert sha_file(path) == entry['result_sha256']
        r = read_json(path); j = r['job']; c = config(entry['dataset'], ARMS[0])
        assert (j['key'], j['dataset'], j['seed'], j['config']['arm']) == (entry['key'], entry['dataset'], entry['seed'], 'CLEAN')
        assert r['epochs'] == 50 and r['test_accessed'] is False
        for field in ('epochs', 'batch_size', 'lr', 'weight_decay', 'rho', 'gate_training_detached'):
            assert j['config'][field] == c[field], (path, field)
        expected = 50*c['steps_per_epoch']
        assert r['global_step'] == r['counts']['optimizer_updates'] == r['counts']['ema_updates'] == expected
        key = f'{j["dataset"]}/{j["seed"]}'
        assert r['initial_hash'] == original['initial_hashes'][key]
        epochs = historical_epochs(path.parent); verify_selection(r, epochs)
        epoch_receipts = []
        for epoch, (p, record) in sorted(epochs.items()):
            assert record['epoch_order_hash'] == original['epoch_order_hashes'][f'{key}/{epoch}']
            epoch_receipts.append(dict(epoch=epoch, path=str(p), sha256=sha_file(p)))
        bound.append(dict(entry, epochs=epoch_receipts))
    return dict(entries=bound, initial_hashes=original['initial_hashes'],
        epoch_order_hashes=original['epoch_order_hashes'], original_index_path=receipt['path'],
        original_index_sha256=receipt['sha256'], selection_hash=digest(SELECTION),
        environment=original['environment'],
        shared_operator_provenance=dict(pilot_source_receipt_path=str(source_path),pilot_source_receipt_sha256=sha_file(source_path),
            shared_conditional_files=shared,pilot_run_manifest_sha256=sha_file(pilot/'artifacts/heterogeneity_aware_20261115/attempt1/run_manifest.json')),
        test_accessed=False)


def check_review(path):
    r = read_json(path); s = source_manifest()
    assert r['verdict'] in ('PASS', 'PASS WITH FIXES') and not r.get('blocking_issues')
    assert r['cpu_checks_passed'] is True and r['source_hash'] == s['sha256']
    assert r['design_bundle_sha256'] == s['design_bundle_sha256']
    assert r['reviewer_id'] and len(set(r['coder_ids'])) >= 2 and r['reviewer_id'] not in r['coder_ids']
    return r


def prepare(attempt, cache_dir, review_path):
    attempt = cycle_path(attempt); assert not stop_requested()
    check_review(review_path)
    source = source_manifest(); controls = reference_index(); environment = environment_identity()
    attempt.mkdir(parents=True, exist_ok=True)
    immutable(attempt/'source_receipt.json', source)
    immutable(attempt/'reused_reference_index.json', controls)
    immutable(attempt/'OLD_INVENTORY_BEFORE.json', old_inventory())
    for dataset in DATASETS:
        bundle, data = dataset_cache(dataset, cache_dir)
        immutable(attempt/f'{dataset}_data_manifest.json', data); del bundle
    jobs = [job(d,a,s,source['sha256']) for s in SEEDS for d in DATASETS for a in ARMS]
    for j in jobs:
        immutable(attempt/'jobs'/f'{j["key"]}.json', j)
    manifest = dict(cycle_id='cycle_02', source_hash=source['sha256'], design_bundle=design_bundle(),
        design_bundle_sha256=source['design_bundle_sha256'], protected_c1_source_sha256=PROTECTED,
        research_receipt=design()['sourcebase']['publication_receipt'],
        review_path=str(Path(review_path).resolve()), review_sha256=sha_file(review_path),
        controls_index_hash=digest(controls), selection_hash=digest(SELECTION), environment=environment,
        configs={d:{a:config(d,a) for a in ARMS} for d in DATASETS}, jobs=jobs,
        budget=design()['budget'], cache_dir=str(DEFAULT_CACHE), model_storage='RAM_ONLY_BytesIO',
        seeds=list(SEEDS), epochs=50, batch_size=40960, precision='float32', test_accessed=False)
    immutable(attempt/'run_manifest.json', manifest)
    return dict(status='PREPARED', source_hash=source['sha256'], jobs=30, test_accessed=False)


def validate_attempt(attempt, review_path=None, require_smoke=False):
    attempt = cycle_path(attempt); m = read_json(attempt/'run_manifest.json'); s = source_manifest()
    assert m['source_hash'] == s['sha256'] and m['design_bundle_sha256'] == s['design_bundle_sha256']
    assert read_json(attempt/'source_receipt.json') == s and m['environment'] == environment_identity()
    assert m['test_accessed'] is False and m['selection_hash'] == digest(SELECTION)
    assert sha_file(m['review_path']) == m['review_sha256']; check_review(m['review_path'])
    if review_path is not None:
        check_review(review_path)
    controls = read_json(attempt/'reused_reference_index.json')
    assert digest(controls) == m['controls_index_hash']
    assert sha_file(controls['original_index_path']) == controls['original_index_sha256']
    for entry in controls['entries']:
        assert sha_file(entry['result_path']) == entry['result_sha256']
        assert all(sha_file(e['path']) == e['sha256'] for e in entry['epochs'])
    assert m['jobs'] == [job(d,a,seed,s['sha256']) for seed in SEEDS for d in DATASETS for a in ARMS]
    assert_old_unchanged(attempt)
    if require_smoke:
        smoke = read_json(attempt/'SMOKE_REPORT.json'); signoff = read_json(attempt/'SMOKE_REVIEW.json')
        assert smoke['status'] == 'PASS' and smoke['test_accessed'] is False
        assert smoke['source_hash'] == s['sha256'] and smoke['design_bundle_sha256'] == s['design_bundle_sha256']
        assert smoke['run_manifest_sha256'] == sha_file(attempt/'run_manifest.json')
        assert {(r['dataset'], r['arm']) for r in smoke['cases']} == {(d,a) for d in DATASETS for a in ARMS}
        assert len(smoke['cases']) == 6 and all(r['status'] == 'PASS' and r['recovery_verified'] for r in smoke['cases'])
        assert smoke['owner_endpoint_hash'] == read_json(attempt/'STORAGE_OWNER.json')['endpoint_hash']
        for case in smoke['cases']:
            assert case['purpose'] == 'engineering_smoke' and case['seed'] == 2022 and case['precision'] == 'float32'
            assert case['full_rows'] == 40960 and case['tail_rows'] == config(case['dataset'],case['arm'])['tail_rows']
            assert case['source_hash'] == s['sha256'] and case['design_bundle_sha256'] == s['design_bundle_sha256']
            assert case['identities'] == [['diagnostic_on_off','lambda_zero','all_single_class']]*2
            p = case['profile']
            assert p['measured'] and p['context_measured'] and p['full_process_peak_bytes'] > 0 and p['torch_reserved_peak_bytes'] > 0
            assert all(p['phases'][name]['samples'] > 0 for name in ('full_diagnostic','tail_diagnostic',
                'validation_online','validation_fixed_ema','cpu_snapshot_serialization','ram_roundtrip','recovery'))
        assert signoff['verdict'] in ('PASS', 'PASS WITH FIXES') and not signoff.get('blocking_issues')
        assert signoff['source_hash'] == s['sha256'] and signoff['design_bundle_sha256'] == s['design_bundle_sha256']
        assert signoff['smoke_sha256'] == sha_file(attempt/'SMOKE_REPORT.json') and signoff['resource_checks_passed'] is True
        code = read_json(m['review_path'])
        assert signoff['reviewer_id'] and signoff['reviewer_id'] not in code['coder_ids']
    return m


def commits(attempt, key):
    attempt = cycle_path(attempt); p = attempt/'epoch_commit_ledger.jsonl'
    rows = [r for r in map(json.loads, p.read_text().splitlines()) if r['key'] == key] if p.exists() else []
    assert [r['epoch'] for r in rows] == list(range(1, len(rows)+1))
    for r in rows:
        expected = attempt/'runs'/key/f'attempt{r["attempt"]}'/f'epoch_{r["epoch"]:02d}.json'
        assert local_path(attempt, r['epoch_file']) == expected and r['status'] == 'RAM_COMMITTED'
    return rows


def validate_epoch_state(state, record):
    assert state['job'] == record['job'] and state_digest(state) == record['continuation_hash']
    for name in ('epoch', 'global_step', 'counts', 'selected_epochs', 'peak_allocated_bytes', 'peak_reserved_bytes', 'peak_full_process_bytes'):
        assert state[name] == record[name], name
    for name in ('model', 'ema', 'optimizer', 'rng', 'loader'):
        assert state_digest(state[name]) == record[name+'_hash'], name
    assert {k:state_digest(v) for k,v in state['selected_models'].items()} == record['selected_hashes']
    if state['job'].get('purpose') != 'engineering_smoke':
        expected = state['epoch']*state['job']['config']['steps_per_epoch']
        assert state['global_step'] == expected and state['counts'] == dict(optimizer_updates=expected, ema_updates=expected)


def epoch_files(attempt, result):
    attempt = cycle_path(attempt); j = result['job']
    if j['config']['arm'] == 'CLEAN':
        index = read_json(attempt/'reused_reference_index.json')
        entry = next(e for e in index['entries'] if e['key'] == j['key'])
        pairs = [(e['epoch'], Path(e['path']), e['sha256']) for e in entry['epochs']]
    else:
        assert j == job(j['dataset'],j['arm'],j['seed'],read_json(attempt/'run_manifest.json')['source_hash'])
        pairs = []
        for ack in commits(attempt,j['key']):
            path = Path(ack['epoch_file']); receipt = read_json(path.with_name(f'ack_{ack["epoch"]:02d}.json'))
            assert receipt['key'] == j['key'] and receipt['attempt'] == ack['attempt'] and receipt['epoch'] == ack['epoch']
            assert receipt['continuation_hash'] == read_json(path)['continuation_hash']
            pairs.append((ack['epoch'], path, receipt['epoch_file_sha256']))
    epochs = {}
    for epoch, path, expected in pairs:
        assert sha_file(path) == expected
        epochs[epoch] = (path,read_json(path))
    return epochs


def results(attempt):
    attempt = cycle_path(attempt); m = validate_attempt(attempt, require_smoke=True)
    paths = [attempt/'runs'/j['key']/'result.json' for j in m['jobs']]
    paths += [Path(e['result_path']) for e in read_json(attempt/'reused_reference_index.json')['entries']]
    assert len(paths) == 40 and len(set(paths)) == 40
    records = [read_json(p) for p in paths]
    assert [r['job'] for r in records[:30]] == m['jobs']
    assert [r['job']['key'] for r in records[30:]] == [e['key'] for e in read_json(attempt/'reused_reference_index.json')['entries']]
    for r in records:
        assert r['epochs'] == 50 and r['test_accessed'] is False
        verify_selection(r, epoch_files(attempt,r))
    return records
