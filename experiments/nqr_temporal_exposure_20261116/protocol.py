"""Cycle03 identities and numeric receipts; frozen numerical helpers stay unchanged."""
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
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
PLAN = ROOT/'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03'
CYCLE = ROOT/REL/'astra_cycles/cycle_03'
ATTEMPT = CYCLE/'attempt1'
HERE = ROOT/'experiments/nqr_temporal_exposure_20261116'
C2 = ROOT/REL/'astra_cycles/cycle_02/attempt1'
C1 = ROOT/REL/'astra_cycles/cycle_01/attempt1'
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache')
ARMS = ('NQR-EARLY10', 'NQR-LATE10')
KINDS = ('online', 'fixed_ema')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
PACKAGE_FILES = ('__init__.py', 'core.py', 'checks.py', 'protocol.py', 'runner.py', 'tables.py', 'README.md')
DESIGN_FILES = ('ALGORITHM_DESIGNS.md', 'CODEX_EXECUTION_PLAN.md', 'CRITICAL_REVIEW.md',
                'DESIGN_SPECIFICATION.md', 'EXPERIMENT_MANIFEST.json',
                'LITERATURE_AND_NOVELTY.md', 'RESEARCH_EXPLORATION.md')
PROTECTED = '6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1'
FROZEN_C2 = 'ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae'
PHASE_FIELDS = ('active_updates', 'off_updates', 'active_rows', 'off_rows',
                'active_task_entries', 'off_task_entries', 'lambda_update_sum', 'nonzero_opportunity_updates')
OPERATOR = 'experiments.natural_quadratic_ranking_20261116.core.train_step'
ASTRA_SIX = '58aa43e9093429bf232da9c07ff8820e952216eb151e8aa603077aa611b9c03e'
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
    early = arm == ARMS[0]
    return dict(cycle_id='cycle_03', dataset=dataset, arm=arm,
        objective='nqr_temporal_exposure_v1', source_operator=OPERATOR, inner_arm='NQR-025',
        probability_margin=1.0, completed_risk='Q_R', schedule_id='early10_v1' if early else 'late10_v1',
        active_epoch_start=1 if early else 41, active_epoch_end=10 if early else 50,
        active_lambda=.25, off_lambda=0.,
        epochs=50, batch_size=40960, precision='float32', lr=.01, betas=[.9, .999], eps=1e-8,
        weight_decay=p['weight_decay'], rho=p['rho'], steps_per_epoch=d['steps_per_epoch'],
        tail_rows=d['tail_rows'], loader_seed_offset=11000, gate_training_detached=True,
        probe_gate_full_derivative=False, algorithm_rng_draws=0,
        data_hash=digest(p['source']), selection_hash=digest(SELECTION))


def protected_source():
    base = design()['sourcebase']; receipt = base['c2_source_receipt']
    assert sha_file(receipt['path']) == receipt['sha256']
    recorded = base['c2_frozen_files']
    saved = read_json(receipt['path'])
    assert saved['files'] == recorded and saved['sha256'] == FROZEN_C2
    actual = [dict(path=r['path'], sha256=sha_file(ROOT/r['path'])) for r in recorded]
    assert len(actual) == 87 and actual == recorded and digest(actual) == FROZEN_C2
    c1 = read_json(C1/'RESOURCE_1_SUPERSESSION.json')['source_manifest']['files']
    assert len(c1) == 79 and digest(c1) == PROTECTED
    assert all(r in actual for r in c1)
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
                selection_hash=digest(SELECTION), protected_c1_source_sha256=PROTECTED,
                protected_c2_source_sha256=FROZEN_C2, frozen_c2_count=87,
                source_operator=OPERATOR, source_operator_sha256=sha_file(ROOT/'experiments/natural_quadratic_ranking_20261116/core.py'),
                new_package_files=list(PACKAGE_FILES),
                additional_files=[r for r in files if r not in frozen and not r['path'].startswith(str(HERE.relative_to(ROOT))+'/')])


def job(dataset, arm, seed, source_hash):
    assert type(seed) is int and seed in SEEDS
    c = config(dataset, arm)
    value = dict(cycle_id='cycle_03', dataset=dataset, arm=arm, seed=seed, config=c,
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
    roots = (ROOT/REL/'attempt1', C1, C2)
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
    tree_names = ('auxiliary_objective_attribution_20261115', 'conditional_information_momentum_20261102_v3',
                  'task_view_gradient_residual_momentum_20261116', 'task_axis_even_views_20261116',
                  'natural_quadratic_ranking_20261116')
    trees = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha_file(p))
             for name in tree_names for p in sorted((ROOT/'experiments'/name).rglob('*')) if p.is_file()]
    return dict(roots=records, trees=trees, protected_c2_source_sha256=digest(protected_source()),
                protected_c1_source_sha256=PROTECTED, reused_index_sha256=sha_file(C2/'reused_reference_index.json'))


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
            assert all(value is not None and math.isfinite(value) for name,value in v.items() if name.endswith(('_auc','_logloss')))
        best = -1.; chosen = None
        for e in range(1,51):
            record = epochs[e][1]
            if record['validation'][kind]['macro_auc'] > best:
                best = record['validation'][kind]['macro_auc']; chosen = e
            assert record['selected_epochs'][kind] == chosen, (e,kind,'SELECTION_PREFIX')
        selected = max(range(1, 51), key=lambda e: epochs[e][1]['validation'][kind]['macro_auc'])
        assert result['selected_epochs'][kind] == selected
        record = epochs[selected][1]
        assert result['selected_validation'][kind] == record['validation'][kind]
        assert result['selected_hashes'][kind] == record['model_hash' if kind == 'online' else 'ema_hash']


def reference_index():
    controls = design()['controls']; receipt = controls['CLEAN_index']
    assert sha_file(receipt['path']) == receipt['sha256']
    original = read_json(receipt['path'])
    assert original['selection_hash'] == controls['selection_hash'] == digest(SELECTION)
    expected_environment = design()['environment']
    for k, v in original['environment'].items():
        observed = expected_environment[k]
        if k == 'package_lock_receipt':
            assert all(observed[n] == x for n, x in v.items())
        else:
            assert observed == v, k
    assert len(original['initial_hashes']) == 10 and len(original['epoch_order_hashes']) == 500
    provenance = original['shared_operator_provenance']
    assert sha_file(provenance['pilot_source_receipt_path']) == provenance['pilot_source_receipt_sha256']
    assert all(sha_file(ROOT/r['path']) == r['sha256'] for r in provenance['shared_conditional_files'])
    pilot = Path(provenance['pilot_source_receipt_path']).parent
    assert sha_file(pilot/'run_manifest.json') == provenance['pilot_run_manifest_sha256']
    pilot_run = read_json(pilot/'run_manifest.json')
    assert pilot_run['selection'] == SELECTION and pilot_run['epochs'] == 50 and pilot_run['batch_size'] == 40960
    assert pilot_run['seeds'] == list(SEEDS)
    pilot_root = pilot.parents[2]
    for name, expected in (('core.py','ec3fcfafc0e3487439b9552204fb14f1514eb18a42a13323648f00796836819d'),
                           ('protocol.py','a72a5eb9345cd9b0889cbe99541a41173e110ec892cc207274eec98757394f5a')):
        assert sha_file(pilot_root/'experiments/heterogeneity_aware_20261115'/name) == expected
    for dataset in DATASETS:
        d = design()['datasets'][dataset]; receipt_data = d['recorded_data_manifest']
        assert sha_file(receipt_data['path']) == receipt_data['sha256']
        assert digest(load_protocol()['datasets'][dataset]['source']) == d['source_data_config_hash']
    clean = {e['key']: e for e in original['entries']}
    entries = controls['entries']
    assert len(entries) == 20
    assert {(e['dataset'], e['arm'], e['seed']) for e in entries} == {
        (d, a, seed) for d in DATASETS for a in ('CLEAN', 'NQR-025') for seed in SEEDS}
    bound = []
    for entry in entries:
        path = Path(entry['result']['path'])
        assert sha_file(path) == entry['result']['sha256'], str(path)
        r = read_json(path); j = r['job']; c = config(entry['dataset'], ARMS[0])
        assert (j['key'], j['dataset'], j['seed'], j['config']['arm']) == (entry['key'], entry['dataset'], entry['seed'], entry['arm'])
        assert r['epochs'] == 50 and r['test_accessed'] is False
        for field in ('epochs', 'batch_size', 'lr', 'weight_decay', 'rho', 'gate_training_detached'):
            assert j['config'][field] == c[field], (str(path), field)
        if 'precision' in j['config']: assert j['config']['precision'] == 'float32'
        total = 50*c['steps_per_epoch']
        assert r['global_step'] == r['counts']['optimizer_updates'] == r['counts']['ema_updates'] == total
        reference_key = f'{j["dataset"]}/{j["seed"]}'
        assert r['initial_hash'] == original['initial_hashes'][reference_key]
        if entry['arm'] == 'CLEAN':
            source = clean[entry['key']]
            assert source['result_path'] == str(path) and source['result_sha256'] == entry['result']['sha256']
            receipts = source['epochs']
        else:
            receipts = [dict(epoch=e, path=str(Path(entry['epoch_directory'])/f'epoch_{e:02d}.json'),
                             sha256=sha_file(Path(entry['epoch_directory'])/f'epoch_{e:02d}.json')) for e in range(1, 51)]
        assert len(receipts) == 50 and digest(receipts) == entry['epoch_receipts_sha256']
        epochs = {}; available = {}
        for e in receipts:
            assert sha_file(e['path']) == e['sha256'], e['path']
            record = read_json(e['path']); epoch = e['epoch']
            assert record['epoch'] == epoch and record['global_step'] == epoch*c['steps_per_epoch']
            assert record['epoch_order_hash'] == original['epoch_order_hashes'][f'{reference_key}/{epoch}']
            required = ('model_hash', 'ema_hash', 'optimizer_hash', 'epoch_order_hash', 'global_step', 'validation', 'selected_epochs')
            assert all(n in record for n in required), e['path']
            if entry['arm'] == 'NQR-025':
                assert record['test_accessed'] is False
                assert record['counts'] == dict(optimizer_updates=epoch*c['steps_per_epoch'], ema_updates=epoch*c['steps_per_epoch'])
                assert all(n in record for n in ('rng_hash', 'loader_hash', 'continuation_hash'))
            epochs[epoch] = (Path(e['path']), record)
            available[str(epoch)] = {n: n in record for n in required+('rng_hash', 'loader_hash', 'counts', 'continuation_hash', 'test_accessed')}
        verify_selection(r, epochs)
        bound.append(dict(entry, result_path=str(path), result_sha256=entry['result']['sha256'],
                          epochs=receipts, available_fields=available, reused=True))
    return dict(entries=bound, initial_hashes=original['initial_hashes'],
        epoch_order_hashes=original['epoch_order_hashes'], selection_hash=digest(SELECTION),
        environment=original['environment'], shared_operator_provenance=provenance,
        source_index_path=receipt['path'], source_index_sha256=receipt['sha256'], test_accessed=False)


def compare_prefix(record, reference):
    fields = ['model_hash', 'ema_hash', 'optimizer_hash', 'epoch_order_hash', 'global_step', 'validation']
    fields += [n for n in ('rng_hash', 'loader_hash') if n in record and n in reference]
    for name in fields:
        assert record[name] == reference[name], ('PREFIX_MISMATCH', record['job']['key'], record['epoch'], name)
    return fields


def prefix_check(attempt, record):
    j = record['job']; limit = 10 if j['arm'] == ARMS[0] else 40
    if record['epoch'] > limit:
        return None
    arm = 'NQR-025' if j['arm'] == ARMS[0] else 'CLEAN'
    index = read_json(cycle_path(attempt)/'reused_reference_index.json')
    entry = next(e for e in index['entries'] if (e['dataset'], e['seed'], e['arm']) == (j['dataset'], j['seed'], arm))
    receipt = entry['epochs'][record['epoch']-1]
    assert sha_file(receipt['path']) == receipt['sha256']
    fields = compare_prefix(record, read_json(receipt['path']))
    return dict(reference_path=receipt['path'], reference_sha256=receipt['sha256'], fields=fields, status='PASS')


def check_review(path):
    r = read_json(path); s = source_manifest()
    assert r['verdict'] in ('PASS', 'PASS WITH FIXES') and not r.get('blocking_issues')
    assert r['cpu_checks_passed'] is True and r['source_hash'] == s['sha256']
    assert r['design_bundle_sha256'] == s['design_bundle_sha256']
    assert r['reviewer_id'] and len(set(r['coder_ids'])) >= 2 and r['reviewer_id'] not in r['coder_ids']
    return r


def prepare(attempt, cache_dir, review_path):
    started = time.perf_counter()
    attempt = cycle_path(attempt); assert not stop_requested()
    reports = [CYCLE/n for n in ('CORE_IMPLEMENTATION_REPORT.md', 'RUNTIME_IMPLEMENTATION_REPORT.md')]
    assert all(p.is_file() for p in reports), 'Both coder reports must be integrated before source freeze'
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
    manifest = dict(cycle_id='cycle_03', source_hash=source['sha256'], design_bundle=design_bundle(),
        design_bundle_sha256=source['design_bundle_sha256'], protected_c1_source_sha256=PROTECTED, protected_c2_source_sha256=FROZEN_C2,
        model=design()['model'], loader_rng=design()['loader_rng'], selection=SELECTION,
        research_receipt=design()['sourcebase']['publication_receipt'],
        review_path=str(Path(review_path).resolve()), review_sha256=sha_file(review_path),
        coder_reports=[dict(path=str(p),sha256=sha_file(p)) for p in reports],
        controls_index_hash=digest(controls), selection_hash=digest(SELECTION), environment=environment,
        configs={d:{a:config(d,a) for a in ARMS} for d in DATASETS}, jobs=jobs,
        budget=design()['budget'], cache_dir=str(DEFAULT_CACHE), model_storage='RAM_ONLY_BytesIO',
        seeds=list(SEEDS), epochs=50, batch_size=40960, precision='float32', test_accessed=False)
    immutable(attempt/'run_manifest.json', manifest)
    if not (attempt/'PREPARATION_COST.json').exists():
        immutable(attempt/'PREPARATION_COST.json', dict(seconds=time.perf_counter()-started, test_accessed=False))
    return dict(status='PREPARED', source_hash=source['sha256'], jobs=20, test_accessed=False)


def validate_attempt(attempt, review_path=None, require_smoke=False):
    attempt = cycle_path(attempt); m = read_json(attempt/'run_manifest.json'); s = source_manifest()
    assert m['source_hash'] == s['sha256'] and m['design_bundle_sha256'] == s['design_bundle_sha256']
    assert read_json(attempt/'source_receipt.json') == s and m['environment'] == environment_identity()
    assert m['test_accessed'] is False and m['selection_hash'] == digest(SELECTION)
    assert sha_file(m['review_path']) == m['review_sha256']; check_review(m['review_path'])
    assert all(sha_file(p['path']) == p['sha256'] for p in m['coder_reports'])
    if review_path is not None:
        check_review(review_path)
    controls = read_json(attempt/'reused_reference_index.json')
    assert digest(controls) == m['controls_index_hash']
    assert sha_file(controls['source_index_path']) == controls['source_index_sha256']
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
        assert len(smoke['cases']) == 4
        assert {(r['dataset'], r['arm']) for r in smoke['cases']} == {(d, a) for d in DATASETS for a in ARMS}
        assert len({r['device'] for r in smoke['cases']}) == 2
        for dataset in DATASETS:
            assert len({r['device'] for r in smoke['cases'] if r['dataset'] == dataset}) == 2
        assert smoke['owner_endpoint_hash'] == read_json(attempt/'STORAGE_OWNER.json')['endpoint_hash']
        for case in smoke['cases']:
            assert case['status'] == 'PASS' and case['recovery_verified'] and case['source_equivalence_verified']
            assert case['purpose'] == 'engineering_smoke' and case['clock_kind'] == 'engineering_virtual'
            assert case['seed'] == 2022 and case['precision'] == 'float32'
            assert case['full_rows'] == 40960 and case['tail_rows'] == config(case['dataset'], case['arm'])['tail_rows']
            assert case['source_hash'] == s['sha256'] and case['design_bundle_sha256'] == s['design_bundle_sha256']
            assert case['physical_train_calls'] == 10 and case['engineering_steps'] == 4 and case['engineering_commits'] == 2
            assert case['scientific_epochs_completed'] == 0 and case['effective_lambdas'] == [0., .25]
            p = case['profile']
            assert p['measured'] and p['context_measured'] and p['full_process_peak_bytes'] > 0 and p['torch_reserved_peak_bytes'] > 0
            for name in ('on_full_diagnostic', 'on_tail_diagnostic', 'off_full_diagnostic', 'off_tail_diagnostic',
                         'validation_online', 'validation_fixed_ema', 'cpu_snapshot_serialization', 'ram_roundtrip', 'recovery'):
                assert p['phases'][name]['samples'] > 0, name
        assert signoff['verdict'] in ('PASS', 'PASS WITH FIXES') and not signoff.get('blocking_issues')
        assert signoff['source_hash'] == s['sha256'] and signoff['design_bundle_sha256'] == s['design_bundle_sha256']
        assert signoff['smoke_sha256'] == sha_file(attempt/'SMOKE_REPORT.json') and signoff['resource_checks_passed'] is True
        assert signoff['cross_arm_device_compatibility_passed'] is True
        assert signoff['reviewer_id'] and signoff['reviewer_id'] not in read_json(m['review_path'])['coder_ids']
    return m


def commits(attempt, key):
    attempt = cycle_path(attempt); p = attempt/'epoch_commit_ledger.jsonl'
    rows = [r for r in map(json.loads, p.read_text().splitlines()) if r['key'] == key] if p.exists() else []
    assert [r['epoch'] for r in rows] == list(range(1, len(rows)+1))
    for r in rows:
        expected = attempt/'runs'/key/f'attempt{r["attempt"]}'/f'epoch_{r["epoch"]:02d}.json'
        assert local_path(attempt, r['epoch_file']) == expected and r['status'] == 'RAM_COMMITTED'
    return rows


def empty_phase_counters():
    return {n: 0. if n == 'lambda_update_sum' else 0 for n in PHASE_FIELDS}


def validate_formal_clock(j, epoch, batch, global_step, counts, phase_counters):
    from .core import schedule
    c = j['config']; d = design()['datasets'][j['dataset']]
    assert type(epoch) is int and 1 <= epoch <= 50
    assert type(batch) is int and 0 <= batch <= c['steps_per_epoch']
    assert type(global_step) is int and global_step == (epoch-1)*c['steps_per_epoch']+batch
    assert set(counts) == {'optimizer_updates', 'ema_updates'}
    assert all(type(v) is int and v == global_step for v in counts.values())
    assert set(phase_counters) == set(PHASE_FIELDS)
    prior = sum(schedule(j['arm'], e)['scheduled_active'] for e in range(1, epoch))
    active = schedule(j['arm'], epoch)['scheduled_active']
    updates = c['steps_per_epoch']*prior+batch*active
    rows = d['train_rows']*prior+min(c['batch_size']*batch, d['train_rows'])*active
    total_rows = (epoch-1)*d['train_rows']+min(c['batch_size']*batch, d['train_rows'])
    expected = dict(active_updates=updates, off_updates=global_step-updates, active_rows=rows,
        off_rows=total_rows-rows, active_task_entries=2*rows, off_task_entries=2*(total_rows-rows),
        lambda_update_sum=.25*updates)
    for n, v in expected.items():
        assert phase_counters[n] == v and (type(phase_counters[n]) is int if n != 'lambda_update_sum' else type(phase_counters[n]) in (float, int)), n
    opportunity = phase_counters['nonzero_opportunity_updates']
    assert type(opportunity) is int and 0 <= opportunity <= updates


def validate_epoch_state(state, record):
    from .core import schedule
    assert state['job'] == record['job'] and state_digest(state) == record['continuation_hash']
    fields = ('epoch', 'global_step', 'counts', 'selected_epochs', 'peak_allocated_bytes', 'peak_reserved_bytes',
              'peak_full_process_bytes', 'clock_kind', 'next_epoch', 'schedule_epoch', 'schedule', 'next_schedule',
              'phase_counters', 'adam_step_counters', 'last_parameter_activity')
    for name in fields:
        assert state[name] == record[name], name
    for name in ('model', 'ema', 'optimizer', 'rng', 'loader'):
        assert state_digest(state[name]) == record[name+'_hash'], name
    assert {k: state_digest(v) for k, v in state['selected_models'].items()} == record['selected_hashes']
    j = state['job']; purpose = j.get('purpose', 'formal'); c = j['config']
    assert j['config'] == config(j['dataset'], j['arm'])
    assert j['config_hash'] == digest(c) and type(j['seed']) is int and j['seed'] in SEEDS
    assert set(state['counts']) == {'optimizer_updates', 'ema_updates'}
    assert set(state['phase_counters']) == set(PHASE_FIELDS)
    assert all(type(v) is bool for v in state['last_parameter_activity'].values())
    assert list(state['last_parameter_activity']) == list(state['adam_step_counters'])
    parameter_ids = [i for group in state['optimizer']['param_groups'] for i in group['params']]
    assert len(parameter_ids) == len(state['adam_step_counters'])
    for name, pid in zip(state['adam_step_counters'], parameter_ids):
        saved = state['optimizer']['state'].get(pid, {})
        step = int(saved['step'].item()) if 'step' in saved else None
        assert step == state['adam_step_counters'][name], name
        assert step is None or 0 <= step <= state['global_step']
    if purpose == 'formal':
        assert state['clock_kind'] == 'formal'
        e = state['epoch']; validate_formal_clock(j, e, c['steps_per_epoch'], state['global_step'], state['counts'], state['phase_counters'])
        assert state['schedule_epoch'] == e and state['next_epoch'] == e+1
        assert state['schedule'] == dict(schedule(j['arm'], e), clock_kind='formal')
        assert state['next_schedule'] == (dict(schedule(j['arm'], e+1), clock_kind='formal') if e < 50 else None)
        assert set(state['selected_epochs']) == set(KINDS)
        for kind, selected in state['selected_epochs'].items():
            assert type(selected) is int and 1 <= selected <= e
    else:
        assert purpose in ('engineering_check', 'engineering_smoke') and state['clock_kind'] == 'engineering_virtual'
        assert j['fixture_id'] == 'temporal_populated_boundary_v1'
        commit = state['epoch']; assert type(commit) is int and commit in (1, 2)
        steps = 2*commit; assert state['global_step'] == state['engineering_steps'] == steps
        assert state['engineering_commits'] == commit and state['scientific_epochs_completed'] == 0
        assert state['counts'] == dict(optimizer_updates=steps, ema_updates=steps)
        before = 10 if j['arm'] == ARMS[0] else 40; virtual = before+commit-1
        assert state['schedule_epoch'] == virtual and state['next_epoch'] is None
        assert state['schedule'] == dict(schedule(j['arm'], virtual), clock_kind='engineering_virtual')
        assert state['next_schedule'] == (dict(schedule(j['arm'], before+1), clock_kind='engineering_virtual') if commit == 1 else None)
        assert state['sample_epoch'] == 1 and state['sample_batches'] == [1, c['steps_per_epoch']]
        p = state['phase_counters']; active_steps = (2 if j['arm'] == ARMS[0] or commit == 2 else 0)
        active_rows = (40960+c['tail_rows']) if active_steps else 0
        assert p['active_updates'] == active_steps and p['off_updates'] == steps-active_steps
        assert p['active_rows'] == active_rows and p['off_rows'] == commit*(40960+c['tail_rows'])-active_rows
        assert p['active_task_entries'] == 2*p['active_rows'] and p['off_task_entries'] == 2*p['off_rows']
        assert p['lambda_update_sum'] == .25*active_steps and 0 <= p['nonzero_opportunity_updates'] <= active_steps
        assert not state['selected_models'] and not state['selected_epochs'] and not state['selected_validation']
        for n in ('engineering_steps', 'engineering_commits', 'scientific_epochs_completed', 'sample_epoch', 'sample_batches'):
            assert state[n] == record[n], n


def epoch_files(attempt, result):
    attempt = cycle_path(attempt); j = result['job']
    if j['config']['arm'] in ('CLEAN', 'NQR-025'):
        index = read_json(attempt/'reused_reference_index.json')
        entry = next(e for e in index['entries'] if e['key'] == j['key'])
        pairs = [(e['epoch'], Path(e['path']), e['sha256']) for e in entry['epochs']]
    else:
        assert j == job(j['dataset'],j['arm'],j['seed'],read_json(attempt/'run_manifest.json')['source_hash'])
        pairs = []
        for ack in commits(attempt,j['key']):
            path = Path(ack['epoch_file']); receipt = read_json(path.with_name(f'ack_{ack["epoch"]:02d}.json'))
            assert receipt['key'] == j['key'] and receipt['attempt'] == ack['attempt'] and receipt['epoch'] == ack['epoch']
            assert receipt['source_hash'] == j['source_hash'] and receipt['design_bundle_sha256'] == j['design_bundle_sha256']
            assert receipt['owner_endpoint_hash'] == read_json(attempt/'STORAGE_OWNER.json')['endpoint_hash']
            assert receipt['continuation_hash'] == read_json(path)['continuation_hash']
            pairs.append((ack['epoch'], path, receipt['epoch_file_sha256']))
    epochs = {}
    for epoch, path, expected in pairs:
        assert sha_file(path) == expected
        record = read_json(path)
        if j['config']['arm'] in ARMS:
            assert record['job'] == j and record['epoch'] == epoch and record['clock_kind'] == 'formal'
            assert record['source_hash'] == j['source_hash'] and record['design_bundle_sha256'] == j['design_bundle_sha256']
            validate_formal_clock(j,epoch,j['config']['steps_per_epoch'],record['global_step'],record['counts'],record['phase_counters'])
        epochs[epoch] = (path,record)
    return epochs


def results(attempt):
    attempt = cycle_path(attempt); m = validate_attempt(attempt, require_smoke=True)
    paths = [attempt/'runs'/j['key']/'result.json' for j in m['jobs']]
    paths += [Path(e['result_path']) for e in read_json(attempt/'reused_reference_index.json')['entries']]
    assert len(paths) == 40 and len(set(paths)) == 40
    records = [read_json(p) for p in paths]
    assert [r['job'] for r in records[:20]] == m['jobs']
    assert [r['job']['key'] for r in records[20:]] == [e['key'] for e in read_json(attempt/'reused_reference_index.json')['entries']]
    for r in records:
        assert r['epochs'] == 50 and r['test_accessed'] is False
        epochs=epoch_files(attempt,r); verify_selection(r, epochs)
        if r['job']['config']['arm'] in ARMS:
            for name in ('global_step','counts','clock_kind','next_epoch','schedule_epoch','schedule','next_schedule',
                         'phase_counters','adam_step_counters','last_parameter_activity'):
                assert r[name] == epochs[50][1][name], name
    return records
