"""Review-gated shared dispatch, RAM-only endpoints and seed-level exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.mdmt_history_information_20261103 import run as history
from experiments.conditional_information_momentum_20261102_v3 import run as pair
from experiments.mdmt_history_information_20261103.reporting import stats, write_csv

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'research_plans/26-11-03_v2'
REPORT = ROOT / 'research_reports/2026-11-03_mdmt_five_seed_completion'
ATTEMPT = ROOT / 'artifacts/mdmt_five_seed_completion_20261103/attempt1'
BASE = ROOT / 'artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720'
FAMILIES = {'history': history, 'pair': pair}
SEEDS = set(range(2022, 2027))
ARMS = ('C_MIX', 'H_MIX', 'P01', 'M')
KINDS = ('online', 'fixed_ema')
CHECKPOINTS = ('selected', 'E10', 'E50')
write_json = history.write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def source_manifest():
    paths = sorted(HERE.rglob('*.py')) + [PLAN / name for name in (
        'DESIGN_SPECIFICATION.md', 'PHASE1_MATRIX.csv', 'DESIGN_REUSE_AUDIT.json')]
    files = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    families = {name: module.source_manifest()['sha256'] for name, module in FAMILIES.items()}
    return dict(files=files, families=families, sha256=history.digest(dict(files=files, families=families)))


def matrix():
    rows = read_csv(PLAN / 'PHASE1_MATRIX.csv')
    assert len(rows) == 40
    assert sum(r['action'] == 'RUN' for r in rows) == 30
    assert sum(r['action'] == 'REUSE' for r in rows) == 10
    identities = [(r['dataset'], r['arm'], int(r['seed'])) for r in rows]
    assert len(set(identities)) == 40
    for dataset in ('movielens', 'kuairand'):
        for arm in ARMS:
            assert {s for d, a, s in identities if (d, a) == (dataset, arm)} == SEEDS
    return rows


def native_job(row):
    module = FAMILIES[row['family']]
    args = (row['dataset'], row['arm'], int(row['seed']), 'FIVE_SEED')
    job = module.job(*args, lambda_pair=.05) if row['family'] == 'pair' else module.job(*args)
    assert job['source_hash'] == row['source_hash']
    assert job['config']['epochs'] == 50 and job['config']['batch_size'] == 40960
    return job


def prepare(attempt):
    attempt = Path(attempt)
    rows = matrix()
    jobs = []
    for row in rows:
        job = native_job(row)
        if row['action'] == 'REUSE':
            path = ROOT / row['source_result']
            assert sha(path) == row['source_result_sha256']
            old = read_json(path)
            assert old['epochs'] == 50 and old['job']['key'] == job['key']
            assert old['job']['config'] == job['config']
        else:
            jobs.append(dict(family=row['family'], job=job))
    baseline_files = {name: sha(BASE / name) for name in (
        'test_metrics.csv', 'validation_metrics.csv', 'validation_epoch_metrics.csv',
        'epoch_metrics.csv', 'costs.csv', 'formal_configuration_frozen.json', 'selection_frozen.json', 'data_manifest.json')}
    value = dict(source=source_manifest(), matrix=rows, jobs=jobs,
                 baseline_files=baseline_files, new_runs=30, reused_runs=10)
    path = attempt / 'PREPARED.json'
    if path.exists():
        assert read_json(path) == value, 'Prepared protocol or source changed'
    else:
        write_json(path, value)
        for item in jobs:
            write_json(attempt / item['family'] / 'jobs' / (item['job']['key'] + '.json'), item['job'])
    write_json(REPORT / 'IMPLEMENTATION_SOURCE_MANIFEST.json', value['source'])
    return value


def review_check(path):
    receipt = read_json(path)
    assert receipt['verdict'] in ('PASS', 'PASS WITH FIXES') and not receipt.get('blocking_issues')
    assert receipt['source_hash'] == source_manifest()['sha256'], 'Reviewed source changed'
    return receipt


def record(attempt, name, value):
    with (Path(attempt) / name).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def freeze_entries(results):
    entries = []
    for result in results:
        assert result['epochs'] == 50
        for kind in KINDS:
            for checkpoint in CHECKPOINTS:
                entries.append(dict(key=result['job']['key'], job=result['job'], model_kind=kind,
                    checkpoint=checkpoint, epoch=result['selected_epochs'][kind] if checkpoint == 'selected'
                    else int(checkpoint[1:]), model_hash=result[checkpoint + '_hashes'][kind]))
    return entries


def run(attempt, devices, review):
    review_check(review)
    prepared = prepare(attempt)
    attempt = Path(attempt)
    if (attempt / 'COMPLETE.json').exists():
        return read_json(attempt / 'COMPLETE.json')
    for name, module in FAMILIES.items():
        if not (attempt / name / 'SHUTDOWN.json').exists():
            module.ensure_guardian(attempt / name)
    processes = {}
    started = time.time()
    while not (attempt / 'SELECTION_FROZEN.json').exists():
        states = {name: module.request(attempt / name, 'status')['jobs']
                  for name, module in FAMILIES.items()}
        pending, active, failed = [], [], []
        for item in prepared['jobs']:
            family, job = item['family'], item['job']
            state = states[family].get(job['key'], {})
            process = processes.get(job['key'])
            if state.get('status') == 'COMPLETE':
                continue
            if history.alive(state.get('worker_pid')) or (process and process.poll() is None):
                active.append(dict(family=family, key=job['key'], epoch=state.get('epoch', 0),
                                   allocated=state.get('allocated', False)))
                continue
            if state or process:
                infrastructure = state.get('failure', {}).get('oom') or state.get('status') == 'RUNNING'
                if infrastructure and state.get('attempts', 0) < 2:
                    pending.append(item)
                else:
                    failed.append(dict(family=family, key=job['key'], state=state,
                                       returncode=process.poll() if process else None))
            else:
                pending.append(item)
        progress = dict(complete=30-len(pending)-len(active)-len(failed), total=30,
                        active=active, pending=len(pending), failed=failed, test_opened=False,
                        wall_seconds=time.time()-started)
        write_json(attempt / 'progress.json', progress)
        if failed:
            raise RuntimeError('Worker failure: inspect progress.json and launch_logs')
        if not pending and not active:
            break
        if pending and all(r['allocated'] for r in active) and history.available_ram() > 6 * 1024**3:
            gpu_reading = history.gpu_status()
            gpu = history.choose_device(gpu_reading, devices, 2 * 1024**3)
            if gpu is not None:
                item = pending[0]
                family, job = item['family'], item['job']
                previous = states[family].get(job['key'], {})
                log = attempt / 'launch_logs' / (job['key'] + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('ab') as stream:
                    process = subprocess.Popen([sys.executable, '-m', FAMILIES[family].__name__,
                        '_worker', '--attempt', str(attempt / family), '--key', job['key'],
                        '--device', f'cuda:{gpu}'], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=stream, start_new_session=True)
                processes[job['key']] = process
                record(attempt, 'launch_ledger.jsonl', dict(family=family, key=job['key'], pid=process.pid,
                    gpu=gpu, time=time.time(), previous_attempts=previous.get('attempts', 0),
                    reason='infrastructure_replay' if previous else 'new', gpu_status=gpu_reading))
        time.sleep(1)
    if not (attempt / 'SELECTION_FROZEN.json').exists():
        frozen = {}
        for family, module in FAMILIES.items():
            results = module.results(attempt / family)
            expected = {i['job']['key'] for i in prepared['jobs'] if i['family'] == family}
            assert {r['job']['key'] for r in results} == expected
            frozen[family] = dict(source_hash=module.source_manifest()['sha256'],
                training_complete=True, validation_only_selection=True, test_opened=False,
                endpoints=freeze_entries(results))
        assert sum(len(f['endpoints']) for f in frozen.values()) == 180
        write_json(attempt / 'CONFIGURATION_FROZEN.json', prepared)
        write_json(attempt / 'SELECTION_FROZEN.json', frozen)
    frozen = read_json(attempt / 'SELECTION_FROZEN.json')
    for family, value in frozen.items():
        if family == 'history':
            write_json(attempt / family / 'CONFIGURATION_FROZEN.json', dict(training_complete=True))
        write_json(attempt / family / ('SELECTION_FROZEN.json' if family == 'history'
                                      else 'selection_frozen.json'), value)
    for family, module in FAMILIES.items():
        family_attempt = attempt / family
        shutdown_path = family_attempt / 'SHUTDOWN.json'
        if not shutdown_path.exists():
            module.final_evaluate(family_attempt, devices)
        evaluation = read_json(family_attempt / 'final_evaluation.json')
        assert evaluation['status'] == 'COMPLETE'
        assert evaluation['endpoints'] == len(frozen[family]['endpoints'])
        if not shutdown_path.exists():
            write_json(shutdown_path, module.request(family_attempt, 'shutdown'))
    value = dict(status='COMPLETE', new_runs=30, new_test_endpoints=180,
                 started=started, finished=time.time(), source_hash=prepared['source']['sha256'])
    write_json(attempt / 'COMPLETE.json', value)
    write_json(attempt / 'progress.json', dict(value, complete=30, total=30, active=[], pending=0))
    return value


def candidate_records(attempt):
    """Return each candidate once, with scalar provenance and all endpoint views."""
    prepared = read_json(Path(attempt) / 'PREPARED.json')
    assert prepared['source'] == source_manifest()
    evaluations = {}
    records = []
    for row in prepared['matrix']:
        reused = row['action'] == 'REUSE'
        job = native_job(row)
        result_path = ROOT / row['source_result'] if reused else Path(attempt) / row['family'] / 'runs' / job['key'] / 'result.json'
        result = read_json(result_path)
        assert result['epochs'] == 50
        origin = result_path.parents[2]
        evaluation_path = origin / 'final_evaluation.json'
        if evaluation_path not in evaluations:
            evaluations[evaluation_path] = read_json(evaluation_path)['rows']
        base = dict(dataset=row['dataset'], arm=row['arm'], seed=int(row['seed']),
                    key=job['key'], configuration=history.digest(job['config']),
                    config_json=json.dumps(job['config'], sort_keys=True), historical_reference=reused,
                    source_result=str(result_path.relative_to(ROOT)), source_result_sha256=sha(result_path))
        endpoints, test_costs = [], []
        for kind in KINDS:
            for checkpoint in CHECKPOINTS:
                identity = dict(base, model_kind=kind, checkpoint=checkpoint,
                    epoch=result['selected_epochs'][kind] if checkpoint == 'selected' else int(checkpoint[1:]),
                    model_hash=result[checkpoint+'_hashes'][kind])
                endpoints.append(dict(identity, split='validation', **result[checkpoint+'_validation'][kind]))
                matches = [r for r in evaluations[evaluation_path] if r['key'] == job['key']
                           and r['model_kind'] == kind and r['checkpoint'] == checkpoint]
                assert len(matches) == 1
                test = matches[0]
                assert (test['epoch'], test['model_hash']) == (identity['epoch'], identity['model_hash'])
                endpoints.append(dict(identity, split='test', **test['metrics']))
                test_costs.append(dict(identity, cost_role='historical_test_reference' if reused else 'test',
                                       **test['cost']))
        records.append(dict(base=base, result=result, result_path=result_path, endpoints=endpoints,
                            test_costs=test_costs))
    return records


def controls(prepared):
    for filename, expected in prepared['baseline_files'].items():
        assert sha(BASE / filename) == expected
    endpoints = []
    for split, filename in [('test', 'test_metrics.csv'), ('validation', 'validation_metrics.csv')]:
        for row in read_csv(BASE / filename):
            assert row['arm'] in ('A0', 'A0_EMA', 'A1_EMA')
            arm = 'Gaussian' if row['arm'] == 'A1_EMA' else 'A0'
            kind = 'online' if row['arm'] == 'A0' else 'fixed_ema'
            assert row['model_kind'] == ('online' if kind == 'online' else
                                         'H2' if row['dataset'] == 'movielens' else 'ema995')
            metrics = {k: float(v) for k, v in row.items() if
                       k.endswith(('_auc', '_logloss', '_count', '_positive')) and v != ''}
            endpoints.append(dict(dataset=row['dataset'], arm=arm, seed=int(row['seed']),
                key=row['run_hash'], configuration=row['config_hash'], model_kind=kind,
                checkpoint=row['checkpoint'], epoch=int(row['epoch']), model_hash=row['model_hash'],
                split=split, historical_reference=True, source_result=str((BASE/filename).relative_to(ROOT)),
                source_result_sha256=prepared['baseline_files'][filename], **metrics))
    assert len(endpoints) == 180
    return endpoints


def verdict(rows):
    if len(rows) != 5 or {r['seed'] for r in rows} != SEEDS:
        return dict(status='INCOMPLETE', n=len(rows))
    mean_auc = stats([r['delta_macro_auc'] for r in rows])['mean']
    mean_ll = stats([r['delta_macro_logloss'] for r in rows])['mean']
    cell_deltas = {f'd{d}t{task}': stats([r[f'delta_d{d}t{task}_auc'] for r in rows])['mean']
                   for d in range(3) for task in range(2)}
    macro_pass = mean_auc >= .003
    cell_pass = all(value >= -.001 for value in cell_deltas.values())
    return dict(status='COMPLETE', n=5, mean_delta_auc=mean_auc, mean_delta_logloss=mean_ll,
                macro_auc_pass=macro_pass, cell_constraint_pass=cell_pass,
                auc_status=('AUC合格且满足逐格约束' if cell_pass else '宏AUC达标、逐格约束未满足')
                if macro_pass else '宏AUC未达0.003', logloss_decreased=mean_ll < 0,
                cell_mean_delta_auc=cell_deltas,
                violated_cells={k: v for k, v in cell_deltas.items() if v < -.001})


def summarize(attempt):
    attempt = Path(attempt)
    assert read_json(attempt / 'COMPLETE.json')['status'] == 'COMPLETE'
    prepared = read_json(attempt / 'PREPARED.json')
    candidates = candidate_records(attempt)
    endpoints = [e for record in candidates for e in record['endpoints']] + controls(prepared)
    identities = [(r['dataset'], r['arm'], r['seed'], r['model_kind'], r['checkpoint'], r['split'])
                  for r in endpoints]
    assert len(set(identities)) == len(endpoints) == 660
    lookup = dict(zip(identities, endpoints))
    metric_names = [k for k in endpoints[0] if k.endswith(('_auc', '_logloss'))]
    paired = []
    for row in endpoints:
        if row['arm'] == 'A0':
            continue
        for baseline_kind in KINDS:
            control = lookup[(row['dataset'], 'A0', row['seed'], baseline_kind, row['checkpoint'], row['split'])]
            comparison = 'total' if baseline_kind == 'online' else 'component'
            out = {k: row[k] for k in ('dataset', 'arm', 'seed', 'model_kind', 'checkpoint', 'split')}
            out.update(comparison=comparison, baseline_kind=baseline_kind,
                       candidate_epoch=row['epoch'], baseline_epoch=control['epoch'],
                       candidate_hash=row['model_hash'], baseline_hash=control['model_hash'])
            for metric in metric_names:
                out['candidate_'+metric] = row[metric]
                out['baseline_'+metric] = control[metric]
                out['delta_'+metric] = row[metric] - control[metric]
            paired.append(out)
    summaries, groups = [], {}
    for row in paired:
        identity = tuple(row[k] for k in ('dataset', 'arm', 'model_kind', 'checkpoint', 'split', 'comparison'))
        groups.setdefault(identity, []).append(row)
    decisions = []
    for identity, rows in sorted(groups.items()):
        base = dict(zip(('dataset', 'arm', 'model_kind', 'checkpoint', 'split', 'comparison'), identity))
        assert {r['seed'] for r in rows} == SEEDS and len(rows) == 5
        for metric in metric_names:
            summaries.append(dict(base, metric=metric, **stats([r['delta_'+metric] for r in rows])))
        if base['model_kind'] == 'fixed_ema' and base['checkpoint'] == 'selected' and base['split'] == 'test':
            decisions.append(dict(base, **verdict(rows)))
    absolute_groups = {}
    for row in endpoints:
        identity = tuple(row[k] for k in ('dataset', 'arm', 'model_kind', 'checkpoint', 'split'))
        absolute_groups.setdefault(identity, []).append(row)
    absolute = []
    for identity, rows in sorted(absolute_groups.items()):
        base = dict(zip(('dataset', 'arm', 'model_kind', 'checkpoint', 'split'), identity))
        assert {r['seed'] for r in rows} == SEEDS and len(rows) == 5
        for metric in metric_names:
            absolute.append(dict(base, metric=metric, **stats([r[metric] for r in rows])))
    epochs, costs, run_manifest = [], [], []
    for item in candidates:
        base, result = item['base'], item['result']
        costs.extend(item['test_costs'])
        run_manifest.append(dict(base, job=result['job'], epochs=result['epochs'],
                                 selected_epochs=result['selected_epochs']))
        by_epoch = {}
        for path in sorted(item['result_path'].parent.glob('attempt*/epoch_*.json')):
            epoch = read_json(path)
            by_epoch[epoch['epoch']] = epoch
        assert set(by_epoch) == set(range(1, 51))
        for epoch in by_epoch.values():
            for kind, values in epoch['validation'].items():
                epochs.append(dict(base, epoch=epoch['epoch'], model_kind=kind,
                    training_loss=epoch['loss'], training_real_loss=epoch['real_loss'], **values))
        costs.append(dict(base, cost_role='historical_reference' if base['historical_reference'] else 'new_training',
            new_training_seconds=0 if base['historical_reference'] else result['worker_seconds'],
            worker_seconds=result['worker_seconds'], peak_reserved_bytes=result.get('peak_reserved_bytes'),
            **result['counts']))
        if not base['historical_reference']:
            for path in sorted(item['result_path'].parent.glob('attempt*/physical_cost.json')):
                physical = read_json(path)
                costs.append(dict(base, cost_role='physical_attempt', **physical))
            for path in sorted(item['result_path'].parent.glob('attempt*/failure.json')):
                failure = read_json(path)
                costs.append(dict(base, cost_role='failed_physical_attempt', **failure))
    # Select epoch histories by exact run hash; the two A0 estimators share one training trajectory.
    baseline_lookup = {(r['dataset'], r['seed'], r['key'], r['model_kind']): r for r in endpoints
                       if r['arm'] in ('A0', 'Gaussian') and r['checkpoint'] == 'selected' and r['split'] == 'test'}
    baseline_losses = {(r['run_hash'], int(r['epoch'])): r for r in read_csv(BASE / 'epoch_metrics.csv')}
    for row in read_csv(BASE / 'validation_epoch_metrics.csv'):
        kind = 'online' if row['model_kind'] == 'online' else 'fixed_ema'
        if row['model_kind'] not in ('online', 'H2' if row['dataset'] == 'movielens' else 'ema995'):
            continue
        base = baseline_lookup.get((row['dataset'], int(row['seed']), row['run_hash'], kind))
        if base is None:
            continue
        values = {k: float(v) for k, v in row.items() if k.endswith(('_auc', '_logloss', '_count', '_positive')) and v != ''}
        training = baseline_losses[(row['run_hash'], int(row['epoch']))]
        epochs.append(dict(dataset=base['dataset'], arm=base['arm'], seed=base['seed'],
            key=base['key'], model_kind=kind, epoch=int(row['epoch']), historical_reference=True,
            training_loss=float(training['reference_fixed_mixed_loss']),
            training_real_loss=float(training['train_real_bce']),
            source_result=str((BASE/'validation_epoch_metrics.csv').relative_to(ROOT)), **values))
    baseline_hashes = {r['key'] for r in endpoints if r['arm'] in ('A0', 'Gaussian')}
    for row in read_csv(BASE / 'costs.csv'):
        if row['run_hash'] in baseline_hashes:
            costs.append(dict(row, cost_role='historical_control_reference', historical_reference=True,
                              new_training_seconds=0))
    write_csv(REPORT / 'endpoints.csv', endpoints)
    write_csv(REPORT / 'paired_seed.csv', paired)
    write_csv(REPORT / 'paired_summary.csv', summaries)
    write_csv(REPORT / 'absolute_summary.csv', absolute)
    write_csv(REPORT / 'validation_epochs.csv', epochs)
    write_csv(REPORT / 'costs.csv', costs)
    write_json(REPORT / 'RUN_MANIFEST.json', run_manifest)
    write_json(REPORT / 'DECISIONS.json', decisions)
    write_json(REPORT / 'COMPLETION.json', dict(read_json(attempt/'COMPLETE.json'),
        logical_candidates=40, candidate_test_endpoints=240, all_endpoint_rows=660,
        paired_rows=len(paired), validation_epoch_rows=len(epochs)))
    return dict(status='COMPLETE', endpoint_rows=660, decisions=len(decisions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['manifest', 'prepare', 'run', 'summarize'])
    parser.add_argument('--attempt', type=Path, default=ATTEMPT)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--review', type=Path, default=REPORT / 'INDEPENDENT_REVIEW_RECEIPT.json')
    args = parser.parse_args()
    if args.command == 'manifest':
        value = source_manifest()
    elif args.command == 'prepare':
        prepared = prepare(args.attempt)
        value = dict(new_runs=prepared['new_runs'], reused_runs=prepared['reused_runs'],
                     source_hash=prepared['source']['sha256'])
    elif args.command == 'run':
        value = run(args.attempt.resolve(), [int(d) for d in args.devices.split(',')], args.review)
    else:
        value = summarize(args.attempt)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
