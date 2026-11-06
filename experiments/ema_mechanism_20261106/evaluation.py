"""Complete-budget endpoint freezing and an isolated, RAM-only Test reader."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file, state_digest, digest
from .analysis import ROOT, SEEDS, Incomplete, _jobs, _fid, load_records, input_hashes, write_table, _plain


def _binding(study, out):
    out = Path(out)
    binding = input_hashes(study, out)
    for name in ['STAGE1_DECISION.json', 'STAGE2_DECISION.json', 'STAGE3_DECISION.json',
                 'STAGE4_DECISION.json', 'CONTROL_RECEIPT.json', 'CONTROL_RESULTS.csv',
                 'PREDICTIVE_FOLDS.json', 'PREDICTION_SEED.csv', 'ARCHIVED_FACTORIAL.csv']:
        if not (out/name).exists():
            raise Incomplete('Cannot freeze without '+name)
        binding[name] = sha_file(out/name)
    return binding


def _complete_summaries(study, out):
    from .state import request
    records = load_records(study, out)
    summaries = request(out, 'summaries')['summaries']
    expected = {job['job_id']: job for job in _jobs(study)}
    if set(summaries) != set(expected):
        raise Incomplete('All twenty live RAM trajectory summaries are required')
    by_job = {}
    for row in records:
        by_job.setdefault(row['job_id'], {})[row['epoch']] = row
    for job_id, summary in summaries.items():
        job = expected[job_id]
        if not summary['complete'] or summary['epoch'] != 50 or summary['global_step'] != 50*job['steps_per_epoch']:
            raise Incomplete('RAM trajectory budget incomplete: '+job_id)
        physical = ['online']+list(study.manifest['datasets'][job['dataset']]['rho_by_physical_shadow'])
        if set(summary['snapshots']) != set(physical):
            raise ValueError('RAM candidate set differs from the frozen physical readouts')
        for rid, markers in summary['snapshots'].items():
            if set(markers) != {'selected', 'E10', 'E50'}:
                raise Incomplete('Missing RAM endpoint markers: '+job_id+'/'+rid)
            selected = max(range(1, 51), key=lambda epoch: (by_job[job_id][epoch]['validation'][rid]['macro_auc'], -epoch))
            for marker, endpoint in markers.items():
                epoch = endpoint['epoch']
                if epoch != (selected if marker == 'selected' else int(marker[1:])):
                    raise ValueError('RAM checkpoint epoch violates the earliest-strict-maximum rule')
                if endpoint['model_hash'] != by_job[job_id][epoch]['model_hash'][rid]:
                    raise ValueError('RAM endpoint hash differs from the committed Validation curve')
    return summaries


def freeze_endpoints(study, out):
    out = Path(out)
    prepared = read_json(out/'PREPARED.json')
    if prepared['manifest_hash'] != study.manifest_hash or prepared['identity'] != study.identity:
        raise ValueError('Prepared source, plan or implementation differs at freeze')
    binding = _binding(study, out)
    summaries = _complete_summaries(study, out)
    endpoints = []
    for job in _jobs(study):
        summary = summaries[job['job_id']]
        for rid, markers in summary['snapshots'].items():
            fid = _fid(study, job['dataset'])
            canonical = 'F' if rid == fid else rid
            for marker in ('selected', 'E10', 'E50'):
                point = markers[marker]
                endpoints.append(dict(job_id=job['job_id'], dataset=job['dataset'], objective=job['objective'],
                    seed=job['seed'], physical_readout_id=rid, readout_id=canonical,
                    aliases=['F', 'H2'] if job['dataset'] == 'movielens' and rid == fid else [rid],
                    checkpoint=marker, epoch=point['epoch'], state_hash=point['model_hash'],
                    validation=point['validation'], layout_hash=summary['layout_hash'], group_hash=summary['group_hash'],
                    data_hash=prepared['data_identity'][job['dataset']]['data_hash'],
                    source_commit=study.identity['source_commit'], implementation_hash=study.identity['implementation_hash']))
    if len(endpoints) > 270:
        raise ValueError('Frozen Test endpoint budget exceeded')
    selection = dict(schema_version='1.0', study_id=study.manifest['study_id'], endpoints=endpoints,
        input_hashes=binding, seed_ids=list(SEEDS), scientific_unit='seed',
        comparison_rules=dict(regular_only=True, checkpoint_rule='complete_validation_macro_auc_earliest_strict_max',
                              checkpoints=['selected', 'E10', 'E50'], pulse_test=False,
                              metric_semantics='source_FP32_probability_AUC_and_float64_LogLoss_log_floor_minus100'),
        historical_test_exposure=True)
    selection_path, receipt_path = out/'SELECTION_FROZEN.json', out/'FREEZE_RECEIPT.json'
    if selection_path.exists():
        if read_json(selection_path) != _plain(selection):
            raise ValueError('Frozen endpoint identity changed')
        validate_freeze(study, out)
        return dict(status='ALREADY_COMMITTED', endpoint_count=len(endpoints))
    write_json(selection_path, _plain(selection))
    receipt = dict(schema_version='1.0', study_id=study.manifest['study_id'],
        selection_hash=sha_file(selection_path), input_hashes=binding, seed_ids=list(SEEDS),
        endpoint_count=len(endpoints), frozen_at_unix=time.time(), fresh_test_accessed=False,
        historical_test_exposure=True, comparison_rules_hash=digest(selection['comparison_rules']))
    write_json(receipt_path, receipt)
    return receipt


def validate_freeze(study, out):
    out = Path(out)
    selection = read_json(out/'SELECTION_FROZEN.json')
    receipt = read_json(out/'FREEZE_RECEIPT.json')
    if sha_file(out/'SELECTION_FROZEN.json') != receipt['selection_hash']:
        raise ValueError('Frozen endpoint file was modified')
    if selection['input_hashes'] != receipt['input_hashes'] or _binding(study, out) != receipt['input_hashes']:
        raise ValueError('Source, observations, analysis or frozen comparisons changed')
    if selection['seed_ids'] != list(SEEDS) or receipt['seed_ids'] != list(SEEDS):
        raise Incomplete('Frozen scientific seed set differs')
    if receipt['comparison_rules_hash'] != digest(selection['comparison_rules']):
        raise ValueError('Frozen Test comparison rules changed')
    expected = {(job['dataset'], job['objective'], int(job['seed'])) for job in _jobs(study)}
    actual = {(r['dataset'], r['objective'], int(r['seed'])) for r in selection['endpoints']}
    if actual != expected or len(expected) != 20:
        raise Incomplete('Test gate lacks the complete twenty-job coverage')
    return selection, receipt


def _archive_index(study, out):
    """Only the pinned, audited same-protocol source tables are eligible."""
    from .analysis import audit_archives
    # Audit was frozen already; validate source bytes without overwriting its products.
    import hashlib
    specs = study.manifest['source']['archive_sources']
    result = {}
    for objective in ('clean', 'gaussian_d0'):
        spec = specs[objective]
        path = ROOT/spec['path']
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob != spec['git_blob']:
            raise ValueError('Pinned archived Test source changed')
        frame = pd.read_csv(path)
        if objective == 'clean':
            frame = frame[(frame['arm'] == 'A0') & frame['model_kind'].isin(['online', 'fixed_ema'])]
        else:
            frame = frame[frame['candidate_id'].isin(['online', 'F'])]
        frame = frame[frame['split'] == 'test']
        for row in frame.to_dict('records'):
            key = row['dataset'], objective, int(row['seed']), row['model_hash']
            row_hash = digest(_plain(row))
            item = dict(metrics={k: v for k, v in row.items() if (k.endswith(('_auc', '_logloss', '_count', '_positive', '_negative')))
                                 and isinstance(v, (int, float)) and pd.notna(v)},
                        source_path=spec['path'], source_commit=spec['commit'],
                        source_sha256=sha_file(path), source_row_hash=row_hash)
            if key in result and result[key]['metrics'] != item['metrics']:
                raise ValueError('Conflicting archived metrics for the same parameter state')
            result[key] = item
    return result


def _evaluate_process(study, out):
    import torch
    from .state import request
    from .protocol import verify_data_identity
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.evaluation import evaluate_model
    from experiments.aug_history_20261105.model_adapter import build_model

    out = Path(out)
    selection, receipt = validate_freeze(study, out)
    prepared = read_json(out/'PREPARED.json')
    cache = prepared['cache_root']
    current_data = verify_data_identity(study, cache, prepared['paths'])
    if current_data != prepared['data_identity']:
        raise ValueError('Test data identity differs from frozen preflight')
    # All gates run before include_test=True or any Test model prediction.
    _complete_summaries(study, out)
    archives = _archive_index(study, out)
    prior_path = out/'ENDPOINTS.csv'
    evaluation_receipt = out/'TEST_EVALUATION_RECEIPT.json'
    if prior_path.exists() and evaluation_receipt.exists():
        prior = read_json(evaluation_receipt)
        if prior['freeze_receipt_hash'] != sha_file(out/'FREEZE_RECEIPT.json') or prior['endpoint_hash'] != sha_file(prior_path):
            raise ValueError('Existing Test endpoint transaction conflicts')
        return dict(prior, status='ALREADY_COMMITTED')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    jobs = {job['job_id']: job for job in prepared['jobs']}
    grouped = {}
    for endpoint in selection['endpoints']:
        grouped.setdefault(endpoint['job_id'], []).append(endpoint)
    rows, test_costs, evaluated = [], [], {}
    for job_id, endpoints in grouped.items():
        job = jobs[job_id]
        snapshot = request(out, 'get', key=job_id)['state']
        model = bundle = None
        for endpoint in endpoints:
            point = snapshot['snapshots'][endpoint['physical_readout_id']][endpoint['checkpoint']]
            if point['model_hash'] != endpoint['state_hash'] or state_digest(point['model_state']) != endpoint['state_hash']:
                raise ValueError('RAM endpoint changed after freezing')
            key = endpoint['dataset'], endpoint['objective'], int(endpoint['seed']), endpoint['state_hash']
            started = time.perf_counter()
            if key in evaluated:
                metrics, exposure, provenance = evaluated[key]
                exposure = 'duplicate_frozen_state'
            elif key in archives:
                item = archives[key]
                metrics, exposure = item['metrics'], 'archived'
                provenance = {k: v for k, v in item.items() if k != 'metrics'}
                evaluated[key] = metrics, exposure, provenance
            else:
                if model is None:
                    bundle = cached_bundle(cache, job['dataset'], include_test=True)
                    model = build_model(bundle, job['config']).to(device)
                model.load_state_dict(point['model_state'])
                metrics = evaluate_model(model, bundle, 'test', 'M3OE', job['config']['eval_batch_size'])
                exposure = 'new_evaluation_of_historically_exposed_population'
                provenance = dict(source_path='RAM_GUARDIAN', source_commit=study.identity['source_commit'])
                evaluated[key] = metrics, exposure, provenance
            base = {k: v for k, v in endpoint.items() if k != 'validation'}
            base['aliases'] = json.dumps(base['aliases'])
            rows.append(dict(base, split='validation', model_hash=endpoint['state_hash'],
                             test_exposure='validation', **endpoint['validation']))
            rows.append(dict(base, split='test', model_hash=endpoint['state_hash'], test_exposure=exposure,
                             **provenance, **metrics))
            test_costs.append(dict(job_id=job_id, dataset=job['dataset'], objective=job['objective'], seed=job['seed'],
                                  state_hash=endpoint['state_hash'], checkpoint=endpoint['checkpoint'],
                                  test_seconds=time.perf_counter()-started, exposure=exposure,
                                  new_test_forward=int(exposure.startswith('new_evaluation'))))
        del snapshot, model, bundle
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    write_table(out/'ENDPOINTS.csv', rows)
    write_table(out/'TEST_COSTS.csv', test_costs)
    result = dict(status='COMPLETED', process_id=os.getpid(), isolated_test_process=True,
        freeze_receipt_hash=sha_file(out/'FREEZE_RECEIPT.json'), endpoint_hash=sha_file(out/'ENDPOINTS.csv'),
        logical_endpoints=len(selection['endpoints']), unique_states=len(evaluated),
        new_test_evaluations=sum(c['new_test_forward'] for c in test_costs),
        archived_or_duplicate_reuse=sum(not c['new_test_forward'] for c in test_costs),
        device=device, completed_at_unix=time.time())
    write_json(evaluation_receipt, result)
    return result


def evaluate_frozen(study, out):
    """The public CLI always creates a fresh process before opening Test arrays."""
    out = Path(out).resolve()
    validate_freeze(study, out)
    prepared = read_json(out/'PREPARED.json')
    manifest = prepared['paths']['manifest']
    command = [sys.executable, '-m', 'experiments.ema_mechanism_20261106.evaluation',
               '--manifest', manifest, '--out', str(out), '--internal-evaluate']
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError('Independent Test evaluator failed with exit code '+str(result.returncode))
    return read_json(out/'TEST_EVALUATION_RECEIPT.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--internal-evaluate', action='store_true', required=True)
    args = parser.parse_args(argv)
    from .protocol import resolve_study
    study = resolve_study(args.manifest)
    prepared = read_json(Path(args.out)/'PREPARED.json')
    study.paths.update(prepared['paths'])
    return _evaluate_process(study, args.out)


if __name__ == '__main__':
    try:
        main()
    except Incomplete as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(5)
