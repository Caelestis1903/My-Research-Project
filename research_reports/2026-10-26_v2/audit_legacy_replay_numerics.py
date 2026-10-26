"""Technical data-only replay audit; never a scientific-effectiveness analysis."""
from pathlib import Path
import hashlib
import json
import numpy as np
from experiments.mdmt_rcf_20261026.protocol import RoleBundle, evaluate
from experiments.mdmt_rcf_20261026.campaign import S1_METHODS

ROOT = Path('artifacts/mdmt_rcf_20261026')
OLD = ROOT / 'v1'
NEW = ROOT / 'v2_four_seed'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def numeric_differences(a, b):
    if isinstance(a, dict):
        return [v for k in a.keys() & b.keys() for v in numeric_differences(a[k], b[k])]
    if isinstance(a, list):
        return [v for x, y in zip(a, b) for v in numeric_differences(x, y)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [abs(a-b)]
    return []


def run():
    rows = []
    for dataset in ('movielens', 'kuairand'):
        data = RoleBundle(dataset)
        ids = data.ids('Q_explore')
        for seed in (5201, 5202):
            old_path = OLD / 'summaries' / f'{dataset}_{seed}.json'
            old = json.loads(old_path.read_text())
            prefix = NEW / 'prefix' / dataset / f'seed{seed}'
            cache = json.loads((prefix / 'ROW_DATA_CACHE.json').read_text())
            mappings = {r['label']: r for r in cache['mappings']}
            previous_prefix = json.loads((OLD / 'prefix' / dataset / f'seed{seed}' / 'PREFIX_COMPLETE.json').read_text())
            current_prefix = json.loads((prefix / 'PREFIX_COMPLETE.json').read_text())
            hash_fields = ('native_hash', 'native_epoch24_hash', 'pilot_hash', 'selected_epoch', 'data_hash')
            prefix_checks = {k: dict(previous=previous_prefix['identity'][k], current=current_prefix['identity'][k],
                                    equal=previous_prefix['identity'][k] == current_prefix['identity'][k]) for k in hash_fields}
            assert all(r['equal'] for r in prefix_checks.values()), 'Scientific prefix values changed'
            methods = []
            missing = []
            for method in S1_METHODS:
                label = str((method, 'probabilities', 'Q_explore'))
                record = mappings.get(label)
                receipt_path = NEW / 'runs' / dataset / f'seed{seed}' / method / 'TRAINING_COMPLETE.json'
                if record is None or not receipt_path.exists():
                    missing.append(method)
                    continue
                probabilities = np.load(record['path'], mmap_mode='r', allow_pickle=False)
                assert probabilities.shape == (len(ids), 2) and np.isfinite(probabilities).all()
                current = evaluate(data.y[ids], probabilities, data.domain[ids])
                previous = old['metrics'][method]
                ds = numeric_differences(previous, current)
                receipt = json.loads(receipt_path.read_text())
                row = dict(method=method, max_abs_numeric_metric_difference=max(ds, default=0.),
                           abs_macro_logloss_difference=abs(previous['macro_logloss']-current['macro_logloss']),
                           abs_macro_auc_difference=abs(previous['macro_auc']-current['macro_auc']),
                           max_cell_abs_logloss_difference=max(abs(a['logloss']-b['logloss']) for a, b in zip(previous['cells'], current['cells'])),
                           max_cell_abs_auc_difference=max(abs(a['auc']-b['auc']) for a, b in zip(previous['cells'], current['cells'])),
                           prediction_data_path=record['path'], prediction_array_hash=record['array_hash'],
                           completion_receipt=str(receipt_path), completion_receipt_sha256=sha(receipt_path),
                           old_selected_policy='NOT_RECORDED', new_selected_policy=receipt['selected_policy'])
                if method == 'ETA_DIRECT_MATCHED':
                    policy = receipt['selected_policy']
                    ordered = sorted(policy['candidates'], key=lambda r: (r['p_cal_macro_ll'], r['pass_id']))
                    assert ordered[0]['pass_id'] == policy['selected_pass']
                    row['new_policy_is_Pcal_minimum'] = True
                    row['new_Pcal_best_to_runner_up_margin'] = ordered[1]['p_cal_macro_ll'] - ordered[0]['p_cal_macro_ll']
                    row['old_policy_equality'] = 'UNVERIFIABLE_NOT_RECORDED'
                    row['directly_used_by_S2_S3_trigger_conditions'] = False
                methods.append(row)
            rows.append(dict(dataset=dataset, seed=seed, legacy_summary_sha256=sha(old_path),
                             scientific_prefix_checks=prefix_checks, available_S1_methods=len(methods),
                             required_S1_methods=len(S1_METHODS), missing_S1_methods=missing,
                             later_old_stage_methods_not_replayed_by_S1=sorted(set(old['metrics'])-set(S1_METHODS)),
                             methods=methods))
    archive_files = [dict(path=str(p), relative_path=str(p.relative_to(OLD / 'summaries')),
                          sha256=sha(p), bytes=p.stat().st_size)
                     for p in sorted((OLD / 'summaries').rglob('*')) if p.is_file()]
    complete = all(not r['missing_S1_methods'] for r in rows)
    return dict(status='COMPLETE_TECHNICAL_AUDIT' if complete else 'INCOMPLETE_WAIT_FOR_DRAIN',
                scope='Absolute numeric replay differences only; no effectiveness directions or research conclusion',
                formal_training=False, models_loaded=False, outer_roles_used=False,
                old_reference_used_in_new_scientific_gates=False, rows=rows,
                original_summary_archive_files=archive_files,
                legacy_ETA_policy='NOT_RECORDED; equality cannot be claimed',
                numerical_cause='BLAS thread/reduction order and mapped-buffer layout are plausible concrete sources; unique causation not established')


if __name__ == '__main__':
    result = run()
    destination = Path('research_reports/2026-10-26_v2/LEGACY_REPLAY_NUMERIC_INDEPENDENT_AUDIT.json')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], contexts=[{k: r[k] for k in ('dataset', 'seed', 'available_S1_methods', 'required_S1_methods', 'missing_S1_methods')} for r in result['rows']]), ensure_ascii=False))
