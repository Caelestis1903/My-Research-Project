"""Reconstruct frozen test summaries and declared paired condition contrasts."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import v3_readout_20261016 as r
import v3_primary_fullstate_reconstruction_20261016 as raw

RAW_SHA = '90075adf80bc3b9351e7b602fa785c89675ee2fac9096887788e71936b6f2e59'
SUMMARY_SHA = '149430001d72e57f5de6e745afb7614462b7c9c5f6a8c5b19482fef3ab58c2aa'


def load(path):
    return json.loads(Path(path).read_text())


def seed_summary(values):
    values = np.asarray(values, float)
    n = len(values)
    assert 1 <= n <= 4 and np.isfinite(values).all()
    rng = np.random.default_rng(20261016)
    boot = values[rng.integers(0, n, size=(20000, n))].mean(axis=1)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=n)))
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(n=n, mean=float(values.mean()), sd=float(values.std(ddof=1)) if n > 1 else None,
        median=float(np.median(values)), positive=int((values > 0).sum()), negative=int((values < 0).sum()),
        seed_values=values.tolist(), ci_low=float(lo), ci_high=float(hi),
        paired_sign_randomization_p=float((abs((values*signs).mean(axis=1)) >= abs(values.mean())-1e-15).mean()))


def evaluate(curves, receipt, label, budget):
    """Use only frozen validation decisions and literal registered test labels."""
    average = False
    if label.startswith('common_epoch'):
        epochs = dict.fromkeys(('SN', 'SB'), int(label.removeprefix('common_epoch')))
    elif label == 'final_epoch':
        epochs = dict.fromkeys(('SN', 'SB'), budget)
    elif label.startswith(('AULC_1_', 'common_pre_AULC_1_')):
        epochs = dict.fromkeys(('SN', 'SB'), int(label.rsplit('_', 1)[1]))
        average = True
    elif label.startswith('selected_'):
        epochs = receipt['selected_epochs'][label.removeprefix('selected_')]
    elif label.startswith(('common_SN_selected_', 'common_SB_selected_')):
        choosing = label.split('_')[1]
        rule = label.split('_selected_', 1)[1]
        epochs = dict.fromkeys(('SN', 'SB'), receipt['selected_epochs'][rule][choosing])
    else:
        raise AssertionError(label)
    assert all(1 <= e <= budget for e in epochs.values())
    values = {arm: curves[arm].loc[1:epochs[arm]].mean() if average else curves[arm].loc[epochs[arm]]
              for arm in ('SN', 'SB')}
    return values, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    freeze_path = r.OUT/'test_readout_frozen.json'
    freeze = load(freeze_path)
    assert freeze['status'] == 'FROZEN' and freeze['all_condition_definitions_frozen'] is True
    assert freeze['analysis_rules_sha'] == r.base.sha(r.OUT/'analysis_rules_frozen.json')
    assert freeze['frozen_test_summary_verifier_manifest_sha'] == r.base.sha(args.manifest)
    assert freeze['frozen_test_summary_verifier_source_sha'] == r.base.sha(__file__)
    assert r.base.sha(Path(raw.__file__)) == RAW_SHA
    assert r.base.sha(Path(r.__file__).with_name('v3_test_summary_20261016.py')) == SUMMARY_SHA
    review = load(r.OUT/'frozen_test_summary_verifier_partial_support_v2_self_review.json')
    assert review['status'] == 'PASS_PRETEST' and review['source_sha'] == r.base.sha(__file__)
    inputs_path = Path(freeze['test_source_inputs_file'])
    assert r.base.sha(inputs_path) == freeze['test_source_inputs_sha']
    inputs = {str(Path(x['run']).resolve()): x for x in load(inputs_path)['records']}
    legacy = freeze.get('legacy_pair_producer_receipt_allowlist', [])
    assert not args.out.exists()
    spec = load(args.manifest)
    maximum, checks, aliases = 0., 0, 0
    sources, cache, data, epochs_by_key = {}, {}, {}, {}

    def close(actual, expected):
        nonlocal maximum, checks
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
        error = float(np.max(abs(actual-expected)))
        assert error < 2e-12, error
        maximum, checks = max(maximum, error), checks+actual.size

    for cohort in spec['cohorts']:
        name, manifest_path, summary_dir = cohort['name'], Path(cohort['summary_manifest']), Path(cohort['summary_dir'])
        assert r.base.sha(manifest_path) in freeze['test_summary_manifest_hashes']
        summary_receipt = load(summary_dir/'receipt.json')
        assert summary_receipt['status'] == 'COMPLETE' and summary_receipt['source_sha'] == SUMMARY_SHA
        assert summary_receipt['manifest_sha'] == r.base.sha(manifest_path)
        assert summary_receipt['test_global_freeze_sha'] == r.base.sha(freeze_path)
        mf = load(manifest_path)
        table = pd.read_csv(summary_dir/'seed_test_readouts.csv')
        cell_table = pd.read_csv(summary_dir/'seed_cell_test_readouts.csv')
        uncertainty = pd.read_csv(summary_dir/'test_selection_uncertainty.csv')
        assert not table.duplicated(['group', 'seed', 'readout', 'metric']).any()
        assert not cell_table.duplicated(['group', 'seed', 'readout', 'domain', 'task', 'metric']).any()
        pair_map = {}
        for group, directories in mf['groups'].items():
            observed = min(load(Path(x)/'receipt.json')['common_pre_degradation_epochs'] for x in directories)
            window = int(mf.get('common_pre_windows', {}).get(group, observed))
            assert 1 <= window <= observed
            for directory in map(Path, directories):
                receipt = load(directory/'receipt.json'); seed = receipt['seed']
                assert receipt['status'] == 'COMPLETE' and (group, seed) not in pair_map
                with np.load(directory/'bootstrap_curves.npz') as arr:
                    budget = len(arr['epochs']); assert np.array_equal(arr['epochs'], np.arange(1, budget+1))
                curves = {}
                for arm, run, old_sha in zip(('SN', 'SB'), receipt['source_runs'], receipt['source_run_receipt_sha']):
                    run = Path(run); key = str(run.resolve()); pin = inputs[key]
                    current_sha = r.base.sha(run/'receipt.json')
                    assert current_sha == pin['producer_receipt_sha']
                    if old_sha != current_sha:
                        assert any(x['pair'] == str(directory) and x['arm'] == arm and x['old_receipt_sha'] == old_sha
                            and x['current_receipt_sha'] == current_sha for x in legacy)
                    if key not in cache:
                        producer, config, completion = load(run/'receipt.json'), load(run/'config.json'), load(run/'test_readout_receipt.json')
                        assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
                        assert producer['epochs'] == pin['epochs'] >= budget
                        assert r.base.sha(run/'config.json') == pin['config_sha'] == producer['artifact_hashes']['config.json']
                        assert config['seed'] == seed and config['arm'] == arm
                        assert completion['status'] == 'COMPLETE' and completion['test_global_freeze_sha'] == r.base.sha(freeze_path)
                        assert completion['freeze_sha'] == freeze['analysis_rules_sha']
                        assert completion['source_sha'] in (r.base.sha(r.__file__), freeze['test_prefix_reuse_source_sha'])
                        for file, digest in completion['hashes'].items():
                            assert r.base.sha(run/file) == digest
                        frame = pd.read_csv(run/'metrics_test.csv')
                        assert len(frame) == producer['epochs']*6 and not frame.duplicated(['epoch', 'domain', 'task']).any()
                        assert set(frame.epoch) == set(range(1, producer['epochs']+1))
                        assert set(frame.split) == {'test'} and set(frame.arm) == {arm} and set(frame.seed) == {seed}
                        curve = raw.cell_curves(frame, config['p']).xs(('test', arm))
                        cache[key] = curve
                        sources[key] = dict(producer_receipt_sha=current_sha, config_sha=pin['config_sha'],
                            test_receipt_sha=r.base.sha(run/'test_readout_receipt.json'), metrics_sha=r.base.sha(run/'metrics_test.csv'))
                    curves[arm] = cache[key]
                pair_map[group, seed] = (receipt, budget, curves)
                labels = {f'common_epoch{e}' for e in (5, 10, 15, 20, 30, 40, 50, 100) if e <= budget}
                labels.add('final_epoch')
                labels.update(f'{prefix}{rule}' for rule in receipt['selected_epochs']
                    for prefix in ('selected_', 'common_SN_selected_', 'common_SB_selected_'))
                labels.update(f'AULC_1_{e}' for e in (10, 20, 50) if e <= budget)
                labels.add(f'common_pre_AULC_1_{window}')
                pair_rows = table[(table.group == group) & (table.seed == seed)]
                pair_cells = cell_table[(cell_table.group == group) & (cell_table.seed == seed)]
                metrics = {v for v in curves['SN'].columns if not (v.startswith('d') and len(v) > 3 and v[2] == 't')}
                assert len(metrics) == 16
                assert set(zip(pair_rows.readout, pair_rows.metric)) == set(itertools.product(labels, metrics))
                assert set(zip(pair_cells.readout, pair_cells.domain, pair_cells.task, pair_cells.metric)) == set(
                    itertools.product(labels, range(3), range(2), ('auc', 'logloss')))
        assert set(zip(table.group, table.seed)) == set(pair_map)
        assert set(zip(cell_table.group, cell_table.seed)) == set(pair_map)
        for frame, cells in [(table, False), (cell_table, True)]:
            for row in frame.itertuples():
                receipt, budget, curves = pair_map[row.group, row.seed]
                vals, ep = evaluate(curves, receipt, row.readout, budget)
                if row.readout.startswith('common_pre_AULC_1_'):
                    observed = min(load(Path(x)/'receipt.json')['common_pre_degradation_epochs'] for x in mf['groups'][row.group])
                    assert int(row.readout.rsplit('_', 1)[1]) == int(mf.get('common_pre_windows', {}).get(row.group, observed))
                variable = f'd{row.domain}t{row.task}_{row.metric}' if cells else row.metric
                sn, sb = (float(vals[arm][variable]) for arm in ('SN', 'SB'))
                close([row.SN, row.SB, row.difference], [sn, sb, sb-sn])
                if not np.isnan(row.SN_epoch):
                    assert row.SN_epoch == ep['SN'] and row.SB_epoch == ep['SB']
                data[name, row.group, row.seed, row.readout, variable] = (sn, sb)
                epochs_by_key[name, row.group, row.seed, row.readout, variable] = (ep['SN'], ep['SB'], budget)
                if row.readout == 'final_epoch':
                    alias, _ = evaluate(curves, receipt, 'selected_fixed_budget', budget)
                    close([sn, sb], [float(alias[arm][variable]) for arm in ('SN', 'SB')]);aliases += 1
        for frame, filename, cell in [(table, 'test_seed_summary.json', False), (cell_table, 'test_cell_seed_summary.json', True)]:
            for item in load(summary_dir/filename):
                part = frame[(frame.group == item['group']) & (frame.readout == item['readout']) & (frame.metric == item['metric'])]
                if cell:
                    part = part[(part.domain == item['domain']) & (part.task == item['task'])]
                part = part.sort_values('seed'); assert not part.seed.duplicated().any()
                expected = seed_summary(part.difference)
                for field, number in expected.items():
                    if number is None: assert item[field] is None
                    else: close(item[field], number)
                close([item['SN_mean'], item['SB_mean']], [part.SN.mean(), part.SB.mean()])
        for item in uncertainty.itertuples():
            receipt, budget, curves = pair_map[item.group, item.seed]
            directory = next(Path(x) for x in mf['groups'][item.group] if load(Path(x)/'receipt.json')['seed'] == item.seed)
            bs = pd.read_csv(directory/'selection_bootstrap.csv');part = bs[bs.rule == item.rule]
            assert len(part) == 1000
            values = curves['SB'].loc[part.SB_epoch.to_numpy(), 'macro_auc'].to_numpy()-curves['SN'].loc[part.SN_epoch.to_numpy(), 'macro_auc'].to_numpy()
            close([item.test_auc_difference_at_reselected_epochs_ci_low, item.ci_high], np.quantile(values, [.025, .975]))
    contrast_rows, excluded = [], []
    for comparison in spec['comparisons']:
        cohort, baseline = comparison['cohort'], comparison['baseline']
        for condition in comparison['conditions']:
            left = {(s, ro, v) for c, g, s, ro, v in data if c == cohort and g == baseline}
            right = {(s, ro, v) for c, g, s, ro, v in data if c == cohort and g == condition}
            excluded.append(dict(cohort=cohort, baseline=baseline, condition=condition, baseline_only=sorted(left-right), condition_only=sorted(right-left)))
            for seed, label, variable in sorted(left & right):
                sn, sb = data[cohort, condition, seed, label, variable]
                nsn, nsb = data[cohort, baseline, seed, label, variable]
                ce = epochs_by_key[cohort, condition, seed, label, variable]
                be = epochs_by_key[cohort, baseline, seed, label, variable]
                contrast_rows.append(dict(cohort=cohort, baseline=baseline, condition=condition, seed=seed, readout=label,
                    variable=variable, interaction=(sb-sn)-(nsb-nsn), SN_condition_minus_baseline=sn-nsn, SB_condition_minus_baseline=sb-nsb,
                    condition_SN_epoch_or_horizon=ce[0], condition_SB_epoch_or_horizon=ce[1], condition_budget=ce[2],
                    baseline_SN_epoch_or_horizon=be[0], baseline_SB_epoch_or_horizon=be[1], baseline_budget=be[2],
                    same_actual_epochs_or_horizons=ce[:2] == be[:2],
                    scientific_alias_of_selected_fixed_budget=label == 'final_epoch'))
    contrasts = pd.DataFrame(contrast_rows);summaries = []
    for keys, frame in contrasts.groupby(['cohort', 'baseline', 'condition', 'readout', 'variable']):
        frame = frame.sort_values('seed'); assert not frame.seed.duplicated().any()
        item = dict(zip(['cohort', 'baseline', 'condition', 'readout', 'variable'], keys))
        item.update(seed_summary(frame.interaction));item.update(SN_condition_minus_baseline_mean=float(frame.SN_condition_minus_baseline.mean()), SB_condition_minus_baseline_mean=float(frame.SB_condition_minus_baseline.mean()))
        item['scope'] = ('Fourseedconditionalcontrast' if len(frame) == 4 else
                         f'Partialseed_descriptive_n{len(frame)}_eligibility_conditional_on_actualbudget_notfullfour-seed')
        summaries.append(item)
    args.out.mkdir(parents=True, exist_ok=False)
    contrasts.to_csv(args.out/'seed_test_contrasts.csv', index=False)
    r.base.dump(args.out/'test_contrast_summary.json', summaries)
    r.base.dump(args.out/'excluded_support_keys.json', excluded)
    r.base.dump(args.out/'receipt.json', dict(status='PASS', time=time.time(), independent_review=False, source_sha=r.base.sha(__file__),
        manifest_sha=r.base.sha(args.manifest), test_global_freeze_sha=r.base.sha(freeze_path), sources=sources,
        cohorts=len(spec['cohorts']), numeric_values_checked=checks, max_error=maximum, final_alias_rows_checked=aliases,
        contrast_rows=len(contrasts), contrast_summary_groups=len(summaries), test_used_for_choices=False,
        scope='Frozen test summary reconstruction from raw metrics; existing declared comparisons only, no new endpoint/selection/training rule. Final epoch aliases fixed budget, not independent evidence.'))
    print(json.dumps(dict(status='PASS', values=checks, max_error=maximum, contrasts=len(contrasts), aliases=aliases)))


if __name__ == '__main__':
    main()
