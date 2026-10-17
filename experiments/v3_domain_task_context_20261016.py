"""Join existing final V8 cell performance, support, exposure and telemetry."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'activation', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    review = load(r.OUT/'V8_final_domain_task_context_self_review.json')
    assert review['status'] == 'PASS_PREPARED_ONLY' and review['source_sha'] == r.base.sha(__file__)
    approval = load(args.activation)
    assert approval['status'] == 'PASS_ACTUAL_ALL24_CELLS_AND_COUPLING'
    assert approval['manifest_sha'] == r.base.sha(args.manifest)
    assert not args.out.exists()
    spec = load(args.manifest)
    pins = {item['cell_dir']: item for item in approval['cell_records']}
    rows, telemetry, sources, support_reference = [], [], [], {}
    for item in spec['records']:
        pair = Path(item['pair']); receipt = load(pair/'receipt.json')
        assert receipt['status'] == 'COMPLETE' and receipt['seed'] == item['seed']
        for name, digest in receipt['hashes'].items():
            assert r.base.sha(pair/name) == digest
        cells = pd.read_csv(pair/'cell_trajectory.csv')
        for i, (arm, source, digest, cell_dir) in enumerate(zip(('SN', 'SB'), receipt['source_runs'], receipt['source_run_receipt_sha'], item['cell_dirs'])):
            source, cell_dir = Path(source), Path(cell_dir)
            producer, config = load(source/'receipt.json'), load(source/'config.json')
            assert r.base.sha(source/'receipt.json') == digest
            assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
            for name in ('config.json', 'trajectory.csv', 'optimizer_observations.csv'):
                assert r.base.sha(source/name) == producer['artifact_hashes'][name]
            pin = pins[str(cell_dir)]
            for filename, key in [('receipt.json', 'cell_receipt_sha'), ('cell_support.csv', 'support_sha'), ('cell_train_performance.csv', 'performance_sha')]:
                assert r.base.sha(cell_dir/filename) == pin[key]
            completion = load(cell_dir/'receipt.json')
            assert completion['status'] == 'PASS' and Path(completion['run']) == source
            assert completion['config_sha'] == r.base.sha(source/'config.json')
            if 'training_receipt_sha' in completion:
                assert completion['training_receipt_sha'] == digest and completion['actual_budget'] == producer['epochs']
            else:
                assert producer['epochs'] == 50
            support = pd.read_csv(cell_dir/'cell_support.csv')
            performance = pd.read_csv(cell_dir/'cell_train_performance.csv')
            trajectory = pd.read_csv(source/'trajectory.csv').set_index('epoch')
            observations = pd.read_csv(source/'optimizer_observations.csv')
            batches_per_epoch = int(np.ceil(sum(config['data']['train_domain_counts'].values()) / config['batch_size']))
            assert config['dataset'] == 'kuairand' and batches_per_epoch == 49
            assert len(support) == 24 and not support.duplicated(['split', 'domain', 'task']).any()
            assert set(support.split) == {'train', 'val_all', 'selection', 'diagnostic'}
            train_support = support[support.split == 'train'].set_index(['domain', 'task'])
            diag_support = support[support.split == 'diagnostic'].set_index(['domain', 'task'])
            baseline_support = support[['split', 'domain', 'task', 'n', 'positive']].sort_values(['split', 'domain', 'task']).reset_index(drop=True)
            if item['seed'] not in support_reference:
                support_reference[item['seed']] = baseline_support
            else:
                pd.testing.assert_frame_equal(support_reference[item['seed']], baseline_support)
            stages = dict(common10=10, common50=50, own_final=producer['epochs'], producer_selected=producer['selected_epoch'])
            assert producer['selected_epoch'] == receipt['selected_epochs']['macro_auc'][arm]
            assert sorted(set(stages.values())) == completion['epochs']
            for stage, epoch in stages.items():
                train = performance[performance.epoch == epoch].set_index(['domain', 'task'])
                diagnostic = cells[(cells.split == 'diagnostic') & (cells.arm == arm) & (cells.epoch == epoch)].set_index(['domain', 'task'])
                assert len(train) == len(diagnostic) == 6
                current = trajectory.loc[epoch]
                assert abs(train.empirical_objective_contribution.sum()-current.train_empirical_objective) < 1e-6+1e-5*abs(current.train_empirical_objective)
                for domain in range(3):
                    for task in range(2):
                        tr, di = train.loc[(domain, task)], diagnostic.loc[(domain, task)]
                        ts, ds = train_support.loc[(domain, task)], diag_support.loc[(domain, task)]
                        count, draws, unique = int(ts.n), int(current[f'exposure_d{domain}']), int(current[f'unique_epoch_d{domain}'])
                        assert count == int(config['data']['train_domain_counts'][str(domain)])
                        assert 0 <= unique <= min(draws, count)
                        assert int(tr.unique_exposed) == int(current[f'unique_cumulative_d{domain}'])
                        assert abs(tr.unique_exposed/count-current[f'coverage_d{domain}']) < 1e-12
                        cumulative_draws = int(trajectory.loc[:epoch, f'exposure_d{domain}'].sum())
                        assert int(tr.actual_exposure) == cumulative_draws
                        assert int(ds.n) == int(di.n)
                        rows.append(dict(group=item['group'], seed=item['seed'], arm=arm, stage=stage, epoch=epoch,
                            actual_final_budget=producer['epochs'], domain=domain, task=task, train_n=count,
                            train_positive=int(ts.positive), train_positive_rate=float(ts.positive_rate),
                            diagnostic_n=int(ds.n), diagnostic_positive=int(ds.positive), diagnostic_positive_rate=float(ds.positive_rate),
                            diagnostic_auc=float(di.auc), diagnostic_logloss=float(di.logloss),
                            train_auc=float(tr.auc), train_logloss=float(tr.logloss), J_contribution=float(tr.empirical_objective_contribution),
                            epoch_draws=draws, epoch_unique=unique, epoch_repeated=draws-unique,
                            minimum_repeats_at_fixed_quota=max(0, draws-count), repeat_histogram=current[f'repeat_histogram_d{domain}'],
                            cumulative_draws=cumulative_draws, cumulative_positive_exposure=float(tr.positive_exposure),
                            cumulative_unique=int(tr.unique_exposed), cumulative_coverage=float(current[f'coverage_d{domain}']), run=str(source)))
                observed = observations[observations.epoch == epoch]
                for block in 'CPRE':
                    part = observed[observed.block == block].sort_values('batch')
                    assert part.batch.tolist() == list(range(6))
                    assert part.steps.tolist() == list(range((epoch-1)*batches_per_epoch+1, (epoch-1)*batches_per_epoch+7))
                    telemetry.append(dict(group=item['group'], seed=item['seed'], arm=arm, stage=stage, epoch=epoch, block=block,
                        gradient_norm_first6_mean=float(part.gradient_norm.mean()), update_norm_first6_mean=float(part.update_norm.mean()),
                        actual_step_ids=';'.join(map(str, part.steps)), sampled_batches=6, run=str(source),
                        scope='Movingtheta_first6datagradients_notdomain-specific_or_full-epoch_orfixedtheta'))
            sources.append(dict(group=item['group'], seed=item['seed'], arm=arm, run=str(source), producer_receipt_sha=digest,
                config_sha=r.base.sha(source/'config.json'), trajectory_sha=r.base.sha(source/'trajectory.csv'),
                observations_sha=r.base.sha(source/'optimizer_observations.csv'), cell_receipt_sha=r.base.sha(cell_dir/'receipt.json'),
                support_sha=r.base.sha(cell_dir/'cell_support.csv'), train_cell_sha=r.base.sha(cell_dir/'cell_train_performance.csv')))
    frame = pd.DataFrame(rows)
    assert len(sources) == 24 and len(frame) == 576 and len(telemetry) == 384
    keys = ['seed', 'arm', 'stage', 'domain', 'task']
    baseline = frame[frame.group == 'clean'].set_index(keys)
    comparisons = []
    for group in ('mild', 'strong'):
        for row in frame[frame.group == group].to_dict('records'):
            original = baseline.loc[tuple(row[k] for k in keys)]
            for metric in ('diagnostic_auc', 'diagnostic_logloss', 'train_auc', 'train_logloss', 'J_contribution'):
                comparisons.append(dict(group=group, seed=row['seed'], arm=row['arm'], stage=row['stage'], domain=row['domain'], task=row['task'],
                    condition_epoch=row['epoch'], baseline_epoch=int(original.epoch), metric=metric, condition=row[metric], baseline=float(original[metric]),
                    difference=row[metric]-float(original[metric]), matched_epoch=row['epoch'] == int(original.epoch)))
    args.out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out/'domain_task_context.csv', index=False)
    pd.DataFrame(telemetry).to_csv(args.out/'sparse_block_gradient_context.csv', index=False)
    pd.DataFrame(comparisons).to_csv(args.out/'severity_minus_clean_context.csv', index=False)
    r.base.dump(args.out/'receipt.json', dict(status='PASS', time=time.time(), source_sha=r.base.sha(__file__), independent_review=False,
        manifest_sha=r.base.sha(args.manifest), activation_sha=r.base.sha(args.activation), sources=sources,
        context_rows=len(frame), telemetry_rows=len(telemetry), descriptive_comparison_rows=len(comparisons), test_access=False,
        no_new_training_probes_or_statistical_estimators=True, scope='Descriptive source joins. Sparse block norms are moving-theta first6 data gradients, not domain-specific/fixedtheta/full-epoch norms; alias stages add no independent observations.'))


if __name__ == '__main__':
    main()
