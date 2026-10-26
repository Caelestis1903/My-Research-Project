"""Reconstruct V11 statistics from raw cells and export declared sensitivities."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def load(path):
    return json.loads(path.read_text())


def cell_curves(frame, p):
    """Direct reductions of six source cells; no aggregate table is an oracle."""
    records = []
    for (split, arm, epoch), cells in frame.groupby(['split', 'arm', 'epoch']):
        assert len(cells) == 6 and set(zip(cells.domain, cells.task)) == {
            (d, t) for d in range(3) for t in range(2)}
        item = dict(split=split, arm=arm, epoch=int(epoch))
        for metric in ('auc', 'logloss'):
            values = cells[metric].to_numpy()
            item['macro_' + metric] = float(values.sum() / 6)
            item['sample_weighted_cell_' + metric] = float(
                np.dot(values, cells.n) / cells.n.sum())
            weights = np.asarray(p)[cells.domain.to_numpy(dtype=int)]
            item['train_objective_weighted_cell_' + metric] = float(
                np.dot(values, weights) / weights.sum())
            for domain in range(3):
                item[f'd{domain}_{metric}'] = float(
                    cells.loc[cells.domain == domain, metric].sum() / 2)
            for task in range(2):
                item[f't{task}_{metric}'] = float(
                    cells.loc[cells.task == task, metric].sum() / 3)
            for row in cells.itertuples():
                item[f'd{row.domain}t{row.task}_{metric}'] = float(getattr(row, metric))
        records.append(item)
    return pd.DataFrame(records).set_index(['split', 'arm', 'epoch']).sort_index()


def chosen_rows(curve, receipt, split, label, joint):
    if label.startswith('common_epoch'):
        epochs = dict.fromkeys(('SN', 'SB'), int(label.removeprefix('common_epoch')))
        average = False
    elif label.startswith('AULC_1_') or label == 'common_pre_AULC':
        end = joint if label == 'common_pre_AULC' else int(label.rsplit('_', 1)[1])
        epochs, average = dict.fromkeys(('SN', 'SB'), end), True
    elif label.startswith('selected_'):
        epochs = receipt['selected_epochs'][label.removeprefix('selected_')]
        average = False
    elif label in ('common_SN_selected_macro_auc', 'common_SB_selected_macro_auc'):
        choosing = label.split('_')[1]
        epochs = dict.fromkeys(('SN', 'SB'), receipt['selected_epochs']['macro_auc'][choosing])
        average = False
    else:
        raise AssertionError(label)
    values = {}
    for arm in ('SN', 'SB'):
        part = curve.loc[(split, arm)]
        values[arm] = part.loc[1:epochs[arm]].mean() if average else part.loc[epochs[arm]]
    return values, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parents[1], r.OUT
    manifest = out / 'V11_final_fullstate_comparison_manifest.json'
    activation = out / 'V11_final_fullstate_analysis_activation_review.json'
    approved = load(activation)
    assert approved['status'] == 'PASS_ACTUAL_ALL16_FULL_PARENTS'
    assert approved['manifest_sha'] == r.base.sha(manifest)
    review = load(out / 'V11_fullstate_numeric_reconstruction_self_review.json')
    assert review['status'] == 'PASS_PREPARED_ONLY_BEFORE_ACTUAL_RUN'
    assert review['source_sha'] == r.base.sha(__file__)
    aggregate = out / 'analysis/V11_fullstate_final_aggregate'
    marginals = out / 'analysis/V11_fullstate_final_joint_marginals'
    contrasts = out / 'analysis/V11_fullstate_final_supported_joint_contrasts'
    ar, mr, cr = (load(directory / 'receipt.json') for directory in
                  (aggregate, marginals, contrasts))
    assert ar['status'] == mr['status'] == cr['status'] == 'COMPLETE'
    assert mr['joint_window'] and len(set(mr['windows'].values())) == 1
    joint = next(iter(mr['windows'].values()))
    assert joint == approved['joint_window']
    for name, digest in mr['aggregate_sources'].items():
        assert r.base.sha(aggregate / name) == digest
    assert cr['marginal_receipt_sha'] == r.base.sha(marginals / 'receipt.json')
    assert cr['marginal_csv_sha'] == r.base.sha(marginals / 'seed_readouts.csv')
    sources = {item['pair']: item for item in approved['records']}
    curves, receipts, trajectories, budgets, provenance = {}, {}, {}, {}, []
    for group, pairs in load(manifest)['groups'].items():
        for directory in map(Path, pairs):
            receipt = load(directory / 'receipt.json')
            assert receipt['status'] == 'COMPLETE'
            assert r.base.sha(directory / 'receipt.json') == sources[str(directory)]['receipt_sha']
            assert r.base.sha(directory / 'cell_trajectory.csv') == receipt['hashes']['cell_trajectory.csv']
            seed = receipt['seed']
            config = load(Path(receipt['source_runs'][0]) / 'config.json')
            curves[group, seed] = cell_curves(pd.read_csv(directory / 'cell_trajectory.csv'), config['p'])
            receipts[group, seed] = receipt
            for arm, run, digest in zip(('SN', 'SB'), receipt['source_runs'],
                                        receipt['source_run_receipt_sha']):
                run = Path(run)
                assert r.base.sha(run / 'receipt.json') == digest
                producer = load(run / 'receipt.json')
                assert r.base.sha(run / 'trajectory.csv') == producer['artifact_hashes']['trajectory.csv']
                assert r.base.sha(run / 'config.json') == producer['artifact_hashes']['config.json']
                frame = pd.read_csv(run / 'trajectory.csv').set_index('epoch')
                assert frame.index.tolist() == list(range(1, producer['epochs'] + 1))
                trajectories[group, seed, arm] = frame
                budgets[group, seed, arm] = producer['epochs']
                provenance.append(dict(group=group, seed=seed, arm=arm, run=str(run),
                    producer_receipt_sha=digest, config_sha=r.base.sha(run / 'config.json'),
                    trajectory_sha=r.base.sha(run / 'trajectory.csv'),
                    pair_receipt_sha=r.base.sha(directory / 'receipt.json'), epochs=producer['epochs']))
    marginal = pd.read_csv(marginals / 'seed_readouts.csv')
    errors, marginal_values = [], {}
    for (group, seed, split, label), frame in marginal.groupby(['group', 'seed', 'split', 'readout']):
        values, epochs = chosen_rows(curves[group, seed], receipts[group, seed], split, label, joint)
        for row in frame.itertuples():
            sn, sb = (float(values[arm][row.variable]) for arm in ('SN', 'SB'))
            errors.extend((abs(sn-row.SN), abs(sb-row.SB), abs((sb-sn)-row.difference)))
            assert row.SN_epoch_or_horizon == epochs['SN'] and row.SB_epoch_or_horizon == epochs['SB']
            marginal_values[group, seed, split, label, row.variable] = (sn, sb)
    seed_contrasts = pd.read_csv(contrasts / 'seed_contrasts.csv')
    for row in seed_contrasts.itertuples():
        sn, sb = marginal_values[row.group, row.seed, row.split, row.readout, row.variable]
        nsn, nsb = marginal_values['nativeAdam', row.seed, row.split, row.readout, row.variable]
        for actual, expected in ((row.interaction, (sb-sn)-(nsb-nsn)),
                                 (row.SN_condition_minus_baseline, sn-nsn),
                                 (row.SB_condition_minus_baseline, sb-nsb)):
            errors.append(abs(actual-expected))
        expected_match = row.readout.startswith(('common_epoch', 'AULC_', 'common_pre_'))
        assert bool(row.matched_epoch_or_horizon) == expected_match
    summaries = load(contrasts / 'summary.json')
    for item in summaries:
        part = seed_contrasts[(seed_contrasts.group == item['group']) &
            (seed_contrasts.split == item['split']) & (seed_contrasts.readout == item['readout']) &
            (seed_contrasts.variable == item['variable'])].sort_values('seed')
        assert part.seed.tolist() == [2022, 2023, 2024, 2025]
        summary = r.paired_summary(part.interaction)
        for key in ('mean', 'ci_low', 'ci_high', 'positive', 'negative', 'paired_sign_randomization_p'):
            errors.append(abs(float(summary[key])-float(item[key])))
        errors.extend((abs(part.SN_condition_minus_baseline.mean()-item['SN_condition_minus_baseline_mean']),
                       abs(part.SB_condition_minus_baseline.mean()-item['SB_condition_minus_baseline_mean'])))
        assert bool(item['matched_epoch_or_horizon']) == bool(part.matched_epoch_or_horizon.all())
    assert max(errors) < 1e-12
    supplemental, health = [], []
    for (group, seed), curve in curves.items():
        for split in ('selection', 'diagnostic'):
            for label in ('original_native_gate_AULC_1_22', 'own_final_epoch'):
                vals = {}
                for arm in ('SN', 'SB'):
                    part = curve.loc[(split, arm)]
                    epoch = 22 if label.startswith('original_native_gate') else budgets[group, seed, arm]
                    vals[arm] = part.loc[1:22].mean() if label.startswith('original_native_gate') else part.loc[epoch]
                for variable in curve.columns:
                    sn, sb = (float(vals[arm][variable]) for arm in ('SN', 'SB'))
                    supplemental.append(dict(group=group, seed=seed, split=split, readout=label,
                        variable=variable, SN=sn, SB=sb, difference=sb-sn,
                        SN_epoch_or_horizon=22 if label.startswith('original_native_gate') else budgets[group, seed, 'SN'],
                        SB_epoch_or_horizon=22 if label.startswith('original_native_gate') else budgets[group, seed, 'SB']))
            for arm in ('SN', 'SB'):
                for label, epoch in (('common10', 10), ('common50', 50),
                                      ('own_final', budgets[group, seed, arm])):
                    row = curve.loc[(split, arm, epoch)]
                    health.append(dict(group=group, seed=seed, split=split, arm=arm,
                        stage=label, epoch=epoch, macro_auc=float(row.macro_auc),
                        macro_logloss=float(row.macro_logloss),
                        train_empirical_objective=float(trajectories[group, seed, arm].loc[epoch, 'train_empirical_objective'])))
    extra = pd.DataFrame(supplemental)
    interactions, extra_summary = [], []
    for key, part in extra.groupby(['seed', 'split', 'readout', 'variable']):
        ref = part[part.group == 'nativeAdam'].iloc[0]
        for row in part[part.group != 'nativeAdam'].itertuples():
            interactions.append(dict(seed=key[0], split=key[1], readout=key[2], variable=key[3],
                group=row.group, interaction=(row.SB-row.SN)-(ref.SB-ref.SN),
                SN_condition_minus_baseline=row.SN-ref.SN, SB_condition_minus_baseline=row.SB-ref.SB,
                matched_epoch_or_horizon=key[2].startswith('original_native_gate'),
                condition_SN_epoch=row.SN_epoch_or_horizon, condition_SB_epoch=row.SB_epoch_or_horizon,
                native_SN_epoch=int(ref.SN_epoch_or_horizon), native_SB_epoch=int(ref.SB_epoch_or_horizon)))
    interactions = pd.DataFrame(interactions)
    for key, part in interactions.groupby(['group', 'split', 'readout', 'variable']):
        part = part.sort_values('seed')
        assert part.seed.tolist() == [2022, 2023, 2024, 2025]
        extra_summary.append(dict(zip(('group', 'split', 'readout', 'variable'), key)) |
            r.paired_summary(part.interaction) |
            dict(SN_condition_minus_baseline_mean=float(part.SN_condition_minus_baseline.mean()),
                 SB_condition_minus_baseline_mean=float(part.SB_condition_minus_baseline.mean()),
                 matched_epoch_or_horizon=bool(part.matched_epoch_or_horizon.all())))
    args.out.mkdir(parents=True, exist_ok=False)
    extra.to_csv(args.out / 'supplemental_seed_readouts.csv', index=False)
    interactions.to_csv(args.out / 'supplemental_seed_contrasts.csv', index=False)
    pd.DataFrame(health).to_csv(args.out / 'absolute_health.csv', index=False)
    r.base.dump(args.out / 'supplemental_summary.json', extra_summary)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE', time=time.time(),
        source_sha=r.base.sha(__file__), independent_review=False,
        specification_sha=r.base.sha(out / 'V11_fullstate_numeric_reconstruction_and_endpoint_specification.json'),
        review_sha=r.base.sha(out / 'V11_fullstate_numeric_reconstruction_self_review.json'),
        activation_sha=r.base.sha(activation), marginal_rows=len(marginal), contrast_rows=len(seed_contrasts),
        summary_groups=len(summaries), max_numeric_error=max(errors), joint_window=joint,
        original_native_fixed_window=22, original_native_window_not_joint_preoverfit=True,
        ownfinal_comparisons_explicitly_unmatched=True, provenance=provenance,
        input_receipt_hashes={str(d): r.base.sha(d/'receipt.json') for d in (aggregate, marginals, contrasts)},
        test_access=False, hashes={p.name:r.base.sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='PASS_NUMERIC_RECONSTRUCTION', marginal_rows=len(marginal),
        contrast_rows=len(seed_contrasts), summary_groups=len(summaries), max_numeric_error=max(errors))))


if __name__ == '__main__':
    main()
