"""Verify matched factor statistics directly from producer six-cell tables."""
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


def load(path):
    return json.loads(Path(path).read_text())


def independent_summary(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (4,) and np.isfinite(values).all()
    rng = np.random.default_rng(20261016)
    draws = values[rng.integers(0, 4, size=(20000, 4))].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    signs = np.asarray(list(itertools.product([-1, 1], repeat=4)))
    return dict(mean=values.mean(), median=np.median(values), sd=values.std(ddof=1),
                positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                ci_low=lo, ci_high=hi, paired_sign_randomization_p=float(
                    (abs((values * signs).mean(axis=1)) >= abs(values.mean())-1e-15).mean()))


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'activation', 'aggregate', 'marginals', 'contrasts', 'budgets', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    assert r.base.sha(Path(raw.__file__)) == RAW_SHA
    review = load(r.OUT / 'matched_factor_numeric_verifier_self_review.json')
    assert review['status'] == 'PASS_PREPARED_ONLY' and review['source_sha'] == r.base.sha(__file__)
    approved, manifest, budgets = load(args.activation), load(args.manifest), load(args.budgets)
    assert approved['status'].startswith('PASS') and approved['manifest_sha'] == r.base.sha(args.manifest)
    assert set(budgets) == {'2022', '2023', '2024', '2025'}
    pins = {item['pair']: item['receipt_sha'] for item in approved['records']}
    assert args.baseline in manifest['groups']
    assert not args.out.exists()
    for directory in (args.aggregate, args.marginals, args.contrasts):
        assert load(directory / 'receipt.json')['status'] == 'COMPLETE'
    mr, cr = load(args.marginals/'receipt.json'), load(args.contrasts/'receipt.json')
    assert mr['joint_window'] and len(set(mr['windows'].values())) == 1
    joint = next(iter(mr['windows'].values()))
    for name, digest in mr['aggregate_sources'].items():
        assert r.base.sha(args.aggregate/name) == digest
    assert cr['marginal_receipt_sha'] == r.base.sha(args.marginals/'receipt.json')
    assert cr['marginal_csv_sha'] == r.base.sha(args.marginals/'seed_readouts.csv')
    cells = pd.read_csv(args.aggregate/'cell_trajectory.csv')
    trajectories = pd.read_csv(args.aggregate/'domain_task_trajectory.csv')
    curves, receipts, sources = {}, {}, []
    maximum, checks = 0., 0

    def close(actual, expected):
        nonlocal maximum, checks
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
        error = float(np.max(abs(actual-expected)))
        assert error < 2e-12, error
        maximum, checks = max(maximum, error), checks+actual.size

    for group, directories in manifest['groups'].items():
        seeds = []
        for directory in map(Path, directories):
            receipt = load(directory/'receipt.json')
            assert receipt['status'] == 'COMPLETE'
            assert r.base.sha(directory/'receipt.json') == pins[str(directory)]
            for name, digest in receipt['hashes'].items():
                assert r.base.sha(directory/name) == digest
            seed = receipt['seed']; seeds.append(seed)
            for arm, run, digest in zip(('SN', 'SB'), receipt['source_runs'], receipt['source_run_receipt_sha']):
                run = Path(run); producer = load(run/'receipt.json'); config = load(run/'config.json')
                assert r.base.sha(run/'receipt.json') == digest
                assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
                assert producer['epochs'] == int(budgets[str(seed)])
                assert config['seed'] == seed and config['arm'] == arm
                for name in ('config.json', 'trajectory.csv'):
                    assert r.base.sha(run/name) == producer['artifact_hashes'][name]
                assert pd.read_csv(run/'trajectory.csv').epoch.tolist() == list(range(1, producer['epochs']+1))
                sources.append(dict(group=group, seed=seed, arm=arm, run=str(run),
                    producer_receipt_sha=digest, config_sha=r.base.sha(run/'config.json'),
                    pair=str(directory), pair_receipt_sha=r.base.sha(directory/'receipt.json'),
                    epochs=producer['epochs']))
            original = pd.read_csv(directory/'cell_trajectory.csv')
            actual = cells[(cells.group == group) & (cells.seed == seed)].drop(columns='group').reset_index(drop=True)
            pd.testing.assert_frame_equal(original.reset_index(drop=True), actual, check_exact=False, atol=1e-14, rtol=0)
            config = load(Path(receipt['source_runs'][0])/'config.json')
            curve = raw.cell_curves(original, config['p'])
            curves[group, seed], receipts[group, seed] = curve, receipt
            part = trajectories[(trajectories.group == group) & (trajectories.seed == seed)]
            for row in part.itertuples():
                point = curve.loc[(row.split, row.arm, row.epoch)]
                for variable in point.index:
                    if hasattr(row, variable):
                        close(float(getattr(row, variable)), float(point[variable]))
        assert sorted(seeds) == [2022, 2023, 2024, 2025]
    marginal = pd.read_csv(args.marginals/'seed_readouts.csv')
    values = {}
    for (group, seed, split, label), part in marginal.groupby(['group', 'seed', 'split', 'readout']):
        selected, epochs = raw.chosen_rows(curves[group, seed], receipts[group, seed], split, label, joint)
        for row in part.itertuples():
            sn, sb = (float(selected[arm][row.variable]) for arm in ('SN', 'SB'))
            close([row.SN, row.SB, row.difference], [sn, sb, sb-sn])
            assert row.SN_epoch_or_horizon == epochs['SN'] and row.SB_epoch_or_horizon == epochs['SB']
            values[group, seed, split, label, row.variable] = (sn, sb)
    contrasts = pd.read_csv(args.contrasts/'seed_contrasts.csv')
    for row in contrasts.itertuples():
        sn, sb = values[row.group, row.seed, row.split, row.readout, row.variable]
        nsn, nsb = values[args.baseline, row.seed, row.split, row.readout, row.variable]
        close([row.interaction, row.SN_condition_minus_baseline, row.SB_condition_minus_baseline],
              [(sb-sn)-(nsb-nsn), sn-nsn, sb-nsb])
        if hasattr(row, 'matched_epoch_or_horizon'):
            assert bool(row.matched_epoch_or_horizon) == row.readout.startswith(('common_epoch', 'AULC_', 'common_pre_'))
    summary_count = 0
    for directory, frame, value in [(args.marginals, marginal, 'difference'), (args.contrasts, contrasts, 'interaction')]:
        for item in load(directory/'summary.json'):
            part = frame[(frame.group == item['group']) & (frame.split == item['split']) &
                (frame.readout == item['readout']) & (frame.variable == item['variable'])].sort_values('seed')
            assert part.seed.tolist() == [2022, 2023, 2024, 2025]
            close(item['seed_values'], part[value].to_numpy())
            for key, number in independent_summary(part[value].to_numpy()).items():
                close(float(item[key]), float(number))
            if value == 'interaction':
                close([item['SN_condition_minus_baseline_mean'], item['SB_condition_minus_baseline_mean']],
                    [part.SN_condition_minus_baseline.mean(), part.SB_condition_minus_baseline.mean()])
            summary_count += 1
    args.out.mkdir(parents=True, exist_ok=False)
    r.base.dump(args.out/'receipt.json', dict(status='PASS', time=time.time(), independent_review=False,
        source_sha=r.base.sha(__file__), raw_reduction_dependency_sha=RAW_SHA,
        manifest_sha=r.base.sha(args.manifest), activation_sha=r.base.sha(args.activation),
        marginal_receipt_sha=r.base.sha(args.marginals/'receipt.json'),
        contrast_receipt_sha=r.base.sha(args.contrasts/'receipt.json'), baseline=args.baseline,
        actual_matched_budgets=budgets, joint_window=joint, sources=sources,
        marginal_rows=len(marginal), contrast_rows=len(contrasts), summary_groups=summary_count,
        numeric_values_checked=checks, max_error=maximum, test_access=False,
        scope='Direct raw six-cell reductions; all fixed/selected/AULC seed readouts and paired interactions; manually reconstructed four-seed bootstrap and exact sign summaries.'))
    print(json.dumps(dict(status='PASS', values=checks, max_error=maximum, summaries=summary_count)))


if __name__ == '__main__':
    main()
