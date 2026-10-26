"""Apply frozen factorial contrasts on common complete marginal support."""
import argparse
import json
from pathlib import Path

import pandas as pd
import v3_factorial_contrasts_20261016 as factor
import v3_readout_20261016 as r


def common_support(frame, groups):
    keys = ['split', 'readout', 'variable']
    supports = {g: set(map(tuple, frame[frame.group == g][keys].to_numpy())) for g in groups}
    common = set.intersection(*supports.values())
    assert common
    mask = frame[keys].apply(tuple, axis=1).isin(common)
    return frame[mask].copy(), {
        g: [list(key) for key in sorted(supports[g] - common)] for g in groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marginals', type=Path, required=True)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt_file = args.marginals / 'receipt.json'
    receipt = json.loads(receipt_file.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['joint_window']
    assert len(set(receipt['windows'].values())) == 1
    for name, digest in receipt['aggregate_sources'].items():
        assert r.base.sha(args.aggregate / name) == digest
    aggregate = json.loads((args.aggregate / 'receipt.json').read_text())
    assert aggregate['status'] == 'COMPLETE'
    budgets = []
    for item in aggregate['sources']:
        pair = Path(item['pair_directory'])
        assert r.base.sha(pair / 'receipt.json') == item['receipt_sha']
        source = json.loads((pair / 'receipt.json').read_text())
        assert source['status'] == 'COMPLETE'
        for arm, directory, digest in zip(('SN', 'SB'), source['source_runs'], source['source_run_receipt_sha']):
            producer = Path(directory) / 'receipt.json'
            assert r.base.sha(producer) == digest
            actual = json.loads(producer.read_text())
            budgets.append(dict(group=item['group'], seed=item['seed'], arm=arm,
                epochs=actual['epochs'], run=directory, receipt_sha=digest))
    source_frame = args.marginals / 'seed_readouts.csv'
    frame = pd.read_csv(source_frame)
    groups = list(receipt['windows'])
    assert set(frame.group) == set(groups)
    assert all(sorted(frame[frame.group == g].seed.unique().tolist()) == [2022, 2023, 2024, 2025] for g in groups)
    filtered, excluded = common_support(frame, groups)
    result = factor.contrasts(filtered, args.baseline, groups)
    summaries = []
    for key, part in result.groupby(['group', 'split', 'readout', 'variable']):
        part = part.sort_values('seed')
        assert part.seed.tolist() == [2022, 2023, 2024, 2025]
        summaries.append(dict(zip(('group', 'split', 'readout', 'variable'), key)) |
            r.paired_summary(part.interaction) |
            dict(SN_condition_minus_baseline_mean=float(part.SN_condition_minus_baseline.mean()),
                 SB_condition_minus_baseline_mean=float(part.SB_condition_minus_baseline.mean()),
                 matched_epoch_or_horizon=bool(part.matched_epoch_or_horizon.all())))
    args.out.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.out / 'seed_contrasts.csv', index=False)
    r.base.dump(args.out / 'summary.json', summaries)
    r.base.dump(args.out / 'excluded_support_keys.json', excluded)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        source_sha=r.base.sha(__file__), frozen_contrast_function_source_sha=r.base.sha(factor.__file__),
        original_readout_sha=r.base.sha(r.__file__), marginal_receipt_sha=r.base.sha(receipt_file),
        marginal_csv_sha=r.base.sha(source_frame), baseline=args.baseline, groups=groups,
        windows=receipt['windows'], actual_budgets=budgets,
        original_rows=len(frame), common_support_rows=len(filtered),
        excluded_rows=len(frame) - len(filtered), independent_review=False,
        estimand='(SB-SN)_condition - (SB-SN)_baseline; paired by training seed',
        support_rule='Intersection of original complete split/readout/variable keys; missing native100 is never fabricated.',
        scope='No new statistical rule, selected epochs remain explicitly unmatched, no test or automatic mechanistic gate.'))


if __name__ == '__main__':
    main()
