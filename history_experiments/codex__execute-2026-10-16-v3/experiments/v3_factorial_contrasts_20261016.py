"""Paired contrasts of sampling gaps from completed joint-window marginals."""
import argparse
import json
from pathlib import Path

import pandas as pd
import v3_readout_20261016 as r


def contrasts(frame, baseline, groups):
    keys = ['seed', 'split', 'readout', 'variable']
    assert set(frame.group) == set(groups) and baseline in groups
    assert not frame.duplicated(['group'] + keys).any()
    reference = frame[frame.group == baseline].drop(columns='group')
    rows = []
    for group in groups:
        current = frame[frame.group == group].drop(columns='group')
        joined = current.merge(reference, on=keys, suffixes=('_condition', '_baseline'),
                               how='outer', validate='one_to_one', indicator=True)
        assert (joined['_merge'] == 'both').all(), group
        common = joined.readout.str.startswith(('common_epoch', 'AULC_', 'common_pre_'))
        for arm in ('SN', 'SB'):
            assert (joined.loc[common, f'{arm}_epoch_or_horizon_condition'] ==
                    joined.loc[common, f'{arm}_epoch_or_horizon_baseline']).all(), group
        joined['group'] = group
        joined['baseline'] = baseline
        joined['interaction'] = joined.difference_condition - joined.difference_baseline
        joined['SN_condition_minus_baseline'] = joined.SN_condition - joined.SN_baseline
        joined['SB_condition_minus_baseline'] = joined.SB_condition - joined.SB_baseline
        joined['matched_epoch_or_horizon'] = common
        rows.append(joined.drop(columns='_merge'))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marginals', type=Path, required=True)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--archived-receipts', type=Path)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.marginals / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['joint_window'] is True
    assert len(set(receipt['windows'].values())) == 1
    for name, sha in receipt['aggregate_sources'].items():
        assert r.base.sha(args.aggregate / name) == sha
    aggregate = json.loads((args.aggregate / 'receipt.json').read_text())
    assert aggregate['status'] == 'COMPLETE'
    archived = (json.loads(args.archived_receipts.read_text())
                if args.archived_receipts else {})
    budgets = []
    for source in aggregate['sources']:
        pair_dir = Path(source['pair_directory'])
        assert r.base.sha(pair_dir / 'receipt.json') == source['receipt_sha']
        pair = json.loads((pair_dir / 'receipt.json').read_text())
        assert pair['status'] == 'COMPLETE'
        for arm, directory, sha in zip(('SN', 'SB'), pair['source_runs'],
                                       pair['source_run_receipt_sha']):
            path = Path(directory) / 'receipt.json'
            if r.base.sha(path) != sha:
                path = Path(archived[sha])
            assert r.base.sha(path) == sha
            producer = json.loads(path.read_text())
            budgets.append(dict(group=source['group'], seed=source['seed'], arm=arm,
                                epochs=producer['epochs'], run=directory, receipt_sha=sha,
                                actual_receipt=str(path)))
    frame_path = args.marginals / 'seed_readouts.csv'
    frame = pd.read_csv(frame_path)
    groups = list(receipt['windows'])
    seed_sets = {group: sorted(frame[frame.group == group].seed.unique().tolist())
                 for group in groups}
    assert all(seeds == seed_sets[args.baseline] for seeds in seed_sets.values())
    assert set(seed_sets[args.baseline]) >= {2022, 2023, 2024, 2025}
    result = contrasts(frame, args.baseline, groups)
    summaries = []
    for key, part in result.groupby(['group', 'split', 'readout', 'variable']):
        part = part.sort_values('seed')
        assert part.seed.tolist() == seed_sets[key[0]]
        summaries.append(dict(zip(('group', 'split', 'readout', 'variable'), key)) |
                         r.paired_summary(part.interaction) |
                         dict(SN_condition_minus_baseline_mean=float(
                                  part.SN_condition_minus_baseline.mean()),
                              SB_condition_minus_baseline_mean=float(
                                  part.SB_condition_minus_baseline.mean()),
                              matched_epoch_or_horizon=bool(
                                  part.matched_epoch_or_horizon.all())))
    args.out.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.out / 'seed_contrasts.csv', index=False)
    r.base.dump(args.out / 'summary.json', summaries)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        source_sha=r.base.sha(__file__), marginal_receipt_sha=r.base.sha(receipt_path),
        marginal_csv_sha=r.base.sha(frame_path), baseline=args.baseline,
        archived_receipts_manifest_sha=(r.base.sha(args.archived_receipts)
                                        if args.archived_receipts else None),
        seeds=seed_sets, windows=receipt['windows'], actual_budgets=budgets,
        independent_review=False,
        estimand='(SB-SN)_condition - (SB-SN)_baseline, paired by training seed',
        scope='Joint fixed horizons; condition-dependent selected epochs are explicitly '
              'unmatched. Seed uncertainty only, no mechanistic gate or test access. '
              'Actual budgets remain pinned in source marginal/aggregate/producer receipts.'))


if __name__ == '__main__':
    main()
