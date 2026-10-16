"""All prespecified V11 local state contrasts, with training-seed uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v3_readout_20261016 as r

KEYS = ['seed', 'epoch', 'split', 'variable']


def state_contrasts(frame):
    """Every comparison uses the same stage, metric and seed; no selection."""
    rows = []
    for key, part in frame.groupby(KEYS):
        base = part[part.condition == 'BASE']
        assert len(base) == 1
        base_value = float(base.value.iloc[0])
        active = part[part.condition != 'BASE']
        lookup = active.set_index(['condition', 'norm_control', 'current_arm']).value
        assert lookup.index.is_unique

        def add(condition, norm, arm, contrast, value):
            rows.append(dict(zip(KEYS, key)) | dict(condition=condition,
                norm_control=norm, current_arm=arm, contrast=contrast,
                difference=float(value)))

        for x in active.itertuples():
            reference = ('native', x.norm_control, x.current_arm)
            assert reference in lookup.index
            add(x.condition, x.norm_control, x.current_arm, 'condition_minus_BASE',
                x.value - base_value)
            add(x.condition, x.norm_control, x.current_arm, 'condition_minus_native',
                x.value - lookup.loc[reference])
            raw = (x.condition, 'raw', x.current_arm)
            if x.norm_control != 'raw':
                assert raw in lookup.index
                add(x.condition, x.norm_control, x.current_arm,
                    'normmatched_minus_raw', x.value - lookup.loc[raw])
        gaps = {}
        for (condition, norm), values in active.groupby(['condition', 'norm_control']):
            arms = set(values.current_arm)
            assert arms <= {'SN', 'SB'}
            if arms != {'SN', 'SB'}:
                continue
            gap = lookup.loc[condition, norm, 'SB'] - lookup.loc[condition, norm, 'SN']
            native_gap = lookup.loc['native', norm, 'SB'] - lookup.loc['native', norm, 'SN']
            gaps[condition, norm] = gap
            add(condition, norm, 'SB_minus_SN', 'current_q_gap', gap)
            add(condition, norm, 'SB_minus_SN', 'history_by_current_q_interaction',
                gap - native_gap)
        for (condition, norm), gap in gaps.items():
            if norm != 'raw':
                add(condition, norm, 'SB_minus_SN', 'normmatched_minus_raw_q_gap',
                    gap - gaps[condition, 'raw'])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert r.base.sha(r.__file__) == '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'
    assert json.loads((r.OUT / 'analysis_rules_frozen.json').read_text())['status'] == 'FROZEN'
    manifest = json.loads(args.manifest.read_text())
    records, sources = [], []
    expected_seeds = [2022, 2023, 2024, 2025]
    for directory in map(Path, manifest['parents']):
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['test_access'] is False
        assert receipt['source_sha'] == '9127a433f34e1143d5280112c7f54f08f592fb82f1ac429d5bcbdb22922aa9d5'
        assert receipt['stages'] == [1, 10, 40]
        for source, sha in zip(receipt['source_runs'], receipt['source_receipt_sha']):
            assert r.base.sha(Path(source) / 'receipt.json') == sha
        config = json.loads((Path(receipt['source_runs'][0]) / 'config.json').read_text())
        p = np.asarray(config['p'])
        summary = pd.read_csv(directory / 'summary.csv')
        cells = pd.read_csv(directory / 'cells.csv')
        metadata = ['seed', 'epoch', 'current_arm', 'condition', 'norm_control',
                    'norm_informative']
        assert summary.seed.unique().tolist() == [receipt['seed']]
        assert not summary.duplicated(metadata).any()
        assert set(summary.epoch) == {1, 10, 40}
        expected_conditions = {'native', 'swap_m', 'swap_v', 'full_shadow',
                               'beta1_zero_v_only', 'beta2_zero', 'm_only'}
        for epoch in (1, 10, 40):
            assert len(summary[(summary.epoch == epoch) &
                               (summary.condition == 'BASE')]) == 1
            for arm in ('SN', 'SB'):
                raw = summary[(summary.epoch == epoch) &
                              (summary.current_arm == arm) &
                              (summary.norm_control == 'raw')]
                assert set(raw.condition) == expected_conditions and len(raw) == 7
        assert set(cells[metadata].itertuples(index=False, name=None)) == set(
            summary[metadata].itertuples(index=False, name=None))
        assert (cells.groupby(metadata).split.nunique() == 2).all()
        assert set(cells.split) == {'selection', 'diagnostic'}
        for key, part in cells.groupby(metadata + ['split']):
            assert len(part) == 6 and not part.duplicated(['domain', 'task']).any()
            assert set(part.domain) == {0, 1, 2} and set(part.task) == {0, 1}
            info = dict(zip(metadata + ['split'], key))
            reference = summary
            for field in metadata:
                reference = reference[reference[field] == info[field]]
            assert len(reference) == 1
            variables = {}
            for metric in ('auc', 'logloss'):
                macro = float(part[metric].mean())
                assert abs(macro - reference.iloc[0][info['split']+'_'+metric]) < 1e-12
                variables['macro_'+metric] = macro
                variables['sample_weighted_cell_'+metric] = float(np.average(part[metric], weights=part.n))
                variables['train_objective_weighted_cell_'+metric] = float(np.average(part[metric], weights=p[part.domain.to_numpy(int)]))
                for domain, subset in part.groupby('domain'):
                    variables[f'd{domain}_{metric}'] = float(subset[metric].mean())
                for task, subset in part.groupby('task'):
                    variables[f't{task}_{metric}'] = float(subset[metric].mean())
                for cell in part.itertuples():
                    variables[f'd{cell.domain}t{cell.task}_{metric}'] = float(getattr(cell, metric))
            records.extend(dict(info, variable=variable, value=value)
                           for variable, value in variables.items())
        for row in summary.to_dict('records'):
            records.append({key: row[key] for key in metadata} |
                dict(split='train', variable='train_empirical_objective',
                     value=float(row['train_empirical_objective'])))
        sources.append(dict(seed=receipt['seed'], parent=str(directory),
            hashes={name: r.base.sha(directory / name)
                    for name in ('receipt.json', 'summary.csv', 'cells.csv')}))
    assert sorted(x['seed'] for x in sources) == expected_seeds
    frame = pd.DataFrame(records)
    assert not frame.duplicated(KEYS + ['condition', 'norm_control', 'current_arm']).any()
    values = state_contrasts(frame)
    keys = ['epoch', 'split', 'variable', 'condition', 'norm_control',
            'current_arm', 'contrast']
    summaries = []
    for key, part in values.groupby(keys):
        assert not part.seed.duplicated().any()
        seeds = sorted(part.seed.tolist())
        item = dict(zip(keys, key)) | dict(n_observed=len(seeds), n_required=4,
            observed_seeds=seeds, all_seeds_present=seeds == expected_seeds)
        if item['all_seeds_present']:
            item.update(r.paired_summary(part.sort_values('seed').difference))
        summaries.append(item)
    args.out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out / 'absolute_seed_readouts.csv', index=False)
    values.to_csv(args.out / 'seed_contrasts.csv', index=False)
    r.base.dump(args.out / 'contrast_summary.json', summaries)
    pd.DataFrame([{key: row[key] for key in keys + ['n_observed', 'n_required',
        'observed_seeds', 'all_seeds_present']} for row in summaries]).to_csv(
            args.out / 'support_counts.csv', index=False)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        mode='SHADOW_LOCAL_COUNTERFACTUAL_STATISTICS', source_sha=r.base.sha(__file__),
        paired_statistics_source_sha=r.base.sha(r.__file__),
        analysis_rules_sha=r.base.sha(r.OUT / 'analysis_rules_frozen.json'),
        manifest_sha=r.base.sha(args.manifest), sources=sources,
        design_sha=r.base.sha(r.OUT / 'V11_shadow_statistics_design.json'),
        independent_review=False, test_access=False,
        hashes={p.name: r.base.sha(p) for p in args.out.iterdir() if p.is_file()},
        scope='Local effects along nativeSNtheta, one heldout draw per q/stage/seed. '
              'Four training seeds per CI, never cell/stage pseudoreplicates; '
              'partial norm controls have no seed CI. Not long-run necessity.'))


if __name__ == '__main__':
    main()
