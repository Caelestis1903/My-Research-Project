"""Read-only V7 local counterfactual statistics; no training activation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v3_readout_20261016 as r


def attenuation(natural, balanced, hybrid):
    gap = balanced - natural
    return None if abs(gap) < 1e-8 else 1 - abs(hybrid - natural) / abs(gap)


def local_contrasts(frame):
    rows = []
    for key, part in frame.groupby(['seed', 'epoch', 'split', 'variable']):
        seed, epoch, split, variable = key
        values = part.set_index('condition').value.to_dict()
        assert len(values) == 7
        natural, balanced = values['SN'], values['SB']
        info = dict(seed=int(seed), epoch=int(epoch), split=split, variable=variable)
        def add(condition, contrast, value):
            rows.append(dict(info, condition=condition, contrast=contrast,
                             difference=value))
        for condition, value in values.items():
            if condition != 'BASE':
                add(condition, 'condition_minus_BASE', value - values['BASE'])
        add('SB', 'SB_minus_SN', balanced - natural)
        for block in 'CPRE':
            condition = 'SB_replace_' + block
            value = values[condition]
            add(condition, 'hybrid_minus_SN', value - natural)
            add(condition, 'hybrid_minus_SB', value - balanced)
            add(condition, 'attenuation', attenuation(natural, balanced, value))
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
    expected_conditions = {'BASE', 'SN', 'SB'} | {'SB_replace_' + b for b in 'CPRE'}
    for task in manifest['tasks']:
        command = task['command']
        directory = Path(command[command.index('--run-dir') + 1])
        source = Path(command[command.index('--source-run') + 1])
        receipt = json.loads((directory / 'receipt.json').read_text())
        producer = json.loads((source / 'receipt.json').read_text())
        config = json.loads((source / 'config.json').read_text())
        seed, epoch = receipt['seed'], receipt['epoch']
        assert receipt['status'] == 'COMPLETE' and receipt['mode'] == 'local'
        assert receipt['source_sha'] == '0e90d738016037ca67d9cf733803977f6a641b1aec654f1c8149b3ceceea313b'
        assert receipt['source_sha'] == r.base.sha(Path(__file__).with_name('v3_local_20261016.py'))
        assert receipt['freeze_sha'] == r.base.sha(r.OUT / 'analysis_rules_frozen.json')
        assert config == receipt['config'] and config['seed'] == seed
        assert config['optimizer'] == 'sgd' and config['lr'] == .3 and config['arm'] == 'SN'
        assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        checkpoint = source / f'epoch{epoch:03d}.pt'
        assert receipt['checkpoint'] == str(checkpoint)
        assert receipt['checkpoint_sha'] == r.base.sha(checkpoint) == producer['artifact_hashes'][checkpoint.name]
        assert r.base.sha(source / 'config.json') == producer['artifact_hashes']['config.json']
        summary, cells = [pd.read_csv(directory / name) for name in ('summary.csv', 'cells.csv')]
        assert len(summary) == 7 and set(summary.condition) == expected_conditions
        assert set(summary.seed) == {seed} and set(summary.epoch) == {epoch}
        assert len(cells) == 84 and set(cells.split) == {'selection', 'diagnostic'}
        assert set(cells.seed) == {seed} and set(cells.epoch) == {epoch}
        assert set(cells.condition) == expected_conditions
        assert not cells.duplicated(['condition', 'split', 'domain', 'task']).any()
        p = np.asarray(config['p'])
        for (condition, split), group in cells.groupby(['condition', 'split']):
            assert len(group) == 6 and set(group.domain) == {0, 1, 2} and set(group.task) == {0, 1}
            variables = {}
            reference = summary[summary.condition == condition].iloc[0]
            for metric in ('auc', 'logloss'):
                macro = float(group[metric].mean())
                assert abs(macro - reference[split + '_' + metric]) < 1e-12
                variables['macro_' + metric] = macro
                variables['sample_weighted_cell_' + metric] = float(np.average(group[metric], weights=group.n))
                variables['train_objective_weighted_cell_' + metric] = float(np.average(group[metric], weights=p[group.domain.to_numpy(int)]))
                variables.update({f'd{d}_{metric}': float(g[metric].mean()) for d, g in group.groupby('domain')})
                variables.update({f't{t}_{metric}': float(g[metric].mean()) for t, g in group.groupby('task')})
                variables.update({f'd{x.domain}t{x.task}_{metric}': float(getattr(x, metric)) for x in group.itertuples()})
            records.extend(dict(seed=seed, epoch=epoch, split=split,
                condition=condition, variable=k, value=v) for k, v in variables.items())
        records.extend(dict(seed=seed, epoch=epoch, split='train',
            condition=x.condition, variable='train_empirical_objective',
            value=float(x.train_empirical_objective)) for x in summary.itertuples())
        sources.append(dict(seed=seed, epoch=epoch, directory=str(directory),
            producer=str(source), producer_receipt_sha=r.base.sha(source / 'receipt.json'),
            checkpoint_sha=receipt['checkpoint_sha'], hashes={name: r.base.sha(directory / name)
                for name in ('receipt.json', 'summary.csv', 'cells.csv')}))
    assert sorted((x['seed'], x['epoch']) for x in sources) == [
        (s, e) for s in range(2022, 2026) for e in (1, 10, 40)]
    absolute = pd.DataFrame(records)
    assert not absolute.duplicated(['seed', 'epoch', 'split', 'variable', 'condition']).any()
    values = local_contrasts(absolute)
    summaries = []
    for key, group in values.groupby(['epoch', 'split', 'variable', 'condition', 'contrast']):
        finite = group[np.isfinite(group.difference)].sort_values('seed')
        item = dict(zip(['epoch', 'split', 'variable', 'condition', 'contrast'], key))
        item['epoch'] = int(item['epoch'])
        item.update(n_observed=len(finite), n_required=4,
                    observed_seeds=finite.seed.tolist())
        if finite.seed.tolist() == [2022, 2023, 2024, 2025]:
            item.update(r.paired_summary(finite.difference))
        summaries.append(item)
    screens = []
    for block in 'CPRE':
        stages = []
        for epoch in (1, 10, 40):
            part = values[(values.epoch == epoch) &
                (values.condition == 'SB_replace_' + block) &
                (values.contrast == 'attenuation')]
            auc = part[(part.split == 'selection') & (part.variable == 'macro_auc')].sort_values('seed')
            assert auc.seed.tolist() == [2022, 2023, 2024, 2025]
            passed = bool(np.isfinite(auc.difference).all() and (auc.difference >= .5).all())
            losses = {}
            for split, variable in [('selection', 'macro_logloss'), ('train', 'train_empirical_objective')]:
                loss = part[(part.split == split) & (part.variable == variable)].sort_values('seed')
                assert loss.seed.tolist() == [2022, 2023, 2024, 2025]
                losses[variable] = dict(seed_attenuation=[float(x) if np.isfinite(x) else None for x in loss.difference],
                    all4_informative_nonworsening=bool(np.isfinite(loss.difference).all() and (loss.difference >= 0).all()))
            stages.append(dict(epoch=epoch, selection_AUC_all4_at_least_half=passed,
                selection_AUC_seed_attenuation=[float(x) if np.isfinite(x) else None for x in auc.difference],
                loss_consistency_descriptive=losses))
        qualified = [x['epoch'] for x in stages if x['selection_AUC_all4_at_least_half']]
        screens.append(dict(block=block, qualified_AUC_stages=qualified,
            AUC_screen_pass=len(qualified) >= 2, stages=stages,
            full_training_authorized=False,
            scope='C/E genuinely shared; P mixed requires decomposition if positive; R alreadyprivate. Primary loss-consistency and scope review mandatory.'))
    args.out.mkdir(parents=True, exist_ok=False)
    absolute.to_csv(args.out / 'absolute_seed_readouts.csv', index=False)
    values.to_csv(args.out / 'seed_contrasts.csv', index=False)
    r.base.dump(args.out / 'contrast_summary.json', summaries)
    r.base.dump(args.out / 'selection_AUC_screens_and_loss_readouts.json', screens)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        source_sha=r.base.sha(__file__), manifest_sha=r.base.sha(args.manifest),
        design_sha=r.base.sha(r.OUT / 'V7_local_statistics_design.json'),
        analysis_rules_sha=r.base.sha(r.OUT / 'analysis_rules_frozen.json'),
        sources=sources, independent_review=False, test_access=False,
        no_automatic_training_activation=True,
        hashes={p.name: r.base.sha(p) for p in args.out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
