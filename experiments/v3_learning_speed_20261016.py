"""Exact saved-step batch comparisons and complete descriptive first-hit curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    curves, hits, sources = {}, [], []
    for group, directories in m['groups'].items():
        curves[group] = {}
        for directory in map(Path, directories):
            receipt = json.loads((directory/'receipt.json').read_text())
            assert receipt['status'] == 'COMPLETE'
            cells = pd.read_csv(directory/'cell_trajectory.csv')
            assert r.base.sha(directory/'cell_trajectory.csv') == receipt['hashes']['cell_trajectory.csv']
            seed = receipt['seed']
            curves[group][seed] = {}
            for arm_index, (arm, source) in enumerate(zip(('SN','SB'), map(Path, receipt['source_runs']))):
                producer = json.loads((source/'receipt.json').read_text())
                assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
                assert r.base.sha(source/'receipt.json') == receipt['source_run_receipt_sha'][arm_index]
                assert r.base.sha(source/'trajectory.csv') == producer['artifact_hashes']['trajectory.csv']
                train = pd.read_csv(source/'trajectory.csv').set_index('epoch')
                assert list(train.index) == list(range(1,len(train)+1))
                points = cells[cells.arm==arm].groupby(['split','epoch'])[['auc','logloss']].mean()
                curves[group][seed][arm] = points.join(train[['steps','examples']], on='epoch')
                selection = points.loc['selection'].auc
                for metric, thresholds, values, maximize in [
                    ('selection_macro_auc', np.arange(50,91)/100, selection, True),
                    ('train_empirical_objective', np.arange(20,81)/100,
                     train.train_empirical_objective, False)]:
                    for threshold in thresholds:
                        crossing = values[values>=threshold] if maximize else values[values<=threshold]
                        epoch = None if crossing.empty else int(crossing.index[0])
                        hits.append(dict(group=group,seed=seed,arm=arm,metric=metric,threshold=threshold,
                            reached=epoch is not None,epoch=epoch,
                            steps=None if epoch is None else int(train.loc[epoch,'steps']),
                            examples=None if epoch is None else int(train.loc[epoch,'examples']),
                            censor_epoch=int(train.index.max()),censor_steps=int(train.steps.max()),
                            censor_examples=int(train.examples.max())))
            sources.append(dict(group=group,seed=seed,receipt_sha=r.base.sha(directory/'receipt.json')))
    rows = []
    for seed in sorted(curves['B0_sgd']):
        assert seed in curves['B2_sgd']
        for split in ('selection','diagnostic'):
            frames = {(g,arm):curves[g][seed][arm].loc[split].reset_index().set_index('steps')
                for g in ('B0_sgd','B2_sgd') for arm in ('SN','SB')}
            assert all(f.index.is_unique for f in frames.values())
            for group in ('B0_sgd','B2_sgd'):
                assert frames[group,'SN'].index.equals(frames[group,'SB'].index)
                assert np.array_equal(frames[group,'SN'][['epoch','examples']].to_numpy(),
                                      frames[group,'SB'][['epoch','examples']].to_numpy())
            common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
            for step in common:
                for metric in ('auc','logloss'):
                    values = {(g,arm):float(f.loc[step,metric]) for (g,arm),f in frames.items()}
                    sn, sb = frames['B0_sgd','SN'].loc[step], frames['B2_sgd','SN'].loc[step]
                    assert int(step) == step
                    b0 = values['B0_sgd','SB']-values['B0_sgd','SN']
                    b2 = values['B2_sgd','SB']-values['B2_sgd','SN']
                    rows.append(dict(seed=seed,split=split,steps=step,metric=metric,
                        B0_epoch=int(sn.epoch),B2_epoch=int(sb.epoch),B0_examples=int(sn.examples),
                        B2_examples=int(sb.examples),B0_SN=values['B0_sgd','SN'],B0_SB=values['B0_sgd','SB'],
                        B2_SN=values['B2_sgd','SN'],B2_SB=values['B2_sgd','SB'],
                        B0_gap=b0,B2_gap=b2,interaction=b2-b0))
    frame = pd.DataFrame(rows)
    frame.to_csv(a.out/'exact_matched_step_seed_readouts.csv',index=False)
    summary=[]
    for key, subset in frame.groupby(['split','steps','metric']):
        assert not subset.seed.duplicated().any()
        item=dict(split=key[0],steps=int(key[1]),metric=key[2],n_observed=len(subset),
            n_required=len(curves['B0_sgd']),all_seeds_present=len(subset)==len(curves['B0_sgd']))
        if item['all_seeds_present']:
            item.update(r.paired_summary(subset.sort_values('seed').interaction))
        summary.append(item)
    r.base.dump(a.out/'matched_step_summary.json',summary)
    pd.DataFrame(hits).to_csv(a.out/'descriptive_first_hit_curves.csv',index=False)
    r.base.dump(a.out/'receipt.json',dict(status='COMPLETE',source_sha=r.base.sha(__file__),
        manifest_sha=r.base.sha(a.manifest),sources=sources,test_access=False,
        scope='Exact saved-step intersections; examples differ between batch sizes. Full first-hit threshold curves '
              'are retrospective descriptive and right-censored, never a new discovery or condition gate.'))


if __name__ == '__main__':
    main()
