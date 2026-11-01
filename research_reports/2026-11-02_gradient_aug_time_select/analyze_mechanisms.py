"""Describe completed B1/B2 and A1-controller logs; seed remains the replicate unit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True, type=Path)
args = parser.parse_args()
out = Path(__file__).resolve().parent
protocol = json.loads((args.run_dir / 'resolved_config.json').read_text())
probabilities, projections, shrinkage, rates = [], [], [], []

def correlation(left, right):
    a, b = left.astype(float), right.astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else np.nan

for dataset in ('movielens', 'kuairand'):
    conf = protocol['datasets'][dataset]
    expected_rows = conf['train_domain_counts'][conf['augmented_domain']]
    steps = conf['steps_per_epoch']
    projection_frames = []
    for seed in (2022, 2024):
        # shortcut: the archived development runs use attempt1; follow the ledger for a run with retries.
        base = args.run_dir / 'runs/development' / dataset
        files = sorted((base / 'B1' / f'seed{seed}/attempt1/primary').glob('mask_probabilities_E*.npz'))
        assert len(files) == 50, (dataset, seed, 'B1 must finish before analysis')
        row_ids, previous, first = None, None, None
        for epoch, path in enumerate(files, 1):
            with np.load(path) as raw:
                order = np.argsort(raw['row_id'])
                ids, p = raw['row_id'][order], raw['probability'][order]
            assert len(ids) == expected_rows and np.unique(ids).size == expected_rows
            if row_ids is None:
                row_ids, first = ids, p.copy()
            else:
                assert np.array_equal(ids, row_ids)
            quantiles = np.quantile(p, [0, .1, .25, .5, .75, .9, 1])
            probabilities.append(dict(dataset=dataset, seed=seed, epoch=epoch, rows=len(p),
                mean_probability=float(p.mean()), std_probability=float(p.std()),
                adjacent_epoch_correlation=np.nan if previous is None else correlation(previous, p),
                epoch1_correlation=correlation(first, p),
                **dict(zip(('p_min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p_max'), quantiles))))
            previous = p
        path = base / 'B2' / f'seed{seed}/attempt1/primary/projection_stats.csv'
        frame = pd.read_csv(path)
        assert len(frame) == 4 * 50 * steps
        frame['seed'] = seed
        projection_frames.append(frame)
        for (task, group), table in frame.groupby(['task', 'group']):
            projections.append(dict(dataset=dataset, seed=seed, task=task, group=group,
                training_steps=len(table), conflict_fraction=float((table.coefficient < 0).mean()),
                mean_clean_norm=table.clean_norm.mean(), mean_delta_norm=table.delta_norm.mean(),
                mean_retained_norm=table.retained_norm.mean(), mean_removed_norm=table.removed_norm.mean(),
                retained_norm_sum_ratio=table.retained_norm.sum() / table.delta_norm.sum(),
                mean_cosine=table.cosine.mean(), min_projected_dot=table.projected_dot.min()))
        frozen = json.loads((base / 'A1' / f'seed{seed}/attempt1/rho_frozen.json').read_text())
        rho = np.asarray(frozen['adaptive'])
        assert frozen['multiset_equal'] and np.array_equal(np.sort(rho), np.sort(frozen['replay']))
        assert len(rho) == 50 * steps
        half_life = np.log(.5) / np.log(rho) / steps
        fixed = np.exp(np.log(.5) / 40) if dataset == 'movielens' else .995
        rates.append(dict(dataset=dataset, seed=seed, training_steps=len(rho), rho_min=rho.min(),
            rho_max=rho.max(), rho_mean=rho.mean(), rho_median=np.median(rho),
            fraction_below_fixed=float((rho < fixed).mean()), half_life_epoch_min=half_life.min(),
            half_life_epoch_median=np.median(half_life), half_life_epoch_max=half_life.max(),
            multiset_equal=True))
    full = pd.concat(projection_frames, ignore_index=True)
    for group, table in full.groupby('group'):
        numerator, denominator = table.retained_norm.sum(), table.delta_norm.sum()
        shrinkage.append(dict(dataset=dataset, group=group, seed_count=2, task_step_observations=len(table),
                              numerator=numerator, denominator=denominator,
                              frozen_shrink_ratio=numerator / denominator if denominator else 1.))

pd.DataFrame(probabilities).to_csv(out / 'b1_probability_stability.csv', index=False)
pd.DataFrame(projections).to_csv(out / 'b2_projection_by_seed_task_group.csv', index=False)
pd.DataFrame(shrinkage).to_csv(out / 'b2_frozen_shrink_recalculation.csv', index=False)
pd.DataFrame(rates).to_csv(out / 'a1_rho_by_seed.csv', index=False)
print('Wrote B1 stability, B2 projection/shrink and A1 rho tables from completed primary logs.')
