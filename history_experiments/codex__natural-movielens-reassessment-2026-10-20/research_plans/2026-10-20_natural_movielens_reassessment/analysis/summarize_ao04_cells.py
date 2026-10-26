"""Primary domain/task, fixed-weight and trade-off summaries after AO04 test closure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_core import interval

OUT = Path(__file__).resolve().parent / 'ao04'
ART = Path('/home_new/xzf_natural_movielens_reassessment')


def analyze():
    status = json.loads((OUT / 'analysis_status.json').read_text())
    assert status['complete'] and status['runs'] == 12
    cells = pd.read_csv(OUT / 'seed_cell_metrics.csv')
    pairs = pd.read_csv(OUT / 'paired_cell_deltas.csv')
    rows = []
    for axis in ('cell', 'domain', 'task'):
        columns = ['domain', 'task'] if axis == 'cell' else [axis]
        perseed = pairs.groupby(['contrast', 'checkpoint', 'seed'] + columns, as_index=False)[
            ['auc', 'logloss', 'baseline_auc']].mean()
        for key, g in perseed.groupby(['contrast', 'checkpoint'] + columns):
            assert set(g.seed) == {2022, 2023, 2024, 2025} and len(g) == 4
            for metric in ('auc', 'logloss'):
                rows.append(dict(zip(['contrast', 'checkpoint'] + columns, key), axis=axis,
                    metric=metric, reference_auc=g.baseline_auc.mean(), **interval(g[metric])))
    pd.DataFrame(rows).to_csv(OUT / 'four_seed_cell_domain_task_summary.csv', index=False)
    baseline = cells.groupby(['arm', 'checkpoint', 'domain', 'task'], as_index=False).agg(
        auc=('auc', 'mean'), logloss=('logloss', 'mean'), n=('n', 'mean'), n_pos=('n_pos', 'mean'))
    baseline.to_csv(OUT / 'four_seed_absolute_cell_means.csv', index=False)
    tradeoffs = []
    for key, g in pairs.groupby(['contrast', 'checkpoint', 'seed']):
        d = g.groupby('domain')[['auc', 'logloss']].mean()
        tradeoffs.append(dict(zip(['contrast', 'checkpoint', 'seed'], key),
            two_smaller_AUC_up_largest_down=bool(d.loc[0, 'auc'] > 0 and d.loc[1, 'auc'] > 0 and d.loc[2, 'auc'] < 0),
            smallest_AUC_down_two_larger_up=bool(d.loc[0, 'auc'] < 0 and d.loc[1, 'auc'] > 0 and d.loc[2, 'auc'] > 0),
            all_domain_AUC_up=bool((d.auc > 0).all()), all_domain_LL_improved=bool((d.logloss < 0).all())))
    pd.DataFrame(tradeoffs).to_csv(OUT / 'seed_domain_tradeoffs.csv', index=False)
    shares = json.loads((ART / 'data/movielens_natural.json').read_text())['data']['domain_shares']
    weighted = []
    for key, g in cells.groupby(['arm', 'checkpoint', 'seed']):
        train_p = g.domain.map({int(k): v for k, v in shares.items()}).to_numpy()
        for name, w in [('macro', np.ones(len(g))), ('train_p_cell', train_p), ('eval_p_cell', g.n.to_numpy())]:
            weighted.append(dict(zip(['arm', 'checkpoint', 'seed'], key), weighting=name,
                auc=np.average(g.auc, weights=w), logloss=np.average(g.logloss, weights=w)))
    weighted = pd.DataFrame(weighted)
    weighted.to_csv(OUT / 'seed_fixed_weight_metrics.csv', index=False)
    summaries = []
    for (checkpoint, weighting), g in weighted.groupby(['checkpoint', 'weighting']):
        for ca, ref in [('B1', 'B0'), ('B2', 'B0'), ('B1', 'B2')]:
            a = g[g.arm == ca].set_index('seed')
            b = g[g.arm == ref].set_index('seed')
            assert set(a.index) == set(b.index) == {2022, 2023, 2024, 2025}
            for metric in ('auc', 'logloss'):
                summaries.append(dict(checkpoint=checkpoint, weighting=weighting, contrast=ca+'-'+ref,
                    metric=metric, candidate_mean=a[metric].mean(), reference_mean=b[metric].mean(),
                    **interval(a[metric]-b[metric])))
    pd.DataFrame(summaries).to_csv(OUT / 'four_seed_fixed_weight_summary.csv', index=False)
    print('AO04 cell/domain/task and fixed-weight summaries complete.')


if __name__ == '__main__':
    analyze()
