"""Primary scientific analysis of completed V2 runs; no training or selection."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / 'v2'
OUT.mkdir(exist_ok=True)
NAT = Path('/home_new/xzf_natural_movielens_reassessment/V2')
HIST = Path('/home/xzf/Code/phase0-3-20261016-git/research_plans/2026-10-16_v2/execution_20261016/runs')
SEEDS = (2022, 2023, 2024, 2025)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

frames, provenance, missing = [], [], []
for population, root in [('movielens_natural', NAT),
                         ('movielens_artificial_historical', HIST / 'movielens'),
                         ('kuairand_natural_historical', HIST / 'kuairand')]:
    for optimizer in ('adam', 'sgd'):
        for arm in ('SN', 'SB'):
            for seed in SEEDS:
                run = root / optimizer / arm / f'seed{seed}' if root == NAT else root / optimizer / f'seed{seed}' / arm / 'attempt1'
                path = run / 'metrics_test.csv'
                if not path.exists():
                    missing.append(dict(population=population, optimizer=optimizer, arm=arm, seed=seed, path=str(path)))
                    continue
                config = json.loads((run / 'config.json').read_text())
                if root == NAT:
                    receipt = json.loads((run / 'run_manifest.json').read_text())
                    assert receipt['status'] == 'COMPLETE' and receipt['epochs_completed'] == 50
                    test = json.loads((run / 'test_access_receipt.json').read_text())
                    assert test['status'] == 'COMPLETE'
                f = pd.read_csv(path)
                assert not f.duplicated(['checkpoint', 'domain', 'task']).any()
                assert f.groupby('checkpoint').size().eq(6).all()
                f['population'] = population
                f['run_dir'] = str(run)
                f['stage'] = f.checkpoint.map(lambda x: 'selected' if x == 'selected' else 'final' if x == 'epoch50' else x)
                p = np.asarray(config['alpha'], dtype=float)
                assert abs(p.sum() - 1) < 1e-12
                f['train_cell_weight'] = f.domain.map(dict(enumerate(p))) / 2
                frames.append(f)
                provenance.append(dict(population=population, optimizer=optimizer, arm=arm, seed=seed,
                                       config_sha256=sha(run / 'config.json'), metrics_sha256=sha(path), path=str(path)))
assert sum(x['population'] == 'movielens_natural' for x in provenance) == 16
cell = pd.concat(frames, ignore_index=True)
cell.to_csv(OUT / 'seed_cell_metrics.csv', index=False)
pd.DataFrame(provenance).to_csv(OUT / 'source_index.csv', index=False)
(OUT / 'missing_historical_artifacts.json').write_text(json.dumps(missing, indent=2))

keys = ['population', 'optimizer', 'arm', 'seed', 'stage']
aggregate = []
for key, g in cell.groupby(keys, sort=False):
    row = dict(zip(keys, key))
    row.update(epoch=int(g.epoch.iloc[0]), steps=int(g.steps.iloc[0]),
               macro_auc=g.auc.mean(), macro_logloss=g.logloss.mean(),
               train_p_cell_auc=float((g.auc * g.train_cell_weight).sum()),
               train_p_cell_logloss=float((g.logloss * g.train_cell_weight).sum()),
               eval_p_cell_auc=float(np.average(g.auc, weights=g.n)),
               eval_p_cell_logloss=float(np.average(g.logloss, weights=g.n)))
    aggregate.append(row)
macro = pd.DataFrame(aggregate)
macro.to_csv(OUT / 'seed_macro_metrics.csv', index=False)
metrics = ['macro_auc', 'macro_logloss', 'train_p_cell_auc', 'train_p_cell_logloss',
           'eval_p_cell_auc', 'eval_p_cell_logloss']
pairs, summary = [], []
for key, g in macro.groupby(['population', 'optimizer', 'stage']):
    sn, sb = g[g.arm == 'SN'].set_index('seed'), g[g.arm == 'SB'].set_index('seed')
    common = sorted(set(sn.index) & set(sb.index))
    for metric in metrics:
        values = np.asarray([sb.loc[s, metric] - sn.loc[s, metric] for s in common], dtype=float)
        for seed, value in zip(common, values):
            pairs.append(dict(population=key[0], optimizer=key[1], stage=key[2], metric=metric,
                              seed=seed, sn=sn.loc[seed, metric], sb=sb.loc[seed, metric], delta=value))
        n = len(values)
        sd = float(values.std(ddof=1)) if n > 1 else float('nan')
        half = t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else float('nan')
        summary.append(dict(population=key[0], optimizer=key[1], stage=key[2], metric=metric,
                            n=n, sn_mean=sn.loc[common, metric].mean(), sb_mean=sb.loc[common, metric].mean(),
                            delta_mean=values.mean(), delta_sd=sd, descriptive_t95_low=values.mean()-half,
                            descriptive_t95_high=values.mean()+half, positive=int((values>0).sum()),
                            negative=int((values<0).sum()), zero=int((values==0).sum())))
pd.DataFrame(pairs).to_csv(OUT / 'paired_seed_deltas.csv', index=False)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / 'paired_summary.csv', index=False)

ckeys = ['population', 'optimizer', 'seed', 'stage', 'domain', 'task']
sn = cell[cell.arm == 'SN'].set_index(ckeys)
sb = cell[cell.arm == 'SB'].set_index(ckeys)
joined = sn[['auc','logloss','n','n_pos']].join(sb[['auc','logloss']], lsuffix='_sn', rsuffix='_sb', how='inner').reset_index()
joined['delta_auc'] = joined.auc_sb - joined.auc_sn
joined['delta_logloss'] = joined.logloss_sb - joined.logloss_sn
joined.to_csv(OUT / 'paired_cell_deltas.csv', index=False)
for marginal in ('domain', 'task'):
    groups = ['population', 'optimizer', 'seed', 'stage', marginal]
    joined.groupby(groups)[['auc_sn','auc_sb','delta_auc','logloss_sn','logloss_sb','delta_logloss']].mean().reset_index().to_csv(OUT / f'paired_{marginal}_deltas.csv', index=False)

selected = summary[(summary.stage.isin(['selected','final'])) & (summary.metric.isin(['macro_auc','macro_logloss','train_p_cell_auc']))]
print(selected.to_string(index=False))

# Recalculate the two reported deployment endpoints from their saved predictions.
# This checks the evidence used in the report rather than adding model tests.
binding = np.load('/home_new/xzf_natural_movielens_reassessment/data/movielens_natural.npz')
raw_audit = []
for record in provenance:
    if record['population'] != 'movielens_natural':
        continue
    run = Path(record['path']).parent
    for stage, label in [('selected','selected'), ('final','epoch50')]:
        path = run / f'test_{label}.npz'
        with np.load(path) as z:
            assert np.array_equal(z['row_ids'], binding['test_row_ids'])
            assert np.array_equal(z['labels'], binding['test_y'])
            assert np.array_equal(z['domains'], binding['test_x'][:,-1])
            saved = cell[(cell.population == record['population']) & (cell.optimizer == record['optimizer']) &
                         (cell.arm == record['arm']) & (cell.seed == record['seed']) & (cell.stage == stage)]
            for row in saved.itertuples():
                ix = z['domains'] == row.domain
                y = z['labels'][ix,row.task].astype(np.float64)
                p = z['predictions'][ix,row.task].astype(np.float64)
                auc = roc_auc_score(y,p)
                p = np.clip(p,1e-12,1-1e-12)
                loss = float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
                raw_audit.append(dict(optimizer=row.optimizer,arm=row.arm,seed=row.seed,stage=stage,
                                      domain=row.domain,task=row.task,auc_residual=auc-row.auc,
                                      logloss_residual=loss-row.logloss,prediction_path=str(path),sha256=sha(path)))
audit = pd.DataFrame(raw_audit)
assert len(audit) == 192
assert audit.auc_residual.abs().max() < 1e-12
assert audit.logloss_residual.abs().max() < 1e-12
audit.to_csv(OUT/'prediction_metric_recalculation.csv',index=False)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True)
labels = {'movielens_natural':'ML natural (new)', 'movielens_artificial_historical':'ML artificial (history)',
          'kuairand_natural_historical':'KR natural (history)'}
for ax, optimizer in zip(axes,('adam','sgd')):
    rows = summary[(summary.optimizer==optimizer)&(summary.metric=='macro_auc')&summary.stage.isin(['selected','final'])]
    names=[]
    for index, row in enumerate(rows.itertuples()):
        y = len(rows)-1-index
        vals = [r['delta'] for r in pairs if r['population']==row.population and r['optimizer']==optimizer and r['stage']==row.stage and r['metric']=='macro_auc']
        ax.scatter(vals,np.full(len(vals),y),s=16,color='#64748b',alpha=.65)
        ax.errorbar(row.delta_mean,y,xerr=[[row.delta_mean-row.descriptive_t95_low],[row.descriptive_t95_high-row.delta_mean]],fmt='o',color='#0369a1',capsize=3)
        names.append(labels[row.population]+' / '+row.stage)
    ax.set_yticks(range(len(names)-1,-1,-1),names)
    ax.axvline(0,color='black',lw=.7)
    ax.set_title(optimizer.upper());ax.set_xlabel('SB minus SN: test macro AUC')
    ax.grid(axis='x',alpha=.2)
fig.suptitle('V2: paired seeds and descriptive t95 intervals (n=4 per condition)',fontsize=11)
fig.tight_layout()
for suffix in ('png','pdf'):fig.savefig(OUT/('v2_population_stage_comparison.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
