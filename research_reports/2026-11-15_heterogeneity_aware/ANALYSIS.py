"""Reconcile the completed pilot and export report tables/figures; no training."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.heterogeneity_aware_20261115.selection import paired

ATTEMPT = ROOT/'artifacts/heterogeneity_aware_20261115/attempt1'
OUT = HERE/'tables'
OUT.mkdir(exist_ok=True)
FIG = HERE/'figures'
FIG.mkdir(exist_ok=True)
ARMS = ['CLEAN', 'A', 'B', 'C', 'RANDOM', 'GAUSSIAN']
CELLS = [f'd{d}t{t}' for d in range(3) for t in range(2)]
SEEDS = list(range(2022, 2027))


def md(frame):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.9f}'
        return str(v)
    rows = [list(frame.columns)] + [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |', '| '+' | '.join(['---']*len(rows[0]))+' |'] +
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])


records = [json.loads(p.read_text()) for p in sorted((ATTEMPT/'runs').glob('*/result.json'))]
assert len(records) == 60
assert not list((ATTEMPT/'runs').glob('*/attempt*/failure.json'))
assert not list((ATTEMPT/'test_endpoints').glob('*.json'))
manifest = json.loads((ATTEMPT/'source_receipt.json').read_text())
source_hash = manifest['sha256']
curves, diagnostic_rows, counts = [], [], {}
for result in records:
    job = result['job']; dataset = job['dataset']; arm = job['config']['arm']; seed = job['seed']
    assert job['source_hash'] == source_hash and result['epochs'] == 50
    assert result['global_step'] == (1000 if dataset == 'movielens' else 2450)
    counts.setdefault((dataset, arm), set()).add(seed)
    directory = ATTEMPT/'runs'/job['key']/'attempt1'
    epochs = sorted((json.loads(p.read_text()) for p in directory.glob('epoch_*.json')), key=lambda r:r['epoch'])
    assert [r['epoch'] for r in epochs] == list(range(1, 51))
    for kind in ('online', 'fixed_ema'):
        chosen = max(epochs, key=lambda r:r['validation'][kind]['macro_auc'])
        assert chosen['epoch'] == result['selected_epochs'][kind]
        assert chosen['validation'][kind] == result['selected_validation'][kind]
        assert chosen['model_hash' if kind == 'online' else 'ema_hash'] == result['selected_hashes'][kind]
    for epoch in epochs:
        for kind, values in epoch['validation'].items():
            curves.append(dict(dataset=dataset, arm=arm, seed=seed, epoch=epoch['epoch'], kind=kind, **values))
    totals = dict(proposed_pairs=0, accepted_rows=0, skipped_pairs=0, backtracking_halvings=0,
                  capped_pairs=0, control_fallback=0, optimizer_updates=0, ema_updates=0)
    weighted = {}; maxima = {}; batches = 0; margin_pairs = 0; margin_change = 0.; sensitivity_weight = 0
    for line in (directory/'batches.jsonl').open():
        batch = json.loads(line); batches += 1
        totals['optimizer_updates'] += batch['optimizer_updates']; totals['ema_updates'] += batch['ema_updates']
        diag = batch['diagnostics']; n = diag.get('accepted_rows', 0)
        for key in totals.keys()-{'optimizer_updates', 'ema_updates'}:
            totals[key] += diag.get(key, 0)
        for key in ('delta_norm_mean', 'non_target_drift_mean'):
            weighted[key] = weighted.get(key, 0.) + diag.get(key, 0.)*n
        if 'q_norm_mean' in diag:
            weight = 2*diag['proposed_pairs']; sensitivity_weight += weight
            for key in ('q_norm_mean','g_norm_mean'):
                weighted[key] = weighted.get(key, 0.) + diag[key]*weight
        for key in ('delta_norm_max', 'non_target_drift_max', 'projection_residual_max'):
            maxima[key] = max(maxima.get(key, 0.), diag.get(key, 0.))
        if 'margin_before_mean' in diag:
            margin_pairs += n/2
            margin_change += (diag['margin_after_mean']-diag['margin_before_mean'])*n/2
    assert batches == totals['optimizer_updates'] == totals['ema_updates'] == result['global_step']
    diagnostic_rows.append(dict(dataset=dataset, arm=arm, seed=seed, batches=batches, **totals, **maxima,
        **{k:v/max(sensitivity_weight if k in ('q_norm_mean','g_norm_mean') else totals['accepted_rows'], 1)
           for k,v in weighted.items()},
        skip_fraction=totals['skipped_pairs']/max(totals['proposed_pairs'], 1),
        actual_margin_change=margin_change/max(margin_pairs, 1)))
assert len(counts) == 12 and all(v == set(SEEDS) for v in counts.values())
curves = pd.DataFrame(curves); diagnostics = pd.DataFrame(diagnostic_rows)
curves.to_csv(OUT/'VALIDATION_CURVES.csv', index=False)
diagnostics.to_csv(OUT/'DIAGNOSTICS_SEEDS.csv', index=False)
diagnostics.groupby(['dataset','arm']).mean(numeric_only=True).drop(columns='seed').to_csv(OUT/'DIAGNOSTICS_MEANS.csv')

stages = curves.assign(stage=pd.cut(curves.epoch, [0,10,30,50], labels=['E01-10','E11-30','E31-50']))
stages = stages.groupby(['dataset','arm','seed','kind','stage'], observed=True)[['macro_auc','macro_logloss']].mean().reset_index()
stage_pairs = []
for (dataset, arm, kind, stage), group in stages.groupby(['dataset','arm','kind','stage'], observed=True):
    if arm == 'CLEAN':
        continue
    reference = stages[(stages.dataset == dataset)&(stages.arm == 'CLEAN')&(stages.kind == kind)&(stages.stage == stage)]
    joined = group.merge(reference, on='seed', suffixes=('', '_clean'))
    for metric in ('macro_auc','macro_logloss'):
        stat = paired(joined.sort_values('seed')[metric]-joined.sort_values('seed')[metric+'_clean'])
        stage_pairs.append(dict(dataset=dataset, arm=arm, kind=kind, stage=stage, metric=metric,
            mean=stat['mean'], sample_sd=stat['sample_sd'], low=stat['ci95'][0], high=stat['ci95'][1], positive_seeds=stat['positive_seeds']))
pd.DataFrame(stage_pairs).to_csv(OUT/'TRAINING_STAGE_PAIRS.csv', index=False)

endpoints = pd.read_csv(ATTEMPT/'tables/ENDPOINTS.csv')
selected = endpoints[endpoints.endpoint == 'selected'].copy()
paired_table = pd.read_csv(ATTEMPT/'tables/PAIRED_SUMMARY.csv')
main_pairs = paired_table[(paired_table.kind == 'fixed_ema')&(paired_table.reference_kind == 'fixed_ema')&
                         (paired_table.arm.isin(['A','B','C']))&(paired_table.control == 'CLEAN')]
main_pairs.to_csv(OUT/'PRIMARY_PAIRED_COMPARISONS.csv', index=False)
result_tables = ['# 汇总数表', '', '全部为本轮 validation 指标；独立统计单位为五个训练 seed。Traffic 为验证域人数加权 cell 指标，Pooled 为逐任务跨域池化后平均。', '']
for kind in ('fixed_ema','online'):
    rows = []
    for (dataset,arm), group in selected[selected.kind == kind].groupby(['dataset','arm']):
        item = dict(dataset=dataset,arm=arm)
        for label, a, b in [('Macro','macro_auc','macro_logloss'),
                            ('Traffic','sample_weighted_cell_auc','sample_weighted_cell_logloss'),
                            ('Pooled','pooled_task_mean_auc','pooled_logloss')]:
            item[label] = f'{group[a].mean():.9f} / {group[b].mean():.9f}'
        item['AUC_SD'] = group.macro_auc.std(ddof=1)
        item['LL_SD'] = group.macro_logloss.std(ddof=1)
        item['Worst_cell_AUC'] = group.worst_cell_auc.mean()
        rows.append(item)
    result_tables += [f'## Selected {kind}', '', md(pd.DataFrame(rows)), '']
    rows=[]
    for (dataset,arm), group in selected[selected.kind == kind].groupby(['dataset','arm']):
        item=dict(dataset=dataset,arm=arm)
        item.update({c:f'{group[c+"_auc"].mean():.9f} / {group[c+"_logloss"].mean():.9f}' for c in CELLS})
        rows.append(item)
    result_tables += [f'## 六格绝对值 / {kind}', '', md(pd.DataFrame(rows)), '']
cell_deltas=paired_table[(paired_table.kind=='fixed_ema')&(paired_table.reference_kind=='fixed_ema')&
                        (paired_table.control=='CLEAN')&paired_table.metric.isin([c+'_auc' for c in CELLS])]
result_tables += ['## 六格 AUC 配对差 / fixed EMA − clean fixed EMA', '',
                  md(cell_deltas.pivot(index=['dataset','arm'],columns='metric',values='mean').reset_index()), '']
result_tables += ['## 主比较配对统计', '', md(main_pairs[main_pairs.metric.isin(['macro_auc','macro_logloss'])]
    [['dataset','arm','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']]), '']
(HERE/'RESULT_TABLES.md').write_text('\n'.join(result_tables))

population = []
for dataset in ('movielens','kuairand'):
    data = json.loads((ATTEMPT/f'{dataset}_data_manifest.json').read_text())
    clean = selected[(selected.dataset == dataset)&(selected.arm == 'CLEAN')&(selected.kind == 'fixed_ema')]
    for d in range(3):
        joint = data['train_joint_labels'][str(d)]; n = data['train_domain_counts'][d]
        for t in range(2):
            population.append(dict(dataset=dataset, domain=d, task=t, train_count=n,
                train_positive_rate=sum(v for label,v in joint.items() if label[t] == '1')/n,
                validation_count=int(clean[f'd{d}t{t}_count'].iloc[0]),
                validation_positive_rate=clean[f'd{d}t{t}_positive'].iloc[0]/clean[f'd{d}t{t}_count'].iloc[0],
                clean_ema_auc=clean[f'd{d}t{t}_auc'].mean(), clean_ema_logloss=clean[f'd{d}t{t}_logloss'].mean()))
pd.DataFrame(population).to_csv(OUT/'POPULATION_AND_REFERENCE.csv', index=False)

appendix = ['# 逐 seed 数值附录', '', '全部为 validation-selected 验证指标。每格为 AUC / 未缩放 LogLoss；epoch 和 model_hash 在原始 ENDPOINTS.csv 中对应。', '']
for (dataset,kind), group in selected.groupby(['dataset','kind']):
    appendix += [f'## {dataset} / {kind}', '']
    rows = []
    for _, row in group.sort_values(['arm','seed']).iterrows():
        item = dict(arm=row.arm, seed=int(row.seed), epoch=int(row.epoch),
                    Macro=f'{row.macro_auc:.9f} / {row.macro_logloss:.9f}')
        item.update({cell:f'{row[cell+"_auc"]:.9f} / {row[cell+"_logloss"]:.9f}' for cell in CELLS})
        item['Traffic'] = f'{row.sample_weighted_cell_auc:.9f} / {row.sample_weighted_cell_logloss:.9f}'
        item['Pooled'] = f'{row.pooled_task_mean_auc:.9f} / {row.pooled_logloss:.9f}'
        rows.append(item)
    appendix += [md(pd.DataFrame(rows)), '']
appendix += ['## 主比较逐 seed 差值', '', 'Δ = 候选 fixed EMA − CLEAN fixed EMA。', '']
seed_pairs = pd.read_csv(ATTEMPT/'tables/PAIRED_SEEDS.csv')
subset = seed_pairs[(seed_pairs.kind == 'fixed_ema')&(seed_pairs.reference_kind == 'fixed_ema')&
                    (seed_pairs.arm.isin(['A','B','C']))&(seed_pairs.control == 'CLEAN')&
                    seed_pairs.metric.isin(['macro_auc','macro_logloss'])]
appendix += [md(subset.pivot(index=['dataset','arm','seed'], columns='metric', values='delta').reset_index()), '']
(HERE/'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix))

colors = dict(zip(ARMS, ['black','#b22222','#e68a00','#1765ac','#6f4a8e','#3a9152']))
fig, axes = plt.subplots(2,2,figsize=(12,8))
for i,dataset in enumerate(('movielens','kuairand')):
    for j,metric in enumerate(('macro_auc','macro_logloss')):
        ax = axes[i,j]
        for arm in ARMS:
            values = curves[(curves.dataset == dataset)&(curves.arm == arm)&(curves.kind == 'fixed_ema')].groupby('epoch')[metric]
            mean, sd = values.mean(), values.std(ddof=1)
            ax.plot(mean.index,mean,color=colors[arm],label=arm)
            ax.fill_between(mean.index,mean-sd,mean+sd,color=colors[arm],alpha=.09)
        ax.set(title=f'{dataset}: validation {metric}', xlabel='Epoch', ylabel=metric)
        ax.grid(alpha=.2)
axes[0,0].legend(ncol=3,fontsize=8)
fig.suptitle('Fixed EMA: mean +/- sample SD across five seeds')
fig.tight_layout();fig.savefig(FIG/'VALIDATION_CURVES.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,('macro_auc','macro_logloss')):
    frame=main_pairs[main_pairs.metric==metric].sort_values(['dataset','arm'])
    y=frame['mean'].to_numpy();error=np.vstack((y-frame.ci95_low.to_numpy(),frame.ci95_high.to_numpy()-y))
    ax.bar(range(len(frame)),y,yerr=error,capsize=4,color=[colors[a] for a in frame.arm])
    ax.set_xticks(range(len(frame)),[f'{dict(movielens="ML",kuairand="KR")[d]} {a}' for d,a in zip(frame.dataset,frame.arm)])
    ax.axhline(0,color='black',linewidth=.8);ax.set(title=metric, ylabel='Candidate minus clean fixed EMA')
    ax.grid(axis='y',alpha=.2)
fig.suptitle('Selected validation: paired five-seed mean and 95% t interval')
fig.tight_layout();fig.savefig(FIG/'PAIRED_PRIMARY.png',dpi=180);plt.close(fig)

decision=json.loads((ATTEMPT/'PILOT_DECISION.json').read_text())
assert decision['status']=='NO_CONTINUATION_SIGNAL' and decision['routes']==[]
completion=dict(status='COMPLETED', scientific_decision=decision['status'], trajectories=60, epochs=3000,
    optimizer_updates=sum(r['global_step'] for r in records), ema_updates=sum(r['global_step'] for r in records),
    seed_ids=SEEDS, source_hash=source_hash, test_accessed=False,
    phases=dict(design='COMPLETED',implementation='COMPLETED',review='PASS_WITH_FIXES',smoke='PASS',pilot='COMPLETED',
        hpo='NOT_TRIGGERED',formal_confirmation='NOT_TRIGGERED',test='NOT_TRIGGERED',targeted_ablation='NOT_TRIGGERED'),
    validation_checkpoint_reconciliation='60 trajectories x 2 readouts: earliest strict maximum verified',
    training_failures=0, missing_runs=0)
(HERE/'COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n')
print(json.dumps(completion,indent=2))
print(diagnostics.groupby(['dataset','arm'])[['skip_fraction','delta_norm_mean','non_target_drift_mean',
    'non_target_drift_max','projection_residual_max','actual_margin_change','control_fallback']].mean().to_string())
