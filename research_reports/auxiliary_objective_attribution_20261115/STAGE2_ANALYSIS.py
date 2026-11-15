"""Stage 2 numerical attachments from complete validation-only evidence."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ANALYSIS import records,native,describe,markdown_table,with_seed_deltas,group_descriptions
from experiments.auxiliary_objective_attribution_20261115.stage2.protocol import results,epoch_files,sha_file
from experiments.auxiliary_objective_attribution_20261115.stage2.tables import paired

SEEDS = [2022,2023,2024,2025,2026]
DATASETS = ['movielens','kuairand']
ARMS = ['CLEAN','NDP-A']
MACRO = ['macro_auc','macro_logloss']
CELL_AUC = [f'd{d}t{t}_auc' for d in range(3) for t in range(2)]
CELL_LL = [f'd{d}t{t}_logloss' for d in range(3) for t in range(2)]
AGGREGATES = ['macro_auc','macro_logloss','sample_weighted_cell_auc','sample_weighted_cell_logloss',
    'pooled_task_mean_auc','pooled_logloss','worst_cell_auc']


def phases(curves):
    frame = curves.copy()
    frame['phase'] = np.select([frame.epoch <= 10,frame.epoch <= 30],['E1-10','E11-30'],default='E31-50')
    means = frame.groupby(['dataset','arm','kind','seed','phase'],as_index=False,sort=False)[MACRO].mean()
    lookup = {(d,a,k,p):g.sort_values('seed') for (d,a,k,p),g in means.groupby(['dataset','arm','kind','phase'])}
    differences = []
    for dataset in DATASETS:
        for phase in ('E1-10','E11-30','E31-50'):
            for kind,reference_kind in (('online','online'),('fixed_ema','fixed_ema'),('fixed_ema','online')):
                a = lookup[dataset,'NDP-A',kind,phase]; b = lookup[dataset,'CLEAN',reference_kind,phase]
                assert a.seed.tolist() == b.seed.tolist() == SEEDS
                for metric in MACRO:
                    delta = (a[metric].to_numpy()-b[metric].to_numpy()).tolist()
                    differences.append(dict(dataset=dataset,phase=phase,arm='NDP-A',control='CLEAN',kind=kind,
                        reference_kind=reference_kind,metric=metric,seeds=SEEDS,deltas=delta,**paired(delta)))
    return means,pd.DataFrame(differences)


def natural_weights(weights):
    columns = ['batch_count','count','clean_count','view_count','natural_weight_mass','clean_weight_mass',
        'view_weight_mass','total_weight_mass','clean_numerator','view_numerator','clean_contribution',
        'view_contribution','total_contribution']
    seed = weights.groupby(['dataset','arm','seed','cell_label'],as_index=False,sort=False)[columns].sum()
    assert (seed['count'] == seed.clean_count).all() and (seed['count'] == seed.view_count).all()
    for metric in ('natural_weight_mass','clean_weight_mass','view_weight_mass','total_weight_mass',
                   'clean_contribution','view_contribution','total_contribution'):
        seed[metric+'_per_batch'] = seed[metric]/seed.batch_count
    np.testing.assert_allclose(seed.clean_weight_mass,.75*seed.natural_weight_mass,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(seed.view_weight_mass,.25*seed.natural_weight_mass,rtol=1e-12,atol=1e-12)
    seed['cell'] = seed.cell_label.str[:4]; seed['label'] = seed.cell_label.str[-1].astype(int)
    priors = []
    for key,group in seed.groupby(['dataset','arm','seed','cell'],sort=False):
        positive = group[group.label == 1]
        natural_count = int(group['count'].sum()); view_count = int(group.view_count.sum())
        priors.append(dict(zip(['dataset','arm','seed','cell'],key),natural_count=natural_count,view_count=view_count,
            natural_positive_rate=float(positive['count'].sum()/natural_count),
            view_positive_rate=float(positive.view_count.sum()/view_count)))
    summary = group_descriptions(seed,['dataset','arm','cell_label'],
        ['natural_weight_mass_per_batch','clean_weight_mass_per_batch','view_weight_mass_per_batch',
         'total_weight_mass_per_batch','clean_contribution_per_batch','view_contribution_per_batch','total_contribution_per_batch'])
    return seed,pd.DataFrame(priors),summary


def training_evidence(attempt,runs,output):
    training = []; first_batches = []; audit = []; task_sums = {}; cell_sums = {}
    for run in runs:
        job = run['job']
        if job['config']['arm'] == 'CLEAN':
            continue
        base = dict(dataset=job['dataset'],arm='NDP-A',seed=job['seed'],key=job['key'])
        for epoch,(_,data) in sorted(epoch_files(attempt,run).items()):
            ledger = data['objective_ledger']
            fields = ['loss','clean_loss','view_loss','probe_seconds','construction_seconds','training_seconds',
                'validation_seconds','seconds','ram_usage','examples_per_second','model_hash','ema_hash',
                'optimizer_hash','epoch_order_hash','support_log','support_log_sha256']
            training.append(dict(**base,epoch=epoch,global_step=data['global_step'],**{k:data[k] for k in fields},
                batch_count=ledger['batches'],B=ledger['B'],entries=ledger['entries'],total_weight_mass=ledger['total_weight_mass']))
            path = attempt/data['support_log']; assert sha_file(path) == data['support_log_sha256']
            batches = 0
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    value = json.loads(line); batches += 1; weights = value['ledger']; B = weights['B']
                    assert value['epoch'] == epoch and weights['entries'] == weights['denominator'] == 2*B
                    assert weights['clean_weight_mass'] == .75 and weights['view_weight_mass'] == .25 and weights['total_weight_mass'] == 1
                    error = max(abs(v['total_weight_mass']-v['count']/(2*B)) for v in weights['cells'].values())
                    assert error < 1e-12 and sum(v['count'] for v in weights['cells'].values()) == 2*B
                    row = dict(**base,epoch=epoch,global_step=value['global_step'],batch=value['batch'],
                        B=B,entries=2*B,row_order_hash=value['row_order_hash'],zero_delta_canonical=value['zero_delta_canonical'],
                        clean_weight_mass=.75,view_weight_mass=.25,total_weight_mass=1.,maximum_stratum_mass_error=error,
                        loss=value['loss'],clean_loss=value['clean_loss'],view_loss=value['view_loss'],
                        clean_numerator=weights['clean_numerator'],view_numerator=weights['view_numerator'],
                        total_numerator=weights['total_numerator'],denominator=weights['denominator'])
                    first = dict(**base,epoch=epoch,global_step=value['global_step'],batch=1,B=B,entries=2*B)
                    for task in range(2):
                        info = value['diagnostics'][f'task{task}']; assert not info['forced_zero']
                        assert info['nonzero_count']+info['zero_count'] == B
                        key = (base['dataset'],base['seed'],task)
                        target = task_sums.setdefault(key,dict(dataset=base['dataset'],arm='NDP-A',seed=base['seed'],
                            task=task,batches=0,entries=0,nonzero_count=0,zero_count=0,zero_direction_count=0,zero_rms_count=0))
                        target['batches'] += 1; target['entries'] += B
                        for field in ('nonzero_count','zero_count','zero_direction_count','zero_rms_count'):
                            target[field] += info[field]; first[f't{task}_{field}'] = info[field]
                            row[f't{task}_{field}'] = info[field]
                        for metric in ('delta_norm','target_logit_change','non_target_absolute_drift'):
                            stats = info[metric]; assert stats['count'] == B
                            target[metric+'_weighted_sum'] = target.get(metric+'_weighted_sum',0.)+stats['mean']*B
                            target[metric+'_maximum'] = max(target.get(metric+'_maximum',-float('inf')),stats['maximum'])
                            for quantile in ('median','p90','p99'):
                                name = metric+'_batch_'+quantile+'_sum'
                                target[name] = target.get(name,0.)+stats[quantile]
                            for field,v in stats.items():
                                first[f't{task}_{metric}_{field}'] = v
                        for cell,counts in info['cells'].items():
                            ckey = (base['dataset'],base['seed'],cell)
                            cell_target = cell_sums.setdefault(ckey,dict(dataset=base['dataset'],arm='NDP-A',seed=base['seed'],
                                cell=cell,count=0,zero_count=0))
                            cell_target['count'] += counts['count']; cell_target['zero_count'] += counts['zero_count']
                    for cell,fields in weights['cells'].items():
                        for name in ('count','clean_weight_mass','view_weight_mass','total_weight_mass'):
                            row[cell+'_'+name] = fields[name]
                    audit.append(row)
                    if value['batch'] == 1:
                        first_batches.append(first)
            assert batches == ledger['batches']
    training = pd.DataFrame(training); first_batches = pd.DataFrame(first_batches); audit = pd.DataFrame(audit)
    assert len(training) == len(first_batches) == 500 and len(audit) == 17250
    task_rows = []
    for row in task_sums.values():
        row['zero_fraction'] = row['zero_count']/row['entries']
        for metric in ('delta_norm','target_logit_change','non_target_absolute_drift'):
            row[metric+'_entry_weighted_mean'] = row.pop(metric+'_weighted_sum')/row['entries']
            for quantile in ('median','p90','p99'):
                row[metric+'_mean_batch_'+quantile] = row.pop(metric+'_batch_'+quantile+'_sum')/row['batches']
        task_rows.append(row)
    cells = pd.DataFrame(cell_sums.values()); cells['zero_fraction'] = cells.zero_count/cells['count']
    tasks = pd.DataFrame(task_rows)
    per_seed = training.groupby(['dataset','arm','seed'],as_index=False,sort=False).agg(
        epochs=('epoch','count'),batches=('batch_count','sum'),B=('B','sum'),entries=('entries','sum'),
        mean_loss=('loss','mean'),mean_clean_loss=('clean_loss','mean'),mean_view_loss=('view_loss','mean'),
        probe_seconds=('probe_seconds','sum'),construction_seconds=('construction_seconds','sum'),
        training_seconds=('training_seconds','sum'),validation_seconds=('validation_seconds','sum'),epoch_seconds=('seconds','sum'),
        maximum_ram_usage=('ram_usage','max'),mean_examples_per_second=('examples_per_second','mean'),
        total_weight_mass=('total_weight_mass','sum'))
    per_seed['mean_batch_total_weight_mass'] = per_seed.total_weight_mass/per_seed.batches
    assert (per_seed.entries == 2*per_seed.B).all() and (per_seed.mean_batch_total_weight_mass == 1).all()
    first_metrics = [k for k in first_batches if k not in ('dataset','arm','seed','key','epoch','global_step','batch')]
    first_means = first_batches.groupby(['dataset','arm','seed'],as_index=False,sort=False)[first_metrics].mean()
    output.mkdir(exist_ok=True)
    for name,frame in (('STAGE2_TRAINING_CURVES',training),('STAGE2_FIRST_BATCH_DIAGNOSTICS',first_batches),('STAGE2_BATCH_AUDIT',audit)):
        frame.to_csv(output/f'{name}.csv',index=False)
    return dict(training_seed_summary=records(per_seed),training_summary=group_descriptions(per_seed,['dataset','arm'],
        [k for k in per_seed if k not in ('dataset','arm','seed')]),
        task_diagnostic_seed_summary=records(tasks),task_diagnostic_summary=group_descriptions(tasks,['dataset','arm','task'],
            [k for k in tasks if k not in ('dataset','arm','seed','task')]),zero_cell_seed_summary=records(cells),
        first_batch_seed_means=records(first_means),first_batch_summary=group_descriptions(first_means,['dataset','arm'],first_metrics),
        diagnostic_scope=dict(all_batch_count=17250,new_epoch_count=500,
            task_means='Entry-weighted means over all new batches; maximum is maximum observed entry value',
            task_quantiles='Means of recorded batch medians/p90/p99, not pooled quantiles',
            first_batch='One first natural batch per epoch, 50 batches per seed; within-seed means weight recorded batches equally',
            parameter_gradient_norm_or_cosine='Not collected in Stage 2',natural_support='All original 2B row-task entries retained'))


def figures(curves,pairs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(10.5,4.1),sharex=True)
    for ax,dataset in zip(axes,DATASETS):
        frame = curves[(curves.dataset == dataset)&(curves.kind == 'fixed_ema')]
        for arm in ARMS:
            g = frame[frame.arm == arm].groupby('epoch').macro_auc.agg(['mean','std'])
            x = g.index.to_numpy(); mean = g['mean'].to_numpy(); sd = g['std'].to_numpy()
            ax.plot(x,mean,label=arm); ax.fill_between(x,mean-sd,mean+sd,alpha=.14)
        ax.set(title=dataset,xlabel='Epoch',ylabel='Validation Macro AUC'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output/'STAGE2_VALIDATION_CURVES.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(11,7))
    for column,dataset in enumerate(DATASETS):
        for row,kind in enumerate(('auc','logloss')):
            names = ['macro_'+kind]+(CELL_AUC if kind == 'auc' else CELL_LL)
            frame = pairs[(pairs.dataset == dataset)&(pairs.endpoint == 'selected')&
                (pairs.kind == 'fixed_ema')&(pairs.reference_kind == 'fixed_ema')].set_index('metric').loc[names]
            mean = frame['mean'].to_numpy(); error = np.vstack((mean-frame.ci95_low,frame.ci95_high-mean))
            ax = axes[row,column]; ax.errorbar(mean,np.arange(7),xerr=error,fmt='o',capsize=3)
            ax.axvline(0,color='grey',linestyle='--',linewidth=.8)
            ax.set_yticks(np.arange(7),['Macro']+[k.split('_')[0] for k in names[1:]]); ax.invert_yaxis()
            ax.set(title=f'{dataset}: {kind}',xlabel='NDP-A - CLEAN paired delta, 95% t interval'); ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(output/'STAGE2_PAIRED_CELLS.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,default=ROOT/'artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1')
    args = parser.parse_args(); attempt = args.attempt.resolve(); output = Path(__file__).resolve().parent
    runs = results(attempt); assert len(runs) == 20 and all(r['epochs'] == 50 and not r['test_accessed'] for r in runs)
    names = ['ENDPOINTS','SUMMARY','PAIRED_SUMMARY','PAIRED_SEEDS','VALIDATION_CURVES','WEIGHTS','COSTS']
    data = {name:pd.read_csv(attempt/'tables'/f'{name}.csv') for name in names}
    selected = data['ENDPOINTS'].loc[lambda f:f.endpoint == 'selected'].copy(); curves = data['VALIDATION_CURVES']
    assert len(selected) == 40 and len(curves) == 2000
    assert set(zip(selected.dataset,selected.arm,selected.seed,selected.kind)) == {
        (d,a,s,k) for d in DATASETS for a in ARMS for s in SEEDS for k in ('online','fixed_ema')}
    pairs = data['PAIRED_SUMMARY']; seed_pairs = data['PAIRED_SEEDS']; selected_pairs = pairs[pairs.endpoint == 'selected']
    common = pairs[(pairs.endpoint != 'selected')&pairs.metric.isin(MACRO)]
    phase_seeds,phase_pairs = phases(curves)
    weight_seeds,priors,weight_summary = natural_weights(data['WEIGHTS'])
    training = training_evidence(attempt,runs,output/'tables')
    identity_names = ['run_manifest.json','source_receipt.json','reused_clean_index.json','zero_counterpart_receipt.json',
        'CPU_CHECKS.json','SMOKE_REPORT.json','movielens_data_manifest.json','kuairand_data_manifest.json']
    identities = {name:dict(sha256=sha_file(attempt/name),content=json.loads((attempt/name).read_text())) for name in identity_names}
    cost_summary = group_descriptions(data['COSTS'],['dataset','arm'],['worker_seconds','peak_allocated_bytes','peak_reserved_bytes'])
    evidence = dict(stage='STAGE2',new_trajectories=10,reused_clean_trajectories=10,seeds=SEEDS,statistical_unit='training_seed',
        test_accessed=False,identities=identities,csv_sha256={name:sha_file(attempt/'tables'/f'{name}.csv') for name in names},
        selected_raw=records(selected),selected_summary=records(data['SUMMARY'].loc[lambda f:f.endpoint == 'selected']),
        selected_pairs=with_seed_deltas(selected_pairs,seed_pairs[seed_pairs.endpoint == 'selected']),
        common_epoch_macro_pairs=with_seed_deltas(common,seed_pairs[seed_pairs.endpoint != 'selected']),
        phase_seed_means=records(phase_seeds),phase_pairs=records(phase_pairs),
        selected_epoch_summary=group_descriptions(selected,['dataset','arm','kind'],['epoch']),
        weight_seed_sums=records(weight_seeds),weight_summary=weight_summary,natural_and_view_label_priors=records(priors),
        cost_seed=records(data['COSTS']),cost_summary=cost_summary,
        new_worker_hours=float(data['COSTS'].loc[data['COSTS'].arm == 'NDP-A','worker_seconds'].sum()/3600),
        report_csv_sha256={name:sha_file(output/'tables'/name) for name in
            ('STAGE2_TRAINING_CURVES.csv','STAGE2_FIRST_BATCH_DIAGNOSTICS.csv','STAGE2_BATCH_AUDIT.csv')},**training)
    (output/'STAGE2_EVIDENCE.json').write_text(json.dumps(native(evidence),indent=2,allow_nan=False)+'\n')
    text = ['# Stage 2 数值汇总','统计单位为seed；五seed为2022–2026。主配对为NDP-A−CLEAN，online/online、EMA/EMA和EMA/clean-online总收益分别列出。',
        '## Selected 绝对值',markdown_table(data['SUMMARY'].loc[lambda f:(f.endpoint == 'selected')&f.metric.isin(AGGREGATES)],
            ['dataset','arm','kind','metric','mean','sample_sd']),
        '## Selected 配对 Macro AUC / LogLoss',markdown_table(selected_pairs.loc[lambda f:f.metric.isin(MACRO)],
            ['dataset','kind','reference_kind','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## 训练阶段配对','每seed先对对应epoch求均值，再计算五seed配对差。',
        markdown_table(phase_pairs,['dataset','phase','kind','reference_kind','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## Common epoch',markdown_table(common,['dataset','endpoint','kind','reference_kind','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## 自然分层质量',markdown_table(pd.DataFrame(weight_summary).loc[lambda f:f.metric.str.endswith('weight_mass_per_batch')],
            ['dataset','arm','cell_label','metric','mean','sample_sd']),
        '## 自然与view标签先验',markdown_table(priors),
        '## 训练损失、质量和时间',markdown_table(pd.DataFrame(training['training_summary']).loc[lambda f:f.metric.isin([
            'mean_loss','mean_clean_loss','mean_view_loss','mean_batch_total_weight_mass','probe_seconds','construction_seconds','training_seconds','validation_seconds'])],
            ['dataset','arm','metric','mean','sample_sd']),
        '## 全量task诊断','mean按全部entry数加权；分位数字段为各batch已记录分位数的均值。',
        markdown_table(pd.DataFrame(training['task_diagnostic_summary']).loc[lambda f:f.metric.isin([
            'zero_fraction','delta_norm_entry_weighted_mean','delta_norm_maximum','target_logit_change_entry_weighted_mean',
            'non_target_absolute_drift_entry_weighted_mean','non_target_absolute_drift_maximum'])],
            ['dataset','arm','task','metric','mean','sample_sd']),
        '首batch数据固定为每epoch首个自然batch，每seed50条；该表保存task zero-direction、delta及drift摘要。Stage2未采集参数gradient norm/cos。',
        '## 成本',markdown_table(pd.DataFrame(cost_summary),['dataset','arm','metric','mean','sample_sd']),
        f"新增10条worker时间合计：{evidence['new_worker_hours']:.6f}小时。",
        '## 数值附件','[逐seed附录](STAGE2_NUMERICAL_APPENDIX.md)、[完整机器数值](STAGE2_EVIDENCE.json)、'
        '[epoch训练表](tables/STAGE2_TRAINING_CURVES.csv)、[首batch诊断](tables/STAGE2_FIRST_BATCH_DIAGNOSTICS.csv)、[全batch自然质量核对](tables/STAGE2_BATCH_AUDIT.csv)。',
        '![固定EMA验证曲线](figures/STAGE2_VALIDATION_CURVES.png)','![Selected EMA六格配对差](figures/STAGE2_PAIRED_CELLS.png)']
    (output/'STAGE2_GENERATED_TABLES.md').write_text('\n\n'.join(text)+'\n')
    appendix = ['# Stage 2 selected validation逐seed附录','Traffic为sample_weighted_cell；Pooled AUC为pooled_task_mean_auc。原始完整精度保留在ENDPOINTS.csv和STAGE2_EVIDENCE.json。']
    for dataset in DATASETS:
        for kind in ('online','fixed_ema'):
            frame = selected[(selected.dataset == dataset)&(selected.kind == kind)].sort_values(['arm','seed'])
            appendix += [f'## {dataset} / {kind}',markdown_table(frame,['arm','seed','epoch']+CELL_AUC+CELL_LL+AGGREGATES,digits=9),
                markdown_table(frame,['arm','seed','epoch','model_hash'])]
    (output/'STAGE2_NUMERICAL_APPENDIX.md').write_text('\n\n'.join(appendix)+'\n')
    figures(curves,pairs,output/'figures')
    print(json.dumps(dict(status='COMPLETE',trajectories=20,selected_readouts=40,test_accessed=False,
        outputs=['STAGE2_EVIDENCE.json','STAGE2_GENERATED_TABLES.md','STAGE2_NUMERICAL_APPENDIX.md',
            'tables/STAGE2_TRAINING_CURVES.csv','tables/STAGE2_FIRST_BATCH_DIAGNOSTICS.csv','tables/STAGE2_BATCH_AUDIT.csv',
            'figures/STAGE2_VALIDATION_CURVES.png','figures/STAGE2_PAIRED_CELLS.png']),indent=2))


if __name__ == '__main__':
    main()
