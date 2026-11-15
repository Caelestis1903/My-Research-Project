"""Build Stage 1 numerical attachments from complete, frozen CSV exports."""
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
from experiments.auxiliary_objective_attribution_20261115.protocol import results, epoch_files
from experiments.auxiliary_objective_attribution_20261115.tables import paired

SEEDS = [2022,2023,2024,2025,2026]
DATASETS = ['movielens','kuairand']
ARMS = ['CLEAN','B','C','Z-PAIR','Z-EXAMPLE']
CONTRASTS = [('Z-PAIR','CLEAN'),('Z-EXAMPLE','CLEAN'),('B','Z-PAIR'),('C','Z-EXAMPLE'),('Z-PAIR','Z-EXAMPLE')]
ALL_CONTRASTS = CONTRASTS+[('B','CLEAN'),('C','CLEAN')]
MACRO = ['macro_auc','macro_logloss']
CELL_AUC = [f'd{d}t{t}_auc' for d in range(3) for t in range(2)]
CELL_LL = [f'd{d}t{t}_logloss' for d in range(3) for t in range(2)]
AGGREGATES = ['macro_auc','macro_logloss','sample_weighted_cell_auc','sample_weighted_cell_logloss',
              'pooled_task_mean_auc','pooled_logloss','worst_cell_auc']


def records(frame):
    return frame.astype(object).where(pd.notna(frame),None).to_dict('records')


def native(value):
    if isinstance(value,dict):
        return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [native(v) for v in value]
    if isinstance(value,np.generic):
        return native(value.item())
    if isinstance(value,float):
        assert np.isfinite(value), 'Nonfinite report number'
    return value


def describe(values):
    v = np.asarray(values,dtype=float)
    assert v.shape == (5,) and np.isfinite(v).all()
    return dict(seeds=SEEDS,values=v.tolist(),mean=float(v.mean()),sample_sd=float(v.std(ddof=1)))


def markdown_table(frame,columns=None,digits=7):
    frame = frame if columns is None else frame[columns]
    def fmt(value):
        if value is None or pd.isna(value):
            return ''
        if isinstance(value,(float,np.floating)):
            return f'{value:.{digits}f}'
        return str(value).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def with_seed_deltas(summary,seeds):
    columns = ['dataset','arm','control','endpoint','kind','reference_kind','metric']
    grouped = {k:g.sort_values('seed') for k,g in seeds.groupby(columns,sort=False)}
    out = []
    for row in records(summary):
        group = grouped[tuple(row[k] for k in columns)]
        assert group.seed.tolist() == SEEDS
        row['seeds'] = SEEDS; row['deltas'] = group.delta.tolist()
        out.append(row)
    return out


def phase_statistics(curves):
    curves = curves.copy()
    curves['phase'] = np.select([curves.epoch <= 10,curves.epoch <= 30],['E1-10','E11-30'],default='E31-50')
    seed_means = curves.groupby(['dataset','arm','kind','seed','phase'],as_index=False,sort=False)[MACRO].mean()
    lookup = {(d,a,k,p):g.sort_values('seed') for (d,a,k,p),g in seed_means.groupby(['dataset','arm','kind','phase'])}
    out = []
    for dataset in DATASETS:
        for phase in ('E1-10','E11-30','E31-50'):
            for arm,control in ALL_CONTRASTS:
                kinds = [('online','online'),('fixed_ema','fixed_ema')]
                if control == 'CLEAN':
                    kinds.append(('fixed_ema','online'))
                for kind,reference_kind in kinds:
                    a = lookup[dataset,arm,kind,phase]; b = lookup[dataset,control,reference_kind,phase]
                    assert a.seed.tolist() == b.seed.tolist() == SEEDS
                    for metric in MACRO:
                        delta = (a[metric].to_numpy()-b[metric].to_numpy()).tolist()
                        out.append(dict(dataset=dataset,phase=phase,arm=arm,control=control,kind=kind,
                            reference_kind=reference_kind,metric=metric,seeds=SEEDS,deltas=delta,**paired(delta)))
    return seed_means,pd.DataFrame(out)


def weight_statistics(weights):
    keys = ['dataset','arm','seed','cell_label']
    sums = ['batch_count','clean_count','aux_count','clean_numerator','aux_numerator',
            'clean_weight_mass','aux_weight_mass','clean_contribution','aux_contribution']
    per_seed = weights.groupby(keys,as_index=False,sort=False)[sums].sum()
    for name in ('clean_weight_mass','aux_weight_mass','clean_contribution','aux_contribution'):
        per_seed[name+'_per_batch'] = per_seed[name]/per_seed.batch_count
    per_seed['mean_clean_entry_coefficient'] = per_seed.clean_weight_mass/per_seed.clean_count
    per_seed['mean_aux_entry_coefficient'] = np.divide(per_seed.aux_weight_mass,per_seed.aux_count,
        out=np.zeros(len(per_seed),dtype=float),where=per_seed.aux_count.to_numpy() != 0)
    per_seed['aux_to_clean_mass_ratio'] = per_seed.aux_weight_mass/per_seed.clean_weight_mass
    per_seed['cell'] = per_seed.cell_label.str[:4]
    per_seed['label'] = per_seed.cell_label.str[-1].astype(int)
    priors = []
    for key,g in per_seed.groupby(['dataset','arm','seed','cell'],sort=False):
        total_clean = int(g.clean_count.sum()); total_aux = int(g.aux_count.sum())
        priors.append(dict(zip(['dataset','arm','seed','cell'],key),
            clean_count=total_clean,aux_count=total_aux,
            natural_positive_rate=float(g.loc[g.label == 1,'clean_count'].sum()/total_clean),
            auxiliary_positive_rate=float(g.loc[g.label == 1,'aux_count'].sum()/total_aux) if total_aux else None))
    summaries = []
    metrics = ['clean_count','aux_count','clean_weight_mass_per_batch','aux_weight_mass_per_batch',
               'clean_contribution_per_batch','aux_contribution_per_batch','mean_clean_entry_coefficient',
               'mean_aux_entry_coefficient','aux_to_clean_mass_ratio']
    for (dataset,arm,cell),g in per_seed.groupby(['dataset','arm','cell_label'],sort=False):
        g = g.sort_values('seed'); assert g.seed.tolist() == SEEDS
        for metric in metrics:
            summaries.append(dict(dataset=dataset,arm=arm,cell_label=cell,metric=metric,**describe(g[metric])))
    return per_seed,pd.DataFrame(priors),summaries


def group_descriptions(frame,keys,metrics):
    out = []
    for key,g in frame.groupby(keys,sort=False):
        key = key if isinstance(key,tuple) else (key,)
        g = g.sort_values('seed'); assert g.seed.tolist() == SEEDS
        for metric in metrics:
            out.append(dict(zip(keys,key),metric=metric,**describe(g[metric])))
    return out


def training_diagnostics(attempt,runs,support_overlap,output):
    training = []; first_batches = []
    for result in runs:
        job = result['job']; arm = job['config']['arm']
        if arm == 'CLEAN':
            continue
        base = dict(dataset=job['dataset'],arm=arm,seed=job['seed'],key=job['key'])
        for epoch,(_,data) in sorted(epoch_files(attempt,result).items()):
            ledger = data['objective_ledger']
            fields = ['loss','clean_loss','auxiliary_loss','probe_seconds','training_seconds','seconds',
                'ram_usage','examples_per_second','model_hash','ema_hash','optimizer_hash',
                'epoch_order_hash','support_log','support_log_sha256']
            training.append(dict(**base,epoch=epoch,global_step=data['global_step'],
                **{k:data[k] for k in fields},batch_count=ledger['batches'],
                B=ledger['B'],M=ledger['M'],total_weight_mass=ledger['total_weight_mass']))
            with gzip.open(attempt/data['support_log'],'rt') as f:
                first = json.loads(next(f))
            assert first['epoch'] == epoch and first['batch'] == 1
            first_batches.append(dict(**base,epoch=epoch,global_step=first['global_step'],batch=1,
                rows=first['rows'],**first['diagnostics'],probe_seconds=first['probe_seconds'],
                training_seconds=first['training_seconds'],B=first['ledger']['B'],M=first['ledger']['M'],
                total_weight_mass=first['ledger']['total_weight_mass']))
    training = pd.DataFrame(training); first_batches = pd.DataFrame(first_batches)
    assert len(training) == len(first_batches) == 2000
    output.mkdir(exist_ok=True)
    training.to_csv(output/'TRAINING_CURVES.csv',index=False)
    first_batches.to_csv(output/'FIRST_BATCH_DIAGNOSTICS.csv',index=False)
    seed_summaries = []
    for (dataset,arm,seed),group in training.groupby(['dataset','arm','seed'],sort=False):
        assert group.epoch.tolist() == list(range(1,51))
        B = int(group.B.sum()); M = int(group.M.sum()); batches = int(group.batch_count.sum())
        row = dict(dataset=dataset,arm=arm,seed=int(seed),epochs=50,batches=batches,B=B,M=M,
            ratio_of_totals_B_over_M=B/M if M else None,
            total_weight_mass=float(group.total_weight_mass.sum()),
            mean_batch_total_weight_mass=float(group.total_weight_mass.sum()/batches),
            probe_seconds=float(group.probe_seconds.sum()),training_seconds=float(group.training_seconds.sum()),
            epoch_seconds=float(group.seconds.sum()),max_ram_usage=float(group.ram_usage.max()),
            mean_examples_per_second=float(group.examples_per_second.mean()))
        for metric in ('loss','clean_loss','auxiliary_loss'):
            row['mean_'+metric] = float(group[metric].mean())
        treated = arm in ('B','C')
        matched = support_overlap[(support_overlap.dataset == dataset)&(support_overlap.seed == seed)&
            (support_overlap.treatment == arm if treated else support_overlap.zero == arm)]
        assert len(matched) == batches
        proposed = int(matched.proposed.sum())
        accepted = int(matched['treatment_M' if treated else 'zero_M'].sum())
        assert accepted == M
        row.update(proposed_row_task_entries=proposed,accepted_row_task_entries=accepted,
            skipped_row_task_entries=proposed-accepted,skip_fraction=(proposed-accepted)/proposed if proposed else 0.)
        seed_summaries.append(row)
    seed_summaries = pd.DataFrame(seed_summaries)
    metadata = ['dataset','arm','seed','key','epoch','global_step','batch']
    metrics = [k for k in first_batches if k not in metadata]
    first_seed_means = first_batches.groupby(['dataset','arm','seed'],as_index=False,sort=False)[metrics].mean()
    first_seed_means['recorded_first_batches'] = 50
    first_summary = []
    for (dataset,arm),group in first_seed_means.groupby(['dataset','arm'],sort=False):
        group = group.sort_values('seed')
        for metric in metrics:
            if group[metric].notna().all():
                first_summary.append(dict(dataset=dataset,arm=arm,metric=metric,**describe(group[metric])))
    numeric = [k for k in seed_summaries if k not in ('dataset','arm','seed')]
    return dict(training_seed_summary=records(seed_summaries),
        training_summary=group_descriptions(seed_summaries,['dataset','arm'],numeric),
        first_batch_diagnostic_seed_means=records(first_seed_means),first_batch_diagnostic_summary=first_summary,
        diagnostic_scope=dict(training_curves='All 50 epochs of every new trajectory; B/M/weight mass are epoch sums',
            accepted_proposed_skip='All batches from SUPPORT_OVERLAP.csv; accepted counts cross-checked against epoch M',
            first_batch='One recorded natural first batch per epoch, 50 points per seed; means weight these batches equally',
            ratio_of_totals_B_over_M='sum of natural batch row counts / sum of accepted auxiliary row-task counts; not mean of per-batch B/M'))


def figures(curves,paired_summary,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(11,4.1),sharex=True)
    for ax,dataset in zip(axes,DATASETS):
        data = curves[(curves.dataset == dataset)&(curves.kind == 'fixed_ema')]
        for arm in ARMS:
            group = data[data.arm == arm].groupby('epoch').macro_auc.agg(['mean','std'])
            x = group.index.to_numpy(); mean = group['mean'].to_numpy(); sd = group['std'].to_numpy()
            ax.plot(x,mean,label=arm,linewidth=1.3); ax.fill_between(x,mean-sd,mean+sd,alpha=.12)
        ax.set(title=dataset,xlabel='Epoch',ylabel='Validation Macro AUC'); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8); fig.tight_layout(); fig.savefig(directory/'VALIDATION_CURVES.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(12,6.8))
    labels = [f'{a} - {b}' for a,b in CONTRASTS]
    for column,dataset in enumerate(DATASETS):
        for row,metric in enumerate(MACRO):
            ax = axes[row,column]; means = []; low = []; high = []
            for arm,control in CONTRASTS:
                selection = paired_summary[(paired_summary.dataset == dataset)&(paired_summary.arm == arm)&
                    (paired_summary.control == control)&(paired_summary.endpoint == 'selected')&
                    (paired_summary.kind == 'fixed_ema')&(paired_summary.reference_kind == 'fixed_ema')&
                    (paired_summary.metric == metric)]
                assert len(selection) == 1
                point = selection.iloc[0]; means.append(point['mean'])
                low.append(point['mean']-point.ci95_low); high.append(point.ci95_high-point['mean'])
            ax.errorbar(means,np.arange(5),xerr=np.asarray([low,high]),fmt='o',capsize=3)
            ax.axvline(0,color='grey',linestyle='--',linewidth=.8); ax.set_yticks(np.arange(5),labels)
            ax.invert_yaxis(); ax.set(title=f'{dataset}: {metric}',xlabel='Paired mean delta with 95% t interval')
            ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(directory/'PAIRED_CONTRASTS.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,default=ROOT/'artifacts/auxiliary_objective_attribution_20261115/attempt1')
    args = parser.parse_args(); attempt = args.attempt.resolve(); output = Path(__file__).resolve().parent
    runs = results(attempt)
    assert len(runs) == 50 and all(r['epochs'] == 50 and not r['test_accessed'] for r in runs), 'Full Stage 1 matrix required'
    tables = attempt/'tables'
    csv_names = ['ENDPOINTS','SUMMARY','PAIRED_SUMMARY','PAIRED_SEEDS','VALIDATION_CURVES','WEIGHTS','COSTS','SUPPORT_SUMMARY']
    data = {name:pd.read_csv(tables/f'{name}.csv') for name in csv_names}
    endpoints = data['ENDPOINTS']; selected = endpoints[endpoints.endpoint == 'selected'].copy()
    assert len(selected) == 100
    assert set(zip(selected.dataset,selected.arm,selected.seed,selected.kind)) == {
        (d,a,s,k) for d in DATASETS for a in ARMS for s in SEEDS for k in ('online','fixed_ema')}
    curves = data['VALIDATION_CURVES']; assert len(curves) == 5000
    pair_summary = data['PAIRED_SUMMARY']; pair_seeds = data['PAIRED_SEEDS']
    selected_pairs = pair_summary[pair_summary.endpoint == 'selected']
    common_macro = pair_summary[(pair_summary.endpoint != 'selected')&pair_summary.metric.isin(MACRO)]
    phase_seeds,phase_pairs = phase_statistics(curves)
    weight_seeds,priors,weight_summary = weight_statistics(data['WEIGHTS'])
    epoch_summary = group_descriptions(selected,['dataset','arm','kind'],['epoch'])
    cost_summary = group_descriptions(data['COSTS'],['dataset','arm'],['worker_seconds','peak_allocated_bytes','peak_reserved_bytes'])
    support = data['SUPPORT_SUMMARY'].copy()
    support['different_support_batch_fraction'] = support.different_support_batches/support.batches
    support_summary = group_descriptions(support,['dataset','treatment','zero'],
        ['mean_jaccard','min_jaccard','different_support_batch_fraction','treatment_entries','zero_entries','symmetric_difference_entries'])
    support_overlap = pd.read_csv(tables/'SUPPORT_OVERLAP.csv',
        usecols=['dataset','treatment','zero','seed','proposed','treatment_M','zero_M'])
    diagnostics = training_diagnostics(attempt,runs,support_overlap,output/'tables')
    identity_files = ['run_manifest.json','source_receipt.json','reuse_receipt.json','movielens_data_manifest.json','kuairand_data_manifest.json']
    identities = {name:dict(sha256=hashlib.sha256((attempt/name).read_bytes()).hexdigest(),
        content=json.loads((attempt/name).read_text())) for name in identity_files}
    csv_hashes = {name:hashlib.sha256((tables/f'{name}.csv').read_bytes()).hexdigest() for name in csv_names}
    evidence = dict(stage='STAGE1',seeds=SEEDS,statistical_unit='training_seed',test_accessed=False,
        new_trajectories=40,reused_clean_trajectories=10,identities=identities,csv_sha256=csv_hashes,
        selected_raw=records(selected),selected_summary=records(data['SUMMARY'][data['SUMMARY'].endpoint == 'selected']),
        selected_pairs=with_seed_deltas(selected_pairs,pair_seeds[pair_seeds.endpoint == 'selected']),
        common_epoch_macro_pairs=with_seed_deltas(common_macro,pair_seeds[pair_seeds.endpoint != 'selected']),
        phase_seed_means=records(phase_seeds),phase_pairs=records(phase_pairs),selected_epoch_summary=epoch_summary,
        weight_seed_sums=records(weight_seeds),weight_summary=weight_summary,label_priors=records(priors),
        support_seed=records(support),support_summary=support_summary,cost_seed=records(data['COSTS']),cost_summary=cost_summary,
        new_worker_hours=float(data['COSTS'].loc[data['COSTS'].arm != 'CLEAN','worker_seconds'].sum()/3600),
        max_new_peak_reserved_bytes=int(data['COSTS'].loc[data['COSTS'].arm != 'CLEAN','peak_reserved_bytes'].max()),
        report_csv_sha256={name:hashlib.sha256((output/'tables'/name).read_bytes()).hexdigest()
            for name in ('TRAINING_CURVES.csv','FIRST_BATCH_DIAGNOSTICS.csv')},**diagnostics)
    (output/'EVIDENCE.json').write_text(json.dumps(native(evidence),indent=2,allow_nan=False)+'\n')
    text = ['# Stage 1 数值汇总','统计单位为 seed；五 seed 为2022–2026。AUC 与 LogLoss分别列出。',
        '完整绝对值见 [NUMERICAL_APPENDIX.md](NUMERICAL_APPENDIX.md)；机器可读原值和汇总见 [EVIDENCE.json](EVIDENCE.json)。',
        '## Selected 五 seed 绝对值均值与 sample SD']
    absolute = data['SUMMARY'][(data['SUMMARY'].endpoint == 'selected')&data['SUMMARY'].metric.isin(AGGREGATES)]
    text += [markdown_table(absolute,['dataset','arm','kind','metric','mean','sample_sd']),
        '## Selected 配对 Macro AUC / LogLoss','差值为arm−control；confidence interval为五 seed配对95% t区间。']
    text += [markdown_table(selected_pairs[selected_pairs.metric.isin(MACRO)],
        ['dataset','arm','control','kind','reference_kind','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## 训练阶段配对','先在每个seed内计算epoch均值，再对五seed配对。',
        markdown_table(phase_pairs[(phase_pairs.kind == 'fixed_ema')&(phase_pairs.reference_kind == 'fixed_ema')],
            ['dataset','phase','arm','control','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## Common epoch 配对 Macro AUC / LogLoss',
        markdown_table(common_macro[(common_macro.kind == 'fixed_ema')&(common_macro.reference_kind == 'fixed_ema')],
            ['dataset','endpoint','arm','control','metric','mean','sample_sd','ci95_low','ci95_high','positive_seeds']),
        '## 选点epoch',markdown_table(pd.DataFrame(epoch_summary),['dataset','arm','kind','mean','sample_sd']),
        '## Support overlap',markdown_table(support,['dataset','treatment','zero','seed','batches','mean_jaccard','min_jaccard',
            'different_support_batches','different_support_batch_fraction','symmetric_difference_entries']),
        '## 每domain/task的自然与辅助正例率','由全部50epochs的计数分别聚合，保留seed。',
        markdown_table(priors,['dataset','arm','seed','cell','clean_count','aux_count','natural_positive_rate','auxiliary_positive_rate']),
        '## 目标系数质量','每batch质量先在各seed内由50epoch账本聚合，表中为五seed均值与sample SD。',
        markdown_table(pd.DataFrame(weight_summary).loc[lambda f:f.metric.isin(['clean_weight_mass_per_batch','aux_weight_mass_per_batch','aux_to_clean_mass_ratio'])],
            ['dataset','arm','cell_label','metric','mean','sample_sd']),
        '## 成本',markdown_table(pd.DataFrame(cost_summary),['dataset','arm','metric','mean','sample_sd']),
        f"新增40条worker时间合计：{evidence['new_worker_hours']:.6f}小时；峰值reserved：{evidence['max_new_peak_reserved_bytes']} bytes。",
        '## 完整训练损失与权重诊断',
        '原始2000个epoch的loss、B/M、目标质量、计时、RAM、吞吐和状态/hash见 [TRAINING_CURVES.csv](tables/TRAINING_CURVES.csv)。'
        'ratio_of_totals_B_over_M为每seed全部batch的B总计除以M总计；不计算逐batch比值的均值。',
        markdown_table(pd.DataFrame(diagnostics['training_summary']).loc[lambda f:f.metric.isin([
            'mean_loss','mean_clean_loss','mean_auxiliary_loss','ratio_of_totals_B_over_M',
            'mean_batch_total_weight_mass','skip_fraction','probe_seconds','training_seconds'])],
            ['dataset','arm','metric','mean','sample_sd']),
        '## 每epoch首batch梯度与扰动诊断',
        '每seed固定50个首batch；先在seed内对这些batch等权求均值，再报告五seed描述统计。'
        '逐条数据见 [FIRST_BATCH_DIAGNOSTICS.csv](tables/FIRST_BATCH_DIAGNOSTICS.csv)。',
        markdown_table(pd.DataFrame(diagnostics['first_batch_diagnostic_summary']).loc[lambda f:f.metric.isin([
            'clean_gradient_norm','auxiliary_gradient_norm','clean_auxiliary_gradient_cosine',
            'delta_norm_mean','delta_norm_max','non_target_drift_mean','non_target_drift_max',
            'capped_pairs','skipped_rows'])],['dataset','arm','metric','mean','sample_sd']),
        '## 图表','![固定EMA验证曲线](figures/VALIDATION_CURVES.png)',
        '![Selected EMA配对差](figures/PAIRED_CONTRASTS.png)']
    (output/'GENERATED_TABLES.md').write_text('\n\n'.join(text)+'\n')
    appendix = ['# Selected validation 逐 seed 数值附录',
        '每行对应一条trajectory的一个readout；六格顺序为d0t0,d0t1,d1t0,d1t1,d2t0,d2t1。所有选点依据为验证六格Macro AUC。',
        'Traffic为sample_weighted_cell；Pooled AUC为pooled_task_mean_auc。原始完整精度保留在ENDPOINTS.csv和EVIDENCE.json。']
    for dataset in DATASETS:
        for kind in ('online','fixed_ema'):
            group = selected[(selected.dataset == dataset)&(selected.kind == kind)].sort_values(['arm','seed'])
            appendix += [f'## {dataset} / {kind}',markdown_table(group,['arm','seed','epoch']+CELL_AUC+CELL_LL+AGGREGATES,digits=9),
                markdown_table(group,['arm','seed','epoch','model_hash'])]
    (output/'NUMERICAL_APPENDIX.md').write_text('\n\n'.join(appendix)+'\n')
    figures(curves,pair_summary,output/'figures')
    print(json.dumps(dict(status='COMPLETE',selected_endpoints=100,trajectories=50,test_accessed=False,
        outputs=['EVIDENCE.json','GENERATED_TABLES.md','NUMERICAL_APPENDIX.md','tables/TRAINING_CURVES.csv',
            'tables/FIRST_BATCH_DIAGNOSTICS.csv','figures/VALIDATION_CURVES.png','figures/PAIRED_CONTRASTS.png']),indent=2))


if __name__ == '__main__':
    main()
