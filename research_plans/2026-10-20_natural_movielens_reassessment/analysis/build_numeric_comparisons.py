"""Assemble completed, primary-analyzed stage comparisons; never infer run completion."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
ANA=ROOT/'analysis'
OUT=ANA/'numeric_comparisons'

def build(final=False):
    if final:
        snapshot=json.loads((ROOT/'EXPERIMENT_EXECUTION_SNAPSHOT.json').read_text());assert snapshot['logical_completed']==1060
        assert json.loads((ANA/'core_mechanism/analysis_status.json').read_text())['complete']
        assert json.loads((ANA/'n17/analysis_receipt.json').read_text())['formal_runs']==72
        assert json.loads((ANA/'ao04/analysis_status.json').read_text())['complete']
        transfer_status=json.loads((ANA/'transfer/collection_status.json').read_text())
        assert transfer_status['new_run_count']==824 and not transfer_status['missing']
    OUT.mkdir(exist_ok=True);rows=[];sources=[]
    def collect(path,transform):
        if path.is_file():
            f=pd.read_csv(path);rows.extend(transform(f));sources.append(str(path.relative_to(ROOT)))
    def v2(f):
        for r in f[f.stage.isin(['selected','final'])].itertuples():
            if r.metric not in ('macro_auc','macro_logloss'):continue
            yield dict(family='V2 fixed '+r.optimizer,population=r.population,budget='historical epoch recipe' if 'historical' in r.population else '50epoch/1000main',
                readout=r.stage,contrast='SB-SN',metric=r.metric,reference_mean=r.sn_mean,candidate_mean=r.sb_mean,
                n=r.n,delta=r.delta_mean,ci95_low=r.descriptive_t95_low,ci95_high=r.descriptive_t95_high,
                positive=r.positive,negative=r.negative,zero=r.zero)
    collect(ANA/'v2/paired_summary.csv',v2)
    def b1(f):
        for r in f[(f.split=='test')&(f.quantity=='macro')].itertuples():
            yield dict(family='B1 reconstruction Sample',population='kuairand_natural_reconstruction_historical',
                budget='50epoch/native recipe; not historical exact initialization',readout=r.rule+' selected',contrast='DH-F',
                metric='macro_'+r.metric,reference_mean=r.F_mean,candidate_mean=r.DH_mean,n=r.n,delta=r.mean,
                ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
    collect(ANA/'b1_history/selected_rule_seed_summary.csv',b1)

    def core(f):
        for r in f[f.checkpoint.isin(['selected','final'])].itertuples():
            if r.metric not in ('macro_auc','macro_logloss'):continue
            yield dict(family=r.stage,population=r.population,budget=r.budget,readout=r.checkpoint,contrast=r.contrast,
                metric=r.metric,reference_mean=r.reference_mean,candidate_mean=r.candidate_mean,n=r.n,delta=r.mean,
                ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
    for stage in ('AC-2D','AC-V9','AC-S5','AC-S6'):collect(ANA/'core'/stage/'paired_summary.csv',core)
    def old_controls(f):
        yield from core(f[(f.task==-1)&f.population.str.contains('historical')])
    collect(ANA/'controlled_history/paired_summary.csv',old_controls)
    def v9(f):
        for r in f[f.readout.isin(['AULC_1_10','AULC_1_16','AULC_1_20','AULC_1_50'])].itertuples():
            yield dict(family='AC-V9 full trajectory',population=r.population,
                budget='50epoch/250main' if 'historical' in r.population else '50epoch/1000main',
                readout=r.readout,contrast='SB-SN',metric='macro_'+r.metric,
                reference_mean=r.reference_mean,candidate_mean=r.candidate_mean,n=r.n,delta=r.mean,
                ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
    collect(ANA/'v9_history/paired_summary.csv',v9)
    if (ANA/'n17/analysis_receipt.json').exists():
        receipt=json.loads((ANA/'n17/analysis_receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['formal_runs']==72
        pairs=pd.read_csv(ANA/'n17/paired_seed_deltas.csv')
        read=pd.read_csv(ANA/'n17/seed_readouts.csv').drop_duplicates('run_dir').set_index('run_dir')
        def n17(f):
            for r in f[(f.task==-1)&f.metric.isin(['auc','logloss'])&f.readout.isin(['selected','final','AULC1_20','AULC1_100','AULC1_200'])].itertuples():
                pair=pairs[pairs.comparison==r.comparison].iloc[0]
                a=read.loc[pair.candidate_run_dir];b=read.loc[pair.reference_run_dir]
                yield dict(family='N17',population=a.population,budget=f'{a.group}/{b.group}; LR{a.lr:g}/{b.lr:g}',
                    readout=r.readout,contrast=r.comparison,metric='macro_'+r.metric,
                    reference_mean=r.reference_mean,candidate_mean=r.candidate_mean,n=r.n,delta=r.mean,
                    ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
        collect(ANA/'n17/paired_summary.csv',n17)
        sources.extend(['analysis/n17/paired_seed_deltas.csv','analysis/n17/seed_readouts.csv'])
    if (ANA/'ao04/analysis_status.json').exists():
        assert json.loads((ANA/'ao04/analysis_status.json').read_text())['complete']
        def ao04(f):
            for r in f[(f.task==-1)&f.metric.isin(['macro_auc','macro_logloss'])&f.checkpoint.isin(['selected','final'])].itertuples():
                yield dict(family='AO04',population='movielens_natural',budget='50epoch/1000main/50arch',readout=r.checkpoint,
                    contrast=r.contrast,metric=r.metric,reference_mean=r.reference_mean,candidate_mean=r.candidate_mean,n=r.n,
                    delta=r.mean,ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
        collect(ANA/'ao04/paired_summary.csv',ao04)
    if (ANA/'transfer/collection_status.json').exists() and json.loads((ANA/'transfer/collection_status.json').read_text()).get('new_run_count',0)>0:
        configs=pd.read_csv(ANA/'transfer/run_config_index.csv').set_index('run_id')
        definitions={d['comparison']:d for d in json.loads((ANA/'transfer/comparison_definitions.json').read_text())}
        def transfer(f):
            for r in f[(f.task==-1)&f.metric.isin(['macro_auc','macro_logloss'])].itertuples():
                d=definitions[r.comparison];c=configs.loc[d['candidate_run_ids'][0]];b=configs.loc[d['reference_run_ids'][0]]
                metric='auc' if r.metric=='macro_auc' else 'logloss'
                yield dict(family=c.study,population=c.population,budget=f'{c.scope}/{b.scope}; {c.budget}/{b.budget}',
                    readout=r.checkpoint,contrast=r.comparison,metric=r.metric,
                    reference_mean=getattr(r,'reference_'+metric),candidate_mean=getattr(r,'candidate_'+metric),n=r.n,
                    delta=r.mean,ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=int(r.n-r.positive-r.negative))
        collect(ANA/'transfer/paired_summary.csv',transfer)
        sources.extend(['analysis/transfer/run_config_index.csv','analysis/transfer/comparison_definitions.json'])
    table=pd.DataFrame(rows);assert table.n.eq(4).all()
    table.to_csv(OUT/'completed_four_seed_macro_comparisons.csv',index=False)
    lines=['# 历史与自然人口数值比较（阶段性版本）','',
        '**整体实验仍在进行。** 本表只合并已完成主代理分析、具备完整四种子比较的阶段；待完成训练和未解锁test均不填零。当前源文件清单见[JSON](analysis/numeric_comparisons/source_files.json)。', '',
        '差值均为候选−参照；AUC越大越好，LogLoss越小越好。宏值为六个domain×task cell的等权平均；不是pooled AUC。区间为四个优化seed的点态描述性t95，不是新总体或多重比较校正后的确认。旧V2/V9优化seed同时改变人工子集，新人口固定。', '',
        '## 1. 自然完整人口的同配置比较', '',
        '|协议|预算|阶段|候选−参照|参照AUC|候选AUC|ΔAUC（正/负seed）|ΔLL（变差/改善seed）|',
        '|---|---|---|---|---:|---:|---:|---:|']
    fmt=lambda v:str(v).replace('|',chr(92)+'|')
    n=table[(table.population=='movielens_natural')&table.metric.eq('macro_auc')]
    for r in n.itertuples():
        l=table[(table.family==r.family)&(table.population==r.population)&(table.budget==r.budget)&
                (table.readout==r.readout)&(table.contrast==r.contrast)&table.metric.eq('macro_logloss')]
        assert len(l)==1;l=l.iloc[0]
        lines.append(f'|{fmt(r.family)}|{fmt(r.budget)}|{fmt(r.readout)}|{fmt(r.contrast)}|{r.reference_mean:.6f}|{r.candidate_mean:.6f}|{r.delta:+.6f}（{r.positive}/{r.negative}）|{l.delta:+.6f}（{l.positive}/{l.negative}）|')
    lines+=['', '## 2. 人口、预算及历史对照', '',
        '人工MovieLens、自然KuaiRand、新自然MovieLens、匹配总N与匹配曝光条件逐行保存在[完整宏比较CSV](analysis/numeric_comparisons/completed_four_seed_macro_comparisons.csv)。不同family有不同融合、优化器、选模和主/arch时钟，不能把它们当单因素消融合并排名。', '',
        '- [V2阶段报告](V2_PHASE_ANALYSIS.md)：完整固定模型Adam/SGD及旧两环境，selected/final、逐域任务、预测复算。',
        '- [AC-2D性能报告](AC2D_PERFORMANCE_ANALYSIS.md)：四臂×四人口/预算条件，含匹配总N与同250步全支持；固定点机制诊断尚待完成。',
        '- [S5完整阶段报告](ACS5_PHASE_ANALYSIS.md)：32配置、两预算、历史/当前梯度、完整逐cell与预算交互，selected/末轮反转及域间取舍。',
        '- [S6阶段报告](ACS6_PHASE_ANALYSIS.md)：Sample F/DH，两预算全部16配置已分析，保留预算×阶段反转。',
        '- [V9阶段报告](ACV9_PERFORMANCE_ANALYSIS.md)：SGD .3八训练及八只读任务闭合，400份全轨迹预测2400cell复算，保留四个预设窗口。',
        '- [N18报告](N18_PHASE_ANALYSIS.md)：自然/人工人口的同信息LP、审核SE/RMSE，非训练AUC比较。',
        '- [历史B1、N17、N19/CM19及AO04重建](HISTORICAL_NUMERIC_RECONSTRUCTION.md)：1184个旧run、747个完整四seed配方比较；AO04另列旧ML两seed、KR一seed，不混入四seed表。新自然N17的72正式配置test已完成，见[N17完整报告](N17_PHASE_ANALYSIS.md)；新自然transfer与AO04测试尚未闭合。',
        '- [特征支持审计](FEATURE_SUPPORT_AUDIT.md)：匹配总N仍未控制ID支持，属于已量化的竞争解释，不是已识别的AUC中介效应。', '',
        '## 3. 逐cell、任务和阶段的读取规则', '',
        '对应阶段的`paired_cell_deltas.csv`、`paired_domain_deltas.csv`、`paired_task_deltas.csv`与原基线绝对值共同用于科学判断。最大域不自动等于最容易，最小域不自动等于最困难。一个宏均值不能替代两任务/三域是否同时受益，也不能替代selected与final的风险变化。', '',
        '当前尚未完成阶段必须继续完整预算、机制诊断、冻结评价和阶段分析。N17新自然数值现已加入；最终版将补齐AO04、N19/CM19的新自然数值，并明确重新评定各历史主张；本表目前不宣称研究闭环。']
    text='\n'.join(lines)+'\n'
    if final:
        text=text.replace('历史与自然人口数值比较（阶段性版本）','历史与自然人口数值比较（完成版）')
        text=text.replace('**整体实验仍在进行。** 本表只合并已完成主代理分析、具备完整四种子比较的阶段；待完成训练和未解锁test均不填零。','**规定训练、诊断和冻结测试均已完成。** 本表合并完整四优化种子比较，名义配置数不等于独立统计重复。')
        text=text.replace('固定点机制诊断尚待完成。','160项固定点机制诊断已完成，见CORE_MECHANISM_ANALYSIS.md。')
        text=text.replace('新自然transfer与AO04测试尚未闭合。','新自然transfer全部824配置及AO04全部12配置的冻结测试已完成，详见各阶段报告。')
        text=text.replace('当前尚未完成阶段必须继续完整预算、机制诊断、冻结评价和阶段分析。N17新自然数值现已加入；最终版将补齐AO04、N19/CM19的新自然数值，并明确重新评定各历史主张；本表目前不宣称研究闭环。','N17、AO04、N19/CM19全部新自然数值与历史比较现已加入；完整科学结论、阴性结果、矛盾及未识别项见各阶段报告和FINAL_REPORT.md。')
    (ROOT/'NUMERIC_COMPARISONS.md').write_text(text)
    (OUT/'source_files.json').write_text(json.dumps(dict(stage='FINAL' if final else 'INTERIM',sources=sources,comparisons=len(table)),indent=2)+'\n')
    print('Completed four-seed metric rows',len(table))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();build(args.final)
