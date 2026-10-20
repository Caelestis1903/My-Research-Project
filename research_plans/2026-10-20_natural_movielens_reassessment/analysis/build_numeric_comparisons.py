"""Assemble completed, primary-analyzed stage comparisons; never infer run completion."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
ANA=ROOT/'analysis'
OUT=ANA/'numeric_comparisons'

def build():
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
        for r in f[f.readout.isin(['AULC_1_16','AULC_1_50'])].itertuples():
            yield dict(family='AC-V9 full trajectory',population=r.population,
                budget='50epoch/250main' if 'historical' in r.population else '50epoch/1000main',
                readout=r.readout,contrast='SB-SN',metric='macro_'+r.metric,
                reference_mean=r.reference_mean,candidate_mean=r.candidate_mean,n=r.n,delta=r.mean,
                ci95_low=r.ci95_low,ci95_high=r.ci95_high,positive=r.positive,negative=r.negative,zero=r.zero)
    collect(ANA/'v9_history/paired_summary.csv',v9)
    table=pd.DataFrame(rows);assert table.n.eq(4).all()
    table.to_csv(OUT/'completed_four_seed_macro_comparisons.csv',index=False)
    lines=['# 历史与自然人口数值比较（阶段性版本）','',
        '**整体实验仍在进行。** 本表只合并已完成主代理分析、具备完整四种子比较的阶段；待完成训练和未解锁test均不填零。当前源文件清单见[JSON](analysis/numeric_comparisons/source_files.json)。', '',
        '差值均为候选−参照；AUC越大越好，LogLoss越小越好。宏值为六个domain×task cell的等权平均；不是pooled AUC。区间为四个优化seed的点态描述性t95，不是新总体或多重比较校正后的确认。旧V2/V9优化seed同时改变人工子集，新人口固定。', '',
        '## 1. 自然完整人口的同配置比较', '',
        '|协议|预算|阶段|候选−参照|参照AUC|候选AUC|ΔAUC（正/负seed）|ΔLL（变差/改善seed）|',
        '|---|---|---|---|---:|---:|---:|---:|']
    n=table[(table.population=='movielens_natural')&table.metric.eq('macro_auc')]
    for r in n.itertuples():
        l=table[(table.family==r.family)&(table.population==r.population)&(table.budget==r.budget)&
                (table.readout==r.readout)&(table.contrast==r.contrast)&table.metric.eq('macro_logloss')]
        assert len(l)==1;l=l.iloc[0]
        lines.append(f'|{r.family}|{r.budget}|{r.readout}|{r.contrast}|{r.reference_mean:.6f}|{r.candidate_mean:.6f}|{r.delta:+.6f}（{r.positive}/{r.negative}）|{l.delta:+.6f}（{l.positive}/{l.negative}）|')
    lines+=['', '## 2. 人口、预算及历史对照', '',
        '人工MovieLens、自然KuaiRand、新自然MovieLens、匹配总N与匹配曝光条件逐行保存在[完整宏比较CSV](analysis/numeric_comparisons/completed_four_seed_macro_comparisons.csv)。不同family有不同融合、优化器、选模和主/arch时钟，不能把它们当单因素消融合并排名。', '',
        '- [V2阶段报告](V2_PHASE_ANALYSIS.md)：完整固定模型Adam/SGD及旧两环境，selected/final、逐域任务、预测复算。',
        '- [AC-2D性能报告](AC2D_PERFORMANCE_ANALYSIS.md)：四臂×四人口/预算条件，含匹配总N与同250步全支持；固定点机制诊断尚待完成。',
        '- [V9阶段报告](ACV9_PERFORMANCE_ANALYSIS.md)：SGD .3公共点；完整50轮冻结读数只有全部8任务闭合才纳入新AULC。',
        '- [N18报告](N18_PHASE_ANALYSIS.md)：自然/人工人口的同信息LP、审核SE/RMSE，非训练AUC比较。',
        '- [历史N19/CM19及AO04重建](HISTORICAL_NUMERIC_RECONSTRUCTION.md)：1184个旧run、747个完整四seed配方比较；AO04另列旧ML两seed、KR一seed，不混入四seed表。新自然transfer、N17与AO04测试尚未闭合。',
        '- [特征支持审计](FEATURE_SUPPORT_AUDIT.md)：匹配总N仍未控制ID支持，属于已量化的竞争解释，不是已识别的AUC中介效应。', '',
        '## 3. 逐cell、任务和阶段的读取规则', '',
        '对应阶段的`paired_cell_deltas.csv`、`paired_domain_deltas.csv`、`paired_task_deltas.csv`与原基线绝对值共同用于科学判断。最大域不自动等于最容易，最小域不自动等于最困难。一个宏均值不能替代两任务/三域是否同时受益，也不能替代selected与final的风险变化。', '',
        '当前尚未完成阶段必须继续完整预算、机制诊断、冻结评价和阶段分析。最终版将补齐S5/S6、N17、AO04、N19/CM19的新自然数值，并明确重新评定各历史主张；本表目前不宣称研究闭环。']
    (ROOT/'NUMERIC_COMPARISONS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'source_files.json').write_text(json.dumps(dict(stage='INTERIM',sources=sources,comparisons=len(table)),indent=2)+'\n')
    print('Completed four-seed metric rows',len(table))

if __name__=='__main__':build()
