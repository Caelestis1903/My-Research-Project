# 历史与自然人口数值比较（阶段性版本）

**整体实验仍在进行。** 本表只合并已完成主代理分析、具备完整四种子比较的阶段；待完成训练和未解锁test均不填零。当前源文件清单见[JSON](analysis/numeric_comparisons/source_files.json)。

差值均为候选−参照；AUC越大越好，LogLoss越小越好。宏值为六个domain×task cell的等权平均；不是pooled AUC。区间为四个优化seed的点态描述性t95，不是新总体或多重比较校正后的确认。旧V2/V9优化seed同时改变人工子集，新人口固定。

## 1. 自然完整人口的同配置比较

|协议|预算|阶段|候选−参照|参照AUC|候选AUC|ΔAUC（正/负seed）|ΔLL（变差/改善seed）|
|---|---|---|---|---:|---:|---:|---:|
|V2 fixed adam|50epoch/1000main|final|SB-SN|0.746414|0.742331|-0.004084（0/4）|+0.026449（4/0）|
|V2 fixed adam|50epoch/1000main|selected|SB-SN|0.766690|0.767169|+0.000479（2/2）|-0.001023（1/3）|
|V2 fixed sgd|50epoch/1000main|final|SB-SN|0.643088|0.638829|-0.004258（1/3）|-0.006715（1/3）|
|V2 fixed sgd|50epoch/1000main|selected|SB-SN|0.643088|0.638829|-0.004258（1/3）|-0.006715（1/3）|
|AC-2D|full50|final|SB-SN|0.743770|0.741013|-0.002757（1/3）|+0.007524（3/1）|
|AC-2D|full50|final|EB-EN|0.744743|0.740080|-0.004663（0/4）|+0.033817（4/0）|
|AC-2D|full50|final|EN-SN|0.743770|0.744743|+0.000973（3/1）|+0.012372（4/0）|
|AC-2D|full50|final|EB-SB|0.741013|0.740080|-0.000933（1/3）|+0.038666（4/0）|
|AC-2D|full50|selected|SB-SN|0.767673|0.767625|-0.000047（2/2）|+0.003182（3/1）|
|AC-2D|full50|selected|EB-EN|0.767060|0.766964|-0.000096（2/2）|+0.003564（2/2）|
|AC-2D|full50|selected|EN-SN|0.767673|0.767060|-0.000613（1/3）|-0.000510（2/2）|
|AC-2D|full50|selected|EB-SB|0.767625|0.766964|-0.000661（0/4）|-0.000128（2/2）|
|AC-2D|matched250|final|SB-SN|0.766507|0.767554|+0.001047（3/1）|-0.001959（1/3）|
|AC-2D|matched250|final|EB-EN|0.767117|0.766461|-0.000656（1/3）|-0.004189（2/2）|
|AC-2D|matched250|final|EN-SN|0.766507|0.767117|+0.000610（2/2）|+0.004799（3/1）|
|AC-2D|matched250|final|EB-SB|0.767554|0.766461|-0.001093（0/4）|+0.002569（3/1）|
|AC-2D|matched250|selected|SB-SN|0.766009|0.767270|+0.001261（3/1）|+0.000492（3/1）|
|AC-2D|matched250|selected|EB-EN|0.766745|0.768083|+0.001337（4/0）|+0.001339（2/2）|
|AC-2D|matched250|selected|EN-SN|0.766009|0.766745|+0.000736（2/2）|+0.004663（3/1）|
|AC-2D|matched250|selected|EB-SB|0.767270|0.768083|+0.000813（3/1）|+0.005510（3/1）|
|AC-V9|full50|final|SB-SN|0.724842|0.724685|-0.000157（2/2）|-0.005543（1/3）|
|AC-V9|full50|selected|SB-SN|0.727801|0.726186|-0.001615（1/3）|-0.014968（1/3）|

## 2. 人口、预算及历史对照

人工MovieLens、自然KuaiRand、新自然MovieLens、匹配总N与匹配曝光条件逐行保存在[完整宏比较CSV](analysis/numeric_comparisons/completed_four_seed_macro_comparisons.csv)。不同family有不同融合、优化器、选模和主/arch时钟，不能把它们当单因素消融合并排名。

- [V2阶段报告](V2_PHASE_ANALYSIS.md)：完整固定模型Adam/SGD及旧两环境，selected/final、逐域任务、预测复算。
- [AC-2D性能报告](AC2D_PERFORMANCE_ANALYSIS.md)：四臂×四人口/预算条件，含匹配总N与同250步全支持；固定点机制诊断尚待完成。
- [V9阶段报告](ACV9_PERFORMANCE_ANALYSIS.md)：SGD .3公共点；完整50轮冻结读数只有全部8任务闭合才纳入新AULC。
- [N18报告](N18_PHASE_ANALYSIS.md)：自然/人工人口的同信息LP、审核SE/RMSE，非训练AUC比较。
- [历史N19/CM19及AO04重建](HISTORICAL_NUMERIC_RECONSTRUCTION.md)：1184个旧run、747个完整四seed配方比较；AO04另列旧ML两seed、KR一seed，不混入四seed表。新自然transfer、N17与AO04测试尚未闭合。
- [特征支持审计](FEATURE_SUPPORT_AUDIT.md)：匹配总N仍未控制ID支持，属于已量化的竞争解释，不是已识别的AUC中介效应。

## 3. 逐cell、任务和阶段的读取规则

对应阶段的`paired_cell_deltas.csv`、`paired_domain_deltas.csv`、`paired_task_deltas.csv`与原基线绝对值共同用于科学判断。最大域不自动等于最容易，最小域不自动等于最困难。一个宏均值不能替代两任务/三域是否同时受益，也不能替代selected与final的风险变化。

当前尚未完成阶段必须继续完整预算、机制诊断、冻结评价和阶段分析。最终版将补齐S5/S6、N17、AO04、N19/CM19的新自然数值，并明确重新评定各历史主张；本表目前不宣称研究闭环。
