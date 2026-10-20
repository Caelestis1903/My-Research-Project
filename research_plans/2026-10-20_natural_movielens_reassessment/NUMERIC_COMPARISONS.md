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
|AC-S5|full50|final|T-F|0.743962|0.728406|-0.015556（0/4）|-0.063643（0/4）|
|AC-S5|full50|final|HT-F|0.743962|0.749927|+0.005965（4/0）|-0.066344（0/4）|
|AC-S5|full50|final|DH-F|0.743962|0.760022|+0.016059（4/0）|-0.087078（0/4）|
|AC-S5|full50|final|DH-HT|0.749927|0.760022|+0.010094（4/0）|-0.020734（0/4）|
|AC-S5|full50|selected|T-F|0.772135|0.740972|-0.031164（0/4）|+0.015686（4/0）|
|AC-S5|full50|selected|HT-F|0.772135|0.761098|-0.011038（0/4）|+0.001562（3/1）|
|AC-S5|full50|selected|DH-F|0.772135|0.769065|-0.003070（1/3）|+0.009985（4/0）|
|AC-S5|full50|selected|DH-HT|0.761098|0.769065|+0.007968（4/0）|+0.008422（3/1）|
|AC-S5|matched250|final|T-F|0.767067|0.735865|-0.031202（0/4）|+0.010652（4/0）|
|AC-S5|matched250|final|HT-F|0.767067|0.757892|-0.009175（0/4）|-0.000480（2/2）|
|AC-S5|matched250|final|DH-F|0.767067|0.759977|-0.007090（0/4）|-0.000207（2/2）|
|AC-S5|matched250|final|DH-HT|0.757892|0.759977|+0.002085（3/1）|+0.000273（3/1）|
|AC-S5|matched250|selected|T-F|0.767579|0.738540|-0.029039（0/4）|+0.004551（3/1）|
|AC-S5|matched250|selected|HT-F|0.767579|0.757820|-0.009759（0/4）|-0.007626（0/4）|
|AC-S5|matched250|selected|DH-F|0.767579|0.760251|-0.007328（0/4）|-0.004891（1/3）|
|AC-S5|matched250|selected|DH-HT|0.757820|0.760251|+0.002430（4/0）|+0.002735（2/2）|
|AC-S6|full50|final|DH-F|0.743109|0.758767|+0.015659（4/0）|-0.079539（0/4）|
|AC-S6|full50|selected|DH-F|0.772478|0.769032|-0.003445（0/4）|+0.014732（4/0）|
|AC-S6|matched250|final|DH-F|0.766854|0.761512|-0.005342（0/4）|+0.003304（4/0）|
|AC-S6|matched250|selected|DH-F|0.767585|0.759869|-0.007716（0/4）|+0.000620（2/2）|
|AC-V9 full trajectory|50epoch/1000main|AULC_1_10|SB-SN|0.565929|0.566409|+0.000480（2/2）|+0.000186（2/2）|
|AC-V9 full trajectory|50epoch/1000main|AULC_1_16|SB-SN|0.571613|0.572086|+0.000473（4/0）|+0.000708（3/1）|
|AC-V9 full trajectory|50epoch/1000main|AULC_1_20|SB-SN|0.575119|0.575590|+0.000471（3/1）|+0.000692（2/2）|
|AC-V9 full trajectory|50epoch/1000main|AULC_1_50|SB-SN|0.628514|0.628010|-0.000504（1/3）|-0.003140（1/3）|
|N17|formal/formal; LR0.0003/0.0003|AULC1_100|formal_sameLR0.0003_SB-SN|0.752355|0.752125|-0.000230（2/2）|+0.001311（3/1）|
|N17|formal/formal; LR0.0003/0.0003|AULC1_20|formal_sameLR0.0003_SB-SN|0.751475|0.751943|+0.000468（4/0）|+0.000153（3/1）|
|N17|formal/formal; LR0.0003/0.0003|final|formal_sameLR0.0003_SB-SN|0.741064|0.741874|+0.000811（3/1）|-0.000108（3/1）|
|N17|formal/formal; LR0.0003/0.0003|selected|formal_sameLR0.0003_SB-SN|0.761950|0.762467|+0.000516（3/1）|-0.001656（0/4）|
|N17|formal/formal; LR0.001/0.001|AULC1_100|formal_sameLR0.001_SB-SN|0.741348|0.740454|-0.000894（2/2）|+0.008266（4/0）|
|N17|formal/formal; LR0.001/0.001|AULC1_20|formal_sameLR0.001_SB-SN|0.758993|0.758706|-0.000286（1/3）|+0.000510（3/1）|
|N17|formal/formal; LR0.001/0.001|final|formal_sameLR0.001_SB-SN|0.719539|0.718253|-0.001286（0/4）|+0.014578（3/1）|
|N17|formal/formal; LR0.001/0.001|selected|formal_sameLR0.001_SB-SN|0.762752|0.762879|+0.000127（2/2）|-0.000661（1/3）|
|N17|formal/formal; LR0.01/0.01|AULC1_100|formal_sameLR0.01_SB-SN|0.746437|0.745605|-0.000832（1/3）|+0.006265（3/1）|
|N17|formal/formal; LR0.01/0.01|AULC1_20|formal_sameLR0.01_SB-SN|0.762221|0.762287|+0.000066（2/2）|+0.000036（2/2）|
|N17|formal/formal; LR0.01/0.01|final|formal_sameLR0.01_SB-SN|0.729910|0.728723|-0.001187（1/3）|+0.016585（3/1）|
|N17|formal/formal; LR0.01/0.01|selected|formal_sameLR0.01_SB-SN|0.767539|0.767283|-0.000256（1/3）|+0.001775（2/2）|
|N17|matched_budget/matched_budget; LR0.0003/0.0003|AULC1_100|matched_budget_sameLR0.0003_SB-SN|0.750920|0.751078|+0.000158（4/0）|-0.000091（3/1）|
|N17|matched_budget/matched_budget; LR0.0003/0.0003|AULC1_20|matched_budget_sameLR0.0003_SB-SN|0.706907|0.706065|-0.000842（1/3）|-0.000128（1/3）|
|N17|matched_budget/matched_budget; LR0.0003/0.0003|final|matched_budget_sameLR0.0003_SB-SN|0.760994|0.761503|+0.000509（3/1）|-0.001220（1/3）|
|N17|matched_budget/matched_budget; LR0.0003/0.0003|selected|matched_budget_sameLR0.0003_SB-SN|0.762065|0.762431|+0.000366（3/1）|-0.002472（2/2）|
|N17|matched_budget/matched_budget; LR0.001/0.001|AULC1_100|matched_budget_sameLR0.001_SB-SN|0.756870|0.756782|-0.000089（2/2）|+0.000708（3/1）|
|N17|matched_budget/matched_budget; LR0.001/0.001|AULC1_20|matched_budget_sameLR0.001_SB-SN|0.736158|0.735932|-0.000226（1/3）|-0.000282（1/3）|
|N17|matched_budget/matched_budget; LR0.001/0.001|final|matched_budget_sameLR0.001_SB-SN|0.759339|0.760515|+0.001176（4/0）|-0.000460（2/2）|
|N17|matched_budget/matched_budget; LR0.001/0.001|selected|matched_budget_sameLR0.001_SB-SN|0.763433|0.763131|-0.000302（2/2）|-0.004533（0/4）|
|N17|quota/quota; LR0.01/0.01|AULC1_100|quota_sameLR0.01_SB-SN|0.745121|0.745350|+0.000229（3/1）|+0.003080（3/1）|
|N17|quota/quota; LR0.01/0.01|AULC1_20|quota_sameLR0.01_SB-SN|0.762496|0.762484|-0.000012（2/2）|-0.000504（1/3）|
|N17|quota/quota; LR0.01/0.01|final|quota_sameLR0.01_SB-SN|0.726298|0.727314|+0.001016（2/2）|+0.003246（3/1）|
|N17|quota/quota; LR0.01/0.01|selected|quota_sameLR0.01_SB-SN|0.767658|0.766754|-0.000904（1/3）|+0.005308（3/1）|
|N17|H200/H200; LR0.01/0.01|AULC1_100|H200_sameLR0.01_SB-SN|0.746437|0.745605|-0.000832（1/3）|+0.006265（3/1）|
|N17|H200/H200; LR0.01/0.01|AULC1_20|H200_sameLR0.01_SB-SN|0.762221|0.762287|+0.000066（2/2）|+0.000036（2/2）|
|N17|H200/H200; LR0.01/0.01|AULC1_200|H200_sameLR0.01_SB-SN|0.734448|0.734207|-0.000241（1/3）|+0.006098（3/1）|
|N17|H200/H200; LR0.01/0.01|final|H200_sameLR0.01_SB-SN|0.718136|0.719219|+0.001083（3/1）|+0.009069（2/2）|
|N17|H200/H200; LR0.01/0.01|selected|H200_sameLR0.01_SB-SN|0.767539|0.767283|-0.000256（1/3）|+0.001775（2/2）|
|N17|formal/formal; LR0.01/0.01|AULC1_100|formal_commonLR_SB-SN|0.746437|0.745605|-0.000832（1/3）|+0.006265（3/1）|
|N17|formal/formal; LR0.01/0.01|AULC1_20|formal_commonLR_SB-SN|0.762221|0.762287|+0.000066（2/2）|+0.000036（2/2）|
|N17|formal/formal; LR0.01/0.01|final|formal_commonLR_SB-SN|0.729910|0.728723|-0.001187（1/3）|+0.016585（3/1）|
|N17|formal/formal; LR0.01/0.01|selected|formal_commonLR_SB-SN|0.767539|0.767283|-0.000256（1/3）|+0.001775（2/2）|
|N17|formal/formal; LR0.01/0.01|AULC1_100|formal_ownLR_SB-SN|0.746437|0.745605|-0.000832（1/3）|+0.006265（3/1）|
|N17|formal/formal; LR0.01/0.01|AULC1_20|formal_ownLR_SB-SN|0.762221|0.762287|+0.000066（2/2）|+0.000036（2/2）|
|N17|formal/formal; LR0.01/0.01|final|formal_ownLR_SB-SN|0.729910|0.728723|-0.001187（1/3）|+0.016585（3/1）|
|N17|formal/formal; LR0.01/0.01|selected|formal_ownLR_SB-SN|0.767539|0.767283|-0.000256（1/3）|+0.001775（2/2）|
|N17|quota/formal; LR0.01/0.01|AULC1_100|quota-minus-IID_SN|0.746437|0.745121|-0.001316（0/4）|+0.004692（4/0）|
|N17|quota/formal; LR0.01/0.01|AULC1_20|quota-minus-IID_SN|0.762221|0.762496|+0.000275（4/0）|+0.001781（4/0）|
|N17|quota/formal; LR0.01/0.01|final|quota-minus-IID_SN|0.729910|0.726298|-0.003612（1/3）|+0.024313（4/0）|
|N17|quota/formal; LR0.01/0.01|selected|quota-minus-IID_SN|0.767539|0.767658|+0.000119（2/2）|-0.004477（1/3）|
|N17|formal/matched_budget; LR0.0003/0.0003|AULC1_100|formal-minus-matched_budget_SN_lr0.0003|0.750920|0.752355|+0.001434（3/1）|+0.040126（4/0）|
|N17|formal/matched_budget; LR0.0003/0.0003|AULC1_20|formal-minus-matched_budget_SN_lr0.0003|0.706907|0.751475|+0.044568（4/0）|-0.035385（0/4）|
|N17|formal/matched_budget; LR0.0003/0.0003|final|formal-minus-matched_budget_SN_lr0.0003|0.760994|0.741064|-0.019931（0/4）|+0.116052（4/0）|
|N17|formal/matched_budget; LR0.0003/0.0003|selected|formal-minus-matched_budget_SN_lr0.0003|0.762065|0.761950|-0.000115（2/2）|-0.000036（2/2）|
|N17|formal/matched_budget; LR0.001/0.001|AULC1_100|formal-minus-matched_budget_SN_lr0.001|0.756870|0.741348|-0.015523（0/4）|+0.128863（4/0）|
|N17|formal/matched_budget; LR0.001/0.001|AULC1_20|formal-minus-matched_budget_SN_lr0.001|0.736158|0.758993|+0.022835（4/0）|-0.010771（0/4）|
|N17|formal/matched_budget; LR0.001/0.001|final|formal-minus-matched_budget_SN_lr0.001|0.759339|0.719539|-0.039801（0/4）|+0.286953（4/0）|
|N17|formal/matched_budget; LR0.001/0.001|selected|formal-minus-matched_budget_SN_lr0.001|0.763433|0.762752|-0.000681（2/2）|-0.001214（2/2）|
|N17|H200/formal; LR0.01/0.01|AULC1_100|H200-minus-H100_SN|0.746437|0.746437|+0.000000（0/0）|+0.000000（0/0）|
|N17|H200/formal; LR0.01/0.01|AULC1_20|H200-minus-H100_SN|0.762221|0.762221|+0.000000（0/0）|+0.000000（0/0）|
|N17|H200/formal; LR0.01/0.01|final|H200-minus-H100_SN|0.729910|0.718136|-0.011774（0/4）|+0.102583（4/0）|
|N17|H200/formal; LR0.01/0.01|selected|H200-minus-H100_SN|0.767539|0.767539|+0.000000（0/0）|+0.000000（0/0）|
|N17|quota/formal; LR0.01/0.01|AULC1_100|quota-minus-IID_SB|0.745605|0.745350|-0.000256（2/2）|+0.001508（3/1）|
|N17|quota/formal; LR0.01/0.01|AULC1_20|quota-minus-IID_SB|0.762287|0.762484|+0.000196（2/2）|+0.001241（3/1）|
|N17|quota/formal; LR0.01/0.01|final|quota-minus-IID_SB|0.728723|0.727314|-0.001409（1/3）|+0.010974（2/2）|
|N17|quota/formal; LR0.01/0.01|selected|quota-minus-IID_SB|0.767283|0.766754|-0.000529（1/3）|-0.000944（1/3）|
|N17|formal/matched_budget; LR0.0003/0.0003|AULC1_100|formal-minus-matched_budget_SB_lr0.0003|0.751078|0.752125|+0.001047（2/2）|+0.041527（4/0）|
|N17|formal/matched_budget; LR0.0003/0.0003|AULC1_20|formal-minus-matched_budget_SB_lr0.0003|0.706065|0.751943|+0.045878（4/0）|-0.035104（0/4）|
|N17|formal/matched_budget; LR0.0003/0.0003|final|formal-minus-matched_budget_SB_lr0.0003|0.761503|0.741874|-0.019629（0/4）|+0.117165（4/0）|
|N17|formal/matched_budget; LR0.0003/0.0003|selected|formal-minus-matched_budget_SB_lr0.0003|0.762431|0.762467|+0.000036（1/3）|+0.000779（3/1）|
|N17|formal/matched_budget; LR0.001/0.001|AULC1_100|formal-minus-matched_budget_SB_lr0.001|0.756782|0.740454|-0.016328（0/4）|+0.136422（4/0）|
|N17|formal/matched_budget; LR0.001/0.001|AULC1_20|formal-minus-matched_budget_SB_lr0.001|0.735932|0.758706|+0.022774（4/0）|-0.009980（0/4）|
|N17|formal/matched_budget; LR0.001/0.001|final|formal-minus-matched_budget_SB_lr0.001|0.760515|0.718253|-0.042262（0/4）|+0.301990（4/0）|
|N17|formal/matched_budget; LR0.001/0.001|selected|formal-minus-matched_budget_SB_lr0.001|0.763131|0.762879|-0.000252（1/3）|+0.002658（2/2）|
|N17|H200/formal; LR0.01/0.01|AULC1_100|H200-minus-H100_SB|0.745605|0.745605|+0.000000（0/0）|+0.000000（0/0）|
|N17|H200/formal; LR0.01/0.01|AULC1_20|H200-minus-H100_SB|0.762287|0.762287|+0.000000（0/0）|+0.000000（0/0）|
|N17|H200/formal; LR0.01/0.01|final|H200-minus-H100_SB|0.728723|0.719219|-0.009504（0/4）|+0.095068（4/0）|
|N17|H200/formal; LR0.01/0.01|selected|H200-minus-H100_SB|0.767283|0.767283|+0.000000（0/0）|+0.000000（0/0）|

## 2. 人口、预算及历史对照

人工MovieLens、自然KuaiRand、新自然MovieLens、匹配总N与匹配曝光条件逐行保存在[完整宏比较CSV](analysis/numeric_comparisons/completed_four_seed_macro_comparisons.csv)。不同family有不同融合、优化器、选模和主/arch时钟，不能把它们当单因素消融合并排名。

- [V2阶段报告](V2_PHASE_ANALYSIS.md)：完整固定模型Adam/SGD及旧两环境，selected/final、逐域任务、预测复算。
- [AC-2D性能报告](AC2D_PERFORMANCE_ANALYSIS.md)：四臂×四人口/预算条件，含匹配总N与同250步全支持；固定点机制诊断尚待完成。
- [S5完整阶段报告](ACS5_PHASE_ANALYSIS.md)：32配置、两预算、历史/当前梯度、完整逐cell与预算交互，selected/末轮反转及域间取舍。
- [S6阶段报告](ACS6_PHASE_ANALYSIS.md)：Sample F/DH，两预算全部16配置已分析，保留预算×阶段反转。
- [V9阶段报告](ACV9_PERFORMANCE_ANALYSIS.md)：SGD .3八训练及八只读任务闭合，400份全轨迹预测2400cell复算，保留四个预设窗口。
- [N18报告](N18_PHASE_ANALYSIS.md)：自然/人工人口的同信息LP、审核SE/RMSE，非训练AUC比较。
- [历史B1、N17、N19/CM19及AO04重建](HISTORICAL_NUMERIC_RECONSTRUCTION.md)：1184个旧run、747个完整四seed配方比较；AO04另列旧ML两seed、KR一seed，不混入四seed表。新自然N17的72正式配置test已完成，见[N17完整报告](N17_PHASE_ANALYSIS.md)；新自然transfer与AO04测试尚未闭合。
- [特征支持审计](FEATURE_SUPPORT_AUDIT.md)：匹配总N仍未控制ID支持，属于已量化的竞争解释，不是已识别的AUC中介效应。

## 3. 逐cell、任务和阶段的读取规则

对应阶段的`paired_cell_deltas.csv`、`paired_domain_deltas.csv`、`paired_task_deltas.csv`与原基线绝对值共同用于科学判断。最大域不自动等于最容易，最小域不自动等于最困难。一个宏均值不能替代两任务/三域是否同时受益，也不能替代selected与final的风险变化。

当前尚未完成阶段必须继续完整预算、机制诊断、冻结评价和阶段分析。N17新自然数值现已加入；最终版将补齐AO04、N19/CM19的新自然数值，并明确重新评定各历史主张；本表目前不宣称研究闭环。
