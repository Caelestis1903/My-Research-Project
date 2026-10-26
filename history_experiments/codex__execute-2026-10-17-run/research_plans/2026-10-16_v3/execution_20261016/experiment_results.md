# 实验结果报告：真实全流程读数

唯一总分类 **BOUNDARY-CONFIRMED**，方法决定 **NO_GO**。冻结test确认的是有条件效应：固定SGD.1共同早期50轮AUC差SB−SN=−.001430（四负，95%seed CI[−.002317,−.000862]），校准Adam.003共同22轮−.009821（四负），校准SGD.3共同37轮则+.001614（四正）；AUC自选test分别−.002518、−.008698、−.000972。固定SGD早期LL差−.000411与AUC方向相反，不能称全面退化。固定SGD/校准Adam的common-SN/common-SB选择读数也保留负向，故selection不是普遍唯一解释。

测试同时限制机制外推：ORDER固定50轮交互−.000123、CI跨零，而AUC自选交互+.003419四正伴SN绝对下降；V8强mask最终d2交互+.003823四正，与诊断−.013904四负相反，两臂绝对test AUC下降；AdaTask早期交互+.003887四正却自选交互−.010219四负。ML共同16轮+.003015四正，固定50轮−.000986四负、自选+.000539为1正3负。阶段、评价人口、选择与绝对健康不能混成一个统一优势。

## 七组baseline的冻结test

宏AUC为六个物理domain×task单元AUC等权均值，不是pooled AUC；按test样本数和train p加权的单元指标另列。

|组|端点|指标|n|SN|SB|SB−SN|95%seed CI|正/负|
|---|---|---|---:|---:|---:|---:|---|---|
|fixedAdam_001|common_epoch50|macro_auc|4|0.668277|0.659714|-0.008562|[-0.020739,+0.003173]|2/2|
|fixedAdam_001|common_epoch50|macro_logloss|4|0.444586|0.446477|+0.001891|[-0.002755,+0.006946]|2/2|
|fixedAdam_001|final_epoch|macro_auc|4|0.668277|0.659714|-0.008562|[-0.020739,+0.003173]|2/2|
|fixedAdam_001|final_epoch|macro_logloss|4|0.444586|0.446477|+0.001891|[-0.002755,+0.006946]|2/2|
|fixedAdam_001|selected_macro_auc|macro_auc|4|0.668646|0.661577|-0.007069|[-0.018602,+0.003197]|2/2|
|fixedAdam_001|selected_macro_auc|macro_logloss|4|0.440392|0.443549|+0.003156|[-0.000227,+0.006539]|3/1|
|fixedSGD_01|common_epoch50|macro_auc|4|0.623542|0.622137|-0.001405|[-0.002183,-0.000627]|0/4|
|fixedSGD_01|common_epoch50|macro_logloss|4|0.451402|0.451110|-0.000293|[-0.012092,+0.011506]|2/2|
|fixedSGD_01|final_epoch|macro_auc|4|0.623542|0.622137|-0.001405|[-0.002183,-0.000627]|0/4|
|fixedSGD_01|final_epoch|macro_logloss|4|0.451402|0.451110|-0.000293|[-0.012092,+0.011506]|2/2|
|fixedSGD_01|selected_macro_auc|macro_auc|4|0.623050|0.620532|-0.002518|[-0.004312,-0.000882]|0/4|
|fixedSGD_01|selected_macro_auc|macro_logloss|4|0.453870|0.456929|+0.003058|[-0.001005,+0.008200]|3/1|
|calAdam_0003|common_epoch50|macro_auc|4|0.639787|0.630713|-0.009074|[-0.017349,-0.000965]|1/3|
|calAdam_0003|common_epoch50|macro_logloss|4|0.510412|0.606793|+0.096381|[+0.084736,+0.108027]|4/0|
|calAdam_0003|final_epoch|macro_auc|4|0.637244|0.626853|-0.010391|[-0.018665,-0.002270]|1/3|
|calAdam_0003|final_epoch|macro_logloss|4|0.525400|0.638733|+0.113333|[+0.096033,+0.135940]|4/0|
|calAdam_0003|selected_macro_auc|macro_auc|4|0.664174|0.655476|-0.008698|[-0.012818,-0.005096]|0/4|
|calAdam_0003|selected_macro_auc|macro_logloss|4|0.446948|0.458027|+0.011079|[-0.000107,+0.022320]|3/1|
|calSGD_03|common_epoch50|macro_auc|4|0.631678|0.632753|+0.001075|[-0.001033,+0.004089]|2/2|
|calSGD_03|common_epoch50|macro_logloss|4|0.447063|0.451497|+0.004434|[-0.006990,+0.013484]|3/1|
|calSGD_03|final_epoch|macro_auc|4|0.635143|0.634710|-0.000434|[-0.004068,+0.003784]|1/3|
|calSGD_03|final_epoch|macro_logloss|4|0.447168|0.446753|-0.000415|[-0.008662,+0.006975]|2/2|
|calSGD_03|selected_macro_auc|macro_auc|4|0.636117|0.635145|-0.000972|[-0.001453,-0.000527]|0/4|
|calSGD_03|selected_macro_auc|macro_logloss|4|0.445168|0.451186|+0.006018|[+0.001106,+0.010929]|3/1|
|batch2SGD_03|common_epoch50|macro_auc|4|0.619139|0.618164|-0.000975|[-0.002953,+0.001846]|1/3|
|batch2SGD_03|common_epoch50|macro_logloss|4|0.448565|0.449771|+0.001207|[-0.009477,+0.012756]|2/2|
|batch2SGD_03|final_epoch|macro_auc|4|0.621333|0.621546|+0.000213|[-0.001851,+0.002277]|2/2|
|batch2SGD_03|final_epoch|macro_logloss|4|0.450608|0.452351|+0.001742|[-0.009477,+0.013292]|3/1|
|batch2SGD_03|selected_macro_auc|macro_auc|4|0.621480|0.621228|-0.000253|[-0.002401,+0.002265]|1/3|
|batch2SGD_03|selected_macro_auc|macro_logloss|4|0.456065|0.457008|+0.000944|[-0.008595,+0.006375]|3/1|
|MovieLensSGD_03|common_epoch50|macro_auc|4|0.569751|0.568765|-0.000986|[-0.001331,-0.000419]|0/4|
|MovieLensSGD_03|common_epoch50|macro_logloss|4|0.585458|0.583436|-0.002023|[-0.003170,-0.000967]|0/4|
|MovieLensSGD_03|final_epoch|macro_auc|4|0.569751|0.568765|-0.000986|[-0.001331,-0.000419]|0/4|
|MovieLensSGD_03|final_epoch|macro_logloss|4|0.585458|0.583436|-0.002023|[-0.003170,-0.000967]|0/4|
|MovieLensSGD_03|selected_macro_auc|macro_auc|4|0.573755|0.574293|+0.000539|[-0.001331,+0.003391]|1/3|
|MovieLensSGD_03|selected_macro_auc|macro_logloss|4|0.585720|0.584025|-0.001695|[-0.003099,-0.000776]|0/4|
|AdaTask_0003|common_epoch50|macro_auc|4|0.646879|0.641072|-0.005807|[-0.016714,+0.003575]|2/2|
|AdaTask_0003|common_epoch50|macro_logloss|4|0.478859|0.517902|+0.039043|[+0.031602,+0.044978]|4/0|
|AdaTask_0003|final_epoch|macro_auc|4|0.647109|0.640283|-0.006826|[-0.019770,+0.003575]|2/2|
|AdaTask_0003|final_epoch|macro_logloss|4|0.481948|0.533021|+0.051073|[+0.031615,+0.079570]|4/0|
|AdaTask_0003|selected_macro_auc|macro_auc|4|0.667236|0.648319|-0.018917|[-0.022075,-0.015759]|0/4|
|AdaTask_0003|selected_macro_auc|macro_logloss|4|0.442278|0.458686|+0.016408|[+0.011732,+0.022356]|4/0|


每组四seed真实值、逐域/任务、六cell、AUC/LL以及AULC/四选择/common-SN/common-SB选择规则均见[14cohort清单](test_summary_verifier_manifest.json)。上表final与fixed-budget为别名；最终预算可随seed不同，不能把own-final当统一epoch。固定100可能只有预算条件n2/n1支持，Q45要求不写四seed。n4精确双侧符号随机化最小p=.125；CI为20k训练seed配对bootstrap，1000选择bootstrap是另一个条件不确定性。

[宏观test图PNG](figures/final_test_selected_macro_gaps.png)/[PDF](figures/final_test_selected_macro_gaps.pdf)、[六cell图PNG](figures/final_test_all_six_cells.png)/[PDF](figures/final_test_all_six_cells.pdf)、[原始画图数值](archive/formal/final_test_selected_all_six_cell_plot_values.csv)。六cell/域/任务不是独立训练seed；图中正负方向需分别看AUC与LL。

## 既定反事实的冻结test交互

|cohort|条件|端点|n|SN绝对变化|SB绝对变化|q-gap交互|95%CI|正/负|
|---|---|---|---:|---:|---:|---:|---|---|
|V10_final|AdaTask_final|common_epoch50|4|+0.007092|+0.010359|+0.003267|[-0.011422,+0.013639]|3/1|
|V10_final|AdaTask_final|common_pre_AULC_1_14|4|-0.000344|+0.003542|+0.003887|[+0.002443,+0.005560]|4/0|
|V10_final|AdaTask_final|final_epoch|4|+0.009865|+0.013430|+0.003565|[-0.010528,+0.013639]|3/1|
|V10_final|AdaTask_final|selected_macro_auc|4|+0.003062|-0.007156|-0.010219|[-0.014092,-0.006345]|0/4|
|V11_final|beta1_zero_v_only|common_epoch50|4|+0.019035|+0.015731|-0.003304|[-0.012358,+0.009404]|1/3|
|V11_final|beta1_zero_v_only|common_pre_AULC_1_4|4|+0.000168|+0.006399|+0.006232|[+0.001737,+0.011546]|4/0|
|V11_final|beta1_zero_v_only|final_epoch|4|-0.008845|+0.004726|+0.013570|[+0.005326,+0.026289]|4/0|
|V11_final|beta1_zero_v_only|selected_macro_auc|4|-0.012374|-0.002397|+0.009978|[-0.004563,+0.024997]|3/1|
|V11_final|beta2_zero|common_epoch50|4|-0.141397|-0.129958|+0.011439|[+0.002305,+0.021188]|3/1|
|V11_final|beta2_zero|common_pre_AULC_1_4|4|-0.115653|-0.108872|+0.006780|[+0.004406,+0.009720]|4/0|
|V11_final|beta2_zero|final_epoch|4|-0.137244|-0.126853|+0.010391|[+0.002270,+0.018665]|3/1|
|V11_final|beta2_zero|selected_macro_auc|4|-0.159244|-0.149361|+0.009882|[+0.006610,+0.014297]|4/0|
|V11_final|m_only|common_epoch50|4|-0.103293|-0.093929|+0.009364|[+0.001517,+0.017675]|3/1|
|V11_final|m_only|common_pre_AULC_1_4|4|-0.097996|-0.091836|+0.006159|[+0.003608,+0.008711]|4/0|
|V11_final|m_only|final_epoch|4|-0.059724|-0.050371|+0.009353|[+0.001749,+0.017389]|3/1|
|V11_final|m_only|selected_macro_auc|4|-0.087313|-0.078085|+0.009228|[+0.005256,+0.013162]|4/0|
|V4_batch_final|B2_sgd|common_epoch50|4|-0.012539|-0.014588|-0.002049|[-0.003669,-0.000308]|1/3|
|V4_batch_final|B2_sgd|common_pre_AULC_1_37|4|-0.022048|-0.024901|-0.002853|[-0.003950,-0.001757]|0/4|
|V4_batch_final|B2_sgd|final_epoch|4|-0.013810|-0.013164|+0.000647|[-0.002171,+0.004419]|2/2|
|V4_batch_final|B2_sgd|selected_macro_auc|4|-0.014636|-0.013918|+0.000719|[-0.001735,+0.003097]|3/1|
|V4_optimizer_final|calAdam_0003|common_epoch50|4|+0.008109|-0.002040|-0.010149|[-0.018297,-0.003555]|0/4|
|V4_optimizer_final|calAdam_0003|common_pre_AULC_1_22|4|+0.076705|+0.065256|-0.011449|[-0.014339,-0.008969]|0/4|
|V4_optimizer_final|calAdam_0003|final_epoch|4|+0.002101|-0.007856|-0.009957|[-0.014597,-0.006054]|0/4|
|V4_optimizer_final|calAdam_0003|selected_macro_auc|4|+0.028057|+0.020330|-0.007727|[-0.011288,-0.004589]|0/4|
|V5_Adam_final|Adam_PERM|common_epoch50|4|+0.010401|-0.007410|-0.017810|[-0.030685,-0.002256]|1/3|
|V5_Adam_final|Adam_PERM|common_pre_AULC_1_22|4|+0.002953|+0.000887|-0.002066|[-0.003183,-0.000921]|0/4|
|V5_Adam_final|Adam_PERM|final_epoch|4|+0.005067|-0.003740|-0.008807|[-0.026680,+0.009065]|2/2|
|V5_Adam_final|Adam_PERM|selected_macro_auc|4|+0.008692|+0.001293|-0.007399|[-0.017120,+0.003432]|1/3|
|V5_SGD_final|SGD_ORDER|common_epoch50|4|+0.001224|+0.001101|-0.000123|[-0.003748,+0.003087]|3/1|
|V5_SGD_final|SGD_ORDER|common_pre_AULC_1_37|4|+0.000622|-0.000306|-0.000929|[-0.002529,+0.000657]|2/2|
|V5_SGD_final|SGD_ORDER|final_epoch|4|+0.000872|+0.000469|-0.000403|[-0.003884,+0.002942]|2/2|
|V5_SGD_final|SGD_ORDER|selected_macro_auc|4|-0.001998|+0.001420|+0.003419|[+0.000938,+0.006087]|4/0|
|V5_SGD_final|SGD_PERM|common_epoch50|4|+0.001135|-0.001528|-0.002663|[-0.007673,+0.002347]|2/2|
|V5_SGD_final|SGD_PERM|common_pre_AULC_1_37|4|+0.000869|-0.000774|-0.001643|[-0.004842,+0.002001]|1/3|
|V5_SGD_final|SGD_PERM|final_epoch|4|+0.000335|-0.001872|-0.002207|[-0.007673,+0.003258]|2/2|
|V5_SGD_final|SGD_PERM|selected_macro_auc|4|+0.001447|+0.001097|-0.000351|[-0.004296,+0.003595]|2/2|
|V8_final|mild|common_epoch50|4|-0.001719|-0.001088|+0.000631|[-0.001887,+0.003149]|2/2|
|V8_final|mild|common_pre_AULC_1_37|4|-0.001992|-0.002116|-0.000125|[-0.000707,+0.000272]|2/2|
|V8_final|mild|final_epoch|4|-0.001986|+0.000125|+0.002110|[-0.003225,+0.007446]|2/2|
|V8_final|mild|selected_macro_auc|4|-0.004696|+0.001703|+0.006399|[+0.002218,+0.012309]|4/0|
|V8_final|strong|common_epoch50|4|-0.008458|-0.008324|+0.000134|[-0.002850,+0.001942]|3/1|
|V8_final|strong|common_pre_AULC_1_37|4|-0.007668|-0.007092|+0.000576|[-0.000128,+0.001056]|3/1|
|V8_final|strong|final_epoch|4|-0.008656|-0.009497|-0.000841|[-0.004115,+0.002433]|2/2|
|V8_final|strong|selected_macro_auc|4|-0.009425|-0.008394|+0.001031|[-0.002426,+0.004119]|3/1|


上述只是全量表中预定义共同早期、E50、最终与AUC自选的宏AUC摘录；所有LL/域任务/六cell及每seed绝对变化、支持交集/排除key均在[完整交互](analysis/test_all_cohorts_primary_verification/test_contrast_summary.json)、[逐seed表](analysis/test_all_cohorts_primary_verification/seed_test_contrasts.csv)。不同训练预算/选择池/阶段不能混作同一因果对照；差距缩小还须结合SN/SB绝对健康，不能自动叫修复。

## 诊断轨迹、训练目标与机制读数

固定SGD.1共同pre50 AUC−.003043四负，fixedAdam.01 pre27−.006980四负，calAdam.003 pre22−.013951四负，calSGD.3 pre37−.000493为1正3负。V5 ORDER E50交互+.005816四正，但seed25 FINAL350交互−.041913；自选AUC交互+.007723四正伴SN绝对四负。V7 C局部AUC阳性与E40 LL/J冲突并列；V11 β2=0最终floor、v-only健康/差距阶段冲突保留。全部具体CI、每域任务/六cell和trainJ参见下列阶段报告。

|读数|完整报告|
|---|---|
|固定SGD与参考interaction|[V2](V2_fixedSGD_and_interaction_sensitivity_analysis.md)|
|selection与7组baseline|[V3](V3_final_baseline_selection_analysis.md)|
|LR/batch/足预算/20等步/2448hit与censor|[V4](V4_final_budget_and_learning_speed_analysis.md)|
|WR/PERM/ORDER最终|[SGD](V5_SGD_final_matched_analysis.md)、[Adam](V5_Adam_final_matched_analysis.md)|
|均值/trace/谱/CPRE/任务/稀疏时间|[V6](V6_geometry_observational_analysis.md)|
|12局部×7条件|[V7](V7_local_counterfactual_analysis.md)|
|规模/输入proxy/正例率/曝光/576cell上下文|[V8](V8_final_matched_diagnostic_analysis.md)|
|第二环境|[V9](V9_final_cross_environment_analysis.md)|
|AdaTask固定同目标组件|[V10](V10_final_vs_calAdam_analysis.md)|
|完整状态与1740方向记录|[V11](V11_fullstate_final_diagnostic_analysis.md)|

所有实际停止依据、各seed预算、56臂onset与选择epoch在[执行报告](execution_report.md)和[onset CSV](archive/formal/final_baseline_onsets_and_selection_56arm_rows.csv)。完整证据分类与竞争解释见[机制报告](mechanism_analysis.md)；14章最终规范见[FINAL_REPORT](FINAL_REPORT.md)。
