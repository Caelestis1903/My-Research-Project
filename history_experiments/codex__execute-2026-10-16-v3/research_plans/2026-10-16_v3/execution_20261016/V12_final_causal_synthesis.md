# V12 最终因果综合：全部规定诊断和冻结测试完成

**唯一总分类：BOUNDARY-CONFIRMED；方法决定NO_GO。** 全14cohort冻结重建PASS：748376个数值，最大误差5.7732e−15，54208条既定配对对照，4480条final/fixed-budget别名。全部13500原生test NPZ物理SHA通过；36精确前缀不是额外独立模型。主控负责设计/实现/自审/分析，机械runner只执行进程，independent_review=false。

## 核心效应、选择、过拟合与优化器

冻结test确认的是有条件效应：固定SGD.1共同早期50轮AUC差SB−SN=−.001430（四负，95%seed CI[−.002317,−.000862]），校准Adam.003共同22轮−.009821（四负），校准SGD.3共同37轮则+.001614（四正）；AUC自选test分别−.002518、−.008698、−.000972。固定SGD早期LL差−.000411与AUC方向相反，不能称全面退化。固定SGD/校准Adam的common-SN/common-SB选择读数也保留负向，故selection不是普遍唯一解释。

测试同时限制机制外推：ORDER固定50轮交互−.000123、CI跨零，而AUC自选交互+.003419四正伴SN绝对下降；V8强mask最终d2交互+.003823四正，与诊断−.013904四负相反，两臂绝对test AUC下降；AdaTask早期交互+.003887四正却自选交互−.010219四负。ML共同16轮+.003015四正，固定50轮−.000986四负、自选+.000539为1正3负。阶段、评价人口、选择与绝对健康不能混成一个统一优势。

共同早期、固定轮次、own-final和validation自选是不同估计量。操作性onset不证明绝无过拟合；全部56臂原始onset/选轮已保留。校准Adam相对SGD共同22 test交互−.011449、四负，CI[−.014339,−.008969]；E50与selected交互也四负。它支持配置内optimizer响应差异（MEDIUM），并不唯一识别历史。fixed参考interaction不触发的诊断反例仍保留；无动量SGD反驳普遍Adam必要性。

## 六项规定反事实

|控制|实际完成与结果|有限判断与边界|
|---|---|---|
|CF1 同J改q|16θ×512/q，完整图导数字段与精确控制；四seed轨迹、选择及test|同图/同均值不同协方差STRONG；配置内性能路径MEDIUM；不唯一识别中介|
|CF2 同q改optimizer|固定/校准Adam/SGD、V10与V11所有触发状态和影子已完成|optimizer/LR/阶段交互；整体状态映射MEDIUM，不是唯一m/v历史必要性|
|CF3 域内sampler|Adam最终50/100/100/50、SGD50/50/50/350，40训练4200臂轮205800批；test完整|Adam PERM early交互−.002066四负；SGD PERM early/selected交互混合。R=E−U≥max(0,E−N)，coverage/repetition/依赖未单独分离|
|CF4 同multiset改order|四seed所有最终和22规定前缀、964/1000短尾移动审计|diagnostic E50+.005816与test−.000123不稳定复现；test selected+.003419四正伴SN绝对下降。不是课程策略或稳定健康恢复|
|CF5 同θ换CPRE|12局部×7条件、全部4788绝对读数/12996对照/3249组重建；12影子与1740方向行|C E10/E40 AUC筛选阳性；E40 LL/J各2正2负且平均衰减负，原联合门槛不满足；full-training NOT_TRIGGERED。R已私有，globalnorm不保持块尺度|
|CF6 batch/确定性|2B全四seed足预算、20等步记录、2448hit/censor、exact/stochastic对照|B2 test early−.001236四负、selected−.000253混合；不是普遍单纯速度证明。预算实用平台不等于全局收敛|

## 训练阶段与评价人口的反证

V5 ORDER：diagnostic E50正交互四seed而test同轮不稳定；两个总体可有不同排序，不能只留一个总体。own-final交互test−.000403为2正2负，selected+.003419四正，其中SN−.001998/SB+.001420；selected loss交互−.014422四负，SN loss+.008022/SB−.006400。gap收缩包含SN变差，不能当全面有效方法。早期与final宽CI没有被用来宣称等效零，因此不以该null触发或豁免核心必要性判断。

V8 strong d2 own-final：diagnostic交互−.013904四负，test+.003823四正、CI[+.001232,+.006566]；test绝对SN−.030257/SB−.026434，两臂受损。mild d2 own-final test交互+.004506四正也伴SN−.008404/SB−.003898。mild selected宏交互+.006399四正，但目标d2两臂分别−.009954/−.006239。mask并非健康恢复方法；固定N/标签/域序列三severity内操纵身份STRONG，但跨severity J_c与表示law改变、只有一个固定嵌套mask，不是自然困难度唯一因果。

V10 AdaTask：test early14交互+.003887四正、CI[+.002443,+.005560]，SN−.000344/SB+.003542；selected交互−.010219四负，SN+.003062/SB−.007156；own-final交互+.003565、3正1负、CI跨零，与diagnostic最终交互四负不同。原目标固定组件保留q敏感性，不能判断完整论文方法全调参失败。

V11 v-only：test E50两臂绝对AUC+.019035/+.015731而交互−.003304不稳定；own-final交互+.013571四正，但SN−.008845/SB+.004726，且预算不匹配native；selected交互+.009978、3正1负而两臂绝对−.012374/−.002397。m-only selected交互+.009228四正伴两臂−.087313/−.078085；β2=0 own-final q-gap=0而两臂AUC=.5。破坏性floor不能识别必要中介，也不能把相对gap改善等同绝对健康。

## 环境、域、任务、规模与困难度

冻结test共同16轮：KR +.001185（3正1负）、ML +.003015（四正）；E50分别+.001075（2正2负）/−.000986（四负）；AUC自选分别−.000972（四负）/+.000539（1正3负）。ML E50出现负向但early为正，不能遗漏也不能反向改动冻结前基于diagnostic early判据而未触发的深层复制分支。49/5步、数据/标签/子集配置束和总体评价人口依然限制归因。

所选test的校准Adam：d1/d2 AUC差分别−.008258/−.013382，均四负；两task分别−.008077/−.009319，均四负；d0混合。AdaTask selected六个域×任务AUC单元全部各四负，同时d0/d2 loss变差。校准SGD selected最小d2 AUC+.001803（3正1负），最大d1−.001854（1正3负），存在局部小/大AUC取舍；但d2 loss+.002772四正，B2和ML阶段/seed反例及中等d0更低基线AUC阻止普遍取舍结论（WEAK）。固定SGD early AUC−.001430四负而LL−.000411较好，进一步限制“全面退化”。

N、p、输入/标签、正例率与baseline难度在3自然域中共变。2x/4x别名不引入新独立支持；只有3域，不能用伪回归证明N主因。domain size作为主要因果解释不可识别；其单独解释当前反例的证据WEAK。不能把小域当困难域，也不能因目标mask AUC下降就将全部q路径归于difficulty。

## 稳定现象、矛盾与竞争解释

- **STRONG**：同前向J/相同detach图期望、精确采样/权重/CPRE支持和模型身份；不是STRONG泛化机制。
- **MEDIUM**：固定SGD、校准Adam与AdaTask配置的共同早期AUC负向在diagnostic和test保留；calibrated optimizer交互；C局部AUC与整体状态映射的条件作用。
- **MEDIUM边界**：calSGD test early为正而selected为负；ORDER E50、V8 strong d2与AdaTask own-final的评价人口反例；ML early/fixed50/selected不同。反例使统一效应和统一必要机制主张不成立。
- **WEAK/不可识别**：唯一长期中介、自然困难度或N主因、普遍小/大域取舍、课程/动量有效、新方法全面收益。

竞争解释包括有限预算速度/步钟、参数路径、估计器高阶矩、输入/标签人口与representation law、coverage/repetition条件依赖、短尾位置、状态步幅/进度与多块共同作用。当前最有判别力的顺序实验是固定短尾位置及batch大小时钟后置换内容；重复/覆盖则需可行E≤N和匹配支持的另行设计；长期C须新预注册AUC-only假设并匹配容量/初始化/预算和全部单元健康。它们服务更强新问题，本轮原规定/触发最小控制均已完成。

四seed20k配对bootstrap CI不含外层LR选择不确定性；n4双侧符号随机化最小p=.125。固定100的部分n1–3有预算资格选择，不冒充四seed。历史test已有暴露，不称pristine holdout；ML seed还改变训练子集。未以覆盖±.003的关键null等效关闭核心，额外2026–2029分支未触发。

[全部14测试cohort](test_summary_verifier_manifest.json)、[全部交互](analysis/test_all_cohorts_primary_verification/test_contrast_summary.json)、[每seed对照/支持与时钟](analysis/test_all_cohorts_primary_verification/seed_test_contrasts.csv)、[原始56臂onset](archive/formal/final_baseline_onsets_and_selection_56arm_rows.csv)、[最终方法判断](V13_final_method_decision.md)。
