# AO04完整阶段分析：自然人口的融合自由度与局部敏感性

## 1. 完成范围与结论

B0/B1/B2×四seed共12配置，全部完成原50轮、1000main/50arch更新及40,008,350 main曝光，architecture实际曝光2,009,934。原selected规则为全validation六cell宏AUC严格最大值的最早轮次，影子停止仅记录、不提前终止。每seed三臂共用配对初始状态SHA。B0对称融合，B1每domain×task独立q共6个，B2每domain跨任务共用q共3个；未新增B3/B4或结果后调参。

12训练、12局部诊断及配置/选模冻结均已闭环。一个原始root test动作覆盖全部12模型，receipt记录optimization_steps=0，冻结SHA和所有模型/配置/selection身份一致。test保存selected、final及E10/E20/E30/E40/E50七个读数，504个逐域逐任务AUC/LL均已从NPZ独立复算，残差<1e−12，并逐数组核对row-ID、labels和共同test人口。E50与final是同训练阶段的重复命名，不能当额外独立证据。

**MEDIUM：完整自然人口没有复现清晰、稳定的B1/B2整体性能收益。** B1−B0 selected宏AUC−.0000555（二正二负），final−.0007756（一正三负）；B2−B0为−.0001392/−.0004123，各二正二负。所有12个端点宏AUC/LL点态四seed区间都跨0，不建立优越、劣效或严格等效。

**MEDIUM有限敏感性、WEAK全局识别或收益：** 36个固定开发探针点的q Jacobian均非零，因此不能简单把本次缺少完整收益归为该有限探针完全不敏感；但Jacobian非零不证明参数全局唯一可识别，也不保证增加自由度会改善测试风险。该判断与历史阴性结果并存。

本次测试复用历史已使用人口，属于探索性复现；四优化seed不是四个独立数据总体。B0 seed2023首次attempt exit−11、无有效receipt，完整同seed重跑已成功，失败attempt保留，不计科研阴性或额外seed；不能把空stdout当作已知零训练。

## 2. 宏指标、四seed及阶段

### 绝对性能

|checkpoint|arm|macro_auc|macro_logloss|
|---|---|---|---|
|final|B0|0.744192|0.611457|
|final|B1|0.743416|0.616822|
|final|B2|0.74378|0.613615|
|selected|B0|0.771979|0.500741|
|selected|B1|0.771924|0.499413|
|selected|B2|0.77184|0.501875|

### 候选−参照

|checkpoint|contrast|metric|reference_mean|candidate_mean|mean|ci95_low|ci95_high|positive|negative|
|---|---|---|---|---|---|---|---|---|---|
|final|B1-B0|macro_auc|0.744192|0.743416|-0.000775863|-0.0029987|0.00144697|1|3|
|final|B1-B0|macro_logloss|0.611457|0.616822|0.0053641|-0.00984057|0.0205688|3|1|
|final|B2-B0|macro_auc|0.744192|0.74378|-0.000412224|-0.00179142|0.000966977|2|2|
|final|B2-B0|macro_logloss|0.611457|0.613615|0.00215762|-0.00300433|0.00731958|3|1|
|final|B1-B2|macro_auc|0.74378|0.743416|-0.000363639|-0.00310173|0.00237445|1|3|
|final|B1-B2|macro_logloss|0.613615|0.616822|0.00320647|-0.0144347|0.0208477|2|2|
|selected|B1-B0|macro_auc|0.771979|0.771924|-5.54918e-05|-0.00209726|0.00198628|2|2|
|selected|B1-B0|macro_logloss|0.500741|0.499413|-0.00132787|-0.00680365|0.0041479|2|2|
|selected|B2-B0|macro_auc|0.771979|0.77184|-0.000139156|-0.000980922|0.000702611|2|2|
|selected|B2-B0|macro_logloss|0.500741|0.501875|0.00113399|-0.00681424|0.00908223|2|2|
|selected|B1-B2|macro_auc|0.77184|0.771924|8.36638e-05|-0.00173461|0.00190194|2|2|
|selected|B1-B2|macro_logloss|0.501875|0.499413|-0.00246187|-0.00938883|0.0044651|1|3|

LL负值表示改善。B1 selected相对B0平均LL−.001328但二好二差，末轮+.005364三差一好；B2 selected+.001134二差二好、final+.002158三差一好。B1−B2 selected AUC+.0000843但二正二负，LL−.002462三好一差，final AUC−.0003634一正三负且LL+.003206二好二差。排序/概率损失与阶段不能合并成一个赢家。

共同点也保留反例：E20 B1−B0 AUC−.000929四负、点态t95[−.001712,−.000146]，LL+.001531四差但区间跨0；E40 AUC+.000475三正一负、LL−.002861三好一差，两个区间跨0；末轮又负。E10 B2−B0 AUC四负但区间仍跨0。所有检查点已在结果前固定，不按最有利点选择结论；这些相关点及多重比较未经全族校正。

各臂选中模型明显强于自身末轮，但这种共同后期退化不能单独识别B1/B2相对差异的原因。开发selected B1−B0 AUC−.000334一正三负，test变为−.0000555二正二负；B1−B2由开发−.000300变成test+.0000843。保留这种跨split/指标差异，不用开发最大值替代测试。

## 3. 逐域、逐任务与基线难度

自然三个物理域训练N=[155949,318599,325619]；test N=[27687,41434,30900]。B0六cell基线如下，auc/logloss是四seed均值。

|arm|checkpoint|domain|task|auc|logloss|n|n_pos|
|---|---|---|---|---|---|---|---|
|B0|final|0|0|0.742576|0.726356|27687|14281|
|B0|final|0|1|0.754292|0.504891|27687|4964|
|B0|final|1|0|0.748203|0.69378|41434|22072|
|B0|final|1|1|0.765407|0.46296|41434|7031|
|B0|final|2|0|0.723236|0.730368|30900|18099|
|B0|final|2|1|0.731438|0.55039|30900|6324|
|B0|selected|0|0|0.765492|0.598092|27687|14281|
|B0|selected|0|1|0.781523|0.410845|27687|4964|
|B0|selected|1|0|0.774769|0.578265|41434|22072|
|B0|selected|1|1|0.793456|0.390235|41434|7031|
|B0|selected|2|0|0.751183|0.585969|30900|18099|
|B0|selected|2|1|0.765453|0.441043|30900|6324|

d2最大却在两任务selected AUC都最低；d0最小却非最低。LL的排序又不完全相同，例如task0 d0风险高于d2，不能用大小、AUC或标签熵中的单个量给域定义统一难度。

下表按域平均两任务、按任务平均三域；每行仍四seed配对。完整六cell比较另存CSV。

|contrast|checkpoint|axis|domain|task|metric|mean|ci95_low|ci95_high|positive|negative|
|---|---|---|---|---|---|---|---|---|---|---|
|B1-B0|final|domain|0||auc|-0.00176419|-0.00614955|0.00262116|2|2|
|B1-B0|final|domain|0||logloss|0.00658531|-0.0151008|0.0282714|2|2|
|B1-B0|final|domain|1||auc|-0.00019496|-0.0059784|0.00558848|2|2|
|B1-B0|final|domain|1||logloss|0.00793163|-0.00974794|0.0256112|3|1|
|B1-B0|final|domain|2||auc|-0.000368439|-0.00441903|0.00368215|1|3|
|B1-B0|final|domain|2||logloss|0.00157535|-0.013682|0.0168327|1|3|
|B1-B0|selected|domain|0||auc|-2.57482e-05|-0.00467993|0.00462844|1|3|
|B1-B0|selected|domain|0||logloss|-0.000731214|-0.00781852|0.00635609|2|2|
|B1-B0|selected|domain|1||auc|0.000448352|-0.000477002|0.00137371|3|1|
|B1-B0|selected|domain|1||logloss|-0.00208054|-0.011697|0.00753593|1|3|
|B1-B0|selected|domain|2||auc|-0.000589079|-0.00189582|0.000717657|1|3|
|B1-B0|selected|domain|2||logloss|-0.00117186|-0.00540835|0.00306462|1|3|
|B1-B2|final|domain|0||auc|0.00121123|-0.00292493|0.00534739|3|1|
|B1-B2|final|domain|0||logloss|0.000329592|-0.0245828|0.025242|2|2|
|B1-B2|final|domain|1||auc|-0.000133625|-0.00740011|0.00713286|3|1|
|B1-B2|final|domain|1||logloss|0.00606119|-0.0185209|0.0306433|2|2|
|B1-B2|final|domain|2||auc|-0.00216852|-0.00895653|0.00461948|1|3|
|B1-B2|final|domain|2||logloss|0.00322864|-0.0118742|0.0183315|3|1|
|B1-B2|selected|domain|0||auc|-8.44776e-05|-0.00264547|0.00247652|2|2|
|B1-B2|selected|domain|0||logloss|-0.00273846|-0.00901432|0.0035374|1|3|
|B1-B2|selected|domain|1||auc|0.00126244|-0.00229788|0.00482276|3|1|
|B1-B2|selected|domain|1||logloss|-0.00270097|-0.0113507|0.00594879|1|3|
|B1-B2|selected|domain|2||auc|-0.000926969|-0.00233529|0.000481354|1|3|
|B1-B2|selected|domain|2||logloss|-0.00194618|-0.00859639|0.00470403|1|3|
|B2-B0|final|domain|0||auc|-0.00297542|-0.00480489|-0.00114595|0|4|
|B2-B0|final|domain|0||logloss|0.00625572|-0.016527|0.0290384|3|1|
|B2-B0|final|domain|1||auc|-6.13346e-05|-0.00621676|0.00609409|2|2|
|B2-B0|final|domain|1||logloss|0.00187044|-0.0154714|0.0192123|2|2|
|B2-B0|final|domain|2||auc|0.00180008|-0.0041603|0.00776047|2|2|
|B2-B0|final|domain|2||logloss|-0.00165329|-0.0161326|0.012826|2|2|
|B2-B0|selected|domain|0||auc|5.87295e-05|-0.00217775|0.00229521|2|2|
|B2-B0|selected|domain|0||logloss|0.00200724|-0.0075398|0.0115543|2|2|
|B2-B0|selected|domain|1||auc|-0.000814086|-0.00370301|0.00207484|2|2|
|B2-B0|selected|domain|1||logloss|0.000620422|-0.0115169|0.0127578|1|3|
|B2-B0|selected|domain|2||auc|0.000337889|-0.00110797|0.00178375|2|2|
|B2-B0|selected|domain|2||logloss|0.000774314|-0.0020768|0.00362543|2|2|
|B1-B0|final|task||0|auc|-0.00115932|-0.0049561|0.00263746|1|3|
|B1-B0|final|task||0|logloss|0.00645699|-0.0161484|0.0290624|2|2|
|B1-B0|final|task||1|auc|-0.000392406|-0.00113754|0.000352723|1|3|
|B1-B0|final|task||1|logloss|0.00427121|-0.00715314|0.0156956|3|1|
|B1-B0|selected|task||0|auc|4.96475e-05|-0.00220013|0.00229943|2|2|
|B1-B0|selected|task||0|logloss|-0.00010752|-0.00471483|0.00449979|3|1|
|B1-B0|selected|task||1|auc|-0.000160631|-0.00201387|0.0016926|2|2|
|B1-B0|selected|task||1|logloss|-0.00254823|-0.0100523|0.00495585|2|2|
|B1-B2|final|task||0|auc|-0.00035259|-0.00630374|0.00559856|1|3|
|B1-B2|final|task||0|logloss|0.00176594|-0.0276158|0.0311477|1|3|
|B1-B2|final|task||1|auc|-0.000374688|-0.00245139|0.00170201|1|3|
|B1-B2|final|task||1|logloss|0.00464701|-0.0051123|0.0144063|3|1|
|B1-B2|selected|task||0|auc|0.000120853|-0.00205065|0.00229235|1|3|
|B1-B2|selected|task||0|logloss|-0.00172522|-0.00817176|0.00472132|1|3|
|B1-B2|selected|task||1|auc|4.64742e-05|-0.00148901|0.00158196|2|2|
|B1-B2|selected|task||1|logloss|-0.00319851|-0.0109321|0.00453507|1|3|
|B2-B0|final|task||0|auc|-0.00080673|-0.00374408|0.00213062|1|3|
|B2-B0|final|task||0|logloss|0.00469105|-0.00845835|0.0178404|2|2|
|B2-B0|final|task||1|auc|-1.77175e-05|-0.00245809|0.00242266|2|2|
|B2-B0|final|task||1|logloss|-0.000375799|-0.00534694|0.00459534|2|2|
|B2-B0|selected|task||0|auc|-7.12059e-05|-0.00076548|0.000623068|2|2|
|B2-B0|selected|task||0|logloss|0.0016177|-0.00630972|0.00954513|2|2|
|B2-B0|selected|task||1|auc|-0.000207105|-0.00122764|0.000813428|2|2|
|B2-B0|selected|task||1|logloss|0.000650286|-0.00762113|0.0089217|1|3|

selected B1−B0 d0/d1/d2 AUC−.000026/+.000448/−.000589，只有d1均值正且三正一负；final三个域均值均负。selected d1/task0 AUC+.000329、四正、点态区间[+.000022,+.000635]，是唯一端点cell AUC/LL区间不含0的读数；它未经多重比较校正，也没有形成宏收益。B2−B0 final最小d0 AUC−.002975四负、按域点态区间[−.004805,−.001146]；d2均值+.001800却只有二正二负。不能用少数正cell给方案背书，也不能因宏均值接近0忽略最小域的条件损害。

两个任务selected B1−B0 AUC约+.000050/−.000161，各二正二负；final−.001159/−.000392，各一正三负。B2 selected两任务AUC均二正二负，final task0一正三负/task1二正二负；没有稳定跨任务收益。

## 4. 大小取舍与评价权重

严格取舍要求同一个seed下d0、d1域平均AUC都上升、最大d2下降，不由宏均值判断。下表是四seed中满足该条件的次数。

|contrast|checkpoint|two_smaller_AUC_up_largest_down|smallest_AUC_down_two_larger_up|all_domain_AUC_up|all_domain_LL_improved|
|---|---|---|---|---|---|
|B1-B0|final|1|0|0|0|
|B1-B0|selected|1|1|0|2|
|B1-B2|final|3|0|0|0|
|B1-B2|selected|1|0|1|3|
|B2-B0|final|0|0|0|0|
|B2-B0|selected|0|0|1|1|

B1−B0 selected/final各1/4，B2−B0均0/4；B1−B2 selected1/4、final3/4。后者末轮宏AUC更差，且四seed平均d1差为负（其中一个负差幅度较大），不能把3/4改写为所有均值或所有seed的规律。参照更换就改变取舍状态；本轮没有单独操纵域大小，规模作为唯一解释仍WEAK。

|checkpoint|weighting|contrast|metric|mean|ci95_low|ci95_high|positive|negative|
|---|---|---|---|---|---|---|---|---|
|final|eval_p_cell|B1-B0|auc|-0.000682935|-0.00305708|0.00169121|1|3|
|final|eval_p_cell|B1-B0|logloss|0.00559528|-0.00925722|0.0204478|3|1|
|final|eval_p_cell|B2-B0|auc|-0.000292931|-0.00171887|0.00113301|1|3|
|final|eval_p_cell|B2-B0|logloss|0.00199573|-0.00380899|0.00780046|3|1|
|final|eval_p_cell|B1-B2|auc|-0.000390005|-0.003487|0.00270699|1|3|
|final|eval_p_cell|B1-B2|logloss|0.00359954|-0.0146687|0.0218678|2|2|
|final|macro|B1-B0|auc|-0.000775863|-0.0029987|0.00144697|1|3|
|final|macro|B1-B0|logloss|0.0053641|-0.00984057|0.0205688|3|1|
|final|macro|B2-B0|auc|-0.000412224|-0.00179142|0.000966977|2|2|
|final|macro|B2-B0|logloss|0.00215762|-0.00300433|0.00731958|3|1|
|final|macro|B1-B2|auc|-0.000363639|-0.00310173|0.00237445|1|3|
|final|macro|B1-B2|logloss|0.00320647|-0.0144347|0.0208477|2|2|
|final|train_p_cell|B1-B0|auc|-0.000571391|-0.00278366|0.00164087|1|3|
|final|train_p_cell|B1-B0|logloss|0.00508262|-0.00948833|0.0196536|3|1|
|final|train_p_cell|B2-B0|auc|0.000128207|-0.00118677|0.00144319|2|2|
|final|train_p_cell|B2-B0|logloss|0.00129117|-0.00265355|0.00523589|3|1|
|final|train_p_cell|B1-B2|auc|-0.000699598|-0.00343812|0.00203892|1|3|
|final|train_p_cell|B1-B2|logloss|0.00379145|-0.0134701|0.021053|1|3|
|selected|eval_p_cell|B1-B0|auc|-3.38344e-06|-0.0018059|0.00179913|2|2|
|selected|eval_p_cell|B1-B0|logloss|-0.00142631|-0.00723531|0.00438269|2|2|
|selected|eval_p_cell|B2-B0|auc|-0.000216594|-0.00105585|0.000622661|2|2|
|selected|eval_p_cell|B2-B0|logloss|0.00105185|-0.00729267|0.00939638|1|3|
|selected|eval_p_cell|B1-B2|auc|0.000213211|-0.00174113|0.00216756|2|2|
|selected|eval_p_cell|B1-B2|logloss|-0.00247816|-0.00958366|0.00462733|1|3|
|selected|macro|B1-B0|auc|-5.54918e-05|-0.00209726|0.00198628|2|2|
|selected|macro|B1-B0|logloss|-0.00132787|-0.00680365|0.0041479|2|2|
|selected|macro|B2-B0|auc|-0.000139156|-0.000980922|0.000702611|2|2|
|selected|macro|B2-B0|logloss|0.00113399|-0.00681424|0.00908223|2|2|
|selected|macro|B1-B2|auc|8.36638e-05|-0.00173461|0.00190194|2|2|
|selected|macro|B1-B2|logloss|-0.00246187|-0.00938883|0.0044651|1|3|
|selected|train_p_cell|B1-B0|auc|-6.6219e-05|-0.00160176|0.00146932|2|2|
|selected|train_p_cell|B1-B0|logloss|-0.00144779|-0.00657295|0.00367738|2|2|
|selected|train_p_cell|B2-B0|auc|-0.000175194|-0.000957284|0.000606895|2|2|
|selected|train_p_cell|B2-B0|logloss|0.000953332|-0.00669299|0.00859966|1|3|
|selected|train_p_cell|B1-B2|auc|0.000108975|-0.001576|0.00179395|2|2|
|selected|train_p_cell|B1-B2|logloss|-0.00240112|-0.00951764|0.0047154|1|3|

train-p与eval-p均为预定cell AUC/LL加权，不是pooled AUC。selected B1−B0宏/训练p/test p的AUC约−.0000555/−.0000662/−.0000034，B2−B0约−.0001392/−.0001750/−.0002167，所有相应区间跨0。权重改变小数值和效用定义，没有产生明确整体增益；不得结果后挑权重给候选排名。

## 5. 局部几何与完整性能是不同证据

每run在initial/selected/final、每域原validation前8行（24行）诊断专家差、a_D(1−b_D)、logit对q/rho Jacobian及q±.1扰动风险，零训练更新、零test读取。B0载入同theta的B1 q=0反事实，不是重新训练的B1。单类探针AUC保持未定义，不填.5。

36点q Jacobian RMS非零。B1 selected .005227–.017806、final .018269–.033461；B2 selected .008757–.022449、final .029844–.050672。a_D(1−b_D)初始.196612，selected约.194654–.196030、final .190259–.193173，专家差RMS亦非零。相同初始函数B2的raw q Jacobian RMS约为B1的sqrt(2)倍来自坐标数归一化，不是更好可识别性或性能。

raw q扰动既不是相同rho幅度也不是相同函数距离，不能以小探针扰动比较给两种参数化作全局优劣排名。当前数据只排除“这些探针点的融合自由度完全没有一阶功能影响”这一强解释；不能排除正则/有效位移、有限样本噪声、选择准则或表达冗余解释，也不证明某个q是长期必要中介。完整几何与开发阶段见[AO04_DEVELOPMENT_ANALYSIS.md](AO04_DEVELOPMENT_ANALYSIS.md)。

## 6. 历史对应、证据强度与后续问题

历史MovieLens AO04实际是人工 **d0_large**，不是其它主线常用d2_large；只有2022/23两个seed。旧B1−B0 selected test AUC+.000015592（一正一负）、LL+.00062045（两差），B1−B2 AUC−.000012657（一正一负）。旧自然KR只有一个完整B1 seed，validation AUC+.001494而test−.008385。它们均不支持普遍的融合自由度收益，原有限/阴性结论保留；不能与新四seed池化构造统计确认。旧预测缺row-ID，旧模型重加载或全局Jacobian身份未在本次冒称完成。

- **STRONG有限事实**：12训练完整预算、固定配置/选模、零更新test、504cell预测复算及归档SHA。
- **MEDIUM条件现象**：局部敏感性非零而完整收益不稳定；阶段、域任务与参照会改变效果方向；最小d0的B2末轮条件损害。
- **WEAK**：严格等效、全局参数识别、域大小主因、某个融合自由度的长期必要性或跨总体收益。没有据小均值形成新算法主张。

最有辨别力的后续候选应先用同初始函数/固定theta、等函数距离或等实际位移的参数化对照，检验是否仍出现局部敏感性与风险效用脱节；若仍无效，则反驳“只需释放该自由度即可提高性能”。只有预先指定完整候选/选择规则后才用新总体确认。当前没有理由结果后扩B3/B4、挑cell或追加seed寻找正号；这些问题不自动新增本次实验。

## 7. 证据索引与整体状态

- [完整状态及504cell计数](analysis/ao04/analysis_status.json)、[预测复算](analysis/ao04/prediction_recalculation.csv)、[训练/测试/探针来源](analysis/ao04/source_index.csv)。
- [逐seed宏/任务](analysis/ao04/seed_macro_task_metrics.csv)、[逐cell差](analysis/ao04/paired_cell_deltas.csv)、[四seed域任务/六cell](analysis/ao04/four_seed_cell_domain_task_summary.csv)、[权重](analysis/ao04/four_seed_fixed_weight_summary.csv)。
- [局部敏感性](analysis/ao04/local_sensitivity.csv)、[扰动风险](analysis/ao04/local_probe_perturbation_risks.csv)、[旧记录](analysis/ao04_history/historical_selected_summary.csv)。
- [冻结及release](analysis/ao04_freeze/)、[实际751文件SHA/113小JSON快照](analysis/family_artifacts/AO04/all/)。大工件位于服务器/home_new/xzf_natural_movielens_reassessment/AO04，672,123,849 bytes。

AO04本阶段已完成；N19/CM19及最终总报告仍待闭环，不能以此宣称整项任务完成。
