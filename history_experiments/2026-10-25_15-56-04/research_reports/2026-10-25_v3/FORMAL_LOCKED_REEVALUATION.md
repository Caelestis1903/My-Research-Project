# 锁定的新 seed 复评

## 进入依据与固定范围

在52核心、10旧aux/外部、7条件轨迹全部24epoch完成后，主agent独立分析并于 `2026-10-25T10:12:43.229970+00:00` 冻结本包。RHD仅KuaiRand通过两开发seed对R0–R5的restricted门槛；必要PB_HISTORY也已完整结束，R6在3101 selected/24的LL均较低。此单seed归因不当作确认，用后续完整比较检验是否可重复。MovieLens未晋级。LPS global主比较失败FEATURE、KR restricted失败L2一个seed，明确NO_START；P2变体不替换L4。

不可变spec SHA256：`697c9f3250986d7687a26a21e1f4bc57b4a8077c62e1ff5208537d91d96c934b`。

方法固定为R6_PB_HB、R0_BASE(BEST_SIMPLE)、R5_PB_H1、R7_PB_HISTORY、fresh STGU_DOMAIN_ADAM；seeds4101–4104，共20条逻辑轨迹，全部固定50epoch，加4条物理共享native10warmup。已审查320份v2/v3运行身份元数据，未找到这些seed的先前研究执行；这只是在可访问审查范围内的新训练seed，不是新样本或全世界未使用的seed。

## 选点与评价角色

主要终点是 `benchmark_test_exposed` 上validation macroLL-selected的macro LL；epoch≥12且精确平局取最早epoch。`Q_legacy_confirm_exposed` 为companion，`Q_explore` 仅开发诊断。每条另报告共同epoch50，保留AUC/LL-best验证别名次表和逐任务/域/六格、naturalLL、task-pooledAUC、最差cell。没有patience或因阴性提前结束。

旧test和Q_confirm均有历史暴露，因此本包叫“锁定方法后的新seed复评”，不能叫未见确认集验证。所有算法、阈值、方法集合、seed、角色和预算在任何正式进程前冻结；正式测试只在最终方法冻结后评分，不据其选择方法。统计单位为seed，四对配对差、样本SD和df3 Student t95% CI分别列出；不把开发两seed与正式四seed混为n6，也不把域/任务/epoch当独立重复。

## 父状态、优化器与方法生命周期

所有方法保留自然fit数据、fit-only编码、batch40960（末批保留）、FP32/无AMP/TF32、固定fusion M3oE及相同耦合weight_decay Adam。RHD共享每新seed的native10完整状态；HISTORY再共享同seed HB初始化与首主块，之后独立闭环。未来父状态只在文件实际生成后原子记录SHA/full-state/data/seed/kind/boundary证书，冻结配置中不编造未来hash，续跑不重初始化Adam或RNG。

PB独立部署到50每条245H shadow分支：HB在KR为4900shadow逻辑步，H1为245步；HISTORY物理suffix237H，另8H初始化首块物理共享、独立部署仍补全245H。实际记录逐步次数，不把同一物理前缀重复计入研究实际。BEST_SIMPLE按两个开发seed validation-selected LL平均锁定；BASE/STATIC精确平局后BASE的冻结独立部署成本估计较低，故锁BASE，不用Q或test挑弱对照。

STGU从各新seed epoch0新建模型、Adam、控制器和主行序列；导入经核实的原sequential-group step算法，拒绝旧3101桥接状态及24epoch上限。实际optimizer group更新数另计，不能等同native逻辑batch数。它直接按LL选择，与旧AUC-primary原表分开。

## 预算与执行证据

任何正式进程（含warmup）之前，初始完整组件预算已冻结为26591.016908秒（7.3864小时）保守串行预留。开发真实组件率按模型/域迁移、STGU显式3倍保守容量余量，统一1.35调度余量；它不是实际耗时或GPU-hours，也不是硬超时。真实retained工作在epoch11保留完整状态后续至50，修订预算单独留痕，不修改方法锁定。

具体命令、进程起止、逐run状态、成本组件与父证书保存在本地manifest、campaign执行报告及每run目录。推理统一4096同输入、5warm+30计时，仅已选检查点；重复推理不构成研究重复。

20条正式计时片段均完整保留到epoch11，4条预热均到10。按实际率修订剩余预留为26804.153988秒，比初始预留显式增加213.137080秒（约0.80%）；原冻结预算和101份SHA匹配的日志前缀全部保存。不可变方法spec不改写，预算修订另有收据，继续全部固定50epoch。

## 完整正式结果

全部20条固定50epoch完成，主进程exit0。正式bulk真实UTC为2026-10-25 10:23:58.733350至12:01:34.890994，5856.157509秒。此前保留的epoch11计时片段及4条warmup另有记录，不能把bulk时间当成整个正式包的全部成本。

全部89条逻辑轨迹终点核验后，最终方法冻结于2026-10-25T12:02:26.282158Z，SHA256为`8a0698db50ed4d129909b9decdd400e1ee8db89cfa142591f942282f9405e382`。之后两个历史暴露角色全部178/178评分、388个predictor-checkpoint-role记录完成；主报告的数值从逐行预测独立重算，角色行身份、标签、域、checkpoint epoch/SHA与验证选点逐一核验。

### 主要终点：历史test，validation LL-selected

差为HB减参照，LL负有利；四差按4101–4104排序。n=4，样本SD与df3配对t95%区间如下，不能把方向数或区间解释为新数据确认。

| reference | paired_deltas | mean_delta | sample_sd_delta | favorable_direction_count | paired_t95_low | paired_t95_high |
| --- | --- | --- | --- | --- | --- | --- |
| R0_BASE | [-0.003073657, +0.005376639, +0.001959519, -0.001299502] | +0.000740750 | +0.003727787 | 2 | -0.005190992 | +0.006672491 |
| R5_PB_H1 | [+0.000193624, +0.004108768, -0.000276876, -0.002372669] | +0.000413212 | +0.002704484 | 2 | -0.003890227 | +0.004716650 |
| R7_PB_HISTORY | [-0.002386860, +0.001255934, +0.000545001, -0.001650622] | -0.000559137 | +0.001736425 | 2 | -0.003322176 | +0.002203903 |
| STGU_DOMAIN_ADAM | [-0.000560314, +0.005938680, +0.003460244, +0.001290517] | +0.002532282 | +0.002803022 | 1 | -0.001927951 | +0.006992515 |

HB相对BASE、H1、fresh STGU的平均LL均更高；相对HISTORY均值略低但仅2/4 seed有利。所有主LL区间包含0，当前证据不足以支持稳定优势，也不足以证明所有比较中都存在真实劣势。开发KR两seed的selected优势没有在新seed主评价稳定重复。主selected有效性证据WEAK，本版不推广PB_HB为优先方法。

test selected AUC的HB−BASE/H1/HISTORY/STGU均值分别−.00208573/−.00233581/−.00175266/−.00087210；对HISTORY四seed均较低。不能仅以LL对HISTORY微小均值改善宣布排序收益。全AUC/LL配对值、median、naturalLL和task-pooledAUC见[正式比较摘要](APPENDIX_FORMAL_COMPARISON_SUMMARY.md)及exposed附录。

### 预定共同epoch50

| reference | paired_deltas | mean_delta | sample_sd_delta | favorable_direction_count | paired_t95_low | paired_t95_high |
| --- | --- | --- | --- | --- | --- | --- |
| R0_BASE | [-0.009324243, +0.005483131, -0.028131072, -0.003044926] | -0.008754277 | +0.014272182 | 3 | -0.031464504 | +0.013955949 |
| R5_PB_H1 | [-0.002075774, -0.010974230, -0.030771780, -0.011952984] | -0.013943692 | +0.012066657 | 4 | -0.033144436 | +0.005257052 |
| R7_PB_HISTORY | [-0.013180435, -0.033070233, -0.064626091, -0.032255966] | -0.035783181 | +0.021311960 | 4 | -0.069695265 | -0.001871097 |
| STGU_DOMAIN_ADAM | [-0.027522836, -0.001750079, -0.024255752, -0.017654419] | -0.017795771 | +0.011457623 | 4 | -0.036027406 | +0.000435864 |

终点HB相对H1、HISTORY、STGU的LL均为4/4更低，对BASE为3/4；对HISTORY差均值−.03578318，t区间[−.06969527,−.00187110]。该重复晚期现象为MEDIUM，值得保留，不改成“机制完全无效”。但test五方法全部20条轨迹的epoch50 LL都高于各自合法selected；终点优势描述的是相对较小的后期恶化，不能替代主要selected目标，也不能把相对减损称为比早期模型更好的绝对风险。

### Companion与开发诊断的矛盾

legacy-confirm selected的HB−BASE/H1/HISTORY LL分别+.00179551/+.00328706/+.00099053，均4/4较坏；对STGU均值−.00117373、2/4较好。Q_explore selected对应BASE/H1/HISTORY均值+.00189249/+.00305883/+.00107814，各仅1/4较好，对STGU近似平局。test中对HISTORY的微小LL均值改善未跨角色重复，不能挑test一列主张当前验收稳定有效。

相对HISTORY的终点LL三角色均4/4改善：test−.03578318、legacy−.02925051、Q_explore−.02714531；相对H1与STGU也均4/4改善。三角色共享原数据来源，不能当成12个独立seed，但晚期方向重复确实存在。legacy的H1-selected AUC四seed均较HB高，终点却四seed均较HB低，是训练阶段与排序目标的直接反例。

### 域、任务与规模

test selected相对BASE，域0/1/2 LL均值分别−.00121162/−.00099322/+.00442709；域0四seed LL均好，但AUC四seed均坏（平均−.00353828）。两个任务的macro LL均值分别+.00057091/+.00091059，排序均值也均下降。域0的排序/校准取舍及最小域2风险上升，不能被总均值抹去。

test selected相对HISTORY，域1 LL平均−.00758411，域2却+.00541252；epoch50域0/1/2分别−.02445990/−.00429573/−.07859391，域0/2为4/4改善，两任务也4/4改善。晚期差主要集中在域2，仍需保留域1部分cell受损；全部六格及方向数见图与完整表。

BASE selected在test的域级AUC约.645822/.698509/.655480，LL约.238389/.421331/.653100。最小域2不是最低AUC域，域0的最低AUC与最低LL并存，标签基率明显影响尺度。自然fit数量138225/1395118/71561不变；test各域行数2613/246284/1870，macro仍等权。域规模、测试支持、基率、特征与难度混杂，规模作为主要因果解释证据WEAK。

test selected相对BASE的小域AUC改善/大域AUC下降只有1/4 seed，legacy为0/4；test终点AUC为2/4。selected LL取舍两角色均0/4，不支持稳定普遍的“小域获益换大域下降”。不能将late域2改善反推为规模因果。

![预定正式配对](figures/formal_primary_paired_forest.png)

![六格与seed](figures/formal_primary_cell_logloss.png)

### 机制与最小可证伪后续

HB四seed实际接受34/35/25/28个事件（各80），H1为45/39/34/37。HISTORY共享首块后实际记录79个suffix事件，逻辑总块数仍80；其suffix非STATIC为79/77/79/69，HB全程非STATIC为34/35/25/28。HISTORY日志中的P_cal acceptance只更新未来记录，不能冒充当前动作被验收。当前验收把更多块回退BASE，与较小晚期恶化相容，但尚不能区分“当前响应信息有价值”与“保守地减少控制即可”。HB对H1还多付shadow训练，不能称等算力的跨度独立效应。

最有判别力的下一项是另立新计划，在预先封存的自然时间窗口中，以当前验收HB对比预先固定阶段非STATIC预算的保守HISTORY，并保留BASE；阶段预算只能由当前开发资料锁定，不借用新run未来响应。预先同时写明selected主LL与共同晚期交互，完整新seed运行。如果保守HISTORY匹配晚期收益，当前标签验收的独有必要性被进一步削弱；若仍有独立增量再评估成本。此为后续建议，本版不追加。
