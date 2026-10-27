**Relation-Basis v2 完整实验报告｜2026-10-27**

已完成两自然数据集×六方法×四seed（2022–2025）的48条正式适配轨迹，8份共同前缀、960轮适配和96份selected/final历史E评估。研究决策为：**停止当前来源基底＋有向关系路线的结构扩展，保留简单目标残差适配。** 两数据集按Q预注册规则均选择TARGET_ONLY为C*；主模型均未达到推进门槛，动态分裂、来源约束网格及新正式训练未触发。完整训练与最终评估均已结束。

**总体证据强度：MEDIUM。** 当前固定协议没有建立主模型超过强简单对照的实用增量。MovieLens的测试LL均值改善与四seed一致的AUC下降并存；KuaiRand的LL与AUC四seed均变差。下面保留这些相互矛盾的结果，而不把单个指标的正向均值写成整体有效。

方法定义、数据和预算已在训练前锁定：[设计](DESIGN_SPECIFICATION.md)、[数值验收补充](DESIGN_RUNTIME_NUMERICS_ADDENDUM.md)。native为固定融合M3oE，24轮、physical batch40960；来源pilot为12 pass；适配20轮包含1轮reader warmup，batch2048，Adam lr=.001，FP32，无AMP/TF32。native/source/预处理冻结，目标风险为自然有效标签平均BCE；不重采样、不增强。V六格macro LL在候选epoch5–20选择完整状态，Q决定推进，E仅用于冻结后的历史复评。用户指定四seed覆盖原规划310x/610x；相同身份轨迹复用于最终复评，未重复计为独立训练。

**冻结与实际完成。** canonical identity为 `e7964525d8a176867abc3c36d68d8e25f903a8e5f42ef20bca6dee2c5f5167c2`。4个MovieLens worker保留R1诊断身份，4个KuaiRand使用标准FP32诊断R2；37个非诊断/非测试源码逐字节相同，共同training source为 `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`。两个兼容冻结view绑定完全相同8个真实worker、48个selected/final状态、配置和阶段决策，所有E在冻结后释放。来源快照与等价证明见 [SOURCE_REVISIONS](SOURCE_REVISIONS/REVISION_EQUIVALENCE.json)、[独立复审](RUNTIME_FIX2_REVIEW.md)，阶段决策见 [STAGE_DECISION.json](STAGE_DECISION.json)。

**开发与历史E结果。** Δ定义为LOWRANK_RELATION−预先锁定C*；LL负值为改善、AUC正值为改善。所有区间是四个seed配对差的95% t区间，df=3；域、任务和epoch不作为独立训练重复。

| 数据/角色 | Δmacro LL [95% CI] | Δmacro AUC [95% CI] | LL改善seed | AUC改善seed |
|---|---:|---:|---:|---:|
| movielens/Q | +0.00032514 [-0.00031550, +0.00096577] | -0.00020896 [-0.00048196, +0.00006403] | 1/4 | 1/4 |
| movielens/E | -0.00071533 [-0.00456772, +0.00313706] | -0.00062292 [-0.00112724, -0.00011861] | 3/4 | 0/4 |
| kuairand/Q | +0.00094887 [-0.00020777, +0.00210551] | -0.00192220 [-0.00642549, +0.00258108] | 0/4 | 1/4 |
| kuairand/E | +0.00115698 [-0.00032516, +0.00263912] | -0.00272675 [-0.00617607, +0.00072258] | 0/4 | 0/4 |

MovieLens Q通过GUARD，但PRACTICAL=false、LOCAL=false；KuaiRand Q的macro AUC平均下降0.001922，超过−0.0005保护界线，GUARD/PRACTICAL/LOCAL均false。两数据的结构扩展在E开放前已停止。E没有改变C*或选择checkpoint：MovieLens上DIRECT的平均E LL优于TARGET_ONLY，而预先锁定的C*仍为TARGET_ONLY。

历史E selected四seed均值 ± sample SD如下。六格macro与自然请求加权LL分别报告。

| 数据 | 方法 | macro AUC | macro LL | natural LL |
|---|---|---:|---:|---:|
| movielens | B_REFERENCE | 0.75971011 ± 0.00029507 | 0.50594239 ± 0.00091786 | 0.50407948 ± 0.00106522 |
| movielens | DIRECT | 0.75985988 ± 0.00032874 | 0.50524379 ± 0.00131199 | 0.50336287 ± 0.00135967 |
| movielens | READ | 0.75961190 ± 0.00037425 | 0.50537029 ± 0.00145870 | 0.50343410 ± 0.00151906 |
| movielens | TARGET_ONLY | 0.75991622 ± 0.00039597 | 0.50558507 ± 0.00187099 | 0.50366125 ± 0.00173597 |
| movielens | LOWRANK_RELATION | 0.75929330 ± 0.00046734 | 0.50486974 ± 0.00108341 | 0.50287763 ± 0.00095581 |
| movielens | FULL_RELATION | 0.75934376 ± 0.00051252 | 0.50528784 ± 0.00162623 | 0.50326898 ± 0.00153201 |
| kuairand | B_REFERENCE | 0.66256467 ± 0.00338617 | 0.43710618 ± 0.00047850 | 0.42880272 ± 0.00152427 |
| kuairand | DIRECT | 0.66275631 ± 0.00322807 | 0.43698823 ± 0.00107322 | 0.42816927 ± 0.00252739 |
| kuairand | READ | 0.66189427 ± 0.00388940 | 0.43741789 ± 0.00062878 | 0.42838954 ± 0.00223022 |
| kuairand | TARGET_ONLY | 0.66380107 ± 0.00390010 | 0.43661005 ± 0.00127458 | 0.42739242 ± 0.00258643 |
| kuairand | LOWRANK_RELATION | 0.66107432 ± 0.00399042 | 0.43776703 ± 0.00066482 | 0.42840593 ± 0.00211776 |
| kuairand | FULL_RELATION | 0.66020808 ± 0.00338089 | 0.43776168 ± 0.00065349 | 0.42932035 ± 0.00367425 |

逐seed的主要E配对差如下，所有方法及V/Q/E的原始值见 [SEED_METRICS.csv](SEED_METRICS.csv)。

| 数据 | seed | Δmacro LL | Δmacro AUC | Δnatural LL |
|---|---:|---:|---:|---:|
| movielens | 2022 | +0.00287605 | -0.00103739 | +0.00263140 |
| movielens | 2023 | -0.00228486 | -0.00026529 | -0.00252991 |
| movielens | 2024 | -0.00202517 | -0.00059917 | -0.00219853 |
| movielens | 2025 | -0.00142734 | -0.00058984 | -0.00103743 |
| kuairand | 2022 | +0.00058429 | -0.00262743 | +0.00010418 |
| kuairand | 2023 | +0.00242204 | -0.00580654 | +0.00127133 |
| kuairand | 2024 | +0.00034456 | -0.00151969 | +0.00082990 |
| kuairand | 2025 | +0.00127705 | -0.00095332 | +0.00184864 |

**重复现象与相互矛盾的证据。** MovieLens的E macro AUC四seed全部下降，配对均值−0.00062292的区间完全低于0；这是本固定协议内较强的排序损害证据（STRONG）。但E LL有3/4改善，均值−0.00071533、区间跨0；Q上则只有1/4改善。LL收益没有跨开发角色与seed稳定成立。KuaiRand在Q与E上均四seed LL变差，E AUC也四次变差（MEDIUM）；n=4的区间较宽，效应大小仍有明显seed不确定性。

**任务、macro与pooled的区分。** 下表每行是三个域的任务macro指标与整任务pooled AUC的主模型−C*差。pooled衡量跨域混合排序，不能替代域内macro。

| 数据/任务 | Δ任务macro AUC | Δpooled AUC | Δ任务macro LL |
|---|---:|---:|---:|
| movielens/0 | -0.00052195 | -0.00026744 | -0.00074447 |
| movielens/1 | -0.00072389 | +0.00049983 | -0.00068620 |
| kuairand/0 | -0.00364218 | -0.00026543 | +0.00129526 |
| kuairand/1 | -0.00181131 | -0.00021412 | +0.00101871 |

MovieLens任务1的pooled AUC改善约0.000500，而同任务域内macro AUC下降约0.000724；若只汇报pooled会改变结论。两个任务的macro AUC都下降，LL都在均值上改善，符合排序与概率风险存在权衡的现象。当前矩阵未把具体校准变化与来源知识形成单独分离。全部pooled均值、SD及区间已补齐在 [METRIC_SUMMARY_ALL.csv](METRIC_SUMMARY_ALL.csv)，任务结果见 [PER_TASK_SUMMARY.csv](PER_TASK_SUMMARY.csv)。

**域大小、标签与基线难度。** 下表使用预先锁定TARGET_ONLY的E基线。F规模是真实目标训练样本数，E正例率解释不同任务LL的基准差异。

| 数据/域/任务 | F行数 | E行数 | E正例率 | C* AUC | C* LL | 主模型ΔAUC | 主模型ΔLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| movielens/0/0 | 9357 | 27687 | 0.51580 | 0.75587712 | 0.59630424 | -0.00123359 | -0.00037508 |
| movielens/1/0 | 19116 | 41434 | 0.53270 | 0.76481708 | 0.58420604 | -0.00020710 | -0.00165198 |
| movielens/2/0 | 19537 | 30900 | 0.58573 | 0.73649700 | 0.59696195 | -0.00012517 | -0.00020634 |
| movielens/0/1 | 9357 | 27687 | 0.17929 | 0.76868967 | 0.41830443 | -0.00128045 | -0.00243063 |
| movielens/1/1 | 19116 | 41434 | 0.16969 | 0.78522791 | 0.39147919 | -0.00009869 | -0.00149486 |
| movielens/2/1 | 19537 | 30900 | 0.20466 | 0.74838852 | 0.44625458 | -0.00079254 | +0.00186690 |
| kuairand/0/0 | 10366 | 2613 | 0.08726 | 0.63024155 | 0.28853171 | -0.00777244 | +0.00098341 |
| kuairand/1/0 | 104634 | 246284 | 0.21052 | 0.68240510 | 0.50751839 | -0.00011544 | +0.00142586 |
| kuairand/2/0 | 5367 | 1870 | 0.58824 | 0.64835655 | 0.64745808 | -0.00303867 | +0.00147650 |
| kuairand/0/1 | 10366 | 2613 | 0.04592 | 0.64269287 | 0.18329199 | -0.00216022 | +0.00117083 |
| kuairand/1/1 | 104634 | 246284 | 0.11131 | 0.70839631 | 0.34800747 | -0.00009307 | +0.00059419 |
| kuairand/2/1 | 5367 | 1870 | 0.44225 | 0.67071401 | 0.64485266 | -0.00318064 | +0.00129111 |

KuaiRand最小域2的基线AUC高于规模更大的域0，两个任务均如此；小域并不自动等同于难域。域0正例率仅0.087/0.046，域2为0.588/0.442，原始LL绝对值还受标签熵影响。Q上的损害主要集中两个小域，E上小域AUC下降仍更大，但大域1的LL同样恶化。域规模/自然梯度份额是一个合理竞争解释，其作为“主要原因”的证据为WEAK：本矩阵没有把样本量、标签语义、基线难度和域分布分开操纵。

不存在跨seed/任务稳定的“小域改善、大域下降”收益交换。MovieLens E任务1上确实有域0平均LL改善、域2平均LL恶化的表象，但域1也改善、Q方向不同，而且小域0改善/大域2恶化的配对只出现2/4 seed。KuaiRand六格平均E LL全部变差、六格AUC全部下降。完整逐格区间和所有对照的配对差见 [PAIRED_ALL_CONTROLS.csv](PAIRED_ALL_CONTROLS.csv)，真实行数及正负样本见 [DATA_ROLE_PROFILE.csv](DATA_ROLE_PROFILE.csv)。

模型选择面对的输入支持也不同：MovieLens Q的user OOV为0，V为76.16%，E为19.32%，词典只在T0拟合。这为Q/V/E表现与最优训练阶段不一致提供了具体的分布解释；它没有被本次实验独立确认为唯一原因。数据见 [MOVIELENS_INPUT_SUPPORT.json](MOVIELENS_INPUT_SUPPORT.json)。自然行切分与fit-only预处理均保留原锁定协议。

**训练阶段。** KuaiRand六种方法在四seed上，epoch20的Q和E macro LL全部高于各自V选中的checkpoint；这些是同四seed的相关方法观察，不增加重复单位。LOWRANK的E final−selected平均+0.01221577，TARGET_ONLY为+0.01074762，选点后主模型已劣于简单方法，不能用晚期轨迹替代selected结果。MovieLens则是混合现象：LOWRANK的E平均final−selected为+0.00045577，四seed各有两次变好/变差；TARGET_ONLY均值为−0.00070295。相同20轮预算没有产生跨数据集一致的晚期保护。逐方法/seed阶段差见 [TRAINING_STAGE_COMPARISON.csv](TRAINING_STAGE_COMPARISON.csv)。

**形成与读取机制。** 在同一selected状态上，只替换f为根函数、保留关系/NULL/w/g的Q风险差ALL_ROOT−FULL，MovieLens为+0.00000302、+0.00008672、+0.00007650、−0.00039024；KuaiRand为−0.00001271、−0.00043298、−0.00003210、−0.00003196。MovieLens前三次有小的可观测形成贡献，第四次反向；KuaiRand四次换回根函数都更好。固定F平均关系的干预在MovieLens方向2正2负，在KuaiRand3次更好、1次更差。函数输出距离与非零U证明发生了适配，收益稳定性仍弱（WEAK）；固定状态风险依赖当前组合，路由质量不作为因果收益。详见 [FUNCTION_DIAGNOSTICS.csv](FUNCTION_DIAGNOSTICS.csv) 及完整原始诊断 [RUNNER_RAW_RESULTS.json](RUNNER_RAW_RESULTS.json)。

FULL_RELATION也未超过C*。因此“仅rank2欠拟合导致失败”不是当前最有支持的解释；增加关系自由度未挽回实用价值。简单目标适配足够、来源残差形成/条件读取引入额外优化噪声，以及自然F风险与宏V选点/输入分布的差异，是更符合当前结果的竞争解释（MEDIUM/WEAK，按上述各条证据区分）。若将来重启此问题，最有区分力的证伪实验是：保持同底座、自然数据角色、冻结特征容量和读取预算，直接比较foreign来源库与同容量目标域监督特征库；目标域库若同样有效，就能把来源知识增量与额外预训练非线性容量分开。

**参数与执行成本。** LOWRANK新增可训练62166，FULL62846，TARGET_ONLY62406；LOWRANK与TARGET_ONLY相差240。关系部分112对792，仅节省680，来源函数调用数相同。下表为四seed实际均值，包含真实并发条件；逐seed参数/显存/时延/MAC范围见 [RUN_PARAMETERS_AND_COSTS.csv](RUN_PARAMETERS_AND_COSTS.csv)。GPU0同时有他人训练，ML两个seed在GPU0，KR最终四seed在GPU1。这里记录共享环境下的实测执行时间，结构节省与执行时间分别判断。

| 数据/方法 | 部署参数 | 适配秒 | raw-X batch256 P95(ms) |
|---|---:|---:|---:|
| movielens/TARGET_ONLY | 835380 | 33.26 | 24.23 |
| movielens/READ | 873036 | 104.01 | 93.33 |
| movielens/LOWRANK_RELATION | 884484 | 133.64 | 124.64 |
| movielens/FULL_RELATION | 885164 | 110.15 | 89.32 |
| kuairand/TARGET_ONLY | 1676660 | 30.94 | 9.53 |
| kuairand/READ | 1714316 | 80.56 | 32.08 |
| kuairand/LOWRANK_RELATION | 1725764 | 94.35 | 33.17 |
| kuairand/FULL_RELATION | 1726444 | 75.41 | 20.66 |

成功矩阵共同前缀一次计1124.338743秒，适配3194.465599秒，合计4318.804342秒；功能诊断另75.450676秒。部署计时/MAC/收据文件时间戳观察间隔1173.010647秒另列，原失败/中止/重放投入下界1686.134897秒另列，均不与重叠组件重复相加。8worker总wall含READY等待15564.834477秒，READY→E释放观察等待9613.428153秒，扣除后近似active wall5951.406324秒；E释放→完成观察间隔合计193.613430秒。计时边界、未记录完整timer的启动失败及观测下界见 [COST_REPORT.md](COST_REPORT.md)、[COST_LEDGER.json](COST_LEDGER.json)。线性层MAC为明确的部分算子清单，未把STAR矩阵等遗漏项当作零；本版执行了来源函数，没有稀疏跳过。

**实际缺陷与修订。** 初始GPU运行在跨batch纯绝对1e-6 parity处退出；同native/source/READ hash重放显示同上下文h/z/phi完全相同、完整Q指标差为0，跨形状最大logit差2.384e-6。R1加入相对容差后，KR近零logit仍被过严绝对界拒绝，4worker在B_REFERENCE20轮后失去RAM。R2采用PyTorch标准FP32 atol1e-5/rtol1.3e-6并保存真正最坏归一化误差点；教师转换与RAM恢复仍为绝对1e-6。训练/模型/cache/LL/AUC/选点没有改变，独立复审PASS。失败与重放全部保留并计账，只重跑失去RAM组，ML完整轨迹保留。stdout、CPU验收与三次独立审查/复审见同目录报告和 [RUNNER_LOG.md](RUNNER_LOG.md)。

**交付与结论。** 实验完成状态为 DEVELOPMENT_COMPLETE + FROZEN_HISTORICAL_E_COMPLETE；新正式轨迹与阶段2未触发。保留TARGET_ONLY是Q规则冻结的研究决策，最终E用于报告这一决策及其矛盾证据。当前固定M3oE前缀、J2/R2、20轮适配与两自然数据集的结果，不支持继续扩容当前路线；MovieLens的LL/AUC权衡值得作为机制现象保留，KuaiRand的重复损害支持停止。所有48条配置、状态身份、逐seed/域/任务AUC与LL、selected/final、真实成本、失败历史和复现代码已交付。模型状态依用户要求仅存在worker内存，最终评估后进程自然退出；本地保留数据、预测、日志、代码和报告。复现与结果索引见 [REPRODUCE.md](REPRODUCE.md)。
