# 科研机制分析（进行中，非终局判定）

当前最明确的结论是：在保持Sample objective的条件下，改变q会改变有限batch估计器及训练轨迹；两个KuaiRand Adam配置的早期diagnostic表现存在四seed同向差异。但现有证据尚不能指认唯一必要中介，也没有资格给出GO-METHOD。V5/V7/V10/V11及冻结test仍在执行，本文件不会把未完成分支写成反证或不可识别。

## 稳定现象与证据等级

**STRONG（数值身份）**：完整经验目标、域梯度和importance加权分解一致；固定θ有限batch估计器存在明显不同的协方差，SB/SN trace在已审计16个点全部>2。确定性固定子集在去掉随机估计器后四seed路径距离为0，随机估计器路径不同。它支持有限batch估计器参与轨迹形成，但不是实际模型最终AUC机制的反事实证明。Monte Carlo均值处于估计不确定性内与代数无偏身份必须分开；plug-in Chebyshev不是已证明的有限样本FWER保证。

**MEDIUM（配置内表现）**：固定参考Adam .01在共同pre27的宏AUC差−.006980、四seed同负；校准Adam .003在pre22为−.013951，LogLoss也四seed更差。共同epoch读出排除了“只有独立选择不同epoch才产生差距”的解释；冻结的操作性过拟合分类也反驳了这两个配置中纯late-rescue解释。n=4、LR选择种子重叠、Adam下网格边界、历史验证集暴露限制外推。

**STRONG（操纵身份），性能机制仍待**：SGD前三seed共300个PERM arm-epoch保持精确物理域序列/配额，300个ORDER arm-epoch保持相同batch集合且顺序实际改变。ORDER不改变epoch覆盖或重复直方图；PERM共同改变覆盖、重复位置、时间依赖。它们是已验证的干预操作，不能因训练完成便声称性能差距被解释。

**MEDIUM（局部谱形状），WEAK（必要性）**：SGD epoch10 effective rank下降及各向异性提高四seed一致，Adam存在rank略升反例。完整坐标经验谱的秩上限是511，不能当作总体谱精确值。不同谱不自动证明它是downstream AUC的必要中介。真实无动量SGD同样存在强相邻梯度方向交替，不能把时间方向现象直接命名为Adam memory。

## 矛盾与重要反例

早期差异和固定末轮差异不一致。固定参考Adam固定50为两正两负；校准Adam在seed24固定50反向改善、selected也微正。校准SGD固定horizon点读出显示前10轮四seed微正、20轮符号混合、50轮多数负，完整共同pre分类及充分预算尚待。不能从Adam更大差距跳到“SGD不存在现象”或“所有optimizer早期均负”。

MovieLens pre16宏差为+.001391、CI跨零；最小域早期+.005270、四seed正，但固定50均值转负且seed不一致。当前既不支持均匀跨环境负效应，也不支持不分阶段的小域获益叙事。KuaiRand与MovieLens的数据、标签、域配置、步数和训练阶段共同改变，有限对照不能唯一归因于某个环境属性。

历史SGD负效应从全集validation转到新selection半集时接近零，已由完全相同batch、参数、预测的身份审计定位为评分总体改变加少量epoch重选。不能忽略这个反例，也不能将其解释为训练器漂移后随意重训。

## 竞争解释和最小区分实验

| 解释 | 现有证据 | 关键限制 | 既定区分动作 |
|---|---|---|---|
| objective错配 | 完整经验梯度/权重/单batch身份通过 | 只限当前模型实现定义的可微图 | V1已完成；保留routing detach边界 |
| selection/验证噪声 | 同epoch早期差距存在，selected符号可变 | 小cell阳性数量有限、重选不确定性与seed不确定性不同 | 完成1000行bootstrap、全部规则及冻结后test |
| 训练速度/有限预算 | 多个末轮或selected反例；seed25需要延长 | 实用平台不是全局最优，batch2B同时改变step数 | V4充分预算、等examples与另列等steps比较 |
| replacement/coverage | 精确PERM配额不变、覆盖与重复位置改变 | 固定E下R=E−U，无法独立识别两者 | V5完整四seed WR/PERM性能；保留联合解释 |
| 顺序 | ORDER的同batch multiset身份通过 | 梯度函数相同不代表移动θ上的梯度相同 | V5完整四seed ORDER性能 |
| geometry/shared blocks | q改变完整梯度分布，部分块方向/位移变化 | 关联不等于必要性，R本已private | V7同θ同state局部替换；仅selection门控触发全训练 |
| Adam状态 | Adam早期效应较大 | SGD强相邻方向交替、有效步幅和阶段均为竞争解释 | V11全局共同窗口interaction先门控；条件触发后新设计 |
| task-wise state | AdaTask原始八条已训练 | 7/8在50轮仍未平台，task更新合成还可改变步幅 | 完成selection判定和Adam匹配预算，再区分state与step scale |

## Domain size、difficulty与task/domain结构

当前数据不能识别domain size为主要解释变量。KuaiRand三域训练量分别172,782/1,743,898/89,452，同时标签率、分布、p、模型路径和baseline AUC都不同。clean八次cell审计已完成，132条性能和192条支持记录经主agent复核；两个任务共享行支持但标签不同。三域的多因素共变不能用训练seed伪造更多独立domain或做高自由度回归。

2×/4×别名复制在WR下保持完整输入序列和目标，16个映射/实际梯度/更新身份通过，故纯名义池大小的该操作不改变轨迹分布。它没有增加独立样本支持；PERM下也不保证复制不改变重复调度。最小识别设计需要在受控分布/难度下独立改变真实独特样本量，当前三个现成域不提供这种随机分配。Q7/Q9的边界是完成最小实际审计后的结论，不免除污染条件。

困难域必须以baseline性能、目标loss及标签结构描述，不能由size命名。轻/强特征遮蔽固定size、labels和采样预算，识别的是该输入扰动的敏感性；跨污染级别目标函数J_c本身随输入改变，只在每个c内部SN/SB保持同目标。V8完整四seed匹配预算仍待，不能提前声称自然difficulty是原因。

稳定的小域改善/大域下降trade-off目前证据WEAK：校准Adam早期三个域都下降；MovieLens最小域早期改善、最大域平均微负但CI跨零，阶段更换后小域方向变化。必须保留每域每任务绝对AUC/loss、有效支持及stage，而非按域大小挑选符号。

## 文献、创新与方法门控

D3和SAF的已发表动态权重机制改变目标，不能作为同目标机制解释；也不据此声称它们无法改善实际性能。把动态权重设常数会禁用关键机制，不能伪装成已完成的原方法对照。AdaTask的目标身份和官方数值oracle已核验，是否覆盖本问题仍待完整实际效果与预算匹配。

近期domain-order、optimizer-memory和importance-sampling文献已按启动日审计，见 `objective_literature_audit.md` 和source receipt。MDMT场景本身不构成新颖性；相同multiset顺序机制、状态传播或一般minibatch扰动已有理论覆盖时必须明确对照。尚无最终novelty或GO/NO-GO判定。

Q34矩范数与Q35稀疏lag标签均仅影响观测解释，不影响实际更新或性能。分别使用精确平方差恢复的m范数和轮内真实step-lag，保留原始字段和hash。任何后续因果叙事都必须服从对应反事实而非遥测曲线。终局CF1–CF6、全部动态问题、最终test与方法门控完成前，整体保持RUNNING。

Q36补充限制：onset的观察prefix峰值和多窗口扫描未包含同时误差校准，故“确认过拟合”是冻结的操作性分类。预算停止和pre-window解释均按这一层级表述；未检测到onset不证明潜在过拟合不存在。本限制不改变预设规则，固定epoch与AULC10/20提供不依赖onset切窗的保留读出。


更新：校准SGD四组初始50完整读出及正式V11门槛已完成，见 `V4_calSGD_initial50_and_V11_actual_analysis.md`。共同pre22交互−.01391006、4/4负、CI排除0并超过.003，实际触发状态实验；这是LR特定optimizer交互，尚无状态因果结论。SGD早期1–10小正、1–20不稳定、固定50/selection-selected转负的阶段反例完整保留。V5/V8 seed25因非平台且不能确认selection过拟合，整组最低150；100产物不覆盖，V4自己baseline100仍独立。所有新test仍未读取，任务继续。

更新：固定SGD.1四组完整读数已核验，pre50宏AUC平均−.00304265、4负，宏LogLoss却略改善；固定LR参考交互NOT_TRIGGERED，与校准组TRIGGERED形成配置敏感性反例。详见 `V2_fixedSGD_and_interaction_sensitivity_analysis.md`。Q3“Adam普遍必要”被反驳，仍需继续Q4已触发的状态机制条件实验，未泛化所有LR/optimizer。


校准交互的统一1–22窗口域/任务分解已实际完成并核验，见 `V11_joint_early_domain_task_interaction_analysis.md`。三个域及两个任务的平均AUC交互均4/4负；cell层保留d0t1、d2t0单seed反例。该结果仍是LR配置特定交互，状态因果拆解正在运行。


状态实验的首个seed2022初始50出现明确退化，已登记Q40，详见V11_state_seed2022_single_seed_health_observation.json。beta2zero的SN/SB selection AUC约0.5，训练J分别40.68/13.99；m-only AUC约0.56014且学习较慢。当前只有单seed描述，证据WEAK，不将gap收缩解释为状态必要性。继续预注册四seed、共同theta状态交换/范数控制；不擅自调LR/eps，操作性plateau不等于全局收敛。


AdaTask四seed初始50匹配分析已完成并主核验，见V10_initial50_vs_calAdam_analysis.md：共同pre14 q-gap−.00783937（3负1正、CI排除0），未消除早期敏感性；减弱交互+.00628296的CI跨±.003，不能宣称稳定救援或等价。epoch50 gap反而更负但两arm绝对AUC平均均优于native；AUC与logloss阶段方向不同，全部反例保留。最终匹配100 selected比较尚待native23完整readout，test仍未读。


新增判别边界：大批量前50早期q差距未消除且绝对两臂学习较慢；等步数与等样本量不能同时保持，exact1225step不确定，不能把批量结果单独归因噪声。V10同目标task-wise组件在完整匹配预算仍有早期/晚期q敏感性，但gap扩大可与两臂绝对性能提升并存，不能以gap一项宣布方法更差、充分覆盖或创新。详见两份实际分析；V5/V7/V11仍为待完成机制判别，不把相关性升级必要性。


Q41：环境自身pre窗口不同可产生表观均值符号差；共同1–16均为小正均值，且轮次仍不等优化进度。不得声称已识别dataset、size或difficulty导致符号翻转。V11方向遥测漏项已记录独立补充设计，不修改冻结训练/原shadow，不从范数 alone 推断方向机制。


新增V3综合支持selection修饰但不是全部早期轨迹差异的唯一来源；所有112000重选来源可核验。V8final200同q跨corruption所有1200实际批次SHA一致、label/support/exposure相同，只识别特征分布敏感性，不自动认定困难度。V11状态23最低150规则继续执行，坏有限性能不豁免预算，也不证明状态必要性。


## 四种子影子性能反事实完成，完整状态实验继续

四个原始影子回放及其19,836行绝对指标、71,820行对比、17,955组四种子统计均实际完成、主代理核验PASS（最大4.1e−16）。局部第10轮m/full-shadow改善两臂绝对AUC，但当前q的AUC差距仍四负；范数与阶段对loss/AUC影响不同。详见`V11_shadow_local_statistics_analysis.md`及图表，局部影响MEDIUM，长期唯一必要机制未识别。方向补充4回放已注册第41GPU队列，匹配延长22到100/23到150/25到100与24初始选择判据仍待完成。Q40已保留四种子beta2饱和和m-only学习不足，也保留v-only两臂较好的反例。Adam24四个actual100输入PASS，CPU2完整读数进行中；WR_SN尚须selectionbootstrap确认预算。新test仍未读取，未达到终局停止条件。


## 实际预算继续延长：seed25最少250、state24最少100

Seed25全部10个匹配条件actual200/exit0及60个关键文件SHA已核验。九个满足平台，mildSN改善率.005263345；完整1,000次selection确认的三个窗口均未通过，因此按冻结规则整组最少250，复制新目录进行中。为避免额外完整中间预算计算，新增只缩减bootstrap轮次的精确工程读数：所有200轮point/1200cell先从真实预测重算，实际完整CPU缓存四轮×1000与单臂子集逐位相同，100个合成确认分类与原函数一致，实际8轮缓存的三个量化窗口已独立复核。它不替代最终完整配对读数，100/150/200共12份中间前缀义务保留。State24三对actual50完整读数均PASS，v-only两臂均需延长，整6条件最少100复制中。四种子Adam抽样审计已实际完成并核验600个配对arm-epoch、3600覆盖行。新test仍未读取，V7尚待V5最终分析，不能停止整轮任务。
