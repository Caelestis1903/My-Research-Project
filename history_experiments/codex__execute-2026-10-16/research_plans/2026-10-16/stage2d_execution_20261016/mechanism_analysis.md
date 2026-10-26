# 科研机制分析与条件流程判定

主代理，2026-10-16。依据本目录16个正式训练及 [结果报告](experiment_results.md)，并结合前轮8个零训练case、既有B1 selection法证和15篇novelty matrix。以下STRONG/MEDIUM/WEAK均限定具体命题、数据和模型，不作为整条研究路线的笼统评分。

## 稳定现象与证据强度

| 命题 | 强度 | 支持与边界 |
|---|---|---|
| 同alpha不同q可改变固定theta梯度covariance，并改变Adam平均局部更新 | STRONG，有限pool/投影范围 | 旧新oracle数值一致、两数据集四seed的C/P局部证据；不是新理论，不能直接外推训练AUC |
| 本KR IID/M3OE/Adam协议中，SB相对SN存在稳定selected性能损伤 | STRONG，本设置描述性干预 | validation Δmacro −0.010406、4/4负，test forensic亦4/4负；common epoch均值和d1条件区间支持；无新holdout、无跨backbone复制 |
| Balanced对Equal有稳定macro收益 | WEAK / 不支持 | 仅1/4 validation seed正、均值−0.006547，无seed达+.003；test仅2/4正；两个AUC guards失败 |
| checkpoint selection显著改变apparent q-effect | STRONG，当前轨迹上的代数/描述事实 | 8组timing恒等式通过；EB被选在5–8 epoch，减轻后期损伤；不能独立识别selection的因果贡献 |
| q-policy通过Adam的m或v造成上述AUC损伤 | WEAK，未识别 | 训练theta/unique exposure和全部状态共同变化；没有正常训练SGD对照、同theta m-only/v-only/current分离 |
| 域规模是主要解释变量 | WEAK | 仅3域，label prior/任务/物理域/训练阶段混杂；Equal的d0/d2同为小域却效应反向 |
| 已有可升级的新方法或通用momentum/curriculum收益 | WEAK / 不支持 | 强Sample对照未被击败，独立setting/新holdout/分量因果/创新门未满足；没有运行课程学习实验 |

q是受控干预，alpha、初始化、更新预算和选点固定，因此可把本研究视为**采样策略整体效应**的证据。q同时改变每批梯度分布、行重复、unique exposure和由此产生的轨迹；这些路径没有被单独隔离。不能把“same expected objective”误读成“所有其他训练统计量都相同”，也不能由state差的存在推出Adam是唯一承载机制。

## 反例与不一致

1. **不同objective。** Sample下Balanced的宏损伤4/4重复；Equal下validation有seed2023的小正值，test正负各半。不能将Sample的结论改写为所有alpha的统一方向。
2. **AUC与LogLoss。** selected EB−EN宏LL四seed均改善，但宏AUC三seed变差；d2的LL条件区间支持给定模型损失改善，AUC区间仍跨0。降低BCE不是提高排序AUC的充分条件。
3. **训练阶段。** EB早期被选中，final validation macro LL约1.000、d2约2.220；EN约0.506/0.834。selected负效应小于common/final负效应。不能只报告最有利的早期状态，也不能只用final夸大预注册selected损伤。
4. **selection与轨迹。** candidate timing项平均为Sample +0.011069、Equal +0.027277；但baseline选点同epoch八组宏差均负。2023 EB的selected微小正值由该锚下timing补偿跨零，不能把它当稳定trajectory收益。
5. **val/test。** SB宏方向一致；EB不同seed出现反号。d0有限正例和不同窗口不能被4个训练seed消除，row区间也未包含选择不确定性。没有证据把所有反号定位为distribution shift。
6. **分区。** C/P theta和m/v显著不同；A_arch theta相对差虽小，m/v也变化。分区维数、范数尺度和跨seed幅度不同，不支持“shared C充分”“A足以解释”或“主要是first momentum”。
7. **历史身份。** 新IID SN不是旧shuffle-S1，原S5/S6链条缺失仍在。新负效应不证明原S5效应不存在，也不补成原S6 prediction-level reversal的NO。

## 规模、难度、baseline、任务和曝光

训练d0/d1/d2=172782/1743898/89452，d1最大。Sample q干预的selected域平均差在三个域均负，没有“小域改善/大域下降”的普遍trade-off。Equal下d2点估计改善而d0/d1下降，是特定objective/selection的trade-off线索（MEDIUM描述；d2 AUC收益本身WEAK），并不覆盖两个小域。d2后期LL恶化尤其严重，进一步反对“增加小域曝光必然有益”。

SN的validation域AUC d0/d1/d2约.676/.707/.670；test约.641/.708/.661。最弱域的排序跨split改变；两任务及正例数也不同，不能用大小直接给困难度命名。M3OE只验证了这一个底座，不声称其在困难域必然失败，不外推MMoE/PLE。

Balanced每epoch可用行的平均曝光率约d0 3.87次、d1 .383次、d2 7.48次，Natural均1次；首epochunique行约81.4万 vs126.8万，最终两者均全部覆盖。于是“最终unique覆盖相同”不足以排除早期路径和重复率作用。原始sample count、unique支持、exposure、q和alpha必须分别报告，不能把重复曝光叫作新增数据。

d1 click train→validation正例率37.78%→20.97%，long_view25.61%→11.15%；validation→test则接近20.97%→21.05%、11.15%→11.13%。粗域占比validation/test都约98.2%属于d1。train→val shift和val→test差异是不同问题；这些汇总不足以证明具体reversal由shift造成。

KuaiRand的两行为标签高度相关但并非所有域严格嵌套：train存在773条long_view=1/click=0，d0无此例外；val为47条。详见 `data_joint_labels.csv`。不能把KR任务简单当独立，也不能仅依据它和MovieLens嵌套rating标签的差异作任务关系因果结论。没有为阶段6补跑未触发的任务关系因果训练。

## 竞争解释及可证伪预言

| 解释 | 当前证据 | 最有区分力的后续证据 |
|---|---|---|
| 梯度估计器几何经adaptive非线性改变优化 | STRONG局部、WEAK性能分量因果 | 正常训练中预注册q×optimizer交叉；若SGD也产生同强损伤，削弱“Adam特有”主张 |
| 重复曝光、unique支持和隐式正则化影响泛化 | MEDIUM关联、WEAK单路径因果 | 固定可用pool与曝光预算、明确改变批次组装的独立实验；不能在不同q下假称原始行频率不变 |
| checkpoint selection掩盖后期不稳定 | STRONG描述、非唯一解释 | 现有same-epoch/common证据已排除纯选点解释；未来固定selection的新setting，而非试新test选点规则 |
| C/A current与历史状态耦合 | WEAK已识别分量 | 原S5恢复并通过A0/A1后，按Stage4 C×A及Stage4B同theta shadow histories进行m/v/current分离 |
| 有限正例、校准与窗口差异 | MEDIUM描述 | 新untouched holdout及user/time-aware不确定性；不是在已反复开发test上继续挑正结果 |

当前最有区分力的新增问题是：**同目标q损伤是否依赖adaptive optimizer，而非曝光/正常非线性训练普遍产生？** 这比再证明Adam公式或追加fusion自由度更接近证伪。但原第一轮上限是16个Adam runs，当前候选收益/guard已失败；此处记录后续鉴别实验，不把它自动扩大为本轮无限追加的训练网格。原Track C的4B以已核实S5/C-A现象为研究对象；新IID state不能替代原身份来宣称完成该因果审计。

## 最终条件判断

- **2D：COMPLETE，候选实用收益NO。** 两个primary contrast都没有达到+.003、3/4正向及guard。没有改alpha、seed、lr、budget、指标或选点来救援。
- **Track A不是“效应消失”的STOP。** Sample存在稳定负效应，跨轨迹state差也存在，所以原§7联合STOP、§18“无稳定selected effect”不能被伪造为满足。准确结论是采样策略整体效应已观测，Adam-specific性能机制尚未识别；不能据此提出state-aware新方法。
- **Stage6：NOT_TRIGGERED。** 结果前冻结的Stage2D设计只在稳定实用收益且guard通过时升级边界验证；当前均未通过，且局部数值事实不是正常训练的m/v因果确认。不因阴性结果追加domain/backbone网格救援故事。
- **E1/候选方法升级：NO_GO。** 原§14/§18要求相对强Sample baseline的稳定改善及其余全部条件，当前不满足；这是停止方法开发的实际门，不是假称算法已实现并失败。CLOSED的momentum/curriculum/fusion路线没有重开。
- **原Stage3B/4/4B：INCOMPLETE/BLOCKED/NOT_RUN_DEPENDENCY。** 当前缺失原S5/S6证据；不写为科学NO，也不让这些历史缺口再阻塞已可独立执行的2D。Stage3与可恢复Stage5证据沿用前次法证范围，缺失全epoch/seed覆盖仍明示。

本轮预定可执行的16run及分析归档完成，条件性升级没有触发，历史依赖的未完成原因有明确记录。**这不等于所有条件阶段都跑过，也不等于全部机制被否定。** 若以后取得原材料，依照原依赖恢复Track C/原S6法证；若另行研究optimizer-specific q机制，应预注册鉴别实验，不能把已有test再包装成新的确认集。
