# 01 — 既有证据、已完成重分析与研究决策

## 1. 当前结论

**历史训练信息已经是有效研究依据，下一步应研究其使用方式如何改善正常MDMT训练。** 固定参数EMA在两数据上的AUC收益、共享/Embedding主要贡献、超年龄匹配效应和历史互补均有累积证据。A21在KuaiRand的历史Test人口上提供同PAIR控制之外的五seed增量。新方案由这些事实出发，比较两个具体可执行的历史使用机制。

当前最值得研究的缺口是：**原始辅助梯度被保存之后，共同Adam二阶状态及其随时间的变化，是否影响这些信息在共享参数上的有效表达。** C1改变来源间二阶耦合；C2改变预条件与历史累积的顺序。这个问题与已完成的时间尺度、条件数量、残差历史和动态选源实验有明确区别。

证据基点为 [main@358d3a4](https://github.com/Caelestis1903/My-Research-Project/tree/358d3a465747eda6af1d55acc937759fa651a10b)。主入口为 [01索引](../../mdmt_research_context/01_AI_EXPERIMENT_INDEX.md)、[02设计](../../mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md)、[03数表](../../mdmt_research_context/03_EXPERIMENT_RESULTS.md)。来源与内容哈希见 [evidence/source_manifest.json](evidence/source_manifest.json)。

## 2. 相关证据的收敛台账

| 实验 | 已做干预及匹配对照 | 当前接受的结论 | 本轮研究决策 |
| --- | --- | --- | --- |
| EXP001–019、EXP020/S0 | 域条件梯度历史、uniform梯度EMA、长β₁普通Adam、角色和目标权重对照 | comp1−Adam约+.005307，但comp1−验证选uniform EMA约−.003033、4/4负；域条件衰减独特收益未保住 | 不重新扫描按域规模分配动量；S0的梯度EMA与现代非反馈参数EMA分开 |
| EXP021–024 | Sample/Equal、单域参照、共享访问、raw norm与实际step norm匹配 | 正常Sample未显示普遍的实用剩余迁移损失；KR Equal共享访问损害有条件成立，两个简单范数控制未解释该旧救援 | 从当前强自然训练观察问题，不恢复退化Equal底座 |
| EXP025–028 | 同目标下history/current来源交叉，KR/ML与Equal/Sample | KR Equal d1救援跟随持久history；HT恢复约99.2%/95.9%，DH约2.1%/1.9%；跨设置边界明确 | 接受完整state载体局部作用；本轮分开一阶与二阶组织，不复刻HT/DH |
| EXP029–032 | 同θ短续训、完整状态续训、选点重审、参数交换 | 局部梯度utility和晚期减损未转成更好的正常selected模型；旧KR Test阳性不伴随Validation阳性 | 局部诊断不再承担完整方法否决权 |
| EXP033–042 | 固定θ replay、Adam/无动量SGD、q×LR、WR/PERM/ORDER、quota/count | 同目标不同采样可改变轨迹，Adam历史不是通用必要条件；影响随预算/LR/阶段变化；随机域计数不是所测协方差主要解释 | 关闭“证明Adam普遍必要”和重复采样审计；保留正常Adam中的具体新干预 |
| EXP055–060 | Gaussian、缩放、L2、IID/反向配对、SAD/AIG、条件/教师/梯度视图 | 固定EMA有效；降方差/更复杂门未给稳定额外排序收益；尺度与正则化是强替代解释 | C1/C2配实际更新尺度控制；不以方差下降代替AUC |
| EXP061–064 | 动态EMA、PUB同α控制、参数组/时间窗口/函数组合、反馈、两态、KD | 共享/Embedding与近期/较早互补已定位；预测组合只覆盖部分收益；退化PUB修复、直接反馈、已测KD不足以超过强参照 | 接受EMA位置与互补事实；不重做分组/窗口/普通teacher蒸馏 |
| EXP065–069 | 条件PAIR、ECCM、C/H/M历史、五seed补全、发表模型重选、Gaussian覆盖 | 监督、历史和EMA需分层；ML Gaussian D0+EMA对M3oE +.004167且六格保护通过，LL上升；KR未胜HyperGate | 固定PAIR对照与固定EMA保留；Gaussian为后续独立组合因素 |
| EXP070–071 | 方向响应、分组时间尺度、漂移增强、年龄匹配、收益预测 | 固定EMA稳定且超年龄匹配；更稳定/更可预测不能替代更高AUC；动态动作无稳定额外增益 | 不重复方向门/ρ调节/收益预测筛选 |
| EXP072–073 | 异质性投影增强、同支持零扰动、自然权重NDP-A | 投影抑制非目标漂移，但selected AUC失败；零扰动复现主要退化，监督权重是关键组成；自然权重修复后仍无稳定AUC增量 | 本轮保持自然BCE和同PAIR支持、质量，不增加未匹配重加权 |
| EXP074–075 | TV-GRM、shuffle/instant、axis-even/IID2/isotropic | 已测raw任务视图残差历史+clean v+cap未给稳定selected增量；对弱NDP-A的修复不等于超过clean | 不复刻残差历史；C2明确改为PAIR来源的历时预条件方向 |
| EXP076–079 | NQR/Brier/mean-gap、早晚剂量、完成权重/总质量/格质量 | 早期正向不能替代selected；KR完成权重LL −.003942、d0t0 AUC −.003141；单独格质量无稳定排序增量 | 阶段曲线用于解释，单格损害进入决策；不重复NQR时序和格权重 |
| EXP080 | A20多条件、A21单辅助历史、A22长mixed Adam、λ=.025、S_DENSE定位；B真/伪源与脉冲 | A21的KR Test同PAIR增量成立；ML组件增量非正；B动态选源无持续价值；Gaussian身份不符为技术暂停 | 保留A21线索；改二阶来源组织/预条件历史顺序；不重启旧B策略 |
| EXP081 SCALE | 常数损失缩放及冷热state匹配 | Adam中`cL,wd,eps`与`L,wd/c,eps/c`在对应state下等价；SCALE自身对照上Macro改善可伴Traffic/Pooled下降 | 直接采用尺度恒等式，优先做本候选的方向/尺度匹配，不重新证明缩放现象 |

每一行的具体设计与原始链接可由主上下文对应EXP章节追溯。早期四seed、人工人口和旧选点数字保持原身份，不并入现代五seed性能均值。S0的uniform EMA作用于共享梯度，现代固定EMA作用于参数读出，二者不得同名混写。

### 2.1 A21确切做了什么

[原设计](../../research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md)与[实现](../../experiments/conditional_information_momentum_20261102_v3/optimizers.py)一致：自然BCE+coupled WD的一阶态按β=.9更新；单份PAIR历史按活动事件β=.99更新；两者分别偏差校正后相加；分母仍来自当前完整mixed梯度的β₂=.999二阶矩。S_DENSE之外普通Adam处理同一完整PAIR目标。S_DENSE不含Embedding；辅助梯度本身仍沿完整模型计算图。

A20与A21的区别包括条件重组、w/q历史和时钟；A22同时延长自然与辅助的一阶历史。两者均是已完成对照，不能再用“需要先试多份历史/更长Adam”延缓新候选。

### 2.2 核心历史数值

下表来自固定S5原数表，完整六格与SD见 [confirmed_reference_metrics.csv](evidence/confirmed_reference_metrics.csv)。A00为clean+EMA，A10为PAIR Adam+EMA，A21为单辅助历史+EMA；发表模型采用其原冻结online读出。

| 数据集 | A00 Test Macro AUC | A10 Test Macro AUC | A21 Test Macro AUC | 最强发表模型 Test Macro AUC |
| --- | ---: | ---: | ---: | ---: |
| MovieLens | .779038514 | .778556686 | .778474048 | M3oE .775190 |
| KuaiRand | .676522998 | .676703430 | .680765767 | HyperGate .682113 |

A21−A10的KR Test差为 **+.004062337**，五seed全正，95%配对t区间约[+.000469,+.007656]；同一比较的Validation为 **−.001988511**。A21−A10的ML Test差为−.000082638。KR A21相对clean+EMA的最大域d1两个任务平均下降；相对HyperGate最差格d0t0约−.014493。因此，接受“KR历史Test人口上的辅助历史增量”，本轮研究继续解决可选点性、跨格保护和强模型竞争力。

SCALE独立四seed2022–2025的Dconst01_NO_AUG−B在ML/KR历史Test Macro AUC约+.004049/+.007588；KR Traffic/Pooled AUC约−.004516/−.004450。其人口与B对照不同，不把这些数字拼入A21表。

## 3. 本次已完成的历史端点重分析

本次从 `artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv` 的固定Git blob `d420cdf0043f88b66a6bb886e0c73d72175023a8` 解析6,480条原端点记录，按seed配对计算online与fixed EMA。产物包含1,050条逐seed差和210条汇总（2数据×3选点口径×7格/宏指标×5种比较）；均为Validation历史重分析，新增训练0条。

设`ΔO=AUC(A21,online)−AUC(A10,online)`，`ΔF=AUC(A21,EMA)−AUC(A10,EMA)`，交互`I=ΔF−ΔO`。

| 数据集/口径 | ΔO | ΔF | I | 解释 |
| --- | ---: | ---: | ---: | --- |
| ML，各自selected | +.000411788（5/5） | +.000195919（4/5） | −.000215869 | selected部署策略的描述性重叠，I的95%区间跨零 |
| KR，各自selected | −.000748854（1/5） | −.001988511（1/5） | −.001239657 | I五seed均负，但区间[−.002675,+.000196]，没有必要把它写成已识别统一机制 |
| KR，共同E10 | +.004548371（5/5） | +.002022029（2/5） | −.002526342 | A21有早期online积极信号；EMA早期滞后 |
| KR，共同E50 | −.007118963（0/5） | −.005104004（0/5） | +.002014959 | A21晚期online退化，EMA缓和其中一部分 |

A10、A21各自的selected EMA−online在两数据均5/5正：ML约+.002961/+.002745，KR约+.004107/+.002867。**EMA在A21训练轨迹上仍有独立读出价值；A21与EMA的交互随阶段变化。** 这已经足以保留两条机制的研究价值，并把新问题收敛到历史如何在线使用。selected的四个数采用各自Validation最佳epoch，解释为部署流程交互；共同E10/E50才是同训练时刻交互。二者分开报告。

## 4. 本轮不重复的工作

直接引用固定EMA有效、年龄匹配、主要参数组、近期/较早窗口与函数组合结果；不新增同目的轨迹。A20、A22、PAIR低剂量、TV-GRM随机/instant、动态源选择、NQR早晚暴露、旧q×LR/域计数研究均已提供相应边界。

新诊断中的“同状态换分母”和“当时D与当前D的累积顺序”此前没有被隔离。近期源代码审计显示SAD、ECCM、TV-GRM先累计raw梯度再用当前D；H_REAL/H_MIX在当前D下作投影后仍返回raw空间存储。C1/C2改变了实际运算关系，因此有完整训练价值。

## 5. MDMT问题导向与本轮成功形态

条件PAIR由真实域、任务和联合标签构成，S_DENSE同时接收不同来源的信号。本轮需要在正常训练里观察这些信号经过历史与预条件后的跨格净排序收益及损害，并验证新处理是否改善该分配。六格同时测量真实纠正、破坏、共同stage和最终selected；域规模单独记录，禁止代替困难度。

如果候选只给出与通用LaProp/两来源AdaTask适配相同的效果，优先保留简单版本，并把结论定位到普适优化机制在MDMT的应用。如果选择性处理PAIR历史还带来稳定额外收益及六格协调改善，则具备MDMT问题导向的方法机会。两种情况都可推动研究，研究价值由问题、机制与匹配比较共同决定。
