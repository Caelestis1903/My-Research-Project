# Experimental Open Questions


> **2026-10-26问题更新：** OQ001–OQ020继续保留为各自提出日期的问句，不以“没有触发实验”自动关闭。新增OQ021–OQ028重点从“反复做一个老解释的防御性否定”转向**辨别真实样本收益、来源可用增量、读写不同利益与局部功能必要性**；当前研究决策见[跨阶段总报告](Experimental_Branch_Synthesis_20261026.md)。最新RCF仍为设计，所有问题没有预设阳性结论。



> **更新于 2026-10-18：** OQ001–OQ010 保留 10-09 原始提问，不因新实验“未触发”而直接关闭。下表是最新去向；新增 OQ011–OQ020 仅登记真正需要新的可区分反事实/证据的研究问题。详细保留 Q1–Q53 的**历史状态与最终限度**：[跨分支问题JSON](Experimental_Question_Disposition_20261018.json)。多轮证据与反例：[综合报告](Experimental_Branch_Synthesis_20261018.md)。

| 旧问题 | 10-18 状态 | 已缩窄/仍缺的证据 |
| --- | --- | --- |
| OQ001 Sample极端自然比例的minority selected gap | OPEN | 固定mask、别名复制不等于独立新真实样本；未得极端自然比例可推广结果 |
| OQ002 domain size独立因果 | NOT_IDENTIFIABLE | N2仅排除固定θ域count波动占主导，不识别真实support N；**小域≠困难域从来不是默认待证假设** |
| OQ003 Equal/S5 history m/v具体承载 | OPEN / HISTORICAL_LINEAGE_MISSING | 新Sample q影子状态不是原Equal S5同θ谱系 |
| OQ004 Equal access/history跨backbone与target | OPEN | 后续多数是同目标q干预，非原access对照的普适复现 |
| OQ005 ML late DH的过拟合与optimizer-policy归因 | BOUNDARY / MEDIATION_OPEN | S8 late rescue不提高原best；N17阶段反例不能识别原DH中介 |
| OQ006 Domain×Task gradient结构跨设置 | PARTIAL / OPEN | 固定θ谱及block有结果，缺跨设置性能因果机制 |
| OQ007 split/label/population shift与objective | ARITHMETIC_RESOLVED / CAUSAL_OPEN | 同预测宏/加权AUC反号可分解，域内shift与目标差的因果不可分 |
| OQ008 S5-ML partial replication的调节来源 | OPEN | ML/KR同时改变任务/预算/支持/模型人口 |
| OQ009 KR S6-R2 selected validation/test反转 | UNRESOLVED / B1_NONEXACT | B1新谱系没有稳定复制，但旧精确状态缺失，不能判旧观察错误 |
| OQ010 S6-R2 swaps collapse与共享分区来源 | OPEN | 后来的局部C不是原intact DH历史参数分量因果 |


本清单只保留已有实验没有回答的问题。每条说明需要什么证据类型；不提出具体算法或下一阶段实验方案。

## OQ001 — Sample 在更极端域比例下是否存在 selected minority-domain performance gap？

- **Scientific Question:** 在已测 Sample 自然比例未见 practical NT 后，更极端 ratio 是否仍保持 selected joint-vs-single/oracle 表现，尤其是少数域单元？
- **Why Unresolved:** S1/S2/S6 覆盖的是指定自然域结构；目前未有该问题所需的 extreme-ratio selected comparison。E009 不能证明未测 ratio 结果。
- **Relevant Evidence IDs:** E009、E018、E019、E027。
- **Competing Explanations:** 可能不存在可观测 gap；可能随 ratio、域身份、目标权重、sample diversity、checkpoint 或训练 exposure 改变。
- **Evidence That Would Resolve It:** 在明确对齐的 selected checkpoint、per-domain/task AUC 与 LogLoss、同 seed joint/reference comparison 下，重复观察不同 imbalance ratios 的 gap 大小与方向。

## OQ002 — Domain size 是因果变量，还是 objective/identity/exposure/difficulty 的代理？

- **Scientific Question:** 当数据域身份、objective、训练曝光和难度结构被分离时，单独改变可用样本量会否改变 target/non-target performance？
- **Why Unresolved:** P0 轮换改变多项数据属性；S4/S5 固定自然域结构；S6 targets 是最大域；没有在同一域身份与目标下独立改变 size 的因果对照。
- **Relevant Evidence IDs:** E002–E004、E010、E014、E020。
- **Competing Explanations:** loss coefficient、sampling exposure、label prior/data shift、domain identity、gradient scale、convergence speed。
- **Evidence That Would Resolve It:** 同一数据身份与任务下只改变一个规模相关因素，同时保持或明确记录目标权重、暴露次数、数据切分和 checkpoint 规则。

## OQ003 — Equal KuaiRand d1 的 persistent-history effect 具体由哪些状态部分承载？

- **Scientific Question:** S5 识别到的整体 Adam history carrier 中，m、v、step/bias-correction 与 history-induced parameter path 各自贡献多少？
- **Why Unresolved:** S5 只操纵完整 history source；Moment gate 15/16；S6 Sample Phase B 的 KuaiRand M1 gate 未通过。
- **Relevant Evidence IDs:** E015、E016、E019、E021。
- **Competing Explanations:** signed m history、v preconditioning、effective step count、weight decay at arm-specific θ、trajectory divergence。
- **Evidence That Would Resolve It:** 经过独立 validity review 的状态分量干预证据，并保留同一配置下的 realized states、参数路径和完整指标；不能从当前 H×C contrast 直接断言状态子成分。

## OQ004 — Equal 下的 access/history effect 能否跨 backbone、目标域和数据集复现？

- **Scientific Question:** S3/S5 的窄条件效应是否会在不同 backbone、非最大 target、小域 target 或其他 domain structure 上保持方向与代价？
- **Why Unresolved:** S3–S6 主机制矩阵以 M3OE 为主；P7 是组件迁移而不是相同 mechanism test；S5ML 只部分复制 Pattern A。
- **Relevant Evidence IDs:** E011、E017、E021、E023、E027。
- **Competing Explanations:** objective × dataset × target × task interaction、imbalance severity、architecture sharing structure。
- **Evidence That Would Resolve It:** 使用同一 intervention definitions、seed pairing、checkpoint/metric protocol 的独立复制，同时对目标与非目标结果进行同等检查。

## OQ005 — MovieLens late DH gain 中普通 overfitting 与 optimizer-policy 贡献各占多少？

- **Scientific Question:** full-state continuation 中看到的 late macro rescue 有多少是 F reference 过拟合导致的“少恶化”，有多少是可泛化的 optimizer-policy effect？
- **Why Unresolved:** S8 显示 selected-anchor null、late macro rescue seed-skewed，且不超过原 F best；它没有将所有 overfit/regularization 路径分开。
- **Relevant Evidence IDs:** E020、E022、E026、E027。
- **Competing Explanations:** ordinary overtraining、implicit step-size regularization、checkpoint timing、non-target contribution、seed variance。
- **Evidence That Would Resolve It:** 在预先明确正常 selected 与 overfit anchors 的比较中，完整状态和相同 batch continuation 下，对照 late/selected best、target/non-target、AUC/LogLoss 和多个 seeds 的共同轨迹。

## OQ006 — Domain×Task gradient structures 的跨设置稳定范围

- **Scientific Question:** Domain-organized cosine pattern、harmful source-target pairs 与 U_k 条件性结果是否跨模型、数据集、域分布和训练阶段存在？
- **Why Unresolved:** S7 只探测两个数据集与一个 M3OE 配置；KuaiRand 的 negative-U counterexample 和 MovieLens overfit pattern 显示效应依赖阶段。
- **Relevant Evidence IDs:** E023–E025。
- **Competing Explanations:** Domain structure、task interaction、norm scale、Adam preconditioning、overfitting、module reachability。
- **Evidence That Would Resolve It:** 在多个独立 backbone/dataset 配置中重复相同的 Domain×Task probe、报告绝对 norm 与方向量，并对性能观测做独立验证。

## OQ007 — E4 的 train-test mixture / label shift 与 objective effect 如何区分？

- **Scientific Question:** KuaiRand 的训练与测试域占比、label prior 变化能解释多少 Equal/Sample 下的 residual differences？
- **Why Unresolved:** E4 审计测得 mixture TV 与 d1 click prior 的变化，但实验未独立操纵该 shift 与 loss objective。
- **Relevant Evidence IDs:** E004、E021。
- **Competing Explanations:** objective coefficient、sampling/exposure、temporal split distribution shift、domain/task difficulty。
- **Evidence That Would Resolve It:** 同一 objective 对照下可追踪的 split/mixture variation，且训练与测试统计同表报告。

## OQ008 — S5-ML 的 history-dominant partial replication 是数据结构还是训练阶段造成？

- **Scientific Question:** S5-ML 中 DH 的 selected/late contribution 与 T rescue 衰减，主要由温和域比例、target/task 结构还是 late overfit 引起？
- **Why Unresolved:** KuaiRand 与 MovieLens 同时改变数据集、域比例、domain identity 与收敛轨迹。
- **Relevant Evidence IDs:** E015–E017、E020。
- **Competing Explanations:** imbalance severity、target identity、task difficulty、checkpoint timing、overfit、Adam history/current interaction。
- **Evidence That Would Resolve It:** 在独立变化这些因素的受控复制中，保留 selected/common/final 与 per-cell 视图；不从两个自然数据集差异反推原因。

## OQ009 — KuaiRand S6-R2 selected-checkpoint 的 validation/test 反转及其跨数据集范围

- **Scientific Question:** 为什么同一组 KuaiRand R0-selected checkpoints 的 DH−F non-target AUC 在 validation 上为负、在 test 上为正；该差异是否能在精确的替代 selection rules 和其他 dataset 上复现？
- **Why Unresolved:** E029 显示 validation ΔN = −0.0039、test ΔN = +0.0068；canonical selected epochs 不在完整 test checkpoint grid，test-layer cross-epoch decomposition 不可得。R1/R3/R4 的正向对照来自 A-grid sensitivity；精确重选 test 仅 R4 有 1/4 seed。MovieLens 的 selected test result 近零，KuaiRand seed2025 也近零。
- **Relevant Evidence IDs:** E028–E030。
- **Competing Explanations:** validation/test metric-layer difference、checkpoint timing、selection-rule sensitivity、dataset/target difference、seed variation。
- **Evidence That Would Resolve It:** 在精确 selected checkpoints 上同时具备 paired validation/test per-cell 指标与同 epoch trajectory 视图，并在多个 seeds 和数据设置下核对方向；现有证据不足以从指标反转指定单一机制。

## OQ010 — S6-R2 parameter-swap collapse 能否定位 DH selected-test gain 的来源？

- **Scientific Question:** 在 shared/architecture/private manifest partition 下，DH 的 test-side selected gain 是否能被分配到某一参数分区，还是只能观察到跨分区不可组合？
- **Why Unresolved:** E032 显示 C/D hybrids 在四 seeds 均明显退化，不能定位 intact DH gain；seed2025 的 ΔN 分母近零，retention ratio 不可解释。现有检查只覆盖 KuaiRand、M3OE 和一种 parameter partition。
- **Relevant Evidence IDs:** E032。
- **Competing Explanations:** 跨分区 co-adaptation、partition boundary 定义、单 seed near-zero denominator；这些解释目前不能区分。
- **Evidence That Would Resolve It:** 对候选参数/功能分区提供独立完整性验证，并同时保留 intact parent 与 hybrid 的 seed-level validation/test outputs；不从当前失败 swaps 断言增益由某一分区产生。


---

## 2026-10-18 新增/进一步明确的待辨别问题

### OQ011 — Adam同θ瞬时平均应用更新差是否真实存在，且历史是否必要？

- **Current evidence:** 新 N17 Q51 的Adam原生与固定滞后分母局部对照已执行；观测差/MC误差量级不足，未能确定总体均值偏移。v2无动量SGD反例已否定“所有q差都须Adam”的通用必要性。
- **Unresolved:** 在相同θ/m/v/step的正常可达状态，是否存在非零平均更新差，以及该**局部映射**是否为长期selected性能的必要中介？与旧Equal/S5分开追踪。
- **Minimal discriminating evidence:** 预注册足够精度、相同状态、同一目标和固定梯度bank的直接更新对照；仅在局部差存在且具性能连接时进一步做正常训练干预。不要再跑一般Adam公式验证。
- **Links:** Q4、Q23、Q51；E036、E038、E041。

### OQ012 — C共享路径的独立长期必要性与健康收益是否存在？

- **Current evidence:** v3局部C AUC筛查有阳性但loss/J联合门失败；N17独立embedding/E10 full约+3e−5、4/4方向同号，实际阈值和方向门失败，N4没触发。
- **Unresolved:** 能否在容量/init/预算匹配下，独立操纵C-sharing实现稳定真实selected收益、AUC/LL同向及六cell保护；还是只有局部可达性/冗余修饰？
- **Minimal discriminating evidence:** 仅在提出新的长期唯一中介主张时，提前定义完整 C-private/shared反事实及模型健康门；不能将N4未运行等同阴性。
- **Links:** Q6、Q22、Q29、Q42、Q46、Q50；E038、E042。

### OQ013 — exposure / repetition / coverage / order 的可识别部分是什么？

- **Current evidence:** v3 WR/PERM/ORDER与N17 quota实施多项控制；固定E,N下R=E−U、E>N无法在保持配额同时无重复；ORDER同时移动短尾batch。
- **Unresolved:** 哪些高阶矩、域内支持与时间顺序可真正独立干预？不能从相互代数绑定的统计量回归推因果。
- **Minimal discriminating evidence:** 明确先检验可构造性，若不可独立操作则报告结构不可识别、不要做更多同一矩阵的防御性统计；若新干预可行，固定目标、总预算与外部选点并区分尾batch时钟。
- **Links:** Q5、Q18–Q20、Q37–Q38、Q49。

### OQ014 — 协方差各向异性/shape 是否是独立于 scalar trace 的必要性能机制？

- **Current evidence:** v3完整冻结谱、所选坐标trace与完整C反号；N17 quota将 count 主要解释从固定点中排除，但同时改依赖/高阶矩。
- **Unresolved:** shape优于trace是否仅为描述性预测，还是具有独立、可重复的训练性能作用？
- **Minimal discriminating evidence:** 先确定能否构造同均值同trace而shape不同、同θ可达的估计器对照；不能仅将更复杂谱图当作因果证据。
- **Links:** Q21、Q42、Q49、Q51；E034、E041。

### OQ015 — ML晚期 own-LR 反号是 q 效应，还是LR/进度与selection联合作用？

- **Current evidence:** N17固定subset、8优化seed下 own selected Δ−0.010875 但E100 +0.021145（八正），SN lr=.001、SB lr=.0003；非劣下界未达标准。
- **Unresolved:** 在**同一LR和相同步数**下 E100正向是否保留？是否改善绝对SN、SB及每cell AUC/LL并具可部署的保护？
- **Minimal discriminating evidence:** 下一轮先冻结ML同一人口、q×LR(.0003,.001)完整2×2、独立seed、固定外部计算预算与validation选择和指标；若固定LR对照不保留，优先LR/阶段解释；若保留才谈新机制。
- **Links:** Q17、Q41、Q48；E040。

### OQ016 — 评价权重的纯算术反号与域内样本shift的性能因果如何分开？

- **Current evidence:** N0/N17证明同一预测在macro/train-p/test-p加权cell AUC下可反号，校准后AUC排名不变而LL可以改善。
- **Unresolved:** 部署目标、域内时间/标签改变与train/validation/test人口之间的差异究竟对应哪个科学估计对象？
- **Minimal discriminating evidence:** 在问题定义时事先固定**一个主评价人口与损失指标**，同时报告其余描述性权重，不以事后挑权重救援候选；若追寻shift因果须操纵或有稳定分层外部数据。
- **Links:** Q2、Q10、Q24、Q47。

### OQ017 — KR无动量SGD q效果是否小到具有实际等效证据？

- **Current evidence:** N17共同LR SGD selected Δ+0.000837、t95[−0.004309,+0.005984]、Holm p=.6953125，横跨实用界±0.003。
- **Unresolved:** 小负、小正、零附近实用等效均未排除；不把显著性失败叫等效。
- **Minimal discriminating evidence:** 仅当此问题会改变下一阶段决策时，预注册所需区间宽度、最小有意义差和独立seed预算；不得无期限追加seed直至找到显著差异。
- **Links:** Q3、Q53；E039。

### OQ018 — 新机制的文献覆盖边界与 MDMT 独特增益能否同时成立？

- **Current evidence:** importance sampling、adaptive optimizers、AdaTask、D3/SAF、顺序及selection原理已被相关文献覆盖；D3/SAF一般改变经验目标而非仅改变q。没有新方法达到本轮数值收益与七项保护。
- **Unresolved:** 是否存在**在MDMT特定任务/共享结构下**尚未被解决、可独立于既有机制鉴别、又有实际性能价值的问题？当前不能用“在MDMT重做通用采样/动量”本身主张新颖。
- **Minimal discriminating evidence:** 先明确单篇先验假设/迁移失败边界及最小可证伪反例，再决定是否提出方法与完整文献新颖性检索；不要把没有跑完整D3/SAF解释成其无效。
- **Links:** Q12–Q14、Q25–Q27、Q52；E043。

### OQ019 — 真实独立支持样本数N、困难度、正例稀疏与训练曝光谁在起作用？

- **Current evidence:** 不存在一致“小域改善大域下降”模式；KR Adam 0/8、SGD 1/8满足严格双小域收益/大域损失；ML小域d1明显下降，但三物理域自然性质共变。
- **Unresolved:** 独立新增unique样本数、标签支持、物理域身份、任务难度与q曝光的各自贡献；N2 fixed-count/alias、V8 corruption均不是这些自然属性的独立因果检验。
- **Minimal discriminating evidence:** 若真正研究样本数因果，需在同域同任务上获得不同真实独立支持的资料或预注册严格可构造对照；否则维持“尚未识别”而非无止境反驳用户原本已否定的“小域=困难域”命题。
- **Links:** Q7–Q9、Q43；E035、E038、E040。

### OQ020 — 何时可把局部/后期收益提升为selected泛化改进而非只是慢学习/减轻过拟合？

- **Current evidence:** S8 late救援不超过原best；N17 KL/ML有早期、长期与selected反号，并且统一test曾在历史使用。某些onset只是操作性规则、无完整FWER。
- **Unresolved:** 配置/时间窗口改变的是真正有用的外部性能还是仅把选点/预算或已过拟合状态重新排列？test过去被开发使用，不足作独立方法验收。
- **Minimal discriminating evidence:** 对已明确的单一候选预注册主要selection和固定预算、基线绝对性能、AUC/LL及全部cell guard，再使用新的未开发外部/时间后置确认人口；若没有新候选则不追加无差别验证。
- **Links:** OQ005、OQ009、Q16–Q17、Q36、Q45、Q48；E027、E037、E040。

**执行/证据状态规则：** 原始历史原件找不到写 NOT_IDENTIFIABLE / INCOMPLETE；实验触发条件未满足写 NOT_TRIGGERED；方法性表现未达冻结阈值写 NO_GO（仅对该候选）；置信区间太宽写 INCONCLUSIVE；不能将这些状态统一改成“实验否定”。以上问题和Q1–Q53不以数量配额封顶。


## 2026-10-26 更新的可区分问题 OQ021–OQ028

### OQ021 — D套件的增益到底来自样本区分、总loss尺度还是复制/噪声？
- **Existing evidence:** ML D−B AUC +0.006303，Dconst01−B +0.005676，各4/4正；D权重快速靠近0.1，Dconst01并未对D通过非劣门。KR方向与区间更复杂。E050–E051。
- **Still unidentified:** 常数0.1和Adam耦合衰减/eps/步长交互、真实样本权重初值、噪声与复制、持续样本差异各自增量。
- **Discriminating test:** 优先做**同0.1损失尺度且无复制/噪声**的最小匹配控制，两个数据×四seed，不新搜超参；若这一臂能解释Dconst01相对B收益，先收紧增强主张，再决定是否需要分拆动态权重。[源报告](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。

### OQ022 — 自然ML下N19/CM19那些人工子集结论能保留到什么程度？
- **Existing evidence:** 人工187500与自然800167差异被审计；10-20部分子阶段完成，但当时N19/CM19完整transfer仍进行中，不能将旧N19/CM19结果冒称自然复现。E048–E049。
- **Still unidentified:** 人口比例、支持覆盖、有效训练曝光和真实域任务分布对条件路径/容量作用的分离因果。
- **Discriminating evidence:** 若后续要升级外部效度，先取原规划的**完整且冻结的自然scope终报**或新的独立支持窗口，再做同身份对照；不要把源目录存在视为完整实验。[原状态](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)。

### OQ023 — 强目标直接预测之后，foreign来源内容还有什么独立可读增量？
- **Existing evidence:** LPS有部分foreign读取信号；DFD ML需求读优于常量，但完整形成器输H_DIRECT/ETA_DIRECT；不同可达读取范围的对照尚缺足够支持。E056–E057。
- **Still unidentified:** 收益来自来源监督、可读隐藏特征、额外目标拟合、容量或单纯概率校准？
- **Discriminating test:** 固定目标/来源/标签预算，分别比较H_DIRECT、ETA_DIRECT、target-only、普通source logit/hidden读取、相同容量读取器，再谈重新形成。该判别被列为[RCF-v1 S1](../research_plans/2026-10-26-v2/00_README.md)，**尚未运行**。

### OQ024 — 域内部群体需求能否指导真正不同的来源后继函数？
- **Existing evidence:** DFD的非恒定责任与梯度存在，但GEOM/SHUFFLE及UNIFORM强对照削弱“形成特定需求功能”解释；KR当前高维需求选读入口失败。E057。
- **Still unidentified:** 在相同总参数/目标数据和source-retention预算下，两个来源函数是否有交叉群体优势？是否只由更强的单函数、统一锚定微调或残差校准复制？
- **Discriminating test:** 用目标群体OOF剩余风险形成函数后冻结，检测交换群体/置乱配对、单宽函数、固定来源读和统一微调的**同reader**风险差；先满足来源保持及逐格保护再宣称功能分工。[RCF设计](../research_plans/2026-10-26-v2/00_README.md)。

### OQ025 — 读取来源与反向更新来源的利益是否可能不同甚至相反？
- **Existing evidence:** S3/S4共享access、CM19路径数值活性与v3 LPS来源读取不一致；固定路径失活不能反证读取的价值。E046–E047、E056。
- **Still unidentified:** 只读的有用内容、允许目标标签重写来源函数、对来源owner造成的损害及目标短期收益之间的非对称关系。
- **Discriminating test:** 固定函数内容及目标读法，分开readonly、target-to-source受约束后继更新及source owner retention；报告所有受影响任务的LL/AUC和成本。不将来源损伤隐藏在宏平均中。

### OQ026 — selected优先的真实泛化增量与晚期减损如何分开？
- **Existing evidence:** S8/N17、10-24和v3 RHD多次出现selected与final反号；v3 RHD e50有减损但selected未胜强参照。E053–E055。
- **Still unidentified:** 动态策略改变的是最佳可用前沿，还是只是减轻某个固定过训状态的下降？
- **Discriminating test:** 预注册合法主selected与共同阶段readout，至少包括每seed/六格/AUC+LL，不用历史暴露test调整选点；有新时间/人口时再做正式确认。

### OQ027 — 新RCF的来源行为保留约束能在目标有益时真正成立吗？
- **Existing evidence:** v4没有完整来源外层refitη/μ分组导出；旧方案没有来源目标风险与source owner风险共同满足的已验收多函数实例。E057。
- **Still unidentified:** 目标剩余风险获得的增量是否只是透支来源质量、容量或额外监督？后继函数身份与强直接拟合能否区分？
- **Discriminating test:** RCF方案的源保留LL、KL和修正幅度约束须在冻结后逐项按实际预测核验，并与统一微调、固定读取、target-only保持匹配；**当前仅设计**。

### OQ028 — 如何获得真正可升级的独立人口确认，而不是重复使用历史test？
- **Existing evidence:** 10-25/26 v3/DFD的旧test、旧Q曾暴露；四seed重复主要是训练随机性，非新的用户/物品/时间支持；KuaiRand缺可信原始用户ID用于用户cluster统计。
- **Still unidentified:** 对新人口与真实场景的跨域可迁移收益、漂移与特征支持如何影响方法净效益。
- **Discriminating evidence:** 先明确尚未用于开发的新自然时间窗口、可归属用户/任务及主要统计单位；无这些材料时保留探索性标签，不为多seed或更多Q-readout强行制造独立确认。

**优先级：** OQ023–OQ025/OQ027组成一个具有独立主张与强简单对照的共享研究问题；OQ021是有限预算的样本重加权归因支线；OQ022/OQ028是能否升级外推结论的条件，**不是要求所有新研究开始前把旧问题全补完**。
