# Experimental Open Questions

## 2026-11-16 / OQ-NQR-C02 — 排序替换损失后的阶段与权重问题

1. ML E10 EMA AUC五seed改善、selected五seed下降：瞬时优化路径、阶段依赖还是总干预剂量解释该反转？需要等暴露的时序干预与完整50epoch匹配选点证据，不能从早期正值推定curriculum有效。
2. 三新臂两数据集主均值均负；去方差后ML AUC几乎不变，但LL较NQR差且4/5一致。均值分离、方差项、共同BCE系数变化和Adam/decay尺度各承担什么作用？四个机制AUC CI跨0，仍缺可分离证据。
3. KR d0/d2两较小域反向、最大d1 LL五seed改善；d0标签正先验稀少而d2较均衡。自然机会权重、先验/特征/task与规模如何区分？固定人口观察未识别规模因果，不能把小域等同困难域。
4. KR Macro AUC负而Traffic/Pooled正，早期LL改善强于排序，EMA总效应正而匹配EMA增量负。读出、校准、阶段和选点如何分别影响可迁移增量？

依据：[E-NQR-C02完整报告](../../research_reports/natural_quadratic_ranking_20261116/REPORT.md)、[配对seed及CI](../../research_reports/natural_quadratic_ranking_20261116/ALL_PAIRED_SUMMARY.csv)、[完整数值来源](../../research_reports/natural_quadratic_ranking_20261116/DATA_AVAILABILITY.md)。下一fresh Astra同时读取C1/TV/NDP及较早历史，保留已知条件反例，探索不同机制；本轮无追加救援实验。


## 2026-11-16 / OQ-AXISEVEN-C01 — 完整even-view比较后的未解问题

1. 同状态下AXIS-EVEN/IID2条件期望目标相同，端点却不同；差异由均值正则、有限梯度噪声/Adam路径还是选点与seed方差主导？KR IID2正均值仅2/5、CI跨零，现有端点未分离这些解释。需要共同状态梯度均值/方差与预算对齐路径干预证据。
2. KR两个较小域d0/d2为何反向，而最大域近零？ML最小域基线最高却退化，task0退化较一致。域规模、标签/特征结构、task和训练阶段的作用仍未独立识别；需要能排除规模单因解释的域/task证据。
3. 轴向局部E/O损失响应显著高于匹配norm isotropic，为什么没有稳定CLEAN之外的selected EMA排序增量？去odd相对NDP-A的恢复只在KR均值为正、两数据集均仅2/5正，不能指定单一损失来源。
4. Online/EMA、Macro/Traffic/Pooled与共同epoch的相反方向如何与泛化机制区分？注册主结果保持NO_CONTINUATION_SIGNAL，局部正值和LL收益仍需分别保留。

依据：[E-AXISEVEN-C01报告](../../research_reports/task_axis_even_views_20261116/REPORT.md)、[原始配对](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/tables/PAIRED_SEEDS.csv)。下一fresh Astra同时读取TV-GRM/NDP-A历史；不预设curriculum、动量有效或小域更难。

## 2026-11-16 / OQ-TVGRM — 完整首轮后的未解问题

1. 定向任务视图梯度相对等实际注入norm随机方向的AUC均值为正，但两个数据集CI都跨零；什么信息能在CLEAN+固定EMA之外提供排序增量？需要自然支持、同源五seed、匹配读出下的可否证比较。
2. KR相对NDP-A恢复损失，同时改变clean二阶矩路由、残差历史与实际幅度；三者各自贡献尚未分离。INSTANT比较跨数据集方向相反，不能预设额外动量有效。
3. 同阶段晚期正值、selected近零/负值、Traffic/Pooled正而Macro负并存；新机制应如何保留可迁移信息而不只改变过拟合阶段或格间分配？以selected主口径裁定，阶段/聚合诊断单列。

已知：ML γ=.20仅WEAK局部正信号；预设全非正NO-GO条件未触发，首轮NO_CONTINUATION_SIGNAL，无可升级champion。后续fresh Astra须同时读取[完整TV-GRM报告](../../research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md)和NDP-A全报告/原数表。人口只作协议元数据，不作为困难度代理。执行仅限当前会话、新上下文子智能体，Test封存。

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
