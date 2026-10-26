# Experimental Open Questions

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
