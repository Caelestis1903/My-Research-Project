# Experimental Hypothesis History

本文件把历史解释与当前允许的最强陈述分开。状态截至 2026-10-09；Hypothesis 状态只使用 `ACTIVE / PARTIALLY_SUPPORTED / WEAKENED / FALSIFIED / SUPERSEDED / OPEN`。

## H001 — M3oE 独有的大域牺牲

- **Initial Hypothesis:** 大域退化是 M3oE 特有的共享结构问题。
- **Origin:** 早期 MovieLens/P5 中观察到的 L-role loss 与 comp1 gain。
- **Tests Performed:** P0-3 增加 MMoE；P0-1/2 轮换物理域与大小角色。
- **Supporting Evidence:** E003 中 M3oE 的早期结果。
- **Contradicting Evidence:** E003 中 MMoE 也出现同向 L-role 退化。
- **Evolution:** 初期 M3oE-specific → P0-3 后改为多骨干现象 → 后续 objective 研究又限制了它的通用表述。
- **Current Status:** SUPERSEDED。
- **Current Allowed Claim:** 在若干旧 MovieLens 配置中，较大的训练角色出现较低 joint-vs-reference transfer；该现象不只见于 M3oE。
- **Claims That Should No Longer Be Made:** “M3oE 固有地牺牲大域”或“Star topology 导致此现象”。

## H002 — Domain size 是退化的独立主因

- **Initial Hypothesis:** 大域/小域样本量差异直接造成 observed trade-off。
- **Origin:** L/M/S 角色结果及 early P0 系列。
- **Tests Performed:** P0-1/2 role rotation、P0-3 backbone、S2 objective contrasts、S4/S5 KuaiRand natural、S6 两个最大域 target。
- **Supporting Evidence:** E003、E010 有所测 Large-role 关联。
- **Contradicting Evidence:** E003/E010；S6 的 MovieLens d0 实为中域、d1 才是最小域；不同域/目标/阶段方向不一致，且无独立 size 操纵。
- **Evolution:** role 相关性 → 怀疑 size → 识别 objective、域身份、weight、收敛等混杂，独立因果未识别。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** Domain size 可以是调节变量或代理变量，但现有矩阵不能认定为主要解释。
- **Claims That Should No Longer Be Made:** “小域就是困难域”“大域必然受损”“size 已被证明是根因”。

## H003 — 小域梯度 SNR 必然更低，因此需要按域动量

- **Initial Hypothesis:** 小域样本少、噪声大，需更长 EMA 记忆。
- **Origin:** P1 的早期设计动机。
- **Tests Performed:** P1 D0.4 两个 batch size；P6 SNR-derived γ；S0 匹配平滑对照。
- **Supporting Evidence:** 某些梯度范数在少量设置下不同，但未形成稳定 SNR 规律。
- **Contradicting Evidence:** E001；P6 的 SNR-derived γ 未胜过公式控制，E008 未显示 per-domain momentum 独特收益。
- **Evolution:** 直觉性前提 → 实测不稳定 → 不再作为方法前提。
- **Current Status:** FALSIFIED（限已测 MovieLens 统计口径）。
- **Current Allowed Claim:** P1 的两种 batch 条件没有显示稳定的小域低 SNR。
- **Claims That Should No Longer Be Made:** “小域更噪已被证实”或“p_d 可直接作为梯度噪声补偿标尺”。

## H004 — comp1 的收益来自独有的 domain-conditioned momentum

- **Initial Hypothesis:** comp1 的收益不能由普通平滑或 Adam 记忆解释。
- **Origin:** P5/P6 中 comp1 大幅优于 baseline 的结果。
- **Tests Performed:** P6 uniform γ、手工/SNR-derived γ；S0 validation-selected uniform EMA 与记忆控制。
- **Supporting Evidence:** E005 的早期 Sample+comp1 residual 与 P6 旧结果。
- **Contradicting Evidence:** E008：uniform EMA 在 S0 的旋转均值中逐 seed 优于 comp1。
- **Evolution:** 强机制主张 → 找到 residual → 匹配控制否定独特性，保留“旧 setting 下可复现 residual”而非机制因果。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** comp1 在某些旧目标/配置中有正向差异；它没有显示相对可选 uniform-smoothing family 的独立优势。
- **Claims That Should No Longer Be Made:** “SNR/域条件动量机制已被确立”或“uniform EMA 全面无效”。

## H005 — comp1 收益随 imbalance ratio 单调扩大

- **Initial Hypothesis:** ratio 越极端，comp1 越有效。
- **Origin:** P5 高 ratio 结果。
- **Tests Performed:** P5 ratio sweep；P0-2 固定 N ratio sweep。
- **Supporting Evidence:** 若干高 ratio 设置差值较大。
- **Contradicting Evidence:** E002 中 P5 曲线非单调，r30/r100 又改变总 N；P0-2 只支持 ratio 部分贡献。
- **Evolution:** 对高 ratio 的描述性观察 → 固定 N 后仍有限关联 → 单调规律不成立。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** 在某些固定 N 条件下，ratio 与方法差异相关。
- **Claims That Should No Longer Be Made:** “comp1 效果随 ratio 单调增长”或“高 ratio 结果证明 size 机制”。

## H006 — Equal/Sample objective mismatch 影响域级性能

- **Initial Hypothesis:** 等权六项目标在样本占比悬殊时可能改变 large-cell/target 表现；Sample 可能缓解。
- **Origin:** P0-4 的权重与选点对照。
- **Tests Performed:** P0-4/5、E1/E3/E4、S2–S6 objective contrasts。
- **Supporting Evidence:** E004、E010、E021。
- **Contradicting Evidence:** E4 的 train-test mixture/label shift、MovieLens checkpoint reversals、S6/S8 的 dataset/stage dependence；权重与 exposure 的单独贡献仍未完全拆分。
- **Evolution:** 大域牺牲解释 → objective-sensitive observation → 受数据、目标、阶段限制的解释。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** Equal 与 Sample 是不同有效目标；在所测 Equal cells 可见局部退化，Sample 在指定 selected 条件下减弱它。
- **Claims That Should No Longer Be Made:** “Sample universally optimal”“Equal universally wrong”或“系数变化是唯一中介”。

## H007 — Sample objective 下仍普遍存在 practical residual negative transfer

- **Initial Hypothesis:** 即使 Sample 修正目标，shared MTL 仍普遍损害某些域，足以支撑 Safe-MTL 主线。
- **Origin:** 早期单元级负差值。
- **Tests Performed:** S1 多数据集、多骨干、joint-vs-SD；S2 Equal/Sample；S6 selected gates。
- **Supporting Evidence:** 零星小负单元；E009 未达到 practical 阈值。
- **Contradicting Evidence:** E009 的 0/72 practical NT；E018 的两个 Sample target gates 均 STOP。
- **Evolution:** 个别负 cell → 配对扩展未重复稳定实用损害 → 该普遍主张停止。
- **Current Status:** FALSIFIED（S1/S6 已测协议内）。
- **Current Allowed Claim:** 已测 Sample 自然比例与 reference 条件未显示普遍 practical NT。
- **Claims That Should No Longer Be Made:** “Sample 下已有稳定普遍负迁移问题”。

## H008 — Curriculum / 时间 schedule 稳定优于静态 Sample

- **Initial Hypothesis:** 按域难度排序的训练课程可进一步改善 Sample。
- **Origin:** P1/P6 组件设计。
- **Tests Performed:** P1-2 forward/reverse、P6 component/sensitivity arms、E5 schedules。
- **Supporting Evidence:** 少数描述性小正值；未达到稳定门槛。
- **Contradicting Evidence:** E006；P1-2 差异在噪声级，E5 未通过 GO。
- **Evolution:** 被当成潜在核心组件 → 对照不稳定 → 从主线移除。
- **Current Status:** FALSIFIED（已测 curriculum/schedule）。
- **Current Allowed Claim:** 只可报告这些具体 schedule 的阴性结果。
- **Claims That Should No Longer Be Made:** “难度次序已验证”或“curriculum 已胜过 static Sample”。

## H009 — Adaptive calibration/EMA 能稳定胜过 Sample 且安全

- **Initial Hypothesis:** 在线估计贡献并使用 EMA 可减少 objective mismatch，同时保持所有域安全。
- **Origin:** E6 adaptive method design。
- **Tests Performed:** E6 Adaptive-NoEMA/Adaptive-EMA 对 Sample/Equal；E7 preregistered GO gate。
- **Supporting Evidence:** E6 某些 secondary metrics 为正，Adaptive-EMA 与 Equal 的差异大。
- **Contradicting Evidence:** E006：对 Sample 的 overall gain 不过 margin；Adaptive-EMA 的 Medium-domain safety gate 失败；E7 0 runs。
- **Evolution:** 方法候选 → performance/safety gate 未通过 → 禁止当成可冻结方法推广。
- **Current Status:** FALSIFIED（限 E6 测试门槛）。
- **Current Allowed Claim:** E6 在该协议下 STOP。
- **Claims That Should No Longer Be Made:** “adaptive calibration 已成为优于 Sample 的主方法”。

## H010 — Equal/KuaiRand/tab1 的 shared-access 导致局部 target loss

- **Initial Hypothesis:** FULL 跨域 shared updates 是 Equal 下 tab1 target 表现差的重要条件因素。
- **Origin:** S2 的 Equal target observation。
- **Tests Performed:** S3 objective×access intervention；S4 controls；S5 history/current decomposition。
- **Supporting Evidence:** E011；S3 的 target rescue 与 interaction 4/4 同向。
- **Contradicting Evidence:** Sample arm 的差异近零；MovieLens/late checkpoint 与 non-target tradeoffs 限制外推。
- **Evolution:** 观察性 Equal loss → 局部 access intervention 支持因果影响 → 机制进一步定位到 history。
- **Current Status:** ACTIVE。
- **Current Allowed Claim:** 在 M3OE/KuaiRand natural/Equal/tab1/指定 selected 对照下，shared-access 干预会因果改变 target 结果。
- **Claims That Should No Longer Be Made:** “跨域梯度几何冲突已被证明是根因”或“此效应适用于所有域/模型”。

## H011 — 纯 gradient geometry/conflict 是 S3 rescue 的机制

- **Initial Hypothesis:** target-only rescue 来自跨域梯度方向冲突。
- **Origin:** S3 access contrast 的机制解释。
- **Tests Performed:** S4 norm controls、S5 history/current split、S7-A probes、S7-B same-θ continuation、S8 effective update analysis。
- **Supporting Evidence:** E011 access effect；E023/E024 显示晚期 gradient utility 结构。
- **Contradicting Evidence:** E013 幅度控制不解释；E015 history 承载 Equal d1 rescue；E025 gradient 变化未改善短程 validation；E026/E027 无 selected best gain。
- **Evolution:** access effect 被误同义为 geometry mechanism → 分拆 current/history 与 effective update 后，纯几何因果解释被削弱。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** gradient structure 与 late MovieLens 现象相关，但不是已确立的独立性能根因。
- **Claims That Should No Longer Be Made:** “gradient conflict 已被证明”“负 U_k 足以导致性能损害”。

## H012 — 仅匹配 raw gradient norm 或 applied-step norm 足以解释 rescue

- **Initial Hypothesis:** T 的收益只是梯度或更新幅度变小。
- **Origin:** S3/S4 的机制候选。
- **Tests Performed:** S4 N/U arms。
- **Supporting Evidence:** T 对 F 有大差异。
- **Contradicting Evidence:** E013 两类幅度控制均未复制 rescue。
- **Evolution:** 幅度候选 → 匹配控制失败 → 作为充分解释被排除。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** 这两类 norm matching 不足以复现指定 rescue。
- **Claims That Should No Longer Be Made:** “幅度匹配解释了 S4 target rescue”。

## H013 — Equal KuaiRand d1 rescue 跟随 persistent Adam history

- **Initial Hypothesis:** S4 中的差异可能依赖当前梯度方向、history 或两者。
- **Origin:** S4 norm controls 失败后的机制分解问题。
- **Tests Performed:** S5 F/T/HT/DH，四种 source policy。
- **Supporting Evidence:** E015：HT≈T，DH≈F；target rescue 约 96–99% 由 history carrier 承载。
- **Contradicting Evidence:** 15/16 moment gate，MovieLens S5ML 的 DH 非零且晚期改变。
- **Evolution:** 联合组成/历史候选 → KuaiRand Equal d1 下定位为 history-dominant → 限定为完整 optimizer carrier 与该配置。
- **Current Status:** ACTIVE。
- **Current Allowed Claim:** 在 KuaiRand natural/M3OE/Equal/d1/tab1 的 S5 条件下，persistent history source 跟随 target rescue。
- **Claims That Should No Longer Be Made:** “m 单独是原因”“v 单独是原因”“history mechanism 跨 objective 不变”。

## H014 — current-target source 对 rescue 的贡献在数据集间恒为零或恒为主导

- **Initial Hypothesis:** S5 KuaiRand 中 DH≈F，可能说明 current-target contribution 很小；也可能该模式普遍。
- **Origin:** S5 history×current factorial。
- **Tests Performed:** S5ML Equal MovieLens；S6 Sample 2×2；S7/S8 trajectory controls。
- **Supporting Evidence:** E015 的 KuaiRand Equal DH≈F；E020 的 MovieLens Sample late DH−F。
- **Contradicting Evidence:** E017 MovieLens Equal selected/final 中 DH 有非零成分；E019 selected Sample contrasts 小。
- **Evolution:** KuaiRand 单一结果 → S5ML 显示数据/阶段不同 → S6 又显示 objective 与训练阶段交互。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** current-source effect depends on dataset/objective/checkpoint; cannot assign a universal zero or universal positive effect。
- **Claims That Should No Longer Be Made:** “DH 一定等于 F”或“target current 一定带来 rescue”。

## H015 — Equal 下 history rescue 在 Sample 下仍达到 practical selected magnitude

- **Initial Hypothesis:** S5/S5ML history-dominant rescue 可直接推广到 Sample。
- **Origin:** Equal objective 下两数据集的 F/T results。
- **Tests Performed:** S6 Phase A F/T；Phase B Sample F/T/HT/DH。
- **Supporting Evidence:** MovieLens common-late DH effects (E020)；但不满足 selected practical target gate。
- **Contradicting Evidence:** E018/E019 两数据集 selected target contrasts 均低于 +0.003；S8 selected continuation null。
- **Evolution:** 跨 objective 外推 → Phase A 双 STOP → Phase B 显示晚期 MovieLens residual 仍存在，但不是 selected target rescue。
- **Current Status:** FALSIFIED（针对已测 Sample selected practical rescue claim）。
- **Current Allowed Claim:** Sample 已测最大域 target selected gates 没有 practical rescue；这不是证明所有设置效应严格为零。
- **Claims That Should No Longer Be Made:** “Equal 机制已在 Sample 下同幅复现”或“Sample 完全没有任何 optimizer-policy trajectory effect”。

## H016 — target-only / HT 是安全可部署方法

- **Initial Hypothesis:** 只更新 target shared gradients 可保留 target gain 并避免 non-target loss。
- **Origin:** S3 target rescue。
- **Tests Performed:** S3–S6 记录 target/non-target cells、macro/worst、AUC/LogLoss。
- **Supporting Evidence:** 部分 Equal target cells 有大幅提升；S5 HT 对 KuaiRand d0 selected cost 有平均缓解。
- **Contradicting Evidence:** E014/E016 non-target penalties与 stage reversals；S5ML d0 cost/d1 gain pattern；E019–E021 Sample under threshold；没有部署型比较。
- **Evolution:** 诊断 intervention 被误写成 candidate method → trade-off、objective dependence 明确 → 仅保留诊断身份。
- **Current Status:** FALSIFIED（作为“已验证安全方法”的主张）。
- **Current Allowed Claim:** target-only/HT 是用于识别更新来源的诊断臂。
- **Claims That Should No Longer Be Made:** “target-only/HT 已解决负迁移”或“没有 non-target cost”。

## H017 — late DH/history effect 能提高最终会选择的模型性能

- **Initial Hypothesis:** S6 late-stage DH gain 可能代表 optimizer-history 方法改善整体性能。
- **Origin:** MovieLens common/final gain 与 S6 trajectory reanalysis。
- **Tests Performed:** S7-A/B、S8 full-state continuation，以及 S6-R2 对 KuaiRand / MovieLens 保存 checkpoint 的 selected-solution 评估。
- **Supporting Evidence:** E020/E022 的固定 late checkpoint observations；E028 显示 KuaiRand R0-selected checkpoints 的 test non-target 与 macro 优势。
- **Contradicting Evidence:** E019 的 Sample selected-target gates 未过；E026/E027 的 S8 selected-best 对照；E029 的 validation/test 方向反转；E030 的 MovieLens selected-checkpoint null。
- **Evolution:** MovieLens late fixed-epoch gain → S8 未提高 MovieLens 原始 selected best → S6-R2 在 KuaiRand 发现 test-side selected gain，但 validation 同期不占优且 MovieLens 未复现。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** KuaiRand Sample/M3OE 的 R0-selected DH 相对 F 有 test non-target mean +0.0068（4/4）和 macro mean +0.0044；target mean 为 −0.0004，validation non-target mean 为 −0.0039。该结果限于特定 dataset、objective、selection rule 和 test layer；S8 的 MovieLens null 仍成立。
- **Claims That Should No Longer Be Made:** “late DH 在所有数据集都提高正常选点性能”或“late DH 在任何 setting 都没有 selected test-side gain”；test-side observation 也不证明 validation optimum 或 optimizer mechanism。

## H018 — negative validation-gradient utility 是独立且充分的性能退化原因

- **Initial Hypothesis:** late negative U_k/harmful contribution 造成 MovieLens 性能下降，可作为梯度方法干预靶点。
- **Origin:** S7-A 的 late MovieLens alignment。
- **Tests Performed:** KuaiRand negative-U counterexample、S7-B same-θ current intervention、S8 effective-update continuation。
- **Supporting Evidence:** E024 的 MovieLens late 时序关联；E025 的同 θ 梯度层响应。
- **Contradicting Evidence:** KuaiRand negative U_k 但 AUC 继续升；S7-B utility 改善后 validation 未改善；S8 effective alignment 与 rescue seed 不对应。
- **Evolution:** 相关性 → same-θ 改变 gradient geometry → 未见 validation benefit，过拟合/Adam 轨迹为更强解释。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** U_k 是条件性局部一阶统计，在 MovieLens late overfit regime 与下降同步。
- **Claims That Should No Longer Be Made:** “negative U_k 必然造成性能下降”或“修复 U_k 就会改善 validation”。

## H019 — Sample 在极端 ratio 下存在 selected minority-domain under-optimization

- **Initial Hypothesis:** 当前自然比例下无 practical residual NT，但更极端不均衡可能出现 minority-domain gap。
- **Origin:** S8 之后的研究问题陈述；不是已观测事实。
- **Tests Performed:** S1/S2 的已测自然设置；未见本问题的 extreme-ratio selected test。
- **Supporting Evidence:** 目前无直接 evidence。
- **Contradicting Evidence:** E009 的自然设置 Sample joint-vs-SD 无 practical NT。
- **Evolution:** 作为 open question 保留；不得把小域/困难域等同，也不得当作已有问题。
- **Current Status:** OPEN。
- **Current Allowed Claim:** 现有资料没有建立该 gap；问题在未测 ratio/target 设置中仍开放。
- **Claims That Should No Longer Be Made:** “Sample 已知存在 minority under-optimization”或“它必然存在”。

## H020 — Domain×Task gradient structure 可普遍跨模型/数据集推广

- **Initial Hypothesis:** S7-A 的结构可能是普遍 MDMT 梯度规律。
- **Origin:** 两数据集 Domain×Task probe。
- **Tests Performed:** S7-A 两数据集、S7-B same-θ 对照；未有相同 probe 的跨 backbone 扩展。
- **Supporting Evidence:** E023 的两数据集 domain-organized cosine observation。
- **Contradicting Evidence:** E024 的 utility/performance 差异；KuaiRand 对照与 MovieLens 不同；P7 不是相同机制 probe。
- **Evolution:** 局部结构被观察到 → 可推广性未检验，性能机制解释被后续结果收窄。
- **Current Status:** OPEN。
- **Current Allowed Claim:** 所测模型与数据中可见 Domain×Task 结构；跨骨干、数据、域任务不确定。
- **Claims That Should No Longer Be Made:** “已证明普遍梯度机制”或“P7 组件迁移等于机制复制”。

## H021 — KuaiRand DH selected-test gain 只是选择规则或单点尖峰造成

- **Initial Hypothesis:** S6-R2 的 KuaiRand R0 test non-target gain 可能只依赖 canonical selection rule，或由单个窄 validation peak 偶然产生。
- **Origin:** R0-selected test 对照中观察到 DH−F non-target AUC 提升。
- **Tests Performed:** S6-R2 Part A 的 R0–R4 validation selection 与 A-grid sensitivity；Part C 的 selected-window 分类。
- **Supporting Evidence:** E028/E031 中存在边界：R1/R3/R4 的 exact reselected test 指标不完整，A-grid 仅限六个保存点；seed 2025 对 alternative-rule gain 有反向响应；DH seed 2023 是 narrow spike。
- **Contradicting Evidence:** E028 的 R0 test ΔN 为 +0.0068、4/4 正；R1/R3/R4 A-grid ΔN 均值为正且各 3/4 正；E031 的 DH windows 3/4 为 moderate，而 F 为 3/4 narrow spike。
- **Evolution:** 单一 R0 观察引出 selection/peak artifact 解释 → 多规则 sensitivity 与 window 分类削弱“完全由单点/单规则造成” → exact alternative-rule test 缺失与 seed caveats 保留不确定性。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** 在已存 checkpoint 上，测试结果不只出现在 canonical R0；但 R1/R3/R4 的支持来自 A-grid sensitivity，不能称为 exact-rule replication，也不能推广到其他 dataset。
- **Claims That Should No Longer Be Made:** “R0 的 test gain 已证明对所有 selection rule、seed 或 dataset 稳健”或“已彻底排除 checkpoint timing”。

## H022 — KuaiRand DH selected-test gain 可由单一参数分区定位

- **Initial Hypothesis:** 把 DH 与 F 的 selected parameters 按 shared/architecture/private 分区交换，可以定位 DH test-side gain 由哪一侧承载。
- **Origin:** S6-R2 Part D 的 parameter-swap analysis。
- **Tests Performed:** KuaiRand 四 seeds 的 manifest-based cross-partition hybrid inference；验证 state-dict key/shape parity 和锚点重现。
- **Supporting Evidence:** E028 确立了 KuaiRand 完整模型的 test-side DH−F 差异，这是提出分区定位假设的动机；该证据本身没有识别参数分区。
- **Contradicting Evidence:** E032 中 C/D cross-partition hybrids 的 test macro 在四 seeds 均降至约 0.52–0.58；E_hybrid≈C_hybrid，F_hybrid≈F_full，未能保留或定位 DH gain。
- **Evolution:** 预期由参数交换识别承载分区 → 两个方向的 cross-partition swap 均大幅退化 → 结果与分区间 co-adaptation 一致，无法分配完整 DH gain 的来源。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** 该 manifest partition 下的已训练组件不能直接互换；swap 结果不支持“某一侧单独承载增益”，也不能证明 co-adaptation 是 test gain 的原因。
- **Claims That Should No Longer Be Made:** “DH shared/architecture 或 private 参数分区已被独立识别为 selected-test gain 的来源”。

## H023 — Checkpoint timing alone explains the KuaiRand DH selected-test gap

- **Initial Hypothesis:** DH 与 F 的 selected test 差异完全来自所选 epoch 不同，而非同一训练阶段的 policy difference。
- **Origin:** S6-R2 Part B decomposed selected validation differences into policy and timing terms。
- **Tests Performed:** S6-R2 val-layer anchor decomposition and same-epoch test trajectory on the saved checkpoint grid。
- **Supporting Evidence:** E029 shows large validation timing terms (TimingAlongF = −0.0120 for non-target AUC; 4/4 negative) and anchor-dependent policy terms。
- **Contradicting Evidence:** E029 shows same-epoch test non-target Δ(e) positive at 16/20 saved seed×epoch points, so timing alone is not established as the entire test-side explanation。
- **Evolution:** canonical selected epochs 不同使 timing 成为候选解释 → validation decomposition 显示对 timing 敏感 → 实际 selected epochs 的 test decomposition 不可得，test-side attribution 仍开放。
- **Current Status:** OPEN。
- **Current Allowed Claim:** Checkpoint timing contributes to validation-layer differences; whether it explains most of the KuaiRand test-side selected gain is unresolved。
- **Claims That Should No Longer Be Made:** “Timing fully explains the test gain” or “timing is irrelevant to the selected contrast”。

## 假设状态计数

| 状态 | 数量 |
|---|---:|
| ACTIVE | 2 |
| PARTIALLY_SUPPORTED | 3 |
| WEAKENED | 5 |
| FALSIFIED | 9 |
| SUPERSEDED | 1 |
| OPEN | 3 |
| **合计** | **23** |
