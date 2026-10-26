# Experimental Evidence Ledger

状态截至 2026-10-09。Evidence 只记录直接观察与审计事实；Interpretation、status 和可接受外推单独列出。AUC 差均注明方向；除非写明，跨 seed 的均值使用报告口径，不能把 cell/epoch 当作独立 seed。

## E001 — MovieLens 梯度 SNR 不支持“小域必然更噪”

- **Observation:** P1 的共享参数诊断中，batch=40960 时 d0/d2 SNR 为 0.5085/0.5584；batch=4096 时为 0.0524/0.0511，大小差异消失并反向。梯度 cosine 为弱正。
- **Source Experiments:** EXP001。
- **Source Reports:** [`P1_diagnosis.md`](reports/history_report/P1_diagnosis.md)；[`P6_ablation.md`](reports/history_report/P6_ablation.md)（SNR 反推 γ 的后续 ablation）。
- **Dataset / Objective / Model:** MovieLens-1M；旧 as-published objective；M3oE。
- **Seeds:** 5（2022–2026）；SNR 汇总含 epoch×seed 配对观测。
- **Effect Direction / Magnitude:** Mixed / batch dependent；SNR 差约 −0.050（40960）與 +0.0013（4096），不支持稳定的小域劣势。
- **Robustness / Scope:** Weak；单数据集、单共享参数定义、两种 batch size。
- **Current Status:** TENTATIVE。
- **Current Interpretation:** 该测量条件下，小域较低 SNR 不是稳定现象；P6 的 SNR-derived momentum 未改善也不说明所有数据域都没有噪声差异。
- **Alternative Explanations:** batch size、样本组成、共享参数集合、梯度方差估计方式。
- **Evidence Strength:** WEAK。

## E002 — 历史 comp1 增益随 ratio 并非单调，ratio 与总 N 曾混杂

- **Observation:** MovieLens 早期 comp1 相对 baseline 的 AUC 差随 downsample ratio 依次约为 +0.0064、+0.0034、−0.0012、+0.0053、+0.0166、+0.0202；后两档同时改变总训练样本量。P0-2 固定 N 设计仍观察到 ratio 对部分增益有贡献。
- **Source Experiments:** EXP003、EXP006。
- **Source Reports:** [`ML_P5_REPORT.md`](reports/history_report/ML_P5_REPORT.md)、[`P0_2_STAGE_REVIEW.md`](reports/history_report/P0_2_STAGE_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens-1M；历史训练目标；M3oE。
- **Seeds:** P5 各 ratio 为 3 或 5；P0-2 为 4。
- **Effect Direction / Magnitude:** Mixed；P5 不是单调曲线；P0-2 ratio=50 的 L-role Δtransfer −0.015545 ± 0.010894。
- **Robustness / Scope:** Medium；旧实验与固定 N 复核方向不完全相同，且 role/ratio/样本结构不可视为单一变量。
- **Current Status:** SUPERSEDED（针对“ratio 单调解释”）。
- **Current Interpretation:** ratio 与 early comp1 表现有关，但不足以支持单调 ratio law 或 domain-size 单因果。
- **Alternative Explanations:** 总 N、域身份、有效 loss 权重、样本多样性、checkpoint、初始化方差。
- **Evidence Strength:** MEDIUM。

## E003 — 大域角色上的旧退化不局限于 M3oE，但不识别 size 单因果

- **Observation:** 固定 N 角色轮换中，P0-1 的 L-role Δtransfer 为 −0.014557 ± 0.007439；P0-2 ratio=50 的 L-role 为 −0.015545 ± 0.010894。P0-3 的 MMoE 也出现 L-role 负 transfer（约 −0.0116），与 M3oE 同向。
- **Source Experiments:** EXP005–EXP007。
- **Source Reports:** [`P0_1_STAGE_REVIEW.md`](reports/history_report/P0_1_STAGE_REVIEW.md)、[`P0_2_STAGE_REVIEW.md`](reports/history_report/P0_2_STAGE_REVIEW.md)、[`P0_3_STAGE_REVIEW.md`](reports/history_report/P0_3_STAGE_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens；旧 objective；M3oE 与 MMoE。
- **Seeds:** 4 per matrix; role/cell 配对数见各报告。
- **Effect Direction / Magnitude:** L-role 负向；约 −0.01 AUC 量级。
- **Robustness / Scope:** Medium；跨两个骨干和多个 role 条件，但仍受 objective、有效权重、域身份、训练收敛共同影响。
- **Current Status:** SUPPORTED（限所测 fixed-N 配置）。
- **Current Interpretation:** 早期退化不是 M3oE 独有；角色轮换比某一个物理域的解释更可信，但域大小未被独立操纵。
- **Alternative Explanations:** Objective 权重、域身份与难度、样本多样性、gradient scale、训练收敛。
- **Evidence Strength:** MEDIUM。

## E004 — Equal 与 Sample 是不同目标；Sample 在已测选点减轻 Large-cell 损伤

- **Observation:** P0-4 中，Equal-train / Equal-select 的 L-role Δ 为 −0.0155，Sample-train / Equal-select 为 −0.0012；P0-5 与 S2 的 Sample 条件也未达到预设 practical negative-transfer 门槛。S2 Equal−SD 有 8/24 组达到 −0.003 或更差，严格组合 gate 为 3/24；Sample−SD 为 0/24。
- **Source Experiments:** EXP008–EXP009、EXP022。
- **Source Reports:** [`P0_4_STAGE_REVIEW.md`](reports/history_report/P0_4_STAGE_REVIEW.md)、[`P0_5_STAGE_REVIEW.md`](reports/history_report/P0_5_STAGE_REVIEW.md)、[`stage_S2_analysis.md`](reports/stage_S2_analysis.md)。
- **Dataset / Objective / Model:** 主要为 MovieLens；S2 另含 KuaiRand；M3oE。
- **Seeds:** 4 per later stage; P0 summary uses paired role/seed blocks。
- **Effect Direction / Magnitude:** Sample 相对 Equal 缩小 L-role 损伤约 0.0143 AUC；S2 KuaiRand tab1 click Equal−SD −0.045006 ± 0.000602，4/4 同向、5/5 common epochs 为负。
- **Robustness / Scope:** Strong within tested objective, dataset, target, and checkpoint definitions；selected/common/final 并非同一 estimand。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 目标加权会改变观察到的域级结果；Sample 减轻了所测 Large-cell selected 损伤，但不证明 Sample 对所有域/任务普遍最优或机制已识别。
- **Alternative Explanations:** Effective objective coefficient、采样 exposure、dataset split/label shift、checkpoint selection、task/domain interaction。
- **Evidence Strength:** STRONG。

## E005 — comp1 的早期 residual 存在，但不证明 domain-conditioned momentum 独特

- **Observation:** P0-5 Sample+comp1 的 residual AUC 为 +0.005407 ± 0.000620（12 paired blocks）。S0 中 comp1 相对 sample Adam 有正差，但相对 validation-selected uniform EMA 的 macro AUC 差为 −0.003033 ± 0.000815，4-seed 均值支持统一 EMA。
- **Source Experiments:** EXP009、EXP018、EXP020。
- **Source Reports:** [`P0_5_STAGE_REVIEW.md`](reports/history_report/P0_5_STAGE_REVIEW.md)、[`P6_ablation.md`](reports/history_report/P6_ablation.md)、[`stage_S0_analysis.md`](reports/stage_S0_analysis.md)。
- **Dataset / Objective / Model:** MovieLens；P0/P6 历史目标与 S0 Sample；M3oE。
- **Seeds:** P0-5 12 paired blocks；P6 5；S0 4 seeds × 3 rotations。
- **Effect Direction / Magnitude:** 历史 residual 为正；S0 comp1−uniform-EMA macro AUC −0.003033 ± 0.000815。
- **Robustness / Scope:** Medium；不同阶段协议不能把数值直接合并；S0 是机制特异性更强的匹配对照。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 旧协议下 residual 确实出现；可接受的当前结论是它没有显示出相对可选 uniform-smoothing family 的独立优势，不是“EMA 在任何配置都无效”。
- **Alternative Explanations:** 训练 exposure、objective、验证集选点、平滑策略族、任务/域聚合。
- **Evidence Strength:** MEDIUM。

## E006 — 已测 curriculum / schedule / adaptive calibration 未稳定胜过 Sample

- **Observation:** P1-2 forward curriculum−none overall AUC +0.000839 ± 0.001018（p=0.198）；E5 Forward 对 static Sample 的 fixed-final 正向 seed 为 1/4，未过预设 GO gate；E6 Adaptive-NoEMA 未过 margin，Adaptive-EMA 未过 remaining-domain safety gate；E7 因 E6 STOP 而 0 runs。
- **Source Experiments:** EXP010、EXP015–EXP017、EXP018（P6 component arms）。
- **Source Reports:** [`P1_2_STAGE_REVIEW.md`](reports/history_report/P1_2_STAGE_REVIEW.md)、[`stage_E5_analysis.md`](reports/history_report/stage_E5_analysis.md)、[`stage_E5_go_no_go.md`](reports/history_report/stage_E5_go_no_go.md)、[`stage_E6_analysis.md`](reports/history_report/stage_E6_analysis.md)、[`stage_E6_go_no_go.md`](reports/history_report/stage_E6_go_no_go.md)、[`stage_E7_gate_stop.md`](reports/history_report/stage_E7_gate_stop.md)。
- **Dataset / Objective / Model:** MovieLens 与 KuaiRand；M3oE；比较围绕 Sample baseline。
- **Seeds:** P1-2 为 4-seed 配对设计；E6 为 4 seeds、24 runs；E7 未运行。
- **Effect Direction / Magnitude:** Mixed / 未达到适用 gate；E6 Adaptive-EMA overall AUC +0.001584，但 Medium-domain safety gate 失败。
- **Robustness / Scope:** Strong against the claim that these tested variants passed their own gates；不覆盖所有 schedule/controller。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 在报告覆盖的实现、预算与 gate 内，未建立课程或自适应方法稳定超过静态 Sample 的证据。
- **Alternative Explanations:** 预算、schedule 范围、域定义、selected vs common/final 指标、单个配置失配。
- **Evidence Strength:** STRONG。

## E007 — P7 组件迁移结果受骨干实现范围限制

- **Observation:** MovieLens D=3 的 comp1−none 平均 AUC：MMoE +0.002978、PLE +0.005538、ShareBottom 适配 +0.004012；STAR 因复现差距过大而未纳入 comp1 transfer 结论。
- **Source Experiments:** EXP019。
- **Source Reports:** [`P7_generalization.md`](reports/history_report/P7_generalization.md)。
- **Dataset / Objective / Model:** MovieLens；M3oE、MMoE、PLE、STAR、ShareBottom 本地适配；历史 objective。
- **Seeds:** 依子矩阵；ShareBottom 单独补跑 5 seeds × 2 arms。
- **Effect Direction / Magnitude:** 已测 MMoE/PLE/ShareBottom 均为小正值；STAR 不可比。
- **Robustness / Scope:** Weak for cross-model mechanism；适配结构、跑批窗口和复现状态不同。
- **Current Status:** TENTATIVE。
- **Current Interpretation:** 仅能报告已测移植条件的表现；不构成 S3–S6 shared-access/history mechanism 的跨模型复制。
- **Alternative Explanations:** 参数量、层适配、任务数 D/T、跑批窗口、objective、初始化。
- **Evidence Strength:** WEAK。

## E008 — S0 没有支持 comp1 相对统一 EMA 的独特机制优势

- **Observation:** S0 comp1−sample Adam 的 macro AUC 差 +0.005307 ± 0.000523；comp1−validation-selected uniform EMA 为 −0.003033 ± 0.000815，3 个 rotation 平均后每个 seed 都是 uniform EMA 更好。
- **Source Experiments:** EXP020。
- **Source Reports:** [`stage_S0_analysis.md`](reports/stage_S0_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** MovieLens 强不均衡；Sample；M3OE。
- **Seeds:** 4，且每 seed 先平均 3 个角色轮换。
- **Effect Direction / Magnitude:** comp1 对 Adam 正；对可选 uniform EMA 负。
- **Robustness / Scope:** Strong within matched exposures and tested controls。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 残余效应可复现，但被统一平滑对照覆盖；不能再称 domain-conditioned momentum 的独立收益已建立。
- **Alternative Explanations:** 平滑强度、选点准则、rotation aggregation、特定 MovieLens 配置。
- **Evidence Strength:** STRONG。

## E009 — Sample joint training 在 S1 测试范围内未显示稳定实用 residual NT

- **Observation:** 192 个运行覆盖两个数据集、三个 backbone 和单域/联合设置；288 paired seed units 中，72 组汇总有 63 正、9 负，但 0/72 达到稳定 practical NT 门槛。最负候选约 −0.002923。
- **Source Experiments:** EXP021。
- **Source Reports:** [`stage_S1_analysis.md`](reports/stage_S1_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** MovieLens 与 KuaiRand；Sample；M3OE/MMoE/PLE。
- **Seeds:** 4 per configuration。
- **Effect Direction / Magnitude:** Overall joint−SD 有利；少数 per-cell 小负值低于预设 practical threshold。
- **Robustness / Scope:** Strong for the preregistered Sample + SD-reference matrix; not all objectives or ratios。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 不支持把普遍 Sample residual NT 作为当前已确认问题；不证明每一 cell 或未测极端 ratio 均无损伤。
- **Alternative Explanations:** SD reference variance、backbone、dataset、cell/sample counts、selected/common checkpoint。
- **Evidence Strength:** STRONG。

## E010 — S2 Equal 的局部 selected loss 与 Sample 的缓解均受目标/数据条件约束

- **Observation:** S2 Equal−SD 的 8/24 cell groups 达到 ≤−0.003，严格 gate 3/24；全部八组位于 Large role。KuaiRand tab1 click 为 −0.045006 ± 0.000602（4/4 seeds、5/5 common epochs）；MovieLens d2 rating>3 为 −0.018093 ± 0.004687（4/4）。Sample−SD 的 0/24 cell groups 达阈值。
- **Source Experiments:** EXP022。
- **Source Reports:** [`stage_S2_analysis.md`](reports/stage_S2_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural 与 MovieLens d2_large；Equal/Sample；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** Equal 下目标/角色依赖；Sample 的 selected threshold 结果近零或正。
- **Robustness / Scope:** Strong for the identified KuaiRand Equal/tab1 cell; medium for cross-dataset size-role generalization because MovieLens late/common views can change direction。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 等权目标下有窄范围、可重复的 selected large-cell loss；不应改写成所有“大域受损”或“Equal 一概错误”。
- **Alternative Explanations:** Objective weights、dataset/target identity、基线、收敛和 checkpoint selection。
- **Evidence Strength:** STRONG。

## E011 — S3 在 KuaiRand Equal/tab1 中识别 shared-access 对 target 的条件性影响

- **Observation:** TARGET_ONLY_SHARED−FULL 在 selected target 上 click/long_view AUC 分别 +0.038876 ± 0.003950 / +0.039576 ± 0.004107（4/4）；相应 interaction 为 +0.038579 / +0.039440（4/4）。Sample 下相同干预约 +0.0003/+0.0001。
- **Source Experiments:** EXP023。
- **Source Reports:** [`stage_S3_analysis.md`](reports/stage_S3_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal 与 Sample；M3OE；target=d1/tab1。
- **Seeds:** 4；8 个严格配对 blocks 覆盖两个数据集与两个 objective。
- **Effect Direction / Magnitude:** Equal 下 target rescue 大；Sample 下接近零。
- **Robustness / Scope:** Strong within KuaiRand/natural/M3OE/Equal/tab1 and selected endpoint。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** shared parameters 的跨域 update access 是该 Equal 条件 target 结果的重要因果因素；该对照本身不证明纯几何 gradient conflict。
- **Alternative Explanations:** 更新组成、objective coefficients、optimizer history、checkpoint differences、domain/task interaction。
- **Evidence Strength:** STRONG。

## E012 — S3 MovieLens selected replication 有 late-training 与 non-target 例外

- **Observation:** MovieLens Equal selected target ages_35_45_50_56 的 access rescue 为 +0.019177（task0）/+0.014118（task1），4/4 同向；macro/non-target 指标不总是同向，后期训练有收敛例外。
- **Source Experiments:** EXP023。
- **Source Reports:** [`stage_S3_analysis.md`](reports/stage_S3_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large；Equal/Sample；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** selected target 正；non-target 有代价，common/final 方向依阶段。
- **Robustness / Scope:** Medium; one dataset and target, with later checkpoint counterexamples。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** 该方向在 MovieLens selected 点也出现，但不足以升级为跨数据集、跨训练阶段的普遍安全收益。
- **Alternative Explanations:** checkpoint、过拟合、dataset-specific target/task、non-target aggregate choice。
- **Evidence Strength:** MEDIUM。

## E013 — raw-gradient norm 与 applied-step norm 匹配都没有复现 S4 target rescue

- **Observation:** KuaiRand d1/tab1 中 N−F 两任务 AUC −0.004048/−0.004873（0/4 正）；U−F 为 −0.000587/−0.000942（2/4 正）；T−F 为 +0.038876/+0.039576（4/4 正，且 5/5 common epochs 为正）。
- **Source Experiments:** EXP024。
- **Source Reports:** [`stage_S4_analysis.md`](reports/stage_S4_analysis.md)、[`stage_S4_A2_review.md`](reports/stage_S4_A2_review.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** T rescue 稳定；两幅度控制未复现。
- **Robustness / Scope:** Strong; execution/control audits passed for the canonical A2 matrix。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 匹配这两种幅度统计不足以解释 S4 的 rescue；梯度组成/方向与持久 history 的联合解释仍需按后续 S5 限定。
- **Alternative Explanations:** m/v history、梯度方向、每臂参数轨迹、objective-specific behavior。
- **Evidence Strength:** STRONG。

## E014 — S4 non-target 代价与 d2 late gain 是 first-class trade-off

- **Observation:** T−F 在 KuaiRand d0 selected 两 task 约 −0.0314/−0.0321；d2 从 selected 混合方向转为 final +0.019810/+0.036160（4/4），且 LogLoss final 明显降低。
- **Source Experiments:** EXP024。
- **Source Reports:** [`stage_S4_analysis.md`](reports/stage_S4_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** selected non-target damage; late d2 gain。
- **Robustness / Scope:** Medium; domain/task and checkpoint dependent; domain sizes were not experimentally changed here。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** target rescue 与全域表现并不等价；late d2 result cannot support a monotonic domain-size claim。
- **Alternative Explanations:** checkpoint selection、训练阶段、域身份、过拟合和 history effects。
- **Evidence Strength:** MEDIUM。

## E015 — S5 Equal KuaiRand d1 rescue follows persistent Adam history

- **Observation:** T−F selected d1 AUC +0.041426/+0.042254（4/4）；HT−F +0.041085/+0.040528（4/4，约恢复 96–99%）；DH−F +0.000875/+0.000821（约 2%）。S5 moment gate 为 15/16；seed 2024/epoch1/m/HT 有限定。
- **Source Experiments:** EXP025。
- **Source Reports:** [`stage_S5_analysis.md`](reports/stage_S5_analysis.md)、[`stage_S5_review.md`](reports/stage_S5_review.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural；Equal；M3OE；target d1/tab1。
- **Seeds:** 2022–2025（4）。
- **Effect Direction / Magnitude:** HT follows T; DH remains close to F on target selected.
- **Robustness / Scope:** Strong within the tested Equal / d1 / M3OE configuration; keep moment-gate exception visible。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 在该条件下，d1 rescue 跟随完整 persistent history carrier；不识别 m、v 或 bias-correction 的单独作用。
- **Alternative Explanations:** Objective、parameter trajectory、moment component、checkpoint rules、seed 2024 deviation。
- **Evidence Strength:** STRONG。

## E016 — S5 Equal 的 d0 history×current interaction 与 d2 late gains 有 seed/stage dependence

- **Observation:** KuaiRand d0 T−F selected 为 −0.027338/−0.037392（0/4 正）；HT−F 均值较小但 seed 2024 仍明显为负，final HT−F 转为 +0.017840/+0.023840（4/4）。d2 HT−F selected/final 分别增加，final 4/4 正。
- **Source Experiments:** EXP025。
- **Source Reports:** [`stage_S5_analysis.md`](reports/stage_S5_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** Mixed by domain and checkpoint; HT mitigates selected d0 cost on average; late d2 benefit。
- **Robustness / Scope:** Medium; the seed 2024 exception and selected/final reversal matter。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** history×current interaction is a viable description for the tested trajectory, not a general safety or size law。
- **Alternative Explanations:** seed variance、checkpoint timing、domain/task identity、objective、optimization trajectory。
- **Evidence Strength:** MEDIUM。

## E017 — S5-ML Equal target rescue partially replicates S5, while DH is not zero

- **Observation:** MovieLens d2 target T−F selected +0.019408/+0.014270（4/4）；HT−F +0.018114/+0.014149（4/4）；DH−F +0.005859/+0.003583（2–3/4）。晚期 DH−F final/common target AUC 约 +0.0098/+0.0103（4/4），而 T rescue 衰减。
- **Source Experiments:** EXP026。
- **Source Reports:** [`stage_S5ML_analysis.md`](reports/stage_S5ML_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large；Equal；M3OE。
- **Seeds:** 4；16 canonical runs；moment gate 8/8。
- **Effect Direction / Magnitude:** Selected F→T rescue replicated; HT≈T holds; DH≈F does not fully hold and is checkpoint dependent。
- **Robustness / Scope:** Medium; mechanism factorization differs from KuaiRand and late MovieLens is overfit。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** history-dominant HT half replicates; the complete “current effect is zero under FULL history” pattern does not generalize to this dataset.
- **Alternative Explanations:** imbalance severity、task structure、late overfitting、checkpoint timing、domain identity。
- **Evidence Strength:** MEDIUM。

## E018 — S6 Phase A Sample F/T 未达到两个预注册 target practical gates

- **Observation:** KuaiRand d1 selected T−F −0.000657/−0.000790（仅 1/4 正）；MovieLens d2 +0.001899/+0.002555（2/4 正）；四项均低于 +0.003。共同训练阶段也未出现稳定 target rescue。Sample T 对 selected non-target cells 有损伤。
- **Source Experiments:** EXP027。
- **Source Reports:** [`stage_S6_phaseA_analysis.md`](reports/stage_S6_phaseA_analysis.md)、[`stage_S6_phaseA_final_analysis_20260929.md`](reports/stage_S6_phaseA_final_analysis_20260929.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 每数据集 4；共 16 新 runs。
- **Effect Direction / Magnitude:** KR small negative; ML small positive; neither meets the practical gate。
- **Robustness / Scope:** Strong for the preregistered Phase A STOP decision; does not establish an exact zero effect。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** S5 Equal 的 target rescue 不能外推为 Sample objective 下同样大小的 selected rescue。
- **Alternative Explanations:** Seed variance、objective coefficient、target identity、sample split、checkpoint selection。
- **Evidence Strength:** STRONG。

## E019 — S6 Phase B Sample selected target 仍无 practical rescue；KuaiRand M1 未通过

- **Observation:** 四臂 selected target T−F、HT−F、DH−F means 在两数据集均未达到 +0.003；KuaiRand HT selected LogLoss 对 target 4/4 改善但 AUC 无 practical mean rescue。KuaiRand early m conjunction 仅 4/8 通过、v 8/8；MovieLens m/v 均 8/8。代码/执行及配置一致性审查通过，但 KuaiRand 完整 M1 identification gate 为 FAIL。
- **Source Experiments:** EXP028。
- **Source Reports:** [`stage_S6_analysis.md`](reports/stage_S6_analysis.md)、[`stage_S6_phaseB_audit_review.md`](reports/stage_S6_phaseB_audit_review.md)、[`stage_S6_phaseB_run_report.md`](reports/stage_S6_phaseB_run_report.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 4 per dataset; 16 new HT/DH runs plus Phase A F/T anchors。
- **Effect Direction / Magnitude:** Selected target mixed and below threshold; metric-specific effects remain。
- **Robustness / Scope:** Medium; KuaiRand optimizer-history interpretation is limited by M1 failure; do not treat structural validity as all gates passed。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** Sample 下 selected target practical gate 继续为 STOP；完整的 history/current mechanism 在 KuaiRand 不能作无条件 identification claim。
- **Alternative Explanations:** Arm-specific parameter paths、effective weight decay、seed variance、AUC/LogLoss difference、checkpoint selection。
- **Evidence Strength:** MEDIUM。

## E020 — S6 Sample MovieLens late DH target gain accompanies severe F deterioration

- **Observation:** MovieLens d2 target DH−F at common epoch39/49 approximately +0.003138/+0.010500 (task0) and +0.008317/+0.008107 (task1); final target LogLoss differences −0.073890/−0.045756 (4/4). Meanwhile F target AUC falls from selected .721797/.739511 to final .693747/.710896, and its LogLoss rises substantially. KuaiRand does not show the same late target rescue.
- **Source Experiments:** EXP028; S6 DH trajectory reanalyses reuse this matrix.
- **Source Reports:** [`stage_S6_analysis.md`](reports/stage_S6_analysis.md)、[`stage_S6_ML_DH_trajectory_reanalysis.md`](stage_S6_ML_DH_trajectory_reanalysis.md)、[`stage_S6_KR_DH_trajectory_reanalysis.md`](stage_S6_KR_DH_trajectory_reanalysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large and KuaiRand d1; Sample; M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** MovieLens late DH target metric positive; selected target rescue null; KuaiRand late target pattern absent。
- **Robustness / Scope:** Medium; fixed late checkpoints, dataset specific, and F is severely overtrained at MovieLens late epochs。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** This is evidence of a late-stage trajectory effect under the tested protocol; it is not evidence that a method improves the normally selected model.
- **Alternative Explanations:** Ordinary overfitting/overtraining、checkpoint timing、history/current policy、metric/target mismatch。
- **Evidence Strength:** MEDIUM。

## E021 — Equal/Sample history effects are objective dependent, but “Sample removes every optimizer effect” is too broad

- **Observation:** Equal target T−F selected: KuaiRand +0.041426/+0.042254; MovieLens +0.019408/+0.014270. Under Sample, Phase A T−F is near zero or below threshold in both datasets. Phase B later shows MovieLens common-late DH−F target gains, while selected Sample target contrasts remain below threshold.
- **Source Experiments:** EXP025–EXP028。
- **Source Reports:** [`stage_S5_analysis.md`](reports/stage_S5_analysis.md)、[`stage_S5ML_analysis.md`](reports/stage_S5ML_analysis.md)、[`stage_S6_analysis.md`](reports/stage_S6_analysis.md)、[`exp_sn.md`](reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand and MovieLens; Equal vs Sample; M3OE。
- **Seeds:** 4 per dataset/stage。
- **Effect Direction / Magnitude:** Equal selected rescue is large; Sample selected rescue is below practical gate; late MovieLens Sample DH pattern remains metric/stage specific。
- **Robustness / Scope:** Medium; matched objectives and dataset targets differ; S6 M1 limitation applies to KuaiRand Phase B。
- **Current Status:** SUPERSEDED（for the broad claim that all optimizer/history effects disappear under Sample）。
- **Current Interpretation:** Current safe claim is that the Equal selected target-rescue pattern does not transfer at practical magnitude to the tested Sample selected target gates; other late optimizer-policy effects are not all zero。
- **Alternative Explanations:** Objective aggregation、overfitting、domain size and identity、AUC vs LogLoss、selected vs common checkpoint。
- **Evidence Strength:** MEDIUM。

## E022 — S6 derived trajectory analyses show non-target checkpoint effects, not independent replications

- **Observation:** MovieLens DH−F test non-target gain at selected checkpoint has 4-seed mean approximately 0, but its longest common-epoch run above +0.003 spans K=5; KuaiRand selected non-target mean is +0.0068, while its qualifying common run is K=3 and is mixed by domain.
- **Source Experiments:** EXP028 (same source runs; no new training).
- **Source Reports:** [`stage_S6_ML_DH_trajectory_reanalysis.md`](stage_S6_ML_DH_trajectory_reanalysis.md)、[`stage_S6_KR_DH_trajectory_reanalysis.md`](stage_S6_KR_DH_trajectory_reanalysis.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand; Sample; M3OE。
- **Seeds:** 4 each; five common checkpoint epochs in trajectory reports。
- **Effect Direction / Magnitude:** Selected and common-stage views differ; MovieLens has a late non-target pattern, KuaiRand is weaker/mixed.
- **Robustness / Scope:** Medium for the descriptive trajectory; not independent replication.
- **Current Status:** SUPERSEDED (as a claim of normal-checkpoint method benefit by S8); underlying fixed-epoch observations remain in the source record。
- **Current Interpretation:** Stage-dependent test observations must be retained, but S8's best-vs-original-best test narrows their practical interpretation to overfit-trajectory rescue。
- **Alternative Explanations:** Overfit deterioration in F、different selected epochs、non-target weighting、domain-specific response。
- **Evidence Strength:** MEDIUM。

## E023 — S7-A cosine structure follows domain, while ConflictMass is not a stable performance marker

- **Observation:** Across all probe seeds/checkpoints, same-domain cross-task cosine exceeds cross-domain pair values; cross-domain cosine changes sign across epochs. ConflictMass has no robust shared stage trend (MovieLens split; KuaiRand falls in 3/4).
- **Source Experiments:** EXP029。
- **Source Reports:** [`stage_S7_analysis.md`](reports/stage_S7_analysis.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 4 per dataset; 8 probe runs。
- **Effect Direction / Magnitude:** Domain-structured cosine; no stable monotonic cross-domain conflict trend。
- **Robustness / Scope:** Medium; replicated across snapshots within these two datasets, but measurements are local gradient geometry。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** Domain×Task decomposition reveals structure hidden by task aggregation; cosine/ConflictMass alone does not establish harmful performance effect。
- **Alternative Explanations:** Gradient norm scale、module reachability/dead ReLU、sample weighting、dataset-specific task structure。
- **Evidence Strength:** MEDIUM。

## E024 — S7-A validation-gradient utility is performance-aligned in late MovieLens, not universally sufficient

- **Observation:** MovieLens F target performance deteriorates from early to late (target AUC −0.0208; 4/4); U_k becomes increasingly negative over later epochs. KuaiRand target AUC continues rising to epoch49 (4/4), even though target U_k is negative in 3/4 seeds at that snapshot.
- **Source Experiments:** EXP029。
- **Source Reports:** [`stage_S7_analysis.md`](reports/stage_S7_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large and KuaiRand natural；Sample；M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** MovieLens late U_k and validation deterioration co-move; KuaiRand negative U_k does not coincide with realized AUC damage。
- **Robustness / Scope:** Medium; observations are stable inside MovieLens but the KuaiRand counterexample limits causal sufficiency。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** U_k is a conditional local first-order statistic; a negative value is not sufficient for realized performance decline.
- **Alternative Explanations:** Adam preconditioning、multi-step trajectory、overfitting、gradient norm growth、metric noise。
- **Evidence Strength:** MEDIUM。

## E025 — S7-B changes late MovieLens gradient utility without validation improvement

  - **Observation:** At same θ, MovieLens anchor39 target-current changes min-U on steps1–4 positively for all four seeds (approximately +0.0074 to +0.169); anchor9 is near zero/mixed. At anchor39 step5, target AUC mean difference is −0.001226 but only 3/4 seeds are negative (seed2025 is +0.002359); LogLoss mean is +0.011130 with 3/4 worse. KuaiRand has no stable early/late response contrast. S7-B uses fresh-Adam private parameters after step0, a declared approximation.
- **Source Experiments:** EXP030。
- **Source Reports:** [`stage_S7B_analysis.md`](reports/stage_S7B_analysis.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand; Sample; M3OE。
- **Seeds:** 4 per dataset × 2 anchors (16 runs)。
- **Effect Direction / Magnitude:** Gradient utility changed at late ML anchor; short-window validation did not improve.
- **Robustness / Scope:** Medium; at anchor39 step5 AUC is negative in 3/4 seeds and LogLoss is worse in 3/4, the window is short, and private optimizer state is approximate。
- **Current Status:** SUPPORTED（for the short-window observation）。
- **Current Interpretation:** Gradient-geometry change is not enough to infer validation benefit; the proposed immediate gradient-to-performance chain is weakened。
- **Alternative Explanations:** Short horizon、private-state approximation、Adam effective update、ordinary overfitting trajectory。
- **Evidence Strength:** MEDIUM。

## E026 — S8 finds no practical continuation benefit at selected checkpoints

- **Observation:** S8 selected-anchor h=2-epoch continuation T−F and DH−F mean best-AUC differences are both +0.000062 on MovieLens; KuaiRand T−F −0.000445 and DH−F +0.000068. These are below +0.003; no MovieLens branch consistently sets a new best above the frozen F original best.
- **Source Experiments:** EXP031。
- **Source Reports:** [`stage_S8_analysis.md`](reports/stage_S8_analysis.md)、[`S8_REVIEW.md`](reports/S8_REVIEW.md)。
- **Dataset / Objective / Model:** KuaiRand and MovieLens；Sample；M3OE。
- **Seeds:** 4 per dataset; 8 base runs and 24 continuations。
- **Effect Direction / Magnitude:** Selected-anchor branch contrast null / far below threshold。
- **Robustness / Scope:** Strong for the tested seeds, anchors, branches, 2-epoch horizon, and frozen-best endpoint; the run validation gates passed。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** The optimizer/current-source interventions did not improve normally selected performance to a practical magnitude in this protocol。
- **Alternative Explanations:** Narrow continuation horizon、selected-state proximity to optimum、dataset-specific overfit behavior。
- **Evidence Strength:** STRONG。

## E027 — S8 late MovieLens gain is an overfit rescue pattern, not a better best model

- **Observation:** MovieLens late-anchor DH−F best-window macro AUC +0.006826 (4/4) and final-step +0.010732 (4/4), but about 89% of the latter mean comes from seeds 2024/2025. Target-domain gain is much smaller (+0.000441 best; +0.002566 final), and every continuation best remains below that seed's frozen F original best. KuaiRand late T is harmful (final −0.010239, 0/4 positive) and DH is near zero.
- **Source Experiments:** EXP031; reinterprets E020/E022.
- **Source Reports:** [`stage_S8_analysis.md`](reports/stage_S8_analysis.md)、[`S8_REVIEW.md`](reports/S8_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand；Sample；M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** Late MovieLens macro DH appears positive; target/best endpoint is small; no branch beats original best. KuaiRand is a negative control.
- **Robustness / Scope:** Medium; late macro effect is seed-skewed and metric dependent; execution was fully validated.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The evidence supports partial recovery from an overtrained late anchor, not a method that improves checkpoint-selected model performance.
- **Alternative Explanations:** Ordinary overfitting/overtraining、non-target macro contribution、seed leverage、checkpoint horizon。
- **Evidence Strength:** MEDIUM。

## E028 — KuaiRand DH has a selected-checkpoint test-side non-target gain under Sample

- **Observation:** At the canonical R0 validation-macro selected checkpoints, KuaiRand test Δ(DH−F) is N +0.0068 ± 0.0046 (4/4 positive; 3/4 ≥ +0.005), M +0.0044 ± 0.0029 (4/4 positive), and T −0.0004 ± 0.0031. Seed 2023 has target ΔT = −0.0047. On the six-point A-grid sensitivity variant, R1/R3/R4 test ΔN means are +0.0041/+0.0060/+0.0041, each 3/4 positive; their mean target deltas remain above −0.003. Exact test availability for truly reselected R1/R3/R4 is 0/4, 0/4, and 1/4 seeds respectively, so those results do not establish full-seed exact-rule robustness.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partA_gate_summary_kuairand.csv`](stage_S6_R2_raw_results/kuairand/s6r2_partA_gate_summary_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Positive test non-target and macro means at canonical R0; target mean is near zero/slightly negative. Alternative-rule test robustness is A-grid sensitivity only.
- **Robustness / Scope:** Medium; R0 N is 4/4 positive, but seed 2025 is near zero, one target seed crosses −0.003, and the alternative rules rely on a restricted checkpoint grid.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The tested KuaiRand/Sample/M3OE DH checkpoints have a test-side selected non-target advantage relative to F. This is not a validation-side advantage or evidence of a universal optimizer mechanism.
- **Alternative Explanations:** Validation-to-test generalization gap、checkpoint timing、selection rule、seed variation。
- **Evidence Strength:** MEDIUM。

## E029 — KuaiRand selected-checkpoint advantage reverses between validation and test

- **Observation:** At the same R0-selected checkpoint pairs, KuaiRand ΔN is −0.0039 ± 0.0081 on validation (1/4 positive) and +0.0068 ± 0.0046 on test (4/4 positive); ΔM is −0.0027 on validation and +0.0044 on test. The validation decomposition has N PolicyAtF = −0.0111 and PolicyAtDH = +0.0082 (four-seed means), with TimingAlongF = −0.0120. Test cross-epoch decomposition is unavailable because canonical epochs are off the saved grid; the saved same-epoch test N difference is positive at 16/20 seed×epoch points.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partB_decomposition_kuairand.csv`](stage_S6_R2_raw_results/kuairand/s6r2_partB_decomposition_kuairand.csv)、[`s6r2_partB_grid_trajectory_kuairand.csv`](stage_S6_R2_raw_results/kuairand/s6r2_partB_grid_trajectory_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Validation non-target/macro means favor F; test non-target/macro means favor DH; target test mean is −0.0004.
- **Robustness / Scope:** Medium for the layer-specific observations; the causal source of the val–test reversal is unresolved, and the test decomposition is blocked by missing off-grid checkpoints.
- **Current Status:** UNRESOLVED。
- **Current Interpretation:** The direction depends on evaluation layer at the same selected checkpoints. The result does not show a validation-visible improvement or identify why the held-out test gap changes direction.
- **Alternative Explanations:** Checkpoint timing、validation/test distribution differences、seed-level heterogeneity。
- **Evidence Strength:** MEDIUM。

## E030 — MovieLens is a selected-checkpoint negative control in S6-R2

- **Observation:** Under MovieLens R0, selected test ΔN is approximately −0.000002 ± 0.0031 (2/4 positive), ΔM is approximately 0.0000 ± 0.0029 (2/4 positive), and ΔT is +0.0001 ± 0.0026. No available exact or A-grid selection rule shows a stable test non-target gain; the saved-grid same-epoch ΔN reaches +0.0334 at epoch 19, while canonical selected epochs are 7–13.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partA_gate_summary_movielens.csv`](stage_S6_R2_raw_results/movielens/s6r2_partA_gate_summary_movielens.csv)、[`s6r2_partB_grid_trajectory_movielens.csv`](stage_S6_R2_raw_results/movielens/s6r2_partB_grid_trajectory_movielens.csv)。
- **Dataset / Objective / Model:** MovieLens d2_large; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Selected test effect is near zero; late saved-grid effects can be positive but do not coincide with selected epochs.
- **Robustness / Scope:** Medium within the four-seed negative-control setting; some alternative exact-rule test values are unavailable.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The KuaiRand selected test result does not reproduce on MovieLens under the tested selection protocol; the dataset/stage contrast does not identify its cause.
- **Alternative Explanations:** Dataset and target identity、selected-epoch timing、training-stage differences、seed variation。
- **Evidence Strength:** MEDIUM。

## E031 — KuaiRand DH selected validation windows are usually broader than F windows

- **Observation:** In the four-seed KuaiRand window analysis, DH has 3 moderate plateaus and 1 narrow spike; F has 1 moderate plateau and 3 narrow spikes. DH seed 2023 is itself a narrow spike, and DH seed 2024 is edge-clipped at epoch 48.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partC_window_kuairand.csv`](stage_S6_R2_raw_results/kuairand/s6r2_partC_window_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Window classification favors more moderate DH than F selections (3/4 vs 1/4).
- **Robustness / Scope:** Weak; one dataset, four seeds, a categorical window rule, one DH spike, and one edge-clipped window.
- **Current Status:** TENTATIVE。
- **Current Interpretation:** A single narrow peak is not a sufficient description of all DH selections in this sample; the result does not establish generally broader or more stable DH optima.
- **Alternative Explanations:** Seed-specific peak shape、edge clipping、the pre-registered window threshold。
- **Evidence Strength:** WEAK。

## E032 — KuaiRand cross-partition parameter swaps collapse and do not localize the gain

- **Observation:** On the four KuaiRand seeds, C_hybrid test macro ranges 0.5450–0.5842 and D_hybrid 0.5196–0.5791, versus F_full mean 0.6730 and DH_full mean 0.6774. E_hybrid closely matches C_hybrid; F_hybrid differs from F_full by at most 0.00011 macro and 0.00016 non-target. Seed 2025 has a DH−F non-target denominator of about 1.1e−5, so its retention ratio is not informative. The manifest partitions and state-dict shapes pass the reported integrity checks.
- **Source Experiments:** EXP032; underlying KuaiRand F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partD_swap_kuairand.csv`](stage_S6_R2_raw_results/kuairand/s6r2_partD_swap_kuairand.csv)、[`s6r2_integrity_report.json`](stage_S6_R2_raw_results/s6r2_integrity_report.json)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE; manifest split into shared, architecture, and private parameter sets。
- **Seeds:** 4, 2022–2025; evaluation-only swaps, no new training。
- **Effect Direction / Magnitude:** Both cross-partition hybrids have large negative test-macro changes relative to the intact parent solutions; the F architecture-only hybrid is nearly unchanged from F_full.
- **Robustness / Scope:** Medium for these tested swaps across four seeds; one seed's retention ratio is undefined in practice and the result is limited to one model/data/partition definition.
- **Current Status:** SUPPORTED（for the observed swap outcome）。
- **Current Interpretation:** The tested partitions are not independently interchangeable and are consistent with cross-partition co-adaptation. These swaps do not identify which component causes the intact DH test-side advantage.
- **Alternative Explanations:** Cross-partition co-adaptation、parameter-manifest boundary、seed 2025's near-zero denominator。
- **Evidence Strength:** MEDIUM。

## Strength tally

| Evidence Strength | Count |
|---|---:|
| STRONG | 10 |
| MEDIUM | 19 |
| WEAK | 3 |
| **Total** | **32** |
