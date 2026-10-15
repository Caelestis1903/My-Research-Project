# EXP_AI_CONTEXT_V2.md
# 面向后续 AI 实验的最小充分科研上下文

## 0. 文档用途与裁决规则

本文件供 M3oE / 多域多任务学习项目的后续 AI 维持当前科研状态、边界和执行约束；它是从 EXP_AI_CONTEXT.md 去重压缩的常驻上下文，不替代完整归档、已签署阶段报告、原始 CSV 或运行日志。文中每个科学事实以其所在 Core Claim 或 Anchor 为主记录位置，其他章节只引用编号。

当前裁定优先于早期解释。若本文件与归档冲突，回到对应正式 Reviewer / analysis 和原始数据核对；统计值以原始 CSV 为准，机制结论以最新正式裁定为准。没有正式裁定的冲突保持 OPEN，不自行择边。FACT 指指定协议下直接观察或核验；INTERPRETATION 指有支持但因果边界有限；OPEN 指尚未区分；STOP / DROP 指当前路线裁定；证据强度使用 STRONG / MEDIUM / WEAK。不得把诊断性结果写成已验证方法或更广泛因果事实。

---

## 1. 当前研究状态

### 1.1 当前研究问题

项目从复现 M3oE 和检验 size-driven per-domain momentum 起步。固定总样本量、轮换大小角色、操纵 objective 等后续审计逐步改写早期归因；核心结论与其适用边界见 C2–C7，数字集中在第 5 章。

后续研究先检验普遍 residual negative transfer，再比较 Equal / Sample 目标，随后将 Equal 下的局部 target rescue 定位到 KuaiRand-Pure、M3OE、tab1 的 Adam history 机制。S6 将同一 F/T access 对照切换到 Sample objective，在 KuaiRand 和 MovieLens 两个最大域 target 上均未达到稳定 practical rescue gate；两数据集 STOP，预注册 Phase B（HT/DH）未触发。因此现有证据支持 history effect 是 Equal 条件下的 objective-contingent 现象，不支持把它当成 Sample 下仍需方法化的残余问题。target-only 对非目标域的代价仍存在，MovieLens final 的宏观改善又受 F 晚期过拟合混淆。研究线当前裁决为 STOP METHOD LINE；不得自动继续 m/v 拆分、ratio boundary 或方法开发，也不得将局部结果包装成通用梯度冲突或安全 MTL 方法。

### 1.2 当前核心科学结论

**C1 — 基线复现成立，但旧 AutoML 贡献与 optimizer grouping 需分开判读。**  
Evidence anchor：A1–A2。Boundary：复现和无可测 AutoML 差异只适用于所测旧协议；B10 修复前后不可无标记地合并方差，也不据此否定其他 architecture search。Status：CONFIRMED。

**C2 — 在已测 MovieLens 梯度诊断中，小域 SNR 并不稳定低于大域。**  
Evidence anchor：A3。Boundary：仅限该数据、共享参数集合、batch size 和统计口径，不证明其他数据集也无域间噪声差异。Status：SUPPORTED。

**C3 — comp1 的早期性能增益是真实历史现象，但单调 ratio、单一 size 机制和真 Pareto 外推均不成立。**  
Evidence anchor：A4–A6。Boundary：固定 N 结果说明 ratio 可贡献部分增益，但不识别 size 单因果；KuaiRand 旧 oracle 对照仅支持前沿移动的方向性证据，不能当作当前方法净收益。Status：SUPPORTED。

**C4 — 固定 N 的轮换实验中，退化随 Large 角色移动且不局限于 M3oE；角色定位比 size 单因果更可信。**  
Evidence anchor：A6。Boundary：size、有效权重、份额、多样性、梯度尺度和收敛共同变化；size 的独立因果强度为 MEDIUM。Status：SUPPORTED。

**C5 — Equal 与 Sample 是不同有效目标；修正目标可大幅缓解已测 Large-cell 损伤，Sample 是当前稳健基线。**  
Evidence anchor：A7–A9。Boundary：该效应不等于单一中介已识别；E2 未执行，E1 是观察性诊断，E4 跨数据集方向为 STRONG 但归因受 train-test shift 限制；Sample+comp1 的总体均值不等于每 cell 都为正。固定 exposure 的目标效应证据为 STRONG。Status：CONFIRMED。

**C6 — 已测 curriculum、预设时间 schedule 与 adaptive calibration 均未稳定超过静态 Sample。**  
Evidence anchor：A10。Boundary：只覆盖已测机制、实现与预算；E6 虽可降低目标 mismatch，但未稳定胜过 Sample，EMA 并非必要机制且中间域触发安全限制；curriculum 未完整到 rho=1，E7 被 gate 停止且没有运行。Status：CONFIRMED。

**C7 — comp1 的 Sample 口径残余效应存在，但 S0 不支持 domain-conditioned momentum 有独立优势。**  
Evidence anchor：A11。Boundary：匹配 uniform-smoothing family 足以解释该残余，不证明 uniform EMA 在所有 setting 最优；效应存在与原机制归因是两件事。S0 范围内裁定为 STRONG。Status：CONFIRMED。
补充限定：comp1 对单一预注册 C1 match 在部分 common-epoch 口径仍为正；STOP 的含义是其未表现出相对 validation-selectable uniform-smoothing family 的独立机制优势，而非对所有 uniform control 全面劣势。

**C8 — Sample + 架构匹配 SD reference 下，普遍 residual NT / Safe-MTL 主线未达稳定实用门槛。**  
Evidence anchor：A12。Boundary：S1 不排除 Equal 目标或特定任务的条件性退化；候选不得脱离 reference 方差、共同训练阶段和阈值解释。S1 预注册范围内证据为 STRONG。Status：CONFIRMED。

**C9 — Equal 下存在窄的条件性 Large-cell NT，Sample 在 selected practical threshold 上缓解；指标与训练阶段会改变表观结论。**  
Evidence anchor：A13–A16。Boundary：KuaiRand tab1 条件效应证据为 STRONG，size role 的独立因果解释为 MEDIUM；基线难度不足以单独解释全部效应，AUC / LogLoss、selected / common / late-stage 不可互换。Status：SUPPORTED。

**C10 — S3 在 KuaiRand Equal/tab1 识别出 shared-access 对目标表现的因果影响，同时观察到跨域代价。**  
Evidence anchor：A17–A18。Boundary：结论限于 M3OE、KuaiRand、Equal、tab1 与指定 checkpoint；MovieLens selected replication 有 late-stage 反例，target-only 仅为诊断干预，目标救援不等于全局改善。条件内证据为 STRONG。Status：CONFIRMED。

**C11 — S4 的 raw-gradient norm 与即时 applied-step norm 匹配均未复现 d1 救援；当时只能将差异定位到写入 Adam moments 的梯度流组成 / 方向及其历史。**  
Evidence anchor：A19–A21。Boundary：N−F、U−F 在两个 d1 task 上均未达到实用救援门槛，控制执行审计通过；S4 尚不能分开每步方向与 persistent history，probe 的逐步方向并非一致，纯几何梯度冲突未证明。S4 同时观察到 d0 selected 代价及 d2 late AUC / LogLoss 改善；固定 KuaiRand natural 域结构没有操纵 domain size，不能把 spillover 归因于 size。Magnitude-control 结论为 STRONG，联合机制解释为 MEDIUM-STRONG。Status：由 S5 细化，见 C13。

**C12 — 当前跨架构证据与 S2–S6 的机制识别范围不同，不能相互替代。**  
Evidence anchor：A22、A24。Boundary：P7 的组件迁移限于所测 MovieLens 结构；ShareBottom 为本地适配，D=5 只测 T=2；S2–S6 的机制识别仍以 M3OE 为主。S6ML 只复核 Sample 下 F/T target contrast，不运行 HT/DH，不能视为 history-carrier 机制的跨数据集复现。外推仍为 OPEN。Status：OPEN。

**C13 — S5 将 Equal 下的 d1 rescue 定位到 persistent Adam history；d0 selected 代价依赖 history × current 的共同作用。**  
Evidence anchor：A23。Boundary：KuaiRand-Pure / M3OE / natural / Equal / tab1、seeds 2022–2025 下，selected 的 T−F 为 +0.0414/+0.0423（4/4）；HT 恢复 99.2%/95.9%，DH 仅约 2%，且 DH−F 在所有 checkpoint 的绝对值 ≤0.0027。因此当前 target direction 对该 Equal 条件的 d1 rescue 既非必要也非充分。d0 T−F 为 −0.0273/−0.0374；HT 均值将 selected 代价削减约 80%，但 seed 2024 HT 仍有 −0.038/−0.030 的例外，跨 seed 一致性 MEDIUM；HT final 转正不能替代 selected 代价。history carrier 是 m/v/step counter 的整体，m、v、bias-correction 有效步长及几何原因尚未拆分。Moment gate 为 15/16：seed 2024 / epoch 1 / m / HT 偏差经 Reviewer 裁定不影响操纵识别，但不得作该 seed 的 epoch-1 m-proximity 主张。条件内 d1 结论 STRONG；d0 交互限 selected，STRONG 方向证据、MEDIUM 跨 seed 一致性。Status：SUPPORTED（Equal / d1 history-dominant rescue）/ SUPPORTED（d0 interaction，selected）。

**C14 — S6 的 Sample-objective Phase A 在 KuaiRand 与 MovieLens 均未达到稳定 practical target-rescue 门槛；S5 的 history effect 不能外推为 objective-independent。**  
Evidence anchor：A24。Boundary：KuaiRand/natural/M3OE 的 target d1 两 task selected T−F 为 −0.0007/−0.0008（各 1/4 seeds 为正），MovieLens/d2_large/M3OE 的 target d2 为 +0.0019/+0.0026（各 2/4），均未达 +0.003 practical threshold；共同 epoch 不支持稳定正向。MovieLens final 全域 T−F 转正由 F 的晚期过拟合崩塌驱动，不是 d2 rescue。两目标均为各自最大域，不能据此推断小域 target 或 size 因果。两数据集均按预注册 gate STOP，HT/DH Phase B 未运行；所以 Sample 下 history decomposition 未被直接测量，且“不达门槛”不等于效应严格为零。Gate 裁决证据为 STRONG；Equal pathology 是机制解释，不超出 MEDIUM。Status：CONFIRMED（限预注册设置的 STOP gate）/ OPEN（机制与更广外推）。

---

## 2. 已关闭、已推翻与已降级方向

| ID | 方向 / 假设 | 早期认识 | 决定性证据 | 当前裁定 |
|---|---|---|---|---|
| CLOSED-1 | 小域梯度噪声必然更大，因此需要逐域 EMA | 小域被认为因样本少而更噪 | 见 C2 / A3 | REJECTED |
| CLOSED-2 | M3oE 可微 AutoML 是主要性能来源 | 架构搜索被视为主要增益来源 | 见 C1 / A2 | NOT PROVEN |
| CLOSED-3 | comp1 收益随 imbalance ratio 单调放大 | 强档增益被概括为单调 size 效应 | 见 C3 / A4 | SUPERSEDED |
| CLOSED-4 | M3oE 固有的大域牺牲，或 size 单独导致退化 | 见 C4 / A6 | CLOSED-4a — M3oE-specific largest-domain sacrifice → SUPERSEDED |CLOSED-4b — Domain size alone causes the degradation → NOT PROVEN
| CLOSED-5 | 与 objective 无关的普遍 small-large trade-off | 大域损失被认为是普遍 sharing 代价 | 见 C5 / A7–A9、C9 | REJECTED |
| CLOSED-6 | comp1 是修复 objective mismatch 的必要方法 | 方法增益被当作目标修正的替代 | 见 C5 / C7 | SUPERSEDED |
| CLOSED-7 | domain-conditioned momentum 有独立机制优势 | residual gain 被归因于域条件动量 | 见 C7 / A11 | STOP |
| CLOSED-8 | 域条件 curriculum 可稳定改善静态 Sample | 预期时间变化可进一步优化训练 | 见 C6 / A10 | DROP |
| CLOSED-9 | 已测时间顺序 contribution / loss schedule 可胜静态 Sample | E3 诊断被视为 schedule 有效线索 | 见 C6 / A10 | STOP |
| CLOSED-10 | 在线 adaptive calibration / EMA 可补足静态 Sample | 精确校准被认为会带来更好表现 | 见 C6 / A10；E7 runs=0 | STOP |
| CLOSED-11 | Sample 下普遍 residual NT 仍支持 Safe-MTL 主线 | 个别负候选被外推为广泛问题 | 见 C8 / A12 | STOP |
| CLOSED-12 | target-only shared update 是可部署的安全方法 | 局部救援被理解为整体改进 | 见 C10 / A18、C11 / A21 | DIAGNOSTIC ONLY |
| CLOSED-13 | 仅匹配 raw gradient norm 或即时 applied step 即可解释救援 | 更新幅度被当作充分机制 | 见 C11 / A19 | REJECTED AS A SUFFICIENT EXPLANATION |
| CLOSED-14 | 已证明通用的几何梯度冲突机制 | shared-access 救援被等同于纯方向冲突 | 见 C10–C11 | NOT PROVEN |
| CLOSED-15 | 结论可任意外推到 backbone、任务和数据集 | 少数迁移结果被视为普遍泛化 | 见 C12 / A22–A24 | NOT PROVEN |
| CLOSED-16 | Equal 下观察到的 target rescue/history effect 在 Sample 下仍达到 practical magnitude | 将 Equal 机制直接视为 objective-independent | 见 C14 / A24 | STOP WITHIN TESTED SAMPLE SETTINGS；not evidence of exact zero |

---

## 3. 当前开放问题与 Research Frontier

### RQ1 — Equal 下的 history-mediated rescue 是否也存在于 Sample objective？

**已回答（S5/S6，C13–C14）：** Equal 下 KuaiRand d1 rescue 跟随 persistent Adam history（HT 恢复约 96–99%，DH 约 2%）；但切换为 Sample 后，KuaiRand d1 与 MovieLens d2 的 F/T target contrast 都未通过 practical gate，故 Equal 下的机制不是 objective-independent 的已证事实。  
**已停止的分支：** 两数据集均 STOP，未运行 Sample 下 HT/DH，也不进入 m/v factorization。  
**边界问题：** S6 仅测两个最大域作为 target；若未来有独立理由重开，应重新设计小域 target / 不同 ratio 的可证伪对照，而不能把本次 STOP 自动延伸到未测设置。当前机制子成分问题不授权继续运行。

### RQ2 — target rescue 能否在不牺牲 non-target domains 的条件下成立？

**已知：** S3–S5 Equal 下 target-only / HT 对目标域有局部收益，也出现非目标代价与训练阶段差异（C10–C13）。S6 Sample 下两个 target 都未达到 stable practical rescue，但 target-only T 仍造成 selected 非目标域损伤；MovieLens final 的全局 T 优势来自 F 过拟合，不能当成 target rescue（C14）。HT/T 都只是诊断臂，不能当成安全方法。  
**未知：** Equal 下目标救援和非目标损伤的结构能否在其他 objective / domain-size target 下复制；Sample 下因为 STOP 未拆分 history/current。  
**下一实验必须区分：** target cell、non-target cells、全域 macro、worst-domain、AUC 与 LogLoss 的共同变化。target-only 与 HT 都仍是诊断臂，不能当作安全方法。

### RQ3 — shared-access 机制能否跨 backbone、dataset 与 target domain 复现？

**已知：** S6 在 KuaiRand 与 MovieLens 两个最大域 target 上显示 Sample 下 F/T target rescue 均不达门槛；MovieLens 的 Equal history pattern 只部分复现（C12、C14）。这不是 HT/DH history-carrier 的跨数据集验证。  
**未知：** history mechanism 是否在其他 backbone、较小 target、不同数据切分或自然域结构下复现，及其代价是否相同。  
**下一实验必须区分：** 重现的是相同的 objective × shared-access 交互，还是只出现类似目标指标变化；同时保留 source identity、数据切分、任务定义和架构 fidelity。

### RQ4 — domain size 是因果变量，还是 objective、share、难度与收敛的代理？

**已知：** Large role 与部分退化相关，但 role rotation、Sample / Equal 对照及 KuaiRand 自然分布不能单独识别 size（C4–C5、C9）。S4/S5 固定 KuaiRand natural 域身份与规模；S6 的两个 target 都是最大域（KuaiRand d1≈86.9%，MovieLens d2≈80%），小域 target 未测试。S6 中 target-only 对小域 d0 的损伤与 size 相关，但未独立操纵 size，不能作因果结论。  
**未知：** 样本数、loss coefficient、采样 exposure、样本多样性、梯度尺度和收敛速度各自贡献多少。  
**下一实验必须区分：** 在同一数据身份和目标定义下独立操纵 size role 与有效权重 / exposure，并预先定义结果视图。该问题是次级因果缺口，不应与 RQ1 的优化器机制拆分混成一个 claim。

---

## 4. 实验协议与不可违反约束

### 4.1 Baseline / objective 定义

- 原始 as-published M3oE 六项 BCE 为等权求和；它与后续 Px_n、E、S 系列的目标、数据和评估协议不同，报告不得只写“baseline”。
- S1/S2 的 Sample 定义为 w_d=1、有效系数 a_d=p_d；Equal 定义为 w_d=1/(3p_d)、有效系数 a_d=1/3，整体目标按平均权重归一。P0-4/5 使用各自总尺度归一实现；不能因名称相同就假定损失尺度完全相同。
- S1/S2 的 SD reference 是保留双任务的 D=1,T=2 单域参考；负迁移判断使用 Joint−SD、Equal−SD 或 Sample−SD，不把历史 oracle 或 Equal−Sample 差当成同义定义。
- S3 Equal FULL / Sample FULL 与 target-only 比较 shared-update access；S4 的 N 是 raw-gradient norm matching，U 是 applied-step matching；S5 的 F/T/HT/DH 操纵 history source × current source；S6/S6ML 的 Phase A 仅运行 Sample objective 下 F/T。S6 双 STOP 后未运行 Phase B。SD legacy reference、target-only、N、U、HT、DH 都不可误标为已验证新方法。
- 每个 arm 定义要写清数据、objective、优化器参数分组、更新访问、训练 exposure、checkpoint 规则与初始化映射。

### 4.2 数据与域定义

- MovieLens-1M 早期三域为 d0={1,18}、d1={25}、d2={35,45,50,56}；任务为 rating>3 和 rating>4。Px_n 固定 N=187,500 并轮换物理域的 L/M/S 角色；固定 N 下 ratio 仍贡献部分增益，ratio、role 与物理域身份不可混称。
- KuaiRand-Pure 本地 tab0/tab1/tab4 对应 d0/d1/d2；S/E 系列任务为 is_click / long_view。全量域占比约 7.10%/89.18%/3.72%，S 系列 train split 约 8.6%/86.9%/4.5%；引用占比时必须说明统计 split。
- KuaiRand 按全局 date、time_ms 排序后作 8:1:1 时序切分，是论文未公开切分规则下的重建方案，不是官方唯一管线；train-test domain mixture 与 label prior shift 是解释结果时必须保留的竞争因素。
- MovieLens 的人工角色轮换与 KuaiRand natural setting 回答不同问题。P7 的 MMoE / PLE 为层移植，ShareBottom 是本地 4-expert+mean 适配，STAR 复现失败并排除；P7 与 S2–S6 的外推边界按 C12。

### 4.3 配对、统计单位与模型选择

- seed 是重复与配对单位；同 seed 配对。domain、task、epoch、probe 不能当独立 seed 扩大 n。未另行预注册并经审查时使用 seeds 2022、2023、2024、2025。
- checkpoint 只能用 validation 选择，test 不可见。Selected、fixed-final、common epochs 是不同 estimand；分别报告，不能择有利视图。S2 selected 使用六 cell unweighted macro AUC 和最早并列规则；引用其他阶段时保留该阶段定义。
- P5/P6 的 test_auc_mean pooled AUC 不等于六个 per-(d,t) AUC 再平均，P7 pooled 也不等于 per-cell macro。每个表注明指标、聚合顺序、样本权重、ddof、paired n 和 seed block。
- AUC 衡量排序，LogLoss 衡量概率损失。S2–S6 报告须同时包含目标与非目标 cell、全局 macro / worst-domain、AUC / LogLoss 及 selected / final / common-stage；S6 MovieLens final 的 macro gain 由 F late overfitting 主导，不得等同于 target rescue。两项指标不同向时保留差异，不另立为研究假设。
- 显著性、实用阈值和证据强度分开。名义 p 值须注明是否多重校正；不显著不等于等效，单次小样本不显著不等于零效应。S2 AUC practical threshold 为 0.003、strong threshold 为 0.005，按阶段预注册门槛执行，不事后修改。

### 4.4 优化器、RNG、实现与数据记录

- B10 是历史真实 bug：Python 位取反被误当逻辑非，旧版主 Adam 包含全部参数，四个 architecture scalars 又被第二个 Adam 更新。复现或比较须标明 grouping 版本，不得混合 bug 与修正版结果。
- 创建额外 DataLoader iterator 可能从全局 RNG 抽取 base seed 并改变后续 shuffle；额外评估 / 诊断要保存并恢复 torch RNG state，并核验训练轨迹。M3oE forward 返回 CPU prediction，调用侧需转到训练 device；Windows 下 KuaiRand 使用 num_workers=0 以避免 spawn pickle 问题。
- 因果对照除预注册变量外锁定训练目标、每步 loss、optimizer state、数据 exposure、初始化映射和源码身份；改变 target access 时不可同时改变 private 参数或 loss scaling。
- P5/P6/P7 主台账没有完整留存 test LogLoss；不得从其他阶段或 validation 推算。后续声称 test LogLoss 必须有本阶段直接评估记录。
- 保留逐 seed 原始结果、失败与重跑行、执行窗口、对照配对、manifest 和 reviewer 核查路径；failed 行不得静默删除。

### 4.5 项目执行角色

- Main agent 只担任 ORCHESTRATOR 和最终科研分析者，不直接实现研究算法。
- 每个新算法实验必须依次 DESIGN → IMPLEMENT → REVIEW → RUN → ANALYZE；阶段不可跳过。Designer 先给 DESIGN SPECIFICATION，未完成设计前不得实现。
- Luna-coder 只按批准规格实现，可作语法和基础检查，不分析研究结果；Reviewer 独立检查规格、实现与 coder report。只有 PASS 或 PASS WITH FIXES 且 blocking issues 修复后才能 RUN。
- Luna-runner 只记录原始结果、不作科学结论；最后由 main agent 独立分析证据。执行新实验前还须有明确研究目标与可区分竞争解释的设计。

---

## 5. 关键数字锚点

除注明外，差值为 AUC 原始比例单位。统计量、聚合方式、seed 配对和 checkpoint 视图均沿用对应源报告；不同口径不能直接拼接。Core Claim 通过本表引用数字，其他章节不重复数字细节。

| Anchor | 条件 / 比较 | 数值锚点与限定 |
|---|---|---|
| A1 | MovieLens as-published 复现 | 五 seed 早期六个 domain-task 汇总 AUC 为 0.7686±0.0017，论文值 0.7702；旧训练和评估协议。 |
| A2 | 旧 AutoML 与 B10 | frozen 0.7311 相对原版 +0.0007，p=0.32；修复后均值无显著变化，seed SD 约 0.0017→0.0002。 |
| A3 | MovieLens 梯度 SNR | batch 40960 小 / 大域约 0.509 / 0.558；batch 4096 域差消失并反号；cosine 弱正而非明显冲突，P6 由 SNR 反推的 gamma 也显著偏离公式 gamma。 |
| A4 | MovieLens 早期 comp1 ratio sweep | 六档 comp1 差值为 +0.0064、+0.0034、−0.0012、+0.0053、+0.0166、+0.0202；r30 / r100 同时改变总 N，序列非单调。 |
| A5 | KuaiRand 早期 d1 oracle 对照 | comp1 使 t0 的 Δoracle 收窄约 +0.009102（名义 p=0.0228），t1 收窄 +0.017077（p=0.369）；15 项检验未过 Bonferroni，均未追平 oracle。 |
| A6 | Px_n 固定 N / role | P0-1 Large-role Δtransfer −0.014557±0.007439；P0-2 ratio=50 时 Large-role Δtransfer −0.015545±0.010894；固定 N 因子对照仍支持 ratio 贡献部分 comp1 增益，大小角色轮换与 MMoE 复现撤销 M3oE-specific 解释。 |
| A7 | Equal → Sample 与 P0-5 | P0-4 的 Large loss damage 修复描述值约 92%；P0-5 Sample+comp1 残余 +0.005407±0.000620、p=0.00041、12/12 paired blocks 正；约 69% 旧增益消失是描述性拆分，不是中介率。 |
| A8 | E1 / E3 objective contrast | ML r50 六 cell selected AUC 的 Equal→Sample 差约 +0.021085±0.009768（12 blocks）；E1 中约 98.5% mismatch 削减来自系数的数学效应，但 routing 未排除；E3 端点 bit-exact 复现 E1，响应面随 alpha 增加总体改善、无稳定 interior 胜者。E1 机制测量仍是观察性，E2 未执行，不能独立区分系数和 sampling exposure。 |
| A9 | E4 KuaiRand split shift | Equal 下 dominant d1 两任务均有负 transfer，Sample 改善 d1 与全局指标但 d1 仍约 −0.006；train-test mixture TV=0.112839，d1 click 正例率约 0.378→0.211。shift 是强竞争解释。 |
| A10 | Curriculum / schedule / adaptive | P1-2 forward curriculum B0−A0 为 +0.000839±0.001018、p=0.198；E5 中 Forward 相对 static Sample 的 fixed-final 差仅 1/4 seeds 为正，GO gate 9 项仅 3 项通过；E6 calibration 可压低 mismatch 但没有稳定胜过 Sample，EMA 非必要且中间域触发预设安全限制；E7 因 STOP 为 0 runs。结论仅覆盖已测路线。 |
| A11 | S0 Momentum matched control | Sample 下 comp1 相对 C_val 的 selected macro AUC 差 −0.003033±0.000815，0/4 seed 为正；残余可复现，但输给验证集可选 uniform-smoothing family。 |
| A12 | S1 Sample + SD reference | 72 组均值 63 正、9 负；0/72 达到 stable practical NT；最负候选 −0.002923，幅度被 reference seed 方差覆盖，未跨条件 / common-stage 稳定。 |
| A13 | S2 Equal / Sample selected cells | Equal−SD 中 8/24 组 ΔAUC≤−0.003，完整严格组合 gate 3/24；Sample−SD 为 0/24 达阈值。8 个 Equal 受损组均为 Large 角色。 |
| A14 | S2 KuaiRand tab1 click | Equal−SD selected AUC −0.045006±0.000602，4/4 seeds 同向、5/5 common epochs 为负；为最强条件性靶点。 |
| A15 | S2 MovieLens d2_large rating>3 | Equal−SD selected AUC −0.018093±0.004687，4/4 seeds 为负；部分后期 / common-final 视图缩小或反转，限制机制外推。 |
| A16 | S2 指标和阶段差异 | 7 个 Equal−SD 组出现 AUC 为正而 LogLoss 变差；selected 不能替代 late / common-final，KuaiRand tab1 的退化则随训练加深更稳定。 |
| A17 | S3 KuaiRand Equal/tab1 target contrast | target-only−FULL selected AUC：click +0.038876±0.003950、long-view +0.039576±0.004107，4/4 seeds 同向；Sample 下相同干预约 +0.0003 / +0.0001；Equal−Sample 的目标差距缩小约 89% / 92%，仍有残余。 |
| A18 | S3 非目标域代价 | Equal 条件下 target-only−FULL 非目标域 AUC 平均 −0.017040；16 个 task-seed 中 14 个为负。 |
| A19 | S4 magnitude controls | d1 的 norm-matched N−FULL 为 −0.004048 / −0.004873；applied-step matched U−FULL 为 −0.000587 / −0.000942，均未复现 rescue。 |
| A20 | S4 target rescue replication | target-only 相对 N / U：click +0.042924 / +0.044448，long-view +0.039463 / +0.040517；4 seeds 与 5 个 common epochs 稳定。 |
| A21 | S4 non-target 与 training stage | d0 selected AUC 约 −0.031386 / −0.032142；d2 在 late checkpoint 转正，约 +0.019810 / +0.036160。 |
| A22 | P7 backbone transfer | MovieLens D=3 下 comp1−none：MMoE +0.002978、PLE +0.005538、ShareBottom +0.004012；仅适用于所测结构，ShareBottom 为本地适配，不能当 S3/S4 泛化证据。 |
| A23 | S5 history×current 拆分 | selected ΔAUC（4 seeds）：d1 的 R_T=+0.0414/+0.0423（4/4）、R_H=+0.0411/+0.0405（4/4，约 96–99% R_T）、R_D=+0.0009/+0.0008（约 2%，各 checkpoint |·|≤0.0027）；d0 的 T−F=−0.0273/−0.0374（0/4）、HT−F=−0.0067/−0.0064（2/4）、DH−F=+0.0077/+0.0114；d2 的 HT−F selected +0.0150/+0.0192（4/4）、final +0.0459/+0.0535（4/4）。反例：T seed2024 final d1 塌陷（T−F=−0.0169）；HT seed2024 selected d0 −0.038/−0.030；moment gate 15/16。 |
| A24 | S6 Sample-objective residual gate | KuaiRand d1 selected T−F −0.000657±0.004359/−0.000790±0.005074（1/4 正）；MovieLens d2 +0.001899±0.004733/+0.002555±0.005244（2/4 正）；四项均低于 +0.003 且 checkpoint/metric 不稳。T 仍损伤 non-target；MovieLens final macro +0.0205 来自 F 过拟合。两数据集 STOP，未运行 HT/DH Phase B。结论限两个最大域 target 与本 M3OE/Sample 协议。 |

---

## 6. 溯源索引与交接

此处只提供定位路径，不重述结论。应先读最新签署的 Reviewer 与 analysis，再查 paired / aggregate CSV，最后查 run-level 原始数据和日志；源报告或本摘要存在冲突且无正式裁定时保留 OPEN。

| 关联 Claim / Anchor | 优先源报告与数据 |
|---|---|
| C1 / A1–A2 | P1_diagnosis.md；P2_baselines.md；P0_repro.csv；results/p2/P2_baselines.csv |
| C2 / A3 | P1_diagnosis.md；results/p1/D04_grad_stats.csv |
| C3 / A4–A5 | ML_P5_REPORT.md；KUAI_STAGE1_ORACLE.md；KUAI_P5_MAIN_REPORT.md；FINAL_RESEARCH_REPORT.md；results/p5/P5_raw.csv；P5_oracle.csv |
| C4–C5 / A6–A8 | P0_1_STAGE_REVIEW.md 至 P0_5_STAGE_REVIEW.md；P1_2_STAGE_REVIEW.md；stage_E1_analysis.md；stage_E3_analysis.md；对应 P0 / E1 / E3 paired 与 summary 文件 |
| C5 / A9 | stage_E4_dataset_audit.md；stage_E4_analysis.md；stage_E4 raw / paired / summary 与 split audit manifest |
| C6 / A10 | P1_2_STAGE_REVIEW.md；stage_E5_go_no_go.md；stage_E6_go_no_go.md；stage_E7_gate_stop.md |
| C7 / A11 | stage_S0_review.md；stage_S0_analysis.md；stage_S0 paired / aggregate / subgroup summaries |
| C8 / A12 | stage_S1_review.md；stage_S1_analysis.md；stage_S1_paired_aggregates.csv；common-epoch 与 reference-variation summaries |
| C9 / A13–A16 | stage_S2_review.md；stage_S2_analysis.md；stage_S2_raw_results.csv；stage_S2_paired_results.csv；training-stage / rotation summaries |
| C10 / A17–A18 | stage_S3_review.md；stage_S3_amendment_review.md；stage_S3_analysis.md；stage_S3_aggregates.csv；interactions / paired results |
| C11 / A19–A21 | stage_S4_A2_review.md；stage_S4_A2_execution_report.md；stage_S4_analysis.md；stage_S4_A2_outputs / paired results / gradient steps / probe / manifest |
| C12 / A22 | P7_generalization.md；results/p7/P7_backbones.csv；results/P7_summary.csv |
| C13 / A23 | stage_S5_review.md（含 moment-gate investigation）；stage_S5_run_report.md；stage_S5_analysis.md；stage_S5_outputs / paired / aggregates / moment_audit / manifest |
| C14 / A24 | stage_S6_design.md；stage_S6_analysis.md；stage_S6_phaseA_analysis.md（MovieLens supplemental，historical scope）；stage_S6_outputs / paired / aggregates / config_lock_audit / manifest；stage_S6ML_outputs / paired / aggregates / manifest；两个 Phase A run reports |

当前研究停止点见 C14：Equal 下的 history-mediated rescue 已限定为 objective-contingent；Sample 下两个数据集均未通过 practical gate，Phase B 未触发。不能把 no practical residual 写成 exact zero，也不能把 MovieLens final 的全域 gain 写成 target rescue。本文仅作上下文，不授权自动实现或运行；如未来重开须作为新阶段设计并遵守 §4.5 的阶段门槛。



