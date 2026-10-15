# R0 旧实验证据审计（Experimental Evidence Audit）

本文件是 Stage R0 交付物，整理自 `research_R0/work/exp_evidence_review.md`（实验证据审查报告，2026-10-10），并以 `experiments/` 目录下 8 份汇总档案（截止口径 2026-10-09）作为交叉核对依据：**Experimental_Global_Summary.md、Experimental_Stable_Findings.md、Experimental_Evidence_Ledger.md、Experimental_Dead_Ends.md、Experimental_Hypothesis_History.md、Experimental_Open_Questions.md、Experimental_Contradiction_Register.md、Experimental_Experiment_Index.md**。

**重要限制**：本工作区内不存在 S0–S8 的原始阶段报告与日志（`stage_*.md`、`exp_sn.md`、`stage_S6_R2_*.csv` 等）。所有数值来自上述 8 份汇总档案的转引，无法二次核验原始 CSV 或 stage 报告；无法二次核验的条目已在各节内显式标注。本审计只重建实验侧证据，不做文献侧判断，不做 AO 方向的最终判决。

---

## 1. 实验总览表

阶段 × 数据集 × objective × 干预对象 × 状态。来源：`Experimental_Experiment_Index.md`（EXP001–EXP032）。

| ID | 阶段 | 数据集 | Objective | 干预/对照对象 | 状态 |
|---|---|---|---|---|---|
| EXP001 | P1 诊断（D0.2/0.4/0.6/0.7） | MovieLens-1M | as-published（旧） | 共享梯度 SNR/cosine 诊断（无训练干预） | 完成（5 seeds） |
| EXP002 | P2 baseline/AutoML/optimizer 修复 | MovieLens-1M | 旧 | 架构权重固定、B10 修复 | 完成（5 seeds） |
| EXP003 | P5 imbalance-ratio sweep | MovieLens-1M | 旧 | baseline/comp1；ratio 0–100（30/100 同时改总 N） | 完成（3 或 5 seeds） |
| EXP004 | P5 主矩阵 + oracle | KuaiRand-Pure | 旧 | baseline/fixed073/comp1/comp2 | 完成（5 seeds） |
| EXP005 | P0-1 固定 N 角色轮换 | MovieLens | 旧 | 物理域 × 大小角色轮换 | 完成（4 seeds） |
| EXP006 | P0-2 固定 N ratio sweep | MovieLens | 旧 | ratio × 角色 × 方法 | 完成（4 seeds） |
| EXP007 | P0-3 跨骨干 | MovieLens | 旧 | M3oE vs MMoE | 完成（4 seeds） |
| EXP008 | P0-4 训练权重 × 选点目标 | MovieLens | Equal/Sample/sqrt × Equal/Sample 选点 | loss weighting × checkpoint objective | 完成（4 seeds） |
| EXP009 | P0-5 comp1 × weighting 2×2 | MovieLens | Equal/Sample | comp1 on/off | 完成（12 paired blocks） |
| EXP010 | P1-2 curriculum | MovieLens | 旧 | static/forward/reverse curriculum | 完成（12 配对单元） |
| EXP011 | E1 objective contrast | MovieLens r50 | Equal vs Sample | objective 对照（系数与 exposure 未分离） | 完成 |
| EXP012 | E2 机制拆分 | — | — | 拟区分 loss coefficient 与 sampling exposure | **未运行（0 runs）** |
| EXP013 | E3 α response | MovieLens r50 | Equal/Sample 混合系数 α 扫描 | objective mixing | 完成（4 seeds） |
| EXP014 | E4 objective + split audit | KuaiRand-Pure | Equal/Sample | objective 对照 + train/test mixture 审计 | 完成（4 seeds） |
| EXP015 | E5 schedule/curriculum | KuaiRand-Pure | vs static Sample | Forward schedule | 完成；GO gate 未通过 |
| EXP016 | E6 adaptive calibration/EMA | KuaiRand-Pure | Adaptive vs Sample/Equal | 自适应系数 + EMA | 完成（24 runs/4 seeds）；STOP |
| EXP017 | E7 generalization gate | — | — | E6 STOP 后门控 | **未运行（0 runs，gate 停止）** |
| EXP018 | P6 组件消融 | MovieLens ml_r30 | 旧 | 17 arms（动量/课程/EMA 等） | 完成（85 runs/5 seeds） |
| EXP019 | P7 骨干迁移 | MovieLens | 旧 | comp1 迁移到 MMoE/PLE/STAR/ShareBottom | 完成（STAR 因复现差距排除） |
| EXP020 | S0 momentum 匹配对照 | MovieLens（强不均衡） | **Sample** | comp1 vs Adam vs uniform EMA（匹配平滑） | 完成（84 runs/4 seeds×3 轮换） |
| EXP021 | S1 joint vs SD reference | MovieLens + KuaiRand | **Sample** | 3 backbones（M3OE/MMoE/PLE）× joint/SD | 完成（192 runs/4 seeds） |
| EXP022 | S2 Equal/Sample vs SD | KuaiRand natural + MovieLens d2_large | **Equal + Sample** | objective × SD reference | 完成（4 seeds） |
| EXP023 | S3 objective × shared-access | KuaiRand natural + MovieLens d2_large | Equal + Sample | FULL vs TARGET_ONLY_SHARED | 完成（24 新+8 复用/4 seeds） |
| EXP024 | S4 幅度对照 | KuaiRand natural | **Equal** | F/T/N/U（N=raw-grad norm，U=step norm 匹配） | 完成（16 runs/4 seeds） |
| EXP025 | S5 history×current | KuaiRand natural | **Equal** | F/T/HT/DH（Adam history source × current source） | 完成（16 runs/4 seeds；moment gate 15/16） |
| EXP026 | S5-ML 跨数据集复现 | MovieLens d2_large | **Equal** | 同 S5 四臂 | 完成（16 runs/4 seeds；gate 8/8）；**注意：exp_sn 漏登，索引补登（C018）** |
| EXP027 | S6 Phase A | KuaiRand natural + MovieLens d2_large | **Sample** | F/T 两臂 | 完成（16 runs/4 seeds×2 数据集）；双 STOP |
| EXP028 | S6 Phase B | 同上 | **Sample** | F/T/HT/DH 2×2 | 完成（16 新 runs）；ML M1 PASS，**KR M1 FAIL**（C014） |
| EXP029 | S7-A 梯度探针 | KuaiRand + MovieLens | Sample | Domain×Task 梯度 probe（非训练干预） | 完成（8 probes/4 seeds×2） |
| EXP030 | S7-B same-θ 短程续训 | KuaiRand + MovieLens | Sample | F vs TARGET-current，anchor e9/e39；private fresh-Adam 近似 | 完成（16 runs/4 seeds×2 anchors） |
| EXP031 | S8 full-state 续训 kill test | KuaiRand + MovieLens | Sample | F/T/DH 分支 × selected/middle/late anchors | 完成（8 base+24 continuation/4 seeds） |
| EXP032 | **S6-R2** | KuaiRand-Pure（MovieLens 为 negative control） | Sample | R0–R4 重选、A-grid sensitivity、val 分解、window 分类、manifest 参数交换 | **档案登记为「已完成的 evaluation-only」**（复用 16 个既有 F/DH checkpoint，无新增训练）；源报告 KuaiRand gate 判 scoped GO。**与 Stage R0 任务书口径矛盾，详见第 5 节** |

完整性核对结论：
- S0–S8、S5-ML、S6 Phase A/B、S6-R2、S7-A/B、S8 **均有登记**，无缺失阶段。
- 标记未运行/门控的条目：**E2（EXP012，0 runs，未实施）、E7（EXP017，0 runs，因 E6 STOP 门控未启动）**。二者不产生任何性能证据。
- 派生分析（S6 DH 轨迹重分析两份）复用 EXP028 训练数据，索引明确标注**非独立复现**；P3/P4 为数据/实现审计，不计实验。
- 缺口说明：汇总引用的原始阶段报告（`stage_*.md` 等）与 S6-R2 原始 CSV 在本工作区不可见，数值无法二次核验；`exp_sn.md`（2026-10-06 快照）本身不可见，其陈旧结论仅以 Contradiction Register 的记录为准。

---

## 2. 稳定结论清单

在 SF001–SF012 基础上重组。**每条结论的 objective/dataset 边界是外推判断的关键，务必逐条核对。** 强度沿用档案口径（STRONG/MEDIUM/WEAK），来源均为 `Experimental_Stable_Findings.md`，证据 ID 与数值以 `Experimental_Evidence_Ledger.md` 为准。

### SF001 — 已测 Sample + SD reference 下无普遍 practical residual NT【STRONG】
- 结论：S1 Sample joint 整体优于 single-domain reference；72 个汇总单元 0 个达 stable practical NT 门槛（288 paired seed units 中 63 正 9 负，最负候选约 −0.002923）。
- 证据：E009、E018。Seed：4。
- 边界：**仅 Sample objective**；已测自然比例；MovieLens+KuaiRand；M3OE/MMoE/PLE 三骨干；S1 的 reference 与 selected checkpoint 规则。
- 例外：单元级仍有小负差；不覆盖 Equal、未测极端 ratio、未测 target 域（OQ001）。

### SF002 — Equal 与 Sample 的 selected 域级表现不同【STRONG】
- 结论：Equal 在 S2 特定 Large-role/target cells 有可重复 selected loss（Equal−SD 8/24 组 ≤ −0.003，严格 gate 3/24，全部位于 Large role；KuaiRand tab1 click −0.045006 ± 0.000602，4/4 同向、5/5 common epochs 负；MovieLens d2 −0.018093 ± 0.004687，4/4）；Sample−SD 为 **0/24**。
- 证据：E004、E010。Seed：4。
- 边界：S2 的 KuaiRand natural / MovieLens d2_large、M3OE、SD reference、selected six-cell macro 规则。
- 例外：MovieLens late/common 视图可能减小或反转差异；不证明 Sample 普遍更优。

### SF003 — KuaiRand Equal/tab1 的 shared-access 干预有条件性 target 效果【STRONG】
- 结论：S3 TARGET_ONLY_SHARED−FULL 在 selected target 上 click/long_view AUC +0.038876 ± 0.003950 / +0.039576 ± 0.004107（**4/4 同向**）；interaction +0.038579/+0.039440（4/4）。**同一配对结构下 Sample 干预效应仅约 +0.0003/+0.0001（近零）**。
- 证据：E011。Seed：4（8 个严格配对 blocks 覆盖两数据集×两 objective）。
- 边界：**仅 KuaiRand natural / M3OE / Equal / target=d1/tab1 / selected endpoint**。这是 intervention effect 的因果识别，**不证明 gradient-conflict 几何机制**。
- 例外：MovieLens 复现为 DATASET_SPECIFIC（E012，MEDIUM：d2 target +0.019/+0.014，4/4，但 non-target 有代价、late checkpoint 有例外）。

### SF004 — 两类幅度匹配不能解释该 rescue【STRONG】
- 结论：S4 中 raw-gradient-norm 匹配（N−F −0.004048/−0.004873，0/4 正）与 applied-step-norm 匹配（U−F −0.000587/−0.000942，2/4 正）均未复制 T−F（+0.038876/+0.039576，4/4，5/5 common epochs 正）。
- 证据：E013。Seed：4。
- 边界：KuaiRand natural / M3OE / Equal / S4 A2 四臂。
- 例外：仅排除这两种充分解释，不确认其他机制。

### SF005 — Equal KuaiRand d1 rescue 跟随完整 persistent Adam history【STRONG】
- 结论：S5 中 T−F +0.041426/+0.042254（4/4）；HT−F +0.041085/+0.040528（4/4，恢复约 **96–99%**）；DH−F 仅 +0.000875/+0.000821（约 2%）。
- 证据：E015。Seed：2022–2025（4）。
- 边界：**仅 KuaiRand-Pure natural / M3OE / Equal / d1/tab1 / selected**。
- 例外：**moment gate 15/16**（seed2024/epoch1/m/HT 限定）；**不能拆成 m、v 或 bias-correction 单独机制；不能外推到 Sample**（OQ003）。

### SF006 — rescue/history 结构随 objective、dataset、checkpoint 改变【MEDIUM】
- 结论：Equal 下 selected rescue 在 S5/S5-ML 可观察；**Sample 下 S6 Phase A/B 所有四臂 selected target contrasts 均未达 +0.003 practical gate**；Sample MovieLens 仍有 late DH trajectory effect。
- 证据：E017–E021。Seed：4/数据集。
- 边界：两数据集、M3OE、Equal/Sample 阶段对照；Sample Phase B KuaiRand 保留 **M1 gate FAIL** 限定（机制识别不能作无条件 claim）。
- 例外：「below threshold」≠ 严格零；MovieLens late fixed-checkpoint 结果不能代替 selected gate。

### SF007 — S8 未证明正常 selected model 的 practical optimizer-policy gain【STRONG】
- 结论：selected-anchor h=2 continuation 的 T−F / DH−F best-AUC 差：MovieLens 均 +0.000062；KuaiRand −0.000445 / +0.000068；均远低于 +0.003。**所有 MovieLens continuation best 均低于该 seed 冻结的原 F best**。
- 证据：E026、E027。Seed：4/数据集（8 base + 24 continuations）。
- 边界：S8 完整 state、selected/middle/late anchors、h=2-epoch primary horizon、Sample/M3OE。
- 例外：MovieLens late-anchor macro final-step 可正（+0.010732，4/4）但 **seed-skewed**（约 89% 来自 seed 2024/2025）、target gain 小（best +0.000441），且未形成更高 best——属 overfit-anchor rescue，非方法收益。

### SF008 — MovieLens late gradient utility 与 overfit decline 同步，但非充分性能机制【MEDIUM】
- 结论：S7-A MovieLens late U_k 与 target validation 恶化同步（early→late target AUC −0.0208，4/4）；**KuaiRand 负 U_k snapshot 没有相同后果（target AUC 持续上升至 e49，4/4）**。S7-B 同 θ 改变 late utility 后短程 validation 未稳定改善：anchor39 step5 target AUC 均值 −0.001226、**仅 3/4 seeds 为负**（seed2025 为 +0.002359），LogLoss 均值 +0.011130、3/4 变差。
- 证据：E023–E025。Seed：4/数据集。
- 边界：Sample/M3OE、所测 anchors/checkpoints；S7-B 用短窗 + fresh-Adam private-state 近似。
- 例外：口径修正后只能说「未见稳定收益」，不能说逐 seed 一致恶化（C012）；S8 full-state 仍不支持 selected-best gain。

### SF009 — 现有数据未识别 domain size 的独立因果效应【STRONG（关于「尚未识别」）】
- 结论：早期 large-role 关联存在（P0-1 L-role Δ −0.014557 ± 0.007439；P0-2 ratio=50 L-role −0.015545 ± 0.010894），但后续对照**从未独立操纵 domain size**；目标权重、域身份、难度、样本多样性、曝光、checkpoint 均共同变化。MovieLens 实际样本量 **d2=150,000、d0=34,500、d1=3,000（d1 最小、d0 居中）**，旧「d0 最小」标签为事实错误（C015）。
- 证据：E002–E004、E010、E014、E020。
- 边界：跨 P0–S6 实验矩阵。对 size 本身效应强弱**不作判断**（OQ002）。

### SF010 — 必须同时报告 target/non-target、AUC/LogLoss 与 checkpoint【STRONG（报告口径结论）】
- 结论：S4–S8 多处 aggregate gain 与 subgroup/target metric、selected 与 final/common checkpoint 方向不同；单一 overall metric 会掩盖 trade-off。典型：S4 中 T rescue 伴随 d0 selected 代价 −0.0314/−0.0321，而 d2 final 转为 +0.019810/+0.036160（4/4）（E014）；S6 MovieLens final macro 正但 target selected 低于门槛且 F 严重过拟合（C009）。
- 证据：E014、E016–E020、E022、E027–E032。

### SF011 —（S6-R2 相关，按任务书原则**隔离使用**）KuaiRand DH test selected non-target gain 不出现在 validation【MEDIUM】
- 结论：Sample/M3OE、canonical R0-selected KuaiRand checkpoints：test Δ(DH−F) ΔN **+0.0068 ± 0.0046（4/4 正，3/4 ≥ +0.005）**、ΔM +0.0044 ± 0.0029（4/4 正）、ΔT −0.0004 ± 0.0031；同一 checkpoint 对的 **validation ΔN = −0.0039 ± 0.0081（仅 1/4 正）**、ΔM −0.0027。R1/R3/R4 的正向仅在 A-grid sensitivity（ΔN +0.0041/+0.0060/+0.0041，各 3/4 正），exact 重选 test 可得性 0/4、0/4、1/4。
- 证据：E028、E029。Seed：2022–2025（4 paired）。
- 例外：seed2023 target ΔT = −0.0047；seed2025 test non-target 近零；不识别机制；**MovieLens selected contrast 近零（E030，negative control）**。

### SF012 —（S6-R2 相关，**隔离使用**）KuaiRand 跨分区参数交换崩溃，未定位增益来源【MEDIUM】
- 结论：四 seeds 下 C/D hybrids test macro 0.5450–0.5842 / 0.5196–0.5791，远低于 F_full 0.6730 / DH_full 0.6774；E_hybrid≈C_hybrid；F_hybrid 与 F_full 差 ≤0.00011 macro。
- 证据：E032。Seed：4。边界：仅 KuaiRand、Sample/M3OE、一种 manifest 分区（shared/architecture/private）、evaluation-only。
- 例外：seed2025 分母近零（~1.1e−5），retention ratio 不可解释；结果与跨分区 co-adaptation 一致，但**不证明 co-adaptation 是 gain 的原因，也不定位增益分区**。

### 补充：未列入 SF 但档案中强度较高的证据
- **E006【STRONG】**：已测 curriculum/schedule/adaptive calibration（P1-2、E5、E6）均未通过各自对 static Sample 的预设 gate；E7 未运行。边界：仅所测实现与 gate，不覆盖所有 schedule/controller。
- **E008【STRONG】**：S0 中 comp1 对 sample Adam 为 +0.005307 ± 0.000523，但对 validation-selected uniform EMA 为 **−0.003033 ± 0.000815（3 轮换均值后逐 seed 均为 uniform EMA 更好）**；边界：MovieLens 强不均衡/Sample/M3OE/4 seeds。

---

## 3. 修订链清单

「早期结论 → 修订实验 → 当前口径」，附 Contradiction Register 编号。原则：以后续更严格、能明确修订先前结论的实验为准。来源：`Experimental_Contradiction_Register.md` 与 `Experimental_Hypothesis_History.md`。

| # | 早期结论（出处） | 修订实验 | 当前口径 | C 编号 |
|---|---|---|---|---|
| R1 | 「大域牺牲是 M3oE 特有」（早期 P5/MovieLens） | P0-3 加入 MMoE（EXP007） | 非 M3oE 特有，MMoE 同向（L-role 约 −0.0116）；H001 SUPERSEDED | C003 |
| R2 | 「domain size 是退化主因」 | P0 轮换、S2/S4/S5/S6 多阶段 | size 从未被独立操纵，只能是调节/代理变量；H002 WEAKENED | C003 |
| R3 | 「小域梯度 SNR 必然低，需要 per-domain momentum」 | P1 双 batch 测量 + P6 SNR-derived γ + S0 匹配对照 | 已测口径下不成立（batch=4096 时 SNR 差消失并反向）；H003 FALSIFIED（限已测口径） | C002 |
| R4 | 「comp1 有独有 domain-conditioned momentum 优势」 | S0 匹配 uniform-EMA 对照（EXP020） | 残余效应存在但被统一平滑覆盖，无独立优势；H004 FALSIFIED | C001、C002 |
| R5 | 「comp1 收益随 ratio 单调增长」 | P0-2 固定 N 复核（EXP006） | 非单调，高 ratio 与总 N 混杂；H005 FALSIFIED | —（E002 SUPERSEDED） |
| R6 | 「curriculum/schedule/adaptive calibration 胜过 Sample」 | P1-2、E5、E6（E7 门控未启动） | 已测实现均未过各自 gate；H008/H009 FALSIFIED（限所测实现） | — |
| R7 | 「Sample 下存在普遍 practical NT（Safe-MTL 主线前提）」 | S1（0/72）+ S2 Sample（0/24）+ S6 Phase A 双 STOP | 已测协议内被否定；极端 ratio 仍 OPEN；H007 FALSIFIED（限已测协议） | C004 |
| R8 | 「S3 rescue 即 gradient-conflict 几何机制」 | S4 幅度对照失败 → S5 定位 history → S7/S8 | 纯几何解释被削弱为 WEAKENED；H011、H012（幅度充分性 FALSIFIED） | C006 |
| R9 | 「S5 KuaiRand DH≈F 说明 current-source 贡献普遍为零」 | S5-ML（EXP026） | MovieLens Equal 下 DH 非零（selected +0.005859/+0.003583，2–3/4），late 更大；模式仅部分跨数据集复现；H014 PARTIALLY_SUPPORTED | C007 |
| R10 | 「换 Sample 后 optimizer/history effect 消失」（exp_sn 2026-10-06 口径） | S6 Phase B、S8、S6-R2 | 过宽：仅 selected practical target gate 未通过成立；MovieLens late DH 轨迹效应等仍存在；E021 SUPERSEDED、H015 FALSIFIED（限 selected practical rescue claim） | C019（+S6-R2 update） |
| R11 | 「S6 Phase B 未触发」（exp_sn 快照） | Phase B 扩展 2026-10-08 完成 | 快照过期；16 新 runs 已完成，属诊断性扩展非确认性 | C016 |
| R12 | exp_sn 未含 S5-ML | 索引补登 EXP026 / E017 | S5-ML 是独立 16-run Equal 复现，**S6ML ≠ S5ML** | C018 |
| R13 | 「MovieLens d0 是最小域受害者」 | S6 样本量复核 | d1=3,000 最小、d0=34,500 居中、d2=150,000；事实标签错误已更正 | C015 |
| R14 | 「S6/S6 轨迹重分析的 late gain 代表方法收益」 | S8 full-state continuation（EXP031） | 收窄为 overfit-trajectory rescue（「少恶化」）；所有 continuation best 低于原 F best；E022 SUPERSEDED | C010、C017 |
| R15 | 「S8 late macro rescue = best performance improvement」 | S8 best-vs-original-best endpoint | 89% 来自 seed 2024/2025；target best gain +0.000441；只允许「partial late-anchor rescue」 | C017 |
| R16 | 「S7-B anchor39 step5 target AUC 4/4 seeds 为负」 | 原始 CSV 复核（2026-10-09） | 均值 −0.001226、**3/4 为负**（seed2025 +0.002359）；「未见稳定改善」仍成立，逐 seed 一致性降级 | C012 |
| R17 | 「负 U_k 是充分性能退化原因」 | KuaiRand 反例 + S7-B same-θ + S8 | WEAKENED：条件性局部一阶统计，负值不充分；H018 | C011、C012 |
| R18 | 「S7-B 短窗近似结论可代表 full-state 续训」 | S8 完整状态恢复 | S8 是更强的后续检验：仍无 selected/best gain；S7-B 局限不外推 | C013 |
| R19 | 「S6 Phase B 16/16 结构完成 = 机制识别通过」 | 独立审查 | 执行有效但 **KuaiRand M1 gate FAIL**（m-conjunction 仅 4/8）；结构有效 ≠ 机制识别通过 | C014 |
| R20 | 「S6-R2 KuaiRand test gain 只是选择规则/单点尖峰 artifact」 | S6-R2 Part A/C 多规则与 window 分析 | 部分削弱（A-grid 多规则为正、DH window 3/4 moderate），但 exact 重选缺失、seed 例外保留；H021 WEAKENED | （S6-R2 专项，见第 5 节） |
| R21 | 「参数交换可定位 DH gain 来源分区」 | S6-R2 Part D | 双向 cross-partition hybrid 均崩溃，无法定位；H022 WEAKENED | C023 |
| R22 | 「checkpoint timing 完全解释 KuaiRand test gap」 | S6-R2 Part B 分解 | validation 层 timing 贡献大（TimingAlongF = −0.0120），但 same-epoch test 16/20 点为正，test 层归因未决；H023 OPEN | C021 |

---

## 4. 死路清单

`Experimental_Dead_Ends.md` 共 12 条（D001–D012），压缩表（「什么被否定 / 在什么条件下」）：

| # | 被否定的方向 | 一句话：否定内容与条件 |
|---|---|---|
| D001 | per-domain momentum（基于小域低 SNR） | 已测 MovieLens 两种 batch 下小域低 SNR 不稳定（E001），且 comp1 未胜过匹配 uniform EMA（E008）；限已测口径 CLOSED |
| D002 | M3oE 特有大域牺牲 | MMoE 在 P0-3 同向退化（E003），非 M3oE 特有；SUPERSEDED |
| D003 | comp1 必要且收益随 ratio 单调 | ratio 响应非单调且与总 N 混杂（E002），改 objective 即减轻 Large-cell loss（E004），S0 否定机制独特性（E008）；CLOSED |
| D004 | Sample 下普遍 practical NT 需 Safe-MTL 主线 | S1 0/72、S2 Sample 0/24、S6 Phase A 双 STOP（E009/E018）；限已测 Sample 设置 CLOSED，极端 ratio 保留 OQ001 |
| D005 | curriculum/已测 schedule 胜过 static Sample | P1-2 差异噪声级、E5 未过 GO、E7 未运行（E006）；限已测实现 CLOSED |
| D006 | E6 adaptive calibration/EMA 是已验证主方法 | 未过 Sample 的 performance/safety gate（E006）；CLOSED/STOP |
| D007 | 幅度匹配足以复制 target rescue | S4 的 N/U 两臂均未复制（E013）；作为充分解释 FALSIFIED |
| D008 | Equal 的 HT/DH 结果可直接当 Sample 方法依据 | S6 selected Sample contrasts 全低于 practical gate（E018/E019）；作为跨 objective 推断 CLOSED |
| D009 | target-only/HT 是无 non-target 代价的安全部署方法 | E014/E016/S5ML 显示 non-target 代价与 checkpoint/数据集 trade-off；作为「已验证安全方法」CLOSED（仅保留诊断臂身份） |
| D010 | universal gradient-conflict/U_k 修正研究线 | KuaiRand 负 U_k 反例（E024）、S7-B 同 θ 改 utility 无 validation 收益（E025）、S8 无 selected/best gain（E026/E027）；作为方法前提 CLOSED |
| D011 | 从 MovieLens late DH gain 推断通用 selected-performance gain | S8 所有 continuation best 低于原 F best（E026/E027）；CLOSED（不作普遍 null claim） |
| D012 | 「域大小」作为现成解释/结论 | size 从未独立操纵，MovieLens 大小标签曾标错（C015）；作为独立因果结论 CLOSED，OQ002 保留 |

---

## 5. S6-R2 专项

### 5.1 状态矛盾记录

- **任务书口径（Stage R0）**：「尚未完成的 S6-R2 不作为任何方向的支持或反证」——即任务书认为 S6-R2 **未完成**。
- **档案口径（`experiments/`，2026-10-09）**：
  - Experiment Index EXP032 明确登记 S6-R2 为**「已完成的 evaluation-only」**条目：复用 16 个既有 F/DH checkpoint（4 seeds × 2 数据集），无新增训练；含 R0–R4 重选、A-grid sensitivity、validation 分解、window 分类、KuaiRand manifest 参数交换（Part A–D）；源报告 KuaiRand gate 判 **scoped GO**；integrity report 通过（state-dict parity、锚点重现；seed2025 retention 分母近零已限定）。
  - Global Summary Q1 第 3 条、Stable Findings SF011/SF012、Evidence Ledger E028–E032、Hypothesis History H021–H023、Contradiction Register C019-update/C021/C022/C023、Open Questions OQ009/OQ010 均把 S6-R2 作为**已完成证据**纳入。
  - 索引明确注明：S6-R2 在 exp_sn.md（2026-10-06）快照**之后**完成，故旧快照无此条目；这不是登记遗漏，而是时间顺序。
- **结论**：两个口径矛盾的真实来源是**任务书撰写时间早于（或未更新到）S6-R2 完成登记**。档案内部自洽且多处交叉登记（索引/证据/稳定结论/矛盾登记四处一致），档案口径可信度更高；但本审计不裁决任务书与档案谁对谁错，只如实记录矛盾。

### 5.2 处理原则（按任务书指令执行）

**无论档案是否登记为完成，S6-R2 的结果一律不作为任何 AO 方向的支持或反证。** 以下内容为「如果采纳它会意味着什么」的如实记录，仅供 Stage R0 识别哪些判断被隔离。

### 5.3 S6-R2 内容摘要（隔离，不用于支持/反证）

1. **KuaiRand test-side selected non-target gain（E028/E029，SF011）**：R0-selected DH−F test ΔN +0.0068（4/4 正）、ΔM +0.0044（4/4 正）、ΔT −0.0004；同一 checkpoint 对 validation ΔN −0.0039（1/4 正）——**validation/test 方向反转未解（C021，OQ009）**。R1/R3/R4 仅 A-grid sensitivity 支持，exact 重选 test 基本缺失；seed2025 近零、seed2023 target ΔT −0.0047。
2. **MovieLens negative control（E030）**：selected test ΔN ≈ 0（2/4 正），晚期 saved-grid e19 可达 +0.0334 但与 canonical selected epoch（7–13）不重合。
3. **Window 分析（E031，WEAK）**：DH 3/4 moderate plateau vs F 3/4 narrow spike；单数据集单规则，TENTATIVE。
4. **参数交换（E032，SF012）**：跨分区 hybrid 全面崩溃（test macro 降至 0.52–0.58 vs parent 0.67+），所测分区不可互换、与 co-adaptation 一致，但**未定位 intact DH gain 来源**（OQ010，H022 WEAKENED）。
5. **对早期结论的修订作用（若采纳）**：它是 C019「Sample 后 history effect 全消失」过宽口径的最新反例之一（test-side selected gain 存在）；它收窄了 H017（允许 claim 为 KuaiRand-specific test-side observation）；它削弱了「gain 纯属选择规则 artifact」（H021）但未能完全排除。

### 5.4 「不用于支持/反证」原则下对 Stage R0 的影响

在 S6-R2 隔离使用的前提下，以下判断**不能依赖 S6-R2 证据**，只能依靠其余档案：

- 「Sample 下 DH/optimizer-policy 在 selected checkpoint 有 test-side non-target 收益」——**无可用证据**（E028 被隔离；S8 的 selected-anchor continuation 为 null，S8 才是可用的最近证据）。
- 「validation/test 反转是普遍现象或已解释」——无证据；OQ009 仍 OPEN 但其唯一证据来源被隔离。
- 「DH gain 可由参数分区定位 / 分区不可互换」——E032 被隔离；参数级 co-adaptation 与不可组合性在可用证据中**无登记**。
- 「Sample 下所有 optimizer/history effect 为零」这一过宽说法——注意：即使不用 S6-R2，S6 Phase B 的 MovieLens late DH 轨迹效应（E020）与 S8 late rescue（E027）已足以否定该说法（C019 的主体修订不依赖 S6-R2）。
- S8 的 MovieLens endpoint 结论（SF007）、S6 Phase A/B 的 selected gate STOP（E018/E019）均**不依赖 S6-R2**，隔离后依然成立。

---

## 6. 证据边界声明（本审计对 AO 核对"不能回答什么"）

以下逐条对照 Evidence Ledger / Experiment Index / Open Questions 核实后列出。**这是 Stage R0 判决规则「不得以缺乏历史数据判 CONTRADICTED」的操作基础**：落入本清单的 AO 相关主张，实验侧一律无支持也无反证。

### 6.1 训练目标与干预形式从未测过

1. **没有测过任何逐格（per-cell）AUC 约束式 / 排序指标约束式训练目标**。历史上的 objective 仅有：as-published 旧目标、Equal、Sample、sqrt 加权（P0-4）、α 混合系数扫描（E3）。所有机制实验的性能指标是 AUC/LogLoss，但训练目标本身从未以逐格 AUC 约束形式构造。
2. **没有测过掩码（mask）类参数效率方法**：无结构化/非结构化稀疏掩码训练、无掩码参数效率测量。唯一涉及参数子集的操作是 S6-R2 的 manifest 参数交换评估（evaluation-only，且按本审计原则隔离），它不是掩码训练方法。
3. **没有测过 top-k 稀疏部署 / 稀疏化推理**：无任何部署形态（稀疏、剪枝、量化、蒸馏）实验；亦无延迟/吞吐/显存等部署指标。
4. **没有测过非均匀他域融合 / 学习到的跨域融合权重**：shared 参数上的干预只有「全开（FULL）vs 仅 target（TARGET_ONLY_SHARED）」二态，以及 history×current 源替换（F/T/HT/DH）。从未实验过连续可调的、逐域逐参数的融合系数或门控。
5. **没有测过 loss coefficient 与 sampling exposure 的独立分离**——E2（EXP012）正是为此设计但**未运行（0 runs）**；Equal/Sample 差异中「权重 vs 曝光」的贡献拆分在实验侧是空缺（OQ007 相关）。
6. **没有测过任何新算法相对于 Sample 之外的现代基线**：对照基线始终是 SD reference、static Sample、Equal 及内部诊断臂；E5/E6 的 schedule/adaptive 方法均已 STOP，且无后续新方法的训练实验。

### 6.2 已测结论无法外推的区域

7. **极端域比例下 Sample 是否有 minority-domain gap：未测**（OQ001；S1/S2/S6 只覆盖指定自然比例；SF001 的「0/72 无 practical NT」不覆盖未测 ratio）。
8. **domain size 的独立因果效应：从未独立操纵**（OQ002；SF009）。任何「按域大小分配干预强度」的设计在实验侧既无依据也无反证。
9. **Equal 的 access/history 机制跨 backbone 复现：未测**。S3–S8 机制矩阵全部在 M3OE；S1 虽测 MMoE/PLE，但只测 Sample 下 joint-vs-SD 的 NT 筛查，**没有**在 MMoE/PLE 上做过 access/history 干预。P7 是旧 comp1 组件迁移，非同一机制检验（OQ004）。
10. **非最大域 target / 小域 target 的干预效应：未测**。S3–S6 的 target 均为所测配置下的指定大域（KuaiRand d1/tab1、MovieLens d2_large）；（注意 MovieLens d1 才是最小域，但从未作为 target）。
11. **Equal 结论不能外推到 Sample；Sample 结论不能外推到 Equal**（SF002/SF003/SF006，C004/C005）。特别是：**Equal 下 d1 的 shared-access/history 退化与 rescue（SF003/SF005）在 Sample 下不存在同幅对应物**（S3 Sample arm 近零、S6 selected gates 全 STOP）；反向，Sample 下 MovieLens late DH 轨迹效应在 Equal 下无对应测量。
12. **KuaiRand 结论不能外推到 MovieLens，反之亦然**：S5 vs S5-ML 部分复现（C007）、S6 late 效应仅 MovieLens（E020）、S7 U_k 跨数据集反例（C011）、S8 late rescue 仅 MovieLens 且 KuaiRand late T 有害（E027）。两数据集同时改变域身份、比例、任务结构、收敛轨迹（OQ008）。
13. **Adam history 的子成分（m / v / step / bias-correction）各自作用：未识别**（OQ003；moment gate 15/16；S6 KuaiRand M1 FAIL）。任何声称特定优化器状态子成分机制的 AO 无实验侧依据。
14. **Sample 下能改善正常 selected 模型的 optimizer-policy：已测范围内为 null**（S8，SF007，STRONG）。注意这是「已测干预在所测协议下未达 practical」而非「证明不存在」；horizon 仅 h=2 epoch、anchor 仅三档。
15. **validation 与 test 层一致性**：唯一系统检查来自被隔离的 S6-R2；在可用证据中，selected 结论均建立在 validation-macro 选择 + 各自报告口径上，跨评估层的稳健性基本未测（SF010 只要求同时报告，未提供跨层规律）。
16. **checkpoint/选择规则稳健性**：canonical 规则为 R0 validation-macro selection；替代规则（R1/R3/R4）的 exact 重选评估只出现在被隔离的 S6-R2 中且不完整。可用证据中**没有**对选择规则敏感性的系统结论。
17. **梯度统计→性能的因果链：已削弱至不成立**（H011/H018 WEAKENED；D010 CLOSED）。即：不能引用「gradient conflict / U_k 修正」作为已被实验支持的方法前提；但也注意其反向——梯度结构本身（Domain×Task cosine 按域组织，E023）作为**描述性观察**仍成立，只是无性能因果含义。
18. **seed 数量与统计功效**：所有 S 阶段均 4 seeds（2022–2025），旧 P 阶段 3–5 seeds；任何小于 ~0.003 AUC 的效应在多数配置中处于预设 practical 门槛之下，档案对亚门槛效应一律不作方向断言。
19. **第三数据集 / 其他 backbone / 其他 MDMT 模型结构：完全未测**。全部证据限于 MovieLens（1M 及其衍生配置）与 KuaiRand-Pure 两个数据集、M3OE 为主（MMoE/PLE 仅 S1 与旧 P 系列）。
20. **AUC/LogLoss 之外的指标**（如校准度、公平性、排序指标 NDCG/Recall、业务指标）：8 份汇总中无登记，实验侧不能回答。

### 6.3 汇总档案本身的覆盖缺口（如实声明）

21. **原始阶段报告与日志在本工作区缺失**：所有数值转引自 8 份汇总，无法二次核验；Global Summary 声称做过抽查（S5 聚合表、S6 Phase B audit、S6-R2 CSV 与 integrity report、S7-B 原始 CSV、S8 paired endpoints），但抽查底稿不可见。
22. **S7-B 曾有过一次报告级数值错误并已更正**（C012：4/4 → 3/4），提示其余未抽查条目的逐 seed 口径存在同类风险；涉及逐 seed 一致性的 AO 判断应保守。
23. **exp_sn.md（2026-10-06 快照）不可见且已知三处过时**（C016 Phase B、C018 S5-ML 漏登、C019 过宽结论）；任何只来自该快照的说法均不应作为当前口径。
24. **E2、E7 两个计划实验未运行**：objective 机制拆分（E2）与 adaptive 方法泛化门控（E7）是**设计上已识别但从未执行**的空缺，相应的因果拆分问题（OQ007 等）在实验侧完全无数据。

---

## 7. 未决问题

按任务书指定的五个主题归类（objective 差异 / 域比例 / 域大小 / 梯度机制 / 优化器状态），并标注对 AO 核对的相关性。来源：`Experimental_Open_Questions.md`。

| OQ | 主题 | 问题 | Stage R0 相关性 |
|---|---|---|---|
| OQ001 | 域比例 | Sample 在更极端域比例下是否出现 selected minority-domain gap？已测自然比例无 practical NT，不能外推未测 ratio | **高**：凡以「极端比例下小域退化」为前提的 AO 方向，历史上**无支持也无反证**——属未测区域 |
| OQ002 | 域大小 | domain size 是因果变量还是 objective/identity/exposure/difficulty 的代理？从未独立操纵 size | **高**：任何「按域大小设计干预」的 AO 不能声称 size 因果已建立，也不能声称已否定 |
| OQ003 | 优化器状态 | Equal KuaiRand d1 的 history effect 由 m/v/bias-correction/参数路径中哪些承载？moment gate 15/16 + S6 KuaiRand M1 FAIL | **高**：凡声称利用特定 optimizer 状态子成分的 AO，实验侧尚无子成分级识别 |
| OQ004 | 跨设置复现 | Equal access/history effect 能否跨 backbone、非最大 target、小域 target、其他数据集复现？机制矩阵几乎全在 M3OE | **高**：限制一切跨 backbone/跨 target 外推 |
| OQ005 | 优化器状态 | MovieLens late DH gain 中普通 overfit 与 optimizer-policy 各占多少？S8 未分离 | 中：late-stage 类 AO 的证据基调是 overfit rescue，未分离前不能记为方法收益 |
| OQ006 | 梯度机制 | Domain×Task 梯度结构（cosine、harmful pairs、U_k 条件性）跨模型/数据集/阶段的稳定范围？仅两数据集一配置 | **高**：凡以梯度冲突结构为靶点的 AO，跨设置普遍性未建立（且 U_k→性能的因果链已 WEAKENED） |
| OQ007 | objective 差异 | KuaiRand 的 train/test mixture 与 label prior shift 能解释多少 Equal/Sample 残余差异？shift 与 objective 未独立操纵 | **高**：Equal/Sample 差异的部分可能是数据 split 而非目标权重本身——影响「objective 是有效干预轴」的解读 |
| OQ008 | 跨数据集 | S5-ML 部分复现由数据结构还是训练阶段调节？两数据集多因素同变 | 中：跨数据集外推的不确定性来源 |
| OQ009 | 梯度/评估机制 | S6-R2 KuaiRand validation/test 反转的原因及跨数据集范围 | 中（S6-R2 隔离使用，见第 5 节） |
| OQ010 | 参数定位 | 参数交换崩溃后能否定位 DH test gain 来源 | 中（S6-R2 隔离使用） |

---

## 附：证据强度分布（档案口径）

- Evidence Ledger 32 条：STRONG 10 / MEDIUM 19 / WEAK 3。
- Stable Findings 12 条：STRONG 8（SF001–SF005、SF007、SF009、SF010）；MEDIUM 4（SF006、SF008、SF011、SF012）。
- Historical Hypotheses 23 条：ACTIVE 2、PARTIALLY_SUPPORTED 3、WEAKENED 5、FALSIFIED 9、SUPERSEDED 1、OPEN 3。
- Contradiction Register 23 条（C001–C023）；Dead Ends 12 条（D001–D012）；Open Questions 10 条（OQ001–OQ010）；Experiment Index 32 条（其中 29 个有正式运行/探测记录，2 个未运行/门控 E2/E7，1 个 evaluation-only S6-R2）。
- 本审计对 AO 核对的总体提示：**STRONG 级证据几乎全部集中在「否定性/收窄性」结论**（无普遍 NT、幅度解释不充分、无 selected continuation 收益、objective 条件性）；**正向机制识别类结论最高仅达条件性 STRONG 且边界极窄**（SF003/SF005，仅限 KuaiRand/Equal/d1/tab1）。
