# MDMTRec 实验设计、现象与结论

按 [01_AI_EXPERIMENT_INDEX.md](01_AI_EXPERIMENT_INDEX.md) 的触发条件定位个别记录。实验编号、证据层级和冻结方案定义了解释范围。数表、逐 seed 和归档核验见 [03_EXPERIMENT_RESULTS.md](03_EXPERIMENT_RESULTS.md)。本文件历史材料中的旧计划、恢复指令和阶段门只记录当时状态，当前执行边界以用户指令和根目录 AGENTS.md 为准。

## 常用协议与身份

现代 M3oE fixed1 矩阵多用自然原行序 80/10/10、train-only 预处理、3 域×2 任务、保留自然尾批、50 epoch、batch 40960、FP32。ML 任务为 rating>3/rating>4，KR 为 click/long_view；KR 的 d0/d1/d2 对应原 tab 0/1/4，训练人数 172782/1743898/89452。ML 训练人数为 155949/318599/325619。常用单 coupled Adam lr=.01、betas(.9,.999)、eps=1e−8、ML/KR 的 WD 分别为 1e−4/1e−5。固定 EMA ρ 分别 .9828205985452511/.995，从初始化开始，每次有效 Adam 更新后更新；online 和 EMA 分开按验证六格 Macro AUC 的最早最高点选取。具体版本以对应记录为准，早期人工人口、V-LL、四 seed、开发种子和历史暴露 Test 均独立标记。域规模和困难度严格区分。历史材料中的规模/困难度假设仅保留其年代和历史证据。

## EXP001-019

| ID | 阶段 / 实验组 | 数据集与模型 | 训练条件与对照 | Seeds / 运行 | Metrics / 来源 | 目的与范围 |
| --- | --- | --- | --- | --- | --- | --- |
| EXP001 | P1 诊断（D0.2/D0.4/D0.6/D0.7） | MovieLens-1M；M3oE | as-published 训练；检查架构参数、共享梯度 SNR/cosine、逐域曲线及全局早停相对各域峰值 | 5 seeds（2022–2026）；D0.4 含 batch 40960/4096 | SNR、梯度范数/cosine、验证/测试 AUC、peak epoch；[`P1_diagnosis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P1_diagnosis.md) | 检查“小域更噪/更难”及 per-domain momentum 前提；只支持该数据与测量口径。 |
| EXP002 | P2 baseline / AutoML / optimizer 修复对照 | MovieLens-1M；M3oE | as-published、固定架构权重 0.7311/0.5、修复 B10 的对照 | 5 seeds（2022–2026） | 六单元汇总 test AUC、配对差；[`P2_baselines.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P2_baselines.md) | 量化旧 AutoML 与 optimizer-grouping 修复的影响；不代表其他架构搜索。 |
| EXP003 | P5 MovieLens imbalance-ratio sweep | MovieLens-1M；M3oE | baseline / comp1 等历史方法；downsample ratio 0/1/3/10/30/100；ratio 30/100 同时改变总 N | ratio 不同为 3 或 5 seeds（详见原表） | test AUC、per-cell / oracle 差；[`ML_P5_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/ML_P5_REPORT.md) | 检查 comp1 与不均衡强度的关系；ratio 序列非单调，且高 ratio 与总样本量混杂。 |
| EXP004 | P5 KuaiRand 主矩阵与 oracle 对照 | KuaiRand-Pure；M3oE | baseline、fixed073、comp1、comp2、组合臂；与单域 oracle 对照 | 5 seeds（2022–2026） | 六个 domain-task AUC、oracle gap；[`KUAI_P5_MAIN_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/KUAI_P5_MAIN_REPORT.md)、[`KUAI_STAGE1_ORACLE.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/KUAI_STAGE1_ORACLE.md) | 检查旧方法在 KuaiRand 上的迁移；oracle 未被追平，早期显著性未经多重检验校正。 |
| EXP005 | P0-1 固定总样本量与 L/M/S 角色轮换 | MovieLens；M3oE | 固定 N；物理域与大小角色轮换；joint 对比单域 oracle | 4 seeds；24 joint + 36 oracle runs | per-cell Δtransfer、AUC；[`P0_1_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_1_STAGE_REVIEW.md)、[`P0_1_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_1_RAW_REPORT.md) | 检查退化是否跟随大域角色或物理域身份；不能单独识别样本规模因果。 |
| EXP006 | P0-2 固定 N 的 ratio sweep | MovieLens；M3oE | 多 ratio、角色与方法配对；复用 P0-1 部分 run | 4 seeds；80 joint（56 新跑+24 复用）、120 oracle（84 新跑+36 复用） | per-cell transfer、总体/最差域 AUC；[`P0_2_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_2_STAGE_REVIEW.md)、[`P0_2_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_2_RAW_REPORT.md) | 分开考察 ratio 与总 N；结果仍不能把 ratio 等同于 domain size 的单因果。 |
| EXP007 | P0-3 M3oE / MMoE 跨骨干对照 | MovieLens；M3oE、MMoE | strong imbalance；baseline 与 comp1；含 joint/oracle 参考 | 4 seeds；32 joint + 48 oracle runs | per-role/per-cell AUC、transfer；[`P0_3_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_3_STAGE_REVIEW.md)、[`P0_3_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_3_RAW_REPORT.md) | 检查大域角色效应是否只属于 M3oE；MMoE 也出现同向现象，削弱 M3oE-specific 解释。 |
| EXP008 | P0-4 训练 loss weighting × checkpoint objective | MovieLens；M3oE | Equal / Sample / sqrt 训练目标 × Equal / Sample 选点；多个组合 | 4 seeds（配对单元详见报告） | per-role transfer、overall/worst-domain AUC、选点 epoch；[`P0_4_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_4_STAGE_REVIEW.md)、[`P0_4_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_4_RAW_REPORT.md) | 检查大域损失是否随目标权重改变；区分训练目标与选点指标。 |
| EXP009 | P0-5 comp1 × loss weighting 2×2 | MovieLens；M3oE | Equal/Sample × comp1 off/on；同设置配对 | 12 paired blocks（4 seeds × 3 角色轮换） | overall/worst-domain、L/M/S cell AUC、交互；[`P0_5_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_5_STAGE_REVIEW.md)、[`P0_5_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_5_RAW_REPORT.md) | 分解 comp1 与 objective 的共同变化；早期“小域 residual”解释后来被 S0 约束。 |
| EXP010 | P1-2 curriculum life-or-death | MovieLens；M3oE | static、forward、reverse curriculum；并含 comp1 分层 | 12 配对单元；具体 run 计数见报告 | overall/worst-domain AUC、peak epoch、梯度轨迹；[`P1_2_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P1_2_STAGE_REVIEW.md)、[`P1_2_RAW_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P1_2_RAW_REPORT.md) | 检验 curriculum 是否稳定超过静态目标；测得差异处于噪声级。 |
| EXP011 | E1 objective contrast / diagnosis | MovieLens r50；M3oE | Equal 与 Sample 对照；objective 系数与 routing/exposure 未完全分离 | 12 paired blocks（报告口径） | selected AUC、coefficient / mismatch 诊断；[`stage_E1_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E1_analysis.md) | 建立 objective 变化的观察性证据；不作单一中介因果判断。 |
| EXP012 | E2 机制拆分提案 | UNKNOWN | 计划区分 loss coefficient 与 sampling exposure；未实施 | 0 runs | 无结果；在 [`EXP_AI_CONTEXT_V2.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/EXP_AI_CONTEXT_V2.md) 标注 E2 未执行 | 明确记录未运行，不进入性能证据。 |
| EXP013 | E3 alpha response / endpoint replication | MovieLens r50；M3oE | 固定端点复现 E1，并扫描 objective mixing coefficient α | 4 seeds（12 blocks）；正式配置见报告 | selected AUC、响应曲线；[`stage_E3_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E3_analysis.md) | 检查 objective response 是否有稳定 interior winner；未识别 routing 与 loss coefficient 的独立效应。 |
| EXP014 | E4 KuaiRand objective 与 split audit | KuaiRand-Pure；M3oE | Equal / Sample；采用时间排序重建 split | 4 seeds（正式对照）；数据集审计见 [`stage_E4_dataset_audit.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E4_dataset_audit.md) | per-domain/task AUC、train/test mixture 与 label prior；[`stage_E4_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E4_analysis.md) | 检查 objective 对大域 target 的影响并审计 split；train-test shift 是竞争解释。 |
| EXP015 | E5 schedule / curriculum | KuaiRand-Pure；M3oE | Forward schedule 与 static Sample 对照；其余配置见正式设计/分析 | 4 seeds；formal run count 以 [`stage_E5_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E5_run_report.md) 为准 | selected/final/common AUC、LogLoss、GO gate；[`stage_E5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E5_analysis.md)、[`stage_E5_go_no_go.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E5_go_no_go.md) | 检验预设训练 schedule 是否稳定超过 Sample；报告裁决未通过 GO。 |
| EXP016 | E6 adaptive calibration / EMA | KuaiRand-Pure；M3oE | Adaptive-NoEMA / Adaptive-EMA / static Sample / Equal 等 | 24 formal runs；4 seeds（2022–2025） | overall/worst-domain AUC、LogLoss、negative-transfer、common epoch；[`stage_E6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E6_analysis.md)、[`stage_E6_go_no_go.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E6_go_no_go.md) | 检验自适应系数与 EMA；STOP，未建立优于 Sample 且满足 safety gate 的方法。 |
| EXP017 | E7 generalization gate | 未运行 | E6 STOP 后未冻结可推广的 adaptive 方法，E7 gate 不允许启动 | 0 runs | [`stage_E7_gate_stop.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E7_gate_stop.md) | gate 记录，不是实验结果；不把失败方法改名为可推广方法。 |
| EXP018 | P6 component ablation / sensitivity | MovieLens ml_r30；M3oE | 17 arms：动量、课程、架构梯度 EMA、组合与超参数；历史 objective | 17 × 5 = 85 runs（2022–2026） | test AUC、配对检验、γ/κ 敏感性；[`P6_ablation.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P6_ablation.md) | 历史结果支持 comp1，但后续 S0 的匹配 uniform EMA 对照削弱“域条件动量独特性”。 |
| EXP019 | P7 backbone 与 D/T 扩展 | MovieLens；M3oE、MMoE、PLE、STAR、ShareBottom 适配 | comp1 vs none；D/T 扩展和骨干层移植；STAR 复现差距过大而排除 | 主要矩阵按报告任务清单；ShareBottom 单独补跑 10/10；seeds 依子矩阵 | per-backbone AUC、D/T transfer；[`P7_generalization.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P7_generalization.md) | 评估历史组件迁移；仅支持所测结构，不等于 S3–S6 的机制泛化。 |

## EXP020-032

| ID | 阶段 / 实验组 | 数据集与模型 | 训练条件与对照 | Seeds / 运行 | Metrics / 来源 | 目的与范围 |
| --- | --- | --- | --- | --- | --- | --- |
| EXP020 | S0 momentum matched controls | MovieLens；M3OE | Sample objective；7 arms × 3 固定角色轮换 × 4 seeds；comp1、Adam、uniform EMA 与记忆控制 | 84/84 有效 runs；12/12 配对区块通过 | macro/per-cell AUC、LogLoss、worst-cell；[`stage_S0_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S0_analysis.md)、[`stage_S0_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_S0_run_report.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | 检验 comp1 是否有独特于普通平滑的收益；不是新写的 stage summary，而是当前事实核查。 |
| EXP021 | S1 Sample joint vs single-domain reference | MovieLens d0/d1/d2_large、KuaiRand natural；M3OE/MMoE/PLE | Sample objective；3 backbones × 4 run types × 4 seeds | 192/192 有效 runs；288 paired seed units | selected/common per-cell AUC、LogLoss 与 reference variance；[`stage_S1_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S1_analysis.md)、[`stage_S1_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_S1_run_report.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | 检验 Sample 下普遍 residual negative transfer；按预注册门槛未发现稳定实用 NT。 |
| EXP022 | S2 Equal / Sample / SD reference | KuaiRand natural、MovieLens d2_large；M3OE | Equal 与 Sample 对照 single-domain multitask reference；重用 S1 M3OE runs | 80/80 记录验证；4 seeds | selected/final/common per-cell AUC/LogLoss、macro/worst-cell、角色轮换；[`stage_S2_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S2_analysis.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | 比较 objective 条件性与 checkpoint dependence。 |
| EXP023 | S3 objective × shared-update access | KuaiRand natural、MovieLens d2_large；M3OE | Equal/Sample × FULL/TARGET_ONLY_SHARED；成对控制 shared 参数梯度访问 | 24 新跑 + 8 复用 Equal-FULL；4 seeds；严格配对区块 8/8 | target/non-target、macro、AUC/LogLoss、selected/common；[`stage_S3_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S3_analysis.md) | 在窄配置下识别 shared-access 对照的因果影响；不单独证明梯度夹角机制。 |
| EXP024 | S4 A2 magnitude controls | KuaiRand natural；M3OE / Equal | F/T/N/U；N 匹配 raw-gradient norm，U 匹配 applied-step norm | 16/16 canonical runs；4 seeds；早期 S4/A1 失败尝试不进入分析 | target/per-cell AUC、LogLoss、common epochs、执行审计；[`stage_S4_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S4_analysis.md)、[`stage_S4_A2_execution_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S4_A2_execution_report.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | 测试更新幅度是否足以解释 T−F rescue。 |
| EXP025 | S5 KuaiRand history × current | KuaiRand natural；M3OE / Equal | F/T/HT/DH，交叉操纵持久 Adam history source 与 current update source | round 3 的 16/16 canonical runs；4 seeds；round 1/2 失败记录不纳入结果 | selected/final/common AUC/LogLoss、moment audit；[`stage_S5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_analysis.md)、[`stage_S5_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_run_report.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | 将 d1 Equal rescue 定位到 history carrier；保留 15/16 moment-gate 限定。 |
| EXP026 | S5-ML cross-dataset extension | MovieLens d2_large；M3OE / Equal | S5 同定义 F/T/HT/DH；natural domain shares | 16/16 canonical runs；4 seeds；moment gate 8/8 | selected/final/common、non-target tradeoffs；[`stage_S5ML_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5ML_analysis.md) | 单独补足 exp_sn 未登记的 S5 MovieLens 复现；完整 Pattern A 仅部分复现。 |
| EXP027 | S6 Phase A: Sample F/T | KuaiRand natural + MovieLens d2_large；M3OE / Sample | F/T 对照；测试原 Equal rescue 是否在 Sample 下达到 practical target gate | 16/16 新 runs；2 datasets × 4 seeds × 2 arms | selected/common/final target + non-target、AUC/LogLoss；[`stage_S6_phaseA_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseA_analysis.md)、[`stage_S6_phaseA_final_analysis_20260929.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseA_final_analysis_20260929.md)；旧汇总 [`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md) | Phase A 的双 STOP 仍是 gate 事实；旧汇总未包含后续扩展。 |
| EXP028 | S6 Phase B: Sample 2×2 extension | KuaiRand natural + MovieLens d2_large；M3OE / Sample | 在原 F/T 上补 HT/DH；history source × current source | 16 新 runs + 16 Phase A arms；每数据集 16-arm records、4 seeds | target/non-target/macro、AUC/LogLoss、common trajectories、moment gates；[`stage_S6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_analysis.md)、[`stage_S6_phaseB_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseB_run_report.md)、[`stage_S6_phaseB_audit_review.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseB_audit_review.md) | 用户在双 STOP 后要求完成的诊断扩展；MovieLens M1 PASS，KuaiRand M1 FAIL。 |
| EXP029 | S7-A Domain×Task gradient probes | KuaiRand natural、MovieLens d2_large；M3OE / Sample | 8 source runs 在 selected/common checkpoints 上做 gradient probe；非训练干预 | 8/8 probes；4 seeds × 2 datasets | gradient norm/cosine、U_k、H、validation trajectory；[`stage_S7_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S7_analysis.md) | 描述 Domain×Task 结构与性能的时序关联；局部一阶指标不等同于因果。 |
| EXP030 | S7-B same-θ early/late continuation | KuaiRand、MovieLens；M3OE / Sample | F vs TARGET-current，anchor e9/e39；短程 same-θ 对照；private fresh-Adam 是已声明近似 | 16/16 runs；2 datasets × 4 seeds × 2 anchors | steps、gradient geometry、validation AUC/BCE；[`stage_S7B_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S7B_analysis.md) | 检验 gradient utility 改变是否在短程带来 validation 改善。 |
| EXP031 | S8 full-state optimizer / trajectory kill test | KuaiRand、MovieLens；M3OE / Sample | 8 F base retrains + 24 full-state continuations；selected/middle/late anchors；F/T/DH branches | 8/8 base + 24/24 continuations；4 seeds | best AUC vs frozen original F best、trajectory、effective Adam updates；[`stage_S8_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S8_analysis.md)、[`S8_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/S8_REVIEW.md) | 检验 late-stage optimizer residual 是否能改善正常选择性能；主结论为 late overfitting rescue only。 |
| EXP032 | S6-R2 selected-solution robustness + parameter-swap evaluation | KuaiRand-Pure；MovieLens negative control；M3OE / Sample | 复用 F/DH saved states；R0–R4 validation selection、A-grid sensitivity、val decomposition、selection windows；KuaiRand manifest-based parameter swaps | 4 seeds per dataset；16 个既有 F/DH dataset×arm×seed checkpoints；无新增训练 | selected/grid test and validation N/T/M、epoch trajectory、window classes、hybrid evaluation；[`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`stage_S6_R2_design.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_design.md)、[`s6r2_integrity_report.json`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json) | 检验 KuaiRand DH test-side selected non-target gain 是否依赖选择规则、尖峰或参数分区；源报告 Gate 为 scoped GO，但机制定位仍不清；结果限于 saved-state evaluation，非独立训练重复。 |

## EXP033-042

| ID | 阶段 | 范围与冻结协议 | 新训练 / 计算 | 结果与原报告 | 关键结论/遗留 |
| --- | --- | --- | --- | --- | --- |
| EXP033 | 10-15 / B1 F-DH重建 | KR natural / Sample / M3OE，F/DH×4seed×50epoch，全状态、R0–R4 validation selection | **8/8 新训练**；400state/800预测，**非旧S6 exact** | [B1重建分析](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-15/research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016/scientific_analysis.md) | R0 test非目标+.000579 2/4正未复制旧+.006757 4/4；lineage和新旧机制差异未识别，R1/R4线索不可变主 |
| EXP034 | 10-15/16 C2.1 / Stage2B-C固定θ | 两数据集、4seed、样本pool、同α变q、梯度trace / 64-step SGD/Adam/AdamW固定场replay | **0正常训练**；8case×后续4096replay，不可算8正式run | [Stage2+数值](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/execution_20261016/experiment_results.md) | 同期望梯度不同covariance；投影与完整trace判断可反号，非长期性能机制 |
| EXP035 | 10-16 Stage2D | KR/M3OE，**learned fusion / main+arch双Adam**，Sample/Equal×Natural/Balanced、4seed×50epoch | **16/16正式训练**，无新历史exact S5 | [Stage2D正式结果](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/experiment_results.md) | SB−SN validation−.010406四负；EB−EN−.006547；改善/无害门失败，m/v中介未证 |
| EXP036 | 10-16 v2 P1及中期P3–P7 | KR fixed fusion / 单optimizer，IID Sample q；Adam vs 无动量SGD，LR受网格界限 | **P1完成，中期非最终全矩阵**；其它P2/P8不能按队列状态算完成 | [v2中期机制分析](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v2/research_plans/2026-10-16_v2/execution_20261016/mechanism_analysis.md) | SGD SB−SN−.002104四负，Adam通用必要性被反例否定，非与EXP035同协议 |
| EXP037 | 10-16-v3 完整V0–V13 | KR/ML fixed fusion，同Sample J、q控制；Adam/SGD/AdaTask、LR/多预算、ORDER/PERM、谱和C局部状态 | **全部规定及触发分支完成**；继承更早祖先，**与v2 Git历史分叉**；家族数/可用seed按实际各表，不与EXP035叠加 | [v3 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md) | 配置/阶段/selection反例、共享C局部阳性而长期门未触发、method NO_GO；旧Q1–Q46源 |
| EXP038 | 10-17/18 N0历史读数与人口/校准审计 | 七家族旧冻结模型，不重训；same-model macro/train-p/test-p weighting，既有行预测校准 | **0新正式训练**；历史派生分析，不计8seed新确认 | [N0评价分解](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/METRIC_AND_POPULATION_DECOMPOSITION.md) | 同预测评价权重可反号；校准改善LL但不变AUC；split内部因果未识别 |
| EXP039 | 10-17/18 N1 q×LR正式矩阵 | KR fixed fusion、同Sample J，Adam/SGD×SN/SB；调参seed2030/31、正式seed2032–39，H100 | **32 tuning +32 formal =64新H100**；N1 common/own模型复用同轨迹不重复计 | [N17 FINAL_ANALYSIS](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | KR Adam selected−.005203/SGD+.000837，预注册Holm都未过；Adam AULC early负是次级 |
| EXP040 | 10-17/18 N2 quota + 定θcount分解 | KR 16个有效固定θ锚×四臂×512draw；formal追加同8seed quota SN/SB H100，IID父N1复用 | **16新quota H100** +固定θ数值诊断；不把16 IID父重计 | [N2 quota/counterfactual](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/DOMAIN_COUNT_COUNTERFACTUAL.md) | count项占trace仅.001297%–.105936%；quota效应仍存、interactionCI跨0；机制不唯一 |
| EXP041 | 10-17/18 N3 shared-C局部/方向 | 原生θ/状态下16开发、16 SGD C-controls、8独立confirm局部锚，冻结embedding/E10 | **40局部干预锚，不是40 H100训练**；四confirm优化seed独立 | [N3局部与N4门](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/C_LOCAL_AND_TOPOLOGY_ANALYSIS.md) | 两路径方向4/4正但效应~3e−5，实际/方向门失败，**N4 NOT_TRIGGERED**，不当长期否定 |
| EXP042 | 10-17/18 N5-C ML固定人口、对称/own LR边界 | ML固定subset_seed2022，16 tuning×H100，8seed×SN.001/SB.001/SB.0003共24 formal×H100 | **16 tuning +24 formal =40新H100**；common/own复用SN同轨迹 | [N17 ML/阶段结论](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | common selected−.010392、own selected−.010875与own E100+.021145并存；LR/时钟非纯q；guard未过 |

## EXP043-051

| ID | 实验 | 执行范围 | 结论 / 边界 | 原报告 |
| --- | --- | --- | --- | --- |
| EXP043 | N18 P1–P6六方向首轮 | **COMPLETED（已触发的有界诊断/冻结头）**；六方向受控构造与四seed头拟合，长期训练按各gate决定 | 存在理论与局部反例，但各当前方案没有稳定的强对照完整算法价值；DATA-BLOCKED不写作训练失败 | [N18 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md) |
| EXP044 | N19六方向完整端到端 | **COMPLETED：328次正式网络训练＋冻结测试**，四seed，依正式ledger去重 | 容量/通道失活及模型选择、阶段、指标反转，尚不支持通用独立新算法；ML训练人口多为人工固定subset，不是自然全量 | [N19 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md) |
| EXP045 | CM19共享调制路径 | **COMPLETED：856次全预算训练＋856组冻结测试**；独立STAR/PEPNet/HAMUR等强对照 | KR fixed M3oE选点输入乘性调制数值中性与局部优化修复得到限定机制支持；恢复路径不等于改善合法selected性能 | [CM19 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md) |
| EXP046 | 10-20自然ML追溯/匹配N/阶段复核 | **PARTIAL / IN_PROGRESS**：自然base-train人口审计与V2/S5/S6/AO04/N17等已完成子项；N19/CM19自然transfer截至已存状态未全闭环 | 人工187500对自然800167同时改变比例、真实支持与训练预算；若只看人工旧N19/CM19不得声称全自然复现 | [人口审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)、[当前工作](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md) |
| EXP047 | 10-21原生M3oE+D/G套件 | **COMPLETED：48条有效正式训练**（2数据×6臂×4seed）及12单epoch检查 | ML D与Dconst01均优于B，动态specificity非该套件收益必要条件；G−Gtau0终点宏AUC负。其目标legacy_group_sum，非其他Sample/Equal实验的对照复现 | [10-21 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md) |
| EXP048 | 10-24 I1权限/来源使用 | **COMPLETED：44主pilot＋4基线pilot＋40正式训练** | I1_PERM在ML/KR selected test AUC未胜固定RW；KR final LL的有限改善仅为次级 | [10-24 README](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/10-24/research_reports/2026-10-24/README.md) |
| EXP049 | 10-25 MDMT-NEXT-v2 | **COMPLETED核心48条24epoch；正式四seed50epoch为NO_START；部分aux和基线预算停止** | 晋级门未过，版本STOP_NO_PROMOTION，未启动正式不是实际负性能样本 | [v2 README](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-25/research_reports/2026-10-25/README.md) |
| EXP050 | 10-25 MDMT-NEXT-v3（RHD/LPS） | **COMPLETED：69条24epoch逻辑轨迹＋20条KR四seed50epoch轨迹**；部分条件NO_START | RHD相对BASE selected test LL较差但e50较好；LPS来源可读与特定形成器有用不等价；测试/旧Q已历史暴露 | [v3 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md) |
| EXP051 | 10-26 DFD-v4（需求条件来源函数形成） | **COMPLETED已触发P1及ML P2；KR P2与两数据P3/P4 NO_START**；184冻结预测、12结构计时 | ML P1需求读法有限优于常量；形成器与整体方法输强直接对照，版本不支持；outer refitη固定μ分组未导出（MAJOR） | [v4 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md) |

## EXP052-060

| ID | 来源 | 完成工作、重复单位及对照 | 结果/最终状态 |
| --- | --- | --- | --- |
| EXP052 | 10-27；26-10-27 | Relation-Basis v2，2自然数据×6方法×4seed，共48正式适配；Q选TARGET_ONLY | LOWRANK/FULL未胜简单目标适配，E宏AUC ML−0.000623、KR−0.002727（均4/4负），当前版停止；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md) |
| EXP053 | 10-28；26-10-28 | BASP-A两自然集×6臂×2seed，24条正式轨迹 | CG对TARGET_MATCH无稳定收益，留出收益探测μ不胜常量、NULL读取占优；P2–P5未触发；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md) |
| EXP054 | 10-29；baseline/mdmt-sota-2526 | M3OE_FIXED1、HyperGate、AutoIFS；2自然数据×3模型×5seed，30正式训练+8开发 | 两新模型四组macro AUC相对M3OE均5/5提高，KR natural LL有权衡；并非论文严格复现；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md) |
| EXP055 | 10-31；26-10-31 | GA31-v2，loss_scale1/.1×NR-Gauss有无增强，2自然集×4臂×5seed，40正式训练 | ML c=.1无增强selected AUC+0.003473，KR缩放方向相反，增强未过门；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md) |
| EXP056 | 11-01；26-11-01 | GA31-R3，ML普通L2控制、M1/IID2/ANTI2；45正式训练 | L2可作ML强普通参照；ANTI2降低方差但不胜IID2，未晋级；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md) |
| EXP057 | 11-01；26-11-01_v2 | 4训练臂×2自然集×5seed=40正式训练；online+参数EMA shadow与SAD βδ控制 | EMA selected AUC实益、ML有LL代价；SAD长差分未胜普通增强，P2未触发；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md) |
| EXP058 | 11-01；26-11-01_v3 | AIGM01：56条唯一完整训练、跨开发/正式复用8个RAM身份；正式A0/A1为五seed，AIG只开发 | EMA排序收益再现；KR Gaussian有域任务交换，固定SAD/AIG动态门不晋级；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md) |
| EXP059 | 11-02；26-11-02 | GATS02：31条新增完整50epoch科学轨迹；B1/B2/B3/C2与匹配控制，KR B2保留3seed确认 | B2开发正而统一test五seed对A1宏AUC约+0.000005；不认整体晋级；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md) |
| EXP060 | 11-02；26-11-02_v2 | EMA02V2：A0/A1/J0/J1/F0/F1/RG/AG/TG，两自然集×2开发seed；36正式训练、144统一测试端点 | ML F1弱测试信号不足门，KR TG/AG损害明显；确认/W1未触发；[原报告](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md) |

## EXP061-064

| ID | 来源 | 完成工作、重复单位及对照 | 结果/最终状态 |
| --- | --- | --- | --- |
| EXP061 | 11-09；26-11-09 EMA_ADAPTIVE A0 | 复用11/07的20条轨迹；0新增训练、0 Adam更新、0 Test访问；训练面板/验证侧预测与动作审计 | C_persist未迁移为正验证增量，C_carry相对统一Fast近零，静态task策略退化为Fast；A0=NO_ACTIONABLE_SIGNAL，A1/B/C未触发；[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md) |
| EXP062 | 11-09；26-11-09_v2 PUB-MDMT | 两数据×P0/P1/P2及同α G控制；70条正式轨迹、3500 epoch、120750 Adam更新，五seed | 当前P0/P1/P2未胜A/F；G控制接近退化，C−G不能替代正常基线比较；当前迁移不晋级；[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md) |
| EXP063 | 11-10；26-11-11 EMA_CAUSAL_ATTRIBUTION | ML/KR×Clean/Gaussian四条件、20条正式轨迹、1000 epoch、34500 Adam更新，五seed；共享/非共享参数、时间窗口、函数与方向干预 | 固定EMA收益主要来自共享S/EMBED；recent×old互补；预测组合在固定面板有收益但无独立Test；当前自适应菜单无稳定动作增量；[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md) |
| EXP064 | 11-11；26-11-11 EMA_NEXT | 95个job、1415 epoch、46280 Adam更新；五seed；A反馈、B两态读出、C条件动作、KD | A_MATCH反馈损害F；ML两态B主要改善LL、无新增AUC；KR B_H277保留小幅诊断AUC信号；C无future signal，KD未形成独立AUC晋级；[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md) |

## EXP065

[原报告 65](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md)

**PAIR/ECCM 因子消融。** 两自然数据、seeds 2022/2024，原 24 条正式训练在 A0/A1 下比较静态监督 P00 与动态监督加 ECCM P11 的 λ=.05/.20；随后补 8 条 λ=.05 P01/P10 以闭合 2×2。区分静态/动态 PAIR 与普通 Adam/ECCM。KR P01−P00 selected Test +.012099，但 Validation −.000101；KR Test 交互 −.012347，两 seed 负，ML −.000384、方向不齐。原开发门停止，S2/S3 未触发。固定 EMA、早期/选点、各格与概率损失分别保留。五种子后续见 EXP067/080。

## EXP066

[原报告 66](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_history_information/REPORT.md)

**条件信息与真实历史。** 从 A0f/A1 的冻结身份出发，改变 M、T0/TM、H_REAL/H_MIX、C_MIX、SCALE 中的监督、任务与历史作用点。单一候选 seed 2022，新训 ML 7/KR 6 条。KR TM 完整批成本 4.61×，为成本阻塞。KR C_MIX Test Macro AUC/LL +.009531/−.005302，验证 d0t0 −.006295；H_MIX Test Macro +.012815 而 Traffic −.000647。ML 全评估候选 Test 主 AUC 非正，T0 LL 改善。后续五 seed 覆盖部分候选。A0b、D1/D2/Confirm 原门未触发。

## EXP067

[原报告 67](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)

**五 seed 补全。** 2×4×5 的40个逻辑槽位含 30 新/10 复用，按同一六格 AUC 选点。C/H/P01 对普通 M3oE online Test AUC 的 ML 增量 .003253/.003059/.003225，KR .003720/.005579/.008702，六格保护均通过；先区分 EMA 贡献，再检查组件。KR P01 对 clean EMA +.004086，但 d1t0/t1 −.001334/−.001047。所有读出与阶段按源轨迹去重。

## EXP068

[原报告 68](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)

**统一发表模型与组合。** 自然人口 Test（V 按 Macro AUC 选点），统一报告 M3oE、HyperGate、AutoIFS；保留各架构实际训练协议。结合 C_MIX＋P01 的组合 CP 与独立组件作内部对比。HyperGate 在 KR .682113，M3oE 在 ML .775190。CP fixedEMA ML/KR .777525/.678168，相对 P01 −.000890/−.002441。以不同预处理和 V-LL 选择的 10/29 表未替代本轮排序。

## EXP069

[原报告 69](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)

**Gaussian 覆盖。** KR 固定 GA31-R3 κ=.1、λ=.5、EMA .995；仅改涵盖域至 [0] 或 [0,2]，各新五条；[2] 精确复用。Macro AUC D0/D02/D2=.679180/.674604/.677999，LL=.435518/.436172/.434283；D02−D0 d0t0 −.014949。ML 复用既有 A1/G-D0，不计新训练，Test Macro AUC .779357、对发表 M3oE +.004167，通过六格保护，平均 LL 上升。KR D0 对 M3oE 的总值 +.007274，仍低于 HyperGate；覆盖策略能改变各格, 全域共享增量仍依赖对照。

## EXP070

[原报告 70](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md)

**AUG_HISTORY。** 固定 EMA 与参数方向响应、路径/参数组时间尺度、以梯度漂移协方差定向增强比较。自然 ML/KR、五 seed，独立选点和 E10/E50 分开。候选改变实际更新与响应，selected 没有稳定额外排序优势；相对固定 EMA 的群组局部差、Gaussian 交互和 LL 方向按人群/阶段记录。受影响或被预注册门阻断的变体见原报告。

## EXP071

[原报告 71](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)

**EMA_MECHANISM。** 自然两数据、Clean/Gaussian；固定 EMA 对照同轨迹年龄匹配 online 与预测稳定性。配对证据支持 fixed EMA 的 selected 排序收益超过纯年龄匹配；早期学习滞后与后期平滑并存。降低排序变化率未建立充分收益。常驻分组交叉仅在 KR Gaussian 成立。复杂收益预测可提高预测准确度，动作比较未增加稳健排序；Gaussian×EMA 交互随阶段/选点改变。

## EXP072

**HAT，DT-PGNA / TGA-Pair / TGA-Example。** 2 数据×6 臂×5 seed，60×50 epoch；A/B 取各格至多 128 同域正负对，C 按同规模抽自然标签 example；共同目标 `L_clean+.25 L_aug`，PAIR 的 `L_aug=mean_pairs(BCE_pos+BCE_neg)`。A 对非目标 logit 梯度作正交投影，半径 .03 RMS(e)，有 margin cap 与实际非目标漂移 ≤.001；B 去掉投影和实际约束；RANDOM、GAUSSIAN 沿 A 支持与半径改方向，C 无配对/margin cap。敏感度探针恢复 gate 输入导数，生成 δ 后 detach；训练保留原模型路径。A 平均 drift ML/KR .000009/.000015，相比 B .016546/.014192，确有投影作用。A/B/C 对 clean EMA 在每数据均 0/5 seed AUC 正向。ML C LL −.004180、五 seed 同向；KR C 胜通用随机/高斯，但对 clean AUC −.009188。KR A/B E1–10 +.013854/+.013042，中晚期反号。ML A/B 晚期微正而各自 selected 负。matched-norm B_MATCH、HPO 和 Test 均未触发。后续零扰动归因见 EXP073。

## EXP073

[原报告 73](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/auxiliary_objective_attribution_20261115/REPORT.md)

**辅助目标归因与 NDP-A。** Stage1 保持原支持/重加权，仅令 δ=0。Z-PAIR/Z-EXAMPLE 相对 clean EMA 的 selected AUC ML −.001547/−.001995，KR −.019672/−.009075；对应 B/C 的 −.001353/−.001643 和 −.021557/−.009188。旧总质量可到 1.5，KR pair d0t1 正例额外质量约 23.4×；零扰动复现主要损失。Stage2 NDP-A 为 `.75 L_clean+.25 L_view`，自然 2B 支持，entry 质量 1/(2B)，每 task 质量 .5；各 task 的 detached BCE 上升视图仅回到本任务。50 epoch、两数据五 seed，零扰动 canonical 状态与 CLEAN model/Adam/EMA 完全相等。NDP-A−clean selected online AUC ML/KR +.000360/−.004705，EMA −.000101/−.005775。自然权重修复通过数值验收，EMA 额外排序没有出现；原 pair 质量解释不能再直接覆盖 NDP-A 失败。

## EXP074

[原报告 74](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md)

**TV-GRM。** 比较任务视图残差历史、shuffle、instant、不同 γ 和 NDP-A/CLEAN 对照；两自然数据各五 seed，50 新轨迹、20 个精确历史控制。γ=.05/.10/.20 对 clean EMA 的 selected AUC ML −.000329/−.000062/+.000154，KR −.000227/−.000166/−.001311。ML γ=.20 AUC 3/5 且区间跨零，LL 5/5 降；KR γ=.10 对 NDP-A +.005609、5/5 正，仍弱于 clean。与 shuffle 的 AUC 区间均跨零，instant 相较历史方向依数据而异。NO_CONTINUATION_SIGNAL；E50、Traffic/Pooled 不替代指定 selected Macro。

## EXP075

[原报告 75](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/task_axis_even_views_20261116/REPORT.md)

**AXIS-EVEN。** 30 条轴向反向对称、轴向 IID2、匹配范数的各向同性 even 新轨迹；20 条 CLEAN/NDP-A 复用。注册 AXIS-EVEN selected EMA 相对 CLEAN ML −.000237520、KR −.003540710（0/5、1/5 正）。ML LL +.001264852，五 seed 均恶化。KR IID2−clean +.001269552、2/5 正、区间跨零。条件期望相同的 even/IID2 隔离了部分噪声路径，当前统计未建立轴向、反向配对或 Adam 路径的独特收益。Test 封存。

## EXP076-077

[原报告 77](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/nqr_temporal_exposure_20261116/REPORT.md)

[原报告 76](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/natural_quadratic_ranking_20261116/REPORT.md)

**NQR 与时序。** C02 三臂 NQR-025、BRIER-OPPORTUNITY、MEAN-GAP；两数据×五 seed 共 30 新轨迹，10 CLEAN 控制复用。NQR selected EMA AUC 对 CLEAN ML −.000343517（0/5），KR −.000328734（2/5）；ML E10 +.001151311（5/5）而 selected task0 AUC/LL 均变差，KR 最大域及 pooled LL 有稳定改善。C03 EARLY10/LATE10 各仅暴露十 epoch，20 新轨迹且 exact reuse CLEAN/constant NQR 各10。EARLY selected AUC−clean ML −.000340045、KR −.000068613。ML EARLY 对 constant 的 LL −.003069812、5/5，AUC 近零；LATE ML selected 恰为干预前 CLEAN 最优，不能评价后置干预。E50 两数据两读出 EARLY−LATE 区间均跨零；E10 重合点不作重复。阶段效应与 selected 结论分别保留。

## EXP078-079

[原报告 79](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)

[原报告 78](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/completion_weight_attribution_20261117/REPORT.md)

**完成权重和格质量。** C04 W=COMPLETION-BCE-025；S=MASS-SCALE-025 在相同总 BCE 质量下构成纯标量参照；C=CLEAN，N=NQR-025。20 新、20 历史准确控制。W−C selected EMA AUC/LL ML −.000154621/−.000305316，KR +.002555086/−.003942334；KR d0t0 −.003141373，d2 域 +.009201240，Traffic AUC −.000166779；KR W−S LL −.003621037（5/5）。W 同时改变 class/cell/task 权重，S 保留尺度、coupled WD/eps/Adam 的交互。N−W 对 ML E1 大幅正、E50 负；同状态虚拟 Adam 与 EMA 代数对应限局部。C05 U=CELL-MASS-BCE-025 保持 W 的格/task 质量并取消格内 class 异质性，10 新+30 历史控制。U−C selected AUC ML/KR −.000221160/−.000235705，LL 均正；KR d0t0 −.007526250。格质量单独无稳定排序增益，C04/C05 Test 未开，C05 后 USER_STOPPED。C04 曾验收的 RAM owner 后来退出，历史收据只证明当时可用。

## EXP080

[原报告 80](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md) · [冻结设计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md)

**条件辅助历史。** A 核心补 P00/P10；同 PAIR/BCE 支持区分普通 Adam A10 与条件 24 历史 A20、单份辅助历史 A21。按 ML Validation 规则冻结 A21（λ=.05、S_DENSE、固定 EMA）。A 核心 12 新/28 历史复用，历史和剂量测试各20新；低剂量 λ=.025 的匹配控制两数据均未改善 Validation。80 同状态锚点用于作用域定位，未触发两种限制。A21 对发表模型、clean＋EMA、同 PAIR＋Adam＋EMA 的 Test AUC 增量分别为 ML +.003284/−.000564/−.000083，KR −.001347/+.004243/+.004062。KR 同 PAIR 比较五 seed 正向（CI [.000469,.007656]），最大域的两个任务相对 clean EMA 均下降；对发表 HyperGate 不达总门且最差格 −.014493。ML 内部强对照 d0t1 −.001040，组件无独立 AUC 增益。

**共享更新 B。** 10 条 clean 重放，40 个真实 Adam 锚点，虚拟 JOINT 与真实同状态更新身份逐项相合；2080 虚拟 Adam/720 来源梯度，对等尺度、动态/固定源、伪源及 NO_ACTION。40 脉冲槽含 7 NO_ACTION，1120 个实际反事实续步。KR online 动态相对 JOINT +.0000063、相对等预算伪源 −.0000562；ML 对 JOINT −.0000690；晚期留出两数据动态均低于固定留一 seed 来源。B_SEARCH 关闭，AB 未触发。共同冻结 S5 有 90 端点（58 个登记历史值/32 个新 Test 评价）。10 条计划 Gaussian 训练未派发：既有批级 GA31-R3 与任务要求按行键的 EMA02 不同，ML 仅两种子身份一致、KR 为零。52 科学轨迹与 10 clean 重放分列。

## EXP081

**RCF-v1 及独立 SCALE，执行版。** 旧设计索引标为 DESIGN_ONLY，后续分支已完成。RCF 对 NATIVE、H_DIRECT、固定概率读出、等监督本地/池化和随机特征，分开比较强目标后的来源增量与后继函数形成。ML/KR seeds 5201–5204；104 S1、24 S2、8 KR S3 方法完成收据。6201–6204 独立锁定种子未开；S4 两数据 NO_START。Q 上 RCF−H 的 AUC/LL ML −.00051304/+.00024518、KR +.01119579/−.00304660，但 KR 不胜 NATIVE，且普通本地/池化强对照的 LL 幅度或方向不满足门。KR 对重训 K1_WIDE 在 Q 的 AUC/LL +.00172312/−.00060349（4/4，区间跨零），不足以取代强参照。ML 历史暴露 Test 对 H 的 LL −.00186226（4/4）且 AUC +.00010003（3/4），在 legacy 反号，POOLED/LOCAL 接近；KR Test 对 NATIVE AUC −.00750254。ML SIMPLE_READER_SUFFICIENT；KR ORDINARY_ANCHORED_FINETUNING，保留 TARGET_REFITTING_EXPLANATION。P_form 与 OOF/refit 背景不同，KR Q 上出现 source LL 预算超界，训练约束与分布外保留分别裁定。

SCALE 独立使用 seeds 2022–2025，八条训练。冷/对应热 Adam 严格匹配 `cL,wd,eps ≍ L,wd/c,eps/c`，因而常数损失尺度效应包含有效 WD/eps 改变。历史暴露 Test 下 Dconst01_NO_AUG−B 的 Macro AUC/LL 为 ML +.00404890/−.00924025、KR +.00758835/−.00655322；KR 流量格 AUC −.00451560、pooled −.00445033，最大域 AUC −.00486585。主 RCF 与 SCALE 有不同设计和参照。当前停止扩展 RCF，保留强简单读出及有限阳性。

## 历史交接摘录

早期技术与当前研究的边界见索引。本节为旧根上下文的科研口径、归因、关键判据与数据锚点，研究状态按对应 EXP 章节最新裁定。历史运行会话与 GPU/RAM 编排留在原收据。



### 早期协议、结果与旧结论修订

#### 历史实验协议与数据锚点
### 4.1 先明确比较的实验对象

- P0–P7 的旧 baseline、as-published、修复 B10 的 fixedbug、Px_n 的 equal/sample、S 系列的 FULL/SD reference 是不同条件。写报告时必须把 objective、优化器参数分组、数据、checkpoint 规则写进 arm 定义；不能只用“baseline”一词。
- 原始 M3oE 训练的六项 BCE 是等权求和；后续 Px_n/E/S 对有效目标做过明确操纵。S2 的口径为 Sample：w_d=1、有效系数 a_d=p_d；Equal：w_d=1/(3p_d)、a_d=1/3，目标总体平均权重归一。P0-4/5 报告采用其自身的总尺度归一实现。跨实验不能仅凭“sample”同名就假定逐项损失和尺度完全相同。
- S1/S2 的 single-domain reference 是保留双任务的 D=1,T=2 参考，用来隔离跨域联合训练与独立单域训练的差异；不是 P5 中所有历史 oracle 的同义词。对负迁移只用 Joint−SD 或 Equal−SD / Sample−SD 判断，Equal−Sample 差本身不是负迁移定义。
- S3 的 Equal FULL/Sample FULL 与 target-only 是共享更新访问对照；S4 另含 raw-gradient norm 匹配 N 和 applied-step 匹配 U。S3 的历史 D=1 reference 仅作 legacy context，不是严格因果臂。不要把 target-only、N 或 U 当成 proposed method。

### 4.2 数据、域、任务与目标适用范围

- MovieLens-1M 早期三域按年龄：d0={1,18}、d1={25}、d2={35,45,50,56}；任务分别为 rating>3 和 rating>4。Px_n 固定 N=187,500 并轮换 L/M/S 的物理域身份；ratio、role、域名不可混为一谈。
- KuaiRand-Pure 本地 tab0/tab1/tab4 映射 d0/d1/d2；S/E 系列任务为 is_click / long_view。全量统计占比约 7.10%/89.18%/3.72%，S 系列 train split 实测约 8.6%/86.9%/4.5%；使用哪一组必须注明 split。数据按全局 (date,time_ms) 排序作 8:1:1 时序切分，是论文未公开切分规则下的重建方案，不是官方唯一管线。
- KuaiRand 有明显 train→test domain-mixture 与 label-prior shift；结论不得忽略该竞争解释。MovieLens 人工角色轮换与 KuaiRand natural setting 回答不同问题，不能把“域大小”合并成一个共同操纵变量。
- P7 MMoE/PLE 为 FuxiCTR 层移植；ShareBottom 是本地 4-expert+mean 适配，非逐字移植；STAR 复现失败并排除。P7 组件 1 泛化只到所测结构/ML 条件；D=5 证据只覆盖 T=2。S2–S4 机制结论为 M3OE 的窄范围发现。

### 4.3 配对、统计单位与模型选择

- seed 是重复与配对单位。配对比较按同 seed 对齐；同一实验的域、任务、epoch、probe 不得冒充独立 seed 增大 n。S 系列主要 n=4；旧 P5 有 n=3/n=5 混合，不能直接比较跨档 p 值。新实验若无经审查的其他预注册种子方案，使用 seeds 2022、2023、2024、2025。
- 以验证集作 checkpoint 选择；test 不可见。Selected、fixed-final 与 common epochs 是不同 estimand，分别报告，不可择其有利视图。S1 joint 用六 cell macro val AUC、reference 用双 task macro；S2 selected 用六 cell unweighted macro AUC 和最早并列规则。复用数据/代码须核对初始化、曝光、source identity 和 block validity。
- P5/P6 的 test_auc_mean pooled AUC 与六个 per-(d,t) AUC 再平均不同；P7 pooled 结果也不是 per-cell macro。每个表明确指标、聚合顺序、样本权重、ddof 和 paired n。
- AUC 衡量排序、LogLoss 衡量概率损失；两者不同向时必须保留差异。S2/S3/S4 应并报目标和非目标 cell、全局 macro/worst-domain、LogLoss 与 AUC、selected/final/common-stage。
- 显著性、实用阈值和证据强度分开表述。许多旧比较为名义 p 值且未过多重校正；不显著不等于等效，单次小样本不显著不得当成零效应。S2 的 AUC practical threshold 为 0.003，强效应 0.005；按各阶段预注册门槛，不事后变更。

### 4.4 优化器、RNG、实现与数据记录

- B10 是真实历史 bug：Python 的位取反被误作逻辑非，旧版主 Adam 包含全部参数，architecture scalars 又由第二个 Adam 更新。比较旧结果或复现时必须注明 grouping 版本；不得将含 bug 与修正版结果无标识混合。P2 显示均值影响不显著、方差明显变化。
- 任何额外 DataLoader iterator（即使 num_workers=0）会从全局 RNG 抽取 base seed，可能改变后续 shuffle。若做额外评估/诊断，应保存并恢复 torch RNG state，且验证训练轨迹未漂移。M3oE forward 返回 CPU prediction，调用侧需转到训练 device。KuaiRand 大数据使用 num_workers=0 以避免 Windows spawn pickle 失败。
- 保留训练目标、每步 loss、optimizer state 规则、数据 exposure、初始化映射与源代码身份。因果对照除预注册变量外锁定其余项；target access 的更改不得同时偷偷改变 private 参数或 loss scaling。
- 测试集 LogLoss 没有在 P5/P6/P7 主台账留存；不得从 P0/P2 或验证集 LogLoss 推算旧主矩阵的 test LogLoss。后续报告若声称 test LogLoss，须有本阶段直接评估记录。
- 报告至少保留逐 seed 原始结果、失败/重跑行、执行窗口、对照配对、manifest 与 reviewer 核查路径；failed 行不静默删除。重跑应依项目 AGENTS.md 的 DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE 顺序；当前证据不授权跳过独立设计/审查直接开跑。

---

### 4.5 项目执行角色

Main agent 只负责 ORCHESTRATOR 与最终科研分析，不直接实现研究算法。新算法实验依次由 designer 给出 DESIGN SPECIFICATION、luna-coder 按该规格实现、reviewer 独立审查、luna-runner 只记录原始结果，最后 main agent 独立分析。Reviewer 未给出 PASS 或 PASS WITH FIXES 且 blocking issues 未修复前不得 RUN。Runner 不作科研结论分析；所有阶段不得跳过。
## 5. 关键证据链

本章仅作溯源地图；当前结论见 F1–F18，不在此重复完整结果。

### 5.1 P0–P7：复现与早期算法候选

P0/P1 建立 M3oE 复现、代码事实和梯度诊断：baseline 可复现，但“小域低 SNR”设计前提失败（F1–F3）。P2 发现 B10 optimizer grouping bug，P3 重建 KuaiRand 数据切分，P4 实现三个候选组件并验证 all-off 等价；后续使用时须保留 B10/数据拆分边界（第4节）。

P5/P6/P7 先记录了 comp1 在 MovieLens 强失衡档的显著收益、KuaiRand 大域前沿移动、课程无稳定独立增益及有限 backbone 移植；同时暴露非单调 ratio、oracle 混合口径、多重检验、test LogLoss 缺失和不同 backbone fidelity。后续 P0-1–P0-5 与 S0 已重审其主要机制解释。今天仍可用的证据价值是：旧现象确实出现过、comp1 可产生增益、KuaiRand 大域负迁移量级很大；不能继续沿用“单调 size 因果”“真 Pareto”或“domain-conditioned mechanism 已证实”。来源：**P1_diagnosis.md、P2_baselines.md、ML_P5_REPORT.md、KUAI_P5_MAIN_REPORT.md、P6_ablation.md、P7_generalization.md、FINAL_RESEARCH_REPORT.md**。

### 5.2 P x_n：固定 N、域角色、objective 的归因修正

P0-1/2 在固定 N 下操纵大小角色与 imbalance ratio，支持效应位置随 Large 角色移动；P0-3 在 MMoE 复现方向，撤销 M3oE-specific 叙事。P0-4 发现 equal loss weighting 是 Large 牺牲的主要可操纵因素；P0-5 的 2×2 显示 comp1 的大部分旧增益是对等权错配的补偿，sample 下仍有小残余。P1-2 在 sample 目标下 DROP Curriculum。独立审计纠正了逐 seed 全正、完全消失、阈值即显著等过强表述。详见 F6–F8。来源：**P0_1_STAGE_REVIEW.md、P0_2_STAGE_REVIEW.md、P0_3_STAGE_REVIEW.md、P0_4_STAGE_REVIEW.md、P0_5_STAGE_REVIEW.md、P1_2_STAGE_REVIEW.md、research_diagnosis_20260909/**。

### 5.3 E 系列：机制诊断、跨数据集和动态方法淘汰

E1 支持但不确证 shared contribution distortion；E2 未执行，故 E3 响应面缺少系数与 sampling 的因果拆分。E3 不支持稳定优于 Sample 的 interior 权重；E4 在 KuaiRand 重现 Equal/Sample 方向，但 temporal shift 是残余的强竞争解释。E5 STOP 所测时间 schedule；E6 STOP 自适应校准；E7 因 gate 未授权，0 runs。详见 F9–F11。来源：**stage_E1_analysis.md、stage_E3_analysis.md、stage_E4_dataset_audit.md、stage_E4_analysis.md、stage_E5_go_no_go.md、stage_E6_go_no_go.md、stage_E7_gate_stop.md**。

### 5.4 S 系列：当前研究前沿

- **S0**：comp1 的约 +0.0054 残余可新鲜复现，但在 matched uniform-smoothing family 下失败 KEEP gate；STOP domain-conditioned Momentum。见 F12。
- **S1**：Sample+SD reference 下 STOP 普遍 residual NT/Safe-MTL 线；analysis 中“72/72 均正”和“候选在 weaker domain”两句已被审计纠正。见 F13。
- **S2**：Equal 目标下出现 Large 角色条件性 NT，Sample 在 selected 口径缓解；size 仍是关联因素，不是已隔离因果。见 F14–F15。
- **S3**：KuaiRand Equal/tab1 shared-access intervention 大幅救援 d1，但损害其他域；MovieLens selected replication 有 late-stage 反例。见 F16、F18。
- **S4**：fresh run 复现 S3；norm 与 applied-step magnitude 对照失败救援；证据收窄到梯度组成/方向及 Adam history 的联合候选，未拆分。见 F17–F18。

优先溯源：**stage_S0_analysis.md / stage_S0_review.md；stage_S1_analysis.md / stage_S1_review.md / stage_S1_paired_aggregates.csv；stage_S2_analysis.md / stage_S2_review.md / stage_S2_raw_results.csv / stage_S2_paired_results.csv；stage_S3_analysis.md / stage_S3_amendment_review.md / stage_S3_aggregates.csv；stage_S4_analysis.md / stage_S4_A2_review.md / stage_S4_A2_execution_report.md / stage_S4_A2_outputs/**。

---

## 6. 关键锚点数字

只保留会改变研究判断的锚点。除标明外，差值为 AUC 原始比例单位；跨报告的口径、paired n、指标不能互换。

| ID | 条件 | 指标 / 对比 | 核心数字 | 科学意义与边界 | 主要来源 |
|---|---|---|---:|---|---|
| A1 | MovieLens，as-published | 复现 AUC | 0.7686±0.0017；论文 0.7702 | baseline 复现成立；旧协议 | P0_repro.csv |
| A2 | P2，freeze 0.7311 | vs 原版 | +0.0007，p=0.32 | 旧 AutoML 无可测贡献 | P2_baselines.csv |
| A3 | P2，修 B10 | 5-seed SD | 约 0.0017→0.0002 | bug 主要放大 seed 方差 | P2_baselines.csv |
| A4 | P1，MovieLens | 小域/大域 SNR | 0.509/0.558；小 batch 差消失 | 小域更噪声大假设不成立 | p1/D04_grad_stats.csv |
| A5 | ML P5，ratio 30/100 | comp1−baseline | +0.0166 (p=.0002,d=5.80) / +0.0202 (p=.0061,d=2.37) | 旧强效应真实，但该下采样同时减少总 N | results/p5/P5_raw.csv |
| A6 | ML P5，六档 | comp1 差值序列 | +.0064, +.0034, −.0012, +.0053, +.0166, +.0202 | 不单调；3:1 为待复核负凹陷 | ML_P5_REPORT.md |
| A7 | KuaiRand P5，大域 local d1 | baseline Δoracle t0/t1 | −.070659/−.099016 | 严重负迁移基线 | P5_oracle.csv、P5_raw.csv |
| A8 | KuaiRand P5，comp1 vs baseline | 大域收窄 t0/t1 | +.009102 (p=.0228) / +.017077 (p=.369) | 只有 t0 名义显著；均未追平 oracle | KUAI_P5_MAIN_REPORT.md |
| A9 | P0-1，固定 N 角色轮换 | Large-role Δtransfer | −.014557±.007439 | SD 取自 P0-2 同配置；现象随角色移动 | P0_1 raw/review |
| A10 | P0-2，N 固定 ratio=50 | Large-role Δtransfer | −.015545±.010894 | ratio 效应在固定 N 下仍存在 | P0_2 raw/review |
| A11 | P0-4，Equal→Sample | Large loss damage 修复 | 约 92% | 特定分解描述值，不是中介率 | P0_4 headline/review |
| A12 | P0-5，Sample+comp1 | 残余 gain | +.005407±.000620；p=.00041；12/12 配对正 | sample 下还有小增量；69% 是描述性拆分 | P0_5_STAGE_REVIEW、audit_statistics.json |
| A13 | P1-2，forward curriculum | B0−A0 AUC | +.000839±.001018（seed-block n=4）；p=.198；CI 跨 0 | Curriculum 在 sample 口径未通过 KEEP | P1_2_STAGE_REVIEW |
| A14 | E1/E3，Equal→Sample，ML r50 | 六 cell selected AUC（12 blocks） | +.021085±.009768；E3 端点 bit-exact | 强目标差异证据；E1 机制本身仍观察性 | stage_E1/E3_summary.csv |
| A15 | E4，KuaiRand | train-test shift | mixture TV=.112839；d1 click rate .378→.211 | temporal shift 是强竞争解释 | stage_E4_dataset_audit.md |
| A16 | E5，预设 schedule | Forward 相对静态/controls | 对静态 Sample 仅 1/4 seed 正；GO gate 9 项 3 项过 | 所测 schedule STOP，不代表所有时间机制 | stage_E5_go_no_go.md |
| A17 | S0，Sample，comp1 vs C_val | selected macro AUC | −.003033±.000815；0/4 seed 正 | residual 真实，但非 domain-conditioned 独有 | stage_S0 paired/analysis |
| A18 | S1，Sample−SD reference | 组级结果 | 63/72 均值正；0/72 stable practical NT | 普遍 residual NT 主线 STOP | stage_S1_paired_aggregates.csv |
| A19 | S2，KuaiRand tab1 click | Equal−SD selected AUC | −.045006±.000602；4/4 seed 负；5/5 common epochs 负 | 最强条件性 NT 靶点 | stage_S2_paired_results.csv |
| A20 | S2，ML d2_large rating>3 | Equal−SD selected AUC | −.018093±.004687；4/4 seed 负 | 第二靶点；后期反转限制机制外推 | stage_S2_paired_results.csv |
| A21 | S2，全部目标条件 | selected practical NT | Equal 8/24；完整严格 gate 3/24；Sample 0/24 | objective 条件性；不是普遍 NT | stage_S2_analysis/paired_results |
| A22 | S3，KuaiRand Equal tab1 | target-only−FULL AUC | click +.038876±.003950；long-view +.039576±.004107 | 4/4 同向；selected 条件 | stage_S3_aggregates.csv |
| A23 | S3，同条件非目标域 | target-only−FULL AUC | −.017040 平均；14/16 为负 | rescue 是性能搬移风险，不是全局改进 | stage_S3_aggregates.csv |
| A24 | S4，KuaiRand d1 | norm臂 N−FULL | −.004048/−.004873 | raw norm match 未救回目标 | stage_S4_A2_outputs/stage_S4_aggregates.csv |
| A25 | S4，KuaiRand d1 | applied-step 臂 U−FULL | −.000587/−.000942 | 即时 Adam 步长 match 也未救回目标 | 同 A24 |
| A26 | S4，d1 target-only−N/U | selected AUC | +.042924/+.044448 click；+.039463/+.040517 long-view | 方向/组成或 moment history 更符合，但两者未分离 | stage_S4_A2_outputs/stage_S4_aggregates.csv |
| A27 | S4，非目标域 | d0 selected / d2 final | d0 −.031386/−.032142；d2 +.019810/+.036160 | 训练阶段和 domain-specific cost 必须报告 | 同 A24 |
| A28 | P7，D=3 backbone transfer | comp1−none | MMoE +.002978；PLE +.005538；ShareBottom +.004012 | 限 ML；ShareBottom 为本地适配；非 S3/S4 泛化证据 | P7_summary.csv、P7_generalization.md |

---

## 7. 结论演化与 Superseded Findings

| 早期认识 | 后续证据 | 当前裁定 |
|---|---|---|
| 小域梯度噪声更大，因此需用 per-domain EMA 补偿小域 | P1 SNR 在两 batch 条件下无稳定域差；P6 的 SNR 反推 γ 显著差于公式 γ | **SUPERSEDED — 不得用“小域噪声更大”作为当前机制前提。** 组件可能有效的历史事实与原动机需分开。 |
| comp1 在 MovieLens 强不均衡下收益巨大，收益随 ratio 增大 | 六档序列非单调；r30/r100 同时缩小 N；固定 N 的 Px_n 确认 imbalance 有贡献；P0-5 又发现大部分增益补偿 Equal mismatch | **PARTIALLY SUPERSEDED — 强档增益现象保留；单调性和单一 size 机制归因失效。** |
| comp1 是大域保护 / Pareto 改善方法 | KuaiRand 仅名义改善大域 t0，t1 不显著，仍远低于 oracle；固定 N 与 objective 对照改变解释 | **SUPERSEDED — 旧证据为前沿移动，不是真 Pareto 外推，也不是当前保留的方法结论。** |
| M3oE 特有的 Largest-domain sacrifice | 负值随 Large role 移动；MMoE 也复现；Equal/Sample 操纵改变方向 | **SUPERSEDED — 不能称为 M3oE 固有，也不能称为不依赖目标的普遍 trade-off。** |
| comp1 在修正目标后仍是独立的 domain-conditioned 机制 | P0-5 残余 +0.0054 新鲜复现；S0 对均匀平滑 family 的 KEEP gate 失败 | **SUPERSEDED — residual effect 可是真实，但 domain-conditioning 归因已 STOP；普通时序平滑足以解释。** |
| Sample 目标下 residual NT 候选证明 Safe-MTL 仍有广泛问题 | S1 以匹配 SD reference、backbone、数据与 common stage 检验，未达到任何 stable practical NT 门 | **SUPERSEDED — 普遍 residual NT 主线 STOP。** S1 的负候选不得脱离其阈值和 reference 方差单独引用。 |
| “negative transfer 在此任务不存在”可从 S1 推出 | S1 只用 Sample；S2 用相同 exposure 下的 Equal/Sample 对比发现 Equal 特定 Large cell 退化 | **SUPERSEDED — S1 结论限 Sample；S2 证明目标条件化现象，两者不是同条件矛盾。** |
| 负迁移来自域难度，或 Large 受损是普遍“救小牺牲大” | S2 effect 随 size role 关联，难度相关弱；Sample 使多个域同升；AUC/LogLoss 与 stage 有反例 | **DOWNGRADED — size role 是定位标签，独立 size 因果未识别；“小域受益、大域付出”不成立为普遍机制。** |
| Target-only rescue 证明 shared gradient conflict，target-only 可能是方法 | S3/S4 识别 shared-access intervention；非目标域明显受损；norm 与 applied-step 控制失败但 Adam history 未拆 | **PARTIALLY SUPPORTED / DIAGNOSTIC ONLY — access effect 有因果证据；纯几何冲突未证明，target-only 不可作安全方法。** |
| E3 的 early/late α 诊断表示 schedule 会优于 Sample | E5 预注册 forward/reverse/exposure-matched 对照未通过 GO gate | **SUPERSEDED — E3 是诊断性线索，所测 schedule 被 STOP；不得把早晚选择相关直接改写为有效时间干预。** |
| adaptive calibration、EMA 或 curriculum 可补足静态 Sample | P1-2、E5、E6 均未达到预设 KEEP/GO；E7 STOP、0 runs | **STOP / DROP — 这些当前路线不能重新包装成已验证方法或新贡献。** |

### 尚无最终机制裁定的主张

没有发现最新 S3/S4 与 Reviewer 裁定之间存在需要在两种结论中择一的未裁冲突。仍开放的是**因果不可辨识**而非相互矛盾：S4 的救援可由梯度方向/组成、其诱发的 Adam moment history 或二者共同作用解释；当前数据不能单独判定。域大小与份额/有效权重/多样性共变也未拆开。E3 late-phase 选择计数因 phase 切分口径出现 10/12 与 11/12 的差异，方向相近但精确计数应回源，不应把任一数字写成普遍事实。E2 未执行使“objective 系数 vs sampling exposure”仍缺因果拆分。

---



### 现代历史交接与源状态

固定 EMA、PUB、两态读出、NDP-A 与 C01–C05 的当前裁定见对应 EXP 章节。逐项保留历史交接的可定位证据。[原交接与状态](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/EXP_AI_CONTEXT.md) · [C05 研究报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)。

## 快照独有的 C05 记录

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

KR4/5 AUC为正而均值为负：seed2024差−0.009762547必须保留；不删异常seed。U−W KR d0t0 LL方向改善而AUC变差，F3不成立。E50小正AUC与selected负并存。KR较小d0与最小d2反向，ML最小域baseline最好，不能小=难或宣称普遍小升大降。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

在当前λ=.25、源M3OE/Adam/固定EMA/自然划分/50epoch协议下，CELL-MASS-BCE-025不进入候选继续通道。此为限定协议的失败，不是证明所有cell质量方法无效；KR总体AUC效应仍不确定。不得以挑seed、重选终点、自动加lambda搜索来改写结论。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

证据等级：筛选不通过的协议事实STRONG；总体AUC方向WEAK（CI跨零）；移除类内政策损失KR校准收益STRONG，ML幅度MEDIUM；规模主因WEAK。正式物理17333次返回vs17250有用更新；工程33vs32。缺失terminal的PID2982386成本未知仍null，不能以已知pending求和0推断全历史0。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

C05：U唯一新增arm；ML+KR×5seed×50epochs；DESIGN/IMPLEMENT/独立代码REVIEW/工程及资源REVIEW/RUN/主ANALYZE完成。父RAM验收20选中模型/500ACK，18CSV含28900配对seed及5780汇总。报告与归档进入最终证据复审和发布收尾，交付后停止。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

当前最新候选未通过主筛选，不能用E50正点均值或U_ema−C_online优势替换matched EMA selected主终点。当前验收与发布完成后执行约定停止。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

F2方向未通过：U−S selected EMA task1 LL为ML−.000733901、KR+.000712949；仅cell/task质量不足复现W−S双数据校准方向。F3未通过：KR d0t0 U−W AUC−.004384878而LL−.001922553。长期政策差异包括Adam/EMA/轨迹，不能给出中介贡献比例或普遍必要性。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

最有鉴别力的未来实验：同domain内独立操纵类别率与样本量、固定配对seed和训练预算，并预设baseline难度分层；另以全新独立seed检验KR轨迹分叉。共同U状态中S−C梯度共线但Adam更新不同，需独立消融才能区分历史、耦合WD、eps。以上均为建议，本次不执行C06。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

### C05 CELL-MASS-BCE-025 — 2026-11-18 完整五seed证据

10条新U×50epochs已全部完成；复用C/W/S数值30条，共40条唯一轨迹，配对n=5（2022–2026）。selected EMA U−C AUC：ML−0.000221160，CI[−0.000624936,+0.000182617]、1正4负；KR−0.000235705，CI[−0.007145879,+0.006674468]、4正1负。ML/KR LL差+0.000320050/+0.000576420。结论NO_CONTINUATION_SIGNAL：两套AUC均未达+.003，KR d0t0差−0.007526250未通过−.001保护线。Test仍封闭，无champion。

STRONG：U在E1两数据AUC均5/5下降且区间全负；U−W的KR Macro/task1 LL均5/5变差且区间全正。两数据U−W task1均5/5变差，但ML区间跨零，ML幅度证据MEDIUM。W−S task1的原有双数据5/5下降保持。

完整报告：[C05 REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/cell_mass_bce_20261117/REPORT.md)；方法、逐seed/cell/stage数值附录、成本及失败恢复见同目录。冻结source55=f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。父直接验收500ACK/20模型/95508960字节；C04已丢失模型仍不可用，仅复用数值，不重建。按用户最终指令本轮交付后停止，不开C06；未来实验建议不自动执行。

## Q1–Q53 问题去向

以下保留每个问题的提问、历史状态/结论、冻结裁定、边界和反证字段。发现同题冲突时按 ID 定位。机器可读原字段见 [源 JSON](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)。

### Q1 · Objective-equivalent sampling 是否真的改变 training trajectory？
- historical status · RESOLVED
- historical conclusion · Objective-equivalent q changes can produce seed-consistent early diagnostic trajectory performance in the frozen fixed-reference Adam.01 configuration: commonpre1..27 AULC mean-.006980, fournegative, CI[-.011226,-.002734]. Not a selected-only or late-only observation in that configuration.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · N1与N5独立确认完整H100；KR selected主Holm均未通过，但Adam早期AULC8/8负，固定ML selected两种调参估计对象均8/8负。条件轨迹效应保留，非普遍有害。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不能将有限配置推广为所有optimizer/人口/阶段。
- next discriminating step · 新假说须冻结其配置与评价对象；不重跑旧存在性审计。
- sources · ["FINAL_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q2 · 是否只是 checkpoint-selection artifact？
- historical status · RESOLVED
- historical conclusion · RESOLVED/MEDIUM：selection会改变符号、大小及两臂健康解释，但不是所有q效应的唯一来源。冻结test固定SGD早期−.001430及共同E50−.001405四负，common-SN/common-SB AUC选轮分别−.002747/−.001380四负；校准Adam早期−.009821及共同选择−.011520/−.010323四负。calSGD早期+.001614四正却AUC自选−.000972四负，ML早期+.003015四正、E50−.000986四负、自选均值+.000539仅1正3负，显示规则/阶段/人口边界。全部四规则、1000重选区间与共同轮次分别归档，不把条件选择区间当外层调参或pristine holdout不确定性。
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · N0固定同预测权重分解及N1固定轮/AULC与selected并列；ML own E100正而selected负，selection改变大小/符号但不能解释所有早期效应。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不同评价人口与选轮不是同estimand。
- next discriminating step · 独立预注册ML完整q×两LR外部预算，检验late作用。
- sources · ["METRIC_AND_POPULATION_DECOMPOSITION.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q3 · 是否依赖 Adam？
- historical status · FALSIFIED
- historical conclusion · UniversalAdamnecessityfalsified: fixedSGD.1 commonpre1..50 AUCmean-.00304265 fournegativeCI[-.00515485,-.00177798]; common20/fixed50/selected alsonegative. Adam notnecessary forANY q-dependent AUCtrajectoryeffect. LogLosscanimprove, noallmetricsworseclaim.
- status at 2026 10 18 · FALSIFIED
- latest review tag · HISTORICAL_FROZEN
- evidence strength · STRONG
- latest evidence · 原无动量SGD反例及FALSIFIED不改。新SGD1.0 selected宏CI跨±.003，不能消除旧配置存在性反例，也不能宣称所有optimizer等效。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 新配置主效应精度不足见Q53；普遍Adam必要性已由旧反例反驳。
- next discriminating step · 不重做通用必要性；新精度设计另登记。
- sources · ["FINAL_performance/primary_common_eta_Holm.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q4 · 若 optimizer-specific，是否为 m/v/history？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · Actualcalibratedinteractiontriggeredallmandatoryfour-seedwhole-stateandshadowcontrols; allfinalbudgets250/150/100/200 and39pairscompleted. Whole-state update mapping influencesearlygap, but fixed22SNdegradation, E50healthyabsoluteimprovementwithunstablegap, beta2floor and unmatchedlongrunbudgets prevent identification of unique necessary m/v/history mediator.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · WEAK
- latest evidence · 同θ/m/v/step的N2映射对比与固定分母完成；ALL对比/边际MC尺度约.48–.62，固定分母后不明确消失。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 未确认总体即时漂移，不识别长期历史必要性。
- next discriminating step · 仅在明确定义的新状态假说下预注册健康保留干预。
- sources · ["N2_FIXEDPOINT_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q5 · 非 Adam 机制是 exposure/repetition/coverage/order/geometry 中哪一个？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · FullWR/PERM/ORDER all4Adam/SGD controlsandfinalrawsampler/coverage/weight auditscomplete. FixedE,N enforcesR=E-U>=max(0,E-N); SBsmallE>N preventsfixedquota/supportzero-repeatcontrol. PERMchangescoverage/dependenceandretainsrepeats; ORDERjointlychanges tailclock. CannotassignuniquenecessarynonAdammediator.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · MEDIUM
- latest evidence · 固定quota的域计数份额小，q trace和全程q性能差保留；N3局部作用小。不同路径仍可共同解释。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · quota改变批内依赖/高阶矩，不唯一分解order/repetition/coverage。
- next discriminating step · 选择一个可操纵剩余解释另立估计对象；不强求唯一中介。
- sources · ["DOMAIN_COUNT_COUNTERFACTUAL.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q6 · shared/private parameter 是否必要？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · All12minimallocalcontrolscomplete: CpositiveAUCscreenE10/E40, E40loss/Jconflictpreventscompletefrozengate; conditionalfulltrainingNOT_TRIGGERED. Currentlocalpathwayeffectnotunique/long-termnecessity.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · MEDIUM
- latest evidence · N3双路径/多draw及独立seed完成；embedding/E10四项MC达标但实际/方向门槛失败，N4未触发。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 未运行长期拓扑，不是拓扑阴性或共享必要性证明。
- next discriminating step · 保持长期未知；新拓扑假说需新的独立局部/长期依据。
- sources · ["C_LOCAL_AND_TOPOLOGY_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q7 · domain size 是否解释 effect？
- historical status · BOUNDARY
- historical conclusion · Under fixed p,q,N example budget and WR sampling, uniform exact aliases change nominal pool size but not empirical input/label distribution, gradients, optimizer updates or trajectory law. Nominal alias count is not a sufficient explanation of SB-SN.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · WEAK
- latest evidence · N0及新模型按N、baseline、标签率、task和阶段分解；严格小域双赢/大域损伤不稳定。没有新真实支持N干预。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · N、p、输入/标签、自然难度共变；原alias大小结构边界保持。
- next discriminating step · 真实独立支持量试验须另定义J；复制行不作为新支持。
- sources · ["METRIC_AND_POPULATION_DECOMPOSITION.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q8 · domain difficulty 是否解释 effect？
- historical status · BOUNDARY
- historical conclusion · All4severitycomparisonsatmatched50/200/50/350complete. Targetd2SNmeanAUC decreaseswithseverity, butE50mildSNlossimproves andqgapnonmonotonic/stagedependent. Onefixednestedmaskper split, changesfeaturelaw/J_c, notpurelatentdifficultycause.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · WEAK
- latest evidence · 新结果不把小域等同困难域；ML最小域下降与最大域略改善，baseline不随N统一单调。原mask敏感性仍限定corruption。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · N5-D未触发；自然难度唯一因果仍未知。
- next discriminating step · 只有明确难度因果主张才多mask实现/独立optimizer seed。
- sources · ["FINAL_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q9 · task/domain imbalance 是否解释 effect？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · Threephysicaldomains jointlyconfound input/labelpopulation,size,positiveprevalence,p,baselineperformance andmodelpaths; q isintervenedbutthesefactorsarenotindependentlyassigned. Trainingseeds donot create new domainpopulations.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · WEAK
- latest evidence · N2限定计数波动，N0/N1/N5报告人口与任务；同一自然域N下两task可异向。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 三个自然域不分离N/p/标签/输入/baseline，seed不增加域因子水平。
- next discriminating step · 独立分配候选因子，不做三个域的高自由度因果回归。
- sources · ["FINAL_domain_context/domain_task_baseline_population_context.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q10 · validation selection 是否改变结论？
- historical status · BOUNDARY
- historical conclusion · AllfrozenAUC/loss/early/fixedselection andcommonSN/SBchosenrulescomplete. Selectionmodifieseffectsize/sign andpopulationQ32; fixed/pre effects remaininconfigurations. Frozennewtest confirmationseparatelyQ2.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 固定E10/20/50/100、selected、AULC与三权重均保留；ML选轮约11–21轮，E100 own正而selected负。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 选择人口最优与test/不同权重排名不可互换。
- next discriminating step · 保留预注册主estimand及全部反向读数。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q11 · 是否跨环境成立？
- historical status · BOUNDARY
- historical conclusion · Requiredsecond-environment4seedSGD.3same-objective qcomparisoncomplete, no stableadversemacroprimaryreplication. MLearly1..16+.0013913positive1negativeCI[-.000131,.003849], selected+.000241CI[-.001222,.001127]. KRcalSGDatjoint16+.000078also3positive1negative, so ownwindow37/16oppositesignsnotisolateddatasetflip.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 固定subset2022的ML八optimizer seed完成，selected两估计对象均8/8负，与历史耦合subset/SGD配置不是相同实验。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 两环境bundle与进度共变；只识别该固定ML人口下稳定性。
- next discriminating step · 需要单一环境driver才做同框架单因子改变。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q12 · SAF/D3/AdaTask 是否覆盖？
- historical status · BOUNDARY
- historical conclusion · D3/SAFfulllearnedweightstargetchange(STRONGformulaboundary). TaskAdamcomponentatfixed.003doesnotremoveq-sensitivity(MEDIUMscopedempirical). Genericimportance-samplingvariance/speed,order/memoryandreshufflingalreadyhavepriorcoverage. Neitheruniversalcoverage nornewMDMTmechanismidentifiedbythesealone.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 八篇原文逐公式对照完成；D3/SAF一般改变J，AdaTask组件旧结论不冒充完整方法，EMS/MIS覆盖通用采样思想。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 有限检索不证明新颖性穷尽，也未训练完整方法性能基线。
- next discriminating step · 具体方法候选出现时再公平完整baseline。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q13 · 是否存在未被已有工作覆盖的 mechanism gap？
- historical status · BOUNDARY
- historical conclusion · Generalobjective-equivalentsamplingcovariance,orderandhistorythemes overlappriorwork. CurrentMDMTconditionalCPREpathway/optimizerboundaries donotidentifyunique necessarymediator oruncoverspecificnovelmethod. Ninesourceauditnotexhaustivenessproof.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · WEAK
- latest evidence · 本轮提供q×LR×预算/人口及固定点反证，未识别独有通用机制/新算法；N3长期门槛失败。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 场景差异不保证创新；仍有非唯一未排除解释。
- next discriminating step · 把下一具体预测与原文假设逐一对照。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q14 · 是否值得方法化？
- historical status · RESOLVED
- historical conclusion · 十项GO要求未同时满足：明确配置有共同早期和非普遍selection-only证据（MEDIUM），并得到测试确认；跨环境存在可描述边界，局部C和整体状态映射有条件反事实作用。但唯一长期必要中介、对速度/coverage/repetition/short-tail clock等替代的独立排除、独特现有工作缺口均未成立，也没有本轮新方法在真实primary endpoint、四seed稳定且所有域任务/原目标无损的完整证据。C局部AUC阳性和loss/J冲突如实保留，未触发原完整联合训练门槛；不以宽CI证明零效应，不从test生成候选或调参。NO_GO只关闭本轮方法化，不宣称所有未来方法失败。
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 旧NO_GO保持；新轮条件GO-EFFECT、固定点限定GO-MECHANISM分开。所有数值读数无宏收益与七项保护联合通过。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 未触发新候选训练不能推断未来方法全失败；E100 own保护失败是区间精度。
- next discriminating step · 有冻结候选、主收益、保护、成本与文献依据才方法化。
- sources · ["FINAL_method_guards/numerical_benefit_guard_summary.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q15 · SN/SB 的“same objective”在代码和 empirical gradient 上是否严格成立？
- historical status · RESOLVED
- historical conclusion · Same sample objective and same established graph-gradient expectation identity verified; no implementation mismatch detected
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 原同图同目标身份保持；新quota实际批域系数n*w/B=p全部核验，新增数值身份验证修正不改估计器。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · routing detach字段非普通总导数；quota依赖不同。
- next discriminating step · 不重复旧审计；保持本轮公式及来源。
- sources · ["N2_QUOTA_IID_PAIR_IDENTITY.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q16 · 现象是否主要发生在 overfitting 之后？
- historical status · RESOLVED
- historical conclusion · A late-only interpretation is contradicted in both frozenAdamconfigurations undertheprespecifieddiagnosticdegradationclassifier: fixed.01commonpre27AULC-.006980 andcal.003commonpre22-.013951, each4/4negativewithpairedseedCIexcluding0 andfourworseLogLoss. Theseearly common-epoch effectsdonotdependonindependentbestcheckpointselection.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 外部固定AULC1..20 KR Adam8/8负；固定E10/E20也负。没有以候选失败重新缩短early窗口。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不以onset未检出证明无过拟合，不证明所有阶段有害。
- next discriminating step · 主早期窗口始终外部固定，时钟与LL同时报告。
- sources · ["FINAL_STAGE_MACRO_AUC.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q17 · effect 是否只是 learning-speed / finite-budget effect？
- historical status · BOUNDARY
- historical conclusion · AllV4LR/batch/requiredextendedbudget controlscompleted. EffectdependsLR/stage/metric/budget: calibratedB0seed25gapchanges-.009424E50to+.000713E100, final4seedendpointmixed; Adamselectedearlynegative andB2selected4negativepersist. Neitheruniversalspeed-onlynoruniversalpermanentgap established. exact1225stepinteractionmixed/uncertain; B0/B2equalepochschangeupdatesteps, equalstepschangedexamples.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 完整H100与所有端点支持阶段依赖；ML own早期负、E100正；H200原rho门槛均未触发。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 没有证明永久差异/纯速度；匹配J是处理后描述。
- next discriminating step · ML完整q×两LR新seed矩阵，以共同外部预算证伪纯LR解释。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q18 · 在相同 sample multiset 下，仅改变 optimizer-step 顺序是否改变结果？
- historical status · RESOLVED
- historical conclusion · Exactbatchmultiset/ordercontrols complete all4final50/50/50/350. E50ORDERqinteraction+.0058155all4positive, finalmean-.0067918 due2025interaction-.0419133. Stage/selection-sensitivepath effect established; notcurriculumoruniformabsoluteimprovement.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 原same-multiset ORDER结果保持；本轮未作新的纯内容顺序或课程策略主张，N5-A未触发。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 原ORDER同时移动短尾，纯内容/size-clock仍未分离。
- next discriminating step · 若顺序成为核心，尾固定等长置换与尾移动分别比较。
- sources · ["CONDITIONAL_BRANCH_DECISIONS_PRETEST.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q19 · repetition 是否是 exposure effect 的必要组成？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · FullWR/PERM/ORDER all4Adam/SGD controlsandfinalrawsampler/coverage/weight auditscomplete. FixedE,N enforcesR=E-U>=max(0,E-N); SBsmallE>N preventsfixedquota/supportzero-repeatcontrol. PERMchangescoverage/dependenceandretainsrepeats; ORDERjointlychanges tailclock. CannotassignuniquenecessarynonAdammediator.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · STRONG
- latest evidence · R=E-U及E>N时零重复不可能的结构边界保留；本轮quota未伪装成阻断重复。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 固定E/N/支持下不能独立零重复；N5-B未触发。
- next discriminating step · 研究新的可行时序estimand或真实独立支持，另立问题。
- sources · ["CONDITIONAL_BRANCH_DECISIONS_PRETEST.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q20 · coverage 是否独立于 exposure/repetition？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · At fixed physicaldomain support N and exposure E, cumulativecoverage U/N and totalrepeats R=E-U are deterministic transforms, so currentWR/PERM cannot identify their independent scalar mediation. PERM changescoverage/repeats jointly; ORDER preservesboth andchangesorder.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · STRONG
- latest evidence · 固定E时unique与repeat代数绑定；quota与N3均未宣称独立coverage中介。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 多seed不修复结构不识别。
- next discriminating step · 仅同E/U/R的间隔或真实支持设计可另回答限定问题。
- sources · ["DOMAIN_COUNT_COUNTERFACTUAL.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q21 · estimator covariance 的 anisotropy 是否比 scalar variance 更有解释力？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · Shapehasdescriptiveinformationbeyondtrace, but independentdownstreamexplanatorypower/necessitynotidentified. qchangesbothandotherfactors; repeatednoisebatchesnotindependentYtrials. V7blockandV11single-stepnormcontrolsnotdistribution-tracecontrols.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · WEAK
- latest evidence · N2真实trace分解及跨块差异完成，但未作同均值trace匹配shape干预。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不据q方差/局部AUC推出shape优于trace；该主张未承担结论。
- next discriminating step · 仅shape成为核心命题时预注册受控残差库。
- sources · ["CONDITIONAL_BRANCH_DECISIONS_PRETEST.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q22 · shared C/R 是否是必要 mediator 而非相关参数块？
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · All12minimallocalcontrolscomplete: CpositiveAUCscreenE10/E40, E40loss/Jconflictpreventscompletefrozengate; conditionalfulltrainingNOT_TRIGGERED. Currentlocalpathwayeffectnotunique/long-termnecessity.
- status at 2026 10 18 · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- latest review tag · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- evidence strength · MEDIUM
- latest evidence · N3整体C/子块、双路径、四格及独立确认已完成；共享拓扑门槛未触发，R仍按既有私有支持处理。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 局部更新替换不等于共享C/R长期必要性。
- next discriminating step · 保持N4未触发而非阴性；新必要性问题另立。
- sources · ["C_LOCAL_AND_TOPOLOGY_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q23 · optimizer effect 与 sampling effect 是否存在真实 interaction？
- historical status · RESOLVED
- historical conclusion · CalibratedLRspecificcommonpre22 interactionTRIGGERED mean-.01391006 all4negativeCIexcludes0; fixedLRreferencecommonpre27 NOTTRIGGERED mean-.00514906 3negative/1positiveCIcrosses0. Strong evidenceconditionalconfigurationinteraction, notuniversalAdamhistorycause.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 原特定配置optimizer×q interaction不改；新对称LR下Adam与SGD各条件效应及N3作用大小不同。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · optimizer和LR/state共同变化，不能唯一归于m/v历史。
- next discriminating step · 针对实际映射定义单因子控制，不重做旧整体state矩阵。
- sources · ["FINAL_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q24 · 第二环境失败时，boundary 究竟来自哪里？
- historical status · BOUNDARY
- historical conclusion · Explicitboundarydecompositioncompletedforobjective,optimizer,architecture,data/labels,imbalance/size,overfit/stage. Sameobjectiveform/SGDfamily/LR/M3OEfamilycontrolled, butfinitepopulations/labels/p/inputdim/updateclockco-vary. UniqueenvironmentdriverNOT_IDENTIFIABLEinexistingtwo-environmentdesign, no claimsize/difficultycausaldriver.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · ML固定支持新八seed完成；与KR并列labels、pop、N/steps及相同宏/加权读数，得到条件环境边界。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不是只改N或只改难度；旧耦合subset方差未分解。
- next discriminating step · 若要求唯一driver，在同框架只改该因子。
- sources · ["METRIC_AND_POPULATION_DECOMPOSITION.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q25 · existing literature 是否已经覆盖“same objective + different q + MDMT trajectory”机制？
- historical status · BOUNDARY
- historical conclusion · D3/SAFfulllearnedweightstargetchange(STRONGformulaboundary). TaskAdamcomponentatfixed.003doesnotremoveq-sensitivity(MEDIUMscopedempirical). Genericimportance-samplingvariance/speed,order/memoryandreshufflingalreadyhavepriorcoverage. Neitheruniversalcoverage nornewMDMTmechanismidentifiedbythesealone.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 原文对同目标采样/协方差/顺序/状态及MIS/EMS的直接覆盖已对照；本轮不将MDMT应用名称当创新。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 有限检索不是穷尽；新颖方法尚未被实测授权。
- next discriminating step · 未来具体主张逐公式/假设/估计器比较。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q26 · D3/SAF 的 objective change 是否使其不能作为 mechanism-equivalent prior work？
- historical status · RESOLVED
- historical conclusion · Published D3 DBO and SAF learned sample weights generally change the empirical target measure and are not mechanism-equivalent to objective-preserving SN/SB. Constant-weight degenerations disable their studied mechanism. This does not establish that either method cannot improve metrics.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 完整D3/SAF一般改变目标的公式边界保持，原文核对完成。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不推断其完整方法效果差，也不把退化常权版冒充原方法。
- next discriminating step · 只有性能问题需要完整等预算基线。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q27 · AdaTask 的 optimizer-state intervention 是否能消除 sampling effect？
- historical status · RESOLVED
- historical conclusion · Auditedsame-objectiveTaskAdamcomponentfixedLR.003 doesnoteliminateq-sensitivityinthisconfiguration: earlyjoint14gap-.00783937 3negative1positiveCI[-.014013,-.001666], actualfinalmatched50/100/50/50gap-.01916006all4negative. Selectedgapandmitigationinteractionsuncertain, nozero/equivalencestatement.
- status at 2026 10 18 · RESOLVED
- latest review tag · HISTORICAL_FROZEN
- evidence strength · MEDIUM
- latest evidence · 旧同目标AdaTask固定组件未消除q敏感性的RESOLVED原样保留；本轮未重训完整AdaTask。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 旧组件/LR结果不等于完整方法调优失败。
- next discriminating step · 有真实完整方法候选比较才公平对称调参。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q28 · routing 的 stop-gradient 边界
- historical status · BOUNDARY
- historical conclusion · Same expected loss values and same expected graph gradients are preserved; no claim this graph computes the ordinary total derivative through routing. Do not silently restore missing path.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · routing_input_detached及原forward始终保持，config/source SHA冻结。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 同图graph-gradient身份不等于恢复routing的普通总导数。
- next discriminating step · 移除detach属于新架构实验，不作为旧错误修复。
- sources · ["GLOBAL_MODEL_AND_PROTOCOL_FREEZE.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q29 · 参数块名称与实际共享支持
- historical status · RESOLVED
- historical conclusion · R/towers cell-private; slots domain-specific/task-shared; C/E tensor sharing across6cells. Correct necessity interventions accordingly.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · C/E共享、R单元私有及P混合的真实坐标支持沿用；N3参数计数和实际更新范数完整。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 张量名称、实际梯度支持与共享必要性不同。
- next discriminating step · 未来拓扑必须按真实支持设计。
- sources · ["N3_development_summary/parameter_counts.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q30 · Adam 学习率赢家位于网格下边界
- historical status · BOUNDARY
- historical conclusion · Finite-grid winner established by frozen rule; global LR optimality not identified and not claimed.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 新四点对称网格含低邻点；KR Adam .01/SGD1.0为上边界赢家，ML SB .0003为下边界赢家，均不扩网格。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 预设网格不证明连续全局最优。
- next discriminating step · 未来边界邻点属于新预注册调参，不改本轮结果。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q31 · pilot 与 finalist 种子重合
- historical status · BOUNDARY
- historical conclusion · All4 selected optimizer/pilotseed pairs have matching initialization/data/source metadata and exact20epoch draw prefixes in formal SN confirmation. Fresh processes do not supply independent tuning-validation seeds. Four-seed contrasts and CIs are conditional on frozen LR; fixed-reference results are retained separately.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 新调参2030/31与正式2032–39不重合；N3提名2032–35与确认2036–39分开，复用别名不计新n。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 仍是条件所选LR确认，不估计完整外层选择方差。
- next discriminating step · 保持分离与来源索引；不重做旧配置。
- sources · ["PRETEST_TRAINING_STAGE_CLOSURE.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q32 · 历史固定 SGD 数字变化来自哪里
- historical status · RESOLVED
- historical conclusion · The apparent fixedSGD v2-v3 selected-effect discrepancy is generated by changing evaluation population plus a small epoch-reselection component, with zero measured trajectory/numerical contribution. It is not a new training implementation failure.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 原fixedSGD读数差源于评价人口/小选轮差的结论保持；N0同预测三权重和新轮明确split。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不重新解释为旧训练数值错误。
- next discriminating step · 保持来源/人口/时钟，不重训旧轨迹。
- sources · ["METRIC_AND_POPULATION_DECOMPOSITION.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q33 · 只用 SN 调 LR 的估计对象
- historical status · BOUNDARY
- historical conclusion · Current claims are conditional commonLR sampling contrasts. No separate SB-optimal tuning or globally optimized SN/SB superiority is established or claimed.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · MEDIUM
- latest evidence · 同时给common eta q干预与等预算own估计对象；KR模型相同复用，ML ownSB .0003与common .001不同且均实际确认。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 有限网格性能不等于全局优化；own跨LR差不作纯q机制。
- next discriminating step · ML晚期优势以同LR新seed格证伪。
- sources · ["FINAL_COMPARISONS.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q34 · 一阶矩遥测混入二阶矩范数
- historical status · RESOLVED
- historical conclusion · 原exp_avg是sqrt(||m||²+||v||²)，8个实际Adam/AdaTask状态块独立核验后，最终254归档快照756800行全部刷新，compact1016组；序列化范数公式最大相对误差3.324e-11，原mixed与corrected m最大相对差99.973%。训练更新/预测/指标不受影响，无需重训。
- status at 2026 10 18 · RESOLVED
- latest review tag · HISTORICAL_FROZEN
- evidence strength · STRONG
- latest evidence · 原corrected_m_norm/v_norm修复及无训练改变结论保留，本轮不再全量刷新遥测。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 范数修正不识别状态因果。
- next discriminating step · 无需新增训练或工程审计。
- sources · ["原original_registry_entry证据链", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q35 · 稀疏遥测 lag 与真实 step lag
- historical status · RESOLVED
- historical conclusion · Cross-epochtelemetrylags measuregapsinobservations, not actualoptimizersteplags. Existinginitial50readoutnowusesonlybatch>=lag andcounts5/4/1 within6firstbatches. No performance/trainingchange. No causalconclusionhadused unfilteredlags.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 本轮保留真实optimizer step与锚计数；N3每draw从同状态重置，不将draw序列称连续记忆lag。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 没有新的连续步相关性/长期记忆证明。
- next discriminating step · 仅有具体时间相关假说才预设连续步窗口。
- sources · ["N3_DESIGN_ADDENDUM.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q36 · onset 程序不提供全族错误率保证
- historical status · BOUNDARY
- historical conclusion · Theprespecifiedclassifierisreportedasanoperationaldefinitiononly. No simultaneousFWER guarantee orproof ofabsenceofoverfit. Frozenbudget/thresholdsunchanged; earlyfixedhorizoncontrastsremainvisible.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 新主端点采用外部E10/20/50/100与AULC，不作全局onset/FWER保证。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 原操作性onset不是过拟合不存在证明。
- next discriminating step · 只有提出全局onset命题才校准同时程序。
- sources · ["FINAL_READOUT_PRETEST_REVIEW.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q37 · IID 批次置换与课程策略
- historical status · BOUNDARY
- historical conclusion · Equal-cardinalityIIDnondatadependentpermutationhasinvariantjointlaw, butactualwholeORDERincludesone shorttail(Q38), so theorem cannotestablishfullproductiondistributionequivalence. Existingexperimentvalidlyreadsconditionalrealizedfull-batch-set sensitivity only.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 本轮没有数据依赖课程策略实验；原随机ORDER条件路径结论保留。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · IID等长置换同分布不能证明课程收益；尾批因素原样保留。
- next discriminating step · 若策略成为目标另预注册独立确认。
- sources · ["CONDITIONAL_BRANCH_DECISIONS_PRETEST.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q38 · ORDER 也移动短尾批次
- historical status · BOUNDARY
- historical conclusion · FinalORDER1000armepochs/964tailmovesverified (old286/300 +new678/700), all4200epochs205800batchweightsandORIGINALtail-last verified. Samecontentmultisetdoesnotisolatepurecontentfromsize-clock.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 原短尾移动964/1000事实保留，新完整训练均保留尾40052(KR)/23660(ML)。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 本轮没有分离纯内容与size-clock的新ORDER分支。
- next discriminating step · 需要纯顺序主张时尾固定/移动分别处理。
- sources · ["FINAL_TRAINING_RUN_LEDGER.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q39 · GPU 跨预算前缀数值复用
- historical status · RESOLVED
- historical conclusion · ActualGPU100->CPU50 full1000/200 arrays within1e-12; all4000epochchoices,onsets/selected/budgets exact. SourceRNG/epochseparableformula andactualcrossT support literal-parent-cache reuse; no approximate ornew bootstrap replicates.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 同配置父轨迹/别名复用不计新seed；每formal100纯model和7200预测模型SHA对应实际核验。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不推断所有CPU/GPU普遍bit相同。
- next discriminating step · 保持producer/model/pred索引，不重复prefix作为新证据。
- sources · ["TEST_READOUT_VALIDATION.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q40 · 状态消融的健康、幅度和学习进度
- historical status · BOUNDARY
- historical conclusion · UnderfixedLR.003 beta2zeroall4ownfinalbothqdiagnosticAUC=.5 withlargeLL/J; m-onlylearnsovertimebutlagsnativeAUC, whileLLcanimprove. Healthyv-onlycounterexample: fixedE50bothqabsoluteAUCimproves4/4yetgapinteraction3negative1positive; original22gapattenuation4positivechieflySNdegradation. Gapcollapsealonecannotidentifynecessaryhistoryorhealthyrescue.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 未重做破坏性beta2=0；N3使用原生状态的块norm×direction四格、实际范数与绝对AUC/LL/J。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · floor/gap修复不证明历史必要性；一步健康不代表长期。
- next discriminating step · 有新状态假说才健康保留长期控制。
- sources · ["C_LOCAL_AND_TOPOLOGY_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q41 · 自身早期窗口造成的环境表观翻转
- historical status · BOUNDARY
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 固定外部早期窗口且并列KR49步/ML5步每epoch；ML own早期负与晚期正的反例保留。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 同epoch、相同J或相同AUC均不消除环境bundle差异。
- next discriminating step · 未来明确外部steps/暴露estimand，不能同时匹配不同N的两钟。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q42 · 全局等范数没有隔离角度与块尺度
- historical status · NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN
- historical conclusion · Globalnormequivalence isonescalarconstraint anddoesnotfix4blockmagnitudes. Existinginterventions jointlychangeangles/blockscales andlackcrossedcounterfactuals; whole-state local effects identified, pure-angle contribution not uniquely identified.
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 逐块交叉幅度/方向四格和交互已实际测量，方向未定义预设处理；实际落地范数/目标误差保留。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 单个全局范数不能隔离块尺度；局部方向作用仍非唯一长期原因。
- next discriminating step · 本轮局部实际门槛失败，长期拓扑未知。
- sources · ["N3_confirmation_summary/actual_block_update_norms.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q43 · MovieLens 的 seed 同时改变训练子集
- historical status · BOUNDARY
- historical conclusion · MLfourpairedseedreplicates jointlyvarytrainingsubset andoptimizationrandomness, withfixedvalidation. Report uncertaintyoverthiscoupledpipeline, notpurefixedtrainpopulationoptimization. Same-seedobjective-equivalentqcomparisons remainvalid.
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 固定subset_seed2022及行/分半SHA，独立optimizer2032–39；同人口优化随机性新24formal完整确认。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 未交叉多个subset，不能分解旧耦合总体方差。
- next discriminating step · 只有子集方差仍用于主张时才3subset×4optimizer交叉。
- sources · ["N5_confirmation_validation.json", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q44 · 破坏性控制改变共同早期窗口
- historical status · BOUNDARY
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 本轮外部窗口不由失败arm决定，未以beta2=0等破坏臂缩短共同早期问题。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 保留旧破坏控制/窗口边界。
- next discriminating step · 始终比较固定外部时钟及绝对健康。
- sources · ["FINAL_READOUT_PRETEST_REVIEW.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q45 · 长时钟读数只有预算条件下的部分种子支持
- historical status · BOUNDARY
- status at 2026 10 18 · RESOLVED
- latest review tag · RESOLVED
- evidence strength · STRONG
- latest evidence · 新120条全H100，各正式组8seed均有E100；旧N0 E100只有1–2seed的资格不更改，H200未触发。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 旧非随机缺失不能补成四seed；未触发更长预算不等于阴性。
- next discriminating step · 每端点继续给实际n/模型支持集。
- sources · ["FINAL_TRAINING_RUN_LEDGER.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q46 · 局部C的AUC阳性与loss/J冲突
- historical status · BOUNDARY
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 新N3 full embedding独立AUC小正、LL/J多数改善，但方向残差J/LL不统一、实际门槛失败；旧C阳性及loss冲突保留。
- excluded explanations · 仅排除所链接报告中对应明确干预/算术可排除的解释；不扩张原历史科学状态。
- remaining uncertainty · 不把attenuation接近SN称绝对健康，也不倒改旧联合门槛。
- next discriminating step · 保留多度量/路径/阶段反例；长期拓扑未触发。
- sources · ["C_LOCAL_AND_TOPOLOGY_ANALYSIS.md", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE

### Q47 · 宏评价、train-p与test-p评价的反转由哪些cell贡献？评价权重变化与域内人口变化如何区分？
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · N0同预测权重精确反转；新ML own macro−.010875、train-p+.000744(区间在±.003)、test-p−.013586。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · 权重算术已识别；不同split域内人口因果仍未识别。
- next discriminating step · 如要求人口因果，固定权重并独立操纵/抽样域内人口。
- sources · ["METRIC_AND_POPULATION_DECOMPOSITION.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q48 · 短pilot选出的LR赢家是否只优化早期速度，且双方等预算调参后采样差距是否保留？
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · KR H10/H100共同LR赢家不同；独立确认全量预算；ML own E100+.021145(8/8正)而selected−.010875(8/8负)。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · 有限网格/外部预算/选轮不同，own跨LR不是纯q。
- next discriminating step · 新独立seed固定ML人口完整q×LR(.0003,.001)，证伪纯LR/进度解释。
- sources · ["Q_LR_BUDGET_RESPONSE.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q49 · 域计数波动相对于域内样本噪声贡献多少，固定域配额是否改变downstream采样效应？
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 固定点计数项.001297%–.105936%，quota后约2倍trace保持；全程quota q差−.009191但I_quota−.003988 CI跨0。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · 固定点排除计数主导；长期不是纯trace中介，绝对lift未确认。
- next discriminating step · 后续仅针对具体剩余依赖/分布解释，不能用gap变化推断纯计数。
- sources · ["DOMAIN_COUNT_COUNTERFACTUAL.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q50 · C局部作用来自哪些子模块、幅度/方向及其交互？共享拓扑的长期作用是否成立？
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · MEDIUM
- latest evidence · 开发一次提名embedding/E10，独立两路径full各4/4正约3e−5；方向/实际界失败且MC达标，不补64；N4未触发。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · pointwise实际界结论限当前状态；长期拓扑未知。
- next discriminating step · 保持未触发状态，未来新假说单独预注册。
- sources · ["C_LOCAL_AND_TOPOLOGY_ANALYSIS.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q51 · 在相同theta、m/v历史及step下，Adam瞬时非线性是否把不同梯度分布映射为不同平均更新？
- status at 2026 10 18 · INCONCLUSIVE
- latest review tag · INCONCLUSIVE
- evidence strength · WEAK
- latest evidence · 同状态原生Adam/固定滞后分母/单位eta仿射SGD全部计算；原生ALL q对比/边际MC尺度.479–.624，固定分母比例.911–1.054。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · 边际RMS不是CI，有限bank非零范数不确认总体均值漂移；历史必要性未知。
- next discriminating step · 只有该均值问题成为核心时预注册向量/投影精度及固定draw预算，不增draw追阳性。
- sources · ["N2_FIXEDPOINT_ANALYSIS.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q52 · EMS/variance-adapted LR与MIS/OMIS现有工作是否已覆盖最直接的候选解释和方法？
- status at 2026 10 18 · BOUNDARY
- latest review tag · BOUNDARY
- evidence strength · STRONG
- latest evidence · 八原文逐式覆盖IS/variance、bucket、memory/order、EMS及MIS通用方案；D3/SAF目标不同等边界明确。
- excluded explanations · 以对应最终报告中明确反证/算术界为限
- remaining uncertainty · 有限检索不证穷尽；无新方法candidate/完整性能baseline声明。
- next discriminating step · 具体候选出现时比较对应原工作与公平调参成本。
- sources · ["LITERATURE_COVERAGE.md", "FINAL_NUMERIC_FINDINGS.json"]
- executed status · COMPLETE

### Q53 · KR无动量SGD共同eta的selected宏AUC q效应在±.003实际界内是否足够精确？
- status at 2026 10 18 · INCONCLUSIVE
- latest review tag · INCONCLUSIVE
- evidence strength · WEAK
- latest evidence · 区间同时包含实际伤害−.003与收益+.003，当前不支持等效或确认改善/损伤；不加seed直至显著。
- excluded explanations · 不是缺失H100或调参seed复用造成的n不足；轨迹完整且调参/正式分离。
- remaining uncertainty · 小n优化随机性和seed尾部变异；固定test人口条件的区间也不估计自然人口泛化。
- next discriminating step · 下一轮先指定置信宽度/实际界、样本预算和分布稳健性分析，再做独立确认；当前闭合，不追加seed搜索阳性。
- sources · ["FINAL_performance/primary_common_eta_Holm.json", "FINAL_performance/paired_seed_endpoints.csv", "FINAL_ANALYSIS.md"]
- executed status · COMPLETE_CURRENT_ROUND

## 冲突登记

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Contradiction_Register.md)

> **2026-11-15 最新修订：** 历史矛盾/反例原文和适用边界继续保留；新增C052–C057承接11/09–11/11的跨报告差异。当前总体结论见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)、[Evidence Ledger](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)。不要将矛盾登记解释为所有相关研究方向已被否定。

> **2026-10-26 增量入口：** C001–C034是截至10-18的历史矛盾；新补C035–C042。各组须区分**同一估计对象真正矛盾**与**不同数据人口、训练目标、测试角色、比较臂或时钟的表面冲突**，并以 [当前跨轮综合](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md) 核对。不能为追求一致而覆盖原始负结果。



> **2026-10-18：** C001–C023 保留 10-09 历史冲突/反例；本文件新加 C024–C034。区分 **真正相同实验定义的数值冲突** 与 **协议/人口/时钟/选点/统计门不同产生的表面冲突**；同分支后续最终状态覆盖早期进度快照。入口：[跨分支整合](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)。


记录跨报告中的方向差异、后期修正或口径不一致。只有同一条件、同一指标、同一 checkpoint 才构成直接矛盾；不同阶段的结果大多是范围边界，不应强行合并。状态截至 2026-10-09。

#### C001 — comp1 的大幅历史收益 vs S0 匹配 uniform EMA

- **Evidence A:** P0-5/P6 在旧配置中观察到 comp1 正收益，见 E005。
- **Evidence B:** S0 comp1−selected uniform EMA macro AUC −0.003033 ± 0.000815，4/4 seed 均由 uniform EMA 胜出，见 E008。
- **Difference in setting:** 旧 P0/P6 与 S0 的 objective、exposure、训练流程和 control selection 不完全相同；S0 的 uniform EMA 是新匹配控制。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** 旧性能观察保留；comp1 的域条件化机制特异性被 S0 限定。不得把旧的 comp1-vs-baseline 数字与 S0 当作同一配对效应。

#### C002 — comp1 residual 曾被解释为小域机制 vs S0 未支持域条件化独特性

- **Evidence A:** P0-5 Sample+comp1 residual +0.005407 ± 0.000620，曾被解释为小域残余收益。
- **Evidence B:** S0 uniform EMA 在汇总 rotation 后优于 comp1；P1 SNR 也未显示稳定小域低 SNR，见 E001/E005/E008。
- **Difference in setting:** P0-5 是 comp1 × objective factorial；S0 针对 momentum identity 作匹配控制。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** Residual 现象不是同义于 domain-conditioned momentum 机制；统一平滑、选点与训练设置是竞争解释。

#### C003 — 大域角色退化 vs “domain size 是主因”

- **Evidence A:** P0 fixed-N role-rotation 报告中 L-role 有负 transfer，见 E003。
- **Evidence B:** S2/S4/S5/S6 结果受 objective、domain/task identity 和 checkpoint 影响；size 未被独立操纵，MovieLens d1 才是最小域，见 E010/E014/E020。
- **Difference in setting:** 早期 role rotation 改变域样本结构；后期 KuaiRand natural 与固定 target 没有 size intervention。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** 大域角色相关性存在于早期 fixed-N 条件；size-only 解释仍未识别。

#### C004 — Equal 下 selected loss vs Sample 下未见稳定 practical NT

- **Evidence A:** S2 Equal 在特定 Large cells（尤其 KuaiRand tab1）出现稳定 selected loss。
- **Evidence B:** Sample−SD 在 S2 为 0/24 practical losses，S1 为 0/72 stable practical NT，见 E009/E010。
- **Difference in setting:** objective、reference、target cell、dataset 与 checkpoint 口径不同。
- **Can the contradiction currently be explained?** Yes, by the objective and scope differences supported by the designs; no single mechanism is identified。
- **Current interpretation:** 不把 Equal 结果外推到 Sample，也不把 Sample 结果用于否认 Equal 下的局部 loss。

#### C005 — S3 的 Sample access 差异接近零 vs Equal 下强 rescue

- **Evidence A:** KuaiRand Equal target-only−FULL 为 +0.038876/+0.039576，4/4。
- **Evidence B:** 同一 S3 配对结构下 Sample access contrast 约 +0.0003/+0.0001。
- **Difference in setting:** Objective 不同；Equal/Sample 有效域权重不同。
- **Can the contradiction currently be explained?** Partially; objective dependence is directly observed, mediator is not fully identified。
- **Current interpretation:** Shared-access effect is objective contingent in the measured configuration; Equal mechanism must not be assumed under Sample。

#### C006 — T rescue vs norm-matched N/U no rescue

- **Evidence A:** S3/S4 T−F target AUC 约 +0.039、4/4 同向。
- **Evidence B:** S4 N−F 与 U−F 未复现正向 practical rescue，见 E013。
- **Difference in setting:** Same Equal target setting; control arms match different magnitude summaries while current-gradient composition/history remain different。
- **Can the contradiction currently be explained?** Yes, in the limited sense that the matched magnitude controls do not reproduce T; the remaining mechanism was later narrowed by S5。
- **Current interpretation:** raw-gradient norm and applied-step norm alone are insufficient explanations.

#### C007 — S5 KuaiRand DH≈F vs S5-ML MovieLens DH has a nonzero contribution

- **Evidence A:** S5 KuaiRand Equal d1 DH−F about +0.0009/+0.0008; HT follows T.
- **Evidence B:** S5-ML MovieLens Equal d2 DH−F selected +0.005859/+0.003583, with late common gains larger,见 E015/E017。
- **Difference in setting:** Dataset/domain composition, target physical domain, imbalance severity, task structure, and training-stage behavior differ。
- **Can the contradiction currently be explained?** Partially; these setting changes are documented, but the causal moderator is not isolated。
- **Current interpretation:** S5 history-dominant pattern only partially generalizes; DH≈F is not universal。

#### C008 — S6 Phase A STOP vs Phase B MovieLens late DH target gains

- **Evidence A:** S6 Phase A Sample F/T selected target effects are below +0.003 in both datasets,见 E018。
- **Evidence B:** S6 Phase B MovieLens DH−F target AUC at common e49 is about +0.0105/+0.0081; KuaiRand lacks the same late pattern,见 E020。
- **Difference in setting:** Phase A is F/T target-only contrast at selected/common endpoints; Phase B adds history/current policies and late checkpoints。
- **Can the contradiction currently be explained?** Yes as endpoint and arm differences; the late MovieLens effect does not overturn the Phase A selected gate。
- **Current interpretation:** Phase A STOP remains valid; Phase B provides a stage-specific optimizer-policy observation, not a practical selected rescue。

#### C009 — MovieLens final macro gain vs target-domain AUC/LogLoss and F overfit

- **Evidence A:** S6 final T−F six-cell macro AUC is positive in MovieLens.
- **Evidence B:** Target-domain selected T−F is below practical threshold; F target AUC and LogLoss deteriorate severely late,见 E018/E020。
- **Difference in setting:** Macro combines target and non-target; final/common checkpoints are far beyond F validation-selected epoch。
- **Can the contradiction currently be explained?** Yes, metric composition and checkpoint stage explain why aggregate and target outcomes differ descriptively。
- **Current interpretation:** Final macro gain is not target rescue and not a normal selected-model improvement。

#### C010 — S6 DH trajectory reanalysis reports long non-target runs vs S8 no better selected best

- **Evidence A:** S6 reanalysis finds MovieLens K=5 common checkpoints with non-target gain over threshold; selected non-target mean near zero,见 E022。
- **Evidence B:** S8 full-state best endpoint shows all MovieLens continuation bests below original F best, with selected-anchor contrast near zero,见 E026/E027。
- **Difference in setting:** S6 compares fixed checkpoints across full training; S8 continues from anchors and compares best continuation against frozen original selected best。
- **Can the contradiction currently be explained?** Yes; later S8 is the discriminating performance endpoint and does not erase the earlier fixed-epoch values。
- **Current interpretation:** The old trajectory effect is narrowed to late overfit rescue / “less deterioration,” not improvement of the model chosen by the normal protocol。

#### C011 — MovieLens negative U_k tracks decline vs KuaiRand negative U_k without decline

- **Evidence A:** S7-A MovieLens U_k becomes negative as target validation AUC declines.
- **Evidence B:** KuaiRand has negative target U_k snapshots while target AUC continues to rise,见 E024。
- **Difference in setting:** Dataset, overfitting regime, gradient norm evolution, and trajectory differ。
- **Can the contradiction currently be explained?** Partially; it disproves sufficiency but does not fully explain the dataset interaction。
- **Current interpretation:** Negative U_k is conditionally associated with performance in late MovieLens and is not a universal cause marker。

#### C012 — S7-A late gradient structure vs S7-B same-θ validation response

- **Evidence A:** S7-A reports late MovieLens harmful utility structure aligned with overfit degradation.
- **Evidence B:** S7-B changes that gradient utility at the same θ, but target AUC/BCE does not improve within the short window,见 E024/E025。
  - **Traceability correction (2026-10-09):** S7-B report overstated MovieLens anchor39 step5 AUC seed consistency as 4/4 negative. The raw CSV gives mean −0.001226, 3/4 negative, and seed2025 +0.002359; LogLoss mean is +0.011130, 3/4 worse. The no-stable-improvement conclusion remains, but the report and E025/SF008 have been corrected and evidence strength is limited accordingly.
- **Difference in setting:** S7-A is an observational 4-batch gradient probe across training; S7-B is a brief single-actual-batch intervention with a declared private fresh-Adam approximation。
- **Can the contradiction currently be explained?** Partially; different estimands and horizon matter, but the failed validation GO condition directly weakens the proposed mechanism chain。
- **Current interpretation:** Gradient metrics respond to intervention; their immediate correction is not shown to improve validation。

#### C013 — S7-B approximate private optimizer state vs S8 complete full-state continuation

- **Evidence A:** S7-B resets private parameters to fresh-Adam after step0 and flags an approximation。
- **Evidence B:** S8 restores full optimizer state including private m/v, with same-state, replay, and validation checks passing。
- **Difference in setting:** S8 adds complete state restoration, longer continuation, and comparison to original best。
- **Can the contradiction currently be explained?** Yes; S8 is the stronger later test for long-horizon performance claims, while S7-B remains a short-window gradient diagnostic。
- **Current interpretation:** S7-B limitations cannot be generalized to the S8 full-state result; S8 still finds no selected/best performance improvement。

#### C014 — S6 Phase B structural run success vs KuaiRand M1 gate failure

- **Evidence A:** Runner reported 16/16 Phase B runs structurally completed; code/update/pairing audit passed。
- **Evidence B:** Independent review found KuaiRand M1 m-conjunction passes 4/8 only; MovieLens passes 8/8,见 EXP028 source review。
- **Difference in setting:** Structural validity and mechanism-identification validity are distinct gates。
- **Can the contradiction currently be explained?** Yes.
- **Current interpretation:** Describe execution as valid and KuaiRand M1 as failed; do not say every scientific identification check passed。

#### C015 — S6 old “MovieLens d0 smallest” description vs domain counts

- **Evidence A:** Some earlier Phase A narrative called d0 the smallest-domain casualty。
- **Evidence B:** S6 full analysis checks train counts: d2=150,000, d0=34,500, d1=3,000; therefore d1 is smallest, d0 is middle。
- **Difference in setting:** Earlier text used a mistaken role label; the later analysis reconciles it against the configured counts。
- **Can the contradiction currently be explained?** Yes; this is a factual labeling error, not a new effect。
- **Current interpretation:** Current database uses d1 as MovieLens minimum and d0 as middle; old label is not reused。

#### C016 — exp_sn Phase B “not triggered” vs later completed extension

- **Evidence A:** `exp_sn.md` (dated 2026-10-06) says S6 Phase B had not triggered after both Phase A STOP gates。
- **Evidence B:** `stage_S6_phaseB_extension_scope.md`, run report, audit review, and final `stage_S6_analysis.md` document 16 new Phase B runs and analysis completed by 2026-10-08。
- **Difference in setting:** These are dated snapshots; the later extension followed Phase A STOP and was explicitly requested as a diagnostic continuation。
- **Can the contradiction currently be explained?** Yes temporally; the earlier statement was true at its date but is stale as current state。
- **Current interpretation:** Keep the Phase A STOP result; update the overall S6 record to include the later non-confirmatory diagnostic extension。

#### C017 — S8 late macro rescue vs “best performance improvement”

- **Evidence A:** MovieLens late DH−F macro final-step is +0.010732 (4/4), with middle/late gains。
- **Evidence B:** About 89% of final macro mean is carried by seeds 2024/2025; target best gain is +0.000441; no continuation best beats the frozen original F best,见 E027。
- **Difference in setting:** Macro/final-step vs target-domain/best endpoint; late anchor is severely overfit。
- **Can the contradiction currently be explained?** Yes; endpoint and seed distribution resolve the apparent conflict。
- **Current interpretation:** “Partial late-anchor rescue” is the narrow allowed statement; “better selected model” is contradicted。

#### C018 — S5-ML absent from exp_sn summary

- **Evidence A:** `exp_sn.md` summarizes S0–S6 and S6ML, but has no S5ML section or occurrence。
- **Evidence B:** `stage_S5ML_analysis.md` records a separate 16-run MovieLens Equal extension with partial Pattern A replication。
- **Difference in setting:** S5ML is a separate dataset extension, not the S6ML Sample Phase A included in exp_sn。
- **Can the contradiction currently be explained?** No conflict in results; this is a coverage omission in the prior summary。
- **Current interpretation:** S5ML is added as EXP026 and E017; do not mistake S6ML for S5ML。

#### C019 — exp_sn “Sample history effect disappears” vs later Sample late-stage evidence

- **Evidence A:** exp_sn concludes that after switching Equal→Sample, history effect disappears below practical threshold, based on Phase A selected gates。
- **Evidence B:** S6 Phase B finds MovieLens late DH effects; S8 finds late macro rescue but no selected/best improvement。
- **Difference in setting:** The older wording generalizes Phase A selected target gate to all checkpoints and all history/current effects。
- **Can the contradiction currently be explained?** Yes by narrowing the estimand; the broad wording is overextended.
- **Current interpretation:** Phase A establishes no practical selected target rescue; it does not establish zero late optimizer-policy effects。

**S6-R2 update:** The later KuaiRand R0-selected test ΔN is +0.0068 (4/4 positive) while target ΔT is −0.0004 (E028–E030). Phase A's no-practical-selected-target result remains valid, but it does not establish zero effects on every metric or evaluation layer; this later result is test-side and is not validation-visible.

#### C020 — exp_sn coverage end date vs later S6-R2/S7/S8 reports

- **Evidence A:** exp_sn summarizes through S6 and is dated 2026-10-06。
- **Evidence B:** S6-R2, S7, S7B, S8 analyses and S8 review are dated after the exp_sn cutoff and are absent from the file。
- **Difference in setting:** Later reports did not exist at the earlier summary cutoff。
- **Can the contradiction currently be explained?** Yes temporally; no claim of error at the earlier cutoff。
- **Current interpretation:** New global outputs include those later evidence and superseding conclusions; exp_sn remains the historical snapshot.

#### C021 — S6-R2 KuaiRand selected-checkpoint validation gap vs test gain

- **Evidence A:** At the same R0-selected checkpoint pairs, validation ΔN is −0.0039 (1/4 positive) and validation ΔM is −0.0027。
- **Evidence B:** On test at those same checkpoints, ΔN is +0.0068 (4/4 positive) and ΔM is +0.0044; target ΔT is −0.0004 (E028/E029)。
- **Difference in setting:** Same dataset, objective, seed pairs, and selected checkpoints; the evaluation layer differs。
- **Can the contradiction currently be explained?** No; the direction reversal is observed, but its source is unresolved。
- **Current interpretation:** The DH test-side result is not a validation-visible selected improvement; do not collapse validation and test into one performance claim。

#### C022 — S6-R2 KuaiRand selected test gain vs MovieLens negative control

- **Evidence A:** KuaiRand R0-selected test ΔN = +0.0068 and ΔM = +0.0044; A-grid R1/R3/R4 test ΔN means are positive (E028)。
- **Evidence B:** MovieLens R0-selected test ΔN is approximately zero, and available alternative-rule results do not show stable non-target gain; MovieLens canonical selected epochs are 7–13 (E030)。
- **Difference in setting:** Dataset, target domain, selected epoch range, and late-training trajectory differ, even though both use Sample/M3OE F vs DH source checkpoints。
- **Can the contradiction currently be explained?** Partially as a dataset/stage boundary; the moderator is not identified。
- **Current interpretation:** S6-R2's selected test gain is KuaiRand-specific in the tested matrix; the MovieLens null cannot be used to erase it, nor can the KuaiRand result be generalized to MovieLens。

#### C023 — Intact DH selected solution gain vs cross-partition hybrid collapse

- **Evidence A:** Intact DH has KuaiRand R0-selected test non-target gain relative to F (E028)。
- **Evidence B:** Under the manifest-based Part D swap, C/D hybrids fall to test macro ranges 0.545–0.584 / 0.520–0.579; F_hybrid remains within about 0.00011 macro of F_full (E032)。
- **Difference in setting:** Complete trained solutions are compared with evaluation-only hybrids assembled from partitions across distinct solutions; seed2025 has a near-zero DH−F denominator for retention ratios。
- **Can the contradiction currently be explained?** Partially; cross-partition co-adaptation is consistent with the collapse, but does not identify the source of the intact DH gain。
- **Current interpretation:** The tested swaps show non-composability under this partition definition, not that DH's test-side effect is absent or localized to one component。


---

#### 2026-10-18 跨分支补充的冲突、修订与处理结果

#### C024 — Stage2+ “0正式训练” vs 同分支 Stage2D “16/16完成”

- **Evidence A:** [10-16 early execution](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/execution_20261016/execution_report.md) 在阶段审计时写0新训练、四臂阻塞。
- **Evidence B:** [10-16 later Stage2D final](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/execution_report.md) 明确16/16、四seed×四臂、数值核验成功。
- **Classification / Current status:** **RESOLVED / TEMPORAL SNAPSHOT SUPERSEDED**。较晚最终执行优先；早期“NO训练”仅属快照，不能用于新综合报告的阶段完成数。历史 S5/C×A 等未触发范围保持不变。

#### C025 — 旧 S6-R2 KR selected test DH非目标阳性 vs 10-15 B1重建接近零

- **A/B:** 旧R0 +0.006757、4/4正；新重建 R0 +0.000579、2/4正，R1/R4 +0.011584、4/4正且LL代价。
- **Classification / Current status:** **OBSERVATIONAL DIFFERENCE / MECHANISM UNRESOLVED**，并非同init、同历史完整state、同谱系的成对重放；B1文件身份明确 RECONSTRUCTION_NOT_HISTORICAL_EXACT。保留原历史观察但不据此开发部署策略（E028–E029、E033）。

#### C026 — Stage2D fixed-J同q显著负向 vs v2 Adam selected 区间跨0

- **A/B:** Stage2D SB−SN validation宏−0.010406，四seed负；v2−0.005556，3/4负且CI跨0。
- **Difference:** Stage2D **learned fusion、双Adam**；v2 **固定fusion、单optimizer**，后者还改变源码底座/SGD校准。
- **Classification:** **PROTOCOL BOUNDARY / NOT A DIRECT CONTRADICTION**。不能将两效应做“同设置复现实验成功/失败”的配对差；E035–E036。

#### C027 — “Adam是必要条件” vs 无动量SGD也出现采样损伤

- **A/B:** 局部旧S5/Equal Adam历史carrier有作用；v2固定fusion/SGD SB−SN −0.002104（4/4负）；v3 fixedSGD早期也可四负。
- **Current interpretation:** **通用必要性 FALSIFIED**，但S5原命题是Equal/特定target的完整history source局部效应而非“所有现象要Adam”；不是同一量词。m/v分量、长期纯中介继续未识别；E015、E036–E038。

#### C028 — v3多家族四seed负向 vs N17 KR八seed选点主Holm不通过

- **A/B:** v3 calibratedAdam test selected −0.008698（四负）等；N17 common LR KR Adam −0.005203（7/8负）、t95跨0、Holm p=.171875，SGD +0.000837、Holm p=.6953125。
- **Difference:** optimizer LR/selection预算、独立seed数、确认性家族与模型/人口细项不同。v3四seed同号的双侧随机化最小p=.125，不能硬标p<.05。
- **Classification:** **EVIDENCE STRENGTH REVISED**：条件效应保留，**N17主selected未确认**；不是证明所有q效果为零。E037、E039。

#### C029 — N17 ML own-LR E100大幅正向 vs own selected负向

- **A/B:** 同 ML固定subset 8seed own对照：E100 SB−SN +0.021145、8正；各自验证选点 selected −0.010875、8负。
- **Difference:** 固定终轮vs各自validation选轮；own定义 SN lr=.001、SB lr=.0003，q与LR同时变化。
- **Classification:** **BUDGET × SELECTION × LR ESTIMAND BOUNDARY**，不是两份训练报告互相矛盾；E100 cell非劣置信下界不达−.003，不能升格成安全方法。

#### C030 — 同一组模型的宏AUC损伤 vs 按训练/测试人口加权AUC获益

- **A/B:** 历史calAdam E50同预测macro −.009074、train-p +.001016、test-p +.003108；新 ML own macro −.010875、train-p +.000744、test-p −.013586。
- **Classification:** **RESOLVED MATHEMATICAL AGGREGATION DIFFERENCE**。cell macro与cell加权的定义不同，且均不是pooled AUC；不能以切换指标冒充目标上的稳健性能；E040。

#### C031 — v3 C局部AUC阳性 vs loss/J不一致与N17 N4未触发

- **A/B:** V7 C的early/mid局部gap attenuation屏幕AUC 4/4正，但第二阶段 LL/trainJ联合门不通过；N17 embedding/E10两路径 full 4/4正、绝对~+3e−5，方向/实用门失败。
- **Classification:** **LOCAL POSITIVE / LONG-TERM NECESSITY NOT TESTED**。复核明确非“C全无效”和非“C必要机制”，N4 NOT_TRIGGERED，未运行不是阴性结果；E038、E042。

#### C032 — same-J q方差差来自count vs N17 quota仍约2倍trace

- **A/B:** 按q改变域计数和每域梯度统计都可影响covariance；N2 fixed θ将域count固定后，ratio IID 2.009–2.584、quota 1.995–2.551，count项只有.001297%–.105936%。
- **Classification:** **COUNT-DOMINANT HYPOTHESIS FALSIFIED IN ANCHORS**；不能再声称这几锚主要来自count波动，但不能推出全部由shape而非trace或Adam history造成。quota的长期交互CI跨0；E041。

#### C033 — S5/S8 late optimizer rescue 与“新q算法有效”混淆

- **A/B:** Equal历史S5 target rescue具有特定history carrier；Sample S8 的late rescue不提高原F best；v3/N17 q路径早期也可影响AUC。
- **Classification:** **DIFFERENT OBJECTIVE / INTERVENTION / WINDOW**。不能把旧的 history rescue改名为新的Sample采样方法，亦不能从Sample无新方法收益推历史Equal效应不存在。保持方法NO_GO、机制问题有界开放；E015–E027、E038–E043。

#### C034 — N17 Adam bootstrap区间负 vs 预定t/符号Holm主准则未确认

- **A/B:** 8seed KR Adam selected: bootstrap95 [−0.010924,−0.000367] 排0，但t95 [−0.012006,+0.001600]跨0、预定Holm p=.171875。
- **Classification:** **INFERENCE PROCEDURE DIFFERENCE, NOT DATA CONTRADICTION**。两种区间如实记录，不能事后择一有利方法将主终点改写成“确认”；E039。

**整体说明：** 分支结束时“完成全部触发训练”和“独立因果机制已识别”是两个不同命题；53问题中大量 BOUNDARY / NOT_IDENTIFIABLE / INCONCLUSIVE 正是研究剩余空间，不能用发布完成状态自动关闭。完整对照请转[问题去向](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)。


#### 2026-10-20—2026-10-26 新增跨轮冲突与解释（C035–C042）

#### C035 — “旧MovieLens即自然人口” vs 全自然/人工比例审计
- **A/B:** 早期多个suite=natural或固定ML结果实际使用人工187500行（34500/3000/150000）；新自然800167行（155949/318599/325619）；N19/CM19的人工子集并未因文件名natural变成自然完整训练。
- **Classification:** **DATA POPULATION IDENTITY CORRECTION**；结果仍有效于旧设置，新自然selected/late部分反向，但改变了N、支持和时钟，不能仅归于比例。E048–E049。

#### C036 — 自然S5/S6 selected不复制旧救援 vs 自然final仍有改善
- **A/B:** 旧人工配置中的Equal/Sample target rescue在选点下曾有正结果；自然matchedN/full有选点反向或缩弱，又在部分full末态改善。
- **Classification:** **POPULATION + TIME + METRIC SCOPE**。同名HT/DH不代表同初始化/更新数；晚期少恶化不推出最佳模型较优。自然扩展仍PARTIAL。E049。

#### C037 — CM19路径中和预测完全不变 vs 局部AdamW/正则修复后有响应
- **A/B:** KR fixed选点输入乘性调制24/24数值中性；同步消除两个factor的耦合L2或局部AdamW使响应恢复，独立STAR等并未同样中性。
- **Classification:** **SPECIFIC PATH/OPTIMIZER CONDITION, NOT UNIVERSAL DEATH**。即时数值贡献、重训必要性、最终性能不同。E046–E047。

#### C038 — 数据生成D获正收益 vs 固定权重Dconst01同样正收益
- **A/B:** ML D−B test AUC +0.006303（4/4）而Dconst01−B +0.005676（4/4）；D的observed权重接近0.1。Dconst01对D尚未证实非劣。
- **Classification:** **SUITE EFFECT VS DYNAMIC SAMPLE WEIGHT NECESSITY**。可认D套件正结果，不能归功动态样本区分或单独噪声；还需固定0.1无增强对照。E050–E051。

#### C039 — 首轮G−D含多种变化 vs 匹配计算路径G−Gtau0负向
- **A/B:** KR G−D macro AUC −0.003302（2正2负、区间跨0），G−Gtau0 −0.006484（4/4负，区间仍跨0），Gtau0−D +0.003182（4/4正）；G早期validation加速而selected test较弱。
- **Classification:** **DIFFERENT GRADIENT IMPLEMENTATION + EPOCH/ROLE/AGGREGATION**，不是“梯度调幅有时万能有时必坏”的同条件冲突。E052。

#### C040 — 10-24权限策略/10-25 RHD晚期改善 vs 合法selected未领先
- **A/B:** I1_PERM selected test宏AUC不胜RW；RHD正式HB−BASE selected LL +0.00074075、e50 −0.00875428，且所有方法e50比自己的合法selected差。
- **Classification:** **SELECTED PRIMARY VS LATE RELATIVE RISK**。不能用晚期风险减损宣称最佳已选模型提升。E053–E055。

#### C041 — LPS有foreign可读性 vs DFD形成收益弱于简单读法
- **A/B:** LPS部分来源内容关闭后变差；DFD ML P1对CONST有利，但完整P2输H_DIRECT/ETA_DIRECT和GEOM等；KR P1没有进入形成。
- **Classification:** **ACCESSIBLE SOURCE CONTENT VS NECESSARY SOURCE FUNCTION FORMATION**。普通读取有用不证明原型、meta权重、跨域μ外推有效；局部失败不否定来源可读性。E056–E057。

#### C042 — “实验方案/发布完成” vs 实际NO_START与机制工件缺项
- **A/B:** NEXT-v2核心48完成但正式0 NO_START；v3有实际50epoch；DFD P3/P4因门未触发、184导出和12推理已完成但外层η/μ分组漏导出；RCF只有实施设计、无新训练。
- **Classification:** **EXECUTION STATUS / ARTIFACT AVAILABILITY**。不能一概写成“全部方法完整阴性”或“所有机制信息已齐”，更不能把RCF方案的理论价值记为测得AUC。E054–E057。

**修订原则：** “小域=困难域”始终不是研究者认可的先验；共享数值失活、来源可读、形成器有用和共享可提升泛化分别是不同问题。若没有同人口/同参数/同指标的直接对照，保留竞争解释而不强行判某一机制为根因。

#### 2026-10-27—2026-11-02 新增跨分支/指标分歧 C043–C051

#### C043 — Relation-Basis在ML的弱LL改进与AUC一致受损
- **A/B:** LOWRANK−TARGET_ONLY的历史E宏AUC ML−0.000623（4/4负）、LL−0.000715（3/4改善且区间跨0）；KR AUC/LL均不利。
- **Classification:** **METRIC TRADEOFF, NOT PROOF OF UNIQUE METHOD VALUE**；保存局部损失信号，不覆写排序阴性及版本STOP。E058。

#### C044 — BASP-A部分固定来源读取有依赖，动态原型政策却未晋级
- **A/B:** 部分读取对固定源内容的替换会引起评价变化，但CG的留出μ不胜常量、NULL占优，Q/E不稳定胜目标匹配。
- **Classification:** **READ DEPENDENCE ≠ NEED TO FORM PROTOTYPES ≠ DEPLOYED BENEFIT**。E059；不能把旧来源可读性重新当作新机制发现。

#### C045 — GA31缩放产生selected收益与普通L2形成近似参照并存
- **A/B:** ML c=.1对c=1无增强test selected AUC+0.003473；后续普通L2校准在ML的validation排序接近缩放参照、test AUC细微反号。
- **Classification:** **DIFFERENT CONTROLS/OPTIMIZER PARAMETERIZATION**；缩放现象成立，但不能被单独归因动态样本specificity或纯LR变化。E061–E062。

#### C046 — ANTI2降低方差/相对普通基线部分提高，但不胜同K IID2
- **A/B:** KR ANTI2−B-KR selected test宏AUC+0.001202（4/5）；ANTI2−IID2−0.001237；validation/最差格保护不支持整体晋级。
- **Classification:** **BASELINE DEPENDENCE & MECHANISM/UTILITY SEPARATION**；既不删除方差下降事实，也不晋级当前方法。E063。

#### C047 — 非反馈EMA有稳定选点AUC收益却可带来ML LogLoss代价
- **A/B:** ML无增强EMA−online AUC+0.004399（5/5）、LL+0.007800（4/5恶化）；调参后AUC+0.003849（5/5）、LL+0.001249（4/5恶化）。
- **Classification:** **AUC RANKING VS PROBABILITY LOSS / TRAINING CONFIGURATION**；允许确认排序有效，不推断LL或改动Adam一阶动量有益。E064。

#### C048 — KR Gaussian+EMA宏LL改善而大流量域AUC受损
- **A/B:** AIGM01 Gaussian相对无增强EMA test宏LL−0.001664（5/5）、宏AUC+0.001476（3/5）；d1/task0与task1 AUC−0.004654/−0.004929，pooled/traffic AUC下降。
- **Classification:** **TASK/CELL BENEFIT DISTRIBUTION AND METRIC AGGREGATION**；不能拿一项宏均值掩盖受损目标。E066。

#### C049 — GATS B2开发通过与确认衰减、test近零同时成立
- **A/B:** KR B2−A1开发validation宏AUC+0.008184（2/2），确认n=3 +0.000640（2/3），统一test五seed+0.000005（3/5）。部分目标任务的验证正值在test反向。
- **Classification:** **DEVELOPMENT/CONFIRMATION/TEST ESTIMAND DIFFERENCE**；开发阳性是真实观察，但新方法晋级的整体证据已不足。E067。

#### C050 — EMA02V2 F1的ML微小test收益与validation反向
- **A/B:** F1−A0+EMA test AUC+0.000359（2/2），validation均值−0.000424，低于+0.001实用门；KR总体又是另一方向。
- **Classification:** **WEAK, WINDOW/SEED-CONDITIONAL TEST SIGNAL**；不因2/2符号正重新打开全预算。E069。

#### C051 — 同一个TG方法在selected/E50的ML排序结论不同
- **A/B:** ML TG−A0+EMA selected test宏AUC−0.002418（0/2正），共同E50却+0.000376（2/2）；KR selected和E50仍退化。
- **Classification:** **CHECKPOINT SPECIFICITY**；不能用E50后期小幅正值替换冻结selected主裁定，不否认阶段性现象。E069。

#### C052 — 训练面板Fast−Slow方向持续性 vs 验证侧动作价值接近零
- **A/B:** 11/09 A0中KR-G训练面板P_A→P_B方向一致率0.96，验证侧0.60；C_persist的P_B和验证ΔAUC均负，C_carry虽有面板正值但相对统一Fast的验证增量约0。
- **Classification:** **TRAINING-PANEL PERSISTENCE ≠ FUTURE VALIDATION ACTION VALUE**；A0没有新训练/Test，不称完整模型失败。E070。

#### C053 — PUB候选相对同α G控制大幅改善 vs 相对正常A/F没有收益
- **A/B:** P0/P1/P2相对A均未过门；G控制本身E50宏AUC接近0.5，使C−G差值偏离正常算法对照。
- **Classification:** **CONTROL DEGENERATION / COMPARATOR IDENTITY**；保留更新空间相对G的差值，但不将其当作正式算法收益。E071。

#### C054 — current-preconditioned单步局部正向 vs 六轮真实反馈使F下降
- **A/B:** 同状态同范数单步干预约+0.00032至+0.00041 AUC；11/11 A_MATCH六轮反馈在ML/KR均使F受损。
- **Classification:** **ONE-STEP INTERVENTION ≠ CLOSED-LOOP TRAINING EFFECT**；单步局部机制和长期递推必须分开评价。E074、E076。

#### C055 — MovieLens两态B验证AUC增量 vs Test新增AUC接近零
- **A/B:** ML Clean两态B验证ΔAUC +0.00040544，Test相对F仅+0.00003113且区间跨0；Gaussian验证增量+0.00023892，Test +0.00009666且区间跨0。LogLoss Test两组均稳定下降。
- **Classification:** **VALIDATION RANKING GAIN / TEST RANKING GAIN / LOGLOSS ARE DISTINCT ESTIMANDS**；保留LL收益，不把它包装为AUC增量或校准证据。E077。

#### C056 — KuaiRand B_SHARED验证候选 vs Test未复现及最强发表基线未通过
- **A/B:** B_SHARED验证有正均值，但Test相对F为Clean−0.00013886、Gaussian−0.00169040；相对HyperGate/AutoIFS未过正式AUC门。另一个H277两态读出Test有+0.00069911诊断信号。
- **Classification:** **CANDIDATE SELECTION / FROZEN TEST / BASELINE IDENTITY**；B_SHARED与B_H277不是同一读出，不能混为一个结果。E077。

#### C057 — LogLoss稳定改善 vs AUC没有新增排序增量
- **A/B:** ML两态B的Test LogLoss两条件均五seed改善且区间低于0，但AUC增量区间跨0。
- **Classification:** **PROBABILITY LOSS VS RANKING**；没有直接校准诊断，不将LogLoss改善解释为已证明校准。E077。

**最新判断原则：****最新判断原则：** 相同数据/目标/算法/选点/评价人口的可比性优先；真实相反的metric和阶段效应必须并列，不将不一致机械判成研究失败，也不把历史反例反复包装成新问题。

## 已停止、已降级与未启动

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Dead_Ends.md)

> **2026-11-15 最新结论：** 历史条目保留原日期和当时证据强度；新增D034–D037仅停止11/09–11/11中已测试的具体版本，不扩大为对PUB、梯度反馈或自适应EMA整个方向的否定。最新判断见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)。

> **2026-10-26更新：** 早期D001–D018保留为**具体过强主张或当时版本**的停止记录，非对动量、梯度、增强或差异化共享学科方向的永久否决。追加D019–D024；其对应的新可区分问题见 [Open Questions](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md)、[2026-10-26总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)。



> **2026-10-18 补充：** “closed/dead end”在此只否定**特定过强命题或当前可用方法候选**，不自动否定整个MDMT采样、共享、梯度与优化研究方向。D001/D012 是历史待检验解释，**不代表用户曾认定“小域=困难域”或“小域天然低SNR”**。保留 D001–D012 的原历史证据，续增D013–D018；参见[多分支终局](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)。


“Dead end”指按原主张继续推进缺少当前证据基础；不否定历史结果。若更窄的设置仍有证据或未解决问题，会在末列明确保留。

#### D001 — 以“小域低 SNR”为依据发展 per-domain momentum

> **来源澄清（2026-10-18）：** 该条是历史对竞争解释的否定，并非用户最初认同过“小域=困难域”或低SNR等式；不要因此设计重复证伪的实验。

- **Why Initially Considered:** 早期假设小域梯度噪声更高，per-domain EMA 可补偿。
- **Evidence Against:** E001 显示 SNR 排序随 batch size 消失/反转；E008 未显示 comp1 相对 uniform EMA 的独特优势。
- **Current Status:** CLOSED / FALSIFIED in measured settings。
- **Narrower Version Remaining Open:** 其他数据或统计口径可能有域间噪声差异，但不能用本项目已测数据支持该前提。

#### D002 — M3oE-specific largest-domain sacrifice

- **Why Initially Considered:** 早期 M3oE MovieLens 结果显示 L-role negative transfer。
- **Evidence Against:** E003 的 P0-3 中 MMoE 同向；E004 后显示 objective dependence。
- **Current Status:** SUPERSEDED。
- **Narrower Version Remaining Open:** 某些 Equal/旧配置下的 L-role loss 仍成立；不再称 M3oE 特有。

#### D003 — comp1 是解决 objective mismatch 的必要方法，且增益随 ratio 单调增长

- **Why Initially Considered:** P5/P6 中 comp1 高 ratio AUC 增益显著。
- **Evidence Against:** E002 的 ratio response 非单调且高 ratio 与 N 混杂；E004 显示更改 objective 会减轻 Large-cell loss；E008 显示 comp1 未胜过 uniform EMA。
- **Current Status:** CLOSED as a general method premise。
- **Narrower Version Remaining Open:** 旧协议中 comp1 的历史 performance difference 可作为限定观察保留；其独特机制优势未建立。

#### D004 — Sample objective 下存在普遍、稳定、实用的 negative transfer，因此需 Safe-MTL 主线

- **Why Initially Considered:** 个别单元曾出现负差。
- **Evidence Against:** E009 的 S1 0/72 practical NT；S2 Sample 0/24；S6 Phase A 两个已测 target gate STOP（E018）。
- **Current Status:** CLOSED within tested Sample settings。
- **Narrower Version Remaining Open:** OQ001 的更极端 ratio / 未测 target 仍没有结论；当前没有已知 selected performance gap。

#### D005 — Curriculum 或已测时间 schedule 稳定胜过 static Sample

- **Why Initially Considered:** P6 与 E-series 希望利用域难度顺序或时间变化。
- **Evidence Against:** E006；P1-2 curriculum 量级低于门槛，E5 未通过 GO，E7 没有运行。
- **Current Status:** CLOSED for the tested implementations and gates。
- **Narrower Version Remaining Open:** 仅未测试的不同配置仍未被排除；现有证据不授权把旧方向重新当成有效前提。

#### D006 — E6 adaptive calibration/EMA 是已验证的主方法

- **Why Initially Considered:** adaptive coefficient calibration 可减 mismatch，EMA 可能平滑估计。
- **Evidence Against:** E006 中未超过 Sample 的预注册 performance/safety gate；E7 0 runs。
- **Current Status:** CLOSED / STOP。
- **Narrower Version Remaining Open:** 无已冻结 adaptive method；secondary metric 不改变裁决。

#### D007 — 仅匹配 gradient norm 或 applied-step norm 就能复制 target rescue

- **Why Initially Considered:** S3/S4 的 T rescue 可能只是幅度缩放。
- **Evidence Against:** E013 的 N/U 对照均未复制 T 的 d1 效应。
- **Current Status:** FALSIFIED as a sufficient explanation。
- **Narrower Version Remaining Open:** 方向、history 和轨迹因素仍有条件性问题；幅度本身仍是测量量，但不能作为充分机制。

#### D008 — Equal 下的 HT/DH 结果可直接当作 Sample 下的方法依据

- **Why Initially Considered:** S5/S5-ML 在 Equal 下发现 target rescue 与 history/current structure。
- **Evidence Against:** E018/E019 selected Sample target contrasts below practical threshold; E020 shows a MovieLens late-stage residual only; E021 confirms objective/stage dependence。
- **Current Status:** CLOSED as an objective-independent inference。
- **Narrower Version Remaining Open:** Equal 下 S5 的 history-dominant KuaiRand d1 fact remains valid within its scope；Sample 下状态机制不可写成 exact zero。

#### D009 — Target-only/HT 是无 non-target cost 的安全部署方法

- **Why Initially Considered:** S3 的 Equal target rescue 看起来显著。
- **Evidence Against:** E014/E016/S5ML show non-target losses and checkpoint/dataset trade-offs；E019 no Sample practical rescue；这些 arms 是诊断干预。
- **Current Status:** CLOSED as a validated safe method claim。
- **Narrower Version Remaining Open:** 某些 target cells 可通过 intervention 有所改善；是否存在满足指定多指标目标的设置尚未建立。

#### D010 — Universal gradient-conflict correction / U_k correction 研究线

- **Why Initially Considered:** S7-A 在 MovieLens late overfit checkpoints 观察到 harmful gradient structure 与 AUC decline 同步。
- **Evidence Against:** E024 的 KuaiRand negative-U counterexample；E025 same-θ utility change 未改善 validation；E026/E027 full-state S8 无 selected/best gain。
- **Current Status:** CLOSED as a method-development premise under the tested evidence。
- **Narrower Version Remaining Open:** S7 的 Domain×Task gradient structures 作为 MovieLens late-stage observation 保留；跨配置普遍性在 OQ006，而性能根因/方法价值未获支持。

#### D011 — 从 MovieLens late DH checkpoint gain 推断通用 selected-performance gain

- **Why Initially Considered:** S6/S6 trajectory reanalysis 中 MovieLens late common checkpoints 的 macro/non-target gains。
- **Evidence Against:** E026/E027 的 S8 MovieLens primary best endpoint；selected continuation null，且无 continuation best 超过 original F best。S6-R2 的 E028–E030 显示结果依赖 dataset 和 metric layer：KuaiRand 有 test-side selected non-target gain，validation 不占优，MovieLens selected contrast 仍近零。
- **Current Status:** CLOSED as a general inference from MovieLens late checkpoints to selected performance; not a universal null claim。
- **Narrower Version Remaining Open:** MovieLens already-overfit late anchor 有 partial rescue/less deterioration；KuaiRand S6-R2 有 test-side selected gain，但其 validation/test reversal 与原因仍未解决（OQ009）。

#### D012 — 把“域大小”当作当前解释或实验结论

> **来源澄清（2026-10-18）：** 该条是历史对竞争解释的否定，并非用户最初认同过“小域=困难域”或低SNR等式；不要因此设计重复证伪的实验。

- **Why Initially Considered:** P0 role findings与旧的 size-based 叙述。
- **Evidence Against:** E003/E004/E010/E014/E020；域身份、objective、样本分布、target 与 checkpoint 同时变化；S6 full analysis 纠正了 MovieLens d0/d1 大小标签。
- **Current Status:** CLOSED as a stand-alone causal conclusion。
- **Narrower Version Remaining Open:** Domain size 是否是独立调节因素仍是 OQ002；没有预设小域更困难或大域更易受损。


---

#### 10-15—10-18 新增关闭的过强命题 / 当前方法化路线

#### D013 — “保持 Sample 期望目标，直接Balanced + p/q就能稳定提高宏AUC并无害”

- **Evidence Against:** Stage2D 16run validation SB−SN宏−0.010406、4/4负，收益/非劣门失败；v3多LR/阶段与N17八seed不同主结果亦未出现通过完整七项保护的候选（E035、E037、E039–E043）。
- **Current Status:** **CLOSED AS A READY-TO-DEPLOY METHOD UNDER TESTED SETTINGS**。
- **Narrower Version Remaining Open:** 特定阶段下 q×LR 或共享局部反事实能否在**新的**独立场景、冻结指标上满足绝对收益与全cell保护；不能仅靠差距gap改善代替健康提升。

#### D014 — “所有同目标采样效应都必须经 Adam 历史状态产生”

- **Evidence Against:** fixed fusion 下无动量SGD负向实例；固定θ预期场与局部协方差变化并不自动需要Adam；v2 SGD 4/4负（E034、E036–E037）。
- **Current Status:** **FALSIFIED FOR UNIVERSAL NECESSITY**。
- **Narrower Version Remaining Open:** 特定 Equal/S5历史carrier效应、v3特定q×LR interaction、同θm/v/ratio及长期必要性均尚可研究，不能据此写“Adam状态普遍无效”。

#### D015 — “所有差异都只由 validation 选择或晚期过拟合造成”

- **Evidence Against:** B1某seed同epoch仍val/test反号；v3和N17早期固定时钟轨迹已有差，N17 KR Adam AULC1..20−0.007480、8/8负（E033、E037、E040）。
- **Current Status:** **FALSIFIED AS AN EXCLUSIVE EXPLANATION**。
- **Narrower Version Remaining Open:** 选择、有限正例、LR/速度及普通过拟合仍是有作用的**修饰项**，ML selected/E100反号未独立分解。

#### D016 — “q间固定点方差差异主要由随机域计数波动造成”

- **Evidence Against:** N17 quota固定域计数后约2倍trace仍在；count项只占0.001297%–0.105936%（E041）。
- **Current Status:** **CLOSED AS DOMINANT EXPLANATION IN THOSE 16 ANCHORS ONLY**。
- **Narrower Version Remaining Open:** 域内真实梯度分布、重复/coverage/时序依赖及优化器非线性相互作用；未否定所有其他θ、batch模型中的count作用。

#### D017 — “共享 C 已被局部AUC屏幕证实为长期必要机制，立即方法化”

- **Evidence Against:** v3局部C AUC有筛查阳性但loss/J门失败；N17独立局部仅约+3e−5且预设实际/方向门失败，N4未触发（E038、E042）。
- **Current Status:** **CLOSED AS A CLAIM OF ALREADY-DEMONSTRATED LONG-TERM NECESSITY OR READINESS**。
- **Narrower Version Remaining Open:** 局部C有效路径和容量匹配的长期必要性仍未识别；N4没运行不意味着C没有作用。

#### D018 — “在历史S6 init/optimizer谱系缺失时，重建B1即可验证或推翻原S6-R2”

- **Evidence Against:** 新B1与旧S6-R2的init/lineage不同，数值虽有显著结论差却不属于同run精确复现；新R1/R4亦不可据test事后替换R0（E028–E030、E033）。
- **Current Status:** **CLOSED AS AN INFERENCE METHOD**。
- **Narrower Version Remaining Open:** 原selected validation/test反转的具体机制、可用历史状态恢复及独立协议下的多seed适用范围（OQ009/OQ010）仍保留。

**不能新增虚假死路：** “小域=困难域”不是起始结论，不需要按轮次反复验证“推翻”；“momentum/课程/gradient天然无价值”并未得到普遍证明；旧 S5 Equal history 不是被新 SGD 反例整体推翻。所有**未运行、未触发、检验功效不足、缺模型状态或暂时NO_GO**必须归入其对应的开放/条件状态，而非偷换成科学失败。


#### 2026-10-26 后续方法停线/主张收缩（D019–D024）

#### D019 — “D相对原始基线增益＝动态specificity重权必要”
- **Evidence Against:** ML D−B有稳定增益，但Dconst01−B仍4/4有利，D观测样本权重快速到0.1附近；Dconst01与D尚无非劣结论。E050–E051。
- **Current Status:** **CLOSED AS A CLAIM OF PROVEN DYNAMIC NECESSITY**。
- **Narrower Opportunity:** 有正收益的整体套件仍成立；同尺度无增强控制可判别全局loss scale与复制/噪声，然后再讨论额外动态区分。

#### D020 — “调幅增加小域梯度范数就普遍提高最终宏AUC”
- **Evidence Against:** 10-21同逐格G计算路径下τ0.5对τ0的ML/KR macro AUC四seed方向均负；KR区间跨0，早期速度和若干域/pooled仍不同向。E052。
- **Current Status:** **CLOSED FOR THIS TAU/PROTOCOL AS SAFE GENERAL METHOD**。
- **Narrower Opportunity:** 真正的实际共享更新与样本重加权如何影响局部目标的净风险，仍可在不同问题上研究；不能用通用否定覆盖GradCraft等不同机制。

#### D021 — “KR M3oE一个输入调制路径中性＝全部MDMT条件共享参数失效”
- **Evidence Against:** CM19仅在特定fixed/Adam/人口/选点对输入乘性路径给出中性；STAR等独立结构反例仍活跃。E046–E047。
- **Current Status:** **CLOSED AS A UNIVERSAL ARCHITECTURE CLAIM**。
- **Narrower Opportunity:** M3oE耦合正则、路径激活、梯度/Adam位移和最优性能间的条件关系尚有可解释价值。

#### D022 — “恢复共享调制路径就能稳定得到更好的最佳泛化模型”
- **Evidence Against:** CM19去双factor耦合正则或局部AdamW可以修复响应，但selected AUC/LL与强对照无通用稳定增益；即时中和与无模块重训可反号。E047。
- **Current Status:** **NOT SUPPORTED AS DEPLOYABLE METHOD; STOP UNQUALIFIED PROMOTION**。
- **Narrower Opportunity:** 保留结构特有的数值机制与可判别成本/模型选择反事实，不把活性当目标本身。

#### D023 — “RHD/LPS/DFD当前动态形成方法已验证优于强简单方案”
- **Evidence Against:** 10-24 I1 selected未胜RW；v3 RHD只有late减损而selected HB−BASE LL更差，LPS形成器无稳定额外优势；DFD ML P2输H_DIRECT/ETA_DIRECT并被预设保护门关闭。E053–E057。
- **Current Status:** **CLOSED FOR TESTED VERSIONS/CLAIMS OF ALREADY-VALIDATED FULL BENEFIT**。
- **Narrower Opportunity:** 来源可读内容和需求异质性有局部信号；RCF可通过目标群体剩余风险、源保留约束和普通reader重建差异，但当前**DESIGN_ONLY**，不能提前登记成功。

#### D024 — “补充自然MovieLens只需重标现有人工结果即可完成”
- **Evidence Against:** 人工187500/自然800167的人口/支持/时钟不同；自然补充旧分支最终状态仍IN_PROGRESS，N19/CM19大矩阵未得到完整自然终报。E048–E049。
- **Current Status:** **CLOSED AS A RESEARCH ACCOUNTING SHORTCUT**。
- **Narrower Opportunity:** 已完成自然子项可限定使用；如后续要确认跨人口方法收益，先取得真实完整冻结记录或新人口，而不是按目录名替代实证。

**不能错误关闭的事项：** 已有方法在相关方向尝试过，不等于目标群体内收益/来源可用函数已无研究价值；同样，NO_START、历史原工件缺失、n较小的区间跨0和新RCF未执行都不等于反证了对应科学问题。尤其“小域=困难域”不是研究者曾经认可的起点。

#### 2026-10-27—2026-11-02 新增具体方法/过强主张停线 D025–D033

#### D025 — 当前Relation-Basis低秩/全关系已优于TARGET_ONLY
- **Evidence Against:** E058：两自然数据各4seed，Q均选择TARGET_ONLY，LOWRANK历史E宏AUC两数据各4/4下降。
- **Status:** STOP_THIS_VERSION。
- **Retained:** 来源可读性、原型局部共享的历史动机和其他形式仍可研究。

#### D026 — BASP-A的收益探测器与原型关系已经带来实用选读增量
- **Evidence Against:** E059：留出μ不胜F域常量，CG主要选NULL，强TARGET_MATCH前宏AUC没有稳定实用增量。
- **Status:** STOP_THIS_POLICY；P2–P5 NOT_TRIGGERED。
- **Retained:** 部分固定读取依赖不等于完整政策有效；当前μ不能当已证实的收益相似性先验。

#### D027 — 目标域NR-Gauss独立稳定提高宏AUC
- **Evidence Against:** E062：ML两尺度均未达增强门，KR c=1仅2/5正且最差格有保护损害。
- **Status:** STOP_GA31_NRG_V1。
- **Retained:** 不改写采用不同人口、目标、复制与权重的历史D套件的阳性结论。

#### D028 — 反向配对降低梯度方差即意味着独有泛化改进
- **Evidence Against:** E063：KR ANTI2对普通基线条件正，但相对同预算IID2 AUC均值负，验证/最差格也不支持全门。
- **Status:** NO_UNIQUE_METHOD_PROMOTION。
- **Retained:** 条件方差确实降低，不把它机械解释为AUC增益。

#### D029 — 非零SAD增强差分长期记忆必优于普通增强Adam
- **Evidence Against:** E065：βδ=.99改变更新，ML/KR对A1 selected宏AUC差均负，KR目标域两任务下降。
- **Status:** STOP_TESTED_BETA_DELTA_099。
- **Retained:** 非零状态不是无效实现；未来改变真实递推并注入域任务信息的方案不是当前A3的简单复测。

#### D030 — AIG-M创新门有已完成五seed可部署收益
- **Evidence Against:** E066：有限开发搜索没过门，非零门输出不等于收益，五seed AIG候选**未运行**。
- **Status:** STOP_AT_DEVELOPMENT；FORMAL_NOT_TRIGGERED。
- **Retained:** 不把未识别的剂量/时程中介当成已证实独有信号，不靠扩seed无限维持同版本。

#### D031 — GATS B2两开发seed验证阳性足以确定整体推进
- **Evidence Against:** E067：确认seed验证增益收缩，统一test对A1宏AUC+0.000005，对强A0+EMA仅2/5正。
- **Status:** NO_PROMOTION_AFTER_COMPLETED_CONFIRMATION。
- **Retained:** 开发/局部任务现象保留；同seed再次报告不算独立确认。

#### D032 — GATS难度覆盖B1、延迟B3、CV→rho C2已证明独有提升
- **Evidence Against:** E068：各自匹配视图/日程控制与预定门未过。
- **Status:** STOP_THESE_RULES；部分更大规模后续NOT_TRIGGERED。
- **Retained:** 真正的MDMT任务条件性和时间动态仍是可发展的研究题目，不能把旧规则换名重新作为空白。

#### D033 — EMA02V2 J/F/TG/AG新视图已胜固定无增强EMA，应该继续全预算
- **Evidence Against:** E069：ML F1仅有验证反向的小幅test信号，KR TG/AG的AUC与LL双重显著方向性退化。
- **Status:** STOP_THESE_VIEW_VERSIONS；CONFIRMATION/W1_NOT_TRIGGERED。
- **Retained:** ML F1只是条件线索，不能把当前视图梯度直接当动量式优化融合完成。

#### 2026-11-15 新增当前版本停止项（D034–D037）

#### D034 — EMA_ADAPTIVE A0当前训练面板预测器与静态任务动作
- **Current version:** C_persist、C_carry和TASK0/TASK1静态政策。
- **Why stop:** 训练面板方向持续性未迁移为验证动作价值；C_persist未来面板与验证侧均负，C_carry相对统一Fast的验证增量近零，task策略退化为统一Fast。
- **Scope:** A0仅复用既有轨迹，无新增训练或Test；A1/B/C未触发。停止当前预测/动作规则，不是完整自适应EMA算法失败。
- **Evidence:** E070；[11/09报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

#### D035 — PUB-MDMT当前原始尺度P0/P1/P2 coupled-WD迁移
- **Current version:** P0/P1/P2与同α G控制。
- **Why stop:** 三种迁移相对A均未形成Test AUC正向seed，Macro AUC/LogLoss均不满足晋级要求；G控制本身退化，不能用C−G作为算法成功依据。
- **Scope:** 仅停止当前原始尺度/coupled-WD版本；报告建议的WD=0矩阵未执行，其他PUB参数化不在本停止结论内。
- **Evidence:** E071；[11/09 PUB报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)。

#### D036 — A_MATCH当前current-preconditioned方向的六轮反馈版本
- **Current version:** alpha=0.5/1.0的短反馈A_MATCH。
- **Why stop:** ML与KR两个强度均使F的验证AUC下降；online局部正值没有转化为F持续收益。
- **Scope:** RAW、完整A续训和AB未触发；停止当前A_MATCH版本，不外推为所有梯度/动量反馈无效。
- **Evidence:** E076；[11/11 EMA_NEXT报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### D037 — 当前C类cell系数/未来动作规则
- **Current version:** 11/10 C_design/C_simple及11/11四种cell预测系数规则。
- **Why stop:** 11/10可见训练信号没有超过固定F的实际动作增量；11/11四种cell规则均NO_FUTURE_SIGNAL。
- **Scope:** 不再重复同类规则网格；只有提出会改变真实参数时间递推且可能改变研究决策的新信息机制时才重开。
- **Evidence:** E075、E078；[11/10报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)、[11/11报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

**不是死路的结论：** 固定EMA选点排序的已测收益；原历史D套件在其协议内的正效应；来源读取/群体差异的有界事实。当前版本停线不推导全部梯度、增强、动量或共享思想不成立。

## 证据记录 E001–E078

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)

> **2026-11-15最新入口：** 已增补E070–E078，承接11/09 A0、11/09 PUB-MDMT、11/10 EMA因果归因和11/11 EMA_NEXT。证据条目记录的是研究证据组，不等于新增独立seed或外部样本；最新判断见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)。

> **2026-10-26 总证据增量：** 历史 E001–E043 保持原编号与观测；后文新增 E044–E057（N18/N19/CM19、自然比例审计、重加权/调幅、10-24/25/26方法）。10-09的STRONG/MEDIUM/WEAK计数表仅是旧快照，**不得把新增证据默认加为“更多独立seed”**。总研究结论、后期修订和原始阶段路径以 [2026-10-26跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md) 为准；RCF-v1仍为设计，非实验效果证据。



> **2026-10-18 增量:** 以下 E001–E032 是截至 2026-10-09 的历史证据，不被覆盖；本文件末尾新增 **E033–E043**。后续阶段主证据以[跨分支整合](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)限定的数据集、样本、目标、训练协议、选择函数和统计检验为准。以下新编号代表**研究证据组**，不是每个原始训练的独立重复。


状态截至 2026-10-09。Evidence 只记录直接观察与审计事实；Interpretation、status 和可接受外推单独列出。AUC 差均注明方向；除非写明，跨 seed 的均值使用报告口径，不能把 cell/epoch 当作独立 seed。

#### E001 — MovieLens 梯度 SNR 不支持“小域必然更噪”

- **Observation:** P1 的共享参数诊断中，batch=40960 时 d0/d2 SNR 为 0.5085/0.5584；batch=4096 时为 0.0524/0.0511，大小差异消失并反向。梯度 cosine 为弱正。
- **Source Experiments:** EXP001。
- **Source Reports:** [`P1_diagnosis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P1_diagnosis.md)；[`P6_ablation.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P6_ablation.md)（SNR 反推 γ 的后续 ablation）。
- **Dataset / Objective / Model:** MovieLens-1M；旧 as-published objective；M3oE。
- **Seeds:** 5（2022–2026）；SNR 汇总含 epoch×seed 配对观测。
- **Effect Direction / Magnitude:** Mixed / batch dependent；SNR 差约 −0.050（40960）與 +0.0013（4096），不支持稳定的小域劣势。
- **Robustness / Scope:** Weak；单数据集、单共享参数定义、两种 batch size。
- **Current Status:** TENTATIVE。
- **Current Interpretation:** 该测量条件下，小域较低 SNR 不是稳定现象；P6 的 SNR-derived momentum 未改善也不说明所有数据域都没有噪声差异。
- **Alternative Explanations:** batch size、样本组成、共享参数集合、梯度方差估计方式。
- **Evidence Strength:** WEAK。

#### E002 — 历史 comp1 增益随 ratio 并非单调，ratio 与总 N 曾混杂

- **Observation:** MovieLens 早期 comp1 相对 baseline 的 AUC 差随 downsample ratio 依次约为 +0.0064、+0.0034、−0.0012、+0.0053、+0.0166、+0.0202；后两档同时改变总训练样本量。P0-2 固定 N 设计仍观察到 ratio 对部分增益有贡献。
- **Source Experiments:** EXP003、EXP006。
- **Source Reports:** [`ML_P5_REPORT.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/ML_P5_REPORT.md)、[`P0_2_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_2_STAGE_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens-1M；历史训练目标；M3oE。
- **Seeds:** P5 各 ratio 为 3 或 5；P0-2 为 4。
- **Effect Direction / Magnitude:** Mixed；P5 不是单调曲线；P0-2 ratio=50 的 L-role Δtransfer −0.015545 ± 0.010894。
- **Robustness / Scope:** Medium；旧实验与固定 N 复核方向不完全相同，且 role/ratio/样本结构不可视为单一变量。
- **Current Status:** SUPERSEDED（针对“ratio 单调解释”）。
- **Current Interpretation:** ratio 与 early comp1 表现有关，但不足以支持单调 ratio law 或 domain-size 单因果。
- **Alternative Explanations:** 总 N、域身份、有效 loss 权重、样本多样性、checkpoint、初始化方差。
- **Evidence Strength:** MEDIUM。

#### E003 — 大域角色上的旧退化不局限于 M3oE，但不识别 size 单因果

- **Observation:** 固定 N 角色轮换中，P0-1 的 L-role Δtransfer 为 −0.014557 ± 0.007439；P0-2 ratio=50 的 L-role 为 −0.015545 ± 0.010894。P0-3 的 MMoE 也出现 L-role 负 transfer（约 −0.0116），与 M3oE 同向。
- **Source Experiments:** EXP005–EXP007。
- **Source Reports:** [`P0_1_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_1_STAGE_REVIEW.md)、[`P0_2_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_2_STAGE_REVIEW.md)、[`P0_3_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_3_STAGE_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens；旧 objective；M3oE 与 MMoE。
- **Seeds:** 4 per matrix; role/cell 配对数见各报告。
- **Effect Direction / Magnitude:** L-role 负向；约 −0.01 AUC 量级。
- **Robustness / Scope:** Medium；跨两个骨干和多个 role 条件，但仍受 objective、有效权重、域身份、训练收敛共同影响。
- **Current Status:** SUPPORTED（限所测 fixed-N 配置）。
- **Current Interpretation:** 早期退化不是 M3oE 独有；角色轮换比某一个物理域的解释更可信，但域大小未被独立操纵。
- **Alternative Explanations:** Objective 权重、域身份与难度、样本多样性、gradient scale、训练收敛。
- **Evidence Strength:** MEDIUM。

#### E004 — Equal 与 Sample 是不同目标；Sample 在已测选点减轻 Large-cell 损伤

- **Observation:** P0-4 中，Equal-train / Equal-select 的 L-role Δ 为 −0.0155，Sample-train / Equal-select 为 −0.0012；P0-5 与 S2 的 Sample 条件也未达到预设 practical negative-transfer 门槛。S2 Equal−SD 有 8/24 组达到 −0.003 或更差，严格组合 gate 为 3/24；Sample−SD 为 0/24。
- **Source Experiments:** EXP008–EXP009、EXP022。
- **Source Reports:** [`P0_4_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_4_STAGE_REVIEW.md)、[`P0_5_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_5_STAGE_REVIEW.md)、[`stage_S2_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S2_analysis.md)。
- **Dataset / Objective / Model:** 主要为 MovieLens；S2 另含 KuaiRand；M3oE。
- **Seeds:** 4 per later stage; P0 summary uses paired role/seed blocks。
- **Effect Direction / Magnitude:** Sample 相对 Equal 缩小 L-role 损伤约 0.0143 AUC；S2 KuaiRand tab1 click Equal−SD −0.045006 ± 0.000602，4/4 同向、5/5 common epochs 为负。
- **Robustness / Scope:** Strong within tested objective, dataset, target, and checkpoint definitions；selected/common/final 并非同一 estimand。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 目标加权会改变观察到的域级结果；Sample 减轻了所测 Large-cell selected 损伤，但不证明 Sample 对所有域/任务普遍最优或机制已识别。
- **Alternative Explanations:** Effective objective coefficient、采样 exposure、dataset split/label shift、checkpoint selection、task/domain interaction。
- **Evidence Strength:** STRONG。

#### E005 — comp1 的早期 residual 存在，但不证明 domain-conditioned momentum 独特

- **Observation:** P0-5 Sample+comp1 的 residual AUC 为 +0.005407 ± 0.000620（12 paired blocks）。S0 中 comp1 相对 sample Adam 有正差，但相对 validation-selected uniform EMA 的 macro AUC 差为 −0.003033 ± 0.000815，4-seed 均值支持统一 EMA。
- **Source Experiments:** EXP009、EXP018、EXP020。
- **Source Reports:** [`P0_5_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P0_5_STAGE_REVIEW.md)、[`P6_ablation.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P6_ablation.md)、[`stage_S0_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S0_analysis.md)。
- **Dataset / Objective / Model:** MovieLens；P0/P6 历史目标与 S0 Sample；M3oE。
- **Seeds:** P0-5 12 paired blocks；P6 5；S0 4 seeds × 3 rotations。
- **Effect Direction / Magnitude:** 历史 residual 为正；S0 comp1−uniform-EMA macro AUC −0.003033 ± 0.000815。
- **Robustness / Scope:** Medium；不同阶段协议不能把数值直接合并；S0 是机制特异性更强的匹配对照。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 旧协议下 residual 确实出现；可接受的当前结论是它没有显示出相对可选 uniform-smoothing family 的独立优势，不是“EMA 在任何配置都无效”。
- **Alternative Explanations:** 训练 exposure、objective、验证集选点、平滑策略族、任务/域聚合。
- **Evidence Strength:** MEDIUM。

#### E006 — 已测 curriculum / schedule / adaptive calibration 未稳定胜过 Sample

- **Observation:** P1-2 forward curriculum−none overall AUC +0.000839 ± 0.001018（p=0.198）；E5 Forward 对 static Sample 的 fixed-final 正向 seed 为 1/4，未过预设 GO gate；E6 Adaptive-NoEMA 未过 margin，Adaptive-EMA 未过 remaining-domain safety gate；E7 因 E6 STOP 而 0 runs。
- **Source Experiments:** EXP010、EXP015–EXP017、EXP018（P6 component arms）。
- **Source Reports:** [`P1_2_STAGE_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P1_2_STAGE_REVIEW.md)、[`stage_E5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E5_analysis.md)、[`stage_E5_go_no_go.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E5_go_no_go.md)、[`stage_E6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E6_analysis.md)、[`stage_E6_go_no_go.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E6_go_no_go.md)、[`stage_E7_gate_stop.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/stage_E7_gate_stop.md)。
- **Dataset / Objective / Model:** MovieLens 与 KuaiRand；M3oE；比较围绕 Sample baseline。
- **Seeds:** P1-2 为 4-seed 配对设计；E6 为 4 seeds、24 runs；E7 未运行。
- **Effect Direction / Magnitude:** Mixed / 未达到适用 gate；E6 Adaptive-EMA overall AUC +0.001584，但 Medium-domain safety gate 失败。
- **Robustness / Scope:** Strong against the claim that these tested variants passed their own gates；不覆盖所有 schedule/controller。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 在报告覆盖的实现、预算与 gate 内，未建立课程或自适应方法稳定超过静态 Sample 的证据。
- **Alternative Explanations:** 预算、schedule 范围、域定义、selected vs common/final 指标、单个配置失配。
- **Evidence Strength:** STRONG。

#### E007 — P7 组件迁移结果受骨干实现范围限制

- **Observation:** MovieLens D=3 的 comp1−none 平均 AUC：MMoE +0.002978、PLE +0.005538、ShareBottom 适配 +0.004012；STAR 因复现差距过大而未纳入 comp1 transfer 结论。
- **Source Experiments:** EXP019。
- **Source Reports:** [`P7_generalization.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/history_report/P7_generalization.md)。
- **Dataset / Objective / Model:** MovieLens；M3oE、MMoE、PLE、STAR、ShareBottom 本地适配；历史 objective。
- **Seeds:** 依子矩阵；ShareBottom 单独补跑 5 seeds × 2 arms。
- **Effect Direction / Magnitude:** 已测 MMoE/PLE/ShareBottom 均为小正值；STAR 不可比。
- **Robustness / Scope:** Weak for cross-model mechanism；适配结构、跑批窗口和复现状态不同。
- **Current Status:** TENTATIVE。
- **Current Interpretation:** 仅能报告已测移植条件的表现；不构成 S3–S6 shared-access/history mechanism 的跨模型复制。
- **Alternative Explanations:** 参数量、层适配、任务数 D/T、跑批窗口、objective、初始化。
- **Evidence Strength:** WEAK。

#### E008 — S0 没有支持 comp1 相对统一 EMA 的独特机制优势

- **Observation:** S0 comp1−sample Adam 的 macro AUC 差 +0.005307 ± 0.000523；comp1−validation-selected uniform EMA 为 −0.003033 ± 0.000815，3 个 rotation 平均后每个 seed 都是 uniform EMA 更好。
- **Source Experiments:** EXP020。
- **Source Reports:** [`stage_S0_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S0_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** MovieLens 强不均衡；Sample；M3OE。
- **Seeds:** 4，且每 seed 先平均 3 个角色轮换。
- **Effect Direction / Magnitude:** comp1 对 Adam 正；对可选 uniform EMA 负。
- **Robustness / Scope:** Strong within matched exposures and tested controls。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 残余效应可复现，但被统一平滑对照覆盖；不能再称 domain-conditioned momentum 的独立收益已建立。
- **Alternative Explanations:** 平滑强度、选点准则、rotation aggregation、特定 MovieLens 配置。
- **Evidence Strength:** STRONG。

#### E009 — Sample joint training 在 S1 测试范围内未显示稳定实用 residual NT

- **Observation:** 192 个运行覆盖两个数据集、三个 backbone 和单域/联合设置；288 paired seed units 中，72 组汇总有 63 正、9 负，但 0/72 达到稳定 practical NT 门槛。最负候选约 −0.002923。
- **Source Experiments:** EXP021。
- **Source Reports:** [`stage_S1_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S1_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** MovieLens 与 KuaiRand；Sample；M3OE/MMoE/PLE。
- **Seeds:** 4 per configuration。
- **Effect Direction / Magnitude:** Overall joint−SD 有利；少数 per-cell 小负值低于预设 practical threshold。
- **Robustness / Scope:** Strong for the preregistered Sample + SD-reference matrix; not all objectives or ratios。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 不支持把普遍 Sample residual NT 作为当前已确认问题；不证明每一 cell 或未测极端 ratio 均无损伤。
- **Alternative Explanations:** SD reference variance、backbone、dataset、cell/sample counts、selected/common checkpoint。
- **Evidence Strength:** STRONG。

#### E010 — S2 Equal 的局部 selected loss 与 Sample 的缓解均受目标/数据条件约束

- **Observation:** S2 Equal−SD 的 8/24 cell groups 达到 ≤−0.003，严格 gate 3/24；全部八组位于 Large role。KuaiRand tab1 click 为 −0.045006 ± 0.000602（4/4 seeds、5/5 common epochs）；MovieLens d2 rating>3 为 −0.018093 ± 0.004687（4/4）。Sample−SD 的 0/24 cell groups 达阈值。
- **Source Experiments:** EXP022。
- **Source Reports:** [`stage_S2_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S2_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural 与 MovieLens d2_large；Equal/Sample；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** Equal 下目标/角色依赖；Sample 的 selected threshold 结果近零或正。
- **Robustness / Scope:** Strong for the identified KuaiRand Equal/tab1 cell; medium for cross-dataset size-role generalization because MovieLens late/common views can change direction。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 等权目标下有窄范围、可重复的 selected large-cell loss；不应改写成所有“大域受损”或“Equal 一概错误”。
- **Alternative Explanations:** Objective weights、dataset/target identity、基线、收敛和 checkpoint selection。
- **Evidence Strength:** STRONG。

#### E011 — S3 在 KuaiRand Equal/tab1 中识别 shared-access 对 target 的条件性影响

- **Observation:** TARGET_ONLY_SHARED−FULL 在 selected target 上 click/long_view AUC 分别 +0.038876 ± 0.003950 / +0.039576 ± 0.004107（4/4）；相应 interaction 为 +0.038579 / +0.039440（4/4）。Sample 下相同干预约 +0.0003/+0.0001。
- **Source Experiments:** EXP023。
- **Source Reports:** [`stage_S3_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S3_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal 与 Sample；M3OE；target=d1/tab1。
- **Seeds:** 4；8 个严格配对 blocks 覆盖两个数据集与两个 objective。
- **Effect Direction / Magnitude:** Equal 下 target rescue 大；Sample 下接近零。
- **Robustness / Scope:** Strong within KuaiRand/natural/M3OE/Equal/tab1 and selected endpoint。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** shared parameters 的跨域 update access 是该 Equal 条件 target 结果的重要因果因素；该对照本身不证明纯几何 gradient conflict。
- **Alternative Explanations:** 更新组成、objective coefficients、optimizer history、checkpoint differences、domain/task interaction。
- **Evidence Strength:** STRONG。

#### E012 — S3 MovieLens selected replication 有 late-training 与 non-target 例外

- **Observation:** MovieLens Equal selected target ages_35_45_50_56 的 access rescue 为 +0.019177（task0）/+0.014118（task1），4/4 同向；macro/non-target 指标不总是同向，后期训练有收敛例外。
- **Source Experiments:** EXP023。
- **Source Reports:** [`stage_S3_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S3_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large；Equal/Sample；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** selected target 正；non-target 有代价，common/final 方向依阶段。
- **Robustness / Scope:** Medium; one dataset and target, with later checkpoint counterexamples。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** 该方向在 MovieLens selected 点也出现，但不足以升级为跨数据集、跨训练阶段的普遍安全收益。
- **Alternative Explanations:** checkpoint、过拟合、dataset-specific target/task、non-target aggregate choice。
- **Evidence Strength:** MEDIUM。

#### E013 — raw-gradient norm 与 applied-step norm 匹配都没有复现 S4 target rescue

- **Observation:** KuaiRand d1/tab1 中 N−F 两任务 AUC −0.004048/−0.004873（0/4 正）；U−F 为 −0.000587/−0.000942（2/4 正）；T−F 为 +0.038876/+0.039576（4/4 正，且 5/5 common epochs 为正）。
- **Source Experiments:** EXP024。
- **Source Reports:** [`stage_S4_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S4_analysis.md)、[`stage_S4_A2_review.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S4_A2_review.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** T rescue 稳定；两幅度控制未复现。
- **Robustness / Scope:** Strong; execution/control audits passed for the canonical A2 matrix。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** 匹配这两种幅度统计不足以解释 S4 的 rescue；梯度组成/方向与持久 history 的联合解释仍需按后续 S5 限定。
- **Alternative Explanations:** m/v history、梯度方向、每臂参数轨迹、objective-specific behavior。
- **Evidence Strength:** STRONG。

#### E014 — S4 non-target 代价与 d2 late gain 是 first-class trade-off

- **Observation:** T−F 在 KuaiRand d0 selected 两 task 约 −0.0314/−0.0321；d2 从 selected 混合方向转为 final +0.019810/+0.036160（4/4），且 LogLoss final 明显降低。
- **Source Experiments:** EXP024。
- **Source Reports:** [`stage_S4_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S4_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** selected non-target damage; late d2 gain。
- **Robustness / Scope:** Medium; domain/task and checkpoint dependent; domain sizes were not experimentally changed here。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** target rescue 与全域表现并不等价；late d2 result cannot support a monotonic domain-size claim。
- **Alternative Explanations:** checkpoint selection、训练阶段、域身份、过拟合和 history effects。
- **Evidence Strength:** MEDIUM。

#### E015 — S5 Equal KuaiRand d1 rescue follows persistent Adam history

- **Observation:** T−F selected d1 AUC +0.041426/+0.042254（4/4）；HT−F +0.041085/+0.040528（4/4，约恢复 96–99%）；DH−F +0.000875/+0.000821（约 2%）。S5 moment gate 为 15/16；seed 2024/epoch1/m/HT 有限定。
- **Source Experiments:** EXP025。
- **Source Reports:** [`stage_S5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_analysis.md)、[`stage_S5_review.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_review.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural；Equal；M3OE；target d1/tab1。
- **Seeds:** 2022–2025（4）。
- **Effect Direction / Magnitude:** HT follows T; DH remains close to F on target selected.
- **Robustness / Scope:** Strong within the tested Equal / d1 / M3OE configuration; keep moment-gate exception visible。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** 在该条件下，d1 rescue 跟随完整 persistent history carrier；不识别 m、v 或 bias-correction 的单独作用。
- **Alternative Explanations:** Objective、parameter trajectory、moment component、checkpoint rules、seed 2024 deviation。
- **Evidence Strength:** STRONG。

#### E016 — S5 Equal 的 d0 history×current interaction 与 d2 late gains 有 seed/stage dependence

- **Observation:** KuaiRand d0 T−F selected 为 −0.027338/−0.037392（0/4 正）；HT−F 均值较小但 seed 2024 仍明显为负，final HT−F 转为 +0.017840/+0.023840（4/4）。d2 HT−F selected/final 分别增加，final 4/4 正。
- **Source Experiments:** EXP025。
- **Source Reports:** [`stage_S5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural；Equal；M3OE。
- **Seeds:** 4。
- **Effect Direction / Magnitude:** Mixed by domain and checkpoint; HT mitigates selected d0 cost on average; late d2 benefit。
- **Robustness / Scope:** Medium; the seed 2024 exception and selected/final reversal matter。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** history×current interaction is a viable description for the tested trajectory, not a general safety or size law。
- **Alternative Explanations:** seed variance、checkpoint timing、domain/task identity、objective、optimization trajectory。
- **Evidence Strength:** MEDIUM。

#### E017 — S5-ML Equal target rescue partially replicates S5, while DH is not zero

- **Observation:** MovieLens d2 target T−F selected +0.019408/+0.014270（4/4）；HT−F +0.018114/+0.014149（4/4）；DH−F +0.005859/+0.003583（2–3/4）。晚期 DH−F final/common target AUC 约 +0.0098/+0.0103（4/4），而 T rescue 衰减。
- **Source Experiments:** EXP026。
- **Source Reports:** [`stage_S5ML_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5ML_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large；Equal；M3OE。
- **Seeds:** 4；16 canonical runs；moment gate 8/8。
- **Effect Direction / Magnitude:** Selected F→T rescue replicated; HT≈T holds; DH≈F does not fully hold and is checkpoint dependent。
- **Robustness / Scope:** Medium; mechanism factorization differs from KuaiRand and late MovieLens is overfit。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** history-dominant HT half replicates; the complete “current effect is zero under FULL history” pattern does not generalize to this dataset.
- **Alternative Explanations:** imbalance severity、task structure、late overfitting、checkpoint timing、domain identity。
- **Evidence Strength:** MEDIUM。

#### E018 — S6 Phase A Sample F/T 未达到两个预注册 target practical gates

- **Observation:** KuaiRand d1 selected T−F −0.000657/−0.000790（仅 1/4 正）；MovieLens d2 +0.001899/+0.002555（2/4 正）；四项均低于 +0.003。共同训练阶段也未出现稳定 target rescue。Sample T 对 selected non-target cells 有损伤。
- **Source Experiments:** EXP027。
- **Source Reports:** [`stage_S6_phaseA_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseA_analysis.md)、[`stage_S6_phaseA_final_analysis_20260929.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseA_final_analysis_20260929.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 每数据集 4；共 16 新 runs。
- **Effect Direction / Magnitude:** KR small negative; ML small positive; neither meets the practical gate。
- **Robustness / Scope:** Strong for the preregistered Phase A STOP decision; does not establish an exact zero effect。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** S5 Equal 的 target rescue 不能外推为 Sample objective 下同样大小的 selected rescue。
- **Alternative Explanations:** Seed variance、objective coefficient、target identity、sample split、checkpoint selection。
- **Evidence Strength:** STRONG。

#### E019 — S6 Phase B Sample selected target 仍无 practical rescue；KuaiRand M1 未通过

- **Observation:** 四臂 selected target T−F、HT−F、DH−F means 在两数据集均未达到 +0.003；KuaiRand HT selected LogLoss 对 target 4/4 改善但 AUC 无 practical mean rescue。KuaiRand early m conjunction 仅 4/8 通过、v 8/8；MovieLens m/v 均 8/8。代码/执行及配置一致性审查通过，但 KuaiRand 完整 M1 identification gate 为 FAIL。
- **Source Experiments:** EXP028。
- **Source Reports:** [`stage_S6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_analysis.md)、[`stage_S6_phaseB_audit_review.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseB_audit_review.md)、[`stage_S6_phaseB_run_report.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_phaseB_run_report.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 4 per dataset; 16 new HT/DH runs plus Phase A F/T anchors。
- **Effect Direction / Magnitude:** Selected target mixed and below threshold; metric-specific effects remain。
- **Robustness / Scope:** Medium; KuaiRand optimizer-history interpretation is limited by M1 failure; do not treat structural validity as all gates passed。
- **Current Status:** CONFIGURATION_SPECIFIC。
- **Current Interpretation:** Sample 下 selected target practical gate 继续为 STOP；完整的 history/current mechanism 在 KuaiRand 不能作无条件 identification claim。
- **Alternative Explanations:** Arm-specific parameter paths、effective weight decay、seed variance、AUC/LogLoss difference、checkpoint selection。
- **Evidence Strength:** MEDIUM。

#### E020 — S6 Sample MovieLens late DH target gain accompanies severe F deterioration

- **Observation:** MovieLens d2 target DH−F at common epoch39/49 approximately +0.003138/+0.010500 (task0) and +0.008317/+0.008107 (task1); final target LogLoss differences −0.073890/−0.045756 (4/4). Meanwhile F target AUC falls from selected .721797/.739511 to final .693747/.710896, and its LogLoss rises substantially. KuaiRand does not show the same late target rescue.
- **Source Experiments:** EXP028; S6 DH trajectory reanalyses reuse this matrix.
- **Source Reports:** [`stage_S6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_analysis.md)、[`stage_S6_ML_DH_trajectory_reanalysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_ML_DH_trajectory_reanalysis.md)、[`stage_S6_KR_DH_trajectory_reanalysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_KR_DH_trajectory_reanalysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large and KuaiRand d1; Sample; M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** MovieLens late DH target metric positive; selected target rescue null; KuaiRand late target pattern absent。
- **Robustness / Scope:** Medium; fixed late checkpoints, dataset specific, and F is severely overtrained at MovieLens late epochs。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** This is evidence of a late-stage trajectory effect under the tested protocol; it is not evidence that a method improves the normally selected model.
- **Alternative Explanations:** Ordinary overfitting/overtraining、checkpoint timing、history/current policy、metric/target mismatch。
- **Evidence Strength:** MEDIUM。

#### E021 — Equal/Sample history effects are objective dependent, but “Sample removes every optimizer effect” is too broad

- **Observation:** Equal target T−F selected: KuaiRand +0.041426/+0.042254; MovieLens +0.019408/+0.014270. Under Sample, Phase A T−F is near zero or below threshold in both datasets. Phase B later shows MovieLens common-late DH−F target gains, while selected Sample target contrasts remain below threshold.
- **Source Experiments:** EXP025–EXP028。
- **Source Reports:** [`stage_S5_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5_analysis.md)、[`stage_S5ML_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S5ML_analysis.md)、[`stage_S6_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S6_analysis.md)、[`exp_sn.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/exp_sn.md)。
- **Dataset / Objective / Model:** KuaiRand and MovieLens; Equal vs Sample; M3OE。
- **Seeds:** 4 per dataset/stage。
- **Effect Direction / Magnitude:** Equal selected rescue is large; Sample selected rescue is below practical gate; late MovieLens Sample DH pattern remains metric/stage specific。
- **Robustness / Scope:** Medium; matched objectives and dataset targets differ; S6 M1 limitation applies to KuaiRand Phase B。
- **Current Status:** SUPERSEDED（for the broad claim that all optimizer/history effects disappear under Sample）。
- **Current Interpretation:** Current safe claim is that the Equal selected target-rescue pattern does not transfer at practical magnitude to the tested Sample selected target gates; other late optimizer-policy effects are not all zero。
- **Alternative Explanations:** Objective aggregation、overfitting、domain size and identity、AUC vs LogLoss、selected vs common checkpoint。
- **Evidence Strength:** MEDIUM。

#### E022 — S6 derived trajectory analyses show non-target checkpoint effects, not independent replications

- **Observation:** MovieLens DH−F test non-target gain at selected checkpoint has 4-seed mean approximately 0, but its longest common-epoch run above +0.003 spans K=5; KuaiRand selected non-target mean is +0.0068, while its qualifying common run is K=3 and is mixed by domain.
- **Source Experiments:** EXP028 (same source runs; no new training).
- **Source Reports:** [`stage_S6_ML_DH_trajectory_reanalysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_ML_DH_trajectory_reanalysis.md)、[`stage_S6_KR_DH_trajectory_reanalysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_KR_DH_trajectory_reanalysis.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand; Sample; M3OE。
- **Seeds:** 4 each; five common checkpoint epochs in trajectory reports。
- **Effect Direction / Magnitude:** Selected and common-stage views differ; MovieLens has a late non-target pattern, KuaiRand is weaker/mixed.
- **Robustness / Scope:** Medium for the descriptive trajectory; not independent replication.
- **Current Status:** SUPERSEDED (as a claim of normal-checkpoint method benefit by S8); underlying fixed-epoch observations remain in the source record。
- **Current Interpretation:** Stage-dependent test observations must be retained, but S8's best-vs-original-best test narrows their practical interpretation to overfit-trajectory rescue。
- **Alternative Explanations:** Overfit deterioration in F、different selected epochs、non-target weighting、domain-specific response。
- **Evidence Strength:** MEDIUM。

#### E023 — S7-A cosine structure follows domain, while ConflictMass is not a stable performance marker

- **Observation:** Across all probe seeds/checkpoints, same-domain cross-task cosine exceeds cross-domain pair values; cross-domain cosine changes sign across epochs. ConflictMass has no robust shared stage trend (MovieLens split; KuaiRand falls in 3/4).
- **Source Experiments:** EXP029。
- **Source Reports:** [`stage_S7_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S7_analysis.md)。
- **Dataset / Objective / Model:** KuaiRand natural、MovieLens d2_large；Sample；M3OE。
- **Seeds:** 4 per dataset; 8 probe runs。
- **Effect Direction / Magnitude:** Domain-structured cosine; no stable monotonic cross-domain conflict trend。
- **Robustness / Scope:** Medium; replicated across snapshots within these two datasets, but measurements are local gradient geometry。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** Domain×Task decomposition reveals structure hidden by task aggregation; cosine/ConflictMass alone does not establish harmful performance effect。
- **Alternative Explanations:** Gradient norm scale、module reachability/dead ReLU、sample weighting、dataset-specific task structure。
- **Evidence Strength:** MEDIUM。

#### E024 — S7-A validation-gradient utility is performance-aligned in late MovieLens, not universally sufficient

- **Observation:** MovieLens F target performance deteriorates from early to late (target AUC −0.0208; 4/4); U_k becomes increasingly negative over later epochs. KuaiRand target AUC continues rising to epoch49 (4/4), even though target U_k is negative in 3/4 seeds at that snapshot.
- **Source Experiments:** EXP029。
- **Source Reports:** [`stage_S7_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S7_analysis.md)。
- **Dataset / Objective / Model:** MovieLens d2_large and KuaiRand natural；Sample；M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** MovieLens late U_k and validation deterioration co-move; KuaiRand negative U_k does not coincide with realized AUC damage。
- **Robustness / Scope:** Medium; observations are stable inside MovieLens but the KuaiRand counterexample limits causal sufficiency。
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** U_k is a conditional local first-order statistic; a negative value is not sufficient for realized performance decline.
- **Alternative Explanations:** Adam preconditioning、multi-step trajectory、overfitting、gradient norm growth、metric noise。
- **Evidence Strength:** MEDIUM。

#### E025 — S7-B changes late MovieLens gradient utility without validation improvement

  - **Observation:** At same θ, MovieLens anchor39 target-current changes min-U on steps1–4 positively for all four seeds (approximately +0.0074 to +0.169); anchor9 is near zero/mixed. At anchor39 step5, target AUC mean difference is −0.001226 but only 3/4 seeds are negative (seed2025 is +0.002359); LogLoss mean is +0.011130 with 3/4 worse. KuaiRand has no stable early/late response contrast. S7-B uses fresh-Adam private parameters after step0, a declared approximation.
- **Source Experiments:** EXP030。
- **Source Reports:** [`stage_S7B_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S7B_analysis.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand; Sample; M3OE。
- **Seeds:** 4 per dataset × 2 anchors (16 runs)。
- **Effect Direction / Magnitude:** Gradient utility changed at late ML anchor; short-window validation did not improve.
- **Robustness / Scope:** Medium; at anchor39 step5 AUC is negative in 3/4 seeds and LogLoss is worse in 3/4, the window is short, and private optimizer state is approximate。
- **Current Status:** SUPPORTED（for the short-window observation）。
- **Current Interpretation:** Gradient-geometry change is not enough to infer validation benefit; the proposed immediate gradient-to-performance chain is weakened。
- **Alternative Explanations:** Short horizon、private-state approximation、Adam effective update、ordinary overfitting trajectory。
- **Evidence Strength:** MEDIUM。

#### E026 — S8 finds no practical continuation benefit at selected checkpoints

- **Observation:** S8 selected-anchor h=2-epoch continuation T−F and DH−F mean best-AUC differences are both +0.000062 on MovieLens; KuaiRand T−F −0.000445 and DH−F +0.000068. These are below +0.003; no MovieLens branch consistently sets a new best above the frozen F original best.
- **Source Experiments:** EXP031。
- **Source Reports:** [`stage_S8_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S8_analysis.md)、[`S8_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/S8_REVIEW.md)。
- **Dataset / Objective / Model:** KuaiRand and MovieLens；Sample；M3OE。
- **Seeds:** 4 per dataset; 8 base runs and 24 continuations。
- **Effect Direction / Magnitude:** Selected-anchor branch contrast null / far below threshold。
- **Robustness / Scope:** Strong for the tested seeds, anchors, branches, 2-epoch horizon, and frozen-best endpoint; the run validation gates passed。
- **Current Status:** SUPPORTED。
- **Current Interpretation:** The optimizer/current-source interventions did not improve normally selected performance to a practical magnitude in this protocol。
- **Alternative Explanations:** Narrow continuation horizon、selected-state proximity to optimum、dataset-specific overfit behavior。
- **Evidence Strength:** STRONG。

#### E027 — S8 late MovieLens gain is an overfit rescue pattern, not a better best model

- **Observation:** MovieLens late-anchor DH−F best-window macro AUC +0.006826 (4/4) and final-step +0.010732 (4/4), but about 89% of the latter mean comes from seeds 2024/2025. Target-domain gain is much smaller (+0.000441 best; +0.002566 final), and every continuation best remains below that seed's frozen F original best. KuaiRand late T is harmful (final −0.010239, 0/4 positive) and DH is near zero.
- **Source Experiments:** EXP031; reinterprets E020/E022.
- **Source Reports:** [`stage_S8_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/stage_S8_analysis.md)、[`S8_REVIEW.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/reports/S8_REVIEW.md)。
- **Dataset / Objective / Model:** MovieLens and KuaiRand；Sample；M3OE。
- **Seeds:** 4 per dataset。
- **Effect Direction / Magnitude:** Late MovieLens macro DH appears positive; target/best endpoint is small; no branch beats original best. KuaiRand is a negative control.
- **Robustness / Scope:** Medium; late macro effect is seed-skewed and metric dependent; execution was fully validated.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The evidence supports partial recovery from an overtrained late anchor, not a method that improves checkpoint-selected model performance.
- **Alternative Explanations:** Ordinary overfitting/overtraining、non-target macro contribution、seed leverage、checkpoint horizon。
- **Evidence Strength:** MEDIUM。

#### E028 — KuaiRand DH has a selected-checkpoint test-side non-target gain under Sample

- **Observation:** At the canonical R0 validation-macro selected checkpoints, KuaiRand test Δ(DH−F) is N +0.0068 ± 0.0046 (4/4 positive; 3/4 ≥ +0.005), M +0.0044 ± 0.0029 (4/4 positive), and T −0.0004 ± 0.0031. Seed 2023 has target ΔT = −0.0047. On the six-point A-grid sensitivity variant, R1/R3/R4 test ΔN means are +0.0041/+0.0060/+0.0041, each 3/4 positive; their mean target deltas remain above −0.003. Exact test availability for truly reselected R1/R3/R4 is 0/4, 0/4, and 1/4 seeds respectively, so those results do not establish full-seed exact-rule robustness.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partA_gate_summary_kuairand.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partA_gate_summary_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Positive test non-target and macro means at canonical R0; target mean is near zero/slightly negative. Alternative-rule test robustness is A-grid sensitivity only.
- **Robustness / Scope:** Medium; R0 N is 4/4 positive, but seed 2025 is near zero, one target seed crosses −0.003, and the alternative rules rely on a restricted checkpoint grid.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The tested KuaiRand/Sample/M3OE DH checkpoints have a test-side selected non-target advantage relative to F. This is not a validation-side advantage or evidence of a universal optimizer mechanism.
- **Alternative Explanations:** Validation-to-test generalization gap、checkpoint timing、selection rule、seed variation。
- **Evidence Strength:** MEDIUM。

#### E029 — KuaiRand selected-checkpoint advantage reverses between validation and test

- **Observation:** At the same R0-selected checkpoint pairs, KuaiRand ΔN is −0.0039 ± 0.0081 on validation (1/4 positive) and +0.0068 ± 0.0046 on test (4/4 positive); ΔM is −0.0027 on validation and +0.0044 on test. The validation decomposition has N PolicyAtF = −0.0111 and PolicyAtDH = +0.0082 (four-seed means), with TimingAlongF = −0.0120. Test cross-epoch decomposition is unavailable because canonical epochs are off the saved grid; the saved same-epoch test N difference is positive at 16/20 seed×epoch points.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partB_decomposition_kuairand.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partB_decomposition_kuairand.csv)、[`s6r2_partB_grid_trajectory_kuairand.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partB_grid_trajectory_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Validation non-target/macro means favor F; test non-target/macro means favor DH; target test mean is −0.0004.
- **Robustness / Scope:** Medium for the layer-specific observations; the causal source of the val–test reversal is unresolved, and the test decomposition is blocked by missing off-grid checkpoints.
- **Current Status:** UNRESOLVED。
- **Current Interpretation:** The direction depends on evaluation layer at the same selected checkpoints. The result does not show a validation-visible improvement or identify why the held-out test gap changes direction.
- **Alternative Explanations:** Checkpoint timing、validation/test distribution differences、seed-level heterogeneity。
- **Evidence Strength:** MEDIUM。

#### E030 — MovieLens is a selected-checkpoint negative control in S6-R2

- **Observation:** Under MovieLens R0, selected test ΔN is approximately −0.000002 ± 0.0031 (2/4 positive), ΔM is approximately 0.0000 ± 0.0029 (2/4 positive), and ΔT is +0.0001 ± 0.0026. No available exact or A-grid selection rule shows a stable test non-target gain; the saved-grid same-epoch ΔN reaches +0.0334 at epoch 19, while canonical selected epochs are 7–13.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partA_gate_summary_movielens.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/movielens/s6r2_partA_gate_summary_movielens.csv)、[`s6r2_partB_grid_trajectory_movielens.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/movielens/s6r2_partB_grid_trajectory_movielens.csv)。
- **Dataset / Objective / Model:** MovieLens d2_large; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Selected test effect is near zero; late saved-grid effects can be positive but do not coincide with selected epochs.
- **Robustness / Scope:** Medium within the four-seed negative-control setting; some alternative exact-rule test values are unavailable.
- **Current Status:** DATASET_SPECIFIC。
- **Current Interpretation:** The KuaiRand selected test result does not reproduce on MovieLens under the tested selection protocol; the dataset/stage contrast does not identify its cause.
- **Alternative Explanations:** Dataset and target identity、selected-epoch timing、training-stage differences、seed variation。
- **Evidence Strength:** MEDIUM。

#### E031 — KuaiRand DH selected validation windows are usually broader than F windows

- **Observation:** In the four-seed KuaiRand window analysis, DH has 3 moderate plateaus and 1 narrow spike; F has 1 moderate plateau and 3 narrow spikes. DH seed 2023 is itself a narrow spike, and DH seed 2024 is edge-clipped at epoch 48.
- **Source Experiments:** EXP032; underlying F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partC_window_kuairand.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partC_window_kuairand.csv)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE。
- **Seeds:** 4 paired seeds, 2022–2025。
- **Effect Direction / Magnitude:** Window classification favors more moderate DH than F selections (3/4 vs 1/4).
- **Robustness / Scope:** Weak; one dataset, four seeds, a categorical window rule, one DH spike, and one edge-clipped window.
- **Current Status:** TENTATIVE。
- **Current Interpretation:** A single narrow peak is not a sufficient description of all DH selections in this sample; the result does not establish generally broader or more stable DH optima.
- **Alternative Explanations:** Seed-specific peak shape、edge clipping、the pre-registered window threshold。
- **Evidence Strength:** WEAK。

#### E032 — KuaiRand cross-partition parameter swaps collapse and do not localize the gain

- **Observation:** On the four KuaiRand seeds, C_hybrid test macro ranges 0.5450–0.5842 and D_hybrid 0.5196–0.5791, versus F_full mean 0.6730 and DH_full mean 0.6774. E_hybrid closely matches C_hybrid; F_hybrid differs from F_full by at most 0.00011 macro and 0.00016 non-target. Seed 2025 has a DH−F non-target denominator of about 1.1e−5, so its retention ratio is not informative. The manifest partitions and state-dict shapes pass the reported integrity checks.
- **Source Experiments:** EXP032; underlying KuaiRand F/DH source checkpoints in EXP027/EXP028。
- **Source Reports:** [`stage_S6_R2_selected_solution_analysis.md`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_selected_solution_analysis.md)、[`s6r2_partD_swap_kuairand.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partD_swap_kuairand.csv)、[`s6r2_integrity_report.json`](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json)。
- **Dataset / Objective / Model:** KuaiRand-Pure natural; Sample; M3OE; manifest split into shared, architecture, and private parameter sets。
- **Seeds:** 4, 2022–2025; evaluation-only swaps, no new training。
- **Effect Direction / Magnitude:** Both cross-partition hybrids have large negative test-macro changes relative to the intact parent solutions; the F architecture-only hybrid is nearly unchanged from F_full.
- **Robustness / Scope:** Medium for these tested swaps across four seeds; one seed's retention ratio is undefined in practice and the result is limited to one model/data/partition definition.
- **Current Status:** SUPPORTED（for the observed swap outcome）。
- **Current Interpretation:** The tested partitions are not independently interchangeable and are consistent with cross-partition co-adaptation. These swaps do not identify which component causes the intact DH test-side advantage.
- **Alternative Explanations:** Cross-partition co-adaptation、parameter-manifest boundary、seed 2025's near-zero denominator。
- **Evidence Strength:** MEDIUM。

#### Strength tally

| Evidence Strength | Count |
|---|---:|
| STRONG | 10 |
| MEDIUM | 19 |
| WEAK | 3 |
| **Total** | **32** |


---

#### 2026-10-15—2026-10-18 分支执行增量（不回写 E001–E032 的历史记录）

#### E033 — KR Sample F/DH B1 重建未稳定复制旧 S6-R2 test 非目标增益

- **Observation:** 四seed×F/DH 共8/8新50epoch训练，400完整状态/800预测工件数值核验。R0 selected test 宏ΔAUC +0.000363（2/4正）、非目标 +0.000579（2/4正）、target −0.000069，target LL +0.004007。旧 S6-R2 非目标 test +0.006757（4/4正）、validation −0.003851。
- **Source:** [10-15 B1 scientific_analysis](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-15/research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016/scientific_analysis.md)。
- **Scope:** KR natural / M3OE / Sample，四seed、50epoch、validation macro R0选轮。**RECONSTRUCTION_NOT_HISTORICAL_EXACT**（init/lineage不同）；原test不是全新holdout。
- **Evidence Strength:** STRONG（工件/数字），MEDIUM（没有稳定实用selected改善的本研究判断），WEAK（旧新差异原因）。
- **Contrary evidence:** R1/R4 test 非目标 +0.011584、4/4正，但选同批epoch且target LL损伤；同轮次 val/test 也可反号。不能以 R0阴性擦除次要阳性或以R1事后重定主规则。

#### E034 — 固定目标同 q 干预的数学身份与有限θ局部 optimizer 差异

- **Observation:** C2.1固定θ两dataset×4seed×8case，以及 Stage2B/2C新8case×4096 replay×64步验证同α不同q期望梯度一致但covariance/Adam局部应用更新可以不同。SGD的平均解析身份和样本路径分叉需区分；完整C与投影坐标 trace 可能相反（KR Sample：所选坐标比0.505344、旧完整C比2.497026）。
- **Source:** [10-16 experiment_results Stage2B/C](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/execution_20261016/experiment_results.md)；[10-15 fixed θ](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-15/research_plans/2026-10-15/execution_20261015/C2_numeric_oracle_20261016/numeric_analysis.md)。
- **Scope:** 有限pool、fixed θ、给定梯度图/少量所选坐标；**0新增正常训练**。Expected objective 恒等不等于模型选点等价；不证明S5历史必要性或普适降噪。
- **Evidence Strength:** STRONG（数值/身份），WEAK（长期性能中介）。

#### E035 — KR Stage2D 16/16 实训：same-J Balanced 候选损伤而非无效应

- **Observation:** 16/16配对IID正式训练完成。validation macro selected SB−SN = −0.010406（4/4负），test法证−0.012674；EB−EN validation −0.006547（3/4负），test−0.002714。原预注册收益和nonharm门不通过。Equal某些域/LL与宏AUC方向不一致。
- **Source:** [10-16 Stage2D experiment_results](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/experiment_results.md)；[final execution](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/execution_report.md)。
- **Scope:** KR/M3OE；learned fusion / main+arch双Adam；2022–2025；50epoch、独立validation selection。q的总体干预同时改变重复、unique支持、梯度统计与后续θ；历史Stage2+文档“0run”是同分支较早快照。
- **Evidence Strength:** STRONG（协议内负向重复和完成状态）；WEAK（归因为Adam m/v或唯一曝光中介）。

#### E036 — v2 fixed-fusion SGD 反例否定“Adam 对 q 效应通用必要”

- **Observation:** P1 validation selected SB−SN：Adam −0.005556，3/4负，CI[−0.012286,+0.005721]；SGD −0.002104，4/4负，CI[−0.002478,−0.001680]。两optimizer interaction CI跨0。SGD LR=.1为[.1,1,10]校准下边界。
- **Source:** [10-16 v2 mechanism_analysis](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v2/research_plans/2026-10-16_v2/execution_20261016/mechanism_analysis.md)；[P1 decision](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v2/research_plans/2026-10-16_v2/execution_20261016/p1_decision.json)。
- **Scope:** fixed fusion / single optimizer，新协议不同E035双Adam；v2正式最终报告未归档完成，P2/P8中期不可当最终。
- **Evidence Strength:** STRONG（该SGD设置的反例存在），WEAK（归纳所有SGD/LR或声称Adam与SGD等效）。

#### E037 — v3 complete：同目标 q 效应在optimizer、LR和checkpoint间有反号

- **Observation:** 原规定/触发实验完成、综合分类BOUNDARY-CONFIRMED、候选方法NO_GO。冻结test selected SB−SN：固定Adam.01 −0.007069（2/4负）、固定SGD.1 −0.002518（4/4负）、校准Adam.003 −0.008698（4/4负）、校准SGD.3 −0.000972（4/4负）、ML SGD.3 +0.000539（1/4正）、AdaTask.003 −0.018917（4/4负）。calSGD共同37轮 +0.001614（4/4正）与selected −0.000972不同。
- **Source:** [v3 FINAL_REPORT §§1–4](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md)。
- **Scope:** 各家族不同LR/阶段/图定义，n4；测试为已开发并冻结后读取，不是纯净新holdout。SGD有反例，不推出所有 q 对SGD皆负。
- **Evidence Strength:** MEDIUM（多配置存在阶段/选择边界），WEAK（总体方法收益、跨域普适性）。

#### E038 — v3局部共享C/状态作用存在，但长期必要性与健康门未过

- **Observation:** fixedθ全坐标 q协方差不同；V7早/中C的局部selection AUC缩小差距筛查4/4合格，但loss/train J联合门不通过，C-private full training NOT_TRIGGERED。V11 m/v消融含beta2=0的灾难性健康floor；全局等norm不保持各参数块norm，不能识别纯方向、单一m/v必要性。真实C/E共享、R私有、P混合。
- **Source:** [v3 FINAL_REPORT §§6–9](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md)；[v3 V7局部](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/V7_local_counterfactual_analysis.md)。
- **Scope:** 给定theta/状态/冻结候选及阈值；不能用“full-training 未触发”宣称C机制被否定。
- **Evidence Strength:** MEDIUM（局部路径），WEAK（长期唯一中介）；STRONG（参数支持及协议边界）。

#### E039 — N17 的 KR 主选点 q 效应未过预设多重检验

- **Observation:** 新120条H100训练（48 tuning/72 formal）完成；正式固定test KR选点：Adam SB−SN −0.005203、8seed t95[−0.012006,+0.001600]、Holm p=.171875、7/8负；SGD +0.000837，CI[−0.004309,+0.005984]、Holm p=.6953125、5/8正。Adam bootstrap负区间不能事后代替预注册t/符号Holm主准则。
- **Source:** [N17 FINAL_ANALYSIS](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md)；[paired data](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_performance/paired_seed_endpoints.csv)。
- **Scope:** fixed fusion、Sample J、独立8 optimizer seeds2032–39、已开发历史test。Holm针对两KR主比较；次级AULC/ML/六cell不自动获得该主确认。
- **Evidence Strength:** STRONG（实训完成/数值），WEAK/INCONCLUSIVE（KR selected primary统计判断）；不得写“等效”“普遍改善/损伤”。

#### E040 — N17 长期预算与评价人口可使表面效应变号

- **Observation:** KR Adam早期 AULC1..20 −0.007480、8/8负，CI[−0.010031,−0.004930]；ML固定subset selected common−0.010392、own−0.010875、各8/8负，own E100则 **+0.021145，8/8正**。ML own对比 SN lr=.001 vs SB lr=.0003，既改q也改LR/训练速度，六cell保护未通过。相同历史calAdam E50预测的macro −0.009074、train-p cell +0.001016、test-p cell +0.003108。
- **Source:** [N17 FINAL_ANALYSIS](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md)；[metric/population decomposition](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/METRIC_AND_POPULATION_DECOMPOSITION.md)。
- **Scope:** 选点vs固定E100、early AULC、macro vs cell加权平均**都是不同估计对象**；加权cell AUC不等于 pooled AUC。
- **Evidence Strength:** STRONG（同预测评价权重算术）；MEDIUM（阶段/配置条件边界），WEAK（把E100解释为pure-q部署收益）。

#### E041 — N17 固定quota排除了“随机域计数主导固定θ trace 差”的窄解释

- **Observation:** N2 16有效fixedθ锚×四臂×512draw；SB/SN全部ALL trace比例：IID约2.009–2.584，quota约1.995–2.551；随机域计数项占预测trace的0.001297%–0.105936%。16条quota H100训练后选点q差 −0.009191、8seed t95[−0.015795,−0.002588]；quota×q交互 −0.003988，CI[−0.010310,+0.002333]。
- **Source:** [N17 DOMAIN_COUNT_COUNTERFACTUAL](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/DOMAIN_COUNT_COUNTERFACTUAL.md)。
- **Scope:** 固定锚点与给定梯度估计器；quota亦改变批内依赖/高阶矩，既不单独识别trace/performance中介，也不等同改变自然域真实数据集规模。原两项float32身份测试失败后仅更改独立验证逻辑、同seed同draw重验，原错误保存。
- **Evidence Strength:** STRONG（所测固定点反证）、MEDIUM（quota后的有限条件训练现象）、WEAK（唯一长期因果解释）。

#### E042 — N17 C局部独立验证不足以触发长期拓扑机制实验

- **Observation:** N3一次冻结embedding/E10候选，独立4seed两路径 full局部效应方向均4/4正、量级约+3e−5，MC精度条件达标，但实际界与方向门不过，**N4未触发**。这保留有限局部作用而非“C完全无效”；不能声称N4验证为NO。
- **Source:** [N17 C_LOCAL_AND_TOPOLOGY_ANALYSIS](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/C_LOCAL_AND_TOPOLOGY_ANALYSIS.md)。
- **Scope:** 同θ/optimizer状态、冻结微干预；与原 S5/S6 历史共享参数交换不是同一被解释效应。
- **Evidence Strength:** MEDIUM（局部实际界与未触发条件），WEAK（长期共享必要性）。

#### E043 — 文献边界与现有方法候选最终 NO_GO 不代表全研究命题关闭

- **Observation:** 15篇Stage2 targeted 和后续v3/N17原文核对已确认同目标重要性采样、variance-aware、Adam/SGD相互作用、动量/任务历史、顺序及selection是先例。D3/SAF原方法通常改变目标J，不能作为固定J改变q的严格同构反事实；AdaTask组件q-gap读数不代表完整方法最优效果。v3/N17没有候选同时通过预定macro收益和全cell/LL保护。
- **Source:** [v3 existing-work coverage](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md)；[N17 LITERATURE_COVERAGE](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/LITERATURE_COVERAGE.md)。
- **Scope:** 有限针对性查新（非全面检索）、已测协议与候选；未运行完整D3/SAF性能基线，不宣称这些论文方法失败。
- **Evidence Strength:** STRONG（已核对的公式边界），MEDIUM（本轮 NO_GO 决策），WEAK（主张全球独有创新）。


#### 2026-10-18—2026-10-26 跨轮证据E044–E057（日期和模型身份不可混用）

#### E044 — N18完成有界六方向探索，不等于六个方法都具独立贡献
- **Observation:** P1–P6进行数学/数据检查、冻结教师的四seed头拟合与实测诊断；当前候选没有同时取得预设强对照优势与实质独立方法贡献。某些方向标DATA-BLOCKED，按门未做长期训练。
- **Interpretation:** 相关科学问题和理论/局部反例仍可推进，不能把DATA-BLOCKED当作被负性能推翻。
- **Strength / Scope:** STRONG执行与受控构造，WEAK方法推广；N18固定人口、冻结头范围。
- **Source:** [N18 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)。

#### E045 — N19完整328次训练未确立普适新算法，保留容量与阶段机制事实
- **Observation:** 328正式训练、对应冻结测试完成；容量数值失活、通道误设、选点/末态、不同指标与总体方向反转有具体受控读数；没有稳定足够强的独立算法贡献证据。
- **Interpretation:** 当前候选NO_GO不等于所有域间收益/知识通道原理被否定。
- **Strength / Scope:** STRONG执行身份，MEDIUM受控现象，WEAK泛化；人工ML训练人口与固定设置不能代替自然完整ML。
- **Source:** [N19 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)。

#### E046 — CM19中KR fixed M3oE的输入乘性调制路径存在限定数值失活
- **Observation:** 856次全预算网络训练及冻结测试完成；KR fixed/Adam selected 24/24 cell中和输入乘性调制不改变预测（固定数值判据），final多数cell仍中性或极小响应。
- **Interpretation:** 不能推广为STAR/全部共享参数失活，也不能将中和预测不变等同于参数完全无梯度。
- **Strength / Scope:** STRONG（同实现/固定干预/人口），不支持跨模型普遍机制。
- **Source:** [CM19 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)。

#### E047 — 双factor正则/局部AdamW恢复路径响应但不保证selected性能
- **Observation:** CM19中同步去两组factor的耦合L2或局部替换AdamW可显著改变该路径的数值贡献；只改一组与两组不等价。STAR/PEPNet/HAMUR/MMoE/PLE存在活跃路径反例；若干结构移除从头重训与即时中和效果反号。
- **Interpretation:** 路径活性、训练必要性、相对强基线的最佳泛化是三个不同命题。AdamW是实现/状态特定干预，非所有共享问题的通用修复。
- **Strength / Scope:** MEDIUM–STRONG特定干预，WEAK通用性能因果。
- **Source:** [CM19 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)。

#### E048 — 历史ML训练人口与自然比例存在被审计确认的系统差异
- **Observation:** 10-15以来多轮固定ML人口187500（d0/d1/d2=34500/3000/150000），完整自然基础train 800167（155949/318599/325619），自然同N替代36543/74656/76301；自然支持量为人工4.2676倍。N19/CM19人工ML还以固定subset_seed2022保持优化seed独立。
- **Interpretation:** 不能把historical suite=natural字段视为完整自然训练人口；新旧比较同时改变数量、角色、特征支持和更新步数。研究者原本也未主张小域必然更难。
- **Strength / Scope:** STRONG人口/编码/身份；仅靠人口审计不确定任何性能机制。
- **Source:** [自然人口审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)。

#### E049 — 自然ML局部重评削弱旧selected强效应的外推，完整transfer仍未完成
- **Observation:** 自然matchedN/full下部分S5/S6 selected改善变负或很弱，同时full末态又出现有益响应；自然V2、N17及AO04有已完成可读结果。归档的CURRENT_WORK仍记录N19/CM19自然矩阵不完整。
- **Interpretation:** 不将差值唯一归于“比例”，不按末态替换selected；部分阴性不使原人工结果无效，也不能将设计+放行算为全矩阵完成。
- **Strength / Scope:** MEDIUM已完固定人口子项，STRONG执行状态，WEAK跨人口因果。
- **Source:** [阶段性重评](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/RESEARCH_CONCLUSION_REASSESSMENT.md)、[当前状态](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)。

#### E050 — D套件在自然MovieLens有重复收益，但独立因子尚未辨别
- **Observation:** 10-21在ML/KR共48有效正式训练；ML D−B test macro AUC +0.006303（4/4）、macro LL −0.015962（4/4）；对原生移植O也有有限收益。D包括specificity权重、最小域复制、embedding噪声及尺度改变。
- **Interpretation:** 可以保留“上传D套件在此协议有效”，不能说完整历史VUG/mask已复现、独立动态权重或增强必定是收益来源；O采用learned fusion双Adam而B固定fusion。
- **Strength / Scope:** STRONG（ML四seed特定协议），WEAK单因素归因；使用legacy_group_sum非标准Sample/Equal。
- **Source:** [M3oE复现REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。

#### E051 — 常数权重套件阳性与D权重接近下界削弱动态样本区分的必要性
- **Observation:** ML Dconst01−B AUC +0.005676（4/4）且LL −0.018445；D/G在epoch5观测批样本权重接近下界0.1。Dconst01−D的预设非劣条件**未通过**，不能当成“等效”。
- **Interpretation:** D的动态specificity不是获取ML套件收益的必要条件，但是否额外增益未完全判别；常数控制仍有复制/噪声/整体损失尺度，且D初期权重轨迹不同。
- **Strength / Scope:** STRONG受控套件存在性、MEDIUM机制限制。
- **Source:** [M3oE复现REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。

#### E052 — 匹配G梯度计算路径后τ=0.5未带来最终宏AUC优势
- **Observation:** G−Gtau0 test宏AUC，ML −0.003466（4/4下降，t描述区间负），KR −0.006484（4/4下降但区间跨0）；Gtau0−D在KR test方向相反。KR早期G反而有优化速度改善。
- **Interpretation:** G−D不是“只改变调幅”，还混入grad=None与零Tensor及逐格求导路径；G−Gtau0才是匹配对照。本协议终点不利不否定所有调幅理论，也不能忽略验证/test或pooled反向。
- **Strength / Scope:** STRONG ML对比方向、MEDIUM KR趋势（区间宽）、WEAK通用论断。
- **Source:** [M3oE复现REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。

#### E053 — 10-24 I1权限策略未获得selected主优势
- **Observation:** 44主pilot、4基线适配pilot、40正式训练均按记录完成；I1_PERM相对固定RW的ML/KR selected test macro AUC为−0.000334/−0.000513，描述性区间跨零；KR final LL有限正向。
- **Interpretation:** 未确认权限动态控制的独立selected改进；不能把final LL取代ranking主目标。
- **Strength / Scope:** MEDIUM本轮阴性门控，条件性次级改善保留。
- **Source:** [10-24 README](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/10-24/research_reports/2026-10-24/README.md)。

#### E054 — NEXT-v2核心pilot完成而正式NO_START是正确停止，不是阴性正式训练
- **Observation:** 48核心24epoch轨迹完成，全部晋级门失败；四seed/50epoch正式矩阵0条，部分aux/外部基线因预算停止。
- **Interpretation:** 当前版本STOP_NO_PROMOTION；不能虚构0条正式的AUC效应。
- **Strength / Scope:** STRONG执行状态，方法价值按已完成核心比较限定。
- **Source:** [v2 README](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-25/research_reports/2026-10-25/README.md)。

#### E055 — NEXT-v3 RHD的晚期风险减损不等于合法selected最优改善
- **Observation:** 69条24epoch与20条KR正式50epoch轨迹完成。KR正式HB−BASE test LL在selected +0.00074075、epoch50 −0.00875428（负才有利）；所有方法50epoch自身LL仍高于合法selected。
- **Interpretation:** 相对末态减损有条件中等证据，主要selected收益未建立；历史响应控制的某些候选动作未执行，不能用非零超参掩饰。
- **Strength / Scope:** MEDIUM KR正式四seed阶段比较，WEAK独立方法价值。
- **Source:** [v3 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)。

#### E056 — LPS来源内容可读不等于特定风险形成器必要
- **Observation:** v3 LPS部分foreign读取在开发存在有限信号，但L4未稳定胜FEATURE/简单标量，正式延伸未过gate；CAL/OWNER/GEOM及原型输出提供简单竞争解释。
- **Interpretation:** 不能因为关掉foreign后部分快照下降就证明meta形成/多函数必要；同时不能由形成器失败推出来源知识毫无价值。
- **Strength / Scope:** MEDIUM来源读取有限性；WEAK形成独有性。
- **Source:** [v3 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)。

#### E057 — DFD-v4：目标需求有限可预测，完整来源函数形成未晋级且有机制记录缺项
- **Observation:** ML P1 CONST−DEMAND LL +0.00108935，KR P1 CONST−DEMAND −0.000400568未过；ML P2 DFD−H_DIRECT/ETA_DIRECT LL +0.001639660/+0.001814021（正值较差），对UNIFORM增量很小，GEOM/SHUFFLE不能稳胜；已完成184导出/12计时。KR P2和P3/P4按gate NO_START。Q/outer原refitη固定μ分组导出缺失且原权重已清理。
- **Interpretation:** 当前版本VERSION_NOT_SUPPORTED；需求选择信号、责任非恒定、形成独有性和可用主方法是四个不同层级。原缺项降低外层机制解释完整度，但不改变已保存P1/P2门控或主要负结论。
- **Strength / Scope:** MEDIUM当前固定自然ML/KR开发、WEAK功能必要性；机制信息MISSING/MAJOR。
- **Source:** [v4 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md)、[PROCESS_REVISIONS](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/PROCESS_REVISIONS.md)。

**新方案不等于新证据：** [2026-10-26-v2 RCF-v1](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)是针对E055–E057提出的可证伪后继函数设计，状态DESIGN_ONLY。未实现、未训练、未评价；不能在Evidence Ledger登记正向效应量。

#### 2026-10-27—2026-11-02 跨分支新增直接证据 E058–E069

**证据层级：** 实测效应、机制诊断、解释及研究决定分别列出；未触发阶段和旧报告的重复副本不是反例。所有来源使用当时报告的**固定Git提交**；原文件仍在各分支，main只保留文字结论。

#### E058 — Relation-Basis v2在冻结强TARGET_ONLY对照前没有实用排序增量
- **Observation:** 48条正式适配（2自然数据×6方法×4seed），Q两个数据都选TARGET_ONLY；LOWRANK_RELATION−该C*在历史E的macro AUC：MovieLens−0.00062292（4/4负，95%CI不跨0），KuaiRand−0.00272675（4/4负）。ML LL−0.00071533（3/4改善但区间跨0）；KR LL+0.00115698（4/4恶化）。FULL_RELATION也未改变强对照结论。
- **Decision:** 当前关系基底/来源形成器停止扩展；不否认既有来源读取价值或其他多原型定义。
- **Strength / Scope:** STRONG ML排序方向、MEDIUM版本性停止；固定自然评估人口，4seed。
- **Source:** [10-27 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md)；EXP052。

#### E059 — BASP-A收益预测与动态读取有活动，但未提供独立实用策略收益
- **Observation:** 24正式轨迹，2自然集×2seed；CG−TARGET_MATCH历史E AUC：ML−0.00001955、KR+0.00054782，正向各1/2；Q上μ的留出skill不胜F域常量，soft群组近均匀，CG NULL读取权重约0.99+。部分FIXED_READ状态可检测到来源读依赖。
- **Decision:** 关闭该版本收益μ/关系群组的晋级主张；P2–P5为NOT_TRIGGERED，不是负性能报告。读取价值与完整形成器收益不等价。
- **Strength / Scope:** STRONG记录到的操作状态；MEDIUM条件停线、仅2seed。
- **Source:** [10-28 FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md)；EXP053。

#### E060 — 2025–26 MDMT开源模型强基线在两个自然数据集已有五seed结果
- **Observation:** 30正式训练（M3OE_FIXED1、HyperGate、AutoIFS，ML/KR各五seed）。对M3OE_FIXED1：HyperGate ML/KR Δmacro AUC+0.007379/+0.014412，AutoIFS +0.003551/+0.007308，四组都5/5正。KR AutoIFS Δmacro LL−0.004400（5/5改善）而Δnatural LL+0.006677（5/5恶化）；主要评价域d1占约98.21%，加权反号有算术解释。
- **Decision:** 两者是已完成的实测外部对照，不是仅搜到的名称；HyperGate部分参数UNVERIFIED_PROPOSAL，训练/选点不等于GA31协议、非官方严格复现。
- **Strength / Scope:** STRONG五seed同协议比对及指标算术；WEAK架构纯因果。
- **Source:** [MDMT基线FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md)；EXP054。

#### E061 — 无增强loss统一缩放可改变selected排序与早晚优化轨迹
- **Observation:** GA31-v2四臂、两集×5seed共40训练；ML c=.1−c=1无增强selected test macro AUC+0.003473（4/5），E10−0.004169、E50+0.026978；KR selected−0.006026（1/5）。耦合Adam将相对weight_decay和eps与c耦合。
- **Decision:** 确认ML条件性缩放排序效应与KR反向边界；不是“纯LR减小”或“动态样本权重本身提高AUC”的直接证据。
- **Strength / Scope:** STRONG ML阶段数值，MEDIUM KR版本性反例。
- **Source:** [GA31-v2](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md)；EXP055。

#### E062 — NR-Gauss增强独有增量未过门，ML普通L2与缩放参照接近
- **Observation:** GA31-v2目标域NR-Gauss增量：ML c=1 −0.000267、c=.1 −0.000188；KR c=1 +0.000188（2/5正）、c=.1 −0.004243。KR c=1最差格−0.008536（5/5损害）。GA31-R3在ML采用普通wd1e−4的L2控制，与c=.1缩放基线selected验证AUC差−0.000185、test AUC+0.000197（3/5），LL+0.001519（4/5损害）。
- **Decision:** 停止这版NR-Gauss投入；普通L2是简单强参照，但不宣称完整eps/历史状态代数完全相等。早期10-21 D套件采用不同协议，旧阳性结果仍成立。
- **Strength / Scope:** MEDIUM方法停线，STRONG实测因子对照。
- **Source:** [GA31-v2](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md)、[GA31-R3](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md)；EXP055–EXP056。

#### E063 — 反向配对降低同参数视图方差但不等于比同K IID2更好
- **Observation:** R3同θ反向配对降低条件方差。KR ANTI2−B-KR selected test AUC+0.001202（4/5），LL−0.001709（4/5），但ANTI2−G-IID2 AUC−0.001237；validation相对B仅2/5正、worst平均−0.003056（4/5损害）。ML ANTI2对工作基线−0.000687。
- **Decision:** 保留STRONG方差构造事实及KR局部条件收益；**当前反向配对没有独有selected AUC晋级价值**。
- **Strength / Scope:** STRONG诊断，MEDIUM当前方法选择。
- **Source:** [GA31-R3](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md)；EXP056。

#### E064 — 参数EMA具有明确selected AUC收益，ML LL可能付出代价
- **Observation:** 11-01动量实验中无增强shadow−online，ML selected test AUC+0.004399（5/5，全部30/30 cell×seed正），LL+0.007800（4/5恶化）；KR AUC+0.004616（4/5），LL−0.001225（3/5）。AIGM01调参后的ML AUC+0.003849（5/5，区间[+0.003384,+0.004314]），LL+0.001249（4/5恶化）；KR AUC同为+0.004616。
- **Decision:** **接受固定EMA为当前AUC强参考**；非反馈参数shadow未更改在线Adam，不能混称更新动量公式或LL全面改善；后续同seed重建不作为新人口。
- **Strength / Scope:** STRONG ML排序、MEDIUM KR排序；当前fixed1自然训练人口。
- **Source:** [EMA/SAD REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md)、[AIGM01](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md)；EXP057–EXP058。

#### E065 — SAD长差分记忆非零且真实改变更新，但未产生实用独有增量
- **Observation:** SAD βδ=.99 A3−普通增强A1 online selected test macro AUC ML−0.000336、KR−0.001554；KR目标域两任务相对A1均5/5下降。差分q/局部实际更新都不为0。βδ=.9 A2与普通Adam在实数代数等价，却可因长期FP32轨迹/选点分离出现差异。
- **Decision:** 停止当前长差分记忆公式独有收益主张，不误判为“差分没有计算”或“所有动量不可用”。
- **Strength / Scope:** STRONG代数/执行，MEDIUM当前版NO_PROMOTION。
- **Source:** [11-01 momentum](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md)；EXP057。

#### E066 — AIGM01再次支持EMA，但Gaussian收益有任务/域交换；创新门仅停于开发
- **Observation:** 56条唯一完整训练，ML Gaussian+EMA−无增强EMA macro AUC+0.000318（2/5），KR +0.001476（3/5），KR macro LL−0.001664（5/5）。KR最大域d1/task0,task1 test AUC分别−0.004654（5/5）/−0.004929（4/5），pooled/traffic均下降；AIG-M动态门q/λ确实非零，但开发两seed未过晋级门，未执行被拒候选五seed正式评价。
- **Decision:** EMA继续作强基线；**KR增强存在真实域任务损害、LL改善与macro增量弱支持并存**；AIG-M版内STOP_AT_DEV而非全预算失败。
- **Strength / Scope:** STRONG KR最大域损害，MEDIUM EMA结果，WEAK创新门独有性。
- **Source:** [AIGM01](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md)；EXP058。

#### E067 — GATS B2任务投影开发阳性未在确认与统一test形成强基线整体收益
- **Observation:** KR B2−A1开发验证mean AUC+0.008183500（2/2），B2−匹配R2+0.006231554；确认n=3对A1 +0.000640287（2/3），合并验证n=5 +0.003657572（4/5）。统一test n=5 B2−A1 macro AUC+0.000005156（3/5）、B2−更强A0固定EMA +0.001481447（仅2/5），pooled/traffic对A0均值负，worst/域任务有损害。
- **Decision:** 已完成实际确认并停止晋级，保留“开发局部有益”的经验事实，不能再用同种子重复包装成新算法稳定成功。
- **Strength / Scope:** STRONG数值/程序记录，MEDIUM停止决策。
- **Source:** [GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md)；EXP059。

#### E068 — 难度依赖增强、延迟时程与CV→rho当前规则均未过匹配门
- **Observation:** GATS B1在ML/KR不胜A1及已触发匹配R1；B3 delay5不胜固定/随机时程R3；C2以梯度范数CV调rho相对固定EMA或rho均值回放增量很小，未触发确认；EMA早期滞后、后期增益是已知阶段现象，不说明C2成功。
- **Decision:** STOP当前B1、B3、C2设计；没有独立搜索其他所有时程/任务增强规则，未触发阶段不登记为阴性。
- **Strength / Scope:** MEDIUM预注册版本停线，WEAK方法独有价值。
- **Source:** [GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md)；EXP059。

#### E069 — 教师约束与任务梯度输入视图有真实计算，但未胜固定EMA
- **Observation:** EMA02V2完成36/36正式轨迹、144/144端点，只有开发2seed。ML教师约束字段F1−A0+EMA test宏AUC+0.000358547（2/2）、验证均值−0.000424325，F1−F0 test +0.000969953；KR梯度任务双视图TG/AG−A0+EMA test AUC−0.017173896/−0.017534（两seed负），macro LL分别+0.012519/+0.012365。投影Jacobian秩2、梯度方向cos和扰动记录证明实际执行。
- **Decision:** 停止当前J/F/TG/AG视图版本；F1 ML只是弱条件线索、确认/W1未触发。梯度视图不同不等于真实动量递推被更改。
- **Strength / Scope:** MEDIUM KR版本伤害，WEAK ML F1边际收益；2seed。
- **Source:** [EMA02V2](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md)；EXP060。

#### E070 — EMA_ADAPTIVE A0训练面板方向持续性没有转化为验证动作增量
- **Observation:** 复用20条既有轨迹，0新增训练/Adam/Test。KR-G主面板Fast−Slow方向一致率0.96，验证侧0.60；C_persist的P_B与验证ΔAUC均为负，C_carry相对统一Fast的验证增量约为0，task静态政策退化为统一Fast。
- **Decision:** A0=NO_ACTIONABLE_SIGNAL；A1/B/C未触发。不能写成完整自适应模型Test失败。
- **Strength / Scope:** STRONG（本轮训练面板/动作诊断）；不增加独立seed。
- **Source:** [11/09 EMA_ADAPTIVE](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

#### E071 — PUB-MDMT当前P0/P1/P2原始尺度迁移未胜正常A/F参照
- **Observation:** 70条正式轨迹、五seed；P0/P1/P2相对A的Test Macro AUC均0/5正向，六格均值下降且LogLoss升高。C相对同α G控制有较大差值，但G自身E50宏AUC接近0.5。
- **Decision:** 当前coupled-WD迁移版本不晋级；G退化使C−G不能作为正常性能证据。WD=0后续矩阵未执行。
- **Strength / Scope:** STRONG（当前配置/协议）；不否定其他PUB参数化。
- **Source:** [11/09 PUB-MDMT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)。

#### E072 — 固定EMA收益主要由共享参数S承担，N删除效应接近零
- **Observation:** 四条件删除S的晚期验证AUC损失+0.0037572至+0.0043627；删除N为−0.0001632至+0.0001580。S内部EMBED删除损失+0.0018738至+0.0030311，STAR_SKIP次之，EXPERT接近0。
- **Decision:** 当前M3oE fixed1的EMA收益主要定位在共享参数，不能将其归因成专有参数普遍更需要EMA。
- **Strength / Scope:** STRONG（四条件、五seed机制归因）；不自动外推到其他模型。
- **Source:** [11/10 EMA causal attribution](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### E073 — recent与old存在正交互，历史预测组合重现部分固定面板排序收益
- **Observation:** 删除recent损失+0.0045636至+0.0049918，删除old损失+0.0007898至+0.0025160；EMBED recent×old交互四条件均正。q_logit/q_probability相对F在固定验证面板提高+0.0006661至+0.0026413 AUC并降低LL。
- **Decision:** 近期历史是主要贡献，较早历史存在组合价值；预测组合是机制解释之一，不是独立Test算法晋级。
- **Strength / Scope:** STRONG（固定面板/窗口归因）；q无独立选点Test。
- **Source:** [11/10 EMA causal attribution](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### E074 — Adam坐标预调节影响单步方向收益，不能把结果归结为纯动量
- **Observation:** current-preconditioned同状态同范数干预约+0.0003183至+0.0004076 AUC；momentum_isotropic约−0.0111386至−0.0127663，后者区间均低于0。
- **Decision:** 保留“当前Adam坐标预调节是局部正向作用的重要条件”；单步诊断不等于完整反馈算法。
- **Strength / Scope:** STRONG（当前方向/范数控制）；仅Clean条件触发。
- **Source:** [11/10 EMA causal attribution](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### E075 — H1/H2不是当前自适应EMA失败的主解释，H3仅条件性支持
- **Observation:** 机制/rho菜单存在标签侧oracle空间，但可见训练信号未产生可部署动作增量；rho7的static、C_design、C_simple多次退化为单一动作，四条件H2匹配未触发。Gaussian×EMA晚期交互区间跨0。
- **Decision:** 不再扩展同类预测器/动作规则网格；H3仅在当前已测动作空间下条件性支持，不宣称所有自适应空间耗尽。
- **Strength / Scope:** MEDIUM；训练可见信号和动作空间受本轮定义限制。
- **Source:** [11/10 EMA causal attribution](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### E076 — A_MATCH单步方向反馈在六轮训练中使F受损
- **Observation:** ML alpha=.5/1.0的F验证ΔAUC为−0.00064031/−0.00177519；KR为−0.00254234/−0.00085929。RAW、完整A续训与AB未触发。
- **Decision:** 停止当前A_MATCH反馈版本；不把未触发分支记为阴性，也不外推为全部方向反馈无效。
- **Strength / Scope:** STRONG（本轮强度、启动点和六轮窗口）。
- **Source:** [11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### E077 — 两态B的Test增量表现为数据集/指标条件性结果
- **Observation:** ML两态B−F Test ΔAUC +0.00003113/+0.00009666，区间跨0；LogLoss −0.00375756/−0.00271183，区间低于0。KR B_SHARED−F为−0.00013886/−0.00169040；KR Clean两态B_H277诊断+0.00069911、5/5正向，95%区间下界−0.00000416。
- **Decision:** ML主要保留LogLoss收益；KR H277只保留为中等强度候选。正式发表模型比较中ML通过三基线，KR B_SHARED未通过HyperGate/AutoIFS。
- **Strength / Scope:** STRONG（ML LogLoss和正式基线裁定）；MEDIUM（KR H277小幅诊断AUC）。
- **Source:** [11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### E078 — 当前cell动作策略无未来信号，KD未形成独立AUC晋级
- **Observation:** 四种cell系数规则均NO_FUTURE_SIGNAL；ML蒸馏学生相对父F的AUC +0.00015231但区间跨0，KR蒸馏未触发。
- **Decision:** 停止当前C规则；KD仅为部分教师收益压缩，不晋级为独立排序方法。
- **Strength / Scope:** MEDIUM；KD的ML结果不代表所有历史软标签方法失败。
- **Source:** [11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

**所有新增证据共同的边界：****所有新增证据共同的边界：** 这些测试多次使用同一历史划分、既有seed池及参考轨迹。五seed是训练随机性，不是新外部样本/用户/时间窗口。强简单对照未被超越时可以作具体版本STOP；不能把已证明的其他现象归零。

## 历史跨阶段总裁定

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)

#### 2026-11-03 新增最高优先级：正式基线只允许已发表论文模型；统一算法指标台账

> **研究者强制指令，后续一切设计、实验、代码与论文报告优先执行。** 本条与紧随其后的“五seed/AUC/LogLoss”新规则共同适用；覆盖下文历史报告中将本项目自研算法、无增强+固定EMA、当前最优候选等称为“基线”的惯例。历史文件中的原称谓保留为当时的研究术语，不表示今后获准作为正式论文baseline。

1. **正式基线资格为“已有正式发表论文的模型”。** 论文基线（baseline model）只能来自**在可核验会议论文集或期刊正式发表过论文的模型**，应登记模型名称、论文名称、venue、年份、DOI/正式出版页以及对应的实现来源。未核实正式发表状态的模型暂不得列入正式baseline表。例：M3oE（SIGIR 2024）、HyperGate（WWW 2025 Companion）、AutoIFS（WSDM 2026）均可作为已发表模型基线候选；其仓库适配版本、参数配置和与原论文的一致程度须明确报告，不可伪称原论文严格复现。
2. **本项目自行设计、改造、探索的所有算法、组件、组合与变体一律不是正式基线。** 包括但不限于Gaussian/MASK变体、SAD、ANTI2、AIG-M、PAIR/ECCM、M/H/C_MIX/T、各类动态权重/共享/原型算法以及本项目自己的A0+固定EMA训练/评估变体。它们只能命名为**候选方法、研究配置、内部参考、匹配控制或消融对照**；可以用于解释组件作用、增量及组合交互，但不能列入“正式发表模型baseline”名单，也不能因是当前最强内部配置而升级为正式基线。固定EMA若是对已发表架构额外加入的仓库内方法，正式报告应明确区分“论文模型原实现/适配”与“内部EMA后处理版本”。
3. **比较协议必须对应。** 与已发表模型进行正式论文性能对比时，须使用同一数据人口/划分、评价口径、合法checkpoint选择和可比较的训练协议，并明确各模型自身有据可查的训练设置；不可将不同分支协议的绝对AUC/LL跨表排序。对自研组件做内部消融时允许相互比较，但标题、列名和结论必须标为“内部消融/控制”，不能以这种比较冒充已发表模型的baseline实验。最新的五seed配对平均AUC≥0.003、逐域任务格平均跌幅不超过0.001、五seed平均LL下降的规则同样用于相应可比的正式评价。
4. **主目录性能总台账为长期维护入口：** [MDMT_ALGORITHM_METRICS_LEDGER.md](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/MDMT_ALGORITHM_METRICS_LEDGER.md)（对于 experiments/ 目录文件请使用相对链接 `../MDMT_ALGORITHM_METRICS_LEDGER.md`）。新增算法实验结束后，必须回填**每个域×任务格的绝对AUC与LogLoss、总体Macro AUC与Macro LogLoss，以及有据可查的traffic/pooled指标**；同时记录数据集、seed数量、模型/算法配置、在线/EMA状态、selected/common epoch、协议身份、来源报告和原始数表。不同协议分组存放；不得猜测缺失数字或将少于5seed的结果写为已充分否决。台账中的自研方法数据仅为**历史观测记录和内部方法分析**，不赋予其正式baseline资格。
5. **执行落地。** 后续Pro方案与Codex自动执行队列应直接继承本条及前述新五seed规则；在编写基线表、自动决策、生成论文比较表时强制检查“论文正式发表”标签。若历史任务中的旧停机门、旧基线用语或单格/seed否决门与本条冲突，以当前强制规则为准。原报告不得追溯性修改，新增研究任务也不得以旧报告为由重启已经充分确认的研究事实。

---


#### 2026-11-03 当前强制实验评价规则：五seed、AUC收益与域任务保护（优先于下方历史阶段门）

> **研究者最新明确指令；自2026-11-03起对后续所有新算法/新实验生效。** 本节优先于下文旧分支的单seed/双seed D0/D1 停止门、要求4/5或5/5 seed 同向的门、旧版域任务下降0.002保护线及“未触发后续阶段=算法性能失败”等措辞。以下规则用于**新的研究规划、Codex运行配置、审查、分析和算法取舍**。既有实验的原始报告、当时按冻结协议得到的 STOP/NO_GAIN 状态及数值**不回写、不篡改**；在新规则下应重新区分“旧协议停止”与“已完成五seed的性能否决”。

1. **任何算法不得以单一seed作为最小性能测试或否决依据。** 判断一个算法/固定候选配置在指定数据和训练协议下是否合格、是否因性能不足而停止，**必须至少完成5个不同随机seed的完整有效训练**，并与相同seed、可比训练/选点协议的基线配对。1–4 seed 的验证/测试结果只能记录为**初步信号或尚未充分评价**，不能据此宣告算法无效，也不能以其平均AUC不佳、正向seed不足、某一个cell退化、95%区间跨0等理由自动阻断五seed正式性能评价。允许少量CPU/单步/单seed工程smoke检查正确性，但它们不是科学性能淘汰门；明确的实现错误、不可运行、数值故障或资源阻塞只能分别记**工程阻塞/成本阻塞**，不能伪装成五seed阴性。不同epoch、六格、online/EMA和同一轨迹复用不计为新增seed。有限调参可按事先约定进行，但**不能用少于5seed的预筛选否决整个算法方向**；最终受评配置必须满足五seed。
2. **AUC按五seed配对平均收益判断，不强求每个seed为正。** 预先指定主AUC口径，当前默认**validation选点后的test六格等权macro AUC**；记 \(\Delta AUC_s=AUC_{method,s}-AUC_{baseline,s}\)，五seed均值 \(\overline{\Delta AUC}=\frac15\sum_{s=1}^{5}\Delta AUC_s\)。**\(\overline{\Delta AUC}\ge +0.003\) 即达到总体AUC收益标准（含恰好0.003）**，不另设4/5、5/5同向、显著性或置信区间必须完全为正的硬门槛。同时执行明确的**域任务保护约束**：六个域×任务cell各自的五seed配对平均 \(\overline{\Delta AUC}_{d,t}\ge-0.001\)，即**任一单格平均AUC最多下降0.001**；无需每个seed的该格均非负。两项都满足才标记“**AUC合格且满足逐格约束**”；宏平均达标但某格低于−0.001，则标记“**宏AUC达标、逐格约束未满足**”，保留实际收益和具体代价，不抹去观察、更不据此否定整个研究方向。validation、traffic、pooled、各cell、selected/common epoch及seed方向数仍必须报告，作为解释与后续改进依据，不擅自加额外一票否决门。
3. **LogLoss同样按五seed综合表现，不要求每个seed都下降。** 统一使用真实评价人口、同一选点协议下未经训练缩放的LogLoss，记 \(\overline{\Delta LL}=\frac15\sum_s(LL_{method,s}-LL_{baseline,s})\)。**\(\overline{\Delta LL}<0\) 即可认定“综合LogLoss下降”**，不以单seed恶化、只有部分seed改善或区间跨0自动否决LL改善。AUC与LL是**分别报告、分别裁定**的目标：以AUC为主的算法不能仅因LL未下降而被取消AUC合格资格，但须完整披露LL代价；若明确宣称“同时改善AUC与LogLoss”，则两项各自满足上述标准。
4. **比较对象、组合收益与研究状态必须清楚。** 每轮预先固定主比较基线，并采用严格匹配seed/数据人口/模型/训练预算/选点的配对差；对采用固定参数EMA的算法，需分别报告“完整算法＋EMA vs 普通训练”的**总收益**及“完整算法＋EMA vs 无新增组件的固定EMA”的**组件增量**，不得混为一谈，也不得因组件增量不够大就将已有整体收益写成“没有收益”。某项收益的“合格”只对清楚标记的比较对象成立。旧报告中D0/D1因1–2 seed或旧保护门而未触发的后续阶段，今后应标注为**不足五seed、按新规则尚不能性能否决**，不是自动断言方法成功；是否优先补做由当前研究价值排序决定，不为防御性确认无限延长实验。
5. **执行约束。** 后续Pro设计、Codex计划及自动runner应把上述规则直接写入实验矩阵、阶段门和最终裁定代码；如任何旧任务模板与此冲突，以本节为准，不得沿用单seed淘汰或强制全部seed正向规则。不得仅修改报告文字而让实际派发逻辑继续在1–2 seed时关停科学候选。

---


#### 2026-11-15 最新跨分支整合（11/09–11/11）

#### 11/09 EMA_ADAPTIVE A0：训练方向持续，但没有可执行动作增量

本轮复用11/07的20条轨迹（ML/KR × Clean/Gaussian D0 × seeds2022–2026），新增训练、Adam更新和Test访问均为0。KR Gaussian主范围GLOBAL的Fast−Slow训练面板方向一致率为0.96，但验证侧为0.60，晚期相关性接近0。C_persist的P_B未来ΔAUC为−0.000421092±0.000109264，验证侧为−0.000103584±0.000169881；C_carry的P_B为+0.000445201±0.000224756、验证侧为+0.000101319±0.000244342，但相对统一Fast动作的验证增量仅−0.000000478。TASK0/TASK1静态政策全部退化为统一Fast，task−uniform增量精确为0。

**裁定：A0=NO_ACTIONABLE_SIGNAL。** A1、闭环B和正式评价C均NOT_TRIGGERED。本轮是训练面板/验证侧预测诊断，不是完整自适应模型的训练或Test失败。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

#### 11/09 PUB-MDMT：完整五seed迁移与同α控制

**协议与完成量。** FULLTRAIN_NATURAL80_10_10，M3oE fixed1，Clean自然人口，ML/KR各seeds2022–2026；50epoch、batch40960、lr.01、coupled Adam。完成18组数值校准、40条2epoch频率比较、70条正式轨迹（3500epoch、120750次更新），A轨迹同步维护F。固定共同solver niter100/tol1e-4/ridge1e-8、α频率10。正式训练无失败、回退或重跑，参数与优化器状态留RAM。

- **STRONG：当前原始α尺度、coupled-WD的P0/P1/P2迁移均未达到主性能目标。** selected Test Macro AUC，ML A/F=.775190/.779039，C0/C1/C2=.557814/.602752/.637816；KR A/F=.671907/.676523，C0/C1/C2=.572765/.623796/.619789。所有C相对A均0/5 seed正向，六格均值全部下降、Macro LL全部升高。两数据Validation组合门均未触发，PUB+EMA轨迹0。
- **STRONG：更新空间相对真实同α梯度控制有独立轨迹增量。** C0/C1/C2−G0/G1/G2的Test Macro AUC，ML=+.030275/+.077394/+.118074，KR=+.049397/+.105126/+.098194，六个比较均5/5正向且六格均值全正。G在E50的宏AUC接近.5，这一控制自身的排序退化须进入解释；相对G的合格与相对A/F的未合格分别保留。
- **拓扑条件性。** C1−C0在ML/KR为+.044938/+.051031，均5/5正向。C2−C1为ML+.035064（4/5，按均值/逐格门合格、LL升高）、KR−.004007（1/5，LL下降）；局部拓扑没有跨数据集重复优势。
- **STRONG机械观察，MEDIUM因果解释。** 数据方向满足Gα=1/α，刷新时范数约√任务数。coupled-WD公共差分在首步显著改变实际效用；ML C1五seed首步30格实际效用均负、S探针BCE均上升。S晚期实际/同状态自然Adam候选范数比，C1/C2为ML.0394/.0310，KR.00206/.00186。原始半径、动态α下的正则/数据比例与性能损失仍待分离。
- **诊断与研究边界。** actual utility对负cell BCE排序的seed平均Spearman约.04–.17，排序对应证据WEAK。同步S探针ML50窗/seed、KR5窗/seed，统计单位为seed。域人数与采样固定；各格相对A均下降，未出现均值上的小域改善/大域下降补偿，不以规模构造困难度。

**当前决定。** 当前三种PUB迁移不晋级，保留固定EMA F为内部强对照。PUB作为优化方法候选记录。下一项有判别力的设计是A/C1/G1统一WD=0的两数据五seed配对，与本轮原WD条件比较，区分正则交互和原始更新半径解释；该新矩阵需单独设计执行。

[完整分析报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)；[全部逐seed端点/状态hash](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/ENDPOINTS.csv)；[配对均值/SD/95%CI](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/PAIRED.csv)；[机制诊断](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/artifacts/pub_mdmt_20261109/attempt1/tables/DIAGNOSTIC_SEEDS.csv)。六格AUC/LL与Macro/Traffic/Pooled已回填指标总台账。A/F的同seed身份保持，独立统计单位仍为五个seed。

#### 11/10 EMA_CAUSAL_ATTRIBUTION：固定EMA收益来源的参数与时间归因

4条件、5 seeds，共20条正式轨迹、1000 epoch和34,500次Adam更新。F−O selected Test Macro AUC为ML-C +0.003848845、KR-C +0.004616399、ML-G +0.003856802、KR-G +0.007766927；对应LogLoss差为+0.001248644、−0.001225342、+0.005334282、+0.000327687（正值表示F损失更高）。

| 删除项 | ML-C | KR-C | ML-G | KR-G |
|---|---:|---:|---:|---:|
| 共享参数S | +0.0040052 | +0.0043627 | +0.0037572 | +0.0042313 |
| 非共享参数N | −0.0001144 | −0.0001632 | −0.0001518 | +0.0001580 |
| EMBED | +0.0029602 | +0.0018738 | +0.0030311 | +0.0024302 |
| STAR_SKIP | +0.0006165 | +0.0014707 | +0.0005041 | +0.0016471 |
| EXPERT | −0.0000482 | +0.0003747 | −0.0000309 | −0.0002263 |
| recent | +0.0048642 | +0.0049918 | +0.0045636 | +0.0046705 |
| old | +0.0007898 | +0.0017536 | +0.0008662 | +0.0025160 |

删除共享S和recent会损失主要排序收益；N删除效应接近0。EMBED的recent×old交互在四条件下均为正，说明历史作用存在互补，不能简单将EMA收益拆成互相独立的参数份额。q_logit/q_probability预测组合在固定验证面板上相对F增加约+0.00067至+0.00264 AUC、同时降低LL，但没有独立选点完整Test，不构成新算法晋级。

同状态同范数current-preconditioned单步干预带来约+0.00032至+0.00041局部AUC变化；momentum_isotropic替代约−0.011至−0.013。前者保留Adam坐标预调节，不能归因于纯一阶动量。H1/H2不是当前主解释，H3得到条件性支持；Gaussian×EMA晚期交互区间跨0，不支持统一协同。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### 11/11 EMA_NEXT：反馈方向、两态历史读出与动作策略

A_MATCH短反馈在ML alpha=0.5/1.0下F验证AUC分别−0.00064031/−0.00177519；KR分别−0.00254234/−0.00085929。单步current-preconditioned的局部正值没有延续到六轮反馈，当前A版本停止。

MovieLens两态B相对F的Test ΔAUC为Clean +0.00003113、Gaussian +0.00009666，95%区间均跨0；LogLoss分别改善−0.00375756和−0.00271183，区间均低于0。KuaiRand冻结主候选B_SHARED相对F的Test ΔAUC为Clean −0.00013886、Gaussian −0.00169040；另一个两态B_H277诊断读出Clean Test +0.00069911、5/5正向，但95%区间下界−0.00000416，保留为候选而非晋级结果。

MovieLens两态B对三个发表模型均通过当前AUC门；KuaiRand B_SHARED只超过M3oE，未超过HyperGate和AutoIFS。因此不存在跨数据集均胜最强发表基线的统一算法。四种cell系数规则均NO_FUTURE_SIGNAL；ML蒸馏保留部分教师收益但没有足够独立AUC增量，KR蒸馏未触发。

[来源报告](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### 11/09–11/11跨研究裁定

- 固定EMA F继续作为强内部参照；其收益主要由共享参数S承担，近期与较早历史存在互补。
- 当前自适应预测器/动作规则没有形成稳定的Test排序增量；已测A_MATCH和C类规则按当前版本停止。
- 两态B的概率损失改善与排序增量必须分开。ML新增AUC近零、LL改善；KR两态诊断有小幅正值但区间下界接近0，值得一次有界新seed判别。
- PUB迁移、A_MATCH反馈、C动作策略的当前版本停止，不推导全部相关研究方向无效。
- 本轮没有合并任何来源分支的实现代码、原始报告文件或训练工件。

---

---

#### 2026-11-04 最新跨分支实证整合（优先于下方11/02历史裁定）

**本轮范围和合并截止。** 原11/02总裁定已整合EXP052–EXP060九组结果，本轮只新增11/02–03补充分支、11/03条件信息及真实历史实验、11/03五seed补全（11/04发布）、11/04 A+B统一发表模型及CP组合、11/04 Gaussian域覆盖及MovieLens既有来源核验。原始报告仍留源分支、本文只归档结论；性能总台账中11/03五seed数字原已存在，新增11/04发表模型/CP及KR G-D0/G-D02完整六格指标。不重复计算同轨迹的多个估计器、E10/E50或重命名别名。

| 实验来源与执行情况 | 关键实证（除特别说明外，为validation六格宏AUC选点后的test） | 当前结论与决策 |
| --- | --- | --- |
| [11/02–03 PAIR/ECCM及P01/P10补充](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md) | 原S1两数据各2seed、24条训练；新增8条训练补充λ=.05的P00/P01/P10/P11两seed完整2×2。KR P01−P00的test宏AUC **+0.012099**、validation **−0.000101**；KR交互P11−P10−P01+P00的test **−0.012347**（两seed均负），ML交互**−0.000384**且种子异向 | 此配置有KR测试条件性增益及负交互线索；**n=2不能五seed性能否决**，旧S2/S3 NOT_TRIGGERED不构成阴性。不能据此证明ECCM独立必要性 |
| [11/03 条件信息增强与真实历史更新](https://github.com/Caelestis1903/My-Research-Project/blob/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72/research_reports/2026-11-03_mdmt_history_information/REPORT.md) | 候选仅seed2022，新训13条、650epoch；KR C_MIX相对A0f的test宏AUC **+0.009531**、LogLoss **−0.005302**，但validation d0t0 **−0.006295**；H_MIX的test宏AUC **+0.012815**，traffic AUC **−0.000647**。ML全部候选主AUC未超过A0f，T0 test LL **−0.007020** | 原D0门停止、后续不触发是历史执行事实，不是五seed全否定。C/H/M后有完整五seed应以新裁定为准；T0/TM/H_REAL保持早期信号；KR TM为COST_BLOCKED |
| [11/03五seed补全，11/04发布](https://github.com/Caelestis1903/My-Research-Project/blob/652ee0235fec2d0d4cfc67b4a6ef9977519619bf/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md) | 两数据四候选40条逻辑轨迹（30新、10复用）。对普通M3oE online，ML C/H/P01固定EMA总宏AUC **+0.003253/+0.003059/+0.003225**、KR **+0.003720/+0.005579/+0.008702**，对应六格保护均满足。KR P01对无组件固定EMA AUC **+0.004086**但d1t0/t1 **−0.001334/−0.001047**。ML Gaussian＋EMA总AUC **+0.004167**、LL **+0.000931**；KR GATS B2历史总AUC **+0.006098**、六格保护通过 | **接受所列完整方案相对普通模型的五seed整体AUC收益**；不得把EMA贡献说成组件独有作用。KR P01对EMA宏收益达标但逐格保护未通过，ML C/H/P01相对EMA AUC均负；五seed原值已进main台账 |
| [11/04 Overnight A+B统一AUC选点和CP组合](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) | 30条A发表模型槽位、40条B组合槽位，合计30条新增训练。ML M3/HG/AutoIFS宏AUC **0.775190/0.771412/0.772425**，最强M3；KR **0.671907/0.682113/0.678597**，最强HG。CP fixedEMA ML/KR **0.777525/0.678168**，对M3总AUC **+0.002335/+0.006261**，对单独P01 **−0.000890/−0.002441** | 论文基线已统一自然人口、AUC合法选点。旧10/29 V按LL及人口/预处理不同不得跨表排名。CP对ML最强M3未达+.003、KR不胜HG，排序优先不替代P01；其对KR M3的总AUC及LL收益保留 |
| [11/04 Gaussian领域覆盖](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)／[ML源核验](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md) | KR G-D0/G-D02各新训5seed，G-D2沿用旧5seed，三者宏AUC **0.679180/0.674604/0.677999**、LL **0.435518/0.436172/0.434283**。D02−D0宏AUC **−0.004576**，d0t0 **−0.014949**。ML G-D0即旧Gaussian，宏AUC **0.779357**、相对M3 **+0.004167**且六格保护通过，ML复用无新训练 | KR宏AUC优先G-D0、宏LL优先G-D2，双域G-D02不优先。KR三臂均不胜最强HG，G-D0对M3总AUC **+0.007274**且六格通过。ML G-D0可按同协议胜最强发表M3，但LL平均上升 |

**跨轮收敛结论。** EMA是独立明确有效的时间平滑内部参照。旧一、两seed的停止门不能自动推翻后续完整五seed的整体排序收益；整体方案收益同样不能直接证明C、H、PAIR、ECCM单个新增部件各自有效。完整2×2已足以拒绝把CP作为当前排序优先组合。统一AUC选点后ML的正式发表模型排序与10/29旧协议相反，本轮以M3、HyperGate、AutoIFS三种合法发表模型为正式baseline；其自带训练设置/架构不同，不声称同超参训练或严格复现原论文。Gaussian改变增强域可改变六格与宏/traffic/LL取舍，直接增加覆盖并未获得排序加成。

**适用条件与后续决策。** 两自然数据均原行序80/10/10、train-only预处理，ML train 800167行、KR train 2006132行；固定M3与内部候选batch40960、Adam lr=.01、50epoch，EMA rho为ML .9828205985452511、KR .995。各模型按validation六格macro AUC严格最大、并列最早选点后以test为主；E10/E50仅作阶段诊断，online/EMA是同轨迹不同估计器。五seed总AUC配对均值≥+.003、逐cell均值≥−.001分别裁定，LL配对均值<0单独报告。KR test流量主要来自d1，宏/traffic会因权重不同方向分歧。**当前决策**是保留ML G-D0相对最强论文模型的已达标优势、KR的已有完整方案总收益但不声称超过HG；优先研究可改变算法选择的域任务收益分配与组件增量，不重启已完成论文基线统一或进行缺乏决策价值的重复确认。

**数值入口：** [MDMT算法性能总台账](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/MDMT_ALGORITHM_METRICS_LEDGER.md)。Overnight增量为正式M3/HG/AI与CP的六格及selected/E10/E50；KR G-D0/G-D02新数据与旧G-D2锚点同协议并列。ML G-D0与原Gaussian、KR G-D2与原Gaussian各为一组源轨迹的别名，不能累加独立seed或训练次数。

---

#### 2026-11-08 本周实验分支整合（11/02–11/07）

#### 分支覆盖与归档审计

**固定研究边界：小域不是困难域。** 不得把域大小当作困难度代理，不设计或解释任何实验来证明或否定这一判断。样本支持、正例稀疏和训练曝光如需研究，必须作为各自独立、可直接观测的变量处理，不映射成域难度。

| 实验分支 | 核心报告（已归档到main） | 整合状态 |
|---|---|---|
| 26-11-02 | [GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md) | 结论已在11/02历史裁定及SF035–SF036中；本次补齐主报告归档 |
| 26-11-02_v2 | [EMA02V2](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-02_ema_view_design_v2/REPORT.md) | 结论已在11/02历史裁定及SF037中；本次补齐主报告归档 |
| 26-11-02_v3 | [PAIR/ECCM与条件监督](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md) | 结论已在11/04整合中；本次补齐主报告归档 |
| 26-11-03_history_information | [条件信息与真实历史](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_history_information/REPORT.md) | 结论已在11/04整合中；本次补齐主报告归档 |
| research-plan/26-11-03-overnight-ab-auc | [Overnight A+B](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) | 统一发表模型AUC选点、CP组合及五seed比较已在11/04整合中；本次补齐主报告归档 |
| research/26-11-04-gaussian-domain-coverage | [Gaussian域覆盖](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md) | 已在11/04整合中；本次补齐主报告归档 |
| 26-11-06 | [AUG_HISTORY](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md) | 新增11/05实验结论与96条配置×数据集×端点台账记录 |
| 26-11-07 | [EMA_MECHANISM](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) | 新增11/06机制研究结论与8行selected Test绝对指标 |

11/03五seed补全报告已在main，不重复复制。research-plan/26-11-2-v3-conditional-information-momentum与research-plan/26-11-03-mdmt-history-information是方案分支，不作为独立实验结果；26-11-01系列在本次日期范围之外。按日期前缀检查未发现独立26-11-08实验分支，26-11-07为本周最后一个实验分支。本次只归档八份核心Markdown主报告及五份总上下文文件；没有合并实现代码、运行器、原始训练数据或大体量原始CSV/JSON，报告中的支持文件链接仍指向来源分支。

#### 11/05 AUG_HISTORY：当前自适应EMA候选没有超过固定EMA

本轮50次物理完整训练对应40条唯一配置×数据集×seed路径；B/A存在同父轨迹复用，C仅完成MovieLens，KuaiRand C为DEFERRED_NOT_RUN。

| 路径 | 关键结果（selected Test） | 裁定 |
|---|---|---|
| B：方向响应EMA | D−F Macro AUC：ML +0.000000104，KR −0.000004946 | 方向响应确实执行，但没有实用增量 |
| A：路径/分组时间尺度 | ML GROUP/GLOBAL：−0.000750/−0.000728；KR GROUP/GLOBAL：−0.000175/−0.000213 | 当前分组/路径策略不晋级；早期验证优势不等于selected Test收益 |
| C：梯度漂移协方差增强 | ML C_MERGED−F Macro AUC −0.000667190，ΔLL +0.000357811，0/5 seed AUC正向 | 当前版本不晋级；旋转控制未显示真实方向具有独立必要性 |
| 完整配置总收益 | ML C_MERGED+EMA相对普通M3oE：Macro AUC +0.003499743，最差cell +0.003065726，ΔLL +0.001289159 | 完整配置总排序收益保留；不能把它归因成C组件增量 |

**研究决策：** 停止当前B/A/C版本晋级，保留固定EMA F作为强内部控制。若继续方向机制，优先在匹配协方差谱/尺度的条件下区分真实方向与旋转方向；不继续扩展缺少判别力的分组候选。

#### 11/06 EMA_MECHANISM：固定EMA的收益超过年龄匹配延迟对照

| 数据/目标 | F−online selected Test Macro AUC | 正向seed | F−online selected Test Macro LogLoss |
|---|---:|---:|---:|
| MovieLens clean | +0.003848845 | 5/5 | +0.001248644 |
| MovieLens Gaussian | +0.003856802 | 5/5 | +0.005334282 |
| KuaiRand clean | +0.004616399 | 4/5 | −0.001225342 |
| KuaiRand Gaussian | +0.007766927 | 5/5 | +0.000327687 |

年龄匹配online对照在晚期epoch31–50的Validation中，F仍分别领先+0.002993、+0.003153、+0.004503、+0.003746，四项95%配对区间下界均大于0。因此收益不只是选了更早的历史checkpoint或固定延迟。早期F有明显学习滞后；晚期排序变化率下降。当前结论是历史组合价值与学习滞后成本并存，低波动本身不是收益充分条件。

逐格常驻时间尺度交叉评价只有KuaiRand Gaussian为正（+0.000375±0.000331，5/5）；其余三组均为负。C_design未来收益数值预测MAE在四组均优于M2/M3，但整体动作增益小。KuaiRand Gaussian d0t1共享组S的M3_direct局部净益+0.000346；相对C_simple的五seed增量+0.000048355±0.000199264，复杂指标独立必要性仍弱。Gaussian×EMA交互在selected Test为ML +0.000007957、KR +0.003150528，均只有3/5 seed正向，且Validation selected和Test E50方向不同，不支持统一协同规则。ML E40校准诊断解释了该锚点大部分LL差异，但selected Test的ML LL代价仍需如实保留。

**研究决策：** 固定EMA的排序收益及其阶段性代价获得强支持；组级时间尺度和复杂预测控制只保留窄范围研究价值，不晋级为普遍算法。后续若设计自适应EMA，核心问题应是找到可跨seed预测未来收益、且能实际改变参数递推的信号。

#### 本周整合后的统一裁定

- 11/02–04的结论沿用既有11/04整合，不重复计训练、不重复计seed、不把未触发分支改写成阴性。
- 11/05–06的新增结果已经回填性能台账；online/F为同轨迹读出，不增加独立seed。
- 固定EMA继续作为内部强参照；自研方法不能升级为正式发表模型baseline。
- 组件增量与完整算法总收益分开；AUC与LogLoss分开裁定。
- 当前优先问题从“固定EMA是否有效”收敛为“什么信号能预测并调节EMA历史组合的未来收益，同时控制阶段滞后”。

#### 2026-11-02 历史跨分支综合裁定（10-27—11-02；以11/04新整合为准）

**总状态：** main的系统性结论此前主要截至2026-10-26；此次补入9组不同已执行实验/基线（EXP052–EXP060），不把同源代码、重复训练、多个endpoint和未触发计划重复计数。报告和原始实验记录仍保留各自分支，本次仅将结论、证据与停止/开放状态纳入main。此前的历史实证与明确已接受的研究结论保留，不以新一轮防御性分析“重新归零”。

| 时间与实验 | 主要实证结果 | 当下研究决策 |
| --- | --- | --- |
| 10-27 Relation-Basis v2 | 两自然数据各4seed，48条正式适配；LOWRANK相对Q选择的TARGET_ONLY历史E宏AUC：ML−0.000623、KR−0.002727（均4/4负）；FULL也无强优势 | **停止当前来源关系基底版本**；保留局部来源读取的历史价值。[10-27 Relation-Basis](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md) |
| 10-28 BASP-A | 两自然数据各2seed、24条正式轨迹；CLUSTER_RELATION相对TARGET_MATCH的E宏AUC ML−0.000020、KR+0.000548（方向都1/2），留出μ预测不胜F常量、读取NULL权重约0.99+ | **当前原型关系/μ政策不晋级**，P2–P5未触发；不否定收益驱动原型概念。[10-28 BASP-A](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md) |
| 10-29 HyperGate/AutoIFS基线 | 两数据×3模型×5seed共30正式训练。HyperGate相对M3OE_FIXED1宏AUC：ML+0.007379、KR+0.014412；AutoIFS：+0.003551/+0.007308，四组都5/5正。KR AutoIFS宏LL−0.004400但natural LL+0.006677（5/5变差） | **两基线已经正式运行可用**；HyperGate配置有UNVERIFIED_PROPOSAL，V选择、训练协议不同，不称原论文严格复现。[10-29基线](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md) |
| 10-31 GA31-v2 | 40条训练；ML无增强c=.1对c=1 selected AUC+0.003473（4/5），E10−0.004169、E50+0.026978；KR缩放selected−0.006026；NR-Gauss在ML两个尺度分别−0.000267/−0.000188 | 确认**loss scale影响轨迹/选点**，NR-Gauss本版无独立晋级收益；耦合wd与eps不能忽略。[10-31 GA31-v2](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md) |
| 11-01 GA31-R3 | 45条训练；普通L2控制可作为ML缩放的简单排序参照。KR ANTI2相对普通基线test AUC+0.001202（4/5），但ANTI2−同K IID2均值−0.001237，validation及最差格不支持全门 | **反向配对减条件方差是真的，独有方法增量未成立**；不继续堆叠SAGU/SAOS。[11-01 GA31-R3](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md) |
| 11-01 参数EMA＋SAD | 两数据×4臂×5seed共40正式训练；ML无增强shadow−online selected AUC+0.004399（5/5），LL+0.007800（4/5恶化）；KR AUC+0.004616（4/5）。SAD βδ=.99相对普通增强Adam宏AUC ML−0.000336、KR−0.001554 | **固定EMA接受为排序有效的强简单参照**；当前差分长记忆版本停止，不将非零差分误当收益。[11-01动量/EMA](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md) |
| 11-01 AIGM01 | 56条唯一完整训练；调参后ML EMA selected AUC+0.003849（5/5）、KR+0.004616（4/5）；KR Gaussian+EMA相对无增强EMA宏LL−0.001664（5/5）、宏AUC+0.001476（3/5），但最大域d1两任务AUC−0.004654（5/5）/−0.004929（4/5） | EMA稳定参照；**增强表现为明确域/任务与指标取舍**。固定SAD/AIG-M开发门没过，不虚构正式五seedAIG阴性。[11-01 AIGM01](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md) |
| 11-02 GATS02 | 31条新增科学轨迹50epoch；KR B2开发验证对A1宏AUC+0.008184（2/2），保留seed确认缩小，五seed统一test对A1宏AUC+0.000005（3/5）；对更强A0固定EMA+0.001481（仅2/5）。B1、B3、当前C2规则均未晋级 | **B2开发局部信号保留，但不是可晋级整体算法**；当前B1/B3/C2停止。[11-02 GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md) |
| 11-02 EMA02V2 | 36/36正式轨迹，2开发seed，144/144端点；ML F1−A0+EMA test AUC+0.000359（2/2）、验证均值−0.000424；KR TG/AG−A0+EMA test宏AUC−0.017174/−0.017534，LL均约+0.0124~0.0125，两seed同向 | 当前J/F/TG/AG视图构造**不晋级**，F1只是微弱test线索，确认/W1未触发；保留无增强固定EMA。[11-02 EMA02V2](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md) |

**跨轮收敛的判断。** EMA的selected AUC收益不再是待“发现”的问题；用EMA影子参数本身不是对Adam动量公式的改造。缩放产生AUC效应、普通L2是强竞争者、方差下降不蕴含排名改善、非零差分历史不蕴含实用收益、复杂视图未胜简单EMA、某一任务收益可能转移成其他任务损失，这些都有各自已执行的实证支持。10-21 D套件历史结果与新GA31不同目标/人口，不互相取代。新研究应重点考虑**域×任务特征/标签需求如何影响增强与梯度，以及这些信息怎样进入真实动量/参数时间递推**，而不是重复先前阴性配置或再给EMA取新名字。

**证据边界与下一行动：** 用户已明确认可的研究前提无需专门复验。10-29基线使用不同训练/选模口径，不能与GA31 AUC选点绝对值直接排名；GATS的开发、确认和test必须分开；ML EMA排序改善与LL代价并存；任何声称新总体收益的方法须至少胜当前强EMA和必要匹配控制。历史test/seed池多次用于研究判断，这不是独立新增用户/时间人口。未触发阶段只记NOT_TRIGGERED。后续如果有具体新增设计改变研究选择，再安排最小有区分力的实验；不为绝对确定性无限追加。

**统一索引：** [实验EXP052–EXP060](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Experiment_Index.md)、[证据E058–E069](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)、[稳定事实SF029–SF037](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Stable_Findings.md)、[版本停线](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Dead_Ends.md)、[指标分歧](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Contradiction_Register.md)。

---

> **2026-10-26 总结更新：** 本文件的当前科研裁定以 [2026-10-26跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md) 为准；后文标为10-18/10-09的Q1–Q7、训练计数及“当前”陈述均保留为当时的历史快照，不能覆盖晚于它的证据。原研究阶段报告未修改，原始文件与固定SHA见 [history_experiments](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/README.md)。

#### 2026-10-26 当前研究结论（优先于下方历史快照）

1. **长期研究轴是样本重加权与差异化共享。** Equal/Sample是训练目标的局部口径；S5 Equal的特定Adam-history作用、v2/N17无动量SGD的采样效应并存：Adam不是全体效应的必要条件，也没有被Sample“完全消除”。研究仍要分开样本曝光、目标加权、动量状态和共享读写权限。
2. **先修正数据人口，再谈相互印证。** 10-15之后多轮MovieLens在人为固定子集187500行（34500/3000/150000）上训练；自然base-train为800167行（155949/318599/325619）。[自然人口审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)显示比例、真实支持与训练时钟一起变化。自然复核已经有部分可靠结果，但扩展分支整体仍为IN_PROGRESS，不能把N19/CM19人工结果宣称为全自然重复。
3. **N18/N19已经实质完成，不是只有“方案”或都没有尝试。** [N18](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)进行了六方向有界诊断；[N19](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)完成328次全预算训练。方法独有性和稳定强对照收益未获支持，但局部反例、容量/通道边界和方向机会保留，不把NO_GO扩展为领域失败。
4. **共享路径的数值功能与实际泛化收益不同。** [CM19](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)完成856次训练：KR fixed M3oE选点的输入乘性调制路径有明确数值中性，局部AdamW/耦合L2干预可以恢复响应；STAR等独立路径保持活跃，且恢复响应并未产生通用selected收益。
5. **10-21的D套件在自然MovieLens确有收益，但“动态样本区分导致收益”尚未成立。** [复现报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)中D−B test宏AUC +0.006303、LL −0.015962（4/4），Dconst01−B AUC +0.005676（4/4）；常数权重仍带复制/噪声/总尺度，不能从这组对照确定哪个模块有效。匹配G−Gtau0调幅的宏AUC：ML −0.003466、KR −0.006484（均4/4负，但KR区间跨零）。其目标为legacy_group_sum，非其他阶段的Sample/Equal。
6. **动态权限/收益/形成新方法须通过强简单对照。** 10-24 I1_PERM在ML/KR selected未胜固定RW；10-25 v2核心48条完成但正式门未触发；10-25 v3 RHD相对BASE selected test LL +0.00074075、e50 −0.00875428，LPS特定形成未稳定优于简单读法；10-26 DFD-v4的ML需求预测有限有利，但完整DFD相对H_DIRECT/ETA_DIRECT LL更差、KR P1失败，P3/P4未触发，另缺原refitη外层μ分组。各轮**版本性结论**不等于来源知识价值或多原型研究全盘不成立。
7. **最新研究决策是RCF设计而非新实验阳性。** [2026-10-26-v2方案](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)保留目标剩余风险指导来源后继函数形成的科学问题，以来源保持、有限容量、固定读出及强目标直接头作判别；目前**DESIGN_ONLY，无新增RCF训练/评价结果**。
8. **评价主张必须绑定协议。** 宏cell AUC、人口加权cell AUC、pooled AUC、LL，selected、末轮、长期AULC、Q/validation/test与真正新封存数据互不等价。早期“小域=困难域”从来不是用户认同的前提，不再把重复推翻它算作研究成果。

**合并结论：** 现有证据支持目标/域/群体条件性收益与共享损害的研究价值，支持一些真实局部机制与反例；没有建立一个在当前完整强对照和不同自然支持下通用、安全、独有的新算法。更值得检验的是**强目标预测之外的来源功能增量、群体需求驱动形成，以及样本权重收益的真实贡献结构**。完整时间线、已完成/未触发/部分完成和历史结论更正见[跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)。

---



> **2026-10-18 跨分支更新。** 下方原“整理截止 2026-10-09”以及 Q1–Q7/历史计数保持**历史快照**身份，不再代表最新所有实验的最终统计或执行状态。当前权威入口：[多分支综合报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)；[53条问题逐项去向](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)。证据编号 E033–E041、SF013–SF020 和新增 EXP 索引补录见相应文件。**未把代码/权重/原始数据从执行分支全量合并。**

#### 2026-10-18 历史科研摘要（优先于当时10-09快照；已由10-26总报告更新）

1. **同 Sample 目标不等于同训练路径。** KR Stage2D（learned fusion、双 Adam、四seed）validation-selected SB−SN macro AUC = −0.010406、4/4负；v2（fixed fusion、单优化器）无动量SGD仍为 −0.002104、4/4负。两组不构成协议同构的直接复现，却共同说明 Adam **不是所有 q 相关效应的通用必要条件**；不能从此反驳旧 S5 Equal 下持久 history carrier 的窄结论。
2. **最新 N17 的统计强度低于简单符号多数投票。** 120条新H100轨迹全部完成；正式 KR selected test：Adam SB−SN = −0.005203，8seed t95[−0.012006,+0.001600]、Holm p=.171875；SGD = +0.000837，t95[−0.004309,+0.005984]、Holm p=.6953125。两项**预定主比较均未确认**，不能称普遍损伤或等效。KR Adam AULC1..20 −0.007480、8/8负是**阶段性次级证据**，不是取代主门。
3. **晚期、选点、LR与评价权重会真实改变方向。** 固定 ML subset 的 own-LR E100 q-gap +0.021145、8/8正，但 selected −0.010875、8/8负；两臂采用不同 LR，不能称纯 q 优势，且cell保护门未过。同一个历史 calAdam E50 预测：宏cell AUC差 −0.009074、train-p加权 +0.001016、test-p加权 +0.003108（均不是 pooled AUC）。
4. **机制从“可能”缩窄为“局部反证与仍未识别”。** N2固定点 quota 计数项只占目标梯度trace约0.001297%–0.105936%，不支持“域计数波动主导”的特定解释；quota长期效应仍存在但交互区间跨0。N3局部共享路径约 +3e−5的阳性达不到独立方向/实用门，N4长期拓扑实验**未触发**；独立 shape vs trace、m/v/history、重复/coverage与共享 C 长期中介仍未唯一识别。
5. **研究主线不再围绕反复否定“小域=困难域”。** 这原本就不是成立前提。数据量、身份、标签/特征、域任务与曝光共同变化；未做真实独立数据支持 N 的纯因果操纵。当前结论是**条件效应有证据、普适/唯一中介不足、已测试候选方法 NO_GO**；不预设动量/课程有效，不从未触发分支推断方法阴性。

**跨时间纠正：** 10-16 早期 Stage2+ “正式训练 0、后续阻塞”已经被同分支 Stage2D **16/16 COMPLETE** 的后续报告覆盖。旧 S6-R2 KR test non-target +0.006757（四正）与10-15 B1重建 +0.000579（两正）来自不同init/lineage，不是同协议相互否定。v2是中期研究（不能声称全部完成）；v3在 Git 历史上与 v2 发散；N17继承v3。三个方案分支的原计划正文已经在 main，不额外计数实验。

**问题状态：** 旧 OQ001–OQ010 均逐项再评估（一些计数/身份问题已排除，一些机制依然开放）；N17 保留 Q1–Q46历史对象并新增Q47–Q53，完整53条见链接。若将“已执行/未触发”直接等同于“已解决/已否定”，会误报；也不限制新增开放问题的数量。

---

整理截止：2026-10-09。本总结把已复核的实验事实与未验证的解释分开；证据强度和适用范围以 [Evidence Ledger](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)、[Hypothesis History](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Hypothesis_History.md) 及 [Open Questions](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md) 为准。

#### Q1 — 目前最强的实验事实是什么？

1. **已测自然比例下，Sample joint training 没有显示稳定的实用负迁移。** S1 的 72 个已测配对单元中为 0 个 practical NT；S2 的 Sample 对照为 0/24。S6 预注册 selected-target gates 也未达到 practical rescue 阈值。这是对已测协议的结论，不覆盖未测的极端比例或目标域设置（E009、E010、E018、E019）。
2. **Equal/KuaiRand/tab1 下的 target 结果对 shared-access/history 干预有条件性响应。** S3 access 对照 4/4 seeds 同向；S5 显示该配置的 rescue 主要随完整 persistent Adam history source，而非单独 current source。此事实限于 M3OE、KuaiRand natural、Equal、d1/tab1 及相应 checkpoint；S5 的 moment gate 为 15/16，不能据此声称 m 或 v 单独致因（E011、E015、OQ003）。
3. **S6-R2 在 KuaiRand 观察到 test-side selected non-target gain，但 validation 方向相反。** R0-selected test ΔN 为 +0.0068（4/4 正）、ΔM 为 +0.0044（4/4 正），target ΔT 为 −0.0004；同一组 checkpoint 的 validation ΔN 是 −0.0039。R1/R3/R4 的同向结果只在 A-grid sensitivity 中成立，且 MovieLens negative control selected contrast 近零（E028–E030）。源报告在其 KuaiRand gate 上判为 GO；这不解决 validation/test 反转或增益机制。
4. **S8 的 MovieLens continuation 未提高冻结的原 F best。** Full-state selected-anchor 对照接近零，所有 MovieLens continuation best 均低于原 F best；late macro rescue 是 seed-skewed 的 overfit-anchor 现象，不能替代 S6-R2 的 KuaiRand test-side observation，也不能证明普遍 optimizer-policy gain（E026、E027）。

#### Q2 — 哪些早期结论后来被修正？

- “小域梯度必然更噪、需要 per-domain momentum”没有得到稳定支持；comp1 的独特动量机制被 S0 的匹配 uniform-EMA 对照削弱，domain size 也没有被独立识别为主因（E001、E008、E009）。
- “Sample 消除了所有 optimizer/history effect”范围过宽。S6 Phase A 只支持 selected practical target gate 未通过；后续 Phase B 与 S8 仍观察到 MovieLens late trajectory effects，S6-R2 又在 KuaiRand 发现 test-side selected non-target gain，但 validation 方向相反。KuaiRand Phase B 的 M1 gate 失败，不能写成机制识别已全面通过（E018–E020、E028–E030、C014、C019）。
- `exp_sn.md` 是 2026-10-06 的历史快照：其“Phase B 未触发”后来已过时；该汇总遗漏了独立的 S5-ML，而 S6ML 与 S5ML 不是同一实验。S6 样本量复核显示 MovieLens d2=150,000、d0=34,500、d1=3,000，故 d1 最小、d0 居中（C015、C016、C018）。
- S7-B 报告曾将 MovieLens anchor39 step5 target AUC 写成 4/4 seeds 为负。原始 CSV 实为均值 −0.001226、3/4 为负、seed2025 为 +0.002359；LogLoss 均值 +0.011130、3/4 变差。已修正源报告与对应 Evidence/Stable Finding；“短窗内未见稳定改善”仍成立，但逐 seed 一致性证据降低（E025、C012）。

#### Q3 — 哪些现象只在特定 dataset/objective/setting 下成立？

- Equal 下 KuaiRand d1/tab1 的 access/history rescue 是局部识别结果；不能直接外推到 Sample、其他域、其他 backbone 或 MovieLens。
- S5-ML 的 MovieLens Equal target/history pattern 只部分复现 KuaiRand S5；S6 Sample 下 selected gates 未达阈值，而 MovieLens late DH effects 出现在较晚训练阶段，受严重过拟合和 seed 影响。
- S7 的 Domain×Task/validation-gradient 观察只覆盖 MovieLens 与 KuaiRand 的 M3OE/Sample 设置。MovieLens late overfit 段的 U_k 与性能下降同步，KuaiRand 的 negative-U snapshot 没有同样后果；它不是跨数据集的充分性能标记。
- S6-R2 的 selected test-side gain 只见于所测 KuaiRand/Sample/M3OE F-vs-DH 对照；MovieLens selected effect 近零。KuaiRand 同一 selected checkpoint pair 的 validation/test 方向相反，且替代规则的 test 支持来自 A-grid sensitivity，并非精确重选 checkpoint。
- 旧 MovieLens ratio/role 结果描述的是特定数据生成、目标权重和训练协议，不能单独证明“大域必然受损”或“小域等于困难域”。

#### Q4 — 哪些机制目前真正有证据？

- 在 S3 的 Equal/KuaiRand/tab1 配置中，shared-access 干预对 target 结果有因果影响；这证明的是干预效应，不是 gradient-conflict 根因。
- 在 S5 的 Equal/KuaiRand/d1/tab1 配置中，完整 persistent Adam history source 承载了大部分 target rescue；状态子成分 m、v、step/bias-correction 与参数路径尚未区分。S6 Sample Phase B 的 KuaiRand M1 未通过，限制跨 objective 机制主张。
- S4 已显示 raw-gradient norm matching 和 applied-step norm matching 单独不足以复制该 target rescue。
- S7-B 能改变部分 gradient utility 指标，但短程内没有稳定 validation 收益；S8 的 MovieLens endpoint 也没有 selected/best 性能收益。S6-R2 的 KuaiRand parameter swaps 在四 seeds 均退化，支持所测分区不可直接互换，但没有定位 DH gain 的来源（E032）。因此，gradient correction 或某个 optimizer component 造成 test-side selected gain 仍未被证明。

#### Q5 — 哪些所谓机制其实仍然只是 hypothesis？

以下仍不是已建立机制：domain size 是退化主因；小域低 SNR 导致需要更长动量；Sample 下存在普遍 practical NT；negative U_k/gradient conflict 是充分性能原因；Equal history 机制跨 objective 保持不变；KuaiRand S6-R2 的 validation/test reversal 由某个单一机制造成；parameter swap collapse 已定位增益分区；MovieLens 与 KuaiRand 差异由某一个数据因素单独造成。KuaiRand test-side selected gain 是条件性 observation，不等同于普遍机制或部署收益。相关历史解释已在 Hypothesis History 中标为 WEAKENED、FALSIFIED、SUPERSEDED 或 OPEN，不能当作 Phase 3 前提。

#### Q6 — 哪些问题仍然 unresolved？

- Sample 在更极端域比例下是否出现 selected minority-domain gap；domain size 是否有独立因果效应（OQ001–OQ002）。
- Equal/KuaiRand rescue 中 Adam history 的具体状态来源，以及 shared-access/history 效应能否跨 backbone、target 和数据集复现（OQ003–OQ004）。
- MovieLens late DH 现象中普通 overfitting 与 optimizer-policy 各自贡献；Domain×Task gradient pattern 的跨设置稳定范围（OQ005–OQ006）。
- KuaiRand 的 train/test mixture 或 label shift 如何与 objective effect 区分；S5-ML 部分复现由数据结构还是训练阶段调节（OQ007–OQ008）。
- KuaiRand S6-R2 的 selected validation/test reversal 与替代 selection rules 在精确 checkpoint 上的结果；跨分区 swaps collapse 后能否定位 intact DH test gain（OQ009–OQ010）。

#### Q7 — 哪些旧研究线程不应在 Phase 3 中被重新当成有效前提？

不能预设小域更噪或更难、域大小已解释退化、M3oE 独有大域牺牲、comp1/域条件动量有独特优势、curriculum 或 adaptive calibration 已胜过 Sample、Sample 下存在普遍 practical NT、Equal 下 history 结果可推广到 Sample、target-only/HT 是安全部署方法、gradient conflict/U_k 修正会改善性能，或 DH 的 selected test gain 普遍存在/普遍不存在。S6-R2 的 KuaiRand 结果不能被当作某个机制或部署方法的既定前提；其 metric-layer reversal 和分区定位问题仍开放。

#### Traceability Audit

- `reports/exp_sn.md` 已先行核对；其中 S0–S6 在索引中各登记一次，未重写重复阶段总结。S5-ML 独立补登；S6 DH trajectory analyses 重用同一批训练数据，标为非独立证据。S6-R2 在该快照之后完成，作为 evaluation-only EXP032 单独纳入；它复用既有 S6/S6ML checkpoints，不新增训练。
- 抽查 S5 聚合表、S6 Phase B audit、S6-R2 gate-summary / window / parameter-swap CSV 与 integrity report、S7-B validation 原始 CSV、S8 paired endpoints；核对了 dataset/objective/seed/checkpoint 口径和关键数值。S6-R2 复用了 16 个既有 F/DH checkpoints、没有新训练；Part D 的 manifest partitions / state-dict parity 检查通过，seed2025 retention 分母近零已限定说明。保留 S6 KuaiRand M1 失败与 MovieLens d0/d1 样本量标签更正。
- 报告中的观察值与机制解释分开登记；单数据集、单 objective、固定 checkpoint 的结果没有推广为普遍规律；S6/S7 派生分析没有计为独立复现。
- 本整理未做 literature matching，也未提出或设计新算法。

#### Completion Counts and Phase 3 Readiness

| 项目 | 数量 / 判定 |
|---|---:|
| Experiment Index | 32 条：29 个既有实验组有正式运行/探测记录，2 个未运行/门控（E2、E7），另有 1 个 evaluation-only（S6-R2）；共 30 个条目有运行或评估结果 |
| Evidence Ledger | 32 条：STRONG 10、MEDIUM 19、WEAK 3 |
| Historical hypotheses | ACTIVE 2、PARTIALLY_SUPPORTED 3、WEAKENED 5、FALSIFIED 9、SUPERSEDED 1、OPEN 3 |
| Stable Findings | 48 |
| Open Questions | 36 |
| Dead Ends | 12 |
| Phase 3 条件 | **具备进入证据驱动的下一阶段梳理条件**；尚不构成任何特定算法或普遍机制已成立的依据。Phase 3 应以这些限定事实和未解决问题为输入。 |

> 计数说明：Stable Findings=48、Open Questions=36为2026-11-08更新后的当前文件条目数；Experiment Index/Evidence Ledger的32条仍是此前历史索引快照，本次未改动这两个文件。


#### 2026-11-23 · 条件历史与共享更新

A21 在 MovieLens 相对已发表 M3oE 的五种子 Test Macro AUC +.003284，但相对 clean+EMA −.000564，d0t1 低于保护线。KuaiRand 相对 clean+EMA +.004243、相对同 PAIR Adam +.004062，相对 HyperGate −.001347。两数据均未达到最终资格。较小 λ 未改善匹配对照，B 动态选源未进入完整训练。 [完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md)

## 历史假设及修订

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Hypothesis_History.md)

> **2026-11-15 最新修订：** 历史假说/反例原文和适用边界继续保留；新增H048–H053承接11/09–11/11的已完成研究。当前总体结论见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)、[Evidence Ledger](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)。不要将假说清单解释为研究者曾经认可的错误先验。

> **2026-10-26 假设更新：** H001–H032是截至10-18的研究假说/竞争解释清单，**不是用户曾经持有的信念清单**。新增H033–H040承接后期自然人口、梯度/共享、RHD/LPS/DFD和RCF设计。旧状态计数是历史截面，不重新汇总为“目前否定了多少科研方向”；以[跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)为当前裁定。



> **2026-10-18 补充与归因纠正：** H001–H023 是 10-09 的历史研究假设/竞争解释索引，**不是用户曾认可这些假设的清单**。尤其用户从研究早期便不把“小域=困难域”当作自己的结论；H002/H003 中域大小、低SNR等是**供数据挑战的候选解释**，不得反过来当成研究起点或把反复证伪它们冒充新创新。下方新增H024–H032，以及[跨分支综合结论](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)和[53问题去向](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)为当前覆盖口径。历史快照计数不自动代表更新后的条目数。


本文件把历史解释与当前允许的最强陈述分开。状态截至 2026-10-09；Hypothesis 状态只使用 `ACTIVE / PARTIALLY_SUPPORTED / WEAKENED / FALSIFIED / SUPERSEDED / OPEN`。

#### H001 — M3oE 独有的大域牺牲

- **Initial Hypothesis:** 大域退化是 M3oE 特有的共享结构问题。
- **Origin:** 早期 MovieLens/P5 中观察到的 L-role loss 与 comp1 gain。
- **Tests Performed:** P0-3 增加 MMoE；P0-1/2 轮换物理域与大小角色。
- **Supporting Evidence:** E003 中 M3oE 的早期结果。
- **Contradicting Evidence:** E003 中 MMoE 也出现同向 L-role 退化。
- **Evolution:** 初期 M3oE-specific → P0-3 后改为多骨干现象 → 后续 objective 研究又限制了它的通用表述。
- **Current Status:** SUPERSEDED。
- **Current Allowed Claim:** 在若干旧 MovieLens 配置中，较大的训练角色出现较低 joint-vs-reference transfer；该现象不只见于 M3oE。
- **Claims That Should No Longer Be Made:** “M3oE 固有地牺牲大域”或“Star topology 导致此现象”。

#### H002 — Domain size 是退化的独立主因

> **2026-10-18来源注记：** 这里是供检验的竞争解释，不等于用户最初认定“小域就是困难域”或认定小域更噪。后续不应以反复否定这一非前提代替真正的新问题探索。

- **Initial Hypothesis:** 大域/小域样本量差异直接造成 observed trade-off。
- **Origin:** L/M/S 角色结果及 early P0 系列。
- **Tests Performed:** P0-1/2 role rotation、P0-3 backbone、S2 objective contrasts、S4/S5 KuaiRand natural、S6 两个最大域 target。
- **Supporting Evidence:** E003、E010 有所测 Large-role 关联。
- **Contradicting Evidence:** E003/E010；S6 的 MovieLens d0 实为中域、d1 才是最小域；不同域/目标/阶段方向不一致，且无独立 size 操纵。
- **Evolution:** role 相关性 → 怀疑 size → 识别 objective、域身份、weight、收敛等混杂，独立因果未识别。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** Domain size 可以是调节变量或代理变量，但现有矩阵不能认定为主要解释。
- **Claims That Should No Longer Be Made:** “小域就是困难域”“大域必然受损”“size 已被证明是根因”。

#### H003 — 小域梯度 SNR 必然更低，因此需要按域动量

> **2026-10-18来源注记：** 这里是供检验的竞争解释，不等于用户最初认定“小域就是困难域”或认定小域更噪。后续不应以反复否定这一非前提代替真正的新问题探索。

- **Initial Hypothesis:** 小域样本少、噪声大，需更长 EMA 记忆。
- **Origin:** P1 的早期设计动机。
- **Tests Performed:** P1 D0.4 两个 batch size；P6 SNR-derived γ；S0 匹配平滑对照。
- **Supporting Evidence:** 某些梯度范数在少量设置下不同，但未形成稳定 SNR 规律。
- **Contradicting Evidence:** E001；P6 的 SNR-derived γ 未胜过公式控制，E008 未显示 per-domain momentum 独特收益。
- **Evolution:** 直觉性前提 → 实测不稳定 → 不再作为方法前提。
- **Current Status:** FALSIFIED（限已测 MovieLens 统计口径）。
- **Current Allowed Claim:** P1 的两种 batch 条件没有显示稳定的小域低 SNR。
- **Claims That Should No Longer Be Made:** “小域更噪已被证实”或“p_d 可直接作为梯度噪声补偿标尺”。

#### H004 — comp1 的收益来自独有的 domain-conditioned momentum

- **Initial Hypothesis:** comp1 的收益不能由普通平滑或 Adam 记忆解释。
- **Origin:** P5/P6 中 comp1 大幅优于 baseline 的结果。
- **Tests Performed:** P6 uniform γ、手工/SNR-derived γ；S0 validation-selected uniform EMA 与记忆控制。
- **Supporting Evidence:** E005 的早期 Sample+comp1 residual 与 P6 旧结果。
- **Contradicting Evidence:** E008：uniform EMA 在 S0 的旋转均值中逐 seed 优于 comp1。
- **Evolution:** 强机制主张 → 找到 residual → 匹配控制否定独特性，保留“旧 setting 下可复现 residual”而非机制因果。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** comp1 在某些旧目标/配置中有正向差异；它没有显示相对可选 uniform-smoothing family 的独立优势。
- **Claims That Should No Longer Be Made:** “SNR/域条件动量机制已被确立”或“uniform EMA 全面无效”。

#### H005 — comp1 收益随 imbalance ratio 单调扩大

- **Initial Hypothesis:** ratio 越极端，comp1 越有效。
- **Origin:** P5 高 ratio 结果。
- **Tests Performed:** P5 ratio sweep；P0-2 固定 N ratio sweep。
- **Supporting Evidence:** 若干高 ratio 设置差值较大。
- **Contradicting Evidence:** E002 中 P5 曲线非单调，r30/r100 又改变总 N；P0-2 只支持 ratio 部分贡献。
- **Evolution:** 对高 ratio 的描述性观察 → 固定 N 后仍有限关联 → 单调规律不成立。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** 在某些固定 N 条件下，ratio 与方法差异相关。
- **Claims That Should No Longer Be Made:** “comp1 效果随 ratio 单调增长”或“高 ratio 结果证明 size 机制”。

#### H006 — Equal/Sample objective mismatch 影响域级性能

- **Initial Hypothesis:** 等权六项目标在样本占比悬殊时可能改变 large-cell/target 表现；Sample 可能缓解。
- **Origin:** P0-4 的权重与选点对照。
- **Tests Performed:** P0-4/5、E1/E3/E4、S2–S6 objective contrasts。
- **Supporting Evidence:** E004、E010、E021。
- **Contradicting Evidence:** E4 的 train-test mixture/label shift、MovieLens checkpoint reversals、S6/S8 的 dataset/stage dependence；权重与 exposure 的单独贡献仍未完全拆分。
- **Evolution:** 大域牺牲解释 → objective-sensitive observation → 受数据、目标、阶段限制的解释。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** Equal 与 Sample 是不同有效目标；在所测 Equal cells 可见局部退化，Sample 在指定 selected 条件下减弱它。
- **Claims That Should No Longer Be Made:** “Sample universally optimal”“Equal universally wrong”或“系数变化是唯一中介”。

#### H007 — Sample objective 下仍普遍存在 practical residual negative transfer

- **Initial Hypothesis:** 即使 Sample 修正目标，shared MTL 仍普遍损害某些域，足以支撑 Safe-MTL 主线。
- **Origin:** 早期单元级负差值。
- **Tests Performed:** S1 多数据集、多骨干、joint-vs-SD；S2 Equal/Sample；S6 selected gates。
- **Supporting Evidence:** 零星小负单元；E009 未达到 practical 阈值。
- **Contradicting Evidence:** E009 的 0/72 practical NT；E018 的两个 Sample target gates 均 STOP。
- **Evolution:** 个别负 cell → 配对扩展未重复稳定实用损害 → 该普遍主张停止。
- **Current Status:** FALSIFIED（S1/S6 已测协议内）。
- **Current Allowed Claim:** 已测 Sample 自然比例与 reference 条件未显示普遍 practical NT。
- **Claims That Should No Longer Be Made:** “Sample 下已有稳定普遍负迁移问题”。

#### H008 — Curriculum / 时间 schedule 稳定优于静态 Sample

- **Initial Hypothesis:** 按域难度排序的训练课程可进一步改善 Sample。
- **Origin:** P1/P6 组件设计。
- **Tests Performed:** P1-2 forward/reverse、P6 component/sensitivity arms、E5 schedules。
- **Supporting Evidence:** 少数描述性小正值；未达到稳定门槛。
- **Contradicting Evidence:** E006；P1-2 差异在噪声级，E5 未通过 GO。
- **Evolution:** 被当成潜在核心组件 → 对照不稳定 → 从主线移除。
- **Current Status:** FALSIFIED（已测 curriculum/schedule）。
- **Current Allowed Claim:** 只可报告这些具体 schedule 的阴性结果。
- **Claims That Should No Longer Be Made:** “难度次序已验证”或“curriculum 已胜过 static Sample”。

#### H009 — Adaptive calibration/EMA 能稳定胜过 Sample 且安全

- **Initial Hypothesis:** 在线估计贡献并使用 EMA 可减少 objective mismatch，同时保持所有域安全。
- **Origin:** E6 adaptive method design。
- **Tests Performed:** E6 Adaptive-NoEMA/Adaptive-EMA 对 Sample/Equal；E7 preregistered GO gate。
- **Supporting Evidence:** E6 某些 secondary metrics 为正，Adaptive-EMA 与 Equal 的差异大。
- **Contradicting Evidence:** E006：对 Sample 的 overall gain 不过 margin；Adaptive-EMA 的 Medium-domain safety gate 失败；E7 0 runs。
- **Evolution:** 方法候选 → performance/safety gate 未通过 → 禁止当成可冻结方法推广。
- **Current Status:** FALSIFIED（限 E6 测试门槛）。
- **Current Allowed Claim:** E6 在该协议下 STOP。
- **Claims That Should No Longer Be Made:** “adaptive calibration 已成为优于 Sample 的主方法”。

#### H010 — Equal/KuaiRand/tab1 的 shared-access 导致局部 target loss

- **Initial Hypothesis:** FULL 跨域 shared updates 是 Equal 下 tab1 target 表现差的重要条件因素。
- **Origin:** S2 的 Equal target observation。
- **Tests Performed:** S3 objective×access intervention；S4 controls；S5 history/current decomposition。
- **Supporting Evidence:** E011；S3 的 target rescue 与 interaction 4/4 同向。
- **Contradicting Evidence:** Sample arm 的差异近零；MovieLens/late checkpoint 与 non-target tradeoffs 限制外推。
- **Evolution:** 观察性 Equal loss → 局部 access intervention 支持因果影响 → 机制进一步定位到 history。
- **Current Status:** ACTIVE。
- **Current Allowed Claim:** 在 M3OE/KuaiRand natural/Equal/tab1/指定 selected 对照下，shared-access 干预会因果改变 target 结果。
- **Claims That Should No Longer Be Made:** “跨域梯度几何冲突已被证明是根因”或“此效应适用于所有域/模型”。

#### H011 — 纯 gradient geometry/conflict 是 S3 rescue 的机制

- **Initial Hypothesis:** target-only rescue 来自跨域梯度方向冲突。
- **Origin:** S3 access contrast 的机制解释。
- **Tests Performed:** S4 norm controls、S5 history/current split、S7-A probes、S7-B same-θ continuation、S8 effective update analysis。
- **Supporting Evidence:** E011 access effect；E023/E024 显示晚期 gradient utility 结构。
- **Contradicting Evidence:** E013 幅度控制不解释；E015 history 承载 Equal d1 rescue；E025 gradient 变化未改善短程 validation；E026/E027 无 selected best gain。
- **Evolution:** access effect 被误同义为 geometry mechanism → 分拆 current/history 与 effective update 后，纯几何因果解释被削弱。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** gradient structure 与 late MovieLens 现象相关，但不是已确立的独立性能根因。
- **Claims That Should No Longer Be Made:** “gradient conflict 已被证明”“负 U_k 足以导致性能损害”。

#### H012 — 仅匹配 raw gradient norm 或 applied-step norm 足以解释 rescue

- **Initial Hypothesis:** T 的收益只是梯度或更新幅度变小。
- **Origin:** S3/S4 的机制候选。
- **Tests Performed:** S4 N/U arms。
- **Supporting Evidence:** T 对 F 有大差异。
- **Contradicting Evidence:** E013 两类幅度控制均未复制 rescue。
- **Evolution:** 幅度候选 → 匹配控制失败 → 作为充分解释被排除。
- **Current Status:** FALSIFIED。
- **Current Allowed Claim:** 这两类 norm matching 不足以复现指定 rescue。
- **Claims That Should No Longer Be Made:** “幅度匹配解释了 S4 target rescue”。

#### H013 — Equal KuaiRand d1 rescue 跟随 persistent Adam history

- **Initial Hypothesis:** S4 中的差异可能依赖当前梯度方向、history 或两者。
- **Origin:** S4 norm controls 失败后的机制分解问题。
- **Tests Performed:** S5 F/T/HT/DH，四种 source policy。
- **Supporting Evidence:** E015：HT≈T，DH≈F；target rescue 约 96–99% 由 history carrier 承载。
- **Contradicting Evidence:** 15/16 moment gate，MovieLens S5ML 的 DH 非零且晚期改变。
- **Evolution:** 联合组成/历史候选 → KuaiRand Equal d1 下定位为 history-dominant → 限定为完整 optimizer carrier 与该配置。
- **Current Status:** ACTIVE。
- **Current Allowed Claim:** 在 KuaiRand natural/M3OE/Equal/d1/tab1 的 S5 条件下，persistent history source 跟随 target rescue。
- **Claims That Should No Longer Be Made:** “m 单独是原因”“v 单独是原因”“history mechanism 跨 objective 不变”。

#### H014 — current-target source 对 rescue 的贡献在数据集间恒为零或恒为主导

- **Initial Hypothesis:** S5 KuaiRand 中 DH≈F，可能说明 current-target contribution 很小；也可能该模式普遍。
- **Origin:** S5 history×current factorial。
- **Tests Performed:** S5ML Equal MovieLens；S6 Sample 2×2；S7/S8 trajectory controls。
- **Supporting Evidence:** E015 的 KuaiRand Equal DH≈F；E020 的 MovieLens Sample late DH−F。
- **Contradicting Evidence:** E017 MovieLens Equal selected/final 中 DH 有非零成分；E019 selected Sample contrasts 小。
- **Evolution:** KuaiRand 单一结果 → S5ML 显示数据/阶段不同 → S6 又显示 objective 与训练阶段交互。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** current-source effect depends on dataset/objective/checkpoint; cannot assign a universal zero or universal positive effect。
- **Claims That Should No Longer Be Made:** “DH 一定等于 F”或“target current 一定带来 rescue”。

#### H015 — Equal 下 history rescue 在 Sample 下仍达到 practical selected magnitude

- **Initial Hypothesis:** S5/S5ML history-dominant rescue 可直接推广到 Sample。
- **Origin:** Equal objective 下两数据集的 F/T results。
- **Tests Performed:** S6 Phase A F/T；Phase B Sample F/T/HT/DH。
- **Supporting Evidence:** MovieLens common-late DH effects (E020)；但不满足 selected practical target gate。
- **Contradicting Evidence:** E018/E019 两数据集 selected target contrasts 均低于 +0.003；S8 selected continuation null。
- **Evolution:** 跨 objective 外推 → Phase A 双 STOP → Phase B 显示晚期 MovieLens residual 仍存在，但不是 selected target rescue。
- **Current Status:** FALSIFIED（针对已测 Sample selected practical rescue claim）。
- **Current Allowed Claim:** Sample 已测最大域 target selected gates 没有 practical rescue；这不是证明所有设置效应严格为零。
- **Claims That Should No Longer Be Made:** “Equal 机制已在 Sample 下同幅复现”或“Sample 完全没有任何 optimizer-policy trajectory effect”。

#### H016 — target-only / HT 是安全可部署方法

- **Initial Hypothesis:** 只更新 target shared gradients 可保留 target gain 并避免 non-target loss。
- **Origin:** S3 target rescue。
- **Tests Performed:** S3–S6 记录 target/non-target cells、macro/worst、AUC/LogLoss。
- **Supporting Evidence:** 部分 Equal target cells 有大幅提升；S5 HT 对 KuaiRand d0 selected cost 有平均缓解。
- **Contradicting Evidence:** E014/E016 non-target penalties与 stage reversals；S5ML d0 cost/d1 gain pattern；E019–E021 Sample under threshold；没有部署型比较。
- **Evolution:** 诊断 intervention 被误写成 candidate method → trade-off、objective dependence 明确 → 仅保留诊断身份。
- **Current Status:** FALSIFIED（作为“已验证安全方法”的主张）。
- **Current Allowed Claim:** target-only/HT 是用于识别更新来源的诊断臂。
- **Claims That Should No Longer Be Made:** “target-only/HT 已解决负迁移”或“没有 non-target cost”。

#### H017 — late DH/history effect 能提高最终会选择的模型性能

- **Initial Hypothesis:** S6 late-stage DH gain 可能代表 optimizer-history 方法改善整体性能。
- **Origin:** MovieLens common/final gain 与 S6 trajectory reanalysis。
- **Tests Performed:** S7-A/B、S8 full-state continuation，以及 S6-R2 对 KuaiRand / MovieLens 保存 checkpoint 的 selected-solution 评估。
- **Supporting Evidence:** E020/E022 的固定 late checkpoint observations；E028 显示 KuaiRand R0-selected checkpoints 的 test non-target 与 macro 优势。
- **Contradicting Evidence:** E019 的 Sample selected-target gates 未过；E026/E027 的 S8 selected-best 对照；E029 的 validation/test 方向反转；E030 的 MovieLens selected-checkpoint null。
- **Evolution:** MovieLens late fixed-epoch gain → S8 未提高 MovieLens 原始 selected best → S6-R2 在 KuaiRand 发现 test-side selected gain，但 validation 同期不占优且 MovieLens 未复现。
- **Current Status:** PARTIALLY_SUPPORTED。
- **Current Allowed Claim:** KuaiRand Sample/M3OE 的 R0-selected DH 相对 F 有 test non-target mean +0.0068（4/4）和 macro mean +0.0044；target mean 为 −0.0004，validation non-target mean 为 −0.0039。该结果限于特定 dataset、objective、selection rule 和 test layer；S8 的 MovieLens null 仍成立。
- **Claims That Should No Longer Be Made:** “late DH 在所有数据集都提高正常选点性能”或“late DH 在任何 setting 都没有 selected test-side gain”；test-side observation 也不证明 validation optimum 或 optimizer mechanism。

#### H018 — negative validation-gradient utility 是独立且充分的性能退化原因

- **Initial Hypothesis:** late negative U_k/harmful contribution 造成 MovieLens 性能下降，可作为梯度方法干预靶点。
- **Origin:** S7-A 的 late MovieLens alignment。
- **Tests Performed:** KuaiRand negative-U counterexample、S7-B same-θ current intervention、S8 effective-update continuation。
- **Supporting Evidence:** E024 的 MovieLens late 时序关联；E025 的同 θ 梯度层响应。
- **Contradicting Evidence:** KuaiRand negative U_k 但 AUC 继续升；S7-B utility 改善后 validation 未改善；S8 effective alignment 与 rescue seed 不对应。
- **Evolution:** 相关性 → same-θ 改变 gradient geometry → 未见 validation benefit，过拟合/Adam 轨迹为更强解释。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** U_k 是条件性局部一阶统计，在 MovieLens late overfit regime 与下降同步。
- **Claims That Should No Longer Be Made:** “negative U_k 必然造成性能下降”或“修复 U_k 就会改善 validation”。

#### H019 — Sample 在极端 ratio 下存在 selected minority-domain under-optimization

- **Initial Hypothesis:** 当前自然比例下无 practical residual NT，但更极端不均衡可能出现 minority-domain gap。
- **Origin:** S8 之后的研究问题陈述；不是已观测事实。
- **Tests Performed:** S1/S2 的已测自然设置；未见本问题的 extreme-ratio selected test。
- **Supporting Evidence:** 目前无直接 evidence。
- **Contradicting Evidence:** E009 的自然设置 Sample joint-vs-SD 无 practical NT。
- **Evolution:** 作为 open question 保留；不得把小域/困难域等同，也不得当作已有问题。
- **Current Status:** OPEN。
- **Current Allowed Claim:** 现有资料没有建立该 gap；问题在未测 ratio/target 设置中仍开放。
- **Claims That Should No Longer Be Made:** “Sample 已知存在 minority under-optimization”或“它必然存在”。

#### H020 — Domain×Task gradient structure 可普遍跨模型/数据集推广

- **Initial Hypothesis:** S7-A 的结构可能是普遍 MDMT 梯度规律。
- **Origin:** 两数据集 Domain×Task probe。
- **Tests Performed:** S7-A 两数据集、S7-B same-θ 对照；未有相同 probe 的跨 backbone 扩展。
- **Supporting Evidence:** E023 的两数据集 domain-organized cosine observation。
- **Contradicting Evidence:** E024 的 utility/performance 差异；KuaiRand 对照与 MovieLens 不同；P7 不是相同机制 probe。
- **Evolution:** 局部结构被观察到 → 可推广性未检验，性能机制解释被后续结果收窄。
- **Current Status:** OPEN。
- **Current Allowed Claim:** 所测模型与数据中可见 Domain×Task 结构；跨骨干、数据、域任务不确定。
- **Claims That Should No Longer Be Made:** “已证明普遍梯度机制”或“P7 组件迁移等于机制复制”。

#### H021 — KuaiRand DH selected-test gain 只是选择规则或单点尖峰造成

- **Initial Hypothesis:** S6-R2 的 KuaiRand R0 test non-target gain 可能只依赖 canonical selection rule，或由单个窄 validation peak 偶然产生。
- **Origin:** R0-selected test 对照中观察到 DH−F non-target AUC 提升。
- **Tests Performed:** S6-R2 Part A 的 R0–R4 validation selection 与 A-grid sensitivity；Part C 的 selected-window 分类。
- **Supporting Evidence:** E028/E031 中存在边界：R1/R3/R4 的 exact reselected test 指标不完整，A-grid 仅限六个保存点；seed 2025 对 alternative-rule gain 有反向响应；DH seed 2023 是 narrow spike。
- **Contradicting Evidence:** E028 的 R0 test ΔN 为 +0.0068、4/4 正；R1/R3/R4 A-grid ΔN 均值为正且各 3/4 正；E031 的 DH windows 3/4 为 moderate，而 F 为 3/4 narrow spike。
- **Evolution:** 单一 R0 观察引出 selection/peak artifact 解释 → 多规则 sensitivity 与 window 分类削弱“完全由单点/单规则造成” → exact alternative-rule test 缺失与 seed caveats 保留不确定性。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** 在已存 checkpoint 上，测试结果不只出现在 canonical R0；但 R1/R3/R4 的支持来自 A-grid sensitivity，不能称为 exact-rule replication，也不能推广到其他 dataset。
- **Claims That Should No Longer Be Made:** “R0 的 test gain 已证明对所有 selection rule、seed 或 dataset 稳健”或“已彻底排除 checkpoint timing”。

#### H022 — KuaiRand DH selected-test gain 可由单一参数分区定位

- **Initial Hypothesis:** 把 DH 与 F 的 selected parameters 按 shared/architecture/private 分区交换，可以定位 DH test-side gain 由哪一侧承载。
- **Origin:** S6-R2 Part D 的 parameter-swap analysis。
- **Tests Performed:** KuaiRand 四 seeds 的 manifest-based cross-partition hybrid inference；验证 state-dict key/shape parity 和锚点重现。
- **Supporting Evidence:** E028 确立了 KuaiRand 完整模型的 test-side DH−F 差异，这是提出分区定位假设的动机；该证据本身没有识别参数分区。
- **Contradicting Evidence:** E032 中 C/D cross-partition hybrids 的 test macro 在四 seeds 均降至约 0.52–0.58；E_hybrid≈C_hybrid，F_hybrid≈F_full，未能保留或定位 DH gain。
- **Evolution:** 预期由参数交换识别承载分区 → 两个方向的 cross-partition swap 均大幅退化 → 结果与分区间 co-adaptation 一致，无法分配完整 DH gain 的来源。
- **Current Status:** WEAKENED。
- **Current Allowed Claim:** 该 manifest partition 下的已训练组件不能直接互换；swap 结果不支持“某一侧单独承载增益”，也不能证明 co-adaptation 是 test gain 的原因。
- **Claims That Should No Longer Be Made:** “DH shared/architecture 或 private 参数分区已被独立识别为 selected-test gain 的来源”。

#### H023 — Checkpoint timing alone explains the KuaiRand DH selected-test gap

- **Initial Hypothesis:** DH 与 F 的 selected test 差异完全来自所选 epoch 不同，而非同一训练阶段的 policy difference。
- **Origin:** S6-R2 Part B decomposed selected validation differences into policy and timing terms。
- **Tests Performed:** S6-R2 val-layer anchor decomposition and same-epoch test trajectory on the saved checkpoint grid。
- **Supporting Evidence:** E029 shows large validation timing terms (TimingAlongF = −0.0120 for non-target AUC; 4/4 negative) and anchor-dependent policy terms。
- **Contradicting Evidence:** E029 shows same-epoch test non-target Δ(e) positive at 16/20 saved seed×epoch points, so timing alone is not established as the entire test-side explanation。
- **Evolution:** canonical selected epochs 不同使 timing 成为候选解释 → validation decomposition 显示对 timing 敏感 → 实际 selected epochs 的 test decomposition 不可得，test-side attribution 仍开放。
- **Current Status:** OPEN。
- **Current Allowed Claim:** Checkpoint timing contributes to validation-layer differences; whether it explains most of the KuaiRand test-side selected gain is unresolved。
- **Claims That Should No Longer Be Made:** “Timing fully explains the test gain” or “timing is irrelevant to the selected contrast”。

#### 假设状态计数

| 状态 | 数量 |
|---|---:|
| ACTIVE | 2 |
| PARTIALLY_SUPPORTED | 3 |
| WEAKENED | 5 |
| FALSIFIED | 9 |
| SUPERSEDED | 1 |
| OPEN | 3 |
| **合计** | **23** |


---

#### 新增假设演化 H024–H032（2026-10-15—2026-10-18）

#### H024 — 同 Sample 目标下改变q不会产生selected性能影响
- **Initial/competing claim:** 由于期望目标相同，Natural/Balanced训练效果应相同，或“Sample消除全部optimizer/sampling影响”。
- **Evidence:** Stage2D同目标 SB−SN validation −0.010406、4/4负；v2固定fusion也观察负向；N17主selected宽CI，不能断言普遍负/非零。
- **Current Status:** **FALSIFIED**（“相同期望目标必然同训练”的通用命题）。
- **Allowed Claim:** 相同期望目标下有限batch、状态与trajectory可变；效应大小/方向及统计确认视协议而异，N17 KR主检验未确认。E034–E039。

#### H025 — Adam是同目标采样效应发生的通用必要条件
- **Initial/competing claim:** 不采用Adam或不保留其历史状态，就不会出现q影响训练性能。
- **Evidence:** v2无动量SGD SB−SN −0.002104、4/4负；v3 SGD亦有条件性差异。
- **Current Status:** **FALSIFIED**（通用必要性）。
- **Allowed Claim:** 不排除Adam历史对某些特定Equal/S5或q×LR/状态局部效果有作用；m/v唯一中介尚未被证实。E015、E036–E038。

#### H026 — Balanced总是降噪并可普遍提高Sample宏AUC
- **Evidence:** 固定θ完整C/投影坐标trace比可反号；Stage2D Sample SB−SN selected −0.010406、nonharm失败；N17主宏未确认但无一致候选改善。
- **Current Status:** **FALSIFIED**（“总是/普遍”陈述及已测候选）。
- **Allowed Claim:** 对特定α/θ/块/q可缩小某个方差量；不等于全模型或选点性能收益。E034–E035、E039。

#### H027 — 同目标Balanced在任意optimizer、LR、阶段均普遍损害AUC
- **Evidence:** v2/v3部分配置负向；N17 KR SGD selected +0.000837（宽CI）；v3 calSGD共同早期可正；N17 ML own E100 +0.021145（同时改LR）而selected负向。
- **Current Status:** **FALSIFIED**（“所有/任何”普遍量词）。
- **Allowed Claim:** 在精确给定的某些KR配置/早期预算存在可重复负效应；不能将 own-LR 跨LR正值当纯q有益证明。E037、E039–E040。

#### H028 — 随机域计数波动主导SB与SN的固定θ梯度协方差差异
- **Evidence:** N17固定quota后SB/SN全部trace仍2.0–2.6左右；计数分量仅0.001297%–0.105936%。
- **Current Status:** **FALSIFIED**（所测16个固定θ锚的“主导”陈述）。
- **Allowed Claim:** 域内分布、批内高阶矩与依赖可能更重要；不能外推到所有theta、真实support N或唯一长期selected机制。E041。

#### H029 — Covariance shape/anisotropy 而不是scalar trace是唯一性能中介
- **Evidence:** 固定θ谱、块间trace方向与rank差已测；没有“同均值同trace、仅shape不同”的正式训练因果干预。
- **Current Status:** **OPEN**。
- **Allowed Claim:** 各向异性是具体可测的候选解释，不可声称优于trace已证实，更不能根据更多派生图表自动判定必要性。E034、E038、E041。

#### H030 — 共享C是Sample q效应的长期、唯一、可实用利用中介
- **Evidence:** v3局部C AUC屏幕阳性但loss/J联合门未过；N17局部independent full效果约3e−5且绝对实际界/方向门未过，N4未触发。
- **Current Status:** **OPEN**（长期必要性）；“局部完全无效”同样已被限定。
- **Allowed Claim:** 给定状态可能有小的C路径作用，尚非容量/预算匹配长期模型收益证据。E038、E042。

#### H031 — Checkpoint selection 或后期过拟合单独足以解释所有 q差
- **Evidence:** v2/v3/N17的共同早期固定轮/AULC已有负向（KR Adam N17 AULC1..20 −0.007480、8/8负）；v3同epoch仍可不同。ML own E100与selected反号。
- **Current Status:** **FALSIFIED**（“所有差异只因选点/晚期”的强版本）。
- **Allowed Claim:** 选点、预算、overfitting仍可显著修饰方向，不能说它们完全无关。E033、E037、E040。

#### H032 — 单一总体AUC符号对评价权重或部署人口不敏感
- **Evidence:** 历史calAdam E50相同预测 macro −0.009074、train-p +0.001016、test-p +0.003108；N17 ML own macro−0.010875、train-p +0.000744、test-p −0.013586。
- **Current Status:** **FALSIFIED**（该强版本）。
- **Allowed Claim:** 必须冻结主评价函数；cell加权AUC≠pooled AUC，两个任务和六cell不应视独立seed。E040。

**当前研究立场：** 不预设momentum、课程学习、domain size困难性，也不把新出现的若干识别缺口机械转化为无限追加的防御性实验。新假设要以既有MDMT方法的具体前提/迁移边界/未解决问题为对象，并提前规定可否定它的最小观测。


#### 2026-10-26 假设演化续记 H033–H040

#### H033 — 10-15之后MovieLens的“natural”就是未做人为域角色抽样
- **Tests:** 自然ML数据谱系、固定subset_seed2022、原始配置与base-train ID/行数审计。
- **Current Status:** **FALSIFIED（强人口等同命题）**。
- **Allowed Claim:** 旧187500人工subset与新800167自然训练都合法，但协议不同；真实比例、N和样本覆盖混杂。E048–E049。

#### H034 — D收益必须由持续区分样本的dynamic specificity引起
- **Tests:** 10-21 D权重下界探针，Dconst01固定0.1并保留复制/噪声的多seed对照。
- **Current Status:** **FALSIFIED AS NECESSARY FOR SUITE VERSUS B; OPEN FOR D'S INDEPENDENT MARGINAL GAIN**。
- **Allowed Claim:** ML D和Dconst01均胜B，但不能证明固定权重与D性能等效；权重/噪声/复制/总尺度仍需严格匹配。E050–E051。

#### H035 — 当前τ=0.5梯度范数调幅可稳定提高最佳宏AUC
- **Tests:** G与相同逐格求导、累加、zero tensor赋值的Gtau0。
- **Current Status:** **FALSIFIED IN TESTED CONFIGURATION**（ML差−0.003466，KR−0.006484，四seed方向均负，KR区间跨0）。
- **Allowed Claim:** KR早期训练可加速，且部分域或pooled指标方向不同；不否定所有梯度调幅机制。E052。

#### H036 — 所有MDMT共享/条件调制参数最终都会像KR fixed输入乘性路径一样失活
- **Tests:** CM19固定路径中和对照、STAR/PEPNet/HAMUR及MMoE/PLE跨架构实际干预。
- **Current Status:** **FALSIFIED（跨结构普遍性）**。
- **Allowed Claim:** KR fixed的输入乘性路径确有数值弱化/失活；只有特定路径、任务和实现范围内可主张。E046。

#### H037 — 修复数值调制失活本身足以稳定改善selected模型性能
- **Tests:** CM19双factor耦合正则/局部AdamW/旁路与从头重训，相对强对照检查selected AUC/LL。
- **Current Status:** **WEAKENED / NOT ESTABLISHED AS A METHOD**。
- **Allowed Claim:** 有效更新和路径响应能被改动，但selected收益不普遍；读取、训练后贡献、从头训练必要性需分别判断。E047。

#### H038 — RHD末轮少恶化/来源可读就证明当前动态控制或来源形成器优于简单方法
- **Tests:** 10-24 I1 selected、10-25 v2门控与v3 RHD/LPS、10-26 DFD强直接对照。
- **Current Status:** **FALSIFIED AS A GENERAL CLAIM ABOUT ALREADY-DEMONSTRATED FULL METHODS**。
- **Allowed Claim:** 特定RHD late-risk减损、foreign read和ML需求读益仍保留；各版本主指标及方法独有形成尚未成立。E053–E057。

#### H039 — 能预测群体共享需求，就已经证明需要训练多个来源函数
- **Tests:** DFD-v4 ML P1/完整ML P2的UNIFORM/GEOM/SHUFFLE及H_DIRECT/ETA_DIRECT对照，KR P1未过。
- **Current Status:** **WEAKENED; SOURCE FUNCTION FORMATION NECESSITY NOT ESTABLISHED**。
- **Allowed Claim:** 需求条件预测可与常量区别，函数形成增量必须在固定可达读取范围与强目标拟合下另证。E056–E057。

#### H040 — 目标群体剩余风险指导受约束来源后继函数可以增加独立可读知识
- **Origin:** 10-26 RCF-v1研究修订；保留多原型收益驱动局部共享问题并纠正DFD固定μ外推与同标签正权BCE形成的不足。
- **Current Status:** **OPEN / DESIGN_ONLY**（未训练）。
- **Allowed Claim:** 是有明确对照和预算的待验新假设；只有超过固定来源读、直接目标头、目标专有/统一微调、单宽后继等控制，才能谈新方法贡献。[RCF-v1](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)。

**历史记述与研究立场：** 研究者一直区分“小域”和“困难域”，并非提出后才被迫否定。既有方法相近不导致方向被机械丢弃；如果真实新增收益在强参照下仍不能建立，则明确版本NO_GO和能重新建立差异的最小设计更改。

#### 2026-11-02 新增已检验主张/候选解释状态 H041–H047

**说明：** H编号代表需被实证裁决的研究命题，不等于研究者曾相信该命题。历史假设状态维持其原日期快照；新版优先取证据的**对应协议**与当时强对照。

#### H041 — 当前低秩/全关系及BASP-A群组可直接提供强目标适配以外的稳定收益
- **Evidence:** E058–E059；Relation-Basis在自然E对强C*的AUC方向不利，BASP-A的μ/NULL和群组责任没达到晋级门。
- **Status:** WEAKENED_IN_TESTED_VERSIONS。
- **Allowed Claim:** 所测路线停线；来源有条件可读性、其他原型形成方式仍为不同命题。

#### H042 — 10-21 D套件正效应已证明源于动态权重，新GA31会自动复制
- **Evidence:** 早期Dconst01套件亦阳性（E050–E051），新GA31缩放/NR-Gauss使用不同目标、总尺度、人口和控制（E061–E062）。
- **Status:** FALSIFIED_AS_DIRECT_CAUSAL_ATTRIBUTION。
- **Allowed Claim:** 历史D与新缩放的各自实测现象保留，不把相似外形当同协议因果复现。

#### H043 — 降低增强方差或累计非零差分记忆足以保证selected泛化收益
- **Evidence:** E063 ANTI2降方差但不胜IID2；E065 SAD改变更新而未胜A1；E066 AIG动态门非零而没过开发增量门。
- **Status:** FALSIFIED_AS_A_SUFFICIENT_CRITERION_IN_TESTED_PROTOCOLS。
- **Allowed Claim:** 具体方差和状态诊断仍有效，必须用强基线的真实性能决定晋级。

#### H044 — 非反馈参数EMA在当前fixed1设置没有可靠selected排序价值
- **Evidence:** E064，ML两组五seed对照selected AUC+0.004399/+0.003849且均5/5正，KR+0.004616（4/5）；LL与AUC并不一致。
- **Status:** FALSIFIED_FOR_CURRENT_FIXED1_AUC。
- **Allowed Claim:** 采用EMA为强AUC基线；不是LL统一收益或新的Adam更新递推。

#### H045 — GATS B2两开发seed验证正差已经足以晋级
- **Evidence:** E067，确认验证收缩、统一test五seed对A1宏AUC近零且对强A0固定EMA仅2/5正。
- **Status:** FALSIFIED_AS_AN_ALGORITHM_PROMOTION_RULE。
- **Allowed Claim:** 开发期任务投影的条件性现象成立，但本轮整体NO_GO。

#### H046 — 显式使用两个任务梯度构造输入视图就必然是有益MDMT协同
- **Evidence:** E069，TG与AG真实生成视图且教师/任务方向诊断均正常，但KR selected test宏AUC和LL同时恶化。
- **Status:** WEAKENED_FOR_THIS_VIEW_FORMULA。
- **Allowed Claim:** 不否定所有任务条件增强；需要检查它是否真正改变有用更新和时间状态，而不是仅改变输入视图。

#### H047 — HyperGate和AutoIFS在本仓库尚未取得正式训练证据
- **Evidence:** E060，10-29已完成两个自然数据×三模型×五seed正式训练与冻结评价。
- **Status:** FALSIFIED_AS_EXECUTION_STATUS。
- **Allowed Claim:** 已有当前适配协议内的强模型实测对照；不可称原作者参数完全核实或GA31协议下严格排名。

#### H048 — PUB原始尺度P0/P1/P2迁移可在当前coupled-WD设置中带来独立AUC收益
- **Evidence:** E071；C0/C1/C2相对A的Test Macro AUC均0/5正向，六格均值下降且LL升高；同α G控制本身退化。
- **Status:** FALSIFIED_FOR_TESTED_P0_P1_P2.
- **Allowed Claim:** 当前原始尺度/coupled-WD版本不晋级；不能据此否定其他PUB参数化或未执行的WD=0版本。

#### H049 — 固定EMA的主要排序收益由非共享参数N承担
- **Evidence:** E072；删除共享S损失约0.00376–0.00436，删除N效应接近0。
- **Status:** FALSIFIED_FOR_CURRENT_FIXED1_CONDITIONS.
- **Allowed Claim:** 当前四条件下共享S是主要收益承载位置；不外推到其他模型结构。

#### H050 — 近期历史单独足以解释固定EMA全部收益，较早历史没有组合价值
- **Evidence:** E073；recent删除损失较大，old删除也有正损失，EMBED recent×old交互四条件均为正。
- **Status:** FALSIFIED_AS_A_COMPLETE_EXPLANATION.
- **Allowed Claim:** recent是主要来源；old单独效应较弱/条件性，但与recent组合有额外作用，不能简单剔除。

#### H051 — 同状态单步current-preconditioned方向收益可以转化为持续反馈训练收益
- **Evidence:** E074、E076；单步局部AUC小幅正向，六轮A_MATCH反馈却在ML/KR均使F下降。
- **Status:** FALSIFIED_FOR_TESTED_A_MATCH_FEEDBACK.
- **Allowed Claim:** 单步方向诊断和训练轨迹反馈是不同估计对象；停止当前反馈版本，不否定所有新递推设计。

#### H052 — 当前训练面板预测器和cell系数规则能够产生未来动作价值
- **Evidence:** E070、E075、E078；A0未形成可执行动作增量，11/10可见策略未超F，11/11四种cell规则NO_FUTURE_SIGNAL。
- **Status:** WEAKENED_IN_TESTED_SIGNAL_AND_ACTION_SPACES.
- **Allowed Claim:** 当前规则不再扩展同类网格；不推导所有训练期信号均无用。

#### H053 — 两态历史读出在两个数据集都能提供稳定的独立Test AUC增量
- **Evidence:** E077；ML两态B相对F的AUC增量近零、区间跨0；KR主候选B_SHARED未胜F，H277诊断读出有小幅正值但区间下界略低于0。
- **Status:** NOT_SUPPORTED_AS_A_CROSS_DATASET_CLAIM.
- **Allowed Claim:** 保留ML LogLoss收益和KR H277诊断候选，需新seed判别；不将相对发表模型的总收益归因成B的独立增量。

**总体演化：****总体演化：** 已建立的EMA排序正效应与旧共享/样本重加权局部事实不因新候选不晋级而被归零。新的研究问题是怎样让域×任务信息在增强/梯度之外进入真实更新或动量递推，以及怎样相对强EMA得到独立净收益；不重跑被当前实验已经裁定的相同假说。

## 历史问题及决策边界

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md)

> **2026-11-15 最新结论：** 历史问题保留原始提出范围；已按11/09–11/11证据更新OQ029、OQ030、OQ033、OQ036，并新增OQ037。最新研究决策见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)。不把未触发阶段当作阴性，也不重复同seed验证。

> **2026-10-26问题更新：** OQ001–OQ020继续保留为各自提出日期的问句，不以“没有触发实验”自动关闭。新增OQ021–OQ028重点从“反复做一个老解释的防御性否定”转向**辨别真实样本收益、来源可用增量、读写不同利益与局部功能必要性**；当前研究决策见[跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)。最新RCF仍为设计，所有问题没有预设阳性结论。



> **更新于 2026-10-18：** OQ001–OQ010 保留 10-09 原始提问，不因新实验“未触发”而直接关闭。下表是最新去向；新增 OQ011–OQ020 仅登记真正需要新的可区分反事实/证据的研究问题。详细保留 Q1–Q53 的**历史状态与最终限度**：[跨分支问题JSON](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)。多轮证据与反例：[综合报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)。

| 旧问题 | 10-18 状态 | 已缩窄/仍缺的证据 |
| --- | --- | --- |
| OQ001 Sample极端自然比例的minority selected gap | OPEN | 固定mask、别名复制不等于独立新真实样本；未得极端自然比例可推广结果 |
| OQ002 domain size独立因果 | CLOSED_BY_RESEARCH_BOUNDARY | **小域不是困难域**是固定研究边界；域大小不得作为困难度代理，不设计相关验证/否定实验。历史N2结果仅保留原始记录，不作为该边界的证据。 |
| OQ003 Equal/S5 history m/v具体承载 | OPEN / HISTORICAL_LINEAGE_MISSING | 新Sample q影子状态不是原Equal S5同θ谱系 |
| OQ004 Equal access/history跨backbone与target | OPEN | 后续多数是同目标q干预，非原access对照的普适复现 |
| OQ005 ML late DH的过拟合与optimizer-policy归因 | BOUNDARY / MEDIATION_OPEN | S8 late rescue不提高原best；N17阶段反例不能识别原DH中介 |
| OQ006 Domain×Task gradient结构跨设置 | PARTIAL / OPEN | 固定θ谱及block有结果，缺跨设置性能因果机制 |
| OQ007 split/label/population shift与objective | ARITHMETIC_RESOLVED / CAUSAL_OPEN | 同预测宏/加权AUC反号可分解，域内shift与目标差的因果不可分 |
| OQ008 S5-ML partial replication的调节来源 | OPEN | ML/KR同时改变任务/预算/支持/模型人口 |
| OQ009 KR S6-R2 selected validation/test反转 | UNRESOLVED / B1_NONEXACT | B1新谱系没有稳定复制，但旧精确状态缺失，不能判旧观察错误 |
| OQ010 S6-R2 swaps collapse与共享分区来源 | OPEN | 后来的局部C不是原intact DH历史参数分量因果 |


本清单只保留已有实验没有回答的问题。每条说明需要什么证据类型；不提出具体算法或下一阶段实验方案。

#### OQ001 — Sample 在更极端域比例下是否存在 selected minority-domain performance gap？

- **Scientific Question:** 在已测 Sample 自然比例未见 practical NT 后，更极端 ratio 是否仍保持 selected joint-vs-single/oracle 表现，尤其是少数域单元？
- **Why Unresolved:** S1/S2/S6 覆盖的是指定自然域结构；目前未有该问题所需的 extreme-ratio selected comparison。E009 不能证明未测 ratio 结果。
- **Relevant Evidence IDs:** E009、E018、E019、E027。
- **Competing Explanations:** 可能不存在可观测 gap；可能随 ratio、域身份、目标权重、sample diversity、checkpoint 或训练 exposure 改变。
- **Evidence That Would Resolve It:** 在明确对齐的 selected checkpoint、per-domain/task AUC 与 LogLoss、同 seed joint/reference comparison 下，重复观察不同 imbalance ratios 的 gap 大小与方向。

#### OQ002 — 域大小与困难度的关系不作为研究假设

- **Status:** CLOSED_BY_RESEARCH_BOUNDARY。
- **固定边界：** 小域不是困难域。不得把域大小、样本量或域人口标签当作困难度代理；不得设计、执行或解释实验来证明或否定这一判断。
- **可区分的其他变量：** 若某个算法问题确实需要，可单独研究正例稀疏、训练曝光、标签支持或数据覆盖等直接可观测量；必须明确其独立定义和作用路径，不将其重命名为“域难度”，也不将结果用于挑战上述边界。
- **处理规则：** 本条不再列入开放实验清单；历史OQ002相关实验只保留其原始实验事实，不作为该固定边界的证据。

#### OQ003 — Equal KuaiRand d1 的 persistent-history effect 具体由哪些状态部分承载？

- **Scientific Question:** S5 识别到的整体 Adam history carrier 中，m、v、step/bias-correction 与 history-induced parameter path 各自贡献多少？
- **Why Unresolved:** S5 只操纵完整 history source；Moment gate 15/16；S6 Sample Phase B 的 KuaiRand M1 gate 未通过。
- **Relevant Evidence IDs:** E015、E016、E019、E021。
- **Competing Explanations:** signed m history、v preconditioning、effective step count、weight decay at arm-specific θ、trajectory divergence。
- **Evidence That Would Resolve It:** 经过独立 validity review 的状态分量干预证据，并保留同一配置下的 realized states、参数路径和完整指标；不能从当前 H×C contrast 直接断言状态子成分。

#### OQ004 — Equal 下的 access/history effect 能否跨 backbone、目标域和数据集复现？

- **Scientific Question:** S3/S5 的窄条件效应是否会在不同 backbone、非最大 target、小域 target 或其他 domain structure 上保持方向与代价？
- **Why Unresolved:** S3–S6 主机制矩阵以 M3OE 为主；P7 是组件迁移而不是相同 mechanism test；S5ML 只部分复制 Pattern A。
- **Relevant Evidence IDs:** E011、E017、E021、E023、E027。
- **Competing Explanations:** objective × dataset × target × task interaction、imbalance severity、architecture sharing structure。
- **Evidence That Would Resolve It:** 使用同一 intervention definitions、seed pairing、checkpoint/metric protocol 的独立复制，同时对目标与非目标结果进行同等检查。

#### OQ005 — MovieLens late DH gain 中普通 overfitting 与 optimizer-policy 贡献各占多少？

- **Scientific Question:** full-state continuation 中看到的 late macro rescue 有多少是 F reference 过拟合导致的“少恶化”，有多少是可泛化的 optimizer-policy effect？
- **Why Unresolved:** S8 显示 selected-anchor null、late macro rescue seed-skewed，且不超过原 F best；它没有将所有 overfit/regularization 路径分开。
- **Relevant Evidence IDs:** E020、E022、E026、E027。
- **Competing Explanations:** ordinary overtraining、implicit step-size regularization、checkpoint timing、non-target contribution、seed variance。
- **Evidence That Would Resolve It:** 在预先明确正常 selected 与 overfit anchors 的比较中，完整状态和相同 batch continuation 下，对照 late/selected best、target/non-target、AUC/LogLoss 和多个 seeds 的共同轨迹。

#### OQ006 — Domain×Task gradient structures 的跨设置稳定范围

- **Scientific Question:** Domain-organized cosine pattern、harmful source-target pairs 与 U_k 条件性结果是否跨模型、数据集、域分布和训练阶段存在？
- **Why Unresolved:** S7 只探测两个数据集与一个 M3OE 配置；KuaiRand 的 negative-U counterexample 和 MovieLens overfit pattern 显示效应依赖阶段。
- **Relevant Evidence IDs:** E023–E025。
- **Competing Explanations:** Domain structure、task interaction、norm scale、Adam preconditioning、overfitting、module reachability。
- **Evidence That Would Resolve It:** 在多个独立 backbone/dataset 配置中重复相同的 Domain×Task probe、报告绝对 norm 与方向量，并对性能观测做独立验证。

#### OQ007 — E4 的 train-test mixture / label shift 与 objective effect 如何区分？

- **Scientific Question:** KuaiRand 的训练与测试域占比、label prior 变化能解释多少 Equal/Sample 下的 residual differences？
- **Why Unresolved:** E4 审计测得 mixture TV 与 d1 click prior 的变化，但实验未独立操纵该 shift 与 loss objective。
- **Relevant Evidence IDs:** E004、E021。
- **Competing Explanations:** objective coefficient、sampling/exposure、temporal split distribution shift、domain/task difficulty。
- **Evidence That Would Resolve It:** 同一 objective 对照下可追踪的 split/mixture variation，且训练与测试统计同表报告。

#### OQ008 — S5-ML 的 history-dominant partial replication 是数据结构还是训练阶段造成？

- **Scientific Question:** S5-ML 中 DH 的 selected/late contribution 与 T rescue 衰减，主要由温和域比例、target/task 结构还是 late overfit 引起？
- **Why Unresolved:** KuaiRand 与 MovieLens 同时改变数据集、域比例、domain identity 与收敛轨迹。
- **Relevant Evidence IDs:** E015–E017、E020。
- **Competing Explanations:** imbalance severity、target identity、task difficulty、checkpoint timing、overfit、Adam history/current interaction。
- **Evidence That Would Resolve It:** 在独立变化这些因素的受控复制中，保留 selected/common/final 与 per-cell 视图；不从两个自然数据集差异反推原因。

#### OQ009 — KuaiRand S6-R2 selected-checkpoint 的 validation/test 反转及其跨数据集范围

- **Scientific Question:** 为什么同一组 KuaiRand R0-selected checkpoints 的 DH−F non-target AUC 在 validation 上为负、在 test 上为正；该差异是否能在精确的替代 selection rules 和其他 dataset 上复现？
- **Why Unresolved:** E029 显示 validation ΔN = −0.0039、test ΔN = +0.0068；canonical selected epochs 不在完整 test checkpoint grid，test-layer cross-epoch decomposition 不可得。R1/R3/R4 的正向对照来自 A-grid sensitivity；精确重选 test 仅 R4 有 1/4 seed。MovieLens 的 selected test result 近零，KuaiRand seed2025 也近零。
- **Relevant Evidence IDs:** E028–E030。
- **Competing Explanations:** validation/test metric-layer difference、checkpoint timing、selection-rule sensitivity、dataset/target difference、seed variation。
- **Evidence That Would Resolve It:** 在精确 selected checkpoints 上同时具备 paired validation/test per-cell 指标与同 epoch trajectory 视图，并在多个 seeds 和数据设置下核对方向；现有证据不足以从指标反转指定单一机制。

#### OQ010 — S6-R2 parameter-swap collapse 能否定位 DH selected-test gain 的来源？

- **Scientific Question:** 在 shared/architecture/private manifest partition 下，DH 的 test-side selected gain 是否能被分配到某一参数分区，还是只能观察到跨分区不可组合？
- **Why Unresolved:** E032 显示 C/D hybrids 在四 seeds 均明显退化，不能定位 intact DH gain；seed2025 的 ΔN 分母近零，retention ratio 不可解释。现有检查只覆盖 KuaiRand、M3OE 和一种 parameter partition。
- **Relevant Evidence IDs:** E032。
- **Competing Explanations:** 跨分区 co-adaptation、partition boundary 定义、单 seed near-zero denominator；这些解释目前不能区分。
- **Evidence That Would Resolve It:** 对候选参数/功能分区提供独立完整性验证，并同时保留 intact parent 与 hybrid 的 seed-level validation/test outputs；不从当前失败 swaps 断言增益由某一分区产生。


---

#### 2026-10-18 新增/进一步明确的待辨别问题

#### OQ011 — Adam同θ瞬时平均应用更新差是否真实存在，且历史是否必要？

- **Current evidence:** 新 N17 Q51 的Adam原生与固定滞后分母局部对照已执行；观测差/MC误差量级不足，未能确定总体均值偏移。v2无动量SGD反例已否定“所有q差都须Adam”的通用必要性。
- **Unresolved:** 在相同θ/m/v/step的正常可达状态，是否存在非零平均更新差，以及该**局部映射**是否为长期selected性能的必要中介？与旧Equal/S5分开追踪。
- **Minimal discriminating evidence:** 预注册足够精度、相同状态、同一目标和固定梯度bank的直接更新对照；仅在局部差存在且具性能连接时进一步做正常训练干预。不要再跑一般Adam公式验证。
- **Links:** Q4、Q23、Q51；E036、E038、E041。

#### OQ012 — C共享路径的独立长期必要性与健康收益是否存在？

- **Current evidence:** v3局部C AUC筛查有阳性但loss/J联合门失败；N17独立embedding/E10 full约+3e−5、4/4方向同号，实际阈值和方向门失败，N4没触发。
- **Unresolved:** 能否在容量/init/预算匹配下，独立操纵C-sharing实现稳定真实selected收益、AUC/LL同向及六cell保护；还是只有局部可达性/冗余修饰？
- **Minimal discriminating evidence:** 仅在提出新的长期唯一中介主张时，提前定义完整 C-private/shared反事实及模型健康门；不能将N4未运行等同阴性。
- **Links:** Q6、Q22、Q29、Q42、Q46、Q50；E038、E042。

#### OQ013 — exposure / repetition / coverage / order 的可识别部分是什么？

- **Current evidence:** v3 WR/PERM/ORDER与N17 quota实施多项控制；固定E,N下R=E−U、E>N无法在保持配额同时无重复；ORDER同时移动短尾batch。
- **Unresolved:** 哪些高阶矩、域内支持与时间顺序可真正独立干预？不能从相互代数绑定的统计量回归推因果。
- **Minimal discriminating evidence:** 明确先检验可构造性，若不可独立操作则报告结构不可识别、不要做更多同一矩阵的防御性统计；若新干预可行，固定目标、总预算与外部选点并区分尾batch时钟。
- **Links:** Q5、Q18–Q20、Q37–Q38、Q49。

#### OQ014 — 协方差各向异性/shape 是否是独立于 scalar trace 的必要性能机制？

- **Current evidence:** v3完整冻结谱、所选坐标trace与完整C反号；N17 quota将 count 主要解释从固定点中排除，但同时改依赖/高阶矩。
- **Unresolved:** shape优于trace是否仅为描述性预测，还是具有独立、可重复的训练性能作用？
- **Minimal discriminating evidence:** 先确定能否构造同均值同trace而shape不同、同θ可达的估计器对照；不能仅将更复杂谱图当作因果证据。
- **Links:** Q21、Q42、Q49、Q51；E034、E041。

#### OQ015 — ML晚期 own-LR 反号是 q 效应，还是LR/进度与selection联合作用？

- **Current evidence:** N17固定subset、8优化seed下 own selected Δ−0.010875 但E100 +0.021145（八正），SN lr=.001、SB lr=.0003；非劣下界未达标准。
- **Unresolved:** 在**同一LR和相同步数**下 E100正向是否保留？是否改善绝对SN、SB及每cell AUC/LL并具可部署的保护？
- **Minimal discriminating evidence:** 下一轮先冻结ML同一人口、q×LR(.0003,.001)完整2×2、独立seed、固定外部计算预算与validation选择和指标；若固定LR对照不保留，优先LR/阶段解释；若保留才谈新机制。
- **Links:** Q17、Q41、Q48；E040。

#### OQ016 — 评价权重的纯算术反号与域内样本shift的性能因果如何分开？

- **Current evidence:** N0/N17证明同一预测在macro/train-p/test-p加权cell AUC下可反号，校准后AUC排名不变而LL可以改善。
- **Unresolved:** 部署目标、域内时间/标签改变与train/validation/test人口之间的差异究竟对应哪个科学估计对象？
- **Minimal discriminating evidence:** 在问题定义时事先固定**一个主评价人口与损失指标**，同时报告其余描述性权重，不以事后挑权重救援候选；若追寻shift因果须操纵或有稳定分层外部数据。
- **Links:** Q2、Q10、Q24、Q47。

#### OQ017 — KR无动量SGD q效果是否小到具有实际等效证据？

- **Current evidence:** N17共同LR SGD selected Δ+0.000837、t95[−0.004309,+0.005984]、Holm p=.6953125，横跨实用界±0.003。
- **Unresolved:** 小负、小正、零附近实用等效均未排除；不把显著性失败叫等效。
- **Minimal discriminating evidence:** 仅当此问题会改变下一阶段决策时，预注册所需区间宽度、最小有意义差和独立seed预算；不得无期限追加seed直至找到显著差异。
- **Links:** Q3、Q53；E039。

#### OQ018 — 新机制的文献覆盖边界与 MDMT 独特增益能否同时成立？

- **Current evidence:** importance sampling、adaptive optimizers、AdaTask、D3/SAF、顺序及selection原理已被相关文献覆盖；D3/SAF一般改变经验目标而非仅改变q。没有新方法达到本轮数值收益与七项保护。
- **Unresolved:** 是否存在**在MDMT特定任务/共享结构下**尚未被解决、可独立于既有机制鉴别、又有实际性能价值的问题？当前不能用“在MDMT重做通用采样/动量”本身主张新颖。
- **Minimal discriminating evidence:** 先明确单篇先验假设/迁移失败边界及最小可证伪反例，再决定是否提出方法与完整文献新颖性检索；不要把没有跑完整D3/SAF解释成其无效。
- **Links:** Q12–Q14、Q25–Q27、Q52；E043。

#### OQ019 — 正例稀疏与训练曝光如何分别影响梯度统计和更新轨迹？

- **Current evidence:** 既有记录包含不同任务/域下的正例支持、重复曝光与采样变化，但这些因素必须分开描述；不将其合并成域难度指标。
- **Unresolved:** 在固定任务定义和评价协议下，正例稀疏、重复曝光与采样权重分别如何影响梯度方差、更新方向和selected AUC/LogLoss。
- **Minimal discriminating evidence:** 只有当这些直接可观测变量会改变具体算法选择时，才对单一因素做预算受控的干预；保持域身份和任务定义清晰，不以域大小推断困难度，也不把结果用于证明或否定“小域不是困难域”。
- **Links:** Q7–Q9、Q43；E035、E038、E040。

#### OQ020 — 何时可把局部/后期收益提升为selected泛化改进而非只是慢学习/减轻过拟合？

- **Current evidence:** S8 late救援不超过原best；N17 KL/ML有早期、长期与selected反号，并且统一test曾在历史使用。某些onset只是操作性规则、无完整FWER。
- **Unresolved:** 配置/时间窗口改变的是真正有用的外部性能还是仅把选点/预算或已过拟合状态重新排列？test过去被开发使用，不足作独立方法验收。
- **Minimal discriminating evidence:** 对已明确的单一候选预注册主要selection和固定预算、基线绝对性能、AUC/LL及全部cell guard，再使用新的未开发外部/时间后置确认人口；若没有新候选则不追加无差别验证。
- **Links:** OQ005、OQ009、Q16–Q17、Q36、Q45、Q48；E027、E037、E040。

**执行/证据状态规则：** 原始历史原件找不到写 NOT_IDENTIFIABLE / INCOMPLETE；实验触发条件未满足写 NOT_TRIGGERED；方法性表现未达冻结阈值写 NO_GO（仅对该候选）；置信区间太宽写 INCONCLUSIVE；不能将这些状态统一改成“实验否定”。以上问题和Q1–Q53不以数量配额封顶。


#### 2026-10-26 更新的可区分问题 OQ021–OQ028

#### OQ021 — D套件的增益到底来自样本区分、总loss尺度还是复制/噪声？
- **Existing evidence:** ML D−B AUC +0.006303，Dconst01−B +0.005676，各4/4正；D权重快速靠近0.1，Dconst01并未对D通过非劣门。KR方向与区间更复杂。E050–E051。
- **Still unidentified:** 常数0.1和Adam耦合衰减/eps/步长交互、真实样本权重初值、噪声与复制、持续样本差异各自增量。
- **Discriminating test:** 优先做**同0.1损失尺度且无复制/噪声**的最小匹配控制，两个数据×四seed，不新搜超参；若这一臂能解释Dconst01相对B收益，先收紧增强主张，再决定是否需要分拆动态权重。[源报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。

#### OQ022 — 自然ML下N19/CM19那些人工子集结论能保留到什么程度？
- **Existing evidence:** 人工187500与自然800167差异被审计；10-20部分子阶段完成，但当时N19/CM19完整transfer仍进行中，不能将旧N19/CM19结果冒称自然复现。E048–E049。
- **Still unidentified:** 人口比例、支持覆盖、有效训练曝光和真实域任务分布对条件路径/容量作用的分离因果。
- **Discriminating evidence:** 若后续要升级外部效度，先取原规划的**完整且冻结的自然scope终报**或新的独立支持窗口，再做同身份对照；不要把源目录存在视为完整实验。[原状态](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)。

#### OQ023 — 强目标直接预测之后，foreign来源内容还有什么独立可读增量？
- **Existing evidence:** LPS有部分foreign读取信号；DFD ML需求读优于常量，但完整形成器输H_DIRECT/ETA_DIRECT；不同可达读取范围的对照尚缺足够支持。E056–E057。
- **Still unidentified:** 收益来自来源监督、可读隐藏特征、额外目标拟合、容量或单纯概率校准？
- **Discriminating test:** 固定目标/来源/标签预算，分别比较H_DIRECT、ETA_DIRECT、target-only、普通source logit/hidden读取、相同容量读取器，再谈重新形成。该判别被列为[RCF-v1 S1](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)，**尚未运行**。

#### OQ024 — 域内部群体需求能否指导真正不同的来源后继函数？
- **Existing evidence:** DFD的非恒定责任与梯度存在，但GEOM/SHUFFLE及UNIFORM强对照削弱“形成特定需求功能”解释；KR当前高维需求选读入口失败。E057。
- **Still unidentified:** 在相同总参数/目标数据和source-retention预算下，两个来源函数是否有交叉群体优势？是否只由更强的单函数、统一锚定微调或残差校准复制？
- **Discriminating test:** 用目标群体OOF剩余风险形成函数后冻结，检测交换群体/置乱配对、单宽函数、固定来源读和统一微调的**同reader**风险差；先满足来源保持及逐格保护再宣称功能分工。[RCF设计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)。

#### OQ025 — 读取来源与反向更新来源的利益是否可能不同甚至相反？
- **Existing evidence:** S3/S4共享access、CM19路径数值活性与v3 LPS来源读取不一致；固定路径失活不能反证读取的价值。E046–E047、E056。
- **Still unidentified:** 只读的有用内容、允许目标标签重写来源函数、对来源owner造成的损害及目标短期收益之间的非对称关系。
- **Discriminating test:** 固定函数内容及目标读法，分开readonly、target-to-source受约束后继更新及source owner retention；报告所有受影响任务的LL/AUC和成本。不将来源损伤隐藏在宏平均中。

#### OQ026 — selected优先的真实泛化增量与晚期减损如何分开？
- **Existing evidence:** S8/N17、10-24和v3 RHD多次出现selected与final反号；v3 RHD e50有减损但selected未胜强参照。E053–E055。
- **Still unidentified:** 动态策略改变的是最佳可用前沿，还是只是减轻某个固定过训状态的下降？
- **Discriminating test:** 预注册合法主selected与共同阶段readout，至少包括每seed/六格/AUC+LL，不用历史暴露test调整选点；有新时间/人口时再做正式确认。

#### OQ027 — 新RCF的来源行为保留约束能在目标有益时真正成立吗？
- **Existing evidence:** v4没有完整来源外层refitη/μ分组导出；旧方案没有来源目标风险与source owner风险共同满足的已验收多函数实例。E057。
- **Still unidentified:** 目标剩余风险获得的增量是否只是透支来源质量、容量或额外监督？后继函数身份与强直接拟合能否区分？
- **Discriminating test:** RCF方案的源保留LL、KL和修正幅度约束须在冻结后逐项按实际预测核验，并与统一微调、固定读取、target-only保持匹配；**当前仅设计**。

#### OQ028 — 如何获得真正可升级的独立人口确认，而不是重复使用历史test？
- **Existing evidence:** 10-25/26 v3/DFD的旧test、旧Q曾暴露；四seed重复主要是训练随机性，非新的用户/物品/时间支持；KuaiRand缺可信原始用户ID用于用户cluster统计。
- **Still unidentified:** 对新人口与真实场景的跨域可迁移收益、漂移与特征支持如何影响方法净效益。
- **Discriminating evidence:** 先明确尚未用于开发的新自然时间窗口、可归属用户/任务及主要统计单位；无这些材料时保留探索性标签，不为多seed或更多Q-readout强行制造独立确认。

**优先级：** OQ023–OQ025/OQ027组成一个具有独立主张与强简单对照的共享研究问题；OQ021是有限预算的样本重加权归因支线；OQ022/OQ028是能否升级外推结论的条件，**不是要求所有新研究开始前把旧问题全补完**。

#### 2026-11-08 更新后的决策问题（OQ029–OQ036）

#### OQ029 — ML selected Test LogLoss代价是否能通过独立校准改善？
- **已有证据：** EMA_MECHANISM的F−online selected Test LL差，ML clean为+0.001248644、ML Gaussian为+0.005334282；KR clean为−0.001225342、KR Gaussian为+0.000327687。ML E40分别校准后，F−online LL由+0.004593/+0.003986转为−0.000255/−0.000451，说明概率尺度能解释该E40锚点的大部分差异，但不等于selected Test LL代价已消失。
- **仍待识别：** selected Test的ML LL代价中，概率尺度与轨迹/选点造成的部分各占多少；在保留AUC增益的情况下是否有独立校准收益。
- **最小判别路径：** 只有当论文目标或明确LL保护要求影响候选去留时，才用未参与模型选择的校准集冻结温度/截距，再在独立Test评估；不重做EMA AUC有效性实验。
- **11/15更新：** ML两态B相对F的Test LogLoss在Clean/Gaussian分别下降0.00375756/0.00271183，95%区间均低于0，但新增AUC仅+0.00003113/+0.00009666且区间跨0。该结果说明LL改善可以独立存在；没有直接校准实验，不将其称为校准收益。

#### OQ030 — 如何使域×任务信息实际进入EMA参数时间递推并带来未来收益？
- **已有证据：** E065–E069中的SAD/AIG/B2/TG/AG当前版本没有形成稳定强增量；AUG_HISTORY的方向响应、路径分组和梯度漂移协方差当前版本均未超过固定EMA。EMA_MECHANISM中M2/M3对同刻局部风险有较高对应，但对独立未来AUC的预测明显变弱；C_design未来收益数值预测MAE在四组更好。
- **设计要求：** 不能只增加梯度视图或域任务标签；必须说明域×任务观测如何产生可预测的未来收益信号，以及该信号如何进入实际EMA参数更新、时间平均状态或动量递推项。
- **替代设计比较：** 至少比较固定EMA、简单C_design类预测、M2/M3类梯度指标，以及同预算的简单自适应规则。只在新增信号可能改变算法选择时开展五seed实验。
- **11/15更新：** 11/10同状态current-preconditioned单步干预约+0.00032至+0.00041 AUC，但11/11六轮A_MATCH反馈在ML/KR均使F下降。当前A_MATCH反馈版本停止；OQ030仍保留为更广泛的研究问题，但后续设计必须改变反馈信息/状态进入递推的方式，不能重复当前alpha混合方向。
- **优先级：HIGH。** 这是下一算法设计主问题；当前已测B/A/C版本不重复运行。

#### OQ031 — 局部动作收益如何转化为六格保护下的整体净收益？
- **已有证据：** EMA_MECHANISM中F−online六格AUC均值在四个条件下均为正；但单组动作整体增益小。KuaiRand Gaussian d0t1共享组S的M3_direct局部净益+0.000346，相对C_simple的增量仅+0.000048355、SD 0.000199264。AUG_HISTORY的完整配置总收益与组件增量也出现分离。
- **仍待识别：** 局部动作在新时间窗口/独立人口下能否复现，以及对其余五格、Traffic/Pooled和LogLoss的联动是否仍受保护。
- **最小判别路径：** 若下一设计确实采用局部动作，只冻结该局部范围与最强简单参照，在未参与开发的新时间窗口评价全部六格；不扩大为无边界的动作搜索。

#### OQ032 — 论文基线对比的当前协议差异是否还需要额外处理？
- **状态：CLOSED_FOR_CURRENT_SCOPE。** 11/04已完成M3oE、HyperGate、AutoIFS统一自然人口和Validation六格Macro AUC最早严格最大值选点；当前协议内ML由M3oE最高、KR由HyperGate最高。旧10/29 V-LogLoss选点结果不跨表排名。
- **保留边界：** 三个模型仍使用各自有依据的训练设置和工程适配，不能声称同超参训练或严格复现原论文。只有正式论文主表需要更强协议同构且该差异会改变结论时才重开；当前不安排重复基线实验。

#### OQ033 — 固定EMA超越年龄匹配online的收益由哪些可操作信号预测？
- **已有证据：** 四组晚期Validation年龄匹配online对照均为正；固定EMA早期存在明显滞后，晚期排序变化率降低。单纯低波动无法解释效用方向；同刻梯度风险指标预测未来AUC较弱，C_design的未来数值预测更好。
- **仍待识别：** 哪些训练期可观测量能在不读取未来Test的条件下，预测某个时间尺度/历史组合对未来排序的净收益，并同时反映早期滞后成本。
- **最小判别路径：** 先用已有轨迹做严格按时间和seed隔离的预测审计；若某信号改变时间尺度选择，再冻结一个简单规则做五seed验证。不再重复固定EMA与online的基本对比。
- **11/15更新：** 11/09 A0中训练面板方向持续性没有迁移为稳定验证动作增量；11/10的H1/H2不是当前主解释、H3仅条件性支持；11/11四种cell系数规则均NO_FUTURE_SIGNAL。当前预测器/规则不再扩展同类网格，只有提出能改变实际参数时间递推且可能改变算法选择的新信号时才重开。
- **优先级：HIGH。**

#### OQ034 — 分组常驻时间尺度的交叉收益能否在其他条件复现？
- **已有证据：** 四组交叉评价只有KuaiRand Gaussian为正（+0.000375±0.000331，5/5），其余三组为负；AUG_HISTORY的GROUP/GLOBAL版本也没有稳定selected增益。
- **仍待识别：** 当前正向结果是可迁移的条件性时间尺度偏好，还是仅限本轮KuaiRand Gaussian协议。
- **最小判别路径：** 仅当下一方法明确依赖分组rho时，才用训练期可得统计量预注册分组规则，并在新的时间窗口进行交叉评价；不做全域大网格搜索。
- **优先级：MEDIUM，条件触发。**

#### OQ035 — C_design/M3局部动作的增益能否跨时间窗口保持？
- **已有证据：** C_design在四组未来数值预测MAE最好，但整体动作增益小；M3_direct在KuaiRand Gaussian d0t1共享组S有+0.000346局部净益，M3相对C_simple的增量弱。
- **仍待识别：** 局部动作是否有超出简单身份/时间偏好的稳定增量，以及是否保持其他五格的AUC保护和LL约束。
- **最小判别路径：** 只冻结该局部范围、M3/C_simple/统一动作三种规则，在一个未参与开发的新时间窗口比较；若没有新窗口，不把重复seed当作独立外推。
- **优先级：MEDIUM，局部且条件触发。**

#### OQ036 — Gaussian×EMA交互是否存在可复用的阶段条件？
- **已有证据：** selected Test交互AUC在ML/KR分别+0.000007957/+0.003150528，但均仅3/5 seed正向；Validation selected与Test E50交互方向不同。
- **仍待识别：** 是否存在可由训练期信号识别的阶段条件，能在新seed/新时间窗口中稳定预测交互方向。
- **执行条件：** 只有当后续方法明确依赖Gaussian×EMA协同主张时才研究；当前不把交互写成普遍机制，也不为无决策价值的协同验证追加实验。
- **11/15更新：** 11/10四条件晚期Gaussian×EMA差中差的AUC区间均跨0；不支持统一额外协同。仅当后续算法明确依赖该交互时才重开。
- **优先级：LOW，条件触发。**

#### OQ037 — 两态历史读出B的增量排序价值能否在新seed上复现？

- **已有证据：** MovieLens两态B相对F的Test AUC增量在Clean/Gaussian仅+0.00003113/+0.00009666，区间跨0，但LogLoss两组均稳定下降；KuaiRand的B_SHARED主候选未胜F，B_H277两态诊断读出Test ΔAUC +0.00069911、5/5正向，95%区间下界−0.00000416。MovieLens完整候选虽通过发表模型AUC门，但大部分总收益来自既有F。
- **仍待识别：** KR两态B_H277的小幅Test排序信号能否在新的独立seed池中保持；ML两态B的LogLoss改善是否同时保留且不牺牲AUC；B相对F的增量与B相对发表模型的总收益必须分开。
- **最小判别路径：** 若继续，只用新的五seed，冻结ML的B_H40_all_logit、KR的B_H277_all_logit，并与F、B_SHARED比较；沿用相同数据/训练/选点协议，报告六格AUC、Macro/Traffic/Pooled和LogLoss。不得在这轮扩大H/scope/mode搜索，也不得把已有2022–2026 seed重算当作新独立确认。
- **优先级：MEDIUM，单次有界确认；若没有新seed预算则保持候选状态，不追加同seed网格。**

**优先级收敛：** OQ030/OQ033是自适应EMA研究的主要问题；OQ034/OQ035只在具体设计依赖分组时间尺度或局部动作时触发；OQ029/OQ036属于明确目标触发时的评价问题；OQ032在当前范围关闭。历史OQ001–OQ028保留原始裁定，不自动成为下一轮必须补完清单。



#### 2026-11-23 · 条件历史与共享更新

KuaiRand 的 A21−A10 从 Validation −.001989 转为 Test +.004062（5/5），且相对 clean 的最大域两个任务都下降。若有独立未暴露人口，固定模型、样本对、选点规则，比较 A00/A10/A21 的宏指标与 d1t0/d1t1，可检验当前收益能否迁移。原 B 选源规则的完整训练门已关闭。 [完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md)

## 稳定事实 SF001–SF056 及后续记录

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Stable_Findings.md)

> **2026-11-15 最新结论：** 历史条目保留原日期和当时证据强度；最新新增SF049–SF056，统一判断见[Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)。不把旧日期的“最新决定”覆盖新的正式实验，也不重复登记相同seed或未触发阶段。

> **2026-10-26增量：** 原SF001–SF020保留在各自历史估计对象和数据人口内。新增SF021–SF028，区分事实、局部反例和方法独有性；[跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)为当前统一结论。此处“稳定”表示可在限定协议内复述，不暗示新用户/时间窗口的普适性。



> **2026-10-18 补充：** SF001–SF012 是 10-09 历史协议内的稳定事实，后续不同 lineage 的阴性不能直接将其全局抹除。新证据与适用边界见 SF013–SF020；当前总体解释以[多分支综合报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)为准。


以下是当前可安全复述的保守结论。每条都限定数据集、objective、模型、训练阶段或 checkpoint；已知例外列在同一条中。强度按阶段报告证据与后续反证综合判断。

#### SF001 — 已测 Sample + SD reference 下未发现普遍 practical residual NT

- **Finding:** S1 的 Sample joint training 在所测 MovieLens/KuaiRand 与 M3OE/MMoE/PLE 条件下整体优于 single-domain reference；72 个汇总单元中 0 个达到 stable practical NT 门槛。
- **Evidence IDs:** E009、E018。
- **Scope:** Sample objective、已测自然数据比例、四 seeds、S1 reference 与 checkpoint 规则。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 单元级仍有小负差；不覆盖 Equal、极端未测 ratio 或所有 target 域。

#### SF002 — Equal 与 Sample 的 selected 域级表现不同

- **Finding:** Equal 在 S2 的特定 Large-role / target cells 上出现可重复 selected loss；Sample 在同一报告预设门槛下未出现 practical selected loss。
- **Evidence IDs:** E004、E010。
- **Scope:** S2 的 KuaiRand natural/MovieLens d2_large、M3OE、SD reference 和 selected six-cell macro 规则。
- **Evidence Strength:** STRONG。
- **Known exceptions:** S2 MovieLens late/common views可能减小或反转差异；这不证明 Sample 在任何 objective 上普遍更好。

#### SF003 — KuaiRand Equal/tab1 的 shared-access 干预有条件性 target 效果

- **Finding:** 在 S3 KuaiRand natural/M3OE/Equal/tab1 selected 对照中，TARGET_ONLY_SHARED 相对 FULL 有约 +0.039 AUC target effect，且四 seeds 同向。
- **Evidence IDs:** E011。
- **Scope:** 指定模型、dataset、objective、target 和 checkpoint。
- **Evidence Strength:** STRONG。
- **Known exceptions:** Sample 同一干预近零；MovieLens 与 later training 有不同结构；该因果效果不证明 pure gradient-conflict geometry。

#### SF004 — S4 两类 magnitude controls 都不能解释该 target rescue

- **Finding:** raw-gradient-norm match 与 applied-step-norm match 均未复制 T−F rescue。
- **Evidence IDs:** E013。
- **Scope:** KuaiRand natural/M3OE/Equal、S4 A2 的 N/U/F/T arms。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 这排除两种充分解释，不单独确认其他机制。

#### SF005 — Equal KuaiRand d1 rescue 在 S5 跟随完整 persistent history carrier

- **Finding:** S5 中 HT 保留了约 96–99% 的 T−F target rescue，而 DH 对 F 的 selected target 差约 0.001；这是完整 persistent Adam history carrier 的配置内事实。
- **Evidence IDs:** E015。
- **Scope:** KuaiRand-Pure/natural/M3OE/Equal/tab1、4 seeds。
- **Evidence Strength:** STRONG。
- **Known exceptions:** moment gate 为 15/16；不能拆成 m、v 或 bias-correction 单独机制，也不能外推到 Sample。

#### SF006 — Target rescue/history structure 随 objective、dataset 与 checkpoint 改变

- **Finding:** Equal 下 selected rescue 在 S5/S5-ML 可观察，但 Sample 下 S6 Phase A/B selected target contrasts 均未达到 +0.003 practical gate；Sample MovieLens 仍有 late DH trajectory effect。
- **Evidence IDs:** E017–E021。
- **Scope:** 两数据集、M3OE、Equal/Sample 阶段对照；Sample Phase B KuaiRand 保留 M1 限定。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** “selected effect below threshold”不等于严格零；MovieLens late fixed checkpoints 结果不能代替 selected gate。

#### SF007 — S8 未证明正常 selected model 的 practical optimizer-policy gain

- **Finding:** Selected-anchor continuation 的 T/DH 对 F best-AUC 差距接近零且低于 practical threshold；所有 MovieLens continuation best 仍低于该 seed 的冻结 F original best。
- **Evidence IDs:** E026、E027。
- **Scope:** S8 的完整 state、四 seeds、两数据集、selected/middle/late anchors 与 h=2-epoch primary horizon。
- **Evidence Strength:** STRONG。
  - **Known exceptions:** MovieLens late-anchor macro final-step 可有 positive rescue，但 seed-skewed、target gain小、未形成更高 best。S6-R2 另见 KuaiRand R0-selected test-side gain（E028）；其 validation 方向相反，且不改变 S8 的 MovieLens endpoint 结论。

#### SF008 — MovieLens late gradient utility 与 overfit decline 同步，但不是充分性能机制

  - **Finding:** S7-A MovieLens late U_k 与 target validation deterioration 同步；KuaiRand 负 U_k snapshot 没有相同性能后果。S7-B 同 θ 改变 late utility 后，短程 validation 未形成稳定改善；anchor39 step5 target AUC 均值为 −0.001226，但仅 3/4 seeds 为负，LogLoss 均值为 +0.011130、3/4 变差。结论应限于“未见稳定收益”，不能称为逐 seed 一致恶化。
- **Evidence IDs:** E023–E025。
- **Scope:** Sample/M3OE、KuaiRand 与 MovieLens 的所测 anchors/checkpoints。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** S7-B 使用短窗口与 fresh-Adam private-state 近似；S8 full-state continuation仍不支持 selected-best gain。

#### SF009 — 早期实验未单独识别 domain size 的独立因果效应

- **Finding:** 早期large-role关联存在，但后续对照未单独操纵domain size；目标权重、域身份、样本多样性、曝光与checkpoint等因素共同变化。
- **Evidence IDs:** E002–E004、E010、E014、E020。
- **Scope:** 跨P0–S6的历史实验记录；仅描述旧实验的可识别边界。
- **Evidence Strength:** STRONG（关于旧设计未单独识别size-only效应）；不对域难度作推断。
- **Research boundary:** 小域不是困难域。本条不用于证明或否定该固定判断，也不构成继续研究域大小与困难度关系的开放问题。

#### SF010 — 当前阶段必须同时报告 target/non-target、AUC/LogLoss 与 checkpoint

- **Finding:** S4–S8 中多个 aggregate gain 与 subgroup/target metric、或 selected 与 final/common checkpoint 出现不同方向；任何单一 overall metric 都可能掩盖 trade-off。
- **Evidence IDs:** E014、E016–E020、E022、E027–E032。
- **Scope:** 已报告的 AUC/LogLoss、macro/worst、per-cell、validation/test layer、parameter-swap 和 training-stage 视图。
- **Evidence Strength:** STRONG。
- **Known exceptions:** 这是报告口径结论，不等于所有指标都同等重要或存在一个统一部署目标。

#### SF011 — KuaiRand S6-R2 的 DH test selected non-target gain 不出现在 validation

- **Finding:** Sample/M3OE 下，canonical R0-selected KuaiRand checkpoints 的 DH−F test ΔN 为 +0.0068（4/4 正，3/4 ≥ +0.005），test ΔM 为 +0.0044（4/4 正），而 test ΔT 均值为 −0.0004。相同 checkpoint 对的 validation ΔN 为 −0.0039（仅 1/4 正）。
- **Evidence IDs:** E028–E029。
- **Scope:** KuaiRand-Pure natural；F vs DH；四 seeds；R0 validation-macro selection。R1/R3/R4 的 test robustness 仅见于 A-grid sensitivity。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** seed2023 target test delta 为 −0.0047；seed2025 的 test non-target delta 近零；alternate-rule exact test values 不完整；此 finding 不识别 test-side gain 的机制。

#### SF012 — S6-R2 的 KuaiRand cross-partition swaps 未保留完整 DH 解的表现

- **Finding:** 四 seeds 中，C/D hybrids 的 test macro 均显著低于完整 F/DH parent solutions；E_hybrid 近似 C_hybrid，F_hybrid 近似 F_full。此为所测参数交换下的非组合结果，不是对 DH 增益来源的定位。
- **Evidence IDs:** E032。
- **Scope:** KuaiRand-Pure natural；Sample/M3OE；四 seeds；parameter_manifest 的 shared/architecture/private 分区。
- **Evidence Strength:** MEDIUM。
- **Known exceptions:** seed2025 的 DH−F non-target denominator 近零，retention ratio 不可解释；结果不能推广到其他 partition definition 或模型。


---

#### 10-15 至 10-18 跨分支追加稳定事实（编号续接，不重写历史）

#### SF013 — B1 的“未复现”与旧 S6-R2 阳性必须同时保留

- **Finding:** KR Sample F/DH 新B1完整8run、四seed R0 test non-target AUC +0.000579（2/4正）未复现旧 S6-R2 +0.006757（4/4正），但原初始化和状态谱系不相同；R1/R4仍有test条件阳性和target LL代价。
- **Evidence IDs:** E028–E030、E033。
- **Scope:** 两个独立谱系与validation-only选点，不能当旧实验的同身份反证。
- **Evidence Strength:** STRONG（数值及身份），MEDIUM（新B1主选点无稳定实用收益）。

#### SF014 — 同经验目标的自然/平衡域采样会产生协议条件性的总体训练差异

- **Finding:** 同α/不同q的固定θ均值梯度可相同但covariance不同；Stage2D KR Sample/Balanced验证选点宏ΔAUC −0.010406，四seed负。该q整体干预同时改变抽样高阶矩、重复曝光与轨迹。
- **Evidence IDs:** E034–E035。
- **Scope:** Stage2D KR/M3OE、learned fusion/dual Adam、特定50epoch；并非所有fixed fusion/optimizer LR/数据集。
- **Evidence Strength:** STRONG（所测条件），WEAK（归因单一中介）。

#### SF015 — Adam 不是同目标采样效应的通用必要条件

- **Finding:** v2 的无动量SGD在fixed fusion协议下产生 SB−SN selected负效应（−0.002104，4/4负）；v3也有固定SGD的早期负向读数。由此足以反驳“只要改q造成差异，就必须有Adam历史”的**通用量词命题**。
- **Evidence IDs:** E036–E037。
- **Scope:** 否定通用必要性，不否定 Equal/S5 history bundle的历史局部结果，不宣称SGD与Adam机制等价。
- **Evidence Strength:** STRONG（存在反例）；长期 m/v 单独必要性仍未识别。

#### SF016 — 最新N17 KR选点宏结果统计主门未通过，不应按种子多数宣布普适损伤

- **Finding:** 8seed配对下Adam SB−SN −0.005203、t95[−0.012006,+0.001600]、Holm p=.171875；SGD +0.000837、t95[−0.004309,+0.005984]、Holm p=.6953125。早期 KR Adam AULC1..20 −0.007480 是单独次级阶段现象。
- **Evidence IDs:** E039–E040。
- **Scope:** N17固定模型/冻结选点test；test有历史使用，不是pristine holdout。
- **Evidence Strength:** STRONG（数值与检验状态），WEAK/INCONCLUSIVE（普遍selected效应）。

#### SF017 — 训练预算与选轮规则可以出现同组不同方向，不能以单一端点替代其他端点

- **Finding:** ML固定subset独立八seed own-LR q-gap selected −0.010875，而E100 +0.021145；两臂LR不同。v3固定/校准 optimizer 与early/selected 也有反号。
- **Evidence IDs:** E037、E040。
- **Scope:** 不同估计对象，E100正向并不是同LR固定q策略的已证明可部署收益。
- **Evidence Strength:** STRONG（数值方向），MEDIUM（训练阶段/预算影响），WEAK（独立归因）。

#### SF018 — 评价加权方式可以对同一模型同一预测给出相反AUC差值

- **Finding:** 历史calAdam E50 macro Δ−0.009074、train-p-cell +0.001016、test-p-cell +0.003108；ML own在新轮 macro −0.010875、train-p-cell +0.000744、test-p-cell −0.013586。AUC cell加权不等于pooled AUC。
- **Evidence IDs:** E040。
- **Scope:** 同预测不同评价权重的精确算术；不识别域内分布变化的因果原因。
- **Evidence Strength:** STRONG。

#### SF019 — 观察到的固定点协方差差异不能主要归于域计数波动

- **Finding:** N17 N2 16锚的q trace比在IID及quota约2倍以上，计数波动仅占模型预测trace的0.001297%–0.105936%；已排除**这组固定点**“随机域count主导”的解释。
- **Evidence IDs:** E041。
- **Scope:** 给定锚/数据/估计器；quota还改变批内依赖，未识别长期trace、m/v、domain size独立因果。
- **Evidence Strength:** STRONG（局部反证）。

#### SF020 — 共享C仅有小规模条件局部作用，长期唯一必要性没有完成检验

- **Finding:** v3 C局部AUC筛查曾阳性但loss/J联合门失败；N17 embedding/E10独立确认两个路径均4/4方向为正、绝对约3e−5，实用/方向门未过，N4长期拓扑按预设不触发。参数支持已核对C/E共享、R私有、P混合。
- **Evidence IDs:** E038、E042。
- **Scope:** 同θ/局部干预；既不是“C全无效”，也不是“共享C必要中介已证实”。
- **Evidence Strength:** MEDIUM（局部及门槛），WEAK（唯一长期中介）。

**截至2026-10-18的历史总体阶段决定：** 条件效应存在，部分历史统一解释被反例削弱；当前没有已测候选同时达到预定收益与非劣保护，方法决定为**本轮 NO_GO**。这不是未来所有MDMT方法的否定，尤其不能把“小域=困难域”当作曾经成立的前提去反复证伪（E043）。 


#### SF021–SF028：2026-10-26更新后的限定稳定事实

#### SF021 — 人工与自然MovieLens是两种不同训练人口
- **Finding:** 多项历史ML使用人工187500行/域34500:3000:150000；自然基础训练800167行/155949:318599:325619。比较还改变N、支持和训练时钟；自然补充整体尚处IN_PROGRESS。
- **Evidence IDs:** E048–E049；[自然人口审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)。
- **Strength:** STRONG人口事实，WEAK比例单因果。原人工实验保留条件效度。

#### SF022 — N18/N19实质完成，方法增量仍需强对照
- **Finding:** N18完成六方向有界诊断/冻结头；N19完成328正式网络训练与冻结测试，未建立稳定足够强的独立新方法贡献。DATA-BLOCKED/NO_GO仅限各自候选。
- **Evidence IDs:** E044–E045；[N18](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)、[N19](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)。
- **Strength:** STRONG执行，MEDIUM条件反例，WEAK通用方法化。

#### SF023 — 共享乘性路径数值中性不等于所有模型共享无效
- **Finding:** CM19 KR fixed M3oE某输入乘性路径selected 24/24 cell中和预测不变；双factor正则/局部AdamW干预恢复响应，STAR等独立架构仍活跃，但selected整体泛化收益不稳定。
- **Evidence IDs:** E046–E047；[CM19](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)。
- **Strength:** STRONG特定路径数值事实，WEAK普遍性能解释。

#### SF024 — D套件有效但动态样本区分的独立增量未证明
- **Finding:** 10-21 ML D−B test宏AUC +0.006303/LL −0.015962、Dconst01−B AUC +0.005676，均4/4有利。D权重趋向0.1下界；Dconst01对D没有通过正式非劣判定，仍保留复制/噪声与总尺度。
- **Evidence IDs:** E050–E051；[D/G报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。
- **Strength:** STRONG ML套件效果，WEAK动态权重单独因果；legacy_group_sum不可混作Sample/Equal。

#### SF025 — 当前匹配梯度计算路径的τ调幅没有终点宏AUC优势
- **Finding:** G−Gtau0的ML/KR test宏AUC分别−0.003466/−0.006484，均4/4负，但KR置信区间跨0；KR早期validation更快、部分域或pooled方向可不同。
- **Evidence IDs:** E052；[梯度对照](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)。
- **Strength:** STRONG ML匹配终点，MEDIUM KR方向，WEAK跨模型推论；G−D非纯τ对照。

#### SF026 — 历史收益控制的末轮减损不同于最佳selected改善
- **Finding:** 10-25 v3 KR RHD HB−BASE正式test LL在selected +0.00074075而e50 −0.00875428；各方法自身e50仍差于合法selected。10-24 I1和10-25 v2没有通过各自selected主优势/晋级门。
- **Evidence IDs:** E053–E055；[v3](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)。
- **Strength:** MEDIUM阶段性实证，不是可部署最佳性能证明。

#### SF027 — 来源可读、需求可预测与特定来源函数值得形成是不同命题
- **Finding:** v3 LPS有部分foreign读益但L4形成未稳定超过简单对照；DFD ML P1需求选择优于常量，KR P1失败，ML完整方法弱于H_DIRECT/ETA_DIRECT。
- **Evidence IDs:** E056–E057；[v3](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md)、[v4](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md)。
- **Strength:** MEDIUM限定观察，WEAK独有形成机制。非零责任不足以证明原型分工。

#### SF028 — 版本性停止、漏导出与尚未训练不能被误记为阴性实验
- **Finding:** NEXT-v2四seed正式50epoch NO_START；DFD P3/P4 NO_START、184预测/12计时完成但外层refitη/μ分组MISSING/MAJOR；最新RCF-v1仅为研究设计。
- **Evidence IDs:** E054、E057；[DFD-v4](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md)、[RCF方案](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)。
- **Strength:** STRONG执行/资料状态；不能据此否定全部局部/多原型共享思路。

**最新总体判断：** 优化响应、样本重加权和来源内容的条件收益值得研究，已测动态方案暂未建立普适强增量。研究者从未将“小域=困难域”作为成立先验；后期研究应先区分具体机制和强简单基线，而非反复防御性否定同一命题。

#### 2026-11-02 新增有边界的稳定事实 SF029–SF037

#### SF029 — 所测来源关系与BASP-A原型政策尚未胜强目标直接适配
- **Finding:** 10-27 Relation-Basis v2低秩/全关系在两自然数据四seed的冻结Q决策下没有胜TARGET_ONLY，低秩在历史E宏AUC各4/4负；10-28 BASP-A的CG相对TARGET_MATCH不稳定，留出收益μ与NULL读取记录限制当前政策解释。
- **Evidence IDs:** E058–E059；**Strength:** MEDIUM具体版本停线，STRONG所测机制状态；**Boundary:** 保留历史来源读取与局部群体收益问题，不否定整个共享方向。

#### SF030 — HyperGate与AutoIFS已在当前仓库正式五seed比较
- **Finding:** 10-29两自然数据的HyperGate/AutoIFS相对M3OE_FIXED1宏AUC四组配对均5/5改善；KR AutoIFS macro LL改善而natural LL五seed变差。原论文参数完整核实及同训练预算的架构纯比较并未完成。
- **Evidence IDs:** E060；**Strength:** STRONG本协议实测，不能宣称严格论文复现。

#### SF031 — 统一损失缩放在ML可提升selected AUC，简单L2解释必须优先考虑
- **Finding:** GA31无增强ML c=.1相对c=1 selected AUC+0.003473（4/5）、KR相同缩放方向负；ML普通L2对缩放基线的selected验证差较小。耦合Adam同时改变相对wd和eps，早/末训练时钟与selected结果不等价。
- **Evidence IDs:** E061–E062；**Strength:** STRONG ML条件性效应，MEDIUM控制比较；**Boundary:** 不当成旧D动态权重的独立收益。

#### SF032 — 固定参数EMA在当前协议具有可重复排序收益
- **Finding:** ML两组五seed的EMA−online selected AUC分别+0.004399和+0.003849，均5/5正；KR+0.004616、4/5正。ML LogLoss在两组协议中多数seed恶化。EMA是非反馈shadow，不是对在线Adam动量递推的修改。
- **Evidence IDs:** E064；**Strength:** STRONG ML AUC，MEDIUM KR AUC；**Boundary:** 不是LL一致收益、不是独立新时间窗口。

#### SF033 — 当前增强的局部收益不能代替总体/流量与LL保护
- **Finding:** NR-Gauss在GA31未过主门；KR ANTI2对普通基线有条件test增量但不胜同K IID2；KR Gaussian+EMA的LL五seed改善伴随d1两任务AUC多数seed下降、pooled/traffic整体下降。必须保留六格及主指标口径。
- **Evidence IDs:** E062–E063、E066；**Strength:** STRONG某些域损害，MEDIUM独有性停线。

#### SF034 — 非零差分梯度/动态门不自动增加实用排序
- **Finding:** SAD βδ=.99实际修改Adam更新但两集selected宏AUC未胜普通增强A1；βδ=.9与普通Adam实数代数等价仍可能出现FP32长期轨迹差。AIG-M动态注入不为0而只停于开发门。
- **Evidence IDs:** E065–E066；**Strength:** STRONG实现与代数身份，MEDIUM已测性能停线。

#### SF035 — GATS任务投影B2开发收益真实存在，却未形成稳定强基线增量
- **Finding:** KR B2开发两seed在验证对A1/R2正向，保留seed确认均值大幅缩小；统一test五seed对A1宏AUC+0.000005156（3/5），对A0固定EMA只有2/5正。已完成当前阶段且明确不晋级。
- **Evidence IDs:** E067；**Strength:** MEDIUM当前版本NO_PROMOTION；**Boundary:** 保留任务级/验证期条件性观察。

#### SF036 — GATS难度覆盖、延迟增强与CV自适应EMA当前版本未晋级
- **Finding:** B1/B3/C2各在匹配控制或预定门下未建立独有macro增量，确认门未触发的臂不能声称正式阴性。EMA早期滞后/后期提升的阶段性事实并不证明CV→rho的新机制。
- **Evidence IDs:** E068；**Strength:** MEDIUM当前规则停止，WEAK普遍外推。

#### SF037 — EMA02V2任务梯度视图确实产生不同方向，但不是有益融合证据
- **Finding:** 两自然数据各2开发seed、36训练全部完成；ML F1 test对A0+EMA +0.000359（2/2）但验证均值负、幅度不足晋级；KR TG/AG对A0+EMA test宏AUC约−0.0172/−0.0175并伴随LL恶化，两seed一致。梯度探针与实际扰动正常执行。
- **Evidence IDs:** E069；**Strength:** MEDIUM当前KR退化，WEAK ML F1微增量；**Boundary:** 不是新数据确认、也没有改在线动量公式。

#### 2026-11-08 新增稳定事实（SF038–SF048）

#### SF038 — 完整候选组合的总收益不等于新增组件的独立收益
- **Finding:** 11/03五seed补全中，C/H/P01+固定EMA相对普通M3oE online的Macro AUC增量：ML +0.003253/+0.003059/+0.003225，KR +0.003720/+0.005579/+0.008702，六格保护均满足。相对固定EMA的组件增量则不同：ML三候选均未超过EMA；KR P01虽有+0.004086宏增量，但d1t0/d1t1分别−0.001334/−0.001047，未满足逐格保护。
- **Source:** 11/04 Global Summary；[五seed补全报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)。
- **Strength:** STRONG（完整配置总收益）；MEDIUM（组件增量与逐格取舍）。

#### SF039 — 统一AUC选点后的正式发表模型排序已明确
- **Finding:** 在统一自然人口、validation六格Macro AUC最早严格最大值选点的协议下，ML M3oE/HyperGate/AutoIFS Macro AUC为0.775190/0.771412/0.772425，M3oE最高；KR为0.671907/0.682113/0.678597，HyperGate最高。该排序替代旧V-LogLoss选点结果作为当前同协议参照；不同训练协议的绝对值不跨表排名。
- **Source:** [Overnight A+B报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)。
- **Strength:** STRONG（当前协议内排序）；适用范围限于已运行适配版本和本轮协议。

#### SF040 — Gaussian覆盖的收益具有数据集与指标条件性
- **Finding:** ML Gaussian D0 Macro AUC为0.779357，相对M3oE +0.004167且六格保护通过，Macro LogLoss差+0.000931。KR G-D0/G-D2/G-D02 Macro AUC为0.679180/0.677999/0.674604，均未超过HyperGate 0.682113；KR不同覆盖的AUC与LL排序不一致。
- **Source:** [Gaussian覆盖报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)。
- **Strength:** STRONG（所测配置和协议）；不外推为普遍增强规律。

#### SF041 — CP组合未形成相对P01的加性排序优势
- **Finding:** CP fixedEMA的ML/KR Macro AUC为0.777525/0.678168，相对M3oE总增量+0.002335/+0.006261；但相对单独P01分别−0.000890/−0.002441。KR对M3oE的总收益保留，但当前CP组合不替代P01排序方案。
- **Source:** [Overnight A+B报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)。
- **Strength:** MEDIUM（当前2×2组合和协议）。

#### SF042 — 当前方向响应EMA没有超过固定EMA
- **Finding:** AUG_HISTORY中B方向响应确实执行，但D−F selected Test Macro AUC仅ML +0.000000104、KR −0.000004946；没有实用增量。
- **Source:** [AUG_HISTORY报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（当前方向定义、模型、候选网格和数据协议）。

#### SF043 — 当前路径/分组时间尺度版本未获得selected收益
- **Finding:** AUG_HISTORY的GROUP/GLOBAL时间尺度相对F，ML Macro AUC分别−0.000750/−0.000728，KR分别−0.000175/−0.000213；部分早期验证优势没有转化为selected Test增益。
- **Source:** [AUG_HISTORY报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（当前实现和候选范围；不否定其他适应性信号）。

#### SF044 — 当前梯度漂移协方差增强没有超过固定EMA
- **Finding:** MovieLens C_MERGED−F selected Test Macro AUC为−0.000667190、LogLoss差+0.000357811、AUC正向seed为0/5；C3_WEAK与旋转控制的均值也未超过F。当前数据不支持真实方向信息具有独立必要性；KuaiRand C为未运行，不记阴性。
- **Source:** [AUG_HISTORY报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-05_aug_history/REPORT.md)。
- **Strength:** MEDIUM（MovieLens当前候选）；KuaiRand C为NOT_RUN。

#### SF045 — 固定EMA的selected排序收益超过年龄匹配online对照
- **Finding:** EMA_MECHANISM四组F−online selected Test Macro AUC为ML clean +0.003848845、ML Gaussian +0.003856802、KR clean +0.004616399、KR Gaussian +0.007766927；正向seed分别5/5、5/5、4/5、5/5。晚期Validation年龄匹配online对照仍有+0.002993/+0.003153/+0.004503/+0.003746，四项95%区间下界均大于0。早期F存在明显学习滞后，晚期排序变化率下降。
- **Source:** [EMA_MECHANISM最终报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** STRONG（四个已测条件的同轨迹配对和年龄匹配对照）。

#### SF046 — 常驻时间尺度交叉收益目前只在KuaiRand Gaussian条件下成立
- **Finding:** 逐格常驻时间尺度交叉评价：ML clean −0.000047、ML Gaussian −0.000082、KR clean −0.000152、KR Gaussian +0.000375±0.000331（5/5）。这是条件性局部证据，不支持普遍的分组rho策略。
- **Source:** [EMA_MECHANISM最终报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（单条件局部交叉收益）。

#### SF047 — 未来收益预测准确度与动作增益不是同一结论
- **Finding:** C_design在四组未来收益数值预测MAE均优于M2/M3，但整体单组动作收益较小。KuaiRand Gaussian d0t1共享组S的M3_direct局部净益+0.000346；M3相对C_simple的五seed增量+0.000048355、SD 0.000199264，复杂指标独立必要性证据弱。
- **Source:** [EMA_MECHANISM最终报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（C_design预测MAE）；WEAK（复杂指标独立动作价值）。

#### SF048 — Gaussian×EMA交互随阶段和选点改变，不支持统一协同规则
- **Finding:** selected Test交互AUC为ML +0.000007957、KR +0.003150528，均仅3/5 seed正向；Validation selected和Test E50交互方向不同。ML E40分别校准的诊断将原F−online LL差+0.004593/+0.003986变为−0.000255/−0.000451，但这只解释E40锚点，不能据此消除selected Test LL代价。
- **Source:** [EMA_MECHANISM最终报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)。
- **Strength:** MEDIUM（当前四条件和阶段）；不外推为通用Gaussian×EMA协同。

#### 2026-11-15 新增稳定事实（SF049–SF056）

#### SF049 — EMA_ADAPTIVE A0没有找到可执行的未来动作增量
- **Finding:** 复用既有20条轨迹、没有新增训练或Test。KR Gaussian训练面板的Fast−Slow方向持续性高于晚期验证侧；C_persist未来P_B和验证侧均为负，C_carry虽有面板正值但相对统一Fast的验证动作增量约为0，静态任务策略全部退化为统一动作。
- **Evidence IDs:** E070；**Strength:** STRONG（本轮面板/动作读出）；**Boundary:** A0不是完整闭环训练，A1/B/C未触发，不能写成完整自适应算法Test失败。
- **Source:** [11/09 EMA_ADAPTIVE](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

#### SF050 — PUB-MDMT当前P0/P1/P2迁移未胜固定EMA或普通模型参照
- **Finding:** 原始α尺度、coupled-WD下C0/C1/C2相对A均0/5 seed正向，六格均值下降且Macro LogLoss升高。C相对同α G控制有较大正差，但G的E50宏AUC接近0.5，不能把该差值当作正常性能晋级。
- **Evidence IDs:** E071；**Strength:** STRONG（当前配置/协议）；**Boundary:** 仅停止当前迁移版本，WD=0建议未执行，不否定所有PUB变体。
- **Source:** [11/09 PUB-MDMT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09_v2/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)。

#### SF051 — 固定EMA收益主要由共享参数S承担
- **Finding:** 删除S使四条件晚期验证AUC下降约0.00376–0.00436；删除N效应接近0。S内部EMBED删除效应最大（约0.00187–0.00303），STAR_SKIP次之，EXPERT接近0。
- **Evidence IDs:** E072；**Strength:** STRONG（当前M3oE fixed1与四个已测条件）；**Boundary:** 不外推为所有MDMT模型或所有共享结构的普遍参数规律。
- **Source:** [11/10 EMA因果归因](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### SF052 — 近期与较早历史存在互补，历史预测组合可复现部分排序收益
- **Finding:** 删除recent的AUC损失约0.00456–0.00499，删除old损失约0.00079–0.00252；EMBED的recent×old交互四条件均为正。q_logit/q_probability组合在固定验证面板上增加约0.00067–0.00264 AUC并降低LL，但没有独立Test选点。
- **Evidence IDs:** E073；**Strength:** STRONG（固定面板归因）；**Boundary:** 不将非线性模型收益拆成可加的唯一参数份额，也不将面板读出升级为算法Test收益。
- **Source:** [11/10 EMA因果归因](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

#### SF053 — 单步current-preconditioned收益没有自动转化为持续反馈收益
- **Finding:** 同状态同范数current-preconditioned单步干预约+0.00032至+0.00041 AUC；momentum_isotropic替代约−0.011至−0.013。11/11六轮A_MATCH反馈使F在ML/KR的两个强度下均下降。
- **Evidence IDs:** E074、E076；**Strength:** STRONG（当前方向定义与反馈窗口）；**Boundary:** 停止当前A_MATCH反馈版本，不推导所有梯度/动量反馈无效。
- **Source:** [11/10因果归因](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)、[11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### SF054 — 当前自适应EMA预测器与cell动作规则未产生稳定动作增量
- **Finding:** 11/09 A0的C_persist/C_carry和静态task政策未胜统一Fast；11/10 H1/H2不是当前主解释、H3仅条件性支持；11/11四种cell系数规则均NO_FUTURE_SIGNAL。
- **Evidence IDs:** E070、E075、E078；**Strength:** MEDIUM（已测预测器/动作空间）；**Boundary:** 只停止已测规则，不表示所有训练期信号均不可用。
- **Source:** [11/09 A0](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-09/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)、[11/10归因](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)、[11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### SF055 — 两态历史读出在ML主要改善LogLoss，在KR保留一个小幅AUC诊断信号
- **Finding:** ML Clean/Gaussian两态B相对F的Test ΔAUC仅+0.00003113/+0.00009666（区间跨0），LogLoss分别改善−0.00375756/−0.00271183（区间低于0）。KR主候选B_SHARED相对F没有AUC增量；KR Clean两态B_H277诊断读出+0.00069911、5/5正向，95%区间下界−0.00000416。
- **Evidence IDs:** E077；**Strength:** STRONG（ML LogLoss差）；MEDIUM（KR两态B诊断AUC）；**Boundary:** 不把ML LL改善解释为已证明校准，也不把KR诊断结果当作正式晋级。
- **Source:** [11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

#### SF056 — 两态B没有形成跨数据集均胜最强发表基线的统一方法
- **Finding:** MovieLens两态B在Clean/Gaussian下通过M3oE、HyperGate、AutoIFS；KuaiRand B_SHARED只超过M3oE，未超过HyperGate和AutoIFS。完整方案相对发表模型的收益包含固定EMA F已有收益，不能归因成B的独立增量。
- **Evidence IDs:** E077；**Strength:** STRONG（本轮冻结Test和既定基线门）；**Boundary:** 不否定KR两态B_H277的小幅诊断信号，后者仍需单独复核。
- **Source:** [11/11 EMA_NEXT](https://github.com/Caelestis1903/My-Research-Project/blob/26-11-11/research_reports/2026-11-11_ema_next/FINAL_REPORT.md)。

**研究结论采用：** 固定EMA的排序收益作为已接受的强内部参照；当前B/A/C自适应候选不晋级。完整配置总收益与新增组件增量分开；未来自适应设计必须提供跨seed的未来收益预测证据，并实际改变参数时间递推。AUC与LogLoss分别报告，所有结论限定于对应数据、模型、seed和选点协议。




#### 2026-11-23 · 条件历史与共享更新

STRONG（限当前冻结人口和指定对照）：A21 在 MovieLens 的 Test Macro AUC 相对已发表 M3oE 五种子全正，在 KuaiRand 相对同 PAIR 普通 Adam 五种子全正。后一个差值在同数据集 Validation 为负，ML 的组件 Test 也未改善，故不能把跨数据的稳定增益归于条件历史。两数据上晚期留出动态选源均未超过留一 seed 静态对照。 [完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/conditional_history_shared_update_20261122/FINAL_REPORT.md)

## 十月上旬至十八日分支审查

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)

> **历史快照/已被后续实验扩展（2026-10-26修订说明）。** 本文只覆盖至N17的2026-10-18状态；其中“当前”“最新一轮”“未合入实验文件”“下一轮优先q×LR”等是当时语境，不是2026-10-26的研究决策。随后已经完成N18/N19、CM19、10-21复现、10-24 I1、10-25 RHD/LPS与10-26 DFD；10-20自然MovieLens重评完成部分子项，但扩展矩阵尚未闭环。实验原件已按分支存入 [history_experiments](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/README.md)。完整更新与早期结论修订请以 **[2026-10-26跨阶段总报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)** 为准；该新文件不覆盖本文原始历史数值。


> **历史定位：本文曾是main在2026-10-18时的跨轮结论入口。** 按提交时间和真实实验依赖，整合 2026-10-15—2026-10-18 未直接合入 main 的实验执行记录。仅抽取科研事实、方案状态、反例、论断演化及未解决问题，不 cherry-pick 实验代码、checkpoint、原始预测或历史阶段进度快照。2026-10-09 的 E001–E032、SF001–SF012、OQ001–OQ010 等仍保留为原始历史记录；与本报告不一致时以**同一估计对象中较晚且最终核验的结果**为准。不同训练协议/种子/数据人口不可直接合并效应量。

#### 1. 分支与时间序列——先判断是新证据还是重复祖先

| 时间（UTC，按阶段完成） | 分支及报告入口 | 处理方式 |
| --- | --- | --- |
| 10-15 至 10-16 05:36 | [execute-2026-10-15：B1、C2 与范围审计](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-15/research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016/scientific_analysis.md) | 新 B1 重建及数值审计；不冒充 S6 历史精确复现 |
| 10-16 至 09:23 | [execute-2026-10-16：Stage2D 最终结果](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/experiment_results.md) | 该分支继承 B1；早期 Stage2+ 的“0 训练/阻塞”是已被 16/16 Stage2D 覆盖的**过程快照** |
| 10-16 至 12:00 | [execute-2026-10-16-v2：P1 与机制中期分析](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v2/research_plans/2026-10-16_v2/execution_20261016/mechanism_analysis.md) | 继承 Stage2D，但 v2 改为固定 fusion/单优化器；最新文件仍标 INTERIM，不能宣布整个 v2 已终局 |
| 10-16 至 10-17 05:19 | [execute-2026-10-16-v3：完整终局报告](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md) | 从较早祖先独立开展、与 v2 **历史分叉**；必须并列保留不同固定 fusion / 校准 LR / optimizer /时钟协议 |
| 10-17 至 10-18 07:00 | [execute-2026-10-17-run：N17 最终研究报告](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | 继承 v3；追加独立优化随机种子、固定 quota、ML 固定人口和局部反事实；当前最新正式终局 |
| 10-16—10-17 方案分支 | 2026-10-16-v2-research-plan、research-plan-2026-10-16-v3、add-research-plans-2026-10-17 | 方案正文已在 main 的 research_plans/2026-10-16_v2、2026-10-16_v3、2026-10-17。**不计新增实验证据**，不重复全量复制 |

前三个实验分支在 Git DAG 上连续，v3 与 v2 发散，N17 连续继承 v3。方案分支共有 3 条，实验执行分支 5 条，总计盘点 8 条。源分支未做 Git merge，仍保持独立可复核。

#### 2. 按时间先后提取的科研发现、限制与历史修订

#### 2.1 10-09 之前的既有主干：P0–S8 与 S6-R2（不是本轮重训）

- **Equal / Sample 是不同优化目标**；Equal 的特定 selected 损伤不能直接视为 Sample 下的实用负迁移。既有 S1/S2 对已测 Sample/自然比例未检出稳定实用 negative transfer，**不等于 Sample 任何情况下都最优**。
- S3–S5 的 KuaiRand/Equal/d1 shared-access 与持久 Adam-history carrier 影响，在特定 checkpoint 和模型中为**有边界的局部现象**。S5 的完整 history 效果不识别一阶矩 m、二阶矩 v 或共享/architecture 哪一部分独立必要；不应扩大成 Adam 对一切 MDMT 现象的必要性。
- S6 Sample selected 的 Equal 类 target rescue 不再达到原预定实用门；但后期 MovieLens 过拟合缓和效应存在。S8 未使正常 validation 选择的最佳模型普遍优于原 F best，因此“late rescue = 最佳泛化提升”这条方法化思路被否定。
- 原 S6-R2 同一组 KuaiRand R0 checkpoint：test DH−F 非 target AUC +0.006757（4/4 正），validation −0.003851（1/4 正），MovieLens 接近零。属于**历史保存工件的局部观察**，不是已经识别了可部署因果机制；不能以 test 筛选规则。
- **“小域就是困难域”从来不是本项目应采纳的结论。** 域数据量、域特征和标签、任务、曝光及评价人口共同影响指标。后续未出现“规模决定困难度”的因果证据；针对该断言的反例应作为边界事实，不应被不断当成新的核心假设反复证伪。

#### 2.2 10-15 执行分支：B1 历史现象的独立重建与 C2.1

**新增** KR natural、Sample/M3OE、F/DH×4 seed×50 epoch＝8/8 新训练；400 个完整状态、800 个预测工件数值核验通过。由于历史 init/lineage 不一致，身份为 **RECONSTRUCTION_NOT_HISTORICAL_EXACT**。R0（只按 validation macro AUC 选）test DH−F：macro +0.000363（2/4 正）、non-target +0.000579（2/4 正）、target −0.000069；target LogLoss +0.004007（恶化）。没有复现旧 S6-R2 的 +0.006757/4 正组合，**但不是证明旧报告造假或错误**。R1/R4 次选点的 test non-target +0.011584（4/4 正），两规则选择完全同一批 epoch、不计两次复制，且 target loss 代价存在。选点不是全部反号的解释：seed2022 两臂同轮仍有 val/test 差异。

**零训练** C2.1 两数据集四seed/8 case：同 alpha 改 q 的期望梯度相同，协方差不同；观察到“Balanced 必降噪”随 alpha 和所测参数分区改变的反例。仅固定 theta / 有限 pool，不能宣称训练提升或历史 Adam causality 已识别。S5/S6 原始 checkpoint、完整 history 和同θ m/v 追踪未找到；保留身份不可恢复缺口，不再当作新训练统一禁令。

#### 2.3 10-16 Stage 2+ 与 Stage2D：把“未跑”的历史状态更正为“已跑但候选失败”

本分支早期 Stage2+ 文档有“0 新正式训练/依赖阻塞”的时间快照；同分支随后完成 **16/16 Stage2D**：KR/M3OE，四 seed，Sample/Equal × Natural/Balanced，**learned fusion、双 Adam 优化器**，配对 IID，50 epoch。最终效应：validation selected 宏 AUC **SB−SN = −0.010406（4/4 负）**；**EB−EN = −0.006547（3/4 负）**。Test（已用于历史开发，仅法证）分别 −0.012674 / −0.002714。Sample 同目标改变采样有明显**负向总体干预效果**，所以“Sample 下 optimizer/sampling 效应全部消失”被修订；但 q 同时改变 batch 梯度分布、重复曝光与后续优化路径，不能锁定 Adam m/v 为性能因果。两种 Balanced 候选都未通过原定实用收益/无害性门，**本配置方法化 NO_GO**；不否认某些 cell/LL 方向不同。

固定 theta 的 Stage 2B/2C 共8 case×4096 replay×64步证实 SGD/Adam 局部递推区别，**完整 C trace 与投影坐标 trace 可显示相反的 Balanced/Natural 比值**；这对“单 scalar variance 解释全模型”构成限制，而不是选中模型收益证据。旧 S5/C×A 与 m/v 专项因缺原状态而未运行；“未运行”不等于“机制无效”。

#### 2.4 10-16 v2：无动量 SGD 反例，固定 fusion/单优化器的中期结论

v2 改为固定 fusion / 单一 optimizer；与 Stage2D **不是**只改 Adam/SGD 的严格历史配对。已核验 P1 Sample q 差：
- KR validation-selected Adam SB−SN **−0.005556**，3/4 负，seed 区间 [−0.012286,+0.005721]；不能写成稳健普遍负效应。
- 无动量 SGD SB−SN **−0.002104**，4/4 负，区间 [−0.002478,−0.001680]（按此 v2 协议）；说明 **“Adam 是采样损伤的通用必要条件”已被反例证伪**。Adam 与 SGD 的 q-gap 差区间跨 0，不能据两均值判定 Adam 额外中介必要。
- SGD calibration .1 是所测 [.1,1,10] 网格下界，不是全局调参充分；P3 参数距离、P4 域任务异质性、P5 selection、P6 投影几何只支持对应描述范围。v2 最终执行/结果/问题注册仍标 RUNNING，P2/P8 不宣称已最终完成。此中期结果与后续采用其他 LR / seed 的 v3、N17 不构成一组同估计对象的直接矛盾。

#### 2.5 10-16 v3（17日完成）：条件效应成立、唯一长期中介未成立

完成 v3 规定及触发实验与冻结 test 法证；总分类 **BOUNDARY-CONFIRMED**，方法 **NO_GO**。同 Sample J / 同 routing-detach 计算图，同目标 q 变动可改变实际有限 batch 协方差和轨迹，冻结全坐标固定θ协方差比约2.3155–2.7522；该平均梯度不变/协方差改变原理不是创新本身。

四训练 seed 的**不同配置、不同评价估计量**，不得混成一种“Sample 比 Equal 好/坏”：
| 家族 | 冻结 test selected 宏 AUC SB−SN | 主要限制 |
| --- | ---: | --- |
| fixed Adam .01 | −0.007069（2/4 负） | 区间跨0、依赖选点 |
| fixed SGD .1 | −0.002518（4/4 负） | 不是所有无动量 SGD 配置 |
| calibrated Adam .003 | −0.008698（4/4 负） | 配置/LR 作用 |
| calibrated SGD .3 | −0.000972（4/4 负） | 同共同早期37轮却 +0.001614 四正；阶段反号 |
| 第二批 SGD .3 | −0.000253（3/4 负） | 平均接近0，不是严格等效 |
| MovieLens SGD .3 | +0.000539（1/4 正） | 不跨环境普遍复现 |
| AdaTask .003（组件） | −0.018917（4/4 负） | 不代表完整 AdaTask 最优方案失败 |

**状态与机制边界：** V11 的原生/影子 m/v/history 与破坏性消融存在健康 floor（beta2=0 最终 AUC≈.5）；同θ局部状态映射作用不证明长期唯一必要性；校准 Adam 与 SGD 有有条件交互，SGD 仍给 Adam 通用必要性的反例。V7 在早期和中期的共享 C 局部 AUC 筛查 4/4 达到衰减条件，但 loss/J 联合门未通过，**长期 C 私有化训练未触发**；不能说“C 全阴性”，也不能说“共享 C 已识别为必要中介”。真实共享 C/E、私有 R gates/towers、混合 P 的参数支持已核对。

**曝光/时间：** 同 multiset 的 ORDER 在固定50轮可呈有条件正向 q 交互，但最终预算和绝对 SN 健康反向；ORDER 还移动短尾 batch，不能叫纯课程收益。重复 R=E−U、coverage 与曝光在许多配额下代数耦合；N/2x alias 复制不增加独立新样本，不能由此识别自然域数据量 N 的因果作用。谱 shape vs scalar trace 缺同 trace 的独立干预，继续 **NOT_IDENTIFIABLE** 而非无限复做固定点谱图。

**适用边界：** MovieLens 早期共同16轮 q 差正，自选又混合；V8 strong mask 的最终 d2 q gap 表面变好但两臂绝对模型均损伤；少域/大域不能定义难度。D3、SAF 往往改变 J，不与本研究固定 J 的 q 操作同构；AdaTask 是相关状态机制先例；重要性采样、方差、shuffle-memory 与选点风险有前人工作，故目前没有已证实的独特 MDMT 必要机制或“新方法四seed主终点稳定改善且全 cell 无害”。

#### 2.6 10-17—10-18 N17：独立 seed、统计主门和新的反例（目前最后一轮）

N17 **120 条新的 H100 训练轨迹（48 tuning +72 formal）全部完成**，其中 72 条正式轨迹×100 个 epoch＝7200 个已冻结模型/预测读取；N2 的16有效固定点、N3独立局部确认另行计数，**不能把沿同轨迹重复选点当独立训练**。Test 虽统一冻结读数，但曾在历史开发使用，**非 pristine holdout**。若引用本节主结果，以8个独立配对训练seed及预定 Holm 家族判定；二级阶段/cell区间只能有条件解读。

| 对比：SB−SN（固定test，selection选点） | 宏AUC差 | 8 seed t95 | 解释 |
| --- | ---: | --- | --- |
| KR Adam/common LR | **−0.005203** | [−0.012006,+0.001600]，7/8负，Holm p=.171875 | **主统计未确认**；bootstrap 区间负不能事后替换预定 t/符号检验 |
| KR SGD/common LR | **+0.000837** | [−0.004309,+0.005984]，5/8正，Holm p=.6953125 | 既不能确认为正，也不能证明 ±.003 实用等效 |
| KR Adam/fixed quota | −0.009191 | [−0.015795,−0.002588]，6/8负 | **次级 pointwise**；quota×q交互 −0.003988，区间跨0 |
| ML Adam/common LR、固定 subset | **−0.010392** | [−0.014599,−0.006185]，8/8负 | 固定 MovieLens 人口/配置的次级条件现象 |
| ML Adam/each-arm own LR、固定 subset | **−0.010875** | [−0.014447,−0.007303]，8/8负 | 与同组 **E100 +0.021145**、8/8正同时成立；SN .001 与 SB .0003 跨LR，不能叫纯 q 长期收益 |

**新机制缩窄而非全面识别：**
1. KR Adam 早期 AULC1..20 −0.007480，8/8负，t95 [−0.010031,−0.004930]；晚期同组 AULC1..100 −0.005324。故效应并非全由过拟合之后或选点伪象造成，但也非所有阶段、所有优化器都负。
2. **N2 固定θ四臂计数/方差分解**：16 锚的 SB/SN trace 比 IID 2.009–2.584、quota 1.995–2.551；域计数波动仅占所算 trace 约0.001297%–0.105936%。**“固定点协方差差异主要是域计数随机波动”在此可否定**；域内梯度结构仍贡献，但 quota 同时变批内依赖/高阶矩，长期交互未定，不证明纯 trace 中介。
3. **N3 共享路径独立四seed确认**：预先选择 embedding/E10，两个 full 局部路径 4/4正但效应仅约 +0.00003，方向门和实际阈值失败。长期 N4 topology 实验按预冻结条件 **NOT_TRIGGERED**；不等于“共享参数没作用”。
4. **评价权重可以翻转同一预测的数值符号**：历史 calAdam E50 macro −0.009074、训练 p 加权 cell AUC +0.001016、test-p 加权 +0.003108；新 ML own selected macro −0.010875，train-p +0.000744、test-p −0.013586。皆为**各 cell AUC 加权平均而非 pooled AUC**。不同宏/部署目标不可交换比较。
5. **ML own E100 正向不能升级方法**：宏 q gap +0.021145（8/8正，点区间正），但 E100 的较小 LR 与更长进度/selection 不同；较小域两任务的非劣保护置信下界 −0.005968、−0.009874，未达到 −0.003 门。保护未通过表示证据不足，**并非断言均值有损**。所有预定方法门仍 NO_GO。
6. **审计更正**：N2 原2项 float32 身份检查失败经同 seed、同 checkpoint、原估计器不变的验证脚本修复才纳入16有效锚；原失败保留。N17 主要 Q1–Q46 历史字段不改写，新增 Q47–Q53；N5-C 固定 ML subset_seed=2022，再用2032–39独立优化seed消除了 v3“seed同时改变subset”的部分混杂，**未**操纵第二个subset。
7. **论文方法门**：无已执行候选同时满足宏收益与七项保护；可说 **GO-EFFECT（指定设置/阶段）**、**GO-MECHANISM（仅固定点域计数贡献与有限局部作用）**、**NO_GO-METHOD（本轮）**，不可以从后者推出所有优化器/动量/MDMT研究无意义。

#### 3. 跨轮潜在冲突与最终裁决

| 历史说法/表面冲突 | 现在如何登记 |
| --- | --- |
| Stage2+ 曾写 0正式训练；Stage2D 说16/16 | **解决——阶段性执行状态覆盖**：同分支 16/16 成功，旧报告保留历史；不可将其计成“未实施”。 |
| 旧 S6-R2 KR test 非 target +.006757 四正；新 B1 +.000579 两正 | **真实差异，原因未识别**：初始化、完整 lineage、selected checkpoint 身份不同；不是同一模型的相互否定。 |
| Stage2D 强负向、v2 Adam CI跨0、N17 Adam主Holm未过、v3多条件不同方向 | **不同协议/估计对象 + 证据强度修订**：learned/fixed fusion、双/单 optimizer、LR、population、seed数、选择/时钟和确认统计都不同；保留局部负效应，不宣称“普遍稳定 q 损伤”。 |
| S5 Equal Adam-history 局部救援；v2/v3 无动量 SGD 也有 q 效应 | **命题不同**：SGD反例只否定 Adam 对“所有 q 相关损伤”的必要性，不否定 Equal 下历史 carrier 的局部条件效应；旧S5 m/v必要性仍未唯一识别。 |
| v3 C 局部 AUC 正、N17 N3绝对约3e-5/长期未触发 | **筛查对象/阈值不同**：早期 C pathway 有局部作用；联合 loss/J 或独立实际界不满足；共享长时必要性未证明，也未被反证。 |
| 标量 trace、谱 shape、Adam 均值更新哪个解释 q 效应？ | **进一步排除计数主导，但唯一中介仍开放**：固定θ均值/cov身份可证，参数分区trace可反号；N2计数份额小，不等于排除重复/coverage/依赖或识别 m/v/谱shape。 |
| ML own E100 正 +.021145、selected −.010875；calAdam 宏负/加权正 | **不是同指标矛盾**：时钟/选点/LR或评价权重定义不同；不以事后更有利的端点改主评价。 |
| “小域=困难域”“课程/动量一定有效”反复否定 | **非前提、非核心创新**：历史已排除从数据量直接推困难度；新轮只是记录适用范围，禁止以反复证伪该稻草人代替新科学问题。 |

#### 4. 问题去向：旧 OQ001–OQ010 与新 Q1–Q53 统一追踪

**机器可读完整清单：[Experimental_Question_Disposition_20261018.json](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)**。保留53条问题的历史状态、必要原结论、最后一次具体证据、未排除解释和最小后续判别；Q1–Q46 的源历史字段仍见 N17 branch 的 [QUESTION_REGISTRY_FINAL.json](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/QUESTION_REGISTRY_FINAL.json)。主文件**不**把“某一轮已执行完所有触发实验”误认为“所有机制问题已解决”。

| 旧主报告问题 | 截至 10-18 的裁决 | 新证据或仍缺什么 |
| --- | --- | --- |
| OQ001 Sample 极端域比例仍否有少数域损伤 | **开放** | V8 mask/物理复制与自然独立新样本量不等价；不能以无新增极端自然比例终局解决 |
| OQ002 size 独立因果，还是身份/任务/曝光代理 | **未识别；拒绝以大小直接定义难度** | N2 只排除所测固定点 count-fluctuation 主导 trace，不识别真实新增 unique N 的性能因果 |
| OQ003 S5 Equal persistent history 的 m/v/共享/架构组成 | **未识别，历史身份不足** | v3/N17 的**新 Sample q** shadow state 不是原 S5 Equal 同θ/同 lineage |
| OQ004 Equal access/history 跨模型/target/数据集范围 | **开放** | 后续主要是固定 J 的 q，未完成 Equal access 的跨 backbone 因果复制 |
| OQ005 ML late DH overfit vs optimizer-policy | **边界缩窄，唯一机制未识别** | S8 原 best 不改善；v3/N17 提供阶段/绝对健康反例，未把 old DH 的每条路径单独归因 |
| OQ006 Domain×Task gradient 结构跨设置稳定 | **部分机制描述，普适性开放** | 完整定θ谱/任务方向及局部块支持，缺稳定 performance 中介 |
| OQ007 训练/测试人口或标签漂移 vs objective | **人口权重算术已解决，域内 shift 因果开放** | N0/N17 三权重同预测反号，无法解释域内 split 的因果来源 |
| OQ008 S5-ML 不同复现来自数据/阶段/结构哪一项 | **开放** | 跨环境改变配置束，无法从 ML vs KR 自然观测剥离单因素 |
| OQ009 S6-R2 val/test 反转及精确协议再现 | **依旧未解释；重建未稳定重现** | B1 新模型/lineage 不同，已识别 selection 部分但非充分原因 |
| OQ010 S6-R2 参数置换 collapse 的定位 | **开放** | 原 swap 不能定位；V7/N3 是新状态局部控制，不是原 intact DH 分区唯一中介 |

**Q1–Q53 新增/修订的状态分层（截至最新证据）：**

- **已解决事实/审计身份（但不许扩大外推）：** Q15 同图/同目标公式及 q 校正、Q26 D3/SAF 目标边界、Q29 实际参数支持、Q31 调参/确认seed独立、Q32 固定SGD旧读数口径更正、Q33 common/own LR估计对象、Q39 同轨迹前缀身份、Q43 ML固定subset 复制、Q45 本轮全H100端点完成；Q3 的“Adam对一切效应必需”原命题为 FALSIFIED，Q34 moment telemetry 的旧修复保留。**已解决的是精确定义的局部命题，不是整个研究线全关闭。**
- **在已测设置内观察到效应、但推广或独立解释受限：** Q1–Q2、Q7–Q8、Q10–Q11、Q12–Q14、Q16–Q18、Q23–Q25、Q28、Q30、Q35–Q38、Q40–Q42、Q44、Q46–Q50、Q52。要逐条看完整清单的原始状态和最新边界，而不是按“BOUNDARY”统一叫未完成。
- **当前方案不能唯一识别（应转为明确新辨别研究，而不是反复复核同批数据）：** Q4（m/v/history必要性）、Q5（重复/曝光/覆盖/顺序分解）、Q6/Q22（共享长期必要性）、Q9（域/任务独立因果）、Q19–Q21（重叠支持与谱shape vs trace）、以及 Q51（Adam即时均值更新，**INCONCLUSIVE**）。
- **证据精度明确不足：** Q53 KR SGD共同LR selected差的t95 [−.004309,+.005984] 同时跨±.003，既不能宣告为零、也不能宣告实用提升/损伤；不要把“增seed追显著”作为机械下一步，除非先固定效应界与统计精度目标。
- **新Q47–Q53 全数保留：** Q47评价权重/人口，Q48调LR和预算，Q49计数与quota，Q50共享C局部路径，Q51 Adam瞬时漂移，Q52 EMS/MIS先例，Q53 SGD精度。N17 对它们的状态分别是 **5个 BOUNDARY、2个 INCONCLUSIVE**；“本轮执行完成”不意味着理论问题解决。

问题不能因“数量限制”自动关闭；新增问题/未识别项不设数额上限。“未触发”必须和“已实验否定”分开。任何新的机制研究应写清将从哪篇 MDMT 方法的哪项假设、迁移边界或尚未解决的矛盾切入，而不是再以早已不主张的“小域必然困难”作为研究动机。

#### 5. 当前能成立的最终结论及行动边界

1. **存在条件效应：** 同一 Sample 经验目标下，改变域采样 q（配以 p/q 校正）能改变批梯度高阶统计、优化路径和某些配置/阶段的 AUC/LL；统计与泛化范围取决于 LR、optimizer、fusion、dataset、物理人口、时钟及 validation selection，**没有普遍收益/损伤定律**。
2. **已证伪通用必要性，不等于确认单一替代中介：** 无动量 SGD 可以出现 q 差，故 Adam 不是普遍必要条件；但某些 Adam 状态响应存在，不能宣布 Adam 历史无作用或 S5 Equal 效应错误。固定点域计数波动不足以解释 trace 差；谱shape/域内噪声/重复/覆盖/长期 C 必要性尚未分离。
3. **不应以数据量代替困难度：** 三自然域的规模、输入、正例率、域任务分布、split、baseline 等高度共变，大小不是独立已识别原因。历史一开始就已拒绝“小域=困难域”；不应再把反复否定它算作新创新。
4. **评价是研究结论的一部分：** 报告必须绑定 macro vs train-p/test-p weighted cell AUC（均不是pooled AUC）、AUC vs LL、validation vs test、selected vs same-clock/E100/AULC、同q不同LR与独立seed；不能只看使单一方法显得有利的一组端点。
5. **方法化暂不成立：** 已测试静态 Balanced、历史 late rescue、局部 C/state 方案均不满足本项目的可用收益和全单元非劣/证据门。既有文献覆盖许多一般采样/优化机制；目前适合主张**条件边界与可证伪问题**，而不是新 MDMT 优化算法已获验证。**NO_GO 只针对本轮所测候选和当前证据**。
6. **最少必要的下一轮辨别**（不是本次合并自动新训）：优先预注册固定 ML 人口上的 **q×LR(.0003,.001) 完整2×2、独立 optimizer seed、相同外部进度和主要评价**，针对 ML own E100 +.021145 与 selected −.010875 这个最有研究价值的冲突；保留所有六cell和 LL保护，避免事后调主终点。是否进一步独立测 S5历史状态或共享C长期机制，必须以新的明确可区分反事实与可获得的原工件为前提，不做无止境的重复防御测试。

#### 6. 阅读路径和计数口径

1. 先读本文件（跨轮最终口径），再读 [Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)、[Stable Findings](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Stable_Findings.md)、[Evidence Ledger](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)、[Contradiction Register](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Contradiction_Register.md)。
2. 逐问题查 [53条问题去向 JSON](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json) 与 [Open Questions](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md)，识别“现象已证实 / 机制未识别 / 缺独立场景 / 科学NO / 条件未触发”。
3. 原始分支报告链接列于第1节；精确数字、置信区间、每seed和数据来源须回查各分支的 CSV/JSON/审计收据。**main 本次只汇总文字与提取后的问题清单，不复制所有原始数据、所有训练提交、状态 checkpoint 或试验算法。**
4. 新 EXP 编号作为**研究阶段索引**而非把每个派生分析都算新独立实验。B1、Stage2D、v2、v3、N17 之间的祖先共享、读取旧模型和派生统计不得重复加总；16/16 Stage2D、v2已核验子集、v3规定训练和 N17 的120条新H100分别按**自己协议**报告。

## 十月二十六日跨分支综合

[该记录历史版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md)

> **汇总截止：2026-10-26；性质：main 中的跨阶段科研事实与结论索引。** 本文整合新归档的实验分支与最新研究方案，不修改各阶段原始报告，也不把方案写成已运行实验。原始数值、配置和源码以归档文件为准。此前 [2026-10-18 综合报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)是截至当时的历史快照，已不是当前结论入口。

#### 1. 研究主线、证据身份与解释纪律

长期主线是 **MDMT推荐中的样本重加权、优化响应与差异化共享**：研究不同域/任务/目标群体的真实知识收益、局部共享需要及负迁移。Equal/Sample是某些实验中的具体损失聚合口径，不能替代整个长期研究主题。

- **分清执行状态：** COMPLETED表示已触发的规定工作完成；NO_START/NOT_TRIGGERED表示前置条件没有通过，不是运行后的阴性；PARTIAL/IN_PROGRESS只能使用已经核验的子集；DESIGN_ONLY不能宣称算法性能。
- **分清四层证据：** 数据/代码身份、训练与预测观测、可识别的受控干预、尚待验证的解释或技术假设。非零梯度、非恒定权重、可生长原型和激活路径本身不等于独立方法增益。
- **分清估计对象：** 自然/人工训练人口、目标与采样q、参数图及优化器、训练时钟、selected与最终epoch、六格macro与按人口加权cell AUC、pooled AUC、LogLoss及测试/验证/Q角色，均不得擅自交换。
- [history_experiments](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/README.md)保存各原分支相对归档前main不同的Git文件对象与相对路径，不是每个分支独立完整工作树；引用阶段事实时核对原提交SHA和原报告，不跨阶段相加共享训练/前缀。

#### 2. 贯穿2026年9—10月的实验时间线

| 时间/轮次 | 原始报告或方案 | 已取得结果/当前状态 | 不能外推的部分 |
|---|---|---|---|
| 9月—10-09 P0–P7、E、S0–S8/S6-R2 | [历史索引](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Experiment_Index.md)、[旧证据](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md) | 样本聚合和共享更新权限的条件效应；Equal下S5特定Adam history承载；S8 late rescue不提高合法selected best | 不同客观目标/协议的结果不可合并成普遍“Sample优于Equal/Adam是根因” |
| 10-15—16 B1/C2与Stage2D | [B1与审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-16-v2/research_plans/2026-10-15/execution_20261015/README.md) | B1重建8/8完成，但与旧S6-R2初始化谱系不同；Stage2D 16/16新训练完成，覆盖早期“零正式训练”的**过程快照** | 旧S6-R2非目标增益与新B1不能当同协议互相推翻 |
| 10-16—17 v2/v3 | [v3终报](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md) | v2固定fusion的无动量SGD仍有采样效应，驳斥Adam普遍必要性；v3规定V0–V14及触发项完成、当前方法NO_GO | v2保留中期范围；v3与v2为不同Git实验谱系，不互称完整复现 |
| 10-17—18 N17 | [FINAL_ANALYSIS](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | 120条新H100、72条唯一formal。KR Adam selected SB−SN宏AUC −0.005203（Holm p=.171875）、SGD +0.000837（p=.6953125），两个预定主比较未确认；ML ownLR E100 +0.021145而selected −0.010875 | ML固定人为subset，LR/时钟和选点不可混称纯q效果；N4未触发 |
| 10-18 N18 P1–P6 | [六方向首轮](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md) | 数学反例、数据与预算审计、四种子冻结头及有界诊断完成；尚无六方向中稳定且具独立增量的方法候选 | DATA-BLOCKED不代表科学方向已失败；非触发长期训练不得记作阴性训练 |
| 10-18—19 N19 | [N19完整终报](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md) | 328条完整正式网络训练及冻结测试；报告容量路径数值弱化、通道误设、指标/阶段反转和受控机会；未支持通用新方法贡献 | MovieLens此轮主要是人工固定subset，不是自然full人口 |
| 10-19—20 CM19 | [共享路径终报](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md) | 856次全预算训练及对应冻结测试。KR fixed M3oE某个输入乘性调制路径selected 24/24 cell数值中性，双factor耦合Adam-L2或局部AdamW干预能恢复路径响应，selected泛化收益未稳定建立 | 不等于全部共享参数失活；独立STAR、PEPNet、HAMUR及MMoE/PLE相关路径仍活跃；恢复活性不等于模型必优 |
| 10-20 自然MovieLens历史复核 | [人口审计](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)、[已完成子项重评](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/RESEARCH_CONCLUSION_REASSESSMENT.md) | 审计确认多项10-15以来ML实验为人为187500行/34500:3000:150000，自然base-train为800167行/155949:318599:325619；已完成V2/S5/S6/AO04、N17和若干局部机制比较，原部分selected/late结论须限制人口 | 该分支 [最后状态](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)仍IN_PROGRESS，**N19/CM19全套自然重训未完成**，不得当作完整复现 |
| 10-21 原生M3oE、数据生成/重加权、调幅 | [48次正式训练](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md) | ML D−B macro AUC +0.006303、LL −0.015962（4/4有利）；Dconst01−B AUC +0.005676（4/4有利），动态样本specificity不是获得套件收益的必要条件；匹配路径G−Gtau0 AUC ML −0.003466、KR −0.006484（均4/4负，KR区间跨0） | D含权重、复制、噪声、总尺度；原生O是受控移植；目标为legacy_group_sum，不能与标准Sample目标直接混比 |
| 10-24 I1权限 | [阶段报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/10-24/research_reports/2026-10-24/README.md) | 44主pilot、4基线适配pilot、40正式训练；I1_PERM比固定RW的ML/KR selected test宏AUC −0.000334/−0.000513，未过主优势 | KR final LL次要改善不替代selected主结论 |
| 10-25 MDMT-NEXT-v2 | [v2报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-25/research_reports/2026-10-25/README.md) | 48条24epoch核心轨迹完成；晋级门均未过，正式四seed/50epoch为NO_START；部分aux/基线预算停止 | 不把未触发的正式训练当作实际阴性实验 |
| 10-25 MDMT-NEXT-v3（RHD/LPS） | [v3完整报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md) | 69条24epoch逻辑轨迹＋20条KR正式50epoch。RHD HB−BASE：selected test LL +0.00074075（差），epoch50 −0.00875428（好）；LPS部分foreign可读取，但特定形成器未稳定超过简单控制 | late减损≠最佳selected改善；测试和旧确认Q历史已暴露 |
| 10-26 DFD-v4 | [完整执行报告](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md) | ML P1需求选择有限优于常量（CONST−DEMAND LL +0.00108935）；KR P1未过。ML P2 DFD−H_DIRECT/+0.00163966、DFD−ETA_DIRECT/+0.00181402，完整方法不支持；184份冻结预测导出、12项推理测量完成 | KR P2、P3/P4按门NO_START；原Q/outer refitη固定μ分组缺失MAJOR，不能称所有机制收据齐全 |
| 10-26 研究重设计RCF | [RCF-v1方案](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)、[事实审查](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/01_实验事实与科学问题.md) | **DESIGN_ONLY**：在来源行为保留/预算约束下，让目标群体剩余风险指导来源后继残差函数形成；配强直接、固定来源读、统一微调与target-only对照 | 没有新的RCF训练或结果；不能将新方向的理论计划写成已被实验验证 |

> 各行训练数属于不同实验定义。祖先/后继分支、复用的前缀与预测、相关cell及fold均不能被当成独立新训练或独立样本数。

#### 3. 早期说法应如何更正

| 早期或过强说法 | 目前可使用的最强陈述 | 状态 |
|---|---|---|
| “10-15后MovieLens都是自然比例，结论可外推原人口” | 多轮用了明确人为角色采样，旧实验仍有效但只是条件结论；自然全量实验同时改变N、支持、相对比例和时钟，不能归因单一规模变量 | **修订人口前提** |
| “Equal普遍伤大域，Sample在所有场景更好并消除共享损伤” | Equal/Sample是不同优化目标；S_n指定配置中Sample缓解大域损害是真实条件结果，不能上升为普遍目标优劣，也不能代替长期样本重加权主线 | **局部保留，普遍性撤回** |
| “Adam历史是所有采样/共享效应根因”或“Sample使Adam历史完全没用” | 无动量SGD仍出现q效应，故Adam**不具通用必要性**；历史Equal S5的特定persistent history承载仍成立，不能由SGD反例把它整体删掉 | **必要性被驳，局部中介开放** |
| “小域=困难域”或“反复推翻小域=困难域是新贡献” | 这原本就不是研究者认可的前提；规模、物理域身份、标签/特征支持、基准性能不等同，仍未识别域规模独立因果 | **避免虚构起点** |
| “仅大域更新可稳定提高小域/共享路径恢复即收益” | S6-R2阳性、B1及自然复评异向且不同谱系；CM19恢复活性并未一致改善selected指标，其他域/任务可能承担代价 | **缩限且保留冲突** |
| “匹配梯度范数就可复现共享救援” | S4 raw/applied范数匹配均不足；10-21调幅匹配路径在本协议下AUC负向，不能推广为所有梯度调节无价值 | **通用机制未支持** |
| “动态权重D获得增益，就证明样本级区分有效” | ML D、Dconst01相对B均有益，D权重很快接近下界；常数方案尚未被证明不劣于D，复制/噪声/总尺度也未独立拆分 | **性能事实保留；归因未建立** |
| “原型/历史控制非零说明有共享功能分工” | N18/N19、10-24、RHD/LPS及DFD仅对各自当前版本给出NO_GO或有限边界；foreign可读与功能需要形成是不同命题。RCF仍是待验证方案 | **版本NO_GO，不是方向否定** |
| “最终epoch相对好就说明最优可部署模型更好” | S8、N17、D/G、RHD的selected/final/AUC/LL/角色可反号，需优先合法选点和逐格保护 | **评价口径修正** |
| “10-16仍没有正式实验”或“10-20自然复核已全数闭环” | Stage2D后续16/16完成；10-20自然矩阵有大量已完成子项但总项目标记IN_PROGRESS，N19/CM19自然正式复制不可虚报 | **按最终状态分别保留** |

**数据角色补充：** 10-25 v3 / 10-26 DFD在自然base-train内另外分割fit及P_form/P_read/P_cal等角色；DFD的ML/KR fit分别为640133/1604904，不等于整个自然base-train的800167/2006132。原始M3oE的learned-fusion双Adam与fixed-Sample单Adam也不是可直接相减的模型。

#### 4. 当前可信的综合科研认识

1. **样本重加权/梯度/优化：** 它们的作用确实可能随目标、population、优化器、学习率和训练阶段而变化；当前尚无证据支持一个通用的“增大小域权重”或“动量修复”的安全算法。10-21套件的正结果表明仍有可深入的尺度、增强与权重交互，但不能把套件收益直接归因某个动态权重函数。
2. **共享机制：** 部分M3oE乘性路径可在特定条件下数值中性，存在设计/优化器相关的恢复；并非所有结构都会失活，也不意味着激活恢复就能提升性能。普通foreign读取的有限价值与特定来源功能形成的必要性需分开判定。
3. **局部需求驱动共享：** ML中目标条件选读相对常量有信号，KR未过，完整DFD未超过强直接控制；需求外推及正权形成机制缺口已清楚，因而可以保留“目标真实剩余需求—来源可用函数”的问题，并用RCF的强对照检验新增价值，而不是宣称其已有效。
4. **评价与科研证据：** 逐格AUC/LL、宏与不同部署人口、selected与末态、validation与历史exposed-test必须并存。当前大部分正式测试沿用已暴露的人口，独立新时间/真实样本支持才有助于升级外部效度。

#### 5. 后续最有判别力的工作（不是已执行结果）

主问题是**强目标直接头已可预测后，来源行为监督形成的函数是否仍提供有限样本/容量/预算下的独立增量**。按照 [RCF-v1规范](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_plans/2026-10-26-v2/00_README.md)先比较完整来源读取、H_DIRECT/ETA_DIRECT、目标专有/统一微调和固定来源，再决定是否值得目标群体组织多个来源后继残差；同时设置来源保持约束并检查真实成本。RCF尚未运行。

样本权重支线保留最小对照：10-21建议固定loss尺度0.1且关闭复制与噪声，以拆分全局scale与增强的收益，再判断动态specificity是否有必要；不要用未运行的“理论有用”填实验结论。

#### 6. 阅读顺序与历史保护

1. 本文件及 [Global Summary](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md)是截至2026-10-26的跨轮入口；[Experiment Index](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Experiment_Index.md)标记最新阶段；[Evidence Ledger](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md)记录证据边界。
2. [Stable Findings](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Stable_Findings.md)、[Contradiction Register](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Contradiction_Register.md)、[Hypothesis History](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Hypothesis_History.md)、[Open Questions](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md)、[Dead Ends](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Dead_Ends.md)保留历史编号并标出后期限制。
3. [2026-10-18旧综合](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md)及[旧Q1–Q53](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json)是日期固定的历史记录。新证据不可回填改写它们。
4. 全部阶段原文、不同实验分支的代码与表格在 [history_experiments](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/history_experiments/README.md)中；归档内容和各阶段原始报告不做编辑。

## 研究分支覆盖复审

下列为迁移前远端可见的 47 个非 main 分支。按各分支头和结果目录核对；超过比较接口文件上限的分支补查研究报告/方案树。表中 Git 状态描述提交关系，科学状态描述能够用于设计的实际证据。继承的报告、重放、相同数据的再分析不重复计数。旧归档结果保留原协议。

| 原分支 | 与迁移前 main 的关系 | 结果在上下文中的位置 |
| --- | --- | --- |
| [`10-24`](https://github.com/Caelestis1903/My-Research-Project/tree/b64b29222fcd36ca8ecbb323ccb127f7da1ccd57) | 分叉 | EXP048 |
| [`2026-10-25_15-56-04`](https://github.com/Caelestis1903/My-Research-Project/tree/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0) | 分叉 | EXP048–050 |
| [`26-10-25`](https://github.com/Caelestis1903/My-Research-Project/tree/619371d9c4d57f6f0b3e8a71b686498f9e5f36d4) | 分叉 | EXP048–049 |
| [`26-10-26`](https://github.com/Caelestis1903/My-Research-Project/tree/dc32f60a6aafce98a25838a485e02cead5425fc6) | 已在 main 历史 | EXP051 |
| [`26-10-27`](https://github.com/Caelestis1903/My-Research-Project/tree/03c338092008a71b2d360a61b06d985eac551ec7) | 分叉 | EXP052 |
| [`26-10-28`](https://github.com/Caelestis1903/My-Research-Project/tree/97151c74544c3651feb805d1898bc9ce1b8d12d3) | 分叉 | EXP052–053 |
| [`26-10-31`](https://github.com/Caelestis1903/My-Research-Project/tree/ee6e81fdfd37284e6447e8149cb93d04736dba02) | 分叉 | EXP055 |
| [`26-11-01`](https://github.com/Caelestis1903/My-Research-Project/tree/a6f496b54f467cdf7b814b6bb8defdbdd457bc38) | 分叉 | EXP056 |
| [`26-11-01_v2`](https://github.com/Caelestis1903/My-Research-Project/tree/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b) | 分叉 | EXP057 |
| [`26-11-01_v3`](https://github.com/Caelestis1903/My-Research-Project/tree/6f5742c2085c1e49d1767b0a7079bb9f28160eda) | 分叉 | EXP057–058 |
| [`26-11-02`](https://github.com/Caelestis1903/My-Research-Project/tree/e59efa2fd954c4d2231b679a742406cfbe2a1077) | 分叉 | EXP057–059 |
| [`26-11-02_v2`](https://github.com/Caelestis1903/My-Research-Project/tree/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27) | 分叉 | EXP060 |
| [`26-11-02_v3`](https://github.com/Caelestis1903/My-Research-Project/tree/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288) | 分叉 | EXP065 |
| [`26-11-03_history_information`](https://github.com/Caelestis1903/My-Research-Project/tree/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72) | 分叉 | EXP066 |
| [`26-11-06`](https://github.com/Caelestis1903/My-Research-Project/tree/40705dfe0bb4fe614bdca835c0bae567348c87d0) | 分叉 | EXP068–070 |
| [`26-11-07`](https://github.com/Caelestis1903/My-Research-Project/tree/ffa3de067fcda5a1293af83a465d741e069493f1) | 分叉 | EXP068–071 |
| [`26-11-09`](https://github.com/Caelestis1903/My-Research-Project/tree/1391affa5f08556c79b4cb2bfad15228a8b9129f) | 分叉 | EXP061、EXP068–071 |
| [`26-11-09_v2`](https://github.com/Caelestis1903/My-Research-Project/tree/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8) | 分叉 | EXP062 |
| [`26-11-11`](https://github.com/Caelestis1903/My-Research-Project/tree/fb120946c50519ea3993bbd6a2532a2c457731fc) | 分叉 | EXP063–064 |
| [`26-11-15`](https://github.com/Caelestis1903/My-Research-Project/tree/52b48c9aab066d56fa91a060864c7c9abf12e50c) | 分叉 | EXP072 正式登记原始 HAT 六臂；旧 NDP-A 已引用它，逐 seed/六格证据补入 |
| [`26-11-23`](https://github.com/Caelestis1903/My-Research-Project/tree/9df48737dd6693f892b85db2344cdee76fb05d6f) | 同一提交 | EXP080 |
| [`baseline/mdmt-sota-2526`](https://github.com/Caelestis1903/My-Research-Project/tree/d2fdd272a2cdf7da626e29b97f61555d25271f89) | 分叉 | EXP052–054 |
| [`codex/2026-10-16-v2-research-plan`](https://github.com/Caelestis1903/My-Research-Project/tree/04ea96173b6302f5a4d7fd436add644b57629e7c) | 分叉 | v2 方案，该分支增量无独立性能结果 |
| [`codex/add-mdmt-frontier-plan-2026-10-18`](https://github.com/Caelestis1903/My-Research-Project/tree/9f6d89683b4a96d3b9cc5e493f8eadf325aa65df) | 分叉 | 研究前沿计划，该分支增量无独立性能结果 |
| [`codex/add-research-plans-2026-10-17`](https://github.com/Caelestis1903/My-Research-Project/tree/7a7cf54313ea41bb40f4099e16b60e0d4d777999) | 分叉 | 已归档前序证据的分析与后续计划，该分支增量无独立性能结果 |
| [`codex/conditional-paths-2026-10-19`](https://github.com/Caelestis1903/My-Research-Project/tree/eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c) | 分叉 | EXP045 |
| [`codex/execute-2026-10-15`](https://github.com/Caelestis1903/My-Research-Project/tree/b17cb2707fc2448f09f21fd12279b2b3856c7756) | 分叉 | EXP033–034 |
| [`codex/execute-2026-10-16`](https://github.com/Caelestis1903/My-Research-Project/tree/aa39e7b27285985bae9fa9a093f16c6d9634989b) | 分叉 | EXP033–035 |
| [`codex/execute-2026-10-16-v2`](https://github.com/Caelestis1903/My-Research-Project/tree/c31b531e210daab4cbfb2e53eaa44c22b913264b) | 分叉 | EXP033–036 |
| [`codex/execute-2026-10-16-v3`](https://github.com/Caelestis1903/My-Research-Project/tree/2d04c6de6c925eca8563f81b66af3f0e95f99183) | 分叉 | EXP037 |
| [`codex/execute-2026-10-17-run`](https://github.com/Caelestis1903/My-Research-Project/tree/5550afbc57f76e0b94c5783db16f4c1796933541) | 分叉 | EXP037–042 |
| [`codex/execute-2026-10-18`](https://github.com/Caelestis1903/My-Research-Project/tree/14cf09a3865303f5e451636dc6407fd0e2cc1b8c) | 分叉 | EXP043 |
| [`codex/execute-n19-2026-10-18`](https://github.com/Caelestis1903/My-Research-Project/tree/c440350681a7a3e41ab3f05eb1c4d31f419b989f) | 分叉 | EXP043–044 |
| [`codex/m3oe-reproduction-2026-10-21`](https://github.com/Caelestis1903/My-Research-Project/tree/c5618e281785a4b7063ff9bf2a61cf38d4faf1be) | 分叉 | EXP047 |
| [`codex/natural-movielens-reassessment-2026-10-20`](https://github.com/Caelestis1903/My-Research-Project/tree/f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e) | 分叉 | EXP046（已完成部分；自然 transfer 未闭环） |
| [`codex/research-plan-2026-10-16-v3`](https://github.com/Caelestis1903/My-Research-Project/tree/0698fe31ea752811b1d71f0f2bb343679d9a5685) | 分叉 | V0–V13 方案，该分支增量无独立性能结果 |
| [`docs/mdmt-ap-codex-plan-20261027`](https://github.com/Caelestis1903/My-Research-Project/tree/b176d6aa6618a51f2e615aeb9d1981fec9fd820b) | 已在 main 历史 | MDMT-AP 方案，该分支增量无独立性能结果 |
| [`planning/ga31-v2-reassessment`](https://github.com/Caelestis1903/My-Research-Project/tree/d857e4cba05d1c64976cfb88668418448d78a22a) | 分叉 | GA31 再评估方案，该分支增量无独立性能结果 |
| [`research-plan/2026-10-19-modulation-path-generalization`](https://github.com/Caelestis1903/My-Research-Project/tree/3bd84f10971791a264e598bcb4b8114aea28bbe7) | 分叉 | 共享调制方案，该分支增量无独立性能结果 |
| [`research-plan/2026-10-26-v2-mechanisms`](https://github.com/Caelestis1903/My-Research-Project/tree/e9e14ab9643131dc30292e8f28bcc12f0a4c4be8) | 分叉 | EXP081 补入实际执行的 RCF 与 SCALE；取代旧索引 DESIGN_ONLY 状态 |
| [`research-plan/26-10-27-v2`](https://github.com/Caelestis1903/My-Research-Project/tree/ed94caa82761d4b9e1ddb8c5cda454e75e6eeed3) | 分叉 | Relation-Basis 实施方案，该分支增量无独立性能结果 |
| [`research-plan/26-11-03-mdmt-history-information`](https://github.com/Caelestis1903/My-Research-Project/tree/75f698d1aff221adee2b6c2d0a9e59ba9c1c7e64) | 已在 main 历史 | 条件信息和历史方案，该分支增量无独立性能结果 |
| [`research-plan/26-11-03-overnight-ab-auc`](https://github.com/Caelestis1903/My-Research-Project/tree/9e682f3e3f608c7d63c2e9306b8657b16d822541) | 分叉 | EXP068 |
| [`research-plan/26-11-2-v3-conditional-information-momentum`](https://github.com/Caelestis1903/My-Research-Project/tree/2c4ab338798e4bd98d3263bfd719b14345edc27e) | 已在 main 历史 | PAIR/ECCM 方案，该分支增量无独立性能结果 |
| [`research/26-11-01-gradient-augmentation-momentum-plan`](https://github.com/Caelestis1903/My-Research-Project/tree/f2444e72f93a1955a3dfb066801473855890ac65) | 分叉 | 梯度增强与动量方案，该分支增量无独立性能结果 |
| [`research/26-11-04-gaussian-domain-coverage`](https://github.com/Caelestis1903/My-Research-Project/tree/c8675ae2923c15b6965884b140b5b646690cd95e) | 分叉 | EXP068–069 |
| [`review/ga31-r3-codex-ready-20261031`](https://github.com/Caelestis1903/My-Research-Project/tree/2f5f30589da9e91bd0396b029984cf50c14f4401) | 已在 main 历史 | GA31 审查分支，该分支增量无独立性能结果 |

RCF 的冻结效果、作用域和移交见 [EXP081](#exp081)。早期分析中的 DESIGN_ONLY 描述的是当时快照。HAT 的实际六臂先导及对照见 [EXP072](#exp072)。

## 迁移复审与旧路径索引

源主线为 `9df48737dd6693f892b85db2344cdee76fb05d6f`。复审以 Git blob、旧 EXP001–EXP064、53 个独立问题的原字段、历史稳定事实/矛盾/证据记录及 11 月最新判定逐项对应。下表链接为版本化溯源；历史报告中的当时操作指令、未执行设想与原计数仍按各自快照解释。旧版重复记录使用单一当前入口，差异与可核验原字段沿来源提交回溯。

| 旧入口 | 本次迁移位置 | 源快照 |
| --- | --- | --- |
| `EXP_AI_CONTEXT.md` | [01_AI_EXPERIMENT_INDEX.md](01_AI_EXPERIMENT_INDEX.md)、[历史交接摘录](#历史交接摘录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/EXP_AI_CONTEXT.md) |
| `MDMT_ALGORITHM_METRICS_LEDGER.md` | [03_EXPERIMENT_RESULTS.md](03_EXPERIMENT_RESULTS.md#历次性能数表) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/MDMT_ALGORITHM_METRICS_LEDGER.md) |
| `experiments/Experimental_Branch_Synthesis_20261018.md` | [十月上旬至十八日分支审查](#十月上旬至十八日分支审查) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261018.md) |
| `experiments/Experimental_Branch_Synthesis_20261026.md` | [十月二十六日跨分支综合](#十月二十六日跨分支综合) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Branch_Synthesis_20261026.md) |
| `experiments/Experimental_Contradiction_Register.md` | [冲突登记](#冲突登记) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Contradiction_Register.md) |
| `experiments/Experimental_Dead_Ends.md` | [已停止、已降级与未启动](#已停止已降级与未启动) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Dead_Ends.md) |
| `experiments/Experimental_Evidence_Ledger.md` | [证据记录 E001–E078](#证据记录-e001e078) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Evidence_Ledger.md) |
| `experiments/Experimental_Experiment_Index.md` | [01_AI_EXPERIMENT_INDEX.md](01_AI_EXPERIMENT_INDEX.md)、[EXP001–019](#exp001-019) 至 [EXP061–064](#exp061-064) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Experiment_Index.md) |
| `experiments/Experimental_Global_Summary.md` | [历史跨阶段总裁定](#历史跨阶段总裁定) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Global_Summary.md) |
| `experiments/Experimental_Hypothesis_History.md` | [历史假设及修订](#历史假设及修订) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Hypothesis_History.md) |
| `experiments/Experimental_Open_Questions.md` | [历史问题及决策边界](#历史问题及决策边界) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Open_Questions.md) |
| `experiments/Experimental_Question_Disposition_20261018.json` | [Q1–Q53 逐项结论与原字段](#q1q53-问题去向) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Question_Disposition_20261018.json) |
| `experiments/Experimental_Stable_Findings.md` | [稳定事实 SF001–SF056 及后续记录](#稳定事实-sf001sf056-及后续记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/Experimental_Stable_Findings.md) |
| `research_context/experiments/Experimental_Contradiction_Register.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Contradiction_Register.md) |
| `research_context/experiments/Experimental_Dead_Ends.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Dead_Ends.md) |
| `research_context/experiments/Experimental_Evidence_Ledger.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Evidence_Ledger.md) |
| `research_context/experiments/Experimental_Experiment_Index.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Experiment_Index.md) |
| `research_context/experiments/Experimental_Global_Summary.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Global_Summary.md) |
| `research_context/experiments/Experimental_Hypothesis_History.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Hypothesis_History.md) |
| `research_context/experiments/Experimental_Open_Questions.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Open_Questions.md) |
| `research_context/experiments/Experimental_Stable_Findings.md` | 同名记录章节、[快照独有的 C05 记录](#快照独有的-c05-记录) | [原版本](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_context/experiments/Experimental_Stable_Findings.md) |
