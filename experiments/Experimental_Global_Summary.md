# Experimental Global Summary


> **2026-10-26 总结更新：** 本文件的当前科研裁定以 [2026-10-26跨阶段总报告](Experimental_Branch_Synthesis_20261026.md) 为准；后文标为10-18/10-09的Q1–Q7、训练计数及“当前”陈述均保留为当时的历史快照，不能覆盖晚于它的证据。原研究阶段报告未修改，原始文件与固定SHA见 [history_experiments](../history_experiments/README.md)。

## 2026-10-26 当前研究结论（优先于下方历史快照）

1. **长期研究轴是样本重加权与差异化共享。** Equal/Sample是训练目标的局部口径；S5 Equal的特定Adam-history作用、v2/N17无动量SGD的采样效应并存：Adam不是全体效应的必要条件，也没有被Sample“完全消除”。研究仍要分开样本曝光、目标加权、动量状态和共享读写权限。
2. **先修正数据人口，再谈相互印证。** 10-15之后多轮MovieLens在人为固定子集187500行（34500/3000/150000）上训练；自然base-train为800167行（155949/318599/325619）。[自然人口审计](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)显示比例、真实支持与训练时钟一起变化。自然复核已经有部分可靠结果，但扩展分支整体仍为IN_PROGRESS，不能把N19/CM19人工结果宣称为全自然重复。
3. **N18/N19已经实质完成，不是只有“方案”或都没有尝试。** [N18](../history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)进行了六方向有界诊断；[N19](../history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)完成328次全预算训练。方法独有性和稳定强对照收益未获支持，但局部反例、容量/通道边界和方向机会保留，不把NO_GO扩展为领域失败。
4. **共享路径的数值功能与实际泛化收益不同。** [CM19](../history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)完成856次训练：KR fixed M3oE选点的输入乘性调制路径有明确数值中性，局部AdamW/耦合L2干预可以恢复响应；STAR等独立路径保持活跃，且恢复响应并未产生通用selected收益。
5. **10-21的D套件在自然MovieLens确有收益，但“动态样本区分导致收益”尚未成立。** [复现报告](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)中D−B test宏AUC +0.006303、LL −0.015962（4/4），Dconst01−B AUC +0.005676（4/4）；常数权重仍带复制/噪声/总尺度，不能从这组对照确定哪个模块有效。匹配G−Gtau0调幅的宏AUC：ML −0.003466、KR −0.006484（均4/4负，但KR区间跨零）。其目标为legacy_group_sum，非其他阶段的Sample/Equal。
6. **动态权限/收益/形成新方法须通过强简单对照。** 10-24 I1_PERM在ML/KR selected未胜固定RW；10-25 v2核心48条完成但正式门未触发；10-25 v3 RHD相对BASE selected test LL +0.00074075、e50 −0.00875428，LPS特定形成未稳定优于简单读法；10-26 DFD-v4的ML需求预测有限有利，但完整DFD相对H_DIRECT/ETA_DIRECT LL更差、KR P1失败，P3/P4未触发，另缺原refitη外层μ分组。各轮**版本性结论**不等于来源知识价值或多原型研究全盘不成立。
7. **最新研究决策是RCF设计而非新实验阳性。** [2026-10-26-v2方案](../research_plans/2026-10-26-v2/00_README.md)保留目标剩余风险指导来源后继函数形成的科学问题，以来源保持、有限容量、固定读出及强目标直接头作判别；目前**DESIGN_ONLY，无新增RCF训练/评价结果**。
8. **评价主张必须绑定协议。** 宏cell AUC、人口加权cell AUC、pooled AUC、LL，selected、末轮、长期AULC、Q/validation/test与真正新封存数据互不等价。早期“小域=困难域”从来不是用户认同的前提，不再把重复推翻它算作研究成果。

**合并结论：** 现有证据支持目标/域/群体条件性收益与共享损害的研究价值，支持一些真实局部机制与反例；没有建立一个在当前完整强对照和不同自然支持下通用、安全、独有的新算法。更值得检验的是**强目标预测之外的来源功能增量、群体需求驱动形成，以及样本权重收益的真实贡献结构**。完整时间线、已完成/未触发/部分完成和历史结论更正见[跨阶段总报告](Experimental_Branch_Synthesis_20261026.md)。

---



> **2026-10-18 跨分支更新。** 下方原“整理截止 2026-10-09”以及 Q1–Q7/历史计数保持**历史快照**身份，不再代表最新所有实验的最终统计或执行状态。当前权威入口：[多分支综合报告](Experimental_Branch_Synthesis_20261018.md)；[53条问题逐项去向](Experimental_Question_Disposition_20261018.json)。证据编号 E033–E041、SF013–SF020 和新增 EXP 索引补录见相应文件。**未把代码/权重/原始数据从执行分支全量合并。**

## 2026-10-18 历史科研摘要（优先于当时10-09快照；已由10-26总报告更新）

1. **同 Sample 目标不等于同训练路径。** KR Stage2D（learned fusion、双 Adam、四seed）validation-selected SB−SN macro AUC = −0.010406、4/4负；v2（fixed fusion、单优化器）无动量SGD仍为 −0.002104、4/4负。两组不构成协议同构的直接复现，却共同说明 Adam **不是所有 q 相关效应的通用必要条件**；不能从此反驳旧 S5 Equal 下持久 history carrier 的窄结论。
2. **最新 N17 的统计强度低于简单符号多数投票。** 120条新H100轨迹全部完成；正式 KR selected test：Adam SB−SN = −0.005203，8seed t95[−0.012006,+0.001600]、Holm p=.171875；SGD = +0.000837，t95[−0.004309,+0.005984]、Holm p=.6953125。两项**预定主比较均未确认**，不能称普遍损伤或等效。KR Adam AULC1..20 −0.007480、8/8负是**阶段性次级证据**，不是取代主门。
3. **晚期、选点、LR与评价权重会真实改变方向。** 固定 ML subset 的 own-LR E100 q-gap +0.021145、8/8正，但 selected −0.010875、8/8负；两臂采用不同 LR，不能称纯 q 优势，且cell保护门未过。同一个历史 calAdam E50 预测：宏cell AUC差 −0.009074、train-p加权 +0.001016、test-p加权 +0.003108（均不是 pooled AUC）。
4. **机制从“可能”缩窄为“局部反证与仍未识别”。** N2固定点 quota 计数项只占目标梯度trace约0.001297%–0.105936%，不支持“域计数波动主导”的特定解释；quota长期效应仍存在但交互区间跨0。N3局部共享路径约 +3e−5的阳性达不到独立方向/实用门，N4长期拓扑实验**未触发**；独立 shape vs trace、m/v/history、重复/coverage与共享 C 长期中介仍未唯一识别。
5. **研究主线不再围绕反复否定“小域=困难域”。** 这原本就不是成立前提。数据量、身份、标签/特征、域任务与曝光共同变化；未做真实独立数据支持 N 的纯因果操纵。当前结论是**条件效应有证据、普适/唯一中介不足、已测试候选方法 NO_GO**；不预设动量/课程有效，不从未触发分支推断方法阴性。

**跨时间纠正：** 10-16 早期 Stage2+ “正式训练 0、后续阻塞”已经被同分支 Stage2D **16/16 COMPLETE** 的后续报告覆盖。旧 S6-R2 KR test non-target +0.006757（四正）与10-15 B1重建 +0.000579（两正）来自不同init/lineage，不是同协议相互否定。v2是中期研究（不能声称全部完成）；v3在 Git 历史上与 v2 发散；N17继承v3。三个方案分支的原计划正文已经在 main，不额外计数实验。

**问题状态：** 旧 OQ001–OQ010 均逐项再评估（一些计数/身份问题已排除，一些机制依然开放）；N17 保留 Q1–Q46历史对象并新增Q47–Q53，完整53条见链接。若将“已执行/未触发”直接等同于“已解决/已否定”，会误报；也不限制新增开放问题的数量。

---

整理截止：2026-10-09。本总结把已复核的实验事实与未验证的解释分开；证据强度和适用范围以 [Evidence Ledger](Experimental_Evidence_Ledger.md)、[Hypothesis History](Experimental_Hypothesis_History.md) 及 [Open Questions](Experimental_Open_Questions.md) 为准。

## Q1 — 目前最强的实验事实是什么？

1. **已测自然比例下，Sample joint training 没有显示稳定的实用负迁移。** S1 的 72 个已测配对单元中为 0 个 practical NT；S2 的 Sample 对照为 0/24。S6 预注册 selected-target gates 也未达到 practical rescue 阈值。这是对已测协议的结论，不覆盖未测的极端比例或目标域设置（E009、E010、E018、E019）。
2. **Equal/KuaiRand/tab1 下的 target 结果对 shared-access/history 干预有条件性响应。** S3 access 对照 4/4 seeds 同向；S5 显示该配置的 rescue 主要随完整 persistent Adam history source，而非单独 current source。此事实限于 M3OE、KuaiRand natural、Equal、d1/tab1 及相应 checkpoint；S5 的 moment gate 为 15/16，不能据此声称 m 或 v 单独致因（E011、E015、OQ003）。
3. **S6-R2 在 KuaiRand 观察到 test-side selected non-target gain，但 validation 方向相反。** R0-selected test ΔN 为 +0.0068（4/4 正）、ΔM 为 +0.0044（4/4 正），target ΔT 为 −0.0004；同一组 checkpoint 的 validation ΔN 是 −0.0039。R1/R3/R4 的同向结果只在 A-grid sensitivity 中成立，且 MovieLens negative control selected contrast 近零（E028–E030）。源报告在其 KuaiRand gate 上判为 GO；这不解决 validation/test 反转或增益机制。
4. **S8 的 MovieLens continuation 未提高冻结的原 F best。** Full-state selected-anchor 对照接近零，所有 MovieLens continuation best 均低于原 F best；late macro rescue 是 seed-skewed 的 overfit-anchor 现象，不能替代 S6-R2 的 KuaiRand test-side observation，也不能证明普遍 optimizer-policy gain（E026、E027）。

## Q2 — 哪些早期结论后来被修正？

- “小域梯度必然更噪、需要 per-domain momentum”没有得到稳定支持；comp1 的独特动量机制被 S0 的匹配 uniform-EMA 对照削弱，domain size 也没有被独立识别为主因（E001、E008、E009）。
- “Sample 消除了所有 optimizer/history effect”范围过宽。S6 Phase A 只支持 selected practical target gate 未通过；后续 Phase B 与 S8 仍观察到 MovieLens late trajectory effects，S6-R2 又在 KuaiRand 发现 test-side selected non-target gain，但 validation 方向相反。KuaiRand Phase B 的 M1 gate 失败，不能写成机制识别已全面通过（E018–E020、E028–E030、C014、C019）。
- `exp_sn.md` 是 2026-10-06 的历史快照：其“Phase B 未触发”后来已过时；该汇总遗漏了独立的 S5-ML，而 S6ML 与 S5ML 不是同一实验。S6 样本量复核显示 MovieLens d2=150,000、d0=34,500、d1=3,000，故 d1 最小、d0 居中（C015、C016、C018）。
- S7-B 报告曾将 MovieLens anchor39 step5 target AUC 写成 4/4 seeds 为负。原始 CSV 实为均值 −0.001226、3/4 为负、seed2025 为 +0.002359；LogLoss 均值 +0.011130、3/4 变差。已修正源报告与对应 Evidence/Stable Finding；“短窗内未见稳定改善”仍成立，但逐 seed 一致性证据降低（E025、C012）。

## Q3 — 哪些现象只在特定 dataset/objective/setting 下成立？

- Equal 下 KuaiRand d1/tab1 的 access/history rescue 是局部识别结果；不能直接外推到 Sample、其他域、其他 backbone 或 MovieLens。
- S5-ML 的 MovieLens Equal target/history pattern 只部分复现 KuaiRand S5；S6 Sample 下 selected gates 未达阈值，而 MovieLens late DH effects 出现在较晚训练阶段，受严重过拟合和 seed 影响。
- S7 的 Domain×Task/validation-gradient 观察只覆盖 MovieLens 与 KuaiRand 的 M3OE/Sample 设置。MovieLens late overfit 段的 U_k 与性能下降同步，KuaiRand 的 negative-U snapshot 没有同样后果；它不是跨数据集的充分性能标记。
- S6-R2 的 selected test-side gain 只见于所测 KuaiRand/Sample/M3OE F-vs-DH 对照；MovieLens selected effect 近零。KuaiRand 同一 selected checkpoint pair 的 validation/test 方向相反，且替代规则的 test 支持来自 A-grid sensitivity，并非精确重选 checkpoint。
- 旧 MovieLens ratio/role 结果描述的是特定数据生成、目标权重和训练协议，不能单独证明“大域必然受损”或“小域等于困难域”。

## Q4 — 哪些机制目前真正有证据？

- 在 S3 的 Equal/KuaiRand/tab1 配置中，shared-access 干预对 target 结果有因果影响；这证明的是干预效应，不是 gradient-conflict 根因。
- 在 S5 的 Equal/KuaiRand/d1/tab1 配置中，完整 persistent Adam history source 承载了大部分 target rescue；状态子成分 m、v、step/bias-correction 与参数路径尚未区分。S6 Sample Phase B 的 KuaiRand M1 未通过，限制跨 objective 机制主张。
- S4 已显示 raw-gradient norm matching 和 applied-step norm matching 单独不足以复制该 target rescue。
- S7-B 能改变部分 gradient utility 指标，但短程内没有稳定 validation 收益；S8 的 MovieLens endpoint 也没有 selected/best 性能收益。S6-R2 的 KuaiRand parameter swaps 在四 seeds 均退化，支持所测分区不可直接互换，但没有定位 DH gain 的来源（E032）。因此，gradient correction 或某个 optimizer component 造成 test-side selected gain 仍未被证明。

## Q5 — 哪些所谓机制其实仍然只是 hypothesis？

以下仍不是已建立机制：domain size 是退化主因；小域低 SNR 导致需要更长动量；Sample 下存在普遍 practical NT；negative U_k/gradient conflict 是充分性能原因；Equal history 机制跨 objective 保持不变；KuaiRand S6-R2 的 validation/test reversal 由某个单一机制造成；parameter swap collapse 已定位增益分区；MovieLens 与 KuaiRand 差异由某一个数据因素单独造成。KuaiRand test-side selected gain 是条件性 observation，不等同于普遍机制或部署收益。相关历史解释已在 Hypothesis History 中标为 WEAKENED、FALSIFIED、SUPERSEDED 或 OPEN，不能当作 Phase 3 前提。

## Q6 — 哪些问题仍然 unresolved？

- Sample 在更极端域比例下是否出现 selected minority-domain gap；domain size 是否有独立因果效应（OQ001–OQ002）。
- Equal/KuaiRand rescue 中 Adam history 的具体状态来源，以及 shared-access/history 效应能否跨 backbone、target 和数据集复现（OQ003–OQ004）。
- MovieLens late DH 现象中普通 overfitting 与 optimizer-policy 各自贡献；Domain×Task gradient pattern 的跨设置稳定范围（OQ005–OQ006）。
- KuaiRand 的 train/test mixture 或 label shift 如何与 objective effect 区分；S5-ML 部分复现由数据结构还是训练阶段调节（OQ007–OQ008）。
- KuaiRand S6-R2 的 selected validation/test reversal 与替代 selection rules 在精确 checkpoint 上的结果；跨分区 swaps collapse 后能否定位 intact DH test gain（OQ009–OQ010）。

## Q7 — 哪些旧研究线程不应在 Phase 3 中被重新当成有效前提？

不能预设小域更噪或更难、域大小已解释退化、M3oE 独有大域牺牲、comp1/域条件动量有独特优势、curriculum 或 adaptive calibration 已胜过 Sample、Sample 下存在普遍 practical NT、Equal 下 history 结果可推广到 Sample、target-only/HT 是安全部署方法、gradient conflict/U_k 修正会改善性能，或 DH 的 selected test gain 普遍存在/普遍不存在。S6-R2 的 KuaiRand 结果不能被当作某个机制或部署方法的既定前提；其 metric-layer reversal 和分区定位问题仍开放。

## Traceability Audit

- `reports/exp_sn.md` 已先行核对；其中 S0–S6 在索引中各登记一次，未重写重复阶段总结。S5-ML 独立补登；S6 DH trajectory analyses 重用同一批训练数据，标为非独立证据。S6-R2 在该快照之后完成，作为 evaluation-only EXP032 单独纳入；它复用既有 S6/S6ML checkpoints，不新增训练。
- 抽查 S5 聚合表、S6 Phase B audit、S6-R2 gate-summary / window / parameter-swap CSV 与 integrity report、S7-B validation 原始 CSV、S8 paired endpoints；核对了 dataset/objective/seed/checkpoint 口径和关键数值。S6-R2 复用了 16 个既有 F/DH checkpoints、没有新训练；Part D 的 manifest partitions / state-dict parity 检查通过，seed2025 retention 分母近零已限定说明。保留 S6 KuaiRand M1 失败与 MovieLens d0/d1 样本量标签更正。
- 报告中的观察值与机制解释分开登记；单数据集、单 objective、固定 checkpoint 的结果没有推广为普遍规律；S6/S7 派生分析没有计为独立复现。
- 本整理未做 literature matching，也未提出或设计新算法。

## Completion Counts and Phase 3 Readiness

| 项目 | 数量 / 判定 |
|---|---:|
| Experiment Index | 32 条：29 个既有实验组有正式运行/探测记录，2 个未运行/门控（E2、E7），另有 1 个 evaluation-only（S6-R2）；共 30 个条目有运行或评估结果 |
| Evidence Ledger | 32 条：STRONG 10、MEDIUM 19、WEAK 3 |
| Historical hypotheses | ACTIVE 2、PARTIALLY_SUPPORTED 3、WEAKENED 5、FALSIFIED 9、SUPERSEDED 1、OPEN 3 |
| Stable Findings | 12 |
| Open Questions | 10 |
| Dead Ends | 12 |
| Phase 3 条件 | **具备进入证据驱动的下一阶段梳理条件**；尚不构成任何特定算法或普遍机制已成立的依据。Phase 3 应以这些限定事实和未解决问题为输入。 |
