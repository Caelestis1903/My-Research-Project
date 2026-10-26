# Experimental Global Summary

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
