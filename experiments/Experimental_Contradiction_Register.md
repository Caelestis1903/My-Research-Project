# Experimental Contradiction Register


> **2026-10-18：** C001–C023 保留 10-09 历史冲突/反例；本文件新加 C024–C034。区分 **真正相同实验定义的数值冲突** 与 **协议/人口/时钟/选点/统计门不同产生的表面冲突**；同分支后续最终状态覆盖早期进度快照。入口：[跨分支整合](Experimental_Branch_Synthesis_20261018.md)。


记录跨报告中的方向差异、后期修正或口径不一致。只有同一条件、同一指标、同一 checkpoint 才构成直接矛盾；不同阶段的结果大多是范围边界，不应强行合并。状态截至 2026-10-09。

## C001 — comp1 的大幅历史收益 vs S0 匹配 uniform EMA

- **Evidence A:** P0-5/P6 在旧配置中观察到 comp1 正收益，见 E005。
- **Evidence B:** S0 comp1−selected uniform EMA macro AUC −0.003033 ± 0.000815，4/4 seed 均由 uniform EMA 胜出，见 E008。
- **Difference in setting:** 旧 P0/P6 与 S0 的 objective、exposure、训练流程和 control selection 不完全相同；S0 的 uniform EMA 是新匹配控制。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** 旧性能观察保留；comp1 的域条件化机制特异性被 S0 限定。不得把旧的 comp1-vs-baseline 数字与 S0 当作同一配对效应。

## C002 — comp1 residual 曾被解释为小域机制 vs S0 未支持域条件化独特性

- **Evidence A:** P0-5 Sample+comp1 residual +0.005407 ± 0.000620，曾被解释为小域残余收益。
- **Evidence B:** S0 uniform EMA 在汇总 rotation 后优于 comp1；P1 SNR 也未显示稳定小域低 SNR，见 E001/E005/E008。
- **Difference in setting:** P0-5 是 comp1 × objective factorial；S0 针对 momentum identity 作匹配控制。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** Residual 现象不是同义于 domain-conditioned momentum 机制；统一平滑、选点与训练设置是竞争解释。

## C003 — 大域角色退化 vs “domain size 是主因”

- **Evidence A:** P0 fixed-N role-rotation 报告中 L-role 有负 transfer，见 E003。
- **Evidence B:** S2/S4/S5/S6 结果受 objective、domain/task identity 和 checkpoint 影响；size 未被独立操纵，MovieLens d1 才是最小域，见 E010/E014/E020。
- **Difference in setting:** 早期 role rotation 改变域样本结构；后期 KuaiRand natural 与固定 target 没有 size intervention。
- **Can the contradiction currently be explained?** Partially。
- **Current interpretation:** 大域角色相关性存在于早期 fixed-N 条件；size-only 解释仍未识别。

## C004 — Equal 下 selected loss vs Sample 下未见稳定 practical NT

- **Evidence A:** S2 Equal 在特定 Large cells（尤其 KuaiRand tab1）出现稳定 selected loss。
- **Evidence B:** Sample−SD 在 S2 为 0/24 practical losses，S1 为 0/72 stable practical NT，见 E009/E010。
- **Difference in setting:** objective、reference、target cell、dataset 与 checkpoint 口径不同。
- **Can the contradiction currently be explained?** Yes, by the objective and scope differences supported by the designs; no single mechanism is identified。
- **Current interpretation:** 不把 Equal 结果外推到 Sample，也不把 Sample 结果用于否认 Equal 下的局部 loss。

## C005 — S3 的 Sample access 差异接近零 vs Equal 下强 rescue

- **Evidence A:** KuaiRand Equal target-only−FULL 为 +0.038876/+0.039576，4/4。
- **Evidence B:** 同一 S3 配对结构下 Sample access contrast 约 +0.0003/+0.0001。
- **Difference in setting:** Objective 不同；Equal/Sample 有效域权重不同。
- **Can the contradiction currently be explained?** Partially; objective dependence is directly observed, mediator is not fully identified。
- **Current interpretation:** Shared-access effect is objective contingent in the measured configuration; Equal mechanism must not be assumed under Sample。

## C006 — T rescue vs norm-matched N/U no rescue

- **Evidence A:** S3/S4 T−F target AUC 约 +0.039、4/4 同向。
- **Evidence B:** S4 N−F 与 U−F 未复现正向 practical rescue，见 E013。
- **Difference in setting:** Same Equal target setting; control arms match different magnitude summaries while current-gradient composition/history remain different。
- **Can the contradiction currently be explained?** Yes, in the limited sense that the matched magnitude controls do not reproduce T; the remaining mechanism was later narrowed by S5。
- **Current interpretation:** raw-gradient norm and applied-step norm alone are insufficient explanations.

## C007 — S5 KuaiRand DH≈F vs S5-ML MovieLens DH has a nonzero contribution

- **Evidence A:** S5 KuaiRand Equal d1 DH−F about +0.0009/+0.0008; HT follows T.
- **Evidence B:** S5-ML MovieLens Equal d2 DH−F selected +0.005859/+0.003583, with late common gains larger,见 E015/E017。
- **Difference in setting:** Dataset/domain composition, target physical domain, imbalance severity, task structure, and training-stage behavior differ。
- **Can the contradiction currently be explained?** Partially; these setting changes are documented, but the causal moderator is not isolated。
- **Current interpretation:** S5 history-dominant pattern only partially generalizes; DH≈F is not universal。

## C008 — S6 Phase A STOP vs Phase B MovieLens late DH target gains

- **Evidence A:** S6 Phase A Sample F/T selected target effects are below +0.003 in both datasets,见 E018。
- **Evidence B:** S6 Phase B MovieLens DH−F target AUC at common e49 is about +0.0105/+0.0081; KuaiRand lacks the same late pattern,见 E020。
- **Difference in setting:** Phase A is F/T target-only contrast at selected/common endpoints; Phase B adds history/current policies and late checkpoints。
- **Can the contradiction currently be explained?** Yes as endpoint and arm differences; the late MovieLens effect does not overturn the Phase A selected gate。
- **Current interpretation:** Phase A STOP remains valid; Phase B provides a stage-specific optimizer-policy observation, not a practical selected rescue。

## C009 — MovieLens final macro gain vs target-domain AUC/LogLoss and F overfit

- **Evidence A:** S6 final T−F six-cell macro AUC is positive in MovieLens.
- **Evidence B:** Target-domain selected T−F is below practical threshold; F target AUC and LogLoss deteriorate severely late,见 E018/E020。
- **Difference in setting:** Macro combines target and non-target; final/common checkpoints are far beyond F validation-selected epoch。
- **Can the contradiction currently be explained?** Yes, metric composition and checkpoint stage explain why aggregate and target outcomes differ descriptively。
- **Current interpretation:** Final macro gain is not target rescue and not a normal selected-model improvement。

## C010 — S6 DH trajectory reanalysis reports long non-target runs vs S8 no better selected best

- **Evidence A:** S6 reanalysis finds MovieLens K=5 common checkpoints with non-target gain over threshold; selected non-target mean near zero,见 E022。
- **Evidence B:** S8 full-state best endpoint shows all MovieLens continuation bests below original F best, with selected-anchor contrast near zero,见 E026/E027。
- **Difference in setting:** S6 compares fixed checkpoints across full training; S8 continues from anchors and compares best continuation against frozen original selected best。
- **Can the contradiction currently be explained?** Yes; later S8 is the discriminating performance endpoint and does not erase the earlier fixed-epoch values。
- **Current interpretation:** The old trajectory effect is narrowed to late overfit rescue / “less deterioration,” not improvement of the model chosen by the normal protocol。

## C011 — MovieLens negative U_k tracks decline vs KuaiRand negative U_k without decline

- **Evidence A:** S7-A MovieLens U_k becomes negative as target validation AUC declines.
- **Evidence B:** KuaiRand has negative target U_k snapshots while target AUC continues to rise,见 E024。
- **Difference in setting:** Dataset, overfitting regime, gradient norm evolution, and trajectory differ。
- **Can the contradiction currently be explained?** Partially; it disproves sufficiency but does not fully explain the dataset interaction。
- **Current interpretation:** Negative U_k is conditionally associated with performance in late MovieLens and is not a universal cause marker。

## C012 — S7-A late gradient structure vs S7-B same-θ validation response

- **Evidence A:** S7-A reports late MovieLens harmful utility structure aligned with overfit degradation.
- **Evidence B:** S7-B changes that gradient utility at the same θ, but target AUC/BCE does not improve within the short window,见 E024/E025。
  - **Traceability correction (2026-10-09):** S7-B report overstated MovieLens anchor39 step5 AUC seed consistency as 4/4 negative. The raw CSV gives mean −0.001226, 3/4 negative, and seed2025 +0.002359; LogLoss mean is +0.011130, 3/4 worse. The no-stable-improvement conclusion remains, but the report and E025/SF008 have been corrected and evidence strength is limited accordingly.
- **Difference in setting:** S7-A is an observational 4-batch gradient probe across training; S7-B is a brief single-actual-batch intervention with a declared private fresh-Adam approximation。
- **Can the contradiction currently be explained?** Partially; different estimands and horizon matter, but the failed validation GO condition directly weakens the proposed mechanism chain。
- **Current interpretation:** Gradient metrics respond to intervention; their immediate correction is not shown to improve validation。

## C013 — S7-B approximate private optimizer state vs S8 complete full-state continuation

- **Evidence A:** S7-B resets private parameters to fresh-Adam after step0 and flags an approximation。
- **Evidence B:** S8 restores full optimizer state including private m/v, with same-state, replay, and validation checks passing。
- **Difference in setting:** S8 adds complete state restoration, longer continuation, and comparison to original best。
- **Can the contradiction currently be explained?** Yes; S8 is the stronger later test for long-horizon performance claims, while S7-B remains a short-window gradient diagnostic。
- **Current interpretation:** S7-B limitations cannot be generalized to the S8 full-state result; S8 still finds no selected/best performance improvement。

## C014 — S6 Phase B structural run success vs KuaiRand M1 gate failure

- **Evidence A:** Runner reported 16/16 Phase B runs structurally completed; code/update/pairing audit passed。
- **Evidence B:** Independent review found KuaiRand M1 m-conjunction passes 4/8 only; MovieLens passes 8/8,见 EXP028 source review。
- **Difference in setting:** Structural validity and mechanism-identification validity are distinct gates。
- **Can the contradiction currently be explained?** Yes.
- **Current interpretation:** Describe execution as valid and KuaiRand M1 as failed; do not say every scientific identification check passed。

## C015 — S6 old “MovieLens d0 smallest” description vs domain counts

- **Evidence A:** Some earlier Phase A narrative called d0 the smallest-domain casualty。
- **Evidence B:** S6 full analysis checks train counts: d2=150,000, d0=34,500, d1=3,000; therefore d1 is smallest, d0 is middle。
- **Difference in setting:** Earlier text used a mistaken role label; the later analysis reconciles it against the configured counts。
- **Can the contradiction currently be explained?** Yes; this is a factual labeling error, not a new effect。
- **Current interpretation:** Current database uses d1 as MovieLens minimum and d0 as middle; old label is not reused。

## C016 — exp_sn Phase B “not triggered” vs later completed extension

- **Evidence A:** `exp_sn.md` (dated 2026-10-06) says S6 Phase B had not triggered after both Phase A STOP gates。
- **Evidence B:** `stage_S6_phaseB_extension_scope.md`, run report, audit review, and final `stage_S6_analysis.md` document 16 new Phase B runs and analysis completed by 2026-10-08。
- **Difference in setting:** These are dated snapshots; the later extension followed Phase A STOP and was explicitly requested as a diagnostic continuation。
- **Can the contradiction currently be explained?** Yes temporally; the earlier statement was true at its date but is stale as current state。
- **Current interpretation:** Keep the Phase A STOP result; update the overall S6 record to include the later non-confirmatory diagnostic extension。

## C017 — S8 late macro rescue vs “best performance improvement”

- **Evidence A:** MovieLens late DH−F macro final-step is +0.010732 (4/4), with middle/late gains。
- **Evidence B:** About 89% of final macro mean is carried by seeds 2024/2025; target best gain is +0.000441; no continuation best beats the frozen original F best,见 E027。
- **Difference in setting:** Macro/final-step vs target-domain/best endpoint; late anchor is severely overfit。
- **Can the contradiction currently be explained?** Yes; endpoint and seed distribution resolve the apparent conflict。
- **Current interpretation:** “Partial late-anchor rescue” is the narrow allowed statement; “better selected model” is contradicted。

## C018 — S5-ML absent from exp_sn summary

- **Evidence A:** `exp_sn.md` summarizes S0–S6 and S6ML, but has no S5ML section or occurrence。
- **Evidence B:** `stage_S5ML_analysis.md` records a separate 16-run MovieLens Equal extension with partial Pattern A replication。
- **Difference in setting:** S5ML is a separate dataset extension, not the S6ML Sample Phase A included in exp_sn。
- **Can the contradiction currently be explained?** No conflict in results; this is a coverage omission in the prior summary。
- **Current interpretation:** S5ML is added as EXP026 and E017; do not mistake S6ML for S5ML。

## C019 — exp_sn “Sample history effect disappears” vs later Sample late-stage evidence

- **Evidence A:** exp_sn concludes that after switching Equal→Sample, history effect disappears below practical threshold, based on Phase A selected gates。
- **Evidence B:** S6 Phase B finds MovieLens late DH effects; S8 finds late macro rescue but no selected/best improvement。
- **Difference in setting:** The older wording generalizes Phase A selected target gate to all checkpoints and all history/current effects。
- **Can the contradiction currently be explained?** Yes by narrowing the estimand; the broad wording is overextended.
- **Current interpretation:** Phase A establishes no practical selected target rescue; it does not establish zero late optimizer-policy effects。

**S6-R2 update:** The later KuaiRand R0-selected test ΔN is +0.0068 (4/4 positive) while target ΔT is −0.0004 (E028–E030). Phase A's no-practical-selected-target result remains valid, but it does not establish zero effects on every metric or evaluation layer; this later result is test-side and is not validation-visible.

## C020 — exp_sn coverage end date vs later S6-R2/S7/S8 reports

- **Evidence A:** exp_sn summarizes through S6 and is dated 2026-10-06。
- **Evidence B:** S6-R2, S7, S7B, S8 analyses and S8 review are dated after the exp_sn cutoff and are absent from the file。
- **Difference in setting:** Later reports did not exist at the earlier summary cutoff。
- **Can the contradiction currently be explained?** Yes temporally; no claim of error at the earlier cutoff。
- **Current interpretation:** New global outputs include those later evidence and superseding conclusions; exp_sn remains the historical snapshot.

## C021 — S6-R2 KuaiRand selected-checkpoint validation gap vs test gain

- **Evidence A:** At the same R0-selected checkpoint pairs, validation ΔN is −0.0039 (1/4 positive) and validation ΔM is −0.0027。
- **Evidence B:** On test at those same checkpoints, ΔN is +0.0068 (4/4 positive) and ΔM is +0.0044; target ΔT is −0.0004 (E028/E029)。
- **Difference in setting:** Same dataset, objective, seed pairs, and selected checkpoints; the evaluation layer differs。
- **Can the contradiction currently be explained?** No; the direction reversal is observed, but its source is unresolved。
- **Current interpretation:** The DH test-side result is not a validation-visible selected improvement; do not collapse validation and test into one performance claim。

## C022 — S6-R2 KuaiRand selected test gain vs MovieLens negative control

- **Evidence A:** KuaiRand R0-selected test ΔN = +0.0068 and ΔM = +0.0044; A-grid R1/R3/R4 test ΔN means are positive (E028)。
- **Evidence B:** MovieLens R0-selected test ΔN is approximately zero, and available alternative-rule results do not show stable non-target gain; MovieLens canonical selected epochs are 7–13 (E030)。
- **Difference in setting:** Dataset, target domain, selected epoch range, and late-training trajectory differ, even though both use Sample/M3OE F vs DH source checkpoints。
- **Can the contradiction currently be explained?** Partially as a dataset/stage boundary; the moderator is not identified。
- **Current interpretation:** S6-R2's selected test gain is KuaiRand-specific in the tested matrix; the MovieLens null cannot be used to erase it, nor can the KuaiRand result be generalized to MovieLens。

## C023 — Intact DH selected solution gain vs cross-partition hybrid collapse

- **Evidence A:** Intact DH has KuaiRand R0-selected test non-target gain relative to F (E028)。
- **Evidence B:** Under the manifest-based Part D swap, C/D hybrids fall to test macro ranges 0.545–0.584 / 0.520–0.579; F_hybrid remains within about 0.00011 macro of F_full (E032)。
- **Difference in setting:** Complete trained solutions are compared with evaluation-only hybrids assembled from partitions across distinct solutions; seed2025 has a near-zero DH−F denominator for retention ratios。
- **Can the contradiction currently be explained?** Partially; cross-partition co-adaptation is consistent with the collapse, but does not identify the source of the intact DH gain。
- **Current interpretation:** The tested swaps show non-composability under this partition definition, not that DH's test-side effect is absent or localized to one component。


---

## 2026-10-18 跨分支补充的冲突、修订与处理结果

### C024 — Stage2+ “0正式训练” vs 同分支 Stage2D “16/16完成”

- **Evidence A:** [10-16 early execution](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/execution_20261016/execution_report.md) 在阶段审计时写0新训练、四臂阻塞。
- **Evidence B:** [10-16 later Stage2D final](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/execution_report.md) 明确16/16、四seed×四臂、数值核验成功。
- **Classification / Current status:** **RESOLVED / TEMPORAL SNAPSHOT SUPERSEDED**。较晚最终执行优先；早期“NO训练”仅属快照，不能用于新综合报告的阶段完成数。历史 S5/C×A 等未触发范围保持不变。

### C025 — 旧 S6-R2 KR selected test DH非目标阳性 vs 10-15 B1重建接近零

- **A/B:** 旧R0 +0.006757、4/4正；新重建 R0 +0.000579、2/4正，R1/R4 +0.011584、4/4正且LL代价。
- **Classification / Current status:** **OBSERVATIONAL DIFFERENCE / MECHANISM UNRESOLVED**，并非同init、同历史完整state、同谱系的成对重放；B1文件身份明确 RECONSTRUCTION_NOT_HISTORICAL_EXACT。保留原历史观察但不据此开发部署策略（E028–E029、E033）。

### C026 — Stage2D fixed-J同q显著负向 vs v2 Adam selected 区间跨0

- **A/B:** Stage2D SB−SN validation宏−0.010406，四seed负；v2−0.005556，3/4负且CI跨0。
- **Difference:** Stage2D **learned fusion、双Adam**；v2 **固定fusion、单optimizer**，后者还改变源码底座/SGD校准。
- **Classification:** **PROTOCOL BOUNDARY / NOT A DIRECT CONTRADICTION**。不能将两效应做“同设置复现实验成功/失败”的配对差；E035–E036。

### C027 — “Adam是必要条件” vs 无动量SGD也出现采样损伤

- **A/B:** 局部旧S5/Equal Adam历史carrier有作用；v2固定fusion/SGD SB−SN −0.002104（4/4负）；v3 fixedSGD早期也可四负。
- **Current interpretation:** **通用必要性 FALSIFIED**，但S5原命题是Equal/特定target的完整history source局部效应而非“所有现象要Adam”；不是同一量词。m/v分量、长期纯中介继续未识别；E015、E036–E038。

### C028 — v3多家族四seed负向 vs N17 KR八seed选点主Holm不通过

- **A/B:** v3 calibratedAdam test selected −0.008698（四负）等；N17 common LR KR Adam −0.005203（7/8负）、t95跨0、Holm p=.171875，SGD +0.000837、Holm p=.6953125。
- **Difference:** optimizer LR/selection预算、独立seed数、确认性家族与模型/人口细项不同。v3四seed同号的双侧随机化最小p=.125，不能硬标p<.05。
- **Classification:** **EVIDENCE STRENGTH REVISED**：条件效应保留，**N17主selected未确认**；不是证明所有q效果为零。E037、E039。

### C029 — N17 ML own-LR E100大幅正向 vs own selected负向

- **A/B:** 同 ML固定subset 8seed own对照：E100 SB−SN +0.021145、8正；各自验证选点 selected −0.010875、8负。
- **Difference:** 固定终轮vs各自validation选轮；own定义 SN lr=.001、SB lr=.0003，q与LR同时变化。
- **Classification:** **BUDGET × SELECTION × LR ESTIMAND BOUNDARY**，不是两份训练报告互相矛盾；E100 cell非劣置信下界不达−.003，不能升格成安全方法。

### C030 — 同一组模型的宏AUC损伤 vs 按训练/测试人口加权AUC获益

- **A/B:** 历史calAdam E50同预测macro −.009074、train-p +.001016、test-p +.003108；新 ML own macro −.010875、train-p +.000744、test-p −.013586。
- **Classification:** **RESOLVED MATHEMATICAL AGGREGATION DIFFERENCE**。cell macro与cell加权的定义不同，且均不是pooled AUC；不能以切换指标冒充目标上的稳健性能；E040。

### C031 — v3 C局部AUC阳性 vs loss/J不一致与N17 N4未触发

- **A/B:** V7 C的early/mid局部gap attenuation屏幕AUC 4/4正，但第二阶段 LL/trainJ联合门不通过；N17 embedding/E10两路径 full 4/4正、绝对~+3e−5，方向/实用门失败。
- **Classification:** **LOCAL POSITIVE / LONG-TERM NECESSITY NOT TESTED**。复核明确非“C全无效”和非“C必要机制”，N4 NOT_TRIGGERED，未运行不是阴性结果；E038、E042。

### C032 — same-J q方差差来自count vs N17 quota仍约2倍trace

- **A/B:** 按q改变域计数和每域梯度统计都可影响covariance；N2 fixed θ将域count固定后，ratio IID 2.009–2.584、quota 1.995–2.551，count项只有.001297%–.105936%。
- **Classification:** **COUNT-DOMINANT HYPOTHESIS FALSIFIED IN ANCHORS**；不能再声称这几锚主要来自count波动，但不能推出全部由shape而非trace或Adam history造成。quota的长期交互CI跨0；E041。

### C033 — S5/S8 late optimizer rescue 与“新q算法有效”混淆

- **A/B:** Equal历史S5 target rescue具有特定history carrier；Sample S8 的late rescue不提高原F best；v3/N17 q路径早期也可影响AUC。
- **Classification:** **DIFFERENT OBJECTIVE / INTERVENTION / WINDOW**。不能把旧的 history rescue改名为新的Sample采样方法，亦不能从Sample无新方法收益推历史Equal效应不存在。保持方法NO_GO、机制问题有界开放；E015–E027、E038–E043。

### C034 — N17 Adam bootstrap区间负 vs 预定t/符号Holm主准则未确认

- **A/B:** 8seed KR Adam selected: bootstrap95 [−0.010924,−0.000367] 排0，但t95 [−0.012006,+0.001600]跨0、预定Holm p=.171875。
- **Classification:** **INFERENCE PROCEDURE DIFFERENCE, NOT DATA CONTRADICTION**。两种区间如实记录，不能事后择一有利方法将主终点改写成“确认”；E039。

**整体说明：** 分支结束时“完成全部触发训练”和“独立因果机制已识别”是两个不同命题；53问题中大量 BOUNDARY / NOT_IDENTIFIABLE / INCONCLUSIVE 正是研究剩余空间，不能用发布完成状态自动关闭。完整对照请转[问题去向](Experimental_Question_Disposition_20261018.json)。
