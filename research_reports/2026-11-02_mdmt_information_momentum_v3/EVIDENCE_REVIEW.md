# Final evidence review

Verdict: **PASS**. Reviewer: `/root/reviewer`. The final report supports retaining A0 and closing this implementation under its registered S1 stopping rule. No further experiments are required by the triggered matrix.

## Numerical and provenance checks

- All **288 endpoint CSV rows** (144 validation, 144 test) match the underlying run/result and test-endpoint JSON metrics within 1e-14. No duplicate scientific endpoints were introduced.
- All **23,760 paired summary rows** independently reproduce the paired-seed CSV counts, means, sample SDs and 95% Student-t intervals within 1e-13. The unit is seed, n=2 and df=1; epochs and tasks are not treated as independent replicates.
- All **24 trajectories** retain 50 epoch records. For both online and fixed EMA, recorded selected epochs equal the earliest maximum of full-validation macro AUC. The 144 frozen endpoint identities, epochs and state hashes correspond one-to-one to final evaluation rows. The report copy of SELECTION_FROZEN matches the original bytes and its configuration-recorded SHA-256.
- The main report's validation/test macro AUC differences and seed directions match the tables. Both datasets have four candidate configurations with nonpositive mean validation differences and no 2/2-positive configuration. Therefore S2, S2_EXTRA and S3 remain NOT_TRIGGERED under the frozen S1 rule.
- The MovieLens AUC-decline/LogLoss-improvement pattern holds across both seeds for each of the four PAIR configurations. The report correctly avoids counting these shared-seed configurations as independent repetitions.
- KuaiRand P11 .05 has test macro AUC difference +0.0014036927, but only 1/2 positive seeds. Independently checked traffic AUC −0.0026566133, pooled AUC −0.0026006007, worst-cell AUC −0.0028243182 and traffic/pooled LogLoss +0.0016956740 match the narrative. All six cell AUC differences are one-positive/one-negative across seeds.

## Interpretation and required correction

The initial sentence that all selected test AUC paired intervals cross zero was too broad: the full comparison table includes, for example, MovieLens EMA P11 .05 minus P00 .20 with an interval above zero, and KuaiRand online P00 .20 minus A0 with an interval below zero. The report has been corrected to refer specifically to **main-endpoint EMA selected, candidates versus A0, test macro AUC**. That corrected statement is supported and does not alter the stopping decision.

The final report distinguishes the weak two-seed performance evidence from strong execution-completion facts, separates domain size from baseline predictive difficulty, describes macro/traffic and AUC/LogLoss conflicts, and limits P11–P00 interpretation because sampling and history change together. Unrun P10/P01/CV/PCOLL formal comparisons are described as untriggered rather than negative. No causal optimizer advantage or independent five-seed confirmation is claimed.

## Delivery scope

Design, implementation mapping/report, source and review identity, CPU and real S0 evidence, run manifest, stage decisions, configuration/selection freeze, complete endpoint/paired/cost CSVs, numerical appendix and final analysis are present in the report/package or their referenced attempt directory. The runner records 24 completed trajectories, 1,200 epochs, 41,400 optimizer/EMA steps, 144 completed test endpoints and released RAM state. Repository publication is a subsequent orchestrator action; this review makes no claim that a remote branch has already been created.

No training, GPU evaluation or algorithm changes were performed in this review. Checks were limited to the delivered evidence and scientific conclusions.


## 2026-11-03：P01/P10补充实验最终证据审查

Verdict: **PASS**；无阻塞修正。独立审查覆盖原REPORT末尾补充章节、NUMERICAL_APPENDIX追加数值、supplement表格和独立supplement_20261103原始attempt。

- 96行新validation/test端点逐项匹配8条新轨迹的result/test JSON；144行A0/P00 .05/P11 .05复用端点匹配原冻结表，原表SHA256仍与补充来源身份记录一致。新旧身份未混作新增独立重复。
- 独立重算14,256个逐seed效应：四个简单效应、P11−P10−P01+P00交互及四个相对A0差均正确，覆盖两数据、validation/test、online/EMA、selected/E10/E50以及全部六格、域任务、宏、traffic、pooled和worst指标。7,128组均值、样本SD及95% t区间匹配，n=2、df=1。
- 8条新轨迹各有50个epoch，合计400epoch、13,800次optimizer/EMA更新；累计worker时间10,207.5999秒与成本说明一致。八份数据manifest的raw/数组/行号/encoder/scaler/域数及split大小匹配原v3。每个估计器的selected epoch均是完整validation宏AUC最早最佳；48个新test身份、epoch和hash逐一匹配冻结清单。
- 主表、A0对照及区间正确。KR静态ECCM相对P00的EMA selected test AUC均值+.012099且2/2正，与validation均值−.000101、1/2正形成报告所述分歧；KR P01/P10相对A0的test宏均值各只有1/2正，ML两新增方法均0/2正。故“没有组件获得稳定超过A0证据”与本次数据相符。
- KR EMA selected交互test−.012347、validation−.005217均2/2负，test LL交互+.004646且2/2不利；静态ECCM与普通Adam动态选择的test增量不构成正协同。报告将已测局部方向标MEDIUM、内部机制标WEAK，未把两seed结果推广成普遍因果结论。
- 阶段与估计器边界也已核对：KR EMA test交互E10/E50均值−.001088/−.014387，但E10仅1/2负；KR online selected test交互−.010877且2/2负，而online E10均值+.002277。ML EMA selected交互一正一负、E50均值转正；ML online selected test交互+.001459且2/2正。报告明确以EMA为主、不给在线模型作无差别外推，完整online口径保留在表中。
- 域任务解释与记录一致：KR P01/P10的最小d2相对A0均2/2改善，最大d1平均下降但各仅1/2负；P10 d1t1 AUC均值−.003128且2/2负。KR负交互的d0/d2均2/2负，两个任务宏均值均负，但task1仅1/2负。报告未将平均大域代价夸大为一致牺牲，亦未将域大小认定为独立原因。

原S1停止仍保留为当时决策，新增研究明确发生于原test公开之后。补充固定8条已完成，未扩展S3、搜索或PCV/PCOLL；那些控制的缺口仍被列为未解决问题。无需为本次结论追加训练。

本轮审查只读取原始证据并重算数值，未运行训练/模型评价、未修改算法、未新建报告文件。
