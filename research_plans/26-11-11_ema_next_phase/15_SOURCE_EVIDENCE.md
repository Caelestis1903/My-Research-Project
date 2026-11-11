# 15｜来源锁、证据覆盖与一手文献

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。固定来源机器清单为 [SOURCE_LOCK.json](SOURCE_LOCK.json)，已完成的数表复核为 [SOURCE_AUDIT.json](SOURCE_AUDIT.json)。本文件区分历史事实、本轮设计推导和未来实验，来源锁不等于模型恢复文件。

## 1. 固定版本

| 用途 | 仓库版本 | 固定 SHA |
|---|---|---|
| 本轮主证据及未来实现底座 | [26-11-11](https://github.com/Caelestis1903/My-Research-Project/tree/5ef45b18b973ed11be6fbb5f654562349998de60) | `5ef45b18b973ed11be6fbb5f654562349998de60` |
| main 上下文读取基准、旧因果计划声明 | [main 读取快照](https://github.com/Caelestis1903/My-Research-Project/tree/5cf92af0d3a8ed77688762616248166b9b68266d) | `5cf92af0d3a8ed77688762616248166b9b68266d` |
| 11/05 增强历史研究 | [26-11-06](https://github.com/Caelestis1903/My-Research-Project/tree/40705dfe0bb4fe614bdca835c0bae567348c87d0) | `40705dfe0bb4fe614bdca835c0bae567348c87d0` |
| 11/06–07 机制、数据/模型底座 | [26-11-07](https://github.com/Caelestis1903/My-Research-Project/tree/ffa3de067fcda5a1293af83a465d741e069493f1) | `ffa3de067fcda5a1293af83a465d741e069493f1` |
| 11/09 A0 与未触发后继 | [26-11-09](https://github.com/Caelestis1903/My-Research-Project/tree/1391affa5f08556c79b4cb2bfad15228a8b9129f) | `1391affa5f08556c79b4cb2bfad15228a8b9129f` |
| PUB 迁移 | [26-11-09_v2](https://github.com/Caelestis1903/My-Research-Project/tree/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8) | `a5f461de714232a416a2e95ca6fdc57fa7ce7ae8` |
| 正式发表模型统一端点 | [11/04 交付](https://github.com/Caelestis1903/My-Research-Project/tree/c8675ae2923c15b6965884b140b5b646690cd95e) | `c8675ae2923c15b6965884b140b5b646690cd95e` |

新计划的提交 SHA 由实际 Git 提交产生，在后续 `PREPARED` 中记录；不能用 main 读取基准冒充本规划版本。最新因果报告声明的源底座 `ffa3...` 也不是包含最新实验代码的交付 SHA `5ef45...`。

## 2. 最新报告、计划和实现核对

读取并交叉核对了以下报告文件，均在固定 `5ef45...` 的 [报告目录](https://github.com/Caelestis1903/My-Research-Project/tree/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/)：

- `FINAL_REPORT.md`：总结果、共享来源、时间互补、预测组合、Adam局部方向、独立选点Test与研究选择。
- `QUANTITATIVE_APPENDIX.md`、`EVIDENCE_SUMMARY.json`：全部1,604条汇总、五seed原值、逐格/阶段/端点身份。
- `REPORT_ADDITIONAL_EVIDENCE.json`：桥接、同状态方向与必要补充口径。
- `FINDINGS_CLARIFICATIONS.md`：allFast范围、不同策略身份及解释边界。
- `FINAL_REPORT_REVIEW.md`、`APPENDIX_REVIEW.md`、`EXPORT_REVIEW.md`：审阅修正与统计归约。
- `PROVENANCE.md`：实现身份、源环境、耗时口径、RAM释放及发布范围。

同步核对 [原研究计划](https://github.com/Caelestis1903/My-Research-Project/tree/5ef45b18b973ed11be6fbb5f654562349998de60/research_plans/26-11-10_ema_causal_attribution/) 的主00–11文档、实现/性能/调度修订及review，而不是只继承早期计划中的预期操作。以最终实际代码与完成收据裁定发生了什么。

核对 [因果实验包](https://github.com/Caelestis1903/My-Research-Project/tree/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_causal_20261110/) 的 `protocol/trajectory/history/interventions/functional/pairs/bridge/analysis/state/block/run` 及相关测试定义。核心区别包括：旧history存R而新B需存F；旧一步方向只是虚拟状态；新A必须改变后续梯度；旧全step q含初始化历史质量；selected和atF属于不同统计对象。本轮没有运行这些模型或重新生成归因数据。

| 已核对内容 | 直接来源 | 对本轮设计的约束 |
|---|---|---|
| F−O四条件Test AUC/LL | FINAL_REPORT、EVIDENCE_SUMMARY的09端点区 | 直接接受，不重复证明；LL单独记代价 |
| EMBED recent×old交互 | 04_temporal_EMBED及实现history/interventions | 保留两段历史，不先验删除旧历史 |
| q_logit/q_prob−F | 06_ensemble_minus_F、trajectory递推 | 固定面板依据，不能当新完整Test |
| 当前numerator、vhat和实际范数 | source_direction_states、08桥接区 | A保留二阶、明确coupled WD、native→patch→F |
| 参数/函数切口与精确pair | functional/pairs、02/03/05区 | 新B只做必要净纠正解释，不重做全归因 |
| 实际rho、optimizer、Gaussian D0 | protocol、源worker、objectives | 继承真实数值，不用嵌套旧domain元数据改目标 |
| gate detach与S/EMBED支持 | model_adapter、group_registry、vendor reference | 训练图不启用probe；domain embedding不纳入A |
| PA/PB人口 | readouts.pulse_rows、panels | rank用于AUC/risk，natural用于LL；PARAM真实前向 |
| RAM-only与资源边界 | start.md、state、外部scheduler、PROVENANCE | 参数已释放；新父轨迹用于新实验；双预算保持 |

`start.md` 的实际编码是GB18030。本轮继承其中五seed、独立审阅、RAM-only、设备准入等仍适用约束；用户当前研究指令决定这次仅规划、直接提交main的范围。

## 3. 本次实际完成的数表复核

从 `EVIDENCE_SUMMARY.json` 的全部1,604条已保存记录重新计算五seed均值、样本SD和t4置信区间，统计单位仍是seed。最大绝对误差：均值 `8.673617379884035e-19`，SD `4.336808689942018e-19`，区间 `1.734723475976807e-18`；无不一致。

原46个源文件索引、摘要与missing=0一起保留在SOURCE_LOCK。该复核覆盖**已保存汇总到统计表**的算术一致性，不声称重新训练、重跑所有参数干预或独立重建全部压缩原始表。最新报告的科学实现摘要为 `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`；其定义是包内特定源码路径的摘要，不等于git commit。

SOURCE_LOCK为所列上游文件保存git blob、SHA-256和字节数；已取得的本地审计副本逐字节核对到相应git对象。future PREPARED还须核对实际运行机器的数据和环境，历史机器“GPU可见”不替代未来预检。

## 4. main 与历史证据覆盖

main的三份上下文：[EXP_AI_CONTEXT](https://github.com/Caelestis1903/My-Research-Project/blob/5cf92af0d3a8ed77688762616248166b9b68266d/EXP_AI_CONTEXT.md)、[Experimental_Global_Summary](https://github.com/Caelestis1903/My-Research-Project/blob/5cf92af0d3a8ed77688762616248166b9b68266d/experiments/Experimental_Global_Summary.md)、[MDMT_ALGORITHM_METRICS_LEDGER](https://github.com/Caelestis1903/My-Research-Project/blob/5cf92af0d3a8ed77688762616248166b9b68266d/MDMT_ALGORITHM_METRICS_LEDGER.md)。读取当前有效状态、统一协议及与本任务相关的历史条目；冲突时采用后续明确修正和实际五seed结果，不将旧口头推测升级成当前事实。

| 历史研究 | 报告/代码 | 本次实际采用的判断 |
|---|---|---|
| 11/05 B、A、C | [REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)；directed_ema、grouped_ema、group_registry | B有效变化极小；A真实读出变化未转selected收益；C真反馈已测，KR未跑 |
| 11/06–07固定机制 | [FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md) | 一epoch pulse与未来新pulse不同于持久闭环；局部条件空间保留 |
| 11/09自适应A0 | [FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)；analysis.py | 误差改善但动作未变；task静态策略统一Fast；A1/B/C后继未触发 |
| 11/09 PUB | [FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)、[SOURCE_AUDIT](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/SOURCE_AUDIT.md)；optimizer.py | 完整迁移失败、solver并非失效；实际半径与公共WD异常支持A状态/范数控制 |
| 正式模型统一结果 | [ENDPOINTS.csv](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv)、[FINAL_REPORT](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md) | M3oE/HyperGate/AutoIFS；保留工程适配与超参来源标签 |

发表模型端点过滤：`role=PUBLISHED_BASELINE`、`checkpoint=selected`、`estimator=online`、`protocol_id=AUC_SELECTED_V1`、`data_protocol=FULLTRAIN_NATURAL80_10_10`，method仅 `M3OE_FIXED1_ADAPTED/HYPERGATE/AUTOIFS`。Test表取split=test，同dataset/seed配对。不能使用同文件的旧内部C_MIX或E10/E50端点冒充正式主基线。

## 5. 相关一手文献与使用范围

| 一手来源 | 本轮采用的内容 | 不据此声称的结果 |
|---|---|---|
| Izmailov et al., [Averaging Weights Leads to Wider Optima and Better Generalization](https://proceedings.mlr.press/r16/izmailov18a.html), UAI 2018 | 参数平均与集成之间有成熟关系，B应研究当前余量 | 不把其他模型的平坦性归因直接写为本项目已测机制 |
| Laine & Aila, [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242), ICLR 2017 | 历史预测可作训练目标 | 不将固定面板预测数组当新输入可部署教师 |
| Tarvainen & Valpola, [Mean Teachers are Better Role Models](https://arxiv.org/abs/1703.01780), NeurIPS 2017 | 参数平均教师是合理简单参照 | 不用Mean Teacher替代本轮额外知识和等预算控制 |
| Hinton, Vinyals & Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015预印本 | 集成信息压缩到单模型的直接路径 | 不预设本项目学生已保留两态AUC增量 |
| [PyTorch 2.9 Adam官方文档](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html) | coupled WD、m/v、偏差校正和step语义 | 实际源配置仍以冻结源码为准，不跟随文档后续版本默认值 |
| Yuan, Cai & Dong, [A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems](https://arxiv.org/abs/2410.05806), ICDM 2024 | 明确PUB论文身份、参数更新视角 | 不把旧迁移失败等同所有更新空间方法无效 |

文献为可采用的方法依据；算法是否值得继续由本项目已有证据与本轮有限五seed门决定。不存在“已有类似工作所以停止”的门，也没有以绝对原创为前提的设计。

## 6. 证据与下一阶段的界线

已接受：固定F有效、共享表示主导、EMBED近期/旧历史互补、面板预测组合余量、保留v的局部当前方向改善，以及历史各自确切失败环节。

本轮必须判别：A真实反馈是否在F上留下增量；B两态能否兑现预测余量及是否可压缩；C训练可见条件信息是否比统一策略增加价值。

不要求本轮解决：更多backbone泛化、全历史最优集成、每个神经元的完整解释、PUB迁移的全部竞争原因。它们不阻止执行当前最小算法比较。小域不是困难域直接继承，不列入假设或实验。
