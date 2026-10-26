> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`39620f8ddb5fb72a98484ef2f4846034ed8a6a7ea7b8200ee5765330067a7c57`。

# MDMT-NEXT-v2 最终报告

本轮已执行到预定不晋级停止条件：**两条MVP的48条核心24epoch轨迹全部完成，但没有方法通过晋级门槛。正式四seed/50epoch矩阵锁为0条，未开放外层评价。** 这不等于全部可选解释已排除；必要aux及近期基线均有明确预算停止记录。

## 执行范围

| 阶段 | 实际结果 |
|---|---|
| P0与实现 | 独立包、数据身份/角色、原生适配、C8/RO梯度与完整恢复检查；通过必要检查 |
| S1 | 8条接受的10epoch warmup；另保留2条mean修正前尝试；H1/H5与冻结读取诊断完成 |
| 静态校准 | 16条固定5epoch分支；ML锁TRANSFER_1_TO_2，KR锁BASE |
| S2核心 | **48/48完成24epoch，48个exit0**；selected、AUC/LL-best及epoch24均保留 |
| 必要aux0 | 4条执行后预算停止，末epoch22/10/9/9；没有完整24epoch析因 |
| 近期基线 | 6条预算停止；ML末epoch13/13/19，KR7/7/7；完整外部比较未解决 |
| 正式与外层 | STOP_NO_PROMOTION；0条正式训练，外层评价调用0 |
| 新确认 | UNSEEN_CONFIRM_PENDING_DATA；没有重切旧数据伪造确认 |

84个实际训练尝试的配置、状态和checkpoint均保留；其中两条旧mean尝试不进入接受的S1比较。checkpoint已从临时路径归档到仓库工件目录，原临时链接仍保留作来源记录。

## 主要结果

下表为候选减参照的两个开发seed均值，统一在validation按预设规则选点后评分Q_explore。LL越低越好，AUC越高越好。

| 比较 | 主指标 | ML平均差 | KR平均差 | 四配对方向 |
|---|---|---:|---:|---|
| LAST−STATIC | macro LL | +.000228 | +.003962 | 4/4变差 |
| HIST−STATIC | macro LL | 0 | 0 | 全部相同 |
| ridge / NN−HIST | macro LL | 0 | 0 | 全部相同 |
| LPS−DIRECT | macro AUC | −.001713 | −.000386 | 4/4变差 |
| K4 DIRECT−K1 DIRECT | macro AUC | −.002189 | −.005638 | 仅1/4改善 |

**当前版本无独立主增量，证据MEDIUM。** 开发seed有限且Q属于开发，不能把这一判断扩大为所有历史机制、状态模型或多原型方向都无效。HIST、ridge、NN各自1568个控制期主batch的非静态动作数均为0，对“本实现没有实际使用额外动作”这一受限事实，证据为STRONG。所有LPS主选点及epoch24快照的深度cap也都是0；这说明被评价的主策略没有采用域内树分区，不能把它当成充分检验了所有局部需求模型。

**反向证据必须保留。** 累计均值、EMA与warmup冻结规则在四条BASE时间后段trace的同状态H1效用均为正；这是MEDIUM强度的机制机会，尚无完整闭环收益。ML3101的K4−K1 AUC为+.000864，但ML3102为−.005242，KR两个seed均负。共同epoch24，ML的K4−K1 LL改善，而主selected ranking未改善；KR的LPS−DIRECT AUC也出现阶段反转。不能挑这些较好指标替换原主张。

**校准是竞争解释，而非预设答案。** CAL_OFF在ML两个seed相对OWN_OFF的AUC提高约+.008475/+.007925并降低LL；KR一正一负。校准解释在ML有MEDIUM支持，但不是普遍规律；完整CAL_PROB_SELECT闭环因reserve不足仍为PENDING_BUDGET。当前证据不足以把收益归因于可迁移多单元知识，也不能声称已排除aux、actor容量或额外元标签拟合。

## 域规模、难度与取舍

域规模作为主要因果解释的证据为**WEAK**。ML两个大域规模接近却有不同native AUC；KR最小域并非AUC最低域，且低基率域的低LL与低AUC并存。K4−K1最大AUC下降发生在KR中等规模d0，不能用“小域更困难”解释全部结果。逐任务的域宏AUC与自然pooled AUC有时方向不同，主目标始终保持预设macro。

**确实存在条件性的“小域改善/大域下降”，但没有稳定普遍定律。** DIRECT−OWN_OFF的selected AUC在3/4配对出现该模式，到共同epoch24仅1/4保留；K4−K1只有1/4 selected配对出现。现象依赖比较模型、任务指标、seed与训练阶段。完整六格、任务、基线难度和规模图均保留，不只汇报有利小域。

## 未解决事项与协议偏离

1. 四条aux0都未完成24epoch，AUX_ATTRIBUTION_PENDING_BUDGET；不提出已排除aux混杂的主张。
2. 三个近期方法均有来源与关键语义收据，但六个pilot未完成24epoch，BASELINE_UNRESOLVED；不声称外部最优或复现原论文分数。STGU是明确的三域Adam/分组适配。
3. 早期9条S2存在预算计费遗漏；后39条已修正，主训练/动作/主指标选点未变。aux的epoch级预算检查还造成ML32.099s、KR33.524s超额。这些是执行偏离，不能称严格等预算比较；原始账本和旧budget_best均保留。后续baseline改用进程期限。
4. 共租负载和未完全分离的研究shadow使纯部署/GPU kernel成本不能精确复原。本轮不提出精确效率倍数或严格同预算优越性结论。
5. 无正式四seed结果、无新数据确认；旧test与旧Q_confirm仍是历史已暴露基准，本轮没有使用它们替代确认。

这些限制不改变48条核心同曝光/预定selected比较的结果，但限制成本、机制归因和外部泛化主张。没有通过追加seed、换选点、删不利数据或超预算补跑来制造阳性。

## 最有判别力的下一实验

RHD优先检验一个冻结简单反馈规则的完整闭环是否仍优于STATIC/HIST，以反驳“同状态H1小收益自然转化为训练收益”；不先扩NN。LPS优先补全有预算的aux析因，并把CAL_OFF及容量匹配K1作为强参照，尝试推翻“多单元本身带来增量”的解释。域规模研究需要固定域身份和评价支持、独立操纵数据量并控制曝光。它们是下一版本明确问题，不是本轮隐含追加授权。

## 工件入口

- [S1报告](S1_REPORT.md)、[S2完整表与分析](S2_REPORT.md)、[时间重放](REPLAY_REPORT.md)。
- [近期基线](BASELINE_REPORT.md)、[成本与偏离](COST_REPORT.md)、[正式停止与外层状态](FORMAL_REPORT.md)。
- [实现复核](IMPLEMENTATION_REVIEW.md)、[开放问题](OPEN_QUESTIONS.md)、工件索引（`../ARTIFACT_INDEX.md`，服务器留存，未随本次发布）。
- 每seed主选点表（`../tables/S2_selected.csv`，服务器留存，未随本次发布）、全配对表（`../tables/S2_all_contrasts.csv`，服务器留存，未随本次发布）、六格配对（`../tables/S2_cell_contrasts.csv`，服务器留存，未随本次发布）、晋级记录（`../tables/promotion.json`，服务器留存，未随本次发布）。
- [RHD训练曲线](figures/RHD_validation_curves.png)、[LPS训练曲线](figures/LPS_validation_curves.png)、[规模与表现](figures/domain_size_vs_auc_delta.png)。

本报告结论以本轮真实工件为依据，未预设Curriculum、Momentum有效，未预设M3OE在困难域失败，也未将小域等同于困难域。
