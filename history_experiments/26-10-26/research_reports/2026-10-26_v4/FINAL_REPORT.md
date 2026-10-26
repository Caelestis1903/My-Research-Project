# DFD-v4.0 完整阶段研究报告

**结论：当前DFD-v4没有获得强简单对照之外的完整方法支持，按预注册规则结束。** ML需求可预测但P2形成/完整增量未晋级；KR在P1即失败。所有已触发训练、184个冻结预测导出、重建与12结构推理计时已完成；P3/P4及正式/外部训练均NO_START。另有一项规定机制信息导出遗漏，下面明确披露，不把核心实验完成概括为全部机制信息无缺失。

## H1/H2/H3

- **H1：ML有限支持，KR当前版本未支持。** ML CONST−DEMAND LL平均+.001089350，两seed六格都非负；KR平均−.000400568，两seed只有2/6、1/6格非负。两数据不合并成一个平均结论，证据**MEDIUM**。
- **H2：特定需求形成未获支持。** ML对UNIFORM的平均LL增量仅+.000027408，GEOM/SHUFFLE未被稳定超过；责任确实变化、非退化，但不能据此证明功能。独有机制证据**WEAK**。
- **H3：完整实践增量未成立。** ML九对照仅三项通过；对H_DIRECT、ETA_DIRECT开发LL平均更坏+.001639660/+.001814021，并触发最差cell保护。固定exposed test中相应LL均值更坏+.008976175/+.009290476，AUC两seed也更低。当前版本为**VERSION_NOT_SUPPORTED**，有限反例证据**MEDIUM**；不是所有需求/迁移方法不可能有效的证明。

## 稳定现象与反例

DFD对UNIFORM的LL在四角色两seed均略好，但尺度小、AUC不稳定；相对SHUFFLE有seed反转，GEOM四角色平均LL均更好。普通条件读取胜常量、胜LOCAL或LOCAL_CAP仍可保留，强直接概率头足以竞争，native自身还是必要参照。test相对NATIVE的DFD LL一好一坏，不能用平均小改善宣称稳定成功。

ML相对H_DIRECT的AUC在Q两seed略高，test却两seed更低，说明指标与人口角色不能择一。四个native的test epoch24 LL均差于合法selected；ML Q/legacy的晚期AUC反而上升，不能把晚期排序或少恶化改写为最佳泛化。域规模未被识别为主要解释，未见可普遍推广的小域改善/大域下降规律。

## 数据、重复与适用范围

自然ML/KR三域两任务、fit-only编码，原生Sample目标/full40960/coupledAdam/FP32，4native24完整轨迹；ML选epoch13/13，KR14/16。每个数据独立决定，seed是重复单位，n=2描述性t区间与每seed1000次配对bootstrap分列，未把六格、fold或角色当额外seed。ML用原始用户，KR无可信原始用户ID而用row退路。

test/legacy历史暴露，冻结hash `7163b07065bfdb7462dd05723b7c761acb8ea7cd41693e9fdb6d93a7dad401d4` 之后才统一评分。没有真正新数据确认、没有6101–6104正式复评或STEM/AutoIFS领先主张。完整macro/natural/pooled/六格、配对差、区间、selected/固定终态、身份与修订均附在分报告、[tables](tables/)和[工件索引](ARTIFACT_INDEX.json)。

## 执行可信度、缺项与存储

DESIGN→IMPLEMENT→独立REVIEW→RUN→主agent分析全部完成；实际raw-X重建最大8.28890167393e-07≤1e-6，184/184导出及12/12结构计时通过。训练源码/数据/行计划身份冻结，修复和过程偏差在[PROCESS_REVISIONS](PROCESS_REVISIONS.md)，未选择性重跑负结果。

**MAJOR机制信息缺口：Q/outer原refitη的固定μ分组未导出；清理后不能由hash/摘要唯一恢复，不能用P_read后的ETA_DIRECT替代。** 该项清理前漏检，限制outer需求分层/迁移解释，不改变已保存H1/P2门控输入或主要负结论。按start最小必要原则记录MISSING，不为补机制分组重新训练已关闭版本，也不把ETA_DIRECT微调结果冒充原η。

遵从用户要求，模型/优化器和大中间缓存已删，未改动其他`/home_new`目录；清理移除8.32GB，无.pt/.pth或模型symlink。保留日志、原始预测、指标、源码/角色/选择/责任信息。指标可从预测复算；重做模型前向需重训。

## 成本与下一项可区分问题

已记录训练/开发leaf5674.226502秒，共享GPU进程跨度和/并集另列，缺失setup等不伪称0或精确总量。ML DFD完整raw-X P50约5.709ms，H_DIRECT2.535ms；更高实现成本没有获得完整风险收益支持。三个成本账重叠，不相加，不称GPU小时/能耗。

下一项优先在真正封存的自然数据窗口，比较强直接概率与普通来源读取，先重建来源监督的独立增量；不继续堆K、元优化或额外seed追逐当前版本阳性。该未来研究本轮未执行。

详见[报告索引](REPORT_INDEX.md)、[P1](P1_DEMAND_REPORT.md)、[完整流水线](FULL_PIPELINE_REPORT.md)、[归因](ATTRIBUTION_REPORT.md)、[成本](COST_REPORT.md)、[开放问题](OPEN_QUESTIONS.md)。
