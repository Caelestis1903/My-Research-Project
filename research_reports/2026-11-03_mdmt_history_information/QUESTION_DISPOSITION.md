# 研究问题处置表

本轮13条新轨迹均为seed2022；下列新方法科学推断均为WEAK。D0门是本轮预先冻结的研究决策规则。

| 问题 | 当前证据 | 处置 |
| --- | --- | --- |
| 当前采用哪个方案？ | ML所有候选validation selected AUC低于A0f；KR正宏候选均触发具体cell保护门 | 保留A0f |
| 是否胜固定A0f？ | 没有候选通过完整准入规则；KR C_MIX有+.005088验证/+.009531测试宏增益，但验证d0t0−.006295 | 保留收益与代价；不接受总体替代 |
| 是否胜有界调参A0b？ | D1未触发，A0b未运行 | 未解决，不能声称胜过 |
| 条件增强是否必要？ | H_MIX−H_REAL：ML test−.000856，KR test+.012979但validation−.002958 | 无稳定必要性证据 |
| 应处理当前梯度还是旧历史？ | KR C_MIX验证宏/traffic/pooled优于H_MIX；H_MIX测试宏更高但traffic较低 | 处理位置有不同代价；旧历史必要性未成立 |
| 另一任务条件是否有增量？ | donor优先条件确实被使用；U_MIX/H_U/T_U未触发 | 未解决，支持率不能代替增量比较 |
| 方向性是否超过等范数缩放？ | H/T有非零干预；H_NORM/T_NORM无正式轨迹 | 未解决 |
| 简单缩放是否足够？ | SCALE未通过D0；ML test宏−.000406，KR+.000078，KR d2任务损失明显 | 本配置未提供总体替代；不能代替NORM控制 |
| T联合机制有交互吗？ | ML validation/test AUC交互+.000592/+.001691；TM仍不胜M及A0f | 存在单seed正交互，不保留已获益联合机制主张 |
| KR T联合机制如何？ | 默认TM在S0为4.61×A0成本，被阻塞 | COST_BLOCKED，无正式性能结论 |
| LL与AUC是否一致？ | ML T0/TM LL改善、selected AUC下降；KR多数test LL改善 | 两目标分离，不能把LL改善写成排名改善 |
| 是否有小域收益/大域代价？ | KR M/H_MIX test有；C_MIX六格同升；H_REAL多为反向 | 方法和评价人口相关，非普遍规律 |
| 规模是否是主要解释？ | ML最大域基线最弱；KR最小域基线优于d0；同域随方法方向不同 | 规模不足以单独解释；不以小域代理困难域 |
| 什么现象重复？ | ML selected AUC普遍负，KR C_MIX跨几个端点宏正，T类ML LL改善 | 同seed内重复；没有新候选的跨seed稳定性证据 |
| 哪些证据相矛盾？ | KR M/H_MIX/A1验证负测试正；ML若干E50正selected负；KR T0 EMA/online selected反向 | 保留人口、阶段、选点与估计器竞争解释 |
| 置信区间与方向数？ | 新候选n=1；各对照差原值全部导出 | 新候选SD/95%t区间留空，不把格子或epoch扩充为n |
| 是否做五seed确认？ | 两数据D0无合格家族 | CONFIRM/三个未开发seed NOT_TRIGGERED |
| 简单方案胜出后删什么？ | 本轮未接受任何新方案 | 当前工作方案不加入增强、H修复或T运输；代码保留用于复现 |
| 是否存在足以改写结论的已知实现问题？ | 独立审查修订关闭，CPU/S0通过，650epoch和冻结评估完整 | 未发现未解决的结果相关实现阻塞；科学主要缺口是单seed及未触发控制 |
| 最优下一实验？ | KR C_MIX的cell验证/测试反转与阶段变化最直接影响接受决定 | 下一独立任务固定C_MIX/A0f补2024，预先检验同格、人口与阶段方向；本轮不执行 |

全部逐seed数值、选点hash和配对差见RUN_MANIFEST.json、endpoints.csv、six_cell.csv、paired_seed.csv、paired_summary.csv及NUMERICAL_APPENDIX.md。D1、REPLICATE、D2和CONFIRM未触发；本轮没有DEFERRED_BY_BUDGET家族，也没有把未运行控制记为失败。
