# V13 方法化条件：诊断闭合后的冻结判据

GO需要十项同时成立。当前没有已验证的新方法primary endpoint/四seed稳定全域任务收益；C局部AUC筛选阳性不能补齐这些要求。诊断证据不足以GO，最终决策在冻结test与全部报告完成后签发，不以此表提前结束任务。

|必要条件|终局诊断证据|判断/边界|
|---|---|---|
|1 稳定pre-overfit effect|固定SGD.1与Adam.01等早期四seed负，MEDIUM|在明确配置成立，calSGD不同；操作onset不证明绝无overfit|
|2 非selection artifact|共同固定轮/早期、四规则/1000重选均完成，MEDIUM|非普遍selection-only，但selected规则敏感；冻结test确认待实际完成|
|3 非单纯速度|LR/2B/足预算及精确时钟完成，阶段方向改变|唯一速度解释未被普遍排除，实用平台不等于全局收敛|
|4 第二环境或可解释边界|KR/ML共同16/各自窗口、子集seed边界已完成，MEDIUM|无跨环境稳定同向宏观损害；环境束/N不唯一可识别|
|5 至少一反事实mediator|V11whole-state与V7 C局部AUC路径有条件影响，MEDIUM|长期/唯一必要中介未识别；C联合loss门槛未满足，full-training条件未触发|
|6 排除主要替代|目标/图身份STRONG，late/selection/Adam通用说法有反例|重复/coverage、shape/trace、content/tail-clock和状态/进度仍联合变化|
|7 现有方法不能直接覆盖|D3/SAF目标不同；AdaTask固定组件残余q敏感；九篇原文已审计|一般采样/方差/order/memory已有先例，无普遍覆盖否定或独特创新证明|
|8 真primary endpoint改善|无本轮新方法全训练候选|局部一步/gap收缩/late-rescue不满足此项；test不用于新选方法|
|9 四seed方法稳定|baseline/受控效应四seed均保留，反例不删|baseline重复不等于方法四seed收益；部分100轮仅n2|
|10 原目标/全部域任务不退化|全AUC/loss/J/六单元均实际归档|C/V11/V8都有指标/阶段冲突；未证实全面健康恢复|

不提出balancing/weighting/Adam组合以获取GO，不把当前无完整新方法证据解释成算法在所有调参均不可能。没有依据关键零效应的等效假设关闭核心；不触发额外4seed。下一步只运行已经冻结的原有条件test，随后关闭Q2/Q10/Q13/Q14并给出唯一总分类、NO-GO或GO及理由。全部46问题和Q28–Q46共19项原样保留。
