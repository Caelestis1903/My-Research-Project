# 全部问题终局闭合表

全部46项均为允许终态；不是把未做实验标不可识别。所有Q28–Q46共19项的原始观察、六字段与下一最小实验完整保留于[FINAL_REPORT第13章](FINAL_REPORT.md)。机器可读的每项证据和边界见[question_registry](question_registry.json)。

|ID|问题|状态|证据强度|回答/边界|
|---|---|---|---|---|
|Q1|Objective-equivalent sampling 是否真的改变 training trajectory？|RESOLVED|MEDIUM|明确配置内同目标q改变路径/早期性能；不普遍外推。|
|Q2|是否只是 checkpoint-selection artifact？|RESOLVED|MEDIUM|RESOLVED/MEDIUM：selection会改变符号、大小及两臂健康解释，但不是所有q效应的唯一来源。冻结test固定SGD早期−.001430及共同E50−.001405四负，common-SN/common-SB AUC选轮分别−.002747/−.001380四负；校准Adam早期−.009821及共同选择−.011520/−.010323四负。calSGD早期+.001614四正却AUC自选−.000972四负，ML早期+.003015四正、E50−.000986四负、自选均值+.000539仅1正3负，显示规则/阶段/人口边界。全部四规则、1000重选区间与共同轮次分别归档，不把条件选择区间当外层调参或pristine holdout不确定性。|
|Q3|是否依赖 Adam？|FALSIFIED|MEDIUM|Adam通用必要性被无动量SGD反例反驳。|
|Q4|若 optimizer-specific，是否为 m/v/history？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|状态映射有条件影响，唯一长期历史机制未识别。|
|Q5|非 Adam 机制是 exposure/repetition/coverage/order/geometry 中哪一个？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|非Adam候选路径未唯一分离，最小采样/几何控制完成。|
|Q6|shared/private parameter 是否必要？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|C局部AUC阳性但联合loss失败；长期必要性未识别。|
|Q7|domain size 是否解释 effect？|BOUNDARY|STRONG|精确别名改变名义池而不改变WR规律，未操纵新独立支持。|
|Q8|domain difficulty 是否解释 effect？|BOUNDARY|MEDIUM|目标AUC对mask敏感，loss/阶段/q不单调；不是自然困难度唯一因果。|
|Q9|task/domain imbalance 是否解释 effect？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|STRONG|N/p/标签/输入/基线共同变化，不能单独归于imbalance。|
|Q10|validation selection 是否改变结论？|BOUNDARY|MEDIUM|规则/评价人口/预算池会改变结论，所有预定规则均保留。|
|Q11|是否跨环境成立？|BOUNDARY|MEDIUM|第二环境没有稳定同向宏观损害，保留配置束/阶段边界。|
|Q12|SAF/D3/AdaTask 是否覆盖？|BOUNDARY|MEDIUM|D3/SAF目标不同、AdaTask固定组件有残差，一般主题部分覆盖。|
|Q13|是否存在未被已有工作覆盖的 mechanism gap？|BOUNDARY|MEDIUM|没有识别独特必要中介；既不宣称穷尽覆盖，也不强称创新。|
|Q14|是否值得方法化？|RESOLVED|MEDIUM|十项GO条件未同时满足；NO_GO。|
|Q15|SN/SB 的“same objective”在代码和 empirical gradient 上是否严格成立？|RESOLVED|STRONG|同forward J及同图导数字段期望身份成立；保留detach边界。|
|Q16|现象是否主要发生在 overfitting 之后？|RESOLVED|MEDIUM|明确配置在操作性pre阶段已有差值，不能归为全部late-only。|
|Q17|effect 是否只是 learning-speed / finite-budget effect？|BOUNDARY|MEDIUM|LR/batch/预算与clock改变响应，既不全确认也不全排除速度解释。|
|Q18|在相同 sample multiset 下，仅改变 optimizer-step 顺序是否改变结果？|RESOLVED|MEDIUM|同multiset的整体顺序干预改变阶段响应，长预算反向保留。|
|Q19|repetition 是否是 exposure effect 的必要组成？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|STRONG|固定E,N零重复不可行，PERM不能单独识别必要重复。|
|Q20|coverage 是否独立于 exposure/repetition？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|STRONG|R=E−U使覆盖和重复不能在当前资源下独立识别。|
|Q21|estimator covariance 的 anisotropy 是否比 scalar variance 更有解释力？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|谱描述有信息，shape独立于trace的必要性未识别。|
|Q22|shared C/R 是否是必要 mediator 而非相关参数块？|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|同Q6；C局部路径阳性不等于长期必要；R原已私有。|
|Q23|optimizer effect 与 sampling effect 是否存在真实 interaction？|RESOLVED|STRONG|calibrated共同22交互满足预定触发；fixed reference未触发。|
|Q24|第二环境失败时，boundary 究竟来自哪里？|BOUNDARY|MEDIUM|窗口/步钟/数据标签/训练子集等配置束边界已分解，非单一N。|
|Q25|existing literature 是否已经覆盖“same objective + different q + MDMT trajectory”机制？|BOUNDARY|MEDIUM|通用sampling/order/memory部分覆盖，不构成MDMT创新证明。|
|Q26|D3/SAF 的 objective change 是否使其不能作为 mechanism-equivalent prior work？|RESOLVED|STRONG|完整D3/SAF学习权重一般改变J，不是同目标机制等价控制。|
|Q27|AdaTask 的 optimizer-state intervention 是否能消除 sampling effect？|RESOLVED|MEDIUM|固定同目标AdaTask组件未消除q敏感性，不推出全论文失败。|
|Q28|routing 的 stop-gradient 边界|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q29|参数块名称与实际共享支持|RESOLVED|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q30|Adam 学习率赢家位于网格下边界|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q31|pilot 与 finalist 种子重合|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q32|历史固定 SGD 数字变化来自哪里|RESOLVED|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q33|只用 SN 调 LR 的估计对象|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q34|一阶矩遥测混入二阶矩范数|RESOLVED|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q35|稀疏遥测 lag 与真实 step lag|RESOLVED|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q36|onset 程序不提供全族错误率保证|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q37|IID 批次置换与课程策略|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q38|ORDER 也移动短尾批次|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q39|GPU 跨预算前缀数值复用|RESOLVED|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q40|状态消融的健康、幅度和学习进度|BOUNDARY|MEDIUM|原始观察及当前边界按终局报告第13章逐项保留。|
|Q41|自身早期窗口造成的环境表观翻转|BOUNDARY|MEDIUM|原始观察及当前边界按终局报告第13章逐项保留。|
|Q42|全局等范数没有隔离角度与块尺度|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|MEDIUM|原始观察及当前边界按终局报告第13章逐项保留。|
|Q43|MovieLens 的 seed 同时改变训练子集|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q44|破坏性控制改变共同早期窗口|BOUNDARY|MEDIUM|原始观察及当前边界按终局报告第13章逐项保留。|
|Q45|长时钟读数只有预算条件下的部分种子支持|BOUNDARY|STRONG|原始观察及当前边界按终局报告第13章逐项保留。|
|Q46|局部C的AUC阳性与loss/J冲突|BOUNDARY|MEDIUM|原始观察及当前边界按终局报告第13章逐项保留。|

## 不可识别项目的已完成最小实验及原因

### Q4

- 已完成：四seed三状态最终250/150/100/200；12最终+27历史配对、12影子同θ/范数控制与1740方向行均完成。
- 当前不可识别原因：整体更新映射与步幅/进度/块间尺度联合变化，状态最终预算与native不同且有floor；没有唯一长期历史必要性识别。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q5

- 已完成：Adam/SGD全部WR/PERM及SGD ORDER四seed最终训练/采样身份、完整几何与任务方向均完成。
- 当前不可识别原因：重复/覆盖/条件依赖联动，shape与trace及后续θ共同变化，不能唯一归因非Adam中介。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q6

- 已完成：12规定同θ局部×7条件全完成；C E10/E40 AUC阳性，E40 LL/J冲突，原联合完整训练条件未触发。
- 当前不可识别原因：局部单draw直接作用不等于长程架构必要性；没有事后放宽loss一致条件。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q9

- 已完成：真实域任务N/正例率/曝光/六cell支持审计、2x/4x别名与三mask强度完整四seed控制完成。
- 当前不可识别原因：3自然域的N/p/输入/标签率/基线难度共同变化，不能单独识别task/domain imbalance。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q19

- 已完成：全部四seed两optimizer WR/PERM最终采样/coverage/repeat身份及性能完成，SB小域E>N已实计。
- 当前不可识别原因：固定E,N使R=E−U≥max(0,E−N)，不存在同配额同支持的零重复控制，PERM残差不能排除必要性。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q20

- 已完成：4200最终臂轮的unique/repeat/exposure、first-seen与累计coverage及采样操纵核验完成。
- 当前不可识别原因：固定曝光与支持下repeat和unique coverage受代数约束，当前设计不能独立改变coverage而保持repeat。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q21

- 已完成：16固定θ×512/q完整谱、320移动θ/块记录、8完整任务梯度及下游控制分析完成。
- 当前不可识别原因：q同时改变trace/shape/direction/高阶矩/θ；没有独立trace-matched shape操纵，相关谱不识别额外因果解释力。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q22

- 已完成：同Q6全部12局部及共享支持审计完成，C AUC阳性、原完整联合条件未触发；R已私有。
- 当前不可识别原因：C局部路径不是唯一长期必要性；已私有R不能伪装成共享R私有化反事实。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

### Q42

- 已完成：12同θshadow/raw与全局等范数控制、1740方向/块幅度记录和12检查点replay完成。
- 当前不可识别原因：全局一个范数约束不固定CPRE块尺度；swap_v的R/P/E仍约1.48/1.35/1.18，纯角度作用无法隔离。
- 阻塞范围：只阻塞更强唯一/必要/独立因果主张，不阻塞有限配置边界结论。

额外2026–2029的触发条件未满足：没有用覆盖±.003的关键null等效关闭核心。C的AUC阳性与loss冲突是Q46的真实反例，不是以不显著替代机制不存在。所有条件未触发分支和最小控制都已记录；最终Git推送和远程验证后结束任务。
