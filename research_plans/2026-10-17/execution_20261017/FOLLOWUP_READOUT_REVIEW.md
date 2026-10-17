# 后续清单与局部确认读数自审

审查依据：已冻结DESIGN、N2/N3补充、原规划N1.5/N3.6；尚未读取新正式N1确认或N3结果。遵循start第11条主agent自审。

n17_followups.py仅从已冻结N1门槛与赢家生成命令，不运行训练、不额外筛选seed。N2/N3主锚严格2032–2035 × SN/SB父路径 × E10/E40，共16；native checkpoint文件及SHA逐一冻结。Adam当前映射仅Adam重要残差触发时加开，SGD整体C对照仅两优化器都触发时加开。quota仅所选优化器重要残差触发，保留16条IID父轨迹，不重复跑。

H200清单只收预设训练rho触发的优化器全部16条共同eta轨迹，先原样保留H100 receipt/selection/selected state/曲线/指标及SHA，命令对原run完整resume至200。原config的epochs=100记录最初预算，实际延长预算明确在H200命令和新receipt；不静默将原H100选点改为H200。清单生成在延长前进行，重复调用遇已延长轨迹会拒绝，防止覆盖原件。

n17_local_readout.py原始draw、每seed均值、四seed平均三个层级分开；MCSE用sqrt(sum(var_s/K_s))/4，不将draw当训练seed。32→64前后文件要求checkpoint、源码、学习率、batch、stream完全一致；重复prefix拒绝合并。独立确认只读已冻结一个子块/阶段，两个锚路径各自检查完整替换和full−norm残差的.0001/3-of-4/预期符号及MCSE≤.000025。精度不足才建议补到64；不能依据符号或新排名换候选。开发补精度时不得再次调用--nominate重选，原candidate_frozen.json及符号原件继续使用。

N4_trigger还要求N1重要残差；与方法收益、历史必要性和唯一方向机制均分开。绝对AUC/LL/J及反例留在原始/汇总表，不由资源门槛抹去。

Verdict: PASS_SELF_REVIEW。只批准既定前置条件满足后的清单生成与主agent读数；没有新正式N2/N3结果，不提前解封test。
