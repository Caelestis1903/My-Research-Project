# N3 局部计算与筛选实现细则

DESIGN COMPLETE；锚点必须来自N1实际完成的原生E10/E40状态。首批16锚和32独立draw按原规划，采样RNG以seed、阶段和固定独立stream派生，两q使用共同底层均匀随机数；与N1训练draw及N2矩估计流独立。所有候选的梯度在同一冻结theta与原生optimizer state计算，用原生torch optimizer实际一步得到θ_N/θ_B，不把线性梯度插值误称Adam更新替换。

首批每draw计算native SN、native SB、整体C四格、五C子块各四格、P完整替换（四格共享native SB，避免重复计算）。各完整替换直接复制对应θ_N坐标；范数或方向控制只改指定块，其余坐标精确取θ_B。r<1e−12标方向未定义，保留完整替换，不造方向。未更新anchor每锚只评价一次。

每候选在完整固定validation人口预测，selection/diagnostic分别给宏及六cell AUC/LL；完整训练人口实际J单独扫，不能用训练batch loss替代。同时保存相对anchor/SB/SN signed变化、四格交互和实际应用块更新范数。无候选状态用于长期训练，预测SHA与行ID指纹足以复现；不需保留每候选的大二进制。

机制开发使用2032..2035的diagnostic人口，确认使用2036..2039独立训练seed与独立draw bank；明确diagnostic已用于预设资源触发，不能称新holdout。

从五个C子块、E10/E40选择至多一个子块/阶段，不额外添加原文没有规定的开发阶段显著性门槛。对方向均有定义且四seed/两路径都有完整draw的候选，按两路径完整替换相对SB的四seed均值之平均的绝对值最大者选择；精确平局按E10优先、embedding/shared_STAR/skip/star_mlp/experts顺序。冻结该跨路径均值的signed方向；方向残差另冻结其跨路径均值的signed方向。开发阶段路径方向相反或低于实用界也保留，独立确认可证伪所选候选。原文.0001、3/4及双路径门槛用于独立确认，不在开发阶段新增提前停止规则。不将整体C的阳性自动归给任何子块；整体C无论是否筛选均完整报告。

候选四seed平均的MCSE=sqrt(sum_s Var_draw(Δ_s)/K)/4。32draw若相关平均的MCSE>.0001/4，只对冻结候选/阶段补draw32..63，并保留原32，最多64；不依据新符号挑另一个候选。若所有候选都有结构性的方向未定义而无法提出方向假说，记录识别限制；不追加draw搜索阳性。确认四seed遵循同样的32→64精度规则；达到64仍不足为INCONCLUSIVE_PRECISION。另报训练seed的不确定性，MCSE不替代seed区间。

确认两个路径各自检查原文N3.6的完整替换、方向残差、预期方向和≥3/4规则；只有还满足N1.6才触发N4。AUC与LL/J/绝对健康冲突必须保留，不把gap接近SN当收益。只有两个优化器均触发N1.6且主为Adam时，在SGD重复整体C四格，不展开另一套子块。

N3.5条件映射复用N2同theta512draw的在线均值/矩，原生Adam数学更新与参数落地浮点舍入分开记录。共同滞后分母固定在原state，确保与当前draw独立；线性SGD/固定分母均值用于核对。未满足其资源条件不运行，未做trace-matched shape干预不宣称shape唯一机制。

实施及自审后才运行；目前尚未运行N3。
