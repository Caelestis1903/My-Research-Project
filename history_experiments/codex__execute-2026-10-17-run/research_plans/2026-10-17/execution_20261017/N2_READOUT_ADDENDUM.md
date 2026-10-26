# N2固定点读数规格（正式固定点数据产生前）

主agent设计；遵守start§11，不派发科学工作。此为原N2.3/N3.5的必要汇总，不新增训练、干预、门槛或选择规则。模型训练文件保持冻结，test保持封存。

## 输入与身份

正式N2 manifest的16锚：2032–2035×SN/SB路径×E10/E40。只接受COMPLETE、engineering=false、每臂512draw；检查checkpoint/config/source SHA，四臂与坐标集合齐全。不同theta、阶段、路径或batch不拼为同一covariance。

## 输出

1. 完整四臂×坐标块原始表，保留总trace、reference-centered energy及其MC SE、域内/计数预测、均值偏差范数和向量均值MC RMS。
2. 每锚/块的quota−IID、SB−SN以及两者交互；另列计数项占比、empirical/predicted trace。有限MC不要求样本均值完全相等，不用偏差范数强设通过阈值。
3. 从原始向量矩计算同锚q差的梯度均值范数；Adam同时报告平均实际映射/固定前v分母映射差及各臂Jensen残差。均值差的MC RMS上界使用两臂边际RMS之和（允许共同随机数相关），仅作噪声尺度，不当置信界。SGD仅报告由线性恒等式推导的lr倍梯度均值差，明确不是新增native实验。
4. 以路径×阶段分别汇总4个训练seed，保留均值/标准差/min/max及原始seed表，不跨路径/阶段抵消，不将draw数当训练样本数。正范数有MC偏差，不从其非零直接宣称均值漂移。
5. receipt记录全部输入SHA和读数代码SHA；大型向量仍在本地，只纳入索引。

## 解释边界

配额保留目标和预期曝光，但同时改变高阶矩/批内依赖。trace分解阳性不证明trace唯一中介；Adam瞬时映射差异不证明历史累积必要性或长期损害。固定点读数与完整训练的gap交互分别判断。所有实际科学结论必须等待正式结果。

## 验收/顺序

DESIGN先于IMPLEMENT；完成实现后主agent对照上述输入/公式/输出自审（independent_review=false），做语法检查。无需增加工程训练；正式16锚结束后才执行读数，核对完整性和真实输出，再分析。
