# 机制分析与证据边界

状态：COMPLETE。按实际干预区分目标、采样、更新映射、局部作用与长期性能。

本轮固定Sample经验目标及原M3OE fixed_model计算图：SN的q=p，SB的q=(1/3,1/3,1/3)，IID用p/q修正；routing detach、固定fusion、B40960、WD1e−5保留。训练、调参、机制和评价估计对象分别报告。主不确定性单位是8个配对训练seed，N3每路径是4seed；多个q、task、epoch或draw均不增加独立n。历史test已被过去研究使用，本轮按全局冻结后统一读取，不称全新hidden holdout。

## 得到什么

1. **STRONG（算术/冻结对象）：** N0同预测评价权重可翻转符号，严格单调校准保持cell AUC而改善部分LL。新ML macro、train-p、test-p方向/量级继续不同；它们不是同一个评价函数。
2. **STRONG（条件固定点）：** IID计数项占ALL预测trace .001297%–.105936%，固定quota仍保持SB/SN约2倍；数据不支持这些锚的方差差主要由域计数波动产生。quota系统舍入只在相邻整数间变动，域权系数n_d*w_d/B恒为p_d。
3. **MEDIUM（轨迹与边界）：** KR Adam早期AULC8/8负；ML selected两种调参估计对象8/8负，own E100却8/8正。LR、阶段、metric及人口必须进入结论。
4. **MEDIUM（局部限定）：** 独立embedding full两路径各4/4正但仅约3e−5，方向残差未满足实际/一致性规则，四项MC精度达标。N4未触发。SGD整体C的许多full条件区间在±.0001内，但E40方向反例仍保留。

## 尚未识别什么

- **WEAK/INCONCLUSIVE：** 同状态Adam总体q均值漂移。有限bank范数有抽样偏差；MC RMSsum不是置信界。固定分母对比/native约.91–1.05，并未出现明确消失；不能从中断言历史必要性。
- **未识别：** C长期必要性、唯一方向中介、协方差shape优于trace、独立重复/覆盖中介、自然困难度与支持规模单因果。未触发控制不冒充阴性。
- N3局部J/LL改善与AUC很小共存；E40部分STAR数据梯度采样为零、原生Adam更新非零也不证明模块全程无用或某激活原因。
- 全程quota q差−.009191的次级CI不跨0，但I_quota=−.003988的CI跨0，两个单臂quota提升也未确认。quota改变依赖与高阶矩，不能当纯trace旋钮。

## 规模、难度与公平性

KR最大域两task负向、最小域也多负向；ML最小域大幅负而最大域略正。严格小域双赢/大域下降在KR Adam0/8、SGD1/8。相同域N下task可异向，baseline和标签率也不单调；没有自然难度等于小域的识别。ML固定subset只分离优化seed，不估计子集方差。KR H1004900更新/200613200样本，ML500/18750000；同epoch不表示同step或同计算量。

## 方法与最小下一步

没有比较同时通过宏收益和七项保护；ML own E100虽宏与六cell点均值改善，但小域保护区间不够精确，不能说已满足非劣，也不能说已证明有害。没有新方法候选授权、课程收益或普遍动量有效性证据。文献中importance sampling、EMS与MIS已覆盖直接通用思想。

最直接的证伪是下一轮ML固定人口的q×两LR完整独立矩阵，检验晚期优势能否在相同LR下存活；当前不追加实验。另以Q53登记SGD主区间横跨±.003的精度不足，不追加seed搜索阳性。

完整量化与原始数据入口见FINAL_ANALYSIS、N2_FIXEDPOINT_ANALYSIS、C_LOCAL_AND_TOPOLOGY_ANALYSIS及各summary/receipt。
