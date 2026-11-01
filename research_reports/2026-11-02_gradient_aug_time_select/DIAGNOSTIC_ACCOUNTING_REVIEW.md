# 扰动能量计量口径独立复核

复核时间：2026-11-02 02:42（Asia/Shanghai）。对象为运行目录 `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005`，执行代码 `ac2ec5cc70be7dcb880a7dde1015e4634726fd96`，manifest `ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`。本次只读取源码、计量字段及日志schema，不评价AUC/LogLoss结果，不修改源码、原始日志或此前REVIEW_REPORT。

**分类：Relevant——诊断计量与展示缺口。应修正派生表口径，不需要为此追加完整训练。**

## 字段事实

`run.py:30–34` 将 `mask_fields`、`perturbation_energy` 初始化为0；`run.py:69–73` 仅在 `info['mask_record']` 存在时累计。`objectives.py:65–69` 只有B1/R1进入 `masked_losses` 并返回该记录，A1/B2/B3走原 `source_losses`。R2/R3也沿普通Gaussian分支。因此普通增强臂的这两个原始0是未进入mask计量通道的占位值，不能读成实际Gaussian扰动为零。

| 臂/字段 | 原始字段的实际语义 | 派生展示 |
|---|---|---|
| A0 perturbation_energy | 增强关闭、没有view，能量可由结构判定为0 | 0，标注structural_zero |
| A0 mask_fields | 没有字段mask机制 | N/A |
| A1/B2/B3及同分支R2/R3 perturbation_energy | 未计量实际扰动能量 | N/A，标注not_measured |
| A1/B2/B3及同分支R2/R3 mask_fields | 未采用字段选择mask；普通Gaussian作用于全部允许字段 | N/A；不得解释为零字段被扰动 |
| B1/R1 mask_fields与perturbation_energy | 实测选中字段数与实际输入扰动平方能量之和 | 保留数值，并注明pass范围 |

读取时，已完成A1/B2的runtime记录均同时存在非零 `extra_views`/`view_draws` 与0 `perturbation_energy`/`mask_fields`。28份 `view_stats.csv` 的schema只有epoch/batch/step、view_key/seed/hash及target_rows；没有实际embedding RMS或扰动能量。8份B1 primary/replay `mask_stats.csv` 则包含counts、energies、embedding_rms、energy_scale、target_rows和mask身份。普通臂的batch/epoch/projection日志也没有另一个实际输入扰动能量列。根 `costs.csv` 在本次读取时尚未生成，其汇总代码沿用上述计数器。

`mask_stats.embedding_rms` 是扰动前clean embedding的尺度统计；`energies` 是实际施加的 `κ × clean_RMS × Gaussian × mask` 的平方和，不乘目标混合系数λ。扰动自身RMS可由能量和明确的元素数分母推导。总runtime计数会合计primary/replay两次实际计算；研究同一online轨迹的剂量时应使用primary日志，避免把重算计成另一个独立剂量。

## 对本轮比较的影响

这些计数器不反馈loss、gradient、Adam、shadow、checkpoint选择或门槛。缺少普通臂能量日志没有改变实际训练与AUC/LogLoss效用比较，也没有改变真实前反向/曝光/时间成本。

B1/R1所需的数量与能量匹配直接读取B1逐step/field的实测 `mask_records`；R1在自身训练分支按这些值匹配，不依赖A1的占位0。该对照检验难度位置关联时，所需计量路径完整。B1相对A1的比较则同时改变字段覆盖与扰动总量，不能写成与A1实测能量相等。

B2/R2的固定减量对照由训练期投影前后任务/参数组梯度范数及其冻结比例q定义。缺少输入embedding能量不改变这个对照；q不能被称为实测输入扰动能量匹配比例。

B3/R3登记的等剂量是累计epoch混合权重 `Σλ=25`、相同启用epoch数和原训练协议。各轨迹embedding状态会变化，现有日志未测得其累计实际输入扰动能量，报告只声明注册的λ预算相同。

C2比较同一online轨迹的不同shadow；确定性回放有逐step参数hash、view hash和行序核验。普通臂未记录能量不改变该同轨迹时间平均比较。

## 最小处理

保留不可变执行代码和原始日志。在最终派生成本/诊断表中增加计量状态，按上表将普通增强臂占位0显示为未测量/N/A，A0结构性零单列；也可将原列明确命名为mask分支计量。报告说明B1/R1实测匹配、R2梯度范数比例和B3累计λ预算各自的定义，不用占位0推断无扰动或能量等价。

普通Gaussian的key/seed/hash可重建标准噪声，但不能提供训练中每一步变化的embedding RMS。当前效用比较与注册匹配控制不要求为此重算整条训练；本次不追加训练、消融或源码修改。
