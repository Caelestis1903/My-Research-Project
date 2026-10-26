# C2.1 实现与 SELF_REVIEW（主 agent）

设计先保存于 `c2_numeric_design_spec.md`。新增实现 `experiments/c2_numeric_oracle_20261016.py`，未改B1冻结worker、调度器、旧训练源码或控制文件。此为数值oracle，不是新训练或方法实现。

复核：每数据集×4 seeds独立初始化并记录真实模型hash；每域16row只来自训练split，记录row IDs和全split/schema。前向固定eval且禁止BatchNorm/Dropout，要求架构sigmoid形式；还原仅在此形式下无输出作用的tau属性。用实际autograd取每row任务梯度，对固定θ进行有限总体枚举，并与整批目标autograd交叉对照。

α/q应用于固定分母样本均值，Equal和Sample目标分别核验所有参数期望。协方差矩阵只覆盖确定规则选择的C/P坐标及全A坐标，另用每row全梯度平方范数核算分区全坐标trace；不声称计算了完整参数两两covariance。缺域batch-cell normalized明确除以3且空域为0，区分原旧sum口径6倍尺度。

Adam oracle只创建detached坐标参数副本和fresh step0状态，原M3OE参数不更新。每个配置抽取3个MC梯度及期望梯度与实际torch.Adam作double数值复核；不混合异轨迹状态，不把E[delta]与delta(E[g])差异写作训练收益。全过程保存原模型hash及param.grad=None不变量，末尾检查数据与源码hash。

正式RUN之前执行语法基本检查。允许CPU执行此有限数值检查；结果只有在实际8/8 cases、期望/MC/Adam不变量通过后才能称oracle完成。若失败，保留case目录和诊断，不自动覆盖/重试。C2训练、C1 probe、B2和总目标不自动获得GO；独立review仍未满足，按用户最新要求使用主agent自审并披露限制。
