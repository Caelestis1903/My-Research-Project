# 域计数反事实：固定点与完整quota训练

状态：固定点16锚及新增quota16条H100全部完成并验证；最终性能读数等待全局test解封，当前IN_PROGRESS。仅主agent设计/自审/分析，independent_review=false。

## 估计对象和实际处理

固定KR人口、Sample经验目标J、Adam共同eta=.01、B40960、WD1e−5。IID采样用p_d/q_d修正；quota按系统舍入生成n_d>0、域内WR，实际逐batch用B_t*p_d/n_d。quota固定域配额并改变批内依赖/高阶矩；不是纯scalar trace开关，也不改变自然域大小。两种估计器在冻结图上具有相同人口期望，batch公式不应要求字面相同。

固定点4个训练seed2032–2035×两原生路径×E10/E40，16条件锚，各四臂×512draw。锚参数、原生optimizer状态/step及draw bank冻结；路径/阶段/draw不是新训练seed。下表为四seed比值均值，字段历史minus名实际存trace比值，并非AUC差或trace相减：

| anchor_arm | epoch | n_training_seeds | IID_SB_minus_SN_ratio_mean | quota_SB_minus_SN_ratio_mean | SN_quota_minus_IID_ratio_mean | SB_quota_minus_IID_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- |
| SB | 10 | 4 | 2.478118 | 2.449725 | 1.014960 | 1.003729 |
| SB | 40 | 4 | 2.123074 | 2.101472 | 1.002827 | 0.993100 |
| SN | 10 | 4 | 2.505010 | 2.498725 | 1.003812 | 1.001487 |
| SN | 40 | 4 | 2.378294 | 2.374368 | 0.995438 | 0.993817 |

全部ALL锚的IID SB/SN trace约2.009–2.584，quota约1.995–2.551；计数项占IID预测trace仅.001297%–.105936%。**STRONG（仅这些条件固定点）**：固定域计数后，q间trace差保持，计数波动不能构成这些锚点方差差的主要解释。同bank内估计的域内协方差不作为独立oracle；零梯度块的比值保留未定义。跨块差异保留，不用权重二阶矩或ESS替代梯度trace。详见N2_FIXEDPOINT_ANALYSIS.md和N2_fixedpoint_summary原表。

同状态Adam q均值更新差仍未在现有MC分辨率下确定，固定滞后分母也未使对比范数明确消失；这不证明瞬时非线性漂移或历史必要性。单位eta仿射SGD只复用同bank矩，理论均值恒等与有限bank差异分开，不伪造测量值0。E40部分STAR块采样数据梯度0而Adam更新非零，限制逐阶段解释。

## 完整训练已经执行

按N1.6既定重要残差触发，新增16quota轨迹：2032–2039×SN/SB×H100；对应16IID父轨迹复用。N2_quota_validation.json核对16/16无失败、每条4900更新/200613200样本、每epoch49步、尾40052、完整曲线/指标/逐epoch模型。N2_QUOTA_IID_PAIR_IDENTITY.json再核对实际q配对与IID父的初始化、数据行、optimizer、LR、B、WD、fusion/detach及目标；每批记录的n_d*w_d/B_t=p_d，舍入偏差≤1。importance_weight配置明确只是IID参考，actual quota权重逐batch改变。

所有原始科学产物保留；原主队列62COMPLETE和2个原N2验证失败不改写。验证修正重试只替换身份检查，原失败/数值诊断与逐字估计器等价证明完整保留；16quota无失败且从未重复训练。恢复时实际最后quota结束时间2026-10-17T17:26:14UTC，见RESUME_20261018.json。

## 待全局冻结后的性能分析

八配对训练seed将报告四臂绝对AUC/LL/J、IID q差、quota q差、I_quota=(SB−SN)_quota−(SB−SN)_IID及两q各自quota−IID绝对变化。固定E10/20/50/100、selected和AULC1..20/100全部保留；次级pointwise不做唯一trace中介推断，不以gap接近0替代两臂健康。test当前封存，故此处无补造性能值。

竞争解释包括域内梯度分布、高阶矩/依赖、累计训练路径与学习进度。仅三个自然域不能识别规模/难度单因果；局部方差差不自动预测小域收益/大域损伤。与独立N3确认共同判断限定机制范围，不追加搜索矩阵。
