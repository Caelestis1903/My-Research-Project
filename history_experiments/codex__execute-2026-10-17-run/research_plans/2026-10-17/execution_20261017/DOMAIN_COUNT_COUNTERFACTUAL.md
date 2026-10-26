# 域计数反事实：固定点与完整quota训练

状态：COMPLETE。固定点16锚、新增quota16条H100及全局test均全部完成并验证。仅主agent设计/自审/分析，independent_review=false。

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

## 八seed全程性能：四臂绝对值

每条selected checkpoint由各自selection宏AUC选择；同一固定test评价，未由test调选。

| metric | SN_IID | SB_IID | SN_quota | SB_quota |
| --- | --- | --- | --- | --- |
| macro_auc | 0.667714 | 0.662511 | 0.667628 | 0.658437 |
| macro_logloss | 0.439815 | 0.447272 | 0.444782 | 0.450989 |
| train_p_cell_logloss | 0.418883 | 0.421958 | 0.424312 | 0.424244 |
| train_J | 0.487479 | 0.493750 | 0.486789 | 0.493548 |

宏AUC配对差及交互：

| effect | mean | t95 | pos_neg |
| --- | --- | --- | --- |
| IID_SB_minus_SN | -0.005203 | [-0.012006, +0.001600] | 1/7 |
| quota_SB_minus_SN | -0.009191 | [-0.015795, -0.002588] | 2/6 |
| quota_interaction | -0.003988 | [-0.010310, +0.002333] | 3/5 |
| SN_quota_minus_IID | -0.000085 | [-0.003289, +0.003118] | 4/4 |
| SB_quota_minus_IID | -0.004074 | [-0.010089, +0.001941] | 3/5 |

**MEDIUM（次级pointwise）**：quota下SB−SN仍为−.009191，t95不跨0、6/8负，不能说固定quota消除长期q差。I_quota均值−.003988但t95[−.010310,+.002333]跨0；没有充分证据确定quota扩大或缩小q差。SN quota−IID近0但区间[−.003289,+.003118]同时越过±.003，不能声称等效；SB quota−IID−.004074也跨0。

宏LL差/交互：

| effect | mean | t95 | pos_neg |
| --- | --- | --- | --- |
| IID_SB_minus_SN | +0.007457 | [+0.004913, +0.010002] | 8/0 |
| quota_SB_minus_SN | +0.006207 | [-0.000571, +0.012985] | 7/1 |
| quota_interaction | -0.001250 | [-0.006799, +0.004298] | 4/4 |
| SN_quota_minus_IID | +0.004967 | [-0.000469, +0.010404] | 7/1 |
| SB_quota_minus_IID | +0.003717 | [-0.002022, +0.009455] | 7/1 |

训练J q差IID+.006271、quota+.006760；I_quota的J差+.000488、CI跨0。train-p加权test LL的I_quota=−.003143，CI[−.012165,+.005880]亦跨0。所有SN/SB绝对quota lift均未通过预设宏收益和七项保护界，不把gap变化替代健康提升。完整六cell、所有固定轮/selected/AULC、原seed和保护界见N2_quota_performance与N2_absolute_quota_method_guards。

## 机制结论与竞争解释

固定点域计数主导解释被反驳，不等于自然域支持规模无作用；长期q差保留与小计数项相容。quota同时改变批内依赖、舍入与高阶矩，不能以单个interaction识别纯trace中介。当前没有识别Adam瞬时总体均值漂移、历史必要性或共享C唯一中介；N3独立局部门槛不满足，N4未触发。

域内梯度分布、累计路径、LR/预算及评价人口仍是竞争解释。三个域共变N、标签率、特征和baseline，未操纵自然难度；不能从方差比推导小域受益/大域受损。下一步应针对具体未排除解释预注册单一辨别实验，不能追踪新seed或重复bank直至阳性。
