# 03｜首轮实验矩阵与 Codex 实施规范

**本文件是首轮实验的执行入口。** 本轮仅完成规划，不启动训练。实施时先复核当前 main 源码和实验配置，再按本文件执行。

## 1. 首轮实验要回答的决策问题

第一轮优先选择 A：**分离 Sample 下的统一损失缩放与增强收益**，随后用低成本机制探针判断是否值得进入真实/增强梯度分流。不是直接开展完整三因素全排列。

原因：
- 10-21 D 和 Dconst01 的结果无法将增强与总尺度分离；
- 10-21 使用 legacy_group_sum，不能直接推断 Sample 下结果；
- 若在 Sample 下增强本身没有增量，直接开发双流梯度和独立 Adam moments 没有投入价值；
- 若增强收益存在但由统一缩放解释，则研究重点应转为优化器尺度/正则化交互，而不是增强梯度质量。

## 2. 固定研究底座

### 2.1 数据与模型

- 主模型：仓库当前 fixed-fusion M3oE 研究底座；禁止恢复 learned fusion、AutoML/DARTS 附加更新。
- 数据：真实自然 MovieLens-1M 与 KuaiRand-Pure，使用仓库已审计的数据 split 和人口。首轮不重新构造人工 MovieLens 子集。
- 种子：正式比较 2022、2023、2024、2025，配对种子。
- optimizer：先使用当前主实验底座的 Adam，默认 betas=(0.9,0.999)、eps 和 weight_decay 必须从实际代码读取并写入 manifest。不得在不知道现有配置时擅自切换 Adam/AdamW。
- batch、训练 epoch 上限、patience、validation-selected 规则、初始化与数据顺序保持所有方法臂一致；从当前 canonical Sample 实验配置读取，不能照搬 10-21 legacy_group_sum 配置。
- Sample objective：
[
L_{S}=rac1Tsum_tsum_d p_darell_{d,t}.
]
代码必须验证有效域系数为 p_d，而非 1/D；记录每域人口、每步实际 domain-task count 和有效 loss coefficient。
- 验证集选模：六个 domain-task cell 的 macro AUC，严格提升时更新，平局取最早（沿用当前主线规范）。测试集仅在 checkpoint 冻结后评估一次。

### 2.2 必须明确的增强语义

首轮采用一个可识别身份的轻量增强操作，不恢复旧 VUG/mask：
- 对连续/embedding 表示添加小幅高斯扰动；
- 明确排除 domain ID、task ID、离散路由索引和任何决定样本归属的字段；
- 首轮噪声只作用于可扰动的内容 embedding，不能照搬 10-21 将 domain embedding 一并扰动的实现；
- 同一个真实样本和增强视图保持标签、域和任务一致；
- augmentation strength 在开发检查后冻结，正式四种子不调参；
- 同时记录增强前后 embedding norm、预测差、每任务 loss、样本级扰动强度。

若现有 M3oE 输入无法区分内容 embedding 与 domain embedding，Codex 必须先定位实际 embedding 拼接位置并给出最小实现方案；不得用整段展平 embedding 加噪声来冒充语义保持增强。

## 3. 最小方法矩阵

### Stage 0：低成本身份与机制检查

| ID | Sample loss | 增强 | 总损失缩放 | optimizer | 目的 |
|---|---|---|---|---|---|
| P0 | 原始 Sample | 关 | 1.0 | Adam | canonical baseline |
| P1 | 原始 Sample | 关 | 0.1 | Adam | 识别统一尺度效应 |
| P2 | 原始 Sample | 开 | 1.0 | Adam | 增强净增量 |
| P3 | 原始 Sample | 开 | 0.1 | Adam | 增强×尺度交互 |

这是 2×2 最小设计。不能把 P2−P0 直接解释为增强效果而忽略 P3/P1；需报告差分中的差分：
[
I_{aug	imes scale}=(P3-P1)-(P2-P0).
]
对 AUC 与 LogLoss 分别计算，并按 dataset、seed、cell 报告。P1/P3 的 0.1 是对完整 Sample loss 的统一乘性缩放，不是重新分配域权重。

### Stage 1：机制探针（不增加完整训练种子数）

在相同 θ、相同真实 batch、相同增强随机种子下，分别计算：
- 真实梯度 g_r；
- 增强视图对应的梯度 g_a；
- 两者混合梯度；
- shared-only、all-parameter 的 cosine 和 norm；
- 使用当前 Adam state 将各梯度映射成反事实 effective update，比较 update cosine/norm；
- 一阶矩和二阶矩在真实/增强两种输入下的变化；
- 同一参数点上真实 validation minibatch loss 的短步变化（仅机制探针，不当作正式性能指标）。

重点统计：
[
\cos(g_r^{sh},g_a^{sh}),quad
\|g_a^{sh}\|/\|g_r^{sh}\|,quad
\cos(\Delta_r^{sh},\Delta_{mix}^{sh}).
]
按 epoch 早/中/selected 附近固定三个锚点采集，不做每一步全量双 backward。每个数据集先选一个固定 seed 做开发探针，再以四 seed 正式运行中少量固定锚点复核。

### Stage 2：只有 Stage 0 增强有效时才执行

| ID | 更新方式 | 增强梯度处理 | 目的 |
|---|---|---|---|
| Q0 | 普通 Adam | 真实/增强先按固定 λ 混合 | 强简单基线 |
| Q1 | SAGU fixed | 固定比例混合 | 检查控制机制是否必要 |
| Q2 | SAGU gate | shared-only cosine gate | 检查简单动态门控增量 |
| Q3 | SAGU projection | shared-only 平行/正交分量控制 | 检查方向增量是否有用 |

如果 Q0 与 Q1/Q2/Q3 相当，停止发展复杂调节。若 Q2/Q3 在至少两个数据集或多个配对 seed 上有一致 selected 增量，才进入 SAOS 独立 moments 对照。SAOS 不应与 SAGU 在首轮同时叠加。

## 4. 实验预算与执行顺序

1. 先审计代码与配置，做一个 batch 的 smoke test，确认 P0 的 loss 数值和梯度与 canonical Sample 实现一致。
2. Stage 0 先用一个开发 seed 做单 epoch/短训练检查，只检查运行语义和日志，不根据单种子 test 选择方法。
3. 冻结增强强度、数据处理、方法臂和训练预算后，正式执行两数据集 × 4 方法臂 × 4 seeds，共 32 次训练。
4. Stage 1 复用正式 run 的指定 checkpoint / batch 做低成本机制探针；若额外 backward 成本可接受，最多对每个数据集 2–3 个固定训练阶段做测量。
5. 只有 Stage 0 的 P2−P0 或交互结果显示增强在 Sample 下有可重复 selected 增量时，才执行 Stage 2；首轮不预先承诺完整 Q0–Q3 全矩阵。
6. 正式结果不得由单种子或 final epoch 决定。原则上四个配对种子；若出现强而稳定的配对差异，后续再扩展，不为追求形式而无止境增加种子。

## 5. 评价指标与机制记录

### 性能主指标
- validation-selected test macro AUC（主指标）；
- macro LogLoss；
- 六个 domain-task cell 的 AUC/LogLoss；
- 每域、每任务 AUC/LogLoss；
- worst-cell AUC；
- pooled AUC；
- 样本人口加权 cell 指标（与 pooled AUC 分开）；
- selected epoch、固定共同 epoch、final epoch；
- train/validation loss 曲线与过拟合间隔。

### 机制指标
- 原始 Sample loss、缩放后 loss、每个 cell 的有效系数；
- 真实/增强梯度 norm、cosine、正交分量 norm；
- shared-only 与 all-parameter 梯度统计；
- Adam m/v norm、更新 norm、effective update cosine；
- 真实/增强样本数量、复制/曝光次数、增强扰动统计；
- 每步耗时、总 wall time、peak CUDA memory；
- checkpoint、optimizer state、RNG state、loader state 的可恢复性审计。

不得仅报告宏 AUC。必须识别大域/小域和任务间收益交换，且不能将域规模直接解释为困难度。

## 6. 结果驱动决策树

| 结果 | 结论 | 下一步 |
|---|---|---|
| P2≈P0，P3≈P1 | Sample 下增强没有独立增量 | 停止 SAGU/SAOS；检查增强是否失真或回到更合适的增强目标 |
| P1 优于 P0，P3−P1≈P2−P0 | 统一缩放解释主要收益 | 研究 optimizer/weight decay/epsilon 尺度交互，不宣称增强创新 |
| P2 优于 P0，且 P3 优于 P1 | 增强有独立收益，缩放非唯一解释 | 启动 Stage 1，检查梯度几何与参数块 |
| P2 有收益但 pooled 或某些大域明显下降 | 存在指标/群体 trade-off | 查明目标指标是否符合研究目标；不直接调大增强权重 |
| 增强梯度与真实梯度高一致，但性能无增益 | 增强可能只是重复信息或加大步长 | 固定 λ 与普通 Adam 对照；不发展门控 |
| cosine 低/负但真实验证收益为正 | 方向冲突不是负收益的充分条件 | 暂不使用 cosine gate；考察正则化/一致性解释 |
| cosine 与真实验证收益在多个阶段、seed 同向 | 几何量有预测价值的初步证据 | Q2/Q3 小规模正式验证 |
| SAGU 有增量但普通混合 Adam 不足 | 可能存在 state contamination / history value | 进入 SAOS，比较 shared moments 与统一 Adam |
| 只改善 late/final，不改善 selected-best | 主要是训练轨迹/过拟合缓解 | 不作为性能方法 GO；仅在明确部署训练预算时单独评估 |
| 只有一个数据集/少数 seed 正向 | 条件性结果 | 保留适用边界，不宣称通用 MDMT 方法 |

## 7. Codex 实施规范

### 7.1 修改前必须提交的审计清单

Codex 开始编码前先报告：
1. 当前 Git branch、commit、working tree 状态；
2. 实际 M3oE model/trainer/optimizer 路径；
3. Sample objective 的源码位置与逐项公式；
4. 域采样概率 p_d、实际 batch q_d、loss coefficient；
5. optimizer 类型、betas、eps、weight_decay、scheduler；
6. 当前参数分组：shared/domain/task/tower；
7. 当前自然 MovieLens、KuaiRand 数据版本、split、seed；
8. canonical checkpoint selection 规则；
9. 现有增强入口及其具体作用张量；
10. 当前代码是否存在 AMP、梯度裁剪、梯度累积、额外 optimizer。

如果源码与本计划不一致，先指出差异并给出最小兼容修改，不得默默改变训练语义。

### 7.2 组件开关

必须独立开关：
- augmentation_enabled；
- loss_scale（默认 1.0）；
- gradient_probe_enabled；
- gradient_mix_enabled；
- gradient_gate_enabled；
- shared_only_control；
- optimizer_state_split_enabled（首轮默认 false）。

关闭全部新组件时，必须逐位或在合理确定性容差内复现 canonical Sample baseline。gradient_probe_enabled 只读，不得改变梯度、optimizer state 或 RNG 消耗顺序。

### 7.3 实现要求

- 不恢复 learned fusion、AutoML/DARTS、旧 VUG/mask 未确认代码。
- 不将 Sample 改成 Equal 或 legacy_group_sum。
- 不改变域采样策略，除非明确实现 q 与 p/q 并记录。
- 增强操作不得扰动 domain ID、task ID 或离散路由键。
- 真实和增强分支应共享同一初始化、标签、域任务标记和样本身份。
- 如用两个 forward，控制 dropout/RNG；如拼接 batch，确保 loss 拆分后与独立分支数学等价。
- 梯度混合前检查梯度是否为 None；不得把 None 自动写成零 Tensor 而不审计其对 Adam weight decay/state 的影响。
- 若实现 SAOS，必须完整定义 bias correction、step counter、m/v 更新、缺失梯度处理和 checkpoint 恢复；不能仅写伪代码后交由实现者自行解释。
- 记录每个参数块的 raw gradient 与 effective Adam update；不把 raw gradient balancing 等同于 update balancing。

### 7.4 必须保存的工件

每次 run 保存：
- resolved_config.json、git_commit.txt、data_manifest.json；
- seed/RNG/loader state；
- model checkpoint 与 optimizer state；
- epoch_metrics.csv、per_cell_metrics.csv；
- augmentation_stats.csv；
- gradient_probe.csv；
- optimizer_state_stats.csv；
- runtime_stats.json；
- stdout/stderr 和完整 run manifest。

实验目录使用新 ID，禁止覆盖 10-21 历史输出和现有 S0–S8 报告。

### 7.5 停止条件

- Stage 0 在自然 MovieLens 与 KuaiRand 均未观察到增强独立 selected 增量：停止发展 SAGU/SAOS，除非存在明确且可重复的目标域收益。
- 增强只降低 LogLoss 而 AUC 不变：保留概率校准解释，不宣称排序收益。
- 增强收益完全由 0.1 loss scaling 解释：转向尺度/正则化研究。
- SAGU 的复杂门控未超过固定 λ：停止门控分支。
- SAOS 未超过普通 Adam + fixed mixing / 统一 EMA：停止独立状态设计。
- 任何方法只在 late/final 改善而 selected-best 不改善：不通过性能方法 GO。

## 8. 最终 Codex 交付物

执行完成后要求 Codex 交付：
1. 代码差异与文件路径；
2. 每个组件的开关验证和 baseline identity test；
3. 方法臂、seed、dataset、commit 对应表；
4. 所有性能主表与逐 seed 配对差；
5. 梯度与 optimizer state 机制统计；
6. 失败/中断 run ledger；
7. 对应结果驱动决策树的逐项裁决；
8. 不覆盖历史报告的阶段报告。

本计划只规定首轮研究路线和实施语义，不授权 Codex 自动扩大算法范围或开展无预算的全排列搜索。