# 06｜评价、统计与阶段决策

## 1. 研究阶段评价对象

A0/A1机制分析的主单位为seed。pulse窗口、task、domain、group、panel、scope均为seed内重复观测，不能把样本量写成窗口数×seed数。报告每个seed的时间序列和seed级聚合值。

必须区分：
- 当前窗口 Fast−F/Slow−F收益；
- 下一窗口真实Fast−F/Slow−F收益；
- 对未来收益的预测误差；
- 用预测信号选择动作后的regret/净收益；
- 完整Validation/Test性能；
- 对固定EMA的组件增量；
- 对已发表模型的总收益。

不得将其中一个替代另一个。

## 2. A0/A1持续性指标

主统计包括：
1. lag-1符号一致率：同一seed/cell/group相邻窗口收益符号是否一致；
2. seed内lag-1 Spearman/Kendall及Pearson相关；
3. 当前收益对e+5未来收益的MAE、RMSE、相关；
4. P_A→P_B、P_A→V_A/V_B的方向迁移；
5. 动态动作相对固定F、统一Fast/Slow和身份/阶段参照的收益；
6. 各task三域聚合与六格结果；
7. 信号缺失率、动作切换次数、平均动作持续epoch和反馈延迟。

相关系数不能直接跨所有窗口拼接计算后把每个窗口当独立样本。先对seed内时间序列计算，再汇总5seed均值/SD和配对差。对同一scope的嵌套ACTION_DECISIONS行只按唯一scope/cell/group/window key保留一次。

## 3. 完整模型评价

每个算法配置、每个seed按Validation六格等权Macro AUC最早严格最大值选择checkpoint。不得用Test、LogLoss或单格结果选择epoch。Test仅在算法、超参数、训练配置和选点规则冻结后一次性运行。

正式报告：
- 六格AUC与LogLoss；
- 六格Macro AUC/LogLoss；
- Traffic AUC/LogLoss；
- Pooled AUC/LogLoss；
- 相对固定F的增量；
- 相对已发表最强模型的增量；若使用新seed 2027–2031而发表模型仅有2022–2026历史结果，必须标记为跨seed非配对上下文比较，不能套用配对五seed阈值。若要正式声称相对发表模型的配对收益，须在相同新seed上运行该发表模型的可比实现，或使用同seed已存在的正式模型端点；不得将非配对差包装成配对收益；
- selected epoch、共同E50、每seed配对差；
- 参数状态成本、推理成本与动作序列。

固定F是内部核心对照，不是发表模型基线。正式发表模型基线只使用已发表论文模型，例如M3oE、HyperGate、AutoIFS；ML同协议当前最强为M3oE，KR为HyperGate。需要明确模型工程适配及训练协议差异。

## 4. 五seed规则

至少五个独立seed才进行科学性能否决判断。无需每个seed都正向。正式模型比较的AUC收益标准：
- 五seed配对平均Macro AUC增益≥+0.003：强收益；
- ≥+0.002：候选收益；
- 任一域任务格五seed平均AUC下降不得超过0.001，作为逐格保护；
- LogLoss独立评价，五seed平均下降即报告LL改善，不以单seed方向否决；
- AUC与LogLoss分别裁定，不得把LL恶化抹去，也不能以LL未改善取消AUC合格。

对内部算法相对F的增量，不要求达到发表模型+0.003阈值，但必须给出实际幅度、seed配对分布和六格代价。若均值很小但跨seed稳定，也应准确报告为小幅效果，不得包装为强收益。

## 5. 阶段决策

### A0 → A1
- PERSISTENCE_SUPPORTED：跨seed、跨P_A/P_B具有前瞻方向信息，且优于简单身份/阶段参照，进入A1。
- STATIC_PREFERENCE_ONLY：没有相邻窗口持续性，但任务偏好跨seed、跨面板稳定；进入静态任务偏好替代，不运行动态控制器。
- NO_ACTIONABLE_SIGNAL：持续性和静态偏好均无可迁移信息；停止新增训练，固定F保留。
- SOURCE_INCOMPLETE：原始数据缺失或hash不符；仅恢复归档或离线重算，不能当阴性，也不能跳过审计直接进入A1。

### A1 → B
只有新seed中信号方向/预测性能能够跨P_A/P_B迁移，且至少一个任务或全局范围在五seed综合上优于固定身份/阶段参照，才进入闭环B。若仅单个cell出现小收益，不启动全矩阵控制；记录局部现象并保留固定F。

### B → C
至少一条A/B/C路径相对F有可重复的五seed验证收益，并且：
- 任务条件化路径需证明task0/1收益在其三个domain格中分布合理；
- 不得有单格平均AUC跌幅超过0.001而不报告；
- 信号迁移和动作决策在新seed上成立；
- 控制器不是只改善P_A训练探针。
满足后进入跨数据集C。

### 全部路径无收益
若A/B/C均未优于固定F，停止自适应控制；保留固定EMA的既有经验收益与滞后权衡结论。不得为了挽救方向再增加复杂指标、额外参数组或无边界HPO。

## 6. 超参数

- rho_F、Fast、Slow继承，不重搜。
- β收益平滑：0或0.5；仅在A0支持持续性后启用，按5seed综合验证。
- C λ：默认0与1−rho_F；若有足够依据，最多加2(1−rho_F)，限制λ≤0.1。
- 动作每epoch更新；不增加动作阈值参数。平局选择Fixed；信号缺失时沿用上次动作，首个信号前Fixed。
- 若需要阈值/滞回，先用无阈值argmax作为简单参照；只有动作抖动明显且会改变结论时才增加一个阈值，并按至少5seed搜索。

所有超参数的搜索只能在A1开发数据和Validation上进行；Test冻结后不再调参。
