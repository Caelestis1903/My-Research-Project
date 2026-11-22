# 指标、统计、选择与数据隔离

## 1. 主终点与选择

每个dataset/arm/seed完成50epoch，分别跟踪online和fixed_ema。主读出固定为fixed_ema：

$$
e_s^*=\min\operatorname*{arg\,max}_{e=1,\ldots,50}
\frac16\sum_{d=0}^2\sum_{t=0}^1 AUC^{Val}_{s,e,d,t}.
$$

使用严格大于才更新最优值；平局保留最早epoch。六格LL、Macro LL与该AUC选中的同一模型绑定，不另选LL最好epoch。E10/E50及B的阶段锚点是次级诊断，不能替代主selected结果。online辅助读出可独立按相同规则选点，但不得混用online对照评价一个新EMA组件。

超参数和方法去留使用五seed Validation均值，不能读Test选λ、η、作用组、样本对、epoch或训练预算。同一探索过程中暴露过的Validation仍是开发证据，不称为全新独立确认。

## 2. 六格和总体指标

每格AUC使用该域该任务所有有效评价样本；固定预测分数与源evaluate一致。六格Macro AUC/LL是算术平均；task/domain聚合仅辅助。

Traffic AUC = Σ_c n_c AUC_c / Σ_c n_c，是cell AUC按样本数加权；不是pooled AUC。Pooled按任务分别把三个域预测合并计算AUC，再报告两任务均值，必要时沿用源文件提供的all-entry pooled并明确标签口径。没有原预测支持的指标记NA，不由六格均值猜测pooled。

LogLoss计算沿用源evaluate的概率裁剪和dtype，记录实现身份。均值下降单独标为LL改善，不作为AUC晋级的附加硬门，也不以LL改善替代AUC。

## 3. 五seed配对统计

所有正式seed严格为2022–2026。对每个比较、数据、格和指标：

$$
\Delta_s=M_{\mathrm{method},s}-M_{\mathrm{control},s},
\quad \bar\Delta=\frac15\sum_s\Delta_s.
$$

报告双方绝对均值、SD(ddof=1)、每seed差、均值差、正/负/零个数、最差格。描述性95%配对t区间为
$\bar\Delta\pm2.776445\,SD(\Delta)/\sqrt5$；同时提供原始五个差，避免只展示宽区间。

统计单位为seed；不能将epoch、六格、40锚点、随机分组replicate或online/EMA当额外n。重放同seed同配置不增加独立n，重试只用于恢复正确轨迹，不能挑性能最好attempt。

正式性能门不要求所有seed为正，也不加“95%CI必须全正”的硬门。若开展多比较描述性检验，预注册比较族并报告调整方式；本轮不以探索性p值替代用户均值阈值。

## 4. 三类增量必须同时展示

1. **总收益：** 最终方法−冻结的最强已发表模型。
2. **组件增量：** 相同背景/λ/覆盖/历史策略下，方法−强内部对照或匹配普通Adam。
3. **交互：** 仅在完整对应2×2存在时，计算AB−A−B+control；没有相同轨迹身份就不拼造交互。

A20−A10的五seed完整比较识别条件历史更新包；A20−A21进一步检查条件分离。A40的匹配普通Adam是A41，不能用A10(.05)代替A41(.025)。Gaussian背景组件比较使用同覆盖Gaussian+F，不能减clean+F后声称整个差都是新优化机制。

## 5. 算法资格

对每个数据集：

- 对最强已发表模型ΔMacro AUC≥.003且六格各自平均Δ≥−.001：强总结果。
- Δ≥.002且保护满足：候选级总结果。
- 宏达标但任一格<−.001：宏收益成立、逐格保护未满足。
- 新组件独立Δ同时单列；相对强内部参照没有新增AUC时，不把一个包含EMA的强总结果称为新增算法贡献。

最终新算法还应满足相对强内部对照的六格保护；该参照在冻结前登记，不能事后换弱参照。近零独立增量的解释依其数值/不确定性，不用通过发表模型总门来救援。

方法优选时的.0001简化带只用于采用更简单实现，不是正式的统计等效证明。A/B机制不受+.003强算法门约束。

## 6. B 面板与排序变化

训练控制面板P_A只来自train，固定row IDs和正负pair IDs；B报告面板P_V来自Validation且不与P_A混用。面板只是原split内固定抽样，不重新改变数据划分。

对每格固定正负对，定义
$I_\theta(i,j)=1[p_\theta(i)>p_\theta(j)]+.5\,1[p_\theta(i)=p_\theta(j)]$。
ΔQ = mean(I_method−I_control)。记录纠正、破坏、平局进入/离开及净变化；存在ties时使用上述差的精确平均，不能只用严格翻转数推导全部AUC差。

固定pair指标用于低成本动作评价，主selected AUC仍使用完整Validation/Test。panel Q是有限样本估计；同一批干预共享同pair有助于配对，但不把pair数量当训练重复数。

B每seed先在预注册的锚点/阶段内平均，再跨五seed汇总。按训练规则选q以后，在Validation评价该选择的效用；不能对Validation六个LOO挑最大的数后宣称训练期可用。随机分组选择预算和真源一致：都六选一+NO_ACTION；第二随机分组是重复控制，不把它并进12选一增加伪策略搜索能力。

## 7. 三种隔离

- **数据隔离：** Test在最终冻结前不可用于新动作、模型、HPO、阈值或范围选择。
- **时间隔离：** 早锚点/当前train panel决定规则，晚锚点Validation评价；禁止读取未来面板来倒推当前动作。
- **seed隔离：** 固定源/跨轨迹统计用留一seed法；评价seed s的q_static只能从其他四个seed的训练面板汇总。完整最终训练仍使用五个规定seed，这不是另开训练seed。

P_A若用作控制器输入可以随模型更新重新前向，但面板IDs和抽样方案事先冻结；不反向传播面板loss，不把它变成额外训练目标。记录额外前向和标签查询成本。

## 8. 历史Test与本轮冻结

P01、P00两seed、Gaussian和论文表的历史Test已暴露。它们可提出本次研究问题、复用数值和列明已知代价；不得用历史Test的具体坏格定制新门控或选择λ。

本轮FINAL_FREEZE锁定主表行和比较后，只评价必要新状态；已存在端点不重复读出，除非指标身份缺失且确有判断需要。同一Test人口上的后续结果标为历史数据集上的冻结比较，不能称独立新人口确认。

若A未晋级，仍完整报告五seedValidation归因以及已有历史Test线索；不为了补齐表格给所有淘汰配置运行额外Test。只有进入最终候选阶段，才展开完整统一Test性能表。

## 9. 数据输出schema

- endpoints.csv：study_id、dataset、arm、seed、split、readout、checkpoint_rule、epoch、model_hash、cell、auc、logloss、support、source_ref、reuse_status。
- paired_summary.csv：comparison_id、dataset、split、readout、cell、metric、n、mean_left、mean_right、mean_delta、sd_delta、ci95_low/high、positive/negative/zero、guard。
- update_utility.csv：seed、anchor_step、source、group、action、panel、readout、delta_auc、delta_ll、gradient_norm、update_norm、norm_matched、optimizer_pre_hash、post_hash、none_mask_hash。
- costs.csv：cost_role、attempt、wall_seconds、worker_seconds、cuda_event_seconds、main_steps、virtual_steps、pulse_steps、forward_rows、autograd_sweeps、peak_allocated/reserved、process_residency、RAM_bytes。
- decisions.json：预注册门输入、触发状态、支持文件hash、解释与下一合法任务ID。

报告时必须保留source_ref和split/readout，避免把最新Validation值与旧Test值拼成同一性能表。
