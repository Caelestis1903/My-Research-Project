# 历史数值重新聚合及关键反例

本文件记录实际旧CSV/config/receipt的重新聚合，供后续自然结果逐项对应。**不是自然N19/CM19已完成报告。** 不声称旧重型checkpoint或预测已经重读。原KuaiRand自然、MovieLens人工人口分别保留，均不混入新自然结果。旧test暴露及单一人口优化seed的边界继续有效。

## 证据覆盖

`analysis/collect_transfer.py --historical-only`收集1184个旧run（N19 328、CM19 856），29856个test cell记录（含CM19冻结干预）。`analysis/analyze_transfer.py`按原候选与规定强控制形成747条完整四seed比较；配方见 [定义](analysis/transfer/comparison_definitions.json)。本次加列known-rule masking、静态CV、uniform容量、global WD0等关键控制，不增加训练，不按正负删方法。

原指标遵循原sklearn `log_loss`及保存dtype，和core的EPS=1e−12不是同一裁剪协议。禁止直接把跨族LL的绝对差当同口径方法优劣。[全部cell](analysis/transfer/all_cell_metrics.csv)、[配对汇总](analysis/transfer/paired_summary.csv)、[来源SHA](analysis/transfer/source_index.csv)均可复查。

S5/S6另从原用户工作目录的64个正式原CSV恢复，见 [原始索引](analysis/controlled_history/historical_source_index.csv)。新自然独立运行并不补回已丢失的旧初始化/optimizer bytes。

## N19：必须保留的候选/强控制/阶段冲突

下表均为人工MovieLens四seed，Δ=候选−控制，AUC/LL为实际任务cell宏均值；LL负值为改善。数值来自原test CSV重新聚合。

|方向、比较|阶段|Δ宏AUC|AUC正/负seed|Δ宏LL|LL改善seed|
|---|---|---:|---|---:|---:|
|P1 M3OE task-DRO−ERM|selected|+.188437|4/0|−.130544|4/4|
|P1 M3OE task-DRO−known-rule mask|selected|+.004101|4/0|−.004698|3/4|
|P1 MMoE task-DRO−known-rule mask|selected|+.001656|2/2|+.015694|0/4|
|P2 joint-axis−nested-axis|selected|−.002586|2/2|−.003804|4/4|
|P2 固定审核additive joint−independent joint，γ=1|selected|−.003919|0/4|+.014394|0/4|
|P3 rank-protected−uniform，Adam|selected|+.000016|3/1|−.000110|2/4|
|P3 rank-protected−uniform，Adam|final|+.002306|4/0|−.016017|3/4|
|P3 rank-protected−uniform，名义AdamW|selected|+.000055|3/1|−.000120|3/4|
|P3 task/cell字典−domain字典，Adam|selected|+.001762|3/1|−.004927|4/4|
|P3 task/cell字典−domain字典，名义AdamW|final|−.002376|1/3|+.023835|1/4|
|P5 regret−uniform-KL|selected|−.000197|2/2|+.001291|2/4|
|P5 regret−uniform-KL|final|+.007960|4/0|+.000261|2/4|
|P6 residual-CV−static antithetic-CV|selected|+.000896|3/1|−.000292|2/4|
|P6 residual-CV−static independent-CV|selected|−.000412|1/3|+.002754|0/4|
|P6 residual-CV−static independent-CV|final|−.003017|1/3|−.015682|3/4|

P1的+.188不能解释为native双任务方法收益：task1是Bernoulli(.2+.6Z)且有已知兼容映射的受控任务；相对known-rule mask大部分优势消失，MMoE的LL还变差。P2的γ=1反例限制正确加性噪声假设的推广；同一审核draw下四个优化seed不提供独立审核重复。P3/P5/P6都有选中/末轮或AUC/LL矛盾，不能只挑表中正数形成方法主张。

这里的稳定同号是配置内的 **MEDIUM** 经验支持；γ控制是对特定假设的受控反例，仍非真实噪声机制证明。新自然结果必须保持同样强控制和全部阴性，不只比较fixed。

## CM19：原路径记录与数值精度

实际重新收集856个run的69448个“run×检查点×探针×干预”汇总，另有逐cell端点、路径强度、真实gradient/update遥测、同状态局部干预与成本。见 [路径分析脚本](analysis/analyze_cm19_paths.py)、[状态](analysis/cm19_paths/analysis_status.json)、[原表SHA](analysis/cm19_paths/source_index.csv)。大的完整表以gzip保存，解压可恢复字节相同CSV，见 [压缩索引](analysis/cm19_paths/large_table_archives.csv)。

原`validation`探针来自development子集，最多512行/域；`meta`来自train-meta。名称不代表完整验证集。这些记录描述有限固定探针上的实现内中和干预，不是全函数、不可逆吸收态或重新训练的无模块模型。

### 1. 活跃/失活边界不是人口标签

- KR fixed Adam selected：四run的六cell modulation中和全部预测相同，即24/24cell为精确0。独立STAR、PEPNet、HAMUR及MMoE/PLE相应干预均有作用，是“乘性结构普遍失活”主张的反例。
- 同一KR fixed Adam final：3/4 run仍精确0；seed2023在2350步时出现极小差异，平均绝对概率差2.340587e−8、最大4.768372e−7，1651个row-task非零。selected为0不能推出以后绝对不可恢复，也不能把这个浮点级变化称有用恢复。
- 人工ML fixed Adam selected的平均modulation概率差约.002303（四run均非零），名义AdamW约.010101。两者仍都没有证明“越活跃越准确”。
- 旧KR与人工ML通常分别2350/500步，且特征、任务与人口不同。旧long ML和新自然匹配预算用于限制时钟解释；不能从native域数量独立推规模充分性。

以上为旧L2表的 **MEDIUM** 条件经验支持；新自然路径尚待实测，不能预写其方向。

### 2. 名义AdamW不能充当已经测过的正解耦收缩

对G0 global AdamW和G1 global Adam WD0的同model/seed对：

|人口|float32名义乘数|final模型canonical state SHA相同|
|---|---:|---:|
|原人工MovieLens|1.0|12/12|
|原自然KuaiRand|.9999999403953552|0/12|

来源：[逐对SHA/预算](analysis/cm19_paths/nominal_AdamW_vs_WD0_model_identities.csv)。这是训练receipt里的模型tensor canonical hash，不是包含不同optimizer状态的checkpoint文件hash。旧heavy tensors未重载；记录一致性的经验范围与float32可表示性的算术事实分开。

自然ML保留原lr=.001、WD=1e−5、float32/kernel，不改变精度或放大WD。其名义乘数仍舍入1；新实验会报告实际模型SHA和路径记录。**可表示正AdamW收缩的效果仍是未被这些名义配方回答的问题，不能把它们改称已检验后失败。**

### 3. 风险与活跃度分开

旧CM19人工ML fixed Adam：selected宏AUC/LL=.733443/.533803，final=.675340/1.165258；KR对应.666951/.442080与.648995/.487054。阶段风险变化很大，不能因路径变强就推全预算性能改善。

独立STAR task0/task1是单任务模型，必须比较fixed同一任务。汇总脚本已采用这一口径，禁止把STAR task1较低LL与fixed两任务宏LL作方法排名。joint与two-private还改变backbone数量；HAMUR的混合batch BN等实现边界保留。成本为共享主机上的有限shape dense profiler，不是生产加速证据。

## 对下一阶段分析的约束

所有自然候选按旧配方执行完整预算，先冻结后读test。科学判断须同时报告基线绝对表现、任务/域、目标与optimizer、实际step/曝光、selected/final、AUC/LL及seed差异。历史数学/数值身份保留为身份；有限探针机制不能替代端点竞争效果，端点效果也不能反向证明唯一机制。
