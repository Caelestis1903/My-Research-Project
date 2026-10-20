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

从328个旧run的非test机制记录再次核对（[原始记录分析](analysis/analyze_n19_records.py)、[逐cell审核支持](analysis/n19_records/audit_class_support.csv)、[rate RMSE](analysis/n19_records/audit_rate_RMSE.csv)）：人工ML γ=0 的实际审核行数按d0/d1/d2为670/61/2876，总3607行；两个任务都审核，不能把12个class计数总和7214误写成独立审核行数。最小域d1 task1的clean-positive仅13例。固定审核拟合的12个flip-rate宏RMSE，独立cell为.035074、additive axis为.027248；γ=1则为.034253/.074270，axis误设显著增大点估计误差。每个γ内四优化seed给出相同初始审核估计，不能当4次独立审核精度确认。审核类别支持和模型误设都是具体竞争解释，域总N并未被单独识别为主因。

P6实际每row/visit平均使用约1.1个标签。residual-CV的静态lower endpoint比例在四seed为97.7210%–99.7786%，均值98.7588%，见 [完整计数](analysis/n19_records/realized_mask_budgets.csv)。相对static independent-CV，selected AUC反而−.000412、LL四seed更差；不能只用相对antithetic的微小正均值说明自适应策略有竞争收益。计数描述重复访问的模拟观测，原标签静态可缓存，不是实际节省这么多独立标注费用。

## CM19：原路径记录与数值精度

实际重新收集856个run的69448个“run×检查点×探针×干预”汇总，另有逐cell端点、路径强度、真实gradient/update遥测、同状态局部干预与成本。见 [路径分析脚本](analysis/analyze_cm19_paths.py)、[状态](analysis/cm19_paths/analysis_status.json)、[原表SHA](analysis/cm19_paths/source_index.csv)。大的完整表以gzip保存，解压可恢复字节相同CSV，见 [压缩索引](analysis/cm19_paths/large_table_archives.csv)。

原`validation`探针来自development子集，最多512行/域；`meta`来自train-meta。名称不代表完整验证集。这些记录描述有限固定探针上的实现内中和干预，不是全函数、不可逆吸收态或重新训练的无模块模型。

### 1. 活跃/失活边界不是人口标签

- KR fixed Adam selected：四run的六cell modulation中和全部预测相同，即24/24cell为精确0。独立STAR、PEPNet、HAMUR及MMoE/PLE相应干预均有作用，是“乘性结构普遍失活”主张的反例。
- 同一KR fixed Adam final：3/4 run仍精确0；seed2023在2350步时出现极小差异，平均绝对概率差2.340587e−8、最大4.768372e−7，1651个row-task非零。selected为0不能推出以后绝对不可恢复，也不能把这个浮点级变化称有用恢复。
- 人工ML fixed Adam selected的平均modulation概率差约.002303（四run均非零），名义AdamW约.010101。两者仍都没有证明“越活跃越准确”。
- 旧KR与人工ML通常分别2350/500步，且特征、任务与人口不同。旧long ML和新自然匹配预算用于限制时钟解释；不能从native域数量独立推规模充分性。

第一步参数空间L2向量/数据梯度范数比也不能单独解释跨模型失活。KR fixed的shared/slot四seed均值为7.424/12.864，而仍活跃的独立STAR task0为23.419/39.589、task1为12.744/21.527。每个STAR任务的两组比值在全部8个“seed×组”上均高于配对fixed。该跨结构反例不排除L2与尺度/归一化/旁路的交互；它反对仅凭第一步某个比值阈值判定必然失活。见 [逐seed上下文](analysis/cm19_paths/first_step_L2_gradient_context.csv)；真实group optimizer类保存在`group_optimizer_class`，与run名义optimizer分列。

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

## B1重建：自然KR的主规则阴性与敏感度阳性均保留

重新读取原重建的1,360条selected双锚点分解记录和13,600条同epoch记录，按四seed重新汇总340项选点指标、340项阶段指标，核对两种timing分解恒等式；原始表按SHA保存，大表以确定性gzip保留。见[来源](analysis/b1_history/source_index.csv)、[选点结果](analysis/b1_history/selected_rule_seed_summary.csv)、[分阶段结果](analysis/b1_history/four_seed_stage_summary.csv)。本次没有重载旧模型/预测，也没有新增训练。

它是自然KR、Sample目标、F/DH×4seed的`RECONSTRUCTION_NOT_HISTORICAL_EXACT`，target物理域d1；旧完整lineage和初始化不匹配，不能与旧S6-R2当作同一执行身份。本次MovieLens人口审计不直接否定这些KR观测。

|原冻结规则|test Δ宏AUC|test Δnon-target AUC|test Δtarget AUC|test Δ宏LL|test Δtarget LL|
|---|---:|---:|---:|---:|---:|
|R0 validation macro（主规则）|+.000363（2/4正）|+.000579（2/4正）|−.000069（1/4正）|+.001053（3/4差）|+.004007（3/4差）|
|R1 target validation|+.007856（4/4正）|+.011584（4/4正）|+.000399（2/4正）|−.000331（3/4改善）|+.006006（3/4差）|
|R2 non-target validation|+.000464（2/4正）|+.001086（2/4正）|−.000781（1/4正）|−.000867（3/4改善）|+.000933（3/4差）|
|R3 worst-domain validation|+.002152（3/4正）|+.004030（3/4正）|−.001602（3/4正）|+.003045（3/4差）|+.007349（3/4差）|
|R4 validation行频率cell AUC|与R1完全相同|与R1完全相同|与R1完全相同|与R1完全相同|与R1完全相同|

R1与R4全部八个模型选中相同epoch，不能记作两次独立复现；R3 target均值为负但3/4为正，种子幅度不能由符号计数替代。R0没有复现旧S6-R2 non-target test+.006757、4/4正的组合，但历史初始化/lineage缺口阻止将差别唯一归因旧错误或dense saving。主阴性不抹去R1的非目标收益线索，敏感度阳性也不替换原主规则。**MEDIUM**：明确的选点/阶段依赖；泛化方法收益 **WEAK**。

同epoch test non-target AUC差在完成E31–40/E41–50为+.008031/+.007009，均3/4正；末轮seed2024为−.015648，阻止稳定末轮收益解释。target test AUC阶段均值在E1–10四正、E11–20四负。2022的R0两臂同epoch仍有val/test反号，说明timing不是全部解释；轨迹分解是算术恒等式，未进行同theta因果分离。

这些旧KR问题继续保留，包括选择规则、窗口支持、target LL代价和独立后置总体。新自然S6按本轮预定R0宏验证选模提供另一人口的负边界参照，不能声称它已复现或否定未迁移的所有R1–R4规则，也不依据旧回顾性test阳性扩大新训练矩阵。

## N17：八优化种子的历史边界与局部反例

依据原`FINAL_performance/paired_seed_endpoints.csv`的12,768条记录重新计算全部指标、种子均值及点态t95，并由原三份最终表的371个逐seed数组复核均值、SE、区间与正负计数。保存[完整重新汇总](analysis/n17_history/all_endpoint_seed_summary.csv)、[数组复算](analysis/n17_history/recorded_seed_array_recalculation.csv)、[来源SHA](analysis/n17_history/source_index.csv)。这是对已记录指标的数值重建，没有声称重新加载旧预测/大checkpoint。旧正式seed为2032–2039，不能与新2022–2025跨人口配对。

|旧人口/协议|读数|SN / SB宏AUC|ΔAUC（正/负seed）|Δ宏LL（变差/改善seed）|
|---|---|---|---|---|
|自然KR，common Adam|selected|.667714 / .662511|−.005203（1/7）|+.007457（8/0）|
|自然KR，common Adam|E100|.663947 / .656525|−.007422（1/7）|+.008678（7/1）|
|自然KR，common SGD|selected|.648212 / .649049|+.000837（5/3）|+.004028（5/3）|
|自然KR，common SGD|E100|.649216 / .648339|−.000877（5/3）|+.008851（7/1）|
|自然KR，quota Adam|selected|.667628 / .658437|−.009191（2/6）|+.006207（7/1）|
|人工ML，common Adam .001|selected|.733318 / .722926|−.010392（0/8）|+.020309（7/1）|
|人工ML，common Adam .001|E100|.683353 / .685467|+.002114（5/3）|+.045201（8/0）|
|人工ML，own LR SN .001 / SB .0003|selected|.733318 / .722444|−.010875（0/8）|+.020539（6/2）|
|人工ML，own LR SN .001 / SB .0003|E100|.683353 / .704498|+.021145（8/0）|−.313661（0/8）|

人工ML common LR的AULC1..100差−.005166，8/8负；own LR变为+.000389，5正3负，而AULC1..20差−.040981，8/8负。末轮own LR优势不能改称单独的q收益：它包含学习率变化、学习速度及SN后期风险升高。KR common/own获胜配方相同，不能把重复数字当两次独立复现。**MEDIUM**：阶段、损失和学习率口径影响旧结论；跨自然ML推广仍需新完整结果。

### 固定计数未消除KR原固定点trace差

原N2四seed×双路径×E10/E40的16锚，在ALL块上IID SB/SN trace范围2.008914–2.584112；quota仍约1.994669–2.550850。重新计算IID域计数预测项只占0.001297%–0.105936%。完整quota训练selected的q差−.009191，也未表现为消除原差异；两臂quota−IID及配对交互逐seed保存在[控制表](analysis/n17_history/quota_by_q_paired_seed_controls.csv)。这限制这些大batch、固定状态/人口下“主要由域计数波动解释”的主张，不排除域内梯度异质性、高阶依赖或其他batch条件。

ALL的同状态Adam均值差范数/边际MC RMS之和为.479189–.624439；该比值不是CI或显著性门槛。原测量未确认超出MC尺度的q平均更新漂移，更不能证明Adam历史必要性。E40部分共享块采样数据梯度为零但原生更新非零的记录完整保留；有限样本零值不等于全人口梯度为零或路径永久失活。

### N3独立确认没有触发长期拓扑训练

原开发冻结C_embedding/E10，独立2036–2039确认两路径完整替换均4/4 AUC为正，但SN/SB平均仅+.000032525/+.000028166；方向残差约+.000014797/+.000014084，SN方向只有2/4正。均小于原局部实际界.0001，四个相关MCSE达标，原门控`N4_trigger=false`。原[冻结门控](analysis/n17_history/N3_confirmation_summary/confirmation_gate.json)和完整条件表保留，没有把未触发的N4写成失败或缺失必跑训练。

**MEDIUM**：旧局部确认未达到预定资源触发规则。**WEAK**：从这些小局部效应推断长期共享唯一中介或历史状态必要性。本次自然N17四训练seed和两个独立MC bank分别记录；两bank不冒称独立训练确认，也不因旧未触发结论省略新规定N2/N3/precision与完整训练。

## AO04：d0_large、两种子与不完整KR边界

从10/15的原selected逐seed/cell差异表重新汇总，保留测试聚合表、rho日志与身份审计原文，见 [来源和SHA](analysis/ao04_history/source_index.json)、[完整差异汇总](analysis/ao04_history/historical_selected_summary.csv)。不重新运行旧bootstrap，也不声称重载旧模型。

|人口/完整对照数|比较|selected验证Δ宏AUC / LL|selected测试Δ宏AUC / LL|
|---|---|---:|---:|
|人工ML **d0_large**，2 seed|B1−B0|+.000005765 / +.000101|+.000015592 / +.000620|
|人工ML d0_large，2 seed|B1−B2|−.000006842 / +.000202|−.000012656 / +.000173|
|自然KR，只有seed2022完整|B1−B0|+.001494 / −.005557|−.008385 / −.000543|

人工ML B1−B0测试AUC一正一负、LL两seed均变差；B1−B2验证AUC两seed均略负，测试一正一负。旧证据不支持稳定有用的非对称融合改进，也不能由近零差认定rho不可识别。原日志没有expert difference或logit Jacobian；rho未饱和不等于预测对rho敏感。KR 2023缺少完整B1结果，不能补成零或四seed。

历史验证NPZ缺少row-ID，旧审计标为`IDENTITY_INCOMPLETE_NO_ROW_IDS`；文件SHA相符不能独立证明与当前split同序。本次自然AO04采用明确row-ID、四完整seed、原B0/B1/B2配方及冻结模型的局部Jacobian/风险探针；这些新结果尚未完成，旧近零结果不能作为跳过训练的依据。旧人口和预算不与d2_large的V2/V9混写。跨环境推广的现有证据 **WEAK**，特定旧表的数值记录保留。

## 对下一阶段分析的约束

所有自然候选按旧配方执行完整预算，先冻结后读test。科学判断须同时报告基线绝对表现、任务/域、目标与optimizer、实际step/曝光、selected/final、AUC/LL及seed差异。历史数学/数值身份保留为身份；有限探针机制不能替代端点竞争效果，端点效果也不能反向证明唯一机制。
