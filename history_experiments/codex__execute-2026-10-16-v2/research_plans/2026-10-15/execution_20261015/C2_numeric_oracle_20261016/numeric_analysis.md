# C2.1 固定参数数值结果与解释边界

状态：**8/8 数值case完成，0 training，C2训练未触发**。两数据集×seeds 2022–2025；模型M3OE，阶段仅为初始化。主agent完成DESIGN→IMPLEMENT→SELF_REVIEW→CPU RUN→ANALYZE，未派发新子agent。数值正确性与科学推广范围分别评价。

## 执行与数值验收

采用KuaiRand/natural、MovieLens/d2_large原训练split、完整特征schema与原模型。每域固定16个真实训练row构成有限总体，不按label选择，域内均匀有放回抽样。KuaiRand四seed共享pool，MovieLens原训练子集与模型seed同变，实际pool IDs逐case保存。没有读入旧checkpoint、更新原模型参数或执行正式训练。

固定α分别为Equal=1/3、Sample=全训练p，q取Natural=p、Balanced=1/3和Stress=(.0001,.9998,.0001)，使用α/q与固定batch分母。每配置B=16/128、10000次Monte Carlo，共96行配置统计、288行参数分区统计、16行完整目标核对。每row两个任务均通过真实autograd取梯度。

| 验收量 | 实测最大残差 | 支持范围 |
|---|---:|---|
| 全坐标精确枚举梯度相对残差 | 1.865e-16 | 同有限总体、固定θ、正确α/q |
| 整批autograd与逐row目标聚合相对残差 | 2.685e-7 | float32原模型的聚合一致性 |
| MC loss标准化残差 | 2.894 SE | 10000批次；设计上限8 SE |
| MC所选梯度坐标最大标准化残差 | 3.026 SE | 每C/P最多32个非零支持坐标及全部4个A坐标 |
| detached副本上真实Adam vs一步公式绝对残差 | 1.113e-16 | 相同θ、fresh step0、原lr/WD/eps |
| 分批与整批预测最大残差 | 1.193e-7 | eval、LayerNorm、无随机dropout、sigmoid架构 |

原模型θ、param.grad及tau前后不变，8case原始数据/source hashes保持一致。RUN receipt内44个原始产物已逐字节校验。`launch_metadata.json`在进程exit0后补充，明确不属于原producer receipt的快照。

## 稳定现象与反例

**STRONG（有限总体条件下的数值事实）**：相同α、不同q的期望loss与全坐标梯度一致；观测的协方差却依赖q和α。这个条件性等价来自有限总体代数与真实模型核对，不构成新的方法创新。

下表是完整C分区梯度协方差trace的Balanced/Natural比值，四seed均值及范围。B=16与128的比值一致，因为此IID oracle的协方差按1/B缩放；两种B不是额外独立研究重复。

| 数据集 | 目标α | C trace比值均值 | 四seed范围 |
|---|---|---:|---|
| KuaiRand | Equal | 0.214720 | 0.204246–0.231254 |
| KuaiRand | Sample | 2.497026 | 2.394147–2.721071 |
| MovieLens | Equal | 0.133501 | 0.107332–0.151251 |
| MovieLens | Sample | 2.334679 | 2.052109–2.694227 |

P与A分区在当前8个初始化case内方向相同：Equal均小于1，Sample均大于1。详见 `analysis/covariance_ratios_per_seed.csv` 和 `covariance_ratio_seed_summary.csv`。这给出 **STRONG范围内反例**：Balanced不是对所有目标都降低梯度方差。跨训练阶段、全训练总体和模型结构的推广证据为 **WEAK**；当前全部θ仍是初始化。

Stress说明支持很小的q会通过α²/q放大方差。例如MovieLens/Sample的C trace相对Natural平均约8688倍；这是人为stress分布的数值边界，不是实际Natural训练被观察到的故障。不得把它写成M3OE必然失败的证据。

**STRONG（一步数值），WEAK（历史机制解释）**：所选坐标上E[AdamDelta(g)]与AdamDelta(E[g])存在差异，真实torch.Adam核对通过。这里使用fresh step0，bias correction使第一步的β1/β2抵消；因此这些数据不检验persistent momentum、训练过的m/v或历史S5因果机制。C/P与A的lr和坐标数不同，原始delta范数不作跨分区机制强弱排名。

## 缺域归一化与正式batch的反证边界

本oracle明确把batch-cell normalized Equal定义为(1/3)Σ存在域的域内平均BCE，空域贡献0；旧未归一化Σdomain×task口径为其6倍。其期望包含域出现概率，不能在任意小batch下直接替代fixed-p weighting。

在Natural/B=16，四seed平均的相对Equal总体loss偏差为KR −0.161507、ML −0.186143；B=128时为KR −0.000680、ML −0.029046。较低数值在这里反映目标偏差，不能视作性能提高。不同α的loss绝对值不作同目标优劣比较。

**STRONG（条件性概率上界）**：原S5/S6正式main batch=40960，当前Natural训练域比例下，稀少域缺失并非这个小batch例子中的量级。对uniform shuffled无放回batch的无条件边际，(1−p_d)^B是缺域概率上界。考虑较小的末批，KR最小域d2/B=40052的log10上界约为−793.426；ML最小域d1/B=23660约为−165.736（全batch更低）。取宽松整数上界分别小于10^-793、10^-165。因此不能用本小batch的“空域归一化偏差”解释这些正式设置中的主要历史现象。

上界针对域是否出现，不排除类别不平衡、测试集有限正例、时间依赖或参数随训练历史改变。原formal Natural是无放回打乱；当前C2 oracle是有放回IID，两者的方差和曝光条件仍应分别记录。

## 域规模、难度、任务与竞争解释

当前每域pool都为16个不同row，改变的是q和importance coefficient；全训练域规模通过预先定义的p进入Natural和Sample。row不同不等于原始事件统计独立；本oracle的IID是给定pool后的抽样机制。它证明给定α/θ/样本池时抽样分布能改变噪声，不能识别独立样本数量本身是否是历史性能差异的主要原因。规模为主要解释变量的证据仍为 **WEAK**。

没有训练后AUC、baseline selected AUC、per-domain/task selected loss或跨阶段数据。本轮不评价域难度，也不判断“小域改善/大域下降”是否存在；相关历史权衡仍以主结果报告中的S5/S6证据和原协议边界为准。逐任务/域的初始化loss与梯度描述保留于 `cell_gradient_observations.csv`，不能从随机初值loss直接排名任务难度。MovieLens两标签嵌套，跨数据集差异也不能单独归因为任务相关性。

竞争解释包括：importance coefficients与域梯度几何共同决定协方差；Adam坐标非线性使均值更新不能由梯度trace单独预测；训练中θ与m/v演化会改变这些关系；不同q还改变unique exposure和重复使用。当前有限pool的reuse统计不代表数百万row训练集的实际重复率，也没有选中性能证据来区分优化收益与泛化正则效应。

## 下一项最有区分力的工作与停止条件

继续B1既定的实际训练和dense取证。若以后研究C2，先在有效的新checkpoint上固定θ与真实optimizer state、使用正式batch规模核查同α/q的梯度和实际更新差异，并覆盖初始化/common/selected/late阶段；在进入16-run矩阵前确认full-data/exposure/partition协议。若大batch下差异很小或随阶段/seed不稳，保留反证并停止该线。

数值等价通过不自动证明存在值得训练的新问题。需要真正的selected AUC/LL、非目标代价与seed一致性，才能讨论C2训练收益；也不把当前结果与S5历史bundle效应或S6 val/test反转合并成机制已证明。当前`c2_training_go=false`，B1为0/8正式完成，最终完整报告与推送尚未完成。

复现入口：`experiments/c2_numeric_oracle_20261016.py`（CPU、拒绝覆盖已有case），描述性重分析入口：`experiments/c2_numeric_analysis_20261016.py`。关键原始数值、manifest和covariance在本目录；分析图表见 `analysis/fixed_theta_covariance_ratios.pdf`。
