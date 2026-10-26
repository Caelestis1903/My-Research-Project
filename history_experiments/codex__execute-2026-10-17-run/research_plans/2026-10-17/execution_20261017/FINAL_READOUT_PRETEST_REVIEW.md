# 全局解封前的统计读数审查

状态：PASS_SELF_REVIEW，independent_review=false，主agent按start.md §11逐项对照既定规划与实际入口。此记录未启动test、未签署GLOBAL freeze、未关闭待运行分支。

- n17_test_readout仅在主agent全阶段闭合的冻结文件存在且定义、源码、config和逐epoch纯model SHA相符时运行；pilot不进入独立确认test。保存实际预测NPZ及其逐epoch SHA索引、模型SHA、ordered row-ID SHA、metrics/receipt SHA。历史test已用于过去研究，不宣称全新隐藏test。
- n17_performance以每条轨迹selection宏AUC在1..100最早最大值选轮，读取该轮test，两个q可以选择不同轮；固定E10/20/50/100和AULC1..20/100保留，AULC为离散epoch均值。不同预算使用不同组名，本轮H200未触发。当前N3阶段E10，无条件新增E40 endpoint；若未来实际触发的拓扑阶段确为E40，须在全局冻结前显式增加该端点。
- 宏、train-p/test-count加权cell指标、六cell、域均值与task均值均保留。加权cell AUC不是pooled AUC；两task的同域p权重在六cell内归一化。J只来自完整训练人口记录。
- 每个comparison的配对训练seed为不确定性单位；两q、两task、多个epoch、条件draw均不是新训练重复。n17_n1_control.paired给t95、全部2^n符号翻转及20,000次配对seed bootstrap（固定统计seed20261017）；符号随机化需要零假设下符号交换性，t/bootstrap均受小n限制。
- 主Holm族严格为N1_common_adam及N1_common_sgd的selected test macro AUC两个比较。own赢家与common重合时复用同轨迹，只是不同估计对象标签，不增加样本量。其余端点、cell、环境、quota干预为次级pointwise读数，不能从其间挑最好一个当主发现。
- n17_quota_readout使用同8seed、H100，分别给四臂绝对性能、IID q差、quota q差、I_quota及SN/SB各自quota−IID提升；改进对比方向必须在comparison定义中冻结，不能看到test后交换candidate/baseline。
- n17_method_guards对SB标签候选相对SN标签baseline给宏均值≥.003、t95下界>0；六cell单侧下界≥−.003和train-p LL逐seed相对变化单侧上界≤.01，七项各α=.05/7。相对LL先逐seed作比值再聚合。完整数值通过仍不是新方法授权，还需冻结候选、成本与文献新颖性评估；本轮采样臂本身不自动命名为新算法。
- 等效只在相关配对区间整个落入±.003（局部±.0001）时表述；跨零不等于等效。八seed主CI若同时跨越±.003，将登记新的具体精度问题，不追加seed追显著。
- n17_domain_context按同人口记录baseline AUC、LL、N、test prevalence、域/task与训练阶段。域大小只有三个物理域，不做域级因果回归；严格both-smaller与either-smaller模式分开，保留反向模式、seed一致性和实际界。大小、自然难度、模型/目标/优化器、训练进度不能互作定义。
- 训练耗时保留原segment elapsed，同时累加检测到重启的active segment和epoch_seconds；不是独占GPU吞吐，也不把暂停的闲置时间伪作训练时间。无H200时不引入不必要续训。

审查只涉及定义/公式/现有入口的必要对应，不添加训练、合成科学结果或额外ablation。最终比较清单、模型索引和触发终态仍须在所有必做训练完成后由主agent实际签署。
