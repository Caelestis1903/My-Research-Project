# AC-S5：Equal目标下历史/当前梯度控制（匹配预算组已完成）

**状态：matched250完整16/16已完成；full50尚有1个配置在运行。** 本版只分析完整四种子matched250，不能代表尚未完成的全预算结果。选模与配置冻结后，112份测试预测的672个逐域任务指标已由主代理重新计算，误差均小于1e−12。原历史CSV没有重载旧checkpoint/预测，证据层级分开。

## 1. 实际协议与控制身份

自然训练人口800167行，d0/d1/d2=155949/318599/325619；目标物理域固定d2。自然d2占40.694%，是最大域；旧人工d2占80%。两个任务仍为rating>3/rating>4。

原learned-fusion M3OE，两Adam主.01/arch.001、coupled L2=1e−5，50轮。普通拼接loader逐轮shuffle、无放回；本组每轮取187500行，250主更新、9,375,000次主样本曝光，另50次architecture更新。本次architecture实际曝光2,009,934行，含原loader短尾；“匹配预算”指主更新/主曝光及arch步数，不能声称新旧全部样本曝光完全相同。

Equal每域α=1/3，三个逐行权重为1.710318/.837173/.819124，保留原per-row mean目标，不对target部分另行归一化。四臂在共享C和architecture A上控制：

|臂|进入持久Adam状态的梯度|用于参数位移的同前状态假想梯度|
|---|---|---|
|F|FULL|FULL|
|T|TARGET|TARGET|
|HT|TARGET|FULL|
|DH|FULL|TARGET|

保留原T的target-only shared访问、原持久optimizer step及HT/DH参数覆盖公式；private参数始终按FULL更新。这里同时改变m/v的长期历史，且共享/arch控制均存在，不能称为只改变一阶动量。

四个seed各自四臂初始参数SHA相同；每seed全部300个实际批次（250main+50arch）的row-ID哈希和domain counts完全相同。公式最大绝对残差约1.45e−6，loss分解≤2.99e−8，所有已测共享/arch步骤target梯度非零。原状态中的None条件均由producer强制检查。见[配对初始化](analysis/core/AC-S5/paired_initialization_identity.csv)、[批次身份](analysis/controlled_updates/paired_batch_identity.csv)、[公式记录](analysis/controlled_updates/control_formula_and_none_context.csv)。

主训练累计独立支持，seed2022/23/24/25分别为800163/800167/800165/800166行；每轮无重复，跨轮有重复。相对于可用800167行，末轮仍未见主样本的行数分别4/0/2/1。可用人口与实际曝光分列，见[实际预算](analysis/controlled_updates/realized_main_arch_budgets.csv)、[逐域支持](analysis/controlled_updates/realized_main_support_by_domain.csv)。

## 2. 匹配预算结果：F的排名表现优于三种控制

所有数字为四优化seed均值，宏指标等权平均六cell，非pooled AUC。

|臂|selected宏AUC / LL|final宏AUC / LL|
|---|---:|---:|
|F|.767579 / .510646|.767067 / .509515|
|T|.738540 / .515196|.735865 / .520167|
|HT|.757820 / .503020|.757892 / .509035|
|DH|.760251 / .505755|.759977 / .509308|

|比较|selected ΔAUC [点态t95]|正/负seed|selected ΔLL（改善seed）|final ΔAUC|final ΔLL|
|---|---|---|---:|---:|---:|
|T−F|−.029039 [−.032993,−.025085]|0/4|+.004551（1/4）|−.031202|+.010652|
|HT−F|−.009759 [−.012592,−.006926]|0/4|−.007626（4/4）|−.009175|−.000480|
|DH−F|−.007328 [−.012798,−.001859]|0/4|−.004891（3/4）|−.007090|−.000207|
|DH−HT|+.002430 [−.001049,+.005909]|4/0|+.002735（2/4）|+.002085|+.000273|

T/HT/DH相对F的final宏AUC也均4/4下降；HT/DH final LL各2改善2变差。HT selected在四seed均以较低LL伴随较低AUC，不能说“完全无效”，也不能按排名损害推出训练目标损失一定变差。四seed区间仅描述固定人口的优化差异，未校正多重比较，历史test已复用。

## 3. target、non-target与逐任务反例

F selected的域平均AUC为d0=0.768629、d1=0.782360、d2=.751748；最大d2的基线最低，大小不是困难度的定义。

|比较|selected target d2 ΔAUC|selected non-target ΔAUC|task0宏ΔAUC|task1宏ΔAUC|
|---|---:|---:|---:|---:|
|T−F|−.005644|−.040737|−.029928|−.028151|
|HT−F|−.008006|−.010635|−.008987|−.010530|
|DH−F|−.011061|−.005462|−.007959|−.006698|
|DH−HT|−.003055|+.005173|+.001029|+.003832|

前三种控制相对F的target和non-target域平均AUC各4/4下降，两个任务宏AUC也各4/4下降。HT−F的六个cell在四seed中全部AUC下降：

|cell|F selected AUC|HT−F ΔAUC|HT−F ΔLL|
|---|---:|---:|---:|
|d0t0|.761630|−.009267|−.022225|
|d0t1|.775629|−.014170|−.018961|
|d1t0|.773373|−.010174|−.004253|
|d1t1|.791347|−.008930|−.009660|
|d2t0|.744056|−.007521|+.007282|
|d2t1|.759441|−.008491|+.002060|

HT相对F的target LL在四seed变差（域平均+.004671），non-target LL在四seed改善（−.013775）。风险与排名的分布并不相同。[完整逐cell](analysis/core/AC-S5/paired_cell_deltas.csv)、[逐域](analysis/core/AC-S5/paired_domain_deltas.csv)、[逐任务](analysis/core/AC-S5/paired_task_deltas.csv)保留全部臂、阶段及seed。

“小域改善/大域下降”也必须按比较定义：T/HT/DH相对F的selected/final严格“两较小域均改善、最大域下降”均0/4；但DH−HT在selected为3/4、final为2/4。这个候选之间的条件取舍确实出现，不能据前三个0/4推不存在任何取舍；DH与HT的宏AUC又都低于F，不构成整体新方法收益。

## 4. 与旧自然KR、人工ML逐项对应

旧selected宏AUC差：

|比较|旧自然KR|旧人工ML d2_large|新自然ML matched250|
|---|---:|---:|---:|
|T−F|+.002369（4/4正）|+.010839（4/4正）|−.029039（4/4负）|
|HT−F|+.017127（4/4正）|+.020527（4/4正）|−.009759（4/4负）|
|DH−F|+.004218（4/4正）|−.004047（2正2负）|−.007328（4/4负）|

旧HT−F selected的target AUC差在KR/人工ML分别+.040806/+.016132，均4/4正；新自然matched250为−.008006，4/4负。原“保留当前FULL、改变历史能救援target”的经验现象，在这一自然预算条件下没有复现；旧受控数据结果保留，其跨自然环境的一般化必须缩限。

新旧ML主更新250及主曝光预算相同，不能只用“自然总是多训练了”说明这一对照的方向差异。不过可用支持、域比例、域内ID/标签分布、重要性权重、arch短尾曝光及历史初始化身份仍非单因素控制；旧source/init/state缺口也不能由新初始化SHA填补。原F人工selected基线.718728，新F=.767579，反例需连同基线条件解释，而不是归为小域/困难域的普遍规律。详见[历史/新统一表](analysis/controlled_history/paired_summary.csv)。

![已完成cohort的阶段比较](analysis/core/AC-S5/controlled_population_stage.png)

[PDF](analysis/core/AC-S5/controlled_population_stage.pdf)。误差棒为四优化seed的点态t95；不同环境target物理含义不同，没有合并成独立重复。

## 5. 实际m/v、位移与竞争解释

新full-coordinate遥测在HT的C块E41–50显示，四seed的平均实际位移范数相对F为2.512–2.778倍；持久m范数比.392–.521，v范数比.225–.419。输入持久状态的target梯度较小，较小v与FULL假想当前位移放大兼容；但是参数路径、m/v、正则和architecture一起变化，以上比值没有唯一识别v或m的因果中介。

新遥测是拼接全部坐标的L2范数、step后状态；原probe的持久范数是step前按tensor范数求和，不能将两个同名列直接比较。[逐seed窗口比值](analysis/controlled_updates/paired_seed_window_state_norm_context.csv)及[完整窗口统计](analysis/controlled_updates/seed_window_gradient_state_summary.csv)分列了记录来源。位移变大与AUC变好没有一一对应关系。

- **MEDIUM**：本次固定自然人口、matched250及四seed下，三种控制相对F的排名下降稳定；HT selected的AUC/LL冲突及六cell AUC负差重复存在。
- **STRONG（实施身份）**：四臂初始/批次一致、持久状态和覆盖公式按原协议执行；这不是算法收益强证据。
- **WEAK**：由新旧方向变化唯一归因人口规模、比例、Adam一阶历史或某个共享子模块。自然目标梯度的信号与权重、支持/频率以及相对优化时钟均是竞争解释。

最有辨别力的本轮既定工作是完成full50，检验同人口延长主时钟后方向是否保持；并完成Sample负边界S6，判断Equal目标是否构成必要条件。它们仍须完整预算执行。后续若要区分m、v与瞬时步幅，需要预先设计同状态因子控制及冻结评价，不能由当前长期联合控制事后宣称已识别，也不在本次结果后自动增加训练。
