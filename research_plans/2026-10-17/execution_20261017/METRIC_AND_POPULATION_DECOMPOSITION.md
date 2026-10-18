# 评价、人口与校准分解

状态：COMPLETE。N0复算零新增正式训练；本轮120条训练及72 formal统一test也已完成。历史与新轮的模型、人口、时钟和独立seed分别保留。

七家族、原2022–2025四训练seed，所有原固定轮/selection/common选轮/共同早期读数以六cell重算，三套权重共4068行。E100仅真实完成且配置同一的seed有资格，额外两组来自已认证的同轨迹continuation；前缀不算独立seed。见readout_ledger.csv、seed_means.csv及E100_qualification.json。

**STRONG：评价权重本身足以翻转同一预测的符号。** calAdam E50的SB−SN宏AUC=−.0090742877，训练p加权cell AUC=+.00101563，test-p加权=+.003108019。前者1正3负，中者2正2负，后者4正。三域两task平均差依次−.014657326、+.003441643、−.016007180；乘宏域权重后的贡献−.004885775、+.001147214、−.005335727。平均过零alpha≈.7449（均匀域权重到test-p的线性插值）。这是精确算术分解，不能解释域内人口变化的因果机制，也不将加权cell AUC称pooled AUC。

**MEDIUM：训练阶段和选轮会改变方向。** 统一AULC1..20的宏AUC差：fixedAdam−.006373、fixedSGD−.000793、calAdam−.009924、calSGD+.001540、2B SGD−.000111、MovieLens+.003178、AdaTask−.006327。calSGD E10+.005616、E20+.002901、E50+.001075，而宏AUC自选−.000972；MovieLens E20+.005220、E50−.000986、自选+.000539。原early窗另列，不与固定20窗混同。部分E100只有1–2个seed，不能写为四seed重复。

**MEDIUM：部分概率质量差可被校准。** 七家族四seed两臂的宏AUC选轮与E50，合计672个cell校准拟合，selection半集拟合，diagnostic/test仅评价，2016行完整输出。672拟合均成功；变换前后score AUC和概率AUC最大变化均0；从原逐样本预测复算test AUC相对历史表最大残差2.22e−16。宏AUC选轮test的336个cell中279个LL改善，但cell不是独立训练重复。所有七家族两臂平均LL/Brier均改善，亦保留个别变差cell。此结果支持存在可校准成分，不证明学习路径差异全部来自校准，更不证明两q相等。

例：calAdam selected的校准后−前宏LL，SN约−.019843、SB约−.027994；AdaTask SN−.016057、SB−.026971。严格单调转换保持cell排序，故其LL改善不能消除已有AUC差。参数、边界、失败记录、before/after AUC/LL/Brier、浮点ties见calibration_cell_metrics.csv和calibration_diagnostics.json。

**规模与难度：** KR训练域份额约(.08613,.86928,.04459)，test约(.01042,.98212,.00746)；本例两个小域下降而大域改善，方向与“小域收益/大域损伤”相反。不能由此推出规模是主要原因：域身份、标签率、特征与人口同时变化，模型和阶段也影响方向。diagnostic d0t1只有50正例，宏权重仍1/6；split_populations.csv给逐域任务正负和joint-label计数。

竞争解释保留：评价权重/域内人口、LR与优化预算、排序与校准、真实采样几何及共享表示路径。N1对称q×LR与独立确认现已完成，下文给全局冻结后的新test读数；test未用于本轮调参或分支选择。

数据链：readout_manifest.json给模型/时钟/选择人口/评价人口/权重与预测SHA；12个缺少直接文件的预测读数通过原认证prefix引用恢复且SHA匹配（216行校准），不作为新模型。readout ledger中的同义端点与重复预测读数不可当独立重复。

补充：MovieLens原生user_id可经已知原始row IDs恢复，test共1209用户。已补逐split域用户簇数和200次按用户配对条件区间，见ML_user_cluster_counts.csv、ML_paired_user_cluster_intervals.csv与user_cluster_receipt.json。KR当前处理文件仍无user_id；未将其row区间称为user独立。该补充不改变N1配置、阈值或训练选择。

## 域规模、任务与baseline的具体对照

historical_cell_population_context.csv给672个历史seed/readout/cell的SN、SB绝对AUC/LL、四个人口计数及标签率；summary JSON给168组四seed区间。SN只是本对照的参考模型，绝不定义为域的内在难度。旧MovieLens的不同seed还改变子集人口，计数范围单列，不能只当优化方差。

例如KR calAdam E50（四seed均值；该家族名calAdam指历史学习率标定，不指上文概率后校准）：

|域/任务|训练域N|SN AUC|SB−SN AUC|SN LL|SB LL|
|---|---:|---:|---:|---:|---:|
|d0/t0|172782|.602282|+.001434|.350190|.394117|
|d0/t1|172782|.593794|−.030749|.237373|.282714|
|d1/t0|1743898|.688696|+.002134|.530361|.514173|
|d1/t1|1743898|.712942|+.004749|.365184|.351755|
|d2/t0|89452|.613785|−.015204|.792729|1.055428|
|d2/t1|89452|.627223|−.016810|.786633|1.042572|

同一d0、相同训练规模的两task方向不同；d0/t0的排序略改善与LL恶化共存。小域之间baseline与标签率也不同（test d0/t0正率.0873、d2/t0正率.5882）。这些点估计说明不能从本表将所有cell差异直接归于规模；d0/t0的小幅正均值不等于已独立确认的收益，四seed区间保留在summary。排序与LL并非同一估计对象；本表不识别规模或自然难度的因果贡献，也不排除规模参与作用。该补充为既有冻结结果重算，无新增训练或新test读取。


## 两环境的任务与进度定义

数据定义沿用stage_S1_data.py和冻结缓存：KR物理域对应tab0/tab1/tab4，task0为stored_is_click，task1为stored_long_view；MovieLens物理域按原年龄组{1,18}/{25}/{35,45,50,56}划分，task0为1[rating>3]、task1为1[rating>4]。后者标签嵌套；两环境的task1不代表相同事件。模型均沿用对应特征字典的fixed_model及冻结fusion/detached routing，不能把跨环境差异解释为只改人口大小或只改某一个标签。

N5-C明确固定subset_seed=2022、训练row列表与validation分半；optimizer seed独立改变初始化/采样。ML训练187500行，每epoch5更新、尾23660，H100为500更新/18750000样本暴露；KR训练2006132行，每epoch49更新、尾40052，H100为4900更新/200613200暴露。相同epoch是各自人口的相同预期暴露倍数，不是相同step或相同计算量。最终报告记录实际训练产物时钟；固定人口的八优化seed只识别该人口条件下稳定性，不自动给子集抽样方差或跨环境因果结论。

## 新轮：固定模型的评价权重与独立优化seed

以下都是selection选定的同一预测；train-p/test-count加权cell AUC与macro分开，不叫pooled AUC。

| group | macro | train_p_cell | test_p_cell |
| --- | --- | --- | --- |
| N1_common_adam | -0.005203 | -0.005055 | -0.005277 |
| N1_common_sgd | +0.000837 | -0.000951 | -0.001638 |
| N2_quota_adam | -0.009191 | -0.004793 | -0.003571 |
| N5_common_adam | -0.010392 | -0.000482 | -0.012557 |
| N5_own_adam | -0.010875 | +0.000744 | -0.013586 |

KR Adam三种均值均负；SGD macro略正、两种加权略负，但区间均有不确定性，不能据符号点估计宣称稳定人口反转。ML own macro−.010875而train-p cell+.000744；后者t95[−.000649,+.002138]落在±.003内，macro与test-p均显著负向的次级pointwise读数仍保留。它们是不同评价函数，不能以更有利权重替换预定macro主指标。

## 新模型的baseline、域/task与损失

下表为八seed均值，baseline为SN；正LL_delta表示SB概率loss恶化。

| group | domain | task | train_domain_n | test_cell_n | test_positive_rate | SN_reference | SB_candidate | difference | LL_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1_common_adam | 0 | 0 | 172782 | 2613 | 0.087256 | 0.625817 | 0.631443 | 0.005627 | 0.004880 |
| N1_common_adam | 0 | 1 | 172782 | 2613 | 0.045924 | 0.640398 | 0.635195 | -0.005203 | 0.006955 |
| N1_common_adam | 1 | 0 | 1743898 | 246284 | 0.210517 | 0.695560 | 0.689887 | -0.005673 | 0.002414 |
| N1_common_adam | 1 | 1 | 1743898 | 246284 | 0.111315 | 0.722983 | 0.718064 | -0.004919 | 0.002028 |
| N1_common_adam | 2 | 0 | 89452 | 1870 | 0.588235 | 0.650526 | 0.640471 | -0.010055 | 0.014488 |
| N1_common_adam | 2 | 1 | 89452 | 1870 | 0.442246 | 0.671001 | 0.660005 | -0.010996 | 0.013980 |
| N5_common_adam | 0 | 0 | 34500 | 27687 | 0.515802 | 0.735840 | 0.733463 | -0.002377 | 0.003008 |
| N5_common_adam | 0 | 1 | 34500 | 27687 | 0.179290 | 0.749749 | 0.745850 | -0.003899 | 0.001414 |
| N5_common_adam | 1 | 0 | 3000 | 41434 | 0.532703 | 0.717206 | 0.685405 | -0.031801 | 0.079825 |
| N5_common_adam | 1 | 1 | 3000 | 41434 | 0.169692 | 0.744014 | 0.718349 | -0.025664 | 0.039300 |
| N5_common_adam | 2 | 0 | 150000 | 30900 | 0.585728 | 0.720585 | 0.721094 | 0.000509 | -0.000762 |
| N5_common_adam | 2 | 1 | 150000 | 30900 | 0.204660 | 0.732516 | 0.733396 | 0.000880 | -0.000931 |

完整各组、两q的AUC/LL绝对值与配对区间在FINAL_SELECTED_CELL_METRICS及FINAL_domain_context。KR相同d0训练N下t0/t1方向不同且不精确；d1两task8/8负、d2两task7/8负。ML最小域d1的两task大幅下降，最大域d2略改善；baseline AUC并不随N统一单调。标签率、不同域输入和目标p同时变化，不能把小域定义为困难域或从三个域识别N主因。

N1 selected task均值：Adam t0−.003367/t1−.007039，两区间跨0；SGD+.001144/+.000531，两区间跨0。ML common t0−.011223/t1−.009561，own−.010192/−.011557，四个pointwise区间都在负侧。cell/task/域不是独立训练重复。

严格小域双赢/大域下降的seed计数见FINAL_ANALYSIS及FINAL_domain_context；KR Adam0/8、SGD1/8，按实际界严格模式均0/8，不支持普遍该取舍。N0历史小域下降/大域改善和新ML反例共同保留。新test未另做概率校准，N0校准结论仅针对原冻结历史预测。

新轮CI以8训练seed描述固定test人口条件下优化随机性；原ML用户簇bootstrap为固定拟合模型的测试人口条件不确定性，不能合并成训练seed。GLOBAL_MODEL_AND_PROTOCOL_FREEZE、test_metadata逐预测SHA/行序、FINAL_performance原seed和alpha敏感性图提供数据链。
