# 评价、人口与校准分解

N0部分已完成；零新增正式训练。本轮N1/N2/N5新模型的最终加权读数仍待全局冻结后test，不将本文件视为整轮完成。

七家族、原2022–2025四训练seed，所有原固定轮/selection/common选轮/共同早期读数以六cell重算，三套权重共4068行。E100仅真实完成且配置同一的seed有资格，额外两组来自已认证的同轨迹continuation；前缀不算独立seed。见readout_ledger.csv、seed_means.csv及E100_qualification.json。

**STRONG：评价权重本身足以翻转同一预测的符号。** calAdam E50的SB−SN宏AUC=−.0090742877，训练p加权cell AUC=+.00101563，test-p加权=+.003108019。前者1正3负，中者2正2负，后者4正。三域两task平均差依次−.014657326、+.003441643、−.016007180；乘宏域权重后的贡献−.004885775、+.001147214、−.005335727。平均过零alpha≈.7449（均匀域权重到test-p的线性插值）。这是精确算术分解，不能解释域内人口变化的因果机制，也不将加权cell AUC称pooled AUC。

**MEDIUM：训练阶段和选轮会改变方向。** 统一AULC1..20的宏AUC差：fixedAdam−.006373、fixedSGD−.000793、calAdam−.009924、calSGD+.001540、2B SGD−.000111、MovieLens+.003178、AdaTask−.006327。calSGD E10+.005616、E20+.002901、E50+.001075，而宏AUC自选−.000972；MovieLens E20+.005220、E50−.000986、自选+.000539。原early窗另列，不与固定20窗混同。部分E100只有1–2个seed，不能写为四seed重复。

**MEDIUM：部分概率质量差可被校准。** 七家族四seed两臂的宏AUC选轮与E50，合计672个cell校准拟合，selection半集拟合，diagnostic/test仅评价，2016行完整输出。672拟合均成功；变换前后score AUC和概率AUC最大变化均0；从原逐样本预测复算test AUC相对历史表最大残差2.22e−16。宏AUC选轮test的336个cell中279个LL改善，但cell不是独立训练重复。所有七家族两臂平均LL/Brier均改善，亦保留个别变差cell。此结果支持存在可校准成分，不证明学习路径差异全部来自校准，更不证明两q相等。

例：calAdam selected的校准后−前宏LL，SN约−.019843、SB约−.027994；AdaTask SN−.016057、SB−.026971。严格单调转换保持cell排序，故其LL改善不能消除已有AUC差。参数、边界、失败记录、before/after AUC/LL/Brier、浮点ties见calibration_cell_metrics.csv和calibration_diagnostics.json。

**规模与难度：** KR训练域份额约(.08613,.86928,.04459)，test约(.01042,.98212,.00746)；本例两个小域下降而大域改善，方向与“小域收益/大域损伤”相反。不能由此推出规模是主要原因：域身份、标签率、特征与人口同时变化，模型和阶段也影响方向。diagnostic d0t1只有50正例，宏权重仍1/6；split_populations.csv给逐域任务正负和joint-label计数。

竞争解释保留：评价权重/域内人口、LR与优化预算、排序与校准、真实采样几何及共享表示路径。本轮最有辨别力的下一步是已冻结N1对称q×LR和独立种子；不是改变主指标或重训历史矩阵。新训练test尚未用于任何选择。

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

## 新轮终局对照（尚待）

最终需追加N1两优化器及触发quota/固定ML人口的macro、训练p与评价p加权cell读数、各域各任务和八训练seed区间。主指标、alpha范围与test人口均按原冻结协议，不依据本轮结果换权重或倒推新设计。N0固定预测、校准与用户簇bootstrap属于不同不确定性对象，不合并成新训练seed。
