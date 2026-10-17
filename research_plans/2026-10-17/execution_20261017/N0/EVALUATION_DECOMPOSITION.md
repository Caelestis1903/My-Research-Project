# N0 固定预测的评价与校准分解

N0已完成；零新增正式训练。七家族、原2022–2025四训练seed，所有原固定轮/selection/common选轮/共同早期读数以六cell重算，三套权重共4068行。E100仅真实完成且配置同一的seed有资格，额外两组来自已认证的同轨迹continuation；前缀不算独立seed。见readout_ledger.csv、seed_means.csv及E100_qualification.json。

**STRONG：评价权重本身足以翻转同一预测的符号。** calAdam E50的SB−SN宏AUC=−.0090742877，训练p加权cell AUC=+.00101563，test-p加权=+.003108019。前者1正3负，中者2正2负，后者4正。三域两task平均差依次−.014657326、+.003441643、−.016007180；乘宏域权重后的贡献−.004885775、+.001147214、−.005335727。平均过零alpha≈.7449（均匀域权重到test-p的线性插值）。这是精确算术分解，不能解释域内人口变化的因果机制，也不将加权cell AUC称pooled AUC。

**MEDIUM：训练阶段和选轮会改变方向。** 统一AULC1..20的宏AUC差：fixedAdam−.006373、fixedSGD−.000793、calAdam−.009924、calSGD+.001540、2B SGD−.000111、MovieLens+.003178、AdaTask−.006327。calSGD E10+.005616、E20+.002901、E50+.001075，而宏AUC自选−.000972；MovieLens E20+.005220、E50−.000986、自选+.000539。原early窗另列，不与固定20窗混同。部分E100只有1–2个seed，不能写为四seed重复。

**MEDIUM：部分概率质量差可被校准。** 七家族四seed两臂的宏AUC选轮与E50，合计672个cell校准拟合，selection半集拟合，diagnostic/test仅评价，2016行完整输出。672拟合均成功；变换前后score AUC和概率AUC最大变化均0；从原逐样本预测复算test AUC相对历史表最大残差2.22e−16。宏AUC选轮test的336个cell中279个LL改善，但cell不是独立训练重复。所有七家族两臂平均LL/Brier均改善，亦保留个别变差cell。此结果支持存在可校准成分，不证明学习路径差异全部来自校准，更不证明两q相等。

例：calAdam selected的校准后−前宏LL，SN约−.019843、SB约−.027994；AdaTask SN−.016057、SB−.026971。严格单调转换保持cell排序，故其LL改善不能消除已有AUC差。参数、边界、失败记录、before/after AUC/LL/Brier、浮点ties见calibration_cell_metrics.csv和calibration_diagnostics.json。

**规模与难度：** KR训练域份额约(.08613,.86928,.04459)，test约(.01042,.98212,.00746)；本例两个小域下降而大域改善，方向与“小域收益/大域损伤”相反。不能由此推出规模是主要原因：域身份、标签率、特征与人口同时变化，模型和阶段也影响方向。diagnostic d0t1只有50正例，宏权重仍1/6；split_populations.csv给逐域任务正负和joint-label计数。

竞争解释保留：评价权重/域内人口、LR与优化预算、排序与校准、真实采样几何及共享表示路径。本轮最有辨别力的下一步是已冻结N1对称q×LR和独立种子；不是改变主指标或重训历史矩阵。新训练test尚未用于任何选择。

数据链：readout_manifest.json给模型/时钟/选择人口/评价人口/权重与预测SHA；12个缺少直接文件的预测读数通过原认证prefix引用恢复且SHA匹配（216行校准），不作为新模型。readout ledger中的同义端点与重复预测读数不可当独立重复。
