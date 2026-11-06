# CODER IMPLEMENTATION REPORT — Analysis / evaluation

实现责任：按已批准的 `DESIGN_SPECIFICATION.md` 实现分析、有限控制、冻结评价和表/图报告生成。未修改原训练源码、核心采集模块或冻结科学矩阵。

## 文件与共享接口

- `experiments/ema_mechanism_20261106/analysis.py`：`audit_archives`、`analyze_stage`、`fit_predictive_folds`、`run_controls`、`generate_report`。
- `experiments/ema_mechanism_20261106/evaluation.py`：`freeze_endpoints`、`validate_freeze`、`evaluate_frozen` 与独立 Test 子进程入口。
- `tests/test_analysis.py`、`tests/test_evaluation.py`：11项最小统计/隔离/冻结检查。

已与 core/protocol coder 对齐设计第7节的 epoch JSON、prediction receipts、P_A/P_B/V_mech NPZ、`PROBE_MANIFEST`、`ResolvedStudy` 与 RAM `request(summaries/get)`。V 的 `pair_counts` 截断 `-1` padding；空pair诊断写NA。路径从PREPARED冻结身份读取。

## 规格覆盖

S0读取并校验固定Git blob的clean/G-D0原端点，重算四格、逐seed配对和交互；提取A/B/C归档完整V曲线/HPO表，列来源及RAM释放后的缺失观测。归档Test只作原始证据，不读取新Test数组。

S1生成完整Validation曲线、innovation、独立selected/共同epoch/有限年龄匹配online/阶段对比，保留选点±1epoch邻域。固定H偏好同时提供样本内和LOSO＋V行交叉评价，F/H2物理alias去重；按原p与固定pair计算online/F相邻epoch排序变化。

S2保留所有72个cell×group组合，主预测仅纳入冻结24支持关系；输出fast−F/slow−F/fast−slow、全组和非加性、纠正/损害/平局、精确AUC与抽样pair AUC，P_A/P_B稳定性及当期跨人口对应。M2梯度由core即时归约；analysis不新增梯度。

S3实现全部8类固定ridge、4个raw动作与UNIFORM_ACTION。特征只来自origin的P_A/几何；label是origin+5新pulse的V_A或V_B收益。每dataset五LOSO×两V方向，训练576行/评分240行，恰好160次ridge；连续标准化仅拟合折，one-hot字典冻结，目标SSE＋alpha1、截距不罚。F/fast/slow并列固定顺序，raw rank分数标单位，不冒称数值AUC预测。输出逐fold/逐seed/condition/cell/group/cell×group全部范围、MAE/skill/方向/动作效用/regret，逐seed先平均相关观察行再汇总五seed。

S4按冻结五seed均值触发C_CAL/C_BG，未触发写NOT_TRIGGERED。C_CAL只用已保存源p、固定温度边界/优化次数/行交叉评价，AUC保留原p，记录饱和与tie；C_BG仅请求RAM中的E20/E40向量、最多两组/160新状态，顺序scratch读出，仅标量与hash落盘。控制无新增训练且幂等复用相同决策/结果hash。

freeze验证20条全50epoch、五seed/条件、实际RAM selected/E10/E50与曲线的状态hash和最早严格最大值，绑定源/数据/计划/观测/分析/比较规则。evaluate显式启动独立Python进程；全部gate和数据身份核验通过后才允许`include_test=True`。固定源同data/seed/objective/state的旧Test直接复用，其余只评价冻结regular端点；物理alias及相同state去重。完整Test预测、模型、optimizer和参数均不落盘。

报告生成原始端点、绝对值mean/样本SD、五seed配对/CI、完整范围的规则决定、资源成本、Q1–Q5结构和PNG/PDF图。主agent负责最后的科学综合解释。必需产物全部从真实epoch/prediction/RAM来源产生；缺输入抛出Incomplete，不生成伪数值、空成功报告或补零结果。

## 产物入口

`audit_archives`：AUDIT_INVENTORY/EVIDENCE_REUSE/ARCHIVED_FACTORIAL及归档完整曲线/配对统计。

`analyze_stage(1/2)`：VALIDATION_CURVES、INNOVATION_STATS、PULSE_READOUTS、PULSE_METRICS、HISTORY_PREFERENCES、SELECTION_DIAGNOSTICS、TEMPORAL_RANK_STABILITY、RANK_FLIPS、GROUP_INTERACTIONS、PULSE_CONTRASTS/SUMMARY、PANEL_STABILITY、METRIC_CORRESPONDENCE与阶段1/2决定。

`fit_predictive_folds/analyze_stage(3)`：PREDICTIVE_ROWS/OOF/FOLDS/COMPARISONS、PREDICTION_SEED/SUMMARY、阶段3全部规则决定与有限控制触发。

`run_controls/analyze_stage(4)`：CONTROL_RESULTS/RECEIPT、CALIBRATION_SUMMARY、BACKGROUND_SUMMARY、STAGE4_DECISION、MECHANISM_DECISIONS。

`freeze/evaluate/generate_report`：SELECTION_FROZEN、FREEZE_RECEIPT、ENDPOINTS、TEST_COSTS/RECEIPT、ENDPOINT_ABSOLUTE_SUMMARY、PAIRED_SEED/SUMMARY、COSTS、ANALYSIS_COSTS、ACTION_DECISIONS、FINAL_REPORT、PNG/PDF图和REPORT_RECEIPT。SOURCE/DATA/PREPARED/PROBE/验证与smoke等产物由core/protocol维护；report检查完整必需清单并按本轮2GiB观察cap验收。

## 基本检查

命令：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/ema_mechanism_20261106/tests/test_analysis.py experiments/ema_mechanism_20261106/tests/test_evaluation.py`。

结果：**11 passed**。覆盖SSE-ridge与截距、F回退/并列、源LL端点与温度恒等、576/240的seed/时间/行隔离、仅训练折标准化、未来新pulse标签和24支持、pair padding、seed聚合、预算不足拒绝freeze、篡改冻结拒绝、独立Test子进程。两模块py_compile通过。

本实现阶段未运行正式训练、未访问新Test、未计算本轮研究结论。GPU工程smoke、真实20轨迹与全流程产物由独立审查通过后的runner执行。

## 修订 R3：Validation masked cap

按 `DESIGN_SPECIFICATION.md` 第11节，V_mech的每域人数是 `min(4096, available)`。分析端统一由身份NPZ已有 `indices>=0` 派生有效mask，不增加持久字段。`_panel_metric_matrix` 的完整V_mech/行分侧指标只取有效行；这同时覆盖LOSO历史偏好。排序pair在gather前验证每个索引落在有效槽，temporal T和flip按pair_count取实际pair；flip的精确面板AUC另按有效行筛选标签和概率。校准拟合侧、评分侧分别取valid与side==0/1的交集，评分侧使用另一合法side，不能用拟合mask补集包含padding。

C_BG沿用core已修订的 `load_panels/panel_predictions/pulse_rows`，因此其自然V读出和派生pulse/未来标签继承同一mask。evaluation只读取官方完整Validation/Test端点，不消费有界V面板，未增加掩码分支。COSTS已对全部peak_*字段按job取max，自动接入core新增的实际guardian RSS成本字段。

新增一个最小不等长域测试，同时覆盖V_mech/V_A/V_B指标等于仅取真实行、pair不含无效槽、精确flip AUC与校准fit/score均排除-1标签。分析与评价测试现为 **12 passed**。报告另生成 `VALIDATION_PANEL_COUNTS.csv`，直接从固定身份记录逐seed/域有效人数及两行分侧人数。此修订无工程/科学Adam、无新Test读取，原attempt继续。

## 修订 R4：采集后派生表无损压缩与分析身份

20条轨迹完成后，S2的 `PULSE_CONTRASTS.csv` 为337,299,690字节，使观察目录超过2GiB。该表不在manifest必需清单中；唯一持久读取caller是报告图 `_figures`，S2/S3/S4实际统计使用同一epoch记录计算的内存contrast表。为保留完整逐行数值，写入与图读取改为 `PULSE_CONTRASTS.csv.gz`，CSV序列化及全部科研计算保持相同。没有删除原始PULSE_READOUTS、逐seed对比或summary。

等待既有S2进程成功结束后，将已生成CSV流式gzip，解压字节SHA256与原CSV完全一致：`6988ad1e93c9ee8b4eb6d437f364020d79039a5c644d05504d56d68e88b6b3fe`。验证完成才删除未压缩副本；gzip为12,926,160字节，观察目录降至1,891,241,685字节。实际字节与hash记于 `PULSE_CONTRASTS_STORAGE_RECEIPT.json`。S1/S2没有重新计算。

采集实现hash仍为 `e371a4c00352a9eba8f76b11c4f50f790c8c80cd02552feb4eb5a86502810aaf`。新增窄范围 `collection_identity` 对照SOURCE_RECEIPT冻结文件清单，除analysis.py/evaluation.py及各自tests外的删除、新增或修改均拒绝；源、计划、manifest身份也逐项匹配。PREPARED/SOURCE及已完成阶段收据保持原字节；`ANALYSIS_REVISION.json` 单列旧collection hash、当前完整文件hash与analysis文件hash，无时间变化字段，重复调用保持相同字节/hash。input_hashes、freeze与独立Test gate绑定此修订；端点和报告保留原采集hash，不把当前分析实现冒作采集实现。

新增无损CSV roundtrip和采集/分析身份分离测试，检查训练组件变动/额外文件被拒绝、原收据不变和分析修订幂等。分析与评价基本测试现为 **14 passed**。未重训、未增加科学矩阵、未访问新Test；后续S3等待独立复审。
