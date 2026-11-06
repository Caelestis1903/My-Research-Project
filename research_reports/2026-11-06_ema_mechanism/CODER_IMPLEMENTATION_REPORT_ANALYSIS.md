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
