# MovieLens G-D0结果补齐独立审查

结论：**PASS**。无阻塞问题。补充结果、来源记录、双数据集表、主报告入口及本分支台账增量符合更新后的任务要求，可提交推送至原分支。

审查日期：2026-11-04。reviewer与本补充coder为不同agent。任务依据为更新后的buchong.md及既有冻结Gaussian/统一评价协议；本次没有新增研究设计或算法实现。

## 1. 审查范围与实际复用

独立读取supplement_movielens.py、CODER_MOVIELENS_SUPPLEMENT_REPORT.md、MOVIELENS_SOURCE_AUDIT.json、导出CSV/JSON、MOVIELENS_SUPPLEMENT.md、G_D0_TWO_DATASETS.md、build_report.py的5行增量，以及MDMT_ALGORITHM_METRICS_LEDGER.md的新补充段。

- MovieLens：Gaussian G-D0复用5条，M3oE/HyperGate/AutoIFS各复用5条，共20条既有轨迹、40个selected V/Test端点。
- KuaiRand：跨数据集表引用既有G-D0与三个发表模型的20条比较轨迹，原270行端点保留。
- 本次新增训练0条、新增seed0个；每个配置保持2022–2026五seed。同源命名映射及V/Test两行不增加独立重复数。

## 2. 身份、选点与评价核查

47个审计来源文件的SHA256均独立复核通过。40行补充端点按原AB表行号逐字段核对，除明确的arm别名和补充溯源字段外，原字段值完全相同；MovieLens实际trajectory_id去重为20个。

原MovieLens GAUSSIAN_A1/fixed_ema使用augmented_domain=0，固定EMA分支H2，因此映射为G-D0。来源配置保持fixed1 M3oE、自然原行序80/10/10、train-only预处理、batch40960、Adam lr=.01、coupled weight_decay=1e-4、Gaussian κ=.05、λ=.5、ρ=.9828205985452511、50epoch和FP32。原GA31-R3噪声key、loader seed+11000及Gaussian/model关键原语与冻结源码哈希一致。

MovieLens train=800167，validation/test各100021，train域人数155949/318599/325619；六数组、row_id、encoder/scaler及原CSV身份匹配。KuaiRand表明确保留κ=.1、wd=1e-5、ρ=.995，不套用MovieLens参数。

四组模型每seed的合法validation窗口均为完整epoch1–50。Gaussian选中epoch依次为23、20、25、23、22，按六格Macro AUC严格最大重算一致，精确并列取最早；同seed的V/Test具有同epoch和model hash。三个发表模型使用同自然人口与AUC选点协议，HyperGate为train阶段，AutoIFS为reuse阶段；原生封存端点及来源checksum对应一致。

## 3. 数值和裁定

独立从逐seed端点复算双数据集表全部六格AUC/LogLoss、Macro、Traffic、Pooled的均值与样本SD，并独立复算MovieLens三个比较的全部配对均值、SD及df4 t95%区间，通过。

| 数据集G-D0 | n | Test Macro AUC ± SD | Test Macro LogLoss ± SD | 最强发表模型 | 平均ΔAUC | 裁定 |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | 5 | 0.779356602 ± 0.000393344 | 0.495822265 ± 0.001550074 | M3oE | +0.004166933 | 强算法、六格保护通过 |
| KuaiRand | 5 | 0.679180178 ± 0.003474732 | 0.435517854 ± 0.002752362 | HyperGate | -0.002932373 | 未超过最强发表模型、六格保护未通过 |

MovieLens对M3oE的六格平均AUC增益均为正；Macro LL平均上升0.000931348、4/5 seed上升，DECISIONS中logloss_improved=false。AUC等级与LogLoss分别裁定正确。对HyperGate/AutoIFS的平均AUC增益分别为+0.007944846/+0.006932013，六格保护通过，平均LL均下降。置信区间及seed方向数没有被额外用作性能否决门。

补充说明、主报告入口及总台账均明确MovieLens G-D0已经完成五seed；双数据集表展示六格绝对AUC/LL、Macro AUC/LL及各自超参数。原KuaiRand范围及其10条新增训练计数继续保留，补齐未被记为新训练。

## 4. 独立执行与冻结身份

使用项目解释器执行supplement_movielens.py：全部来源、配置、完整选点和数表断言通过，状态COMPLETE_REUSE。独立检查程序另核验47文件SHA、40行精确导入、两个数据集各5seed、20个MovieLens独立trajectory、双数据集均值/SD及全部配对区间。supplement_movielens.py和build_report.py的py_compile通过。

- 补充脚本SHA256：6458828c2cba029ee23e732fcd2c071a0f6ce19db45f91e286b337588438ff15。
- 原270行ENDPOINTS.csv SHA256：5de7c064542d19b79bcae49325c49f7e40fb35f41b21e7e1b0a236cb0bf69875，保持不变。
- 原训练源码hash：c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d，保持不变。

未运行GPU或新增训练；全部所需MovieLens seed已经合法复用。
