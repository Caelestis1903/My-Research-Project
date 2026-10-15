# 原计划全范围及条件节点判定

主 agent，2026-10-16。以 `start.md`、原计划§7–12、§18预算/依赖和本次真实工件为依据。用户最新禁止子agent，后续设计、实现、self-review、运行、分析均由主agent顺序执行；未声称取得独立review。本文件记录科学条件与无法恢复的历史要求，不将未运行项改写为成功。

| 项目 | 实际执行/证据 | 本轮判定及未完成原因 |
|---|---|---|
| A0历史身份与实现审计 | `artifact_inventory.json`、`missing_artifacts.json`、`runner_objective_optimizer_callgraph.md`、数据hash与bounded recovery记录 | 可访问部分已审计；历史S5/S6原始manifest/config/checkpoint/prediction/optimizer lineage缺失，完整历史闭环 **INCOMPLETE**。不补造source SHA或原始状态。 |
| A1历史结构/梯度证据 | 可访问CSV、AO04预测复算、rho及可达性静态检查 | 预测点算术可复核，缺失历史状态使完整logit Jacobian、frozen-θ预测敏感度与所有历史seed分析 **INCOMPLETE**；未把融合可辨识性当作已证实。 |
| A2历史F/DH反转 | S6-R2 CSV重算、AO04 paired row bootstrap、split计数 | 原exact selected/grid身份保留；缺失旧逐epoch预测/state，历史full selected替代规则与不确定性 **INCOMPLETE**。新研究不填充旧空缺。 |
| B1历史精确dense复现 | 历史初始化pin与当前初始化不匹配，原lineage无法恢复 | **NOT_RUN_AS_HISTORICAL_EXACT**；无法证明原协议身份。已执行另列的新重建研究，不能称原8/8精确复现。 |
| B1重建测量 | `B1_reconstruction_20261016/analysis/analysis_receipt.json`；8/8、400state、800预测文件；`scientific_analysis.md` | **COMPLETE_RECONSTRUCTION_ONLY**。主要R0 test宏AUC差+0.000363、2/4正；R1/R4正向线索及target LL代价完整披露。按§7停止该线的自动方法开发；没有把阴性扩大解释为所有policy均无效。 |
| B2 Equal C×A最多16runs | A0有效S5 source/derived state、F/T同协议anchors与独立来源验证前提仍缺失 | **NOT_RUN_UNMET_PREREQUISITES**。Sample F/DH重建不构成Equal F/T anchors。没有因显卡空闲越过条件启动16runs。 |
| C1 m×v×current探针 | 缺有效S5 state及同θ轨迹ghost/shadow histories；B2未运行 | **NOT_RUN_UNMET_PREREQUISITES**。当前F/DH末端state来自异轨迹，不能硬拼后当作m/v因果；fresh step0 oracle不代替persistent-history探针。 |
| C2.1固定参数oracle | `C2_numeric_oracle_20261016/run_receipt.json`及`numeric_analysis.md`；两数据集×4seed完成8case、0training | **COMPLETE_NUMERIC_ONLY**。同α期望梯度匹配，Balanced对Equal减方差、对Sample增方差；支持有限pool/初始化θ的反例，不支持selected泛化收益。 |
| C2.2配对训练每数据集可选16runs | 尚无独立确认的同目标selected实用gap及完整曝光匹配训练协议；本次B1不改变q或alpha | **NOT_RUN_NO_TRAINING_GO**。完成oracle是必要条件而非自动充分条件；不将初始化梯度方差差异提升为训练收益。 |
| D1域身份/比例/任务/骨干/外部数据 | 无跨规则稳健的主要selected收益或已识别机制；没有独立未开发holdout | **NOT_TRIGGERED**。原计划为条件性单因子研究，不是本轮全网格义务；R1/R4回顾性test阳性不自动触发。 |
| E1新方法及其基线/消融 | 新科学gap、Sample强基线稳定selected收益、non-target guard、独立holdout和novelty条件未满足 | **NOT_TRIGGERED**。没有方法候选，因此未启动其后依赖的MMoE/PLE/SGD/AdaTask/PCGrad/CAGrad/FAMO/SAF及模块消融矩阵；不声称这些方法已被本轮比较或否定。 |

## 科学停机依据

R0为主规则，R1–R4仅为预定法证敏感度。R0没有稳定实用selected收益，但R1/R4的test非目标收益4/4正，必须保留为局部线索；两规则在本研究选中完全相同的checkpoint，不能算独立复制。selected target LL代价、seed2024和同epoch/阶段结果限制推广。详见完整科学分析及全部原始统计表。

下一项最能区分解释的是独立时间后置数据上预注册的policy×selection测量，包含AUC与LL双指标、同epoch/selected及目标/非目标成本。这是未来建议，未擅自改变本轮定义或启动新训练。历史原始材料无法恢复的部分继续明确列为未完成；`start.md`最终验收第1项允许明确记录未完成原因，不要求伪造或越过原计划条件。

## 整体交付仍需验收

本轮必须完成三份最终报告、机器可读表图和配置/源码归档，并检查真实Git状态、提交及远端对应分支。仅有本文件或数值PASS不等于整体任务完成。最终完成凭据应记录这些实际文件与远端commit，而不是把历史缺失项标为成功。
