# DESIGN SPECIFICATION — A0：共享 S 的收益持续性与动作价值

设计日期：2026-11-09。状态：设计完成，待独立实现和审查。研究标识：MDMT_ADAPTIVE_EMA_20261109。本规范在读取新 A0 科学结果前冻结；不实施尚未触发的 A/B/C 训练。

## 1. 责任和来源

本阶段回答：当前训练面板可见的 S 组 Fast/Slow 收益，能否预测下一个观测窗口，并在任务或全局范围提供超过身份/阶段和简单参照的动作价值。A0 使用已有20条轨迹、五个 seed（2022–2026），新增训练次数、Adam steps、GPU 使用和 Test 访问均为0。

源码基底为 `26-11-07` 的真实 head `ffa3de067fcda5a1293af83a465d741e069493f1`，交付分支为用户要求的新分支 `26-11-09`。规划目录来自当前 main `21a7926945281333afa0b6ef6037d2428f1143a1`。原计划分支 `research-plan/26-11-09` 已不存在。main 和来源分支的 merge-base 为 `652ee0235fec2d0d4cfc67b4a6ef9977519619bf`；main 相对来源 ahead 83、behind 17。执行 audit 时由本地 Git 再算并记录方向，不能使用过时的本地 main head。

当前 main 已含机制最终报告，但没有 `experiments/ema_mechanism_20261106/` 实现；runbook 关于报告也返回404的陈述已过时。必须在 SOURCE_AUDIT 按实际树分别记录实现、报告、规划和上下文身份。只读复用历史文件，不修改历史代码、报告或总上下文。

用户 start.md 的 ONLY 2022–2026 优先于规划未来2027–2031的建议。保留原规划文件字节，在解析配置中显式记录覆盖。A0 可以完整执行。若阶段门支持继续，A1 的独立确认设计必须先修订并独立审查；不得将同 seed、同轨迹重跑称为新的独立 seed 确认，也不得擅自使用2027–2031。

## 2. 最小实现范围

新增包 `experiments/ema_adaptive_20261109/`，包含少量直接模块（例如 audit/protocol、analysis、run、tests）；不要求照搬规划中尚未触发的控制器文件结构。复用历史统计、ridge 和表格工具；不导入会启动训练或读取 Test 的路径。

CLI 至少支持：

```text
python -m experiments.ema_adaptive_20261109.run audit --manifest <plan_manifest> --out <attempt>
python -m experiments.ema_adaptive_20261109.run analyze --stage A0 --manifest <plan_manifest> --out <attempt>
python -m experiments.ema_adaptive_20261109.run report --manifest <plan_manifest> --out <attempt>
python -m experiments.ema_adaptive_20261109.run release --manifest <plan_manifest> --out <attempt>
```

首条科学执行命令必须为 audit。实现者只做语法和小型正确性检查；独立 reviewer PASS 后，runner 执行 audit/analyze 并记录原始产物。科学解读由主 agent 完成。新代码不保存、加载或生成模型参数/checkpoint 文件；仅处理已有指标、日志、面板身份和报告。

## 3. 来源审计与恢复

审计 manifest 列出的全部13个 `required_source_artifacts`，外加 `PREDICTIVE_ROWS.csv`、`PREDICTIVE_FOLDS.json`、`RESOLVED_PROTOCOL.json`、`GROUP_MAP.json`、`PROBE_MANIFEST.json`、`SOURCE_RECEIPT.json`、`ANALYSIS_REVISION.json`、`raw_epoch_records.tar.gz`、源码依赖和 main 上下文快照。记录路径、字节数、SHA-256、实际 Git blob SHA、来源 commit 和用途。对 tracked 文件将实际字节 blob 与来源 commit 的 Git blob 比较；不把“本次自算 hash”当作预存参考。规划及 main 快照按 REMOTE_SOURCE_IDENTITY 核验。

区分采集实现 `05f75a73...` 与后续无损 CSV gzip 分析修订：历史 SOURCE_RECEIPT 的采集 hash 不能要求等于修订后 analysis.py；按 ANALYSIS_REVISION 验证对应链。`.csv.gz` 解压 checksum 仅在实际使用该表时验证，不为未使用衍生表增加新验算链。

PULSE_READOUTS 已直接含 P_A/P_B/V_mech/V_A/V_B，因此正常路径不需要复制2GB预测文件或重建模型。用归档 JSON 的 pulse_readouts/pulse_metrics 做一遍标量行与对应源表的身份/数值核对；直接流式读取 tar，不在磁盘解包训练状态。归档可恢复丢失的上述两张表，重建到本次运行目录而不覆盖来源。若必要字段仍缺失或与来源 blob/归档冲突，输出 SOURCE_INCOMPLETE，不能记为阴性；只恢复已有日志，不重新训练以绕过来源门。

`PROBE_MANIFEST` 定义的 P_A/P_B 面板文件检查 SHA、panel/pair hash 及训练行无交集；V_A/V_B 为同一 V_mech 有效行的固定 side 0/1，校验 valid_mask 和行侧不交叠。无需读取自然原始 CSV 或 Test 标签。P_A/P_B 的 AUC 来自固定排名探针，LogLoss 来自自然探针；V_A/V_B 均为自然验证行。两种支持不同，报告必须保留这个差别。

## 4. 主数据集、键和差值

只分析 S 参数组。源表保留全部组的身份供审计，不新增其他组实验。源 pulse 是从窗口起点 F 初始化、恰好一个 epoch 的干预，结束于 e=5,10,…,50；观测间隔5 epoch。它不是持续5个 epoch 的干预。

PULSE_READOUTS 主键：`(job_id, epoch, panel_id, physical_readout_id, domain, task)`。先检查重复；完全相同的别名只保留一次，非相同重复为冲突。用 `variant=F, group_id为空` 作本窗固定参照，`variant=fast/slow, group_id=S` 作 S-only 混合读出；不得误用 `all_fast/all_slow`。

规范化键：`(dataset, objective, seed, epoch, panel_id, domain, task, group_id=S)`。每键保留 F/fast/slow 的 auc、logloss、state_hash、实际样本数、missing_reason 及轨迹/面板身份。完整矩阵为2 datasets ×2 objectives ×5 seeds ×10 windows ×5 panels ×6 cells。V_mech 可作为描述，不参与两行侧均值时重复计权。

计算：`g_fast=auc_fast−auc_F`、`g_slow=auc_slow−auc_F`、`D=g_fast−g_slow`；LL 效用另算 `l_fast=LL_F−LL_fast`、`l_slow=LL_F−LL_slow`。正 LL 效用表示损失下降。Fast−F、Slow−F、Fast−Slow 独立标记，不混为一个样本池。缺失保持 NA 并记录原因，不填0；不跨缺失窗口连 lag。核心格缺失先按来源恢复路径处理，完整五 seed 之前不下否决结论。

PULSE_METRICS 键为 `(job_id, epoch, domain, task, group_id, panel_id)`，仅取 S/P_A。与 readout 主键一一对应后复用源特征。`actual_count` 在 metrics 中是参数坐标数，在 readouts 中是面板人数，不能混用。

## 5. 直接持续性与迁移

对每个 seed/condition/panel/cell 和任务/全局聚合，按 epoch 排序配对 e→e+5（e=5…45），输出逐窗当前和未来 g_fast/g_slow/D 及 LL 效用。任务为同一 task 三个 domain 等权均值，全局为六格等权均值；先聚合真实 g 再计算统计。

逐 seed 计算 sign agreement、Pearson、Spearman、Kendall；符号主统计排除任一侧等于0的 pair，另列零收益率、含零完整 sign agreement 和有效pair数。常数序列的相关为 NA，给出有效数。报告全窗和预定义 early origins5/10/15、late origins25/30/35/40/45；origin20 作为时间隔离，不用于拟合或评分。

同时做 P_A(e)→P_B(e+5)、P_A(e)→V_A/V_B(e+5)，及反向 P_B(e)→P_A(e+5)。同窗 P_A→P_B 只描述测量迁移，不能替代前瞻结果。报告 AUC 差和 LL 差，动作仍只用 AUC。

原始 lag 符号率还要与本设计的身份/阶段和静态政策比较；正向自相关或超过0.5的符号率本身不能触发训练。所有 cell/window 为 seed 内观测，最终五 seed 均值/样本SD/CI，不能把窗口数作为科学 n。

## 6. 新的训练面板预测审计

固定 outer LOSO 五折；每次只以其他四 seed 的 origins5/10/15 拟合、held seed 的 origins25/30/35/40/45 评分。target_epoch=origin_epoch+5。按 dataset 分别拟合、含两个 objective，每行仅 S 的一个 cell：144行拟合、60行留出评分。目标是未来新 pulse 收益，而非当前 pulse 延迟生效的因果效应。

主方向的拟合输入与拟合标签均来自 P_A：输入当前窗口特征，双目标为未来 P_A 的 `[g_fast,g_slow]`。评分特征只取 held seed 当前 P_A，所得预测和动作在未来 P_A/P_B/V_A/V_B 上分别评分。P_B 和 V 不进入主政策拟合、动作、超参或模型选择。另作预定义 P_B→P_A 对称诊断，只用于迁移核验，不能替换表现不利的主政策。

参照与候选：

- `C_design`：source 的固定 condition×cell×group 48-way onehot，加 epoch/50；S-only 行仅12个身份列活跃，空列保留或删除等价。
- `C_simple`：源 RESOLVED_PROTOCOL 的 design/geometry/function_amplitude/gradient_amplitude/global_preference 特征，原字段原义。当前 all-fast/all-slow 六格特征仅作为已存在简单参照特征，不当成 S-only 动作真值。
- `C_persist`：C_design 加当前 P_A 的 `[g_fast,g_slow]`，仅这两个额外收益特征。它检验实际收益在身份/阶段之外的信息；不加平滑、阈值、额外 HPO。
- `C_carry`：无需拟合，直接以当前 `[g_fast,g_slow]` 预测未来收益，作为持久性的直观参照。
- M0/M2_full/M2_detach/M3_direct：按源 feature_sets 在相同 S-only、P_A-only、LOSO-forward 方案重拟合，供与原信号家族比较，不增新特征。

直接复用 source `feature_matrices`/`ridge_predict`，传入固定 spec；所有连续量仅用训练折均值与 ddof0 SD 标准化，常数列归0，cos 缺失沿用源规则。ridge alpha=1，截距不罚。固定模型无需内层调参：outer seed 留出与内层 early→late 时间隔离共同保证前瞻评分，不增加内层HPO。若复用 helper 的固定行数检查写死576/240，只在新包适配为144/60，不改历史实现。

反向 P_B 特征没有 M2 梯度：只重复 C_design/C_persist/C_carry 三个可完全由 P_B 当前收益构造的模型，不伪造 P_B 梯度或挑选有利方向。

模型前瞻方向准确率及阶段门中的 raw-carry 与 C_design 方向比较，统一以真实未来 D≠0 的行作共同分母；预测 D=0 计为错误，raw carry同样处理，沿用历史前瞻预测约定。第5节的描述性 lag 符号一致率仍使用两侧均非零分母，并另列零率与完整符号一致率。

原 PREDICTIVE_OOF/PREDICTION_* 另列 `HISTORICAL_VALIDATION_FIT` 对照：其拟合标签来自 V_A/V_B，不能当本次 P_A-only 政策或与新预测器的分布差异混淆。复用时仅保留 S 与唯一 `(dataset,objective,seed,origin_epoch,target_epoch,cell,group_id,model_id,label_panel,row_fold_direction)`。ACTION_DECISIONS 嵌套 scopes 按唯一 scope/cell/group/model/window 身份记录，不累加条数为独立证据。

## 7. 动作和静态参照

动作集合固定 `[F,fast,slow]`，分数 `[0,yhat_fast,yhat_slow]`；精确平局优先F，其后fast、slow。任务动作先将该任务三格预测等权平均，再 argmax；全局先平均六格再 argmax。对真实未来收益取相同聚合和同一动作，不能平均六个各自最优动作后声称全局收益。

记录 chosen−F、chosen−uniform、chosen−identity_stage、chosen−C_simple、regret（同范围 oracle−chosen）、direction accuracy、MAE/RMSE fast/slow/D、Pearson/Spearman/Kendall、动作缺失率、切换次数和连续动作窗口长度。窗口长度同时用“观测窗口数”和跨度epoch表示；不声称观测间隔内动作持续不变。反馈延迟为5个epoch的观测间隔；原pulse本身只有1 epoch。LL 效用按 AUC 选出的同一动作独立评分。

静态政策与统一参照都只从其他四seed的 early P_A 当前收益估计：每个 dataset/objective/heldout_seed，静态任务政策分别按task三格均值选择一个固定动作；统一政策按六格均值选一个固定动作。之后在 held seed late target30/35/40/45/50 的 P_B/V_A/V_B 评分。另作 P_B训练、P_A及V评分的反向检查，保持为诊断而不择优。额外列全程F/统一fast/统一slow三种机会参照。静态task政策必须比相同信息来源训练的统一政策有增量，单独选中非F不构成任务偏好价值。

## 8. 预先冻结的 A0 阶段门

主判断范围只用 KuaiRand Gaussian D0、S 的 `GLOBAL`、`TASK0`、`TASK1`。四个 dataset×objective 和六格结果完整输出；其他条件或单个cell的阳性保留为局部现象，不据此触发全局训练。P_A是固定主信号侧，不按数据改主侧。

所有“正向稳定”定义为：五seed均值严格>0，且逐一去掉任何一个seed后的四seed均值均≥0。这允许单个seed负值，并防止一个seed独占均值。报告全部五个原值、配对差和 `mean ± 2.776445×sampleSD/√5`；CI不是额外否决阈值。

动态门在至少一个主范围同时满足：

1. C_persist 的未来 D 的 MAE 同时优于 C_design 和 C_simple；五seed配对 `MAE_reference−MAE_persist` 均按上述正向稳定规则通过。
2. C_persist 的动作 AUC效用相对F、统一政策、C_design、C_simple的五seed配对增量均正向稳定。
3. 1–2在 P_A→P_B 未来评分成立；同一主范围在 V_A/V_B 两侧先在seed内等权平均后也成立。两侧单列，不以一侧替换另一侧。
4. 原始前瞻 `D(e)` 对 `D(e+5)` 的方向率相对固定身份/阶段预测器有正向五seed增量；P_B→P_A 的方向和动作−F均为正均值。此反向检查不要求缺失的P_B M2特征，也不改主政策。

满足为 `PERSISTENCE_SUPPORTED`。具体幅度和六格代价照实报告，不使用发表模型+.003作为内部组件门。

若动态门不通过，独立判断静态门：KuaiRand Gaussian 的任务政策−F、任务政策−统一政策，在主P_A训练后 P_B和V_A/V_B平均评分均正向稳定；反向P_B训练后P_A和V评分的同类均值为正；至少一个任务的增量为正向稳定，并列出另外一个任务及每任务三个域的真实响应。任务/域反向代价均须保留，不把单个cell好处当task成功。满足为 `STATIC_PREFERENCE_ONLY`，只允许后续静态替代设计，不启动动态控制。

静态门的范围明确为：以上正向稳定与反向正均值先在GLOBAL聚合任务政策判断；此外至少一个TASK在主P_A训练后的P_B及V_A/V_B均值两类评分上，其对F和统一政策的增量均正向稳定。TASK级反向结果仅作诊断，不增加反向稳定性或正均值否决条件。

两个门均不通过为 `NO_ACTIONABLE_SIGNAL`：在已测5-epoch观测分辨率下停止新增训练，固定F保留；逐epoch反馈在本阶段没有观测。来源不完整或hash冲突时为 `SOURCE_INCOMPLETE`，不能落入上述科学阴性。决策 JSON 输出每条 gate 的数值、逐seed输入、通过/未通过及证据路径；主 agent 依据冻结规则解释，runner不撰写科学结论。

## 9. 交付与成本

运行目录 `artifacts/ema_adaptive_20261109/<attempt_id>/`；正式报告 `research_reports/2026-11-09_ema_adaptive/`。至少输出 SOURCE_AUDIT.md、AUDIT_INVENTORY.json、EVIDENCE_REUSE.csv、RESOLVED_PROTOCOL.json、RESOURCE_ESTIMATE.json、A0_PERSISTENCE_ROWS.csv、A0_SEED.csv、A0_SUMMARY.csv、A0_PREDICTIVE_OOF.csv、A0_PREDICTIVE_FOLDS.json、A0_STATIC_ACTIONS.csv、A0_DECISION.json、COSTS.csv、IMPLEMENTATION_REVIEW.md、FINAL_REPORT.md、RELEASE_RECEIPT.json。可按清晰职责拆分逐窗/逐seed表，避免一张宽表伪造不适用字段。

未触发A1/B/C时以阶段状态说明，不生成空的“正式训练结果”代替未运行实验。必须保留解析后的manifest和设计SHA、实现hash及运行命令。报告逐seed、六格/任务/全局AUC和LL效用、阶段、参照、来源state_hash与窗口身份；A0没有新checkpoint，不造selected epoch。历史完整模型端点只从既有报告引用为背景，不能当A0新终点。

资源估计读 GROUP_MAP 的真实 records，求S坐标numel并与counts、metrics actual_count相核：ML483152/KR661920，FP32 payload1932608/2647680字节。A0只作标量核验，不实例化或保留这些参数。记录源输入字节、输出字节、进程RSS峰值、CPU和墙钟时间、新GPU秒/Adam步/Test访问0；RSS及输出低于既有90%主机RAM/2GiB输出门。源预测无需复制。release关闭文件并记录零参数状态与产物hash；不删除历史数据。

## 10. 必要正确性检查

只做影响结论的小型测试：

- F/fast_S/slow_S唯一配对、符号和负LL效用方向；相同alias去重与冲突报错。
- e→e+5且不跨缺失窗口；训练seed/score seed与early/late严格隔离；fit_transform不读取score或V。
- 任务/全局先聚合后选动作；tie优先F；静态task和uniform共享同一训练信息；V更改不能改变预测/动作。
- 常数相关/零D为NA并计数；seed先聚合，五seedCI和配对差正确，窗口不增加n。
- 固定人工小表测试动态/静态/无信号/来源缺失四个门；包含一个负seed但总体稳定的情形。

这些测试使用手算小表，不以研究源数据尝试选择阈值。reviewer 独立对照本规范、代码和 CODER IMPLEMENTATION REPORT；所有blocking修正后才执行A0。

## 11. 报告分析要求

主 agent 分别评价稳定重复现象、与历史结论的矛盾、竞争解释、证据强度和最能区分解释的后续实验。证据标 STRONG/MEDIUM/WEAK，幅度与强度分开：五seed配对95%CI完全正向且跨训练/验证面板一致可支持 STRONG；稳定均值但区间跨0为MEDIUM或WEAK，需结合实际离散度；单cell/同窗/历史拟合背景为WEAK，不升级阶段。

明确区分训练探针噪声、身份/阶段可预测性、真实收益持续性和未来动作价值；保留AUC与LL取舍。域规模只列已有自然人数作为背景，域难度以对应基线AUC/LL描述，不作为控制输入或追加研究变量。A0没有操纵域规模，也不检验small-domain improvement/large-domain decline的因果关系；报告写出实际六格有无相反响应及识别范围，不增加规模实验。停止条件成立即完成本次已触发实验、报告与新分支交付。
