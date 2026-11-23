# S1 共享更新与专有路径顺序诊断

**裁定 `COMPLETE_NO_GO_S1`。** 共享 Adam 子步改变了六格排序和损失。`P_AXIS` 在 KuaiRand 的选择面板 V_A 达标并被锁定。独立面板 V_B 上，两套数据均未通过联合累积对照与阶段复现条件，S2 的持续训练门关闭。 [冻结规则与裁定](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

## 实验和对照

MovieLens 和 KuaiRand 各使用 seeds 2022–2026，运行 canonical M3oE fixed1、自然两任务 BCE、FP32、batch 40,960 和 50 个 epoch。Adam 的学习率为 .01，β 为 (.9, .999)，两套数据的 coupled weight decay 分别为 1e−4 和 1e−5。固定 EMA 的 ρ 分别为 .9828205985452511 和 .995。真实首批和尾批的输出、模型、Adam 和 EMA 状态与 canonical 实现精确匹配。 [配置与来源](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) · [S0 验收](S0_SMOKE_REVIEW.md)

`U` 包含 embedding、STAR、skip、共享 experts。`P_CELL` 包含每格的 Gate/Tower，`P_AXIS` 包含域轴与任务轴的 experts，`P_FULL` 为两者的并集。实际六格审计记录了 Gate 输入 `detach`、词表行支持以及固定融合下的轴外零梯度。`U→P` 在 A 更新 U 后，从更新后的状态在 B 重算梯度并更新 P；反向顺序使用同预算。每个锚点执行 AB/BA 互换、全参数两步 J2、同状态两批梯度合并后一步的 J_ACC，以及 U/P 各自累积对照。 [设计与参数分组](DESIGN_SPECIFICATION.md) · [真实梯度支持](../../artifacts/mdmt_shared_private_coordination_20261123/S0/PARAMETER_SUPPORT.csv)

统计以 seed 为配对单位。先在 E10、E25、E40 各自平均 AB 和 BA，再在每个 seed 内平均三个锚点。V_A 只用于选择，V_B 只确认冻结范围。Online 是诊断主读出，固定 EMA 是次级读出。以下差值为候选减参照，表中以 10⁻⁶ 为单位。 [逐 seed、区间与敏感性](../../artifacts/mdmt_shared_private_coordination_20261123/PAIRED_SEEDS.csv)

## H1 · 共享子步引起六格异质响应

**MEDIUM · 当前模型和人口下的局部响应。** 在完整 Validation 上，MovieLens 的单次 U 更新使 Macro AUC 平均变化 −0.000313，五个 seed 均为负，配对 95% CI [−0.000520, −0.000106]。六格均值全部下降，范围为 −0.000465 至 −0.000187。E10、E25、E40 的 Macro 均值分别为 −0.000420、−0.000353、−0.000166。E10 的 LogLoss 平均降低 0.001125，E25 和 E40 则分别上升 0.002326 和 0.003220。 [完整响应表](S1_NUMERIC_READOUT.md#literal-online-post-u-response-and-after-p-change)

KuaiRand 的两个 d0 cell 分别下降 −0.000310 和 −0.000330，两个 d1 cell 分别上升 +0.000068 和 +0.000094，d2 两格方向不同。d1 的平均绝对 logit 位移约 0.087–0.094，d0 为 0.039–0.046。独立复核发现，d1 两个任务的绝对 logit 位移同时高于 d0 的情况出现于 14/15 个 seed–锚点组合，绝对样本对分差位移在 15/15 个组合都出现相同次序。d0t1 的固定验证对中有 1,091 次顺序等级改善和 1,212 次下降，d1t0 则为 1,270 / 1,203。完整 Validation 的 Macro AUC 平均变化 −0.000075，跨 seed 区间 [−0.000622, +0.000473]，E40 的均值为正。固定对重复用于多个锚点和种子，区间以五个 seed 计算。 [逐格响应](../../artifacts/mdmt_shared_private_coordination_20261123/CELL_RESPONSE.csv) · [固定样本对](../../artifacts/mdmt_shared_private_coordination_20261123/PAIR_TRANSITIONS.csv) · [独立复核](S1_REVIEW.md#scientific-interpretation)

## H2 · 专有路径适配和顺序效应

V_A 上三种范围中只有 KuaiRand 的 `P_AXIS` 同时超过 U_ONLY、反向顺序和 J_ACC，依次为 +9.618、+7.309、+30.862。MovieLens 的三个范围相对 U_ONLY 均为负。按预注册规则，KuaiRand 使 `P_AXIS` 成为唯一合格的冻结范围。 [V_A 全部排序与对照](S1_NUMERIC_READOUT.md#registered-selection-and-confirmation)

| V_B online | U→P_AXIS 对照 | Macro ΔAUC ×10⁶ | 95% CI ×10⁶ | 正向 seed |
| --- | --- | ---: | --- | ---: |
| MovieLens | P_AXIS→U | +11.265 | [−19.429, +41.960] | 4/5 |
| MovieLens | J_ACC | −1.147 | [−103.616, +101.322] | 2/5 |
| KuaiRand | P_AXIS→U | +52.746 | [+3.875, +101.616] | 4/5 |
| KuaiRand | J_ACC | −82.005 | [−251.364, +87.354] | 1/5 |
| 两数据等权 | P_AXIS→U | +32.006 | [+9.704, +54.307] | 5/5 |
| 两数据等权 | J_ACC | −41.576 | [−137.970, +54.818] | 1/5 |

**STRONG · 冻结的 S1 决策。** V_B 两套数据的六格均值均通过 −0.001 保护线。KuaiRand 仅 E25 同时超过反向顺序和 J_ACC，MovieLens 没有达到这两个条件的锚点。两套数据对 J_ACC 的总体均值均为负。J_ACC 对照在 MovieLens 的 E25 和 E40 为五个 seed 全负。这些结果触发的是预注册的推进门，完整训练增量尚无观测。 [分阶段配对结果](S1_NUMERIC_READOUT.md#registered-selection-and-confirmation)

**MEDIUM · 对反向顺序有局部优势。** KuaiRand 对反向顺序的平均增量在三个锚点均为正，V_B 的 AB 和 BA 却分别为 +125.284 和 −19.793。MovieLens 的 V_B 两个映射也反号。完整 Validation 上，KuaiRand 的 E25 专有更新部分收复 U 更新后的排序下降，E40 则在 U 更新已带来正均值的基础上进一步提高。跨阶段专有更新的 ΔAUC 为 +34.533，配对区间 [−113.536, +182.602]。MovieLens 的对应均值为 −2.644，E25/E40 均为五个 seed 全负。批次组成、训练阶段和普通梯度累积仍可解释部分变化。 [批次互换与中间状态](S1_NUMERIC_READOUT.md#endpoint-auc-and-logloss)

**WEAK · 由概率损失推断额外排序收益。** V_B online 下，`P_AXIS` 对 J_ACC 的 Macro LogLoss 分别降低 0.002431 和 0.000239，但两套数据的 Macro AUC 均值都低于 J_ACC。相比双步 J2，两套数据的 Macro AUC 均值分别提高 +0.000489 和 +0.000405，LogLoss 同时下降。J2 对全部参数执行两步 Adam，P_AXIS 顺序臂只各更新指定组一次。全参数 J_ACC 还更新顺序臂保持不动的 `P_CELL`。对反向顺序的比较匹配组更新次数。固定 EMA 的 V_B J_ACC 排序差为 +1.789 和 +0.812 ×10⁻⁶，体现了读出和时间平滑对短步差值的影响。 [AUC、LogLoss、流量加权与 pooled](S1_NUMERIC_READOUT.md#endpoint-auc-and-logloss)

## 域规模、任务和基线难度

| 数据集 | Train 行数 d0 / d1 / d2 | J_ORIG selected EMA 域均 AUC d0 / d1 / d2 | P_AXIS 对 J_ACC 的 V_B 域均 ΔAUC ×10⁶ |
| --- | --- | --- | --- |
| MovieLens | 155,949 / 318,599 / 325,619 | .767988 / .756619 / .753927 | −14.404 / +22.998 / −12.036 |
| KuaiRand | 172,782 / 1,743,898 / 89,452 | .677939 / .717428 / .669996 | −238.367 / −17.299 / +9.651 |

基线排序难度用同一人口下的 AUC 衡量。MovieLens 最小域的基线 AUC 最高，KuaiRand 最大域的基线 AUC 最高。KuaiRand 的 `P_AXIS` 相对 U_ONLY 在较小的 d0 增加 +244.401 ×10⁻⁶，最大域 d1 下降 −34.730，最小域 d2 仍下降 −186.310。换成 J_ACC，d0 转负而 d2 转正。MovieLens 相对 U_ONLY 三域均下降。域规模作为主要解释的证据为 **WEAK**，小域增益与大域下降的交换只在部分对照出现。 [六格、域和任务差](S1_NUMERIC_READOUT.md#endpoint-auc-and-logloss)

相对 J_ACC，MovieLens 两任务的 V_B online ΔAUC 为 +5.187 和 −7.482 ×10⁻⁶，KuaiRand 为 −77.148 和 −86.862。J_ORIG 的 selected Macro AUC 对 MovieLens 为 online .756665、EMA .759512，对 KuaiRand 为 online .686278、EMA .688454。对应 Macro LogLoss 为 .526497/.525966 和 .430564/.429661。KuaiRand 的五个 EMA 选点有三个落在 E50，MovieLens 五个均在 E18–E28。不同阶段和读出会改变基线与局部干预的比较，本轮证据限于 M3oE fixed1。 [基线和逐格结果](S1_NUMERIC_READOUT.md#selected-j_orig-validation-endpoints-and-support)

## 证据边界和下一项区分实验

11-22 的共享 EMBED 来源实验观察到小幅且不均匀的局部反应，后续脉冲未形成持续策略。D2 的局部差值同样依赖读出和对照，同时 KuaiRand 在 V_B 留下了匹配组更新次数的方向信号。两项实验的干预对象不同。 [11-22 B 方向报告](../conditional_history_shared_update_20261122/B_CORE_REPORT.md)

最有区分力的后续验证是在同一冻结 U 位移和同一 P batch 下，只切换 P 梯度来自 U 更新前还是更新后。两个臂从同一 Adam 状态开始，匹配 P 更新次数、实际 batch 位置和 EMA 时间刻度。应在预先锁定的独立面板比较配对 AUC 和六格保护。若优势只随 batch 映射出现，削弱更新后梯度适配这一解释。 [当前注册对照](../../research_plans/11-23/03_CAUSAL_EXPERIMENT_DESIGN.md)

## 执行和成本

十条基线共 500 个 epoch，30 个锚点执行了全部 570 个诊断 fork。正式训练及诊断分别调用原生 Adam 17,250 次和 990 次，自首次准入到收据发布耗时 1,807.5 秒。十条轨迹均在首次尝试以退出码 0 完成。S0 第一次烟测的会话退出使 RAM 服务离线，重跑并独立核验同一前台会话中的活体读回后才开始正式训练。S2 与 S3 未触发，Test 评价为零。 [执行报告](RUNNER_REPORT.md) · [实际成本](../../artifacts/mdmt_shared_private_coordination_20261123/COSTS.csv)
