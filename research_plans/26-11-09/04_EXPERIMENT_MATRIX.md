# 04｜分阶段实验矩阵与决策依赖

## 1. 总体矩阵

| 阶段 | 数据/目标 | Seed | 新在线轨迹 | 核心问题 | 决策 |
|---|---|---|---:|---|---|
| A0 | 11/07已有ML/KR × Clean/G-D0 | 2022–2026历史 | 0 | Fast−F/Slow−F收益是否跨窗口持续；e→e+5是否可预测 | 支持/不支持/缺失 |
| A1 | KR Clean、KR Gaussian D0 | 2027–2031 | 10 | 持续性信号是否跨新seed、P_A/P_B迁移 | 进入B或静态替代 |
| B | A1相同两条件 | 2027–2031 | 0（共用轨迹） | A/B/C闭环是否超过F和统一动作 | 选最简单有效路径 |
| C | ML Clean、ML Gaussian D0；KR确认 | 2027–2031 | 视需复用/新增5seed | 跨数据集完整模型性能 | 正式评价 |
| 静态替代 | A0/A1存在稳定任务偏好但无时序持续性 | 至少5seed | 共用A1轨迹 | 固定task rho / C静态λ是否优于统一F | 仅作为替代，不称动态控制 |

## 2. A0：已有数据零训练持续性审计

必须读取 11/07 的原始 pulse/readout/metrics、PREDICTIVE_OOF、PREDICTION_SEED/SUMMARY、PREDICTIVE_COMPARISONS、ACTION_DECISIONS、HISTORY_PREFERENCE_SUMMARY、PANEL_STABILITY、METRIC_CORRESPONDENCE、TEMPORAL_RANK_STABILITY、GROUP_INTERACTIONS、MECHANISM_DECISIONS。先校验文件是否真实存在、非空、哈希一致；报告摘要不替代逐窗原始数据。

11/07原始pulse每5epoch一次（e=5,10,…,50），A0的相邻窗口lag-1表示相隔5epoch；不能将其称为逐epoch持续性。原预测器用P_A在e提供特征，预测e+5新pulse在V_A/V_B上的Fast−F/Slow−F收益；它不是当前pulse延迟生效的估计。C_design基线为48-way condition-cell-group onehot + epoch/50，ridge alpha=1、无惩罚截距，按dataset独立拟合，4seed early origins 5/10/15训练、heldout seed late origins 25/30/35/40/45评分，并交换V_A/V_B标签侧。

对每个dataset×objective×seed×cell×group计算：
- Fast−F与Slow−F的窗口序列；
- 相邻pulse窗口（间隔5 epoch）的符号一致率；
- 窗口收益的Spearman/Kendall排序稳定性；
- seed内lag-1相关；
- 当前窗口收益预测e+5窗口收益的误差、相关与动作regret；
- P_A与P_B方向一致率，P→V_A/V_B迁移；
- 对照 C_design、C_simple、M0、M2、M3及固定task/group/epoch身份模型。

每个seed先聚合所有窗口/格/组，再对5个seed计算均值、样本SD及配对差。scope有嵌套关系时按唯一key去重，不能累计独立样本。必须分开报告当前收益、未来收益、预测误差、动作价值和闭环性能。

A0必须回答：C_persist相对身份/阶段/简单参照是否带来独立预测增量？收益方向是否足够稳定以支持控制？如果只在单一局部范围成立，应保留局部结果但不能据此启动全局控制。

## 3. A0阶段门

- **PERSISTENCE_SUPPORTED：** 至少一个预先定义的主要范围（优先KR Gaussian、S、任务格聚合）在5seed中显示P_A→P_B前瞻方向一致，且e→e+5收益预测优于C_simple/身份参照；无严重跨seed反转。可进入A1。
- **STATIC_PREFERENCE_ONLY：** 相邻窗口持续性弱，但task0/task1偏好跨seed、P_A/P_B稳定。进入静态替代：固定task动作或单状态静态λ，不启用每epoch动态切换。
- **NO_ACTIONABLE_SIGNAL：** 持续性和静态偏好均无可迁移增量。停止新增训练，保留固定F。
- **SOURCE_INCOMPLETE：** 关键逐窗数据缺失/哈希不符。只允许恢复已有归档文件或用已保存的完整数据离线重算；不得把缺失视为阴性，也不得直接启动A1绕过来源审计。

## 4. A1：前瞻性新seed

只在 A0支持动态或静态路径时运行。KuaiRand Clean/G-D0，各5个新seed 2027–2031，每条50epoch。模型、人口、batch、Adam、Gaussian与11/07完全一致。A1不允许Test参与选择。

同一online轨迹并行维护：
- online；
- 固定F；
- Fast/Slow pulse observer；
- A全局控制状态；
- B两条任务条件状态；
- C一条F与两个task λ；
- 统一Fast/Fixed/Slow动作参照。

控制器观测不得修改loader、训练样本、loss或Adam。新增观测在每epoch结束执行；用e及以前的数据决定e+1动作。A1首先检验信号是否从历史seed迁移至新seed，而不是立即追求性能。

## 5. B：闭环控制器比较

所有方法共用A1在线轨迹。对照包括：
1. online；
2. fixed F；
3. 全程Fast、全程Slow（统一动作参照）；
4. A全局动态动作；
5. B任务条件化动态动作；
6. C-P单状态任务参数插值；
7. C-F函数空间任务logit门控；
8. 若A0判定静态偏好，附加静态任务动作/静态λ。

主终点为Validation六格Macro AUC最早严格最大值。所有动作仅由P_A训练面板的过去收益决定；P_B、V_A、V_B用于独立审计。报告每epoch动作序列、切换次数、收益信号延迟、控制计算成本、六格AUC/LL、Macro/Traffic/Pooled。

A/B/C-P和C-F都不是在线训练算法，不改变Adam参数更新轨迹；A/B/C-P改变参数历史读出，C-F只在函数空间混合online/F的任务输出。若要研究对Adam反馈，必须另立研究路线，不纳入本计划。

## 6. C：跨数据集正式评价

B中至少一个方案在新seed KR验证上优于F，且增量来自可迁移控制信号而非单一局部噪声，才进入C。将冻结的最佳方案扩展到ML Clean/G-D0；KR保留确认集。Test在所有超参、算法、选点规则冻结后一次性评价。

报告：
- 六格域任务AUC与LogLoss；
- 六格Macro AUC/LogLoss；
- Traffic AUC/LogLoss；
- Pooled AUC/LogLoss；
- 对固定F的组件增量；
- 对已发表最强基线（ML M3oE；KR HyperGate）的正式增量；
- 推理吞吐、显存/内存和额外延迟。

不把局部pulse收益相加为Macro AUC，不把内部F称作发表模型baseline。
