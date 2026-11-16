# DESIGN SPECIFICATION — TV-GRM 与无限自主研究循环修订

状态：DESIGN COMPLETE / READY FOR CODER。coder 按本文实现；不同的 reviewer 独立审查设计、实现和 coder 报告，通过后 runner 才运行。本文依据 `start.md`（GB18030）、`buchong.md`、上游计划及 NDP-A 全报告与原始数表；采用 ponytail full、stop-that-shit change。

设计输出目录：`/home/xzf/Code/phase0-3/artifacts/26-11-16_augmentation_gradient_momentum/`。
执行工作树：`/home/xzf/Code/phase0-3_tvg_rm_20261116`，detached、固定 `dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11`。旧工作区无 Git，不能把旧根目录上下文或旧本地 main 当作当前 main。

## 1. Research question / 研究问题
在自然 row-task 支持、相同任务视图和固定 EMA 下，将视图与 clean 的参数梯度之差单独累积为一阶残差状态，并只使用 clean Adam 的二阶预条件，能否产生超过 CLEAN fixed EMA 的验证 AUC 增量？增量是否超过 NDP-A 的普通凸混合、匹配实际残差更新范数的随机方向，以及不累积残差动量的简单方案？

执行问题：如何让每轮实验、报告、主上下文更新、新原生 Astra 研究和下一轮 Codex 实验在后台持续衔接，直至人工停止，同时保持 Test 封存与有限 RAM 占用？

## 2. Hypothesis being tested / 可否证假设
- H1：`g_view-g_clean` 存在 CLEAN/固定 EMA 尚未利用的排序信息。
- H2：在 clean 二阶预条件下，残差的实际方向比等残差更新范数的随机方向有用。
- H3：残差一阶历史的持续累积比仅使用当前残差有用。
- H4：该优化器路由优于已测 NDP-A 普通凸混合。NDP-A 同时改变两个 Adam moment；本比较识别整体路由差异，不能单独证明二阶状态是唯一原因。

允许全部假设不成立。局部导数、梯度余弦、LogLoss 改善、online 收益均不能代替主 AUC 结果。已测 A_MATCH 反馈失败约束持续训练解释；不能把历史单步正值外推为 TV-GRM 有效。

## 3. Variables changed / 改变变量
算法只改变 clean Adam 外的残差通道：全局 cap `gamma∈{.05,.10,.20}`；`.10` 的随机方向控制；`.10` 的无残差动量控制。视图半径固定 `.03`，不搜索 alpha、lambda、beta 或 EMA rho。

编排只纠正既有计划/manifest/state 中的轮数上限、候选停止与整项停止的混淆、Test 条件、交接上下文和实际模型身份；补齐真正后台外循环、人工停止和现有 RAM 状态生命周期。每轮有限矩阵预算与无限外循环分别记录。

## 4. Variables held constant / 固定变量
MovieLens、KuaiRand 真实自然人口，继承 80/10/10 划分和 train-only 编码；M3oE fixed1、3 域×2 task、固定融合系数 1；50 epochs、batch 40960（保留尾 batch）、FP32；seeds **2022–2026 五个且仅这五个**。不改架构、gate/expert/fusion、共享结构、采样比例或数据划分。

单 coupled Adam：lr=.01，betas=(.9,.999)，eps=1e-8，ML/KR WD=1e-4/1e-5，`foreach=False,fused=False,amsgrad=False`；固定 EMA rho=.9828205985452511/.995，每 batch 完整参数更新后一次 EMA。online/EMA 分别按验证六格等权 Macro AUC 的最早严格最大值选点。评价用无扰动输入，AUC/未经缩放的 LogLoss 分别裁定。

小域不是困难域；不围绕该边界设计验证或构造困难度代理。域人口、标签数、各格原始表现属于协议与性能描述。CLEAN/EMA、NDP-A 为内部匹配参考，不冒称正式发表模型基线。

## 5. Mathematical / algorithmic definition / 精确算法
### 5.1 自然支持、detached task views 和两套梯度

一个自然 batch 的实际行数为 B。令 `ell` 为概率 BCE：

`L_c = (1/(2B)) sum_i sum_{t=0,1} ell(p_theta(x_i)_t,y_it)`

沿用 NDP-A 已审查的完整 embedding 边界、probe 输入 Jacobian 和普通训练 gate detach。对当前同一 `theta`：

`a_it = (sigmoid(z_it)-y_it) * grad_e z_it`

`R_i = RMS(e_i)`

`delta_it = stop_gradient(.03 * R_i * a_it/(||a_it||_2+1e-12))`

仅在 `||a_it||>1e-10` 且 `R_i>1e-12` 时使用该式，否则 delta 精确置零并保留原 entry。无 pair、投影、rejection、标签平衡或额外辅助质量。probe 只为输入敏感度临时重接 gate，训练 forward 保留原 gate detach。清理 probe hooks/图；probe 不填充参数 `.grad`。

`L_v = .5 * (mean_i ell(p_theta(x_i;delta_i0)_0,y_i0) + mean_i ell(p_theta(x_i;delta_i1)_1,y_i1))`

视图 0 的 task1、视图 1 的 task0 不进入训练 loss。两项都使用全部 B 行；尾 batch 使用真实 B。训练视图为 `embedding_theta(x)+delta.detach()`，不能 detach 整个 embedding，否则丢失 embedding 参数梯度。不得反传穿过 delta 的构造，无二阶导数。

`g_c = grad_theta L_c; g_v = grad_theta L_v; r = g_v-g_c`

两套梯度在参数修改前计算、按参数名字/固定顺序对应；均为 data gradient，不含 WD。没有 `.75/.25` 混合梯度送入 TV-GRM 的 Adam；`.25` 仅是历史 NDP-A 的目标定义。

### 5.2 每参数 Adam step、残差历史和全局 cap

定义本 batch 活跃参数集合 `A={p : g_c,p is not None}`。当前 `g_c,p=None` 的参数，即使以前已有 Adam state，也跳过本次参数、WD、m/v/q 更新及 step 增量，并排除在本步 cap/random norm 外。不得把 None 转成零张量。`g_c,p` 是零张量时仍执行正常 Adam/WD。若活跃参数的 `g_v,p=None`，其数学梯度按零处理；反向只有 view 活跃的参数仍跳过并计数。

对 `p∈A`，令 `n_p` 为该参数源 Adam 本次更新后的 step。不能用全局 batch counter 替代；参数暂时缺席时 n_p 不增长，重新出现时继续原值：

`h_p = g_c,p + wd*theta_p`

`m_p <- beta1*m_p + (1-beta1)*h_p`

`v_p <- beta2*v_p + (1-beta2)*h_p^2`

`q_p <- beta1*q_p + (1-beta1)*r_p`，初始 q=0，不含 WD。

`mhat_p=m_p/(1-beta1^n_p); vhat_p=v_p/(1-beta2^n_p); qhat_p=q_p/(1-beta1^n_p)`

`D_p=sqrt(vhat_p)+eps; u_c,p=mhat_p/D_p; u_r,p=qhat_p/D_p`

对活跃参数全部坐标拼接后的单一全局 L2 范数：

`C=||u_c||_2; U=||u_r||_2; s=0 if U=0 else min(1,gamma*C/U)`

`b=s*u_r; theta_next=theta-lr*(u_c+b)`

若 C=0，b=0。cap 分母使用明确零分支，避免 epsilon 造成匹配目标不明确；Adam 的 eps 保持源值。范数标量用 detached FP64 累加，参数/m/v/q 保持 FP32。精确不变量为 `||b||≤gamma*||u_c||`。

实现优先复用源 `torch.optim.Adam` 更新 clean m/v 和 clean 参数，再在同一 batch 更新事务中加入已定义的残差增量；EMA、观察者和 RAM 提交均只能在完整更新之后。两套梯度在此事务前取得，不能在 Adam 已改 theta 后再求 g_v。残差不得进入 Adam `.grad`、m、v 或 WD；不引入第二 optimizer。若源更新的舍入路径与直接公式不同，clean 部分保留源算子顺序，尤其 gamma=0 路径。

`gamma=0` 直接走 canonical source clean 训练路径，不运行 view/probe/random，不改变训练 RNG/loader 消耗。精确比较 model/source Adam/EMA/RNG/loader 状态；新增 q 属于单独通道，不把扩展 state 字段误作 source Adam hash 不等。

### 5.3 SHUFFLE 的匹配域：预条件且限幅后的残差更新

废除“先匹配 raw r 范数，再过随机 q/预条件”的定义：非均匀 D 与历史累积会破坏实际更新范数匹配。

SHUFFLE-010 在**自己的当前 theta、clean m/v、真实 residual q** 上，先按 5.2 算出本地反事实 `b`、`R=||b||`。用独立键控随机流 `(dataset,seed,epoch,batch,parameter_name,stream)` 在 A 的全部坐标生成 Rademacher 向量 xi，维数为 N：

`b_shuffle = 0 if R=0 else R*xi/sqrt(N)`

实际更新 `theta_next=theta-lr*(u_c+b_shuffle)`。因此 `||lr*b_shuffle||=||lr*b||`，匹配的是**预条件、cap 后注入的残差参数更新**，不是 raw r、q 或总更新范数。不再对 b_shuffle 做预条件、动量累积或第二次 cap。真实 q 仅产生本地幅度参照，不把其方向应用到 SHUFFLE 轨迹。

SHUFFLE 的参照属于自身当前状态，不能偷读 TV-GRM-010 的在线梯度、theta 或幅度序列；两臂独立训练。该控制移除有结构的残差方向，保留状态相关的幅度信息；不同轨迹之间不要求逐步范数相等。固定参数顺序，随机流独立于初始化/loader；记录 norm 相对误差，FP32 回写后≤1e-5，目标零时检查精确零。

### 5.4 无残差动量控制 TV-GRM-INSTANT-010

clean Adam beta1/beta2、D、视图、gamma=.10、全局 cap 完全相同，只把 `qhat_p` 替换为本步 `r_p`，即残差 beta=0、无需 bias correction。`u_r,p=r_p/D_p`。不把 clean Adam beta1 改为 0。

这是删除残差历史的直接消融，比较整体历史通道贡献；cap 激活和实际注入范数可能随历史改变，必须同时报告。因此单凭此对比不能进一步声称“仅方向平滑”或“仅幅度平滑”是原因。本轮不加第二套范数控制或 beta 搜索。

### 5.5 后台外循环和终止语义

在现有 Codex CLI/RAM guardian 之上，用一个 stdlib 进程 driver 承载持久主会话。首次 `codex exec --json` 使用已验证父模型 `gpt-6.1-sol`/xhigh；记录 `thread.started.thread_id`，后续显式 `codex exec resume <SESSION_ID>` 续同一父会话，不用 `--last`。CLI 使用既有认证，不复制凭据。Astra 始终是该主会话内原生 `spawn_agent` 子上下文 `gpt-6-astra`/xhigh，不能改为独立 exec Astra。

driver 后台脱离前台终端生命周期，持久保存 JSONL 日志、会话 ID 和现有 AUTONOMOUS_STATE。按完成的阶段/轮边界续父会话，不额外搭建任务框架。`turn.completed` 只是一次主会话回合结束；只要状态 ACTIVE 且下一动作明确，继续 resume。`turn.failed/error`、CLI 异常退出或具体技术故障保存诊断并 PAUSED，不伪造新研究结果或静默重建父会话。不能仅凭进程 PID/参数声称外循环已运行：验收需有真实父会话事件、状态推进和已开始的合法下一动作。

每轮流程为：设计完成 → 独立 coder → 不同 reviewer → 修正阻塞项 → runner 原值 → 主会话分析/报告 → **主研究上下文实质更新** → fresh 原生 Astra → 研究计划 → 主会话完整性检查 → 下一轮设计与 Codex 实验。Astra 每轮保留原计划要求的≥3种实质不同机制探索及六件研究输出。新候选设计先于实现，不能机械重复上一矩阵。

`max_cycles=null; max_astra_cycles=null; max_experiments=null`。删除“两轮后结束”“只返修一次即结束”等任何隐藏总上限。gamma 候选数、单矩阵 50 epochs 和五 seed 是单次实验预算，不是外循环终止条件。

| 事件 | 必须动作 | 整项正常终止？ |
|---|---|---|
| 候选 NO-GO / NO_CONTINUATION_SIGNAL | 报告、更新主上下文、释放该候选无用 RAM 状态、fresh Astra | 否 |
| Astra STOP | 记录该候选否决及原因、更新上下文、fresh Astra 再探索 | 否 |
| Astra REVISE / 计划缺项 | 返回具体缺项并修订，完成后继续 | 否 |
| 实际能力/资源/运行技术阻塞 | 保存状态，PAUSED，可恢复的 next_action | 否 |
| 人工 flag/sentinel/明确停止指令 | 停止派发，完成当前原子操作，最终研究总结 | **USER_STOPPED** |

正常终止原因只有 `USER_STOPPED`。成功矩阵、发表门、无候选收益、轮数/实验数/时间耗尽均不能自动结束研究。技术暂停不是正常终止；保持 termination_reason=null，并用 pause_reason 指出具体阻塞。不降低科研门或训练预算来维持运行。

### 5.6 人工停止、sealed Test 和 RAM 生命周期

唯一 sentinel 为执行工作树 `artifacts/26-11-16_augmentation_gradient_momentum/STOP_AUTONOMOUS`，原始交付目录仅保存设计副本。flag 为同目录 `AUTONOMOUS_STATE.json.manual_stop_requested`。用户明确停止、flag=true 或 sentinel 存在任一项即生效并锁存。driver 每次 exec/resume 前、主会话每次 Astra/Codex/coder/reviewer/runner 等新派发前都重读；长期训练在现有 batch 边界检查，不等待整轮矩阵结束才响应。

停止时完成已开始 batch 的两套梯度、完整参数更新、EMA 和 RAM 提交这一原子操作；不能保存半步状态。已有子进程收到停止请求，在安全边界返回。停止新的训练/研究派发，先收集现有日志并生成包含已完成和未完成矩阵的最终总结。若父会话恰在回合边界，driver 可 resume 同一父会话一次进入明确的 FINALIZE 路径，完成日志汇总/最终冻结；该路径禁止新 Astra/Codex 研究派发，不能因 stop 检查而遗失最终总结。状态记录 USER_STOPPED。确有有效候选时，最后冻结候选/比较对象/源码/数据/选择规则/selected 模型 hash 后，才允许单独最终 Test 阶段；部分 seed 不冒充完整五 seed。

`test_allowed = (termination_reason==USER_STOPPED) and final_freeze.complete and final_candidate_eligible`。

**任何 ACTIVE 自主轮、候选 PASS、gamma 冻结、Astra STOP 或技术 PAUSED 均不能开启 Test。** 不在自主循环调用 Test loader/inference/metric；数据身份直接引用已核验 split/hash 收据。人工停止后的最终 Test 结果不能再次触发 HPO 或恢复自主探索。没有有效最终候选则 Test=NOT_TRIGGERED。

复用既有 RAM guardian/admission，不写磁盘 model、optimizer、EMA、checkpoint、张量快照或序列化权重。持久 state 仅保存路径/hash、计数和 guardian 引用。只保留当前原子恢复状态、当前轨迹验证选点状态，以及最多一个按验证规则选定的 incumbent/champion 配置所需五 seed selected 状态；完成/否决的其他候选立即释放。新 champion 替代后释放旧 champion。禁止每个否决周期永久积累模型。当前轨迹/当前矩阵所需选点状态作为暂存，报告/候选决策后即回收；RAM 准入仍<90%。

精确数值恢复依赖存活 guardian；失去 RAM 状态时标 PAUSED，按冻结配置重跑受影响轨迹，不能从 hash 重建权重或用 final 代替 selected。保持现有逐 GPU 显存和功耗均<85%、RAM<90%准入，按单卡独立并发，不减 batch/epoch/seed。

## 6. Exact affected modules / 文件与来源
本 DESIGN 阶段只新增本文。以下是 coder 的精确落点；不覆盖其他 agent 并发编辑。

令 `P=research_plans/26-11-16_augmentation_gradient_momentum/`，在执行工作树修订：

- `P/00_EXECUTION_RUNBOOK.md`、`11_AUTONOMOUS_ORCHESTRATION.md`、`12_ASTRA_RESEARCH_PROMPT.md`、`13_AGENT_HANDOFF_PROTOCOL.md`：无限外循环、后台主会话、stop、模型门、每轮上下文与证据交接。
- `P/01_RESEARCH_CONTEXT.md`、`02_HISTORICAL_EVIDENCE_LEDGER.md`、`03_RESEARCH_QUESTIONS.md`：整合 NDP-A 全报告/原始证据、加入即时残差竞争解释。
- `P/04_FIRST_ROUND_EXPERIMENT_DESIGN.md`、`05_EXPERIMENT_MATRIX.md`、`06_GRADIENT_AND_MOMENTUM_DIAGNOSTICS.md`、`07_HYPERPARAMETER_AND_STATISTICS.md`、`08_CODEX_IMPLEMENTATION_SPEC.md`：本文完整数学定义、50条矩阵、精确 norm 域、per-parameter step、最小检查、人工停止前 Test 封存。
- `P/09_SCHEDULER_AND_RESOURCES.md`：纠正每 epoch/整条更新计数与50条预算，RAM回收。
- `P/10_REPORT_TEMPLATE.md`、`experiment_manifest.json`：直接 NDP-A/SHUFFLE/INSTANT 配对结果、无限策略、Test条件、完整预算。
- `artifacts/26-11-16_augmentation_gradient_momentum/AUTONOMOUS_STATE.json`：扩展既有状态，不重置已有计数/收据。
- 主上下文 `EXP_AI_CONTEXT.md` 与 `experiments/Experimental_{Global_Summary,Stable_Findings,Open_Questions,Experiment_Index,Evidence_Ledger}.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md`：按实际证据实质回填，不追改旧报告。

新增最小独立 package `experiments/task_view_gradient_residual_momentum_20261116/`：`core.py`（梯度/残差更新）、`protocol.py`（身份/config/复用）、`runner.py`（现有 worker/guardian/admission 适配）、`tables.py`（配对输出）、`test_core.py`（紧凑语义检查）、`autonomous.py`（stdlib 后台 driver）；必要的 `__init__.py`。不要为了文件对称拆更多层。若现有纯导表 helper 足够，直接复用、合并 tables 责任。

只读复用：`experiments/auxiliary_objective_attribution_20261115/core.py::{forward,sensitivity}`、`stage2/core.py::make_views` 及纯 loss helper；`stage2/protocol.py` 用于核对身份，不调用其旧 Stage2 prepare/固定 job 队列；`experiments/conditional_information_momentum_20261102_v3/` 的源模型/data/evaluate/EMA/RNG/guardian。不得把 sibling 绝对路径当运行时 Python import，不改 vendor/历史源码。

新日志/收据/报告分别在本任务 artifacts/research_reports 对应目录；无提交、push 或修改历史报告的授权。本轮上下文交接以实际 reviewed 工作树/文件 hash 和报告身份为准，不能把没有提交的内容说成已进入 main。

## 7. Implementation requirements / 具体实施约束
### 7.1 能力门与模型政策冲突

用户提供的宿主回执已确认：父 `gpt-6.1-sol`/xhigh；子 `gpt-6-astra`/xhigh，独立 `source.subagent.thread_spawn`、depth1、无 fork。原文 `6-Astra-exhigh` 是不可用别名；原生实际 ID 为 `gpt-6-astra`。子 agent ID `01a1dfbf-ee0f-7501-ba26-d2f024c769e0` 已读取固定来源，核对160条 ENDPOINTS、3960条 paired 数据并复算20个相关 seed 差。

能力证据由父保存于执行工作树 `artifacts/26-11-16_augmentation_gradient_momentum/PREFLIGHT_RECEIPT.json`，用户报告 verdict=PASS。coder 引用这份原生 metadata/固定来源读取收据，不重复 capability 测试、不改用父模型模拟 Astra。后续真实派发若出现具体能力故障，才记录 AGENT_DISPATCH_BLOCKED/PAUSED，交付已修正计划与可恢复 state，不写模拟循环完成。

**显式政策冲突：** start.md 第11条要求除机械 RUN 外子 agent 与主 agent 同模型；runbook/11 指定 Astra 独立模型。当前用户明确坚持原生 Astra，采用此项显式例外：Astra=`gpt-6-astra`/xhigh；coder 与独立 reviewer 采用实际父 `gpt-6.1-sol`/xhigh；两者不同 agent；runner 只记录原值。保留冲突和当前裁定，不谎称原两条同时满足。

### 7.2 状态/manifest 最小字段

保留既有字段，加入 `max_cycles:null,max_astra_cycles:null,max_experiments:null,manual_stop_requested:false,manual_stop_sentinel,normal_termination:[USER_STOPPED],test_policy`。状态明确区分 `status=ACTIVE/PAUSED/USER_STOPPED`、`current_cycle`、`current_stage`、`current_candidate`、`candidate_decision`、`pause_reason`、`termination_reason`、`next_action`、`parent_session_id`、`astra_dispatch_status`、实际 model/effort 与 receipt 路径、report/raw/context 身份和 guardian/champion 引用。没有实测的字段保持 null，不填计划值冒充实际值。

`planned_trajectories=50` 只属于首轮；后续设计逐轮声明自己的完整五 seed 矩阵。stdout/session 回合完成不覆盖候选科学决策或全项终止状态。manual_stop 标志必须在状态更新时保留，不能被旧内存副本清回 false。

### 7.3 NDP-A substantive context / 精确证据路径

证据根 `E=/home/xzf/Code/phase0-3_auxiliary_20261115`；下列相对路径均拼接 E，可只读访问。首次 Astra 研究 handoff 前，向当前主 EXP_AI_CONTEXT 和证据/结论入口实际加入以下结果、权重语义、阶段矛盾和直接来源，不能仅复制“已整合”字样。

| 文件 | 必须交给 Astra 的内容 |
|---|---|
| `research_reports/auxiliary_objective_attribution_20261115/REPORT.md` | 全报告，含 Stage1 零扰动归因、Stage2、矛盾/竞争解释/停止理由 |
| 同目录 `STAGE2_DESIGN_SPECIFICATION.md`、`STAGE2_FINAL_REVIEW.md`、`STAGE2_DECISION.json`、`STAGE2_RUNNER_REPORT.json`、`SOURCE_MAP.md` | 冻结定义、审查、决策、源码责任与成本；审查不代替原数值 |
| `artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/run_manifest.json`、`source_receipt.json`、`movielens_data_manifest.json`、`kuairand_data_manifest.json` | source/config/data/split/选择身份 |
| 同 attempt `zero_counterpart_receipt.json`、`reused_clean_index.json` | CLEAN/zero精确状态等价和十条 CLEAN 索引 |
| 同 attempt `tables/ENDPOINTS.csv`、`PAIRED_SEEDS.csv`、`PAIRED_SUMMARY.csv`、`VALIDATION_CURVES.csv`、`WEIGHTS.csv`、`COSTS.csv` | 绝对端点、逐seed配对、六格/阶段/权重/成本原值 |
| 同 attempt `runs/*/result.json`、`runs/*/attempt*/epoch_*.json`、`epoch_commit_ledger.jsonl`、`launch_logs/` | run/epoch 原始记录及其指向的compact日志；按需追索，不复制巨型逐entry支持流 |
| `research_reports/auxiliary_objective_attribution_20261115/tables/STAGE2_BATCH_AUDIT.csv`、`STAGE2_FIRST_BATCH_DIAGNOSTICS.csv` | 实测全自然支持、视图与权重审计 |

已读原值：NDP-A fixed-EMA−CLEAN fixed-EMA selected validation Macro AUC，ML `-.00010125623089003443`（1/5正，CI跨零），KR `-.005775319390310641`（0/5正，CI跨零）；ML online 增量 `+.000359628` 不能作为 fixed-EMA 增量。KR d2 两task改善与 d0 两task明显退化并存，LL 与 AUC可反向。报告证据标签：旧零扰动重现主要损失 STRONG；KR退化 MEDIUM；ML新增排序信号 WEAK。NDP-A 自然支持/权重修正已执行，不能把失败简单归咎于旧辅助质量缺陷。

NDP-A 目标 `.75*L_c+.25*L_v`，全2B支持、每entry质量1/(2B)，alpha=.03。源 hash `831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`；selection hash `c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89`。CLEAN 与 NDP-A 复用只计历史同一五 seed，不算新独立确认。不重跑既完成 NDP-A。

当前主上下文以固定 `dda38ec8:AGENT.md`、`dda38ec8:EXP_AI_CONTEXT.md`、`dda38ec8:research_context/experiments/Experimental_Open_Questions.md` 为起点，后者真实路径含 `experiments/`。父提供的 `artifacts/.../source_context/{AGENT.md,EXP_AI_CONTEXT.md,Experimental_Open_Questions.md}` 是这些来源的快照映射；不要使用旧 cwd 根文件替代。各轮 handoff 显式传 NDP-A 全报告与上述 raw evidence，以及该轮完整报告/原始端点/诊断/成本和更新后的主上下文，fresh Astra 不继承父 transcript。

### 7.4 数据与执行身份

优先用父已验证的只读缓存 `/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache`。原始 CSV：`/home_new/xzf/datasets/ml-1m/ml-1m.csv`、`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。缓存路径不是历史 manifest 中旧位置仍存在的证明，引用 PREFLIGHT 数据哈希收据。

项目 Python 固定 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python` 或规定 conda run。source manifest 绑定实际隔离工作树 HEAD、实际 import 文件 hash、设计/config/数据/选择/审查 hash；detached provenance 如实记录，不伪报 main 已更新。父负责已授权的工作树准备，designer 不操作 Git。

## 8. Ablations required / 冻结首轮矩阵
| 臂 | gamma | 残差方向/历史 | ML新训练 | KR新训练 |
|---|---:|---|---:|---:|
| TV-GRM-005 | .05 | 真实残差，beta=.9 | 5 | 5 |
| TV-GRM-010 | .10 | 真实残差，beta=.9 | 5 | 5 |
| TV-GRM-020 | .20 | 真实残差，beta=.9 | 5 | 5 |
| TV-GRM-SHUFFLE-010 | .10 | 5.3实际残差更新 norm 匹配 | 5 | 5 |
| TV-GRM-INSTANT-010 | .10 | 当前残差，无q历史 | 5 | 5 |
| CLEAN / NDP-A | 历史 | 精确身份复用，各10条历史轨迹 | 0 | 0 |

新增 **50条、2500epochs**，每臂/数据均完整 seeds2022–2026。三个 gamma 全跑，不先用1–4 seed筛选。`.10` 是预先固定机制锚点；不能用选择后的 `.05/.20` 与 `.10` 控制比较后宣称已排除对应 gamma 的幅度/动量解释。其他 gamma 若成为候选且需要对应机制证据，下一轮另立设计，不隐加本轮全交叉矩阵。

`steps_per_epoch=ceil(train_rows/40960)`：ML800167行→20，KR2006132行→49。每条50epoch共1000/2450步，**不是每epoch1000/2450**。新增25条ML、25条KR，共 `25*1000+25*2450=86250` 次完整参数更新及同次数EMA。CLEAN复用为0新增；若精确复用失败，记录具体差异并修复路径，不静默追加完整参考训练。

必须报告每个候选 vs CLEAN fixed EMA、`.10` vs SHUFFLE、`.10` vs INSTANT，及每个候选 **直接逐seed vs NDP-A fixed EMA/online对应读出**；同时报告候选+EMA vs CLEAN online总收益。旧NDP-A只称历史控制，不冒称本轮随机分配的纯二阶状态因果消融。

## 9. Expected logging requirements / 日志与统计
每epoch保存online/EMA验证六格AUC/LL、Macro、Traffic、Pooled、逐domain/task与worst cell；50epoch曲线、E10/E30/E50、selected epoch/hash；每seed配对原值、差、mean、sample SD、`95% CI=mean±2.776445*SD/sqrt(5)` 和正向seed数。seed是统计单位，不把epoch/cell当额外重复。

每batch只留紧凑标量：B、row-order hash、clean/view loss及实际分母、自然支持计数、活跃/None参数数、完整更新/EMA数、cap系数/激活、`||u_c||,||u_r||,||b||,||lr*b||`；SHUFFLE加本地目标norm/实际norm/误差与独立流键。E1/10/25/40/50固定诊断batch记录g_c/g_v/r/q/m/v的范数、cosine、per-parameter step范围和精确源参数组汇总。无全梯度逐batch落盘。

成本记录实际同步计时、wall/吞吐、GPU峰值allocated/reserved、RAM占比及guardian保留/释放数量。NDP-A历史运行成本仅作估算参照，TV-GRM耗时由smoke测量，不承诺50条在固定夜间窗口内结束。

编排JSONL保留父thread/session事件、原生child实际metadata、cycle/stage、dispatch前stop检查、完整性/独立review结果、report/context/raw路径和hash、next_action、暂停与人工停止理由。成功外循环需真实“上一轮报告→上下文更新→fresh Astra→新计划→下一实验派发”事件链；不能仅靠driver alive或ACTIVE字段替代。

## 10. Acceptance criteria for reviewer / 审查验收
1. **文档/状态一致。** 00–13、manifest、state无轮数/实验数总上限；NO-GO与Astra STOP都不终止全项；只有USER_STOPPED正常终止；自主Test始终false；首轮50条/86250更新一致。读取回执PASS、模型例外与实际ID明确。
2. **实现符合数学。** 同theta两套data gradient；delta detach且embedding参数仍可训练；target-task专属loss/自然支持；clean Adam/WD一次；per-parameter n_p与None语义；q无WD；全局cap；SHUFFLE在实际残差更新域匹配；INSTANT只删残差历史；一次完整参数事务后一次EMA。
3. **最小必要工程证据。** 复用仍有效的已审查Jacobian/finite-difference证据，不重建测试工程。新CPU检查集中四组：(a)自然loss/梯度、target选择、detached view；(b)两参数活跃/缺席/重入的Adam/q计数、WD、INSTANT递推；(c)cap/SHUFFLE正值与零值norm、独立随机流；(d)gamma0多步源状态精确相等及一次RAM保存恢复与不间断轨迹一致。覆盖直接改变本次结论的分支即可。
4. **真实full-batch工程smoke。** 每数据集一个真实自然首batch harness，固定同初始化，检查source clean、gamma0、TV-.10、SHUFFLE-.10、INSTANT-.10各一次完整更新；合计两套harness/10个单步分支，不是正式性能训练。gamma0与source状态hash相等，其他臂finite、更新/EMA计数与norm定义正确，记录实际资源。无需额外gamma网格GPU smoke或新有限差分矩阵。
5. **独立审查门。** coder只实现本设计/做基本正确性检查；另一reviewer独立读本文、源码、CODER_IMPLEMENTATION_REPORT和相应证据。PASS或修完全部blocking issues的PASS WITH FIXES后才派正式runner；runner不分析科学结论。
6. **后台与人工停止真实可用。** 对最小driver解析/续会话/stop边界用短事件fixture检查一次即可，不派假科研循环。正式启动后记录真实thread.started、持久session_id、合法阶段推进；人工flag/sentinel检查在driver和每次子派发前生效。不能为了证明stop功能停止正式实验，也不能把第一矩阵队列当无限外循环。
7. **科学判别门。** 完成五seed再按数据集分别裁定。主指标是selected验证fixed-EMA Macro AUC相对CLEAN的均值；两数据所有gamma均≤0则TV-GRM本版本NO-GO，报告后fresh Astra。存在正均值仅保留候选/机制线索；`.10`未胜SHUFFLE或INSTANT时相应机制主张不支持，不把控制被击败当默认事实。CI跨零、非全seed正、某seed/cell负值不另设否决门；LL单独裁定。
8. **选择规则。** 若需保留首轮单一champion，按两数据集五seed Macro AUC增量均值的等权平均，在三种TV-GRM gamma中选严格最大，平局取较小gamma；如两数据均无正增量则不升级。完整披露每个gamma/数据表现；这个验证排序只确定RAM保留，不是论文合格门，也不掩盖单数据退化。后续候选沿用同一验证标量与精确协议后才替换champion。
9. **最终Test独立门。** 只在USER_STOPPED+最终freeze+有效五seed候选后运行。沿用与最强适用正式发表模型的配对平均Macro AUC≥.003、各cell均值≥-.001；LL均值<0单独标记改善，不要求所有seed正或CI排零。冻结前核实比较对象/选点状态存在；缺权重则具体暂停处理，禁止换selected身份。Test阶段结束仅形成最终报告，不再反馈自主选参。
10. **最小范围与资源。** 无源/vendor/历史报告改写、无磁盘模型、无新依赖/任务框架/泛化hardening；每轮无用RAM状态实际回收。若原生能力后来真实阻塞，结果为修正计划+PAUSED可恢复state，而不是模拟后台探索。

## 11. Conditions that would falsify the hypothesis / 否证条件
- H1：完整五seed下三个gamma在两数据集均不提高CLEAN fixed-EMA selected验证Macro AUC；或online小收益在fixed EMA主比较中消失。该结果否决本次TV-GRM版本，接入fresh Astra，不结束全项研究。
- H2：`.10` 的真实方向未超过预条件/cap后等残差更新norm的SHUFFLE，尤其SHUFFLE更好；不能宣称残差方向提供独立有效信息。
- H3：`.10` 的INSTANT等于或优于动量臂，或动量臂的阶段收益不能超过各自selected参考；不能宣称残差历史是必要增益来源。若只见norm差异，保留幅度解释，不补写方向结论。
- H4：TV-GRM未胜历史匹配NDP-A，或其相对CLEAN的结果仍重现NDP-A的格间搬移与总体退化；不能宣称新路由解决了该输入视图路线的问题。
- 任一假设只有LL/代理诊断正值、单数据集/阶段正值或少于五seed时，证据限于该现象。报告重复现象、矛盾、竞争解释与STRONG/MEDIUM/WEAK，不能升级为跨数据集通用收益。
- 实现错误、工程/能力故障、未完成矩阵属于具体阻塞，不是科学阴性；修复后按冻结方案续行。质量门不随无限循环放宽。

本设计阶段文件变更：仅新增 `artifacts/26-11-16_augmentation_gradient_momentum/DESIGN_SPECIFICATION.md`。算法实现、训练、后台启动与提交/push由本文之后的授权执行阶段处理。
