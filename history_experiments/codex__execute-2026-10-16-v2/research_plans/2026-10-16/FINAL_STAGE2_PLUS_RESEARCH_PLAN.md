# My-Research-Project：阶段 2+ 最终科研探索与实验方案规划

**制定日期**：2026-10-16  
**基准分支**：codex/execute-2026-10-15  
**基准提交**：de80055be56682c47fc3cd4aedd1c722c2bdb5d6  
**依据**：agent_promt.md、2026-10-15 后续实验规划、阶段 0/1 审计、S0-S8、S6-R2、research_R2/AO04、最新 B1 reconstruction、C2.1 numeric oracle 与实验注册信息。

> **状态：最终后续科研规划，不等同于运行授权。**
> 所有正式实验仍必须遵守 main → designer → coder → reviewer → runner → main；本规划不授权绕过 F0/科学协议，也不授权直接修改代码或启动 GPU。

## 1. 核心研究判断

阶段 0/1 之后，不再把 gradient conflict、momentum、sample weighting 当作一个混合问题。当前必须拆成四层：

1. **Objective**：优化什么总体目标，由 alpha_d 决定。
2. **Estimator / sampling**：怎样从有限 batch 估计该目标，由 domain sampling q_d 与 importance weighting 决定。
3. **Optimizer state**：Adam 的 m、v 如何积累 estimator 输出。
4. **Selection**：最终从训练轨迹中选择哪个 checkpoint。

统一抽象：

objective alpha + sampling q → stochastic gradient estimator → Adam state → parameter trajectory → validation selection → selected model → test/generalization。

核心研究问题：

> 在 MDMT 中，objective、domain sampling estimator、adaptive optimizer state 与 checkpoint selection 是否存在传统 MTL 工作没有充分区分的 interaction？这种 interaction 是否能够解释 Equal/Sample、S5 history effect、S6-R2 validation/test reversal 与 AO04 fusion boundary？

这是待证伪假设，不是既定结论。

## 2. 已关闭路线

默认 CLOSED：

- “小域天然更困难”；
- domain size 本身就是困难度；
- per-domain momentum 普遍有效；
- generic EMA / curriculum 叠加必然改善；
- 仅 gradient norm correction 可以解释 S3/S4；
- 仅 applied-step norm correction 可以解释 S3/S4；
- negative cosine 是充分性能因果指标；
- Sample 下存在普遍稳定的 DH/target rescue；
- AO04 fine-grained asymmetric fusion 当前已有可用收益；
- 仅增加 routing/free parameters 就是创新；
- 动态 task/domain weighting 本身就是创新。

PCGrad/CAGrad 已覆盖 gradient manipulation/conflict；AdaTask 已直接研究 task-wise accumulated gradients 与 Adam 等 adaptive optimizer 的 dominance；FAMO 已代表动态 task weighting。因此未来方法必须明确超越这些概念。

## 3. 新主线排序

| Track | 科学问题 | 状态 | 优先级 |
|---|---|---|---|
| A | sampling estimator × adaptive optimizer | OPEN | 1 |
| B | checkpoint selection × training trajectory | OPEN | 2 |
| C | Equal-specific persistent optimizer state | OPEN but narrow | 3 |
| D | C/A_arch parameter partition | BLOCKED | 4 |
| E | AO04 asymmetric fusion | LOW PRIORITY | 5 |
| F | 新算法 | NOT ALLOWED YET | 最后 |

## 4. Stage 2A：文献与理论边界审查（0 training）

必须系统回答：

### Q1
现有 MTL/MDMT 是否明确区分 objective weighting 与 sampling distribution？

### Q2
是否已有工作研究 same expected objective + different minibatch estimator 对 Adam trajectory 的影响？

### Q3
是否已有 MDMT 工作明确研究 domain sampling → gradient variance → adaptive optimizer？

### Q4
checkpoint selection 是否被作为 training policy 的一部分研究？

每篇直接相关论文必须记录：

- objective；
- sampling distribution；
- weighting；
- optimizer；
- optimizer state；
- gradient aggregation；
- architecture；
- dataset/domain definition；
- checkpoint rule；
- primary endpoint；
- negative-transfer definition；
- limitations；
- failure cases；
- code availability。

必须主动寻找与候选机制同构的方法，不能只找名字相似的方法。

至少覆盖 M3oE、AdaTask、PCGrad、CAGrad、FAMO、Adam/AdamW、importance sampling/variance reduction、MDMT adaptive weighting/fusion、SAF，以及 checkpoint selection/generalization reversal 相关工作。

输出必须是 **novelty matrix**，不是论文摘要合集。

## 5. Stage 2B：固定参数数值 Oracle（0 training）

核心定义：

目标权重 alpha 满足 sum(alpha)=1；从 domain distribution q 采样，使用 weight = alpha/q。

基础条件：

| Objective alpha | Sampling q | Importance weight |
|---|---|---|
| Equal = 1/3 | Natural = p | 1/(3p_d) |
| Equal = 1/3 | Balanced = 1/3 | 1 |
| Sample = p | Natural = p | 1 |
| Sample = p | Balanced = 1/3 | 3p_d |

固定 model parameters、exact batch rows、labels、loss、optimizer state，做 Monte Carlo。

必须测：

1. per-domain/per-task loss；
2. gradient mean；
3. per-coordinate covariance；
4. covariance trace；
5. gradient norm；
6. effective sample size；
7. rare/zero-domain batch probability；
8. domain exposure；
9. importance-weight distribution；
10. one-step SGD update；
11. one-step Adam update；
12. m/v change；
13. bias correction；
14. weight decay；
15. clipping（如实际训练存在）。

额外比较 fixed-p Equal 与 batch-cell-normalized Equal。不得默认二者同一期望。

**硬 STOP：** estimator mean 不匹配、数值等价条件不成立、差异只来自实现 bug、或 variance/state difference 不稳定。

## 6. Stage 2C：Adam state propagation Oracle（0 training）

验证：

same expected gradient 不必然意味着 same Adam state。

在完全相同 theta 与 paired gradient streams 下比较：

- Natural estimator；
- Balanced + importance estimator；
- SGD；
- Adam；
- 必要时 AdamW。

追踪 m、v、bias-corrected moments、applied delta、parameter change。

关键判断：

如果 same alpha、different q 只在 Adam 下出现稳定 state divergence，而 SGD 基本没有，才支持 estimator × adaptive optimizer interaction。

## 7. Stage 2D：same objective / different estimator 训练验证

仅当 2A-2C 均通过后申请。

第一轮只做一个数据集，优先 KuaiRand-natural / M3OE。

四臂：

| Arm | alpha | q | Weight |
|---|---|---|---|
| SN | Sample p | Natural p | 1 |
| SB | Sample p | Balanced | 3p_d |
| EN | Equal | Natural p | 1/(3p_d) |
| EB | Equal | Balanced | 1 |

最多 4 arms × 4 seeds = 16 runs。

Primary contrasts 只有：

- SB − SN；
- EB − EN。

Equal vs Sample 只作为 interaction analysis，不能称为 sampling-only effect。

严格固定：

- model；
- initialization；
- optimizer；
- batch size；
- effective main steps；
- compute budget；
- validation/test；
- checkpoint rule；
- capacity；
- data preprocessing。

完整记录 domain exposure。

若同目标实际 selected-performance gap 小于预注册阈值且 state/gradient 差异不稳定，则 Track A STOP。

## 8. Stage 3：Selection × Trajectory

B1 已经使该方向成为独立研究线。

必须区分：

### Same-epoch effect

Delta_same(e) = M_DH(e) − M_F(e)

### Selected effect

Delta_selected = M_DH(e_DH) − M_F(e_F)

### Timing contribution

仅在数学定义成立时计算 selected checkpoint timing contribution。

必须绘制：

1. validation macro AUC trajectory；
2. validation target AUC；
3. validation non-target AUC；
4. test same-epoch trajectory；
5. selected epoch；
6. selected vs common epoch；
7. per-cell ranking；
8. validation-test rank agreement。

核心问题：

> DH 的 apparent improvement 是 trajectory effect，还是 selection functional effect？

新的方法实验不得用 test 选择 checkpoint。

## 9. Stage 3B：S6-R2 reversal forensic audit

如果历史 checkpoint/prediction artifacts 可以恢复，先做 0 training。

要求：

- same-row paired predictions；
- per-cell AUC；
- per-cell LogLoss；
- macro；
- sample-weighted；
- target/non-target；
- same epoch；
- selected epoch；
- exact selection；
- paired row bootstrap；
- d0 long_view 低 positive count 的不确定性分析。

检验：

validation gain 是否真的不转化为 test gain。

同时检查 validation ranking 与 test ranking 的一致性。

如果 prediction-level 证据无法恢复，状态必须为 INCOMPLETE，而不是 NO。

## 10. Stage 4：Equal-specific persistent state

只有 S5 historical artifacts 恢复并通过 A0/A1 后进入。

四臂：

| Arm | C current | A_arch current |
|---|---|---|
| CF_AF | FULL | FULL |
| CT_AF | TARGET | FULL |
| CF_AT | FULL | TARGET |
| CT_AT | TARGET | TARGET |

固定：

- KuaiRand natural；
- M3OE；
- Equal；
- d1 target；
- P = FULL；
- 50 epochs；
- seeds 2022–2025；
- original main/arch loop。

最多 16 runs。

目标是判断 S5 effect 主要由 shared C 还是 architecture A_arch 承载。

若 A_arch 单独已经解释主要 effect，则不得继续把现象写成 shared optimizer history。

## 11. Stage 4B：m/v/current causal probe

禁止直接拼接异轨迹状态长期训练。

合法流程：

1. 沿同一 theta trajectory 使用 shadow optimizer；
2. 同时记录不同 gradient source 下的 m/v；
3. 在同一 theta 做 one-step counterfactual；
4. 分别测试 m only、v only、current gradient、full m+v；
5. 必要时做合法 multi-step replay。

不能把不属于真实可达 trajectory 的 (m_F, v_T, theta_DH) 直接当作正常 Adam 状态。

若 m/v effect 高度依赖 trajectory、block/seed 方向不稳定，或 one-step effect 与 selected AUC 无对应，则 STOP。

## 12. Stage 5：AO04 fusion diagnostic

暂不继续 B3/B4/B5。

对 B1：

R_dk = b_D E_d + (1-b_D)[rho_dk E_j + (1-rho_dk)E_r]

计算局部 sensitivity：

dF_dk / d rho_dk = a_D(1-b_D) rho_dk(1-rho_dk)(E_j-E_r)

逐 epoch/seed/cell 检查：

- rho saturation；
- a_D(1-b_D)；
- expert-output difference；
- prediction/logit sensitivity；
- gate entropy；
- branch collapse；
- validation-test ranking agreement。

若 rho 大幅变化但 prediction sensitivity 很低，优先研究 identifiability，不增加自由度。

若 sensitivity 高但 AUC 无收益，转向 generalization/overfitting/shift，而不是继续扩展 fusion。

## 13. Stage 6：条件性边界验证

只有此前至少一个机制被可靠确认后才启动。

顺序：

### 6.1 Domain identity
固定 natural ratio，改变 target d0/d1/d2。记录 sample count、label prior、gradient magnitude、exposure、reachability。不得把 d2 小等同于困难。

### 6.2 Ratio / exposure
分开 total N、unique samples、exposure、q、alpha。重复曝光不能称为新增独立数据。

### 6.3 Task relation
MovieLens click/like 为嵌套阈值标签；KuaiRand 两任务定义不同。先做 label agreement、conditional dependence、positive-rate diagnostics，再讨论任务关系因果。

### 6.4 Backbone
必要时 M3OE → MMoE → PLE。必须 capacity matched，并重新定义 reachability。

Go/no-go：至少两个数据/结构在预注册 selection protocol 下重现指定机制方向；若只在一个 setting 有效，则写成 narrow-boundary mechanism。

## 14. Stage 7 / E1：方法开发

只有全部满足才允许：

1. 标准 Sample 强基线；
2. mechanism ≥4 seeds 可重复；
3. 独立验证 setting；
4. 非 selection artifact；
5. 新 holdout / external data；
6. 与 AdaTask、PCGrad、CAGrad、FAMO、相关 MDMT 方法明确区分；
7. mechanism 可写成 causal hypothesis；
8. simple intervention 已验证；
9. 新方法只解决已确认 bottleneck；
10. 不重新打开 CLOSED 路线。

### E1-A：state-oriented
若 Track C 成立，研究 objective-contingent shared optimizer state。不是再发明一个 generic momentum；必须与 AdaTask 明确区分。

### E1-B：estimator-aware
若 Track A 成立，目标是：

固定 objective alpha，同时控制 estimator variance 与 optimizer-state distortion。

不是简单 balanced sampling，也不是简单 importance weighting。

### E1-C：reachability
只有证明名义 private component 存在不期望的跨域梯度可达且与稳定损伤相关，才设计最小结构约束。

## 15. 评价协议

Primary endpoint：

validation-selected six-cell macro AUC。

Secondary：

- per-domain AUC；
- per-task AUC；
- per-cell AUC；
- LogLoss；
- sample-weighted AUC；
- worst-domain；
- worst-cell；
- target/non-target；
- same epoch；
- selected epoch；
- final epoch。

默认 seeds：

2022, 2023, 2024, 2025。

历史 exploratory gate：

- practical gain +0.003；
- strong gain +0.005；
- 至少 3/4 seeds 同方向。

这些不是 p-value，也不是 deployment threshold。

所有 guard 必须在训练前冻结。

## 16. Test lock

已经被多次读取的 test 只能用于历史 forensic analysis。

任何新方法确认必须使用新的 temporal holdout 或完全未用于开发的 external dataset。

KuaiRand 的 train→validation 与 validation→test 分布差异必须分开分析，不能用粗 domain ratio 单独解释 validation/test reversal。

## 17. 统一 run schema

每个正式 run 至少保存：

study_id, stage, dataset, split_hash, setting, arm, seed, objective_alpha, domain_sampling_q, effective_importance_weight, target_domain, model_code_sha, optim_code_sha, config_sha, init_sha, batch_sha, epoch, selected_epoch, step_main, step_arch, state_complete, checkpoint_sha, val_cell_auc, val_cell_ll, test_cell_auc, test_cell_ll, source_grade, reviewer_verdict。

同时保存 stdout、stderr、config、seed/RNG、optimizer state、model checkpoint、per-domain/task metrics、domain exposure、batch fingerprints、state hashes。

## 18. Hard Go/No-Go

| 阶段 | STOP 条件 |
|---|---|
| A0 | split/init/state/batch/evaluator 不一致 |
| 2B | estimator expectation 不等价 |
| 2C | Adam divergence 不可重复 |
| 2D | same-objective different-q 无稳定 selected effect |
| 3 | selection effect 消失或无法从 prediction level 复核 |
| 4 | C/A decomposition 无法解释 |
| 5 | fusion sensitivity 低或无独立收益 |
| 6 | 机制只在单一 setting 且无边界证据 |
| E1 | 相对强 Sample baseline 无稳定改善 |

NO、INCOMPLETE、NOT_RUN 必须严格区分。

## 19. 最终执行顺序

A0 → A1/A2 → Stage 2A → Stage 2B → Stage 2C → Stage 2D → Stage 3 → Stage 4 → Stage 5/6 → E1

这是条件链，不是强制跑完全套。

任何阶段出现 STOP：

- 停止对应 track；
- 不修改 hypothesis 救援；
- 不为了论文故事追加实验；
- 阴性结果作为正式科学证据；
- INCOMPLETE 不写成 NO；
- NOT_RUN 不写成失败。

## 20. 当前第一批工作

在任何新训练前，完成三个 0-run work packages：

### WP1：证据恢复
逐项审计 S4/S5/S6/S6-R2/S8/research_R2 的 config、manifest、metrics、prediction、checkpoint、optimizer state、parameter manifest、selection 文件。

### WP2：历史结果法证
完成 R2 AO04 prediction-level audit、S6-R2 exact selected vs grid audit、paired bootstrap、selection timing decomposition。

### WP3：文献/理论
完成 objective vs sampling taxonomy、adaptive optimizer state taxonomy、checkpoint-selection taxonomy、novelty matrix。

WP1-WP3 未完成前，不申请新训练。

## 21. 当前正式研究主张的边界

目前不能写：

- “DH 在 Sample 下有效”；
- “Adam history 导致负迁移”；
- “sampling variance 导致性能变化”；
- “validation/test reversal 是 distribution shift”；
- “AO04 asymmetric fusion 失败证明不可辨识”。

目前只能写：

- 历史 S6-R2 存在 selected validation/test reversal 线索，但 reconstruction 未复制为稳定同强度现象；
- C2.1 已提供 same-objective/different-q 的数值层 variance 机制线索，但尚未证明 downstream performance effect；
- B1 显示 selection functional 可能显著改变 apparent method gain，因此 selection 必须被视为独立分析因素；
- Equal-specific history effect 是条件性、版本相关、需要更严格 C/A/source-state 分离的机制现象。

## 22. 最终研究策略

整个项目从此遵守：

> **先证明现象真实，再证明机制，最后才设计方法。**

目标不是尽快得到一个正结果，而是建立：

observation → mechanism → controlled intervention → independent replication

的闭环。

只有当这条闭环成立，才进入算法设计。

