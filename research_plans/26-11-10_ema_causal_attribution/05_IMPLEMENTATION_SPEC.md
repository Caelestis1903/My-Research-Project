# 05｜实现规范：状态构造、函数干预与轨迹纯度

> 状态：**DESIGN_ONLY**。本文件规定未来 Codex 的实现责任；本文交付时新包和命令尚未实现，没有启动训练。科学比较、矩阵和统计分别以 [03](03_CAUSAL_EXPERIMENT_DESIGN.md)、[04](04_EXPERIMENT_MATRIX.md)、[07](07_METRICS_AND_STATISTICS.md) 为准。本文把公式映射为可执行的数据结构与操作。

## 1. 固定源码与复用边界

实现基底固定为 `ffa3de067fcda5a1293af83a465d741e069493f1`（原分支 `26-11-07`）。未来只新增 `experiments/ema_causal_20261110/`；复用下表的纯组件，不能调用旧整套 runner 冒充新协议，也不能更改源训练函数以方便观察。

| 已有模块 | 复用内容 | 新任务必须保留的语义 |
|---|---|---|
| [model_adapter.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/model_adapter.py) | `build_model`、`forward_logits` | fixed1 结构、原始 tower 索引、原训练 gate detach |
| [registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/registry.py) | `ParameterLayout`、参数展平/复制、alias/shape/hash | 只有同布局向量可替换；`flatten` 会 detach，不能用于建立导数图 |
| [group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/group_registry.py) | `build_path_groups` | domain embedding 按行；S/D/T/DT 完整不交划分 |
| [test_a.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/aug_history_20261105/tests/test_a.py) | `test_group_partition_and_actual_lookup`、`test_single_cell_gradient_paths` | 复用已存在的真实输入支持审计；补充 KR 与完整功能导数语义 |
| [readouts.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py) | 原概率和 raw logit 同次读取、面板 valid mask、完整 V 评价 | 原 FP32 概率计算 AUC；原 LogLoss 定义；不转发 padding |
| [shadows.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/shadows.py) | `FixedShadowBank`、`PulseWindow` | after-Adam 时钟、从初始参数初始化、有限 pulse 更新 |
| [worker.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/worker.py) | 原 loss/Adam/RNG/loader/选点的调用方式 | 不沿用旧全部候选矩阵；不复制每 epoch 全部 retained state |
| [state.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/state.py) | RAM guardian、内存 IPC、事务与 replay 思路 | 新增不可变 packet 的增量提交；参数类状态不落盘 |
| [data_bridge.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/overnight_ab_20261103/data_bridge.py) | `prepare_cache`、`cached_bundle` | train-only 预处理、原始行/split 身份、`include_test=False` |
| [selection.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/overnight_ab_20261103/selection.py) | 原六格 Macro AUC score | 最早严格最大值；不用容差改变 tie 选择 |
| [原 objectives.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py) | `losses` | Clean 原归约；Gaussian D0 replacement、同 view、源 noise key |

源模型图见 [reference.py](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/reference.py)，真实训练配置见 [RESOLVED_PROTOCOL.json](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json)。解析后必须显式固定 `target_domain=0`；KR 嵌套 source 元数据中的旧 `augmented_domain=2` 不参与本轮目标。

## 2. 新包的最小职责与公共接口

不引入第二套模型、数据预处理、指标或训练框架。新模块可合并文件，但以下接口语义必须存在。

| 新模块 | 必须实现的接口 | 责任 |
|---|---|---|
| `protocol.py` | `resolve_plan(plan_dir, paths) -> ResolvedStudy`；`prepare(study, out)` | 冻结 20 jobs、源码/数据/布局、观察事件、资源上限、假设及 Test 隔离 |
| `trajectory.py` | `create_session`、`train_one_step`、`advance_epoch` | 仅在源路径上执行一次真实 Adam；调用纯观察器 |
| `history.py` | `SparseHistory.capture/consume`；`ScoreEMA.update` | 稀疏 R 事件、窗口分解、FP64 数学审计、逐 step 分数 EMA |
| `interventions.py` | `build_state(packet, descriptor)`；`evaluate_state` | S/N、细分、近期/早期、有限动作菜单；输出状态身份 |
| `functional.py` | `capture_cut`、`mixed_readout`、`capture_fusion_parts`、`integrate_path` | 完整图割、真实混合读出、路径积分及闭合 |
| `pairs.py` | `exact_transition_table`、`panel_pair_statistics` | 全量精确 3×3 排序迁移表；固定样本对诊断 |
| `state.py` | `put_packet/get_packet/release_packet`；`commit_epoch` | 不可变 RAM packet、原子 epoch、恢复、内存计数 |
| `analysis.py` | `analyze_block`、`analyze_global`、`write_report` | 五 seed 配对、交互、H1–H3、Q1–Q5证据链与决策 |
| `evaluation.py` | `freeze_endpoints`、`evaluate_frozen` | 完整 V 选点、全局冻结、独立 Test |
| `run.py` | 统一 CLI | `audit, prepare, smoke, run, status, resume, analyze, freeze, evaluate, report, release` |

所有公共子命令使用 `--plan-dir` 和 `--out` 定位当前计划与 attempt。`prepare` 生成 `PREPARED.json`、`RESOLVED_PROTOCOL.json`、`OBSERVATION_SCHEDULE.json`；后续从其恢复冻结身份。运行参数详见 00；不得在 `resume` 时重新解释旧 manifest 或隐式搜索新超参数。

### 2.1 状态对象

~~~python
ResolvedStudy:
    study_id, plan_hash, source_commit, source_file_hashes
    jobs, configs, group_registry, panels, observation_schedule
    resource_caps, comparison_registry, data_identity

LiveSession:                      # 只驻 RAM
    online_model, optimizer, fixed_ema_fp32
    ideal_R_fp64, theta_before_step_fp32
    score_ema_prob_fp64, score_ema_logit_fp64
    sparse_history, pulse_fast, pulse_slow
    matched_horizon_pulse_fast, matched_horizon_pulse_slow, pulse_clocks
    loader_generator, train_rng, epoch, optimizer_step
    selected_online, selected_F_packet, observation_cursor

AnchorPacket:                     # 不可变、RAM-only
    job_id, epoch, optimizer_step, state_hashes, layout_hash
    online_fp32, R_fp64, old_R_fp64_by_effective_h
    numerical_error_scalars, panel_prediction_receipts
    optional_source_assay_ref
    matched_horizon_fast_fp32, matched_horizon_slow_fp32  # E35/E50 only
    matched_horizon_start_step, matched_horizon_steps

EndpointPacket:                   # RAM-only
    selected_online_state_fp32
    selected_F_state_fp32, online_at_selected_F_fp32
    selected_epochs, validation_metrics, hashes

InterventionDescriptor:            # 可持久化元数据
    contrast_id, anchor_id, parameter_mask_id, window_id
    background_id, coefficients, graph_cut_id, donor_id
    exact_or_approximate, numerical_audit_ref
~~~

Test冻结清单固定为Osel、Fsel、OatF、S-only atF、N-only atF。后两者从OatF/Fsel逐坐标重构，不额外训练，不按Test挑组或时间。模型端点在全部四块完成且Stage4机制冻结之前不交给Test进程。

参数向量、梯度、moment、R、历史向量和可恢复 RNG 状态均属于模型状态，不能借 `npz/json/pickle/tmpfs` 名称落盘。面板行身份、标量、有限预测、参数 hash 可以保存。非可恢复的 RNG namespace/seed 配置可持久化；真实恢复状态留 RAM。

## 3. 一次真实更新的顺序

### 3.1 原训练过程

每条逻辑 job 独立调用源 `history.setup(seed, device)`，loader 使用原 CPU generator `seed+11000`。每 epoch 一次原 `randperm`，batch40960且保留原尾批。不得由于观察改变 batch、梯度归约、Adam step 数或 Gaussian 随机流。

~~~python
theta0 = flatten(source_model).clone()
F32 = theta0.clone()
ideal_R64 = zeros_like(theta0, dtype=float64)
q_prob64, q_logit64 = predictions(theta0, fixed_panels).to(float64)

for source_batch in original_order:
    theta_before = flatten(online).clone()
    optimizer.zero_grad(set_to_none=True)
    loss = source_losses(online, batch, target_domain=0, ...)
    loss.backward()

    # 仅预注册的 e20/e35 source assay，可保存当前真实梯度。
    capture_pre_step_if_registered(online, optimizer, current_data_grad)
    optimizer.step()                 # 唯一真实 Adam 更新
    capture_post_step_moments_if_registered(optimizer)
    t += 1
    theta_after = flatten(online)

    # 必须保持源 FP32 mul_/add_ 顺序。
    F32.mul_(rho).add_(theta_after, alpha=1-rho)

    delta64 = theta_after.double() - theta_before.double()
    ideal_R64 = rho * ideal_R64 - rho * delta64

    # 观察全部在独立 scratch 上；模型/loader/训练 RNG 不变。
    observation_guard.begin()
    p32, z32 = read_fixed_panels(theta_after, eval_mode=True)
    q_prob64 = rho*q_prob64 + (1-rho)*p32.double()
    q_logit64 = rho*q_logit64 + (1-rho)*z32.double()
    update_pulses_started_before_t(theta_after, t)
    finish_pulses_ending_at(t)
    if is_regular_pulse_endpoint(t):
        collect_bridge_rank_gradient_summary(F32, P_A, full_functional_graph=True)
    capture_sparse_events_and_source_statistics(t, theta_after, F32)
    begin_pulses_starting_at(t, F32)  # 新窗口首个更新必须为 t+1
    observation_guard.end()
~~~

初始化的 `q0` 不能置零，否则权重与参数 EMA 不同。所有pulse时钟、含E35/E50匹配H的两个临时shadow，均进入RAM恢复状态；窗口开始step本身不更新新shadow，首个更新为下一step。每个真实 optimizer step 都更新分数 EMA；不能用每 epoch/anchor 预测冒充全历史组合。分数 EMA 没有反馈到 loss、在线参数、Adam 或参数 EMA。重复 row 可在固定面板内去重一次前向，再按冻结索引 gather 回原角色。

每个常规pulse终点必须现场执行08§4的桥接梯度观察：在F的P_A rank面板上，对六格各自全部256×256正负组合的平均softplus风险再六格等权，只求一次完整功能梯度，用同一向量归约all/S/N的RMS并保存标量。全研究最多200次该观察，不能沿用旧4096随机pairs的梯度定义，也不能等block最终化再索取已经释放的E15/E25等F状态。梯度仅存在scratch/RAM，使用 `autograd.grad`，不写online `.grad`，归约后立即释放。

在 epoch 结束时，先计算完整 V 的 O/F，按各自最早严格最大值更新 selected。F 选点改变时，更新同刻 O/F 与四个旧历史分量的 packet；不得用 online 独立 selected 充当“同刻 O”。随后完成本 epoch 观察事件并提交事务。

非固定anchor的selected面板预测和窗口参考预测随当前selected packet驻RAM替换；50epoch结束后只持久化最终selected的一套数组。历次选点身份与完整V曲线仍保存为标量，不能将每次改善时的数组都永久累积而突破06的输出预算。

### 3.2 轨迹纯度验收

只做能排除实际污染的必要工程检查：

- 相同源 batch、Clean/Gaussian、原尾批，对比观察开/关后在线参数、原梯度、Adam state、loader状态、训练 RNG、F 的一致性；同一环境优先逐位相等。
- scratch 不得共享 online 的可变参数存储；读出不得调用真实 optimizer。梯度观察使用 `autograd.grad`，不写 online `.grad`。
- 每个 hook 在 `finally` 中移除，恢复 scratch 的模式；不复用带旧 hook 的 scratch。
- regular evaluation 前后记录原 RNG 和参数/optimizer digest；正式每 epoch 记录 `init_hash/order_hash/online_hash/F_hash`。工程检查通过后，不为“多验证一次”重复全部训练。
- `torch.func.functional_call` 的输入参数/buffer可受模块原地操作影响，因此必须使用 scratch 与独立 buffer，采用完整映射、`strict=True`，不得将 online 的 buffer 引用直接传入。参见 [PyTorch 2.9 functional_call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html)。

源模型无 BatchNorm/Dropout，但仍枚举 `named_buffers` 并核对身份。若执行版本出现新的有状态 buffer，停止该语义不匹配，不静默选择“平均 buffer”或“重新估计统计量”。

## 4. R 的构造、FP32 舍入与时间状态

### 4.1 实数公式与实际状态

主干预永远以实际源 FP32 O/F 为准：

$$
R_t^{act}=\operatorname{float64}(F_t)-\operatorname{float64}(O_t).
$$

S/非S组合优先逐坐标从原 O 或原 F 拷贝，完整F直接引用原F，保证参数端点不因“先减再加”改变。时间分解采用：

$$
B_{t,h}=\rho^{h_{\rm eff}}R^{act}_{t-h_{\rm eff}},
\qquad A_{t,h}=R^{act}_t-B_{t,h},\qquad h_{\rm eff}=\min(h,t).
$$

`A` 是近期修正，`B` 是更早修正；二者相加在 FP64 中闭合到实际R。模型前向只把最终合成参数一次转回 FP32，不逐项 FP32 相加。每个混合状态记录：

~~~python
recent64 = R_actual64 - old64
candidate32 = (online32.double() + selected_components64).float()
~~~

相同有效窗口只生成一个物理状态；初期 `t-h_eff=0` 的旧分量严格零，不分配 P 维零副本。

### 4.2 不把浮点误差隐藏进历史机制

理想实数更新恒等式通过 `ideal_R64 = rho*ideal_R64-rho*delta64` 同步审计。这是一份只读 FP64 向量，不替代实际F。定义：

$$
E_t=R^{act}_t-R^{ideal}_t.
$$

事件点保存 `max_abs(E_t), RMS(E_t)` 标量；近期修正偏离理想加权更新的误差上界为

$$
\|E_t-\rho^hE_{t-h}\|_\infty
\le \|E_t\|_\infty+\rho^h\|E_{t-h}\|_\infty.
$$

无需为了该审计再保存一套历史参数向量。所有固定锚点即时读出 `O+R_ideal` 的面板预测，与实际F比较概率/排序；只保存误差标量和必要有限预测。理想递推、实际R重构、正负更新符号和时间指数必须先在短序列算例闭合。

实际O/F的正式结果不因理论舍入差异重定义。若理想/实际预测的差异达到会改变主要机制方向的程度，将该部分单列为数值残差，并定位源算术；不能把它解释为历史参数效应。对 margin 归因，若数值误差界覆盖零点，则该 pair 标记为 `NUMERICALLY_UNRESOLVED`；真实 p 的精确 AUC/迁移表仍保留。

### 4.3 稀疏事件缓存，不保留逐 step 参数 ring

目标集合包括每个 epoch 末尾——因为F的 selected epoch 事先未知——以及预注册 pulse/source assay事件。为每个目标 `t=e*S` 预先生成需求 `max(0,t-h)`。到达该 source step 时保存一份 FP64 R，按目标引用计数：

~~~python
needs = build_schedule(all_epoch_ends, registered_h)
for t in optimizer_steps:
    if t in needs.capture_steps:
        history.put(t, R_actual64.clone(), consumers=needs[t])
    if t in needs.target_steps:
        old = {h: rho**h_eff * history.get(t-h_eff) for h in unique_h_eff}
        complete_this_epoch_readouts(old)
        retain_packet_only_if_fixed_anchor_or_new_F_selected(...)
        history.consume_all_targets_at(t)
        history.release_zero_refcount()
~~~

锚点为E5/10/20/35/50，另加F selected同刻packet；重复锚点使用同一对象。四窗口ML20/40/80/160 steps，KR69/138/277/553 steps。调度缓存的安全向量上界采用

$$
N_{\rm pending}\le \sum_h\lceil h/S\rceil+4,
$$

ML最多19、KR最多27个FP64 R副本；`prepare`应从实际事件表算出更紧的精确峰值。保存权重后的 old 分量只在完整 anchor packet 中发生，不能长期再留所有原始 R 副本。

同一稀疏事件还在RAM缓存该时刻的在线面板p/z及所需训练目标统计，供实际窗口端点比较；引用归零即释放。持久化仅保留五固定anchor和最终tF对应的至多四个窗口起点的p，最多120×4份；每份源形状上限147,456 bytes，共70,778,880 bytes。窗口logit/rank-risk等统计在状态可用时现场聚合，不将全部scheduled epoch×h的p/z写盘。每step的面板分数始终流式消费，不存在完整逐step预测档案。

## 5. 参数干预与跨组/时间组合

`build_state` 接收显式背景与mask，禁止根据参数名字临时猜组。

~~~python
# 共享/非共享二因素
O  = online32
S  = coordinate_splice(O, F, mask_S)
N  = coordinate_splice(O, F, mask_N)
SN = F

# 时间二因素
A  = cast32(O.double() + recent64)
B  = cast32(O.double() + old64)
AB = F

# 重要组 G 的分量删除，背景必须写在descriptor里
F_without_G_recent = cast32(F.double() - mask_G * recent64)
O_with_G_recent    = cast32(O.double() + mask_G * recent64)
~~~

每个状态返回 `state_hash` 与 `InterventionDescriptor`。参数 hash相同的读出可复用预测，但逻辑比较标签仍保留；同hash状态不能增加独立样本数。细分及组×时间展开仅按03/04的五seed规则触发，按完整五seed执行。S与N均重要时，S的非routing embedding/共享STAR与skip/shared experts和N的D/T/DT共六类，各自单独与删除读出合计最多12个额外状态/anchor。固定1的路由支持成立时，可由族读出在相应cell上精确得到Dd/Tt/DTdt的leaf指标，其他cell拼回父背景；不再为leaf追加全组合。不得只报告单组增量后求和替代真实组合。

实际模型计数与support见01和源GROUP_MAP。工程审计至少验证：坐标全覆盖无交叠；domain embedding行正确；gate/tower `3*t+d` 映射正确；源与full功能图的不支持cell仍无响应。单个样本的零梯度不能用来把结构上可达的参数从组中删除。

## 6. 预测与样本对分析

### 6.1 面板身份

复用旧P_A/P_B训练互不交池与rank/natural角色：每cell256正+256负，每域512自然行。V_mech每域最多4096真实Validation行，V_A/V_B用冻结row hash分侧；KR不足4096的域保留实际人数，`valid_mask=(indices>=0)`。标签、原始row_id、角色索引只保存一次。

固定32768个均匀正负pair/cell用于细粒度诊断；若可用对少于上限，使用全部。保留旧4096 pairs子面板用于历史指标桥接，不能因哪个面板效果更好换面板。pair随机流与训练噪声、loader完全分离；同数据集同seed的Clean/Gaussian使用同一面板身份。

### 6.2 全量精确排序迁移表

对每cell的O与干预状态，令 `a=2B(m_O), b=2B(m_I)`，值为0/1/2。实现二维秩计数：

1. 分别压缩O/I的负样本预测坐标，保留相等组。
2. 对O分数排序扫描，用Fenwick树统计I坐标的小于/等于/大于计数。
3. O相等组查询必须先于整组插入；用组内I排序计数补齐O ties。
4. 得到全部9格 `N_ab`；使用int64，先验证正负pair总数不溢出。
5. 验证 `sum(N_ab)=n_pos*n_neg`，行/列边缘各自复原准确O/I AUC，并验证
   $$
   \Delta AUC=\frac{\sum_{a,b}(b-a)N_{ab}/2}{n_+n_-}.
   $$

只在小型包含ties/全同分/饱和概率的算例中与暴力枚举比较；正式数据不枚举全部pairs。全量预测允许在一次评价的RAM缓冲中暂存，用完释放；不把全量每状态预测持久化。复杂度O(N log N)，空间O(N)。

固定面板的32768 pairs分块计算margin、纠正、损害及特征分层；只持久化有限预测和聚合计数，不保存带长字符串的逐pair明细CSV。pair不能视为独立seed。

## 7. 从参数到函数的干预

### 7.1 历史预测组合

主比较为每步源概率的指数组合 `q_prob`；`sigmoid(q_logit)` 为单列次要比较。两者都是相同完整时钟、相同真实rho、相同初始化的**固定面板精确递推**，不是完整人口预测集成。保存每个anchor的 q；非anchor q仅当前值和恢复状态留RAM。

报告F、q_prob、q_logit、O的真实p排序、纠正/损害重合和margin差。不能把 `ΔAUC(q)/ΔAUC(F)` 自动写成唯一解释百分比；F与q相近或不同都必须结合真实pair与函数对照。

### 7.2 完整计算图割

预定义两个图割：

- `h=star_mlp(...) + skip_conn(input_emb)`，必须同时替换进入**全部gate、shared experts、domain experts、task experts**的h。
- 每个 `u_dt=fused_fea[3*t+d]`，替换进入对应tower的整个输入。

对既定对比端点 `theta_a/theta_b`，在相同行/同batch下先捕获donor的cut tensor，再在recipient scratch中进行前向pre-hook替换，得到 `s_ab=H_b(z_a)`。不能只替换gate输入就声称切断了整个h路径。endpoint重构 `s_00`/`s_11` 必须分别等于原两个端点的源前向；混合读出输出的是该具体干预效果。

每个batch消费完立即释放激活。GPU同时最多一份donor激活与一份recipient计算图，不保存全数据中间表示。记录每row/每pair的上游、下游与交互恒等式闭合，另用真实p重新计算AUC。AUC交互与logit交互分别命名。

若S×D交互是主要发现，再执行03规定的STAR乘积交叉项控制；记录固定embedding和所有背景，不能把embedding同时变化的效应归给单独的 `R_S*R_D`。

### 7.2.1 必做的融合函数分解

在E35与F-selected（重合则去重）对O→F，捕获同一行的四个shared expert输出 `e_j`、对应cell gate权重 `g_j`、domain输出 `d`、task输出 `t` 和融合输入 `u`。捕获tensor通过注册在shared/domain/task expert与gate的forward hooks获取；tower pre-hook读取实际融合 `u`。不重写原训练forward。

固定1融合给出精确恒等式：

$$
\Delta u=
\underbrace{\sum_jg_j^O\Delta e_j}_{\text{expert content}}
+\underbrace{\sum_j\Delta g_j e_j^O}_{\text{routing}}
+\underbrace{\sum_j\Delta g_j\Delta e_j}_{\text{content-routing interaction}}
+\underbrace{\Delta d}_{\text{domain output}}
+\underbrace{\Delta t}_{\text{task output}}.
$$

`capture_fusion_parts(scratch, x)` 返回按cell索引的 `{gate, shared_experts, domain_output, task_output, actual_fused}`；`fusion_part_readout(O, F, part, mode)` 在相同batch先计算两端对应输出，FP64形成五项，再一次转回FP32注入tower输入。FP64五项恒等式对FP64重新计算的融合差验收；实际source FP32 fused差与五项和之间的舍入残差单列，不能强制其误差达到1e−10。真实删除干预的背景始终是source实际 `u_F`，不以FP64重算值悄悄替换源端点。按源 `3*t+d` 路由，不根据模块名称猜cell。

五个必做删除读出为 `tower_F(u_F - part_k)`，保留F的其他功能状态；每个在完整V和固定面板真实前向，记录删除后纠正/损害及margin。五seed的V_A选择最多一个最重要功能分量k。第6个读出针对已经定位的主要参数组或时间修正X，明确端点 `theta0=F-X, theta1=F`，重新计算同一功能类别的 `part_k^(X)`，在F融合接口删除它，真实读出 `tower_F(u_F-part_k^(X))`。在V_B检查原来由恢复X纠正的pairs有多少因此丢失，直接连接“X→该功能成分→排序纠正”。不重新搜索其他功能类别，不再添加O背景注入。对应选择与行侧规则遵循03/07。共最多6个新读出×2anchor×20轨迹=240个新增完整V状态。它们是具体函数干预，不是新的参数训练算法或Test候选。

`part_k`必须针对每个真实样本计算，不能以均值向量替换；全V分batch流式处理，保存标量与固定pair聚合后释放tensor。不要把五个删除AUC相加。五项表示恒等式回答函数改了什么，真实删除与来源对比回答该变化是否导致纠错，二者都必须保留。

### 7.3 参数路径与积分

参数路径 `theta(alpha)=theta0+alpha*R` 在scratch上计算。必须开启 `full_functional_graph=True`：源训练gate的detach保留，但函数导数要包含gate输入变化的路径。源训练梯度作为不同的 `source_detach_proxy` 单列，不能拿它验证积分闭合。

固定路径节点0/.25/.5/.75/1用于真实forward形状；JVP积分默认8点，只对未满足数值闭合的已注册对比升到16、32点，最多32。以 [PyTorch 2.9 jvp](https://docs.pytorch.org/docs/2.9/generated/torch.func.jvp.html) 为接口依据；若某算子缺forward AD，使用固定小pair子面板的reverse-AD/VJP等价方向导数，记实际成本，不能改在线图或无界增加积分点。

归因主证据仍为真实状态/图割干预。积分的端点score与margin闭合误差逐pair保存聚合；未闭合的近tie样本保留残差，不能用一阶梯度代理替代原endpoint。

## 8. 训练来源、历史指标和有限动作桥接

D3只观察原训练：每step流式读取固定训练/验证面板O分数，保存窗口需求点的分数、源真实loss及冻结自然面板loss统计；不把全部step×面板预测写盘。原始window端点与参数R事件共享调度。对同一pair集合比较实际历史排序破坏与当前EMA纠正的重合，区分“实际参数读出反转某类变化”与“若删去过去更新后仍保持其余梯度不变”的不同命题。

H桥接复用10个one-epoch pulse（E5/10/…/50），只增两份临时fast/slow shadow。两组各7动作菜单、拟合折、简单参照和前瞻评分按03/07/08冻结。除真实菜单预测外，必须在终点现场采集上文规定的F完整rank梯度all/S/N摘要；已归约标量可在block最终化时拟合。H的fast/slow只在真实pulse终点存在；若F selected不是这些epoch，不能伪造对应pulse。H相关完整V评价只在预注册固定core anchors，其他终点用固定面板。

### 8.1 H2的时域与幅度匹配控制

此控制是H2准备指导下一阶段B方向时的必要判别，不扩超参数网格。采集阶段仅在E35/E50，为 `H=round(true_half_life_steps)` 额外维护global fast/slow：在 `start=t-H` 完成正常F更新后复制当时F，随后从 `start+1` 到t恰好H步，分别用 `rho_fast=rho²`、`rho_slow=sqrt(rho)` 更新。不能用epoch近似start，也不能把旧one-epoch pulse当作这一配平控制。到anchor保留两向量与时钟；未触发时写 `NOT_TRIGGERED` 并释放。

触发后最多两个共同scope `G∈{all,S,N}`：all加已定位的主要S或N。每scope分别用P_A六格Macro AUC在 `{F,fast,slow}` 与 `{F,drop_recent,drop_old}` 中选一个动作，fast/slow仅替换G坐标，其余仍为F；nonrho修正使用同H。令两动作相对F位移为 `d_rho,d_nonrho`，同一scope内计算

```python
r = min(norm(d_rho), norm(d_nonrho))
alpha_rho = r / norm(d_rho)
alpha_nonrho = r / norm(d_nonrho)
matched_rho = F + alpha_rho*d_rho
matched_nonrho = F + alpha_nonrho*d_nonrho
```

两个alpha均不超过1，不放大任一动作。若任一位移为零，记录 `NOT_INFORMATIVE_ZERO_DISPLACEMENT` 与该seed原值；不能把相同F读出记作反对H2。其余情况真实前向两个配平状态，V_B比较，最多2state×2scope×2anchor×20轨迹=160个新增完整V状态；只使用冻结的P_A选择，不按V_B重选动作。

记录action horizon、scope、两动作选择人口、原范数/缩放系数、原始与配平后真实效用。差异配平后仍存在，才支持这个有限范围的方向/操作类型H2；若消失，解释收窄为时域或幅度差，不能声称rho全族无效，也不自动判A胜出。所有触发condition均完成五seed。

### 8.2 有限optimizer来源读出

有限optimizer来源读出只针对Clean的E20/E35，且仅在会改变下一阶段方向时消费。为避免事后再训练，可在两锚点保留最近真实step的 `theta_before/data_grad/mhat/vhat` 四份FP32向量；coupled-WD梯度和实际Δ由这些与锚点O推得，不重复保留。实际、同预条件当前梯度、同方向等向尺度三类读出均做规定的组范数匹配；没有新的真实 `optimizer.step`，不续训，不调beta。它识别给定状态下的分子/预条件作用，不能宣称干预了整个Adam历史。

## 9. 文件格式、数值记录与事务

下表的科学表名与07的逻辑名一致；物理文件可使用 `.csv.gz`。必须生成 `ARTIFACT_INDEX.json`，登记逻辑名→物理路径、schema版本、内容hash和行数；分析器按此索引读取，禁止各模块自行重命名形成两套表。最少输出集合：

| 输出 | 关键字段 |
|---|---|
| `SOURCE_AUDIT.json`、`RESOLVED_PROTOCOL.json` | 源commit/hash、原始数据/cache/row/split身份、rho/step、布局、组、软件与资源 |
| `OBSERVATION_SCHEDULE.json` | 每job capture step、target step、h/h_eff、consumer、预计峰值、逻辑/物理读出计数 |
| `EPOCHS.csv.gz` | seed/condition/epoch/step、O/F full V六格AUC/LL、hash、selected flag、训练loss、成本 |
| `INTERVENTION_METRICS.csv` | descriptor/hash、五seed身份、anchor/background/group/window、六格指标、数值残差 |
| `PAIR_TRANSITIONS.csv` | 每状态/cell的9格int64、总pairs、重构AUC、精确差及误差 |
| `FUNCTIONAL_PATCH.csv`、`FUSION_COMPONENTS.csv`、`PATH_CLOSURE.csv` | graph cut/path/endpoint、表示/下游/交互、margin闭合、真实翻转、residual |
| `ACTION_MENU.csv`、`BRIDGE_PREDICTIONS.csv`、`BRIDGE_VALUE.csv` | 冻结menu/feature/fold、当前及未来效用、选中动作、oracle诊断、simple参照 |
| `COSTS.csv`、`REPLAY_LEDGER.jsonl` | 逻辑与物理计数、训练/观察/IPC/分析、峰值、重试与身份核验 |
| `BLOCK_COMPLETE.json`、`MECHANISM_FREEZE.json` | 五seed完整性、展开与不展开的理由、分析/假设/比较hash、端点清单 |

prediction-only NPZ/分块数组禁object、`allow_pickle=False`；schema逐项白名单。原始p/z为FP32；q及差值计算为FP64。若为保存压缩q转FP32，必须保留FP64计算出的指标和转换误差，并以FP64为分析权威；优先保存anchor q为FP64，成本很小。标量表在写前选择gzip，不能先制造巨型冗余CSV再靠删除救预算。

epoch事务采用“immutable新packet登记 → 可恢复active状态提交 → 标量/有限预测原子发布 → commit ledger确认”。只有已确认epoch进入分析；写入失败回滚未发布观察，不倒退已有commit。packet按独立ID引用，旧anchor不得每epoch重新deepcopy。源实现的内存序列化方式可以复用，但新提交必须只传本epoch增量和active状态。

必要数值验收包括：源轨迹纯度；R/window符号与指数；完整S/N/时间重构；support；原概率ties及全量迁移表；q递推初始化；完整图割endpoint；full导数与积分闭合；RAM事务恢复。验收不是科学算法筛选，工程seed不进入五seed统计。

## 10. 实现完成条件

未来实现者须交付接口、CLI、数据schema、真实运行计数器、恢复分支及有限必要检查；独立reviewer确认计算图、矩阵、Test门和内存路径后才能进入正式队列。每个重要公式必须对应某个状态构造或数值恒等式检查。

本次规划到此不创建Python模块、执行工程smoke或正式训练。执行时的源码修改仅限新包，原研究报告与原始结论保持独立；新结果通过新报告解释，不覆盖旧记录。

