# 路线 A：共享嵌入的一阶历史修正

返回[执行入口](00_EXECUTION_RUNBOOK.md)；实验数量见[矩阵](05_EXPERIMENT_MATRIX.md)，统一工程约定见[实现规范](06_CODEX_IMPLEMENTATION.md)，最终裁定见[评价与决策](09_EVALUATION_AND_DECISION.md)。本文件定义待实现算法，本次仅交付规划。

## 1. 当前结论与研究动作

**先检验 EMBED 的当前方向能否在真实反馈后给固定 EMA 带来额外收益。** 最新 Clean 同状态、同范数方向干预在 V_B 的五 seed 平均 ΔAUC 为 MovieLens `+0.0003336`、KuaiRand `+0.0003559`；移除二阶预调节则分别约 `−0.0123103`、`−0.0112120`。前者支持有限反馈实验，后者要求保留二阶坐标预调节。此前结果是虚拟一步，尚未检验长期反馈；不能据此称 Adam 动量整体有害。

来源固定为发布 SHA `5ef45b18b973ed11be6fbb5f654562349998de60`：

- [最终报告第5、8节](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。该报告原“方向 C”对应本方案的**路线 A**。
- [source_direction_states](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_causal_20261110/interventions.py)：当前 data gradient 加 coupled WD，保留本步真实 vhat，匹配组内实际位移范数。
- [原始训练顺序](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_mechanism_20261106/worker.py)、[EMA 递推](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_mechanism_20261106/shadows.py)、[坐标分组](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/aug_history_20261105/group_registry.py)。

只比较 `A_MATCH` 与较简单的 `A_RAW`，不新增全共享参数网格、梯度冲突指标或自适应网络。共享表示的收益已成立；本轮不重新证明它，也不以域规模定义难度或更新质量。

## 2. 保持的训练协议

|项目|MovieLens|KuaiRand|
|---|---:|---:|
|batch / eval batch|40960|40960|
|epoch / 每 epoch 步数|50 / 20|50 / 49|
|Adam lr / β / eps|.01 / (.9,.999) / 1e−8|相同|
|coupled weight decay|1e−4|1e−5|
|固定参数 EMA ρ|.9828205985452511|.995|

FP32；无 AMP、TF32、梯度裁剪；`foreach=False, fused=False, amsgrad=False`。保持源 Clean 损失 `real.mean(1).mean()`、自然数据顺序、原始采样与 gate input detach。不得把机制 probe 的 `probe_input_full_graph=True` 改成训练图。普通 F 从 θ0 初始化，每个真实更新之后按源 FP32 顺序递推。

作用组 `G` 使用 registry 的 **S∩embedding**，即 `group_masks()['EMBED']`。domain embedding 的坐标属于 D，必须排除；不得按模块名把全部 embedding 纳入。每步另排除 `grad is None` 的张量。零梯度但非 None 的稠密张量仍按源 Adam 处理，包括其 WD 与历史更新。

## 3. A_MATCH 的完整定义

记 θₜ₋₁ 为**当前分支**步前参数，gₜ=∇L_data(θₜ₋₁)，λ 为耦合 WD。源 Adam 递推为：

\[
q_t=g_t+\lambda\theta_{t-1},\quad
m_t=\beta_1m_{t-1}+(1-\beta_1)q_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)q_t^2.
\]

\[
\widehat m_t=m_t/(1-\beta_1^{k}),\qquad
\widehat v_t=v_t/(1-\beta_2^{k}).
\]

`k` 为该张量 optimizer state 的 step，不能用全局 step 替代可能不同的张量时钟。

先执行一次原生 Adam，得到 θ_native。令组内实际位移 aₜ=(θ_native−θₜ₋₁)ᴳ、rₜ=‖aₜ‖₂；计算当前 numerator 方向：

\[
c_t=-\eta_t\frac{q_t^G}{\sqrt{\widehat v_t^G}+\epsilon},\quad
b_t=r_t\frac{c_t}{\|c_t\|_2},\quad
w_t=(1-\alpha)a_t+\alpha b_t,\quad
d_t=r_t\frac{w_t}{\|w_t\|_2}.
\]

\[
\theta_t^G=\operatorname{FP32}(\theta_{t-1}^G+d_t),\qquad
\theta_t^{\neg G}=\theta_{\mathrm{native}}^{\neg G},\qquad
F_t=\rho F_{t-1}+(1-\rho)\theta_t.
\]

候选 α 为 `{0.5, 1}`。α=1 对应当前方向完全替换；α=.5 保留部分一阶方向。α=0 是工程等价控制，必须直接保留原生结果，不做参数往返拷贝。

**状态与数值约定：**

- g 是原始 data gradient；q 仅用于候选计算。不得把已加 WD 的 q 写回 `.grad` 后再调用 Adam，造成双重 WD。
- m、v、step 均由原生 Adam 更新一次。未直接应用的一阶 m 继续积累；后续 g 来自修改后的 θ。v 保留继承状态并随真实分支递推，不能冻结或回填父轨迹 v。
- 范数参考为**本分支本步**原生 Adam 位移，不是父轨迹的未来位移。分支演化后，双方范数可以不同。
- c、范数、混合用 FP64，最终写回 FP32；Adam 状态与 F 保持源 dtype 和操作顺序。EMA 必须在坐标改写之后更新。
- r=0 时 no-op；‖c‖=0 或 ‖w‖≤1e−12·r 时回退 a 并记录；非有限量是工程失败，不是性能淘汰。不增加隐式比例 clipping。
- 记录 FP32 写回后的实际范数；相对匹配误差以 `max(r,1e−12)` 为分母，工程验收阈值 1e−4。异常先定位实现，不能调科研门补救。
- `grad is None` 的张量保持原生跳过规则，包括 WD 和 step；源模型预期稠密梯度，出现 sparse gradient 则停止工程执行。

同一状态的学习率、WD、二阶矩及组位移幅度均已控制，因此不再增加学习率网格来回答这一方向问题。范数匹配是否值得保留，由下一节更简单替代直接判断。

## 4. A_RAW：更简单的普通 Adam 替代

仅当该数据集至少一个 `A_MATCH` 通过短程门后，执行 `A_RAW`。将 EMBED 独立 paramgroup 的 β1 从 .9 改为0；非 EMBED 仍为.9。参数对象、per-tensor step、v、F、随机状态全部保留。第一步起正常递推：

\[
m_t=q_t,\quad\widehat m_t=q_t,\quad
v_t=.999v_{t-1}+.001q_t^2,\quad
\theta_t^G=\theta_{t-1}^G-\eta_t q_t^G/(\sqrt{\widehat v_t^G}+\epsilon).
\]

不做范数匹配，不重置 v/step，不另扫 β1。已有 m 会被 β1=0 的正常公式覆盖。与 `A_MATCH` 一样，只在固定起点后实施，F 保留此前完整历史。必须报告实际步幅变化；若其性能相当或更好，优先简单替代，不能把归一化层预设为必要贡献。

## 5. 五 seed 短程、选择及完整续训

**每个数据集独立裁定，不跨数据集平均抵消。** Seeds 固定2022–2026。Stage 1 普通父轨迹共两数据×五 seed×50 epochs，同时服务 B/C；在 E20、E35 **结束后**保存完整 RAM anchor。原研究状态已释放，须重放这一次公共轨迹采集；它用于获得缺失状态，不是重新证明 F 有效。

每个 α 从两个 anchor 分别运行 E21–26、E36–41。普通对照直接读取父轨迹相同 epoch。定义每 seed 的主效应：

\[
\delta_{s,\alpha}^{F}=\frac14\sum_{e\in\{25,26,40,41\}}
\left[\mathrm{MacroAUC}(F_{s,\alpha,e})-\mathrm{MacroAUC}(F_{s,0,e})\right].
\]

主均值为五 seed 的平均。每 cell 用相同四 epoch 与五 seed 计算差，E20 起点辅助均值只用 E25/E26。候选通过必须同时满足：

1. 五 seed 主均值 ≥1e−4；
2. 六个 cell 的平均差均 ≥−.001；
3. E20 起点的末两 epoch 五 seed Macro AUC 平均差 >0。

不要求每个 seed 正向或置信区间下界>0。若前两项通过但第三项不满足，记 `REVISE_TIMING`，不擅自改为 E35 启动。两个 α 都通过时按主均值排序；差≤1e−6优先.5；每数据最多一个 `A_MATCH` 进入完整续训。

随后才做 `A_RAW` 的相同两个 anchor×6 epoch×五 seed，使用相同三条门。通过则也进入完整续训；完整阶段每数据最多 `A_MATCH` 赢家和 `A_RAW` 两种配置。

完整算法**固定 E20 后开启**，续接其 E26 RAM 状态至 E50；E35 分支只提供边界信息。保留父轨迹 E1–20 的验证曲线及当时最优状态，与候选 E21–50 合并形成完整选点序列。online 和 F 各自按完整 Validation 六格 Macro AUC **最早严格最大值**选点。若最优仍在公共前缀，该端点不能归功于干预；仍按候选与父F各自完整选点报告真实增量，可能为负。仅在比较端点epoch/state hash及评价协议相同时，差值为0，不能删除早期最优来制造提升。

所有候选同时报告同刻 online/F 效应。只有 online 改善、F没有额外增量，记录 ONLINE_GAIN_WITHOUT_EXTRA_F_GAIN；F变化接近零时与既有F覆盖该改善相容，F为负时直接报告代价。不能仅由这个分支直接确认吸收机制。两种 α 都未过门则停止 A 并进入 B。动作已明显变化但反馈无收益，结论为局部干预优势未转为实际反馈优势，不继续扩充控制器。

上述 1e−4 是机制探索门，**不是正式算法 +.003 晋级标准**。完整方案、发表基线、固定 F 独立增量和 LogLoss 分别按[统一决策](09_EVALUATION_AND_DECISION.md)裁定，不能把原有 F 收益归给新组件。Test 在配置与 checkpoint 冻结后读取。若进入 Gaussian 边界，沿用该数据集 Clean 冻结赢家与 E20 切换，不重新调 α。A+B 的条件组合仅增加已冻结 B 读出，不增加 Adam 轨迹；条件由[矩阵](05_EXPERIMENT_MATRIX.md)定义。

## 6. 实现位置、伪代码与验收

新包 `experiments/ema_next_20261111/`，方向操作放 `optimizer_direction.py`，训练接线放 `worker.py`。复用源 model adapter、registry、loss、data cache、metrics、FixedShadowBank；不原地修改冻结源。建议接口：

```python
capture_embed_inputs(model, optimizer, registry) -> PreStep
apply_embed_direction_mix(model, optimizer, pre, alpha) -> StepDiagnostics
set_embed_beta1(optimizer, registry, value=0.0) -> None
```

规格伪代码：

```python
optimizer.zero_grad(set_to_none=True)
loss = source_losses(model, batch, frozen_objective)
loss.backward()
pre = capture_embed_inputs(...) if active_A_MATCH else None
if first_active_A_RAW_step:
    set_embed_beta1(..., value=0.0)  # 保留对象及 optimizer state
optimizer.step()                   # 全部状态恰好更新一次
if active_A_MATCH:
    apply_embed_direction_mix(..., pre, alpha)
bank.after_adam(flatten(model), global_step + 1)
```

无额外模型前向或梯度；训练增加 EMBED 向量计算，推理仍为一份 selected F。检查实现只用工程 seed2022，不用 smoke 性能淘汰：α=0 的 θ/m/v/step/F/RNG/loader 等价；α=1 与冻结 `source_direction_states` 单步一致；WD只加一次；None/零范数边界；非EMBED单步不变；fork不共享storage；恢复后下一步与不中断一致；第二步梯度确实来自修改后 θ。检查两数据的源 mask、loader 与真实 batch/tail，不新写一套训练目标。

## 7. RAM 生命周期与成本

遵守 `start.md`：**任何模型、optimizer、EMA、anchor、selected 或恢复 checkpoint 均不得写磁盘。** 使用 guardian RAM 和不可变 epoch 事务；磁盘只保存标量、配置、身份、成本与状态摘要。RAM anchor包含 θ/buffers、optimizer与paramgroups、F/EMA clock、epoch/globalstep、loader/RNG、前缀最优selected及来源hash。父轨迹、短程分支、续训互相不共享可变张量。

worker中断从存活guardian最后完整epoch恢复；guardian丢失只能按[自动执行规则](08_AUTOMATED_EXECUTION.md)最多一次确定性重放，并核对原记录。不能从JSON恢复参数；不能将重放计为新seed。仅在依赖该anchor的任务、冻结Test和最终记录全部完成后释放相应状态。

|工作量，两数据均触发时|新增 epochs|新增 Adam updates|
|---|---:|---:|
|公共父轨迹，A/B/C共用，只计一次|500|17,250|
|两α×两anchor×五seed的短程|240|8,280|
|条件 A_RAW 短程|120|4,140|
|一个完整候选从E26续至E50，两个数据|240|8,280|

完整候选最多 matched+raw 两个，最后一行最多发生两次；未触发任务不预支预算。原参数量 ML773470/KR1614254，单份FP32模型约2.95/6.16MiB；θ/m/v/F四份核心约11.80/24.63MiB；两anchor×五seed×两数据的核心约364.3MiB。激活、grad、scratch、selected、group mask、IPC旧/新副本另计，参数大小不能代替显存峰值。

[旧 Clean smoke](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/artifacts/ema_causal_20261110/attempt2/SMOKE_MOVIELENS.json)每epoch ML13.3294s；[KR](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/artifacts/ema_causal_20261110/attempt2/SMOKE_KUAIRAND.json)42.3881s，均含旧观察。按其乘短程工作量约1.86累计worker小时，仅作预算参考；新smoke须分项测训练、验证、改写与RAM提交。降低并发处理资源不足，不能减少seed或改batch。原guardian固定四block/50epoch完成条件不能直接套分支；按[工程规范](06_CODEX_IMPLEMENTATION.md)适配stage DAG与分支终止epoch。
