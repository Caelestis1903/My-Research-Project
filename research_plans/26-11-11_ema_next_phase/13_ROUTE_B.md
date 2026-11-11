# 路线 B：共享历史的有界预测组合与单模型蒸馏

本文件是 [算法总规范](04_ALGORITHM_SPECIFICATIONS.md) 的路线 B 详细定义。实验身份、运行依赖见 [实验矩阵](05_EXPERIMENT_MATRIX.md)，晋级、停止、路线 C 教师替换和正式 Test 冻结规则统一服从 [评价与决策](09_EVALUATION_AND_DECISION.md)。本轮交付仅为规划，以下算法尚未实施或取得新实验结果。

## 1. 研究选择与直接证据

**主研究对象是“参数重心不变、历史信息在预测函数中组合”的 `B_SPLIT`；更简单替代是单次前向的 `B_SHARED_ONLY`。** 先在同一批普通 Adam 父轨迹上比较两者，只有已冻结教师具备值得迁移的额外信息时，才按决策文件进入 `B_DISTILL`。不重新搜索 EMA 系数，不保存全部历史模型，不把固定面板上的全步预测 EMA 当成可直接部署的模型。

冻结证据源：`26-11-11`，commit `5ef45b18b973ed11be6fbb5f654562349998de60`。

| 源码或报告 | 本方案继承的具体内容 |
|---|---|
| [trajectory.py](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_causal_20261110/trajectory.py) | `create_session/train_batch` 在每次实际 Adam 更新之后计算固定面板 online 预测，按与参数 F 相同的逐步 ρ 累积；q 初值为初始模型预测。 |
| [history.py](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_causal_20261110/history.py) | `ScoreEMA` 的逐步递推；稀疏历史按引用计数保留/释放的工程思路。 |
| [FINAL_REPORT.md §4、§10](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md) | 全步 logit/probability 组合的固定面板收益；旧参数、Adam、EMA 与 selected 状态已经释放。 |
| [QUANTITATIVE_APPENDIX.md §06](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/QUANTITATIVE_APPENDIX.md) | 五 seed 配对差、区间、anchor 和原始数表身份。 |
| [group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/aug_history_20261105/group_registry.py) | 共享坐标 S 的真实划分；domain embedding 的领域行不属于 S。 |
| [readouts.py](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/experiments/ema_mechanism_20261106/readouts.py) | 一次真实模型前向同时取得 tower raw logits 与原概率；完整 Validation 指标口径。 |

晚期 E35/E50、固定 `V_mech` 面板、五 seed 的已有结果如下。它们是本路线的进入依据，**不是新算法独立选点的 Test 性能**。

| 条件 | q_logit−F AUC | q_logit−F LL | q_probability−F AUC | q_probability−F LL |
|---|---:|---:|---:|---:|
| MovieLens Clean | +0.0007681 | −0.0055907 | +0.0006661 | −0.0077528 |
| KuaiRand Clean | +0.0021033 | −0.0037093 | +0.0021162 | −0.0035134 |
| MovieLens Gaussian D0 | +0.0007821 | −0.0063159 | +0.0006708 | −0.0085027 |
| KuaiRand Gaussian D0 | +0.0026413 | −0.0032157 | +0.0025154 | −0.0029617 |

两种集成在四条件下 AUC 差五 seed 均正、LL 差均负。已有 pair 重叠表显示，它们与 F 的大部分纠正相同，同时存在新增纠正。这已经足以推进有界近似研究；本轮不再采集一套全 optimizer-step q 来重复确证。

共享参数是 F 的主要收益来源，但原 q 使用完整历史模型。本方案以 `S/all` 同轨迹对照判断新增函数组合是否也主要来自共享状态，不先把原 q 的优势归为 MDMT 特有收益。

## 2. 参数平均、预测平均和本方案的连接

设一次源 Adam 更新后的在线参数为 θ_t，固定 EMA 为 F_t，初值 F_0=θ_0。理想数学递推为：

\[
F_t=\rho F_{t-1}+(1-\rho)\theta_t
=\rho^t\theta_0+(1-\rho)\sum_{i=1}^{t}\rho^{t-i}\theta_i
=\sum_{i=0}^{t}w_{t,i}\theta_i.
\]

权重和为 1，包含初始模型的质量 \(w_{t,0}=\rho^t\)。参数平均的预测是 \(z(x;F_t)\)，全步 logit 平均是 \(q_t^z(x)=\sum_iw_{t,i}z(x;\theta_i)\)，一般不相等。局部可作二阶展开时，一阶差因加权位移均值为零而抵消，剩余差与历史参数分散度及预测函数的非线性有关。此展开用于解释设计动机，不用于断言曲率符号、ReLU 跨区域行为或 AUC 必然改善。

两种函数组合分别为：

\[
p_t^{\rm logit}(x)=\sigma\!\left(\sum_iw_{t,i}z(x;\theta_i)\right),\qquad
p_t^{\rm prob}(x)=\sum_iw_{t,i}\sigma(z(x;\theta_i)).
\]

sigmoid 对单个分数单调，却不能与平均交换，因此两者的排序也可能不同。概率平均的 BCE 凸性只约束它与组件 BCE 的加权平均之间的关系，不保证它优于 F。

`B_SPLIT` 用两段历史的**参数均值模型**代替全部历史模型，再在函数空间组合。它保留全部旧尾部及 F 的参数重心；它能否保留原 q 相对 F 的增量，是要回答的实验问题。

## 3. 固定配置、参数组和时钟

| 项目 | MovieLens | KuaiRand |
|---|---:|---:|
| 每 epoch 实际 optimizer steps S_d | 20 | 49 |
| 源固定 EMA ρ | 0.9828205985452511 | 0.995 |
| H 候选，单位 optimizer step | 20、40、80 | 69、138、277 |
| 基准 H | 40 | 138 |
| 参数总数 P | 773,470 | 1,614,254 |
| 共享坐标数 P_S | 483,152 | 661,920 |
| 原 train 行数 | 800,167 | 2,006,132 |
| 原 Validation 行数 | 100,021 | 250,767 |

H 直接继承最新归因代码的前三个窗口。KuaiRand 的两倍半衰期独立取整是 **277**，不能先将半衰期取整为 138 再乘 2 改成 276。每个实际 optimizer step 只更新一次 F，观察与评估不推进时钟。

参数组采用冻结 `build_path_groups` 生成的坐标 mask \(M_S\)。共享 embedding、shared_weight/bias、skip_conn、star_mlp、expert 属于 S；域、任务及域任务专属坐标遵循原 registry。禁止按名称粗略重建分组，也不因某个 tensor 含有共享行就把整个 tensor 都判为共享。

父轨迹首先为两个数据集 Clean×seeds 2022–2026×50 epoch。普通 Adam、batch、原损失、数据划分与自然采样完全继承源协议；这批轨迹同时为路线 A 分叉、B 读出和 C 可见信号提供状态。不要为每一个 H、Scope 或聚合方式分别训练。

## 4. `B_SPLIT` 完整数学定义

### 4.1 两段历史参数均值

在评估步 t，对预注册 H 定义：

\[
h=\min(H,t),\qquad a=\rho^h.
\]

t=0 时只返回 F_0。t>0 时：

\[
\mu_{\mathrm{old},t,H}=F_{t-h},\qquad
\mu_{\mathrm{recent},t,H}=\frac{F_t-aF_{t-h}}{1-a}.
\]

因此：

\[
(1-a)\mu_{\mathrm{recent},t,H}+a\mu_{\mathrm{old},t,H}=F_t.
\]

\(\mu_{\rm recent}\) 是最近 h 次在线参数状态的归一化指数均值；\(\mu_{\rm old}\) 是此前全部历史及初始化质量的均值。旧尾部没有被截断。两者不是前一轮 `recent_R/old_R` 参数修正，不得互换公式或名称。

### 4.2 `Scope=all` 与 `Scope=S`

全参数版本：

\[
\phi_r=\mu_{\rm recent},\qquad \phi_o=\mu_{\rm old}.
\]

共享版本：

\[
\phi_r=M_S\odot\mu_{\rm recent}+(1-M_S)\odot F_t,
\]

\[
\phi_o=M_S\odot\mu_{\rm old}+(1-M_S)\odot F_t.
\]

共享版本的两个模型共用 F_t 的非共享坐标。两个 Scope 均满足 \((1-a)\phi_r+a\phi_o=F_t\)。因此将这两个模型重新做参数平均只会回到 F，无需另开“参数融合”训练臂。

### 4.3 推理读出

对同一输入 x，各执行一次 eval/no_grad 前向，取得两个任务的 raw logits \(z_r,z_o\) 和原模型概率 \(p_r,p_o\)：

\[
z_B=(1-a)z_r+az_o,\qquad p_{B,\rm logit}=\sigma(z_B),
\]

\[
p_{B,\rm prob}=(1-a)p_r+ap_o.
\]

同一对模型的 logit/probability 读出共用两次前向，不重复执行四次。正式 AUC/LL/Traffic/Pooled 使用相应概率与原评价函数；raw logits 用于排序诊断和 C 的可见权重拟合。不能对经过截断的概率反求 logit。

训练期只有源在线模型接受反向和 Adam 更新，F 仍按源递推；B 的参数组合不反向、不改变 m/v 或在线参数。`B_SPLIT` 是完整可部署的两模型预测算法，但不是已形成训练反馈的优化器算法。

### 4.4 数值和 buffers

1. 实际 F 仍按源 FP32 递推；差分构造时将 F_t 和 F_(t−h) 转 FP64。用 `-expm1(h*log(rho))` 计算 \(1-a\)，不得用随意加 epsilon 改动归一化。
2. FP64 中记录重心恒等式残差；两态 cast 到 FP32 后另记重心误差。实际前向用 cast 后参数，不以理想 FP64 端点替代科学结果。
3. 所有组件使用当时源模型的同一完整 buffers 映射；buffers 不参与参数差分、EMA 或 Scope mask。记录其名称、dtype、shape、hash。
4. 已审计源模型无 stateful batch layers，构建时 assert 无 BatchNorm/Dropout。仍必须核对 buffers 身份；若与冻结源不符，进入实现错误处理，不暗中重算统计或改变训练行为。
5. eval/readout 必须恢复原 training 标志，参数替换只在 scratch/functional 映射进行，源模型与 optimizer 状态及 RNG 均不能变化。

## 5. `B_SHARED_ONLY`：必须比较的单模型替代

定义：

\[
\psi_t=M_S\odot F_t+(1-M_S)\odot\theta_t.
\]

使用 \(p(x;\psi_t)\) 单次前向。它保留共享 EMA、非共享坐标使用当前 online 参数；不需要 lag F、不增加 H，不改训练。buffers 与同刻 O/F 完全一致。

前一轮 `S_atF` 是 F 选点时刻的干预读出，本轮 `B_SHARED_ONLY` 有自己的完整 Validation 选点曲线。不能把前一轮同刻干预的数值冒充本轮已完成的独立选点结果。

该对照能回答：是否只需去掉收益较弱的非共享参数平滑，就能获得与两模型预测组合相近的结果。若它已经满足相同收益目标，应优先保留单模型方案；不得为了路线名称强行选择 B_SPLIT。

## 6. 选点、配置冻结与必要观测

每个 Scope×H×聚合组合得到独立的 50 epoch 完整 Validation 曲线；B_SHARED_ONLY 也独立选点。每 seed 以六格等权 Macro AUC **最早严格最大值**选择自己的 epoch。随后用五 seed 所选 Validation 分数的均值比较统一配置，不能逐 seed 选择不同 H 或 Scope。

B_SPLIT内部配置均值相差不超过 `1e-6` 视为配置选择同分；此容差不改变单条曲线最早严格最大规则。内部同分顺序：S优先all、logit优先probability、H基准值优先短窗口再长窗口。

B_SHARED_ONLY与最佳B_SPLIT跨家族比较时，若二者通过09规定的门且U差不超过`1e-4`，优先单次前向的B_SHARED_ONLY。该成本简化容差与内部HPO同分阈值分别记录。

F 是内部参照，B 的获胜配置是否值得保留、何时进入蒸馏、能否被 C 教师替换，均由 [决策规则](09_EVALUATION_AND_DECISION.md) 计算，不在本文件另设冲突门槛。

除标准指标外，仅保留以下必要机制观测：

- B−F、B_SHARED_ONLY−F、B_SPLIT−B_SHARED_ONLY 的五 seed 配对差。
- 同 H、同聚合、同 epoch 的 S/all 差；不能将两个 independently-selected epoch 的差当作纯 Scope 因果效应。
- 固定 P_A/P_B/V_mech 上 B−F 的 corrected、broken、net 及与 F 的纠正重叠；复用既有 pair 缓存与平局口径。
- 两态 logit 差、B−F logit 差及参数重心数值误差，用于区分动作太小与动作有效但净收益有限。
- 组件 AUC/LL 可作解释，不能因 old 组件单独较差就删去旧历史；两态组合才是待评价对象。

固定面板旧 q 数表用于背景对照，不要求本轮重跑全 step q。新 B 需要完整 Validation 和冻结后 Test，以回答有界近似是否兑现收益。

## 7. 稀疏缓存、伪代码和接口

### 7.1 快照需求

预先为 `t_e=e*S_d, e=1..50` 和全部 H 构建 `origin=max(0,t_e−H)`。保存 `origin → 尚未消费的 target steps`。实际 step 结束、F 更新完成后才 capture；目标消费完后移除引用，无引用即释放。

当前需求日程按“先 capture 后 consume”精确模拟：MovieLens 的历史 F 快照峰值 5 个，KuaiRand 峰值 11 个，包含初始化和同刻新 capture。不能假设 t−h 的旧 RAM 状态仍在，不能释放后再索引。

```text
initialize online theta_0, Adam m/v, F_0=theta_0
build all epoch-end target -> origin dependencies for the three H values
capture F_0 if it has users

for each original natural batch:
    compute original source loss
    backward; original Adam.step()
    F <- rho*F + (1-rho)*online
    step <- step+1
    if step is a registered origin: capture F and its identity in guardian RAM

    if step is an epoch end:
        evaluate O, F and B_SHARED_ONLY on full Validation
        for H in frozen_H_values:
            h <- min(H, step)
            old <- registered_snapshot(step-h)
            recent <- stable_FP64_difference(F, old, rho**h)
            for scope in [S, all]:
                construct exactly two states; attach the same audited buffers
                stream their logits/probabilities on full Validation
                accumulate B_logit and B_probability metrics
                if H == source_main_H and scope == S:
                    evaluate the single registered C family from these logits
                record small train/validation-panel logits and scalar diagnostics
                update earliest-strict-max checkpoint references in guardian RAM
                release full-Validation temporary score arrays
        consume this target's origin references

after all five seeds have complete curves:
    freeze one dataset/condition-wide configuration using the rules above
    retain its per-seed selected states until every dependent task is finished
```

建议新增实验模块中的接口（是待实现规范，不是本轮现有命令）：

```python
HistoryFCache(capture_users, layout_hash, buffer_hash)
HistoryFCache.capture(step, fixed_ema_state)
HistoryFCache.get(origin_step) -> immutable_parameter_vector
HistoryFCache.consume(target_step)

split_means(fixed_t, fixed_origin, rho, h) -> recent, old, numerical_audit
split_states(recent, old, fixed_t, scope_mask, buffers) -> two_states
shared_only_state(fixed_t, online_t, scope_mask, buffers) -> one_state
stream_split_readouts(two_states, weights, bundle, visible_policies) -> metrics
```

其中 `visible_policies` 只允许接收上一 epoch 已冻结的 C 权重，不允许在当前 full Validation 标签上拟合动作。完整曲线流式计算可以避免为 C 再重放一遍父轨迹。

## 8. 计算、RAM 与观察输出预算

### 8.1 成本公式

单个 FP32 参数向量是 \(4P\) bytes。历史环缓存为 ML `5×4P=15,469,400 bytes`（14.75 MiB），KR `11×4P=71,027,176 bytes`（67.74 MiB）。该数字只含历史 F；工作模型、F、Adam、梯度、best 候选和路线 A 分叉状态另计，并按真实存储去重计入 guardian 预算。

理论部署参数量：B_SPLIT-all 为 \(2P\)；B_SPLIT-S 可打包共同非共享参数及两份共享参数，为 \(P+P_S\)。两者仍需要两个完整函数前向，不能因复用参数就按一次推理计。

| 部署参数表示 | ML bytes | KR bytes |
|---|---:|---:|
| B_SHARED_ONLY：P 个 FP32 | 3,093,880 | 6,457,016 |
| B_SPLIT-S：P+P_S 个 FP32 | 5,026,488 | 9,104,696 |
| B_SPLIT-all：2P 个 FP32 | 6,187,760 | 12,914,032 |

每 epoch，B_SPLIT 是 `3 H × 2 Scope × 2 状态 = 12` 次完整 Validation 前向；logit/prob 共用结果。B_SHARED_ONLY 另需 1 次，O/F 两参照另需 2 次，因此该完整读出集合共 **15 次 full-Validation 前向/epoch**。C_PRED的预定义权重直接复用两态logits，不额外前向；C_PARAM的三个控制最多增加5个全batch等效前向，所以完整父期注册集合上限为20次/epoch。Validation 约为 train 的八分之一；这项评估开销必须实测，不能写成零成本观察。

### 8.2 有界分数存储

一个完整 Validation 的双任务 FP32 分数数组占 `8*N_val` bytes：ML800,168 bytes、KR2,006,136 bytes。按当前 H/Scope 顺序计算，指标完成后释放；不跨50 epoch、全部Scope/H累积完整人口 logits。

AUC 需保留当前读出的全人口分数或等价准确排序状态，不能简单平均 batch AUC。可保存的是六格/宏/Traffic/Pooled 标量曲线、固定小面板有界预测、身份哈希和控制决策。

### 8.3 RAM-only 持久性约束

继承仓库 `start.md` 的参数状态限制：模型、optimizer、EMA、历史快照、selected 参数均不落盘。它们只驻 guardian RAM，元数据与预测缓存可按预算保存。

同一 guardian 尚存时，恢复状态包含 online、Adam m/v/step、F、history ring、引用计数、best checkpoint 引用、loader/RNG、global step、配置/代码/数据/分组/buffer 哈希。guardian RAM 丢失时，必须按自动执行文档从已知身份确定性重放所需前缀，并计入预算；不得声称可从磁盘模型 checkpoint 恢复。

每次候选刷新 best 时，除两态/权重之外，须同时保留该epoch真实FP32父F及buffers的RAM引用，供后续学生初始化；不能在选点后用已cast的两态重构父F来替代原状态。可按状态hash去重。selected 教师要保留到其软标签缓存、蒸馏依赖和最终冻结 Test 完成。B/C 模型算法上可部署，不等于本轮获准将权重文件落盘。最终报告保留可重建的配置、来源及选择身份。

## 9. `B_DISTILL`：将有益历史组合迁移到一个模型

### 9.1 教师冻结与训练初值

只在 [决策规则](09_EVALUATION_AND_DECISION.md) 允许时执行。教师默认为冻结 B_SPLIT；C 若满足预注册的相对简单策略增量条件，则可由冻结 C 预测赢家替换。不能由 Codex 临时混合多个赢家。

每个 seed 的教师身份至少包含 dataset、condition、source trajectory、选中 epoch/step τ、两态参数 hash、H、Scope、聚合方式，以及 C 时的权重与训练来源。教师 eval/no_grad，固定不随学生变化。

学生初值统一为**同一教师选中 epoch τ 的父轨迹 F_τ**：

\[
\theta^{\rm student}_0=F_\tau.
\]

不能换成另一个 independently-selected F epoch；所有学生臂以及 F-teacher 控制共享此初值。F-teacher 就是这个 F_τ 的预测，避免额外教师选点差异。

先对完整原 train 人口生成教师的两任务软概率。可以使用 P_A/P_B 以外的原 train 行，且保留这些原 train 行；不能把研究面板改成新的正式训练集。缓存必须以原 `train_row_id` 对齐，附 data/preprocessing/teacher/config hash。Validation/Test 不得进入软标签缓存、学生 loss 或教师权重拟合。

教师选择允许使用规定的 Validation，Test 始终只用于全部方案冻结后的评价。该流程不会为未知 Test 输入缓存历史 q；推理学生是单一参数模型。

### 9.2 归一化训练目标

本轮蒸馏先限 Clean。设当前自然 minibatch 大小 n、任务 k∈{0,1}，教师软概率为 q_ik：

\[
\mathcal L_y=\frac{1}{2n}\sum_{i,k}\operatorname{BCE}(p_{ik},y_{ik}),\qquad
\mathcal L_q=\frac{1}{2n}\sum_{i,k}\operatorname{BCE}(p_{ik},\operatorname{stopgrad}(q_{ik})),
\]

\[
\mathcal L_{\mathrm{KD}}=(1-\lambda)\mathcal L_y+\lambda\mathcal L_q.
\]

硬、软标签均使用源 Clean 相同的 probability BCE 实现与自然样本/两任务均值；一次学生前向同时供两项使用。不要把 Macro AUC 的六格等权口径错误地用于训练重加权。温度固定 T=1，不额外搜温度。

两项权重和为1，teacher=y时退化原监督损失；标签项权重减少是蒸馏操作本身，F-teacher同λ控制其一般软标签效应。不能改成额外加和又忽略由总体损失尺度变化产生的混淆。软概率需 finite 且位于[0,1]，不能按Test统计重新校准。

### 9.3 optimizer、EMA 和严格预算

- fresh Adam：m_0=v_0=0、step=0；不能把父 online optimizer 的历史误接到 F_τ 初值。
- LR 候选 `{0.001,0.003}`；β、ε、耦合 weight decay、foreach/fused/AMP/TF32 等继承源协议。
- λ 候选 `{0.25,0.5}`，所有比较均五 seed。
- 固定5 epoch，无性能早停；ML每臂100次Adam步、KR每臂245次。自然batch与样本集合不变。
- 同 seed 的全部臂使用相同初始化、loader RNG和batch序列；教师cache只读。
- 学生参数 EMA 初始化 `F_student,0=theta_student,0`，每个实际学生 Adam.step 后按该dataset原ρ更新；它的Adam时钟是新训练时钟。
- e0和e1–e5均评估online/F，两种读出分别按最早严格最大Validation Macro AUC选点。e0是正式候选，不能因续训损害而被排除。

### 9.4 七臂矩阵与控制

| 臂 | 教师 | LR | λ | 用途 |
|---|---|---|---|---|
| KD-1 | 冻结B/C教师 | .001 | .25 | 候选 |
| KD-2 | 同上 | .001 | .5 | 候选 |
| KD-3 | 同上 | .003 | .25 | 候选 |
| KD-4 | 同上 | .003 | .5 | 候选 |
| LABEL-1 | 无 | .001 | 0 | 同预算普通继续训练 |
| LABEL-2 | 无 | .003 | 0 | 同预算普通继续训练 |
| F-TEACHER | 父F_τ | 冻结KD赢家LR | 冻结KD赢家λ | 识别新增历史信息与一般自蒸馏 |

先完成前六臂的五 seed，再冻结 KD 的 LR/λ，补齐 F-TEACHER 五 seed；无需为F教师重开全网格。候选选择、对同LR LABEL和最佳LABEL的比较、是否优于F教师及最终输出模式按 [09](09_EVALUATION_AND_DECISION.md) 执行。

每dataset为 `7臂×5seed×5epoch=175epoch`；两个数据集为350epoch。对应新增实际Adam步为ML3500、KR8575，共12075步；教师cache前向、e0/每epoch评估另计，无Adam步。这个预算不能与父轨迹500epoch混为一项。

若教师优于F但学生未超LABEL或F-TEACHER，结论是本轮压缩目标未兑现额外信息；保留合格的两模型读出，暂停增加蒸馏网络/新损失。不能把它写成历史预测组合本身失败。

### 9.5 软标签缓存成本与恢复

每教师两任务FP32软标签为 `8*N_train` bytes：ML6,401,336 bytes（6.10 MiB），KR16,049,056 bytes（15.31 MiB）。B/C教师与F教师各一份，多个学生臂共享cache，不按λ/LR复制。若不复用既有row_id数组，额外ID索引为 `8*N_train` bytes。

软标签属于预测数据，可在观察输出预算内落盘；模型参数仍禁止落盘。cache hash不匹配则拒绝训练，不默默重对齐。学生恢复状态包括其 freshAdam状态、F_student、loader/RNG和教师cache身份；RAM丢失按固定初值与相同批次序列重放，并计入自动执行预算。

```text
freeze exactly one teacher per seed under the already-frozen method
theta0 <- parent fixed EMA at teacher's selected epoch
cache teacher probabilities on ORIGINAL TRAIN ONLY
cache probabilities of the same theta0 for the later F-teacher control

for each pre-registered KD or LABEL arm:
    student <- theta0; Adam <- fresh; student_F <- theta0
    evaluate e0
    for 5 fixed epochs using identical paired batch orders:
        p <- student(original clean batch)
        loss_y <- source_natural_mean_BCE(p, y)
        loss <- loss_y for LABEL
        loss <- (1-lambda)*loss_y + lambda*source_natural_mean_BCE(p,q) for KD
        backward; Adam.step; update student_F once
        evaluate full Validation online and student_F; retain best RAM states

after the four KD and two LABEL five-seed arms complete:
    freeze KD learning rate and lambda by Validation
    run five-seed F-teacher arm with that same pair
    apply the pre-registered decision; freeze final Test readouts
```

## 10. 与已有方法的关系及本轮不扩展的事项

- [SWA，UAI 2018](https://proceedings.mlr.press/r16/izmailov18a.html) 将训练轨迹参数平均与预测集成联系起来。本方案借鉴两种平均的关系；不据此断言本M3oE/Adam/EMA的损失面已更平坦。
- [Temporal Ensembling，ICLR 2017](https://arxiv.org/abs/1610.02242) 用历史预测共识作为训练目标。旧q是固定面板观测，不是已完成的TemporalEnsembling训练；本轮蒸馏才实际把软目标用于学生训练。
- [Mean Teacher，NIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) 使用参数EMA教师；它支持保留F-teacher对照，不能把原来仅用于评估的F读出自动称为MeanTeacher训练。
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531) 提供集成信息压缩到单模型的思路；本推荐场景是否成功由匹配成本的学生对照决定。

这些相似工作不构成否决本路线的理由。当前研究价值在于用已确认的共享历史收益定位具体操作、保留完整旧尾部并量化函数组合的额外价值。

本轮不自动增加K=4/更多历史模型、半衰期新网格、额外增强蒸馏、温度搜索或复杂校准器。唯一Gaussian边界按09：Gaussian父教师/初始化，Clean学生5epoch，沿用冻结λ/LR和三个匹配臂，condition单列`gaussian_d0_parent__clean_student`；不改变下面主Clean算法的损失配方。若两态近似与单模型替代都不能带来满足决策规则的收益，保留固定F；若C没有超过匹配的全局策略，停止复杂分域控制。其后开放问题由最终报告说明，不把停止条件改成继续增加超参数。

Gaussian学生参数映射补充（与09§8一致）：三个学生共用Clean主赢家的student_lr；B/C教师学生与Fteacher学生的非零distill_weight取Clean已冻结KD赢家的λ；LABEL固定λ=0。若Clean主赢家是LABEL，沿用其获胜LR，但不把λ=0复制到两个soft目标臂。teacher family保持Clean已选对象，冻结至GAUSSIAN_TRANSFER_FREEZE，不增加任何候选。
