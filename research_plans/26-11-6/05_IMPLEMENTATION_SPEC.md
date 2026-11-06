# 05｜Codex 工程实现规范

## 1. 实施边界与固定源码

本文件规定 Codex 后续应实现的观测、读出、统计和执行工具。**本次交付只有规划文档与配置；下述 `experiments.ema_mechanism_20261106` 包、类、函数和 CLI 均待 Codex 实现，不是仓库已有功能。** 科学定义以 [03_METRICS_AND_MATH.md](03_METRICS_AND_MATH.md) 为准，实验范围以 [04_EXPERIMENT_MATRIX.md](04_EXPERIMENT_MATRIX.md) 和 `experiment_manifest.json` 为准。

实现基底固定为 `40705dfe0bb4fe614bdca835c0bae567348c87d0`，不能直接在仅含规划的 main 上假定全部依赖已经存在。由 [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md) 建立独立执行 worktree，并导入本目录。源训练文件保持原样；新增代码集中于 `experiments/ema_mechanism_20261106/`。本研究所有新 shadow、probe、参数组读出均无训练反馈，不实现新的自适应 EMA 算法。

### 已核验的真实组件

下表链接均固定到实现 SHA，可用于逐项核对，不能把较早方案中的计划状态当作未实现。

| 文件 / 接口 | 复用内容 | 不得改变的边界 |
|---|---|---|
| [model_adapter.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/model_adapter.py)：`build_model`、`ModelAdapter.forward_logits` | 固定 M3oE 模型、raw logit、完整功能导数选项 | 普通训练继续保留源 gate 输入 detach |
| [registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/registry.py)：`ParameterLayout` | 参数名、shape、offset、别名、dtype 的唯一布局 | `.flatten()` 会 detach；不能用它建立功能导数图 |
| [group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/group_registry.py)：`build_path_groups` | 12 个互斥组，domain embedding 按行拆分 | 不按域规模重命名或推断组行为 |
| [source objectives.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py)：`losses` | 原 clean/Gaussian 标量目标及数值次序 | 分母为 `2*N_actual`；增强替换、非新增全额损失 |
| [source views.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/views.py) | 字段 RMS、Gaussian view 和独立 RNG | 保留原 `GA31-R3` 噪声 key，不加入新 study/候选参数 |
| [data_bridge.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/overnight_ab_20261103/data_bridge.py)：`cached_bundle` | 自然人口 mmap、`include_test=False` | 训练 worker 不持有 Test 数组 |
| [evaluation.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/overnight_ab_20261103/evaluation.py)、[selection.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/overnight_ab_20261103/selection.py) | 完整 Validation、六格 Macro AUC 严格最大选点 | 精确并列保留较早 epoch；AUC 与 LogLoss 分开 |
| [state.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/overnight_ab_20261103/state.py)、[guardian.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/guardian.py) | CPU RAM 深拷贝、hash、epoch 事务、worker 所有权 | 不将参数或优化器状态写磁盘 |

旧 `aug_history_20261105.protocol.resolve` 与 `worker` 不能直接承担本研究：它们校验固定 D0 Gaussian、lambda=.5、15 个候选，并绑定 A/B/C 逻辑。新实现应引用上述纯组件，独立组织 worker；不修改旧接口来容纳新研究。

## 2. 新增文件、类与接口合同

只建立下列必要模块；可以在同文件内实现私有辅助函数，不另建通用实验框架。

| 新文件 | 必须提供的接口 | 输入与产物 |
|---|---|---|
| `protocol.py` | `resolve_study(manifest, paths)`；`audit_sources()`；`verify_data_identity()` | 解析冻结设置、源文件与数据 hash；输出 `ResolvedStudy`、`PREPARED.json` |
| `panels.py` | `build_panels(bundle, seed, spec)`；`load_panel_identity()` | 仅 train/validation；输出 P_A、P_B、V_mech、固定 pair 索引及 hash |
| `shadows.py` | `FixedShadowBank`；`PulseWindow` | 常驻标准 EMA、1 epoch 临时 fast/slow 脉冲；没有可学习控制器 |
| `readouts.py` | `read_predictions()`；`group_hybrid()`；`evaluate_full_validation()` | 单份 scratch 顺序读出；返回原概率和 raw logits、指标及有限观测 |
| `observers.py` | `observe_innovation()`；`observe_pulse()`；`rank_direction_scores()` | r 的分组即时归约、M0–M3；只输出标量与白名单预测 |
| `worker.py` | `create_session()`；`train_batch()`；`advance_epoch()`；`export_session()`；`restore_session()` | 保持源训练语义，输出完整 epoch 事务 |
| `state.py` | `MechanismGuardian`；`commit_epoch()`；`PredictionWriter` | RAM 状态、有界预测 NPZ、幂等事务与资源计数 |
| `analysis.py` | `analyze_stage(stage)`；`fit_predictive_folds()`；`run_controls()` | 按 06 的规则生成配对表、跨 seed/时间预测、有限控制结论 |
| `evaluation.py` | `freeze_endpoints()`；`evaluate_frozen()` | 全部五 seed 完成后冻结；复用同状态历史 Test 或评价新冻结端点 |
| `run.py` | `main(argv=None)` | 第 9 节定义的唯一 CLI 合同 |
| `tests/` | 工程单测与 smoke 辅助 | 第 10 节列出的必要正确性检查 |

### 关键 Python 签名

```python
def create_session(job: dict, bundle, device: str, panels) -> dict: ...
def train_batch(session: dict, x, y, row_ids) -> dict: ...
def advance_epoch(session: dict) -> dict: ...
def export_session(session: dict) -> dict: ...  # 只用于 RAM IPC
def restore_session(job, bundle, device, panels, snapshot: dict) -> dict: ...

class FixedShadowBank:
    def __init__(self, theta0, rho_by_physical_id: dict, aliases: dict): ...
    def after_adam(self, theta_after, step: int) -> None: ...
    def get(self, readout_id: str): ...  # resolves alias; returns physical vector
    def state_dict(self) -> dict: ...

class PulseWindow:
    def begin(self, phi_f, end_epoch: int, start_step: int) -> None: ...
    def after_adam(self, theta_after, step: int) -> None: ...
    def finish(self, expected_end_step: int) -> dict: ...

def read_predictions(scratch, x, *, params=None,
                     full_functional_graph=False) -> "PredictionBatch": ...
def group_hybrid(base, changed, group_index, group_id: int, out) -> None: ...
def rank_direction_scores(scratch, phi_f, psi_fast, psi_slow,
                          panel_a, layout, groups) -> list[dict]: ...
```

所有新科学汇总以五 seed 为单位。工程 smoke 可用 seed2022，结果显式标记 `ENGINEERING_ONLY`，不能进入科学表。

## 3. 张量、参数组与状态定义

### 输入输出形状

| 对象 | shape / dtype | 说明 |
|---|---|---|
| 训练输入 `x` | `[F,N]`，源特征 dtype | batch 转置后进入模型；最后一行是 domain |
| 标签 `y` | `[N,2]`，源 float32 | 两任务标签不改写 |
| 模型概率、raw logit | `[N,2]`，float32 | 不是 `[N,6]`；内部按 `3*t+d` 路由汇合 |
| theta / phi / psi | `[P]`，float32 | ML P=773470；KR P=1614254 |
| 参数组编号 | `[P]`，int64 | 12 组完整互斥分区，源 group hash 必须一致 |
| 排名 panel 每 cell | positive256、negative256 | 固定行 ID；每 cell 4096 个固定正负 pair |
| 排名存储矩阵 | `[6,512]` | 每 cell 只存该任务输出，不重复另一任务 |
| natural train panel | `[3,512,2]` | 每域512自然抽样行，两任务 |
| V_mech | `[3,4096,2]` | 每域4096 Validation 自然抽样行，两任务 |
| 排名 pair index | `[6,4096,2]`，int32 | 索引对应本 cell 的正、负样本位置 |

12 组顺序固定为 `S,D0,D1,D2,T0,T1,DT00,DT01,DT10,DT11,DT20,DT21`。参数名到组的映射直接继承源码；domain embedding 按真实 lookup 行拆开。`gate/tower[3*t+d]` 归 DTdt。全部其余 sparse embedding 归 S，不新增基于样本共现的 embedding 行聚类。

冻结函数图支持 `support[b,d,t]`：每个cell只对 `S,Dd,Tt,DTdt` 为真，共24个可达 `(cell,group)`；其余48个为结构性不可达。完整功能导数恢复gate输入路径后，仍遵守同一支持。主指标比较和预测评分使用这24行；完整72格作为附表保留。**不能按实际测得的收益是否为0筛选行。** support mask及其源图证据hash写入 `GROUP_MAP.json` 和每条指标记录。

### 常驻与临时状态

`session` 至少包含：

```text
job / source_identity / data_identity / observer_identity
model / optimizer / scratch / layout / groups
shadow_bank.phi / shadow_bank.rho / shadow_bank.aliases / shadow_bank.updates
pulse.{active,end_epoch,start_step,last_step,fast,slow}
panels / loader_generator_state / python_numpy_torch_cuda_rng
epoch / global_step / commit_index / selection_trackers
snapshots[physical_readout_id][selected|E10|E50]
retained_pulse_states[E20|E40][F_end|fast_end|slow_end]
observer_scalar_accumulators / cost / prediction_receipts
```

模型、Adam 的一二阶状态、所有参数影子及选中 checkpoint 只驻 RAM。scratch 不加入 optimizer，且不写回 online。旧 phi、innovation、group hybrid 等临时向量用完立即释放；**不保存逐步全参数/全梯度历史，不新增 theta_prev 或 delta_prev 的长期缓存**。每步创新量仅即时按组归约。

常驻半衰期为1、2、4 epochs，另保留源 F。`rho(H)=2**(-1/(H*steps_per_epoch))`。MovieLens H2 与 F 为同一物理影子，以源 F 的精确rho为规范；物理ID固定为H2，别名方向为F→H2，`bank.get("F")`先解析alias。`after_adam`只遍历物理ID，不能再次更新alias。KR 的 F 半衰期约2.82209 epochs，另占一份。因此物理 EMA 数分别为3、4；selected/E10/E50 与性能候选计数不得重复 alias。

## 4. 训练与观测的精确顺序

### 固定真实训练语义

- seeds `[2022,2023,2024,2025,2026]`；50 epochs；batch40960；自然人口与原始split；无 early stop。
- 标准 `torch.optim.Adam`：lr=.01、betas=(.9,.999)、eps=1e-8、coupled weight_decay ML1e-4/KR1e-5；foreach=False、fused=False、amsgrad=False；无 AMP、TF32、梯度裁剪。
- Gaussian D0：原 source `losses`，target=0、`num_views=1, view_family='gaussian', pairing='iid'`、lambda=.5、kappa ML.05/KR.1。
- Clean：同一 source `losses`，`num_views=0`，保留源码 `real.mean(1).mean()` 的求值次序；不得为了统一表达式改写浮点归约顺序。
- Gaussian 尺度为字段 embedding 的 detached RMS；只扰动原非routing sparse字段；两个任务共用原 view；不改变标签和损失分母。
- 原噪声 key 为 `GA31-R3|dataset|seed|train|step|gaussian|1`。新增观测绝不修改这个 key。
- 初始化调用源 `history.setup(seed,device)`；数据顺序使用独立 CPU generator `seed+11000`，每 epoch 一次原 `randperm`。

### 单步伪代码

```python
# 对下一个真实 step k；所有观测都无反馈。
optimizer.zero_grad(set_to_none=True)
loss, info = source_losses(model, x, y, target=0, config=source_view_config,
                          dataset=dataset, seed=seed, step=k,
                          mix_lambda=0.5, kappa=frozen_kappa)
assert finite(loss)
loss.backward()
optimizer.step()                         # 本 batch 唯一真实优化器更新
theta = layout.flatten(model.parameters())

# F尚未更新，此时才是规定的 r_k。
r = theta - bank.get('F')
innovation_rows = reduce_rms_by_group(r, groups)
del r                                   # 只保留12组标量

bank.after_adam(theta, k)                # 所有常驻标准EMA各更新一次
if pulse.active:
    pulse.after_adam(theta, k)           # 两份影子；不影响bank/model/optimizer
session.global_step = k
```

`observe_innovation` 不调用 backward，不清空/改写真实 `.grad`，不修改 Adam 一二阶状态。M0 的终点主量按 03 使用 `gap=theta-F`、`delta=psi_fast-psi_slow` 的分组 RMS 与 cosine；零范数写 `null` 和原因码，不填造0方向。

### 每 epoch 顺序

1. 完成该 epoch 的全部自然训练 batch；记录实际尾 batch 和全局 step。
2. 在完整 Validation 上依次评价 online 和全部物理常驻 EMA，逐个调用源选点规则；保存 selected 与 E10/E50 RAM 状态。原批内顺序、全量 Validation 行顺序不变。
3. 从本次完整 Validation 前向中抽取 V_mech 行输出；不额外跑一轮相同 regular V_mech 前向。
4. online/F 在 P_A、P_B 上读出。若为 pulse 终点，F 的这些输出与 V_mech 输出直接复用；其余26种 pulse 读出顺序计算。
5. pulse 终点计算 M0–M3，并在 E20/E40 深拷贝且仅保留 `[F_end,fast_end,slow_end]` 到 guardian。其余终点即时归约后释放 pulse 全参数。
6. 若本 epoch 是下一个 pulse 的起点 e−1，则在 F 已完成本 epoch 更新后复制下一窗口的两份状态。
7. 恢复观测前训练 RNG；生成一个完整 epoch 事务；先提交 RAM 状态，再原子提交标量/预测索引；完成后递增 commit_index。

模型没有 BatchNorm/Dropout，使用 LayerNorm；不增加 BN 重新校准或 dropout 处理实验。仍须验证 observer 开关不改变 online/Adam/RNG。

## 5. Pulse、反事实读出与 M2 实现

### 时间定义

对终点 e∈{5,10,…,50}，`k0=(e-1)*S`，`k1=e*S`。两份 pulse 在 k0 的 F 更新完成后复制，随后恰好处理 `k0+1,...,k1`：

```python
pulse.fast = F_at_k0.clone()
pulse.slow = F_at_k0.clone()
rho_fast = rho_F ** 2
rho_slow = math.sqrt(rho_F)
for k in range(k0 + 1, k1 + 1):
    pulse.fast.mul_(rho_fast).add_(theta_k, alpha=1-rho_fast)
    pulse.slow.mul_(rho_slow).add_(theta_k, alpha=1-rho_slow)
```

每个终点共27个研究读出：F；12个只替换一个组的 fast；12个只替换一个组的 slow；全 fast；全 slow。对于组 b，`hybrid=F+P_b*(psi-F)`。实现用 `out.copy_(F)` 后仅复制该组范围，避免数值上多做一次减加改变组内参数；公式只用于定义。

该读出与“上一 epoch 仅组 b 改用对应 rho、其余组继续 F”的独立 shadow 数值等价。单元测试必须验证。它只定义固定轨迹上的局部历史读出效应；多个组的收益不可直接相加。`all-fast/all-slow` 用于检查组合效应，不作为训练方法。

### M2：完整功能梯度优先，源 detach 代理并列记录

在 F 上对 P_A 的每个 cell 计算固定4096 pair 的 softplus 排名损失，输出三种正向分数：`-g_c^T P_b*(psi_fast-F)`、`-g_c^T P_b*(psi_slow-F)`、`-g_c^T P_b*(psi_fast-psi_slow)`。只保存每组标量、梯度组范数、导数语义和有效样本数，不写 P 维梯度。

```python
flat = F.detach().clone().requires_grad_(True)
params = {name: flat[left:right].view(shape)
          for name, shape, (left, right) in layout_entries}
z = scratch.forward_logits(panel_A.x, params=params,
                           probe_input_full_graph=True)
losses_c = fixed_pairwise_softplus(z, panel_A.pairs)  # [6]
for c in range(6):
    g = autograd.grad(losses_c[c], flat,
                      retain_graph=(c < 5), create_graph=False)[0]
    write_scalar_group_dots(g, psi_fast-F, contrast='fast_minus_F',
                           derivative_semantics='full_functional')
    write_scalar_group_dots(g, psi_slow-F, contrast='slow_minus_F',
                           derivative_semantics='full_functional')
    write_scalar_group_dots(g, psi_fast-psi_slow, contrast='fast_minus_slow',
                           derivative_semantics='full_functional')
    del g
del flat, params, z
```

随后使用完全相同的 panel、pairs、F 和归约尺度，设置 `probe_input_full_graph=False` 重算一遍，标记 `derivative_semantics='source_detach'`。这是固定代理对照，不新增可调超参数，也不修改旧 A 控制器。二者前向 raw logits 应相同，导数允许不同。每次调用后移除 hook；观测只能作用于 scratch/functional 参数。

源默认 autograd 含 gate 输入 detach，因此不能把 source_detach 的 `g·delta` 直接写成完整模型函数的一阶 Taylor 导数。完整功能导数须与有限差分在工程测试中校验。

## 6. Panel 与预测保存协议

### 固定 panel 构建

1. P_A/P_B 只来自原训练人口。以 `SHA256(EMA_MECHANISM_20261106|train_partition|dataset|seed|original_row_id)` 最低位将源训练行分到两个不交池，再以role key采样；保证两 panel 涉及的原始 row ID 集合不相交。样本仍保留在原 online 训练中。
2. 每 panel 每 cell 均匀抽256正例和256负例，用于排名/AUC探针；每域另均匀抽512自然行，用于自然分布 LogLoss/预测波动。允许同 panel 不同任务复用同一原始行，但记录角色索引，不能把角色重复视为独立样本。
3. 每 cell 从256×256正负组合中用独立 generator 无放回固定选4096 pairs；全研究期间不重抽。
4. V_mech 在完整 Validation 每域均匀抽4096自然行，两任务使用相同行。不得从 Test 抽 panel；不得按观察到的收益筛选行。
5. namespace 固定 `EMA_MECHANISM_20261106|probe|dataset|seed|panel|role`，通过 SHA256 派生独立 generator；不触动训练 RNG。P_A/P_B/V_mech 与 pair 的内容 hash 在 preflight 冻结。
6. preflight 核验上述配额可满足。若源人口不满足，记录 `PANEL_POPULATION_MISMATCH` 并停止受影响工程 job；不替换数据集、不推断域难度、不以此作科学否决。

V_mech另外按 `SHA256(dataset|row_id)` 最低位分成 V_A/V_B。此划分不包含seed，同一原始行在所有seed中始终属于同一侧；保存 `validation_row_side`、分割规则版本和行集合hash。该二分仅用于机制指标与偏好交叉验证，不改变官方完整Validation checkpoint选点。若某子集缺少某cell标签类别则报告其不可用，不以伪值填补。

V_mech在preflight另存每格4096个均匀无放回正负pair索引，用于03第7.3节的online/F排序时间变化率；不足用全部可用pair，缺类为NA。仅增加索引元数据，不新增forward或Validation梯度。P_A/P_B复用既有pair。

### 同次 forward 得到原概率和 raw logits

新增 `read_predictions` 用临时 tower hook 或等价只读捕获，在一次原 forward 中收集：原模型最终输出 `p:[N,2]` 及按真实 domain 路由后的 raw `z:[N,2]`。**官方指标使用原 `p`，不能用 CPU 再 sigmoid(z) 代替。** 捕获过程须在 finally 中移除全部 hook，并恢复 train/eval 状态。

官方 AUC 对源 FP32 概率计算，保留 sigmoid 饱和造成的 ties。LogLoss 沿用源 float64 `max(log(p),-100)` 与 `max(log1p(-p),-100)`；不替换为 sklearn 默认 clamp 或另一个 logit BCE。研究排名 softplus 使用 raw z，指标来源分别标记。

### 白名单预测 NPZ

允许使用 **prediction-only NPZ** 保存本次明确要求的观测；这是对预测文件格式的限定，不改变 RAM-only 模型状态约束。写入与读取均禁用 object/pickle；读取 `allow_pickle=False`，文件采用临时文件+原子rename。字段白名单为 `p`、`z`、只读 row/pair/role 索引、标签、cell/readout索引及短hash元数据，shape 必须符合本节。禁止参数名、模型 state_dict、Adam状态、P维梯度、Q/参数投影等内容。

| 观测 | 保存内容 | 避免重复 |
|---|---|---|
| 每 epoch regular V_mech：online/F | `[3,4096,2]` 的 z+p | F/H2 alias只存一次 |
| 每 epoch regular V_mech：其余H | `[3,4096,2]` 的 p | 不存对应 z |
| 每 epoch online/F：P_A/P_B | 排名 `[6,512]` 的对应任务 z+p；natural `[3,512,2]` 的 z+p | 同 row多角色可索引复用 |
| 每 anchor 的26个新增pulse读出 | P_A/P_B同上；V_mech只存 p | F已经由regular输出提供，不再存一份 |
| panel身份 | row IDs、labels、pair indices | 每trajectory/panel只保存一次 |

所有 M0–M3、六格绝对指标、相对F和fast−slow配对量在采集时生成标量。paired ranking flips 以源 p 的比较及 tie规则计算。完整 Validation/Test 预测不落盘。预测与日志合计硬上限2 GiB，具体预算见07；接近上限时先阻止下一次原子写入并记录资源失败，不静默删观测或改panel规模。

## 7. 结构化数据与自动分析接口

### 身份层

`online_trajectory_id` 由数值训练源、数据身份、dataset、objective、seed、初始化/顺序/噪声语义、Adam设置计算；不含 rho候选、panel 或 observer参数。`job_id` 另包含 observer配置与实现hash。旧归档的trajectory ID生成规则可能不同，因此与旧记录的等价性通过数据身份、init/order/model hash证明，不能要求不同研究的字符串ID碰巧相同。

每个表或NPZ receipt须能无歧义追溯：

```text
schema_version, study_id, source_commit, source_files_hash,
implementation_commit, implementation_hash, manifest_hash,
data_hash, split_row_hash, parameter_layout_hash, group_map_hash,
dataset, objective, seed, online_trajectory_id, job_id,
epoch, global_step, readout_id, physical_readout_id, state_hash,
panel_id, panel_hash, pair_hash, group_id, domain, task,
functional_support, validation_row_side, validation_partition_hash,
rho, half_life_steps, half_life_epochs,
pulse_start_step, pulse_end_step, derivative_semantics,
metric_name, metric_value, actual_count, missing_reason,
selection_status, test_exposure, commit_index, observation_hash
```

不适用字段填null并保留schema，不能混用F与alias或selected与固定epoch。不必每行重复长元数据；可以存immutable identity表并用identity_id引用，但解析后的记录必须具备上述字段。

### 必须生成的产物

| 产物 | 内容 |
|---|---|
| `AUDIT_INVENTORY.json` | 既有结果、可复用字段、缺失参数状态、所需观测重放理由 |
| `SOURCE_RECEIPT.json`、`DATA_MANIFEST.json`、`RESOLVED_PROTOCOL.json`、`PROBE_MANIFEST.json` | 冻结身份与准确数值 |
| `PREPARED.json`、`RUN_MANIFEST.json` | 20个逻辑job、源/数据/观测/资源绑定 |
| `VALIDATION_CURVES.csv` | 每 epoch、每物理常驻读出的全部源指标 |
| `INNOVATION_STATS.csv` | 每步即时归约，持久保存每epoch均值/样本SD/最大值及anchor最后一步瞬时量；无逐step序列、无P维序列 |
| `PULSE_READOUTS.csv` | 各anchor、panel、27读出的六格绝对/配对指标 |
| `PULSE_METRICS.csv` | M0–M3、full/source_detach两种M2及其组梯度RMS、全组P_A宏收益、M1/M3的A/B panel一致性 |
| `EVIDENCE_REUSE.csv`、`ARCHIVED_FACTORIAL.csv` | S0来源身份、复用决定和归档四格重算 |
| `HISTORY_PREFERENCES.csv`、`SELECTION_DIAGNOSTICS.csv` | 时间尺度样本内/交叉评价空间及选点/延迟比较 |
| `RANK_FLIPS.csv`、`TEMPORAL_RANK_STABILITY.csv`、`GROUP_INTERACTIONS.csv` | 从原预测/读出派生的排序转换、相邻epoch固定pair变化率及组非加性，不重复保存参数 |
| `PREDICTION_SUMMARY.csv`、`CONTROL_RESULTS.csv`、`MECHANISM_DECISIONS.json` | 逐seed预测比较、条件控制、最终分层研究决定 |
| `PREDICTIVE_ROWS.csv`、`PREDICTIVE_OOF.csv` | 按06固定时间与seed折划分的特征、未来label、折外预测 |
| `PAIRED_SEED.csv`、`PAIRED_SUMMARY.csv` | 五seed配对效应、样本SD、方向及区间；不增加伪seed |
| `STAGE1_DECISION.json`…`STAGE4_DECISION.json` | 有限阶段决策、保留/修订/停止及具体证据 |
| `SELECTION_FROZEN.json`、`FREEZE_RECEIPT.json` | 冻结端点、比较规则、checkpoint hash、五seed覆盖 |
| `ENDPOINTS.csv` | selected/E10/E50 的Validation/Test绝对指标及reuse来源 |
| `COSTS.csv`、`REPLAYS.jsonl`、`epoch_commit_ledger.jsonl` | 逻辑预算、实际开销、物理重放、幂等事务 |
| `FINAL_REPORT.md`、`figures/` | 按01–08研究问题生成报告和标准静态图 |

阶段3预测实现必须严格遵守：e的特征预测e+5的**新pulse偏好**；训练origin=[5,10,15]，最终评分origin=[25,30,35,40,45]。外层leave-one-seed-out，加时间向前及Validation行交叉拟合：其他四seed早期 V_A 标签拟合，留出seed晚期 V_B 评价，然后交换A/B；每seed先平均两行折，再汇总五seed。标准化、类别编码、ridge拟合都在训练折内完成，固定ridge=1，不新增HPO。默认每数据集独立报告，主评分使用24条函数图可达行。精确ridge目标与八类特征集合以06为准；C_M01与C_simple必须实现，不能把M2/M3加入新信息后的优势与单一弱参照比较。P_B不额外求M2梯度；对已算P_A梯度即时归约组RMS。

回归目标为 `[N,2]` 的未来 `[fast−F,slow−F]`，主偏好差预测为两列之差。离线选择诊断在 `{0,预测fast−F,预测slow−F}` 中取最大，精确并列先F、再fast、再slow；不能强迫在两种有损读出中选一个。另报告不拟合的M2/M3直接分数选择，使用同样的F选项。所有选择仅作研究诊断，绝不反馈训练。阶段1时间尺度偏好也采用leave-one-seed-out与相同V_A/V_B行交叉验证；官方完整Validation选点协议保持独立。同一轨迹的epochs/cells/readouts/两个行折不能当额外seed。

S4第二背景只保存派生标量与混合参数状态hash，不额外持久保存该控制的原始预测；临时预测计算后释放，以保持07的磁盘上限。

条件控制仅按 `STAGE3_DECISION.json`：温度诊断读取已有预测；第二背景只用E20/E40保留三向量，最多两个已冻结组。不得为这些控制新训练，也不进入Test调参。具体触发与组选择规则沿用04/06。

## 8. 选点、冻结与 Test 入口

每个物理 regular readout 和 online 各自用完整Validation六格 Macro AUC选checkpoint；精确并列取最早。保存selected/E10/E50，不能用cell最佳epoch替换正式macro选点。

`freeze` 必须检查：全部20个逻辑trajectory完整、每个dataset×objective具有五seed、1..50epoch完整、候选列表未变、所有selected/E10/E50状态hash可用、阶段性配置与比较规则已冻结、分析代码hash已记录。随后生成两个相互绑定的文件：

- `SELECTION_FROZEN.json`：所有合法Test端点及其dataset/objective/seed/readout/epoch/state_hash。
- `FREEZE_RECEIPT.json`：上述文件hash、源/数据/manifest/analysis hash、五seed覆盖、冻结时间及历史Test暴露声明。

`evaluate` 在单独进程中首先验证两个文件和当前状态一致，之后才加载Test数组。正式Test仅包含online和常驻H/F的selected/E10/E50；局部pulse、逐cell多模型选择、条件第二背景和温度诊断不扩大到Test比较。

归档Test满足同数据/顺序、指标函数、state_hash和seed身份时直接复用，写明源表路径、源commit和原行hash；否则只评价已冻结端点。按state_hash去重selected=E10等重复状态。历史复用结果保留 `test_exposure=archived`，新评价也不伪称数据集从未被历史研究使用。

## 9. 新 CLI 合同与退出码

以下全部为**待实现的新 CLI**。在实现和 `--help` 验证完成前不得启动。解释器使用源项目环境，不安装替换环境、不使用sudo。

所有子命令均要求 `--manifest <file> --out <directory>`：

| 子命令 | 额外参数 | 功能 |
|---|---|---|
| `audit` | 无 | 读取归档证据与源码，输出inventory；不训练 |
| `preflight` | `--cache-root <dir>`；`--ml-csv <file>`；`--kr-csv <file>` | 后二默认manifest核验路径；冻结数据、panel、job身份 |
| `validate` | 无 | 必要工程单测，输出精确source/implementation hash绑定的receipt |
| `smoke` | `--dataset {movielens,kuairand} --device cuda:0` | 该dataset的clean/G-D0原batch、尾批、pulse和观测纯度工程检查 |
| `train` | `--devices 0 1 --resume` | 仅调度manifest全部20job；最多两个并发worker；resume幂等 |
| `analyze` | `--stage {1,2,3,4}` | 只读取归档/Validation/probe，不能加载新Test |
| `controls` | `--decision <out>/STAGE3_DECISION.json` | 执行预定义有限控制，或输出NOT_TRIGGERED |
| `freeze` | 无 | 所有五seed和配置冻结，生成两个Test gate文件 |
| `evaluate` | 无 | 从out读取并核验冻结文件；复用或评价合法Test端点 |
| `report` | 无 | 生成最终统计表、图、研究结论及预算报告 |
| `release` | 无 | Test与报告完成后释放本任务guardian；不触碰他人进程 |

`--cache-root` 未提供时，先检查已读取历史 `DATA_MANIFEST`/`RUN_MANIFEST` 中真实记录的路径；不猜测迁移路径。若没有可用 cache，但原 CSV 的 SHA 匹配，则用源 `prepare_cache(root,p)` 在 manifest 指定的新数据缓存目录 `artifacts/ema_mechanism_20261106/data_cache` 重建；显式给出的不存在目录也可作为新 cache 目标。重建前检查磁盘，随后核对原数组、row IDs/fingerprint、split、encoder/scaler hash；全部一致才使用。已有文件身份冲突时停止，不能静默覆盖。数据缓存单列资源，不占2GiB观察输出额度。原构建器可生成封存的 Test 数组，但训练/分析始终 `include_test=False`，不读取 Test 指标。smoke/train只读取preflight已冻结的路径和数据身份，不再接受cache/CSV路径参数。

示例格式（实现后）：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run preflight --manifest research_plans/26-11-6/experiment_manifest.json --out artifacts/ema_mechanism_20261106/attempt1 --cache-root "$EMA_MECH_CACHE"
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run validate --manifest research_plans/26-11-6/experiment_manifest.json --out artifacts/ema_mechanism_20261106/attempt1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run smoke --manifest research_plans/26-11-6/experiment_manifest.json --out artifacts/ema_mechanism_20261106/attempt1 --dataset movielens --device cuda:0
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run smoke --manifest research_plans/26-11-6/experiment_manifest.json --out artifacts/ema_mechanism_20261106/attempt1 --dataset kuairand --device cuda:0
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.ema_mechanism_20261106.run train --manifest research_plans/26-11-6/experiment_manifest.json --out artifacts/ema_mechanism_20261106/attempt1 --devices 0 1 --resume
```

`EMA_MECH_CACHE` 须设置为明确存在的待核验cache路径，或专用于本任务重建的新目录；其身份由preflight核验后冻结，不是预设服务器路径。单个GPU可使用 `--devices 0`，不改变科学预算。

退出码统一：0成功/合法无控制；2配置、来源或hash冲突；3工程数值失败；4资源不足；5五seed/依赖未完成。2–5均不能自动转换为科研方向失败。

## 10. 必要工程验证与 GPU smoke

以下测试对应会改变结论的具体风险，不扩展为重新验证历史方法性能。

1. **源语义等价**：clean与G-D0分别比较原source loss、梯度、一次Adam更新及F；尾批分母为实际2N；Gaussian字段、RMS detach和原随机key不变。
2. **观测纯度**：observer on/off在线参数、Adam、loader、Python/NumPy/CPU/GPU RNG一致；scratch前向/梯度不能修改online `.grad`；新shadow数量和panel规模不改变online hash。
3. **EMA时钟**：theta0初始化、after-Adam次序、源F逐坐标等价、半衰期换算、ML H2/F alias、零改动与常量参数退化。
4. **pulse等价**：窗口恰好S步；两全向量组合结果等于独立逐组shadow；未激活窗口无更新；E20/E40保留状态不被后续原位修改。
5. **组与函数导数**：完整互斥坐标；domain embedding按行；full功能导数与有限差分；full/source_detach前向相等；hook清理；M2符号与M3方向单元例子。
6. **指标准确**：官方概率AUC含ties与源LogLoss log-floor一致；排名softplus独立标记；零范数/缺类返回missing；同一state差值为0。
7. **采样与前瞻划分**：P_A/P_B行集合不交；train probe无Validation/Test行；V_mech只来自Validation；V_A/V_B行角色跨seed不变；固定pair复现；训练折没有未来label、held-out seed或另一Validation行侧；标准化无泄漏；24/72支持来自冻结图而非观测零值。
8. **冻结与保存**：不足五seed不能freeze；篡改freeze文件或数据hash拒绝Test；prediction NPZ禁object/参数字段与非法shape；2GiB/4GiB硬上限；参数只通过RAM IPC。
9. **恢复与幂等**：连续运行与从RAM边界恢复一致；重复epoch/prediction事务不重复计数；冲突hash拒绝覆盖；丢失RAM重放只记物理成本、不增加seed。

GPU smoke使用seed2022工程标记，覆盖两个dataset的真实batch40960、实际尾批，以及一个长度S的pulse窗口和一次完整观测。为控制成本，可在该窗口重复使用固定真实batch作为工程输入；**其结果不进入科学曲线**。对照worker采用完全相同工程输入序列。测量训练、完整Validation、完整功能梯度和27读出各自峰值GPU/RSS/耗时；不能用缩小batch替代正式资源准入。

## 11. 恢复、所有权与执行停止

epoch事务的唯一键为 `(job_id,epoch,commit_index)`；观测子键增加panel/readout/group/metric。先把可恢复状态提交RAM guardian，再原子发布NPZ及标量索引；只有全部成功才能发布完成receipt。标量发布失败时回滚RAM事务。重复一致内容只返回ALREADY_COMMITTED；冲突必须停止并保留两份hash证据，不覆盖旧结果。

恢复状态必须包含online、Adam、所有常驻EMA、活跃pulse、loader/RNG、选点状态、保留anchor和观测事务位置。参数RAM丢失则从theta0重放同一逻辑job；逐epoch核验init/order/model hash，新增物理开销写 `REPLAYS.jsonl`。每逻辑trajectory最多一次额外物理重放；再次丢失则记录工程阻塞，不能无限重试。

并发上限2，不抢占其他进程；只对本任务登记且PID/启动身份匹配的worker/guardian发停止信号。不使用sudo。资源不足可以降低本任务并发，不能减batch、减seed、删科学候选或缩短训练来伪装完成。

全部五seed统计、条件控制、冻结Test和最终报告完成后，`release` 自动释放本任务guardian及临时参数状态；先确认输出hash和磁盘观测齐全。此时阶段性研究完成，不因仍有一般性开放问题继续添加训练。
