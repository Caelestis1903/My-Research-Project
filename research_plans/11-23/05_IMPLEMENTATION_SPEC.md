# 05｜Implementation Specification

## 1. 新代码包与边界

新增包：
`experiments/mdmt_shared_private_coordination_20261123/`

建议模块：
- `protocol.py`：数据、seed、batch、optimizer、EMA、指标常量。
- `source_adapter.py`：canonical M3oE fixed1 载入、输出适配。
- `parameter_groups.py`：U/P_CELL/P_AXIS/P_FULL 参数 manifest 与梯度支持审计。
- `optimizer_blocks.py`：原生 Adam state-preserving 的参数子集更新。
- `gradient_accumulation.py`：J_ACC 梯度累计，按样本数精确加权。
- `fork_state.py`：RAM clone、状态 hash、RNG/loader 保存恢复。
- `diagnostic.py`：S1 anchors、AB/BA forks、V_A/V_B/pair panels。
- `trainer.py`：J_ORIG/J_ACC/P→U/U→P 训练循环。
- `evaluation.py`：Validation 选点、冻结、Test 一次性评价。
- `statistics.py`：paired seed、CI、cell guard、Macro/Traffic/Pooled。
- `report.py`：完整 CSV/JSON/Markdown 报告。
- `cli.py`：prepare/audit/check/run/analyze/freeze/evaluate/report。

不得修改 `m3oe/MDMTRec/models/multi_domain/model.py` 或既有 canonical adapter；新 adapter 只能调用源实现并保持字节级身份。

## 2. 参数分组实现合同

构造 `PARAMETER_MANIFEST.json`，每个 named parameter 记录：
- name、shape、numel、dtype、requires_grad；
- group_id（U/P_CELL/P_AXIS/FROZEN/OTHER）；
- cell/domain/task support；
- gradient nonzero support；
- forward dependency support；
- optimizer membership；
- fixed1 是否允许训练；
- source tensor hash。

### U
初始包含 embedding、STAR shared weight/bias、slot weight/bias、skip_conn、star_mlp、共享 expert 列表 `expert`。slot 参数虽然按 domain 索引，但其输出供该域两个任务共同消费，因此属于上游域条件共享表示，而非 cell-specific head。

### P_CELL
`gate[3*t+d]` 与 `tower[3*t+d]`。索引顺序来自源代码，不得误用 domain-major 的六格报表顺序。

### P_AXIS
`domain_expert[d]` 与 `task_expert[t]`。前者跨同域两任务共享，后者跨同任务三域共享。

### P_FULL
P_CELL ∪ P_AXIS。

### 组检查
- U/P_CELL/P_AXIS 必须互斥且完备覆盖全部可训练参数。
- 若 fixed1 有训练参数不在上述集合，先独立审计并暂停。
- `_weight_exp_d/_weight_exp_t/_weight_bal_d/_weight_bal_t` 的 fixed1 状态必须从 canonical constructor 确认；不得按名字硬塞进某组。
- 用单 cell 自然 batch 分别反传，验证支持范围与预期相符。
- 记录 gate detach 对梯度支持的影响。

## 3. 原生 Adam 子集更新

不能创建六个独立 optimizer 来近似源优化器。保留一个原生 Adam state dict，参数对象与 param groups 不变。

每个子步：
1. model.zero_grad(set_to_none=True)；
2. 对当前自然 batch 计算完整两任务自然 BCE；
3. 只保留本次目标 group 的梯度，其他参数 grad=None；
4. 执行原生 optimizer.step()；
5. 核验仅指定参数发生变化；其余参数、optimizer state 和 step counter 完全不变；
6. 记录实际 Adam 位移、m/v/step 和 weight decay 贡献。

对于未更新参数必须 grad=None，不能设置成零张量；否则 Adam step、momentum 和 coupled weight decay 语义都会改变。

## 4. J_ACC 精确定义

在相同 pre-state θ 上，对 batch A、B 分别计算梯度，不在 A 后更新模型。按实际行数累计：
[
g_{acc}=\frac{|A|g_A+|B|g_B}{|A|+|B|}.
]
对两个任务的原始 BCE 口径一致；自然尾批按真实 B 加权。完成梯度累计后仅执行一次原生 Adam step。

验证 J_ACC 与将 A/B 拼接成一个大 batch 的 full-batch gradient 在 FP32 容差内一致。若显存不足，不允许改变 batch 语义或使用 AMP；可用梯度累计实现相同数学结果。

## 5. U→P / P→U 语义

以两个自然 microbatch 构成一个 cycle：
- U→P：U(A) → P(B)，P 的 forward 必须使用 U 更新后的参数。
- P→U：P(A) → U(B)，U 的 forward 必须使用 P 更新后的参数。
- AB/BA 映射分别执行并记录。

每个 cycle 的两个 microbatch 各有一个 EMA time tick。J_ACC 的 EMA 在第一个 microbatch 后以未更新模型执行一次 tick，第二个 microbatch 的 Adam 更新后再 tick；U→P/P→U 在两个子步后各 tick；J_ORIG 每个完整 Adam step 后 tick。online 为主要机制读出；EMA 作为次级 readout。所有 arms 的 EMA rho 固定，不搜索。

## 6. 训练 tail 规则

每 epoch 只对两个完整 batch 组成的 pair 使用 block schedule。剩余一个或两个 batch 对所有 arms 使用完全相同的 canonical joint Adam tail rule。不得 drop_last，不跨 epoch 拼接，不改变自然 row order。具体 pair/tail 索引写入每个 run manifest。

## 7. 随机数与状态隔离

每条轨迹固定 seed 和独立 DataLoader generator。诊断 fork 的 RNG 与 baseline 隔离；评价、模型克隆、V_A/V_B 构建不得消耗训练 RNG。保存完整 Python/NumPy/CPU/CUDA RNG 和 loader permutation/cursor。

所有 model/optimizer/EMA/fork tensors RAM_ONLY。磁盘只允许 source、config、hash、标量、CSV、JSON、日志和报告；禁止写 `.pt/.pth/.ckpt`、pickle 模型、参数数组或完整 optimizer state。状态由 RAM guardian 保存，实际 get/hash 验收后才登记为有效。

## 8. 正确性测试

必须包括：
1. canonical forward equality；
2. U/P 参数集合无重叠且完备；
3. 每个 cell 的真实梯度支持；
4. J_ACC 与 concatenated batch gradient 一致；
5. None vs zero gradient；
6. 子集 Adam 更新只改变指定参数与状态；
7. AB/BA fork RNG/loader 不变；
8. fork 不污染 baseline；
9. U→P 中 P gradient 由 U+ 后态计算；
10. P→U 中 U gradient 由 P+ 后态计算；
11. EMA 每 microbatch tick 语义；
12. 50 epoch 的 pair/tail row coverage 无遗漏、无重复；
13. Validation earliest strict max；
14. Test freeze 前拒绝 Test；
15. 断点恢复后下一 batch、optimizer step、EMA 和结果可复现。

coder 完成后交 CODER_IMPLEMENTATION_REPORT；由未编写该实现的独立 reviewer 逐条核查设计、源码、测试与真实 smoke。Reviewer PASS 前不允许正式训练。
