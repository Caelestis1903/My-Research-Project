# 03 — 统一状态快照、隔离与数据身份

## 1. 快照必须包含

每个 baseline/anchor/fork 的完整状态至少包括：
- model parameters、buffers、training/eval mode；
- Adam m/v/step、param groups、lr、betas、eps、weight decay 和 None/zero gradient semantics；
- EMA parameters、rho、update counter；
- scheduler state；
- Python、NumPy、Torch CPU、CUDA RNG；
- sampler/DataLoader generator、epoch permutation、batch cursor、next row IDs；
- dataset/split/feature encoder/scaler identity；
- config、source commit、dependency closure、environment；
- epoch/global step、checkpoint selection history；
- validation panel and calibration panel identities。

只保存 model weights 不构成可恢复训练状态。

## 2. D2 state 复用

先读取 D2 10 条 J_ORIG trajectories 与 30 anchors 的 receipts。每个 state 必须 live readback：
1. state 对象真实存在；
2. hash 与 source/config/data/anchor identity 一致；
3. restore 后 model prediction parity；
4. Adam、EMA、RNG、loader cursor parity；
5. next batch row IDs parity。

任一失败则从冻结 source/config 重建。endpoint CSV 或 hash index 不能替代完整 state。

## 3. Fork 隔离

- baseline 和 anchor 只读；
- 每个机制 variant / arm / seed / dataset / anchor 独立 clone；
- 不共享可变 model、optimizer、EMA、loader、RNG；
- 每个 fork 写 parent_state_hash、start_state_hash、end_state_hash；
- 训练前后写 batch row IDs、optimizer transitions 和 EMA ticks；
- fork 间结果只通过 immutable receipts 汇总。

## 4. Adam 子集更新

M1 P-only consistency step：
- U 保持 requires_grad=True，构造 g_U,c 的二阶计算图；
- consistency loss 对 P 求导；
- backward 后显式将 U.grad=None；
- 再调用同一个 native Adam.step()，让 Adam 跳过 U；
- 核验 U 参数、m/v/step bitwise unchanged；
- P 按原生 Adam 语义更新；
- 不得将 None gradient 改成 zero gradient。

M1 的 coordinate-shuffled control 对每个 cell、每个 U parameter tensor 使用独立且固定的可逆坐标 permutation，保留各张量范数但破坏 cell gradients 之间的坐标对齐；所有 cell 共用同一 permutation 会保持 cosine，不是有效 control。简单置换 cell ID 在等权求和下不改变目标，不是有效 control。

## 5. 虚拟 Adam 与 M2

M2 的 full 与 leave-one-row-task-out 虚拟更新必须从同一 optimizer state 克隆计算：
- full gradient；
- full gradient 减去指定 row-task 的 BCE gradient contribution；
- 两次 native Adam virtual displacement；
- ΔU_i,t = ΔU_full − ΔU_without(i,t)。

两次虚拟更新不得修改 live optimizer、EMA、RNG 或模型状态。验证 row-task contributions 按 canonical loss reduction 求和可重建完整 batch gradient。

## 6. Evaluation 与 RNG

所有 Validation forward、pair-margin calculation、feature permutation audit 和 diagnostic forward 必须保存并恢复训练 RNG。额外创建 DataLoader iterator 可能消耗全局 RNG；必须使用隔离 generator 或保存/恢复 RNG，并核验后续 batch order。

## 7. Panel 与数据隔离

- Train calibration panel 不与训练 batch 重叠。
- V_A 与 V_B row IDs 和 pair IDs 互斥。
- 所有 panel 在结果读取前冻结。
- Test arrays 在 S0–S3 不加载、不解码、不评分。
- 变换数据保留原始 row ID、domain membership 和原始 labels。
- M5 每种变换保留 before/after feature hash、label pair、domain ID、合法性和一致性 receipts。

## 8. 原子持久化

采用 temp write → fsync → atomic rename。每个 checkpoint 包含 state.pt、state_manifest.json、hashes.json、next_batch_receipt.json、validation_readout.json、costs.json。读回验证通过后才能标记 COMPLETE_VERIFIED。
