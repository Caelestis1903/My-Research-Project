# 03 — 统一基础设施与状态快照

## 1. 状态身份

每个基线状态必须绑定：
- repo commit、branch、source closure SHA-256；
- raw dataset SHA-256、split ID、train/validation row ID hashes；
- model class、fixed1 config、完整 parameter manifest；
- optimizer class/config、Adam m/v/step；
- EMA state 与 update counter；
- Python/NumPy/Torch CPU/CUDA RNG；
- sampler/DataLoader generator、epoch permutation、batch cursor；
- scheduler state（如有）；
- epoch/global step、last completed batch；
- environment versions、device identity；
- validation selection history。

## 2. 基线来源与复用

优先审计 D2 S1 的十条 J_ORIG trajectories 和 30 anchors。S1_RUN.json 表明运行曾完成，但 RAM guardian 记录不代表当前仍可读取。每个状态必须通过 live readback：
1. state file/guardian object 存在；
2. SHA 与 ANCHOR_INDEX / source / config / data identity 一致；
3. 加载后重新预测与存储 readout 一致；
4. Adam、EMA、RNG、loader cursor 一致；
5. 下一批 row IDs 完全一致。

不满足任一项则从固定源重建，不得复用 endpoint CSV 代替 state。

## 3. 克隆隔离

- Anchor baseline 只读。
- 每个 arm fork 独立 model、optimizer、EMA、RNG、loader。
- fork 前后写入 state hash。
- 不同机制间不共享可变对象。
- 每个 fork 记录 parent anchor ID、arm ID、batch A/B row IDs。
- 诊断评价 forward 必须保存/恢复所有 RNG 状态，不能改变后续 batch permutation。

## 4. Adam 子集更新语义

M1 的 P-consistency step 必须：
- 对 U 设置 requires_grad=False 或冻结更新，但保留计算 g_c 对 U 的依赖图；
- 通过 create_graph=True 获取 g_c；
- consistency loss 对 P 求导；
- 只允许 P 参数更新；
- U 的 tensor、Adam state、step counter 逐项 bitwise unchanged；
- P 的 Adam state 按原生 Adam semantics 更新；
- P-BCE 对照使用相同参数组、相同 step 数和 weight decay。

不得为方便将 None gradient 改成 zero gradient。None/zero 对 Adam momentum 与 coupled weight decay 的行为不同。

## 5. 二阶图与内存

不显式构造 Hessian。用 autograd.grad(..., create_graph=True) 得到 g_U,c，再对标量 L_cons 求 P 梯度。按 cell 顺序逐个构造/释放图；只保留计算一致性目标所需的张量。GPU smoke 必须记录峰值显存与墙钟。

若 full U 二阶图 OOM：
- 不可直接切换到一阶 proxy 并保留 M1 名称；
- 先检查图生命周期、checkpointing 和 cell-wise accumulation；
- 若仍无法运行，TECHNICAL_BLOCKED；
- 如需缩小 U 子集，必须提交新 estimand amendment 与独立审查，不能在运行中临时缩小。

## 6. 数据和面板

- Train calibration panel 与训练 batch 不重叠。
- Validation V_A / V_B 行互斥，按 domain-task 分层。
- 正负 pair 不重叠；固定 row IDs 与 pair IDs。
- Test arrays 不加载；只保留预先存在的 sealed provenance hash。
- 所有样本保持原始 domain membership 和自然 row weight，除 M2 预注册的域任务内权重处理外不得改变。

## 7. 原子保存

快照保存采用临时文件 + fsync + atomic rename。写入：
- state.pt（模型、Adam、EMA、RNG、loader、scheduler）；
- state_manifest.json；
- hashes.json；
- validation_readout.json；
- next_batch_receipt.json。

恢复后必须执行预测 parity 与 next-batch identity test。
