# GAM01 独立实现审查

日期：2026-11-01。审查者与 coder 为不同 agent，均继承协调者模型。审查输入为已批准的 `DESIGN_SPECIFICATION.md`、新包实际实现、`IMPLEMENTATION_REPORT.md` 及规划 04/05。未修改源码，未启动 CUDA Stage0 或正式训练。

**最终 verdict：PASS。初审为 PASS WITH FIXES；原 coder 已修正 Blocking B1，独立复核通过，当前无未解决阻塞项。可交 runner 执行 Stage0；真实 Stage0 通过后方可进入正式40run。**

## 已核对的实现

| 范围 | 实际行为与核对结果 |
|---|---|
| 梯度目标 | `objectives.py` 复用 GA31-R3 标量目标；A0 保留任务均值再样本均值，A1 保留增强标量求和次序。SAD 的真实通道为 full-batch 梯度，差分为 target augmented 减 target clean，分母为原始 B×2；参数更新前完成两次 autograd.grad。 |
| Adam/SAD | beta_delta=0、关闭 SAD、关闭增强/λ0 等由标准 Adam 分支执行；A2=.9 是分源等价控制，A3=.99 改变差分时间尺度。m_real 包含 coupled decay，q_delta 不含 decay，唯一 v 使用当前混合梯度加 decay；三种 bias correction 共用逐参数 step。状态初始化为零，更新在 no_grad 内。 |
| None | full-real 与 delta 均 None 时跳过全部状态；单通道连接时缺失通道按零，显式零仍更新；λ0 不激活仅增强连接的参数。目标缺失但真实通道活动时已有 q 按固定系数衰减。 |
| Shadow | deepcopy 初始化且 requires_grad=false；每次在线更新后做 ρ=.995 EMA，buffer 复制；不反馈训练。关闭时不创建模型或额外 validation。在线与 shadow 严格独立按完整 validation macro AUC 选点，平局最早、E0 排除。 |
| 快照/随机流 | online、shadow、m/q/v/step、Python/NumPy/Torch RNG、loader generators 和选点身份均仅存 RAM。复用 R3 独立视图 Generator/key；恢复使用保存的 shadow，未重新初始化。磁盘只写数值、哈希、配置、日志。 |
| 数据/模型复用 | 完整命名空间导入未改 R3 的数据、fixed1 模型、视图和评价；继承 raw/六数组/split/encoder/scaler 身份和迁移后的 CSV 路径。无新增依赖、网络参数或采样规则。来源字节核验复用协调者已提供的收据。 |
| 矩阵/调度 | 构造两数据集×五 seed×四臂；每 run 50 epoch、batch40960，一 batch 一次参数更新。每张卡按自身显存/功耗阈值并发；首次分配回执后继续调度。OOM 仅按原 batch 降并发重试，失败保留 ledger；不删 seed 或缩训练。 |
| 选择与 test | 40 run 完成后核验同 seed 初始化/50轮 row 顺序；先以 validation+D1 冻结 P2 判断。触发时保留 RAM 等独立设计/review后的 P2；未触发时冻结全部 online/shadow selected/E10/E50 身份后统一 test。没有依据 test 决定 P2。 |
| 数值表与成本 | epoch/per-cell/selected、pooled/traffic/worst、域/任务、五 seed 配对 mean/sample SD/df4 区间均有输出。真实训练 forward、梯度遍历、一次 step、增强曝光和额外 shadow validation 分别计数。D1 标为 epoch 最后实际 batch；局部 Adam 对照明确为同映射状态的一步方向。 |

## Blocking B1：自动 GO 遗漏 validation 依据和保护项

初审位置：`experiments/gradient_aug_momentum_20261101/analyze.py:119`。代码已经计算两种 split 的 gate/guards，但最终 `status` 仅以 test 的 guards/gain_gate 决定。规划 04 §3.4 要求 P0 有 validation 的实用收益且保护项未触发，再由 test 支持；P1 也要求开发方向支持。此缺陷会把 validation 退化而 test 正向的结果自动记为 GO，直接改变交付裁决。

最小合成表复现：五 seed 的 P0 selected validation Δmacro AUC 均为 −.003，test 均为 +.002；validation 五项保护均触发、test 无保护退化。现代码仍输出 `status=GO`、`validation_gain_gate=false`。合成数据不参与研究统计，临时目录已自动清理。

协调者批准的最小修正条件：

- P0：GO 必须同时满足 validation/test 既有 gain_gate，且双方 guards 均空。
- P1：相对 A0 与 A1 的 selected validation 平均 Δmacro AUC 都须大于零，并同时通过两项 test 既有 gain_gate；双方 guards 均空才 GO。
- 任一 split 保护触发时保留 TRADEOFF，明确触发 split；其他未过条件记 NO_PROMOTION，保留完整 validation/test 数值以分析方向分裂。
- 增加最小合成 assert，覆盖 validation 反向而 test 过门不会 GO，以及 validation 保护项被保留。训练公式和正式矩阵无需修改。

## 复用与新增验证

独立运行同一必要 CPU 测试：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/gradient_aug_momentum_20261101/tests/test_semantics.py`，结果 **3 passed**。公式收据包含 FP32/FP64 三步退化、beta=.99 手算及 None/逐参数 step；语义与纯 RAM 恢复检查均 PASS。已有证据覆盖公式、目标分母、shadow 非干扰和状态恢复，无需重复扩展。

真实 Stage0 已实现，尚待 runner 在 B1 修正和最终 review 通过后执行：原 batch40960、两个数据集 seed2022、每数据集12次实际开发更新，不评价 test。固定源真实模型预测/loss/梯度/Adam 一步、CUDA 显存、shadow on/off 与恢复后的下一步必须实际通过后，才可进入正式40run。

## 审订记录

1. 初审：核心训练与状态实现符合设计；发现 B1 并给出上述最小修正，verdict PASS WITH FIXES。
2. 原 coder 修正 `analyze.py`：新增直接的 `method_verdict`，P0/P1 分别执行上述 validation 开发条件；test 原增益门仍必须通过；双方 guards 汇入 `guard_splits` 并优先保留 TRADEOFF。`build_tables` 实际调用该函数，未改训练公式、矩阵、选点或 P2 触发。
3. 独立复核该实现、新增合成 assert 及实施报告修订。定向运行 `python -m pytest -q experiments/gradient_aug_momentum_20261101/tests/test_semantics.py -k method_verdict`：**1 passed，3 deselected**。coder 已报告同文件全套 **4 passed**；未重复已有效的完整 CPU 公式检查。
4. B1 已关闭，最终 verdict **PASS**。CUDA Stage0 尚未运行，交由 runner 实际执行。
