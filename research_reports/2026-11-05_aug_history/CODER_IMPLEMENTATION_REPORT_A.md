# CODER IMPLEMENTATION REPORT A

状态：A_IMPLEMENTED，等待独立 REVIEW。实施依据 `DESIGN_SPECIFICATION_A.md`、route_A 四份冻结文件及原 manifest；B 的原始数据、冻结、报告和已执行数学路径保持不变。未运行 A 正式训练、GPU smoke 或性能分析。

## 范围与实现

新增 `group_registry.py`、`ranking_probe.py`、`grouped_ema.py`、聚焦的 `tests/test_a.py`。仅对现有 protocol/worker/guardian/CLI/selection/report 增加 A 路线分派，没有新增依赖。

- GroupRegistry 以完整源布局生成互斥坐标映射，动态定位实际 domain embedding lookup 行；固定 gate/tower index `3*t+d`，未知参数报错。两数据组数与 DESIGN 的 S/D/T/DT 数字完全核对，INACTIVE=0。A preflight 输出 `GROUP_MAP.json` 和新源码身份。
- 排序探针仅访问 train 数组，排除当前 batch 的 raw row IDs；每格独立 SHA256 键的局部 Generator，从正负池无放回抽最多16行，缺类跳过。每个刷新时刻所有候选共享行，控制器/φ 独立。
- 代理参数用旧 φ 与当前 Adam 后 θ 的 detached 值构造，只有 q 可微。源 functional raw logits 保留 gate 输入 detach；只 `autograd.grad(loss,q)`，不写在线 `.grad` 或 Adam。q 的无穷范数归一化、[-1,1] 边界、无有效格和非有限值按规格处理。
- 15 影子从 θ0 初始化：U/F三种、四固定 profile、四 GROUP、四 GLOBAL。GROUP q 为12维，GLOBAL为1维；时钟从 K20/K40 开始，在 Adam 后、一次 EMA 前更新 q。F/q0 使用源 `mul_(rho).add_(theta,alpha=1-rho)` 顺序；固定 profile 不运行探针。
- worker 的 A 分支保留原唯一 loss/backward/Adam，不进入 B 的 auxiliary gradient/Q 路径。RAM 状态包含 group hash、全部 φ/q/η/K/刷新时钟、probe 身份、controller 次数、候选成本和既有 online/Adam/loader/selection/端点。恢复使用原 guardian 事务/重放机制。
- 每候选独立记录 controller、EMA 与 validation 实测时间，共享抽样成本单列。HPO 按各候选实际成本排序，固定 profile 偏离以绝对 log2 半衰期倍率之和表示；自适应候选共用 q0/边界，静态偏离为0。
- A 的家族选择、固定家族中的 F、中性去重、GROUP 赢家的匹配 GLOBAL、五 seed 门和 test 冻结门按合同实现。A validation 决策只使用冻结端点及成本，满足正增量和逐格保护时保留，否则输出 RUN_C；不读取新 test 调度。

## 新阶段身份与公开流程

A 必须使用新 output（约定 `artifacts/aug_history_20261105/attempt_A1`）。`preflight --route A` 记录新 implementation/source/data/group 身份；独立审查收据为 `REVIEW_ACCEPTANCE_A.json`，绑定此 source hash。后续公开命令加 `--route A`，private worker 从准备的 A job 获得路线，数据和表格输出位于 A 目录。

跨阶段 `continue` 读取传入 B DECISION 所属目录的 PREPARED/selection 冻结身份，不调用当前源码=B旧源码的 gate。它只在 A 输出写 CONTINUATION 收据，包含 B 决策、冻结与 PREPARED 的路径/hash/身份、允许动作和已解析命令。B 冻结文件不回写。

A GPU smoke 的实现以原 batch40960执行到 step20，比较同 seed/step 的源 online 更新，并实际触发四个 K20 GROUP/GLOBAL 控制器；只判数值、路径与资源。正式50epoch×5seed矩阵须等独立审查和此 smoke 通过后运行。

## 必要检查

全部项目 Python 命令使用 `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

```text
python -m pytest experiments/aug_history_20261105/tests/test_a.py -q
8 passed, 1 warning in 161.04s
python -m experiments.aug_history_20261105.run --help
exit 0
python -m pytest experiments/aug_history_20261105/tests/test_b.py -q
11 passed, 2 warnings in 16.57s
```

覆盖两数据完整分组/真实 lookup/六头映射，单 cell 禁止梯度路径与全部活动组的可响应实例；train-only probe 的排除、无放回、缺类、独立 RNG；q0/固定 profile 公式、实际 q 梯度、GLOBAL/GROUP 维数、边界和空/非有限分支；两数据41步源 online/正式梯度逐位一致、K20/K40 控制器顺序、q/φ/时钟/probe 的连续与 RAM 恢复一致；旧 B 身份的只读继续收据。

共享/B 的11项回归检查全部通过，既有数学与生命周期 gate 保持可用。GPU smoke、完整预检和正式 A 训练由 REVIEW 后的 RUN 执行；C 尚未实施。
