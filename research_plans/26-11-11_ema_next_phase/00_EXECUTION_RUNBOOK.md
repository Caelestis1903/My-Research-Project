# 固定 EMA 之后：下一阶段 Codex 执行入口

日期：2026-11-11（Asia/Shanghai）。研究 ID：`EMA_NEXT_20261111`。本次交付：**研究规划、算法规范和配置；未实现新算法，未启动训练。**

## 1. 明确选择

**主顺序：A 的最小真实反馈 → B 的有界历史预测组合 → C 的域任务条件化价值 → 必要的单模型蒸馏 → 冻结后的正式评价。** B/C 的无 SGD 读出随公共父轨迹采集，不重复训练。A 失败立即转 B；A 成功也只保留一个匹配方向配置和一个更简单的普通 Adam 替代。C 没有比统一策略增加收益时结束当前复杂分域控制支线。

首要问题是新增机制相对固定 EMA 的独立增量。固定 EMA 的有效性、共享表示主导和 EMBED recent×old 正交互已经接受。论文性能仍与已发表模型 M3oE、HyperGate、AutoIFS 比较；内部 F 控制不得列为发表模型基线。

## 2. 交付导航和定义权威

| 文件 | 内容 |
|---|---|
| [01_RESEARCH_CONTEXT.md](01_RESEARCH_CONTEXT.md) | 冻结事实、历史失败分类、复用与禁止重复事项 |
| [02_MECHANISM_ANALYSIS.md](02_MECHANISM_ANALYSIS.md) | 参数/预测平均、Adam、排序与本轮可识别问题 |
| [03_ALGORITHM_EXPLORATION.md](03_ALGORITHM_EXPLORATION.md) | 候选选择、更简单替代、优先级 |
| [04_ALGORITHM_SPECIFICATIONS.md](04_ALGORITHM_SPECIFICATIONS.md) | 全局符号、数据与状态契约 |
| [05_EXPERIMENT_MATRIX.md](05_EXPERIMENT_MATRIX.md) | ID、完整展开规则、成本、触发依赖 |
| [06_CODEX_IMPLEMENTATION.md](06_CODEX_IMPLEMENTATION.md) | 已有路径、应新增接口、必要测试 |
| [07_HYPERPARAMETER_PLAN.md](07_HYPERPARAMETER_PLAN.md) | 有限五 seed 搜索、选参与冻结 |
| [08_AUTOMATED_EXECUTION.md](08_AUTOMATED_EXECUTION.md) | 调度、RAM 生命周期、恢复、资源计数 |
| [09_EVALUATION_AND_DECISION.md](09_EVALUATION_AND_DECISION.md) | 所有数值门、比较对象、状态裁定 |
| [10_CODEX_TASKS.md](10_CODEX_TASKS.md) | 实现、独立审阅、执行与报告清单 |
| [11_RESEARCH_REVIEW.md](11_RESEARCH_REVIEW.md) | 研究与工程审阅及修正 |
| [12_ROUTE_A.md](12_ROUTE_A.md) | A_MATCH / A_RAW 完整数学与反馈规范 |
| [13_ROUTE_B.md](13_ROUTE_B.md) | B_SPLIT / B_SHARED_ONLY / B_DISTILL |
| [14_ROUTE_C.md](14_ROUTE_C.md) | C_GLOBAL / C_CELL 与同系数参数替代 |
| [15_SOURCE_EVIDENCE.md](15_SOURCE_EVIDENCE.md) | 固定 SHA、路径、一手文献和读取范围 |
| [experiment_manifest.json](experiment_manifest.json) | 算法菜单、数值门、预算、运行阶段的机器输入 |
| [SOURCE_LOCK.json](SOURCE_LOCK.json) / [SOURCE_AUDIT.json](SOURCE_AUDIT.json) | 来源锁与已执行的数表复算 |

用户当前指令优先。数值门以 09 和 manifest 的同名字段为准；算法以 04、12–14 为准；矩阵以 05 为准。`prepare` 必须校验这些定义一致，有差异先修复方案/实现，不允许选择有利的一套解释。

## 3. 先取得包含真实实现的工作树

本轮规划提交在 main；实验底座固定为 `26-11-11` 的 `5ef45b18b973ed11be6fbb5f654562349998de60`。main 只汇总了部分实现/报告，不能假设它可直接 import 所有实验包。该底座包含源 `26-11-07` 的模型/数据代码和最新归因实现。不要 merge 整个实验分支进 main。

后续 Codex 接手执行时：

```bash
git fetch origin main 26-11-11
EMA_NEXT_PLAN_REF=$(git rev-parse origin/main)
git cat-file -e "$EMA_NEXT_PLAN_REF:research_plans/26-11-11_ema_next_phase/00_EXECUTION_RUNBOOK.md"
git worktree add -b research/26-11-11-ema-next ../phase0-3_ema_next_20261111 5ef45b18b973ed11be6fbb5f654562349998de60
git -C ../phase0-3_ema_next_20261111 restore --source "$EMA_NEXT_PLAN_REF" -- research_plans/26-11-11_ema_next_phase
```

目标存在时核对 SOURCE/PREPARED 后复用，或选择未占用名字；不覆盖旧工作树。记录实际 plan commit 与全目录 hash，不能将本次读取 main 的旧 SHA 当成规划提交 SHA。

## 4. 自动阶段

| 阶段 | 操作 | 自动进入/退出 |
|---|---|---|
| S0 | 源码/数据/协议审计，最小实现测试，两数据 GPU smoke，资源展开 | 独立实现审阅通过且工程收据齐全 |
| S1-P | Clean 每数据 5 条父轨迹各 50 epoch；保留 E20/E35 RAM 状态；同步 B/C 全 V 曲线 | 每数据 5 seed 完整后才计算科学门 |
| S1-A | 两个 alpha × 两个起点 × 五 seed，各真实反馈 6 epoch | 依据 G_A 决定继续或停止 A |
| S2-A | 条件运行 A_RAW；保留一个 A_MATCH，符合门的 A_RAW；从 E26 续到 E50 | 完整算法固定 E20 开始修改；不临时改起点 |
| S2-BC | 汇总 B、C 的完整曲线和未来训练面板价值；选择简单有效读出 | B/C 无额外 SGD；参数替代与条件信息分别裁定 |
| S2-AB | A 和 B 分别有独立证据时，在 A 完整续训时附带冻结 B 读出 | 无额外训练；完整 2×2 比较；不自动承认叠加 |
| S3-KD | 有价值的两态教师才进入 5 epoch 蒸馏及 label/F-teacher 控制 | 四参数组合均五 seed；保持同预算 |
| S3-G | 每数据至多一个 Clean 主候选进入 Gaussian D0 边界 | 沿用 Clean 选择，不重搜；仍五 seed |
| S4 | 全局方案/端点冻结，独立 Test，四类比较，报告、资源释放 | 规定任务完成或满足预设 STOP；工程阻塞单列 |

物理顺序以**一个数据集的五 seed 块**为单位：父轨迹 → 短反馈 → 条件续训/蒸馏 → 冻结需保留的端点 → 释放无消费者的 RAM。可以资源准入地预取下一块，不让已释放状态变成后续依赖。S1-P 中 B/C 的完整评价是保存可执行候选的必要采集，不代表三条路线并列启动大型训练。

## 5. 未来 CLI 合同

以下命令由 Codex 按 06、10 实现后使用；当前提交不包含该新实验包。

```bash
EMA_NEXT_PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
EMA_NEXT_PLAN=research_plans/26-11-11_ema_next_phase
EMA_NEXT_OUT=artifacts/ema_next_20261111/attempt1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$EMA_NEXT_PY" -m experiments.ema_next_20261111.run audit --plan-dir "$EMA_NEXT_PLAN" --out "$EMA_NEXT_OUT"
"$EMA_NEXT_PY" -m experiments.ema_next_20261111.run prepare --plan-dir "$EMA_NEXT_PLAN" --out "$EMA_NEXT_OUT"
"$EMA_NEXT_PY" -m experiments.ema_next_20261111.run smoke --out "$EMA_NEXT_OUT" --dataset movielens
"$EMA_NEXT_PY" -m experiments.ema_next_20261111.run smoke --out "$EMA_NEXT_OUT" --dataset kuairand
"$EMA_NEXT_PY" -m experiments.ema_next_20261111.run execute --out "$EMA_NEXT_OUT" --devices 0 1 --through report
```

`execute` 自动完成阶段判定、进入预先写明的分支、冻结、Test 和归档；不等待研究者每天再写提示词。`status` 只读，`resume` 使用冻结 PREPARED、原 guardian 和原科学配置，不重新选参。设备号由预检实际确认。资源不足时排队，不以时间到了或某 seed 不佳代替科学停止条件。

## 6. 执行限制与完成标准

- 五 seed 只用 2022–2026；源自然数据、loss 归约和 batch 保持；小域不是困难域，不设计相关假设/分析。
- 遵循仓库 `start.md`：参数、Adam、EMA、历史重心及恢复状态仅驻 RAM；禁止模型/检查点落盘，包括临时 NPZ、mmap、tmpfs 绕行。预测数据、标量与收据可以按 2 GiB 预算落盘。
- 每卡显存与功耗均低于 85%、主机 RAM 低于 90% 时结合预留量准入；继承 guardian 4 GiB commit 与外部 7 GiB tensor 双约束，不设置无依据的两 worker 固定上限。
- 所有结果可反查 experiment/config/source/data/state hash；完整 V 每 epoch 选点，Test 只在全局冻结后打开。已经发布过的历史 Test 只作背景，不进入本轮自动分支。
- 结束时报告 A 是否产生 F 尚未吸收的真实反馈收益、B 是否有可部署增量、C 是否增加统一策略不能提供的价值，以及压缩后成本；所有门都有实际状态。未触发项记 NOT_TRIGGERED，资源/实现阻塞记 BLOCKED，不冒充性能失败。

本轮规划只提交本目录。未来实验的代码与报告在实验工作分支交付，不自动修改 main 的历史上下文或历史原始结果。
