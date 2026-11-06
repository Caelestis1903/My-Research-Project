# EMA 收益机制研究：Codex 唯一执行入口

## 1. 执行目标与当前交付状态

**先实现无反馈的历史观测工具，再按已冻结的 20 条轨迹完成机制研究。** 本轮研究 EMA 的经验收益、六格/路径/时间尺度分布、收益指标的描述与预测价值，以及 Gaussian 的交互结构；不设计或运行新的自适应 EMA 算法。

本目录的当前交付仅包含规划文档、已归档结果重算和执行配置，未启动 GPU 实验，未改动正式训练代码。`experiments.ema_mechanism_20261106` 是本方案要求 Codex 新建的包；下面的研究 CLI **在实现和工程验收后才可执行**，不是仓库已有命令。

任务来源中，最前的“MDMTRec 中 EMA 参数历史收益机制研究”明确规定本次目标与 `research_plans/26-11-6/` 交付位置。其后拼接的旧 A/B/C 规划与探索内容按历史背景处理；最新分支已经实现并运行的 A/B/C 不能重新标成待完成路线。

### 不可变规则

- 科学样本固定为每 dataset×objective 的五个 seed `[2022,2023,2024,2025,2026]`。工程 smoke 不参与科学结论；同轨迹多个影子、epoch、cell、panel、重放不增加独立样本数。
- 使用自然人口 `FULLTRAIN_NATURAL80_10_10`，50 epoch、batch 40960、源 Adam/Gaussian/EMA 语义。完整 Validation 六格等权 Macro AUC 的最早严格最大值选点。
- 小域不是困难域；本研究不设置相关假设、解释变量或验证实验。
- online、F、多个 half-life、pulse 均为内部研究对照。正式发表基线只引用已有 M3oE、HyperGate、AutoIFS；不重新训练这些基线。
- Test 只在配置、比较规则、checkpoint 冻结后进入独立评价进程。旧 Test 可以作归档证据，不能再声称是未使用的新测试人口。
- 不把算法平均 AUC≥.003 的资格线套到机制问题。报告实际五 seed 效应、样本 SD、条件和决策；不设置全 seed 同向或置信区间排除零的统一成功门。
- 参数、Adam、shadow、完整梯度、checkpoint 始终 RAM-only。预测/标量允许有界落盘；不得记录完整逐步参数轨迹。

## 2. 阅读依赖与唯一事实源

| 顺序 | 文件 | 执行者应获得的决定 |
|---|---|---|
| 1 | [01_EXISTING_EVIDENCE.md](01_EXISTING_EVIDENCE.md) 与 [evidence_snapshot.json](evidence_snapshot.json) | 已成立事实、原始数值、版本区别、可复用与真正缺失的观测 |
| 2 | [02_SCIENTIFIC_QUESTIONS.md](02_SCIENTIFIC_QUESTIONS.md) | H1–H4、竞争解释、四层结论和下一研究决定 |
| 3 | [03_METRICS_AND_MATH.md](03_METRICS_AND_MATH.md) | EMA、创新量、pulse、收益方向、M0–M3的准确数学定义 |
| 4 | [04_EXPERIMENT_MATRIX.md](04_EXPERIMENT_MATRIX.md) | S0–S4的有限矩阵、共享轨迹与停止条件 |
| 5 | [05_IMPLEMENTATION_SPEC.md](05_IMPLEMENTATION_SPEC.md) | 实际组件、新模块、张量、更新次序、CLI、事务和工程验收 |
| 6 | [06_EVALUATION_AND_DECISIONS.md](06_EVALUATION_AND_DECISIONS.md) | 五 seed 统计、未来预测、联合简单参照、控制触发与研究决策 |
| 7 | [07_RESOURCES_AND_REUSE.md](07_RESOURCES_AND_REUSE.md) | 资源上限、重放理由、估算来源、恢复与释放 |
| 8 | [08_RELATED_WORK.md](08_RELATED_WORK.md) | 相关原始文献能支持什么、不能替本研究断言什么 |
| 9 | [experiment_manifest.json](experiment_manifest.json) 与 [09_PLAN_REVIEW.md](09_PLAN_REVIEW.md) | 冻结数值/依赖/产物合同及交付前检查结果 |

数值配置由 manifest 驱动；公式语义由 03 定义，科学评价由 06 定义，状态/接口由 05 定义。三者已按本轮审阅统一。若后续实现发现真实冲突，先记录具体键/公式并修复一致性，不自行扩大实验或选择对结果有利的解释。运行时生成 `RESOLVED_PROTOCOL.json`，同时绑定本目录所有文件的 SHA256。

## 3. 必须从正确源码建立执行工作区

审计时分支状态：

| 对象 | 固定 SHA / 事实 |
|---|---|
| 规划前 main | `2645f1f724f52f5ed5dea4b479335c0dc12fabcd` |
| `26-11-06` 实现基底 | `40705dfe0bb4fe614bdca835c0bae567348c87d0` |
| 共同祖先 | `652ee0235fec2d0d4cfc67b4a6ef9977519619bf` |
| 最新实现相对当时 main | ahead 13、behind 2；不是 main 的简单同义引用 |

本次只把本目录提交 main，没有把整条 `26-11-06` 实现分支合并 main。**不能直接在 main 假定最新模型适配器、12组实现及历史结果都已存在。** 使用独立 worktree，以固定实现 SHA 为父基底，只导入 main 的规划目录。

在已有仓库根目录执行以下现有 Git 命令；它们不启动训练：

```bash
set -euo pipefail
git fetch origin main 26-11-06
EMA_PLAN_REF="$(git rev-parse origin/main)"
EMA_SOURCE_REF=40705dfe0bb4fe614bdca835c0bae567348c87d0
EMA_REPO_ROOT="$(git rev-parse --show-toplevel)"
EMA_WORKTREE="$(dirname "$EMA_REPO_ROOT")/My-Research-Project-ema-mechanism-20261106"
git cat-file -e "$EMA_SOURCE_REF^{commit}"
git cat-file -e "$EMA_PLAN_REF:research_plans/26-11-6/experiment_manifest.json"
git worktree add -b research/ema-mechanism-20261106 "$EMA_WORKTREE" "$EMA_SOURCE_REF"
git archive "$EMA_PLAN_REF" research_plans/26-11-6 | tar -x -C "$EMA_WORKTREE"
cd "$EMA_WORKTREE"
```

若同名 worktree/分支已存在，核验其 source ancestor、plan hash 和任务所有权后恢复；不覆盖现有工作，也不再次创建重复实验目录。其他用户工作区可以保持未提交状态，本工作区独立进行新增代码。将 `EMA_PLAN_REF`、实际 HEAD、源文件 hash 记录进 `SOURCE_RECEIPT.json`，不能只记分支名。

## 4. 固定数值与研究读出

| 设置 | MovieLens | KuaiRand |
|---|---:|---:|
| Clean / Gaussian D0 逻辑轨迹 | 5 / 5 | 5 / 5 |
| 每轨迹 epoch | 50 | 50 |
| Adam step/epoch | 20 | 49 |
| 常驻 half-life（epoch） | 1、2、4；F=H2 | 1、2、4；另保留F |
| 历史 F 每步 rho | .9828205985452511 | .995 |
| F 的实际 half-life（epoch） | 2 | 2.8220933262 |
| Gaussian κ / replacement λ | .05 / .5 | .1 / .5 |
| Adam coupled weight decay | 1e-4 | 1e-5 |

所有其他训练参数已在 manifest 固定。训练时同时采集 S1–S3 所需观测，分析阶段不重新训练。

**局部历史吸收读出：** 在终点 e=5,10,…,50 的前一 epoch 末，从 F 复制 fast/slow 两向量。接下来一个 epoch 只读取相同 online，分别用 `rho_F²` 和 `sqrt(rho_F)` 更新。终点仅替换一个组，依次读取 12 fast、12 slow、全 fast、全 slow；F复用。这个操作精确对应一组在该窗口改变吸收速度，只需两份临时历史，完全不反馈 online。

**收益指标：** M0 几何量、M1 函数幅度、M2 完整功能排序导数、M3 直接训练排序效用。在统一固定 ridge 下，与设计/阶段、M0、M1、M0+M1、包含梯度 RMS 和全局收益的 C_simple 比较。八类回归使用同一数据折；正则为 **平方误差总和＋1×参数平方和**，截距不罚，不搜索系数。

**预测检验：** P_A 的 e 时刻信息预测 e+5 新 pulse 的 `[fast−F,slow−F]` 两列 V 效用。每数据集留一 seed，其他四 seed 的 origins 5/10/15 和 V_A 拟合；留出 seed 的 origins 25/30/35/40/45 和 V_B 评分，再交换行分区。评价 MAE、相对所有简单参照的 skill、regret 和实际动作净益；允许选 F。另在拟合折冻结一个统一快慢诊断动作，检验方法是否超出全局偏好。收益数值预测与动作价值分别判定，简单指标也可支持下一步定向设计。24个函数可达 cell×group 关系进入主平均，其余48个结构零保留检查，不充当预测成功样本。

## 5. 工程实现与验收顺序

### DESIGN 已冻结

不重新设计 H 网格、增强目标、指标家族、种子、观察窗口、数据分折或触发条件。先完成 S0 的归档复现，继承 EMA 已有收益和 A/B/C 的版本性结论。必要缺项记具体来源，不为旧性能表重跑基线。

执行角色继承源AGENTS：主代理负责统筹和最终分析，`designer`核对本已冻结设计，`luna-coder`实现，`reviewer`独立审阅，`luna-runner`仅运行并记录原始结果。审阅结论必须为PASS，或PASS WITH FIXES且阻塞项已修复。旧四seed默认和旧域规模/难度分析要求由本次用户的五seed与明确排除规则覆盖。

### IMPLEMENT

按05仅新增 `experiments/ema_mechanism_20261106/` 包及该包测试。引用已核实的模型/数据/指标/状态纯组件，独立组织新 worker，不调用旧 A/B/C resolver 冒充新协议。源正式训练实现不修改。

1. 实现协议/数据身份、固定面板、常驻 shadow 和 pulse。
2. 实现只读 scratch、完整功能与 source-detach 两种 M2、即时梯度 RMS 归约、原概率指标及有限预测输出。
3. 实现 RAM guardian、epoch 原子事务、重放身份、8类固定预测器、条件控制和冻结入口。
4. 实现全部 CLI 子命令及产物 schema，运行 `--help` 验证不存在虚构参数。

### REVIEW 与工程准入

至少一次独立审阅覆盖“源训练语义未变”和“统计/未来信息隔离”。必要验证为05第10节九类检查，不额外增加科学性能确认实验。

`preflight` 核验原 CSV SHA、实际缓存、split/row/encoder/scaler身份、字段与12组、环境、磁盘和可用GPU。原缓存缺失时可用原 `prepare_cache` 在新的任务数据目录重建；身份必须完全匹配。数据缓存与观察输出分开记账。

`validate` 通过后，对两数据集各做一次 seed2022 工程 smoke，覆盖 clean/G-D0、batch40960、实际尾批、一个完整 pulse 时钟和观察纯度，标 `ENGINEERING_ONLY`。根据实际峰值决定一或两 worker，并生成 `RESOURCE_ESTIMATE.json`。不能缩小 batch/seed/epoch 以通过资源准入。

## 6. 实现后的一次完整执行序列

下列均为05要求实现的新 CLI。变量只减少重复；**本次规划交付不运行这段命令**。解释器默认使用已审计环境；实际 cache 可显式传给 preflight，之后冻结到 out。

```bash
set -euo pipefail
EMA_PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
EMA_MANIFEST=research_plans/26-11-6/experiment_manifest.json
EMA_OUT=artifacts/ema_mechanism_20261106/attempt1
EMA_MODULE=experiments.ema_mechanism_20261106.run

"$EMA_PY" -m "$EMA_MODULE" audit --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" preflight --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" validate --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" smoke --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --dataset movielens --device cuda:0
"$EMA_PY" -m "$EMA_MODULE" smoke --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --dataset kuairand --device cuda:0
"$EMA_PY" -m "$EMA_MODULE" train --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --devices 0 1 --resume
"$EMA_PY" -m "$EMA_MODULE" analyze --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --stage 1
"$EMA_PY" -m "$EMA_MODULE" analyze --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --stage 2
"$EMA_PY" -m "$EMA_MODULE" analyze --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --stage 3
"$EMA_PY" -m "$EMA_MODULE" controls --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --decision "$EMA_OUT/STAGE3_DECISION.json"
"$EMA_PY" -m "$EMA_MODULE" analyze --manifest "$EMA_MANIFEST" --out "$EMA_OUT" --stage 4
"$EMA_PY" -m "$EMA_MODULE" freeze --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" evaluate --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" report --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
"$EMA_PY" -m "$EMA_MODULE" release --manifest "$EMA_MANIFEST" --out "$EMA_OUT"
```

示例设备0/1由 preflight 映射到允许使用的设备；只有一个可用GPU时用 `--devices 0`，科学预算不变。不得结束他人进程、使用 sudo 或覆盖历史 attempt。合法恢复继续使用同一 out/manifest/job_id；内容一致的重复事务返回 `ALREADY_COMMITTED`。

## 7. 阶段依赖、停止与研究决定

| 阶段 | 必需输入 | 主要产物 / 完成标准 |
|---|---|---|
| S0 / audit | 固定源码、历史数表、本计划 | `AUDIT_INVENTORY.json`、`EVIDENCE_REUSE.csv`、`ARCHIVED_FACTORIAL.csv`；明确已有与缺失观测 |
| 工程准入 | S0、实现、原数据 | `SOURCE_RECEIPT.json`、`DATA_MANIFEST.json`、`PROBE_MANIFEST.json`、验证/smoke收据；观察不改训练 |
| 共享采集 | 准入通过、20个job | 1…50epoch全V曲线、innovation汇总、pulse读出/指标、有界预测、RAM端点；五seed齐全 |
| S1 | 全V曲线、固定H | 时间阶段/六格收益、年龄匹配online、选点差异、交叉评价偏好空间；不扩大H网格 |
| S2 | 已采集pulse与预测 | fast−F、slow−F、fast−slow、排序转换、组非加性；不因增量小重开训练 |
| S3 | 固定特征/标签/折 | 八类预测器及四类无拟合诊断的五seed结果、`STAGE3_DECISION.json` |
| S4 | 仅预注册触发 | 温度最多960 fits、第二背景最多160新增状态；未触发记 `NOT_TRIGGERED`，之后停止追加控制 |
| 冻结/Test | S1–S4完成、20job全预算 | `SELECTION_FROZEN.json`+`FREEZE_RECEIPT.json`；仅常驻读出的selected/E10/E50；合法旧端点直接复用 |
| 报告/释放 | 全部表、来源与成本 | `FINAL_REPORT.md`、`MECHANISM_DECISIONS.json`、图和成本；释放本任务参数RAM |

工程失败不能写成科学阴性。一个逻辑job最多额外重放一次，真实成本另记，不形成新seed。任何未解决问题必须具体到缺失数据/公式/状态或冲突效应；不使用笼统“需进一步验证”替代已有结论。

科学输出按范围给出 `READY_FOR_TARGETED_DESIGN`、`DESCRIPTIVE_ONLY`、`UNIFORM_HISTORY_SUFFICIENT_IN_SCOPE`，必要时标具体 `SPECIFIC_QUESTION_UNRESOLVED`。简单参照有效时优先它；复杂指标有效时接受其经验预测价值。S4结束即报告，不自动设计下一代算法。

## 8. 资源与生命周期总账

| 资源 | 正常完整方案 | 故障或条件上界 |
|---|---:|---:|
| online逻辑轨迹 / epoch / Adam | 20 / 1000 / 34500 | 每逻辑一次额外重放：40物理 / 2000 / 69000 |
| 完整 Validation 模型读出 | 4500 | 全部一次重放：9000 |
| pulse新增状态读出 | 5200（F另复用200次） | S4第二背景再加≤160 |
| M2小面板梯度归约 | 2400（P_A、六格、两语义） | 不增加P_B反向 |
| 完整Test端点 | ≤270，状态重复与合法归档去重 | 不评价pulse/条件最优模型 |
| Guardian参数payload | 约3.124GB，模型状态硬上限4GiB | 不含只读数据mmap；RSS/IPC另测 |
| 有界预测主体 | 1,857,945,600 bytes≈1.73035GiB | 加256MiB标量/元数据预留仍<2GiB；S4只留派生标量 |
| 并发 | 最多2 worker | 资源不足先降并发 |
| 耗时排程量级 | 约8–12累计worker小时 | 按smoke重估；非GPU核时或墙钟承诺 |

S4只保留 e20/e40 的 F/fast/slow 三向量；不用的锚点即时释放。selected/E10/E50留到冻结Test完成。模型状态不落盘，不能从CSV恢复参数；guardian丢失按同逻辑job确定性重放。完成全部报告即自动释放本任务guardian，不把释放作为再次征求用户确认的步骤。

## 9. 最终交付与验收

Codex 的最终报告必须独立回答 Q1–Q5 和是否值得进入定向算法设计，按“当前结论→核心证据→适用范围→研究决定”组织。每项引用实际表/图、五seed数据、source/data/plan/implementation hash。

验收的关键不是找到更高AUC的新方法，而是：完整继承已成立事实；清楚定位新的历史吸收效应；公平比较简单与复杂指标；允许有用的局部结论；对没有预测空间的方向明确停止。全局上下文和台账可生成更新提案，不回写或篡改旧结果。

本次规划提交与未来实现/结果提交分开。本目录是交给执行者的唯一入口，后续无需另行科研设计。
