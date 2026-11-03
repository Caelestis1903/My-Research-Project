# 2026-11-03 晚间 A＋B 联合实验：Codex 唯一执行入口

**交付状态：工程实验规划已冻结；不是已经启动或完成 GPU 实验的证明。** 本次交付包含可运行的矩阵生成/规则自检程序；训练、调度适配代码由 Codex 按以下确定规格实现，然后实际运行。不得再开展开放式算法设计，也不得只生成另一份研究建议而停止。

目标分支：`research-plan/26-11-03-overnight-ab-auc`。规划目录：`research_plans/26-11-03_overnight_AB/`。人类只需将本文件交给 Codex；其余文件均由本入口导航，不存在另一份需要自行选择的启动提示词。

## 1. 已作出的决定

1. **主表统一完整自然 train 人口及 train-only 预处理。** 10/29 的 T0∪F/T0-only 基线不能仅换 checkpoint 后与近期候选横比。保留历史表；旧轨迹的 AUC 重选只进入独立历史附录。正式 A 表复用近期 M3oE fixed1 online 10 条，原则上新增 HyperGate/AutoIFS 各10条；若服务器已有完全等价新轨迹则继续复用。
2. **主组合只选 C_MIX＋P01。** C_MIX 修正当前真实梯度，P01 保留静态 PAIR 的 ECCM 历史项；融合为一次二阶矩递推、一次参数更新、一次固定 EMA 更新。不得依次调用两个完整 optimizer.step。完整公式见03。
3. **简单替代为 C_MIX＋Gaussian。** 已冻结其完整2×2和参数，仅在主组合工程/资源阻塞时替代，或在核心队列全部完成且事先授权追加时执行；不能因主组合1–4seed表现差而切换。默认不搜索 MASK，不枚举第三个组合。
4. **核心预计新增30条完整训练**：A的20条＋主组合10条。A0、C_MIX、P01控制优先复用或等待正在运行的五seed补评。不是把A的全部30条、B的全部40个消融槽位重跑。
5. 两数据都完成2022–2026五seed。KuaiRand可先运行，MovieLens不是通过单seed门才获准运行的后续阶段。A和B独立实现、独立工程验收、共享公平GPU队列，任何一条线不得被另一条线的全部训练阻塞。

## 2. 阅读及执行次序

首先读取最新 main 的 `EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md`，再读取最新本地历史五seed进度/结果。本规划审计基准 main 为 `92a592418d3f11a588f1e628fbe9364ceab71e18`；读取最新规则不等于任意改变本次固定配置。

依次执行：

| 顺序 | 文件 | 必须完成的工作 |
|---|---|---|
| 1 | `01_AUDIT_AND_REUSE.md` | 原清单修正、逐轨迹复用和外部任务认领；输出真实审计表 |
| 2 | `protocol.json`、`plan.py` | 校验并展开固定矩阵；冻结数据/代码/配置身份 |
| 3 | `02_TASK_A_IMPLEMENTATION.md` | 发表模型数据、训练阶段、AUC选点适配 |
| 4 | `03_TASK_B_IMPLEMENTATION.md` | 融合优化器、两套完整2×2、有限工程测试 |
| 5 | `04_RUNTIME_AND_REPORTING.md` | 队列、RAM恢复、逐seed封存、汇总与发布 |

实现统一放入新包 `experiments/overnight_ab_20261103/`。禁止修改正在补评的 history/pair 包、其 vendor、SOURCE_MANIFEST、旧 decisions/schedule 和历史报告。不能在原工作目录切分支或更新正在运行的源码。

### 工作树准备

Codex从现有仓库保存 `HISTORY_REPO=$(git rev-parse --show-toplevel)`；读取进程/工作树及未提交改动，不清理它们。在独立工作树中，以审计基准 main 为代码基础建立 `execution/26-11-03-overnight-ab`（若已存在则恢复该工作树，不强制覆盖）。从原方案分支复制本规划目录。需要更新的最高优先级main规则另存审计快照；若只是新增结果，可接入复用清单，不改变冻结数学定义。主目录代码存在缺失时按01的固定源补齐，不使用整分支合并覆盖运行目录。

示例首次准备命令（变量均由当前仓库实际解析）：

```sh
HISTORY_REPO=$(git rev-parse --show-toplevel)
git fetch origin main research-plan/26-11-03-overnight-ab-auc baseline/mdmt-sota-2526
WORKTREE="${HISTORY_REPO}_overnight_ab"
git worktree add -b execution/26-11-03-overnight-ab "$WORKTREE" 92a592418d3f11a588f1e628fbe9364ceab71e18
cd "$WORKTREE"
git restore --source=origin/research-plan/26-11-03-overnight-ab-auc -- research_plans/26-11-03_overnight_AB
```

上述仅适用于首次建立；重复执行用 `git worktree list --porcelain` 找已有目录，不能重复创建、reset或删除。将解析后的HISTORY_REPO写入本轮运行配置；其 artifacts 只读接入，新产物使用独立路径。

## 3. 唯一评价规则

选点协议 `AUC_SELECTED_V1`；数据协议 `FULLTRAIN_NATURAL80_10_10`。每个模型/估计器在最终训练阶段epoch1–50中，取 validation 六格等权 macro AUC **严格最大、并列最早**的checkpoint。AutoIFS只允许reuse1–50，不允许search进入候选。禁止LL选点、限制M3为12–24、跨seed平均后挑同一epoch、按test挑epoch。online与EMA各自选点，不要求二者选中同一epoch；**每个端点的validation和test必须是同一个模型状态**。

五seed配对test平均宏ΔAUC≥0.003即宏收益达标；六格各自五seed平均ΔAUC≥−0.001独立判断逐格保护。两项均满足才写“AUC合格且满足逐格约束”。LL平均差<0独立判为改善。不要增加正向seed数量、单seed单格、置信区间、validation/test同向、traffic同向等硬门。不足五seed只写INCOMPLETE。

发表基线只有M3oE fixed1适配、HyperGate、AutoIFS。A0+EMA、Gaussian、C_MIX、P01、组合均不是发表基线。分别报告：对三个发表模型的正式比较、相对A0 online的完整方案总收益、相对A0 fixedEMA的组件增量、相对单组件的组合增量及2×2交互。不能因组件增量未达0.003抹去总收益，也不能把仅胜M3说成胜过三个模型。

**小域不是困难域。** 本任务不提出、检验或反向分析这一命题。只继承已固定的数据和目标域设置。

## 4. Codex必须实现的命令契约

现已交付、无需GPU即可运行的规划程序：

```sh
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
PLAN=research_plans/26-11-03_overnight_AB
"$PY" "$PLAN/plan.py" self-test
"$PY" "$PLAN/plan.py" materialize --out /tmp/mdmt-overnight-ab-plan
```

然后实现新包及04中全部子命令；以下是**实现后**的真实自动化启动命令，不得在模块尚未实现时声称已运行：

```sh
ATTEMPT="$HISTORY_REPO/artifacts/mdmt_overnight_ab_20261103/attempt1"
"$PY" -m pytest experiments/overnight_ab_20261103/tests -q
"$PY" -m experiments.overnight_ab_20261103.run prepare \
  --plan "$PLAN/protocol.json" --history-root "$HISTORY_REPO" --attempt "$ATTEMPT"
"$PY" -m experiments.overnight_ab_20261103.run smoke --attempt "$ATTEMPT" --devices 0,1
mkdir -p "$ATTEMPT"
nohup "$PY" -m experiments.overnight_ab_20261103.run run \
  --attempt "$ATTEMPT" --devices 0,1 --resume \
  >"$ATTEMPT/coordinator.log" 2>&1 < /dev/null &
```

`run`自动完成调度、逐seed冻结/评价、汇总、阶段报告，不要求训练后再由用户手动发起summarize。实现阶段先让已通过工程验收的A或B进入共享队列，另一条线可继续CPU实现；不等待所有可选功能完成才给两项实际运行机会。未通过的具体arm不得派发。

没有GPU权限时须提交实现、测试和READY/WAIT_RESOURCE清单，不能虚报启动。本次规划撰写本身不拥有服务器训练进程。

## 5. 完成条件

- **审计完成**：A旧30条逐条说明是否仅历史重选可用、为何主表不可复用；近期A0/C/P/G逐seed标明REUSE_DONE、WAIT_EXTERNAL或确切缺失。不能把GitHub未推送结果当作服务器未运行。
- **A完成**：两数据×3模型×5seed共30个合法online selected端点，含六格AUC/LL、macro/traffic/pooled、选点和成本身份。预计10复用20新训；逐项以审计实际数为准。
- **B主组合完成**：两数据各完整2×2×5seed，A0/C/P/CP均有固定EMA selected端点；40个消融槽位并非40个新增训练。四格来自同协议匹配seed。若主组合工程阻塞但替代完成，写“替代组合完成；主组合工程阻塞”，不可写主组合性能失败。
- **运行完成**：没有未解释缺项；全部授权job为DONE或明确BLOCKED/WAIT_EXTERNAL。科学完整性与队列结束分开报告，不把部分完成强行记FINAL_COMPLETE。
- **交付完成**：04约定的原始表、配对表、2×2表、审计、进度、费用、FINAL_REPORT.md及台账增量生成；提交执行分支。更新main台账前读取其最新版本并做增量合并，不覆盖并行补评成果，不篡改旧报告。

## 6. 停止边界

不使用任何1–4seed性能停止门。允许明确代码错误、NaN、不可用数据、RAM/GPU资源不足导致工程暂停。自动基础设施重试为首次失败后最多2次，正常epoch让出不算失败。OOM不改batch/精度/目标；只等待资源后原配置恢复。主机或guardian丢失权重时仅重放缺失轨迹，不能宣称RAM-only支持断电续训。

夜间预算不足时保留固定队列继续或可恢复暂停，不能为了“跑满一夜”添加新机制，也不承诺某个时点一定跑完。核心矩阵不因test表现或历史候选补评结论而缩减。
