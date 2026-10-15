# REVIEW REPORT — Stage E1（第三次/最终独立复审）

## 1. Verdict

**PASS**

冻结后的实现已修复上一轮三个 blocking MAJOR：

1. equal/sample trace 长度因合法 early stop 不同时，比较器按共同 `(kind, epoch, step)` 支持比较，并仅把停止后的非共同尾部视为 censoring；共同支持中的任何 fingerprint/order 差异仍为 `INVALID`。
2. formal pair 与 resume 会检查初始参数 hash、受控 config、数据/split/model/partition identity，以及 50 epoch upper bound、patience=10 等正式协议；3-epoch smoke 不会进入 formal `done_set`。
3. ON/OFF 专用比较器只忽略预期的 `is_probe` 差异，仍强制 batch key/fingerprint/order、逐 step loss、逐 epoch/summary metrics、初始/最终参数 hash 与 RNG 恢复证据。

实现与 DESIGN 的训练目标、梯度公式、参数分区、mask/routing、optimizer 路径及 baseline cadence 一致。未发现 CRITICAL 或仍会阻止正式 RUN 的 MAJOR 问题。

本轮未读取 `group2.md`/`group3.md`，未修改实现代码，未启动 24-run 正式矩阵，也未分析研究结果。

## 2. Requirement-by-requirement verification

### 2.1 上轮三个 blocking 的最终复核

| Blocking item | Result | Independent evidence |
|---|---|---|
| common-support + censoring | **RESOLVED — PASS** | `compare_pair_runs` 以 `(kind, epoch, step)` 建 key，检查重复 key、共同支持顺序及共同支持内的 fingerprint/size/domain/task/probe 字段；只有短臂 manifest 明确记录 observed stop 且非共同项全部晚于 stop 时才允许尾部。现成单测覆盖合法 2-vs-1 epoch 为 `VALID`，以及共同支持 fingerprint 被改后为 `INVALID`。 |
| formal pair identity/protocol + resume | **RESOLVED — PASS** | config diff 仅放行 weighting、loss_weighting、run path/hash；manifest 比较 dataset、D/T、role、train counts/share、split counts、positive rates、initial hash，以及 partition groups/counts/shapes/manifest/state hash。`FORMAL_PROTOCOL` 固定 epoch=50、batch=40960、lr、arch-lr、weight decay、expert=4、probe=1、instrument=1、role sizes、patience=10、backbone/task/domain/scheduler。resume 另检查规范化 dataset path、observed epoch 连续性、stop/censor 边界和 50-epoch metric cardinality。3-epoch真实 smoke 被 formal validator明确拒绝。 |
| ON/OFF comparator | **RESOLVED — PASS** | `compare_on_off_runs` 只将 `is_probe` 加入 ignored fields，并开启逐 step loss `1e-7` gate；metrics_epoch 与 metrics_summary 使用 `1e-8` gate；初始/最终 hash 和逐 epoch RNG evidence 必须存在、恢复成功且两臂完全相同。本轮真实 3-epoch CUDA ON/OFF 返回 `VALID`，无任何 mismatch。 |

### 2.2 实现忠实性与正式矩阵

| Requirement | Result | Evidence |
|---|---|---|
| 固定 24-run matrix | PASS | `build_tasks` 为 `2 weighting × 3 settings × 4 seeds`；本轮 full dry-run 显示 24 pending 并逐一枚举 equal/sample、三个 role rotation 和 2022–2025，未启动 child。 |
| 唯一训练差异 | PASS | 正式两臂除 `weighting/loss_weighting` 与输出路径/hash 外受控 config相同；未启用 sqrt、curriculum、momentum、EMA、Polyak、新 router/sampler/scheduler。 |
| objective coefficients | PASS | equal 为六个 1；sample 为 task-major `[2.4, 0.552, 0.048] × 2`；两者系数和均为6。 |
| P0-4 baseline cadence | PASS | role subsample、loader、MDMTRec、main Adam、独立 architecture Adam、每 epoch architecture update、validation checkpoint及 patience=10 均保持原协议。前轮 reviewer 已独立确认 E1-OFF 与 P0-4 三 epoch最终 state_dict逐 tensor完全相等。 |
| checkpoint/censoring | PASS | checkpoint只用六格 equal mean validation AUC，test不参与；early stop 后写空 metric 的 censored rows，不前向填充。 |

### 2.3 参数分区、公式、shape/mask、梯度流

| Requirement | Result | Evidence |
|---|---|---|
| 完整互斥分区 | PASS | production `strict=True` 拒绝未知参数；111/111 MDMTRec named parameters 被完整互斥覆盖。`_weight_*` 独立为 architecture，domain_expert/task_expert分别为 domain/task-specific。每 run 保存参数名、count、shape、manifest hash和state hash。 |
| cell order与mask | PASS | 全路径统一 `idx=t*D+d`；masked BCE 为每 cell 精确 sample mean；空 cell 保留可微零项并记录 count。 |
| observer梯度路径 | PASS | 同一 forward graph 对六项执行 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`；返回 None 时零填充并记录 `unused_param_count`；observer不写 `.grad`。真实训练仍执行原始 `loss.backward(); optimizer.step()`。 |
| actual/counterfactual | PASS | 一次 raw gradient collection 同时生成 actual/equal/sample coefficient view，没有重复 forward 或额外 optimizer step。 |
| actual update alignment | PASS | main optimizer step 前后读取参数，使用 `theta_before-theta_after`；architecture 因不属于 main optimizer 而显式记录不可定义原因。 |
| routing | PASS | hook 仅 detach 后累计；cell gate只统计对应 physical domain样本，并额外输出 domain-task-average row；不改变 forward、参数或 state_dict。 |

### 2.4 数学验收与日志

本轮最新真实 ON run（seed=2022、d0_large、equal、3 epoch、batch=40960、CUDA）独立检查：

| Check | Observed | Threshold/result |
|---|---:|---|
| gradient_step rows | 450 | 3 epoch × 5 probes × 6 cells × 5 groups，PASS |
| `weighted_norm - |a|×raw_norm` max error | 0 | PASS |
| normalized contribution sum max error | `1.8012e-7` | `≤1e-5`, PASS |
| domain signed contribution reconstruction | 0 | PASS |
| shared real-backward reconstruction max relative L2 | `1.6031e-6` | `≤1e-5`, PASS |
| minimum probes per epoch/cell/group | 5 | `≥2`, PASS |
| negative finite variance count | 0 | PASS |
| finite cosine range | all within `[-1,1]` | PASS |
| metrics_epoch rows | 18 | 3 epoch × 6 cells, PASS |
| missing train/val/test AUC or LogLoss | 0 | PASS |
| nonfinite inventory | 162 listed, 0 unexplained | PASS |

non-finite 项的原因仅为两类设计允许情形：architecture group没有 main-optimizer update alignment，以及 private norm-scalar variance不定义 vector SNR。CSV中的 `NA` 与真正空白字段保持区分。

summary/report 路径已确认：固定枚举24个 expected run，保留每 seed/epoch/domain/task/split；读取并 join cell routing与domain-task-average routing；缺失 run/file 在 manifest中显式列出，不计算科研结论。

### 2.5 真实 ON/OFF 无损性复验

本轮独立重跑同一 seed/setting/weighting 的3-epoch observer OFF/ON：

| Check | Result |
|---|---|
| train + architecture trace | 18 vs 18，key/order/fingerprint/count全同 |
| per-step train loss max abs diff | 0 (`≤1e-7`) |
| architecture loss max abs diff | 0 (`≤1e-7`) |
| metrics_epoch + metrics_summary max abs diff | 0 (`≤1e-8`) |
| initial parameter hash | identical |
| final parameter state | 111 tensors逐元素相等；max abs diff=0；hash identical |
| RNG restoration | 每个 observed epoch 的 CPU+all-CUDA digest均 `before==after`，且 ON/OFF evidence完全相同 |
| comparator | `VALID`; mismatch_count=0；ignored field恰为 `is_probe` |

同一 smoke 分别送入 formal validator时均为 `False`；原因包含 epoch不是50、formal marker缺失、observed/metric cardinality不符，OFF还包含 instrument不为1。这证明 smoke 不会被 resume 当作正式完成。

## 3. Bugs/deviations found

### Blocking

无。

### Non-blocking

1. **MINOR — baseline loss scalar日志存在表达式归约舍入差。** E1使用 tensor stack/sum，历史 P0-4日志使用 Python sum；前轮对照中逐 step scalar最大差约 `4.77e-7`，但真实梯度、所有最终参数和metrics完全相同。这是日志浮点归约差，不是训练行为差异。
2. **MINOR — private-group variance口径受内存约束。** shared为完整 vector-trace Welford；private/architecture为明确标记的 `norm_scalar` dispersion，不能在后续分析中与 shared vector variance直接横比。该限制已在 implementation report如实记录，符合 DESIGN 允许分支。

## 4. Corrections made

无。reviewer 未修改任何实现代码；仅以本报告替换上一轮 FAIL 结论。

implementation report 已纠正旧的不实陈述：它明确说明旧 generic comparator 曾因 `is_probe` 返回 `INVALID`，并将该说法标为被第二轮修正取代；其最新 `compare_on_off_runs=VALID` 声明与本轮独立结果一致。

## 5. Remaining risks

- equal/sample 正式训练可能在不同 epoch early stop；当前 comparator会保留共同支持、把合法尾部标为censored。主分析必须按 common support比较，不得把未观测尾部填充或视为失败。
- formal 24-run 尚未执行，因此本报告只批准实现与执行协议，不预判运行时资源故障或任何科学方向。
- private `norm_scalar` 与 shared `vector_trace` 的统计语义不同，main analysis必须按 `variance_scope` 分层。
- oracle artifact 缺失时字段会明确为 `missing`；不得从报告均值反推 seed-level oracle。

## 6. Whether experiments are safe to run

**YES — SAFE TO RUN THE FORMAL 24-RUN MATRIX.**

批准进入严格工作流的 RUN 阶段，由 `luna-runner` 执行并只记录原始结果。推荐使用 implementation report中已审计的正式命令；runner不得在 RUN 阶段解释科研结论。

## 7. Reviewer checks executed

- `python -m py_compile`：六个 Stage E1模块/测试，exit 0。
- PyTorch环境下 pytest：`9 passed`；仅有既有 `.pytest_cache` permission warning。
- orchestrator full `--dry_run --max_runs 24`：24/24命令枚举正确，未启动训练。
- 最新真实 CUDA 3-epoch OFF/ON：两次单 run成功；专用 comparator `VALID`。
- formal validator对两份3-epoch smoke：均拒绝。
- 最新 ON raw artifact：逐项复核梯度不变量、metrics完整性、nonfinite inventory与RNG evidence。

以上均为非正式 reviewer 检查，不属于24-run正式实验，也未用于研究结论分析。
