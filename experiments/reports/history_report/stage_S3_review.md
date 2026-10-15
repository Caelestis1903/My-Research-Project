# REVIEW REPORT

Verdict: PASS

## 1. Verdict

**PASS.** 先前两个 MAJOR 均已通过最小、局部修复，并由本次复审使用真实 accepted artifacts 和独立恢复状态重放验证。Stage S3 的数学干预、控制变量、复用证据、正式矩阵、fail-fast、pair recovery、collector 结构和代表性双进程 CUDA 资源路径均满足设计要求。

本 verdict 明确授权：**无需再次取得用户批准，可自动进入 `luna_runner` 正式执行 Stage S3。**

本次复审未修改任何实现代码、测试、S1/S2/M3OE 文件或既有实验产物；未启动正式训练或科研分析。本文件是唯一更新的审核产物。

## 2. Requirement-by-Requirement Verification

1. **PASS — exact 16-key fresh matrix.** 2 settings × 2 objectives × seeds 2022–2025，共 16 个唯一 `TARGET_ONLY_SHARED` M3OE joint keys；不存在额外 setting/model/access condition。dry-run queue 为 16 tasks、8 pairs、0 formal launches。
2. **PASS — exact 40-key reuse inventory.** 独立重新运行完整 read-only audit：40/40 `REUSE_OK`、0 rejected；组成精确为 8 S1 Sample FULL、8 S2 Equal FULL、24 S1 D=1 references。prediction-level AUC/LogLoss、artifact hashes、checkpoint finite 与 block pairing 均重新核验。
3. **PASS — objective formula and coefficients.** Sample 为 `w_d=1`；Equal 为冻结 full-training share 的 `1/(3p_d)`；target component 保留原 batch denominator `B`。没有 sampler、target-count normalization、observed minibatch share、重复 `p_d`、gradient norm scaling 或 coefficient exposure 变化。
4. **PASS — exhaustive C/P/A_arch partition.** `domain_expert.*`、`task_expert.*` 及 embedding/shared/skip/star/shared experts 属于 C；slot/gate/tower 属于 P；四个 architecture `deep_weights` 属于 A_arch。KuaiRand 结构为 80/744,848 + 54/869,406 + 4/4 = 1,614,258；MovieLens 为 53/566,080 + 54/207,390 + 4/4 = 773,474。
5. **PASS — real-model reachability and corruption rejection.** 两个真实 schema 均通过跨域 reachability；独立将 `tower.0.*` 故意误分类为 shared 后 audit 明确拒绝，证明 reachability test 不只是名称或计数检查。
6. **PASS — loss decomposition and gradient substitution.** `L_full=L_target+L_non`；C 只安装 `grad(L_target)`，P 安装 `grad(L_full)`，A_arch 只安装 `grad(L_target)`；removed non-target gradients 仅记录 norm，不进入 optimizer。一次 batch 只有一次 forward 与一次对应 optimizer step。
7. **PASS — real FULL compatibility.** 真实 KuaiRand M3OE 上，S3 FULL 与 S2/direct backward 的 main loss、一步参数、Adam state 精确一致；architecture FULL 的 loss、一步参数、Adam state 也精确一致。相同 fixture batch fingerprint 为 `413f51d2bf24758d9e7e8d46d57dece2cb19086bdd8f93f943dbef26b6d98f5a`。
8. **PASS — fresh/reused controlled pairing.** pair validator 覆盖 dataset bytes、splits/views、initial state、model manifest/count/FLOPs、optimizer、exposure、step counts、batch hashes、metric/checkpoint/runtime contracts；正式 child 只有通过自身 validator 和同-objective FULL pairing 才能 promotion。objective/access identity 是唯一允许差异。
9. **PASS — baseline/model preservation.** S3 仅新增隔离层；未改变 M3OE forward、capacity、S1/S2 baseline、optimizer/scheduler/sampler，也未引入 momentum、clipping、projection、freezing 或新模型路径。复用审计确认冻结文件/产物一致。
10. **PASS — checkpoint/test discipline.** validation-only six-cell macro-AUC selection、earliest tie break、deferred test、selected/final/common 9/19/29/39/49 全部保留；validator 要求 50 epochs、完整 artifacts、prediction-level recomputation 与 final/common checkpoint contract。
11. **PASS — two-process queue.** 8 个 reciprocal same-setting/same-seed cross-objective pairs，固定 `max_workers=2`，wave-specific output isolation，parent queue atomic replace，且只允许 pair promotion。
12. **PASS — append-only pair recovery.** mixed `[success, retryable failure]` 与双 retryable 均整对归档并让两个 siblings 在全新 wave-2 directories 从 epoch 0 重跑；OOM/numerical、protocol、validator、zero-exit-invalid 等为 fatal；wave-2 失败整对归档且无第三次尝试；成功时双成员一起 promotion。
13. **PASS — full test/smoke set.** `EDDAenv` 下 S3 suite 20/20 通过；包括 objective/access、real reachability、corruption、真实 manifest collector、恢复状态矩阵。真实 TARGET_ONLY CPU main+architecture 更新通过；代表性 batch size 40,960 双进程 CUDA smoke 2/2 return code 0，各执行恰好 1 main + 1 architecture step，未生成 formal result。
14. **PASS — scope/report/immutability.** implementation report 准确列出修复文件和验证范围；修复仅涉及 S3 collector、orchestrator、对应测试与报告。S1/S2/M3OE source/artifacts 未修改。正式 S3 raw artifact 文件数仍为 0。
15. **PASS — real collector contract.** 使用 40 个真实 accepted S1/S2 sources，加仅位于 `TemporaryDirectory` 的 16 个 fresh-shaped fixtures，完整 collector 得到精确：raw 1,680、direct contrasts 2,688、interactions 336、inventory 56。list-shaped 与 Mapping-shaped domain maps 都正确；missing/wrong-length/task/pair identity corruption 均 fail-closed。
16. **PASS — run authorization.** 所有 blocking acceptance 均通过；允许 orchestrator 自动调用 `luna_runner`，不再暂停等待用户批准。

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- `prediction/target=[B,2]`、`domain_ids=[B]` 和 domain range 均 fail-fast 检查。
- target loss 为 `(w[d_i] * per_example_i * 1[d_i=d*]).mean()`，没有按 target count 重归一。
- task-major `t*D+d` 与 private gate/tower 的 `index % D` owner 对齐；domain/task experts 因真实 forward 混合被正确归为跨域参数。
- interaction 的 AUC 与 `Q=-LogLoss` 方向公式可从 raw rows 重建；collector 唯一性与 cardinality 检查通过。

### Gradient flow & Loss participation

- main update 从同一 forward graph 获得 target-shared、diagnostic non-target-shared 与 full-private gradients，只执行一次 main step。
- architecture update从同一 forward 只给四个 A_arch scalars 安装 target gradients，只执行一次 architecture step。
- `.detach().clone()` 仅用于写入已计算 gradient；未新增 activation detach。模型原有 gate-input detach 保持不变。
- prediction、loss、gradient、parameter 都有 nonfinite 检查；CPU/CUDA smoke 的 shared/private/architecture gradient norms 均有限且非零。

### Controlled variables & Baseline preservation

- 固定 50 epochs、batch size 40,960、ordinary shuffled concatenated loader、`drop_last=False`、`num_workers=0`。
- Adam main/architecture 分别使用 lr 0.01/0.001、betas 0.9/0.999、weight decay 1e-5，无 scheduler。
- 同 objective/setting/seed 的 fresh 与 FULL source 要求相同初始化、batch orders、exposure、steps、dataset/split/view、metric/checkpoint contract。
- 40/40 source artifacts 已重新完成不可变哈希和 prediction-level metric 审计。

### Fail-fast enforcement

- 未发现 silent config fallback、自动 tensor reshape/broadcast 修复、broad exception suppression、singleton repair、concurrency/matrix 自动改变或 mid-run resume。
- list/Mapping domain schema 的 missing、wrong length、empty/ambiguous index、D=1 task mismatch、pair identity mismatch 都明确抛错。
- pair failure 只按失败成员分类；验证失败不能伪装成 transient success，fatal 状态立即停止。

## 4. Over-Engineering & Code Bloat Audit

修复是局部的：一个显式 list-or-mapping domain accessor 和一个 pair recovery helper/state classifier，均直接服务于已冻结的实验正确性与 fail-fast。未引入注册表、框架层、生产监控或未来兼容抽象。测试使用 injected child outcomes 与临时目录，不触碰正式路径。

## 5. Bugs / Deviations Found

本轮未发现 CRITICAL、MAJOR 或 MINOR 未解决问题。

先前 MAJOR 状态：

- **RESOLVED — real manifest physical mapping.** 真实 KuaiRand joint local domain 1 与 D=1 reference 均解析为 `(1, "tab1")`；Mapping equivalent 同样通过；corrupt forms fail-closed；全 collector 临时重放达到 1,680/2,688/336/56。
- **RESOLVED — mixed pair recovery.** 成功 sibling 不再使 pair fatal；mixed/double retryable、fatal/OOM/protocol/validator、wave-2 failure、pair archive、fresh wave-2 directories 和 atomic pair promotion 均由独立执行覆盖。

## 6. Corrections Made

Reviewer 未修改实现。Coder 的两处最小修复经独立复审接受。

## 7. Remaining Risks

- 正式运行尚未发生，因此真实 16 fresh runs 的训练稳定性、时间和显存表现仍属于运行阶段风险；pair-level fail-fast/retry 策略已就绪。
- 本 PASS 仅证明实验按设计安全可运行，不预判任何科研结论。
- 正式 run 完成后必须进行独立 post-run review：验证 16 fresh + 40 reused artifacts、hashes、prediction-level metrics、batch traces、1,680/2,688/336/56 cardinalities、direct deltas/interactions 和机械 aggregates，之后才进入 Main analysis。

## 8. Whether Experiments Are Safe to Run

**Yes.** Stage S3 可安全进入 `luna_runner`。根据用户当前指令，**不需要额外批准停顿**；runner 应立即按冻结 8-pair/16-run 队列执行 seeds 2022、2023、2024、2025。若任一 fatal validator/协议/数值异常出现，必须按已审核策略 fail-fast，不得改变矩阵或并发。

## Independent execution evidence

- Environment: `C:\Users\86199\.conda\envs\EDDAenv\python.exe`
- S3 unit/regression suite: 20/20 passed.
- Fresh read-only reuse audit: 40/40 valid, 0 rejected.
- Real collector replay: list joint `(1, tab1)`, D=1 `(1, tab1)`, Mapping compatible, corruption rejected; cardinalities 1,680 / 2,688 / 336 / 56.
- Recovery replay: mixed success+retryable, dual retryable, fatal OOM/protocol/validator, wave-2 failure, full-pair archive and pair-only promotion all passed.
- Real M3OE reachability: KuaiRand 1,614,258 and MovieLens 773,474 parameters; deliberate tower-as-shared corruption rejected.
- Real FULL compatibility: main and architecture loss/parameters/Adam state exact-match.
- TARGET_ONLY real-model CPU: Equal and Sample each completed exactly one main and one architecture step with finite nonzero norms.
- Representative two-process CUDA: PASS, return codes `[0,0]`, batch size 40,960 each, one main + one architecture step each.
- Queue remains `DRY_RUN`, 16 tasks / 8 pairs, `launched=0`, `formal_launches=0`, `resume_state=PLANNED`; official Stage S3 raw artifacts remain 0.
