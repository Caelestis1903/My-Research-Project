# DESIGN SPECIFICATION — Stage E6：Adaptive Domain Contribution Calibration

## 0. Scope, ordering, and evidence gate

本次 DESIGN 严格只读取了提示词文件 `promt/验证性补全和实验方向寻找/group3.md`。没有读取、搜索、打开或委托其他代理读取其他 prompt 文件。

本规格只设计 group3 内部顺序当前项 E6；不设计、不实现、不运行 E7。E5 已正式得到 `STOP — DROP Curriculum`，所以 E6 不包含任何 Curriculum baseline、warmup 或 schedule，且不得用 E6 重新包装 E5 的负结果。

E6 的开发动机来自已经完成的正式证据，而不是预设 adaptive 或 EMA 有效：

- E1：sample weighting 在 12/12 setting×seed pairs 提高 overall AUC（均值约 `+0.0211`），并把 shared contribution mismatch 从约 `0.640` 降至 `0.133`；但 sample 下 L/M/S magnitude shares 约 `0.696/0.241/0.062`，仍未等于 data shares `0.800/0.184/0.016`。
- E3：`alpha=1`（sample）是 overall endpoint 最佳或实际等价最佳 static；shared mismatch 随 alpha 下降，在 `alpha=1` 仍约 `0.111`。没有固定 interior alpha 稳定胜过 sample，因此不再手调 alpha。
- E4：KuaiRand 上 sample 相对 equal 提高 overall AUC、worst-domain AUC 与 dominant-domain transfer，shared mismatch 从约 `0.711` 降到 `0.084`，说明 contribution allocation 的方向可跨数据集出现；但 sample 仍留有 residual negative transfer。
- E5：预注册 Forward schedule 未稳定超过 sample，且 gradient variance/SNR 仅是 seed-dependent 的竞争解释。故 training stage 不进入 controller；variance/SNR 只记录，不作为首版输入。

这些结果只说明“把实际 contribution 更接近 data exposure”是可证伪的机制候选。它们不证明精确匹配 data share 最优，也不证明 EMA 必需。

## 1. Research question

在 MovieLens-1M、fixed-N r50、canonical `d0_large`、MDMTRec/M3OE、自然采样不变时：

> 使用仅依赖训练 batch 的 shared raw-gradient magnitude，在线修正 domain loss coefficients，使实际 shared magnitude contribution 朝 domain data share 靠近，能否稳定超过简单 sample weighting，并减少 asymmetric negative transfer？

需要独立回答：

1. 动态 state 是否在 sample size prior 之上产生增益；
2. data-share target 是否必要，还是任何 contribution equalization 都可产生类似结果；
3. EMA 是否改善 estimator/controller 稳定性，还是无效甚至有害；
4. 改善是 overall AUC、worst-domain、Large/Small transfer 或 negative-transfer robustness 中哪一种；
5. 增益是否跨至少 3/4 seeds，且不由单个 domain/task 驱动；
6. domain size 与 oracle operational difficulty 是否提供竞争解释。Small 不得等同 difficult。

## 2. Hypotheses being tested

### 2.1 Primary mechanism hypothesis H-C

Static sample coefficients `c_d=Dp_d` 不能补偿不同 domain 的当前 raw shared-gradient magnitude。使用训练内 raw-gradient state 计算校准系数，使 coefficient-adjusted shared magnitude share 更接近 `p_d`，将比 static sample 提高 performance 或 transfer robustness。

### 2.2 EMA hypothesis H-E

若 instantaneous raw-gradient magnitude 的 batch noise 会引起 coefficient 抖动，则用一 epoch half-life 的 EMA 平滑该训练统计，将比相同规则的 NoEMA 版本获得更低 coefficient variation，并在至少 3/4 seeds 提供可重复的 performance/robustness 增益。

H-E 不预设成立。EMA 仅是 temporal estimator，不是 optimizer momentum、parameter EMA 或 checkpoint ensemble。

### 2.3 Null hypotheses

- **H0-C**：Adaptive 与 sample/static alpha 实际等价，或结果 seed/domain/task 不稳定；残余 mismatch 不是值得校准的性能瓶颈。
- **H0-E**：EMA 仅使曲线更平滑但不改善 AUC、LogLoss 或 transfer；EMA unnecessary。
- **H0-size**：data-share target 不优于 uniform target，说明 domain size prior 不是 controller 的必要信息。

### 2.4 Competing explanations

- **C1 — sample already sufficient**：E3 的 static alpha=1 已达到可用 optimum，进一步追踪 contribution 只增加方差/复杂度。
- **C2 — raw-gradient norm is not causal**：magnitude share 是描述量；gradient direction/cosine 或 private/routing path 才决定 transfer。
- **C3 — optimizer/instrumentation artifact**：额外 `autograd.grad` 改变 `.grad`、RNG、architecture path 或数值轨迹。matched EMA control 专门检验此项。
- **C4 — EMA lag**：平滑 state 在快速变化阶段滞后，NoEMA 反而更好。
- **C5 — size prior wrong target**：真实最优 contribution 不应等于 sample share；uniform-target ablation 可能等价或更优。
- **C6 — single-domain/task redistribution**：overall gain 来自牺牲另一 domain/task，或只修复一个 cell。
- **C7 — difficulty/identity**：canonical d0/d1/d2 的 oracle difficulty 与 size role 混杂。单 setting E6 不能证明 size 是唯一主因。
- **C8 — checkpoint/convergence**：adaptive 改变 peak epoch而非最终轨迹；共同 epoch/last-common comparisons 需保留。
- **C9 — noise/SNR**：若性能变化跟随 variance/SNR 而非 calibration error，首版 controller 的状态选择不充分。

## 3. Variables changed

### 3.1 Formal arm registry

仅比较下列六个 unique arms：

| arm key | coefficient behavior | dynamic statistic | purpose |
|---|---|---|---|
| `equal_static` | `c_d=1` | none | original equal baseline |
| `sample_static` | `c_d=Dp_d` | none | key baseline；同时是 E3 best static `alpha=1` |
| `sample_ema_control` | `c_d=Dp_d` | 维护与 Adaptive-EMA 相同的 EMA，但不反馈到 loss | matched constant EMA/control-overhead baseline |
| `adaptive_noema` | data-share calibration | current-batch raw shared norm | without EMA |
| `adaptive_ema` | data-share calibration | EMA raw shared norm | full candidate |
| `adaptive_ema_uniform` | uniform-share calibration | EMA raw shared norm | without size prior |

E3 的 best static alpha 就是 `alpha=1`，与 sample weighting 完全相同，因此由同一个 `sample_static` artifact 同时满足两个 baseline 语义，不运行伪重复。E5 Curriculum 已 DROP，不加入第七臂。

### 3.2 Formal matrix

- dataset/ratio/setting：MovieLens-1M × r50 × canonical `d0_large`：1
- arms：6
- seeds：2022、2023、2024、2025：4
- 正式 joint runs：`1 × 6 × 4 = 24`

Transfer 复用 E3 canonical setting 的 `3 domains × 4 seeds = 12` 个 frozen oracle references；它们不计为 E6 新 runs。不得在看到 E6 test 后增加 target、EMA decay、clip、ratio 或 schedule arm。

## 4. Variables held constant

全部 24 runs 固定为 E3 r50 canonical joint protocol：

- MovieLens-1M；fixed training N=187,500；
- `d0_large` role mapping：d0=L、d1=M、d2=S；role counts `150000/34500/3000`，shares `0.800/0.184/0.016`；
- 已有 chronological train/validation/test split、seed-specific subsample 与 feature processing；
- model：MDMTRec/M3OE，`backbone=mdmoe`，D=3，T=2，model/expert/tower dimensions 不变；
- main Adam 与 architecture Adam 的 lr、betas、weight decay、parameter grouping 不变；
- batch size=40960、num_workers=0、自然 sampling/shuffle 与 architecture-update cadence 不变；
- upper bound=50 epochs、early-stop patience=10、validation six-cell macro AUC checkpoint rule不变；
- task 始终等权；controller 只输出 domain-level coefficient 并复制给两个 tasks；
- architecture loss 保持 E3 unmasked objective，不使用 controller coefficient；
- 无 Curriculum、training-stage input、optimizer momentum modification、parameter EMA、adaptive sampler、新 router 或 validation/test feedback；
- test 不参与 controller、checkpoint 或 early stop；
- 同一 seed 六 arms 的 initialization、split rows、train/architecture batch order、model/optimizer config一致；合法 early-stop 差异只允许 suffix censoring，共同支持内必须一致。

## 5. Mathematical and algorithmic definition

### 5.1 Cell loss and raw shared state

cell order 固定为 `idx=task*D+domain`。对 main train batch：

\[
\ell_{dt}(\theta)=\frac{1}{|B_d|}\sum_{i\in B_d}
\operatorname{BCE}(\hat y_{i,t},y_{i,t}).
\]

shared parameter partition 必须复用 E1/E3 已审查定义。先用同一次 forward 的 cell losses 计算：

\[
g^S_{dt,k}=\nabla_{\theta_S}\ell_{dt,k},
\qquad
r_{d,k}=\sum_{t=1}^{T}\|g^S_{dt,k}\|_2.
\]

`r_d` 是唯一动态 controller input。它使用当前 train batch、当前模型、pre-update state；必须 `detach`，不得产生二阶梯度。domain share：

\[
p_d=n_d/\sum_j n_j
\]

是唯一静态输入。

不使用 validation/test、oracle transfer、AUC、loss trend、routing、training stage、gradient variance 或 SNR 作为 controller input。variance/SNR 因 E5 仅有竞争性、非稳定证据，只作为诊断日志。

### 5.2 Target contribution

Full candidate 与 NoEMA 的 target 为训练 data share：

\[
\tau_d=p_d.
\]

依据是 E1/E3/E4 中 sample weighting 相对 equal 的稳定改善及 mismatch 减少，但 residual mismatch 仍存在。该 target 是待证伪假设，不保证最优。

Without-size-prior ablation 使用：

\[
\tau_d=1/D.
\]

它保持完全相同动态 state 与 EMA，只移除 size information。不得使用 oracle difficulty 构造 target，因为这会引入 evaluation leakage 并把 size 与 difficulty 混为一谈。

### 5.3 NoEMA calibration rule

令 `q_{d,k}=r_{d,k}`。数值稳定项：

\[
\epsilon_k=\operatorname{finfo}(q.dtype).eps\cdot
\max(1,\operatorname{mean}_j q_{j,k}).
\]

先计算 unnormalized inverse-magnitude correction：

\[
z_{d,k}=\frac{\tau_d}{q_{d,k}+\epsilon_k},
\]

再固定总 loss scale：

\[
c_{d,k}=D\frac{z_{d,k}}{\sum_j z_{j,k}}.
\]

对两个 tasks 复制相同 `c_d`，main objective：

\[
L_k=\sum_d\sum_t c_{d,k}\ell_{dt,k}.
\]

若 `epsilon` 可忽略，则 coefficient-adjusted magnitude share 满足：

\[
\frac{c_{d,k}r_{d,k}}{\sum_j c_{j,k}r_{j,k}}\approx\tau_d.
\]

该 closed-form rule 无 controller learning rate、手调 alpha 或额外 optimizer。归一化保证 `c_d>0`、`sum_d c_d=D`、`sum_{d,t}c_d=DT=6`，自然把单 domain coefficient 限在 `(0,D)`。

若任一 domain 在 formal train batch 缺失、`r_d` 非有限或所有 `r_d` 为零，该 run 必须 FAIL 并记录原因；不得用 validation、previous best 或人工 coefficient 静默替代。

### 5.4 EMA calibration rule

EMA 只平滑 raw-gradient magnitude。每个 run 第一 main batch：

\[
\hat r_{d,0}=r_{d,0}.
\]

之后：

\[
\hat r_{d,k}=\beta\hat r_{d,k-1}+(1-\beta)r_{d,k}.
\]

其中 decay 不搜索，按一 epoch half-life 预注册：

\[
S=\lceil N/B\rceil=5,\qquad \beta=2^{-1/S}\approx0.8705506.
\]

Adaptive-EMA 与 uniform ablation 令 `q_d=hat r_d` 后使用同一 closed-form calibration。EMA state 是 detached float64 training statistic；不进入 state_dict parameters、不接受梯度、不更新 optimizer，不对模型参数做平均。

`sample_ema_control` 计算完全相同的 `r_d` 与 `hat r_d`，但 main loss 始终使用 `c_d=Dp_d`。它检验额外 autograd/EMA bookkeeping 是否改变训练结果，并不是 matched optimizer EMA。

### 5.5 Update order and gradient-chain invariant

每个 main batch 严格顺序：

1. single model forward；
2. 构造六 cell masked-mean losses；
3. 对 shared parameters 调用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` 得到 raw cell gradients；
4. detach norms，更新或读取 controller statistic；
5. 计算并 detach `c_d`；
6. 构造一次 `L_k`，执行原 main `backward()` 与一次 optimizer step；
7. architecture update 按原 cadence 和原 objective 单独执行。

`autograd.grad` 不得写入或累加 `.grad`。系数不能反向传播进 `r_d`，否则变成未设计的 second-order rule。实际 main gradients必须与 detached-coefficient weighted loss 一致。

### 5.6 Calibration and stability metrics

对当前 raw state 与应用 coefficient：

\[
s_{d,k}=\frac{c_{d,k}r_{d,k}}{\sum_j c_{j,k}r_{j,k}+\epsilon},
\]

\[
M_k=\frac12\sum_d|s_{d,k}-p_d|,
\qquad
E_k(\tau)=\frac12\sum_d|s_{d,k}-\tau_d|.
\]

EMA tracking error：`|hat_r_d-r_d|/(r_d+epsilon)`。Coefficient variation：

\[
V_c=\frac{1}{K-1}\sum_{k>0}\|c_k-c_{k-1}\|_1/D.
\]

同时记录 raw/weighted norm、signed contribution、actual-update alignment、gradient variance/SNR。低 `E_k` 只证明 controller 做到了定义目标，不证明 performance 改善。

### 5.7 Performance, transfer, size and difficulty

每 epoch、每 domain、每 task 记录 train/validation/test AUC 与 exact sample-mean LogLoss。selected checkpoint 只由 validation six-cell macro AUC 决定，tie 取 earliest epoch。

汇总：overall macro AUC/LogLoss、per-domain、per-task、L/M/S、worst-domain AUC、worst-cell AUC、peak/best/last-common epoch、trajectory。

使用 E3 frozen oracle：

\[
Transfer_{d,t}=AUC^{joint}_{d,t}-AUC^{oracle}_{d,t}.
\]

报告 Large/Medium/Small transfer 与 negative-transfer rate。Operational difficulty 仅由 matching oracle AUC/LogLoss 描述，domain size 由 train count/share 描述；不得因为 d2=S 就直接称其 difficult。因 E6 只有 `d0_large`，size 与 physical identity/difficulty 仍不能完全分离。

## 6. Exact affected modules/files

### 6.1 New implementation files

- `experiments/stage_E6_calibration.py`
  - arm registry；raw domain state；NoEMA/EMA state；target 与 coefficient closed form；calibration metrics；block validation。
- `experiments/stage_E6_run.py`
  - 单个 `(arm,seed)` 训练入口；controller 只接 main loss。
- `experiments/stage_E6_orchestrator.py`
  - 固定 24-run resume-safe matrix 与 E3 oracle resolver。
- `experiments/stage_E6_report.py`
  - raw/seed-level structural summary，不作科研结论。
- `experiments/tests/test_stage_E6_calibration.py`
  - 公式、EMA、detach、loss-gradient 与 target tracking focused tests。
- `experiments/tests/test_stage_E6_matrix.py`
  - 24-run matrix、identity、artifact completeness 与 matched-control tests。

### 6.2 Read/reuse without unnecessary edits

- `experiments/stage_E3_weighting.py`
- `experiments/stage_E3_run.py`
- `experiments/stage_E3_orchestrator.py`
- `experiments/stage_E3_raw_results/oracle_reference/`
- `experiments/stage_E1_instrumentation.py`
- `m3oe/MDMTRec/models/multi_domain/model.py`
- `m3oe/MDMTRec/utils/data.py`
- `m3oe/MDMTRec/trainers/ctr_trainer.py`

优先隔离新实现；不得为 controller 修改核心 model forward、sampler 或 router。

### 6.3 Required artifacts

- `experiments/reports/stage_E6_design.md`（本文件，authoritative design）
- `experiments/reports/stage_E6_implementation.md`
- `experiments/reports/stage_E6_review.md`
- `experiments/stage_E6_raw_results/`
- `experiments/stage_E6_ablation.csv`
- `experiments/stage_E6_summary.csv`
- `experiments/stage_E6_summary_manifest.json`
- `experiments/reports/stage_E6_run_report.md`
- `experiments/reports/stage_E6_analysis.md`（仅 main analysis 写）
- `experiments/reports/stage_E6_go_no_go.md`（仅 main analysis 写）

## 7. Implementation requirements

1. 六 arm registry 封闭；formal CLI 不接受额外 target、beta、clip、alpha 或 schedule。
2. `sample_static` 同时是 sample 与 best static alpha=1；不得伪造重复 arm 来扩大样本数。
3. `p_d` 必须从审计通过的实际 train counts 计算；formal counts 必须为 150000/34500/3000。
4. Controller 每个 main batch 使用当前 forward 的 shared raw cell gradients；不得使用 previous epoch validation/test/oracle。
5. domain raw magnitude 是两个 task raw shared-gradient L2 norms 之和；task coefficient 保持相同。
6. NoEMA、EMA 与 uniform ablation 只在 `q_d` 或 `tau_d` 指定处不同，其余代码路径完全相同。
7. beta 精确由 `2**(-1/steps_per_epoch)` 计算并写入 config；不得搜索或 seed-specific 调整。
8. coefficient 必须 detached 且 sum-domain=D；main loss只 backward/step 一次；controller 无可训练参数。
9. `sample_ema_control` 执行同样的 autograd/EMA state 更新但始终应用 `Dp_d`；plain sample 与 matched control 的差异必须完整报告。
10. architecture loss、optimizer、sampling、routing、batch cadence 与 E3 完全不变。
11. instrumentation 记录实际应用 coefficient与 post-hoc reconstruction；不得只记录 controller proposal 而 loss 用另一向量。
12. test evaluation 保持/恢复 CPU/CUDA RNG，且不影响后续训练；复用既有实现，不重复大规模防御性验证。
13. oracle resolver 只接受 formal-valid、seed/setting/domain/task/split identity 完整的 E3 artifacts；Markdown 均值不是 oracle。
14. report 固定枚举 24 expected paths，summary保留 seed/cell rows；缺失、非法 early-stop 尾部或 identity mismatch 显式 INVALID。
15. coder 只做 syntax、focused unit tests 和一个短 smoke；不得启动 24-run matrix，不得分析结果。
16. 不实现 E7、Curriculum、parameter EMA、optimizer momentum modification、adaptive sampling、多输入 learned controller 或手调 alpha。

## 8. Ablations required

最小必要消融与映射：

1. **without dynamic state**：`sample_static` vs adaptive；
2. **matched estimator overhead**：`sample_static` vs `sample_ema_control`；
3. **without size prior**：`adaptive_ema_uniform` vs `adaptive_ema`；
4. **without EMA**：`adaptive_noema` vs `adaptive_ema`；
5. **static alpha replacement**：`sample_static`（E3 best alpha=1）vs adaptive；
6. **original baseline**：`equal_static`，只用于定位旧基线，不能作为新方法关键胜利条件；
7. selected checkpoint、last-common 与 common-support trajectory；
8. per-domain/per-task、size-role 与 oracle-difficulty views；
9. calibration error vs performance、coefficient variation vs EMA gain、variance/SNR competing diagnostics。

不得增加 dynamic-state combinations、EMA decay sweep、target blend、clip sweep 或 Curriculum arm。

## 9. Expected logging requirements

每 run：`experiments/stage_E6_raw_results/<arm_key>/d0_large/seed<seed>/`，至少包含：

- `config.json`
- `run_manifest.json`
- `stdout.log`
- `metrics_epoch.csv`
- `metrics_summary.csv`
- `controller_step.csv`
- `controller_epoch.csv`
- `shared_gradient_step.csv`
- `shared_gradient_epoch.csv`
- `batch_trace.csv`
- `nonfinite_inventory.json`

`controller_step.csv` 每 main batch 至少记录：seed/arm/epoch/step、batch fingerprint、domain counts、`p_d`、`tau_d`、raw `r_d`、EMA `hat_r_d`（适用时）、beta、proposed/applied `c_d`、coefficient sum、actual magnitude share、target/data-share calibration error、EMA tracking error、coefficient variation、fallback/error flag。

`shared_gradient_step/epoch.csv`：raw/weighted norm与RMS、signed contribution、magnitude share、mismatch、update alignment、variance/SNR、probe/count；必须能按 domain/task重建 applied weighted gradient。

`metrics_epoch.csv` 每 `(epoch,domain,task)`：train/validation/test count、positive rate、AUC、LogLoss、physical domain、size role、domain size/share、oracle AUC/LogLoss/source、transfer、selected/censored flag。

`config/manifest`：arm/target/controller formula/version hash、beta derivation、controller inputs whitelist、dataset/model/optimizer/sampling identity、initial/final hash、split/batch fingerprints、best/stopped/censored epoch、controller update count、coefficient extrema/variation、oracle coverage、RNG evidence。

`stage_E6_ablation.csv` 与 `stage_E6_summary.csv` 必须是 seed-level/cell-level raw summaries，不写“有效”等科研结论。至少包含 AUC、LogLoss、overall/worst-domain、per-domain/task、L/M/S transfer、negative-transfer、calibration error、coefficient variation、variance/SNR、best/last-common epoch。

Runner 只记录 raw results、完整性与失败原因，不解释优劣。

## 10. Acceptance criteria for reviewer

Reviewer 重点是算法正确性，不做无边界防御验证。只有以下均满足才可 PASS：

1. toy gradients 下 `r_d=sum_t ||g_dt||` 与手算一致；
2. NoEMA closed form 的 coefficient 正、sum为D，且应用后的 magnitude share 在数值容差内等于 target；
3. EMA 初值、更新式与 `beta=2^(-1/5)` 精确；NoEMA/EMA 除 state 外一致；
4. uniform ablation 只把 target 从 `p` 改为 `1/D`，没有移除 dynamic state；
5. sample EMA control 维护相同统计但 loss coefficient 始终严格 `Dp`；
6. 当前 batch coefficient 确实作用于 main loss/backward，且 architecture loss/sampler未改变；
7. coefficient 和 raw state detached，无 second-order chain，`autograd.grad` 不污染 `.grad`，每 batch仅一次 main optimizer step；
8. 一个短 deterministic smoke 证明 logged applied coefficient 等于 loss 实际 coefficient，并证明 plain sample 与 matched control 在相同 batch上训练语义一致；无需重做 E1 全套防御检查；
9. 同 seed arm identity、common batch order、合法 early-stop尾部 comparator正确；
10. validation/test/oracle没有进入 controller，test不参与checkpoint；
11. per-domain/per-task AUC、LogLoss、transfer、size、difficulty 与 controller trajectory日志齐全；
12. orchestrator精确枚举24 runs和seeds 2022–2025；无Curriculum/E7/额外grid；
13. implementation report如实记录算法路径、测试、限制和偏离。

以下为 blocking FAIL：controller proposal未实际进入loss；EMA作用于parameters/optimizer；采样被改变；使用validation/test/oracle state；gradient chain未detach；uniform ablation同时改变其他组件；或矩阵超出/缺少预注册arms。

## 11. Conditions that would falsify the hypotheses

### 11.1 Falsify H-C / method justification

任一情形足以拒绝“adaptive calibration 值得作为复杂方法”：

1. `adaptive_noema` 与 `adaptive_ema` 均未相对 `sample_static` 达到至少3/4 seeds同向的预注册性能或robustness优势；
2. adaptive overall AUC 与 sample 实际等价（绝对mean delta<=0.001），worst-domain/Large transfer/negative-transfer也无明确补偿；
3. adaptive calibration error显著下降，但AUC/LogLoss/transfer不改善，说明精确匹配data share不是足够机制；
4. gain仅来自一个domain/task，另一个domain平均退化超过0.001；
5. `adaptive_ema_uniform` 等于或稳定优于data-share target，反驳size prior必要性；
6. `sample_ema_control` 与plain sample出现实质性能差异，说明额外计算/数值路径而非controller可能造成结果；
7. adaptive只改变peak epoch，在last-common/common-support上优势消失；
8. coefficient variation、variance/SNR或gradient direction比calibration error更一致地解释结果，当前state选择不充分；
9. seed方向不稳定或反例超过1/4。

### 11.2 Falsify H-E / EMA necessity

- EMA只降低coefficient variation但相对NoEMA的关键指标mean gain<=0.001或不足3/4 seeds；
- NoEMA通过method gate而EMA无额外实际增益：EMA unnecessary；
- EMA因lag在至少3/4 seeds比NoEMA差超过0.001：EMA harmful under this protocol；
- 两者都失败时，不能把EMA平滑曲线包装为方法贡献。

## 12. Go / Conditional / Stop rules

seed是唯一replication unit。AUC/transfer与LogLoss practical margin均为0.001；报告paired mean、sample std、median、min/max、improved seeds/4、反例，可报告seed bootstrap CI但不以p-value单独判定。

### GO — Method justified for E7

`adaptive_noema` 或 `adaptive_ema` 至少一个必须同时满足：

- selected-checkpoint overall macro AUC 相对 `sample_static` mean gain `>0.001`；
- 至少3/4 seeds为正；
- 相对 `sample_ema_control` 方向一致，排除observer/EMA bookkeeping；
- overall LogLoss不恶化超过0.001；
- 至少两个domain的mean AUC/transfer不为负，且任一其余domain退化不超过0.001；
- worst-domain AUC、Large transfer、Small transfer均无超过0.001的平均退化；
- 优势不只来自单个task；
- calibration target确实被实现，且common-support/last-common不与selected结论实质矛盾。

通过后进入E7的算法必须冻结为通过gate的单一版本；若NoEMA与EMA都通过，按下述EMA判据决定更简单版本，不得看E7结果再选。

### CONDITIONAL — robustness/transfer only

若 overall AUC 相对sample的绝对mean delta `<=0.001`，但至少一项满足：

- worst-domain AUC gain `>0.001` 且至少3/4 seeds为正；或
- Large transfer gain `>0.001` 且至少3/4 seeds为正；或
- negative-transfer rate在至少3/4 seeds不增加且聚合至少减少一个cell比例单位（`1/6`），

同时 LogLoss、Small transfer与其余domains无超过0.001退化，则可以 CONDITIONAL 进入E7，但论文目标必须明确为 `transfer allocation / robustness`，不能宣称average AUC SOTA。

### STOP — Method too complex for gain

若Adaptive无法通过GO或CONDITIONAL，或只胜过equal而未胜过sample/static alpha，则停止复杂算法开发，不增加input、learned controller、EMA decay、clip、target blend或schedule。

### EMA conclusion

- **EMA necessary**：Adaptive-EMA通过GO/CONDITIONAL，NoEMA不通过，并且EMA-NoEMA关键指标gain>0.001且至少3/4 seeds为正。
- **EMA unnecessary**：NoEMA通过，EMA相对NoEMA未超过0.001或不足3/4 seeds；冻结NoEMA。
- **EMA harmful**：EMA相对NoEMA下降>0.001且至少3/4 seeds同向。
- 若两者均失败：写 `EMA not justified because the adaptive method failed`，不得仅凭平滑度声称必要。

Curriculum结论固定为 **Curriculum unnecessary / dropped by E5**，E6不得重新打开。

## 13. Required main-analysis reporting

Main analysis 必须独立报告：

- AUC、LogLoss、overall、worst-domain、per-domain、per-task；
- L/M/S AUC与transfer、negative-transfer rate；
- four-seed paired effect、std、direction consistency与反例；
- domain size/count/share与frozen-oracle operational difficulty；明确Small不等于difficult；
- equal、sample/best-static、matched EMA control、NoEMA、EMA、without-size-prior baselines；
- MDMTRec/M3OE model identity、selected/best/last-common training stage与trajectory；
- applied coefficient、magnitude share、target/data mismatch、signed contribution、variance/SNR、coefficient variation与EMA lag；
- stable repeated phenomena、contradictory experiments、competing explanationsC1–C9；
- evidence strength labels：STRONG/MEDIUM/WEAK；
- domain size是否仍是合理主解释变量、是否存在small-improvement/large-decline trade-off；
- `Method justified`或`Method too complex for gain`；`EMA necessary/unnecessary/harmful/not justified`；`Curriculum unnecessary`；
- GO/CONDITIONAL/STOP 与最能区分剩余解释的下一实验。

不允许因sample weighting已很强而降低Adaptive门槛，也不允许为进入E7隐藏失败domain、seed反例或EMA负结果。
