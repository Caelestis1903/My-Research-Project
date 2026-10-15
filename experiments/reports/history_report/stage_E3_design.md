# DESIGN SPECIFICATION — Stage E3：Domain Weighting Response Surface

## 0. Scope and ordering declaration

本次 DESIGN 严格只读取了提示词文件 `promt/验证性补全和实验方向寻找/group2.md`。没有读取、搜索、打开或以其他方式接触 `group1.md` 与 `group3.md`，也没有委托任何子代理读取它们。

`group2.md` 内含两个有先后约束的实验方向：Prompt 3（MovieLens r50 weighting response surface）与 Prompt 4（KuaiRand replication）。本规格只设计顺序在前的 Prompt 3，即 Stage E3；不设计、不实现、不运行 Prompt 4。只有 E3 正式分析完成并给出 weighting law 判定后，才允许为 KuaiRand 单独进入下一轮 DESIGN。

本阶段不实现算法、不运行实验、不生成或推断新结果。现有 E1 报告只作为协议与 instrumentation 的依据，不把其结论当作 E3 的预定结果。

## 1. Research question

在 MovieLens-1M、fixed-N、r50、canonical role rotation、MDMTRec/M3OE、自然采样不变的条件下，将 domain loss weighting 从 equal 连续改变到 sample proportional 时：

> Large、Medium、Small 的 transfer、overall AUC、worst-domain AUC、negative-transfer rate、LogLoss 与 shared optimization contribution 是否呈现稳定、可重复、具有方向性的响应曲线？

需要回答：

1. equal → sample 是否近似单调改善，而非只在两个端点偶然不同；
2. Large 改善是否以 Small 退化为代价；
3. 是否有固定的中间 `alpha` 稳定优于 `alpha=0` 与 `alpha=1`；
4. 最优区域是否随 size role、physical domain 或 task 改变；
5. 多目标指标间是否存在可重复的 Pareto frontier；
6. 静态 sample weighting 是否已经足够，还是只有证据支持继续研究 adaptive 或 temporal weighting。

E3 是静态 objective response-surface 实验。它可以为 adaptive/temporal 方法提供动机，但不能直接证明一个尚未运行的 controller 或 curriculum 有效。

## 2. Hypotheses being tested

### 2.1 Primary hypothesis H-R：stable weighting response

随着 `alpha` 从 0 增加到 1，domain coefficient 与 shared effective contribution 的变化会伴随可重复的 transfer/performance 响应；该响应主要随 Large/Medium/Small size role 轮换，而非固定绑定某个 physical domain。

### 2.2 Null hypothesis H0：no stable performance law

除 coefficient 与 coefficient-adjusted gradient magnitude 的确定性变化外，AUC、LogLoss、transfer 与 negative-transfer rate 对 `alpha` 的变化在 seeds、role rotations 或 tasks 间不稳定，或者效应小于预注册的实际意义阈值。此时不得据此开发复杂 weighting 方法。

### 2.3 Static-sufficiency hypothesis H-S

`alpha=1` 对 overall、worst-domain 与各 size role 指标同时达到最优或实际等价于最优；任何中间 `alpha` 的优势均不稳定或小于实际意义阈值。若成立，static sample weighting sufficient。

### 2.4 Competing explanations

- **C1 — deterministic coefficient effect only**：shared contribution ratio 仅按公式机械变化，performance 不随之稳定变化。
- **C2 — raw-gradient geometry/noise**：响应主要由不同 `alpha` 训练状态产生的 raw gradient norm、alignment 或 variance 变化解释，而非 coefficient 本身。
- **C3 — checkpoint/convergence**：曲线来自各臂 early-stop/checkpoint epoch 不同；在共同 epoch 或固定终点上消失。
- **C4 — physical-domain/difficulty**：曲线跟随 d0/d1/d2 或 label prevalence，而不随 L/M/S role rotation。Small 不得被直接等同于 difficult。
- **C5 — task heterogeneity**：task 0 与 task 1 的响应方向不同，宏平均掩盖冲突。
- **C6 — sampling exposure**：若 batch sequence 或 sampling strategy 在 arms 间不一致，观察到的响应不能归因于 objective weighting。
- **C7 — finite-grid/noise**：五点曲线的“中间最优”是 test-set 事后挑选或随机波动，而不是可复现规律。

本设计必须允许 H0、H-S 或任一竞争解释胜出，不以证明 adaptive weighting 为目标。

## 3. Variables changed

### 3.1 Sole main manipulated variable

唯一主要训练变量为：

\[
\alpha \in \{0, 0.25, 0.50, 0.75, 1.00\}.
\]

不得把 optimizer、sampling、model、routing、Curriculum、Momentum/EMA 或 checkpoint rule 与 `alpha` 联动。

### 3.2 Formal main matrix

- alpha：5
- canonical role assignment：3
- seed：4（2022、2023、2024、2025）
- 正式 joint runs：`5 × 3 × 4 = 60`

Role assignments 固定为：

- `d0_large`：d0=L，d1=M，d2=S
- `d1_large`：d0=S，d1=L，d2=M
- `d2_large`：d0=M，d1=S，d2=L

60 个 run 之外不加入新的 joint training arm。r5/r20 是 E3 分析后才可决定的 extension，不得在本轮 coder/reviewer/runner 阶段预先展开。

### 3.3 Auxiliary oracle reference

Transfer 必须使用相同 `(setting, physical_domain, task, seed)` 的 single-domain oracle。oracle 是跨所有 alpha 共享的冻结参考，不是 alpha response-surface 的训练臂。

当前工作树中的 E1 config 指向 `experiments/P0_2_ratio_sweep`，但该目录当前不存在，E1 的 oracle 字段均标记为 missing。因此实现必须先做只读 artifact audit：

1. 若能找到并验证 P0-2 r50 的 seed-level `metrics_summary.csv`，直接复用；
2. 严禁从 Markdown 报告均值、跨 seed 汇总或 E1 结果反推 oracle；
3. 若 seed-level artifact 仍缺失，runner 在正式 joint matrix 前补跑 `3 settings × 3 physical domains × 4 seeds = 36` 个冻结 oracle reference runs；
4. oracle 只计算一次并供五个 alpha 共享，不参与 alpha 选择，不改变 60-run 主矩阵定义。

若 oracle 工程入口经 reviewer 判断无法正确实现，则 transfer 与 negative-transfer rate 必须标记 unresolved，不能以 equal endpoint 代替 oracle；但 overall/worst-domain/AUC/LogLoss/contribution response 仍可运行并分析。

## 4. Variables held constant

全部 joint runs 固定为 E1/P0-4 r50 协议：

- Dataset：MovieLens-1M；
- fixed total training N：187,500；
- role sizes：L=150,000，M=34,500，S=3,000；
- role shares：`p=(0.800, 0.184, 0.016)`，按 role 映射到 physical domain；
- train/validation/test：既有 80%/10%/10% 时间顺序切分；
- role subsample：`RandomState(seed*10+d)`；
- model：MDMTRec/M3OE，`backbone=mdmoe`，D=3，T=2；
- `fcn_dims=[512,256,256,64]`，shared experts=4，`exp_d=exp_t=bal_d=bal_t=1`；
- main optimizer：Adam，lr=1e-2，betas=(0.9,0.999)，weight decay=1e-5；
- architecture optimizer：Adam，lr=1e-3，weight decay=1e-5；
- `fix_optim_grouping=1`；
- batch size=40,960；num_workers=0；
- upper bound=50 epochs；early-stop patience=10；
- checkpoint objective：validation 上六个 domain-task cell AUC 的等权宏平均；
- task 维度始终等权；
- train sampling/loader/shuffle、architecture-update cadence、validation cadence 全部不变；
- 无 scheduler、Curriculum、Momentum、EMA、matched EMA、adaptive controller、新 router 或新 sampler；
- test 不参与 checkpoint 或 alpha 选择；
- instrumentation cadence 在全部 alpha 间一致。

同一 `(setting, seed)` 的五个 alpha 必须具有相同：

- role-sampled train rows、validation/test rows；
- initial state_dict hash；
- 每 epoch/step 的 train 与 architecture batch fingerprint/order；
- model/optimizer/architecture optimizer 初始化；
- split counts、positive rates 与 domain/task schema。

允许不同 alpha 因 early stop 产生不同尾长；比较轨迹时只使用共同观测 epoch，且不前向填充 censored epochs。

## 5. Mathematical and algorithmic definition

### 5.1 Cell loss

cell 顺序固定为 `idx = task * D + domain`。对 batch 中 domain `d`、task `t` 的 masked-mean BCE：

\[
\ell_{dt}(\theta)=\frac{1}{|\mathcal B_d|}
\sum_{i\in\mathcal B_d}\operatorname{BCE}(\hat y_{i,t},y_{i,t}).
\]

若某 batch 的 domain `d` 为空，沿用 E1 的可微零项语义并记录 count=0，不改变其他 cell 的 coefficient。

### 5.2 Alpha weighting family and normalization

令训练集 domain share 为：

\[
p_d=\frac{n_d}{\sum_j n_j},\qquad p_d>0.
\]

定义每个 domain 的 coefficient：

\[
c_d(\alpha)=\frac{D\,p_d^\alpha}{\sum_{j=1}^{D}p_j^\alpha}.
\]

每个 task 复制相同 domain coefficient：

\[
a_{dt}(\alpha)=c_d(\alpha).
\]

训练目标：

\[
L_\alpha(\theta)=\sum_{d=1}^{D}\sum_{t=1}^{T}
a_{dt}(\alpha)\ell_{dt}(\theta).
\]

该归一化保证：

\[
\sum_d c_d(\alpha)=D,\qquad
\sum_{d,t}a_{dt}(\alpha)=DT=6,
\]

从而不同 alpha 的总体 coefficient scale 与 E1 相同。它还保证：

- `alpha=0`：每个 cell coefficient=1，严格等价于 equal-domain weighting；
- `alpha=1`：`c_d=3p_d`，严格等价于 sample-proportional weighting；
- task 不按正例率、task sample count 或难度再次加权。

r50 下按 L/M/S role 的预期 coefficient（每个 task 相同）为：

| alpha | L | M | S |
|---:|---:|---:|---:|
| 0.00 | 1.000000000 | 1.000000000 | 1.000000000 |
| 0.25 | 1.450270421 | 1.004340436 | 0.545389143 |
| 0.50 | 1.850704291 | 0.887566598 | 0.261729111 |
| 0.75 | 2.165590261 | 0.719237241 | 0.115172498 |
| 1.00 | 2.400000000 | 0.552000000 | 0.048000000 |

实现必须从实际审计通过的 `n_d` 计算，而非按 physical-domain id 硬编码这些数；表中数值用于 reviewer 验证。

### 5.3 Shared optimization contribution

沿用 E1 的严格 shared parameter 分区：embedding、shared star/skip/MLP、shared experts；不得把 domain expert、task expert、cell gate/tower 或 `_weight_*` architecture parameters 纳入 shared。

对 shared parameters：

\[
g^{S}_{dt,k}=\nabla_{\theta_S}\ell_{dt,k},\qquad
h^{S}_{dt,k}(\alpha)=a_{dt}(\alpha)g^{S}_{dt,k},
\]

\[
G^S_k(\alpha)=\sum_{d,t}h^S_{dt,k}(\alpha).
\]

本阶段只需记录 shared 组，避免重复 E1 的全参数组重型诊断。主要 contribution 指标：

1. coefficient-adjusted magnitude share：

\[
s^S_{d,k}(\alpha)=
\frac{\sum_t\|h^S_{dt,k}(\alpha)\|_2}
{\sum_{j,t}\|h^S_{jt,k}(\alpha)\|_2+\epsilon};
\]

2. signed effective contribution：

\[
q^S_{dt,k}(\alpha)=
\frac{\langle h^S_{dt,k}(\alpha),G^S_k(\alpha)\rangle}
{\|G^S_k(\alpha)\|_2^2+\epsilon},
\qquad q^S_{d,k}=\sum_t q^S_{dt,k};
\]

3. data-share mismatch：

\[
D^S_k(\alpha)=\frac12\sum_d|s^S_{d,k}(\alpha)-p_d|.
\]

4. actual-update alignment（诊断）：

\[
A^S_{dt,k}=\cos(h^S_{dt,k},\theta^S_{before}-\theta^S_{after}).
\]

必须同时记录 raw gradient norm 与 weighted gradient norm，以区分 deterministic coefficient scaling 与 trajectory-mediated raw-gradient change。`weight × norm` 或 mismatch 单独不能证明 performance mechanism。

### 5.4 Performance, transfer and fairness metrics

每个 run 的 selected checkpoint 由 validation 宏平均 AUC 决定，随后报告 test：

- cell AUC/LogLoss：`AUC_{d,t}`、`LL_{d,t}`；
- domain AUC：`AUC_d=(AUC_{d,0}+AUC_{d,1})/2`；
- role AUC/transfer：把 physical domain 映射到 L/M/S，再对两个 task 等权；
- overall AUC：六个 cell AUC 的等权宏平均；
- overall LogLoss：六个 cell LogLoss 的等权宏平均；
- worst-domain AUC：`min_d AUC_d`，不是六个 cell 的最小值；
- worst-cell AUC：作为 secondary diagnostic 单独记录。

若存在匹配 oracle：

\[
Transfer_{d,t}(\alpha)=AUC^{joint}_{d,t}(\alpha)-AUC^{oracle}_{d,t},
\]

\[
Transfer_d(\alpha)=\frac12\sum_t Transfer_{d,t}(\alpha).
\]

negative-transfer rate 定义为所有匹配 `(setting,seed,domain,task)` cell 中 `Transfer_{d,t}<0` 的比例；同时报告每个 role/task 的比例。缺 oracle 的 cell 从分母排除并显式给 coverage；coverage 不是 100% 时不得给完整 negative-transfer verdict。

LogLoss transfer 辅助定义为 `joint - oracle`，负值表示 joint 更好，必须与 AUC transfer 的符号语义分开。

### 5.5 Oracle reference semantics

若需补跑 oracle，对每个 `(setting, physical_domain, seed)`：

- 训练样本是该 setting/seed role subsample 后属于该 physical domain 的全部行；
- validation/test 使用原时间切分中同一 physical domain 的行；
- 两 task 联合训练、task 等权；
- 相同 feature processing、base dimensions、expert count、optimizer、lr、weight decay、batch/epoch/patience；
- checkpoint 使用该 oracle 两 task validation AUC 等权均值；
- 不使用 alpha、Curriculum、Momentum/EMA 或 adaptive weighting；
- 保存原 physical-domain id、role、train count、split count 与 seed；
- 同一个 oracle artifact 供该 setting/seed/domain 的五个 alpha 共用。

若必须用 `domain_num=1` 入口，artifact 必须保留 original physical-domain mapping，不能把局部 domain 0 错当作原 d0。

## 6. Exact affected modules/files

建议采用隔离的 E3 文件，原则上不修改核心模型或 E1 产物。

### 6.1 New implementation files

- `experiments/stage_E3_weighting.py`
  - `alpha -> coefficients` 的唯一实现；
  - shared-only gradient/contribution helpers；
  - alpha-pair/matrix identity checks。
- `experiments/stage_E3_run.py`
  - 单个 `(alpha,setting,seed)` joint run；
  - 复用 E1 数据、模型、checkpoint 与 RNG-safe evaluation 语义。
- `experiments/stage_E3_oracle_run.py`
  - 仅在 verified oracle artifact 缺失时使用的冻结 single-domain reference 入口。
- `experiments/stage_E3_orchestrator.py`
  - 60-run 固定矩阵；先做 oracle audit，必要时建立 36 个 reference；resume-safe。
- `experiments/stage_E3_report.py`
  - 结构化汇总原始数据，不输出科研结论。
- `experiments/tests/test_stage_E3_weighting.py`
- `experiments/tests/test_stage_E3_matrix.py`

### 6.2 Reference/reuse, no unnecessary edits

- `experiments/stage_E1_run.py`
- `experiments/stage_E1_instrumentation.py`
- `experiments/stage_E1_orchestrator.py`
- `experiments/stage_E1_report.py`
- `m3oe/MDMTRec/models/multi_domain/model.py`
- `m3oe/MDMTRec/utils/data.py`
- `m3oe/MDMTRec/trainers/ctr_trainer.py`
- `experiments/history_exp/first/p4_run.py`（仅参考旧 single-domain oracle 语义；不得盲目依赖其历史路径）

### 6.3 Required artifacts

- `experiments/reports/stage_E3_design.md`
- `experiments/reports/stage_E3_implementation.md`
- `experiments/reports/stage_E3_review.md`
- `experiments/stage_E3_raw_results/`
- `experiments/stage_E3_summary.csv`
- `experiments/stage_E3_summary_manifest.json`
- `experiments/reports/stage_E3_run_report.md`
- `experiments/reports/stage_E3_analysis.md`（只能由 main analysis 阶段生成）

## 7. Implementation requirements

1. `alpha` 使用数值参数且只接受注册网格；目录键使用无歧义规范名：`alpha_0p00`、`alpha_0p25`、`alpha_0p50`、`alpha_0p75`、`alpha_1p00`。
2. coefficient 只由 `c_d(alpha)=D*p_d^alpha/sum(p^alpha)` 生成，task-major 顺序复制到六项；严禁对 task 再 normalization。
3. 正式训练仍使用一次原 forward、`L_alpha.backward()` 与原 optimizer step；observer 使用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`，不得写入 `.grad` 或额外 step。
4. shared-only observer 必须复用 E1 的参数分区定义；不重复记录 private/routing 全套诊断。
5. 每个 observed epoch 的每个实际 train step 记录 shared contribution。若实现成本需降采样，允许固定 `probe_interval`，但全部 alpha 必须一致且每 epoch至少 2 个 probe；不得按结果动态改变 cadence。
6. 每 run 逐 epoch记录 train/validation/test 的 cell AUC 与 exact sample-mean LogLoss；test evaluation 必须保存/恢复 CPU 与全部 CUDA RNG state。
7. checkpoint 只由六 cell validation AUC 宏平均决定；test 不得用于 checkpoint、early stop 或挑 alpha。
8. 五个 alpha 在同一 `(setting,seed)` 使用相同 train/architecture batch order；若共同支持内 fingerprint 不同，该五臂 block 无效。合法 early-stop 尾部只标记 censored。
9. 记录实际 `n_d`、`p_d`、role mapping 与 coefficient vector。若实际 counts 不是 150000/34500/3000，formal run 立即失败。
10. oracle resolver 只接受 seed-level、cell-level、setting/domain/task identity 完整的 artifact；Markdown 汇总不是合法 oracle source。
11. 需要补跑 oracle 时，它们必须先经与 joint 数据相同的 row/split/feature identity audit；oracle 不得包含任何 alpha weighting。
12. report 固定枚举 60 个 expected joint paths，即使缺失也列出；oracle coverage 独立列出，不能因 oracle 缺失把 joint run 标作不存在。
13. 允许复用 E1 helper，但 E1 的 `equal/sample` API 与既有 E1 raw results 不得被破坏或覆盖。
14. 不实现 r5/r20、KuaiRand、Curriculum、Momentum/EMA、adaptive controller 或 temporal schedule。
15. coder 只做语法、单元测试与短 smoke；不得启动 60-run/36-oracle 正式矩阵，不得分析科研结论。

## 8. Required ablations and comparisons

本阶段的“消融”就是预注册 response surface 本身，不添加其他机制臂：

1. **Five-point alpha sweep**：0、0.25、0.5、0.75、1；
2. **Endpoint controls**：alpha=0 必须复现 equal objective，alpha=1 必须复现 sample objective；
3. **Role rotation falsification**：d0_large、d1_large、d2_large；
4. **Task stratification**：task 0、task 1 分开报告，再作等权宏平均；
5. **Physical-domain vs size-role view**：两套视角都保留，检验 size 与 difficulty/identity 竞争解释；
6. **Selected-checkpoint vs common-support epoch**：用于排除 selection-only 解释；
7. **Raw vs coefficient-adjusted shared gradients**：用于区分训练轨迹与公式直接缩放。

明确不要求再次做 E1 已完成的 instrumentation ON/OFF 全套防御验证。reviewer 只需一个短程算法正确性 smoke；正式研究证据来自 runner 阶段。

## 9. Expected logging requirements

### 9.1 Per joint run

目录：`stage_E3_raw_results/<alpha_key>/<setting>/seed<seed>/`

至少包含：

- `config.json`
- `run_manifest.json`
- `stdout.log`
- `metrics_epoch.csv`
- `metrics_summary.csv`
- `shared_gradient_step.csv`
- `shared_gradient_epoch.csv`
- `batch_trace.csv`
- `nonfinite_inventory.json`

`config/manifest` 至少记录：alpha 数值与 key、实际 domain counts/shares、六项 coefficient、seed、setting、role mapping、split counts/positive rates、dataset/model/optimizer identity、initial/final parameter hash、best/stopped/censored epoch、RNG restoration evidence、oracle source/coverage。

`metrics_epoch.csv` 每 `(epoch,domain,task)`：train/validation/test count、positive rate、AUC、LogLoss、selected checkpoint、censored flag、oracle AUC/LogLoss/source、transfer。

`shared_gradient_step.csv` 每 `(epoch,step,domain,task)`：alpha、coefficient、batch fingerprint、cell count、raw norm/RMS、weighted norm/RMS、magnitude share、signed contribution、domain signed contribution、mismatch、actual-update alignment、reconstruction error。

`shared_gradient_epoch.csv`：上述 contribution 的 epoch mean/std、raw/weighted norm、probe count；gradient variance/SNR 可直接复用 E1 shared Welford，若实现则所有 alpha 必须同口径。

### 9.2 Oracle references

`stage_E3_raw_results/oracle_reference/<setting>/d<domain>/seed<seed>/` 至少包含 config、manifest、metrics_epoch、metrics_summary、stdout；另生成 `oracle_reference_manifest.csv`，标出 reused/new/missing/invalid 与 source path。

### 9.3 Stage summary

`stage_E3_summary.csv` 必须保留 seed-level rows，不得只输出均值。至少含：

- alpha、setting、seed、physical domain、size role、task；
- selected epoch；
- test AUC/LogLoss；
- domain/overall/worst-domain/worst-cell AUC；
- oracle 与 transfer；
- negative-transfer indicator；
- shared raw/weighted norm、magnitude share、signed contribution、mismatch、update alignment；
- common-support/censored 标记。

runner 与 report 只记录数据和完整性状态，不生成“最好 alpha”或机制结论。

## 10. Acceptance criteria for reviewer

Reviewer 以算法正确实现为重点；复审最多三次，不增加无关的防御性框架。只有以下 blocking 条件全部满足才可 PASS / PASS WITH FIXES：

### 10.1 Weighting correctness

- 公式与本规格完全一致；
- `alpha=0` 六项全为 1；
- `alpha=1` 为 `[2.4,0.552,0.048]` 按 task-major 重复；
- 五个 alpha 的 coefficient 与 §5.2 表在 `1e-8` 内一致；
- 每个 alpha coefficient 全为正，domain 和为3、六 cell 和为6（容差 `1e-7`）；
- loss 在 synthetic six-cell terms 上等于显式点积，且 endpoint 与 E1 equal/sample helper 数值一致。

### 10.2 Training-path fidelity

- config diff 显示五臂除 alpha/coefficient/output identity 外无训练设置差异；
- formal queue 恰为60个不重不漏 task；
- 无 prohibited mechanism；sampling strategy 相同；
- shared parameter partition 与 E1 一致；observer 不改变 `.grad` 或 optimizer state；
- checkpoint/test/early-stop 语义符合 §4 与 §7。

### 10.3 Minimal focused verification

只要求一个 `(setting=d0_large, seed=2022)` 的短程五-alpha smoke：

- initial parameter hash 相同；
- common-support train/architecture batch fingerprints 相同；
- coefficient 与实际训练 loss 匹配；
- `weighted_norm = coefficient × raw_norm`；
- signed cell contribution 在非退化 step 上和为1，容差 `1e-5`；
- reconstructed shared combined gradient 与 real backward shared `.grad` relative L2 error `<=1e-5`；
- schema 与 artifact 可被 report 读取。

smoke 不得被 resume 当作 formal完成。无需重复 E1 的完整 ON/OFF 等价性实验，除非 reviewer 找到 observer 实际改变训练路径的具体证据。

### 10.4 Oracle integrity

- reused oracle 必须有 seed/setting/original-domain/task/count/split identity；
- 若缺失则由独立冻结 oracle run 补齐，或明确 unresolved；
- 不从报告均值伪造 seed-level oracle；
- transfer 的 subtraction 方向与 role mapping 正确。

### 10.5 Review gate

- reviewer 不以结果是否支持 adaptive weighting 为 PASS 条件；
- reviewer 不运行正式 60-run 或解释科研结果；
- 最多三轮复审。第三轮仍有影响 objective、数据 identity、matrix 或指标方向的 blocking bug，则 verdict=FAIL 并交回主代理决定，不继续无限加固。

## 11. Conditions that would falsify the hypotheses

### 11.1 Falsify or strongly weaken H-R

以下任何稳定结果都会削弱或证伪“有可用 weighting response law”：

1. performance/transfer 曲线在 seeds 或 role rotations 间方向反复，aggregate effect 小于实际意义阈值；
2. contribution ratio 随 alpha 机械变化，但 AUC/LogLoss/transfer 与之无稳定关联；
3. apparent response 固定跟随 d0/d1/d2，而不是随 L/M/S role 轮换；
4. task 0 与 task 1 呈系统性相反响应，宏平均结论不可泛化；
5. 中间 alpha 优势仅来自一两个 seed、test 事后挑选或不同 checkpoint epoch，在 common-support 分析中消失；
6. negative-transfer rate 与 worst-domain AUC 无稳定改善；
7. oracle coverage 不足以检验 transfer，则关于 transfer law 的 H-R 保持 unresolved，而不是视为支持。

### 11.2 Falsify H-S

任一固定中间 alpha 在预注册检验下同时稳定优于 alpha=0 与 alpha=1，或 alpha=1 对某一 role/task 产生可重复且有实际意义的损害而另一个固定 alpha 避免该损害，则 static sample sufficiency 被证伪。

### 11.3 Evidence against adaptive/temporal motivation

若 alpha=1 在所有核心指标上占优或与最优实际等价，且没有稳定 role/task 异质性，则没有足够证据开发 adaptive weighting。若 epoch-wise 最优区域不发生稳定变化，也没有 temporal weighting 动机。

## 12. Main analysis plan and decision criteria

### 12.1 Unit of analysis and uncertainty

- 基本配对单位：同一 `(setting,seed,role,task)` 的五点 alpha curve；
- 汇总同时报告 12 个 setting×seed block 的方向一致性，以及先对三个 rotations 平均后的 4 个 seed-level effect；
- 置信区间采用 seed-cluster bootstrap（seed 为 cluster，setting 留在 cluster 内），不得把 epoch、task 或 cell 当作独立重复扩大 n；
- 全部 test 结论使用固定五点比较，不挑选 test 最优点后再对同一 test 做无校正声明；
- practical margins：AUC/transfer `delta_AUC=0.001`，LogLoss `delta_LL=0.001`。低于阈值的差异视为实际等价，而不是自动称为改善。

证据标签：

- **STRONG**：效应超过 practical margin，seed-cluster 95% CI 排除0，至少10/12 block 同方向，且4/4 seed rotation-average 同方向；
- **MEDIUM**：超过 margin，并满足 CI 或方向一致性之一；
- **WEAK**：其余、仅趋势或 coverage 不完整。

只有四个 seed 时 CI 可能较宽；不得因 p 值不显著把有一致方向的效应说成“证明无效”，应报告等价区间与证据强度。

### 12.2 Q1 — Approximate monotonicity

对 Large/Medium/Small transfer、overall AUC、worst-domain AUC、negative-transfer rate、macro LogLoss 分别：

1. 报告五点均值与 seed-cluster CI；
2. 报告四个相邻 alpha 差分；
3. 报告每 block 的 Spearman rho；
4. 定义“近似单调改善”为：aggregate curve 没有任何相邻反向变化超过 practical margin，endpoint 方向改善，且至少9/12 block 的 rho 方向一致。

不满足即报告 non-monotonic/mixed，不强行拟合单调故事。

### 12.3 Q2 — Large/Small trade-off

相对 alpha=0，只有在同一个 alpha 上同时满足：

- Large transfer/AUC 改善超过 `delta_AUC`；
- Small transfer/AUC 下降超过 `delta_AUC`；
- 两者 seed-cluster CI 均排除0，且至少9/12 block 呈对应方向；

才称为稳定 trade-off。若 Large 与 Small 同时改善，则明确否定该 alpha 的 trade-off；若方向不稳定，结论为 unresolved/mixed。

### 12.4 Q3 — Interior alpha superiority

对每个固定 interior alpha（0.25/0.5/0.75）分别与两个端点做 paired comparison，并对三种 interior 候选做 Holm correction。只有某一固定 alpha：

- 对 alpha=0 与 alpha=1 的 overall AUC 均提高超过0.001；
- 两个差值 CI 均排除0；
- 至少9/12 block 同方向；
- worst-domain AUC 不劣于最佳端点超过0.001；

才称“interior alpha 明显优于两个端点”。不得用 test 上 `max(0.25,0.5,0.75)` 作为未经校正的证据。

### 12.5 Q4 — Domain/task-specific optimum

- 展示每个 role×task 与 physical-domain×task 的五点曲线；
- 用 validation curve 选择 candidate alpha，再采用 leave-one-seed-out：其余三 seed 的 validation 选择 alpha，在 held-out seed 的 test 上评估；
- 只有不同 role/task 的 selected alpha 相隔至少两个 grid steps，且差异在至少3/4 held-out seeds 重复并带来 >0.001 test gain，才称稳定 heterogeneous optimum；
- 否则只称表观异质性，不足以支持 adaptive controller。

### 12.6 Q5 — Pareto frontier

Pareto 向量为：最大化 Large/Medium/Small transfer、overall AUC、worst-domain AUC，最小化 negative-transfer rate 与 macro LogLoss。采用 practical margins 定义 epsilon-dominance。报告：

- 基于 paired mean 的 frontier；
- seed-cluster bootstrap 中每个 alpha 成为 non-dominated 的概率；
- 只有 dominance probability >=0.95 才称明确支配；否则称 uncertain frontier。

### 12.7 Q6 and final Go/Stop verdict

Main Agent 最终必须在以下四类中选一，并附证据强度：

1. **Static weighting sufficient**：alpha=1 对所有核心指标为最优或实际等价；无稳定 interior advantage、role/task heterogeneity 或 temporal shift。
2. **Adaptive weighting justified**：固定 interior alpha 稳定优于两端，或不同 role/task/physical conditions 需要稳定不同 alpha；这只批准下一轮 adaptive DESIGN，不证明 controller 有效。
3. **Temporal weighting justified**：在 common-support epoch 上，早/中/晚期最佳 alpha 区域稳定改变，至少3/4 seeds、三个 rotations 中多数重复，且 effect >0.001。该结论只批准后续 temporal causal experiment，不得直接宣称 curriculum 有效。
4. **No stable weighting law**：曲线平坦、冲突、对 physical domain/seed 敏感，或不确定性覆盖 practical margin，无法支持前三类。

如果 alpha=1 基本全局占优，优先进入下一顺序的 KuaiRand DESIGN；如果中间 alpha 或异质性稳定存在，才进入 adaptive weighting 的独立 DESIGN；只有训练阶段最优区域稳定变化，才允许重新考虑 Contribution Curriculum。r5/r20 只能在 r50 已显示明确规律后，由主代理另行决定并重新走完整工作流。

## 13. Required plots for main analysis

runner 不画结论图。main analysis 至少生成带 seed-level points/uncertainty 的：

1. alpha vs Large transfer；
2. alpha vs Small transfer；
3. alpha vs overall AUC；
4. alpha vs worst-domain AUC；
5. alpha vs effective shared gradient contribution ratio（L/M/S magnitude share 为主，signed contribution 与 data share 作伴随层）。

并以表格补充 Medium、task-stratified、negative-transfer rate、LogLoss、physical-domain view 和 oracle coverage。
