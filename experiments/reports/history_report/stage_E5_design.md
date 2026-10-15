# DESIGN SPECIFICATION — Stage E5：Conditional Contribution Curriculum

## 0. 范围、顺序与条件门控

本次 DESIGN 严格只读取了提示词文件 `promt/验证性补全和实验方向寻找/group3.md`。没有读取、搜索、打开或委托其他代理读取其他提示词文件。

`group3.md` 内含有先后约束的 E5、E6、E7。本规格只设计当前顺序最前且条件允许的 E5；不设计 E6 adaptive controller，也不设计 E7 generalization matrix。E5 完成 reviewer、runner 与 main analysis 并给出 GO/STOP 后，才能单独开始下一轮 DESIGN。

E5 motivation gate 判定为 **GO（仅允许实验，不代表方法成立）**，依据已有正式结果：

- E3 early common-support validation 在 12/12 blocks 选择 `alpha<1`；late training 在 11/12 blocks 选择 `alpha=1`；
- early 选择相对 `alpha=1` 的描述性 test AUC 平均增益为 `0.016251`，11/12 blocks 为正、10/12 超过 0.001，4/4 seed 聚合方向为正；
- early 的精确最优 alpha 在 0、0.25、0.5、0.75 间变化，因此证据只支持“early 较低、late 较高”的区域变化，不支持按 seed、domain 或 test 结果挑选 schedule；
- E3 endpoint 上 `alpha=1` 是最佳或实际等价最佳静态方案，`alpha=0.75` 是最有信息量的 distinct interior control；没有固定 interior alpha 通过 superiority gate；
- E4 在另一数据集上支持 contribution allocation 的方向性，但 residual error 仍可由 temporal/data shift 解释，不能当作 Curriculum 的正面结果。

因此本实验必须区分“顺序本身有效”与“平均 coefficient exposure、checkpoint、optimization budget 或偶然 seed 波动”。若 Forward 未稳定超过 Static 与 Reverse，E5 必须 DROP Curriculum；E6 是否开展由其自身条件决定，不得用 E5 失败阻止或强行包装 adaptive calibration。

## 1. Research question

在 MovieLens-1M、r50、canonical `d0_large`、MDMTRec/M3OE、自然采样固定时：

> 仅改变 domain contribution exponent 随训练阶段的时间顺序，是否能比最佳静态 weighting、反向顺序和相同积分暴露的静态控制产生稳定的 final performance 与 transfer 改善？

具体回答：

1. `0.75 -> 1.00` 是否优于 `1.00 -> 0.75`，从而支持 temporal order 而不只是 coefficient 范围；
2. Forward 是否优于 static `alpha=1`、static `alpha=0.75` 与 exposure-matched static `alpha=0.875`；
3. 改善是否同时出现在 fixed-final 与 validation-selected checkpoint，而非 checkpoint selection artifact；
4. 是否存在 Large gain / Small decline、某一 task 独占增益或 worst-domain 退化；
5. contribution trajectory 是否按预注册 schedule 改变，以及 performance 是否仅跟随累计 exposure；
6. domain size 与 operational difficulty 是否提供不同解释。canonical setting 不能将 size 与 physical-domain identity/difficulty 完全解耦，因此本阶段只作受限判断，Small 不等同 difficult。

## 2. Hypothesis being tested

### 2.1 Primary hypothesis H-T

在总 optimizer steps 与累计 alpha exposure 匹配时，Forward `0.75 -> 1.00` 比 Reverse `1.00 -> 0.75` 和 exposure-matched Static `0.875` 获得更高的 final overall AUC；相对 static `1.00` 的改善在至少 3/4 seeds 同向，并且不依赖单一 domain/task。

### 2.2 Null H0

Forward、Reverse 与 exposure-matched Static 的差异小于实际意义阈值或 seed 方向不稳定。此时 E3 的 early/late 选择只是静态轨迹的描述性现象，不能支持 temporal allocation。

### 2.3 Competing explanations

- **C1 — integrated exposure**：收益只由平均 alpha/累计 coefficient exposure 决定；Static `0.875` 应与两条 schedule 等价。
- **C2 — endpoint objective**：收益只由训练末期 alpha 决定；Forward 接近 static `1.00`，Reverse 接近 static `0.75`，而非体现顺序效应。
- **C3 — checkpoint selection**：schedule 只改变 peak epoch；在 fixed-final epoch 上优势消失或反转。
- **C4 — generic optimization continuation**：不同 early stop/step 数制造差异。E5 通过固定 18 epochs、相同 batch order 与相同 optimizer-step 数排除此项。
- **C5 — one-domain/task effect**：总体均值由一个 cell 驱动，Large/Small 或两 task 出现冲突。
- **C6 — physical identity / operational difficulty**：结果来自 canonical d0/d1/d2 固有难度，而非 size role。用 frozen single-domain oracle AUC/LogLoss 描述 operational difficulty；E5 不凭单一 role assignment 宣称 size 是主因。
- **C7 — sampling/noise**：自然采样下某些 batch 的梯度噪声与 schedule 偶然耦合。全部 arms 使用完全相同 batch sequence，并记录 raw norm/variance/SNR 作为竞争诊断。
- **C8 — architecture path**：若 schedule 意外进入 architecture optimizer，实验不再是 E3 objective schedule。architecture loss 必须保持 E3 的原始 unmasked objective。

## 3. Variables changed

### 3.1 唯一主要操纵变量：alpha trajectory

令固定训练 epoch 数 `E=18`，epoch index `e in {0,...,17}`，归一化阶段：

\[
u_e = \frac{e}{E-1}.
\]

预注册五个 arms：

| arm key | alpha at epoch e | purpose |
|---|---:|---|
| `static_sample` | `1.000` | E3 最佳 static / sample baseline |
| `static_interior` | `0.750` | E3 最有信息量 distinct interior/no-schedule control |
| `static_matched` | `0.875` | 与 Forward/Reverse 相同离散平均 alpha 的 exposure control |
| `forward` | `0.750 + 0.250*u_e` | 预注册 low/intermediate -> sample schedule |
| `reverse` | `1.000 - 0.250*u_e` | 相同范围、相同积分暴露、相反时间顺序 control |

对 18 个离散 epochs，Forward 与 Reverse 的平均 alpha 均严格为 `0.875`，且逐 epoch 有 `alpha_forward(e)+alpha_reverse(e)=1.75`。不得改为 cosine、piecewise、warmup、按 validation 自适应或按 seed 调参。

E3 的全局最优 static `alpha*=1` 与 baseline A 语义重合，因此不创建重复的 `alpha=1` run；`alpha=0.75` 作为 E3 明确推荐的 distinct interior control。

### 3.2 Formal matrix

- dataset/ratio/setting：MovieLens-1M × r50 × canonical `d0_large`：1
- arms：5
- seeds：4（2022、2023、2024、2025）
- 正式 joint runs：`1 × 5 × 4 = 20`

Transfer 使用 E3 已有 canonical setting 的 `3 domains × 4 seeds = 12` 个 frozen oracle reference artifacts；它们不是 E5 新训练臂。runner 先验证 identity 后复用，不因 E5 重跑。若某 reference 缺失或 identity 不匹配，相关 transfer 标记 unavailable 并阻止 transfer-based GO 判定；不得静默以 joint baseline 代替。

本阶段不得扩展 role reversal、r1/r5/r20、KuaiRand、backbone、warmup、schedule shape、transition point 或 ratio 网格。

## 4. Variables held constant

所有 20 runs 固定：

- Dataset：MovieLens-1M；fixed-N r50；canonical `d0_large`；
- role mapping：d0=L、d1=M、d2=S；训练 counts `150000/34500/3000`，shares `0.800/0.184/0.016`；
- 现有 temporal split、feature processing 与 seed-specific role subsample；
- model：MDMTRec/M3OE，`backbone=mdmoe`，D=3，T=2；所有 architecture/model dimensions 与 E3 相同；
- main Adam、architecture Adam、learning rates、betas、weight decay、parameter grouping 与 E3 相同；
- batch size 40960、num_workers 0、自然采样、shuffle、architecture-update cadence；
- task 等权；无 optimizer scheduler、momentum method、EMA、adaptive controller、新 sampler、新 router；
- architecture optimizer 的 unmasked loss 完全不使用 alpha schedule；
- 每 arm 固定运行 18 epochs，不启用 early termination；validation 仍用于记录 peak/selected checkpoint，但不得改变训练长度；
- 同一 seed 五 arms 的 initialization hash、split fingerprints、train/architecture batch order、每 epoch step count、总 optimizer steps 必须相同；
- test 不参与 schedule、checkpoint 或任何训练决策；
- instrumentation cadence 全 arms 一致，至少每 epoch 两个 shared-gradient probes；
- frozen oracle、baseline performance、模型和训练阶段定义一致。

固定 18 epochs 是 E5 的识别约束：E3 的 common-support 实际为约 15–18 epochs，且 18 个固定 epochs 可使 Forward、Reverse 与 Static `0.875` 的累计 exposure 和 optimization budget 严格匹配。不得用不同 early-stop 尾长破坏该比较。

## 5. Mathematical / algorithmic definition

### 5.1 Scheduled contribution objective

沿用 E3 cell order `idx=task*D+domain`。batch 中 cell masked-mean BCE 为：

\[
\ell_{dt}(\theta)=\frac{1}{|B_d|}\sum_{i\in B_d}
\operatorname{BCE}(\hat y_{i,t},y_{i,t}).
\]

训练集 domain share：

\[
p_d=n_d/\sum_j n_j.
\]

对 arm `a`、epoch `e`，先按上表得到 `alpha_a(e)`，再计算：

\[
c_{d,e}^{(a)}=
\frac{D p_d^{\alpha_a(e)}}{\sum_{j=1}^{D} p_j^{\alpha_a(e)}} ,
\qquad
L_e^{(a)}(\theta)=\sum_{d=1}^D\sum_{t=1}^T
c_{d,e}^{(a)}\ell_{dt}(\theta).
\]

每 task 复制相同 `c_d`，因此每 epoch：

\[
\sum_d c_{d,e}^{(a)}=D,\qquad
\sum_{d,t}c_{d,e}^{(a)}=DT=6.
\]

Schedule 在 epoch 开始时更新一次并对该 epoch 的所有 main-optimizer batches 固定；不得按 batch loss、gradient、validation 或 test 更新。每 run 只进行一次正常 forward/backward/step。观察器不得改写 `.grad` 或额外更新参数。

注意：相同平均 alpha 不意味着每个 domain coefficient 的算术平均完全相同，因为 `c_d(alpha)` 是非线性函数。因此必须同时报告：

1. `sum_e alpha(e)`；
2. 每个 domain/task 的 `sum_e c_d(e)`；
3. 每个 domain/task 的实际 `sum_steps c_d(e)`。

`static_matched=0.875` 是 alpha-exposure matched control；Forward 与 Reverse 彼此在 alpha 范围、平均、步数上严格匹配。若 coefficient 非线性造成与 static-matched 的 domain coefficient integral 小偏差，分析必须显式量化，不能宣称完全 coefficient-matched。

### 5.2 Contribution trajectory

沿用 E3 的 shared parameter partition。对 shared parameters：

\[
g^S_{dt,k}=\nabla_{\theta_S}\ell_{dt,k},\quad
h^S_{dt,k}=c_{d,e}g^S_{dt,k},\quad
G^S_k=\sum_{d,t}h^S_{dt,k}.
\]

至少记录：raw gradient norm、weighted norm、magnitude share、signed effective contribution、data-share mismatch、actual-update alignment、raw gradient variance/SNR。Schedule 实现正确性要求 recorded coefficient 与当 epoch 公式一致；contribution 变化本身不是 performance 因果证明。

### 5.3 Performance、transfer、difficulty 与 training-stage metrics

每 epoch 按 domain × task 记录 train/validation/test AUC 与 exact sample-mean LogLoss。汇总：

- overall macro AUC / macro LogLoss（六 cells 等权）；
- Large、Medium、Small domain AUC 与 LogLoss；
- worst-domain AUC；
- per-task 与 per-domain/per-task 指标；
- fixed-final epoch=`17`；
- peak epoch：validation macro AUC 最大的 earliest epoch；
- selected-checkpoint test metrics：只在训练结束后使用预先保存的 validation-best state 评估/汇总；
- early/middle/late：固定划分 epochs `0–5 / 6–11 / 12–17`，不得按每 arm peak 重新切分。

使用 E3 frozen oracle：

\[
Transfer_{d,t}=AUC^{joint}_{d,t}-AUC^{oracle}_{d,t}.
\]

domain/role transfer 为两 task 等权；negative-transfer rate 为六个 matched cells 中 `Transfer<0` 的比例。

Operational difficulty 只由相同 seed/physical-domain/task 的 frozen oracle AUC（低 AUC=较难）和 oracle LogLoss 描述。domain size 使用固定 train count/share。不得把 Small 直接标记为 difficult；因 E5 只有 canonical role，size-vs-difficulty 主效应不可识别，分析必须明确这一限制。

### 5.4 Primary estimand and paired analysis

seed 是唯一 replication unit。Primary estimand：

\[
\Delta_{F-X}^{final} = \frac14\sum_s
[AUC_{overall,s}^{forward,epoch17}-AUC_{overall,s}^{X,epoch17}],
\]

其中 `X in {static_sample, static_matched, reverse}`。同时报告四个 paired seed deltas、mean、sample std、median、min/max、improved seeds/4 与反例。

Selected-checkpoint 作为 confirmatory secondary estimand；Large/Small/worst-domain/per-task/LogLoss/transfer/contribution trajectory 为机制与安全性诊断。n=4 时不以单独 p-value 作为结论；可报告 paired bootstrap CI，但 evidence label 主要依据 effect size、方向一致性和反例。

实际意义阈值预注册为：AUC/transfer `0.001`，LogLoss `0.001`（改善定义为 baseline minus candidate）。

## 6. Exact affected modules / files

### 6.1 新实现文件（由 coder 阶段负责）

- `experiments/stage_E5_schedule.py`：arm registry、`alpha(epoch)`、coefficient/exposure 计算、matrix/block validator；
- `experiments/stage_E5_run.py`：单个 `(arm,seed)` 训练入口，复用 E3 data/model/instrumentation；
- `experiments/stage_E5_orchestrator.py`：固定 20-run resume-safe matrix；
- `experiments/stage_E5_report.py`：结构化汇总原始结果，不作科研结论；
- `experiments/tests/test_stage_E5_schedule.py`：公式、boundary、exposure、loss-path focused tests；
- `experiments/tests/test_stage_E5_matrix.py`：20-run matrix、block identity、artifact completeness tests。

### 6.2 只读复用，原则上不修改

- `experiments/stage_E3_weighting.py`
- `experiments/stage_E3_run.py`
- `experiments/stage_E3_orchestrator.py`
- `experiments/stage_E3_report.py`
- `experiments/stage_E3_raw_results/oracle_reference/`
- `experiments/stage_E1_instrumentation.py`
- `m3oe/MDMTRec/models/multi_domain/model.py`
- `m3oe/MDMTRec/utils/data.py`
- `m3oe/MDMTRec/trainers/ctr_trainer.py`

若必须修改上述共享文件，coder 必须先说明 E5 隔离实现为何不足，并证明 E1/E3/E4 API/产物不受破坏；优先通过新文件封装，禁止改核心模型结构。

### 6.3 阶段产物

- `experiments/reports/stage_E5_design.md`（本文件）
- `experiments/reports/stage_E5_implementation.md`
- `experiments/reports/stage_E5_review.md`
- `experiments/stage_E5_raw_results/`
- `experiments/stage_E5_summary.csv`
- `experiments/stage_E5_summary_manifest.json`
- `experiments/reports/stage_E5_run_report.md`
- `experiments/reports/stage_E5_go_no_go.md`（仅 main analysis 写）
- `experiments/reports/stage_E5_analysis.md`（仅 main analysis 写）

## 7. Implementation requirements

1. 五个 arm key 与公式必须是封闭 registry；formal run 拒绝任意额外 schedule 参数。
2. `alpha(epoch)` 使用 epoch 0–17，Forward 边界严格为 0.75/1.0，Reverse 为 1.0/0.75；Static 不随 epoch 变。
3. 每 epoch 由实际 audited domain shares 调用 E3 coefficient formula，不按 physical-domain id 硬编码。
4. scheduled coefficients 只作用于 main six-cell loss；architecture loss、sampler、routing 与 optimizer 不改变。
5. 全部 arms 固定完成 18 epochs；不得 early terminate。数值非有限或运行错误应 FAIL，而非产生短尾作为合法结果。
6. 同 seed 五 arms 必须具有相同 initialization、split rows、每 epoch train/architecture batch fingerprints、step counts 与 total processed examples。
7. 记录 epoch/step 实际 alpha 与六项 coefficients；manifest 记录 planned/observed alpha trajectory hash、per-domain coefficient integrals 与 optimizer-step-weighted integrals。
8. checkpoint 只由 validation macro AUC 选择，tie 取更早 epoch；test 不参与训练、schedule 或选择。
9. test trajectory evaluation 必须保持/恢复 CPU 与 CUDA RNG state，避免改变后续 batch/optimization；只需复用 E3 已验证机制，不重复大规模防御测试。
10. shared-gradient observer 复用 E3 分区与 cadence；不得写 `.grad`、不得额外 optimizer step、不得把 architecture/private 参数误作 shared。
11. oracle resolver 只接受 E3 formal-valid、seed/canonical-setting/physical-domain/task/split identity 完整的 reference；不从 Markdown 均值或跨 seed 汇总构造 oracle。
12. report 固定枚举 `5 arms × 4 seeds=20` expected paths，缺失即列出；summary 保留 seed-level/cell-level rows，不只输出均值。
13. coder 只运行 syntax、focused unit tests 和至多一个极短 smoke；不得启动 20-run 正式矩阵，不得解释科研结果。
14. 不实现 adaptive rule、EMA/momentum estimator、role reversal、第二数据集、第二 backbone 或新算法组件。

## 8. Ablations required

只做以下必要比较，不增加组合：

1. `forward` vs `reverse`：主要 temporal-order falsification；
2. `forward` vs `static_matched`：排除平均 alpha exposure；
3. `forward` vs `static_sample`：关键简单 baseline；
4. `forward` vs `static_interior`：排除仅使用低 alpha；
5. fixed-final vs validation-selected checkpoint：排除 selection-only；
6. early/middle/late trajectory：验证阶段效应是否按预测出现；
7. per-domain/per-task 与 L/M/S transfer：检查单 cell 驱动及 Large/Small trade-off；
8. raw vs weighted shared-gradient/contribution trajectory：区分 coefficient 直接作用与 state/noise 解释。

不做 `kappa × schedule × warmup × ratio` 网格，不事后修改 transition、shape 或 endpoint。

## 9. Expected logging requirements

每 run 目录：`experiments/stage_E5_raw_results/<arm_key>/d0_large/seed<seed>/`，至少含：

- `config.json`
- `run_manifest.json`
- `stdout.log`
- `metrics_epoch.csv`
- `metrics_summary.csv`
- `shared_gradient_step.csv`
- `shared_gradient_epoch.csv`
- `schedule_epoch.csv`
- `batch_trace.csv`
- `nonfinite_inventory.json`

`schedule_epoch.csv` 每 epoch 至少记录：arm、seed、epoch、phase、alpha、domain/task coefficients、coefficient sum、cumulative alpha exposure、per-domain cumulative coefficient exposure、main step count。

`metrics_epoch.csv` 每 `(epoch,domain,task)` 记录 train/validation/test count、positive rate、AUC、LogLoss、size role、domain size/share、oracle AUC/LogLoss/source、transfer、selected/final flag。

`shared_gradient_step/epoch.csv` 记录 alpha、coefficients、raw/weighted norm、variance/SNR、magnitude share、signed contribution、mismatch、update alignment、probe count 与 reconstruction error。

`config/manifest` 记录 dataset/model/optimizer/training-stage identity、18-epoch fixed-budget flag、schedule formula/version/hash、planned/observed trajectories、initial/final hash、split/batch fingerprints、best/peak epoch、fixed-final epoch、total steps/examples、oracle coverage 与 RNG restoration evidence。

`stage_E5_summary.csv` 至少保留 seed、arm、epoch view（fixed-final/selected）、physical domain、size role、task、AUC、LogLoss、transfer、negative-transfer、overall/worst-domain、peak epoch、contribution statistics、alpha/coefficient integrals。`stage_E5_summary_manifest.json` 列出 20/20 completeness、block validity、oracle coverage 与所有 missing/invalid artifacts。

Runner 只记录 raw results 和完整性，不写科研结论。

## 10. Acceptance criteria for reviewer

Reviewer 重点审查算法是否忠实实现，不进行不必要的防御性扩张。只有以下均满足才可 PASS：

1. 五 arm schedule 在 epoch 0、8/9、17 的 alpha 与公式一致；Forward/Reverse 离散平均均为 0.875，方向相反；
2. coefficient 使用 E3 normalization，六 cell coefficient sum 每 epoch为 6，task 复制顺序正确；
3. main loss 实际使用当前 epoch scheduled coefficients；architecture loss 和 sampling 未被改变；
4. 18 fixed epochs、main/architecture optimizer steps 与 processed examples 在同 seed 五 arms 一致；
5. 一个 deterministic toy-loss test 证明 scheduled weighted sum 等于手算结果，反向梯度进入预期 main parameters；
6. 一个短 smoke 证明 schedule logging 与训练实际 coefficient 一致，observer 不改变参数更新；无需重复 E1 的全面 ON/OFF 防御验证；
7. 同 seed initialization/split/batch-order identity comparator 能检测真实 mismatch；
8. per-epoch/per-domain/per-task AUC、LogLoss、transfer、peak/final 与 contribution trajectory 字段齐全；
9. oracle 不含 validation/test leakage，test 不参与 schedule/checkpoint；
10. formal orchestrator 只枚举 20 runs，seeds 精确为 2022–2025，无额外 grid；
11. 未实现 E6/E7、EMA/momentum、adaptive controller 或事后 schedule tuning；
12. implementation report 如实列出检查、限制与任何偏离。

Blocking FAIL 条件包括：schedule 未实际作用于 main loss、意外改变 sampler/architecture objective、Forward/Reverse exposure/steps 不匹配、使用 test 调 schedule、缺少 per-domain/task 指标、或正式矩阵不是预注册 20 runs。

## 11. Conditions that would falsify the hypothesis

以下任一核心现象都削弱或证伪 H-T：

1. Forward 相对 Reverse 或 Static `0.875` 的 final overall AUC mean gain `<=0.001`，或仅 `<=2/4` seeds 为正；
2. Forward 与 `static_sample` 实际等价（绝对 mean delta `<=0.001`）且没有 worst-domain/transfer 的稳定补偿优势；
3. Reverse 与 Forward 同等或更好，说明方向/顺序预测错误；
4. Forward 优势只存在于 validation-selected checkpoint，在 fixed-final epoch 消失或反转；
5. 平均 gain 由一个 seed、一个 domain 或一个 task 驱动，其余方向冲突；
6. Forward 提高 overall AUC 但 Large 或 Small transfer 平均下降超过 0.001，或 worst-domain AUC下降超过 0.001，不能称为普遍 temporal allocation 改善；
7. performance 只跟随累计 coefficient exposure，Static `0.875` 复现 Forward；
8. schedule 产生预期 contribution trajectory，但 performance 不改善，说明 contribution timing 不是足够机制；
9. raw gradient variance/SNR 而非时间顺序更一致地解释 seed/domain 结果；
10. 四 seed 的效应方向不稳定或反例超过预注册门槛。

## 12. Go / Stop rule for main analysis

### GO — temporal allocation has research value

必须同时满足：

- Forward fixed-final overall AUC 相对 `static_sample`、`static_matched`、`reverse` 的 mean gain 均 `>0.001`；
- 上述三个关键 contrast 各至少 3/4 seeds 为正；
- selected-checkpoint 对三个 contrast 方向一致；
- LogLoss 不比 static sample 恶化超过 0.001；
- Large transfer、Small transfer、worst-domain AUC 相对 static sample 均无超过 0.001 的平均退化；
- 优势不只来自单一 domain/task，并存在符合预注册方向的 contribution trajectory。

只有满足这些条件，`stage_E5_go_no_go.md` 才可写：`Contribution Curriculum retained for later comparison`。这不证明 Curriculum 为最终方法，也不授权扩展 schedule grid。

### STOP — DROP Curriculum

若 Forward 约等于 Reverse/Static、seed consistency 不足、只产生 checkpoint artifact，或未通过上述 GO gate，则写：

`Contribution Curriculum currently lacks empirical support and is dropped.`

之后 E5 不再增加 schedule、warmup、ratio 或 shape 组件。后续 E6 若独立进入，主 baseline 不包含 Curriculum；只有 E5 GO 时才把最佳预注册 Contribution Curriculum 作为 E6 的附加 baseline。

## 13. Required main-analysis reporting

Main analysis 必须报告且不能预设结论：

- AUC、LogLoss、per-domain、per-task、L/M/S、worst-domain、transfer、negative-transfer；
- four-seed paired deltas、mean/std、direction consistency 与反例；
- domain size/count/share；oracle operational difficulty，明确 Small 不等于 difficult；
- baseline performance：static 1、0.75、0.875 与 frozen single-domain oracle；
- model/backbone 与 fixed training stage/peak epoch；
- early/middle/late performance 与 contribution trajectory；
- stable repeated phenomena、contradictions、evidence strength（STRONG/MEDIUM/WEAK）；
- C1–C8 竞争解释；
- 是否存在 small-domain improvement / large-domain decline trade-off；
- Go/Stop 判定与最能区分剩余解释的下一实验。

本设计不允许从尚未运行的 E5 数据推断结果，也不允许为了“顶会故事”放宽 GO gate。
