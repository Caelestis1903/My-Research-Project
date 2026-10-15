# DESIGN SPECIFICATION — Stage E4：KuaiRand Mechanism Replication

## 0. Scope and ordering declaration

本轮 DESIGN 只读取并使用了 `promt/验证性补全和实验方向寻找/group2.md`，没有读取、搜索、打开或以其他方式接触 `group1.md` 或 `group3.md`，也没有委托子代理处理提示词。

Stage E3 已完成 main analysis；根据 `group2.md` 内部顺序，本轮只设计 Prompt 4 / KuaiRand Mechanism Replication。E3 的 temporal diagnostic 只作为竞争解释记录，不把 curriculum、temporal schedule、adaptive controller、Momentum/EMA 或任何动态权重加入本 replication。

本设计不实现源码、不运行训练、不产生或解释 E4 结果。

## 1. Research question

在不改变现有 KuaiRand-Pure 数据、特征和时间切分的条件下：

> MovieLens controlled r50 中观察到的 objective/domain-weighting law，能否迁移到真实自然失衡、且存在明显时间分布变化的 KuaiRand？

具体回答：

1. equal weighting 是否使 dominant d1 相对 single-domain oracle 出现负迁移；
2. sample-proportional weighting 是否缩小 dominant-domain 负迁移，并改善 overall、worst-domain 与 negative-transfer rate；
3. sqrt weighting 是否比 equal/sample 更合适；
4. MovieLens 上从 equal 到 sample 的正向规律是否跨数据集复现；
5. 历史 KuaiRand Comp1 的不稳定/0-of-5-positive 现象是否与 objective mismatch 解释相容；
6. train→validation/test 的 domain-mixture 与 label-prior shift 是否是更强竞争解释。

E4 只能检验 static objective weighting 的跨数据集可迁移性。它不能单独识别 temporal shift 的因果效应，也不能证明 adaptive 或 temporal 方法有效。

## 2. Hypotheses being tested

### 2.1 Primary H-K：cross-dataset objective mechanism

相对 equal weighting，sample-proportional weighting 会把 shared optimization contribution 向训练数据中的 dominant d1 重新分配，稳定缩小 d1 的负迁移，并改善或至少不实质损害 macro overall AUC、worst-domain AUC 与小域表现。sqrt 应表现为端点之间的中间响应，或在自然异质性下形成稳定折中。

### 2.2 Null H0

weighting 只机械改变 coefficient-adjusted gradient contribution；AUC、LogLoss、transfer 与 negative-transfer rate 在四个 seeds 中没有稳定、有实际意义的响应。MovieLens 机制不能直接外推。

### 2.3 Alternative H-R：reversal under natural heterogeneity

sample weighting 过度强调 dominant d1，导致 macro overall、worst-domain 或小域 transfer 稳定恶化；固定 sample proportional 并非通用解。

### 2.4 Competing explanations

- **C1 — temporal distribution shift**：train d1 share 为86.93%，validation/test 约98.2%；d1 两 task positive rate 大幅下降。跨时间分布变化可能比 objective mismatch 更主导泛化。
- **C2 — natural domain/task heterogeneity**：domains 的用户、物品、label prevalence 和条件难度不同；size 不等同于 difficulty。
- **C3 — task conflict**：click 与 long-view 对 weighting 的响应方向不同，宏平均掩盖冲突。
- **C4 — deterministic coefficient effect only**：shared contribution ratio 改变，但 raw gradient geometry 和 performance 不相应变化。
- **C5 — gradient noise/exposure**：自然 sampling 下小域 batch information 和梯度方差不同；performance 跟随 noise 而非目标权重。
- **C6 — checkpoint/convergence**：不同 weighting 的 early-stop epoch 不同造成 selected-checkpoint 差异。
- **C7 — oracle mismatch**：若 single-domain oracle 使用不同时间边界、编码或 seed，则 transfer 结论无效。
- **C8 — historical implementation/window**：旧 Comp1 结果来自不同代码路径、执行窗口和五-seed集合，不能仅凭均值归因于 weighting。

H-K、H0 与 H-R 均是允许的结果；不得为了保持 MovieLens 叙事而弱化 KuaiRand 反例。

## 3. Variables changed

### 3.1 Sole primary manipulated variable

只改变 static domain-loss weighting：

- A — Equal：`alpha=0`
- B — Sqrt：`alpha=0.5`
- C — Sample proportional：`alpha=1`

`alpha` 只决定 loss coefficient；不决定 sampling、optimizer、early stop、architecture update 或 evaluation。

### 3.2 Formal joint matrix

- weighting conditions：3
- seeds：2022、2023、2024、2025
- formal joint runs：`3 × 4 = 12`

KuaiRand 使用自然 physical domains，不做 MovieLens role rotation。

### 3.3 Oracle references

需要 `3 physical domains × 4 seeds = 12` 个 matched single-domain oracle references。它们不是 weighting arms，而是所有 A/B/C 共享的冻结 transfer baseline。

当前工作树中只有历史 oracle 报告，没有可验证的 seed-level raw artifacts/config/model/split fingerprints；因此 Markdown 数值不可作为 E4 正式 oracle。runner 应生成新的12个 matched references。若 reviewer 发现成本或工程路径不可接受，则 transfer/negative-transfer rate 必须标记 unresolved，不得制造近似 oracle。

### 3.4 Explicitly omitted condition D

不加入 optional matched constant EMA。历史 `fixed073` 并不是当前 E4 data/split/fingerprint/protocol 下已经审计的低成本 artifact；加入它会扩大研究问题且不能帮助 A/B/C 的 objective replication。Comp1、Curriculum、differential Momentum、adaptive controller 和 temporal weighting 均不进入正式矩阵。

## 4. Variables held constant

### 4.1 Dataset and split

- Dataset：现有 KuaiRand-Pure processed pipeline；
- path：`D:/CODE/data/kuairand/processed/kuai_rand.csv`；
- retained tabs：0、1、4，映射 local d0、d1、d2；
- tasks：t0=`is_click`，t1=`long_view`；
- rows：2,507,666；
- global `(date,time_ms)` chronological order 已在 preprocessing 阶段冻结；
- split：按现有 row order 80%/10%/10%；
- 不重新排序、不随机切分、不下采样、不重平衡、不增删特征或 labels；
- 现有 34 sparse-feature representation（含 `domain_indicator`）与全数据 LabelEncoder 语义保持不变。

正式 run 必须满足 `stage_E4_dataset_audit.md` 的 exact counts、positive counts/rates 与 split fingerprints。

### 4.2 Model and optimization

- model：MDMTRec/M3OE；D=3，T=2；
- `fcn_dims=[512,256,256,64]`；expert_num=4；
- `exp_d=exp_t=bal_d=bal_t=1`；
- main optimizer：Adam，lr=1e-2，betas=(0.9,0.999)，weight_decay=1e-5；
- architecture optimizer：Adam，lr=1e-3，weight_decay=1e-5；
- architecture parameters 与 main parameters 使用已修正的互斥 grouping；
- architecture update：沿 E1/E3 的既有 once-per-epoch train-batch path；
- batch size=40,960；num_workers=0；shuffle=True；drop_last=False；
- epoch upper bound=50；early-stop patience=10；
- scheduler=None。`run_kuaishou.py` 虽传入 scheduler parameters，但未传 scheduler class，现有 trainer 的实际行为是无 scheduler；
- checkpoint：六个 validation cell AUC 的等权宏平均；
- test 不参与 checkpoint、early stop 或方法选择；
- seeds 固定为2022–2025。

### 4.3 Cross-arm identity

同一 seed 的 A/B/C 必须具有相同：

- processed file identity、feature schema/encoders；
- train/validation/test ordered row fingerprints；
- initial state_dict；
- train/architecture batch order与fingerprints；
- model/optimizer/architecture-update配置；
- instrumentation cadence；
- validation/test cadence与checkpoint rule。

允许 early stop 导致不同尾长；共同轨迹比较只使用 common support，不填充 censored epochs。

## 5. Mathematical and algorithmic definition

### 5.1 Dataset facts fixed before training

Train proportions：

\[
p=(p_0,p_1,p_2)=
(0.0861269348,\ 0.8692837759,\ 0.0445892892).
\]

这些比例来自 train split，不得使用 full-data `(7.10%,89.18%,3.72%)` 或 validation/test proportions 替换。

### 5.2 Cell objective

cell order 固定为：

\[
idx=tD+d.
\]

对 batch 内 domain `d`、task `t`：

\[
\ell_{dt}(\theta)=\frac{1}{|\mathcal B_d|}
\sum_{i\in\mathcal B_d}\operatorname{BCE}(\hat y_{i,t},y_{i,t}).
\]

定义：

\[
c_d(\alpha)=\frac{D p_d^\alpha}{\sum_jp_j^\alpha},
\qquad a_{dt}(\alpha)=c_d(\alpha),
\]

\[
L_\alpha(\theta)=\sum_{d=0}^{2}\sum_{t=0}^{1}
a_{dt}(\alpha)\ell_{dt}(\theta).
\]

每个 task 复制相同 domain coefficient，不按 label prevalence 再加权。所有条件满足 `sum_d c_d=3`、`sum_dt a_dt=6`，固定总体 loss scale。

按 d0/d1/d2 的预期 coefficients：

| condition | alpha | d0 | d1 | d2 |
|---|---:|---:|---:|---:|
| Equal | 0 | 1.000000000000 | 1.000000000000 | 1.000000000000 |
| Sqrt | 0.5 | 0.612684844677 | 1.946473203086 | 0.440841952237 |
| Sample | 1 | 0.258380804454 | 2.607851327829 | 0.133767867718 |

实现必须从审计后的 train counts 动态计算，表中数值仅作为 acceptance oracle。

### 5.3 Shared optimization contribution

只复用 E1/E3 的 low-cost shared parameter partition 与 observer，不重复 private/routing 全套诊断。

\[
g^S_{dt,k}=\nabla_{\theta_S}\ell_{dt,k},\quad
h^S_{dt,k}=a_{dt}g^S_{dt,k},\quad
G^S_k=\sum_{dt}h^S_{dt,k}.
\]

记录：

\[
s^S_{d,k}=
\frac{\sum_t\|h^S_{dt,k}\|_2}
{\sum_{j,t}\|h^S_{jt,k}\|_2+\epsilon},
\]

\[
q^S_{dt,k}=
\frac{\langle h^S_{dt,k},G^S_k\rangle}
{\|G^S_k\|_2^2+\epsilon},
\qquad q^S_d=\sum_tq^S_{dt,k},
\]

\[
D^S_k=\frac12\sum_d|s^S_{d,k}-p_d|.
\]

同时记录 raw/weighted norm、shared gradient variance/SNR 与 actual-update alignment。coefficient-adjusted contribution 的机械变化不能单独证明 H-K；必须与 raw trajectory 和 performance 分开讨论。

KuaiRand 每 epoch 约49个 train batches。为控制成本，固定 observer probe steps 为 `step % 10 == 0`，即通常每 epoch 5 probes；A/B/C 完全相同，不按结果动态改变。

### 5.4 Evaluation metrics

Primary metrics 使用 cell-macro 口径：

- per-cell AUC/LogLoss；
- per-domain AUC/LogLoss：两个 tasks 等权平均；
- per-task macro AUC/LogLoss：三个 domains 等权平均；
- overall macro AUC/LogLoss：六 cells 等权平均；
- worst-domain AUC：三个 per-domain task-average AUC 的最小值；
- worst-cell AUC：secondary；
- pooled/micro AUC/LogLoss：secondary，单独报告，禁止与 macro 口径混合。

区分 macro/micro 是必要的，因为 validation/test 的 d1 share约98.2%，显著高于 train 的86.93%。

### 5.5 Transfer and negative transfer

对 matched `(domain,task,seed)`：

\[
Transfer^{AUC}_{dt}=AUC^{joint}_{dt}-AUC^{oracle}_{dt}.
\]

domain transfer 为两个 task transfer 的等权平均。负迁移 indicator：

\[
I_{dt}=1[Transfer^{AUC}_{dt}<0].
\]

negative-transfer rate 分别报告 overall、per-domain、per-task。LogLoss transfer 定义为 `joint-oracle`，负值表示 joint 更好；不能与 AUC transfer 共用符号解释。

### 5.6 Matched oracle definition

每个 `(seed, original physical domain)`：

1. 先用 joint pipeline 完成全数据编码与全局 chronological 80/10/10 split；
2. 再分别从 train、validation、test 中按 original domain 过滤；
3. 过滤后把模型输入中的 local domain id remap 为0，但 manifest保留 original d0/d1/d2；
4. 两 task 联合训练、task等权；
5. MDMTRec `domain_num=1, task_num=2`，保持相同 fcn dims、expert_num、embedding dim、optimizer、lr、weight decay、batch、epoch/patience 和 architecture-update语义；不为匹配总参数量而加宽模型；
6. checkpoint 使用该 physical domain 两个 validation task AUC 的等权均值；
7. 同一 seed；无 alpha、Momentum/EMA、Curriculum 或 adaptive机制；
8. 保存 ordered row fingerprints、original/local domain mapping、split counts/positive rates、parameter count。

任何“先过滤整个 domain 再做80/10/10切分”的旧 oracle 都不匹配当前 joint 时间边界，必须拒绝。

## 6. Exact affected modules/files

### 6.1 New files

- `experiments/stage_E4_weighting.py`
  - 可薄封装/复用 E3 coefficient、shared observer 与比较器；
  - 保存 KuaiRand E4 formal protocol 和 dataset audit constants。
- `experiments/stage_E4_run.py`
  - 单个 `(weighting,seed)` joint run；使用现有 KuaiRand loader/split。
- `experiments/stage_E4_oracle_run.py`
  - matched split-first/filter-second single-domain references。
- `experiments/stage_E4_orchestrator.py`
  - 12 oracle audit/run + 12 joint run；resume-safe，只记录状态。
- `experiments/stage_E4_report.py`
  - 结构化汇总，不做科研分析。
- `experiments/tests/test_stage_E4_weighting.py`
- `experiments/tests/test_stage_E4_data_oracle.py`

### 6.2 Reuse/reference; avoid unnecessary modification

- `m3oe/examples/run_kuaishou.py`
- `m3oe/MDMTRec/models/multi_domain/model.py`
- `m3oe/MDMTRec/utils/data.py`
- `m3oe/MDMTRec/trainers/ctr_trainer.py`
- `experiments/stage_E1_instrumentation.py`
- `experiments/stage_E3_weighting.py`
- `experiments/stage_E3_run.py`
- `experiments/stage_E3_oracle_run.py`
- `experiments/history_exp/first/p3_prepare.py`
- `experiments/history_exp/first/p4_run.py`（只作历史语义参考）

核心 model/trainer/data 文件原则上不修改；E4 逻辑放在隔离 runner 中。不得覆盖 E1/E3 raw artifacts。

### 6.3 Required outputs

- `experiments/reports/stage_E4_dataset_audit.md`
- `experiments/reports/stage_E4_design.md`
- `experiments/reports/stage_E4_implementation.md`
- `experiments/reports/stage_E4_review.md`
- `experiments/stage_E4_raw_results/`
- `experiments/stage_E4_summary.csv`
- `experiments/stage_E4_summary_manifest.json`
- `experiments/reports/stage_E4_run_report.md`
- `experiments/reports/stage_E4_analysis.md`（仅 main analysis 生成）

## 7. Implementation requirements

1. E4 必须读取现有 processed CSV，不得运行或修改 preprocessing、下载数据或生成新 split。
2. loader 输出必须与 dataset audit 的 total/split/domain/task positive counts 一致，并保存 processed-file identity 与 ordered split fingerprints。
3. coefficient 公式只实现一处；task-major复制，无 task 二次 normalization。
4. 正式 joint loss 是六个 masked-mean BCE 与 coefficients 的显式点积，进入一次真实 backward/optimizer step；observer 不写 `.grad` 或额外 step。
5. strict optimizer grouping：`_weight_*` 只进入 architecture optimizer，其余 trainable parameters 只进入 main optimizer，二者互斥且完整。
6. 自然 DataLoader sampling 在 A/B/C 间相同；不加入 balanced sampler、importance correction、downsampling 或 domain oversampling。
7. 同 seed 三臂必须具有相同 initial model hash 与 common-support batch fingerprints；任何共同 step 不一致使该 seed block INVALID。
8. validation checkpoint 必须要求六个 cell AUC 全部 finite，再取等权平均；不得因 d0/d2 validation 样本少而静默跳过 cell。
9. 每 observed epoch 记录 train/validation/test AUC 与 exact sample-mean LogLoss。额外 test evaluation必须保存/恢复 CPU与全部CUDA RNG state。
10. shared observer cadence固定为每10 steps一次；同一 forward graph上计算六个 raw cell gradients，随后真实 loss backward，不重复forward。
11. oracle 按 §5.6 split-first/filter-second；不接受只有报告均值、缺config、缺seed、缺row fingerprint或旧时间边界的 artifact。
12. oracle 必须12/12 coverage后才能计算完整 transfer；若缺失，summary明确 coverage并禁止伪造。
13. report固定枚举12 joint和12 oracle expected paths；缺失/invalid逐项列出。
14. 不实现 matched EMA、Comp1、Curriculum、temporal schedule、adaptive controller或group3实验。
15. coder只做syntax/unit tests与短 smoke；不得启动正式矩阵，不得分析结果。

## 8. Required ablations/comparisons

1. **Primary A/B/C**：equal、sqrt、sample；
2. **Equal→sqrt→sample dose order**：检验三点方向，而非只挑最好条件；
3. **Matched single-domain oracle**：计算每 domain/task transfer；
4. **Dominant vs minority domains**：d1与d0/d2分开，不把两个小域合并后掩盖差异；
5. **Task stratification**：click与long-view；
6. **Macro vs pooled evaluation**：主要结论以macro为准，pooled作为composition-sensitive secondary；
7. **Selected checkpoint vs common-support epochs**：排除selection-only解释；
8. **Raw vs weighted shared contribution**：区分trajectory与deterministic scaling；
9. **Historical Comp1 compatibility check**：只在main analysis中与可验证的历史seed-level证据作描述性方向对照，不增加Comp1新run，不声称因果 mediation。

不增加其他训练 ablation。

## 9. Expected logging requirements

### 9.1 Joint run directory

`stage_E4_raw_results/<weighting>/seed<seed>/`

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
- `model.pth`

config/manifest 必须记录：dataset/file/split fingerprints、domain mapping、每split domain/task counts与positive rates、weighting/alpha/coefficient、seed、model/optimizer/architecture config、initial/final hash、observed/best/stopped/censored epoch、RNG restore evidence、oracle coverage。

### 9.2 Per-epoch metrics

每 `(epoch,domain,task)` 保存：

- train/validation/test counts与positive rates；
- train/validation/test AUC与LogLoss；
- macro/pooled口径标识；
- selected checkpoint与censor flag；
- oracle AUC/LogLoss/source；
- AUC/LogLoss transfer。

### 9.3 Shared contribution

每 probe/cell：raw/weighted gradient norm与RMS、coefficient、magnitude share、signed contribution、domain contribution、mismatch、variance/SNR、update alignment、reconstruction error、batch fingerprint和cell count。

### 9.4 Oracle directory

`stage_E4_raw_results/oracle_reference/d<original_domain>/seed<seed>/`

保存 config、manifest、metrics_epoch、metrics_summary、stdout、model及split row fingerprints。生成 `oracle_reference_manifest.csv`。

### 9.5 Stage summary

`stage_E4_summary.csv` 保留每 seed/domain/task 行，至少含：

- weighting、alpha、seed、domain、task；
- split counts/proportions/positive rates；
- selected test AUC/LogLoss；
- macro overall/per-domain/per-task/worst-domain/worst-cell；
- pooled metrics（单独列）；
- oracle/transfer/negative-transfer indicator；
- shared raw/weighted contribution metrics；
- checkpoint/common-support/censoring和artifact validity。

runner/report只记录raw结果与完整性，不输出cross-dataset结论。

## 10. Acceptance criteria for reviewer

Reviewer重点审查算法是否正确实现；最多三轮复审，不构建无关防御框架。

### 10.1 Dataset and split correctness

- processed row count、split boundaries、domain/task counts与`stage_E4_dataset_audit.md`逐项一致；
- domain mapping d0=tab0、d1=tab1、d2=tab4正确；
- split fingerprints在三臂间一致；
- oracle严格先全局split后per-split filter；
- weighting使用train proportions，不使用full/val/test比例。

### 10.2 Weighting correctness

- alpha=0/0.5/1和§5.2 coefficient表在`1e-8`内一致；
- 每项正、domain和3、六cell和6；
- task-major顺序正确，两个tasks系数相同；
- synthetic six-loss点积与实际loss helper一致。

### 10.3 Training fidelity

- queue恰为12 joint + 12 oracle；
- A/B/C除weighting/output identity外训练config相同；
- natural sampling、no scheduler、optimizer grouping、architecture update与checkpoint符合本规格；
- shared observer不改变真实gradient flow；
- 无禁止机制。

### 10.4 Minimal focused verification

无需在review阶段跑完整数据epoch。只要求：

1. synthetic coefficient/loss/gradient tests；
2. 从真实processed data加载并只执行同一seed的前2个batches A/B/C smoke：initial hash和batch fingerprints相同、loss coefficient参与backward、weighted norm identity与gradient reconstruction通过；
3. 一个physical domain的2-batch oracle smoke，核对split-first/filter-second row fingerprint与original/local mapping；
4. report能读取smoke schema，且smoke不能被resume视为formal。

不重复E1/E3完整ON/OFF防御验证，除非发现observer改变训练路径的具体证据。

### 10.5 Numeric/logging gates

- shared reconstructed gradient relative L2 error `<=1e-5`；
- signed contribution非退化step求和误差`<=1e-5`；
- `weighted_norm=coefficient×raw_norm`在float误差内；
- selected checkpoint的六个joint cells或两个oracle tasks完整finite；
- transfer方向为joint-oracle；
- macro与pooled字段不可互换。

Reviewer不得以结果是否支持H-K作为PASS条件。第三次仍有影响objective、split、oracle、gradient flow或matrix的blocking bug，则FAIL并交回主代理，不无限加固。

## 11. Conditions that would falsify hypotheses

### 11.1 Falsify/strongly weaken H-K

1. sample相对equal未稳定缩小d1负迁移，或四seed方向冲突且效应小于0.001；
2. sample改善weighted contribution但overall/worst-domain/transfer不改善；
3. sample对d0/d2产生稳定>0.001损害，导致worst-domain或negative-transfer rate恶化；
4. sqrt/sample响应不呈任何稳定顺序，且raw gradient/性能曲线跨seed不一致；
5. effect只存在于一个task，另一个task稳定反向；
6. common-support分析不复现selected-checkpoint差异；
7. matched oracle缺失或时间边界不一致，则transfer机制结论不成立，只能Insufficient evidence。

### 11.2 Falsify H0

sample或sqrt在四seed上对预注册核心指标给出一致、超过practical margin且seed-cluster CI排除0的效果，并伴随shared contribution结构变化，则“无稳定law”被拒绝。

### 11.3 Support H-R

若sample相对equal在四seed中稳定降低macro overall/worst-domain或显著增加minority negative transfer，即使d1改善，也支持自然异质性下的reversal/trade-off，反对sample作为通用静态解。

## 12. Main analysis plan

### 12.1 Statistical unit and evidence strength

- independent replication unit：seed；n=4；
- domain/task/epoch不是额外独立样本；
- 所有方法比较为同seed paired differences；
- 报告每seed原始值、均值、sample std、seed bootstrap CI与方向一致性；
- practical margin：AUC/transfer=0.001，LogLoss=0.001；negative-transfer rate按实际cell变化同时报告，不以极小比例变化夸大结论。

Evidence labels：

- **STRONG**：mean effect超过margin、4/4 seeds同方向、seed-bootstrap 95% CI排除0，且两个tasks无>0.001稳定反向；
- **MEDIUM**：mean超过margin，并满足CI或至少3/4方向一致之一；
- **WEAK**：其余、仅描述性历史对照或oracle coverage不完整。

四seed统计能力有限；未显著不等于证明等价。必须报告effect interval。

### 12.2 Q1 — Equal dominant-domain degradation

主要检验 equal 下 d1 两task的matched transfer。若domain-average transfer<-0.001、4/4 seeds为负且CI排除0，判为稳定dominant-domain degradation。分别报告两个tasks，不能由均值掩盖一正一负。

### 12.3 Q2 — Sample improvement

计算 sample-equal paired delta：d1 transfer、overall macro AUC、worst-domain AUC、negative-transfer rate和macro LogLoss。H-K的强支持要求d1 transfer改善>0.001且4/4同方向，同时overall/worst-domain不劣于equal超过0.001。若只改善d1但损害minority/worst，则为trade-off而非通用改善。

### 12.4 Q3 — Sqrt comparison

sqrt分别与equal和sample做固定paired comparison，不在test上先选最佳再检验。只有sqrt对两个端点均改善overall macro AUC>0.001、CI排除0、至少3/4 seeds同方向且worst-domain不劣时，才称sqrt稳定优于两端。

### 12.5 Q4 — MovieLens law transfer

比较三点ordered response与E3预注册方向：

- d1/dominant transfer；
- overall/worst-domain；
- shared magnitude share/mismatch；
- d0/d2 minority transfer；
- task consistency。

不能把MovieLens effect size直接作为KuaiRand阈值，也不能因为sample不是绝对最好就自动判为contradicted。

### 12.6 Q5 — Historical Comp1 explanation

E4不含Comp1 factorial，因此只能判定“objective mismatch explanation是否相容”，不能证明Comp1被weighting mediation：

- 若sample修复的domain/task方向与历史Comp1变化一致，标记compatible，证据WEAK；
- 若sample表现稳定而历史Comp1失败，说明Comp1的其他机制可能抵消objective收益；
- 若sample无效/恶化，则objective mismatch不足以解释Comp1 failure；
- 真正因果归因需要后续`Comp1 on/off × equal/sample` factorial，必须另走完整workflow。

历史raw artifact当前缺失时，只引用已审计报告中的方向，不做新的显著性或seed级合并。

### 12.7 Q6 — Temporal shift competition

使用dataset audit固定事实及各arm逐epoch/各split指标：

- domain-mixture TV(train,test)=0.112839；
- d1 label-prior大幅变化；
- macro与pooled结果是否分歧；
- weighting effect是否在val/test与task间一致。

若A/B/C均无法改善d1 transfer，且误差集中在label shift最大的d1 tasks，temporal shift成为更强竞争解释；但E4仍不能称其为因果主因。不得据此在本阶段追加temporal schedule。

## 13. Cross-dataset mechanism verdict

Main Agent必须按以下优先级给出唯一结论，并标证据强度：

1. **Insufficient evidence**：formal matrix/oracle coverage不完整、关键identity无效，或四seed区间太宽无法区分>0.001效果。
2. **Supported**：sample相对equal稳定改善d1 transfer及overall/worst-domain，降低或不增加negative transfer，shared contribution方向与MovieLens一致，且无minority稳定损害。
3. **Contradicted**：sample对预注册主要指标呈稳定反向效果，例如d1 transfer不改善且overall/worst-domain稳定恶化。
4. **Dataset-dependent**：KuaiRand存在稳定但不同的law，例如sqrt优于两端、sample只改善d1却损害minority，或task/domain规律系统分化。
5. **Partially supported**：objective weighting在部分核心指标/domain/task上复现，但未满足Supported，且没有达到Contradicted或稳定Dataset-dependent模式。

决策后续：

- Supported：objective mechanism获得跨数据集支持，只批准下一轮adaptive-method DESIGN；
- 无作用/Partially/Insufficient：优先研究temporal shift与data heterogeneity；
- Contradicted或Dataset-dependent：固定sample proportional不是通用解，可形成adaptive研究动机，但仍须独立因果设计；
- 无论何种结果，本E4不得回填或运行E3建议的temporal schedule。
