# DESIGN SPECIFICATION — Stage E1：Shared Optimization Contribution 机制诊断

本次 DESIGN 严格只读取了指定提示词文件 `promt/验证性补全和实验方向寻找/group1.md`，没有读取、搜索或接触其他提示词文件。本轮仅设计 `group1.md` 中顺序最前的 E1，不提前设计 E2；未修改代码、未运行实验、未生成新结果。

## 1. Research question

在 MovieLens-1M、固定总训练样本量且严重域规模失衡的 r50 条件下：

> equal domain/task weighting 导致 Large-role domain 负迁移，而 sample-proportional weighting 缓解该现象，是否主要因为两种 loss aggregation 改变了各 domain/task 对 shared parameters 的有效优化贡献？

需要区分：

1. loss coefficient 带来的必然、即时缩放；
2. 不同 weighting 训练轨迹造成的 raw gradient geometry、方差、SNR 和更新对齐变化；
3. shared、domain-specific、task-specific、cell-specific 参数上的变化；
4. routing topology、checkpoint timing、physical-domain difficulty 等竞争解释。

E1 是观察性 instrumentation，不得把相关性写成因果性。无论结果支持与否，完成分析后均进入 E2 的 objective × sampling 因果拆分。

## 2. Hypothesis being tested

### 主假设 H1：Shared contribution distortion

equal weighting 在 r50 下使 Small-role domain 对 shared optimization 的贡献相对其样本占比过高，并使 Large-role domain 的贡献结构不利；sample weighting 改变该结构，其轨迹变化先于或伴随 Large AUC 修复。

### 零假设 H0

除 loss coefficient 的确定性乘法外，两臂在 shared-gradient geometry、trajectory-mediated contribution、actual Adam update alignment 与 performance trajectory 上不存在稳定、可重复关系。

### 竞争解释

- H2—Raw-gradient/noise：现象主要由各域原始梯度范数、方差或 SNR 差异解释，而非 weighting 后的 shared contribution。
- H3—Private-parameter：关键变化主要发生在 domain/task/cell-private 参数，而非 shared components。
- H4—Routing：sample weighting 主要改变 expert routing topology，shared contribution 只是伴随变量。
- H5—Checkpoint/convergence：差异主要源于收敛速度或 early-stop 选点，而不是训练轨迹中的结构变化。
- H6—Physical-domain/difficulty：效应跟随固定物理域或基线难度，而不随 Large/Medium/Small size role 轮换。
- H7—Instrumentation artifact：额外梯度观测或逐 epoch test evaluation 改变了 RNG、batch 顺序或参数轨迹。

## 3. Variables changed

唯一训练自变量是 `loss_weighting`：

- Condition A：`equal`
- Condition B：`sample`

实验矩阵：

- weighting：2
- role assignment：3
- seed：4
- 新正式 run：`2 × 3 × 4 = 24`

Role assignments：

- `d0_large`：d0=L，d1=M，d2=S
- `d1_large`：d0=S，d1=L，d2=M
- `d2_large`：d0=M，d1=S，d2=L

Seeds 固定：

- 2022
- 2023
- 2024
- 2025

只读诊断维度，不构成额外训练臂：

- parameter group：shared / domain-specific / task-specific / cell-specific
- actual versus counterfactual weighting evaluation
- physical domain versus size role
- training epoch/step
- task 0/1
- gate/expert utilization

禁止加入 sqrt weighting、Curriculum、Comp1、differential momentum、EMA、Polyak、新 optimizer、新 router 或额外 sampling strategy。

## 4. Variables held constant

全部沿用 P0-4 r50 baseline 协议：

- Dataset：MovieLens-1M
- Model/backbone：MDMTRec/M3OE，`backbone=mdmoe`
- D=3，T=2，shared experts=4
- fixed total training N=187,500
- role sizes：L/M/S = 150,000/34,500/3,000
- domain shares：pL=0.800，pM=0.184，pS=0.016
- train/validation/test split：既有 80%/10%/10% 时间顺序切分
- role subsample：沿用 `RandomState(seed*10+d)` 规则
- batch size：40,960
- epochs upper bound：50
- early-stop patience：10
- checkpoint objective：6-cell equal mean validation AUC
- main optimizer：Adam，lr=1e-2，betas=(0.9,0.999)
- architecture-weight optimizer：Adam，lr=1e-3
- weight decay：1e-5
- `fix_optim_grouping=1`
- 无激活 LR scheduler
- model initialization、batch sequence、validation cadence、architecture update cadence全部匹配
- `_weight_*` architecture parameters 仍按既有独立 optimizer 和 unmasked batch BCE 路径更新，两臂一致
- 所有额外评估/观测必须保存并恢复 CPU 与 CUDA RNG state

同一 `(setting, seed)` 的 equal/sample 必须具有相同：

- 训练样本集合与顺序候选；
- 初始 state_dict；
- 每 step batch fingerprint；
- domain/task label distribution；
- validation/test 数据；
- instrumentation cadence。

Domain size 与 domain difficulty 必须分开记录：

- size：训练样本数和样本占比；
- operational difficulty：同 setting、seed、physical domain、task 的既有 single-domain oracle AUC/LogLoss；
- physical-domain intrinsic difficulty 辅助量：全量数据 baseline；
- label prevalence：每 split、domain、task 正例率。

不得用 Small 作为 difficult 的同义词。

## 5. Mathematical/algorithmic definition

### 5.1 Training objectives

对 batch 中 domain d、task t 的 masked-mean BCE：

\[
\ell_{dt}(\theta)=\frac{1}{|\mathcal B_d|}
\sum_{i\in\mathcal B_d}\operatorname{BCE}
(\hat y_{i,t},y_{i,t})
\]

Condition A：

\[
L_{\mathrm{equal}}=\sum_{d=0}^{2}\sum_{t=0}^{1}\ell_{dt},
\qquad a^{E}_{dt}=1
\]

Condition B：

\[
L_{\mathrm{sample}}=\sum_{d=0}^{2}\sum_{t=0}^{1}3p_d\ell_{dt},
\qquad a^{S}_{dt}=3p_d
\]

r50 的 role-based weights：

- L：2.400
- M：0.552
- S：0.048

每个 task 重复同一 domain weight；六个 coefficient 总和均为 6，因此总体 loss scale 与有效学习率尺度保持一致。

### 5.2 Parameter partition

参数分区必须互斥、可审计：

- `shared`
  - `embedding.*`
  - `shared_weight`
  - `shared_bias`
  - `skip_conn.*`
  - `star_mlp.*`
  - `expert.*`
- `domain_specific`
  - `slot_weight.*`
  - `slot_bias.*`
  - `domain_expert.*`
- `task_specific`
  - `task_expert.*`
- `cell_specific`
  - `gate.*`
  - `tower.*`
- `architecture`
  - `_weight_exp_d*`
  - `_weight_exp_t*`
  - `_weight_bal_d*`
  - `_weight_bal_t*`

`architecture` 单独记录但不纳入 shared contribution 主分析。不得沿用 P1 中把 `domain_expert`、`task_expert` 一并称为 shared 的旧口径。

### 5.3 Raw and weighted gradients

对参数组 \(P\)：

\[
g^{P}_{dt,k}=\nabla_{\theta_P}\ell_{dt,k}
\]

\[
h^{P}_{dt,k}=a_{dt}g^{P}_{dt,k}
\]

\[
G^{P}_k=\sum_{d,t}h^{P}_{dt,k}
\]

必须分别记录：

\[
r^{P}_{dt,k}=\|g^{P}_{dt,k}\|_2
\]

\[
m^{P}_{dt,k}=\|h^{P}_{dt,k}\|_2
=a_{dt}r^{P}_{dt,k}
\]

并记录 parameter-count normalized RMS：

\[
\operatorname{RMSGrad}^{P}_{dt,k}
=\frac{\|g^{P}_{dt,k}\|_2}{\sqrt{|P|}}
\]

跨参数组比较必须用 RMSGrad，不得直接比较不同参数量组的 L2 norm。

### 5.4 Effective contribution

主贡献定义采用对 combined training gradient 的有符号投影：

\[
e^{P}_{dt,k}
=
\frac{\langle h^{P}_{dt,k},G^{P}_k\rangle}
{\|G^{P}_k\|_2+\epsilon}
\]

其单位与 gradient norm 一致，表示该 cell 沿实际 combined-gradient 方向提供的更新强度；负值表示与 combined direction 对抗。

归一化贡献：

\[
q^{P}_{dt,k}
=
\frac{\langle h^{P}_{dt,k},G^{P}_k\rangle}
{\|G^{P}_k\|_2^2+\epsilon}
\]

在非退化情况下：

\[
\sum_{d,t}q^{P}_{dt,k}=1
\]

domain contribution：

\[
q^P_{d,k}=\sum_t q^P_{dt,k}
\]

同时记录非负 magnitude share：

\[
s^P_{d,k}
=
\frac{\sum_t\|h^P_{dt,k}\|_2}
{\sum_{j,t}\|h^P_{jt,k}\|_2+\epsilon}
\]

相对样本占比：

\[
\operatorname{overrep}^P_{d,k}
=\frac{s^P_{d,k}}{p_d}
\]

整体 mismatch：

\[
D^P_k=\frac12\sum_d|s^P_{d,k}-p_d|
\]

`weight × gradient norm` 只能作为 magnitude 指标，不能单独证明机制；主分析必须同时使用 signed projection、actual-update alignment 和 counterfactual decomposition。

### 5.5 Counterfactual decomposition

每次只计算一套 raw gradients，但在日志层同时施加两套系数：

\[
h^{P,E}_{dt}=a^E_{dt}g^P_{dt}, \qquad
h^{P,S}_{dt}=a^S_{dt}g^P_{dt}
\]

因此每个模型状态均产生：

- actual weighting contribution；
- equal-counterfactual contribution；
- sample-counterfactual contribution。

用于区分：

1. direct coefficient effect：固定 \(\theta\) 和 batch，仅替换 coefficient；
2. trajectory/state effect：equal-trained 与 sample-trained 模型在共同 reference weighting 下比较；
3. total actual effect：两臂各自 actual contribution 的差异。

只有 direct coefficient effect 是数学必然；若没有 trajectory/state effect、update alignment 或时间关系，不得称为机制证据。

### 5.6 Actual Adam update alignment

在观测 step 保存 optimizer step 前后的 shared 参数，定义下降方向：

\[
u_k^P=\theta^P_{\mathrm{before}}-\theta^P_{\mathrm{after}}
\]

记录：

\[
A^P_{dt,k}
=
\cos(h^P_{dt,k},u^P_k)
\]

\[
U^P_{dt,k}
=
\frac{\langle h^P_{dt,k},u^P_k\rangle}
{\|u^P_k\|_2+\epsilon}
\]

该量包含 Adam 历史动量、预条件和 weight decay 的影响，只作为“当前 cell gradient 与真实参数更新是否对齐”的诊断，不解释为严格 Adam attribution。

### 5.7 Gradient variance and SNR

在每个 epoch 的实际训练 steps 上，用在线 Welford 统计 raw gradient：

\[
\bar g^P_{dt,e}
=\frac1{K_e}\sum_{k=1}^{K_e}g^P_{dt,k}
\]

\[
V^P_{dt,e}
=
\frac1{K_e-1}
\sum_k\|g^P_{dt,k}-\bar g^P_{dt,e}\|_2^2
\]

\[
\operatorname{SNR}^P_{dt,e}
=
\frac{\|\bar g^P_{dt,e}\|_2^2}
{V^P_{dt,e}+\epsilon}
\]

同时记录：

- `mean_gradient_norm`
- `norm_of_mean_gradient`
- `gradient_variance_trace`
- `weighted_gradient_variance = a²V`
- `gradient_SNR`
- `probe_count`
- 每个 probe 的 cell sample count

正 coefficient 缩放不会改变同一状态下的 SNR；两臂 SNR 差异只能来自训练状态演化，不得将 coefficient 的确定性变化误写为 SNR 改善。

### 5.8 Pairwise cosine

domain aggregate raw gradient：

\[
g^P_{d,k}=\sum_t g^P_{dt,k}
\]

\[
\cos^P_{d,d',k}
=
\frac{\langle g^P_{d,k},g^P_{d',k}\rangle}
{\|g^P_{d,k}\|_2\|g^P_{d',k}\|_2+\epsilon}
\]

记录 d0-d1、d0-d2、d1-d2；cosine 仅为辅助指标，不作为主要假设检验。

### 5.9 Routing state

通过只读 hook 记录 cell 对应 gate 的：

- 每个 shared expert 的 mean probability；
- gate entropy；
- effective expert count \(e^{H(g)}\)；
- top-1 expert frequency；
- domain-level task average。

只统计 cell 对应 physical domain 的样本。不得修改 forward 返回值、模型参数或 state_dict。

## 6. Exact affected modules/files

建议 coder 采用隔离实现，避免污染核心训练器：

### 新增

- `experiments/stage_E1_run.py`
  - E1 单 run 入口；
  - 复用 P0-4 数据、模型、weighting、checkpoint 语义；
  - prediction、gradient、update 和 routing instrumentation。
- `experiments/stage_E1_instrumentation.py`
  - 参数分区；
  - gradient norm/dot/Welford；
  - counterfactual contribution；
  - gate observer；
  - schema writer。
- `experiments/stage_E1_orchestrator.py`
  - 24-run 矩阵、resume、锁、日志与 config。
- `experiments/stage_E1_report.py`
  - 只做结构化汇总，不作科研结论。
- `experiments/tests/test_stage_E1_instrumentation.py`
- `experiments/tests/test_stage_E1_equivalence.py`

### 主要复用/参考，不应无必要修改

- `experiments/history_exp/first/p4_run.py`
  - role reversal、loss weighting、metrics/checkpoint/RNG 保护的既有参考实现。
- `experiments/history_exp/first/p1_run.py`
  - 旧梯度 instrumentation 参考；不得复用其旧 shared 参数定义。
- `m3oe/MDMTRec/models/multi_domain/model.py`
- `m3oe/MDMTRec/trainers/ctr_trainer.py`
- `m3oe/MDMTRec/utils/data.py`
- `m3oe/examples/run_movielens.py`

如必须修改核心文件，所有新逻辑必须由默认关闭的显式 E1 开关保护；不开启时行为与当前 baseline 一致。

### 产物

- `experiments/reports/stage_E1_design.md`
- `experiments/reports/stage_E1_implementation.md`
- `experiments/reports/stage_E1_review.md`
- `experiments/stage_E1_raw_results/`
- `experiments/stage_E1_summary.csv`

## 7. Implementation requirements

1. 正式训练必须仍调用原始 loss backward 和 optimizer step；观测使用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`，不得把 observer gradient 写入 `.grad`。
2. 六个 cell 顺序固定为 `idx=t*D+d`。
3. 每个实际训练 step 均观测 shared gradients，以获得每 epoch 约 5 个 probe；一次 raw gradient 计算同时服务 actual/counterfactual 两套 weighting，禁止重复 forward。
4. domain/task/cell-private 使用同一次 autograd 调用返回的梯度做 norm/RMS；完整 Welford tensor至少对 shared 组实施，其他组若内存受限可记录 norm dispersion，但必须在 implementation report 明确。
5. `allow_unused` 返回 None 时按零向量统计，但必须记录 `unused_param_count`，不得静默丢弃。
6. 每个 probe 后重建实际 combined shared gradient，并与真实 `loss.backward()` 得到的 `.grad` 比较。
7. 训练指标用 detached prediction：
   - per-domain/task exact sample-mean BCE；
   - per-domain/task train AUC；
   - 不得用“batch BCE mean 再等权平均”近似 epoch loss。
8. 每 epoch 记录 train/validation/test AUC 和 LogLoss。额外 DataLoader 迭代必须保存并恢复：
   - `torch.get_rng_state()`
   - `torch.cuda.get_rng_state_all()`
9. 同 setting/seed 的 equal/sample 必须记录 batch fingerprint；任何不匹配使该 paired run 无效。
10. 记录每 split、domain、task 的样本数、正例率和物理域/size role。
11. routing hook 只能 detach 后累计 scalar，不保留 graph。
12. actual update alignment 只能在主 optimizer step 前后读取参数，不得额外 step 或操作 optimizer state。
13. checkpoint 始终由 equal 6-cell validation AUC mean 决定；test metric 绝不参与选点。
14. early stop 后的缺失 epoch 标记为 censored，不做前向填充。
15. 不新增 defensive framework；复用既有 CSV/config/stdout 格式。
16. baseline linkage：
   - 优先只读复用 P0-2 r50 同 seed oracle；
   - 不为 E1 新跑 oracle；
   - 若原始 oracle artifact 不可用，字段标记 missing，并在分析中仅使用 equal/sample paired difference，严禁从报告均值伪造 seed-level baseline。

## 8. Ablations required

E1 不增加训练方法，所需对照为：

1. 核心训练对照：equal vs sample。
2. Role-rotation falsification：d0_large、d1_large、d2_large。
3. Parameter-location diagnostic：
   - shared
   - domain-specific
   - task-specific
   - cell-specific
4. Counterfactual weighting diagnostic：
   - actual
   - equal_cf
   - sample_cf
5. Geometry versus routing：
   - shared contribution trajectory
   - gate/expert allocation trajectory
6. Optimization versus selection：
   - all common-support epochs
   - best equal-selection checkpoint
   - terminal observed epoch
7. Instrumentation bypass：
   - observer OFF
   - observer ON
   - 同 seed/setting/weighting 的短程等价性检查

明确禁止把 sqrt、Momentum、Curriculum、EMA 或新 backbone 当作本阶段 ablation。

## 9. Expected logging requirements

### 每 run 的原始目录

`stage_E1_raw_results/<weighting>/<setting>/seed<seed>/`

至少包含：

- `config.json`
- `stdout.log`
- `run_manifest.json`
- `metrics_epoch.csv`
- `gradient_step.csv`
- `gradient_epoch.csv`
- `gradient_cosine_epoch.csv`
- `gate_epoch.csv`
- `batch_trace.csv`
- `metrics_summary.csv`

### `metrics_epoch.csv`

每 `(seed, epoch, domain, task)`：

- weighting
- setting
- physical_domain
- size_role
- domain_size
- domain_share
- train/val/test sample count
- positive rate
- train AUC / LogLoss
- validation AUC / LogLoss
- test AUC / LogLoss
- selected checkpoint flag
- stopped/censored flag
- oracle baseline AUC/LogLoss 与来源（如可用）
- transfer AUC/LogLoss（仅 baseline 可验证时）

### `gradient_step.csv`

每 `(epoch, step, domain, task, parameter_group)`：

- batch fingerprint
- cell sample count
- loss coefficient
- raw gradient norm/RMS
- weighted gradient norm/RMS
- effective projection
- normalized contribution
- magnitude share
- overrepresentation ratio
- equal_cf/sample_cf contribution
- actual-update norm、alignment、projection
- unused parameter count
- gradient reconstruction error

### `gradient_epoch.csv`

至少：

- seed
- epoch
- domain
- task
- weighting
- parameter group
- gradient norm
- weighted gradient norm
- gradient variance
- weighted variance
- gradient SNR
- norm of mean gradient
- probe count
- contribution mean/std
- mismatch index

### `stage_E1_summary.csv`

每 seed 必须保留，不得只存总体均值。至少含提示词要求的：

- seed
- epoch
- domain
- task
- weighting
- AUC
- LogLoss
- gradient_norm
- weighted_gradient_norm
- gradient_variance
- gradient_SNR

并增加 setting、role、domain size/share、split、parameter group、contribution、baseline/transfer、routing fields。

## 10. Acceptance criteria for reviewer

Reviewer 只有满足以下条件才可给 PASS 或 PASS WITH FIXES：

### 实现忠实性

- 正式矩阵严格为 24 run；
- config diff 显示 equal/sample 除 loss coefficient 外无训练设置差异；
- 无禁止机制被启用；
- loss weights 精确为 1 或 role-based 2.400/0.552/0.048；
- 两臂六项 coefficient 总和均为 6。

### 参数分区

- 非 architecture 的 main-optimizer 参数被四个主组完全覆盖；
- parameter group 两两无交集；
- `_weight_*` 不进入 shared；
- `domain_expert` 不进入 shared；
- `task_expert` 不进入 shared；
- 每 run 保存参数名清单、param count 和 hash。

### 数学不变量

- `weighted_gradient_norm ≈ coefficient × raw_gradient_norm`；
- 所有 cell 的 normalized contribution 和为 1，误差 ≤1e-5；
- domain contribution 等于两 task contribution 之和；
- cosine 在 [-1,1]；
- variance 非负；
- actual combined gradient 与 real backward `.grad` 的 shared relative L2 error ≤1e-5。

### 无损性

至少执行 3-epoch、seed 2022、同一 setting 的 instrumentation OFF/ON 检查：

- batch fingerprints 全同；
- per-step training loss 最大绝对差 ≤1e-7；
- validation/test AUC 最大绝对差 ≤1e-8；
- final parameter max absolute difference 为 0；若运行环境只能达到非零浮点差，必须定位来源并证明低于既有环境漂移，不能直接放宽。
- 额外 test evaluation 前后 CPU/CUDA RNG state 精确恢复。

### 日志完整性

- Train/validation/test AUC 和 LogLoss 均逐 epoch、domain、task 保存；
- test LogLoss 不缺失；
- 每 epoch 至少两个有效 gradient probes，否则 variance/SNR 标记 NA 并解释；
- batch fingerprint 在 paired equal/sample 之间完全匹配；
- 所有 NaN/Inf 均被列出且有原因；
- schema 单测通过；
- stage implementation report 列出修改文件、命令、检查结果和已知限制。

Reviewer 不以“结果是否支持 H1”为验收条件；阴性结果同样是合格结果。

## 11. Conditions that would falsify the hypothesis

以下任一核心模式可削弱或证伪 H1：

1. sample 仅产生 coefficient 导致的即时 `weight × norm` 变化；在 common counterfactual weighting 下，两臂没有稳定 trajectory/state effect。
2. shared contribution mismatch 的变化不早于 Large AUC divergence，或者系统性晚于性能变化。
3. sample 修复 Large AUC，但 shared signed contribution、actual-update alignment 和 mismatch 均无相应结构变化。
4. 主要变化位于 domain/task/cell-private 参数，shared 组效应弱或不稳定。
5. routing topology 的稳定变化先于 shared contribution，并比 contribution 更一致地解释 AUC。
6. 现象固定跟随 d0/d1/d2，而不是随 L/M/S role 轮换。
7. contribution 变化由 oracle difficulty、正例率或 raw SNR 更好解释，domain size 不再是可信主解释变量。
8. 三个 role rotations 中仅一个成立，或 12 个 setting×seed 配对方向接近随机。
9. equal 下 Small contribution 并未相对其 pS=0.016 过度，或 sample 反而扩大 mismatch。
10. instrumentation ON/OFF 改变 batch sequence、RNG 或参数轨迹，导致观测无法解释。
11. Large AUC 改善只来自更早/更晚 checkpoint，而 common-support trajectory 没有稳定改善。

证据分级预注册：

- STRONG：3/3 role rotations 同向，且每个 rotation 至少 3/4 seeds 同向；trajectory/state contribution 先于或同步于 Large 性能变化；shared 证据强于 private/routing 竞争解释。
- MEDIUM：2/3 rotations 且总体至少 8/12 paired runs 同向，但时间顺序或参数定位不完整。
- WEAK：只有 arithmetic coefficient effect、单一 physical domain、少数 seed 或事后 checkpoint 关系。
- 不得根据 weighted norm 的必然缩放单独给出 STRONG 证据。

## 执行顺序与分析判据

Coder 完成后，reviewer 必须独立检查设计、代码与 implementation report；只有 PASS 或修复完阻塞问题后的 PASS WITH FIXES 才能交 runner。

Runner 仅保存 24 个正式 run 的原始数据，不分析。Main analysis 应报告：

- mean、std、median、min/max；
- seed-level paired difference 和 improved-seed count；
- per-domain、per-task、role aggregate；
- shared/private 参数位置；
- domain size 与 operational difficulty；
- baseline/transfer；
- common-support trajectory 与 censoring；
- contribution onset 和 AUC divergence onset；
- direct coefficient、trajectory/state、total actual 三部分；
- stable phenomena、contradictions、competing explanations；
- Observed / Supported / Hypothesized；
- STRONG / MEDIUM / WEAK。

## 本设计依据的现有文件

- `D:\CODE\phase0-3\promt\验证性补全和实验方向寻找\group1.md`
- `D:\CODE\phase0-3\PROJECT_ARCHITECTURE.md`
- `D:\CODE\phase0-3\experiments\reports\P0_2_RAW_REPORT.md`
- `D:\CODE\phase0-3\experiments\reports\P0_4_RAW_REPORT.md`
- `D:\CODE\phase0-3\experiments\reports\P0_4_STAGE_REVIEW.md`
- `D:\CODE\phase0-3\experiments\reports\P0_5_RAW_REPORT.md`
- `D:\CODE\phase0-3\experiments\reports\P0_5_STAGE_REVIEW.md`
- `D:\CODE\phase0-3\experiments\reports\P1_diagnosis.md`
- `D:\CODE\phase0-3\experiments\reports\P1_2_STAGE_REVIEW.md`
- `D:\CODE\phase0-3\experiments\reports\research_diagnosis_20260909\RESEARCH_DIAGNOSIS.md`
- `D:\CODE\phase0-3\experiments\reports\research_diagnosis_20260909\EXPERIMENT_EVIDENCE_TABLE.md`
- `D:\CODE\phase0-3\experiments\history_exp\first\p1_run.py`
- `D:\CODE\phase0-3\experiments\history_exp\first\p4_run.py`
- `D:\CODE\phase0-3\experiments\history_exp\second\p0_4_orchestrator.py`
- `D:\CODE\phase0-3\experiments\tests\test_p0_4_weighting.py`
- `D:\CODE\phase0-3\experiments\tests\test_equivalence.py`
- `D:\CODE\phase0-3\m3oe\MDMTRec\models\multi_domain\model.py`
- `D:\CODE\phase0-3\m3oe\MDMTRec\trainers\ctr_trainer.py`
- `D:\CODE\phase0-3\m3oe\MDMTRec\utils\data.py`
- `D:\CODE\phase0-3\m3oe\examples\run_movielens.py`
