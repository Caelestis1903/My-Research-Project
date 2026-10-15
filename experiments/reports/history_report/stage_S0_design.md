# DESIGN SPECIFICATION

## 所选阶段

**Stage 0：Matched Momentum Closure（匹配动量封口实验）**

仓库状态判定：

- `promt/saft_mdmtl.md` 的顺序入口是 Stage 0。
- 历史 `P1-3` 脚本虽已包含 uniform EMA / Adam β1 控制的雏形，但仓库中没有对应原始结果或正式阶段报告，不能视为已执行。
- 旧脚本的 `γ*=0.94318` 被既有诊断报告指出只匹配“按样本占比加权的时间常数”，并非唯一合理的 memory matching；只击败该单点不能证明 domain-conditioned 结构独有。
- `stage_E1`–`stage_E7` 属于先前研究计划，不能据其编号跳过本计划的 Stage 0。

本规格只设计 Stage 0。**未设计 Stage 1、Stage 2 或任何后续阶段。**

---

## 1. Research question

在 sample-proportional objective 已控制大部分 objective mismatch 后，历史上约 `+0.0054` 的 comp1 残余 AUC 增量能否新鲜复现；若能，它是否必须依赖 domain-conditioned per-domain momentum，还是可由下列简单机制解释：

1. 普通 uniform temporal smoothing；
2. Adam β1 / effective first-moment memory 改变；
3. checkpoint selection 或收敛速度差异；
4. 外层 EMA 无偏差校正造成的启动期隐式域权重；
5. 随机种子、执行窗口或旧结果复用造成的偶然差异。

Stage 0 只做旧 Momentum 路线的封口，不开发新 momentum 方法。

---

## 2. Hypotheses being tested

### H-M（待证伪主假设）

在相同 sample objective、数据、初始化、batch 顺序、训练步数和 checkpoint 规则下，domain-conditioned comp1 比经过公平匹配和小规模 validation-only 选择的 uniform EMA 与 Adam β1 controls 均有稳定、具有实际量级且无 role-level 明显代价的优势。

### 竞争解释

- **H-U：普通平滑解释。** 任一合理 uniform EMA control 能复现 comp1 的主要增量。
- **H-A：普通 Adam memory 解释。** 调整共享参数的 Adam β1 即可复现 comp1。
- **H-C：checkpoint/convergence 解释。** comp1 只在 validation-selected checkpoint 占优，在固定步数或共同 epoch 不占优。
- **H-I：启动期隐式权重解释。** 无偏差校正的多 γ EMA 主要通过早期系数 `1-γ_d^k` 改变域贡献，并非稳态 domain-conditioned smoothing 的独立收益。
- **H-N：随机波动解释。** `+0.0054` 无法在四个固定 seeds 和三种 role rotations 中稳定复现。
- **H-R：role redistribution。** overall gain 来自牺牲某个 Large/Medium/Small role，而不是无明显代价的改善。

不把 Curriculum、Momentum 有效、M3OE 特有失败、域大小或梯度冲突预设为真。

---

## 3. Variables changed

### 3.1 正式训练臂

所有训练臂均为 fresh runs，不复用历史 A/B 结果作为正式证据。

| 臂 | 共享梯度处理 | 共享 Adam β1 | 作用 |
|---|---|---:|---|
| A `sample_adam` | 无外层 EMA | 0.9 | Sample baseline |
| B `comp1_domain` | domain-conditioned EMA | 0 | 当前 comp1 |
| C0 `uniform_090` | uniform EMA，γ=0.9 | 0 | uniform 边界控制 |
| C1 `uniform_p_tau` | uniform EMA，γ=0.9431818 | 0 | 样本占比加权 memory match，预注册主 matched-uniform control |
| C2 `uniform_eq_tau` | uniform EMA，γ≈0.980453 | 0 | 等域平均 memory match |
| D1 `adam_p_tau` | 无外层 EMA | 0.9431818 | 预注册主 matched-Adam control |
| D2 `adam_eq_tau` | 无外层 EMA | ≈0.980453 | Adam memory sensitivity control |

正式矩阵：

- 7 arms
- 3 large-domain role rotations：`d0_large`、`d1_large`、`d2_large`
- 4 seeds：2022、2023、2024、2025
- **共 84 个 fresh joint runs**

不得依据 test 结果追加 γ、β1、学习率或 schedule。

### 3.2 Validation-selected simple-control families

为避免只击败任意单一 match 定义：

- `C_val`：在 C0/C1/C2 中，只按同一 validation checkpoint score 选择；
- `D_val`：在 A/D1/D2 中，只按 validation score 选择。

选择必须在每个 `(rotation, seed)` block 内完成，test 完全不可见。B 必须同时与预注册的 C1/D1 和更强的 C_val/D_val 比较。

---

## 4. Variables held constant

- Dataset：MovieLens 当前已审计数据和 split。
- Model：当前 MDMTRec/M3OE 实现；本阶段不比较 backbone。
- 强不平衡训练规模：Large/Medium/Small=`150000/34500/3000`，份额约 `0.800/0.184/0.016`，约 50:1。
- 三种 physical-domain role rotations。
- 两个既有任务及原标签定义。
- Sample-proportional objective。
- 数据缓存、训练/验证/测试划分和 preprocessing fingerprint。
- 每 seed 的初始化参数、数据子集和 batch permutation。
- Batch size `40960`。
- 最大训练预算 `50 epochs`，学习率 `0.01`，DARTS 学习率 `0.001`，weight decay `1e-5`。
- β2=`0.999`；private parameters 均保持 Adam β1=`0.9`。
- architecture optimizer、scheduler、更新频率和所有非 momentum 开关完全一致。
- 不启用 Curriculum、adaptive controller、新 gamma schedule、stacked momentum 或其他新机制。
- 所有臂完成相同 examples processed 和 optimizer steps；不得因 early stopping 实际提前终止某臂。
- 相同 validation-only checkpoint rule。
- 测试集只参与最终报告，不参与 checkpoint、γ、β1 或 control-family 选择。

---

## 5. Mathematical / algorithmic definition

### 5.1 Sample objective

设域份额为 $p_d=n_d/\sum_j n_j$，域数 $D=3$，任务数 $T=2$。保持历史实现的损失尺度：

$$
w_d = Dp_d,
\qquad
L = \sum_{d=1}^{D}\sum_{t=1}^{T} w_d\,\ell_{d,t}.
$$

所有臂必须对 shared、domain-private、task-private 和 architecture parameters 使用同一 objective；禁止某条训练路径记录或优化未加权 loss 而另一条使用加权 loss。

### 5.2 当前 comp1

对 shared parameters：

$$
g_{d,k}
=
\nabla_{\theta_s}
\left(
\sum_t w_d\ell_{d,t}
\right),
$$

$$
m_{d,k}
=
\gamma_d m_{d,k-1}
+
(1-\gamma_d)g_{d,k},
$$

$$
\tilde g_k = \sum_d m_{d,k}.
$$

$$
\gamma_d =
\operatorname{clip}
\left(
1-(1-\gamma_0)\frac{p_d}{p_{\max}},
\gamma_0,\gamma_{\max}
\right),
$$

其中 $\gamma_0=0.9,\gamma_{\max}=0.99$。在当前份额下应为约：

$$
\gamma_L=0.9,\quad
\gamma_M=0.977,\quad
\gamma_S=0.99.
$$

共享参数随后进入 β1=0、β2=0.999 的 Adam；private parameters 保持标准 Adam `(0.9,0.999)`。

### 5.3 Uniform EMA

C0/C1/C2 与 B 使用完全相同的逐域反传、buffer 数量、外层 EMA、β1=0 Adam 和计算路径，唯一变化是：

$$
\gamma_d \equiv \gamma^\* \quad \forall d.
$$

这样 B–C 对比只改变 γ 是否依域而异，不能让 C 使用更便宜或不同的反传路径。

### 5.4 Memory matching

定义：

$$
\tau(\gamma)=\frac{1}{1-\gamma},
\qquad
h_{1/2}(\gamma)=\frac{\log 0.5}{\log \gamma},
\qquad
N_{\mathrm{eff}}(\gamma)=\frac{1+\gamma}{1-\gamma}.
$$

两种预注册 scalar matches：

1. 样本占比加权时间常数：

$$
\tau_p=\sum_d p_d\tau(\gamma_d)=17.6,
$$

$$
\gamma_p^\*=1-\frac1{17.6}=0.9431818.
$$

2. 等域平均时间常数：

$$
\tau_{\mathrm{eq}}=\frac1D\sum_d\tau(\gamma_d)\approx51.159,
$$

$$
\gamma_{\mathrm{eq}}^\*
=
1-\frac1{\tau_{\mathrm{eq}}}
\approx0.980453.
$$

必须明确：单一指数核不能完全复制多个 γ 的混合核；C1/C2 是两种标量匹配，而非数学等价。

### 5.5 Matched Adam controls

D1/D2 不使用 per-domain EMA：

$$
g_k=\sum_d g_{d,k},
$$

共享参数使用标准、带 Adam bias correction 的一阶矩：

$$
v^{(1)}_k
=
\beta_1v^{(1)}_{k-1}
+
(1-\beta_1)g_k,
$$

分别令 $\beta_1=\gamma_p^\*$ 和 $\beta_1=\gamma_{\mathrm{eq}}^\*$。

D 臂不是 B/C 的数学等价，因为其二阶矩输入、bias correction 和外层 EMA 位置不同；它只检验普通 optimizer memory 是否足以解释收益。

### 5.6 Checkpoint protocol

- 所有臂固定训练完整 50 epochs。
- 同时模拟历史 patience=10 的“shadow early-stop”状态，但不得据此停止训练。
- 正式 selected checkpoint 使用所有臂相同的 validation macro-AUC 规则和 first-argmax tie break；为复现历史 residual，选择窗口不得超过 shadow early-stop 本来可见的 epoch。
- 另报告 fixed-final epoch 49 以及共同 epochs `{9,19,29,39,49}`。
- 只有 selected checkpoint 优势而 fixed/common-epoch 不支持时，H-C 不得排除。

### 5.7 Transfer definitions

对匹配的 single-domain/STL oracle：

$$
\Delta^{AUC}_{d,t}
=
AUC^{joint}_{d,t}
-
AUC^{oracle}_{d,t},
$$

$$
\Delta^{LL}_{d,t}
=
LogLoss^{joint}_{d,t}
-
LogLoss^{oracle}_{d,t}.
$$

分别报告：

- raw AUC NT：$\Delta^{AUC}<0$；
- raw LogLoss NT：$\Delta^{LL}>0$；
- practical AUC NT：$\Delta^{AUC}\le-0.003$；
- concordant practical NT：AUC practical NT 且 LogLoss 同时恶化。

现有 oracle 仅可在 seed、split、role subset、task definition 和数据 fingerprint 全部匹配时复用；否则必须在 Stage 0 内补齐 matching oracles，不能用不匹配参考计算 NT rate。

---

## 6. Exact affected modules/files if identifiable

建议仅新增 Stage 0 文件，不改历史报告或原始数据：

- `experiments/stage_S0_run.py`
- `experiments/stage_S0_orchestrator.py`
- `experiments/stage_S0_report.py`
- `experiments/tests/test_stage_S0_matrix.py`
- `experiments/tests/test_stage_S0_momentum.py`
- `experiments/tests/test_stage_S0_pairing.py`
- `experiments/reports/stage_S0_implementation.md`
- `experiments/reports/stage_S0_review.md`
- `experiments/reports/stage_S0_run_report.md`
- 运行后原始目录：`experiments/stage_S0_raw_results/`

可只读复用/导入：

- `experiments/history_exp/first/p4_run.py` 中已验证的 `PerDomainMomentum`、sample weighting 和 role-rotation 数据逻辑；
- 已验证的 matching oracle artifacts；
- 现有 RNG-safe per-epoch evaluation 方式。

不应修改：

- 历史 P0/P1 原始结果；
- Stage E1–E7 artifacts；
- core M3OE 模型定义；
- Curriculum 代码；
- 已封存报告。

若现有历史 runner 无法满足固定 50 epochs、统一 loss logging 或新增审计字段，应在新 `stage_S0_run.py` 中封装/继承，不应改变旧实验语义。

---

## 7. Implementation requirements

1. `build_tasks()` 必须恰好生成 84 个唯一 joint tasks；每臂 12 个。
2. 每个 `(rotation, seed)` 的七臂必须具有相同：
   - initialization hash；
   - train/val/test fingerprint；
   - selected training indices；
   - batch-order hash；
   - examples processed；
   - optimizer steps；
   - scheduler steps。
3. B/C 外层 EMA 必须每 optimizer step、每个 domain 恰好更新一次。
4. B/C shared Adam β1 必须为 0；A/D shared Adam β1 必须分别为规定值；所有 private β1 必须为 0.9。
5. C 与 B 除 γ vector 外必须计算路径一致。
6. D 与 A 除 shared β1 和必要的 shared/private param-group 拆分外必须一致；默认 β1=0.9 时，拆分组需通过更新等价测试。
7. 所有 loss 日志必须使用同一归一化 sample-weighted定义。
8. 额外评估必须保存/恢复 RNG 状态，不能改变后续 batch 顺序。
9. 禁止 test-based γ/β1/epoch selection。
10. 运行器必须 resume-safe；失败行保留，成功唯一键不可静默覆盖。
11. A/B 必须 fresh run；历史数据只作外部复现对照，不进入正式 paired gate。
12. 正式结果不得把 rotation×seed 的 12 个 block 当作 12 个独立随机 seeds。seed 是主要重复单位，rotation 是每 seed 内的固定交叉条件。
13. 不实现 bias-corrected comp1、新 controller、gamma schedule、stacked momentum 或额外复杂 optimizer。

---

## 8. Ablations required

### Primary comparisons

1. B vs A：`+0.0054` residual 能否 fresh reproduce。
2. B vs C1：domain-conditioned γ 是否优于样本占比加权 memory match。
3. B vs D1：是否优于相同标量 memory 的普通 Adam β1。
4. B vs C_val：是否优于经过公平 validation-only 小网格选择的 uniform EMA。
5. B vs D_val：是否优于经过公平 validation-only 选择的标准 Adam family。

### Required sensitivity/diagnostic comparisons

- C0/C1/C2：结论是否依赖任意 memory matching 定义。
- A/D1/D2：普通 Adam memory sensitivity。
- selected checkpoint vs fixed-final vs共同 epochs：checkpoint/convergence 解释。
- Large/Medium/Small role gain：检查 redistribution。
- task 0/task 1：检查收益是否只来自单一 task。
- 三个 physical-domain rotations：检查固定物理域解释。
- A/B 历史结果 vs fresh A/B：只做描述性复现审计，不混合作正式效应。

不得加入 Curriculum 或新 momentum 算法消融。

---

## 9. Expected logging requirements

### 每 run 的 manifest

- stage、arm、seed、rotation；
- dataset/split/cache/code fingerprints；
- physical domain→L/M/S 映射；
- domain counts/shares；
- 全部 optimizer 参数；
- γ vector、β1、β2；
- $\tau,h_{1/2},N_{\mathrm{eff}}$；
- initialization/batch-order hashes；
- examples processed、optimizer/scheduler/EMA update counts；
- selected epoch、shadow-stop epoch、fixed-final epoch；
- wall time、peak memory、status、异常信息。

### 每 epoch × domain × task

- sample-weighted train loss；
- validation AUC、LogLoss；
- test AUC、LogLoss；
- oracle AUC、LogLoss 与两个 transfer delta；
- negative-transfer indicators；
- physical domain、size role、task id、domain size。

### 每 epoch 的优化诊断

低成本记录：

- 每域 raw shared gradient norm；
- 每域 EMA buffer norm；
- 聚合 shared gradient norm；
- 实际 shared update norm；
- Adam first/second moment norms；
- `||m_d|| / (||g_d||+ε)`；
- EMA update count；
- 启动系数 $1-\gamma_d^k$；
- learning rate。

不得因日志开启改变数值结果；需要 ON/OFF 等价测试。

### 汇总输出

- overall/macro AUC；
- macro LogLoss；
- 合理定义的 sample-weighted/pooled AUC，且不得与 macro 混称；
- per-domain/task AUC 和 LogLoss；
- worst-domain、worst-cell；
- B 相对 A/C/D 的 paired gains；
- Large/Medium/Small role gains；
- 每 seed 聚合 paired difference；
- 每 rotation paired difference；
- raw/practical/concordant NT rates；
- selected/fixed/common-epoch trajectory；
- mean、median、sample SD、range、3/4 seed方向计数；
- 四-seed paired interval只作不确定性描述，不用伪精确 p 值宣称机制成立。

---

## 10. Acceptance criteria for reviewer

### 10.1 实现验收

Reviewer 必须确认：

1. 84-run 矩阵完整且无重复/缺格；
2. A/B 为 fresh runs；
3. B/C 仅 γ vector 不同；
4. A/D 仅 shared β1 和必要 param grouping 不同；
5. 默认 split-group Adam 在 β1=0.9 时与 baseline 更新等价；
6. sample objective 在所有 arm 路径数学一致；
7. examples、steps、batches、初始化完全配对；
8. fixed-budget training 未被 early stop 截断；
9. checkpoint 与 control-family 选择不接触 test；
10. momentum buffers、Adam states 和 raw metrics 均实际落盘；
11. oracle fingerprints 合法；
12. instrumentation OFF/ON 在短程测试中参数和指标一致；
13. Curriculum 与任何新 momentum 变体未进入代码或矩阵。

任一项失败均为 blocking issue，不得 RUN。

### 10.2 科学 KEEP gate

只有以下全部成立，Main Agent 才可暂时 `KEEP comp1 as a secondary component`：

1. **Fresh residual reproduction**  
   B−A 的四-seed聚合 macro-AUC 均值至少 `+0.003`，至少 3/4 seed 为正，且三个 role rotations 的均值方向均为正。

2. **胜过 uniform family**  
   B−C_val 的 macro-AUC 均值至少 `+0.002`，至少 3/4 seed 为正；同时 B−C1 方向为正。

3. **胜过 Adam family**  
   B−D_val 的 macro-AUC 均值至少 `+0.002`，至少 3/4 seed 为正；同时 B−D1 方向为正。

4. **不是单 checkpoint 偶然性**  
   B 对 C_val/D_val 在 `{9,19,29,39,49}` 中至少 3/5 个共同 epochs 为正，fixed-final 不得低于任一 family best 超过 `0.001`。

5. **无明显 role 代价**  
   相对 A，每个 L/M/S role 的平均 AUC gain 均不得低于 `-0.001`；任何 role 不得在至少 2/4 seeds 中出现 `≤-0.003` 的聚合损失。

6. **LogLoss 不矛盾**  
   overall macro LogLoss 不得恶化超过 `0.005`，任一 role 平均 LogLoss 不得恶化超过 `0.01`；若 AUC 与 LogLoss 稳定相反，只能判为 trade-off，不能 KEEP 为无代价组件。

即使通过，也只能说明当前 MovieLens 强不平衡协议下存在独立研究价值，不能声称普适 Momentum 机制或 M3OE 特有现象。

---

## 11. Conditions that would falsify the hypothesis / failure and stop conditions

以下任一成立即否定“domain-conditioned momentum 具有独立价值”或使其证据不足：

1. B−A 小于 `+0.003`、少于 3/4 seeds 同方向，或三种 rotation 中有稳定反例；
2. B 与 C_val 的差落入实用等价区间 `[-0.001,+0.001]`，或 C_val 不劣于 B；
3. B 与 D_val 的差落入 `[-0.001,+0.001]`，或 D_val 不劣于 B；
4. B 只击败 C1，却不能击败 C0/C2 中 validation 选出的简单 control；
5. B 只在 selected checkpoint 占优，在共同 epoch/fixed-final 不支持；
6. overall gain 伴随任一 role 的预注册明显损害；
7. 增益仅来自一个 task、一个 physical domain 或一个 seed；
8. AUC 改善伴随稳定 LogLoss 恶化；
9. fresh A/B 无法复现历史 `+0.0054`，显示旧 residual 可能是窗口或方差现象；
10. 数据、batch、初始化、steps、checkpoint 或 oracle pairing 不一致；
11. test 被用于调 γ、β1 或选择 epoch；
12. instrumentation 改变训练轨迹。

科学 STOP 结论应按证据写为以下之一：

- `STOP Momentum — residual did not reproduce`；
- `STOP Momentum — uniform smoothing is sufficient`；
- `STOP Momentum — ordinary Adam memory is sufficient`；
- `STOP Momentum — gain is checkpoint/role redistribution`；
- `INCONCLUSIVE due to protocol failure`，仅在完整性问题时使用，不得借此继续调 gamma。

Stage 0 完成后不得继续围绕 gamma、EMA、controller 或 momentum schedule 调参。是否进入 Stage 1 必须由 Main Agent 根据 Stage 0 的正式结果单独决定；**本次没有设计任何后续 stage。**
