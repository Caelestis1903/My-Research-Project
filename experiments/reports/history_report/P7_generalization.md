# P7 泛化与 D/T 扩展报告

> 编排：`experiments/p7_orchestrator.py --concurrency 2`（30 s 错峰、断点可续、按 `experiments/results/p7/P7_backbones.csv` 已有行去重）。
> Runner：`experiments/p7_run.py`；分析：`experiments/p7_report.py`（幂等；stdout 已落 `experiments/results/p7/p7_report_stdout.txt`）。
> 报告完成时间：2026-09-08；任务分配：D/T 5-domain 由 P7 编排器生成；组件 1 移植与 backbone 复现同批跑完；STAR 因复现差距过大排除出组件 1 移植结论（详见 §"方法失效或不适用的场景"）。sharebottom 原跑批窗口内因 `ShareBottom_Layer` 切片 bug 全 FAILED，于本次补跑（10/10 OK，none + per_domain_only 各 5 seeds），与 mmoe/ple/star 三 backbone 不在同一跑批窗口；§A/§B 中 sharebottom 的复现与配对结论仅在 sharebottom 自己的 none vs per_domain_only 窗口内自洽，与其余 backbone 的横向比较仅作复现差距量级参考。
> 执行人：Runner（实验调度与记录官）；组件实现层与 P0–P5 同人。

---

## 0. backbone 选型说明

候选来自 `p7.md` 第 4 行：`PLE、MMoE、AdaTT、STAR`。本项目最终选定 `mmoe / ple / star / sharebottom` 四个 backbone，理由按 `experiments/p7_models.py` 头注释的来源事实复述：

- **MMoE / PLE**：层数学逐字取自 FuxiCTR v2.3.9 的 `model_zoo/multitask/{MMoE,PLE}`（Apache-2.0），仅把 FuxiCTR 自带的 `BaseModel` 替换为 MDMTRec 同构的 embedding/tower 包装以消费同输入张量；层前向不变。FuxiCTR 包源码已存档于 `experiments/tmp_pip/FuxiCTR-2.3.9/`（`p5_orchestrator.py` / 此前 p0 期间 `p0.md` 已多次使用同一份 codeload 下载）。
- **ShareBottom**：FuxiCTR v2.3.9 原版（`model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`）是**单个** `MLP_Block` 共享底部（默认 [64,64,64]）+ `num_tasks` 个 tower，**不存在 num_experts 参数，也不存在对 expert 输出取平均**。本仓库 P7 移植因 `Backbone` 包装器将 `num_tasks = D*T` 传入层（原 bug：对 expert 维按 task 索引切片导致 IndexError），将层改写为 `num_experts` 个并行 `MLPBlock` + 沿 expert 维 mean 聚合成单一共享底部表示 + `[shared] * num_tasks` 广播。该实现**是仓库自引入的适配**，**不是 FuxiCTR ShareBottom 的逐字移植**；底部参数量约为原单 MLP 的 `num_experts` 倍。语义（单共享底部喂所有任务头）保持，层数学不保持。详见 §D.2。
- **STAR**：自实现，遵循 L. Sheng et al., "One Model to Serve All: Star Topology Adaptive Recommender", RecSys 2021 的 per-domain `W_d` 逐元素乘共享 `W_0` 再加 per-domain bias 结构。结构上与 MDMTRec 自带的单层 star fusion（`m3oe/MDMTRec/models/multi_domain.py:151-153`）同源——后者是同一拓扑的单层实例；本实现为多层 FCN。
- **AdaTT**：候选列表中给出但**未入选**。原因：`p7_models.py` 头注释未收录 AdaTT 实现，且本仓库当前会话窗口未发现可靠的 Apache-2.0 / MIT 开源 PyTorch 实现可直接借用（如需新增 backbone，须先完成实现+冒烟+1 配置复现才能叠加我们的组件）；为不超出 P7 编排器已固化的 60 任务清单，本次不纳入；如后续需要，按 p7.md 约束（先复现一篇原报告结果再叠加组件）单独追加。
- **复现验证门控（p7.md 约束，定性）**：先在每个 backbone 上用 momentum=none 跑完整 50 epoch × 5 seeds，对照 M3oE Table 2 的 MDMT 行（M3oE-MDMT 76.35 / PLE-MDMT 75.68 / STAR 76.08 / ShBot-MDMT 76.26），复现差距过大者按 p7.md "如实在报告中写、无法复现则排除"的定性规则排除出组件 1 移植结论。p7.md 仅规定定性门控、未规定任何数值阈值；§A 的"±1 点内"等数值化判定为 Runner 对已得数据的事后归纳。
- **复现严格性差异提示**：mmoe / ple 为 FuxiCTR 逐字移植（参数对齐、形状对齐），sharebottom 为仓库自引入的语义适配（4-expert + mean 聚合，参数对齐、形状适配），star 为自实现。**sharebottom 的"复现成立"判定严格性弱于 mmoe / ple**，仅作为"语义一致的复现参照"采信，不与 mmoe / ple 等量齐观。

---

## §A 跨 backbone 复现对照表（D=3, momentum=none, 5 seeds）

`test_auc_mean` 列是 `evaluate_multi_domain`（`m3oe/MDMTRec/trainer/ctr_trainer.py:228` 等）对全测试集（两任务合并）的 **pooled AUC**（与 P5 `P5_raw.csv` `test_auc_mean` 同口径），并非 6 个 per-(d,t) 格的均值（逐行核验 60/60 行二者均不等）。单位为原始比例（×1），与 P5/P0 同口径。

| backbone | n | mean | std | M3oE Table 2 | diff (ours−paper)*100 | P5_mdmtoe_ref (ml_orig) | δ vs P5_mdmtoe*100 | 复现判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mmoe | 5 | 0.767107 | 0.001336 | 76.35 | +0.360740 | 0.770098 ± 0.001776 | -0.299051 | **成立**（与论文差 +0.36 AUC 点，方向为我们高于论文；5 seeds 稳定，std 0.00134） |
| ple | 5 | 0.764977 | 0.001610 | 75.68 | +0.817680 | 0.770098 ± 0.001776 | -0.512111 | **成立**（与论文差 +0.82 AUC 点，方向为我们高于论文；5 seeds 稳定，std 0.00161） |
| star | 5 | 0.533898 | 0.006611 | 76.08 | -22.690200 | 0.770098 ± 0.001776 | -23.619991 | **不成立**（远低于论文 76.08，差 22.69 点；详见 §"方法失效或不适用的场景"） |
| sharebottom | 5 | 0.769253 | 0.001345 | 76.26 | +0.665260 | 0.770098 ± 0.001776 | -0.084531 | **成立**（与论文差 +0.67 AUC 点，方向为我们高于论文；补跑窗口，5 seeds × 2 momentum 全 OK；详见 §"方法失效或不适用的场景" 与 §0 复现严格性提示） |
| mdmtoe (P5 in-repo) | 5 | 0.770098 | 0.001776 | — | — | — | — | 作为基准参照（非本表对照对象） |

> 注：
> 1. `M3oE Table 2` 列按 `p7.md` / `exp.md` 使用的"原始比例×100"口径展示（76.35 即 0.7635）；论文中报告为 AUC*100。
> 2. `diff` 列符号约定：`diff (ours−paper)*100 = (mean − paper_val) × 100`，正值表示我们高于论文值。mmoe / ple / sharebottom 三者 `diff` 均为正（+0.36 / +0.82 / +0.67），且 95% CI 下界（按 t 分布 n=5 算：mean ± 2.776 × std/√5；mmoe [76.45,76.88] / ple [76.30,76.70] / sharebottom [76.76,77.09]）**均高于**论文值 76.35 / 75.68 / 76.26——即我们**显著高于**论文参照值。这与"覆盖论文值"方向相反但语义更强：差距为正且统计显著，判定不受影响。
> 3. 论文 AUC 口径（MDMT 设定下 per-(d,t) 10 格平均）与本表 pooled AUC 口径不完全一致（MDMT 设定下两任务合并的全局 ROC-AUC），二者仅作量级参照；正负号与小差距数值不应直接当作同口径绝对比较。`δ vs P5_mdmtoe` 列为我们的 backbone 相对仓库自带 MDMTRec（P5 ml_orig baseline）同口径的差距。
> 4. `p7.md` 仅规定**定性**复现门控（"先复现原报告结果，无法复现则报告差距并排除"），**没有规定任何数值阈值**；下文的"±1 点内"等表述是 Runner 对已得数据的事后归纳。

**判定逻辑（事后陈述，非 p7.md 规定）**：基于"与论文参照值同量级（±1 点内，方向不限）且 5 seeds 稳定（std ≲ 0.002）"两条事后归纳做分类——
- **成立**（mmoe / ple / sharebottom）：与论文参照值同量级，5 seeds 全部 OK，std 0.001–0.002 之间，无单点跳变。
- **不成立**（star）：与论文参照值差 22.69 点，远大于 mmoe/ple/sharebottom 的 0.36–0.82 点（差近 30 倍），且冒烟记录首 epoch 即 ~0.53 稳定停留该水平 → 非欠训练/欠拟合。按 p7.md "如实在报告中写、无法复现则排除"的定性规则排除出组件 1 移植结论。

sharebottom 的"成立"判定需附加 §0 复现严格性提示：sharebottom 的 backbone 实现是仓库自引入的 4-expert + mean 聚合适配（详见 §D.2），并非 FuxiCTR ShareBottom 的逐字移植，**复现严格性弱于 mmoe / ple**；本行作为"语义一致的复现参照"采信，不与 mmoe/ple 等量齐观。

---

## §B 组件 1 移植对照表（D=3, paired-t: per_domain_only vs none, 5 seeds）

| backbone | n | diff (mean_A − mean_B) | t | p (双侧) | Cohen's d | 结论 |
|---|---:|---:|---:|---:|---:|---|
| mmoe | 5 | +0.002978 | 5.182900 | 0.006593 | 2.317863 | **显著正效应** |
| ple | 5 | +0.005538 | 5.319129 | 0.006009 | 2.378787 | **显著正效应** |
| star | 5 | -0.006649 | -3.268831 | 0.030823 | -1.461866 | **不纳入结论**：STAR backbone 在 §A 中判定为"复现不成立"（mean ≈ 0.534，远低于论文 76.08），按 p7.md 与本报告 §0 复现门控规则，本 backbone 上的组件 1 配对差异不具有方法学意义。 |
| sharebottom | 5 | +0.004012 | 4.616984 | 0.009904 | 2.064778 | **显著正效应**（补跑窗口；p=0.0099，d=2.06） |

> 配对方式：按 seed 配对，`a = test_auc_mean(per_domain_only, seed)`，`b = test_auc_mean(none, seed)`；`diff = a − b`；`stats.ttest_rel(a, b)`（双侧）；`d = diff.mean() / diff.std(ddof=1)`。所有数值来自 `p7_report.py` stdout（`experiments/results/p7/p7_report_stdout.txt` §B）。
>
> **结论**：在 §A 判定为"成立"的 backbone 上（mmoe、ple、sharebottom），组件 1 移植均给出**显著正效应**（p < 0.01，d > 2），与 P5 强不均衡档观察一致——这三个 backbone 都属于"参数共享在所有 (D, T) 槽位"的多任务结构（mmoe 共享 expert；ple 共享 expert；sharebottom 共享 bottom），per-domain 动量改造直接作用于这些共享参数，与 MDMTRec 的实现路径同构。注：sharebottom 的 none vs per_domain_only 配对比较在 sharebottom 自己的补跑窗口内自洽，与 mmoe/ple 的配对不构成跨窗口横向比较（仅在 §A 复现差距量级上可比）。

---

## §C D/T 扩展结果（D=5, T=2, MovieLens 按年龄切 5 域）

**5×2 分域 AUC 矩阵**（test AUC mean across 5 seeds；原始比例，6 位小数）：

| backbone | momentum | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | d3t0 | d3t1 | d4t0 | d4t1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mdmtoe | none | 0.720192 | 0.748497 | 0.763644 | 0.776280 | 0.770305 | 0.789804 | 0.751293 | 0.759944 | 0.720283 | 0.747290 |
| mdmtoe | per_domain_only | 0.732749 | 0.761317 | 0.766061 | 0.780540 | 0.773682 | 0.791682 | 0.756008 | 0.763288 | 0.732530 | 0.759937 |
| mmoe | none | 0.737250 | 0.767892 | 0.765404 | 0.772196 | 0.767168 | 0.788448 | 0.748798 | 0.762159 | 0.719163 | 0.750069 |
| mmoe | per_domain_only | 0.726395 | 0.754784 | 0.764464 | 0.775009 | 0.770939 | 0.790794 | 0.748848 | 0.759254 | 0.728803 | 0.754625 |

> 来源：`P7_backbones.csv` 中 D=5 行；`p7_report.py` 的 `load_p7` 与 `per_domain_matrix` 本次重构为按行内 `auc_d*` 列**动态识别 (D, T)**，D=5 行的 4 个无表头列（d3t0..d4t1）已正确归位（不再有 NaN 警告）；矩阵与均值口径均直接来自 `p7_report.py` stdout。`experiments/_p7_dt_extract.py` 与 `experiments/results/p7/P7_d5_per_dt_matrix.csv` 保留作留底。
> 矩阵性质：D=5、T=2 下全部 10 个 (d,t) 格均产出有效 AUC，无 NaN、无接近 0.5 的退化格——证明 P7 中 `GeneralizedTrainer`（p7_run.py:94-132）解除 C3 硬编码后，对 D=5 同样给出正常训练与评估。

**10 项 per_dt 均值与配对检验**（5 seeds, ddof=1）：

| backbone | momentum | n | mean | std |
|---|---|---:|---:|---:|
| mdmtoe | none | 5 | 0.763934 | 0.001304 |
| mdmtoe | per_domain_only | 5 | 0.768839 | 0.001633 |
| mmoe | none | 5 | 0.764111 | 0.001841 |
| mmoe | per_domain_only | 5 | 0.765196 | 0.001249 |

配对 t 检验（per_domain_only vs none, 5 seeds）：

| backbone | n | diff | t | p | Cohen's d |
|---|---:|---:|---:|---:|---:|
| mdmtoe | 5 | +0.004904 | 7.106125 | 0.002072 | 3.177956 |
| mmoe | 5 | +0.001086 | 3.138101 | 0.034912 | 1.403401 |

> 数值与矩阵均来自本次重构后的 `p7_report.py` stdout（`experiments/results/p7/p7_report_stdout.txt` §C），`P7_summary.csv` 同步刷新。
>
> **结论（口径收敛）**：
> - mdmtoe backbone 在 D=5 下组件 1 显著正效应（p=0.0021，d=3.18），与 P5 强不均衡档方向一致；
> - mmoe backbone 在 D=5 下组件 1 名义显著正效应（p=0.035，d=1.40），d<2，按 Reviewer 此前对组件 1 的口径归类为"中效应、待复核"；
> - **D=3 → D=5 的掩码路径泛化有效**（实证覆盖 (D=3, T=2) 与 (D=5, T=2) 两点），证明组件 1 公式对 D 变化不敏感（仅依赖 `p_d`），P7 的 `GeneralizedTrainer` 对 D=5 解码路径正常。
> - **口径声明**：实证仅覆盖 T=2（两层任务）；T≠2（如 T=1 / T=3 / T≥4）**未实证**。§C 标题级"方法在 D=5 下成立"成立，但**不应外推至"任意 (D, T) 下成立"**；外推需后续实验。

---

## §D 方法失效或不适用的场景

1. **STAR backbone 复现失败 — 已排除出组件 1 移植结论**。
   - 5 seeds × 2 momentum 共 10/10 OK（程序完整跑完 50 epoch 早停，未崩溃），但 mean test AUC 仅 0.533898 ± 0.006611；论文 STAR 报告值 76.08 → 差距 -22.69 AUC 点（绝对值），与 §A 成立三 backbone（差 0.36–0.82 点）差近 30 倍。
   - 已知冒烟记录（`experiments/results/p7_smoke/P7_backbones.csv`）首 epoch 即 ~0.53，完整 50 epoch 仍停留该水平 → 不是欠拟合或欠训练。
   - 诚实结论：STAR 自实现（在 M3oE star 单层基础上扩展为多层 FCN）**无法复现 RecSys 2021 论文在 MovieLens MDMT 设定下的报告水平**。按 p7.md "若某 backbone 无法复现其原报告结果，报告差距并将其排除，不要在未验证的实现上做结论" 的定性规则，STAR 不进入 §B 的方法学结论；其 CSV 数据保留用于将来实现修正后的复算。

2. **sharebottom backbone 切片 bug — 已修复并补跑完成；复现严格性弱于 mmoe/ple**。
   - 原 5 seeds × 2 momentum 共 10/10 FAILED（见 `experiments/logs/p7_orchestrator.out` 中 `[done] p7 sharebottom ... FAILED last='IndexError: index 4 is out of bounds for dimension 1 with size 4'`），wallclock 10s 即崩溃。
   - 根因：`p7_models.py` 原 `ShareBottom_Layer.forward` 返回 `[expert_output[:, i, :] for i in range(self.num_tasks)]`，其中 `expert_output` 在 dim 1 上大小为 `num_experts=4`，但 `num_tasks = D*T = 6`（Backbone 构造处 `n_task = domain_num * task_num`）。FuxiCTR 原版 ShareBottom 的设计意图是"单个共享 MLP 底部 + 每 task 一个 tower"，**不是**"per task 切 expert"。
   - FuxiCTR v2.3.9 原版事实（对照 `experiments/tmp_pip/FuxiCTR-2.3.9/model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`）：单个 `MLP_Block`（默认 [64,64,64]）作共享底部，输出 `num_tasks` 个 tower；**无 num_experts 参数、无 expert 维、无任何对 expert 输出取平均的操作**。故"原版即对所有 expert 输出做平均/共享到各 task"系错误陈述（已被 Reviewer 核验）。
   - 修复方式（Coder，已落地 `experiments/p7_models.py:109-125` 及头注释）：保留 `num_experts` 个并行 expert `MLPBlock`（`torch.stack` on dim=1），把 `dim=1` 视为 expert 维，对 expert 维取 mean 聚合成单一共享底部表示 `shared`（shape `(batch, tower_input_dim)`），再 `[shared] * num_tasks` 广播到所有 tower 输入。**该修复是仓库自引入的适配**（为消化 `Backbone` 包装器传 `num_tasks = D*T` 而保留 `num_experts` 形参），**不是 FuxiCTR ShareBottom 的逐字移植**；底部参数量约为 FuxiCTR 单 MLP 的 4 倍。语义（单共享底部喂所有任务头）与 FuxiCTR 一致；层数学不一致。
   - 冒烟与补跑：1-epoch 冒烟（validation auc 0.7365）通过；本次编排器以 `--concurrency 2`、30s 错峰重跑 10 任务，10/10 OK（wall 300–445s/任务）；§A/§B 数值与配对均已落表。
   - 复现严格性提示：sharebottom 的"成立"判定基于**仓库自引入的 4-expert + mean 聚合**实现（与 FuxiCTR 单 MLP 共享底部参数量差 4 倍），与 mmoe / ple 的 FuxiCTR 逐字移植不可同口径比较；本行作为"语义一致的复现参照"采信，**不与 mmoe / ple 等量齐观**。`p7_models.py` 头注释已由 Coder 修正（移除了对 ShareBottom 的"layer math is unchanged"声明，明确标注为"repo-local adaptation"）。
   - 诚实标注：补跑窗口与 mmoe/ple/star 的原跑批窗口**不重合**（不同日期、不同随机种子次序、不同 GPU 温度），故 sharebottom 在 §A 中的复现判定仅基于自己窗口内的 5 seeds 协议；§B 中 sharebottom 的 none vs per_domain_only 配对在补跑窗口内自洽（同一窗口、同一 seed 序列），与 mmoe/ple 的配对不构成跨窗口横向比较。结论按 5 seeds × 2 momentum 全量协议给出（与 p7.md 与 exp.md 的复现门控一致）。

3. **组件 2（域条件课程）对 mmoe / ple / star / sharebottom 全部不适用**。
   - 组件 2（p4_run.py 实现、`MDMTRecBeta`）需要一个可调的跨域融合权重标量（per-domain `α_d` / per-task `β_t` / 课程 sigmoid）。mmoe/ple 的 gate 输出 softmax over experts → 没有"标量融合"自由度；star 的 domain weight 是逐元素乘矩阵（向量化的 per-channel 权重）→ 不能坍缩为单一标量；sharebottom 无任何跨域融合机制。
   - 因此本报告 §B 只能验证组件 1（per-domain 动量）的可移植性；组件 2 在这些 backbone 上的"不存在性"已在 p7.md 与 exp.md §"剩余未完成的实验与任务" C.10 明确，不强行构造。

4. **组件 2 在 D=5 下的进一步审视**。
   - §C 中 mdmtoe backbone 在 D=5 下走的就是组件 1 路径；组件 2 课程功能未在 P7 中重新测（仍是 MDMTRec 的 wrapper）。这与 exp.md §"剩余" C.10 中"组件 2 在这些 backbone 上不适用"一致；P6 消融（ml_r30）才是组件 2 跨档敏感性扫描的主战场。

5. **GPU 显存异常**。
   - P7 跑批过程中观察到单 run 阶段峰值显存一度达到 7.17 GB（`nvidia-smi` 抽样：2026-09-08 15:48），逼近 8 GB 笔记本卡上限；并发 2 仍跑完（未见 OOM 报错），但建议后续若并发提升需先在空闲状态下做 1 run 冒烟验证。

---

## §E 交付清单

| 路径 | 内容 |
|---|---|
| `experiments/results/p7/P7_backbones.csv` | 60 行 OK 数据（12 个 (backbone, domain_version, momentum) cell × 5 seeds），含 6+10 per-(d,t) AUC 原始列 |
| `experiments/results/p7/p7_report_stdout.txt` | `p7_report.py` 完整 stdout（§A/§B/§C 三节；本次按动态 (D,T) 识别重构后已消除 D=5 NaN 警告） |
| `experiments/results/P7_summary.csv` | `p7_report.py` 生成的长表（按 cell × section，本次同步刷新） |
| `experiments/results/p7/P7_d5_per_dt_matrix.csv` | 5×2 分域 AUC 矩阵（_p7_dt_extract.py 留底；§C 表改由 p7_report.py stdout 直接生成） |
| `experiments/p7_models.py` | backbone 库；头注释已明确 ShareBottom 为"repo-local adaptation"，非 FuxiCTR 逐字移植 |
| `experiments/p7_report.py` | 报告生成；`load_p7` 与 `per_domain_matrix` 已重构为按行内 auc_d* 列动态识别 (D, T) |
| `experiments/logs/p7_*.log` | 单 run 日志（60 OK；sharebottom 10 个为补跑产物） |
| `experiments/logs/p7_orchestrator.out` | 编排器主日志（含 `P7_ORCHESTRATOR_DONE`） |
| `experiments/reports/P7_generalization.md` | 本报告 |

---

## Reviewer 审计与统计仲裁结论

> 审计人：Reviewer（科研审计与统计仲裁员）；审计日期：2026-09-08；方式：只读审计 + scipy 独立复算。未修改任何实验代码与原始 CSV。

**1. 真实性核验（PASS）**。抽 7 个 (backbone, momentum, domain_version, seed)：`sharebottom/none/three/2022`、`sharebottom/per_domain_only/three/2026`、`mmoe/none/three/2024`、`ple/per_domain_only/three/2023`、`star/none/three/2025`、`mdmtoe/none/five/2024`（D=5）、`mmoe/per_domain_only/five/2024`（D=5），日志尾部 `[p7]` 行的 test_auc_mean 及全部 auc_d* 列与 `P7_backbones.csv` 逐位一致（含 D=5 行的 d3/d4 无表头列）。台账 60 数据行、60 个唯一 (backbone,domain_version,momentum,seed) 键、12 cell × 5 seeds 无缺失无重复；`logs/p7_*.log` 60 个 run 日志齐全（另有 1 个冒烟日志）。

**2. 统计复算（PASS）**。独立 scipy 复算 6 个配对检验，与 `P7_summary.csv` / `p7_report_stdout.txt` 全部一致（|Δ| ≤ 1e-6，远小于容差 1e-4）：

| backbone | 窗口 | diff | t | p | d |
|---|---|---:|---:|---:|---:|
| mmoe | D=3 | +0.002978 | 5.182900 | 0.006593 | 2.317863 |
| ple | D=3 | +0.005538 | 5.319129 | 0.006009 | 2.378787 |
| sharebottom | D=3 | +0.004012 | 4.616984 | 0.009904 | 2.064778 |
| star | D=3 | -0.006649 | -3.268831 | 0.030823 | -1.461866 |
| mdmtoe | D=5 | +0.004904 | 7.106125 | 0.002072 | 3.177956 |
| mmoe | D=5 | +0.001086 | 3.138101 | 0.034912 | 1.403401 |

§A 的 mean/std/diff 复算亦全部一致（mmoe +0.360740 / ple +0.817680 / star -22.690200 / sharebottom +0.665260，单位 1e-2 AUC）。

**3. ShareBottom 修复的语义核验（裁定：语义声明部分不成立，需修正表述；复现结论可采信）**。对照 `experiments/tmp_pip/FuxiCTR-2.3.9/model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`：原版为**单个** `MLP_Block` 共享 bottom（默认 [64,64,64]）+ `num_tasks` 个 tower，**不存在 num_experts，也不存在"对 expert 输出取平均"**。故：① 修复后"单一共享底部表示广播到全部 tower"在架构语义上与原版一致，此部分声明成立；② 报告 §D.2 括号内"其原版即对所有 expert 输出做平均/共享到各 task"**与原版源码不符，系错误陈述**；③ "4 个并行 expert 取 mean 聚合"是本仓库修复自引入的机制，原版无此结构，`p7_models.py` 头注释对 ShareBottom 的 "layer math is unchanged / verbatim" 表述自修复起不再成立。影响评估：修复后的底部参数量约为单 MLP 的 4 倍，并非 FuxiCTR ShareBottom 的逐字移植，"复现验证"的严格性弱于 mmoe/ple；但核心语义（共享底部喂所有任务头）保持，且 §A 复现差距 +0.665 点与 mmoe(+0.36)/ple(+0.82) 同量级、与 star(-22.69) 差两个数量级，复现门控实证上仍通过。裁定：sharebottom 复现"成立"的判定可采信，但 §D.2 与 §0 的语义表述必须修正。

**4. §A 复现判定仲裁（裁定：判定本身成立；三处表述需修正）**。
- 差距计算数值正确，但表头 `diff (paper−ours)*100` 与 `p7_report.py` 实际计算（ours−paper，正值=我们高于论文）**符号标注相反**，需修正。
- "95% CI 覆盖论文值"对 mmoe/ple/sharebottom **均不成立**：复算 95% CI 分别为 [76.545, 76.877] / [76.298, 76.698] / [76.758, 77.092]，论文值 76.35/75.68/76.26 全部低于 CI 下界（我们显著高于论文）。该句需删除或改写；事实上"显著高于论文"比"覆盖"更强，不影响"成立"结论。
- 判定阈值问题：`p7.md` 仅规定定性门控（"先复现原报告结果，无法复现则报告差距并排除"），**没有任何数值阈值**；§A 的"≤0.76 点"、§D.1 的"p7.md 排除阈值 0.70"、"≥20 点"均为 Runner 自定且误归于 p7.md；且 ple 差 +0.82 已超出其自定 0.76 阈值仍判成立，阈值内部不一致。尽管如此，三个"成立"与一个"不成立并排除"的判定在 p7.md 定性规则下稳健（差距 0.36–0.82 点 vs 22.69 点），裁定维持。
- 另：§A 引言"测试集 6 项平均 test AUC"表述不准确——`test_auc_mean` 是 `evaluate_multi_domain`（ctr_trainer.py:228）对全测试集（两任务合并）的 pooled AUC，并非 6 个 per-(d,t) 格的均值（逐行核验 60/60 行二者均不等）。因与 P5 同口径，横向比较不受影响；措辞需修正。

**5. §B 跨窗口可比性仲裁（裁定：诚实标注充分，sharebottom 配对结论可采信）**。窗口不重合在头部第 5 行、§B 注、§D.2 三处披露，限定语（"窗口内自洽"、"横向比较仅作复现差距量级参考"）与数据事实一致（`p7_orchestrator.out` 确认原窗口 sharebottom 10/10 FAILED `IndexError`，补跑 10/10 OK）。按 seed 配对的 t 检验只要求同一 backbone 两 momentum 臂同窗口同 seed 序列，sharebottom 满足，裁定其 §B 结论（p=0.0099，d=2.06，显著正效应）可采信；跨窗口横向比较报告已自我限定，裁定充分。

**6. §C D/T 扩展仲裁（裁定：数据链成立；"任意 (D,T)"属过度宣称，需收敛措辞）**。`P7_d5_per_dt_matrix.csv` 与 `P7_backbones.csv` 原始 auc_d* 列逐格一致（max|diff| = 0.0），`_p7_dt_extract.py` 重构精确；`p7_report.py` 的 D=5 矩阵读取确仍写死 D=3 列名（stdout 有 d3/d4 NaN 警告），报告已如实披露，且 §C 均值与配对取自 test_auc_mean 列、不受该缺陷影响。但实证仅覆盖 (D=3,T=2) 与 (D=5,T=2) 两个点，且 mmoe 在 D=5 下仅名义显著（p=0.0349，d=1.40）：证据支持"组件 1 在 D=3→D=5 路径有效"，**不足以支持"方法在任意 (D,T) 下成立"**；T 未做任何变化（恒为 2）。§C 结论句中"本节实证 D=3 → D=5 路径有效"是恰当口径，标题级"方法在 D=5 下成立"可保留，但任何向"任意 (D,T)"的外推需删改。

**总裁定**：数据真实、统计正确、各节判定方向维持（§A mmoe/ple/sharebottom 成立、star 不成立并排除；§B 三 backbone 显著正效应；§C D=5 路径有效）。需 Runner 修正的表述（不影响数值与结论）：① §A 表头 diff 符号；② §A "95% CI 覆盖论文值"；③ §A "6 项平均"措辞；④ §A/§D.1 阈值误归于 p7.md；⑤ §D.2 ShareBottom "原版即对 expert 输出做平均"及 §0/`p7_models.py` 头注释的 verbatim 表述；⑥ §C 向"任意 (D,T)"的外推措辞。另建议（非阻塞）：Runner 修正 `p7_report.py` 的 D=5 列读取，使流水线自洽、消除对 `_p7_dt_extract.py` 外部重构的依赖。
