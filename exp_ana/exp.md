# M3oE 实验最终整合文档（exp.md）

> **文档性质声明**：本文档为 P0–P7 全部实验的**整合最终文档**，整合自以下源文件，**读者只读本文档即可获得全部实验记录**。所有数字逐字照搬源文件，6 位小数；原始 CSV / 日志以文件引用方式索引，**不内嵌原始 CSV 全量行**。
> **整合日期**：2026-09-08
> **整合前快照备份**：`experiments/logs/exp_handover_primerge_20260908.md`（287 行原 exp.md 完整备份）
>
> **整合源文件清单**（均位于 `experiments/`，源文件全部保留不动，供溯源）：
> 1. `experiments/EXP_CONTEXT.md`（环境 §1 / 24 条断言核查表 §2 / 各阶段摘要 §7-§8）
> 2. `experiments/reports/P1_diagnosis.md`（P1 训练插桩四项诊断）
> 3. `experiments/reports/P2_baselines.md`（P2 AutoML 贡献与 B10 bug 影响对照）
> 4. `experiments/reports/P3_kuairand.md`（P3 KuaiRand-Pure 数据管线）
> 5. `experiments/reports/P4_implementation.md`（P4 三组件实现与等价性测试）
> 6. `experiments/reports/ML_P5_REPORT.md`（P5 MovieLens 六档梯度扫尾报告，Reviewer 签署）
> 7. `experiments/reports/KUAI_STAGE1_ORACLE.md`（KuaiRand 阶段 1：单域 Oracle 真理基准 + 历史 Log 解剖，Reviewer 签署）
> 8. `experiments/reports/KUAI_P5_MAIN_REPORT.md`（KuaiRand P5 主矩阵实验报告，Reviewer 签署）
> 9. `experiments/reports/FINAL_RESEARCH_REPORT.md`（P5 终审研究报告，Reviewer 终审签署生效 2026-09-08）
> 10. `experiments/reports/P6_ablation.md`（P6 消融实验报告，Reviewer 签署并审计）
> 11. `experiments/reports/P7_generalization.md`（P7 泛化与 D/T 扩展报告，Reviewer 签署 + 审计勘误）
>
> **冲突处理原则**：内容冲突时**以已签署报告 > 旧 exp.md 摘要**为准，并在脚注注明取舍；已签署报告的所有 Reviewer 签署节完整保留并注明来源报告名。
> **章节内交叉引用规则**：所有"见 §X.Y"已按本文档新编号重写；原始报告中的"见 §X.Y"在嵌入时同步更新。
>
> **免责声明**：源文件仍保留于 `experiments/` 供溯源；本文档为唯一权威整合版（"整合后快照"）。任何后续数字更正请同时修订两份。

---

## §1 总览

> 本节保留 Master 2026-09-08 复审修正后的"核心科学结论"措辞与八阶段状态表；后续 §3–§11 给出各阶段证据与原始数据。

- **研究对象**：M3oE（SIGIR 2024，arXiv:2404.18465v3）官方 PyTorch 代码（本地 `m3oe/`，非 git 仓库）。
- **八阶段实验计划**（p0.md–p7.md）。当前状态：

| 阶段 | 状态 |
|---|---|
| P0 环境+复现+代码核查 | ✅ 完成 |
| P1 训练插桩四项诊断 | ✅ 完成 |
| P2 fixed-0.7311/fixed-0.5/修bug 对照 | ✅ 完成 |
| P3 KuaiRand-Pure 数据管线 | ✅ 完成（管线与论文 Table 1 逐位复现） |
| P4 三组件实现+等价性测试 | ✅ 完成 |
| P5 主实验 | ✅ 完成（2026-09-08 闭环：ML 六档主矩阵 144 行 OK + oracle 45 行 OK；KuaiRand 30/30 + oracle 15/15；三份报告均经 Reviewer 签署 + 终审） |
| P6 七组消融 | ✅ 完成（2026-09-08 闭环：85/85 OK，17 变体 × 5 seeds；ml_r30 上 9 组消融全部出报告；Reviewer 签署消融报告） |
| P7 backbone 泛化 + D/T 扩展 | ✅ 完成（2026-09-08 闭环：60/60 OK，12 cell × 5 seeds；MMoE/PLE/ShareBottom 复现成立 + 组件 1 移植 + D=5 扩展，STAR 复现失败排除；Reviewer 签署泛化报告） |

- **核心科学结论**（一句话版，按出现顺序）：
  1. M3oE 复现成功（0.7686±0.0017 vs 论文 0.7702）；
  2. 其"可微 AutoML"实际贡献≈0（fixed-0.7311 无显著差异，p=0.32）；
  3. 优化器分组 bug（B10）不影响均值但把种子方差放大 8 倍；
  4. "小域梯度噪声更大"这一组件设计前提**不成立**（SNR 域间无差异）；
  5. 尽管前提不成立，组件 1（域规模驱动 per-domain 动量）在 MovieLens 强不均衡档产生巨大收益：**+0.0166（30:1，p=0.0002，d=5.8）→ +0.0202（100:1，p=0.006）**；但全六档（补全 3:1/10:1 后）**非单调**：1:1 +0.0064、1.9:1 +0.0034、3:1 −0.0012（名义显著凹陷，n=3 待复核）、10:1 +0.0053（n.s.）；组件 2 单独无效；组件 1&2 组合在强档点估计最优且在 100:1 档收窄域间差距（**但 P6 裁定 comp2 相对 comp1 的边际贡献与种子噪声不可区分，"组合最优"不得解读为 comp2 有独立增益**）；
  6. KuaiRand（24:1 极端不均衡）上：comp1/comp1&2/comp123 在大域（本地 d1=论文 d2）**显著收窄负迁移**（Δ_oracle d1t0 从 baseline −0.070659 收窄至约 −0.0616，配对 p≤0.024（**名义显著，15 个检验未过 Bonferroni**，作方向性结论采信）；d1t1 收窄 +0.015~+0.017 但**不显著**（comp1 p=0.369、comp1&2 p=0.159、comp123 p=0.116，不得与 d1t0 并列为"显著收窄"））且小域 Δ 全为正，但 Δ_oracle(大域) 仍 ≈ −0.06/−0.08——Reviewer 裁定**"前沿移动"而非真 Pareto 外推**；
  7. P6 消融（ml_r30）：comp1 增益几乎全部来自 per-domain 动量本体（d=10.16）；均一γ/手工γ/SNR反推γ均不及公式γ；comp2/comp3 单独及课程方向反转均不显著；κ0 平台区宽 20×（**限定：κ0 控制的是主效应与噪声不可区分的 comp2，"无效应组件的强度旋钮不敏感"近乎恒真，该证据强度远低于 γ0 平台区**）、γ0 平台区右端 0.95（0.99 显著跌落 d=−3.87）；EMA 应替换而非叠加 Adam β1（Reviewer 补算直接配对 p=0.000776 d=4.11；**边界：未排除"总动量预算"替代解释，stacked 臂双重动量等效过平滑，彻底排除需半强度 β1 叠加臂**）。另注口径：P6 数字为 pooled-AUC 口径（baseline r30=0.722514），P5 主表为六项平均口径（baseline r30=0.7268），绝对水平差 ~0.004，方法间差值两口径同向。g2 进一步闭环 P1 的否定性发现：**收益绑定在"按域规模分档"本身，而非梯度噪声匹配**（若噪声假说成立，SNR 反推 γ 应最优，实测它显著最差 d=−5.38）——组件 1 的有效机理是规模分档动量，其设计前提（小域噪声更大）虽不成立，分档结构本身恰好是对的东西；
  8. P7 泛化：组件 1 移植到 MMoE/PLE/ShareBottom 均显著为正（d 2.06–2.38），方法非 M3oE 特有；STAR 复现失败排除；D=3→D=5 掩码路径泛化有效（T≠2 未实证）；组件 2 在无跨域融合标量的 backbone 上不适用。

> **章节指引**：§2 给环境与 24 条代码事实核查；§3–§6 给 P1–P4 阶段报告全文（诊断、对照、管线、组件实现）；§7–§9 给 P5 全部主实验（§7 MovieLens、§8 KuaiRand、§9 终审综合报告）；§10 给 P6 消融全文；§11 给 P7 泛化全文；§12 补 test logloss 留底；§13 给 Master 批判性复审与机理重归因；§14 整合工程坑与风险提示；§15 文件地图；§16 交接清单。

---

## §2 实验环境与复现（合并自 `EXP_CONTEXT.md` §1、§2 与 exp.md §2.1-§2.3）

### §2.1 环境信息（`EXP_CONTEXT.md` §1 / `experiments/requirements.lock`）

| 项 | 值 |
|---|---|
| OS | Windows 10 (build 26200), Git Bash |
| Python | 3.13.5 (Anaconda base, `D:\Programs\anaconda3`) |
| PyTorch | 2.9.0+cu126 |
| CUDA | 12.6 (driver 592.82) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8188 MiB |
| pandas / numpy / scikit-learn / tqdm | 2.2.3 / 2.1.3 / 1.6.1 / 0.4.x（完整清单见 `experiments/requirements.lock`） |
| 虚拟环境 | `D:\CODE\phase0-3\venv`（`python -m venv venv --system-site-packages`） |
| venv 复用 base torch | `--system-site-packages` 复用 Anaconda base 已装 torch 2.9.0+cu126（pip 在本机网络下装 3GB CUDA 轮子过慢；仓库内 `__pycache__/*.cpython-313.pyc` 证明同 Python 3.13 此前曾运行过该仓库） |
| 数据 | 仓库自带 `ml-1m.csv`（1,000,209 条，与官方 zip 字节一致 75,514,352） |

**环境调整及理由（全部记录，无静默升级）**：
1. 仓库**没有** requirements.txt / setup.py（官方 GitHub 树确认），无可对照的官方依赖清单，因此不存在"过时依赖需要调整"的问题；按现状使用。
2. venv 以 `--system-site-packages` 创建：复用 Anaconda base 已装好的 torch 2.9.0+cu126。
3. `run_movielens.py` 的 `--dataset_path` 默认值是作者本机路径 `D:/CODE/data/ml-1m`；不改代码，调用时显式传 `--dataset_path D:/CODE/phase0-3/m3oe/examples/data/ml-1m`。
4. 运行方式注意事项：本机曾观察到一次 DataLoader 停滞（py-spy 抓栈：主进程 join worker、8 个 worker 空等索引队列），以 `python -u`（无缓冲）重跑同一脚本，训练正常且逐 epoch 验证 AUC 与前台运行**逐位一致**。**所有正式运行均使用 `python -u` 并以前台/受控后台方式执行**。
5. `git status`：`m3oe/` 目录不是 git 仓库（无 .git），仅是官方仓库文件的拷贝；对官方仓库的状态核对通过 GitHub API 完成（commit tree `1b30217477836cac8f9bb68329a91e1c805aa4f5`）。

### §2.2 as-published 复现（5 seeds，`experiments/results/P0_repro.csv`）

- 6 项平均 test AUC：**0.7686 ± 0.0017**（逐 seed：0.77025/0.76855/0.76579/0.76971/0.76870）。
- 论文报告 0.7702 → 差 −0.0016（95% CI ±0.0021 覆盖论文值）→ **复现成功**。
- per-(d,t)：d0t0 0.7634±0.0028，d0t1 0.7775±0.0020，d1t0 0.7746±0.0030，d1t1 0.7930±0.0021，d2t0 0.7445±0.0028，d2t1 0.7587±0.0011。
- 全部种子在 18–25 epoch 触发早停；最优验证 epoch 7–14；每 seed 墙钟 633–874s。
- 早停统计：5 个种子在第 18–25 epoch 触发早停；最优验证 epoch（argmax 6 项平均 val AUC）为 7–14；墙钟 633–874s/种子（RTX 4060 Laptop）。
- 明显的域间结构：d2t0（0.7445）是持续的最弱项，d1t1（0.7930）是最强项；小域 d0 并不是最弱的。

> 逐种子明细：`experiments/results/P0_repro.csv`；原始日志：`experiments/p0_seed{2022..2026}.log`。

### §2.3 代码事实核查表（24 条断言全部核到；`EXP_CONTEXT.md` §2）

图例：✅成立 | ❌不成立 | ⚠️部分成立 | ❓未找到
核查方式：源码精读（附 文件:行号）、运行时验证（`experiments/verify_assertions.py`，纯构造无训练）、论文核对（arXiv:2404.18465v3 HTML 全文）、官方仓库核对（GitHub API）。

#### A. 特征与网络维度

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| A1 | EmbeddingLayer 对每个 SparseFeature 建独立 embedding table，RandomNormal(0, 1e-4) 初始化，embed_dim=16；DenseFeature embed_dim=1 直接透传 | ✅成立 | `MDMTRec/basic/layers.py:56-62`（embed_dict 每个 feature 一张表）、`features.py:61`（默认 `initializer=RandomNormal(0, 0.0001)`）、`run_movielens.py:68`（`embed_dim=16`，未覆盖默认 initializer）、`features.py:91`（DenseFeature `embed_dim=1`）、`layers.py:91`（dense 值直接 `x[fea.name].float()` 拼接，不经过 embedding） |
| A2 | 7 sparse × 16 + 1 dense = input_dim = 113 | ✅成立 | `run_movielens.py:44-46`（dense=['age'] 1 个；sparse 7 个）、`model.py:81`（`input_dim=sum(embed_dim)`）。运行时验证：`input_dim=113` |
| A3 | fcn_dims=[512,256,256,64]，域表示输出 h_d 是 256 维 | ✅成立 | `run_movielens.py:150`；`model.py:83,94-97`（star_dim=[113,512,256]，余下 fcn_dim=[256,64]）、`model.py:105`（star_mlp=MLP_N([512,256])）、`model.py:149-154`（emb = star_mlp(emb)+skip，两者均为 256 维） |
| A4 | skip_conn 是单层 Linear(input_dim,256)+LayerNorm+ReLU | ✅成立 | `model.py:97`（`skip_conn=MLP_N([113,256])`）；MLP_N 构造 `model.py:61-65`（n-1=1 组）。运行时打印：`Linear(113→256), LayerNorm(256), ReLU` |
| A5 | 专家网络是一层 MLP：Linear(256,64)+LayerNorm(64)+ReLU；prediction tower 64→64→1 | ✅成立 | `model.py:113-115`（expert=MLP_N([256,64])，运行时打印 `Linear(256→64), LayerNorm(64), ReLU`）；`model.py:127-136`（tower=`Linear(64,64)+LayerNorm+ReLU+Linear(64,1)`） |
| A6 | domain_indicator 既作为 sparse 特征参与 embedding，又作为最后一列充当域 ID（domain_id = x[-1,:]） | ✅成立 | `run_movielens.py:46`（在 sparse_features 中）、`run_movielens.py:26`（x_used_cols 最后一列）；`model.py:140`（`domain_id = x[-1, :].clone().detach()`）。trainer 侧同样用 `x_dict[-1,:]` 取域 ID（`ctr_trainer.py:89,192`） |

#### B. 融合权重与 AutoML

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| B1 | 只有共享专家有 gating；gating 是 `nn.Sequential(nn.Linear(256, expert_num), nn.Softmax(dim=1))`，共 D×T 个 | ✅成立 | `model.py:125`（`for i in range(domain_num*task_num)`，共 6 个；运行时打印 `Linear(256→4)+Softmax(dim=1)`）；`model.py:190-193`（bmm 只作用于共享专家输出 `fea`；domain/task 专家走标量权重） |
| B2 | gating 输入被 detach（`self.gate[i](emb.detach())`），梯度不回流到 h_d | ✅成立 | `model.py:156`（原文即 `emb.detach()`）。注意：gate 自身参数正常训练，只是 h_d 不接收 gate 的梯度 |
| B3 | 架构参数只有 4 个标量（α_d,α_t,β_d,β_t），由 Weights 经 Sigmoid 生成；β_d 不随 domain 变、β_t 不随 task 变，即 `Weights(1,...)` 而非 `Weights(D,...)` | ✅成立 | `model.py:88-91`（四个 `Weights(1, ...)`）；`model.py:44-46`（softmax_type=3 → `F.sigmoid`）；`model.py:176-188`（同一个 `_weight_bal_d()` 标量对所有 domain 循环复用，`_weight_bal_t()` 对所有 task 复用）。运行时：`deep_weights` shape=[1] |
| B4 | softmax_type 有 0/1/2/3 四种；0/1/2 被 `assert 0` 硬禁用；默认 3（sigmoid）；tau/tau_step 在 sigmoid 路径不生效 | ✅成立 | `model.py:33-43`（0/1/2 分支各有 `assert 0`）；`model.py:78`（默认 `softmax_type=3`，run_movielens 未传）；sigmoid 分支不用 tau（`model.py:44-46`）；`model.py:30-31` 的 tau 递减对输出无影响。附注：`model.py:48` 引用未定义名 `TAU`（NameError 隐患），仅 softmax_type 非法时可达，默认路径不触发 |
| B5 | 四个架构参数初值为 1.0，sigmoid(1)≈0.7311 | ✅成立 | `run_movielens.py:218-221`（exp_d/exp_t/bal_d/bal_t 默认 1）→ `model.py:19-24`（initial_deep=1 → `deep_weights=[1.0]`）。运行时：raw=[1.]，sigmoid=[0.7310586] |
| B6 | 论文 Eq.(7) 中 α_d/α_t 的归属，正文公式与其后讨论段落矛盾；核对代码实际是 α_d 乘哪个模块 | ✅成立（矛盾存在）；代码 = 讨论段落，≠ 公式 | 论文 Eq.(7)：`h̄_d = S(h_d) + α_d·T(h_d) + α_t·D(h_d)`——公式里 **α_d 乘 task 模块**；但其后讨论写 "α_d controls the overall weight of the domain expert module"——**说 α_d 控制域模块**，两处矛盾。代码 `model.py:190-193`：`_weight_exp_d() * weighted_domain_fea[...]`、`_weight_exp_t() * weighted_task_fea[...]`，即 **α_d 乘域专家、α_t 乘任务专家**，与讨论段落一致、与公式相反 |
| B7 | 架构参数更新：先训模型参数一整个 epoch，再取一个 mini-batch 算 loss 反传更新；一阶方法，无 unrolled step、无 Hessian | ✅成立（所述机制存在）；但注意 B10 使其实际动力学不同 | `ctr_trainer.py:124-146`（fit：先 `train_one_epoch` 整个 epoch，再取一个 batch `_loss.backward()` → `optimizer_loss_weight.step()`）；单次反传、无展开。**因 B10，架构参数同时还在每个训练 step 被主优化器更新**，实际更新频率远高于"每 epoch 一次" |
| B8 | 该 mini-batch 取自 train_dataloader 而非验证集（形如 `valid_data_iter = iter(train_dataloader)`） | ✅成立 | `ctr_trainer.py:125`：`valid_data_iter = iter(train_dataloader)`（变量名有误导性，实际取训练集 batch，且 train loader 是 shuffle 的） |
| B9 | arch loss 用未按域掩码的整体 BCE，与训练的 6 项掩码求和不是同一目标 | ✅成立 | `ctr_trainer.py:143`：`_loss = self.criterion(_y_pred, _val_y.float())`，BCELoss 对 [bs,2] 全 batch 求均值，三域样本混在一起；训练 loss 是按 domain mask 分桶后 6 项 BCE 求和（`ctr_trainer.py:88-114`）。两个目标不同 |
| B10 | 【重点】优化器参数分组 `if ~darts_para_flag(name)`：`~True==-2`、`~False==-1` 均为真值 → 主优化器含全部参数（含 4 个架构标量），架构参数被两个 Adam 同时更新 | ✅成立（bug 真实存在） | `ctr_trainer.py:52-53`：`darts_para_flag = lambda name: name.startswith('_weight_')`；`self.optimizer = optimizer_fn([param for name, param in self.model.named_parameters() if ~ darts_para_flag(name)], ...)`。Python 位取反：`~True=-2`、`~False=-1`，均为 truthy，过滤失效。运行时验证：主优化器 param 张量 **111/111**（含 `_weight_exp_d.deep_weights` 等 4 个架构参数）；`optimizer_loss_weight` 组内恰为这 4 个。两个 Adam 均会更新它们（主 Adam lr=1e-2 每 step、darts Adam lr=1e-3 每 epoch） |

#### C. 损失与训练配置

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| C1 | loss = 6 项 BCE 等权求和（`sum(torch.stack(logloss_list))`），无任务/域加权、无 load balancing、无辅助正则 | ✅成立 | `ctr_trainer.py:114`；全文件无其他 loss 项（weight_decay=1e-5 是唯一的正则，来自优化器参数） |
| C2 | 附近有被注释的可学习任务权重（形如 `# loss = sum(self.model.loss_weight() * ...)`） | ✅成立 | `ctr_trainer.py:113`：`# loss = sum(self.model.loss_weight() * torch.stack(logloss_list))` |
| C3 | 掩码分支硬编码：`if not (domain_num == 3 and task_num == 2)` 才走单一 BCE，只有 3 域 2 任务走 MDMT 掩码路径 | ✅成立 | `ctr_trainer.py:84`：原文即 `if not (domain_num == 3 and task_num == 2): loss = self.criterion(y_pred, y.float())`，else 走掩码求和路径 |
| C4 | 主优化器 Adam lr 默认 1e-2；架构优化器 Adam lr 默认 1e-3；weight_decay=1e-5 | ✅成立 | `run_movielens.py:209-210`（`learning_rate=1e-2`、`learning_rate_darts=1e-3`）、`:162`（两者均传 `weight_decay=args.weight_decay`=1e-5，`:212`）。注：trainer 内部默认 lr=1e-3（`ctr_trainer.py:50`）但被 run 脚本覆盖 |
| C5 | batch_size=4096*10=40960；epoch 上限 50；早停 patience=10，监控验证集 6 项平均 AUC | ✅成立 | `run_movielens.py:211,208,162`；`ctr_trainer.py:149-150`（`auc = sum(auc_list)/(domain_num*task_num)`）。⚠️附注：早停触发时才回滚到 best weights（`:154-157`）；若 50 epoch 跑满未触发早停，最终测试用的是**最后一轮**权重而非最优轮（`:159` 直接保存当前权重）——as-published 行为，后续阶段复现时须知晓 |
| C6 | scheduler_params 被传入但 scheduler_fn=None，调度未启用 | ✅成立 | `run_movielens.py:162`（传 `scheduler_params={"step_size":4,"gamma":0.85}`）、`:163`（`scheduler_fn=...StepLR...` 被注释）；`ctr_trainer.py:58-60`（`scheduler_fn is not None` 才建 scheduler） |
| C7 | train:val:test = 8:1:1，按行序切分而非随机打散 | ✅成立 | `run_movielens.py:84-87`（`int(len*0.8), int(len*0.9)` 连续切片）。运行时：800,167 / 100,021 / 100,021。（注意 DataLoader 的 shuffle=True 只影响训练 epoch 内的 batch 顺序，不影响切分本身） |

#### D. MovieLens 预处理

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| D1 | 域切分依据 age：{1,18}→d0，{25}→d1，{35,45,50,56}→d2 | ✅成立 | `run_movielens.py:28-30`（group1/2/3）、`:34`（`map_group_indicator` 按列表序返回 0/1/2） |
| D2 | click = rating > 3，like = rating > 4 | ✅成立 | `run_movielens.py:101-112`（`convert_target`：click 阈值 3，like 阈值 4，`v > thre → 1`）、`:60-61` |
| D3 | age 同时用作域划分依据和一个输入特征 | ✅成立 | `run_movielens.py:44`（age 在 dense_features 中，MinMax 缩放后入模）、`:34,44-45`（同时生成 domain_indicator）。附注：域 ID `domain_indicator` 也是 sparse 特征之一（`:46`），即"age 信息"以两种形式进入模型 |
| D4 | 实际三域样本量与占比，与论文 Table 1（210747/395556/393906，21.07%/39.55%/39.38%）核对 | ✅一致（我们的统计与论文完全相同） | 由 `ml-1m.csv` 全量计算：d0=210,747（21.07%）、d1=395,556（39.55%）、d2=393,906（39.38%），总数 1,000,209。与论文 Table 1 逐位一致 |

#### E. 仓库完整性（对官方 GitHub 仓库核对，tree `1b30217`）

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| E1 | KuaiRand-Pure 的预处理与运行脚本完全未开源 | ✅成立（对官方仓库） | 官方仓库全树（GitHub API 递归列出）无 run_kuaishou.py、无任何 KuaiRand 相关文件；README 只写 MovieLens-1M 流程。⚠️注意：**本地拷贝**多出一份 `examples/run_kuaishou.py`（P3 提示词说明系另行获得），非官方内容 |
| E2 | `__pycache__` 残留 `ablation.cpython-37.pyc` 等但对应 .py 未提交 | ✅成立 | 官方树含 `MDMTRec/models/multi_domain/__pycache__/ablation.cpython-37.pyc`、`adapter.cpython-37.pyc`、`adapter_dcn.cpython-37.pyc`、`adapter_wd.cpython-37.pyc`（另有 cpython-310 版），而 `models/multi_domain/` 下仅提交了 `model.py` 与 `__init__.py`。本地拷贝同现状 |
| E3 | 是否有 LICENSE | ❌不成立（即：没有 LICENSE） | 官方仓库树中无 LICENSE/COPYING 文件，README 无许可声明。**许可状态：未授权**，研究复现用途需自行注意 |

**核查范围**：A1–E3 共 24 条全部核到，无"未找到"项。

**代码侧其他值得记住的事实**：
1. sparse 特征被 LabelEncoder 编码**两次**：第一次 `+1`（run_movielens.py:56-58），第二次不带 +1 重新 fit_transform（:63-65）覆盖第一次结果。`vocab_size=nunique()`（:68）与最终编码一致，实际无害，但 `+1` 是死代码。
2. Weights 在 `forward()` 内修改 `self.tau`（model.py:30-31）——forward 有副作用；sigmoid 路径下无影响。
3. `fit()` 的 exp_d/exp_t/bal_d/bal_t 形参被忽略（实际初值在模型构造时传入）。
4. 早停判据是 6 项平均验证 AUC（等权），且"提升"定义为严格大于（callback.py:24）。
5. 训练 loss 的 6 项 BCE 对全 batch 等权，意味着小域 d0（21% 样本）在梯度中天然占少数——这正是 §3 P1 D0.4 要量化的问题。
6. **B10 bug 的实证**：3 epoch 冒烟运行后 4 个架构参数从 0.7311 移动到 0.7124/0.7224/0.7533/0.7395（主优化器逐 step 更新的效果）。

### §2.4 关键文件与关键函数清单（`EXP_CONTEXT.md` §4）

| 位置 | 内容 | 行号 |
|---|---|---|
| `MDMTRec/models/multi_domain/model.py` | `class Weights`（架构参数，sigmoid 路径 :44-46） | :12-48 |
| 同上 | `class MLP_N`（Linear+LayerNorm+ReLU 堆叠） | :51-72 |
| 同上 | `class MDMTRec.__init__`（4 个 Weights :88-91；star/star_mlp :94-105；专家 :113-123；gate :125；tower :127-136） | :75-136 |
| 同上 | `MDMTRec.forward`（域掩码 :140-145；embedding :147；skip/星型融合 :149-154；gate 输入 detach :156；三类专家输出 :159-173；加权 :175-193；tower+sigmoid :195；按域掩码回填 :197-207） | :138-207 |
| `MDMTRec/trainers/ctr_trainer.py` | `CTRTrainer.__init__`（**B10 bug 位置 :53**；双优化器 :53-56） | :26-66 |
| 同上 | `train_one_epoch`（C3 硬编码 :84；掩码分桶 :88-99；6 项 BCE 求和 :107-114；C2 注释行 :113） | :68-122 |
| 同上 | `fit`（B8 :125；每 epoch 后单 batch arch 更新 :133-146；6 项平均 AUC 早停 :149-157；保存 :159） | :124-159 |
| 同上 | `evaluate_multi_domain`（逐 (d,t) AUC/logloss） | :177-229 |
| `examples/run_movielens.py` | `get_movielens_data_rank_multidomain`（域划分 :28-34；特征列 :26,42-46；两次 LabelEncoder :56-58 与 :63-65；标签 :60-61；8:1:1 切分 :84-87） | :21-91 |
| 同上 | `main`（fcn_dims :150；模型构造 :156；trainer 构造 :162；结果写 CSV :170-201） | :144-201 |
| 同上 | argparse 默认值（lr/lr_darts/batch/epoch/expert_num/exp_d/exp_t/bal_d/bal_t 等） | :203-227 |
| `examples/run_kuaishou.py` | 非官方文件（详见 §5），KuaiRand 预处理 | :1-262 |
| `MDMTRec/basic/layers.py` | `EmbeddingLayer` | :29-116 |
| `MDMTRec/basic/features.py` | `SparseFeature`（默认 initializer :61）、`DenseFeature`（embed_dim=1 :91） | :49-94 |
| `MDMTRec/basic/callback.py` | `EarlyStopper`（严格大于才算提升） | :4-33 |
| `MDMTRec/utils/data.py` | `DataGenerator.generate_dataloader`（num_workers=8 默认；train shuffle=True，val/test shuffle=False） | :58-82 |

### §2.5 复现风险与偏差清单（`EXP_CONTEXT.md` §5）

**环境/运行层面的调整（均非代码行为修改）**
1. venv 使用 `--system-site-packages` 复用 torch 2.9.0+cu126。
2. 数据路径通过 CLI 参数显式指定（默认值指向作者本机路径）。
3. 所有运行加 `python -u`（规避一次观察到的 DataLoader 停滞；已验证训练轨迹逐位不变）。
4. 仓库无 LICENSE（E3）——仅做研究性复现，未再分发。
5. 本地 `run_kuaishou.py` 非官方文件（E1），§5 阶段单独调试，不混入 P0 结论。

**论文与代码不一致处**
1. B6：Eq.(7) 与讨论段落对 α_d/α_t 的归属矛盾；代码遵循讨论段落（α_d×域专家、α_t×任务专家）。
2. 论文未报告：优化器参数分组的 bug（B10）、早停未触发时用最后一轮权重评测（C5 附注）、架构参数实际每 step 被主优化器更新——论文把 AutoML 描述为"每 epoch 用验证 batch 更新架构权重"，与实际实现不符。
3. 论文 Table 1 的 MovieLens 三域规模与我们统计完全一致（D4）；KuaiRand 数字在 §5 核对。

### §2.6 P0 交付物索引

- `experiments/results/P0_repro.csv`：seed, domain, task, test_auc, test_logloss, best_epoch, wall_clock_sec
- `experiments/requirements.lock`：pip freeze 全量
- `experiments/verify_assertions.py` + 运行输出：断言的运行时验证证据
- `experiments/p0_seed{2022..2026}.log`：每个种子的完整训练日志
- `experiments/p0_wallclock.txt`：每种子墙钟时间

---

## §3 P1：训练插桩四项诊断（✅，`experiments/reports/P1_diagnosis.md` 全文）

> 一句话结论先行：
> - **D0.2（架构参数轨迹）：部分成立** — 四个架构参数全程偏移 ≤0.037，"几乎没搜动"大体成立；但终值并非全部落在 0.715–0.752（bal_d 系统性上漂到 ~0.767，exp_d 下探到 ~0.714）。
> - **D0.4（per-domain 梯度统计，本任务最关键项）：不成立** — 两个 batch size 下三域 SNR 均无显著差异（batch=40960 下 d0=0.509 vs d2=0.558，配对 t≈−2.5，绝对差仅 0.05；batch=4096 下差异完全消失甚至反号）。**"小域梯度噪声更大"这一前提被数据否定。**
> - **D0.6（per-domain 曲线）：成立** — 各 (d,t) 最优 epoch 显著错开（分散度 max−min=11 epochs）；**大域 d2 最早过拟合（峰值 ~6.4 epoch），小域 d0 居中（~9–12），d1 最晚（~14）**——与小域更早过拟合的直觉相反。
> - **D0.7（早停公平性）：成立但损失不对称** — 全局早停点（~11.4 epoch）对最早过拟合的 d2t0 造成最大损失（test AUC −0.0052），对其余 (d,t) 影响 ≤0.001（部分为负，即全局点反而更优）。

运行配置：5 seeds（2022–2026）× 2 batch sizes（40960 as-published / 4096），k=10 步采集一次 D0.4；MovieLens-1M；其余超参与 P0 as-published 完全一致。D0.4 全部 5 个种子完整采集（未减种子）。

### §3.0 等价性验证（插桩无损性）

固定种子 2022 的三重验证（`experiments/p1_equivalence_check.py`）：

1. instrument=0 与 P0 原版日志：24 个 epoch 的验证 AUC 流**逐位一致**，最终 test AUC 逐位一致（0.7713085167467406）。
2. instrument=1 与 instrument=0：24 个 epoch 验证 AUC 流逐位一致，最终 test AUC 逐位一致。
3. per-step loss 流：最大相对差 1.3e-7（float32 求和的舍入量级，两种记录方式的求和精度不同所致）。

过程中的重要发现（已修复并复验）：给 fit 增加 per-epoch 测试集评测时，**每次创建 DataLoader 迭代器都会从全局 RNG 抽一个 `_base_seed`（torch 2.9 dataloader.py:697，num_workers=0 也一样）**，曾导致 epoch 1 起 batch 序列漂移。修复：额外评测包裹 `torch.get_rng_state()/set_rng_state()` 快照。该坑已记录，后续所有阶段的 runner 都沿用此模式。

### §3.1 D0.2 架构参数轨迹

**结论：部分成立。** 四参数从 sigmoid(1)=0.7311 出发，全程最大偏移 0.022–0.037，训练确实"几乎不动"；但按 0.715–0.752 的判定区间，20 个 (seed,param) 序列中只有 9 个终值落在区间内——主要因为 bal_d（β_d）系统性上漂至 0.7668±0.0145（越过上界 0.752）、exp_d 下探至 0.7135±0.0144（略低于下界 0.715）。

| 参数 | 初值 | 终值（5 seeds 均值±std） | 全程最大偏移均值 |
|---|---|---|---|
| _weight_exp_d (α_d) | 0.7311 | 0.7135 ± 0.0144 | 0.0223 |
| _weight_exp_t (α_t) | 0.7311 | 0.7223 ± 0.0280 | 0.0225 |
| _weight_bal_d (β_d) | 0.7311 | 0.7668 ± 0.0145 | 0.0367 |
| _weight_bal_t (β_t) | 0.7311 | 0.7236 ± 0.0178 | 0.0233 |

机制注记（与 §2.3 P0 核查 B10 一致）：这些参数实际被主 Adam（lr=1e-2，每 step）和 darts Adam（lr=1e-3，每 epoch）同时更新，但即便如此偏移仍只有 ±0.02–0.04——梯度本身就小，AutoML 搜索几乎没有产生有效位移。图：`results/p1/figs/d02_arch_traj.png`；数据：`results/p1/D02_arch_traj.csv`（epoch=−1 为训练前初值快照）。

### §3.2 D0.4 per-domain 梯度统计（核心项）

**结论：不成立。** 共享参数（定义见下）上，小域 d0 的 SNR 与大域无实质差异：

| batch | d0 (21% 样本) | d1 (40%) | d2 (39%) | d0 vs d2 配对 t（112 组 epoch×seed 对） |
|---|---|---|---|---|
| 40960 | SNR 0.5085 ± 0.1692 | 0.5218 ± 0.1785 | 0.5584 ± 0.2005 | −2.46（绝对差仅 −0.050，相对差 ~9%） |
| 4096 | SNR 0.0524 ± 0.0122 | 0.0495 ± 0.0115 | 0.0511 ± 0.0139 | +0.65（反号，无差异） |

- batch=4096 下三域 SNR 一致地掉到 ~0.05（batch 缩小 10×，SNR 掉 ~10×，符合噪声主导的预期），域间差异完全消失。
- batch=40960 下 d2 略高，但绝对差只有 0.05，且 112 个 epoch×seed 对并非独立样本（同 epoch 内强自相关），名义显著但效应量可忽略。**判定：无显著差异。**
- 平均梯度范数（补充观察）：d0 = 0.0897，d1 = 0.0673，d2 = 0.0705（batch=40960）——小域梯度范数反而**更大**而非更小。
- 跨域梯度余弦（每 epoch 内 K' 个采集批次上的均值，再对 epoch 取均值±std）：

| batch | cos(d0,d1) | cos(d0,d2) | cos(d1,d2) |
|---|---|---|---|
| 40960 | 0.213 ± 0.163 | 0.176 ± 0.162 | 0.263 ± 0.170 |
| 4096 | 0.040 ± 0.027 | 0.038 ± 0.027 | 0.063 ± 0.034 |

  跨域梯度呈**弱正相关**（大 batch 下 ~0.2，小 batch 下近似正交），没有出现负余弦意义上的梯度冲突。域间的矛盾更接近"信息不重叠"而非"方向相反"。

**共享参数清单（本项目的统一定义，§6 组件 1 沿用）**
- 包含：7 张 embedding table（`embedding.embed_dict.*`）；`shared_weight`、`shared_bias`（星型融合的 W_sh/b_sh）；`skip_conn.*`；`star_mlp.*`；共享专家 `expert.*`（4 个）；域专家 `domain_expert.*`（3 个）；任务专家 `task_expert.*`（2 个）。
- 排除：per-(d,t) 的 `tower.*`（6 个）、per-(d,t) 的 `gate.*`（6 个）、各域私有的 `slot_weight.*`/`slot_bias.*`（各 3 个）、以及 4 个 `_weight_*` 架构标量（属于优化变量，不在 P1 的包含列表中，故排除并在此声明）。
- 共享参数展平维度约 58 万维；采集方式为对当前 batch 的 6 项掩码 BCE 按 d 取 L_d = BCE(t0|d)+BCE(t1|d)，逐域 backward（retain_graph=True）后克隆共享梯度、清零共享梯度，再走正常训练路径（清零-反传-step 前插桩不污染更新）。

图：`results/p1/figs/d04_snr.png`；数据：`results/p1/D04_grad_stats.csv`、`D04_grad_cosine.csv`。

### §3.3 D0.6 per-domain 训练曲线

**结论：成立，方向与直觉相反。** 各 (d,t) 验证 AUC 峰值 epoch（5 seeds）：

| (d,t) | 峰值 epoch（均值±std） | 各 seed |
|---|---|---|
| d0t0 | 12.4 ± 1.2 | 13, 11, 11, 14, 13 |
| d0t1 | 9.2 ± 2.3 | 10, 6, 9, 13, 8 |
| d1t0 | 13.8 ± 2.2 | 13, 10, 16, 14, 16 |
| d1t1 | 11.8 ± 2.4 | 13, 10, 8, 14, 14 |
| d2t0 | 6.4 ± 1.0 | 8, 7, 5, 6, 6 |
| d2t1 | 7.4 ± 0.8 | 9, 7, 7, 7, 7 |

- 最优 epoch 分散度：max−min = 11，std = 3.24。
- **大域 d2 最早过拟合（峰值 6–7 epoch），小域 d0 居中（9–12），中域 d1 最晚（12–14）。** "小域更早过拟合"不成立；反倒是小域在等权 6 项 loss 中被欠拟合方向拖着走（P0 的事实：loss 是 6 项 BCE 等权求和，d0 样本占比 21%）。
图：`results/p1/figs/d06_val_curves.png`；数据：`results/p1/D06_curves.csv`（train_loss 为按 (d,t) 样本数加权的掩码 BCE 均值）。

### §3.4 D0.7 早停判据的公平性

**结论：成立，且损失集中在大域的 click 任务上。** 全局早停点 = argmax(6 项平均验证 AUC)（as-published 协议：早停后回滚最优验证权重，平均最优 epoch ≈ 11.4）：

| (d,t) | 自身最优 epoch | 全局 epoch | 全局点 test AUC 损失 delta（global − own） |
|---|---|---|---|
| d0t0 | 12.4 | 11.4 | −0.0012（全局点反而更好） |
| d0t1 | 9.2 | 11.4 | +0.0009 |
| d1t0 | 13.8 | 11.4 | −0.0008 |
| d1t1 | 11.8 | 11.4 | −0.0010 |
| d2t0 | 6.4 | 11.4 | **+0.0052** |
| d2t1 | 7.4 | 11.4 | +0.0005 |

- 单一全局早停点对多数 (d,t) 的代价 ≤0.001 AUC，部分甚至为负（自身最优点的 test AUC 更差，说明 per-(d,t) 验证最优也存在选择噪声）。
- 唯一显著受害者是 **d2t0**（峰值最早 6.4 epoch，被拖到 11.4 评估，损失 0.0052 AUC）。公平性问题的来源是"峰值 epoch 错位"本身，而不是小域被牺牲。
- 数据：`results/p1/D07_earlystop.csv`（delta = test_auc_at_global − test_auc_at_own，正值=全局早停造成的损失）；图：`results/p1/figs/d07_dispersion.png`。
- 一致性校验：D0.7 的 test_auc_at_global 与 P0 复现的同种子 per-(d,t) test AUC 逐位一致（同一协议下的同一权重）。

### §3.5 对后续设计的影响

**per-domain 动量（组件 1）的设计前提不成立。** §3.2 的数据明确否定了"小域（d0）梯度噪声更大/ SNR 更低"的命题：batch=40960 下三域 SNR 0.51/0.52/0.56（差异 ≤0.05，t≈−2.5 但效应量可忽略），batch=4096 下完全无差异。因此"用域规模占比 p_d 驱动 per-domain EMA 窗长以补偿小域噪声"的动机链条断了一环。

但这不等于组件 1 必然无效，只是它的收益来源需要重新解释：§3.2 真正观察到的域间差异是
1. **小域梯度范数更大**（0.090 vs 0.067）——per-domain EMA 的 (1−γ_d) 缩放客观上会压制小域的更新幅度，方向与小域"更需要保护"相反；
2. **跨域梯度弱正相关**（大 batch ~0.2，小 batch ~0）——EMA 平滑的对象（时间噪声）在域间近似独立，Σ_d m_d 与 Σ_d g_d 的差异主要由各域噪声相消，理论上对任何域都是方差削减，不是小域专属收益。

**组件 2（β_d 的域条件课程）的前提部分受损：** 课程压低小域 β_d 的动机（小域先借跨域信息）仍然与 D0.4 的"梯度方向弱相关、无强冲突"相容（借得到、不打架），但"小域信号弱"的证据只有梯度范数更大这一点，方向恰好复杂化。**组件 3（架构参数平滑）的动机被 D0.2 强支持**：架构参数近乎不动（≤0.037），说明现有更新机制（单 batch、未掩码 loss、B10 双优化器）没有产生有效搜索。

后续（§4 P2 / §6 P4 / §7-§8 P5）因此按以下姿态推进：P2 先检验"AutoML 实际贡献"（fixed-0.7311 对照）与 B10 修复对照，确立主基线；P4 照常实现三个组件（默认关闭=基线）；P5 以数据说话——若组件 1 在各不均衡档位下与基线无显著差异，将如实报告"在 SNR 无域间差异的前提下，per-domain 动量退化为等动量"，并把这作为负面结果写入报告。

### §3.6 附：数据与图索引

| 诊断 | CSV | 图 |
|---|---|---|
| D0.2 | `results/p1/D02_arch_traj.csv` | `results/p1/figs/d02_arch_traj.png` |
| D0.4 | `results/p1/D04_grad_stats.csv`、`D04_grad_cosine.csv` | `results/p1/figs/d04_snr.png` |
| D0.6 | `results/p1/D06_curves.csv`（含 `D06_test.csv` 辅助） | `results/p1/figs/d06_val_curves.png` |
| D0.7 | `results/p1/D07_earlystop.csv` | `results/p1/figs/d07_dispersion.png` |

每个 (seed × batch) 的原始运行日志：`results/p1/s{seed}_b{batch}.log`；等价性证据：`results/p1/eq0.log`、`loss_stream_eq0.txt`、`loss_stream_s2022_b40960.txt`、`experiments/p1_equivalence_check.py` 输出（PASS）。

---

## §4 P2：AutoML 贡献与 bug 影响对照（✅，`experiments/reports/P2_baselines.md` 全文）

> 一句话结论：
> - **对照 1（D0.3）：命题成立** — fixed-0.7311 与原版无显著差异（配对 t 检验 p=0.32）；甚至比原版略高 0.0007。可微架构搜索的实际贡献 ≈ 0。论文 w/o AutoML（固定 0.5）的损失在我们复现里也只有 −0.0008，远小于论文声称的 −0.0065。
> - **对照 2（D0.5）：B10 修复对均值无显著影响（p=0.15），但把种子间方差从 ±0.0017 压到 ±0.0002**；修复后架构参数在正确的"每 epoch 一次"机制下几乎完全静止（max|dev| ≤ 0.0025，比带 bug 时小一个量级）——确认带 bug 时的参数移动来自主优化器的逐 step 更新，而非搜索本身。

### §4.1 实验设置

- 5 seeds（2022–2026），MovieLens-1M，其余超参全部 = as-published（§2 P0 配置）。
- 变体（全部经配置开关控制，默认关闭 = as-published）：
  - `orig`：as-published（即 P0 基线；另用 p2_run 重跑了 seed 2022 做交叉验证，最终 test AUC 与 P0 逐位一致 0.7713085167467406）。
  - `fixed073`：4 个架构参数冻结在 sigmoid(1)=0.7311（raw=logit(0.7311)=1.0），requires_grad=False，不在任何梯度图中。
  - `fixed05`：同上但冻结在 0.5（raw=0.0），对应论文 w/o AutoML 消融的"固定 0.5"设定。
  - `fixedbug`：修复 B10——主优化器只含非架构参数（103 个张量），darts 优化器只含 4 个架构参数（正确的布尔过滤替换 `~` 写法）。
- 实现：`experiments/p2_run.py`（`apply_fix_arch` / `rebuild_optimizers`），默认关闭。

### §4.2 主对照表（6 项平均 test AUC，5 seeds，均值±标准差 ddof=1）

| 变体 | 平均 AUC | vs 原版差值 | 配对 t | p（双侧） | 说明 |
|---|---|---|---|---|---|
| orig（as-published） | 0.7686 ± 0.0017 | — | — | — | = P0 复现 |
| fixed-0.7311 | **0.7693 ± 0.0010** | +0.0007 | t=−1.14 | **0.32**（n.s.） | 冻结在初值 |
| fixed-0.5 | 0.7678 ± 0.0022 | −0.0008 | — | n.s.（见 CSV） | 论文 w/o AutoML |
| fixedbug（修 B10） | 0.7700 ± 0.0002 | +0.0014 | t=−1.75 | **0.15**（n.s.） | 方差缩小 ~8× |

per-seed 6 项平均 AUC：

| seed | orig | fixed073 | fixed05 | fixedbug |
|---|---|---|---|---|
| 2022 | 0.7702 | 0.7701 | 0.7714 | 0.7702 |
| 2023 | 0.7686 | 0.7679 | 0.7679 | 0.7699 |
| 2024 | 0.7658 | 0.7687 | 0.7673 | 0.7702 |
| 2025 | 0.7697 | 0.7702 | 0.7656 | 0.7699 |
| 2026 | 0.7687 | 0.7698 | 0.7670 | 0.7697 |

per-(d,t) 明细：`experiments/results/p2/P2_baselines.csv`（由运行日志恢复：val AUC 流全精度、FINAL per-(d,t) 6 位小数）。

### §4.3 对照 1（D0.3）结论

**待检验命题："fixed-0.7311 与原版无显著差异，即 AutoML 的实际贡献接近零" —— 成立。**

- 配对（同种子）双侧 t 检验：差值 −0.0007（原版更低！），t=−1.14，p=0.32；Welch（非配对）p=0.46。5 seeds 下检测效力有限，但点估计的方向与"搜索有益"相反。
- 与论文数字的关系：论文 Table 3 报 w/o AutoML = 0.7637，比完整 M3oE 低 0.65 个 AUC 点。我们的 fixed-0.5 只低 0.08 个点（0.0008），且不显著。结合 §3.1 D0.2（架构参数全程偏移 ≤0.037），证据链一致：**这个可微搜索在 MovieLens 上没有做任何有意义的事**，论文消融中 0.65 点的差距更可能来自其消融实现与我们重建的"固定 0.5"不同（论文未开源 KuaiRand 与消融代码，无法核对），或来自随机噪声（论文无标准差、无显著性检验）。

### §4.4 对照 2（D0.5）结论

B10（`~` 用作逻辑非导致架构参数被两个 Adam 同时更新）在 §2.3 P0 已核实为真（`ctr_trainer.py:53`，主优化器含全部 111 个参数张量）。修复后的结果：

1. **均值几乎不变**（+0.0014，p=0.15，不显著）——bug 对平均精度影响可以忽略。
2. **方差骤降**：±0.0017 → ±0.0002（~8×）。原版的训练噪声有相当一部分来自"4 个架构标量被 lr=1e-2 的主 Adam 每 step 推着走"这一不受控动力学；修复后训练在种子间高度一致（5 个种子的 6 项平均 AUC 挤在 0.7697–0.7702）。
3. **修复后的架构参数轨迹**（`results/p2/P2_arch_traj_fixed.csv`，种子 2022–2026）：sigmoid 值全程 max|dev| ≤ 0.0025，初值 0.7311 → 终值 ~0.729–0.734。对比 §3.1 D0.2 中带 bug 时的 0.02–0.037 偏移：**正确机制（每 epoch 一次、darts lr=1e-3）下的"搜索"比 bug 机制下的意外漂移更安静**——进一步证实架构搜索是无效组件。
4. 命题"修 bug 后它们是否开始移动、移动到哪里"的答案：**几乎不动，停在初值附近**。

### §4.5 主基线选择

**后续实验（§7-§10 P5/P6）以 as-published 原版为主基线。** 理由：(a) 它与 §2 P0 复现直接衔接，是与论文对话的参照系；(b) 修 bug 对均值无显著影响（p=0.15），主要结论对该选择不敏感；(c) fixedbug 作为"稳定性检查"保留——任何新组件若只在原版上"有效"而幅度小于其 ±0.0017 的种子噪声，都将被判定为不可分辨。fixed-0.7311 作为第二对照进入 §7 P5 的方法列。

### §4.6 数据文件说明

- `results/p2/P2_baselines.csv`：variant, seed, domain, task, test_auc, test_logloss, best_epoch, n_epochs, early_stopped（**test_logloss 列为空：P2 的插桩只记录了 AUC；该缺口已在 P4/P5 的插桩中补上**——见 §12，**实际 P4/P5 评估脚本仍只输出 AUC**，故 §7/§10 P5/P6 主矩阵的 test_logloss 未在台账中留存）。
- `results/p2/P2_arch_traj_fixed.csv`：修 bug 版架构参数轨迹（5 seeds × 每 epoch × 4 参数）。
- `results/p2/*.log`：16 次运行的完整日志（orig 交叉验证 + fixed073 × 5 + fixed05 × 5 + fixedbug × 5）。

---

## §5 P3：KuaiRand-Pure 数据管线（✅；`experiments/reports/P3_kuairand.md` 全文）

> 一句话结论：官方 KuaiRand-Pure 数据已获取并重建预处理管线；按 tab∈{0,1,4} 切三域 + 全局时间排序 + 8:1:1 按行序切分，**三域 users/items/instances/占比与论文 Table 1 全部 12 个数字逐位一致（diff 全为 0）**；run_kuaishou.py 修复 2 处 bug 后端到端跑通冒烟训练。

### §5.1 数据来源与许可

| 项 | 值 |
|---|---|
| 数据集 | KuaiRand-Pure（KuaiRand 官方三版本之一；7,583 候选池视频上的全量日志） |
| 官方发布 | Zenodo record [10439422](https://zenodo.org/records/10439422)（`KuaiRand-Pure.tar.gz`，md5 0820331067a3784d9691136f772b35a7） |
| 实际获取途径 | **Kaggle 镜像** [arashnic/an-unbiased-sequential-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/an-unbiased-sequential-recommendation-dataset)（匿名下载成功；Zenodo 在本环境 DNS 被屏蔽，USTC 官方镜像 rec.ustc.edu.cn 匿名登录当时处于"升级维护中"（API 返回 4101），重试 40 次（约 6.7 小时）未恢复） |
| 许可 | **CC BY-SA 4.0**（Kaggle 页面标注 + 压缩包内 LICENSE 文件均为官方 CC BY-SA 4.0 文本）。允许研究使用、署名 + 相同方式共享。本仓库为研究性复现，不再分发数据 |
| 完整性核对 | 文件结构与官方 README 的 KuaiRand-Pure 清单一致（3 个日志 + user_features + video_basic + video_statistic + load_data_pure.py）；日志 193MB / 特征 10.4MB 与官方"184MB logs + 10MB features"吻合；行数：1,186,059 + 1,141,112 + 295,497 = 2,622,668，与官方 Pure 版"15 policies 下 1,436,609 + random 1,186,059"合计一致 ✓ |

**信息来源声明**：本节的发布渠道、许可、版本统计来自 **KuaiRand 官方文档（GitHub README + 压缩包内文件）**；Table 1 的目标数字来自**论文陈述**（arXiv:2404.18465v3）；下采样/排序等实现决策属于**我们的推定**（论文未开源该预处理，官方文档亦未规定切分顺序），均在下文标明。

### §5.2 重建的预处理管线（`experiments/p3_prepare.py`）

1. 三个日志按官方文件顺序读入并合并（2,622,668 行）。
2. 过滤 tab∈{0,1,4} → 2,507,666 行（95.62%）。**这三个 tab 与论文 Table 1 的三域逐位对应**（见 §5.4）。
3. join user_features_pure（user_id）+ video_features_basic_pure + video_features_statistic_pure（video_id，取 counts 列）：**inner join 零行丢失**（用户表覆盖 27,285 人、视频表覆盖 7,583 个，是日志的超集）。
4. 缺失值填 −1（onehot_feat4: 874 行、onehot_feat12–17: ~714 行、music_type: 203、tag: 96；均为特征类别缺失，非日志缺失）。
5. 全部 34 个特征列转为字符串后写 `D:/CODE/data/kuairand/processed/kuai_rand.csv`（416 MB），列 = run_kuaishou.py 的 x_used_cols + is_click + long_view。

### §5.3 划分规则与候选方案

下游管线（与 MovieLens 同一套）按**行序** 8:1:1 切分。行序的选择即是划分规则。候选方案：

| 候选 | 方案 | 结果 |
|---|---|---|
| A | 文件顺序拼接（std 4/08–4/21 → std 4/22–5/08 → random 4/22–5/08）+ tab∈{0,1,4} | 聚合统计与论文一致，但 **8:1:1 切分损坏**：random 日志只含 tab=1（1,178,025 行全部是主域），val/test 段落在 random 日志内部 → 验证集只有域 1，域 0/2 为空，评估直接崩溃 |
| **B（采用）** | **全局按 (date, time_ms) 时间排序** + tab∈{0,1,4} + 8:1:1 按行序 | 三域在每个 split 内都存在（train: 172,782/1,743,898/89,452；val: 2,692/246,232/1,843；test: 2,613/246,284/1,870）；无时间泄漏；确定性可复现。**与论文 Table 1 全部数字逐位一致** |
| C | 不过滤 tab（15 个 tab 全保留） | 得到 15 个域而非 3 个，与论文的 3 域设置不符 → 否决 |

**采用 B 的理由**：(i) 是唯一同时满足"论文三域聚合统计逐位复现"+"8:1:1 行序切分可用"的无泄漏方案；(ii) 时间切分是推荐日志的自然语义（后续模型不使用时间特征，不影响特征空间）。**声明：论文未公开其切分顺序（§2.3 E1 已核实 KuaiRand 预处理未开源），我们的时间排序属于推定重建；聚合统计（Table 1）不受该选择影响。**

### §5.4 与论文 Table 1 的核对（采用方案 B）

| 域 (tab) | users | items | instances | 占比 | 论文 | diff |
|---|---|---|---|---|---|---|
| d0 (tab=0) | 15,398 | 6,233 | 178,087 | 7.10% | 15,398 / 6,233 / 178,087 / 7.10% | **全 0** |
| d1 (tab=1) | 27,049 | 7,580 | 2,236,414 | 89.18% | 27,049 / 7,580 / 2,236,414 / 89.18% | **全 0** |
| d2 (tab=4) | 11,809 | 4,633 | 93,165 | 3.72% | 11,809 / 4,633 / 93,165 / 3.72% | **全 0** |

（CSV：`experiments/results/P3_dataset_stats.csv`，含候选方案对照。）

### §5.5 run_kuaishou.py 调试记录（该脚本系另行获得、非官方仓库内容）

| # | 问题 | 修复 |
|---|---|---|
| 1 | 构造 CTRTrainer 时漏传 `optimizer_params_darts`（默认 None 被 `**` 展开 → TypeError） | 补传 `{"lr": args.learning_rate_darts, "weight_decay": args.weight_decay}`，并新增 `--learning_rate_darts`（默认 1e-3，与 run_movielens.py 一致） |
| 2 | Windows spawn 无法把 ~700MB 的数据集对象 pickle 给 8 个 DataLoader worker（`reduction.dump: OSError Errno 22` / pickle truncated） | `num_workers=0`（进程内加载；确定性；此规模下速度足够）。**该脚本此前从未成功运行，不存在需要保持一致的"原版轨迹"** |
| 3 | （潜在，未触发）tab 过滤后 `domain_indicator` 若不从 0 重编码会导致域 0 为空 | 脚本已有该修复（原注释所述），实测通过 |

### §5.6 冒烟训练（1 seed × 1 epoch，as-published 超参）

`run_kuaishou.py --dataset_path D:/CODE/data/kuairand/processed --epoch 1 --seed 2022`：

- trainable params = 1,614,322；输入维度 34×16 = 544（34 个 sparse 特征，无 dense）。
- epoch 0 验证 6 项平均 AUC = 0.5601；test：d0t0 0.4576 / d0t1 0.4752 / d1t0 0.5568 / d1t1 0.5883 / d2t0 0.5950 / d2t1 0.6066。
- 1 epoch 的模型远未收敛（MovieLens 侧 1 epoch 也只有 ~0.72），**此结果仅证明管线可运行**，不作为任何方法的证据。主实验（§7-§8 P5）将给足 epoch 并配早停。
- 论文报告的 KuaiRand M3oE 平均 AUC ≈ 0.7460（Table 3，二手信息）——留给 §7-§8 P5 核对。

### §5.7 与论文的偏差声明（汇总）

1. **切分顺序为推定**：论文只说 8:1:1，未说行序依据；我们采用全局时间排序（§5.3 方案 B）。聚合统计不受影响（逐位一致），但 train/val/test 的具体样本与论文可能不同——这会带来微小的、无法消除的复现不确定性。
2. **特征集合是脚本的子集**：论文说"21 user + 12 video + 17 common"共 50 个特征；run_kuaishou.py 实际使用 33 个原始特征列（7 个 user 离散 + 18 个 onehot + 7 个 video + counts + domain_indicator）。论文未给出特征清单，无法进一步对齐。
3. **KuaiRand-Pure 的 is_like 等其他反馈**未使用（管线只定义 click/long_view 双任务，与论文一致）。

### §5.8 EXP_CONTEXT 更新

已把 KuaiRand 重建方案、许可与来源、num_workers 适配写入 `experiments/EXP_CONTEXT.md` §7（已整合进 §2）。§7-§8 P5 编排器的 `kuairand` 配置指向 `D:/CODE/data/kuairand/processed`。

---

## §6 P4：M3oE 三个新组件实现（`experiments/p4_run.py` + `experiments/reports/P4_implementation.md` 全文）

> 一句话结论：三个组件已实现并通过全部等价性验收测试（5/5 PASS）；组件关闭时与主基线**逐位一致**（bit-exact）；组件 1 的 D 次反传开销实测仅 +6% 耗时 / +216 MiB 显存（远低于 3× 上限）。

### §6.0 设计要点与前提声明

§3.2 P1 D0.4 结论（小域 SNR 并不更低）已经削弱了组件 1 的原始动机，但按计划照常实现，让 §7-§8 P5 的数据来裁决其有效性。组件 1 的实现做了两处与"逐域朴素实现"不同的关键选择：

1. **梯度获取 = D 次反传而非 D+1 次**：完整训练 loss 本就是 6 项 BCE 之和，逐域 L_d 反传即可同时得到 (a) 共享参数的逐域梯度 g_d，(b) 私有参数（tower/gate/slot 权重）的完整梯度——每个私有参数只出现在一个 L_d 的图中；(c) 4 个架构标量出现在所有 L_d 中，其跨域累加值恰好等于完整 loss 的梯度。因此无需再做一次全量反传。实测开销见 §6.4。
2. **逐域梯度采集时不污染私有梯度**：每域反传后只把**共享参数**的 .grad 克隆进 g_d 并清零共享参数；私有/架构参数的累加不被打断。

### §6.1 实现位置（`experiments/p4_run.py`，单一文件，未重构仓库代码）

| 组件 | 位置 | 说明 |
|---|---|---|
| 组件 2 模型变体 | `MDMTRecBeta`（:45-185） | 继承 `MDMTRec`；把 `_weight_bal_d`/`_weight_bal_t`（1 个标量 Weights）替换为 per-domain (D 维) / per-task (T 维) Parameter（`e_β` 初值 1.0，与原版一致）；`forward` 复制原版并改为 `bal_d[i]` 逐域取值。掩码 `.cpu()` 等细节与 `model.py:140-145` 保持一致 |
| 组件 2 课程 | `MDMTRecBeta.set_epoch`（:81-84）+ forward 内偏移 | 每 epoch 开始设 ρ(τ)=min(1, epoch/T_c)，T_c=总 epoch/2；forward 中 `sigmoid(e_β − κ_d·(1−ρ))`；κ_d = κ_0·log(p_max/p_d)/log(p_max/p_min)，p 全相等时 κ_d≡0；`curriculum_beta_t=0`（默认关，β_t 不加课程） |
| 组件 1 动量 | `PerDomainMomentum`（:187-241） | γ_d=clip(1−(1−γ_0)·p_d/p_max, [γ_0, γ_max])，p_d 取**训练集**逐域占比；`backward_accumulate`（:208）做 D 次反传并维护逐域 EMA `m_d ← γ_d·m_d + (1−γ_d)·g_d`；`apply_grads`（:233）把共享参数 .grad 替换为 **Σ_d m_d**（与 6 项等权 BCE 的共享梯度 Σ_d g_d 严格对应，drop-in）；私有参数走原累加梯度 |
| 组件 1 与 Adam 的关系 | `fit` 内重建优化器（:372-380） | `per_domain_only`：共享参数所在 param group β1=0，只用我们的 EMA；`per_domain_stacked`：保留 Adam β1=0.9，我们的 EMA 叠在其上；`none`（默认）：Adam 原样。β2 始终 0.999。注：Adam 的 β2 二阶矩会归一化梯度尺度差异，但不会消除方向噪声——本组件针对的是后者 |
| 组件 3 | `_arch_step`（:410-425）+ `_arch_update_at_step`（:472-497） | `arch_grad_ema`：架构参数梯度 EMA（γ_a=0.9，默认关）；`arch_loss_masked`：arch 更新 loss 改为与训练一致的 6 项掩码求和（默认关=原版未掩码整体 BCE）；`arch_update_every`：每 N step 更新一次架构参数（默认 0=每 epoch 一次，对齐基线）。**耦合提示**：原实现 50 epoch 只有 ~50 次架构更新，EMA 在这么少的样本上意义有限；开启 `arch_grad_ema` 时建议同时把 `arch_update_every` 设为 5-10（§7-§8/§10 P5/P6 采用的组合），EMA 才有足够的样本支撑 |
| 数据下采样（P5 用） | `downsample_train`（:288-322） | 只动训练集；保持最大域不变、其余两域按公共因子收缩至目标 max:min；1:1 档（低于现状 1.9:1）数学上必须收缩大域，此时改为较大域收缩到最小域规模；`RandomState(seed)` 确定性可复现 |
| 单域 oracle（P5 用） | 复用原版 `--domain` 机制（未用） | §7-§8 P5 将直接调用原版 run_movielens.py 的单域模式，保持 as-published |

新增配置项及默认值（默认全部 = 主基线 as-published）：

| 开关 | 默认 | 说明 |
|---|---|---|
| `--momentum_mode` | none | none / per_domain_only / per_domain_stacked |
| `--gamma0` / `--gamma_max` | 0.9 / 0.99 | 组件 1 的动量上下界 |
| `--curriculum_mode` | none | none / domain_conditioned |
| `--kappa0` | 0.0 | 可为负（反向课程） |
| `--curriculum_beta_t` | 0 | 是否对 β_t 加课程 |
| `--arch_grad_ema` | 0 | γ_a 由 `--gamma_a`（0.9）控制 |
| `--arch_loss_masked` | 0 | arch 更新用掩码 6 项 loss |
| `--arch_update_every` | 0 | 0 = 每 epoch（对齐基线） |
| `--fix_optim_grouping` | 0 | §4 P2 的 B10 修复开关，本文件复用 |
| `--downsample_ratio` / `--downsample_seed` | 0 / 777 | §7 P5 数据档位 |

### §6.2 等价性测试（验收标准，全部通过）

`experiments/tests/test_equivalence.py`（直接 `python` 运行，退出码 0）：

```
[PASS] (1) all-off val AUC stream == P0 for 3 epochs (bit-exact)
[PASS] (2) all-off per-step loss stream == P0 baseline for 60 steps (bit-exact)
[PASS] (3) curriculum degenerate: rho=1 forward is bit-identical to MDMTRec
[PASS] (4) rho=0 changes the forward (curriculum active)
[PASS] (5) analytic check: max offset at rho=1 is 0.0
```

- (1)(2) 用固定种子 2022 训练 3 个 epoch：关闭全部组件时 p4_run 走**原版** `CTRTrainer.fit/train_one_epoch`（只挂了 §3 P1 验证过的无损观察器），逐 step loss 与 P0 基线 60 步逐位一致，逐 epoch 验证 AUC 与 P0 日志逐位一致。
- (3)(4)(5) 验证组件 2 的关键性质：ρ(τ)=1 时前向与原版**逐位一致**——任何增益只能来自训练轨迹，不可能来自模型容量增加。

### §6.3 实现过程中发现并修复的坑（对后续阶段有用）

1. 模型 forward 返回的 `result` 在 CPU 上（model.py:198-206 未指定 device），原版 trainer 在调用侧做了 `y_pred.to(device)`——组件路径必须保留这一步，否则 CUDA 掩码索引 CPU 张量报错。
2. `arch_named()` 辅助：架构参数在原版是 `_weight_x.deep_weights`（Weights 模块），在 MDMTRecBeta 中是 `_weight_x`（Parameter），统一解析。
3. `--arch_update_every` 的 mid-epoch 更新会额外创建 train loader 迭代器，其 `_base_seed` 抽样会推后续 epoch 的 shuffle 序列（§3.0 P1 已发现的同类坑）；该路径属于非默认实验配置，§7-§8/§10 P5/P6 若使用将记录其 RNG 偏移。

### §6.4 组件开销实测（seed 2022，2 epochs，GPU 与 P2 链共享争用，横向可比）

| 配置 | fit 耗时 | 峰值显存 | 相对 all-off |
|---|---|---|---|
| all-off（基线） | 95.2 s | 942 MiB | 1.00× |
| 组件 1（per_domain_only, fixgroup=1） | 101.1 s | 1158 MiB | **1.06×** / +216 MiB |
| 组件 2（domain_conditioned, κ0=0.5, fixgroup=1） | 94.9 s | 941 MiB | 1.00× / ±0 |
| 组件 3（arch_ema + update_every=5, fixgroup=1） | 110.1 s | 942 MiB | 1.16× / ±0 |

- 组件 1 的 D=3 反传使耗时仅 +6%——反传约占总步时的一半，3 次反传理论 ~1.5-2×，实测更低（小模型下 kernel 启动开销占主导），**远低于 3× 上限，无实现浪费**。显存增量是 3 份共享参数 EMA 状态（~58 万维 × 3 × 4B ≈ 7 MiB 理论值；实测 216 MiB 含采集图与临时量）。
- 组件 2 零开销（等价的前向 + 每 epoch 一次标量更新）。
- 组件 3 的 +16% 来自每 epoch 额外 4 次 arch 更新（update_every=5 × 20 step）。

### §6.5 交付物索引

- `experiments/p4_run.py`：全部组件（单文件，~630 行）
- `experiments/tests/test_equivalence.py`：验收测试（5/5 PASS）
- `experiments/results/p4/P4_results.csv`：变体运行结果（本次为 bench 记录）
- `experiments/results/p4_bench/`、`experiments/bench_c{1,2,3}.log`：开销实测原始输出

### §6.6 组件要点摘要（与 §7-§8 P5 / §10 P6 衔接）

- 组件 1：γ_d=clip(1−(1−γ0)·p_d/p_max, [γ0, γ_max])（p_d=训练集占比）；D 次反传（retain_graph）逐域采集共享梯度并维护 EMA，`shared.grad = Σ_d m_d` drop-in；私有参数与架构标量由同一批反传自然累加（各私有参数只属于一个 L_d，架构标量出现在全部 L_d）；`momentum_mode`: none/per_domain_only（共享参数组 β1=0）/per_domain_stacked。§10 P6 消融附加开关：`--gamma_uniform`、`--gamma_source {formula,manual,snr}`、`--gamma_manual`。
- 组件 2：`MDMTRecBeta`（β_d 展开 D 维、β_t 展开 T 维）+ 课程 sigmoid(e_β − κ_d(1−ρ))，ρ=min(1, epoch/T_c)，T_c=总 epoch/2；κ_d=κ0·log(p_max/p_d)/log(p_max/p_min)；ρ=1 时前向与原版**逐位一致**（§6.2 单元测试通过）；`--kappa0` 可负（反向课程）；`--curriculum_beta_t` 默认关。
- 组件 3：`--arch_grad_ema`（γ_a=0.9）、`--arch_loss_masked`、`--arch_update_every N`（默认 0=每 epoch 对齐基线）。
- 数据开关：`--downsample_ratio/--downsample_seed`（1:1 档为等化收缩）、`--fix_arch X`（冻结架构参数）、`--kuairand 1`、`--single_domain d`（oracle），确定性预处理缓存（`results/cache/`，不消耗 torch RNG、不改变任何轨迹）。
- 等价性验收：`experiments/tests/test_equivalence.py` 5/5 PASS（全关闭=基线 bit-exact；课程退化性质 bit-exact）。

---

## §7 P5 MovieLens 主实验（✅ 全部完成，2026-09-08 闭环；`experiments/reports/ML_P5_REPORT.md` 全文）

> 数据源：`experiments/results/p5/P5_raw.csv`（主矩阵，唯一数值源）、`experiments/results/p5/P5_oracle.csv`（单域 oracle）
> 单 run 日志：`experiments/logs/<config>_<method>_<seed>.log` 与 `experiments/results/p5/logs/`（分属不同执行窗口的两个编排器）
> 编排日志：`experiments/logs/dual_orchestrator_ml_p3.log`（9/8 窗口）、`experiments/logs/dual_orchestrator_main.log`（早期窗口）
> 断点台账：`experiments/logs/stage3_checkpoint.md`
> 生成时间：2026-09-08 04:06；所有数字由 CSV 实算，未引用任何既有文档的转抄值。

### §7.1 运行环境与硬件状态（`ML_P5_REPORT.md` §1）

- 编排器：`experiments/dual_orchestrator.py --dataset ml --concurrency 2`。**并发上限 2**（`assert concurrency <= 2`，8 GiB VRAM 笔记本硬约束），相邻 `[launch]` **错峰 30s**（`STAGGER_SEC = 30`）。断点续跑判据：目标 CSV 中该 `(config, method, seed)` 无 `status == OK` 行才入队。
- 训练入口 `experiments/p4_run.py`，`--epoch 50`，`num_workers = 0`（`p4_run.py:375` 既有设置），各方法共用同一套早停规则。
- 域-任务：MovieLens 3 域 × 2 任务；`per_dt` 六值顺序 **[d0t0, d0t1, d1t0, d1t1, d2t0, d2t1]**。原始 ML-1M 训练集域占比 ≈ 19.49% / 39.82% / 40.69%（日志 `[p4] train domain share`），`--downsample_ratio` 越大则域间越倾斜。

#### 跨日漂移探针（9/8 窗口开跑前置条件）

重跑 `ml_orig` baseline seed 2022（独立 `--tag drift_probe_d0908 --outdir experiments/results/p5/drift_probe_d0908`，**未写入 P5_raw.csv**），日志 `experiments/logs/drift_ml_orig_baseline_2022_d0908.log`：

| 项 | 既有窗口值 | 9/8 探针值 | 漂移 |
|---|---|---|---|
| FINAL test auc | 0.7720050200624052 | 0.7720050200624052 | **0.000000** |
| per_dt d0t0 | 0.766026 | 0.766026 | 0.000000 |
| per_dt d0t1 | 0.776147 | 0.776147 | 0.000000 |
| per_dt d1t0 | 0.747527 | 0.747527 | 0.000000 |
| per_dt d1t1 | 0.783104 | 0.783104 | 0.000000 |
| per_dt d2t0 | 0.794131 | 0.794131 | 0.000000 |
| per_dt d2t1 | 0.761827 | 0.761827 | 0.000000 |
| best val auc | 0.7518606615810101 | 0.7518606615810101 | 0.000000 |
| peak cuda mem | 942 MiB | 942 MiB | — |
| fit wall | 423.1s | 413.8s | 计时抖动，非指标 |

漂移 0.000000 ≤ 阈值 0.003，规程通过后才启动 9/8 跑批。

#### 各档 wall_sec / peak_mem_mb 实读（P5_raw.csv，仅 OK 行）

| config | 下采样比 | n(OK) | seeds | wall min | max | mean | fit min | max | peak_mem_mb |
|---|---|---|---|---|---|---|---|---|---|
| ml_orig | 0（原始 ML-1M，域比 ≈ 1.95:1） | 30 | 5 | 187 | 513 | 309 | 183.5 | 508.0 | 1158/941/942 |
| ml_r1 | 1.0 | 18 | 3 | 146 | 352 | 260 | 140.5 | 346.6 | 1158/941/942 |
| ml_r3 | 3.0 | 18 | 3 | 187 | 845 | 502 | 182.7 | 838.5 | 1158/941/942 |
| ml_r10 | 10.0 | 18 | 3 | 140 | 350 | 245 | 133.3 | 344.6 | 1158/941/942 |
| ml_r30 | 30.0 | 30 | 5 | 64 | 120 | 93 | 60.6 | 116.0 | 1158/941/942 |
| ml_r100 | 100.0 | 30 | 5 | 62 | 200 | 101 | 58.7 | 195.5 | 1158/941/942 |

peak_mem_mb 呈两簇：941 / 942 MiB（baseline、fixed073、comp2）与 1158 MiB（含 comp1 类 per-domain 动量的 comp1 / comp1_comp2 / comp123），与 KuaiRand 侧同一规律（组件 1 引入额外动量缓冲）。

#### 执行窗口与中断事实

- 9/8 窗口（本轮）：`ml_r3` 补齐 14 格 + `ml_r10` 全新 18 格 + ML oracle 45 格，共 77 格入队（编排日志首行 `queued 77 tasks`），全部一次通过，**0 FAILED**、无重投。编排日志末行 `DUAL_ORCHESTRATOR DONE`。
- `ml_orig` / `ml_r1` / `ml_r30` / `ml_r100` 主矩阵与 `ml_r3` 的前 4 格产出于此前窗口，本轮未重跑（断点续跑只补 `status != OK` 的格子）。
- Runner 会话在 9/8 期间多次因 API 过载（HTTP 529）中断，但训练进程由编排器独立托管（`subprocess.Popen` + 全量重定向到独立日志），**不受会话中断影响**；跨班以 `experiments/logs/stage3_checkpoint.md` 无损接续。本轮全部格子的 wall_sec 连续、无被中断截断的 run。

### §7.2 原生指标明细矩阵（`ML_P5_REPORT.md` §2）

> 两个指标口径，**分开列示、不可混用**：
> - **口径 A**：`per_dt` 六个 (d,t) 的 AUC 先各自跨 seed 取均值（表内六列），其再平均记作 A 列；
> - **口径 B**：CSV 的 `test_auc_mean` 列（评估脚本内部按样本聚合的单标量）跨 seed 取均值。
> std 主口径 **ddof = 0（population）**，另附 ddof = 1（样本 std）对照表。所有数值 6 位小数，取自 CSV 原生值。

#### §7.2.1 ml_orig（downsample_ratio = 0；实测 5 seeds：2022, 2023, 2024, 2025, 2026）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.772005 | 0.766026 | 0.776147 | 0.747527 | 0.783104 | 0.794131 | 0.761827 | 320 |
| baseline | 2023 | 0.767875 | 0.765806 | 0.773044 | 0.741030 | 0.776634 | 0.789136 | 0.758122 | 206 |
| baseline | 2024 | 0.769944 | 0.764750 | 0.773352 | 0.742711 | 0.779354 | 0.791826 | 0.758716 | 304 |
| baseline | 2025 | 0.768935 | 0.765913 | 0.772453 | 0.745194 | 0.780081 | 0.791155 | 0.759525 | 285 |
| baseline | 2026 | 0.771731 | 0.766809 | 0.777397 | 0.745286 | 0.779683 | 0.795095 | 0.760218 | 266 |
| fixed073 | 2022 | 0.771999 | 0.766444 | 0.775808 | 0.747065 | 0.783378 | 0.794353 | 0.761626 | 244 |
| fixed073 | 2023 | 0.771053 | 0.766399 | 0.777460 | 0.742109 | 0.776577 | 0.793272 | 0.760679 | 187 |
| fixed073 | 2024 | 0.770659 | 0.764336 | 0.775179 | 0.742452 | 0.779693 | 0.791060 | 0.758872 | 292 |
| fixed073 | 2025 | 0.770584 | 0.766324 | 0.772435 | 0.745301 | 0.780816 | 0.791869 | 0.759278 | 328 |
| fixed073 | 2026 | 0.770084 | 0.765944 | 0.775366 | 0.745525 | 0.777081 | 0.793118 | 0.757779 | 417 |
| comp1 | 2022 | 0.775458 | 0.770381 | 0.776546 | 0.754519 | 0.783863 | 0.795573 | 0.765214 | 412 |
| comp1 | 2023 | 0.773036 | 0.763294 | 0.776367 | 0.752350 | 0.781506 | 0.794766 | 0.764616 | 477 |
| comp1 | 2024 | 0.773538 | 0.767031 | 0.776438 | 0.753147 | 0.782545 | 0.790415 | 0.766567 | 513 |
| comp1 | 2025 | 0.775795 | 0.768509 | 0.778627 | 0.754264 | 0.780244 | 0.796070 | 0.767899 | 346 |
| comp1 | 2026 | 0.770754 | 0.763027 | 0.777372 | 0.748718 | 0.779458 | 0.789995 | 0.764162 | 339 |
| comp2 | 2022 | 0.771860 | 0.764184 | 0.777541 | 0.748594 | 0.779330 | 0.793109 | 0.761895 | 250 |
| comp2 | 2023 | 0.770523 | 0.761893 | 0.777527 | 0.746143 | 0.777468 | 0.792965 | 0.760289 | 261 |
| comp2 | 2024 | 0.768311 | 0.763711 | 0.771692 | 0.740252 | 0.777347 | 0.791672 | 0.756431 | 210 |
| comp2 | 2025 | 0.767701 | 0.763634 | 0.771653 | 0.741003 | 0.779157 | 0.791218 | 0.755244 | 190 |
| comp2 | 2026 | 0.772146 | 0.768503 | 0.778540 | 0.743666 | 0.782070 | 0.794360 | 0.758146 | 261 |
| comp1_comp2 | 2022 | 0.774839 | 0.767268 | 0.776741 | 0.755178 | 0.782145 | 0.793119 | 0.768241 | 348 |
| comp1_comp2 | 2023 | 0.773960 | 0.763485 | 0.779214 | 0.753711 | 0.780615 | 0.794991 | 0.767607 | 336 |
| comp1_comp2 | 2024 | 0.772577 | 0.766266 | 0.774961 | 0.750975 | 0.783634 | 0.791421 | 0.766692 | 389 |
| comp1_comp2 | 2025 | 0.775842 | 0.767676 | 0.778691 | 0.754239 | 0.778319 | 0.796745 | 0.768949 | 335 |
| comp1_comp2 | 2026 | 0.770019 | 0.760277 | 0.775910 | 0.747605 | 0.774803 | 0.791421 | 0.762379 | 302 |
| comp123 | 2022 | 0.772317 | 0.762178 | 0.778395 | 0.752293 | 0.779814 | 0.793894 | 0.766406 | 321 |
| comp123 | 2023 | 0.773349 | 0.765346 | 0.777719 | 0.750888 | 0.779808 | 0.794877 | 0.765441 | 286 |
| comp123 | 2024 | 0.772079 | 0.763157 | 0.775140 | 0.748028 | 0.779287 | 0.791973 | 0.764740 | 282 |
| comp123 | 2025 | 0.770321 | 0.763827 | 0.774640 | 0.744307 | 0.777914 | 0.792495 | 0.760700 | 265 |
| comp123 | 2026 | 0.772677 | 0.766171 | 0.773183 | 0.752366 | 0.779987 | 0.792104 | 0.765261 | 305 |

##### ml_orig 每方法均值 ± std（ddof = 0，n = 5）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.765861±0.000658 | 0.774479±0.001936 | 0.744350±0.002253 | 0.779771±0.002060 | 0.792269±0.002131 | 0.759682±0.001287 | 0.769402±0.001517 | 0.770098±0.001589 |
| fixed073 | 0.765889±0.000797 | 0.775250±0.001621 | 0.744490±0.001907 | 0.779509±0.002498 | 0.792734±0.001150 | 0.759647±0.001357 | 0.769587±0.000972 | 0.770876±0.000641 |
| comp1 | 0.766448±0.002888 | 0.777070±0.000858 | 0.752600±0.002092 | 0.781523±0.001576 | 0.793364±0.002616 | 0.765692±0.001368 | 0.772783±0.001448 | 0.773716±0.001823 |
| comp2 | 0.764385±0.002201 | 0.775391±0.003058 | 0.743932±0.003125 | 0.779074±0.001709 | 0.792665±0.001117 | 0.758401±0.002437 | 0.768975±0.001761 | 0.770108±0.001812 |
| comp1_comp2 | 0.764994±0.002775 | 0.777103±0.001620 | 0.752342±0.002750 | 0.779903±0.003098 | 0.793539±0.002075 | 0.766774±0.002319 | 0.772443±0.001950 | 0.773447±0.002022 |
| comp123 | 0.764136±0.001449 | 0.775815±0.001952 | 0.749576±0.003067 | 0.779362±0.000761 | 0.793069±0.001132 | 0.764510±0.001980 | 0.771078±0.001253 | 0.772148±0.001009 |

##### ml_orig 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.765861±0.000736 | 0.774479±0.002164 | 0.744350±0.002519 | 0.779771±0.002304 | 0.792269±0.002383 | 0.759682±0.001439 | 0.769402±0.001696 | 0.770098±0.001776 |
| fixed073 | 0.765889±0.000891 | 0.775250±0.001812 | 0.744490±0.002132 | 0.779509±0.002793 | 0.792734±0.001285 | 0.759647±0.001517 | 0.769587±0.001087 | 0.770876±0.000716 |
| comp1 | 0.766448±0.003229 | 0.777070±0.000960 | 0.752600±0.002339 | 0.781523±0.001762 | 0.793364±0.002925 | 0.765692±0.001530 | 0.772783±0.001619 | 0.773716±0.002038 |
| comp2 | 0.764385±0.002461 | 0.775391±0.003419 | 0.743932±0.003493 | 0.779074±0.001911 | 0.792665±0.001249 | 0.758401±0.002724 | 0.768975±0.001969 | 0.770108±0.002026 |
| comp1_comp2 | 0.764994±0.003103 | 0.777103±0.001811 | 0.752342±0.003075 | 0.779903±0.003463 | 0.793539±0.002320 | 0.766774±0.002593 | 0.772443±0.002180 | 0.773447±0.002261 |
| comp123 | 0.764136±0.001620 | 0.775815±0.002182 | 0.749576±0.003430 | 0.779362±0.000851 | 0.793069±0.001266 | 0.764510±0.002213 | 0.771078±0.001401 | 0.772148±0.001129 |

#### §7.2.2 ml_r1（downsample_ratio = 1.0；实测 3 seeds：2022, 2023, 2024）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.758864 | 0.757542 | 0.762240 | 0.727238 | 0.775920 | 0.781601 | 0.744444 | 197 |
| baseline | 2023 | 0.758649 | 0.763101 | 0.765615 | 0.729718 | 0.776268 | 0.776634 | 0.741926 | 237 |
| baseline | 2024 | 0.759801 | 0.762309 | 0.763752 | 0.727214 | 0.777541 | 0.780853 | 0.744229 | 146 |
| fixed073 | 2022 | 0.760238 | 0.759362 | 0.765237 | 0.729287 | 0.772319 | 0.783720 | 0.747181 | 228 |
| fixed073 | 2023 | 0.758425 | 0.763693 | 0.766165 | 0.729725 | 0.776084 | 0.777851 | 0.742392 | 223 |
| fixed073 | 2024 | 0.758988 | 0.762601 | 0.763380 | 0.727588 | 0.777756 | 0.780328 | 0.743687 | 252 |
| comp1 | 2022 | 0.766762 | 0.765430 | 0.768293 | 0.734847 | 0.780845 | 0.787924 | 0.753006 | 299 |
| comp1 | 2023 | 0.766278 | 0.764329 | 0.770502 | 0.737830 | 0.783366 | 0.787600 | 0.754167 | 313 |
| comp1 | 2024 | 0.764897 | 0.763631 | 0.769612 | 0.733010 | 0.779363 | 0.785416 | 0.753346 | 281 |
| comp2 | 2022 | 0.760607 | 0.759268 | 0.765479 | 0.729307 | 0.772588 | 0.783752 | 0.747201 | 242 |
| comp2 | 2023 | 0.759130 | 0.763333 | 0.766188 | 0.729751 | 0.776650 | 0.777953 | 0.742693 | 292 |
| comp2 | 2024 | 0.759409 | 0.762157 | 0.763957 | 0.727602 | 0.776702 | 0.780459 | 0.744786 | 195 |
| comp1_comp2 | 2022 | 0.763541 | 0.759208 | 0.767562 | 0.734779 | 0.779738 | 0.784382 | 0.754522 | 311 |
| comp1_comp2 | 2023 | 0.766123 | 0.764661 | 0.770857 | 0.737853 | 0.783522 | 0.787505 | 0.754304 | 352 |
| comp1_comp2 | 2024 | 0.764886 | 0.763035 | 0.769247 | 0.732690 | 0.778906 | 0.785391 | 0.753395 | 251 |
| comp123 | 2022 | 0.768032 | 0.763967 | 0.772347 | 0.739128 | 0.782143 | 0.788534 | 0.757223 | 338 |
| comp123 | 2023 | 0.764504 | 0.761597 | 0.768283 | 0.735542 | 0.778603 | 0.784849 | 0.754204 | 278 |
| comp123 | 2024 | 0.764186 | 0.759494 | 0.768860 | 0.733456 | 0.775492 | 0.784539 | 0.750253 | 247 |

##### ml_r1 每方法均值 ± std（ddof = 0，n = 3）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.760984±0.002455 | 0.763869±0.001380 | 0.728057±0.001175 | 0.776576±0.000697 | 0.779696±0.002187 | 0.743533±0.001140 | 0.758786±0.000475 | 0.759105±0.000500 |
| fixed073 | 0.761885±0.001839 | 0.764927±0.001158 | 0.728867±0.000922 | 0.775386±0.002274 | 0.780633±0.002406 | 0.744420±0.002023 | 0.759353±0.000123 | 0.759217±0.000758 |
| comp1 | 0.764463±0.000741 | 0.769469±0.000907 | 0.735229±0.001986 | 0.781191±0.001652 | 0.786980±0.001114 | 0.753506±0.000487 | 0.765140±0.000915 | 0.765979±0.000790 |
| comp2 | 0.761586±0.001708 | 0.765208±0.000931 | 0.728887±0.000926 | 0.775313±0.001927 | 0.780721±0.002375 | 0.744893±0.001842 | 0.759435±0.000132 | 0.759716±0.000640 |
| comp1_comp2 | 0.762301±0.002286 | 0.769222±0.001345 | 0.735107±0.002121 | 0.780722±0.002009 | 0.785759±0.001301 | 0.754074±0.000488 | 0.764531±0.001368 | 0.764850±0.001055 |
| comp123 | 0.761686±0.001827 | 0.769830±0.001795 | 0.736042±0.002342 | 0.778746±0.002717 | 0.785974±0.001815 | 0.753893±0.002854 | 0.764362±0.002157 | 0.765574±0.001743 |

##### ml_r1 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.760984±0.003007 | 0.763869±0.001691 | 0.728057±0.001439 | 0.776576±0.000853 | 0.779696±0.002678 | 0.743533±0.001396 | 0.758786±0.000581 | 0.759105±0.000612 |
| fixed073 | 0.761885±0.002252 | 0.764927±0.001418 | 0.728867±0.001129 | 0.775386±0.002785 | 0.780633±0.002946 | 0.744420±0.002477 | 0.759353±0.000150 | 0.759217±0.000928 |
| comp1 | 0.764463±0.000907 | 0.769469±0.001111 | 0.735229±0.002433 | 0.781191±0.002024 | 0.786980±0.001364 | 0.753506±0.000597 | 0.765140±0.001120 | 0.765979±0.000968 |
| comp2 | 0.761586±0.002092 | 0.765208±0.001140 | 0.728887±0.001134 | 0.775313±0.002360 | 0.780721±0.002908 | 0.744893±0.002256 | 0.759435±0.000161 | 0.759716±0.000784 |
| comp1_comp2 | 0.762301±0.002800 | 0.769222±0.001648 | 0.735107±0.002597 | 0.780722±0.002460 | 0.785759±0.001594 | 0.754074±0.000598 | 0.764531±0.001675 | 0.764850±0.001292 |
| comp123 | 0.761686±0.002238 | 0.769830±0.002199 | 0.736042±0.002869 | 0.778746±0.003328 | 0.785974±0.002222 | 0.753893±0.003495 | 0.764362±0.002642 | 0.765574±0.002134 |

#### §7.2.3 ml_r3（downsample_ratio = 3.0；实测 3 seeds：2022, 2023, 2024）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.762816 | 0.752906 | 0.767391 | 0.738657 | 0.766228 | 0.785613 | 0.758244 | 330 |
| baseline | 2023 | 0.763930 | 0.756620 | 0.767991 | 0.742297 | 0.769329 | 0.785315 | 0.758560 | 187 |
| baseline | 2024 | 0.762216 | 0.754977 | 0.765577 | 0.739769 | 0.766786 | 0.780707 | 0.755578 | 351 |
| fixed073 | 2022 | 0.762624 | 0.754874 | 0.769867 | 0.742830 | 0.768580 | 0.785246 | 0.757065 | 452 |
| fixed073 | 2023 | 0.764084 | 0.757538 | 0.768048 | 0.741886 | 0.770517 | 0.786333 | 0.757651 | 340 |
| fixed073 | 2024 | 0.760307 | 0.752860 | 0.767915 | 0.740122 | 0.767734 | 0.781327 | 0.757177 | 400 |
| comp1 | 2022 | 0.758995 | 0.750916 | 0.764295 | 0.737446 | 0.768200 | 0.784286 | 0.756653 | 435 |
| comp1 | 2023 | 0.763159 | 0.751757 | 0.763104 | 0.741643 | 0.766774 | 0.785314 | 0.761692 | 535 |
| comp1 | 2024 | 0.760645 | 0.749012 | 0.767116 | 0.736079 | 0.766545 | 0.781436 | 0.758114 | 670 |
| comp2 | 2022 | 0.761356 | 0.751212 | 0.767632 | 0.736753 | 0.768001 | 0.781915 | 0.753787 | 450 |
| comp2 | 2023 | 0.763252 | 0.755864 | 0.763827 | 0.741141 | 0.768586 | 0.782690 | 0.759451 | 385 |
| comp2 | 2024 | 0.761556 | 0.758685 | 0.764424 | 0.737516 | 0.769503 | 0.779207 | 0.752537 | 415 |
| comp1_comp2 | 2022 | 0.761687 | 0.750423 | 0.764978 | 0.736151 | 0.767190 | 0.785031 | 0.758729 | 675 |
| comp1_comp2 | 2023 | 0.760116 | 0.752955 | 0.762739 | 0.734592 | 0.766443 | 0.776845 | 0.754850 | 680 |
| comp1_comp2 | 2024 | 0.762002 | 0.756900 | 0.766980 | 0.741221 | 0.767434 | 0.784278 | 0.757545 | 660 |
| comp123 | 2022 | 0.762502 | 0.754667 | 0.765352 | 0.741278 | 0.767983 | 0.783579 | 0.759066 | 845 |
| comp123 | 2023 | 0.763681 | 0.754867 | 0.764243 | 0.736307 | 0.773506 | 0.784324 | 0.758099 | 580 |
| comp123 | 2024 | 0.755946 | 0.754774 | 0.759958 | 0.738278 | 0.764257 | 0.777884 | 0.755356 | 655 |

##### ml_r3 每方法均值 ± std（ddof = 0，n = 3）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.754834±0.001520 | 0.766986±0.001026 | 0.740241±0.001523 | 0.767448±0.001350 | 0.783878±0.002246 | 0.757461±0.001337 | 0.761808±0.001157 | 0.762987±0.000710 |
| fixed073 | 0.755091±0.001916 | 0.768610±0.000890 | 0.741613±0.001122 | 0.768944±0.001165 | 0.784302±0.002150 | 0.757298±0.000254 | 0.762643±0.001055 | 0.762338±0.001555 |
| comp1 | 0.750562±0.001148 | 0.764838±0.001682 | 0.738389±0.002367 | 0.767173±0.000732 | 0.783679±0.001640 | 0.758820±0.002117 | 0.760577±0.000839 | 0.760933±0.001712 |
| comp2 | 0.755254±0.003081 | 0.765294±0.001671 | 0.738470±0.001914 | 0.768697±0.000618 | 0.781271±0.001493 | 0.755258±0.003008 | 0.760707±0.000880 | 0.762055±0.000851 |
| comp1_comp2 | 0.753426±0.002665 | 0.764899±0.001732 | 0.737321±0.002830 | 0.767022±0.000422 | 0.782051±0.003694 | 0.757041±0.001623 | 0.760294±0.001767 | 0.761268±0.000825 |
| comp123 | 0.754769±0.000082 | 0.763184±0.002326 | 0.738621±0.002044 | 0.768582±0.003800 | 0.781929±0.002876 | 0.757507±0.001571 | 0.760765±0.001660 | 0.760710±0.003402 |

##### ml_r3 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.754834±0.001861 | 0.766986±0.001257 | 0.740241±0.001865 | 0.767448±0.001653 | 0.783878±0.002750 | 0.757461±0.001638 | 0.761808±0.001417 | 0.762987±0.000870 |
| fixed073 | 0.755091±0.002347 | 0.768610±0.001091 | 0.741613±0.001375 | 0.768944±0.001427 | 0.784302±0.002633 | 0.757298±0.000311 | 0.762643±0.001292 | 0.762338±0.001904 |
| comp1 | 0.750562±0.001406 | 0.764838±0.002060 | 0.738389±0.002899 | 0.767173±0.000897 | 0.783679±0.002009 | 0.758820±0.002593 | 0.760577±0.001027 | 0.760933±0.002097 |
| comp2 | 0.755254±0.003774 | 0.765294±0.002046 | 0.738470±0.002344 | 0.768697±0.000757 | 0.781271±0.001829 | 0.755258±0.003684 | 0.760707±0.001077 | 0.762055±0.001042 |
| comp1_comp2 | 0.753426±0.003264 | 0.764899±0.002122 | 0.737321±0.003466 | 0.767022±0.000516 | 0.782051±0.004525 | 0.757041±0.001988 | 0.760294±0.002164 | 0.761268±0.001010 |
| comp123 | 0.754769±0.000100 | 0.763184±0.002849 | 0.738621±0.002503 | 0.768582±0.004654 | 0.781929±0.003523 | 0.757507±0.001925 | 0.760765±0.002034 | 0.760710±0.004167 |

#### §7.2.4 ml_r10（downsample_ratio = 10.0；实测 3 seeds：2022, 2023, 2024）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.750572 | 0.747176 | 0.754583 | 0.721501 | 0.752244 | 0.775427 | 0.743933 | 170 |
| baseline | 2023 | 0.748596 | 0.740672 | 0.754676 | 0.726492 | 0.757687 | 0.773469 | 0.738977 | 145 |
| baseline | 2024 | 0.743077 | 0.741214 | 0.749462 | 0.718966 | 0.739679 | 0.768691 | 0.742699 | 185 |
| fixed073 | 2022 | 0.751199 | 0.746786 | 0.754350 | 0.722212 | 0.751888 | 0.775427 | 0.744185 | 160 |
| fixed073 | 2023 | 0.748383 | 0.740399 | 0.754588 | 0.726771 | 0.758482 | 0.773338 | 0.739417 | 140 |
| fixed073 | 2024 | 0.742171 | 0.741224 | 0.748756 | 0.719085 | 0.739356 | 0.768064 | 0.742877 | 180 |
| comp1 | 2022 | 0.754979 | 0.749459 | 0.752927 | 0.732141 | 0.759948 | 0.775568 | 0.753627 | 295 |
| comp1 | 2023 | 0.747472 | 0.741937 | 0.744108 | 0.737783 | 0.751954 | 0.767461 | 0.755022 | 315 |
| comp1 | 2024 | 0.753730 | 0.750818 | 0.751136 | 0.733238 | 0.758082 | 0.774556 | 0.752760 | 270 |
| comp2 | 2022 | 0.750370 | 0.739450 | 0.753357 | 0.726221 | 0.746449 | 0.776486 | 0.751094 | 190 |
| comp2 | 2023 | 0.748742 | 0.744433 | 0.754923 | 0.724727 | 0.753687 | 0.775948 | 0.745229 | 175 |
| comp2 | 2024 | 0.745783 | 0.741628 | 0.749663 | 0.720502 | 0.741952 | 0.770633 | 0.742034 | 220 |
| comp1_comp2 | 2022 | 0.755388 | 0.749367 | 0.751316 | 0.732890 | 0.760366 | 0.777541 | 0.754807 | 350 |
| comp1_comp2 | 2023 | 0.749822 | 0.741979 | 0.744799 | 0.738194 | 0.750548 | 0.769467 | 0.755424 | 320 |
| comp1_comp2 | 2024 | 0.753759 | 0.752687 | 0.748471 | 0.736135 | 0.760340 | 0.771866 | 0.755587 | 300 |
| comp123 | 2022 | 0.752187 | 0.750721 | 0.747216 | 0.731515 | 0.759108 | 0.772911 | 0.750151 | 340 |
| comp123 | 2023 | 0.754572 | 0.746067 | 0.749443 | 0.738687 | 0.756075 | 0.773672 | 0.755809 | 330 |
| comp123 | 2024 | 0.754379 | 0.751929 | 0.751293 | 0.734752 | 0.758212 | 0.773801 | 0.753220 | 320 |

##### ml_r10 每方法均值 ± std（ddof = 0，n = 3）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.743021±0.002947 | 0.752907±0.002436 | 0.722320±0.003127 | 0.749870±0.007541 | 0.772529±0.002829 | 0.741870±0.002107 | 0.747086±0.002577 | 0.747415±0.003172 |
| fixed073 | 0.742803±0.002836 | 0.752565±0.002695 | 0.722689±0.003156 | 0.749909±0.007933 | 0.772276±0.003098 | 0.742160±0.002012 | 0.747067±0.002718 | 0.747251±0.003772 |
| comp1 | 0.747405±0.003906 | 0.749390±0.003806 | 0.734387±0.002443 | 0.756661±0.003415 | 0.772528±0.003607 | 0.753803±0.000932 | 0.752362±0.001887 | 0.752060±0.003284 |
| comp2 | 0.741837±0.002040 | 0.752648±0.002205 | 0.723817±0.002422 | 0.747363±0.004834 | 0.774356±0.002641 | 0.746119±0.003752 | 0.747690±0.002359 | 0.748298±0.001899 |
| comp1_comp2 | 0.748011±0.004475 | 0.748195±0.002668 | 0.735740±0.002183 | 0.757085±0.004622 | 0.772958±0.003385 | 0.755273±0.000336 | 0.752877±0.001988 | 0.752989±0.002337 |
| comp123 | 0.749572±0.002527 | 0.749317±0.001667 | 0.734985±0.002933 | 0.757798±0.001272 | 0.773461±0.000393 | 0.753060±0.002313 | 0.753032±0.000809 | 0.753713±0.001082 |

##### ml_r10 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.743021±0.003609 | 0.752907±0.002984 | 0.722320±0.003829 | 0.749870±0.009236 | 0.772529±0.003465 | 0.741870±0.002580 | 0.747086±0.003156 | 0.747415±0.003884 |
| fixed073 | 0.742803±0.003474 | 0.752565±0.003301 | 0.722689±0.003865 | 0.749909±0.009715 | 0.772276±0.003795 | 0.742160±0.002464 | 0.747067±0.003329 | 0.747251±0.004619 |
| comp1 | 0.747405±0.004784 | 0.749390±0.004661 | 0.734387±0.002991 | 0.756661±0.004182 | 0.772528±0.004418 | 0.753803±0.001141 | 0.752362±0.002311 | 0.752060±0.004022 |
| comp2 | 0.741837±0.002498 | 0.752648±0.002701 | 0.723817±0.002966 | 0.747363±0.005921 | 0.774356±0.003235 | 0.746119±0.004595 | 0.747690±0.002889 | 0.748298±0.002325 |
| comp1_comp2 | 0.748011±0.005481 | 0.748195±0.003267 | 0.735740±0.002674 | 0.757085±0.005661 | 0.772958±0.004146 | 0.755273±0.000411 | 0.752877±0.002434 | 0.752989±0.002862 |
| comp123 | 0.749572±0.003095 | 0.749317±0.002041 | 0.734985±0.003592 | 0.757798±0.001558 | 0.773461±0.000481 | 0.753060±0.002832 | 0.753032±0.000991 | 0.753713±0.001325 |

#### §7.2.5 ml_r30（downsample_ratio = 30.0；实测 5 seeds：2022, 2023, 2024, 2025, 2026）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.721162 | 0.721819 | 0.730750 | 0.709671 | 0.725771 | 0.739577 | 0.724820 | 86 |
| baseline | 2023 | 0.726003 | 0.714387 | 0.734635 | 0.714763 | 0.729293 | 0.741646 | 0.726925 | 70 |
| baseline | 2024 | 0.719956 | 0.721153 | 0.730431 | 0.709772 | 0.727848 | 0.736689 | 0.728754 | 75 |
| baseline | 2025 | 0.725296 | 0.728034 | 0.739048 | 0.714909 | 0.722526 | 0.753942 | 0.736531 | 70 |
| baseline | 2026 | 0.720152 | 0.718215 | 0.724676 | 0.707862 | 0.720538 | 0.734180 | 0.733951 | 77 |
| fixed073 | 2022 | 0.713686 | 0.715921 | 0.724135 | 0.709940 | 0.714244 | 0.724642 | 0.727367 | 81 |
| fixed073 | 2023 | 0.727030 | 0.714114 | 0.734235 | 0.715648 | 0.728156 | 0.742041 | 0.728087 | 64 |
| fixed073 | 2024 | 0.720194 | 0.720686 | 0.730808 | 0.709902 | 0.728063 | 0.737134 | 0.728934 | 73 |
| fixed073 | 2025 | 0.725898 | 0.727074 | 0.737786 | 0.715059 | 0.722497 | 0.752818 | 0.737084 | 71 |
| fixed073 | 2026 | 0.717896 | 0.721566 | 0.729682 | 0.710574 | 0.721880 | 0.743546 | 0.738348 | 74 |
| comp1 | 2022 | 0.740356 | 0.731347 | 0.747408 | 0.728380 | 0.739416 | 0.760123 | 0.750822 | 106 |
| comp1 | 2023 | 0.743523 | 0.733681 | 0.749955 | 0.728198 | 0.739972 | 0.758945 | 0.748342 | 105 |
| comp1 | 2024 | 0.739445 | 0.735037 | 0.744505 | 0.727574 | 0.739622 | 0.753654 | 0.750237 | 120 |
| comp1 | 2025 | 0.743330 | 0.733745 | 0.750676 | 0.729214 | 0.744332 | 0.761627 | 0.751305 | 99 |
| comp1 | 2026 | 0.742558 | 0.736470 | 0.745981 | 0.727866 | 0.744212 | 0.760384 | 0.748018 | 100 |
| comp2 | 2022 | 0.718150 | 0.713076 | 0.733672 | 0.713621 | 0.716087 | 0.734727 | 0.736083 | 90 |
| comp2 | 2023 | 0.724863 | 0.713348 | 0.734894 | 0.717425 | 0.723680 | 0.733601 | 0.737153 | 83 |
| comp2 | 2024 | 0.723732 | 0.725169 | 0.728433 | 0.710204 | 0.734556 | 0.738771 | 0.729247 | 88 |
| comp2 | 2025 | 0.728754 | 0.731222 | 0.735934 | 0.710986 | 0.734251 | 0.748578 | 0.730493 | 87 |
| comp2 | 2026 | 0.720217 | 0.712494 | 0.724739 | 0.717765 | 0.721399 | 0.736639 | 0.739107 | 94 |
| comp1_comp2 | 2022 | 0.741589 | 0.733252 | 0.749273 | 0.727944 | 0.745288 | 0.759243 | 0.749030 | 105 |
| comp1_comp2 | 2023 | 0.745195 | 0.738532 | 0.748847 | 0.723053 | 0.748315 | 0.761658 | 0.746461 | 106 |
| comp1_comp2 | 2024 | 0.742578 | 0.735193 | 0.750090 | 0.726458 | 0.747425 | 0.758952 | 0.748643 | 106 |
| comp1_comp2 | 2025 | 0.746131 | 0.734997 | 0.750828 | 0.728333 | 0.744069 | 0.765290 | 0.750044 | 99 |
| comp1_comp2 | 2026 | 0.741909 | 0.734871 | 0.749099 | 0.728136 | 0.742826 | 0.757629 | 0.750366 | 106 |
| comp123 | 2022 | 0.742750 | 0.735281 | 0.746702 | 0.725823 | 0.741036 | 0.755779 | 0.748742 | 119 |
| comp123 | 2023 | 0.744869 | 0.736656 | 0.748108 | 0.722681 | 0.747513 | 0.760733 | 0.745462 | 111 |
| comp123 | 2024 | 0.747254 | 0.738164 | 0.749597 | 0.728171 | 0.752268 | 0.764107 | 0.751215 | 112 |
| comp123 | 2025 | 0.746861 | 0.733541 | 0.750564 | 0.729702 | 0.741551 | 0.768466 | 0.749033 | 104 |
| comp123 | 2026 | 0.743346 | 0.737056 | 0.744828 | 0.725548 | 0.743353 | 0.756651 | 0.748626 | 111 |

##### ml_r30 每方法均值 ± std（ddof = 0，n = 5）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.720722±0.004500 | 0.731908±0.004780 | 0.711395±0.002891 | 0.725195±0.003255 | 0.741207±0.006853 | 0.730196±0.004380 | 0.726771±0.003104 | 0.722514±0.002603 |
| fixed073 | 0.719872±0.004564 | 0.731329±0.004580 | 0.712225±0.002573 | 0.722968±0.005107 | 0.740036±0.009221 | 0.731964±0.004739 | 0.726399±0.004084 | 0.720941±0.004982 |
| comp1 | 0.734056±0.001695 | 0.747705±0.002332 | 0.728246±0.000557 | 0.741511±0.002262 | 0.758947±0.002780 | 0.749745±0.001326 | 0.743368±0.001111 | 0.741842±0.001644 |
| comp2 | 0.719062±0.007704 | 0.731534±0.004266 | 0.714000±0.003148 | 0.725995±0.007295 | 0.738463±0.005354 | 0.734417±0.003857 | 0.727245±0.002576 | 0.723143±0.003695 |
| comp1_comp2 | 0.735369±0.001726 | 0.749627±0.000731 | 0.726785±0.001980 | 0.745585±0.002041 | 0.760554±0.002701 | 0.748909±0.001378 | 0.744472±0.000616 | 0.743480±0.001834 |
| comp123 | 0.736140±0.001593 | 0.747960±0.002043 | 0.726385±0.002405 | 0.745144±0.004227 | 0.761147±0.004723 | 0.748616±0.001838 | 0.744232±0.001877 | 0.745016±0.001809 |

##### ml_r30 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.720722±0.005032 | 0.731908±0.005344 | 0.711395±0.003232 | 0.725195±0.003639 | 0.741207±0.007662 | 0.730196±0.004897 | 0.726771±0.003471 | 0.722514±0.002910 |
| fixed073 | 0.719872±0.005103 | 0.731329±0.005120 | 0.712225±0.002876 | 0.722968±0.005709 | 0.740036±0.010309 | 0.731964±0.005299 | 0.726399±0.004566 | 0.720941±0.005570 |
| comp1 | 0.734056±0.001895 | 0.747705±0.002607 | 0.728246±0.000623 | 0.741511±0.002529 | 0.758947±0.003108 | 0.749745±0.001482 | 0.743368±0.001242 | 0.741842±0.001838 |
| comp2 | 0.719062±0.008614 | 0.731534±0.004770 | 0.714000±0.003519 | 0.725995±0.008156 | 0.738463±0.005986 | 0.734417±0.004312 | 0.727245±0.002880 | 0.723143±0.004132 |
| comp1_comp2 | 0.735369±0.001930 | 0.749627±0.000817 | 0.726785±0.002214 | 0.745585±0.002282 | 0.760554±0.003020 | 0.748909±0.001540 | 0.744472±0.000689 | 0.743480±0.002051 |
| comp123 | 0.736140±0.001781 | 0.747960±0.002284 | 0.726385±0.002689 | 0.745144±0.004726 | 0.761147±0.005280 | 0.748616±0.002055 | 0.744232±0.002099 | 0.745016±0.002022 |

#### §7.2.6 ml_r100（downsample_ratio = 100.0；实测 5 seeds：2022, 2023, 2024, 2025, 2026）

| 方法 | seed | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2022 | 0.671146 | 0.660522 | 0.684064 | 0.702835 | 0.676197 | 0.694350 | 0.718106 | 75 |
| baseline | 2023 | 0.691279 | 0.679975 | 0.700434 | 0.707258 | 0.682032 | 0.709805 | 0.714534 | 63 |
| baseline | 2024 | 0.667750 | 0.652542 | 0.684987 | 0.711669 | 0.661742 | 0.699297 | 0.734662 | 75 |
| baseline | 2025 | 0.682254 | 0.678827 | 0.690110 | 0.697751 | 0.674793 | 0.710006 | 0.723307 | 66 |
| baseline | 2026 | 0.685212 | 0.674386 | 0.704299 | 0.704199 | 0.685137 | 0.719375 | 0.722529 | 67 |
| fixed073 | 2022 | 0.672216 | 0.661056 | 0.686084 | 0.702926 | 0.677605 | 0.696848 | 0.717285 | 70 |
| fixed073 | 2023 | 0.690527 | 0.678717 | 0.699919 | 0.707181 | 0.681745 | 0.709635 | 0.714077 | 62 |
| fixed073 | 2024 | 0.667101 | 0.661549 | 0.686615 | 0.703970 | 0.666515 | 0.702851 | 0.724778 | 69 |
| fixed073 | 2025 | 0.682302 | 0.677859 | 0.689922 | 0.697818 | 0.676176 | 0.710054 | 0.722806 | 66 |
| fixed073 | 2026 | 0.681366 | 0.664382 | 0.702556 | 0.700952 | 0.682021 | 0.718858 | 0.719993 | 66 |
| comp1 | 2022 | 0.712460 | 0.691416 | 0.714483 | 0.720506 | 0.701614 | 0.733131 | 0.744397 | 95 |
| comp1 | 2023 | 0.695753 | 0.684494 | 0.691837 | 0.729024 | 0.692294 | 0.706780 | 0.737804 | 108 |
| comp1 | 2024 | 0.712892 | 0.699815 | 0.706207 | 0.721035 | 0.713466 | 0.733213 | 0.742181 | 96 |
| comp1 | 2025 | 0.705814 | 0.673606 | 0.716413 | 0.721203 | 0.692419 | 0.730789 | 0.744127 | 103 |
| comp1 | 2026 | 0.715956 | 0.701391 | 0.713621 | 0.722810 | 0.704294 | 0.738162 | 0.744180 | 118 |
| comp2 | 2022 | 0.664358 | 0.656795 | 0.684994 | 0.700594 | 0.662268 | 0.684949 | 0.723457 | 191 |
| comp2 | 2023 | 0.694345 | 0.675362 | 0.705832 | 0.705251 | 0.678159 | 0.725274 | 0.714937 | 146 |
| comp2 | 2024 | 0.669758 | 0.662145 | 0.683895 | 0.705338 | 0.661328 | 0.702869 | 0.721144 | 148 |
| comp2 | 2025 | 0.672296 | 0.679194 | 0.683511 | 0.704817 | 0.649815 | 0.698057 | 0.723301 | 98 |
| comp2 | 2026 | 0.686287 | 0.666881 | 0.706836 | 0.707719 | 0.685948 | 0.726847 | 0.727915 | 89 |
| comp1_comp2 | 2022 | 0.712650 | 0.687989 | 0.713976 | 0.719504 | 0.699045 | 0.734781 | 0.744072 | 100 |
| comp1_comp2 | 2023 | 0.699976 | 0.695299 | 0.703790 | 0.714594 | 0.698552 | 0.716220 | 0.731250 | 112 |
| comp1_comp2 | 2024 | 0.714689 | 0.697772 | 0.708384 | 0.718904 | 0.706516 | 0.736493 | 0.742551 | 99 |
| comp1_comp2 | 2025 | 0.712143 | 0.691350 | 0.715185 | 0.722396 | 0.707210 | 0.730148 | 0.745220 | 103 |
| comp1_comp2 | 2026 | 0.710528 | 0.695173 | 0.711380 | 0.722717 | 0.699194 | 0.721825 | 0.743938 | 106 |
| comp123 | 2022 | 0.702960 | 0.683354 | 0.711230 | 0.714442 | 0.694757 | 0.723207 | 0.737685 | 119 |
| comp123 | 2023 | 0.705097 | 0.701106 | 0.703115 | 0.716770 | 0.710413 | 0.714858 | 0.737100 | 109 |
| comp123 | 2024 | 0.706316 | 0.693314 | 0.707055 | 0.721686 | 0.703381 | 0.728170 | 0.742013 | 110 |
| comp123 | 2025 | 0.713167 | 0.697267 | 0.705770 | 0.720006 | 0.704774 | 0.724732 | 0.740749 | 112 |
| comp123 | 2026 | 0.707275 | 0.692113 | 0.712375 | 0.720839 | 0.689180 | 0.725291 | 0.743530 | 200 |

##### ml_r100 每方法均值 ± std（ddof = 0，n = 5）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B(test_auc_mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.669250±0.010849 | 0.692779±0.008187 | 0.704742±0.004629 | 0.675980±0.008058 | 0.706567±0.008814 | 0.722628±0.006801 | 0.695324±0.004689 | 0.679528±0.008795 |
| fixed073 | 0.668713±0.007905 | 0.693019±0.006889 | 0.702569±0.003116 | 0.676812±0.005631 | 0.707649±0.007417 | 0.719788±0.003819 | 0.694758±0.003475 | 0.678703±0.008204 |
| comp1 | 0.690144±0.010267 | 0.708512±0.009028 | 0.722916±0.003150 | 0.700817±0.007948 | 0.728415±0.011081 | 0.742538±0.002499 | 0.715557±0.004976 | 0.708575±0.007214 |
| comp2 | 0.668075±0.008253 | 0.693014±0.010892 | 0.704744±0.002310 | 0.667504±0.012901 | 0.707599±0.016182 | 0.722151±0.004226 | 0.693848±0.007080 | 0.677409±0.011141 |
| comp1_comp2 | 0.693517±0.003443 | 0.710543±0.004106 | 0.719623±0.002935 | 0.702103±0.003898 | 0.727893±0.007740 | 0.741406±0.005148 | 0.715848±0.003146 | 0.709997±0.005184 |
| comp123 | 0.693431±0.005948 | 0.707909±0.003443 | 0.718749±0.002722 | 0.700501±0.007563 | 0.723252±0.004494 | 0.740215±0.002474 | 0.714009±0.001819 | 0.706963±0.003421 |

##### ml_r100 同表 ddof = 1 对照

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 口径A再平均 | 口径B |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.669250±0.012129 | 0.692779±0.009153 | 0.704742±0.005175 | 0.675980±0.009009 | 0.706567±0.009855 | 0.722628±0.007603 | 0.695324±0.005242 | 0.679528±0.009834 |
| fixed073 | 0.668713±0.008838 | 0.693019±0.007702 | 0.702569±0.003484 | 0.676812±0.006295 | 0.707649±0.008292 | 0.719788±0.004270 | 0.694758±0.003885 | 0.678703±0.009172 |
| comp1 | 0.690144±0.011479 | 0.708512±0.010094 | 0.722916±0.003521 | 0.700817±0.008887 | 0.728415±0.012389 | 0.742538±0.002794 | 0.715557±0.005564 | 0.708575±0.008066 |
| comp2 | 0.668075±0.009227 | 0.693014±0.012177 | 0.704744±0.002583 | 0.667504±0.014424 | 0.707599±0.018092 | 0.722151±0.004725 | 0.693848±0.007916 | 0.677409±0.012456 |
| comp1_comp2 | 0.693517±0.003850 | 0.710543±0.004591 | 0.719623±0.003282 | 0.702103±0.004358 | 0.727893±0.008654 | 0.741406±0.005756 | 0.715848±0.003518 | 0.709997±0.005796 |
| comp123 | 0.693431±0.006651 | 0.707909±0.003849 | 0.718749±0.003043 | 0.700501±0.008455 | 0.723252±0.005025 | 0.740215±0.002766 | 0.714009±0.002034 | 0.706963±0.003825 |

#### §7.2.7 跨档汇总（口径 A 再平均 / 口径 B，均值±std ddof=0）

| 方法 | ml_orig | ml_r1 | ml_r3 | ml_r10 | ml_r30 | ml_r100 |
|---|---|---|---|---|---|---|
| baseline | 0.769402 / 0.770098 | 0.758786 / 0.759105 | 0.761808 / 0.762987 | 0.747086 / 0.747415 | 0.726771 / 0.722514 | 0.695324 / 0.679528 |
| fixed073 | 0.769587 / 0.770876 | 0.759353 / 0.759217 | 0.762643 / 0.762338 | 0.747067 / 0.747251 | 0.726399 / 0.720941 | 0.694758 / 0.678703 |
| comp1 | 0.772783 / 0.773716 | 0.765140 / 0.765979 | 0.760577 / 0.760933 | 0.752362 / 0.752060 | 0.743368 / 0.741842 | 0.715557 / 0.708575 |
| comp2 | 0.768975 / 0.770108 | 0.759435 / 0.759716 | 0.760707 / 0.762055 | 0.747690 / 0.748298 | 0.727245 / 0.723143 | 0.693848 / 0.677409 |
| comp1_comp2 | 0.772443 / 0.773447 | 0.764531 / 0.764850 | 0.760294 / 0.761268 | 0.752877 / 0.752989 | 0.744472 / 0.743480 | 0.715848 / 0.709997 |
| comp123 | 0.771078 / 0.772148 | 0.764362 / 0.765574 | 0.760765 / 0.760710 | 0.753032 / 0.753713 | 0.744232 / 0.745016 | 0.714009 / 0.706963 |

> 每格式样为「口径 A 再平均 / 口径 B」。seeds 数不一致（ml_orig / ml_r30 / ml_r100 为 5，ml_r1 / ml_r3 / ml_r10 为 3），跨档比较时须带上该口径差异。

### §7.3 与 Oracle 对比的 Δ_oracle 表（`ML_P5_REPORT.md` §3）

> Δ_oracle = 主矩阵该 (d,t) 的口径 A 均值 − 同档 oracle 对应 (域, 任务) 的 5 seed 均值。oracle 为单域独立训练参考：Δ > 0 表示联合训练在该 (d,t) 上高于单域 oracle，Δ < 0 表示低于。本节只列实算数值，不作方向性解读。

#### oracle 基准（P5_oracle.csv 实算，均值±std ddof=0）

| config | n/域 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|
| ml_orig | 5 | 0.756866±0.002350 | 0.770364±0.003253 | 0.765861±0.001913 | 0.789030±0.000928 | 0.730768±0.003280 | 0.744937±0.003120 |
| ml_r30 | 5 | 0.756866±0.002350 | 0.770364±0.003253 | 0.765861±0.001913 | 0.789030±0.000928 | 0.730768±0.003280 | 0.744937±0.003120 |
| ml_r100 | 5 | 0.756866±0.002350 | 0.770364±0.003253 | 0.765861±0.001913 | 0.789030±0.000928 | 0.730768±0.003280 | 0.744937±0.003120 |

> ML oracle 仅覆盖 `ml_orig` / `ml_r30` / `ml_r100` 三档（共 45 行 OK = 3 档 × 3 域 × 5 seeds），`ml_r1` / `ml_r3` / `ml_r10` 无 oracle，故这三档不出 Δ 表。

> **重要事实（三档 oracle 数值逐位相同）**：上表三行完全一致，并非笔误。核验：15 个 `(domain, seed)` 组合上三档的 `auc_d0t0` / `auc_d0t1` 逐位相同（15/15 相同，0 差异）。原因是 `p4_run.py` 的单域路径与下采样的交互 —— `_load_data()`（`p4_run.py:476`）在 `--single_domain d` 下只取域 d 的数据、`domain_num = 1`，随后 `downsample_train()`（`p4_run.py:314`）按域计数排序得到 `counts`，仅对 `counts[1:]`（即非最大域）执行 `shrink`；单域数据只有一个域，`counts[1:]` 为空，**下采样从不执行**。因此三档单域 oracle 训练的是同一份未裁剪的单域训练集，只是缓存键（`p4_run.py:447` 含 `downsample_ratio`）不同而各自重算了一遍（三份不同 pkl：`data_e557b0124f61` / `data_2567db2d5177` / `data_7902a310de64`，训练 wall 179.6 / 181.8 / 179.0s，peak 均 730 MiB）。
>
> 该行为的直接后果：**oracle 基准不随下采样档位变化**，它衡量的是「单域全量数据独立训练」的水平，而主矩阵在 r30 / r100 档的训练集已被大幅裁剪。因此 §7.3 中 r30 / r100 的 Δ_oracle 同时混入了「联合训练 vs 单域训练」与「裁剪后训练集 vs 全量训练集」两种差异，不是纯粹的迁移效应度量。是否需要重定义 r30 / r100 档的 oracle（例如让单域 oracle 也按同比例裁剪），提交 Reviewer 裁定，本报告不代为判断。

#### Δ_oracle：ml_orig（干净口径 —— 主矩阵与 oracle 两侧均未下采样）

> 本档 `downsample_ratio = 0`，主矩阵与单域 oracle 训练的都是未裁剪数据，Δ 为纯迁移效应度量，可作 Pareto ≥ 0 判据。

| 方法 | d0t0 Δ | d0t1 Δ | d1t0 Δ | d1t1 Δ | d2t0 Δ | d2t1 Δ |
|---|---|---|---|---|---|---|
| baseline | 0.008995 | 0.004115 | -0.021511 | -0.009258 | 0.061500 | 0.014744 |
| fixed073 | 0.009023 | 0.004886 | -0.021370 | -0.009521 | 0.061966 | 0.014709 |
| comp1 | 0.009582 | 0.006706 | -0.013261 | -0.007506 | 0.062596 | 0.020754 |
| comp2 | 0.007519 | 0.005027 | -0.021929 | -0.009955 | 0.061897 | 0.013464 |
| comp1_comp2 | 0.008128 | 0.006740 | -0.013519 | -0.009126 | 0.062771 | 0.021836 |
| comp123 | 0.007270 | 0.005452 | -0.016284 | -0.009668 | 0.062300 | 0.019572 |

#### Δ_oracle：ml_r30（**混合口径** —— 基准为单域全量训练参考）

> 基准为**单域全量训练参考**：oracle 侧未下采样（原因见上文 §7.3 说明），主矩阵侧训练集按 30:1 裁剪。故本表 Δ 的绝对水平**混入了「裁剪 vs 全量」的训练集规模效应**，不是纯迁移效应，**不得作为 Pareto ≥ 0 的判据**。同档内方法间的 Δ 差值不受影响（oracle 常数逐项抵消）。

| 方法 | d0t0 Δ | d0t1 Δ | d1t0 Δ | d1t1 Δ | d2t0 Δ | d2t1 Δ |
|---|---|---|---|---|---|---|
| baseline | -0.036145 | -0.038456 | -0.054465 | -0.063834 | 0.010439 | -0.014741 |
| fixed073 | -0.036994 | -0.039034 | -0.053636 | -0.066062 | 0.009268 | -0.012973 |
| comp1 | -0.022810 | -0.022659 | -0.037614 | -0.047519 | 0.028178 | 0.004807 |
| comp2 | -0.037804 | -0.038829 | -0.051860 | -0.063035 | 0.007695 | -0.010521 |
| comp1_comp2 | -0.021497 | -0.020736 | -0.039076 | -0.043445 | 0.029786 | 0.003971 |
| comp123 | -0.020727 | -0.022404 | -0.039476 | -0.043885 | 0.030379 | 0.003678 |

#### Δ_oracle：ml_r100（**混合口径** —— 基准为单域全量训练参考）

> 同 ml_r30：基准为**单域全量训练参考**，oracle 侧未下采样而主矩阵侧按 100:1 裁剪，Δ 绝对水平**混入训练集规模效应**，**不得作为 Pareto ≥ 0 的判据**；同档内方法间差值不受影响。

| 方法 | d0t0 Δ | d0t1 Δ | d1t0 Δ | d1t1 Δ | d2t0 Δ | d2t1 Δ |
|---|---|---|---|---|---|---|
| baseline | -0.087616 | -0.077585 | -0.061118 | -0.113049 | -0.024202 | -0.022310 |
| fixed073 | -0.088154 | -0.077344 | -0.063291 | -0.112217 | -0.023119 | -0.025150 |
| comp1 | -0.066722 | -0.061851 | -0.042945 | -0.088212 | -0.002353 | -0.002400 |
| comp2 | -0.088791 | -0.077350 | -0.061117 | -0.121526 | -0.023169 | -0.022787 |
| comp1_comp2 | -0.063350 | -0.059821 | -0.046238 | -0.086926 | -0.002875 | -0.003531 |
| comp123 | -0.063435 | -0.062455 | -0.047112 | -0.088529 | -0.007517 | -0.004722 |

### §7.4 客观异常记录（`ML_P5_REPORT.md` §4）

- **行数清点（P5_raw.csv，config 以 `ml_` 开头）**：OK 合计 144 行 —— ml_orig 30、ml_r1 18、ml_r3 18、ml_r10 18、ml_r30 30、ml_r100 30。ML oracle（P5_oracle.csv）45 行 OK，0 FAILED。
- **历史 FAILED 行**：2 条，按规程保留未删：
  - `ml_orig fixed073 2026` wall=3s，note = `AttributeError: 'P4Runner' object has no attribute '_load_cache'`
  - `ml_orig comp1 2022` wall=3s，note = `AttributeError: 'P4Runner' object has no attribute '_load_cache'`
  这两条均有同键的后续 OK 行，本报告对同一 `(config, method, seed)` 取 OK 行优先，故不影响任何统计。
- **seeds 数不对称**：`ml_orig` / `ml_r30` / `ml_r100` 为 5 seeds（2022–2026），`ml_r1` / `ml_r3` / `ml_r10` 为 3 seeds（2022–2024）。这是既有计划内的降级，非本轮丢格；跨档比较与任何跨档趋势陈述都必须带此口径差异。
- **oracle 覆盖不对称**：oracle 只跑了 `ml_orig` / `ml_r30` / `ml_r100` 三档，`ml_r1` / `ml_r3` / `ml_r10` 无 oracle 基准，因此 §7.3 只有三档 Δ 表，梯度上的 Δ 序列存在空档。
- **执行窗口不齐**：各档产出于不同日期窗口（`ml_r3` 的 4 格与 `ml_orig` / `ml_r1` / `ml_r30` / `ml_r100` 属早期窗口；`ml_r3` 余 14 格、`ml_r10` 全部、oracle 全部属 9/8 窗口）。9/8 开跑前的漂移探针在 `ml_orig` baseline 2022 上实测漂移 0.000000（§7.1），该探针只覆盖 ml_orig 一格，不逐位担保其他档位。
- **会话中断**：Runner 会话在 9/8 多次因 API 529 中断，训练由编排器独立托管未受影响，本轮 77 格全部一次通过、0 FAILED、无重投；断点台账见 `experiments/logs/stage3_checkpoint.md`。
- **数据载体 reconcile**：`ml_r30` / `ml_r100` 主矩阵行确实存在于 `P5_raw.csv`（各 30 行 OK、5 seeds、6 方法齐全），此前交接中「r30/r100 只有 oracle、无主矩阵」的说法有误，已在本报告纠正。另有 `experiments/results/p5/P5_main.csv` 为同批结果的长表展开（每 run 6 行 (d,t)，含 `mean_auc` 列），本报告一律以 `P5_raw.csv` 宽表为唯一数值源，未混用两表。`P5_main.csv` 的 config 覆盖为 kuairand / ml_orig / ml_r1 / ml_r3(部分) / ml_r30 / ml_r100，缺 `ml_r10`，故不适合作梯度扫尾的统一数值源。
- **三档 oracle 数值逐位相同**：`ml_orig` / `ml_r30` / `ml_r100` 的 45 行 oracle 在 15 个 `(domain, seed)` 组合上逐位一致（15/15 相同）。根因为 `p4_run.py` 单域路径下 `downsample_train()` 的 `counts[1:]` 为空、下采样不生效（详见 §7.3 的说明与代码行号）。这是既有代码的既定行为，本轮**未修改任何代码**；由此带来的 r30 / r100 档 Δ_oracle 口径混淆（迁移效应与训练集规模效应叠加）提交 Reviewer 裁定。
- **best_epoch 未入 CSV**：编排器 `RAW_FIELDS` 无该字段（既有 schema），每 run 的 `validation: best auc` 与逐 epoch 曲线完整保留在单 run 日志中，可按需回溯。

### §7.5 配对 t 检验汇总（与 exp.md §6.2 衔接）

> 全部 vs baseline，**按 seed 配对**，**双侧**。显著性 + 效应量阈值：p<0.05（粗体为显著正效应），d>2 为大效应量。

| 档 | comp1 | comp1&2 | comp1&2&3 | comp2 | fixed073 |
|---|---|---|---|---|---|
| ml_orig | **+0.0034, p=0.0278, d=1.51** | +0.0030, p=0.0962 | +0.0017, p=0.137 | −0.0004, p=0.59 | +0.0002, p=0.77 |
| ml_r1 | **+0.0064, p=0.0162, d=4.48** | +0.0057, p=0.0257 | +0.0056, p=0.096 | +0.0006, p=0.27 | +0.0006, p=0.31 |
| ml_r3 | **−0.0012, p=0.0327, d=−3.12** | −0.0015, p=0.539 | −0.0010, p=0.316 | −0.0011, p=0.123 | +0.0008, p=0.158 |
| ml_r10 | +0.0053, p=0.178, d=1.18 | +0.0058, p=0.166 | +0.0059, p=0.122 | +0.0006, p=0.317 | −0.0000, p=0.883 |
| ml_r30 | **+0.0166, p=0.0002, d=5.80** | **+0.0177, p=0.0001, d=6.32** | +0.0175, p=0.0003 | +0.0005, p=0.50 | −0.0004, p=0.83 |
| ml_r100 | **+0.0202, p=0.0061, d=2.37** | **+0.0205, p=0.0039, d=2.68** | **+0.0187, p=0.0013** | −0.0015, p=0.40 | −0.0006, p=0.51 |

> **与 exp.md §6.2 既有四档拼合的 comp1 六档梯度（均值 Δ，Reviewer 复核）**：1:1 +0.006354 / 1.9:1 +0.003381 / 3:1 −0.001231 / 10:1 +0.005277 / 30:1 +0.016598 / 100:1 +0.020233（§9 FINAL 与 Reviewer 仲裁复用此 6 数序列）。

### §7.6 核心发现（`ML_P5_REPORT.md` Reviewer 签署第三节）

1. **组件 1 的增益在强不均衡档（≥30:1）大幅放大**：+0.0166（30:1，p=0.0002，d=5.80）、+0.0202（100:1，p=0.0061，d=2.37），r30 档全部 6 项 (d,t) 一致提升（+0.013~+0.020）。**但"随不均衡度单调放大"在全六档序列上不成立**（Reviewer 2026-09-08 仲裁）：1:1 +0.0064 → 1.9:1 +0.0034 → **3:1 −0.0012（名义显著负凹陷，3/3 seeds 同号，幅度≈噪声底，n=3 效力不足，定性为待复核异常而非机理证据）** → 10:1 +0.0053（不显著）→ 30:1/100:1 大幅显著。
2. **组件 2（域条件课程）单独从不显著**（所有档位 |diff|≤0.0015）；与 comp1 组合后不叠加增益（与 comp1 持平），但在 r100 档**收窄域间差距**（gap：baseline 0.033583 → comp1&2 0.032620 → comp123 0.031064；comp1 单独反升至 0.036148）。（2026-09-08 终审修正：旧文"comp123 到 0.0285"无台账背书。）
3. **组件 3（arch 平滑）无稳定贡献**（comp123 − comp1&2 六档差 −0.0018~+0.0005，r3/r10 档为正值；2026-09-08 终审修正：旧文"低 0.001~0.003"仅在 orig/r100 成立）——与 §3.1 P1 D0.2 / §4 P2"架构搜索本就无效"一致。
4. fixed-0.7311 在所有档位与 baseline 无差异（复证 §4 P2）。
5. 域间差距（max−min per-domain AUC）：comp1 在 ml_orig 略降（0.0125 vs 0.0139）；r30 档 comp1 的 gap 略升（0.0195 vs 0.0174）——增益是大域小域共享的，不是单纯补小域。
6. **诚实负面记录**：comp2 全线不显著；comp123 在 r30/r100 略低于 comp1&2；ml_r3 档 comp1 出现名义显著负凹陷（见第 1 条）；kuairand 上大域负迁移仅收窄未逆转（见 §8）。
7. **oracle 口径警示（Reviewer 裁决四）**：ML 三档 oracle 数值逐位相同——`p4_run.py` 单域分支不执行下采样，oracle 恒为"单域全量训练"。ml_orig 档 Δ_oracle 干净可用；**r30/r100 的 Δ_oracle 为混合口径（混入裁剪 vs 全量规模效应），不得作为 Pareto ≥0 判据**；同档内方法间差值不受影响（oracle 常数抵消）。纯迁移口径的备用修复规约见 §9 FINAL 终审签署第四节。

### §7.7 环境敏感性（重要！影响跨会话可比性）

- 在会话中段发现：**绝对数值依赖 GPU 桌面负载状态**（同一 seed 的 epoch-0 验证 AUC 在两个环境窗口分别为 0.71866 与 0.71232，各自窗口内逐位可复现）。原因推断：桌面应用占用的显存改变分配器布局 → cuBLASLt 按指针对齐选核 → 浮点归约顺序变化。`CUBLAS_WORKSPACE_CONFIG=:4096:8` 未消除。
- 对策与结论：P0–P2 的绝对数字是其当时窗口的记录；§7 P5 全矩阵在同一窗口内完成、**方法间对比内部自洽**；跨窗口差异（~0.002 AUC）远小于 r30/r100 档的效应（+0.017~+0.020），主结论稳固。交接后若跨天续跑，建议先重跑 ml_orig baseline 2022 一格并与 `P5_raw.csv` 对比以检测窗口漂移。

### §7.8 效率（fit 墙钟 / 峰值显存，同档内可比）

- ml_orig：baseline 272s/942MiB；comp1 413s/1158MiB（+52%/+216MiB）；comp2 231s/941MiB；comp1&2 338s。
- ml_r30：baseline 72s；comp1 102s；comp1&2 101s。ml_r100：baseline 66s；comp1 100s。
- KuaiRand：baseline ~975s；comp1 ~2787s（D 次反传在 2M 行上的代价）。

### §7.9 Reviewer 审计与不均衡机理仲裁结论（来源：`ML_P5_REPORT.md` 末段）

**审计日期：2026-09-08　署名：Reviewer**

**一、真实性核验：通过。** P5_raw.csv 行数声明属实（ML 主矩阵 OK 144 行 = ml_orig 30 / ml_r1 18 / ml_r3 18 / ml_r10 18 / ml_r30 30 / ml_r100 30；历史 FAILED 恰 2 条，note 与 §7.4 一致，均有同键 OK 行）；P5_oracle.csv 60 行全 OK（kuairand 15 + ML 45）。§7.2 全部 36 个（档×方法）格子的六项 per_dt 均值、口径 A/B 及双 ddof std 经全量复算 0 不符；§7.3 oracle 基准与三档 Δ 表 108 格全量复算 0 不符；§7.1/§7.4 各项与编排日志（`queued 77 tasks`、0 FAILED、`DUAL_ORCHESTRATOR DONE`）、断点台账、漂移探针日志（漂移 0.000000）逐条相符。

**二、新增档位显著性（vs baseline，口径 A，双侧配对 t，n=3，df=2）。** ml_r3：comp1 **−0.001231（p=0.032670，d=−3.12）**，fixed073 +0.000835（p=0.158437），comp2 −0.001101（p=0.123442），comp1_comp2 −0.001515（p=0.539285），comp123 −0.001043（p=0.316428）。ml_r10：comp1 +0.005277（p=0.178438，d=+1.18），fixed073 −0.000019（p=0.882995），comp2 +0.000604（p=0.317018），comp1_comp2 +0.005791（p=0.165664），comp123 +0.005946（p=0.122405）。与 §7.5 既有四档拼合的 comp1 六档梯度（均值 Δ）：1:1 +0.006354 / 1.9:1 +0.003381 / 3:1 −0.001231 / 10:1 +0.005277 / 30:1 +0.016598 / 100:1 +0.020233。

**三、机理仲裁："组件 1 增益随不均衡度单调放大"在全六档序列上不成立。** 3:1 档出现名义显著的负向凹陷（3/3 seeds 同号为负，幅度 ≈0.0012 ≈ 噪声底，n=3 效力不足），10:1 档回正但不显著且低于 1:1 档；两档均落在 r1→r30 log 插值包络（≈+0.0097/+0.0133）之下。裁定改写为：组件 1 的增益在强不均衡档（≥30:1）大幅放大且高度显著（p=0.000204/0.006084，d=5.80/2.37）；1:1–10:1 区间非单调，3:1 凹陷定性为待复核异常、不作机理证据。"大域保护"命题在 MovieLens 不适用其原形式（最大域仅 40%，无 KuaiRand 式 89% 极端大域），但"相对最大域 d1 负迁移收窄"同向复现于全部三个 oracle 档（comp1 vs baseline 收窄量：orig +0.008250/+0.001752，r30 +0.016851/+0.016315，r100 +0.018173/+0.024837，t0/t1），且为同档内方法间对比，不受 oracle 口径缺陷影响，予以采信。

**四、oracle 口径缺陷裁决：不需重跑。** 根因核验属实（`p4_run.py:476` 单域分支 + `p4_run.py:314` 的 `counts[1:]` 为空，下采样不执行；缓存键 `p4_run.py:447` 含 ratio 故三档各自重算同一份全量单域数据）。裁定：(a) r30/r100 的 Δ_oracle 绝对水平混入"裁剪 vs 全量"规模效应，**不得作为 Pareto ≥0 判据**；ml_orig 档两侧均未下采样，其 Δ 表干净可用；同档内方法间差值 oracle 常数抵消，不受影响。(b) 本阶段全部核心结论均为矩阵内部对比，现有证据足够，不重跑 30 格。(c) §7.3 的 r30/r100 Δ 表须标注基准为"单域全量训练参考（混合口径）"；若未来需纯迁移效应口径，最小改动为：`_load_data()` 单域分支先全量加载、经 `downsample_train()` 同参数裁剪后再掩码取域 d，缓存键加版本标志，重跑 ml_r30/ml_r100 oracle 共 30 格。

**五、其余事项。** seeds 不对称（3 vs 5）：跨档趋势只许用均值 Δ，p 值跨档不可比（n=3 档 t_crit=4.303，r10 的 +0.005277 不显著不得解读为"无效"）。ml_r1/r3/r10 无 oracle：该三档禁止任何 Δ_oracle/Pareto 陈述。执行窗口不齐：r3 为跨窗混合矩阵（baseline×3 与 fixed073 2022 属早期窗口），漂移探针 0.000000 支持两窗等价但仅覆盖 ml_orig 一格；r3 凹陷 3 seeds 同号、窗口伪影可能性低，保留为注意事项。§7.4 异常记录八条全部与台账/日志实情相符。

**签署结论：本报告数据层与异常记录核验通过，准予签署；机理主结论按第三节改写后生效，oracle 口径按第四节受限使用。—— Reviewer，2026-09-08**

---

## §8 P5 KuaiRand 主实验（✅ 已完成，2026-09-07/08；合并 `KUAI_STAGE1_ORACLE.md` + `KUAI_P5_MAIN_REPORT.md` 全文）

> 数据源：`experiments/results/p5/P5_raw.csv`（主矩阵）、`experiments/results/p5/P5_oracle.csv`（单域 oracle）
> 单 run 日志：9/7 窗口在 `experiments/logs/kuairand_<method>_<seed>.log`；9/6 窗口在 `experiments/results/p5/logs/kuairand_<method>_<seed>.log`
> 编排日志：`experiments/logs/dual_orchestrator_s2.log`（阶段 2 续跑）、`experiments/logs/dual_orchestrator_s2_part1.log`（中断前片段）、`experiments/logs/dual_orchestrator_main.log`（9/7 前段）
> 生成时间：2026-09-07 23:57（Runner 自动出稿，数值全部由 CSV 实算）

### §8.0 执行方式（来自 exp.md §6.6 + KUAI 两份报告）

按 `re.md` 重启，`experiments/dual_orchestrator.py --dataset kuairand --concurrency 2`（30s 错峰、断点可续、只补 status≠OK、日志逐 run 独立重定向到 `experiments/logs/`）。中途一次会话中断导致 comp123 2024/2025 两条 FAILED（外部杀进程、无 traceback），续跑后全部补齐；FAILED 行按红线保留未删改。

**数据**：`results/p5/P5_raw.csv` kuairand **30/30 OK**（6 方法 × 5 seeds）；`results/p5/P5_oracle.csv` **15/15 OK**（3 域 × 5 seeds）。窗口归属按单 run 日志 mtime 实测：baseline×5、fixed073×5、comp1 2022–2024 属 9/6 窗口（`results/p5/logs/`）；comp1 2025/2026、comp2×5、comp1&2×5、comp123×5 属 9/7 窗口（`experiments/logs/`）。

### §8.1 运行环境与硬件状态（`KUAI_P5_MAIN_REPORT.md` §1）

- 编排器：`experiments/dual_orchestrator.py`，`--dataset kuairand --concurrency 2`。**并发上限 2**，相邻 `[launch]` 之间**错峰 30s**（`STAGGER_SEC = 30`，规避数据反序列化 + CUDA context 初始化的内存尖峰）。断点续跑判据：仅当目标 CSV 中该 `(config, method, seed)` 无 `status == OK` 行时才入队。
- 硬件：单机单卡，8 GiB VRAM（编排器内置 `assert concurrency <= 2` 硬上限）。
- 数据：KuaiRand，3 域 × 2 任务；训练集实测域占比 d0/d1/d2 = 8.61% / 86.93% / 4.46%（日志 `[p4] train domain share`），设计口径为本地 d0 = 论文 d1 小域 7.10%、本地 d1 = 论文 d2 大域 89.18%、本地 d2 = 论文 d3 小域 3.72%。
- 训练配置：`--epoch 50`，各方法共用同一套早停规则（无方法独立设置）。

#### §8.1.1 阶段 0 环境漂移自检（KUAI_STAGE1_ORACLE §1）

重跑 ml_orig baseline seed 2022，FINAL test auc = 0.7720050200624052，与 `P5_raw.csv` 既有记录**逐位一致**（6 项 per-(d,t) 亦逐位一致）；对 P0 历史值 0.77025 的偏差 +0.00175，在 ±0.002 容忍带内 → 当前窗口与 P5 主矩阵同窗口，无漂移。

#### §8.1.2 各方法单 run wall_sec 实测范围（P5_raw.csv `wall_sec` 列，仅 OK 行）

| 方法 | n(OK) | wall_sec min | max | mean | peak_mem_mb | 窗口分布 |
|---|---|---|---|---|---|---|
| baseline | 5 | 672 | 1352 | 983 | 1045 | 9/6×5 |
| fixed073 | 5 | 592 | 641 | 612 | 1046 | 9/6×5 |
| comp1 | 5 | 1967 | 3340 | 2795 | 1339 | 9/6×3、9/7×2 |
| comp2 | 5 | 1235 | 1880 | 1526 | 1044 | 9/7×5 |
| comp1_comp2 | 5 | 2990 | 5330 | 4181 | 1339 | 9/7×5 |
| comp123 | 5 | 3500 | 5015 | 4251 | 1339 | 9/7×5 |

#### §8.1.3 本次中断—续跑事实

- 阶段 2 的 comp123 在两次会话中完成：`comp123` 2022/2023 于 9/7 18:20 / 18:51 正常收尾并落盘 OK；随后编排器与其两个子进程在 9/7 21:20–21:21 被会话中断一并终止 —— 2024 / 2025 各只跑到 epoch 0（wall 170s / 140s，两者相差恰为 30s 错峰间隔，且单 run 日志在无任何 traceback 的情况下截断，可判定为外部终止而非训练报错），编排器把这两个 run 记为 `FAILED` 行；2026 在被终止前刚 `[launch]`，未产生 CSV 行。
- 本班（9/7 21:47）以同一命令重启编排器续跑，日志首行 `queued 3 tasks`，与剩余缺格数（comp123 2024/2025/2026）严格一致，确认续跑判据未重复投放已 OK 的 2022/2023。
- P5_raw.csv 中中断留下的 FAILED 行按规程保留、未删改；OK 行以后写的成功行为准（本报告对同一 `(method, seed)` 取 OK 行优先）。

### §8.2 阶段 1：单域 Oracle 真理基准 + 历史 Log 解剖（`KUAI_STAGE1_ORACLE.md` 全文）

> 状态：Oracle 15 runs 运行中（双进程并发，`dual_orchestrator.py`）。本稿先落历史解剖与运行环境，Oracle 明细表待跑批完成后补齐。

#### §8.2.1 历史 Log 深度解剖（`parse_kuairand_detail.py`，数据源 `results/p5/P5_raw.csv`）

域映射：本地 d0=论文 d1（小域 7.10%），本地 d1=**论文 d2（大域 89.18%）**，本地 d2=论文 d3（小域 3.72%）。

| 方法 | n | 6 项均值 | 大域 d2 (local d1) t0/t1 | 小域 d1 (local d0) 均值 | 小域 d3 (local d2) 均值 |
|---|---|---|---|---|---|
| baseline | 5 | 0.647658 | 0.630129 / 0.630305（均值 0.630217） | 0.644338 | 0.668419 |
| fixed073 | 5 | 0.651328 | 0.634131 / 0.642236（均值 0.638184） | 0.644227 | 0.671573 |
| comp1 | 3 | 0.645326 | 0.640842 / 0.633933（均值 0.637387） | 0.633460 | 0.665130 |

配对差值（vs baseline，**配对 n=3，共同 seeds = {2022, 2023, 2024}**）：

| 方法 | 配对 n | 大域 d2 (local d1) 域均值 Δ | 其中大域 t0 Δ | 其中大域 t1 Δ | 小域 d1 (local d0) Δ | 小域 d3 (local d2) Δ | 6 项均值 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| comp1 | 3 | **+0.003321** | +0.010389 | **−0.003746** | **−0.009756** | −0.002501 | −0.002979 |
| fixed073（参照，非配对 n=5） | 5 | — | — | — | — | — | +0.003670 |

> 说明：原 §8.2.1 表格"大域 d2 0.630217 → 0.637387（+0.0072）""小域 d1 0.644338 → 0.633460（−0.0109）""小域 d3 −0.0033"为**非配对**口径（baseline n=5 vs comp1 n=3，跨集合均值差），夸大真实效应约 **2.2 倍**。本次 Reviewer 复算改用配对口径（仅共同 seeds 2022/2023/2024，每个 seed 内做 per-(d,t) 配对差再求均值），原始 6 位小数均值表保留不动，仅差值段落修正。

**解剖结论（事实陈述，配对口径）**：comp1 的 6 项均值 −0.002979 **并非大域 d2 下跌所致**——大域 d2 域均值配对 Δ = **+0.003321**，但**全部增益来自 t0（+0.010389），t1 实为微跌（−0.003746）**；跌幅全部来自小域 d1（local d0）：配对 Δ = **−0.009756**，小域 d3 微跌 −0.002501。即 comp1 在 KuaiRand 上的已有信号是"**大域 t0 单边增益、t1 实为微跌；小域全面受损**"，方向与 MovieLens 的"大小域共享增益"不同。

#### §8.2.2 单域 Oracle 明细表（`KUAI_STAGE1_ORACLE.md` §3）

数据源：`results/p5/P5_oracle.csv`（config=kuairand，3 域 × 5 seeds = 15 runs，全部 status=OK，0 FAILED）。下方 AUC 均为 6 位小数原生精度；均值/标准差按 n=5 在原生全精度上计算后截断显示。

##### §8.2.2.1 逐 seed 明细（6 位小数）

| domain (local) | seed | auc d0t0 | auc d0t1 | wall_sec |
|---:|---:|---:|---:|---:|
| d0 (paper d1 小域 7.10%) | 2022 | 0.593821 | 0.587803 | 215 |
| d0 (paper d1 小域 7.10%) | 2023 | 0.542253 | 0.552463 | 85 |
| d0 (paper d1 小域 7.10%) | 2024 | 0.539067 | 0.523619 | 100 |
| d0 (paper d1 小域 7.10%) | 2025 | 0.541599 | 0.518312 | 65 |
| d0 (paper d1 小域 7.10%) | 2026 | 0.525007 | 0.531283 | 80 |
| d1 (paper d2 大域 89.18%) | 2022 | 0.700494 | 0.728517 | 2125 |
| d1 (paper d2 大域 89.18%) | 2023 | 0.701678 | 0.730260 | 2120 |
| d1 (paper d2 大域 89.18%) | 2024 | 0.700514 | 0.728985 | 2120 |
| d1 (paper d2 大域 89.18%) | 2025 | 0.698455 | 0.727071 | 2115 |
| d1 (paper d2 大域 89.18%) | 2026 | 0.702798 | 0.731773 | 2025 |
| d2 (paper d3 小域 3.72%) | 2022 | 0.590856 | 0.589540 | 55 |
| d2 (paper d3 小域 3.72%) | 2023 | 0.617366 | 0.622311 | 120 |
| d2 (paper d3 小域 3.72%) | 2024 | 0.597830 | 0.599899 | 65 |
| d2 (paper d3 小域 3.72%) | 2025 | 0.597083 | 0.594339 | 45 |
| d2 (paper d3 小域 3.72%) | 2026 | 0.589154 | 0.591334 | 55 |

##### §8.2.2.2 每域 5-seed 均值±std（population std，n=5，6 位小数）

| domain (local) | paper 域 | t0 mean ± std | t1 mean ± std |
|---|---|---|---|
| d0 | d1 小域 7.10% | 0.548349 ± 0.023586 | 0.542696 ± 0.025372 |
| d1 | d2 大域 89.18% | 0.700788 ± 0.001445 | 0.729321 ± 0.001595 |
| d2 | d3 小域 3.72% | 0.598458 ± 0.010041 | 0.599485 ± 0.011942 |

#### §8.2.3 阶段 1 客观异常记录

- 2026-09-06 历史遗留：kuairand comp2/comp1_comp2/comp123/comp1(seed 2025,2026) 曾以残缺参数（`--kuairand` 缺值）启动，全部 argparse 报错，对应日志为 20 行 usage 文本，无训练发生；本次由编排器按 status!=OK 重投。
- **Oracle 跑批（2026-09-07 本次）**：15/15 runs 全部 status=OK，**0 FAILED**，0 早停异常。
  - wall_sec 范围（按域）：
    - 本地 d0（小域）：65–215s；中位 ~85s；seed 2022 单跑 215s 明显高于本域其它 4 跑（65/80/85/100s），但 6 项 AUC 与其它 seed 同量级，未触发异常标记。
    - 本地 d1（大域）：2025–2125s；5 跑全部跑满 50 epoch（无早停），wall 集中在 2115–2125s 区间，仅 seed 2026 略快（2025s）。
    - 本地 d2（小域）：45–120s；中位 ~55s；seed 2023 单跑 120s 略高，其它 4 跑均 ≤65s。
  - 编排器行为：`experiments/logs/dual_orchestrator_main.log` 共计 16 条 [launch] / 15 条 [done]，无重投记录，无 argparse / CUDA / OOM 类报错；d1 大域与 d0/d2 小域混跑时 GPU 显存无溢出（自检空闲显存 7096/8187 MiB）。
  - 历史遗留对照：本次已成功覆盖 §8.2.3 首条提到的 `--kuairand` 缺值残缺参数任务（其原残缺日志为 20 行 usage 文本，本次全部以完整参数启动并 OK）。

#### §8.2.4 阶段 1 Reviewer 审计与 Pareto 仲裁结论

- **真实性核验**：通过。全部 15 个 oracle run 日志与 `P5_oracle.csv` 逐位一致；§8.2.2.2 均值/std 独立复算吻合（std 为 population std，ddof=0，与报告标注一致）。小疵：§8.2.3 称编排器 16 条 [launch]，实测 **17 条**（15 oracle + 2 条阶段 2 comp1 重投）。

- **区间核验**：部分符合。oracle 大域 d2 t0 = 0.700788 与论文单域区间（约 0.69~0.70+）基本相容略偏高；t1 = 0.729321 超出论文 d2t2（0.7007）约 +0.0286。判定：本仓库自测 oracle 是比论文更苛刻的基准（backbone 不同所致）。

- **历史解剖仲裁**：结论方向成立（跌幅不来自大域），但原 +0.0072 / −0.0109 为**非配对**口径，放大约 **2.2 倍**；配对口径如上（§8.2.1 已修订），且大域 t1 实为 −0.003746 微跌。

- **准入意见**：阶段 1 通过，准入阶段 2。
  - 预先记录：oracle 大域 (0.700788 / 0.729321) vs joint baseline (0.630129 / 0.630305) 负迁移 **−0.070659 / −0.099016**；comp1 (0.640842 / 0.633933) 仍差 **−0.059946 / −0.095388**，与现有方法效应量级（±0.003~0.011）相差近一个数量级，**Δ_oracle(d2) ≥ 0 在阶段 2 大概率不可达**。
  - 阶段 2 现实目标应为"**大域负迁移显著收窄且不损小域**"；若坚持 Δ ≥ 0 判据须如实记录不达成。

### §8.3 阶段 2：主矩阵逐 seed 明细（`KUAI_P5_MAIN_REPORT.md` §2）

> 域-任务映射（本地口径）：本地 d0 = 论文 d1（小域 7.10%）；本地 d1 = 论文 d2（**大域** 89.18%）；本地 d2 = 论文 d3（小域 3.72%）。
>
> `per_dt` 列为 `np.float64` 六值，顺序 **[d0t0, d0t1, d1t0, d1t1, d2t0, d2t1]**；下表每格 6 位小数（CSV 原生值，未做任何取舍或再平均）。
>
> **窗口列判定方式**：取该 `(method, seed)` 含 `FINAL test auc` 的单run 日志（优先 `experiments/logs/`，其次 `experiments/results/p5/logs/`），窗口 = 该日志文件 mtime 的日期。实测结果与既有约定一致：baseline / fixed073 全 5 seed + comp1 的 2022–2024 落在 9/6 窗口（日志位于 `experiments/results/p5/logs/`），其余（comp1 2025/2026、comp2 全 5、comp1_comp2 全 5、comp123 全 5）落在 9/7 窗口（日志位于 `experiments/logs/`）。

#### §8.3.1 baseline

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/6 | 0.676392 | 0.623340 | 0.656907 | 0.627461 | 0.617766 | 0.684591 | 0.649506 | 1206 |
| 2023 | 9/6 | 0.677533 | 0.623952 | 0.658806 | 0.629002 | 0.630592 | 0.686179 | 0.646494 | 1352 |
| 2024 | 9/6 | 0.676812 | 0.639608 | 0.656687 | 0.634897 | 0.664679 | 0.684641 | 0.654372 | 978 |
| 2025 | 9/6 | 0.674774 | 0.639648 | 0.654953 | 0.633367 | 0.632240 | 0.683821 | 0.661420 | 706 |
| 2026 | 9/6 | 0.677001 | 0.632338 | 0.657146 | 0.625916 | 0.606247 | 0.685856 | 0.647314 | 672 |

#### §8.3.2 fixed073

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/6 | 0.675865 | 0.634115 | 0.655951 | 0.633870 | 0.626514 | 0.683990 | 0.656723 | 595 |
| 2023 | 9/6 | 0.676642 | 0.622116 | 0.657057 | 0.627489 | 0.635874 | 0.685649 | 0.654850 | 599 |
| 2024 | 9/6 | 0.674826 | 0.630209 | 0.654801 | 0.642976 | 0.643468 | 0.683311 | 0.666445 | 592 |
| 2025 | 9/6 | 0.677772 | 0.638003 | 0.657403 | 0.633794 | 0.664705 | 0.686598 | 0.660690 | 641 |
| 2026 | 9/6 | 0.677396 | 0.634779 | 0.657841 | 0.632528 | 0.640617 | 0.686334 | 0.651137 | 632 |

#### §8.3.3 comp1

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/6 | 0.663583 | 0.640432 | 0.642628 | 0.633113 | 0.647744 | 0.672867 | 0.642575 | 1967 |
| 2023 | 9/6 | 0.671794 | 0.608123 | 0.650985 | 0.635367 | 0.623071 | 0.681344 | 0.644960 | 2493 |
| 2024 | 9/6 | 0.673395 | 0.605515 | 0.653080 | 0.654047 | 0.630983 | 0.681755 | 0.667277 | 3194 |
| 2025 | 9/7 | 0.672750 | 0.633462 | 0.651989 | 0.640619 | 0.666406 | 0.680963 | 0.667245 | 3340 |
| 2026 | 9/7 | 0.661390 | 0.629484 | 0.639523 | 0.633006 | 0.668706 | 0.670039 | 0.656988 | 2980 |

#### §8.3.4 comp2

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/7 | 0.673916 | 0.638870 | 0.653960 | 0.648141 | 0.639601 | 0.681707 | 0.663529 | 1235 |
| 2023 | 9/7 | 0.676320 | 0.645605 | 0.656623 | 0.607650 | 0.667867 | 0.685452 | 0.635254 | 1880 |
| 2024 | 9/7 | 0.674002 | 0.623131 | 0.653084 | 0.627325 | 0.621808 | 0.682718 | 0.654371 | 1590 |
| 2025 | 9/7 | 0.671300 | 0.636598 | 0.651101 | 0.632838 | 0.639601 | 0.680210 | 0.649353 | 1470 |
| 2026 | 9/7 | 0.671423 | 0.635924 | 0.651069 | 0.640400 | 0.631695 | 0.679271 | 0.654184 | 1455 |

#### §8.3.5 comp1_comp2

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/7 | 0.664451 | 0.626171 | 0.645032 | 0.638572 | 0.621153 | 0.672058 | 0.651018 | 5330 |
| 2023 | 9/7 | 0.672115 | 0.613840 | 0.650960 | 0.639106 | 0.646875 | 0.681081 | 0.658332 | 4310 |
| 2024 | 9/7 | 0.670209 | 0.640432 | 0.649340 | 0.634097 | 0.664384 | 0.678680 | 0.649727 | 4535 |
| 2025 | 9/7 | 0.675911 | 0.621336 | 0.655190 | 0.647419 | 0.642435 | 0.684854 | 0.665144 | 3740 |
| 2026 | 9/7 | 0.664450 | 0.631209 | 0.642791 | 0.636260 | 0.659308 | 0.672842 | 0.655929 | 2990 |

#### §8.3.6 comp123

| seed | 窗口 | test_auc_mean | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | wall_sec |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 9/7 | 0.670757 | 0.650123 | 0.650045 | 0.631557 | 0.656221 | 0.679342 | 0.648255 | 4775 |
| 2023 | 9/7 | 0.672294 | 0.629722 | 0.652168 | 0.641202 | 0.646804 | 0.679728 | 0.660877 | 3700 |
| 2024 | 9/7 | 0.668346 | 0.647449 | 0.647700 | 0.643827 | 0.663792 | 0.676872 | 0.672956 | 5015 |
| 2025 | 9/7 | 0.673034 | 0.627417 | 0.653012 | 0.639123 | 0.631705 | 0.681835 | 0.648187 | 4265 |
| 2026 | 9/7 | 0.675483 | 0.633789 | 0.654729 | 0.641043 | 0.626123 | 0.683997 | 0.663750 | 3500 |

#### §8.3.7 每方法 6 项均值 ± std（口径 A：per-(d,t) 先算 AUC 再跨 seed 平均）

> std 口径 **ddof = 0（population std）**，与 `KUAI_STAGE1_ORACLE.md` §8.2.2.2 约定一致；ddof = 1（样本 std）见紧随其后的对照表。n = 5 seed。

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|
| baseline | 0.631777±0.007155 | 0.656900±0.001227 | 0.630129±0.003445 | 0.630305±0.019603 | 0.685018±0.000873 | 0.651821±0.005526 |
| fixed073 | 0.631844±0.005459 | 0.656611±0.001100 | 0.634131±0.005002 | 0.642236±0.012623 | 0.685176±0.001302 | 0.657969±0.005238 |
| comp1 | 0.623403±0.014011 | 0.647641±0.005490 | 0.639230±0.007906 | 0.647382±0.018312 | 0.677394±0.004939 | 0.655809±0.010550 |
| comp2 | 0.636026±0.007300 | 0.653167±0.002062 | 0.631271±0.013740 | 0.640114±0.015341 | 0.681872±0.002149 | 0.651338±0.009258 |
| comp1_comp2 | 0.626598±0.008981 | 0.648663±0.004382 | 0.639091±0.004529 | 0.646831±0.015116 | 0.677903±0.004874 | 0.656030±0.005536 |
| comp123 | 0.637700±0.009317 | 0.651531±0.002438 | 0.639350±0.004174 | 0.644929±0.014250 | 0.680355±0.002409 | 0.658805±0.009519 |

#### §8.3.8 同表 ddof = 1 对照（样本 std）

| 方法 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|
| baseline | 0.631777±0.008000 | 0.656900±0.001372 | 0.630129±0.003852 | 0.630305±0.021917 | 0.685018±0.000976 | 0.651821±0.006179 |
| fixed073 | 0.631844±0.006104 | 0.656611±0.001230 | 0.634131±0.005592 | 0.642236±0.014113 | 0.685176±0.001455 | 0.657969±0.005857 |
| comp1 | 0.623403±0.015665 | 0.647641±0.006138 | 0.639230±0.008840 | 0.647382±0.020474 | 0.677394±0.005522 | 0.655809±0.011795 |
| comp2 | 0.636026±0.008162 | 0.653167±0.002305 | 0.631271±0.015362 | 0.640114±0.017152 | 0.681872±0.002402 | 0.651338±0.010351 |
| comp1_comp2 | 0.626598±0.010041 | 0.648663±0.004899 | 0.639091±0.005064 | 0.646831±0.016900 | 0.677903±0.005450 | 0.656030±0.006189 |
| comp123 | 0.637700±0.010417 | 0.651531±0.002725 | 0.639350±0.004667 | 0.644929±0.015932 | 0.680355±0.002693 | 0.658805±0.010642 |

#### §8.3.9 每方法 test_auc_mean 列均值（口径 B，与口径 A 不同，不可混用）

> `test_auc_mean` 是评估脚本内部按样本聚合出的单标量（口径见 m3oe 的 test 评估定义），与上表"先按 (d,t) 求 AUC 再跨 (d,t) / seed 平均"不是同一个量；两者数值不可互相替换，也不应混在同一张对比表里。

| 方法 | test_auc_mean 均值（ddof=0 std） | 6 项均值的再平均（口径 A 参考） |
|---|---|---|
| baseline | 0.676502±0.000939 | 0.647658 |
| fixed073 | 0.676500±0.001063 | 0.651328 |
| comp1 | 0.668582±0.005051 | 0.648477 |
| comp2 | 0.673392±0.001869 | 0.648965 |
| comp1_comp2 | 0.669427±0.004459 | 0.649186 |
| comp123 | 0.671983±0.002374 | 0.652112 |

### §8.4 阶段 2 Δ_oracle 表（`KUAI_P5_MAIN_REPORT.md` §3）

> Δ_oracle = 各方法该 (d,t) 的 6 项均值口径值 − `P5_oracle.csv` 中 kuairand 对应域任务的 5 seed 均值（oracle = 单域独立训练参考）。Δ > 0 表示主矩阵联合训练在该 (d,t) 上高于单域 oracle，Δ < 0 表示低于。本节只列实算数值，不作漂移/优劣裁定。

#### §8.4.1 oracle 基准（P5_oracle.csv 实算，5 seed 均值，6 位小数）

| oracle domain | 论文域 | n(OK) | t0 均值 | t1 均值 |
|---|---|---|---|---|
| 0 | d1（本地 d0，小域 7.10%） | 5 | 0.548349 | 0.542696 |
| 1 | d2（本地 d1，大域 89.18%） | 5 | 0.700788 | 0.729321 |
| 2 | d3（本地 d2，小域 3.72%） | 5 | 0.598458 | 0.599485 |

#### §8.4.2 Δ_oracle（6 项）

| 方法 | d0t0 Δ | d0t1 Δ | d1t0 Δ | d1t1 Δ | d2t0 Δ | d2t1 Δ |
|---|---|---|---|---|---|---|
| baseline | 0.083428 | 0.114204 | -0.070659 | -0.099016 | 0.086560 | 0.052336 |
| fixed073 | 0.083495 | 0.113914 | -0.066656 | -0.087085 | 0.086719 | 0.058484 |
| comp1 | 0.075054 | 0.104945 | -0.061557 | -0.081939 | 0.078936 | 0.056324 |
| comp2 | 0.087676 | 0.110471 | -0.069517 | -0.089207 | 0.083414 | 0.051853 |
| comp1_comp2 | 0.078248 | 0.105966 | -0.061697 | -0.082490 | 0.079445 | 0.056545 |
| comp123 | 0.089351 | 0.108835 | -0.061437 | -0.084392 | 0.081897 | 0.059320 |

### §8.5 阶段 2 客观异常（`KUAI_P5_MAIN_REPORT.md` §4）

- **状态清点**：kuairand OK 30 / 30（6 方法 × 5 seed）。P5_raw.csv 中另有 2 条 FAILED 行（保留未删）：
  - `comp123 2024` wall=170s，note = `epoch: 0 validation: auc: 0.5326414139489806`
  - `comp123 2025` wall=140s，note = `epoch: 0 validation: auc: 0.5396064193167749`
- **FAILED 实情**：上列 FAILED 行均出现在 9/7 21:20–21:21 的会话中断时刻，两个 run 的 wall 差恰为 30s 错峰间隔，单 run 日志在 epoch 0 后无 traceback 直接截断（对照正常 run 会写出 `fit wall time` / `FINAL per-(d,t) test auc` / `FINAL test auc` 三行）。据此判定为**外部进程终止**，非训练/配置错误；`dual_orchestrator.py` 的 `finalize()` 在 `rc != 0` 且未解析到 AUC 时记 FAILED，行为符合预期，本次**未修改编排器或任何训练脚本**。
- **中断续跑记录**：中断前编排日志片段保存为 `experiments/logs/dual_orchestrator_s2_part1.log`；续跑段追加在 `experiments/logs/dual_orchestrator_s2.log`，两段均含 `queued 3 tasks` 首行。重投的 3 个 seed 使用与前 2 个 seed 完全相同的命令行参数（`--momentum_mode per_domain_only --curriculum_mode domain_conditioned --kappa0 0.5 --arch_grad_ema 1 --arch_update_every 5 --fix_optim_grouping 1`）与相同缓存数据（`cache/data_9836d80cd0f3.pkl`），仅进程重启。
- **窗口归属判定方式**：见 §8.3 表头说明（按含 `FINAL test auc` 的单 run 日志 mtime 实测，非人工指派）。9/6 与 9/7 两窗口的单 run 日志分别落在两个不同目录，可独立复核。
- **跨窗口可比性**：baseline / fixed073 / comp1(2022–2024) 与其余方法分属 9/6、9/7 两个执行窗口，同一方法内部也存在跨窗口（comp1）。两窗口是否存在系统性偏移、以及跨窗口的方法间对比是否成立，**待 Reviewer 仲裁**；本报告不给出任何具体漂移数值结论。
- comp123 的 2024/2025/2026 三个 seed 与 2022/2023 分属中断前后两次进程，属同一 9/7 窗口内的不同进程实例，此事实一并提交 Reviewer 判定是否影响该方法内部 seed 间可比性。

### §8.6 阶段 2 Reviewer 审计与 Pareto 仲裁结论（`KUAI_P5_MAIN_REPORT.md` 末段）

> 审计日期：2026-09-07；仲裁员：Reviewer（科研审计与统计仲裁员，只读复核）。
> 数据口径：P5_raw.csv kuairand 30 行 OK（排除 2 条历史 FAILED 行）、P5_oracle.csv 15 行 OK；per_dt 六值顺序 [d0t0,d0t1,d1t0,d1t1,d2t0,d2t1]；std 为 population std（ddof=0）。

**1. 真实性核验：通过。** §8.3 逐 seed 明细 30 行 × 8 格（test_auc_mean + per_dt 六值 + wall_sec）共 240 格与 P5_raw.csv 逐位一致（0 不符）；抽查 4 条单 run 日志（baseline_2025、comp1_2026、comp1_comp2_2023、comp123_2022）的 FINAL 三行与 CSV 逐位一致；§8.3.7/§8.3.8 两张均值±std 表（ddof=0/1）、§8.3.9 口径 B 表、§8.1.2 wall_sec 表全部复算吻合。oracle 基准复核：d1 t0 = 0.700788 ± 0.001445、t1 = 0.729321 ± 0.001595（population std），与 P5_oracle.csv 及 §8.2.2.2 一致。§8.4.2 Δ_oracle 表 36 格与独立实算逐格一致。§8.5 的 2 条 FAILED 行描述与编排日志（s2_part1 / s2）相符（保留意见：被中断 run 的单 run 日志已被续跑覆盖，截断细节无法独立复核，但现存证据与之相容）。

**2. 显著性检验（vs baseline，paired t 双侧，n=5 seed，Cohen's d 按差值列计）**：

| 口径 | 方法 | Δ | t | p | Cohen's d |
|---|---|---|---|---|---|
| 口径A(6项均值) | fixed073 | +0.003670 | +1.9905 | 0.117385 | +0.8902 |
| 口径A | comp1 | +0.000818 | +0.2809 | 0.792714 | +0.1256 |
| 口径A | comp2 | +0.001306 | +0.3262 | 0.760619 | +0.1459 |
| 口径A | comp1_comp2 | +0.001528 | +0.8766 | 0.430210 | +0.3920 |
| 口径A | comp123 | +0.004453 | +1.8702 | 0.134796 | +0.8364 |
| d1t0 | fixed073 | +0.004003 | +2.1035 | 0.103247 | +0.9407 |
| d1t0 | comp1 | +0.009102 | +3.6002 | **0.022753** | +1.6101 |
| d1t0 | comp2 | +0.001142 | +0.1510 | 0.887289 | +0.0675 |
| d1t0 | comp1_comp2 | +0.008962 | +3.5284 | 0.024266 | +1.5779 |
| d1t0 | comp123 | +0.009222 | +4.5512 | 0.010408 | +2.0354 |
| d1t1 | fixed073 | +0.011931 | +1.1705 | 0.306775 | +0.5235 |
| d1t1 | comp1 | +0.017077 | +1.0115 | 0.369000 | +0.4523 |
| d1t1 | comp2 | +0.009810 | +0.7003 | 0.522354 | +0.3132 |
| d1t1 | comp1_comp2 | +0.016526 | +1.7274 | 0.159161 | +0.7725 |
| d1t1 | comp123 | +0.014624 | +2.0009 | 0.116001 | +0.8948 |

comp1 跨窗口说明：其 2022–2024 三个 seed 与 baseline 同属 9/6 窗口，d1t0 上 3 对窗内差 +0.005652/+0.006365/+0.019150 全为正（均值 +0.010389），显著性不依赖 2025/2026 两对跨窗口配对。15 个检验均未过 Bonferroni（α=0.0033），d1t0 显著性为名义显著，但方向与组件设计意图及阶段 1 预登记一致，作方向性结论采信。

**3. Pareto 仲裁：裁定 comp1 / comp1_comp2 / comp123 在大域 d1 上为"前沿移动"（负迁移收窄但仍 < 0），真 Pareto 外推不达成。** 依据：oracle 坐标 d1 = 0.700788 / 0.729321；baseline Δ(d1) = −0.070659 / −0.099016；comp1 收窄至 −0.061557 / −0.081939（+0.009102 / +0.017077，t0 p=0.022753）；comp1_comp2 −0.061697 / −0.082490（t0 p=0.024266）；comp123 −0.061437 / −0.084392（t0 p=0.010408）。小域条件满足（d0 Δ +0.075054~+0.089351；d2 Δ t0 +0.078936~+0.086560、t1 +0.052336~+0.059320），但 Δ_oracle(d1) 距 ≥ 0 仍差约 0.061 / 0.082，收窄量仅覆盖缺口的 13–17%，与阶段 1 预登记"Δ(d1) ≥ 0 大概率不可达"一致。fixed073 与 comp2 的收窄不显著，不计入前沿移动。

**4. 跨窗口可比性裁决：结论可信，无需重跑对齐。** (a) 9/7 重跑探针 drift_ml_orig_baseline_2022.log 与 9/6 窗口 CSV 逐位一致（0.7720050200624052），实测窗口漂移为 0（探针为 ml_orig，不逐位担保 kuairand）；(b) 受跨窗口影响的唯一方法间结论（comp1 d1t0）在纯窗内配对上独立成立；(c) comp123 中断前后两进程实例（2022/2023 vs 2024/2025/2026）同窗口、同命令行、同数据缓存，口径 A 两组取值区间完全重叠，seed 间可比。

**签署：以上六项核验全部通过，无数字不符点，准予签署。—— Reviewer，2026-09-07**

---

## §9 P5 终审综合报告（`experiments/reports/FINAL_RESEARCH_REPORT.md` 全文）

# M3oE 极端不均衡下大域负迁移消解与动量优化研究报告

> 最终成果报告（re.md 阶段 4 交付物）。起草：Master；终审签署：Reviewer。
> 数据基础：P0–P5 全部实验（2026-09-05 至 2026-09-08），所有数字来自 `experiments/results/` 原始 CSV 与三份已签署阶段报告（`KUAI_STAGE1_ORACLE.md`、`KUAI_P5_MAIN_REPORT.md`、`ML_P5_REPORT.md`，已分别整合进 §8.2、§8.3–§8.6、§7.1–§7.9）。指标一律 6 位小数原生值；显著性为双侧配对 t 检验（按 seed 配对），Cohen's d 按差值列计算。

---

### §9.1 研究问题

原 M3oE 论文（SIGIR 2024）在 KuaiRand-Pure（24:1 极端不均衡，大域占 89.18%）上暴露出大域相对单域独立模型（Oracle）的严重负迁移（AUC 倒跌 1.7~3.2 点）。本研究验证三个自研组件：

- **组件 1**：域规模驱动的逐域梯度动量解耦（per-domain EMA 动量，`momentum_mode=per_domain_only`，含优化器分组修复）；
- **组件 2**：域条件课程（domain-conditioned curriculum，κ0=0.5）；
- **组件 3**：架构参数 EMA 平滑 + 低频更新。

核心判定准则（oracle 坐标）：Δ_oracle(d) = AUC_Our(d) − AUC_Oracle(d)；大域 Δ ≥ 0 且小域 Δ > 0 方为**真 Pareto 外推**。

### §9.2 实验基础与可信度声明

- **复现成功**：as-published 5 seeds 0.7686±0.0017 vs 论文 0.7702（95% CI 覆盖），论文 Table 1 数据集统计 12/12 逐位复现（§2 P0 / §5 P3）。
- **代码事实核查**：24 条断言全部核到；实锤 B10 优化器分组 bug（架构参数被两个 Adam 同时更新），修复后均值不变但种子方差缩小 ~8 倍（§4 P2）。
- **环境可控性**：跨会话存在 GPU 桌面负载导致的数值窗口漂移风险；两次实测漂移探针（9/7、9/8 跨日）均 0.000000 逐位一致。全矩阵方法间对比均经窗口归属核验，Reviewer 两次裁决"结论可信、无需重跑对齐"。
- **数据规模**：MovieLens 主矩阵 144 runs（6 档 × 6 方法，3/5 seeds）+ oracle 45 runs；KuaiRand 主矩阵 30 runs + oracle 15 runs。全部 status=OK，历史 FAILED 行如实保留。

### §9.3 组件有效性结论（MovieLens，不均衡梯度 1:1 → 100:1）

**3.1 组件 1 在强不均衡档（≥30:1）收益巨大且高度显著**（口径 A，vs baseline 配对）：

| 档 | Δ | p | Cohen's d |
|---|---|---|---|
| ml_r30（30:1，n=5） | +0.016598 | 0.000204 | +5.80 |
| ml_r100（100:1，n=5） | +0.020233 | 0.006084 | +2.37 |

r30 档全部 6 项 (d,t) 同号提升（六项均值口径 +0.013~+0.020），非单域偶发。

**3.2 但"增益随不均衡度单调放大"不成立**（六档全序列，Reviewer 2026-09-08 仲裁）：
1:1 +0.006354（p=0.016174）→ 1.9:1 +0.003381（p=0.027781）→ **3:1 −0.001231（p=0.032670，d=−3.12）** → 10:1 +0.005277（p=0.178438）→ 30:1/100:1 大幅显著。3:1 档为 3/3 seeds 同号的名义显著负凹陷，幅度 ≈0.0012 接近种子噪声底、n=3 效力不足，定性为**待复核异常**，不作机理证据。1:1–10:1 区间整体非单调。

**3.3 组件 2 单独无效**（全部六档 |Δ|≤0.0015 且不显著）；**组件 3 无稳定贡献**（comp123 − comp1&2 六档差 −0.0018~+0.0005，r3/r10 档甚至为正值）——与 §3.1 P1 D0.2 / §4 P2"架构搜索实际贡献≈0"（fixed-0.7311 vs 原版 p=0.32）自洽。组合最优为 comp1&2（r30 +0.017701 p=0.000145、r100 +0.020523 p=0.003880），并在 100:1 档收窄域间差距（gap：baseline 0.033583 → comp1&2 0.032620 → comp123 0.031064；注意 comp1 单独反升至 0.036148）。

**3.4 组件设计前提被证伪**：§3.2 P1 梯度诊断显示"小域梯度 SNR 更低"不成立（域间无差异，小域梯度范数反而更大；跨域梯度弱正相关 0.18–0.26）。组件 1 的收益机制因此不能归因于原设计叙事，机理归因留给 §10 P6 消融。

### §9.4 KuaiRand 大域负迁移（核心命题的终审结论）

**4.1 oracle 基准比论文更苛刻**：单域 oracle（5 seeds，population std）大域 d2 t0 = 0.700788±0.001445（与论文区间相容）、t1 = 0.729321±0.001595（超论文值 +0.0286）。

**4.2 大域负迁移在 baseline 上复现且幅度更大**：Δ_oracle(大域) = −0.070659（t0）/ −0.099016（t1）。

**4.3 组件 1 系方法显著收窄负迁移（前沿移动）**：

| 方法 | Δ_oracle d1t0（收窄量，p） | Δ_oracle d1t1（收窄量，p） |
|---|---|---|
| baseline | −0.070659 | −0.099016 |
| comp1 | −0.061557（+0.009102，p=0.022753，d=1.61） | −0.081939（+0.017077，p=0.369000） |
| comp1&2 | −0.061697（+0.008962，p=0.024266） | −0.082490（+0.016526，p=0.159161） |
| comp123 | −0.061437（+0.009222，p=0.010408，d=2.04） | −0.084392（+0.014624，p=0.116001） |

小域各方法 Δ 全为正（d0 +0.075~+0.114；d2 +0.052~+0.087），"小域 > 0"条件满足。comp1 的 d1t0 优势在 3 对纯同窗口配对上独立成立（+0.005652/+0.006365/+0.019150），不受跨窗口混合影响。（注：KuaiRand 本地域编号 d1 = 论文大域 d2，本节表头 d1t0/d1t1 即论文大域的两任务。）

**4.4 Reviewer 裁定：前沿移动，非真 Pareto 外推。** Δ_oracle(大域) 距 ≥0 仍差约 0.061/0.082，收窄量仅覆盖缺口的 13–17%。15 个检验均未过 Bonferroni（α=0.0033），d1t0 显著性为名义显著、因方向与设计意图及阶段 1 预登记一致而作方向性结论采信。

**4.5 MovieLens 侧的呼应**：ML 无 89% 极端大域（最大域 40%），"大域保护"原命题不适用；但"相对最大域 d1 负迁移收窄"同向复现于全部三个 oracle 档（comp1 vs baseline：orig +0.008250/+0.001752；r30 +0.016851/+0.016315；r100 +0.018173/+0.024837），且为同档内方法间对比、oracle 常数抵消，不受口径缺陷污染。

### §9.5 口径与证据边界（诚实声明）

1. **ML oracle 混合口径**：`p4_run.py` 单域分支不执行下采样，三档 oracle 恒为"单域全量训练"。ml_orig 档 Δ_oracle 干净；**r30/r100 的 Δ_oracle 绝对水平不得作为 Pareto ≥0 判据**（Reviewer 裁决四）；纯迁移口径的备用修复规约已存档（§7.9 终审签署第四节），本阶段结论均不依赖它。
2. **seeds 不对称**：ml_orig/r30/r100 为 5 seeds，r1/r3/r10 为 3 seeds（计划内降级）；跨档趋势只用均值 Δ，p 值跨档不可比（n=3 档 t_crit=4.303）。
3. **oracle 覆盖空档**：ml_r1/r3/r10 无 oracle，该三档无任何 Pareto 陈述。
4. **已知负面**：comp2 全线无效；comp3 轻微拖累；3:1 档负凹陷未解释；KuaiRand 大域缺口未闭合。
5. **多重比较**：全部检验为名义显著口径，未做多重校正通过声明；关键结论均标注了对应 p 值与效应量。
6. **执行窗口不齐**：ml_r3 为跨窗混合矩阵（baseline×3 与 fixed073 2022 属早期窗口，其余 14 格属 9/8 窗口）；漂移探针 0.000000 仅覆盖 ml_orig 一格，不逐位担保其他档位（Reviewer 裁决保留为注意事项：r3 凹陷 3 seeds 同号、窗口伪影可能性低）。

### §9.6 效率成本

组件 1 的 D 次反传代价：ml_orig 上 wall +52%（272s→413s）、显存 +216MiB；KuaiRand 上 ~975s→~2787s（comp1 5-seed fit 均值，区间 1962–3328s，2M 行）。r30/r100 档绝对成本小（66–102s/run）。

### §9.7 最终结论

1. M3oE 的"可微 AutoML"贡献≈0（冻结初值无显著差异），其论文的 w/o AutoML 损失在我们的复现中不成立。
2. 组件 1（逐域动量解耦）是本研究唯一稳定有效的组件：在 MovieLens 强不均衡档（≥30:1）取得大效应量显著提升，在 KuaiRand 极端不均衡下显著收窄大域负迁移（前沿移动）；但**真 Pareto 外推（大域追平单域 oracle）未达成**，残余缺口约为已收窄量的 5–7 倍。
3. "小域梯度噪声更大"的设计前提不成立，组件 1 的生效机理需 §10 P6 消融进一步归因（γ 公式 vs 均匀 vs SNR 反推等 17 变体已就绪）。
4. 上述每条断言均有 p 值、Cohen's d 与 oracle 数据背书；三份阶段报告经 Reviewer 逐项核验（KuaiRand 240 格 + ML 36 格子/108 Δ 格全量复算均 0 不符）并签署。

### §9.8 档案索引

| 产物 | 路径 |
|---|---|
| 主台账 | `experiments/results/p5/P5_raw.csv`（178 数据行：OK 174 = ML 144 + kuairand 30；历史 FAILED 4 条保留） |
| oracle 台账 | `experiments/results/p5/P5_oracle.csv`（60 行 OK） |
| 阶段报告（已签署） | `experiments/reports/KUAI_STAGE1_ORACLE.md`（§8.2）、`KUAI_P5_MAIN_REPORT.md`（§8.3–§8.6）、`ML_P5_REPORT.md`（§7.1–§7.9） |
| 综合报表 | `experiments/results/p5/P5_main.csv`（174 runs 长表）、`figs/p5_trend.png`（六档 Δ 曲线） |
| 断点台账 | `experiments/logs/stage3_checkpoint.md` |
| 交接文档 | `exp.md`（本文档，含 P6/P7 后续清单） |

---

### §9.9 Reviewer 终审签署

**审计日期：2026-09-08　署名：Reviewer　状态：已生效**

**终审范围**：本报告全部定量断言逐条对照 `P5_raw.csv` / `P5_oracle.csv` 独立复算（KuaiRand Δ_oracle 全表、d1 收窄配对 p/Cohen's d、comp1 纯窗口三对配对、ML 六档梯度、ML 三档 d1 收窄量、域间差距、效率均值、行数与文件清点），并对照三份已签署阶段报告与 exp.md / stage3_checkpoint.md 核验转抄值。

**零不符声明**：§9.3.1/§9.3.2 六档梯度与 r30/r100 显著性表、§9.4.1–§9.4.4 全部 oracle 与 Δ_oracle 数字（含 d1t1 三个 p 值 0.369000/0.159161/0.116001 与 comp123 d=2.04，经本审独立复算确认）、§9.4.5 ML d1 收窄六数、§9.2 复现值 0.7686±0.0017 与 B10/方差 8× 声明、§9.6 效率数字、§9.8 全部索引与行数——均与台账一致。表述审查：3:1 凹陷定性（待复核异常、不作机理证据）、前沿移动而非真 Pareto、名义显著口径，均与本人此前仲裁一致，无越界断言。

**8 项修正已复核落地**：首轮终审提出的 8 处数字/表述错误（F1 小域 d0 范围、F2 r30 六项提升范围、F3 域间差距数值与归属、F4 组件 3 六档差表述、F5 漂移探针次数、F6 §9.5 补执行窗口边界、F7 KuaiRand comp1 耗时、F8 P5_raw.csv 行数构成）及 2 处措辞建议（§9.4.3 域编号统一标注、§9.7.2 缺口倍数表述），经本轮逐字复核全部按本人给出的正确值落实，exp.md 两处同源过时表述亦同步修正并留痕。

**结论**：本报告数据层、统计层与证据边界声明全部核验通过，予以签署生效。本人同意"re.md 四阶段闭环、P5 宣告完成"：MovieLens 六档梯度与 KuaiRand 极端不均衡两条证据链完整；组件 1 在强不均衡档（≥30:1）的显著收益与 KuaiRand 大域负迁移收窄（前沿移动、非真 Pareto 外推）成立；组件 2 无效、组件 3 无稳定贡献、"单调放大"命题不成立等负面结论均已如实载明；全部口径限制（oracle 混合口径、seeds 不对称、oracle 空档、多重比较、执行窗口不齐）已在 §9.5 完整声明。—— **Reviewer，2026-09-08**

---

## §10 P6 消融实验报告（`experiments/reports/P6_ablation.md` 全文）

> 数据源：`experiments/results/p6/P6_raw.csv`（P6 编排器原始落账，85 行全 OK）+ `experiments/results/p5/P5_raw.csv`（ml_r30 对照臂）。
> 编排器：`experiments/p6_dual_orchestrator.py --concurrency 2`，17 变体 × 5 seeds = 85 任务，30s 错峰；运行至 `P6_DUAL_ORCHESTRATOR_DONE`。
> 显存峰值：max=1158 MB、mean=1107 MB（p4_run 单进程口径，显著低于 7 GB 警戒）。
> 分析：`experiments/p6_report.py`，配对 t 检验（同一 seed 内 variant vs ref），Cohen's d = mean(diff)/std(diff, ddof=1)，n=5。
> 全部数值为 6 位小数原生绝对值；引自 `experiments/results/P6_ablation.csv`（25 行长表）与 `experiments/results/p6/p6_report_stdout.txt`。
> 生成时间：2026-09-08 13:30（Runner 自动出稿）。

### §10.0 总览

P6 共 17 变体 × 5 seeds（2022..2026）= 85 任务，单进程 + 30 s 错峰 + 并发 2 跑完，无 FAILED、无自愈重启。控制臂 baseline / fixed073 / comp1 / comp2 / comp1_comp2 / comp123 来自 §7 P5 ml_r30 配置（与本批次 P6 共用同一套训练脚本与早停）。

各组 arm 级均值 / 标准差 / 配对 p / Cohen's d 详见 `experiments/results/P6_ablation.csv`，下表逐组展开。

---

### §10.1 g1 动量分档形式

> 三档：`per_domain_only`（comp1，公式 γ_d = 1-(1-γ_0)p_d/p_max）、`uniform_gamma`（g1_uniform_gamma，γ_d ≡ γ_0 = 0.9，全域均一、无规模分档）、`baseline`（momentum_mode=none，即 Adam β1）。
> **勘误（2026-09-08 Master 复审）**：本节初稿将 uniform_gamma 误写为"γ_d ≡ 1"；按 `p4_run.py:203` 实际语义为 γ_d ≡ γ_0 = 0.9（γ≡1 会冻结全部共享参数，与实测 0.723894 矛盾）。仅标签勘误，数据与结论不变。

| arm | n | mean ± std (test_auc) | vs comp1 | vs baseline |
|---|---|---|---|---|
| baseline（无 EMA） | 5 | 0.722514 ± 0.002910 | — | — |
| g1_uniform_gamma（γ_d ≡ 1） | 5 | 0.723894 ± 0.004975 | diff = -0.017948, p = 0.000675, d = -4.265255 | diff = +0.001380, p = 0.420079, d = 0.401478 |
| comp1（per_domain，公式 γ_d） | 5 | 0.741842 ± 0.001838 | — | diff = +0.019329, p = 0.000022, d = 10.163744 |

**结论**：
- comp1 相对 baseline 配对 diff +0.019329（p = 2.2e-5，d = 10.16），**显著**。
- g1_uniform_gamma 相对 baseline 配对 diff +0.001380（p = 0.420，d = 0.40），**不显著**。
- comp1 相对 g1_uniform_gamma 配对 diff -0.017948（p = 6.8e-4，d = -4.27），**显著**。
- **归因解读**：动量 EMA 的增益不是"任何 EMA 都能给"，而是"按域差异化的 EMA"才能给。把 γ 强行压成全域统一常数等价于把动量信息压回 Adam 等价类，提升落进噪声。

---

### §10.2 γ_d 的来源

> 三个来源：公式（comp1，已确认优）/ 手工设定（g2_manual_gamma，给 γ_manual=0.95,0.90,0.90）/ SNR 反推（g2_snr_gamma，P1 的 D0.4 实测 SNR）。ref = comp1。

| arm | n | mean ± std | diff vs comp1 | p | Cohen's d |
|---|---|---|---|---|---|
| comp1（公式） | 5 | 0.741842 ± 0.001838 | — | — | — |
| g2_manual_gamma（手工 γ=0.95/0.90/0.90） | 5 | 0.727853 ± 0.004711 | -0.013990 | 0.002328 | -3.080590 |
| g2_snr_gamma（SNR 反推） | 5 | 0.724179 ± 0.003149 | -0.017664 | 0.000273 | -5.381670 |

**结论**：
- 手工 γ 与 SNR 反推 γ **均显著差于**公式 γ（p = 0.0023 与 p = 0.0003），效应量 d = -3.08 与 -5.38。
- **归因解读**：γ_d 的最优解对先验非常敏感——手工拍脑袋、SNR 反推都给出系统性负增益，唯一可靠的是把 p_d 当做 γ_d 的标尺（公式 γ_d = 1-(1-γ_0)·p_d/p_max）。

---

### §10.3 β_d 粒度：per-domain vs 全局单标量

> 关课程，仅隔离 β 参数化粒度。ref = baseline。

| arm | n | mean ± std | diff vs baseline | p | Cohen's d |
|---|---|---|---|---|---|
| baseline | 5 | 0.722514 ± 0.002910 | — | — | — |
| g3_beta_perdomain_nocur（per-domain β，无课程） | 5 | 0.721447 ± 0.004494 | -0.001067 | 0.394056 | -0.426692 |

**结论**：diff = -0.001067（p = 0.394，d = -0.43），**不显著**。

**归因解读**：单把 β 扩成 per-domain、但不开课程，参数化扩展本身不带来可观测的 AUC 改进；这给 §10.4 的"课程 vs 固定终值"归因提供了干净的零基线。

---

### §10.4 课程 vs 固定终值

> 同为 per-domain β 参数化；区别：comp2 开课程（domain_conditioned, κ_0=0.5），g3_beta_perdomain_nocur 用同样 per-domain β 但**关课程**（κ_0=0.0）。ref = g3_beta_perdomain_nocur。

| arm | n | mean ± std | diff vs g3_beta_perdomain_nocur | p | Cohen's d |
|---|---|---|---|---|---|
| g3_beta_perdomain_nocur（关课程） | 5 | 0.721447 ± 0.004494 | — | — | — |
| comp2（开课程，κ_0=0.5） | 5 | 0.723143 ± 0.004132 | +0.001696 | 0.191530 | 0.702071 |

**结论**：diff = +0.001696（p = 0.192，d = 0.70），**不显著**。

**归因解读**：参数化扩展（per-domain β）单独拿出来不增不减；课程调度（per-seed 的训练轨迹）也单独拿出来不增不减。两条合流到 comp1_comp2 时才出现"组件正交"的稳定增益（见 §10.6）。这说明组件1和组件2单独都不足以撑起方法名头，但**互为前置**：组件2 的课程轨迹必须建立在组件1 的差异化 EMA 之上才能生效。

---

### §10.5 课程方向反转

> κ_0 = +0.5（comp2，正向：易→难） vs κ_0 = -0.5（g5_reverse_curriculum，反向）。ref = comp2。

| arm | n | mean ± std | diff vs comp2 | p | Cohen's d |
|---|---|---|---|---|---|
| comp2（κ_0=+0.5，正向） | 5 | 0.723143 ± 0.004132 | — | — | — |
| g5_reverse_curriculum（κ_0=-0.5，反向） | 5 | 0.721957 ± 0.001621 | -0.001187 | 0.596935 | -0.256531 |

**结论**：diff = -0.001187（p = 0.597，d = -0.26），**不显著**。

**归因解读**：在 ml_r30 / 三域 / 6 位小数原生值下，方向反转带来的 diff 远小于 seed 间噪声，**不能用此实验区分课程方向是否重要**。但反向课程并未拉低至显著更差，意味着 comp2 的正向方向选择本身也未在本数据上展现统计优势——这与 p6.md 中"CHEST 反转课程实验"对比的关键差异在于：CHEST 数据集有显式的域难度梯度信号，KuaiRand-Pure ml_r30 的三域 d0/d1/d2 = 7.10% / 89.18% / 3.72% 没有显式难度注释，p_d 仅是占比代理、不是难度。

---

### §10.6 组件正交性

> 7 个 arm：单组件 3、两两组合 3、三联合 1。ref = baseline。配对 t 见下表，diff = arm − baseline。

| arm | n | mean ± std | diff vs baseline | p | Cohen's d |
|---|---|---|---|---|---|
| baseline（无 EMA、无课程、无 arch_grad_ema） | 5 | 0.722514 ± 0.002910 | — | — | — |
| comp1（仅 per_domain 动量） | 5 | 0.741842 ± 0.001838 | +0.019329 | 0.000022 | 10.163744 |
| comp2（仅课程） | 5 | 0.723143 ± 0.004132 | +0.000630 | 0.657240 | 0.214015 |
| g6_comp3（仅 arch_grad_ema） | 5 | 0.723050 ± 0.003746 | +0.000537 | 0.641133 | 0.225133 |
| comp1&2（comp1 + comp2） | 5 | 0.743480 ± 0.002051 | +0.020966 | 0.000004 | 16.055391 |
| g6_comp1+comp3（comp1 + comp3） | 5 | 0.742602 ± 0.001961 | +0.020088 | 0.000361 | 5.013211 |
| g6_comp2+comp3（comp2 + comp3） | 5 | 0.724136 ± 0.004242 | +0.001622 | 0.278714 | 0.560006 |
| comp123（全部三个） | 5 | 0.745016 ± 0.002022 | +0.022502 | 0.000084 | 7.260780 |

**两两配对差矩阵（mean(arm_row) − mean(arm_col)，同 seed 配对）**

| 行＼列 | comp1 | comp2 | g6_comp3 | comp1&2 | g6_comp1+comp3 | g6_comp2+comp3 | comp123 |
|---|---|---|---|---|---|---|---|
| comp1 | — | +0.018699 | +0.018792 | -0.001638 | -0.000760 | +0.017706 | -0.003173 |
| comp2 | -0.018699 | — | +0.000093 | -0.020337 | -0.019459 | -0.000993 | -0.021872 |
| g6_comp3 | -0.018792 | -0.000093 | — | -0.020430 | -0.019552 | -0.001086 | -0.021965 |
| comp1&2 | +0.001638 | +0.020337 | +0.020430 | — | +0.000878 | +0.019344 | +0.001536 |
| g6_comp1+comp3 | +0.000760 | +0.019459 | +0.019552 | -0.000878 | — | +0.018466 | +0.002414 |
| g6_comp2+comp3 | -0.017706 | +0.000993 | +0.001086 | -0.019344 | -0.018466 | — | -0.020880 |
| comp123 | +0.003173 | +0.021872 | +0.021965 | +0.001536 | +0.002414 | +0.020880 | — |

**结论**：
- 单组件：**comp1 显著**（d = 10.16），**comp2 不显著**（d = 0.21），**g6_comp3 不显著**（d = 0.23）。
- 含 comp1 的组合（comp1&2、g6_comp1+comp3、comp123）均**显著高于 baseline**（p ≤ 0.0004，d ≥ 5.01）。
- 不含 comp1 的组合（g6_comp2+comp3）**不显著**（p = 0.279，d = 0.56）。
- comp123 vs comp1 配对差 = +0.003173，comp1&2 vs comp1 配对差 = +0.001638，**绝对值都很小**，处于 seed 噪声带边缘。
- **归因解读**：comp1 是主增益源；comp2 / comp3 单独不增、必须配 comp1 才出可观测的正向；comp1 与 comp2/3 接近可加但叠加量级（≈+0.002~+0.003）显著小于 comp1 自身的 +0.019，绝大部分增益归 comp1。

---

### §10.7 超参敏感性（κ_0 与 γ_0）

> 平台区宽度的客观描述：连续改变 κ_0 ∈ {0.1, 0.25, 0.5, 1.0, 2.0}（20 倍）与 γ_0 ∈ {0.8, 0.85, 0.9, 0.95, 0.99}，以 comp1_comp2（κ_0=0.5）为 κ 参照、comp1（γ_0=0.9）为 γ 参照。
> 曲线图：`experiments/results/p6/figs/p6_sensitivity.png`（左 κ_0、右 γ_0；横轴 log 仅 κ_0）。

#### §10.7.1 κ_0 扫描（ref = comp1_comp2, κ_0=0.5）

| κ_0 | arm | n | mean ± std | diff vs comp1_comp2 | p | Cohen's d |
|---|---|---|---|---|---|---|
| 0.10 | g7_kappa0_0.1 | 5 | 0.743601 ± 0.001387 | +0.000120 | 0.856611 | 0.086161 |
| 0.25 | g7_kappa0_0.25 | 5 | 0.743148 ± 0.001262 | -0.000332 | 0.800360 | -0.120846 |
| 0.50 | comp1_comp2 | 5 | 0.743480 ± 0.002051 | — | — | — |
| 1.00 | g7_kappa0_1.0 | 5 | 0.742090 ± 0.002273 | -0.001390 | 0.141117 | -0.818703 |
| 2.00 | g7_kappa0_2.0 | 5 | 0.743290 ± 0.002213 | -0.000190 | 0.826179 | -0.104830 |

- 范围 κ_0 ∈ [0.1, 2.0]（20 倍），均值跨度 0.7421–0.7436（极差 0.0015）。
- **所有配对 t 均不显著**（p ≥ 0.141，最大 |d| = 0.82 出现在 κ_0=1.0，但仍未达 5%）。
- **平台区宽度客观描述**：在 ml_r30 三域 / 5 seeds 噪声带（≈±0.001–0.002）下，κ_0 在 0.1 到 2.0 的整段取值均落在该噪声带内且与 κ_0=0.5 不可区分。**整段 20 倍范围为有效平台区**。

#### §10.7.2 γ_0 扫描（ref = comp1, γ_0=0.9）

| γ_0 | arm | n | mean ± std | diff vs comp1 | p | Cohen's d |
|---|---|---|---|---|---|---|
| 0.80 | g7_gamma0_0.8 | 5 | 0.741704 ± 0.001595 | -0.000139 | 0.910701 | -0.053406 |
| 0.85 | g7_gamma0_0.85 | 5 | 0.742651 ± 0.000770 | +0.000808 | 0.368317 | 0.453068 |
| 0.90 | comp1 | 5 | 0.741842 ± 0.001838 | — | — | — |
| 0.95 | g7_gamma0_0.95 | 5 | 0.739562 ± 0.002764 | -0.002281 | 0.224239 | -0.642378 |
| 0.99 | g7_gamma0_0.99 | 5 | 0.727887 ± 0.003886 | -0.013955 | 0.000985 | -3.866053 |

- 范围 γ_0 ∈ [0.8, 0.99]，均值跨度 0.7279–0.7427（极差 0.0148）。
- γ_0 ∈ {0.8, 0.85, 0.95} vs 0.9 均**不显著**（p ≥ 0.224，|d| ≤ 0.64）；γ_0=0.95 已有可见负趋势（-0.00228）但仍处噪声带。
- γ_0=0.99 **显著变差**（p = 0.000985，d = -3.87，单次配对测试）。
- **平台区宽度客观描述**：γ_0 在 [0.8, 0.95]（宽 0.15 的窗口）内不显著区分；从 0.95 越过即开始负偏离，至 0.99 跌出噪声带形成可见悬崖。平台区右端点 = 0.95。

**§10.7 总结**：κ_0 的平台区宽度（≥20 倍）显著宽于 CoHeat 等课程学习工作自陈的"过快的调度即过拟合"敏感区；γ_0 的平台区右端点明确落在 0.95。γ_0=0.99 触发悬崖的原因可能是 γ 接近 1 时 EMA 退化为接近不更新的滑动平均，与 §10.2 中"γ 偏离公式标尺即掉点"的方向一致——把 γ 推得太高，等同于在 §10.1 维度上接近 g1_uniform_gamma 的退化解。

---

### §10.8 Adam 动量关系（g8）

> 三档 `momentum_mode`：none（baseline）、`per_domain_only`（comp1）、`per_domain_stacked`（g8_stacked）。ref = baseline。
> 直答审稿人问题："我们的 EMA 是否只是重复了 Adam β1？"

| arm | n | mean ± std | diff vs baseline | p | Cohen's d |
|---|---|---|---|---|---|
| baseline（momentum_mode=none，即 Adam β1 单独） | 5 | 0.722514 ± 0.002910 | — | — | — |
| comp1（momentum_mode=per_domain_only，**替换** Adam β1） | 5 | 0.741842 ± 0.001838 | +0.019329 | 0.000022 | 10.163744 |
| g8_stacked（momentum_mode=per_domain_stacked，**叠加**在 Adam β1 之上） | 5 | 0.729893 ± 0.002042 | +0.007379 | 0.014405 | 1.850386 |

**结论**：
- comp1 vs baseline：diff = +0.019329（p = 2.2e-5，d = 10.16），**显著**。
- g8_stacked vs baseline：diff = +0.007379（p = 0.014，d = 1.85），**显著但效应量小一个数量级**。
- 直接差值 comp1 vs g8_stacked（按 §10.6 配对矩阵）：+0.011950（≈+0.012），g8_stacked 与 comp1 的配对 t 在 stdout 中未直接报告，但从均值差 + 配对种子数与 std 量级可判断**差异量级远超** g8_stacked 自身的提升。
- **结论：我们的 EMA 不是 Adam β1 的复制品**。comp1（用 per-domain γ_d 替换 β1）才是产生主增益的方式；把我们的 EMA 叠加在 Adam β1 之上（g8_stacked）只留下小幅增益（+0.007），说明我们 EMA 的正确用法是**替代**而非**叠加**——这点和 §10.1 维度上"统一 γ 即退化"、§10.2 维度上"γ 偏离公式标尺即掉点"互相印证。

---

### §10.9 哪些组件的贡献无法与噪声区分

> 噪声口径：ml_r30 三域 5-seed 下，相同 arm 内 seed 间 std 通常 ≤ 0.003（comp2 / g6_comp3 等不带 comp1 的 arm 略高，约 ±0.004）；同 seed 内两 arm 的配对 diff std 由 paired t 给出（|d|<1 时视为噪声级）。
> 严格遵循 p6.md：以下组件的贡献**不显著**，明确写"不显著"，不使用"略有提升"等模糊词。

| 组件 / 维度 | 实验组 | 配对 diff | p | Cohen's d | 结论 |
|---|---|---|---|---|---|
| 统一常数 γ_d（取代 per-domain γ_d） | §10.1 g1_uniform_gamma vs baseline | +0.001380 | 0.420079 | 0.401478 | **不显著** |
| 仅 per-domain β，关课程（β 参数化单独的贡献） | §10.3 g3_beta_perdomain_nocur vs baseline | -0.001067 | 0.394056 | -0.426692 | **不显著** |
| 仅课程（comp2 单独的贡献） | §10.6 comp2 vs baseline | +0.000630 | 0.657240 | 0.214015 | **不显著** |
| 仅 arch_grad_ema（comp3 单独的贡献） | §10.6 g6_comp3 vs baseline | +0.000537 | 0.641133 | 0.225133 | **不显著** |
| 课程方向（正 vs 反） | §10.5 g5_reverse_curriculum vs comp2 | -0.001187 | 0.596935 | -0.256531 | **不显著** |
| 课程开启相对关课程（comp2 vs g3_beta_perdomain_nocur） | §10.4 | +0.001696 | 0.191530 | 0.702071 | **不显著** |
| κ_0 = 0.1 / 0.25 / 1.0 / 2.0 vs 0.5 | §10.7.1 | ±0.0001 ~ 0.0014 | 0.141~0.857 | \|d\| ≤ 0.82 | **不显著**（整段平台） |
| γ_0 = 0.8 / 0.85 / 0.95 vs 0.9 | §10.7.2 | ±0.0001 ~ 0.0023 | 0.224~0.911 | \|d\| ≤ 0.64 | **不显著**（0.8–0.95 平台） |
| 不含 comp1 的两组件组合 g6_comp2+comp3 vs baseline | §10.6 | +0.001622 | 0.278714 | 0.560006 | **不显著** |

**读法**：
1. 上述"不显著"项目均落在 ml_r30 5-seed 噪声带内，**配对样本下亦无法区分**。审稿人若质疑"这些组件到底有无贡献"，本批次数据无法支持显著结论。
2. 与之对照，**显著**的项目均为：含 comp1 的所有 arm（comp1、comp1&2、g6_comp1+comp3、comp123、§10.2 中"公式 vs 手工/SNR"、§10.7.2 γ_0=0.99 跌出噪声、§10.8 g8_stacked 的小幅但显著）。
3. 关于课程方向反转（§10.5）：本数据下**不能**支持"难度序必须正确"的强结论——反转课程与正向课程未拉出显著差，详见 §10.5 解读。

---

### §10.10 文件清单

- `experiments/results/p6/P6_raw.csv` — 85 行 OK 长表（17 变体 × 5 seeds，含 wall_sec / peak_mem_mb / note）。
- `experiments/results/P6_ablation.csv` — 25 行配对 t 长表（9 组 × 若干 pair）。
- `experiments/results/p6/figs/p6_sensitivity.png` — κ_0 / γ_0 双子图敏感性曲线。
- `experiments/results/p6/p6_report_stdout.txt` — `p6_report.py` 的完整 stdout（组表 + 两两差矩阵）。
- `experiments/logs/p6_dual_orchestrator.out` — 编排器主日志，末行 `P6_DUAL_ORCHESTRATOR_DONE`。
- `experiments/logs/p6_<variant>_<seed>.log` — 85 个子 run 独立日志。
- `experiments/p6_dual_orchestrator.py` — 编排器（不修改）。
- `experiments/p6_report.py` — 报告脚本（不修改）。

---

### §10.11 Reviewer 审计与统计仲裁结论

> 签署：Reviewer（科研审计与统计仲裁员），2026-09-08。审计范围为只读核验 + 独立复算，未改动任何实验代码与原始 CSV。

#### 1. 真实性核验（抽样 7/85，全一致）

随机抽 7 个 (variant, seed)，逐位比对日志尾部 `FINAL test auc` 与 `P6_raw.csv`：

| 样本 | 日志值 = CSV 值（逐位一致） |
|---|---|
| g1_uniform_gamma / 2024 | 0.7236295830563022 |
| g2_snr_gamma / 2023 | 0.7270172530818306 |
| g3_beta_perdomain_nocur / 2025 | 0.7249355103954831 |
| g5_reverse_curriculum / 2026 | 0.7204022639555729 |
| g7_gamma0_0.99 / 2022 | 0.728144136899606 |
| g7_kappa0_2.0 / 2025 | 0.7466190637595296 |
| g8_stacked / 2023 | 0.7267435237995553 |

另做全量结构核验：85 行 = 17 variant × 5 seed（2022–2026），每 (variant,seed) 恰好 1 行无重复，status 全 OK（0 行非 OK），日志目录恰有 85 个 run 日志 + 1 个编排器日志。无篡改痕迹。

#### 2. 统计复算（独立 scipy 复算 6 组，全部命中）

用 venv scipy 独立重建 arm 表并复算配对 t（含任务指定的 4 组 + 2 组抽查），与 `P6_ablation.csv` 对比，diff/t/p/d 全部逐位一致（偏差 < 1e-6，远优于 1e-4 容差）：

| 检验 | 复算 diff / t / p / d | CSV | 一致 |
|---|---|---|---|
| g1 comp1 vs baseline | 0.019329 / 22.726823 / 0.000022 / 10.163744 | 同左 | ✓ |
| g1 uniform vs baseline | 0.001380 / 0.897732 / 0.420079 / 0.401478 | 同左 | ✓ |
| g7 gamma0_0.99 vs comp1 | -0.013955 / -8.644758 / 0.000985 / -3.866053 | 同左 | ✓ |
| g8 stacked vs baseline | 0.007379 / 4.137588 / 0.014405 / 1.850386 | 同左 | ✓ |
| g5 reverse vs comp2 | -0.001187 / -0.573620 / 0.596935 / -0.256531 | 同左 | ✓ |
| g7 kappa0_2.0 vs comp1_comp2 | -0.000190 / -0.234407 / 0.826179 / -0.104830 | 同左 | ✓ |

#### 3. 方法学审查（`p6_report.py`）：通过

- 配对方式：按 seed 交集配对（`paired_stats` 取 `set(a) & set(b)` 排序后对齐），本批 n=5 全满配，正确。
- ddof=1 用于 arm std 与 diff std；t/p 用 `scipy.stats.ttest_rel`（默认双侧）；Cohen's d = mean(diff)/std(diff, ddof=1)，即标准配对 d_z。均正确。
- 敏感性图参照线：`plot_sensitivity` 取 `arms["baseline"]` 均值（复算 = 0.722514）作虚线，与 PNG 中 `baseline = 0.7225` 一致；κ0 图五点、γ0 图五点与 CSV 均值吻合，γ0=0.99 悬崖可见。正确。
- 提示（非错误）：25 个配对检验未做多重比较校正。Bonferroni（α=0.002）下，除 §10.8 g8_stacked（p=0.0144，仅名义显著）外，全部"显著"断言仍成立；§10.7.2 已将 γ0=0.99 标注"单次配对测试"，口径诚实。

#### 4. 结论仲裁

- **§10.1–§10.4、§10.6、§10.9 全部"显著/不显著"断言**：与复算数据一致，**成立**。
- **§10.5 课程方向反转不显著**：**成立**，且报告措辞恰当。仲裁意见：该结果对"课程收益绑定难度序"命题**既非证伪也非证实——是无信息（检验力不足）**。n=5 下的不显著不能证伪强形式；更根本的是 comp2 单独相对 baseline 本身不显著（p=0.657），主效应不可检时调节效应（方向）无从检验。报告 §10.5/§10.9 已如实表述为"不能用此实验区分""不能支持强结论"，未越界，无需修正。
- **§10.8 Adam 关系**：**成立**。替换（comp1, +0.0193, d=10.16）显著优于叠加（g8_stacked, +0.0074, d=1.85），足以回答"EMA 是否只是重复 Adam β1"——若只是重复，替换 β1 应无增益，实测相反；叠加仍保留显著小增益说明 EMA 含有 β1 之外的信息。补充：报告未直接报告 comp1 vs g8_stacked 的配对 t，Reviewer 复算得 diff = +0.011949、t = 9.197632、p = 0.000776、d = 4.113306，**直接显著**，比报告"从量级判断"的保守措辞更强；建议 Runner 可将此数补入 §10.8（非必须修正项）。另注意 g8_stacked 的 p=0.0144 在多重校正下仅为名义显著，§10.8 断言"显著但效应量小一个数量级"可接受。
- **§10.7 平台区宽度**：**成立**。κ0 ∈ [0.1, 2.0]（20×）四点对 0.5 全不显著（p ≥ 0.141）、均值极差 0.0015 处于噪声带，"整段 20 倍为有效平台区"被数据支持；γ0 ∈ [0.8, 0.95] 对 0.9 全不显著（p ≥ 0.224）而 0.99 显著跌落（p = 0.000985），"平台右端点 = 0.95"被支持，且报告已如实标注 0.95 处的负趋势（-0.0023, d = -0.64）。

#### 5. 总裁定

台账真实、统计产物可复现、脚本方法学正确、报告全部显著性断言与数据一致。**无需 Runner 修正的硬性问题**；唯二建议（非强制）：① §10.8 可补 comp1 vs g8_stacked 直接配对检验（p = 0.000776）；② 全文涉及 25 次配对检验，可在方法说明处补一句多重比较口径（Bonferroni 下除 g8_stacked 外结论不变）。

---

## §11 P7 泛化与 D/T 扩展报告（`experiments/reports/P7_generalization.md` 全文）

# P7 泛化与 D/T 扩展报告

> 编排：`experiments/p7_orchestrator.py --concurrency 2`（30 s 错峰、断点可续、按 `experiments/results/p7/P7_backbones.csv` 已有行去重）。
> Runner：`experiments/p7_run.py`；分析：`experiments/p7_report.py`（幂等；stdout 已落 `experiments/results/p7/p7_report_stdout.txt`）。
> 报告完成时间：2026-09-08；任务分配：D/T 5-domain 由 P7 编排器生成；组件 1 移植与 backbone 复现同批跑完；STAR 因复现差距过大排除出组件 1 移植结论（详见 §11.D.1）。sharebottom 原跑批窗口内因 `ShareBottom_Layer` 切片 bug 全 FAILED，于本次补跑（10/10 OK，none + per_domain_only 各 5 seeds），与 mmoe/ple/star 三 backbone 不在同一跑批窗口；§A/§B 中 sharebottom 的复现与配对结论仅在 sharebottom 自己的 none vs per_domain_only 窗口内自洽，与其余 backbone 的横向比较仅作复现差距量级参考。
> 执行人：Runner（实验调度与记录官）；组件实现层与 P0–P5 同人。

---

### §11.0 backbone 选型说明

候选来自 `p7.md` 第 4 行：`PLE、MMoE、AdaTT、STAR`。本项目最终选定 `mmoe / ple / star / sharebottom` 四个 backbone，理由按 `experiments/p7_models.py` 头注释的来源事实复述：

- **MMoE / PLE**：层数学逐字取自 FuxiCTR v2.3.9 的 `model_zoo/multitask/{MMoE,PLE}`（Apache-2.0），仅把 FuxiCTR 自带的 `BaseModel` 替换为 MDMTRec 同构的 embedding/tower 包装以消费同输入张量；层前向不变。FuxiCTR 包源码已存档于 `experiments/tmp_pip/FuxiCTR-2.3.9/`（`p5_orchestrator.py` / 此前 p0 期间 `p0.md` 已多次使用同一份 codeload 下载）。
- **ShareBottom**：FuxiCTR v2.3.9 原版（`model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`）是**单个** `MLP_Block` 共享底部（默认 [64,64,64]）+ `num_tasks` 个 tower，**不存在 num_experts 参数，也不存在对 expert 输出取平均**。本仓库 P7 移植因 `Backbone` 包装器将 `num_tasks = D*T` 传入层（原 bug：对 expert 维按 task 索引切片导致 IndexError），将层改写为 `num_experts` 个并行 `MLPBlock` + 沿 expert 维 mean 聚合成单一共享底部表示 + `[shared] * num_tasks` 广播。该实现**是仓库自引入的适配**，**不是 FuxiCTR ShareBottom 的逐字移植**；底部参数量约为原单 MLP 的 `num_experts` 倍。语义（单共享底部喂所有任务头）保持，层数学不保持。详见 §11.D.2。
- **STAR**：自实现，遵循 L. Sheng et al., "One Model to Serve All: Star Topology Adaptive Recommender", RecSys 2021 的 per-domain `W_d` 逐元素乘共享 `W_0` 再加 per-domain bias 结构。结构上与 MDMTRec 自带的单层 star fusion（`m3oe/MDMTRec/models/multi_domain.py:151-153`）同源——后者是同一拓扑的单层实例；本实现为多层 FCN。
- **AdaTT**：候选列表中给出但**未入选**。原因：`p7_models.py` 头注释未收录 AdaTT 实现，且本仓库当前会话窗口未发现可靠的 Apache-2.0 / MIT 开源 PyTorch 实现可直接借用（如需新增 backbone，须先完成实现+冒烟+1 配置复现才能叠加我们的组件）；为不超出 P7 编排器已固化的 60 任务清单，本次不纳入；如后续需要，按 p7.md 约束（先复现一篇原报告结果再叠加组件）单独追加。
- **复现验证门控（p7.md 约束，定性）**：先在每个 backbone 上用 momentum=none 跑完整 50 epoch × 5 seeds，对照 M3oE Table 2 的 MDMT 行（M3oE-MDMT 76.35 / PLE-MDMT 75.68 / STAR 76.08 / ShBot-MDMT 76.26），复现差距过大者按 p7.md "如实在报告中写、无法复现则排除"的定性规则排除出组件 1 移植结论。p7.md 仅规定定性门控、未规定任何数值阈值；§A 的"±1 点内"等数值化判定为 Runner 对已得数据的事后归纳。
- **复现严格性差异提示**：mmoe / ple 为 FuxiCTR 逐字移植（参数对齐、形状对齐），sharebottom 为仓库自引入的语义适配（4-expert + mean 聚合，参数对齐、形状适配），star 为自实现。**sharebottom 的"复现成立"判定严格性弱于 mmoe / ple**，仅作为"语义一致的复现参照"采信，不与 mmoe / ple 等量齐观。

---

### §11.A 跨 backbone 复现对照表（D=3, momentum=none, 5 seeds）

`test_auc_mean` 列是 `evaluate_multi_domain`（`m3oe/MDMTRec/trainer/ctr_trainer.py:228` 等）对全测试集（两任务合并）的 **pooled AUC**（与 P5 `P5_raw.csv` `test_auc_mean` 同口径），并非 6 个 per-(d,t) 格的均值（逐行核验 60/60 行二者均不等）。单位为原始比例（×1），与 P5/P0 同口径。

| backbone | n | mean | std | M3oE Table 2 | diff (ours−paper)*100 | P5_mdmtoe_ref (ml_orig) | δ vs P5_mdmtoe*100 | 复现判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mmoe | 5 | 0.767107 | 0.001336 | 76.35 | +0.360740 | 0.770098 ± 0.001776 | -0.299051 | **成立**（与论文差 +0.36 AUC 点，方向为我们高于论文；5 seeds 稳定，std 0.00134） |
| ple | 5 | 0.764977 | 0.001610 | 75.68 | +0.817680 | 0.770098 ± 0.001776 | -0.512111 | **成立**（与论文差 +0.82 AUC 点，方向为我们高于论文；5 seeds 稳定，std 0.00161） |
| star | 5 | 0.533898 | 0.006611 | 76.08 | -22.690200 | 0.770098 ± 0.001776 | -23.619991 | **不成立**（远低于论文 76.08，差 22.69 点；详见 §11.D.1） |
| sharebottom | 5 | 0.769253 | 0.001345 | 76.26 | +0.665260 | 0.770098 ± 0.001776 | -0.084531 | **成立**（与论文差 +0.67 AUC 点，方向为我们高于论文；补跑窗口，5 seeds × 2 momentum 全 OK；详见 §11.D.2 与 §11.0 复现严格性提示） |
| mdmtoe (P5 in-repo) | 5 | 0.770098 | 0.001776 | — | — | — | — | 作为基准参照（非本表对照对象） |

> 注：
> 1. `M3oE Table 2` 列按 `p7.md` / `exp.md` 使用的"原始比例×100"口径展示（76.35 即 0.7635）；论文中报告为 AUC*100。
> 2. `diff` 列符号约定：`diff (ours−paper)*100 = (mean − paper_val) × 100`，正值表示我们高于论文值。mmoe / ple / sharebottom 三者 `diff` 均为正（+0.36 / +0.82 / +0.67），且 95% CI 下界（按 t 分布 n=5 算：mean ± 2.776 × std/√5；mmoe [76.45,76.88] / ple [76.30,76.70] / sharebottom [76.76,77.09]）**均高于**论文值 76.35 / 75.68 / 76.26——即我们**显著高于**论文参照值。这与"覆盖论文值"方向相反但语义更强：差距为正且统计显著，判定不受影响。
> 3. 论文 AUC 口径（MDMT 设定下 per-(d,t) 10 格平均）与本表 pooled AUC 口径不完全一致（MDMT 设定下两任务合并的全局 ROC-AUC），二者仅作量级参照；正负号与小差距数值不应直接当作同口径绝对比较。`δ vs P5_mdmtoe` 列为我们的 backbone 相对仓库自带 MDMTRec（P5 ml_orig baseline）同口径的差距。
> 4. `p7.md` 仅规定**定性**复现门控（"先复现原报告结果，无法复现则报告差距并排除"），**没有规定任何数值阈值**；下文的"±1 点内"等表述是 Runner 对已得数据的事后归纳。

**判定逻辑（事后陈述，非 p7.md 规定）**：基于"与论文参照值同量级（±1 点内，方向不限）且 5 seeds 稳定（std ≲ 0.002）"两条事后归纳做分类——
- **成立**（mmoe / ple / sharebottom）：与论文参照值同量级，5 seeds 全部 OK，std 0.001–0.002 之间，无单点跳变。
- **不成立**（star）：与论文参照值差 22.69 点，远大于 mmoe/ple/sharebottom 的 0.36–0.82 点（差近 30 倍），且冒烟记录首 epoch 即 ~0.53 稳定停留该水平 → 非欠训练/欠拟合。按 p7.md "如实在报告中写、无法复现则排除"的定性规则排除出组件 1 移植结论。

sharebottom 的"成立"判定需附加 §11.0 复现严格性提示：sharebottom 的 backbone 实现是仓库自引入的 4-expert + mean 聚合适配（详见 §11.D.2），并非 FuxiCTR ShareBottom 的逐字移植，**复现严格性弱于 mmoe / ple**；本行作为"语义一致的复现参照"采信，不与 mmoe/ple 等量齐观。

---

### §11.B 组件 1 移植对照表（D=3, paired-t: per_domain_only vs none, 5 seeds）

| backbone | n | diff (mean_A − mean_B) | t | p (双侧) | Cohen's d | 结论 |
|---|---:|---:|---:|---:|---:|---|
| mmoe | 5 | +0.002978 | 5.182900 | 0.006593 | 2.317863 | **显著正效应** |
| ple | 5 | +0.005538 | 5.319129 | 0.006009 | 2.378787 | **显著正效应** |
| star | 5 | -0.006649 | -3.268831 | 0.030823 | -1.461866 | **不纳入结论**：STAR backbone 在 §A 中判定为"复现不成立"（mean ≈ 0.534，远低于论文 76.08），按 p7.md 与本报告 §11.0 复现门控规则，本 backbone 上的组件 1 配对差异不具有方法学意义。 |
| sharebottom | 5 | +0.004012 | 4.616984 | 0.009904 | 2.064778 | **显著正效应**（补跑窗口；p=0.0099，d=2.06） |

> 配对方式：按 seed 配对，`a = test_auc_mean(per_domain_only, seed)`，`b = test_auc_mean(none, seed)`；`diff = a − b`；`stats.ttest_rel(a, b)`（双侧）；`d = diff.mean() / diff.std(ddof=1)`。所有数值来自 `p7_report.py` stdout（`experiments/results/p7/p7_report_stdout.txt` §B）。
>
> **结论**：在 §A 判定为"成立"的 backbone 上（mmoe、ple、sharebottom），组件 1 移植均给出**显著正效应**（p < 0.01，d > 2），与 P5 强不均衡档观察一致——这三个 backbone 都属于"参数共享在所有 (D, T) 槽位"的多任务结构（mmoe 共享 expert；ple 共享 expert；sharebottom 共享 bottom），per-domain 动量改造直接作用于这些共享参数，与 MDMTRec 的实现路径同构。注：sharebottom 的 none vs per_domain_only 配对比较在 sharebottom 自己的补跑窗口内自洽，与 mmoe/ple 的配对不构成跨窗口横向比较（仅在 §A 复现差距量级上可比）。

---

### §11.C D/T 扩展结果（D=5, T=2, MovieLens 按年龄切 5 域）

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

### §11.D 方法失效或不适用的场景

#### §11.D.1 STAR backbone 复现失败 — 已排除出组件 1 移植结论

- 5 seeds × 2 momentum 共 10/10 OK（程序完整跑完 50 epoch 早停，未崩溃），但 mean test AUC 仅 0.533898 ± 0.006611；论文 STAR 报告值 76.08 → 差距 -22.69 AUC 点（绝对值），与 §A 成立三 backbone（差 0.36–0.82 点）差近 30 倍。
- 已知冒烟记录（`experiments/results/p7_smoke/P7_backbones.csv`）首 epoch 即 ~0.53，完整 50 epoch 仍停留该水平 → 不是欠拟合或欠训练。
- 诚实结论：STAR 自实现（在 M3oE star 单层基础上扩展为多层 FCN）**无法复现 RecSys 2021 论文在 MovieLens MDMT 设定下的报告水平**。按 p7.md "若某 backbone 无法复现其原报告结果，报告差距并将其排除，不要在未验证的实现上做结论" 的定性规则，STAR 不进入 §B 的方法学结论；其 CSV 数据保留用于将来实现修正后的复算。

#### §11.D.2 sharebottom backbone 切片 bug — 已修复并补跑完成；复现严格性弱于 mmoe/ple

- 原 5 seeds × 2 momentum 共 10/10 FAILED（见 `experiments/logs/p7_orchestrator.out` 中 `[done] p7 sharebottom ... FAILED last='IndexError: index 4 is out of bounds for dimension 1 with size 4'`），wallclock 10s 即崩溃。
- 根因：`p7_models.py` 原 `ShareBottom_Layer.forward` 返回 `[expert_output[:, i, :] for i in range(self.num_tasks)]`，其中 `expert_output` 在 dim 1 上大小为 `num_experts=4`，但 `num_tasks = D*T = 6`（Backbone 构造处 `n_task = domain_num * task_num`）。FuxiCTR 原版 ShareBottom 的设计意图是"单个共享 MLP 底部 + 每 task 一个 tower"，**不是**"per task 切 expert"。
- FuxiCTR v2.3.9 原版事实（对照 `experiments/tmp_pip/FuxiCTR-2.3.9/model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`）：单个 `MLP_Block`（默认 [64,64,64]）作共享底部，输出 `num_tasks` 个 tower；**无 num_experts 参数、无 expert 维、无任何对 expert 输出取平均的操作**。故"原版即对所有 expert 输出做平均/共享到各 task"系错误陈述（已被 Reviewer 核验）。
- 修复方式（Coder，已落地 `experiments/p7_models.py:109-125` 及头注释）：保留 `num_experts` 个并行 expert `MLPBlock`（`torch.stack` on dim=1），把 `dim=1` 视为 expert 维，对 expert 维取 mean 聚合成单一共享底部表示 `shared`（shape `(batch, tower_input_dim)`），再 `[shared] * num_tasks` 广播到所有 tower 输入。**该修复是仓库自引入的适配**（为消化 `Backbone` 包装器传 `num_tasks = D*T` 而保留 `num_experts` 形参），**不是 FuxiCTR ShareBottom 的逐字移植**；底部参数量约为 FuxiCTR 单 MLP 的 4 倍。语义（单共享底部喂所有任务头）与 FuxiCTR 一致；层数学不一致。
- 冒烟与补跑：1-epoch 冒烟（validation auc 0.7365）通过；本次编排器以 `--concurrency 2`、30s 错峰重跑 10 任务，10/10 OK（wall 300–445s/任务）；§A/§B 数值与配对均已落表。
- 复现严格性提示：sharebottom 的"成立"判定基于**仓库自引入的 4-expert + mean 聚合**实现（与 FuxiCTR 单 MLP 共享底部参数量差 4 倍），与 mmoe / ple 的 FuxiCTR 逐字移植不可同口径比较；本行作为"语义一致的复现参照"采信，**不与 mmoe / ple 等量齐观**。`p7_models.py` 头注释已由 Coder 修正（移除了对 ShareBottom 的"layer math is unchanged"声明，明确标注为"repo-local adaptation"）。
- 诚实标注：补跑窗口与 mmoe/ple/star 的原跑批窗口**不重合**（不同日期、不同随机种子次序、不同 GPU 温度），故 sharebottom 在 §A 中的复现判定仅基于自己窗口内的 5 seeds 协议；§B 中 sharebottom 的 none vs per_domain_only 配对在补跑窗口内自洽（同一窗口、同一 seed 序列），与 mmoe/ple 的配对不构成跨窗口横向比较。结论按 5 seeds × 2 momentum 全量协议给出（与 p7.md 与 exp.md 的复现门控一致）。

#### §11.D.3 组件 2（域条件课程）对 mmoe / ple / star / sharebottom 全部不适用

- 组件 2（p4_run.py 实现、`MDMTRecBeta`）需要一个可调的跨域融合权重标量（per-domain `α_d` / per-task `β_t` / 课程 sigmoid）。mmoe/ple 的 gate 输出 softmax over experts → 没有"标量融合"自由度；star 的 domain weight 是逐元素乘矩阵（向量化的 per-channel 权重）→ 不能坍缩为单一标量；sharebottom 无任何跨域融合机制。
- 因此本报告 §B 只能验证组件 1（per-domain 动量）的可移植性；组件 2 在这些 backbone 上的"不存在性"已在 p7.md 与 exp.md §"剩余未完成的实验与任务" C.10 明确，不强行构造。

#### §11.D.4 组件 2 在 D=5 下的进一步审视

- §C 中 mdmtoe backbone 在 D=5 下走的就是组件 1 路径；组件 2 课程功能未在 P7 中重新测（仍是 MDMTRec 的 wrapper）。这与 exp.md §"剩余" C.10 中"组件 2 在这些 backbone 上不适用"一致；P6 消融（ml_r30）才是组件 2 跨档敏感性扫描的主战场。

#### §11.D.5 GPU 显存异常

- P7 跑批过程中观察到单 run 阶段峰值显存一度达到 7.17 GB（`nvidia-smi` 抽样：2026-09-08 15:48），逼近 8 GB 笔记本卡上限；并发 2 仍跑完（未见 OOM 报错），但建议后续若并发提升需先在空闲状态下做 1 run 冒烟验证。

---

### §11.E 交付清单

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

### §11.F Reviewer 审计与统计仲裁结论

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

**3. ShareBottom 修复的语义核验（裁定：语义声明部分不成立，需修正表述；复现结论可采信）**。对照 `experiments/tmp_pip/FuxiCTR-2.3.9/model_zoo/multitask/ShareBottom/src/ShareBottom.py:52-66`：原版为**单个** `MLP_Block` 共享 bottom（默认 [64,64,64]）+ `num_tasks` 个 tower，**不存在 num_experts，也不存在"对 expert 输出取平均"**。故：① 修复后"单一共享底部表示广播到全部 tower"在架构语义上与原版一致，此部分声明成立；② 报告 §11.D.2 括号内"其原版即对所有 expert 输出做平均/共享到各 task"**与原版源码不符，系错误陈述**；③ "4 个并行 expert 取 mean 聚合"是本仓库修复自引入的机制，原版无此结构，`p7_models.py` 头注释对 ShareBottom 的 "layer math is unchanged / verbatim" 表述自修复起不再成立。影响评估：修复后的底部参数量约为单 MLP 的 4 倍，并非 FuxiCTR ShareBottom 的逐字移植，"复现验证"的严格性弱于 mmoe/ple；但核心语义（共享底部喂所有任务头）保持，且 §A 复现差距 +0.665 点与 mmoe(+0.36)/ple(+0.82) 同量级、与 star(-22.69) 差两个数量级，复现门控实证上仍通过。裁定：sharebottom 复现"成立"的判定可采信，但 §11.D.2 与 §11.0 的语义表述必须修正。

**4. §A 复现判定仲裁（裁定：判定本身成立；三处表述需修正）**。
- 差距计算数值正确，但表头 `diff (paper−ours)*100` 与 `p7_report.py` 实际计算（ours−paper，正值=我们高于论文）**符号标注相反**，需修正。
- "95% CI 覆盖论文值"对 mmoe/ple/sharebottom **均不成立**：复算 95% CI 分别为 [76.545, 76.877] / [76.298, 76.698] / [76.758, 77.092]，论文值 76.35/75.68/76.26 全部低于 CI 下界（我们显著高于论文）。该句需删除或改写；事实上"显著高于论文"比"覆盖"更强，不影响"成立"结论。
- 判定阈值问题：`p7.md` 仅规定定性门控（"先复现原报告结果，无法复现则报告差距并排除"），**没有任何数值阈值**；§A 的"≤0.76 点"、§11.D.1 的"p7.md 排除阈值 0.70"、"≥20 点"均为 Runner 自定且误归于 p7.md；且 ple 差 +0.82 已超出其自定 0.76 阈值仍判成立，阈值内部不一致。尽管如此，三个"成立"与一个"不成立并排除"的判定在 p7.md 定性规则下稳健（差距 0.36–0.82 点 vs 22.69 点），裁定维持。
- 另：§A 引言"测试集 6 项平均 test AUC"表述不准确——`test_auc_mean` 是 `evaluate_multi_domain`（ctr_trainer.py:228）对全测试集（两任务合并）的 pooled AUC，并非 6 个 per-(d,t) 格的均值（逐行核验 60/60 行二者均不等）。因与 P5 同口径，横向比较不受影响；措辞需修正。

**5. §B 跨窗口可比性仲裁（裁定：诚实标注充分，sharebottom 配对结论可采信）**。窗口不重合在头部第 5 行、§B 注、§11.D.2 三处披露，限定语（"窗口内自洽"、"横向比较仅作复现差距量级参考"）与数据事实一致（`p7_orchestrator.out` 确认原窗口 sharebottom 10/10 FAILED `IndexError`，补跑 10/10 OK）。按 seed 配对的 t 检验只要求同一 backbone 两 momentum 臂同窗口同 seed 序列，sharebottom 满足，裁定其 §B 结论（p=0.0099，d=2.06，显著正效应）可采信；跨窗口横向比较报告已自我限定，裁定充分。

**6. §C D/T 扩展仲裁（裁定：数据链成立；"任意 (D,T)"属过度宣称，需收敛措辞）**。`P7_d5_per_dt_matrix.csv` 与 `P7_backbones.csv` 原始 auc_d* 列逐格一致（max|diff| = 0.0），`_p7_dt_extract.py` 重构精确；`p7_report.py` 的 D=5 矩阵读取确仍写死 D=3 列名（stdout 有 d3/d4 NaN 警告），报告已如实披露，且 §C 均值与配对取自 test_auc_mean 列、不受该缺陷影响。但实证仅覆盖 (D=3,T=2) 与 (D=5,T=2) 两个点，且 mmoe 在 D=5 下仅名义显著（p=0.0349，d=1.40）：证据支持"组件 1 在 D=3→D=5 路径有效"，**不足以支持"方法在任意 (D,T) 下成立"**；T 未做任何变化（恒为 2）。§C 结论句中"本节实证 D=3 → D=5 路径有效"是恰当口径，标题级"方法在 D=5 下成立"可保留，但任何向"任意 (D,T)"的外推需删改。

**总裁定**：数据真实、统计正确、各节判定方向维持（§A mmoe/ple/sharebottom 成立、star 不成立并排除；§B 三 backbone 显著正效应；§C D=5 路径有效）。需 Runner 修正的表述（不影响数值与结论）：① §A 表头 diff 符号；② §A "95% CI 覆盖论文值"；③ §A "6 项平均"措辞；④ §A/§11.D.1 阈值误归于 p7.md；⑤ §11.D.2 ShareBottom "原版即对 expert 输出做平均"及 §11.0/`p7_models.py` 头注释的 verbatim 表述；⑥ §C 向"任意 (D,T)"的外推措辞。另建议（非阻塞）：Runner 修正 `p7_report.py` 的 D=5 列读取，使流水线自洽、消除对 `_p7_dt_extract.py` 外部重构的依赖。

---

## §12 test logloss 留底情况（来自 exp.md §6.7，整合自源文件相关声明）

- **P0**：`P0_repro.csv` 含 `test_logloss` 列（per-(d,t) 全精度）。5 seeds × 6 (d,t) = 30 个原生 6 位小数（用于 §2.2 报告中的 r0/d0/d1 域平均 loss 计算）。
- **P2 / P5 / P6 / P7**：`p4_run.py` 的末尾两行（`p4_run.py:701-702`）**仅输出 test AUC**，不输出 test logloss：
  ```
  [p4] FINAL per-(d,t) test auc: [...]
  [p4] FINAL test auc: ...
  ```
  评估集日志里只有逐 epoch 的 `validation: auc / log loss`（验证集 BCE），没有 test logloss。`P2_baselines.csv` 的 `test_logloss` 列全部为空，`P5_raw.csv` / `P6_raw.csv` / `P7_backbones.csv` 三个 CSV **没有该列**；`P2_baselines.md §4.6` 已诚实标注"P2 插桩只记录了 AUC；该缺口已在 P4/P5 的插桩中补上"——但 P4/P5 实际未补，P5/P6/P7 评估脚本仍只输出 AUC。
- **结论**：P5 / P6 / P7 主矩阵的 **test logloss 未在台账中留存**（评估脚本仅输出 AUC），如需绝对值须改 `p4_run.py` 加 test logloss 计算 + 重跑评估。**严禁凭 P0/P2 的零星 test logloss 推断 P5 主档数字**——P0 与 P5 训练集 / 早停 / 模型超参均不同。

---

## §13 Master 批判性复审与机理重归因（2026-09-08，来自 exp.md §10 全文）

> 性质：不是数字正确性复核（Runner 已做，5 处修正），而是基于全部已签署报告与代码语义对结论链的批判性审视。以下每条均标注证据出处。

### §13.1 机理重归因（正向发现，重构组件 1 的生效故事）

P6 三臂合证 + EMA 代码语义（`p4_run.py:242`：`m_d ← γ_d·m_d + (1−γ_d)·g_d`，无偏校正，`shared.grad = Σ_d m_d`）：
- g1 均一 γ（γ_d ≡ γ_0=0.9，全域无分档）：+0.001380，**不显著**；
- g2-snr（SNR≈域间相等 → 实际也是近均一 γ）：显著差于公式（d=−5.38）；
- 公式 γ_d（大域 0.9、小域 →0.99 的单调档差）：+0.019329，d=10.16。
- **结论：增益完全由"小域更重平滑"的档差结构承载**。机理上与 KuaiRand 大域负迁移收窄互洽：小域梯度的高频波动被重 EMA 衰减后才进入共享参数，大域信号主导共享表示——**它保护的其实是大域免受小域梯度方差污染，而非补偿小域**。这与 P5 发现 5（"增益是大域小域共享的，不是单纯补小域"，§7.6）完全一致。设计前提（小域 SNR 更低）错了，但处方歪打正着——这是"P1 证伪前提 × P5/P6 实证有效"悖论的收敛解释。

### §13.2 证据链薄弱点与替代解释（按严重度排序）

1. **r3 凹陷的排除理由无效**（最重要）：FINAL 报告 §9.3 与 §7.6 以"3/3 seeds 同号、窗口伪影可能性低"淡化 r3 凹陷——**该推理不成立**：窗口漂移是系统性偏移，对所有 seed 同向，"全同号"恰恰是窗口伪影的表现形式而非反证。且 r3 为跨窗混合矩阵（baseline×3 属早期窗口、comp1 属 9/8 窗口），配对本身跨窗。凹陷幅度 −0.0012 处于实测漂移幅度（~0.002）之内。**重定性：r3 凹陷"无法与窗口漂移区分"，既不能作机理证据，也不能排除为纯伪影**；唯一干净裁定方式是同窗口重跑 r3 全部 6 臂（~18 runs）。
2. **"增益随不均衡度放大"与训练集总规模混杂**：r30/r100 档在下采样制造不均衡的同时使总训练量锐减（baseline 0.7694→0.7268/0.6953）；comp1 增益与 baseline 跌幅同步放大，跨档趋势部分反映"抵抗数据稀缺损伤"而非纯不均衡效应。同档内方法间比较不受影响（总量恒定）；跨档叙事需带此限定。Reviewer 已裁定单调性不成立，本条进一步指出趋势轴本身不纯。
3. **κ0 平台区（20×）的证据强度被高估**：κ0 控制的是 comp2——其主效应与噪声不可区分（p=0.657）。"无效应组件的强度旋钮不敏感"近乎恒真，不能正面回答 p6.md 引用的 CoHeat 式质疑。真正承载"本方法超参不敏感"断言的只有 γ0 平台区 [0.8, 0.95] + 悬崖点 0.99。结论 7 已加限定。
4. **"EMA 替换而非叠加 β1"存在未排除的替代解释**：g8_stacked = Adam β1(0.9) + 我们的 EMA = 总动量预算翻倍；comp1 vs stacked 之差（+0.0119）同样可由"过平滑"解释，无需诉诸"EMA 含有 β1 之外的结构信息"。审稿人问题的回答方向成立（替换显著更优），但强形式结论需半强度叠加臂（如 β1=0.45+EMA）才能锁死。记为证据边界，不改结论。
5. **KuaiRand d1t1 收窄不显著**（comp1 p=0.369）：此前 exp.md 一句话版把 d1t1 与 d1t0 并列呈现易误导为两者同等显著；已在结论 6 修正。KuaiRand 大域证据实际上只由 d1t0 单任务承载（名义显著）。
6. **口径差异**：P6（pooled AUC）与 P5 主表（六项平均）绝对水平差 ~0.004（baseline r30：0.722514 vs 0.7268），diff/p/d 两口径同向同显著性；结论 7 已标注。P7 亦用 pooled 口径（Reviewer 已裁）。
7. **P6 报告 §10.4"互为前置"措辞越界**：comp1&2 vs comp1 = +0.001638 未达可分辨阈值，"课程必须建立在差异化 EMA 之上才生效"是合理假说而非本数据支持的结论。已签署报告正文不改，在此留痕；exp.md 结论 5 的对应表述已先行修正。
8. **P7 泛化边界**：三个第三方 backbone 的移植验证均在 ml_orig（1.9:1 温和不均衡）上进行，组件 1 的强效应区（≥30:1）未在第三方 backbone 上测——"方法非 M3oE 特有"目前仅在温和不均衡域成立。
9. **ShareBottom 复现严格性**：修复后的 ShareBottom 为仓库自引入适配（4-expert mean 聚合），非 FuxiCTR 原版单 MLP 底部；其"复现成立"判定 fidelity 弱于 mmoe/ple，结论 8 与 §11 P7 报告已如实标注，此处确认该限定必要且充分。

### §13.3 事实性勘误（已落地）

- **P6_ablation.md §10.1 arm 标签错误**：`g1_uniform_gamma` 描述写"γ_d ≡ 1"，实际代码语义为 γ_d ≡ γ_0 = 0.9（`p4_run.py:203` 均一填充 γ0）。若 γ≡1，EMA 冻结（m 恒 0 → 共享参数零梯度），AUC 必崩至随机水位，与实测 0.723894 矛盾——证明实际执行的是均一 0.9。已在 §10.1 做勘误修订（仅标签，不动任何数字与签署节）。

---

## §14 已知工程坑与风险提示（整合 exp.md §8 + 交接清单 E 项）

### §14.1 已知工程坑（来自 exp.md §8）

1. **DataLoader 迭代器创建消耗全局 RNG**（`_base_seed`，num_workers=0 也一样）→ 额外评测必须 RNG 快照包裹（`p4_run.py` fit 中已做；新代码照做）。
2. **模型 forward 返回 CPU 张量**（model.py:198 未指定 device）→ 调用侧必须 `y_pred.to(device)`（原版 trainer 有；自写循环别漏）。
3. **Windows spawn 不能 pickle ~700MB 数据集** → KuaiRand 管线 num_workers=0。
4. **架构参数键名双形态**：原版 `_weight_x.deep_weights`（Weights 模块），Beta 版 `_weight_x`（Parameter）→ 用 `arch_named()` 统一解析。
5. **orchestrator 断点续跑只认 status==OK**（曾因无过滤导致 FAILED 格被跳过；已修复）。
6. **环境敏感性**见 §7.7。
7. 运行统一用 `python -u`（历史上管道缓冲曾伴随一次 DataLoader 停滞）。

### §14.2 已知风险与提示（来自 exp.md 交接清单 E 项）

- **环境窗口漂移**（§7.7）：跨会话续跑先做基线漂移检查（重跑 `--seed 2022 --epoch 50` baseline 一格对比 P5_raw）；若漂移 >0.003，需在同一窗口内重跑受影响的对照臂。（2026-09-07 复测：9/7 重跑 ml_orig baseline 2022 与 9/6 窗口 CSV 逐位一致、实测漂移为 0，探针日志 `experiments/logs/drift_ml_orig_baseline_2022.log`；Reviewer 据此裁决 KuaiRand 跨窗口结论可信。仍建议跨天续跑保留漂移检查步骤。）
- **P7 sharebottom 跨窗口风险**（2026-09-08 复盘新增）：sharebottom 为修复 `ShareBottom_Layer` 切片 bug（IndexError，原 10/10 FAILED）后的**补跑窗口**产物，与 mmoe/ple/star 的原跑批窗口**不重合**（不同日期 / 不同随机种子次序 / 不同 GPU 温度）。sharebottom 的 none vs per_domain_only 配对结论（p=0.0099 d=2.06）仅在 sharebottom 自己窗口内自洽（同一窗口、同一 seed 序列），与 mmoe/ple 的配对**不构成跨窗口横向比较**；§A 复现差距量级（+0.665 AUC 点）可参考，但严格性弱于 mmoe/ple 的 FuxiCTR 逐字移植。
- ml_r1/r3/r10 是 3 seeds（原计划允许的降级）；r3/r100 为 5 seeds；统计口径在报告里写明。
- GPU 是用户日常在用的笔记本（8GB），并发桌面负载会造成上表所述的环境漂移——不要并行跑多个训练。
- 若磁盘紧张：`m3oe/examples/model_para/` 下每次运行都会存一个 ~45MB 的 model.pth（uuid 前缀），可定期清理（结果不依赖它们，P2 的教训：它们与日志的对应关系没记录，无法追溯）。

---

## §15 文件地图与数据索引（整合自 exp.md §9 并扩展）

> 每份源报告标注"已整合进本文档 §X"——读者只读本文档即可获得全部实验记录；原始 CSV / 日志以文件引用方式索引。

| 文件 | 内容 | 已整合进 |
|---|---|---|
| `experiments/EXP_CONTEXT.md` | 共享上下文（环境/核查表/复现/各阶段摘要） | §2（环境 + 24 断言核查表）、§2.4-§2.6（关键文件清单 / 风险 / 交付索引） |
| `experiments/reports/P1_diagnosis.md` | P1 训练插桩四项诊断 | §3（全文） |
| `experiments/reports/P2_baselines.md` | P2 AutoML 贡献与 B10 bug 影响对照 | §4（全文） |
| `experiments/reports/P3_kuairand.md` | P3 KuaiRand-Pure 数据管线 | §5（全文） |
| `experiments/reports/P4_implementation.md` | P4 三组件实现与等价性测试 | §6（全文） |
| `experiments/reports/ML_P5_REPORT.md` | MovieLens 六档梯度扫尾报告（Reviewer 签署） | §7（全文，含 Reviewer 签署） |
| `experiments/reports/KUAI_STAGE1_ORACLE.md` | KuaiRand 阶段 1：单域 Oracle 真理基准 + 历史 Log 解剖（Reviewer 签署） | §8.2（全文，含 Reviewer 签署） |
| `experiments/reports/KUAI_P5_MAIN_REPORT.md` | KuaiRand P5 主矩阵实验报告（Reviewer 签署） | §8.3–§8.6（全文，含 Reviewer 签署） |
| `experiments/reports/FINAL_RESEARCH_REPORT.md` | P5 终审研究报告（Reviewer 终审签署生效 2026-09-08） | §9（全文） |
| `experiments/reports/P6_ablation.md` | P6 消融实验报告（Reviewer 签署并审计） | §10（全文，含 Reviewer 签署） |
| `experiments/reports/P7_generalization.md` | P7 泛化与 D/T 扩展报告（Reviewer 签署 + 审计勘误） | §11（全文，含 Reviewer 签署） |
| `experiments/results/P0_repro.csv` | P0 复现（30 行） | §2.2 |
| `experiments/results/p1/`（D02/D04/D06/D07_*.csv + figs/） | P1 诊断数据与图 | §3.1–§3.4 |
| `experiments/results/p2/`（P2_baselines.csv、P2_arch_traj_fixed.csv、日志） | P2 对照 | §4 |
| `experiments/results/P3_dataset_stats.csv`、`D:/CODE/data/kuairand/processed/kuai_rand.csv` | P3 | §5 |
| `experiments/results/p5/P5_raw.csv`、`P5_main.csv`、`figs/p5_trend.png`、`logs/` | P5 主矩阵（178 数据行：OK 174 = ML 144 + kuairand 30；历史 FAILED 4 条保留） | §7、§8 |
| `experiments/p1_run.py` / `p2_run.py` / `p4_run.py` / `p7_models.py` / `p7_run.py` | 各阶段 runner | §3、§4、§6、§11 |
| `experiments/p5_orchestrator.py` / `p5_oracle.py` / `p6_orchestrator.py` | 编排器（断点可续） | §7、§8、§10 |
| `experiments/dual_orchestrator.py` | 双进程并发编排器（30s 错峰，`--dataset kuairand/ml`，断点可续） | §7、§8 |
| `experiments/results/p5/P5_oracle.csv` | P5 单域 oracle 台账（kuairand 15 行 OK；ML 45 行 OK） | §7.3、§8.2.2、§8.4.1 |
| `experiments/reports/KUAI_STAGE1_ORACLE.md`、`KUAI_P5_MAIN_REPORT.md` | KuaiRand 阶段 1/2 报告（Reviewer 已签署） | §8 |
| `experiments/reports/ML_P5_REPORT.md` | MovieLens 六档梯度扫尾报告（Reviewer 已签署） | §7 |
| `experiments/reports/FINAL_RESEARCH_REPORT.md` | **最终研究报告（Reviewer 终审签署生效，2026-09-08）** | §9 |
| `experiments/p6_dual_orchestrator.py` / `p6_report.py` | P6 编排（17 变体 × 5 seeds，85/85 OK）+ 出报告 | §10 |
| `experiments/results/p6/P6_raw.csv` / `P6_ablation.csv` / `figs/p6_sensitivity.png` / `p6_report_stdout.txt` | P6 原始数据 / 配对 t 长表 / κ_0-γ_0 双子图 / 完整 stdout | §10 |
| `experiments/reports/P6_ablation.md` | **P6 消融报告（Reviewer 已签署并审计，2026-09-08）** | §10 |
| `experiments/p7_orchestrator.py` / `p7_models.py` / `p7_run.py` / `p7_report.py` | P7 编排 / backbone 库（mmoe/ple/star/sharebottom） / runner / 出报告 | §11 |
| `experiments/results/p7/P7_backbones.csv` / `P7_summary.csv` / `P7_d5_per_dt_matrix.csv` / `p7_report_stdout.txt` | P7 原始数据（60/60 OK）/ 长表 / D=5 矩阵留底 / 完整 stdout | §11 |
| `experiments/reports/P7_generalization.md` | **P7 泛化报告（Reviewer 已签署 + 审计勘误，2026-09-08）** | §11 |
| `experiments/logs/stage3_checkpoint.md` | 阶段 3/4 断点台账（含漂移探针、跑批、报表核验全记录） | §7.1、§7.4、§7.9 |
| `experiments/parse_p0.py` / `parse_p2_logs.py` / `parse_p5.py` / `p5_report.py` / `p1_analysis.py` | 解析与出报告 | §4.6、§7、§10 |
| `experiments/tests/test_equivalence.py` | P4 验收测试 | §6.2 |
| `experiments/verify_assertions.py`、`p0_seed*.log`、`requirements.lock` | P0 证据 | §2 |
| `experiments/_p7_dt_extract.py` | D=5 矩阵留底 | §11.C |
| `experiments/logs/exp_handover_primerge_20260908.md` | 整合前 exp.md 完整备份 | 整合前快照 |

> **唯一权威整合版声明**：本文档 §1–§16 为整合最终版本；源文件全部保留于 `experiments/` 供溯源。如需引用某一数字，优先从本文档 §X 取得（已逐字照搬源文件、6 位小数、与台账一致）；若与源文件有出入，**以源文件为绝对权威**并同步修订本文档对应章节。

---

## §16 交接清单（A–E 全部 ✅ 完结版）

> 执行环境：进入 `D:\CODE\phase0-3\`，用 `venv/Scripts/python.exe`；所有编排器断点可续（只补 status!=OK 的格子）。**re.md 四阶段流水线已于 2026-09-08 全部闭环（P5 两数据集主矩阵 + oracle + 三份签署报告 + 综合报表）；下表剩余项仅为 P6 消融与 P7 泛化（不在 re.md 范围内），亦已于 2026-09-08 闭环。**

### A. P5 收尾（✅ 全部完成，2026-09-08 闭环）

- ✅ KuaiRand：主矩阵 30/30 OK + oracle 15/15 OK + 两份 Reviewer 签署报告（`KUAI_STAGE1_ORACLE.md` 见 §8.2、`KUAI_P5_MAIN_REPORT.md` 见 §8.3–§8.6）。
- ✅ MovieLens：ml_r3 / ml_r10 补齐（各 18 行 OK）、全矩阵 144 行 OK 零 FAILED、ML oracle 45 行 OK、`ML_P5_REPORT.md` 经 Reviewer 签署（含机理仲裁与 oracle 口径裁决，见 §7.6、§7.9）；跨日漂移探针 0.000000。
- ✅ 综合报表：`p5_report.py` 已刷新 `P5_main.csv`（174 runs 长表）与六档趋势图 `figs/p5_trend.png`。
- 可选遗留（非阻塞）：`bootstrap_ci()` 仍未接入 `p5_report.py` 主流程；若未来需要纯迁移效应口径的 r30/r100 Δ_oracle，按 §7.9 终审签署第四节的备用规约改 `p4_run.py` 单域分支并重跑 30 格 oracle。

### B. P6 消融（✅ 已完成，2026-09-08 闭环；ml_r30 参照点，85/85 OK，Reviewer 签署报告）

**产物路径**（全部已整合进 §10）：
- `experiments/results/p6/P6_raw.csv` — 85 行 OK 长表（17 变体 × 5 seeds）
- `experiments/results/P6_ablation.csv` — 25 行配对 t 长表（9 组消融）
- `experiments/results/p6/figs/p6_sensitivity.png` — κ_0/γ_0 双子图敏感性曲线
- `experiments/results/p6/p6_report_stdout.txt` — `p6_report.py` 完整 stdout
- `experiments/reports/P6_ablation.md` — **P6 消融报告（Reviewer 已签署并审计，含统计仲裁）**
- `experiments/p6_dual_orchestrator.py` / `p6_report.py` — 编排与出报告

**关键数（6 位小数，与 §10 / `P6_ablation.csv` 逐位一致）**：comp1 vs baseline +0.019329 p=2.2e-5 d=10.16；g1 uniform_gamma 不显著（+0.001380 p=0.420 d=0.40）；g2 公式γ优于手工（d=−3.08 p=0.0023）与 SNR 反推（d=−5.38 p=0.0003）；g3 per-domain β 关课程不显著（d=−0.43）；g4 课程 vs 关课程不显著（d=0.70）；g5 反向课程不显著（d=−0.26）；g6 仅含 comp1 的组合显著（comp1&2 d=16.06、comp1+comp3 d=5.01、comp123 d=7.26）；g7 κ0∈[0.1, 2.0] 平台区 20×（4 点对 0.5 全 p≥0.141）、γ0 平台区至 0.95（0.8/0.85/0.95 vs 0.9 全 p≥0.224，0.99 显著跌落 d=−3.87 p=0.000985）；g8 Adam：comp1 +0.0193 vs stacked +0.0074，替换优于叠加（Reviewer 补算直接配对 p=0.000776 d=4.11）。

### C. P7 泛化（✅ 已完成，2026-09-08 闭环；ML only，60/60 OK，Reviewer 签署并勘误）

**产物路径**（全部已整合进 §11）：
- `experiments/results/p7/P7_backbones.csv` — 60 行 OK（12 cell × 5 seeds；含 D=3 与 D=5 全部 per-(d,t) AUC 列）
- `experiments/results/P7_summary.csv` — `p7_report.py` 长表（按 cell × section）
- `experiments/results/p7/P7_d5_per_dt_matrix.csv` — 5×2 分域 AUC 矩阵留底
- `experiments/results/p7/p7_report_stdout.txt` — §A/§B/§C 完整 stdout（本次按动态 (D,T) 识别重构后已无 NaN 警告）
- `experiments/reports/P7_generalization.md` — **P7 泛化报告（Reviewer 已签署并审计 + 勘误，含统计仲裁）**
- `experiments/p7_orchestrator.py` / `p7_models.py` / `p7_report.py` — 编排 / backbone 库 / 出报告（已重构为按行内 auc_d* 动态识别 (D, T)）

**关键数（6 位小数，与 §11 / `P7_summary.csv` 逐位一致）**：跨 backbone 复现（D=3, momentum=none, pooled AUC）mmoe 0.767107±0.001336 / ple 0.764977±0.001610 / sharebottom 0.769253±0.001345 复现成立（高于论文参照 +0.36/+0.82/+0.67 AUC 点，量级参照；sharebottom 为仓库自引入的 4-expert+mean 聚合适配，复现严格性弱于 mmoe/ple 的 FuxiCTR 逐字移植）；star 0.533898±0.006611 复现失败排除（差论文 −22.69 AUC 点，远超 mmoe/ple/sharebottom 30 倍）。组件 1 移植（D=3, per_domain_only vs none, 配对 t）：mmoe +0.002978 p=0.0066 d=2.32、ple +0.005538 p=0.0060 d=2.38、sharebottom +0.004012 p=0.0099 d=2.06 均显著。D/T 扩展（D=5, T=2）：mdmtoe +0.004904 p=0.0021 d=3.18、mmoe +0.001086 p=0.035 d=1.40；**掩码路径泛化有效**（实证覆盖 D=3、D=5 两点，T≠2 未实证）。组件 2 对 mmoe/ple/star/sharebottom **不适用**（无可调的跨域融合标量，未强行构造）。

### D. 收尾（✅ 已完成，2026-09-08）

13. ✅ 更新 `experiments/EXP_CONTEXT.md`（追加 §8 "P5–P7 最终结论摘要" 一节，详含 P5 一句话结论 + P6/P7 要点与产物路径）——已整合进 §2 与 §15。
14. ✅ P5–P7 五份签署报告（ML_P5_REPORT.md / KUAI_P5_MAIN_REPORT.md / KUAI_STAGE1_ORACLE.md / P6_ablation.md / P7_generalization.md）结论已合入本文档 §1 核心结论 5–8 与 §7–§11 全文。签署归属：P5 三份经 Reviewer 签署并由 FINAL_RESEARCH_REPORT.md 终审生效（2026-09-08，覆盖 P5）；P6/P7 两份各有独立 Reviewer 签署（2026-09-08），不在 FINAL_RESEARCH_REPORT 覆盖范围内。

### E. 已知风险与提示（详见 §14.2）

---

> **整合完成声明**：本节（§16）保留 A–E 五个清单节、产物路径与关键数均按源文件 6 位小数逐字照搬。源文件全部保留于 `experiments/` 供溯源；本文档为唯一权威整合版。任何后续数字更正请同步修订 §X 章节与对应源文件。





















