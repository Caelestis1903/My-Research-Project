# EXP_CONTEXT.md — M3oE 复现与核查上下文（P0 产出，随阶段更新）

> 本文件是整个研究项目（P0–P7）的共享上下文。后续所有阶段先读本文件。
> 所有"我们跑出的"数字均来自实际运行，与"论文报告的"数字明确区分。
> 生成时间：2026-09-06（P0）；P1/P4 产出见文末第 7 节。运行环境见第 1 节。

## 1. 环境信息

| 项 | 值 |
|---|---|
| OS | Windows 10 (build 26200), Git Bash |
| Python | 3.13.5 (Anaconda base, `D:\Programs\anaconda3`) |
| PyTorch | 2.9.0+cu126 |
| CUDA | 12.6 (driver 592.82) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8188 MiB |
| pandas / numpy / scikit-learn / tqdm | 2.2.3 / 2.1.3 / 1.6.1 / 0.4.x（完整清单见 `experiments/requirements.lock`） |
| 虚拟环境 | `D:\CODE\phase0-3\venv`（`python -m venv venv --system-site-packages`） |

**环境调整及理由（全部记录，无静默升级）：**
1. 仓库**没有** requirements.txt / setup.py（官方 GitHub 树确认），无可对照的官方依赖清单，因此不存在"过时依赖需要调整"的问题；我们按现状使用。
2. 独立 venv 以 `--system-site-packages` 创建：pip 在本机网络下安装 CUDA 版 torch（约 3 GB 轮子）实测过慢（仅 scikit-learn 即耗时 6 分钟），故复用 Anaconda base 已装好的 torch 2.9.0+cu126。仓库内残留的 `__pycache__/*.cpython-313.pyc` 证明同一 Python 3.13 此前成功运行过该仓库，版本兼容性有旁证。
3. `run_movielens.py` 的 `--dataset_path` 默认值是作者本机路径 `D:/CODE/data/ml-1m`。我们不改代码，调用时显式传 `--dataset_path D:/CODE/phase0-3/m3oe/examples/data/ml-1m`（仓库自带数据，`ml-1m.csv` 与 `ml-1m.zip` 内文件字节数一致 75,514,352，即 zip 的解压产物；MovieLens-1M 共 1,000,209 条评分，完整）。
4. 运行方式注意事项：本机（Windows + num_workers=8 的 DataLoader）曾观察到一次训练停滞——在管道缓冲/并发干扰下主进程卡在 DataLoader 的 `_shutdown_workers`（py-spy 抓栈证实：主进程 join worker、8 个 worker 空等索引队列）。以 `python -u`（无缓冲）重跑同一脚本，训练正常且逐 epoch 验证 AUC 与前台运行**逐位一致**。结论：属于运行方式问题而非代码问题；`num_workers=8` 等代码行为保持 as-published 未动。**所有正式运行均使用 `python -u` 并以前台/受控后台方式执行。**
5. `git status`：`m3oe/` 目录不是 git 仓库（无 .git），仅是官方仓库文件的拷贝；对官方仓库的状态核对通过 GitHub API 完成（commit tree `1b30217477836cac8f9bb68329a91e1c805aa4f5`）。

## 2. 代码事实核查表（P0 任务 3）

图例：✅成立 | ❌不成立 | ⚠️部分成立 | ❓未找到
核查方式：源码精读（附 文件:行号）、运行时验证（`experiments/verify_assertions.py`，纯构造无训练）、论文核对（arXiv:2404.18465v3 HTML 全文）、官方仓库核对（GitHub API）。

### A. 特征与网络维度

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| A1 | EmbeddingLayer 对每个 SparseFeature 建独立 embedding table，RandomNormal(0, 1e-4) 初始化，embed_dim=16；DenseFeature embed_dim=1 直接透传 | ✅成立 | `MDMTRec/basic/layers.py:56-62`（embed_dict 每个 feature 一张表）、`features.py:61`（默认 `initializer=RandomNormal(0, 0.0001)`）、`run_movielens.py:68`（`embed_dim=16`，未覆盖默认 initializer）、`features.py:91`（DenseFeature `embed_dim=1`）、`layers.py:91`（dense 值直接 `x[fea.name].float()` 拼接，不经过 embedding） |
| A2 | 7 sparse × 16 + 1 dense = input_dim = 113 | ✅成立 | `run_movielens.py:44-46`（dense=['age'] 1 个；sparse 7 个）、`model.py:81`（`input_dim=sum(embed_dim)`）。运行时验证：`input_dim=113` |
| A3 | fcn_dims=[512,256,256,64]，域表示输出 h_d 是 256 维 | ✅成立 | `run_movielens.py:150`；`model.py:83,94-97`（star_dim=[113,512,256]，余下 fcn_dim=[256,64]）、`model.py:105`（star_mlp=MLP_N([512,256])）、`model.py:149-154`（emb = star_mlp(emb)+skip，两者均为 256 维） |
| A4 | skip_conn 是单层 Linear(input_dim,256)+LayerNorm+ReLU | ✅成立 | `model.py:97`（`skip_conn=MLP_N([113,256])`）；MLP_N 构造 `model.py:61-65`（n-1=1 组）。运行时打印：`Linear(113→256), LayerNorm(256), ReLU` |
| A5 | 专家网络是一层 MLP：Linear(256,64)+LayerNorm(64)+ReLU；prediction tower 64→64→1 | ✅成立 | `model.py:113-115`（expert=MLP_N([256,64])，运行时打印 `Linear(256→64), LayerNorm(64), ReLU`）；`model.py:127-136`（tower=`Linear(64,64)+LayerNorm+ReLU+Linear(64,1)`） |
| A6 | domain_indicator 既作为 sparse 特征参与 embedding，又作为最后一列充当域 ID（domain_id = x[-1,:]） | ✅成立 | `run_movielens.py:46`（在 sparse_features 中）、`run_movielens.py:26`（x_used_cols 最后一列）；`model.py:140`（`domain_id = x[-1, :].clone().detach()`）。trainer 侧同样用 `x_dict[-1,:]` 取域 ID（`ctr_trainer.py:89,192`） |

### B. 融合权重与 AutoML

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

### C. 损失与训练配置

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| C1 | loss = 6 项 BCE 等权求和（`sum(torch.stack(logloss_list))`），无任务/域加权、无 load balancing、无辅助正则 | ✅成立 | `ctr_trainer.py:114`；全文件无其他 loss 项（weight_decay=1e-5 是唯一的正则，来自优化器参数） |
| C2 | 附近有被注释的可学习任务权重（形如 `# loss = sum(self.model.loss_weight() * ...)`） | ✅成立 | `ctr_trainer.py:113`：`# loss = sum(self.model.loss_weight() * torch.stack(logloss_list))` |
| C3 | 掩码分支硬编码：`if not (domain_num == 3 and task_num == 2)` 才走单一 BCE，只有 3 域 2 任务走 MDMT 掩码路径 | ✅成立 | `ctr_trainer.py:84`：原文即 `if not (domain_num == 3 and task_num == 2): loss = self.criterion(y_pred, y.float())`，else 走掩码求和路径 |
| C4 | 主优化器 Adam lr 默认 1e-2；架构优化器 Adam lr 默认 1e-3；weight_decay=1e-5 | ✅成立 | `run_movielens.py:209-210`（`learning_rate=1e-2`、`learning_rate_darts=1e-3`）、`:162`（两者均传 `weight_decay=args.weight_decay`=1e-5，`:212`）。注：trainer 内部默认 lr=1e-3（`ctr_trainer.py:50`）但被 run 脚本覆盖 |
| C5 | batch_size=4096*10=40960；epoch 上限 50；早停 patience=10，监控验证集 6 项平均 AUC | ✅成立 | `run_movielens.py:211,208,162`；`ctr_trainer.py:149-150`（`auc = sum(auc_list)/(domain_num*task_num)`）。⚠️附注：早停触发时才回滚到 best weights（`:154-157`）；若 50 epoch 跑满未触发早停，最终测试用的是**最后一轮**权重而非最优轮（`:159` 直接保存当前权重）——as-published 行为，后续阶段复现时须知晓 |
| C6 | scheduler_params 被传入但 scheduler_fn=None，调度未启用 | ✅成立 | `run_movielens.py:162`（传 `scheduler_params={"step_size":4,"gamma":0.85}`）、`:163`（`scheduler_fn=...StepLR...` 被注释）；`ctr_trainer.py:58-60`（`scheduler_fn is not None` 才建 scheduler） |
| C7 | train:val:test = 8:1:1，按行序切分而非随机打散 | ✅成立 | `run_movielens.py:84-87`（`int(len*0.8), int(len*0.9)` 连续切片）。运行时：800,167 / 100,021 / 100,021。（注意 DataLoader 的 shuffle=True 只影响训练 epoch 内的 batch 顺序，不影响切分本身） |

### D. MovieLens 预处理

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| D1 | 域切分依据 age：{1,18}→d0，{25}→d1，{35,45,50,56}→d2 | ✅成立 | `run_movielens.py:28-30`（group1/2/3）、`:34`（`map_group_indicator` 按列表序返回 0/1/2） |
| D2 | click = rating > 3，like = rating > 4 | ✅成立 | `run_movielens.py:101-112`（`convert_target`：click 阈值 3，like 阈值 4，`v > thre → 1`）、`:60-61` |
| D3 | age 同时用作域划分依据和一个输入特征 | ✅成立 | `run_movielens.py:44`（age 在 dense_features 中，MinMax 缩放后入模）、`:34,44-45`（同时生成 domain_indicator）。附注：域 ID `domain_indicator` 也是 sparse 特征之一（`:46`），即"age 信息"以两种形式进入模型 |
| D4 | 实际三域样本量与占比，与论文 Table 1（210747/395556/393906，21.07%/39.55%/39.38%）核对 | ✅一致（我们的统计与论文完全相同） | 由 `ml-1m.csv` 全量计算：d0=210,747（21.07%）、d1=395,556（39.55%）、d2=393,906（39.38%），总数 1,000,209。与论文 Table 1 逐位一致 |

### E. 仓库完整性（对官方 GitHub 仓库核对，tree `1b30217`）

| # | 断言 | 结果 | 证据 |
|---|---|---|---|
| E1 | KuaiRand-Pure 的预处理与运行脚本完全未开源 | ✅成立（对官方仓库） | 官方仓库全树（GitHub API 递归列出）无 run_kuaishou.py、无任何 KuaiRand 相关文件；README 只写 MovieLens-1M 流程。⚠️注意：**本地拷贝**多出一份 `examples/run_kuaishou.py`（P3 提示词说明系另行获得），非官方内容 |
| E2 | `__pycache__` 残留 `ablation.cpython-37.pyc` 等但对应 .py 未提交 | ✅成立 | 官方树含 `MDMTRec/models/multi_domain/__pycache__/ablation.cpython-37.pyc`、`adapter.cpython-37.pyc`、`adapter_dcn.cpython-37.pyc`、`adapter_wd.cpython-37.pyc`（另有 cpython-310 版），而 `models/multi_domain/` 下仅提交了 `model.py` 与 `__init__.py`。本地拷贝同现状 |
| E3 | 是否有 LICENSE | ❌不成立（即：没有 LICENSE） | 官方仓库树中无 LICENSE/COPYING 文件，README 无许可声明。**许可状态：未授权**，研究复现用途需自行注意 |

## 3. as-published 复现结果（5 seeds，MovieLens-1M）

**论文报告的**（arXiv:2404.18465v3 Table 3）：完整 M3oE MovieLens 全域全任务平均 AUC ≈ 0.7702；w/o AutoML ≈ 0.7637。

**我们跑出的**（as-published 配置：lr=1e-2 / darts lr=1e-3 / wd=1e-5 / batch=40960 / epoch≤50 / patience=10 / expert_num=4 / 架构参数初值 1.0；5 seeds 全部触发早停）：

### 3.1 per (domain, task) test AUC（5 seeds，均值±标准差，ddof=1）

| (d,t) | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|
| test AUC | 0.7634 ± 0.0028 | 0.7775 ± 0.0020 | 0.7746 ± 0.0030 | 0.7930 ± 0.0021 | 0.7445 ± 0.0028 | 0.7587 ± 0.0011 |

### 3.2 6 项平均 AUC（按种子）

| seed | 2022 | 2023 | 2024 | 2025 | 2026 | 均值±标准差 |
|---|---|---|---|---|---|---|
| 平均 AUC | 0.77025 | 0.76855 | 0.76579 | 0.76971 | 0.76870 | **0.7686 ± 0.0017** |

- 与论文目标 0.7702 的差：**−0.0016**，在 0.005 容差内；且 0.7686 的 95% 置信区间（±0.0021, t(4)）覆盖 0.7702。复现判定：**成功，无需排查偏差**。
- 早停统计：全部 5 个种子在第 18–25 个 epoch 触发早停；最优验证 epoch（argmax 6 项平均 val AUC）为 7–14。墙钟 633–874 s/种子（RTX 4060 Laptop）。
- 明显的域间结构（后续阶段的背景事实）：d2t0（0.7445）是持续的最弱项，d1t1（0.7930）是最强项；小域 d0 并不是最弱的。

逐种子明细：`experiments/results/P0_repro.csv`；原始日志：`experiments/p0_seed{2022..2026}.log`。

## 4. 关键文件与关键函数清单（后续阶段使用）

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
| `examples/run_kuaishou.py` | 非官方文件（P3 待调试），KuaiRand 预处理 | :1-262 |
| `MDMTRec/basic/layers.py` | `EmbeddingLayer` | :29-116 |
| `MDMTRec/basic/features.py` | `SparseFeature`（默认 initializer :61）、`DenseFeature`（embed_dim=1 :91） | :49-94 |
| `MDMTRec/basic/callback.py` | `EarlyStopper`（严格大于才算提升） | :4-33 |
| `MDMTRec/utils/data.py` | `DataGenerator.generate_dataloader`（num_workers=8 默认；train shuffle=True，val/test shuffle=False） | :58-82 |

**代码侧其他值得记住的事实（核查表之外）**
1. sparse 特征被 LabelEncoder 编码**两次**：第一次 `+1`（run_movielens.py:56-58），第二次不带 +1 重新 fit_transform（:63-65）覆盖第一次结果。`vocab_size=nunique()`（:68）与最终编码一致，实际无害，但 `+1` 是死代码。（对照：run_kuaishou.py 采用了 `nunique()+1` 并重编码 domain_indicator 的"修复"，见 P3。）
2. Weights 在 `forward()` 内修改 `self.tau`（model.py:30-31）——forward 有副作用；sigmoid 路径下无影响。
3. `fit()` 的 exp_d/exp_t/bal_d/bal_t 形参被忽略（实际初值在模型构造时传入）。
4. 早停判据是 6 项平均验证 AUC（等权），且"提升"定义为严格大于（callback.py:24）。
5. 训练 loss 的 6 项 BCE 对全 batch 等权，意味着小域 d0（21% 样本）在梯度中天然占少数——这正是 P1 D0.4 要量化的问题。
6. **B10 bug 的实证**：3 epoch 冒烟运行后 4 个架构参数从 0.7311 移动到 0.7124/0.7224/0.7533/0.7395（主优化器逐 step 更新的效果）。

## 5. 复现风险与偏差清单

**环境/运行层面的调整（均非代码行为修改）**
1. venv 使用 `--system-site-packages` 复用 torch 2.9.0+cu126（理由见第 1 节）。
2. 数据路径通过 CLI 参数显式指定（默认值指向作者本机路径）。
3. 所有运行加 `python -u`（规避一次观察到的 DataLoader 停滞；已验证训练轨迹逐位不变）。
4. 仓库无 LICENSE（E3）——仅做研究性复现，未再分发。
5. 本地 `run_kuaishou.py` 非官方文件（E1），P3 阶段单独调试，不混入 P0 结论。

**论文与代码不一致处**
1. B6：Eq.(7) 与讨论段落对 α_d/α_t 的归属矛盾；代码遵循讨论段落（α_d×域专家、α_t×任务专家）。
2. 论文未报告：优化器参数分组的 bug（B10）、早停未触发时用最后一轮权重评测（C5 附注）、架构参数实际每 step 被主优化器更新——论文把 AutoML 描述为"每 epoch 用验证 batch 更新架构权重"，与实际实现不符。
3. 论文 Table 1 的 MovieLens 三域规模与我们统计完全一致（D4）；KuaiRand 数字待 P3 核对。

**核不上的断言**：无（A1–E3 共 24 条全部核到，无"未找到"项）。

## 6. P0 交付物索引

- `experiments/results/P0_repro.csv`：seed, domain, task, test_auc, test_logloss, best_epoch, wall_clock_sec
- `experiments/requirements.lock`：pip freeze 全量
- `experiments/verify_assertions.py` + 运行输出：断言的运行时验证证据
- `experiments/p0_seed{2022..2026}.log`：每个种子的完整训练日志
- `experiments/p0_wallclock.txt`：每种子墙钟时间

## 7. 阶段产出更新（P1 / P4 摘要，详情见对应报告）

### P1（诊断，`reports/P1_diagnosis.md`）
- 等价性验证：插桩开关关闭/开启时训练轨迹与 P0 **逐位一致**（val AUC 流、最终 test AUC bit-exact；loss 流差 ≤ float32 舍入）。
- D0.2：架构参数全程偏移 ≤0.037，"几乎没搜动"大体成立；但 bal_d 上漂至 ~0.767、exp_d 下探至 ~0.714，不全部落在 0.715–0.752。
- **D0.4：不成立** — 两档 batch 下三域 SNR 无显著差异（40960: 0.51/0.52/0.56；4096: ~0.05 全域一致）。小域梯度范数反而更大（0.090 vs 0.067）。跨域梯度余弦弱正相关（0.18–0.26 @40960；~0.04 @4096）。
- D0.6：各 (d,t) 峰值 epoch 分散 max−min=11；**大域 d2 最早过拟合（~6.4）**，d0 居中（9–12），d1 最晚（~14）。
- D0.7：全局早停（~11.4）对 d2t0 损失最大（−0.0052 test AUC），其余 ≤0.001。
- 共享参数清单（D0.4 定义，全项目沿用）：embedding 全部 7 表、shared_weight/bias、skip_conn、star_mlp、expert×4、domain_expert×3、task_expert×2；排除 tower/gate/slot_weight/slot_bias/架构标量。
- 工程坑（后续 runner 通用）：创建任何 DataLoader 迭代器都会抽全局 RNG 的 `_base_seed`（torch 2.9 dataloader.py:697，num_workers=0 也不例外）；额外评测必须用 `get_rng_state/set_rng_state` 包裹。

### P4（实现，`reports/P4_implementation.md`）
- 三组件全部实现于 `experiments/p4_run.py`，默认关闭 = 主基线；等价性测试 5/5 PASS（bit-exact）。
- 组件 1 per-domain 动量（γ_d 按训练集占比，共享参数 Σ_d m_d 替换 .grad；momentum_mode: none/per_domain_only/per_domain_stacked）；组件 2 域条件课程（MDMTRecBeta，β_d 展开 D 维，ρ=1 时前向与原版逐位一致）；组件 3 arch 平滑（EMA/掩码 loss/更新频率）。
- 开销实测：组件 1 +6% 耗时 / +216 MiB；组件 2 ±0；组件 3 +16%（含额外 arch 更新）。
- 主基线选定：待 P2 结果（`reports/P2_baselines.md`）。

### P3（KuaiRand 管线，`reports/P3_kuairand.md`）
- 数据源：官方 KuaiRand-Pure（Zenodo 官方渠道在本环境被屏蔽、USTC 镜像匿名登录维护中），经 Kaggle 官方镜像获取；**许可 CC BY-SA 4.0**（压缩包内 LICENSE 与官方一致）。
- 管线：三日志合并 → tab∈{0,1,4} 过滤（2,507,666 行）→ join user/video 特征（零行丢失）→ 全局 (date,time_ms) 时间排序 → 8:1:1 行序切分。**三域 users/items/instances/占比与论文 Table 1 全部逐位一致（12/12 diff=0）**。
- 重建偏差声明：切分顺序属推定（论文未公开，E1 已核实预处理未开源）；特征集合为 run_kuaishou.py 的 33 列（论文说 50，无清单可对齐）。
- run_kuaishou.py 修复 2 处 bug（漏传 optimizer_params_darts；Windows spawn 无法 pickle ~700MB 数据集 → num_workers=0），1-epoch 冒烟训练端到端跑通（val AUC 0.5601，仅证明可运行）。
- P5 的 kuairand 配置：`--kuairand 1 --dataset_path D:/CODE/data/kuairand/processed`。

### 用户指示（2026-09-06）：停止 KuaiRand 的运行与测试
- P5 的 kuairand 档位在完成 baseline(5 seeds) 与 fixed-0.7311(5 seeds)、comp1(2 seeds) 后按指示停止；后续所有实验（oracle/P6/P7）仅使用 MovieLens。
- 已采集的 kuairand 行保留在 `results/p5/P5_raw.csv` 供参考，报告中标注为"不完整（用户指示终止）"。

---

## 8. P5–P7 最终结论摘要（2026-09-08 闭环）

> 本节为 P5–P7 三阶段闭环后追加的最终结论摘要。P5 详情见 §6.6 与 `exp.md` §6，P6/P7 详情见各自报告（路径见下）。

### 8.1 P5（主实验，2026-09-08 闭环）
- **一句话结论**：组件 1（per-domain 动量）在 MovieLens 强不均衡档（≥30:1，ml_r30: +0.0166 p=0.0002 d=5.80；ml_r100: +0.0202 p=0.0061 d=2.37）显著为正，但全六档**非单调**（ml_r3 出现 −0.0012 名义显著负凹陷，n=3 待复核）；组件 2/3 单独不显著；KuaiRand（24:1）上仅"前沿移动"未达真 Pareto 外推（Reviewer 裁定）。
- **产物**：`results/p5/P5_raw.csv`（ML 144/144 OK + kuairand 30/30 OK）、`P5_main.csv`、`P5_oracle.csv`（kuairand 15/15 OK）、`figs/p5_trend.png`；`reports/ML_P5_REPORT.md`、`KUAI_STAGE1_ORACLE.md`、`KUAI_P5_MAIN_REPORT.md`、`FINAL_RESEARCH_REPORT.md`（均 Reviewer 签署）。

### 8.2 P6（消融，2026-09-08 闭环，ml_r30 参照点，85/85 OK）
- **要点**：comp1 增益几乎全部来自 per-domain 动量本体（d=10.16 p=2.2e-5）；均一 γ/手工 γ/SNR 反推 γ 均不及公式 γ（d=−3.08 / −5.38 均 p<0.003）；comp2/comp3 单独及课程方向反转均不显著；含 comp1 的组合（comp1&2 / comp1+comp3 / comp123）均显著（d≥5.01）；κ0∈[0.1, 2.0]（20×）整段平台区不显著区分；γ0∈[0.8, 0.95] 平台，0.99 显著跌落（d=−3.87）；EMA 应**替换**而非**叠加** Adam β1（Reviewer 补算 comp1 vs g8_stacked 直接配对 p=0.000776 d=4.11）。
- **产物**：`experiments/results/p6/P6_raw.csv`、`P6_ablation.csv`、`figs/p6_sensitivity.png`、`p6_report_stdout.txt`；`experiments/reports/P6_ablation.md`（Reviewer 签署并审计）。
- **关键负结果**：comp2 单独、comp3 单独、课程方向反转（正向 vs κ0=−0.5）、不含 comp1 的两组件组合（g6_comp2+comp3）均**不显著**，落在 5-seed 噪声带内——本批次数据无法支持"这些组件单独有可观测贡献"的强结论。

### 8.3 P7（泛化，2026-09-08 闭环，ML only，60/60 OK）
- **要点**：跨 backbone 复现（D=3, momentum=none, pooled AUC）：mmoe 0.767107±0.001336 / ple 0.764977±0.001610 / sharebottom 0.769253±0.001345 **复现成立**（高于论文参照 +0.36/+0.82/+0.67 AUC 点；sharebottom 为仓库自引入 4-expert+mean 聚合适配，严格性弱于 mmoe/ple 的 FuxiCTR 逐字移植）；star 0.533898±0.006611 **复现失败排除**（差论文 −22.69 AUC 点）。**组件 1 移植**到 mmoe（+0.002978 p=0.0066 d=2.32）、ple（+0.005538 p=0.0060 d=2.38）、sharebottom（+0.004012 p=0.0099 d=2.06）均**显著正效应**，证明方法非 M3oE 特有。**D/T 扩展**（D=5, T=2）mdmtoe +0.004904 p=0.0021 d=3.18、mmoe +0.001086 p=0.035 d=1.40，掩码路径泛化有效（T≠2 未实证）。**组件 2** 对 mmoe/ple/star/sharebottom **不适用**（无可调跨域融合标量）。
- **产物**：`experiments/results/p7/P7_backbones.csv`（60/60 OK）、`P7_summary.csv`、`P7_d5_per_dt_matrix.csv`、`p7_report_stdout.txt`；`experiments/reports/P7_generalization.md`（Reviewer 签署 + 审计勘误）。
- **已知限制**（Reviewer 仲裁）：sharebottom 补跑窗口与 mmoe/ple/star 不重合（仅 sharebottom 自窗口内 none vs per_domain_only 配对自洽）；§A pooled AUC 口径与论文 per-(d,t) 10 格平均口径不完全一致，仅作量级参照；D=5 仅覆盖 T=2，向"任意 (D,T)"外推需后续实验。

### 8.4 跨阶段总览
- **P0–P7 全部闭环**（2026-09-08）：八阶段报告均经 Reviewer 签署（含统计仲裁与勘误），台账真实、统计产物可复现、脚本方法学正确。
- **方法学价值定位**：组件 1（per-domain 动量）是 P5 主效应源、P6 消融确认其本体性、P7 跨 backbone 验证其非 M3oE 特有——三阶段互相印证。组件 2/3 在 P5/P6 均不显著，P7 上无可调标量故未移植。
- **诚实边界**：跨窗口可比性靠 2026-09-07 漂移探针与 P7 窗口披露两重护栏；sharebottom 复现严格性弱于 mmoe/ple；STAR 复现失败、AdaTT 未入选；D=5 仅 T=2，T≠2 未实证。
