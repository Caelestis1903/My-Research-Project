# 项目架构与多版本演进全景文档

> **适用范围**：本项目是 SIGIR 2024 论文 **M³oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework** 官方代码（`base_m3oe`）之上的多版本研究仓库，围绕"多域多任务推荐"展开两条改进线：**稀缺域反事实数据增强**（`mask` 分支）与**优化器动量机制**（`v1_momentum` / `v2_momentum` 分支）。
>
> **阅读约定**：
> - 各版本目录结构完全一致，统一以 `<版本>/MDMTRec/...` 表示，如 `base_m3oe/MDMTRec/models/multi_domain/model.py`；
> - "cell（单元）"指领域×任务组合，扁平索引 **i = t·D + d**（任务优先），6 个 cell 为 d0t0、d0t1、d1t0、d1t1、d2t0、d2t1；
> - 除特别说明，性能数字均来自各版本 `examples/output/*.csv`，MovieLens-1M、seed=2022、lr=0.01、expert_num=4。

---

## 0. 版本演进总览（Evolution Matrix）

| 版本代号 | 对应代码路径 | 核心创新 / 改动动机 | 关键机制变动 | 相比基线的性能指标变化（ML-1M, seed 2022, 总 AUC） |
|---|---|---|---|---|
| **base_m3oe**（基线） | `base_m3oe/` | M³oE 原版：跨域 Star 融合 + 跨任务 MMoE | Star 拓扑域调制（slot_weight ⊙ shared_weight）；共享/领域/任务三类专家 + 门控 + 平衡加权；6 个预测塔 | 基准 **0.7688**（重复运行波动 0.7688–0.7738；KuaiRand 0.6747） |
| **mask**（稀缺域反事实增强分支，2026-04） | `mask/` | 基线在**稀缺域（样本最少的 d0）** 过拟合、负迁移；引入反事实数据增强（CDA）+ 掩码去噪 | ① VUG 注意力虚拟样本生成器；② DBO 样本特异性重加权；③ 三阶段训练（真实→随机替换+冻结→注意力增强）；④ 样本级自适应掩码；⑤（文档级）任务对抗掩码 | 无 AutoML **0.7709**；AutoML 参数 **0.7713**；**+DBO 0.7771**（+0.83pt）；任务对抗掩码 **0.7769**；其余掩码配置 0.7722–0.7741；逐 cell/LogLoss 全录见 §2.1【实验结果全录】（注：output 工作日志另有 0.7750–0.7808 未入档运行，含"0.7799 备份版本"与最高 0.7808，配置不可判定） |
| **v1_momentum**（M3OE-MC，2026-06） | `v1_momentum/` | 基线单一 β₁=0.9 的 Adam 对**跨域共享组件**收敛慢、对**难域 d2** 学习不足；提出分层动量 + 域难度课程 | ① 按组件共享度分配 β₁（0.99/0.95/0.90/0.80）；② 随 epoch 演进的域级课程损失加权（cosine/linear，clip [0.5,2]）；③ Fixed-Anchor 冻结消融 | 消融协议：Baseline 0.7688 / FixedAnchor(25) 0.7672 / **M3OE-MC 0.7683**；调优后 MC 稳定 **0.7748–0.7752**，GateCurriculum 变体 **0.7755**（+0.67pt）；逐 cell/LogLoss 全录见 §2.2【实验结果全录】 |
| **v2_momentum**（优化器消融实验场，2026-08） | `v2_momentum/` | 在 v1 之上系统性回答"动量该怎样作用于难 cell（d2t0/d2t1）"：7 条干预通道全部环境变量门控、默认关闭即逐位复现基线 | ① v1 两开关解耦（分层动量 / 课程加权）；② 6 个自定义 Adam 子类（TAM/HTMW/CVR/MoMo-DN/LRMD/Polyak 及 Polyak×LRMD 组合）；③ CAM 梯度冲突校正；④ vGRA 门控残差适配器（模型侧）；⑤ 谱观测（SVD）只读通道 | 实验营基线 R0' = **0.7721507**（分层动量开/课程关，独立于 v1 消融协议）；**PolyakAdam(0.995) 唯一跨种子保留**：0.776768，n=6 配对 mean **+0.004022**（p=0.00056，36/36 cell·种子全正）；CAM 有害 **0.767433**（−0.0047）；LRMD 0.775084 / vGRA 0.774437 单种子过线、跨种子归零；R14 组合单种子最优 **0.778063**；KuaiRand 0.6701 / 0.6710（基线 0.6747）；全录见 §2.3【实验结果全录】 |

**版本谱系**（按文件修改时间排序）：

```
base_m3oe (SIGIR'24 原版)
   ├── mask            (2026-04)  反事实增强/掩码分支 —— 面向"数据稀缺域"
   └── v1_momentum     (2026-06)  分层动量+课程分支 —— 面向"优化动力学"
         └── v2_momentum (2026-08) v1 的系统化消融实验场（含 v1 全部机制）
```

---

## 1. 基线模型全解（Baseline Specification）

基线为 M³oE（`base_m3oe`），核心思想：**用 Star 拓扑（共享权重 ⊗ 域槽位权重）解决跨域知识共享，用 MMoE（共享专家 + 领域专家 + 任务专家 + 门控）解决跨任务平衡**。

### 1.1 核心问题与设计假设

**目标任务**：多域多任务 CTR 式预估（Multi-Domain Multi-Task，MTMD 设定）。

**形式化定义**：
- 输入：一批混合样本 $x \in \mathbb{N}^{F \times B}$（特征优先布局，由 trainer 中 `x_dict.transpose(0,1)` 得到），其中**最后一行**是 `domain_indicator`（域 ID）；
- 输出：$\hat{Y} \in (0,1)^{B \times T}$，每个样本在其所属域的塔上取值；
- 域数 $D=3$、任务数 $T=2$（MTMD 设定时），共 $D \times T = 6$ 个 cell。

**数据集设定**（`base_m3oe/examples/run_movielens.py`）：

| 数据集 | 域划分（domain_indicator） | 任务（基于 rating 二值化） | 特征 |
|---|---|---|---|
| MovieLens-1M | d0 = 年龄∈{1,18}；d1 = 25；d2 = {35,45,50,56} | click: rating>3；like: rating>4 | 稀疏 7 个：user_id, movie_id, gender, occupation, zip, cate_id, domain_indicator（embed_dim=16）；稠密 1 个：age（MinMax 归一化，dim=1） |
| KuaiRand（`run_kuaishou.py`） | `tab` 字段（基线固定 tab∈{0,1,4}） | 同上式样的双任务 | 33 个稀疏特征 |

**设计假设**：
1. 域间存在可共享的底层模式（由 Star 共享权重与共享专家承载），也存在域特异模式（由槽位权重与领域专家承载）；
2. 任务间同理（任务专家 + 每域每任务独立门控）；
3. **d0（年龄 ≤18）是样本最稀缺的域**：ML-1M 全量统计为 d0 = 210,747 条 vs d1 = 395,556 / d2 = 393,906 条，`mask` 分支的稀缺域增强即以此为靶点；
4. **d2（年龄 35+）是天然难域**：样本量不少但基线 6 个 cell 中 d2t0 AUC 最低（0.7449 vs d0t0 0.7656），这一"难而不少"的现象是 `v1/v2_momentum` 动量机制的直接动机；
4. 门控基于 `emb.detach()`，即路由权重不向表征回传梯度（路由与表征解耦）。

### 1.2 网络架构与数据流向

**代码地图**：

| 组件 | 类/函数 | 位置 |
|---|---|---|
| 模型主体 | `MDMTRec`（`__init__` / `forward`） | `base_m3oe/MDMTRec/models/multi_domain/model.py` |
| 通用 MLP 块（Linear+LayerNorm+ReLU） | `MLP_N` | 同上 |
| 嵌入层 | `EmbeddingLayer` | `base_m3oe/MDMTRec/basic/layers.py` |
| 特征声明 | `SparseFeature` / `DenseFeature` | `base_m3oe/MDMTRec/basic/features.py` |
| 训练器（损失/评估/早停） | `CTRTrainer`（`train_one_epoch` / `fit` / `evaluate_multi_domain`） | `base_m3oe/MDMTRec/trainers/ctr_trainer.py` |
| 早停 | `EarlyStopper` | `base_m3oe/MDMTRec/basic/callback.py` |
| 数据管道 | `DataGenerator` | `base_m3oe/MDMTRec/utils/data.py` |
| 启动脚本 | `run_movielens.py` / `run_kuaishou.py` / `run_MDMT.sh` | `base_m3oe/examples/` |

**Step-by-Step 数据流**（记批次大小为 $B$，输入总维度 $D_{in}=7\times16+1=113$）：

```
x [F=8, B]（最后一行 = domain_id）
  │ ① domain_id = x[-1,:]，生成 3 个布尔域掩码 m_d ∈ {0,1}^B
  ▼
② EmbeddingLayer(x, features, squeeze_dim=True)          → X ∈ R^{B×113}
  │  （7 个稀疏查表 16 维拼接 + age 稠密值直接拼接）
  ▼
③ Star 跨域融合：
  │   skip = SkipMLP(X) = ReLU(LN(W_skip·X + b_skip))     → [B×256]（MLP_N([113,256])）
  │   对每个域 d：h_d = X·(W_d ⊙ W_shared) + b_d + b_shared → [B×512]
  │     （slot_weight_d ∈ R^{113×512} 逐元素调制 shared_weight ∈ R^{113×512}）
  │   emb⁰[b] = Σ_d m_d(b)·h_d[b]（torch.where 按域掩码逐行填充）
  │   emb = StarMLP(emb⁰) + skip                           → E ∈ R^{B×256}（MLP_N([512,256])）
  ▼
④ 专家层（全部为 MLP_N([256,64])，即 Linear(256→64)+LN+ReLU）：
  │   共享专家 F_sh[i] = Expert_i(E), i=1..4               → 堆叠 [B×4×64]
  │   领域专家 F_D[d] = DomainExpert_d(E), d=0..2          → [B×3×64]
  │   任务专家 F_T[t] = TaskExpert_t(E), t=0..1            → [B×2×64]
  ▼
⑤ 门控（每 cell 一个）：g_i = Softmax(Linear_{256→4}(E.detach()))  → [B×4]
  ▼
⑥ 平衡加权（bal_d / bal_t）：
  │   F̃_D[d] = bal_d·F_D[d] + (1-bal_d)/(D-1)·Σ_{j≠d} F_D[j]
  │   F̃_T[t] = bal_t·F_T[t] + (1-bal_t)/(T-1)·Σ_{j≠t} F_T[j]
  ▼
⑦ 融合（每 cell i=(t,d)）：
  │   z_i = g_i·F_sh + exp_d·F̃_D[d] + exp_t·F̃_T[t]        → [B×64]
  ▼
⑧ 预测塔（每 cell 一个：Linear(64→64)+LN+ReLU+Linear(64→1)+Sigmoid）：
  │   ŷ_i = σ(Tower_i(z_i))                                → [B]
  ▼
⑨ 重组装：output=[B×6] 按 cell 排布 → Ŷ[b,t] = Σ_d m_d(b)·ŷ_{(t,d)}[b]  → [B×2]
```

**关键形状速查**（默认超参下）：`X: [B,113]` → `emb: [B,256]` → `专家输出/gate输入: [B,64]/[B,4]` → `z_i: [B,64]` → `ŷ_i: [B]` → `Ŷ: [B,2]`。

**训练/评估循环要点**（`CTRTrainer`）：
- MTMD 设定（`domain_num==3 and task_num==2`）下，按域掩码把 `y_pred`/`y` 拆成 6 组，**逐 cell 求 BCE 再求和**作为总损失；
- 评估（`evaluate_multi_domain`）逐 cell 计算 AUC / LogLoss，另计算全体样本的 pooled AUC / LogLoss；
- 模型选择：以 6 个 cell AUC 的**平均值**做 EarlyStopping（patience=10），训练结束保存最优权重。

### 1.3 核心数学建模与损失函数

**记号**：$X\in\mathbb{R}^{B\times113}$ 为嵌入矩阵；$m_d(b)=\mathbb{1}[\text{domain}_b=d]$；cell 索引 $i=(t,d)$。

**（1）Star 跨域融合**

$$
\text{skip}=\sigma_{\text{ReLU}}\!\big(\mathrm{LN}(W_{\text{skip}}X+b_{\text{skip}})\big),\qquad
h_d = X\,(W_d \odot W_{\text{sh}}) + (b_d + b_{\text{sh}})
$$

$$
E^{(0)}[b]=\sum_{d=0}^{D-1} m_d(b)\,h_d[b],\qquad
E=\mathrm{StarMLP}\big(E^{(0)}\big)+\text{skip}
$$

其中 $\odot$ 为逐元素乘（域槽位权重对共享权重的"调制"），$W_{\text{sh}},W_d\in\mathbb{R}^{113\times512}$，Xavier 均匀初始化。

**（2）三类专家与门控路由**

$$
F_{\text{sh}}^{(k)}=\mathrm{MLP}_{\text{exp}}(E),\ k=1..K\ (K=4);\quad
F_D^{(d)}=\mathrm{MLP}_{\text{exp}}(E);\quad
F_T^{(t)}=\mathrm{MLP}_{\text{exp}}(E)
$$

$$
g_{(t,d)}=\mathrm{Softmax}\!\big(W_g\,\mathrm{sg}(E)+b_g\big)\in\Delta^{K-1},\qquad \mathrm{sg}(\cdot)=\text{stop-gradient}
$$

**（3）域/任务平衡特征**

$$
\tilde F_D^{(d)} = \beta_d F_D^{(d)} + \frac{1-\beta_d}{D-1}\sum_{j\neq d}F_D^{(j)},\qquad
\tilde F_T^{(t)} = \beta_t F_T^{(t)} + \frac{1-\beta_t}{T-1}\sum_{j\neq t}F_T^{(j)}
$$

$\beta_d=\text{bal\_d},\ \beta_t=\text{bal\_t}$（默认 1.0，此时退化为纯自身专家）。

**（4）cell 融合与预测**

$$
z_{(t,d)} = g_{(t,d)}^{\top}F_{\text{sh}} + \gamma_d\,\tilde F_D^{(d)} + \gamma_t\,\tilde F_T^{(t)},\qquad
\hat y_{(t,d)} = \sigma\!\big(\mathrm{Tower}_{(t,d)}(z_{(t,d)})\big)
$$

$\gamma_d=\text{exp\_d},\ \gamma_t=\text{exp\_t}$（默认 1.0）；$\mathrm{Tower}=\text{Linear}(64{\to}64)\to\text{LN}\to\text{ReLU}\to\text{Linear}(64{\to}1)$。

**（5）优化目标**（MTMD 设定，`ctr_trainer.py: train_one_epoch`）

$$
\mathcal{L}_{\text{M}^3\text{oE}}=\sum_{t=0}^{T-1}\sum_{d=0}^{D-1}\ \ell_{(t,d)},\qquad
\ell_{(t,d)}=\frac{1}{|\mathcal B_d|}\sum_{b\in\mathcal B_d}\mathrm{BCE}\big(\hat y_{(t,d),b},\,y_{b,t}\big)
$$

$\mathcal B_d$ 为批次内属于域 $d$ 的样本集合；即 **6 个 cell 的 BCE 均值再求和**（注意：不是全体样本的平铺 BCE）。非 MTMD 设定退化为单一 BCE。

**（6）评估指标**：逐 cell AUC（`roc_auc_score`）/ LogLoss；pooled AUC / LogLoss；模型选择用 6 cell 平均 AUC。

### 1.4 关键超参数与训练配置

**数据/优化**（`base_m3oe/examples/run_movielens.py` 的 argparse 默认值）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `learning_rate` | **1e-2** | Adam（`torch.optim.Adam`） |
| `weight_decay` | 1e-5 | Adam L2 正则（coupled） |
| `batch_size` | 4096×10 = **40960** | Kuaishou 脚本同为 40960 |
| `epoch` | 50 | 配合 EarlyStopping（patience=10） |
| LR 调度 | `StepLR(step_size=4, gamma=0.85)`（注释态） | epoch 级衰减 |
| 损失 | `torch.nn.BCELoss` | MTMD 下逐 cell 求和 |
| 划分 | train/val/test = 80%/10%/10% | 时间顺序切分 |
| 种子 | 2022–2026 各跑一轮（`run_MDMT.sh`） | `cudnn.deterministic=True` |
| 设备 | `cuda:0` | 支持 `DataParallel` |

**模型结构**：

| 参数 | 默认值 | 派生形状 |
|---|---|---|
| `fcn_dims` | [512, 256, 256, 64] | 全维序列 [113,512,256,256,64] |
| Star 层 | 前 3 层 | skip: MLP_N([113,256])；StarMLP: MLP_N([512,256])；shared/slot_weight: [113,512] |
| `expert_num` | 4 | 共享专家 4 + 领域专家 3 + 任务专家 2，均为 MLP_N([256,64]) |
| `embed_dim` | 16 | input_dim = 113 |
| `exp_d` / `exp_t` | 1 / 1 | 域/任务专家融合系数 $\gamma_d,\gamma_t$ |
| `bal_d` / `bal_t` | 1 / 1 | 平衡系数 $\beta_d,\beta_t$ |
| `tau`, `tau_step`, `softmax_type` | 1, 5e-5, 3 | 构造器预留的温度参数（基线 forward 中未使用） |

**结果落盘**：`examples/output/MDMTRec_domain_{domain}_task_{task}_exp_{expert_num}_{lr}.csv`，列含逐 cell `auc_dXtY` / `log_dXtY` 与 pooled `auc` / `log`。

**基线参考成绩**（seed 2022，lr=0.01）：pooled AUC **0.7688**，逐 cell AUC — d0t0 0.7656 / d0t1 0.7776 / d1t0 0.7731 / d1t1 0.7901 / **d2t0 0.7449（最弱）** / d2t1 0.7561；KuaiRand pooled AUC 0.6747。

---

## 2. 迭代版本差异化详解（Version Diffs）

> 本章按时间顺序（mask → v1 → v2）展开，**每个版本只写相对其父版本的增量**；未提及的文件/机制与基线完全一致。

### 2.1 mask — 稀缺域反事实增强（VUG + DBO + 三阶段训练 + 自适应掩码）

#### 【修改动机与假设】

基线把三个年龄域混合训练，**样本量最小的域（自动检测，通常 d0）严重欠拟合且共享层被其他域主导**。本分支的核心假设：

1. **反事实增强（CDA）**：为稀缺域人工制造"虚拟优质样本"，让共享层见到难以自然出现的数据，缓解稀缺造成的过拟合；
2. **增强数据有噪声，必须隔离**：随机替换生成的样本不可信，需冻结共享参数 + 降低 loss 权重（0.1），防止噪声负迁移到其他域；注意力生成的样本更可信，后期可放开共享层；
3. **样本异质性（DBO）**：不同样本的"域特异性"不同，应按特异性重新加权 loss；
4. **掩码去噪**：对生成的噪声样本按难度自适应掩码——难样本少掩码（保信息）、易样本多掩码（加挑战），并把掩码上升到任务层面（掩掉"任务间共享但不特异"的特征，倒逼任务解耦）。

#### 【代码级增量修改（Code Delta）】

**① 新增 VUG 虚拟样本生成器** — `mask/MDMTRec/models/multi_domain/model.py: UnnormalizedScoreAttentionGenerator`（别名 `VUG_Generator`）：

```python
class UnnormalizedScoreAttentionGenerator(nn.Module):
    # mode='attention'：稀缺样本嵌入为 Q，整批嵌入为 K/V
    #   attention_scores = Q·Kᵀ/√d_k                [A, B]
    #   attention_weights = dropout(scores)          # use_softmax=False → 不做 Softmax（"未归一化分数"）
    #   virtual_emb = attention_weights·V            [A, D]
    #   output = LayerNorm(virtual_emb + target_emb) # 残差 + LN 稳定尺度
    # mode='random'：target_emb + N(0, noise_std)
```

`MDMTRec.__init__` 新增 `self.vug_generator = VUG_Generator(input_dim, mode=gen_mode)`。

**② forward 增加增强/掩码钩子** — `MDMTRec.forward`（签名从 `(x, test_flag)` 扩为 `(x, test_flag, aug_mask, noise_std, feature_mask)`）：

```diff
+ def forward(self, x, test_flag=False, aug_mask=None, noise_std=0.1, feature_mask=None):
      ...
+     # 特征级掩码：训练态且带 feature_mask 时，将 movie_id（特征行 1）置 0（PAD/MASK）
+     if self.training and feature_mask is not None and feature_mask.sum() > 0:
+         x_processed = x.clone(); x_processed[1, feature_mask] = 0
      input_emb = self.embedding(x_processed, self.features, squeeze_dim=True)
+     # VUG：aug_mask 标记的（稀缺域复制）样本，其嵌入被注意力生成结果原位替换
+     if self.training and aug_mask is not None and aug_mask.sum() > 0:
+         target_emb = input_emb[aug_indices]; context_emb = input_emb.clone()
+         input_emb[aug_indices] = self.vug_generator(target_emb, context_emb, noise_std)
+     specificity = self.dbo_specificity(input_emb)      # DBO 特异性 [B]
      ...
+     if self.training: return result, specificity       # 训练态额外返回特异性
+     else:             return result
```

**③ 新增 DBO 层** — `mask/MDMTRec/basic/layers.py`：

```python
class DBO_Specificity_Layer(nn.Module):      # 前向使用
    # mlp = Linear(input_dim→64) → ReLU → Linear(64→2) → Softmax
    # forward: g = softmax(mlp(x)); return g[:, 0]   # g_sp ∈ [0,1]，[B]

class DBO_Complexity_Layer(nn.Module):       # 注意力熵复杂度（D3 公式 6/7/15），当前 MDMTRec 未启用
```

**④ trainer 全面重写三阶段训练** — `mask/MDMTRec/trainers/ctr_trainer.py`：

- `__init__` 新增：`scarce_domain`、`cda_weight`、`aug_col_index`（movie_id 列）、`aug_vocab_size`、`random_aug_ratio`、`use_adaptive_masking`、`mask_base_ratio`、`mask_temp`、`mask_feature_indices`；
- 新增 `_compute_adaptive_mask_prob`（自适应掩码概率）与 `_apply_feature_mask`（Bernoulli 采样 + 置 0）；
- 新增 `calculate_loss`：统一计算入口，支持 `sample_mask`（只对选中样本求损失）与 DBO 样本权重；
- `train_one_epoch` 按 epoch 分相（详见下方机制）；`fit` 向其传入 `epoch_i`。

```diff
+ # ===== Phase 2（5 ≤ epoch < 10）：随机替换 + 自适应掩码 + 冻结共享层 =====
+ if self.scarce_domain is not None and scarce_mask.sum() > 0 and self.model.training and 5 <= epoch_i < 10:
+     num_aug = max(1, int(len(scarce_indices) * self.random_aug_ratio))
+     y_noise = uniform_(0.4, 0.6)                        # 软标签置信度
+     x_noise[aug_col_index, :] = randint(1, aug_vocab_size)  # 随机替换 movie_id
+     if self.use_adaptive_masking:                       # no_grad 前向算样本 loss → 掩码概率 → Bernoulli 掩码
+         mask_probs = self._compute_adaptive_mask_prob(sample_losses, base_ratio, temp)
+         x_noise_masked, applied_mask = self._apply_feature_mask(x_noise, mask_probs, feature_indices)
+     # 冻结：名字含 embedding/shared/task_expert/expert 且不含 domain/slot 的参数 requires_grad=False
+     loss_noise = self.calculate_loss(...) * 0.1; loss_noise.backward(); 解冻
+
+ # ===== Phase 3（epoch ≥ 10）：VUG 注意力增强，不冻结 =====
+ if ... and epoch_i >= 10:
+     x_input = cat([x_dict, x_aug])                      # 稀缺域样本整批复制拼接
+     aug_mask[original_bs:] = True
+     loss_aug = self.calculate_loss(x_input, y_input, aug_mask, sample_mask=aug_mask) * 0.1
+     loss_aug.backward()                                 # 增强分支独立反传（共享层开放）
+     loss_real = self.calculate_loss(x_dict, y, None); loss_real.backward()
+     run_normal_training = False
```

**⑤ 入口脚本** — `mask/examples/run_movielens.py`：自动检测稀缺域 `auto_scarce_domain = argmin(域样本计数)`；新增命令行参数 `--scarce_domain / --cda_weight / --gen_mode {attention,random} / --random_aug_ratio / --use_adaptive_masking / --mask_base_ratio / --mask_temp / --mask_feature_indices`（`str2bool` 助手）。

**⑥ 其余**：`utils/data.py` 的 dataloader `num_workers` 0→8；`basic/__init__.py`、`models/multi_domain/__init__.py` 导出新符号；`model.py` 追加三个**接口兼容的对比基线**（PEPNet / HiNet / MDRAU，均适配 `forward(x, aug_mask, noise_std)` + 训练态返回元组的协议）供对比实验。

**配套文档**（`mask/` 根目录）：`ADAPTIVE_MASKING.md`（自适应掩码）、`MASK_MODIFICATIONS.md`（逐行修改记录，注明对照"0.7799 备份版本"）、`GRADIENT_ADAPTIVE_SCALING.md`（Phase2 冻结/Phase3 开放的梯度策略）、`TASK_CONFUSION_MASKING_PATCH.md`（任务对抗掩码设计；**对应 trainer 文件已从仓库移除，仅存文档与结果 CSV**）、`AGENTS.md`（项目总览）。

#### 【数学与机制变化】

**三阶段训练课程**：

| 阶段 | Epoch | 生成方式 | 共享层梯度 | 增强 loss 权重 |
|---|---|---|---|---|
| Phase 1 | 0–4 | 无增强，纯真实数据 | 正常 | — |
| Phase 2 | 5–9 | 随机替换 movie_id（比例 `random_aug_ratio=0.1`），标签 $y_{noise}\sim U(0.4,0.6)$ | **冻结**（embedding/shared/task_expert/expert） | ×0.1 |
| Phase 3 | 10+ | VUG 注意力生成（稀缺样本复制后嵌入被替换，`aug_mask` 标记） | **开放** | ×0.1（独立反传），真实数据正常反传 |

**样本级自适应掩码**（`_compute_adaptive_mask_prob`）：设归一化难度 $\tilde\ell=\frac{\ell-\ell_{\min}}{\ell_{\max}-\ell_{\min}}\in[0,1]$（1=最难），温度 $\tau$（`mask_temp`，>1 差异减小）：

$$
p_{\text{mask}}=\mathrm{clip}\Big(b+(1-2b)\big(1-\tilde\ell^{\,1/\tau}\big),\ 0.05,\ 0.95\Big),\qquad b=\text{mask\_base\_ratio}=0.3
$$

即**难样本低掩码率、易样本高掩码率**；随后按 Bernoulli($p_{mask}$) 逐样本采样，将 `mask_feature_indices`（或模型内 movie_id 行）置 0。**仅作用于 Phase 2 的随机替换样本**，真实数据不掩码。

**DBO 样本重加权**（进入 `calculate_loss` 的损失项）：

$$
g_{\text{sp}}=\mathrm{softmax}\big(\mathrm{MLP}(E)\big)[:,0],\qquad
w_{\text{spf}}=\lambda_2+\alpha_2\,g_{\text{sp}}\quad(\lambda_2{=}0.1,\ \alpha_2{=}0.5)
$$

$$
\mathcal{L}=\sum_{t,d}\frac{1}{|\mathcal B_d|}\sum_{b\in\mathcal B_d} w_{\text{spf}}(b)\cdot \mathrm{BCE}\big(\hat y_{(t,d),b},y_{b,t}\big)
$$

即总损失形式仍为基线 $\mathcal L_{\text{M}^3\text{oE}}$，但逐样本 BCE 乘上与域特异性正相关的权重。

**任务对抗掩码**（文档级，`TASK_CONFUSION_MASKING_PATCH.md`；结果 CSV `MDMTRec_task_confusion_domain_all_task_all.csv` 佐证）：对每任务 $t$ 估计特征梯度重要性 $I_{b,f,t}$，任务特异性取跨任务方差 $S_{b,f}=\mathrm{Var}_t(I_{b,f,t})$，混淆分 $c_{b,f}=\bar I_{b,f}/(S_{b,f}+\varepsilon)$，Top-K（比例 0.3）掩码"重要性高但特异性低"（共享但不特异）的特征，另附特异性辅助损失（权重 0.1）。

#### 【超参数新增/变动】

| 参数 | 默认值 | 作用 |
|---|---|---|
| `scarce_domain` | **自动**（样本最少的域） | 增强目标域 |
| `cda_weight` | 0.5 | CDA 损失权重（传入 trainer，实际三阶段内固定用 0.1 系数） |
| `gen_mode` | `attention` | VUG 模式（attention / random） |
| `random_aug_ratio` | 0.1 | Phase 2 随机替换比例 |
| `use_adaptive_masking` | True | 自适应掩码开关 |
| `mask_base_ratio` | 0.3 | 基础掩码率 $b$ |
| `mask_temp` | 1.0 | 掩码温度 $\tau$ |
| `mask_feature_indices` | None | 指定掩码的特征行（None → 模型内默认掩 movie_id） |
| DBO 内部常数 | $\lambda_2{=}0.1,\ \alpha_2{=}0.5$ | 样本权重 $w_{\text{spf}}=\lambda_2+\alpha_2 g_{\text{sp}}$ |
| 任务对抗掩码 | `confusion_mask_ratio=0.3`, `confusion_aux_weight=0.1` | 见文档（代码已移除） |

**实证结果**（`mask/examples/output/`，总 AUC）：原始代码（AutoML 搜索的 exp_d=0.6953 / exp_t=0.7248 / bal_d=0.7467 / bal_t=0.7348）**0.7713**；固定参数（无 AutoML）**0.7709**；**加入 DBO 0.7771**（相对基线 0.7688 +0.83pt）；引入时机设为第 10 轮等配置 0.7722–0.7741；任务对抗掩码（ratio 0.3 / aux 0.1）**0.7769**。

#### 【实验结果全录】mask 分支的逐 cell AUC 与 LogLoss（自 `mask/examples/output/*.csv` 提取，2026-09-06 补齐）

**表 M-A｜有标签配置的 test AUC**（`MDMTRec_domain_all_task_all_exp_4_0.01.0001.csv` 内中文标签行 + `MDMTRec_task_confusion_*.csv`；总 AUC 与上方实证结果逐项吻合）

| 配置（CSV 内标签） | 总 AUC | Δ vs 0.7688 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|---|
| 原始代码（AutoML：exp_d 0.6953 / exp_t 0.7248 / bal_d 0.7467 / bal_t 0.7348） | 0.7713 | +0.25pt | 0.7667 | 0.7809 | 0.7758 | 0.7931 | 0.7447 | 0.7603 |
| 没有AutoML（固定参数） | 0.7709 | +0.21pt | 0.7657 | 0.7807 | 0.7754 | 0.7917 | 0.7455 | 0.7602 |
| 加入DBO | 0.7771 | +0.83pt | 0.7719 | 0.7892 | 0.7773 | 0.7969 | 0.7520 | 0.7669 |
| 加入引入时机（第 10 轮） | 0.7732 | +0.44pt | 0.7653 | 0.7820 | 0.7735 | 0.7961 | 0.7498 | 0.7652 |
| 加入DBO（第二次运行，疑叠加引入时机，变体无法判定） | 0.7757 | +0.69pt | 0.7723 | 0.7900 | 0.7743 | 0.7940 | 0.7500 | 0.7678 |
| 任务对抗掩码（ratio 0.3 / aux 0.1） | 0.7769 | +0.81pt | 0.7709 | 0.7861 | 0.7768 | 0.7962 | 0.7521 | 0.7670 |

**表 M-B｜上表各配置的 pooled LogLoss 与逐 cell LogLoss**

| 配置 | pooled LL | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|
| 原始代码（AutoML） | 0.4902 | 0.5989 | 0.4250 | 0.5775 | 0.3964 | 0.6099 | 0.4508 |
| 没有AutoML（固定参数） | 0.4475 | 0.5961 | 0.4354 | 0.5783 | 0.3940 | 0.5989 | 0.4523 |
| 加入DBO | 0.4577 | 0.5810 | 0.4049 | 0.5722 | 0.3830 | 0.5795 | 0.4318 |
| 加入引入时机（第 10 轮） | 0.4551 | 0.6056 | 0.4190 | 0.5832 | 0.3856 | 0.5929 | 0.4411 |
| 加入DBO（第二次运行） | 0.4229 | 0.5860 | 0.4106 | 0.5844 | 0.3951 | 0.5841 | 0.4352 |
| 任务对抗掩码（ratio 0.3 / aux 0.1） | 0.4957 | 0.5765 | 0.3988 | 0.5680 | 0.3740 | 0.5764 | 0.4334 |

**无标签运行与判定边界（重要）**：

1. `MDMTRec_domain_all_task_all_exp_4_0.01.csv` 为**无标签工作日志**（74 次运行记录，含空行分隔与 AutoML/DARTS 搜索轨迹行——exp_d/exp_t/bal 列出现 0.4/0.8/1158.5 等搜索中间值的行不是最终结果）。其中 **9 次运行总 AUC 落在文档所述"其余掩码配置 0.7722–0.7741"区间附近**（0.7741 / 0.7733 / 0.7733 / 0.7722 / 0.7723 / 0.7734 / 0.7722 / 0.7750 / 0.7729，逐 cell 数值在 CSV 中可查），但 **CSV 不含配置标签，无法判定每次运行对应哪个掩码配置**。
2. 该文件另含**文档未记录的后续运行约 60 次**（总 AUC 0.7663–0.7808，多数无标签）：总 AUC **0.7799** 的运行出现 9 次（逐 cell 一致），即 §2.1 配套文档注明的"**0.7799 备份版本**"；带文字标签的运行有"只用在生成数据"（0.7796）、"应用Phase2"（0.7796）、"0.5温度系数"（0.7796）；全文件最高 **0.7808**（无标签，逐 cell 0.7789/0.7928/0.7837/0.8020/0.7572/0.7675）。这些运行均无配对超参记录，**本次仅登记、不归因**；引用时不可与 0.7771/0.7769 混用。
3. 判定方法：有标签行以"中文标签 + 总 AUC 与文档数字双吻合"为准（如 0.7713/0.7709/0.7771）；task_confusion 行以 confusion_mask_ratio=0.3 / confusion_aux_weight=0.1 列值与文档参数双重吻合为准。

---

### 2.2 v1_momentum — 分层动量 + 域难度课程（M3OE-MC）

#### 【修改动机与假设】

基线用统一 β₁=0.9 的 Adam 训练所有参数，存在两个痛点：

1. **跨域共享组件收敛慢**：`shared_weight / star_mlp / skip_conn` 承载跨域共识，梯度在域间相互抵消、信噪比低，需要**更大的动量（更强的历史平滑）**；
2. **路由需要快适应、难域需后程发力**：gate 的路由决策应快速调整；而天然难域 d2（AUC 最低）若全程高权重会被噪声主导，应按"课程学习"**先易后难**地调度各域 loss 权重。

由此提出 **M3OE-MC**（M³oE-Momentum-Curriculum）：**分层动量优化器 + 域难度课程加权**两条正交机制，另设 Fixed-Anchor 冻结消融用于归因。

#### 【代码级增量修改（Code Delta）】

**唯一改动的库文件：`v1_momentum/MDMTRec/trainers/ctr_trainer.py`**（模型结构零改动）。

**① `__init__` 新增参数与分层参数组构建**：

```diff
  def __init__(self, model, optimizer_fn=..., optimizer_params=..., ...,
+             use_momentum_curriculum=False, domain_difficulty=None,
+             curriculum_type='cosine', domain_num=3, fixed_anchor_epochs=0):
      ...
+     if domain_difficulty is None:   # 默认难度线性 0.2(最易)→0.8(最难)
+         domain_difficulty = [0.2 + 0.6 * i / (domain_num - 1) for i in range(domain_num)]
+     if self.use_momentum_curriculum:
+         param_groups = self._build_momentum_param_groups(optimizer_params)
+         self.optimizer = optimizer_fn(param_groups)      # 分组 β₁
+     else:
+         self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
```

**② `_build_momentum_param_groups`**——按"组件共享度"分 4 组赋予不同 β₁（β₂ 恒 0.999，lr/wd 同基线）：

```python
group_specs = [
    (['shared_weight', 'shared_bias', 'skip_conn', 'star_mlp'], 0.99, 'star_shared_base'),  # 跨域共识：极高动量
    (['expert.'],                                              0.95, 'shared_expert'),      # 共享专家：高动量
    (['slot_weight', 'slot_bias', 'domain_expert.', 'task_expert.',
      'tower.', 'embedding.'],                                0.90, 'medium_momentum'),    # 域/任务私有：标准
    (['gate.'],                                                0.80, 'gate'),               # 路由：低动量快适应
]
# 兜底组 'others'：β₁=0.90；DataParallel 的 'module.' 前缀由 _strip_module_prefix 剥离
```

**③ `get_curriculum_weights(epoch)`**——域级课程权重（cosine / linear 两种调度，clip 到 [0.5, 2.0] 防梯度消失）：

```python
progress = epoch / (self.n_epoch - 1)
for diff in self.domain_difficulty:
    if self.curriculum_type == 'cosine':
        if diff <= 0.5:   # 易域：高 → 1.0
            w = 1.0 + (1.0 - 2.0*diff) * math.cos(progress * math.pi / 2)
        else:             # 难域：1.0 → 高
            w = 1.0 + (2.0*diff - 1.0) * (1.0 - math.cos(progress * math.pi / 2))
    elif self.curriculum_type == 'linear': ...
    w = max(0.5, min(w, 2.0))
```

**④ `train_one_epoch` 损失改为课程加权并归一化**：

```diff
- loss = sum(torch.stack(logloss_list))                    # 基线：6 cell BCE 求和
+ curriculum_weights = self.get_curriculum_weights(epoch_i) if self.use_momentum_curriculum else [1.0]*domain_num
+ weighted_losses.append(curriculum_weights[d] * logloss_list[t*domain_num + d])
+ total_weight = sum(curriculum_weights) * task_num
+ loss = sum(torch.stack(weighted_losses)) / total_weight  # 加权后按总权重归一化
+ # 注意：即使用 MC 关闭（权重全 1），loss 也从“求和”变为“均值”（等价 lr ×1/6 的尺度变化）
```

**⑤ `fit` 增加 Fixed-Anchor 消融钩子**：`fixed_anchor_epochs=N` 时，epoch 0 冻结 `_anchor_param_names()=['shared_weight','shared_bias','skip_conn','star_mlp','expert.']`，第 N 轮解冻。另加每 batch `torch.cuda.empty_cache()`。

**⑥ 实验脚本**：`examples/run_ablation.sh`（三配置：Baseline / FixedAnchor(25) / M3OE-MC，seed 2022，batch 40960，lr 1e-2，50 epoch）+ `examples/analyze_ablation.py`（解析训练日志 → `ablation_summary.csv`、验证曲线 `ablation_curves.png`、训练平滑度指标 = 逐 epoch 验证 AUC 差分的 std）。

**⑦ 入口脚本**：`run_movielens.py` 新增 `--use_momentum_curriculum`、`--domain_difficulty "0.2,0.5,0.8"`、`--curriculum_type {cosine,linear}`、`--fixed_anchor_epochs`；`run_kuaishou.py` 的域划分改为**动态取样本数前 3 的 tab**。

#### 【数学与机制变化】

**（1）分层动量**：对参数组 $G$，一阶动量递推的 β₁ 按"共享度↑ → β₁↑"分配：

$$
m \leftarrow \beta_1^{(G)}\,m + (1-\beta_1^{(G)})\,g,\qquad
\beta_1^{(\cdot)}\in\{\underbrace{0.99}_{\text{star\_shared\_base}},\underbrace{0.95}_{\text{shared\_expert}},\underbrace{0.90}_{\text{medium}},\underbrace{0.80}_{\text{gate}}\}
$$

直觉：共享参数的有效动量窗口更长（平滑域间冲突梯度），gate 窗口更短（快速改路由）。

**（2）课程加权损失**：

$$
\mathcal{L}_{\text{MC}}(e)=\frac{\displaystyle\sum_{t,d} w_d(e)\,\ell_{(t,d)}}{T\sum_d w_d(e)},\qquad
w_d(e)=\mathrm{clip}\Big(1+\big|2\delta_d-1\big|\cdot\phi_e,\ 0.5,\ 2.0\Big)
$$

其中 $\delta_d$ 为域难度（默认 0.2/0.5/0.8），$\phi_e$ 为调度因子：易域（$\delta\le0.5$）$\phi_e=\cos(\frac{e\pi}{2(E-1)})$ 从 1 衰减到 0；难域（$\delta>0.5$）$\phi_e=1-\cos(\frac{e\pi}{2(E-1)})$ 从 0 升到 1。即**易域先学、难域后攻**，全程保持损失归一化。

**（3）Fixed-Anchor 消融**：前 $N$（=25）个 epoch 冻结跨域锚点参数（Star 共享件 + 共享专家），仅训练私有组件，用于验证"共享锚点是否应在课程初期保持稳定"。

#### 【超参数新增/变动】

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_momentum_curriculum` | False | 总开关（分层动量 + 课程加权） |
| `--domain_difficulty` | "0.2,0.5,0.8" | 域难度 δ（d0 最易 → d2 最难，与实测 AUC 排序一致） |
| `--curriculum_type` | cosine | 课程调度形（cosine / linear） |
| `--fixed_anchor_epochs` | 0（关闭） | 锚点冻结轮数（消融用 25） |
| 分组 β₁ | 0.99 / 0.95 / 0.90 / 0.80 | 见 `_build_momentum_param_groups` |
| 权重裁剪 | [0.5, 2.0] | 防梯度消失 |
| **隐含变动** | loss 归一化 | `sum(6 cell BCE)` → `加权/(T·Σw)`；MC 关闭时等价于取均值 |

**实证结果**（`v1_momentum/examples/output/ablation_summary.csv`，seed 2022）：

| 方法 | 总 AUC | 总 LogLoss | 最弱 cell d2t0 AUC |
|---|---|---|---|
| Baseline | 0.7688 | 0.4037 | 0.7449 |
| FixedAnchor(25) | 0.7672 | **0.3968** | 0.7413 |
| M3OE-MC | 0.7683 | 0.3994 | 0.7430 |

后续调优运行的 MC 配置稳定在 **0.7748–0.7752**（`MDMTRec_MC_*.csv` 后段），`MDMTRec_MC_GateCurriculum`（课程权重同步作用于门控）达 **0.7755**（相对基线 +0.67pt）。logloss 一致性上 FixedAnchor/MC 均优于 Baseline，AUC 提升主要体现在 d2 的两个 cell（0.7430→0.7534 档）。

#### 【实验结果全录】v1_momentum 的逐 cell AUC 与 LogLoss（自 `v1_momentum/examples/output/*.csv` 提取，2026-09-06 补齐）

**表 V-A｜消融协议（`ablation_summary.csv`；总 AUC/LogLoss 与上方消融表逐项一致，逐 cell 为本次补齐）**

| 配置 | 总 AUC | 总 LogLoss | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.7688 | 0.4037 | 0.7655 | 0.7782 | 0.7726 | 0.7902 | 0.7449 | 0.7564 |
| FixedAnchor(25) | 0.7672 | 0.3968 | 0.7589 | 0.7757 | 0.7693 | 0.7914 | 0.7413 | 0.7602 |
| M3OE-MC | 0.7683 | 0.3994 | 0.7618 | 0.7789 | 0.7743 | 0.7905 | 0.7430 | 0.7576 |

**表 V-B｜消融协议逐 cell LogLoss**

| 配置 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|
| Baseline | 0.6044 | 0.4396 | 0.5826 | 0.4126 | 0.5993 | 0.4568 |
| FixedAnchor(25) | 0.6040 | 0.4143 | 0.5746 | 0.3837 | 0.5873 | 0.4359 |
| M3OE-MC | 0.6135 | 0.4273 | 0.5733 | 0.3829 | 0.6013 | 0.4453 |

**表 V-C｜调优后运行（seed 2022；总 AUC 与文档 0.7748–0.7752 / 0.7755 吻合）**

| 运行 | 总 AUC | Δ vs 0.7688 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|---|
| 调优 MC（`MDMTRec_MC_*.csv` 第 4 个数据行） | 0.7752 | +0.64pt | 0.7689 | 0.7831 | 0.7760 | 0.7959 | 0.7534 | 0.7655 |
| 调优 MC（默认名 `MDMTRec_…_0.01.csv`） | 0.7752 | +0.64pt | 0.7687 | 0.7832 | 0.7762 | 0.7964 | 0.7530 | 0.7645 |
| 调优 MC（`MDMTRec_MC_*.csv` 第 5 个数据行） | 0.7748 | +0.60pt | 0.7685 | 0.7835 | 0.7759 | 0.7956 | 0.7530 | 0.7643 |
| GateCurriculum（`MDMTRec_MC_GateCurriculum_*.csv`） | 0.7755 | +0.67pt | 0.7695 | 0.7844 | 0.7766 | 0.7963 | 0.7527 | 0.7643 |

**表 V-D｜调优后运行 LogLoss（pooled + 逐 cell）**

| 运行 | pooled LL | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|
| 调优 MC 0.7752（MC 文件） | 0.4806 | 0.5910 | 0.4114 | 0.5723 | 0.3852 | 0.5797 | 0.4372 |
| 调优 MC 0.7752（默认名文件） | 0.4818 | 0.5901 | 0.4091 | 0.5723 | 0.3835 | 0.5806 | 0.4391 |
| 调优 MC 0.7748（MC 文件） | 0.4825 | 0.5919 | 0.4105 | 0.5725 | 0.3844 | 0.5802 | 0.4382 |
| GateCurriculum 0.7755 | 0.4654 | 0.5913 | 0.4106 | 0.5741 | 0.3855 | 0.5818 | 0.4414 |

**无标签/无法判定运行与判定边界（重要）**：

1. `MDMTRec_MC_*.csv` 中部另有 4 次无标签运行（总 AUC 0.7681 / 0.7682 / 0.7684 / 0.7659，消融协议量级）与默认名文件中 1 次离群运行（0.7611）——疑为 cosine/linear 课程或不同超参变体，**CSV 无配置信息，无法判定归属**。
2. `MDMTRec_MC_smoke_*.csv`（0.7305）为冒烟测试，非正式结果。
3. `MDMTRec_domain_all_task_all_exp_4_0.001.csv` 为**非 MTMD 模式**（表头 3 域×6 任务共 18 个 auc 列，AUC ~0.99/LogLoss ~0.06），与本文档任何实验均不对应；`…_0.003.csv` 为 lr=0.003 调试运行（0.9966 过拟合形态与 0.7930 病态逐 cell），文档无对应记载。三者**均不纳入记录**。
4. `MDMTRec_FixedAnchor_*.csv` 内混入一行 Baseline 数据（0.7688/0.4335），系 §1.4 引用的原始基线行复制，非 FixedAnchor 结果。

**对文档既有表述的确认与修正**：

- §1.4 的基线逐 cell 引用（0.7656/0.7776/0.7731/0.7901/0.7449/0.7561）= `MDMTRec_Baseline_*.csv` 第 2 个数据行，其 **pooled LogLoss 0.4335** 为本次补齐；该行与消融协议 Baseline 行（0.7655/…，pooled LL 0.4037）是**两次不同运行**（总 AUC 同为 0.7688）。
- "d2 两个 cell 0.7430→0.7534 档"确认：消融 MC d2t0 0.7430 → 调优 MC d2t0 0.7530/0.7534、GateCurriculum d2t0 0.7527；d2t1 同步 0.7576 → 0.7643–0.7655。
- 调优运行的 pooled LogLoss（0.4654–0.4825）**劣于**消融协议三配置（0.3968–0.4037）：两批运行的损失归一化口径不同（求和 vs 均值），LogLoss 不可跨批比较——引用时须注明口径。

---

### 2.3 v2_momentum — 动量机制消融实验场（vGRA + 六大优化器变体 + CAM）

#### 【修改动机与假设】

v1 证明"动量分层 + 课程"方向可行但**未回答归因问题**：收益究竟来自哪条通道？对难 cell（d2t0/d2t1，即 flat index 2/5）应采取哪种动量干预？v2 把 v1 重构为一个**全部开关默认关闭、打开任一开关即相对基线逐位可复现（bit-exact off-path）**的消融实验场，覆盖三类干预：

- **A. 参数组静态/调度类**（TAM、HTMW）：只改分组与 β₁ 取值；
- **B. 状态重置/分解类**（CVR、LRMD、MoMo-DN）：改 Adam 的 `exp_avg` / `exp_avg_sq` 递推本身；
- **C. 梯度/参数后处理类**（CAM、Polyak）与**模型侧插件**（vGRA）：在 backward→step 之间或参数轨迹上做文章。

#### 【代码级增量修改（Code Delta）】

**① 模型侧（`v2_momentum/MDMTRec/models/multi_domain/model.py`）**：

```diff
+ class VGRAAdapter(nn.Module):
+     """二阶矩膨胀驱动的门控残差适配器（仅用于难 cell）
+         h' = h + g_c · W2·ReLU(W1·h) + b2 ;  g_c = sigmoid(a_c·s_c + b_c)
+        - s_c 为 trainer 注入的外部标量信号（detached float），梯度只流向 a/b/W1/W2/b2
+        - W2、b2 零初始化 → 初始严格恒等映射；bottleneck=16"""
+
  class MDMTRec(nn.Module):
      ...
+     # 只读元数据标签（纯 Python 属性，不改 state_dict）：tower[i].cell_id=(t,d)、
+     # domain_expert[d].domain_id、task_expert[t].task_id —— 供优化器侧按 cell 分组
+     if os.environ.get('USE_VGRA', '0') == '1':   # 惰性创建：OFF 路径不加任何子模块/参数
+         self.vgra = nn.ModuleDict({str(i): VGRAAdapter(self.fcn_dim[-1], 16) for i in (2, 5)})
+         self._vgra_signal = {'2': 0.0, '5': 0.0}
      ...
+     # forward 尾部：仅当 vgra 存在时，对 fused_fea[2]/[5] 施加适配器（eval 路径沿用最近信号）
+     if hasattr(self, 'vgra'):
+         for i in (2, 5):
+             fused_fea[i] = self.vgra[str(i)](fused_fea[i], _vgra_signal.get(str(i), 0.0))
```

**② trainer 重构（`v2_momentum/MDMTRec/trainers/ctr_trainer.py`，约 2260 行）**：

- **开关解耦**：v1 的 `use_momentum_curriculum` 拆为 `use_hierarchical_momentum` 与 `use_curriculum_weighting` 两个独立开关；分组细化为 6 组（`star_shared_base` 0.99 / `shared_expert` 0.95 / `star_domain_modulation` 0.90 / `domain_task_expert` 0.90 / `gate` 0.80 / `tower_embedding` 0.90）；
- **环境变量注入通道**：所有实验开关支持 kwargs 与环境变量双入口（env 优先）：`USE_TAM / USE_VRESET / USE_WARMUP / USE_MOMO / USE_LOW_RANK / USE_POLYAK / USE_VGRA / USE_CAM / CAM_OBSERVE / SPEC_OBSERVE` 及各自超参（如 `TAM_BETA1_HARD`、`CAM_ALPHA`）；
- **互斥规则**（构造期校验）：WARMUP 与 TAM/VRESET 互斥；MOMO 与 TAM/VRESET/WARMUP/CAM 互斥；LOW_RANK、POLYAK 除与彼此组合（→`PolyakLowRankAdam`）外互斥；
- **优化器类选择链**：`PolyakLowRankAdam > ConditionalResetAdam > WarmupAdam > MoMoAdam > LowRankAdam > PolyakAdam > 基线 Adam`；
- **新增 6 个 Adam 子类**（全部默认不实例化）：

| 子类 | 实验代号 | 核心机制 | 关键方法 |
|---|---|---|---|
| `ConditionalResetAdam` | D4-CVR | 不重写 `step()`；`on_validation` 检测 cell AUC 平台期后**条件重置二阶矩** | `on_validation` / `_reset_cell_v` |
| `WarmupAdam` | D3-HTMW | 仅重写 `step()` 前置调度：`tower_hard` 组 β₁ 线性升温，更新数学全部委托原版 Adam | `_beta1_eff` |
| `MoMoAdam` | D5 MoMo-DN | 手写 Adam：共享参数维护**每域一阶动量**并做方向归一化加权融合；私有参数走精确单张量 Adam | `register_domain_grad` / `_momo_step_shared` |
| `LowRankAdam` | D6-LRMD | 对 5 个私有专家首层矩阵（256×64），把偏置校正后的一阶矩投影到**Top-k SVD 子空间**再更新 | `_select_k` / `_low_rank_update` |
| `PolyakAdam` | D7 | 原版 Adam 更新后维护参数轨迹 **EMA shadow**；验证/保存时切换到 shadow 权重 | `swap_to_shadow` / `restore_from_shadow` |
| `PolyakLowRankAdam` | R14 组合 | LRMD 在线去噪 × Polyak 报告点平均（`POLYAK_EVAL_OFF=1` 可关闭其 swap） | 继承二者 |

- **trainer 侧新增通道**：`_build_tam_param_groups`（TAM/HTMW 分组）、`_apply_cam_correction`（CAM）、`_vgra_compute_signal/_vgra_inject_signal/_vgra_cache_g2`（vGRA 信号）、`_collect_spec_matrix_params/_spec_spectrum/_log_spec_stats`（谱观测）、`get_curriculum_weights` / `_set_anchor_requires_grad`（沿用 v1）；
- **训练循环介入点顺序**（`train_one_epoch`）：注入 vGRA 信号 → 前向 → 课程加权/均值损失 → （MoMo：3 次域隔离 backward；否则单次 backward + CAM 校正）→ 谱梯度快照（仅每 epoch 第 0 步）→ 缓存 g² → `optimizer.step()`；每 epoch 末统一落盘各通道统计到 `logs/momentum_stats_{tag}.log`；
- **`fit` 钩子**：`steps_per_epoch` 交接（WarmupAdam 调度锚定）、Polyak 的 `swap_to_shadow/restore_from_shadow` 环绕验证与保存、CVR 的 `on_validation(auc_list)` 只读回调。

#### 【数学与机制变化】

以下 $\mathcal{L}_{total}$ 默认仍为 v1 的课程加权形式（开关关闭时为 6 cell 均值 BCE），**各通道均不引入新的损失项**，只作用于优化动力学；唯一新增损失耦合是 vGRA 的可学习参数（仍由主损失端到端训练）。

**（1）D1-TAM（Task-Aware Momentum，静态）**：`tower_embedding` 组按 cell 拆分为 `tower_hard`（默认 cell {2,5}，β₁=`tam_beta1_hard`=0.95）/ `tower_easy`（0.90）/ `embedding`（0.90），其余 5 组与分层基线完全一致。取 `tam_beta1_hard=0.90` 时数值上等于基线。

**（2）D3-HTMW（Hard-Task Momentum Warmup，开环调度）**：

$$
\beta_1^{\text{eff}}(e)=\beta_{\text{start}}+(\beta_{\text{end}}-\beta_{\text{start}})\cdot\frac{e+1}{W}\quad(e<W),\ \text{否则}\ \beta_{\text{end}};\qquad \beta_{\text{start}}{=}0.5,\ \beta_{\text{end}}{=}0.95,\ W{=}6
$$

仅作用于 `tower_hard` 组；bias-correction 仍按常数 β₁ 近似（文档声明的有界误差）。

**（3）D4-CVR（Cell-wise Conditional Second-Moment Reset，闭环触发）**：对监控 cell 的 AUC 连续 `patience=2` 轮提升不超过 `tol=5e-4`、距上次重置 ≥ `cooldown=3` 轮且累计 < `max=2` 次时，在该 cell 全部参数上执行

$$
v_e \leftarrow (1-\beta_2^{s_p})\cdot\max\big(\overline{\hat v}_{\text{cell}},\,10^{-10}\big),\qquad \hat v=\frac{\text{exp\_avg\_sq}}{1-\beta_2^{s_p}}
$$

保持 bias-correction 严格连续（下一步 $\hat v$ 从均值出发）、cell 级均值尺度守恒，无阶跃爆炸。

**（4）D5 MoMo-DN（Multi-Objective Momentum, Direction-Normalized）**：每 batch 1 次前向 + **3 次域隔离 backward**（$\mathcal L_d=(\ell_{t0,d}+\ell_{t1,d})/2$，以 $\mathcal L_d/D$ 反传使存储梯度精确等于基线贡献）。共享参数 $p$ 的更新（WD 拆分 $wd/D$ 保证 uniform 模式精确退化为基线）：

$$
m_d \leftarrow \beta_1 m_d + (1-\beta_1)\big(g_d+\tfrac{wd}{D}p\big),\qquad
v \leftarrow \beta_2 v + (1-\beta_2)\big(g_{\text{comb}}+wd\cdot p\big)^2
$$

$$
\text{dirnorm 模式：}\ n_d=\|m_d\|,\ s=\tfrac{1}{D}\textstyle\sum_j n_j,\ w_d=\frac{1/(n_d+\varepsilon_n)}{\sum_j 1/(n_j+\varepsilon_n)},\quad
m_{\text{comb}}=D\,s\sum_d w_d\,\frac{m_d}{n_d+\varepsilon_n}
$$

$$
p \leftarrow p-\frac{lr}{1-\beta_1^t}\cdot\frac{m_{\text{comb}}}{\sqrt{v/(1-\beta_2^t)}+\varepsilon}
$$

弱信号域（动量范数小，如 d2）的方向获得更大权重；$\|m_{\text{comb}}\|\le Ds=\sum_j n_j$ 保证步长不超基线对齐尺度。

**（5）D6-LRMD（Low-Rank Momentum Decomposition）**：对目标矩阵 $P$（5 个私有专家首层 256×64），每 $N{=}20$ 步对 $\hat m$ 做 SVD 并缓存 Top-$k$ 基（$k$ 默认自动 = 谱能量 90% 的最小值，上限 32），中间步复用缓存：

$$
m_k=U_k\big(U_k^{\top}\hat m\,V_k\big)V_k^{\top},\qquad
P \leftarrow P - lr\cdot m_k\big/\big(\sqrt{v/(1-\beta_2^t)}+\varepsilon\big)
$$

二阶矩路径与基线完全一致，投影只替换更新方向。配套 **Round-A 谱观测**（只读）：$\hat m$ 与第 0 步梯度快照的 Top-16 奇异值、$k\in\{4,8,16,32\}$ 能量占比、90% 能量有效秩、Eckart–Young 投影误差 $\sqrt{1-e_k}$。

**（6）D7-Polyak（参数空间 EMA）**：每次原版 Adam 更新后

$$
\text{shadow} \leftarrow \delta\cdot\text{shadow}+(1-\delta)\cdot p,\qquad \delta=\text{polyak\_decay}=0.995
$$

验证、EarlyStopper 记账与最终 `torch.save` 均作用于 shadow 权重（早停路径免二次 swap）。

**（7）CAM（Conflict-Aware Momentum，梯度注入校正）**：对 $g\cdot m<0$ 的冲突元素，目标更新为 $m_t=(1-\alpha)\beta_1 m_{t-1}+(1-\beta_1)g_t$，等价改写为对梯度注入 $g_t \leftarrow g_t-\kappa m_{t-1}$，$\kappa=\alpha\beta_1/(1-\beta_1)$；实施时加幅值帽：

$$
\mathrm{inj}=\mathrm{sign}(m)\cdot\min\big(\kappa|m|,\ \rho|g|\big),\qquad \alpha{=}0.5,\ \rho{=}2.0;\quad p.\mathrm{grad}\ \mathrel{-}=\ \mathrm{inj}
$$

只改 `p.grad` 不动 `optimizer.state`；`CAM_OBSERVE=1` 时为只读统计（冲突率、$\cos(g,m)$、$\overline{\hat v}/\overline{g^2}$ 膨胀诊断）。

**（8）P2-A1 vGRA（模型侧门控残差适配器，与优化器通道正交可叠加）**：耦合信号取自优化器二阶矩统计（trainer 只读计算、detached 注入）：

$$
s_c=\log\frac{\overline{\hat v}_c}{\overline{g_c^2}},\qquad
g_c=\sigma(a_c s_c+b_c),\qquad
h'=h+g_c\,W_2\,\mathrm{ReLU}(W_1 h)+b_2
$$

$W_2,b_2$ 零初始化（初始严格恒等）；仅在难 cell 2/5 的 `fused_fea` 上生效（bottleneck 16，每 cell 2130 参数）。

#### 【超参数新增/变动】

| 通道 | 关键新增超参（默认值） | 说明 |
|---|---|---|
| v1 沿用 | `use_hierarchical_momentum`(False)、`use_curriculum_weighting`(False)、`domain_difficulty`("0.2,0.5,0.8")、`fixed_anchor_epochs`(0) | 由 `use_momentum_curriculum` 拆分 |
| TAM | `USE_TAM`(0)、`TAM_BETA1_HARD`(0.95)、`TAM_HARD_CELLS`("2,5") | 静态 β₁ 提升 |
| HTMW | `USE_WARMUP`(0)、`WARMUP_BETA1_START`(0.5)、`WARMUP_BETA1_END`(0.95)、`WARMUP_EPOCHS`(6)、`WARMUP_CELLS`("2,5") | 开环 β₁ 升温 |
| CVR | `USE_VRESET`(0)、`VRESET_PATIENCE`(2)、`VRESET_TOL`(5e-4)、`VRESET_COOLDOWN`(3)、`VRESET_MAX`(2)、`VRESET_CELLS`("2,5") | 闭环二阶矩重置 |
| MoMo-DN | `USE_MOMO`(0)、`MOMO_WEIGHT_MODE`("dirnorm"/"uniform")、`MOMO_EPS_N`(1e-12) | 多目标动量融合 |
| LRMD | `USE_LOW_RANK`(0)、`LOW_RANK_K`(0=自动90%)、`LOW_RANK_N`(20) | 低秩动量投影 |
| Polyak | `USE_POLYAK`(0)、`POLYAK_DECAY`(0.995)、`POLYAK_EVAL_OFF`(0) | 参数 EMA（R14 组合开关） |
| CAM | `USE_CAM`(0)、`CAM_ALPHA`(0.5)、`CAM_RHO`(2.0)、`CAM_OBSERVE`(0) | 冲突梯度校正/观测 |
| vGRA | `USE_VGRA`(0，**模型与 trainer 两侧需同开**)、bottleneck(16)、目标 cell(2,5) | 门控残差适配器 |
| 观测 | `SPEC_OBSERVE`(0)、各通道 `*_LOG_TAG` | 只读统计落盘 `logs/` |

**实证参考（修订 2026-09-06）**：v1 消融协议数字（Baseline 0.7688 / FixedAnchor(25) 0.7672 / MC 0.7683，见 `examples/output/ablation_summary.csv`）属 v1 代码谱系；v2 实验营（R0'–R14）有**独立基线体系**（见下方【实验结果全录】），两套数字不可直接比较。KuaiRand 两组运行 0.6701 / 0.6710（基线 0.6747）的对应配置未在当前 `v2_momentum/logs/` 存档中（该目录 82 个日志均为 ML-1M）。v2 各通道主结果的完整 AUC / LogLoss 记录（6 cell + 总量）见下。默认配置（全部开关关闭）与基线逐位一致，用于对照的可信性保障。

#### 【实验结果全录】v2 实验营（R0'–R14）全部配置的 AUC 与 LogLoss（自 `v2_momentum/logs/*.log` 末尾测试块提取，2026-09-06 补齐）

**基线体系（RESULTS.md 裁决 3，双基线协议）**：

- 实验营统一命令 = `--use_hierarchical_momentum`（分层动量**开**、课程权重**关**）+ 各环境变量通道；全部 Δ 的基准为 **R0' = 0.7721507（test logloss 0.461217）**；
- R0''（再叠加 `--use_curriculum_weighting`）= **0.7741784**，逐位复现 v5/v6 锚点（环境自洽校验）；两基线差 **+0.002028 即课程加权通道的效应量**（该通道已封闭、不再干预）；
- 等价性护栏：T1_default_R1/R2/R4、T2_observe_R1、d7_t1_equiv、r5_t1_equiv、r6_t1_default、r7_t1_default、r7_t2_spec、rt_t1_baseline、r14_t1_default 共 11 个默认/观测轮全部与 R0' 逐位一致（0.7721506503976439）。

**表 A｜单种子主实验 test AUC（seed=2022，Δ 基准 = R0'）**

| 配置 | 总 AUC | Δ | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | 终审（RESULTS.md） |
|---|---|---|---|---|---|---|---|---|---|
| R0' 基线（分层动量开/课程关） | 0.772151 | — | 0.763857 | 0.778475 | 0.774814 | 0.792444 | 0.747955 | 0.765934 | Δ 基准 |
| R0'' 锚点校验（+课程权重） | 0.774178 | +0.002028 | 0.766934 | 0.781720 | 0.776450 | 0.794864 | 0.748132 | 0.767461 | 环境锚 |
| R1 CAM（α=0.5，ρ=2.0） | 0.767433 | −0.004718 | 0.761338 | 0.777115 | 0.769741 | 0.790111 | 0.742397 | 0.757280 | 封闭（有害，6 cell 全降） |
| R2 TAM β1_hard=0.95 | 0.772105 | −0.000046 | 0.763618 | 0.777374 | 0.775111 | 0.793160 | 0.749307 | 0.766364 | 存疑（<σ） |
| R3 TAM β1_hard=0.97 | 0.772135 | −0.000016 | 0.763786 | 0.778461 | 0.774805 | 0.792992 | 0.749239 | 0.766288 | 存疑（剂量峰） |
| R3 TAM β1_hard=0.99 | 0.771750 | −0.000401 | 0.764497 | 0.780475 | 0.775401 | 0.793701 | 0.750882 | 0.764040 | 封闭（d2t1 护栏破） |
| R4 CVR 二阶矩条件重置 | 0.772358 | +0.000207 | 0.763907 | 0.778565 | 0.774830 | 0.792507 | 0.748142 | 0.765803 | 封闭（<σ，触发精确但非 d2 定向） |
| R5 HTMW β1 预热 0.5→0.95 | 0.772508 | +0.000358 | 0.765751 | 0.782177 | 0.776624 | 0.794584 | 0.751669 | 0.764276 | 封闭（<σ，d2t1 越哨兵线） |
| R6 MoMo uniform（管线对照） | 0.772042 | −0.000108 | 0.763833 | 0.778463 | 0.774726 | 0.792361 | 0.747880 | 0.765688 | 管线漂移 −1.1e-4 |
| R6 MoMo dirnorm | 0.771637 | −0.000514 | 0.766758 | 0.783100 | 0.771138 | 0.786792 | 0.751356 | 0.764936 | 封闭（d1 双 cell 破 −0.002 红线） |
| R7 LRMD 低秩动量投影 | 0.775084 | +0.002933 | 0.766463 | 0.781601 | 0.779939 | 0.795228 | 0.753354 | 0.765704 | 跨种子拒（表 C） |
| R9 Polyak decay=0.995 | 0.776768 | +0.004617 | 0.771791 | 0.785682 | 0.779376 | 0.797905 | 0.753366 | 0.768045 | **保留 ✅（表 C）** |
| R11 vGRA-only | 0.774437 | +0.002286 | 0.765080 | 0.780948 | 0.776126 | 0.794760 | 0.751866 | 0.765506 | 跨种子拒（表 C） |
| R11 vGRA+Polyak 联合 | 0.776838 | +0.004687 | 0.772525 | 0.786825 | 0.779114 | 0.798312 | 0.755069 | 0.769221 | 封闭（交互 −0.0022 亚可加） |
| R13 Polyak decay=0.99 | 0.776703 | +0.004552 | 0.770578 | 0.785217 | 0.779301 | 0.797022 | 0.753803 | 0.768445 | 平台档（不升级） |
| R13 Polyak decay=0.998 | 0.770116 | −0.002035 | 0.771317 | 0.784554 | 0.775981 | 0.794976 | 0.748348 | 0.763238 | 封闭（崩落，双护栏破） |
| R14 Polyak×LRMD 组合 | 0.778063 | +0.005912 | 0.772669 | 0.785853 | 0.780994 | 0.798216 | 0.754744 | 0.768443 | 封闭（vs Polyak +0.0013 < +0.0020 升级线） |
| R14 兼容性轮（影子禁用） | 0.775084 | +0.002933 | 0.766463 | 0.781601 | 0.779939 | 0.795228 | 0.753354 | 0.765704 | 逐位 = R7 LRMD ✅ |

**表 B｜单种子主实验 test LogLoss（seed=2022）**

| 配置 | 总 LL | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|---|
| R0' 基线（分层动量开/课程关） | 0.461217 | 0.588503 | 0.408171 | 0.568486 | 0.382841 | 0.580423 | 0.432658 |
| R0'' 锚点校验（+课程权重） | 0.455905 | 0.586550 | 0.408525 | 0.568885 | 0.382937 | 0.581436 | 0.432119 |
| R1 CAM（α=0.5，ρ=2.0） | 0.455034 | 0.598889 | 0.423770 | 0.578886 | 0.386086 | 0.592174 | 0.442453 |
| R2 TAM β1_hard=0.95 | 0.463315 | 0.587952 | 0.408000 | 0.568673 | 0.384050 | 0.580156 | 0.431783 |
| R3 TAM β1_hard=0.97 | 0.451223 | 0.591431 | 0.407639 | 0.572954 | 0.384457 | 0.581367 | 0.433567 |
| R3 TAM β1_hard=0.99 | 0.491361 | 0.595523 | 0.407583 | 0.571779 | 0.380980 | 0.582932 | 0.434697 |
| R4 CVR 二阶矩条件重置 | 0.451803 | 0.586366 | 0.411177 | 0.569462 | 0.385675 | 0.581513 | 0.433856 |
| R5 HTMW β1 预热 0.5→0.95 | 0.475612 | 0.598401 | 0.410070 | 0.576539 | 0.385336 | 0.583992 | 0.438714 |
| R6 MoMo uniform（管线对照） | 0.461452 | 0.588806 | 0.408595 | 0.568626 | 0.382709 | 0.580481 | 0.432782 |
| R6 MoMo dirnorm | 0.452249 | 0.592659 | 0.404368 | 0.576160 | 0.387812 | 0.581119 | 0.433433 |
| R7 LRMD 低秩动量投影 | 0.475789 | 0.594750 | 0.412933 | 0.568973 | 0.385139 | 0.582808 | 0.438228 |
| R9 Polyak decay=0.995 | 0.442746 | 0.586880 | 0.413926 | 0.568190 | 0.383777 | 0.578465 | 0.431032 |
| R11 vGRA-only | 0.422520 | 0.600953 | 0.422441 | 0.580237 | 0.385402 | 0.588735 | 0.445085 |
| R11 vGRA+Polyak 联合 | 0.432135 | 0.599245 | 0.415597 | 0.574789 | 0.386282 | 0.578633 | 0.430145 |
| R13 Polyak decay=0.99 | 0.432901 | 0.602093 | 0.421714 | 0.577991 | 0.392670 | 0.586623 | 0.439628 |
| R13 Polyak decay=0.998 | 0.478847 | 0.582538 | 0.405830 | 0.573862 | 0.381050 | 0.580821 | 0.437395 |
| R14 Polyak×LRMD 组合 | 0.461715 | 0.584829 | 0.411040 | 0.564357 | 0.378582 | 0.577559 | 0.431481 |
| R14 兼容性轮（影子禁用） | 0.475789 | 0.594750 | 0.412933 | 0.568973 | 0.385139 | 0.582808 | 0.438228 |

**表 C｜跨种子确认（n=6 配对，seeds 2022–2027；2022 列来自主实验）**

| 配置 | s2022 | s2023 | s2024 | s2025 | s2026 | s2027 | 配对 Δ 均值 ± SD | t | p |
|---|---|---|---|---|---|---|---|---|---|
| R0' 基线 | 0.772151 | 0.773074 | 0.774132 | 0.774005 | 0.772826 | 0.775627 | — | — | — |
| LRMD | 0.775084 | 0.773217 | 0.771761 | 0.774449 | 0.773973 | 0.774740 | +0.000235 ± 0.001801 | 0.319 | ≫0.05 |
| Polyak 0.995 | 0.776768 | 0.777223 | 0.775994 | 0.779360 | 0.777674 | 0.778931 | **+0.004022 ± 0.001266** | 7.785 | **0.00056** |
| vGRA-only | 0.774437 | 0.773192 | 0.771867 | 0.772621 | 0.774336 | 0.771562 | −0.000633 ± 0.002395 | −0.648 | 0.546 |

**表 C-2｜跨种子逐 cell 配对 Δ 均值**（自 `xs_*.log` 配对重算；Polyak 行与 RESULTS.md R10 逐项一致）

| 配置 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---|---|---|---|
| LRMD | +0.00086 | +0.00209 | −0.00137 | −0.00017 | +0.00196 | +0.00061 |
| Polyak 0.995 | +0.00708 | +0.00594 | +0.00310 | +0.00507 | +0.00390 | +0.00318 |
| vGRA-only | −0.00055 | +0.00044 | −0.00112 | −0.00060 | +0.00050 | +0.00027 |

**结果读数**：

1. **唯一跨种子保留 = PolyakAdam（R9/R10）**：总 AUC 单种子 +0.004617（≈3σ），跨种子 mean +0.004022（t=7.785，p=0.00056），6/6 种子全正、36/36 cell·种子组合全正；d2t1（对前 7 轮梯度动量空间干预免疫的 cell）首次显著为正（+0.00318，p=0.00039）；总 LogLoss 0.4427 与 AUC 同步改善。
2. **CAM 是全表最大劣化**（−0.004718，6 cell 全降，d2t0 0.742397 为全表最低）——元素级"冲突"以噪声为主，衰减动量即摧毁唯一信号平均器，失稳正反馈。
3. **TAM 剂量响应不对称**：d2t0 随 β1_hard 单调升（0.749307→0.749239→0.750882），d2t1 反向降（0.766364→0.766288→0.764040），0.99 档 d2t1 护栏失败——两个难 cell 不能共用统一 hard 档。
4. **HTMW**：d2t0 +0.0037 但 d2t1 −0.0017 且总 LogLoss 恶化至 0.4756；val 全程正向、best 段退化为静态行为。
5. **MoMo dirnorm 域间零和再分配**：d0 两 cell +0.0029/+0.0046，d1 两 cell −0.0037/−0.0057；机制层证伪（域间动量 cos ≥0.9，无拖曳对象）。
6. **LRMD**：单种子全面上涨且 d2t0 涨幅最大（0.753354，+0.0054），但跨种子均值归零，呈"d0/d2 受益、d1 代价"的零和形态（表 C-2）。
7. **vGRA-only 总 LogLoss 全表最低（0.422520）但跨种子 null**——logloss 单一指标不可作为保留依据的实例。
8. **Polyak 剂量倒 U 型**：0.99（0.776703）→ **0.995（0.776768，峰）** → 0.998（0.770116，崩落且总 LL 0.478847）；0.995 确认为最终剂量。
9. **R14 组合单种子最优（0.778063）但未过跨种子升级线**（vs Polyak-only 仅 +0.001295 < +0.0020）；可加性 78.3%，LRMD 与 Polyak 作用于同一噪声源。

**数据来源与勘误**：

- 表 A/B 数值提取自 `v2_momentum/logs/*.log` 末尾测试块（脚本 `v2_momentum/extract_results.py`，汇总 `extracted_results.json`；表格生成 `gen_tables.py`、跨种子配对 `cell_paired.py`）；与 `RESULTS.md` 所载全部 Δ 值逐项核对一致，表 C/C-2 的配对统计与 R8/R10/R12 逐项吻合。
- **`有意义的log/P2-A1 vGRA 门控适配器.log` 与 `有意义的log/D7 单独.log` 为同一文件**（MD5 相同，归档失误）；真正的 vGRA 日志为 `logs/rt_t2_vgra_only.log` 与 `logs/rt_t3_vgra_polyak.log`。
- R0'（0.7721507）与 v1 消融协议基线（0.7688）不可比较：前者在分层动量开启、损失为 6 cell 均值归一化的代码谱系（v5→v7）上测得；引用时须注明所用基线体系。
