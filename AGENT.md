# My-Research-Project — M3oE / MDMTRec 项目上下文与 Codex 工作约定

> 本文件供 Codex 在 `My-Research-Project` 中进行代码阅读、修改、调试、实验执行和科研分析时使用。
>
> 上游参考：`Applied-Machine-Learning-Lab/M3oE`，审阅基准 `main@1b30217477836cac8f9bb68329a91e1c805aa4f5`。
>
> **重要：本项目当前研究协议与上游 M3oE 不完全相同。V1/V2 有意移除原 M3oE 的可学习融合权重及 AutoML/DARTS 权重搜索/附加优化路径。此移除是后续实验的刻意控制条件，不是 bug、遗漏或需要恢复的功能。**

## 0. Codex 最高优先级规则

1. 这是科研实验代码。优先保证实验变量可辨识、可复现、可回滚，不进行无关重构。
2. 必须区分：上游 M3oE baseline、当前固定实验底座、V1 DBO-inspired reweighting、V2 gradient magnitude balancing，以及真正的代码 bug。
3. **不得恢复被有意移除的 `_weight_*` 可学习融合权重、AutoML/DARTS 权重搜索或其附加 optimizer/update。** 如果需要研究这些机制，必须作为独立 baseline/ablation 显式启用。
4. 任何训练语义改变都必须明确记录：loss、domain/task 聚合方式、optimizer、参数归属、batch size、seed、数据 split、checkpoint、软件/CUDA/PyTorch 版本。
5. 不得把“稀缺域”自动等同于“困难域”，也不得预设某种权重、梯度修正或 Adam 历史状态一定有益。
6. 未实际运行的 AUC、logloss、收敛速度、显存和稳定性一律标为“未测”。科研结论必须由当前代码、当前数据和当前实验支持。
7. 修改前先检查 `git status`、当前 commit 和实际模块路径。不要覆盖既有实验结果或 checkpoint。

---

# 1. 上游 M3oE 基础结构

上游项目核心类 `MDMTRec` 位于：

- `MDMTRec/models/multi_domain/model.py`
- `MDMTRec/trainers/ctr_trainer.py`
- `MDMTRec/basic/layers.py`
- `MDMTRec/basic/features.py`
- `MDMTRec/utils/data.py`
- `examples/run_movielens.py`

默认 MovieLens-1M 示例是 3 domains × 2 tasks。训练输入经过 `x.transpose(0,1)` 后，模型 `forward()` 接受 `[feature_num, batch_size]`；输出为 `[batch_size, task_num]` 的 sigmoid 概率。

原始主干：

```text
Embedding
  ↓
STAR / domain-specific transformation + skip
  ↓
shared experts + domain experts + task experts
  ↓
6 个 domain-task gate
  ↓
domain/task expert fusion
  ↓
6 个 domain-task towers
  ↓
按样本 domain 选择 tower
  ↓
[B, 2]
```

原始 3×2 Equal-style 训练目标是六个 `(domain, task)` 组内 BCE 均值求和：

\[
L_{\rm M3oE}
=
\sum_{t=1}^{T}\sum_{d=1}^{D}
\frac{1}{|B_d|}
\sum_{i\in B_d}
\ell_{i,t},
\qquad
\ell_{i,t}={\rm BCE}(p_{i,t},y_{i,t}).
\]

注意这里没有除以 `D*T`。它不是全数据集严格域均衡，而是 minibatch 内的 group-mean 写法。

---

# 2. 当前研究的统一实验底座

后续 V1/V2 实验默认建立在：

- 固定 `exp_d / exp_t / bal_d / bal_t`；
- 单个 Adam；
- 不使用原 M3oE `_weight_*` learned parameters；
- 不执行 AutoML/DARTS 权重搜索或 epoch-level 附加更新；
- 明确记录固定融合系数；
- 在此固定底座上再加入新的研究变量。

## 2.1 为什么必须保持这个设计

原 M3oE 的 `_weight_exp_d()`、`_weight_exp_t()`、`_weight_bal_d()`、`_weight_bal_t()` 是可学习参数，经 `sigmoid` 得到 effective weights；原 trainer 还存在针对这些参数的额外 optimizer/update 路径。

当前研究**主动删除这些机制**，目的不是修复代码，而是防止 AutoML/learned fusion 在后续实验中成为额外自由度，从而干扰对 DBO、sample reweighting、gradient balancing 等研究变量的判断。

因此：

- 不要为了兼容旧 CSV 而重新加入 `_weight_*`；
- 不要把固定值伪装成 learned effective weight；
- 不要因为旧脚本读取 `_weight_*` 就恢复 learned weights；
- 若需要上游对照，单独建立 `M3oE-original` baseline；
- 当前研究底座本身就是一个**有意定义的实验条件**。

与上游比较时仍应承认协议不同，但应描述为“研究设计差异”，而不是“V1 的错误”。

---

# 3. V1：DBO-inspired sample reweighting

## 3.1 文件

V1 由三个修改文件组成：

```text
layers(1).py
model(2).py
ctr_trainer(2).py
```

对应：

```text
MDMTRec/basic/layers.py
MDMTRec/models/multi_domain/model.py
MDMTRec/trainers/ctr_trainer.py
```

V1 增加：

- `DBO_Complexity_Layer`
- `DBO_Specificity_Layer`
- sample-level complexity weight
- sample-level specificity weight
- weighted BCE

同时保留 M3oE 主干 STAR + shared/domain/task experts + gate + tower。

## 3.2 Complexity

对每个样本的 field embedding：

\[
X_i\in\mathbb R^{F\times E}.
\]

V1：

\[
Q_i=X_iW_Q,\qquad K_i=X_iW_K,
\]

\[
S_i=\frac{Q_iK_i^\top}{\sqrt{E}},
\]

\[
A_i={\rm softmax}(S_i).
\]

然后计算平均 attention entropy：

\[
c_i
=
-\frac1F
\sum_{a=1}^{F}\sum_{b=1}^{F}
A_{i,a,b}\log(A_{i,a,b}+\epsilon).
\]

这里的 `c_i` 是模型当前 embedding/attention 状态下的 field-level attention entropy，不能未经实验直接称为“真实 domain difficulty”。

V1 在 batch 内标准化：

\[
\mu_B={\rm mean}(c_i),\qquad
\sigma_B={\rm std}(c_i)+10^{-9},
\]

\[
c'_i=
{\rm clip}
\left(
\frac{c_i-\mu_B}{\sigma_B}+0.5,
0.1,1.0
\right),
\]

\[
w_i^{cpl}=0.1+0.5c'_i.
\]

因此有限值下：

\[
0.15\le w_i^{cpl}\le0.60.
\]

### 与 D3 的差异

当前代码不是 D3 的逐式复制：

1. V1 标准化没有 `/2`；
2. V1 使用标准 Shannon entropy 的负号；
3. attention scaling 使用 `sqrt(E)`，其中 `E` 是 field embedding 最大维，而不是 `att_dim`；
4. V1 的 attention 只用于计算 loss weight，不直接参与 D3 对应的主干 shared-specific fusion。

因此只能称 **DBO-inspired**，不能称严格 D3 reproduction。

### 数值风险

`torch.std()` 默认使用 sample std。若 batch 只有一个样本，`std` 可能为 NaN，后续 weight 也会 NaN。正式实验必须覆盖 `batch_size=1` 和极端小 batch。

## 3.3 Specificity

V1 对展平 embedding：

\[
h_i\in\mathbb R^{H}
\]

执行：

\[
v_i=W_2{\rm ReLU}(W_1h_i+b_1)+b_2,
\qquad v_i\in\mathbb R^2,
\]

\[
(g_{i,sp},g_{i,sh})={\rm softmax}(v_i),
\]

\[
w_i^{spf}=0.1+0.5g_{i,sp}.
\]

注意该 gate **没有实际控制主干 shared/specific fusion**；其输出只进入 loss weight。因此它更准确地是一个 learnable sample-level loss gate，而不是已经被证明的 domain-specificity estimator。

## 3.4 V1 最终 loss

默认 3 domains × 2 tasks：

\[
w_i=w_i^{cpl}w_i^{spf}.
\]

\[
L_{\rm V1}
=
\sum_{t=1}^{T}\sum_{d=1}^{D}
\frac1{|B_d|}
\sum_{i\in B_d}
w_i\ell_{i,t}.
\]

它仍然是 domain-task group mean，但组内样本贡献被重新加权。

因为 weight 没有 `detach()`：

\[
\nabla(w_i\ell_i)
=
w_i\nabla\ell_i+
\ell_i\nabla w_i.
\]

因此辅助 weight network 不只是“计算一个外部权重”，它自身也可能通过降低权重来降低训练目标。不能假设 learned weight 一定学习出“更重要样本”。

---

# 4. V1 与上游的差异必须如何解释

以下变化全部是**有意的实验协议**：

### 4.1 learned fusion → fixed fusion

原：

\[
\sigma(\text{learned parameter})
\]

当前：

\[
e_d,e_t,b_d,b_t
\]

为固定 Python 标量。

这意味着跨 domain/task expert 的混合强度发生改变，但这是研究设计的一部分。

### 4.2 AutoML/DARTS → disabled

当前：

- 单 Adam；
- 不再有 `_weight_*` 参数；
- `optimizer_params_darts` 不参与；
- 不执行原 epoch-level learned-weight update。

**Codex 不得把这一变化写成 bug fix。**

### 4.3 上游 run script

如果旧 `run_movielens.py` 读取 `_weight_*`：

正确处理：

```text
删除 learned-weight 日志字段
或改成记录 fixed coefficients + AutoML/DARTS disabled
```

错误处理：

```text
为了兼容旧 CSV 恢复 `_weight_*`
```

---

# 5. V2：specificity-only + domain-task gradient magnitude balancing

## 5.1 文件关系

第四个文件：

```text
ctr_trainer加入梯度修正(2).py
```

是**独立的第二版本**，不是 V1 trainer 的简单补丁。

本次材料没有提供 V2 配套 model。

V2 trainer 要求：

```python
model(
    x_input,
    aug_mask=aug_mask,
    noise_std=0.1
)
```

而 V1 `MDMTRec.forward()` 不接受这些参数。因此不要直接把 V2 trainer 和 V1 model 拼接；这属于独立版本的配套接口问题，不代表需要恢复 V1。

## 5.2 V2 的 specificity-only weight

V2 仍计算 complexity，但最终：

```python
weights_raw = w_spf_raw
```

即：

\[
w_i=0.1+0.5g_{i,sp}.
\]

因此 V2 **有意只研究 specificity weight**。

不要擅自恢复：

\[
w_i=w_i^{cpl}w_i^{spf}.
\]

如果要比较 complexity 的作用，应另设 ablation。

## 5.3 稀缺域复制

设目标稀缺域为 \(d_*\)：

\[
B_*=\{i:d_i=d_*\}.
\]

V2：

\[
B'=B\Vert B_*.
\]

即把当前 batch 中目标域样本复制一次。

trainer 本身只完成复制和 `aug_mask` 标记：

```text
original batch
    +
scarce-domain copy
    ↓
model(x_input, aug_mask, noise_std)
```

是否真的加入噪声、噪声在哪里加入、是否改变 embedding/feature，取决于 V2 配套 model；本文件没有提供这些实现，因此不能由 trainer 单独证明“noise augmentation 已实现”。

复制样本使用：

\[
\bar w_B=\frac1{|B|}\sum_iw_i
\]

作为统一权重，并 `detach()`。

重要：在 Equal-style domain-task group mean 中，**复制样本不自动等价于给稀缺域更大的 group-level 权重**。

---

# 6. V2：gradient magnitude adjustment

默认 3×2 情况，V2 不是对两个 task 求梯度，而是对六个 `(task,domain)` group 分别求梯度。

定义：

\[
L_{t,d}
=
\frac1{|B_d'|}
\sum_{i\in B_d'}
w_i\ell_{i,t}.
\]

每组梯度：

\[
g_{t,d}=\nabla_\theta L_{t,d}.
\]

全模型梯度范数：

\[
G_{t,d}
=
\|g_{t,d}\|_2
=
\sqrt{
\sum_{p\in\mathcal P}
\sum_a
(g_{t,d,p,a})^2
}.
\]

令：

\[
G_{\max}=\max_{t,d}G_{t,d}.
\]

V2 使用：

\[
s_{t,d}
=
(1-\tau)
+
\tau\frac{G_{\max}}{G_{t,d}},
\]

\[
\tilde g_{t,d}=s_{t,d}g_{t,d}.
\]

最后：

\[
g_{\rm final}
=
\sum_{t,d}\tilde g_{t,d}.
\]

然后：

```text
param.grad = g_final
Adam.step()
```

所以：

- \(\tau=0\)：不进行幅值修正；
- \(\tau=1\)：非零 group gradient norms 被拉到相同的最大范数；
- \(0<\tau<1\)：部分向最大范数靠拢。

对于 \(G_{t,d}>0\)：

\[
\|\tilde g_{t,d}\|
=
(1-\tau)G_{t,d}
+
\tau G_{\max}.
\]

## 6.1 V2 不是完整 GradCraft

V2 实现的是 **magnitude adjustment**。

它没有实现：

- task/group gradient cosine；
- negative conflict detection；
- global direction projection；
- 冲突梯度的投影消除。

因此科研文档中应写：

> `GradCraft-inspired gradient magnitude balancing`

而不是：

> `full GradCraft`

---

# 7. V2 梯度修正的重要解释

## 7.1 不一定等于“照顾稀缺域”

梯度范数小不等于样本少。

小域可能：

- 梯度大；
- 梯度小；
- 梯度噪声大；
- 已经饱和；
- 发生共享参数梯度抵消。

因此最大范数对齐不能直接解释为 scarce-domain balancing。

## 7.2 当前范数包含 private parameters

V2 对所有模型参数计算：

\[
\|g_{t,d}\|_2.
\]

其中包括：

- embedding；
- STAR；
- shared experts；
- domain experts；
- task experts；
- tower；
- 其他辅助网络。

很多 private parameter 只属于某一个 group，却仍参与该 group 的 norm。

因此必须额外实验：

\[
\|g^{shared}_{t,d}\|_2
\]

与

\[
\|g^{all}_{t,d}\|_2
\]

是否给出相同结论。

## 7.3 magnitude scaling 不改变单个 group 的方向

因为：

\[
\tilde g_{t,d}=s_{t,d}g_{t,d},
\quad s_{t,d}\ge0
\]

在合法 \(\tau\in[0,1]\) 且 \(G>0\) 时，单组方向不变。

如果原本：

\[
\langle g_i,g_j\rangle<0,
\]

单纯正比例缩放不会改变负夹角的符号。

所以 V2 **不能宣称解决 gradient conflict**。

---

# 8. V2 当前代码需要区分的真实 bug 与实验设计

## 真正需要修复/验证的代码问题

1. V2 缺域分支使用 `logloss_list.append(...)`，但 trainer 中创建的是 `task_losses`。这是实际变量名错误，应修复并做空组测试。
2. V2 的多次 `backward(retain_graph=True)` 会增加计算/显存成本，应 benchmark。
3. `grad_tau` 没有范围检查；正式实验需确保 \(\tau\in[0,1]\)。
4. 强制把原本 `grad=None` 的参数写成零 Tensor，可能改变 Adam 的 weight decay / state 行为，需要单独测试。
5. V1 singleton batch 的 `std` NaN 问题需要单独测试。

## **绝对不要当成 bug 的项目设计**

以下均是**有意设计**：

- 删除 `_weight_*` learned fusion；
- 删除 AutoML/DARTS optimizer/update；
- 固定 `exp_d/exp_t/bal_d/bal_t`；
- V2 从 `complexity × specificity` 拆成 specificity-only；
- V2 使用独立 trainer/model 接口；
- 后续实验以 fixed-fusion + single-Adam 为统一底座。

Codex 不得自行恢复。

---

# 9. 推荐实验矩阵

## 9.1 固定底座

| ID | 设置 | 目的 |
|---|---|---|
| B0 | 原 M3oE learned fusion + 原优化路径 | 上游参照 |
| B1 | fixed fusion + single Adam + no AutoML/DARTS + no DBO | **当前研究统一底座** |
| B2 | B1 + complexity-only | complexity 单因子 |
| B3 | B1 + specificity-only | specificity 单因子 |
| B4 | B1 + complexity×specificity | 双因子 |
| B5 | B4 + weight detach | 检查 weight-network 自我降低路径 |
| B6 | B4 + weight normalization | 分离绝对 loss scale |
| B7 | B4 + D3 `/2std` | 检查公式差异 |

B0 只是上游参照；不要把 B0 混入后续固定底座。

## 9.2 V2

| ID | 设置 | 目的 |
|---|---|---|
| G0 | specificity-only，关闭 gradient balancing，无增强 | V2 baseline |
| G1 | G0 + \(\tau=0\) | 验证开关等价性 |
| G2 | G0 + \(\tau=0.25,0.5,0.75,1\) | 梯度幅值强度 |
| G3 | 全参数 norm vs shared-only norm | private parameter 干扰 |
| G4 | scarce-domain copy，无噪声 | 判断 copy 本身作用 |
| G5 | copy + 真实 V2 model noise | 判断 augmentation |
| G6 | G2 + G5 | 交互作用 |
| G7 | 若研究完整 GradCraft，再独立加入 direction projection | 与 magnitude-only 对照 |

---

# 10. 必须记录的机制数据

正式实验不能只保存最终 AUC。

至少记录：

### 数据

- seed；
- Git commit；
- dataset/version；
- train/val/test split；
- 每域样本数；
- 每域正例率；
- batch size。

### V1

- `complexity` mean/std/min/max；
- `specificity` mean/std/min/max；
- `w_cpl`；
- `w_spf`；
- product weight；
- weight 与 BCE 的 correlation；
- weight network gradient；
- unweighted BCE；
- weighted BCE。

### V2

- 每个 `(task,domain)` 的 \(G_{t,d}\)；
- 每个 scale \(s_{t,d}\)；
- shared-only gradient norm；
- all-parameter gradient norm；
- gradient cosine matrix；
- negative-gradient-conflict rate；
- scarce-domain copy 数量；
- augmentation 前后预测/embedding 差异；
- wall time；
- peak CUDA memory；
- Adam state/update statistics。

### 性能

必须至少报告：

\[
AUC_{d,t},\quad
LogLoss_{d,t}
\]

以及明确公式的：

- macro AUC；
- macro logloss；
- sample-weighted aggregate。

---

# 11. Adam 与梯度实验的特别要求

不要把：

```text
gradient balancing
```

直接解释成：

```text
parameter update balancing
```

Adam 实际维护：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

因此 V2 的 `g_final` 会同时改变：

- 当前梯度；
- 一阶历史；
- 二阶历史；
- 后续 step 的有效更新。

如果研究 Adam 历史状态，checkpoint 必须至少保存：

- model state；
- optimizer state；
- scheduler state（若存在）；
- RNG state；
- epoch/step；
- Git commit。

只保存 `model.pth` 无法完整恢复 Adam 历史。

---

# 12. Codex 执行协议

当用户要求修改或运行 V1/V2 时，Codex 必须先报告：

```text
1. 当前 Git commit
2. 使用 V1 还是 V2
3. 实际 model/trainer 路径
4. forward 签名
5. learned fusion 是否存在
6. AutoML/DARTS 是否启用
7. optimizer 数量
8. loss 数学形式
9. domain-task 聚合方式
10. DBO weight 公式
11. gradient balancing 是否启用
12. scarce-domain augmentation 是否启用
```

然后：

1. 先做小规模 smoke test；
2. 再做单 epoch；
3. 再跑多 seed；
4. 保存原始日志和中间机制数据；
5. 不覆盖已有实验目录；
6. 结果按实验 ID 隔离；
7. 报告所有异常和未运行项目。

### 特别禁止

- 不得自动恢复 `_weight_*`；
- 不得自动恢复 AutoML/DARTS；
- 不得因为旧日志字段缺失而恢复 learned weights；
- 不得把 V1 称为严格 D3；
- 不得把 V2 称为完整 GradCraft；
- 不得把 complexity 在 V2 中自动补回；
- 不得把 magnitude adjustment 称为 conflict projection；
- 不得把 scarce domain 自动解释成 difficult domain；
- 不得仅凭最终平均 AUC 宣称机制有效。

---

# 13. 最终研究解释框架

当前两版应理解为两个连续但独立的实验方向：

### V1

\[
\boxed{
\text{Fixed M3oE base}
+
\text{DBO-inspired sample reweighting}
}
\]

研究问题：

> attention-derived complexity 与 sample-level specificity 是否能够提供比原始 Equal objective 更有用的样本重加权信号？

关键风险：

- weight self-suppression；
- batch-dependent normalization；
- complexity/specifity 与真实 difficulty 的可辨识性；
- D3 → MDMT 的迁移假设。

### V2

\[
\boxed{
\text{Fixed M3oE base}
+
\text{specificity-only weighting}
+
\text{domain-task gradient magnitude balancing}
+
\text{optional scarce-domain augmentation}
}
\]

研究问题：

> 当 MDMT 中六个 domain-task objective 的梯度规模不一致时，梯度幅值重平衡是否比单纯 loss reweighting 更有效？

关键风险：

- gradient norm 不等于 domain difficulty；
- private parameter 污染 norm；
- direction conflict 未解决；
- Adam history interaction；
- scarce-domain copy 不等于 group-level upweight；
- augmentation 依赖未提供的配套 model。

**当前没有真实多 seed AUC/logloss 证据支持任一版本优于其他版本。任何性能结论必须由后续实验产生。**

---

# 14. 上游参考

- M3oE：`https://github.com/Applied-Machine-Learning-Lab/M3oE`
- M3oE SIGIR 2024：`https://doi.org/10.1145/3626772.3657686`
- D3：`https://ojs.aaai.org/index.php/AAAI/article/view/28699/29354`
- GradCraft：`https://arxiv.org/html/2407.19682v2`
- GradCraft code：`https://github.com/baiyimeng/GradCraft`

> 本文件是 Codex 的项目上下文，不替代当前源码。若本地代码与本文不一致，以当前 Git checkout 的实际源码为准，并在实验报告中说明差异。
