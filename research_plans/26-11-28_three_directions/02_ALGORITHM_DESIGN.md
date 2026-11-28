# 02 — 候选算法数学定义与更新流程

## 0. 共同符号和训练目标

- 域任务格集合：\(\mathcal C=\{(d,t):d\in\{0,1,2\},t\in\{0,1\}\}\)。
- 一个样本行 \(i\) 的 M3oE 预测概率：\(p_{it}=f_{\theta}(x_i)_t\)，logit：\(z_{it}\)。
- \(\theta=(\theta_S,\phi)\)：\(\theta_S\) 是现有 optimizer manifest 中 S_DENSE 的共享参数，\(\phi\) 是其余参数。
- BCE 自然目标按项目当前 Sample/fixed1 训练实现执行。A/C pairwise 辅助目标使用 macro6 sampler 的六格宏平均权重；B 的 PAIR source 严格沿用 A21 natural sampler。单独记录两类 sampler 的原始 loss 和梯度范数。
- 固定 EMA 参数：\(\bar\theta_t=\rho\bar\theta_{t-1}+(1-\rho)\theta_t\)，使用仓库已冻结 dataset-specific rho；EMA 只读、不反馈 online optimizer。

## 1. A：Gaussian Pairwise Ranking Augmentation

### 1.1 训练 pair 定义与 Macro-AUC 目标

A/C 的 pairwise 辅助目标直接对齐六格 Macro AUC；B 则保持 A21 原有的 natural condition sampler。两者不得混用。

A/C 使用 Train-global pair pools，不从当前 BCE batch 临时抽 pair。对每个 domain-task cell \(c=(d,t)\)，令联合标签状态集合为 \(a,b\in\{0,1,2,3\}\)，正负组分别为 \(G_{d,a}\)、\(G_{d,b}\)。仅当状态 a 对 task t 为正、状态 b 对 task t 为负，且两组均非空时建立 condition \(i=(d,t,a,b)\)。计算
\[
r_i=\frac{|G_{d,a}|}{n^+_{d,t}}\frac{|G_{d,b}|}{n^-_{d,t}},
\quad Z_{d,t}=\sum_{i\in A_{d,t}}r_i,
\quad w_i^{macro}=\frac{1}{6}\frac{r_i}{Z_{d,t}}.
\]
\(n^+_{d,t}\)、\(n^-_{d,t}\) 是 Train split 中该 cell 的正负行数，\(A_{d,t}\) 是 Train split 中实际存在的 condition。每个 cell 的可用 condition 权重和为 1/6，六格总权重为 1。S0 必须确认六个 cell 均有正负样本；若某 cell 缺失正例或负例，则该 dataset 的 A/C pairwise 方向阻塞，不能对剩余 cells 重新归一化冒充 Macro AUC。

新增 `macro_pair_sampler.py::sample_macro6`：
1. 从全局固定权重计算 proposal \(q_i^{macro}=0.5/|A|+0.5w_i^{macro}\)，确保 \(q\) 和为 1 且 \(w_i^{macro}/q_i^{macro}\le 2\)。
2. 使用独立的 `macro6_condition` RNG stream 按 q 抽取一个 condition。
3. 从该 condition 的 Train-global positive/negative row pools 有放回抽取 2048 pairs，使用独立的 `macro6_pair_indices` RNG stream。
4. 通过稳定 row IDs 从 canonical Train tensor/cache 取出 4096 endpoint rows；顺序固定为 positive endpoints、negative endpoints。记录 condition、pair row IDs、pair hash 和曝光次数。

PAIR loss：
\[
L_{pair}^{macro}=\frac{w_i^{macro}}{q_i^{macro}}\frac{1}{2048}
\sum_{j=1}^{2048}\operatorname{softplus}(-(z_j^+-z_j^-)/\tau),\quad \tau=1.
\]
该重要性采样估计六格 Macro-AUC 对应的平均 pairwise risk；BCE 主目标仍为 Sample/natural，不是 Equal BCE。A/C 所有对照臂共享完全相同的 macro6 condition 和 pair IDs。

B 不调用 `sample_macro6`，必须原样调用 `boundary_sampler.py::sample`，保持 A21 的 natural 24-condition proposal、原始 w/q、2048 pairs、pair event clock 和 RNG。B 的梯度校正只改变 PAIR 梯度状态，不改变 A21 的采样目标。

### 1.2 Gaussian view

沿用 GA31-R3 D0 embedding perturbation 的字段范围、scale、自然 D0 覆盖和 Gaussian key 身份。对样本 i 的可扰动字段 f：
\[
e^{view}_{i,f}=e_{i,f}+\kappa\,\operatorname{stopgrad}(\sqrt{\operatorname{mean}(e_{i,f}^2)})\,\epsilon_{i,f}.
\]
\(\kappa\) 采用既有 GA31-R3 配置：MovieLens 0.05、KuaiRand 0.1；不重新扫描噪声强度。domain embedding 不扰动。必须复用原始 GA31-R3 RNG identity：device-local `torch.Generator`、每个扰动字段按目标 batch 顺序调用一次 `torch.randn`，noise key 为 `GA31-R3|{dataset}|{seed}|train|{global_step}|gaussian|1`，global_step 从 1 起。这里的随机数是按 batch 顺序生成的 per-row/per-field 矩阵，不是 row-keyed RNG；不得改用 EMA02V2 SplitMix64/Box-Muller row-keyed noise，因为 main 的 Gaussian audit 已明确其与 GA31-R3 archived draw 不同且 `row_key_ema02_replacement_allowed=false`。需要验证同一 seed、batch row order、global_step 和 field draw order 可复现，并核对原始首批噪声 hash；不要求 batch reorder invariance。pair row IDs 仍用于 pair hash，不参与重定义 GA31-R3 noise。

A1 的 pairwise forward 始终使用 macro6 sampler 返回的 4096 Train endpoint rows。P0 对这些 rows 使用 clean view；GP0 在 selected condition 属于 D0 时使用独立 endpoint Gaussian view（key stream 为 `gaussian_pairwise`），非 D0 condition 使用 clean view。该 pairwise view 与 full-batch GA31-R3 BCE view 是两个独立 view stream；base BCE 的 GA31-R3 key 不变。A1 四臂均执行相同 pair batch forward，B0/G0 不对 pairwise logits 回传梯度，以匹配额外计算量。若 A1 通过，A2 才比较 pair-level independent/shared noise：仅当 sampler 选中 D0 condition 时，independent arm 对每个 endpoint 独立采样，shared arm 对每个 pair 的正负 endpoint 使用同一字段噪声向量；两臂使用相同 pair IDs、pair-expanded rows 和 view forward 数。若 selected condition 非 D0，两臂均使用 clean pair logits。共享噪声只是一种匹配控制，不能默认更优。

### 1.3 四臂最小 2×2 矩阵

定义：
- B0：clean BCE，EMA。
- P0：clean BCE + clean pairwise，EMA。
- G0：原 GA31-R3 Gaussian BCE augmentation，EMA。
- GP0：原 GA31-R3 Gaussian BCE augmentation + independent Gaussian pairwise view，EMA。pairwise view 对 D0 condition 的 endpoint 使用独立 Gaussian noise；非 D0 condition 使用 clean view。这样 Gaussian 只影响原注册的 D0 范围，不把噪声扩展到其他 cell。

总目标统一写为：
\[
L=L_{BCE}^{base}+\lambda_p L_{pair}^{(view)}.
\]
G0 的 base BCE 完全复现原 GA31-R3；GP0 仅增加 pairwise 项。P0 的 base BCE 与 B0 完全一致。BCE 增强项不能因为新增 pairwise loss 而被重新归一化或改变剂量。

A 的 pairwise coefficient 固定为 \(\lambda_A=0.05\)，不做额外 HPO；A1 四个 arm 的 pairwise arms 共享 macro6 sampler、pair IDs 和 pair count。记录每步 macro6 `w/q`、pair count 和 pair/base gradient norm。B 的 \(\lambda_{PAIR}=0.05\) 仍严格沿用 A21 原始配置，二者是不同配置字段，不得混用。

### 1.4 前向与反向顺序

1. 使用原 clean input 做一次 full-batch forward，获得 clean logits 和 Sample BCE。
2. 调用 macro6 Train-global sampler，获得一个 condition 和 2048 pairs；通过 row IDs gather 4096 endpoint rows，并对四臂执行同样的 pair forward。B0/G0 在 pair forward 上使用 no-grad；P0 使用 clean pair logits；GP0 若 condition 属于 D0 则使用独立 `gaussian_pairwise` view，否则使用 clean pair logits。
3. 对 G0/GP0 生成原始 GA31-R3 full-batch D0 view，执行一次 D0 view forward，仅用于原 GA31-R3 base BCE augmentation。
4. G0 的 base loss 完全复现原 GA31-R3 mixed BCE；GP0 在此基础上加入 pairwise Gaussian loss；P0 在 clean BCE 上加入 clean pairwise loss；B0 只有 clean BCE。
5. 按 `L = L_BCE + lambda_A * L_pair_macro` 组合 P0/GP0；B0/G0 不加入 pairwise loss。
6. 每个 arm 对其训练目标 backward 一次；optimizer 更新一次；随后更新固定 EMA 一次。
7. 每 epoch 对 online 与 EMA 独立做 Validation 六格 Macro AUC 选点。

四臂均执行相同的 4096 endpoint pair forward，B0/G0 不回传该 forward 的梯度；Gaussian arms 额外执行原始 full-batch D0 BCE view，GP0 的 pair forward 使用独立 Gaussian pair view。训练步数和 BCE row exposure 相同，完整记录 pair forward、Gaussian view、pair 数和 wall time。

### 1.5 关键解释和停止条件

- GP0−G0：Gaussian 背景上的 pairwise 增量。
- P0−B0：无 Gaussian 时 pairwise 监督本身的增量。
- GP0−P0：Gaussian 对 pairwise objective 的额外贡献。
- 若 P0 有收益而 GP0 不超过 P0：保留 clean pairwise，关闭 Gaussian Pairwise 组合。
- 若 GP0 不超过 G0+0.001：A 方向不晋级；不得继续堆叠排序损失、噪声类型或温度超参数。
- 若 GP0 通过改进门且六格保护，则 A2 才触发噪声耦合方式审计。

## 2. B：MARS-inspired PAIR Freshness Correction

### 2.1 研究对象

A21 的 PAIR 辅助梯度历史只在 S_DENSE 共享参数上发挥作用。B 仅修正 PAIR 来源梯度，不改自然 BCE 的原生 Adam 梯度，不改固定 EMA，也不修改私有参数优化规则。

设第 t 步采样的 PAIR 集合为 \(\mathcal P_t\)，当前私有参数为 \(\phi_t\)，当前共享参数为 \(\theta_{S,t}\)，上一步提交前共享参数快照为 \(\theta_{S,t-1}\)。

### 2.2 同 batch 梯度差

当前参数状态 PAIR 梯度：
\[
g_t=\nabla_{\theta_S}L_{pair}(\theta_{S,t},\phi_t;\mathcal P_t).
\]
旧共享参数状态、当前私有参数固定的 PAIR 梯度：
\[
g_t^{-}=\nabla_{\theta_S}L_{pair}(\theta_{S,t-1},\phi_t;\mathcal P_t).
\]
Freshness-corrected gradient：
\[
\tilde g_t=g_t+\gamma(g_t-g_t^{-}),\quad \gamma\in\{0.25,0.5\}.
\]
主候选预注册 \(\gamma=0.5\)；0.25 只在 0.5 通过 Validation 门、但出现过度校正迹象时触发。\(\gamma=0\) 是 A21 原始辅助梯度控制。第一步初始化 \(\theta_{S,t-1}=\theta_{S,t}\)，因此首步 correction 必须严格为零。不得直接称为 MARS；这是 MARS 同样本梯度差校正思想在 PAIR 来源和共享参数子空间上的适配。

保持 A21 的 collapsed-history 数值合同：令 \(a_t=(w_s/q_s)\tilde g_t\)，并更新
\[
Q_t=0.99Q_{t-1}+0.01a_t,\quad k_t=k_{t-1}+1.
\]
在 Adam mixed gradient 中扣除当前 PAIR 项 \(\lambda_{PAIR}a_t\)，再在 bias-corrected first-moment numerator 中加回 \(\lambda_{PAIR}Q_t/(1-0.99^{k_t})\)。这逐项沿用 `PairOptimizer.step(mode='collapsed')` 的时钟、重要性比和注入顺序。辅助注入保持 A21 的来源权重 \(\lambda_{PAIR}\)、重要性比 \(w/q\)、S_DENSE scope、时钟、Adam 原生 BCE 梯度和 EMA 设置完全不变。\(\lambda_{PAIR}\) 沿 A21 已冻结值，不再调节。

### 2.3 精确更新顺序

在每个训练 batch：
1. 依据冻结 sampler 得到 batch rows、PAIR cell 和 pair indices；同一 pair set 必须用于 \(g_t\) 与 \(g_t^-\)。
2. 在当前 \((\theta_{S,t},\phi_t)\) 做 clean forward，先用 `autograd.grad(auxiliary, S_DENSE, retain_graph=True)` 计算 PAIR 梯度 \(g_t\)，再对 BCE-only loss backward，保存自然 BCE 的完整 native gradients。
3. 暂存当前 \(\theta_{S,t}\) 的梯度；将 S_DENSE 临时替换为上一步快照 \(\theta_{S,t-1}\)，保持 \(\phi_t\)、buffers、dropout/RNG 和 pair indices 固定，重新计算 PAIR loss 和 \(g_t^-\)。该路径不产生 BCE 梯度，不更新任何状态。
4. 恢复当前 \(\theta_{S,t}\)；检查 model、optimizer、EMA、buffers、RNG、loader 和 sampler state 哈希守恒。
5. 计算 \(\tilde g_t\)，先完成 native BCE backward，再对每个 S_DENSE 参数显式加上 \(\lambda_{PAIR}(w_s/q_s)\tilde g_t\)，使传入的 optimizer gradient 与 A21 原实现的“BCE + 当前 PAIR 项”形式一致；随后调用 collapsed optimizer 的 `step(pair, r=tilde_g_t)`。optimizer 内部会扣除同一个当前 PAIR 项，并把 corrected gradient 写入 Q 历史，再将 bias-corrected Q 注入 numerator。不得只 backward BCE 却直接调用现有 `step(pair,r)`，否则 optimizer 会扣除一个未加进来的 PAIR 项。
6. 执行原生 optimizer 一次，更新 online 参数一次；更新 EMA 一次；将当前 S_DENSE 参数快照更新为新参数，供下一 batch 使用。
7. 记录 \(\|g_t\|\)、\(\|g_t^-\|\)、\(\|g_t-g_t^-\|\)、cos(g_t,g_t^-)、cos(\tilde g_t,g_t)、校正比例和额外 forward/backward 成本。

若无 PAIR，按 A21 原始无事件规则处理：不增加 pair_clock、不更新 Q 的事件内容；不得以空梯度产生虚假的校正。若 pair loss 梯度为零但 pair event 有效，仍按原始时钟合同衰减 Q 并记录 zero event。None gradient 按参数支持表处理，不填零后误更新 Adam state。任何非有限值立即暂停该 job，不能静默跳步。

### 2.4 参数快照与坐标范围

- 只保存 S_DENSE 参数的上一步快照，不保存完整模型快照。
- 计算 \(g_t^-\) 时使用当前私有参数 \(\phi_t\)，因此估计的是“共享参数坐标变化”对 PAIR 梯度的局部影响，不声称是完整参数轨迹的全状态 transport。
- 必须验证所有 S_DENSE 参数在当前和旧共享状态下可参与同一 pair loss 计算；参数顺序和 shape 来自现有 optimizer manifest，禁止手写名字猜测。
- 旧共享参数临时替换必须使用 try/finally 保证恢复；并对 buffers、dropout masks、随机数、optimizer、EMA 做守恒检查。建议用 functional_call/显式参数映射实现，避免写入 live Parameter。
- 额外成本为每 batch 一次 PAIR forward/backward；不得在报告中只报 optimizer step 数而忽略该成本。

### 2.5 B 对照

- B0：普通 clean BCE + PAIR Adam + EMA。
- B1：原 A21 + EMA。
- B2：A21 + same-batch freshness correction，γ=0.5 + EMA。
- B3：B2 的校正项仅作为诊断 shadow，不注入更新，用于核对方向变化是否与性能收益一致；B3 不计完整候选训练，使用同轨迹 observer 即可。
- 触发控制：若 B2 相对 A21 Validation ΔAUC≥+0.001 且六格保护通过，才跑 γ=0.25 的有限确认；否则 B 关闭。
- 如果 B2 只改善 LogLoss、不改善 AUC，不晋级为排序优化机制。

## 3. C：历史正确排序关系记忆

### 3.1 记忆对象

记忆单元不是参数、梯度或整条样本，而是合法训练行构成的有序 pair：
\[
m_j=(i_j^+,i_j^-,c_j,\Delta_j^{hist},e_j,\text{snapshot-id}).
\]
其中两行来自同一训练 cell，\(\Delta_j^{hist}=z_{i^+}^{hist}-z_{i^-}^{hist}\) 是历史 snapshot 的 logit margin。只保留在该历史 checkpoint 上正确排序的 pair（\(\Delta_j^{hist}>0\)），并保留 margin、row ids、epoch 和 snapshot hash。训练标签用于确认正负身份；不读取 Validation/Test 标签来建立记忆。

### 3.2 记忆损失

对当前参数 \(\theta_t\)：
\[
\Delta_j(\theta_t)=z_{i_j^+}(\theta_t)-z_{i_j^-}(\theta_t).
\]
历史排序软目标：
\[
q_j=\sigma(\Delta_j^{hist}/T),\quad p_j(\theta_t)=\sigma(\Delta_j(\theta_t)/T).
\]
记忆损失沿用 macro6 sampler 的 selected condition 和 (w_i^{macro}/q_i^{macro})，对 2048 memory pairs 求均值：
\[
L_{mem}=\frac{w_i^{macro}}{q_i^{macro}}\cdot\frac{1}{2048}\sum_{j=1}^{2048}
\operatorname{BCE}(p_j(\theta_t),q_j).
\]
C4 使用相同 pair identities，将 memory target 换成 hard pairwise label，即 \(L_{hard}=\frac{w_i^{macro}}{q_i^{macro}}\frac{1}{2048}\sum_j\operatorname{softplus}(-\Delta_j(\theta_t))\)。
默认 T=1。若历史 margin 非常大，按预注册 clip [-5,5] 保存 teacher margin，避免极端 logits 支配训练。该损失保留历史 pairwise confidence，而非仅重复当前 pair label。所有记忆 pair 仍然是 Train 行。

### 3.3 构建和更新

- 在训练早期固定 epoch 5 和 epoch 10 作为候选历史快照；只使用训练标签和 Train 行。快照对象统一为 fixed EMA 参数；C0 比较 EMA-E5、EMA-E10 到各自 selected EMA checkpoint 的关系变化。
- 对每个 active condition 构建 pair candidate pool，按 pair sampling seed 固定最多 4096 个候选；使用 EMA-E10 预测，保存所有 pair margin，并从中筛选历史正确 pair。
- 从每个 active condition 的合法 Train-global pair pool 中抽取最多 2048 个 unique memory pair（最多 24×2048），不足时记录实际容量。训练时调用 macro6 sampler 获得 selected condition、(w_i^{macro}/q_i^{macro})，再从该 condition 的 memory pool 有放回抽取 2048 pair 替换 current pair identities；不改变 macro6 condition probability 或 pair dose。C1/C2/C3/C4 均执行相同 4096 endpoint forward；C0 也执行该 forward 但不回传 pair loss，以匹配计算量。epoch 5 只用于估计历史关系变化；主训练 memory bank 仅由 epoch 10 snapshot 构建，并从 epoch 11 开始使用。
- C1/C2/C3/C4 在 epoch 1–10 使用完全相同的 current pairwise objective 和 pair exposure；epoch 10 结束后从 EMA-E10 构建并冻结 memory，epoch 11–50 中 C2/C3/C4 使用各自 memory bank 替换 current pair identities。C0 始终无 pairwise。memory 不随 Validation/Test 更新，不按当前预测重新筛选。
- 若当前 batch 的 macro6 sampler 返回 None，则该步没有 pair event；若 memory pool 对选中 condition 为空，记录 memory miss，并在 C2/C3/C4 对该步回退到相同的 current macro6 pair indices，使各臂 pair dose 对齐。
- 为控制曝光，memory pair 的每个 epoch 曝光次数与当前 pairwise control 相同；记忆组与随机记忆组使用相同 pair 数和更新次数。

### 3.4 C 最小对照

- C0：BCE + EMA，无 pairwise/memory。
- C1：BCE + current pairwise，EMA。
- C2：BCE + random historical pair memory（同 cell、同 pair count，保留所有 pair 的历史 margin，不按历史正确性筛选），EMA。
- C3：BCE + historical correct pair memory，EMA。
- C4：BCE + historical correct pair identities with hard pairwise labels（同样正确 pair 记忆，但不蒸馏历史 margin），仅在 C3 相对 C2 通过改进门后触发。
- 所有组 pair 曝光、loss 权重、采样时钟和模型训练步数相同。C3−C2 衡量筛选历史正确关系的增量；C3−C1 衡量历史记忆是否超出普通当前 pairwise supervision；C3−C4 衡量历史 margin 蒸馏是否提供额外信息。

### 3.5 C0 离线审计

在任何训练前，先确认 main 可访问的合法 selected checkpoints、Train row IDs 和预测缓存。优先复用现有数据，不重跑模型。如果权重只存在旧 RAM guardian 且当前不可读取，不能假定其存在；不得从磁盘临时保存的未知快照推断。

离线指标：
- selected epoch 的 Train pair 正确率、margin 分布和 cell 统计；
- 从 epoch 5/10 到 selected epoch 的历史正确 pair flip rate；
- flip pair 的 selected margin、未 flip pair 的 selected margin；
- 对应 pair 在 selected checkpoint 上的分布与各 cell 的 AUC；
- 历史正确 pair 与当前 pairwise sampler 的重合率；
- 固定记忆容量下有效 pair 数和每 pair 平均曝光成本。

顺序：先分析每个 seed 实际 selected epoch；再分析 selected 前关系变化；最后才分析 selected 后变化。不得用 E50 代替 selected checkpoint 结论。若 selected checkpoint 之前没有稳定历史正确 pair 被遗忘，或记忆关系与 current pair sampler 高度重复且无历史 margin 增量，则 C 关闭，不训练。

## 4. 替代机制和选择

- B 的更低成本替代：不运输历史梯度，只使用当前 PAIR 梯度 + A21 原 EMA。若 freshness correction 的额外计算成本不值得，回退 A21。
- B 的高潜力替代：输出残差历史动量（BOM 思路），将 pairwise residual/margin residual 保存于输出空间，再经当前网络计算梯度。它需另立研究方案，不得混入 B2。
- C 的更简单替代：current pairwise loss。若 C3 不超过 C1，则采用简单 pairwise objective，不保留记忆。
- A 的更简单替代：clean pairwise ranking。若 P0 已经达到 GP0 的效果，关闭 Gaussian pairwise 叠加。

