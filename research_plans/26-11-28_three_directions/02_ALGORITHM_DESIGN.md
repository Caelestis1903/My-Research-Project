# 02 — 候选算法数学定义与更新流程

## 0. 共同符号和训练目标

- 域任务格集合：\(\mathcal C=\{(d,t):d\in\{0,1,2\},t\in\{0,1\}\}\)。
- 一个样本行 \(i\) 的 M3oE 预测概率：\(p_{it}=f_{\theta}(x_i)_t\)，logit：\(z_{it}\)。
- \(\theta=(\theta_S,\phi)\)：\(\theta_S\) 是现有 optimizer manifest 中 S_DENSE 的共享参数，\(\phi\) 是其余参数。
- 六格 BCE 自然目标按项目当前 fixed1 训练实现执行。所有新增 pairwise 项按六格等权平均，且单独记录原始 loss 和梯度范数。
- 固定 EMA 参数：\(\bar\theta_t=\rho\bar\theta_{t-1}+(1-\rho)\theta_t\)，使用仓库已冻结 dataset-specific rho；EMA 只读、不反馈 online optimizer。

## 1. A：Gaussian Pairwise Ranking Augmentation

### 1.1 训练 pair 定义

在每个 batch 内按 cell \(c=(d,t)\) 分层。对 cell 中正例集合 \(P_c\) 与负例集合 \(N_c\)，从合法 Train 行抽取 pair：
\[
\mathcal P_{c,t}=\{(i^+,i^-):y_{i^+,t}=1,y_{i^-,t}=0, d(i^+)=d(i^-)=d\}.
\]
不跨域、不跨任务格配对。严格复用 main 的 `experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py::sample`：每个 batch 按原 condition distribution 选择一个 cell，并有放回抽取 2048 个正负 pair；保留原 `active/w/q/index/selected`、pair event clock 和 context RNG。所有对照臂共享完全相同的 pair indices。不得改成 256 pair/cell 或自行改成全格均匀采样，因为这会改变 A21 已有的采样目标。pair sampling 只访问 Train 标签；记录 row-id pair hash。

每格 pair loss：
\[
L_{pair,c}=\frac{1}{|\mathcal P_c|}\sum_{(i^+,i^-)\in\mathcal P_c}
\operatorname{softplus}(-(z_{i^+}-z_{i^-})/\tau),
\quad \tau=1.
\]
总体 pair loss 由当前 sampler 的单一 selected cell 估计，并乘以原始 `w/q` 重要性比：
\[
L_{pair}=\frac{w_s}{q_s}\cdot\frac{1}{2048}\sum_{j=1}^{2048}\operatorname{softplus}(-(z_j^+-z_j^-)/\tau),\quad \tau=1.
\]
其中 s 是本步 sampler 选中的 condition。严格使用原 `w/q` 重要性比，不能再叠加自定义 cell 权重。若 sampler 返回 None，则 pair loss 为零，不产生 NaN 或伪造历史事件。

### 1.2 Gaussian view

沿用 GA31-R3 D0 embedding perturbation 的字段范围、scale、自然 D0 覆盖和 Gaussian key 身份。对样本 i 的可扰动字段 f：
\[
e^{view}_{i,f}=e_{i,f}+\kappa\,\operatorname{stopgrad}(\sqrt{\operatorname{mean}(e_{i,f}^2)})\,\epsilon_{i,f}.
\]
\(\kappa\) 采用既有 GA31-R3 配置：MovieLens 0.05、KuaiRand 0.1；不重新扫描噪声强度。domain embedding 不扰动。所有 noise RNG 使用独立且可复现的 dataset/seed/global_step/row-id/field key；不得使用 batch 内位置作为唯一行身份。若源实现只支持 batch key，先复现其原始调用顺序并做 row-order invariance 审计，未通过前不得宣称逐行噪声复现。

pairwise Gaussian logits 使用正负样本各自的 view logits。最小矩阵先使用 independent noise（正负行独立噪声）。若 A1 通过，A2 才比较 coupled noise：对同一 pair 的同一字段使用共享噪声向量 \(\epsilon_f^{pair}\) 加到两行；其余字段保持独立。共享噪声只是一种匹配控制，不能默认更优。

### 1.3 四臂最小 2×2 矩阵

定义：
- B0：clean BCE，EMA。
- P0：clean BCE + clean pairwise，EMA。
- G0：原 GA31-R3 Gaussian BCE augmentation，EMA。
- GP0：原 GA31-R3 Gaussian BCE augmentation + Gaussian-view pairwise，EMA。

总目标统一写为：
\[
L=L_{BCE}^{base}+\lambda_p L_{pair}^{(view)}.
\]
G0 的 base BCE 完全复现原 GA31-R3；GP0 仅增加 pairwise 项。P0 的 base BCE 与 B0 完全一致。BCE 增强项不能因为新增 pairwise loss 而被重新归一化或改变剂量。

\(\lambda_{PAIR}\) 固定复用现有 PAIR 训练入口的 `lambda_pair=0.05`（以及现有 sampler 返回的 `w/q`）。不得另行做梯度范数标定，因为这会改变 A21 的监督剂量并引入新的超参数。四个 arm 对 pair sampler 和 pair exposure 完全匹配；记录每步 pair loss、`w/q`、pair count 和 pair/base gradient norm。

### 1.4 前向与反向顺序

1. 使用原 clean input 做一次 forward，获得 clean logits 和原始 BCE。
2. 构造每格 pair index，clean pairwise 复用 clean logits。
3. 对 G0/GP0 生成 GA31-R3 view input，执行一次 view forward。
4. G0 按原始 GA31-R3 mixed BCE objective 计算 base loss；GP0 的 pairwise 只从 Gaussian view logits 计算。
5. 对 P0 使用 clean logits 计算 pairwise loss。
6. 总 loss backward 一次；optimizer 正常更新一次；随后更新固定 EMA 一次。
7. 每 epoch 对 online 与 EMA 独立做 Validation 六格 Macro AUC 选点。

额外的 view forward 只发生于 Gaussian arm；pairwise loss 不重复采样另一个 pair set。训练预算按同一 epoch/row exposure 匹配，并完整记录额外 forward、pair 数和 wall time。

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
主候选预注册 \(\gamma=0.5\)；0.25 只在 0.5 通过 Validation 门、但出现过度校正迹象时触发。\(\gamma=0\) 是 A21 原始辅助梯度控制。不得直接称为 MARS；这是 MARS 同样本梯度差校正思想在 PAIR 来源和共享参数子空间上的适配。

保持 A21 的 collapsed-history 数值合同：令 \(a_t=(w_s/q_s)\tilde g_t\)，并更新
\[
Q_t=0.99Q_{t-1}+0.01a_t,\quad k_t=k_{t-1}+1.
\]
在 Adam mixed gradient 中扣除当前 PAIR 项 \(\lambda_{PAIR}a_t\)，再在 bias-corrected first-moment numerator 中加回 \(\lambda_{PAIR}Q_t/(1-0.99^{k_t})\)。这逐项沿用 `PairOptimizer.step(mode='collapsed')` 的时钟、重要性比和注入顺序。辅助注入保持 A21 的来源权重 \(\lambda_{PAIR}\)、重要性比 \(w/q\)、S_DENSE scope、时钟、Adam 原生 BCE 梯度和 EMA 设置完全不变。\(\lambda_{PAIR}\) 沿 A21 已冻结值，不再调节。

### 2.3 精确更新顺序

在每个训练 batch：
1. 依据冻结 sampler 得到 batch rows、PAIR cell 和 pair indices；同一 pair set 必须用于 \(g_t\) 与 \(g_t^-\)。
2. 在当前 \((\theta_{S,t},\phi_t)\) 做 clean forward/backward，保存自然 BCE 的完整 native gradients，并计算 PAIR 梯度 \(g_t\)。
3. 暂存当前 \(\theta_{S,t}\) 的梯度；将 S_DENSE 临时替换为上一步快照 \(\theta_{S,t-1}\)，保持 \(\phi_t\)、buffers、dropout/RNG 和 pair indices 固定，重新计算 PAIR loss 和 \(g_t^-\)。该路径不产生 BCE 梯度，不更新任何状态。
4. 恢复当前 \(\theta_{S,t}\)；检查 model、optimizer、EMA、buffers、RNG、loader 和 sampler state 哈希守恒。
5. 计算 \(\tilde g_t\)，先完成 native BCE backward，再对每个 S_DENSE 参数显式加上 \(\lambda_{PAIR}(w_s/q_s)\tilde g_t\)，使传入的 optimizer gradient 与 A21 原实现的“BCE + 当前 PAIR 项”形式一致；随后调用 collapsed optimizer 的 `step(pair, r=tilde_g_t)`。optimizer 内部会扣除同一个当前 PAIR 项，并把 corrected gradient 写入 Q 历史，再将 bias-corrected Q 注入 numerator。不得只 backward BCE 却直接调用现有 `step(pair,r)`，否则 optimizer 会扣除一个未加进来的 PAIR 项。
6. 执行原生 optimizer 一次，更新 online 参数一次；更新 EMA 一次；将当前 S_DENSE 参数快照更新为新参数，供下一 batch 使用。
7. 记录 \(\|g_t\|\)、\(\|g_t^-\|\)、\(\|g_t-g_t^-\|\)、cos(g_t,g_t^-)、cos(\tilde g_t,g_t)、校正比例和额外 forward/backward 成本。

若无 PAIR，按 A21 原始无事件规则处理：不更新 q 的事件内容；不得以空梯度产生虚假的校正。若 pair loss 梯度为零，仍按原始时钟合同更新历史衰减；记录 zero event。None gradient 按参数支持表处理，不填零后误更新 Adam state。任何非有限值立即暂停该 job，不能静默跳步。

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
记忆损失：
\[
L_{mem}=\frac{1}{|M_t|}\sum_{j\in M_t}
\operatorname{BCE}(p_j(\theta_t),q_j).
\]
默认 T=1。若历史 margin 非常大，按预注册 clip [-5,5] 保存 teacher margin，避免极端 logits 支配训练。该损失保留历史 pairwise confidence，而非仅重复当前 pair label。所有记忆 pair 仍然是 Train 行。

### 3.3 构建和更新

- 在训练早期固定 epoch 5 和 epoch 10 作为候选历史快照；只使用训练标签和 Train 行。
- 对每个 cell 构建 pair candidate pool，按 pair sampling seed 固定最多 4096 个候选；使用对应 snapshot 预测，保留历史正确 pair。
- 从每个 cell 的合法 Train 候选中抽取最多 2048 个 unique memory pair（总计最多 12288），不足时记录实际容量。训练时先调用现有 condition sampler 获得 selected cell、w/q 和 pair clock，再从该 cell 的 memory pool 有放回抽取 2048 pair 替换 pair identities；不改变 condition probability 或 pair dose。
- 记忆在 epoch 10 结束后冻结，不随 Validation/Test 更新，不按当前预测重新筛选；训练时每个 epoch 以固定曝光预算取样。
- 若任何 cell 无合法 pair，跳过该 cell并按 active cells 等权归一。
- 为控制曝光，memory pair 的每个 epoch 曝光次数与当前 pairwise control 相同；记忆组与随机记忆组使用相同 pair 数和更新次数。

### 3.4 C 最小对照

- C0：BCE + EMA，无 pairwise/memory。
- C1：BCE + current pairwise，EMA。
- C2：BCE + random pair memory（同 cell、同 pair count、无历史 margin），EMA。
- C3：BCE + historical correct pair memory，EMA。
- C4：BCE + recent pair replay（按同样历史 snapshot、同样容量，但不筛正确排序关系），仅在 C3 相对 C2 通过改进门后触发。
- 所有组 pair 曝光、loss 权重、采样时钟和模型训练步数相同。C3−C2 衡量历史正确关系/历史 margin 的额外价值；C3−C1 衡量历史记忆是否超出普通当前 pairwise supervision。

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

