# My-Research-Project：证据驱动的完整后续科研实验规划

**编制日期**：2026-10-15  
**仓库**：[Caelestis1903/My-Research-Project](https://github.com/Caelestis1903/My-Research-Project)  
**审计固定提交**：`3e6642820b7b2f17f130a5ccd9d382a16e023811`（`main`，initial commit）  
**工作状态**：实验**方案**，不是运行授权、运行记录、已实现方法或论文创新性认定。  
**总体原则**：不覆盖原始产物；不把历史报告视为原始指标；不先开发算法再倒找证据；严格区分可用目标、实际估计器、Adam 状态及 checkpoint 选择。

## 0. 结论先行：最合理的研究战略

1. 以 **S5 Equal/history-dependent effect** 和 **S6-R2 Sample validation/test reversal** 作为两个*相互独立*的研究问题，不把 Sample 和 Equal 的结果直接相加。
2. 以 **R2 AO-04 B1 负结果及 KuaiRand 未完成矩阵** 为可检验的融合自由度边界，不将低收益直接写成不可辨识定理。
3. 在 S1/S2 与 S0/E3–E6 既有反证下，暂不恢复“小域天然难”“域动量特异有效”“课程/EMA 靠叠加必然收益”等已被削弱路线。
4. 在已检查到的 F0 v1.5 审核范围内，B6 historical `OPEN/FAIL`、forward `NOT_GRANTED`、execution `NO_GO`；故该路径上的训练、checkpoint load、模型运行不得据此计划自动开始。证据审计、理论推导和**批准范围内**只读采集可分别办理。
5. **第一项可能获批的新训练**，应优先是 S6-R2 预先许可的 `F/DH × 4 seeds = 8 runs` checkpoint-dense 科学取证，而不是新方法搜索。其可执行性仍受独立 F0、科学协议与工件门控约束。

## 1. 已审材料、证据等级和不能忽略的新事实

### 1.1 五份附件审查

- `research_checkpoint.json`：固定了提交、22 个首次检查文件、11 个待定位工件和阶段状态；该 22 不是全仓文件数。
- `reading_coverage.md`：精确区分“已读返回全文”“部分读取”“没有查验”；应继承这种证据语言。
- `phase_0_1_research_report.md`：详细的前置事实重建；S4、S8 指标只读自报告，而 R2 差值有指标层面的独立算术复核。
- `R2_numeric_audit.md`：固定两个数据集的少量配对差值，不能从其反推预测级方差。
- `My_Research_Project_phase_0_1_audit.zip`：包含 README/报告/覆盖表/检查点、指标转录、重算脚本及结果等 9 个内部文件；在当前审计环境，`SHA256SUMS.txt` 9/9 通过，`audit_r2.py` 可重算指标。注意：这些只是 ZIP 内部一致性，不是原始数据与预测工件全链条的 SHA 证明。

### 1.2 新访问的仓库索引

截至 2026-10-09 的 `experiments/Experimental_Global_Summary.md` 整理了 **32 个实验或门控条目**、**12 个 stable findings**、**10 个 open questions**、**12 条 dead ends**。与附件的首轮 22 文件覆盖互补，不能混合称作已独立核验全部原始记录。索引主入口：

- `experiments/Experimental_Experiment_Index.md`
- `experiments/Experimental_Evidence_Ledger.md`
- `experiments/Experimental_Stable_Findings.md`
- `experiments/Experimental_Open_Questions.md`
- `experiments/Experimental_Contradiction_Register.md`
- `experiments/Experimental_Dead_Ends.md`
- `experiments/Experimental_Hypothesis_History.md`

### 1.3 先明确证据等级

| 级别 | 定义 | 是否能成为论文结果 |
|---|---|---|
| L0：设计 | 规格书、提案、预注册计划 | 否 |
| L1：报告 | 文字报告/图表而原始 bytes 未复核 | 仅可作为已知线索 |
| L2：指标算术 | 独立从已存储逐 run CSV 复算，校验选择规则 | 限定的可复制数字 |
| L3：预测复算 | 原始 labels、predictions、row IDs、split hash 对齐后重算 AUC/LogLoss | 合格的评估证据 |
| L4：执行复现 | 数据+源码+配对 RNG/批次、模型+optim state、独立审查、独立重复 | 受协议限定的强实验证据 |
| L5：跨设定验证 | 按预注册选择规则跨数据、结构、种子且多重检验控制 | 可评估方法性普遍结论 |

**禁止**：用 L1 数值说“我们重新证明 S4”；用 L2 数值说“R2 B1 发生因果梯度冲突”；用 L3 的 test 向后调参并把 test 当独立验证。

## 2. 重要历史研究状态：哪些问题已回答，哪些还没有

| 线程 | 当前有效观察或裁决 | 下一步姿态 |
|---|---|---|
| S0 | domain-conditioned EMA/comp1 未表现超过 validation-selectable uniform EMA family 的独特机制优势 | 不以“小域低 SNR”恢复 comp1 |
| S1 | Sample + 匹配 SD reference，所测 72 单元 0 个稳定 practical NT（源报告） | Sample 强基线，先复核 reference fidelity |
| S2 | Equal 特定大域目标 selected loss；Sample 相应配置 0/24 practical NT（源报告） | 同一目标不同估计器/选点严格分离 |
| S3/S4 | KuaiRand/Equal/d1 target-only +0.0389/+0.0396 AUC；norm controls N/U 不可复制；d0 代价 | 已证明局部干预，未独立证明负 cosine 冲突 |
| S5 | F/T/HT/DH 交叉：HT 恢复 99.2%/95.9% target rescue；DH ~2%；moment gate 15/16 | 当前/历史总体因子已做；下一步只研究组件、参数分区及边界 |
| S5-ML | Equal MovieLens 对 history pattern 仅部分复制 | 不把域/任务/overfit 混为大小因果 |
| S6/S8 | Sample selected practical target rescue 未过门；S8 selected continuation 无优于原最优 checkpoint 的实用收益 | 不按 selected method claim 重启 optimizer history |
| S6-R2 | Sample KuaiRand F/DH R0 同一 selected checkpoint：test ΔN +0.0068、validation ΔN −0.0039；coarse A-grid 其他选择规则 only | 仅一项 checkpoint-dense 取证，未授权方法创新 |
| research_R2 | AO04 B1 MovieLens NO、KuaiRand B1 2023 INCOMPLETE；KR seed2022 val+0.001494/test−0.008385 | 原预测核验与 rho 路径诊断，不继续 B3/B4 搜索 |

**概念区分**：仓库 `experiments/stage_S6_R2_*` 是 Sample/F-DH 的既有 checkpoint 再评估；`research_R2/*` 是 M3OE 的 AO04 非对称融合 B0–B4。两者不能用“R2”简称放到同一矩阵。

## 3. 已验证的数据口径与新的分布事实

### 3.1 MovieLens

- 域年龄组 `{1,18}`, `{25}`, `{35,45,50,56}`；任务 `click=1[rating>3]`、`like=1[rating>4]` 为**嵌套阈值标签**，并非两个独立行为。
- L/M/S 角色训练样本数量 150000/34500/3000，按 `RandomState(10*seed+physical_domain)` 抽子集；网络 seed 与抽样 seed 同时改变，需在解释方差时保留身份。
- `setting=d2_large` (S3 等) 与 `d0_large` (research_R2) 不可当作一个共享的物理域角色设置。
- S1 DataBundle.reference_view 对联合样本 row-id remap 做部分指纹断言；仍需核实 single-domain 模型架构/初始化、预算和 evaluator。

### 3.2 KuaiRand-Pure

- 物理域 local d0=tab0, d1=tab1, d2=tab4；任务 `is_click` 和 `long_view`。
- 旧 E4 数据审计记载先按 `(date,time_ms)` 排序，再按全局行顺序切分 80/10/10，因此该处理版本的切分为**时间顺序切分**。本次没有重新拿原始 CSV bytes 独立复核排序，这条归为“报告+预处理路径证据”。
- **训练域数** (172782,1743898,89452)，**验证域数** (2692,246232,1843)，**测试域数** (2613,246284,1870)。对应 train d1 86.93%、val/test d1 98.19%/98.21%；train–val mixture TV 0.112632，val–test TV 0.000315。
- d1 click positive train 0.377832 → val 0.209737 → test 0.210517；long_view train 0.256083 → val 0.111521 → test 0.111315。Train→val/test label prior 变化明显，但 val→test marginal 几乎一致；**不能以粗域比例单独解释 val/test AUC 反号**。
- test d0 click 正样本 228/2613；long_view 正样本 120/2613；宏观六格指标的置信区间需要 paired row-level bootstrap，不能只报 seed variance。
- 任一单域 reference：必须先对全局有序数据完成 80/10/10，再在各 split 内按域过滤。绝不可先按域过滤后各自 80/10/10。

## 4. 共享、目标与优化器技术合同

### 4.1 M3OE 可达性

固定 forward 为 STAR/shared experts/domain experts/task experts、global four architecture scalars 与 cell gate/tower。模型 cell 使用 `i=t*D+d` 索引；gate `emb.detach()` 只阻断 gate-input 路径到 emb，而不冻结 emb 或 gate 参数。其他域 expert 在 cross-domain 混合中仍被当前域预测读取；模块命名为 domain_expert **不等于该参数只收到该域梯度**。

`stage_S3_gradient_access.py` 的分区：

- C：embedding、shared STAR/skip/star MLP、shared/domain/task experts；
- P：slot weights/bias、gate、tower；
- A_arch：四个全局 fusion raw scalars，单独 arch optimizer。

未来每个对照必须按 `autograd.grad(L_dt, partition)` 建立非零 reachability matrix，并独立审计 P 是否真的仅从预想 cell 更新；对专家参数必须允许跨域交叉可达。不能把多任务“grad conflict”直接等价为某个参数名字冲突。

### 4.2 四种不得混用的 loss

对于域 d、任务 t、batch 域内计数 B_d，

1. 原始旧 Equal：`L_old=Σ_{d,t} mean_{i:d_i=d} BCE_{i,t}`（未除 D·T；空域边界需要保护）。
2. S0 历史 Sample：`L_S0=Σ_{d,t} D p_d \bar ℓ_{dt}`（每个 batch 使用训练比例历史尺度）。
3. S1 Sample：`L_sample=(1/(B T))Σ_{i,t} BCE_{i,t}`。
4. S2 Equal：`w_d=1/(D p_d)`，`L_equal=(1/(BT))Σ_{i,t} w_{d_i} BCE_{i,t}`；有限 batch 不保证域严格等权。

**所有新计划必须首先以固定参数、固定 batch 做 loss 和 autograd 的单元 oracle**：数值精度、缺域、规模、有效权重、相同 Adam state 下一步实际更新。不能用相同名字推定相同期望或相同轨迹。

### 4.3 相同期望目标与不同估计器

令目标域权重 `α_d` 满足 `Σ_d α_d=1`：Sample 时 `α=p`，Equal 时 `α_d=1/D`。从域分布 `q_d>0` 抽样时，以 `α_{d_i}/q_{d_i}` 的 importance weight 构造 fixed-denominator mean，可在独立抽样、支持与其他常规条件下使期望梯度对应同一目标；但样本重复、无放回抽样、batch normalization、gradient clipping、weight decay 和动态参数变化可破坏简单等价。保留同一样本分布下的 paired estimators oracle。

**优先测试的是** `same α, different q`，而不是把 `Equal vs Sample` 同时改了损失、重采样和校准后归因为“一种新 reweighting 算法”。

### 4.4 Adam 有效更新与状态一致性

对于每个参数 block 与 optimizer，日志至少包括：原始 per-domain/per-task gradient、进入 optimizer 前 weight-decay 处理、moment `m/v`、step counter、bias correction、真正 applied delta、已更新 θ、grad source/history source、初始化指纹、main/arch batch 指纹。PyTorch Adam 与 AdamW 的 weight-decay 位置不同，不能只用 `β1` 或时间常数声称相同优化器。对 S0 `comp1_domain` 及所有 uniform EMA 臂，仓库协议 `shared Adam beta1=0`；禁止把它写成额外叠加标准 Adam(beta1=.9) 的双动量。

原始 trainer 的 `if ~darts_para_flag(name)` 使 arch 参数进入双 Adam；S1 修为 `if not is_architecture_parameter(name)`。历史双 Adam 重叠不能未经追踪用于解释新版本；每个新 run 必须断言 C/P/A_arch optimizer param-id 分组互斥。

## 5. 必须先满足的执行/科学门控：A0（0 runs）

**A0.1 F0 合法执行域**：独立读取当前最新 control/hold/staged authorization。已见 2026-10-14 `B6 historical OPEN/FAIL; forward NOT_GRANTED; execution NO_GO`，只有指定 bounded post-incident evidence baseline ACCEPTED。任何代码编译/模型 load/GPU 训练/历史 checkpoint copy 不能从本计划自动推导许可；应由既有审批流程明确批准 exact action/scope 后操作；保存旧 hold/失败审计不可抹除。

**A0.2 不变量**：锁定仓库 SHA/tree、当前工作树变更、所有源码内容 SHA256、数据版本 hash、预处理排序/切分、6-cell 统计、optim state schema、设备/torch/cudnn、RNG、同 seed sampler、batch fingerprint、配置 sha、输出路径保护与备份。旧仓库只有 initial commit，不能依赖 git history 找到历史中间源码；使用归档 hash 比对。

**A0.3 工件恢复（只读清单）**：逐个定位 S4、S5、S6、S6-R2、S8 每 run `config.json`、`run_manifest.json`、`metrics_epoch/selected/test.csv`、`predictions_*.npz`、`parameter_manifest`、`selected/common/final checkpoint` 与完整 optimizer state。仓库 `stage_S4_A2_outputs/`、`stage_S5_outputs/`、`stage_S6_outputs/`、`stage_S8_outputs/` 对应已检查规范目录缺主表；部分 `stage_S5_raw_results`/`stage_S6_raw_results` 子目录在 GitHub 只有 `stdout.log`。不得把输出目录存在当作指标已存在；不得臆测本地磁盘也不存在。`research_R2/formal_runs_20261012/` 则有多 run 完整 CSV/NPZ/PT，优先用于端到端审计。

**A0.4 交付**：`artifact_inventory.json`（每工件 source, size, SHA256, immutable path, generation command），`missing_artifacts.json`，`code_path_audit.md`（runner callgraph，loss、C/P/A、arch step、epoch selection）和独立 review；复核通过才进入后续可执行阶段。

**STOP**：任意配对中 seed、数据 split、参数初始化、train-batch/hash、effective loss、optimizer state 或 checkpoint evaluator 不一致；无法恢复时明确 `INCOMPLETE`，不是“科学 NO”。

## 6. 低成本高回报的先行分析：A1/A2（0 runs）

### A1.1 R2 AO04 原预测/运行链追溯

按 `dataset × arm × seed × attempt` 收集九个完成 run 的 `run_manifest`/`selection.json`/`metrics_epoch`/`metrics_test`/`predictions_test`/`validation_predictions_selected`/`fusion_log`/`batch_trace`/`config`/`parameter_manifest`。第一步严格复算每个 cell AUC、LogLoss 和 aggregate 规则，再复算每个 seed 及配对Δ；对于 missing `.npz` 标识不可核实，不作替代。

**特别检查**：有 run config 记录 `metric_implementation.logloss_epsilon=1e-12`，与 charter text `1e-7` 不一致，但报告说明按实际版本冻结。跨阶段合并 loss 前，必须分别保留原口径和一致化诊断口径，不能混写为代码错误已经造成性能差异。

期望重建：

- MovieLens B1−B0 val macro AUC 约 `+0.000005765`、B1−B2 `−0.000006842`（2 paired seeds）；decision `NO` 严格只限该设计。
- KuaiRand B1−B0 seed2022 val `+0.001494010`，test `−0.008385198`，其中 d0 click `−0.022434`，d0 long_view `−0.030011`；B1 seed2023 `INCOMPLETE`，不是统计阴性。
- `B3/B4` 没有被有效执行，不能把它们计算为被否定的模型。

### A1.2 R2 可辨识性与函数敏感度（先不训练）

模型 `B1` 的局部导数：

`dF_dt/dq_dt = a_D * (1-b_D) * rho_dt*(1-rho_dt)*(E_j-E_k)`。

逐 seed × epoch × cell 量化：(i) `rho_dt` 与 q trajectory 及饱和标志；(ii) `a_D(1-b_D)`；(iii) `||E_j-E_k||` 与其输出方向；(iv) Jacobian 对 logits 的尺度；(v) Q 对 predictions/AUC surrogate 的 frozen-θ 改动；(vi) gate entropy / branch collapse；(vii) per-cell validation-test ranking agreement。若 q 几乎无预测敏感度，优先研究“当前融合形式为什么难以识别”，不是继续扩大 B1 自由度。

**反证**：即便 q 变化很大，若冻结预测敏感度低，不能声称学到有效跨域非对称性；即便预测敏感度高而 AUC 差，须转向 generalization/regularization，而不是不可辨识。

### A2 S6-R2 F/DH 指标反转重审

- 先恢复既有 16 个 F/DH dataset×arm×seed checkpoints 与原输出，不新增训练；将 S6-R2 自动生成的 A-grid test robustness 与 exact reselected test 明确隔开。
- 同一 checkpoint 的 `Δ_{DH-F}` 拆为 domain-task、macro、N/T、LogLoss、加权指标。比较 `val→test` 绝对值与同 epoch change；标明 valley/peak 与 selection_epoch。
- test d0 long_view 正样本仅 120，使用**按 cell 与标签分层的 paired row bootstrap**。同一 row 的 F/DH predictions 配对重抽；同一个测试数据集上的不同 seed 不是独立测试样本，报告两个变异层次，不重复计数为独立重复。
- 先行报告每 cell `n_pos/n_neg`、bootstrap CI、分数分布/校准曲线及按时间窗口的抽样不确定性。观察的 val/test 方向可能来自不同观察窗口和选择时间，不能仅由 train→val mixture shift 指定因果解释。
- 已有 test 被研究多次，后续任何基于 test 的选择均只作 exploratory/forensic；若发展方法，必须预留新的严格时间后置数据或外部数据。

**交付**：`r2_end_to_end_audit.md`、`s6r2_selected_exact_vs_grid.md`、`split_shift_evidence.md`、prediction-level 误差/不确定性表。通过条件：source SHA + run身份 + split rowhash + selected epoch + paired predictions 与 CSV 一致。

## 7. 第一项条件性新训练：B1 S6-R2 dense-checkpoint（8 runs）

**来源与严格范围**：仓库 `experiments/stage_S6_R2_selected_solution_analysis.md` §9 已明确：只授权一项**measurement replication**，相同 KuaiRand S6 F/DH、模型、Sample、比例、seed、50 epochs、超参；新增每 1 epoch 或每 2 epoch 的完整 state dump。不得捆绑新方法/调参/新损失。

| 维度 | 固定/变化 |
|---|---|
| 数据、切分 | KuaiRand 当前核实版本、原 row IDs、原 val/test、无重分割 |
| 设置/骨干 | natural / M3OE / S6 Sample |
| arms | F=FULL-history/FULL-current；DH=FULL-history/TARGET-current；private FULL |
| seeds | 2022, 2023, 2024, 2025；相同样本序列/初始参数 |
| horizon | 原 S6 的 50 epochs；非再搜索 epoch 预算 |
| checkpoint | 每 epoch 的 model weights、C/P/A 对应全部 optimizer state、step counts、RNG/iterator 必要状态，SHA256 |
| 评估规则 | 仅 validation 决定 R0–R4 的选点；test 是冻结规则下的一次预先规定的法证对比 |
| 估计量 | 同 epoch `Δ_policy(e)=M_DH(e)-M_F(e)`；各策略各自选点下 `Δ_selected`；timing/path decomposition 必须保留可定义性约束 |
| 主报告 | 6 cell val/test AUC、LL；ΔN/T/M、worst、selected/common；per seed，非仅 aggregate |

**验收**：8/8有效 paired runs、fingerprints 与原协议的预期一致、完整 epoch test endpoints 可在许可条件下重建；对 exact alternative selection 给真实选中 epoch 的 test 而不再使用稀疏 grid 的近似。对 S6 已读 test 的重用不算独立外部检验。

**STOP**：若重新按同一选点无可重复的 F/DH difference，或其表观增益主要来自选择时序/测试噪声，结束该线并保留否证；若仍存在同 epoch test 反转，只申请下一项**预注册机制诊断**，不得自动转算法开发。

## 8. 第二项条件性训练：B2 将共享 C 与架构 A_arch 正交隔离（最多 16 runs）

**必要条件**：A0 拿到有效 S5 source/derived artifacts；S5 既有 F/T/HT/DH 已独立复核；确认历史 runner 并没有做等价 C/A 分解；F0/科学协议授权 exact new action。

固定 KR natural、M3OE、Equal、d1 target、50 epoch、seeds 2022–2025、原 main/arch loop（main batch、arch persistent iterator）和 P=FULL。新臂为：

| arm | C 当前/写入来源 | A_arch 当前/写入来源 | 目的 |
|---|---|---|---|
| CF_AF | FULL | FULL | F 对照 |
| CT_AF | TARGET | FULL | 是否 C 足够 |
| CF_AT | FULL | TARGET | 是否 arch 足够 |
| CT_AT | TARGET | TARGET | T 对照 |

**注意**：这里没有复用 S5 的 `HT/DH` 术语。S5 中 HT/DH 是 history/current 来源交叉；这里是 C/A 参数分区交叉。

预注册主要 Δ 为 target d1 两任务 validation AUC 与 selected paired d0 non-target cost，以及 macro/worst/LL。除了 4/4 seeds 与 practical +0.003 之外，必须同时看 d0 >=合理 noninferiority guard，且对 seed2024 例外单独报告。

**解释与停机**：
- CT_AF ≈ CT_AT，CF_AT ≈ CF_AF ⇒ C 可承载主要效应；下一步才做 C 内部可达块。
- CF_AT ≈ CT_AT，CT_AF ≈ CF_AF ⇒ arch 分区优先；“shared Adam m/v 是根因”需改写。
- 两个单因子都不足而交互强 ⇒ C/A trajectory co-adaptation；不直接断言时域历史来源。
- CF_AF 与原 F 不稳定、CT_AT 与原 T 不稳定 ⇒ 新 runner 不合格，所有机制判读停止。

**新训练量**：四臂×4 seed=16 runs；如果能从保留原始工件按严格代码/数据/批次同一性复用 F/T，可减少新训练，但**只有同一有效执行协议**才允许称 paired reuse。

## 9. 低运行成本诊断：C1 共享 Adam 历史的 m/v/current 因子

在有效的 S5 checkpoint / source clone 与权限范围内：

1. `m_source ∈ {Full, Target}`、`v_source ∈ {Full, Target}`、`g_current ∈ {Full, Target}` 的 2×2×2 探针；分 C 与 A_arch，P 始终 FULL。
2. 同 θ、同 batch、同 step counter、同 bias correction、同 lr/decay/eps；生成 8 种 one-step `Δθ`，记录 update norm/cos、per-cell validation loss 的冻结局部变化。
3. **不能**直接把异轨迹优化器状态硬插另一条末端 θ 后训练并把跌幅叫“m 的因果贡献”。先沿同一 θ 轨迹用 ghost-state/shadow optimizer 收集两个 gradient source 形成的 m/v；再在同一 θ 做合法反事实一步。
4. 明确检查 Adam 的 m 与 v 通常来自同一梯度随机过程，混合的 `(m_F,v_T)` 不一定属于任何可达常规 Adam 轨迹；这些为数值干预不是天然部署算法。
5. 若出现可重复局部差异，还要有多步 replay/same-state 更新验证，确保 one-step utility 真的关联正常 selected AUC；否则 C1 只得出机制描述。

**STOP**：若 m/v 效应对 θ 所处轨迹、block/seed 极不稳定，或没有 selected/per-cell validation 对应，不继续发明 per-domain optimizer。

## 10. 受控目标估计器与域抽样：C2（理论 oracle 后再决定 16 runs/数据集）

核心假说：**Adam 对梯度噪声与坐标二阶矩敏感，相同期望目标的不同域抽样/重加权方式可以导致不同的选中解与域级权衡。** 这是要证伪的假说，不是已识别的 MDMT 规律。

### C2.1 先做固定参数、固定数据的数值 oracle（0 training）

| α（期望目标） | q（域采样） | 重要性权重 α/q | 是否应匹配同一总体目标 |
|---|---|---|---|
| Equal α=1/3 | Natural q=p | 1/(3p_d) | 与下行同目标期望 |
| Equal α=1/3 | Balanced q=1/3 | 1 | 与上行同目标期望 |
| Sample α=p_d | Natural q=p | 1 | 与下行同目标期望 |
| Sample α=p_d | Balanced q=1/3 | 3p_d | 与上行同目标期望 |

进行每域每任务独立梯度及 Adam update oracle：统计 loss/grad 期望，per-coordinate covariance、norm、zero-domain batch probability、effective sample reuse 与 minibatch exposure，使用多次 Monte Carlo 和分层统计，检验数值等价条件与极端 q 下的方差爆炸。额外计算 `Equal fixed-p weighting` 与 `batch-cell-normalized Equal` 的差异；后者不必具有同一期望，不可默认等价。

### C2.2 仅 oracle 后才允许的配对训练

一个数据集：4 arms × 4 seeds =16 runs。其他结构、batch size、优化器与 val selection 固定；要对齐**有效 main step 总数**和每种 q 的数据曝光记录，区分“相同计算预算”与“相同 unique samples”，后者通常不能完全同时保持。

主要 contrast 是 Equal natural-weighting vs Equal balanced-sampling、Sample natural vs Sample balanced-importance；跨 Equal/Sample 差仅用于 objective interaction，而不能标作单纯 sampling effect。若同目标各对比 gap 低于预注册阈值且状态/梯度差随 seed 不稳，STOP，不做新课程/权重算法。

## 11. 条件性边界：D1 域身份、规模、任务及骨干（不是一次全量网格）

启动仅在 A1/B1/B2/C2 至少有一个明确被证实的机制或反例且与既有 baseline 有效差异之后。按“只变一个因素”的顺序小批次执行：

1. **Domain identity**：在自然 ratio 固定的情况下分别选择 target d0/d1/d2，但 C、A、P 的有效梯度 mask/域 role 需对齐；不同 target 的损失规模和样本可用性记录，不把 d2 小当困难。
2. **Ratio 与 exposure**：在同一物理域和固定全局时间切分下，对训练子集做预先指定的限制/增强；分开总 N、独立 unique samples、曝光步数、训练目标 α 与 domain q。没有一个有限数据集能无限增广真实独立样本；不把重复曝光当成新增独立数据。
3. **Task relation**：MovieLens 二元任务嵌套，KuaiRand 任务关系不同；不能用两个数据集的差异证明任务相关性因果。先以同一输入样本构造合法的任务相关性诊断（labels agreement/conditional mutual info/positive rates），再考虑控制改变任务定义是否仍是同一个应用问题。
4. **Backbone**：M3OE → MMoE → PLE；先保证共享参数 reachable source mask 与 optimizer state 定义可比（专家访问拓扑不同，C/P 名字不可直接迁移）。容量、参数数、epoch/LR search budget 必须报告；P7 的历史 comp1 迁移不是这里的机制复制。
5. **至少一个不经此前实验选择的严格时间后置或外部数据集**，否则不能声称 test-reversal 对真实部署通用或新方法能泛化。

Go/no-go：两个数据集/至少两种结构在预注册选择协议下能重现指定机制方向；若只在 Equal/KR/d1 有效，则写成窄边界机制研究，不为了“跨域泛化”删去反例。

## 12. E1 方法发展：只有在明确问题成立时才允许

### E1.1 合法的创新空间

- **源-状态分离方法**：在不直接扔掉非目标域样本的前提下，针对实证确认的 C 或 A_arch 状态瓶颈，仅改变必要的 optimizer-state 构造；要求与 AdaTask 的 per-task adaptive history 机制正面区分。
- **估计器/预调节器耦合**：若 C2 证明同一总体目标下重采样与重加权造成不可忽略、可预测且对 selected checkpoint 有实用收益的差异，提出更低方差或更稳定的实现；不得把“重要性采样与 Adam 不同轨迹”本身叫作第一次发现。
- **域/任务专家可达性约束**：若 A1/B2 证明表面私有专家的跨域梯度可达导致特定损伤，在保持容量与共享带来的正迁移条件下做最小结构约束；需与现有门控/正则/SAF 对照。

### E1.2 至少的必要基线

- 强基线 `Sample` 与 `Equal`，matched SD/joint reference，原 S5/S6 diagnostic arms（仅诊断，不作为“已部署安全方法”）。
- 对应问题的 MMoE/PLE 原实现及 capacity matched baseline。
- 优化器路线：SGD/Adam（仅机制对照）、AdaTask (AAAI 2023)；方向类至少 PCGrad、CAGrad；动态权重类考虑 FAMO（需尊重各方法计算开销）。
- 自适应 MDMT 融合/重加权路线：SAF (AAAI 2026) 等最新同行工作，确保和训练目标、任务定义、骨干、专家数一致。
- 任一新方法至少一项“删除新增模块”“共享 vs arch only”“m only/v only（如相关）”“简单静态权重/统一平滑”“matched compute/params”消融。

**严禁**仅相对旧 Equal baseline 获益，忽略 Sample 的正常 selected best 与 non-target guard。

## 13. 验证、测试与统计协议（适用于全部阶段）

1. **主要 endpoint**：预注册 validation-selected checkpoint 的 six-cell macro AUC，以及若设置明确 target 则 `ΔTarget/ΔNonTarget`；不能先用 test 选择阈值、最终再报告 test 同一差。
2. **必须并列**：per-domain × per-task AUC、LogLoss、macro、sample-weighted、worst-domain/worst-cell；每 run selected epoch、common epoch、final。说明 validation/test 与训练权重 α/q 的不同。
3. **实用门槛**：历史项目使用 ΔAUC +0.003 practical / +0.005 strong，至少 3/4 seeds 同方向。沿用可作 exploratory gate，**不是** p-value，也不是普适 deploy threshold。
4. **预注册负向 guard（新提案）**：例如任何重要 cell 的 selected AUC 损害不超过 0.003，或 worst-domain 不劣于基线超过 0.003；是否使用以及什么权重必须在训练前根据应用价值冻结，不要随结果更改。目标的 +0.04 与 d0 的 −0.03 不构成“整体安全改进”。
5. **两种不确定性**：AUC 测试数据自身抽样不确定性做 same-row paired stratified bootstrap（cell 类比例保留）；训练随机性以 seed-block 配对 CI/散点/符号、异质性单独呈现，不能把 4 seeds × N测试样本乘作独立实验数。
6. **多重比较**：首次探索可以并列图表、分层标记；正式确认必须冻结 primary comparison、domain targets、选点策略、种子集、guard，并用适当的 familywise/FDR 校正、或保留未经开发选择的 confirmation set。
7. **split/test lock**：验证与测试按原协议冻结。KuaiRand 旧时间 split 不得随意 shuffle；已反复读取的 test 仅为历史审计，未来方法要有真正新的未用 holdout。
8. **按 seed 一一配对**：同初始化、同 input row ids、同 batch fingerprints（只在可实现同 exposure 下）、同随机数消费；修复实现 bug 必须声明“新协议”，旧 run 不可未标记混合。
9. **完整状态的可复现性**：任何 continuation 需包含 main+arch 所有 optimizer state（private m/v 亦然），scheduler/RNG/loader iterator；S8 报告明确 S7-B private fresh-Adam 的近似边界。
10. **停止决策**：允许 `NO`（完整运行且 gate 失败）、`INCOMPLETE`（未完成/工件缺失）、`NOT_RUN`（未授权/未触发），严禁统一写作“失败”。

## 14. 最小复核用例、日志结构与伪代码级断言

A0 编写单元/集成检查（可在受批准的独立环境开展）：

```python
# 伪代码；不是未经当前代码签名验证即可执行的替换实现。
assert set(C_ids).isdisjoint(P_ids)
assert set(C_ids).isdisjoint(A_ids)
assert set(P_ids).isdisjoint(A_ids)
assert len(C_ids | P_ids | A_ids) == len(all_trainable_param_ids)

# 对每个域任务的梯度访问（允许 0 / None，需与 forward 可达性一致）
for d, t in all_cells:
    grad = autograd.grad(loss_cell[d,t], relevant_parameters,
                         retain_graph=True, allow_unused=True)
    record_reachability_norms(d,t,grad)

# 强制配对 oracle：相同初值、batch、所有分组状态和 grad source
assert initial_model_hash(F) == initial_model_hash(T)
assert paired_data_row_hash(F) == paired_data_row_hash(T)
assert selected_epoch == argmax_val_macro_auc_with_frozen_tie_rule()
assert not test_used_to_select_hyperparameters
assert manifest_claim_matches_file_sha256()
```

每个 run 推荐统一 schema：`study_id, stage, dataset, split_hash, setting, arm, seed, objective_alpha, domain_sampling_q, effective_importance_weight, target_domain, model_code_sha, optim_code_sha, config_sha, init_sha, batch_sha, epoch, selected_epoch, step_main, step_arch, state_complete, checkpoint_sha, val_cell_auc, val_cell_ll, test_cell_auc[locked], source_grade, reviewer_verdict`。

## 15. 工作顺序与 run 成本上限（不是自动执行）

| 节点 | 工作 | 新训练 | 启动条件 | 停止/移交 |
|---|---|---:|---|---|
| A0 | 代码/工件/F0 门禁 | 0 | 立即仅可做合法只读工作 | 字节/权限/配对不合格则不启动 GPU |
| A1 | R2 AO04 val/test/rho 复核、E4 shift | 0 | 原始文件可读且使用合规 | 不可读则清单列 INCOMPLETE |
| A2 | S6-R2 F/DH 既有 artifact review | 0 | 既有工件与 manifest 完整 | 仅 coarse grid 不得声称 exact robustness |
| B1 | KR Sample F/DH dense 50ep | 8 | 严格独立 RUN 授权与工件检查；源报告单项许可 | 把 test/timing 解释边界缩小，不调参 |
| B2 | KR Equal C×A source | 最多16 | S5 关键材料核实、新设计和授权 | C/A 解释仍不清则停止算法 |
| C1 | m×v×current same-θ probes | 0 新训练优先 | B2 or existing checkpoint 支持合法状态 | 若 probe 与 selected 无对应，STOP |
| C2 | Equal/Sample × q oracle + pilot | 0 + 可选16/数据 | 理论数值 oracle、独立有效问题 | 同目标实际 gap 不足 -> STOP |
| D1 | 角色/比例/任务/backbone | 条件性 | 明确机制、预注册矩阵和资源 | 只剩单 setting 则局部研究 |
| E1 | 最小可发表候选 | 未定 | Sample 强基线真实可改善、独立 holdout | 任何 substantial guard violation -> STOP |

**初期实际优先预算**：先完成 A0–A2 的 0-run 审计，再最多请求 B1 8 runs；B2 16 runs 必须经额外 GO，不应与 B1 同时批量启动。若 B1 阴性，保留为科学反例，不为挽救假说扩大训练。

## 16. 文献对照与潜在创新性冲突（初步公开来源核验，不是系统全文创新性结论）

| 论文/时间 | 已确认相关主张 | 本研究不能重命名成创新的内容 | 若要新研究还需证明 |
|---|---|---|---|
| M3oE, SIGIR 2024 | 多域多任务 MoE/可微融合 | 再加少量域/任务门控就是新原理 | 对现有融合假设识别具体失效边界 |
| AdaTask, AAAI 2023 | per-task parameter update dominance 与独立自适应梯度累计/学习率 | 首次提出 task-specific Adam/moment separation | MDMT 结构中 domain×task reachability 与 coupled arch/main 条件下的新可证伪机制 |
| PCGrad, NeurIPS 2020；CAGrad, NeurIPS 2021 | 多任务梯度冲突与方向/优化目标处理 | 正负 cosine 相关，就应修改梯度 | 实际 applied Adam update、持久历史及非目标成本的因果价值 |
| FAMO, NeurIPS 2023 | 低成本的自适应多任务权重调整 | 又提出动态任务 reweighting 名词 | fixed objective 下的估计器/历史交互或 MDMT 特殊失效假设 |
| SAF, AAAI 2026 | MDMT 自适应特征/专家融合、sample weights、HSIC 稳定表征 | MDMT sample reweight + expert gate 作为泛称新颖 | 与 covariate/label shift、专家 reachability 和状态分离明确差别 |

权威入口：
- [M3oE, arXiv 2404.18465](https://arxiv.org/abs/2404.18465)
- [AdaTask, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26275)
- [CAGrad, NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
- [PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420-Abstract.html)
- [FAMO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html)
- [SAF, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38491)

**文献执行计划**：针对每个证明/假设完成全文阅读与引文追踪；记录 dataset/MDMT definition、optimizer state 更新式、loss 权重/采样假设、评价约束、负迁移定义、适用边界、已有 failure analysis、开源实现，主动寻找与自身候选完全同构的方法。没有完成这一步前，新颖性仅能标记 `OPEN`。

## 17. 论文研究方向分流与终止条件

### Track A（优先）：objective-contingent shared Adam state

可检验主张：在指定 MDMT 结构中，跨域 loss source 形成的 persistent optimizer state 能在 Equal 条件下调节目标与非目标域更新，并表现出与 Sample 不同的边界。**科学论文可能以否证/机制识别为主，不必有新算法。** 失败条件：代码/工件不可复核、C/A 隔离不成立、仅与某个不可比 optimizer 相关或无法超越协议差异。

### Track B（优先取证）：selection-time × generalization reversal

可检验主张：既有 KR Sample F/DH selected 解在 val/test 反号不是稀疏网格/错位 artifact，且有统计和时间结构解释。失败条件：精确 selected test 按 per-cell bootstrap 无可信优势、same-epoch 差消失、或现象仅是已见 test 的开发偏差。此方向必须有新独立检验才可进方法性 claim。

### Track C（备选）：same objective, different estimator & Adam

可检验主张：在固定 α、结构、有效计算预算下，改变 q 与 importance weighting 引起可复现的 Adam state/selected performance 交互，且对强 Sample 也有价值。失败条件：oracle 权重不等价、差异只来自任务设置/曝光、效应小于阈值或被 SGD/Adam 对照解释为非 MDMT 特有。

### Track D（暂缓）：fine-grained asymmetric fusion

研究 R2 B1 的 q 不可辨识、过拟合或 shift 边界，但在 MovieLens 当前阴性/KuaiRand 未完成条件下，不直接扩展参数或训练 B3/B4。需要预测级敏感度及独立数据正面证据后才能重新授权。

## 18. 面向 Codex 的下一条最小执行指令（建议直接复制）

> 在 `Caelestis1903/My-Research-Project@3e6642820b7b2f17f130a5ccd9d382a16e023811` 进行只读证据清点。不得修改现有代码、原实验目录、原 checkpoint、hold/gate 文件，也不得运行训练、恢复 optimizer、加载模型 checkpoint 或进行现有 F0 未批准的科学 RUN。先核对 F0 v1.5 B6 及最新有效独立 authorization，发现 NO_GO 则保持 STOP。只在独立且被许可的审计工作区建立 `artifact_inventory.json`、`missing_artifacts.json`、`runner_objective_optimizer_callgraph.md`、`r2_source_trace.md`，覆盖 S4/S5/S6/S6-R2/S8 与 `research_R2`。固定数据/代码/manifest SHA256、不同 R2 的身份、Equal/Sample 实际 loss、C/P/A_arch 梯度来源、test 解锁协议。每个文件注明可访问、来源、实际哈希、是否原始或仅报告；对缺失素材记 INCOMPLETE，不补造结果。提交审计报告及所有不能证明的假设清单，未经独立 review 不进入训练设计。

## 19. 固定提交的重要仓库出处

- [实验总览](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/Experimental_Global_Summary.md)
- [开放问题](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/Experimental_Open_Questions.md)
- [已停止方向](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/Experimental_Dead_Ends.md)
- [M3OE 主模型](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/m3oe/MDMTRec/models/multi_domain/model.py)
- [S3 梯度访问](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/stage_S3_gradient_access.py)
- [S4 报告](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/reports/stage_S4_analysis.md)
- [S5 报告](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/reports/stage_S5_analysis.md)
- [S8 报告](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/reports/stage_S8_analysis.md)
- [S6-R2 精确选择问题与许可的一项实验](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/stage_S6_R2_selected_solution_analysis.md)
- [E4 Chronological split 审计](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/reports/history_report/stage_E4_dataset_audit.md)
- [R2 模型实现](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/research_R2/src/ao04_v12_model.py)
- [R2 已存储指标](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/research_R2/analysis/selected_aggregate_metrics.csv)
- [F0 B6 事故后基线审核](https://github.com/Caelestis1903/My-Research-Project/blob/3e6642820b7b2f17f130a5ccd9d382a16e023811/experiments/reports/stage_F0_v15_B6_postincident_baseline_attempt03_review_v1.md)

---

### 最终判断

本仓库下一项最有价值的科研工作**不是立刻开发新的共享/动量/梯度/重加权模块**，而是把 S5 的条件性历史效应、S6-R2 的评估层反转和 AO04 R2 的非对称融合边界还原成可复制、可证伪、分区明确的证据链。只在某条链条揭示标准 Sample、匹配容量和已有方法尚无法解决的具体缺口后，才提出最小算法。正反结果均应进入正式科研证据。
