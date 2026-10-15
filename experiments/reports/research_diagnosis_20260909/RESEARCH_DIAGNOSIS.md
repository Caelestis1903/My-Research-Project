# 连续研究诊断：从规模角色不对称迁移，转向训练目标与优化器的交互

诊断日期：2026-09-09。材料边界：上一阶段为 2026-09-08 整合的 P0–P7；本阶段为 P0-1、P0-2、P0-3、P0-4、P0-5、P1-2。旧 P1 诊断与新 P1-2 不是同一批实验。

**研究决策：明显调整方向。停止以“Curriculum + Momentum 解决 M3OE 固有的大小域结构性 trade-off”为论文主线。优先研究不平衡条件下，训练目标、模型选择和优化器如何共同决定迁移结果；动量的剩余贡献需要匹配对照，Curriculum 暂从核心方法中移除。**

这不等于样本比例加权天然正确，也不等于动量无效。当前最扎实的发现是：在这套 MovieLens 训练协议下，一个简单的 loss-weighting 改动，解释了大量此前归于复杂优化机制的性能优势；修正目标后仍存在可重复的动量残余收益，其机制尚未验证。

## 证据范围、定义与审计纪律

读取工作目录 23 份原始/阶段/整合 Markdown 材料，并程序化读取其中全部表格。进一步读取六个新阶段共 400 个已保存 run 的 config.json、metrics_summary.csv，整理出 1,728 条 domain-task AUC 记录、432 行跨 seed 证据表。400 是目录中的训练记录数，包含重复验证与跨阶段锚点，不能当成 400 个独立统计重复。P0-2 复用的 P0-1 数据不另算独立证据。历史 P5–P7 的全部原始日志没有重新逐行审计；历史定量断言以对应原始报告为来源，不继承其中“Reviewer 已签署”所暗示的科学结论权威。

完整新阶段证据表：[EXPERIMENT_EVIDENCE_TABLE.md](D:/CODE/phase0-3/experiments/reports/research_diagnosis_20260909/EXPERIMENT_EVIDENCE_TABLE.md)。表中未保存的超参写“未报告”，不会用当前脚本默认值填历史缺失项。此前报告里的数值与此次复算不一致时，本报告明确标出。未运行新训练、未改动已有训练代码或实验报告。P1_3_matched_momentum 仅发现空指标表和已准备的脚本，不能当作完成实验。

三种量必须分开：

\[
T_{d,t}=AUC_{joint,d,t}-AUC_{single-domain,d,t},\quad
G_{d,t}=AUC_{method,d,t}-AUC_{baseline,d,t}.
\]

T 是相对单域参考的迁移差；G 是方法收益。小域 T>0 不代表方法没有损害小域，必须同时看 G。所谓 oracle 只是本项目训练出来的单域双任务参考，不是理论上界，也不是 M3OE 论文所用单域单任务 MLP 的同义词。

宏平均主指标为六个 domain-task AUC 的算术平均。P5–P7 中另有 pooled AUC；把不同任务、域的预测池化后计算 AUC，不等于宏平均，也不等于样本加权的分域 AUC。它可能受跨域/跨任务分数尺度影响。

**证据等级：**实验事实=记录或实现可核实；稳定现象=在明确重复单位上反复出现；统计证据=说明估计对象、重复单位、区间和假设；机制推测=与数据相容但未通过鉴别性干预；待验证假设=可被后续实验否定；论文结论=有明确数据、协议和适用边界的限定陈述。

本报告新增统计：先在同 setting、seed 下作差；再将同一个 seed 的三个 setting 平均，形成 n=4 的 seed-block 差值。报告样本 SD（ddof=1）、双侧 t 区间与名义 p；同时保留 12 个配对的描述统计。这样避免把同一个 run 的两个任务、多个域当作独立样本，但四个 seed 仍很少，正态近似不可验证，不能外推到新数据划分、用户总体或数据集总体。四个块的双侧精确符号检验最小 p 为 0.125，因此参数检验的小 p 不等于分布无关强证据。下文未宣称新增检验通过多重比较校正。

## 第一部分：恢复上一阶段研究状态

### 真正得到支持的旧 Observation

1. 旧 MovieLens 强不平衡设置中，comp1 的宏平均增益较大：r30 +0.016598、r100 +0.020233。但六档序列不单调，r3 为 −0.001231，且存在跨窗口及样本总量混杂。
2. 旧 KuaiRand baseline 大域相对单域参考明显落后；comp1 收窄大域 t0 缺口约 +0.009102，p=0.022753，为名义显著；t1 的 +0.017077 不显著，p=0.369。缺口未闭合。
3. comp1 在旧 MMoE/PLE 与适配的 ShareBottom 上也有效，已提示方法并非 M3OE 专属；多数旧移植实验只覆盖温和不平衡。
4. Curriculum 单独及相对 comp1 的边际收益没有稳定证据；旧“互为前置”“协同成立”不受数据支持。
5. 旧梯度诊断没有提供稳健的“小域 SNR 更低”证据；平均跨域余弦为弱正值。它既没有证明强冲突，也不能排除局部、短时或特定层冲突。

| Hypothesis | Previous Evidence | Confidence | Missing Evidence |
| --- | --- | --- | --- |
| M3OE 存在大小域结构性 trade-off | KuaiRand 大域 T<0、小域 T>0；ML 单域参照部分口径不齐 | 现象中等，结构性低 | 同子集单域参照、角色反转、目标权重控制 |
| 规模失衡是原因 | 强档方法收益较大 | 低至中 | 固定 N；区分域身份、总量、采样与 loss 权重 |
| Curriculum 有独立价值 | 旧 P6 主效应 p=0.657；正反方向无可靠分离 | 低 | 修正目标后的终点和动态对照 |
| 域条件动量有特有机制 | 公式 γ 优于未充分匹配的 uniform/SNR 控制 | 有效性中等，机制低 | 匹配记忆、启动偏差、Adam 二阶矩和 LR |
| Curriculum×Momentum 协同 | 组合点估计略高；没有可靠 interaction 检验 | 很低 | 2×2 差中差，而非组合对 baseline 的检验 |
| 小域噪声导致负迁移 | mild ML 上 SNR 不稳定，平均余弦正 | 不支持 | 强档、分层、逐步梯度及因果干预 |

上一阶段最关键的 RQ 是：**观察到的大域损失究竟是失衡数据本身造成，还是训练/评估协议造成；comp1 在排除协议解释后还有没有贡献？** P0-1 至 P0-5 基本按这个顺序提供了新证据。

## 第二部分：本阶段 Experiment Evidence Table

先列配置与观测，不在本节解释机制。完整逐域逐任务表见附表；正文汇总表不是对 432 行证据的替代。

### 共同设置

Dataset=MovieLens-1M；d0 年龄组 {1,18}，d1={25}，d2={35,45,50,56}；t0=rating>3，t1=rating>4，属于从评分构造的二元标签。N=187,500；seeds=2022–2025；LR=0.01、架构 LR=0.001、Adam、weight decay=1e-5、batch=40,960、最多50 epochs、patience=10，fix_optim_grouping=1。代码的当前 role-setting 路径没有 scheduler.step，不能因构造时传过 scheduler_params 就写“使用学习率衰减”。每 run 的保存配置是超参记录的优先来源。

comp1：共享梯度逐域 EMA，γ0=0.9、γmax=0.99；共享 Adam β1=0，β2=0.999；私有参数仍 β1=0.9。ratio50 的 L/M/S 对应 γ=(0.9,0.977,0.99)。Curriculum none/forward/reverse 分别 κ0=0/+0.5/−0.5，作用于融合权重，T_c=25 epochs。不同组件参数化差异见第六部分。

| Experiment ID | Model/arms | L/M/S domain size | L:S | 新跑/复用 | Baseline、改动与评价 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | M3OE baseline/comp1/逐域参考 | 150000/30000/7500 | 20 | 新60；运行事故重复已去重 | 3种角色轮换；equal loss；T、G |
| P0-2 | 同上 | 62500/62500/62500；125000/37500/25000；150000/30000/7500；150000/34500/3000 | 1/5/20/50 | 新140，复用60 | 固定N ratio sweep；ratio1只一个退化setting |
| P0-3 | MMoE baseline/comp1/逐域参考 | 平衡或150000/34500/3000 | 1/50 | 新80；M3OE参照复用 | 共享数据缓存；不同backbone |
| P0-4 | M3OE baseline | 150000/34500/3000 | 50 | 新36；oracle复用 | equal/sample/sqrt训练权重×equal/sample验证选点 |
| P0-5 | M3OE，weighting×comp1 | 同上 | 50 | 新12；其余三臂复用 | equal选点；2×2 interaction |
| P1-2 | M3OE A0/B0/C0/A1/B1/C1 | 同上 | 50 | 新72；A臂为锚点复验 | sample权重；课程正反×动量；AUC、peak、norm |

ratio50 样本占比 L/M/S=0.8/0.184/0.016。每任务的 equal 权重=1；sample=2.4/0.552/0.048；sqrt约1.851/0.888/0.262；两个任务合计权重归一为6。这固定了标量权重总和，**不保证共享梯度或 Adam 实际步长尺度相同**。

### ratio sweep 的事实

| Ratio | Baseline 宏AUC | comp1 宏AUC | G overall | T_L | T_M | T_S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7405 | 0.7435 | +0.0030 | 不定义大小角色 | 不定义大小角色 | 不定义大小角色 |
| 5 | 0.7338 | 0.7405 | +0.0067 | −0.0018 | +0.0235 | +0.0304 |
| 20 | 0.7185 | 0.7334 | +0.0149 | −0.0146 | +0.0194 | +0.0487 |
| 50 | 0.7098 | 0.7271 | +0.0174 | −0.0155 | +0.0186 | +0.0598 |

以上为报告均值，跨setting总体SD原报告未按独立seed统一报告；逐setting×domain×task的SD已复算入附表。ratio1三个域的平均T分别约+0.0159/+0.0212/+0.0074，不将它们命名为L/M/S。

### P0-4/P0-5 的四臂结果

| Weighting | comp1 | 宏AUC Mean | T_L Mean | T_M Mean | T_S Mean | 对本权重baseline的G |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| equal | off | 0.709760 | −0.015545 | +0.018645 | +0.059787 | 0 |
| equal | on | 0.727120 | −0.003405 | +0.031019 | +0.087352 | +0.017359 |
| sample | off | 0.730845 | −0.001210 | +0.030075 | +0.097276 | 0 |
| sample | on | 0.736252 | −0.000067 | +0.032678 | +0.109752 | +0.005407 |

| 配对效应 | Mean | seed-block SD | 95% t CI，n=4 | 名义p | 12个setting×seed方向 | 与旧结论 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| sample−equal，无comp1 | +0.021085 | 0.006968 | [0.009997,0.032172] | 0.0091 | 12/12正 | 新发现，削弱结构性解释 |
| comp1 on−off，equal | +0.017359 | 0.007170 | [0.005950,0.028769] | 0.0168 | 12/12正 | 方向一致 |
| comp1 on−off，sample | +0.005407 | 0.000620 | [0.004421,0.006394] | 0.00041 | 12/12正 | 残余真实存在于当前协议 |
| 上述两种G的差，sample−equal | −0.011952 | 0.006866 | [−0.022878,−0.001026] | 0.0400 | 12/12负 | 新的负交互证据 |

这里SD是“跨三个setting平均后的配对差”的seed间SD，不是方法AUC的SD。原P0-5中T_L旁的±0.0023等来自cell级SE，不能当成seed SD。

### P1-2 的完整四臂与反向控制

Mean为六项宏AUC；SD为先跨三个setting平均后四个seed的样本SD。Curriculum=none仍有Adam动量；“comp1 off”不等于无任何动量。

| Arm | Curriculum | comp1 | Mean | seed-block SD | Δ vs A0 | 是否稳定、与旧结论一致性 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A0 | none | off | 0.730845 | 0.001866 | 0 | 参照 |
| B0 | forward +0.5 | off | 0.731683 | 0.002375 | +0.000839 | 配对仅7/12正；重复旧课程弱证据 |
| C0 | reverse −0.5 | off | 0.732007 | 0.002684 | +0.001163 | 8/12正；未显示正向优势 |
| A1 | none | on | 0.736252 | 0.001933 | +0.005407 | 动量收益12/12正 |
| B1 | forward +0.5 | on | 0.736310 | 0.001923 | +0.005465 | 相对A1均值仅+0.000058，7/12正 |
| C1 | reverse −0.5 | on | 0.736347 | 0.002012 | +0.005502 | 相对A1均值仅+0.000095，7/12正 |

P1-2原报告给的是配对median/IQR，不能把它们当Mean/SD；本表是此次独立复算。峰值epoch正向课程配对中位变化为0或±1；没有稳定≥2 epoch平移。正反课程记录到的rho最大仅0.88/0.92。

### 必须纠正的原报告表述

| 原说法 | 本次核对 | 对结论的影响 |
| --- | --- | --- |
| P0-1“36/36 cell提升” | 3setting×3域×2任务只有18个均值cell；18/18正，逐seed为69/72正 | 广谱平均提升成立，逐cell无损不成立 |
| 平衡下全部cell正迁移 | ratio1 d2t1 seed2023 T=−0.000961 | 应写所有域任务均值正，保留个别反例 |
| strong M3OE L为23/24负 | P0-2 CSV复算为22/24负 | 数量小误差，不改变主要趋势 |
| P0-4有18个run选点被截断 | 18是重复到6个任务域的cell行；独立run只有3个，均在equal训练/sample选点 | 3/12该臂受限；不是18/72 run |
| “截断只会使test效果保守” | 更多epoch可能提高最大val，但test可升可降 | 不能称test收窄幅度的严格下界 |
| P0-5交互超过0.005所以显著 | 效应阈值不是统计检验；本次块级p≈0.040，未校正 | 存在名义证据，但原论证无效 |
| 69%收益是补偿机制 | 68.85%是条件收益衰减的描述比例 | 不能当作因果中介贡献率 |
| sample+comp1负迁移完全消失 | 24个L任务×seed中11个仍T<0；6个L均值cell中3个T<0 | 只能说角色总体均值近零 |
| equal目标“错误”、sample目标“正确” | 两者分别对应不同风险偏好 | 改为当前协议下不同目标的效果，不作普遍价值判断 |
| 旧P2修bug“方差缩小8倍” | 报的是SD约8倍；variance是平方量 | 稳定性术语需修正 |
| P6“uniform γ=1” | 实际是0.9；报告部分表头仍残留错误 | γ=1将冻结EMA，不能照抄 |
| P6把ml_r30写成KuaiRand占比 | 数据集与域比例串写 | 机制解释不采用该段 |

另有三个不同的“worst-domain”：P0-2先跨setting/seed平均后取六个cell最小值；P0-4/P0-5对setting×d×t均值取最小值；P1-2每run先对任务平均再取最差域。0.6785与0.6425的差不能当作新阶段退步。本报告只在相同定义内部比较；后续统一为每run的 min_d mean_t AUC，并另报worst-cell。

## 第三部分：本阶段真正增加的知识

### A. 重复验证

- **规模角色与迁移方向有关，但受训练目标条件约束。** equal loss下，角色轮换使平均负迁移落在L；在ratio20、50及两个backbone中重复。不是“域d2天生受损”。
- **固定总量后，失衡关联仍在。** 当前四档中G均值约0.003→0.0067→0.0149→0.0174。排除了总N下降是唯一原因；没有排除每域绝对样本稀缺和算法γ随ratio变化。
- **comp1有效性比其机制证据强。** sample权重后12/12整体配对仍为正；三个setting均值增益分别+0.003723/+0.007898/+0.004601。
- **Curriculum没有稳定边际贡献。** 旧P6和新P1-2方向一致，新阶段补上了反向、动态及interaction检查。

### B. 新现象

最关键是loss weighting的作用大于新增动量：sample−equal宏AUC约+0.0211；L缺口均值由−0.015545变−0.001210，点估计收窄92.2%；S获益也增加。comp1的剩余G分角色约L +0.001143、M +0.002603、S +0.012476，收益集中位置发生变化。

### C. 消失或失去资格的旧现象

- 大域大幅负迁移在sample权重下缩小到接近零的角色均值，原“不可避免牺牲大域”的强叙事失去支持。
- “组合最好”不再能承担协同结论；P1-2 B1−A1均值仅约0.000058。
- M3OE特有问题的定位失去支持；MMoE复现同型现象。不过“两架构同型”不等于所有MDMT架构都如此。
- 旧r3负凹陷未被本阶段直接复核：新sweep没有3:1且协议变化，不能写成“已消失”。

### D. 必须保留的反例

1. ratio50、d0_large、d2t0、seed2022，小域baseline T=−0.016563，小域并非永远正迁移。
2. 同ratio与setting，d0t0、seed2024，大域baseline T=+0.000598，且comp1 G=−0.002033；大域也不是每seed都受损或被修复。
3. P0-1 d0_large seed2022的d1t1、d2t1在comp1下下降；“18个均值cell全正”不能掩盖它们。
4. 固定150k的单域参考中d2 AUC最低；d2作为小域时comp1收益反而小于其作为中/大域时，否定size单变量解释所有收益。
5. **旧KuaiRand是重要跨数据集反例。** comp1相对baseline的小域d0t0/d0t1约−0.008374/−0.009259，小域d2t0约−0.007624，d2t1约+0.003988。小域虽仍高于单域参考，却受到了方法损害；不能宣称“大域改善且小域无损”，也不能据此宣称Pareto前沿外移。
6. P1-2动量层加forward课程时，worst-domain配对只有3/12为正，中位差−0.000909；反向课程并未稳定更差。

## 第四部分：H1–H5 的 Bayesian-style 更新

Previous Confidence按旧P0–P7可支持的程度给出，不沿用旧报告过强结论。Updated Confidence严格使用指定标签。

| Hypothesis | Previous Confidence | New Evidence | Counter Evidence | Updated Confidence |
| --- | --- | --- | --- | --- |
| H1：M3OE在高度失衡中有结构性的small gain↔large degradation | 中等现象证据、低结构证据 | equal目标下固定N/角色反转重复 | sample同时改善L/M/S；MMoE也出现；L平均缺口降92% | Strongly weakened |
| H2：trade-off与规模失衡本身有关，非仅身份/难度 | 低至中 | 固定N和身份轮换排除身份恒定解释；ratio剂量响应 | 每域稀缺、γ和ratio联动；目标权重改变主要现象 | Moderately strengthened |
| H3：当前Curriculum能改变节奏并改善该问题 | 低 | 完成正反×动量及动态诊断 | 无稳定终点/peak改变；无正向优势；课程未完整结束 | Moderately weakened |
| H4：动量有跨域稳定机制，超出普通优化器加速 | 有效性中、机制低 | sample下仍+0.005407且12/12正 | 缺匹配uniform/Adam；无偏差校正；二阶矩改变；SD未下降 | Insufficient evidence |
| H5：Curriculum+Momentum存在真正协同 | 很低 | 完整2×2差中差 | interaction均值−0.000781，CI跨0；组合几乎等于动量 | Moderately weakened |

H2增强的是“规模分配在当前协议中有因果作用”，不是“imbalance脱离objective就独立造成不可避免trade-off”。H1中“M3OE独有”的子命题可在当前两个实现的范围内拒绝；普遍结构性命题尚不宜声称数学上被彻底证伪。H3/H5应停止作为当前论文卖点，但不能由有限阴性结果推翻所有Curriculum设计。

## 第五部分：Domain Size 到底是什么变量

| 变量 | 当前证据/控制 | 能否作为G的机制解释 |
| --- | --- | --- |
| domain size n_d | 主动改变且角色轮换 | 有影响；未与每域统计精度分离 |
| sample ratio p_d | 固定N时p_d=n_d/N | 与n_d数学共线，不能估计独立效应 |
| domain difficulty | 同150k单域d2仍最低 | 排除small=hard；只能用同预算性能作困难代理 |
| baseline AUC | ratio升高时baseline降低，G升高 | 相关仍可能是低基线/可改善空间；G本身包含−baseline，直接相关有数学耦合 |
| convergence speed | comp1层peak较晚；域间峰值错开 | 可疑调节变量；峰值晚不等于更稳定 |
| class imbalance | 新报告未给逐域逐任务正负率 | 未排除；两个rating阈值标签相互依赖 |
| distribution shift | 身份轮换固定域定义，未消除联合混合分布改变 | 不能声称完全控制；KuaiRand时间切分尤其不同 |
| task difficulty | 两任务都测，但不是两个独立问题 | 需要任务分层，禁止把两任务当独立重复 |
| label sparsity | 两标签由rating构造；有效正样本数未报告 | 不等于domain size；新数据缺标问题尚未测 |
| gradient magnitude | 旧mild设置小域范数较大；新有norm | 与loss权重、二阶矩和参数块共同决定影响，不等于样本贡献 |

当前最合理的层次是：**T的符号更紧密地跟随size role×objective；G的大小更紧密地跟随objective与优化器的交互，同时被域身份、样本信息量和基线质量调节。** 不支持“G只与size有关”或“只与difficulty有关”。

固定N使全局数据稀缺解释明显减弱，却不消除小域从25k降到3k所导致的统计稀缺。小域单域参考同步变差，所以T_S上升部分是参照降低。ratio20→50保持L=150k而压缩S、增加M，是较有价值的组成对照，但仍不是只改变一个抽象ratio且不改变分布的实验。

要分离独特样本信息与训练频次：固定各域unique样本池，改变域抽样频率，同时用重要性权重固定目标风险；再固定频次改变unique样本量。这比继续把size、baseline AUC和G直接画相关图更有鉴别力。若使用回归，须按域、任务、setting分层，使用独立seed估计baseline，避免G与其baseline共享测量误差；本批数据不足以作可靠多变量因果回归。

## 第六部分：Curriculum 到底在做什么

当前方法不是按难度重排样本，而是在early epochs改变小域自身与跨域信息的融合比例。三个域仍参与每个batch的训练。难度次序没有被独立测量，所以称“易到难训练”不准确。

| 排序 | 解释 | 证据等级 | 判断 |
| --- | --- | --- | --- |
| 1 | I 当前无法判断产生有效收益的机制 | Directly supported | 终点与有限动态不足以识别有益机制 |
| 2 | H training schedule effect | Directly supported（实现层） | 明确改变融合权重时间轨迹；未证明有益性能效应 |
| 3 | G implicit reweighting | Plausible | 改变前向会改变反传贡献；未测等效域权重 |
| 4 | C 改变域学习顺序 | Unsupported | peak配对中位无稳定≥2变化；不排除更细时间尺度变化 |
| 5 | A 小域representation更好 | Unsupported | 无冻结表征probe或迁移质量检验 |
| 6 | B 减少early dominant-domain interference | Unsupported | 未记录方向性干扰；“dominant”不能仅按样本数定义 |
| 7 | D 降低gradient conflict | Unsupported | norm不能检测角度或冲突概率 |
| 8 | E expert specialization | Unsupported | 无新专家分工/消融证据 |
| 9 | F 改善gate allocation | Unsupported | 融合标量被操纵不等于learned gate分配改善 |

这里没有把未知解释全部写为Contradicted。当前数据主要反对“指定课程已经带来可重复可用的独立增益”；没有直接观察并否定每个潜在机制。

还应限制三项原有强论断：

- forward初始小域融合由sigmoid(1)=0.731降到sigmoid(0.5)=0.622，不是“sigmoid 1.0→0.62”。
- MDMTRecBeta将两个标量扩成D+T个可学习参数，D=3/T=2时增加3个标量。初始化且rho=1时等价测试，不证明训练后容量或所有参数状态等价。当前B/C与A对照仍混有这一参数化差异；它很小但不能说零容量变化。
- 所有课程run未到rho=1。当前测的是“部分课程＋共同早停”，不能证明完整课程无效。基于预算可以drop当前组件；无需为了保留方法不断扫κ。若未来重启，需同参数化κ=0参照和训练结束后的统一rho=1评测。

## 第七部分：Momentum 到底在做什么

实现上，输入每域加权梯度g_d后：

\[
m_{d,k}=\gamma_dm_{d,k-1}+(1-\gamma_d)g_{d,k},\quad
\widetilde g_k=\sum_dm_{d,k}.
\]

这直接证明存在逐域时间平滑，不证明它是收益来源。所有域每步更新EMA，**小域不是因抽到次数少而少更新buffer**。

| 待判断机制 | 当前证据 | 充分性 |
| --- | --- | --- |
| 平滑domain-specific gradient | EMA公式直接存在 | 算法事实充分；收益归因不足 |
| 减少振荡/梯度方差 | 无同checkpoint同batch的PSD、方差干预 | 不充分 |
| 改变聚合方向 | 不同γ理论上可以改变 | 未记录实际方向差 |
| 改善cosine similarity | 旧有原始梯度均值，新无严格raw/post-EMA对照 | 不充分 |
| 延长小域梯度记忆 | γ_S更大，核衰减更慢 | 数学事实；小域受益原因未验证 |
| 减少大域短期支配 | sample下权重大域更强，EMA启动还可能更抑小域 | 无支持，方向不能先验断言 |
| 改善课程顺序 | 课程×动量差中差没有正向证据 | 不支持当前叙事 |
| 提高seed stability | A0/A1块级AUC SD约0.001866/0.001933 | 当前未见降低；不能将配对G稳定等同于方法方差下降 |

### 最危险的可检验替代机制：启动期隐式权重

若初始化m=0且梯度近似常量，m_{d,k}=(1−γ_d^k)g_d。未做偏差校正时，小域大γ导致启动期累计系数更小。当前ratio50每epoch约5步；若到第40步，L系数约0.985、M约0.606、S约0.331。故本方法可在有限训练窗内产生**随时间变化的域权重**。这是根据实现推导的机制候选，不是已测得的因果中介。

不能把每步的(1−γ)当作永久缩小梯度：在常量梯度稳态，EMA单位增益回到1。也不能把上述系数直接当作参数更新比例，因为Adam随后做二阶矩归一化。

### 第二阶矩与普通Adam不等价

baseline Adam的v跟踪原始聚合梯度的平方；comp1把EMA后的\tilde g交给β1=0的Adam，v跟踪的是滤波后梯度的平方。加上无偏差校正和Adam中的weight decay，其差异超出“只改变一阶记忆长度”。因此matched Adam是重要控制，但不是数学等价控制。现有stacked比较只能说明该设置下叠加较差，不能证明超出普通momentum的创新机制。

### 已准备P1-3脚本的匹配定义需修正

脚本推导τ*=0.8×10+0.184×43.478+0.016×100=17.6，γ*=0.94318，算术成立。但它同时将f_d写成“每域每步更新，f_d=1”，随后使用p_d当f_d。这个解释不一致：17.6是样本占比加权的时间常数，不是按EMA实际更新频率加权的时间常数。均等域权重下时间常数均值约51.159，对应γ≈0.98045，又是不同匹配准则。

二者都只能匹配一个标量摘要；单一指数核不可能同时复制多个γ混合核的完整时域和频域响应。保留0.94318可作为预指定参照，必须准确命名，并给uniform控制一个小而公平的验证集选择预算。只击败0.94318不足以判定domain-conditioned结构独有。

## 第八部分：Curriculum×Momentum 是否协同

明确按每seed/setting计算：

\[
I=(B1-A1)-(B0-A0).
\]

当前I均值−0.000781，seed-block 95% CI约[−0.002520,+0.000957]；median−0.001015，12配对仅5个正。不能将两个median直接相减得到interaction median。B1−A1均值约+0.000058，95%块级CI约[−0.000420,+0.000535]，相对当前0.002实际效应门槛非常小。

**最符合情况D：主要来自Momentum。** 加性预测的组合增益约0.005407+0.000839=0.006246，实际约0.005465；有负交互倾向，但不足以声称已证明拮抗。不同域可能有异质性，现有分域信息不足以把情况E升级为稳定科学结论。

当前可作方法选择决策：drop Curriculum。当前不能作普遍理论结论：Curriculum从不有效、所有协同已被证伪。

## 第九部分：审稿人最危险的替代解释

| 因素 | 已控制部分 | 剩余风险 |
| --- | --- | --- |
| 1 random seed | 同seed配对、4seed | 不是12/24个独立seed；同数据划分的适用范围有限 |
| 2 learning rate | 名义LR一致 | 实际更新范数未匹配；各臂公平LR调优缺失 |
| 3 early stopping | 同patience和宏选点 | 不同轨迹选不同预算；课程未完成；posthoc受截断 |
| 4 optimizer | 新阶段fixgroup统一 | β1、偏差校正、二阶矩输入同时变；记忆控制未完成 |
| 5 loss weighting | P0-4/5主动干预 | equal/micro偏好不同；共享和私有同时加权，未定位作用路径 |
| 6 baseline tuning | 固定配置可复现 | 未证明简单Adam+公平调参、正则化已足够强 |
| 7 total steps | joint固定N、batch | 单域小池每epoch约1步，joint约5；早停进一步改变总步数 |
| 8 sampling | 缓存与角色子集可追溯 | unique样本量、频次、覆盖度尚耦合；抽样seed与训练seed未正交 |
| 9 domain difficulty | 身份轮换、等量oracle | 子集代表性、冷启动用户/物品与质量仍可能变化 |
| 10 ceiling effect | 没有AUC接近1的极端天花板 | baseline更低有更大改善空间；与G共享误差 |
| 11 distribution difference | 新ML分割固定 | KuaiRand预处理非完全原论文协议，不能用总体统计一致证明分布对齐 |
| 12 expert capacity | 同backbone各臂大体一致 | 跨backbone及单域模型容量不同，不能叫理论oracle |
| 13 gate capacity | comp1 gate结构未改 | Curriculum扩展融合参数；无路由机制统计 |
| 14 parameter count | comp1仅增buffer，不增模型参数 | 旧ShareBottom适配有4倍底部；新课程增加标量 |
| 15 regularization | weight decay数值相同 | Adam耦合L2、EMA、选点共同改变有效正则化 |

**最危险五项：loss weighting、optimizer/LR与基线调优、early stopping/总steps、统计单位及单域参考定义、跨数据集分布与样本频次。**

其中，loss weighting已破坏“固有trade-off只能靠新方法缓解”的核心前提；若matched optimizer＋公平调优解释全部残余，便直接破坏“域条件Momentum提供独有机制”的核心结论。若第二数据集不复现weighting作用，“一般MDMT目标设计问题”也必须降级为单协议案例研究。

另一个审计问题是train_loss日志：comp1路径记录sum未加权BCE，sample/off记录加权BCE。当前跨动量层的train_loss曲线不能直接比较。新metrics_summary含逐cell test logloss，而旧P5–P7缺失test logloss；不能把旧缺失结论外推到新批次。新A0/A1宏logloss均值约0.552124/0.528411，方向支持残余收益不限于AUC，但并非机制证明。

## 第十部分：研究方向排序

1–5分仅为导师判断；“重叠/质疑风险”越高越不利，其他越高越有利。novelty是限定的一手文献定位，不是已完成系统新颖性检索。

| 排名 | Direction | 事实匹配 | 科学意义 | 新颖潜力 | 机制清晰 | 实验证据 | 与M3OE重叠 | 质疑风险 | story潜力 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | D Optimization Dynamics under Imbalanced Multi-Domain Learning | 5 | 4 | 3 | 2 | 4 | 2 | 3 | 4 |
| 2 | B Imbalanced Multi-Domain Multi-Task Learning | 4 | 4 | 2 | 2 | 4 | 2 | 3 | 3 |
| 3 | C Mitigating Small-/Large-Domain Trade-off in M3OE | 2 | 3 | 2 | 2 | 2 | 4 | 5 | 2 |
| 4 | E Gradient/Expert Competition under Domain Imbalance | 2 | 4 | 3 | 1 | 1 | 3 | 5 | 2 |
| 5 | A Hard Domain Learning in M3OE | 1 | 3 | 2 | 1 | 1 | 4 | 5 | 1 |

D应收窄为“目标权重与优化器交互”；B只是问题背景。C可作为被审计的初始现象，不能继续描述为已经证实的结构性问题。E需要明确梯度/专家的直接机制证据，当前不该改名抢跑。A缺乏独立hardness定义，最不合适。

文献定位：M3OE原论文已经明确提出MDMT seesaw和域任务知识融合，因此“发现MDMT有此消彼长”本身新颖性不足。[M3OE原文](https://arxiv.org/html/2404.18465v3)

复杂多任务优化收益可被普通正则化/稳定化解释的审稿问题已有直接先例；本项目不能把这一一般批评包装为首次发现。[Kurin等，In Defense of the Unitary Scalarization](https://arxiv.org/abs/2201.04122)

按loss分别维护矩的思路也已有MTAdam；2026年MAdam预印本进一步讨论了loss权重与Adam二阶矩的交互。它们不证明本项目机制，但提高了“域条件EMA”创新举证要求。[MTAdam](https://arxiv.org/abs/2006.14683)、[MAdam，预印本](https://arxiv.org/abs/2606.03904)

## 第十一部分：当前真正可能的论文核心

**Observation：**在fixed-N身份轮换实验中，equal目标下迁移差跟随规模角色；sample weighting同时改善L/M/S均值，并使comp1边际收益大幅缩小。

**Problem：**将未经目标权重、模型选择和优化器对照的迁移差，直接归于架构竞争或复杂优化机制，会产生错误归因。相同的“负迁移”标签可能混合统计共享收益、风险偏好、有限步优化和参考模型差异。

**Hypothesis：**小域在equal-domain目标下拥有更高单样本贡献，有限训练中的目标估计方差和更新尺度可能导致泛化受损；不同EMA记忆还可能产生启动期的隐式权重与二阶矩变化。其因果中介尚未验证。

**Intervention：**先显式声明所需风险目标并校准loss/sampling/selection，再测试轻量域条件时间滤波是否优于强普通优化器。不保留缺乏独立价值的Curriculum。

**Evidence：**角色轮换、固定N sweep、第二backbone同型、weighting×selection、weighting×momentum、课程正反阴性结果。

**Missing Evidence：**匹配优化器控制＋独立数据集的目标干预；若要机制论文，还需要启动权重与二阶矩的鉴别性干预。

目前可写的限定结论：

> 在MovieLens、三个年龄域、双评分阈值任务及当前训练协议中，loss weighting强烈调节规模角色不对称迁移；简单样本比例加权大幅降低大域平均缺口，且保留集中于小域的动量增量。当前课程没有稳定边际价值。

不能写：现有MDMT必然存在结构性trade-off；已恢复所有域的Pareto最优；已证实gradient conflict/expert competition；已提出有普适性的全新动量机制。

建议的一句话主线（诊断型，避免把待验证方法写成成果）：

> Existing methods combine domain-wise objectives, but under unequal domain sample sizes they can exhibit protocol-dependent transfer asymmetries. We find that, in our controlled MovieLens study, changing loss weighting removes most of the average large-domain deficit and substantially reduces the apparent benefit of domain-conditioned momentum. Therefore, we propose an objective-aware evaluation protocol to distinguish objective-design effects from residual optimization benefits.

## 第十二部分：下一阶段最小实验设计

下面的P0/P1/P2是“下一轮优先级”，不是已完成阶段的编号。记全局H1为用户原命题“结构性small gain↔large degradation”；机制实验另外说明局部假设。一个实验指一个有判别力的对照矩阵，不是一次训练。先完成已有数据的统计/命名修正，不需要额外训练。

### P0-A：残余动量的匹配对照——唯一首选

**Question：**sample weighting下+0.005407残余，是否需要domain-conditioned EMA？

**Competing Hypotheses：**域条件时间结构；普通uniform EMA/Adam的记忆长度；有效步长/选择时点差异。

**Design：**保留ratio50、三角色轮换、相同子集、sample loss和equal验证选点。四臂：A标准Adam；B当前comp1；C共享梯度uniform EMA＋Adam β1=0；D普通Adam共享β1匹配控制。先利用已规划P1-3的12+12新增控制run作筛查，A/B复用但严格标为非独立复现；不改变现有训练时域来强行冒充matched-step。C的0.94318明确称p加权时间常数匹配。若C落后，在宣称B结构特有之前，给C一个小验证集预算（例如0.9/0.94318/约0.98045），与B同等LR选择机会，并在新seed确认。记录所有臂同step的指标以及val选点结果；不可用test挑γ或LR。

**Metric：**宏AUC、逐域任务G、worst-domain、test logloss、配对SD；实际更新范数、更新总步数、每域EMA更新次数；γ对应的核质量和记忆定义。

**Expected if H1 true：**在公平控制下仍存在稳定的域间相反迁移，并且无法通过简单对照同时改善；但B>C本身不证明H1。

**Expected if H1 false：**可通过普通目标/优化器获得同步改善；局部机制上，若C或D与B实际等效，则残余不需要域条件结构。

**Interpretation：**C≈B：简化为普通稳定化；D≈B：普通optimizer调参解释力增强；B>C且固定step也优：支持继续检查域条件结构；只在早停结果优：优先研究schedule/selection。

**Research Decision：**C/D等效→Change mechanism或Pivot research direction；B稳定胜出→Continue机制研究；只胜一个未充分调优的C→Modify hypothesis，不宣布胜利。

### P0-B：第二数据集的weighting×momentum×架构交叉验证

**Question：**MovieLens目标权重作用能否迁移到KuaiRand；它是否只是共同训练器的局部现象？

**Competing Hypotheses：**跨数据集的目标条件效应；MovieLens标签/年龄分域特例；backbone或时间分布差异。

**Design：**KuaiRand已有时间划分上先做M3OE与MMoE的equal/sample×off/on，所有臂fixgroup、同split、相同预算与选点。先在M3OE配对四臂做低成本准入，再有条件扩展MMoE；单域参考必须取joint相同训练子集且预算明确，旧不齐的参考只作历史背景。同步报告训练、验证、测试各自占比，不使用全量24:1替代训练约19.5:1。

**Metric：**两任务逐域T/G、宏与sample加权AUC、worst-domain、logloss、单域缺口CI；特别关注此前受损的小域。

**Expected if H1 true：**目标权重干预后，大域损失与小域收益仍稳定对立，且跨backbone存在。

**Expected if H1 false：**简单weighting可使角色均值同时改善，所谓牺牲主要依赖目标配置。

**Interpretation：**同型复现→加强“目标条件”主线；ML有效而Kuai无效→研究数据分布/目标交互，不声称普适修正；仅M3OE有效→增加架构限定。

**Research Decision：**复现→Continue；反向/消失→Modify hypothesis；只单协议成立→Pivot research direction为诊断案例。

### P0-C：训练目标与模型选择的独立预算确认

**Question：**weighting改善是否依赖早停/预算；改变loss究竟修复泛化还是只挑到了不同checkpoint？

**Competing Hypotheses：**训练目标改变整条泛化轨迹；相同轨迹上的选点效应；effective LR/训练不足。

**Design：**ML ratio50三角色，equal/sample两训练臂，不加comp1。统一运行满预先固定的共同step预算，保存每epoch；分别用equal/sample验证指标选点，并报告预指定共同step的性能。只用val选点。必要的LR小网格同预算用于两臂，不能仅调方法臂；已完成P0-4数据可作启动估计，但补充训练需统一重新运行，不能拿已截断轨迹冒充完整预算。

**Metric：**逐step T/G、train与val同定义loss、过拟合间隔、被选epoch、宏/加权/worst-domain AUC。

**Expected if H1 true：**在充分训练、公平调参和两种选点下仍存在不能同时消除的角色损害。

**Expected if H1 false：**sample在共同step或共同选点下仍改善，或equal充分调优也能消除牺牲。

**Interpretation：**共同step优势保留→目标干预效应更可信；只有选点改善→改为模型选择研究；equal调优追平→原baseline优化不足。

**Research Decision：**Continue目标研究／Change mechanism到selection／Pivot到baseline诊断，依上述结果执行。

### P1-A：EMA启动偏差 vs 历史平滑

**Question：**域条件EMA收益来自长记忆还是启动期\(1-\gamma_d^k\)隐式权重？

**Competing Hypotheses：**真实历史滤波；零初始化下的时间域权重。

**Design：**仅在P0-A显示B有剩余优势后做：当前EMA；每域bias-corrected EMA \(m_d/(1-\gamma_d^k)\)；无历史EMA但共享梯度乘\(1-\gamma_d^k\)的schedule控制。固定sample目标、参数块、β2路径及验证规则，记录而非假定实际步长相同。

**Metric：**B相对两控制的G、early域贡献比例、更新角度、固定探针梯度自相关。

**Expected if H1 true：**原trade-off仍需在共同目标下独立证明；本实验不单独判H1。局部长记忆假说为真时，bias correction不应使所有优势消失，schedule不应完全追平。

**Expected if H1 false：**可观察同步改善；若schedule追平而校正EMA掉点，更支持隐式权重而非历史记忆。

**Interpretation：**校正EMA保留优势支持历史因素；schedule足够则重命名机制为时间权重；二者都变化则做中介谨慎解释。

**Research Decision：**Continue／Change mechanism；不以任何单臂掉点强行证明竞争机制。

### P1-B：定位Adam二阶矩与共享/私有参数路径

**Question：**残余是否主要由二阶预条件或私有头正则化改变驱动？

**Competing Hypotheses：**共享域条件方向收益；滤波后v改变；私有头loss尺度改变。

**Design：**以P0-A优胜臂为参照，固定一阶方向构造raw聚合梯度v与filtered-gradient v的对照；另作weighting仅施于共享、仅施于私有、全部参数的少量路径控制。该分组梯度实验是诊断性干预，不宣称仍是同一个标量目标的梯度下降。相同初始化、样本和step，记录真实\(\Delta\theta\)。

**Metric：**分块更新范数、cosine、v统计、AUC/logloss；以配对优势是否保留为主，不能只展示热图。

**Expected if H1 true：**若结构性牺牲确由共享竞争，应对共享路径干预敏感；这仍是必要线索，非充分证明。

**Expected if H1 false：**普通二阶路径或私有正则化就能解释收益，无须结构性冲突。

**Interpretation：**v控制解释收益→转向preconditioning；共享路径主导→再检查跨域方向；私有路径主导→放弃“共享冲突修复”。

**Research Decision：**Change mechanism或Continue，视优胜路径决定。

### P1-C：固定样本池的频次与信息量分离

**Question：**关键变量是unique domain size还是sampling ratio？

**Competing Hypotheses：**统计稀缺；域曝光频次；两者交互。

**Design：**用同一个域身份：第一组固定unique池，改变有放回抽样频次q_d，并以目标p_d/q_d作重要性校正固定风险；第二组固定q_d，改变unique池大小并保持step与目标固定。小范围2×2即可；所有臂均记录重复率与有效正负例数。

**Metric：**域G/T、训练-验证gap、梯度估计方差、有效样本量、实际权重。

**Expected if H1 true：**若“domain size imbalance本身”是关键，固定曝光后改变unique大小仍应驱动角色不对称；若只随频次变化，需修改H1的变量定义。

**Expected if H1 false：**目标固定后不对称消失，或收益只随统计稀缺/曝光而非ratio标签变化。

**Interpretation：**unique主导→统计学习问题；q主导→sampling/optimization问题；二者交互→不再用单变量size故事。

**Research Decision：**Modify hypothesis或Pivot research direction。

### P2-A：独立seed与数据子集的确认性重复

**Question：**关键效应是否超出目前四seed和一次split？

**Competing Hypotheses：**可泛化的协议效应；子集偶然性/seed选择。

**Design：**只重跑最终保留的简单baseline与优胜方法；训练seed和抽样seed分别控制，至少一批从未用于设计的方法选择的seed；如使用时间数据，留后续时间段。先注册唯一主效应与非劣界ε，按先导配对方差决定重复数。

**Metric：**配对区间、实际效应、逐cell非劣性；用户/item相关时采用合适簇单位估计数据不确定性，不能只bootstrap记录行。

**Expected if H1 true：**预注册的相反角色效应在新重复中维持。

**Expected if H1 false：**对立消失或被简单baseline消除；方法收益不稳定则不支持推广。

**Interpretation/Research Decision：**确认→Continue；效应不确定→Warning并收窄结论；反向→Modify hypothesis/Pivot。

### P2-B：最小泛化、敏感性与计算成本

**Question：**最终机制能否在额外backbone或D=5保留，收益是否值得成本？

**Competing Hypotheses：**可移植轻量机制；特定模型/参数偶然效果。

**Design：**只选择P0-B未覆盖的一种可靠backbone或D=5；采用最终已确定目标与参数，γ上下邻近两点；同硬件无额外负载差异地记录每step和训练总成本。无需复活STAR或Curriculum。

**Metric：**逐域任务AUC/logloss、更新步数、wall/GPU memory、收益/成本；T保持2时明确不声称T泛化。

**Expected if H1 true：**跨设置仍有原结构性pattern；若论文已改为目标条件假说，则需复现目标交互而非强行寻找牺牲。

**Expected if H1 false：**结构性pattern不稳；优化收益可有可无，不能用它反推H1。

**Interpretation/Research Decision：**机制同型→Continue；仅局部有效→Modify hypothesis；无实用增益→Abandon新增方法的推广主张。

## 第十三部分：预算只有一个或三个实验

**只做一个：P0-A，sample-weighted matched-momentum controls。** 原因：已有证据足以暂停Curriculum和固有trade-off故事；现在唯一可能支撑新方法的资产，是+0.005407残余。应首先检验它是否真的需要域条件结构。先完成已准备矩阵的两个控制臂，可利用旧A/B，约24个新增run；这一步是筛查，若优胜才付出公平调优与新seed确认预算。

**能做三个：P0-A＋P0-B＋P0-C。** 分别回答方法是否独有、问题能否跨数据集、结果是否依赖选点和预算。三者的信息互补，大于继续γ大网格或增加多个弱消融。P1插桩按P0-A结果决定，不先铺开全量昂贵轨迹。

## 第十四部分：停止条件

预先区分“停止机制”“停止组件”“停止核心问题”。ε必须在看下一轮test前确定；如将0.002 AUC作为实践门槛，应报告CI与该门槛的关系，不能将不显著当等效。

| 决策级别 | 可操作条件 | 应采取行动 |
| --- | --- | --- |
| Continue condition | 公平调优后B优于matched C/D，配对区间支持有意义优势；第二数据集复现目标交互 | 继续目标×优化研究，再做P1机制 |
| Warning condition | 均值正但区间覆盖0或实际门槛；只一任务/一角色有效；固定step优势消失 | 缩小scope，暂停新模块，先补鉴别证据 |
| Pivot condition | 普通uniform/Adam达到B；或weighting效果仅ML存在 | 转向协议诊断/通用稳定化或数据特异泛化问题 |
| Abandon condition | 新seed与公平baseline后残余不足实践价值；第二数据集也不能支持目标交互，主要现象由口径错误解释 | 放弃当前新方法核心命题；保留可复现阴性/诊断结果 |

分假说执行：

- H1强结构性版本：简单目标干预持续同时改善各域且无实际重要损害，就停止“不可避免牺牲”叙事。现有ML均值证据已经触发转向；严格逐cell非劣尚未证明。
- H2纯size版本：固定unique量和风险后效应只随sampling或baseline调优变化，就改写关键变量，不再称纯imbalance causality。
- H3：当前课程未达到保留标准，已经停止作为核心组件；没有新动机和新判别设计，不再为它追加κ搜索。
- H4：匹配uniform/Adam非劣且成本更低，停止域条件独有机制主张；若启动schedule复现收益，则改为时间权重解释。
- H5：当前无协同证据，停止协同卖点；只有在预注册interaction及其CI支持实用正效应后才重启。

## 第十五部分：最终研究诊断报告

### 1. 本阶段最重要的实验发现（最多5条）

1. equal目标下，角色反转与固定N sweep重复规模角色不对称迁移，MMoE也出现。
2. sample weighting使大域平均缺口收窄约92%，同时提升中小域，强烈削弱结构性trade-off解释。
3. comp1边际收益由+0.017359降为+0.005407；后者12/12整体配对正，集中于S，但没有证明特有机制。
4. 当前Curriculum及其与Momentum的正协同没有稳定证据，应移出核心。
5. 原报告的逐cell无损、完全消失、独立样本数和worst指标存在越界或口径问题，必须修正。

### 2. 相比上一阶段，我们现在知道了什么新东西

固定总量没有消除现象，但改变loss权重显著改变它；由此将主要解释从“数据/架构固有缺陷”移到“规模分配×训练协议”。动量在sample目标下仍有剩余价值，已成为最有信息价值的后续鉴别对象。

### 3. 哪些旧结论被加强

size role确有影响；comp1有效性不局限M3OE；Curriculum原有弱证据在修正口径下继续弱；困难域不等于小域。

### 4. 哪些旧结论被削弱或推翻

M3OE特有性、不可避免的大小域牺牲、通过总N下降解释全部收益、课程协同、小域必然是受害者均需放弃或限定。原“全部提升/负迁移消失”不是事实；旧KuaiRand的小域方法损害不能隐藏。

### 5. 当前最可信的机制解释

直接支持的是loss weighting的干预效应；潜在中介是目标估计、有限训练动态与泛化。动量的启动期隐式权重、历史平滑和Adam二阶矩均是候选，尚未验证哪一个主导。

### 6. 当前最大的未解决问题

sample目标下的+0.005407，是否超出充分匹配与调优的普通EMA/Adam？

### 7. 当前最大的论文风险

为相同效应不断更换解释，却始终不击败强简单控制；以及把目标偏好选择写成通用“错误修正”，把统计不显著写成问题消失。

### 8. 是否需要修改研究方向

**明显调整。**

### 9. 当前推荐研究方向

中文：**不平衡多域多任务学习中，目标权重与优化器交互对迁移结果的影响。**

英文：**Objective–Optimizer Interactions in Imbalanced Multi-Domain Multi-Task Learning.**

### 10. 当前推荐 Research Question

RQ1：规模角色不对称迁移有多少由loss权重与model selection决定，能否跨数据集复现？

RQ2：在风险目标与普通优化器公平匹配后，域条件EMA是否仍有独立、实际重要的增益？

RQ3：若存在残余，来自历史滤波、启动期隐式权重，还是Adam二阶矩预条件？

### 11. 当前推荐论文故事线

Problem：MDMT迁移现象和方法收益可能被目标/训练协议混杂。

↓

Observation：固定N角色轮换下出现不对称；weighting干预大幅改变现象与动量收益。

↓

Mechanism Hypothesis：目标估计与优化器时间/二阶动态共同影响有限样本泛化。

↓

Method：先建立目标明确的评估协议和强简单控制；仅保留被证明有增量的优化干预。

↓

Evidence：现有控制链＋matched optimizer＋第二数据集＋必要机制反事实。

↓

Conclusion：只能保留已被验证的目标条件现象与机制；暂不预写“新方法普遍解决trade-off”。

### 12. 下一阶段实验

P0：matched momentum → 第二数据集目标交互 → 完整预算与选点控制。

P1（条件触发）：EMA启动偏差 → 二阶矩/参数路径 → unique size与sampling分离。

P2：独立seed/子集确认 → 一项额外泛化与成本。

### 13. 如果只能做一个实验

**P0-A：sample weighting下的matched uniform EMA与matched Adam控制。**

### 14. 当前研究成熟度

| 维度 | 评分 | 理由 |
| --- | ---: | --- |
| Problem definition | 3/5 | 有控制链，但目标偏好与结构性问题仍需分清 |
| Empirical evidence | 3/5 | 新ML证据较密，独立seed和跨数据集干预不足 |
| Mechanism evidence | 2/5 | 有weighting干预，动量因果中介未拆解 |
| Method novelty | 1/5 | 样本加权/EMA本身已有基础，域条件价值待证 |
| Experimental completeness | 2/5 | matched controls、预算、独立确认尚缺 |
| Paper story maturity | 3/5 | 诊断型故事形成；方法型故事未成熟 |

### 15. 当前最重要的导师建议

**不要继续围绕既定方法寻找能讲通的机制。先修正报告中的无损与显著性声明，停掉Curriculum主线；用一个强matched-momentum实验决定是否还值得保留域条件动量。若简单控制追平，接受简化结果，把论文转为目标与训练协议的严谨诊断，而不是再添加一个组件。**
