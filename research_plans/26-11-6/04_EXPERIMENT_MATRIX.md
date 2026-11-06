# 分阶段实验矩阵与预算

## 1. 总体结构

本轮科学配置只有 **2 个数据集 × 2 种训练目标 × 5 个 seed = 20 条在线轨迹**。每条训练 50 epoch，新增真实 Adam 更新总数 34,500。多时间尺度、参数组脉冲、收益指标均依附同一在线轨迹，不另训练模型，不进入在线 loss/Adam/Gaussian 生成。

阶段是分析依赖，不是五轮重训。阶段 1–3 所需观测一次采齐；阶段 4 的两个条件控制只使用已保留的预测与有限 RAM 锚点。历史 A/B 的 15 影子算法矩阵、C 条件增强和发表模型不重新运行。

| 阶段 | 核心任务 | 新在线训练 | 主要产物 | 下一阶段条件 |
|---|---|---:|---|---|
| S0 | 现有证据/代码/产物身份审计 | 0 | `EVIDENCE_REUSE.csv`, `ARCHIVED_FACTORIAL.csv`, `AUDIT_INVENTORY.json` | 完成真实来源与缺失观测登记；不重新裁定 EMA 是否有效 |
| S1 | 全程时间尺度、同 epoch/选点、阶段、六格分布 | 与 S2/S3 共用 20 条 | `VALIDATION_CURVES.csv`, `HISTORY_PREFERENCES.csv`, `SELECTION_DIAGNOSTICS.csv` | 五 seed 齐全且工程验收通过；按现象描述进入 S2 |
| S2 | 一 epoch 参数历史吸收干预与函数收益 | 0 条额外训练 | `PULSE_READOUTS.csv`, `RANK_FLIPS.csv`, `GROUP_INTERACTIONS.csv` | 不因增量小阻断已采集指标的分析 |
| S3 | 候选指标比较、未来预测、简单替代 | 0 | `PREDICTIVE_OOF.csv`, `PREDICTION_SUMMARY.csv`, `STAGE3_DECISION.json` | 依据预先定义规则触发最多两个 S4 控制，或直接冻结并报告 |
| S4 | 校准取舍/第二参数背景的必要区分 | 正常 0 | `CONTROL_RESULTS.csv`, `MECHANISM_DECISIONS.json` | 每个触发问题一次控制；之后停止追加实验 |
| Final | 冻结正式读出、必要 Test、形成报告 | 0 | `SELECTION_FROZEN.json`, `ENDPOINTS.csv`, `FINAL_REPORT.md` | 完成既定科学问题，释放本任务 RAM 状态 |

S0 已在本次规划中完成核心原始端点重算，见 [01](01_EXISTING_EVIDENCE.md) 与 `evidence_snapshot.json`。Codex 的 S0 负责本地复现和完整曲线提取，不把它变成一轮新研究设计。

## 2. 冻结训练配置

| 配置 | ML-C | ML-G | KR-C | KR-G |
|---|---|---|---|---|
| dataset / condition | movielens / clean | movielens / gaussian_d0 | kuairand / clean | kuairand / gaussian_d0 |
| seed | 2022–2026，各五个 | 同左 | 同左 | 同左 |
| 训练次数 | 5 | 5 | 5 | 5 |
| epoch / 每 epoch steps | 50 / 20 | 50 / 20 | 50 / 49 | 50 / 49 |
| Adam 更新数 | 5,000 | 5,000 | 12,250 | 12,250 |
| Gaussian | 关闭，源 clean loss | D0，κ=.05，λ=.5 | 关闭，源 clean loss | D0，κ=.1，λ=.5 |
| coupled Adam wd | .0001 | .0001 | .00001 | .00001 |
| 固定 F 的 rho | .9828205985452511 | 同左 | .995 | 同左 |
| 常驻 EMA 影子 | H1/H2/H4；F=H2 | 同左 | H1/H2/H4/F | 同左 |

其余共同设置：batch 40960；lr=.01；betas=(.9,.999)；eps=1e-8；50 epoch 全预算；FP32；无 AMP/TF32/clipping/early stop；fixed1 M3oE，embedding16、FCN[512,256,256,64]、4 shared experts。自然人口、tail batch 分母、数据行序、编码与原始 Gaussian RNG key 均继承源实现。准确映射见 [05](05_IMPLEMENTATION_SPEC.md)。

**H1/H2/H4 指 epoch 单位半衰期，不是 rho 数值。** 对 S 个 optimizer steps/epoch，`rho(H)=2**(-1/(H*S))`。ML 的 H2 与 F 去重，只分配一次参数状态。KR 的历史 F 半衰期为 2.8220933262 epoch，作为独立历史锚点保留。比较范围被冻结，不再按照结果增加 .5、8、16 等尺度。

该网格用于机制响应，不是寻找最优性能。历史 Gaussian 的 U_FAST/F/U_SLOW 与 Clean 长短 EMA 数表全部复用；网格内的重复轨迹只为补充缺失的参数与预测观测。

## 3. S0：必须先用现有资料回答的内容

### 3.1 读取与去重

固定源 `40705dfe0bb4fe614bdca835c0bae567348c87d0` 下读取：

- `artifacts/aug_history_20261105/attempt1/B/{ENDPOINTS.csv,VALIDATION_CURVES.csv,HPO_SUMMARY.csv}`。
- `artifacts/aug_history_20261105/attempt_A1/A/` 和 `attempt_C1/C/` 的同类数表及对应报告。
- `research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv`。
- `research_reports/2026-11-03_overnight_AB/`、`research_reports/2026-11-04_gaussian_domain_coverage/` 的原始端点、归档说明和身份信息。
- [01](01_EXISTING_EVIDENCE.md) 标明的 11/01 长短 EMA、Polyak 和已有控制结果；仅按其原协议引用。

用 `trajectory_id/model_hash/data_hash/seed` 去重：A/B 相同 online 不算独立复现，ML G-D0 别名不重复计数；KR 旧 G-D2 不能代替当前 G-D0。若归档为压缩文件，解压标量而不是声称模型 checkpoint 可用。

### 3.2 零训练分析

1. 四格 Clean/G-D0 × online/F，分别报告 selected、E10、E50 的 V/Test、六格、Macro、Traffic、Pooled。
2. 全 V 曲线的逐 epoch online/F/U_FAST/U_SLOW；对不同时间尺度先换算成 step/epoch 半衰期与平均历史年龄。
3. 共同 epoch 的收益与各自 selected 收益分开；抽取 selected epoch 与峰值邻域稳定性。
4. 已归档选点结果及 A/B/C 版本性结论直接继承，不扩大其因果范围。
5. 建立资源清单：标量可复用；RAM checkpoint 已释放；预测/参数干预需要观测重放；数据缓存需在目标机器核验。

**停止规则：** 源文件不全时仅将缺项记为 `MISSING_ARCHIVED_ARTIFACT`，优先从既有归档恢复。不能为了补一张旧性能表重训发表模型。关键数据身份/训练语义不一致属于工程阻塞，修正后继续；本轮科学问题不因旧报告使用较少 seed 而被自动关闭。

## 4. S1：时间尺度与收益分布

### 4.1 每条轨迹的全量评价

每 epoch 结束评价 online 和所有常驻影子，使用完整 Validation 六格 Macro AUC 选点。记录六格绝对 AUC/LL、Macro/Traffic/Pooled、model/data/layout hash，保存每个读出的 selected、E10、E50 状态于 RAM。

完整 V 模型读出次数：

\[
10\times50\times4+10\times50\times5=4,500.
\]

这里每数据集有 Clean/G 两条件共 10 条轨迹；不能误算成 2,250。参数更新仍只有 34,500 次。

### 4.2 固定分析视图

- 固定阶段：early=1–10、middle=11–30、late=31–50；在每个 seed 内先平均再跨 seed 汇总。
- 固定比较 epoch：10、20、30、40、50；局部参数脉冲观察 epoch=5,10,…,50。
- `G_same(c,e,H)=U_c(EMA_H,e)-U_c(online,e)`。
- `G_selected`：双方分别按完整 V 六格 Macro AUC 最早最大值选点。
- 共同锚点：online-selected 与 F-selected 两套 epoch 上，同时读取现有 V 曲线中所有读出；不另按每格选 checkpoint。
- 延迟 online：按 F 的有限时刻平均历史年龄，四舍五入到 epoch 得到理论延迟。在同一 V 人口上读取旧 online 曲线；历史索引不足 1 时标 NA，主要解释使用 e≥5 的锚点。不得按 V/Test 调整延迟使其最优。
- 排序时间变化率：从固定P_A/P_B/V_mech pairs的每epoch online/F概率计算相邻epoch变化 T（e≥2）；与净收益并列，区别稳定了损害还是延后了改进，见03第7.3节。
- 选点稳定性：各 seed 峰值前后 ±1 epoch 的平均 V AUC 与峰值差、selected epoch 分散度；这是描述性稳定性，不另选一个“更稳定 checkpoint”。

### 4.3 域任务历史偏好

在 e=10,20,30,40,50 比较每格的 1/2/4 epoch 与 F。报告：

1. 全局统一候选的五 seed 平均响应。
2. 每格样本内最优相对统一最优的诊断空间，明确它包含额外选择自由度。
3. 留一 seed 并交叉观察行：其他四 seed 的 V_A 选择统一 H 或各格 H，留出 seed 的 V_B 同 epoch 评价，再互换 A/B；每 seed 平均两折后五 seed 汇总。候选集完全相同，不增加模型、不读取 Test；具体规则见 06。
4. 六份条件读出没有组成一个可部署单模型，结果名称必须是 `CELL_CONDITIONAL_DIAGNOSTIC`。

**继续/停止：** 五 seed 完整即进行 S2 的已计划观测分析；没有异质性时也给出明确限定结论，不扩大半衰期网格寻找阳性。

## 5. S2：受控参数历史吸收

### 5.1 唯一主干预

目标 epoch e=5,10,…,50。令每 epoch optimizer steps 为 S，在 k0=(e−1)S 时从 F 复制两份向量。下一 epoch 只读同一 online θ：

- fast 脉冲：rho_fast=rho_F²，半衰期为 F 的一半。
- slow 脉冲：rho_slow=sqrt(rho_F)，半衰期为 F 的两倍。

终点分别对 12 组构造 `F + P_b(pulse−F)`，并评价两份全组 pulse。与 F 合计 27 个读出；同一 scratch 顺序完成。它等价于“仅这组在刚过去的一 epoch 改变吸收速度”，不需要 24 份历史状态。

读取面板：P_A、P_B、V_mech。P_A/P_B 均来自 train 且按行互斥；V_mech 按每域自然抽样 4096 行或该域全体可用行。每面板与每个 readout 均保存来源 hash，保持整个训练期间不变。具体抽样/缺类规则见 [05](05_IMPLEMENTATION_SPEC.md)。

### 5.2 必须记录的量

- fast−F、slow−F、fast−slow 的六格 AUC、LL 与训练 pairwise logistic risk。
- 源概率上的正负样本对纠正/损害/平局转换；4096 固定 pairs 的分解与精确面板 AUC 单列。
- 组内参数 gap、pulse 位移 RMS、cos；F 更新前的 innovation 分组能量按 epoch 汇总。
- 全组快慢差与各组快慢差之和的非加性残差；同样记录 fast−F、slow−F 的残差。
- 训练 M0/M1/M2/M3 与独立 V_mech 当期效用的对应关系。

**评价对象是只读估计器。** 既不更新 online/Adam，也不向未来 F 写入脉冲。某组较优不能直接变为训练策略；共享组的六格效应必须全部展示。

### 5.3 成本与状态上限

20 轨迹 × 10 锚点 = 200 个脉冲窗口，覆盖 200 个已在预算内的 epoch。每窗口两份 P 维向量，无新增 Adam。最多 5,400 个参数状态读出（包括可复用的 F），每个状态读取 P_A/P_B/V_mech 三面板，计最多 16,200 次面板 forward（可合批并复用 F）；每次在有界样本面板上完成，不做 5,400 次完整 Validation。

只有 e20/e40 的 F、fast、slow 三份向量保留到阶段 4；其他锚点完成观测即释放。全部状态 RAM-only。若面板统计有效但效应很小，记录幅度和观察精度，不把“此强度下无可利用增量”扩大为不存在历史偏好。

## 6. S3：指标的描述、预测与选择价值

本阶段不再训练推荐模型。按 [03](03_METRICS_AND_MATH.md) 计算 M0–M3；以 [06](06_EVALUATION_AND_DECISIONS.md) 的五折时间预测协议为唯一主评价。

主目标：e 时刻的 train-only 指标，预测 e+5 时新脉冲的 `V_mech fast−slow`。训练 origins=[5,10,15]，评价 origins=[25,30,35,40,45]。训练 label 最晚 e20，评价 feature 最早 e25；目标窗口不重叠，参数历史本身允许持续存在，不能据此假装训练阶段独立。

数据集分别拟合；每折留出一个 seed，其他四 seed 的 early origins / V_A 标签拟合固定 ridge（penalty=1，无搜索），留出 seed 的 late origins / V_B 评价后交换 A/B。identity/phase、M0、M1、C_M01、C_simple、M2_full、M2_detach、M3_direct 使用同样行、同样标准化和固定ridge规则；C_simple含已计算的组梯度RMS和全局读出收益，不新增GPU观测。另报告无需拟合的有符号 M2/M3，避免把学习器的记忆当指标本身。

输出包括 MAE、相对简单控制的技能分数、非零目标方向准确率、样本内/外的选择 regret、实际被选读出的效用及其对 F 的净差。所有行先在 seed 内汇总，再五 seed 报告均值/SD；不能把 200 窗口、12 组、6 格当独立样本增加显著性。

**停止规则：** 无预测价值就接受“当前指标仅描述性/当前可用信息不足”，不追加神经网络预测器或重新搜索指标。部分设置有效就保留局部结论，不强求两个数据集同向。

## 7. S4：只保留两个会改变研究结论的控制

| ID | 触发原因 | 固定做法 | 训练成本 | 支持的结论 |
|---|---|---|---:|---|
| C_CAL | online/F 在 e20/e40 的五 seed 平均 AUC 收益与负 LL 收益相反，等价于原始 AUC 差与 LL 差同号 | V_mech 稳定哈希二分，正温度 T∈[.25,4]，一半拟合另一半评价再交换；每格单参数；五 seed；不改 checkpoint | 0 | 概率尺度能解释多少 LL 差；不解释 AUC 来源 |
| C_BG | 组效应与全组效应非加性会改变“优先哪组”的判断 | 最多两组，用 e20/e40 已留 RAM 向量；在其他组全 slow、全 fast 两个背景下各改目标组；五 seed | 0 | 组效应是否依赖背景，能否作局部定位 |

触发算法和排序见 [06](06_EVALUATION_AND_DECISIONS.md)。每个控制只执行一次，没有新的超参数搜索。C_CAL 不访问 Test；C_BG 不重训、不改变原 online。控制未触发写 `NOT_TRIGGERED`，不是阴性结果。

若状态丢失，先检查已有 guardian。确需重放时，仅重放受影响的逻辑 job，保持原五 seed 组身份，记录为恢复成本和原 trajectory_id；不能当新独立实证。恢复预算见 [07](07_RESOURCES_AND_REUSE.md)。

## 8. 冻结与正式评价

常驻全程 shadow/online 的研究候选、5 seed、比较规则在开始前冻结；各自 checkpoint 在完整 V 结束后冻结。所有 S1–S4 的开发判断只使用归档证据和新 Validation/训练探针。之后才允许加载 Test。

Test 仅评价 online 与全程 H1/H2/H4/F 的 selected/E10/E50，不对 pulse、每格条件最优读出或未冻结指标调参。若某旧 endpoint 的 data/model/readout/epoch/hash 完全匹配，可以直接复用旧 Test 指标；不重复评价冒充新盲测。新半衰期的 Test 只作冻结配置结果报告。

最终判定写入 `MECHANISM_DECISIONS.json`，状态与证据层次见 [02](02_SCIENTIFIC_QUESTIONS.md) / [06](06_EVALUATION_AND_DECISIONS.md)。完成后输出结论更新提案，不自动改写历史原始实验结果。
