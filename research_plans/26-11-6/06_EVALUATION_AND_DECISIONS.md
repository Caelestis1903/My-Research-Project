# 评价、统计与研究决策

## 1. 优先级与估计对象

本轮遵守五 seed、自然人口、完整 Validation 六格 Macro AUC 最早最大值选点。AUC 为主指标，LogLoss 独立报告。源 [评价实现](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/overnight_ab_20261103/evaluation.py) 和原概率指标语义必须保留，尤其不能更换 LogLoss 截断规则后声称同协议。

新观测只来自训练探针和 Validation。历史已归档 Test 可以解释已有事实，但不当作重新封存的测试集。新的 Test 读取发生在配置、checkpoint 和比较规则冻结之后；不参与阶段门或指标选择。

五种子为 `[2022,2023,2024,2025,2026]`。同 seed 的 Clean/G、多个影子、多个 epoch、六格和多个探针均为相关观测，不能增加独立 seed 数。旧 AGENTS 的四 seed 默认与单轮旧门不适用于本任务。

### 五类效应单独命名

| 名称 | 定义与解释 |
|---|---|
| 同时刻净益 `G_same` | 同轨迹、同 epoch，EMA−online；估计器读出效应 |
| 合法选点净益 `G_selected` | 两读出各自按完整 V Macro AUC 选点后的差；包含估计器和选点位置差异 |
| 固定锚点净益 `G_common` | 在 online-selected 或 F-selected 的共同 epoch 比较；只在 V 曲线/已冻结端点合法计算 |
| 短窗吸收效应 `D_pulse` | 固定在线轨迹，一组在刚过去一 epoch 快/慢吸收的 readout 效应 |
| Gaussian 交互 `I` | 各自轨迹内的 EMA 净益相减；Clean/G 是不同训练目标 |

Macro/Traffic/Pooled、AUC/LL、Validation/Test、selected/common/fixed 不能跨列混用。六格按 d0t0、d0t1、d1t0、d1t1、d2t0、d2t1 的展示顺序；内部模型索引 `3*t+d` 必须显式转换。

## 2. 五 seed 统计协议

对每个冻结对比计算 seed 内配对差 \(\Delta_s\)，报告：

\[
\bar\Delta=\frac15\sum_s\Delta_s,
\quad SD_\Delta=\sqrt{\frac{\sum_s(\Delta_s-\bar\Delta)^2}{4}},
\quad CI_{95}=\bar\Delta\pm2.776445\,SD_\Delta/\sqrt5.
\]

同时报告两臂绝对均值/样本 SD、逐 seed 原值、selected epoch、方向数、全部六格。区间说明训练随机性，不代表新用户/新时间人口的抽样区间。样本探针的 bootstrap 若执行，只用于观察误差，不替代五 seed。

阶段平均先在一个 seed 内算：early=1–10、middle=11–30、late=31–50。预测性能先在一个 seed 内平均其观察 epoch、条件、支持格/组和两个观察面板折，再对五 seed 汇总。不能对成百上千行做独立样本 t 检验。

不要求 5/5 或 4/5 同号，不以 CI 跨 0 作为自动否决。若均值方向和效应量已经足够回答有限问题，应直接接受该经验判断；若仍会改变下一步选择，具体指出哪项效应、差异或预测比较未区分。

**正式模型性能资格与机制判断分开。** 相对发表模型的正式总收益仍采用平均 AUC≥.003、六格平均下降不超过 .001，LL 均差<0 单独判断。这个门不用于决定一个历史偏好、参数效应或指标是否有研究价值。内部 F/H/pulse 不是发表模型 baseline。

## 3. 独立观察面板与函数支持

### 3.1 数据用途

- `P_A`：训练集固定观测面板，是 M1/M2/M3 主特征来源。
- `P_B`：与 P_A 按原始行号互斥的固定训练面板，只用于采样稳定性复测；不按结果挑选较好面板。
- `V_mech`：每域自然抽样最多 4096 行，不进入在线 loss、增强生成或 EMA 更新。全量 V 仍用于官方选点。
- `V_A/V_B`：V_mech 内稳定哈希二分，用 `SHA256(dataset|original_row_id)` 最低位决定，不含模型 seed。各 seed 的行成员可能因 V_mech 抽样不同而不同，但同一原始行在所有 seed 中总归同一半，不发生跨折标签流动。

抽样由源标签/行号决定，不能查看模型预测后挑样本。P_A/P_B 仍属于原训练人口，不从 online loader 删除，避免改变训练目标。V_mech 的抽样 hash、行号、标签、fold、正负数量保存一次；后续所有 readout 共用。

### 3.2 结构性零与科学零

函数支持由源码和工程测试冻结：对于 c=(d,t)，主分析包含 `S,Dd,Tt,DTdt` 四组。六格合计 24 个 (c,b) 支持关系。其余 48 个结构性不可达组合仍保存和检查，但不进入主 MAE/方向准确率/regret 的平均，以免大量必然零使指标看似准确。

不能按观测到的收益是否为 0 过滤样本。支持组内观测零是真实结果，必须保留。源训练 detach 导数与完整预测函数导数分开标记；结构支持以完整预测函数为准。

## 4. 时间尺度偏好与候选数量偏差

全程 H 候选为 1、2、4 epoch 加历史 F（重复 rho 去重）。对每个固定 epoch 10/20/30/40/50，报告全局统一选择和逐格条件选择的完整响应。

### 4.1 样本内空间

在全 V 的五 seed 均值上计算每格最优和统一最优的差，命名 `IN_SAMPLE_PREFERENCE_SPACE`。它量化现有候选中的潜在空间，但含逐格多次选优优势。不能直接称为新算法增量。

### 4.2 交叉评价的空间

每个留出 seed s、每个 V 面板方向执行：

1. 仅用其他四 seed 在 V_A 上的同 epoch 指标选择 H。统一策略按六格平均选择，条件策略按每格选择；二者候选完全相同。
2. 在 seed s 的 V_B 上评价已经选好的 H；不再选 epoch。
3. 交换 A/B 重复。每 seed 平均两折，最后汇总五 seed。
4. tie 顺序固定为 F、H1、H2、H4；删除重复项后保持顺序。精确最大值，无结果驱动的容忍带。

这同时降低行级选优和种子偶然性影响，支持本研究人口内的诊断空间。它没有把共享模型变成六份可部署模型，也不证明新人口泛化。官方 checkpoint 仍由全量 V Macro AUC 选择，面板分折不改官方协议。

## 5. 指标比较的精确数值协议

### 5.1 必须比较的特征集合

所有拟合器固定为多输出 ridge：目标为 `sum_i ||Y_i - intercept - X_i B||_2^2 + 1 * ||B||_F^2`，两目标平方误差求和，截距不惩罚，无超参数搜索。对训练折中心化的 Xc/Yc，解 `(Xc.T @ Xc + I) @ B = Xc.T @ Yc`；若使用 sklearn Ridge，固定 `alpha=1`。这是未除N的平方误差，等价于平均误差加 `(1/N)||B||²`；不能误用平均误差加1而把稀疏身份参照过度收缩。只用拟合折均值/SD标准化连续变量；常量变量置 0 并保存标记；one-hot 原样使用。几何的零范数情况使用 zero flag，cos 不定义时置 0 只供拟合并同时输入 missing flag，报告仍为 NA。

基础 `C_design` 包含受支持的 `(condition,c,b)` 身份 one-hot（最多 48 项）和 `epoch/50`，允许这个简单参照记住稳定的组/格差异。不存在的类别以冻结字典表示，不用评价折学习字典或缩放值。

| 模型 ID | 输入，在 C_design 基础上 | 目的 |
|---|---|---|
| C_design | 无额外量 | 稳定身份/阶段是否已足够 |
| M0 | 组内 gap RMS、pulse 快慢差 RMS、二者 cos 与 zero/missing flag | 几何参照 |
| M1 | P_A 自然面板上 fast−F、slow−F、fast−slow 的 logit RMS | 函数幅度参照 |
| C_M01 | M0+M1 | 单纯合并几何与函数幅度是否已足够 |
| C_simple | M0+M1，加 `grad_rank_rms_full(c,b)` 与 P_A 全 fast−F、全 slow−F 的六格 Macro AUC 差 | 已有梯度幅度与全局历史偏好是否已足够 |
| M2_full | C_simple，加完整函数梯度的 fast−F、slow−F 有符号 rank-risk 近似 | 在同样幅度/全局信息之上，功能方向是否有增量 |
| M2_detach | C_simple，加源 detach 代理的同两列；其余输入与 M2_full 相同 | 代理导数语义是否影响收益近似 |
| M3_direct | C_simple，加 P_A 的 fast−F、slow−F 精确 AUC 差和 rank-risk 差 | 直接局部效用是否在简单参照之上预测独立效用 |

简单参照固定为 C_design、M0、M1、C_M01、C_simple，复杂指标必须对全部五个报告同折差；新增幅度/全局特征均由已计算的 M2 梯度和 all-fast/all-slow 读出归约，零额外 GPU 观测。Innovation 的 epoch 统计只用于描述，不未经登记加入回归。共八类固定 ridge，不增加神经网络、自动特征搜索、多个 ridge 系数或按 Test 选指标。P_B 上的 M1/M3 同定义结果用于报告一致性；M2 只在 P_A 求导，并比较其分数与 P_B 的实际风险变化，不增加 P_B 反向；不得事后替换 P_A 主结果。另报告四个不用 ridge 的符号/动作诊断：M2_full_raw、M2_detach_raw、M3_auc_raw、M3_rank_raw，分别用各自 fast−F/slow−F 两分数与0比较，不能按结果二选一隐藏不利版本。rank-risk 分数不是未经标定的 AUC 数值预测。

### 5.2 预测目标

在观察 epoch e，特征只能使用截至 e 的状态、P_A 和身份信息。标签来自 e+5 重新进行同定义一 epoch 脉冲后的 V 面板：

\[
Y_{e+5}=[U(\phi^{fast,b}_{e+5})-U(F_{e+5}),\ U(\phi^{slow,b}_{e+5})-U(F_{e+5})].
\]

`Y:[N,2]`，主方向目标 `D=Y[:,0]-Y[:,1]`。这衡量后续阶段重新出现的历史吸收偏好，**不是当前脉冲对未来训练的延迟作用**，因为脉冲从不反馈。

主 U 为 cell AUC；LL 效用用 `−LL` 独立报告，不参与 AUC 特征模型选择。训练 rank risk 的方向为“风险降低=正收益”。

### 5.3 五 seed × 时间 × 行的固定切分

每个数据集单独执行 5 个留一 seed 折，每折再做两个 V 行方向：

- 拟合：其他 4 seed，origins=[5,10,15]，只用 V_A 的 future 标签（目标 epochs=[10,15,20]）。
- 评价：留出 seed，origins=[25,30,35,40,45]，只用 V_B 的 future 标签（目标 epochs=[30,35,40,45,50]）。
- 然后互换 V_A/V_B。两个条件 Clean/G 都在拟合中，以固定 condition 身份区分；不得漏掉不利条件。
- 禁止随机打散 epoch 行、将同 seed 相邻 epoch 分到普通随机 train/test，或以当期效用预测当期同一标签后声称有未来预测力。

拟合和评价窗口之间留出 e20–25 的间隔。历史状态仍连续，这是目标场景的真实依赖；seed 与行分折并不把时间行变成独立样本。全部十折结束才作一个数据集的五 seed 科学判断。

### 5.4 报告量

1. `MAE_D` 与两列 `MAE_Y`；按 cell/group/phase 分解。
2. 相对每个简单控制的 `skill=1−MAE_method/MAE_control`；分母为 0 则 NA，报告绝对差，不加 epsilon 制造巨大技能值。
3. D 的方向准确率：目标精确为 0 的行单列；全部行的误差照常保留。不要为得到高准确率事后删掉小效应。
4. 冻结诊断动作 `argmax([0,yhat_fastF,yhat_slowF])`，tie 优先 F，再 fast，再 slow。两预测均非正就回到 F。
5. 该动作在独立观察折上的实际 `U_chosen−U_F`，以及 `regret=max(U_F,U_fast,U_slow)−U_chosen`。
6. 由 P_A 直接分数选动作的无需拟合版本，作为低成本可用信息诊断。

动作效用是“单个参数组读出机会”的平均，不能把不同 c/b/epoch 的动作加起来，声称存在一个同时获得该 Macro AUC 的模型。每条建议必须保留实际 readout 身份。

另固定 `UNIFORM_ACTION`：每个 dataset×condition、每个LOSO/行分折，只用拟合侧四seed的早期未来标签，在24个受支持单组机会和全部拟合epoch上等权平均 `[0,Y_fastF,Y_slowF]` 后选一个统一动作，tie顺序仍为F/fast/slow。这个动作冻结后用于该折留出侧的同一批单组机会；不能拿全参数all-fast的AUC与单组机会平均相减。它不新增ridge/GPU，只用于判断一个统一快慢偏好是否已足够。

## 6. 样本噪声与观察精度

P_A/P_B 用于回答指标是否对少量训练样本敏感。报告同定义分数差、相关性和动作一致率，不把面板 A/B 当独立 seed。若 M2 与精确 train 风险接近，而 train 与 V 对应差，则问题在 train 到 holdout 的迁移；若完整导数明显优于 detach 代理，则优先修订指标解释，而不是直接改源训练。

4096 个固定正负 pairs 的净排序纠正恒等于该 pair 集上的 AUC 差；与精确自然面板 AUC 可能不完全相同。分别命名并保留二者误差。每格 AUC 用原概率排名计算，不对全部自然正负对做 O(N²) 枚举。

若小效应的样本噪声明显影响某个实际决策，本轮先使用已保存的 V_A/V_B 与自然面板全量指标界定不确定性。不得自动加 seed、扩大训练预算或从大量新指标中寻找显著相关。

## 7. 自动触发的两项控制

### C_CAL：概率尺度能解释多少 LL 取舍

触发单位为 dataset×condition。在 e20/e40 任一共同 epoch，五 seed 平均 `EMA−online AUC` 与原始 `EMA−online LL` 同为正或同为负，记为排序/LL 取舍。若存在该情况，评价两个 epoch 和全部六格，不能只选最有利格。

在 V_A 上为每个模型/格拟合一个正温度 T∈[.25,4]，在 V_B 评价，交换后取平均。目标为源 LogLoss，固定最多 100 次有界标量优化、绝对收敛容差 1e-8；无额外 HPO。对源概率 p 使用正温度映射 `sigmoid(logit(p)/T)`，p=0/1 按极限保持端点，T=1 精确返回源 p，以保留原始平局关系；不调整截距、不改标签基率、不重选 checkpoint。

报告未校准 LL 差、双方分别校准后的 LL 差、差值缩小量。正温度在实数下保持每格排序；浮点 sigmoid 可能产生新饱和和 ties，因此本控制 AUC 沿用原 p 排序，只重算 LL，并记录新增饱和/tie 计数。校准后 LL 差的缩小才是这项控制的经验发现。观察集拟合参数不会发布为新方法，不进入 Test。

### C_BG：组效应是否依赖其他参数背景

用 e20/e40、五 seed 平均的 pulse AUC 效应，计算每个 dataset×condition×cell 的

\[
R=D_{all}-\sum_bD_b,
\quad q=|R|/(|D_{all}|+\sum_b|D_b|).
\]

分母为 0 时 q=0，且不触发。只有 `q>.5` 且 `max(|D_all|,sum|D_b|)≥.0002` 时触发。`.0002` 仅是省去近零组配合诊断的资源触发线，不是机制有效/无效门，更不是算法性能标准。

触发后在所有触发 strata 上按 `mean(abs(D_b))` 排序，选最多两组；并列用冻结组顺序 S,D0,D1,D2,T0,T1,DT00,…,DT21。对这两组的全部六格、两个保留 epoch、五 seed，分别使用“其他组全 fast”和“其他组全 slow”背景计算目标组快慢差。参数向量来自已保留的 F/ψfast/ψslow，不重新优化。

若效应随背景翻转或大幅改变，就接受组间配合的限定结论，停止单组加和贡献主张。若不变，保留其局部路径定位价值。一次控制后不扩展到所有参数组合。

## 8. 研究判定规则

| 证据结果 | 可以接受的结论 | 后续决策 |
|---|---|---|
| 共同 epoch EMA 有益 | 同轨迹历史读出有效；现有事实继续成立 | 不重复有效性验证 |
| EMA 对年龄匹配旧 online 仍有益 | 超出该简单延迟对照的历史组合价值 | 结合排序翻转/路径读出解释 |
| 六格快慢偏好在交叉评价中仍有差异 | 本人口/模型范围内存在可重复的历史偏好差异 | 看其是否可预测，不立即设计控制器 |
| 仅样本内逐格选优有空间 | 目前只有描述性空间或选择机会优势 | 标 `DESCRIPTIVE_ONLY` |
| 指标样本外 MAE 更低 | 接受收益大小的预测价值 `EFFECT_SIZE_PREDICTIVE` | 不强求同时改善动作；报告各自作用 |
| 指标的留出动作效用更高 | 接受选择价值 `ACTION_USEFUL`，即使绝对数值标定未更好 | 与F及统一诊断动作比较，决定是否有差异化空间 |
| 几何/阶段参照已与直接收益指标相当或更好 | 简单信息足够，复杂指标没有独立必要性 | 优先简单参照，保留机制解释 |
| 只在部分数据/任务/阶段预测有效 | 接受并清楚限定条件性价值 | 不用另一设置自动推翻 |
| 所有候选无样本外增量 | 当前信号没有提供可用的历史控制依据 | 结束指标扩张，维持固定 EMA |
| 新脉冲效应很小或互相抵消 | 该干预范围下可利用空间有限 | 不将其泛化成所有历史机制不存在 |

`ACTION_USEFUL(m)` 的操作性要求：在预先列出的 dataset×condition 或其支持cell/组范围，五seed留出评价的 `mean(U_m−U_F)>0` 且 `mean(U_m−U_uniform)>0`。所有预注册范围完整报告，不能只保留最有利切片。相同机会上的regret降低与动作效用增加是代数等价，不作为两份独立证据。

`READY_FOR_TARGETED_DESIGN`：交叉评价中存在cell/路径偏好或受控组脉冲的可利用空间，且至少一个train-only方法满足上述ACTION_USEFUL。方法可以是C_design/M0/M1等简单参照，不要求它战胜自己，也不要求MAE同时更低。仅在声称M2/M3的复杂信息具有独立必要性时，才要求其同折动作效用超过全部五个简单参照；否则保留其有效性并优先较简单方法。若仅胜F、未胜统一动作，则接受全局历史调整价值，不宣称差异化空间。无需全部seed同向或CI排除0。若正向动作优势小于.0002 AUC，仍接受经验值，但标 `LOW_PRIORITY_SMALL_EFFECT`，不夸大复杂算法投入优先级。

若不同证据冲突到足以改变上述状态，写 `SPECIFIC_QUESTION_UNRESOLVED`，给出相关两项效应、范围、为什么会改变决定。S4 是本轮最后控制，不自动增加实验。机制尚未完整不影响 L1/L2 已有结论。

## 9. 冻结记录与报告合同

`STAGE3_DECISION.json` 至少包含：

```json
{
  "schema_version": "1.0",
  "study_id": "EMA_MECHANISM_20261106",
  "input_hashes": {},
  "scientific_seed_count": 5,
  "fresh_test_accessed": false,
  "metric_primary_id": "P_A_LOSO_FORWARD_ROW_CROSSFIT",
  "controls": {
    "C_CAL": {"triggered": false, "strata": [], "reason": "computed_from_validation"},
    "C_BG": {"triggered": false, "groups": [], "reason": "computed_from_validation"}
  },
  "next_action": "CONTROLS_OR_FREEZE",
  "decisions": []
}
```

示例中的 false/空表是 schema 示例，不是预判结果。实际文件由分析脚本计算，绑定 manifest/source/data/probe/fold/results hash。`SELECTION_FROZEN.json` 固定每个 regular readout×seed 的 epoch/state hash；`FREEZE_RECEIPT.json` 绑定研究比较、阶段决定、配置和选点的共同摘要。二者一致且五 seed 齐全才允许新 Test。

最终 `MECHANISM_DECISIONS.json` 的每项结论至少含：`question_id`、`hypothesis_id`、`claim_level`、`status`、`scope`、`effect_size`、`n_seeds`、`source_artifact`、`competing_explanation_disposition`、`next_research_decision`。不使用单一 `success=true/false` 抹去不同结论层次。

### 自动报告必须生成的表/图

| 产物 | 必须包含 |
|---|---|
| 四格与交互表 | Clean/G × online/F；六格与Macro；selected/E10/E50；归档/新观测标记 |
| 时间曲线图 | online/F/H 的共同 epoch 收益，六格小图，阶段分界与选点标记 |
| 历史偏好热图 | cell×H×phase，样本内与交叉评价空间并列 |
| 脉冲效应表/图 | 12组×6格，fast−F、slow−F、fast−slow，全组与非加性 |
| 排序与LL图 | 纠正/损害/平局、相邻epoch固定pair变化T；精确AUC与pair近似分开，LL自然权重 |
| 指标比较表 | 每fold、每seed、MAE/skill/regret/动作效用，P_A/P_B一致性 |
| 成本表 | 真实训练、额外forward/backward、CPU分析、等待、GPU峰值、guardian RAM、磁盘 |
| 最终回答 | Q1–Q5及未来定向算法是否有依据；可接受结论、限定竞争解释与停止决定 |

图由标准绘图库生成 PNG/PDF，保留表格数据。最终报告每个结论链接到对应原始数表/配置/哈希。全局上下文和性能台账只生成更新提案；本轮执行者按用户后续授权决定是否回写 main，不能追溯修改历史记录。
